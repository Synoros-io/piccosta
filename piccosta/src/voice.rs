// Voice, brought by the user. piccosta records and synthesizes nothing: two commands named in
// settings do that, and piccosta passes text to and from them. Voice is a call with one session at
// a time, the way the user is on a call with a person:
//
// - A call starts with `/call` in the terminal UI (the session shown there, or `/call assistant`)
//   or `piccosta call [assistant | <session id or name>]` (the assistant when none is named). It's
//   a record in the agent dir naming the session's file and an id of its own; starting another ends
//   it, and so do `/hangup`, `piccosta hangup` and closing that session (a socket close, the UI's
//   /close).
// - Push-to-talk talks into the call. With none, `piccosta voice start` and `voice stop` do nothing
//   and exit 0, so the button can be the one the user already talks to people with.
// - What push-to-talk hears, and what's said, belongs to the call it started in, by its id: once
//   the record holds another id or none, listening stops with nothing delivered and speaking stops
//   within CALL_POLL, whichever process ended the call.
// - During a call, every answer the session gives is said, whatever started the turn: something
//   the user said or typed, a background job's end, a lane's news (the assistant calling the user).
//   Whichever runtime runs the session says it, checking the record as each turn starts; what it's
//   saying ends as the session closes. Outside a call nothing is said.
//
// The commands:
//
// - `voice.listen`, a shell command, starts when push-to-talk goes down (`piccosta voice start`).
//   It records from the microphone itself, from when it starts until its stdin closes, which
//   piccosta does when the key comes up (`piccosta voice stop`). It writes what it heard to stdout
//   as UTF-8 lines, each a finished piece of the transcript in order, as soon as it has them (a
//   streaming recognizer while the key is held, a batch one like whisper.cpp after), then exits,
//   within LISTEN_FINISH_SECONDS of its stdin closing. The lines joined by spaces are the user's
//   message, delivered to the session in the call the way `piccosta tell` delivers one (queued in
//   its file when nobody runs it, for the next run to answer).
// - `voice.speak`, a shell command, starts once for each turn said. piccosta writes it UTF-8
//   lines, each a piece of the answer that ends at a sentence or a paragraph, the moment the model
//   has written it, and closes its stdin when the turn ends. It says each line as it comes (Piper
//   reads its stdin this way) and exits after the last. Turns are said one after another, never
//   over each other.
// - `voice.quick`, optional: a fast model (provider/id) that answers each of the user's messages
//   first, from the end of the conversation, while the session's model does the work. Its lines go
//   to the speak command first, then the session's answer's, which wait their turn. It's spoken
//   only: the session never sees its words, so its transcript and its prompt cache are as they'd
//   be without it; what it cost is counted in the session's statistics.

use crate::agent::{AgentConfig, Event, EventSink};
use crate::message::{AgentMessage, Message, SystemMessage, TextOrBlocks, Usage, UserMessage};
use crate::model::{Model, ProviderConfig};
use crate::provider::{Request, StreamEvent};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;

// In the agent dir: the record (pid and start time) of the hold that's listening, the lock that
// lets one `voice start` or `voice stop` at a time touch it, and the call's record.
const HOLD_FILE: &str = "voice.pid";
const LOCK_FILE: &str = "voice.lock";
const CALL_FILE: &str = "call.json";
// Where a hold's errors and the listen command's stderr go, in the agent dir.
const LOG_FILE: &str = "voice.log";
// How long a hold listens when no `voice stop` comes (a missed key release), unless
// `voice.maxSeconds` says otherwise: two minutes is longer than anyone talks at a button, and short
// enough that a microphone left open by mistake doesn't stay open.
const MAX_HOLD_SECONDS: u64 = 120;
// How long the listen command has, after the key comes up, to write the rest and exit before it and
// everything it started are killed: a batch recognizer like whisper.cpp transcribes a two-minute
// clip in a few seconds on a CPU, so thirty is room for a slow machine and still bounds the mic.
const LISTEN_FINISH_SECONDS: u64 = 30;
// The speech worker's bounds. A write to the speak command that takes longer than
// SPEAK_WRITE_SECONDS means it has stopped reading (a pipe holds far more than it says in that
// time), and the rest of that turn isn't said. After its stdin closes it gets SPEAK_EXIT_SECONDS
// to finish saying what it has, about what a 900-token answer takes to say aloud, then it's killed.
// The quick model gets QUICK_SECONDS: it's meant to answer in one. A print run waits up to
// DRAIN_SECONDS, once its answer is done, for what's still being said. A frontend that exits gives
// the worker STOP_SECONDS to kill the speak command and end: a killed process is reaped at once, so
// five is room for a loaded machine and still no wait anyone notices.
const SPEAK_WRITE_SECONDS: u64 = 10;
const SPEAK_EXIT_SECONDS: u64 = 300;
const QUICK_SECONDS: u64 = 30;
const DRAIN_SECONDS: u64 = 300;
const STOP_SECONDS: u64 = 5;
// How often listening and speaking look at the call's record to see that their call is still on: a
// spoken line is a sentence, a couple of seconds, so half a second ends speech within the line
// being said, and reading a small file twice a second costs nothing.
const CALL_POLL: Duration = Duration::from_millis(500);
// How many turns can wait to be said, and how many pieces of one turn's answer: a few turns is
// more than anyone asks while one is said, and four thousand streamed pieces is several thousand
// tokens, minutes of speech. Past either, what doesn't fit is dropped, never waited for.
const TURN_QUEUE: usize = 4;
const ANSWER_QUEUE: usize = 4096;
// How much of the end of the conversation the quick model reads: about 6k tokens, the last several
// exchanges, which a fast model reads in well under a second.
const QUICK_CONTEXT_CHARS: usize = 24_000;
// The kind of the session's usage entries for quick answers.
pub const QUICK_KIND: &str = "voice-quick";
const QUICK_PROMPT: &str = "You are the voice that answers first. The user has just spoken to their assistant, which is now working on the full answer and will speak it after you. Say at once, in one to three short sentences, what you can from what you know and the conversation below: answer when you can, else say what is being looked into. Don't claim to have done or checked anything. Write for speech: plain sentences, no markdown, lists or code.";

// A shell command line, as `sh -c` runs it, in a process group of its own so it can be killed
// along with whatever it starts.
fn shell(command: &str) -> Command {
    let mut shell = Command::new("sh");
    shell.arg("-c").arg(command).process_group(0).kill_on_drop(true);
    shell
}

// Kill a process group that's gone past its time, and reap its leader.
async fn kill_group(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        // SAFETY: killpg() only sends a signal.
        unsafe { libc::killpg(pid as i32, libc::SIGKILL) };
    }
    let _ = child.wait().await;
}

// The voice log, opened to append: a hold's errors and the listen command's stderr, and in a
// session's piccosta (the terminal UI's screen among them) the speak command's and the quick
// model's.
fn log() -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new().create(true).append(true).open(crate::config::agent_dir().join(LOG_FILE))
}

pub(crate) fn note(text: &str) {
    if let Ok(mut file) = log() {
        let _ = writeln!(file, "piccosta voice: {text}");
    }
}

fn text_setting(settings: &crate::config::Settings, key: &str) -> Option<String> {
    settings.get(key).and_then(Value::as_str).filter(|text| !text.trim().is_empty()).map(str::to_string)
}

// --- Process records ---

// Write `record` to `path`, beside the old one and renamed over it, so a reader finds one or the
// other, never half of one.
fn write_record(path: &Path, record: Value) -> std::io::Result<()> {
    let written = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&written, record.to_string())?;
    std::fs::rename(written, path)
}

// The hold's record at `path`: its pid, when the process that wrote it is still running. The
// record holds its boot and start time too (proc.rs recorded_start), to tell it from a process
// given the same pid since.
fn live_record(path: &Path) -> Option<u32> {
    let record: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let pid = u32::try_from(record.get("pid")?.as_u64()?).ok()?;
    let started = crate::proc::recorded_start(record.get("started")?.as_str()?)?;
    crate::proc::Process::started(pid as i32, started).map(|_| pid)
}

// The call the user is on: its session's file and its id, or None when there's no call. A read that
// fails for any reason but the file being gone (too many open files, say) is an error, not a hangup.
fn reading(agent_dir: &Path) -> std::io::Result<Option<(PathBuf, String)>> {
    let text = match std::fs::read_to_string(agent_dir.join(CALL_FILE)) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        read => read?,
    };
    // The record is replaced whole (write_record), so one that doesn't parse isn't a call.
    let record: Value = serde_json::from_str(&text).unwrap_or_default();
    Ok(record["session"].as_str().zip(record["id"].as_str()).map(|(file, id)| (PathBuf::from(file), id.to_string())))
}

fn current(agent_dir: &Path) -> Option<(PathBuf, String)> {
    reading(agent_dir).ok().flatten()
}

// Whether the call with id `id` is known to be over: no call, or another one. A failed read says
// nothing, so speech and listening go on until a read says so.
fn over(agent_dir: &Path, id: &str) -> bool {
    reading(agent_dir).is_ok_and(|call| call.is_none_or(|(_, now)| now != id))
}

// The file of the session the user is on a call with, if they're on one.
pub fn call(agent_dir: &Path) -> Option<PathBuf> {
    current(agent_dir).map(|(file, _)| file)
}

// Done once the call with id `id` is over, looked for every CALL_POLL.
async fn ended(agent_dir: &Path, id: &str) {
    while !over(agent_dir, id) {
        tokio::time::sleep(CALL_POLL).await;
    }
}

// Call the session in `file`, under a new id, which ends any other call; None hangs up.
pub fn set_call(agent_dir: &Path, file: Option<&Path>) -> std::io::Result<()> {
    match file {
        Some(file) => write_record(&agent_dir.join(CALL_FILE), json!({ "session": file, "id": uuid::Uuid::now_v7().to_string() })),
        None => std::fs::remove_file(agent_dir.join(CALL_FILE)).or_else(|error| if error.kind() == std::io::ErrorKind::NotFound { Ok(()) } else { Err(error) }),
    }
}

// Hang up if the call is with the session in `file`, which is closing.
pub fn hang_up(agent_dir: &Path, file: &Path) {
    if call(agent_dir).as_deref() == Some(file) {
        let _ = set_call(agent_dir, None);
    }
}

// The session `wanted` names for `piccosta call`: the assistant's for none or "assistant", the file
// for a path, else the session whose id starts with it or whose name it is.
fn session_named(agent_dir: &Path, wanted: Option<&str>) -> Result<PathBuf, String> {
    match wanted {
        None | Some("assistant") => crate::manager::session_file(agent_dir).ok_or_else(|| "no assistant yet: start it with `piccosta assistant`".into()),
        Some(path) if path.contains('/') || path.ends_with(".jsonl") => std::path::absolute(path).map_err(|error| error.to_string()),
        Some(wanted) => crate::host::all_sessions(agent_dir, usize::MAX).into_iter().find(|summary| summary.id.starts_with(wanted) || summary.title == wanted).map(|summary| summary.path).ok_or_else(|| format!("No session matches \"{wanted}\" (an id, the start of one, or a name)")),
    }
}

// The lock `voice start`, `voice stop` and a hold that's ending take before touching the hold's
// record, so no two of them act on it at once. It's let go when the file is dropped.
fn lock(agent_dir: &Path) -> Result<std::fs::File, String> {
    let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(agent_dir.join(LOCK_FILE)).map_err(|error| error.to_string())?;
    file.lock().map_err(|error| error.to_string())?;
    Ok(file)
}

// --- Push-to-talk ---

// `piccosta call [assistant | <session>]`, `piccosta hangup`, `piccosta voice start`, `piccosta
// voice stop`, and `voice hold <call id>`, the process start leaves listening. Output: an
// error to print, or Ok when it's done.
pub async fn run(arguments: &[String]) -> Result<(), String> {
    let agent_dir = crate::config::agent_dir();
    let hold_file = agent_dir.join(HOLD_FILE);
    match arguments.first().map(String::as_str) {
        Some("call") => set_call(&agent_dir, Some(&session_named(&agent_dir, arguments.get(1).map(String::as_str))?)).map_err(|error| error.to_string()),
        Some("hangup") => set_call(&agent_dir, None).map_err(|error| error.to_string()),
        Some("start") => {
            // Off a call, push-to-talk is someone else's button.
            let Some((target, id)) = current(&agent_dir) else { return Ok(()) };
            let settings = crate::config::Settings::load(&agent_dir, std::env::current_dir().ok().as_deref());
            text_setting(&settings, "voice.listen").ok_or("no listen command: set voice.listen in settings")?;
            let _lock = lock(&agent_dir)?;
            // A hold still listening (its release was missed) ends first, and sends what it heard.
            stop(&hold_file);
            // The hold runs on after this returns, in a process group of its own, so the
            // push-to-talk script that ran this can end without taking it along. Its record is
            // written before the lock goes, so a `voice stop` right after finds it.
            let exe = std::env::current_exe().map_err(|error| error.to_string())?;
            let log = log().map_err(|error| error.to_string())?;
            let mut hold = std::process::Command::new(exe);
            hold.arg("voice").arg("hold").arg(&id).arg(&target).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log);
            let hold = std::os::unix::process::CommandExt::process_group(&mut hold, 0).spawn().map_err(|error| format!("couldn't start listening: {error}"))?;
            write_record(&hold_file, json!({ "pid": hold.id(), "started": crate::proc::start(hold.id() as i32).map(|start| format!("{} {start}", crate::proc::boot_id())) })).map_err(|error| error.to_string())
        }
        Some("stop") => {
            let _lock = lock(&agent_dir)?;
            stop(&hold_file);
            Ok(())
        }
        Some("hold") => {
            let (Some(id), Some(target)) = (arguments.get(1), arguments.get(2)) else { return Err("voice hold needs a call id and its session file".into()) };
            let listened = hold(&agent_dir, id, Path::new(target)).await;
            // However it ended, its record goes, unless a newer start has written its own.
            let _lock = lock(&agent_dir)?;
            if live_record(&hold_file) == Some(std::process::id()) {
                let _ = std::fs::remove_file(&hold_file);
            }
            listened
        }
        _ => Err("usage: piccosta voice start | piccosta voice stop".into()),
    }
}

// End the hold that's listening, if its record names one still running: SIGTERM, which it answers
// by closing the listen command's stdin. The caller holds the lock.
fn stop(hold_file: &Path) {
    if let Some(pid) = live_record(hold_file) {
        // SAFETY: kill() only sends a signal.
        unsafe { libc::kill(pid as i32, libc::SIGTERM) };
    }
    let _ = std::fs::remove_file(hold_file);
}

// Listen until SIGTERM (`voice stop`) or the time limit, then deliver what was heard to the session
// in the call with id `id` (its session's file, `target`), if that call is still on. One that ends
// first ends the listening.
async fn hold(agent_dir: &Path, id: &str, target: &Path) -> Result<(), String> {
    let settings = crate::config::Settings::load(agent_dir, std::env::current_dir().ok().as_deref());
    let listen = text_setting(&settings, "voice.listen").ok_or("no listen command: set voice.listen in settings")?;
    let limit = settings.get("voice.maxSeconds").and_then(Value::as_u64).unwrap_or(MAX_HOLD_SECONDS);
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).map_err(|error| error.to_string())?;
    let mut listener = shell(&listen).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().map_err(|error| format!("couldn't run voice.listen: {error}"))?;
    let stdin = listener.stdin.take();
    let mut lines = tokio::io::BufReader::new(listener.stdout.take().ok_or("voice.listen has no stdout")?).lines();
    let mut heard = Vec::new();
    // While the key is down the lines are read as they come, so a recognizer that streams is never
    // held up by a full pipe. One that ends by itself has nothing more to say.
    let time_up = tokio::time::sleep(Duration::from_secs(limit));
    let call_ended = ended(agent_dir, id);
    tokio::pin!(time_up, call_ended);
    loop {
        tokio::select! {
            _ = terminate.recv() => break,
            () = &mut call_ended => {
                kill_group(&mut listener).await;
                return Ok(());
            }
            () = &mut time_up => {
                eprintln!("piccosta voice: no stop after {limit} s (voice.maxSeconds); stopped listening");
                break;
            }
            line = lines.next_line() => match line {
                Ok(Some(line)) => heard.push(line),
                _ => break,
            },
        }
    }
    // The key is up: closing stdin tells the listen command to finish, which it has
    // LISTEN_FINISH_SECONDS to do; then it, and anything it started that still holds the pipe, is
    // killed, and what it wrote by then is what was heard.
    drop(stdin);
    let finish = async {
        while let Ok(Some(line)) = lines.next_line().await {
            heard.push(line);
        }
        let _ = listener.wait().await;
    };
    if tokio::time::timeout(Duration::from_secs(LISTEN_FINISH_SECONDS), finish).await.is_err() {
        eprintln!("piccosta voice: voice.listen didn't finish within {LISTEN_FINISH_SECONDS} s of the key coming up; killed");
        kill_group(&mut listener).await;
    }
    let words = heard.iter().map(|line| line.trim()).filter(|line| !line.is_empty()).collect::<Vec<_>>().join(" ");
    if words.is_empty() || over(agent_dir, id) {
        return Ok(());
    }
    // The assistant hears from the user with the time, as `piccosta tell` sends it.
    let text = if crate::manager::session_file(agent_dir).as_deref() == Some(target) { crate::manager::from_user(&words) } else { words };
    crate::runtime::send(target, &text, None).map(|_| ())
}

// --- Speaking ---

// The quick model and what it takes to call it, carried onto the speech worker.
#[derive(Clone)]
struct Quick {
    model: Model,
    config: Option<ProviderConfig>,
    credentials: crate::auth::CredentialPaths,
    transport: String,
    cwd: PathBuf,
}

// A turn to say, as the speech worker gets it: the speak command, the words it answers, what the
// quick model reads (no quick model for a turn the user didn't start), and the session's answer as
// it streams.
struct Turn {
    speak: String,
    // Cancelled as its session closes (jobs.rs shutdown), which ends it said or waiting.
    closed: CancellationToken,
    // The id of the call it's said in, whose end ends it too.
    call: String,
    session_id: String,
    words: String,
    transcript: String,
    quick: Option<Quick>,
    answer: mpsc::Receiver<String>,
}

enum Job {
    Turn(Box<Turn>),
    // Answered once every turn queued before it has been said.
    Drain(oneshot::Sender<()>),
}

// The speech worker, one a process, which says turns one after another on a thread of its own (an
// agent runs on whatever runtime its frontend has). Started with the first turn said; `stop`
// ends it, and `thread` is its thread with a receiver that closes as that thread returns.
struct Worker {
    jobs: mpsc::Sender<Job>,
    stop: CancellationToken,
    thread: Mutex<Option<(std::thread::JoinHandle<()>, oneshot::Receiver<()>)>>,
}

static WORKER: OnceLock<Worker> = OnceLock::new();
// What quick answers cost, as (session id, provider, model, usage), until that session's agent
// counts it. A session's task watches it, so what comes while the session is idle is counted then.
pub type Spent = Vec<(String, String, String, Usage)>;
static SPENT: LazyLock<watch::Sender<Spent>> = LazyLock::new(|| watch::Sender::new(Vec::new()));

fn worker() -> &'static Worker {
    WORKER.get_or_init(|| {
        let (jobs, mut queue) = mpsc::channel(TURN_QUEUE);
        let (stop, (ended, thread_ended)) = (CancellationToken::new(), oneshot::channel::<()>());
        let stopping = stop.clone();
        let thread = std::thread::spawn(move || {
            // Dropped last, once the runtime and the turns still queued are gone.
            let _ended = ended;
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return };
            runtime.block_on(async {
                loop {
                    match tokio::select! { job = queue.recv() => job, () = stopping.cancelled() => None } {
                        Some(Job::Turn(turn)) => play(*turn, &stopping).await,
                        Some(Job::Drain(done)) => drop(done.send(())),
                        None => break,
                    }
                }
            });
        });
        Worker { jobs, stop, thread: Mutex::new(Some((thread, thread_ended))) }
    })
}

// Stop the speech worker, as a frontend exits: the turns waiting are dropped, the speak command
// saying one is killed with everything it started, and the worker's thread is joined, within
// STOP_SECONDS. With `drain` (a print run) what's queued is said first, for up to DRAIN_SECONDS.
pub async fn shutdown(drain: bool) {
    let Some(worker) = WORKER.get() else { return };
    let (done, said) = oneshot::channel();
    let drained = async {
        if worker.jobs.send(Job::Drain(done)).await.is_ok() {
            let _ = said.await;
        }
    };
    if drain && tokio::time::timeout(Duration::from_secs(DRAIN_SECONDS), drained).await.is_err() {
        note(&format!("still speaking after {DRAIN_SECONDS} s; the rest isn't said"));
    }
    worker.stop.cancel();
    let Some((thread, ended)) = worker.thread.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take() else { return };
    match tokio::time::timeout(Duration::from_secs(STOP_SECONDS), ended).await {
        Ok(_) => drop(thread.join()),
        Err(_) => note(&format!("the speech worker didn't stop within {STOP_SECONDS} s")),
    }
}

// What quick answers cost in session `session_id` since this was last asked, as (session id,
// provider, model, usage), each for a usage entry of kind QUICK_KIND.
pub fn spent(session_id: &str) -> Spent {
    let mut mine = Vec::new();
    SPENT.send_if_modified(|spent| {
        (mine, *spent) = std::mem::take(spent).into_iter().partition(|(id, ..)| id == session_id);
        !mine.is_empty()
    });
    mine
}

// Changes when a quick answer's cost comes in, for a session's task to count it (`spent`).
pub fn spent_changes() -> watch::Receiver<Spent> {
    SPENT.subscribe()
}

// A session's events, with its answers said while the user is on a call with it. Input: the
// settings, the agent's config (for the quick model), the session, and where the events go on to.
// Output: the sink to give the agent, which is `emit` itself when `voice.speak` isn't set.
pub fn speaking(settings: &crate::config::Settings, config: &AgentConfig, session: &crate::session::Session, mut emit: EventSink) -> EventSink {
    let Some(speak) = text_setting(settings, "voice.speak") else { return emit };
    let closed = config.tool_shared.jobs.closed.clone();
    let quick = text_setting(settings, "voice.quick").and_then(|name| config.registry.resolve(&name).ok().cloned()).map(|model| Quick {
        config: config.registry.provider_config(&model.provider).cloned(),
        model,
        credentials: config.credentials.clone(),
        transport: config.transport.clone(),
        cwd: config.cwd.clone(),
    });
    let (session_id, file, agent_dir) = (session.id().to_string(), session.file().map(Path::to_path_buf), crate::config::agent_dir());
    let mut transcript = String::new();
    session.context().messages.iter().for_each(|message| remember(&mut transcript, message));
    // The turn being said: where the session's answer goes, a piece at a time. Each message from
    // the user ends the one before's and starts its own, while the call is with this session; a
    // job's end or a lane's news starts one when none is under way.
    let mut turn: Option<mpsc::Sender<String>> = None;
    Box::new(move |event: &Event| {
        match event {
            Event::MessageStart { message: message @ (AgentMessage::User(_) | AgentMessage::Custom(_)) } => {
                let user = matches!(message, AgentMessage::User(_));
                if user || turn.is_none() {
                    let on_call = current(&agent_dir).filter(|(called, _)| Some(called) == file.as_ref());
                    turn = on_call.map(|(_, call)| (call, mpsc::channel(ANSWER_QUEUE))).and_then(|(call, (sender, answer))| {
                        let words = if let AgentMessage::User(said) = message { said.content.text() } else { String::new() };
                        let said = Turn { speak: speak.clone(), closed: closed.clone(), call, session_id: session_id.clone(), words, transcript: transcript.clone(), quick: quick.clone().filter(|_| user), answer };
                        worker().jobs.try_send(Job::Turn(Box::new(said))).map(|()| sender).inspect_err(|_| note("too many turns waiting to be said; this one isn't")).ok()
                    });
                }
            }
            Event::MessageUpdate { assistant_message_event, .. } => {
                if assistant_message_event.get("type").and_then(Value::as_str) == Some("text_delta")
                    && let (Some(turn), Some(delta)) = (&turn, assistant_message_event.get("delta").and_then(Value::as_str))
                {
                    // Never waited on: what the worker has no room for is dropped.
                    let _ = turn.try_send(delta.to_string());
                }
            }
            Event::MessageEnd { message } => remember(&mut transcript, message),
            // The run is over: the speak command gets the rest, then its stdin closes.
            Event::AgentSettled { .. } => turn = None,
            _ => {}
        }
        emit(event)
    })
}

// Add a message's text to the end of the conversation the quick model reads, keeping the last
// QUICK_CONTEXT_CHARS. Tool calls and their results are left out: they're long and not for speech.
fn remember(transcript: &mut String, message: &AgentMessage) {
    let (speaker, text) = match message {
        AgentMessage::User(user) => ("User", user.content.text()),
        AgentMessage::Assistant(reply) => ("Assistant", TextOrBlocks::Blocks(reply.content.clone()).text()),
        _ => return,
    };
    if text.trim().is_empty() {
        return;
    }
    transcript.push_str(&format!("{speaker}: {}\n\n", text.trim()));
    if transcript.len() > QUICK_CONTEXT_CHARS {
        let cut = (transcript.len() - QUICK_CONTEXT_CHARS..).find(|&index| transcript.is_char_boundary(index)).unwrap_or(0);
        transcript.drain(..cut);
    }
}

// Say one turn: start the speak command, give it the quick answer when there's a quick model, then
// the session's answer, which has waited meanwhile; close its stdin and give it its time to finish.
// A speak command that stops taking lines, a worker told to stop, the session closing or the call
// ending ends the turn at once: the command and everything it started are killed, and the rest of
// the answer is let go. A turn whose session closed or whose call ended while it waited isn't
// started.
async fn play(mut turn: Turn, stop: &CancellationToken) {
    let agent_dir = crate::config::agent_dir();
    if turn.closed.is_cancelled() || over(&agent_dir, &turn.call) {
        return;
    }
    let mut command = shell(&turn.speak);
    command.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(log().map_or(Stdio::null(), Stdio::from));
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return note(&format!("couldn't run voice.speak: {error}")),
    };
    let (group, closed, call) = (child.id(), turn.closed.clone(), turn.call.clone());
    let cut = stop.child_token();
    let mut speaker = Speaker { stdin: child.stdin.take(), pending: String::new(), cut: cut.clone() };
    let said = async {
        if let Some(quick) = &turn.quick {
            if tokio::time::timeout(Duration::from_secs(QUICK_SECONDS), quick_answer(quick, &turn, &mut speaker)).await.is_err() {
                note(&format!("voice.quick gave no answer within {QUICK_SECONDS} s"));
            }
            speaker.add("", true).await;
        }
        while let Some(delta) = turn.answer.recv().await {
            speaker.add(&delta, false).await;
        }
        speaker.add("", true).await;
        speaker.stdin = None;
        tokio::time::timeout(Duration::from_secs(SPEAK_EXIT_SECONDS), child.wait()).await.is_ok() || {
            note(&format!("voice.speak still running {SPEAK_EXIT_SECONDS} s after its turn ended; killed"));
            false
        }
    };
    let finished = tokio::select! { exited = said => exited, () = cut.cancelled() => false, () = closed.cancelled() => false, () = ended(&agent_dir, &call) => false };
    if !finished {
        kill_group(&mut child).await;
    } else if let Some(group) = group {
        // A speak command can exit leaving a child still playing (a backgrounded player); its group
        // goes too. Linux keeps a group's number reserved while any member lives, so this reaches
        // only them; with none left it finds nothing.
        // SAFETY: killpg() only sends a signal.
        unsafe { libc::killpg(group as i32, libc::SIGKILL) };
    }
}

// The speak command's stdin, what the model has written that isn't ready to say yet, and the token
// that ends the turn when the command stops taking lines.
struct Speaker {
    stdin: Option<ChildStdin>,
    pending: String,
    cut: CancellationToken,
}

impl Speaker {
    // Add text the model wrote, and send the speak command the part that's ready: up to the last
    // sentence end (., ! or ? before a space) or line break, or all of it once the text is
    // `finished`. Each piece goes as one line, its line breaks made spaces. A speak command that
    // has gone, or hasn't taken a line in SPEAK_WRITE_SECONDS, ends the turn.
    async fn add(&mut self, text: &str, finished: bool) {
        self.pending.push_str(text);
        let pending = &self.pending;
        let boundary = pending.char_indices().zip(pending.chars().skip(1)).filter(|((_, character), next)| *character == '\n' || (".!?".contains(*character) && next.is_whitespace()));
        let Some(end) = (if finished { Some(pending.len()) } else { boundary.map(|((index, character), _)| index + character.len_utf8()).last() }) else { return };
        let piece = self.pending.drain(..end).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
        if let (false, Some(stdin)) = (piece.is_empty(), &mut self.stdin)
            && !matches!(tokio::time::timeout(Duration::from_secs(SPEAK_WRITE_SECONDS), stdin.write_all(format!("{piece}\n").as_bytes())).await, Ok(Ok(())))
        {
            note("voice.speak stopped taking lines; it's killed and the rest of this turn isn't said");
            self.stdin = None;
            self.cut.cancel();
        }
    }
}

// Ask the quick model for its answer to the turn's words, with the conversation's end, and give the
// speaker its text as it streams (or all at once, from a provider that didn't stream it). Uncached,
// under a session id of its own, and never written anywhere; what it cost goes to SPENT, for the
// session's statistics. A model that fails says nothing.
async fn quick_answer(quick: &Quick, turn: &Turn, speaker: &mut Speaker) -> Option<()> {
    let credential = crate::auth::credential_for(&quick.model.provider, quick.config.as_ref(), &quick.credentials).await.map_err(|error| note(&format!("voice.quick: {error}"))).ok()?;
    let messages = [
        Message::System(SystemMessage { content: TextOrBlocks::Text(QUICK_PROMPT.into()), sections: None, tools_added: None, tools_removed: None, timestamp: 0 }),
        Message::User(UserMessage { content: TextOrBlocks::Text(format!("The conversation so far, its end:\n\n{}The user just said: {}", turn.transcript, turn.words)), timestamp: crate::clock::now_ms() }),
    ];
    let session_id = uuid::Uuid::now_v7().to_string();
    let (sender, mut events) = mpsc::unbounded_channel();
    let request = Request {
        model: &quick.model,
        messages: &messages,
        credential: &credential,
        session_id: Some(&session_id),
        cache: false,
        transport: &quick.transport,
        thinking_level: crate::provider::clamp_thinking_level(&quick.model, "off"),
        session_file: None,
        cwd: &quick.cwd,
        cancel: CancellationToken::new(),
    };
    let mut streamed = false;
    let read = async {
        while let Some(event) = events.recv().await {
            if let StreamEvent::TextDelta { delta, .. } = event {
                streamed = true;
                speaker.add(&delta, false).await;
            }
        }
    };
    let (reply, ()) = tokio::join!(crate::provider::stream(request, sender), read);
    if !streamed {
        speaker.add(&TextOrBlocks::Blocks(reply.content.clone()).text(), false).await;
    }
    if let Some(error) = &reply.error_message {
        note(&format!("voice.quick: {error}"));
    }
    (reply.usage.input + reply.usage.output > 0).then(|| SPENT.send_modify(|spent| spent.push((turn.session_id.clone(), reply.provider, reply.model, reply.usage))))
}
