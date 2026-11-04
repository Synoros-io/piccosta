// Voice, brought by the user. piccosta records and synthesizes nothing: commands named in
// settings do that, and piccosta passes text and captured audio between them. Voice is a call with one session at
// a time, the way the user is on a call with a person:
//
// - A call starts with `/call` in the terminal UI (the session shown there, or `/call assistant`)
//   or `piccosta call [assistant | <session id or name>]` (the assistant when none is named). It's
//   a record in the agent dir naming the session's file and an id of its own; starting another ends
//   it, and so do `/hangup`, `piccosta hangup` and closing that session (a socket close, the UI's
//   /close). `voice.call`, when it's set, runs for the length of the call (below).
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
// - `voice.reply`, optional: the first voice, from the user's audio itself. When set, voice.listen
//   must also save a complete mono WAV at $PICCOSTA_VOICE_AUDIO (a private path supplied for this
//   hold). Nothing changes for listen when reply is unset. The reply command gets that WAV on
//   stdin, plays its own audio, writes its reply as a UTF-8 line to stdout as soon as it knows
//   it, and exits after its one sentence plays. Quick and the session generate meanwhile.
//   $PICCOSTA_VOICE_REPLY_AUDIO is a scratch WAV path it can use, for example with
//   `lfm reply --server "$socket" -o "$PICCOSTA_VOICE_REPLY_AUDIO" && player "$PICCOSTA_VOICE_REPLY_AUDIO"`.
//   It runs in the same queue as speak, before quick and the session, never over either. Its
//   process group ends with the call or session; capture and reply files are removed afterwards.
// - `voice.speak`, a shell command, starts once per spoken turn. piccosta streams UTF-8 lines,
//   closes stdin at the turn's end, and waits for playback to exit. With reply set, a ready session
//   line drops early lines still in piccosta's queue; lines already handed to speak finish first.
//   The command owns playback of the lines it accepts. Turns never speak over each other.
// - `voice.call`, an optional shell command, starts with each call and is killed, with everything it
//   started, when the call ends. It's for what listening and speaking share and shouldn't start
//   over each push or turn, like a speech model held loaded (`lfm serve`) that the listen and speak
//   commands use. A detached supervisor owns its process group until the call ends.
// - `voice.quick`, optional: a fast model (provider/id) that answers each of the user's messages
//   first, from the end of the conversation, while the session's model does the work. With reply
//   set, at most two sentences enter the queue; otherwise all quick's lines precede the session.
//   It's spoken only; its cost goes into session statistics.
//   Quick sees reply's text. When early speech finishes or the session overtakes it, one trailing
//   message says what was spoken, asking for only what's missing (which can be nothing).
//   No system prompt or earlier message changes; the session never waits to speak its first line.

use crate::agent::{AgentConfig, Event, EventSink};
use crate::message::{AgentMessage, CustomMessage, Message, SystemMessage, TextOrBlocks, Usage, UserMessage};
use crate::model::ModelRegistry;
use crate::provider::{Request, StreamEvent};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;

// In the agent dir: the record (pid and start time) of the hold that's listening, the lock that
// lets one `voice start` or `voice stop` at a time touch it, and the call's record.
const HOLD_FILE: &str = "voice.pid";
const LOCK_FILE: &str = "voice.lock";
const CALL_FILE: &str = "call.json";
// Supervisors hold this through cleanup, so a replacement can't bind a socket the old call still owns.
const CALL_LOCK_FILE: &str = "voice.call.lock";
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
// A pipe holds much more than a line of speech: ten seconds without accepting a write means
// the command has stopped reading. After stdin closes, allow five minutes to finish playback,
// including paragraphs without sentence punctuation. A stalled command's whole group is killed.
const SPEAK_WRITE_SECONDS: u64 = 10;
// The quick model gets QUICK_SECONDS: it's meant to answer in one. A print run waits up to
// DRAIN_SECONDS, once its answer is done, for what's still being said. A frontend that exits gives
// the worker STOP_SECONDS to kill the speak command and end: a killed process is reaped at once, so
// five is room for a loaded machine and still no wait anyone notices.
const SPEAK_EXIT_SECONDS: u64 = 300;
const QUICK_SECONDS: u64 = 30;
const DRAIN_SECONDS: u64 = 300;
const STOP_SECONDS: u64 = 5;
// How often listening and speaking look at the call's record to see that their call is still on: a
// spoken line is a sentence, a couple of seconds, so half a second ends speech within the line
// being said, and reading a small file twice a second costs nothing.
const CALL_POLL: Duration = Duration::from_millis(500);
// Give a call's command a second to unlink sockets and close devices before killing its leftovers.
const CALL_EXIT_GRACE: Duration = Duration::from_secs(1);
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
const SAID_TYPE: &str = "voice-said";
pub(crate) const INPUT_TYPE: &str = "voice-input";
// The baseline is at most two sentences; the session, not this layer, owns the full answer.
const QUICK_SENTENCES: usize = 2;
const QUICK_PROMPT: &str = "You are the voice that answers first. The user has just spoken to their assistant, which is now working on the full answer and will speak it after you. Say at once, in one to three short sentences, what you can from what you know and the conversation below: answer when you can, else say what is being looked into. Don't claim to have done or checked anything. Write for speech: plain sentences, no markdown, lists or code.";
pub const SILENT: &str = "[PICCOSTA_SILENT]";
// What a message said on a call carries, so its answer is written to be heard.
const SAID_ON_A_CALL: &str = "said on a voice call: answer in a few plain spoken sentences, no markdown or lists";
const LAYERED_PROMPT: &str = "You are the quick baseline voice. The user has just spoken to their assistant, which is now working on the full answer. Say at once, in at most two tiny sentences, what you can from what you know and the conversation below: answer when you can, else say what is being looked into. Lead with the shortest useful fact. Each sentence is just a few words, one fact, never a restatement of the question or an elaboration. Don't claim to have done or checked anything. Don't repeat what the first voice already said; add what's missing. Write for speech: plain sentences, no markdown, lists or code.";

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

// A process's record, its pid and its boot and start time, for live_record to find it by.
fn running(pid: u32) -> Value {
    json!({ "pid": pid, "started": crate::proc::start(pid as i32).map(|start| format!("{} {start}", crate::proc::boot_id())) })
}

// Publish the call before starting anything. All transitions share the voice lock; old supervisors
// see their id disappear, and only ever signal the children they own.
pub fn set_call(agent_dir: &Path, file: Option<&Path>) -> std::io::Result<()> {
    let _lock = lock(agent_dir).map_err(std::io::Error::other)?;
    let record = agent_dir.join(CALL_FILE);
    let Some(file) = file else { return remove_call(&record) };
    let id = uuid::Uuid::now_v7().to_string();
    write_record(&record, json!({ "session": file, "id": id }))?;
    let mut supervisor = std::process::Command::new(std::env::current_exe()?);
    supervisor.arg("voice").arg("run-call").arg(id).env("PICCOSTA_AGENT_DIR", agent_dir).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log()?);
    match std::os::unix::process::CommandExt::process_group(&mut supervisor, 0).spawn() {
        Ok(mut child) => { std::thread::spawn(move || child.wait()); Ok(()) }
        Err(error) => { let _ = remove_call(&record); Err(error) }
    }
}

fn remove_call(record: &Path) -> std::io::Result<()> {
    std::fs::remove_file(record).or_else(|error| if error.kind() == std::io::ErrorKind::NotFound { Ok(()) } else { Err(error) })
}

// Compare and remove under the same lock, so closing one session never hangs up a newer call.
pub fn hang_up(agent_dir: &Path, file: &Path) {
    let Ok(_lock) = lock(agent_dir) else { return };
    if call(agent_dir).as_deref() == Some(file) {
        let _ = remove_call(&agent_dir.join(CALL_FILE));
    }
}

// Keep the group's leader unreaped until both signals have gone to its group. Even a shell that
// exits after `command &` keeps its pid reserved, so teardown can't signal a reused group number.
async fn run_call(agent_dir: &Path, id: &str) -> Result<(), String> {
    let command_lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(agent_dir.join(CALL_LOCK_FILE)).map_err(|e| e.to_string())?;
    command_lock.lock().map_err(|e| e.to_string())?;
    if !reading(agent_dir).map_err(|e| e.to_string())?.is_some_and(|(_, now)| now == id) { return Ok(()) }
    let settings = crate::config::Settings::load(agent_dir, std::env::current_dir().ok().as_deref());
    // This supervisor also owns captures left by a receiver that exited before saying them.
    let _recordings = Recording(agent_dir.join("voice").join(uuid::Uuid::parse_str(id).map_err(|e| e.to_string())?.to_string()));
    let command = text_setting(&settings, "voice.call");
    // The call ending, or this process being told to end (a logout, a kill), takes the same way out,
    // so its command never outlives it. The signals are caught before the command starts, so none
    // can end this process between the two.
    use tokio::signal::unix::{signal, SignalKind};
    let caught = |kind| signal(kind).map_err(|error| format!("couldn't catch a signal: {error}"));
    let (mut term, mut hangup, mut interrupt) = (caught(SignalKind::terminate())?, caught(SignalKind::hangup())?, caught(SignalKind::interrupt())?);
    let mut child = command
        .map(|command| {
            shell(&command).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log().map_err(|e| e.to_string())?).spawn().map_err(|e| e.to_string())})
        .transpose()?;
    tokio::select! {
        () = ended(agent_dir, id) => {}
        _ = term.recv() => {}
        _ = hangup.recv() => {}
        _ = interrupt.recv() => {}
    }
    if let Some(child) = &mut child {
        if let Some(pid) = child.id() {
            // SAFETY: this unreaped child still owns the process-group number.
            unsafe { libc::killpg(pid as i32, libc::SIGTERM) };
            tokio::time::sleep(CALL_EXIT_GRACE).await;
        }
        kill_group(child).await;
    }
    Ok(())
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

// The lock for call transitions and the hold's record. It's let go when the file is dropped.
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
        Some("run-call") => run_call(&agent_dir, arguments.get(1).ok_or("voice run-call needs a call id")?).await,
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
            write_record(&hold_file, running(hold.id())).map_err(|error| error.to_string())
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
    let recording = if text_setting(&settings, "voice.reply").is_some() {
        Some(Recording::new(agent_dir, id)?)
    } else {
        None
    };
    let mut command = shell(&listen);
    if let Some(recording) = &recording {
        command.env("PICCOSTA_VOICE_AUDIO", recording.0.join("input.wav"));
    }
    let mut listener = command
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().map_err(|error| format!("couldn't run voice.listen: {error}"))?;
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
    // The session knows from its first word that its answer will be heard, not read: the words go
    // with a note of it, in the user's own message (new text at the end, so no cache is disturbed).
    let words = format!("{words} ({SAID_ON_A_CALL})");
    let text = if crate::manager::session_file(agent_dir).as_deref() == Some(target) { crate::manager::from_user(&words) } else { words };
    if let Some(recording) = recording
        && recording.0.join("input.wav").is_file()
    {
        if crate::runtime::send_voice(target, &text, &recording.0)? {
            std::mem::forget(recording);
        }
        return Ok(());
    }
    crate::runtime::send(target, &text, None).map(|_| ())
}

// The directory belongs to one capture, not to a session or the user's recorder. The speaker
// owns it after delivery; an undelivered capture is removed by the hold instead.
struct Recording(PathBuf);
impl Recording {
    fn new(agent_dir: &Path, call: &str) -> Result<Self, String> {
        let dir = agent_dir
            .join("voice")
            .join(call)
            .join(uuid::Uuid::now_v7().to_string());
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&dir).map_err(|e| e.to_string())?;
        Ok(Self(dir))
    }
}
impl Drop for Recording {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// The early voices complete while the main model works. The agent takes their note at its next
// tool boundary (or before settling), then appends it as an ordinary message, never a prompt edit.
#[derive(Clone, Default)]
pub struct State(Arc<Mutex<Option<oneshot::Receiver<Option<CustomMessage>>>>>);
impl State {
    pub async fn finish(&self, cancel: &CancellationToken, wait: bool) -> Option<CustomMessage> {
        let mut receiver = self.0.lock().unwrap_or_else(|p| p.into_inner()).take()?;
        // Tools don't wait for speech. Take a ready note, otherwise keep working; the first
        // answer line will overtake the early voices and release the note before settling.
        match receiver.try_recv() {
            Ok(note) => return note,
            Err(oneshot::error::TryRecvError::Empty) if !wait => {
                *self.0.lock().unwrap_or_else(|p| p.into_inner()) = Some(receiver);
                return None;
            }
            Err(oneshot::error::TryRecvError::Closed) => return None,
            _ => {}
        }
        tokio::select! { result = receiver => result.ok().flatten(), () = cancel.cancelled() => None }
    }
}

enum Speech {
    Text(String),
    End,
}

// Only this queue feeds playback. Ready marks the end of quick's queued lines.
enum Line {
    Say(&'static str, String),
    Ready,
}

// --- Speaking ---

// The quick model and what it takes to call it, carried onto the speech worker.
#[derive(Clone)]
struct Quick {
    name: String,
    credentials: crate::auth::CredentialPaths,
    transport: String,
    cwd: PathBuf,
}

// A turn to say, as the speech worker gets it: the speak command, the words it answers, what the
// quick model reads (no quick model for a turn the user didn't start), and the session's answer as
// it streams.
struct Turn {
    speak: Option<String>,
    // Cancelled as its session closes (jobs.rs shutdown), which ends it said or waiting.
    closed: CancellationToken,
    // The id of the call it's said in, whose end ends it too.
    call: String,
    session_id: String,
    words: String,
    transcript: String,
    quick: Option<Quick>,
    reply: Option<String>,
    recording: Option<Recording>,
    earlier: Option<oneshot::Sender<Option<CustomMessage>>>,
    answer: mpsc::Receiver<Speech>,
    started: Instant,
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
    let Some((thread, ended)) = worker.thread.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take() else { return ;
    };
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
// Output: the sink to give the agent. It also owns delivered recordings without a speak command.
pub fn speaking(settings: &crate::config::Settings, config: &AgentConfig, session: &crate::session::Session, mut emit: EventSink,
) -> EventSink {
    let speak = text_setting(settings, "voice.speak");
    let closed = config.tool_shared.jobs.closed.clone();
    let state = config.voice.clone();
    let reply = text_setting(settings, "voice.reply");
    let quick = text_setting(settings, "voice.quick").map(|name| Quick {
        name,
        credentials: config.credentials.clone(),
        transport: config.transport.clone(),
        cwd: config.cwd.clone(),
    });
    let (session_id, file, agent_dir) = (session.id().to_string(), session.file().map(Path::to_path_buf), crate::config::agent_dir(),
    );
    let mut transcript = String::new();
    session.context().messages.iter().for_each(|message| remember(&mut transcript, message));
    // The turn being said: where the session's answer goes, a piece at a time. Each message from
    // the user ends the one before's and starts its own, while the call is with this session; a
    // job's end or a lane's news starts one when none is under way.
    let mut turn: Option<mpsc::Sender<Speech>> = None;
    Box::new(move |event: &Event| {
        match event {
            Event::MessageStart { message: message @ (AgentMessage::User(_) | AgentMessage::Custom(_)) } => {
                if matches!(message, AgentMessage::Custom(note) if note.custom_type == SAID_TYPE) {
                    return emit(event);
                }
                // Claim a delivered capture even if the call ended or speech was disabled meanwhile.
                let recording = match message {
                    AgentMessage::Custom(input) if input.custom_type == INPUT_TYPE => input.details.as_ref()
                        .and_then(|d| d["recording"].as_str()).map(PathBuf::from)
                        .filter(|dir| dir.parent().and_then(Path::parent) == Some(agent_dir.join("voice").as_path())
                            && [dir.file_name(), dir.parent().and_then(Path::file_name)].into_iter()
                                .all(|name| name.and_then(|n| n.to_str()).is_some_and(|n| uuid::Uuid::parse_str(n).is_ok())))
                        .map(Recording),
                    _ => None,
                };
                let user = matches!(message, AgentMessage::User(_))
                    || matches!(message, AgentMessage::Custom(note) if note.custom_type == INPUT_TYPE);
                if user || turn.is_none() {
                    let on_call = current(&agent_dir).filter(|(called, _)| Some(called) == file.as_ref() && (speak.is_some() || reply.is_some()));
                    turn = on_call.map(|(_, call)| (call, mpsc::channel(ANSWER_QUEUE))).and_then(|(call, (sender, answer))| {
                        let words = match message {
                            AgentMessage::User(said) => said.content.text(),
                            AgentMessage::Custom(said) if user => said.content.text(),
                            _ => String::new(),
                        };
                        let (earlier, done) = oneshot::channel();
                        let earlier = reply.is_some().then(|| {
                            *state.0.lock().unwrap_or_else(|p| p.into_inner()) = Some(done);
                            earlier
                        });
                        // A recording plays only under the call it was made in (one from a call just
                        // hung up is dropped, and so deleted, when another has started).
                        let recording = recording.filter(|recording| recording.0.parent().and_then(Path::file_name).and_then(|name| name.to_str()) == Some(call.as_str()));
                        let said = Turn { speak: speak.clone(), closed: closed.clone(), call, session_id: session_id.clone(), words,
                            transcript: transcript.clone(), quick: quick.clone().filter(|_| user && speak.is_some()), reply: reply.clone(), recording,
                            earlier, answer, started: Instant::now() };
                        worker().jobs.try_send(Job::Turn(Box::new(said))).map(|()| sender).inspect_err(|_| note("too many turns waiting to be said; this one isn't")).ok()
                    });
                }
            }
            Event::MessageUpdate { assistant_message_event, .. } => {
                if assistant_message_event.get("type").and_then(Value::as_str) == Some("text_delta")
                    && let (Some(turn), Some(delta)) = (&turn, assistant_message_event.get("delta").and_then(Value::as_str),
                    )
                {
                    // Never waited on: what the worker has no room for is dropped.
                    let _ = turn.try_send(Speech::Text(delta.to_string()));
                }
            }
            Event::MessageEnd { message } => {
                if matches!(message, AgentMessage::Assistant(_)) && let Some(turn) = &turn {
                    let _ = turn.try_send(Speech::End);
                }
                remember(&mut transcript, message);
            }
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
        AgentMessage::Assistant(reply) => ("Assistant", TextOrBlocks::Blocks(reply.content.clone()).text(),
        ),
        AgentMessage::Custom(input) if input.custom_type == INPUT_TYPE => {
            ("User", input.content.text())
        }
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

// Generate concurrently, feed one playback process. With reply set, a ready session line drops
// early lines still queued here, not lines already accepted by the command. Without reply,
// quick's original prompt, line boundaries and order stay intact.
async fn play(mut turn: Turn, stop: &CancellationToken) {
    let agent_dir = crate::config::agent_dir();
    if turn.closed.is_cancelled() || over(&agent_dir, &turn.call) { return; }
    // Start loading the synthesizer while the models work, once for the entire turn.
    let mut child = if let Some(command) = &turn.speak {
        match shell(command).stdin(Stdio::piped()).stdout(Stdio::null())
            .stderr(log().map_or(Stdio::null(), Stdio::from)).spawn() {
            Ok(child) => Some(child),
            Err(error) => return note(&format!("couldn't run voice.speak: {error}")),
        }
    } else { None };
    let group = Group(child.as_ref().and_then(tokio::process::Child::id));
    let mut stdin = child.as_mut().and_then(|child| child.stdin.take());
    let (lines, mut queue) = mpsc::unbounded_channel();
    let session_lines = lines.clone();
    let (text, first_text) = oneshot::channel::<String>();
    let overtaken = CancellationToken::new();
    let mut answer = std::mem::replace(&mut turn.answer, mpsc::channel(1).1);
    let mut earlier = turn.earlier.take();
    let recording = turn.recording.take();
    let layered = turn.reply.is_some();
    let (baseline_done, baseline) = oneshot::channel();
    let session = async {
        if !layered { let _ = baseline.await; }
        let mut speaker = Speaker::new(session_lines, "session", usize::MAX, layered);
        while let Some(piece) = answer.recv().await {
            match piece {
                Speech::Text(delta) => speaker.add(&delta, false),
                Speech::End => speaker.add("", true),
            }
            if layered && speaker.sent > 0 { overtaken.cancel(); }
        }
        speaker.add("", true);
        if layered && speaker.sent > 0 { overtaken.cancel(); }
    };
    let quick = async {
        let mut speaker = Speaker::new(lines, "voice.quick", if layered { QUICK_SENTENCES } else { usize::MAX }, layered);
        let first = first_text.await.unwrap_or_default();
        if let Some(quick) = &turn.quick {
            if tokio::time::timeout(Duration::from_secs(QUICK_SECONDS), quick_answer(quick, &turn, &mut speaker, &first, &overtaken)).await.is_err() {
                note(&format!("voice.quick gave no answer within {QUICK_SECONDS} s"));
            }
            if !overtaken.is_cancelled() { speaker.add("", true); }
        }
        let _ = speaker.lines.send(Line::Ready);
        let _ = baseline_done.send(());
    };
    let playback = async {
        let _group = group;
        let mut said = if !overtaken.is_cancelled()
            && let (Some(command), Some(recording)) = (&turn.reply, &recording)
        { reply_audio(command, recording, &turn, stop, text).await.unwrap_or_default() }
        else { drop(text); String::new() };
        // The transcript has arrived and the audio reply is done. No later speech needs the WAVs.
        drop(recording);
        let mut first = [true; 2];
        loop {
            let Some(line) = queue.recv().await else { break };
            let (source, text) = match line {
                Line::Say(source, text) if source == "session" || !overtaken.is_cancelled() => (source, text),
                Line::Say(..) => continue,
                Line::Ready => ("", String::new()),
            };
            if (source == "session" || source.is_empty()) && let Some(earlier) = earlier.take() {
                let note = (!said.trim().is_empty()).then(|| CustomMessage {
                    custom_type: SAID_TYPE.into(),
                    content: TextOrBlocks::Text(format!("During this call, answer in a few spoken sentences, only about what the user asked. piccosta's own notes, including machine load and disk space, are for you, not to read aloud. Said aloud already: {}; don't repeat it, add only what's missing. Your answer above is also being spoken; don't restate it either. If nothing is missing, write exactly {SILENT} and nothing else. That marker means silence, not a word to say. No acknowledgment, filler or stage direction.", said.trim())),
                    display: false, details: None, timestamp: crate::clock::now_ms(),
                });
                let _ = earlier.send(note);
            }
            if text.is_empty() { continue; }
            let Some(input) = &mut stdin else { continue; };
            if !matches!(tokio::time::timeout(Duration::from_secs(SPEAK_WRITE_SECONDS), input.write_all(format!("{text}\n").as_bytes())).await, Ok(Ok(()))) {
                note("voice.speak stopped taking lines; the rest of this turn isn't said");
                if let Some(child) = &mut child { kill_group(child).await; }
                break;
            }
            let index = usize::from(source == "session");
            if first[index] {
                note(&format!("{source} first line to voice.speak in {:.3} s (session {})", turn.started.elapsed().as_secs_f64(), turn.session_id));
                first[index] = false;
            }
            if source != "session" { said.push(' '); said.push_str(&text); }
        }
        // A failed player must release the agent waiting for its note, too.
        drop(earlier.take());
        drop(stdin);
        if let Some(child) = &mut child {
            if tokio::time::timeout(Duration::from_secs(SPEAK_EXIT_SECONDS), child.wait()).await.is_err() {
                note("voice.speak didn't finish its turn; killed");
            }
            kill_group(child).await;
        }
    };
    tokio::select! {
        () = async { tokio::join!(session, quick, playback); } => {},
        () = stop.cancelled() => {},
        () = turn.closed.cancelled() => {},
        () = ended(&agent_dir, &turn.call) => {},
    }
}

// A player's entire process group belongs to its turn, including on cancellation.
struct Group(Option<u32>);
impl Drop for Group {
    fn drop(&mut self) {
        if let Some(group) = self.0 { unsafe { libc::killpg(group as i32, libc::SIGKILL) }; }
    }
}

// Run the BYO audio reply in the speaker's queue. Its stdout is only the words it actually
// played. A failure is silent to the later voices, not an invented acknowledgment.
async fn reply_audio(
    command: &str,
    recording: &Recording,
    turn: &Turn,
    stop: &CancellationToken,
    first_text: oneshot::Sender<String>,
) -> Option<String> {
    let input = std::fs::File::open(recording.0.join("input.wav")).ok()?;
    let mut child = shell(command)
        .env("PICCOSTA_VOICE_REPLY_AUDIO", recording.0.join("reply.wav"))
        .stdin(input)
        .stdout(Stdio::piped())
        .stderr(log().map_or(Stdio::null(), Stdio::from))
        .spawn()
        .map_err(|e| note(&format!("couldn't run voice.reply: {e}")))
        .ok()?;
    // Dropping the outer speech future must kill the reply's player too, even in the middle
    // of a read. The child stays unreaped until this guard exists.
    let _group = Group(child.id());
    let mut output = String::new();
    let read = async {
        use tokio::io::AsyncReadExt;
        let mut lines = tokio::io::BufReader::new(child.stdout.take()?.take(QUICK_CONTEXT_CHARS as u64)).lines();
        let first = lines.next_line().await.ok()?.unwrap_or_default();
        let _ = first_text.send(first.clone());
        output = first;
        while let Some(line) = lines.next_line().await.ok()? { output.push(' '); output.push_str(&line); }
        child.wait().await.ok().filter(|status| status.success())
    };
    let agent_dir = crate::config::agent_dir();
    let finished = tokio::select! {
        result = tokio::time::timeout(Duration::from_secs(LISTEN_FINISH_SECONDS), read) => matches!(result, Ok(Some(_))),
        () = stop.cancelled() => false,
        () = turn.closed.cancelled() => false,
        () = ended(&agent_dir, &turn.call) => false,
    };
    kill_group(&mut child).await;
    if !finished {
        note("voice.reply didn't finish; no reply text is passed on");
        return None;
    }
    note(&format!(
        "voice.reply finished in {:.3} s (session {})",
        turn.started.elapsed().as_secs_f64(),
        turn.session_id
    ));
    Some(output.trim().to_string())
}

// Split streamed text into individual queue entries, including whole non-streaming responses.
struct Speaker {
    lines: mpsc::UnboundedSender<Line>,
    pending: String,
    source: &'static str,
    limit: usize,
    sent: usize,
    sentences: bool,
}

impl Speaker {
    fn new(lines: mpsc::UnboundedSender<Line>, source: &'static str, limit: usize, sentences: bool) -> Self {
        Self { lines, pending: String::new(), source, limit, sent: 0, sentences }
    }

    fn add(&mut self, text: &str, finished: bool) {
        if self.sent == self.limit { return; }
        self.pending.push_str(text);
        while self.sent < self.limit {
            let mut boundaries = self.pending.char_indices().zip(self.pending.chars().skip(1))
                .filter(|((_, character), next)| *character == '\n' || (".!?".contains(*character) && next.is_whitespace()))
                .map(|((index, character), _)| index + character.len_utf8());
            let boundary = if self.sentences { boundaries.next() } else if finished { Some(self.pending.len()) } else { boundaries.last() };
            let Some(end) = boundary.or_else(|| finished.then_some(self.pending.len())) else { break };
            if end == 0 { break; }
            let piece = spoken(&self.pending.drain(..end).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" "));
            if !piece.is_empty() && piece != SILENT {
                let _ = self.lines.send(Line::Say(self.source, piece));
                self.sent += 1;
            }
        }
    }
}

// A line as speech: markdown's marks (a heading's #s, a list's bullet, a quote's >, emphasis and
// code marks) aren't words, so whatever speaks it gets plain text.
fn spoken(line: &str) -> String {
    let line = line.trim_start();
    let line = match line.trim_start_matches('#') {
        rest if rest.len() < line.len() && rest.starts_with(' ') => rest,
        _ => line,
    };
    let line = ["- ", "* ", "+ ", "• ", "> "].iter().find_map(|marker| line.strip_prefix(marker)).unwrap_or(line);
    // An asterisk is emphasis when it touches a word on one side only, as markdown's flanking rule
    // has it; with a word on both sides (2*3) or space on both (2 * 3) it's maths and stays.
    let characters: Vec<char> = line.chars().collect();
    let word = |at: Option<&char>| at.is_some_and(|character| character.is_alphanumeric());
    let space = |at: Option<&char>| at.is_none_or(|character| character.is_whitespace());
    let mark = |index: usize| {
        let (before, after) = (index.checked_sub(1).and_then(|before| characters.get(before)), characters.get(index + 1));
        !(word(before) && word(after)) && !(space(before) && space(after))
    };
    characters.iter().enumerate().filter(|&(index, &character)| character != '`' && !(character == '*' && mark(index))).map(|(_, character)| character).collect::<String>().trim().to_string()
}

// Ask the quick model for its answer to the turn's words, with the conversation's end, and give the
// speaker its text as it streams (or all at once, from a provider that didn't stream it). Uncached,
// with no saved session (a fresh routing id for HTTP), what it cost goes to SPENT, for the
// session's statistics. A model that fails says nothing.
async fn quick_answer(quick: &Quick, turn: &Turn, speaker: &mut Speaker, first: &str, overtaken: &CancellationToken) -> Option<()> {
    if overtaken.is_cancelled() { return None; }
    // Lists arrive in the background. A failed lookup at session creation mustn't disable every
    // later turn; read the current list here, waiting for startup only if this model is missing.
    let path = crate::config::agent_dir().join("models.json");
    let resolve = || {
        let registry = ModelRegistry::load(&path)?;
        let model = registry.resolve(&quick.name)?.clone();
        let config = registry.provider_config(&model.provider).cloned();
        Ok::<_, String>((model, config))
    };
    let mut resolved = resolve();
    if resolved.is_err() && let Some(listing) = crate::model::listing(&crate::config::agent_dir()) {
        listing.await;
        resolved = resolve();
    }
    let (model, config) = resolved.map_err(|error| note(&format!("voice.quick {} unavailable: {error}", quick.name))).ok()?;
    note(&format!("voice.quick trying {} (session {})", quick.name, turn.session_id));
    let credential = crate::auth::credential_for(&model.provider, config.as_ref(), &quick.credentials).await.map_err(|error| note(&format!("voice.quick: {error}"))).ok()?;
    let messages = [
        Message::System(SystemMessage { content: TextOrBlocks::Text(if turn.reply.is_some() { LAYERED_PROMPT } else { QUICK_PROMPT }.into()), sections: None, tools_added: None, tools_removed: None, timestamp: 0 ,
        }),
        Message::User(UserMessage { content: TextOrBlocks::Text(format!("The conversation so far, its end:\n\n{}The user just said: {}{}", turn.transcript, turn.words,
            if turn.reply.is_some() { format!("\n\nThe first voice already said: {first}") } else { String::new() })),  timestamp: crate::clock::now_ms() ,
        }),
    ];
    let session_id = uuid::Uuid::now_v7().to_string();
    let (sender, mut events) = mpsc::unbounded_channel();
    let cancel = overtaken.child_token();
    let _cancel = cancel.clone().drop_guard();
    let request = Request {
        model: &model,
        messages: &messages,
        credential: &credential,
        session_id: (model.provider != crate::provider::claude::PROVIDER).then_some(session_id.as_str()),
        cache: false,
        transport: &quick.transport,
        thinking_level: crate::provider::clamp_thinking_level(&model, "off"),
        session_file: None,
        cwd: &quick.cwd,
        cancel: cancel.clone(),
    };
    let mut streamed = false;
    let read = async {
        while let Some(event) = events.recv().await {
            if let StreamEvent::TextDelta { delta, .. } = event {
                if !streamed {
                    note(&format!("voice.quick first text in {:.3} s (session {})", turn.started.elapsed().as_secs_f64(), turn.session_id));
                }
                streamed = true;
                if !overtaken.is_cancelled() { speaker.add(&delta, false); }
                if speaker.sent == speaker.limit { cancel.cancel(); }
            }
        }
    };
    let (reply, ()) = tokio::join!(crate::provider::stream(request, sender), read);
    if !streamed {
        let text = TextOrBlocks::Blocks(reply.content.clone()).text();
        if text.trim().is_empty() {
            note("voice.quick returned no speech");
        }
        if !overtaken.is_cancelled() { speaker.add(&text, true); }
    }
    if let Some(error) = &reply.error_message {
        note(&format!("voice.quick: {error}"));
    }
    (reply.usage.input + reply.usage.output > 0).then(|| {
        SPENT.send_modify(|spent| {
            spent.push((turn.session_id.clone(), reply.provider, reply.model, reply.usage,))
}
)
    })
}
