// Voice is brought by the user: shell commands record, transcribe and play audio. A call names
// one session in the agent directory; push-to-talk and every player belong to that call's id.
// Replacing it or hanging up stops old audio, but releasing a frontend leaves the call intact.
//
// voice.listen reads until stdin closes and writes one UTF-8 transcript to stdout. Only its final
// protocol newline is removed: internal newlines and spaces are the user's words, not ours to edit.
// With voice.reply set, listen also writes a mono WAV to $PICCOSTA_VOICE_AUDIO. reply receives that
// WAV on stdin and may write its generated text to stdout; it owns its own playback.
// voice.quick names any configured provider/model. It gets the recent conversation and the exact
// transcript, with a bounded head start. Its result travels WITH the user message through queues,
// then becomes an attributed append before main runs. No prompt or cached prefix is rewritten.
// voice.speak takes UTF-8 lines on stdin, once for quick and once for main, in that order. Exiting
// successfully is the BYO command's playback-completed acknowledgment, not proof the user heard it.
// voice.call optionally runs for the call's lifetime (a shared speech server, for example).

use crate::agent::{Event, EventSink};
use crate::message::{AgentMessage, AssistantMessage, CustomMessage, InputOrigin, Message, SystemMessage, TextOrBlocks, UserMessage};
use crate::provider::Request;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

const HOLD_FILE: &str = "voice.pid";
const LOCK_FILE: &str = "voice.lock";
const CALL_FILE: &str = "call.json";
const CALL_LOCK_FILE: &str = "voice.call.lock";
const LOG_FILE: &str = "voice.log";
// Two minutes bounds a missed key release; batch recognizers get thirty seconds after release.
const MAX_HOLD_SECONDS: u64 = 120;
const LISTEN_FINISH: Duration = Duration::from_secs(30);
// Quick is a head start, not a gate that can stall the conversation. Five seconds includes auth
// and generation; a failed or slow quick model leaves main to answer the unchanged transcript.
const QUICK_WAIT: Duration = Duration::from_secs(5);
// About 6k tokens of recent conversation, without tools or their potentially enormous output.
const QUICK_CONTEXT_CHARS: usize = 24_000;
// Half a second bounds hangup detection without watching a file continuously.
const CALL_POLL: Duration = Duration::from_millis(500);
// Reap within an audio frame (LFM emits 80 ms frames) without reusing the group id too early.
const REAP_POLL: Duration = Duration::from_millis(20);
const CALL_EXIT_GRACE: Duration = Duration::from_secs(1);
// A player that won't accept a line is stuck. Long answers still have five minutes to play.
const SPEAK_WRITE: Duration = Duration::from_secs(10);
const SPEAK_EXIT: Duration = Duration::from_secs(300);
const STOP_WAIT: Duration = Duration::from_secs(5);
// A few waiting turns and their complete answers never backpressure the model. Queue whole
// answers, not deltas: providers without text deltas work too, and overflow cannot garble text.
const TURN_QUEUE: usize = 4;
const QUICK_KIND: &str = "voice-quick";
pub const SILENT: &str = "[PICCOSTA_SILENT]";

// A provider can split one marker across text blocks. Join only for this decision; normal speech
// keeps its block separators, and the journal always keeps the provider's original blocks.
pub fn is_silent(message: &AssistantMessage) -> bool {
    let text: String = message.content.iter().filter_map(|block| if let crate::message::Content::Text { text, .. } = block { Some(text.as_str()) } else { None }).collect();
    text.trim() == SILENT
}
// Notices can arrive between a voice input and its answer; only another ordinary user input
// changes the rendering convention. Historical voice replies keep it after hangup or reopen.
pub fn is_voice_turn(messages: &[AgentMessage]) -> bool {
    messages.iter().rev().find_map(|message| match message { AgentMessage::User(user) if !user.origin.is_notice() => Some(user.voice.is_some()), _ => None }).unwrap_or(false)
}

const QUICK_PROMPT: &str = "You are the quick voice in a conversation. The main assistant will see your exact reply and answer after you. From the recent conversation, answer the user's latest words in one or two short spoken sentences. Answer what you know, but never claim to have checked or done something. No markdown or filler. Prior main-assistant and quick-voice replies are attributed separately below.";
const MAIN_PROMPT: &str = "This is a voice call. Quick's generated reply above is attributed to the quick voice, not the user. Your audio follows its playback. Add only information it did not cover, in brief plain spoken sentences; do not repeat or paraphrase its answer. Begin directly with new information, not a recap or acknowledgment. Generated is not proof of completed playback: a hangup can cut speech short. Do not read harness notes or machine statistics aloud unless asked. If nothing is missing, write exactly [PICCOSTA_SILENT] and nothing else. That marker means silence, not speech.";

// A single queued/steering input keeps these together. The session retains the receipt with the
// user entry, but projection removes it: main gets the exact user body and a separate notice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Delivery {
    pub id: String,
    pub call: String,
    pub quick: Option<AssistantMessage>,
    pub playback: bool,
    pub recording: Option<PathBuf>,
}

fn shell(command: &str) -> Command {
    let mut shell = Command::new("sh");
    shell.arg("-c").arg(command).process_group(0).kill_on_drop(true);
    shell
}

// Keep the group number while its leader is unreaped, including if an async future is cancelled.
struct Group(Option<u32>);
impl Group {
    fn kill(&mut self) {
        if let Some(group) = self.0.take() {
            unsafe { libc::killpg(group as i32, libc::SIGKILL) };
        }
    }
}
impl Drop for Group { fn drop(&mut self) { self.kill(); } }

// Observe exit without reaping the leader. Its pid still reserves the group number until all
// descendants have been signalled, so a late cleanup can never kill a reused process group.
async fn wait_group(child: &mut tokio::process::Child, group: &mut Group) -> std::io::Result<std::process::ExitStatus> {
    let pid = child.id().ok_or_else(|| std::io::Error::other("voice child already reaped"))?;
    loop {
        let mut status = unsafe { std::mem::zeroed::<libc::siginfo_t>() };
        if unsafe { libc::waitid(libc::P_PID, pid, &mut status, libc::WEXITED | libc::WNOHANG | libc::WNOWAIT) } != 0 { return Err(std::io::Error::last_os_error()); }
        if unsafe { status.si_pid() } != 0 { break; }
        tokio::time::sleep(REAP_POLL).await;
    }
    group.kill();
    child.wait().await
}

async fn kill_group(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() { unsafe { libc::killpg(pid as i32, libc::SIGKILL) }; }
    let _ = child.wait().await;
}

fn log() -> std::io::Result<std::fs::File> { std::fs::OpenOptions::new().create(true).append(true).open(crate::config::agent_dir().join(LOG_FILE)) }

pub(crate) fn note(text: &str) { if let Ok(mut file) = log() { let _ = writeln!(file, "piccosta voice: {text}"); } }

fn text_setting(settings: &crate::config::Settings, key: &str) -> Option<String> {
    settings.get(key).and_then(Value::as_str).filter(|text| !text.trim().is_empty()).map(str::to_string)
}

fn write_record(path: &Path, record: Value) -> std::io::Result<()> {
    let written = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&written, record.to_string())?;
    std::fs::rename(written, path)
}

fn live_record(path: &Path) -> Option<u32> {
    let record: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let pid = u32::try_from(record["pid"].as_u64()?).ok()?;
    let started = crate::proc::recorded_start(record["started"].as_str()?)?;
    crate::proc::Process::started(pid as i32, started).map(|_| pid)
}

// Failed reads are not hangups. Records are renamed whole, so malformed JSON is not a call.
fn reading(agent_dir: &Path) -> std::io::Result<Option<(PathBuf, String)>> {
    let text = match std::fs::read_to_string(agent_dir.join(CALL_FILE)) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        read => read?,
    };
    let record: Value = serde_json::from_str(&text).unwrap_or_default();
    Ok(record["session"].as_str().zip(record["id"].as_str()).map(|(file, id)| (PathBuf::from(file), id.to_string())))
}

fn over(agent_dir: &Path, id: &str) -> bool { reading(agent_dir).is_ok_and(|call| call.is_none_or(|(_, now)| now != id)) }
async fn ended(agent_dir: &Path, id: &str) { while !over(agent_dir, id) { tokio::time::sleep(CALL_POLL).await; } }

fn open_lock(agent_dir: &Path, name: &str) -> Result<std::fs::File, String> {
    std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(agent_dir.join(name)).map_err(|error| error.to_string())
}

fn lock(agent_dir: &Path) -> Result<std::fs::File, String> {
    let file = open_lock(agent_dir, LOCK_FILE)?;
    file.lock().map_err(|error| error.to_string())?;
    Ok(file)
}

fn session_settings(agent_dir: &Path, file: &Path) -> Result<(crate::session::Session, crate::config::Settings), String> {
    let session = crate::session::Session::read(file).map_err(|error| error.to_string())?;
    let settings = crate::config::Settings::load(agent_dir, Some(Path::new(&session.header().cwd)));
    Ok((session, settings))
}

// Voice is unattended ingress, even when a person holds the button. Refuse incompatible models
// and absent owners before recording or spending quick's tokens, rather than promising a call.
fn callable(agent_dir: &Path, file: &Path) -> Result<crate::config::Settings, String> {
    let (session, settings) = session_settings(agent_dir, file)?;
    let model = session.state().0.map(|(provider, id)| format!("{provider}/{id}")).or_else(|| settings.default_model()).ok_or("voice session has no model")?;
    let registry = crate::model::ModelRegistry::load(&agent_dir.join("models.json"))?;
    if registry.resolve(&model)?.interactive_only { return Err("voice calls cannot use an interactiveOnly main model; choose another model".into()); }
    if !crate::runtime::running(file) { return Err("voice session is not running; open it before calling or speaking".into()); }
    Ok(settings)
}

fn remove_call(record: &Path) -> std::io::Result<()> { std::fs::remove_file(record).or_else(|error| if error.kind() == std::io::ErrorKind::NotFound { Ok(()) } else { Err(error) }) }

pub fn set_call(agent_dir: &Path, file: Option<&Path>) -> std::io::Result<()> {
    // A fresh TUI session has an owner but no journal yet. Only that owner may materialize its
    // header; an outside writer could otherwise overwrite the metadata it still holds in memory.
    if let Some(file) = file.filter(|file| !file.exists()) {
        let answer = crate::runtime::ask(file, json!({"do": "voice-call"}), crate::runtime::ASK_WAIT);
        return match answer {
            crate::runtime::Asked::Answered(answer) if answer["ok"] == true => Ok(()),
            _ => Err(std::io::Error::other("could not start the call: open this session first")),
        };
    }
    if let Some(file) = file { callable(agent_dir, file).map_err(std::io::Error::other)?; }
    let _lock = lock(agent_dir).map_err(std::io::Error::other)?;
    let record = agent_dir.join(CALL_FILE);
    let Some(file) = file else { return remove_call(&record) };
    let id = uuid::Uuid::now_v7().to_string();
    write_record(&record, json!({"session": file, "id": id}))?;
    let mut supervisor = std::process::Command::new(std::env::current_exe()?);
    supervisor.arg("voice").arg("run-call").arg(id).env("PICCOSTA_AGENT_DIR", agent_dir).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log()?);
    match std::os::unix::process::CommandExt::process_group(&mut supervisor, 0).spawn() {
        Ok(mut child) => {
            std::thread::spawn(move || child.wait());
            Ok(())
        }
        Err(error) => {
            let _ = remove_call(&record);
            Err(error)
        }
    }
}

// Compare and remove under one lock, so closing an old session cannot hang up a newer call.
pub fn hang_up(agent_dir: &Path, file: &Path) {
    let Ok(_lock) = lock(agent_dir) else { return };
    if reading(agent_dir).ok().flatten().is_some_and(|(called, _)| called == file) {
        let _ = remove_call(&agent_dir.join(CALL_FILE));
    }
}

async fn run_call(agent_dir: &Path, id: &str) -> Result<(), String> {
    let command_lock = open_lock(agent_dir, CALL_LOCK_FILE)?;
    // The old supervisor cleans up sockets before its replacement starts a server.
    command_lock.lock().map_err(|e| e.to_string())?;
    if over(agent_dir, id) {
        return Ok(());
    }
    let _recordings = Recording(agent_dir.join("voice").join(uuid::Uuid::parse_str(id).map_err(|e| e.to_string())?.to_string()));
    let Some((target, current)) = reading(agent_dir).map_err(|error| error.to_string())? else { return Ok(()) };
    if current != id {
        return Ok(());
    }
    let (_, settings) = session_settings(agent_dir, &target)?;
    use tokio::signal::unix::{SignalKind, signal};
    let caught = |kind| signal(kind).map_err(|error| error.to_string());
    let (mut term, mut hangup, mut interrupt) = (caught(SignalKind::terminate())?, caught(SignalKind::hangup())?, caught(SignalKind::interrupt())?);
    let mut child = text_setting(&settings, "voice.call").map(|command| shell(&command).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log().map_err(|e| e.to_string())?).spawn().map_err(|e| e.to_string())).transpose()?;
    tokio::select! { () = ended(agent_dir, id) => {}, _ = term.recv() => {}, _ = hangup.recv() => {}, _ = interrupt.recv() => {} }
    if let Some(child) = &mut child {
        if let Some(pid) = child.id() {
            unsafe { libc::killpg(pid as i32, libc::SIGTERM) };
            tokio::time::sleep(CALL_EXIT_GRACE).await;
        }
        kill_group(child).await;
    }
    Ok(())
}

fn session_named(agent_dir: &Path, wanted: Option<&str>) -> Result<PathBuf, String> {
    match wanted {
        None | Some("assistant") => crate::manager::session_file(agent_dir).ok_or_else(|| "no assistant yet: start it with `piccosta assistant`".into()),
        Some(path) if path.contains('/') || path.ends_with(".jsonl") => std::path::absolute(path).map_err(|error| error.to_string()),
        Some(wanted) => {
            crate::host::all_sessions(agent_dir, usize::MAX).into_iter().find(|summary| summary.id.starts_with(wanted) || summary.title == wanted).map(|summary| summary.path).ok_or_else(|| format!("No session matches \"{wanted}\""))
        }
    }
}

pub async fn run(arguments: &[String]) -> Result<(), String> {
    let agent_dir = crate::config::agent_dir();
    let hold_file = agent_dir.join(HOLD_FILE);
    match arguments.first().map(String::as_str) {
        Some("call") => set_call(&agent_dir, Some(&session_named(&agent_dir, arguments.get(1).map(String::as_str))?)).map_err(|e| e.to_string()),
        Some("hangup") => set_call(&agent_dir, None).map_err(|e| e.to_string()),
        Some("run-call") => run_call(&agent_dir, arguments.get(1).ok_or("voice run-call needs an id")?).await,
        Some("start") => {
            let Some((target, id)) = reading(&agent_dir).map_err(|e| e.to_string())? else {
                return Ok(());
            };
            let settings = callable(&agent_dir, &target)?;
            text_setting(&settings, "voice.listen").ok_or("no listen command: set voice.listen in settings")?;
            let _lock = lock(&agent_dir)?;
            stop(&hold_file);
            let mut hold = std::process::Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
            hold.arg("voice").arg("hold").arg(id).arg(target).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log().map_err(|e| e.to_string())?);
            // A release can arrive before exec finishes. Keep SIGTERM pending until hold() has
            // installed its receiver, rather than losing a very short push-to-talk recording.
            unsafe {
                std::os::unix::process::CommandExt::pre_exec(&mut hold, || mask_release(libc::SIG_BLOCK));
            }
            let hold = std::os::unix::process::CommandExt::process_group(&mut hold, 0).spawn().map_err(|e| e.to_string())?;
            let pid = hold.id();
            write_record(&hold_file, json!({"pid": pid, "started": crate::proc::start(pid as i32).map(|start| format!("{} {start}", crate::proc::boot_id()))})).map_err(|e| e.to_string())
        }
        Some("stop") => {
            let _lock = lock(&agent_dir)?;
            stop(&hold_file);
            Ok(())
        }
        Some("hold") => {
            let (Some(id), Some(target)) = (arguments.get(1), arguments.get(2)) else {
                return Err("voice hold needs a call id and session file".into());
            };
            let result = hold(&agent_dir, id, Path::new(target)).await;
            // close() lets go asynchronously. Finish reaping and deleting the one-shot's history
            // before this short-lived process drops its runtime, even when no audio kept it alive.
            crate::provider::shutdown().await;
            let _lock = lock(&agent_dir)?;
            if live_record(&hold_file) == Some(std::process::id()) {
                let _ = std::fs::remove_file(hold_file);
            }
            result
        }
        _ => Err("usage: piccosta call [session] | hangup | voice start | voice stop".into()),
    }
}

// Only async-signal-safe calls: this also runs between fork and exec for the listening child.
fn mask_release(how: i32) -> std::io::Result<()> {
    let mut signals = unsafe { std::mem::zeroed::<libc::sigset_t>() };
    unsafe {
        libc::sigemptyset(&mut signals);
        libc::sigaddset(&mut signals, libc::SIGTERM);
    }
    if unsafe { libc::sigprocmask(how, &signals, std::ptr::null_mut()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn stop(hold_file: &Path) {
    if let Some(pid) = live_record(hold_file) {
        unsafe { libc::kill(pid as i32, libc::SIGTERM) };
    }
    let _ = std::fs::remove_file(hold_file);
}

struct Recording(PathBuf);
impl Recording {
    fn new(agent_dir: &Path, call: &str) -> Result<Self, String> {
        let dir = agent_dir.join("voice").join(call).join(uuid::Uuid::now_v7().to_string());
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

async fn hold(agent_dir: &Path, id: &str, target: &Path) -> Result<(), String> {
    let settings = callable(agent_dir, target)?;
    let listen = text_setting(&settings, "voice.listen").ok_or("no listen command: set voice.listen in settings")?;
    let recording = Recording::new(agent_dir, id)?;
    let mut command = shell(&listen);
    command.env("PICCOSTA_VOICE_AUDIO", recording.0.join("input.wav"));
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).map_err(|e| e.to_string())?;
    mask_release(libc::SIG_UNBLOCK).map_err(|e| e.to_string())?;
    let mut listener = command.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
    let mut group = Group(listener.id());
    let stdin = listener.stdin.take();
    let mut output = listener.stdout.take().ok_or("voice.listen has no stdout")?;
    let mut words = String::new();
    {
        let read = output.read_to_string(&mut words);
        tokio::pin!(read);
        let limit = Duration::from_secs(settings.get("voice.maxSeconds").and_then(Value::as_u64).unwrap_or(MAX_HOLD_SECONDS));
        let complete = tokio::select! {
            result = &mut read => { result.map_err(|e| e.to_string())?; true }
            _ = terminate.recv() => false,
            () = tokio::time::sleep(limit) => false,
            () = ended(agent_dir, id) => { group.kill(); let _ = listener.wait().await; return Ok(()) }
        };
        drop(stdin);
        if !complete {
            tokio::select! {
                result = tokio::time::timeout(LISTEN_FINISH, &mut read) => match result {
                    Ok(result) => { result.map_err(|e| e.to_string())?; }
                    Err(_) => { group.kill(); let _ = listener.wait().await; return Err("voice.listen did not finish after release".into()) }
                },
                () = ended(agent_dir, id) => { group.kill(); let _ = listener.wait().await; return Ok(()) }
            }
        }
    }
    let status = tokio::time::timeout(LISTEN_FINISH, wait_group(&mut listener, &mut group)).await.map_err(|_| "voice.listen did not exit")?.map_err(|e| e.to_string())?;
    drop(group);
    if !status.success() {
        return Err(format!("voice.listen exited {status}"));
    }
    // One final line terminator is framing, not speech. Everything else is untouched.
    if words.ends_with('\n') {
        words.pop();
        if words.ends_with('\r') {
            words.pop();
        }
    }
    if words.is_empty() || over(agent_dir, id) {
        return Ok(());
    }
    let settings = callable(agent_dir, target)?;
    let (session, pending) = crate::runtime::voice_context(target)?;
    let (quick, query) = generate(&settings, &session, &pending, &words, id).await;
    let mut receipt = Delivery { id: uuid::Uuid::now_v7().to_string(), call: id.into(), quick, playback: false, recording: Some(recording.0.clone()) };
    let mut submitted = false;
    let deliver = async {
        if over(agent_dir, id) {
            return Ok(());
        }
        callable(agent_dir, target)?;
        let quick_text = receipt.quick.as_ref().filter(|answer| answer.error_message.is_none()).map(|answer| TextOrBlocks::Blocks(answer.content.clone()).text()).unwrap_or_default();
        let speak = text_setting(&settings, "voice.speak").filter(|_| !quick_text.trim().is_empty());
        receipt.playback = speak.is_some();
        let mut message = UserMessage::new(&words, Vec::new(), InputOrigin::frontend("voice"));
        message.voice = Some(Box::new(receipt.clone()));
        // Queue text immediately; only the main player's audio waits for quick's receipt.
        let sent = crate::runtime::send_voice(target, &message);
        if !matches!(sent, Ok(crate::runtime::Sent::Delivered)) {
            let _ = write_record(&recording.0.join("quick-playback.json"), json!({"completed": false}));
            return Err("voice delivery was not confirmed; the transcript may be queued, open the session before speaking again".into());
        }
        submitted = true;
        if let Some(command) = speak {
            let play = async {
                let _audio = playback_lock(agent_dir).await?;
                Ok::<_, String>(player(&command, &quick_text, "quick").await)
            };
            let completed = tokio::select! {
                result = play => result.unwrap_or_else(|error| { note(&error); false }),
                () = ended(agent_dir, id) => false,
            };
            let _ = write_record(&recording.0.join("quick-playback.json"), json!({"completed": completed}));
        }
        Ok(())
    };
    // Cancellation drains accounting alongside playback, never extending main's head start.
    let cleanup = async {
        match query {
            Some(query) => query.finish().await,
            None => None,
        }
    };
    let (result, cancelled) = tokio::join!(deliver, cleanup);
    // The receiver or the call supervisor owns cleanup after any attempted durable delivery,
    // so let go of the recording before anything below can return early.
    if submitted || result.is_err() {
        std::mem::forget(recording);
    }
    if let Some(answer) = cancelled.or_else(|| (!submitted).then(|| receipt.quick.clone()).flatten()) {
        receipt.quick = Some(answer);
        crate::runtime::voice_usage(target, &receipt)?;
    }
    result
}

// A fresh provider identity per attempt, no tools and no session file. Always close the provider
// query after success, failure or timeout; Claude's ordinary saved-session ownership is unchanged.
type PendingAnswer = std::pin::Pin<Box<dyn std::future::Future<Output = Result<AssistantMessage, String>>>>;

struct Query {
    id: String,
    pending: Option<PendingAnswer>,
    sent: Arc<AtomicBool>,
    unknown: AssistantMessage,
}

impl Query {
    async fn finish(self) -> Option<AssistantMessage> {
        let answer = if let Some(pending) = self.pending {
            match tokio::time::timeout(STOP_WAIT, pending).await {
                Ok(Ok(mut answer)) => {
                    let usage = &mut answer.usage;
                    if [usage.input, usage.output, usage.cache_read, usage.cache_write, usage.total_tokens].iter().all(|count| *count == 0) && usage.reported.is_none() {
                        usage.tokens_unknown = true;
                        usage.cost.state = crate::message::CostState::Unknown;
                    }
                    Some(answer)
                }
                Ok(Err(error)) => { note(&format!("voice.quick: {error}")); None }
                Err(_) => self.sent.load(Ordering::Acquire).then_some(self.unknown),
            }
        } else {
            None
        };
        if let Err(error) = crate::provider::close(&self.id).await {
            note(&format!("voice.quick cleanup: {error}"));
        }
        answer
    }
}

async fn generate(settings: &crate::config::Settings, session: &crate::session::Session, pending: &[UserMessage], words: &str, call: &str) -> (Option<AssistantMessage>, Option<Query>) {
    let Some(name) = text_setting(settings, "voice.quick") else { return (None, None) };
    let resolve = || {
        let registry = crate::model::ModelRegistry::load(&crate::config::agent_dir().join("models.json"))?;
        let model = registry.resolve(&name)?.clone();
        if model.interactive_only {
            return Err("voice.quick cannot run an interactive-only provider".into());
        }
        let config = registry.provider_config(&model.provider).cloned();
        Ok::<_, String>((model, config))
    };
    let (model, config) = match resolve() {
        Ok(found) => found,
        Err(error) => {
            note(&error);
            return (None, None);
        }
    };
    let mut unknown = crate::provider::new_output(&model, &model.api);
    unknown.usage.tokens_unknown = true;
    unknown.usage.cost.state = crate::message::CostState::Unknown;
    let mut transcript = String::new();
    for message in session.context().messages.iter().cloned().chain(pending.iter().cloned().map(AgentMessage::User)) {
        remember(&mut transcript, &message);
    }
    let messages = [
        Message::System(SystemMessage { content: TextOrBlocks::Text(QUICK_PROMPT.into()), sections: None, tools_added: None, tools_removed: None, timestamp: 0 }),
        Message::User(UserMessage::new(&format!("Recent conversation:\n{transcript}\nThe user just said:\n{words}"), Vec::new(), InputOrigin::Unknown)),
    ];
    let id = uuid::Uuid::now_v7().to_string();
    let cancel = CancellationToken::new();
    let (key, cancellation, sent) = (id.clone(), cancel.clone(), Arc::new(AtomicBool::new(false)));
    let started = sent.clone();
    let (cwd, transport) = (PathBuf::from(&session.header().cwd), settings.transport());
    let mut pending: PendingAnswer = Box::pin(async move {
        let paths = crate::auth::CredentialPaths::from_environment(&crate::config::agent_dir());
        let credential = crate::auth::credential_for(&model.provider, config.as_ref(), &paths).await?;
        let request = Request {
            model: &model,
            provider_config: config.as_ref(),
            interaction: None,
            interactive: false,
            elicitation: false,
            yolo: false,
            messages: &messages,
            credential: &credential,
            session_id: Some(&key),
            in_place: false,
            manager: false,
            transport: &transport,
            thinking_level: crate::provider::clamp_thinking_level(&model, "off"),
            session_file: None,
            fresh_conversation: true,
            ownership: None,
            cwd: &cwd,
            cancel: cancellation,
        };
        let (sender, mut events) = mpsc::unbounded_channel();
        started.store(true, Ordering::Release);
        let (answer, ()) = tokio::join!(crate::provider::stream(request, sender), async { while events.recv().await.is_some() {} });
        Ok(answer)
    });
    let agent_dir = crate::config::agent_dir();
    let mut finished = false;
    let answer = tokio::select! {
        result = &mut pending => { finished = true; match result { Ok(answer) => Some(answer), Err(error) => { note(&format!("voice.quick: {error}")); None } } },
        () = tokio::time::sleep(QUICK_WAIT) => { note("voice.quick timed out; main proceeds without it"); None },
        () = ended(&agent_dir, call) => None,
    };
    cancel.cancel();
    let pending = if finished { None } else { Some(pending) };
    (answer, Some(Query { id, pending, sent, unknown }))
}

fn remember(transcript: &mut String, message: &AgentMessage) {
    let (speaker, text) = match message {
        AgentMessage::User(user) => {
            let mut text = user.content.text();
            if let Some(answer) = user.voice.as_ref().and_then(|delivery| delivery.quick.as_ref()).filter(|answer| answer.error_message.is_none()) {
                text.push_str(&format!("\n\nQuick voice: {}", TextOrBlocks::Blocks(answer.content.clone()).text()));
            }
            (if user.origin.is_notice() { user.origin.label() } else { "User" }, text)
        }
        AgentMessage::Assistant(reply) => ("Main assistant", TextOrBlocks::Blocks(reply.content.clone()).text()),
        AgentMessage::Custom(notice) if notice.origin.is_notice() && notice.custom_type != QUICK_KIND => (notice.origin.label(), notice.content.text()),
        _ => return,
    };
    if text.trim().is_empty() || text.trim() == SILENT {
        return;
    }
    transcript.push_str(&format!("{speaker}: {text}\n\n"));
    if transcript.len() > QUICK_CONTEXT_CHARS {
        let cut = (transcript.len() - QUICK_CONTEXT_CHARS..).find(|&index| transcript.is_char_boundary(index)).unwrap_or(0);
        transcript.drain(..cut);
    }
}

// A receipt keeps its cost through requeue and shutdown too. Only a successful append counts
// as recorded; a later take retries without losing or double-counting the pending usage.
pub(crate) fn record_usage(session: &mut crate::session::Session, delivery: &Delivery) -> std::io::Result<Option<crate::session::SessionEntry>> {
    use crate::session::SessionEntry;
    let Some(answer) = &delivery.quick else { return Ok(None) };
    let note = format!("voice:{}", delivery.id);
    if session.entries().iter().any(|entry| matches!(entry, SessionEntry::Usage { note: Some(saved), .. } if saved == &note)) {
        return Ok(None);
    }
    let entry = SessionEntry::Usage { base: session.next_base(), kind: QUICK_KIND.into(), provider: answer.provider.clone(), model: answer.model.clone(), usage: answer.usage.clone(), note: Some(note) };
    session.append(entry.clone())?;
    Ok(Some(entry))
}

// Persisted before main's first request, even for quick-only configurations and no-tool answers.
pub(crate) fn notice(delivery: &Delivery) -> CustomMessage {
    let generated = delivery.quick.as_ref().filter(|answer| answer.error_message.is_none()).map(|answer| TextOrBlocks::Blocks(answer.content.clone()).text()).unwrap_or_default();
    let content = if generated.trim().is_empty() {
        "This is a voice call. No quick reply was generated. You are the only responding voice: answer the user's message normally, in brief plain spoken sentences. Do not read harness notes or machine statistics aloud unless asked.".into()
    } else {
        format!("Quick voice generated (not yet confirmed played):\n{generated}\n\n{MAIN_PROMPT}")
    };
    CustomMessage {
        custom_type: QUICK_KIND.into(),
        origin: InputOrigin::Harness { name: QUICK_KIND.into() },
        content: TextOrBlocks::Text(content),
        display: true,
        details: Some(json!({"id": delivery.id, "call": delivery.call, "generated": generated, "playback": delivery.playback, "recording": delivery.recording})),
        timestamp: crate::clock::now_ms(),
    }
}

struct Turn {
    call: String,
    closed: CancellationToken,
    speak: Option<String>,
    reply: Option<String>,
    recording: Option<Recording>,
    quick_pending: bool,
    answer: mpsc::Receiver<String>,
}
enum Job {
    Turn(Turn),
    Drain(oneshot::Sender<()>),
}
struct Worker {
    jobs: mpsc::Sender<Job>,
    stop: CancellationToken,
    thread: Mutex<Option<(std::thread::JoinHandle<()>, oneshot::Receiver<()>)>>,
}
static WORKER: OnceLock<Worker> = OnceLock::new();

fn worker() -> &'static Worker {
    WORKER.get_or_init(|| {
        let (jobs, mut queue) = mpsc::channel(TURN_QUEUE);
        let (stop, (ended, thread_ended)) = (CancellationToken::new(), oneshot::channel());
        let stopping = stop.clone();
        let thread = std::thread::spawn(move || {
            let _ended = ended;
            let Ok(runtime) = tokio::runtime::Builder::new_current_thread().enable_all().build() else { return };
            runtime.block_on(async {
                loop {
                    match tokio::select! { job = queue.recv() => job, () = stopping.cancelled() => None } {
                        Some(Job::Turn(turn)) => play(turn, &stopping).await,
                        Some(Job::Drain(done)) => {
                            let _ = done.send(());
                        }
                        None => break,
                    }
                }
            });
        });
        Worker { jobs, stop, thread: Mutex::new(Some((thread, thread_ended))) }
    })
}

// Drain only this process's speech before releasing a print runtime's session ownership. A UI
// exit cuts local playback but leaves the call record, so its replacement can carry on.
pub async fn shutdown(drain: bool) {
    let Some(worker) = WORKER.get() else { return };
    if drain {
        let (done, said) = oneshot::channel();
        let finish = async {
            if worker.jobs.send(Job::Drain(done)).await.is_ok() {
                let _ = said.await;
            }
        };
        if tokio::time::timeout(SPEAK_EXIT, finish).await.is_err() {
            note("speech drain timed out");
        }
    }
    worker.stop.cancel();
    let Some((thread, ended)) = worker.thread.lock().unwrap_or_else(|p| p.into_inner()).take() else {
        return;
    };
    if tokio::time::timeout(STOP_WAIT, ended).await.is_ok() {
        let _ = thread.join();
    } else {
        note("speech worker did not stop");
    }
}

pub fn speaking(settings: &crate::config::Settings, session: &crate::session::Session, closed: CancellationToken, mut emit: EventSink) -> EventSink {
    let (speak, reply) = (text_setting(settings, "voice.speak"), text_setting(settings, "voice.reply"));
    let (file, agent_dir) = (session.file().map(Path::to_path_buf), crate::config::agent_dir());
    let mut voiced = is_voice_turn(&session.context().messages);
    let mut turn: Option<mpsc::Sender<String>> = None;
    Box::new(move |event| {
        match event {
            Event::MessageStart { message } => {
                let voice = match message {
                    AgentMessage::User(user) => {
                        if !user.origin.is_notice() {
                            voiced = user.voice.is_some();
                        }
                        user.voice.as_deref()
                    }
                    _ => None,
                };
                if matches!(message, AgentMessage::User(_)) || turn.is_none() && matches!(message, AgentMessage::Custom(note) if note.custom_type != QUICK_KIND) {
                    turn = None;
                    let recording = voice
                        .and_then(|delivery| delivery.recording.clone())
                        .filter(|dir| {
                            dir.parent().and_then(Path::parent) == Some(agent_dir.join("voice").as_path())
                                && [dir.file_name(), dir.parent().and_then(Path::file_name)].into_iter().all(|name| name.and_then(|name| name.to_str()).is_some_and(|name| uuid::Uuid::parse_str(name).is_ok()))
                        })
                        .map(Recording);
                    let on_call = reading(&agent_dir).ok().flatten().filter(|(called, id)| Some(called) == file.as_ref() && voice.is_none_or(|delivery| delivery.call == *id));
                    if let Some((_, call)) = on_call.filter(|_| speak.is_some() || reply.is_some()) {
                        let (sender, answer) = mpsc::channel(TURN_QUEUE);
                        let job = Turn { call, closed: closed.clone(), speak: speak.clone(), reply: reply.clone(), recording, quick_pending: voice.is_some_and(|delivery| delivery.playback), answer };
                        turn = worker().jobs.try_send(Job::Turn(job)).map(|()| sender).inspect_err(|_| note("speech queue full; this turn is not played")).ok();
                    }
                }
            }
            Event::MessageEnd { message: AgentMessage::Assistant(reply) } if !(matches!(reply.stop_reason, crate::message::StopReason::Error | crate::message::StopReason::Aborted) || voiced && is_silent(reply)) => {
                if let Some(turn) = &turn {
                    let _ = turn.try_send(TextOrBlocks::Blocks(reply.content.clone()).text()).inspect_err(|_| note("answer queue full; this answer is not played"));
                }
            }
            Event::AgentSettled { .. } => turn = None,
            _ => {}
        }
        emit(event)
    })
}

// Every process uses one playback lock. A replacement runtime cannot speak over a player that
// the old runtime has not stopped yet. Try-lock keeps hangup and cancellation runnable.
async fn playback_lock(agent_dir: &Path) -> Result<std::fs::File, String> {
    let file = open_lock(agent_dir, "voice.play.lock")?;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) => tokio::time::sleep(CALL_POLL).await,
            Err(error) => return Err(error.to_string()),
        }
    }
}

async fn player(command: &str, text: &str, source: &str) -> bool {
    let spawned = shell(command).env("PICCOSTA_VOICE_SOURCE", source).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(log().map_or(Stdio::null(), Stdio::from)).spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            note(&format!("voice.speak: {error}"));
            return false;
        }
    };
    let mut group = Group(child.id());
    let mut input = child.stdin.take().unwrap();
    let line = format!("{text}\n");
    let write = input.write_all(line.as_bytes());
    let submitted = matches!(tokio::time::timeout(SPEAK_WRITE, write).await, Ok(Ok(())));
    drop(input);
    let completed = if submitted { matches!(tokio::time::timeout(SPEAK_EXIT, wait_group(&mut child, &mut group)).await, Ok(Ok(status)) if status.success()) } else { false };
    if !completed {
        note(&format!("{source} playback failed or timed out"));
    }
    drop(group);
    kill_group(&mut child).await;
    completed
}

// Reserve the output only while playing. Holding it while waiting for main's tokens would
// stop a new quick reply from speaking during a slow request or tool call.
async fn say(command: &str, text: &str) {
    let _audio = match playback_lock(&crate::config::agent_dir()).await {
        Ok(lock) => lock,
        Err(error) => return note(&error),
    };
    let _ = player(command, text, "main").await;
}

async fn play(mut turn: Turn, stop: &CancellationToken) {
    let agent_dir = crate::config::agent_dir();
    if turn.closed.is_cancelled() || over(&agent_dir, &turn.call) {
        return;
    }
    let playback = async {
        if turn.quick_pending
            && let Some(recording) = &turn.recording
        {
            let ready = async {
                while !recording.0.join("quick-playback.json").is_file() {
                    tokio::time::sleep(REAP_POLL).await;
                }
            };
            if tokio::time::timeout(SPEAK_EXIT, ready).await.is_err() {
                note("quick playback receipt missing; main continues");
            }
        }
        // Optional speech-to-speech is independent of quick. Its text is never mislabeled as
        // heard; quick and main share only the generated quick reply recorded in the session.
        if let (Some(command), Some(recording)) = (&turn.reply, &turn.recording)
            && let Ok(input) = std::fs::File::open(recording.0.join("input.wav"))
        {
            let _audio = match playback_lock(&agent_dir).await {
                Ok(lock) => lock,
                Err(error) => return note(&error),
            };
            let mut child = match shell(command).env("PICCOSTA_VOICE_REPLY_AUDIO", recording.0.join("reply.wav")).stdin(input).stdout(Stdio::null()).stderr(log().map_or(Stdio::null(), Stdio::from)).spawn() {
                Ok(child) => child,
                Err(error) => return note(&format!("voice.reply: {error}")),
            };
            let mut group = Group(child.id());
            let _ = tokio::time::timeout(SPEAK_EXIT, wait_group(&mut child, &mut group)).await;
            drop(group);
            kill_group(&mut child).await;
        }
        if let Some(command) = &turn.speak {
            while let Some(text) = turn.answer.recv().await {
                if !text.trim().is_empty() {
                    say(command, &text).await;
                }
            }
        }
    };
    tokio::select! { () = playback => {}, () = stop.cancelled() => {}, () = turn.closed.cancelled() => {}, () = ended(&agent_dir, &turn.call) => {} }
}
