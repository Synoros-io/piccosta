// The session runtime: what runs a session, the same in every piccosta that runs one (the terminal
// UI, print mode, rpc, ACP, the assistant). It holds the session's agent and does what it's asked
// one thing at a time, except a message, a stop and a close, which it takes at once, mid-turn too.
// It listens on an owner-only local stream named for the session (ipc.rs), so another piccosta
// reaches it the way a frontend in its own process does, with a JSON line:
// - {"do": "message", "text": ...}: steers the turn when one runs (its commands move to the
//   background, so the model reads it at the next step), else starts one;
// - {"do": "stop"}: Esc: running commands move to the background and the turn stops; a run of its
//   own (print mode) then ends;
// - {"do": "close"}: stop, kill its jobs and end;
// - {"do": "status"}: whether it's working and since when, and the session whose command started
//   it and its job, when one did (a subagent);
// - {"do": "subscribe"}: a line when it's subscribed, one for each turn that ends, then one when it
//   ends itself.
// A session runs where its socket takes connections (or would, but for a full backlog: connect),
// and nowhere else. Asking it anything (connecting, writing, reading the answer) has one deadline,
// so nothing here waits on a runtime that's stopped or stuck for longer. A message for a session
// nobody runs goes into its file as a queued message, never sent to a model, which the next runtime
// takes ahead of anything else; a lane gets a run of its own for it (launch).
//
// The session file's lock is held for a moment at each change of hands: a runtime claiming the
// session (binding its socket, reading the file, listening: one step), a sender delivering or
// queueing a message, and a runtime giving the session up (it stops listening and puts what's
// unread into the file). So a message either reaches a runtime, which keeps it until it's read or
// written into the file, or is in the file before the next runtime reads it, and nothing writes the
// file of a session a runtime holds but that runtime.

use crate::agent::{Agent, Event};
use crate::host::{Host, Team};
use crate::message::{AgentMessage, Content, StopReason};
use crate::model::Model;
use crate::session::Session;
use crate::tools::ToolShared;
use serde_json::{json, Value};
use sha2::Digest;
use std::collections::VecDeque;
use std::io::{BufRead, Read, Write};
use crate::ipc::{connect, Timed};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot};

// What a refused start says, after the session file's path.
pub const LIVE_ELSEWHERE: &str = "is open in another piccosta";
// How long a start or a sender waits out another's moment with the session file's lock. A lock
// held longer is a piccosta from before runtimes (0.11.3 and older), which held it as long as it
// ran the session.
const LOCK_WAIT: Duration = Duration::from_secs(2);
const LOCK_POLL: Duration = Duration::from_millis(20);
// How long a question gets for its answer: a runtime answers messages at once; a stop or close
// answers once the turn has stopped, which an interrupted Claude Code answer can take seconds to
// do. A runner that hasn't answered a stop or close by then is killed (end).
const ASK_WAIT: Duration = Duration::from_secs(2);
pub const STOP_WAIT: Duration = Duration::from_secs(15);
// How long status gets. Its listener answers it at once, off the session's thread, so one that
// hasn't within this is stopped or stuck. Kept short because the terminal UI asks every runtime
// in its sessions' folders on the thread its own sessions run on (a stopped assistant held each
// of its turns up by the whole ASK_WAIT, every scan, 2026-11-03).
const STATUS_WAIT: Duration = Duration::from_millis(250);
// Where a session's messages waited before runtimes (0.11.3 and older): <session file>.inbox, a
// JSON line each, {"text": ...}. The runtime that claims the session takes them in (claim).
const INBOX_SUFFIX: &str = ".inbox";
// How long a runtime waits for a request's line once a client connects.
const READ_WAIT: Duration = Duration::from_secs(1);

// What a runtime is asked. Message, Stop and Close are taken at once; the rest wait their turn.
// Stop and Close answer on their sender, when there is one, once they've taken effect.
pub enum Command {
    Prompt(String, Vec<Content>),
    Message(String),
    Stop(Option<oneshot::Sender<()>>),
    Close(Option<oneshot::Sender<()>>),
    // A weak ticket expires when the frontend shows it again or edits its draft.
    // Only the runtime knows whether it has settled.
    ReleaseIfIdle(std::sync::Weak<()>),
    Model(String),
    Thinking(String),
    Name(String),
    Compact(Option<String>),
    AutoCompaction(bool),
    AutoRetry(bool),
    // Work with other models from the next prompt on (none: alone again).
    Team(Team),
}

// What the runtime reports after each piece of work: how it went (`notice`: work no frontend asked
// for, a message or a job's end), and the model and level the session has now.
pub struct Done {
    pub result: Result<(), String>,
    pub notice: bool,
    pub model: Model,
    pub thinking: String,
}

// What the socket reads of a runtime: since when it's working, and who hears its turn ends.
#[derive(Default)]
struct Watch {
    since: Option<i64>,
    subscribers: Vec<mpsc::UnboundedSender<String>>,
}

impl Watch {
    fn tell(&mut self, line: &str) {
        self.subscribers.retain(|subscriber| subscriber.send(line.to_string()).is_ok());
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// --- The socket ---

// The socket: the session file's name with SOCKET_SUFFIX after it.
pub const SOCKET_SUFFIX: &str = ".sock";

pub fn socket_path(file: &Path) -> PathBuf {
    PathBuf::from(format!("{}{SOCKET_SUFFIX}", file.display()))
}

// The session file's lock, for a moment. None when the file isn't there yet: a session not on disk
// has a new name nobody else has.
fn hold(file: &Path) -> Result<Option<std::fs::File>, String> {
    let Ok(handle) = std::fs::File::open(file) else { return Ok(None) };
    let deadline = Instant::now() + LOCK_WAIT;
    loop {
        match crate::platform::try_session_lock(&handle) {
            Ok(()) => return Ok(Some(handle)),
            Err(std::fs::TryLockError::WouldBlock) if Instant::now() < deadline => std::thread::sleep(LOCK_POLL),
            Err(std::fs::TryLockError::WouldBlock) => return Err(format!("{} {LIVE_ELSEWHERE} (an older one, which needs a restart to let go of it)", file.display())),
            Err(std::fs::TryLockError::Error(error)) => return Err(format!("{}: {error}", file.display())),
        }
    }
}

// The way into a claimed session's runtime: what the socket takes goes through it, until it's shut
// (None) when the runtime gives the session up, or the claim goes unrun.
type Gate = Arc<Mutex<Option<mpsc::UnboundedSender<Command>>>>;

// The session, claimed by this process: its socket, listening, and what it has taken, which the
// runtime reads (`taken`). This process runs the session while this lives.
pub struct Claim {
    file: PathBuf,
    listening: Option<tokio::task::JoinHandle<()>>,
    gate: Gate,
    taken: mpsc::UnboundedReceiver<Command>,
    watch: Arc<Mutex<Watch>>,
}

// A claim no runtime gave up (its open failed after claiming, say): the messages it took go into
// the file for the next runtime, under the file's lock, then the socket goes.
impl Drop for Claim {
    fn drop(&mut self) {
        if lock(&self.gate).is_some() {
            let _held = hold(&self.file);
            lock(&self.gate).take();
            let mut session = Session::open(&self.file, self.file.parent().unwrap_or(Path::new("/"))).ok();
            while let Ok(command) = self.taken.try_recv() {
                if let (Command::Message(text), Some(session)) = (command, session.as_mut())
                    && !text.is_empty()
                {
                    let _ = session.queue(&text);
                }
            }
        }
        if let Some(listening) = self.listening.take() {
            listening.abort();
        }
        crate::ipc::remove(&self.file);
    }
}

// Claim a session for this process to run, in one step under its file's lock: bind its socket (one
// left by a piccosta that died takes no connections and is taken over), read the file again so the
// session holds whatever was queued meanwhile, take in an old inbox, and start listening. Output:
// the claim, None for a session kept in memory, or an error when another piccosta runs it, or its
// old inbox couldn't be kept.
pub fn claim(session: &mut Session) -> Result<Option<Claim>, String> {
    let Some(file) = session.file().map(Path::to_path_buf) else { return Ok(None) };
    let _held = hold(&file)?;
    let failed = |error: std::io::Error| format!("{}: {error}", socket_path(&file).display());
    std::fs::create_dir_all(file.parent().unwrap_or(Path::new("/"))).map_err(failed)?;
    let listener = crate::ipc::bind(&file).map_err(|error| if error.kind() == std::io::ErrorKind::AddrInUse { format!("{} {LIVE_ELSEWHERE}", file.display()) } else { failed(error) })?;
    if file.exists() {
        let cwd = PathBuf::from(&session.header().cwd);
        *session = Session::open(&file, &cwd).map_err(|error| error.to_string())?;
    }
    // An old inbox, read whole under its own lock as its senders wrote it, goes into the session as
    // one queued entry holding its messages and naming it (its size and sha256), and is unlinked
    // after. That entry is written whole or not at all, so an inbox still there once it's written
    // (the piccosta taking it died before the unlink) is known by it and only unlinked. One that
    // can't be locked or read whole, or beside a session not on disk yet, is left where it is.
    let inbox = PathBuf::from(format!("{}{INBOX_SUFFIX}", file.display()));
    let mut old = String::new();
    if file.exists()
        && let Ok(Some(handle)) = hold(&inbox)
        && (&handle).read_to_string(&mut old).is_ok()
    {
        let hash: String = sha2::Sha256::digest(&old).iter().map(|byte| format!("{byte:02x}")).collect();
        let name = json!({"size": old.len(), "sha256": hash});
        let taken = session.entries().iter().any(|entry| matches!(entry, crate::session::SessionEntry::Custom { custom_type, data: Some(data), .. } if custom_type == crate::session::QUEUED_TYPE && data["inbox"] == name));
        let texts: Vec<Value> = old.lines().filter_map(|line| serde_json::from_str::<Value>(line).ok()?.get("text").filter(|text| text.is_string()).cloned()).collect();
        if !taken && !texts.is_empty() {
            session.append_custom(crate::session::QUEUED_TYPE, Some(json!({"texts": texts, "inbox": name}))).map_err(|error| format!("{}: {error}", file.display()))?;
        }
        let _ = std::fs::remove_file(&inbox);
    }
    let (taken_by, taken) = mpsc::unbounded_channel();
    let (gate, watch): (Gate, Arc<Mutex<Watch>>) = (Arc::new(Mutex::new(Some(taken_by))), Arc::default());
    let listening = Some(listen(listener, gate.clone(), watch.clone()));
    Ok(Some(Claim { file, listening, gate, taken, watch }))
}

// Take requests on the socket through the gate, one connection at a time, until the task is
// aborted. That can come mid-request, since it comes under the file's lock (the runtime giving the
// session up, or an unrun claim going), and a message is only ever sent under it, so the one cut
// short is never a message. A request is a whole line: one its sender didn't finish writing is
// never acted on, so a sender whose write failed knows it wasn't (ask). A stop or close is answered
// by a task of its own once it has taken effect, a subscriber's lines by another.
fn listen(listener: crate::ipc::Listener, gate: Gate, watch: Arc<Mutex<Watch>>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let Ok(stream) = crate::ipc::accept(&listener).await else { return };
            let (read, mut write) = tokio::io::split(stream);
            let mut line = String::new();
            if !matches!(tokio::time::timeout(READ_WAIT, tokio::io::BufReader::new(read).read_line(&mut line)).await, Ok(Ok(1..))) || !line.ends_with('\n') {
                continue;
            }
            let request: Value = serde_json::from_str(&line).unwrap_or_default();
            let send = |command| lock(&gate).as_ref().is_some_and(|taken| taken.send(command).is_ok());
            let reply = match request["do"].as_str().unwrap_or_default() {
                "status" => {
                    let since = lock(&watch).since;
                    let (parent, job) = (std::env::var(crate::tools::subagent::SESSION_ENV).ok(), std::env::var(crate::tools::subagent::CHILD_ENV).ok());
                    json!({"working": since.is_some(), "since": since, "parent": parent, "job": job})
                }
                "message" => json!({"ok": send(Command::Message(request["text"].as_str().unwrap_or_default().to_string()))}),
                kind @ ("stop" | "close") => {
                    let (answer, answered) = oneshot::channel();
                    let sent = send(if kind == "stop" { Command::Stop(Some(answer)) } else { Command::Close(Some(answer)) });
                    tokio::spawn(async move {
                        let done = sent && answered.await.is_ok();
                        let _ = write.write_all(format!("{}\n", json!({"ok": done})).as_bytes()).await;
                    });
                    continue;
                }
                "subscribe" => {
                    let (subscriber, mut lines) = mpsc::unbounded_channel();
                    lock(&watch).subscribers.push(subscriber);
                    tokio::spawn(async move {
                        let mut next = Some(json!({"ok": true}).to_string());
                        while let Some(line) = next {
                            if write.write_all(format!("{line}\n").as_bytes()).await.is_err() {
                                break;
                            }
                            next = lines.recv().await;
                        }
                    });
                    continue;
                }
                _ => json!({"ok": false, "error": "no such request"}),
            };
            let _ = write.write_all(format!("{reply}\n").as_bytes()).await;
        }
    })
}

// --- Asking a runtime from outside ---

// A process known for sure: its pid, its start time and the boot it started in. A pid passes on
// once its process is gone, and a start time counts from boot, so only all three name one process:
// a run launch started, as it started, or the runtime on a socket, as it was connected to.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Run {
    pub pid: i32,
    start: u64,
    boot: String,
}

impl Run {
    // The live process `pid` names now; None once it's gone.
    fn of(pid: i32) -> Option<Run> {
        Some(Run { pid, start: crate::proc::start(pid)?, boot: crate::proc::boot_id() })
    }
}

// What came of asking a runtime (ask).
pub enum Asked {
    // Nothing listens on the session's socket: nobody runs it.
    Nobody,
    // The request never reached it whole: its socket isn't taking connections (stopped or stuck,
    // with its backlog full), or the write didn't finish in time. It can't have acted on it.
    Unread,
    // Its answer; Null when the connection ended without one, or with a garbled one: it ended or
    // died, maybe after acting on the request.
    Answered(Value),
    // It has the request whole and didn't answer in time (it's stopped, or stuck); who it is, by
    // its socket's peer credentials, when they could be read.
    Silent(Option<Run>),
}

// Ask the runtime of the session in `file`, all of it (connecting, writing, reading the answer)
// within `wait`.
pub fn ask(file: &Path, request: Value, wait: Duration) -> Asked {
    let deadline = Instant::now() + wait;
    let stream = match connect(file) {
        Ok(stream) => stream,
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Asked::Unread,
        Err(_) => return Asked::Nobody,
    };
    let runner = crate::ipc::peer(&stream).and_then(Run::of);
    if Timed(&stream, deadline).write_all(format!("{request}\n").as_bytes()).is_err() {
        return Asked::Unread;
    }
    let mut line = String::new();
    match std::io::BufReader::new(Timed(&stream, deadline)).read_line(&mut line) {
        Err(error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => Asked::Silent(runner),
        _ => Asked::Answered(serde_json::from_str(&line).unwrap_or_default()),
    }
}

// What the runtime of the session in `file` says of itself (status above): None when nothing runs
// it, Null when it didn't answer in time (it's stopped, or stuck), though it runs it.
pub fn status(file: &Path) -> Option<Value> {
    match ask(file, json!({"do": "status"}), STATUS_WAIT) {
        Asked::Nobody => None,
        Asked::Answered(status) => Some(status),
        Asked::Unread | Asked::Silent(_) => Some(Value::Null),
    }
}

// Whether something runs the session in `file`: its socket takes a connection, or would but for a
// full backlog (connect).
pub fn running(file: &Path) -> bool {
    connect(file).map_or_else(|error| error.kind() == std::io::ErrorKind::WouldBlock, |_| true)
}

// Stop or close (`what`) the session in `file` from outside, as a manager does. A runner that
// doesn't answer within STOP_WAIT (stopped, stuck) is killed, and the commands it ran go with it
// (its watcher, reaper.rs), but only when it's the lane run the manager itself started (`ours`):
// the same pid, start time and boot. A terminal UI or the assistant that took that pid since holds
// other sessions, as does anyone else's run, and is never killed for one. Output: None when nothing
// runs it; else whether it answered, or what was done instead.
pub fn end(file: &Path, what: &str, ours: Option<&Run>) -> Option<Result<(), String>> {
    let waited = STOP_WAIT.as_secs();
    Some(match ask(file, json!({"do": what}), STOP_WAIT) {
        Asked::Nobody => return None,
        Asked::Answered(_) => Ok(()),
        Asked::Silent(Some(runner)) if ours != Some(&runner) => {
            Err(format!("it didn't answer in {waited} s; the piccosta running it (pid {}) isn't the run you started (not the same pid, start time and boot), so it was left alone", runner.pid))
        }
        Asked::Silent(Some(runner)) if crate::proc::Process::started(runner.pid, runner.start).is_some_and(|process| process.signal(crate::proc::KILL)) => {
            Err(format!("it didn't answer in {waited} s, so its run (pid {}) was killed, with everything it ran", runner.pid))
        }
        Asked::Silent(_) => Err(format!("it didn't answer in {waited} s, and the piccosta running it couldn't be found to kill")),
        Asked::Unread => Err("the piccosta running it isn't taking requests (it's stopped, or stuck), so it couldn't be told, nor known for sure to kill: it was left alone".into()),
    })
}

// Change the session in `file` while nothing runs it, under its lock so no runtime starts meanwhile.
// Output: None when a runtime runs it.
pub fn when_idle<T>(file: &Path, change: impl FnOnce(&mut Session) -> T) -> Result<Option<T>, String> {
    let _held = hold(file)?;
    if running(file) {
        return Ok(None);
    }
    let mut session = Session::open(file, file.parent().unwrap_or(Path::new("/"))).map_err(|error| error.to_string())?;
    Ok(Some(change(&mut session)))
}

// What became of a message (send). Unconfirmed: the runtime running it has it whole but didn't
// answer in time (it's stopped, or stuck), and reads it if it goes on; or its connection ended
// without an answer, maybe after it took the message. Either way it isn't sent again.
pub enum Sent {
    Delivered,
    Unconfirmed,
    Queued,
    Started(Run),
}

// A message for the session in `file`, under the file's lock: to the runtime running it, else into
// the file as a queued message for the next one, and with `launch` (the directory and log for a run
// of its own) a run started now, unless it's a manager's session, whose runs spend its model and
// wait for its service or a terminal. A message once sent is never sent again.
pub fn send(file: &Path, text: &str, launch: Option<(&Path, &Path)>) -> Result<Sent, String> {
    let held = hold(file)?;
    match ask(file, json!({"do": "message", "text": text}), ASK_WAIT) {
        Asked::Nobody => {}
        Asked::Answered(answer) if answer["ok"] == true => return Ok(Sent::Delivered),
        Asked::Answered(Value::Null) | Asked::Silent(_) => return Ok(Sent::Unconfirmed),
        Asked::Answered(_) => return Err(format!("{} is running but didn't take the message", file.display())),
        Asked::Unread => return Err(format!("{} is running but isn't taking messages (it's stopped, or stuck), so it wasn't sent", file.display())),
    }
    if held.is_none() {
        return Err(format!("{}: no such session", file.display()));
    }
    let mut session = Session::open(file, file.parent().unwrap_or(Path::new("/"))).map_err(|error| error.to_string())?;
    session.queue(text).map_err(|error| format!("{}: {error}", file.display()))?;
    let lane = crate::manager::role(&session).is_none();
    drop(held);
    match launch.filter(|_| lane) {
        Some((cwd, log)) => self::launch(file, cwd, log).map(Sent::Started).map_err(|error| format!("couldn't start its run: {error}")),
        None => Ok(Sent::Queued),
    }
}

// The runs launch started that have ended, for a manager watching its lanes.
static ENDED: Mutex<Vec<Run>> = Mutex::new(Vec::new());

pub fn ended(run: &Run) -> bool {
    lock(&ENDED).contains(run)
}

// Run the session in `file` in a piccosta of its own (print mode), apart from whoever starts it: in
// a session and process group of its own, so it carries on when the manager, its terminal or its
// service stops (the service's unit has KillMode=process for this). It answers what's queued in
// its file, then whatever it's sent while it runs, and ends when nothing is left. What it prints
// goes to `log`. Output: the run, by its identity as it started.
pub fn launch(file: &Path, cwd: &Path, log: &Path) -> std::io::Result<Run> {
    std::fs::create_dir_all(log.parent().unwrap_or(Path::new(".")))?;
    let output = std::fs::File::create(log)?;
    let mut command = std::process::Command::new(crate::proc::exe()?);
    command.arg("-p").arg("--session").arg(file).current_dir(cwd);
    // A lane's run is the manager's, not a command's, even when an agent's `sessions message`
    // starts it (which wakes a project manager's lane, or the manager itself).
    command.env_remove(crate::tools::subagent::SESSION_ENV);
    command.stdin(std::process::Stdio::null()).stdout(output.try_clone()?).stderr(output);
    let mut child = crate::proc::own_session(&mut command).spawn()?;
    // Read before it can be reaped (below), so its pid is still its own.
    let run = Run::of(child.id() as i32);
    // Waited for, so it isn't left a zombie while this piccosta runs; if this one ends first, init
    // takes it over.
    let ending = run.clone();
    std::thread::spawn(move || {
        let _ = child.wait();
        lock(&ENDED).extend(ending);
    });
    run.ok_or_else(|| std::io::Error::other("it ended as it started; its log says why"))
}

// Subscribe to the turn ends of the runtime running the session in `file`: each line it sends goes
// to `heard`, until it ends. Output: whether it ended saying so, rather than going silent.
pub async fn subscribe(file: &Path, mut heard: impl FnMut(Value)) -> bool {
    let Ok(mut stream) = crate::ipc::connect_async(file).await else { return false };
    if stream.write_all(format!("{}\n", json!({"do": "subscribe"})).as_bytes()).await.is_err() {
        return false;
    }
    let mut lines = tokio::io::BufReader::new(stream).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let line: Value = serde_json::from_str(&line).unwrap_or_default();
        if line["end"] == true {
            return true;
        }
        heard(line);
    }
    false
}

// --- The runtime ---

enum Work {
    Command(Command),
    Notice(crate::message::CustomMessage),
}

// Start a session's runtime. Input: what Host::open made; `wake`, whether it starts turns no
// frontend asked for (a job's end, a message, what was queued for it); `until_idle`, whether it
// ends when it has nothing left to do (a run of its own) rather than when its frontend lets go of
// it; and where its reports go. Output: where its commands go, and its task, which hands the agent
// back when it ends. Runs on a LocalSet: the agent's future isn't Send (rustc can't prove it
// through the providers' higher-ranked lifetimes).
pub fn spawn(host: Arc<Host>, mut agent: Agent, shared: Arc<ToolShared>, wake: bool, until_idle: bool, mut done: impl FnMut(Done) + 'static) -> (mpsc::UnboundedSender<Command>, tokio::task::JoinHandle<Agent>) {
    let (sender, mut commands) = mpsc::unbounded_channel();
    let mut claim = agent.take_claim();
    let watch = claim.as_ref().map_or_else(Arc::default, |claim| claim.watch.clone());
    // Each turn the model ends goes to the subscribers, with how many jobs still run.
    let (heard, jobs) = (watch.clone(), shared.clone());
    agent.tap(move |event| {
        if let Event::AgentEnd { messages, will_retry: false } = event
            && let Some(AgentMessage::Assistant(last)) = messages.last()
            && last.stop_reason != StopReason::ToolUse
        {
            lock(&heard).tell(&json!({"turn_end": {"jobs": jobs.jobs.running_count()}}).to_string());
        }
    });
    let wait_minutes = host.settings.get("manager.inactiveMinutes").and_then(Value::as_u64).unwrap_or(crate::manager::INACTIVE_MINUTES);
    let watcher = agent.manager_home().map(|home| tokio::task::spawn_local(crate::manager::watch(home, sender.downgrade(), wait_minutes)));
    let task = tokio::task::spawn_local(async move {
        let handle = agent.handle();
        let mut team = crate::host::team(&agent.session);
        let mut queue: VecDeque<Command> = VecDeque::new();
        let mut preparing = Box::pin(host.ready(&mut agent));
        let ready = loop {
            tokio::select! {
                result = &mut preparing => break result,
                command = next(&mut commands, &mut claim) => match command {
                    Some(Command::Close(answer)) => {
                        if let Some(answer) = answer { let _ = answer.send(()); }
                        break Err("Session closed during plugin discovery".into());
                    }
                    Some(command) => queue.push_back(command),
                    None => break Err("Session closed during plugin discovery".into()),
                }
            }
        };
        drop(preparing);
        if let Err(error) = ready {
            done(Done { result: Err(error), notice: false, model: agent.model().clone(), thinking: agent.thinking_level().to_string() });
            if let Some(watcher) = watcher { watcher.abort(); }
            shared.jobs.shutdown("plugin discovery ended");
            shared.web.shutdown();
            return agent;
        }
        // What the last runner left queued is taken at once.
        if wake && !agent.session.queued().is_empty() {
            queue.push_back(Command::Message(String::new()));
        }
        let mut closing: Option<Option<oneshot::Sender<()>>> = None;
        // The session file's lock, while a run of its own lets go.
        let mut held = None;
        loop {
            let work = match queue.pop_front() {
                Some(command) => Work::Command(command),
                None => tokio::select! {
                    command = next(&mut commands, &mut claim) => match command {
                        Some(command) => Work::Command(command),
                        None => break,
                    },
                    Some(notice) = agent.notice(), if wake => Work::Notice(notice),
                },
            };
            let work = match work {
                Work::Command(Command::Stop(answer)) => {
                    answer.map(|answer| answer.send(()));
                    if until_idle {
                        break;
                    }
                    continue;
                }
                Work::Command(Command::Close(answer)) => {
                    closing = Some(answer);
                    break;
                }
                Work::Command(Command::ReleaseIfIdle(generation)) => {
                    if generation.upgrade().is_none() { continue; }
                    held = agent.session.file().map(hold);
                    if queue.is_empty() && commands.is_empty() && claim.as_ref().is_none_or(|claim| claim.taken.is_empty()) && shared.jobs.running_count() == 0 && !agent.has_pending_work() {
                        break;
                    }
                    held = None;
                    continue;
                }
                // A frontend that starts every turn itself (rpc, ACP) gets an idle message with
                // its next prompt.
                Work::Command(Command::Message(text)) if !wake && !text.is_empty() => {
                    if let Err(error) = agent.session.queue(&text) {
                        eprintln!("piccosta: could not queue a message: {error}");
                    }
                    continue;
                }
                Work::Notice(notice) if crate::tools::jobs::stopped_by_user(&notice) => {
                    agent.hold_notice(notice);
                    continue;
                }
                work => work,
            };
            let notice = matches!(work, Work::Notice(_) | Work::Command(Command::Message(_)));
            lock(&watch).since = Some(crate::clock::now_ms());
            let mut stops = Vec::new();
            let result = {
                let run = perform(&host, &mut agent, &mut team, work);
                tokio::pin!(run);
                loop {
                    tokio::select! {
                        result = &mut run => break result,
                        command = next(&mut commands, &mut claim), if closing.is_none() => match command {
                            Some(Command::Message(text)) => {
                                handle.steer(text);
                                shared.jobs.detach_all();
                            }
                            Some(Command::Stop(answer)) => {
                                shared.jobs.detach_all();
                                handle.abort();
                                stops.push(answer);
                            }
                            Some(Command::Close(answer)) => {
                                shared.jobs.detach_all();
                                handle.abort();
                                closing = Some(answer);
                            }
                            Some(other) => queue.push_back(other),
                            None => {
                                handle.abort();
                                closing = Some(None);
                            }
                        },
                    }
                }
            };
            lock(&watch).since = None;
            // What a stop left unread stays queued here, for the session's next turn (or rpc's
            // clear_queue); it goes into the file when the session is given up.
            let stopped = !stops.is_empty();
            stops.into_iter().flatten().for_each(|answer| {
                let _ = answer.send(());
            });
            done(Done { result, notice, model: agent.model().clone(), thinking: agent.thinking_level().to_string() });
            if closing.is_some() || (until_idle && stopped) {
                break;
            }
            // A run of its own lets go of the session when nothing's left, unless something came
            // in as it did, which it answers first. Under the file's lock, which a sender delivers
            // under too: once it's held, every message sent is in, and a sender after it finds
            // nobody and queues its message in the file.
            if until_idle && queue.is_empty() {
                held = agent.session.file().map(hold);
                if commands.is_empty() && claim.as_ref().is_none_or(|claim| claim.taken.is_empty()) {
                    break;
                }
                held = None;
            }
        }
        // The end, under the file's lock: the socket stops taking requests and the gate shuts, what's
        // unread goes into the file, the socket goes, then the jobs.
        let held = held.or_else(|| agent.session.file().map(hold));
        commands.close();
        let mut left = Vec::new();
        if let Some(claim) = claim.as_mut() {
            if let Some(listening) = claim.listening.take() {
                listening.abort();
                let _ = listening.await;
            }
            lock(&claim.gate).take();
            left.extend(std::iter::from_fn(|| claim.taken.try_recv().ok()));
        }
        for command in left.into_iter().chain(std::iter::from_fn(|| commands.try_recv().ok())) {
            match command {
                Command::Message(text) if !text.is_empty() => handle.steer(text),
                Command::Stop(Some(answer)) | Command::Close(Some(answer)) => {
                    let _ = answer.send(());
                }
                _ => {}
            }
        }
        requeue(&mut agent);
        lock(&watch).tell(&json!({"end": true}).to_string());
        drop(claim);
        drop(held);
        if let Some(watcher) = watcher {
            watcher.abort();
        }
        crate::provider::claude::close(agent.session.id());
        shared.jobs.shutdown("quit");
        shared.web.shutdown();
        if let Some(Some(answer)) = closing {
            let _ = answer.send(());
        }
        agent
    });
    (sender, task)
}

// The next command for a runtime: its frontend's, or one its socket took. None once the frontend
// has let go of it.
async fn next(commands: &mut mpsc::UnboundedReceiver<Command>, claim: &mut Option<Claim>) -> Option<Command> {
    let taken = async {
        match claim {
            Some(claim) => claim.taken.recv().await,
            None => std::future::pending().await,
        }
    };
    tokio::select! {
        command = commands.recv() => command,
        Some(command) = taken => Some(command),
    }
}

// One piece of work: a frontend's command, a message (a turn on it, with the session's team), or a
// background job's end.
async fn perform(host: &Host, agent: &mut Agent, team: &mut Team, work: Work) -> Result<(), String> {
    let written = |result: std::io::Result<()>| result.map_err(|error| format!("could not write the session: {error}"));
    let Work::Command(command) = work else {
        let Work::Notice(notice) = work else { unreachable!("a command is matched above") };
        return agent.take_notice(notice).await;
    };
    match command {
        Command::Prompt(text, images) => host.answer(agent, team, &text, images).await,
        Command::Message(text) => host.answer(agent, team, &text, Vec::new()).await,
        Command::Team(asked) => {
            *team = crate::host::record_team(&host.registry()?, &mut agent.session, &asked)?;
            Ok(())
        }
        Command::Compact(instructions) => agent.compact(instructions.as_deref()).await,
        Command::Thinking(level) => written(agent.set_thinking_level(&level)),
        Command::Name(name) => written(agent.set_name(&name)),
        Command::AutoCompaction(enabled) => {
            agent.set_auto_compaction(enabled);
            Ok(())
        }
        Command::AutoRetry(enabled) => {
            agent.set_auto_retry(enabled);
            Ok(())
        }
        Command::Model(name) => host.switch_model(agent, &name),
        Command::Stop(_) | Command::Close(_) | Command::ReleaseIfIdle(_) => Ok(()),
    }
}

// Put what the agent has queued and not taken (steering, follow-ups) into the session file, where
// its next turn, here or in the next runtime, takes it first.
fn requeue(agent: &mut Agent) {
    let (steer, follow_up) = agent.handle().clear();
    for text in steer.into_iter().chain(follow_up) {
        if let Err(error) = agent.session.queue(&text) {
            eprintln!("piccosta: could not keep a message: {error}");
        }
    }
}
