// Outside control of a terminal UI, on the same owner-only IPC as session runtimes. The marker
// names this terminal's endpoint even on Windows, where named pipes aren't directory entries.
use super::App;
use crate::runtime::{ask, Asked};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot};

// Like runtime's ordinary asks: a local request gets two seconds, not the duration of a turn.
const WAIT: Duration = Duration::from_secs(2);
// Local IPC budgets, not model limits: room for a path and JSON, and a handful of callers.
// Excess work is refused rather than letting it grow with incoming clients.
const MAX_REQUEST_BYTES: usize = 16 * 1024;
const MAX_CLIENTS: usize = 8;
const MAX_QUEUED_REQUESTS: usize = 4;
type Opened = Result<(crate::session::Session, Option<std::time::SystemTime>), String>;
pub(super) struct Request {
    value: Value,
    opened: Option<Opened>,
    answer: oneshot::Sender<Value>,
    written: oneshot::Receiver<()>,
}
struct Endpoint(PathBuf);
impl Drop for Endpoint {
    fn drop(&mut self) {
        crate::ipc::remove(&self.0);
        let _ = std::fs::remove_file(&self.0);
    }
}
pub(super) struct Control {
    _endpoint: Endpoint,
    task: tokio::task::JoinHandle<()>,
    pub requests: mpsc::Receiver<Request>,
}
impl Drop for Control {
    fn drop(&mut self) { self.task.abort(); }
}
impl Control {
    pub fn bind(snapshot: &Path) -> Result<Self, String> {
        let file = snapshot.with_extension("control");
        let listener = crate::ipc::bind(&file).map_err(|error| format!("TUI control {}: {error}", file.display()))?;
        let endpoint = Endpoint(file.clone());
        std::fs::write(&file, "").and_then(|()| crate::platform::permissions(&file, 0o600)).map_err(|error| error.to_string())?;
        let (send, requests) = mpsc::channel(MAX_QUEUED_REQUESTS);
        let task = tokio::spawn(async move {
            let mut clients = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    _ = clients.join_next(), if !clients.is_empty() => {},
                    stream = crate::ipc::accept(&listener) => {
                        let Ok(stream) = stream else { break };
                        if clients.len() >= MAX_CLIENTS { continue; }
                        let send = send.clone();
                        clients.spawn(async move {
                            let (read, mut write) = tokio::io::split(stream);
                            let mut line = Vec::new();
                            let mut read = tokio::io::BufReader::new(read.take(MAX_REQUEST_BYTES as u64));
                            tokio::time::timeout(WAIT, read.read_until(b'\n', &mut line)).await.ok()?.ok()?;
                            if !line.ends_with(b"\n") { return None; }
                            let value: Value = serde_json::from_slice(&line).ok()?;
                            let opened = if value["do"] == "open" {
                                let file = value["file"].as_str().unwrap_or_default().to_string();
                                // Keep this client counted until its bounded read finishes. Timing out
                                // spawn_blocking would leave work behind and let new clients pile it up.
                                Some(tokio::task::spawn_blocking(move || read_session(&file)).await.ok()?)
                            } else { None };
                            let (answer, reply) = oneshot::channel();
                            let (written, received) = oneshot::channel();
                            send.try_send(Request { value, opened, answer, written: received }).ok()?;
                            let _ = tokio::time::timeout(WAIT, async move {
                                write.write_all(format!("{}\n", reply.await.ok()?).as_bytes()).await.ok()?;
                                let _ = written.send(());
                                Some(())
                            }).await;
                            Some(())
                        });
                    }
                }
            }
        });
        Ok(Self { _endpoint: endpoint, task, requests })
    }
}

// Read and validate away from the UI. Check the opened handle too: a path can be replaced
// between metadata and open. Nonblocking open keeps that race from waiting on a Unix FIFO.
fn read_session(file: &str) -> Opened {
    let read = || -> Result<_, std::io::Error> {
        use std::io::Read;
        let path = Path::new(file).canonicalize()?;
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        { use std::os::unix::fs::OpenOptionsExt; options.custom_flags(libc::O_NONBLOCK); }
        if !path.metadata()?.is_file() { return Err(std::io::Error::other("TUI open requires a regular session file")); }
        let input = options.open(&path)?;
        let metadata = input.metadata()?;
        if !metadata.is_file() { return Err(std::io::Error::other("TUI open requires a regular session file")); }
        // No size cap: sessions run to tens of megabytes, and a regular file is finite (the
        // checks above already refuse /dev/zero and FIFOs).
        let mut bytes = Vec::new();
        let mut input = input;
        input.read_to_end(&mut bytes)?;
        Ok((path, bytes, metadata.modified().ok()))
    };
    let (path, bytes, modified) = read().map_err(|error| error.to_string())?;
    crate::session::Session::parse(&path, &String::from_utf8_lossy(&bytes)).map(|session| (session, modified)).map_err(|error| error.to_string())
}

// CLI control never starts a model, refreshes a catalog or writes a session. Status comes from
// the UI itself, not an old snapshot, and only actual terminal input advances its activity time.
pub fn cli(arguments: &[String]) -> Result<(), String> {
    let usage = "Usage: piccosta tui | piccosta open <id or name> [--tui <id or pid>] | piccosta restart [--tui <id or pid>]";
    let mut words = arguments.to_vec();
    let which = if let Some(at) = words.iter().position(|word| word == "--tui") {
        if at + 1 >= words.len() { return Err(usage.into()); }
        let which = words.remove(at + 1);
        words.remove(at);
        Some(which)
    } else { None };
    let command = words.first().map(String::as_str).unwrap_or_default();
    if !matches!((command, words.len()), ("tui" | "restart", 1) | ("open", 2)) || words.iter().skip(1).any(|word| word.starts_with("--")) || (command == "tui" && which.is_some()) {
        return Err(usage.into());
    }
    let agent_dir = crate::config::agent_dir();
    let mut running: Vec<_> = std::fs::read_dir(&agent_dir).into_iter().flatten().flatten().filter_map(|entry| {
        let file = entry.path();
        let name = file.file_name()?.to_str()?;
        let id = name.strip_prefix("tui-state-")?.strip_suffix(".control")?.to_string();
        let Asked::Answered(mut status) = ask(&file, json!({"do": "status"}), WAIT) else { return None };
        status["id"] = json!(id);
        Some((file, status))
    }).collect();
    running.sort_by_key(|(_, status)| std::cmp::Reverse(status["active"].as_u64().unwrap_or_default()));
    if command == "tui" {
        println!("{}", serde_json::to_string_pretty(&running.iter().map(|(_, status)| status).collect::<Vec<_>>()).unwrap_or_default());
        return Ok(());
    }
    let chosen: Vec<_> = running.iter().filter(|(_, status)| which.as_ref().is_none_or(|which| status["id"].as_str().is_some_and(|id| id.starts_with(which)) || status["pid"].as_u64().is_some_and(|pid| which.parse() == Ok(pid)))).collect();
    if which.is_some() && chosen.len() > 1 { return Err("Ambiguous TUI; use its full id or pid from `piccosta tui`".into()); }
    let (file, status) = chosen.first().ok_or("No matching running TUI; see `piccosta tui`")?;
    let mut request = json!({"do": command});
    if command == "open" {
        let host = crate::host::Host::for_args(&crate::cli::Args::default());
        request["file"] = json!(host.find_session(&words[1])?.canonicalize().map_err(|error| error.to_string())?);
    }
    match ask(file, request, WAIT) {
        Asked::Answered(reply) if reply["ok"] == true => { println!("{}", json!({"tui": status["id"], "result": reply})); Ok(()) }
        Asked::Answered(reply) => Err(reply["error"].as_str().unwrap_or("TUI refused request").into()),
        _ => Err("TUI did not confirm the request; inspect `piccosta tui` before retrying".into()),
    }
}

impl App {
    pub(super) fn control(&mut self, request: Request) {
        if request.answer.is_closed() { return; }
        let restart = request.value["do"] == "restart";
        let result = match request.value["do"].as_str() {
            Some("status") => Ok(json!({
                "pid": std::process::id(), "active": self.active, "focus": self.focus,
                "panes": self.panes.iter().map(|pane| self.slots[pane.slot].path.clone()).collect::<Vec<_>>(),
                "open": self.slots.iter().filter(|slot| slot.opened).map(|slot| json!({"id": slot.id, "file": slot.path, "title": slot.title, "working": slot.busy() || slot.working_elsewhere})).collect::<Vec<_>>(),
                "restarting": self.restart_waiting
            })),
            Some("open") if self.single => Err(super::SINGLE.into()),
            Some("open") => request.opened.unwrap().and_then(|(session, modified)| self.open_background(session, modified)).map(|()| json!({"ok": true})),
            Some("restart") => {
                self.command("restart");
                Ok(json!({"ok": true, "waiting": self.slots.iter().any(super::Slot::busy)}))
            }
            _ => Err("No such TUI control request".into()),
        };
        let reply = result.unwrap_or_else(|error| json!({"ok": false, "error": error}));
        let _ = request.answer.send(reply);
        // Only exec needs the wire acknowledgement. The event loop still takes input and updates
        // while a slow client reads; ordinary replies never hold up another request.
        if restart { self.restart_replies.push(request.written); }
    }

    fn open_background(&mut self, session: crate::session::Session, modified: Option<std::time::SystemTime>) -> Result<(), String> {
        let path = session.file().unwrap().to_path_buf();
        let existing = self.slots.iter().position(|slot| slot.path.as_ref() == Some(&path));
        let slot = existing.unwrap_or_else(|| {
            self.slots.push(super::Slot { saved: true, ..super::Slot::new(Some(path), super::NEW_TITLE.into()) });
            self.slots.len() - 1
        });
        // Just a sidebar row, not a new runtime or a prompt. Existing live views stay untouched.
        if self.slots[slot].live.is_none() { self.follow_session(slot, session, modified); }
        self.slots[slot].opened = true;
        self.save_state().map_err(|error| error.to_string())
    }
}
