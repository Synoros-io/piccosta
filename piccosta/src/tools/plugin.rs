// Executables in the agent dir's plugins folder describe their tools once at session open.
// Calls share one lazy process per plugin per session. Its group belongs to the same registry
// and watcher as bash, not to the turn: cancelling a call sends a message, closing kills it.
use super::{Tool, ToolContext, ToolFuture, ToolOutput};
use crate::message::{Content, ToolDefinition};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use base64::Engine;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Mutex;

pub const PLUGINS_DIR: &str = "plugins";
pub struct Discovery(pub tokio::task::JoinHandle<Vec<Arc<dyn Tool>>>);
impl Drop for Discovery { fn drop(&mut self) { self.0.abort(); } }
// Describe is local metadata, not a model call. A broken executable must not hold session open.
const DESCRIBE_WAIT: Duration = Duration::from_secs(5);
// Local desktop startup can take seconds. Two minutes leaves room for it, not an endless turn.
// Override with plugins.callTimeoutSeconds in settings.json (positive seconds).
pub const CALL_WAIT: Duration = Duration::from_secs(120);
// A plugin is an interactive tool, not a batch job: allow at most an hour per call.
pub const MAX_CALL_WAIT: Duration = Duration::from_secs(60 * 60);
// Enough for an uncompressed 2000x2000 RGBA screenshot in base64, plus its text and JSON.
const MAX_LINE_BYTES: usize = 32 * 1024 * 1024;
// A plugin gets a short chance to acknowledge cancellation; otherwise its protocol is reset.
const CANCEL_WAIT: Duration = Duration::from_secs(1);

#[derive(Deserialize)]
pub struct Description {
    pub name: String,
    pub version: String,
    pub tools: Vec<OfferedTool>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfferedTool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub kinds: Option<Vec<String>>,
    #[serde(default)]
    pub read_only: bool,
    pub asks: Option<Asks>,
}

// A single condition keeps the original wire shape; a list asks when any condition matches.
#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub enum Asks { One(Ask), Any(Vec<Ask>) }
impl Asks {
    fn matching(&self, arguments: &Map<String, Value>) -> Option<&Ask> {
        let asks = match self { Self::One(ask) => std::slice::from_ref(ask), Self::Any(asks) => asks };
        asks.iter().find(|ask| ask.when.iter().all(|(key, value)| arguments.get(key) == Some(value)))
    }
}

// Every key in `when` must equal the argument's value. An empty object asks on every call.
#[derive(Clone, Deserialize)]
pub struct Ask {
    pub when: Map<String, Value>,
    pub question: String,
}

struct Plugin {
    path: PathBuf,
    name: String,
    process: Mutex<Option<Process>>,
}

struct PluginTool {
    plugin: Arc<Plugin>,
    offered: OfferedTool,
}

fn executable(path: &Path) -> bool {
    if !path.is_file() { return false; }
    #[cfg(unix)]
    { use std::os::unix::fs::PermissionsExt; path.metadata().is_ok_and(|m| m.permissions().mode() & 0o111 != 0) }
    #[cfg(windows)]
    { path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("exe")) }
}

// Discovery has no session identity. The same supervisor, sandbox and watcher as calls own it.
// Async pipe reads and one deadline bound both an endless writer and a helper holding stdout.
async fn describe(path: &Path, cwd: &Path, agent_dir: &Path, read_only: bool) -> Result<Description, String> {
    if read_only { super::readonly::check()?; }
    let mut host = super::plugin_host::command(path, read_only).map_err(|e| e.to_string())?;
    let command = host.command.as_mut().unwrap();
    command.arg("describe").current_dir(cwd).env(crate::config::AGENT_DIR_ENV, agent_dir)
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    for (key, _) in std::env::vars_os().filter(|(key, _)| key.to_string_lossy().starts_with("PICCOSTA_SESSION_")) { command.env_remove(key); }
    crate::proc::shell_setup(command, false);
    let mut child = tokio::process::Command::from(host.command.take().unwrap()).spawn().map_err(|e| e.to_string())?;
    let pid = child.id().ok_or("describe has no pid")?;
    let guard = match host.attach(pid) { Ok(owner) => owner, Err(error) => { let _ = child.start_kill(); return Err(error.to_string()); } };
    let owner = guard.clone();
    let stdout = child.stdout.take().ok_or("describe has no stdout")?;
    let waited = tokio::spawn(async move {
        let status = child.wait().await;
        owner.kill();
        #[cfg(unix)]
        super::reaper::track_plugin(pid, owner.start, false);
        crate::proc::release(pid);
        status
    });
    let mut bytes = Vec::new();
    stdout.take((MAX_LINE_BYTES + 1) as u64).read_to_end(&mut bytes).await.map_err(|e| e.to_string())?;
    if bytes.len() > MAX_LINE_BYTES { return Err("describe output is too large".into()); }
    if !waited.await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?.success() { return Err("describe exited unsuccessfully".into()); }
    drop(guard);
    let description: Description = serde_json::from_slice(&bytes).map_err(|e| format!("describe: {e}"))?;
    let valid_name = |name: &str| !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if description.name.is_empty() || description.version.is_empty() || description.tools.iter().any(|tool| !valid_name(&tool.name) || !tool.parameters.is_object() || tool.parameters.get("type").and_then(Value::as_str) != Some("object")) {
        return Err("describe needs a name, version and tools with names and object parameters schemas".into());
    }
    Ok(description)
}

// Snapshot installed executables at session open, then do discovery on a runtime worker, not the
// TUI's LocalSet. Each executable has one bounded async read; no reader threads survive a timeout.
pub fn discover(agent_dir: &Path, cwd: &Path, kind: Option<String>, read_only: bool, existing: Vec<Arc<dyn Tool>>) -> Option<tokio::task::JoinHandle<Vec<Arc<dyn Tool>>>> {
    let mut paths: Vec<_> = std::fs::read_dir(agent_dir.join(PLUGINS_DIR)).into_iter().flatten().flatten().map(|entry| entry.path()).filter(|path| executable(path)).collect();
    if paths.is_empty() { return None; }
    paths.sort();
    let (agent_dir, cwd) = (agent_dir.to_path_buf(), cwd.to_path_buf());
    Some(tokio::spawn(async move {
    let (kind, deadline) = (kind.as_deref(), tokio::time::Instant::now() + DESCRIBE_WAIT);
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    for path in paths {
        if tokio::time::Instant::now() >= deadline { break; }
        let described = match tokio::time::timeout_at(deadline, describe(&path, &cwd, &agent_dir, read_only)).await {
            Ok(result) => result,
            Err(_) => { eprintln!("Plugin {}: describe timed out", path.display()); break; }
        };
        let description = match described {
            Ok(description) => description,
            Err(error) => { eprintln!("Plugin {}: {error}", path.display()); continue; }
        };
        let plugin = Arc::new(Plugin { path, name: description.name, process: Mutex::new(None) });
        for offered in description.tools {
            if offered.kinds.as_ref().is_some_and(|kinds| !kind.is_some_and(|kind| kinds.iter().any(|k| k == kind))) || (read_only && !offered.read_only) { continue; }
            if existing.iter().chain(&tools).any(|tool| tool.name() == offered.name) {
                eprintln!("Plugin {}: duplicate tool {}", plugin.name, offered.name);
                continue;
            }
            tools.push(Arc::new(PluginTool { plugin: plugin.clone(), offered }));
        }
    }
    tools
    }))
}

impl Tool for PluginTool {
    fn name(&self) -> &str { &self.offered.name }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition { name: self.offered.name.clone(), description: self.offered.description.clone(), parameters: self.offered.parameters.clone(), constrained_sampling: None }
    }
    fn prompt_snippet(&self) -> Option<&str> { Some(&self.offered.description) }
    fn read_only(&self) -> bool { self.offered.read_only }
    fn approval(&self, arguments: &Map<String, Value>) -> Option<String> {
        self.offered.asks.as_ref().and_then(|asks| asks.matching(arguments)).map(|ask| ask.question.clone())
    }
    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let approved = self.approval(&arguments).is_some();
            self.plugin.call(&self.offered.name, arguments, approved, context).await.map_err(|e| format!("Plugin {}: {e}", self.plugin.name))
        })
    }
}

struct Process {
    _lifetime: super::plugin_host::Owner,
    stdin: tokio::process::ChildStdin,
    stdout: Arc<crate::proc::Receiver>,
    buffered: Vec<u8>,
}

impl Process {
    fn spawn(path: &Path, context: &ToolContext) -> Result<Self, String> {
        let (reader, writer) = std::io::pipe().map_err(|e| e.to_string())?;
        let stdout = Arc::new(crate::proc::pipe(reader).map_err(|e| e.to_string())?);
        let read_only = context.shared.read_only;
        if read_only { super::readonly::check()?; }
        let mut host = super::plugin_host::command(path, read_only).map_err(|e| e.to_string())?;
        let command = host.command.as_mut().unwrap();
        command.current_dir(&context.cwd).env(crate::config::AGENT_DIR_ENV, context.shared.bin_dir.parent().unwrap_or(&context.cwd))
            .env(super::subagent::SESSION_ENV, &context.session_id).env("PICCOSTA_PROVIDER", &context.provider)
            .env("PICCOSTA_MODEL", &context.model_id).env("PICCOSTA_REASONING_LEVEL", &context.thinking_level)
            // Protocol errors belong in answer content, not on top of the terminal UI.
            .stdin(Stdio::piped()).stdout(writer).stderr(Stdio::null());
        command.env_remove("PICCOSTA_SESSION_FILE");
        if let Some(file) = &context.session_file { command.env("PICCOSTA_SESSION_FILE", file); }
        crate::proc::shell_setup(command, false);
        let mut child = tokio::process::Command::from(host.command.take().unwrap()).spawn().map_err(|e| e.to_string())?;
        let pid = child.id().ok_or("process has no pid")?;
        let owner = match host.attach(pid) { Ok(owner) => owner, Err(error) => { let _ = child.start_kill(); return Err(error.to_string()); } };
        let stdin = child.stdin.take().expect("piped plugin stdin");
        context.shared.jobs.shells().push(super::jobs::Shell { pid, own: false, plugin: Some(owner.clone()), kill_at: None, pipe: stdout.clone(), handed: false });
        let lifetime = owner.clone();
        let shared = context.shared.clone();
        tokio::spawn(async move {
            let _ = child.wait().await;
            // Helpers must not outlive even a plugin that crashes. Check identity before touching
            // a group whose leader has just been reaped, as the main watcher does.
            owner.kill();
            shared.jobs.shells().retain(|shell| shell.pid != pid);
            #[cfg(unix)]
            super::reaper::track_plugin(pid, owner.start, false);
            crate::proc::release(pid);
        });
        Ok(Process { _lifetime: lifetime, stdin, stdout, buffered: Vec::new() })
    }

    async fn send(&mut self, value: Value) -> Result<(), String> {
        let line = format!("{value}\n");
        self.stdin.write_all(line.as_bytes()).await.map_err(|e| format!("stdin: {e}"))
    }

    async fn answer(&mut self, id: &str, limits: super::image::Limits, deadline: tokio::time::Instant, cancel: &tokio_util::sync::CancellationToken) -> Result<ToolOutput, String> {
        let mut scanned = 0;
        loop {
            if let Some(offset) = self.buffered[scanned..].iter().position(|byte| *byte == b'\n') {
                let newline = scanned + offset;
                if newline >= MAX_LINE_BYTES { return Err("answer is too large".into()); }
                let line: Vec<_> = self.buffered.drain(..=newline).collect();
                let id = id.to_string();
                // Large JSON/base64 and image conversion all belong off the session thread.
                // One codec can't be interrupted, but a dropped future must stop the worker
                // before its next block. Bound the decoded pixels across the whole response.
                let stopped = cancel.child_token();
                let _stop_on_drop = stopped.clone().drop_guard();
                return tokio::task::spawn_blocking(move || {
                    #[derive(Deserialize)]
                    struct Answer { call: String, content: Vec<Content>, #[serde(rename = "isError")] is_error: bool }
                    let answer: Answer = serde_json::from_slice(&line).map_err(|e| format!("bad answer: {e}"))?;
                    if answer.call != id || answer.content.iter().any(|c| !matches!(c, Content::Text { .. } | Content::Image { .. })) {
                        return Err("bad answer: expected this call's id and text or image blocks".into());
                    }
                    let mut content = Vec::new();
                    let mut pixels_left = super::image::MAX_PLUGIN_RESPONSE_PIXELS;
                    for block in answer.content {
                        if stopped.is_cancelled() || tokio::time::Instant::now() >= deadline {
                            return Err("answer conversion stopped at cancellation or deadline".into());
                        }
                        match block {
                            Content::Image { data, mime_type } => {
                                let bytes = base64::engine::general_purpose::STANDARD.decode(data).map_err(|e| format!("bad image base64: {e}"))?;
                                if super::image::sniff_mime_type(&bytes) != Some(mime_type.as_str()) {
                                    return Err("bad image MIME: expected a known image type matching its bytes".to_string());
                                }
                                let prepared = super::image::prepare_plugin(&bytes, limits, &mut pixels_left)?;
                                content.extend(prepared.hints.into_iter().map(Content::text));
                                content.push(Content::Image { data: prepared.data, mime_type: prepared.mime_type });
                            }
                            text => content.push(text),
                        }
                    }
                    Ok(ToolOutput { content, details: None, is_error: answer.is_error })
                }).await.map_err(|e| format!("answer conversion: {e}"))?;
            }
            if self.buffered.len() >= MAX_LINE_BYTES { return Err("answer is too large".into()); }
            scanned = self.buffered.len();
            // A ready pipe may never yield on its own. Keep buffered bursts cooperative too,
            // and scan each byte once rather than rescanning a base64 image on every read.
            tokio::task::yield_now().await;
            let mut bytes = [0; 8192];
            let count = crate::proc::read_pipe(&self.stdout, &mut bytes).await.map_err(|e| format!("stdout: {e}"))?;
            if count == 0 { return Err("exited mid-call (stdout closed)".into()); }
            self.buffered.extend_from_slice(&bytes[..count]);
        }
    }
}

impl Plugin {
    async fn call(&self, tool: &str, arguments: Map<String, Value>, approved: bool, context: &ToolContext) -> Result<ToolOutput, String> {
        // One deadline includes the queue, startup and stdin, not just reads. Reset while still
        // holding the lock: dropping it first lets a queued call take the broken process back.
        let deadline = tokio::time::Instant::now().checked_add(context.shared.plugin_call_wait).ok_or("call timeout cannot be represented")?;
        let mut held = tokio::select! {
            held = tokio::time::timeout_at(deadline, self.process.lock()) => held.map_err(|_| "call timed out waiting for plugin")?,
            () = context.cancel.cancelled() => return Err("Operation aborted".into()),
        };
        if context.cancel.is_cancelled() { return Err("Operation aborted".into()); }
        if held.is_none() { *held = Some(Process::spawn(&self.path, context)?); }
        let process = held.as_mut().unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        let result = async {
            process.send(json!({"call": id, "tool": tool, "arguments": arguments, "cwd": context.cwd, "approved": approved})).await?;
            process.answer(&id, context.image_limits, deadline, &context.cancel).await
        };
        let result = tokio::select! {
            result = tokio::time::timeout_at(deadline, result) => result.unwrap_or_else(|_| Err("call timed out; plugin process reset".into())),
            () = context.cancel.cancelled() => {
                let acknowledged = tokio::time::timeout(CANCEL_WAIT, async {
                    process.send(json!({"cancel": id})).await?;
                    process.answer(&id, context.image_limits, tokio::time::Instant::now() + CANCEL_WAIT, &tokio_util::sync::CancellationToken::new()).await
                }).await;
                if !matches!(acknowledged, Ok(Ok(_))) { held.take(); }
                return Err("Operation aborted".into());
            }
        };
        if result.is_err() { held.take(); }
        result
    }
}
