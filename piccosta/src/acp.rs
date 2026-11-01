// ACP, the Agent Client Protocol (agentclientprotocol.com, version 1), so Zed, JetBrains, Neovim,
// Emacs and the other ACP clients can run piccosta: JSON-RPC 2.0, one message per line on stdin
// and stdout. Written from the v1 schema and from what Zed's client does (the contract, cited to
// the spec, the schema and Zed's source, was read out on 2026-10-29), not on an SDK, so a bug in
// it is ours to fix in Rust.
//
// What a client gets:
// - Sessions: new, load (the conversation replayed), list, close, several at once on one
//   connection. Each is a piccosta session file under its own cwd, written as it goes, so a thread
//   survives Zed killing the process (it SIGKILLs) and opens in the TUI too.
// - Prompts: text, images, @-mentioned files (embedded by the client, or linked and read here).
//   The reply streams as thought and message chunks; each tool call as tool_call then
//   tool_call_update, with its kind, path, output and, for edit and write, the file before and
//   after; usage_update after each response.
// - Config options: model (those with a login), thinking level, and approval mode (ask before
//   anything that isn't read-only, let edits through, or ask for nothing). A permission request
//   names the tool call already on screen; "always" is remembered for that tool in that session.
// - Commands /compact, /init and /review, as in the TUI.
// Not used: the client's file system and terminals (piccosta's own tools do that work), auth (the
// model logins are piccosta's own) and MCP servers (piccosta has no MCP client: they are ignored,
// and the client is told when it takes notices).
//
// The rules Zed holds an agent to beyond the spec, each followed below: a cancelled prompt is
// always answered, with stopReason "cancelled"; a tool call is announced before any update to it,
// and every open one is finished before a prompt is answered; model, thinking and mode all go in
// configOptions, load's answer included; nothing but protocol lines on stdout.

use crate::agent::{Event, EventSink, Gate, Handle};
use crate::host::{Command, Done, Host};
use crate::message::{AgentMessage, Content, StopReason, TextOrBlocks};
use crate::model::Model;
use crate::rpc::write_line;
use crate::session::Session;
use serde_json::{json, Map, Value};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::oneshot;

// The protocol version spoken. v2 is a draft no client uses by default; one that asks for it gets
// 1, the documented fallback.
const PROTOCOL_VERSION: u64 = 1;
// How long a cancelled prompt waits for its run to stop before it is answered anyway: Zed waits for
// that answer with no timeout, so an unanswered one wedges the thread.
const CANCEL_WAIT: Duration = Duration::from_secs(5);
// How long the sessions get to stop their runs at exit.
const STOP_WAIT: Duration = Duration::from_secs(5);
// The most of a tool's output a tool call shows; the model still gets all of it.
const TOOL_OUTPUT_CHARS: usize = 8000;
// The largest file a prompt's link is read in for, and the largest an edit's before/after is
// shown for; past that the model gets the path, the client the plain output.
const FILE_MAX_BYTES: u64 = 1024 * 1024;
// Tools that only read, which no mode asks about.
const READ_ONLY: [&str; 4] = ["read", "web_search", "fetch_content", "get_search_content"];
// JSON-RPC error codes (ACP v1 ErrorCode).
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;
const NOT_FOUND: i64 = -32002;

// When a tool call needs the user's yes.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Ask,
    Edits,
    Auto,
}

// Each mode's id, name and description, as the client shows them.
const MODES: [(Mode, &str, &str, &str); 3] = [
    (Mode::Ask, "ask", "Ask", "Ask before commands, edits and anything else that isn't read-only"),
    (Mode::Edits, "edits", "Edits run", "Edits and writes run; ask before commands"),
    (Mode::Auto, "auto", "Auto", "Run every tool without asking"),
];

fn needs_approval(mode: Mode, tool: &str) -> bool {
    match mode {
        Mode::Auto => false,
        Mode::Edits => !READ_ONLY.contains(&tool) && !matches!(tool, "edit" | "write"),
        Mode::Ask => !READ_ONLY.contains(&tool),
    }
}

// The user's answer to a permission request.
enum Answer {
    Once,
    Always,
    No,
}

enum Input {
    Line(String),
    Closed,
    // From a session's task, keyed by its session id and which start of it this was.
    Event(String, u64, Box<Event>),
    Done(String, u64, Box<Done>),
    // A tool call waiting for the user's yes.
    Ask { session: String, serial: u64, call_id: String, reply: oneshot::Sender<Answer> },
    // A cancelled prompt's time to stop is up; the number is which prompt of the session it was.
    CancelDeadline(String, u64, u64),
}

// A tool call announced and not finished: its name, arguments, the file it works on, and (for edit
// and write) that file's text before it ran, None when there was no such file.
struct OpenCall {
    name: String,
    arguments: Map<String, Value>,
    path: Option<PathBuf>,
    before: Option<Option<String>>,
    // A shell command's output already written to its client terminal.
    shown: String,
}

// Count commands where they execute, so a delayed start event cannot abort a newer run.
#[derive(Default)]
struct Runs {
    handle: Handle,
    sent: u64,
    finished: u64,
    cancelled_through: u64,
}

// One session being served, and what this loop knows of it while its task has the agent.
struct Live {
    serial: u64,
    host: Arc<Host>,
    shared: Arc<crate::tools::ToolShared>,
    handle: Handle,
    runs: Arc<Mutex<Runs>>,
    commands: UnboundedSender<Command>,
    task: tokio::task::JoinHandle<()>,
    model: Model,
    thinking: String,
    mode: Rc<Cell<Mode>>,
    always: Rc<RefCell<HashSet<String>>>,
    // One per command sent to the task, oldest first: the prompt request it answers, if it is one.
    owed: VecDeque<Option<Value>>,
    // Commands whose events are no longer part of a client prompt, through a cancelled deadline.
    muted: usize,
    // The prompt being answered, whether it was cancelled, and how many prompts came before it.
    prompt: Option<Value>,
    cancelled: bool,
    prompts: u64,
    // How the last response ended, for the prompt's answer.
    stop: Option<StopReason>,
    error: Option<String>,
    // The assistant message streaming in, numbered for its messageId.
    message: u64,
    open: HashMap<String, OpenCall>,
    titled: bool,
    // The context the last response filled and the session's cost so far, for usage_update
    // (ACP's `used` is tokens in context now, its `cost` the whole session's).
    used: u64,
    cost: f64,
}

// Tell the client how full the context is and what the session has cost.
fn usage_update(session: &str, live: &Live) {
    update(session, json!({"sessionUpdate": "usage_update", "used": live.used, "size": live.model.context_window, "cost": {"amount": live.cost, "currency": "USD"}}));
}

struct Acp {
    args: crate::cli::Args,
    inputs: UnboundedSender<Input>,
    sessions: HashMap<String, Live>,
    hosts: HashMap<PathBuf, Arc<Host>>,
    initialized: bool,
    // Whether the client takes `notice` updates.
    notices: bool,
    // Whether the client shows terminals the agent runs (Zed's `_meta.terminal_output`). Zed shows
    // a command's output only that way, so shell output then streams into one per call.
    terminals: bool,
    // The models with a login, found once per connection.
    models: Option<Vec<Model>>,
    next_request: u64,
    next_session: u64,
    asks: HashMap<u64, (String, String, oneshot::Sender<Answer>)>,
    // Updates that go out after the response being written (a new session's commands).
    after: Vec<(String, Value)>,
}

// An error answer: code, message and data.
struct Failure(i64, String, Option<Value>);

fn invalid(message: impl Into<String>) -> Failure {
    Failure(INVALID_PARAMS, message.into(), None)
}

fn unknown_session(id: &str) -> Failure {
    Failure(NOT_FOUND, "Resource not found".into(), Some(json!({"sessionId": id})))
}

fn reply(id: Value, result: Value) {
    write_line(&json!({"jsonrpc": "2.0", "id": id, "result": result}));
}

fn fail(id: Value, Failure(code, message, data): Failure) {
    let mut error = json!({"code": code, "message": message});
    if let Some(data) = data {
        error["data"] = data;
    }
    write_line(&json!({"jsonrpc": "2.0", "id": id, "error": error}));
}

fn update(session: &str, update: Value) {
    write_line(&json!({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session, "update": update}}));
}

fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

pub async fn run(args: crate::cli::Args) -> Result<ExitCode, String> {
    tokio::task::LocalSet::new().run_until(serve(args)).await
}

async fn serve(args: crate::cli::Args) -> Result<ExitCode, String> {
    let (inputs, mut received) = unbounded_channel();
    // stdin on a plain thread, as rpc mode reads it.
    let lines = inputs.clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else { break };
            if lines.send(Input::Line(line)).is_err() {
                return;
            }
        }
        let _ = lines.send(Input::Closed);
    });
    let mut acp = Acp { args, inputs, sessions: HashMap::new(), hosts: HashMap::new(), initialized: false, notices: false, terminals: false, models: None, next_request: 0, next_session: 0, asks: HashMap::new(), after: Vec::new() };
    use tokio::signal::unix::{signal, SignalKind};
    let (mut term, mut hangup) = (signal(SignalKind::terminate()).map_err(|error| error.to_string())?, signal(SignalKind::hangup()).map_err(|error| error.to_string())?);
    let code = loop {
        let input = tokio::select! {
            input = received.recv() => input,
            _ = term.recv() => break 143,
            _ = hangup.recv() => break 129,
        };
        match input {
            Some(Input::Line(line)) => acp.line(&line).await,
            Some(Input::Event(session, serial, event)) if acp.current(&session, serial) => acp.event(&session, *event),
            Some(Input::Done(session, serial, done)) if acp.current(&session, serial) => acp.done(&session, *done),
            Some(Input::Ask { session, serial, call_id, reply }) if acp.current(&session, serial) => acp.ask(session, call_id, reply),
            Some(Input::CancelDeadline(session, serial, prompt)) if acp.current(&session, serial) => acp.cancel_deadline(&session, prompt),
            Some(Input::Closed) | None => break 0,
            _ => {}
        }
    };
    let tasks: Vec<_> = acp.sessions.drain().map(|(_, live)| {
        live.handle.abort();
        live.task
    }).collect();
    let _ = tokio::time::timeout(STOP_WAIT, futures_util::future::join_all(tasks)).await;
    crate::provider::claude::shutdown().await;
    crate::tools::bash::kill_all_running();
    Ok(ExitCode::from(code))
}

impl Acp {
    fn current(&self, session: &str, serial: u64) -> bool {
        self.sessions.get(session).is_some_and(|live| live.serial == serial)
    }

    async fn line(&mut self, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        let message: Value = match serde_json::from_str(line) {
            Ok(message) => message,
            Err(error) => return fail(Value::Null, Failure(PARSE_ERROR, format!("Parse error: {error}"), None)),
        };
        if !message.is_object() || message.get("jsonrpc").and_then(Value::as_str) != Some("2.0")
            || message.get("id").is_some_and(|id| !id.is_null() && !id.is_string() && !id.is_number())
            || message.get("method").is_some_and(|method| !method.is_string())
            || (message.get("method").is_none() && (message.get("id").is_none() || (message.get("result").is_none() && message.get("error").is_none())))
        {
            return fail(Value::Null, Failure(INVALID_REQUEST, "Invalid Request".into(), None));
        }
        let params = message.get("params").cloned().unwrap_or_else(|| json!({}));
        match (message.get("method").and_then(Value::as_str), message.get("id").cloned()) {
            (Some(method), Some(id)) => {
                if !message.get("params").is_some_and(Value::is_object) {
                    return fail(id, invalid("params is missing or is not an object"));
                }
                match self.request(method, &params, id.clone()).await {
                    Ok(Some(result)) => reply(id, result),
                    Ok(None) => {}
                    Err(failure) => fail(id, failure),
                }
                for (session, after) in std::mem::take(&mut self.after) {
                    update(&session, after);
                }
            }
            (Some(method), None) => self.notification(method, &params),
            // A response to one of piccosta's requests; one with a null id answers nothing.
            (None, Some(id)) => self.answered(&id, &message),
            (None, None) => {}
        }
    }

    // A request. Output: its result, or None when it is answered later (a prompt).
    async fn request(&mut self, method: &str, params: &Value, id: Value) -> Result<Option<Value>, Failure> {
        if method != "initialize" && !self.initialized {
            return Err(Failure(INVALID_REQUEST, "initialize first".into(), None));
        }
        let session_id = params.get("sessionId").and_then(Value::as_str).unwrap_or("").to_string();
        match method {
            "initialize" => {
                self.initialized = true;
                self.notices = params.pointer("/clientCapabilities/session/notices").is_some_and(Value::is_object);
                self.terminals = params.pointer("/clientCapabilities/_meta/terminal_output") == Some(&Value::Bool(true));
                Ok(Some(json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "agentCapabilities": {
                        "loadSession": true,
                        "promptCapabilities": {"image": true, "audio": false, "embeddedContext": true},
                        "mcpCapabilities": {"http": false, "sse": false},
                        "sessionCapabilities": {"list": {}, "close": {}}
                    },
                    "agentInfo": {"name": "piccosta", "title": "piccosta", "version": env!("CARGO_PKG_VERSION")},
                    "authMethods": []
                })))
            }
            "session/new" => {
                let host = self.host(params)?;
                let session = host.session(None, false, true).map_err(|error| Failure(INTERNAL_ERROR, error, None))?;
                let (model, thinking) = host.launch(&self.args).map_err(|error| Failure(INTERNAL_ERROR, error, None))?;
                let id = self.start(host.clone(), session, model.as_deref(), thinking.as_deref())?;
                let state = self.state(&id).await;
                self.after.push((id.clone(), commands()));
                self.session_notices(params, &id, &host);
                Ok(Some(json!({"sessionId": id, "modes": state.0, "configOptions": state.1})))
            }
            "session/load" => {
                let host = self.host(params)?;
                let file = host.sessions(usize::MAX).into_iter().find(|summary| summary.id == session_id).map(|summary| summary.path).ok_or_else(|| unknown_session(&session_id))?;
                let session = Session::open(&file, &host.cwd).map_err(|error| Failure(INTERNAL_ERROR, error.to_string(), None))?;
                if Path::new(&session.header().cwd) != host.cwd {
                    return Err(invalid("cwd does not match the session"));
                }
                replay(&session_id, &session, &host.cwd, self.terminals);
                if !self.sessions.contains_key(&session_id) {
                    self.start(host.clone(), session, None, None)?;
                }
                let state = self.state(&session_id).await;
                self.after.push((session_id.clone(), commands()));
                self.session_notices(params, &session_id, &host);
                Ok(Some(json!({"modes": state.0, "configOptions": state.1})))
            }
            "session/list" => {
                let mut listed = if params.get("cwd").and_then(Value::as_str).is_some() {
                    self.host(params)?.sessions(usize::MAX)
                } else if let Some(dir) = &self.args.session_dir {
                    crate::host::sessions_in(Path::new(dir), None, usize::MAX)
                } else {
                    std::fs::read_dir(crate::config::agent_dir().join("sessions")).into_iter().flatten().flatten()
                        .flat_map(|entry| crate::host::sessions_in(&entry.path(), None, usize::MAX)).collect()
                };
                listed.sort_by_key(|summary| std::cmp::Reverse(summary.modified));
                let sessions: Vec<Value> = listed
                    .into_iter()
                    .map(|summary| {
                        let updated = summary.modified.duration_since(std::time::UNIX_EPOCH).map(|since| crate::clock::iso_from_ms(since.as_millis() as i64)).ok();
                        json!({"sessionId": summary.id, "cwd": summary.cwd, "title": summary.title, "updatedAt": updated})
                    })
                    .collect();
                Ok(Some(json!({"sessions": sessions})))
            }
            "session/close" => {
                self.asks.retain(|_, (session, ..)| session != &session_id);
                if let Some(mut live) = self.sessions.remove(&session_id) {
                    live.handle.abort();
                    drop(live.commands);
                    if tokio::time::timeout(STOP_WAIT, &mut live.task).await.is_err() {
                        live.task.abort();
                        let _ = live.task.await;
                        live.shared.jobs.shutdown("quit");
                        live.shared.web.shutdown();
                    }
                    if let Some(prompt) = live.prompt {
                        for call_id in live.open.keys() {
                            update(&session_id, json!({"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": "failed", "content": [{"type": "content", "content": text_block("Cancelled.")}]}));
                        }
                        reply(prompt, json!({"stopReason": "cancelled"}));
                    }
                }
                Ok(Some(json!({})))
            }
            "session/prompt" => {
                let blocks = params.get("prompt").and_then(Value::as_array).ok_or_else(|| invalid("prompt is missing or is not an array"))?;
                let live = self.sessions.get_mut(&session_id).ok_or_else(|| unknown_session(&session_id))?;
                if live.prompt.is_some() {
                    return Err(Failure(INVALID_REQUEST, "a prompt is already running in this session".into(), None));
                }
                let (text, images) = prompt_input(blocks, &live.host.cwd);
                let trimmed = text.trim();
                let (name, argument) = trimmed.strip_prefix('/').map(|line| line.split_once(' ').map_or((line, ""), |(name, rest)| (name, rest.trim()))).unwrap_or(("", ""));
                let command = match name {
                    "compact" => Command::Compact(Some(argument.to_string()).filter(|text| !text.is_empty())),
                    _ => Command::Prompt(crate::prompt::command_prompt(&text).unwrap_or_else(|| text.clone()), images),
                };
                if !live.titled {
                    live.titled = true;
                    let title: String = trimmed.lines().next().unwrap_or("").chars().take(crate::host::TITLE_CHARS).collect();
                    update(&session_id, json!({"sessionUpdate": "session_info_update", "title": title, "updatedAt": crate::clock::iso_from_ms(crate::clock::now_ms())}));
                }
                if live.commands.send(command).is_err() {
                    return Err(Failure(INTERNAL_ERROR, "the session stopped".into(), None));
                }
                live.runs.lock().unwrap().sent += 1;
                live.owed.push_back(Some(id.clone()));
                live.prompt = Some(id);
                Ok(None)
            }
            "session/set_config_option" => {
                let value = params.get("value").and_then(Value::as_str).unwrap_or("").to_string();
                match params.get("configId").and_then(Value::as_str).unwrap_or("") {
                    "mode" => self.set_mode(&session_id, &value)?,
                    "model" => self.set_model(&session_id, &value).await?,
                    "thinking" => self.set_thinking(&session_id, &value)?,
                    other => return Err(invalid(format!("no config option {other}"))),
                }
                Ok(Some(json!({"configOptions": self.state(&session_id).await.1})))
            }
            "session/set_mode" => {
                let mode = params.get("modeId").and_then(Value::as_str).unwrap_or("").to_string();
                self.set_mode(&session_id, &mode)?;
                self.after.push((session_id, json!({"sessionUpdate": "current_mode_update", "currentModeId": mode})));
                Ok(Some(json!({})))
            }
            _ => Err(Failure(METHOD_NOT_FOUND, format!("Method not found: {method}"), None)),
        }
    }

    fn notification(&mut self, method: &str, params: &Value) {
        // Cancel: stop the run, and answer the prompt when it has stopped, or after CANCEL_WAIT if
        // it hasn't.
        if method == "session/cancel"
            && let Some(session) = params.get("sessionId").and_then(Value::as_str)
            && let Some(live) = self.sessions.get_mut(session)
            && live.prompt.is_some()
        {
            live.cancelled = true;
            {
                let mut runs = live.runs.lock().unwrap();
                runs.cancelled_through = runs.sent;
            }
            live.handle.abort();
            let (inputs, session, serial, prompt) = (self.inputs.clone(), session.to_string(), live.serial, live.prompts);
            tokio::task::spawn_local(async move {
                tokio::time::sleep(CANCEL_WAIT).await;
                let _ = inputs.send(Input::CancelDeadline(session, serial, prompt));
            });
        }
    }

    // The client's answer to a permission request: allow once or always, else no.
    fn answered(&mut self, id: &Value, message: &Value) {
        let Some((session, call_id, reply)) = id.as_u64().and_then(|id| self.asks.remove(&id)) else { return };
        if !self.sessions.get(&session).is_some_and(|live| live.open.contains_key(&call_id)) {
            return;
        }
        let answer = match (message.pointer("/result/outcome/outcome").and_then(Value::as_str), message.pointer("/result/outcome/optionId").and_then(Value::as_str)) {
            (Some("selected"), Some("allow-once")) => Answer::Once,
            (Some("selected"), Some("allow-always")) => Answer::Always,
            _ => Answer::No,
        };
        if !matches!(answer, Answer::No) {
            update(&session, json!({"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": "in_progress"}));
        }
        let _ = reply.send(answer);
    }

    // Ask the client about a tool call it already shows.
    fn ask(&mut self, session: String, call_id: String, reply: oneshot::Sender<Answer>) {
        let Some(call) = self.sessions.get(&session).and_then(|live| live.open.get(&call_id)) else {
            let _ = reply.send(Answer::No);
            return;
        };
        let id = self.next_request;
        self.next_request += 1;
        let options = json!([
            {"optionId": "allow-once", "name": "Allow", "kind": "allow_once"},
            {"optionId": "allow-always", "name": format!("Always allow {}", call.name), "kind": "allow_always"},
            {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"}
        ]);
        let tool_call = json!({"toolCallId": call_id, "title": title(&call.name, &call.arguments), "kind": kind(&call.name)});
        write_line(&json!({"jsonrpc": "2.0", "id": id, "method": "session/request_permission", "params": {"sessionId": session, "toolCall": tool_call, "options": options}}));
        self.asks.insert(id, (session, call_id, reply));
    }

    fn event(&mut self, session: &str, event: Event) {
        let Some(live) = self.sessions.get_mut(session) else { return };
        if live.muted > 0 {
            return;
        }
        match event {
            Event::MessageStart { message: AgentMessage::Assistant(_), .. } => live.message += 1,
            Event::MessageUpdate { assistant_message_event, .. } => {
                let kind = match assistant_message_event.get("type").and_then(Value::as_str) {
                    Some("text_delta") => "agent_message_chunk",
                    Some("thinking_delta") => "agent_thought_chunk",
                    _ => return,
                };
                let delta = assistant_message_event.get("delta").and_then(Value::as_str).unwrap_or("");
                if !delta.is_empty() {
                    update(session, json!({"sessionUpdate": kind, "messageId": format!("live-{}", live.message), "content": text_block(delta)}));
                }
            }
            Event::MessageEnd { message: AgentMessage::Assistant(assistant), .. } => {
                let usage = &assistant.usage;
                let used = usage.input + usage.output + usage.cache_read + usage.cache_write;
                live.cost += usage.cost.total;
                if used > 0 {
                    live.used = used;
                    usage_update(session, live);
                }
                live.error = assistant.error_message.clone();
                live.stop = Some(assistant.stop_reason);
            }
            Event::ToolExecutionStart { tool_call_id, tool_name, args } => {
                let path = args.get("path").and_then(Value::as_str).map(|path| crate::tools::paths::resolve(path, &live.host.cwd));
                let before = path.as_deref().filter(|_| matches!(tool_name.as_str(), "edit" | "write")).and_then(|path| {
                    match (read_small(path), path.try_exists()) {
                        (Some(text), _) => Some(Some(text)),
                        (None, Ok(false)) => Some(None),
                        _ => None,
                    }
                });
                let waits = needs_approval(live.mode.get(), &tool_name) && !live.always.borrow().contains(&tool_name);
                let mut call = json!({
                    "sessionUpdate": "tool_call", "toolCallId": tool_call_id, "title": title(&tool_name, &args), "name": tool_name,
                    "kind": kind(&tool_name), "status": if waits { "pending" } else { "in_progress" }, "rawInput": Value::Object(args.clone())
                });
                if let Some(path) = &path {
                    call["locations"] = json!([{"path": path}]);
                }
                // The change it is about to make, for a permission card to show.
                if let (Some(path), Some(before)) = (&path, &before)
                    && let Some(after) = planned(&tool_name, &args, before.as_deref())
                {
                    call["content"] = json!([{"type": "diff", "path": path, "oldText": before, "newText": after}]);
                }
                if self.terminals && tool_name == "bash" {
                    call["_meta"] = json!({"terminal_info": {"terminal_id": tool_call_id, "cwd": live.host.cwd}});
                    call["content"] = json!([{"type": "terminal", "terminalId": tool_call_id}]);
                }
                update(session, call);
                live.open.insert(tool_call_id, OpenCall { name: tool_name, arguments: args, path, before, shown: String::new() });
            }
            Event::ToolExecutionUpdate { tool_call_id, partial_result, .. } if live.open.contains_key(&tool_call_id) => {
                let text = output_text(&partial_result);
                let mut message = json!({"sessionUpdate": "tool_call_update", "toolCallId": tool_call_id, "status": "in_progress"});
                match live.open.get_mut(&tool_call_id).filter(|call| self.terminals && call.name == "bash") {
                    Some(call) => message["_meta"] = json!({"terminal_output": {"terminal_id": tool_call_id, "data": terminal_bytes(&mut call.shown, &text)}}),
                    None => message["content"] = json!([{"type": "content", "content": text_block(&text)}]),
                }
                update(session, message);
            }
            Event::ToolExecutionEnd { tool_call_id, result, is_error, .. } => {
                self.asks.retain(|_, (asked_session, call_id, _)| asked_session != session || call_id != &tool_call_id);
                let call = live.open.remove(&tool_call_id);
                let after = call.as_ref().filter(|call| !is_error && call.before.is_some()).and_then(|call| call.path.as_deref()).and_then(read_small);
                let content = match (call.as_ref(), after) {
                    (Some(OpenCall { path: Some(path), before: Some(before), .. }), Some(after)) => json!([{"type": "diff", "path": path, "oldText": before, "newText": after}]),
                    _ => json!([{"type": "content", "content": text_block(&output_text(&result))}]),
                };
                let mut message = json!({"sessionUpdate": "tool_call_update", "toolCallId": tool_call_id, "status": if is_error { "failed" } else { "completed" }, "content": content});
                if let Some(mut call) = call.filter(|call| self.terminals && call.name == "bash") {
                    message["content"] = json!([{"type": "terminal", "terminalId": tool_call_id}]);
                    message["_meta"] = json!({
                        "terminal_output": {"terminal_id": tool_call_id, "data": terminal_bytes(&mut call.shown, &output_text(&result))},
                        "terminal_exit": {"terminal_id": tool_call_id}
                    });
                }
                update(session, message);
            }
            // Without the client's compaction support, a compaction is told as a message.
            // After a compaction the context is the summary's size.
            Event::CompactionEnd { result: Some(result), .. } => {
                live.used = result.get("estimatedTokensAfter").and_then(Value::as_u64).unwrap_or(live.used);
                usage_update(session, live);
                let before = result.get("tokensBefore").and_then(Value::as_u64).unwrap_or(0);
                live.message += 1;
                let text = format!("Compacted the conversation: {before} → about {} tokens.", live.used);
                update(session, json!({"sessionUpdate": "agent_message_chunk", "messageId": format!("live-{}", live.message), "content": text_block(&text)}));
            }
            Event::CompactionEnd { error_message, aborted, .. } if !aborted => {
                let text = error_message.unwrap_or_else(|| "Compacted the conversation.".into());
                live.message += 1;
                update(session, json!({"sessionUpdate": "agent_message_chunk", "messageId": format!("live-{}", live.message), "content": text_block(&text)}));
            }
            _ => {}
        }
    }

    fn done(&mut self, session: &str, done: Done) {
        let Some(live) = self.sessions.get_mut(session) else { return };
        live.model = done.model;
        live.thinking = done.thinking;
        let prompt = live.owed.pop_front().flatten();
        if live.muted > 0 {
            live.muted -= 1;
        } else if let Some(prompt) = prompt {
            self.finish(session, prompt, done.result);
        }
    }

    // Answer a prompt: every tool call still open ends failed first, then the stop reason, or an
    // error when the model's response failed.
    fn finish(&mut self, session: &str, prompt: Value, result: Result<(), String>) {
        let Some(live) = self.sessions.get_mut(session) else { return };
        if live.prompt.as_ref() != Some(&prompt) {
            return;
        }
        live.prompt = None;
        self.asks.retain(|_, (asked_session, ..)| asked_session != session);
        live.prompts += 1;
        for (call_id, _) in live.open.drain() {
            update(session, json!({"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": "failed", "content": [{"type": "content", "content": text_block("Cancelled.")}]}));
        }
        let (cancelled, stop, error) = (std::mem::take(&mut live.cancelled), live.stop.take(), live.error.take());
        match (cancelled || stop == Some(StopReason::Aborted), result, stop) {
            (true, _, _) => reply(prompt, json!({"stopReason": "cancelled"})),
            (false, Err(error), _) => fail(prompt, Failure(INTERNAL_ERROR, error, None)),
            (false, Ok(()), Some(StopReason::Error)) => fail(prompt, Failure(INTERNAL_ERROR, error.unwrap_or_else(|| "the model's response failed".into()), None)),
            (false, Ok(()), Some(StopReason::Length)) => reply(prompt, json!({"stopReason": "max_tokens"})),
            (false, Ok(()), _) => reply(prompt, json!({"stopReason": "end_turn"})),
        }
    }

    // A cancelled prompt that still hasn't stopped is answered now.
    fn cancel_deadline(&mut self, session: &str, prompt: u64) {
        let Some(live) = self.sessions.get_mut(session) else { return };
        if live.prompts == prompt && live.cancelled && let Some(id) = live.prompt.clone() {
            // Reports stay in command order. Drain this run before accepting a later command's
            // events or stop reason, even if the client has already sent another prompt.
            live.muted = live.owed.len();
            self.finish(session, id, Ok(()));
        }
    }

    // The host for a request's cwd. The spec says it is absolute; a relative one is taken from
    // piccosta's own directory, the one the client started it in (the SDK's own yopo client sends
    // ".", rust-sdk src/yopo/src/lib.rs:114).
    fn host(&mut self, params: &Value) -> Result<Arc<Host>, Failure> {
        let cwd = params.get("cwd").and_then(Value::as_str).filter(|cwd| !cwd.is_empty()).ok_or_else(|| invalid("cwd is missing"))?;
        // components() drops the "." parts, so "." and the absolute path name the same session dir.
        let cwd: PathBuf = std::env::current_dir().map(|here| here.join(cwd)).unwrap_or_else(|_| PathBuf::from(cwd)).components().collect();
        let args = &self.args;
        Ok(self.hosts.entry(cwd.clone()).or_insert_with(|| {
            let host = Host::for_args_at(args, cwd);
            host.show_settings_errors();
            Arc::new(host)
        }).clone())
    }

    // Settings errors and unconnected MCP servers go to the user when the client takes notices.
    fn session_notices(&mut self, params: &Value, session: &str, host: &Host) {
        if self.notices {
            for error in &host.settings.load_errors {
                self.after.push((session.to_string(), json!({"sessionUpdate": "notice", "severity": "warning", "title": "Invalid settings file", "description": error})));
            }
        }
        let count = params.get("mcpServers").and_then(Value::as_array).map_or(0, Vec::len);
        if count == 0 {
            return;
        }
        eprintln!("piccosta: {count} MCP server(s) left unconnected (piccosta has no MCP client)");
        if self.notices {
            self.after.push((session.to_string(), json!({"sessionUpdate": "notice", "severity": "warning", "title": "MCP servers are not connected", "description": format!("piccosta has no MCP client; the {count} server(s) configured here are left unconnected.")})));
        }
    }

    // Start serving a session: its agent on a task whose events and reports come back as inputs.
    fn start(&mut self, host: Arc<Host>, session: Session, model: Option<&str>, thinking: Option<&str>) -> Result<String, Failure> {
        let id = session.id().to_string();
        let serial = self.next_session;
        self.next_session += 1;
        let events = (self.inputs.clone(), id.clone());
        let runs = Arc::new(Mutex::new(Runs::default()));
        let starts = runs.clone();
        let emit: EventSink = Box::new(move |event: &Event| {
            if matches!(event, Event::AgentStart | Event::CompactionStart { .. }) {
                let runs = starts.lock().unwrap();
                if runs.finished < runs.cancelled_through {
                    runs.handle.abort();
                }
            }
            let _ = events.0.send(Input::Event(events.1.clone(), serial, Box::new(event.clone())));
        });
        // A prompt stays open until the background jobs it started end, so a job's message reaches
        // the model; Zed can't take a turn the agent starts on its own.
        let (mut agent, shared) = host.open(session, model, thinking, emit, true).map_err(|error| Failure(INTERNAL_ERROR, error, None))?;
        agent.session.persist_now().map_err(|error| Failure(INTERNAL_ERROR, format!("could not write the session: {error}"), None))?;
        let first_mode = if self.args.yolo { Mode::Auto } else { Mode::Ask };
        let (mode, always) = (Rc::new(Cell::new(first_mode)), Rc::new(RefCell::new(HashSet::new())));
        agent.set_gate(gate(self.inputs.clone(), id.clone(), serial, mode.clone(), always.clone()));
        let titled = crate::host::name(&agent.session).is_some() || agent.session.context().messages.iter().any(|message| matches!(message, AgentMessage::User(_)));
        let cost = agent.session.context().messages.iter().filter_map(|message| if let AgentMessage::Assistant(assistant) = message { Some(assistant.usage.cost.total) } else { None }).sum();
        let used = crate::compaction::estimate_projected_context_tokens(&agent.session.projection(), &agent.session.path_to_leaf()).tokens;
        let (handle, live_model, live_thinking) = (agent.handle(), agent.model().clone(), agent.thinking_level().to_string());
        runs.lock().unwrap().handle = handle.clone();
        let (commands, receiver) = unbounded_channel();
        let reports = (self.inputs.clone(), id.clone());
        let finishes = runs.clone();
        let task = host.clone().spawn(agent, shared.clone(), false, receiver, move |done| {
            finishes.lock().unwrap().finished += 1;
            let _ = reports.0.send(Input::Done(reports.1.clone(), serial, Box::new(done)));
        });
        let live = Live {
            serial, host, shared, handle, runs, commands, task, model: live_model, thinking: live_thinking, mode, always, owed: VecDeque::new(), muted: 0, prompt: None, cancelled: false,
            prompts: 0, stop: None, error: None, message: 0, open: HashMap::new(), titled, used, cost,
        };
        self.sessions.insert(id.clone(), live);
        Ok(id)
    }

    fn set_mode(&mut self, session: &str, value: &str) -> Result<(), Failure> {
        let live = self.sessions.get(session).ok_or_else(|| unknown_session(session))?;
        let (mode, ..) = MODES.iter().find(|(_, id, ..)| *id == value).ok_or_else(|| invalid(format!("no mode {value}")))?;
        live.mode.set(*mode);
        // Choosing a mode starts its asking afresh: an earlier "always allow" doesn't outlive it.
        live.always.borrow_mut().clear();
        Ok(())
    }

    // A new model, for the session's next turn.
    async fn set_model(&mut self, session: &str, value: &str) -> Result<(), Failure> {
        let model = self.usable().await.into_iter().find(|model| format!("{}/{}", model.provider, model.id) == value).ok_or_else(|| invalid(format!("no model {value} with a login")))?;
        let live = self.sessions.get_mut(session).ok_or_else(|| unknown_session(session))?;
        // A model picked in the editor is the one new sessions start on, as the terminal's /model.
        if let Some((provider, id)) = value.split_once('/')
            && let Err(error) = crate::config::remember_model(&live.host.agent_dir, provider, id)
        {
            eprintln!("piccosta: {error}");
        }
        if live.commands.send(Command::Model(value.to_string())).is_ok() {
            live.runs.lock().unwrap().sent += 1;
            live.owed.push_back(None);
            // The level the session will run at, as the task sets it: the model's own, else this one.
            let level = live.host.remembered_thinking_level(&model).unwrap_or_else(|| live.thinking.clone());
            live.model = model;
            live.thinking = crate::provider::clamp_thinking_level(&live.model, &level).to_string();
            usage_update(session, live);
        }
        Ok(())
    }

    fn set_thinking(&mut self, session: &str, value: &str) -> Result<(), Failure> {
        let live = self.sessions.get_mut(session).ok_or_else(|| unknown_session(session))?;
        if !crate::provider::thinking_levels(&live.model).contains(&value) {
            return Err(invalid(format!("{} doesn't take thinking level {value}", live.model.id)));
        }
        let model = format!("{}/{}", live.model.provider, live.model.id);
        if let Err(error) = crate::config::remember_thinking_level(&live.host.agent_dir, &model, Some(value)) {
            eprintln!("piccosta: {error}");
        }
        if live.commands.send(Command::Thinking(value.to_string())).is_ok() {
            live.runs.lock().unwrap().sent += 1;
            live.owed.push_back(None);
            live.thinking = value.to_string();
        }
        Ok(())
    }

    async fn usable(&mut self) -> Vec<Model> {
        if self.models.is_none() {
            let host = Host::load();
            self.models = Some(host.usable_models().await.unwrap_or_default());
        }
        self.models.clone().unwrap_or_default()
    }

    // A session's modes and config options (mode, model, thinking), as session/new, load and
    // set_config_option answer them.
    async fn state(&mut self, session: &str) -> (Value, Value) {
        let mut models = self.usable().await;
        let Some(live) = self.sessions.get(session) else { return (Value::Null, json!([])) };
        let current = format!("{}/{}", live.model.provider, live.model.id);
        if !models.iter().any(|model| format!("{}/{}", model.provider, model.id) == current) {
            models.insert(0, live.model.clone());
        }
        let mode = MODES.iter().find(|(mode, ..)| *mode == live.mode.get()).map_or("ask", |(_, id, ..)| *id);
        let modes: Vec<Value> = MODES.iter().map(|(_, id, name, description)| json!({"id": id, "name": name, "description": description})).collect();
        let mode_options: Vec<Value> = MODES.iter().map(|(_, id, name, description)| json!({"value": id, "name": name, "description": description})).collect();
        let model_options: Vec<Value> = models.iter().map(|model| json!({"value": format!("{}/{}", model.provider, model.id), "name": model.name})).collect();
        let thinking_options: Vec<Value> = crate::provider::thinking_levels(&live.model).into_iter().map(|level| json!({"value": level, "name": level})).collect();
        let options = json!([
            {"id": "mode", "name": "Mode", "category": "mode", "type": "select", "currentValue": mode, "options": mode_options},
            {"id": "model", "name": "Model", "category": "model", "type": "select", "currentValue": current, "options": model_options},
            {"id": "thinking", "name": "Thinking", "category": "thought_level", "type": "select", "currentValue": live.thinking, "options": thinking_options}
        ]);
        (json!({"currentModeId": mode, "availableModes": modes}), options)
    }
}

// The gate a session's tool calls pass: through at once when the mode doesn't ask about the tool
// or the user said "always" for it, else after the client's yes.
fn gate(inputs: UnboundedSender<Input>, session: String, serial: u64, mode: Rc<Cell<Mode>>, always: Rc<RefCell<HashSet<String>>>) -> Gate {
    Rc::new(move |call_id: String, name: String, _arguments: Map<String, Value>| {
        let (inputs, session, mode, always) = (inputs.clone(), session.clone(), mode.clone(), always.clone());
        Box::pin(async move {
            if !needs_approval(mode.get(), &name) || always.borrow().contains(&name) {
                return Ok(());
            }
            let (reply, answer) = oneshot::channel();
            let _ = inputs.send(Input::Ask { session, serial, call_id, reply });
            match answer.await {
                Ok(Answer::Always) => {
                    always.borrow_mut().insert(name);
                    Ok(())
                }
                Ok(Answer::Once) => Ok(()),
                _ => Err(format!("The user declined this {name} call.")),
            }
        })
    })
}

// The commands a client offers as completions.
fn commands() -> Value {
    json!({"sessionUpdate": "available_commands_update", "availableCommands": [
        {"name": "compact", "description": "Summarize the conversation to free context", "input": {"hint": "what to keep (optional)"}},
        {"name": "init", "description": "Write this repo's AGENTS.md, direction.md and ledger.md"},
        {"name": "review", "description": "Hand a review of the changes to the reviewer job", "input": {"hint": "what to review (optional)"}}
    ]})
}

// A prompt's blocks as piccosta's prompt: the text, with each mentioned file named in place and
// its contents attached after (embedded by the client, or read here for a file link small enough),
// and the images.
fn prompt_input(blocks: &[Value], cwd: &Path) -> (String, Vec<Content>) {
    let (mut words, mut attached, mut images) = (Vec::new(), Vec::new(), Vec::new());
    for block in blocks {
        let field = |key: &str| block.get(key).and_then(Value::as_str).unwrap_or("").to_string();
        match block.get("type").and_then(Value::as_str) {
            Some("text") => words.push(field("text")),
            Some("image") => images.push(Content::Image { data: field("data"), mime_type: field("mimeType") }),
            Some("resource") => {
                let uri = block.pointer("/resource/uri").and_then(Value::as_str).unwrap_or("");
                let label = file_path(uri, cwd).map_or(uri.to_string(), |path| path.display().to_string());
                words.push(format!("`{label}`"));
                match block.pointer("/resource/text").and_then(Value::as_str) {
                    Some(text) => attached.push(format!("{label}:\n```\n{text}\n```")),
                    None => attached.push(format!("{label}: (binary, not included)")),
                }
            }
            Some("resource_link") => match file_path(&field("uri"), cwd) {
                Some(path) => {
                    words.push(format!("`{}`", path.display()));
                    if let Some(text) = read_small(&path) {
                        attached.push(format!("{}:\n```\n{text}\n```", path.display()));
                    }
                }
                None => words.push(format!("[{}]({})", field("name"), field("uri"))),
            },
            _ => {}
        }
    }
    let mut text = words.join(" ");
    if !attached.is_empty() {
        text.push_str("\n\n");
        text.push_str(&attached.join("\n\n"));
    }
    (text, images)
}

// A link's file: a file:// URI, or a bare path (Zed's own tests send one), taken from the cwd.
fn file_path(uri: &str, cwd: &Path) -> Option<PathBuf> {
    match url::Url::parse(uri) {
        Ok(url) if url.scheme() == "file" => url.to_file_path().ok(),
        Ok(_) => None,
        Err(_) => Some(cwd.join(uri)),
    }
}

// A text file's contents, when it is one and no bigger than FILE_MAX_BYTES.
fn read_small(path: &Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    (metadata.is_file() && metadata.len() <= FILE_MAX_BYTES).then(|| std::fs::read_to_string(path).ok()).flatten()
}

// What an edit or write is about to leave in the file, when it can be worked out: a write's
// content, or an edit's replacements applied to the text it found.
fn planned(tool: &str, arguments: &Map<String, Value>, before: Option<&str>) -> Option<String> {
    if tool == "write" {
        return arguments.get("content").and_then(Value::as_str).map(str::to_string);
    }
    use crate::tools::Tool;
    let arguments = crate::tools::edit::EditTool.prepare_arguments(arguments.clone());
    let edits: Vec<crate::tools::edit::Edit> = serde_json::from_value(arguments.get("edits")?.clone()).ok()?;
    crate::tools::edit::edit_content(before?, &edits, arguments.get("path")?.as_str()?).ok().map(|(content, ..)| content)
}

// The text of a tool result ({content: [...]}) cut to TOOL_OUTPUT_CHARS.
fn output_text(result: &Value) -> String {
    let blocks: Vec<Content> = result.get("content").and_then(|content| serde_json::from_value(content.clone()).ok()).unwrap_or_default();
    let text = TextOrBlocks::Blocks(blocks).text();
    if text.chars().count() <= TOOL_OUTPUT_CHARS {
        return text;
    }
    let cut: String = text.chars().take(TOOL_OUTPUT_CHARS).collect();
    format!("{cut}\n… (cut; the model has all of it)")
}

// What a client terminal still needs to show `text` when it already shows `shown`: the new tail,
// or all of it over a cleared screen when the output was cut from the front. Line ends become
// CRLF, since a terminal on a bare \n moves down without going back to the start of the line.
fn terminal_bytes(shown: &mut String, text: &str) -> String {
    let data = match text.strip_prefix(shown.as_str()) {
        Some(tail) => tail.to_string(),
        None => format!("\x1b[2J\x1b[H{text}"),
    };
    *shown = text.to_string();
    data.replace('\n', "\r\n")
}

// A tool call in a few words, the way the client lists it.
fn title(tool: &str, arguments: &Map<String, Value>) -> String {
    let get = |key: &str| arguments.get(key).and_then(Value::as_str).unwrap_or("");
    let first = |text: &str| text.lines().next().unwrap_or("").chars().take(120).collect::<String>();
    match tool {
        "bash" => first(get("command")),
        "read" => format!("Read {}", get("path")),
        "edit" => format!("Edit {}", get("path")),
        "write" => format!("Write {}", get("path")),
        "web_search" => format!("Search: {}", first(get("query"))),
        "fetch_content" => format!("Fetch {}", get("url")),
        // A subagent runs with its job's own tools and asks nothing, so the card says which.
        "subagent" => format!("{} ({}): {}", get("agent"), crate::tools::subagent::tools_of(get("agent")), first(get("task"))),
        _ => tool.to_string(),
    }
}

fn kind(tool: &str) -> &'static str {
    match tool {
        "read" => "read",
        "edit" | "write" => "edit",
        "bash" => "execute",
        "web_search" => "search",
        "fetch_content" | "get_search_content" => "fetch",
        _ => "other",
    }
}

// A loaded session's conversation, sent once: prompts as user chunks, responses as thought and
// message chunks, each tool call as one tool_call with its final status and output, summaries of
// compacted history as messages.
fn replay(session: &str, from: &Session, cwd: &Path, terminals: bool) {
    let messages: Vec<_> = from.path_to_leaf().into_iter().flat_map(crate::session::entry_context_messages).collect();
    let mut outcomes: HashMap<String, (String, bool)> = HashMap::new();
    for message in &messages {
        if let AgentMessage::ToolResult(result) = message {
            outcomes.insert(result.tool_call_id.clone(), (TextOrBlocks::Blocks(result.content.clone()).text(), result.is_error));
        }
    }
    for (index, message) in messages.iter().enumerate() {
        let id = format!("replay-{index}");
        let chunk = |kind: &str, text: &str| update(session, json!({"sessionUpdate": kind, "messageId": id, "content": text_block(text)}));
        match message {
            AgentMessage::User(user) => {
                chunk("user_message_chunk", &user.content.text());
                if let TextOrBlocks::Blocks(blocks) = &user.content {
                    for block in blocks {
                        if let Content::Image { data, mime_type } = block {
                            update(session, json!({"sessionUpdate": "user_message_chunk", "messageId": id, "content": {"type": "image", "data": data, "mimeType": mime_type}}));
                        }
                    }
                }
            }
            AgentMessage::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        Content::Thinking { thinking, .. } if !thinking.is_empty() => chunk("agent_thought_chunk", thinking),
                        Content::Text { text, .. } if !text.is_empty() => chunk("agent_message_chunk", text),
                        Content::ToolCall { id: call_id, name, arguments, .. } => {
                            let (output, failed) = outcomes.get(call_id).cloned().unwrap_or_else(|| ("No result was recorded.".into(), true));
                            let output = output_text(&json!({"content": [{"type": "text", "text": output}]}));
                            let mut call = json!({
                                "sessionUpdate": "tool_call", "toolCallId": call_id, "title": title(name, arguments), "name": name, "kind": kind(name),
                                "status": if failed { "failed" } else { "completed" }, "rawInput": Value::Object(arguments.clone()),
                                "content": [{"type": "content", "content": text_block(&output)}]
                            });
                            if let Some(path) = arguments.get("path").and_then(Value::as_str) {
                                call["locations"] = json!([{"path": crate::tools::paths::resolve(path, cwd)}]);
                            }
                            if !(terminals && name == "bash") {
                                update(session, call);
                                continue;
                            }
                            call["_meta"] = json!({"terminal_info": {"terminal_id": call_id, "cwd": cwd}});
                            call["content"] = json!([{"type": "terminal", "terminalId": call_id}]);
                            update(session, call);
                            let data = terminal_bytes(&mut String::new(), &output);
                            update(session, json!({"sessionUpdate": "tool_call_update", "toolCallId": call_id, "_meta": {"terminal_output": {"terminal_id": call_id, "data": data}, "terminal_exit": {"terminal_id": call_id}}}));
                        }
                        _ => {}
                    }
                }
            }
            AgentMessage::CompactionSummary(summary) => chunk("agent_message_chunk", &format!("Summary of the earlier conversation:\n\n{}", summary.summary)),
            AgentMessage::BranchSummary(summary) => chunk("agent_message_chunk", &format!("Summary of the earlier conversation:\n\n{}", summary.summary)),
            AgentMessage::Custom(custom) if custom.display => chunk("agent_message_chunk", &custom.content.text()),
            _ => {}
        }
    }
}
