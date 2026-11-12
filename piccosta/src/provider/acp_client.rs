// ACP as a provider: the child owns its tools and login, piccosta owns the local transcript and
// the user's interactions. There is one bidirectional reader, including while an approval waits.

use super::{Interact, Interaction, Request, StreamEvent, acp_wire as wire, child::Child};
use crate::message::{AssistantMessage, Content, CostState, Message, StopReason, ToolResultMessage};
use crate::model::ProviderConfig;
use futures_util::{StreamExt, stream::FuturesUnordered};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub const API: &str = "acp";
pub const NO_COMPACTION: &str = "Piccosta compaction is unavailable for ACP agents: they own their context. Use a new session or the agent's own context controls.";
// ACP server mode already gives a cancelled prompt five seconds to settle. A child that never
// acknowledges cancellation is closed, not reused with an outstanding prompt.
const CANCEL_WAIT: Duration = Duration::from_secs(5);
const PROTOCOL_VERSION: u64 = 1;
// ACP's existing defensive budget leaves room for large tool results, not an endless frame.
const FRAME_MAX_BYTES: usize = 16 * 1024 * 1024;
const HISTORY_MISMATCH: &str = "ACP history no longer matches this conversation (model switch, branch, context edit, instructions or agent command changed). Start a new session; history migration is not supported.";
const UNCERTAIN: &str =
    "The previous ACP prompt may have run but was not durably acknowledged. It will not be resent. Inspect it in the agent's own UI and start a new piccosta session.";

static LIVE: LazyLock<Mutex<HashMap<String, Connection>>> = LazyLock::new(Default::default);
static CLOSING: Mutex<Vec<tokio::task::JoinHandle<()>>> = Mutex::new(Vec::new());
fn live() -> MutexGuard<'static, HashMap<String, Connection>> {
    LIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

// The remote identity is valid only for this command, cwd, instructions and exact local prefix.
// In-flight is written and synced before sending. A crash can lose an answer, never resend an
// action whose completion we don't know. The checkpoint comes after all local records are saved.
#[derive(Serialize, Deserialize)]
struct Record {
    identity: String,
    remote: String,
    prints: Vec<String>,
    replay: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_tail: Option<Value>,
    in_flight: bool,
}
impl Record {
    // Load replays one stream, not separate turns. Keep only its last text item so adjacent user
    // chunks still merge after empty/cancelled turns, without storing a second whole transcript.
    fn append_replay(&mut self, turn: Replay) {
        let mut normalized = Replay::default();
        if let Some(tail) = self.replay_tail.take() {
            self.replay.pop();
            normalized.items.push(tail);
        }
        for item in turn.items {
            if let (Some(role), Some(text)) = (item["role"].as_str(), item["text"].as_str()) {
                normalized.text(role, text);
            } else {
                normalized.items.push(item);
            }
        }
        self.replay_tail = normalized.items.last().filter(|item| item["role"].is_string()).cloned();
        self.replay.extend(normalized.prints());
    }
}

fn hash(mut value: Value) -> String {
    // JSON key order and the way a tool update was split are not differences in history.
    value.sort_all_objects();
    Sha256::digest(value.to_string()).iter().map(|byte| format!("{byte:02x}")).collect()
}
// Setup diagnostics never reached the agent. Every prefix/count operation must skip exactly
// the same messages; uncertain failures have no such marker and remain part of the history.
fn conversation(messages: &[Message]) -> impl Iterator<Item = &Message> {
    messages.iter().filter(|message| !matches!(message, Message::System(_)) && !matches!(message, Message::Assistant(assistant) if assistant.request_not_sent))
}
fn prints(messages: &[Message]) -> Vec<String> {
    conversation(messages)
        .filter_map(|message| {
            Some(hash(match message {
                Message::System(_) => return None,
                Message::User(user) => json!({"role": "user", "content": user.content.clone().into_blocks()}),
                Message::Assistant(assistant) => {
                    json!({"role": "assistant", "provider": assistant.provider, "model": assistant.model, "content": assistant.content, "stop": assistant.stop_reason})
                }
                Message::ToolResult(result) => json!({"role": "toolResult", "id": result.tool_call_id, "content": result.content, "isError": result.is_error}),
            }))
        })
        .collect()
}
fn sidecar(file: &Path) -> PathBuf {
    let mut name = file.as_os_str().to_os_string();
    name.push(".acp.json");
    name.into()
}
fn save(path: Option<&Path>, record: &Record) -> Result<(), String> {
    let Some(path) = path else { return Ok(()) };
    let partial = path.with_extension(format!("{}.partial", uuid::Uuid::new_v4().simple()));
    let saved = (|| {
        crate::platform::private_dir(path.parent().ok_or("ACP record has no directory")?).map_err(|error| error.to_string())?;
        crate::auth::write_private_json(&partial, &serde_json::to_value(record).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
        std::fs::File::open(&partial).and_then(|file| file.sync_all()).map_err(|error| error.to_string())?;
        std::fs::rename(&partial, path).map_err(|error| error.to_string())?;
        #[cfg(unix)]
        std::fs::File::open(path.parent().ok_or("ACP record has no directory")?).and_then(|dir| dir.sync_all()).map_err(|error| error.to_string())?;
        Ok(())
    })();
    if saved.is_err() {
        let _ = std::fs::remove_file(partial);
    }
    saved
}

struct Connection {
    child: Child,
    next_id: u64,
    initialized: Value,
    state: Value,
    record: Record,
    file: Option<PathBuf>,
    acknowledged: Option<String>,
}
impl Connection {
    fn spawn(config: &ProviderConfig, cwd: &Path, record: Record, file: Option<PathBuf>) -> Result<Self, String> {
        let (program, args) = config.command.split_first().filter(|(program, _)| !program.is_empty()).ok_or("ACP provider needs command (executable and argv) in models.json")?;
        let mut command = std::process::Command::new(program);
        command.args(args).current_dir(cwd);
        Ok(Self {
            child: Child::start(command).map_err(|error| format!("ACP child: {error}"))?,
            next_id: 0,
            initialized: Value::Null,
            state: Value::Null,
            record,
            file,
            acknowledged: None,
        })
    }

    // Keep dispatching notifications and requests while our response or a frontend answer is
    // outstanding. Cancellation drops every frontend waiter, replies to the child, then drains
    // the prompt. There is no automatic retry on a lost response.
    async fn call(
        &mut self,
        method: &str,
        params: Value,
        cancel: &CancellationToken,
        interaction: Option<Interact>,
        yolo: bool,
        updates: &mut (dyn FnMut(Value) + Send),
    ) -> Result<Value, String> {
        self.next_id += 1;
        let id = json!(self.next_id);
        let message = wire::request(id.clone(), method, params);
        tokio::select! {
            result = self.child.write(&message) => result?,
            () = cancel.cancelled() => return Err("ACP request delivery interrupted; outcome uncertain".into()),
        }
        let mut pending = FuturesUnordered::new();
        let mut waiting = HashMap::<String, (Value, String)>::new();
        let mut cancelled = false;
        let mut deadline = None;
        loop {
            tokio::select! {
                line = self.child.next_line(Some(FRAME_MAX_BYTES)) => {
                    let line = line.ok_or_else(|| format!("ACP child closed its output or exceeded the frame limit. {}", self.child.stderr_so_far()))?;
                    if line.trim().is_empty() { continue; }
                    let message: Value = serde_json::from_str(&line).map_err(|error| format!("Invalid ACP JSON: {error}"))?;
                    if !wire::valid(&message) { return Err("Invalid ACP JSON-RPC envelope".into()); }
                    if let Some(method) = message["method"].as_str() {
                        let params = message.get("params").cloned().unwrap_or_else(|| json!({}));
                        if method == "session/update" {
                            if params["sessionId"].as_str() == Some(&self.record.remote) {
                                let update = &params["update"];
                                if update["sessionUpdate"] == "config_option_update" && update["configOptions"].is_array() { self.state["configOptions"] = update["configOptions"].clone(); }
                                updates(update.clone());
                            }
                        } else if let Some(request_id) = message.get("id") {
                            if !matches!(method, "session/request_permission" | "elicitation/create") {
                                self.child.write(&wire::error(request_id.clone(), -32601, "ACP client method is not supported", None)).await?;
                                continue;
                            }
                            if params["sessionId"].as_str() != Some(&self.record.remote) {
                                self.child.write(&wire::error(request_id.clone(), -32602, "Unknown session", None)).await?;
                                continue;
                            }
                            if method == "session/request_permission" {
                                let mut call = params["toolCall"].clone();
                                call["sessionUpdate"] = "tool_call_update".into();
                                updates(call);
                            }
                            let request = Interaction { method: method.into(), params };
                            let request_id = request_id.clone();
                            if waiting.contains_key(&request_id.to_string()) { return Err("Duplicate ACP interaction ID".into()); }
                            waiting.insert(request_id.to_string(), (request_id.clone(), method.into()));
                            let interaction = interaction.clone();
                            pending.push(async move {
                                let answer = if cancelled {
                                    Ok(request.answer(None))
                                } else if let Some(interaction) = interaction {
                                    // The frontend's current mode overrides the startup flag.
                                    interaction(request.clone()).await
                                } else {
                                    request.auto_allow(yolo).ok_or_else(|| "No frontend is available for this interaction".into())
                                };
                                let answer = answer.and_then(|answer| {
                                    if request.method == "session/request_permission" {
                                        let outcome = answer.pointer("/outcome/outcome").and_then(Value::as_str);
                                        let selected = answer.pointer("/outcome/optionId").and_then(Value::as_str);
                                        if outcome != Some("cancelled") && !(outcome == Some("selected") && request.choices().iter().any(|(id, _)| Some(id.as_str()) == selected)) {
                                            return Err("Frontend selected an option the agent did not offer".into());
                                        }
                                    } else if !matches!(answer["action"].as_str(), Some("accept" | "decline" | "cancel")) {
                                        return Err("Invalid elicitation answer".into());
                                    }
                                    Ok(answer)
                                });
                                (request_id, answer)
                            });
                        }
                    } else if message["id"] == id {
                        if !waiting.is_empty() { return Err("ACP prompt ended with unanswered interactions; outcome uncertain".into()); }
                        if let Some(error) = message.get("error") { return Err(format!("ACP {method}: {error}")); }
                        return Ok(message["result"].clone());
                    }
                }
                Some((request_id, answer)) = pending.next(), if !pending.is_empty() => {
                    waiting.remove(&request_id.to_string());
                    let response = match answer {
                        Ok(answer) => wire::reply(request_id, answer),
                        Err(error) => wire::error(request_id, -32601, error, None),
                    };
                    self.child.write(&response).await?;
                }
                () = cancel.cancelled(), if !cancelled => {
                    cancelled = true;
                    pending.clear();
                    for (_, (id, method)) in waiting.drain() {
                        let answer = if method == "session/request_permission" { super::permission_answer(None) } else { json!({"action": "cancel"}) };
                        self.child.write(&wire::reply(id, answer)).await?;
                    }
                    if method != "session/prompt" { return Err("ACP setup cancelled".into()); }
                    self.child.write(&wire::notification("session/cancel", json!({"sessionId": self.record.remote}))).await?;
                    deadline = Some(tokio::time::Instant::now() + CANCEL_WAIT);
                }
                () = async { if let Some(deadline) = deadline { tokio::time::sleep_until(deadline).await; } else { std::future::pending().await } } => return Err("ACP prompt did not settle after cancellation; child closed, outcome uncertain".into()),
            }
        }
    }

    async fn initialize(&mut self, cancel: &CancellationToken, elicitation: bool) -> Result<(), String> {
        let capabilities = if elicitation { json!({"elicitation": {"form": {}}}) } else { json!({}) };
        self.initialized = self
            .call(
                "initialize",
                json!({"protocolVersion": PROTOCOL_VERSION, "clientCapabilities": capabilities, "clientInfo": {"name": "piccosta", "version": env!("CARGO_PKG_VERSION")}}),
                cancel,
                None,
                false,
                &mut |_| {},
            )
            .await?;
        if self.initialized["protocolVersion"] != PROTOCOL_VERSION {
            return Err("ACP agent did not negotiate protocol version 1".into());
        }
        Ok(())
    }

    async fn open(&mut self, config: &ProviderConfig, cwd: &Path, prompt: &str, cancel: &CancellationToken) -> Result<(), String> {
        let loading = !self.record.remote.is_empty();
        if loading && self.initialized.pointer("/agentCapabilities/loadSession") != Some(&Value::Bool(true)) {
            return Err("ACP agent cannot load its saved session; start a new session".into());
        }
        let mut params = json!({"cwd": cwd, "mcpServers": []});
        if let Some(meta) = &config.session_meta {
            if prompt_extension(meta, &self.initialized["agentCapabilities"]["_meta"]) == Some(false) {
                return Err("ACP agent does not advertise the configured system-prompt append extension".into());
            }
            params["_meta"] = substitute(meta, prompt);
        }
        if loading {
            params["sessionId"] = self.record.remote.clone().into();
        }
        let mut replay = Replay::default();
        let result = self.call(if loading { "session/load" } else { "session/new" }, params, cancel, None, false, &mut |update| replay.update(&update)).await?;
        if loading {
            if replay.prints() != self.record.replay {
                return Err(format!("{HISTORY_MISMATCH} Remote load replay differs from the acknowledged transcript."));
            }
        } else {
            self.record.remote = result["sessionId"].as_str().filter(|id| !id.is_empty()).ok_or("ACP session/new returned no sessionId")?.into();
        }
        self.state = result;
        save(self.file.as_deref(), &self.record)
    }

    async fn set(&mut self, id: &str, value: &str, cancel: &CancellationToken) -> Result<(), String> {
        let option = options(&self.state).into_iter().find(|option| option["id"] == id).ok_or_else(|| format!("ACP agent no longer offers config option {id}"))?;
        if !selections(&option).iter().any(|(id, _)| id == value) {
            return Err(format!("ACP option {id} does not offer value {value}"));
        }
        if option["currentValue"] == value {
            return Ok(());
        }
        let result = self.call("session/set_config_option", json!({"sessionId": self.record.remote, "configId": id, "value": value}), cancel, None, false, &mut |_| {}).await?;
        if !result["configOptions"].is_array() {
            return Err("ACP set_config_option returned no configOptions".into());
        }
        self.state["configOptions"] = result["configOptions"].clone();
        if !options(&self.state).iter().any(|option| option["id"] == id && option["currentValue"] == value) {
            return Err(format!("ACP agent did not confirm {id}={value}"));
        }
        Ok(())
    }

    async fn configure(&mut self, config: &ProviderConfig, model: &str, effort: Option<&str>, cancel: &CancellationToken) -> Result<(), String> {
        let option = category(&self.state, "model").ok_or("ACP agent offered no model config option")?;
        self.set(option["id"].as_str().ok_or("ACP model option has no id")?, model, cancel).await?;
        // Model selection can replace the option IDs and their allowed efforts.
        if let Some(effort) = effort {
            let option = category(&self.state, "thought_level").ok_or("ACP agent no longer offers a thinking option")?;
            self.set(option["id"].as_str().ok_or("ACP effort option has no id")?, effort, cancel).await?;
        }
        for (id, value) in &config.config_options {
            if options(&self.state).iter().any(|option| option["id"] == *id && matches!(option["category"].as_str(), Some("model" | "thought_level"))) {
                return Err("Choose ACP model and thinking through piccosta, not configOptions".into());
            }
            self.set(id, value, cancel).await?;
        }
        Ok(())
    }
}

// A placeholder must correspond to an advertised append capability, not an arbitrary _meta key
// that the agent may silently ignore. Non-prompt metadata doesn't claim to deliver instructions.
fn prompt_extension(meta: &Value, capabilities: &Value) -> Option<bool> {
    match meta {
        Value::String(text) if text.contains("{systemPrompt}") => Some(capabilities == true),
        Value::Object(fields) => {
            let checks: Vec<bool> = fields.iter().filter_map(|(name, value)| prompt_extension(value, &capabilities[name])).collect();
            (!checks.is_empty()).then(|| checks.into_iter().all(|supported| supported))
        }
        _ => None,
    }
}

fn substitute(value: &Value, prompt: &str) -> Value {
    match value {
        Value::String(value) => value.replace("{systemPrompt}", prompt).into(),
        Value::Array(values) => values.iter().map(|value| substitute(value, prompt)).collect(),
        Value::Object(values) => values.iter().map(|(key, value)| (key.clone(), substitute(value, prompt))).collect::<Map<_, _>>().into(),
        other => other.clone(),
    }
}
fn options(state: &Value) -> Vec<Value> {
    state["configOptions"].as_array().cloned().unwrap_or_default()
}
fn category(state: &Value, category: &str) -> Option<Value> {
    options(state).into_iter().find(|option| option["category"] == category)
}
fn selections(option: &Value) -> Vec<(String, String)> {
    option["options"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|entry| match entry["value"].as_str() {
            Some(value) => vec![(value.to_string(), entry["name"].as_str().unwrap_or(value).to_string())],
            None => selections(entry),
        })
        .collect()
}
// Discover on the existing background catalog refresh, never on the TUI input thread. Limits and
// prices aren't part of ACP's model list, so zero limits and an unknown cost mean not supplied.
pub async fn offered_models(config: &ProviderConfig) -> Result<Vec<Value>, String> {
    let cwd = std::env::current_dir().map_err(|error| error.to_string())?;
    let record = Record { identity: String::new(), remote: String::new(), prints: Vec::new(), replay: Vec::new(), replay_tail: None, in_flight: false };
    let mut connection = Connection::spawn(config, &cwd, record, None)?;
    let cancel = CancellationToken::new();
    let listed = async {
        connection.initialize(&cancel, false).await?;
        connection.open(config, &cwd, "", &cancel).await?;
        let models = category(&connection.state, "model").map(|option| selections(&option)).unwrap_or_default();
        if models.is_empty() {
            return Err("ACP agent offered no selectable models".into());
        }
        let mut listed = Vec::new();
        for (id, name) in models {
            if !config.accepts_model(&id)? {
                continue;
            }
            connection.configure(config, &id, None, &cancel).await?;
            let efforts = category(&connection.state, "thought_level").map(|option| selections(&option)).unwrap_or_default();
            let thinking: Map<String, Value> = super::THINKING_LEVELS
                .into_iter()
                .map(|level| {
                    let offered = efforts.iter().find(|(id, _)| id == level || (level == "off" && id == "none"));
                    (level.into(), offered.map_or(Value::Null, |(id, _)| id.clone().into()))
                })
                .collect();
            listed.push(
                json!({"id": id, "name": name, "api": API, "input": ["text"], "reasoning": !efforts.is_empty(), "thinkingLevelMap": thinking, "contextWindow": 0, "maxTokens": 0}),
            );
        }
        Ok(listed)
    }
    .await;
    connection.child.close().await;
    listed
}

// Normalize replay independent of chunk sizes and tool-update boundaries. Only message text and
// tool records are promised by loadSession; plans, usage, titles and thinking are not history.
#[derive(Default)]
struct Replay {
    items: Vec<Value>,
}
impl Replay {
    fn text(&mut self, role: &str, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(last) = self.items.last_mut().filter(|last| last["role"] == role) {
            if let Value::String(whole) = &mut last["text"] {
                whole.push_str(text);
            }
        } else {
            self.items.push(json!({"role": role, "text": text}));
        }
    }
    fn update(&mut self, update: &Value) {
        match update["sessionUpdate"].as_str() {
            Some("user_message_chunk") => self.text("user", update.pointer("/content/text").and_then(Value::as_str).unwrap_or_default()),
            Some("agent_message_chunk") => self.text("assistant", update.pointer("/content/text").and_then(Value::as_str).unwrap_or_default()),
            Some("tool_call" | "tool_call_update") => {
                let Some(id) = update["toolCallId"].as_str() else { return };
                let at = self.items.iter().position(|item| item["toolCallId"] == id).unwrap_or_else(|| {
                    self.items.push(json!({"toolCallId": id}));
                    self.items.len() - 1
                });
                for field in ["title", "kind", "status", "rawInput", "rawOutput", "content", "locations"] {
                    if let Some(value) = update.get(field) {
                        self.items[at][field] = value.clone();
                    }
                }
            }
            _ => {}
        }
    }
    fn prints(&self) -> Vec<String> {
        self.items.iter().cloned().map(hash).collect()
    }
}

struct Turn<'a> {
    output: AssistantMessage,
    events: &'a mpsc::UnboundedSender<StreamEvent>,
    calls: HashMap<String, (usize, Value, bool)>,
    replay: Replay,
}
impl Turn<'_> {
    fn update(&mut self, update: Value) {
        self.replay.update(&update);
        match update["sessionUpdate"].as_str() {
            Some(kind @ ("agent_message_chunk" | "agent_thought_chunk")) => {
                let Some(text) = update.pointer("/content/text").and_then(Value::as_str) else { return };
                let thinking = kind == "agent_thought_chunk";
                let append = self.output.content.last().is_some_and(|block| matches!((block, thinking), (Content::Text { .. }, false) | (Content::Thinking { .. }, true)));
                if !append {
                    self.output.content.push(if thinking { Content::Thinking { thinking: String::new(), thinking_signature: None, redacted: None } } else { Content::text("") });
                    let index = self.output.content.len() - 1;
                    let _ = self.events.send(if thinking { StreamEvent::ThinkingStart { index } } else { StreamEvent::TextStart { index } });
                }
                let index = self.output.content.len() - 1;
                match &mut self.output.content[index] {
                    Content::Text { text: whole, .. } | Content::Thinking { thinking: whole, .. } => whole.push_str(text),
                    _ => {}
                }
                let _ = self.events.send(if thinking { StreamEvent::ThinkingDelta { index, delta: text.into() } } else { StreamEvent::TextDelta { index, delta: text.into() } });
            }
            Some("tool_call" | "tool_call_update") => self.tool(update),
            Some("usage_update") => self.output.usage.reported = Some(update),
            _ => {}
        }
    }

    fn tool(&mut self, update: Value) {
        let Some(id) = update["toolCallId"].as_str().filter(|id| !id.is_empty()).map(str::to_string) else {
            return;
        };
        let fresh = !self.calls.contains_key(&id);
        let (index, raw, done) = self.calls.entry(id.clone()).or_insert_with(|| {
            let index = self.output.content.len();
            self.output.content.push(Content::ToolCall {
                external: true,
                id: id.clone(),
                name: String::new(),
                arguments: Map::new(),
                namespace: None,
                arguments_error: None,
                thought_signature: None,
            });
            (index, json!({}), false)
        });
        if *done {
            return;
        }
        if let (Some(raw), Some(update)) = (raw.as_object_mut(), update.as_object()) {
            raw.extend(update.clone());
        }
        let Content::ToolCall { name, arguments, .. } = &mut self.output.content[*index] else {
            return;
        };
        if name.is_empty() {
            *name = raw["title"].as_str().or(raw["kind"].as_str()).unwrap_or("Agent tool").into();
        }
        *arguments = match raw.get("rawInput") {
            Some(Value::Object(arguments)) => arguments.clone(),
            Some(value) => Map::from_iter([("input".into(), value.clone())]),
            None => Map::new(),
        };
        if fresh {
            let _ = self.events.send(StreamEvent::ToolCallStart { index: *index, id: id.clone(), name: name.clone() });
            let _ = self.events.send(StreamEvent::ExternalStart { id: id.clone(), name: name.clone(), arguments: arguments.clone() });
        }
        let _ = self.events.send(StreamEvent::ToolCallEnd { index: *index, tool_call: self.output.content[*index].clone() });
        let (name, arguments) = match &self.output.content[*index] {
            Content::ToolCall { name, arguments, .. } => (name.clone(), arguments.clone()),
            _ => unreachable!(),
        };
        let content = tool_content(raw);
        if matches!(raw["status"].as_str(), Some("completed" | "failed")) {
            *done = true;
            let _ = self.events.send(StreamEvent::ExternalEnd(ToolResultMessage {
                tool_call_id: id,
                tool_name: name,
                content,
                details: Some(json!({"acp": raw})),
                usage: None,
                is_error: raw["status"] == "failed",
                timestamp: crate::clock::now_ms(),
                duration_ms: None,
            }));
        } else {
            let _ = self.events.send(StreamEvent::ExternalUpdate { id, name, arguments, result: json!({"content": content, "details": {"acp": raw}}) });
        }
    }

    fn finish_calls(&mut self) {
        let pending: Vec<String> = self.calls.iter().filter(|(_, (_, _, done))| !done).map(|(id, _)| id.clone()).collect();
        for id in pending {
            self.tool(json!({"sessionUpdate": "tool_call_update", "toolCallId": id, "status": "failed", "content": [{"type": "content", "content": {"type": "text", "text": "Outcome unknown: ACP prompt ended without a terminal tool result."}}]}));
        }
    }
}
fn tool_content(raw: &Value) -> Vec<Content> {
    let text: Vec<String> = raw["content"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|content| content.pointer("/content/text").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| content.to_string()))
        .collect();
    vec![Content::text(if text.is_empty() { raw.get("rawOutput").map_or(String::new(), Value::to_string) } else { text.join("\n") })]
}

pub async fn stream(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let mut turn = Turn { output: super::new_output(request.model, API), events, calls: HashMap::new(), replay: Replay::default() };
    turn.output.usage.cost.state = CostState::Unknown;
    turn.output.usage.tokens_unknown = true;
    turn.output.request_not_sent = true;
    let _ = events.send(StreamEvent::Start);
    let result = run(request, &mut turn).await;
    turn.finish_calls();
    super::settle(request, &mut turn.output, result);
    turn.output
}
async fn run(request: &Request<'_>, turn: &mut Turn<'_>) -> Result<(), String> {
    let config = request.provider_config.ok_or("ACP provider has no models.json command")?;
    if request.in_place {
        return Err(NO_COMPACTION.into());
    }
    if !config.accepts_model(&request.model.id)? {
        return Err("ACP model refused by provider modelFilter".into());
    }
    if config.interactive_only && (!request.interactive || request.manager || std::env::var_os(crate::tools::subagent::CHILD_ENV).is_some() || crate::manager::started_by_agent()) {
        return Err("This ACP provider permits user-picked interactive sessions only (interactiveOnly in models.json).".into());
    }
    let session = request.session_id.ok_or("ACP needs a local session identity")?;
    let system = super::transcript::current_system_message(request.messages).map(|message| super::transcript::system_message_text(&message)).unwrap_or_default();
    // Config selections may be corrected between turns; command and instructions may not.
    let identity = hash(json!([session, request.model.provider, config.command, config.session_meta, request.cwd, request.session_file, system]));
    let file = request.session_file.map(sidecar);
    let current = prints(request.messages);
    let mut connection = live().remove(session);
    if connection.is_none() {
        let record = match file.as_ref().map(std::fs::read_to_string).transpose() {
            Ok(Some(text)) => serde_json::from_str::<Record>(&text).map_err(|error| format!("ACP session record is unreadable: {error}"))?,
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(format!("ACP session record: {error}")),
            _ => Record { identity: identity.clone(), remote: String::new(), prints: Vec::new(), replay: Vec::new(), replay_tail: None, in_flight: false },
        };
        if record.remote.is_empty() && !request.fresh_conversation {
            return Err(HISTORY_MISMATCH.into());
        }
        validate(&record, &identity, &current, request.messages)?;
        let mut child = Connection::spawn(config, request.cwd, record, file)?;
        child.initialize(&request.cancel, request.elicitation).await?;
        child.open(config, request.cwd, &system, &request.cancel).await?;
        connection = Some(child);
    }
    let mut connection = connection.ok_or("ACP child is unavailable")?;
    validate(&connection.record, &identity, &current, request.messages)?;
    let new: Vec<_> = conversation(request.messages).skip(connection.record.prints.len()).collect();
    let text = new
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(user.content.text()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    if new.iter().any(|message| matches!(message, Message::User(user) if user.content.clone().into_blocks().iter().any(|block| !matches!(block, Content::Text { .. })))) {
        return Err("This ACP client supports text prompts only".into());
    }
    let effort = request.model.provider_thinking_level(request.thinking_level);
    connection.configure(config, &request.model.id, effort.as_deref(), &request.cancel).await?;
    turn.output.provider_thinking_level = effort;
    let first = connection.record.prints.is_empty();
    let appended = config.session_meta.as_ref().and_then(|meta| prompt_extension(meta, &connection.initialized["agentCapabilities"]["_meta"])).is_some();
    let prompt = if first && !appended && !system.is_empty() { format!("<client_instructions>\n{system}\n</client_instructions>\n\n{text}") } else { text };
    turn.replay.text("user", &prompt);
    connection.record.in_flight = true;
    save(connection.file.as_deref(), &connection.record)?;
    // From here even a partial write is uncertain. Never classify it as a setup failure.
    turn.output.request_not_sent = false;
    let response = connection
        .call(
            "session/prompt",
            json!({"sessionId": connection.record.remote, "prompt": [{"type": "text", "text": prompt}]}),
            &request.cancel,
            request.interaction.clone(),
            request.yolo,
            &mut |update| turn.update(update),
        )
        .await?;
    turn.output.stop_reason = match response["stopReason"].as_str() {
        Some("end_turn") => StopReason::Stop,
        Some("cancelled") => StopReason::Aborted,
        Some("max_tokens" | "max_turn_requests") => StopReason::Length,
        other => return Err(format!("Unknown ACP stop reason {other:?}; prompt will not be resent")),
    };
    let unfinished = turn.calls.values().any(|(_, _, done)| !done);
    turn.finish_calls();
    if unfinished {
        return Err("ACP prompt ended with unfinished tools; outcome uncertain, child will not be reused".into());
    }
    connection.record.append_replay(std::mem::take(&mut turn.replay));
    let response_id = uuid::Uuid::new_v4().to_string();
    turn.output.response_id = Some(response_id.clone());
    connection.acknowledged = Some(response_id);
    live().insert(session.into(), connection);
    Ok(())
}
fn validate(record: &Record, identity: &str, current: &[String], messages: &[Message]) -> Result<(), String> {
    if record.in_flight {
        return Err(UNCERTAIN.into());
    }
    if record.identity != identity || !current.starts_with(&record.prints) {
        return Err(HISTORY_MISMATCH.into());
    }
    let new: Vec<_> = conversation(messages).skip(record.prints.len()).collect();
    if new.is_empty() || new.iter().any(|message| !matches!(message, Message::User(_))) {
        return Err(HISTORY_MISMATCH.into());
    }
    Ok(())
}

// The assistant and all external results are now on disk. Metadata written by the caller (usage,
// duration, prompt-cache notes) does not affect the canonical history correspondence.
pub fn checkpoint(session: &crate::session::Session, assistant: &AssistantMessage) -> Result<(), String> {
    let mut live = live();
    let Some(connection) = live.get_mut(session.id()) else { return Ok(()) };
    // Two missing IDs prove nothing: a credential failure never dispatched its user input.
    let Some(acknowledged) = connection.acknowledged.as_deref().filter(|id| !id.is_empty()) else { return Ok(()) };
    if assistant.response_id.as_deref() != Some(acknowledged) {
        return Ok(());
    }
    connection.record.prints = prints(&crate::message::to_model_messages(&session.context().messages, session.header().notice_token.as_deref()));
    connection.record.in_flight = false;
    if let Err(error) = save(connection.file.as_deref(), &connection.record) {
        connection.record.in_flight = true;
        return Err(format!("Could not checkpoint ACP history: {error}"));
    }
    connection.acknowledged = None;
    Ok(())
}
pub fn close(session: &str) {
    if let Some(mut connection) = live().remove(session) {
        let mut closing = CLOSING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        closing.retain(|task| !task.is_finished());
        closing.push(tokio::spawn(async move {
            connection.child.close().await;
        }));
    }
}
pub async fn shutdown() {
    let connections: Vec<_> = live().drain().map(|(_, connection)| connection).collect();
    futures_util::future::join_all(connections.into_iter().map(|mut connection| async move {
        connection.child.close().await;
    }))
    .await;
    let closing: Vec<_> = CLOSING.lock().unwrap_or_else(std::sync::PoisonError::into_inner).drain(..).collect();
    let _ = futures_util::future::join_all(closing).await;
}
