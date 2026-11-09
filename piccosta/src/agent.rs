// The agent: one prompt in, model turns and tool calls until the model is done, every message
// written to the session as it happens, and events out for whoever is watching (json mode, the
// UI).
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/agent/src/agent-loop.ts (turns, tool
// batches, event order) and the run controller of packages/coding-agent/src/core/agent-session.ts
// (system prompt updates, persistence, retry, queued messages). What pi keeps in memory and syncs
// back to the session, this reads from the session: each request's context is built from the
// session tree (agent-session.ts prepareRequest), so a retry's context edit or a compaction is
// simply what the next request sees.
//
// Event order for a prompt with one tool call, as pi emits it: agent_start, turn_start, the
// starting messages (message_start/end each: a system update first if the prompt or tools
// changed, then the user message), the assistant's message_start, message_update*, message_end,
// tool_execution_start, tool_execution_update*, tool_execution_end, the tool result's
// message_start/end, turn_end, then the next turn_start ... and agent_end. A retry runs another
// agent_start..agent_end; agent_settled comes once at the very end.
//
// Compaction (compaction.rs) is checked where pi checks it: before a prompt is sent, before each
// next response within a run, and after a run, where a context overflow is compacted and asked
// again once.

use crate::auth::{self, CredentialPaths};
use crate::compaction::{self, Reason, RetryNotice};
use crate::config::{CompactionSettings, RetrySettings};
use crate::message::{
    AgentMessage, AssistantMessage, Content, CustomMessage, StopReason, SystemMessage, TextOrBlocks, ToolDefinition, ToolReference,
    ToolResultMessage, Usage, UserMessage,
};
use crate::model::{Model, ModelRegistry};
use crate::provider::{self, transcript, Request, StreamEvent};
use crate::session::{Session, SessionEntry};
use crate::tools::{self, Tool, ToolContext, ToolShared, ToolUpdate};
use crate::{clock, pintask, retry};
use futures_util::stream::{FuturesUnordered, StreamExt};
use indexmap::IndexMap;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// The text a tool call gets when its response hit the output limit: its arguments may be cut off,
// so none of the calls run (agent-loop.ts:493).
fn truncated_call_text(name: &str) -> String {
    format!(
        "Tool call \"{name}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments."
    )
}

// What the agent reports, in pi's AgentSessionEvent shapes so json mode prints the same lines.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", rename_all_fields = "camelCase")]
pub enum Event {
    AgentStart,
    AgentEnd {
        messages: Vec<AgentMessage>,
        will_retry: bool,
    },
    AgentSettled {
        aborted: bool,
    },
    TurnStart,
    TurnEnd {
        message: AgentMessage,
        tool_results: Vec<AgentMessage>,
    },
    MessageStart {
        message: AgentMessage,
    },
    // pi's json shape: the cumulative usage and the stream event without its partial message.
    MessageUpdate {
        usage: Usage,
        assistant_message_event: Value,
    },
    MessageEnd {
        message: AgentMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Map<String, Value>,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Map<String, Value>,
        partial_result: Value,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: Value,
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    AutoRetryStart {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        error_message: String,
    },
    AutoRetryEnd {
        success: bool,
        attempt: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        final_error: Option<String>,
    },
    CompactionStart {
        reason: Reason,
    },
    // `result` is pi's CompactionResult on success; pi leaves it and `errorMessage` out otherwise.
    CompactionEnd {
        reason: Reason,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
        aborted: bool,
        will_retry: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
    },
    SummarizationRetryScheduled {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        error_message: String,
    },
    SummarizationRetryAttemptStart {
        source: &'static str,
        reason: Reason,
    },
    SummarizationRetryFinished,
    // What's still queued, after queued messages were taken into the conversation (pi emits it
    // before their message_start).
    QueueUpdate {
        steering: Vec<String>,
        follow_up: Vec<String>,
    },
    // An entry written outside the message flow: a context edit hiding a failed attempt, a
    // feature's own record.
    EntryAppended {
        entry: SessionEntry,
    },
}

pub type EventSink = Box<dyn FnMut(&Event) + Send>;

// Asked before each tool call runs, with the tool's name and arguments: Ok to run it, or the
// reason it was refused, which becomes the call's error result. A frontend that asks the user
// sets one (rpc mode for T3 Code's approvals). A tool's own question always needs a yes;
// without a frontend it is refused. The last flag is the tool's read-only declaration.
pub type Gate = std::rc::Rc<dyn Fn(String, String, Map<String, Value>, Option<String>, bool) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>>>>>;

// A session's controls for whoever drives it while it runs: abort the current run, and queue
// messages for it. Steering goes in after the current turn's tool calls, a follow-up when the
// model would otherwise stop (pi's steer and followUp, each one at a time by default). The Notify
// wakes print mode's wait for background jobs when a message is queued.
#[derive(Clone, Default)]
pub struct Handle(Arc<(std::sync::Mutex<Inbox>, tokio::sync::Notify)>);

#[derive(Default)]
struct Inbox {
    // The token of the run in progress; each prompt puts in a fresh one, so a handle taken before
    // the prompt still aborts it.
    cancel: CancellationToken,
    steer: Vec<String>,
    follow_up: Vec<String>,
    steering_all: bool,
    follow_up_all: bool,
    // A /fast that came during the run, for its next request.
    fast: Option<bool>,
}

impl Handle {
    pub fn abort(&self) {
        self.lock().cancel.cancel();
    }

    pub fn steer(&self, text: String) {
        self.lock().steer.push(text);
        self.0.1.notify_one();
    }

    pub fn follow_up(&self, text: String) {
        self.lock().follow_up.push(text);
        self.0.1.notify_one();
    }

    pub fn set_steering_mode(&self, mode: &str) {
        self.lock().steering_all = mode == "all";
    }

    pub fn set_follow_up_mode(&self, mode: &str) {
        self.lock().follow_up_all = mode == "all";
    }

    // Put in (Some) or take out (None) a /fast for the run's next request. Output: what was there.
    pub fn swap_fast(&self, on: Option<bool>) -> Option<bool> {
        std::mem::replace(&mut self.lock().fast, on)
    }

    // What's queued and not sent yet: steering, then follow-ups.
    pub fn queued(&self) -> (Vec<String>, Vec<String>) {
        let inbox = self.lock();
        (inbox.steer.clone(), inbox.follow_up.clone())
    }

    // Take back everything queued (pi's clear_queue), e.g. to put it back in the editor on abort.
    pub fn clear(&self) -> (Vec<String>, Vec<String>) {
        let mut inbox = self.lock();
        (std::mem::take(&mut inbox.steer), std::mem::take(&mut inbox.follow_up))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inbox> {
        self.0.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn dequeue(queue: &mut Vec<String>, all: bool) -> Vec<String> {
    let count = if all { queue.len() } else { queue.len().min(1) };
    queue.drain(..count).collect()
}

fn user_message(text: &str) -> AgentMessage {
    with_images(text, Vec::new())
}

// The user's text, then any images they attached (pi puts them after the text).
fn with_images(text: &str, images: Vec<Content>) -> AgentMessage {
    let mut blocks = vec![Content::text(text)];
    blocks.extend(images);
    AgentMessage::User(UserMessage { content: TextOrBlocks::Blocks(blocks), timestamp: clock::now_ms() })
}

// A stream event in pi's json form (json-event.ts:16-35): its type and content index, the delta or
// finished block, and a tool call's id and name on its start.
fn stream_event_json(event: &StreamEvent) -> Option<Value> {
    Some(match event {
        StreamEvent::Start => return None,
        StreamEvent::TextStart { index } => json!({"type": "text_start", "contentIndex": index}),
        StreamEvent::TextDelta { index, delta } => json!({"type": "text_delta", "contentIndex": index, "delta": delta}),
        StreamEvent::TextEnd { index, content } => json!({"type": "text_end", "contentIndex": index, "content": content}),
        StreamEvent::ThinkingStart { index } => json!({"type": "thinking_start", "contentIndex": index}),
        StreamEvent::ThinkingDelta { index, delta } => json!({"type": "thinking_delta", "contentIndex": index, "delta": delta}),
        StreamEvent::ThinkingEnd { index, content } => json!({"type": "thinking_end", "contentIndex": index, "content": content}),
        StreamEvent::ToolCallStart { index, id, name } => json!({"type": "toolcall_start", "contentIndex": index, "id": id, "toolName": name}),
        StreamEvent::ToolCallDelta { index, delta } => json!({"type": "toolcall_delta", "contentIndex": index, "delta": delta}),
        StreamEvent::ToolCallEnd { index, tool_call } => json!({"type": "toolcall_end", "contentIndex": index, "toolCall": tool_call}),
    })
}

// A tool's outcome as pi's AgentToolResult: content and details, details `{}` for a failure.
fn result_json(content: &[Content], details: &Option<Value>) -> Value {
    let mut result = Map::new();
    result.insert("content".into(), serde_json::to_value(content).unwrap_or_default());
    if let Some(details) = details {
        result.insert("details".into(), details.clone());
    }
    Value::Object(result)
}

// Everything the agent runs with.
pub struct AgentConfig {
    pub model: Model,
    pub registry: ModelRegistry,
    pub credentials: CredentialPaths,
    pub thinking_level: String,
    pub tools: Vec<Arc<dyn Tool>>,
    pub tool_shared: Arc<ToolShared>,
    pub cwd: PathBuf,
    // The system prompt as sections (prompt.rs).
    pub sections: IndexMap<String, String>,
    // Whether the prompt carries the session's AGENTS.md files (--no-context-files turns them off);
    // they're looked at before each run, for an edit (prompt_update).
    pub context_files: bool,
    pub retry: RetrySettings,
    // When to compact, resolved for this model.
    pub compaction: CompactionSettings,
    // How Codex requests travel (config.rs transport).
    pub transport: String,
    // Background jobs' finish messages.
    pub notices: mpsc::UnboundedReceiver<CustomMessage>,
    pub gate: Option<Gate>,
    // Print and json mode: hold the run open while a background job runs, so its finish message
    // becomes a turn before the process exits.
    pub hold_for_jobs: bool,
    // The manager's home, for the manager service: its compactions keep only the newest of the
    // user's words verbatim and add its ledger's newest entries (manager.rs).
    pub manager_home: Option<PathBuf>,
    // The session's socket, which its runtime listens on (runtime.rs).
    pub claim: Option<crate::runtime::Claim>,
}

pub struct Agent {
    pub session: Session,
    pub(crate) pending_plugins: Option<tools::plugin::Discovery>,
    config: AgentConfig,
    emit: EventSink,
    // Cancelled when the user aborts; each run gets a fresh one, which the handle then holds.
    cancel: CancellationToken,
    handle: Handle,
    // The retry in progress: attempts made so far.
    retry_attempt: u32,
    // Whether this prompt already compacted and asked again after an overflow; a second overflow
    // then fails instead of looping (agent-session.ts _overflowRecoveryAttempted).
    overflow_recovery_attempted: bool,
    // Background job messages kept for the next prompt instead of starting a turn of their own.
    held: Vec<CustomMessage>,
    // The loop guard's count of the same tool call made back to back: the call, and how many times.
    repeated_call: (String, u32),
    // When the model was last told the machine is overused (machine.rs).
    machine_noted: Option<std::time::Instant>,
}

// The loop guards. A response whose last LOOP_WINDOW
// bytes are one unit of at most LOOP_UNIT_MAX bytes over and over is stuck: hy4-preview streamed one
// placeholder token for ten minutes (sweep, 2026-10-30). Eight thousand bytes of one short unit
// is far past any table rule or run of digits a real answer has; it's checked each time another
// LOOP_CHECK_EVERY bytes have streamed. And the same tool call (tool and arguments) made
// REPEATED_CALL_LIMIT times back to back ends the run: a model that hasn't changed its call after
// five identical results won't on the sixth.
const LOOP_WINDOW: usize = 8000;
const LOOP_UNIT_MAX: usize = 50;
const LOOP_CHECK_EVERY: usize = 500;
const REPEATED_CALL_LIMIT: u32 = 5;
pub const LOOP_NOTE_TYPE: &str = "piccosta-loop-guard";

// The unit `text` ends in when its last LOOP_WINDOW bytes are that unit over and over.
fn looping(text: &str) -> Option<String> {
    let tail = text.as_bytes().get(text.len().checked_sub(LOOP_WINDOW)?..)?;
    let unit = (1..=LOOP_UNIT_MAX).find(|&unit| tail[unit..].iter().zip(tail).all(|(later, earlier)| later == earlier))?;
    Some(String::from_utf8_lossy(&tail[tail.len() - unit..]).into_owned())
}

// The session entry /fast writes, {"on": true or false}. A session without one runs at its model's
// own serviceTier from models.json.
pub const FAST_TYPE: &str = "piccosta-fast";

// How far into its prompt cache's life an idle session compacts: Claude Code 2.1.319's
// lastActivityAt + (expiresAt - lastActivityAt) * 0.9, 54 minutes of an hour.
const IDLE_SHARE: f64 = 0.9;
// How long idle compaction's request may take to go out once it's decided on: a terminal UI draws on
// the sessions' thread, and a debug build drawing a 450 KB transcript held the request back 3.2 to
// 3.6 s (2026-11-08). Closer than this to the cache's expiry, it doesn't run.
const IDLE_SEND_MS: i64 = 5_000;

// The service tier `model`'s requests ask for in `session`: /fast's last choice there, on a model
// that offers fast mode, else the model's own.
pub fn service_tier(session: &Session, model: &Model) -> Option<String> {
    let offered = provider::FAST_APIS.contains(&model.api.as_str());
    let chosen = session.path_to_leaf().into_iter().rev().filter(|_| offered).find_map(|entry| match entry {
        SessionEntry::Custom { custom_type, data: Some(data), .. } if custom_type == FAST_TYPE => data["on"].as_bool(),
        _ => None,
    });
    chosen.map_or_else(|| model.service_tier.clone(), |on| on.then(|| provider::FAST_TIER.to_string()))
}

struct ToolRun {
    index: usize,
    call_id: String,
    name: String,
    output: Result<tools::ToolOutput, String>,
    // How long the tool ran; None for a call that never started (pi leaves durationMs out then).
    duration_ms: Option<u64>,
}

impl Agent {
    // Input: the session, what to run with, and where events go. Records the model and thinking
    // level on the session where they differ from what it last had (sdk.ts:253-278, 449-452).
    pub fn new(mut session: Session, config: AgentConfig, emit: EventSink) -> std::io::Result<Agent> {
        let context = session.context();
        let has_thinking_entry = session.path_to_leaf().iter().any(|entry| matches!(entry, SessionEntry::ThinkingLevelChange { .. }));
        if context.model.as_ref() != Some(&(config.model.provider.clone(), config.model.id.clone())) {
            let base = session.next_base();
            session.append(SessionEntry::ModelChange { base, provider: config.model.provider.clone(), model_id: config.model.id.clone() })?;
        }
        let level = provider::clamp_thinking_level(&config.model, &config.thinking_level).to_string();
        if !has_thinking_entry || context.thinking_level != level {
            let base = session.next_base();
            session.append(SessionEntry::ThinkingLevelChange { base, thinking_level: level.clone() })?;
        }
        let mut config = AgentConfig { thinking_level: level, ..config };
        config.model.service_tier = service_tier(&session, &config.model);
        let handle = Handle::default();
        let cancel = handle.lock().cancel.clone();
        Ok(Agent { session, pending_plugins: None, config, emit, cancel, handle, retry_attempt: 0, overflow_recovery_attempted: false, held: Vec::new(), repeated_call: (String::new(), 0), machine_noted: None })
    }

    // The controls for this agent's runs (abort on Ctrl-C or Escape, queued messages).
    pub fn handle(&self) -> Handle {
        self.handle.clone()
    }

    // Have `watch` see every event before whoever the events go to (the runtime's turn ends).
    pub fn tap(&mut self, mut watch: impl FnMut(&Event) + Send + 'static) {
        let mut emit = std::mem::replace(&mut self.emit, Box::new(|_| {}));
        self.emit = Box::new(move |event| {
            watch(event);
            emit(event);
        });
    }

    pub fn take_claim(&mut self) -> Option<crate::runtime::Claim> {
        self.config.claim.take()
    }

    pub fn manager_home(&self) -> Option<PathBuf> {
        self.config.manager_home.clone()
    }

    // A fresh cancel token for the run about to start, the one the handle aborts.
    fn fresh_cancel(&mut self) {
        self.cancel = CancellationToken::new();
        self.handle.lock().cancel = self.cancel.clone();
    }

    pub(crate) fn tools(&self) -> &[Arc<dyn Tool>] { &self.config.tools }
    pub(crate) fn install_tools(&mut self, tools: Vec<Arc<dyn Tool>>, sections: IndexMap<String, String>) {
        self.config.tools = tools;
        self.config.sections = sections;
    }

    pub fn model(&self) -> &Model {
        &self.config.model
    }

    // Whether the last run was stopped by the user.
    pub fn aborted(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn thinking_level(&self) -> &str {
        &self.config.thinking_level
    }

    // Switch models for the next request. The prompt sections and compaction settings depend on
    // the model, so the caller resolves them again; the thinking level is clamped to the new
    // model. The session records both changes. Picking the model the session is already on records
    // nothing: a model change lets the prompt be brought up to date, which rewrites the cache, and a
    // frontend that sets the model it's already on after each run left a model change after every
    // answer in real sessions (2026-11-01).
    pub fn set_model(&mut self, model: Model, registry: ModelRegistry, sections: IndexMap<String, String>, compaction: CompactionSettings) -> std::io::Result<()> {
        if (&model.provider, &model.id) != (&self.config.model.provider, &self.config.model.id) {
            let base = self.session.next_base();
            self.session.append(SessionEntry::ModelChange { base, provider: model.provider.clone(), model_id: model.id.clone() })?;
        }
        self.config.model = Model { service_tier: service_tier(&self.session, &model), ..model };
        // A provider added since this session opened must bring its credential settings too.
        self.config.registry = registry;
        self.config.sections = sections;
        // The new model's compaction numbers, with the session's on/off kept (/autocompact).
        self.config.compaction = CompactionSettings { enabled: self.config.compaction.enabled, ..compaction };
        let level = self.config.thinking_level.clone();
        self.set_thinking_level(&level)
    }

    // A thinking level for the next request, clamped to what the model supports. Recorded only
    // when it changes.
    pub fn set_thinking_level(&mut self, level: &str) -> std::io::Result<()> {
        let level = provider::clamp_thinking_level(&self.config.model, level).to_string();
        if level != self.config.thinking_level {
            let base = self.session.next_base();
            self.session.append(SessionEntry::ThinkingLevelChange { base, thinking_level: level.clone() })?;
            self.config.thinking_level = level;
        }
        Ok(())
    }

    // Fast mode on or off from the next request on, on the APIs that offer it, kept in the session.
    pub fn set_fast(&mut self, on: bool) -> Result<(), String> {
        if !provider::FAST_APIS.contains(&self.config.model.api.as_str()) {
            return Err(format!("fast mode isn't offered on {}", self.config.model.provider));
        }
        self.session.append_custom(FAST_TYPE, Some(json!({"on": on}))).map_err(|error| format!("could not write the session: {error}"))?;
        self.config.model.service_tier = service_tier(&self.session, &self.config.model);
        Ok(())
    }

    pub fn auto_compaction(&self) -> bool {
        self.config.compaction.enabled
    }

    // Turn automatic compaction or retries on or off for this session (pi's set_auto_compaction,
    // set_auto_retry).
    pub fn set_auto_compaction(&mut self, enabled: bool) {
        self.config.compaction.enabled = enabled;
    }

    pub fn set_auto_retry(&mut self, enabled: bool) {
        self.config.retry.enabled = enabled;
    }

    pub fn set_retry(&mut self, retry: crate::config::RetrySettings) {
        self.config.retry = retry;
    }

    // Name the session (pi's set_session_name), as a session_info entry.
    pub fn set_name(&mut self, name: &str) -> std::io::Result<()> {
        let base = self.session.next_base();
        self.session.append(SessionEntry::SessionInfo { base, name: Some(name.to_string()) }).map(|_| ())
    }

    // Compact now, as the user asked, with their instructions for the summary. As in pi
    // (agent-session.ts compact), compaction_start always comes and a session with nothing to cut
    // gets a compaction_end that says why. Output: an error only when the session can't be
    // written; the outcome is in the compaction_end event.
    pub async fn compact(&mut self, instructions: Option<&str>) -> Result<(), String> {
        self.fresh_cancel();
        if compaction::prepare(&self.session, self.config.compaction).is_none() {
            let already = matches!(self.session.path_to_leaf().last(), Some(SessionEntry::Compaction { .. }));
            let why = if already { "Already compacted" } else { "Nothing to compact (session too small)" };
            (self.emit)(&Event::CompactionStart { reason: Reason::Manual });
            (self.emit)(&Event::CompactionEnd { reason: Reason::Manual, result: None, aborted: false, will_retry: false, error_message: Some(format!("Compaction failed: {why}")) });
            return Ok(());
        }
        self.run_auto_compaction(Reason::Manual, false, instructions).await.map(drop)
    }

    // The prompt the session runs with is built once and kept in it, as its system message, so
    // every request repeats it byte for byte: any change to it rewrites the provider's whole prompt
    // cache (a new build's wording did that to every running session on its next turn, 11.9M tokens
    // written in a day, 2026-11-01). It's built again only when what it's made from changes: the
    // session has none yet; it was compacted or put on another model since (each starts a new cache
    // anyway); or its AGENTS.md files now differ from the instructions its prompt holds, compared
    // file text to file text (prompt::instructions), so piccosta's wording around them doesn't
    // count. Until then a new build's wording, a tool's description or a changed scale wait.
    // Output: the system message to add, None when nothing differs.
    fn prompt_update(&mut self) -> Option<SystemMessage> {
        let event = self.session.path_to_leaf().into_iter().rev().find_map(|entry| match entry {
            SessionEntry::Message { message: AgentMessage::System(_), .. } => Some(false),
            SessionEntry::Compaction { .. } | SessionEntry::ModelChange { .. } => Some(true),
            _ => None,
        });
        let messages = crate::message::to_model_messages(&self.session.context().messages);
        let held = transcript::current_system_message(&messages).and_then(|system| system.sections?.shift_remove("project_context")?);
        let now = if self.config.context_files { crate::prompt::context_now(&self.config.cwd, &crate::config::agent_dir()) } else { Some(None) };
        let edited = now.as_ref().is_some_and(|project| crate::prompt::instructions(project.as_deref()) != crate::prompt::instructions(held.as_deref()));
        if !event.is_none_or(|event| event) && !edited {
            return None;
        }
        if let Some(project) = now {
            crate::prompt::set_project_context(&mut self.config.sections, project);
        }
        self.prompt_patch()
    }

    // What makes the prompt the session holds the one this build makes now: the sections that
    // differ (None for one that went), the tools added or changed, and the tools gone. None when
    // it's the same.
    fn prompt_patch(&self) -> Option<SystemMessage> {
        let current = transcript::current_system_message(&crate::message::to_model_messages(&self.session.context().messages));
        let held: IndexMap<String, Option<String>> = current.as_ref().and_then(|system| system.sections.clone()).unwrap_or_default();
        let held_tools = current.and_then(|system| system.tools_added).unwrap_or_default();
        let tools: Vec<ToolDefinition> = self.config.tools.iter().map(|tool| tool.definition()).collect();
        let mut patch: IndexMap<String, Option<String>> = self.config.sections.iter().filter(|(name, text)| held.get(*name) != Some(&Some((*text).clone()))).map(|(name, text)| (name.clone(), Some(text.clone()))).collect();
        patch.extend(held.keys().filter(|name| !self.config.sections.contains_key(*name)).map(|name| (name.clone(), None)));
        let added: Vec<ToolDefinition> = tools.iter().filter(|tool| !held_tools.contains(tool)).cloned().collect();
        let removed: Vec<ToolReference> = held_tools.iter().filter(|held| !tools.iter().any(|tool| tool.name == held.name)).map(|held| ToolReference { name: held.name.clone() }).collect();
        (!patch.is_empty() || !added.is_empty() || !removed.is_empty()).then(|| SystemMessage {
            content: TextOrBlocks::Text(String::new()),
            sections: (!patch.is_empty()).then_some(patch),
            tools_added: (!added.is_empty()).then_some(added),
            tools_removed: (!removed.is_empty()).then_some(removed),
            timestamp: clock::now_ms(),
        })
    }

    // Add a message to the session and announce it. Custom messages are their own entry type.
    fn add_message(&mut self, message: AgentMessage, new_messages: &mut Vec<AgentMessage>) -> Result<String, String> {
        // A message from the user starts fresh overflow recovery.
        if matches!(message, AgentMessage::User(_)) {
            self.overflow_recovery_attempted = false;
        }
        (self.emit)(&Event::MessageStart { message: message.clone() });
        (self.emit)(&Event::MessageEnd { message: message.clone() });
        let id = match &message {
            AgentMessage::Custom(custom) => self.session.append_custom_message(custom),
            _ => self.session.append_message(message.clone()),
        }
        .map_err(|error| format!("Could not write the session: {error}"))?;
        new_messages.push(message);
        Ok(id)
    }

    // Send the user's prompt, with any images (rpc mode's `images`), and run until the model is
    // done (and any retry or background job it waits on). An empty one with none answers only
    // what's queued in the session. Output: an error only when the session can't be written; model
    // failures end up in the session as failed assistant messages, as in pi.
    pub async fn prompt_with_images(&mut self, text: &str, images: Vec<Content>) -> Result<(), String> {
        // A prompt in a closed session opens it again.
        if let Some(file) = self.session.file() {
            crate::session::set_closed(&crate::config::agent_dir(), file, false);
        }
        let first = (!text.is_empty() || !images.is_empty()).then(|| with_images(text, images));
        self.run_from(first).await
    }

    pub fn set_gate(&mut self, gate: Gate) {
        self.config.gate = Some(gate);
    }

    // A background job's finish message, when one comes while nothing runs. A frontend waits on
    // this between prompts and hands what arrives to `take_notice`.
    pub async fn notice(&mut self) -> Option<CustomMessage> {
        self.config.notices.recv().await
    }

    // Releasing an idle runtime must not throw away notices or a prompt it hasn't taken yet.
    pub fn has_pending_work(&self) -> bool {
        let inbox = self.handle.lock();
        !self.config.notices.is_empty() || !self.held.is_empty() || !self.session.queued().is_empty() || !inbox.steer.is_empty() || !inbox.follow_up.is_empty()
    }

    // Run a turn on a background job's finish message, as longcmd does in pi when the session
    // is idle.
    pub async fn take_notice(&mut self, notice: CustomMessage) -> Result<(), String> {
        self.run_from(Some(AgentMessage::Custom(notice))).await
    }

    // Keep a job's message for the next prompt, which it goes in ahead of: a kill the user asked
    // for (/stop) is news for the model, not a reason to answer.
    pub fn hold_notice(&mut self, notice: CustomMessage) {
        self.held.push(notice);
    }

    // A run, starting with the messages queued in the session (sent while nothing ran it, or left
    // by a stop), then `first`.
    async fn run_from(&mut self, first: Option<AgentMessage>) -> Result<(), String> {
        let queued: Vec<AgentMessage> = self.session.queued().iter().map(|text| user_message(text)).collect();
        if queued.is_empty() && first.is_none() {
            return Ok(());
        }
        self.fresh_cancel();
        // However the run ends, an error included, it ends settled, so nothing waits on it after it's
        // over (a frontend's working state).
        let ran = self.turns(queued, first).await;
        (self.emit)(&Event::AgentSettled { aborted: ran.as_ref().is_ok_and(|aborted| *aborted) });
        ran.map(drop)
    }

    // The turns of one run (run_from): the queued messages and `first`, then whatever follows them
    // (retries, compaction, job messages, queued follow-ups). Output: whether it was aborted.
    async fn turns(&mut self, queued: Vec<AgentMessage>, first: Option<AgentMessage>) -> Result<bool, String> {
        // Compact first when the last response left the context past the threshold; the new
        // prompt replaces any retry that schedules (agent-session.ts _prepareRun, _startRun).
        if let Some((assistant, entry_id)) = self.last_projected_assistant() {
            self.check_compaction(&assistant, Some(&entry_id), false).await?;
        }
        let mut starting: Vec<AgentMessage> = self.held.drain(..).map(AgentMessage::Custom).collect();
        starting.extend(queued);
        starting.extend(first);
        let mut aborted = false;
        loop {
            if let Some(system) = self.prompt_update() {
                starting.insert(0, AgentMessage::System(system));
            }
            let outcome = self.run(std::mem::take(&mut starting)).await?;
            // The end of a run, which a Claude query still parked on tool calls doesn't outlive.
            provider::claude::end_run(self.session.id());
            crate::scale::after_turn();
            if outcome.aborted {
                aborted = true;
                break;
            }
            if let Some(failed) = outcome.retry {
                if !self.wait_to_retry(&failed).await? {
                    aborted = self.cancel.is_cancelled();
                    break;
                }
                continue;
            }
            // After the run: compact when the context filled up, and after an overflow ask again
            // on the compacted context (agent-session.ts _handlePostAgentRun).
            if let Some((assistant, entry_id)) = &outcome.last_assistant
                && self.check_compaction(assistant, Some(entry_id), true).await?
            {
                if self.cancel.is_cancelled() {
                    aborted = true;
                    break;
                }
                continue;
            }
            // A background job's message that came after the last turn starts a turn of its own.
            if let Ok(notice) = self.config.notices.try_recv() {
                starting.push(AgentMessage::Custom(notice));
                continue;
            }
            // What the user queued that no turn took (follow-ups, and steering that came after
            // the last turn) starts the next run.
            let queued = self.take_queued();
            if !queued.is_empty() {
                starting.extend(queued);
                continue;
            }
            if self.config.hold_for_jobs && self.config.tool_shared.jobs.running_count() > 0 {
                match self.wait_for_job().await {
                    Some(messages) => {
                        starting.extend(messages);
                        continue;
                    }
                    None => {
                        aborted = self.cancel.is_cancelled();
                        break;
                    }
                }
            }
            break;
        }
        Ok(aborted)
    }

    // The next of what's queued (steering first, then follow-ups), as user messages, and said.
    fn take_queued(&mut self) -> Vec<AgentMessage> {
        let queued = {
            let mut inbox = self.handle.lock();
            if inbox.steer.is_empty() {
                let all = inbox.follow_up_all;
                dequeue(&mut inbox.follow_up, all)
            } else {
                let all = inbox.steering_all;
                dequeue(&mut inbox.steer, all)
            }
        };
        if !queued.is_empty() {
            self.announce_queue();
        }
        queued.iter().map(|text| user_message(text)).collect()
    }

    // Print and json mode with a job still running after the model's last turn: wait for its
    // finish message (longcmd.ts:533-554), or a message queued meanwhile (one sent to its
    // runtime). PICCOSTA_LONGCMD_HOLD_S caps the wait, then the jobs are killed and
    // their messages say why. Output: what starts the next turn, None when the wait was stopped.
    async fn wait_for_job(&mut self) -> Option<Vec<AgentMessage>> {
        let shared = self.config.tool_shared.clone();
        let jobs = &shared.jobs;
        let cap = jobs.hold.map(|hold| tokio::time::Instant::now() + hold);
        let mut killed = false;
        let handle = self.handle.clone();
        loop {
            tokio::select! {
                notice = self.config.notices.recv() => return notice.map(|notice| vec![AgentMessage::Custom(notice)]),
                () = handle.0.1.notified() => {
                    let queued = self.take_queued();
                    if !queued.is_empty() {
                        return Some(queued);
                    }
                }
                () = self.cancel.cancelled() => return None,
                () = tokio::time::sleep_until(cap.unwrap_or_else(tokio::time::Instant::now)), if cap.is_some() && !killed => {
                    killed = true;
                    let hold = crate::tools::jobs::seconds_text(jobs.hold.unwrap_or_default());
                    jobs.kill_all(&format!("print mode stopped waiting after {hold}, {}", crate::tools::jobs::HOLD_ENV), false);
                }
            }
        }
    }

    // Before a retry: count it, say so, hide the failed response from context, and wait
    // (agent-session.ts:3787-3827). Output: false when the wait was aborted or attempts ran out.
    async fn wait_to_retry(&mut self, failed: &FailedResponse) -> Result<bool, String> {
        self.retry_attempt += 1;
        let retry = self.config.retry;
        if self.retry_attempt > retry.max_retries {
            (self.emit)(&Event::AutoRetryEnd { success: false, attempt: self.retry_attempt - 1, final_error: Some(failed.error.clone()) });
            self.retry_attempt = 0;
            return Ok(false);
        }
        let delay = retry::delay_ms(retry.base_delay_ms, retry.max_delay_ms, self.retry_attempt);
        (self.emit)(&Event::AutoRetryStart { attempt: self.retry_attempt, max_attempts: retry.max_retries, delay_ms: delay, error_message: failed.error.clone() });
        self.omit_attempt(&failed.entry_id, &failed.message)?;
        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(delay)) => Ok(true),
            () = self.cancel.cancelled() => {
                (self.emit)(&Event::AutoRetryEnd { success: false, attempt: self.retry_attempt, final_error: Some("Retry cancelled".into()) });
                self.retry_attempt = 0;
                Ok(false)
            }
        }
    }

    // One agent_start..agent_end cycle: the starting messages, then turns until the model stops
    // calling tools and nothing is queued (agent-loop.ts runLoop).
    async fn run(&mut self, starting: Vec<AgentMessage>) -> Result<RunOutcome, String> {
        let mut new_messages = Vec::new();
        self.repeated_call = (String::new(), 0);
        (self.emit)(&Event::AgentStart);
        (self.emit)(&Event::TurnStart);
        for message in starting {
            self.add_message(message, &mut new_messages)?;
        }
        loop {
            // Before each response, with all it will carry in the session (the starting messages,
            // steering, job notices): compact if it wouldn't fit (agent-session.ts
            // _compactBeforeNextAssistantResponse).
            self.compact_before_next_response().await?;
            let (assistant, entry_id) = self.respond().await?;
            new_messages.push(AgentMessage::Assistant(assistant.clone()));

            if matches!(assistant.stop_reason, StopReason::Error | StopReason::Aborted) {
                (self.emit)(&Event::TurnEnd { message: AgentMessage::Assistant(assistant.clone()), tool_results: Vec::new() });
                let will_retry = self.will_retry(&assistant);
                (self.emit)(&Event::AgentEnd { messages: new_messages, will_retry });
                let aborted = assistant.stop_reason == StopReason::Aborted;
                let retry = will_retry.then(|| FailedResponse {
                    entry_id: entry_id.clone(),
                    error: assistant.error_message.clone().unwrap_or_default(),
                    message: assistant.clone(),
                });
                if !will_retry && !aborted && self.retry_attempt > 0 {
                    (self.emit)(&Event::AutoRetryEnd { success: false, attempt: self.retry_attempt, final_error: assistant.error_message.clone() });
                    self.retry_attempt = 0;
                }
                return Ok(RunOutcome { aborted, retry, last_assistant: Some((assistant, entry_id)) });
            }
            if self.retry_attempt > 0 {
                (self.emit)(&Event::AutoRetryEnd { success: true, attempt: self.retry_attempt, final_error: None });
                self.retry_attempt = 0;
            }

            let has_calls = assistant.content.iter().any(|block| matches!(block, Content::ToolCall { .. }));
            let mut results = Vec::new();
            if has_calls {
                let mut runs = if assistant.stop_reason == StopReason::Length { self.refuse_truncated_calls(&assistant) } else { self.execute_tools(&assistant).await };
                self.append_web_results()?;
                // An overused machine goes on the end of the last result, where the model reads it
                // with what it just ran and the cached prefix stays as it was.
                if let (Some(note), Some(Ok(output))) = (crate::machine::note(&self.config.cwd, &mut self.machine_noted), runs.last_mut().map(|run| &mut run.output)) {
                    output.content.push(Content::text(note));
                }
                for run in runs {
                    let message = tool_result_message(run);
                    self.add_message(AgentMessage::ToolResult(message.clone()), &mut new_messages)?;
                    results.push(AgentMessage::ToolResult(message));
                }
            }
            (self.emit)(&Event::TurnEnd { message: AgentMessage::Assistant(assistant.clone()), tool_results: results });

            // The call loop guard: the same call again and again ends the run, with a note to the
            // user and the model saying which.
            for block in &assistant.content {
                if let Content::ToolCall { name, arguments, .. } = block {
                    let call = format!("{name} {}", Value::Object(arguments.clone()));
                    self.repeated_call = if self.repeated_call.0 == call { (call, self.repeated_call.1 + 1) } else { (call, 1) };
                }
            }
            if self.repeated_call.1 >= REPEATED_CALL_LIMIT {
                let text = format!("piccosta stopped the run: the same call was made {} times in a row ({}).", self.repeated_call.1, crate::tools::jobs::command_preview(&self.repeated_call.0));
                let note = CustomMessage { custom_type: LOOP_NOTE_TYPE.into(), content: TextOrBlocks::Text(text), display: true, details: None, timestamp: clock::now_ms() };
                self.add_message(AgentMessage::Custom(note), &mut new_messages)?;
                (self.emit)(&Event::AgentEnd { messages: new_messages, will_retry: false });
                return Ok(RunOutcome { aborted: false, retry: None, last_assistant: Some((assistant, entry_id)) });
            }

            let aborted = self.cancel.is_cancelled();
            // Background jobs that finished during the turn, and what the user queued to steer,
            // go into the next one.
            let steering = if aborted { Vec::new() } else { self.steering() };
            if aborted || (!has_calls && steering.is_empty()) {
                (self.emit)(&Event::AgentEnd { messages: new_messages, will_retry: false });
                return Ok(RunOutcome { aborted, retry: None, last_assistant: Some((assistant, entry_id)) });
            }
            (self.emit)(&Event::TurnStart);
            // What's left queued, just before the queued messages start, where pi says it.
            if steering.iter().any(|message| matches!(message, AgentMessage::User(_))) {
                self.announce_queue();
            }
            for message in steering {
                self.add_message(message, &mut new_messages)?;
            }
        }
    }

    // What steers the next turn: background jobs' finish messages, then the user's steering.
    fn steering(&mut self) -> Vec<AgentMessage> {
        let mut messages: Vec<AgentMessage> = std::iter::from_fn(|| self.config.notices.try_recv().ok()).map(AgentMessage::Custom).collect();
        let steer = {
            let mut inbox = self.handle.lock();
            let all = inbox.steering_all;
            dequeue(&mut inbox.steer, all)
        };
        messages.extend(steer.iter().map(|text| user_message(text)));
        messages
    }

    fn announce_queue(&mut self) {
        let (steering, follow_up) = self.handle.queued();
        (self.emit)(&Event::QueueUpdate { steering, follow_up });
    }

    fn will_retry(&self, assistant: &AssistantMessage) -> bool {
        let retry = self.config.retry;
        retry.enabled
            && retry::is_retryable(assistant)
            && !retry::is_context_overflow(assistant, self.config.model.context_window)
            && self.retry_attempt < retry.max_retries
    }

    // Ask the model for one response and stream it out as events. Output: the response, already
    // in the session, and its entry id.
    async fn respond(&mut self) -> Result<(AssistantMessage, String), String> {
        // A /fast from during the run applies here; one this model doesn't take waits for the run's end.
        if let Some(on) = self.handle.swap_fast(None) && self.set_fast(on).is_err() {
            self.handle.swap_fast(Some(on));
        }
        let messages = crate::message::to_model_messages(&self.session.context().messages);
        let model = &self.config.model;
        let credential = auth::credential_for(&model.provider, self.config.registry.provider_config(&model.provider), &self.config.credentials).await;
        let message = match credential {
            Ok(credential) => {
                let (sender, mut events) = mpsc::unbounded_channel();
                // The stream loop guard stops the request through its own token, so a stopped loop
                // isn't taken for the user's abort.
                let guard = self.cancel.child_token();
                let response = provider::stream(Self::request(&self.config, &self.session, &messages, &credential, guard.clone()), sender);
                tokio::pin!(response);
                let emit = &mut self.emit;
                let mut started = false;
                // pi sends the partial message's usage, which these APIs fill in only at the end:
                // zeros while streaming.
                let handle = |event: StreamEvent, emit: &mut EventSink, started: &mut bool| match stream_event_json(&event) {
                    None => {
                        *started = true;
                        let mut partial = provider::new_output(model, &model.api);
                        partial.stop_reason = StopReason::Stop;
                        emit(&Event::MessageStart { message: AgentMessage::Assistant(partial) });
                    }
                    Some(json) => emit(&Event::MessageUpdate { usage: Usage::default(), assistant_message_event: json }),
                };
                let (mut streamed, mut checked_at, mut looped) = (String::new(), 0, None);
                let mut message = loop {
                    tokio::select! {
                        biased;
                        Some(event) = events.recv() => {
                            if let StreamEvent::TextDelta { delta, .. } | StreamEvent::ThinkingDelta { delta, .. } = &event {
                                streamed.push_str(delta);
                                if looped.is_none() && streamed.len() >= checked_at + LOOP_CHECK_EVERY {
                                    checked_at = streamed.len();
                                    looped = looping(&streamed);
                                    if looped.is_some() {
                                        guard.cancel();
                                    }
                                }
                            }
                            handle(event, emit, &mut started)
                        }
                        message = &mut response => break message,
                    }
                };
                while let Ok(event) = events.try_recv() {
                    handle(event, emit, &mut started);
                }
                if !started {
                    emit(&Event::MessageStart { message: AgentMessage::Assistant(message.clone()) });
                }
                // A stopped loop is the model's failure, not the user's abort, and not one to retry.
                if let Some(unit) = looped.filter(|_| !self.cancel.is_cancelled()) {
                    message.stop_reason = StopReason::Error;
                    message.error_message = Some(format!("Stopped a looping response: the model kept repeating {:?}", unit.trim()));
                }
                message
            }
            Err(error) => {
                let mut failed = provider::new_output(model, &model.api);
                failed.stop_reason = StopReason::Error;
                failed.error_message = Some(error);
                failed.thinking_level = Some(self.config.thinking_level.clone());
                (self.emit)(&Event::MessageStart { message: AgentMessage::Assistant(failed.clone()) });
                failed
            }
        };
        // A Claude model's first turn can learn that it gets 200K, not 1M (claude::stream); the
        // session compacts against that from here.
        let model = &mut self.config.model;
        if model.provider == provider::claude::PROVIDER {
            model.context_window = model.context_window.min(provider::claude::context_window(&model.id, model.max_tokens));
        }
        (self.emit)(&Event::MessageEnd { message: AgentMessage::Assistant(message.clone()) });
        let id = self.session.append_message(AgentMessage::Assistant(message.clone())).map_err(|error| format!("Could not write the session: {error}"))?;
        // The cache monitor: a response that had to write most of its context to the cache again
        // is recorded with its cause, and the session's views show it as one line.
        if let Some(miss) = crate::cache::check(&self.session, &id, provider::claude::take_rewrite(self.session.id())) {
            let entry = SessionEntry::Custom { base: self.session.next_base(), custom_type: crate::cache::MISS_TYPE.into(), data: Some(miss) };
            self.session.append(entry.clone()).map_err(|error| format!("Could not write the session: {error}"))?;
            (self.emit)(&Event::EntryAppended { entry });
        }
        // The bridge's one notice, that Claude Code is newer than the version it was checked against.
        if let Some(text) = provider::claude::take_notice(self.session.id()) {
            let entry = SessionEntry::Custom { base: self.session.next_base(), custom_type: provider::claude::NOTICE_TYPE.into(), data: Some(json!({"text": text})) };
            self.session.append(entry.clone()).map_err(|error| format!("Could not write the session: {error}"))?;
            (self.emit)(&Event::EntryAppended { entry });
        }
        // A response that neither failed nor ran out of room ends any overflow recovery.
        if !matches!(message.stop_reason, StopReason::Error | StopReason::Length) {
            self.overflow_recovery_attempted = false;
        }
        Ok((message, id))
    }

    // A request on the session's own context, the same for each turn and for idle compaction: its
    // model, level, transport, files and cache key, so the provider keeps the one cached prefix.
    fn request<'a>(config: &'a AgentConfig, session: &'a Session, messages: &'a [crate::message::Message], credential: &'a auth::Credential, cancel: CancellationToken) -> Request<'a> {
        Request { model: &config.model, messages, credential, session_id: Some(session.id()), cache: true, in_place: false, manager: config.manager_home.is_some(), transport: &config.transport,
            thinking_level: &config.thinking_level, session_file: session.file(), cwd: &config.cwd, cancel }
    }

    // When this session compacts while idle (runtime.rs keeps the time), as Claude Code's
    // idleCompaction does: IDLE_SHARE of the way from its last request to the expiry of the prompt
    // cache that request wrote, so the next turn sends the summary rather than writing the whole
    // context to the cache again. Output: that time and the request's (epoch ms), or why not:
    // compaction or compaction.idle is off; the last request failed or was stopped, or the context
    // changed after it (a compaction, another model or level, a new prompt), so its cache isn't what
    // the next request reads; the provider's cache life isn't known, or ends before a request would get
    // there; on the Claude bridge, no query of Claude Code's waits with that cache; the context is under
    // compaction.idleMinTokens; or the model's plan is running low.
    pub fn idle_compaction_due(&self) -> Result<(i64, i64), String> {
        let (settings, model) = (self.config.compaction, &self.config.model);
        if !settings.enabled || !settings.idle {
            return Err("compaction or compaction.idle is off".into());
        }
        let branch = self.session.path_to_leaf();
        let found = branch.iter().rposition(|entry| matches!(entry, SessionEntry::Message { message: AgentMessage::Assistant(_), .. }));
        let Some((at, SessionEntry::Message { message: AgentMessage::Assistant(last), .. })) = found.map(|at| (at, branch[at])) else { return Err("no request yet".into()) };
        if matches!(last.stop_reason, StopReason::Error | StopReason::Aborted) {
            return Err("the last request failed or was stopped, so what it cached isn't known".into());
        }
        let changed = |entry: &&SessionEntry| matches!(entry, SessionEntry::Compaction { .. } | SessionEntry::ModelChange { .. } | SessionEntry::ThinkingLevelChange { .. }
            | SessionEntry::Message { message: AgentMessage::System(_), .. });
        if branch[at + 1..].iter().any(changed) {
            return Err("the cache is cold: the context changed since the last request".into());
        }
        let Some(ttl) = settings.cache_seconds.map(Duration::from_secs).or_else(|| provider::cache_ttl(model, &last.usage)) else {
            return Err(format!("{}'s cache life isn't known (compaction.cacheSeconds sets one)", model.provider));
        };
        let expires = last.timestamp + ttl.as_millis() as i64;
        if crate::clock::now_ms() + IDLE_SEND_MS >= expires {
            return Err(format!("the cache expires at {}, before a request would get there", crate::clock::iso_from_ms(expires)));
        }
        if model.provider == provider::claude::PROVIDER && !provider::claude::waiting(self.session.id()) {
            return Err("no Claude Code query waits with this session's context, and a new one would write all of it again".into());
        }
        let tokens = compaction::estimate_projected_context_tokens(&self.session.projection(), &branch).tokens;
        if tokens < settings.idle_min_tokens {
            return Err(format!("{tokens} tokens of context, under compaction.idleMinTokens ({})", settings.idle_min_tokens));
        }
        if crate::scale::low(&crate::config::agent_dir(), &format!("{}/{}", model.provider, model.id)) {
            return Err("the model's plan is running low".into());
        }
        Ok((last.timestamp + (ttl.as_millis() as f64 * IDLE_SHARE) as i64, last.timestamp))
    }

    // Compact now, while idle, the summary asked for in the session's own request (above).
    pub async fn compact_idle(&mut self) -> Result<(), String> {
        self.fresh_cancel();
        self.run_auto_compaction(Reason::Idle, false, None).await.map(drop)
    }

    // The newest response in the model's context, and the entry it came from.
    fn last_projected_assistant(&self) -> Option<(AssistantMessage, String)> {
        self.session.projection().iter().rev().find_map(|entry| {
            entry.messages.iter().rev().find_map(|message| match message {
                AgentMessage::Assistant(assistant) => Some((assistant.clone(), entry.source.id().to_string())),
                _ => None,
            })
        })
    }

    // Hide a failed attempt from the model's context for good, with the results of any tools it
    // called, so a result never outlives its call (agent-session.ts _omitRecoveryAttempt). The
    // entries stay in the file; context edits leave them out.
    fn omit_attempt(&mut self, entry_id: &str, assistant: &AssistantMessage) -> Result<(), String> {
        let call_ids: Vec<&str> = assistant
            .content
            .iter()
            .filter_map(|block| match block {
                Content::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let mut targets = vec![entry_id.to_string()];
        let projection = self.session.projection();
        if let Some(position) = projection.iter().position(|entry| entry.source.id() == entry_id) {
            'after: for entry in &projection[position + 1..] {
                for message in &entry.messages {
                    match message {
                        AgentMessage::Assistant(_) => break 'after,
                        AgentMessage::ToolResult(result) if call_ids.contains(&result.tool_call_id.as_str()) => {
                            targets.push(entry.source.id().to_string());
                        }
                        _ => {}
                    }
                }
            }
        }
        drop(projection);
        for target_id in targets {
            let entry = SessionEntry::ContextEdit { base: self.session.next_base(), target_id, replacement: None };
            self.session.append(entry.clone()).map_err(|error| format!("Could not write the session: {error}"))?;
            (self.emit)(&Event::EntryAppended { entry });
        }
        Ok(())
    }

    // Whether a response calls for compaction, and run it if so (agent-session.ts
    // _checkCompaction). Three cases: an overflow or a length stop short of the model's output
    // limit is compacted and asked again once; a successful response that overflowed is compacted
    // and kept; and a context past the threshold is compacted. Input: the response, its entry,
    // and whether an aborted response is skipped (after a run) or checked (before a prompt).
    // Output: whether the caller should ask the model again.
    async fn check_compaction(&mut self, assistant: &AssistantMessage, entry_id: Option<&str>, skip_aborted: bool) -> Result<bool, String> {
        let settings = self.config.compaction;
        if !settings.enabled || (skip_aborted && assistant.stop_reason == StopReason::Aborted) {
            return Ok(false);
        }
        // A response from another model says nothing about this one's window: switching from a
        // small-context model to a larger one shouldn't compact for the old one's overflow.
        let model = &self.config.model;
        let same_model = model.provider == assistant.provider && model.id == assistant.model;
        let context_window = model.context_window;
        let max_tokens = model.max_tokens;

        let branch = self.session.path_to_leaf();
        // A response from before the latest compaction measured a context that's gone.
        let compaction_ms = branch
            .iter()
            .rev()
            .find(|entry| matches!(entry, SessionEntry::Compaction { .. }))
            .map(|entry| clock::ms_from_iso(&entry.base().timestamp).unwrap_or(0));
        if compaction_ms.is_some_and(|compacted| assistant.timestamp <= compacted) {
            return Ok(false);
        }

        let projection = self.session.projection();
        let is_projected = entry_id.is_none_or(|id| {
            projection.iter().any(|entry| entry.source.id() == id && entry.messages.iter().any(|message| matches!(message, AgentMessage::Assistant(_))))
        });
        let after: &[&SessionEntry] = match entry_id.and_then(|id| branch.iter().position(|entry| entry.id() == id)) {
            Some(index) => &branch[index + 1..],
            None => &[],
        };
        let edited_after = after.iter().any(|entry| matches!(entry, SessionEntry::ContextEdit { .. }));
        let latest_edit_of_it = after.iter().rev().find_map(|entry| match entry {
            SessionEntry::ContextEdit { target_id, replacement, .. } if Some(target_id.as_str()) == entry_id => Some(replacement),
            _ => None,
        });
        // An explicit overflow is recovered unless the attempt was already hidden or compacted away.
        let retained_for_recovery = entry_id.is_none()
            || (!after.iter().any(|entry| matches!(entry, SessionEntry::Compaction { .. }))
                && latest_edit_of_it.is_none_or(|replacement| replacement.is_some()));
        let usage_matches_projection = is_projected && !edited_after;
        let explicit_overflow = assistant.stop_reason == StopReason::Error && retry::is_context_overflow(assistant, 0);
        let context_overflow = same_model
            && ((explicit_overflow && retained_for_recovery)
                || (usage_matches_projection && retry::is_context_overflow(assistant, context_window)));
        let recoverable_length = same_model && is_projected && retry::is_recoverable_length(assistant, max_tokens);

        if context_overflow || recoverable_length {
            drop(projection);
            drop(branch);
            // A completed response can't be continued, so it's kept and only the context shrinks.
            let will_retry = assistant.stop_reason != StopReason::Stop;
            if !will_retry {
                return self.run_auto_compaction(Reason::Overflow, false, None).await;
            }
            if self.overflow_recovery_attempted {
                let error_message = if context_overflow {
                    "Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or switching to a larger-context model."
                } else {
                    "Truncated response recovery failed after one compact-and-retry attempt."
                };
                (self.emit)(&Event::CompactionEnd {
                    reason: Reason::Overflow,
                    result: None,
                    aborted: false,
                    will_retry: false,
                    error_message: Some(error_message.into()),
                });
                return Ok(false);
            }
            self.overflow_recovery_attempted = true;
            if let Some(id) = entry_id {
                self.omit_attempt(id, assistant)?;
            }
            return self.run_auto_compaction(Reason::Overflow, true, None).await;
        }

        // The threshold. A failed or zero-usage response reports no usable size, so the size is
        // estimated from the last response that did; after a context edit, from the projection.
        let has_context_edits = projection.iter().any(|entry| matches!(entry.source, SessionEntry::ContextEdit { .. }));
        let direct_tokens = provider::estimate::context_tokens(&assistant.usage);
        let context_tokens = if has_context_edits {
            compaction::estimate_projected_context_tokens(&projection, &branch).tokens
        } else if assistant.stop_reason == StopReason::Error || direct_tokens == 0 {
            let messages: Vec<AgentMessage> = projection.iter().flat_map(|entry| entry.messages.iter().cloned()).collect();
            let estimate = compaction::estimate_context_tokens(&messages);
            // Usage from a response kept across the last compaction still measures the old,
            // larger context and would compact again right away.
            if let (Some(index), Some(compacted)) = (estimate.last_usage_index, compaction_ms)
                && let AgentMessage::Assistant(usage_source) = &messages[index]
                && usage_source.timestamp <= compacted
            {
                return Ok(false);
            }
            estimate.tokens
        } else {
            direct_tokens
        };
        drop(projection);
        drop(branch);
        if compaction::should_compact(context_tokens, context_window, settings) {
            return self.run_auto_compaction(Reason::Threshold, false, None).await;
        }
        Ok(false)
    }

    // Right before a response, inside a run (its messages and tool results all in): compact only
    // when the request wouldn't fit, since a compaction rewrites the whole prompt cache (on the Claude
    // bridge, all of it). So compactAt waits for the turn's end, and the room kept for the reply
    // is reserveTokens, except on the bridge, whose window already keeps it (claude context_window).
    // An estimate that falls short still gets the overflow's compact-and-retry after the run.
    async fn compact_before_next_response(&mut self) -> Result<(), String> {
        let context_window = self.config.model.context_window;
        if context_window == 0 {
            return Ok(());
        }
        let tokens = {
            let projection = self.session.projection();
            let branch = self.session.path_to_leaf();
            compaction::estimate_projected_context_tokens(&projection, &branch).tokens
        };
        let reply_room = if self.config.model.provider == provider::claude::PROVIDER { 0 } else { self.config.compaction.reserve_tokens };
        let last_resort = CompactionSettings { compact_at: None, reserve_tokens: reply_room, ..self.config.compaction };
        if compaction::should_compact(tokens, context_window, last_resort) {
            self.run_auto_compaction(Reason::Threshold, false, None).await?;
        }
        Ok(())
    }

    // Compact now (agent-session.ts _runAutoCompaction): summarize what's before the cut, write
    // the compaction entry with the user's own words pinned onto pi's summary (pintask.rs), and
    // report it. A failure is reported and leaves the session as it was. Output: whether to ask the
    // model again, which only an overflow recovery does.
    async fn run_auto_compaction(&mut self, reason: Reason, will_retry: bool, instructions: Option<&str>) -> Result<bool, String> {
        let Some(mut preparation) = compaction::prepare(&self.session, self.config.compaction) else { return Ok(false) };
        (self.emit)(&Event::CompactionStart { reason });
        let pinning = pintask::enabled() || self.config.manager_home.is_some();
        let previous_status = if pinning { pintask::strip_previous_block(&mut preparation, &self.session.path_to_leaf()) } else { "none" };

        let cancel = self.cancel.child_token();
        let model = &self.config.model;
        let credential = auth::credential_for(&model.provider, self.config.registry.provider_config(&model.provider), &self.config.credentials).await;
        let outcome = match credential {
            Err(error) => Err(error),
            Ok(credential) if reason == Reason::Idle => {
                let messages = crate::message::to_model_messages(&self.session.context().messages);
                compaction::summarize_in_place(&preparation, Self::request(&self.config, &self.session, &messages, &credential, cancel.clone())).await
            }
            Ok(credential) => {
                let request = compaction::SummaryRequest {
                    model,
                    credential: &credential,
                    thinking_level: &self.config.thinking_level,
                    retry: self.config.retry,
                    transport: &self.config.transport,
                    cwd: &self.config.cwd,
                    cancel: cancel.clone(),
                };
                let emit = &mut self.emit;
                let mut notify = |notice: RetryNotice| {
                    emit(&match notice {
                        RetryNotice::Scheduled { attempt, max_attempts, delay_ms, error_message } => {
                            Event::SummarizationRetryScheduled { attempt, max_attempts, delay_ms, error_message }
                        }
                        RetryNotice::AttemptStart => Event::SummarizationRetryAttemptStart { source: "compaction", reason },
                        RetryNotice::Finished => Event::SummarizationRetryFinished,
                    })
                };
                compaction::summarize(&preparation, instructions, &request, &mut notify).await
            }
        };

        let summary = match outcome {
            Ok(summary) if !cancel.is_cancelled() => summary,
            outcome => {
                let aborted = cancel.is_cancelled();
                let error_message = match outcome {
                    Err(error) if !aborted => Some(match reason {
                        Reason::Overflow => format!("Context overflow recovery failed: {error}"),
                        Reason::Manual => format!("Compaction failed: {error}"),
                        _ => format!("Auto-compaction failed: {error}"),
                    }),
                    _ => None,
                };
                (self.emit)(&Event::CompactionEnd { reason, result: None, aborted, will_retry: false, error_message });
                return Ok(false);
            }
        };

        let write_error = |error: std::io::Error| format!("Could not write the session: {error}");
        let (budget, ledger) = match &self.config.manager_home {
            Some(home) => (Some(crate::manager::MANAGER_PIN_CHARS), crate::manager::ledger_view(home)),
            None => (None, String::new()),
        };
        let pin = pinning.then(|| pintask::pin(&self.session.path_to_leaf(), &preparation, reason, &summary.summary, previous_status, budget, &ledger));
        let stored_summary = format!("{}{}", summary.summary, pin.as_ref().map_or("", |pin| pin.block.as_str()));
        let compaction_id = self
            .session
            .append_compaction(
                stored_summary.clone(),
                summary.first_kept_entry_id.clone(),
                summary.tokens_before,
                Some(summary.details.clone()),
                Some(false),
                Some(summary.usage.clone()),
            )
            .map_err(write_error)?;
        // The compacted context starts a new cache anyway, so the prompt is brought up to date now,
        // before the first response on it: a compaction inside a tool loop goes straight on to the
        // next response, and an update left for the next run would rewrite the cache a second time.
        if let Some(system) = self.prompt_update() {
            let message = AgentMessage::System(system);
            (self.emit)(&Event::MessageStart { message: message.clone() });
            (self.emit)(&Event::MessageEnd { message: message.clone() });
            self.session.append_message(message).map_err(write_error)?;
        }
        if let Some(pin) = &pin {
            let stats_id = self.session.append_custom(pintask::STATS_ENTRY, Some(pin.stats(&compaction_id))).map_err(write_error)?;
            if let Some(entry) = self.session.entry(&stats_id).cloned() {
                (self.emit)(&Event::EntryAppended { entry });
            }
        }
        let estimated_tokens_after: u64 = self.session.context().messages.iter().map(compaction::estimate_tokens).sum();
        let result = json!({
            "summary": stored_summary,
            "firstKeptEntryId": summary.first_kept_entry_id,
            "tokensBefore": summary.tokens_before,
            "estimatedTokensAfter": estimated_tokens_after,
            "usage": summary.usage,
            "details": summary.details,
        });
        (self.emit)(&Event::CompactionEnd { reason, result: Some(result), aborted: false, will_retry, error_message: None });
        Ok(will_retry)
    }

    // What the web tools stored during the batch, as "web-search-results" custom entries ahead of
    // the batch's results (pi-web-access appends them while its tools run).
    fn append_web_results(&mut self) -> Result<(), String> {
        for data in self.config.tool_shared.web.store.take_entries() {
            let id = self.session.append_custom(tools::web::store::SESSION_ENTRY_TYPE, Some(data)).map_err(|error| format!("Could not write the session: {error}"))?;
            if let Some(entry) = self.session.entry(&id).cloned() {
                (self.emit)(&Event::EntryAppended { entry });
            }
        }
        Ok(())
    }

    fn tool_context<'a>(&self, session: &'a Session) -> ToolContext<'a> {
        ToolContext {
            session: Some(session),
            call_id: String::new(),
            cwd: self.config.cwd.clone(),
            session_id: self.session.id().to_string(),
            session_file: self.session.file().map(PathBuf::from),
            provider: self.config.model.provider.clone(),
            model_id: self.config.model.id.clone(),
            thinking_level: self.config.thinking_level.clone(),
            model_accepts_images: self.config.model.input.iter().any(|kind| kind == "image"),
            image_limits: tools::image::Limits::of(self.config.model.input_limits.as_ref()),
            cancel: self.cancel.clone(),
            updates: None,
            shared: self.config.tool_shared.clone(),
        }
    }

    fn refuse_truncated_calls(&mut self, assistant: &AssistantMessage) -> Vec<ToolRun> {
        let mut runs = Vec::new();
        for (index, block) in assistant.content.iter().enumerate() {
            let Content::ToolCall { id, name, arguments, .. } = block else { continue };
            (self.emit)(&Event::ToolExecutionStart { tool_call_id: id.clone(), tool_name: name.clone(), args: arguments.clone() });
            let run = ToolRun { index, call_id: id.clone(), name: name.clone(), output: Err(truncated_call_text(name)), duration_ms: None };
            Self::emit_tool_end(&mut self.emit, &run);
            runs.push(run);
        }
        runs
    }

    fn emit_tool_end(emit: &mut EventSink, run: &ToolRun) {
        let (result, is_error) = match &run.output {
            Ok(output) => (result_json(&output.content, &output.details), output.is_error),
            Err(message) => (result_json(&[Content::text(message.clone())], &Some(json!({}))), true),
        };
        emit(&Event::ToolExecutionEnd {
            tool_call_id: run.call_id.clone(),
            tool_name: run.name.clone(),
            result,
            is_error,
            duration_ms: run.duration_ms,
        });
    }

    // Run one response's tool calls (agent-loop.ts executeToolCallsParallel). Each call is
    // announced and checked in order; unknown tools and bad arguments fail right there. The rest
    // run at once, ending in whatever order they finish; the results come back in call order.
    async fn execute_tools(&mut self, assistant: &AssistantMessage) -> Vec<ToolRun> {
        let mut finished: Vec<Option<ToolRun>> = Vec::new();
        let mut pending = FuturesUnordered::new();
        let (update_sender, mut updates) = mpsc::unbounded_channel::<(usize, ToolUpdate)>();
        let mut calls: Vec<(String, String, Map<String, Value>)> = Vec::new();
        for block in &assistant.content {
            let Content::ToolCall { id, name, arguments, arguments_error, .. } = block else { continue };
            let slot = finished.len();
            finished.push(None);
            calls.push((id.clone(), name.clone(), arguments.clone()));
            (self.emit)(&Event::ToolExecutionStart { tool_call_id: id.clone(), tool_name: name.clone(), args: arguments.clone() });
            // An unknown tool, bad arguments or an abort fails the call right here.
            let checked = match self.config.tools.iter().find(|tool| tool.name() == name).cloned() {
                None => Err(format!("Tool {name} not found")),
                Some(_) if arguments_error.is_some() => Err(format!("Invalid arguments for tool \"{name}\": {}", arguments_error.as_deref().unwrap_or_default())),
                Some(tool) => tools::validate::validate(name, &tool.definition().parameters, &tool.prepare_arguments(arguments.clone()))
                    .and_then(|validated| if self.cancel.is_cancelled() { Err("Operation aborted".to_string()) } else { Ok((tool, validated)) }),
            };
            // The gate's answer for a call that passed the checks; an abort while it waits is an
            // abort.
            let checked = match (checked, self.config.gate.clone()) {
                (Ok((tool, validated)), Some(gate)) => tokio::select! {
                    allowed = gate(id.clone(), name.clone(), validated.clone(), tool.approval(&validated), tool.read_only()) => allowed.map(|()| (tool, validated)),
                    () = self.cancel.cancelled() => Err("Operation aborted".to_string()),
                },
                (Ok((tool, validated)), None) => match tool.approval(&validated) {
                    Some(question) => Err(format!("{question} No user approval is available here; the call was refused.")),
                    None => Ok((tool, validated)),
                },
                (checked, _) => checked,
            };
            let (tool, validated) = match checked {
                Ok(ready) => ready,
                Err(output) => {
                    let run = ToolRun { index: slot, call_id: id.clone(), name: name.clone(), output: Err(output), duration_ms: None };
                    Self::emit_tool_end(&mut self.emit, &run);
                    finished[slot] = Some(run);
                    continue;
                }
            };
            // Each call gets its own update channel, tagged with its slot on the way out.
            let (sender, mut receiver) = mpsc::unbounded_channel::<ToolUpdate>();
            let forward = update_sender.clone();
            tokio::spawn(async move {
                while let Some(update) = receiver.recv().await {
                    if forward.send((slot, update)).is_err() {
                        break;
                    }
                }
            });
            let context = ToolContext { call_id: id.clone(), updates: Some(sender), ..self.tool_context(&self.session) };
            let (call_id, call_name) = (id.clone(), name.clone());
            pending.push(async move {
                let started = Instant::now();
                let output = tool.execute(validated, &context).await;
                ToolRun { index: slot, call_id, name: call_name, output, duration_ms: Some(started.elapsed().as_millis() as u64) }
            });
        }
        drop(update_sender);

        while !pending.is_empty() {
            tokio::select! {
                Some(run) = pending.next() => {
                    Self::emit_tool_end(&mut self.emit, &run);
                    let index = run.index;
                    finished[index] = Some(run);
                }
                Some((slot, update)) = updates.recv() => {
                    let (call_id, name, args) = &calls[slot];
                    (self.emit)(&Event::ToolExecutionUpdate {
                        tool_call_id: call_id.clone(),
                        tool_name: name.clone(),
                        args: args.clone(),
                        partial_result: result_json(&update.content, &update.details),
                    });
                }
            }
        }
        finished.into_iter().flatten().collect()
    }
}

struct FailedResponse {
    entry_id: String,
    error: String,
    message: AssistantMessage,
}

struct RunOutcome {
    aborted: bool,
    retry: Option<FailedResponse>,
    // The run's last response and its entry, for the compaction check after it.
    last_assistant: Option<(AssistantMessage, String)>,
}

// The tool result message for a finished call (agent-loop.ts:934-950). A failure's text is its
// error message, with empty details, as pi records it.
fn tool_result_message(run: ToolRun) -> ToolResultMessage {
    let (content, details, is_error) = match run.output {
        Ok(output) => (output.content, output.details, output.is_error),
        Err(message) => (vec![Content::text(message)], Some(json!({})), true),
    };
    ToolResultMessage {
        tool_call_id: run.call_id,
        tool_name: run.name,
        content,
        details,
        usage: None,
        is_error,
        timestamp: clock::now_ms(),
        duration_ms: run.duration_ms.map(|ms| ms as f64),
    }
}
