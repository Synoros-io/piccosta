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
// sets one (rpc mode for T3 Code's approvals); without one every call runs.
pub type Gate = std::rc::Rc<dyn Fn(String, String, Map<String, Value>) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>>>>>;

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
}

impl Handle {
    pub fn abort(&self) {
        self.lock().cancel.cancel();
    }

    // Stop the run without killing its commands, the way every interrupt does (Esc, an rpc abort,
    // ACP's cancel). Input: the session's jobs. The move goes out before the abort, and bash's
    // biased select takes it first, so each running command goes on as a background job.
    pub fn interrupt(&self, jobs: &crate::tools::jobs::Jobs) {
        jobs.detach_all();
        self.abort();
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
    // they're read again before each run.
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
}

pub struct Agent {
    pub session: Session,
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
        let config = AgentConfig { thinking_level: level, ..config };
        let handle = Handle::default();
        let cancel = handle.lock().cancel.clone();
        Ok(Agent { session, config, emit, cancel, handle, retry_attempt: 0, overflow_recovery_attempted: false, held: Vec::new(), repeated_call: (String::new(), 0), machine_noted: None })
    }

    // The controls for this agent's runs (abort on Ctrl-C or Escape, queued messages).
    pub fn handle(&self) -> Handle {
        self.handle.clone()
    }

    // A fresh cancel token for the run about to start, the one the handle aborts.
    fn fresh_cancel(&mut self) {
        self.cancel = CancellationToken::new();
        self.handle.lock().cancel = self.cancel.clone();
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
    pub fn set_model(&mut self, model: Model, sections: IndexMap<String, String>, compaction: CompactionSettings) -> std::io::Result<()> {
        if (&model.provider, &model.id) != (&self.config.model.provider, &self.config.model.id) {
            let base = self.session.next_base();
            self.session.append(SessionEntry::ModelChange { base, provider: model.provider.clone(), model_id: model.id.clone() })?;
        }
        self.config.model = model;
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

    // The system message that brings the session's prompt and tools up to date, or None when they
    // already are (system-prompt.ts diffSystemPromptSections, agent-loop.ts declareToolChanges).
    // A session keeps the prompt it has: any change to it rewrites the provider's whole prompt
    // cache (a new build's prompt text did that to every running session on its next turn,
    // 11.9M tokens written in a day, 2026-11-01). So it's brought up to date only when it has none
    // yet, at a compaction or a model change (which start a new cache anyway), or when what it's
    // built from changed: its context files (prompt::INPUT_SECTIONS) or what a tool can be called
    // with, its name and parameters less their descriptions (a subagent job added or removed
    // changes the job enum in the subagent tool's parameters). A tool's wording, like the prompt's,
    // waits: the tools that stayed keep the definitions and tool-list lines the session has.
    fn system_update(&self) -> Option<SystemMessage> {
        let messages = crate::message::to_model_messages(&self.session.context().messages);
        let current = transcript::current_system_message(&messages);
        let previous: IndexMap<String, String> = current
            .as_ref()
            .and_then(|system| system.sections.as_ref())
            .map(|sections| sections.iter().filter_map(|(name, text)| text.clone().map(|text| (name.clone(), text))).collect())
            .unwrap_or_default();
        let current_tools: Vec<ToolDefinition> = current.and_then(|system| system.tools_added).unwrap_or_default();
        let desired: Vec<ToolDefinition> = self.config.tools.iter().map(|tool| tool.definition()).collect();
        let stayed = |tool: &ToolDefinition| {
            current_tools.iter().any(|held| held.name == tool.name && callable_schema(&held.parameters) == callable_schema(&tool.parameters))
        };
        let tools_changed = desired.len() != current_tools.len() || !desired.iter().all(stayed);
        // An input section is the same when its files are: piccosta's header line above them is
        // wording, which waits like any other.
        let same_input = |name: &str| match (previous.get(name), self.config.sections.get(name)) {
            (Some(held), Some(new)) => below_header(held).map_or(held.as_str(), |(_, files)| files) == below_header(new).map_or(new.as_str(), |(_, files)| files),
            (held, new) => held == new,
        };
        let inputs_changed = crate::prompt::INPUT_SECTIONS.iter().any(|name| !same_input(name)) || tools_changed;
        // The newest of a system message, a reply, a compaction or a model change, walking back from
        // the leaf. A reset is free only until the first request after it: once a reply follows it,
        // the cache has been written again with the prompt as it was, and the chance has passed.
        let since_reset = self.session.path_to_leaf().into_iter().rev().find_map(|entry| match entry {
            SessionEntry::Message { message: AgentMessage::System(_) | AgentMessage::Assistant(_), .. } => Some(false),
            SessionEntry::Compaction { .. } | SessionEntry::ModelChange { .. } => Some(true),
            _ => None,
        });
        let reset = previous.is_empty() || since_reset == Some(true);
        if !reset && !inputs_changed {
            return None;
        }
        // After a reset every section is brought up to date. Otherwise only what the prompt is built
        // from goes in: the input sections whose files changed, under the header the session already
        // has, and the tool list when the tools did. The rest of the wording stays as it was.
        let refreshed = |name: &str| reset || (crate::prompt::INPUT_SECTIONS.contains(&name) && !same_input(name)) || (name == "tools" && tools_changed);
        let mut patch: IndexMap<String, Option<String>> = IndexMap::new();
        for (name, text) in &self.config.sections {
            if !refreshed(name) {
                continue;
            }
            let text = match (previous.get(name).and_then(|held| below_header(held)), below_header(text)) {
                _ if reset => text.clone(),
                (Some((header, _)), Some((_, files))) => format!("{header}\n\n{files}"),
                // The tool list the session has, less the lines of tools that went or changed, with
                // the lines of the new ones after its last tool, or after its opening line when none
                // stayed, so its own frame is kept. With no list held, the new one.
                _ if name == "tools" => {
                    let tool_of = |line: &str| line.strip_prefix("- ").and_then(|rest| rest.split_once(": ")).map(|(tool, _)| tool.to_string());
                    let kept = |line: &&str| tool_of(line).is_none_or(|tool| desired.iter().any(|wanted| wanted.name == tool && stayed(wanted)));
                    let mut lines: Vec<&str> = previous.get(name).map_or("", String::as_str).lines().filter(kept).collect();
                    let at = lines.iter().rposition(|line| tool_of(line).is_some()).map(|last| last + 1).or((!lines.is_empty()).then_some(1));
                    match at {
                        Some(at) => {
                            lines.splice(at..at, text.lines().filter(|line| tool_of(line).is_some() && !kept(line)));
                            lines.join("\n")
                        }
                        None => text.clone(),
                    }
                }
                _ => text.clone(),
            };
            if previous.get(name) != Some(&text) {
                patch.insert(name.clone(), Some(text));
            }
        }
        // A section the new prompt doesn't have goes only at a reset, or when it's an input whose
        // files went. The tool list, when the new prompt has none (a SYSTEM.md replaced it), stays
        // as held until a reset, like any other wording.
        for name in previous.keys() {
            if !self.config.sections.contains_key(name) && refreshed(name) && (reset || name != "tools") {
                patch.insert(name.clone(), None);
            }
        }

        // After a reset any tool whose definition differs goes in; otherwise only the ones added or
        // with new parameters.
        let added: Vec<ToolDefinition> = desired.iter().filter(|tool| if reset { !current_tools.contains(tool) } else { !stayed(tool) }).cloned().collect();
        let removed: Vec<ToolReference> = current_tools
            .iter()
            .filter(|tool| !desired.iter().any(|wanted| wanted.name == tool.name))
            .map(|tool| ToolReference { name: tool.name.clone() })
            .collect();
        if patch.is_empty() && added.is_empty() && removed.is_empty() {
            return None;
        }
        Some(SystemMessage {
            content: TextOrBlocks::Text(String::new()),
            sections: (!patch.is_empty()).then_some(patch),
            tools_added: (!added.is_empty()).then_some(added),
            tools_removed: (!removed.is_empty()).then_some(removed),
            timestamp: clock::now_ms(),
        })
    }

    // Read the session's AGENTS.md files again, unless they're off.
    fn reload_context_files(&mut self) {
        if self.config.context_files {
            crate::prompt::reload_context_files(&mut self.config.sections, &self.config.cwd, &crate::config::agent_dir());
        }
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

    // Send the user's prompt and run until the model is done (and any retry or background job
    // it waits on). Output: an error only when the session can't be written; model failures end
    // up in the session as failed assistant messages, as in pi.
    pub async fn prompt(&mut self, text: &str) -> Result<(), String> {
        self.run_from(user_message(text)).await
    }

    // A prompt with images attached (rpc mode's `images`).
    pub async fn prompt_with_images(&mut self, text: &str, images: Vec<Content>) -> Result<(), String> {
        // A prompt in a closed session opens it again.
        if let Some(file) = self.session.file() {
            crate::session::set_closed(&crate::config::agent_dir(), file, false);
        }
        self.run_from(with_images(text, images)).await
    }

    pub fn set_gate(&mut self, gate: Gate) {
        self.config.gate = Some(gate);
    }

    // A background job's finish message, when one comes while nothing runs. A frontend waits on
    // this between prompts and hands what arrives to `take_notice`.
    pub async fn notice(&mut self) -> Option<CustomMessage> {
        self.config.notices.recv().await
    }

    // Run a turn on a background job's finish message, as longcmd does in pi when the session
    // is idle.
    pub async fn take_notice(&mut self, notice: CustomMessage) -> Result<(), String> {
        self.run_from(AgentMessage::Custom(notice)).await
    }

    // Keep a job's message for the next prompt, which it goes in ahead of: a kill the user asked
    // for (/stop) is news for the model, not a reason to answer.
    pub fn hold_notice(&mut self, notice: CustomMessage) {
        self.held.push(notice);
    }

    async fn run_from(&mut self, first: AgentMessage) -> Result<(), String> {
        self.fresh_cancel();
        // Compact first when the last response left the context past the threshold; the new
        // prompt replaces any retry that schedules (agent-session.ts _prepareRun, _startRun).
        if let Some((assistant, entry_id)) = self.last_projected_assistant() {
            self.check_compaction(&assistant, Some(&entry_id), false).await?;
        }
        let mut starting: Vec<AgentMessage> = self.held.drain(..).map(AgentMessage::Custom).collect();
        starting.push(first);
        let mut aborted = false;
        loop {
            self.reload_context_files();
            if let Some(system) = self.system_update() {
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
        (self.emit)(&Event::AgentSettled { aborted });
        Ok(())
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
    // finish message (longcmd.ts:533-554), or a message queued meanwhile (another piccosta's, from
    // the session's inbox). PICCOSTA_LONGCMD_HOLD_S caps the wait, then the jobs are killed and
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
        let messages = crate::message::to_model_messages(&self.session.context().messages);
        let model = &self.config.model;
        let credential = auth::credential_for(&model.provider, self.config.registry.provider_config(&model.provider), &self.config.credentials).await;
        let message = match credential {
            Ok(credential) => {
                let (sender, mut events) = mpsc::unbounded_channel();
                // The stream loop guard stops the request through its own token, so a stopped loop
                // isn't taken for the user's abort.
                let guard = self.cancel.child_token();
                let request = Request {
                    model,
                    messages: &messages,
                    credential: &credential,
                    session_id: Some(self.session.id()),
                    cache: true,
                    transport: &self.config.transport,
                    thinking_level: &self.config.thinking_level,
                    session_file: self.session.file(),
                    cwd: &self.config.cwd,
                    cancel: guard.clone(),
                };
                let response = provider::stream(request, sender);
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
        if let Some(text) = provider::claude::take_notice() {
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
    // is reserveTokens, except on the bridge, whose window already keeps it (claude effective_window).
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
        self.reload_context_files();
        if let Some(system) = self.system_update() {
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

    fn tool_context(&self) -> ToolContext {
        ToolContext {
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
            self.emit_tool_end(&run);
            runs.push(run);
        }
        runs
    }

    fn emit_tool_end(&mut self, run: &ToolRun) {
        let (result, is_error) = match &run.output {
            Ok(output) => (result_json(&output.content, &output.details), output.is_error),
            Err(message) => (result_json(&[Content::text(message.clone())], &Some(json!({}))), true),
        };
        (self.emit)(&Event::ToolExecutionEnd {
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
                    allowed = gate(id.clone(), name.clone(), validated.clone()) => allowed.map(|()| (tool, validated)),
                    () = self.cancel.cancelled() => Err("Operation aborted".to_string()),
                },
                (checked, _) => checked,
            };
            let (tool, validated) = match checked {
                Ok(ready) => ready,
                Err(output) => {
                    let run = ToolRun { index: slot, call_id: id.clone(), name: name.clone(), output: Err(output), duration_ms: None };
                    self.emit_tool_end(&run);
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
            let context = ToolContext { updates: Some(sender), ..self.tool_context() };
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
                    self.emit_tool_end(&run);
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

// What a tool can be called with: its parameter schema with the "description" strings taken out at
// every depth, since they're wording. Only string values go, so a parameter named description stays.
fn callable_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => Value::Object(
            map.iter().filter(|(key, value)| !(*key == "description" && value.is_string())).map(|(key, value)| (key.clone(), callable_schema(value))).collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(callable_schema).collect()),
        other => other.clone(),
    }
}

// A project_context section split at its header line: the header (piccosta's wording, with the
// opening tag) and the files below it. None for other sections.
fn below_header(text: &str) -> Option<(&str, &str)> {
    text.starts_with("<project_context>\n").then(|| text.split_once("\n\n")).flatten()
}
