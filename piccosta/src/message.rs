// Messages: what a conversation is made of, in memory and in session files.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/types.ts (content blocks, the four
// roles a model sees, usage) and packages/coding-agent/src/core/messages.ts (the roles that live
// only in pi's sessions, and how they turn into user messages for the model).
//
// Every field keeps pi's JSON name, so session files pi wrote open here and the other way round.
// Fields pi grew for things piccosta doesn't do (deferred batch responses, codemode's nested tool
// calls, provider diagnostics) aren't modeled: serde skips unknown fields when reading, and the
// session layer keeps each line's original JSON for the cases that copy entries.

use crate::json_number;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// The text around a compaction or branch summary when it goes to the model as a user message.
// Model-visible, so these are pi's strings exactly (core/messages.ts:11-24).
pub const COMPACTION_SUMMARY_PREFIX: &str =
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";
pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

// One block of message content. pi tags each block with `type`. Which blocks a role allows is
// narrower than this enum (an assistant never sends an image, a user never sends a tool call);
// the code that builds messages keeps to that, and reading stays lenient.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Content {
    Text {
        text: String,
        // OpenAI Responses message metadata, replayed so the next request can refer back to it.
        #[serde(rename = "textSignature", skip_serializing_if = "Option::is_none")]
        text_signature: Option<String>,
    },
    Thinking {
        thinking: String,
        // The provider's opaque reasoning record (encrypted reasoning for OpenAI), sent back on
        // the next turn so the model keeps its chain of thought.
        #[serde(rename = "thinkingSignature", skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
        // True when a safety filter withheld the text; the payload is in the signature only.
        #[serde(skip_serializing_if = "Option::is_none")]
        redacted: Option<bool>,
    },
    Image {
        // Base64 image bytes and their type, e.g. "image/png".
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    ToolCall {
        // False for old files and local calls. External agents execute their own tools; this is
        // recorded history, not a request for our executor.
        #[serde(default, skip_serializing_if = "is_false")]
        external: bool,
        id: String,
        name: String,
        arguments: Map<String, Value>,
        // OpenAI Responses namespace, for calls to tools loaded under one.
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        // Set when the model's arguments weren't a valid JSON object. `arguments` is then empty,
        // and this says what was wrong, with the text the model sent.
        #[serde(rename = "argumentsError", skip_serializing_if = "Option::is_none")]
        arguments_error: Option<String>,
        // Gemini signs function-call parts too; the signature belongs to this exact call.
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
}

impl Content {
    // A plain text block, the common case. Input: the text. Output: a block with no signature.
    pub fn text(text: impl Into<String>) -> Content {
        Content::Text { text: text.into(), text_signature: None }
    }
}

// Message content that may be one string or a list of blocks. pi allows both for user, system
// and custom messages, and old sessions contain both forms, so both round-trip as they were.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextOrBlocks {
    Text(String),
    Blocks(Vec<Content>),
}

impl TextOrBlocks {
    // The content as blocks. A bare string becomes one text block, which is how pi turns custom
    // messages into user messages (core/messages.ts:163).
    pub fn into_blocks(self) -> Vec<Content> {
        match self {
            TextOrBlocks::Text(text) => vec![Content::text(text)],
            TextOrBlocks::Blocks(blocks) => blocks,
        }
    }

    // The text blocks' text, joined by newlines; images and other blocks are left out.
    pub fn text(&self) -> String {
        match self {
            TextOrBlocks::Text(text) => text.clone(),
            TextOrBlocks::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    Content::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

// Token counts for one response, and what they cost at the catalog price.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    // Pi's numeric fields stay zero when no per-turn counters were supplied. They are not a
    // measured zero-token turn; consumers must keep that distinction when adding up usage.
    #[serde(default, skip_serializing_if = "is_false")]
    pub tokens_unknown: bool,
    // An external agent's session reading, not per-turn input/output or cost. None means it did
    // not report accounting; keep cumulative/context readings without adding them as fresh usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reported: Option<Value>,
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    // The part of cache_write kept for an hour. Only Anthropic reports the split.
    #[serde(rename = "cacheWrite1h", skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    // Reasoning tokens, already counted inside `output`. None when the provider doesn't say.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    pub total_tokens: u64,
    pub cost: Cost,
}

// Dollar cost per part of a response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    #[serde(serialize_with = "json_number::serialize")]
    pub input: f64,
    #[serde(serialize_with = "json_number::serialize")]
    pub output: f64,
    #[serde(serialize_with = "json_number::serialize")]
    pub cache_read: f64,
    #[serde(serialize_with = "json_number::serialize")]
    pub cache_write: f64,
    #[serde(serialize_with = "json_number::serialize")]
    pub total: f64,
    // How it was billed when it was made: by a plan's quota (true), or by the token. None in a
    // response saved before piccosta kept it (responses::Tally says what that counts as).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<bool>,
    #[serde(default, skip_serializing_if = "CostState::is_known")]
    pub state: CostState,
    // How many pending turns before this one its total pays for: Claude Code reckons a tool turn's
    // cost on the turn that answers it.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub settles: u32,
}

fn is_false(value: &bool) -> bool { !*value }

fn is_zero(count: &u32) -> bool {
    *count == 0
}

// Whether a cost's numbers are what the response cost: known; unknown, billed by the token at prices
// nobody gave (responses::calculate_cost), so they're zero and mean nothing; or pending, zero until
// a later turn's `settles` pays for it, and unknown if none ever does (its query was stopped first).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CostState {
    #[default]
    Known,
    Pending,
    Unknown,
}

impl CostState {
    fn is_known(&self) -> bool {
        *self == CostState::Known
    }
}

// Why a response ended. `Pending` and `Deferred` belong to pi's deferred batch requests, which
// piccosta never makes; they stay only so a session that recorded one still opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

// A tool as the model is told about it: name, description and JSON Schema for its arguments.
// `constrainedSampling` (provider-side schema enforcement) is kept as raw JSON; only the request
// builders look inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolReference {
    pub name: String,
}

// The system prompt and tool set at one point in a conversation (ai/src/types.ts:544-570). The
// first system message is the prompt. Each later one changes it: `content` adds instructions,
// `sections` replaces named sections (None removes one), and the tool lists add or remove
// tools. Replaying them in order gives the current prompt and tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMessage {
    pub content: TextOrBlocks,
    // Ordered by insertion, which is the order the sections render in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sections: Option<IndexMap<String, Option<String>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_added: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_removed: Option<Vec<ToolReference>>,
    // Epoch milliseconds, as in every message.
    pub timestamp: i64,
}

// A system message on its own, outside a message list, written with its `role` as pi writes it.
// A compaction entry keeps one of these as its snapshot of the prompt and tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum SystemSnapshot {
    System(SystemMessage),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    pub content: TextOrBlocks,
    pub timestamp: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessage {
    pub content: Vec<Content>,
    // Which wire protocol, provider and model produced this, e.g. "openai-codex-responses",
    // "openai-codex", "gpt-6-luna". The converters use them to decide what can be replayed.
    pub api: String,
    pub provider: String,
    pub model: String,
    // The concrete model the provider reported, when it differs from `model`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    // Proven local/setup failure, before a prompt could be delivered. Unlike an uncertain turn,
    // this diagnostic is not part of the external agent's conversation and can be retried safely.
    #[serde(default, skip_serializing_if = "is_false")]
    pub request_not_sent: bool,
    // The provider's own effort name used for this response, and pi's level that asked for it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_thinking_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    // When the request started, epoch ms, and how long the response took on a monotonic clock.
    pub timestamp: i64,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "json_number::serialize_option",
        default
    )]
    pub duration_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    // Text and image blocks only.
    pub content: Vec<Content>,
    // Tool-specific structured data for the session and the UI. Never sent to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    // Tokens a tool spent itself (a subagent, say), kept apart from the main context's usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub is_error: bool,
    pub timestamp: i64,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "json_number::serialize_option",
        default
    )]
    pub duration_ms: Option<f64>,
}

// A command the user ran with `!` in the editor. It reaches the model as a user message
// unless the user typed `!!`, which keeps it out of context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashExecutionMessage {
    pub command: String,
    pub output: String,
    // None when the command never exited on its own (cancelled, killed). pi writes no key then.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    pub cancelled: bool,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    pub timestamp: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_from_context: Option<bool>,
}

// A message a built-in feature adds to the conversation (pi's extension sendMessage). It goes to
// the model as a user message; `display` decides whether the UI shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomMessage {
    pub custom_type: String,
    pub content: TextOrBlocks,
    pub display: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchSummaryMessage {
    pub summary: String,
    // The leaf the conversation came back from, or None for the root.
    pub from_id: Option<String>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryMessage {
    pub summary: String,
    pub tokens_before: u64,
    pub timestamp: i64,
}

// The four roles a model understands. Everything sent to a provider is a list of these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

// Everything a session can hold: the four model roles plus pi's own four. The extra roles turn
// into user messages on the way to the model (`to_model_messages`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum AgentMessage {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
    BashExecution(BashExecutionMessage),
    Custom(CustomMessage),
    BranchSummary(BranchSummaryMessage),
    CompactionSummary(CompactionSummaryMessage),
}

// The text a `!` command becomes for the model (core/messages.ts:82-98). Input: the recorded
// command. Output: "Ran `cmd`" plus its output in a fence, and notes for a cancel, a non-zero
// exit, or output cut short with the path of the full copy.
pub fn bash_execution_text(execution: &BashExecutionMessage) -> String {
    let mut text = format!("Ran `{}`\n", execution.command);
    if execution.output.is_empty() {
        text.push_str("(no output)");
    } else {
        text.push_str(&format!("```\n{}\n```", execution.output));
    }
    if execution.cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(code) = execution.exit_code.filter(|code| *code != 0) {
        text.push_str(&format!("\n\nCommand exited with code {code}"));
    }
    if execution.truncated
        && let Some(path) = &execution.full_output_path {
            text.push_str(&format!("\n\n[Output truncated. Full output: {path}]"));
        }
    text
}

// A user message holding one text block, the form all of pi's own roles take for the model.
fn user_text(text: String, timestamp: i64) -> Message {
    Message::User(UserMessage { content: TextOrBlocks::Blocks(vec![Content::text(text)]), timestamp })
}

// What the model sees of a conversation (core/messages.ts:148-196). Input: the session's messages
// in order. Output: the same conversation in the four model roles. `!!` commands are left out,
// and every other pi role becomes a user message.
pub fn to_model_messages(messages: &[AgentMessage]) -> Vec<Message> {
    let mut model_messages = Vec::with_capacity(messages.len());
    for message in messages {
        let converted = match message {
            AgentMessage::System(system) => Message::System(system.clone()),
            AgentMessage::User(user) => Message::User(user.clone()),
            AgentMessage::Assistant(assistant) => Message::Assistant(assistant.clone()),
            AgentMessage::ToolResult(result) => Message::ToolResult(result.clone()),
            AgentMessage::BashExecution(execution) => {
                if execution.exclude_from_context == Some(true) {
                    continue;
                }
                user_text(bash_execution_text(execution), execution.timestamp)
            }
            AgentMessage::Custom(custom) => Message::User(UserMessage {
                content: TextOrBlocks::Blocks(custom.content.clone().into_blocks()),
                timestamp: custom.timestamp,
            }),
            AgentMessage::BranchSummary(branch) => user_text(
                format!("{BRANCH_SUMMARY_PREFIX}{}{BRANCH_SUMMARY_SUFFIX}", branch.summary),
                branch.timestamp,
            ),
            AgentMessage::CompactionSummary(compaction) => user_text(
                format!("{COMPACTION_SUMMARY_PREFIX}{}{COMPACTION_SUMMARY_SUFFIX}", compaction.summary),
                compaction.timestamp,
            ),
        };
        model_messages.push(converted);
    }
    model_messages
}
