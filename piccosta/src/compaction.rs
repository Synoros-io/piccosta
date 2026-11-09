// Compaction: when a conversation nears its model's context window, the older part is summarized by
// the model and the summary takes its place, with the recent part kept whole after it.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/compaction/
// compaction.ts and utils.ts. Everything the summarizer reads is pi's to the byte: the prompts, the
// way the conversation is written out for it, and the file lists after the summary. Sizes count
// UTF-16 units, as JavaScript's `.length` does, at pi's 4 characters per token for compaction (the
// request-size estimate in provider/estimate.rs uses 3.5; that's pi's split too), so compaction
// fires and cuts where pi's does.
//
// The flow: `prepare` picks what to summarize from the session (None when there's nothing to),
// `summarize` asks the model, and the agent writes the entry and says so (agent.rs). Left out:
// codemode's nested calls in the file lists (piccosta has no codemode) and extension hooks.

use crate::auth::Credential;
use crate::config::{CompactionSettings, RetrySettings};
use crate::message::{
    to_model_messages, AgentMessage, AssistantMessage, Content, Message, StopReason, SystemMessage, TextOrBlocks, Usage,
    UserMessage,
};
use crate::model::Model;
use crate::provider::{self, estimate, transcript, Request};
use crate::session::{ProjectedEntry, Session, SessionEntry};
use crate::{clock, js, retry};
use serde::Serialize;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// pi's rule of thumb for compaction, chars / 4, which overestimates tokens (compaction.ts:294-298),
// and what one image counts as (compaction.ts:276).
const CHARS_PER_TOKEN: f64 = 4.0;
const ESTIMATED_IMAGE_CHARS: usize = 4800;
// The most of one tool result the summarizer sees (utils.ts:94).
const TOOL_RESULT_MAX_CHARS: usize = 2000;

pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You write the checkpoint a coding session continues from after its earlier messages are dropped. You read the conversation and write the summary in the format asked for. You don't answer anything in the conversation or carry on with its work.";

const SUMMARIZATION_PROMPT: &str = "The messages above are the conversation so far. Write the checkpoint the session will continue from, in this format:

## The user's instructions
Every instruction, rule or preference the user gave that still applies, each quoted in the user's own words. Never paraphrase or shorten these, and leave none out: they are the user's standing orders for the rest of the session.

## Goal
What the user wants done. More than one item if the session covers several things.

## Done
- What's finished, with the exact files, functions and commands.

## In progress
- What was being worked on when the conversation was cut, and how far it got.

## Blocked
- What's stopping progress, if anything.

## Decisions
- Each decision and the reason for it.

## Next
1. What comes next, in order.

## Facts needed to continue
- Paths, names, numbers, error messages and references, exactly as they appeared.

Keep the rest short, but keep paths, names and error messages exact.";

// Updating a checkpoint: the same sections, keeping what still holds.
const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are new since the checkpoint in <previous-summary> tags. Write the updated checkpoint: keep everything in the old one that still holds, add what's new, move finished work to Done, update Next, and drop only what no longer matters. The user's instructions are never dropped while they still apply. Use this format:

## The user's instructions
Every instruction, rule or preference the user gave that still applies, each quoted in the user's own words. Never paraphrase or shorten these, and leave none out: they are the user's standing orders for the rest of the session.

## Goal
What the user wants done. More than one item if the session covers several things.

## Done
- What's finished, with the exact files, functions and commands.

## In progress
- What was being worked on when the conversation was cut, and how far it got.

## Blocked
- What's stopping progress, if anything.

## Decisions
- Each decision and the reason for it.

## Next
1. What comes next, in order.

## Facts needed to continue
- Paths, names, numbers, error messages and references, exactly as they appeared.

Keep the rest short, but keep paths, names and error messages exact.";

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "The messages above are the start of a turn whose later messages are kept as they are. Write a short checkpoint of this part only, placed before those later messages:

## The user's instructions
Every instruction or rule the user gave here, quoted in their own words.

## The request
What the user asked for.

## Done so far
- Decisions and work in these messages.

## Needed to continue
- What from these messages the later ones depend on.

Only what's in the messages above; don't guess at what comes later.";

// Idle compaction's summary is asked for at the end of the session's own request, under the
// session's system prompt, so the summarizer's role (SUMMARIZATION_SYSTEM_PROMPT) goes in the message.
const IN_PLACE_PROMPT: &str = "piccosta is compacting this session now. Answer this message alone, in text, calling no tool.";
// What the session's views say of an idle compaction (Claude Code's words for its own).
pub const IDLE_LINE: &str = "Compacted while idle, before the prompt cache expired";

// What set a compaction off, as pi's events name it, and idle compaction's, which pi doesn't have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Reason {
    Manual,
    Threshold,
    Overflow,
    Idle,
}

// ---------------------------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------------------------

fn tokens_for_chars(chars: usize) -> u64 {
    (chars as f64 / CHARS_PER_TOKEN).ceil() as u64
}

fn blocks_chars(blocks: &[Content]) -> usize {
    blocks
        .iter()
        .map(|block| match block {
            Content::Text { text, .. } => js::utf16_len(text),
            Content::Image { .. } => ESTIMATED_IMAGE_CHARS,
            _ => 0,
        })
        .sum()
}

fn content_chars(content: &TextOrBlocks) -> usize {
    match content {
        TextOrBlocks::Text(text) => js::utf16_len(text),
        TextOrBlocks::Blocks(blocks) => blocks_chars(blocks),
    }
}

// The estimated tokens of one message (compaction.ts:298-349): its visible characters / 4. A
// system message counts its sections and the JSON of the tools it adds; an assistant counts its
// text, thinking, and each call's name and arguments.
pub fn estimate_tokens(message: &AgentMessage) -> u64 {
    let chars = match message {
        AgentMessage::System(system) => {
            let sections: usize = system.sections.iter().flatten().filter_map(|(_, text)| text.as_deref()).map(js::utf16_len).sum();
            let tools = system
                .tools_added
                .as_ref()
                .map(|tools| js::utf16_len(&js::stringify(&serde_json::to_value(tools).unwrap_or_default())))
                .unwrap_or(0);
            content_chars(&system.content) + sections + tools
        }
        AgentMessage::User(user) => content_chars(&user.content),
        AgentMessage::Assistant(assistant) => assistant
            .content
            .iter()
            .map(|block| match block {
                Content::Text { text, .. } => js::utf16_len(text),
                Content::Thinking { thinking, .. } => js::utf16_len(thinking),
                Content::ToolCall { name, arguments, .. } => {
                    js::utf16_len(name) + js::utf16_len(&js::stringify(&Value::Object(arguments.clone())))
                }
                Content::Image { .. } => 0,
            })
            .sum(),
        AgentMessage::Custom(custom) => content_chars(&custom.content),
        AgentMessage::ToolResult(result) => blocks_chars(&result.content),
        AgentMessage::BashExecution(execution) => js::utf16_len(&execution.command) + js::utf16_len(&execution.output),
        AgentMessage::BranchSummary(branch) => js::utf16_len(&branch.summary),
        AgentMessage::CompactionSummary(compaction) => js::utf16_len(&compaction.summary),
    };
    tokens_for_chars(chars)
}

// A response's reported usage, when it can be trusted as the context size: not from an aborted or
// failed response, and not all zeros (compaction.ts:148-161).
pub fn assistant_usage(message: &AgentMessage) -> Option<&Usage> {
    let AgentMessage::Assistant(assistant) = message else { return None };
    let usable = !matches!(assistant.stop_reason, StopReason::Aborted | StopReason::Error);
    (usable && estimate::context_tokens(&assistant.usage) > 0).then_some(&assistant.usage)
}

// A context size: the last trusted usage plus estimates for what came after it, or estimates
// alone (pi's ContextUsageEstimate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate {
    pub tokens: u64,
    pub usage_tokens: u64,
    pub trailing_tokens: u64,
    // Which message the usage came from; None when every number is an estimate.
    pub last_usage_index: Option<usize>,
}

// The context size of a message list (compaction.ts:196-224).
pub fn estimate_context_tokens(messages: &[AgentMessage]) -> Estimate {
    let found = messages.iter().enumerate().rev().find_map(|(index, message)| assistant_usage(message).map(|usage| (index, usage)));
    let Some((index, usage)) = found else {
        let estimated = messages.iter().map(estimate_tokens).sum();
        return Estimate { tokens: estimated, usage_tokens: 0, trailing_tokens: estimated, last_usage_index: None };
    };
    let usage_tokens = estimate::context_tokens(usage);
    let trailing_tokens = messages[index + 1..].iter().map(estimate_tokens).sum();
    Estimate { tokens: usage_tokens + trailing_tokens, usage_tokens, trailing_tokens, last_usage_index: Some(index) }
}

// The context size of the session's projection (compaction.ts:227-262). A response's usage stops
// counting once a context edit or compaction came after it, since it measured a context that no
// longer exists; then the size is the current prompt plus estimates for everything else.
pub fn estimate_projected_context_tokens(projection: &[ProjectedEntry<'_>], branch: &[&SessionEntry]) -> Estimate {
    let messages: Vec<AgentMessage> = projection.iter().flat_map(|entry| entry.messages.iter().cloned()).collect();
    let estimate = estimate_context_tokens(&messages);
    if let Some(usage_index) = estimate.last_usage_index {
        // The entry the usage came from.
        let mut message_index = 0;
        let mut usage_entry_id = None;
        for entry in projection {
            let next = message_index + entry.messages.len();
            if usage_index < next {
                usage_entry_id = Some(entry.source.id());
                break;
            }
            message_index = next;
        }
        let usage_entry_index = usage_entry_id.and_then(|id| branch.iter().position(|entry| entry.id() == id));
        let latest_invalidating = branch
            .iter()
            .rposition(|entry| matches!(entry, SessionEntry::ContextEdit { .. } | SessionEntry::Compaction { .. }));
        // pi compares indices with -1 standing for "none": a usage entry that isn't on the branch
        // never beats an invalidating one, and any usage entry beats no invalidation at all.
        let valid = match (usage_entry_index, latest_invalidating) {
            (Some(usage), Some(invalidating)) => usage > invalidating,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if valid {
            return estimate;
        }
    }
    let model_messages = to_model_messages(&messages);
    let mut tokens = transcript::current_system_message(&model_messages)
        .map(|system| estimate_tokens(&AgentMessage::System(system)))
        .unwrap_or(0);
    tokens += messages.iter().filter(|message| !matches!(message, AgentMessage::System(_))).map(estimate_tokens).sum::<u64>();
    Estimate { tokens, usage_tokens: 0, trailing_tokens: tokens, last_usage_index: None }
}

// Whether a context this size is past the point to compact: within reserveTokens of the window
// (compaction.ts:267-270).
pub fn should_compact(context_tokens: u64, context_window: u64, settings: CompactionSettings) -> bool {
    settings.enabled && (context_tokens as i64 > context_window as i64 - settings.reserve_tokens as i64 || settings.compact_at.is_some_and(|tokens| context_tokens > tokens))
}

// ---------------------------------------------------------------------------------------------
// Where to cut
// ---------------------------------------------------------------------------------------------

// A message the kept part may start at. Never a tool result: it has to follow its call
// (compaction.ts:351-364).
fn is_cut_point_message(message: &AgentMessage) -> bool {
    matches!(
        message,
        AgentMessage::User(_)
            | AgentMessage::Assistant(_)
            | AgentMessage::BashExecution(_)
            | AgentMessage::Custom(_)
            | AgentMessage::BranchSummary(_)
            | AgentMessage::CompactionSummary(_)
    )
}

// A message that starts a turn: anything the user's side sends (compaction.ts:366-379).
fn is_turn_start_message(message: &AgentMessage) -> bool {
    matches!(
        message,
        AgentMessage::User(_)
            | AgentMessage::BashExecution(_)
            | AgentMessage::Custom(_)
            | AgentMessage::BranchSummary(_)
            | AgentMessage::CompactionSummary(_)
    )
}

fn is_compaction(entry: &ProjectedEntry<'_>) -> bool {
    matches!(entry.source, SessionEntry::Compaction { .. })
}

fn is_projected_turn_start(entry: &ProjectedEntry<'_>) -> bool {
    !is_compaction(entry) && entry.messages.iter().any(is_turn_start_message)
}

// The nearest entry at or before `entry_index`, and not before `start`, that starts a turn.
fn find_projected_turn_start(entries: &[ProjectedEntry<'_>], entry_index: usize, start: usize) -> Option<usize> {
    (start..=entry_index).rev().find(|&index| is_projected_turn_start(&entries[index]))
}

// Where the kept part starts, and the turn it splits, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CutPoint {
    first_kept: usize,
    // The entry starting the turn the cut falls inside; None when the cut starts a turn itself.
    turn_start: Option<usize>,
    is_split_turn: bool,
}

// Where to cut so that about `keep_recent_tokens` stay whole (compaction.ts:789-857). Walks back
// from the newest entry adding up sizes; once the budget is reached, the kept part starts at the
// nearest cut point at or after there. Entries in [start, end) only.
fn find_projected_cut_point(entries: &[ProjectedEntry<'_>], start: usize, end: usize, keep_recent_tokens: u64) -> CutPoint {
    let cut_points: Vec<usize> =
        (start..end).filter(|&index| !is_compaction(&entries[index]) && entries[index].messages.iter().any(is_cut_point_message)).collect();
    let Some(&first_cut_point) = cut_points.first() else {
        return CutPoint { first_kept: start, turn_start: None, is_split_turn: false };
    };

    let mut accumulated = 0;
    let mut exceeded_budget = false;
    let mut cut = first_cut_point;
    for index in (start..end).rev() {
        let tokens: u64 = entries[index].messages.iter().map(estimate_tokens).sum();
        if tokens == 0 {
            continue;
        }
        accumulated += tokens;
        if accumulated >= keep_recent_tokens {
            exceeded_budget = true;
            // The closest cut point at or after here. When the trailing tool results alone are
            // over the budget, keep from their call rather than falling back to the first message.
            cut = cut_points.iter().copied().find(|&candidate| candidate >= index).unwrap_or(*cut_points.last().unwrap_or(&first_cut_point));
            break;
        }
    }

    // A failed attempt that recovery hid, and its hiding edits, give nothing to context after the
    // last visible input. When everything after the cut is exactly that, the cut moves past it, so
    // the attempt isn't what starts the kept part; any other entry there keeps the cut where it is,
    // so it never passes input the model hasn't seen.
    let suffix = &entries[(cut + 1).min(end)..end];
    let intrinsically_visible =
        |entry: &ProjectedEntry<'_>| !matches!(entry.source, SessionEntry::ContextEdit { .. }) && !crate::session::entry_context_messages(entry.source).is_empty();
    let omitted = |entry: &ProjectedEntry<'_>| intrinsically_visible(entry) && entry.messages.is_empty();
    let omitted_ids: Vec<&str> = suffix.iter().filter(|entry| omitted(entry)).map(|entry| entry.source.id()).collect();
    let has_external_replacement = suffix.iter().any(|entry| {
        matches!(entry.source, SessionEntry::ContextEdit { target_id, replacement: Some(_), .. } if !omitted_ids.contains(&target_id.as_str()))
    });
    let is_recovery_omission_suffix = exceeded_budget
        && !has_external_replacement
        && suffix.iter().any(|entry| matches!(entry.source, SessionEntry::Message { message: AgentMessage::Assistant(_), .. }) && omitted(entry))
        && suffix.iter().all(|entry| !is_compaction(entry) && (!intrinsically_visible(entry) || omitted(entry)));
    if is_recovery_omission_suffix {
        cut += 1;
    }

    // Take in the state-only entries just before the cut (a model change, say).
    while cut > start {
        let previous = &entries[cut - 1];
        if is_compaction(previous) || !previous.messages.is_empty() {
            break;
        }
        cut -= 1;
    }
    let starts_turn = entries.get(cut).is_some_and(is_projected_turn_start);
    let turn_start = if starts_turn { None } else { find_projected_turn_start(entries, cut.min(entries.len().saturating_sub(1)), start) };
    CutPoint { first_kept: cut, turn_start, is_split_turn: !starts_turn && turn_start.is_some() }
}

// ---------------------------------------------------------------------------------------------
// What to summarize
// ---------------------------------------------------------------------------------------------

// The files the summarized part read and changed (utils.ts:12-24). Paths are kept as the model
// wrote them.
#[derive(Debug, Clone, Default)]
pub struct FileOps {
    read: Vec<String>,
    written: Vec<String>,
    edited: Vec<String>,
}

fn add_unique(list: &mut Vec<String>, path: &str) {
    if !list.iter().any(|known| known == path) {
        list.push(path.to_string());
    }
}

impl FileOps {
    // Record the read, write and edit calls an assistant message made (utils.ts:30-61).
    fn record(&mut self, message: &AgentMessage) {
        let AgentMessage::Assistant(assistant) = message else { return };
        for block in &assistant.content {
            let Content::ToolCall { name, arguments, .. } = block else { continue };
            let Some(path) = arguments.get("path").and_then(Value::as_str).filter(|path| !path.is_empty()) else { continue };
            match name.as_str() {
                "read" => add_unique(&mut self.read, path),
                "write" => add_unique(&mut self.written, path),
                "edit" => add_unique(&mut self.edited, path),
                _ => {}
            }
        }
    }

    // Files only read, and files changed, each sorted as JavaScript sorts strings: by UTF-16 code
    // units (utils.ts:67-72).
    fn lists(&self) -> (Vec<String>, Vec<String>) {
        let sort = |list: &mut Vec<String>| list.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        let mut modified: Vec<String> = Vec::new();
        for path in self.edited.iter().chain(&self.written) {
            add_unique(&mut modified, path);
        }
        let mut read_only: Vec<String> = self.read.iter().filter(|path| !modified.contains(path)).cloned().collect();
        sort(&mut read_only);
        sort(&mut modified);
        (read_only, modified)
    }
}

// The file lists as the summary's closing tags (utils.ts:77-87), or "" when there are none.
fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut sections = Vec::new();
    if !read_files.is_empty() {
        sections.push(format!("<read-files>\n{}\n</read-files>", read_files.join("\n")));
    }
    if !modified_files.is_empty() {
        sections.push(format!("<modified-files>\n{}\n</modified-files>", modified_files.join("\n")));
    }
    if sections.is_empty() {
        return String::new();
    }
    format!("\n\n{}", sections.join("\n\n"))
}

// Everything a compaction needs before the model is asked (pi's CompactionPreparation).
#[derive(Debug, Clone)]
pub struct Preparation {
    pub first_kept_entry_id: String,
    // The part that's summarized and dropped.
    pub messages_to_summarize: Vec<AgentMessage>,
    // The start of a turn the cut splits, summarized on its own.
    pub turn_prefix_messages: Vec<AgentMessage>,
    pub is_split_turn: bool,
    pub tokens_before: u64,
    // The last compaction's summary, which the new one updates.
    pub previous_summary: Option<String>,
    file_ops: FileOps,
}

// A projected entry's messages as compaction reads them: none from a compaction, and no system
// messages, which are prompt state the compaction entry carries itself (compaction.ts:98-102).
fn messages_for_compaction(entry: &ProjectedEntry<'_>) -> Vec<AgentMessage> {
    if is_compaction(entry) {
        return Vec::new();
    }
    entry.messages.iter().filter(|message| !matches!(message, AgentMessage::System(_))).cloned().collect()
}

// Decide what a compaction of the session at its leaf would summarize (compaction.ts:859-923).
// Output: None when the leaf is already a compaction or there's nothing before the cut.
pub fn prepare(session: &Session, settings: CompactionSettings) -> Option<Preparation> {
    let branch = session.path_to_leaf();
    if matches!(branch.last(), Some(SessionEntry::Compaction { .. })) {
        return None;
    }
    let projection = session.projection();
    // The newest compaction is projected first; older ones inside its kept range project empty.
    let previous_compaction = projection.iter().position(|entry| is_compaction(entry) && !entry.messages.is_empty());
    let (previous_summary, boundary_start) = match previous_compaction {
        Some(index) => {
            let SessionEntry::Compaction { summary, .. } = projection[index].source else { unreachable!("is_compaction matched") };
            (Some(summary.clone()), index + 1)
        }
        None => (None, 0),
    };
    let boundary_end = projection.len();
    let tokens_before = estimate_projected_context_tokens(&projection, &branch).tokens;
    let cut = find_projected_cut_point(&projection, boundary_start, boundary_end, settings.keep_recent_tokens);

    let first_kept_entry_id = projection.get(cut.first_kept)?.source.id().to_string();
    let history_end = cut.turn_start.filter(|_| cut.is_split_turn).unwrap_or(cut.first_kept);
    let messages_to_summarize: Vec<AgentMessage> =
        projection[boundary_start..history_end.max(boundary_start)].iter().flat_map(messages_for_compaction).collect();
    let turn_prefix_messages: Vec<AgentMessage> = match cut.turn_start.filter(|_| cut.is_split_turn) {
        Some(turn_start) => projection[turn_start..cut.first_kept].iter().flat_map(messages_for_compaction).collect(),
        None => Vec::new(),
    };
    if messages_to_summarize.is_empty() && turn_prefix_messages.is_empty() {
        return None;
    }

    // The files from the summarized part, plus those the previous compaction listed, when pi
    // wrote it (compaction.ts:60-88). Its changed files count as edited.
    let mut file_ops = FileOps::default();
    if let Some(index) = previous_compaction
        && let SessionEntry::Compaction { details: Some(details), from_hook, .. } = projection[index].source
        && *from_hook != Some(true)
    {
        let paths = |key: &str| details.get(key).and_then(Value::as_array).cloned().unwrap_or_default();
        for path in paths("readFiles").iter().filter_map(Value::as_str) {
            add_unique(&mut file_ops.read, path);
        }
        for path in paths("modifiedFiles").iter().filter_map(Value::as_str) {
            add_unique(&mut file_ops.edited, path);
        }
    }
    for message in messages_to_summarize.iter().chain(&turn_prefix_messages) {
        file_ops.record(message);
    }

    Some(Preparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: cut.is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
    })
}

// ---------------------------------------------------------------------------------------------
// The conversation as the summarizer reads it
// ---------------------------------------------------------------------------------------------

fn blocks_text(blocks: &[Content], separator: &str) -> String {
    transcript::content_text(&TextOrBlocks::Blocks(blocks.to_vec()), separator)
}

// The first `max_chars` UTF-16 units of a tool result and a note of how many more there were
// (utils.ts:100-104). A cut through a surrogate pair leaves half a character; JavaScript sends
// that half, which providers read as U+FFFD, so U+FFFD is what goes in its place here.
fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let units: Vec<u16> = text.encode_utf16().collect();
    if units.len() <= max_chars {
        return text.to_string();
    }
    let kept = String::from_utf16_lossy(&units[..max_chars]);
    format!("{kept}\n\n[... {} more characters truncated]", units.len() - max_chars)
}

// The conversation as plain text the summarizer won't mistake for its own to continue
// (utils.ts:120-155): each message on its own lines, labeled by role, tool calls as
// name(key=json, ...), long tool results cut. Thinking is left out, and so are system messages.
pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    for message in messages {
        match message {
            Message::User(user) => {
                let text = transcript::content_text(&user.content, "");
                if !text.is_empty() {
                    parts.push(format!("[User]: {text}"));
                }
            }
            Message::Assistant(assistant) => {
                let calls: Vec<String> = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        Content::ToolCall { name, arguments, .. } => {
                            let arguments: Vec<String> =
                                js::object_entries(arguments).into_iter().map(|(key, value)| format!("{key}={}", js::stringify(value))).collect();
                            Some(format!("{name}({})", arguments.join(", ")))
                        }
                        _ => None,
                    })
                    .collect();
                if assistant.content.iter().any(|block| matches!(block, Content::Text { .. })) {
                    parts.push(format!("[Assistant]: {}", blocks_text(&assistant.content, "\n")));
                }
                if !calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
                }
            }
            Message::ToolResult(result) => {
                let text = blocks_text(&result.content, "");
                if !text.is_empty() {
                    parts.push(format!("[Tool result]: {}", truncate_for_summary(&text, TOOL_RESULT_MAX_CHARS)));
                }
            }
            Message::System(_) => {}
        }
    }
    parts.join("\n\n")
}

// ---------------------------------------------------------------------------------------------
// Asking the model
// ---------------------------------------------------------------------------------------------

// What a summary request runs with: the session's model, credential and thinking level, and the
// retry settings for transient failures.
pub struct SummaryRequest<'a> {
    pub model: &'a Model,
    pub credential: &'a Credential,
    pub thinking_level: &'a str,
    pub retry: RetrySettings,
    pub transport: &'a str,
    pub cwd: &'a std::path::Path,
    pub cancel: CancellationToken,
}

// What the summary request's retries report, for pi's summarization_retry_* events.
pub enum RetryNotice {
    Scheduled { attempt: u32, max_attempts: u32, delay_ms: u64, error_message: String },
    AttemptStart,
    Finished,
}

pub type RetryListener<'a> = dyn FnMut(RetryNotice) + Send + 'a;

// The finished summary and what pi stores with it (pi's CompactionResult).
#[derive(Debug, Clone)]
pub struct Summary {
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: u64,
    pub usage: Usage,
    // {readFiles, modifiedFiles}, which the next compaction reads back.
    pub details: Value,
}

// One summary request (compaction.ts:595-638, 682-693): pi's system prompt and the prompt as one
// user message. It goes uncached under a fresh session id: pi's cacheRetention "none", which
// providers that route by session (opencode) still get the id for. Thinking is the session's level
// when the model reasons.
async fn request_once(request: &SummaryRequest<'_>, prompt_text: &str) -> AssistantMessage {
    let messages = [
        Message::System(SystemMessage {
            content: TextOrBlocks::Text(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
            sections: None,
            tools_added: None,
            tools_removed: None,
            timestamp: 0,
        }),
        Message::User(UserMessage { content: TextOrBlocks::Blocks(vec![Content::text(prompt_text)]), timestamp: clock::now_ms() }),
    ];
    let session_id = uuid::Uuid::now_v7().to_string();
    let thinking_level = if request.model.reasoning && request.thinking_level != "off" { request.thinking_level } else { "off" };
    // Nobody watches a summary stream; the receiver only has to outlive it.
    let (sender, _receiver) = mpsc::unbounded_channel();
    provider::stream(
        Request {
            model: request.model,
            messages: &messages,
            credential: request.credential,
            session_id: Some(&session_id),
            cache: false,
            in_place: false,
            manager: false,
            transport: request.transport,
            thinking_level,
            session_file: None,
            cwd: request.cwd,
            cancel: request.cancel.child_token(),
        },
        sender,
    )
    .await
}

// The request, retried on transient failures the way pi's retryAssistantCall does
// (ai/src/utils/retry.ts:205-255): up to maxRetries more attempts with the usual backoff. An abort
// during a wait ends it as an aborted response.
async fn request_with_retry(request: &SummaryRequest<'_>, prompt_text: &str, notify: &mut RetryListener<'_>) -> AssistantMessage {
    let max_attempts = if request.retry.enabled { request.retry.max_retries } else { 0 };
    let mut attempt = 0;
    let mut retrying = false;
    loop {
        let response = request_once(request, prompt_text).await;
        if response.stop_reason != StopReason::Error || attempt >= max_attempts || !retry::is_retryable(&response) {
            if retrying {
                notify(RetryNotice::Finished);
            }
            return response;
        }
        attempt += 1;
        retrying = true;
        let delay_ms = retry::delay_ms(request.retry.base_delay_ms, request.retry.max_delay_ms, attempt);
        let error_message = response.error_message.clone().filter(|error| !error.is_empty()).unwrap_or_else(|| "Unknown error".into());
        notify(RetryNotice::Scheduled { attempt, max_attempts, delay_ms, error_message });
        tokio::select! {
            () = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
            () = request.cancel.cancelled() => {
                notify(RetryNotice::Finished);
                return AssistantMessage { stop_reason: StopReason::Aborted, error_message: None, ..response };
            }
        }
        notify(RetryNotice::AttemptStart);
    }
}

// The text of a summary response, or why it can't be stored (compaction.ts:585-593, 742-750). A
// length stop is a partial summary and must never become a checkpoint, nor may an empty one, which
// would keep nothing of what came before the cut.
fn summary_text(response: &AssistantMessage, label: &str) -> Result<String, String> {
    match response.stop_reason {
        StopReason::Error => {
            let error = response.error_message.as_deref().filter(|error| !error.is_empty()).unwrap_or("Unknown error");
            return Err(format!("{label} failed: {error}"));
        }
        StopReason::Length => return Err(format!("{label} failed: generation hit the token cap and the summary is incomplete")),
        _ => {}
    }
    if response.content.iter().any(|block| matches!(block, Content::ToolCall { .. })) {
        return Err(format!("{label} attempted to call a tool"));
    }
    Some(blocks_text(&response.content, "\n")).filter(|text| !text.trim().is_empty()).ok_or_else(|| format!("{label} returned empty text"))
}

// The prompt that summarizes `messages`, updating `previous_summary` when there is one
// (compaction.ts:696-729).
pub fn history_prompt(messages: &[AgentMessage], previous_summary: Option<&str>, custom_instructions: Option<&str>) -> String {
    // pi tests the previous summary for truth, so an empty one counts as none.
    let previous_summary = previous_summary.filter(|previous| !previous.is_empty());
    let mut instructions = if previous_summary.is_some() { UPDATE_SUMMARIZATION_PROMPT } else { SUMMARIZATION_PROMPT }.to_string();
    if let Some(focus) = custom_instructions.filter(|focus| !focus.is_empty()) {
        instructions = format!("{instructions}\n\nAdditional focus: {focus}");
    }
    let conversation = serialize_conversation(&to_model_messages(messages));
    let mut prompt_text = format!("<conversation>\n{conversation}\n</conversation>\n\n");
    if let Some(previous) = previous_summary {
        prompt_text.push_str(&format!("<previous-summary>\n{previous}\n</previous-summary>\n\n"));
    }
    prompt_text.push_str(&instructions);
    prompt_text
}

// The prompt that summarizes the start of a turn the cut splits (compaction.ts:1076-1081).
pub fn turn_prefix_prompt(messages: &[AgentMessage], custom_instructions: Option<&str>) -> String {
    let conversation = serialize_conversation(&to_model_messages(messages));
    let mut prompt_text = format!("# Conversation\n{conversation}\n\n# Instructions\n{TURN_PREFIX_SUMMARIZATION_PROMPT}");
    if let Some(focus) = custom_instructions.filter(|focus| !focus.is_empty()) {
        prompt_text = format!("{prompt_text}\n\nAdditional focus: {focus}");
    }
    prompt_text
}

// Two requests' usage added up (usage-totals.ts:31-52).
fn combine_usage(first: &Usage, second: &Usage) -> Usage {
    let optional_sum = |a: Option<u64>, b: Option<u64>| (a.is_some() || b.is_some()).then(|| a.unwrap_or(0) + b.unwrap_or(0));
    Usage {
        input: first.input + second.input,
        output: first.output + second.output,
        cache_read: first.cache_read + second.cache_read,
        cache_write: first.cache_write + second.cache_write,
        cache_write_1h: optional_sum(first.cache_write_1h, second.cache_write_1h),
        reasoning: optional_sum(first.reasoning, second.reasoning),
        total_tokens: first.total_tokens + second.total_tokens,
        cost: crate::message::Cost {
            input: first.cost.input + second.cost.input,
            output: first.cost.output + second.cost.output,
            cache_read: first.cost.cache_read + second.cost.cache_read,
            cache_write: first.cost.cache_write + second.cost.cache_write,
            total: first.cost.total + second.cost.total,
            plan: first.cost.plan.or(second.cost.plan),
            state: first.cost.state.max(second.cost.state),
            settles: first.cost.settles + second.cost.settles,
        },
    }
}

// Write the summary for a prepared compaction with the model (compaction.ts:952-1057).
pub async fn summarize(
    preparation: &Preparation,
    custom_instructions: Option<&str>,
    request: &SummaryRequest<'_>,
    notify: &mut RetryListener<'_>,
) -> Result<Summary, String> {
    summarize_with(preparation, custom_instructions, &mut async |prompt: &str| request_with_retry(request, prompt, notify).await).await
}

// Write the summary for a prepared compaction, getting each response from `complete`, which
// sends one prompt. A split turn gets two requests, the history and the turn's start, joined
// into one summary. The file lists go last.
pub async fn summarize_with(
    preparation: &Preparation,
    custom_instructions: Option<&str>,
    complete: &mut impl AsyncFnMut(&str) -> AssistantMessage,
) -> Result<Summary, String> {
    let previous = preparation.previous_summary.as_deref();
    let mut history_summary = async |previous: Option<&str>| -> Result<(String, Usage), String> {
        let response = complete(&history_prompt(&preparation.messages_to_summarize, previous, custom_instructions)).await;
        Ok((summary_text(&response, "Summarization")?, response.usage))
    };
    let (summary, usage) = if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
        let (history, history_usage) = if preparation.messages_to_summarize.is_empty() {
            (previous.unwrap_or("No prior history.").to_string(), None)
        } else {
            let (text, usage) = history_summary(previous).await?;
            (text, Some(usage))
        };
        let response = complete(&turn_prefix_prompt(&preparation.turn_prefix_messages, custom_instructions)).await;
        let (prefix, prefix_usage) = (summary_text(&response, "Turn prefix summarization")?, response.usage);
        let usage = match history_usage {
            Some(history_usage) => combine_usage(&history_usage, &prefix_usage),
            None => prefix_usage,
        };
        (format!("{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}"), usage)
    } else {
        history_summary(previous).await?
    };
    Ok(finish(preparation, summary, usage))
}

// Idle compaction's summary (agent.rs): `request` is the session's own, as its turns send it, and
// the summary prompt goes after its messages as one more, so the provider reads everything before
// it from its cache rather than a request of the summarizer's own (request_once) writing it all
// again. One request: the model sees the whole context, split turn or not. Not retried: a failure
// leaves the session as it was, for the next turn.
pub async fn summarize_in_place(preparation: &Preparation, request: Request<'_>) -> Result<Summary, String> {
    let prompt = format!("{IN_PLACE_PROMPT} {SUMMARIZATION_SYSTEM_PROMPT}\n\n{SUMMARIZATION_PROMPT}");
    let messages = [request.messages, &[Message::User(UserMessage { content: TextOrBlocks::Text(prompt), timestamp: clock::now_ms() })]].concat();
    let (sender, _receiver) = mpsc::unbounded_channel();
    let response = provider::stream(Request { messages: &messages, in_place: true, ..request }, sender).await;
    Ok(finish(preparation, summary_text(&response, "Summarization")?, response.usage))
}

// The summary with the file lists after it, and what pi stores with it.
fn finish(preparation: &Preparation, mut summary: String, usage: Usage) -> Summary {
    let (read_files, modified_files) = preparation.file_ops.lists();
    summary.push_str(&format_file_operations(&read_files, &modified_files));
    Summary {
        summary,
        first_kept_entry_id: preparation.first_kept_entry_id.clone(),
        tokens_before: preparation.tokens_before,
        usage,
        details: json!({"readFiles": read_files, "modifiedFiles": modified_files}),
    }
}
