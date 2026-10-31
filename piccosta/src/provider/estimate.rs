// How many tokens a conversation holds: the last response's reported usage, plus an estimate for
// everything after it.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/utils/estimate.ts. Used to keep a
// request's output limit inside the context window, and by compaction to decide when to summarize.
// Lengths count UTF-16 code units, as JavaScript's `.length` does, so thresholds land where pi's do.

use super::transcript;
use crate::message::{Content, Message, StopReason, TextOrBlocks, Usage};

// pi's rule of thumb for characters per token, and what one image counts as (estimate.ts:15-16).
const CHARS_PER_TOKEN: f64 = 3.5;
const ESTIMATED_IMAGE_CHARS: usize = 4800;

fn utf16_length(text: &str) -> usize {
    text.encode_utf16().count()
}

// The context size a response reported: its total, or the sum of its parts when there's no total.
pub fn context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

fn tokens_for_chars(chars: usize) -> u64 {
    (chars as f64 / CHARS_PER_TOKEN).ceil() as u64
}

fn text_and_image_chars(blocks: &[Content]) -> usize {
    blocks
        .iter()
        .map(|block| match block {
            Content::Text { text, .. } => utf16_length(text),
            _ => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

// The estimated tokens of one message (estimate.ts:46-68).
pub fn message_tokens(message: &Message) -> u64 {
    match message {
        Message::System(system) => {
            let tools = |list: Option<String>| list.map(|json| tokens_for_chars(utf16_length(&json))).unwrap_or(0);
            tokens_for_chars(utf16_length(&transcript::system_message_text(system)))
                + tools(system.tools_added.as_ref().filter(|list| !list.is_empty()).and_then(|list| serde_json::to_string(list).ok()))
                + tools(system.tools_removed.as_ref().filter(|list| !list.is_empty()).and_then(|list| serde_json::to_string(list).ok()))
        }
        Message::User(user) => match &user.content {
            TextOrBlocks::Text(text) => tokens_for_chars(utf16_length(text)),
            TextOrBlocks::Blocks(blocks) => tokens_for_chars(text_and_image_chars(blocks)),
        },
        Message::ToolResult(result) => tokens_for_chars(text_and_image_chars(&result.content)),
        Message::Assistant(assistant) => {
            let chars: usize = assistant
                .content
                .iter()
                .map(|block| match block {
                    Content::Text { text, .. } => utf16_length(text),
                    Content::Thinking { thinking, .. } => utf16_length(thinking),
                    Content::ToolCall { name, arguments, .. } => {
                        utf16_length(name) + utf16_length(&serde_json::Value::Object(arguments.clone()).to_string())
                    }
                    Content::Image { .. } => 0,
                })
                .sum();
            tokens_for_chars(chars)
        }
    }
}

fn timestamp_of(message: &Message) -> i64 {
    match message {
        Message::System(system) => system.timestamp,
        Message::User(user) => user.timestamp,
        Message::Assistant(assistant) => assistant.timestamp,
        Message::ToolResult(result) => result.timestamp,
    }
}

// The estimated context size (estimate.ts:96-110): the latest usable reported usage plus estimates
// for the messages after it, or all estimates when no response reported usage. A response's usage
// stops applying once an earlier-dated message (a compaction summary, say) was put ahead of it.
pub fn context_estimate(messages: &[Message]) -> u64 {
    let mut latest_prefix_timestamp = i64::MIN;
    let mut usage_at: Option<(u64, usize)> = None;
    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message {
            let applies = assistant.timestamp >= latest_prefix_timestamp;
            let usable = !matches!(assistant.stop_reason, StopReason::Aborted | StopReason::Error);
            let tokens = context_tokens(&assistant.usage);
            if applies && usable && tokens > 0 {
                usage_at = Some((tokens, index));
            }
        }
        latest_prefix_timestamp = latest_prefix_timestamp.max(timestamp_of(message));
    }
    match usage_at {
        Some((tokens, index)) => tokens + messages[index + 1..].iter().map(message_tokens).sum::<u64>(),
        None => messages.iter().map(message_tokens).sum(),
    }
}

// Room left for output: the model's maximum, cut to what fits in the context window after the
// conversation and a safety margin (simple-options.ts:15-22).
pub fn max_output_tokens(context_window: u64, max_tokens: u64, messages: &[Message]) -> u64 {
    const CONTEXT_SAFETY_TOKENS: i64 = 4096;
    if context_window == 0 {
        return max_tokens.max(1);
    }
    let available = context_window as i64 - context_estimate(messages) as i64 - CONTEXT_SAFETY_TOKENS;
    max_tokens.min(available.max(1) as u64)
}
