// Which failed responses are worth sending again, how long to wait, and which failures mean the
// conversation no longer fits the model.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/utils/retry.ts and overflow.ts.
// The patterns are pi's, matched case-insensitively anywhere in the error text. pi's status-code
// pattern needs a lookaround the regex crate doesn't have, so that one check is written out.

use crate::message::{AssistantMessage, StopReason};
use regex::{Regex, RegexBuilder};
use std::sync::LazyLock;

fn any_of(patterns: &[&str]) -> Regex {
    RegexBuilder::new(&patterns.join("|")).case_insensitive(true).build().expect("valid pattern")
}

// Account limits that reset in hours or need a person, not seconds (retry.ts:7-27).
static NON_RETRYABLE: LazyLock<Regex> = LazyLock::new(|| {
    any_of(&[
        "GoUsageLimitError",
        "FreeUsageLimitError",
        "Monthly usage limit reached",
        "available balance",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
        "subscription_sharing_usage_limit_exceeded",
        "subscription_sharing_workplace_individual_usage_limit_exceeded",
        // Not pi's: opencode answers a model it has withdrawn with "400 Upstream request failed:
        // Model is unavailable.", which the retryable "upstream request failed" caught, so a
        // withdrawn grok model was retried for four minutes (sweep, 2026-10-30).
        "model is unavailable",
    ])
});

// Load, transport and transient server failures (retry.ts:30-110).
static RETRYABLE: LazyLock<Regex> = LazyLock::new(|| {
    any_of(&[
        "overloaded",
        "server_busy",
        "servers are currently busy",
        "currently experiencing high demand",
        "model is at capacity",
        "rate.?limit",
        "too many requests",
        "service.?(?:is.?)?(?:temporarily.?)?unavailable",
        "server.?error",
        "internal.?error",
        "provider.?returned.?(?:an.?)?error",
        "exceeded request buffer limit while retrying upstream",
        "unable to forward upstream",
        "upstream.?request.?failed",
        "upstream_stream_read_error",
        "stream was interrupted",
        "http/?2 stream failed",
        "network.?error",
        "connection.?error",
        "connection.?refused",
        "connection.?reset",
        "connection.?lost",
        "connection closed before message completed",
        "dns.?error",
        "other side closed",
        "fetch failed",
        r"request to \S+ failed, reason:",
        "getaddrinfo",
        "ENOTFOUND",
        "EAI_AGAIN",
        "upstream.?connect",
        "reset before headers",
        "socket hang up",
        "socket connection was closed",
        "timed? out",
        "timeout",
        "terminated",
        "websocket.?closed",
        "websocket.?error",
        "ended without",
        "stream ended before message_stop",
        "stream ended before a terminal response event",
        "http2 request did not get a response",
        "stream disconnected before completion",
        "pending stream has been canceled",
        "retry delay",
        "you can retry your request",
        "try your request again",
        "please retry your request",
        "ResourceExhausted",
        "subscription_sharing_usage_unavailable",
        "subscription_sharing_user_unavailable",
    ])
});

// Status codes that mean "try again", only when not part of a longer word or number such as a
// request id or a byte count (pi#11057).
const RETRYABLE_STATUS_CODES: [&str; 7] = ["429", "500", "502", "503", "504", "520", "524"];

fn has_status_code(text: &str) -> bool {
    let bytes = text.as_bytes();
    let is_word = |at: Option<&u8>| at.is_some_and(|byte| byte.is_ascii_alphanumeric());
    RETRYABLE_STATUS_CODES.iter().any(|code| {
        text.match_indices(code).any(|(at, _)| !is_word(at.checked_sub(1).and_then(|before| bytes.get(before))) && !is_word(bytes.get(at + 3)))
    })
}

// Whether a response failed in a way worth retrying (retry.ts:266-271).
pub fn is_retryable(message: &AssistantMessage) -> bool {
    let Some(error) = message.error_message.as_deref().filter(|error| !error.is_empty()) else { return false };
    if message.stop_reason != StopReason::Error || NON_RETRYABLE.is_match(error) {
        return false;
    }
    RETRYABLE.is_match(error) || has_status_code(error)
}

// The wait before retry number `attempt` (1-based): base x 2^(attempt-1), capped (retry.ts:142-146).
pub fn delay_ms(base_delay_ms: u64, max_delay_ms: u64, attempt: u32) -> u64 {
    let factor = 2u64.checked_pow(attempt.saturating_sub(1)).unwrap_or(u64::MAX);
    base_delay_ms.saturating_mul(factor).min(max_delay_ms)
}

static OVERFLOW: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"prompt (?:is )?too long",
        r"prompt exceeds max length",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds the context window",
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        r"input token count.*exceeds the maximum",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"too large for model with \d+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"model_context_window_exceeded",
        r"prompt too long; exceeded (?:max )?context length",
        r"range of input length should be",
        r"context[_ ]length[_ ]exceeded",
        r"too many tokens",
        r"token limit exceeded",
        r"^(?:[\w -]+ API error \(413\):|413(?:\s|:|$))",
    ]
    .iter()
    .map(|pattern| RegexBuilder::new(pattern).case_insensitive(true).build().expect("valid pattern"))
    .collect()
});

static NOT_OVERFLOW: LazyLock<Regex> = LazyLock::new(|| any_of(&[r"^(?:Throttling error|Service unavailable):", "rate limit", "too many requests"]));

// Whether a response says the conversation no longer fits the model (overflow.ts:142-176): an
// overflow error, a "successful" response whose input already exceeds the window, or a length
// stop with no output and a full window.
pub fn is_context_overflow(message: &AssistantMessage, context_window: u64) -> bool {
    if message.stop_reason == StopReason::Error
        && let Some(error) = message.error_message.as_deref().filter(|error| !error.is_empty())
        && !NOT_OVERFLOW.is_match(error)
        && OVERFLOW.iter().any(|pattern| pattern.is_match(error))
    {
        return true;
    }
    let input = message.usage.input + message.usage.cache_read;
    if context_window > 0 && message.stop_reason == StopReason::Stop && input > context_window {
        return true;
    }
    context_window > 0 && message.stop_reason == StopReason::Length && message.usage.output == 0 && input as f64 >= context_window as f64 * 0.99
}

// Whether a length stop ended below the output limit the model was meant to have, which points at
// context pressure or a provider cutting it short, so one compact-and-retry may help
// (overflow.ts:184-186). Input: the response and the model's own maximum output, before any
// clamping to the context window.
pub fn is_recoverable_length(message: &AssistantMessage, desired_max_output: u64) -> bool {
    message.stop_reason == StopReason::Length && desired_max_output > 0 && message.usage.output < desired_max_output
}
