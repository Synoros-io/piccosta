// The OpenAI Responses API as other providers serve it ("openai-responses"), e.g. opencode-go's
// GPT and Grok models.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/api/openai-responses.ts, without
// the openai npm SDK: the request is a POST to <base>/responses and the answer streams back as
// server-sent events, which responses.rs turns into a message.
//
// Left out: GitHub Copilot's headers, Sign in with ChatGPT's quirks for api.openai.com, per-turn
// effort via configuration_update items, and service tiers. None of piccosta's providers use them.

use super::estimate;
use super::responses::{self, ConvertOptions, StreamState};
use super::transcript;
use super::{clamp_thinking_level, new_output, Request, StreamEvent, BODY_IDLE_TIMEOUT};
use crate::message::{AssistantMessage, StopReason};
use crate::model::Model;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

const API: &str = "openai-responses";
// Providers whose two-part tool-call ids are kept on replay (openai-responses.ts:36).
const TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];
// The Responses API rejects max_output_tokens below 16 (openai-responses.ts:38).
const MIN_OUTPUT_TOKENS: u64 = 16;
const MAX_PROMPT_CACHE_KEY_CHARS: usize = 64;

fn compat_bool(model: &Model, key: &str, default: bool) -> bool {
    model.compat.as_ref().and_then(|compat| compat.get(key)).and_then(Value::as_bool).unwrap_or(default)
}

// The request body (openai-responses.ts:338-420), keys in pi's order.
pub fn build_body(model: &Model, messages: &[crate::message::Message], cache_key: Option<&str>, thinking_level: &str) -> Value {
    let supports_mid_conversation_system = compat_bool(model, "supportsMidConvoSystemMessages", false);
    let transcript_messages = transcript::resolve_transcript(messages, supports_mid_conversation_system);
    let supports_strict = compat_bool(model, "supportsStrictMode", false);
    let (request_tools, _) = transcript::request_tools(&transcript_messages, responses::tool_support(model));
    let input = responses::convert_messages(
        model,
        &transcript_messages,
        &ConvertOptions { include_system_prompt: true, tool_call_providers: &TOOL_CALL_PROVIDERS, default_strict: Some(false), supports_strict },
    );

    let mut body = Map::new();
    body.insert("model".into(), model.id.clone().into());
    body.insert("input".into(), Value::Array(input));
    body.insert("stream".into(), true.into());
    if let Some(key) = cache_key {
        body.insert("prompt_cache_key".into(), key.into());
    }
    body.insert("store".into(), false.into());
    if compat_bool(model, "supportsMaxOutputTokens", true) {
        let max_tokens = estimate::max_output_tokens(model.context_window, model.max_tokens, &transcript_messages);
        body.insert("max_output_tokens".into(), max_tokens.max(MIN_OUTPUT_TOKENS).into());
    }
    if !request_tools.is_empty() {
        body.insert("tools".into(), Value::Array(responses::convert_tools(&request_tools, Some(false), supports_strict)));
    }
    if model.reasoning {
        let level = clamp_thinking_level(model, thinking_level);
        if level != "off" {
            let effort = match model.thinking_level_map.as_ref().and_then(|map| map.get(level)) {
                Some(Some(value)) => value.clone(),
                _ => level.to_string(),
            };
            body.insert("reasoning".into(), json!({"effort": effort, "summary": "auto"}));
            body.insert("include".into(), json!(["reasoning.encrypted_content"]));
        } else {
            match model.thinking_level_map.as_ref().and_then(|map| map.get("off")) {
                Some(None) => {}
                Some(Some(value)) => {
                    body.insert("reasoning".into(), json!({"effort": value}));
                }
                None => {
                    body.insert("reasoning".into(), json!({"effort": "none"}));
                }
            }
        }
    }
    Value::Object(body)
}

// Stream one response. Output: the finished message, or a failed one with the reason.
pub async fn stream(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let mut send = |event: StreamEvent| {
        let _ = events.send(event);
    };
    let mut state = StreamState::new(request.model, new_output(request.model, API));
    let result = tokio::select! {
        result = run(request, &mut state, &mut send) => result,
        () = request.cancel.cancelled() => Err("Request was aborted".to_string()),
    };
    let mut output = state.output;
    if let Err(message) = result {
        output.stop_reason = if request.cancel.is_cancelled() { StopReason::Aborted } else { StopReason::Error };
        output.error_message = Some(message);
    } else if output.stop_reason == StopReason::Pending {
        output.stop_reason = StopReason::Error;
        output.error_message = Some("OpenAI Responses stream ended without a stop reason".to_string());
    }
    output
}

async fn run(request: &Request<'_>, state: &mut StreamState<'_>, send: &mut dyn FnMut(StreamEvent)) -> Result<(), String> {
    let model = request.model;
    // The id that keys the cache and backend affinity; none for an uncached request
    // (openai-responses.ts:188-189).
    let cache_session = request.session_id.filter(|_| request.cache);
    let cache_key: Option<String> = cache_session.map(|id| id.chars().take(MAX_PROMPT_CACHE_KEY_CHARS).collect());
    let body = build_body(model, request.messages, cache_key.as_deref(), request.thinking_level);

    // Session affinity, by the format the provider asks for (openai-responses.ts:293-303).
    // opencode's models ask for "openai-nosession" (pi's catalog).
    let mut affinity = Vec::new();
    if let Some(session) = cache_session {
        let format = model.compat_str("sessionAffinityFormat").unwrap_or(if super::is_opencode(model) { "openai-nosession" } else { "openai" });
        if format == "openai" {
            affinity.push(("session_id", session.to_string()));
        }
        affinity.push(("x-client-request-id", session.to_string()));
    }
    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
    let response = super::post_openai(model, url, request.credential, request.session_id, affinity, &body).await?;
    send(StreamEvent::Start);
    super::sse::read_events(response, BODY_IDLE_TIMEOUT, "OpenAI Responses", |event| {
        state.handle(&event, send)?;
        let kind = event.get("type").and_then(Value::as_str).unwrap_or_default();
        Ok(matches!(kind, "response.completed" | "response.incomplete"))
    })
    .await?;
    state.finish()
}
