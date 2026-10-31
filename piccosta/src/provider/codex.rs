// OpenAI's Codex backend, reached with a ChatGPT plan's login ("openai-codex-responses").
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/api/openai-codex-responses.ts.
//
// The request is a Responses API body. It goes over a websocket by default (codex_ws.rs: one
// connection per session, only the new input once a response has finished), and is POSTed to
// <base>/codex/responses with the answer as server-sent events when the websocket can't be used.
// Left out: zstd request compression on the SSE path (it saves upload bytes, nothing the model
// sees), pi's provider-level retries, which piccosta never turned on (retries happen in the agent
// loop), and the transport-failure diagnostic pi attaches to a message that fell back to SSE.

use super::codex_ws::{self, Failure, Target};
use super::responses::{self, ConvertOptions, StreamState};
use super::transcript;
use super::{clamp_thinking_level, new_output, user_agent, Request, StreamEvent, BODY_IDLE_TIMEOUT, HEADER_TIMEOUT};
use crate::message::{AssistantMessage, Message, StopReason};
use crate::model::Model;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

const API: &str = "openai-codex-responses";
const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
// The instructions Codex gets when the conversation has no system prompt (openai-codex-responses.ts:578).
const FALLBACK_INSTRUCTIONS: &str = "You are a helpful assistant.";
// Providers whose two-part "call|item" tool-call ids are kept when replayed to Codex (line 65).
const TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];
// Longest prompt_cache_key the backend takes, in code points (openai-prompt-cache.ts:3-8).
const MAX_PROMPT_CACHE_KEY_CHARS: usize = 64;
// Response statuses Codex reports that the Responses stream understands; any other reads as none.
const KNOWN_STATUSES: [&str; 6] = ["completed", "incomplete", "failed", "cancelled", "queued", "in_progress"];

// <base>/codex/responses, accepting a base that already ends in /codex or the full path.
pub(crate) fn endpoint(base_url: &str) -> String {
    let raw = if base_url.trim().is_empty() { DEFAULT_BASE_URL } else { base_url };
    let normalized = raw.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        normalized.to_string()
    } else if normalized.ends_with("/codex") {
        format!("{normalized}/responses")
    } else {
        format!("{normalized}/codex/responses")
    }
}

// The same endpoint as a websocket url: https becomes wss, http becomes ws (line 671).
fn websocket_url(base_url: &str) -> String {
    let url = endpoint(base_url);
    if let Some(rest) = url.strip_prefix("https:") {
        format!("wss:{rest}")
    } else if let Some(rest) = url.strip_prefix("http:") {
        format!("ws:{rest}")
    } else {
        url
    }
}

// Codex takes strict tool schemas unless the model says otherwise (openai-codex-responses.ts:552).
pub(super) fn supports_strict(model: &Model) -> bool {
    model.compat.as_ref().and_then(|compat| compat.get("supportsStrictMode")).and_then(Value::as_bool).unwrap_or(true)
}

// The request body (openai-codex-responses.ts:542-622), keys in pi's order.
pub fn build_body(model: &Model, messages: &[Message], cache_key: Option<&str>, thinking_level: &str) -> Value {
    let transcript_messages = transcript::resolve_transcript(messages, model.compat_flag("supportsMidConvoSystemMessages"));
    let support = responses::tool_support(model);
    let (request_tools, _) = transcript::request_tools(&transcript_messages, support);
    let input = responses::convert_messages(
        model,
        &transcript_messages,
        &ConvertOptions {
            include_system_prompt: false,
            tool_call_providers: &TOOL_CALL_PROVIDERS,
            default_strict: None,
            supports_strict: supports_strict(model),
        },
    );
    let instructions = match transcript_messages.first() {
        Some(Message::System(system)) => transcript::system_message_text(system),
        _ => String::new(),
    };

    let mut body = Map::new();
    body.insert("model".into(), model.id.clone().into());
    body.insert("store".into(), false.into());
    body.insert("stream".into(), true.into());
    body.insert("instructions".into(), if instructions.is_empty() { FALLBACK_INSTRUCTIONS.into() } else { instructions.into() });
    body.insert("input".into(), Value::Array(input));
    body.insert("text".into(), json!({"verbosity": "low"}));
    body.insert("include".into(), json!(["reasoning.encrypted_content"]));
    if let Some(key) = cache_key {
        body.insert("prompt_cache_key".into(), key.into());
    }
    body.insert("tool_choice".into(), "auto".into());
    body.insert("parallel_tool_calls".into(), true.into());
    if !request_tools.is_empty() {
        body.insert("tools".into(), Value::Array(responses::convert_tools(&request_tools, None, supports_strict(model))));
    }

    // Effort: the clamped level through the model's map. "off" asks for the model's own off value
    // (or "none"), and a level the map marks null sends no reasoning field at all.
    let level = clamp_thinking_level(model, thinking_level);
    let off_value = || -> Option<String> {
        match model.thinking_level_map.as_ref().and_then(|map| map.get("off")) {
            Some(None) => None,
            Some(Some(value)) => Some(value.clone()),
            None => Some("none".to_string()),
        }
    };
    if level == "off" {
        if model.reasoning
            && let Some(effort) = off_value() {
                body.insert("reasoning".into(), json!({"effort": effort}));
            }
    } else {
        let effort = match model.thinking_level_map.as_ref().and_then(|map| map.get(level)) {
            Some(None) => None,
            Some(Some(value)) => Some(value.clone()),
            None => Some(level.to_string()),
        };
        if let Some(effort) = effort {
            body.insert("reasoning".into(), json!({"effort": effort, "summary": "auto"}));
        }
    }
    Value::Object(body)
}

// The error text for a failed HTTP response (openai-codex-responses.ts:1618-1647): ChatGPT's
// usage-limit errors get a readable message with the plan and when it resets; others give the
// server's message, or the body, or the status text.
fn error_text(status: reqwest::StatusCode, body: &str) -> String {
    let fallback = if body.is_empty() { status.canonical_reason().unwrap_or("Request failed").to_string() } else { body.to_string() };
    let Ok(parsed) = serde_json::from_str::<Value>(body) else { return fallback };
    let Some(error) = parsed.get("error").filter(|error| error.is_object()) else { return fallback };
    let code = error.get("code").or(error.get("type")).and_then(Value::as_str).unwrap_or_default().to_lowercase();
    let usage_limit = ["usage_limit_reached", "usage_not_included", "rate_limit_exceeded"].iter().any(|needle| code.contains(needle))
        || status.as_u16() == 429;
    let friendly = usage_limit.then(|| {
        let plan = error.get("plan_type").and_then(Value::as_str).map(|plan| format!(" ({} plan)", plan.to_lowercase())).unwrap_or_default();
        let when = error.get("resets_at").and_then(Value::as_f64).map(|resets_at| {
            let minutes = ((resets_at * 1000.0 - crate::clock::now_ms() as f64) / 60_000.0).round().max(0.0);
            format!(" Try again in ~{minutes} min.")
        }).unwrap_or_default();
        format!("You have hit your ChatGPT usage limit{plan}.{when}").trim().to_string()
    });
    // pi throws the friendly text when there is one, else the server's message.
    friendly.or_else(|| error.get("message").and_then(Value::as_str).map(str::to_string)).unwrap_or(fallback)
}

// Stream one response from Codex. Output: the finished message, or a failed one with the reason.
pub async fn stream(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let model = request.model;
    let mut send = |event: StreamEvent| {
        let _ = events.send(event);
    };
    let output = new_output(model, API);
    let mut state = StreamState::new(model, output);
    let result = tokio::select! {
        result = run(request, &mut state, &mut send) => result,
        () = request.cancel.cancelled() => Err("Request was aborted".to_string()),
    };
    let mut output = state.output;
    if let Err(message) = result {
        output.stop_reason = if request.cancel.is_cancelled() { StopReason::Aborted } else { StopReason::Error };
        output.error_message = Some(message);
    }
    output
}

async fn run(request: &Request<'_>, state: &mut StreamState<'_>, send: &mut dyn FnMut(StreamEvent)) -> Result<(), String> {
    let model = request.model;
    let account_id = request.credential.account_id.as_deref().ok_or("Failed to extract accountId from token")?;
    // An uncached request sends no key and no session headers (openai-codex-responses.ts:280-281).
    let cache_key: Option<String> = request.session_id.filter(|_| request.cache).map(|id| id.chars().take(MAX_PROMPT_CACHE_KEY_CHARS).collect());
    let body = build_body(model, request.messages, cache_key.as_deref(), request.thinking_level);

    // The websocket first, unless the transport says SSE or this session already fell back
    // (openai-codex-responses.ts:311-378). A missing continuation on the server, or the
    // connection limit before the response began, is retried once; any other connection failure
    // before the response began moves the session to SSE for the rest of the process; a failure
    // after it began, and Codex's own errors, end the request.
    if request.transport != "sse" && !codex_ws::fallback_active(cache_key.as_deref()) {
        // Without a cached session the socket's request id is a fresh one (line 287).
        let request_id = cache_key.clone().unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
        let mut headers = vec![("originator".to_string(), "pi".to_string()), ("User-Agent".to_string(), user_agent())];
        headers.extend(model.headers.iter().flatten().map(|(name, value)| (name.clone(), value.clone())));
        headers.extend([
            ("Authorization".to_string(), format!("Bearer {}", request.credential.key)),
            ("chatgpt-account-id".to_string(), account_id.to_string()),
            ("OpenAI-Beta".to_string(), codex_ws::OPENAI_BETA.to_string()),
            ("x-client-request-id".to_string(), request_id.clone()),
            ("session-id".to_string(), request_id),
        ]);
        let target = Target {
            url: websocket_url(&model.base_url),
            headers,
            session: cache_key.as_deref(),
            account_id,
            continue_responses: matches!(request.transport, "auto" | "websocket-cached"),
        };
        let Value::Object(body_map) = &body else { unreachable!("build_body builds an object") };
        let (mut retried_missing, mut retried_limit) = (false, false);
        loop {
            let mut started = false;
            let failure = match codex_ws::stream(&target, body_map, model, state, send, &mut started).await {
                Ok(()) => return state.finish(),
                Err(failure) => failure,
            };
            match failure {
                Failure::Api { code: Some(code), .. } if code == "previous_response_not_found" && !retried_missing && !started => {
                    retried_missing = true;
                }
                Failure::Api { code: Some(code), .. } if code == "websocket_connection_limit_reached" && !started => {
                    if retried_limit {
                        codex_ws::mark_fallback(cache_key.as_deref());
                        break;
                    }
                    retried_limit = true;
                }
                Failure::Api { message, .. } | Failure::Protocol(message) => return Err(message),
                Failure::Transport(message) => {
                    if started {
                        return Err(message);
                    }
                    codex_ws::mark_fallback(cache_key.as_deref());
                    break;
                }
            }
        }
    }

    let mut http = reqwest::Client::builder()
        .user_agent(user_agent())
        .build()
        .map_err(|error| error.to_string())?
        .post(endpoint(&model.base_url))
        // Codex wants an originator on every request. pi sends "pi"; it's kept until another value
        // is known to be accepted.
        .header("originator", "pi");
    for (name, value) in model.headers.iter().flatten() {
        http = http.header(name, value);
    }
    http = http
        .header("Authorization", format!("Bearer {}", request.credential.key))
        .header("chatgpt-account-id", account_id)
        .header("OpenAI-Beta", "responses=experimental")
        .header("accept", "text/event-stream")
        .header("content-type", "application/json");
    if let Some(key) = &cache_key {
        http = http.header("session-id", key).header("x-client-request-id", key);
    }

    let response = tokio::time::timeout(HEADER_TIMEOUT, http.body(body.to_string()).send())
        .await
        .map_err(|_| format!("Codex SSE response headers timed out after {}ms", HEADER_TIMEOUT.as_millis()))?
        .map_err(|error| error.to_string())?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(error_text(status, &text));
    }
    send(StreamEvent::Start);

    super::sse::read_events(response, BODY_IDLE_TIMEOUT, "Codex", |event| {
        if handle_codex_event(event, state, send)? { Ok(true) } else { Ok(false) }
    })
    .await?;
    state.finish()
}

// Codex's own error events (openai-codex-responses.ts:752-791): an `error` event, or a failed
// response. Output: the error's text and its code, which the websocket's retry rules read; None
// for any other event.
pub(super) fn api_error(event: &Value) -> Option<(String, Option<String>)> {
    match event.get("type").and_then(Value::as_str)? {
        "error" => {
            let nested = event.get("error");
            let field = |key: &str| {
                event.get(key).and_then(Value::as_str).or_else(|| nested.and_then(|error| error.get(key)).and_then(Value::as_str)).map(str::to_string)
            };
            let code = field("code");
            let detail = field("message").filter(|message| !message.is_empty()).or_else(|| code.clone().filter(|code| !code.is_empty())).unwrap_or_else(|| event.to_string());
            Some((format!("Codex error: {detail}"), code))
        }
        "response.failed" => {
            let message = event.pointer("/response/error/message").and_then(Value::as_str).filter(|message| !message.is_empty());
            let code = event.pointer("/response/error/code").and_then(Value::as_str).map(str::to_string);
            Some((message.unwrap_or("Codex response failed").to_string(), code))
        }
        _ => None,
    }
}

// One Codex event (openai-codex-responses.ts:765-810). Codex's own error events become errors;
// its terminal events (done, completed, incomplete) are read as response.completed with the status
// checked, and end the stream. Output: true at the terminal event.
pub(super) fn handle_codex_event(event: Value, state: &mut StreamState<'_>, send: &mut dyn FnMut(StreamEvent)) -> Result<bool, String> {
    if let Some((message, _)) = api_error(&event) {
        return Err(message);
    }
    let kind = event.get("type").and_then(Value::as_str).unwrap_or_default().to_string();
    match kind.as_str() {
        "" => Ok(false),
        "response.done" | "response.completed" | "response.incomplete" => {
            let mut response = event.get("response").cloned().unwrap_or(Value::Null);
            if let Some(end_turn) = response.get("end_turn").and_then(Value::as_bool) {
                state.output.end_turn = Some(end_turn);
            }
            if let Some(object) = response.as_object_mut() {
                let known = object.get("status").and_then(Value::as_str).is_some_and(|status| KNOWN_STATUSES.contains(&status));
                if !known {
                    object.remove("status");
                }
            }
            state.handle(&json!({"type": "response.completed", "response": response}), send)?;
            Ok(true)
        }
        _ => {
            state.handle(&event, send)?;
            Ok(false)
        }
    }
}
