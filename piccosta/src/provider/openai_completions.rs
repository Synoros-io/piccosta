// The OpenAI Chat Completions API ("openai-completions"), which most of opencode-go's models speak
// (DeepSeek, GLM, Kimi, MiniMax, MiMo, ...), and so do local servers (vLLM, llama.cpp).
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/api/openai-completions.ts, without
// the openai npm SDK: a POST to <base>/chat/completions, answered as server-sent events.
//
// Providers differ in small ways, which pi calls compat. pi also detects a dozen providers from
// their name and URL; here only opencode's gateway is detected, since it's the one piccosta ships,
// and any other provider states what differs in its models.json `compat`. Left out, because no
// piccosta model uses them: the other providers' detection, the zai, qwen, chat-template and
// OpenRouter thinking formats, session-affinity headers, PI_CACHE_RETENTION's 24-hour cache, the
// Mistral-style quirks (tool result names, an assistant turn after tool results, thinking sent
// back as text), servers without finish reasons or streamed usage, GitHub Copilot's headers,
// grammar ("custom") tools and thinking token budgets.

use super::estimate;
use super::responses::{self, calculate_cost, parse_tool_call_arguments};
use super::transcript::{self, ToolSupport};
use super::{clamp_thinking_level, is_opencode, new_output, transform, Request, StreamEvent, BODY_IDLE_TIMEOUT};
use crate::message::{AssistantMessage, Content, Message, StopReason, TextOrBlocks, ToolDefinition, Usage};
use crate::model::Model;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

const API: &str = "openai-completions";
// OpenAI's limit on tool-call ids (openai-completions.ts:1219).
const MAX_TOOL_CALL_ID_CHARS: usize = 40;
// The text standing in for a tool result that had only images, or nothing (openai-completions.ts:1420).
const IMAGE_ONLY_RESULT: &str = "(see attached image)";
const EMPTY_RESULT: &str = "(no tool output)";
// The reasoning fields servers stream thinking in, first non-empty one wins (llama.cpp uses
// reasoning_content, others reasoning).
const REASONING_FIELDS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];

// The provider quirks this file acts on (openai-completions.ts:1603-1705): opencode's defaults,
// then the model's own compat entries over them.
struct Compat {
    supports_store: bool,
    supports_developer_role: bool,
    supports_reasoning_effort: bool,
    max_tokens_field: String,
    requires_reasoning_content_on_assistant_messages: bool,
    thinking_format: String,
    supports_strict_mode: bool,
    supports_mid_conversation_system: bool,
    supports_mid_conversation_tool_additions: bool,
}

fn compat_of(model: &Model) -> Compat {
    let opencode = is_opencode(model);
    // DeepSeek's own API (pi's detection), and DeepSeek's models behind opencode (pi's catalog).
    let deepseek = model.provider == "deepseek" || model.base_url.contains("deepseek.com") || (opencode && model.id.starts_with("deepseek"));
    let flag = |key: &str, detected: bool| model.compat.as_ref().and_then(|compat| compat.get(key)).and_then(Value::as_bool).unwrap_or(detected);
    Compat {
        supports_store: flag("supportsStore", !opencode),
        supports_developer_role: flag("supportsDeveloperRole", !opencode),
        supports_reasoning_effort: flag("supportsReasoningEffort", true),
        max_tokens_field: model.compat_str("maxTokensField").unwrap_or(if opencode { "max_tokens" } else { "max_completion_tokens" }).to_string(),
        requires_reasoning_content_on_assistant_messages: flag("requiresReasoningContentOnAssistantMessages", deepseek),
        thinking_format: model.compat_str("thinkingFormat").unwrap_or(if deepseek { "deepseek" } else { "openai" }).to_string(),
        // OpenAI compatibility alone doesn't mean strict JSON-schema tools; opencode's gateway takes them.
        supports_strict_mode: flag("supportsStrictMode", opencode),
        supports_mid_conversation_system: flag("supportsMidConvoSystemMessages", false),
        supports_mid_conversation_tool_additions: flag("supportsMidConvoToolAdditions", false),
    }
}

fn tool_support(compat: &Compat) -> ToolSupport {
    if compat.supports_mid_conversation_system && compat.supports_mid_conversation_tool_additions { ToolSupport::Additions } else { ToolSupport::None }
}

// Tool declarations for Chat Completions: `strict` only where the provider takes it, true when the
// tool asked for constrained sampling and its schema can be made strict (openai-completions.ts:1492-1527).
fn convert_tools(tools: &[ToolDefinition], compat: &Compat) -> Vec<Value> {
    responses::convert_tools(tools, Some(false), compat.supports_strict_mode)
        .into_iter()
        .map(|declaration| {
            let mut function = Map::new();
            for key in ["name", "description", "parameters", "strict"] {
                if let Some(value) = declaration.get(key) {
                    function.insert(key.into(), value.clone());
                }
            }
            json!({"type": "function", "function": function})
        })
        .collect()
}

// A tool-call id from another API made fit for Chat Completions (openai-completions.ts:1208-1232):
// Responses' "call|item" ids become "call_item" with odd characters made "_", hashed down to 40
// characters when longer.
fn normalize_tool_call_id(id: &str, model: &Model) -> String {
    if let Some((call, item)) = id.split_once('|') {
        let clean = |part: &str| -> String { part.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect() };
        let (call_id, item_id) = (clean(call), clean(item));
        let combined = if item_id.is_empty() { call_id.clone() } else { format!("{call_id}_{item_id}") };
        if combined.len() <= MAX_TOOL_CALL_ID_CHARS {
            return combined;
        }
        let hash: String = responses::short_hash(id).chars().take(8).collect();
        let prefix: String = call_id.chars().take((MAX_TOOL_CALL_ID_CHARS - hash.len() - 1).max(1)).collect();
        return format!("{prefix}_{hash}");
    }
    if model.provider == "openai" && id.len() > MAX_TOOL_CALL_ID_CHARS {
        return id.chars().take(MAX_TOOL_CALL_ID_CHARS).collect();
    }
    id.to_string()
}

fn image_url(mime_type: &str, data: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": format!("data:{mime_type};base64,{data}")}})
}

// The conversation as Chat Completions messages (openai-completions.ts:1199-1490).
fn convert_messages(model: &Model, messages: &[Message], compat: &Compat) -> Vec<Value> {
    let resolved = transcript::resolve_transcript(messages, compat.supports_mid_conversation_system);
    let normalize = |id: &str, _source: &AssistantMessage| normalize_tool_call_id(id, model);
    let transformed = transform::transform_messages(&resolved, model, &normalize);
    let (_, folded_system_messages) = transcript::request_tools(&resolved, tool_support(compat));
    let instruction_role = if model.reasoning && compat.supports_developer_role { "developer" } else { "system" };
    let takes_images = model.input.iter().any(|kind| kind == "image");

    let mut out: Vec<Value> = Vec::new();
    let mut system_index = 0usize;
    let mut index = 0usize;
    while index < transformed.len() {
        match &transformed[index] {
            Message::System(system) => {
                // Tools added after the ones the top-level list holds go in a system message of
                // their own, as Kimi takes them.
                let anchored = system_index >= folded_system_messages;
                system_index += 1;
                if anchored && let Some(added) = system.tools_added.as_ref().filter(|added| !added.is_empty()) {
                    out.push(json!({"role": "system", "tools": convert_tools(added, compat)}));
                }
                let text = if index == 0 { transcript::system_message_text(system) } else { transcript::render_system_message_update(system) };
                if !text.is_empty() {
                    out.push(json!({"role": instruction_role, "content": text}));
                }
            }
            Message::User(user) => {
                match &user.content {
                    TextOrBlocks::Text(text) => out.push(json!({"role": "user", "content": text})),
                    TextOrBlocks::Blocks(blocks) => {
                        let parts: Vec<Value> = blocks
                            .iter()
                            .filter_map(|block| match block {
                                Content::Text { text, .. } if !text.is_empty() => Some(json!({"type": "text", "text": text})),
                                Content::Image { data, mime_type } => Some(image_url(mime_type, data)),
                                _ => None,
                            })
                            .collect();
                        if parts.is_empty() {
                            index += 1;
                            continue;
                        }
                        out.push(json!({"role": "user", "content": parts}));
                    }
                }
            }
            Message::Assistant(assistant) => out.extend(convert_assistant(model, assistant, compat)),
            Message::ToolResult(_) => {
                // A run of tool results goes out together; their images follow as one user message.
                let mut images = Vec::new();
                while let Some(Message::ToolResult(result)) = transformed.get(index) {
                    let text: Vec<&str> = result.content.iter().filter_map(|block| if let Content::Text { text, .. } = block { Some(text.as_str()) } else { None }).collect();
                    let text = text.join("\n");
                    let has_images = result.content.iter().any(|block| matches!(block, Content::Image { .. }));
                    let content = if !crate::js::trim(&text).is_empty() {
                        text
                    } else if has_images {
                        IMAGE_ONLY_RESULT.to_string()
                    } else {
                        EMPTY_RESULT.to_string()
                    };
                    out.push(json!({"role": "tool", "content": content, "tool_call_id": result.tool_call_id}));
                    if has_images && takes_images {
                        for block in &result.content {
                            if let Content::Image { data, mime_type } = block {
                                images.push(image_url(mime_type, data));
                            }
                        }
                    }
                    index += 1;
                }
                if !images.is_empty() {
                    let mut content = vec![json!({"type": "text", "text": "Attached image(s) from tool result:"})];
                    content.extend(images);
                    out.push(json!({"role": "user", "content": content}));
                }
                continue;
            }
        }
        index += 1;
    }
    out
}

// One assistant turn for replay (openai-completions.ts:1301-1400): its text as one string, its
// thinking in the reasoning field it streamed in, and its tool calls. None for a turn with nothing
// left to send, which providers reject.
fn convert_assistant(model: &Model, assistant: &AssistantMessage, compat: &Compat) -> Option<Value> {
    let mut message = Map::new();
    message.insert("role".into(), "assistant".into());
    message.insert("content".into(), Value::Null);

    let text: String = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            Content::Text { text, .. } if !crate::js::trim(text).is_empty() => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let thinking: Vec<(&str, Option<&str>)> = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            Content::Thinking { thinking, thinking_signature, .. } if !crate::js::trim(thinking).is_empty() => {
                Some((thinking.as_str(), thinking_signature.as_deref()))
            }
            _ => None,
        })
        .collect();

    if !text.is_empty() {
        message.insert("content".into(), text.into());
    }
    if let Some((_, signature)) = thinking.first() {
        // The field the thinking streamed in; opencode-go streams `reasoning` but takes it back as
        // `reasoning_content`.
        let field = match *signature {
            Some("reasoning") if model.provider == "opencode-go" => Some("reasoning_content"),
            Some(field) if REASONING_FIELDS.contains(&field) => Some(field),
            _ => None,
        };
        if let Some(field) = field {
            message.insert(field.into(), thinking.iter().map(|(text, _)| *text).collect::<Vec<_>>().join("\n").into());
        }
    }

    let calls: Vec<Value> = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            Content::ToolCall { id, name, arguments, .. } => Some(json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": Value::Object(arguments.clone()).to_string()},
            })),
            _ => None,
        })
        .collect();
    if !calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(calls));
    }
    // DeepSeek rejects a replayed assistant turn without the field once thinking was on.
    if compat.requires_reasoning_content_on_assistant_messages && model.reasoning && !message.contains_key("reasoning_content") {
        message.insert("reasoning_content".into(), "".into());
    }
    let has_content = message.get("content").and_then(Value::as_str).is_some_and(|text| !text.is_empty());
    (has_content || message.contains_key("tool_calls")).then_some(Value::Object(message))
}

// The request body (openai-completions.ts:806-1016), keys in pi's order.
pub fn build_body(model: &Model, messages: &[Message], cache_key: Option<&str>, cache: bool, thinking_level: &str) -> Value {
    let compat = compat_of(model);
    let resolved = transcript::resolve_transcript(messages, compat.supports_mid_conversation_system);
    let (request_tools, _) = transcript::request_tools(&resolved, tool_support(&compat));
    let converted = convert_messages(model, messages, &compat);

    let mut body = Map::new();
    body.insert("model".into(), model.id.clone().into());
    body.insert("messages".into(), Value::Array(converted));
    body.insert("stream".into(), true.into());
    // OpenAI's own API keys its cache by this; an uncached request (pi's "none") sends none.
    if let Some(key) = cache_key.filter(|_| cache && model.base_url.contains("api.openai.com")) {
        body.insert("prompt_cache_key".into(), key.chars().take(64).collect::<String>().into());
    }
    body.insert("stream_options".into(), json!({"include_usage": true}));
    if compat.supports_store {
        body.insert("store".into(), false.into());
    }
    let max_tokens = estimate::max_output_tokens(model.context_window, model.max_tokens, &resolved);
    if max_tokens > 0 {
        body.insert(compat.max_tokens_field.clone(), max_tokens.into());
    }
    if !request_tools.is_empty() {
        body.insert("tools".into(), Value::Array(convert_tools(&request_tools, &compat)));
    } else if resolved.iter().any(|message| matches!(message, Message::ToolResult(_)) || matches!(message, Message::Assistant(assistant) if assistant.content.iter().any(|block| matches!(block, Content::ToolCall { .. })))) {
        // Anthropic behind LiteLLM-style proxies wants `tools` whenever the history has tool use.
        body.insert("tools".into(), json!([]));
    }

    // Thinking: off arrives as no effort at all (openai-completions.ts streamSimple).
    let level = if model.reasoning { clamp_thinking_level(model, thinking_level) } else { "off" };
    let effort: Option<&str> = (level != "off").then_some(level);
    let mapped = |level: &str| -> Option<Option<String>> { model.thinking_level_map.as_ref().and_then(|map| map.get(level)).cloned() };
    let off_is_null = matches!(mapped("off"), Some(None));
    if model.reasoning {
        // DeepSeek's format switches thinking itself; every format names the effort.
        let deepseek = compat.thinking_format == "deepseek";
        if deepseek && effort.is_some() {
            body.insert("thinking".into(), json!({"type": "enabled"}));
        } else if deepseek && !off_is_null {
            body.insert("thinking".into(), json!({"type": "disabled"}));
        }
        if let Some(effort) = effort.filter(|_| compat.supports_reasoning_effort) {
            body.insert("reasoning_effort".into(), mapped(effort).flatten().unwrap_or_else(|| effort.to_string()).into());
        } else if effort.is_none()
            && !deepseek
            && compat.supports_reasoning_effort
            && let Some(Some(off)) = mapped("off")
        {
            body.insert("reasoning_effort".into(), off.into());
        }
    }
    Value::Object(body)
}

// Usage from a chunk (openai-completions.ts:1529-1570). Cached tokens are reads, wherever the
// provider puts them: prompt_tokens_details (OpenAI), prompt_cache_hit_tokens (DeepSeek), or top
// level (Kimi). Writes are counted apart, and input is what's left of the prompt.
fn parse_usage(raw: &Value, model: &Model) -> Usage {
    let number = |pointer: &str| raw.pointer(pointer).and_then(Value::as_u64);
    let prompt = number("/prompt_tokens").unwrap_or(0);
    let cache_read = number("/prompt_tokens_details/cached_tokens").or_else(|| number("/prompt_cache_hit_tokens")).or_else(|| number("/cached_tokens")).unwrap_or(0);
    let cache_write = number("/prompt_tokens_details/cache_write_tokens").unwrap_or(0);
    let input = prompt.saturating_sub(cache_read + cache_write);
    let output = number("/completion_tokens").unwrap_or(0);
    let mut usage = Usage {
        input,
        output,
        cache_read,
        cache_write,
        reasoning: Some(number("/completion_tokens_details/reasoning_tokens").unwrap_or(0)),
        total_tokens: input + output + cache_read + cache_write,
        ..Usage::default()
    };
    calculate_cost(model, &mut usage);
    usage
}

fn map_stop_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        other => (StopReason::Error, Some(format!("Provider finish_reason: {other}"))),
    }
}

// The blocks being built while the response streams.
struct Building {
    text: Option<usize>,
    thinking: Option<usize>,
    // Tool calls by stream index and by id, to the content index, with their argument text so far.
    calls_by_index: std::collections::HashMap<u64, usize>,
    calls_by_id: std::collections::HashMap<String, usize>,
    partial_arguments: std::collections::HashMap<usize, String>,
    has_finish_reason: bool,
}

// Handle one chunk (openai-completions.ts:559-692).
fn handle_chunk(chunk: &Value, model: &Model, output: &mut AssistantMessage, building: &mut Building, send: &mut dyn FnMut(StreamEvent)) {
    if output.response_id.is_none()
        && let Some(id) = chunk.get("id").and_then(Value::as_str)
    {
        output.response_id = Some(id.to_string());
    }
    if output.response_model.is_none()
        && let Some(name) = chunk.get("model").and_then(Value::as_str).filter(|name| !name.is_empty() && *name != model.id)
    {
        output.response_model = Some(name.to_string());
    }
    if let Some(usage) = chunk.get("usage").filter(|usage| usage.is_object()) {
        output.usage = parse_usage(usage, model);
    }
    let Some(choice) = chunk.get("choices").and_then(Value::as_array).and_then(|choices| choices.first()) else { return };
    // Moonshot puts usage on the choice.
    if chunk.get("usage").is_none_or(Value::is_null)
        && let Some(usage) = choice.get("usage").filter(|usage| usage.is_object())
    {
        output.usage = parse_usage(usage, model);
    }
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        output.raw_stop_reason = Some(reason.to_string());
        let (stop_reason, error) = map_stop_reason(reason);
        output.stop_reason = stop_reason;
        if error.is_some() {
            output.error_message = error;
        }
        building.has_finish_reason = true;
    }
    let Some(delta) = choice.get("delta") else { return };

    // Reasoning before content: some proxies put both in one delta, and thinking comes first.
    if let Some((field, piece)) = REASONING_FIELDS.iter().find_map(|field| delta.get(*field).and_then(Value::as_str).filter(|text| !text.is_empty()).map(|text| (*field, text))) {
        let signature = if model.provider == "opencode-go" && field == "reasoning" { "reasoning_content" } else { field };
        let index = *building.thinking.get_or_insert_with(|| {
            output.content.push(Content::Thinking { thinking: String::new(), thinking_signature: Some(signature.to_string()), redacted: None });
            let index = output.content.len() - 1;
            send(StreamEvent::ThinkingStart { index });
            index
        });
        if let Content::Thinking { thinking, .. } = &mut output.content[index] {
            thinking.push_str(piece);
        }
        send(StreamEvent::ThinkingDelta { index, delta: piece.to_string() });
    }

    if let Some(piece) = delta.get("content").and_then(Value::as_str).filter(|text| !text.is_empty()) {
        let index = *building.text.get_or_insert_with(|| {
            output.content.push(Content::text(""));
            let index = output.content.len() - 1;
            send(StreamEvent::TextStart { index });
            index
        });
        if let Content::Text { text, .. } = &mut output.content[index] {
            text.push_str(piece);
        }
        send(StreamEvent::TextDelta { index, delta: piece.to_string() });
    }

    for call in delta.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
        let stream_index = call.get("index").and_then(Value::as_u64);
        let id = call.get("id").and_then(Value::as_str).filter(|id| !id.is_empty());
        let name = call.pointer("/function/name").and_then(Value::as_str).unwrap_or_default();
        let existing = stream_index.and_then(|at| building.calls_by_index.get(&at).copied()).or_else(|| id.and_then(|id| building.calls_by_id.get(id).copied()));
        let index = match existing {
            Some(index) => index,
            None => {
                output.content.push(Content::ToolCall {
                    id: id.unwrap_or_default().to_string(),
                    name: name.to_string(),
                    arguments: Map::new(),
                    namespace: None,
                    arguments_error: None,
                    thought_signature: None,
                });
                let index = output.content.len() - 1;
                send(StreamEvent::ToolCallStart { index, id: id.unwrap_or_default().to_string(), name: name.to_string() });
                index
            }
        };
        if let Some(at) = stream_index {
            building.calls_by_index.entry(at).or_insert(index);
        }
        if let Some(id) = id {
            building.calls_by_id.insert(id.to_string(), index);
        }
        if let Content::ToolCall { id: call_id, name: call_name, .. } = &mut output.content[index] {
            if call_id.is_empty()
                && let Some(id) = id
            {
                *call_id = id.to_string();
            }
            if call_name.is_empty() && !name.is_empty() {
                *call_name = name.to_string();
            }
        }
        let piece = call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or_default();
        building.partial_arguments.entry(index).or_default().push_str(piece);
        send(StreamEvent::ToolCallDelta { index, delta: piece.to_string() });
    }
}

// Close every open block, in content order, with its final value.
fn finish_blocks(output: &mut AssistantMessage, building: &mut Building, send: &mut dyn FnMut(StreamEvent)) {
    for index in 0..output.content.len() {
        match &mut output.content[index] {
            Content::Text { text, .. } => {
                let content = text.clone();
                send(StreamEvent::TextEnd { index, content });
            }
            Content::Thinking { thinking, .. } => {
                let content = thinking.clone();
                send(StreamEvent::ThinkingEnd { index, content });
            }
            Content::ToolCall { arguments, arguments_error, .. } => {
                let text = building.partial_arguments.remove(&index).unwrap_or_default();
                let (parsed, error) = parse_tool_call_arguments(&text);
                *arguments = parsed;
                *arguments_error = error;
                let tool_call = output.content[index].clone();
                send(StreamEvent::ToolCallEnd { index, tool_call });
            }
            Content::Image { .. } => {}
        }
    }
}

pub async fn stream(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let mut send = |event: StreamEvent| {
        let _ = events.send(event);
    };
    let mut output = new_output(request.model, API);
    let mut building = Building {
        text: None,
        thinking: None,
        calls_by_index: Default::default(),
        calls_by_id: Default::default(),
        partial_arguments: Default::default(),
        has_finish_reason: false,
    };
    let result = tokio::select! {
        result = run(request, &mut output, &mut building, &mut send) => result,
        () = request.cancel.cancelled() => Err("Request was aborted".to_string()),
    };
    if result.is_ok() {
        finish_blocks(&mut output, &mut building, &mut send);
    }
    let failure = match result {
        Err(message) => Some(message),
        Ok(()) if output.stop_reason == StopReason::Aborted => Some("Request was aborted".to_string()),
        Ok(()) if output.stop_reason == StopReason::Error => Some(output.error_message.clone().unwrap_or_else(|| "Provider returned an error stop reason".into())),
        Ok(()) if !building.has_finish_reason || output.stop_reason == StopReason::Pending => Some("Stream ended without finish_reason".to_string()),
        Ok(()) => None,
    };
    if let Some(message) = failure {
        output.stop_reason = if request.cancel.is_cancelled() { StopReason::Aborted } else { StopReason::Error };
        output.error_message = Some(message);
    }
    output
}

async fn run(request: &Request<'_>, output: &mut AssistantMessage, building: &mut Building, send: &mut dyn FnMut(StreamEvent)) -> Result<(), String> {
    let model = request.model;
    let body = build_body(model, request.messages, request.session_id, request.cache, request.thinking_level);
    let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
    let response = super::post_openai(model, url, request.credential, request.session_id, Vec::new(), &body).await?;
    send(StreamEvent::Start);
    super::sse::read_events(response, BODY_IDLE_TIMEOUT, "Chat Completions", |chunk| {
        handle_chunk(&chunk, model, output, building, send);
        Ok(false)
    })
    .await
}
