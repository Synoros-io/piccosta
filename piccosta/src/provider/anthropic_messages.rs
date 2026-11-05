// Anthropic Messages through opencode's gateway.
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/api/anthropic-messages.ts.
// Only the key-authenticated Messages protocol lives here, not OAuth, Bedrock or Claude Code.

use super::{BODY_IDLE_TIMEOUT, Request, StreamEvent, clamp_thinking_level, estimate, new_output, responses, transcript, transform};
use crate::message::{AssistantMessage, Content, Message, StopReason};
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio::sync::mpsc;

const API: &str = "anthropic-messages";
// pi's simple-options.ts leaves this many tokens for the answer, then maps levels to budgets.
const MIN_ANSWER_TOKENS: u64 = 1024;
const THINKING_BUDGETS: [u64; 4] = [1024, 2048, 8192, 16384];
// Anthropic's wire version, also used by pi's Messages client.
pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";
// Anthropic's tool ids accept these characters and at most 64 of them (pi's convertMessages).
const TOOL_ID_LIMIT: usize = 64;

fn string(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn input_block(block: &Content) -> Option<Value> {
    match block {
        Content::Text { text, .. } if !text.trim().is_empty() => Some(json!({"type":"text", "text":text})),
        Content::Image { data, mime_type } => Some(json!({"type":"image", "source":{"type":"base64", "media_type":mime_type, "data":data}})),
        _ => None,
    }
}

// Fold prompt updates, repair tool pairs, then render Messages blocks. Consecutive tool results
// share a user turn because compatible endpoints reject one user turn per result.
fn build_body(request: &Request<'_>) -> Value {
    let model = request.model;
    let resolved = transcript::resolve_transcript(request.messages, false);
    let normalize = |id: &str, _: &AssistantMessage| transform::safe_id(id, TOOL_ID_LIMIT);
    let messages = transform::transform_messages(&resolved, model, &normalize);
    let mut system = Vec::new();
    let mut turns: Vec<Value> = Vec::new();
    for message in &messages {
        let (role, blocks) = match message {
            Message::System(prompt) => {
                let text = transcript::system_message_text(prompt);
                if !text.is_empty() {
                    system.push(json!({"type":"text", "text":text}));
                }
                continue;
            }
            Message::User(user) => ("user", user.content.clone().into_blocks().iter().filter_map(input_block).collect::<Vec<_>>()),
            Message::Assistant(assistant) => {
                let blocks = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        Content::Thinking { thinking_signature: Some(signature), redacted: Some(true), .. } => {
                            Some(json!({"type":"redacted_thinking", "data":signature}))
                        }
                        Content::Thinking { thinking, thinking_signature, .. } => {
                            let signature = thinking_signature.as_deref().unwrap_or_default();
                            if thinking.is_empty() && signature.is_empty() {
                                None
                            } else if !signature.is_empty() || model.compat_flag("allowEmptySignature") {
                                Some(json!({"type":"thinking", "thinking":thinking, "signature":signature}))
                            } else {
                                Some(json!({"type":"text", "text":thinking}))
                            }
                        }
                        Content::ToolCall { id, name, arguments, .. } => Some(json!({"type":"tool_use", "id":id, "name":name, "input":arguments})),
                        _ => input_block(block),
                    })
                    .collect();
                ("assistant", blocks)
            }
            Message::ToolResult(result) => {
                let blocks: Vec<_> = result.content.iter().filter_map(input_block).collect();
                let result = json!({"type":"tool_result", "tool_use_id":result.tool_call_id, "is_error":result.is_error, "content":blocks});
                if let Some(last) = turns.last_mut().filter(|last| last["role"] == "user") {
                    last["content"].as_array_mut().unwrap().push(result);
                    continue;
                }
                ("user", vec![result])
            }
        };
        if !blocks.is_empty() {
            turns.push(json!({"role":role, "content":blocks}));
        }
    }
    let tools: Vec<_> =
        transcript::current_tools(messages.iter().filter_map(|message| if let Message::System(system) = message { Some(system) } else { None }))
            .iter()
            .map(|tool| json!({"name":tool.name, "description":tool.description, "input_schema":tool.parameters}))
            .collect();
    // The gateway uses Anthropic's short-lived explicit cache markers, just as pi does.
    if request.cache {
        if let Some(last) = system.last_mut() {
            last["cache_control"] = json!({"type":"ephemeral"});
        }
        if let Some(last) =
            turns.last_mut().filter(|turn| turn["role"] == "user").and_then(|turn| turn["content"].as_array_mut()).and_then(|blocks| blocks.last_mut())
        {
            last["cache_control"] = json!({"type":"ephemeral"});
        }
    }
    let max_tokens = estimate::max_output_tokens(model.context_window, model.max_tokens, &resolved);
    let mut body = json!({"model":model.id, "messages":turns, "system":system, "max_tokens":max_tokens, "stream":true});
    if !tools.is_empty() {
        body["tools"] = tools.into();
    }
    if model.reasoning {
        let level = clamp_thinking_level(model, request.thinking_level);
        if level != "off" {
            if model.compat_flag("forceAdaptiveThinking") {
                body["thinking"] = json!({"type":"adaptive", "display":"summarized"});
                let effort = if level == "minimal" { "low" } else { level };
                body["output_config"] = json!({"effort":model.provider_thinking_level(effort)});
            } else {
                let index = match level {
                    "minimal" => 0,
                    "low" => 1,
                    "medium" => 2,
                    _ => 3,
                };
                let budget = THINKING_BUDGETS[index].min(max_tokens.saturating_sub(MIN_ANSWER_TOKENS));
                body["thinking"] = json!({"type":"enabled", "budget_tokens":budget});
            }
        } else if !matches!(model.thinking_level_map.as_ref().and_then(|map| map.get("off")), Some(None)) {
            body["thinking"] = json!({"type":"disabled"});
        }
    }
    body
}

// Each wire block has its own index. Keep that mapping instead of assuming unknown blocks
// occupy a content slot, and keep JSON fragments separate until the tool call closes.
#[derive(Default)]
struct Building {
    indexes: HashMap<u64, usize>,
    arguments: HashMap<usize, String>,
    stopped: bool,
}

fn usage(raw: &Value, output: &mut AssistantMessage, model: &crate::model::Model) {
    for (key, destination) in [
        ("input_tokens", &mut output.usage.input),
        ("output_tokens", &mut output.usage.output),
        ("cache_read_input_tokens", &mut output.usage.cache_read),
        ("cache_creation_input_tokens", &mut output.usage.cache_write),
    ] {
        if let Some(value) = raw[key].as_u64() {
            *destination = value;
        }
    }
    if let Some(value) = raw.pointer("/cache_creation/ephemeral_1h_input_tokens").and_then(Value::as_u64) {
        output.usage.cache_write_1h = Some(value);
    }
    if let Some(value) = raw.pointer("/output_tokens_details/thinking_tokens").and_then(Value::as_u64) {
        output.usage.reasoning = Some(value);
    }
    output.usage.total_tokens = output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write;
    responses::calculate_cost(model, &mut output.usage);
}

// The two new protocols use the same block events, even though their wire events differ.
pub(super) fn start_block(block: &Content, index: usize, send: &mut dyn FnMut(StreamEvent)) {
    send(match block {
        Content::Text { .. } => StreamEvent::TextStart { index },
        Content::Thinking { .. } => StreamEvent::ThinkingStart { index },
        Content::ToolCall { id, name, .. } => StreamEvent::ToolCallStart { index, id: id.clone(), name: name.clone() },
        Content::Image { .. } => return,
    });
}

pub(super) fn end_block(block: &Content, index: usize, send: &mut dyn FnMut(StreamEvent)) {
    send(match block {
        Content::Text { text, .. } => StreamEvent::TextEnd { index, content: text.clone() },
        Content::Thinking { thinking, .. } => StreamEvent::ThinkingEnd { index, content: thinking.clone() },
        Content::ToolCall { .. } => StreamEvent::ToolCallEnd { index, tool_call: block.clone() },
        Content::Image { .. } => return,
    });
}

fn handle(
    event: Value,
    output: &mut AssistantMessage,
    building: &mut Building,
    model: &crate::model::Model,
    send: &mut dyn FnMut(StreamEvent),
) -> Result<bool, String> {
    let wire_index = event["index"].as_u64().unwrap_or_default();
    match event["type"].as_str().unwrap_or_default() {
        "error" => return Err(event.pointer("/error/message").and_then(Value::as_str).unwrap_or("Anthropic stream error").to_string()),
        "message_start" => {
            let message = &event["message"];
            output.response_id = message["id"].as_str().map(str::to_string);
            output.response_model = message["model"].as_str().filter(|name| *name != model.id).map(str::to_string);
            usage(&message["usage"], output, model);
        }
        "content_block_start" => {
            let block = &event["content_block"];
            let content = match block["type"].as_str().unwrap_or_default() {
                "text" => Content::text(string(block, "text")),
                "thinking" => Content::Thinking { thinking: string(block, "thinking"), thinking_signature: Some(string(block, "signature")), redacted: None },
                "redacted_thinking" => {
                    Content::Thinking { thinking: "[Reasoning redacted]".into(), thinking_signature: Some(string(block, "data")), redacted: Some(true) }
                }
                "tool_use" => Content::ToolCall {
                    id: string(block, "id"),
                    name: string(block, "name"),
                    arguments: block["input"].as_object().cloned().unwrap_or_default(),
                    namespace: None,
                    arguments_error: None,
                    thought_signature: None,
                },
                "fallback" if !output.content.is_empty() => return Err("Anthropic performed an unsupported mid-output model fallback".into()),
                _ => return Ok(false),
            };
            let index = output.content.len();
            start_block(&content, index, send);
            output.content.push(content);
            building.indexes.insert(wire_index, index);
        }
        "content_block_delta" => {
            let Some(&index) = building.indexes.get(&wire_index) else { return Ok(false) };
            let delta = &event["delta"];
            match (&mut output.content[index], delta["type"].as_str().unwrap_or_default()) {
                (Content::Text { text, .. }, "text_delta") => {
                    let piece = string(delta, "text");
                    text.push_str(&piece);
                    send(StreamEvent::TextDelta { index, delta: piece });
                }
                (Content::Thinking { thinking, .. }, "thinking_delta") => {
                    let piece = string(delta, "thinking");
                    thinking.push_str(&piece);
                    send(StreamEvent::ThinkingDelta { index, delta: piece });
                }
                (Content::Thinking { thinking_signature, .. }, "signature_delta") => {
                    thinking_signature.get_or_insert_default().push_str(&string(delta, "signature"))
                }
                (Content::ToolCall { .. }, "input_json_delta") => {
                    let piece = string(delta, "partial_json");
                    building.arguments.entry(index).or_default().push_str(&piece);
                    send(StreamEvent::ToolCallDelta { index, delta: piece });
                }
                _ => {}
            }
        }
        "content_block_stop" => {
            if let Some(index) = building.indexes.remove(&wire_index) {
                if let Content::ToolCall { arguments, arguments_error, .. } = &mut output.content[index]
                    && let Some(raw) = building.arguments.remove(&index)
                {
                    (*arguments, *arguments_error) = responses::parse_tool_call_arguments(&raw);
                }
                end_block(&output.content[index], index, send);
            }
        }
        "message_delta" => {
            if let Some(reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                output.raw_stop_reason = Some(reason.into());
                output.stop_reason = match reason {
                    "end_turn" | "stop_sequence" | "pause_turn" => StopReason::Stop,
                    "max_tokens" | "model_context_window_exceeded" => StopReason::Length,
                    "tool_use" => StopReason::ToolUse,
                    _ => return Err(format!("Provider stopped with: {reason}")),
                };
            }
            usage(&event["usage"], output, model);
        }
        "message_stop" => {
            building.stopped = true;
            return Ok(true);
        }
        _ => {}
    }
    Ok(false)
}

pub async fn stream(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let mut output = new_output(request.model, API);
    let mut send = |event| {
        let _ = events.send(event);
    };
    let mut building = Building::default();
    let result = super::abortable(request, async {
        let body = build_body(request);
        let extra = vec![("x-api-key", request.credential.key.clone()), ("anthropic-version", ANTHROPIC_VERSION.into())];
        let response = super::post_openai(request.model, format!("{}/v1/messages", request.model.base_url.trim_end_matches('/')), request.credential, request.session_id, extra, &body).await?;
        send(StreamEvent::Start);
        super::sse::read_events(response, BODY_IDLE_TIMEOUT, "Anthropic", |event| handle(event, &mut output, &mut building, request.model, &mut send)).await?;
        if !building.stopped || output.stop_reason == StopReason::Pending { return Err("Anthropic stream ended without a stop reason".into()); }
        Ok(())
    })
    .await;
    super::settle(request, &mut output, result);
    output
}
