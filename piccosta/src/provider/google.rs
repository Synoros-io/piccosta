// Google's GenerateContent protocol through opencode's gateway.
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/api/google-generative-ai.ts
// and google-shared.ts. No SDK, OAuth, Vertex or Cloud Code Assist is needed here.

use super::anthropic_messages::{end_block, start_block};
use super::{BODY_IDLE_TIMEOUT, Request, StreamEvent, clamp_thinking_level, estimate, new_output, responses, transcript, transform};
use crate::message::{AssistantMessage, Content, Message, StopReason};
use serde_json::{Value, json};
use tokio::sync::mpsc;

const API: &str = "google-generative-ai";
// Google function-call ids, as normalized in pi's google-shared.ts.
const TOOL_ID_LIMIT: usize = 64;

fn input_part(block: &Content) -> Option<Value> {
    match block {
        Content::Text { text, .. } => Some(json!({"text":text})),
        Content::Image { data, mime_type } => Some(json!({"inlineData":{"mimeType":mime_type,"data":data}})),
        _ => None,
    }
}

// Signatures only mean something on the provider and model that made them. transform_messages
// already clears foreign thinking and text signatures, but tool parts need that check too.
fn assistant_part(block: &Content, same_model: bool) -> Option<Value> {
    let (mut part, signature) = match block {
        Content::Text { text, text_signature } => (json!({"text":text}), text_signature.as_deref()),
        Content::Thinking { thinking, thinking_signature, .. } => (json!({"text":thinking,"thought":true}), thinking_signature.as_deref()),
        Content::ToolCall { id, name, arguments, thought_signature, .. } => {
            (json!({"functionCall":{"id":id,"name":name,"args":arguments}}), thought_signature.as_deref())
        }
        _ => return None,
    };
    if let Some(signature) = signature.filter(|signature| same_model && !signature.is_empty()) {
        part["thoughtSignature"] = signature.into();
    }
    if part.get("text").and_then(Value::as_str).is_some_and(|text| text.trim().is_empty()) && part.get("thoughtSignature").is_none() {
        return None;
    }
    Some(part)
}

fn build_body(request: &Request<'_>) -> Result<Value, String> {
    let model = request.model;
    let resolved = transcript::resolve_transcript(request.messages, false);
    let normalize = |id: &str, _: &AssistantMessage| transform::safe_id(id, TOOL_ID_LIMIT);
    let messages = transform::transform_messages(&resolved, model, &normalize);
    let mut contents: Vec<Value> = Vec::new();
    let mut system = String::new();
    for message in &messages {
        let (role, parts) = match message {
            Message::System(prompt) => {
                system = transcript::system_message_text(prompt);
                continue;
            }
            Message::User(user) => ("user", user.content.clone().into_blocks().iter().filter_map(input_part).collect::<Vec<_>>()),
            Message::Assistant(assistant) => {
                ("model", assistant.content.iter().filter_map(|block| assistant_part(block, transform::is_same_model(assistant, model))).collect())
            }
            Message::ToolResult(result) => {
                let text = result
                    .content
                    .iter()
                    .filter_map(|block| if let Content::Text { text, .. } = block { Some(text.as_str()) } else { None })
                    .collect::<Vec<_>>()
                    .join("\n");
                // Gemini rejects whitespace-only output (a bare \r\n from bash) with a 400, so it goes as
                // empty (pi #8720).
                let text = if text.trim().is_empty() { String::new() } else { text };
                let response = if result.is_error { json!({"error":text}) } else { json!({"output":text}) };
                let mut part = json!({"functionResponse":{"id":result.tool_call_id,"name":result.tool_name,"response":response}});
                let images: Vec<_> = result.content.iter().filter(|block| matches!(block, Content::Image { .. })).filter_map(input_part).collect();
                if !images.is_empty() {
                    part["functionResponse"]["parts"] = images.into();
                }
                if let Some(last) = contents.last_mut().filter(|last| {
                    last["role"] == "user" && last["parts"].as_array().is_some_and(|parts| parts.iter().any(|part| part.get("functionResponse").is_some()))
                }) {
                    last["parts"].as_array_mut().unwrap().push(part);
                    continue;
                }
                ("user", vec![part])
            }
        };
        if !parts.is_empty() {
            contents.push(json!({"role":role,"parts":parts}));
        }
    }
    let mut body =
        json!({"contents":contents,"generationConfig":{"maxOutputTokens":estimate::max_output_tokens(model.context_window, model.max_tokens, &resolved)}});
    if !system.is_empty() {
        body["systemInstruction"] = json!({"parts":[{"text":system}]});
    }
    let tools = transcript::current_tools(messages.iter().filter_map(|message| if let Message::System(system) = message { Some(system) } else { None }));
    if !tools.is_empty() {
        let declarations: Vec<_> =
            tools.iter().map(|tool| json!({"name":tool.name,"description":tool.description,"parametersJsonSchema":tool.parameters})).collect();
        body["tools"] = json!([{"functionDeclarations":declarations}]);
    }
    if model.reasoning {
        let level = clamp_thinking_level(model, request.thinking_level);
        body["generationConfig"]["thinkingConfig"] = if level == "off" {
            json!({"thinkingBudget":0})
        } else {
            // The catalog supplies the supported discrete levels; don't guess them from a model
            // name (pi's older name matching misses new gateway aliases).
            let mapped = model.provider_thinking_level(level).unwrap_or_else(|| level.into()).to_uppercase();
            if !["MINIMAL", "LOW", "MEDIUM", "HIGH"].contains(&mapped.as_str()) {
                return Err(format!("Unsupported Google thinking level: {level} -> {mapped}"));
            }
            json!({"includeThoughts":true,"thinkingLevel":mapped})
        };
    }
    Ok(body)
}

// Keep each signed wire part on its own content block. A signature can belong to text, a thought,
// or a call, and moving it onto a neighbouring part breaks Gemini's next tool-result request.
fn handle(chunk: Value, output: &mut AssistantMessage, model: &crate::model::Model, send: &mut dyn FnMut(StreamEvent)) -> Result<bool, String> {
    if let Some(error) = chunk.get("error") {
        return Err(error["message"].as_str().map(str::to_string).unwrap_or_else(|| error.to_string()));
    }
    if output.response_id.is_none() {
        output.response_id = chunk["responseId"].as_str().map(str::to_string);
    }
    if output.response_model.is_none() {
        output.response_model = chunk["modelVersion"].as_str().filter(|name| *name != model.id).map(str::to_string);
    }
    if let Some(reason) = chunk.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
        return Err(format!("Google blocked prompt: {reason}"));
    }
    if let Some(candidate) = chunk["candidates"].as_array().and_then(|candidates| candidates.first()) {
        for part in candidate.pointer("/content/parts").and_then(Value::as_array).into_iter().flatten() {
            let signature = part["thoughtSignature"].as_str().filter(|value| !value.is_empty()).map(str::to_string);
            let index = output.content.len();
            let block = if let Some(call) = part.get("functionCall") {
                let name = call["name"].as_str().unwrap_or_default().to_string();
                let id = call["id"]
                    .as_str()
                    .filter(|id| !output.content.iter().any(|block| matches!(block, Content::ToolCall { id: existing, .. } if existing == id)))
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("{name}_{}_{}", crate::clock::now_ms(), index));
                Content::ToolCall {
                    external: false,
                    id,
                    name,
                    arguments: call["args"].as_object().cloned().unwrap_or_default(),
                    namespace: None,
                    arguments_error: None,
                    thought_signature: signature,
                }
            } else if part.get("text").is_some() || signature.is_some() {
                let text = part["text"].as_str().unwrap_or_default().to_string();
                if part["thought"] == true {
                    Content::Thinking { thinking: text, thinking_signature: signature, redacted: None }
                } else {
                    Content::Text { text, text_signature: signature }
                }
            } else {
                continue;
            };
            start_block(&block, index, send);
            send(match &block {
                Content::Text { text, .. } => StreamEvent::TextDelta { index, delta: text.clone() },
                Content::Thinking { thinking, .. } => StreamEvent::ThinkingDelta { index, delta: thinking.clone() },
                Content::ToolCall { arguments, .. } => StreamEvent::ToolCallDelta { index, delta: Value::Object(arguments.clone()).to_string() },
                Content::Image { .. } => unreachable!(),
            });
            end_block(&block, index, send);
            output.content.push(block);
        }
        if let Some(reason) = candidate["finishReason"].as_str() {
            output.raw_stop_reason = Some(reason.into());
            output.stop_reason = match reason {
                "STOP" if output.content.iter().any(|block| matches!(block, Content::ToolCall { .. })) => StopReason::ToolUse,
                "STOP" => StopReason::Stop,
                "MAX_TOKENS" => StopReason::Length,
                _ => return Err(format!("Provider stopped with: {reason}")),
            };
        }
    }
    if let Some(raw) = chunk.get("usageMetadata") {
        let number = |key: &str| raw[key].as_u64().unwrap_or_default();
        output.usage.cache_read = number("cachedContentTokenCount");
        output.usage.input = number("promptTokenCount").saturating_sub(output.usage.cache_read);
        output.usage.reasoning = Some(number("thoughtsTokenCount"));
        output.usage.output = number("candidatesTokenCount") + number("thoughtsTokenCount");
        output.usage.total_tokens = raw["totalTokenCount"].as_u64().unwrap_or(output.usage.input + output.usage.output + output.usage.cache_read);
        responses::calculate_cost(model, &mut output.usage);
    }
    Ok(false)
}

pub async fn stream(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let mut output = new_output(request.model, API);
    let mut send = |event| {
        let _ = events.send(event);
    };
    let result = super::abortable(request, async {
        let body = build_body(request)?;
        // The catalog base already includes the API version. Path-encode the id so a user
        // model cannot turn its id into a query string or a different endpoint.
        let mut url = reqwest::Url::parse(&format!("{}/", request.model.base_url.trim_end_matches('/'))).map_err(|error| error.to_string())?;
        url.path_segments_mut().map_err(|_| "Google base URL cannot hold path segments".to_string())?.pop_if_empty().push("models").push(&format!("{}:streamGenerateContent", request.model.id));
        url.query_pairs_mut().append_pair("alt", "sse");
        let extra = vec![("x-goog-api-key", request.credential.key.clone())];
        let response = super::post_openai(request.model, url.into(), request.credential, request.session_id, extra, &body).await?;
        send(StreamEvent::Start);
        super::sse::read_events(response, BODY_IDLE_TIMEOUT, "Google", |chunk| handle(chunk, &mut output, request.model, &mut send)).await?;
        if output.stop_reason == StopReason::Pending { return Err("Google stream ended without a finish reason".into()); }
        Ok(())
    })
    .await;
    super::settle(request, &mut output, result);
    output
}
