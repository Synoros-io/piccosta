// Making a conversation safe to send to one particular model.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/api/transform-messages.ts.
//
// A session can mix models (switching mid-conversation, or a subagent's history), so before a
// request the history is adjusted for the model it's going to:
// - images become a note when the model can't see images;
// - thinking from another model is dropped (its signatures only mean something to the model that
//   wrote them, and readable thinking sent as text would look like part of the reply);
// - another model's tool-call ids are rewritten to what this API accepts, and their results follow;
// - failed and aborted assistant turns are left out, since replaying half a turn can be rejected;
// - a tool call with no result gets a "No result provided" error result, so every call is answered.

use crate::clock;
use crate::message::{AssistantMessage, Content, Message, StopReason, TextOrBlocks, ToolResultMessage};
use crate::model::Model;
use std::collections::{HashMap, HashSet};

// What replaces images for a model without image input (transform-messages.ts:12-13).
const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";
// The result a tool call gets when the conversation has none for it.
const MISSING_TOOL_RESULT_TEXT: &str = "No result provided";

// Content with each run of images replaced by one placeholder text block.
fn replace_images(blocks: &[Content], placeholder: &str) -> Vec<Content> {
    let mut result = Vec::with_capacity(blocks.len());
    let mut previous_was_placeholder = false;
    for block in blocks {
        if matches!(block, Content::Image { .. }) {
            if !previous_was_placeholder {
                result.push(Content::text(placeholder));
            }
            previous_was_placeholder = true;
            continue;
        }
        previous_was_placeholder = matches!(block, Content::Text { text, .. } if text == placeholder);
        result.push(block.clone());
    }
    result
}

// Whether an assistant message came from exactly this model on this API.
pub fn is_same_model(message: &AssistantMessage, model: &Model) -> bool {
    message.provider == model.provider && message.api == model.api && message.model == model.id
}

// A tool-call id as Anthropic and Google take it: each character outside [a-zA-Z0-9_-] made "_",
// then the first `limit` of them (pi's convertMessages, google-shared.ts).
pub fn safe_id(id: &str, limit: usize) -> String {
    id.chars().map(|character| if character.is_ascii_alphanumeric() || character == '_' || character == '-' { character } else { '_' }).take(limit).collect()
}

// The history adjusted for `model`. Input: the conversation and a function that rewrites another
// model's tool-call id for this API (given the id and the message that made the call). Output:
// the conversation to convert to the wire format.
pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: &dyn Fn(&str, &AssistantMessage) -> String,
) -> Vec<Message> {
    let takes_images = model.input.iter().any(|kind| kind == "image");
    let mut renamed_ids: HashMap<String, String> = HashMap::new();

    // First pass: images, thinking and tool-call ids.
    let mut transformed = Vec::with_capacity(messages.len());
    for message in messages {
        let adjusted = match message {
            Message::User(user) if !takes_images => {
                let mut user = user.clone();
                if let TextOrBlocks::Blocks(blocks) = &user.content {
                    user.content = TextOrBlocks::Blocks(replace_images(blocks, NON_VISION_USER_IMAGE_PLACEHOLDER));
                }
                Message::User(user)
            }
            Message::ToolResult(result) => {
                let mut result = result.clone();
                if !takes_images {
                    result.content = replace_images(&result.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER);
                }
                if let Some(new_id) = renamed_ids.get(&result.tool_call_id) {
                    result.tool_call_id = new_id.clone();
                }
                Message::ToolResult(result)
            }
            Message::Assistant(assistant) => {
                let same_model = is_same_model(assistant, model);
                let mut adjusted = assistant.clone();
                adjusted.content = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        Content::Thinking { thinking, thinking_signature, redacted } => {
                            if !same_model {
                                return None;
                            }
                            // Signed or redacted blocks are kept even when empty: OpenAI's
                            // encrypted reasoning has no readable text but is needed for replay.
                            let keep = redacted == &Some(true) || thinking_signature.is_some() || !thinking.trim().is_empty();
                            keep.then(|| block.clone())
                        }
                        Content::Text { text, .. } if !same_model => Some(Content::text(text.clone())),
                        Content::ToolCall { external, id, name, arguments, namespace, arguments_error, .. } if !same_model => {
                            let new_id = normalize_tool_call_id(id, assistant);
                            if &new_id != id {
                                renamed_ids.insert(id.clone(), new_id.clone());
                            }
                            Some(Content::ToolCall {
                                external: *external,
                                id: new_id,
                                name: name.clone(),
                                arguments: arguments.clone(),
                                namespace: namespace.clone(),
                                arguments_error: arguments_error.clone(),
                                thought_signature: None,
                            })
                        }
                        other => Some(other.clone()),
                    })
                    .collect();
                Message::Assistant(adjusted)
            }
            other => other.clone(),
        };
        transformed.push(adjusted);
    }

    // Second pass: leave out failed turns, and answer every tool call. A system message that lands
    // between a call and its results waits until after them, so it never splits a call from its
    // answer.
    let mut result = Vec::with_capacity(transformed.len());
    let mut pending_calls: Vec<(String, String)> = Vec::new();
    let mut answered: HashSet<String> = HashSet::new();
    let mut held_system: Vec<Message> = Vec::new();
    let close_pending = |result: &mut Vec<Message>,
                         pending_calls: &mut Vec<(String, String)>,
                         answered: &mut HashSet<String>,
                         held_system: &mut Vec<Message>| {
        for (id, name) in pending_calls.drain(..) {
            if !answered.contains(&id) {
                result.push(Message::ToolResult(ToolResultMessage {
                    tool_call_id: id,
                    tool_name: name,
                    content: vec![Content::text(MISSING_TOOL_RESULT_TEXT)],
                    details: None,
                    usage: None,
                    is_error: true,
                    timestamp: clock::now_ms(),
                    duration_ms: None,
                }));
            }
        }
        answered.clear();
        result.append(held_system);
    };

    for message in transformed {
        match &message {
            Message::Assistant(assistant) => {
                close_pending(&mut result, &mut pending_calls, &mut answered, &mut held_system);
                if matches!(assistant.stop_reason, StopReason::Error | StopReason::Aborted) {
                    continue;
                }
                pending_calls = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        Content::ToolCall { id, name, .. } => Some((id.clone(), name.clone())),
                        _ => None,
                    })
                    .collect();
                result.push(message);
            }
            Message::ToolResult(tool_result) => {
                answered.insert(tool_result.tool_call_id.clone());
                result.push(message);
            }
            Message::System(_) if !pending_calls.is_empty() => held_system.push(message),
            Message::User(_) => {
                close_pending(&mut result, &mut pending_calls, &mut answered, &mut held_system);
                result.push(message);
            }
            _ => result.push(message),
        }
    }
    close_pending(&mut result, &mut pending_calls, &mut answered, &mut held_system);
    result
}
