// The OpenAI Responses format, shared by the Codex backend and any openai-responses provider:
// conversation to request input, tools to declarations, and the event stream back to a message.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/api/openai-responses-shared.ts,
// with the JSON repair of packages/ai/src/utils/json-parse.ts and the strict-schema transform of
// packages/ai/src/api/constrained-sampling.ts.
//
// Left out: grammar ("custom") tools and the tool_search fallback, which only pi's codemode and
// models piccosta doesn't run use; per-turn effort via configuration_update items, which no
// catalog model asks for; and service-tier pricing, which only pi-subagents' fast mode set.
//
// Every request item is built with its keys in pi's order. Responses caches by prefix, so the same
// history must produce the same bytes turn after turn.

use super::transcript::{self, ToolSupport};
use super::transform;
use super::StreamEvent;
use crate::message::{AssistantMessage, Content, Cost, CostState, Message, StopReason, SystemMessage, ToolDefinition, Usage};
use crate::model::Model;
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};

// Longest item id the Responses API accepts.
const MAX_ITEM_ID_LENGTH: usize = 64;
// Keywords OpenAI's strict mode rejects in a schema (constrained-sampling.ts:14-32).
const UNSUPPORTED_STRICT_SCHEMA_KEYS: [&str; 16] = [
    "$ref", "$defs", "definitions", "allOf", "oneOf", "patternProperties", "dependentSchemas", "dependencies",
    "unevaluatedProperties", "propertyNames", "contains", "prefixItems", "not", "if", "then", "else",
];

// pi's shortHash (utils/hash.ts): two 32-bit multiply-xor hashes over the UTF-16 code units,
// written in base 36. Used to shorten ids, so it must give pi's exact output: an id it produced
// for a past turn has to come out the same when that turn is sent again.
pub fn short_hash(text: &str) -> String {
    let mut h1: u32 = 0xdeadbeef;
    let mut h2: u32 = 0x41c6ce57;
    for unit in text.encode_utf16() {
        let unit = unit as u32;
        h1 = (h1 ^ unit).wrapping_mul(2654435761);
        h2 = (h2 ^ unit).wrapping_mul(1597334677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2246822507) ^ (h2 ^ (h2 >> 13)).wrapping_mul(3266489909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2246822507) ^ (h1 ^ (h1 >> 13)).wrapping_mul(3266489909);
    format!("{}{}", base36(h2), base36(h1))
}

fn base36(mut value: u32) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if value == 0 {
        return "0".to_string();
    }
    let mut digits = Vec::new();
    while value > 0 {
        digits.push(DIGITS[(value % 36) as usize]);
        value /= 36;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap_or_default()
}

// One part of a tool-call id made safe: anything outside [a-zA-Z0-9_-] becomes "_" (one per UTF-16
// unit, as pi's regex does), cut to 64, trailing underscores dropped.
fn normalize_id_part(part: &str) -> String {
    let mut sanitized = String::with_capacity(part.len());
    for character in part.chars() {
        if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
            sanitized.push(character);
        } else {
            for _ in 0..character.len_utf16() {
                sanitized.push('_');
            }
        }
    }
    sanitized.truncate(MAX_ITEM_ID_LENGTH);
    sanitized.trim_end_matches('_').to_string()
}

// The rewrite for a tool-call id another model made, so this API accepts it (shared:200-223).
// Responses ids are "<call id>|<item id>"; the item id has to start with fc_, and one from another
// provider is replaced by a hash of itself.
fn normalize_tool_call_id(id: &str, model: &Model, source: &AssistantMessage, allowed_providers: &[&str]) -> String {
    if !allowed_providers.contains(&model.provider.as_str()) || !id.contains('|') {
        return normalize_id_part(id);
    }
    let mut parts = id.splitn(2, '|');
    let call_id = normalize_id_part(parts.next().unwrap_or_default());
    let item_id = parts.next().unwrap_or_default();
    let foreign = source.provider != model.provider || source.api != model.api;
    let mut normalized_item = if foreign {
        let mut hashed = format!("fc_{}", short_hash(item_id));
        hashed.truncate(MAX_ITEM_ID_LENGTH);
        hashed
    } else {
        normalize_id_part(item_id)
    };
    if !normalized_item.starts_with("fc_") {
        normalized_item = normalize_id_part(&format!("fc_{normalized_item}"));
    }
    format!("{call_id}|{normalized_item}")
}

// A text block's Responses signature: {"v":1,"id","phase"}, or a bare id from older sessions.
fn parse_text_signature(signature: Option<&str>) -> Option<(String, Option<String>)> {
    let signature = signature?;
    if signature.starts_with('{')
        && let Ok(parsed) = serde_json::from_str::<Value>(signature)
            && parsed.get("v").and_then(Value::as_i64) == Some(1)
                && let Some(id) = parsed.get("id").and_then(Value::as_str) {
                    let phase = parsed
                        .get("phase")
                        .and_then(Value::as_str)
                        .filter(|phase| *phase == "commentary" || *phase == "final_answer")
                        .map(str::to_string);
                    return Some((id.to_string(), phase));
                }
    Some((signature.to_string(), None))
}

fn encode_text_signature(id: &str, phase: Option<&str>) -> String {
    let mut payload = Map::new();
    payload.insert("v".into(), 1.into());
    payload.insert("id".into(), id.into());
    if let Some(phase) = phase {
        payload.insert("phase".into(), phase.into());
    }
    Value::Object(payload).to_string()
}

// A tool result's output: plain text, or text and images when the model takes images
// (shared:82-109).
fn tool_result_output(model: &Model, content: &[Content]) -> Value {
    let text = content
        .iter()
        .filter_map(|block| match block {
            Content::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images: Vec<(&str, &str)> = content
        .iter()
        .filter_map(|block| match block {
            Content::Image { data, mime_type } => Some((data.as_str(), mime_type.as_str())),
            _ => None,
        })
        .collect();
    let has_text = !text.trim().is_empty();
    if images.is_empty() || !model.input.iter().any(|kind| kind == "image") {
        let output = if has_text {
            text
        } else if !images.is_empty() {
            "(see attached image)".to_string()
        } else {
            "(no tool output)".to_string()
        };
        return Value::String(output);
    }
    let mut parts = Vec::new();
    if has_text {
        parts.push(json!({"type": "input_text", "text": text}));
    }
    for (data, mime_type) in images {
        parts.push(json!({"type": "input_image", "detail": "auto", "image_url": format!("data:{mime_type};base64,{data}")}));
    }
    Value::Array(parts)
}

// How a model takes tool changes after the first request (shared:166-173).
pub fn tool_support(model: &Model) -> ToolSupport {
    if model.compat_flag("supportsMidConvoSystemMessages") && model.compat_flag("supportsAdditionalTools") {
        ToolSupport::Changes
    } else {
        ToolSupport::None
    }
}

// The guidance before every additional_tools item after the first, and the notice for removed
// tools (shared:181-189). Model-visible; pi verified that smaller models need the restated list.
fn incremental_tools_notice(unavailable: &BTreeSet<String>) -> String {
    let notice = "This is an incremental tools update. Previously declared tools remain available unless explicitly marked unavailable. A tool declared again here is available again, and its latest definition replaces the earlier one.";
    if unavailable.is_empty() {
        notice.to_string()
    } else {
        format!("{notice}\nStill unavailable: {}.", unavailable.iter().cloned().collect::<Vec<_>>().join(", "))
    }
}

fn tool_removal_notice(names: &[String]) -> String {
    let lines: Vec<String> = names.iter().map(|name| format!("- {name}")).collect();
    format!("The following tools are no longer available. Do not call them:\n{}", lines.join("\n"))
}

// Options for turning a conversation into Responses input.
pub struct ConvertOptions<'a> {
    // Whether the first system message goes in `input` (openai-responses) or travels as the
    // request's `instructions` (Codex).
    pub include_system_prompt: bool,
    // Providers whose "call|item" tool-call ids are kept in two parts.
    pub tool_call_providers: &'a [&'a str],
    // Default `strict` for tools without a constrained-sampling setting: Codex sends null, the
    // Responses API false.
    pub default_strict: Option<bool>,
    // Whether the model takes strict tool schemas; `strict` is only sent when it does.
    pub supports_strict: bool,
}

// The conversation as Responses `input` items (shared:191-432).
pub fn convert_messages(model: &Model, messages: &[Message], options: &ConvertOptions) -> Vec<Value> {
    let supports_mid_conversation_system = model.compat_flag("supportsMidConvoSystemMessages");
    let resolved = transcript::resolve_transcript(messages, supports_mid_conversation_system);
    let normalize = |id: &str, source: &AssistantMessage| normalize_tool_call_id(id, model, source, options.tool_call_providers);
    let transformed = transform::transform_messages(&resolved, model, &normalize);
    let support = tool_support(model);
    let (_, folded_system_messages) = transcript::request_tools(&resolved, support);
    // Reasoning models take instructions as `developer`, unless the provider says it has no such role.
    let developer_allowed = model.compat.as_ref().and_then(|compat| compat.get("supportsDeveloperRole")).and_then(Value::as_bool) != Some(false);
    let instruction_role = if model.reasoning && developer_allowed { "developer" } else { "system" };

    let mut items: Vec<Value> = Vec::new();
    let mut declared_additional_tools = false;
    let mut unavailable_tools: BTreeSet<String> = BTreeSet::new();
    let mut message_index = 0usize;
    let mut system_index = 0usize;

    for (source_index, message) in transformed.iter().enumerate() {
        let is_leading_system = source_index == 0 && matches!(message, Message::System(_));
        match message {
            Message::System(system) => {
                let anchors_tool_changes = system_index >= folded_system_messages;
                system_index += 1;
                if anchors_tool_changes {
                    append_tool_additions(
                        &mut items,
                        system,
                        support,
                        instruction_role,
                        &mut declared_additional_tools,
                        &mut unavailable_tools,
                        options,
                    );
                }
                if !is_leading_system || options.include_system_prompt {
                    let mut parts = vec![if is_leading_system {
                        transcript::system_message_text(system)
                    } else {
                        transcript::render_system_message_update(system)
                    }];
                    if anchors_tool_changes {
                        // A redefinition is a removal plus an addition; only true removals are announced.
                        let added: BTreeSet<&str> =
                            system.tools_added.iter().flatten().map(|tool| tool.name.as_str()).collect();
                        let removed: Vec<String> = system
                            .tools_removed
                            .iter()
                            .flatten()
                            .map(|tool| tool.name.clone())
                            .filter(|name| !added.contains(name.as_str()))
                            .collect();
                        if !removed.is_empty() {
                            parts.push(tool_removal_notice(&removed));
                        }
                        unavailable_tools.extend(removed);
                    }
                    parts.retain(|part| !part.is_empty());
                    let text = parts.join("\n\n");
                    if !text.is_empty() {
                        items.push(json!({"role": instruction_role, "content": text}));
                    }
                }
            }
            Message::User(user) => match &user.content {
                crate::message::TextOrBlocks::Text(text) => {
                    items.push(json!({"role": "user", "content": [{"type": "input_text", "text": text}]}));
                }
                crate::message::TextOrBlocks::Blocks(blocks) => {
                    let content: Vec<Value> = blocks
                        .iter()
                        .filter_map(|block| match block {
                            Content::Text { text, .. } => Some(json!({"type": "input_text", "text": text})),
                            Content::Image { data, mime_type } => Some(json!({
                                "type": "input_image", "detail": "auto", "image_url": format!("data:{mime_type};base64,{data}")
                            })),
                            _ => None,
                        })
                        .collect();
                    if content.is_empty() {
                        continue;
                    }
                    items.push(json!({"role": "user", "content": content}));
                }
            },
            Message::Assistant(assistant) => {
                let same_provider_and_api = assistant.provider == model.provider && assistant.api == model.api;
                let same_model = same_provider_and_api && assistant.model == model.id;
                let different_model = same_provider_and_api && assistant.model != model.id;
                let mut output = Vec::new();
                let mut text_block_index = 0usize;
                for block in &assistant.content {
                    match block {
                        Content::Thinking { thinking_signature: Some(signature), .. } => {
                            // The stored reasoning item goes back as it came, encrypted content and all.
                            if let Ok(item) = serde_json::from_str::<Value>(signature) {
                                output.push(item);
                            }
                        }
                        Content::Text { text, text_signature } => {
                            let parsed = parse_text_signature(text_signature.as_deref());
                            let fallback = if text_block_index == 0 {
                                format!("msg_pi_{message_index}")
                            } else {
                                format!("msg_pi_{message_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            let id = match &parsed {
                                None => fallback,
                                Some((id, _)) if id.encode_utf16().count() > MAX_ITEM_ID_LENGTH => format!("msg_{}", short_hash(id)),
                                Some((id, _)) => id.clone(),
                            };
                            let mut item = Map::new();
                            item.insert("type".into(), "message".into());
                            item.insert("role".into(), "assistant".into());
                            item.insert("content".into(), json!([{"type": "output_text", "text": text, "annotations": []}]));
                            item.insert("status".into(), "completed".into());
                            item.insert("id".into(), id.into());
                            if let Some((_, Some(phase))) = &parsed {
                                item.insert("phase".into(), phase.clone().into());
                            }
                            output.push(Value::Object(item));
                        }
                        Content::ToolCall { id, name, arguments, namespace, .. } => {
                            let mut parts = id.splitn(2, '|');
                            let call_id = parts.next().unwrap_or_default();
                            // An item id from a different model, or one not shaped like a function
                            // call id, is left out so OpenAI doesn't pair it with reasoning it never saw.
                            let item_id = parts.next().filter(|item_id| !different_model && item_id.starts_with("fc_"));
                            let mut item = Map::new();
                            item.insert("type".into(), "function_call".into());
                            if let Some(item_id) = item_id {
                                item.insert("id".into(), item_id.into());
                            }
                            item.insert("call_id".into(), call_id.into());
                            item.insert("name".into(), name.clone().into());
                            item.insert("arguments".into(), Value::Object(arguments.clone()).to_string().into());
                            if let (true, Some(namespace)) = (same_model, namespace) {
                                item.insert("namespace".into(), namespace.clone().into());
                            }
                            output.push(Value::Object(item));
                        }
                        _ => {}
                    }
                }
                if output.is_empty() {
                    continue;
                }
                items.extend(output);
            }
            Message::ToolResult(result) => {
                let call_id = result.tool_call_id.split('|').next().unwrap_or_default();
                items.push(json!({"type": "function_call_output", "call_id": call_id, "output": tool_result_output(model, &result.content)}));
            }
        }
        if !is_leading_system {
            message_index += 1;
        }
    }
    items
}

// Declare a system message's added tools in place (shared:231-246): an additional_tools item, with
// the incremental-update notice ahead of every one after the first.
fn append_tool_additions(
    items: &mut Vec<Value>,
    system: &SystemMessage,
    support: ToolSupport,
    instruction_role: &str,
    declared_additional_tools: &mut bool,
    unavailable_tools: &mut BTreeSet<String>,
    options: &ConvertOptions,
) {
    let tools = system.tools_added.as_deref().unwrap_or_default();
    for tool in tools {
        unavailable_tools.remove(&tool.name);
    }
    if tools.is_empty() || support != ToolSupport::Changes {
        return;
    }
    if *declared_additional_tools {
        items.push(json!({"role": instruction_role, "content": incremental_tools_notice(unavailable_tools)}));
    }
    *declared_additional_tools = true;
    items.push(json!({"type": "additional_tools", "role": "developer", "tools": convert_tools(tools, options.default_strict, options.supports_strict)}));
}

// Tools as Responses function declarations (shared:438-475). A tool that asks for JSON-schema
// constrained sampling gets strict mode when the model supports it and its schema fits; the rest
// get `default_strict`. `strict` is sent only to models with strict mode.
pub fn convert_tools(tools: &[ToolDefinition], default_strict: Option<bool>, supports_strict: bool) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let wants_strict = tool.constrained_sampling.as_ref().and_then(|config| config.get("type")).and_then(Value::as_str) == Some("json_schema");
            let strict_schema = (wants_strict && supports_strict).then(|| strict_json_schema(&tool.parameters)).flatten();
            let strict = if strict_schema.is_some() { Some(true) } else { default_strict };
            let mut declaration = Map::new();
            declaration.insert("type".into(), "function".into());
            declaration.insert("name".into(), tool.name.clone().into());
            declaration.insert("description".into(), tool.description.clone().into());
            declaration.insert("parameters".into(), strict_schema.unwrap_or_else(|| tool.parameters.clone()));
            if supports_strict {
                declaration.insert("strict".into(), strict.map(Value::Bool).unwrap_or(Value::Null));
            }
            Value::Object(declaration)
        })
        .collect()
}

// A schema rewritten to OpenAI's strict subset (constrained-sampling.ts:56-139): every property
// required, optional ones made nullable, no other properties allowed. None when the schema uses
// something strict mode can't express; the tool is then sent without strict.
pub fn strict_json_schema(schema: &Value) -> Option<Value> {
    let mut cloned = schema.clone();
    make_strict(&mut cloned)?;
    (cloned.get("type").and_then(Value::as_str) == Some("object")).then_some(cloned)
}

fn is_structured(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else { return false };
    let types: Vec<&str> = match object.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    types.contains(&"object") || types.contains(&"array") || object.contains_key("properties") || object.contains_key("items")
}

fn allows_null(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else { return false };
    let type_allows = match object.get("type") {
        Some(Value::String(kind)) => kind == "null",
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind == "null"),
        _ => false,
    };
    type_allows
        || object.get("const") == Some(&Value::Null)
        || object.get("enum").and_then(Value::as_array).is_some_and(|values| values.contains(&Value::Null))
        || object.get("anyOf").and_then(Value::as_array).is_some_and(|variants| variants.iter().any(allows_null))
}

fn make_strict(schema: &mut Value) -> Option<()> {
    let object = schema.as_object_mut()?;
    if UNSUPPORTED_STRICT_SCHEMA_KEYS.iter().any(|key| object.contains_key(*key)) {
        return None;
    }
    if let Some(any_of) = object.get_mut("anyOf") {
        let variants = any_of.as_array_mut().filter(|variants| !variants.is_empty())?;
        for variant in variants {
            if is_structured(variant) {
                return None;
            }
            make_strict(variant)?;
        }
    }
    if let Some(items) = object.get_mut("items") {
        if items.is_array() {
            return None;
        }
        make_strict(items)?;
    }
    let is_object = object.get("type").and_then(Value::as_str) == Some("object");
    if object.contains_key("properties") && !is_object {
        return None;
    }
    if !is_object {
        return Some(());
    }
    if object.get("additionalProperties").is_some_and(|value| value != &Value::Bool(false)) {
        return None;
    }
    let required: Vec<String> = match object.get("required") {
        None => Vec::new(),
        Some(Value::Array(names)) => names.iter().map(|name| name.as_str().map(str::to_string)).collect::<Option<_>>()?,
        Some(_) => return None,
    };
    let properties = match object.get_mut("properties") {
        None => &mut Map::new(),
        Some(Value::Object(properties)) => properties,
        Some(_) => return None,
    };
    if required.iter().any(|name| !properties.contains_key(name)) {
        return None;
    }
    let names: Vec<String> = properties.keys().cloned().collect();
    for (name, property) in properties.iter_mut() {
        make_strict(property)?;
        if !required.contains(name) && !allows_null(property) {
            *property = json!({"anyOf": [property.clone(), {"type": "null"}]});
        }
    }
    object.insert("required".into(), names.into_iter().map(Value::String).collect());
    object.insert("additionalProperties".into(), Value::Bool(false));
    Some(())
}

// Repair the two mistakes models make in JSON strings (json-parse.ts:32-83): raw control characters,
// which get escaped, and backslashes before characters that aren't escapes, which get doubled.
pub fn repair_json(json: &str) -> String {
    const VALID_ESCAPES: [char; 8] = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't'];
    let characters: Vec<char> = json.chars().collect();
    let mut repaired = String::with_capacity(json.len());
    let mut in_string = false;
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if !in_string {
            repaired.push(character);
            in_string = character == '"';
            index += 1;
            continue;
        }
        if character == '"' {
            repaired.push(character);
            in_string = false;
            index += 1;
            continue;
        }
        if character == '\\' {
            match characters.get(index + 1) {
                None => repaired.push_str("\\\\"),
                Some('u') if characters.get(index + 2..index + 6).is_some_and(|digits| digits.iter().all(char::is_ascii_hexdigit)) => {
                    repaired.push_str("\\u");
                    repaired.extend(&characters[index + 2..index + 6]);
                    index += 6;
                    continue;
                }
                Some(next) if VALID_ESCAPES.contains(next) => {
                    repaired.push('\\');
                    repaired.push(*next);
                    index += 2;
                    continue;
                }
                Some(_) => repaired.push_str("\\\\"),
            }
            index += 1;
            continue;
        }
        match character {
            '\u{8}' => repaired.push_str("\\b"),
            '\u{c}' => repaired.push_str("\\f"),
            '\n' => repaired.push_str("\\n"),
            '\r' => repaired.push_str("\\r"),
            '\t' => repaired.push_str("\\t"),
            control if (control as u32) < 0x20 => repaired.push_str(&format!("\\u{:04x}", control as u32)),
            other => repaired.push(other),
        }
        index += 1;
    }
    repaired
}

// A finished tool call's arguments (json-parse.ts:104-120). Input: the text the model sent.
// Output: the object, or an empty object and an error naming the problem with the text, which the
// tool layer reports back to the model.
pub fn parse_tool_call_arguments(json: &str) -> (Map<String, Value>, Option<String>) {
    if json.trim().is_empty() {
        return (Map::new(), None);
    }
    let parsed = serde_json::from_str::<Value>(json).or_else(|error| {
        let repaired = repair_json(json);
        if repaired != json { serde_json::from_str::<Value>(&repaired) } else { Err(error) }
    });
    match parsed {
        Ok(Value::Object(object)) => (object, None),
        Ok(_) => (Map::new(), Some(format!("Tool arguments must be a JSON object\n\nReceived arguments:\n{json}"))),
        Err(error) => (Map::new(), Some(format!("{error}\n\nReceived arguments:\n{json}"))),
    }
}

// Whether a provider's responses are billed by a plan's quota now, not by the token: then they cost
// no money, whatever the list's prices say. So are the accounts piccosta knows (auth::ACCOUNTS),
// Claude only while Claude Code is logged in to a plan (claude::billed_by_token). Each response
// records the answer as it was made (Cost::plan); totals go by that, never by today's login.
pub fn plan_billed(provider: &str) -> bool {
    match provider {
        crate::provider::claude::PROVIDER => !crate::provider::claude::billed_by_token(),
        provider => crate::auth::account(provider).is_some(),
    }
}

// A cost as people read it: the dollars known, to `places` decimals, and "unknown" for responses
// billed by the token at prices nobody gave (models.json can give them).
pub fn dollars(known: f64, unknown: bool, places: usize) -> String {
    match (unknown, known > 0.0) {
        (false, _) => format!("${known:.places$}"),
        (true, false) => "unknown".into(),
        (true, true) => format!("${known:.places$} + unknown"),
    }
}

// What responses cost, added up in the order they were made from what each recorded then. A plan's
// add nothing: by their record, or for one saved before responses kept it, by the provider (the
// accounts piccosta knows and Claude were plans then). A cost nobody knew, or a pending one no later
// turn paid for, leaves the total partly unknown. One per session: a turn settles its own query's.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tally {
    pub total: f64,
    unknown: bool,
    pending: u32,
}

impl Tally {
    pub fn add(&mut self, usage: &Usage, provider: &str) {
        let old_plan = || provider == crate::provider::claude::PROVIDER || crate::auth::account(provider).is_some();
        if usage.cost.plan.unwrap_or_else(old_plan) {
            return;
        }
        self.total += usage.cost.total;
        self.unknown |= usage.cost.state == CostState::Unknown;
        self.pending = (self.pending + u32::from(usage.cost.state == CostState::Pending)).saturating_sub(usage.cost.settles);
    }

    // Another session's tally, its pending turns now for good unknown.
    pub fn merge(&mut self, other: Tally) {
        self.total += other.total;
        self.unknown |= other.unknown();
    }

    pub fn unknown(&self) -> bool {
        self.unknown || self.pending > 0
    }

    pub fn dollars(&self, places: usize) -> String {
        dollars(self.total, self.unknown(), places)
    }
}

// Anthropic prices a cache write kept for an hour at twice the base input rate (platform.claude.com's
// prompt caching docs, read 2026-11-03): what an hour's write costs when the model's prices give no
// cacheWrite1h.
const HOUR_WRITE_INPUT_MULTIPLIER: f64 = 2.0;

// Set a response's cost from its usage and the model's prices, with the tier for its input size
// (models.ts:1236-1256). A plan's responses cost nothing; one billed by the token at prices nobody
// gave has an unknown cost, not a zero one.
// A Responses API `usage` object as piccosta's Usage, its cost not yet reckoned. OpenAI counts
// cached and cache-write tokens inside input_tokens, so both come off.
pub fn usage_from(usage: &Value) -> Usage {
    let number = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);
    let cached = number("/input_tokens_details/cached_tokens");
    let cache_write = number("/input_tokens_details/cache_write_tokens");
    Usage {
        input: number("/input_tokens").saturating_sub(cached + cache_write),
        output: number("/output_tokens"),
        cache_read: cached,
        cache_write,
        cache_write_1h: None,
        reasoning: Some(number("/output_tokens_details/reasoning_tokens")),
        total_tokens: number("/total_tokens"),
        cost: Default::default(),
    }
}

pub fn calculate_cost(model: &Model, usage: &mut Usage) {
    let plan = plan_billed(&model.provider);
    let state = if plan || model.cost.supplied { CostState::Known } else { CostState::Unknown };
    usage.cost = Cost { plan: Some(plan), state, ..Default::default() };
    if plan || !model.cost.supplied {
        return;
    }
    let input_tokens = usage.input + usage.cache_read + usage.cache_write;
    let mut rates = (model.cost.input, model.cost.output, model.cost.cache_read, model.cost.cache_write);
    let mut matched_threshold: i64 = -1;
    for tier in &model.cost.tiers {
        if input_tokens > tier.input_tokens_above && tier.input_tokens_above as i64 > matched_threshold {
            rates = (tier.input, tier.output, tier.cache_read, tier.cache_write);
            matched_threshold = tier.input_tokens_above as i64;
        }
    }
    let (input_rate, output_rate, cache_read_rate, cache_write_rate) = rates;
    const PER_MILLION: f64 = 1_000_000.0;
    let long_write = usage.cache_write_1h.unwrap_or(0) as f64;
    let short_write = usage.cache_write as f64 - long_write;
    usage.cost.input = input_rate / PER_MILLION * usage.input as f64;
    usage.cost.output = output_rate / PER_MILLION * usage.output as f64;
    usage.cost.cache_read = cache_read_rate / PER_MILLION * usage.cache_read as f64;
    let long_write_rate = model.cost.cache_write_1h.unwrap_or(input_rate * HOUR_WRITE_INPUT_MULTIPLIER);
    usage.cost.cache_write = (cache_write_rate * short_write + long_write_rate * long_write) / PER_MILLION;
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
}

// What a streaming output slot is building.
enum Slot {
    Thinking(usize),
    Text(usize),
    ToolCall { index: usize, partial_json: String },
}

// Turns Responses stream events into an assistant message, sending StreamEvents as it goes
// (shared:511-855). Feed it each event with `handle`; call `finish` at the end of the stream.
pub struct StreamState<'a> {
    model: &'a Model,
    pub output: AssistantMessage,
    slots: HashMap<u64, Slot>,
    // Reasoning blocks by item id, for backfilling encrypted content from the final response.
    reasoning_by_id: HashMap<String, usize>,
    saw_terminal: bool,
}

impl<'a> StreamState<'a> {
    pub fn new(model: &'a Model, output: AssistantMessage) -> StreamState<'a> {
        StreamState { model, output, slots: HashMap::new(), reasoning_by_id: HashMap::new(), saw_terminal: false }
    }

    // Start a slot for a new output item, if it's a kind we build.
    fn create_slot(&mut self, output_index: u64, item: &Value, send: &mut dyn FnMut(StreamEvent)) {
        let index = self.output.content.len();
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => {
                self.output.content.push(Content::Thinking { thinking: String::new(), thinking_signature: None, redacted: None });
                self.slots.insert(output_index, Slot::Thinking(index));
                send(StreamEvent::ThinkingStart { index });
            }
            Some("message") => {
                if item.get("phase").and_then(Value::as_str) == Some("final_answer") {
                    self.output.stop_reason = StopReason::Stop;
                }
                self.output.content.push(Content::text(""));
                self.slots.insert(output_index, Slot::Text(index));
                send(StreamEvent::TextStart { index });
            }
            Some("function_call") => {
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or_default();
                let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default();
                self.output.content.push(Content::ToolCall {
                    id: format!("{call_id}|{item_id}"),
                    name: item.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                    arguments: Map::new(),
                    namespace: item.get("namespace").and_then(Value::as_str).map(str::to_string),
                    arguments_error: None,
                    thought_signature: None,
                });
                let partial_json = item.get("arguments").and_then(Value::as_str).unwrap_or_default().to_string();
                self.slots.insert(output_index, Slot::ToolCall { index, partial_json });
                send(StreamEvent::ToolCallStart {
                    index,
                    id: format!("{call_id}|{item_id}"),
                    name: item.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                });
            }
            _ => {}
        }
    }

    // Handle one stream event. Output: an error message when the event ends the response with an
    // error (the caller then finishes the message as failed).
    pub fn handle(&mut self, event: &Value, send: &mut dyn FnMut(StreamEvent)) -> Result<(), String> {
        let kind = event.get("type").and_then(Value::as_str).unwrap_or_default();
        let output_index = event.get("output_index").and_then(Value::as_u64).unwrap_or(u64::MAX);
        let delta = event.get("delta").and_then(Value::as_str).unwrap_or_default();
        match kind {
            "response.created" => {
                if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                    self.output.response_id = Some(id.to_string());
                }
            }
            "response.output_item.added" => {
                if let Some(item) = event.get("item") {
                    self.create_slot(output_index, item, send);
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" | "response.reasoning_summary_part.done" => {
                let piece = if kind == "response.reasoning_summary_part.done" { "\n\n" } else { delta };
                if let Some(Slot::Thinking(index)) = self.slots.get(&output_index) {
                    let index = *index;
                    if let Content::Thinking { thinking, .. } = &mut self.output.content[index] {
                        thinking.push_str(piece);
                    }
                    send(StreamEvent::ThinkingDelta { index, delta: piece.to_string() });
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                if let Some(Slot::Text(index)) = self.slots.get(&output_index) {
                    let index = *index;
                    if let Content::Text { text, .. } = &mut self.output.content[index] {
                        text.push_str(delta);
                    }
                    send(StreamEvent::TextDelta { index, delta: delta.to_string() });
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(Slot::ToolCall { index, partial_json }) = self.slots.get_mut(&output_index) {
                    partial_json.push_str(delta);
                    send(StreamEvent::ToolCallDelta { index: *index, delta: delta.to_string() });
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(Slot::ToolCall { index, partial_json }) = self.slots.get_mut(&output_index) {
                    let arguments = event.get("arguments").and_then(Value::as_str).unwrap_or_default();
                    if let Some(rest) = arguments.strip_prefix(partial_json.as_str())
                        && !rest.is_empty() {
                            send(StreamEvent::ToolCallDelta { index: *index, delta: rest.to_string() });
                        }
                    *partial_json = arguments.to_string();
                }
            }
            "response.output_item.done" => {
                let Some(item) = event.get("item") else { return Ok(()) };
                if item.get("type").and_then(Value::as_str) == Some("message") && item.get("phase").and_then(Value::as_str) == Some("final_answer") {
                    self.output.stop_reason = StopReason::Stop;
                }
                if !self.slots.contains_key(&output_index) {
                    self.create_slot(output_index, item, send);
                }
                self.finish_item(output_index, item, send);
            }
            "response.completed" | "response.incomplete" => {
                if let Some(response) = event.get("response") {
                    self.finalize_response(response);
                }
            }
            "error" => {
                let code = event.get("code").map(|code| code.as_str().map(str::to_string).unwrap_or_else(|| code.to_string())).unwrap_or_else(|| "undefined".into());
                let message = event.get("message").and_then(Value::as_str).unwrap_or("undefined");
                return Err(format!("Error Code {code}: {message}"));
            }
            "response.failed" => {
                self.saw_terminal = true;
                self.output.raw_stop_reason = event.pointer("/response/status").and_then(Value::as_str).map(str::to_string);
                let error = event.pointer("/response/error");
                let reason = event.pointer("/response/incomplete_details/reason").and_then(Value::as_str);
                return Err(match (error, reason) {
                    (Some(error), _) if !error.is_null() => format!(
                        "{}: {}",
                        error.get("code").and_then(Value::as_str).filter(|code| !code.is_empty()).unwrap_or("unknown"),
                        error.get("message").and_then(Value::as_str).filter(|message| !message.is_empty()).unwrap_or("no message"),
                    ),
                    (_, Some(reason)) => format!("incomplete: {reason}"),
                    _ => "Unknown error (no error details in response)".to_string(),
                });
            }
            _ => {}
        }
        Ok(())
    }

    // Finish the slot an output item built, with the item's final content (shared:761-820).
    fn finish_item(&mut self, output_index: u64, item: &Value, send: &mut dyn FnMut(StreamEvent)) {
        let Some(slot) = self.slots.remove(&output_index) else { return };
        match (slot, item.get("type").and_then(Value::as_str)) {
            (Slot::Thinking(index), Some("reasoning")) => {
                let join = |key: &str| -> String {
                    item.get(key)
                        .and_then(Value::as_array)
                        .map(|parts| parts.iter().filter_map(|part| part.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n\n"))
                        .unwrap_or_default()
                };
                let summary = join("summary");
                let content = join("content");
                if let Content::Thinking { thinking, thinking_signature, .. } = &mut self.output.content[index] {
                    if !summary.is_empty() {
                        *thinking = summary;
                    } else if !content.is_empty() {
                        *thinking = content;
                    }
                    *thinking_signature = Some(item.to_string());
                }
                if let Some(id) = item.get("id").and_then(Value::as_str) {
                    self.reasoning_by_id.insert(id.to_string(), index);
                }
                let content = match &self.output.content[index] {
                    Content::Thinking { thinking, .. } => thinking.clone(),
                    _ => String::new(),
                };
                send(StreamEvent::ThinkingEnd { index, content });
            }
            (Slot::Text(index), Some("message")) => {
                let text: String = item
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                                Some("output_text") => part.get("text").and_then(Value::as_str),
                                _ => part.get("refusal").and_then(Value::as_str),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
                let phase = item.get("phase").and_then(Value::as_str);
                self.output.content[index] = Content::Text { text: text.clone(), text_signature: Some(encode_text_signature(id, phase)) };
                send(StreamEvent::TextEnd { index, content: text });
            }
            (Slot::ToolCall { index, partial_json }, Some("function_call")) => {
                let text = item.get("arguments").and_then(Value::as_str).filter(|text| !text.is_empty()).unwrap_or(&partial_json);
                let (parsed, error) = parse_tool_call_arguments(text);
                if let Content::ToolCall { arguments, arguments_error, namespace, .. } = &mut self.output.content[index] {
                    *arguments = parsed;
                    *arguments_error = error;
                    if let Some(item_namespace) = item.get("namespace").and_then(Value::as_str) {
                        *namespace = Some(item_namespace.to_string());
                    }
                }
                send(StreamEvent::ToolCallEnd { index, tool_call: self.output.content[index].clone() });
            }
            (slot, _) => {
                // The item changed type under its slot; keep the slot so `finish` reports it.
                self.slots.insert(output_index, slot);
            }
        }
    }

    // Usage, cost and stop reason from the final response (shared:630-675).
    fn finalize_response(&mut self, response: &Value) {
        self.saw_terminal = true;
        // Some servers send encrypted reasoning only in the final response; copy it into the
        // stored reasoning items so the next turn can replay them.
        for item in response.get("output").and_then(Value::as_array).into_iter().flatten() {
            let (Some("reasoning"), Some(encrypted), Some(id)) = (
                item.get("type").and_then(Value::as_str),
                item.get("encrypted_content").filter(|value| !value.is_null()),
                item.get("id").and_then(Value::as_str),
            ) else {
                continue;
            };
            let Some(&index) = self.reasoning_by_id.get(id) else { continue };
            if let Content::Thinking { thinking_signature: Some(signature), .. } = &mut self.output.content[index]
                && let Ok(Value::Object(mut stored)) = serde_json::from_str::<Value>(signature)
                    && stored.get("encrypted_content").is_none_or(Value::is_null) {
                        stored.insert("encrypted_content".into(), encrypted.clone());
                        *signature = Value::Object(stored).to_string();
                    }
        }
        if let Some(id) = response.get("id").and_then(Value::as_str) {
            self.output.response_id = Some(id.to_string());
        }
        if let Some(usage) = response.get("usage").filter(|usage| !usage.is_null()) {
            self.output.usage = usage_from(usage);
        }
        calculate_cost(self.model, &mut self.output.usage);
        let status = response.get("status").and_then(Value::as_str);
        let reason = response.pointer("/incomplete_details/reason").and_then(Value::as_str);
        self.output.raw_stop_reason = match (status, reason) {
            (Some(status), Some(reason)) => Some(format!("{status}.{reason}")),
            (Some(status), None) => Some(status.to_string()),
            _ => None,
        };
        let (stop_reason, error_message) = match (status, reason) {
            (None | Some("completed" | "in_progress" | "queued"), _) => (StopReason::Stop, None),
            (Some("incomplete"), Some("max_output_tokens")) => (StopReason::Length, None),
            (Some("incomplete"), Some(reason)) => (StopReason::Error, Some(format!("Response incomplete: {reason}"))),
            (Some("incomplete"), None) => (StopReason::Error, Some("Response incomplete without a provider reason".to_string())),
            (Some(_), _) => (StopReason::Error, None),
        };
        self.output.stop_reason = stop_reason;
        self.output.error_message = error_message;
        if stop_reason == StopReason::Stop && self.output.content.iter().any(|block| matches!(block, Content::ToolCall { .. })) {
            self.output.stop_reason = StopReason::ToolUse;
        }
    }

    // Check the stream ended properly (shared:838-854). Output: an error when no terminal event came,
    // or when a tool call never finished (its arguments may be cut off, so it mustn't run).
    pub fn finish(&self) -> Result<(), String> {
        if !self.saw_terminal {
            return Err("OpenAI Responses stream ended before a terminal response event".to_string());
        }
        if self.output.stop_reason == StopReason::ToolUse {
            for slot in self.slots.values() {
                if let Slot::ToolCall { index, .. } = slot
                    && let Content::ToolCall { name, id, .. } = &self.output.content[*index] {
                        return Err(format!("OpenAI Responses stream completed with an unfinished tool call: {name} ({id})"));
                    }
            }
        }
        Ok(())
    }
}
