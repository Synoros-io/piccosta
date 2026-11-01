// Checking a model's tool arguments against the tool's JSON Schema before the tool runs.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/utils/validation.ts, which nulls
// out optional fields, converts with TypeBox's Value.Convert, then checks with TypeBox. TypeBox is
// a JavaScript library, so what it does here was measured on pi 1.4.0 (2026-10-29) and written
// down rule by rule:
// - string fields take numbers and booleans as their text, and null as "null";
// - number fields take numeric strings (Number() rules: "" is 0, " 7 " is 7, "0x10" is 16),
//   "true"/"false" and booleans as 1/0; integer fields also drop any fraction ("2.5" is 2);
// - boolean fields take "true"/"false", 1/0 and null (false);
// - array fields take a lone value as a one-element array.
// The error text goes back to the model, so it keeps TypeBox's wording and order: a missing-field
// error for an object first, then its fields in schema order, and nothing deeper under a value
// of the wrong type.
//
// Only the schema features piccosta's own tools use are handled: type, properties, required and
// items. Anything else in a schema is ignored, never rejected.

use crate::js;
use serde_json::{Map, Value};

// Input: the tool's name, its parameters schema and the arguments the model sent. Output: the
// arguments after conversion, or the message the model gets back when they don't fit.
pub fn validate(tool_name: &str, schema: &Value, arguments: &Map<String, Value>) -> Result<Map<String, Value>, String> {
    let mut value = Value::Object(arguments.clone());
    strip_optional_nulls(&mut value, schema);
    let value = convert(value, schema);
    let mut errors = Vec::new();
    check(&value, schema, &mut Vec::new(), &mut errors);
    if errors.is_empty() {
        let Value::Object(converted) = value else { unreachable!("an object schema checked an object") };
        return Ok(converted);
    }
    let received = serde_json::to_string_pretty(&Value::Object(arguments.clone())).unwrap_or_default();
    Err(format!("Validation failed for tool \"{tool_name}\":\n{}\n\nReceived arguments:\n{received}", errors.join("\n")))
}

fn types_of(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn required_of(schema: &Value) -> Vec<&str> {
    schema.get("required").and_then(Value::as_array).map(|names| names.iter().filter_map(Value::as_str).collect()).unwrap_or_default()
}

// A strict provider makes every optional field required and nullable, so the model sends null
// for "not given". Those nulls go, so the tool sees the field as absent (validation.ts:283-308).
fn strip_optional_nulls(value: &mut Value, schema: &Value) {
    match value {
        Value::Array(items) => {
            if let Some(item_schema) = schema.get("items").filter(|items| items.is_object()) {
                for item in items {
                    strip_optional_nulls(item, item_schema);
                }
            }
        }
        Value::Object(object) => {
            let Some(properties) = schema.get("properties").and_then(Value::as_object) else { return };
            let required = required_of(schema);
            for (key, property_schema) in properties {
                let Some(field) = object.get_mut(key) else { continue };
                if field.is_null() && !required.contains(&key.as_str()) && !types_of(property_schema).contains(&"null") {
                    object.shift_remove(key);
                } else {
                    strip_optional_nulls(field, property_schema);
                }
            }
        }
        _ => {}
    }
}

// TypeBox's Value.Convert for the schema features above. Values that can't be converted are left
// as they are, and check() reports them.
fn convert(value: Value, schema: &Value) -> Value {
    let kinds = types_of(schema);
    let Some(kind) = kinds.first().copied() else { return value };
    if kinds.len() > 1 {
        return value;
    }
    match kind {
        "string" => match value {
            Value::Number(number) => Value::String(js::number_string(number.as_f64().unwrap_or(0.0))),
            Value::Bool(flag) => Value::String(flag.to_string()),
            Value::Null => Value::String("null".into()),
            other => other,
        },
        "number" | "integer" => {
            let number = match &value {
                Value::String(text) => match text.as_str() {
                    "true" => Some(1.0),
                    "false" => Some(0.0),
                    _ => js::parse_number(text).filter(|number| number.is_finite()),
                },
                Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
                Value::Number(number) if kind == "integer" => number.as_f64(),
                _ => None,
            };
            match number {
                Some(number) => {
                    let number = if kind == "integer" { number.trunc() } else { number };
                    serde_json::Number::from_f64(number).map(Value::Number).unwrap_or(value)
                }
                None => value,
            }
        }
        "boolean" => match &value {
            Value::String(text) if text == "true" => Value::Bool(true),
            Value::String(text) if text == "false" => Value::Bool(false),
            Value::Number(number) if number.as_f64() == Some(1.0) => Value::Bool(true),
            Value::Number(number) if number.as_f64() == Some(0.0) => Value::Bool(false),
            Value::Null => Value::Bool(false),
            _ => value,
        },
        "array" => {
            let items = match value {
                Value::Array(items) => items,
                other => vec![other],
            };
            match schema.get("items").filter(|items| items.is_object()) {
                Some(item_schema) => Value::Array(items.into_iter().map(|item| convert(item, item_schema)).collect()),
                None => Value::Array(items),
            }
        }
        "object" => match value {
            Value::Object(mut object) => {
                if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                    for (key, property_schema) in properties {
                        if let Some(field) = object.get_mut(key) {
                            *field = convert(field.take(), property_schema);
                        }
                    }
                }
                Value::Object(object)
            }
            other => other,
        },
        _ => value,
    }
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "number" => value.as_f64().is_some_and(f64::is_finite),
        "integer" => value.as_f64().is_some_and(|number| number.is_finite() && number.fract() == 0.0),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => true,
    }
}

// TypeBox's error path: JSON Pointer segments joined with dots, "root" for the top level.
fn path_text(path: &[String]) -> String {
    if path.is_empty() { "root".into() } else { path.join(".") }
}

fn check(value: &Value, schema: &Value, path: &mut Vec<String>, errors: &mut Vec<String>) {
    let kinds = types_of(schema);
    if !kinds.is_empty() && !kinds.iter().any(|kind| matches_type(value, kind)) {
        errors.push(format!("  - {}: must be {}", path_text(path), kinds.join(",")));
        return;
    }
    match value {
        Value::Object(object) => {
            let missing: Vec<&str> = required_of(schema).into_iter().filter(|name| !object.contains_key(*name)).collect();
            if let Some(first) = missing.first() {
                path.push((*first).to_string());
                errors.push(format!("  - {}: must have required properties {}", path_text(path), missing.join(", ")));
                path.pop();
            }
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                for (key, property_schema) in properties {
                    if let Some(field) = object.get(key) {
                        path.push(key.clone());
                        check(field, property_schema, path, errors);
                        path.pop();
                    }
                }
            }
        }
        Value::Array(items) => {
            if let Some(item_schema) = schema.get("items").filter(|items| items.is_object()) {
                for (index, item) in items.iter().enumerate() {
                    path.push(index.to_string());
                    check(item, item_schema, path, errors);
                    path.pop();
                }
            }
        }
        _ => {}
    }
}
