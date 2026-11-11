// The small JSON-RPC envelope common to the ACP client and server. Payloads stay JSON: neither
// side needs a second copy of the whole ACP schema.
use serde_json::{Value, json};

pub(crate) fn valid(message: &Value) -> bool {
    message.is_object()
        && message["jsonrpc"] == "2.0"
        && message.get("id").is_none_or(|id| id.is_null() || id.is_string() || id.is_number())
        && match message.get("method") {
            Some(method) => method.is_string() && message.get("result").is_none() && message.get("error").is_none(),
            None => message.get("id").is_some() && (message.get("result").is_some() ^ message.get("error").is_some()),
        }
}
pub(crate) fn request(id: Value, method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}
pub(crate) fn notification(method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "method": method, "params": params})
}
pub(crate) fn reply(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}
pub(crate) fn error(id: Value, code: i64, message: impl Into<String>, data: Option<Value>) -> Value {
    let mut error = json!({"code": code, "message": message.into()});
    if let Some(data) = data {
        error["data"] = data;
    }
    json!({"jsonrpc": "2.0", "id": id, "error": error})
}
