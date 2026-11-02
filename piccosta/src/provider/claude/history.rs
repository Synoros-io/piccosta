// What Claude Code holds of a pi conversation, and the session files that make it hold that.
//
// Claude Code keeps its own copy of the conversation, so on every model call the bridge (mod.rs)
// asks two things of this file. Is pi's context exactly what the parked query already holds plus
// the tool results it waits for (deliver them), or anything else (rewrite)? And for a rewrite,
// which session file makes a fresh Claude Code hold exactly pi's context: a conversion of pi's
// messages, or Claude Code's own records from an earlier query of the same pi session ("kept"),
// which repeat the bytes its prompt cache has already seen.
//
// Based on piccosta's TypeScript bridge (sync.ts, convert.ts, kept.ts, session-write.ts), itself
// built on pi-claude-bridge (MIT, Eli Dickinson), and on cc-session-io 0.4.0 (MIT; both in
// LICENSE.md): its record writer, tool-pairing repair and project-path hash,
// which have to match what Claude Code reads.

use super::{debug, MCP_TOOL_PREFIX, PROVIDER, SETTINGS};
use crate::message::{AssistantMessage, Content, Message, StopReason, TextOrBlocks, ToolDefinition};
use crate::{clock, js};
use indexmap::IndexSet;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::fs;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

// What a rewrite opens with when pi's context ends at a tool result instead of a user message:
// Claude Code can't resume without a new user message, so the harness says plainly what it wants.
pub const CONTINUE_NOTE: &str = "[Continue from the tool results above.]";
// A user message with nothing in it, the same placeholder in the prompt and in a rebuilt history.
const EMPTY_USER_CONTENT: &str = "[empty]";
// An errored tool result with no output: the API refuses an empty one ("content cannot be empty if
// is_error is true"), and a refused block in history would fail every later request.
pub const EMPTY_ERROR_RESULT: &str = "[tool failed with no output]";
// pi's own answer for a tool call that never got a result (transform-messages.ts).
const NO_RESULT_PROVIDED: &str = "No result provided";
// cc-session-io's answer for one its repair finds unanswered, and for a result with no call.
const NO_RESULT_RECORDED: &str = "[no tool result recorded]";
const ORPHAN_REMOVED: &str = "[orphaned tool result removed]";
// Anthropic tool ids: ^[a-zA-Z0-9_-]+$, at most 64 characters (pi's normalizeToolCallId).
const TOOL_ID_MAX: usize = 64;
// Claude Code's project dirs: the path with every non-alphanumeric UTF-16 unit made "-", cut at 200
// with a hash of the whole path after it (cc-session-io paths.ts).
const PROJECT_DIR_MAX: usize = 200;
// The record fields cc-session-io writes for a session it creates.
const RECORD_VERSION: &str = "2.1.83";
const SLUG_ADJECTIVES: &str = "ancient bold bright calm clever cool crimson daring eager fast fierce gentle golden happy hidden iron keen lively mighty nimble pale proud quick rapid sharp silent smooth steady swift vivid warm wild";
const SLUG_NOUNS: &str = "arrow badge beacon blade brook castle cedar cloud comet crest dawn drift eagle ember falcon flame forge gale grove harbor hawk jade kettle lance maple marsh needle oak pearl pine ridge river rune sage shade spark stone thorn tower wave";
// The kept records' sidecar next to pi's session file (pi lists only *.jsonl, so it's left alone).
const SIDECAR_FORMAT: &str = "claude-bridge-kept";
const SIDECAR_VERSION: u64 = 1;
pub const SIDECAR_SUFFIX: &str = ".claude-bridge";
// What a query is started with. A change in any part means a running Claude Code holds something
// pi no longer wants. maxTokens is always unset here (pi passes none); it stays a part so sidecars
// the TypeScript bridge wrote still line up.
pub const SHAPE_PARTS: [&str; 6] = ["systemPrompt", "tools", "cliModel", "effort", "maxTokens", "cwd"];

pub type Shape = Map<String, Value>;

// What a live query holds: its shape, one print per message of pi's it holds, and the tool calls
// it is parked on (None while a turn streams or after the last one).
pub struct Held {
    pub shape: Shape,
    pub prints: Vec<String>,
    pub pending: Option<Vec<String>>,
    // Its turn is over, stopped mid-answer (pi's abort) or answered, and it waits, as Claude Code
    // does at its prompt, for the next message, which goes into this same query (mod.rs, interrupt
    // and hold).
    pub waiting: bool,
}

// --- Fingerprints ---

// Input: any JSON. Output: the sha256 of JSON.stringify of it, hex, so a held context is a list of
// short strings and a sidecar written by the TypeScript bridge compares equal.
fn hash(value: &Value) -> String {
    Sha256::digest(js::stringify(value)).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn role(message: &Message) -> &'static str {
    match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        Message::ToolResult(_) => "toolResult",
        Message::System(_) => "system",
    }
}

// User or tool-result content as the model sees it, a string folded into one text block.
fn canonical_content(blocks: &[Content]) -> Value {
    blocks.iter().map(|block| match block {
        Content::Text { text, .. } => json!({"type": "text", "text": text}),
        Content::Image { data, mime_type } => json!({"type": "image", "mimeType": mime_type, "data": data}),
        Content::Thinking { .. } => json!({"type": "thinking"}),
        Content::ToolCall { .. } => json!({"type": "toolCall"}),
    }).collect()
}

// Input: one pi message. Output: a print of what the model sees of it; two messages with the same
// print convert to the same records. pi clones messages per call and adds fields of its own, so
// identity and whole-object equality can't be used. The provider counts because thinking is
// replayed only from this provider's messages.
pub fn fingerprint(message: &Message) -> String {
    hash(&match message {
        Message::User(user) => json!({"role": "user", "content": canonical_content(&user.content.clone().into_blocks())}),
        Message::Assistant(assistant) => {
            let content: Vec<Value> = assistant.content.iter().map(|block| match block {
                Content::Text { text, .. } => json!({"type": "text", "text": text}),
                Content::Thinking { thinking, thinking_signature, redacted } => json!({"type": "thinking", "thinking": thinking,
                    "signature": thinking_signature.as_deref().unwrap_or(""), "redacted": *redacted == Some(true)}),
                Content::ToolCall { id, name, arguments, .. } => json!({"type": "toolCall", "id": id, "name": name, "arguments": arguments}),
                Content::Image { .. } => json!({"type": "image"}),
            }).collect();
            json!({"role": "assistant", "provider": assistant.provider, "content": content})
        }
        Message::ToolResult(result) => json!({"role": "toolResult", "id": result.tool_call_id, "name": result.tool_name,
            "isError": result.is_error, "content": canonical_content(&result.content)}),
        Message::System(_) => json!({"role": "system"}),
    })
}

// Input: a query's prompt, tools, Claude Code model id, effort ("off" for thinking off) and cwd.
// Output: a print per part, so a rewrite can say which one changed. Tools are compared by what the
// model is told, in order, since that is the order Claude Code lists them in.
pub fn shape(system_prompt: &str, tools: &[ToolDefinition], cli_model: &str, effort: Option<&str>, cwd: &str) -> Shape {
    let tools: Vec<Value> = tools.iter().map(|tool| json!({"name": tool.name, "description": tool.description, "parameters": tool.parameters})).collect();
    let values = [json!(system_prompt), Value::Array(tools), json!(cli_model), json!(effort), Value::Null, json!(cwd)];
    SHAPE_PARTS.iter().zip(values).map(|(part, value)| (part.to_string(), Value::String(hash(&value)))).collect()
}

// --- The decision ---

// Input: what the live query holds (None when there is none), and this call's messages with their
// prints and shape. Output: Ok(n) when the call's context is the held one (its first n messages)
// plus results for exactly the calls the query is parked on, or plus user messages for a query
// that waits, which are then delivered; otherwise Err with the first reason found, and the call
// rewrites.
pub fn decide(live: Option<&Held>, messages: &[Message], prints: &[String], shape: &Shape) -> Result<usize, String> {
    let live = live.ok_or("no live query for this session")?;
    let changed: Vec<&str> = SHAPE_PARTS.into_iter().filter(|part| shape.get(*part) != live.shape.get(*part)).collect();
    if !changed.is_empty() {
        return Err(format!("{} changed", changed.join(", ")));
    }
    // The first differing message is the most useful reason (an edited or compacted history), so it
    // is looked for before lengths.
    let held = live.prints.len();
    if let Some(index) = (0..held.min(messages.len())).find(|&index| prints[index] != live.prints[index]) {
        return Err(format!("message {index} ({}) differs from what Claude Code holds", role(&messages[index])));
    }
    if messages.len() <= held {
        return Err(format!("context has {} messages, Claude Code holds {held}", messages.len()));
    }
    // A waiting query takes what comes next: its context (with its answer, when it gave one; a
    // stopped answer pi drops), then user messages only, which include pi's custom messages (a job's
    // notice, lane news).
    if live.waiting {
        let after: Vec<&Message> = messages[held..].iter().filter(visible).collect();
        if after.is_empty() || !after.iter().all(|message| matches!(message, Message::User(_))) {
            return Err("what came after the waiting query's turn isn't only user messages".into());
        }
        return Ok(held);
    }
    let pending = live.pending.as_ref().ok_or("the live query is not parked at a tool call")?;
    // What pi added has to be tool results and nothing else: a user message here (a steer, an
    // extension's text) reaches Claude Code as part of a rewritten context, not mid-query.
    let mut ids = Vec::new();
    for (offset, message) in messages[held..].iter().enumerate() {
        match message {
            Message::ToolResult(result) => ids.push(result.tool_call_id.as_str()),
            other => return Err(format!("message {} is a {} after the tool results", held + offset, role(other))),
        }
    }
    let unique: HashSet<&&str> = ids.iter().collect();
    if ids.len() != pending.len() || unique.len() != ids.len() || !ids.iter().all(|id| pending.iter().any(|parked| parked == id)) {
        return Err(format!("tool results [{}] are not the calls Claude Code is parked on [{}]", ids.join(", "), pending.join(", ")));
    }
    Ok(held)
}

// A turn that ended in an error or an abort is not part of what the model sees (pi drops it for
// every provider), so it is neither written nor the "last message" a rewrite splits by.
pub fn visible(message: &&Message) -> bool {
    !matches!(message, Message::Assistant(AssistantMessage { stop_reason: StopReason::Error | StopReason::Aborted, .. }))
}

// Where the current user turn starts: the trailing run of user messages, or the end.
pub fn turn_start(messages: &[Message]) -> usize {
    messages.iter().rposition(|message| !matches!(message, Message::User(_))).map_or(0, |index| index + 1)
}

// Text and image blocks in Anthropic's shape, empty ones left out.
pub fn anthropic_blocks(blocks: &[Content]) -> Vec<Value> {
    blocks.iter().filter_map(|block| match block {
        Content::Text { text, .. } if !text.is_empty() => Some(json!({"type": "text", "text": text})),
        Content::Image { data, mime_type } if !data.is_empty() && !mime_type.is_empty() =>
            Some(json!({"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": data}})),
        _ => None,
    }).collect()
}

// Input: the visible messages. Output: where the history ends, the prompt the new query opens
// with (the trailing user run as one prompt, since the API merges consecutive user messages
// anyway), and whether that prompt is the continue note.
pub fn split(messages: &[Message]) -> (usize, Vec<Value>, bool) {
    let start = turn_start(messages);
    if start == messages.len() {
        return (start, vec![json!({"type": "text", "text": CONTINUE_NOTE})], true);
    }
    let mut prompt: Vec<Value> = messages[start..].iter().flat_map(|message| match message {
        Message::User(user) => anthropic_blocks(&user.content.clone().into_blocks()),
        _ => Vec::new(),
    }).collect();
    if prompt.is_empty() {
        prompt.push(json!({"type": "text", "text": EMPTY_USER_CONTENT}));
    }
    (start, prompt, false)
}

// --- Converting pi's messages ---

// pi's rule for Anthropic tool ids: every character outside [a-zA-Z0-9_-] (per UTF-16 unit) becomes
// "_", then the first 64 are kept. A pure function, so a pi id maps the same way on every rebuild.
pub fn sanitize_tool_id(id: &str) -> String {
    id.chars().flat_map(|c| {
        let keep = c.is_ascii_alphanumeric() || c == '_' || c == '-';
        std::iter::repeat_n(if keep { c } else { '_' }, if keep { 1 } else { c.len_utf16() })
    }).take(TOOL_ID_MAX).collect()
}

// A pi tool name as Claude knows it: the served tool's own casing behind the MCP prefix. A tool pi
// ran that isn't served now still gets the prefix; naming it after a Claude Code builtin would tell
// the model about a tool it can't call.
fn sdk_tool_name(name: &str, served: &[ToolDefinition]) -> String {
    let lower = name.to_lowercase();
    let served_name = served.iter().find(|tool| tool.name == name)
        .or_else(|| served.iter().find(|tool| tool.name.to_lowercase() == lower))
        .map_or(name, |tool| tool.name.as_str());
    format!("{MCP_TOOL_PREFIX}{served_name}")
}

// A tool result as text, as Claude Code stores most of them; one carrying an image keeps blocks.
fn tool_result_content(content: &[Content]) -> Value {
    if content.iter().any(|block| matches!(block, Content::Image { data, mime_type } if !data.is_empty() && !mime_type.is_empty())) {
        return Value::Array(anthropic_blocks(content));
    }
    let texts: Vec<&str> = content.iter().filter_map(|block| match block {
        Content::Text { text, .. } if !text.is_empty() => Some(text.as_str()),
        _ => None,
    }).collect();
    json!(texts.join("\n"))
}

// Input: pi messages (no system ones) and the tools being served. Output: the same conversation as
// Anthropic messages, ready for the record writer.
// - A turn that failed, or was aborted before anything streamed, is left out.
// - Thinking is replayed only when this provider minted its signature; any other is not ours to
//   hand back, and Anthropic rejects signatures it can't verify.
// - A turn's tool results all go into one user message directly after it, even when pi recorded a
//   steer between them: cc-session-io's repair pairs only the results in the first user message
//   after a call, and would stub the rest. Claude Code orders a mid-turn steer the same way.
// - A tool call that never got a result gets pi's "No result provided" error result.
pub fn convert(messages: &[Message], served: &[ToolDefinition]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    // The open turn's assistant message and the user message holding its results, by index.
    let mut turn_assistant: Option<usize> = None;
    let mut turn_results: Option<usize> = None;
    // Every turn with tool calls: its assistant's index, the call ids, its results' index.
    let mut tool_turns: Vec<(usize, Vec<String>, Option<usize>)> = Vec::new();
    let mut current_turn: Option<usize> = None;
    for message in messages {
        match message {
            Message::User(user) => {
                let content = match &user.content {
                    TextOrBlocks::Text(text) => json!(if text.is_empty() { EMPTY_USER_CONTENT } else { text }),
                    TextOrBlocks::Blocks(blocks) => {
                        let parts = anthropic_blocks(blocks);
                        if parts.is_empty() { json!("[image]") } else { Value::Array(parts) }
                    }
                };
                out.push(json!({"role": "user", "content": content}));
            }
            Message::Assistant(assistant) => {
                // A turn aborted before anything streamed is dropped rather than given a placeholder,
                // which would invent a reply early in the prefix and cost the cache every rebuild.
                // turn_results stays as it is, so a result after it still joins its own turn.
                if !visible(&message) || assistant.content.is_empty() {
                    continue;
                }
                let mut blocks: Vec<Value> = assistant.content.iter().filter_map(|block| match block {
                    Content::Text { text, .. } if !text.is_empty() => Some(json!({"type": "text", "text": text})),
                    Content::Thinking { thinking, thinking_signature: Some(signature), .. } if assistant.provider == PROVIDER && !signature.is_empty() =>
                        Some(json!({"type": "thinking", "thinking": thinking, "signature": signature})),
                    Content::ToolCall { id, name, arguments, .. } =>
                        Some(json!({"type": "tool_use", "id": sanitize_tool_id(id), "name": sdk_tool_name(name, served), "input": arguments})),
                    _ => None,
                }).collect();
                // Content that was all filtered keeps its slot: an empty message is refused, and
                // dropping it could leave a tool result with no call before it.
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": "[incompatible content omitted]"}));
                }
                let ids: Vec<String> = blocks.iter().filter(|block| block["type"] == "tool_use").filter_map(|block| block["id"].as_str().map(String::from)).collect();
                turn_results = None;
                turn_assistant = Some(out.len());
                current_turn = (!ids.is_empty()).then(|| {
                    tool_turns.push((out.len(), ids, None));
                    tool_turns.len() - 1
                });
                out.push(json!({"role": "assistant", "content": blocks}));
            }
            Message::ToolResult(result) => {
                let content = tool_result_content(&result.content);
                let content = if result.is_error && content == "" { json!(EMPTY_ERROR_RESULT) } else { content };
                let block = json!({"type": "tool_result", "tool_use_id": sanitize_tool_id(&result.tool_call_id), "content": content, "is_error": result.is_error});
                match turn_results {
                    Some(index) => out[index]["content"].as_array_mut().into_iter().for_each(|blocks| blocks.push(block.clone())),
                    None => {
                        // Right after its assistant message; with none before it the history is
                        // malformed, and the result goes last for the repair to discard.
                        let at = turn_assistant.map_or(out.len(), |index| index + 1);
                        out.insert(at, json!({"role": "user", "content": [block]}));
                        turn_results = Some(at);
                        if let Some(turn) = current_turn {
                            tool_turns[turn].2 = Some(at);
                        }
                    }
                }
            }
            Message::System(_) => {}
        }
    }
    // Unanswered calls, once at the end so a late result is never answered twice. The latest turn
    // goes first, so an insertion never moves an index still to be used.
    for (assistant, ids, results) in tool_turns.into_iter().rev() {
        let answered: HashSet<String> = results.and_then(|index| out[index]["content"].as_array().cloned()).unwrap_or_default()
            .iter().filter_map(|block| block["tool_use_id"].as_str().map(String::from)).collect();
        let stubs: Vec<Value> = ids.iter().filter(|id| !answered.contains(*id))
            .map(|id| json!({"type": "tool_result", "tool_use_id": id, "content": NO_RESULT_PROVIDED, "is_error": true})).collect();
        if stubs.is_empty() {
            continue;
        }
        match results {
            Some(index) => out[index]["content"].as_array_mut().into_iter().for_each(|blocks| blocks.extend(stubs.clone())),
            None => out.insert(assistant + 1, json!({"role": "user", "content": stubs})),
        }
    }
    out
}

// cc-session-io's repairToolPairing, which its import applies: every tool_use answered in the user
// message right after it (a missing one stubbed first), and every tool_result there for a call
// that was made (others dropped). Input and output: Anthropic messages.
fn repair(messages: Vec<Value>) -> Vec<Value> {
    let stub = |id: &String| json!({"type": "tool_result", "tool_use_id": id, "content": NO_RESULT_RECORDED, "is_error": true});
    let mut out: Vec<Value> = Vec::new();
    let mut pending: Option<IndexSet<String>> = None;
    let flush = |out: &mut Vec<Value>, pending: &mut Option<IndexSet<String>>| {
        if let Some(ids) = pending.take().filter(|ids| !ids.is_empty()) {
            out.push(json!({"role": "user", "content": ids.iter().map(stub).collect::<Vec<_>>()}));
        }
    };
    for mut message in messages {
        if message["role"] == "assistant" {
            flush(&mut out, &mut pending);
            let ids: IndexSet<String> = message["content"].as_array().into_iter().flatten()
                .filter(|block| block["type"] == "tool_use").filter_map(|block| block["id"].as_str().map(String::from)).collect();
            pending = (!ids.is_empty()).then_some(ids);
            out.push(message);
            continue;
        }
        let blocks = message["content"].as_array().cloned();
        let has_results = blocks.as_ref().is_some_and(|blocks| blocks.iter().any(|block| block["type"] == "tool_result"));
        if pending.is_none() && !has_results {
            out.push(message);
            continue;
        }
        let input = blocks.unwrap_or_else(|| match message["content"].as_str() {
            Some(text) if !text.is_empty() => vec![json!({"type": "text", "text": text})],
            _ => Vec::new(),
        });
        let mut provided = HashSet::new();
        let mut kept: Vec<Value> = input.into_iter().filter(|block| {
            let id = block["tool_use_id"].as_str().unwrap_or_default();
            block["type"] != "tool_result" || (pending.as_ref().is_some_and(|ids| ids.contains(id)) && { provided.insert(id.to_string()); true })
        }).collect();
        if let Some(ids) = pending.take() {
            kept.splice(0..0, ids.iter().filter(|id| !provided.contains(*id)).map(stub));
        }
        if kept.is_empty() {
            if out.is_empty() {
                out.push(json!({"role": "user", "content": [{"type": "text", "text": ORPHAN_REMOVED}]}));
            }
            continue;
        }
        message["content"] = Value::Array(kept);
        out.push(message);
    }
    flush(&mut out, &mut pending);
    out
}

// --- Session files ---

// Claude Code's config dir for the bridge's children: CLAUDE_BRIDGE_CONFIG_DIR, else
// CLAUDE_CONFIG_DIR, else ~/.claude.
fn claude_dir() -> PathBuf {
    SETTINGS.config_dir.clone().or_else(|| std::env::var_os("CLAUDE_CONFIG_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".claude"))
}

// A path with symlinks resolved, or the path itself when it can't be.
pub fn real_or(path: &str) -> String {
    fs::canonicalize(path).map_or_else(|_| path.to_string(), |real| real.to_string_lossy().into_owned())
}

// Input: a project path. Output: Claude Code's dir for it, <config>/projects/<hash>, the hash taken
// of its realpath in NFC.
fn project_dir(project_path: &str) -> PathBuf {
    let path: String = real_or(project_path).nfc().collect();
    let sanitized: String = path.encode_utf16().map(|unit| match u8::try_from(unit) {
        Ok(byte) if byte.is_ascii_alphanumeric() => byte as char,
        _ => '-',
    }).collect();
    let name = if sanitized.len() <= PROJECT_DIR_MAX {
        sanitized
    } else {
        // JavaScript's (h << 5) - h + code | 0 over UTF-16 units, then Math.abs(h).toString(36).
        let mut hash = (path.encode_utf16().fold(0i32, |h, unit| (h << 5).wrapping_sub(h).wrapping_add(unit as i32)) as i64).abs();
        let mut digits = Vec::new();
        while hash > 0 || digits.is_empty() {
            digits.push(char::from_digit((hash % 36) as u32, 36).unwrap_or('0'));
            hash /= 36;
        }
        format!("{}-{}", &sanitized[..PROJECT_DIR_MAX], digits.iter().rev().collect::<String>())
    };
    claude_dir().join("projects").join(name)
}

pub fn session_path(session_id: &str, project_path: &str) -> PathBuf {
    project_dir(project_path).join(format!("{session_id}.jsonl"))
}

// Delete a session's file and companion dir, then the project dir when that was the last thing
// in it (remove_dir refuses a dir that isn't empty, so another session's files are never at risk).
pub fn delete_session(session_id: &str, project_path: &str) {
    let path = session_path(session_id, project_path);
    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir_all(path.with_extension(""));
    let _ = fs::remove_dir(project_dir(project_path));
}

// A session file being written, as cc-session-io's Session builds one: every record a link in the
// uuid chain, timestamps 1 ms apart from now.
pub struct Records {
    pub session_id: String,
    pub path: PathBuf,
    cwd: String,
    model: String,
    slug: String,
    last_uuid: Option<String>,
    next_ms: i64,
    pub records: Vec<Value>,
}

// A few random bytes, from a v4 uuid (no rand crate in the tree).
fn random_bytes() -> [u8; 16] {
    *Uuid::new_v4().as_bytes()
}

impl Records {
    pub fn new(project_path: &str, cwd: &str, model: &str) -> Records {
        let session_id = Uuid::new_v4().to_string();
        let [first, second, third, ..] = random_bytes();
        let adjectives: Vec<&str> = SLUG_ADJECTIVES.split(' ').collect();
        let nouns: Vec<&str> = SLUG_NOUNS.split(' ').collect();
        let pick = |words: &[&str], byte: u8| words[byte as usize % words.len()].to_string();
        Records {
            path: session_path(&session_id, project_path),
            session_id,
            cwd: cwd.to_string(),
            model: model.to_string(),
            slug: format!("{}-{}-{}", pick(&adjectives, first), pick(&adjectives, second), pick(&nouns, third)),
            last_uuid: None,
            next_ms: clock::now_ms(),
            records: Vec::new(),
        }
    }

    // The fields every record starts with, in cc-session-io's order; the new record joins the chain.
    fn base(&mut self, kind: &str) -> Map<String, Value> {
        let uuid = Uuid::new_v4().to_string();
        let record = json!({"type": kind, "uuid": uuid, "parentUuid": self.last_uuid, "sessionId": self.session_id,
            "timestamp": clock::iso_from_ms(self.next_ms), "isSidechain": false, "cwd": self.cwd, "userType": "external",
            "version": RECORD_VERSION, "gitBranch": "HEAD", "slug": self.slug, "entrypoint": "cli"});
        self.next_ms += 1;
        self.last_uuid = Some(uuid);
        record.as_object().cloned().unwrap_or_default()
    }

    fn user(&mut self, content: Value) {
        let mut record = self.base("user");
        record.insert("message".into(), json!({"role": "user", "content": content}));
        self.records.push(Value::Object(record));
    }

    // cc-session-io's importMessages: repaired first, then one record per assistant message and
    // per user message, a user message's tool results split from (and ahead of) its other blocks.
    pub fn import(&mut self, messages: Vec<Value>) {
        let synthetic_id = |prefix: &str| format!("{prefix}{}", &Uuid::new_v4().simple().to_string()[..24]);
        for message in repair(messages) {
            let content = message["content"].clone();
            if message["role"] == "assistant" {
                let content = match content {
                    Value::String(text) => json!([{"type": "text", "text": text}]),
                    other => other,
                };
                let tool_use = content.as_array().is_some_and(|blocks| blocks.iter().any(|block| block["type"] == "tool_use"));
                let mut record = self.base("assistant");
                record.insert("requestId".into(), json!(synthetic_id("req_syn_")));
                record.insert("message".into(), json!({"id": synthetic_id("msg_syn_"), "type": "message", "role": "assistant", "model": self.model,
                    "content": content, "stop_reason": if tool_use { "tool_use" } else { "end_turn" }, "stop_sequence": null,
                    "usage": {"input_tokens": 0, "output_tokens": 0}}));
                self.records.push(Value::Object(record));
                continue;
            }
            let Some(blocks) = content.as_array() else {
                self.user(content);
                continue;
            };
            let (results, rest): (Vec<&Value>, Vec<&Value>) = blocks.iter().partition(|block| block["type"] == "tool_result");
            if !results.is_empty() {
                self.user(results.iter().map(|result| {
                    let mut block = json!({"type": "tool_result", "tool_use_id": result["tool_use_id"], "content": result["content"]});
                    if result["is_error"] == true {
                        block["is_error"] = json!(true);
                    }
                    block
                }).collect());
            }
            if !rest.is_empty() {
                self.user(Value::Array(rest.into_iter().cloned().collect()));
            }
        }
    }

    pub fn save(&self) -> std::io::Result<()> {
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(&self.path, self.records.iter().map(|record| js::stringify(record) + "\n").collect::<String>())
    }
}

// --- Kept records ---

// What a clean query end leaves for its pi session: Claude Code's chain records and the prints and
// roles of the pi messages they hold (failed turns left out).
struct Kept {
    shape: Shape,
    prints: Vec<String>,
    roles: Vec<String>,
    records: Vec<Value>,
}

// Per pi session: where its sidecar lives (None when pi doesn't save the session), what it kept.
type KeptSessions = HashMap<String, (Option<PathBuf>, Option<Kept>)>;
static KEPT: LazyLock<Mutex<KeptSessions>> = LazyLock::new(Default::default);

fn strings(value: &Value) -> Vec<String> {
    value.as_array().into_iter().flatten().filter_map(|item| item.as_str().map(String::from)).collect()
}

// The first call for a pi session: remember its sidecar and load what an earlier process kept.
pub fn attach(key: &str, session_file: Option<&Path>) {
    let mut kept = KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !SETTINGS.keep || kept.contains_key(key) {
        return;
    }
    let sidecar = session_file.map(|file| PathBuf::from(format!("{}{SIDECAR_SUFFIX}", file.display())));
    let loaded = sidecar.as_ref().and_then(|path| fs::read_to_string(path).ok()).and_then(|text| {
        let mut lines = text.lines().filter(|line| !line.trim().is_empty()).map(serde_json::from_str::<Value>);
        let meta = lines.next()?.ok()?;
        if meta["format"] != SIDECAR_FORMAT || meta["version"] != SIDECAR_VERSION {
            return None;
        }
        Some(Kept { shape: meta["shape"].as_object().cloned()?, prints: strings(&meta["prints"]), roles: strings(&meta["roles"]),
            records: lines.collect::<Result<_, _>>().ok()? })
    });
    kept.insert(key.to_string(), (sidecar, loaded));
}

// At a clean query end: keep its session file's chain records for the next rewrite. Input: the pi
// session, Claude Code's session id and project path, and what it holds (shape, prints, roles, its
// final answer included). Written to the sidecar whole and renamed into place.
pub fn keep(key: &str, cc_session_id: &str, project_path: &str, cwd: &str, shape: &Shape, prints: Vec<String>, roles: Vec<String>) {
    if !SETTINGS.keep {
        return;
    }
    let file = session_path(cc_session_id, project_path);
    let Ok(text) = fs::read_to_string(&file) else {
        return debug(&format!("kept: could not read {}, nothing kept", file.display()));
    };
    let records: Vec<Value> = text.lines().filter_map(|line| serde_json::from_str::<Value>(line).ok()).filter(|record| record["uuid"].is_string()).collect();
    let mut map = KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = map.entry(key.to_string()).or_insert((None, None));
    debug(&format!("kept: {} records of {cc_session_id} for {} pi messages", records.len(), prints.len()));
    if let Some(sidecar) = &entry.0 {
        let meta = json!({"format": SIDECAR_FORMAT, "version": SIDECAR_VERSION, "cwd": cwd,
            "shape": shape, "prints": prints, "roles": roles});
        let text: String = std::iter::once(&meta).chain(&records).map(|record| js::stringify(record) + "\n").collect();
        let partial = PathBuf::from(format!("{}.partial", sidecar.display()));
        if fs::write(&partial, text).and_then(|()| fs::rename(&partial, sidecar)).is_err() {
            debug(&format!("kept: could not save {}", sidecar.display()));
        }
    }
    entry.1 = Some(Kept { shape: shape.clone(), prints, roles, records });
}

#[derive(PartialEq, Clone, Copy)]
enum Kind {
    Prompt,
    Results,
    Assistant,
    Other,
}

// Claude Code's own bookkeeping attachments for the prompt it was started with. On a resume it
// sends the last recorded snapshot instead of the prompt it is given (measured 2026-10-29 on 2.1.300
// against a local mock API: a kept rewrite after a prompt change got the old prompt), so copies
// leave them out and the prompt pi gives is the one the model sees.
const PROMPT_ATTACHMENTS: [&str; 3] = ["prompt_snapshot", "prompt_render_point", "deferred_tools_record"];

// Claude Code's notice for a turn that got no reply: it writes one after a resumed session that ends
// at a user record. The bridge has it answer such a turn instead (mod.rs, RESUME_ENV), but kept
// records from before can hold the notice, and read back as the model's own words, a run of them
// teaches it to answer that way.
pub const NO_RESPONSE: &str = "No response requested.";

// Whether an assistant message is that notice.
pub fn no_response(message: &Value) -> bool {
    message["model"] == "<synthetic>" && message["content"].as_array().is_some_and(|blocks| blocks.iter().all(|block| block["text"] == NO_RESPONSE))
}

// Input: records copied from a kept session. Output: the same without the prompt attachments and
// the no-reply notices, each dropped one's child re-parented to the dropped one's own parent so the
// uuid chain stays whole.
fn without_prompt_attachments(records: &[Value]) -> Vec<Value> {
    let mut parents: HashMap<String, Value> = HashMap::new();
    let mut out = Vec::new();
    for record in records {
        let parent = record["parentUuid"].as_str().and_then(|parent| parents.get(parent).cloned()).unwrap_or_else(|| record["parentUuid"].clone());
        let attachment = record["type"] == "attachment" && PROMPT_ATTACHMENTS.iter().any(|kind| record["attachment"]["type"] == *kind);
        if attachment || (record["type"] == "assistant" && no_response(&record["message"])) {
            parents.insert(record["uuid"].as_str().unwrap_or_default().to_string(), parent);
            continue;
        }
        let mut record = record.clone();
        record["parentUuid"] = parent;
        out.push(record);
    }
    out
}

// What a kept record is in the conversation. A meta user record and Claude Code's <synthetic>
// notices ("No response requested.") have no pi message.
fn kind(record: &Value) -> Kind {
    match record["type"].as_str() {
        Some("assistant") if record["message"]["model"] == "<synthetic>" => Kind::Other,
        Some("assistant") => Kind::Assistant,
        Some("user") if record["isMeta"] != true => {
            let results = record["message"]["content"].as_array().is_some_and(|blocks| blocks.iter().any(|block| block["type"] == "tool_result"));
            if results { Kind::Results } else { Kind::Prompt }
        }
        _ => Kind::Other,
    }
}

fn is_message(record: &Value) -> bool {
    record["type"] == "user" || record["type"] == "assistant"
}

// The note Claude Code puts in the user turn after an interrupt ("[Request interrupted by user]",
// or "... for tool use]"). pi's history has no message for it, but the stopped query sent it, so a
// rewrite copies its record as is: matched as part of the user's prompt it ended the copy there, and
// everything after was converted into different bytes, a whole rewrite of the cache (stand-in,
// 2026-11-01; 9.3M tokens over a day of real sessions were turns after an abort).
const INTERRUPT_NOTE: &str = "[Request interrupted by user";

// User content from either side as what the model sees of it: short strings, empty texts and
// Claude Code's interrupt notes left out.
fn prompt_parts(content: &Value) -> Vec<String> {
    let seen = |text: &&str| !text.is_empty() && !text.starts_with(INTERRUPT_NOTE);
    if let Some(text) = content.as_str() {
        return Some(text).filter(seen).map(|text| format!("t:{text}")).into_iter().collect();
    }
    content.as_array().into_iter().flatten().filter_map(|block| match block["type"].as_str() {
        Some("text") => block["text"].as_str().filter(seen).map(|text| format!("t:{text}")),
        Some("image") => Some(format!("i:{}", block["data"].as_str().or(block["source"]["data"].as_str()).unwrap_or_default())),
        _ => None,
    }).collect()
}

// The kept records of answers an interrupt stopped: the assistant records just before an interrupt
// note, back to the user turn they answer. Claude Code keeps a stopped answer's finished blocks and
// sent them on, but pi drops the answer, so they line up with no pi message and are copied with the
// note, as the bytes the cache holds (left in as an assistant turn, they ended the copy there).
fn stopped_answers(records: &[Value]) -> HashSet<usize> {
    let note = |record: &Value| {
        let content = &record["message"]["content"];
        let texts: Vec<&str> = content.as_str().into_iter().chain(content.as_array().into_iter().flatten().filter_map(|block| block["text"].as_str())).collect();
        texts.iter().any(|text| text.starts_with(INTERRUPT_NOTE))
    };
    let mut stopped = HashSet::new();
    for index in (0..records.len()).filter(|&index| kind(&records[index]) == Kind::Prompt && note(&records[index])) {
        let mut back = index;
        while back > 0 && matches!(kind(&records[back - 1]), Kind::Assistant | Kind::Other) {
            back -= 1;
            if kind(&records[back]) == Kind::Assistant {
                stopped.insert(back);
            }
        }
    }
    stopped
}

// Assistant blocks from either side as their text and tool call ids, enough to tell two turns apart
// (thinking is left out: its signature already decided the print, and Claude Code splits it).
fn assistant_key(blocks: &[Value], sanitize: bool) -> String {
    let texts: Vec<&str> = blocks.iter().filter(|block| block["type"] == "text").filter_map(|block| block["text"].as_str()).filter(|text| !text.is_empty()).collect();
    let ids: Vec<String> = blocks.iter().filter(|block| block["type"] == "toolCall" || block["type"] == "tool_use")
        .map(|block| block["id"].as_str().unwrap_or_default()).map(|id| if sanitize { sanitize_tool_id(id) } else { id.to_string() }).collect();
    js::stringify(&json!([texts.join("\n"), ids]))
}

// Line the kept records up with the first `count` history messages. Output: per message, the index
// of the last record holding it, or None where no cut can follow it (inside a run of user messages
// or tool results, which Claude Code holds as one block). Stops at the first message that doesn't
// line up, so the list can be shorter than `count`.
fn align(kept: &Kept, history: &[Message], count: usize) -> Vec<Option<usize>> {
    let records = &kept.records;
    let stopped = stopped_answers(records);
    let kind_of = |r: usize| if stopped.contains(&r) { Kind::Other } else { kind(&records[r]) };
    let content = |message: &Message| serde_json::to_value(match message {
        Message::User(user) => user.content.clone(),
        Message::Assistant(assistant) => TextOrBlocks::Blocks(assistant.content.clone()),
        _ => TextOrBlocks::Blocks(Vec::new()),
    }).unwrap_or_default();
    let (mut ends, mut r, mut i) = (Vec::new(), 0, 0);
    while i < count {
        while r < records.len() && kind_of(r) == Kind::Other {
            r += 1;
        }
        // The continue note a query opened with has no pi message of its own.
        if r < records.len() && kind_of(r) == Kind::Prompt && role(&history[i]) != "user"
            && prompt_parts(&records[r]["message"]["content"]) == [format!("t:{CONTINUE_NOTE}")]
        {
            r += 1;
            continue;
        }
        if r >= records.len() {
            break;
        }
        let role = role(&history[i]);
        let mut run_end = i + 1;
        if role != "assistant" {
            while run_end < kept.roles.len() && kept.roles[run_end] == role {
                run_end += 1;
            }
        }
        if run_end > count {
            break;
        }
        let last = match role {
            "user" | "toolResult" => {
                let (want_kind, want): (Kind, Vec<String>) = if role == "user" {
                    (Kind::Prompt, history[i..run_end].iter().flat_map(|message| prompt_parts(&content(message))).collect())
                } else {
                    let mut ids: Vec<String> = history[i..run_end].iter().filter_map(|message| match message {
                        Message::ToolResult(result) => Some(sanitize_tool_id(&result.tool_call_id)),
                        _ => None,
                    }).collect();
                    ids.sort();
                    (Kind::Results, ids)
                };
                let (mut got, mut last) = (Vec::new(), None);
                while r < records.len() && matches!(kind_of(r), k if k == want_kind || k == Kind::Other) && (want_kind == Kind::Prompt || got.len() < want.len()) {
                    if kind_of(r) == want_kind {
                        let content = &records[r]["message"]["content"];
                        if want_kind == Kind::Prompt {
                            got.extend(prompt_parts(content));
                        } else {
                            got.extend(content.as_array().into_iter().flatten().filter(|block| block["type"] == "tool_result").filter_map(|block| block["tool_use_id"].as_str().map(String::from)));
                        }
                        last = Some(r);
                    }
                    r += 1;
                    if want_kind == Kind::Prompt && got.len() >= want.len() {
                        break;
                    }
                }
                if want_kind == Kind::Results {
                    got.sort();
                }
                match last.filter(|_| got == want) {
                    Some(last) => last,
                    None => break,
                }
            }
            _ => {
                if kind_of(r) != Kind::Assistant {
                    break;
                }
                // Claude Code writes one record per content block, all under the response's id.
                let id = records[r]["message"]["id"].clone();
                let (mut blocks, mut last) = (Vec::new(), r);
                while r < records.len() {
                    match kind_of(r) {
                        Kind::Assistant if records[r]["message"]["id"] == id => {
                            blocks.extend(records[r]["message"]["content"].as_array().cloned().unwrap_or_default());
                            last = r;
                        }
                        Kind::Other => {}
                        _ => break,
                    }
                    r += 1;
                }
                let pi_blocks = content(&history[i]).as_array().cloned().unwrap_or_default();
                if assistant_key(&blocks, false) != assistant_key(&pi_blocks, true) {
                    break;
                }
                last
            }
        };
        ends.extend(std::iter::repeat_n(None, run_end - i - 1));
        ends.push(Some(last));
        r = last + 1;
        i = run_end;
    }
    ends
}

// Input: a pi session, the history the new session must hold with its prints, and what conversion
// needs. Output: the session written from kept records for the longest prefix they hold plus a
// conversion of the rest, or None when nothing could be copied (the caller converts it all).
#[allow(clippy::too_many_arguments)]
fn from_kept(key: &str, history: &[Message], prints: &[String], cwd: &str, project_path: &str, served: &[ToolDefinition], cli_model: &str, shape: &Shape) -> Option<Records> {
    let map = KEPT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let kept = map.get(key)?.1.as_ref()?;
    // The model and cwd are written into Claude Code's own records ("You are powered by", the
    // environment block); copied under another they would tell the model something false.
    if kept.shape.get("cliModel") != shape.get("cliModel") || kept.shape.get("cwd") != shape.get("cwd") {
        debug("kept: not used, the model or cwd changed");
        return None;
    }
    let shared = prints.iter().zip(&kept.prints).take_while(|(print, kept)| print == kept).count();
    let ends = align(kept, history, shared);
    // A cut has to leave every tool call with its results on the same side: a converted result whose
    // call was copied would be dropped as an orphan, a copied call whose results were converted
    // would reach the API unanswered.
    let mut cut = ends.len().min(shared);
    let has_calls = |message: &Message| matches!(message, Message::Assistant(assistant) if assistant.content.iter().any(|block| matches!(block, Content::ToolCall { .. })));
    while cut > 0 && (ends[cut - 1].is_none() || matches!(history.get(cut), Some(Message::ToolResult(_))) || has_calls(&history[cut - 1])) {
        cut -= 1;
    }
    let mut last = ends.get(cut.checked_sub(1)?).copied().flatten()?;
    // The chain links after the last copied message, up to the next one (the attachments Claude Code
    // writes after a prompt) go too.
    while last + 1 < kept.records.len() && !is_message(&kept.records[last + 1]) {
        last += 1;
    }
    let mut records = Records::new(project_path, cwd, cli_model);
    records.records = without_prompt_attachments(&kept.records[..=last]).into_iter().map(|mut record| {
        record["sessionId"] = json!(records.session_id);
        record
    }).collect();
    records.last_uuid = records.records.iter().rev().find(|record| matches!(record["type"].as_str(), Some("user" | "assistant" | "attachment")))
        .and_then(|record| record["uuid"].as_str().map(String::from));
    let copied = records.records.len();
    records.import(convert(&history[cut..], served));
    // The writer chains new records from the last message; Claude Code chains them from the last
    // link, attachments included.
    let last_link = records.records[copied - 1]["uuid"].clone();
    if let Some(first_new) = records.records.get_mut(copied).filter(|_| last_link.is_string()) {
        first_new["parentUuid"] = last_link;
    }
    debug(&format!("kept: {cut} of {} history messages from kept records", history.len()));
    Some(records)
}

// Write the session a rewrite resumes. Input: the pi session, the history (visible messages before
// the prompt) and its prints, the cwd and its realpath, the served tools, the Claude Code model and
// the query's shape. Output: the new session's id. Always a fresh id: a discarded Claude Code may
// still be flushing into the file it had.
#[allow(clippy::too_many_arguments)]
pub fn write_session(key: &str, history: &[Message], prints: &[String], cwd: &str, project_path: &str, served: &[ToolDefinition], cli_model: &str, shape: &Shape) -> std::io::Result<String> {
    let records = from_kept(key, history, prints, cwd, project_path, served, cli_model, shape).unwrap_or_else(|| {
        let mut records = Records::new(project_path, &real_or(cwd).nfc().collect::<String>(), cli_model);
        records.import(convert(history, served));
        records
    });
    records.save()?;
    Ok(records.session_id)
}
