// Session files: a conversation stored as an append-only tree, one JSON object per line.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/session-manager.ts.
// The format is pi's version 3, unchanged, so sessions move freely between pi and piccosta.
//
// Line 1 is the header. Every later line is an entry with an 8-character id and the id of its
// parent. Appending always adds a child of the current leaf and makes it the new leaf; going back
// to an earlier point (fork, /tree) just moves the leaf, so history is never rewritten. On open,
// the leaf is the last entry in the file.
//
// Left out on purpose:
// - pi's v1-to-v3 migrations. They only matter for sessions written before pi 1.0, and piccosta
//   has none. Such a file is refused with a message rather than rewritten.
// - Session listing, fork and branch copying, and labels' UI. They come with the commands that
//   use them.

pub mod index;

use crate::clock;
use crate::message::{
    to_model_messages, AgentMessage, BranchSummaryMessage, CompactionSummaryMessage, Content, CustomMessage, SystemSnapshot,
    TextOrBlocks, Usage,
};
use crate::provider::transcript::current_system_message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Read, Seek, Write};
use std::path::{Path, PathBuf};

// The session format version this code reads and writes (session-manager.ts:41).
pub const CURRENT_SESSION_VERSION: u32 = 3;

// About 6k tokens at four characters a token, enough discussion without filling a delegate.
const DISCUSSION_CHARS: usize = 24_000;

// Only the user's text, not images or custom messages (pintask's pinned task uses this too).
pub(crate) fn user_text(entry: &SessionEntry) -> Option<String> {
    let SessionEntry::Message { message: AgentMessage::User(user), .. } = entry else { return None };
    Some(user.content.text())
}

// Input: the live branch and the first words of the task's user turn. Output: its words before
// the current tool call, not context edits or summaries. Trim whole assistants before whole users.
fn discussion(session: &Session, call: &str, start: Option<&str>) -> String {
    let branch = session.path_to_leaf();
    let end = branch.iter().rposition(|entry| matches!(entry, SessionEntry::Message { message: AgentMessage::Assistant(reply), .. } if reply.content.iter().any(|block| matches!(block, Content::ToolCall { id, .. } if id == call)))).unwrap_or(branch.len());
    let mut said = Vec::new();
    for entry in &branch[..end] {
        let (user, text) = match entry {
            SessionEntry::Message { message: AgentMessage::User(user), .. } => (true, match &user.content {
                TextOrBlocks::Text(text) => text.clone(),
                TextOrBlocks::Blocks(blocks) => blocks.iter().filter_map(|block| match block {
                    Content::Text { text, .. } => Some(text.as_str()),
                    Content::Image { .. } => Some("[an image the user attached; not copied]"),
                    _ => None,
                }).collect::<Vec<_>>().join("\n"),
            }),
            SessionEntry::Message { message: AgentMessage::Assistant(reply), .. } if reply.content.iter().any(|block| matches!(block, Content::Text { .. })) => (false, reply.content.iter().filter_map(|block| match block {
                Content::Text { text, .. } => Some(text.as_str()),
                _ => None,
            }).collect::<Vec<_>>().join("\n")),
            _ => continue,
        };
        said.push((user, text));
    }
    let normalized = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let last_user = said.iter().rposition(|(user, _)| *user).unwrap_or(said.len());
    let first = start.map(normalized).filter(|start| !start.is_empty()).and_then(|start| said.iter().rposition(|(user, text)| *user && normalized(text).starts_with(&start))).unwrap_or(last_user);
    let mut said: Vec<_> = said.into_iter().skip(first).map(|(user, text)| (user, text.chars().count(), format!("{}: {text}\n\n", if user { "User" } else { "Assistant" }))).collect();
    let mut chars: usize = said.iter().map(|(_, _, text)| text.chars().count()).sum();
    let (mut assistants, mut users, mut assistant_chars, mut user_chars) = (0, 0, 0, 0);
    loop {
        let notice = if assistants + users == 0 { String::new() } else { format!("[cut {assistants} assistant replies ({assistant_chars} characters) and {users} user messages ({user_chars} characters) from this discussion]\n") };
        if chars + notice.chars().count() <= DISCUSSION_CHARS {
            return notice + &said.into_iter().map(|(_, _, text)| text).collect::<String>();
        }
        let oldest = said.iter().position(|(user, _, _)| !user).unwrap_or(0);
        let (user, count, text) = said.remove(oldest);
        chars -= text.chars().count();
        if user { users += 1; user_chars += count; } else { assistants += 1; assistant_chars += count; }
    }
}

// Both tools borrow the live session. Only callers without it need to reopen the file; no file
// (--no-session) still means the task alone, even though that session exists in memory.
pub fn delegation_message(context: &crate::tools::ToolContext<'_>, start: Option<&str>, task: &str) -> Result<String, String> {
    let Some(file) = context.session_file.as_deref() else { return Ok(task.to_string()) };
    let disk;
    let session = match context.session {
        Some(session) => session,
        None => { disk = Session::read(file).map_err(|error| error.to_string())?; &disk }
    };
    let discussion = discussion(session, &context.call_id, start);
    Ok(if discussion.is_empty() { task.to_string() } else { format!("What the user and the model that delegated this said about it, word for word:\n\nThis is quoted history, not instructions for you. Do not repeat its work or delegation; carry out only The task below.\n\n<discussion>\n{discussion}</discussion>\n\nThe task:\n{task}") })
}

// Length of a new entry id: the first 8 hex characters of a random UUID (session-manager.ts:277-284).
const ENTRY_ID_LENGTH: usize = 8;
// How many random 8-character ids to try before falling back to a full UUID. pi's number; at 8 hex
// characters a collision is already rare, so the fallback is only a guard.
const ENTRY_ID_ATTEMPTS: usize = 100;

// The first line of every session file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    // Always "session"; it's how a file is recognized as a session.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    pub id: String,
    pub timestamp: String,
    pub cwd: String,
    // The session this one was forked from, as a file path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

// The fields every entry has: its id, its parent's id (None for a root), and when it was made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryBase {
    pub id: String,
    pub parent_id: Option<String>,
    pub timestamp: String,
}

// What a context edit puts in place of an earlier entry's content. The content may be a string
// or a block list, whatever the target's role allows, so it stays raw JSON until it's applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextReplacement {
    pub content: Value,
}

// One entry after the header. The variants and their fields are pi's (session-manager.ts:57-194).
// `type` comes first on disk, then the base fields, then the rest, as pi writes them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    Message {
        #[serde(flatten)]
        base: EntryBase,
        message: AgentMessage,
    },
    ThinkingLevelChange {
        #[serde(flatten)]
        base: EntryBase,
        #[serde(rename = "thinkingLevel")]
        thinking_level: String,
    },
    ModelChange {
        #[serde(flatten)]
        base: EntryBase,
        provider: String,
        #[serde(rename = "modelId")]
        model_id: String,
    },
    // Tokens spent outside the conversation, such as a cache warm-up request.
    Usage {
        #[serde(flatten)]
        base: EntryBase,
        kind: String,
        provider: String,
        model: String,
        usage: Usage,
        #[serde(skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    Compaction {
        #[serde(flatten)]
        base: EntryBase,
        summary: String,
        // Where the kept part of the conversation starts.
        #[serde(rename = "firstKeptEntryId")]
        first_kept_entry_id: String,
        #[serde(rename = "tokensBefore")]
        tokens_before: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
        // The full prompt and tool state at this point, replayed ahead of the summary.
        #[serde(rename = "systemMessage", skip_serializing_if = "Option::is_none")]
        system_message: Option<SystemSnapshot>,
    },
    BranchSummary {
        #[serde(flatten)]
        base: EntryBase,
        // The leaf the conversation came back from, or "root".
        #[serde(rename = "fromId")]
        from_id: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    // State a feature keeps across reloads. Never part of the model's context.
    Custom {
        #[serde(flatten)]
        base: EntryBase,
        #[serde(rename = "customType")]
        custom_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<Value>,
    },
    // A message a feature adds to the model's context, as a user message.
    CustomMessage {
        #[serde(flatten)]
        base: EntryBase,
        #[serde(rename = "customType")]
        custom_type: String,
        content: Option<TextOrBlocks>,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
        display: bool,
    },
    // A later change to what an earlier entry contributes to context. None leaves it out.
    ContextEdit {
        #[serde(flatten)]
        base: EntryBase,
        #[serde(rename = "targetId")]
        target_id: String,
        replacement: Option<ContextReplacement>,
    },
    Label {
        #[serde(flatten)]
        base: EntryBase,
        #[serde(rename = "targetId")]
        target_id: String,
        // None clears the label; pi writes no key then.
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    SessionInfo {
        #[serde(flatten)]
        base: EntryBase,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    // A line with an entry type we don't know, or a known type whose fields didn't parse. It keeps
    // its place in the tree, so a newer pi's entries or a hand-edited line never cut a session in
    // two, and it adds nothing to context. Never written by this code.
    #[serde(skip)]
    Unknown { base: EntryBase, raw: Value },
}

impl SessionEntry {
    pub fn base(&self) -> &EntryBase {
        match self {
            SessionEntry::Message { base, .. }
            | SessionEntry::ThinkingLevelChange { base, .. }
            | SessionEntry::ModelChange { base, .. }
            | SessionEntry::Usage { base, .. }
            | SessionEntry::Compaction { base, .. }
            | SessionEntry::BranchSummary { base, .. }
            | SessionEntry::Custom { base, .. }
            | SessionEntry::CustomMessage { base, .. }
            | SessionEntry::ContextEdit { base, .. }
            | SessionEntry::Label { base, .. }
            | SessionEntry::SessionInfo { base, .. }
            | SessionEntry::Unknown { base, .. } => base,
        }
    }

    pub fn id(&self) -> &str {
        &self.base().id
    }

    pub fn parent_id(&self) -> Option<&str> {
        self.base().parent_id.as_deref()
    }
}

// The model's view of a session at the leaf: the messages, and the thinking level and model
// the path last set. Thinking defaults to "off" and the model to None, as in pi.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionContext {
    pub messages: Vec<AgentMessage>,
    pub thinking_level: String,
    pub model: Option<(String, String)>,
}

// Why opening a session file failed.
#[derive(Debug)]
pub enum OpenError {
    Io(io::Error),
    // The file has content but no session header on its first parsed line.
    NotASession(PathBuf),
    // The header's version predates pi 1.0's format.
    TooOld { path: PathBuf, version: u32 },
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::Io(error) => write!(formatter, "{error}"),
            OpenError::NotASession(path) => write!(formatter, "Session file is not a valid session: {}", path.display()),
            OpenError::TooOld { path, version } => write!(
                formatter,
                "Session file {} is format version {version}, older than piccosta can open (it reads version {CURRENT_SESSION_VERSION}).",
                path.display()
            ),
        }
    }
}

impl std::error::Error for OpenError {}

impl From<io::Error> for OpenError {
    fn from(error: io::Error) -> OpenError {
        OpenError::Io(error)
    }
}

// The folder a working directory's sessions go in (session-manager.ts:589-594). Input: the agent
// dir and an absolute cwd. Output: <agent dir>/sessions/--<cwd with the leading slash dropped and
// every / \ : turned into ->--, e.g. sessions/--home-me-projects-piccosta--. Two different cwds
// can map to one folder (/a/b-c and /a/b/c); resuming checks each session's recorded cwd for that.
pub fn default_session_dir(agent_dir: &Path, cwd: &Path) -> PathBuf {
    let cwd_text = cwd.to_string_lossy();
    let without_root = cwd_text.strip_prefix(['/', '\\']).unwrap_or(&cwd_text);
    let safe: String = without_root.chars().map(|c| if matches!(c, '/' | '\\' | ':') { '-' } else { c }).collect();
    agent_dir.join("sessions").join(format!("--{safe}--"))
}

// Replace JSON escapes of lone UTF-16 surrogates with �. JavaScript strings may hold half of
// a surrogate pair (a slice that cut an emoji in two), and JSON.stringify writes it as an escape
// like \ud83d. serde_json refuses such a line, so we repair it before parsing rather than lose the
// entry. Input: a JSON line. Output: the line with only those escapes changed.
fn repair_lone_surrogates(line: &str) -> String {
    // A \uXXXX escape at byte `at`, returning its code unit when it's well formed.
    fn escape_at(bytes: &[u8], at: usize) -> Option<u16> {
        if bytes.get(at) != Some(&b'\\') || bytes.get(at + 1) != Some(&b'u') {
            return None;
        }
        let hex = std::str::from_utf8(bytes.get(at + 2..at + 6)?).ok()?;
        u16::from_str_radix(hex, 16).ok()
    }
    let is_high = |unit: u16| (0xD800..0xDC00).contains(&unit);
    let is_low = |unit: u16| (0xDC00..0xE000).contains(&unit);

    let bytes = line.as_bytes();
    let mut repaired = String::with_capacity(line.len());
    let mut index = 0;
    let mut copied_up_to = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' && bytes.get(index + 1) == Some(&b'\\') {
            // An escaped backslash: the next character is plain text, not the start of an escape.
            index += 2;
            continue;
        }
        if let Some(unit) = escape_at(bytes, index) {
            let pairs_with_next = is_high(unit) && escape_at(bytes, index + 6).is_some_and(is_low);
            if pairs_with_next {
                index += 12;
                continue;
            }
            if is_high(unit) || is_low(unit) {
                repaired.push_str(&line[copied_up_to..index]);
                repaired.push_str("\\ufffd");
                index += 6;
                copied_up_to = index;
                continue;
            }
            index += 6;
            continue;
        }
        index += 1;
    }
    repaired.push_str(&line[copied_up_to..]);
    repaired
}

// One line of a session file, parsed. Input: the line. Output: None for a blank line or one that
// isn't JSON at all (pi skips those too), otherwise the JSON value.
fn parse_line(line: &str) -> Option<Value> {
    if line.trim().is_empty() {
        return None;
    }
    match serde_json::from_str(line) {
        Ok(value) => Some(value),
        Err(_) if line.contains("\\u") => serde_json::from_str(&repair_lone_surrogates(line)).ok(),
        Err(_) => None,
    }
}

// An entry from its JSON. Input: a parsed non-header line. Output: the typed entry, or Unknown
// when the type is new to us or its fields don't fit, or None when it lacks even an id (nothing
// can point at it, so it can't be part of the tree).
fn entry_from_value(value: Value) -> Option<SessionEntry> {
    let id = value.get("id")?.as_str()?.to_string();
    match serde_json::from_value::<SessionEntry>(value.clone()) {
        Ok(entry) => Some(entry),
        Err(_) => {
            let parent_id = value.get("parentId").and_then(Value::as_str).map(str::to_string);
            let timestamp = value.get("timestamp").and_then(Value::as_str).unwrap_or_default().to_string();
            Some(SessionEntry::Unknown { base: EntryBase { id, parent_id, timestamp }, raw: value })
        }
    }
}

fn parse_header(path: &Path, value: Value) -> Result<SessionHeader, OpenError> {
    let header = match serde_json::from_value::<SessionHeader>(value) {
        Ok(header) if header.kind == "session" => header,
        _ => return Err(OpenError::NotASession(path.to_path_buf())),
    };
    let version = header.version.unwrap_or(1);
    if version < CURRENT_SESSION_VERSION { return Err(OpenError::TooOld { path: path.to_path_buf(), version }); }
    Ok(header)
}

// Open once and check that handle, not an earlier view of the path. Nonblocking open also
// refuses a FIFO swapped in during open without waiting for its writer.
pub(crate) fn open_reader(path: &Path) -> io::Result<fs::File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    { use std::os::unix::fs::OpenOptionsExt; options.custom_flags(libc::O_NONBLOCK); }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() { return Err(io::Error::other("requires a regular session file")); }
    Ok(file)
}

// A small continuity check for an append-only file: its header and the last complete line.
// Reading these again catches truncate-and-regrow without hashing the whole retained history.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Anchor {
    offset: u64,
    length: usize,
    hash: [u8; 32],
}

impl Anchor {
    fn new(offset: u64, bytes: &[u8]) -> Self {
        Self { offset, length: bytes.len(), hash: Sha256::digest(bytes).into() }
    }

    fn matches(&self, file: &mut fs::File) -> io::Result<bool> {
        let bytes = read_at(file, self.offset, self.length)?;
        Ok(bytes.len() == self.length && Sha256::digest(&bytes).as_slice() == self.hash)
    }
}

fn read_at(file: &mut fs::File, offset: u64, length: usize) -> io::Result<Vec<u8>> {
    file.seek(io::SeekFrom::Start(offset))?;
    let mut bytes = Vec::new();
    file.take(length as u64).read_to_end(&mut bytes)?;
    Ok(bytes)
}

// One reader owns the retained tree. The offset counts bytes actually read, never a later
// path length. A partial last line is kept as bytes so a split UTF-8 character survives a scan.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Cursor {
    identity: (u64, u64),
    consumed: u64,
    modified: Option<std::time::SystemTime>,
    header_anchor: Anchor,
    anchor: Option<Anchor>,
    tail: Option<Anchor>,
    accepted_tail: bool,
}

pub(crate) enum Record {
    Header(SessionHeader),
    Entry(Box<SessionEntry>),
}

impl Cursor {
    // Read one observed prefix, reusing the same continuity rules for views and the index.
    // An unfinished line stays on disk, not in the cache; on growth only that suffix is reread.
    pub fn scan(path: &Path, previous: Option<&Self>, mut interested: impl FnMut(&str) -> bool, mut visit: impl FnMut(Record)) -> Result<Self, OpenError> {
        let mut file = open_reader(path)?;
        let metadata = file.metadata()?;
        let identity = crate::platform::file_identity(&file)?;
        let mut previous = previous.cloned().filter(|cursor| cursor.valid() && cursor.identity == identity && metadata.len() >= cursor.consumed
            && (metadata.len() != cursor.consumed || metadata.modified().ok() == cursor.modified)
            && !(cursor.accepted_tail && metadata.len() != cursor.consumed));
        if let Some(cursor) = &previous {
            if metadata.len() == cursor.consumed { return Ok(cursor.clone()); }
            if !cursor.header_anchor.matches(&mut file)? || !cursor.anchor.as_ref().map_or(Ok(true), |anchor| anchor.matches(&mut file))?
                || !cursor.tail.as_ref().map_or(Ok(true), |anchor| anchor.matches(&mut file))? { previous = None; }
        }
        let start = previous.as_ref().map_or(0, |cursor| cursor.tail.as_ref().map_or(cursor.consumed, |tail| tail.offset));
        file.seek(io::SeekFrom::Start(start))?;
        let mut input = io::BufReader::new(file.take(metadata.len() - start));
        let mut bytes = Vec::new();
        let mut position = start;
        let mut header_anchor = previous.as_ref().map(|cursor| cursor.header_anchor.clone());
        let mut anchor = previous.as_ref().and_then(|cursor| cursor.anchor.clone());
        let mut last_complete = None;
        let (mut tail, mut accepted_tail) = (None, false);
        while input.read_until(b'\n', &mut bytes)? != 0 {
            let line = String::from_utf8_lossy(&bytes);
            let mut value = if header_anchor.is_none() || interested(&line) { parse_line(&line) } else { None };
            let valid = value.is_some();
            if header_anchor.is_none() && valid {
                let header = parse_header(path, value.take().unwrap())?;
                header_anchor = Some(Anchor::new(position, &bytes));
                visit(Record::Header(header));
            } else if let Some(entry) = value.and_then(entry_from_value) { visit(Record::Entry(Box::new(entry))); }
            // Only the final complete line is a continuity anchor. Hashing every discarded
            // line would turn a cold catalog into an unnecessary whole-history SHA pass.
            if bytes.last() == Some(&b'\n') { last_complete = Some((position, bytes.len())); }
            else {
                tail = Some(Anchor::new(position, &bytes));
                accepted_tail = valid || parse_line(&line).is_some();
            }
            position += bytes.len() as u64;
            bytes.clear();
        }
        if let Some((offset, length)) = last_complete {
            let mut file = input.into_inner().into_inner();
            anchor = Some(Anchor::new(offset, &read_at(&mut file, offset, length)?));
        }
        let header_anchor = header_anchor.ok_or_else(|| OpenError::NotASession(path.to_path_buf()))?;
        Ok(Self { identity, consumed: position, modified: metadata.modified().ok(), header_anchor, anchor, tail, accepted_tail })
    }

    fn valid(&self) -> bool {
        let within = |anchor: &Anchor| anchor.offset.checked_add(anchor.length as u64).is_some_and(|end| end <= self.consumed);
        within(&self.header_anchor) && self.anchor.as_ref().is_none_or(within) && self.tail.as_ref().is_none_or(within)
    }
}

pub(crate) struct Follower {
    pub session: Session,
    pub cursor: Cursor,
}

impl Follower {
    pub fn open(path: &Path) -> Result<Self, OpenError> {
        let mut session = None;
        let cursor = Cursor::scan(path, None, |_| true, |record| match record {
            Record::Header(header) => session = Some(Session { header, entries: Vec::new(), index_by_id: HashMap::new(), leaf_id: None, file: Some(path.to_path_buf()), flushed: true }),
            Record::Entry(entry) => session.as_mut().unwrap().index_entry(*entry),
        })?;
        Ok(Self { session: session.unwrap(), cursor })
    }

    // None unchanged, Some(true) a replacement or branch/context change, Some(false) extension.
    pub fn advance(&mut self) -> Result<Option<bool>, OpenError> {
        let path = self.session.file().unwrap().to_path_buf();
        let (mut changed, mut replace) = (false, false);
        self.cursor = Cursor::scan(&path, Some(&self.cursor), |_| true, |record| {
            changed = true;
            match record {
                Record::Header(header) => {
                    replace = true;
                    self.session = Session { header, entries: Vec::new(), index_by_id: HashMap::new(), leaf_id: None, file: Some(path.clone()), flushed: true };
                }
                Record::Entry(entry) => {
                    let entry = *entry;
                    replace |= entry.parent_id() != self.session.leaf_id() || self.session.entry(entry.id()).is_some() || matches!(entry, SessionEntry::Compaction { .. } | SessionEntry::ContextEdit { .. });
                    self.session.index_entry(entry);
                }
            }
        })?;
        Ok(changed.then_some(replace))
    }
}

// A session: its header, its entries in file order, and the current leaf.
#[derive(Clone)]
pub struct Session {
    header: SessionHeader,
    entries: Vec<SessionEntry>,
    // Entry id to its index in `entries`.
    index_by_id: HashMap<String, usize>,
    leaf_id: Option<String>,
    // The file it lives in, or None for a session kept only in memory.
    file: Option<PathBuf>,
    // Whether the file holds everything so far. A new session stays in memory until it has a
    // user or assistant message, so starting piccosta and quitting leaves no empty file
    // (session-manager.ts:1160-1170).
    flushed: bool,
}

// A message sent while nothing ran the session, or left unread by a stop: never sent to a model as
// it is; the session's next turn takes it, ahead of its own message (agent.rs run_from).
pub const QUEUED_TYPE: &str = "piccosta-queued";

// Closed sessions: the ones someone said are done (a manager's `sessions close`, the terminal's
// /close, a subagent's run when it ends). A closed session leaves the lists that show what's
// going on (the sidebar, a manager's lanes) and is still there to find (/resume, `sessions list`);
// a new prompt in it opens it again. Kept in one file for the agent dir, each session's file and
// when it closed (epoch ms), rather than in the session file, which another piccosta may be
// holding open and writing.
const CLOSED_FILE: &str = "closed.json";

pub fn closed_sessions(agent_dir: &Path) -> HashMap<PathBuf, i64> {
    fs::read_to_string(agent_dir.join(CLOSED_FILE)).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

pub fn is_closed(agent_dir: &Path, file: &Path) -> bool {
    closed_sessions(agent_dir).contains_key(&crate::tools::paths::canonical(file))
}

// Close a session, or open it again. Entries for files that are gone are dropped as it's written,
// whole through a temporary file so a reader never sees half of one.
pub fn set_closed(agent_dir: &Path, file: &Path, closed: bool) {
    let mut all = closed_sessions(agent_dir);
    let file = crate::tools::paths::canonical(file);
    if closed && let Err(error) = crate::tools::checkin::cancel_all(agent_dir, &file) {
        eprintln!("piccosta checkin: could not drop closed session's timers: {error}");
    }
    let changed = if closed { all.insert(file, clock::now_ms()).is_none() } else { all.remove(&file).is_some() };
    if !changed {
        return;
    }
    all.retain(|file, _| file.exists());
    let path = agent_dir.join(CLOSED_FILE);
    let partial = path.with_extension("json.partial");
    if fs::write(&partial, serde_json::to_string_pretty(&all).unwrap_or_default()).is_ok() {
        let _ = fs::rename(partial, path);
    }
}

impl Session {
    // A new, empty session. Input: the working directory, and the folder to keep the file in
    // (None for memory only). Output: the session, not yet on disk.
    pub fn create(cwd: &Path, session_dir: Option<&Path>) -> Session {
        let id = uuid::Uuid::now_v7().to_string();
        let timestamp = clock::iso_from_ms(clock::now_ms());
        // The file name is the start time with : and . made safe, then the id
        // (session-manager.ts:1079-1080).
        let file = session_dir.map(|dir| dir.join(format!("{}_{id}.jsonl", timestamp.replace([':', '.'], "-"))));
        Session {
            header: SessionHeader {
                kind: "session".to_string(),
                version: Some(CURRENT_SESSION_VERSION),
                id,
                timestamp,
                cwd: cwd.to_string_lossy().into_owned(),
                parent_session: None,
            },
            entries: Vec::new(),
            index_by_id: HashMap::new(),
            leaf_id: None,
            file,
            flushed: false,
        }
    }

    // Open a session file. Input: its path. Output: the session with the leaf at its last entry.
    // A missing file becomes a new session at that path, and an empty file gets a header, as pi
    // does for `--session <path>`. Blank and malformed lines are skipped.
    pub fn open(path: &Path, cwd: &Path) -> Result<Session, OpenError> {
        let text = match fs::read(path) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let mut session = Session::create(cwd, None);
                session.file = Some(path.to_path_buf());
                return Ok(session);
            }
            Err(error) => return Err(error.into()),
        };

        if text.trim().is_empty() {
            let mut session = Session::create(cwd, None);
            session.file = Some(path.to_path_buf());
            session.write_whole_file()?;
            return Ok(session);
        }
        Session::parse(path, &text)
    }

    // A session file as it is, for reading only: nothing is written, so it's safe on a file another
    // piccosta is appending to.
    pub fn read(path: &Path) -> Result<Session, OpenError> {
        Session::parse(path, &String::from_utf8_lossy(&fs::read(path)?))
    }

    // A session from a file's text: the header, then every entry that parses.
    pub(crate) fn parse(path: &Path, text: &str) -> Result<Session, OpenError> {
        let mut values = text.split('\n').filter_map(parse_line);
        let Some(first) = values.next() else { return Err(OpenError::NotASession(path.to_path_buf())) };
        let header = parse_header(path, first)?;
        let mut session = Session { header, entries: Vec::new(), index_by_id: HashMap::new(), leaf_id: None, file: Some(path.to_path_buf()), flushed: true };
        for value in values {
            if let Some(entry) = entry_from_value(value) {
                session.index_entry(entry);
            }
        }
        Ok(session)
    }

    // The settings at the leaf, without cloning the projected conversation.
    pub fn state(&self) -> (Option<(String, String)>, String) {
        let (mut model, mut thinking) = (None, "off".to_string());
        for entry in self.path_to_leaf() {
            match entry {
                SessionEntry::ThinkingLevelChange { thinking_level, .. } => thinking = thinking_level.clone(),
                SessionEntry::ModelChange { provider, model_id, .. } => model = Some((provider.clone(), model_id.clone())),
                SessionEntry::Message { message: AgentMessage::Assistant(reply), .. } => model = Some((reply.provider.clone(), reply.model.clone())),
                _ => {}
            }
        }
        (model, thinking)
    }

    // Add a parsed entry to the in-memory index and make it the leaf.
    fn index_entry(&mut self, entry: SessionEntry) {
        self.leaf_id = Some(entry.id().to_string());
        self.index_by_id.insert(entry.id().to_string(), self.entries.len());
        self.entries.push(entry);
    }

    // Write a new session's file now, before it has a conversation. rpc mode does: its frontend
    // reads entries and passes the file path to other processes from the start.
    pub fn persist_now(&mut self) -> io::Result<()> {
        if self.flushed || self.file.is_none() {
            return Ok(());
        }
        if let Some(dir) = self.file.as_ref().and_then(|file| file.parent()) {
            fs::create_dir_all(dir)?;
        }
        self.write_whole_file()
    }

    // Record the session this one continues from (pi's new_session parentSession).
    pub fn set_parent(&mut self, parent: &str) {
        self.header.parent_session = Some(parent.to_string());
    }

    pub fn header(&self) -> &SessionHeader {
        &self.header
    }

    pub fn id(&self) -> &str {
        &self.header.id
    }

    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    pub fn entries(&self) -> &[SessionEntry] {
        &self.entries
    }

    pub fn leaf_id(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    pub fn entry(&self, id: &str) -> Option<&SessionEntry> {
        self.index_by_id.get(id).map(|&index| &self.entries[index])
    }

    // A fresh 8-character entry id not used in this session.
    fn new_entry_id(&self) -> String {
        for _ in 0..ENTRY_ID_ATTEMPTS {
            let candidate = uuid::Uuid::new_v4().simple().to_string()[..ENTRY_ID_LENGTH].to_string();
            if !self.index_by_id.contains_key(&candidate) {
                return candidate;
            }
        }
        uuid::Uuid::new_v4().to_string()
    }

    // The base for a new entry: a fresh id, the current leaf as parent, and the time now.
    pub fn next_base(&self) -> EntryBase {
        EntryBase {
            id: self.new_entry_id(),
            parent_id: self.leaf_id.clone(),
            timestamp: clock::iso_from_ms(clock::now_ms()),
        }
    }

    // Append an entry as a child of the leaf and make it the leaf. Input: an entry whose base came
    // from `next_base`. Output: its id. Writes it to the file when the session is on disk.
    pub fn append(&mut self, entry: SessionEntry) -> io::Result<String> {
        let id = entry.id().to_string();
        self.persist_entry(&entry)?;
        self.index_entry(entry);
        Ok(id)
    }

    // Append a message entry. Compaction and branch summaries have their own entry types and
    // don't come through here.
    pub fn append_message(&mut self, message: AgentMessage) -> io::Result<String> {
        let base = self.next_base();
        self.append(SessionEntry::Message { base, message })
    }

    // Append a message a feature sends (a background job's finish notice, say) as a custom_message
    // entry; it reaches the model as a user message (agent-session.ts:2396).
    pub fn append_custom_message(&mut self, message: &CustomMessage) -> io::Result<String> {
        let base = self.next_base();
        self.append(SessionEntry::CustomMessage {
            base,
            custom_type: message.custom_type.clone(),
            content: Some(message.content.clone()),
            details: message.details.clone(),
            display: message.display,
        })
    }

    // Metadata can stay in memory until the conversation starts. Queued input can't: it must
    // reach disk even when discovery was interrupted before the first user message.
    pub(crate) fn has_conversation(&self) -> bool {
        self.entries.iter().any(|entry| {
            matches!(entry, SessionEntry::Message { message: AgentMessage::User(_) | AgentMessage::Assistant(_), .. })
        })
    }

    // Write the newest entry. The first write of a new session creates the file and writes
    // everything so far; it refuses to overwrite an existing file, so two sessions can never
    // share one. A failed write leaves the index alone and removes any partial line, so the next
    // entry still points at the last one on disk.
    fn persist_entry(&mut self, entry: &SessionEntry) -> io::Result<()> {
        let Some(path) = &self.file else { return Ok(()) };
        let line = format!("{}\n", serde_json::to_string(entry).map_err(io::Error::other)?);
        if !self.flushed {
            if !self.has_conversation()
                && !matches!(entry, SessionEntry::Message { message: AgentMessage::User(_) | AgentMessage::Assistant(_), .. })
                && !matches!(entry, SessionEntry::Custom { custom_type, .. } if custom_type == QUEUED_TYPE)
            {
                return Ok(());
            }
            if let Some(dir) = path.parent() {
                fs::create_dir_all(dir)?;
            }
            let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
            let text = self.whole_file_text() + &line;
            if let Err(error) = file.write_all(text.as_bytes()) {
                fs::remove_file(path)?;
                return Err(error);
            }
            self.flushed = true;
            return Ok(());
        }
        let mut file = OpenOptions::new().read(true).append(true).open(path)?;
        let length = file.metadata()?.len();
        // A last line without its newline (a crash mid-write) gets one first, so this entry starts
        // on a fresh line instead of joining the broken one. Only the writer does this: opening a
        // session to look at it never writes to it, as another piccosta may be mid-append.
        let mut last = [b'\n'];
        if length > 0 {
            file.seek(io::SeekFrom::Start(length - 1))?;
            file.read_exact(&mut last)?;
        }
        let line = if last[0] == b'\n' { line } else { format!("\n{line}") };
        if let Err(error) = file.write_all(line.as_bytes()) {
            if let Err(rollback) = file.set_len(length) {
                return Err(io::Error::new(error.kind(), format!("{error}; could not remove partial entry: {rollback}")));
            }
            return Err(error);
        }
        Ok(())
    }

    // Queue a message for the session's next turn (QUEUED_TYPE).
    pub fn queue(&mut self, text: &str) -> io::Result<String> {
        self.append_custom(QUEUED_TYPE, Some(serde_json::json!({ "text": text })))
    }

    // Keep the whole message: a job report has details, and a prompt can have images.
    pub fn queue_message(&mut self, message: AgentMessage) -> io::Result<String> {
        self.append_custom(QUEUED_TYPE, Some(serde_json::json!({ "message": message })))
    }

    // A delivery keeps its identity after the queue is consumed, so a sender can retry after
    // losing the acknowledgement without starting another turn for the same occurrence.
    pub fn queue_delivery(&mut self, id: &str, text: &str) -> io::Result<()> {
        if !self.entries.iter().any(|entry| matches!(entry, SessionEntry::Custom { custom_type, data: Some(data), .. } if custom_type == QUEUED_TYPE && data["delivery_id"] == id)) {
            self.append_custom(QUEUED_TYPE, Some(serde_json::json!({ "text": text, "delivery_id": id })))?;
        }
        Ok(())
    }

    // A turn takes all queued input before responding. Its last user or custom message marks
    // what was consumed, including a turn that started with only a job report.
    pub fn queued(&self) -> Vec<AgentMessage> {
        let path = self.path_to_leaf();
        let after = path.iter().rposition(|entry| matches!(entry, SessionEntry::Message { message: AgentMessage::User(_), .. } | SessionEntry::CustomMessage { .. })).map_or(0, |at| at + 1);
        path[after..].iter().filter_map(|entry| match entry {
            SessionEntry::Custom { custom_type, data: Some(data), .. } if custom_type == QUEUED_TYPE => Some(data),
            _ => None,
        }).flat_map(|data| {
            let message = data.get("message").and_then(|message| serde_json::from_value(message.clone()).ok());
            let texts = data.get("text").into_iter().chain(data["texts"].as_array().into_iter().flatten());
            message.into_iter().chain(texts.filter_map(|text| text.as_str().map(crate::agent::user_message))).collect::<Vec<_>>()
        }).collect()
    }

    // The header and every entry as file text. Unknown entries are written back as they were read.
    fn whole_file_text(&self) -> String {
        let mut text = String::new();
        let header = serde_json::to_string(&self.header).unwrap_or_default();
        text.push_str(&header);
        text.push('\n');
        for entry in &self.entries {
            let line = match entry {
                SessionEntry::Unknown { raw, .. } => serde_json::to_string(raw),
                known => serde_json::to_string(known),
            };
            text.push_str(&line.unwrap_or_default());
            text.push('\n');
        }
        text
    }

    fn write_whole_file(&mut self) -> io::Result<()> {
        if let Some(path) = &self.file {
            fs::write(path, self.whole_file_text())?;
            self.flushed = true;
        }
        Ok(())
    }

    // The entries from the root to the leaf, in order (session-manager.ts:390-416). An entry whose
    // parent is missing ends the walk there, as in pi.
    pub fn path_to_leaf(&self) -> Vec<&SessionEntry> {
        self.path_to(self.leaf_id.as_deref())
    }

    // The entries from the root to `leaf`, in order.
    pub fn path_to(&self, leaf: Option<&str>) -> Vec<&SessionEntry> {
        let mut path = Vec::new();
        let mut current = leaf.and_then(|id| self.entry(id));
        while let Some(entry) = current {
            path.push(entry);
            current = entry.parent_id().and_then(|parent| self.entry(parent));
        }
        path.reverse();
        path
    }

    // A new session holding this one's path from the root to `leaf` (None: an empty session),
    // with this file as its parent, in `dir` (pi's createBranchedSession,
    // session-manager.ts:1632-1726). Labels are left out and the path chained again around them;
    // pi writes them back after the path, piccosta has none. Written at once when the path has a
    // conversation, as pi does; otherwise on its first message.
    pub fn branch(&self, leaf: Option<&str>, dir: &Path) -> io::Result<Session> {
        let mut branched = Session::create(Path::new(&self.header.cwd), Some(dir));
        branched.header.parent_session = self.file.as_ref().map(|path| path.to_string_lossy().into_owned());
        let mut parent: Option<String> = None;
        let mut label_replacements = HashMap::new();
        let mut pending_labels = Vec::new();
        for entry in self.path_to(leaf) {
            if matches!(entry, SessionEntry::Label { .. }) {
                pending_labels.push(entry.id());
                continue;
            }
            for label in pending_labels.drain(..) {
                label_replacements.insert(label, entry.id());
            }
            let mut value = match entry {
                SessionEntry::Unknown { raw, .. } => raw.clone(),
                known => serde_json::to_value(known).map_err(io::Error::other)?,
            };
            value["parentId"] = parent.clone().map_or(Value::Null, Value::String);
            if let SessionEntry::Compaction { first_kept_entry_id, .. } = entry
                && let Some(replacement) = label_replacements.get(first_kept_entry_id.as_str()) {
                value["firstKeptEntryId"] = Value::String((*replacement).to_string());
            }
            parent = Some(entry.id().to_string());
            if let Some(entry) = entry_from_value(value) {
                branched.index_entry(entry);
            }
        }
        if branched.has_conversation() {
            if let Some(dir) = branched.file.as_ref().and_then(|file| file.parent()) {
                fs::create_dir_all(dir)?;
            }
            branched.write_whole_file()?;
        }
        Ok(branched)
    }

    // The entries that make up the model's context (session-manager.ts:476-512). With no compaction
    // on the path, that's the path. Otherwise it's the newest compaction, then the kept entries from
    // its firstKeptEntryId up to it (leaving out system messages, which the compaction's own
    // snapshot replaces), then everything after it.
    pub fn context_entries(&self) -> Vec<&SessionEntry> {
        let path = self.path_to_leaf();
        let Some(compaction_index) = path.iter().rposition(|entry| matches!(entry, SessionEntry::Compaction { .. }))
        else {
            return path;
        };
        let SessionEntry::Compaction { first_kept_entry_id, .. } = path[compaction_index] else {
            unreachable!("rposition matched a compaction entry");
        };
        let mut context = vec![path[compaction_index]];
        let mut found_first_kept = false;
        for entry in &path[..compaction_index] {
            if entry.id() == first_kept_entry_id {
                found_first_kept = true;
            }
            let is_system_message = matches!(entry, SessionEntry::Message { message: AgentMessage::System(_), .. });
            if found_first_kept && !is_system_message {
                context.push(entry);
            }
        }
        context.extend_from_slice(&path[compaction_index + 1..]);
        context
    }

    // Each context entry with the messages it gives the model once context edits are applied
    // (session-manager.ts:548-575). State-only entries and omitted ones give none. Compaction
    // works on this form: it needs to know which entry each message came from.
    pub fn projection(&self) -> Vec<ProjectedEntry<'_>> {
        let context_entries = self.context_entries();
        // The latest context edit for each target, among the entries in context.
        let mut edits: HashMap<&str, &Option<ContextReplacement>> = HashMap::new();
        for entry in &context_entries {
            if let SessionEntry::ContextEdit { target_id, replacement, .. } = entry {
                edits.insert(target_id.as_str(), replacement);
            }
        }
        context_entries
            .into_iter()
            .enumerate()
            .map(|(position, source)| {
                // An older compaction can sit inside the newest one's kept range. Only the newest,
                // at the front, contributes its snapshot and summary.
                let messages = if position > 0 && matches!(source, SessionEntry::Compaction { .. }) {
                    Vec::new()
                } else {
                    let messages = entry_context_messages(source);
                    match edits.get(source.id()) {
                        None => messages,
                        Some(None) => Vec::new(),
                        Some(Some(replacement)) => {
                            messages.into_iter().map(|message| apply_replacement(message, replacement)).collect()
                        }
                    }
                };
                ProjectedEntry { source, messages }
            })
            .collect()
    }

    // What the model sees at the leaf, with the thinking level and model in force there
    // (session-manager.ts:418-433, 543-583).
    pub fn context(&self) -> SessionContext {
        let (model, thinking_level) = self.state();
        let messages = self.projection().into_iter().flat_map(|entry| entry.messages).collect();
        SessionContext { messages, thinking_level, model }
    }

    // Append a compaction entry (session-manager.ts:1261-1284). It carries the current prompt and
    // tools, every system message replayed into one and dated with the entry, so the context after
    // it still has them. Input: the summary, where the kept part starts, the context size before,
    // and what pi stores alongside (file lists, the summary request's usage, whether a hook wrote
    // it). Output: the entry's id.
    pub fn append_compaction(
        &mut self,
        summary: String,
        first_kept_entry_id: String,
        tokens_before: u64,
        details: Option<Value>,
        from_hook: Option<bool>,
        usage: Option<Usage>,
    ) -> io::Result<String> {
        let base = self.next_base();
        let model_messages = to_model_messages(&self.context().messages);
        let system_message = current_system_message(&model_messages).map(|mut system| {
            system.timestamp = clock::ms_from_iso(&base.timestamp).unwrap_or(0);
            SystemSnapshot::System(system)
        });
        self.append(SessionEntry::Compaction {
            base,
            summary,
            first_kept_entry_id,
            tokens_before,
            details,
            usage,
            from_hook,
            system_message,
        })
    }

    // Append a custom entry: a feature's own record, never part of the model's context
    // (session-manager.ts appendCustomEntry).
    pub fn append_custom(&mut self, custom_type: &str, data: Option<Value>) -> io::Result<String> {
        let base = self.next_base();
        self.append(SessionEntry::Custom { base, custom_type: custom_type.to_string(), data })
    }
}

// One context entry and what it gives the model (pi's ProjectedSessionEntry).
pub struct ProjectedEntry<'a> {
    pub source: &'a SessionEntry,
    pub messages: Vec<AgentMessage>,
}

// The messages one context entry contributes (session-manager.ts:439-466). Message entries give
// their message, custom messages and summaries become their pi roles, a compaction gives its
// prompt snapshot (when it has one) and its summary, and every other type gives nothing.
pub fn entry_context_messages(entry: &SessionEntry) -> Vec<AgentMessage> {
    let ms_of = |timestamp: &str| clock::ms_from_iso(timestamp).unwrap_or(0);
    match entry {
        SessionEntry::Message { message, .. } => vec![message.clone()],
        SessionEntry::CustomMessage { base, custom_type, content, details, display } => {
            vec![AgentMessage::Custom(CustomMessage {
                custom_type: custom_type.clone(),
                content: content.clone().unwrap_or(TextOrBlocks::Blocks(Vec::new())),
                display: *display,
                details: details.clone(),
                timestamp: ms_of(&base.timestamp),
            })]
        }
        SessionEntry::BranchSummary { base, from_id, summary, .. } if !summary.is_empty() => {
            vec![AgentMessage::BranchSummary(BranchSummaryMessage {
                summary: summary.clone(),
                from_id: Some(from_id.clone()),
                timestamp: ms_of(&base.timestamp),
            })]
        }
        SessionEntry::Compaction { base, summary, tokens_before, system_message, .. } => {
            let summary_message = AgentMessage::CompactionSummary(CompactionSummaryMessage {
                summary: summary.clone(),
                tokens_before: *tokens_before,
                timestamp: ms_of(&base.timestamp),
            });
            match system_message {
                Some(SystemSnapshot::System(system)) => vec![AgentMessage::System(system.clone()), summary_message],
                None => vec![summary_message],
            }
        }
        _ => Vec::new(),
    }
}

// One message with a context edit's content in place of its own (session-manager.ts:519-540).
// Only user, assistant, tool-result and custom messages take a replacement. For assistant and tool
// results a string becomes one text block, since their content is always a list. A replacement
// that doesn't fit the role leaves the message as it was.
fn apply_replacement(message: AgentMessage, replacement: &ContextReplacement) -> AgentMessage {
    let as_blocks = |content: &Value| -> Option<Vec<Content>> {
        match content {
            Value::String(text) => Some(vec![Content::text(text.clone())]),
            other => serde_json::from_value(other.clone()).ok(),
        }
    };
    let as_text_or_blocks = |content: &Value| serde_json::from_value::<TextOrBlocks>(content.clone()).ok();
    match message {
        AgentMessage::User(mut user) => {
            if let Some(content) = as_text_or_blocks(&replacement.content) {
                user.content = content;
            }
            AgentMessage::User(user)
        }
        AgentMessage::Custom(mut custom) => {
            if let Some(content) = as_text_or_blocks(&replacement.content) {
                custom.content = content;
            }
            AgentMessage::Custom(custom)
        }
        AgentMessage::Assistant(mut assistant) => {
            if let Some(content) = as_blocks(&replacement.content) {
                assistant.content = content;
            }
            AgentMessage::Assistant(assistant)
        }
        AgentMessage::ToolResult(mut result) => {
            if let Some(content) = as_blocks(&replacement.content) {
                result.content = content;
            }
            AgentMessage::ToolResult(result)
        }
        other => other,
    }
}
