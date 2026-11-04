// The cache monitor. After each response piccosta compares what the provider read from its prompt
// cache with the context the turn before had; a turn that had to write most of it again is a miss,
// recorded in the session with the cause it can name, and shown as one line. `piccosta cache`
// lists the recorded misses across every session for the last day (or the hours given).

use crate::message::{AgentMessage, AssistantMessage};
use crate::session::{Session, SessionEntry};
use serde_json::{json, Value};
use std::path::Path;

// The custom entry a miss is recorded as.
pub const MISS_TYPE: &str = "piccosta-cache-miss";
// What counts as a miss, as measured on 2026-11-01: a turn that wrote more than 20K tokens while
// reading less than half of the context the turn before had. Smaller writes are a turn's own new
// tokens; a read that big means the cache held up to some point late in the context.
const MISS_MIN_TOKENS: u64 = 20_000;
const MISS_READ_SHARE: f64 = 0.5;
// The window `piccosta cache` looks at when no hours are given.
const DEFAULT_REPORT_HOURS: i64 = 24;

// A token count the short way: 25.3M, 412k, 980.
pub fn tokens(count: u64) -> String {
    match count {
        1_000_000.. => format!("{:.1}M", count as f64 / 1e6),
        1_000.. => format!("{}k", count / 1_000),
        _ => count.to_string(),
    }
}

// The whole context a response was asked with: what it read from cache, wrote to it, and sent
// uncached.
fn context(message: &AssistantMessage) -> u64 {
    message.usage.input + message.usage.cache_read + message.usage.cache_write
}

// Whether the response in entry `id` missed the cache, and why. Input: the session, the entry, and
// the Claude bridge's reason when that call started a new Claude Code query. Output: the miss as
// the custom entry's data (written, read, before, cause, detail), or None when the turn read at
// least half of the context before it, wrote little, or has no turn before. The cause is what
// changed the request's prefix: a compaction since the turn before, another model, a new system
// message; on the Claude bridge, a gap longer than a query waits for its next message (the cache
// had lapsed), else the new query it started, with why; else it's unknown.
pub fn check(session: &Session, id: &str, bridge: Option<String>) -> Option<Value> {
    let entry = session.entry(id)?;
    let SessionEntry::Message { message: AgentMessage::Assistant(current), .. } = entry else { return None };
    let written = context(current) - current.usage.cache_read;
    if written <= MISS_MIN_TOKENS {
        return None;
    }
    // Walk back to the last response that had a context (one stopped before any usage came has
    // none), noting what came between.
    let (mut compacted, mut prompt_changed) = (false, false);
    let mut parent = entry.parent_id();
    let previous = loop {
        let entry = session.entry(parent?)?;
        match entry {
            SessionEntry::Message { message: AgentMessage::Assistant(previous), .. } if context(previous) > 0 => break previous,
            SessionEntry::Compaction { .. } => compacted = true,
            SessionEntry::Message { message: AgentMessage::System(_), .. } => prompt_changed = true,
            _ => {}
        }
        parent = entry.parent_id();
    };
    let before = context(previous);
    if current.usage.cache_read as f64 >= before as f64 * MISS_READ_SHARE {
        return None;
    }
    let gap_ms = current.timestamp - previous.timestamp;
    let bridged = current.provider == crate::provider::claude::PROVIDER;
    let (cause, detail) = if compacted {
        ("compaction", None)
    } else if (&previous.provider, &previous.model) != (&current.provider, &current.model) {
        ("model change", None)
    } else if prompt_changed {
        ("prompt change", None)
    } else if bridged && gap_ms > crate::provider::claude::hold_time().as_millis() as i64 {
        ("cache expired", Some(format!("idle {} min", gap_ms / 60_000)))
    } else {
        (if bridge.is_some() { "query restart" } else { "unknown" }, bridge)
    };
    Some(json!({"written": written, "read": current.usage.cache_read, "before": before, "cause": cause, "detail": detail}))
}

// A miss as one line: "cache miss: 412k written, 4k read of 430k (cache expired: idle 72 min)".
pub fn line(miss: &Value) -> String {
    let number = |key: &str| miss[key].as_u64().unwrap_or(0);
    let detail = miss["detail"].as_str().map(|detail| format!(": {detail}")).unwrap_or_default();
    format!("cache miss: {} written, {} read of {} ({}{detail})", tokens(number("written")), tokens(number("read")), tokens(number("before")), miss["cause"].as_str().unwrap_or("unknown"))
}

// `piccosta cache [hours]`: every miss recorded in the last `hours` (24 when not given) across
// sessions, subagents' included, totals by cause first, then each one, oldest first.
pub fn report(agent_dir: &Path, hours: Option<&str>) -> String {
    let hours = hours.and_then(|hours| hours.parse().ok()).unwrap_or(DEFAULT_REPORT_HOURS);
    let since = crate::clock::now_ms() - hours * 60 * 60 * 1000;
    let listing = |dir: &Path| std::fs::read_dir(dir).into_iter().flatten().flatten().map(|entry| entry.path()).collect::<Vec<_>>();
    let folders = listing(&agent_dir.join("sessions"));
    let files = folders.iter().flat_map(|dir| [listing(dir), listing(&dir.join("subagents"))]).flatten().filter(|path| path.extension().is_some_and(|extension| extension == "jsonl"));
    let changed_since = std::time::UNIX_EPOCH + std::time::Duration::from_millis(since.max(0) as u64);
    let mut misses: Vec<(i64, String, Value)> = Vec::new();
    for path in files.filter(|path| path.metadata().and_then(|metadata| metadata.modified()).is_ok_and(|modified| modified >= changed_since)) {
        let Ok(session) = Session::read(&path) else { continue };
        let project = Path::new(&session.header().cwd).file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        for entry in session.entries() {
            let SessionEntry::Custom { custom_type, data: Some(miss), .. } = entry else { continue };
            if let Some(time) = crate::clock::ms_from_iso(&entry.base().timestamp).filter(|time| custom_type == MISS_TYPE && *time >= since) {
                misses.push((time, project.clone(), miss.clone()));
            }
        }
    }
    misses.sort_by_key(|(time, ..)| *time);
    let written = |miss: &Value| miss["written"].as_u64().unwrap_or(0);
    let mut causes: Vec<(&str, u64, usize)> = Vec::new();
    for (_, _, miss) in &misses {
        let cause = miss["cause"].as_str().unwrap_or("unknown");
        match causes.iter_mut().find(|(name, ..)| *name == cause) {
            Some((_, total, count)) => (*total, *count) = (*total + written(miss), *count + 1),
            None => causes.push((cause, written(miss), 1)),
        }
    }
    causes.sort_by_key(|(_, total, _)| std::cmp::Reverse(*total));
    let total: u64 = misses.iter().map(|(_, _, miss)| written(miss)).sum();
    let mut lines = vec![format!("cache misses in the last {hours} h (times UTC): {} turns, {} tokens written again", misses.len(), tokens(total))];
    lines.extend(causes.iter().map(|(cause, total, count)| format!("  {cause:<16} {:>7}  {count} turns", tokens(*total))));
    lines.push(String::new());
    for (time, project, miss) in &misses {
        let when = crate::clock::iso_from_ms(*time);
        lines.push(format!("{} {project:<20} {}", when[..16.min(when.len())].replace('T', " "), line(miss).trim_start_matches("cache miss: ")));
    }
    lines.join("\n") + "\n"
}
