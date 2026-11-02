// The cache monitor. After each response piccosta compares what the provider read from its prompt
// cache with the context the turn before had; a turn that had to write most of it again is a miss,
// recorded in the session with the cause it can name, and shown as one line. `piccosta cache`
// lists the misses across every session for the last day (or the hours given), the same check run
// over each session file.

use crate::message::{AgentMessage, AssistantMessage, StopReason};
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
// An idle gap worth showing next to a miss: five minutes, the shortest a prompt cache lasts unused
// at any provider here (Anthropic's default; OpenAI keeps one five to ten minutes). It's only what
// was seen, not a cause: Claude Code on a plan asks Anthropic for an hour, and in real bridge
// sessions gaps of five minutes to an hour missed 3 times in 51, gaps over an hour 5 of 5
// (2026-10-31..11-01); DeepSeek keeps its cache for hours.
const IDLE_NOTE_MS: i64 = 5 * 60 * 1000;
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
// the custom entry's data (entryId, written, read, before, cause, seen, bridge, gapMs, model), or
// None when the turn read at least half of the context before it, wrote little, or has no turn
// before. The cause is named only when something that changes the request's prefix happened (a
// compaction since the turn before, another model, a new system message), or, on the Claude
// bridge, when the gap was longer than a query waits for its next message (the cache had lapsed:
// Claude Code asks for an hour); else it's unknown. What else happened in between goes in `seen`,
// as what was seen and not as the cause: a response stopped by the user, a message that came in
// while tools ran, the bridge starting a new Claude Code query (it rebuilds the same bytes when it
// can), an idle gap of five minutes or more. Each of those can keep the cache or lose it.
pub fn check(session: &Session, id: &str, bridge: Option<String>) -> Option<Value> {
    let entry = session.entry(id)?;
    let SessionEntry::Message { message: AgentMessage::Assistant(current), .. } = entry else { return None };
    let written = context(current) - current.usage.cache_read;
    if written <= MISS_MIN_TOKENS {
        return None;
    }
    // Walk back to the last response that had a context, noting what came between. A response
    // stopped before any usage came has no context, so it's passed over, but its stop is noted.
    let (mut compacted, mut prompt_changed, mut aborted, mut arrived_during_tools, mut after_user) = (false, false, false, false, false);
    let mut parent = entry.parent_id();
    let previous = loop {
        let entry = session.entry(parent?)?;
        match entry {
            SessionEntry::Message { message: AgentMessage::Assistant(previous), .. } if context(previous) > 0 => break previous,
            SessionEntry::Message { message: AgentMessage::Assistant(stopped), .. } => aborted |= stopped.stop_reason == StopReason::Aborted,
            SessionEntry::Compaction { .. } => compacted = true,
            SessionEntry::Message { message: AgentMessage::System(_), .. } => prompt_changed = true,
            SessionEntry::Message { message: AgentMessage::ToolResult(_), .. } => arrived_during_tools |= after_user,
            _ => {}
        }
        after_user = matches!(entry, SessionEntry::Message { message: AgentMessage::User(_), .. } | SessionEntry::CustomMessage { .. });
        parent = entry.parent_id();
    };
    let before = context(previous);
    if current.usage.cache_read as f64 >= before as f64 * MISS_READ_SHARE {
        return None;
    }
    let gap_ms = current.timestamp - previous.timestamp;
    let expired = current.provider == crate::provider::claude::PROVIDER && gap_ms > crate::provider::claude::hold_time().as_millis() as i64;
    let cause = if compacted {
        "compaction"
    } else if (&previous.provider, &previous.model) != (&current.provider, &current.model) {
        "model change"
    } else if prompt_changed {
        "prompt change"
    } else if expired {
        "cache expired"
    } else {
        "unknown"
    };
    let seen: Vec<&str> = [
        (aborted || previous.stop_reason == StopReason::Aborted, "abort"),
        (arrived_during_tools, "message during tools"),
        (bridge.is_some(), "query restart"),
        (gap_ms >= IDLE_NOTE_MS && !expired, "idle"),
    ]
    .into_iter()
    .filter_map(|(happened, what)| happened.then_some(what))
    .collect();
    Some(json!({"entryId": id, "written": written, "read": current.usage.cache_read, "before": before, "cause": cause, "seen": seen, "bridge": bridge,
        "gapMs": gap_ms, "model": format!("{}/{}", current.provider, current.model)}))
}

// A miss's cause with what was seen alongside it: "prompt change", "unknown; seen: abort, idle".
fn label(miss: &Value) -> String {
    let cause = miss["cause"].as_str().unwrap_or("unknown");
    let seen: Vec<&str> = miss["seen"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    if seen.is_empty() { cause.to_string() } else { format!("{cause}; seen: {}", seen.join(", ")) }
}

// A miss as one line: "cache miss: 412k written, 4k read of 430k (unknown; seen: abort, idle 72 min)".
pub fn line(miss: &Value) -> String {
    let number = |key: &str| miss[key].as_u64().unwrap_or(0);
    let idle = miss["gapMs"].as_i64().unwrap_or(0) / 60_000;
    let label = label(miss).replace("idle", &format!("idle {idle} min"));
    let bridge = miss["bridge"].as_str().map(|reason| format!("; Claude Code query restarted: {reason}")).unwrap_or_default();
    format!("cache miss: {} written, {} read of {} ({label}{bridge})", tokens(number("written")), tokens(number("read")), tokens(number("before")))
}

// Every session file under the agent dir's sessions, subagents' included.
fn session_files(agent_dir: &Path) -> Vec<std::path::PathBuf> {
    let listing = |dir: &Path| std::fs::read_dir(dir).into_iter().flatten().flatten().map(|entry| entry.path()).collect::<Vec<_>>();
    let directories = listing(&agent_dir.join("sessions"));
    let nested = directories.iter().flat_map(|dir| [listing(dir), listing(&dir.join("subagents"))]).flatten();
    nested.filter(|path| path.extension().is_some_and(|extension| extension == "jsonl")).collect()
}

// `piccosta cache [hours]`: every miss in the last `hours` (24 when not given) across sessions,
// totals by cause first, then each one, oldest first.
pub fn report(agent_dir: &Path, hours: Option<&str>) -> String {
    let hours = hours.and_then(|hours| hours.parse().ok()).unwrap_or(DEFAULT_REPORT_HOURS);
    let since = crate::clock::now_ms() - hours * 60 * 60 * 1000;
    let mut misses: Vec<(i64, String, Value)> = Vec::new();
    for path in session_files(agent_dir) {
        let modified = std::fs::metadata(&path).and_then(|metadata| metadata.modified()).ok();
        if modified.is_some_and(|modified| modified < std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_millis(since.max(0) as u64)) {
            continue;
        }
        let Ok(session) = Session::read(&path) else { continue };
        let project = Path::new(&session.header().cwd).file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        // Every turn is checked again here, so a miss recorded by an older build gets today's
        // attribution; only the bridge's reason, which the session can't show, comes from the record.
        let recorded: std::collections::HashMap<&str, Option<String>> = session.entries().iter().filter_map(|entry| match entry {
            SessionEntry::Custom { custom_type, data: Some(data), .. } if custom_type == MISS_TYPE => Some((data["entryId"].as_str()?, data["bridge"].as_str().map(String::from))),
            _ => None,
        }).collect();
        for entry in session.entries() {
            let SessionEntry::Message { message: AgentMessage::Assistant(message), .. } = entry else { continue };
            if message.timestamp < since {
                continue;
            }
            if let Some(miss) = check(&session, entry.id(), recorded.get(entry.id()).cloned().flatten()) {
                misses.push((message.timestamp, project.clone(), miss));
            }
        }
    }
    misses.sort_by_key(|(time, ..)| *time);
    let written = |miss: &Value| miss["written"].as_u64().unwrap_or(0);
    let mut causes: Vec<(String, u64, usize)> = Vec::new();
    for (_, _, miss) in &misses {
        let cause = label(miss);
        match causes.iter_mut().find(|(name, ..)| *name == cause) {
            Some((_, total, count)) => {
                *total += written(miss);
                *count += 1;
            }
            None => causes.push((cause, written(miss), 1)),
        }
    }
    causes.sort_by_key(|(_, total, _)| std::cmp::Reverse(*total));
    let total: u64 = misses.iter().map(|(_, _, miss)| written(miss)).sum();
    let mut lines = vec![format!("cache misses in the last {hours} h (times UTC): {} turns, {} tokens written again", misses.len(), tokens(total))];
    lines.extend(causes.iter().map(|(cause, total, count)| format!("  {cause:<44} {:>7}  {count} turns", tokens(*total))));
    lines.push(String::new());
    for (time, project, miss) in &misses {
        let when = crate::clock::iso_from_ms(*time);
        lines.push(format!("{} {project:<20} {}", &when[..16.min(when.len())].replace('T', " "), line(miss).trim_start_matches("cache miss: ")));
    }
    lines.join("\n") + "\n"
}
