// pintask: pi's compaction, plus the user's own words from before the cut pinned verbatim onto the
// summary.
//
// Based on the TypeScript piccosta's extensions/pintask.ts with PICCOSTA_PIN_ALL=1, the launcher's default:
// every user message before the cut, in order, with no limit. Why it exists: in the 2026-10-19
// context ablation on gpt-6-luna, stock compaction landed inside the task's own turn and pi's
// split-turn summary turned a 6,951-char task statement into a ~1,000-char paraphrase; those runs
// scored F2P 23/51 against 38/51 for arms that kept the statement. Jack, 2026-10-20: "dont cap how
// much the user input can be ... the user messages should stay right but dont cap them,
// remembering the relevant older ones is important".
//
// What differs from the TS extension, which had to wrap pi's appendCompaction to get in:
// - pi's own summary, cut and request are untouched here by construction too; the block is added
//   where the entry is written (agent.rs), and json mode's compaction_end shows the stored summary
//   with the block, where the TS version could only show pi's.
// - PICCOSTA_PIN_ALL=0 means stock pi, no block. The capped modes (a 15% budget share,
//   PICCOSTA_PIN_RECENT_CHARS with index lines and a history file) were bench arms, superseded by
//   the "dont cap" direction, and aren't carried over; nor is the history file, which only they pointed to.
//
// Licensed FSL-1.1-ALv2, Copyright 2026 Synoros.

use crate::compaction::{Preparation, Reason};
use crate::message::{AgentMessage, Content, TextOrBlocks};
use crate::js;
use crate::session::SessionEntry;
use serde_json::{json, Value};

pub const PIN_ALL_ENV: &str = "PICCOSTA_PIN_ALL";
// The heading selfcompact put over the same words (selfcompact.ts buildSummary), and the blank
// line it joined its sections with.
const TASK_HEADING: &str = "## Task (the user's words, verbatim)";
const SECTION_GAP: &str = "\n\n";
// Between two quoted messages.
const SEPARATOR: &str = "\n\n---\n\n";
// Marks the counting entry after each pinned compaction (never sent to the model).
pub const STATS_ENTRY: &str = "piccosta-pintask";
// How much of each quoted message's first line the counting entry keeps, to tell them apart.
const HEAD_CHARS: usize = 80;

// On unless PICCOSTA_PIN_ALL=0.
pub fn enabled() -> bool {
    std::env::var(PIN_ALL_ENV).as_deref() != Ok("0")
}

// A user message's plain text, images left out (selfcompact.ts userText). Only messages the user
// sent: custom messages (a job's finish notice) aren't the user's words.
fn user_text(entry: &SessionEntry) -> Option<String> {
    let SessionEntry::Message { message: AgentMessage::User(user), .. } = entry else { return None };
    Some(match &user.content {
        TextOrBlocks::Text(text) => text.clone(),
        TextOrBlocks::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                Content::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    })
}

// The first `count` UTF-16 units of `text`, as JavaScript's slice(0, count).
fn utf16_prefix(text: &str, count: usize) -> String {
    let units: Vec<u16> = text.encode_utf16().take(count).collect();
    String::from_utf16_lossy(&units)
}

// Before pi's summarizer runs: cut the block this pinned onto the previous summary back off, so
// the summarizer updates the summary it wrote itself rather than one carrying the user's words
// (which the new block quotes again anyway). The counting entry for that compaction says how many
// UTF-16 units were added. Output: what happened, for the counting entry.
pub fn strip_previous_block(preparation: &mut Preparation, branch: &[&SessionEntry]) -> &'static str {
    let Some(previous) = preparation.previous_summary.as_deref().filter(|previous| !previous.is_empty()) else { return "none" };
    let compaction_id = branch.iter().rev().find(|entry| matches!(entry, SessionEntry::Compaction { .. })).map(|entry| entry.id());
    let appended = branch.iter().find_map(|entry| match entry {
        SessionEntry::Custom { custom_type, data: Some(data), .. }
            if custom_type == STATS_ENTRY && data.get("compactionId").and_then(Value::as_str) == compaction_id =>
        {
            data.get("appendedChars").and_then(Value::as_u64)
        }
        _ => None,
    });
    let Some(appended) = appended.filter(|count| *count > 0) else { return "not_ours" };
    let units: Vec<u16> = previous.encode_utf16().collect();
    let Some(cut_at) = units.len().checked_sub(appended as usize) else { return "mismatch" };
    let tail = String::from_utf16_lossy(&units[cut_at..]);
    if !tail.starts_with(&format!("{SECTION_GAP}{TASK_HEADING}")) {
        return "mismatch";
    }
    preparation.previous_summary = Some(String::from_utf16_lossy(&units[..cut_at]));
    "stripped"
}

// The block for a compaction, and its counting entry's data less the entry id.
pub struct Pin {
    pub block: String,
    stats: Value,
}

impl Pin {
    // The counting entry's data once the compaction entry has an id.
    pub fn stats(&self, compaction_id: &str) -> Value {
        let mut stats = json!({"compactionId": compaction_id});
        if let (Some(stats_out), Some(fields)) = (stats.as_object_mut(), self.stats.as_object()) {
            stats_out.extend(fields.clone());
        }
        stats
    }
}

// Input: the branch (root to leaf), the prepared compaction, why it ran, pi's summary, and what
// strip_previous_block did. Output: every user message before the cut, verbatim and in order,
// under a heading, ready to append to the summary ("" when there are none), and what to count.
pub fn pin(branch: &[&SessionEntry], preparation: &Preparation, reason: Reason, summary: &str, previous_status: &str) -> Pin {
    let first_kept = &preparation.first_kept_entry_id;
    let kept_at = branch.iter().position(|entry| entry.id() == first_kept).unwrap_or(branch.len());
    let dropped: Vec<(&str, String)> = branch[..kept_at]
        .iter()
        .filter_map(|entry| user_text(entry).map(|text| (entry.id(), text)))
        .filter(|(_, text)| !js::trim(text).is_empty())
        .collect();
    let task = dropped.iter().map(|(_, text)| text.as_str()).collect::<Vec<_>>().join(SEPARATOR);
    let block = if task.is_empty() { String::new() } else { format!("{SECTION_GAP}{TASK_HEADING}{SECTION_GAP}{task}") };

    // Each message before the cut and whether the block holds it, read off the block itself.
    let quoted: Vec<Value> = dropped
        .iter()
        .map(|(id, text)| {
            let status = if task.contains(text.as_str()) { "full" } else { "absent" };
            let head = utf16_prefix(js::trim(text).split('\n').next().unwrap_or(""), HEAD_CHARS);
            json!({"id": id, "chars": js::utf16_len(text), "status": status, "head": head})
        })
        .collect();
    let latest = quoted.last().map(|message| json!({"id": message["id"], "chars": message["chars"], "status": message["status"]}));
    let latest_in_full = quoted.last().is_some_and(|message| message["status"] == "full");
    let shown = quoted.iter().filter(|message| message["status"] != "absent").count();
    let stats = json!({
        "pinned": !block.is_empty(),
        "reason": reason,
        "isSplitTurn": preparation.is_split_turn,
        "firstKeptEntryId": first_kept,
        "tokensBefore": preparation.tokens_before,
        "piSummaryChars": js::utf16_len(summary),
        "pinnedChars": js::utf16_len(&task),
        "appendedChars": js::utf16_len(&block),
        "userMessages": shown,
        "droppedUserMessages": quoted.len(),
        "latestUser": latest,
        "latestUserInFull": latest_in_full,
        "previousSummary": previous_status,
        "quoted": quoted,
    });
    Pin { block, stats }
}
