// Cutting tool output down to what a model should see at once.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/tools/truncate.ts.
// Two independent limits, lines and bytes, and whichever is hit first wins. Lines are never cut
// in half except in one case: tail truncation of a single line longer than the byte limit keeps
// its end. Byte counts are UTF-8 bytes, as in pi.

use crate::js;
use serde::Serialize;

// pi's limits for read and bash output: 2000 lines or 50 KB (truncate.ts:11-12). Both are in the
// tool descriptions the model reads.
pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

// What truncation did, kept in a tool result's details for the UI and the session. Field names
// are pi's (TruncationResult), so details read the same either way.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Truncation {
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub output_bytes: usize,
    pub last_line_partial: bool,
    pub first_line_exceeds_limit: bool,
    pub max_lines: usize,
    pub max_bytes: usize,
}

// The lines of a text, for counting: a trailing newline doesn't start another line, and an empty
// text has no lines at all.
fn lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

// Input: a byte count. Output: it in B, KB or MB with one decimal, e.g. "50.0KB".
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{}KB", js::to_fixed(bytes as f64 / 1024.0, 1))
    } else {
        format!("{}MB", js::to_fixed(bytes as f64 / (1024.0 * 1024.0), 1))
    }
}

fn untouched(content: &str, total_lines: usize, max_lines: usize, max_bytes: usize) -> Truncation {
    Truncation {
        content: content.to_string(),
        truncated: false,
        truncated_by: None,
        total_lines,
        total_bytes: content.len(),
        output_lines: total_lines,
        output_bytes: content.len(),
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

// Keep the start of a text: as many whole lines from the top as fit both limits. Used by read.
// When even the first line is over the byte limit nothing is kept and first_line_exceeds_limit
// says so, so the caller can point the model somewhere else.
pub fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let lines = lines_for_counting(content);
    let total_lines = lines.len();
    if total_lines <= max_lines && content.len() <= max_bytes {
        return untouched(content, total_lines, max_lines, max_bytes);
    }
    let mut result = Truncation {
        content: String::new(),
        truncated: true,
        truncated_by: Some(TruncatedBy::Bytes),
        total_lines,
        total_bytes: content.len(),
        output_lines: 0,
        output_bytes: 0,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    };
    if lines[0].len() > max_bytes {
        result.first_line_exceeds_limit = true;
        return result;
    }

    let mut kept = 0;
    let mut kept_bytes = 0;
    let mut by = TruncatedBy::Lines;
    for (index, line) in lines.iter().enumerate().take(max_lines) {
        // Every line after the first costs its newline too.
        let line_bytes = line.len() + usize::from(index > 0);
        if kept_bytes + line_bytes > max_bytes {
            by = TruncatedBy::Bytes;
            break;
        }
        kept += 1;
        kept_bytes += line_bytes;
    }
    if kept >= max_lines && kept_bytes <= max_bytes {
        by = TruncatedBy::Lines;
    }
    result.content = lines[..kept].join("\n");
    result.truncated_by = Some(by);
    result.output_lines = kept;
    result.output_bytes = result.content.len();
    result
}

// Keep the end of a text: as many whole lines from the bottom as fit both limits. Used by bash,
// where the last lines (the error, the summary) matter most. A last line over the byte limit on
// its own keeps its final bytes, cut at a character boundary.
pub fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let total_bytes = content.len();
    let lines = lines_for_counting(content);
    let total_lines = lines.len();
    if total_lines <= max_lines && content.len() <= max_bytes {
        return untouched(content, total_lines, max_lines, max_bytes);
    }

    let mut kept: Vec<&str> = Vec::new();
    let mut partial: Option<&str> = None;
    let mut kept_bytes = 0;
    let mut by = TruncatedBy::Lines;
    for line in lines.iter().rev() {
        if kept.len() >= max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(!kept.is_empty());
        if kept_bytes + line_bytes > max_bytes {
            by = TruncatedBy::Bytes;
            if kept.is_empty() {
                let end = tail_of_str(line, max_bytes);
                kept_bytes = end.len();
                partial = Some(end);
            }
            break;
        }
        kept.push(line);
        kept_bytes += line_bytes;
    }
    if let Some(end) = partial {
        kept.push(end);
    }
    let output_lines = kept.len();
    if output_lines >= max_lines && kept_bytes <= max_bytes {
        by = TruncatedBy::Lines;
    }
    kept.reverse();
    let content = kept.join("\n");
    Truncation {
        output_bytes: content.len(),
        content,
        truncated: true,
        truncated_by: Some(by),
        total_lines,
        total_bytes,
        output_lines,
        last_line_partial: partial.is_some(),
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

// The last max_bytes bytes of a string, moved forward to the next character boundary.
fn tail_of_str(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}
