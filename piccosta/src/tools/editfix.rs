// What a failed edit tells the model, beyond pi's one-line error: enough to get the retry right
// in one go instead of re-reading the file.
//   - when the file changed since the model last saw it, that it changed and where;
//   - for an oldText that wasn't found, the closest region of the current file with line numbers
//     and a short diff (or a plain "nothing close" when nothing scores well enough);
//   - for an oldText found several times, the line of each occurrence;
//   - for a multi-edit call, which of the other edits would have applied, since edits are all or
//     none.
//
// Why: pi's edit failed 10.9% of the time over 187k tool calls from real sessions, and 80% of
// those failures were "Could not find edits[N]", many in parallel worktrees where the file had
// moved under the model after it read it.
//
// "Last saw it" means the last successful read, edit or write of that file in this session, plus
// shell reads (cat/head/tail/sed/nl), because some models read only through bash.
//
// Based on the TypeScript piccosta's extensions/editfix.ts (FSL-1.1-ALv2, Synoros), with three fixes: oldText
// is LF-normalized before it's classified, as pi's edit does, so a CRLF oldText pi would apply
// isn't reported "not found"; occurrences are counted without overlaps, as pi counts them; and
// paths resolve with pi's rules (unicode spaces, file:// URLs). Character bigrams are over
// characters, where the TypeScript took UTF-16 units; the two differ only for text outside the
// Basic Multilingual Plane.

use super::edit::{count_occurrences, normalize_for_fuzzy_match, Edit};
use super::paths;
use crate::js;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

// A region has to be at least this similar (Dice coefficient over character bigrams) before we
// show it as "closest". 0.6 is the cutoff Python's difflib.get_close_matches uses for the same
// "did you mean" job; on the .scratch scenarios real near-misses scored 0.8+ and an unrelated
// block scored under 0.4, so the line sits comfortably between them.
const SIMILARITY_FLOOR: f64 = 0.6;
// Windows one line shorter and one line longer than oldText are tried too, so a line added or
// removed inside the target since the read still lines up.
const WINDOW_SLACK_LINES: usize = 1;
// How many lines of the matched region and of the diff we print. pi's guideline asks for small
// oldText, so 30 covers nearly every real edit without flooding the context.
const MAX_REGION_LINES: usize = 30;
const MAX_DIFF_LINES: usize = 30;
// Upper bound on characters compared in the full sliding-window scan (file lines x oldText size x
// window sizes). Above it we only score windows anchored on a line that appears in both, which
// keeps a 100-line oldText against a 50k-line file under a second.
const SCAN_CHAR_BUDGET: usize = 20_000_000;
// Line diffs (oldText vs region, snapshot vs current file) use a plain LCS table after the common
// head and tail are trimmed off. Past this many cells we stop being exact and report the whole
// differing middle as one block.
const LCS_CELL_BUDGET: usize = 4_000_000;
// Don't reread whole large files just to annotate a later failed edit. Text already in memory
// can keep a hash, but only snapshots up to this size keep line ranges. 2 MB is far above pi's
// 50 KB read cap, so most files a model edits fit.
pub(super) const SNAPSHOT_MAX_BYTES: usize = 2 * 1024 * 1024;
// How many changed ranges we list before summarizing the rest as "and N more".
const MAX_CHANGED_RANGES: usize = 8;
// Shell commands that show a file to the model. These are the read commands mined from
// transcripts in notes/tool-usage.md finding 2 (head, sed -n, cat, tail, nl).
const SHELL_READ_COMMANDS: [&str; 5] = ["cat", "head", "tail", "sed", "nl"];

// ---------------------------------------------------------------------------------------
// Snapshots: what each file looked like when the model last saw it

struct Snapshot {
    hash: u64,
    content: Option<String>,
    // Which tool showed it: read, edit, write, bash.
    via: &'static str,
    at: Instant,
}

// The files this session's model has seen, keyed by real path so "./a.ts", "src/../a.ts" and a
// symlink all meet.
#[derive(Default)]
pub struct SeenFiles {
    files: Mutex<HashMap<PathBuf, Snapshot>>,
}

// A file's text the way edit matches against it: BOM off, CRLF and lone CR made LF.
fn as_edit_sees(text: &str) -> String {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn read_as_edit_sees(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(as_edit_sees(&String::from_utf8_lossy(&bytes)))
}

fn hash_text(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

impl SeenFiles {
    // Input: an absolute path and the text the model was just shown or just wrote, as the file
    // holds it. Records it as what the model last saw.
    pub fn remember(&self, path: &Path, text: &str, via: &'static str) {
        let content = as_edit_sees(text);
        let snapshot = Snapshot {
            hash: hash_text(&content),
            content: (text.len() <= SNAPSHOT_MAX_BYTES).then_some(content),
            via,
            at: Instant::now(),
        };
        self.files.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).insert(paths::canonical(path), snapshot);
    }

    pub(super) fn forget(&self, path: &Path) {
        self.files.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).remove(&paths::canonical(path));
    }

    // Skip oversized files before reading. The bounded reader also handles a file growing after
    // metadata: tracking a shell read must never turn head -n 1 into an unbounded read of its own.
    pub fn record(&self, path: &Path, via: &'static str) {
        use std::io::Read;
        let Ok(metadata) = std::fs::metadata(path) else { return };
        if !metadata.is_file() { return; }
        if metadata.len() > SNAPSHOT_MAX_BYTES as u64 {
            self.forget(path);
            return;
        }
        let Ok(file) = std::fs::File::open(path) else { return };
        let mut bytes = Vec::new();
        if file.take(SNAPSHOT_MAX_BYTES as u64 + 1).read_to_end(&mut bytes).is_err() { return; }
        if bytes.len() > SNAPSHOT_MAX_BYTES {
            self.forget(path);
            return;
        }
        self.remember(path, &String::from_utf8_lossy(&bytes), via);
    }

    // Input: a bash command line and the cwd it ran in. Records a snapshot for every existing file
    // a read-style command (cat, head, ...) named. It only follows "cd X" segments within the same
    // command, because every bash call starts fresh in the session cwd. This is a cheap word
    // split, not a shell parser: quoted paths with spaces are missed, which only means no snapshot
    // for them.
    pub fn record_shell_reads(&self, command: &str, cwd: &Path) {
        let mut base_dir = cwd.to_path_buf();
        for segment in split_segments(command) {
            let words: Vec<String> = segment.split(js::is_whitespace).filter(|word| !word.is_empty()).map(strip_quotes).collect();
            let Some(first) = words.first() else { continue };
            if first == "cd" {
                if let Some(target) = words.get(1) {
                    base_dir = paths::resolve(target, &base_dir);
                }
                continue;
            }
            if !SHELL_READ_COMMANDS.contains(&first.as_str()) {
                continue;
            }
            for word in &words[1..] {
                if word.starts_with('-') || word.contains('>') || word.contains('<') {
                    continue;
                }
                self.record(&paths::resolve(word, &base_dir), "bash");
            }
        }
    }

    fn snapshot_of(&self, path: &Path) -> Option<(u64, Option<String>, &'static str, Instant)> {
        let files = self.files.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        files.get(&paths::canonical(path)).map(|snapshot| (snapshot.hash, snapshot.content.clone(), snapshot.via, snapshot.at))
    }
}

// A command line cut at &&, ||, ;, | and newlines.
fn split_segments(command: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let bytes = command.as_bytes();
    let (mut start, mut index) = (0, 0);
    while index < bytes.len() {
        let two = bytes.get(index..index + 2);
        let width = if two == Some(b"&&") || two == Some(b"||") {
            2
        } else if matches!(bytes[index], b';' | b'|' | b'\n') {
            1
        } else {
            index += 1;
            continue;
        };
        segments.push(&command[start..index]);
        index += width;
        start = index;
    }
    segments.push(&command[start..]);
    segments
}

// One quote off each end, as `word.replace(/^['"]|['"]$/g, "")` takes them.
fn strip_quotes(word: &str) -> String {
    let word = word.strip_prefix(['\'', '"']).unwrap_or(word);
    word.strip_suffix(['\'', '"']).unwrap_or(word).to_string()
}

// ---------------------------------------------------------------------------------------
// Matching, the same way the edit does

enum EditStatus {
    Empty,
    Unique { line: usize, fuzzy: bool },
    Duplicate { lines: Vec<usize> },
    Missing,
}

// 1-based line number of a byte offset.
fn line_at_offset(text: &str, offset: usize) -> usize {
    1 + text.as_bytes()[..offset].iter().filter(|byte| **byte == b'\n').count()
}

// Input: the current file text (as edit sees it) and one LF-normalized oldText. Output: what the
// edit would make of this oldText on its own: exact match first, then the normalized match, with
// uniqueness counted in normalized space.
fn classify_edit(content: &str, old_text: &str) -> EditStatus {
    if old_text.is_empty() {
        return EditStatus::Empty;
    }
    let exact_at = content.find(old_text);
    let normalized_content = normalize_for_fuzzy_match(content);
    let normalized_old = normalize_for_fuzzy_match(old_text);
    let occurrences = count_occurrences(content, old_text);
    if occurrences > 1 {
        let lines = if normalized_old.is_empty() {
            Vec::new()
        } else {
            normalized_content.match_indices(&normalized_old).map(|(at, _)| line_at_offset(&normalized_content, at)).collect()
        };
        return EditStatus::Duplicate { lines };
    }
    if let Some(at) = exact_at {
        return EditStatus::Unique { line: line_at_offset(content, at), fuzzy: false };
    }
    match normalized_content.find(&normalized_old) {
        Some(at) if !normalized_old.is_empty() => EditStatus::Unique { line: line_at_offset(&normalized_content, at), fuzzy: true },
        _ => EditStatus::Missing,
    }
}

// ---------------------------------------------------------------------------------------
// Line diff

#[derive(Debug, Clone, PartialEq)]
pub struct DiffOp<'a> {
    // '=', '-' or '+'.
    pub op: char,
    pub text: &'a str,
    // 1-based line numbers into each side (the number of the next line on the side an op
    // doesn't touch).
    pub old_line: usize,
    pub new_line: usize,
}

// Input: two lists of lines. Output: a line diff as = / - / + ops. The common head and tail are
// peeled off first; the middle gets an LCS table if it fits in LCS_CELL_BUDGET, otherwise it's
// reported as all removed, then all added.
pub fn diff_lines<'a>(old_lines: &[&'a str], new_lines: &[&'a str]) -> Vec<DiffOp<'a>> {
    let mut head = 0;
    while head < old_lines.len() && head < new_lines.len() && old_lines[head] == new_lines[head] {
        head += 1;
    }
    let mut tail = 0;
    while tail < old_lines.len() - head && tail < new_lines.len() - head && old_lines[old_lines.len() - 1 - tail] == new_lines[new_lines.len() - 1 - tail] {
        tail += 1;
    }

    let mut ops: Vec<DiffOp<'a>> = (0..head).map(|i| DiffOp { op: '=', text: old_lines[i], old_line: i + 1, new_line: i + 1 }).collect();
    let old_middle = &old_lines[head..old_lines.len() - tail];
    let new_middle = &new_lines[head..new_lines.len() - tail];
    let (rows, cols) = (old_middle.len(), new_middle.len());
    if rows * cols <= LCS_CELL_BUDGET && rows > 0 && cols > 0 {
        // lcs[i][j] = length of the longest common subsequence of old_middle[i..] and
        // new_middle[j..], flattened into one array of (rows+1) x (cols+1).
        let width = cols + 1;
        let mut lcs = vec![0u32; (rows + 1) * width];
        for i in (0..rows).rev() {
            for j in (0..cols).rev() {
                lcs[i * width + j] = if old_middle[i] == new_middle[j] {
                    lcs[(i + 1) * width + j + 1] + 1
                } else {
                    lcs[(i + 1) * width + j].max(lcs[i * width + j + 1])
                };
            }
        }
        let (mut i, mut j) = (0, 0);
        while i < rows || j < cols {
            let (old_line, new_line) = (head + i + 1, head + j + 1);
            if i < rows && j < cols && old_middle[i] == new_middle[j] {
                ops.push(DiffOp { op: '=', text: old_middle[i], old_line, new_line });
                i += 1;
                j += 1;
            } else if i < rows && (j == cols || lcs[(i + 1) * width + j] >= lcs[i * width + j + 1]) {
                // Removals before additions on a tie, so a changed line reads "-old" then "+new".
                ops.push(DiffOp { op: '-', text: old_middle[i], old_line, new_line });
                i += 1;
            } else {
                ops.push(DiffOp { op: '+', text: new_middle[j], old_line, new_line });
                j += 1;
            }
        }
    } else {
        for (i, text) in old_middle.iter().enumerate() {
            ops.push(DiffOp { op: '-', text, old_line: head + i + 1, new_line: head + 1 });
        }
        for (j, text) in new_middle.iter().enumerate() {
            ops.push(DiffOp { op: '+', text, old_line: head + rows + 1, new_line: head + j + 1 });
        }
    }
    for k in 0..tail {
        let (old_index, new_index) = (old_lines.len() - tail + k, new_lines.len() - tail + k);
        ops.push(DiffOp { op: '=', text: old_lines[old_index], old_line: old_index + 1, new_line: new_index + 1 });
    }
    ops
}

fn range_text(start_line: usize, end_line: usize) -> String {
    if start_line == end_line { format!("line {start_line}") } else { format!("lines {start_line}-{end_line}") }
}

// Input: a diff between the snapshot and the current file. Output: where the current file
// differs, in current line numbers, e.g. "lines 4-6, line 40, 2 line(s) removed before line 80".
fn describe_changed_ranges(ops: &[DiffOp]) -> String {
    let mut parts = Vec::new();
    let mut k = 0;
    while k < ops.len() {
        if ops[k].op == '=' {
            k += 1;
            continue;
        }
        // One run of consecutive - and + ops is one changed place.
        let (mut added, mut removed) = (0, 0);
        let (mut first_new, mut last_new) = (None, 0);
        let run_start_new_line = ops[k].new_line;
        while k < ops.len() && ops[k].op != '=' {
            if ops[k].op == '+' {
                added += 1;
                first_new.get_or_insert(ops[k].new_line);
                last_new = ops[k].new_line;
            } else {
                removed += 1;
            }
            k += 1;
        }
        match first_new {
            Some(first) if added > 0 => parts.push(range_text(first, last_new)),
            _ => parts.push(format!("{removed} line(s) removed before line {run_start_new_line}")),
        }
    }
    if parts.len() > MAX_CHANGED_RANGES {
        let extra = parts.len() - MAX_CHANGED_RANGES;
        return format!("{}, and {extra} more", parts[..MAX_CHANGED_RANGES].join(", "));
    }
    parts.join(", ")
}

// Input: oldText lines and the file region they were compared with. Output: diff lines, "-" for
// the model's oldText and "+" for the file, with the file's line numbers on "+" and on context
// lines. Unchanged lines are kept only next to a change.
fn format_region_diff(old_lines: &[&str], region_lines: &[&str], region_start_line: usize) -> Vec<String> {
    let ops = diff_lines(old_lines, region_lines);
    let is_change = |index: Option<usize>| index.and_then(|index| ops.get(index)).is_some_and(|op| op.op != '=');
    let mut out = Vec::new();
    for (k, op) in ops.iter().enumerate() {
        let near_change = op.op != '=' || is_change(k.checked_sub(1)) || is_change(Some(k + 1));
        if !near_change {
            continue;
        }
        let file_line = region_start_line + op.new_line - 1;
        if op.op == '-' {
            out.push(format!("-      {}", op.text));
        } else {
            out.push(format!("{}{file_line:>5} {}", if op.op == '+' { '+' } else { ' ' }, op.text));
        }
    }
    if out.len() > MAX_DIFF_LINES {
        let more = out.len() - MAX_DIFF_LINES;
        out.truncate(MAX_DIFF_LINES);
        out.push(format!("  ... {more} more diff lines"));
    }
    out
}

enum Mapped {
    Lines { start_line: usize, end_line: usize },
    RemovedBefore(usize),
}

// Input: a diff from the snapshot's lines to the current lines, and a 1-based inclusive line range
// in the snapshot. Output: the lines of the current file those snapshot lines became (kept lines
// plus whatever replaced the removed ones), or RemovedBefore when all of them are gone, naming the
// current line that now follows the gap.
fn map_range_through_diff(ops: &[DiffOp], start_line: usize, end_line: usize) -> Mapped {
    let on_old_side = |op: &DiffOp| op.op != '+';
    let first = ops.iter().position(|op| on_old_side(op) && op.old_line == start_line);
    let last = ops.iter().rposition(|op| on_old_side(op) && op.old_line == end_line);
    let (Some(mut first), Some(mut last)) = (first, last) else { return Mapped::RemovedBefore(1) };
    // A removed line at either edge counts its replacement lines as part of the range.
    while ops[first].op == '-' && first > 0 && ops[first - 1].op == '+' {
        first -= 1;
    }
    while ops[last].op == '-' && ops.get(last + 1).is_some_and(|op| op.op == '+') {
        last += 1;
    }
    let current: Vec<usize> = ops[first..=last].iter().filter(|op| op.op != '-').map(|op| op.new_line).collect();
    match (current.iter().min(), current.iter().max()) {
        (Some(min), Some(max)) => Mapped::Lines { start_line: *min, end_line: *max },
        _ => Mapped::RemovedBefore(ops[last].new_line),
    }
}

// ---------------------------------------------------------------------------------------
// Closest region

// Character-bigram counts of a string with whitespace runs folded to one space.
fn bigram_counts(text: &str) -> (HashMap<(char, char), usize>, usize) {
    let mut folded = String::with_capacity(text.len());
    let mut in_space = false;
    for c in js::trim(text).chars() {
        if js::is_whitespace(c) {
            if !in_space {
                folded.push(' ');
            }
            in_space = true;
        } else {
            folded.push(c);
            in_space = false;
        }
    }
    let chars: Vec<char> = folded.chars().collect();
    let mut counts = HashMap::new();
    for pair in chars.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    }
    let total = chars.len().saturating_sub(1);
    (counts, total)
}

// Dice coefficient of two bigram multisets: 2 * shared / (total in a + total in b), 0..1.
fn dice_similarity(a: &HashMap<(char, char), usize>, a_total: usize, b: &HashMap<(char, char), usize>, b_total: usize) -> f64 {
    if a_total + b_total == 0 {
        return 1.0;
    }
    let shared: usize = b.iter().map(|(pair, count)| a.get(pair).map_or(0, |in_a| (*in_a).min(*count))).sum();
    (2 * shared) as f64 / (a_total + b_total) as f64
}

struct RegionMatch {
    // 1-based, inclusive.
    start_line: usize,
    end_line: usize,
    score: f64,
    // The region equals oldText once all whitespace is removed.
    whitespace_only: bool,
}

fn squash(text: &str) -> String {
    text.chars().filter(|c| !js::is_whitespace(*c)).collect()
}

// Input: the current file's lines and an oldText the edit couldn't find. Output: the window of
// lines most like oldText, or None for an empty file or a blank oldText. Pass 1 looks for a window
// that equals oldText with all whitespace removed (indentation or tabs drifted). Pass 2 scores
// windows by bigram similarity and keeps the best.
fn find_closest_region(file_lines: &[&str], old_text: &str) -> Option<RegionMatch> {
    let old_lines: Vec<&str> = old_text.strip_suffix('\n').unwrap_or(old_text).split('\n').collect();
    if file_lines.is_empty() || js::trim(old_text).is_empty() {
        return None;
    }
    let mut window_sizes = vec![old_lines.len()];
    for slack in 1..=WINDOW_SLACK_LINES {
        if old_lines.len() > slack {
            window_sizes.push(old_lines.len() - slack);
        }
        window_sizes.push(old_lines.len() + slack);
    }

    // Pass 1: same text, different whitespace.
    let squashed_lines: Vec<String> = file_lines.iter().map(|line| squash(line)).collect();
    let squashed_old = squash(old_text);
    for &size in &window_sizes {
        for start in 0..(file_lines.len() + 1).saturating_sub(size) {
            if squashed_lines[start..start + size].concat() == squashed_old {
                return Some(RegionMatch { start_line: start + 1, end_line: start + size, score: 1.0, whitespace_only: true });
            }
        }
    }

    // Pass 2: which window starts to score. All of them when the scan fits the budget; otherwise
    // only starts that line up a line of oldText with an identical file line.
    let candidate_starts: Vec<usize> = if file_lines.len() * js::utf16_len(old_text) * window_sizes.len() <= SCAN_CHAR_BUDGET {
        (0..file_lines.len()).collect()
    } else {
        let mut old_line_indexes: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, line) in old_lines.iter().enumerate() {
            let key = js::trim(line);
            if !key.is_empty() {
                old_line_indexes.entry(key).or_default().push(index);
            }
        }
        let mut starts = HashSet::new();
        for (file_index, line) in file_lines.iter().enumerate() {
            for old_index in old_line_indexes.get(js::trim(line)).into_iter().flatten() {
                if file_index >= *old_index {
                    starts.insert(file_index - old_index);
                }
            }
        }
        let mut starts: Vec<usize> = starts.into_iter().collect();
        starts.sort_unstable();
        starts
    };

    let (target, target_total) = bigram_counts(old_text);
    let mut best: Option<RegionMatch> = None;
    for &size in &window_sizes {
        for &start in &candidate_starts {
            if start + size > file_lines.len() {
                continue;
            }
            let (window, window_total) = bigram_counts(&file_lines[start..start + size].join("\n"));
            let score = dice_similarity(&target, target_total, &window, window_total);
            if best.as_ref().is_none_or(|best| score > best.score) {
                best = Some(RegionMatch { start_line: start + 1, end_line: start + size, score, whitespace_only: false });
            }
        }
    }
    best
}

// Input: the file's lines and a 1-based inclusive range. Output: those lines numbered, in the same
// "    12 | text" shape as the diff, capped at MAX_REGION_LINES.
fn numbered_lines(file_lines: &[&str], start_line: usize, end_line: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = start_line;
    while line <= end_line && out.len() < MAX_REGION_LINES {
        out.push(format!("{line:>6} | {}", file_lines.get(line - 1).copied().unwrap_or_default()));
        line += 1;
    }
    let count = end_line + 1 - start_line;
    if count > MAX_REGION_LINES {
        out.push(format!("       ... {} more lines", count - MAX_REGION_LINES));
    }
    out
}

// ---------------------------------------------------------------------------------------
// Building the extra error text

// Input: the path as the model wrote it, its absolute path, the edits from the call and what the
// model has seen. Output: the lines to append to the edit's error, or nothing when there's
// nothing useful to add (the file is gone, say; the edit's own message covers that).
pub fn explain_failure(display_path: &str, absolute: &Path, edits: &[Edit], seen: &SeenFiles) -> Vec<String> {
    let Some(content) = read_as_edit_sees(absolute) else { return Vec::new() };
    let file_lines: Vec<&str> = content.split('\n').collect();
    let old_texts: Vec<String> = edits.iter().map(|edit| edit.old_text.replace("\r\n", "\n").replace('\r', "\n")).collect();
    let mut out = Vec::new();

    // 1. Did the file move since the model saw it? Said first: when true, it's the reason.
    let snapshot = seen.snapshot_of(absolute);
    let file_changed = snapshot.as_ref().is_some_and(|(hash, ..)| hash_text(&content) != *hash);
    let snapshot_content = snapshot.as_ref().and_then(|(_, content, ..)| content.as_deref());
    let snapshot_lines: Vec<&str> = snapshot_content.map(|text| text.split('\n').collect()).unwrap_or_default();
    // Snapshot-to-now diff, kept for mapping an oldText's old position to its new one below.
    let since_snapshot = (file_changed && snapshot_content.is_some()).then(|| diff_lines(&snapshot_lines, &file_lines));
    if let Some((_, _, via, at)) = &snapshot {
        let seconds_ago = (at.elapsed().as_millis() as f64 / 1000.0).round();
        if file_changed {
            let changed = since_snapshot.as_ref().map(|ops| format!("; changes, in current line numbers: {}", describe_changed_ranges(ops))).unwrap_or_default();
            out.push(format!("{display_path} changed since you last read it ({via}, {seconds_ago}s ago){changed}. Read those lines again before retrying."));
        } else {
            out.push(format!(
                "{display_path} has not changed since you last read it ({via}, {seconds_ago}s ago), so the oldText differs from what was shown to you."
            ));
        }
    }

    // 2. Per edit: where it lands now.
    let statuses: Vec<EditStatus> = old_texts.iter().map(|old_text| classify_edit(&content, old_text)).collect();
    for (index, status) in statuses.iter().enumerate() {
        match status {
            EditStatus::Duplicate { lines } => {
                let lines: Vec<String> = lines.iter().map(usize::to_string).collect();
                out.push(format!("edits[{index}] occurs {} times, at lines {}.", count_occurrences(&content, &old_texts[index]), lines.join(", ")));
                continue;
            }
            EditStatus::Missing => {}
            _ => continue,
        }
        let old_text = &old_texts[index];
        let old_line_count = old_text.strip_suffix('\n').unwrap_or(old_text).split('\n').count();
        // Best case: the file changed since the read and this oldText was there, once, in what the
        // model read. Then we don't guess; we follow those lines to where they are now.
        let in_snapshot = match (&since_snapshot, &snapshot_content) {
            (Some(_), Some(text)) => Some(classify_edit(text, old_text)),
            _ => None,
        };
        let region = if let (Some(ops), Some(EditStatus::Unique { line, .. })) = (&since_snapshot, &in_snapshot) {
            let snapshot_end = line + old_line_count - 1;
            match map_range_through_diff(ops, *line, snapshot_end) {
                Mapped::RemovedBefore(before) => {
                    out.push(format!(
                        "edits[{index}] was at {} when you read it; those lines have since been deleted (the gap is before line {before}).",
                        range_text(*line, snapshot_end)
                    ));
                    continue;
                }
                Mapped::Lines { start_line, end_line } => {
                    out.push(format!(
                        "edits[{index}] was at {} when you read it; that text is now at {}:",
                        range_text(*line, snapshot_end),
                        range_text(start_line, end_line)
                    ));
                    out.extend(numbered_lines(&file_lines, start_line, end_line));
                    (start_line, end_line)
                }
            }
        } else {
            let Some(found) = find_closest_region(&file_lines, old_text) else { continue };
            let range = range_text(found.start_line, found.end_line);
            if found.whitespace_only {
                out.push(format!("edits[{index}] matches {range} if whitespace is ignored. The file's exact text there:"));
            } else if found.score >= SIMILARITY_FLOOR {
                out.push(format!("edits[{index}] closest region: {range} (similarity {}):", js::to_fixed(found.score, 2)));
            } else {
                out.push(format!(
                    "edits[{index}]: nothing in {display_path} is close to this oldText (best similarity {} at {range}, need {}).",
                    js::to_fixed(found.score, 2),
                    js::number_string(SIMILARITY_FLOOR)
                ));
                continue;
            }
            out.extend(numbered_lines(&file_lines, found.start_line, found.end_line));
            (found.start_line, found.end_line)
        };
        out.push("diff (- your oldText, + file):".to_string());
        let old_lines: Vec<&str> = old_text.strip_suffix('\n').unwrap_or(old_text).split('\n').collect();
        let region_lines = &file_lines[region.0 - 1..region.1.min(file_lines.len())];
        out.extend(format_region_diff(&old_lines, region_lines, region.0));
    }

    // 3. For multi-edit calls: which ones were fine on their own.
    if edits.len() > 1 {
        out.push("Nothing was written (edits apply all or none). Each edit against the current file:".to_string());
        for (index, status) in statuses.iter().enumerate() {
            out.push(match status {
                EditStatus::Unique { line, fuzzy } => {
                    format!("  edits[{index}]: would apply (line {line}{})", if *fuzzy { ", after whitespace/quote normalization" } else { "" })
                }
                EditStatus::Duplicate { .. } => format!("  edits[{index}]: not unique ({} matches)", count_occurrences(&content, &old_texts[index])),
                EditStatus::Empty => format!("  edits[{index}]: empty oldText"),
                EditStatus::Missing => format!("  edits[{index}]: not found"),
            });
        }
    }
    out
}
