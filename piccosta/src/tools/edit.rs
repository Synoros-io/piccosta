// The edit tool: exact-text replacements in one file, all applied or none.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/tools/edit.ts and
// edit-diff.ts. Matching is pi's: each oldText is looked up in the original file (not after the
// earlier edits), exactly first, then after pi's fuzzy normalization (NFKC, trailing whitespace,
// smart quotes, dashes, odd spaces); it must be unique and must not overlap another edit. The
// error texts are pi's, word for word, because models retry on them. When an edit fails, the
// diagnosis from piccosta's editfix (editfix.rs) goes under pi's message.
//
// Three things differ from pi, each where pi changes bytes the model never asked to change:
// - Line endings: pi rewrites every line ending in the file to the first one it finds, so a file
//   with mixed endings changes on every line. Here, lines the edit didn't touch keep their own
//   ending, and new text gets the file's first ending. A file with one kind of ending comes out
//   exactly as pi writes it. A lone CR counts as an ending, so an old Mac file stays CR.
// - A file that isn't valid UTF-8 is refused. pi decodes it lossily and writes the replacement
//   characters back, corrupting every invalid byte in the file.
// - pi's fuzzy path throws "different line count" when the last line is whitespace only with no
//   newline after it (normalization empties it and pi's line splitter then drops it). Lines are
//   counted by newlines here, so both sides always agree.

use super::{editfix, paths, Tool, ToolContext, ToolFuture, ToolOutput};
use crate::js;
use crate::message::{Content, ToolDefinition};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use unicode_normalization::UnicodeNormalization;

pub struct EditTool;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

#[derive(Deserialize)]
struct Arguments {
    path: String,
    edits: Vec<Edit>,
}

// The lines of the edit's display diff kept around each change (edit-diff.ts:379).
const DIFF_CONTEXT_LINES: usize = 4;

impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit".into(),
            description: "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.".into(),
            parameters: json!({
                "type": "object",
                "required": ["path", "edits"],
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["oldText", "newText"],
                            "properties": {
                                "oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."},
                                "newText": {"type": "string", "description": "Replacement text for this targeted edit."}
                            }
                        },
                        "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead."
                    }
                }
            }),
            constrained_sampling: Some(json!({"type": "json_schema", "strict": "prefer"})),
        }
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("replace exact pieces of a file's text, several in one call")
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        &["Each oldText in an edit call must appear exactly once in the file as it was before the call, kept just long enough to be unique; changes to one file go in one call, and changes that touch or overlap go in one entry"]
    }

    // Shapes models send instead of an `edits` array (edit.ts:103-134): the array as a JSON string
    // (Opus 4.6, GLM-5.1 do this), one edit object on its own, or the old top-level oldText and
    // newText, which are added to the end of `edits`.
    fn prepare_arguments(&self, mut arguments: Map<String, Value>) -> Map<String, Value> {
        let is_single_edit = |value: &Value| value.get("oldText").is_some_and(Value::is_string) && value.get("newText").is_some_and(Value::is_string);
        match arguments.get("edits") {
            Some(Value::String(text)) => {
                if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                    if parsed.is_array() {
                        arguments.insert("edits".into(), parsed);
                    } else if is_single_edit(&parsed) {
                        arguments.insert("edits".into(), Value::Array(vec![parsed]));
                    }
                }
            }
            Some(edit) if is_single_edit(edit) => {
                let edit = edit.clone();
                arguments.insert("edits".into(), Value::Array(vec![edit]));
            }
            _ => {}
        }
        let (Some(Value::String(old_text)), Some(Value::String(new_text))) = (arguments.get("oldText"), arguments.get("newText")) else {
            return arguments;
        };
        let legacy = json!({"oldText": old_text, "newText": new_text});
        let mut edits = match arguments.get("edits") {
            Some(Value::Array(edits)) => edits.clone(),
            _ => Vec::new(),
        };
        edits.push(legacy);
        arguments.shift_remove("oldText");
        arguments.shift_remove("newText");
        arguments.insert("edits".into(), Value::Array(edits));
        arguments
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: Arguments = serde_json::from_value(Value::Object(arguments)).map_err(|error| error.to_string())?;
            if arguments.edits.is_empty() {
                return Err("Edit tool input is invalid. edits must contain at least one replacement.".into());
            }
            let absolute = paths::resolve(&arguments.path, &context.cwd);
            let _turn = context.shared.mutations.lock(&absolute).await;
            match edit_file(&arguments.path, &absolute, &arguments.edits, context).await {
                Ok(output) => Ok(output),
                // A FIFO or device was refused without opening it; the diagnosis would open it.
                Err(message) if context.cancel.is_cancelled() || paths::refuse_special(&absolute).is_err() => Err(message),
                Err(message) => {
                    // The diagnosis reads the file and may scan it for a while; it runs off the
                    // async workers. If it fails the model still gets pi's message.
                    let shared = context.shared.clone();
                    let (path, edits) = (arguments.path.clone(), arguments.edits.clone());
                    let extra = tokio::task::spawn_blocking(move || editfix::explain_failure(&path, &absolute, &edits, &shared.seen))
                        .await
                        .unwrap_or_default();
                    if extra.is_empty() { Err(message) } else { Err(format!("{message}\n\n{}", extra.join("\n"))) }
                }
            }
        })
    }
}

async fn edit_file(path: &str, absolute: &std::path::Path, edits: &[Edit], context: &ToolContext) -> Result<ToolOutput, String> {
    // Abort is checked between steps, never mid-write (edit.ts:168-172).
    let aborted = || context.cancel.is_cancelled().then(|| "Operation aborted".to_string());
    if let Some(message) = aborted() {
        return Err(message);
    }
    if let Err(error) = paths::access(absolute, libc::R_OK | libc::W_OK) {
        if let Some(message) = aborted() {
            return Err(message);
        }
        return Err(format!("Could not edit file: {path}. Error code: {}.", paths::error_code(&error)));
    }
    paths::refuse_special(absolute)?;
    let bytes = tokio::fs::read(absolute).await.map_err(|error| paths::fs_error(&error, "read", None))?;
    if let Some(message) = aborted() {
        return Err(message);
    }
    let raw = String::from_utf8(bytes).map_err(|error| {
        format!(
            "Could not edit file: {path}. It is not valid UTF-8 (first bad byte at offset {}); edit would corrupt it. Use bash for this file.",
            error.utf8_error().valid_up_to()
        )
    })?;

    let (final_content, diff, first_changed_line) = edit_content(&raw, edits, path)?;
    if let Some(message) = aborted() {
        return Err(message);
    }
    tokio::fs::write(absolute, final_content.as_bytes()).await.map_err(|error| paths::fs_error(&error, "open", Some(absolute)))?;
    if let Some(message) = aborted() {
        return Err(message);
    }
    context.shared.seen.remember(absolute, &final_content, "edit");

    Ok(ToolOutput {
        content: vec![Content::text(format!("Successfully replaced {} block(s) in {path}.", edits.len()))],
        details: Some(json!({"diff": diff, "firstChangedLine": first_changed_line})),
        is_error: false,
    })
}

// Input: the file's UTF-8 text and edits against that original text. Output: the bytes to write
// and the tool's display diff. Permission previews use this too, before any write is allowed.
pub fn edit_content(raw: &str, edits: &[Edit], path: &str) -> Result<(String, String, Option<usize>), String> {
    if edits.is_empty() {
        return Err("Edit tool input is invalid. edits must contain at least one replacement.".into());
    }
    // The BOM is invisible to the model, so it's set aside before matching and put back after.
    let (bom, content) = match raw.strip_prefix('\u{feff}') {
        Some(rest) => ("\u{feff}", rest),
        None => ("", raw),
    };
    let text = EndingMap::new(content);
    let applied = apply_edits(&text.normalized, edits, path)?;
    let output = applied.restore(&text);
    let final_content = format!("{bom}{output}");
    let (diff, first_changed_line) = diff_string(&text.normalized, &applied.new_content);
    Ok((final_content, diff, first_changed_line))
}

// ---------------------------------------------------------------------------------------
// Line endings

// A file's text with every line ending made "\n", plus what each of those "\n" was, so text the
// edit didn't touch can be written back with its own endings.
pub struct EndingMap {
    pub normalized: String,
    // Byte offset in `normalized` of each "\n", and the ending it stands for.
    newlines: Vec<usize>,
    endings: Vec<&'static str>,
    // The first ending in the file, used for new text; "\n" when the file has none.
    pub file_ending: &'static str,
}

impl EndingMap {
    pub fn new(content: &str) -> EndingMap {
        let mut normalized = String::with_capacity(content.len());
        let mut newlines = Vec::new();
        let mut endings = Vec::new();
        let mut chars = content.char_indices().peekable();
        while let Some((_, c)) = chars.next() {
            let ending = match c {
                '\r' if chars.peek().is_some_and(|(_, next)| *next == '\n') => {
                    chars.next();
                    "\r\n"
                }
                '\r' => "\r",
                '\n' => "\n",
                other => {
                    normalized.push(other);
                    continue;
                }
            };
            newlines.push(normalized.len());
            endings.push(ending);
            normalized.push('\n');
        }
        let file_ending = endings.first().copied().unwrap_or("\n");
        EndingMap { normalized, newlines, endings, file_ending }
    }

    // The original text of normalized[start..end], with each line ending as it was in the file.
    fn original(&self, start: usize, end: usize, out: &mut String) {
        let first = self.newlines.partition_point(|at| *at < start);
        let last = self.newlines.partition_point(|at| *at < end);
        let mut from = start;
        for index in first..last {
            let at = self.newlines[index];
            out.push_str(&self.normalized[from..at]);
            out.push_str(self.endings[index]);
            from = at + 1;
        }
        out.push_str(&self.normalized[from..end]);
    }
}

// pi's normalizeToLF: CRLF and lone CR become LF.
fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

// ---------------------------------------------------------------------------------------
// Matching, as pi's edit-diff.ts does it

// pi's normalizeForFuzzyMatch (edit-diff.ts:34-55): NFKC, trailing whitespace off each line,
// smart quotes, dashes and odd spaces made ASCII.
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    let composed: String = text.nfkc().collect();
    let trimmed: Vec<&str> = composed.split('\n').map(js::trim_end).collect();
    trimmed
        .join("\n")
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => '\'',
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => '"',
            '\u{2010}'..='\u{2015}' | '\u{2212}' => '-',
            '\u{a0}' | '\u{2002}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

struct FoundText {
    index: usize,
    length: usize,
    fuzzy: bool,
}

// Exact first, then in fuzzy-normalized space, where offsets are into the normalized content.
fn fuzzy_find_text(content: &str, old_text: &str) -> Option<FoundText> {
    if let Some(index) = content.find(old_text) {
        return Some(FoundText { index, length: old_text.len(), fuzzy: false });
    }
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old_text);
    fuzzy_content.find(&fuzzy_old).map(|index| FoundText { index, length: fuzzy_old.len(), fuzzy: true })
}

// How many times oldText occurs, counted as pi counts it: in fuzzy space, without overlaps, the way
// String.split counts separators. An oldText that normalizes to nothing splits between every
// UTF-16 unit in JavaScript, so it "occurs" length-1 times.
pub fn count_occurrences(content: &str, old_text: &str) -> usize {
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old_text);
    if fuzzy_old.is_empty() {
        return js::utf16_len(&fuzzy_content).saturating_sub(1);
    }
    fuzzy_content.matches(&fuzzy_old).count()
}

fn not_found_error(path: &str, index: usize, total: usize) -> String {
    if total == 1 {
        format!("Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines.")
    } else {
        format!("Could not find edits[{index}] in {path}. The oldText must match exactly including all whitespace and newlines.")
    }
}

fn duplicate_error(path: &str, index: usize, total: usize, occurrences: usize) -> String {
    if total == 1 {
        format!("Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique.")
    } else {
        format!(
            "Found {occurrences} occurrences of edits[{index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        )
    }
}

fn empty_old_text_error(path: &str, index: usize, total: usize) -> String {
    if total == 1 { format!("oldText must not be empty in {path}.") } else { format!("edits[{index}].oldText must not be empty in {path}.") }
}

fn no_change_error(path: &str, total: usize) -> String {
    if total == 1 {
        format!(
            "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
        )
    } else {
        format!("No changes made to {path}. The replacements produced identical content.")
    }
}

#[derive(Clone)]
struct Replacement {
    edit_index: usize,
    start: usize,
    length: usize,
    new_text: String,
}

// The result of matching every edit: what the new file is in LF space (for the no-change check and
// the diff), and enough to write it back with the original line endings.
struct Applied {
    new_content: String,
    replacements: Vec<Replacement>,
    // Some(fuzzy-normalized content) when any edit needed fuzzy matching; offsets are then into it.
    fuzzy_base: Option<String>,
}

// Input: the file's LF content, the edits and the path for messages. Output: the edits matched
// and applied, or pi's error for the first one that doesn't fit (edit-diff.ts:301-370).
fn apply_edits(normalized: &str, edits: &[Edit], path: &str) -> Result<Applied, String> {
    let edits: Vec<Edit> = edits.iter().map(|edit| Edit { old_text: normalize_to_lf(&edit.old_text), new_text: normalize_to_lf(&edit.new_text) }).collect();
    for (index, edit) in edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(empty_old_text_error(path, index, edits.len()));
        }
    }
    // If any edit needs fuzzy matching, every edit is matched in fuzzy space.
    let any_fuzzy = edits.iter().any(|edit| fuzzy_find_text(normalized, &edit.old_text).is_some_and(|found| found.fuzzy));
    let fuzzy_base = any_fuzzy.then(|| normalize_for_fuzzy_match(normalized));
    let base = fuzzy_base.as_deref().unwrap_or(normalized);

    let mut replacements = Vec::new();
    for (index, edit) in edits.iter().enumerate() {
        let found = fuzzy_find_text(base, &edit.old_text).ok_or_else(|| not_found_error(path, index, edits.len()))?;
        let occurrences = count_occurrences(base, &edit.old_text);
        if occurrences > 1 {
            return Err(duplicate_error(path, index, edits.len(), occurrences));
        }
        replacements.push(Replacement { edit_index: index, start: found.index, length: found.length, new_text: edit.new_text.clone() });
    }
    replacements.sort_by_key(|replacement| replacement.start);
    for pair in replacements.windows(2) {
        if pair[0].start + pair[0].length > pair[1].start {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                pair[0].edit_index, pair[1].edit_index
            ));
        }
    }

    let applied = Applied { new_content: String::new(), replacements, fuzzy_base };
    let new_content = applied.build(normalized, None);
    if new_content == normalized {
        return Err(no_change_error(path, edits.len()));
    }
    Ok(Applied { new_content, ..applied })
}

// The byte span of each line of `text`, ending included. Lines are counted by "\n", so a text
// ending in "\n" (or empty) has an empty last line; both sides of the fuzzy overlay always have
// the same count, since normalization never adds or removes a newline.
fn line_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for (at, _) in text.match_indices('\n') {
        spans.push((start, at + 1));
        start = at + 1;
    }
    spans.push((start, text.len()));
    spans
}

impl Applied {
    // Build the new file. With `restore` None the result is in LF space, as pi computes it. With
    // the file's EndingMap, untouched text keeps its original endings and new or rewritten text
    // gets the file's first ending.
    fn build(&self, normalized: &str, restore: Option<&EndingMap>) -> String {
        let mut out = String::with_capacity(normalized.len());
        let ending = restore.map(|map| map.file_ending).unwrap_or("\n");
        let push_new = |out: &mut String, text: &str| {
            if ending == "\n" { out.push_str(text) } else { out.push_str(&text.replace('\n', ending)) }
        };
        let push_untouched = |out: &mut String, start: usize, end: usize| match restore {
            Some(map) => map.original(start, end, out),
            None => out.push_str(&normalized[start..end]),
        };

        let Some(base) = self.fuzzy_base.as_deref() else {
            // Exact matches: the replacements are spans of the normalized content itself.
            let mut from = 0;
            for replacement in &self.replacements {
                push_untouched(&mut out, from, replacement.start);
                push_new(&mut out, &replacement.new_text);
                from = replacement.start + replacement.length;
            }
            push_untouched(&mut out, from, normalized.len());
            return out;
        };

        // Fuzzy matches are offsets into the fuzzy base. Each replacement widens to the lines it
        // touches; those lines are rebuilt from the base, every other line is copied from the
        // file (edit-diff.ts:122-173).
        let base_lines = line_spans(base);
        let original_lines = line_spans(normalized);
        let mut groups: Vec<(usize, usize, Vec<&Replacement>)> = Vec::new();
        for replacement in &self.replacements {
            let end = replacement.start + replacement.length;
            let start_line = base_lines.iter().position(|(start, stop)| replacement.start >= *start && replacement.start < *stop).unwrap_or(0);
            let mut end_line = start_line;
            while end_line < base_lines.len() && base_lines[end_line].1 < end {
                end_line += 1;
            }
            let end_line = (end_line + 1).min(base_lines.len());
            match groups.last_mut() {
                Some(group) if start_line < group.1 => {
                    group.1 = group.1.max(end_line);
                    group.2.push(replacement);
                }
                _ => groups.push((start_line, end_line, vec![replacement])),
            }
        }
        let mut line = 0;
        for (start_line, end_line, replacements) in groups {
            if line < start_line {
                push_untouched(&mut out, original_lines[line].0, original_lines[start_line - 1].1);
            }
            let group_start = base_lines[start_line].0;
            let group_end = base_lines[end_line - 1].1;
            let mut rebuilt = String::new();
            let mut from = group_start;
            for replacement in replacements {
                rebuilt.push_str(&base[from..replacement.start]);
                rebuilt.push_str(&replacement.new_text);
                from = replacement.start + replacement.length;
            }
            rebuilt.push_str(&base[from..group_end]);
            push_new(&mut out, &rebuilt);
            line = end_line;
        }
        if line < original_lines.len() {
            push_untouched(&mut out, original_lines[line].0, normalized.len());
        }
        out
    }

    fn restore(&self, text: &EndingMap) -> String {
        self.build(&text.normalized, Some(text))
    }
}

// ---------------------------------------------------------------------------------------
// The display diff (edit-diff.ts:376-499): changed lines with their line numbers, up to four
// unchanged lines around each change, and "..." for what's skipped. Shown in the UI.

// Input: the old and new LF content. Output: the diff text and the first changed line in the new
// file. The line diff is editfix's LCS (pi uses jsdiff); the formatting is pi's.
pub fn diff_string(old: &str, new: &str) -> (String, Option<usize>) {
    let mut old_lines: Vec<&str> = old.split('\n').collect();
    let mut new_lines: Vec<&str> = new.split('\n').collect();
    let width = old_lines.len().max(new_lines.len()).to_string().len();
    // jsdiff keeps each line's newline, so a final newline is not a line of its own. Both texts
    // ending in one would otherwise show an empty last line as context.
    if old.ends_with('\n') && new.ends_with('\n') {
        old_lines.pop();
        new_lines.pop();
    }
    let ops = editfix::diff_lines(&old_lines, &new_lines);

    // jsdiff's parts: runs of the same kind.
    let mut parts: Vec<(char, Vec<&str>)> = Vec::new();
    for op in &ops {
        match parts.last_mut() {
            Some((kind, lines)) if *kind == op.op => lines.push(op.text),
            _ => parts.push((op.op, vec![op.text])),
        }
    }

    let number = |line: usize| format!("{line:>width$}");
    let skip = format!(" {} ...", " ".repeat(width));
    let mut out: Vec<String> = Vec::new();
    let (mut old_line, mut new_line) = (1usize, 1usize);
    let mut last_was_change = false;
    let mut first_changed = None;
    for (index, (kind, lines)) in parts.iter().enumerate() {
        match kind {
            '+' | '-' => {
                first_changed.get_or_insert(new_line);
                for text in lines {
                    if *kind == '+' {
                        out.push(format!("+{} {text}", number(new_line)));
                        new_line += 1;
                    } else {
                        out.push(format!("-{} {text}", number(old_line)));
                        old_line += 1;
                    }
                }
                last_was_change = true;
            }
            _ => {
                let next_is_change = parts.get(index + 1).is_some_and(|(next, _)| *next != '=');
                let mut lines_cursor = (old_line, new_line);
                if last_was_change && next_is_change {
                    if lines.len() <= DIFF_CONTEXT_LINES * 2 {
                        push_context(&mut out, lines, &mut lines_cursor, width);
                    } else {
                        push_context(&mut out, &lines[..DIFF_CONTEXT_LINES], &mut lines_cursor, width);
                        out.push(skip.clone());
                        let skipped = lines.len() - 2 * DIFF_CONTEXT_LINES;
                        lines_cursor.0 += skipped;
                        lines_cursor.1 += skipped;
                        push_context(&mut out, &lines[lines.len() - DIFF_CONTEXT_LINES..], &mut lines_cursor, width);
                    }
                } else if last_was_change {
                    let shown = lines.len().min(DIFF_CONTEXT_LINES);
                    push_context(&mut out, &lines[..shown], &mut lines_cursor, width);
                    if lines.len() > shown {
                        out.push(skip.clone());
                        lines_cursor.0 += lines.len() - shown;
                        lines_cursor.1 += lines.len() - shown;
                    }
                } else if next_is_change {
                    let skipped = lines.len().saturating_sub(DIFF_CONTEXT_LINES);
                    if skipped > 0 {
                        out.push(skip.clone());
                        lines_cursor.0 += skipped;
                        lines_cursor.1 += skipped;
                    }
                    push_context(&mut out, &lines[skipped..], &mut lines_cursor, width);
                } else {
                    lines_cursor.0 += lines.len();
                    lines_cursor.1 += lines.len();
                }
                (old_line, new_line) = lines_cursor;
                last_was_change = false;
            }
        }
    }
    (out.join("\n"), first_changed)
}

// Unchanged lines shown as context, numbered by the old file. Input: the lines and the old and
// new line numbers, which advance past them.
fn push_context(out: &mut Vec<String>, texts: &[&str], cursor: &mut (usize, usize), width: usize) {
    for text in texts {
        out.push(format!(" {:>width$} {text}", cursor.0));
        cursor.0 += 1;
        cursor.1 += 1;
    }
}
