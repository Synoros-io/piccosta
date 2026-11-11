// The read tool: a file's text, a window of its lines at a time, or an image as an attachment.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/tools/read.ts.
// Every message here is pi's; models are trained to follow the "Use offset=N to continue" lines.

use super::truncate::{self, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};
use super::{image, paths, Tool, ToolContext, ToolFuture, ToolOutput};
use crate::message::{Content, ToolDefinition};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::io::Read as _;

pub struct ReadTool;

#[derive(Deserialize)]
struct Arguments {
    path: String,
    offset: Option<f64>,
    limit: Option<f64>,
}

const NON_VISION_NOTE: &str = "[Current model does not support images. The image will be omitted from this request.]";

impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read".into(),
            description: format!(
                "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.",
                DEFAULT_MAX_BYTES / 1024
            ),
            parameters: json!({
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                    "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                    "limit": {"type": "number", "description": "Maximum number of lines to read"}
                }
            }),
            constrained_sampling: Some(json!({"type": "json_schema", "strict": "prefer"})),
        }
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("show a file's text, or an image")
    }

    fn read_only(&self) -> bool {
        true
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        &["Look at files with read, not cat or sed in bash"]
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: Arguments = serde_json::from_value(Value::Object(arguments)).map_err(|error| error.to_string())?;
            let absolute = paths::resolve(&arguments.path, &context.cwd);
            let task_context = ToolContext { session: None, ..context.clone() };
            let task = tokio::task::spawn_blocking(move || read_file(&arguments, &absolute, &task_context));
            tokio::select! {
                result = task => result.map_err(|error| error.to_string())?,
                () = context.cancel.cancelled() => Err("Operation aborted".to_string()),
            }
        })
    }
}

// The blocking part: everything after the path is resolved. Runs on tokio's blocking pool, so a
// slow disk or a large image resize doesn't stall the other tool calls.
fn read_file(arguments: &Arguments, absolute: &std::path::Path, context: &ToolContext) -> Result<ToolOutput, String> {
    // pi checks readability with access(R_OK) first, so that's the syscall a missing file names.
    paths::access(absolute, false).map_err(|error| super::pathhint::file_error(&error, "access", absolute, &context.cwd))?;
    let mut file = std::fs::File::open(absolute).map_err(|error| super::pathhint::file_error(&error, "open", absolute, &context.cwd))?;
    let mut bytes = Vec::new();
    (&mut file).take(image::SNIFF_BYTES as u64).read_to_end(&mut bytes).map_err(|error| paths::fs_error(&error, "read", None))?;

    if let Some(mime_type) = image::sniff_mime_type(&bytes) {
        file.read_to_end(&mut bytes).map_err(|error| paths::fs_error(&error, "read", None))?;
        // The line under the heading: the resize hints, or why the image couldn't be prepared,
        // then the note for a model without vision.
        let note = (!context.model_accepts_images).then_some(NON_VISION_NOTE);
        let (heading, extra, image) = match image::prepare(&bytes, mime_type, context.image_limits) {
            Ok(prepared) => (prepared.mime_type.clone(), prepared.hints, Some(Content::Image { data: prepared.data, mime_type: prepared.mime_type })),
            Err(message) => (mime_type.to_string(), vec![message], None),
        };
        let lines: Vec<String> = std::iter::once(format!("Read image file [{heading}]")).chain(extra).chain(note.map(str::to_string)).collect();
        let content = std::iter::once(Content::text(lines.join("\n"))).chain(image).collect();
        return Ok(ToolOutput { content, details: None, is_error: false });
    }

    if file.metadata().is_ok_and(|metadata| metadata.len() <= super::editfix::SNAPSHOT_MAX_BYTES as u64) {
        // Bound this read too: a file can grow after metadata. The prefix then feeds the same
        // streaming reader, and only complete, small files get an edit snapshot.
        (&mut file).take(super::editfix::SNAPSHOT_MAX_BYTES as u64 + 1).read_to_end(&mut bytes).map_err(|error| paths::fs_error(&error, "read", None))?;
        if bytes.len() <= super::editfix::SNAPSHOT_MAX_BYTES {
            let output = read_text(bytes.as_slice(), arguments)?;
            context.shared.seen.remember(absolute, &String::from_utf8_lossy(&bytes), "read");
            return Ok(output);
        }
    }
    context.shared.seen.forget(absolute);
    read_text(bytes.as_slice().chain(file), arguments)
}

// Input: a text stream and the requested window. Count omitted text for pi's notices and details,
// but retain only enough of the window to apply the existing truncation rules. UTF-8 decoding
// carries incomplete characters between chunks and, like from_utf8_lossy, keeps the BOM.
fn read_text(mut reader: impl std::io::Read, arguments: &Arguments) -> Result<ToolOutput, String> {
    // 1-indexed, and 0 or nothing means the first line. A fractional offset counts as its whole
    // part, as JavaScript's slice takes it.
    let start_line = match arguments.offset {
        Some(offset) if offset != 0.0 => (offset - 1.0).max(0.0).trunc() as usize,
        _ => 0,
    };
    let start_display = start_line.saturating_add(1);
    let end_line = arguments.limit.map_or(usize::MAX, |limit| start_line.saturating_add(limit.trunc().max(0.0) as usize));
    let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();
    // A read-sized chunk amortizes syscalls; the decoder supplies its worst-case lossy size.
    let mut input = vec![0; DEFAULT_MAX_BYTES];
    let mut decoded = String::new();
    let (mut line, mut total_bytes, mut first_line_bytes) = (0, 0, 0);
    let (mut selected, mut ends_with_newline) = (String::new(), false);
    loop {
        let count = reader.read(&mut input).map_err(|error| paths::fs_error(&error, "read", None))?;
        decoded.clear();
        // A pending partial character changes the decoder's worst case for this chunk.
        decoded.reserve(decoder.max_utf8_buffer_length(count).ok_or("UTF-8 buffer too large")?);
        let (_, consumed, _) = decoder.decode_to_string(&input[..count], &mut decoded, count == 0);
        debug_assert_eq!(consumed, count);
        for part in decoded.split_inclusive('\n') {
            let newline = part.ends_with('\n');
            let content = if newline { &part[..part.len() - 1] } else { part };
            if line >= start_line && line < end_line {
                if line == start_line { first_line_bytes += content.len(); }
                // A newline belongs to the selected text only when the next line is selected too.
                let part = if newline && line.saturating_add(1) < end_line { part } else { content };
                total_bytes += part.len();
                if !part.is_empty() { ends_with_newline = part.ends_with('\n'); }
                if selected.len() <= DEFAULT_MAX_BYTES {
                    let mut end = part.len().min(DEFAULT_MAX_BYTES + 1 - selected.len());
                    while !part.is_char_boundary(end) { end += 1; }
                    selected.push_str(&part[..end]);
                }
            }
            if newline { line += 1; }
        }
        if count == 0 { break; }
    }
    let total_file_lines = line + 1;
    if start_line >= total_file_lines {
        let offset = crate::js::number_string(arguments.offset.unwrap_or(0.0));
        return Err(format!("Offset {offset} is beyond end of file ({total_file_lines} lines total)"));
    }

    let selected_lines = end_line.min(total_file_lines) - start_line;
    let user_limited_lines = arguments.limit.map(|_| selected_lines);
    let mut truncation = truncate::truncate_head(&selected, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    truncation.total_bytes = total_bytes;
    truncation.total_lines = if total_bytes == 0 { 0 } else { selected_lines - usize::from(ends_with_newline) };
    truncation.truncated = truncation.total_lines > DEFAULT_MAX_LINES || total_bytes > DEFAULT_MAX_BYTES;
    if truncation.truncated && !truncation.first_line_exceeds_limit {
        // The prefix can end on a newline with the next line still outside it. Only the full
        // window tells us whether another line hit the byte budget before the line limit.
        truncation.truncated_by = Some(if truncation.output_lines < truncation.total_lines.min(DEFAULT_MAX_LINES) {
            truncate::TruncatedBy::Bytes
        } else {
            truncate::TruncatedBy::Lines
        });
    }
    let limit_text = truncate::format_size(DEFAULT_MAX_BYTES);
    let mut details = None;
    let output = if truncation.first_line_exceeds_limit {
        // The first line alone is over the limit: point the model at a shell command that shows
        // part of it.
        let size = truncate::format_size(first_line_bytes);
        details = Some(json!({"truncation": truncation}));
        format!(
            "[Line {start_display} is {size}, exceeds {limit_text} limit. Use bash: sed -n '{start_display}p' {} | head -c {DEFAULT_MAX_BYTES}]",
            arguments.path
        )
    } else if truncation.truncated {
        let end_display = start_display + truncation.output_lines - 1;
        let next_offset = end_display + 1;
        let notice = match truncation.truncated_by {
            Some(truncate::TruncatedBy::Lines) => {
                format!("[Showing lines {start_display}-{end_display} of {total_file_lines}. Use offset={next_offset} to continue.]")
            }
            _ => format!(
                "[Showing lines {start_display}-{end_display} of {total_file_lines} ({limit_text} limit). Use offset={next_offset} to continue.]"
            ),
        };
        let text = format!("{}\n\n{notice}", truncation.content);
        details = Some(json!({"truncation": truncation}));
        text
    } else if let Some(lines) = user_limited_lines.filter(|lines| start_line + lines < total_file_lines) {
        let remaining = total_file_lines - (start_line + lines);
        let next_offset = start_line + lines + 1;
        format!("{}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]", truncation.content)
    } else {
        truncation.content
    };
    Ok(ToolOutput { content: vec![Content::text(output)], details, is_error: false })
}
