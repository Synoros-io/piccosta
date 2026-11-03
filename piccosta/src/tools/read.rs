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
            let task_context = context.clone();
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
    paths::access(absolute, libc::R_OK).map_err(|error| paths::fs_error(&error, "access", Some(absolute)))?;
    let mut file = std::fs::File::open(absolute).map_err(|error| paths::fs_error(&error, "open", Some(absolute)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|error| paths::fs_error(&error, "read", None))?;

    if let Some(mime_type) = image::sniff_mime_type(&bytes[..bytes.len().min(image::SNIFF_BYTES)]) {
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

    let text = String::from_utf8_lossy(&bytes);
    let output = read_text(&text, arguments)?;
    context.shared.seen.remember(absolute, &text, "read");
    Ok(output)
}

// Input: the file's text and the call's offset and limit. Output: the window the model sees, with
// pi's notice for whatever was left out (read.ts:155-201).
fn read_text(text: &str, arguments: &Arguments) -> Result<ToolOutput, String> {
    let all_lines: Vec<&str> = text.split('\n').collect();
    let total_file_lines = all_lines.len();
    // 1-indexed, and 0 or nothing means the first line. A fractional offset counts as its whole
    // part, as JavaScript's slice takes it.
    let start_line = match arguments.offset {
        Some(offset) if offset != 0.0 => (offset - 1.0).max(0.0).trunc() as usize,
        _ => 0,
    };
    let start_display = start_line + 1;
    if start_line >= total_file_lines {
        let offset = crate::js::number_string(arguments.offset.unwrap_or(0.0));
        return Err(format!("Offset {offset} is beyond end of file ({total_file_lines} lines total)"));
    }

    // A limit from the model is applied first; truncation then cuts within it.
    let (selected, user_limited_lines) = match arguments.limit {
        Some(limit) => {
            let limit = limit.trunc().max(0.0) as usize;
            let end_line = start_line.saturating_add(limit).min(total_file_lines);
            (all_lines[start_line..end_line].join("\n"), Some(end_line - start_line))
        }
        None => (all_lines[start_line..].join("\n"), None),
    };

    let truncation = truncate::truncate_head(&selected, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let limit_text = truncate::format_size(DEFAULT_MAX_BYTES);
    let mut details = None;
    let output = if truncation.first_line_exceeds_limit {
        // The first line alone is over the limit: point the model at a shell command that shows
        // part of it.
        let size = truncate::format_size(all_lines[start_line].len());
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
