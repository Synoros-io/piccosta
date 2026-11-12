// The write tool: create or replace a whole file, making its parent directories as needed.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/tools/write.ts. The
// content is written as given, with no line-ending or BOM handling, inside the file's mutation
// queue so it can't interleave with an edit of the same file.

use super::{paths, Tool, ToolContext, ToolFuture, ToolOutput};
use crate::message::{Content, ToolDefinition};
use serde::Deserialize;
use serde_json::{json, Map, Value};

pub struct WriteTool;

#[derive(Deserialize)]
struct Arguments {
    path: String,
    content: String,
}

impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "write"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write".into(),
            description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.".into(),
            parameters: json!({
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                    "content": {"type": "string", "description": "Content to write to the file"}
                }
            }),
            constrained_sampling: Some(json!({"type": "json_schema", "strict": "prefer"})),
        }
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("create a file, or replace one whole")
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        &["Change part of a file with edit; write is for new files and whole rewrites"]
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: Arguments = serde_json::from_value(Value::Object(arguments)).map_err(|error| error.to_string())?;
            let absolute = paths::resolve(&arguments.path, &context.cwd);
            super::mutate(absolute, context, move |absolute, context, board| {
                if context.cancel.is_cancelled() { return Err("Operation aborted".into()); }
                if board && arguments.content.len() > crate::scratchpad::MAX_BYTES { return Err(format!("scratchpad exceeds {} bytes", crate::scratchpad::MAX_BYTES)); }
                if let Some(parent) = absolute.parent() { std::fs::create_dir_all(parent).map_err(|error| paths::fs_error(&error, "mkdir", Some(parent)))?; }
                if context.cancel.is_cancelled() { return Err("Operation aborted".into()); }
                paths::replace(absolute, arguments.content.as_bytes()).map_err(|error| paths::fs_error(&error, "open", Some(absolute)))?;
                context.shared.seen.remember(absolute, &arguments.content, "write");
                Ok(ToolOutput { content: vec![Content::text(format!("Successfully wrote to {}", arguments.path))], details: board.then(|| json!({"scratchpad": crate::scratchpad::stamp(absolute, Some(&arguments.content)).ok()})), is_error: false })
            }).await
        })
    }
}
