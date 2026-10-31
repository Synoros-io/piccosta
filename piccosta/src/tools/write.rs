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
        Some("Create or overwrite files")
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        &["Use write only for new files or complete rewrites."]
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: Arguments = serde_json::from_value(Value::Object(arguments)).map_err(|error| error.to_string())?;
            let absolute = paths::resolve(&arguments.path, &context.cwd);
            let _turn = context.shared.mutations.lock(&absolute).await;
            // An abort is only checked between steps: a write already started finishes, so the
            // queue never lets the next change in while this one is still landing (write.ts:68-72).
            let aborted = || context.cancel.is_cancelled().then(|| "Operation aborted".to_string());
            if let Some(message) = aborted() {
                return Err(message);
            }
            if let Some(parent) = absolute.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|error| paths::fs_error(&error, "mkdir", Some(parent)))?;
            }
            if let Some(message) = aborted() {
                return Err(message);
            }
            tokio::fs::write(&absolute, arguments.content.as_bytes())
                .await
                .map_err(|error| paths::fs_error(&error, "open", Some(&absolute)))?;
            if let Some(message) = aborted() {
                return Err(message);
            }
            context.shared.seen.record(&absolute, "write");
            Ok(ToolOutput { content: vec![Content::text(format!("Successfully wrote to {}", arguments.path))], details: None, is_error: false })
        })
    }
}
