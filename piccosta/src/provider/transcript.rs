// System messages across a conversation: replaying them into the current prompt and tool set,
// and rendering them as text for the wire.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/utils/transcript.ts and text.ts.
//
// A conversation's first system message is the prompt. Later ones change it: their text is added,
// their sections replace sections of the same name (None removes one), and their tool lists add or
// remove tools. APIs that accept system messages mid-conversation get each one in place; the
// others get one leading message holding the replayed result.

use crate::message::{Content, Message, SystemMessage, TextOrBlocks, ToolDefinition};
use indexmap::IndexMap;

// The text of message content: a string as is, or the text blocks joined by `separator`.
pub fn content_text(content: &TextOrBlocks, separator: &str) -> String {
    match content {
        TextOrBlocks::Text(text) => text.clone(),
        TextOrBlocks::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                Content::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(separator),
    }
}

// A system message as a whole prompt: its text, then each section that isn't removed, joined by
// blank lines with empty parts left out (text.ts:15-22).
pub fn system_message_text(message: &SystemMessage) -> String {
    let mut parts = vec![content_text(&message.content, "\n")];
    if let Some(sections) = &message.sections {
        for text in sections.values().flatten() {
            parts.push(text.clone());
        }
    }
    parts.retain(|part| !part.is_empty());
    parts.join("\n\n")
}

// A later system message as text for an API that takes system messages mid-conversation. Section
// changes are named so the model can relate them to the prompt it already has (text.ts:28-40).
pub fn render_system_message_update(message: &SystemMessage) -> String {
    let mut parts = Vec::new();
    let text = content_text(&message.content, "\n");
    if !text.is_empty() {
        parts.push(text);
    }
    if let Some(sections) = &message.sections {
        for (name, value) in sections {
            parts.push(match value {
                None => format!("Removed system prompt section \"{name}\"."),
                Some(text) => format!("Updated system prompt section \"{name}\":\n\n{text}"),
            });
        }
    }
    parts.join("\n\n")
}

// The tools in force after replaying every system message, in the order they were last added.
pub fn current_tools<'a>(messages: impl Iterator<Item = &'a SystemMessage>) -> Vec<ToolDefinition> {
    let mut tools: IndexMap<String, ToolDefinition> = IndexMap::new();
    for message in messages {
        for removed in message.tools_removed.iter().flatten() {
            tools.shift_remove(&removed.name);
        }
        for added in message.tools_added.iter().flatten() {
            // A redefinition keeps the tool's place, as a JavaScript Map's set does; a tool removed
            // and added again goes to the end.
            tools.insert(added.name.clone(), added.clone());
        }
    }
    tools.into_values().collect()
}

fn system_messages(messages: &[Message]) -> impl Iterator<Item = &SystemMessage> {
    messages.iter().filter_map(|message| match message {
        Message::System(system) => Some(system),
        _ => None,
    })
}

// Every system message replayed into one (transcript.ts:75-99): their texts joined by blank lines,
// sections patched by name, the current tools, and the first one's timestamp. None when there is
// no system message and no tool.
pub fn current_system_message(messages: &[Message]) -> Option<SystemMessage> {
    let mut texts = Vec::new();
    let mut sections: IndexMap<String, String> = IndexMap::new();
    let mut timestamp = None;
    for message in system_messages(messages) {
        timestamp.get_or_insert(message.timestamp);
        let text = content_text(&message.content, "\n");
        if !text.is_empty() {
            texts.push(text);
        }
        for (name, value) in message.sections.iter().flatten() {
            match value {
                None => {
                    sections.shift_remove(name);
                }
                Some(text) => {
                    sections.insert(name.clone(), text.clone());
                }
            }
        }
    }
    let tools = current_tools(system_messages(messages));
    if timestamp.is_none() && tools.is_empty() {
        return None;
    }
    Some(SystemMessage {
        content: TextOrBlocks::Text(texts.join("\n\n")),
        sections: (!sections.is_empty()).then(|| sections.into_iter().map(|(name, text)| (name, Some(text))).collect()),
        tools_added: (!tools.is_empty()).then_some(tools),
        tools_removed: None,
        timestamp: timestamp.unwrap_or(0),
    })
}

// The conversation for an API, with system messages kept in place when it accepts them there,
// otherwise replayed into one leading message and dropped from the rest (transcript.ts:112-125).
pub fn resolve_transcript(messages: &[Message], supports_mid_conversation_system: bool) -> Vec<Message> {
    if supports_mid_conversation_system {
        return messages.to_vec();
    }
    let head = current_system_message(messages);
    let rest = messages.iter().filter(|message| !matches!(message, Message::System(_))).cloned();
    head.map(Message::System).into_iter().chain(rest).collect()
}

// How an API can express tool changes after the first request (transcript.ts:234-245). piccosta's
// Responses models (Codex) take mid-conversation system messages and `additional_tools`, so they
// declare every tool in place ("changes"); everything else sends the current tool list each time
// ("none"). pi's third mode, synthetic tool search for models that can only add tools, has no
// piccosta model that needs it and is left out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSupport {
    None,
    // Tools can be added at a later message but not removed or redefined (Kimi K3 on Chat
    // Completions).
    Additions,
    Changes,
}

// The tools to send in the request's top-level list, and how many leading system messages those
// already cover (transcript.ts:261-271). With Changes the list is empty and every system message
// carries its own tool changes.
pub fn request_tools(messages: &[Message], support: ToolSupport) -> (Vec<ToolDefinition>, usize) {
    match support {
        ToolSupport::Changes => (Vec::new(), 0),
        ToolSupport::None => (current_tools(system_messages(messages)), system_messages(messages).count()),
        // With additions, a removal or a redefinition can't be said in place, so the top-level list
        // is the tool state after the last such change, and later additions stay where they are
        // (transcript.ts:205-262). A name declared before counts as a redefinition even after it
        // was removed: Kimi K3 ignores a removed tool that a later message declares again.
        ToolSupport::Additions => {
            let systems: Vec<&SystemMessage> = system_messages(messages).collect();
            let mut declared = std::collections::HashSet::new();
            let mut last_non_additive: Option<usize> = None;
            for (index, message) in systems.iter().enumerate() {
                if message.tools_removed.as_ref().is_some_and(|removed| !removed.is_empty()) {
                    last_non_additive = Some(index);
                }
                for tool in message.tools_added.iter().flatten() {
                    if !declared.insert(tool.name.clone()) {
                        last_non_additive = Some(index);
                    }
                }
            }
            let leading = usize::from(matches!(messages.first(), Some(Message::System(_))));
            let folded = leading.max(last_non_additive.map_or(0, |index| index + 1));
            (current_tools(systems[..folded.min(systems.len())].iter().copied()), folded)
        }
    }
}
