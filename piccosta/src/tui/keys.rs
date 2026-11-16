// One registry, ordered the way people look for controls: panes, sessions, scratchpad, input,
// lists, keys. Each row gives its context, group, settings name, plain description and defaults.
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use crate::config::Settings;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Context { Global, Pane, Picker, Selection, Editor, Scratch, Keys }
use Context::*;
const DEFAULTS: &[(Context, &str, &str, &str, &[&str])] = &[
    (Pane, "panes", "close-pane", "close this pane", &["ctrl+]", "ctrl+5", "f4"]),
    (Pane, "panes", "cycle-focus", "switch to the next pane", &["f6"]),
    (Pane, "panes", "focus-left", "focus the previous pane", &["alt+left"]),
    (Pane, "panes", "focus-right", "focus the next pane", &["alt+right"]),
    (Pane, "panes", "split", "open a new pane beside this one", &["ctrl+\\", "ctrl+4"]),
    (Pane, "panes", "cycle-layout", "switch grid, columns or rows", &["f7"]),
    (Pane, "panes", "pane-wider", "make this pane wider", &["ctrl+alt+right"]),
    (Pane, "panes", "pane-narrower", "make this pane narrower", &["ctrl+alt+left"]),
    (Pane, "panes", "pane-taller", "make this pane taller", &["ctrl+alt+down"]),
    (Pane, "panes", "pane-shorter", "make this pane shorter", &["ctrl+alt+up"]),
    (Pane, "panes", "toggle-sidebar", "show or hide the session sidebar", &["f9"]),
    (Pane, "panes", "narrow-sidebar", "make the sidebar narrower", &["alt+,"]),
    (Pane, "panes", "widen-sidebar", "make the sidebar wider", &["alt+."]),
    (Pane, "sessions", "sessions", "choose a session to open beside this one", &["ctrl+s"]),
    (Pane, "sessions", "new-session", "start a session in a chosen folder", &["ctrl+n"]),
    (Pane, "sessions", "previous-session", "show the previous sidebar session", &["alt+up", "pageup", "ctrl+pageup", "shift+pageup", "ctrl+shift+pageup"]),
    (Pane, "sessions", "next-session", "show the next sidebar session", &["alt+down", "pagedown", "ctrl+pagedown", "shift+pagedown", "ctrl+shift+pagedown"]),
    (Pane, "sessions", "delete-session", "press twice on an empty draft to trash this session", &["delete"]),
    (Scratch, "scratchpad", "scratch-append", "write a paragraph to append", &["ctrl+e"]),
    (Scratch, "scratchpad", "scratch-edit", "open the scratchpad in your editor", &["f2"]),
    (Scratch, "scratchpad", "scratch-command", "enter a slash command without changing your draft", &["/"]),
    (Pane, "input", "submit", "send a message or submit the current input", &["enter"]),
    (Pane, "input", "send-now", "send queued input now instead of waiting", &["ctrl+x ctrl+s"]),
    (Pane, "input", "newline", "insert a new line without sending", &["alt+enter"]),
    (Pane, "input", "stop", "stop this turn or leave scratchpad editing", &["esc"]),
    (Pane, "input", "cancel", "stop, clear the draft, or press twice to quit", &["ctrl+c"]),
    (Pane, "input", "quit-empty", "press twice on an empty draft to quit", &["ctrl+d"]),
    (Pane, "input", "paste", "paste the clipboard into the input", &["ctrl+v"]),
    (Pane, "input", "clear-draft", "clear the draft and keep it in history", &["ctrl+shift+delete"]),
    (Pane, "input", "complete", "complete a slash command name", &["tab"]),
    (Pane, "input", "history-next", "recall newer input or scroll the scratchpad down", &["down"]),
    (Pane, "input", "history-previous", "recall older input or scroll the scratchpad up", &["up"]),
    (Pane, "input", "scroll-up", "scroll this pane up", &["alt+pageup"]),
    (Pane, "input", "scroll-down", "scroll this pane down", &["alt+pagedown"]),
    (Pane, "input", "toggle-tools", "expand or fold tool output", &["ctrl+o"]),
    (Pane, "input", "toggle-thinking", "expand or fold model thinking", &["ctrl+t"]),
    (Selection, "input", "copy-selection", "copy the selected text", &["ctrl+c"]),
    (Selection, "input", "select-left", "extend the selection left", &["shift+left"]),
    (Selection, "input", "select-right", "extend the selection right", &["shift+right"]),
    (Selection, "input", "select-up", "extend the selection up", &["shift+up"]),
    (Selection, "input", "select-down", "extend the selection down", &["shift+down"]),
    (Selection, "input", "select-home", "extend the selection to the row start", &["shift+home"]),
    (Selection, "input", "select-end", "extend the selection to the row end", &["shift+end"]),
    (Editor, "input", "line-start", "move to the start of this line", &["home", "shift+home"]),
    (Editor, "input", "line-end", "move to the end of this line", &["ctrl+e", "end", "shift+end"]),
    (Editor, "input", "cut-line-start", "delete text before the cursor on this line", &["ctrl+u"]),
    (Editor, "input", "cut-line-end", "delete text after the cursor on this line", &["ctrl+k"]),
    (Editor, "input", "cut-whitespace-word", "delete the whitespace-delimited word before the cursor", &["ctrl+w"]),
    (Editor, "input", "cut-word-back", "delete the word before the cursor", &["alt+backspace", "ctrl+backspace", "ctrl+h"]),
    (Editor, "input", "cut-word-forward", "delete the word after the cursor", &["alt+d"]),
    (Editor, "input", "word-back", "move back one word", &["alt+b", "ctrl+left", "ctrl+shift+left"]),
    (Editor, "input", "word-forward", "move forward one word", &["alt+f", "ctrl+right", "ctrl+shift+right"]),
    (Editor, "input", "undo", "undo the last input edit", &["ctrl+_", "ctrl+7", "ctrl+/"]),
    (Editor, "input", "backspace", "delete the character before the cursor", &["backspace"]),
    (Editor, "input", "delete-character", "delete the character after the cursor", &["delete", "ctrl+d"]),
    (Editor, "input", "cursor-left", "move left in the input", &["left", "shift+left"]),
    (Editor, "input", "cursor-right", "move right in the input", &["right", "shift+right"]),
    (Editor, "input", "cursor-up", "move up in the input", &["up", "shift+up"]),
    (Editor, "input", "cursor-down", "move down in the input", &["down", "shift+down"]),
    (Picker, "lists", "picker-choose", "choose the selected item or confirm captured keys", &["enter"]),
    (Picker, "lists", "picker-cancel", "close this list or cancel captured keys", &["esc", "ctrl+c"]),
    (Picker, "lists", "picker-delete", "ask to delete the selected item", &["delete", "ctrl+d"]),
    (Picker, "lists", "picker-clear", "clear the search text", &["ctrl+u"]),
    (Picker, "lists", "picker-up", "select the previous item", &["up"]),
    (Picker, "lists", "picker-down", "select the next item", &["down"]),
    (Picker, "lists", "picker-backspace", "delete a character from the search", &["backspace"]),
    (Picker, "lists", "picker-paste", "paste into the search", &["ctrl+v"]),
    (Global, "keys", "show-keys", "find and change keyboard controls", &["f1"]),
    (Keys, "keys", "reset-binding", "restore the selected action's defaults", &["ctrl+r"]),
    (Keys, "keys", "disable-binding", "remove every key for the selected action", &["delete"]),
];

#[derive(Clone)]
struct Binding { context: Context, action: &'static str, sequence: Vec<KeyEvent>, label: String }
#[derive(Clone)]
pub(super) struct Keys { bindings: Vec<Binding>, pending: Vec<KeyEvent>, pub context: Option<Context>, pub problems: Vec<String> }
pub(super) enum Match { Action(&'static str), Prefix(String), None }

fn normalize(mut key: KeyEvent) -> KeyEvent {
    if key.code == KeyCode::BackTab { key.code = KeyCode::Tab; key.modifiers |= KeyModifiers::SHIFT; }
    if let KeyCode::Char(character) = key.code {
        if character.is_ascii_uppercase() { key.code = KeyCode::Char(character.to_ascii_lowercase()); key.modifiers |= KeyModifiers::SHIFT; }
        if !character.is_ascii_alphabetic() { key.modifiers.remove(KeyModifiers::SHIFT); }
    }
    KeyEvent::new(key.code, key.modifiers)
}

fn parse(text: &str) -> Result<Vec<KeyEvent>, String> {
    let mut sequence = Vec::new();
    for token in text.split_whitespace() {
        let mut parts = token.split('+').peekable();
        let mut modifiers = KeyModifiers::empty();
        while parts.clone().count() > 1 {
            modifiers |= match parts.next().unwrap().to_ascii_lowercase().as_str() {
                "ctrl" | "control" => KeyModifiers::CONTROL, "alt" | "option" => KeyModifiers::ALT,
                "shift" => KeyModifiers::SHIFT, "super" | "command" | "windows" => KeyModifiers::SUPER,
                "hyper" => KeyModifiers::HYPER, "meta" => KeyModifiers::META,
                _ => return Err(format!("bad modifier in {token:?}")),
            };
        }
        let name = parts.next().unwrap_or_default().to_ascii_lowercase();
        let code = match name.as_str() {
            "enter" | "return" => KeyCode::Enter, "esc" | "escape" => KeyCode::Esc,
            "tab" => KeyCode::Tab, "backtab" => { modifiers |= KeyModifiers::SHIFT; KeyCode::Tab },
            "space" => KeyCode::Char(' '), "plus" => KeyCode::Char('+'),
            "backspace" => KeyCode::Backspace, "delete" | "del" => KeyCode::Delete,
            "insert" => KeyCode::Insert, "home" => KeyCode::Home, "end" => KeyCode::End,
            "left" => KeyCode::Left, "right" => KeyCode::Right, "up" => KeyCode::Up, "down" => KeyCode::Down,
            "pageup" | "pgup" => KeyCode::PageUp, "pagedown" | "pgdn" => KeyCode::PageDown,
            name if name.starts_with('f') && name.len() > 1 => KeyCode::F(name[1..].parse::<u8>().ok().filter(|number| (1..=24).contains(number)).ok_or_else(|| format!("bad function key {token:?}"))?),
            name if name.chars().count() == 1 => KeyCode::Char(name.chars().next().unwrap()),
            _ => return Err(format!("bad key {token:?}")),
        };
        sequence.push(normalize(KeyEvent::new(code, modifiers)));
    }
    if sequence.is_empty() { return Err("empty key sequence (use [] to disable)".into()); }
    Ok(sequence)
}

impl Keys {
    pub fn load(settings: &Settings) -> Self {
        let mut keys = Self { bindings: Vec::new(), pending: Vec::new(), context: None, problems: Vec::new() };
        let configured = settings.get("keys");
        if configured.is_some_and(|value| !value.is_object()) { keys.problems.push("keys must be an object".into()); }
        for name in configured.and_then(Value::as_object).into_iter().flat_map(|values| values.keys()) {
            if !DEFAULTS.iter().any(|(_, _, action, _, _)| action == name) { keys.problems.push(format!("keys.{name}: unknown action")); }
        }
        for custom in [true, false] {
            for &(_, _, action, _, _) in DEFAULTS {
                let value = configured.and_then(|value| value.get(action));
                if value.is_some() == custom && let Err(error) = keys.assign(action, value) { keys.problems.push(format!("keys.{action}: {error}")); }
            }
        }
        keys
    }

    // Loading, rebinding and reset share this validator. Contexts may deliberately reuse keys;
    // a global binding or a same-context sequence must not shadow another binding or prefix.
    fn assign(&mut self, action: &str, value: Option<&Value>) -> Result<(), String> {
        let &(context, _, action, _, defaults) = DEFAULTS.iter().find(|(_, _, name, _, _)| *name == action).ok_or("unknown action")?;
        let labels: Vec<_> = match value {
            None => defaults.iter().map(|label| label.to_string()).collect(), Some(Value::String(label)) => vec![label.clone()],
            Some(Value::Array(labels)) if labels.iter().all(Value::is_string) => labels.iter().map(|label| label.as_str().unwrap().to_string()).collect(),
            _ => return Err("expected a key string or array".into()),
        };
        self.bindings.retain(|binding| binding.action != action);
        // A clash costs only the clashing key: the action's other keys still bind.
        let mut clashes = Vec::new();
        for label in labels {
            let sequence = parse(&label)?;
            if let Some(other) = self.bindings.iter().find(|binding| (binding.context == context || matches!((context, binding.context), (Global, _) | (_, Global))) && (binding.sequence.starts_with(&sequence) || sequence.starts_with(&binding.sequence))) {
                clashes.push(format!("{label:?} overlaps {} ({})", other.action, other.label));
                continue;
            }
            self.bindings.push(Binding { context, action, sequence, label });
        }
        if clashes.is_empty() { Ok(()) } else { Err(clashes.join("; ")) }
    }

    pub fn validate(&self, action: &str, binding: &Value) -> Result<(), String> { self.clone().assign(action, Some(binding)) }
    pub fn defaults(action: &str) -> Value { DEFAULTS.iter().find(|(_, _, name, _, _)| *name == action).map(|(_, _, _, _, defaults)| serde_json::json!(defaults)).unwrap_or_default() }
    pub fn action(item: &str) -> Option<&'static str> { let name = item.split_whitespace().nth(1)?; DEFAULTS.iter().find(|(_, _, action, _, _)| *action == name).map(|(_, _, action, _, _)| *action) }
    pub fn label(&self, action: &str) -> String { let labels: Vec<_> = self.bindings.iter().filter(|binding| binding.action == action).map(|binding| binding.label.as_str()).collect(); if labels.is_empty() { "unbound".into() } else { labels.join(" / ") } }
    pub fn short(&self, action: &str) -> &str { self.bindings.iter().filter(|binding| binding.action == action).map(|binding| binding.label.as_str()).min_by_key(|label| label.len()).unwrap_or("unbound") }
    pub fn rows(&self) -> Vec<String> { DEFAULTS.iter().map(|(_, group, action, about, _)| format!("{group:<11} {action:<22} {:<25} {about}", self.label(action))).collect() }
    pub fn help(&self) -> String { format!("Keys: F1 or /keys; /keys reload rereads settings. Spaces separate chord keys; [] disables an action.\n{}\n{}", self.rows().join("\n"), self.problems.join("\n")) }

    pub fn key_name(key: KeyEvent) -> String {
        let key = normalize(key);
        let name = match key.code { KeyCode::Backspace => "backspace".into(), KeyCode::Delete => "delete".into(), KeyCode::Char('+') => "plus".into(), code => code.to_string().replace(' ', "").to_ascii_lowercase() };
        let modifiers = key.modifiers.to_string().to_ascii_lowercase().replace("control", "ctrl");
        format!("{modifiers}{}{name}", if modifiers.is_empty() { "" } else { "+" })
    }
    pub fn suffix_len(&self, action: &str, keys: &[KeyEvent]) -> usize {
        self.bindings.iter().filter(|binding| binding.action == action).find(|binding| keys.iter().rev().map(|key| normalize(*key)).zip(binding.sequence.iter().rev().copied()).all(|(left, right)| left == right) && keys.len() >= binding.sequence.len()).map_or(0, |binding| binding.sequence.len())
    }

    // Keep an unfinished sequence in its original context. A nonmatching suffix is consumed,
    // so it cannot accidentally send a draft, but capture mode retains every pressed key.
    pub fn resolve(&mut self, contexts: &[Context], key: KeyEvent) -> Match {
        if self.context.is_some_and(|context| !contexts.contains(&context)) { self.pending.clear(); self.context = None; }
        let was_pending = !self.pending.is_empty();
        self.pending.push(normalize(key));
        for &context in contexts {
            if was_pending && self.context != Some(context) { continue; }
            for binding in &self.bindings {
                if binding.context != context || !binding.sequence.starts_with(&self.pending) { continue; }
                self.context = Some(context);
                if binding.sequence == self.pending { self.pending.clear(); return Match::Action(binding.action); }
                return Match::Prefix(format!("Complete {}: {}", binding.action, binding.label));
            }
        }
        self.pending.clear();
        if was_pending { Match::Prefix("Key sequence cancelled".into()) } else { Match::None }
    }
}
