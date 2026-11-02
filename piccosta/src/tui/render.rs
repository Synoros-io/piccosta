// A session's messages as screen rows: the model's text as Markdown with its TeX typeset, one
// line per tool call with its output under it, the user's prompts set off by a bar. Every row is
// already wrapped to the pane's width, so a transcript's height is its row count.

use super::tex::{self, Formula, Tex};
use super::theme;
use crate::message::{AssistantMessage, Content, StopReason};
use pulldown_cmark::{CodeBlockKind, Event as Md, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::{Map, Value};
use std::collections::HashMap;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub type Rows = Vec<Line<'static>>;

// How many rows of a tool's output show while tools are folded, and the most they ever show.
const TOOL_OPEN_ROWS: usize = 400;
// Arguments a tool call is best summed up by, in the order they're looked for.
const SUMMARY_KEYS: [&str; 8] = ["command", "path", "file_path", "pattern", "query", "url", "prompt", "description"];
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// An assistant message's content as the transcript draws it; a streaming one is built from its
// deltas in the same shape.
pub enum Block {
    Text(String),
    Thinking(String),
    Call { id: String, name: String, arguments: Map<String, Value> },
}

pub fn blocks(message: &AssistantMessage) -> Vec<Block> {
    message
        .content
        .iter()
        .filter_map(|block| match block {
            Content::Text { text, .. } => Some(Block::Text(text.clone())),
            Content::Thinking { thinking, .. } => Some(Block::Thinking(thinking.clone())),
            Content::ToolCall { id, name, arguments, .. } => Some(Block::Call { id: id.clone(), name: name.clone(), arguments: arguments.clone() }),
            Content::Image { .. } => None,
        })
        .collect()
}

// A tool call's outcome so far.
#[derive(Default, Clone)]
pub struct ToolView {
    pub text: String,
    pub diff: Option<String>,
    pub is_error: bool,
    pub running: bool,
}

// What the user chose to see.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Flags {
    pub tools_open: bool,
    pub thinking_open: bool,
}

// The user's prompt, behind a bar.
pub fn user(text: &str, width: u16, tex: &mut Tex) -> Rows {
    let mut writer = Writer::new(tex, width);
    writer.levels.push(Level::fixed("▌ ", Style::new().fg(theme::USER)));
    for line in clean(text).lines() {
        writer.begin_row();
        writer.text(line, Style::new());
        writer.flush_line(true);
    }
    writer.blank();
    writer.rows
}

// A note from outside the conversation: a background job's finish, a compaction.
pub fn note(text: &str, width: u16, tex: &mut Tex) -> Rows {
    news(text, width, tex, false)
}

// A notice for the model (a background job's end, a lane's news): its first line, dim, with how
// many more it has; all of them with tools open (ctrl+o). It's for the model to read, so it stays
// out of the way of the conversation.
pub fn news(text: &str, width: u16, tex: &mut Tex, open: bool) -> Rows {
    let mut writer = Writer::new(tex, width);
    writer.levels.push(Level::fixed("◆ ", Style::new().fg(theme::DIM)));
    let text = clean(text);
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    let shown = if open { TOOL_OPEN_ROWS } else { 1 };
    for (index, line) in lines.iter().take(shown).enumerate() {
        let more = if index == 0 && !open && lines.len() > 1 { format!(" · {} more lines (ctrl+o)", lines.len() - 1) } else { String::new() };
        let room = (width as usize).saturating_sub(2 + more.width());
        writer.text(&if open { line.to_string() } else { fit(line, room) }, Style::new().fg(theme::DIM));
        writer.text(&more, Style::new().fg(theme::DIM));
        writer.flush_line(true);
    }
    writer.blank();
    writer.rows
}

// A slash command's output (/status, /diff, /agents …), every line of it, diff lines colored.
pub fn report(text: &str, width: u16, tex: &mut Tex) -> Rows {
    let mut writer = Writer::new(tex, width);
    writer.levels.push(Level::fixed("  ", Style::new()));
    for line in clean(text).lines() {
        let style = match line.chars().next() {
            Some('+') if !line.starts_with("+++") => Style::new().fg(theme::ADDED),
            Some('-') if !line.starts_with("---") => Style::new().fg(theme::REMOVED),
            _ => Style::new().fg(theme::DIM),
        };
        // Leading spaces are kept (a diff's context lines, indented lists); the rest wraps at spaces.
        let rest = line.trim_start_matches(' ');
        writer.word(&line[..line.len() - rest.len()], style);
        writer.text(rest, style);
        writer.flush_line(true);
    }
    writer.blank();
    writer.rows
}

// The model's response: thinking, text and tool calls in order, then how it ended when that
// wasn't normally. `frame` turns running tools' spinners.
pub fn assistant(blocks: &[Block], ended: Option<(&StopReason, Option<&str>)>, tools: &HashMap<String, ToolView>, flags: Flags, frame: usize, width: u16, tex: &mut Tex) -> Rows {
    let mut writer = Writer::new(tex, width);
    for block in blocks {
        match block {
            Block::Thinking(text) if !text.trim().is_empty() => {
                let style = Style::new().fg(theme::DIM).add_modifier(Modifier::ITALIC);
                writer.levels.push(Level::fixed("  ", style));
                if flags.thinking_open {
                    for line in clean(text).lines() {
                        writer.text(line, style);
                        writer.flush_line(true);
                    }
                } else {
                    let first = clean(text).lines().find(|line| !line.trim().is_empty()).unwrap_or("").to_string();
                    writer.word(&fit(&format!("∴ {first}"), writer.room()), style);
                    writer.flush_line(true);
                }
                writer.levels.pop();
                writer.blank();
            }
            Block::Thinking(_) => {}
            Block::Text(text) => writer.markdown(text),
            Block::Call { id, name, arguments } => {
                let view = tools.get(id).cloned().unwrap_or(ToolView { running: ended.is_none(), ..Default::default() });
                writer.tool(name, arguments, &view, flags, frame);
            }
        }
    }
    match ended {
        Some((StopReason::Error, message)) => {
            writer.text(&format!("✗ {}", message.unwrap_or("the request failed")), Style::new().fg(theme::ERROR));
            writer.flush_line(true);
            writer.blank();
        }
        Some((StopReason::Aborted, _)) => {
            writer.text("⏹ stopped", Style::new().fg(theme::DIM));
            writer.flush_line(true);
            writer.blank();
        }
        _ => {}
    }
    writer.rows
}

// One level of indentation: what starts its first row and every row after.
struct Level {
    first: String,
    rest: String,
    style: Style,
    fresh: bool,
}

impl Level {
    fn fixed(text: &str, style: Style) -> Level {
        Level { first: text.into(), rest: text.into(), style, fresh: true }
    }
}

struct Writer<'t> {
    tex: &'t mut Tex,
    width: usize,
    rows: Rows,
    line: Vec<Span<'static>>,
    used: usize,
    // Whether the row has anything past its indentation.
    content: bool,
    levels: Vec<Level>,
    styles: Vec<Style>,
    // Open lists: the next number of an ordered one.
    lists: Vec<Option<u64>>,
    code: bool,
    // A table being read: its rows of cells.
    table: Option<Vec<Vec<String>>>,
}

impl<'t> Writer<'t> {
    fn new(tex: &'t mut Tex, width: u16) -> Writer<'t> {
        Writer { tex, width: usize::from(width).max(8), rows: Vec::new(), line: Vec::new(), used: 0, content: false, levels: Vec::new(), styles: vec![Style::new()], lists: Vec::new(), code: false, table: None }
    }

    fn style(&self) -> Style {
        self.styles.last().copied().unwrap_or_default()
    }

    fn indent(&self) -> usize {
        self.levels.iter().map(|level| if level.fresh { level.first.width() } else { level.rest.width() }).sum()
    }

    // Columns left for text after the indentation.
    fn room(&self) -> usize {
        self.width.saturating_sub(self.indent()).max(1)
    }

    fn begin_row(&mut self) {
        if !self.line.is_empty() {
            return;
        }
        for level in &mut self.levels {
            let text = if level.fresh { &level.first } else { &level.rest };
            self.used += text.width();
            self.line.push(Span::styled(text.clone(), level.style));
            level.fresh = false;
        }
    }

    // End the row; an empty one is kept only when `keep_empty`.
    fn flush_line(&mut self, keep_empty: bool) {
        if self.content || (keep_empty && !self.line.is_empty()) {
            self.rows.push(Line::from(std::mem::take(&mut self.line)));
        }
        self.line.clear();
        self.used = 0;
        self.content = false;
    }

    // One empty row between blocks, never two.
    fn blank(&mut self) {
        self.flush_line(false);
        if self.rows.last().is_some_and(|row| row.width() > 0) {
            self.rows.push(Line::default());
        }
    }

    // A piece that doesn't break at spaces: on this row if it fits, else on the next, split
    // where it's wider than a whole row.
    fn word(&mut self, text: &str, style: Style) {
        self.piece(text.to_string(), text.width(), style);
    }

    fn piece(&mut self, text: String, width: usize, style: Style) {
        self.begin_row();
        if self.content && self.used + width > self.width {
            self.flush_line(false);
            self.begin_row();
        }
        if width <= self.width.saturating_sub(self.used) || width == 0 {
            self.used += width;
            self.line.push(Span::styled(text, style));
            self.content = true;
            return;
        }
        let mut part = String::new();
        let mut part_width = 0;
        for character in text.chars() {
            let character_width = character.width().unwrap_or(0);
            if self.used + part_width + character_width > self.width && !part.is_empty() {
                self.line.push(Span::styled(std::mem::take(&mut part), style));
                self.content = true;
                self.flush_line(false);
                self.begin_row();
                part_width = 0;
            }
            part.push(character);
            part_width += character_width;
        }
        self.used += part_width;
        self.line.push(Span::styled(part, style));
        self.content = true;
    }

    // Text that wraps at spaces. A space that would start a row is dropped.
    fn text(&mut self, text: &str, style: Style) {
        for (index, word) in text.split(' ').enumerate() {
            if index > 0 && self.content && self.used < self.width {
                self.begin_row();
                self.used += 1;
                self.line.push(Span::styled(" ", style));
            }
            if !word.is_empty() {
                self.word(word, style);
            }
        }
    }

    fn markdown(&mut self, text: &str) {
        let options = Options::ENABLE_MATH | Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
        let source = dollar_math(&clean(text));
        for event in Parser::new_ext(&source, options) {
            if let Some(table) = &mut self.table {
                match event {
                    Md::Start(Tag::TableRow | Tag::TableHead) => table.push(Vec::new()),
                    Md::Start(Tag::TableCell) => table.last_mut().into_iter().for_each(|row| row.push(String::new())),
                    Md::Text(text) | Md::Code(text) => table.last_mut().and_then(|row| row.last_mut()).into_iter().for_each(|cell| cell.push_str(&text)),
                    Md::InlineMath(text) => table.last_mut().and_then(|row| row.last_mut()).into_iter().for_each(|cell| cell.push_str(&format!("${text}$"))),
                    Md::End(TagEnd::Table) => {
                        let rows = self.table.take().unwrap_or_default();
                        self.draw_table(rows);
                    }
                    _ => {}
                }
                continue;
            }
            self.markdown_event(event);
        }
        self.blank();
    }

    fn markdown_event(&mut self, event: Md) {
        let push_style = |writer: &mut Writer, change: fn(Style) -> Style| {
            let style = change(writer.style());
            writer.styles.push(style);
        };
        match event {
            Md::Start(Tag::Heading { .. }) => push_style(self, |style| style.fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
            Md::Start(Tag::Emphasis) => push_style(self, |style| style.add_modifier(Modifier::ITALIC)),
            Md::Start(Tag::Strong) => push_style(self, |style| style.add_modifier(Modifier::BOLD)),
            Md::Start(Tag::Strikethrough) => push_style(self, |style| style.add_modifier(Modifier::CROSSED_OUT)),
            Md::Start(Tag::Link { .. }) => push_style(self, |style| style.fg(theme::ACCENT).add_modifier(Modifier::UNDERLINED)),
            Md::End(TagEnd::Heading(_)) => {
                self.styles.pop();
                self.blank();
            }
            Md::End(TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough | TagEnd::Link) => {
                self.styles.pop();
            }
            Md::End(TagEnd::Paragraph) => self.blank(),
            Md::Start(Tag::BlockQuote(_)) => self.levels.push(Level::fixed("▏ ", Style::new().fg(theme::DIM))),
            Md::End(TagEnd::BlockQuote(_)) => {
                self.flush_line(false);
                self.levels.pop();
                self.blank();
            }
            Md::Start(Tag::CodeBlock(kind)) => {
                self.flush_line(false);
                if let CodeBlockKind::Fenced(language) = kind
                    && !language.is_empty()
                {
                    self.word(&language, Style::new().fg(theme::DIM));
                    self.flush_line(false);
                }
                self.levels.push(Level::fixed("│ ", Style::new().fg(theme::DIM)));
                self.code = true;
            }
            Md::End(TagEnd::CodeBlock) => {
                self.flush_line(false);
                self.levels.pop();
                self.code = false;
                self.blank();
            }
            Md::Start(Tag::List(start)) => {
                self.flush_line(false);
                self.lists.push(start);
            }
            Md::End(TagEnd::List(_)) => {
                self.flush_line(false);
                self.lists.pop();
                if self.lists.is_empty() {
                    self.blank();
                }
            }
            Md::Start(Tag::Item) => {
                self.flush_line(false);
                let bullet = match self.lists.last_mut() {
                    Some(Some(number)) => {
                        *number += 1;
                        format!("{}. ", *number - 1)
                    }
                    _ => "• ".to_string(),
                };
                let rest = " ".repeat(bullet.width());
                self.levels.push(Level { first: bullet, rest, style: Style::new().fg(theme::ACCENT), fresh: true });
            }
            Md::End(TagEnd::Item) => {
                self.flush_line(false);
                self.levels.pop();
            }
            Md::Start(Tag::Table(_)) => {
                self.flush_line(false);
                self.table = Some(Vec::new());
            }
            Md::Text(text) if self.code => {
                for (index, line) in text.split('\n').enumerate() {
                    if index > 0 {
                        self.flush_line(true);
                    }
                    if !line.is_empty() {
                        self.word(line, Style::new().fg(theme::CODE));
                    } else {
                        self.begin_row();
                    }
                }
            }
            Md::Text(text) | Md::Html(text) | Md::InlineHtml(text) => {
                let style = self.style();
                self.text(&text, style);
            }
            Md::Code(text) => self.word(&text, Style::new().fg(theme::CODE)),
            Md::SoftBreak => {
                let style = self.style();
                self.text(" ", style);
            }
            Md::HardBreak => self.flush_line(true),
            Md::Rule => {
                self.flush_line(false);
                let rule = "─".repeat(self.room());
                self.word(&rule, Style::new().fg(theme::DIM));
                self.blank();
            }
            Md::TaskListMarker(done) => self.word(if done { "☑ " } else { "☐ " }, Style::new().fg(theme::ACCENT)),
            Md::InlineMath(text) => self.inline_math(&text),
            Md::DisplayMath(text) => self.display_math(&text),
            _ => {}
        }
    }

    fn inline_math(&mut self, source: &str) {
        let room = self.room() as u16;
        match self.tex.formula(source, false, room) {
            Some(Formula::Image { id, columns, .. }) => {
                let (cells, color) = tex::placeholders(*id, 0, *columns);
                let columns = usize::from(*columns);
                self.piece(cells, columns, Style::new().fg(color));
            }
            Some(Formula::Text(rows)) => {
                let row = rows.first().cloned().unwrap_or_default();
                self.word(&row, Style::new().fg(theme::MATH));
            }
            None => self.word(&format!("${source}$"), Style::new().fg(theme::MATH)),
        }
    }

    // Block math, centered: the image's rows, term-maths' rows, or the source.
    fn display_math(&mut self, source: &str) {
        self.flush_line(false);
        let room = self.room();
        let rows: Vec<(String, usize, Style)> = match self.tex.formula(source, true, room as u16) {
            Some(Formula::Image { id, columns, rows }) => (0..*rows)
                .map(|row| {
                    let (cells, color) = tex::placeholders(*id, row, *columns);
                    (cells, usize::from(*columns), Style::new().fg(color))
                })
                .collect(),
            Some(Formula::Text(rows)) => rows.iter().map(|row| (row.clone(), row.width(), Style::new().fg(theme::MATH))).collect(),
            None => format!("$${source}$$").lines().map(|line| (line.to_string(), line.width(), Style::new().fg(theme::MATH))).collect(),
        };
        let widest = rows.iter().map(|(_, width, _)| *width).max().unwrap_or(0);
        let pad = " ".repeat(room.saturating_sub(widest) / 2);
        for (text, width, style) in rows {
            self.begin_row();
            self.line.push(Span::raw(pad.clone()));
            self.used += pad.len();
            self.piece(text, width, style);
            self.flush_line(false);
        }
        self.blank();
    }

    // A table with its columns sized to their cells, the widest shrunk until it fits.
    fn draw_table(&mut self, rows: Vec<Vec<String>>) {
        let count = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut widths = vec![0usize; count];
        for row in &rows {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(cell.trim().width());
            }
        }
        let room = self.room().saturating_sub(3 * count.saturating_sub(1));
        while widths.iter().sum::<usize>() > room {
            let Some(widest) = widths.iter_mut().max() else { break };
            if *widest <= 3 {
                break;
            }
            *widest -= 1;
        }
        let rule = Style::new().fg(theme::DIM);
        for (index, row) in rows.iter().enumerate() {
            self.begin_row();
            for (column, width) in widths.iter().enumerate() {
                if column > 0 {
                    self.line.push(Span::styled(" │ ", rule));
                }
                let cell = row.get(column).map_or("", |cell| cell.trim());
                let style = if index == 0 { Style::new().add_modifier(Modifier::BOLD) } else { Style::new() };
                self.line.push(Span::styled(pad(&fit(cell, *width), *width), style));
            }
            self.content = true;
            self.flush_line(false);
            if index == 0 {
                self.begin_row();
                let line = widths.iter().map(|width| "─".repeat(*width)).collect::<Vec<_>>().join("─┼─");
                self.line.push(Span::styled(line, rule));
                self.content = true;
                self.flush_line(false);
            }
        }
        self.blank();
    }

    // A tool call: one line naming it, with how many lines of output it has hidden; with tools open
    // (ctrl+o), its output (a diff for an edit) under it, up to TOOL_OPEN_ROWS. A run of calls stays
    // a run of lines, so the conversation, not the commands, fills the screen.
    fn tool(&mut self, name: &str, arguments: &Map<String, Value>, view: &ToolView, flags: Flags, frame: usize) {
        let (mark, color) = match (view.running, view.is_error) {
            (true, _) => (SPINNER[frame % SPINNER.len()], theme::ACCENT),
            (false, true) => ("✗", theme::ERROR),
            (false, false) => ("⏺", theme::OK),
        };
        self.word(&format!("{mark} "), Style::new().fg(color));
        self.word(name, Style::new().add_modifier(Modifier::BOLD));
        let text = view.diff.as_deref().unwrap_or(&view.text);
        let text = clean(text);
        let lines: Vec<&str> = text.lines().collect();
        let hidden = if flags.tools_open || lines.is_empty() { String::new() } else { format!(" · {} line{}", lines.len(), if lines.len() == 1 { "" } else { "s" }) };
        let summary = summary(arguments);
        if !summary.is_empty() {
            let room = self.width.saturating_sub(self.used + 2 + hidden.width());
            self.word(&format!("  {}", fit(&summary, room)), Style::new().fg(theme::DIM));
        }
        self.word(&hidden, Style::new().fg(theme::DIM));
        self.flush_line(false);
        if !flags.tools_open {
            return;
        }
        let shown = TOOL_OPEN_ROWS;
        self.levels.push(Level { first: "  ⎿ ".into(), rest: "    ".into(), style: Style::new().fg(theme::DIM), fresh: true });
        for line in lines.iter().take(shown) {
            let style = match line.chars().next() {
                Some('+') if view.diff.is_some() => Style::new().fg(theme::ADDED),
                Some('-') if view.diff.is_some() => Style::new().fg(theme::REMOVED),
                _ if view.is_error => Style::new().fg(theme::ERROR),
                _ => Style::new().fg(theme::DIM),
            };
            self.begin_row();
            self.word(line, style);
            self.content = true;
            self.flush_line(false);
        }
        if lines.len() > shown {
            self.word(&format!("… {} more lines (ctrl+o)", lines.len() - shown), Style::new().fg(theme::DIM));
            self.flush_line(false);
        }
        self.levels.pop();
        self.blank();
    }
}

// A tool call in a few words: its command, path or query, else its arguments as JSON.
fn summary(arguments: &Map<String, Value>) -> String {
    let subagent = arguments.get("agent").and_then(Value::as_str).zip(arguments.get("task").and_then(Value::as_str)).map(|(agent, task)| format!("{agent}: {task}"));
    let found = subagent.or_else(|| SUMMARY_KEYS.iter().find_map(|key| arguments.get(*key).and_then(Value::as_str)).map(str::to_string));
    let text = found.unwrap_or_else(|| if arguments.is_empty() { String::new() } else { Value::Object(arguments.clone()).to_string() });
    clean(&text).lines().next().unwrap_or("").to_string()
}

// A path with the home directory written as ~. Path::strip_prefix matches whole components, so
// /home/me2 isn't under /home/me; a HOME of / is left alone, since every path is under it.
pub fn home_relative(path: &std::path::Path) -> String {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from).filter(|home| home.parent().is_some());
    match home.as_deref().and_then(|home| path.strip_prefix(home).ok()) {
        Some(rest) if rest.as_os_str().is_empty() => "~".into(),
        Some(rest) => format!("~/{}", rest.display()),
        None => path.display().to_string(),
    }
}

// A running time the short way: 42s, 7m, 2h05m.
pub fn duration(running_for: std::time::Duration) -> String {
    match running_for.as_secs() {
        seconds @ 0..60 => format!("{seconds}s"),
        seconds @ 60..3600 => format!("{}m", seconds / 60),
        seconds => format!("{}h{:02}m", seconds / 3600, seconds / 60 % 60),
    }
}

// Text cut to a width, with "…" where it was cut.
pub fn fit(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for character in text.chars() {
        let character_width = character.width().unwrap_or(0);
        if used + character_width + 1 > width {
            break;
        }
        out.push(character);
        used += character_width;
    }
    out.push('…');
    out
}

fn pad(text: &str, width: usize) -> String {
    format!("{text}{}", " ".repeat(width.saturating_sub(text.width())))
}

// Text from a model or a tool, safe to put on screen: tabs as spaces, escape sequences and other
// control characters removed (they would reach the terminal as commands), and no image
// placeholders (they would draw one of our images).
pub fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '\t' => out.push_str("    "),
            '\n' => out.push('\n'),
            // An escape sequence: ESC [ ... final byte, ESC ] ... BEL/ST, or ESC and one more.
            '\x1b' => match characters.next() {
                Some('[') => {
                    for next in characters.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    while let Some(next) = characters.next() {
                        if next == '\x07' || (next == '\x1b' && characters.next_if_eq(&'\\').is_some()) {
                            break;
                        }
                    }
                }
                _ => {}
            },
            '\u{10EEEE}' => {}
            character if character.is_control() => {}
            character => out.push(character),
        }
    }
    out
}

// Models write math as \(..\) and \[..\] as often as $..$ and $$..$$; the Markdown parser knows
// only the dollar forms, so the others are rewritten first, outside code.
fn dollar_math(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut fenced = false;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
        }
        if fenced || trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            out.push_str(line);
            continue;
        }
        let mut in_code = false;
        let mut characters = line.chars().peekable();
        while let Some(character) = characters.next() {
            match character {
                '`' => {
                    in_code = !in_code;
                    out.push('`');
                }
                '\\' if !in_code => match characters.peek() {
                    Some('(' | ')') => {
                        characters.next();
                        out.push('$');
                    }
                    Some('[' | ']') => {
                        characters.next();
                        out.push_str("$$");
                    }
                    _ => out.push('\\'),
                },
                character => out.push(character),
            }
        }
    }
    out
}
