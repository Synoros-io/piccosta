// Copy and paste: the system clipboard, and the text under a mouse selection.
//
// The clipboard is reached through the desktop's own tools, the way Claude Code reaches it:
// wl-copy and wl-paste under Wayland, xclip or else xsel under X. A copy goes to the clipboard and the primary
// selection both, so a middle click elsewhere pastes it too. Where neither tool's desktop is there
// (over ssh), a copy goes to the terminal as OSC 52, and there's no paste but the terminal's own.

use base64::Engine;
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use std::io::Write;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use unicode_width::UnicodeWidthStr;

// A desktop's clipboard tools: the variable that says the desktop is there, then the commands that
// copy to the clipboard and the primary selection and paste from them.
struct Tools {
    desktop: &'static str,
    copy: [&'static [&'static str]; 2],
    paste: [&'static [&'static str]; 2],
}

const TOOLS: [Tools; 3] = [
    Tools {
        desktop: "WAYLAND_DISPLAY",
        copy: [&["wl-copy"], &["wl-copy", "--primary"]],
        paste: [&["wl-paste", "--no-newline"], &["wl-paste", "--no-newline", "--primary"]],
    },
    Tools {
        desktop: "DISPLAY",
        copy: [&["xclip", "-selection", "clipboard"], &["xclip", "-selection", "primary"]],
        paste: [&["xclip", "-o", "-selection", "clipboard"], &["xclip", "-o", "-selection", "primary"]],
    },
    Tools {
        desktop: "DISPLAY",
        copy: [&["xsel", "--clipboard", "--input"], &["xsel", "--primary", "--input"]],
        paste: [&["xsel", "--clipboard", "--output"], &["xsel", "--primary", "--output"]],
    },
];

// The tools of the desktops this runs under, Wayland's first (XWayland sets DISPLAY too).
fn tools() -> impl Iterator<Item = &'static Tools> {
    TOOLS.iter().filter(|tools| std::env::var_os(tools.desktop).is_some())
}

// Commands are argv arrays, never shell lines. Only the user's settings can choose them;
// absent selections use desktop detection, while an empty array disables that selection.
async fn commands(copy: bool, primary: bool) -> Vec<Vec<String>> {
    tokio::task::spawn_blocking(move || {
        let settings = crate::config::Settings::load(&crate::config::agent_dir(), None);
        let key = format!("clipboard.{}.{}", if copy { "copy" } else { "paste" }, if primary { "primary" } else { "clipboard" });
        if let Some(value) = settings.get(&key) {
            return value.as_array().and_then(|values| values.iter().map(|value| value.as_str().map(str::to_string)).collect::<Option<Vec<_>>>())
                .filter(|command| command.first().is_some_and(|program| !program.is_empty())).into_iter().collect();
        }
        tools().map(|tools| (if copy { tools.copy } else { tools.paste })[usize::from(primary)].iter().map(|word| word.to_string()).collect()).collect()
    }).await.unwrap_or_default()
}

// Queue a copy without making a large stdin write or a configured command hold up the screen.
// The task owns and reaps its children; dropping it at shutdown kills a still-blocked command.
pub fn copy(text: &str) -> String {
    static COPIES: std::sync::OnceLock<tokio::sync::mpsc::UnboundedSender<String>> = std::sync::OnceLock::new();
    let copies = COPIES.get_or_init(|| {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        // Preserve copy order even when a custom command is slow.
        tokio::spawn(async move { while let Some(text) = receiver.recv().await { copy_text(text).await; } });
        sender
    });
    let _ = copies.send(text.to_string());
    "the clipboard/terminal (background)".into()
}

async fn copy_text(text: String) {
    let mut copied = false;
    for primary in [false, true] {
        for command in commands(true, primary).await {
            let Ok(mut child) = Command::new(&command[0]).args(&command[1..]).kill_on_drop(true).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() else { continue };
            let written = match child.stdin.take() {
                Some(mut stdin) => stdin.write_all(text.as_bytes()).await.is_ok(),
                None => false,
            };
            if child.wait().await.is_ok_and(|status| status.success()) && written { copied = true; break; }
        }
    }
    if !copied {
        // OSC 52: the terminal sets its clipboard. Inside tmux it goes through as passthrough.
        let encoded = base64::engine::general_purpose::STANDARD.encode(&text);
        let sequence = format!("\x1b]52;c;{encoded}\x07");
        let sequence = if std::env::var_os("TMUX").is_some() { format!("\x1bPtmux;{}\x1b\\", sequence.replace('\x1b', "\x1b\x1b")) } else { sequence };
        let _ = tokio::task::spawn_blocking(move || {
            let mut stdout = std::io::stdout();
            let _ = stdout.write_all(sequence.as_bytes()).and_then(|()| stdout.flush());
        }).await;
    }
}

// A delayed paste belongs to the input that requested it, not a newly focused pane or picker.
pub(super) struct Paste {
    target: super::Target,
    picker: Option<String>,
    input: String,
    text: Option<String>,
}
impl Paste {
    pub(super) fn apply(self, app: &mut super::App) {
        if self.target != app.panes[app.focus].target || self.picker != app.picker.as_ref().map(|picker| picker.title.clone()) || self.input != input(app) {
            app.notice = Some("Clipboard arrived after the input changed; paste again.".into());
        } else if let Some(text) = self.text {
            app.paste_text(&text);
        } else {
            app.notice = Some("Nothing to paste: no clipboard command answered; the terminal's own paste key still works".into());
        }
    }
}
fn input(app: &super::App) -> String {
    app.picker.as_ref().map_or_else(|| app.draft().text.clone(), |picker| picker.filter.clone())
}

// Read clipboard or primary text asynchronously, then wake the UI through its own update queue.
pub(super) fn paste(app: &super::App, primary: bool) {
    let mut paste = Paste { target: app.panes[app.focus].target, picker: app.picker.as_ref().map(|picker| picker.title.clone()), input: input(app), text: None };
    let sender = app.sender.clone();
    tokio::spawn(async move {
        for command in commands(false, primary).await {
            if let Ok(output) = Command::new(&command[0]).args(&command[1..]).kill_on_drop(true).stdin(Stdio::null()).stderr(Stdio::null()).output().await
                && output.status.success() {
                paste.text = Some(String::from_utf8_lossy(&output.stdout).into_owned());
                break;
            }
        }
        let _ = sender.send((0, super::Update::Clipboard(paste)));
    });
}

// A place in a pane's conversation that stays on its text when the text wraps again: the message
// it's in; twice the count of non-blank characters before it in that message's rows, one more when
// it's on a character rather than a blank; and for a blank its row in the message, which tells apart
// the blanks of one run.
pub type Point = (usize, usize, usize);

// A row of a message: its message, its number there, and the count of non-blank characters above it.
pub type Row = (usize, usize, usize);

// A row's non-blank characters.
pub fn solid(row: &Line) -> usize {
    row.spans.iter().flat_map(|span| span.content.chars()).filter(|character| !character.is_whitespace()).count()
}

// A row's characters as ratatui draws them, by grapheme, so an accented letter or a joined emoji is
// one: each one's first column, width, text and place; then the place past the row's end.
pub type Cells<'a> = (Vec<(u16, u16, &'a str, Point)>, Point);

pub fn cells<'a>(row: &'a Line, (message, number, before): Row) -> Cells<'a> {
    let (mut column, mut count) = (0u16, before);
    let cells = row.styled_graphemes(Style::new()).map(|grapheme| {
        let (characters, width) = (grapheme.symbol.chars().filter(|character| !character.is_whitespace()).count(), grapheme.symbol.width() as u16);
        let cell = (column, width, grapheme.symbol, (message, 2 * count + usize::from(characters > 0), if characters > 0 { 0 } else { number }));
        (column, count) = (column.saturating_add(width), count + characters);
        cell
    });
    (cells.collect(), (message, 2 * count, number))
}

// The ends of what a press at column `at` of a row selects: the place under it, its word (a run of
// non-blanks, so a path or URL is one word) for a double click, its whole row for a triple one.
pub fn word((cells, end): &Cells, at: u16, clicks: u8) -> (Point, Point) {
    let under = cells.iter().position(|(column, width, ..)| (*column..column + width).contains(&at));
    let place = under.map_or(*end, |index| cells[index].3);
    match under {
        _ if clicks == 3 => (cells.first().map_or(*end, |cell| cell.3), *end),
        Some(index) if clicks == 2 && place.1 % 2 == 1 => {
            let solid = |cell: &&(u16, u16, &str, Point)| cell.3.1 % 2 == 1;
            let first = cells[..index].iter().rev().take_while(solid).count();
            let last = cells[index..].iter().take_while(solid).count();
            (cells[index - first].3, cells[index + last - 1].3)
        }
        _ => (place, place),
    }
}

// Show reversed the first to last columns `selected` gives each row of `area`, counted from its top.
pub fn highlight(buffer: &mut Buffer, area: Rect, selected: impl Fn(u16) -> Option<(u16, u16)>) {
    for y in area.top()..area.bottom() {
        let Some((first, last)) = selected(y - area.y) else { continue };
        for x in area.x.saturating_add(first)..=area.x.saturating_add(last).min(area.right().saturating_sub(1)) {
            if let Some(cell) = buffer.cell_mut(Position::new(x, y)) {
                cell.set_style(Style::new().add_modifier(Modifier::REVERSED));
            }
        }
    }
}

// The first and last columns of a row selected from `from` to `to` (places only grow along a row, so
// they're one run), to the edge when the selection goes past the row's end. Only a row an end of
// the selection is on is cut into characters to find them.
pub fn span(row: &Line, at @ (message, number, before): Row, from: Point, to: Point) -> Option<(u16, u16)> {
    let (top, bottom) = ((message, 2 * before, 0), (message, 2 * (before + solid(row)), number));
    if bottom < from || top > to {
        return None;
    }
    if from <= top && bottom <= to {
        return Some((0, u16::MAX));
    }
    let (cells, end) = &cells(row, at);
    let selected = cells.iter().filter(|cell| (from..=to).contains(&cell.3));
    let span = selected.fold(None, |span: Option<(u16, u16)>, cell| Some((span.map_or(cell.0, |span| span.0), cell.0 + cell.1.max(1) - 1)));
    let after = cells.last().map_or(0, |cell| cell.0 + cell.1);
    if (from..=to).contains(end) { Some((span.map_or(after, |span| span.0), u16::MAX)) } else { span }
}

// The text from `from` to `to` on these rows, a line per row with the spaces at its ends dropped.
pub fn text<'a>(rows: impl Iterator<Item = Cells<'a>>, from: Point, to: Point) -> String {
    let lines = rows.filter(|(cells, end)| cells.first().map_or(*end, |cell| cell.3) <= to && *end >= from);
    let lines = lines.map(|(cells, _)| cells.iter().filter(|cell| (from..=to).contains(&cell.3)).map(|cell| cell.2).collect::<String>().trim_end().to_string());
    lines.collect::<Vec<_>>().join("\n")
}
