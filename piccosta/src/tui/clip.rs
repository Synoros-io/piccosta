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
use std::process::{Command, Stdio};
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

// Copy `text` to the clipboard and the primary selection. Output: what it went through, for the
// notice. The tools stay behind to serve the copy; each is waited on from its own thread, so a
// slow one never holds up the screen.
pub fn copy(text: &str) -> String {
    for tools in tools() {
        let mut copied = false;
        for command in tools.copy {
            let Ok(mut child) = Command::new(command[0]).args(&command[1..]).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() else { continue };
            if let Some(mut stdin) = child.stdin.take() {
                copied |= stdin.write_all(text.as_bytes()).is_ok();
            }
            std::thread::spawn(move || child.wait());
        }
        if copied {
            return tools.copy[0][0].to_string();
        }
    }
    // OSC 52: the terminal sets its clipboard. Inside tmux it goes through as passthrough.
    let encoded = base64::engine::general_purpose::STANDARD.encode(text);
    let sequence = format!("\x1b]52;c;{encoded}\x07");
    let sequence = if std::env::var_os("TMUX").is_some() { format!("\x1bPtmux;{}\x1b\\", sequence.replace('\x1b', "\x1b\x1b")) } else { sequence };
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(sequence.as_bytes()).and_then(|()| stdout.flush());
    "the terminal (OSC 52)".into()
}

// The clipboard's text, or the primary selection's (what was last selected anywhere, the middle
// click's paste). None when no tool answers.
pub fn paste(primary: bool) -> Option<String> {
    tools().find_map(|tools| {
        let command = tools.paste[usize::from(primary)];
        let output = Command::new(command[0]).args(&command[1..]).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
        output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
    })
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
