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
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use unicode_width::UnicodeWidthStr;

// A desktop's clipboard tools: the variable that says the desktop is there, then the commands that
// copy to the clipboard and the primary selection and paste from them, and, when the tool reads
// images, the commands that list the clipboard's types and read one (the type goes on the end).
struct Tools {
    desktop: &'static str,
    copy: [&'static [&'static str]; 2],
    paste: [&'static [&'static str]; 2],
    image: Option<[&'static [&'static str]; 2]>,
}

// How long a clipboard tool gets to answer: wl-paste waits as long as the clipboard's owner does,
// and an owner that hasn't written in two seconds is stuck. It's looked at every CLIPBOARD_POLL, so
// a tool that's done is seen within 2 ms.
const CLIPBOARD_WAIT: Duration = Duration::from_secs(2);
const CLIPBOARD_POLL: Duration = Duration::from_millis(2);

// The image types a paste takes from the clipboard, the one a screenshot tool puts there first.
const IMAGE_TYPES: [&str; 2] = ["image/png", "image/jpeg"];

const TOOLS: [Tools; 3] = [
    Tools {
        desktop: "WAYLAND_DISPLAY",
        copy: [&["wl-copy"], &["wl-copy", "--primary"]],
        paste: [&["wl-paste", "--no-newline"], &["wl-paste", "--no-newline", "--primary"]],
        image: Some([&["wl-paste", "--list-types"], &["wl-paste", "--type"]]),
    },
    Tools {
        desktop: "DISPLAY",
        copy: [&["xclip", "-selection", "clipboard"], &["xclip", "-selection", "primary"]],
        paste: [&["xclip", "-o", "-selection", "clipboard"], &["xclip", "-o", "-selection", "primary"]],
        image: Some([&["xclip", "-o", "-selection", "clipboard", "-t", "TARGETS"], &["xclip", "-o", "-selection", "clipboard", "-t"]]),
    },
    Tools {
        desktop: "DISPLAY",
        copy: [&["xsel", "--clipboard", "--input"], &["xsel", "--primary", "--input"]],
        paste: [&["xsel", "--clipboard", "--output"], &["xsel", "--primary", "--output"]],
        image: None,
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

// What a paste takes from the clipboard: the image on it, when asked for one and it offers one as
// png or jpeg, else its text.
pub enum Pasted {
    Image(Vec<u8>),
    Text(String),
}

// The clipboard's text, or the primary selection's (what was last selected anywhere, the middle
// click's paste), or with `image` the image on the clipboard first. The text is read while the
// image is looked for, so a text paste waits no longer than it did. None when no tool answers.
pub fn paste(primary: bool, image: bool) -> Option<Pasted> {
    std::thread::scope(|scope| {
        let text = scope.spawn(|| tools().find_map(|tools| run(tools.paste[usize::from(primary)], None)));
        let image = tools().filter(|_| image).find_map(|tools| {
            let [list, read] = tools.image?;
            let types = String::from_utf8_lossy(&run(list, None)?).into_owned();
            let kind = IMAGE_TYPES.into_iter().find(|kind| types.lines().any(|offered| offered.trim() == *kind))?;
            run(read, Some(kind)).filter(|bytes| !bytes.is_empty())
        });
        image.map(Pasted::Image).or_else(|| text.join().ok().flatten().map(|bytes| Pasted::Text(String::from_utf8_lossy(&bytes).into_owned())))
    })
}

// What a clipboard command prints, with `last` added to its arguments; None when it fails, or
// hasn't finished by CLIPBOARD_WAIT, when it's killed.
fn run(command: &[&str], last: Option<&str>) -> Option<Vec<u8>> {
    let mut child = Command::new(command[0]).args(&command[1..]).args(last).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let mut stdout = child.stdout.take()?;
    let read = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let deadline = Instant::now() + CLIPBOARD_WAIT;
    let status = loop {
        match child.try_wait().ok()? {
            Some(status) => break status,
            None if Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            None => std::thread::sleep(CLIPBOARD_POLL),
        }
    };
    read.join().ok()?.ok().filter(|_| status.success())
}

// A selection's two ends in reading order, both inside `rect`.
fn ordered(rect: Rect, anchor: Position, head: Position) -> (Position, Position) {
    let clamp = |at: Position| Position::new(at.x.clamp(rect.x, rect.right().saturating_sub(1)), at.y.clamp(rect.y, rect.bottom().saturating_sub(1)));
    let (anchor, head) = (clamp(anchor), clamp(head));
    if (anchor.y, anchor.x) <= (head.y, head.x) { (anchor, head) } else { (head, anchor) }
}

// The cells from `from` to `to` in reading order, as a terminal selects: the first row from its
// column to the edge, whole rows between, the last row up to its column.
fn cells(rect: Rect, from: Position, to: Position) -> impl Iterator<Item = Position> {
    (from.y..=to.y).flat_map(move |y| {
        let start = if y == from.y { from.x } else { rect.x };
        let end = if y == to.y { to.x } else { rect.right().saturating_sub(1) };
        (start..=end).map(move |x| Position::new(x, y))
    })
}

// Show a selection reversed.
pub fn highlight(buffer: &mut Buffer, rect: Rect, anchor: Position, head: Position) {
    let (from, to) = ordered(rect, anchor, head);
    for at in cells(rect, from, to) {
        if let Some(cell) = buffer.cell_mut(at) {
            cell.set_style(Style::new().add_modifier(Modifier::REVERSED));
        }
    }
}

// The text a selection covers, a line per row with the spaces at their ends dropped. A wide
// character's second cell holds nothing of its own and is skipped.
pub fn text(buffer: &Buffer, rect: Rect, anchor: Position, head: Position) -> String {
    let (from, to) = ordered(rect, anchor, head);
    let mut lines: Vec<String> = vec![String::new()];
    let mut row = from.y;
    let mut skip = 0;
    for at in cells(rect, from, to) {
        if at.y != row {
            (row, skip) = (at.y, 0);
            lines.push(String::new());
        }
        if skip > 0 {
            skip -= 1;
            continue;
        }
        let symbol = buffer.cell(at).map_or(" ", |cell| cell.symbol());
        skip = symbol.width().saturating_sub(1);
        lines.last_mut().into_iter().for_each(|line| line.push_str(symbol));
    }
    lines.iter().map(|line| line.trim_end()).collect::<Vec<_>>().join("\n")
}

// The word under `at` on its row (a run of non-spaces, so a path or URL is one word), or the whole
// row: the two ends to select.
pub fn expand(buffer: &Buffer, rect: Rect, at: Position, whole_row: bool) -> (Position, Position) {
    if whole_row {
        return (Position::new(rect.x, at.y), Position::new(rect.right().saturating_sub(1), at.y));
    }
    let blank = |x: u16| buffer.cell(Position::new(x, at.y)).is_none_or(|cell| cell.symbol().trim().is_empty());
    if blank(at.x) {
        return (at, at);
    }
    let start = (rect.x..=at.x).rev().take_while(|&x| !blank(x)).last().unwrap_or(at.x);
    let end = (at.x..rect.right()).take_while(|&x| !blank(x)).last().unwrap_or(at.x);
    (Position::new(start, at.y), Position::new(end, at.y))
}
