// --single in the terminal's own scrollback, the way Claude Code runs. Each finished message is
// printed once, above the live part, and never drawn again, so the terminal's scrolling, selection
// and copy work on it. Only the live part (the reply being written, a running tool, the working
// line, the editor, the status line, a picker) is drawn again in place: ratatui draws it as usual,
// into a Fixed viewport that this file moves and sizes as the conversation grows. Its rows come
// from render.rs, the same as the full UI's.

use super::render::{self, Block, Flags};
use super::{App, Rows};
use crate::message::AgentMessage;
use pulldown_cmark::{Event as Md, Parser};
use crossterm::cursor::MoveTo;
use crossterm::queue;
use crossterm::style::Print;
use crossterm::terminal::{Clear, ClearType};
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect, Size};
use ratatui::{DefaultTerminal, Terminal, TerminalOptions, Viewport};
use std::io::Write;
use std::sync::atomic::{AtomicU16, Ordering};

// The fewest rows the live part takes, before a frame has said how many it wants: the header, an
// editor of one line in its border, and the status line.
const FEWEST_ROWS: usize = 5;

// Frames drawn at most for one: the second places the live part at the height the first wanted.
const DRAWS: usize = 2;

// The pane's margin: rows are printed a column in from each side, as the pane draws them.
const MARGIN: u16 = 1;

// The live part's first row, for restore (and so a panic) to take it off the screen; NONE when
// there's none on it.
const NONE: u16 = u16::MAX;
static TOP: AtomicU16 = AtomicU16::new(NONE);

pub struct Inline {
    // The live part's first row on the screen and its height, and the screen's size, at the last frame.
    top: u16,
    height: u16,
    size: Size,
    // From the last frame: how wide each live row above the typing cursor was and the cursor's
    // column, so after a resize the live part can be found where the terminal rewrapped it.
    widths: Vec<u16>,
    column: u16,
}

impl Inline {
    // Where the live part starts: the cursor's row, or the next one when the shell left text on it.
    pub fn start() -> std::io::Result<Inline> {
        let (column, row) = crossterm::cursor::position()?;
        let (width, height) = crossterm::terminal::size()?;
        let mut top = row;
        if column > 0 {
            // On the last row the newline scrolls the screen, so the next row is the last again.
            queue!(std::io::stdout(), Print("\r\n"))?;
            top = (row + 1).min(height.saturating_sub(1));
        }
        TOP.store(top, Ordering::Relaxed);
        Ok(Inline { top, height: 0, size: Size::new(width, height), widths: Vec::new(), column: 0 })
    }

    // Before a frame: the messages finished since the last go into the scrollback, and the live
    // part is placed under them at the height the last frame wanted (the whole screen for a picker).
    // After a resize the live part is found again and drawn anew at the new width.
    pub fn frame(&mut self, terminal: &mut DefaultTerminal, app: &mut App) -> std::io::Result<()> {
        let size = terminal.size()?;
        let mut out = std::io::stdout();
        if size != self.size {
            let (_, row) = crossterm::cursor::position()?;
            self.top = row.saturating_sub(self.rows_above(size.width));
            (self.size, self.height) = (size, 0);
        }
        let width = size.width.saturating_sub(2 * MARGIN);
        let rows = app.finished(width);
        let height = self.wanted(app.live_rows);
        if rows.is_empty() && height == self.height {
            return Ok(());
        }
        clear_below(self.top)?;
        let mut y = self.top;
        for row in rows {
            // Past the last row the screen scrolls up one for it, its top row into the scrollback.
            if y >= size.height {
                queue!(out, MoveTo(0, size.height - 1), Print('\n'))?;
                y = size.height - 1;
            }
            let area = Rect::new(0, 0, size.width, 1);
            let mut buffer = Buffer::empty(area);
            buffer.set_line(MARGIN, 0, &row, width);
            // Only the cells that aren't blank, so no row is padded out to this width.
            terminal.backend_mut().draw(Buffer::empty(area).diff(&buffer).into_iter().map(|(x, _, cell)| (x, y, cell)))?;
            y += 1;
        }
        // Room for the live part under them, the screen scrolled up when there isn't enough.
        let short = (y + height).saturating_sub(size.height);
        if short > 0 {
            queue!(out, MoveTo(0, size.height - 1), Print("\n".repeat(usize::from(short))))?;
        }
        (self.top, self.height) = (y - short, height);
        TOP.store(self.top, Ordering::Relaxed);
        // A new terminal on the new place, its buffers empty, so the frame is drawn whole.
        *terminal = Terminal::with_options(CrosstermBackend::new(std::io::stdout()), TerminalOptions { viewport: Viewport::Fixed(Rect::new(0, self.top, size.width, height)) })?;
        Ok(())
    }

    // After a frame: what a resize needs to find the live part, and the cursor put on its first row
    // when nothing is being typed (a picker is open), so it's always somewhere known.
    pub fn drawn(&mut self, buffer: &Buffer, typing_at: Option<Position>) -> std::io::Result<()> {
        let at = typing_at.unwrap_or(Position::new(0, self.top));
        let area = buffer.area;
        self.widths = (area.y..at.y.max(area.y)).map(|y| (area.x..area.right()).rev().find(|&x| buffer[(x, y)].symbol() != " ").map_or(0, |x| x + 1)).collect();
        self.column = at.x;
        let mut out = std::io::stdout();
        if typing_at.is_none() {
            queue!(out, MoveTo(at.x, at.y))?;
        }
        out.flush()
    }

    // The live part's height for a frame that wanted `rows`: at least the fewest, at most the screen.
    fn wanted(&self, rows: usize) -> u16 {
        rows.max(FEWEST_ROWS).min(usize::from(self.size.height)) as u16
    }

    // How far above the typing cursor the live part starts once the terminal has rewrapped it to
    // `width`: a row wider than that takes more rows, and the cursor moves down its own row's wraps.
    // Rows are written by position, never run on, so a wider screen joins none.
    fn rows_above(&self, width: u16) -> u16 {
        if width >= self.size.width || width == 0 {
            return self.widths.len() as u16;
        }
        self.widths.iter().map(|&row| row.div_ceil(width).max(1)).sum::<u16>() + self.column / width
    }

}

// The live part off the screen, the cursor where it began, once.
pub fn leave() {
    let top = TOP.swap(NONE, Ordering::Relaxed);
    if top != NONE {
        let _ = clear_below(top).and_then(|()| std::io::stdout().flush());
    }
}

// Clear the screen from row `top` down. Its first row goes on its own: tmux takes a clear from the
// top left corner to the end as the whole screen's, and keeps it in the scrollback.
fn clear_below(top: u16) -> std::io::Result<()> {
    queue!(std::io::stdout(), MoveTo(0, top), Clear(ClearType::CurrentLine), MoveTo(0, top + 1), Clear(ClearType::FromCursorDown))
}

// One frame. --single places its live part first, and when the frame wanted another height than it
// got, places it again and draws once more, so it's never left cut short.
pub fn draw(terminal: &mut DefaultTerminal, app: &mut App, inline: Option<&mut Inline>) -> std::io::Result<()> {
    let Some(inline) = inline else { return terminal.draw(|frame| app.draw(frame)).map(drop) };
    for _ in 0..DRAWS {
        inline.frame(terminal, app)?;
        let done = terminal.draw(|frame| app.draw(frame))?;
        inline.drawn(done.buffer, app.typing_at)?;
        if inline.wanted(app.live_rows) == inline.height {
            break;
        }
    }
    Ok(())
}

impl App {
    // The rows of the shown session's messages finished since the last frame (each of their tools
    // ended, or the turn is over), counted as printed, then the blocks of the reply streaming in that
    // nothing more can change. Another session in the pane (/clear, /resume) is printed from its
    // start; a /fork or /rewind's copy only past what it shares with the one before (commands.rs fork).
    fn finished(&mut self, width: u16) -> Rows {
        let slot_index = self.session_index();
        if self.printed.0 != slot_index {
            (self.printed, self.streamed) = ((slot_index, 0), Streamed::default());
        }
        let slot = &self.slots[slot_index];
        let count = slot.messages.len();
        let from = self.printed.1.min(count);
        let ended = |index: &usize| !slot.busy() || slot.call_owner.iter().all(|(id, owner)| owner != index || slot.tools.get(id).is_some_and(|tool| !tool.running));
        let to = (from..count).find(|index| !ended(index)).unwrap_or(count);
        // Printed with tools open, since ctrl+o can't open them in the scrollback later.
        let flags = Flags { tools_open: true, inline: true, ..self.flags };
        let mut rows = self.rewound.take().map(|text| render::note(&text, width, &mut self.tex)).unwrap_or_default();
        for index in from..to {
            rows.extend(self.unprinted(slot_index, index, width, flags));
        }
        self.printed.1 = to;
        // The reply streaming in, past what's printed of it: its thinking once the part after it has
        // begun, its text up to the block still being written. A tool call redraws until it has run,
        // so it and what follows it wait for the reply to end.
        let slot = &self.slots[slot_index];
        let (Some(parts), Some((model, started)), true) = (&slot.partial, &slot.partial_model, to == count) else { return rows };
        let before = if self.streamed.0 == *started { self.streamed } else { (*started, 0, 0, 0) };
        let (mut part, mut at) = (before.1, before.2);
        let mut done = Vec::new();
        while let Some(block) = parts.get(part) {
            let last = part + 1 == parts.len();
            match block {
                // A retried request (the Claude bridge drops what it abandoned) can replace the text:
                // what's printed stays, and the new text prints from its start (resume).
                Block::Text(text) if last => {
                    let from = resume(text, at, before.3);
                    let end = from + finished_blocks(&text[from..]);
                    done.push(Block::Text(text[from..end].to_string()));
                    at = end;
                    break;
                }
                Block::Text(text) => done.push(Block::Text(text[resume(text, at, before.3)..].to_string())),
                Block::Thinking(_) if !last => done.push(block.clone()),
                _ => break,
            }
            (part, at) = (part + 1, 0);
        }
        let sum = match parts.get(part) { Some(Block::Text(text)) if at > 0 => checksum(&text[..at]), _ => 0 };
        if (part, at, sum) == (before.1, before.2, before.3) {
            return rows;
        }
        if (before.1, before.2) == (0, 0) {
            rows.extend(super::model_row(&slot.messages, Some(model.clone())));
        }
        rows.extend(render::assistant(&done, None, &slot.tools, flags, self.frame, width, &mut self.tex));
        self.streamed = (*started, part, at, sum);
        rows
    }

    // The reply streaming into a slot, as rows. In --single, only what isn't printed yet, under its
    // model's row when none of it is, the way it's printed once it ends.
    pub(super) fn streaming(&mut self, slot_index: usize, width: u16) -> Option<Rows> {
        let slot = &self.slots[slot_index];
        let parts = slot.partial.as_ref()?;
        if !self.inline {
            return Some(render::assistant(parts, None, &slot.tools, self.flags, self.frame, width, &mut self.tex));
        }
        let (model, started) = slot.partial_model.clone()?;
        let printed = if self.streamed.0 == started { self.streamed } else { (started, 0, 0, 0) };
        let mut rows = if (printed.1, printed.2) == (0, 0) { super::model_row(&slot.messages, Some(model)) } else { Vec::new() };
        rows.extend(render::assistant(&rest(parts, printed), None, &slot.tools, self.flags, self.frame, width, &mut self.tex));
        Some(rows)
    }

    // A message's rows, less what of it --single printed while it streamed.
    pub(super) fn unprinted(&mut self, slot_index: usize, index: usize, width: u16, flags: Flags) -> Rows {
        let slot = &self.slots[slot_index];
        match &slot.messages[index] {
            AgentMessage::Assistant(reply) if self.inline && reply.timestamp == self.streamed.0 => {
                let ended = Some((&reply.stop_reason, reply.error_message.as_deref()));
                render::assistant(&rest(&render::blocks(reply), self.streamed), ended, &slot.tools, flags, self.frame, width, &mut self.tex)
            }
            _ => self.rows_of(slot_index, index, width, flags),
        }
    }
}

// How far into a reply --single printed while it streamed: the reply (when it started; MessageEnd
// moves it to the ended message's time), the parts printed whole, the bytes of the next, and a
// checksum of those bytes, so a replaced text is known (resume).
pub type Streamed = (i64, usize, usize, u64);

// A reply's parts past what was printed of it.
fn rest(parts: &[Block], (_, part, at, sum): Streamed) -> Vec<Block> {
    let mut rest = parts.get(part..).unwrap_or_default().to_vec();
    if let Some(Block::Text(text)) = rest.first_mut() {
        *text = text[resume(text, at, sum)..].to_string();
    }
    rest
}

// Where printing goes on in a part's text: at `at` while the text still starts with what was
// printed of it (its checksum `sum`), else at its start, since a retried request replaced it.
fn resume(text: &str, at: usize, sum: u64) -> usize {
    if at > 0 && text.get(..at).is_some_and(|printed| checksum(printed) == sum) { at } else { 0 }
}

fn checksum(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

// Where the last top-level block of streaming Markdown starts, by the parser render.rs draws it
// with. Nothing written after can change a block before it, while the last can still grow: a
// paragraph or a list goes on, a table takes rows, a code fence stays open until it closes.
fn finished_blocks(text: &str) -> usize {
    let (mut depth, mut last) = (0, 0);
    for (event, range) in Parser::new_ext(text, render::MARKDOWN).into_offset_iter() {
        if depth == 0 {
            last = range.start;
        }
        match event {
            Md::Start(_) => depth += 1,
            Md::End(_) => depth -= 1,
            _ => {}
        }
    }
    last
}
