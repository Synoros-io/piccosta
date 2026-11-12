// TeX in the terminal. A formula is typeset by RaTeX (KaTeX's layout and fonts, in Rust) into a
// PNG and shown through the kitty graphics protocol's Unicode placeholders, which Ghostty, kitty
// and WezTerm draw. The image goes to the terminal once; its cells are ordinary characters in the
// screen buffer, so a formula scrolls and clips with the text around it. Where the terminal shows
// no images, term-maths draws the formula as Unicode text, and what neither can draw stays as its
// source.
//
// PICCOSTA_TEX=image or =text skips asking the terminal and picks the mode. Inside tmux, images
// need `set -g allow-passthrough on`; without it the query goes unanswered and text is used.

use base64::Engine as _;
use ratatui::style::Color;
use ratex_types::color::Color as TexColor;
use ratex_types::math_style::MathStyle;
use std::collections::HashMap;
use std::io::Write;
#[cfg(unix)]
use std::time::{Duration, Instant};

const MODE_ENV: &str = "PICCOSTA_TEX";
// A TeX em as a fraction of a cell's height. Terminal fonts set a line at about 1.25 em, so a
// formula's letters come out the size of the text beside them.
const EM_PER_CELL_HEIGHT: f32 = 0.8;
// How long the terminal gets to answer the startup query, and how long past tmux's own answer
// (tmux answers DA1 itself) the outer terminal's graphics reply may still take.
#[cfg(unix)]
const QUERY_WAIT: Duration = Duration::from_millis(500);
#[cfg(unix)]
const TMUX_GRACE: Duration = Duration::from_millis(150);
// A cell's size in pixels when the terminal doesn't say.
const FALLBACK_CELL: (u32, u32) = (10, 20);
// Base64 bytes per graphics command; the protocol's limit is 4096.
const CHUNK: usize = 4096;
// The protocol's placeholder character. Each cell adds a diacritic for its row and one for its
// column in the image, and its foreground color names the image.
const PLACEHOLDER: char = '\u{10EEEE}';
// The protocol's row/column diacritics, in order (kitty's rowcolumn-diacritics list, as ratatex,
// MIT, carries it).
const DIACRITICS: &str = concat!(
    "\u{305}\u{30D}\u{30E}\u{310}\u{312}\u{33D}\u{33E}\u{33F}\u{346}\u{34A}\u{34B}\u{34C}\u{350}",
    "\u{351}\u{352}\u{357}\u{35B}\u{363}\u{364}\u{365}\u{366}\u{367}\u{368}\u{369}\u{36A}\u{36B}",
    "\u{36C}\u{36D}\u{36E}\u{36F}\u{483}\u{484}\u{485}\u{486}\u{487}\u{592}\u{593}\u{594}\u{595}",
    "\u{597}\u{598}\u{599}\u{59C}\u{59D}\u{59E}\u{59F}\u{5A0}\u{5A1}\u{5A8}\u{5A9}\u{5AB}\u{5AC}",
    "\u{5AF}\u{5C4}\u{610}\u{611}\u{612}\u{613}\u{614}\u{615}\u{616}\u{617}\u{657}\u{658}\u{659}",
    "\u{65A}\u{65B}\u{65D}\u{65E}\u{6D6}\u{6D7}\u{6D8}\u{6D9}\u{6DA}\u{6DB}\u{6DC}\u{6DF}\u{6E0}",
    "\u{6E1}\u{6E2}\u{6E4}\u{6E7}\u{6E8}\u{6EB}\u{6EC}\u{730}\u{732}\u{733}\u{735}\u{736}\u{73A}",
    "\u{73D}\u{73F}\u{740}\u{741}\u{743}\u{745}\u{747}\u{749}\u{74A}\u{7EB}\u{7EC}\u{7ED}\u{7EE}",
    "\u{7EF}\u{7F0}\u{7F1}\u{7F3}\u{816}\u{817}\u{818}\u{819}\u{81B}\u{81C}\u{81D}\u{81E}\u{81F}",
    "\u{820}\u{821}\u{822}\u{823}\u{825}\u{826}\u{827}\u{829}\u{82A}\u{82B}\u{82C}\u{82D}\u{951}",
    "\u{953}\u{954}\u{F82}\u{F83}\u{F86}\u{F87}\u{135D}\u{135E}\u{135F}\u{17DD}\u{193A}\u{1A17}",
    "\u{1A75}\u{1A76}\u{1A77}\u{1A78}\u{1A79}\u{1A7A}\u{1A7B}\u{1A7C}\u{1B6B}\u{1B6D}\u{1B6E}\u{1B6F}",
    "\u{1B70}\u{1B71}\u{1B72}\u{1B73}\u{1CD0}\u{1CD1}\u{1CD2}\u{1CDA}\u{1CDB}\u{1CE0}\u{1DC0}\u{1DC1}",
    "\u{1DC3}\u{1DC4}\u{1DC5}\u{1DC6}\u{1DC7}\u{1DC8}\u{1DC9}\u{1DCB}\u{1DCC}\u{1DD1}\u{1DD2}\u{1DD3}",
    "\u{1DD4}\u{1DD5}\u{1DD6}\u{1DD7}\u{1DD8}\u{1DD9}\u{1DDA}\u{1DDB}\u{1DDC}\u{1DDD}\u{1DDE}\u{1DDF}",
    "\u{1DE0}\u{1DE1}\u{1DE2}\u{1DE3}\u{1DE4}\u{1DE5}\u{1DE6}\u{1DFE}\u{20D0}\u{20D1}\u{20D4}\u{20D5}",
    "\u{20D6}\u{20D7}\u{20DB}\u{20DC}\u{20E1}\u{20E7}\u{20E9}\u{20F0}\u{2CEF}\u{2CF0}\u{2CF1}\u{2DE0}",
    "\u{2DE1}\u{2DE2}\u{2DE3}\u{2DE4}\u{2DE5}\u{2DE6}\u{2DE7}\u{2DE8}\u{2DE9}\u{2DEA}\u{2DEB}\u{2DEC}",
    "\u{2DED}\u{2DEE}\u{2DEF}\u{2DF0}\u{2DF1}\u{2DF2}\u{2DF3}\u{2DF4}\u{2DF5}\u{2DF6}\u{2DF7}\u{2DF8}",
    "\u{2DF9}\u{2DFA}\u{2DFB}\u{2DFC}\u{2DFD}\u{2DFE}\u{2DFF}\u{A66F}\u{A67C}\u{A67D}\u{A6F0}\u{A6F1}",
    "\u{A8E0}\u{A8E1}\u{A8E2}\u{A8E3}\u{A8E4}\u{A8E5}\u{A8E6}\u{A8E7}\u{A8E8}\u{A8E9}\u{A8EA}\u{A8EB}",
    "\u{A8EC}\u{A8ED}\u{A8EE}\u{A8EF}\u{A8F0}\u{A8F1}\u{AAB0}\u{AAB2}\u{AAB3}\u{AAB7}\u{AAB8}\u{AABE}",
    "\u{AABF}\u{AAC1}\u{FE20}\u{FE21}\u{FE22}\u{FE23}\u{FE24}\u{FE25}\u{FE26}\u{10A0F}\u{10A38}",
    "\u{1D185}\u{1D186}\u{1D187}\u{1D188}\u{1D189}\u{1D1AA}\u{1D1AB}\u{1D1AC}\u{1D1AD}\u{1D242}",
    "\u{1D243}\u{1D244}",
);

// What a formula became.
#[derive(Clone)]
pub enum Formula {
    // An uploaded image and the cells it fills.
    Image { id: u32, columns: u16, rows: u16 },
    // term-maths' drawing, one string per row.
    Text(Vec<String>),
}

pub struct Tex {
    images: bool,
    tmux: bool,
    // A cell's size in pixels.
    cell: (u32, u32),
    // The glyphs' color: light on a dark terminal, dark on a light one.
    ink: TexColor,
    // Every formula drawn so far, by source, block or inline, and the widest it may be.
    cache: HashMap<(String, bool, u16), Option<Formula>>,
    first_id: u32,
    ids: std::sync::Arc<std::sync::Mutex<(u32, Vec<u32>)>>,
    epoch: std::sync::Arc<std::sync::atomic::AtomicU64>,
    // Uploads waiting to be written before the next frame.
    pending: Vec<(u32, Vec<u8>)>,
}

impl Tex {
    // Ask the terminal whether it shows kitty images, how big a cell is, and whether its
    // background is dark. Call in raw mode, before anything else reads stdin.
    pub fn detect() -> Tex {
        let tmux = std::env::var_os("TMUX").is_some();
        let forced = std::env::var(MODE_ENV).ok();
        let replies = if forced.is_some() { String::new() } else { query(tmux) };
        let images = match forced.as_deref() {
            Some(mode) => mode == "image",
            None => replies.contains("_Gi=31;OK"),
        };
        let shade = if background_is_light(&replies) { 0.12 } else { 0.88 };
        // Image ids are per terminal, so two piccostas in one window start apart.
        let first_id = (std::process::id() % 4096) << 12 | 1;
        Tex { images, tmux, cell: cell_size(&replies), ink: TexColor::rgb(shade, shade, shade), cache: HashMap::new(), first_id, ids: std::sync::Arc::new(std::sync::Mutex::new((first_id, Vec::new()))), epoch: Default::default(), pending: Vec::new() }
    }

    // A document worker shares only the transport's id allocator, never transcript formula rows.
    pub(super) fn fork(&self) -> Self {
        Self { images: self.images, tmux: self.tmux, cell: self.cell, ink: self.ink, cache: HashMap::new(), first_id: self.first_id, ids: self.ids.clone(), epoch: self.epoch.clone(), pending: Vec::new() }
    }
    pub(super) fn epoch(&self) -> u64 { self.epoch.load(std::sync::atomic::Ordering::Relaxed) }
    pub(super) fn receive(&mut self, bytes: Vec<u8>) { if !bytes.is_empty() { self.pending.push((0, bytes)); } }
    pub(super) fn pending(&mut self) -> Vec<u8> { self.prepared().into_values().flatten().collect() }
    pub(super) fn prepared(&mut self) -> HashMap<u32, Vec<u8>> { std::mem::take(&mut self.pending).into_iter().collect() }
    pub(super) fn recycle(&self, ids: &[u32]) { self.ids.lock().unwrap().1.extend_from_slice(ids); }
    pub(super) fn block_ids(&mut self) -> Vec<u32> {
        self.cache.drain().filter_map(|(_, formula)| match formula { Some(Formula::Image { id, .. }) => Some(id), _ => None }).collect()
    }

    pub fn images(&self) -> bool {
        self.images
    }

    // Text only: --single prints into the scrollback, where an image placed by a frame isn't taken back.
    pub fn text_only(mut self) -> Tex {
        self.images = false;
        self
    }

    // A formula, typeset once and kept. `display` is block math ($$..$$), set in TeX's display
    // style at its natural size; inline math is set in text style and scaled to one row so it
    // stays in its line. Output: None when nothing can draw it.
    pub fn formula(&mut self, tex: &str, display: bool, max_columns: u16) -> Option<&Formula> {
        let key = (tex.to_string(), display, max_columns);
        if !self.cache.contains_key(&key) {
            let image = if self.images { self.image(tex, display, max_columns) } else { None };
            let formula = image.or_else(|| text(tex).map(Formula::Text));
            self.cache.insert(key.clone(), formula);
        }
        self.cache.get(&key).and_then(Option::as_ref)
    }

    fn image(&mut self, tex: &str, display: bool, max_columns: u16) -> Option<Formula> {
        let style = if display { MathStyle::Display } else { MathStyle::Text };
        let ink = self.ink;
        let png = quietly(|| {
            let nodes = ratex_parser::parse(tex).ok()?;
            let layout = ratex_layout::layout(&nodes, &ratex_layout::LayoutOptions { style, color: ink, ..Default::default() });
            let options = ratex_render::RenderOptions {
                font_size: self.cell.1 as f32 * EM_PER_CELL_HEIGHT,
                padding: 1.0,
                background_color: TexColor::new(0.0, 0.0, 0.0, 0.0),
                ..Default::default()
            };
            ratex_render::render_to_png(&ratex_layout::to_display_list(&layout), &options).ok()
        })??;
        // The PNG's size, from its header.
        let width = u32::from_be_bytes(png.get(16..20)?.try_into().ok()?);
        let height = u32::from_be_bytes(png.get(20..24)?.try_into().ok()?);
        let (cell_width, cell_height) = self.cell;
        // The cells it fills; inline math never taller than a row. The terminal fits the image
        // into them without distorting it, so narrowing the box shrinks the formula.
        let scale = if display { 1.0 } else { (cell_height as f32 / height as f32).min(1.0) };
        let mut columns = ((width as f32 * scale) / cell_width as f32).ceil().max(1.0) as u32;
        let mut rows = if display { height.div_ceil(cell_height).max(1) } else { 1 };
        if columns > u32::from(max_columns) {
            rows = (rows * u32::from(max_columns)).div_ceil(columns).max(1);
            columns = u32::from(max_columns).max(1);
        }
        let limit = u32::from(max_cells());
        if rows > limit { columns = (columns * limit).div_ceil(rows).max(1); rows = limit; }
        Some(self.upload(&png, columns.min(limit) as u16, rows as u16))
    }

    // PNGs are already decoded, fitted and encoded by the scratchpad's disk worker. The UI only
    // queues this bounded upload; its placeholder cells use exactly the same transport as TeX.
    pub(super) fn upload(&mut self, png: &[u8], columns: u16, rows: u16) -> Formula {
        let mut ids = self.ids.lock().unwrap();
        let id = ids.1.pop().unwrap_or_else(|| { let id = ids.0; ids.0 += 1; id });
        drop(ids);
        let mut bytes = Vec::new();
        let encoded = base64::engine::general_purpose::STANDARD.encode(png);
        let chunks: Vec<&[u8]> = encoded.as_bytes().chunks(CHUNK).collect();
        for (index, chunk) in chunks.iter().enumerate() {
            let more = u8::from(index + 1 < chunks.len());
            let control = if index == 0 { format!("a=t,f=100,t=d,i={id},q=2,m={more}") } else { format!("m={more}") };
            command(&mut bytes, &control, chunk, self.tmux);
        }
        command(&mut bytes, &format!("a=p,U=1,i={id},c={columns},r={rows},q=2"), b"", self.tmux);
        self.pending.push((id, bytes));
        Formula::Image { id, columns: columns.min(max_cells()), rows: rows.min(max_cells()) }
    }

    pub(super) fn discard(&mut self, id: u32) {
        let mut bytes = Vec::new();
        command(&mut bytes, &format!("a=d,d=I,i={id},q=2"), b"", self.tmux);
        self.receive(bytes);
        self.recycle(&[id]);
    }

    pub(super) fn cell(&self) -> (u32, u32) { self.cell }

    // Write the uploads made since the last frame. Call before drawing it.
    pub fn flush(&mut self, out: &mut impl Write) -> std::io::Result<()> {
        if !self.pending.is_empty() {
            for (_, bytes) in std::mem::take(&mut self.pending) { out.write_all(&bytes)?; }
            out.flush()?;
        }
        Ok(())
    }

    // Forget every image this process uploaded (on exit, or when the terminal lost them).
    pub fn clear(&mut self, out: &mut impl Write) -> std::io::Result<()> {
        let mut bytes = Vec::new();
        let mut ids = self.ids.lock().unwrap();
        if ids.0 > self.first_id {
            command(&mut bytes, &format!("a=d,d=R,x={},y={},q=2", self.first_id, ids.0 - 1), b"", self.tmux);
        }
        ids.1.clear();
        self.epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        drop(ids);
        self.cache.clear();
        self.receive(bytes);
        self.flush(out)
    }
}

// Placeholder foregrounds carry their image id; ordinary colored text doesn't request uploads.
pub(super) fn row_ids<'a>(row: &'a ratatui::text::Line<'_>) -> impl Iterator<Item = u32> + 'a {
    row.spans.iter().filter_map(|span| match span.style.fg { Some(Color::Rgb(red, green, blue)) if span.content.contains(PLACEHOLDER) => Some(u32::from(red) << 16 | u32::from(green) << 8 | u32::from(blue)), _ => None })
}

// Kitty's coordinate alphabet is finite; fitting happens before upload, never by repeating row 0.
pub(super) fn max_cells() -> u16 { DIACRITICS.chars().count() as u16 }

// One row of an image's placeholder cells, and the color that names the image.
pub fn placeholders(id: u32, row: u16, columns: u16) -> (String, Color) {
    static DIACRITIC_CELLS: std::sync::LazyLock<Vec<char>> = std::sync::LazyLock::new(|| DIACRITICS.chars().collect());
    let diacritics = &*DIACRITIC_CELLS;
    let diacritic = |index: u16| diacritics[usize::from(index)];
    if usize::from(row) >= diacritics.len() { return (String::new(), Color::Reset); }
    let columns = columns.min(diacritics.len() as u16);
    let mut cells = String::with_capacity(usize::from(columns) * 8);
    for column in 0..columns {
        cells.push(PLACEHOLDER);
        cells.push(diacritic(row));
        cells.push(diacritic(column));
    }
    (cells, Color::Rgb((id >> 16) as u8, (id >> 8) as u8, id as u8))
}

// term-maths' drawing of a formula, or None when it couldn't parse it (it then prints the
// source, backslashes and all).
fn text(tex: &str) -> Option<Vec<String>> {
    let drawn = quietly(|| term_maths::render(tex).to_string())?;
    let rows: Vec<String> = drawn.lines().map(|line| line.trim_end().to_string()).collect();
    (!drawn.contains('\\') && rows.iter().any(|row| !row.is_empty())).then_some(rows)
}

thread_local! {
    // Set while a formula is being drawn: a panic there is caught, and the panic hook leaves the
    // screen alone instead of restoring the terminal.
    pub static DRAWING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Run a typesetter on text a model wrote; a panic in it counts as "can't draw this".
fn quietly<T>(work: impl FnOnce() -> T) -> Option<T> {
    DRAWING.with(|drawing| drawing.set(true));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).ok();
    DRAWING.with(|drawing| drawing.set(false));
    result
}

// One graphics command; inside tmux, wrapped for passthrough with its escapes doubled.
fn command(out: &mut Vec<u8>, control: &str, payload: &[u8], tmux: bool) {
    let mut body = format!("\x1b_G{control}").into_bytes();
    if !payload.is_empty() {
        body.push(b';');
        body.extend_from_slice(payload);
    }
    body.extend_from_slice(b"\x1b\\");
    if !tmux {
        out.extend(body);
        return;
    }
    out.extend_from_slice(b"\x1bPtmux;");
    for byte in body {
        if byte == 0x1b {
            out.push(0x1b);
        }
        out.push(byte);
    }
    out.extend_from_slice(b"\x1b\\");
}

// Ask for a graphics reply, the cell size, the background color, then DA1, whose answer every
// terminal gives; read what comes back.
#[cfg(windows)]
fn query(_tmux: bool) -> String { String::new() }

#[cfg(unix)]
fn query(tmux: bool) -> String {
    let mut ask = Vec::new();
    command(&mut ask, "i=31,s=1,v=1,a=q,t=d,f=24", b"AAAA", tmux);
    ask.extend_from_slice(b"\x1b[16t\x1b]11;?\x1b\\\x1b[c");
    let mut stdout = std::io::stdout();
    if stdout.write_all(&ask).and_then(|()| stdout.flush()).is_err() {
        return String::new();
    }
    let mut replies = Vec::new();
    let mut deadline = Instant::now() + QUERY_WAIT;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let mut poll = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
        // SAFETY: one pollfd for stdin, then a read into a buffer of the length given.
        let read = unsafe {
            if left.is_zero() || libc::poll(&mut poll, 1, left.as_millis() as i32) <= 0 {
                break;
            }
            let mut buffer = [0u8; 1024];
            let read = libc::read(0, buffer.as_mut_ptr().cast(), buffer.len());
            if read <= 0 {
                break;
            }
            buffer[..read as usize].to_vec()
        };
        replies.extend(read);
        let text = String::from_utf8_lossy(&replies);
        let answered_da1 = text.find("\x1b[?").is_some_and(|start| text[start..].contains('c'));
        if answered_da1 {
            // Inside tmux the outer terminal's graphics reply can come after tmux's own DA1.
            if !tmux || text.contains("_Gi=31;") {
                break;
            }
            deadline = deadline.min(Instant::now() + TMUX_GRACE);
        }
    }
    String::from_utf8_lossy(&replies).into_owned()
}

// A cell's size in pixels: the terminal's answer to CSI 16 t, else the window's pixel size over
// its cells, else a guess.
fn cell_size(replies: &str) -> (u32, u32) {
    let answered = replies.find("\x1b[6;").and_then(|start| {
        let rest = &replies[start + 4..];
        let (numbers, _) = rest.split_once('t')?;
        let (height, width) = numbers.split_once(';')?;
        Some((width.parse().ok()?, height.parse().ok()?))
    });
    let from_window = || {
        let size = crossterm::terminal::window_size().ok()?;
        (size.width > 0 && size.columns > 0 && size.rows > 0)
            .then(|| (u32::from(size.width / size.columns), u32::from(size.height / size.rows)))
    };
    answered.filter(|&(width, height): &(u32, u32)| width > 0 && height > 0).or_else(from_window).unwrap_or(FALLBACK_CELL)
}

// Whether the terminal answered OSC 11 with a light background.
fn background_is_light(replies: &str) -> bool {
    let Some(start) = replies.find("]11;rgb:") else { return false };
    let channels: Vec<f32> = replies[start + 8..]
        .split(['/', '\x1b', '\x07'])
        .take(3)
        .filter_map(|channel| u32::from_str_radix(channel.get(..2)?, 16).ok())
        .map(|value| value as f32 / 255.0)
        .collect();
    channels.len() == 3 && 0.299 * channels[0] + 0.587 * channels[1] + 0.114 * channels[2] > 0.5
}
