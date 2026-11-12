// A document, never a session. Reads, image decoding and cached Markdown block layout all run
// on the existing asker. The UI only takes a window of rows and adopts bounded graphics data.

use super::{App, Editor, Pane, Target, Update, render, tex};
use crate::scratchpad::{self, stamp, Stamp};
use pulldown_cmark::{Event, Parser, Tag};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::Duration;

// The prompt loader's settle window. Eight 512² RGBA previews per board keep four panes' decoded
// previews within 32 MiB; source decoding uses the tools' 16M-pixel guard. Compressed reads get
// one quarter of that guard's RGBA allowance, and PNG output gets room for encoding overhead.
const SAVE_SETTLE: Duration = Duration::from_secs(2);
const MAX_ASSETS: usize = 8;
const PREVIEW_PIXELS: u64 = 512 * 512;
const SOURCE_BYTES: usize = crate::tools::image::MAX_PLUGIN_RESPONSE_PIXELS as usize;
const PREVIEW_BYTES: usize = PREVIEW_PIXELS as usize * 8;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Fit { width: u16, columns: u16, rows: u16, cell: (u32, u32), images: bool }
struct Picture { png: Vec<u8>, columns: u16, rows: u16 }
struct Asset { stamp: Option<Stamp>, picture: Option<Arc<Picture>>, problem: Option<String>, deferred: bool }

#[derive(Default)]
pub(super) struct View {
    text: String,
    stamp: Option<Stamp>,
    references: Vec<String>,
    assets: HashMap<String, Arc<Asset>>,
    fit: Fit,
    epoch: u64,
    blocks: Vec<Arc<render::DocumentBlock>>,
    images: HashMap<String, tex::Formula>,
    graphics: Vec<u8>,
    owned: Vec<u32>,
    created: Vec<u32>,
    retired: Vec<u32>,
}
impl View {
    fn rows(&self) -> impl Iterator<Item = &ratatui::text::Line<'static>> { self.blocks.iter().flat_map(|block| &block.rows) }
}

pub(super) struct Scratchpad {
    pub path: PathBuf,
    pub cwd: PathBuf,
    pub draft: Editor,
    view: Arc<View>,
    ticket: Arc<()>,
    pending: bool,
    appending: bool,
    ready: Vec<Stamp>,
    fit: Fit,
}

fn picture(path: &Path, fit: Fit, pixels_left: &mut u64) -> Result<Picture, String> {
    let mut bytes = Vec::new();
    scratchpad::open(path, false)?.take(SOURCE_BYTES as u64 + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
    if bytes.len() > SOURCE_BYTES { return Err("image exceeds the compressed-source budget".into()); }
    if !matches!(crate::tools::image::sniff_mime_type(&bytes), Some("image/png" | "image/jpeg")) { return Err("only local PNG/JPEG images are shown".into()); }
    let image = crate::tools::image::decode_guarded(&bytes, pixels_left)?;
    let scale = (f64::from(fit.columns) * f64::from(fit.cell.0) / f64::from(image.width()))
        .min(f64::from(fit.rows) * f64::from(fit.cell.1) / f64::from(image.height()))
        .min((PREVIEW_PIXELS as f64 / (f64::from(image.width()) * f64::from(image.height()))).sqrt()).min(1.0);
    let image = image.thumbnail((f64::from(image.width()) * scale).max(1.0) as u32, (f64::from(image.height()) * scale).max(1.0) as u32);
    let columns = image.width().div_ceil(fit.cell.0).max(1).min(u32::from(fit.columns)) as u16;
    let rows = image.height().div_ceil(fit.cell.1).max(1).min(u32::from(fit.rows)) as u16;
    let png = crate::tools::image::encode_png(&image).ok_or("could not encode image preview")?;
    if png.len() > PREVIEW_BYTES { return Err("image exceeds the preview budget".into()); }
    Ok(Picture { png, columns, rows })
}

// Read only changed versions, retaining the last good view while a save settles. A budget-deferred
// asset gets another scan; invalid files don't. Relative URLs resolve from the Markdown document,
// and to_file_path rejects network schemes and remote hosts without fetching anything.
fn scan(path: &Path, previous: &View, fit: Fit, ready: &[Stamp], mut tex: tex::Tex) -> Result<Option<View>, String> {
    let lock = scratchpad::lock(path)?;
    if previous.stamp.is_none() { scratchpad::open(path, true)?; }
    let current = stamp(path, None)?;
    if ready.is_empty() && current.modified.elapsed().is_ok_and(|age| age < SAVE_SETTLE) { return Ok(None); }
    let changed = previous.stamp.as_ref() != Some(&current);
    let text = if changed || !ready.is_empty() { Cow::Owned(scratchpad::read(path)?) } else { Cow::Borrowed(previous.text.as_str()) };
    if stamp(path, None)? != current || (current.modified.elapsed().is_ok_and(|age| age < SAVE_SETTLE) && !ready.contains(&stamp(path, Some(&text))?)) { return Ok(None); }
    drop(lock);
    let references = if changed {
        let mut references = Vec::new();
        for destination in Parser::new_ext(&text, render::MARKDOWN).filter_map(|event| match event { Event::Start(Tag::Image { dest_url, .. }) => Some(dest_url.into_string()), _ => None }) {
            if !references.contains(&destination) { references.push(destination); }
            if references.len() == MAX_ASSETS { break; }
        }
        Cow::Owned(references)
    } else { Cow::Borrowed(previous.references.as_slice()) };
    let epoch = tex.epoch();
    let mut assets = HashMap::new();
    let mut updated = changed || fit != previous.fit || epoch != previous.epoch;
    let mut pixels_left = crate::tools::image::MAX_PLUGIN_RESPONSE_PIXELS;
    if fit.images {
        for reference in references.iter() {
            let Some(path) = url::Url::from_directory_path(path.parent().unwrap()).ok().and_then(|base| base.join(reference).ok()) .and_then(|url| url.to_file_path().ok()) else { continue };
            let current = stamp(&path, None).ok();
            let old = previous.assets.get(reference);
            if fit == previous.fit && old.is_some_and(|old| old.stamp == current && !old.deferred) { assets.insert(reference.clone(), old.unwrap().clone()); continue; }
            let settling = current.as_ref().is_some_and(|stamp| stamp.modified.elapsed().is_ok_and(|age| age < SAVE_SETTLE));
            let decoded = if settling { Err("image save settling".into()) } else if current.is_some() { picture(&path, fit, &mut pixels_left) } else { Err("image unavailable or not a regular file".into()) };
            if stamp(&path, None).ok() != current { return Ok(None); }
            let deferred = settling || (pixels_left < crate::tools::image::MAX_PLUGIN_RESPONSE_PIXELS && decoded.as_ref().is_err_and(|error| error.contains("decoded-pixel budget exceeded")));
            let problem = decoded.as_ref().err().filter(|_| !deferred).cloned();
            let picture = decoded.ok().map(Arc::new).or_else(|| old.and_then(|old| old.picture.clone()).filter(|_| fit == previous.fit));
            assets.insert(reference.clone(), Arc::new(Asset { stamp: current, picture, problem, deferred }));
            updated = true;
        }
    }
    if stamp(path, None)? != current || !updated { return Ok(None); }
    let (mut images, mut created) = (HashMap::new(), Vec::new());
    for (reference, asset) in &assets {
        if let Some(picture) = &asset.picture {
            let old = (epoch == previous.epoch).then(|| previous.assets.get(reference)).flatten().and_then(|asset| asset.picture.as_ref());
            let retained = old.is_some_and(|old| Arc::ptr_eq(old, picture)).then(|| previous.images.get(reference).cloned()).flatten();
            let formula = retained.unwrap_or_else(|| { let formula = tex.upload(&picture.png, picture.columns, picture.rows); if let tex::Formula::Image { id, .. } = formula { created.push(id); } formula });
            images.insert(reference.clone(), formula);
        }
    }
    let graphics = tex.pending();
    let previous_blocks = if fit.width == previous.fit.width && epoch == previous.epoch { previous.blocks.as_slice() } else { &[] };
    let (blocks, formulas) = render::document(&text, fit.width, &mut tex, &images, previous_blocks);
    created.extend(formulas);
    let owned: HashSet<_> = blocks.iter().flat_map(|block| block.ids.iter().copied()).chain(images.values().filter_map(|formula| match formula { tex::Formula::Image { id, .. } => Some(*id), _ => None })).collect();
    let retired = previous.owned.iter().copied().filter(|id| !owned.contains(id)).collect();
    Ok(Some(View { text: text.into_owned(), stamp: Some(current), references: references.into_owned(), assets, fit, epoch, blocks, images, graphics, owned: owned.into_iter().collect(), created, retired }))
}

impl App {
    pub(super) fn draft(&self) -> &Editor { match self.panes[self.focus].target { Target::Session(slot) => &self.slots[slot].draft, Target::Scratchpad(board) => &self.scratchpads[board].draft } }
    pub(super) fn draft_mut(&mut self) -> &mut Editor { match self.panes[self.focus].target { Target::Session(_) => &mut self.focused().draft, Target::Scratchpad(board) => &mut self.scratchpads[board].draft } }
    pub(super) fn focus_cwd(&self) -> &Path { match self.panes[self.focus].target { Target::Session(slot) => &self.slots[slot].cwd, Target::Scratchpad(board) => &self.scratchpads[board].cwd } }
    pub(super) fn focus_busy(&self) -> bool { matches!(self.panes[self.focus].target, Target::Session(slot) if self.slots[slot].busy()) }
    pub(super) fn append_pending(&self) -> bool { matches!(self.panes[self.focus].target, Target::Scratchpad(board) if self.scratchpads[board].appending) }

    pub(super) fn scratch_index(&mut self, path: PathBuf, cwd: PathBuf) -> usize {
        if let Some(index) = self.scratchpads.iter().position(|board| board.path == path) { return index; }
        self.scratchpads.push(Scratchpad { path, cwd, draft: Editor::default(), view: Arc::new(View::default()), ticket: Arc::new(()), pending: false, appending: false, ready: Vec::new(), fit: Fit::default() });
        self.scratchpads.len() - 1
    }

    pub(super) fn scratch(&mut self, argument: &str) {
        if self.single { self.notice = Some("Scratchpad needs the full TUI; start without --single or --inline.".into()); return; }
        if !matches!(argument, "" | "edit") { self.notice = Some("/scratch or /scratch edit".into()); return; }
        let cwd = if self.focus_cwd().as_os_str().is_empty() { self.host.cwd.clone() } else { self.focus_cwd().to_path_buf() };
        let (directory, edit) = (self.host.agent_dir.clone(), argument == "edit");
        if self.asks.send(Box::new(move |_| { let path = scratchpad::path(&cwd, &directory); (0, Update::ScratchPath(cwd, path, edit)) })).is_err() { self.notice = Some("Scratchpad resolver stopped.".into()); }
    }

    pub(super) fn scratch_path(&mut self, cwd: PathBuf, path: PathBuf, edit: bool) {
        if self.quit || self.restart_waiting { return; }
        let board = if let Some(pane) = self.panes.iter().position(|pane| matches!(pane.target, Target::Scratchpad(board) if self.scratchpads[board].path == path)) {
            self.focus = pane;
            let Target::Scratchpad(board) = self.panes[pane].target else { unreachable!() }; board
        } else {
            if self.panes.len() >= super::MAX_PANES { self.notice = Some("All four panes are occupied; close a pane before /scratch.".into()); return; }
            let board = self.scratch_index(path.clone(), cwd);
            self.panes.push(Pane { target: Target::Scratchpad(board), scroll: 0, anchor: None });
            self.focus = self.panes.len() - 1;
            board
        };
        if edit {
            let ticket = Arc::downgrade(&self.scratchpads[board].ticket);
            if self.asks.send(Box::new(move |_| (0, Update::ScratchEdit(board, ticket, scratchpad::lock(&path).and_then(|_lock| scratchpad::open(&path, true).map(drop)))))).is_err() { self.notice = Some("Cannot open scratchpad editor: disk worker stopped.".into()); }
        }
        self.scan_scratchpads();
    }

    pub(super) fn append_scratch(&mut self, text: String) {
        let Target::Scratchpad(board) = self.panes[self.focus].target else { return };
        let (path, paragraph) = (self.scratchpads[board].path.clone(), text.clone());
        if self.deliveries.send(Box::new(move |_| (0, Update::Appended(board, scratchpad::append(&path, &paragraph), paragraph)))).is_err() {
            self.scratchpads[board].draft.insert(&text);
            self.notice = Some("Scratchpad append failed: the disk worker stopped; draft kept.".into());
        } else { self.scratchpads[board].appending = true; }
    }

    pub(super) fn appended(&mut self, board: usize, result: Result<Option<Stamp>, String>, text: String) {
        let board = &mut self.scratchpads[board];
        board.appending = false;
        self.notice = Some(match result {
            Ok(ready) => { if let Some(ready) = ready { board.ready.push(ready); } "Paragraph appended.".into() }
            Err(error) => { let draft = board.draft.take(); board.draft = Editor::with(if draft.is_empty() { text } else { format!("{text}\n\n{draft}") }); format!("Scratchpad append failed: {error}; draft kept.") }
        });
        self.scan_scratchpads();
    }

    // A completion attests committed bytes, not permission to read a fresh outside save.
    // Keep arrivals during a scan and coalesce them into its successor.
    pub(super) fn scratch_ready(&mut self, cwd: &Path, ready: Stamp) {
        for board in &mut self.scratchpads {
            if (board.cwd == cwd || board.path == ready.path || board.view.stamp.as_ref().is_some_and(|stamp| stamp.path == ready.path)) && !board.ready.contains(&ready) { board.ready.push(ready.clone()); }
        }
        self.scan_scratchpads();
    }

    pub(super) fn scan_scratchpads(&mut self) {
        if self.restart_waiting || self.quit { return; }
        for index in 0..self.scratchpads.len() {
            let Some(pane) = self.panes.iter().position(|pane| pane.target == Target::Scratchpad(index)) else { self.scratchpads[index].ready.clear(); self.clear_scratch_images(index); continue };
            let rect = self.pane_rects.get(pane).copied().unwrap_or_default();
            let width = rect.width.saturating_sub(2 + u16::from(rect.x > self.pane_rects.first().map_or(0, |rect| rect.x)));
            let fit = Fit { width, columns: width.saturating_sub(2).max(1).min(tex::max_cells()), rows: rect.height.saturating_sub(super::PANE_MIN_HEIGHT).max(1).min(tex::max_cells()), cell: self.tex.cell(), images: self.tex.images() };
            let board = &mut self.scratchpads[index];
            board.fit = fit;
            if board.pending || board.appending { continue; }
            let (path, previous, ticket, ready, painter) = (board.path.clone(), board.view.clone(), Arc::downgrade(&board.ticket), board.ready.clone(), self.tex.fork());
            board.pending = self.asks.send(Box::new(move |_| { let result = if ticket.strong_count() == 0 { Ok(None) } else { scan(&path, &previous, fit, &ready, painter) }; (0, Update::Scratch(index, ticket, ready, Box::new(result))) })).is_ok();
        }
    }

    pub(super) fn scratch_result(&mut self, index: usize, ticket: Weak<()>, ready: Vec<Stamp>, result: Result<Option<View>, String>) {
        let current = ticket.ptr_eq(&Arc::downgrade(&self.scratchpads[index].ticket));
        if current { self.scratchpads[index].pending = false; }
        let superseded = current && (ready != self.scratchpads[index].ready || result.as_ref().ok().and_then(Option::as_ref).is_some_and(|view| view.fit != self.scratchpads[index].fit || view.epoch != self.tex.epoch()));
        if current && !superseded { self.scratchpads[index].ready.clear(); }
        match result {
            Ok(Some(mut view)) if current && !superseded => {
                for id in view.retired.drain(..) { self.tex.discard(id); }
                self.tex.receive(std::mem::take(&mut view.graphics));
                if let Some((reference, problem)) = view.assets.iter().find_map(|(reference, asset)| asset.problem.as_ref().map(|problem| (reference, problem))) { self.notice = Some(format!("Scratchpad image {reference}: {problem}; keeping its last preview or alt text.")); }
                let old = std::mem::replace(&mut self.scratchpads[index].view, Arc::new(view));
                self.retire(old);
            }
            Ok(Some(view)) => { self.tex.recycle(&view.created); self.retire(view); self.scan_scratchpads(); }
            Err(error) if current && !superseded => self.notice = Some(format!("Scratchpad keeps its last view: {error}")),
            _ => {},
        }
        if superseded { self.scan_scratchpads(); }
    }

    pub(super) fn scratch_edit_current(&self, index: usize, ticket: &Weak<()>) -> bool { ticket.ptr_eq(&Arc::downgrade(&self.scratchpads[index].ticket)) && self.panes.iter().any(|pane| pane.target == Target::Scratchpad(index)) }
    pub(super) fn invalidate_scratch(&mut self, index: usize) { let board = &mut self.scratchpads[index]; board.ticket = Arc::new(()); board.pending = false; }
    pub(super) fn clear_scratch_images(&mut self, index: usize) {
        if self.scratchpads[index].view.stamp.is_none() { return; }
        let old = std::mem::take(&mut self.scratchpads[index].view);
        for &id in &old.owned { self.tex.discard(id); }
        self.retire(old);
    }

    pub(super) fn draw_scratch(&mut self, frame: &mut ratatui::Frame, pane: usize, index: usize, inner: ratatui::layout::Rect) {
        use ratatui::{layout::Rect, style::{Modifier, Style}, text::{Line, Span}, widgets::Paragraph};
        let ([header, body, _, editor], editor_rows, cursor) = Self::editor_layout(&self.scratchpads[index].draft, inner, pane == self.focus, 0);
        frame.render_widget(Paragraph::new(Line::from(vec![Span::styled("scratchpad", Style::new().fg(super::theme::ACCENT).add_modifier(Modifier::BOLD)), Span::styled(format!("  {}", self.scratchpads[index].path.display()), Style::new().fg(super::theme::DIM))])), header);
        let view = self.scratchpads[index].view.clone();
        let length: usize = view.blocks.iter().map(|block| block.rows.len()).sum();
        let scroll = self.panes[pane].scroll.min(length.saturating_sub(usize::from(body.height)));
        if view.stamp.is_some() { self.panes[pane].scroll = scroll; }
        let end = length - scroll;
        let start = end.saturating_sub(usize::from(body.height));
        let shown = Rect { height: (end - start) as u16, ..body };
        // Lists and paragraphs may contain thousands of formulas. Only visible placeholder ids
        // take their prepared uploads, even when the entire document is one cached block.
        let mut first = 0;
        for block in &view.blocks {
            let next = first + block.rows.len();
            if first < end && next > start {
                let mut graphics = block.graphics.lock().unwrap();
                for row in &block.rows[start.saturating_sub(first)..(end - first).min(block.rows.len())] {
                    for id in tex::row_ids(row) { if let Some(bytes) = graphics.remove(&id) { self.tex.receive(bytes); } }
                }
            }
            first = next;
        }
        let rows: render::Rows = view.rows().skip(start).take(end - start).cloned().collect();
        frame.render_widget(Paragraph::new(rows.clone()), shown);
        self.body_rects.push(body);
        if self.selection.as_ref().is_some_and(|selection| selection.pane == pane) {
            let mut count = 0;
            let places: Vec<_> = view.rows().take(end).enumerate().filter_map(|(row, line)| { let place = (0, row, count); count += super::clip::solid(line); (row >= start).then_some((row - start, place)) }).collect();
            if let Some((from, to)) = self.select_rows(frame, pane, shown, &rows, &places) {
                let mut count = 0;
                let cells = view.rows().enumerate().map(|(row, line)| { let cells = super::clip::cells(line, (0, row, count)); count += super::clip::solid(line); cells });
                self.copied(super::clip::text(cells, from, to));
            }
        }
        if pane == self.focus { self.draw_editor(frame, editor, &editor_rows, cursor); }
    }
}
