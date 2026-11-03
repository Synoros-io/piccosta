// The terminal UI. Several sessions live at once in one piccosta: a sidebar lists them (this
// directory's earlier ones under them), one to four panes show them side by side, and the
// focused pane has the editor. Each live session runs on its own task, and its events come back
// here tagged with its slot, so a session keeps working while another one is on screen.

mod clip;
pub(crate) mod commands;
mod render;
mod state;
mod tex;

use crate::agent::{Event, Handle};
use crate::host::{Command, Done, Host};
use crate::message::AgentMessage;
use crossterm::event::{Event as Input, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind};
use futures_util::StreamExt;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph};
use ratatui::Frame;
use render::{Block as Part, Flags, Rows, ToolView};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use unicode_width::UnicodeWidthStr;

// The look: a few colors, used the same way everywhere.
pub mod theme {
    use ratatui::style::Color;
    pub const ACCENT: Color = Color::Rgb(0x5f, 0xb3, 0xa9);
    pub const USER: Color = Color::Rgb(0xe5, 0xa3, 0x5b);
    pub const DIM: Color = Color::Rgb(0x6c, 0x70, 0x80);
    pub const CODE: Color = Color::Rgb(0x9f, 0xc6, 0x8a);
    pub const MATH: Color = Color::Rgb(0xb8, 0xa1, 0xe3);
    pub const ERROR: Color = Color::Rgb(0xe8, 0x73, 0x6b);
    pub const OK: Color = Color::Rgb(0x7f, 0xbf, 0x7f);
    pub const ADDED: Color = Color::Rgb(0x7f, 0xbf, 0x7f);
    pub const REMOVED: Color = Color::Rgb(0xd9, 0x77, 0x6f);
}

// The status line when there's no notice: what's most used first, since a narrow screen cuts the
// end; /help has every key and command.
const HELP: &str = "enter send · esc stop · alt+↑↓ sessions · ^n new · ^s sessions · del twice deletes · drag copies, right-click pastes · /pair, /trio a second model · /help all keys · ^c ^c quit";
// How many of this directory's earlier sessions the sidebar lists, and of other projects': the most
// recently used projects, a few sessions each.
const RECENT_SESSIONS: usize = 30;
const OTHER_PROJECTS: usize = 8;
const RECENT_PER_OTHER_PROJECT: usize = 3;
const SIDEBAR_WIDTH: u16 = 30;
// The widest a picker gets: room for a project, a title and an age.
const PICKER_MAX_WIDTH: u16 = 110;
// Below this screen width the sidebar steps aside for the panes.
const SIDEBAR_MIN_SCREEN: u16 = 90;
const MAX_PANES: usize = 4;
const EDITOR_MAX_ROWS: usize = 8;
// How often running sessions' spinners turn, and how many turns each of the working line's dots
// lasts (about 400 ms).
const FRAME: Duration = Duration::from_millis(80);
const WORKING_DOT_FRAMES: usize = 5;
const WHEEL_ROWS: usize = 3;
// How long a first ctrl+c waits for the second while sessions are running.
const QUIT_CONFIRM: Duration = Duration::from_secs(2);
// How long quitting waits for aborted sessions to write their last messages.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(3);
const NEW_TITLE: &str = "new session";
// How many of this directory's sessions /resume lists, and how many undo steps the editor keeps.
const PICKER_SESSIONS: usize = 500;
const UNDO_STEPS: usize = 100;
// Spaces a pasted tab becomes (pi's editor turns tabs into four).
const TAB_SPACES: &str = "    ";
// Clicks this close together make a double or triple click (GTK's default gtk-double-click-time).
const MULTI_CLICK: Duration = Duration::from_millis(400);
// How many agent scans go between looks at other piccostas' locks on the sessions listed (10 s).
const LOCK_SCAN_EVERY: u64 = 5;
// How often the sidebar looks again for the agents each session has running.
const AGENT_SCAN: Duration = Duration::from_secs(2);

// What comes back from a session's task.
enum Update {
    Event(Box<Event>),
    Done(Box<Done>),
    // A tool's question for the user (Tool::approval), with the call's tool and action, answered
    // with one of ANSWERS.
    Ask(&'static str, String, tokio::sync::oneshot::Sender<&'static str>),
}

// The answers to a tool's question: yes this once, yes for the rest of the session, no.
const ANSWERS: [&str; 3] = ["allow once", "allow for this session", "refuse"];
// How much of the asking session's title a question's picker shows.
const ASKER_CHARS: usize = 30;

#[derive(PartialEq)]
enum State {
    Idle,
    Running,
    Retrying(String),
    Compacting,
}

// One session as the UI keeps it; `live` is None for an earlier session not opened yet.
struct Slot {
    title: String,
    path: Option<PathBuf>,
    cwd: PathBuf,
    live: Option<(Handle, UnboundedSender<Command>)>,
    // Its tools' shared state, for /ps and /stop.
    shared: Option<Arc<crate::tools::ToolShared>>,
    model: String,
    context_window: u64,
    thinking: String,
    messages: Vec<AgentMessage>,
    // The response streaming in, built from its deltas.
    partial: Option<Vec<Part>>,
    tools: HashMap<String, ToolView>,
    // Which message holds each tool call, so its result redraws only that message.
    call_owner: HashMap<String, usize>,
    state: State,
    // Finished while nobody was looking.
    unread: bool,
    // The context the last response filled, and what the session has cost.
    context_tokens: u64,
    cost: f64,
    draft: Editor,
    // Each message's rows, by the width and flags they were drawn for.
    cache: HashMap<usize, (u16, Flags, Rows)>,
    // Its id, and the agents its tools have running (subagents, `piccosta -p` run by hand).
    id: String,
    agents: Vec<crate::tools::subagent::RunningAgent>,
    // When its current run started, for the working line under the conversation.
    working_since: Option<Instant>,
    // Whether it compacts on its own (/autocompact), and the models it works with (/pair, /trio).
    auto_compaction: bool,
    team: crate::host::Team,
    // Run by another piccosta (it holds the file's lock), and, when shown here, the file's time
    // as last read: such a session is followed from its file, never run twice.
    elsewhere: bool,
    followed: Option<std::time::SystemTime>,
    // Another piccosta has it open and is in the middle of a turn (mid_turn), and the file's time
    // when that was last read.
    working_elsewhere: bool,
    elsewhere_read: Option<std::time::SystemTime>,
    // A manager's session (the assistant, or its project's manager), which leads its group; and,
    // for a new one, that it's to be made its project's manager when it opens (/manager).
    manager: bool,
    make_manager: bool,
    // A running subagent's session: the slot of the session that started it.
    parent: Option<usize>,
    // Its own settings, when it was started with flags of its own (/new <dir> --tools …).
    host: Option<Arc<Host>>,
}

impl Slot {
    fn new(path: Option<PathBuf>, title: String) -> Slot {
        Slot {
            title,
            // Known from the file's name before the file is read, so the agents of a session another
            // piccosta runs are found without opening it.
            id: path.as_deref().map_or_else(String::new, id_of),
            path,
            cwd: PathBuf::new(),
            live: None,
            shared: None,
            model: String::new(),
            context_window: 0,
            thinking: String::new(),
            messages: Vec::new(),
            partial: None,
            tools: HashMap::new(),
            call_owner: HashMap::new(),
            state: State::Idle,
            unread: false,
            context_tokens: 0,
            cost: 0.0,
            draft: Editor::default(),
            cache: HashMap::new(),
            agents: Vec::new(),
            auto_compaction: true,
            working_since: None,
            team: Default::default(),
            elsewhere: false,
            followed: None,
            working_elsewhere: false,
            elsewhere_read: None,
            manager: false,
            make_manager: false,
            parent: None,
            host: None,
        }
    }

    fn busy(&self) -> bool {
        self.state != State::Idle
    }

    // Open, idle, and nothing in it yet: no message, no draft, no file (a new session's file is
    // named from the start but made only with its first message).
    fn empty(&self) -> bool {
        self.live.is_some() && !self.busy() && self.messages.is_empty() && self.draft.text.is_empty() && self.path.as_deref().is_none_or(|path| !path.exists())
    }

    // Deleted, or closed before it was ever saved: kept only so the slot numbers stay put.
    fn gone(&self) -> bool {
        self.live.is_none() && self.path.is_none()
    }

    // A message's tool calls changed: draw it again.
    fn touch(&mut self, call_id: &str) {
        if let Some(index) = self.call_owner.get(call_id) {
            self.cache.remove(index);
        }
    }

    fn remember(&mut self, message: AgentMessage) {
        match &message {
            AgentMessage::Assistant(assistant) => {
                for part in render::blocks(assistant) {
                    if let Part::Call { id, .. } = part {
                        self.call_owner.insert(id, self.messages.len());
                    }
                }
                let usage = &assistant.usage;
                if usage.total_tokens > 0 {
                    self.context_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
                }
                self.cost += usage.cost.total;
            }
            AgentMessage::ToolResult(result) => {
                self.tools.insert(
                    result.tool_call_id.clone(),
                    ToolView {
                        text: crate::message::TextOrBlocks::Blocks(result.content.clone()).text(),
                        diff: result.details.as_ref().and_then(|details| details.get("diff")).and_then(Value::as_str).map(str::to_string),
                        is_error: result.is_error,
                        running: false,
                    },
                );
                let id = result.tool_call_id.clone();
                self.touch(&id);
            }
            _ => {}
        }
        self.messages.push(message);
    }
}

struct Pane {
    slot: usize,
    // Rows scrolled up from the bottom; 0 follows the newest.
    scroll: usize,
}

struct App {
    host: Arc<Host>,
    tex: tex::Tex,
    slots: Vec<Slot>,
    panes: Vec<Pane>,
    focus: usize,
    sidebar: bool,
    flags: Flags,
    frame: usize,
    sender: UnboundedSender<(usize, Update)>,
    updates: UnboundedReceiver<(usize, Update)>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    model: Option<String>,
    persist: bool,
    notice: Option<String>,
    quit_armed: Option<Instant>,
    // The Delete key was pressed once on this session; a second press soon after deletes it.
    delete_armed: Option<(Instant, usize)>,
    // ctrl+x was pressed: the next key finishes a two-key chord.
    chord: bool,
    // How many times scan_agents has run.
    scans: u64,
    // The state last saved (state.rs), as written, so it's written again only when it changes.
    saved: String,
    // /restart: start again on the binary as it is now, once everything here has stopped; and
    // asked for, waiting for the sessions' turns to end.
    restart: bool,
    restart_waiting: bool,
    // A file to open in the user's editor (/scale edit), with the screen handed over meanwhile.
    edit: Option<PathBuf>,
    // When this piccosta started, and each session folder's time when last looked at, for the
    // sessions made since (find_new_sessions).
    started: std::time::SystemTime,
    folder_times: HashMap<PathBuf, std::time::SystemTime>,
    // Since when the focused draft has been /d…, on the way to the easter egg (commands.rs).
    descent_since: Option<Instant>,
    // The session files `/resume all` listed, by their number there.
    found: Vec<PathBuf>,
    // A mouse selection, and the last click (when, where, how many in a row).
    selection: Option<Selection>,
    last_click: Option<(Instant, Position, u8)>,
    quit: bool,
    picker: Option<Picker>,
    // Questions waiting for the picker in use to close.
    asks: std::collections::VecDeque<Picker>,
    // The prompts /rewind offered: each one's parent entry and its text.
    rewind: Vec<(Option<String>, String)>,
    history: Vec<String>,
    // Where things were drawn, for the mouse.
    sidebar_rows: Vec<(u16, usize)>,
    pane_rects: Vec<Rect>,
    sidebar_rect: Rect,
}

// Text being selected with the mouse, in one pane: where the drag started and where it is now, in
// screen cells; by cell, by word (a double click) or by row (a triple click); and whether to copy
// it when it's next drawn (the button came up, or ctrl+c).
struct Selection {
    rect: Rect,
    anchor: Position,
    head: Position,
    clicks: u8,
    copy: bool,
}

// A session's id from its file's name (<time>_<id>.jsonl, as session.rs names them).
fn id_of(path: &Path) -> String {
    path.file_stem().and_then(|stem| stem.to_str()).and_then(|stem| stem.rsplit_once('_')).map_or_else(String::new, |(_, id)| id.to_string())
}

// A list to choose from, filtered by what's typed.
struct Picker {
    title: String,
    items: Vec<String>,
    filter: String,
    selected: usize,
    apply: fn(&mut App, String),
    // What ctrl+d does to the chosen item, after enter confirms it.
    delete: Option<fn(&mut App, String)>,
    confirming: bool,
    // In a picker of directories, the directory typed paths start from: what's typed can then be
    // the choice itself (a path, or anything when no item matches).
    free: Option<PathBuf>,
    // A tool's question: where its answer goes. Closed without one, the sender drops and the
    // call is refused.
    answer: Option<tokio::sync::oneshot::Sender<&'static str>>,
}

impl Picker {
    // The item picked, or, in a free picker, what's typed when nothing matches. A free picker
    // matches on what's typed up to its first space; the rest (flags) goes along with the item.
    fn chosen(&self) -> Option<String> {
        let shown = self.shown();
        let typed = self.filter.trim();
        if self.free.is_some() && shown.is_empty() {
            return (!typed.is_empty()).then(|| typed.to_string());
        }
        let item = shown.get(self.selected.min(shown.len().saturating_sub(1)))?;
        let rest = if self.free.is_some() { typed.split_once(' ').map_or("", |(_, rest)| rest) } else { "" };
        Some(if rest.is_empty() { item.to_string() } else { format!("{item} {rest}") })
    }

    // The items holding every word typed. In a picker of directories only the first word filters
    // (the rest are flags), and a path typed there (~, /, . or a slash in it) that is a directory
    // comes first, as typed, so enter takes it rather than the first project under it (~ or
    // ~/projects).
    fn shown(&self) -> Vec<String> {
        let filter = if self.free.is_some() { self.filter.split(' ').next().unwrap_or("") } else { &self.filter };
        let words: Vec<String> = filter.to_lowercase().split_whitespace().map(str::to_string).collect();
        let mut shown: Vec<String> = self.items.iter().filter(|item| words.iter().all(|word| item.to_lowercase().contains(word))).cloned().collect();
        let path = filter.starts_with(['~', '/', '.']) || filter.contains('/');
        if let Some(base) = &self.free
            && path
            && commands::resolve_directory(filter, base).is_some()
        {
            shown.retain(|item| item != filter);
            shown.insert(0, filter.to_string());
        }
        shown
    }
}

// Run the UI until the user quits. Input: the command line (its model, thinking level, session
// choice, and an opening message). Sessions run as tasks on this thread (a LocalSet): they wait on
// the network and on processes, not on the CPU, and the agent's future isn't Send.
pub async fn run(args: crate::cli::Args) -> Result<std::process::ExitCode, String> {
    tokio::task::LocalSet::new().run_until(run_local(args)).await
}

async fn run_local(mut args: crate::cli::Args) -> Result<std::process::ExitCode, String> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
    let mut hangup = signal(SignalKind::hangup()).map_err(|error| error.to_string())?;
    let host = Arc::new(Host::for_args(&args));
    // After /restart: the reaper and the background jobs the image before handed over.
    let handover = if args.restore { state::take_handover(&host.agent_dir) } else { state::Handover::default() };
    if let Some((fd, pid)) = handover.reaper {
        crate::tools::reaper::adopt(fd, pid);
    }
    if args.restore {
        reap_inherited_children(&handover);
    }
    // --use becomes the model and level the scale picks, before anything opens.
    (args.model, args.thinking) = host.launch(&args)?;
    args.provider = None;
    let mut terminal = setup().map_err(|error| format!("could not set up the terminal: {error}"))?;
    let (sender, updates) = unbounded_channel();
    let mut app = App {
        host: host.clone(),
        tex: tex::Tex::detect(),
        slots: Vec::new(),
        panes: Vec::new(),
        focus: 0,
        sidebar: true,
        flags: Flags::default(),
        frame: 0,
        sender,
        updates,
        tasks: Vec::new(),
        model: crate::cli::model_name(&args),
        persist: !args.no_session,
        notice: None,
        quit_armed: None,
        delete_armed: None,
        chord: false,
        scans: 0,
        saved: String::new(),
        restart: false,
        restart_waiting: false,
        edit: None,
        started: std::time::SystemTime::now(),
        folder_times: HashMap::new(),
        descent_since: None,
        found: Vec::new(),
        selection: None,
        last_click: None,
        quit: false,
        picker: None,
        asks: Default::default(),
        rewind: Vec::new(),
        history: Vec::new(),
        sidebar_rows: Vec::new(),
        pane_rects: Vec::new(),
        sidebar_rect: Rect::default(),
    };
    if !app.tex.images() && std::env::var_os("TMUX").is_some() && std::env::var_os("PICCOSTA_TEX").is_none() {
        app.notice = Some("TeX shows as text: for typeset images inside tmux, add `set -g allow-passthrough on` to tmux.conf".into());
    }
    let opened = app.start(&args);
    if let Err(error) = opened {
        restore();
        host.show_settings_errors();
        return Err(error);
    }
    if !host.settings.load_errors.is_empty() {
        app.report(format!("Invalid settings file:\n{}", host.settings.load_errors.join("\n")));
        app.notice = Some("Invalid settings; details are shown above.".into());
    }
    app.adopt_jobs(handover.sessions);
    let mut input = EventStream::new();
    let mut spinner = tokio::time::interval(FRAME);
    let mut agent_scan = tokio::time::interval(AGENT_SCAN);
    let mut stdout = std::io::stdout();
    let mut exit_code = 0;
    while !app.quit {
        if let Err(error) = terminal.draw(|frame| app.draw(frame)).and_then(|_| app.tex.flush(&mut stdout)) {
            restore();
            app.shutdown("error").await;
            return Err(format!("could not draw: {error}"));
        }
        tokio::select! {
            _ = term.recv() => { exit_code = 143; app.quit = true; },
            _ = hangup.recv() => { exit_code = 129; app.quit = true; },
            event = input.next() => match event {
                Some(Ok(event)) => app.input(event),
                _ => app.quit = true,
            },
            Some((slot, update)) = app.updates.recv() => app.update(slot, update),
            _ = spinner.tick(), if app.slots.iter().any(|slot| slot.busy() || slot.working_elsewhere) => app.frame += 1,
            _ = agent_scan.tick() => app.scan_agents(),
        }
        // Everything else already waiting goes in before the next frame.
        while let Ok((slot, update)) = app.updates.try_recv() {
            app.update(slot, update);
        }
        // A /restart waiting for turns to end goes ahead once none is running.
        if app.restart_waiting && !app.slots.iter().any(Slot::busy) {
            (app.restart, app.quit) = (true, true);
        }
        // A file to edit: the user's editor gets the terminal ($VISUAL, else $EDITOR), and the
        // screen comes back when it exits. Sessions carry on meanwhile; their updates wait.
        if let Some(path) = app.edit.take() {
            match std::env::var_os("VISUAL").or_else(|| std::env::var_os("EDITOR")) {
                None => app.notice = Some(format!("Set $EDITOR to edit it here; the file is {}", path.display())),
                Some(editor) => {
                    restore();
                    let ran = std::process::Command::new(&editor).arg(&path).status();
                    terminal = setup().map_err(|error| format!("could not set up the terminal again: {error}"))?;
                    terminal.clear().map_err(|error| error.to_string())?;
                    app.notice = Some(match ran {
                        Ok(_) => match crate::scale::changes(&app.host.agent_dir) {
                            Ok(_) => format!("{} saved; new sessions and subagents use it", path.display()),
                            Err(error) => format!("Not valid, so it's left out: {error}"),
                        },
                        Err(error) => format!("{}: {error}", editor.to_string_lossy()),
                    });
                }
            }
        }
    }
    app.save_state();
    let _ = app.tex.clear(&mut stdout);
    restore();
    // /restart hands the background jobs over before the sessions end, so their end doesn't kill them.
    let handover = if app.restart { app.hand_over() } else { state::Handover::default() };
    app.shutdown(match exit_code { 143 => "SIGTERM", 129 => "SIGHUP", _ => "quit" }).await;
    // /restart: this process becomes the binary now at its path (a build replaces the file, which
    // the kernel then names "… (deleted)"), on the same command line, bringing everything back.
    if app.restart {
        use std::os::unix::process::CommandExt;
        let exe = std::env::current_exe().map_err(|error| error.to_string())?;
        let exe = PathBuf::from(exe.to_string_lossy().trim_end_matches(" (deleted)"));
        state::save_handover(&host.agent_dir, &handover).map_err(|error| format!("could not hand the background jobs over: {error}"))?;
        let error = std::process::Command::new(&exe).args(std::env::args_os().skip(1).filter(|argument| argument != "--restore")).arg("--restore").exec();
        state::take_handover(&host.agent_dir);
        return Err(format!("could not start {} again: {error}", exe.display()));
    }
    Ok(std::process::ExitCode::from(exit_code))
}

// After /restart, the processes the old image started are still this process's children. The
// jobs it handed over are taken back and reaped by their watchers, and the reaper runs on; any
// other child (a Claude process still closing, say) would stay a zombie when it ends, since
// nothing in this image waits on it, so a thread waits on each. Only children that existed before
// this image started any are listed, so none of its own are taken from tokio.
fn reap_inherited_children(handover: &state::Handover) {
    let kept: Vec<u32> = handover.sessions.iter().flat_map(|(_, jobs)| jobs.iter().map(|job| job.pid)).chain(handover.reaper.map(|(_, pid)| pid)).collect();
    let tasks = std::fs::read_dir("/proc/self/task").into_iter().flatten().flatten();
    let children = tasks.filter_map(|task| std::fs::read_to_string(task.path().join("children")).ok());
    for pid in children.flat_map(|text| text.split_whitespace().filter_map(|pid| pid.parse::<u32>().ok()).collect::<Vec<_>>()) {
        if !kept.contains(&pid) {
            // SAFETY: waitpid on a known child pid with a status pointer it may ignore.
            std::thread::spawn(move || unsafe { libc::waitpid(pid as libc::pid_t, std::ptr::null_mut(), 0) });
        }
    }
}

// How much of a session file's end is read for its last message: room for a long answer.
const TAIL_BYTES: u64 = 1024 * 1024;

// Whether a session another piccosta runs is in the middle of a turn: its last message is the
// user's, a tool's result, a background job's notice, or the model's call for tools. Read from the
// file's end, so a long session costs no more than a short one.
fn mid_turn(path: &Path) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut file) = std::fs::File::open(path) else { return false };
    let length = file.metadata().map_or(0, |metadata| metadata.len());
    let mut tail = Vec::new();
    if file.seek(SeekFrom::Start(length.saturating_sub(TAIL_BYTES))).is_err() || file.read_to_end(&mut tail).is_err() {
        return false;
    }
    let tail = String::from_utf8_lossy(&tail);
    let last = tail.lines().rev().filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok()).find(|entry| entry["type"] == "message" || entry["type"] == "custom_message");
    last.is_some_and(|entry| entry["type"] == "custom_message" || entry["message"]["role"] != "assistant" || entry["message"]["stopReason"] == "toolUse")
}

fn setup() -> std::io::Result<ratatui::DefaultTerminal> {
    use crossterm::event::{EnableBracketedPaste, EnableFocusChange, EnableMouseCapture, KeyboardEnhancementFlags, PushKeyboardEnhancementFlags};
    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(
        std::io::stdout(),
        crossterm::terminal::EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture,
        EnableFocusChange,
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    )?;
    // tmux passes on keys like shift+enter and ctrl+enter only to a pane that asks in xterm's way
    // (modifyOtherKeys), and only with `extended-keys on` in tmux.conf; the kitty request above
    // doesn't reach it.
    if std::env::var_os("TMUX").is_some() {
        print!("{MODIFY_OTHER_KEYS_ON}");
    }
    // A panic puts the terminal back first, except inside a typesetter, where it's caught.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if tex::DRAWING.with(std::cell::Cell::get) {
            return;
        }
        restore();
        previous(info);
    }));
    ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(std::io::stdout()))
}

// xterm's modifyOtherKeys, mode 1 (keys that have no code of their own get one), and off again.
const MODIFY_OTHER_KEYS_ON: &str = "\x1b[>4;1m";
const MODIFY_OTHER_KEYS_OFF: &str = "\x1b[>4m";

fn restore() {
    if std::env::var_os("TMUX").is_some() {
        print!("{MODIFY_OTHER_KEYS_OFF}");
    }
    use crossterm::event::{DisableBracketedPaste, DisableFocusChange, DisableMouseCapture, PopKeyboardEnhancementFlags};
    let _ = crossterm::execute!(
        std::io::stdout(),
        PopKeyboardEnhancementFlags,
        DisableFocusChange,
        DisableMouseCapture,
        DisableBracketedPaste,
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::cursor::Show
    );
    let _ = crossterm::terminal::disable_raw_mode();
}

impl App {
    // The first pane's session (named, this directory's latest, or new), and the sidebar's list
    // of earlier ones.
    fn start(&mut self, args: &crate::cli::Args) -> Result<(), String> {
        if let Some(started) = crate::manager::ensure_running(&self.host.settings) {
            self.notice = Some(started);
        }
        // -c (and /restart) bring back what was open when piccosta last closed, when there's any.
        if !((args.restore || args.continue_recent) && self.restore()) {
            let first = self.slots.len();
            self.slots.push(Slot::new(None, NEW_TITLE.into()));
            self.load(first, self.model.clone(), args.thinking.clone(), Some(args))?;
            self.panes.push(Pane { slot: first, scroll: 0 });
        }
        let first = self.panes[self.focus].slot;
        if let Some(name) = &args.name {
            self.slots[first].title = name.clone();
            self.tell(Command::Name(name.clone()));
        }
        if let Some(team) = &args.team {
            self.set_team(team.clone());
        }
        // This directory's earlier sessions, then the latest few of the projects used most recently,
        // so the sidebar can group them by project.
        let mut folders: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(self.host.agent_dir.join("sessions"))
            .into_iter()
            .flatten()
            .flatten()
            .filter(|folder| folder.path() != self.host.session_dir)
            .filter_map(|folder| Some((folder.metadata().ok()?.modified().ok()?, folder.path())))
            .collect();
        folders.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
        // A project whose directory is gone (a deleted scratch run) has nothing to come back to.
        let others = folders
            .iter()
            .map(|(_, folder)| crate::host::sessions_in(folder, None, RECENT_PER_OTHER_PROJECT))
            .filter(|sessions| sessions.first().is_some_and(|summary| summary.cwd.is_dir()))
            .take(OTHER_PROJECTS)
            .flatten();
        // The manager's session, when there is one, heads the sidebar.
        let manager = crate::manager::session_file(&self.host.agent_dir).map(|file| crate::host::SessionSummary {
            title: crate::manager::assistant_name(&self.host.agent_dir), id: String::new(), cwd: crate::manager::home(&self.host.agent_dir), modified: std::time::SystemTime::now(), closed: false, path: file,
        });
        let others: Vec<crate::host::SessionSummary> = manager.into_iter().chain(self.host.sessions(RECENT_SESSIONS)).chain(others).collect();
        self.list_sessions(others);
        // -r with no session named: the picker, over the new session it opened on. A restart
        // doesn't ask again, or send the opening message again.
        if args.resume.as_deref() == Some("") && !args.restore {
            self.sessions("");
        }
        let opening = args.messages.join("\n");
        if !opening.trim().is_empty() && !args.restore {
            self.send(opening);
        }
        Ok(())
    }

    // Bring back the state saved last (state.rs): each pane's session, opened here (or followed,
    // when another piccosta runs it now), scrolled where it was and holding its draft; then the
    // other sessions that were open, as rows. Output: whether anything came back; when nothing did
    // (no state, or its sessions gone), piccosta starts as it would have.
    fn restore(&mut self) -> bool {
        let Some(state) = state::load(&self.host.agent_dir) else { return false };
        let on_screen = state.panes.len();
        for (index, kept) in state.panes.into_iter().chain(state.open).enumerate() {
            let path = kept.session.filter(|path| path.exists());
            // A new session that was never saved comes back only where it was on screen.
            if path.is_none() && (index >= on_screen || !kept.cwd.is_dir()) {
                continue;
            }
            let slot = self.slots.len();
            self.slots.push(Slot { cwd: kept.cwd, ..Slot::new(path.clone(), NEW_TITLE.into()) });
            self.slots[slot].elsewhere = path.as_deref().is_some_and(crate::session::held_elsewhere);
            if index >= on_screen || self.slots[slot].elsewhere {
                self.follow(slot);
            } else if let Err(error) = self.load(slot, None, None, None) {
                self.notice = Some(error);
                self.slots[slot] = Slot::new(None, NEW_TITLE.into());
                continue;
            }
            self.slots[slot].draft.insert(&kept.draft);
            if index < on_screen {
                self.panes.push(Pane { slot, scroll: kept.scroll });
            }
        }
        if self.panes.is_empty() {
            self.slots.clear();
            return false;
        }
        self.focus = state.focus.min(self.panes.len() - 1);
        self.sidebar = state.sidebar;
        true
    }

    // Save the state as it is now (state.rs), when it has changed since it was last saved. Nothing
    // is kept for a piccosta whose sessions aren't (--no-session).
    fn save_state(&mut self) {
        if !self.persist {
            return;
        }
        let kept = |slot: &Slot, scroll: usize| state::Kept { session: slot.path.clone().filter(|path| path.exists()), cwd: slot.cwd.clone(), scroll, draft: slot.draft.text.clone() };
        let panes = self.panes.iter().map(|pane| kept(&self.slots[pane.slot], pane.scroll)).collect();
        let elsewhere = |index: usize| !self.panes.iter().any(|pane| pane.slot == index);
        let open = (0..self.slots.len())
            .filter(|&index| elsewhere(index) && (self.slots[index].live.is_some() || !self.slots[index].draft.text.is_empty()) && self.slots[index].path.as_deref().is_some_and(Path::exists))
            .map(|index| kept(&self.slots[index], 0))
            .collect();
        let text = serde_json::to_string_pretty(&state::State { panes, focus: self.focus, sidebar: self.sidebar, open }).unwrap_or_default();
        crate::voice::focus(&self.host.agent_dir, self.current().path.as_deref());
        if text != self.saved && state::save(&self.host.agent_dir, &text).is_ok() {
            self.saved = text;
        }
    }

    // /restart: each saved session's running background jobs, and the reaper, for the next image.
    fn hand_over(&self) -> state::Handover {
        let sessions = self.slots.iter().filter_map(|slot| Some((slot.path.clone()?, slot.shared.as_ref()?.jobs.hand_over()))).filter(|(_, jobs)| !jobs.is_empty()).collect();
        state::Handover { reaper: crate::tools::reaper::hand_over(), sessions }
    }

    // After /restart: open each session that had background jobs, where restore didn't, and give
    // it its jobs back, so /ps lists them and their messages reach it.
    fn adopt_jobs(&mut self, sessions: Vec<(PathBuf, Vec<crate::tools::jobs::HandedJob>)>) {
        for (path, jobs) in sessions {
            let slot = match self.slots.iter().position(|slot| slot.path.as_ref() == Some(&path)) {
                Some(slot) if self.slots[slot].live.is_some() => Ok(slot),
                // Shown from its file: a fresh view to open, so its messages don't show twice.
                Some(slot) => {
                    self.slots[slot] = Slot { cwd: self.slots[slot].cwd.clone(), ..Slot::new(Some(path.clone()), self.slots[slot].title.clone()) };
                    self.load(slot, None, None, None).map(|()| slot)
                }
                None => self.open(Some(path.clone()), None, None),
            };
            let shared = slot.map(|slot| self.slots[slot].shared.clone());
            for job in jobs {
                let adopted = match &shared {
                    Ok(Some(shared)) => crate::tools::bash::adopt(shared.clone(), job).map_err(|error| error.to_string()),
                    Ok(None) => Err("its session didn't open".into()),
                    Err(error) => Err(error.clone()),
                };
                if let Err(error) = adopted {
                    self.notice = Some(format!("A background job of {} wasn't taken back: {error}", path.display()));
                }
            }
        }
    }

    // Add sessions to the list that aren't in it yet, each marked when it's a manager.
    fn list_sessions(&mut self, summaries: Vec<crate::host::SessionSummary>) {
        let assistant = crate::manager::session_file(&self.host.agent_dir).and_then(|file| file.canonicalize().ok());
        // Closed sessions stay out of the list; /resume still finds them.
        for summary in summaries.into_iter().filter(|summary| !summary.closed) {
            let path = summary.path.canonicalize().unwrap_or(summary.path);
            if self.slots.iter().any(|slot| slot.path.as_ref() == Some(&path)) {
                continue;
            }
            let project_manager = crate::manager::project_manager(&self.host.agent_dir, &summary.cwd).and_then(|file| file.canonicalize().ok());
            let manager = Some(&path) == assistant.as_ref() || Some(&path) == project_manager.as_ref();
            self.slots.push(Slot { cwd: summary.cwd, manager, ..Slot::new(Some(path), summary.title) });
        }
    }

    // Sessions made since this piccosta started (a manager's, another terminal's) join the list,
    // looked for in the session folders whose time has changed since the last look (a new file
    // changes its folder's time).
    fn find_new_sessions(&mut self) {
        let folders = std::fs::read_dir(self.host.agent_dir.join("sessions")).into_iter().flatten().flatten();
        let mut found = Vec::new();
        for (folder, modified) in folders.filter_map(|folder| Some((folder.path(), folder.metadata().ok()?.modified().ok()?))) {
            if self.folder_times.insert(folder.clone(), modified) != Some(modified) {
                let new = crate::host::sessions_in(&folder, None, RECENT_PER_OTHER_PROJECT).into_iter().filter(|summary| summary.modified >= self.started && summary.cwd.is_dir());
                found.extend(new);
            }
        }
        self.list_sessions(found);
    }

    // Open a session in a new slot: a file, or a new session on `model`.
    fn open(&mut self, path: Option<PathBuf>, model: Option<String>, thinking: Option<String>) -> Result<usize, String> {
        let slot = self.slots.len();
        self.slots.push(Slot::new(path, NEW_TITLE.into()));
        self.load(slot, model, thinking, None).inspect_err(|_| {
            self.slots.pop();
        })?;
        Ok(slot)
    }

    // Start a slot's session: its agent, its task, and what it already holds.
    fn load(&mut self, slot: usize, model: Option<String>, thinking: Option<String>, opening: Option<&crate::cli::Args>) -> Result<(), String> {
        let cwd = self.slots[slot].cwd.clone();
        let session = match opening {
            Some(args) => self.host.opening(args)?,
            // A new session for another directory, kept in that directory's session folder.
            None if self.slots[slot].path.is_none() && !cwd.as_os_str().is_empty() && cwd != self.host.cwd => {
                let folder = crate::session::default_session_dir(&self.host.agent_dir, &cwd);
                crate::session::Session::create(&cwd, self.persist.then_some(folder.as_path()))
            }
            None => self.host.session(self.slots[slot].path.as_deref(), false, self.persist)?,
        };
        // The first session ever in a directory is its project's manager, and so is one /manager
        // asked for. Not a session kept in memory, a subagent's, the assistant's, or one in home;
        // and not the first one on its own in a piccosta an agent's command started or on a
        // scratch agent dir, since managers are the user's (/manager still makes one there).
        let mut session = session;
        let dir = PathBuf::from(&session.header().cwd);
        let agent_dir = &self.host.agent_dir;
        let first = session.entries().is_empty()
            && crate::manager::project_manager(agent_dir, &dir).is_none()
            && crate::host::sessions_in(&crate::session::default_session_dir(agent_dir, &dir), Some(&dir), 1).is_empty();
        // Home isn't a project: a session started there (outside every project) is a plain one.
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let eligible = self.persist && std::env::var_os(crate::tools::subagent::CHILD_ENV).is_none() && dir != crate::manager::home(agent_dir) && Some(&dir) != home.as_ref();
        let first = first && !crate::manager::started_by_agent() && !crate::manager::own_agent_dir();
        if eligible && (first || self.slots[slot].make_manager) && crate::manager::role(&session).is_none() {
            crate::manager::make(agent_dir, &mut session, crate::manager::Role::Project).map_err(|error| format!("could not make the project manager: {error}"))?;
        }
        let sender = self.sender.clone();
        let emit: crate::agent::EventSink = Box::new(move |event: &Event| {
            let _ = sender.send((slot, Update::Event(Box::new(event.clone()))));
        });
        let host = self.slots[slot].host.clone().unwrap_or_else(|| self.host.clone());
        let (mut agent, shared) = host.open(session, model.as_deref(), thinking.as_deref(), emit, false)?;
        // The terminal asks nothing but a tool's own question; "for this session" holds for that
        // question until the session closes.
        let (asks, allowed) = (self.sender.clone(), std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashSet::new())));
        agent.set_gate(std::rc::Rc::new(move |_: String, name: String, arguments: serde_json::Map<String, Value>, question: Option<&'static str>| {
            let (asks, allowed) = (asks.clone(), allowed.clone());
            Box::pin(async move {
                let Some(question) = question.filter(|question| !allowed.borrow().contains(question)) else { return Ok(()) };
                let (reply, answer) = tokio::sync::oneshot::channel();
                let action = arguments.get("action").and_then(Value::as_str).map_or(name.clone(), |action| format!("{name} {action}"));
                let _ = asks.send((slot, Update::Ask(question, action, reply)));
                match answer.await {
                    Ok(answer) if answer == ANSWERS[0] => Ok(()),
                    Ok(answer) if answer == ANSWERS[1] => {
                        allowed.borrow_mut().insert(question);
                        Ok(())
                    }
                    _ => Err("The user refused.".to_string()),
                }
            })
        }));
        let view = &mut self.slots[slot];
        view.title = crate::manager::title(&self.host.agent_dir, &agent.session);
        view.manager = crate::manager::role(&agent.session).is_some();
        view.path = agent.session.file().map(|path| path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
        view.cwd = PathBuf::from(&agent.session.header().cwd);
        view.model = format!("{}/{}", agent.model().provider, agent.model().id);
        view.context_window = agent.model().context_window;
        view.thinking = agent.thinking_level().to_string();
        // The prompts it holds join the history, older than what's typed here, and its context is
        // the projected one (after a compaction the last response's usage measured the old one).
        let mut prompts = Vec::new();
        for message in agent.session.context().messages {
            if let AgentMessage::User(user) = &message {
                prompts.push(user.content.text());
            }
            view.remember(message);
        }
        view.context_tokens = crate::compaction::estimate_projected_context_tokens(&agent.session.projection(), &agent.session.path_to_leaf()).tokens;
        view.auto_compaction = agent.auto_compaction();
        view.team = crate::host::team(&agent.session);
        let (commands, receiver) = unbounded_channel();
        view.id = agent.session.id().to_string();
        // Everything queued goes in together at the next step, as Claude Code sends it.
        agent.handle().set_steering_mode("all");
        view.live = Some((agent.handle(), commands));
        view.shared = Some(shared.clone());
        let sender = self.sender.clone();
        let done = move |done: Done| {
            let _ = sender.send((slot, Update::Done(Box::new(done))));
        };
        self.tasks.push(host.spawn(agent, shared, true, receiver, done));
        prompts.retain(|prompt| !self.history.contains(prompt));
        self.history.splice(0..0, prompts);
        Ok(())
    }

    fn current(&self) -> &Slot {
        &self.slots[self.panes[self.focus].slot]
    }

    fn focused(&mut self) -> &mut Slot {
        let slot = self.panes[self.focus].slot;
        &mut self.slots[slot]
    }

    // Show a slot in the focused pane. An earlier session is read from its file and nothing starts:
    // it opens (gets its agent) only when something is typed into it, so stepping through the list
    // never adds an open session. A new session left empty behind it goes away.
    fn show(&mut self, slot: usize) {
        let leaving = self.panes[self.focus].slot;
        if self.slots[slot].live.is_none() {
            self.slots[slot].elsewhere = self.slots[slot].path.as_deref().is_some_and(crate::session::held_elsewhere);
            self.follow(slot);
        }
        self.slots[slot].unread = false;
        self.panes[self.focus] = Pane { slot, scroll: 0 };
        self.put_away(leaving);
    }

    // A session no pane shows any more: one that's empty goes away, and one that's idle with nothing
    // in the background goes back to being an earlier session (its agent ends, its file lock goes,
    // and it starts again when typed into). One still working stays open until it's done.
    fn put_away(&mut self, slot: usize) {
        let view = &self.slots[slot];
        if self.panes.iter().any(|pane| pane.slot == slot) || view.live.is_none() || view.busy() {
            return;
        }
        let background = view.shared.as_ref().is_some_and(|shared| shared.jobs.running_count() > 0) || !view.agents.is_empty();
        if view.empty() {
            self.slots[slot] = Slot::new(None, NEW_TITLE.into());
        } else if !background && view.draft.text.is_empty() {
            self.slots[slot] = Slot { cwd: view.cwd.clone(), ..Slot::new(view.path.clone(), view.title.clone()) };
        }
    }

    // Text pasted into the picker's filter or the editor, with line ends and tabs evened out.
    fn paste_text(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n").replace('\t', TAB_SPACES);
        if let Some(picker) = &mut self.picker {
            picker.filter.push_str(&text);
        } else {
            self.focused().draft.insert(&text);
        }
    }

    // Paste from the clipboard (or the primary selection), read through the desktop's tool.
    fn paste(&mut self, primary: bool) {
        match clip::paste(primary) {
            Some(text) => self.paste_text(&text),
            None => self.notice = Some("Nothing to paste: no clipboard tool (wl-paste, xclip, xsel) answered; the terminal's own paste key still works".into()),
        }
    }

    // Make sure the focused pane's session is open here, opening an earlier one shown from its
    // file. Output: false, with a notice, when it can't be (another piccosta runs it).
    fn ensure_open(&mut self) -> bool {
        if self.current().live.is_some() {
            return true;
        }
        if self.current().elsewhere {
            self.notice = Some("Another piccosta is running this session; this pane follows it, and nothing typed here goes to it.".into());
            return false;
        }
        let index = self.panes[self.focus].slot;
        let view = &self.slots[index];
        self.slots[index] = Slot { cwd: view.cwd.clone(), ..Slot::new(view.path.clone(), view.title.clone()) };
        match self.load(index, None, None, None) {
            Ok(()) => true,
            Err(error) => {
                self.notice = Some(error);
                false
            }
        }
    }

    // A new session on the focused one's model and level, or the focused one itself when it's an
    // empty one nobody has typed into yet, so ^n and /clear don't pile up empty rows.
    fn new_session(&mut self) -> Option<usize> {
        self.new_session_in(None, None)
    }

    // The same in another directory (None: the focused session's), with the command line's flags
    // when given (/new <dir> --model … --pair …): the model and level, its team and name, and
    // its own tools and prompt settings. Without flags it takes the focused session's model and
    // level. The session runs in that directory: its tools, its AGENTS.md, its session folder.
    fn new_session_in(&mut self, dir: Option<PathBuf>, args: Option<crate::cli::Args>) -> Option<usize> {
        let current = self.current();
        let dir = dir.unwrap_or_else(|| if current.cwd.as_os_str().is_empty() { self.host.cwd.clone() } else { current.cwd.clone() });
        if current.empty() && current.cwd == dir && args.is_none() {
            return Some(self.panes[self.focus].slot);
        }
        let flags = args.as_ref();
        let model = flags.and_then(crate::cli::model_name).or_else(|| Some(current.model.clone()).filter(|model| !model.is_empty())).or(self.model.clone());
        let thinking = flags.and_then(|args| args.thinking.clone()).or_else(|| Some(current.thinking.clone()).filter(|level| !level.is_empty()));
        let host = flags.map(|args| Arc::new(Host::for_args_at(args, dir.clone())));
        let slot = self.slots.len();
        self.slots.push(Slot { cwd: dir, host, ..Slot::new(None, NEW_TITLE.into()) });
        if let Err(error) = self.load(slot, model, thinking, None) {
            self.slots.pop();
            self.notice = Some(error);
            return None;
        }
        // A level the model doesn't take runs as its nearest, and says so.
        if let Some(asked) = flags.and_then(|args| args.thinking.as_ref()).filter(|asked| **asked != self.slots[slot].thinking) {
            self.notice = Some(format!("{} doesn't take {asked}; it runs at {}", self.slots[slot].model, self.slots[slot].thinking));
        }
        // Its name and team go to it as the commands /rename and /pair send.
        let commands = self.slots[slot].live.as_ref().map(|(_, commands)| commands.clone());
        if let (Some(args), Some(commands)) = (flags, commands) {
            if let Some(name) = &args.name {
                self.slots[slot].title = name.clone();
                let _ = commands.send(Command::Name(name.clone()));
            }
            if let Some(team) = &args.team {
                self.slots[slot].team = team.clone();
                let _ = commands.send(Command::Team(team.clone()));
            }
        }
        Some(slot)
    }

    // The editor's text: a command, a prompt, or steering for a run in progress.
    fn submit(&mut self) {
        let text = self.focused().draft.take();
        if text.trim().is_empty() {
            return;
        }
        if self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        // A command is its name right after the slash; "/ something" is a message like any other.
        if let Some(command) = text.strip_prefix('/')
            && command.starts_with(|character: char| !character.is_whitespace())
            && self.command(command.trim())
        {
            return;
        }
        self.send(text);
    }

    fn send(&mut self, text: String) {
        let pane = self.focus;
        self.panes[pane].scroll = 0;
        // A message for a session another piccosta runs (the manager, say) goes to its inbox, which
        // that piccosta takes in; typing into one shown from its file opens it here.
        if self.current().live.is_none() && self.current().elsewhere {
            let Some(path) = self.current().path.clone() else { return };
            self.notice = Some(match crate::manager::deliver(&path, &text, false) {
                Ok(()) => "Sent: the piccosta running this session takes it in after its current step.".into(),
                Err(error) => format!("Couldn't leave the message: {error}"),
            });
            return;
        }
        if !self.ensure_open() {
            self.focused().draft.insert(&text);
            return;
        }
        let slot = self.focused();
        let Some((handle, commands)) = &slot.live else { return };
        // While the model works, a message waits for the step it's on; commands that step is
        // running move to the background (they carry on and report when they end), so the message
        // isn't left waiting on them.
        if slot.busy() {
            handle.steer(text, false);
            if let Some(shared) = &slot.shared {
                shared.jobs.detach_all();
            }
            return;
        }
        let _ = commands.send(Command::Prompt(text.clone(), Vec::new(), false));
        slot.state = State::Running;
        if slot.title == NEW_TITLE {
            slot.title = render::fit(text.lines().next().unwrap_or(""), 80);
        }
    }

    // Work with other models in the focused session (none: alone again), and say how.
    fn set_team(&mut self, team: crate::host::Team) {
        let names = team.models.join(" and ");
        self.notice = Some(match (team.models.is_empty(), team.pair) {
            (true, _) => "Working alone again.".into(),
            (false, true) => format!("A pair with {names}: after you ask, the two answer each other in turn until one says it's done, then it's your turn. /pair off ends it."),
            (false, false) => format!("A trio with {names}: each answer is checked by {names}, then it's your turn. /trio off ends it."),
        });
        self.focused().team = team.clone();
        self.tell(Command::Team(team));
    }

    fn tell(&mut self, command: Command) {
        if !self.ensure_open() {
            return;
        }
        if let Some((_, commands)) = &self.focused().live {
            let _ = commands.send(command);
        }
    }

    // Choose from a list; an argument that matches one item picks it at once.
    fn pick(&mut self, title: &'static str, items: Vec<String>, filter: &str, apply: fn(&mut App, String)) {
        let picker = Picker { title: title.to_string(), items, filter: filter.to_string(), selected: 0, apply, delete: None, confirming: false, free: None, answer: None };
        let shown = picker.shown();
        if !filter.is_empty() && (shown.len() == 1 || shown.iter().any(|item| item.as_str() == filter)) {
            let choice = shown.iter().find(|item| item.as_str() == filter).unwrap_or(&shown[0]).to_string();
            apply(self, choice);
            return;
        }
        self.picker = Some(picker);
    }

    // Put an open picker's selection on `item` when it's listed, and leave it at the top when not.
    fn select(&mut self, item: &str) {
        if let Some(picker) = &mut self.picker {
            picker.selected = picker.shown().iter().position(|shown| *shown == item).unwrap_or(0);
        }
    }

    fn split(&mut self) {
        if self.panes.len() >= MAX_PANES {
            return;
        }
        if let Some(slot) = self.new_session() {
            self.panes.push(Pane { slot, scroll: 0 });
            self.focus = self.panes.len() - 1;
        }
    }

    fn close_pane(&mut self) {
        if self.panes.len() > 1 {
            let slot = self.panes.remove(self.focus).slot;
            self.focus = self.focus.min(self.panes.len() - 1);
            self.put_away(slot);
        }
    }

    // Stop the focused session's run; what was queued goes out at once as the next prompt, and the
    // draft stays in the editor (Claude Code's esc).
    // Esc: the commands the turn is running move to the background, where they carry on and report
    // when they end, and the model stops.
    fn interrupt(&mut self) {
        let slot = self.focused();
        let (Some((handle, commands)), Some(shared)) = (&slot.live, &slot.shared) else { return };
        handle.interrupt(&shared.jobs);
        let (steer, follow_up) = handle.clear();
        let queued: Vec<String> = steer.into_iter().chain(follow_up).collect();
        if !queued.is_empty() {
            let _ = commands.send(Command::Prompt(queued.join("\n\n"), Vec::new(), false));
        }
    }

    // Claude Code's "send now": the draft joins the queue and the model reads it all now. Commands
    // the turn is running move to the background, so the turn goes on to the queue at once; when
    // nothing can move (the model is writing), the run stops and the queue is the next prompt. With
    // nothing running it's enter.
    fn send_now(&mut self) {
        if !self.current().busy() {
            self.submit();
            return;
        }
        let text = self.focused().draft.take();
        if !text.trim().is_empty() {
            self.history.push(text.clone());
            if let Some((handle, _)) = &self.current().live {
                handle.steer(text, false);
            }
        }
        let detached = self.current().shared.as_ref().is_some_and(|shared| shared.jobs.detach_all());
        if !detached {
            self.interrupt();
        }
    }

    // Take what's queued back into the editor, one per line ahead of the draft. Output: whether
    // there was anything.
    fn take_back(&mut self) -> bool {
        let slot = self.focused();
        let Some((handle, _)) = &slot.live else { return false };
        let (steer, follow_up) = handle.clear();
        let queued: Vec<String> = steer.into_iter().chain(follow_up).collect();
        if queued.is_empty() {
            return false;
        }
        let text = format!("{}{}", queued.join("\n"), if slot.draft.text.is_empty() { String::new() } else { format!("\n{}", slot.draft.text) });
        slot.draft = Editor::with(text);
        true
    }

    // Look again at what runs here and elsewhere: messages left in open sessions' inboxes, which
    // listed sessions another piccosta has open, what a followed one's file holds now, and the
    // agents of every open or followed session (the assistant, a manager in another terminal, their
    // lanes), an agent's own agents under its row. Kept cheap for an idle UI: /proc is walked once a
    // scan, every scan while a session is busy or has background jobs or agents, else every
    // LOCK_SCAN_EVERY scans, which still finds what a command left running with `&` and what other
    // piccostas start. Other piccostas' locks are looked at as often, a shown session's every scan.
    // Then the state is saved.
    fn scan_agents(&mut self) {
        self.scans += 1;
        let locks_due = self.scans.is_multiple_of(LOCK_SCAN_EVERY);
        // A message left for the focused session while it's shown from its file (push-to-talk sends
        // there) opens it, as typing into it would.
        if self.current().live.is_none() && !self.current().elsewhere && self.current().path.as_deref().is_some_and(crate::manager::waiting) {
            self.ensure_open();
        }
        for index in 0..self.slots.len() {
            let slot = &mut self.slots[index];
            // Messages other piccostas left for a session open here: queued while it runs, else
            // the next prompt.
            if let (Some((handle, commands)), Some(path)) = (&slot.live, &slot.path) {
                for (text, spoken) in crate::manager::take_inbox(path) {
                    if slot.state == State::Idle {
                        let _ = commands.send(Command::Prompt(text, Vec::new(), spoken));
                        slot.state = State::Running;
                    } else {
                        handle.steer(text, spoken);
                    }
                }
            }
            let shown = self.panes.iter().any(|pane| pane.slot == index);
            if slot.live.is_none() && (locks_due || shown) {
                slot.elsewhere = slot.path.as_deref().is_some_and(crate::session::held_elsewhere);
                let modified = slot.path.as_deref().and_then(|path| path.metadata().ok()?.modified().ok());
                if !slot.elsewhere {
                    slot.working_elsewhere = false;
                } else if modified != slot.elsewhere_read {
                    slot.working_elsewhere = slot.path.as_deref().is_some_and(mid_turn);
                    slot.elsewhere_read = modified;
                }
            }
            if slot.followed.is_some() {
                self.follow(index);
            }
        }
        let watched = |slot: &Slot| (slot.live.is_some() || slot.elsewhere) && !slot.id.is_empty();
        let active = |slot: &Slot| slot.busy() || !slot.agents.is_empty() || slot.shared.as_ref().is_some_and(|shared| shared.jobs.running_count() > 0);
        if locks_due || self.slots.iter().any(|slot| watched(slot) && active(slot)) {
            let ids: Vec<&str> = self.slots.iter().filter(|slot| watched(slot)).map(|slot| slot.id.as_str()).collect();
            let mut found = crate::tools::subagent::running_under(&ids);
            for slot in self.slots.iter_mut() {
                slot.agents = if watched(slot) { found.remove(&slot.id).unwrap_or_default() } else { Vec::new() };
            }
        }
        // The assistant's row, once its session exists (it starts a moment after piccosta does).
        if let Some(file) = crate::manager::session_file(&self.host.agent_dir).and_then(|file| file.canonicalize().ok())
            && !self.slots.iter().any(|slot| slot.path.as_ref() == Some(&file))
        {
            self.slots.push(Slot { cwd: crate::manager::home(&self.host.agent_dir), manager: true, ..Slot::new(Some(file), crate::manager::assistant_name(&self.host.agent_dir)) });
        }
        if locks_due {
            self.find_new_sessions();
        }
        // Each running subagent with a session gets a row of its own under its parent, so it can
        // be walked to and opened (followed live); a finished one's row goes unless it's on screen.
        let mut children = Vec::new();
        for (index, slot) in self.slots.iter().enumerate() {
            for agent in &slot.agents {
                let Some(file) = &agent.session else { continue };
                if !self.slots.iter().chain(&children).any(|other: &Slot| other.path.as_ref() == Some(file)) {
                    let model = agent.model.as_deref().map_or("", |model| model.rsplit('/').next().unwrap_or(model));
                    children.push(Slot { cwd: slot.cwd.clone(), parent: Some(index), elsewhere: true, ..Slot::new(Some(file.clone()), format!("{} · {model}", agent.job.as_deref().unwrap_or("agent"))) });
                }
            }
        }
        self.slots.extend(children);
        for index in 0..self.slots.len() {
            let finished = self.slots[index].parent.is_some() && self.slots[index].path.as_deref().is_some_and(|path| !crate::session::held_elsewhere(path));
            if finished && !self.panes.iter().any(|pane| pane.slot == index) {
                self.slots[index].path = None;
            }
        }
        self.save_state();
    }

    // Show a session that isn't open here, from its file: an earlier one, or one another piccosta
    // is running. The file is read again only when it has changed since.
    fn follow(&mut self, slot: usize) {
        let view = &mut self.slots[slot];
        let Some(path) = view.path.clone() else { return };
        let modified = path.metadata().and_then(|metadata| metadata.modified()).ok();
        if view.followed.is_some() && view.followed == modified {
            return;
        }
        let Ok(session) = crate::session::Session::open(&path, &view.cwd) else { return };
        let context = session.context();
        let model = context.model.as_ref().map_or(String::new(), |(provider, id)| format!("{provider}/{id}"));
        let window = self.host.registry().ok().and_then(|registry| registry.resolve(&model).ok().map(|found| found.context_window));
        let view = &mut self.slots[slot];
        view.context_window = window.unwrap_or(0);
        view.messages.clear();
        view.tools.clear();
        view.call_owner.clear();
        view.cache.clear();
        if view.parent.is_none() {
            view.title = crate::manager::title(&self.host.agent_dir, &session);
        }
        view.id = session.id().to_string();
        view.cwd = PathBuf::from(&session.header().cwd);
        view.model = model;
        view.thinking = context.thinking_level.clone();
        view.manager = crate::manager::role(&session).is_some();
        view.team = crate::host::team(&session);
        view.context_tokens = crate::compaction::estimate_projected_context_tokens(&session.projection(), &session.path_to_leaf()).tokens;
        for message in context.messages {
            view.remember(message);
        }
        view.followed = modified;
    }

    // The sessions in the sidebar's order: open ones, then earlier ones, the gone left out.
    // Grouped by project (the session's directory): this one's first, then the others in the order
    // they first come up; open sessions ahead of earlier ones in each, earlier ones newest first.
    fn listed(&self) -> Vec<usize> {
        let mut listed: Vec<usize> = (0..self.slots.len()).filter(|&index| !self.slots[index].gone()).collect();
        let manager = crate::manager::home(&self.host.agent_dir);
        let mut projects = vec![manager.as_path(), self.host.cwd.as_path()];
        for &index in &listed {
            if !projects.contains(&self.slots[index].cwd.as_path()) {
                projects.push(&self.slots[index].cwd);
            }
        }
        let project = |index: usize| projects.iter().position(|project| *project == self.slots[index].cwd).unwrap_or(projects.len());
        listed.sort_by_key(|&index| (project(index), !self.slots[index].manager, self.slots[index].live.is_none()));
        // A subagent's row comes right after its parent's (and after its siblings').
        let (children, mut order): (Vec<usize>, Vec<usize>) = listed.into_iter().partition(|&index| self.slots[index].parent.is_some());
        for child in children {
            let parent = self.slots[child].parent.and_then(|parent| order.iter().position(|&index| index == parent));
            let mut at = parent.map_or(order.len(), |parent| parent + 1);
            while order.get(at).is_some_and(|&index| self.slots[index].parent.is_some()) {
                at += 1;
            }
            order.insert(at, child);
        }
        order
    }

    // Take a session out of the open ones. Its task ends when its command channel closes; its pane
    // closes, or shows another open session (a new one when none is left). Output: false, with a
    // notice, when it's running.
    fn close(&mut self, slot: usize) -> bool {
        if self.slots[slot].busy() {
            self.notice = Some(format!("“{}” is running; esc stops it, then it can go.", render::fit(&self.slots[slot].title, 60)));
            return false;
        }
        let next = self.listed().into_iter().find(|&index| index != slot && self.slots[index].live.is_some()).or_else(|| self.new_session());
        // A fresh view, so opening it again doesn't show its messages twice; never saved, it's gone.
        let path = self.slots[slot].path.clone().filter(|path| path.exists());
        self.slots[slot] = Slot::new(path, self.slots[slot].title.clone());
        let showing: Vec<usize> = (0..self.panes.len()).rev().filter(|&pane| self.panes[pane].slot == slot).collect();
        for pane in showing {
            if self.panes.len() > 1 {
                self.panes.remove(pane);
            } else {
                self.panes[pane] = Pane { slot: next.unwrap_or(slot), scroll: 0 };
            }
        }
        self.focus = self.focus.min(self.panes.len() - 1);
        true
    }

    fn update(&mut self, slot: usize, update: Update) {
        let visible = self.panes.iter().any(|pane| pane.slot == slot);
        let view = &mut self.slots[slot];
        match update {
            // Shown at once unless a list is open, which it waits for. Its title says which session
            // asks and for what, since it may not be the one on screen.
            Update::Ask(question, action, reply) => {
                let items = ANSWERS.map(str::to_string).to_vec();
                let title = format!("{} · {action} · {question}", render::fit(&view.title, ASKER_CHARS));
                let picker = Picker { title, items, filter: String::new(), selected: 0, apply: |_, _| {}, delete: None, confirming: false, free: None, answer: Some(reply) };
                match self.picker {
                    None => self.picker = Some(picker),
                    Some(_) => self.asks.push_back(picker),
                }
            }
            Update::Done(done) => {
                view.model = format!("{}/{}", done.model.provider, done.model.id);
                view.context_window = done.model.context_window;
                view.thinking = done.thinking;
                // A command finished, so the session's task is idle, whatever the last event said.
                view.state = State::Idle;
                view.working_since = None;
                if let Err(error) = done.result {
                    self.notice = Some(error);
                }
            }
            Update::Event(event) => match *event {
                Event::AgentStart | Event::AutoRetryEnd { .. } => {
                    view.state = State::Running;
                    view.working_since.get_or_insert_with(Instant::now);
                }
                Event::AgentSettled { .. } => {
                    view.state = State::Idle;
                    view.working_since = None;
                    view.partial = None;
                    view.unread = !visible;
                }
                Event::MessageStart { message: AgentMessage::Assistant(_) } => view.partial = Some(Vec::new()),
                Event::MessageUpdate { assistant_message_event, .. } => {
                    if let Some(parts) = &mut view.partial {
                        apply_delta(parts, &assistant_message_event);
                    }
                }
                Event::MessageEnd { message } => {
                    if matches!(message, AgentMessage::Assistant(_)) {
                        view.partial = None;
                    }
                    view.remember(message);
                }
                Event::ToolExecutionStart { tool_call_id, .. } => {
                    view.tools.insert(tool_call_id.clone(), ToolView { running: true, ..Default::default() });
                    view.touch(&tool_call_id);
                }
                Event::ToolExecutionUpdate { tool_call_id, partial_result, .. } => {
                    view.tools.insert(tool_call_id.clone(), ToolView { text: result_text(&partial_result), running: true, ..Default::default() });
                    view.touch(&tool_call_id);
                }
                Event::ToolExecutionEnd { tool_call_id, result, is_error, .. } => {
                    let diff = result.pointer("/details/diff").and_then(Value::as_str).map(str::to_string);
                    view.tools.insert(tool_call_id.clone(), ToolView { text: result_text(&result), diff, is_error, running: false });
                    view.touch(&tool_call_id);
                }
                Event::AutoRetryStart { attempt, max_attempts, delay_ms, error_message } => {
                    view.state = State::Retrying(format!("retry {attempt}/{max_attempts} in {}s: {}", delay_ms / 1000, render::fit(&error_message, 60)));
                }
                Event::CompactionStart { .. } => view.state = State::Compacting,
                // A compaction says what it did, and the context is the summary's size from now on.
                Event::CompactionEnd { error_message, result, .. } => {
                    view.state = State::Running;
                    if let Some(result) = result {
                        let before = result.get("tokensBefore").and_then(Value::as_u64).unwrap_or(0);
                        let after = result.get("estimatedTokensAfter").and_then(Value::as_u64).unwrap_or(0);
                        view.context_tokens = after;
                        let text = format!("Compacted: {} → about {} tokens of context", crate::cache::tokens(before), crate::cache::tokens(after));
                        view.messages.push(commands::report_message(text));
                    }
                    if let Some(error) = error_message {
                        self.notice = Some(error);
                    }
                }
                Event::EntryAppended { entry: crate::session::SessionEntry::Custom { custom_type, data: Some(miss), .. } } if custom_type == crate::cache::MISS_TYPE => {
                    view.messages.push(commands::report_message(crate::cache::line(&miss)));
                }
                _ => {}
            },
        }
    }

    fn input(&mut self, input: Input) {
        match input {
            Input::Key(key) if key.kind != KeyEventKind::Release => self.key(key),
            Input::Paste(text) => self.paste_text(&text),
            Input::Mouse(mouse) => {
                let at = Position::new(mouse.column, mouse.row);
                let pane = self.pane_rects.iter().position(|rect| rect.contains(at));
                match mouse.kind {
                    MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                        if let Some(pane) = pane {
                            let scroll = &mut self.panes[pane].scroll;
                            *scroll = if mouse.kind == MouseEventKind::ScrollUp { *scroll + WHEEL_ROWS } else { scroll.saturating_sub(WHEEL_ROWS) };
                        }
                    }
                    // A press in a pane starts a selection there; a second or third quick press on the
                    // same cell selects its word or its row.
                    MouseEventKind::Down(MouseButton::Left) => {
                        if let Some(pane) = pane {
                            self.focus = pane;
                            let clicks = match self.last_click {
                                Some((when, place, count)) if place == at && when.elapsed() < MULTI_CLICK => count % 3 + 1,
                                _ => 1,
                            };
                            self.last_click = Some((Instant::now(), at, clicks));
                            self.selection = Some(Selection { rect: pane_inner(pane, self.pane_rects[pane]), anchor: at, head: at, clicks, copy: false });
                        } else if self.sidebar_rect.contains(at)
                            && let Some(&(_, slot)) = self.sidebar_rows.iter().rev().find(|(row, _)| *row <= mouse.row)
                        {
                            self.show(slot);
                        }
                    }
                    MouseEventKind::Drag(MouseButton::Left) => {
                        if let Some(selection) = &mut self.selection {
                            selection.head = at;
                        }
                    }
                    // Letting go copies what's selected, as Claude Code does; a plain click selects nothing.
                    MouseEventKind::Up(MouseButton::Left) => {
                        if let Some(selection) = &mut self.selection {
                            selection.head = at;
                            selection.copy = selection.anchor != selection.head || selection.clicks > 1;
                            if !selection.copy {
                                self.selection = None;
                            }
                        }
                    }
                    // A right click pastes the clipboard, a middle click the primary selection.
                    MouseEventKind::Down(MouseButton::Right) => self.paste(false),
                    MouseEventKind::Down(MouseButton::Middle) => self.paste(true),
                    _ => {}
                }
            }
            // tmux drops images sent to a hidden pane; send them all again on coming back.
            Input::FocusGained => {
                let _ = self.tex.clear(&mut std::io::stdout());
                self.slots.iter_mut().for_each(|slot| slot.cache.clear());
            }
            _ => {}
        }
    }

    fn key(&mut self, key: KeyEvent) {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        if self.picker.is_some() {
            self.picker_key(key);
            self.picker = self.picker.take().or_else(|| self.asks.pop_front());
            return;
        }
        if !(control && matches!(key.code, KeyCode::Char('c' | 'd'))) {
            self.quit_armed = None;
        }
        let delete_armed = self.delete_armed.take();
        // A notice lasts until the next key.
        self.notice = None;
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        // A mouse selection: ctrl+c copies it, shift+arrows (and shift+home/end) move its end, the
        // page keys scroll under it; any other key lets it go and does what it does.
        if let Some(selection) = &mut self.selection {
            let head = &mut selection.head;
            match key.code {
                KeyCode::Char('c') if control => return selection.copy = true,
                KeyCode::Left if shift => return head.x = head.x.saturating_sub(1),
                KeyCode::Right if shift => return head.x += 1,
                KeyCode::Up if shift => return head.y = head.y.saturating_sub(1),
                KeyCode::Down if shift => return head.y += 1,
                KeyCode::Home if shift => return head.x = selection.rect.x,
                KeyCode::End if shift => return head.x = selection.rect.right(),
                KeyCode::PageUp | KeyCode::PageDown => {}
                _ => self.selection = None,
            }
        }
        // The second key of a ctrl+x chord (Claude Code's and readline's): ctrl+s sends now.
        if std::mem::take(&mut self.chord) {
            if control && key.code == KeyCode::Char('s') {
                self.send_now();
            }
            return;
        }
        match key.code {
            // Text selected in the editor (shift+arrows): ctrl+c copies it.
            KeyCode::Char('c') if control && self.current().draft.selection().is_some() => {
                let draft = &self.current().draft;
                let text = draft.selection().map(|range| draft.text[range].to_string()).unwrap_or_default();
                let via = clip::copy(&text);
                self.notice = Some(format!("Copied {} characters with {via}", text.chars().count()));
            }
            KeyCode::Char('v') if control => self.paste(false),
            // Ctrl+Shift+Delete clears what's typed; ↑ brings it back from the history.
            KeyCode::Delete if control && shift => {
                let text = self.focused().draft.take();
                if !text.trim().is_empty() && self.history.last() != Some(&text) {
                    self.history.push(text);
                }
            }
            // The Delete key with nothing typed deletes the session shown here: the first press says
            // which, a second soon after moves it to the trash and shows the next one in the list.
            KeyCode::Delete if self.current().draft.text.is_empty() => {
                let slot = self.panes[self.focus].slot;
                if delete_armed.is_some_and(|(when, armed)| armed == slot && when.elapsed() < QUIT_CONFIRM) {
                    self.delete_shown();
                } else {
                    self.delete_armed = Some((Instant::now(), slot));
                    self.notice = Some(format!("Delete again moves “{}” to the trash", render::fit(&self.current().title, 60)));
                }
            }
            // The session list, also on ctrl+m where the terminal tells it from enter.
            KeyCode::Char('m') if control => self.sessions(""),
            // Claude Code's ctrl+c: stop the run here; else clear the draft; else a second press
            // within QUIT_CONFIRM quits. ctrl+d on an empty draft quits the same two-press way.
            KeyCode::Char('c') if control && self.current().busy() => self.interrupt(),
            KeyCode::Char('c') if control && !self.current().draft.text.is_empty() => {
                self.focused().draft.take();
            }
            KeyCode::Char('c' | 'd') if control && (key.code == KeyCode::Char('c') || self.current().draft.text.is_empty()) => {
                if self.quit_armed.is_some_and(|armed| armed.elapsed() < QUIT_CONFIRM) {
                    self.quit = true;
                } else {
                    self.quit_armed = Some(Instant::now());
                    let running = if self.slots.iter().any(Slot::busy) { "sessions are running; " } else { "" };
                    let pressed = if key.code == KeyCode::Char('d') { "ctrl+d" } else { "ctrl+c" };
                    self.notice = Some(format!("{running}{pressed} again quits piccosta (to delete a session: the Delete key)"));
                }
            }
            KeyCode::Esc if self.restart_waiting => {
                self.restart_waiting = false;
                self.notice = Some("Not restarting.".into());
            }
            KeyCode::Esc if self.current().busy() => self.interrupt(),
            KeyCode::Esc => self.panes[self.focus].scroll = 0,
            KeyCode::Char('x') if control => {
                self.chord = true;
                self.notice = Some("ctrl+x: ctrl+s sends now".into());
            }
            // Ctrl+enter arrives as itself only where the terminal reports it (not through tmux
            // without extended-keys); ctrl+x ctrl+s works anywhere.
            KeyCode::Enter if control => self.send_now(),
            KeyCode::Enter if alt || key.modifiers.contains(KeyModifiers::SHIFT) => self.focused().draft.insert("\n"),
            KeyCode::Char('j') if control => self.focused().draft.insert("\n"),
            KeyCode::Enter => self.submit(),
            // A new session: where, from a list that starts with this directory (enter keeps it).
            KeyCode::Char('n') if control => {
                self.command("new");
            }
            // Up and down the sidebar's list, one session at a time. Passing an earlier session only
            // shows it, so this never adds an open session.
            KeyCode::Up | KeyCode::Down if alt => {
                let listed = self.listed();
                if let Some(at) = listed.iter().position(|&index| index == self.panes[self.focus].slot) {
                    let next = if key.code == KeyCode::Up { (at + listed.len() - 1) % listed.len() } else { (at + 1) % listed.len() };
                    self.show(listed[next]);
                }
            }
            KeyCode::Char('s') if control => self.sessions(""),
            KeyCode::Left if alt => self.focus = self.focus.saturating_sub(1),
            KeyCode::Right if alt => self.focus = (self.focus + 1).min(self.panes.len() - 1),
            KeyCode::Char('\\') if control => self.split(),
            KeyCode::Char('4') if control => self.split(),
            KeyCode::Char(']') if control => self.close_pane(),
            KeyCode::Char('5') if control => self.close_pane(),
            KeyCode::Char('o') if control => {
                self.flags.tools_open = !self.flags.tools_open;
            }
            KeyCode::Char('t') if control => {
                self.flags.thinking_open = !self.flags.thinking_open;
            }
            KeyCode::Char('b') if control => self.sidebar = !self.sidebar,
            KeyCode::PageUp => {
                let step = self.pane_rects.get(self.focus).map_or(10, |rect| usize::from(rect.height / 2));
                self.panes[self.focus].scroll += step;
            }
            KeyCode::PageDown => {
                let step = self.pane_rects.get(self.focus).map_or(10, |rect| usize::from(rect.height / 2));
                let scroll = &mut self.panes[self.focus].scroll;
                *scroll = scroll.saturating_sub(step);
            }
            // Tab finishes a command name from what's typed: the first that fits.
            KeyCode::Tab if self.current().draft.text.starts_with('/') && !self.current().draft.text.contains(char::is_whitespace) => {
                if let Some(name) = commands::completions(&self.current().draft.text, false).first() {
                    let text = format!("{name} ");
                    self.focused().draft = Editor::with(text);
                }
            }
            // Down from a recalled prompt steps to the newer one, and past the newest to empty.
            KeyCode::Down if !shift && self.history.contains(&self.current().draft.text) => {
                let current = self.current().draft.text.clone();
                let position = self.history.iter().rposition(|entry| *entry == current).unwrap_or(0);
                let text = self.history.get(position + 1).cloned().unwrap_or_default();
                self.focused().draft = Editor::with(text);
            }
            // Up from the editor's first line takes back what's queued, as in Claude Code.
            KeyCode::Up if !shift && !self.current().draft.text[..self.current().draft.cursor].contains('\n') && self.take_back() => {}
            KeyCode::Up if !shift && (self.current().draft.text.is_empty() || self.history.contains(&self.current().draft.text)) => {
                let current = self.current().draft.text.clone();
                let position = self.history.iter().rposition(|entry| *entry == current).unwrap_or(self.history.len());
                if let Some(previous) = position.checked_sub(1).and_then(|index| self.history.get(index)).cloned() {
                    self.focused().draft = Editor::with(previous);
                }
            }
            _ => self.focused().draft.key(key),
        }
    }

    fn picker_key(&mut self, key: KeyEvent) {
        let Some(picker) = &mut self.picker else { return };
        if picker.confirming {
            match key.code {
                KeyCode::Enter => {
                    let (choice, delete) = (picker.chosen(), picker.delete);
                    self.picker = None;
                    if let (Some(choice), Some(delete)) = (choice, delete) {
                        delete(self, choice);
                    }
                }
                // Keeping it closes the list too: whoever asked to delete is done with it.
                KeyCode::Esc => self.picker = None,
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) && picker.delete.is_some() => picker.confirming = true,
            KeyCode::Delete if picker.delete.is_some() => picker.confirming = true,
            KeyCode::Esc => self.picker = None,
            KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
            KeyCode::Down => picker.selected += 1,
            KeyCode::Backspace => {
                picker.filter.pop();
                picker.selected = 0;
            }
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.filter.push(character);
                picker.selected = 0;
            }
            KeyCode::Enter => {
                let (choice, apply, answer) = (picker.chosen(), picker.apply, picker.answer.take());
                self.picker = None;
                match (choice, answer) {
                    (Some(choice), Some(answer)) => {
                        let _ = answer.send(ANSWERS.into_iter().find(|offered| *offered == choice).unwrap_or(ANSWERS[2]));
                    }
                    (Some(choice), None) => apply(self, choice),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // Abort every run, let the sessions write their last messages, then stop what's left.
    async fn shutdown(&mut self, reason: &str) {
        for slot in &mut self.slots {
            if let Some((handle, _)) = slot.live.take() {
                handle.abort();
            }
            if let Some(shared) = &slot.shared {
                shared.jobs.shutdown(reason);
                shared.web.shutdown();
            }
        }
        crate::tools::bash::kill_all_running();
        let tasks = std::mem::take(&mut self.tasks);
        let _ = tokio::time::timeout(SHUTDOWN_WAIT, futures_util::future::join_all(tasks)).await;
        crate::voice::shutdown(false).await;
        crate::provider::claude::shutdown().await;
        crate::tools::bash::kill_all_running();
    }

    fn draw(&mut self, frame: &mut Frame) {
        // How long the focused draft has been on the way to /descent. The screen is drawn again at
        // least every agent scan, so the name shows up within one of the 12 s mark.
        let toward = commands::toward_descent(&self.current().draft.text);
        self.descent_since = if toward { self.descent_since.or_else(|| Some(Instant::now())) } else { None };
        let [main, status] = Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(frame.area());
        let sidebar = self.sidebar && main.width >= SIDEBAR_MIN_SCREEN;
        let [side, panes] = Layout::horizontal([Constraint::Length(if sidebar { SIDEBAR_WIDTH } else { 0 }), Constraint::Fill(1)]).areas(main);
        self.sidebar_rect = side;
        if sidebar {
            self.draw_sidebar(frame, side);
        }
        let rects = Layout::horizontal(vec![Constraint::Fill(1); self.panes.len()]).split(panes);
        self.pane_rects = rects.to_vec();
        for (index, rect) in rects.iter().enumerate() {
            self.draw_pane(frame, index, *rect);
        }
        // A mouse selection, over what was just drawn: a double click's ends grow to whole words, a
        // triple click's to whole rows; copied when due (before the status line, which says so),
        // then shown reversed.
        if let Some(selection) = &mut self.selection {
            let buffer = frame.buffer_mut();
            let (mut anchor, mut head) = (selection.anchor, selection.head);
            if selection.clicks > 1 {
                let whole_rows = selection.clicks == 3;
                let (anchor_start, anchor_end) = clip::expand(buffer, selection.rect, anchor, whole_rows);
                let (head_start, head_end) = clip::expand(buffer, selection.rect, head, whole_rows);
                (anchor, head) = if (anchor.y, anchor.x) <= (head.y, head.x) { (anchor_start, head_end) } else { (anchor_end, head_start) };
            }
            if std::mem::take(&mut selection.copy) {
                let text = clip::text(buffer, selection.rect, anchor, head);
                let via = clip::copy(&text);
                self.notice = Some(format!("Copied {} characters with {via}", text.chars().count()));
            }
            clip::highlight(buffer, selection.rect, anchor, head);
        }
        let line = self.notice.clone().unwrap_or_else(|| HELP.to_string());
        let cwd = render::home_relative(&self.current().cwd);
        let room = usize::from(status.width).saturating_sub(cwd.width() + 2);
        let text = format!(" {:<room$} {cwd}", render::fit(&line, room));
        frame.render_widget(Paragraph::new(text).style(Style::new().fg(theme::DIM)), status);
        if self.picker.is_some() {
            self.draw_picker(frame, main);
        }
    }

    fn draw_sidebar(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::new().borders(Borders::RIGHT).border_style(Style::new().fg(theme::DIM));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let mut lines = vec![Line::from(Span::styled(" piccosta", Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD))), Line::default()];
        self.sidebar_rows.clear();
        let room = usize::from(inner.width).saturating_sub(3);
        let shown = self.panes[self.focus].slot;
        // A heading over each project's sessions; open ones carry a mark, earlier ones are dim.
        let mut project: Option<PathBuf> = None;
        for index in self.listed() {
            let slot = &self.slots[index];
            if project.as_ref() != Some(&slot.cwd) {
                if project.is_some() {
                    lines.push(Line::default());
                }
                let name = if std::env::var_os("HOME").is_some_and(|home| slot.cwd == std::path::Path::new(&home)) { "~".to_string() } else { slot.cwd.file_name().map_or_else(|| slot.cwd.display().to_string(), |name| name.to_string_lossy().into_owned()) };
                lines.push(Line::from(Span::styled(format!(" {}", render::fit(&name, room + 2)), Style::new().fg(theme::DIM).add_modifier(Modifier::BOLD))));
                project = Some(slot.cwd.clone());
            }
            let (mark, color) = match &slot.state {
                State::Running => (render::SPINNER[self.frame % render::SPINNER.len()], theme::ACCENT),
                State::Retrying(_) => ("↻", theme::USER),
                State::Compacting => ("◐", theme::ACCENT),
                State::Idle if slot.elsewhere && slot.working_elsewhere => (render::SPINNER[self.frame % render::SPINNER.len()], theme::ACCENT),
                State::Idle if slot.elsewhere => ("◎", theme::ACCENT),
                State::Idle if slot.unread => ("●", theme::OK),
                State::Idle => (" ", theme::DIM),
            };
            let title_style = match (index == shown, slot.live.is_some()) {
                (true, _) => Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD),
                (false, true) => Style::new(),
                (false, false) => Style::new().fg(theme::DIM),
            };
            self.sidebar_rows.push((inner.y + lines.len() as u16, index));
            // A manager's row is bold, so it reads as the one that heads its group.
            let title_style = if slot.manager { title_style.add_modifier(Modifier::BOLD) } else { title_style };
            let (mark, room) = if slot.parent.is_some() { (format!("  ↳ {mark} "), room.saturating_sub(3)) } else { (format!(" {mark} "), room) };
            lines.push(Line::from(vec![Span::styled(mark, Style::new().fg(color)), Span::styled(render::fit(&slot.title, room), title_style)]));
            // The agents it has running with no session to open, one row each: job (or "agent"),
            // how long, then the model, which is what the sidebar's width cuts.
            for agent in slot.agents.iter().filter(|agent| agent.session.is_none()) {
                let model = agent.model.as_deref().map_or("", |model| model.rsplit('/').next().unwrap_or(model));
                let text = format!("{} {} {model}", agent.job.as_deref().unwrap_or("agent"), render::duration(agent.running_for));
                lines.push(Line::from(Span::styled(format!("   ↳ {}", render::fit(&text, room.saturating_sub(2))), Style::new().fg(theme::ACCENT))));
            }
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_pane(&mut self, frame: &mut Frame, index: usize, rect: Rect) {
        let focused = index == self.focus;
        let block = if index > 0 { Block::new().borders(Borders::LEFT).border_style(Style::new().fg(theme::DIM)) } else { Block::new() };
        let inner = pane_inner(index, rect);
        frame.render_widget(block, rect);
        let slot_index = self.panes[index].slot;
        let width = inner.width;

        // The editor (focused pane only) and what's queued above it.
        let (editor_rows, cursor) = self.slots[slot_index].draft.layout(usize::from(width.saturating_sub(4)).max(1));
        let queued = self.slots[slot_index].live.as_ref().map(|(handle, _)| handle.queued()).unwrap_or_default();
        let queued: Vec<String> = queued.0.into_iter().chain(queued.1).collect();
        // While a command name is being typed, the ones it could be, on one line (tab takes the first).
        let draft = &self.slots[slot_index].draft.text;
        let descent = focused && self.descent_since.is_some_and(|since| since.elapsed() >= commands::DESCENT_REVEAL);
        let completions = if draft.starts_with('/') && !draft.contains(char::is_whitespace) { commands::completions(draft, descent) } else { Vec::new() };
        let editor_height = if focused { editor_rows.len().clamp(1, EDITOR_MAX_ROWS) as u16 + 2 } else { 0 };
        let queued_height = if focused { (queued.len().min(3) + usize::from(!completions.is_empty())) as u16 } else { 0 };
        let [header, body, queue, editor] = Layout::vertical([Constraint::Length(1), Constraint::Fill(1), Constraint::Length(queued_height), Constraint::Length(editor_height)]).areas(inner);

        let slot = &self.slots[slot_index];
        let mut spans = vec![Span::styled(render::fit(&slot.title, usize::from(width) / 2), if focused { Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD) } else { Style::new().add_modifier(Modifier::BOLD) })];
        let mut facts = format!("  {} · {} · {}/{} context", slot.model, slot.thinking, crate::cache::tokens(slot.context_tokens), crate::cache::tokens(slot.context_window));
        if !slot.team.models.is_empty() {
            let names: Vec<&str> = slot.team.models.iter().map(|model| model.rsplit('/').next().unwrap_or(model)).collect();
            facts.push_str(&format!(" · {} with {}", if slot.team.pair { "pair" } else { "trio" }, names.join(" and ")));
        }
        if slot.live.is_none() && slot.elsewhere {
            facts.push_str(" · running in another piccosta, followed here");
        }
        if slot.cost > 0.0 {
            facts.push_str(&format!(" · ${:.2}", slot.cost));
        }
        spans.push(Span::styled(facts, Style::new().fg(theme::DIM)));
        match &slot.state {
            State::Running => spans.push(Span::styled(format!("  {} working", render::SPINNER[self.frame % render::SPINNER.len()]), Style::new().fg(theme::ACCENT))),
            State::Idle if slot.live.is_none() && slot.working_elsewhere => spans.push(Span::styled(format!("  {} working", render::SPINNER[self.frame % render::SPINNER.len()]), Style::new().fg(theme::ACCENT))),
            State::Retrying(text) => spans.push(Span::styled(format!("  ↻ {text}"), Style::new().fg(theme::USER))),
            State::Compacting => spans.push(Span::styled("  ◐ compacting", Style::new().fg(theme::ACCENT))),
            State::Idle => {}
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), header);

        let height = usize::from(body.height);
        let mut rows = self.transcript(slot_index, width, height + self.panes[index].scroll);
        // While it works, a line under the conversation says so, its dots moving and its time
        // counting, so it's plain the model is still at it.
        let slot = &self.slots[slot_index];
        if slot.busy() {
            let what = match &slot.state {
                State::Compacting => "compacting",
                State::Retrying(_) => "waiting to retry",
                _ => "working",
            };
            let dots = ".".repeat(1 + self.frame / WORKING_DOT_FRAMES % 3);
            let seconds = slot.working_since.map_or(String::new(), |since| format!(" · {}", render::duration(since.elapsed())));
            let restart = if self.restart_waiting { " · piccosta restarts when every turn has ended (esc or /restart: don't)" } else { "" };
            rows.push(Line::default());
            rows.push(Line::from(Span::styled(format!("{} {what}{dots:<3}{seconds}{restart}", render::SPINNER[self.frame % render::SPINNER.len()]), Style::new().fg(theme::ACCENT))));
        }
        let scroll = self.panes[index].scroll.min(rows.len().saturating_sub(height));
        self.panes[index].scroll = scroll;
        let end = rows.len() - scroll;
        let start = end.saturating_sub(height);
        // Rows sit at the bottom, as a chat does, until they fill the pane.
        let top = body.y + (height - (end - start)) as u16;
        frame.render_widget(Paragraph::new(rows[start..end].to_vec()), Rect { y: top, height: (end - start) as u16, ..body });
        if scroll > 0 {
            let note = format!(" ↓ {scroll} more ");
            let at = Rect { x: body.right().saturating_sub(note.width() as u16), y: body.bottom().saturating_sub(1), width: note.width() as u16, height: 1 };
            frame.render_widget(Paragraph::new(note).style(Style::new().fg(theme::USER)), at);
        }

        if focused {
            let mut lines: Vec<Line> = queued.iter().take(3).map(|text| Line::from(Span::styled(format!("↳ {}", render::fit(text.lines().next().unwrap_or(""), usize::from(width) - 2)), Style::new().fg(theme::DIM)))).collect();
            if !completions.is_empty() {
                lines.push(Line::from(Span::styled(render::fit(&completions.join("  "), usize::from(width)), Style::new().fg(theme::ACCENT))));
            }
            frame.render_widget(Paragraph::new(lines), queue);
            let border = Block::bordered().border_type(BorderType::Rounded).border_style(Style::new().fg(if self.slots[slot_index].busy() { theme::DIM } else { theme::ACCENT }));
            let text_area = border.inner(editor).inner(ratatui::layout::Margin { horizontal: 1, vertical: 0 });
            frame.render_widget(border, editor);
            let visible = usize::from(text_area.height);
            let first = (cursor.0 + 1).saturating_sub(visible);
            if self.slots[slot_index].draft.text.is_empty() {
                let view = &self.slots[slot_index];
                let hint = match (view.busy(), view.live.is_some(), view.elsewhere) {
                    (true, ..) => "queue a message (read after the current step) · ^x^s sends now · ↑ takes it back · esc stops",
                    (false, false, true) => "another piccosta is running this session; it's shown here as it goes",
                    (false, false, false) => "an earlier session: type to carry on with it",
                    (false, true, _) if view.team.pair => "ask anything · the two models answer each other until one is done, then you · /pair off works alone",
                    (false, true, _) if !view.team.models.is_empty() => "ask anything · the other model checks each answer, then you · /trio off works alone",
                    (false, true, _) => "ask anything · / for commands · /pair or /trio <model> to work with a second model",
                };
                frame.render_widget(Paragraph::new(hint).style(Style::new().fg(theme::DIM)), text_area);
            } else {
                let shown: Vec<Line> = editor_rows.iter().skip(first).take(visible).map(|row| Line::from(row.clone())).collect();
                frame.render_widget(Paragraph::new(shown), text_area);
                // Text selected with shift+arrows, from its first character to its last.
                let draft = &self.slots[slot_index].draft;
                if let Some(range) = draft.selection() {
                    let last = draft.text[..range.end].char_indices().last().map_or(range.start, |(at, _)| at);
                    let width = usize::from(text_area.width).max(1);
                    let place = |(row, column): (usize, usize)| Position::new(text_area.x + column as u16, text_area.y + row.saturating_sub(first) as u16);
                    clip::highlight(frame.buffer_mut(), text_area, place(draft.spot(width, range.start)), place(draft.spot(width, last)));
                }
            }
            if self.picker.is_none() {
                frame.set_cursor_position(Position::new(text_area.x + cursor.1 as u16, text_area.y + (cursor.0 - first) as u16));
            }
        }
    }

    // A session's rows, drawn from the bottom up until `wanted` are there (or the session's start).
    fn transcript(&mut self, slot_index: usize, width: u16, wanted: usize) -> Rows {
        let flags = self.flags;
        let frame = self.frame;
        let slot = &mut self.slots[slot_index];
        let mut chunks: Vec<Rows> = Vec::new();
        let mut count = 0;
        if let Some(parts) = &slot.partial {
            let rows = render::assistant(parts, None, &slot.tools, flags, frame, width, &mut self.tex);
            count += rows.len();
            chunks.push(rows);
        }
        for index in (0..slot.messages.len()).rev() {
            if count >= wanted {
                break;
            }
            let cached = slot.cache.get(&index).filter(|(cached_width, cached_flags, _)| *cached_width == width && *cached_flags == flags);
            let rows = match cached {
                Some((_, _, rows)) => rows.clone(),
                None => {
                    let mut rows = message_rows(&slot.messages[index], &slot.tools, flags, frame, width, &mut self.tex);
                    // A response from a different model than the one before it says whose it is
                    // (a pair's turns, a /model switch).
                    let model_of = |message: &AgentMessage| if let AgentMessage::Assistant(assistant) = message { Some(format!("{}/{}", assistant.provider, assistant.model)) } else { None };
                    if let Some(model) = model_of(&slot.messages[index])
                        && slot.messages[..index].iter().rev().find_map(model_of).is_some_and(|previous| previous != model)
                    {
                        rows.insert(0, Line::from(Span::styled(format!("  {model}"), Style::new().fg(theme::DIM).add_modifier(Modifier::ITALIC))));
                    }
                    // A message with a tool still running keeps redrawing for its spinner.
                    let running = slot.call_owner.iter().any(|(id, owner)| *owner == index && slot.tools.get(id).is_some_and(|tool| tool.running));
                    if !running {
                        slot.cache.insert(index, (width, flags, rows.clone()));
                    }
                    rows
                }
            };
            count += rows.len();
            chunks.push(rows);
        }
        chunks.into_iter().rev().flatten().collect()
    }

    fn draw_picker(&mut self, frame: &mut Frame, area: Rect) {
        let Some(picker) = &self.picker else { return };
        let shown = picker.shown();
        let width = area.width.min(PICKER_MAX_WIDTH.max(picker.title.chars().count() as u16 + 4));
        let height = (shown.len() as u16 + 3).min(area.height.saturating_sub(4)).max(4);
        let rect = Rect { x: area.x + (area.width - width) / 2, y: area.y + area.height / 5, width, height };
        frame.render_widget(Clear, rect);
        let title = if picker.confirming { "delete it? enter deletes · esc keeps it" } else { &picker.title };
        let block = Block::bordered().border_type(BorderType::Rounded).border_style(Style::new().fg(theme::ACCENT)).title(format!(" {title} "));
        let inner = block.inner(rect);
        frame.render_widget(block, rect);
        let selected = picker.selected.min(shown.len().saturating_sub(1));
        let visible = usize::from(inner.height.saturating_sub(1));
        let first = (selected + 1).saturating_sub(visible);
        let mut lines = vec![Line::from(vec![Span::styled("› ", Style::new().fg(theme::ACCENT)), Span::raw(picker.filter.clone())])];
        for (index, item) in shown.iter().enumerate().skip(first).take(visible) {
            let style = match (index == selected, picker.confirming) {
                (true, true) => Style::new().fg(theme::REMOVED).add_modifier(Modifier::BOLD),
                (true, false) => Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD),
                _ => Style::new(),
            };
            lines.push(Line::from(Span::styled(format!("  {item}"), style)));
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }
}

// Where a pane's text goes: inside its left border (panes after the first have one) and a column
// of margin on each side.
fn pane_inner(index: usize, rect: Rect) -> Rect {
    let block = if index > 0 { Block::new().borders(Borders::LEFT) } else { Block::new() };
    block.inner(rect).inner(ratatui::layout::Margin { horizontal: 1, vertical: 0 })
}

fn message_rows(message: &AgentMessage, tools: &HashMap<String, ToolView>, flags: Flags, frame: usize, width: u16, tex: &mut tex::Tex) -> Rows {
    match message {
        // A manager's news (a lane's turn, a schedule) comes in as its message, "[<time>, what]",
        // but it's for the model: shown as a notice. The user's own words through the inbox,
        // "[<time>, from the user]", stay the user's.
        AgentMessage::User(user) if is_news(&user.content.text()) => render::news(&user.content.text(), width, tex, flags.tools_open),
        AgentMessage::User(user) => render::user(&user.content.text(), width, tex),
        AgentMessage::Assistant(assistant) => {
            render::assistant(&render::blocks(assistant), Some((&assistant.stop_reason, assistant.error_message.as_deref())), tools, flags, frame, width, tex)
        }
        AgentMessage::Custom(custom) if custom.custom_type == commands::REPORT => render::report(&custom.content.text(), width, tex),
        // A pair's or trio's turn: one line saying whose turn it is, not the note the model reads.
        // Notes from before pairs went back and forth name a `partner` checking a `lead`.
        AgentMessage::Custom(custom) if custom.custom_type == crate::host::PAIR_TYPE => {
            let name = |key: &str| custom.details.as_ref().and_then(|details| details.get(key)?.as_str()).map(|model| model.rsplit('/').next().unwrap_or(model).to_string());
            let pair = name("mode").as_deref() == Some("pair");
            let text = match (name("speaker").or(name("partner")), name("after").or(name("lead"))) {
                (Some(speaker), Some(after)) if pair => format!("⇄ {speaker} answers {after}"),
                (Some(speaker), Some(after)) => format!("⇄ {speaker} checks {after}'s answer"),
                (Some(speaker), None) if pair => format!("⇄ pair: {speaker} answers first"),
                (Some(speaker), None) => format!("⇄ trio: {speaker} answers first"),
                _ => "⇄ the second model checks the answer".into(),
            };
            vec![Line::default(), Line::from(Span::styled(text, Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD)))]
        }
        AgentMessage::Custom(custom) if custom.display => render::news(&custom.content.text(), width, tex, flags.tools_open),
        AgentMessage::CompactionSummary(_) => render::note("earlier context compacted into a summary", width, tex),
        AgentMessage::BranchSummary(_) => render::note("a branch's summary", width, tex),
        _ => Vec::new(),
    }
}

// Whether a message is a manager's news rather than the user's words: it starts "[<ISO time>, " and
// isn't marked as from the user.
fn is_news(text: &str) -> bool {
    let Some(rest) = text.strip_prefix('[') else { return false };
    let Some((stamp, what)) = rest.split_once(", ") else { return false };
    stamp.len() >= 16 && stamp.as_bytes()[4] == b'-' && stamp.as_bytes()[10] == b'T' && !what.starts_with("from the user")
}

// A streaming response's delta, applied to its parts (pi's assistant message events).
fn apply_delta(parts: &mut Vec<Part>, event: &Value) {
    let index = event.get("contentIndex").and_then(Value::as_u64).unwrap_or(0) as usize;
    let mut set = |part: Part| {
        while parts.len() <= index {
            parts.push(Part::Text(String::new()));
        }
        parts[index] = part;
    };
    let text = |key: &str| event.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    match event.get("type").and_then(Value::as_str).unwrap_or("") {
        "text_start" => set(Part::Text(String::new())),
        "thinking_start" => set(Part::Thinking(String::new())),
        "toolcall_start" => set(Part::Call { id: text("id"), name: text("toolName"), arguments: Default::default() }),
        "toolcall_end" => {
            let call = event.get("toolCall");
            let field = |key: &str| call.and_then(|call| call.get(key)).and_then(Value::as_str).unwrap_or("").to_string();
            let arguments = call.and_then(|call| call.get("arguments")).and_then(Value::as_object).cloned().unwrap_or_default();
            set(Part::Call { id: field("id"), name: field("name"), arguments });
        }
        "text_delta" | "thinking_delta" => {
            if let Some(Part::Text(body) | Part::Thinking(body)) = parts.get_mut(index) {
                body.push_str(&text("delta"));
            }
        }
        _ => {}
    }
}

// A tool result's text blocks, from pi's AgentToolResult shape.
fn result_text(result: &Value) -> String {
    let blocks = result.get("content").and_then(Value::as_array).cloned().unwrap_or_default();
    blocks.iter().filter_map(|block| block.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n")
}

// The prompt being written. `cursor` is a byte offset into `text`.
#[derive(Default)]
struct Editor {
    text: String,
    cursor: usize,
    // Earlier states, newest last, for ctrl+_ (Claude Code's undo).
    undo: Vec<(String, usize)>,
    // Where a shift+arrow selection began; the selection runs from here to the cursor.
    anchor: Option<usize>,
}

impl Editor {
    // An editor holding `text`, the cursor at its end.
    fn with(text: String) -> Editor {
        Editor { cursor: text.len(), text, undo: Vec::new(), anchor: None }
    }

    // What's typed or pasted replaces the selection, if there is one.
    fn insert(&mut self, text: &str) {
        self.save();
        self.remove_selection();
        self.text.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    // The selected text's byte range, when something is selected.
    fn selection(&self) -> Option<std::ops::Range<usize>> {
        let anchor = self.anchor?;
        (anchor != self.cursor).then(|| anchor.min(self.cursor)..anchor.max(self.cursor))
    }

    // Take the selected text out, the cursor where it began.
    fn remove_selection(&mut self) {
        if let Some(range) = self.selection() {
            self.cursor = range.start;
            self.text.drain(range);
        }
        self.anchor = None;
    }

    // The row and column `at` (a byte offset) is drawn at, wrapped at `width`.
    fn spot(&self, width: usize, at: usize) -> (usize, usize) {
        Editor { text: self.text.clone(), cursor: at, undo: Vec::new(), anchor: None }.layout(width).1
    }

    // Keep the state before an edit, for undo.
    fn save(&mut self) {
        if self.undo.len() == UNDO_STEPS {
            self.undo.remove(0);
        }
        self.undo.push((self.text.clone(), self.cursor));
    }

    // Cut a range out of the text, the cursor at its start.
    fn cut(&mut self, range: std::ops::Range<usize>) {
        self.save();
        self.anchor = None;
        self.cursor = range.start;
        self.text.drain(range);
    }

    // The start of the word before the cursor, and the end of the one after it. A word is a run of
    // letters and digits, so `/` and `.` part words, as Claude Code's alt+b and alt+f treat them.
    fn word_back(&self) -> usize {
        let before = self.text[..self.cursor].trim_end_matches(|character: char| !character.is_alphanumeric());
        before.char_indices().rev().take_while(|(_, character)| character.is_alphanumeric()).last().map_or(before.len(), |(index, _)| index)
    }

    fn word_forward(&self) -> usize {
        let after = &self.text[self.cursor..];
        let start = after.len() - after.trim_start_matches(|character: char| !character.is_alphanumeric()).len();
        self.cursor + after[start..].find(|character: char| !character.is_alphanumeric()).map_or(after.len(), |index| start + index)
    }

    fn take(&mut self) -> String {
        self.cursor = 0;
        self.undo.clear();
        self.anchor = None;
        std::mem::take(&mut self.text)
    }

    fn previous(&self) -> Option<usize> {
        self.text[..self.cursor].char_indices().last().map(|(index, _)| index)
    }

    fn next(&self) -> Option<usize> {
        self.text[self.cursor..].chars().next().map(|character| self.cursor + character.len_utf8())
    }

    fn line_start(&self) -> usize {
        self.text[..self.cursor].rfind('\n').map_or(0, |index| index + 1)
    }

    fn line_end(&self) -> usize {
        self.text[self.cursor..].find('\n').map_or(self.text.len(), |index| self.cursor + index)
    }

    // Move to the same column of the line above or below.
    fn vertical(&mut self, up: bool) {
        let start = self.line_start();
        let column = self.text[start..self.cursor].chars().count();
        let target = if up {
            if start == 0 {
                return;
            }
            self.text[..start - 1].rfind('\n').map_or(0, |index| index + 1)
        } else {
            let end = self.line_end();
            if end == self.text.len() {
                return;
            }
            end + 1
        };
        let line = &self.text[target..];
        let line = &line[..line.find('\n').unwrap_or(line.len())];
        self.cursor = target + line.char_indices().nth(column).map_or(line.len(), |(index, _)| index);
    }

    // The readline keys Claude Code and pi share. Alt+letter is a key, never a typed letter. A
    // movement with shift held selects from where it started; one without lets the selection go.
    fn key(&mut self, key: KeyEvent) {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let moves = matches!(key.code, KeyCode::Left | KeyCode::Right | KeyCode::Up | KeyCode::Down | KeyCode::Home | KeyCode::End);
        if moves {
            self.anchor = if shift { Some(self.anchor.unwrap_or(self.cursor)) } else { None };
        }
        match key.code {
            KeyCode::Backspace | KeyCode::Delete if self.selection().is_some() => {
                self.save();
                self.remove_selection();
            }
            KeyCode::Char('a') if control => self.cursor = self.line_start(),
            KeyCode::Char('e') if control => self.cursor = self.line_end(),
            KeyCode::Char('u') if control => self.cut(self.line_start()..self.cursor),
            KeyCode::Char('k') if control => self.cut(self.cursor..self.line_end()),
            // Ctrl+w goes back to whitespace (a whole path); alt+backspace to the word's start.
            KeyCode::Char('w') if control => {
                let start = self.text[..self.cursor].trim_end().rfind([' ', '\n']).map_or(0, |index| index + 1);
                self.cut(start..self.cursor);
            }
            // Terminals that don't report ctrl+backspace as such send it as ctrl+h.
            KeyCode::Backspace if alt || control => self.cut(self.word_back()..self.cursor),
            KeyCode::Char('h') if control => self.cut(self.word_back()..self.cursor),
            KeyCode::Char('d') if alt => self.cut(self.cursor..self.word_forward()),
            KeyCode::Char('b') if alt => self.cursor = self.word_back(),
            KeyCode::Char('f') if alt => self.cursor = self.word_forward(),
            KeyCode::Left if control => self.cursor = self.word_back(),
            KeyCode::Right if control => self.cursor = self.word_forward(),
            // Ctrl+_ (which terminals also send as ctrl+7 or ctrl+/).
            KeyCode::Char('_' | '7' | '/') if control => {
                if let Some((text, cursor)) = self.undo.pop() {
                    (self.text, self.cursor, self.anchor) = (text, cursor, None);
                }
            }
            KeyCode::Char(character) if !control && !alt => self.insert(&character.to_string()),
            KeyCode::Backspace => {
                if let Some(previous) = self.previous() {
                    self.cut(previous..self.cursor);
                }
            }
            // Delete, and ctrl+d with text in the editor: the character after the cursor.
            KeyCode::Delete | KeyCode::Char('d') if key.code == KeyCode::Delete || control => {
                if let Some(next) = self.next() {
                    self.cut(self.cursor..next);
                }
            }
            KeyCode::Left => self.cursor = self.previous().unwrap_or(0),
            KeyCode::Right => self.cursor = self.next().unwrap_or(self.cursor),
            KeyCode::Home => self.cursor = self.line_start(),
            KeyCode::End => self.cursor = self.line_end(),
            KeyCode::Up => self.vertical(true),
            KeyCode::Down => self.vertical(false),
            _ => {}
        }
    }

    // The text cut into rows of `width` columns, and the cursor's row and column.
    fn layout(&self, width: usize) -> (Vec<String>, (usize, usize)) {
        let mut rows = vec![String::new()];
        let mut used = 0;
        let mut cursor = (0, 0);
        for (index, character) in self.text.char_indices() {
            if index == self.cursor {
                cursor = (rows.len() - 1, used);
            }
            if character == '\n' {
                rows.push(String::new());
                used = 0;
                continue;
            }
            let character_width = unicode_width::UnicodeWidthChar::width(character).unwrap_or(0);
            if used + character_width > width {
                rows.push(String::new());
                used = 0;
            }
            rows.last_mut().into_iter().for_each(|row| row.push(character));
            used += character_width;
        }
        if self.cursor >= self.text.len() {
            if used >= width {
                rows.push(String::new());
                used = 0;
            }
            cursor = (rows.len() - 1, used);
        }
        (rows, cursor)
    }
}
