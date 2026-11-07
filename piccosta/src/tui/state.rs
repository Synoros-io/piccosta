// The terminal UI's state, kept so it comes back as it was: what each pane shows, how far it's scrolled
// and what's typed in it and not sent, which pane has focus, whether the sidebar shows, and the
// sessions open besides, until the user closes them. `piccosta -c` and
// /restart bring it back.
//
// One file per terminal, written when it changes (at each agent scan, so a crash loses a couple
// of seconds at most) and as piccosta quits. Without a terminal identity, each UI gets its own
// file and -c reads the newest. /restart passes its file explicitly, even without a terminal.
//
// /restart also leaves a handover for the image it execs: the background jobs still running, by
// session, the watcher (tools/reaper.rs), and the Claude Code processes waiting for their sessions'
// next messages (provider/claude). It's named by the process id, which the exec keeps, so only that
// image reads it, and it reads it once.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// Internal to /restart: its children ignore this unless started with --restore.
pub const RESTORE_FILE: &str = "PICCOSTA_TUI_STATE";

// The 30-day retention window. A daily mtime refresh keeps an unchanged live UI well inside
// it; pruning daily too avoids scanning every terminal's snapshots on each agent scan.
const SNAPSHOT_RETENTION: std::time::Duration = std::time::Duration::from_secs(30 * 24 * 60 * 60);
const SNAPSHOT_MAINTENANCE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

#[derive(Default)]
pub struct Saved {
    text: Option<String>,
    refreshed: Option<std::time::Instant>,
    pruned: Option<std::time::Instant>,
}

pub fn file(agent_dir: &Path) -> PathBuf {
    use sha2::Digest;
    // A tmux pane survives its process (and even a replacement tty after SIGKILL).
    let terminal = std::env::var("TMUX").ok().zip(std::env::var("TMUX_PANE").ok())
        .map(|(server, pane)| format!("{}:{pane}", server.split(',').next().unwrap_or(&server)))
        .or_else(|| std::env::var("WT_SESSION").ok());
    #[cfg(unix)]
    let terminal = terminal.or_else(|| {
        let name = unsafe { libc::ttyname(libc::STDIN_FILENO) };
        (!name.is_null()).then(|| unsafe { std::ffi::CStr::from_ptr(name) }.to_string_lossy().into_owned())
    });
    let id = terminal.map_or_else(|| uuid::Uuid::new_v4().to_string(), |name| sha2::Sha256::digest(name.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect());
    agent_dir.join(format!("tui-state-{id}.json"))
}

#[derive(Serialize, Deserialize)]
pub struct State {
    pub panes: Vec<Kept>,
    pub focus: usize,
    pub sidebar: bool,
    pub open: Vec<Kept>,
}

#[derive(Serialize, Default)]
pub struct Handover {
    // The watcher's pipe descriptor and pid (tools/reaper.rs).
    pub reaper: Option<(i32, u32)>,
    pub sessions: Vec<(PathBuf, Vec<crate::tools::jobs::HandedJob>)>,
    pub claude: Vec<crate::provider::claude::Handed>,
    // As read back: each Claude Code record that couldn't be read but gives its pid and start time,
    // with its pipes' descriptors when those read too.
    #[serde(skip)]
    pub unread: Vec<(u32, u64, Vec<i32>)>,
}

// A session as it was left: its file (None for a new one never saved, which starts again in its
// directory), its directory, how far its pane was scrolled, and what was typed and not sent.
#[derive(Serialize, Deserialize)]
pub struct Kept {
    pub session: Option<PathBuf>,
    #[serde(default)]
    pub title: String,
    pub cwd: PathBuf,
    #[serde(default)]
    pub scroll: usize,
    #[serde(default)]
    pub draft: String,
}

pub fn load(agent_dir: &Path, own: &Path, legacy: bool) -> Option<State> {
    // An older /restart didn't pass a private file. Read exactly what it saved, not another UI.
    // An existing own snapshot never falls back to another UI, even if it can't be read.
    let path = if legacy { agent_dir.join("tui-state.json") } else if own.exists() { own.to_path_buf() } else {
        std::fs::read_dir(agent_dir).ok()?.flatten().filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("tui-state") && !name.starts_with("tui-state-single-") && name.ends_with(".json")
        }).filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
            .max_by_key(|(time, _)| *time)?.1
    };
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

impl Saved {
    // Write changed content whole, so a piccosta starting meanwhile never reads half of one.
    pub fn save(&mut self, path: &Path, text: &str) -> std::io::Result<()> {
        if self.text.as_deref() != Some(text) || !path.exists() {
            let partial = path.with_extension("json.partial");
            std::fs::write(&partial, text)?;
            std::fs::rename(partial, path)?;
            self.text = Some(text.to_string());
            self.refreshed = Some(std::time::Instant::now());
        } else if self.refreshed.is_none_or(|time| time.elapsed() >= SNAPSHOT_MAINTENANCE) {
            std::fs::File::options().write(true).open(path)?.set_modified(std::time::SystemTime::now())
                .map_err(|error| std::io::Error::new(error.kind(), format!("could not refresh UI snapshot: {error}")))?;
            self.refreshed = Some(std::time::Instant::now());
        }
        if self.pruned.is_some_and(|time| time.elapsed() < SNAPSHOT_MAINTENANCE) { return Ok(()); }
        // Never prune the newest recovery point or the old shared state. Check the age at removal,
        // not while listing: another UI may have restored and rewritten an old snapshot meanwhile.
        let files = path.parent().into_iter().flat_map(std::fs::read_dir).flatten().flatten();
        let mut snapshots: Vec<_> = files.filter_map(|entry| {
            let file = entry.path();
            let name = file.file_name()?.to_str()?;
            if !name.starts_with("tui-state-") || !name.ends_with(".json") { return None; }
            Some((entry.metadata().ok()?.modified().ok()?, file))
        }).collect();
        snapshots.sort_by_key(|(time, _)| std::cmp::Reverse(*time));
        for (_, file) in snapshots.into_iter().skip(1) {
            if file.metadata().and_then(|metadata| metadata.modified()).ok().and_then(|time| time.elapsed().ok()).is_some_and(|age| age >= SNAPSHOT_RETENTION) {
                let _ = std::fs::remove_file(file);
            }
        }
        self.pruned = Some(std::time::Instant::now());
        Ok(())
    }
}

fn handover_file(agent_dir: &Path) -> PathBuf {
    agent_dir.join(format!("restart-{}.json", std::process::id()))
}

pub fn save_handover(agent_dir: &Path, handover: &Handover) -> std::io::Result<()> {
    std::fs::write(handover_file(agent_dir), serde_json::to_string(handover)?)
}

// The handover this process's image before left, removed as it's read; empty when there's none.
// Each part (the watcher, the jobs, each Claude Code) is read on its own, so one a different build
// wrote leaves the rest to take back. What can't be read is left out and the file is moved aside,
// kept; the error says what and where.
pub fn take_handover(agent_dir: &Path) -> (Handover, Option<String>) {
    let file = handover_file(agent_dir);
    let mut problems = Vec::new();
    let value = match std::fs::read_to_string(&file) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return (Handover::default(), None),
        text => text.map_err(|error| error.to_string()).and_then(|text| serde_json::from_str(&text).map_err(|error| error.to_string())),
    }.unwrap_or_else(|error| { problems.push(error); serde_json::Map::new() });
    let mut handover = Handover { reaper: part(&value, "reaper", &mut problems), sessions: part(&value, "sessions", &mut problems), ..Handover::default() };
    for record in part::<Vec<serde_json::Value>>(&value, "claude", &mut problems) {
        match crate::provider::claude::Handed::deserialize(&record) {
            Ok(query) => handover.claude.push(query),
            Err(error) => { problems.push(format!("claude: {error}")); handover.unread.extend(record["pid"].as_u64().zip(record["start"].as_u64()).map(|(pid, start)| (pid as u32, start, Vec::deserialize(&record["fds"]).unwrap_or_default()))); }
        }
    }
    if problems.is_empty() {
        let _ = std::fs::remove_file(file);
        return (handover, None);
    }
    let aside = file.with_extension(format!("unreadable-{}.json", crate::clock::now_ms()));
    let kept = std::fs::rename(&file, &aside).map_or_else(|error| format!("still at {} ({error})", file.display()), |()| format!("kept at {}", aside.display()));
    (handover, Some(format!("{}; the file is {kept}", problems.join("; "))))
}

// One part of a handover: its default when there's none, else what's there, or the default with
// the problem noted.
fn part<T: serde::de::DeserializeOwned + Default>(value: &serde_json::Map<String, serde_json::Value>, name: &str, problems: &mut Vec<String>) -> T {
    value.get(name).map_or(Ok(T::default()), T::deserialize).unwrap_or_else(|error| { problems.push(format!("{name}: {error}")); T::default() })
}
