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
// session, and the watcher (tools/reaper.rs). It's named by the process id, which the exec keeps, so only that
// image reads it, and it reads it once.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// Internal to /restart: its children ignore this unless started with --restore.
pub const RESTORE_FILE: &str = "PICCOSTA_TUI_STATE";

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

#[derive(Serialize, Deserialize, Default)]
pub struct Handover {
    // The watcher's pipe descriptor and pid (tools/reaper.rs).
    pub reaper: Option<(i32, u32)>,
    pub sessions: Vec<(PathBuf, Vec<crate::tools::jobs::HandedJob>)>,
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

pub fn load(agent_dir: &Path, own: &Path) -> Option<State> {
    // An existing own snapshot never falls back to another UI, even if it can't be read.
    let path = if own.exists() { own.to_path_buf() } else {
        std::fs::read_dir(agent_dir).ok()?.flatten().filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("tui-state") && name.ends_with(".json")
        }).filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
            .max_by_key(|(time, _)| *time)?.1
    };
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

// Write it whole through a temporary file, so a piccosta starting meanwhile never reads half of one.
pub fn save(path: &Path, text: &str) -> std::io::Result<()> {
    let partial = path.with_extension("json.partial");
    std::fs::write(&partial, text)?;
    std::fs::rename(partial, path)
}

fn handover_file(agent_dir: &Path) -> PathBuf {
    agent_dir.join(format!("restart-{}.json", std::process::id()))
}

pub fn save_handover(agent_dir: &Path, handover: &Handover) -> std::io::Result<()> {
    std::fs::write(handover_file(agent_dir), serde_json::to_string(handover)?)
}

// The handover this process's image before left, removed as it's read; empty when there's none.
pub fn take_handover(agent_dir: &Path) -> Handover {
    let file = handover_file(agent_dir);
    let text = std::fs::read_to_string(&file).unwrap_or_default();
    let _ = std::fs::remove_file(file);
    serde_json::from_str(&text).unwrap_or_default()
}
