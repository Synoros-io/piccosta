// The terminal UI's state, kept so it comes back as it was: what each pane shows, how far it's scrolled
// and what's typed in it and not sent, which pane has focus, whether the sidebar shows, and the
// sessions open besides (running when piccosta quit, or holding a draft). `piccosta -c` and
// /restart bring it back.
//
// One file per agent dir, written when it changes (looked at with each agent scan, so a crash
// loses a couple of seconds at most) and as piccosta quits. With two open, the one that changed
// last is the one that comes back.
//
// /restart also leaves a handover for the image it execs: the background jobs still running, by
// session, and the watcher (tools/reaper.rs). It's named by the process id, which the exec keeps, so only that
// image reads it, and it reads it once.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const FILE: &str = "tui-state.json";

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
    pub cwd: PathBuf,
    #[serde(default)]
    pub scroll: usize,
    #[serde(default)]
    pub draft: String,
}

pub fn load(agent_dir: &Path) -> Option<State> {
    serde_json::from_str(&std::fs::read_to_string(agent_dir.join(FILE)).ok()?).ok()
}

// Write it whole through a temporary file, so a piccosta starting meanwhile never reads half of one.
pub fn save(agent_dir: &Path, text: &str) -> std::io::Result<()> {
    let path = agent_dir.join(FILE);
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
