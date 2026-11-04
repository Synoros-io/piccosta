// Opening a GUI that runs piccosta as its agent: `piccosta zed [dir]` and `piccosta t3 [dir]`, one
// command each, so nobody has to edit a GUI's settings by hand.
//
// Each finds the GUI (settings' `gui.<name>`, a path, else its usual names on PATH), makes sure it
// runs this piccosta, and starts it on the directory:
// - Zed: its settings.json gets piccosta in `agent_servers`, over ACP. A file that's there is
//   changed only when its piccosta entry is missing or different, after a copy of it is kept, and
//   not at all when it isn't plain JSON with comments (then piccosta says what to add).
// - T3 Code: its Pi provider runs a program called `pi` from PATH unless told otherwise, so T3
//   starts with a folder first on its PATH where `pi` is this piccosta. Its telemetry is off unless
//   the environment says otherwise. T3 runs in the foreground, as its own command does.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// The names each GUI's program goes by on PATH (Arch packages Zed as zeditor).
const ZED_NAMES: [&str; 3] = ["zed", "zeditor", "zedit"];
const T3_NAMES: [&str; 1] = ["t3"];
// T3's telemetry switches, set off unless the environment already says.
const T3_QUIET: [(&str, &str); 2] = [("T3CODE_TELEMETRY_ENABLED", "false"), ("DO_NOT_TRACK", "1")];

// Input: the GUI's name and what follows it on the command line (a directory, else this one).
// Output: an error to show, or nothing once the GUI is started (T3 replaces this process).
pub fn open(name: &str, arguments: &[String]) -> Result<(), String> {
    let settings = crate::config::Settings::load(&crate::config::agent_dir(), std::env::current_dir().ok().as_deref());
    let dir = std::fs::canonicalize(arguments.first().map_or(".", String::as_str)).map_err(|error| format!("{}: {error}", arguments.first().map_or(".", String::as_str)))?;
    let exe = this_program()?;
    let names: &[&str] = if name == "zed" { &ZED_NAMES } else { &T3_NAMES };
    let program = find(&settings, name, names)?;
    if name == "zed" {
        if let Some(note) = configure_zed(&exe)? {
            eprintln!("{note}");
        }
        let mut command = Command::new(&program);
        command.arg(&dir).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        let started = crate::proc::own_group(&mut command).spawn();
        return started.map(drop).map_err(|error| format!("{}: {error}", program.display()));
    }
    let shim = crate::config::agent_dir().join("gui-bin");
    link_pi(&shim, &exe)?;
    let path = std::env::join_paths(std::iter::once(shim).chain(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()))).map_err(|error| error.to_string())?;
    let mut command = Command::new(&program);
    command.arg("--auto-bootstrap-project-from-cwd").arg(&dir).current_dir(&dir).env("PATH", path);
    for (variable, value) in T3_QUIET {
        if std::env::var_os(variable).is_none() {
            command.env(variable, value);
        }
    }
    eprintln!("Starting T3 Code on {}. The first time, turn on its Pi provider (Settings, Providers, Pi): its binary path, pi, runs this piccosta.", dir.display());
    Err(format!("{}: {}", program.display(), crate::proc::resume(&mut command)))
}

// This piccosta as it was started: found on PATH when started by name, so a link (~/bin/piccosta)
// stays a link and the GUI follows wherever it points; else the running binary.
fn this_program() -> Result<PathBuf, String> {
    let started = std::env::args_os().next().map(PathBuf::from).unwrap_or_default();
    let on_path = || std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).map(|dir| dir.join(&started)).find(|candidate| candidate.is_file());
    match started.components().count() {
        1 => on_path().map_or_else(|| std::env::current_exe().map_err(|error| error.to_string()), Ok),
        _ => std::path::absolute(&started).map_err(|error| error.to_string()),
    }
}

// The GUI's program: settings' gui.<name>, else the first of its names on PATH.
fn find(settings: &crate::config::Settings, name: &str, names: &[&str]) -> Result<PathBuf, String> {
    if let Some(path) = settings.get(&format!("gui.{name}")).and_then(Value::as_str) {
        return Ok(PathBuf::from(path));
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    names
        .iter()
        .flat_map(|program| std::env::split_paths(&path).map(move |dir| crate::platform::executable(&dir, program)))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| format!("Can't find {name} on PATH ({}); set \"gui\": {{\"{name}\": \"/path/to/it\"}} in piccosta's settings.json", names.join(", ")))
}

// `pi` in the shim folder, pointing at this piccosta.
fn link_pi(shim: &Path, exe: &Path) -> Result<(), String> {
    let link = crate::platform::executable(shim, "pi");
    if std::fs::read_link(&link).ok().as_deref() == Some(exe) {
        return Ok(());
    }
    std::fs::create_dir_all(shim).map_err(|error| format!("{}: {error}", shim.display()))?;
    let _ = std::fs::remove_file(&link);
    crate::proc::link(exe, &link).map_err(|error| format!("{}: {error}", link.display()))
}

// Make Zed's settings run this piccosta as an agent. Output: a note to print when it changed
// something, or an error saying what to add by hand.
fn configure_zed(exe: &Path) -> Result<Option<String>, String> {
    let config = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| Some(crate::config::home().join(".config"))).ok_or("No home folder")?;
    let path = config.join("zed").join("settings.json");
    let entry = json!({"type": "custom", "command": exe, "args": ["--mode", "acp"]});
    let by_hand = || format!("Add this to {} under \"agent_servers\": \"piccosta\": {entry}", path.display());
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let mut settings: Value = if text.trim().is_empty() {
        json!({})
    } else {
        serde_json::from_str(&crate::model::strip_json_comments(&text)).map_err(|_| by_hand())?
    };
    if settings.pointer("/agent_servers/piccosta") == Some(&entry) {
        return Ok(None);
    }
    let servers = settings.as_object_mut().ok_or_else(by_hand)?.entry("agent_servers").or_insert_with(|| json!({}));
    servers.as_object_mut().ok_or_else(by_hand)?.insert("piccosta".into(), entry);
    let kept = path.with_extension("json.before-piccosta");
    if !text.is_empty() && !kept.exists() {
        std::fs::write(&kept, &text).map_err(|error| format!("{}: {error}", kept.display()))?;
    }
    std::fs::create_dir_all(path.parent().unwrap_or(&config)).map_err(|error| error.to_string())?;
    std::fs::write(&path, serde_json::to_string_pretty(&settings).unwrap_or_default()).map_err(|error| format!("{}: {error}", path.display()))?;
    let comments = if text.contains("//") { format!("; its comments weren't kept, the file as it was is {}", kept.display()) } else { String::new() };
    Ok(Some(format!("Zed now runs piccosta as an agent ({}{comments}). Open the agent panel and pick piccosta.", path.display())))
}
