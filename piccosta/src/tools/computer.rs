// The computer tool: look at a desktop and use its mouse and keyboard (Linux, X11).
//
// What the model sees follows Anthropic's computer-use reference (anthropic-quickstarts,
// computer-use-demo/computer_use_demo/tools/computer.py, MIT, see LICENSE.md): one tool whose
// `action` says what to do, coordinates in the screenshot's pixels, xdotool moving the pointer and
// pressing the keys, and a fresh screenshot after every action so the model sees what it did. Two
// actions are added: `tree`, the accessibility tree, read over AT-SPI the way computer-use-linux
// reads it (MIT, src/atspi_tree.rs: from the registry's root down, each node's role, name, place,
// states and text); and `open`, which starts an app on the desktop.
//
// It works on a hidden desktop unless told otherwise, made the way agent-workspace-linux makes its
// X11 one (MIT, src/session/x11.rs): an Xvfb on the display `computer.display` names, with its own
// auth cookie, a session bus of its own and an accessibility bus of its own, so an app started
// there registers with neither of the user's. The accessibility bus is a plain dbus-daemon on
// at-spi's own config, not at-spi-bus-launcher: the launcher runs dbus-broker, which starts the
// registry as a systemd unit, and that fails for a bus systemd doesn't know. There's one per machine: the first call starts it
// and it stays up while any session that used it is open, since they all name the same display,
// which is what a queue of computer jobs sharing it needs; the last of them to close stops it.
// Only an Xvfb this piccosta started is used: its pid is kept, and before each reuse it has to be
// the process holding the display's sockets, so a display some other server has is refused. The
// apps on it get a HOME and a runtime dir of their own and X11 toolkits, so none of them finds the
// user's Wayland compositor, session bus or profiles. The main desktop (the user's own, this process's DISPLAY)
// is used only when a call asks for it, and only after the user says yes to that call
// (Tool::approval, agent.rs). It's reached through X11, so on a Wayland desktop only XWayland
// windows see the input and the screenshot.

use super::{image, Tool, ToolContext, ToolFuture, ToolOutput};
use crate::message::{Content, ToolDefinition};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use zbus::zvariant::{OwnedObjectPath, OwnedValue};

// Defaults for the `computer.*` settings. The display is one no desktop session or xvfb-run (:99)
// takes; the size is WXGA, the largest Anthropic's reference recommends for a model's
// screenshots, and under image.rs's 2000-pixel cap, so a hidden screen's screenshots go out
// unscaled.
const DEFAULT_DISPLAY: &str = ":77";
const DEFAULT_XVFB: &str = "Xvfb";
const DEFAULT_SIZE: &str = "1280x800x24";
// How long the hidden desktop's server and bus get to come up, and how often to look.
const START_WAIT: Duration = Duration::from_secs(10);
const START_POLL: Duration = Duration::from_millis(50);
// The wait before the screenshot that follows an action, for the app to draw what it did.
// Anthropic's reference waits 2 s; GTK apps on an Xvfb with no compositor repaint well within
// half a second of a click or a key.
const SETTLE: Duration = Duration::from_millis(500);
// An app being opened gets longer: Anthropic's 2 s.
const OPEN_SETTLE: Duration = Duration::from_secs(2);
// Between typed characters, and between a double click's two clicks: Anthropic's values.
const TYPING_DELAY_MS: &str = "12";
const DOUBLE_CLICK_DELAY_MS: &str = "10";
// Scroll clicks when the call doesn't say.
const DEFAULT_SCROLL_AMOUNT: u32 = 3;
// The tree's limits: computer-use-linux's default node cap and its snapshot deadline, its per-node
// text cap, and a wait per D-Bus call so one app that doesn't answer can't stall the whole read.
const TREE_MAX_NODES: usize = 1000;
const TREE_DEADLINE: Duration = Duration::from_secs(10);
const TREE_TEXT_CHARS: i32 = 4096;
const TREE_CALL_WAIT: Duration = Duration::from_secs(2);
// The AT-SPI registry's root, whose children are the applications.
const REGISTRY: &str = "org.a11y.atspi.Registry";
const REGISTRY_ROOT: &str = "/org/a11y/atspi/accessible/root";
// AT-SPI's state bits (the StateType enum, atspi-common's State) the tree shows, and SHOWING,
// which says whether a node is on screen at all.
const STATE_NAMES: [(u32, &str); 6] = [(4, "checked"), (7, "editable"), (10, "expanded"), (12, "focused"), (20, "pressed"), (23, "selected")];
const SHOWING: u32 = 25;
// Toolkits pick Wayland when they can, GTK 4 draws with the GPU an Xvfb doesn't have, and Qt hides
// its accessibility unless asked; on the hidden desktop they get X11 (GTK, Qt, SDL, Clutter,
// Firefox, Electron), software drawing and an accessibility tree.
const HIDDEN_ENV: [(&str, &str); 9] = [
    ("GDK_BACKEND", "x11"),
    ("GSK_RENDERER", "cairo"),
    ("QT_QPA_PLATFORM", "xcb"),
    ("QT_LINUX_ACCESSIBILITY_ALWAYS_ON", "1"),
    ("SDL_VIDEODRIVER", "x11"),
    ("CLUTTER_BACKEND", "x11"),
    ("MOZ_ENABLE_WAYLAND", "0"),
    ("ELECTRON_OZONE_PLATFORM_HINT", "x11"),
    ("XDG_SESSION_TYPE", "x11"),
];
// What leads an app to the user's compositor: Wayland's socket and display, and the sockets of
// the compositors that have their own (Hyprland, Sway, i3, niri) or start XWayland on demand
// (GNOME's), and the names of the user's desktop and backend, which apps use to pick a Wayland or
// portal path.
const COMPOSITOR_ENV: [&str; 11] = ["WAYLAND_DISPLAY", "WAYLAND_SOCKET", "HYPRLAND_INSTANCE_SIGNATURE", "SWAYSOCK", "I3SOCK", "NIRI_SOCKET", "GNOME_SETUP_DISPLAY", "XDG_ACTIVATION_TOKEN", "XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "XDG_BACKEND"];
// Where X servers listen for display N, in the filesystem and (with a leading @) in Linux's
// abstract namespace: X11's own convention, the one libxcb connects by.
const X11_SOCKET: &str = "/tmp/.X11-unix/X";
// /proc/net/unix's flag for a listening socket (__SO_ACCEPTCON).
const LISTENING: u32 = 0x10000;
// Where an app keeps its profile, which on the hidden desktop goes under its own HOME instead, so
// none finds the user's (or the single-instance socket of a browser the user has open, which
// would take the request to the user's screen). Chromium and Firefox also take a profile from
// their own variables, which would point them back at the user's.
const PROFILE_ENV: [&str; 8] = ["XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "CHROME_USER_DATA_DIR", "CHROME_CONFIG_HOME", "XRE_PROFILE_PATH", "XRE_PROFILE_LOCAL_PATH"];
// The hidden desktop's three processes, each recorded in its folder as `<name>.pid`.
const SERVICES: [&str; 3] = ["xvfb", "bus", "a11y"];
// at-spi's config for an accessibility bus, whose service folder starts the registry.
const ACCESSIBILITY_CONFIG: &str = "/usr/share/defaults/at-spi2/accessibility.conf";
const MAIN_QUESTION: &str = "Let the agent use your own desktop? It sees your screen and moves your mouse and keyboard.";

// The latest screenshot's scale on each display (screen pixels per screenshot pixel), which the
// next call's coordinates are multiplied by.
static SCALES: LazyLock<Mutex<HashMap<String, f64>>> = LazyLock::new(Mutex::default);

// The `computer.*` settings: the hidden desktop's display, its X server and screen size. Each
// session has its own, which holds its claim on the desktop once it has used it.
#[derive(Clone)]
pub struct Settings {
    display: String,
    xvfb: String,
    size: String,
    held: Arc<Mutex<Held>>,
}

// The session's claim on the desktop, and whether the session has closed, after which it makes no
// new one. Both change under one mutex, so a claim still starting the desktop as the session
// closes either finishes first and is let go with it, or finds it closed.
#[derive(Default)]
struct Held {
    user: Option<User>,
    closed: bool,
}

impl Settings {
    pub fn new(settings: &crate::config::Settings) -> Settings {
        let text = |key: &str, default: &str| settings.get(key).and_then(Value::as_str).map(str::trim).filter(|value| !value.is_empty()).unwrap_or(default).to_string();
        Settings { display: text("computer.display", DEFAULT_DISPLAY), xvfb: text("computer.xvfb", DEFAULT_XVFB), size: text("computer.size", DEFAULT_SIZE), held: Arc::default() }
    }

    // The hidden desktop's folder: in XDG_RUNTIME_DIR (the agent dir without one), named by
    // display.
    fn dir(&self) -> PathBuf {
        let base = std::env::var_os("XDG_RUNTIME_DIR").map_or_else(crate::config::agent_dir, PathBuf::from);
        base.join(format!("piccosta-desktop-{}", self.display.trim_start_matches(':')))
    }

    // The hidden desktop, checked and started when it isn't up, with this session's claim on it
    // held from now until the session closes: what the tool works on, and what bash puts a visual
    // session's commands on (Desktop::hide).
    pub async fn claim(&self) -> Result<Desktop, String> {
        let settings = self.clone();
        tokio::task::spawn_blocking(move || hidden(&settings)).await.map_err(|error| error.to_string())?
    }

    // Let go of this session's claim as it closes (runtime.rs), stopping the desktop when no other
    // session holds one (User's drop), after the session's apps have been stopped. The claim is
    // dropped after the mutex is let go: its drop takes the start lock, which a claim being made
    // holds while it waits for this mutex, so the locks are only ever taken start lock first.
    pub fn release(&self) {
        let user = {
            let mut held = self.held.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            held.closed = true;
            held.user.take()
        };
        drop(user);
    }
}

// A session's claim on the hidden desktop: a shared lock on its `users` file, held from the
// session's first call until it closes. The session that closes with nobody else holding it
// stops the desktop. A piccosta that's killed lets its claim go without stopping it, so the
// desktop then waits for the next session that uses it to close.
struct User {
    file: std::fs::File,
    dir: PathBuf,
}

impl Drop for User {
    fn drop(&mut self) {
        // Under the start lock, so no session starts using it between the look and the stop.
        let Ok(lock) = std::fs::File::create(self.dir.join("lock")) else { return };
        // SAFETY: flock on descriptors this function and its User own.
        unsafe {
            libc::flock(std::os::fd::AsRawFd::as_raw_fd(&lock), libc::LOCK_EX);
            if libc::flock(std::os::fd::AsRawFd::as_raw_fd(&self.file), libc::LOCK_EX | libc::LOCK_NB) == 0 {
                let _ = stop(&self.dir);
            }
        }
    }
}

// Give a command the hidden desktop's display, auth and buses, and a HOME and runtime dir of its
// own, with X11 toolkits and nothing that names the user's compositor or profiles.
fn hide(command: &mut Command, display: &str, dir: &Path) {
    let (home, runtime) = (dir.join("home"), dir.join("runtime"));
    for folder in [&home, &runtime] {
        let _ = std::fs::DirBuilder::new().recursive(true).mode(0o700).create(folder);
    }
    let bus = |name: &str| format!("unix:path={}", dir.join(name).display());
    for name in COMPOSITOR_ENV.iter().chain(&PROFILE_ENV) {
        command.env_remove(name);
    }
    command.env("DISPLAY", display).env("XAUTHORITY", dir.join("Xauthority")).envs(HIDDEN_ENV).env("HOME", home).env("XDG_RUNTIME_DIR", runtime);
    command.env("DBUS_SESSION_BUS_ADDRESS", bus("bus")).env("AT_SPI_BUS_ADDRESS", bus("a11y"));
}

// A desktop to work on: its X display and, for the hidden one, the folder holding its auth file
// and its buses' sockets. Without a folder it's this process's own desktop.
pub struct Desktop {
    display: String,
    dir: Option<PathBuf>,
}

impl Desktop {
    // A program run on this desktop.
    fn command(&self, program: &str) -> Command {
        let mut command = Command::new(program);
        command.env("DISPLAY", &self.display).stdin(Stdio::null());
        self.hide(&mut command);
        command
    }

    // Put a command on the hidden desktop (nothing, on the user's): what bash does in a visual
    // session, so an app it starts opens there and not on the user's screen. That stops accidents
    // only; a command can set its own DISPLAY and HOME and reach the user's desktop on purpose.
    pub fn hide(&self, command: &mut Command) {
        if let Some(dir) = &self.dir {
            hide(command, &self.display, dir);
        }
    }

    // One of the hidden desktop's buses ("bus", the session's, or "a11y"); None on the user's.
    fn bus(&self, name: &str) -> Option<String> {
        self.dir.as_ref().map(|dir| format!("unix:path={}", dir.join(name).display()))
    }

    // Whether its X server answers.
    fn up(&self) -> bool {
        self.command("xdotool").arg("getdisplaygeometry").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|status| status.success())
    }

    // Run a program on it and wait. Output: what it printed, or an error with what it said.
    async fn run(&self, program: &str, args: &[String]) -> Result<Vec<u8>, String> {
        let output = tokio::process::Command::from(self.command(program)).args(args).kill_on_drop(true).output().await.map_err(|error| format!("{program}: {error}"))?;
        match output.status.success() {
            true => Ok(output.stdout),
            false => Err(format!("{program} {}: {}", args.join(" "), String::from_utf8_lossy(&output.stderr).trim())),
        }
    }
}

// Start a program in a session of its own, so it outlives this piccosta, with its output in `log`
// (or nowhere). Waited for on a thread so it isn't left a zombie while this one runs. Output: its
// pid and a pidfd opened at spawn, which names it alone whatever later takes the pid.
fn detach(command: &mut Command, log: Option<PathBuf>) -> Result<(u32, OwnedFd), String> {
    // One file for both, so neither writes over the other.
    let file = log.and_then(|log| std::fs::File::create(log).ok());
    let output = |file: &Option<std::fs::File>| file.as_ref().and_then(|file| file.try_clone().ok()).map_or_else(Stdio::null, Stdio::from);
    command.stdout(output(&file)).stderr(output(&file));
    // SAFETY: setsid() is async-signal-safe and runs in the child between fork and exec.
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let mut child = command.spawn().map_err(|error| format!("{}: {error}", command.get_program().to_string_lossy()))?;
    let pid = child.id();
    // A process piccosta can't signal safely isn't left running (a kernel before 5.3 has no pidfds).
    let Some(pidfd) = pidfd(pid) else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("{} was stopped: piccosta couldn't open a pidfd to track it (needs Linux 5.3 or later)", command.get_program().to_string_lossy()));
    };
    std::thread::spawn(move || child.wait());
    Ok((pid, pidfd))
}

// A pidfd for process `pid`, or None when there's none (or the kernel can't).
fn pidfd(pid: u32) -> Option<OwnedFd> {
    // SAFETY: pidfd_open takes a pid and flags and returns a new descriptor (close-on-exec) or -1.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0) } as i32;
    // SAFETY: a descriptor pidfd_open just returned, owned by nothing else.
    (fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) })
}

// End the process a pidfd names: SIGTERM, then SIGKILL when it hasn't gone within START_WAIT (one
// that's stopped keeps a SIGTERM pending). Output: whether it's gone.
fn end(pidfd: &OwnedFd) -> bool {
    let wait = START_WAIT.as_millis() as i32;
    [libc::SIGTERM, libc::SIGKILL].into_iter().any(|signal| {
        let mut ready = libc::pollfd { fd: pidfd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: a signal through an open pidfd, which can reach no process but its own, then a
        // poll on it, which is readable once that process has exited.
        unsafe {
            libc::syscall(libc::SYS_pidfd_send_signal, pidfd.as_raw_fd(), signal, std::ptr::null::<libc::siginfo_t>(), 0);
            libc::poll(&mut ready, 1, wait) > 0
        }
    })
}

// Wait for `ready`, up to START_WAIT. Output: whether it came.
fn wait_for(ready: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while !ready() {
        if start.elapsed() > START_WAIT {
            return false;
        }
        std::thread::sleep(START_POLL);
    }
    true
}

// A process's start time (field 22 of /proc/<pid>/stat), which tells it from a later one given
// the same pid. None when there's no such process, or it has exited and waits to be reaped.
fn started(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The fields after the name, which is in parentheses and may hold spaces, start at field 3,
    // the state.
    let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    (fields.first() != Some(&"Z")).then(|| fields.get(19)?.parse().ok()).flatten()
}

// One of the hidden desktop's processes, by its record in `dir`: its pid and a pidfd for it while
// it still runs. The pidfd is opened before the start time is checked, so when that matches, the
// pidfd is the recorded process's (it held the pid all along) and not one given the pid later.
fn recorded(dir: &Path, name: &str) -> Option<(u32, OwnedFd)> {
    let record = std::fs::read_to_string(dir.join(format!("{name}.pid"))).ok()?;
    let (pid, start) = record.trim().split_once(' ')?;
    let pid = pid.parse().ok()?;
    let pidfd = pidfd(pid)?;
    (started(pid)? == start.parse::<u64>().ok()?).then_some((pid, pidfd))
}

// The inodes of the sockets X servers listen on for display `number`.
fn listeners(number: &str) -> Vec<String> {
    let table = std::fs::read_to_string("/proc/net/unix").unwrap_or_default();
    let path = format!("{X11_SOCKET}{number}");
    // Num RefCount Protocol Flags Type St Inode Path
    let listening = |fields: &[&str]| u32::from_str_radix(fields[3], 16).is_ok_and(|flags| flags & LISTENING != 0);
    let rows = table.lines().skip(1).map(|line| line.split_whitespace().collect::<Vec<_>>());
    rows.filter(|fields| fields.len() == 8 && fields[7].trim_start_matches('@') == path && listening(fields)).map(|fields| fields[6].to_string()).collect()
}

// Whether process `pid` holds every socket listening for display `number` (and there are some).
fn holds(pid: u32, number: &str) -> bool {
    let sockets = listeners(number);
    let open: Vec<String> = std::fs::read_dir(format!("/proc/{pid}/fd")).into_iter().flatten().flatten().filter_map(|fd| std::fs::read_link(fd.path()).ok()).map(|link| link.to_string_lossy().into_owned()).collect();
    !sockets.is_empty() && sockets.iter().all(|inode| open.contains(&format!("socket:[{inode}]")))
}

// Stop what's left of the hidden desktop in `dir`, all three together, through pidfds for their
// records. A record goes only once its process is gone; one that outlives SIGKILL keeps its record
// and is named in the error. Once all three are gone, so are its apps' HOME and runtime folders,
// which live on the runtime dir's tmpfs and would otherwise keep their profiles and downloads in
// memory.
fn stop(dir: &Path) -> Result<(), String> {
    let mut left = Vec::new();
    for name in SERVICES {
        match recorded(dir, name) {
            Some((pid, pidfd)) if !end(&pidfd) => left.push(format!("{name} (pid {pid})")),
            _ => drop(std::fs::remove_file(dir.join(format!("{name}.pid")))),
        }
    }
    match left.is_empty() {
        true => {
            let _ = std::fs::remove_dir_all(dir.join("home"));
            let _ = std::fs::remove_dir_all(dir.join("runtime"));
            Ok(())
        }
        false => Err(format!("The hidden desktop's {} didn't exit, even on SIGKILL; set computer.display to another one.", left.join(" and "))),
    }
}

// The hidden desktop, started when it isn't running whole: a fresh auth cookie, the X server, then
// its two buses, each recorded. Blocking (it waits for them to come up). Its folder holds a lock so
// two sessions starting it at once start it once, and the users' claims (User). A display other
// than `:N` is refused, since another name for a display (`unix:0`, `localhost:0`) could be the
// user's.
fn hidden(settings: &Settings) -> Result<Desktop, String> {
    let error = |error: std::io::Error| error.to_string();
    let number = settings.display.strip_prefix(':').map(|rest| rest.split('.').next().unwrap_or(rest)).filter(|number| !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit()));
    let number = number.ok_or_else(|| format!("computer.display is {}; the hidden desktop's has to be a local display number, like {DEFAULT_DISPLAY}.", settings.display))?;
    let dir = settings.dir();
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&dir).map_err(error)?;
    let lock = std::fs::File::create(dir.join("lock")).map_err(error)?;
    // SAFETY: flock on descriptors this function and the session's User own; the start lock goes
    // when `lock` is dropped.
    unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&lock), libc::LOCK_EX) };
    let mut held = settings.held.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if held.closed {
        return Err("This session has closed; it doesn't use the desktop any more.".into());
    }
    if held.user.is_none() {
        let file = std::fs::File::create(dir.join("users")).map_err(error)?;
        unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&file), libc::LOCK_SH) };
        held.user = Some(User { file, dir: dir.clone() });
    }
    let desktop = Desktop { display: settings.display.clone(), dir: Some(dir.clone()) };
    // What's checked here is a moment: an Xvfb that exits after it, its display taken by another
    // server before the next xdotool or import connects by number, isn't caught. That window is
    // accepted; the signals, which could reach a process given a recorded pid, go through pidfds.
    let whole = recorded(&dir, "xvfb").is_some_and(|(pid, _)| holds(pid, number)) && recorded(&dir, "bus").is_some() && recorded(&dir, "a11y").is_some();
    if whole && desktop.up() {
        return Ok(desktop);
    }
    stop(&dir)?;
    // A failed start is rolled back through the pidfds from spawn, so a process whose record
    // couldn't be written still goes.
    let mut spawned = Vec::new();
    if let Err(error) = start(settings, &desktop, number, &mut spawned) {
        let left = spawned.iter().filter(|pidfd| !end(pidfd)).count();
        let _ = stop(&dir);
        return Err(match left {
            0 => error,
            _ => format!("{error}; and {left} of its processes didn't exit, even on SIGKILL"),
        });
    }
    Ok(desktop)
}

// Start the hidden desktop's three processes in its folder, recording each and keeping its pidfd
// in `spawned`. A display some other X server listens on is refused.
fn start(settings: &Settings, desktop: &Desktop, number: &str, spawned: &mut Vec<OwnedFd>) -> Result<(), String> {
    let dir = desktop.dir.clone().unwrap_or_default();
    let record = |name: &str, pid: u32| std::fs::write(dir.join(format!("{name}.pid")), format!("{pid} {}", started(pid).unwrap_or_default())).map_err(|error| error.to_string());
    if !listeners(number).is_empty() {
        return Err(format!("Another X server has {}, which piccosta didn't start, so it won't use it; set computer.display to a free one.", settings.display));
    }
    let auth = dir.join("Xauthority").to_string_lossy().into_owned();
    let cookie = uuid::Uuid::new_v4().simple().to_string();
    let added = desktop.command("xauth").args(["-f", &auth, "add", &settings.display, ".", &cookie]).output().map_err(|error| format!("xauth: {error}"))?;
    if !added.status.success() {
        return Err(format!("xauth: {}", String::from_utf8_lossy(&added.stderr).trim()));
    }
    let (pid, pidfd) = detach(desktop.command(&settings.xvfb).args([&settings.display, "-screen", "0", &settings.size, "-auth", &auth]), Some(dir.join("xvfb.log")))?;
    spawned.push(pidfd);
    record("xvfb", pid)?;
    if !wait_for(|| holds(pid, number) && desktop.up()) {
        return Err(format!("{} didn't start on {} (see {}); another X server may have it, in which case set computer.display to a free one.", settings.xvfb, settings.display, dir.join("xvfb.log").display()));
    }
    // Then its buses, which start with the desktop's DISPLAY, so what they start for an app (the
    // accessibility registry among them) belongs to this display. Sockets left by earlier ones go
    // first.
    for (name, config) in [("bus", "--session".to_string()), ("a11y", format!("--config-file={ACCESSIBILITY_CONFIG}"))] {
        let (socket, log) = (dir.join(name), dir.join(format!("{name}.log")));
        let _ = std::fs::remove_file(&socket);
        let address = format!("--address={}", desktop.bus(name).unwrap_or_default());
        let (pid, pidfd) = detach(desktop.command("dbus-daemon").args([&config, "--nofork", "--nopidfile", &address]), Some(log.clone()))?;
        spawned.push(pidfd);
        record(name, pid)?;
        if !wait_for(|| socket.exists()) {
            return Err(format!("dbus-daemon didn't start (see {})", log.display()));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct Arguments {
    action: String,
    coordinate: Option<[f64; 2]>,
    start_coordinate: Option<[f64; 2]>,
    text: Option<String>,
    button: Option<String>,
    scroll_direction: Option<String>,
    scroll_amount: Option<f64>,
    command: Option<Vec<String>>,
    desktop: Option<String>,
}

// The keys of a key action: space-separated key names or combinations of them ("ctrl+shift+t"),
// each name letters, digits and underscores. Each goes to xdotool as a `key` call of its own with
// that one argument: a word after the first one would be read as another xdotool command
// (`key Return exec sh` runs sh), and a lone word that's a command's name (`key exec`) starts
// that command with no arguments, which fails.
fn keys(text: &str) -> Result<Vec<String>, String> {
    let name = |name: &str| !name.is_empty() && name.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    let words: Vec<String> = text.split_whitespace().map(str::to_string).collect();
    match !words.is_empty() && words.iter().all(|word| word.split('+').all(name)) {
        true => Ok(words),
        false => Err(format!("key takes key names and combinations (\"Return\", \"ctrl+a\", \"alt+F4\"), not {text:?}")),
    }
}

pub struct ComputerTool;

impl Tool for ComputerTool {
    fn name(&self) -> &'static str {
        "computer"
    }

    fn definition(&self) -> ToolDefinition {
        let point = |what: &str| json!({"type": "array", "items": {"type": "number"}, "description": format!("{what}: [x, y] in the latest screenshot's pixels")});
        ToolDefinition {
            name: "computer".into(),
            description: "Use a computer: see its screen, move and click the mouse, type, press keys, scroll, read its accessibility tree and open apps. It's a hidden desktop of your own unless desktop is \"main\", the user's own screen, which the user has to allow on every call. Every action but tree returns a screenshot taken after it.".into(),
            parameters: json!({
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "enum": ["screenshot", "click", "double_click", "drag", "move", "type", "key", "scroll", "tree", "open"], "description": "screenshot; click or double_click (at coordinate, else where the pointer is); drag from start_coordinate to coordinate; move the pointer to coordinate; type text; key: press text's keys (xdotool names, space-separated: \"ctrl+a\", \"Return\", \"alt+F4\"); scroll at coordinate; tree: the accessibility tree as text (roles, names, sizes, states, text, and places in screenshot pixels where the app reports them); open: start command (a program and its arguments, not a shell line) as an app on the desktop"},
                    "coordinate": point("Where to act"),
                    "start_coordinate": point("Where a drag starts"),
                    "text": {"type": "string", "description": "What to type, or the keys to press"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "For click (default left)"},
                    "scroll_direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                    "scroll_amount": {"type": "integer", "description": "Scroll wheel clicks (default 3)"},
                    "command": {"type": "array", "items": {"type": "string"}, "description": "For open: the program, then its arguments, e.g. [\"zenity\", \"--entry\"]"},
                    "desktop": {"type": "string", "enum": ["hidden", "main"], "description": "hidden (default): your own desktop; main: the user's, only with their yes"}
                }
            }),
            constrained_sampling: None,
        }
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("Use a hidden desktop of your own: screenshots, mouse, keyboard, accessibility tree, apps")
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        &["Start apps for the computer with its open action. Only a visual session's bash runs on the hidden desktop; in any other session an app started from bash opens on the user's own screen."]
    }

    fn approval(&self, arguments: &Map<String, Value>) -> Option<&'static str> {
        (arguments.get("desktop").and_then(Value::as_str) == Some("main")).then_some(MAIN_QUESTION)
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: Arguments = serde_json::from_value(Value::Object(arguments)).map_err(|error| error.to_string())?;
            let desktop = match arguments.desktop.as_deref() {
                Some("main") => Desktop { display: std::env::var("DISPLAY").map_err(|_| "This session has no DISPLAY, so there's no X11 desktop of the user's to reach.")?, dir: None },
                _ => context.shared.computer.claim().await?,
            };
            tokio::select! {
                result = act(&desktop, &arguments, context) => result,
                () = context.cancel.cancelled() => Err("Operation aborted".to_string()),
            }
        })
    }
}

// Carry out one action on a desktop. Output: the screenshot after it (tree: the tree as text;
// open: the app's pid, then the screenshot). An app opened is the session's, stopped as it closes
// (bash.rs launch).
async fn act(desktop: &Desktop, arguments: &Arguments, context: &ToolContext) -> Result<ToolOutput, String> {
    // xdotool words for moving the pointer to a point given in screenshot pixels.
    let scale = SCALES.lock().unwrap().get(&desktop.display).copied().unwrap_or(1.0);
    let to = |point: [f64; 2]| vec!["mousemove".to_string(), ((point[0] * scale).round() as i64).to_string(), ((point[1] * scale).round() as i64).to_string()];
    let at = arguments.coordinate.map(to).unwrap_or_default();
    let need = |value: &Option<String>, what: &str| value.clone().ok_or_else(|| format!("{} needs {what}", arguments.action));
    let point = |value: Option<[f64; 2]>, what: &str| value.map(to).ok_or_else(|| format!("{} needs {what}", arguments.action));
    let words = |list: &[&str]| list.iter().map(|word| word.to_string()).collect::<Vec<String>>();
    let mut note = None;
    let xdotool: Vec<String> = match arguments.action.as_str() {
        "screenshot" => Vec::new(),
        "tree" => return Ok(ToolOutput { content: vec![Content::text(tree(desktop, scale).await?)], details: None, is_error: false }),
        "open" => {
            let command = arguments.command.clone().filter(|command| !command.is_empty()).ok_or("open needs a command")?;
            let log = desktop.dir.as_ref().map(|dir| dir.join("open.log"));
            let pid = super::bash::launch(&command, context, desktop, log.clone())?;
            let output = log.map_or(String::new(), |log| format!(", its output in {}", log.display()));
            note = Some(format!("Started {command:?} on {} (pid {pid}{output}).", desktop.display));
            Vec::new()
        }
        "click" => {
            let button = match arguments.button.as_deref() {
                Some("right") => "3",
                Some("middle") => "2",
                _ => "1",
            };
            [at, words(&["click", button])].concat()
        }
        "double_click" => [at, words(&["click", "--repeat", "2", "--delay", DOUBLE_CLICK_DELAY_MS, "1"])].concat(),
        "drag" => [point(arguments.start_coordinate, "start_coordinate")?, words(&["mousedown", "1"]), point(arguments.coordinate, "coordinate")?, words(&["mouseup", "1"])].concat(),
        "move" => point(arguments.coordinate, "coordinate")?,
        "type" => [words(&["type", "--delay", TYPING_DELAY_MS, "--"]), vec![need(&arguments.text, "text")?]].concat(),
        "key" => {
            for combination in keys(&need(&arguments.text, "text")?)? {
                desktop.run("xdotool", &[words(&["key", "--"]), vec![combination]].concat()).await?;
            }
            Vec::new()
        }
        "scroll" => {
            // X11's wheel is buttons 4 to 7.
            let button = match need(&arguments.scroll_direction, "scroll_direction")?.as_str() {
                "up" => "4",
                "down" => "5",
                "left" => "6",
                _ => "7",
            };
            let amount = arguments.scroll_amount.map_or(DEFAULT_SCROLL_AMOUNT, |amount| amount.max(1.0) as u32).to_string();
            [at, words(&["click", "--repeat", &amount, button])].concat()
        }
        other => return Err(format!("Unknown action {other}")),
    };
    if !xdotool.is_empty() {
        desktop.run("xdotool", &xdotool).await?;
    }
    match arguments.action.as_str() {
        "screenshot" => {}
        "open" => tokio::time::sleep(OPEN_SETTLE).await,
        _ => tokio::time::sleep(SETTLE).await,
    }
    let mut content: Vec<Content> = note.map(Content::text).into_iter().collect();
    content.extend(screenshot(desktop, context.model_accepts_images, context.image_limits).await?);
    Ok(ToolOutput { content, details: None, is_error: false })
}

// The whole screen, as image.rs prepares any image for a model, with a line saying its size. Its
// scale is kept for the next call's coordinates. A model that doesn't take images gets the line
// and a pointer to the tree.
async fn screenshot(desktop: &Desktop, accepts_images: bool, limits: image::Limits) -> Result<Vec<Content>, String> {
    let png = desktop.run("import", &["-window".into(), "root".into(), "png:-".into()]).await?;
    // A PNG's size is in its IHDR chunk, bytes 16 to 24.
    let size = |range: std::ops::Range<usize>| png.get(range).and_then(|bytes| bytes.try_into().ok()).map(u32::from_be_bytes).unwrap_or(0);
    let (width, height) = (size(16..20), size(20..24));
    let prepared = image::prepare(&png, "image/png", limits)?;
    SCALES.lock().unwrap().insert(desktop.display.clone(), width as f64 / prepared.width.max(1) as f64);
    let mut line = format!("Screenshot of {}, {}x{}", desktop.display, prepared.width, prepared.height);
    if prepared.width != width {
        line += &format!(" (the screen is {width}x{height}; coordinates stay in this screenshot's pixels)");
    }
    if !accepts_images {
        return Ok(vec![Content::text(line + ". This model doesn't take images: the tree action reads the screen as text.")]);
    }
    Ok(vec![Content::text(line + "."), Content::Image { data: prepared.data, mime_type: prepared.mime_type }])
}

// One AT-SPI call to a node (its bus name and path). Output: the reply, or None when it fails or
// the node doesn't have that interface.
async fn call<R>(bus: &zbus::Connection, node: &(String, OwnedObjectPath), interface: &str, method: &str, body: &(impl serde::Serialize + zbus::zvariant::DynamicType)) -> Option<R>
where
    R: serde::de::DeserializeOwned + zbus::zvariant::Type,
{
    let reply = bus.call_method(Some(node.0.as_str()), node.1.as_str(), Some(interface), method, body).await.ok()?;
    reply.body().deserialize().ok()
}

// The desktop's accessibility tree as text, one node a line, indented by depth: role, "name", its
// place (when known) and size in screenshot pixels, the states that matter, and its text. A node that isn't on
// screen is left out with everything under it (the root and the apps say nothing about that).
// Stops at TREE_MAX_NODES or TREE_DEADLINE, and says so.
async fn tree(desktop: &Desktop, scale: f64) -> Result<String, String> {
    let error = |error: zbus::Error| format!("The accessibility bus: {error}");
    // The hidden desktop's is its own; the user's session bus hands out theirs
    // (at-spi-bus-launcher's).
    let address = match desktop.bus("a11y") {
        Some(address) => address,
        None => {
            let session = zbus::Connection::session().await.map_err(error)?;
            let reply = session.call_method(Some("org.a11y.Bus"), "/org/a11y/bus", Some("org.a11y.Bus"), "GetAddress", &()).await.map_err(error)?;
            reply.body().deserialize().map_err(error)?
        }
    };
    let bus = zbus::connection::Builder::address(address.as_str()).map_err(error)?.method_timeout(TREE_CALL_WAIT).build().await.map_err(error)?;
    let root = OwnedObjectPath::try_from(REGISTRY_ROOT).map_err(|error| error.to_string())?;
    let mut stack = vec![((REGISTRY.to_string(), root), 0usize)];
    let (mut lines, start) = (Vec::new(), Instant::now());
    let accessible = "org.a11y.atspi.Accessible";
    while let Some((node, depth)) = stack.pop() {
        if lines.len() >= TREE_MAX_NODES || start.elapsed() > TREE_DEADLINE {
            lines.push(format!("(stopped at {} nodes, {:.0} s: more of the tree is left)", lines.len(), start.elapsed().as_secs_f64()));
            break;
        }
        let states: Vec<u32> = call(&bus, &node, accessible, "GetState", &()).await.unwrap_or_default();
        let has = |bit: u32| states.get((bit / 32) as usize).is_some_and(|word| word & (1 << (bit % 32)) != 0);
        if depth > 1 && !has(SHOWING) {
            continue;
        }
        let role: String = call(&bus, &node, accessible, "GetRoleName", &()).await.unwrap_or_default();
        let mut line = format!("{}{role}", "  ".repeat(depth));
        let name: Option<OwnedValue> = call(&bus, &node, "org.freedesktop.DBus.Properties", "Get", &(accessible, "Name")).await;
        if let Some(name) = name.and_then(|name| String::try_from(name).ok()).filter(|name| !name.is_empty()) {
            line += &format!(" {name:?}");
        }
        let interfaces: Vec<String> = call(&bus, &node, accessible, "GetInterfaces", &()).await.unwrap_or_default();
        let offers = |interface: &str| interfaces.iter().any(|offered| offered == interface);
        // Its size and place on the screen (coordinate type 0), in screenshot pixels like the
        // clicks. GTK 4 has no screen positions (it answers 0,0, and its window ones are off by
        // varying amounts), so a place of 0,0 is left out rather than trusted.
        if offers("org.a11y.atspi.Component")
            && let Some((x, y, width, height)) = call::<(i32, i32, i32, i32)>(&bus, &node, "org.a11y.atspi.Component", "GetExtents", &(0u32,)).await
            && width > 0
        {
            let pixels = |value: i32| (value as f64 / scale).round() as i64;
            if (x, y) != (0, 0) {
                line += &format!(" @{},{}", pixels(x), pixels(y));
            }
            line += &format!(" {}x{}", pixels(width), pixels(height));
        }
        let shown: Vec<&str> = STATE_NAMES.iter().filter(|(bit, _)| has(*bit)).map(|(_, name)| *name).collect();
        if !shown.is_empty() {
            line += &format!(" [{}]", shown.join(", "));
        }
        if offers("org.a11y.atspi.Text")
            && let Some(text) = call::<String>(&bus, &node, "org.a11y.atspi.Text", "GetText", &(0i32, TREE_TEXT_CHARS)).await
            && !text.trim().is_empty()
        {
            line += &format!(" = {text:?}");
        }
        lines.push(line);
        // Children go on the stack last first, so they come off in order.
        let children: Vec<(String, OwnedObjectPath)> = call(&bus, &node, accessible, "GetChildren", &()).await.unwrap_or_default();
        stack.extend(children.into_iter().rev().map(|child| (child, depth + 1)));
    }
    Ok(lines.join("\n"))
}
