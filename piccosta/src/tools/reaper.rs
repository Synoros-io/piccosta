// The watcher: a small second piccosta that outlives this one only to stop its commands.
//
// A runtime stops its commands when it ends (jobs.rs shutdown), but a piccosta killed with SIGKILL
// (how Zed stops an agent, and how a manager ends a runner that doesn't answer, runtime.rs end)
// ends no runtime, and the commands would run on with nobody watching. So the first command a
// piccosta runs also starts this watcher: the same binary, started with REAPER_ENV set, in a session
// of its own so a closing terminal's signals don't reach it. The runtimes' job registry tells it
// each shell's process group as the shell starts ("+pid") and as it's reaped ("-pid") through a
// pipe (bash.rs). When the pipe closes, piccosta is gone, however it went, and the watcher stops
// every group still listed, the way bash.rs stops one, then exits. One per piccosta process: a lane
// runs in a piccosta of its own (runtime.rs launch), with a watcher of its own.
//
// /restart execs a new image in the same process. The pipe goes across the exec (hand_over,
// adopt), so the watcher never sees it close, keeps its list, and goes on covering the jobs the
// new image takes back.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
#[cfg(not(target_os = "linux"))]
use std::io::BufRead;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::process::{Command, Stdio};
use std::sync::Mutex;

// Set in the watcher's environment; main() checks it before anything else.
pub const REAPER_ENV: &str = "PICCOSTA_REAPER";

// The pipe to the watcher and the watcher's pid, once it's started; None until the first command,
// or when it couldn't start (then commands run as they would without it).
static PIPE: Mutex<Option<Option<(File, u32)>>> = Mutex::new(None);

// Tell the watcher a shell's group started (`running`) or ended. Starts the watcher the first time.
pub fn track(pid: u32, running: bool) { record(pid, crate::proc::start(pid as i32), running, false); }
pub fn track_plugin(pid: u32, start: Option<u64>, running: bool) { record(pid, start, running, true); }

#[cfg(target_os = "linux")]
pub fn track_cgroup(path: &std::path::Path, file: &File, running: bool) -> std::io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let mut pipe = PIPE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (stdin, _) = pipe.get_or_insert_with(start).as_mut().ok_or_else(|| std::io::Error::other("plugin watcher could not start"))?;
    let id = file.metadata()?.ino();
    let line = format!("{}{id} {}\n", if running { 'c' } else { 'd' }, serde_json::to_string(path)?);
    let mut data = libc::iovec { iov_base: line.as_ptr().cast_mut().cast(), iov_len: line.len() };
    // SCM_RIGHTS transfers the original open cgroup.kill, not a path the watcher must reopen.
    let mut control = [0usize; CONTROL_WORDS];
    // SAFETY: aligned control storage, one valid descriptor, and a live message buffer.
    unsafe {
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &mut data;
        message.msg_iovlen = 1;
        if running {
            message.msg_control = control.as_mut_ptr().cast();
            message.msg_controllen = std::mem::size_of_val(&control) as _;
            let header = libc::CMSG_FIRSTHDR(&message);
            (*header).cmsg_level = libc::SOL_SOCKET;
            (*header).cmsg_type = libc::SCM_RIGHTS;
            (*header).cmsg_len = libc::CMSG_LEN(size_of::<i32>() as _) as _;
            std::ptr::write(libc::CMSG_DATA(header).cast::<i32>(), file.as_raw_fd());
        }
        let count = libc::sendmsg(stdin.as_raw_fd(), &message, libc::MSG_NOSIGNAL);
        if count < 0 { return Err(std::io::Error::last_os_error()); }
        stdin.write_all(&line.as_bytes()[count as usize..])?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
const CONTROL_WORDS: usize = (unsafe { libc::CMSG_SPACE(size_of::<i32>() as _) } as usize).div_ceil(size_of::<usize>());

// Read through recvmsg, not buffered stdin: buffering could discard SCM_RIGHTS. Each record is
// short and local; one byte at a time keeps descriptor ownership attached to its exact line.
#[cfg(target_os = "linux")]
fn records() -> impl Iterator<Item = (String, Option<File>)> {
    std::iter::from_fn(|| {
        let (mut bytes, mut file) = (Vec::new(), None);
        loop {
            let mut byte = 0u8;
            let mut control = [0usize; CONTROL_WORDS];
            let mut data = libc::iovec { iov_base: (&mut byte as *mut u8).cast(), iov_len: 1 };
            // SAFETY: live data/aligned control buffers. SCM_RIGHTS gives us an owned descriptor.
            unsafe {
                let mut message: libc::msghdr = std::mem::zeroed();
                message.msg_iov = &mut data;
                message.msg_iovlen = 1;
                message.msg_control = control.as_mut_ptr().cast();
                message.msg_controllen = std::mem::size_of_val(&control) as _;
                let count = libc::recvmsg(0, &mut message, libc::MSG_CMSG_CLOEXEC);
                if count < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted { continue; }
                if count <= 0 { return None; }
                let header = libc::CMSG_FIRSTHDR(&message);
                if !header.is_null() && (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                    file = Some(File::from_raw_fd(std::ptr::read(libc::CMSG_DATA(header).cast::<i32>())));
                }
            }
            if byte == b'\n' { return Some((String::from_utf8_lossy(&bytes).into_owned(), file)); }
            bytes.push(byte);
        }
    })
}

fn record(pid: u32, identity: Option<u64>, running: bool, plugin: bool) {
    let mut pipe = PIPE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((stdin, _)) = pipe.get_or_insert_with(start) {
        // A group joins with its leader's start time (proc::group_held), taken while it runs.
        let start = if running { identity.map_or(String::new(), |start| format!(" {start}")) } else { String::new() };
        let _ = stdin.write_all(format!("{}{pid}{start}\n", if !running { '-' } else if plugin { 'p' } else { '+' }).as_bytes());
    }
}

// /restart is about to exec: keep the pipe open across it, and every descriptor the image before
// left that nobody here took back (from a handover that couldn't be read), so what waits at the
// other end (a watcher, a job's shell) never sees it close. Output: the pipe's descriptor and the
// watcher's pid, for the next image's adopt; None when no watcher was started.
pub fn hand_over() -> Option<(i32, u32)> {
    INHERITED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).0.iter().for_each(|&fd| keep_across_exec(fd));
    let pipe = PIPE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (stdin, pid) = pipe.as_ref()?.as_ref()?;
    keep_across_exec(stdin.as_raw_fd());
    Some((stdin.as_raw_fd(), *pid))
}

// The next image takes the watcher back: the same pipe, the same list of groups.
pub fn adopt(fd: i32, pid: u32) {
    *PIPE.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Some((File::from(inherited(fd)), pid)));
}

// Let a descriptor go across the exec that /restart makes; every other one closes there.
pub fn keep_across_exec(fd: i32) {
    // SAFETY: fcntl on a descriptor this process holds open.
    unsafe { libc::fcntl(fd, libc::F_SETFD, 0) };
}

// What an image /restart started inherited, as main found it before anything could spawn: the
// descriptors open past stdin, stdout and stderr (the ones not yet claimed or closed), and the children.
static INHERITED: Mutex<(Vec<i32>, Vec<i32>)> = Mutex::new((Vec::new(), Vec::new()));

// First thing in an image /restart started: each descriptor it inherited closes on exec from now
// on, so no command started here gets one, and they and the children still running are listed.
pub fn inherit() {
    let listed: Vec<i32> = std::fs::read_dir("/dev/fd").into_iter().flatten().flatten().filter_map(|entry| entry.file_name().to_str()?.parse().ok()).collect();
    // SAFETY: fcntl on numbers that may not be open: the listing's own, closed by now, fails.
    let open = listed.into_iter().filter(|&fd| fd > 2 && unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } == 0).collect();
    *INHERITED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = (open, crate::proc::children());
}

// A descriptor the image before this one kept across the exec, claimed and owned here from now on,
// and closed on exec (again, should inherit have found no listing).
pub fn inherited(fd: i32) -> OwnedFd {
    INHERITED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).0.retain(|&open| open != fd);
    // SAFETY: the image before left it open for this one, and nothing else here owns it.
    unsafe {
        libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
        OwnedFd::from_raw_fd(fd)
    }
}

// The children the image before left, once (tui reap_inherited_children).
pub fn inherited_children() -> Vec<i32> { std::mem::take(&mut INHERITED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).1) }

// Reap a child the image before left that nobody here took back, so it's no zombie when it ends.
// One to end gets SIGTERM, then SIGKILL after `ending` (nothing, if it's gone: only this reaps it,
// so its pid is its own till then). Then the `pipes` (a Claude Code's) nobody claimed close.
pub fn reap_inherited(pid: i32, ending: Option<std::time::Duration>, pipes: Vec<i32>) {
    let listed = INHERITED.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).0.clone();
    let pipes: Vec<OwnedFd> = pipes.into_iter().filter(|fd| listed.contains(fd)).map(inherited).collect();
    // SAFETY: signals and waitpid on this process's own unreaped child.
    std::thread::spawn(move || unsafe {
        if let Some(grace) = ending { libc::kill(pid, libc::SIGTERM); std::thread::sleep(grace); libc::kill(pid, libc::SIGKILL); }
        libc::waitpid(pid, std::ptr::null_mut(), 0);
        drop(pipes);
    });
}

// Start the watcher: this binary again, with the pipe as its stdin and nothing else open.
fn start() -> Option<(File, u32)> {
    let mut command = Command::new(crate::proc::exe().ok()?);
    command.env(REAPER_ENV, "1").stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(target_os = "linux")]
    {
        let (writer, reader) = std::os::unix::net::UnixStream::pair().ok()?;
        command.stdin(Stdio::from(OwnedFd::from(reader)));
        let child = crate::proc::own_session(&mut command).spawn().ok()?;
        Some((File::from(OwnedFd::from(writer)), child.id()))
    }
    #[cfg(not(target_os = "linux"))]
    {
        command.stdin(Stdio::piped());
        let mut child = crate::proc::own_session(&mut command).spawn().ok()?;
        Some((File::from(OwnedFd::from(child.stdin.take()?)), child.id()))
    }
}

// The watcher's whole life: keep the list until the pipe closes, then stop what's on it, SIGTERM
// first, then SIGKILL after bash's stop grace (subagents write their last messages then).
pub fn run() {
    let mut groups: HashMap<u32, (Option<u64>, bool)> = HashMap::new();
    #[cfg(target_os = "linux")]
    let mut cgroups: HashMap<u64, (std::path::PathBuf, File)> = HashMap::new();
    #[cfg(target_os = "linux")]
    let lines = records();
    #[cfg(not(target_os = "linux"))]
    let lines = std::io::stdin().lock().lines().map_while(Result::ok).map(|line| (line, None::<File>));
    for (line, _file) in lines {
        let (sign, rest) = line.split_at(line.len().min(1));
        #[cfg(target_os = "linux")]
        if sign == "c" || sign == "d" {
            if let Some((id, path)) = rest.split_once(' ') && let Ok(id) = id.parse::<u64>() {
                if sign == "c" {
                    if let (Some(file), Ok(path)) = (_file, serde_json::from_str(path)) { cgroups.insert(id, (path, file)); }
                } else { cgroups.remove(&id); }
            }
            continue;
        }
        let mut fields = rest.split_whitespace();
        if let Some(Ok(pid)) = fields.next().map(str::parse::<u32>) {
            if sign == "+" || sign == "p" { groups.insert(pid, (fields.next().and_then(|start| start.parse().ok()), sign == "p")) } else { groups.remove(&pid) };
        }
    }
    // A supervisor may have died first. Its cgroup still belongs to us, not to that pid.
    #[cfg(target_os = "linux")]
    for (_, file) in cgroups.values() { super::plugin_host::kill_cgroup(file); }
    #[cfg(target_os = "linux")]
    let empty = groups.is_empty() && cgroups.is_empty();
    #[cfg(not(target_os = "linux"))]
    let empty = groups.is_empty();
    if empty { return; }
    // Each group only while it's still the one registered (proc::group_held).
    for (&pid, &(start, plugin)) in &groups { if plugin { super::plugin_host::kill(pid, start); } }
    let signal_all = |signal| groups.iter().filter(|&(&group, &(start, plugin))| !plugin && crate::proc::group_held(group, start)).for_each(|(&group, _)| crate::proc::kill_group(group, signal));
    signal_all(libc::SIGTERM);
    std::thread::sleep(super::bash::SUBAGENT_STOP_GRACE);
    signal_all(libc::SIGKILL);
    #[cfg(target_os = "linux")]
    for (_, (path, file)) in cgroups { super::plugin_host::remove_cgroup(&path, &file); }
}
