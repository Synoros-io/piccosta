// The reaper: a small second piccosta that outlives this one only to stop its commands.
//
// piccosta stops the shells it started when it quits or gets a signal it can catch. SIGKILL (how
// Zed stops an agent) can't be caught, so those shells and everything under them would run on with
// nobody watching. So the first command a piccosta runs also starts this watcher: the same binary,
// started with REAPER_ENV set, in a session of its own so a closing terminal's signals don't reach
// it. piccosta tells it each shell's process group as the shell starts ("+pid") and as it ends
// ("-pid") through a pipe. When the pipe closes, piccosta is gone, however it went, and the
// watcher stops every group still listed, the way bash.rs stops one, then exits.
//
// /restart execs a new image in the same process. The pipe goes across the exec (hand_over,
// adopt), so the watcher never sees it close, keeps its list, and goes on covering the jobs the
// new image takes back.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

// Set in the watcher's environment; main() checks it before anything else.
pub const REAPER_ENV: &str = "PICCOSTA_REAPER";
// How long the groups get after SIGTERM before SIGKILL: bash.rs's stop grace for subagents, which
// write their last messages then.
const STOP_GRACE: Duration = super::bash::SUBAGENT_STOP_GRACE;

// The pipe to the watcher and the watcher's pid, once it's started; None until the first command,
// or when it couldn't start (then commands run as they did without it).
static PIPE: Mutex<Option<Option<(File, u32)>>> = Mutex::new(None);

// Tell the watcher a shell's group started (`running`) or ended. Starts the watcher the first time.
pub fn track(pid: u32, running: bool) {
    let mut pipe = PIPE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((stdin, _)) = pipe.get_or_insert_with(start) {
        let _ = writeln!(stdin, "{}{pid}", if running { '+' } else { '-' });
    }
}

// /restart is about to exec: keep the pipe open across it. Output: the pipe's descriptor and the
// watcher's pid, for the next image's adopt; None when no watcher was started.
pub fn hand_over() -> Option<(i32, u32)> {
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

// A descriptor the image before this one kept across the exec, owned here from now on and closed
// on the next exec again, so the commands this image starts don't inherit it.
pub fn inherited(fd: i32) -> OwnedFd {
    // SAFETY: the previous image left it open for this one, and nothing else here owns it.
    unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
    unsafe { OwnedFd::from_raw_fd(fd) }
}

// Start the watcher: this binary again, with the pipe as its stdin and nothing else open.
fn start() -> Option<(File, u32)> {
    use std::os::unix::process::CommandExt;
    let exe = std::env::current_exe().ok()?;
    let mut command = Command::new(exe);
    command.env(REAPER_ENV, "1").stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
    // SAFETY: setsid() is async-signal-safe and runs in the child between fork and exec.
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let mut child = command.spawn().ok()?;
    Some((File::from(OwnedFd::from(child.stdin.take()?)), child.id()))
}

// The watcher's whole life: keep the list until the pipe closes, then stop what's on it.
pub fn run() {
    let mut groups: HashSet<i32> = HashSet::new();
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let (sign, pid) = line.split_at(line.len().min(1));
        if let Ok(pid) = pid.parse::<i32>() {
            if sign == "+" { groups.insert(pid) } else { groups.remove(&pid) };
        }
    }
    if groups.is_empty() {
        return;
    }
    // SAFETY: killpg() takes plain integers; a group already gone fails with ESRCH.
    let signal_all = |signal| groups.iter().for_each(|&group| unsafe { libc::killpg(group, signal); });
    signal_all(libc::SIGTERM);
    std::thread::sleep(STOP_GRACE);
    signal_all(libc::SIGKILL);
}
