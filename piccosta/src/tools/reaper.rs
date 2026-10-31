// The reaper: a small second piccosta that outlives this one only to stop its commands.
//
// piccosta stops the shells it started when it quits or gets a signal it can catch. SIGKILL (how
// Zed stops an agent) can't be caught, so those shells and everything under them would run on with
// nobody watching. So the first command a piccosta runs also starts this watcher: the same binary,
// started with REAPER_ENV set, in a session of its own so a closing terminal's signals don't reach
// it. piccosta tells it each shell's process group as the shell starts ("+pid") and as it ends
// ("-pid") through a pipe. When the pipe closes, piccosta is gone, however it went, and the
// watcher stops every group still listed, the way bash.rs stops one, then exits.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

// Set in the watcher's environment; main() checks it before anything else.
pub const REAPER_ENV: &str = "PICCOSTA_REAPER";
// How long the groups get after SIGTERM before SIGKILL: bash.rs's stop grace for subagents, which
// write their last messages then.
const STOP_GRACE: Duration = super::bash::SUBAGENT_STOP_GRACE;

// The pipe to the watcher, once it's started; None until the first command, or when it couldn't
// start (then commands run as they did without it).
static PIPE: Mutex<Option<Option<ChildStdin>>> = Mutex::new(None);

// Tell the watcher a shell's group started (`running`) or ended. Starts the watcher the first time.
pub fn track(pid: u32, running: bool) {
    let mut pipe = PIPE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let stdin = pipe.get_or_insert_with(start);
    if let Some(stdin) = stdin {
        let _ = writeln!(stdin, "{}{pid}", if running { '+' } else { '-' });
    }
}

// Start the watcher: this binary again, with the pipe as its stdin and nothing else open.
fn start() -> Option<ChildStdin> {
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
    command.spawn().ok()?.stdin.take()
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
