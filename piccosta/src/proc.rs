// Processes: knowing one for sure, and starting one apart from this one.
//
// A pid names a process only while it lives: once it's reaped the number passes on. So a process
// found by its number (a record on disk, a list in /proc) is opened as a pidfd, which names that
// one process for as long as it's held, and checked against the start time it was recorded with. A
// start time counts from boot, so a record carries the boot it was made in too. Every piccosta
// process that has to find or stop another goes through here: the lanes' run records
// (manager.rs), the Claude bridge's children, bash's shells, the web tools' clone dirs.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::PathBuf;

// The kernel's id for this boot (random(4)).
const BOOT_ID_FILE: &str = "/proc/sys/kernel/random/boot_id";
// Where /proc/<pid>/stat keeps the parent (field 4), the session (field 6) and the start time
// (field 22, clock ticks since boot), counted among the fields after the name's ')', which start
// at field 3, the state (proc_pid_stat(5)).
const STAT_PARENT: usize = 4 - 3;
const STAT_SESSION: usize = 6 - 3;
const STAT_START: usize = 22 - 3;

pub fn boot_id() -> String {
    std::fs::read_to_string(BOOT_ID_FILE).unwrap_or_default().trim().to_string()
}

// A live process as its stat has it. None once it's gone, or a zombie, which is as good as gone.
#[derive(Clone, Copy)]
pub struct Stat {
    pub parent: i32,
    pub session: i32,
    pub start: u64,
}

pub fn stat(pid: i32) -> Option<Stat> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields: Vec<&str> = text.rsplit_once(')')?.1.split_whitespace().collect();
    (*fields.first()? != "Z").then_some(())?;
    let number = |index: usize| fields.get(index)?.parse().ok();
    Some(Stat { parent: number(STAT_PARENT)?, session: number(STAT_SESSION)?, start: fields.get(STAT_START)?.parse().ok()? })
}

// One process, known for sure: a pidfd, and its pid for reading /proc.
pub struct Process {
    pub fd: OwnedFd,
    pub pid: i32,
}

impl Process {
    // The process `pid` names now; None when there's none.
    pub fn open(pid: i32) -> Option<Process> {
        // SAFETY: pidfd_open takes a pid and flags and returns a new descriptor (close-on-exec) or -1.
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) } as i32;
        // SAFETY: that descriptor, just made and owned from here on.
        (fd >= 0).then(|| Process { fd: unsafe { OwnedFd::from_raw_fd(fd) }, pid })
    }

    // Its stat now, None once it's gone. Read before the pidfd shows it alive: a pid passes on only
    // once its process is reaped, so a stat read while it lives is its own.
    pub fn stat(&self) -> Option<Stat> {
        stat(self.pid).filter(|_| self.signal(0))
    }

    pub fn signal(&self, signal: i32) -> bool {
        // SAFETY: pidfd_send_signal on a descriptor this owns, with no siginfo.
        unsafe { libc::syscall(libc::SYS_pidfd_send_signal, self.fd.as_raw_fd(), signal, std::ptr::null::<libc::siginfo_t>(), 0) == 0 }
    }

    // The process `pid` names now, if it's the one that started at `start`, with its stat now.
    pub fn started(pid: i32, start: u64) -> Option<(Process, Stat)> {
        let process = Process::open(pid)?;
        let now = process.stat().filter(|now| now.start == start)?;
        Some((process, now))
    }

    // What a later process finds a live one by, with its pid: "<boot id> <start time>".
    pub fn record(pid: i32) -> Option<String> {
        Some(format!("{} {}", boot_id(), stat(pid)?.start))
    }
}

// The start time a record (Process::record) names, None when it was made in another boot, where
// its pid could be anything's now.
pub fn recorded_start(record: &str) -> Option<u64> {
    let (boot, start) = record.trim().split_once(' ')?;
    (!boot.is_empty() && boot == boot_id()).then(|| start.parse().ok())?
}

// This process's children, from each of its threads' lists (a child is listed under the thread
// that started it).
pub fn children() -> Vec<i32> {
    let tasks = std::fs::read_dir("/proc/self/task").into_iter().flatten().flatten();
    let lists = tasks.filter_map(|task| std::fs::read_to_string(task.path().join("children")).ok());
    lists.flat_map(|list| list.split_whitespace().filter_map(|pid| pid.parse().ok()).collect::<Vec<i32>>()).collect()
}

// A command started apart from this process: in a session of its own, so a closing terminal's
// signals don't reach it and its process group can be signalled as one.
pub fn own_session(command: &mut std::process::Command) -> &mut std::process::Command {
    use std::os::unix::process::CommandExt;
    // SAFETY: setsid() is async-signal-safe and runs in the child between fork and exec.
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        })
    }
}

// This binary, to start again: once a new build replaces it on disk, /proc/self/exe reads
// "<path> (deleted)", and the path is the new build.
pub fn exe() -> std::io::Result<PathBuf> {
    Ok(PathBuf::from(std::env::current_exe()?.to_string_lossy().trim_end_matches(" (deleted)")))
}
