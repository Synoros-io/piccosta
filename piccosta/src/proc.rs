// Processes: knowing one for sure, and starting one apart from this one.
//
// A pid names a process only while it lives: once it's reaped the number passes on. So a process
// found by its number (a record on disk, a socket's peer) is checked against the start time it was
// found with, through a pidfd, which names that one process for as long as it's held. A start time
// counts from boot, so a record carries the boot it was made in too: the web tools' clone dirs keep
// their owner's. bash waits for its shells through a pidfd too, and a manager kills a session's
// runner that doesn't answer through one (runtime.rs end).

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::PathBuf;

// The kernel's id for this boot (random(4)).
const BOOT_ID_FILE: &str = "/proc/sys/kernel/random/boot_id";
// Where /proc/<pid>/stat keeps the start time (field 22, clock ticks since boot), counted among the
// fields after the name's ')', which start at field 3, the state (proc_pid_stat(5)).
const STAT_START: usize = 22 - 3;

pub fn boot_id() -> String {
    std::fs::read_to_string(BOOT_ID_FILE).unwrap_or_default().trim().to_string()
}

// When a live process started, in clock ticks since boot. None once it's gone, or a zombie, which
// is as good as gone.
pub fn start(pid: i32) -> Option<u64> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields: Vec<&str> = text.rsplit_once(')')?.1.split_whitespace().collect();
    (*fields.first()? != "Z").then_some(())?;
    fields.get(STAT_START)?.parse().ok()
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

    // The process `pid` names now, if it's the one that started at `start`. Its start time is read
    // before the pidfd shows it alive: a pid passes on only once its process is reaped, so a time
    // read while it lives is its own.
    pub fn started(pid: i32, start: u64) -> Option<Process> {
        let process = Process::open(pid)?;
        (self::start(pid) == Some(start) && process.signal(0)).then_some(process)
    }

    // Send it a signal (0 only checks it's alive). Output: whether it went.
    pub fn signal(&self, signal: i32) -> bool {
        // SAFETY: pidfd_send_signal on a descriptor this owns, with no siginfo.
        unsafe { libc::syscall(libc::SYS_pidfd_send_signal, self.fd.as_raw_fd(), signal, std::ptr::null::<libc::siginfo_t>(), 0) == 0 }
    }
}

// The start time a record ("<boot id> <start time>") names, None when it was made in another
// boot, where its pid could be anything's now.
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
