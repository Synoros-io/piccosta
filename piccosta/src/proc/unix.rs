use std::os::unix::process::{CommandExt, ExitStatusExt};
pub use libc::{SIGKILL as KILL, SIGTERM as TERM};
pub use tokio::net::unix::pipe::Receiver;

pub fn own_session(command: &mut std::process::Command) -> &mut std::process::Command {
    // SAFETY: setsid is async-signal-safe between fork and exec.
    unsafe { command.pre_exec(|| { libc::setsid(); Ok(()) }) }
}

pub fn signal_group(pid: u32, signal: i32) {
    let Ok(pid) = i32::try_from(pid) else { return };
    // SAFETY: the shell leads its group; if that group is gone, try the unreaped child alone.
    unsafe { if libc::kill(-pid, signal) != 0 { libc::kill(pid, signal); } }
}

// The group alone, never the pid: for a child that may already have been reaped, whose number may
// be someone else's by now.
pub fn kill_group(pid: u32, signal: i32) {
    let Ok(pid) = i32::try_from(pid) else { return };
    // SAFETY: kill() takes plain integers; a gone group fails with ESRCH.
    unsafe { libc::kill(-pid, signal) };
}

// Whether the process group `pid` led, which started at `start`, is still that group: its leader
// is still that process (a zombie leader, not yet reaped, still holds the number too), or no
// process holds the number and the group has members, which only its own can be then. A number
// another process has taken since is someone else's, never to be signalled.
pub fn group_held(pid: u32, start: Option<u64>) -> bool {
    let Ok(group) = i32::try_from(pid) else { return false };
    match crate::proc::start(group) {
        Some(now) => Some(now) == start,
        // SAFETY: kill() with signal 0 only checks that the group has a member.
        None => unsafe { libc::kill(-group, 0) == 0 },
    }
}

pub fn exit_code(status: std::process::ExitStatus) -> i32 {
    status.code().unwrap_or_else(|| status.signal().map_or(1, |signal| 128 + signal))
}

pub fn resume(command: &mut std::process::Command) -> std::io::Error { command.exec() }

pub fn link(exe: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(exe, link)
}

pub fn own_group(command: &mut std::process::Command) -> &mut std::process::Command { command.process_group(0) }
pub fn group(command: &mut tokio::process::Command) { command.process_group(0); }
pub fn attach(_pid: u32) -> std::io::Result<()> { Ok(()) }
pub fn release(_pid: u32) {}

pub fn pipe(reader: std::io::PipeReader) -> std::io::Result<Receiver> {
    Receiver::from_owned_fd(std::os::fd::OwnedFd::from(reader))
}

pub async fn read_pipe(pipe: &Receiver, buffer: &mut [u8]) -> std::io::Result<usize> {
    loop {
        pipe.readable().await?;
        match pipe.try_read(buffer) {
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {},
            result => return result,
        }
    }
}

pub fn shell_setup(command: &mut std::process::Command, read_only: bool) {
    // SAFETY: these are only raw async-signal-safe calls, after fork and before exec.
    unsafe { command.pre_exec(move || {
        libc::setsid();
        if read_only { crate::tools::readonly::enter()?; }
        Ok(())
    }); }
}

pub fn parent_death(command: &mut tokio::process::Command) {
    #[cfg(target_os = "linux")]
    {
        let parent = std::process::id();
        // SAFETY: raw system calls only in the forked child.
        unsafe { command.pre_exec(move || {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            if libc::getppid() as u32 != parent { return Err(std::io::Error::other("piccosta ended")); }
            Ok(())
        }); }
    }
    #[cfg(target_os = "macos")]
    group(command);
}
