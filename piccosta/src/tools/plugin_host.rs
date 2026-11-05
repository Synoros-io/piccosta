// A plugin's small parent: a watcher-owned cgroup with delegation, a child subreaper otherwise.
// It stays outside the plugin's read-only sandbox, so cleanup can still signal its descendants.
#[cfg(target_os = "linux")]
use std::sync::Arc;

pub struct Host {
    pub command: Option<std::process::Command>,
    #[cfg(target_os = "linux")]
    group: Option<Arc<Cgroup>>,
}

// Keep the spawn-time identity in every owner, including the session's shutdown snapshot.
// Windows keeps the Job Object itself: a later process with this pid cannot replace that handle.
#[derive(Clone)]
pub struct Owner {
    pub pid: u32,
    pub start: Option<u64>,
    #[cfg(target_os = "linux")]
    group: Option<Arc<Cgroup>>,
    #[cfg(windows)]
    job: crate::proc::JobHandle,
}
impl Host {
    pub fn attach(self, pid: u32) -> std::io::Result<Owner> {
        crate::proc::attach(pid)?;
        let owner = Owner { pid, start: crate::proc::start(pid as i32),
            #[cfg(target_os = "linux")]
            group: self.group,
            #[cfg(windows)]
            job: crate::proc::job(pid).ok_or_else(|| std::io::Error::other("plugin has no Job Object"))?,
        };
        #[cfg(unix)]
        super::reaper::track_plugin(pid, owner.start, true);
        Ok(owner)
    }
}
impl Owner {
    pub fn kill(&self) {
        #[cfg(target_os = "linux")]
        if let Some(group) = &self.group { group.kill(); }
        #[cfg(unix)]
        kill(self.pid, self.start);
        #[cfg(windows)]
        self.job.kill();
    }
}
impl Drop for Owner { fn drop(&mut self) { self.kill(); } }

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::path::PathBuf;
    pub const HOST_ENV: &str = "PICCOSTA_PLUGIN_HOST";
    const READ_ONLY_ENV: &str = "PICCOSTA_PLUGIN_READ_ONLY";
    const CGROUP_ENV: &str = "PICCOSTA_PLUGIN_CGROUP";
    const CGROUP_KILL_ENV: &str = "PICCOSTA_PLUGIN_CGROUP_KILL";
    // The kernel's cgroup v2 interface, not a deployment-specific directory.
    const CGROUP_ROOT: &str = "/sys/fs/cgroup";
    const NO_CONTAINMENT: &str = "plugins need a delegated cgroup or usable pidfd signalling; refusing to start";

    pub struct Cgroup { dir: PathBuf, kill: std::fs::File }
    impl Cgroup {
        pub fn kill(&self) { kill_cgroup(&self.kill); }
    }
    impl Drop for Cgroup {
        fn drop(&mut self) {
            self.kill();
            // An old open file cannot name a replacement cgroup at this path. Only remove the
            // directory if it still has that identity; the watcher owns its own descriptor.
            if remove_cgroup(&self.dir, &self.kill) { let _ = super::super::reaper::track_cgroup(&self.dir, &self.kill, false); }
        }
    }
    pub fn kill_cgroup(file: &std::fs::File) {
        use std::io::Write;
        let _ = (&*file).write_all(b"1");
    }
    pub fn remove_cgroup(path: &std::path::Path, file: &std::fs::File) -> bool {
        same_cgroup(path, file) && std::fs::remove_dir(path).is_ok()
    }
    fn same_cgroup(path: &std::path::Path, file: &std::fs::File) -> bool {
        use std::os::unix::fs::MetadataExt;
        file.metadata().ok().zip(std::fs::metadata(path.join("cgroup.kill")).ok()).is_some_and(|(old, now)| old.dev() == now.dev() && old.ino() == now.ino())
    }

    // Register the original kill descriptor before the supervisor can start any descendants.
    // The watcher owns that same kernel object even if its path is removed and reused.
    fn cgroup() -> Option<Arc<Cgroup>> {
        let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
        let parent = text.lines().find_map(|line| line.strip_prefix("0::"))?;
        let dir = PathBuf::from(CGROUP_ROOT).join(parent.trim_start_matches('/')).join(format!("piccosta-plugin-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).ok()?;
        // Check both interfaces before choosing cgroups over pidfds; roll back a failed start.
        let opened = (|| {
            std::fs::OpenOptions::new().write(true).open(dir.join("cgroup.procs"))?;
            std::fs::OpenOptions::new().write(true).open(dir.join("cgroup.kill"))
        })();
        let kill = match opened { Ok(kill) => kill, Err(_) => { let _ = std::fs::remove_dir(&dir); return None; } };
        let group = Arc::new(Cgroup { dir, kill });
        super::super::reaper::track_cgroup(&group.dir, &group.kill, true).ok()?;
        Some(group)
    }

    fn can_signal() -> bool { crate::proc::Process::open(std::process::id() as i32).is_some_and(|process| process.signal(0)) }

    pub fn command(path: &std::path::Path, read_only: bool) -> std::io::Result<Host> {
        let group = cgroup();
        if group.is_none() && !can_signal() { return Err(std::io::Error::other(NO_CONTAINMENT)); }
        let mut command = std::process::Command::new(crate::proc::exe()?);
        command.env(HOST_ENV, path).env(READ_ONLY_ENV, if read_only { "1" } else { "0" }).env_remove(CGROUP_ENV).env_remove(CGROUP_KILL_ENV);
        use std::os::fd::AsRawFd;
        if let Some(group) = &group { command.env(CGROUP_ENV, &group.dir).env(CGROUP_KILL_ENV, group.kill.as_raw_fd().to_string()); }
        let inherited_group = group.clone();
        let parent = std::process::id();
        use std::os::unix::process::CommandExt;
        // SAFETY: raw syscalls only. TERM lets an initialized supervisor clean its children.
        unsafe { command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 { return Err(std::io::Error::last_os_error()); }
            if libc::getppid() as u32 != parent { return Err(std::io::Error::other("piccosta ended before plugin startup")); }
            if let Some(group) = &inherited_group && libc::fcntl(group.kill.as_raw_fd(), libc::F_SETFD, 0) < 0 { return Err(std::io::Error::last_os_error()); }
            Ok(())
        }); }
        Ok(Host { command: Some(command), group })
    }

    pub fn run() -> ! {
        // SAFETY: before threads or children. sigwait cannot lose a close between checking a
        // child's status and waiting for its next signal.
        let mut signals: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            if libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0 { std::process::exit(1); }
            libc::sigemptyset(&mut signals);
            for signal in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP, libc::SIGCHLD] { libc::sigaddset(&mut signals, signal); }
            if libc::sigprocmask(libc::SIG_BLOCK, &signals, std::ptr::null_mut()) != 0 { std::process::exit(1); }
        }
        let group = std::env::var_os(CGROUP_ENV).map(PathBuf::from);
        let kill = std::env::var(CGROUP_KILL_ENV).ok().and_then(|fd| fd.parse().ok()).map(|fd| std::fs::File::from(super::super::reaper::inherited(fd)));
        if let Some(dir) = &group {
            if !kill.as_ref().is_some_and(|kill| same_cgroup(dir, kill)) || std::fs::write(dir.join("cgroup.procs"), std::process::id().to_string()).is_err() { std::process::exit(1); }
        } else if !can_signal() { eprintln!("{NO_CONTAINMENT}"); std::process::exit(1); }
        let path = std::env::var_os(HOST_ENV).expect("plugin host path");
        let read_only = std::env::var(READ_ONLY_ENV).as_deref() == Ok("1");
        let mut command = std::process::Command::new(path);
        command.args(std::env::args_os().skip(1)).env_remove(HOST_ENV).env_remove(READ_ONLY_ENV).env_remove(CGROUP_ENV).env_remove(CGROUP_KILL_ENV);
        use std::os::unix::process::{CommandExt, ExitStatusExt};
        // SAFETY: raw syscalls only between fork and exec. Only the plugin enters Landlock.
        unsafe { command.pre_exec(move || {
            if libc::sigprocmask(libc::SIG_UNBLOCK, &signals, std::ptr::null_mut()) != 0 { return Err(std::io::Error::last_os_error()); }
            if read_only { super::super::readonly::enter()?; }
            Ok(())
        }); }
        let child = command.spawn();
        let pid = child.as_ref().map(|child| child.id() as i32).unwrap_or(-1);
        let mut status = 0;
        if pid > 0 {
            loop {
                // SAFETY: never block in waitpid, including when signalling becomes unavailable.
                if unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) } != 0 { break; }
                let mut signal = 0;
                if unsafe { libc::sigwait(&signals, &mut signal) } != 0 || signal != libc::SIGCHLD { break; }
            }
        }
        // Leave the group before killing descendants so a successful describe stays successful.
        // If we're killed first, the harness and watcher still own the group's cleanup.
        if let Some(dir) = group {
            if let Some(parent) = dir.parent() { let _ = std::fs::write(parent.join("cgroup.procs"), std::process::id().to_string()); }
            if let Some(kill) = &kill { kill_cgroup(kill); }
        }
        // A child stuck in I/O can orphan another generation much later. Stay its subreaper
        // until ECHILD, however long that takes. Session shutdown bounds its own wait, not ours.
        loop {
            for pid in crate::proc::children() {
                if let Some(child) = crate::proc::Process::open(pid) { child.signal(libc::SIGKILL); }
            }
            let mut ignored = 0;
            let ended = unsafe { libc::waitpid(-1, &mut ignored, libc::WNOHANG) };
            if ended > 0 { continue; }
            if ended < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD) { break; }
            if ended < 0 { continue; }
            let mut signal = 0;
            unsafe { libc::sigwait(&signals, &mut signal); }
        }
        std::process::exit(if pid < 0 { 1 } else { crate::proc::exit_code(std::process::ExitStatus::from_raw(status)) });
    }
}
#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(not(target_os = "linux"))]
pub fn command(path: &std::path::Path, _read_only: bool) -> std::io::Result<Host> {
    #[cfg(target_os = "macos")]
    {
        static NOTICE: std::sync::Once = std::sync::Once::new();
        NOTICE.call_once(|| eprintln!("piccosta: macOS plugins have process-group containment only; helpers using setsid can outlive the session"));
    }
    Ok(Host { command: Some(std::process::Command::new(path)) })
}

#[cfg(unix)]
pub fn kill(pid: u32, start: Option<u64>) {
    #[cfg(target_os = "linux")]
    if let Some(process) = start.and_then(|start| crate::proc::Process::started(pid as i32, start)) { process.signal(crate::proc::TERM); }
    #[cfg(not(target_os = "linux"))]
    if crate::proc::group_held(pid, start) { crate::proc::kill_group(pid, crate::proc::KILL); }
}
