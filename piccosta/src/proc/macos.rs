use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

// libproc's timestamps are epoch seconds and microseconds; unlike a pid they survive reuse.
pub fn start(pid: i32) -> Option<u64> {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of_val(&info) as i32;
    let got = unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, (&raw mut info).cast(), size) };
    (got == size && info.pbi_status != libc::SZOMB).then_some(info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec)
}

pub fn boot_id() -> String {
    let mut boot: libc::timeval = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of_val(&boot);
    let mut mib = [libc::CTL_KERN, libc::KERN_BOOTTIME];
    let got = unsafe { libc::sysctl(mib.as_mut_ptr(), mib.len() as u32, (&raw mut boot).cast(), &mut size, std::ptr::null_mut(), 0) };
    if got == 0 { format!("{}-{}", boot.tv_sec, boot.tv_usec) } else { String::new() }
}

// EVFILT_PROC holds a reference to this process, not whatever later takes its pid. Before a
// signal we also check its creation time; an exited child stays unreaped until bash takes it.
pub struct Process { pub fd: OwnedFd, pub pid: i32, start: u64 }
impl Process {
    pub fn open(pid: i32) -> Option<Self> {
        let start = start(pid)?;
        let fd = unsafe { libc::kqueue() };
        if fd < 0 { return None; }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC); }
        let event = libc::kevent { ident: pid as usize, filter: libc::EVFILT_PROC, flags: libc::EV_ADD | libc::EV_ONESHOT, fflags: libc::NOTE_EXIT, data: 0, udata: std::ptr::null_mut() };
        let got = unsafe { libc::kevent(fd.as_raw_fd(), &event, 1, std::ptr::null_mut(), 0, std::ptr::null()) };
        (got == 0 && self::start(pid) == Some(start)).then_some(Self { fd, pid, start })
    }
    pub fn started(pid: i32, start: u64) -> Option<Self> {
        Self::open(pid).filter(|process| process.start == start)
    }
    pub fn signal(&self, signal: i32) -> bool {
        self::start(self.pid) == Some(self.start) && unsafe { libc::kill(self.pid, signal) == 0 }
    }
}

pub fn children() -> Vec<i32> {
    let bytes = unsafe { libc::proc_listchildpids(std::process::id() as i32, std::ptr::null_mut(), 0) };
    if bytes <= 0 { return Vec::new(); }
    let mut pids = vec![0i32; bytes as usize / size_of::<i32>()];
    let got = unsafe { libc::proc_listchildpids(std::process::id() as i32, pids.as_mut_ptr().cast(), bytes) };
    pids.truncate(got.max(0) as usize / size_of::<i32>());
    pids
}
