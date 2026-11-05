// Apps see an allowlisted tmpfs root, not the host tree with a few holes covered. All mounts
// and SysV objects belong to private namespaces and disappear with their last process.
use std::ffi::{CStr, CString};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

const X11_DIR: &CStr = c"/tmp/.X11-unix";
// Landlock ABI 6: filesystem rights through IOCTL_DEV, and abstract-socket/signal scopes.
const FS_ALL: u64 = (1 << 16) - 1;
const FS_READ: u64 = (1 << 0) | (1 << 2) | (1 << 3);
const FS_FILE: u64 = FS_ALL & !((1 << 3) | (0x3ff << 4));
const SCOPE_ABSTRACT_UNIX_SOCKET: u64 = 1;
const SCOPE_SIGNAL: u64 = 1 << 1;
const CAP_VERSION: u32 = 0x2008_0522;
const NOROOT_LOCKED: libc::c_ulong = 1 | 2;
// X11's TCP base, plus the traditional first 64 displays and the actual host display.
const X_TCP_BASE: u16 = 6000;
const X_DISPLAY_COUNT: u16 = 64;
// Loaders, executables, fonts and icons, and from /etc only what apps read to run: names, name
// resolution, certificates, the loader's cache, time zone, MIME types and toolkit settings. /etc as
// a whole and /opt aren't mounted: a socket or a readable secret in them would be reachable.
const SYSTEM: &[&str] = &["/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/var/cache/fontconfig", "/var/cache/font-manager", "/var/cache/icons",
    // Public certificates only: never /etc/ssl/private or /etc/pki/tls/private, where private keys
    // live. openSUSE's /etc/ssl/certs links into /var/lib/ca-certificates.
    "/etc/fonts", "/etc/ssl/certs", "/etc/ssl/cert.pem", "/etc/ssl/ca-bundle.pem", "/etc/ssl/openssl.cnf", "/etc/ca-certificates",
    "/var/lib/ca-certificates", "/etc/pki/tls/certs", "/etc/pki/tls/cert.pem", "/etc/pki/tls/openssl.cnf", "/etc/pki/ca-trust", "/etc/passwd", "/etc/group", "/etc/hosts", "/etc/host.conf", "/etc/nsswitch.conf", "/etc/gai.conf",
    "/etc/resolv.conf", "/etc/services", "/etc/protocols", "/etc/localtime", "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/mime.types",
    "/etc/xdg", "/etc/gtk-2.0", "/etc/gtk-3.0", "/etc/gtk-4.0", "/etc/pango", "/etc/X11/xkb", "/etc/machine-id", "/etc/os-release", "/etc/alternatives", "/etc/firefox"];
const DEVICES: &[&str] = &["/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty"];
const SANDBOX: &str = "--piccosta-app-sandbox";
// close_range(2)'s flag that marks the range close-on-exec instead of closing it (linux/close_range.h).
const CLOSE_RANGE_CLOEXEC: libc::c_uint = 1 << 2;

pub const FALLBACK_QUESTION: &str = "Private app sandbox isolation is unavailable on this machine. Let the agent start an unrestricted app? It can reach your own desktop and files too.";
pub static AVAILABLE: LazyLock<bool> = LazyLock::new(|| probe().is_ok());

pub fn fallback_notice() -> &'static str {
    static NOTICE: std::sync::Once = std::sync::Once::new();
    let mut first = false;
    if !*AVAILABLE { NOTICE.call_once(|| first = true); }
    if first { "Private app sandbox isolation is unavailable; this app is unrestricted, with your approval.\n" } else { "" }
}
fn checked(result: libc::c_long) -> io::Result<libc::c_long> {
    if result < 0 { Err(io::Error::last_os_error()) } else { Ok(result) }
}
fn path(path: &Path) -> io::Result<CString> {
    CString::new(path.as_os_str().as_bytes()).map_err(io::Error::other)
}
struct User { uid: CString, gid: CString }
impl User {
    fn new() -> Self {
        let (uid, gid) = unsafe { (libc::getuid(), libc::getgid()) };
        Self { uid: CString::new(format!("{uid} {uid} 1\n")).unwrap(), gid: CString::new(format!("{gid} {gid} 1\n")).unwrap() }
    }
    fn enter(&self) -> io::Result<()> {
        unsafe {
            checked(libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNS | libc::CLONE_NEWIPC) as _)?;
            for (file, value) in [(c"/proc/self/setgroups", c"deny"), (c"/proc/self/uid_map", self.uid.as_c_str()), (c"/proc/self/gid_map", self.gid.as_c_str())] {
                let fd = checked(libc::open(file.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC) as _)? as i32;
                let written = libc::write(fd, value.as_ptr().cast(), value.to_bytes().len());
                let error = io::Error::last_os_error();
                libc::close(fd);
                if written != value.to_bytes().len() as isize { return Err(error); }
            }
            checked(libc::mount(std::ptr::null(), c"/".as_ptr(), std::ptr::null(), libc::MS_REC | libc::MS_PRIVATE, std::ptr::null()) as _)?;
        }
        Ok(())
    }
}
fn tmpfs(target: &CStr) -> io::Result<()> {
    unsafe { checked(libc::mount(c"tmpfs".as_ptr(), target.as_ptr(), c"tmpfs".as_ptr(), libc::MS_NOSUID | libc::MS_NODEV, c"mode=0700".as_ptr().cast()) as _)?; }
    Ok(())
}
fn mkdir(target: &CStr) -> io::Result<()> {
    if unsafe { libc::mkdir(target.as_ptr(), 0o700) } != 0 && io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST) { return Err(io::Error::last_os_error()); }
    Ok(())
}

// Sources are pinned before covering /tmp, then closed at exec. Never leave the app an fd
// into the discarded root. Reopen after entering the user namespace, for procfd permission.
struct Bind { fd: OwnedFd, source: CString, original: CString, target: CString }
impl Bind {
    fn new(target: &Path) -> io::Result<Self> {
        let target = path(target)?;
        let fd = unsafe { checked(libc::open(target.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) as _)? as i32 };
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let source = CString::new(format!("/proc/self/fd/{}", fd.as_raw_fd())).unwrap();
        Ok(Self { fd, source, original: target.clone(), target })
    }
    fn reopen(&self) -> io::Result<()> {
        unsafe {
            let fd = checked(libc::open(self.original.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) as _)? as i32;
            let result = checked(libc::dup3(fd, self.fd.as_raw_fd(), libc::O_CLOEXEC) as _);
            libc::close(fd);
            result?;
        }
        Ok(())
    }
    fn mount(&self) -> io::Result<()> {
        unsafe { checked(libc::mount(self.source.as_ptr(), self.target.as_ptr(), std::ptr::null(), libc::MS_BIND, std::ptr::null()) as _)?; }
        Ok(())
    }
}
#[repr(C)]
struct Ruleset { fs: u64, net: u64, scoped: u64 }
#[repr(C, packed)]
struct Beneath { access: u64, fd: i32 }
#[repr(C)]
struct CapHeader { version: u32, pid: i32 }
fn restrict(rules: &[(CString, u64)]) -> io::Result<()> {
    let attr = Ruleset { fs: FS_ALL, net: 0, scoped: SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL };
    let header = CapHeader { version: CAP_VERSION, pid: 0 };
    let caps = [0u32; 6];
    unsafe {
        let fd = checked(libc::syscall(libc::SYS_landlock_create_ruleset, &attr, size_of::<Ruleset>(), 0))? as i32;
        let result = (|| {
            for (target, access) in rules {
                let file = checked(libc::open(target.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) as _)? as i32;
                let rule = Beneath { access: *access, fd: file };
                let result = checked(libc::syscall(libc::SYS_landlock_add_rule, fd, 1, &rule, 0));
                libc::close(file);
                result?;
            }
            checked(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) as _)?;
            checked(libc::syscall(libc::SYS_landlock_restrict_self, fd, 0))?;
            checked(libc::prctl(libc::PR_SET_SECUREBITS, NOROOT_LOCKED, 0, 0, 0) as _)?;
            checked(libc::syscall(libc::SYS_capset, &header, caps.as_ptr()))?;
            Ok(())
        })();
        libc::close(fd);
        result
    }
}
fn host_safe(runtime: &Path) -> io::Result<()> {
    if !std::fs::canonicalize(runtime)?.is_dir() { return Err(io::Error::other("cannot identify host runtime directory")); }
    let display = std::env::var("PICCOSTA_HOST_DISPLAY").or_else(|_| std::env::var("DISPLAY")).unwrap_or_default();
    let host_port = display.rsplit_once(':').and_then(|(_, number)| number.split('.').next()?.parse::<u16>().ok()).and_then(|number| X_TCP_BASE.checked_add(number));
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        for line in std::fs::read_to_string(table)?.lines().skip(1) {
            let fields: Vec<_> = line.split_whitespace().collect();
            let port = fields.get(1).and_then(|addr| addr.rsplit_once(':')).and_then(|(_, port)| u16::from_str_radix(port, 16).ok());
            if fields.get(3) == Some(&"0A") && port.is_some_and(|port| (X_TCP_BASE..X_TCP_BASE + X_DISPLAY_COUNT).contains(&port) || Some(port) == host_port) {
                return Err(io::Error::other("host TCP X11 listener prevents app isolation"));
            }
        }
    }
    Ok(())
}

// The complete mount plan is prepared before fork. Targets are under /tmp until pivot_root;
// no recursive bind can accidentally bring a host submount along for the ride.
struct Sandbox {
    user: User, binds: Vec<(Bind, bool, bool)>, folders: Vec<CString>, files: Vec<CString>,
    links: Vec<(CString, CString)>, rules: Vec<(CString, u64)>,
}
impl Sandbox {
    fn new(dir: Option<&Path>, socket: Option<&str>, runtime: &Path) -> io::Result<Self> {
        host_safe(runtime)?;
        let mut sandbox = Self { user: User::new(), binds: Vec::new(), folders: Vec::new(), files: Vec::new(), links: Vec::new(), rules: Vec::new() };
        for system in SYSTEM {
            let target = Path::new(system);
            if !target.try_exists()? { continue; }
            if let Ok(link) = std::fs::read_link(target) {
                sandbox.links.push((path(&link)?, path(&Path::new("/tmp").join(target.strip_prefix("/").unwrap()))?));
            } else { sandbox.bind(target, target, true, false)?; }
            // A file (or a link to one) can't take Landlock's directory-only right to read entries.
            sandbox.rules.push((path(target)?, if target.is_dir() { FS_READ } else { FS_READ & !(1 << 3) }));
        }
        // resolv.conf is often a link into /run. Copy only its target, never the host /run.
        if let Ok(target) = std::fs::canonicalize("/etc/resolv.conf") && !SYSTEM.iter().any(|root| target.starts_with(root)) {
            sandbox.bind(&target, &target, true, false)?;
            sandbox.rules.push((path(&target)?, FS_READ & !(1 << 3)));
        }
        for device in DEVICES {
            sandbox.bind(Path::new(device), Path::new(device), false, true)?;
            sandbox.rules.push((CString::new(*device).unwrap(), FS_FILE));
        }
        for folder in ["/tmp", "/tmp/.X11-unix", "/dev/shm", "/proc", "/.old"] { sandbox.folder(Path::new(folder))?; }
        sandbox.rules.extend([(CString::new("/tmp").unwrap(), FS_ALL), (CString::new("/dev/shm").unwrap(), FS_ALL), (CString::new("/proc").unwrap(), FS_READ | (1 << 1))]);
        if let Some(dir) = dir {
            for name in ["home", "runtime"] {
                let target = dir.join(name);
                sandbox.bind(&target, &target, false, false)?;
            }
            let auth = dir.join("Xauthority");
            sandbox.bind(&auth, &auth, true, false)?;
            if let Some(number) = socket {
                sandbox.bind(&dir.join("x11").join(format!("X{number}")), &Path::new("/tmp/.X11-unix").join(format!("X{number}")), true, false)?;
            }
        }
        sandbox.folders.sort_by_key(|folder| folder.to_bytes().len());
        sandbox.folders.dedup();
        Ok(sandbox)
    }
    fn folder(&mut self, target: &Path) -> io::Result<()> {
        for parent in target.ancestors().filter(|parent| parent.parent().is_some()) {
            self.folders.push(path(&Path::new("/tmp").join(parent.strip_prefix("/").unwrap()))?);
        }
        Ok(())
    }
    fn bind(&mut self, source: &Path, target: &Path, readonly: bool, device: bool) -> io::Result<()> {
        let directory = source.is_dir();
        self.folder(if directory { target } else { target.parent().unwrap() })?;
        let mut bind = Bind::new(source)?;
        bind.target = path(&Path::new("/tmp").join(target.strip_prefix("/").unwrap()))?;
        if !directory { self.files.push(bind.target.clone()); }
        self.binds.push((bind, readonly, device));
        Ok(())
    }
    fn mounts(&self) -> io::Result<()> {
        for (bind, _, _) in &self.binds { bind.reopen()?; }
        tmpfs(c"/tmp")?;
        for folder in &self.folders { mkdir(folder)?; }
        for file in &self.files {
            let fd = checked(unsafe { libc::open(file.as_ptr(), libc::O_CREAT | libc::O_EXCL | libc::O_WRONLY | libc::O_CLOEXEC, 0o600) } as _)? as i32;
            unsafe { libc::close(fd); }
        }
        for (link, target) in &self.links { checked(unsafe { libc::symlink(link.as_ptr(), target.as_ptr()) } as _)?; }
        for (bind, readonly, device) in &self.binds {
            bind.mount()?;
            let flags = libc::MS_BIND | libc::MS_REMOUNT | libc::MS_NOSUID | if *device { 0 } else { libc::MS_NODEV } | if *readonly { libc::MS_RDONLY } else { 0 };
            checked(unsafe { libc::mount(std::ptr::null(), bind.target.as_ptr(), std::ptr::null(), flags, std::ptr::null()) } as _)?;
        }
        unsafe {
            // Mount while the inherited procfs is still attached: Linux rejects a new procfs
            // after the last old one is detached (the mount-too-revealing check).
            checked(libc::mount(c"proc".as_ptr(), c"/tmp/proc".as_ptr(), c"proc".as_ptr(), libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC, c"subset=pid".as_ptr().cast()) as _)?;
            checked(libc::chdir(c"/tmp".as_ptr()) as _)?;
            checked(libc::syscall(libc::SYS_pivot_root, c".".as_ptr(), c".old".as_ptr()))?;
            checked(libc::chdir(c"/".as_ptr()) as _)?;
            checked(libc::umount2(c"/.old".as_ptr(), libc::MNT_DETACH) as _)?;
            checked(libc::rmdir(c"/.old".as_ptr()) as _)?;
        }
        restrict(&self.rules)
    }
}
fn wait(pid: i32) -> io::Result<i32> {
    let mut status = 0;
    loop {
        if unsafe { libc::waitpid(pid, &mut status, 0) } >= 0 { return Ok(status); }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted { return Err(error); }
    }
}
fn runtime() -> io::Result<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).ok_or_else(|| io::Error::other("XDG_RUNTIME_DIR is unset"))
}
fn probe() -> io::Result<()> {
    let sandbox = Sandbox::new(None, None, &runtime()?)?;
    unsafe {
        let pid = checked(libc::fork() as _)? as i32;
        if pid == 0 {
            let ok = (|| {
                sandbox.user.enter()?;
                checked(libc::unshare(libc::CLONE_NEWPID) as _)?;
                let child = checked(libc::fork() as _)? as i32;
                if child == 0 {
                    let result = sandbox.mounts();
                    if let Err(error) = &result { eprintln!("Allowlisted app root probe: {error}"); }
                    libc::_exit(if result.is_ok() { 0 } else { 1 });
                }
                if wait(child)? != 0 { libc::_exit(1); }
                Ok::<(), io::Error>(())
            })().is_ok();
            libc::_exit(if ok { 0 } else { 1 });
        }
        if wait(pid)? == 0 { Ok(()) } else { Err(io::Error::other("allowlisted app root unavailable")) }
    }
}

pub fn isolate(command: &mut Command, display: &str, dir: &Path, cwd: &Path) -> io::Result<Startup> {
    let runtime = runtime()?;
    if !cwd.is_absolute() { return Err(io::Error::other("sandbox cwd must be absolute")); }
    let mut fds = [-1; 2];
    checked(unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } as _)?;
    let startup = Startup { read: unsafe { OwnedFd::from_raw_fd(fds[0]) }, write: unsafe { OwnedFd::from_raw_fd(fds[1]) } };
    let status = fds[1];
    let mut wrapped = Command::new("/proc/self/exe");
    wrapped.args([std::ffi::OsStr::new(SANDBOX), display.as_ref(), dir.as_os_str(), cwd.as_os_str(), runtime.as_os_str(), std::ffi::OsStr::new(&status.to_string()), command.get_program()]);
    wrapped.args(command.get_args()).stdin(std::process::Stdio::null()).current_dir("/");
    for (key, value) in command.get_envs() { if let Some(value) = value { wrapped.env(key, value); } else { wrapped.env_remove(key); } }
    wrapped.env("PICCOSTA_HOST_DISPLAY", std::env::var_os("DISPLAY").unwrap_or_default());
    unsafe { wrapped.pre_exec(move || { checked(libc::fcntl(status, libc::F_SETFD, 0) as _)?; Ok(()) }); }
    *command = wrapped;
    Ok(startup)
}
pub struct Startup { read: OwnedFd, write: OwnedFd }
impl Startup {
    pub fn check(self, wait: std::time::Duration) -> io::Result<()> {
        drop(self.write);
        let mut ready = libc::pollfd { fd: self.read.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        if unsafe { libc::poll(&mut ready, 1, wait.as_millis() as i32) } <= 0 { return Err(io::Error::other("app sandbox startup timed out")); }
        let mut bytes = [0; 4096];
        let count = checked(unsafe { libc::read(self.read.as_raw_fd(), bytes.as_mut_ptr().cast(), bytes.len()) } as _)? as usize;
        if count == 0 { Ok(()) } else { Err(io::Error::other(String::from_utf8_lossy(&bytes[..count]).into_owned())) }
    }
}
// Before Tokio starts threads: fork a PID namespace init, then discard the old root entirely.
pub fn run() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_none_or(|arg| arg != SANDBOX) { return Ok(()); }
    let status = args.get(5).and_then(|fd| fd.to_str()).and_then(|fd| fd.parse::<i32>().ok()).ok_or_else(|| io::Error::other("missing app sandbox status pipe"))?;
    checked(unsafe { libc::fcntl(status, libc::F_SETFD, libc::FD_CLOEXEC) } as _)?;
    let result: io::Result<()> = (|| {
        if args.len() < 7 { return Err(io::Error::other("incomplete app sandbox command")); }
        let (display, dir, cwd, runtime) = (&args[1], Path::new(&args[2]), Path::new(&args[3]), Path::new(&args[4]));
        let number = display.to_str().unwrap_or_default().trim_start_matches(':').split('.').next().unwrap_or_default();
        let sandbox = Sandbox::new(Some(dir), Some(number), runtime)?;
        sandbox.user.enter()?;
        checked(unsafe { libc::unshare(libc::CLONE_NEWPID) } as _)?;
        let parent_fd = unsafe { OwnedFd::from_raw_fd(checked(libc::syscall(libc::SYS_pidfd_open, libc::getpid(), 0))? as i32) };
        let pid = checked(unsafe { libc::fork() } as _)? as i32;
        if pid > 0 { unsafe { libc::close(status); } std::process::exit(piccosta::proc::exit_code(std::process::ExitStatus::from_raw(wait(pid)?))); }
        checked(unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } as _)?;
        // getppid is zero across this PID boundary. A pinned pidfd closes the death-signal race.
        let mut ready = libc::pollfd { fd: parent_fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        if unsafe { libc::poll(&mut ready, 1, 0) } != 0 { return Err(io::Error::other("app supervisor ended during startup")); }
        sandbox.mounts()?;
        let reachable = |dir: &CStr| unsafe { libc::chdir(dir.as_ptr()) } == 0 && {
            let fd = unsafe { libc::open(dir.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC) };
            fd >= 0 && unsafe { libc::close(fd) } == 0
        };
        if !reachable(&path(cwd)?) && !reachable(&path(&dir.join("home"))?) { return Err(io::Error::other("app private home is unreachable")); }
        drop(parent_fd);
        // A descriptor piccosta inherited without close-on-exec (an open file, a connected socket)
        // would cross into the app past the new root and Landlock: every one above stderr closes
        // at exec. The status pipe is already close-on-exec, which is how the parent sees success.
        checked(unsafe { libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, CLOSE_RANGE_CLOEXEC) })?;
        Err(Command::new(&args[6]).args(&args[7..]).exec())
    })();
    let error = format!("Allowlisted app root refused exec: {}", result.unwrap_err());
    unsafe { libc::write(status, error.as_ptr().cast(), error.len()); }
    eprintln!("{error}");
    std::process::exit(1);
}

// Do this independently of the app probe: TCP X, missing runtime or missing Landlock must not
// put desktop profiles back on disk. If even a private tmpfs is impossible, don't start one.
pub fn desktop() -> io::Result<()> {
    User::new().enter()?;
    tmpfs(c"/tmp")
}
pub fn display_view(command: &mut Command, dir: &Path) -> io::Result<()> {
    let user = User::new();
    let mut sockets = Bind::new(&dir.join("x11"))?;
    let desktop = Bind::new(dir)?;
    sockets.target = CString::from(X11_DIR);
    unsafe { command.pre_exec(move || {
        user.enter()?;
        sockets.reopen()?;
        desktop.reopen()?;
        tmpfs(c"/tmp")?;
        mkdir(&desktop.target)?;
        desktop.mount()?;
        mkdir(X11_DIR)?;
        sockets.mount()
    }); }
    Ok(())
}
