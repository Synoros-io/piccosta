use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Instant;
pub type Stream = UnixStream;
pub type Listener = tokio::net::UnixListener;

// Linux can name the long session path through an open directory. macOS cannot, so its socket
// lives in a configurable short runtime directory, keyed by the absolute session path.
fn address(file: &Path) -> std::io::Result<(Option<std::fs::File>, PathBuf)> {
    #[cfg(target_os = "linux")]
    {
        let dir = std::fs::File::open(file.parent().unwrap_or(Path::new("/")))?;
        let socket = crate::runtime::socket_path(file);
        let name = socket.file_name().unwrap_or_default().to_string_lossy();
        let address = PathBuf::from(format!("/proc/self/fd/{}/{name}", dir.as_raw_fd()));
        Ok((Some(dir), address))
    }
    #[cfg(target_os = "macos")]
    {
        use sha2::Digest;
        let dir = std::env::var_os("PICCOSTA_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(|| crate::config::data_home().join("piccosta/run"));
        let parent = std::fs::canonicalize(file.parent().unwrap_or(Path::new(".")))?;
        let path = parent.join(file.file_name().unwrap_or_default());
        let digest = sha2::Sha256::digest(path.as_os_str().as_encoded_bytes());
        let name: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
        Ok((None, dir.join(format!("{name}.sock"))))
    }
}
pub fn remove(file: &Path) {
    if let Ok((_dir, address)) = address(file) { let _ = std::fs::remove_file(address); }
}
pub fn bind(file: &Path) -> std::io::Result<Listener> {
    let (_dir, address) = address(file)?;
    #[cfg(target_os = "macos")]
    {
        let dir = address.parent().unwrap();
        crate::platform::private_dir(dir)?;
        crate::platform::permissions(dir, 0o700)?;
    }
    let listener = match UnixListener::bind(&address) {
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => {
            if crate::runtime::running(file) { return Err(error); }
            let _ = std::fs::remove_file(&address);
            UnixListener::bind(&address)
        }
        bound => bound,
    }?;
    crate::platform::permissions(&address, 0o600)?;
    listener.set_nonblocking(true)?;
    Listener::from_std(listener)
}
pub fn connect(file: &Path) -> std::io::Result<Stream> {
    let (_dir, address) = address(file)?;
    let socket = socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None)?;
    socket.set_nonblocking(true)?;
    socket.connect(&socket2::SockAddr::unix(address)?)?;
    socket.set_nonblocking(false)?;
    Ok(socket.into())
}
pub async fn connect_async(file: &Path) -> std::io::Result<tokio::net::UnixStream> {
    let (_dir, address) = address(file)?;
    tokio::net::UnixStream::connect(address).await
}
pub async fn accept(listener: &Listener) -> std::io::Result<tokio::net::UnixStream> {
    loop {
        let (stream, _) = listener.accept().await?;
        // File permissions are the first boundary; peer credentials also reject another uid
        // even if it could bypass those permissions (or they were changed after binding).
        if stream.peer_cred()?.uid() == unsafe { libc::geteuid() } { return Ok(stream); }
    }
}
pub fn peer(stream: &Stream) -> Option<i32> {
    #[cfg(target_os = "linux")]
    {
        let mut credentials = libc::ucred { pid: 0, uid: 0, gid: 0 };
        let mut length = size_of_val(&credentials) as libc::socklen_t;
        let got = unsafe { libc::getsockopt(stream.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, (&raw mut credentials).cast(), &mut length) };
        (got == 0).then_some(credentials.pid)
    }
    #[cfg(target_os = "macos")]
    {
        let mut pid: libc::pid_t = 0;
        let mut length = size_of_val(&pid) as libc::socklen_t;
        let got = unsafe { libc::getsockopt(stream.as_raw_fd(), libc::SOL_LOCAL, libc::LOCAL_PEERPID, (&raw mut pid).cast(), &mut length) };
        (got == 0).then_some(pid)
    }
}
pub struct Timed<'a>(pub &'a Stream, pub Instant);
impl Timed<'_> {
    fn left(&self) -> std::io::Result<Option<std::time::Duration>> {
        Some(self.1.saturating_duration_since(Instant::now())).filter(|left| !left.is_zero()).map(Some).ok_or(std::io::ErrorKind::TimedOut.into())
    }
}
impl std::io::Read for Timed<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> { self.0.set_read_timeout(self.left()?)?; std::io::Read::read(&mut self.0, bytes) }
}
impl std::io::Write for Timed<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> { self.0.set_write_timeout(self.left()?)?; std::io::Write::write(&mut self.0, bytes) }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}
