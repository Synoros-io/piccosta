use std::io::{Read, Write};
use std::os::windows::io::AsRawHandle;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::net::windows::named_pipe::{NamedPipeServer, NamedPipeClient, ServerOptions, ClientOptions};
use windows_sys::Win32::{Foundation::*, Security::{Authorization::*, *}, System::Pipes::*};

pub type Stream = std::fs::File;
pub struct Listener { name: String, server: tokio::sync::Mutex<NamedPipeServer> }
// NOWAIT pipes report temporarily empty/full without blocking a synchronous caller. This is
// only the polling interval within runtime's existing deadline, not another delivery timeout.
const PIPE_POLL: Duration = Duration::from_millis(2);
fn name(file: &Path) -> std::io::Result<String> {
    use sha2::Digest;
    let parent = std::fs::canonicalize(file.parent().unwrap_or(Path::new(".")))?;
    let path = parent.join(file.file_name().unwrap_or_default());
    let hash: String = sha2::Sha256::digest(path.as_os_str().as_encoded_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    Ok(format!(r"\\.\pipe\piccosta-{hash}"))
}
fn server(name: &str, first: bool) -> std::io::Result<NamedPipeServer> {
    // Only the owner may connect; reject_remote_clients also rules out the network pipe path.
    let sddl: Vec<u16> = "D:P(A;;GA;;;OW)".encode_utf16().chain(Some(0)).collect();
    let mut descriptor = std::ptr::null_mut();
    if unsafe { ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.as_ptr(), SDDL_REVISION_1, &mut descriptor, std::ptr::null_mut()) } == 0 { return Err(std::io::Error::last_os_error()); }
    let mut security = SECURITY_ATTRIBUTES { nLength: size_of::<SECURITY_ATTRIBUTES>() as u32, lpSecurityDescriptor: descriptor, bInheritHandle: 0 };
    let result = unsafe { ServerOptions::new().first_pipe_instance(first).reject_remote_clients(true).create_with_security_attributes_raw(name, (&raw mut security).cast()) };
    unsafe { LocalFree(descriptor); }
    result
}
pub fn bind(file: &Path) -> std::io::Result<Listener> {
    let name = name(file)?;
    let server = server(&name, true).map_err(|error| if error.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32) { std::io::ErrorKind::AddrInUse.into() } else { error })?;
    Ok(Listener { name, server: tokio::sync::Mutex::new(server) })
}
pub fn remove(_file: &Path) {}
pub async fn accept(listener: &Listener) -> std::io::Result<NamedPipeServer> {
    let mut current = listener.server.lock().await;
    loop {
        match current.connect().await {
            Ok(()) => return Ok(std::mem::replace(&mut *current, server(&listener.name, false)?)),
            // A synchronous running() probe may disconnect before ConnectNamedPipe observes it.
            // That instance is spent, not the listener: replace it and keep taking requests.
            Err(error) if matches!(error.raw_os_error(), Some(code) if code == ERROR_NO_DATA as i32 || code == ERROR_PIPE_NOT_CONNECTED as i32 || code == ERROR_BROKEN_PIPE as i32) => {
                *current = server(&listener.name, false)?;
            }
            Err(error) => return Err(error),
        }
    }
}
pub fn connect(file: &Path) -> std::io::Result<Stream> {
    let pipe = std::fs::OpenOptions::new().read(true).write(true).open(name(file)?).map_err(|error| if error.raw_os_error() == Some(ERROR_PIPE_BUSY as i32) { std::io::ErrorKind::WouldBlock.into() } else { error })?;
    let mode = PIPE_READMODE_BYTE | PIPE_NOWAIT;
    if unsafe { SetNamedPipeHandleState(pipe.as_raw_handle(), &mode, std::ptr::null(), std::ptr::null()) } == 0 { return Err(std::io::Error::last_os_error()); }
    Ok(pipe)
}
pub async fn connect_async(file: &Path) -> std::io::Result<NamedPipeClient> { ClientOptions::new().open(name(file)?) }
pub fn client_peer(stream: &NamedPipeServer) -> Option<i32> {
    let mut pid = 0;
    (unsafe { GetNamedPipeClientProcessId(stream.as_raw_handle(), &mut pid) } != 0).then_some(pid as i32)
}
pub fn peer(stream: &Stream) -> Option<i32> {
    let mut pid = 0;
    (unsafe { GetNamedPipeServerProcessId(stream.as_raw_handle(), &mut pid) } != 0).then_some(pid as i32)
}
pub struct Timed<'a>(pub &'a Stream, pub Instant);
impl Timed<'_> {
    fn retry<T>(&self, mut call: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
        loop {
            if Instant::now() >= self.1 { return Err(std::io::ErrorKind::TimedOut.into()); }
            match call() {
                Err(error) if matches!(error.raw_os_error(), Some(code) if code == ERROR_NO_DATA as i32 || code == ERROR_PIPE_BUSY as i32) => std::thread::sleep(PIPE_POLL.min(self.1.saturating_duration_since(Instant::now()))),
                result => return result,
            }
        }
    }
}
impl Read for Timed<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> { let mut stream = self.0; self.retry(|| stream.read(bytes)) }
}
impl Write for Timed<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> { let mut stream = self.0; self.retry(|| match stream.write(bytes) { Ok(0) if !bytes.is_empty() => Err(std::io::Error::from_raw_os_error(ERROR_PIPE_BUSY as i32)), result => result }) }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}
