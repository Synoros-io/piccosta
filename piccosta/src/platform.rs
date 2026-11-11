// Files that carry credentials or agent output are private to this user. Unix needs explicit
// mode bits; Windows inherits the user's profile ACL. Named pipes have their own owner-only ACL.
use std::path::Path;

pub fn permissions(path: &Path, mode: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    { use std::os::unix::fs::PermissionsExt; std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)) }
    #[cfg(windows)]
    { let _ = (path, mode); Ok(()) }
}
pub fn private_file(options: &mut std::fs::OpenOptions) -> &mut std::fs::OpenOptions {
    #[cfg(unix)]
    { use std::os::unix::fs::OpenOptionsExt; options.mode(0o600); }
    options
}
pub fn private_dir(path: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    { use std::os::unix::fs::DirBuilderExt; builder.mode(0o700); }
    builder.create(path)
}
// Windows file locks are mandatory, including for another handle in this same process. Lock
// one byte beyond Windows' maximum file length (a signed 64-bit offset), so the session and
// inbox can still be read/appended while this coordination lock is held. Closing it unlocks.
#[cfg(windows)]
pub fn try_session_lock(file: &std::fs::File) -> Result<(), std::fs::TryLockError> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::{Foundation::ERROR_LOCK_VIOLATION, Storage::FileSystem::*, System::IO::*};
    const LOCK_OFFSET: u64 = i64::MAX as u64;
    let mut position: OVERLAPPED = unsafe { std::mem::zeroed() };
    position.Anonymous.Anonymous = OVERLAPPED_0_0 { Offset: LOCK_OFFSET as u32, OffsetHigh: (LOCK_OFFSET >> 32) as u32 };
    if unsafe { LockFileEx(file.as_raw_handle(), LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &mut position) } != 0 { return Ok(()); }
    let error = std::io::Error::last_os_error();
    Err(if error.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) { std::fs::TryLockError::WouldBlock } else { std::fs::TryLockError::Error(error) })
}
#[cfg(unix)]
pub fn try_session_lock(file: &std::fs::File) -> Result<(), std::fs::TryLockError> { file.try_lock() }

#[cfg(unix)]
pub const NULL: &str = "/dev/null";
#[cfg(windows)]
pub const NULL: &str = "NUL";

// Identity from the open handle, so a replacement between stat and open can't lend its
// cursor to a different file. Windows' file index is scoped to its volume, like Unix's inode.
pub fn file_identity(file: &std::fs::File) -> std::io::Result<(u64, u64)> {
    #[cfg(unix)]
    { use std::os::unix::fs::MetadataExt; let metadata = file.metadata()?; Ok((metadata.dev(), metadata.ino())) }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle};
        let mut information: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) } == 0 { return Err(std::io::Error::last_os_error()); }
        Ok((information.dwVolumeSerialNumber as u64, ((information.nFileIndexHigh as u64) << 32) | information.nFileIndexLow as u64))
    }
}

// Find native executables without making callers know about Windows' suffix.
pub fn executable(dir: &Path, name: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    if cfg!(windows) && path.extension().is_none() { path.with_extension("exe") } else { path }
}
