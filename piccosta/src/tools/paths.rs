// Turning the path a model wrote into a file on disk, and filesystem errors into the text the
// model reads.
//
// Path rules are based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/utils/paths.ts and
// core/tools/path-utils.ts. Left out: Windows drive paths, and read's macOS screenshot-name
// variants (AM/PM narrow spaces, NFD names, curly apostrophes), which don't arise on Linux.

use std::io;
use std::path::{Component, Path, PathBuf};

// Spaces models and terminals put into paths in place of a plain space (paths.ts:5).
fn is_unicode_space(c: char) -> bool {
    matches!(c, '\u{a0}' | '\u{2000}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

fn home_dir() -> PathBuf {
    crate::config::home()
}

// "~" and "~/..." under home; any other path as written.
pub fn expand_home(raw: &str) -> PathBuf {
    match raw.strip_prefix('~') {
        Some("") => home_dir(),
        Some(rest) if rest.starts_with('/') || cfg!(windows) && rest.starts_with('\\') => home_dir().join(rest.trim_start_matches(['/', std::path::MAIN_SEPARATOR])),
        _ => PathBuf::from(raw),
    }
}

// Input: a path as the model wrote it. Output: the path with odd spaces made plain, a leading
// "@" dropped (models copy @file mentions), "~" expanded and a file:// URL turned into its path.
pub fn normalize(raw: &str) -> PathBuf {
    let spaced: String = raw.chars().map(|c| if is_unicode_space(c) { ' ' } else { c }).collect();
    let path = spaced.strip_prefix('@').unwrap_or(&spaced);
    #[cfg(windows)]
    if path.starts_with("file://") {
        return url::Url::parse(path).ok().and_then(|url| url.to_file_path().ok()).unwrap_or_else(|| PathBuf::from(path));
    }
    match path.strip_prefix("file://") {
        Some(rest) => PathBuf::from(percent_decode(rest.strip_prefix("localhost").unwrap_or(rest))),
        None => expand_home(path),
    }
}

// %XX escapes in a file URL's path, as fileURLToPath undoes them. Bytes that don't form valid
// UTF-8 afterwards are replaced, which only affects paths that were never valid text.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && let Some(hex) = text.get(index + 1..index + 3)
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
            index += 3;
            continue;
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// Input: a path as the model wrote it and the session cwd. Output: an absolute path with "." and
// ".." resolved lexically, like Node's path.resolve (symlinks are not followed).
pub fn resolve(raw: &str, cwd: &Path) -> PathBuf {
    let path = normalize(raw);
    let joined = if path.is_absolute() { path } else { cwd.join(path) };
    // A relative cwd is taken from the process's own directory, as path.resolve does.
    let joined = if joined.is_absolute() { joined } else { std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")).join(joined) };
    lexical_clean(&joined)
}

fn lexical_clean(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir | Component::Prefix(_) => out.push(component.as_os_str()),
            Component::CurDir => {}
        }
    }
    out
}

// Node's code and message for an errno, as libuv's uv_strerror writes them. A model reads these
// (pi passes fs errors through), and "ENOENT: no such file or directory, open '/x'" names both
// the problem and the path. Errnos not listed fall back to Rust's own description.
#[cfg(windows)]
fn errno_text(_code: i32) -> Option<(&'static str, &'static str)> { None }

#[cfg(unix)]
fn errno_text(code: i32) -> Option<(&'static str, &'static str)> {
    Some(match code {
        1 => ("EPERM", "operation not permitted"),
        2 => ("ENOENT", "no such file or directory"),
        5 => ("EIO", "i/o error"),
        9 => ("EBADF", "bad file descriptor"),
        13 => ("EACCES", "permission denied"),
        16 => ("EBUSY", "resource busy or locked"),
        17 => ("EEXIST", "file already exists"),
        18 => ("EXDEV", "cross-device link not permitted"),
        20 => ("ENOTDIR", "not a directory"),
        21 => ("EISDIR", "illegal operation on a directory"),
        22 => ("EINVAL", "invalid argument"),
        24 => ("EMFILE", "too many open files"),
        26 => ("ETXTBSY", "text file is busy"),
        27 => ("EFBIG", "file too large"),
        28 => ("ENOSPC", "no space left on device"),
        30 => ("EROFS", "read-only file system"),
        36 => ("ENAMETOOLONG", "name too long"),
        39 => ("ENOTEMPTY", "directory not empty"),
        40 => ("ELOOP", "too many symbolic links encountered"),
        122 => ("EDQUOT", "disk quota exceeded"),
        _ => return None,
    })
}

// The errno's code name alone, e.g. "ENOENT", for messages that print only the code.
pub fn error_code(error: &io::Error) -> String {
    error
        .raw_os_error()
        .and_then(errno_text)
        .map(|(code, _)| code.to_string())
        .unwrap_or_else(|| format!("{:?}", error.kind()))
}

// Input: an io error, the syscall Node would name (open, access, mkdir, read...) and the path,
// or None for calls Node reports without one (read on a directory). Output: Node's message,
// e.g. "EACCES: permission denied, open '/etc/shadow'".
pub fn fs_error(error: &io::Error, syscall: &str, path: Option<&Path>) -> String {
    let (code, description) = match error.raw_os_error().and_then(errno_text) {
        Some((code, description)) => (code.to_string(), description.to_string()),
        None => return format!("{error}, {syscall}{}", path.map(|path| format!(" '{}'", path.display())).unwrap_or_default()),
    };
    match path {
        Some(path) => format!("{code}: {description}, {syscall} '{}'", path.display()),
        None => format!("{code}: {description}, {syscall}"),
    }
}

// Can this user read the file, and (for edit) write it too? Unix keeps Node's access(2) check;
// Windows opens it to check its ACL, rather than treating Unix permission bits as an ACL.
#[cfg(windows)]
pub fn access(path: &Path, writable: bool) -> io::Result<()> {
    std::fs::OpenOptions::new().read(true).write(writable).open(path).map(drop)
}

#[cfg(unix)]
pub fn access(path: &Path, writable: bool) -> io::Result<()> {
    let mode = libc::R_OK | if writable { libc::W_OK } else { 0 };
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|_| io::Error::from_raw_os_error(libc::ENOENT))?;
    // SAFETY: access() only reads the NUL-terminated path we just built.
    if unsafe { libc::access(c_path.as_ptr(), mode) } == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}

// The key the mutation queue and editfix's snapshots use for a file: its real path, so "./a",
// "src/../a" and a symlink to it all meet, or the resolved path when the file doesn't exist yet.
pub fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

// Write `bytes` as the whole of `path`. The file is written in place, so it keeps its inode, its
// other hard links, its owner, permissions and attributes, and a symlink stays a link. A write that
// fails part way (a full disk) puts the old text back, so the file isn't left empty or half written;
// when even that fails, the old text is kept beside it and the error says where. A file this user
// may write but not read is written without that safety, as before.
pub fn replace(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let old = match std::fs::read(path) {
        Ok(old) => Some(old),
        Err(error) if matches!(error.kind(), io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied) => None,
        Err(error) => return Err(error),
    };
    let Err(error) = std::fs::write(path, bytes) else { return Ok(()) };
    let Some(old) = old else { return Err(error) };
    if std::fs::write(path, &old).is_ok() {
        return Err(error);
    }
    let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let backup = path.with_file_name(format!("{name}.before-piccosta"));
    let kept = match std::fs::write(&backup, &old) {
        Ok(()) => format!("its old text is in {}", backup.display()),
        Err(_) => "its old text couldn't be kept anywhere".to_string(),
    };
    Err(io::Error::new(error.kind(), format!("{error}; it couldn't be put back as it was, and {kept}")))
}
