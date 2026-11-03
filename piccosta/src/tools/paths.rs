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
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

// "~" and "~/..." under home; any other path as written.
pub fn expand_home(raw: &str) -> PathBuf {
    match raw.strip_prefix('~') {
        Some("") => home_dir(),
        Some(rest) if rest.starts_with('/') => home_dir().join(&rest[1..]),
        _ => PathBuf::from(raw),
    }
}

// Input: a path as the model wrote it. Output: the path with odd spaces made plain, a leading
// "@" dropped (models copy @file mentions), "~" expanded and a file:// URL turned into its path.
pub fn normalize(raw: &str) -> PathBuf {
    let spaced: String = raw.chars().map(|c| if is_unicode_space(c) { ' ' } else { c }).collect();
    let path = spaced.strip_prefix('@').unwrap_or(&spaced);
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
    let mut out = PathBuf::from("/");
    for component in path.components() {
        match component {
            Component::Normal(part) => out.push(part),
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir | Component::CurDir | Component::Prefix(_) => {}
        }
    }
    out
}

// Node's code and message for an errno, as libuv's uv_strerror writes them. A model reads these
// (pi passes fs errors through), and "ENOENT: no such file or directory, open '/x'" names both
// the problem and the path. Errnos not listed fall back to Rust's own description.
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

// access(2) on a path, as Node's fs.access checks it. Input: the path and libc's R_OK/W_OK bits.
pub fn access(path: &Path, mode: libc::c_int) -> io::Result<()> {
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
