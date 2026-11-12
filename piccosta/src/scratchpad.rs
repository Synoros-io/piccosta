// One Markdown file for a directory, shared by its sessions and the user's pane. The lock lives
// beside the file, so an editor replacing the file can't replace a lock a writer is waiting on.

use crate::tools::{editfix::SeenFiles, paths};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

// Match editfix's bounded snapshots: a board must fit in a read whose version we can remember.
pub const MAX_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Stamp {
    pub path: PathBuf,
    length: u64,
    hash: Option<[u8; 32]>,
    pub modified: std::time::SystemTime,
    created: Option<std::time::SystemTime>,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    mode: u32,
}

// Supplying content binds completion to the bytes we committed, not a later outside save.
// Plain metadata stamps still avoid rereading unchanged documents and assets.
pub(crate) fn stamp(path: &Path, content: Option<&str>) -> Result<Stamp, String> {
    let metadata = std::fs::metadata(path).map_err(|error| error.to_string())?;
    if !metadata.is_file() { return Err(format!("{} isn't a regular file", path.display())); }
    Ok(Stamp { path: paths::canonical(path), length: content.map_or(metadata.len(), |text| text.len() as u64), hash: content.map(|text| Sha256::digest(text.as_bytes()).into()), modified: metadata.modified().map_err(|error| error.to_string())?, created: metadata.created().ok(),
        #[cfg(unix)]
        inode: { use std::os::unix::fs::MetadataExt; metadata.ino() },
        #[cfg(unix)]
        mode: { use std::os::unix::fs::MetadataExt; metadata.mode() },
    })
}

// One stable identity per process, shared by prompt, tools and pane. Call on a disk worker;
// reloading settings or asking Git is necessary only on the first use of this directory.
pub fn path(cwd: &Path, agent_dir: &Path) -> PathBuf {
    static PATHS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<(PathBuf, PathBuf), PathBuf>>> = std::sync::LazyLock::new(Default::default);
    PATHS.lock().unwrap_or_else(|error| error.into_inner()).entry((cwd.to_path_buf(), agent_dir.to_path_buf())).or_insert_with(|| resolve(cwd, agent_dir)).clone()
}

fn resolve(cwd: &Path, agent_dir: &Path) -> PathBuf {
    let settings = crate::config::Settings::load(agent_dir, Some(cwd));
    if let Some(path) = settings.get("scratchpad.path").and_then(serde_json::Value::as_str).filter(|path| !path.is_empty()) {
        return paths::canonical(&paths::resolve(path, cwd));
    }
    let mut identity = paths::canonical(cwd);
    if settings.get("scratchpad.sharedAcrossWorktrees").and_then(serde_json::Value::as_bool) == Some(true)
        && let Ok(output) = std::process::Command::new("git").arg("-C").arg(cwd).args(["rev-parse", "--path-format=absolute", "--git-common-dir"]).output()
        && output.status.success()
    {
        identity = paths::canonical(Path::new(String::from_utf8_lossy(&output.stdout).trim()));
    }
    let digest = Sha256::digest(identity.as_os_str().as_encoded_bytes());
    let name: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    agent_dir.join("scratchpads").join(format!("{name}.md"))
}

pub fn lock(path: &Path) -> Result<File, String> {
    // Aliases must lock the same sidecar, even when a configured symlink's target hasn't been
    // created yet. Other canonicalization failures (including a link loop) are not new files.
    let path = match std::fs::canonicalize(path) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if let Ok(target) = std::fs::read_link(path) {
                let target = if target.is_absolute() { target } else { path.parent().ok_or("scratchpad has no parent directory")?.join(target) };
                return lock(&target);
            }
            path.to_path_buf()
        }
        Err(error) => return Err(error.to_string()),
    };
    let parent = path.parent().ok_or("scratchpad has no parent directory")?;
    std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let mut name = path.as_os_str().to_os_string();
    name.push(".lock");
    let lock = open(&PathBuf::from(name), true)?;
    lock.lock().map_err(|error| error.to_string())?;
    Ok(lock)
}

// Open before checking the handle, but never wait for a FIFO's other end (session::open_reader).
pub fn open(path: &Path, create: bool) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.read(true).append(create).create(create);
    #[cfg(unix)]
    { use std::os::unix::fs::OpenOptionsExt; options.custom_flags(libc::O_NONBLOCK); }
    let file = options.open(path).map_err(|error| error.to_string())?;
    if !file.metadata().map_err(|error| error.to_string())?.is_file() { return Err("scratchpad requires a regular file".into()); }
    Ok(file)
}

pub fn read(path: &Path) -> Result<String, String> {
    let file = open(path, false)?;
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64 + 1).read_to_end(&mut bytes).map_err(|error| error.to_string())?;
    if bytes.len() > MAX_BYTES { return Err(format!("scratchpad exceeds {MAX_BYTES} bytes")); }
    String::from_utf8(bytes).map_err(|error| error.to_string())
}

// The UI appends, never saves an old copy over the agent's work. Keep a failed append retryable;
// while holding the writer lock, take back any partial suffix before returning the error.
pub(crate) fn append(path: &Path, paragraph: &str) -> Result<Option<Stamp>, String> {
    let _lock = lock(path)?;
    let mut file = open(path, true)?;
    let length = file.metadata().map_err(|error| error.to_string())?.len();
    let suffix = format!("\n\n{paragraph}\n");
    if length.saturating_add(suffix.len() as u64) > MAX_BYTES as u64 { return Err(format!("scratchpad exceeds {MAX_BYTES} bytes")); }
    let mut content = read(path)?;
    content.push_str(&suffix);
    if let Err(error) = file.write_all(suffix.as_bytes()) {
        return Err(match file.set_len(length) {
            Ok(()) => error.to_string(),
            Err(rollback) => format!("{error}; partial append couldn't be taken back: {rollback}"),
        });
    }
    Ok(stamp(path, Some(&content)).ok())
}

// Only this session's scratchpad gets collaborative-write protection. Ordinary source edits keep
// their existing semantics. Keep the returned lock through the write and snapshot update.
pub fn guard(absolute: &Path, cwd: &Path, agent_dir: &Path, seen: &SeenFiles) -> Result<Option<File>, String> {
    if paths::canonical(absolute) != paths::canonical(&path(cwd, agent_dir)) { return Ok(None); }
    let lock = lock(absolute)?;
    if absolute.try_exists().map_err(|error| error.to_string())? && !seen.unchanged(absolute, &read(absolute)?) {
        return Err("Scratchpad changed since your last read (or hasn't been read). Read it again before write/edit; no text was replaced.".into());
    }
    Ok(Some(lock))
}
