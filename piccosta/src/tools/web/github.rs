// GitHub repository URLs as a shallow clone the model can read: the tree and README for a repo,
// a directory listing for /tree/, the file for /blob/, and the clone's path for read and bash.
//
// Based on pi-web-access 0.44.0 (MIT, see LICENSE.md): github-extract.ts and
// github-api.ts. The text the model reads is pi-web-access's. What changed:
// - Clones live under <agent dir>/github-clones (or `web.githubClone.clonePath`), not
//   /tmp/pi-github-repos. Each process clones into its own runtime-* directory, removes it when
//   it exits, and sweeps runtime directories whose owning process is gone, as pi-web-access does.
// - Issue and pull request URLs aren't given gh's structured view; they go to the page fetch.

use super::extract::Extracted;
use super::WebShared;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

// github-extract.ts:13-36 and github-api.ts:5-6.
const BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "svg", "tiff", "tif",
    "mp3", "mp4", "avi", "mov", "mkv", "flv", "wmv", "wav", "ogg", "webm", "flac", "aac",
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "zst",
    "exe", "dll", "so", "dylib", "bin", "o", "a", "lib",
    "woff", "woff2", "ttf", "otf", "eot",
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "sqlite", "db", "sqlite3",
    "pyc", "pyo", "class", "jar", "war",
    "iso", "img", "dmg",
];
const NOISE_DIRS: &[&str] = &["node_modules", "vendor", ".next", "dist", "build", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", "target", ".gradle", ".idea", ".vscode"];
const NON_CODE_SEGMENTS: &[&str] = &[
    "issues", "pull", "pulls", "discussions", "releases", "wiki", "actions", "settings", "security", "projects", "graphs",
    "compare", "commits", "tags", "branches", "stargazers", "watchers", "network", "forks", "milestone", "labels",
    "packages", "codespaces", "contribute", "community", "sponsors", "invitations", "notifications", "insights",
];
const MAX_INLINE_FILE_CHARS: usize = 100_000;
const MAX_TREE_ENTRIES: usize = 200;
const README_MAX_CHARS: usize = 8192;
const OWNER_FILE: &str = ".owner.json";
const OWNER_VERSION: u64 = 1;
const DEFAULT_CLONE_DIR: &str = "github-clones";
// gh's own calls: `gh --version` 5 s, the API reads 10 s, the tree 15 s (github-api.ts).
const GH_CHECK_TIMEOUT: Duration = Duration::from_secs(5);
const GH_API_TIMEOUT: Duration = Duration::from_secs(10);
const GH_TREE_TIMEOUT: Duration = Duration::from_secs(15);
// After SIGTERM, how long a clone's process group gets before SIGKILL (github-extract.ts:503).
const KILL_GRACE: Duration = Duration::from_secs(3);
const USE_TOOLS_LINE: &str = "Use `read` and `bash` tools at the path above to explore further.";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Root,
    Tree,
    Blob,
}

#[derive(Debug, Clone)]
pub struct GithubUrl {
    pub owner: String,
    pub repo: String,
    pub git_ref: Option<String>,
    pub ref_is_full_sha: bool,
    pub path: String,
    pub kind: Kind,
}

// decodeURIComponent: None on a malformed escape or bytes that aren't UTF-8.
fn decode_component(segment: &str) -> Option<String> {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b'%' {
            let hex = segment.get(at + 1..at + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            at += 3;
        } else {
            out.push(bytes[at]);
            at += 1;
        }
    }
    String::from_utf8(out).ok()
}

// A github.com repo, tree or blob URL (github-extract.ts parseGitHubUrl).
pub fn parse_github_url(url: &str) -> Option<GithubUrl> {
    let parsed = url::Url::parse(url).ok()?;
    let host = parsed.host_str()?.to_lowercase();
    if host != "github.com" && host != "www.github.com" {
        return None;
    }
    let mut segments = Vec::new();
    for segment in parsed.path().split('/').filter(|s| !s.is_empty()) {
        segments.push(decode_component(segment)?);
    }
    if segments.len() < 2 {
        return None;
    }
    let owner = segments[0].clone();
    let repo = segments[1].strip_suffix(".git").unwrap_or(&segments[1]).to_string();
    let owner_ok = (1..=39).contains(&owner.len())
        && owner.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        && !owner.starts_with('-')
        && !owner.ends_with('-')
        && !owner.contains("--");
    let repo_ok = (1..=100).contains(&repo.len()) && repo.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) && repo != "." && repo != "..";
    if !owner_ok || !repo_ok {
        return None;
    }
    if let Some(third) = segments.get(2)
        && NON_CODE_SEGMENTS.contains(&third.to_lowercase().as_str())
    {
        return None;
    }
    if segments.len() == 2 {
        return Some(GithubUrl { owner, repo, git_ref: None, ref_is_full_sha: false, path: String::new(), kind: Kind::Root });
    }
    let kind = match segments[2].as_str() {
        "blob" => Kind::Blob,
        "tree" => Kind::Tree,
        _ => return None,
    };
    let git_ref = segments.get(3)?.clone();
    if git_ref.is_empty() || git_ref.len() > 1024 || git_ref.chars().any(|c| (c as u32) < 0x20 || c as u32 == 0x7f) {
        return None;
    }
    let ref_is_full_sha = git_ref.len() == 40 && git_ref.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
    Some(GithubUrl { owner, repo, git_ref: Some(git_ref), ref_is_full_sha, path: segments[4..].join("/"), kind })
}

// formatFileSize: B under a KiB, then KB and MB to one decimal.
fn format_file_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{} KB", crate::js::to_fixed(bytes as f64 / 1024.0, 1))
    } else {
        format!("{} MB", crate::js::to_fixed(bytes as f64 / (1024.0 * 1024.0), 1))
    }
}

// Directory entries sorted as JavaScript's default sort does, by UTF-16 code units.
fn sorted_entries(dir: &Path) -> Option<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir).ok()?.flatten().map(|entry| entry.file_name().to_string_lossy().into_owned()).collect();
    names.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    Some(names)
}

// A path inside the clone, or None when it (or a symlink in it) leads outside
// (github-extract.ts resolveWithinRepo).
fn within_repo(root: &Path, relative: &str) -> Option<PathBuf> {
    let mut candidate = root.to_path_buf();
    for part in relative.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if !candidate.pop() || !candidate.starts_with(root) {
                    return None;
                }
            }
            _ => candidate.push(part),
        }
    }
    if !candidate.starts_with(root) {
        return None;
    }
    if !candidate.exists() {
        return Some(candidate);
    }
    let real_root = std::fs::canonicalize(root).ok()?;
    let real = std::fs::canonicalize(&candidate).ok()?;
    real.starts_with(&real_root).then_some(candidate)
}

fn is_binary(path: &Path) -> bool {
    let extension = path.extension().map(|e| e.to_string_lossy().to_lowercase());
    if let Some(extension) = &extension
        && BINARY_EXTENSIONS.contains(&extension.as_str())
    {
        return true;
    }
    use std::io::Read as _;
    let Ok(mut file) = std::fs::File::open(path) else { return false };
    let mut head = [0u8; 512];
    let read = file.read(&mut head).unwrap_or(0);
    head[..read].contains(&0)
}

fn build_tree(root: &Path) -> String {
    fn walk(root: &Path, dir: &Path, rel: &str, entries: &mut Vec<String>) {
        if entries.len() >= MAX_TREE_ENTRIES {
            return;
        }
        let Some(items) = sorted_entries(dir) else { return };
        for item in items {
            if entries.len() >= MAX_TREE_ENTRIES {
                return;
            }
            if item == ".git" {
                continue;
            }
            let path = if rel.is_empty() { item.clone() } else { format!("{rel}/{item}") };
            let Some(safe) = within_repo(root, &path) else {
                entries.push(format!("{path}  [outside repo skipped]"));
                continue;
            };
            let Ok(meta) = std::fs::metadata(&safe) else { continue };
            if meta.is_dir() {
                if NOISE_DIRS.contains(&item.as_str()) {
                    entries.push(format!("{path}/  [skipped]"));
                    continue;
                }
                entries.push(format!("{path}/"));
                walk(root, &safe, &path, entries);
            } else {
                entries.push(path);
            }
        }
    }
    let mut entries = Vec::new();
    walk(root, root, "", &mut entries);
    if entries.len() >= MAX_TREE_ENTRIES {
        entries.push(format!("... (truncated at {MAX_TREE_ENTRIES} entries)"));
    }
    entries.join("\n")
}

fn build_dir_listing(root: &Path, sub_path: &str) -> String {
    let Some(target) = within_repo(root, sub_path) else { return "(path escapes repository root)".to_string() };
    let Some(items) = sorted_entries(&target) else { return "(directory not readable)".to_string() };
    let mut lines = Vec::new();
    for item in items {
        if item == ".git" {
            continue;
        }
        let rel = if sub_path.is_empty() { item.clone() } else { format!("{sub_path}/{item}") };
        let Some(safe) = within_repo(root, &rel) else {
            lines.push(format!("  {item}  (outside repo)"));
            continue;
        };
        match std::fs::metadata(&safe) {
            Ok(meta) if meta.is_dir() => lines.push(format!("  {item}/")),
            Ok(meta) => lines.push(format!("  {item}  ({})", format_file_size(meta.len()))),
            Err(_) => lines.push(format!("  {item}  (unreadable)")),
        }
    }
    lines.join("\n")
}

// The first `max` UTF-16 units of a string.
fn utf16_prefix(text: &str, max: usize) -> String {
    let units: Vec<u16> = text.encode_utf16().take(max).collect();
    String::from_utf16_lossy(&units)
}

fn truncated_readme(content: &str) -> String {
    if crate::js::utf16_len(content) > README_MAX_CHARS { format!("{}\n\n[README truncated at 8K chars]", utf16_prefix(content, README_MAX_CHARS)) } else { content.to_string() }
}

fn read_readme(root: &Path) -> Option<String> {
    for name in ["README.md", "readme.md", "README", "README.txt", "README.rst"] {
        let Some(path) = within_repo(root, name) else { continue };
        if let Ok(bytes) = std::fs::read(path) {
            return Some(truncated_readme(&String::from_utf8_lossy(&bytes)));
        }
    }
    None
}

// What a cloned repo shows for the URL's kind (github-extract.ts generateContent).
fn generate_content(root: &Path, info: &GithubUrl) -> String {
    let mut lines = vec![format!("Repository cloned to: {}", root.display()), String::new()];
    let structure = |lines: &mut Vec<String>| {
        lines.push("## Structure".to_string());
        lines.push(build_tree(root));
    };
    match info.kind {
        Kind::Root => {
            structure(&mut lines);
            lines.push(String::new());
            if let Some(readme) = read_readme(root) {
                lines.extend(["## README.md".to_string(), readme, String::new()]);
            }
            lines.push(USE_TOOLS_LINE.to_string());
        }
        Kind::Tree => {
            let dir = &info.path;
            match within_repo(root, dir).filter(|path| path.exists()) {
                Some(_) => {
                    lines.push(format!("## {}", if dir.is_empty() { "/" } else { dir }));
                    lines.push(build_dir_listing(root, dir));
                }
                None => {
                    lines.extend([format!("Path `{dir}` not found in clone. Showing repository root instead."), String::new()]);
                    structure(&mut lines);
                }
            }
            lines.extend([String::new(), USE_TOOLS_LINE.to_string()]);
        }
        Kind::Blob => {
            let file = &info.path;
            let Some(full) = within_repo(root, file).filter(|path| path.exists()) else {
                lines.extend([format!("Path `{file}` not found in clone. Showing repository root instead."), String::new()]);
                structure(&mut lines);
                lines.extend([String::new(), USE_TOOLS_LINE.to_string()]);
                return lines.join("\n");
            };
            let meta = match std::fs::metadata(&full) {
                Ok(meta) => meta,
                Err(error) => {
                    lines.extend([format!("Could not inspect `{file}`: {error}"), String::new(), USE_TOOLS_LINE.to_string()]);
                    return lines.join("\n");
                }
            };
            if meta.is_dir() {
                lines.push(format!("## {}", if file.is_empty() { "/" } else { file }));
                lines.push(build_dir_listing(root, file));
                lines.extend([String::new(), USE_TOOLS_LINE.to_string()]);
                return lines.join("\n");
            }
            if is_binary(&full) {
                let extension = Path::new(file).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
                lines.push(format!("## {file}"));
                lines.push(format!("Binary file ({extension}, {}). Use `read` or `bash` tools at the path above to inspect.", format_file_size(meta.len())));
                return lines.join("\n");
            }
            let content = match std::fs::read(&full) {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(_) => {
                    lines.extend([format!("Could not read `{file}` as UTF-8 text."), String::new(), USE_TOOLS_LINE.to_string()]);
                    return lines.join("\n");
                }
            };
            lines.push(format!("## {file}"));
            if crate::js::utf16_len(&content) > MAX_INLINE_FILE_CHARS {
                lines.extend([utf16_prefix(&content, MAX_INLINE_FILE_CHARS), String::new(), format!("[File truncated at 100K chars. Full file: {}]", full.display())]);
            } else {
                lines.push(content);
            }
            lines.extend([String::new(), USE_TOOLS_LINE.to_string()]);
        }
    }
    lines.join("\n")
}

fn title_of(info: &GithubUrl) -> String {
    if info.path.is_empty() { format!("{}/{}", info.owner, info.repo) } else { format!("{}/{} - {}", info.owner, info.repo, info.path) }
}

// One gh call's stdout, or None when gh fails, times out or isn't there.
async fn gh(clones: &Clones, args: &[&str], timeout: Duration) -> Option<String> {
    let mut command = tokio::process::Command::new("gh");
    command.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).env("GH_PROMPT_DISABLED", "1");
    let (child, _group) = clones.spawn(&mut command)?;
    let output = tokio::time::timeout(timeout, child.wait_with_output()).await.ok()?.ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

// Whether a runtime directory's owner is provably gone (github-extract.ts cloneRuntimeOwnerIsDead):
// another boot's, or its pid gone, a zombie, or started at another time. The owner record is the
// pid, boot id and start time of the process that made the directory (runtime_dir).
fn owner_is_dead(owner: &Value) -> bool {
    let (Some(pid), Some(boot), Some(start)) = (owner["pid"].as_u64(), owner["bootId"].as_str(), owner["startTime"].as_str()) else { return false };
    crate::proc::recorded_start(&format!("{boot} {start}")).and_then(|start| crate::proc::Process::started(pid as i32, start)).is_none()
}

// The clones this process made, one per repo and ref.
pub struct Clones {
    parent: PathBuf,
    runtime: Mutex<Option<PathBuf>>,
    cache: Mutex<HashMap<String, Arc<OnceCell<Option<PathBuf>>>>>,
    gh_available: OnceCell<bool>,
    children: Arc<Mutex<Option<Vec<i32>>>>,
}

// The child can finish before its helpers. Keep the whole group owned until the call ends,
// including when its future is dropped during cancellation or shutdown.
struct ChildGroup {
    pid: i32,
    children: Arc<Mutex<Option<Vec<i32>>>>,
}

impl Drop for ChildGroup {
    fn drop(&mut self) {
        let mut children = self.children.lock().unwrap();
        if let Some(children) = children.as_mut()
            && let Some(at) = children.iter().position(|pid| *pid == self.pid)
        {
            unsafe { libc::kill(-self.pid, libc::SIGKILL) };
            children.swap_remove(at);
        }
    }
}

impl Clones {
    pub fn new(agent_dir: &Path, configured: Option<&Path>) -> Clones {
        Clones {
            parent: configured.map(Path::to_path_buf).unwrap_or_else(|| agent_dir.join(DEFAULT_CLONE_DIR)),
            runtime: Mutex::new(None),
            cache: Mutex::new(HashMap::new()),
            gh_available: OnceCell::new(),
            children: Arc::new(Mutex::new(Some(Vec::new()))),
        }
    }

    async fn has_gh(&self) -> bool {
        *self.gh_available.get_or_init(|| async { gh(self, &["--version"], GH_CHECK_TIMEOUT).await.is_some() }).await
    }

    fn spawn(&self, command: &mut tokio::process::Command) -> Option<(tokio::process::Child, ChildGroup)> {
        let mut children = self.children.lock().unwrap();
        let children = children.as_mut()?;
        let child = command.process_group(0).kill_on_drop(true).spawn().ok()?;
        let pid = child.id()? as i32;
        children.push(pid);
        Some((child, ChildGroup { pid, children: self.children.clone() }))
    }

    // This process's runtime-* directory, made (and stale ones swept) on first use.
    fn runtime_dir(&self) -> Option<PathBuf> {
        let mut runtime = self.runtime.lock().unwrap();
        if self.children.lock().unwrap().is_none() {
            return None;
        }
        if let Some(path) = runtime.as_ref() {
            return Some(path.clone());
        }
        std::fs::create_dir_all(&self.parent).ok()?;
        let parent = std::fs::canonicalize(&self.parent).ok()?;
        self.sweep_stale(&parent);
        let path = parent.join(format!("runtime-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
        std::fs::create_dir(&path).ok()?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).ok()?;
        let pid = std::process::id();
        let start = crate::proc::start(pid as i32).map(|start| start.to_string());
        let owner = json!({"version": OWNER_VERSION, "pid": pid, "platform": "linux", "bootId": crate::proc::boot_id(), "startTime": start});
        std::fs::write(path.join(OWNER_FILE), owner.to_string()).ok()?;
        *runtime = Some(path.clone());
        Some(path)
    }

    fn sweep_stale(&self, parent: &Path) {
        let Ok(entries) = std::fs::read_dir(parent) else { return };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("runtime-") || !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                continue;
            }
            let owner = std::fs::read_to_string(entry.path().join(OWNER_FILE)).ok().and_then(|text| serde_json::from_str::<Value>(&text).ok());
            if owner.is_some_and(|owner| owner_is_dead(&owner)) {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }

    // Remove this process's clones; called when piccosta exits.
    pub fn shutdown(&self) {
        // Closing the registry under the spawn lock also rules out a new child after shutdown.
        for pid in self.children.lock().unwrap().take().into_iter().flatten() {
            unsafe { libc::kill(-pid, libc::SIGKILL) };
        }
        if let Some(path) = self.runtime.lock().unwrap().take() {
            let _ = std::fs::remove_dir_all(path);
        }
        self.cache.lock().unwrap().clear();
    }

    fn key(info: &GithubUrl) -> String {
        serde_json::to_string(&json!([info.owner, info.repo, info.git_ref])).unwrap_or_default()
    }

    // A clone this process already made or is making for the repo and ref.
    fn has_clone(&self, info: &GithubUrl) -> bool {
        self.cache.lock().unwrap().contains_key(&Self::key(info))
    }

    // Clone once per repo and ref; concurrent fetches of the same repo share the clone.
    async fn clone_repo(&self, info: &GithubUrl, timeout: Duration, cancel: &CancellationToken) -> Option<PathBuf> {
        let key = Self::key(info);
        let cell = self.cache.lock().unwrap().entry(key.clone()).or_insert_with(|| Arc::new(OnceCell::new())).clone();
        let result = cell.get_or_init(|| self.run_clone(info, &key, timeout, cancel)).await.clone();
        if result.is_none() {
            self.cache.lock().unwrap().remove(&key);
        }
        result
    }

    async fn run_clone(&self, info: &GithubUrl, key: &str, timeout: Duration, cancel: &CancellationToken) -> Option<PathBuf> {
        let runtime = self.runtime_dir()?;
        let digest = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut hasher);
            format!("{:016x}{}", hasher.finish(), uuid::Uuid::new_v4().simple())
        };
        let local = runtime.join(digest);
        let mut args: Vec<String> = if self.has_gh().await {
            vec!["gh".into(), "repo".into(), "clone".into(), format!("{}/{}", info.owner, info.repo), local.display().to_string(), "--".into(), "--depth".into(), "1".into(), "--single-branch".into()]
        } else {
            vec!["git".into(), "clone".into(), "--depth".into(), "1".into(), "--single-branch".into()]
        };
        if let Some(git_ref) = &info.git_ref {
            args.extend(["--branch".into(), git_ref.clone()]);
        }
        if args[0] == "git" {
            args.extend([format!("https://github.com/{}/{}.git", info.owner, info.repo), local.display().to_string()]);
        }
        let mut command = tokio::process::Command::new(&args[0]);
        command
            .args(&args[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GCM_INTERACTIVE", "Never")
            .env("GH_PROMPT_DISABLED", "1");
        let (mut child, group) = self.spawn(&mut command)?;
        let pid = child.id().map(|pid| pid as i32);
        let succeeded = tokio::select! {
            status = child.wait() => status.is_ok_and(|status| status.success()),
            () = tokio::time::sleep(timeout) => false,
            () = cancel.cancelled() => false,
        };
        if !succeeded {
            if let Some(pid) = pid {
                unsafe { libc::kill(-pid, libc::SIGTERM) };
                if tokio::time::timeout(KILL_GRACE, child.wait()).await.is_err() {
                    unsafe { libc::kill(-pid, libc::SIGKILL) };
                    let _ = child.wait().await;
                }
            }
            drop(group);
            let _ = std::fs::remove_dir_all(&local);
            return None;
        }
        Some(local)
    }
}

impl Drop for Clones {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// The API view, when a repo is too big to clone, the ref is a commit, or the clone failed
// (github-api.ts fetchViaApi). Needs gh.
async fn fetch_via_api(url: &str, info: &GithubUrl, size_note: Option<String>, clones: &Clones) -> Option<Extracted> {
    if !clones.has_gh().await {
        return None;
    }
    let repo_path = format!("repos/{}/{}", info.owner, info.repo);
    let git_ref = match &info.git_ref {
        Some(git_ref) => git_ref.clone(),
        None => gh(clones, &["api", &repo_path, "--jq", ".default_branch"], GH_API_TIMEOUT).await.map(|b| b.trim().to_string()).filter(|b| !b.is_empty())?,
    };
    let mut lines: Vec<String> = Vec::new();
    if let Some(note) = size_note {
        lines.extend([note, String::new()]);
    }
    let decode = |text: String| -> Option<String> {
        use base64::Engine as _;
        let cleaned: String = text.chars().filter(|c| !c.is_whitespace() && *c != '"').collect::<String>().replace("\\n", "");
        base64::engine::general_purpose::STANDARD.decode(cleaned).ok().map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    };
    if info.kind == Kind::Blob && !info.path.is_empty() {
        let content = gh(clones, &["api", &format!("{repo_path}/contents/{}?ref={git_ref}", info.path), "--jq", ".content"], GH_API_TIMEOUT).await.and_then(decode).filter(|c| !c.is_empty())?;
        lines.push(format!("## {}", info.path));
        if crate::js::utf16_len(&content) > MAX_INLINE_FILE_CHARS {
            lines.extend([utf16_prefix(&content, MAX_INLINE_FILE_CHARS), "\n[File truncated at 100K chars]".to_string()]);
        } else {
            lines.push(content);
        }
        return Some(Extracted { url: url.to_string(), title: format!("{}/{} - {}", info.owner, info.repo, info.path), content: lines.join("\n"), ..Default::default() });
    }
    let (tree_path, readme_path) = (format!("{repo_path}/git/trees/{git_ref}?recursive=1"), format!("{repo_path}/readme?ref={git_ref}"));
    let tree_args = ["api", tree_path.as_str(), "--jq", ".tree[].path"];
    let readme_args = ["api", readme_path.as_str(), "--jq", ".content"];
    let (tree, readme) = tokio::join!(gh(clones, &tree_args, GH_TREE_TIMEOUT), gh(clones, &readme_args, GH_API_TIMEOUT));
    let tree = tree.map(|text| text.trim().split('\n').filter(|line| !line.is_empty()).map(str::to_string).collect::<Vec<_>>()).filter(|paths| !paths.is_empty()).map(|paths| {
        let shown = paths[..paths.len().min(MAX_TREE_ENTRIES)].join("\n");
        if paths.len() > MAX_TREE_ENTRIES { format!("{shown}\n... ({} total entries)", paths.len()) } else { shown }
    });
    let readme = readme.and_then(decode).map(|text| truncated_readme(&text));
    if tree.is_none() && readme.is_none() {
        return None;
    }
    if let Some(tree) = tree {
        lines.extend(["## Structure".to_string(), tree, String::new()]);
    }
    if let Some(readme) = readme {
        lines.extend(["## README.md".to_string(), readme, String::new()]);
    }
    lines.push("This is an API-only view. Clone the repo or use `read`/`bash` for deeper exploration.".to_string());
    Some(Extracted { url: url.to_string(), title: title_of(info), content: lines.join("\n"), ..Default::default() })
}

// Input: a parsed GitHub URL. Output: the repo view, or None to fall through to fetching the
// page (github-extract.ts extractGitHub).
pub async fn extract_github(url: &str, info: &GithubUrl, force_clone: bool, web: &WebShared, cancel: &CancellationToken) -> Option<Extracted> {
    let settings = &web.settings;
    if !settings.github_clone_enabled || cancel.is_cancelled() {
        return None;
    }
    let clones = &web.clones;
    if info.ref_is_full_sha {
        return fetch_via_api(url, info, Some("Note: Commit SHA URLs use the GitHub API instead of cloning.".to_string()), clones).await;
    }
    // A repo already cloned (or being cloned) is used as it is, before any size check.
    if !force_clone
        && !clones.has_clone(info)
        && clones.has_gh().await
        && let Some(size_kb) = gh(clones, &["api", &format!("repos/{}/{}", info.owner, info.repo), "--jq", ".size"], GH_API_TIMEOUT).await.and_then(|text| text.trim().parse::<f64>().ok())
    {
        let size_mb = size_kb / 1024.0;
        if size_mb > settings.github_max_repo_size_mb {
            let note = format!(
                "Note: Repository is {}MB (threshold: {}MB). Showing API-fetched content instead of full clone. Ask the user if they'd like to clone the full repo -- if yes, call fetch_content again with the same URL and add forceClone: true to the params.",
                size_mb.round(),
                crate::js::number_string(settings.github_max_repo_size_mb)
            );
            return fetch_via_api(url, info, Some(note), clones).await;
        }
    }
    let timeout = Duration::from_secs_f64(settings.github_clone_timeout_s);
    match clones.clone_repo(info, timeout, cancel).await {
        Some(local) => {
            let title = title_of(info);
            let view = info.clone();
            let content = tokio::task::spawn_blocking(move || generate_content(&local, &view)).await.ok()?;
            Some(Extracted { url: url.to_string(), title, content, ..Default::default() })
        }
        None if cancel.is_cancelled() => None,
        None => fetch_via_api(url, info, None, clones).await,
    }
}
