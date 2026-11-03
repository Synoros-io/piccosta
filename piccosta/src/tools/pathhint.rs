// "No such file or directory" in a command's output gets up to three existing paths close to the
// missing one, so the model's next call can be the right one instead of an `ls` to look around.
//
// Based on the TypeScript piccosta's extensions/lib/pathhint.ts (FSL-1.1-ALv2, Synoros). The hint text goes to
// the model, so it's the same: "<path> does not exist; did you mean: a, b, c".

use crate::js;
use regex::Regex;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

// How many "did you mean" paths we print. Three is enough to cover a typo plus a couple of
// same-named files in other directories; more is noise.
const MAX_SUGGESTIONS: usize = 3;
// Missing paths we look up per command, so a failing glob over a hundred names can't make the
// hint longer than the output.
const MAX_MISSING_PATHS: usize = 3;
// A name counts as close when its edit distance from the missing name is at most a third of the
// missing name's length, and never less than 2. That catches a typo, a dropped letter or a wrong
// extension (config.yml vs config.yaml is 1) without proposing unrelated names.
const MIN_ALLOWED_DISTANCE: usize = 2;
const NAME_CHARS_PER_DISTANCE: usize = 3;
// The name search under cwd stops at this depth and after this many directory entries. Six
// levels reaches src/a/b/c/d/file in an ordinary repo; 20k entries is a medium monorepo and lists
// in well under 100 ms on local disk.
const SEARCH_MAX_DEPTH: usize = 6;
const SEARCH_MAX_ENTRIES: usize = 20_000;
// Directories the search never descends into: version control, dependency and build trees. A
// suggestion inside them is almost never the file the model meant.
const SEARCH_SKIP_DIRS: [&str; 7] = [".git", "node_modules", "target", ".venv", "__pycache__", "dist", "build"];

fn resolve_from(base_dir: &Path, raw: &str) -> PathBuf {
    let path = super::paths::expand_home(raw);
    let joined = if path.is_absolute() { path } else { base_dir.join(path) };
    super::paths::resolve(&joined.to_string_lossy(), Path::new("/"))
}

// Levenshtein distance over characters, two rows at a time.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut current = vec![i];
        for j in 1..=b.len() {
            let substitution = previous[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            current.push((previous[j] + 1).min(current[j - 1] + 1).min(substitution));
        }
        previous = current;
    }
    previous[b.len()]
}

fn allowed_distance(name: &str) -> usize {
    MIN_ALLOWED_DISTANCE.max(js::utf16_len(name) / NAME_CHARS_PER_DISTANCE)
}

// Every directory a relative path in the command could be relative to: the cwd, plus each "cd X"
// target, since every call starts in a fresh shell at the session cwd.
fn candidate_base_dirs(command: &str, cwd: &Path) -> Vec<PathBuf> {
    static CD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:^|[;&|\n]\s*)cd\s+([^\s;&|]+)").expect("valid regex"));
    let mut dirs = vec![cwd.to_path_buf()];
    let mut current = cwd.to_path_buf();
    for captures in CD.captures_iter(command) {
        let target = &captures[1];
        let target = target.strip_prefix(['\'', '"']).unwrap_or(target);
        let target = target.strip_suffix(['\'', '"']).unwrap_or(target);
        current = resolve_from(&current, target);
        if !dirs.contains(&current) {
            dirs.push(current.clone());
        }
    }
    dirs
}

// The shapes "No such file or directory" takes in the output of common tools. Each pattern
// captures the path:
//   cat: foo.txt: No such file or directory          (coreutils, bash, cd)
//   ls: cannot access 'foo': No such file or directory
//   FileNotFoundError: [Errno 2] No such file or directory: 'foo'   (python)
//   python3: can't open file '/w/foo.py': [Errno 2] No such file or directory
static MISSING_PATH_PATTERNS: LazyLock<[Regex; 4]> = LazyLock::new(|| {
    [
        r"'([^'\n]+)': No such file or directory",
        r"No such file or directory: '([^'\n]+)'",
        r"'([^'\n]+)': \[Errno 2\]",
        r#"(?m)(?:^|[\s:])([^\s:'"]+): No such file or directory"#,
    ]
    .map(|pattern| Regex::new(pattern).expect("valid regex"))
});

// The distinct paths an output says are missing, in order.
fn missing_paths_in(output: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for pattern in MISSING_PATH_PATTERNS.iter() {
        for captures in pattern.captures_iter(output) {
            let path = js::trim(&captures[1]);
            if !path.is_empty() && !found.iter().any(|seen| seen == path) {
                found.push(path.to_string());
            }
        }
    }
    found
}

fn close_entries_in(directory: &Path, name: &str) -> Vec<(usize, PathBuf)> {
    let limit = allowed_distance(name);
    let Ok(entries) = std::fs::read_dir(directory) else { return Vec::new() };
    entries
        .flatten()
        .filter_map(|entry| {
            let entry_name = entry.file_name().to_string_lossy().to_lowercase();
            let distance = edit_distance(&entry_name, &name.to_lowercase());
            (distance <= limit).then(|| (distance, directory.join(entry.file_name())))
        })
        .collect()
}

// Files or directories under root whose name is close to `name`, breadth first within
// SEARCH_MAX_DEPTH and SEARCH_MAX_ENTRIES.
fn close_names_under(root: &Path, name: &str) -> Vec<(usize, PathBuf)> {
    let limit = allowed_distance(name);
    let lower = name.to_lowercase();
    let mut hits = Vec::new();
    let mut seen = 0;
    let mut level = vec![root.to_path_buf()];
    let mut depth = 0;
    while depth <= SEARCH_MAX_DEPTH && !level.is_empty() && seen < SEARCH_MAX_ENTRIES {
        let mut next = Vec::new();
        for directory in &level {
            let Ok(entries) = std::fs::read_dir(directory) else { continue };
            for entry in entries.flatten() {
                seen += 1;
                if seen > SEARCH_MAX_ENTRIES {
                    break;
                }
                let entry_name = entry.file_name().to_string_lossy().into_owned();
                if edit_distance(&entry_name.to_lowercase(), &lower) <= limit {
                    hits.push((edit_distance(&entry_name.to_lowercase(), &lower), directory.join(&entry_name)));
                }
                let is_dir = entry.file_type().is_ok_and(|kind| kind.is_dir());
                if is_dir && !SEARCH_SKIP_DIRS.contains(&entry_name.as_str()) {
                    next.push(directory.join(&entry_name));
                }
            }
        }
        level = next;
        depth += 1;
    }
    hits
}

// Up to MAX_SUGGESTIONS existing paths for one missing path, relative to cwd when under it. First
// the directory the path pointed into (a typo in the last part), then a name search under cwd
// (right file, wrong directory). Ties go to the shorter path.
fn suggestions_for(missing: &str, command: &str, cwd: &Path) -> Vec<String> {
    let Some(name) = Path::new(missing).file_name().map(|name| name.to_string_lossy().into_owned()) else { return Vec::new() };
    let mut scored = Vec::new();
    for base_dir in candidate_base_dirs(command, cwd) {
        let absolute = resolve_from(&base_dir, missing);
        if absolute.exists() {
            // It exists from here; the failure was about something else.
            return Vec::new();
        }
        if let Some(parent) = absolute.parent() {
            scored.extend(close_entries_in(parent, &name));
        }
    }
    if scored.len() < MAX_SUGGESTIONS {
        scored.extend(close_names_under(cwd, &name));
    }
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.as_os_str().len().cmp(&b.1.as_os_str().len())));
    let mut shown: Vec<String> = Vec::new();
    let mut unique = HashSet::new();
    for (_, absolute) in scored {
        let display = match absolute.strip_prefix(cwd) {
            Ok(relative) if !relative.as_os_str().is_empty() => relative.display().to_string(),
            _ => absolute.display().to_string(),
        };
        if unique.insert(display.clone()) {
            shown.push(display);
        }
        if shown.len() == MAX_SUGGESTIONS {
            break;
        }
    }
    shown
}

// Input: the text a command produced, the command and the cwd. Output: hint lines to append (maybe
// none), one per missing path that has close existing paths.
pub fn hints_for(output: &str, command: &str, cwd: &Path) -> Vec<String> {
    missing_paths_in(output)
        .into_iter()
        .take(MAX_MISSING_PATHS)
        .filter_map(|missing| {
            let suggestions = suggestions_for(&missing, command, cwd);
            (!suggestions.is_empty()).then(|| format!("{missing} does not exist; did you mean: {}", suggestions.join(", ")))
        })
        .collect()
}
