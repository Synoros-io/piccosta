// The system prompt: pi's sections, with piccosta's changes made as the sections are built.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/system-prompt.ts
// (the sections) and resource-loader.ts (which AGENTS.md / CLAUDE.md files load). The sections are
// what a session's system message stores, rendered joined by blank lines; each one but the
// preamble is wrapped in a tag of its own name, so a later change can name what it replaces.
//
// piccosta's changes, which pi did as extensions rewriting the finished prompt:
// - lean: no "Pi documentation" section. It points the model at pi's own README and examples,
//   which matter only when you're building pi, and it was about 40% of pi 0.85's system text
//   (bench/results/overhead/report.md).
// - lean: a line in a context file that is only `@some/file.md` is replaced by that file, the way
//   Claude Code does. Paths are relative to the importing file, `~/` is home. One level deep: an
//   imported file's own imports stay as written. A file never imports itself or a context file
//   already in the prompt, and an import that can't be read keeps its `@` line, so nothing
//   disappears silently.
// - constitution: piccosta's constitution (prompts/constitution.md) goes last, as plain text after the tagged sections, once,
//   the same every request, so after the first call it comes from the prompt cache.
// In pi these forced the whole prompt every run, so the session stored one prompt and sent
// another; built here as sections, the session holds exactly what the model gets.

use crate::tools::Tool;
use indexmap::IndexMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// pi's opening line, when no SYSTEM.md replaces it (system-prompt.ts:156-157).
const PREAMBLE: &str = "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.";
// The context files a directory can hold, first one found wins (resource-loader.ts:185).
const CONTEXT_FILE_NAMES: [&str; 5] = ["AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"];
// The largest import inlined: pi's read cap. Anything the model couldn't read in one read call
// doesn't belong in every request either.
const IMPORT_MAX_BYTES: usize = crate::tools::truncate::DEFAULT_MAX_BYTES;

// The sections built from the session's own files rather than piccosta's text: its AGENTS.md /
// CLAUDE.md files and its directory. A running session takes a change to these at once; a change to
// any other section (a new build's wording, the constitution, SYSTEM.md, APPEND_SYSTEM.md, the
// manager's part) waits for its next compaction or model change (agent.rs system_update).
pub const INPUT_SECTIONS: [&str; 2] = ["project_context", "cwd"];
// How recently a context file may have been written for its new text to be taken: a save that
// truncates the file and then writes it is mid-way for milliseconds, and a read then would put half
// a file in the prompt and the whole one in the next, two rewrites for one edit. Two seconds is far
// past any such save and well short of the time to the next message.
const SAVE_SETTLE: std::time::Duration = std::time::Duration::from_secs(2);

// The last good bytes of every context file and import read in this piccosta process, by path.
static SNAPSHOTS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<PathBuf, Vec<u8>>>> = std::sync::LazyLock::new(Default::default);

// The constitution is off unless asked for: benchmarks gave mixed results with it, and it is about
// 2,200 tokens on every request (2026-11-02). PICCOSTA_CONSTITUTION holds the text itself (how the
// bench passes it into a sandbox), PICCOSTA_CONSTITUTION_FILE a path to it, or "built-in" for
// prompts/constitution.md, the text this binary was built with.
const CONSTITUTION_TEXT_ENV: &str = "PICCOSTA_CONSTITUTION";
const CONSTITUTION_FILE_ENV: &str = "PICCOSTA_CONSTITUTION_FILE";
const BUILT_IN_CONSTITUTION: &str = include_str!("../prompts/constitution.md");
const CONSTITUTION_BUILT_IN: &str = "built-in";

pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
}

// What reading the context files ran into: the files that couldn't be read and had no last good
// bytes to stand in, and whether one was still being saved with none (piccosta just started).
#[derive(Default)]
struct Reads {
    failures: Vec<String>,
    unsettled: bool,
}

impl Reads {
    // A file's bytes, or None when it's not there (only a not-found says so). A file that can't be
    // read, or was written under SAVE_SETTLE ago, gives its last good bytes instead, so each file
    // keeps its own and the rest still update; one with none that can't be read is a failure.
    fn read(&mut self, path: &Path) -> Option<Vec<u8>> {
        let read = std::fs::metadata(path).and_then(|metadata| Ok((metadata.modified()?, std::fs::read(path)?)));
        let mut snapshots = SNAPSHOTS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        match read {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                snapshots.remove(path);
                None
            }
            Ok((written, bytes)) if !written.elapsed().is_ok_and(|age| age < SAVE_SETTLE) => {
                snapshots.insert(path.to_path_buf(), bytes.clone());
                Some(bytes)
            }
            // Still being saved: its last good bytes, or with none these, marked unsettled, so a
            // session that holds instructions keeps them and a new one starts with what's there.
            Ok((_, bytes)) => snapshots.get(path).cloned().or_else(|| {
                self.unsettled = true;
                Some(bytes)
            }),
            Err(error) => snapshots.get(path).cloned().or_else(|| {
                self.failures.push(format!("Could not read {}: {error}", path.display()));
                None
            }),
        }
    }
}

fn read_context_file(dir: &Path, reads: &mut Reads) -> Option<ContextFile> {
    for name in CONTEXT_FILE_NAMES {
        let path = dir.join(name);
        if path.is_dir() {
            continue;
        }
        if let Some(bytes) = reads.read(&path) {
            let text = String::from_utf8_lossy(&bytes);
            let content = text.strip_prefix('\u{feff}').unwrap_or(&text).to_string();
            return Some(ContextFile { path, content });
        }
    }
    None
}

// Where a checkout's git data is: the directory holding `.git`, and the shared git dir (the main
// repo's for a linked worktree). None outside a repo (footer-data-provider.ts:16-48).
fn git_paths(cwd: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut dir = cwd.to_path_buf();
    loop {
        let git = dir.join(".git");
        if git.is_file() {
            let content = std::fs::read_to_string(&git).ok()?;
            let target = content.trim().strip_prefix("gitdir: ")?;
            let git_dir = crate::tools::paths::resolve(target.trim(), &dir);
            if !git_dir.join("HEAD").exists() {
                return None;
            }
            let common = match std::fs::read_to_string(git_dir.join("commondir")) {
                Ok(common) => crate::tools::paths::resolve(common.trim(), &git_dir),
                Err(_) => git_dir,
            };
            return Some((dir, common));
        }
        if git.is_dir() {
            return git.join("HEAD").exists().then_some((dir, git));
        }
        if !dir.pop() {
            return None;
        }
    }
}

// A linked worktree nested inside its main checkout (`git worktree add .scratch/lane`) has its own
// copy of the repo's AGENTS.md; loading both would give the model the same instructions twice.
// Output: the main checkout's context file that the worktree's copy stands in for
// (resource-loader.ts:203-227).
fn shadowed_context_file(cwd: &Path, reads: &mut Reads) -> Option<PathBuf> {
    let (repo_dir, common_git_dir) = git_paths(cwd)?;
    let common_git_dir = crate::tools::paths::canonical(&common_git_dir);
    let worktree_root = crate::tools::paths::canonical(&repo_dir);
    let main_root = common_git_dir.parent()?.to_path_buf();
    if worktree_root == main_root || !worktree_root.starts_with(&main_root) {
        return None;
    }
    if crate::tools::paths::canonical(&main_root.join(".git")) != common_git_dir {
        return None;
    }
    let worktree_file = read_context_file(&worktree_root, reads)?;
    Some(main_root.join(worktree_file.path.file_name()?))
}

// The context files for a session (resource-loader.ts:229-273): the agent dir's own first, then
// every ancestor directory's, outermost first, ending with the cwd's.
fn context_files(cwd: &Path, agent_dir: &Path, reads: &mut Reads) -> Vec<ContextFile> {
    let mut files = Vec::new();
    let mut seen = Vec::new();
    if let Some(global) = read_context_file(agent_dir, reads) {
        seen.push(global.path.clone());
        files.push(global);
    }
    let shadowed = shadowed_context_file(cwd, reads);
    let mut ancestors = Vec::new();
    let mut dir = cwd.to_path_buf();
    loop {
        if let Some(file) = read_context_file(&dir, reads) {
            let real_dir = crate::tools::paths::canonical(&dir);
            let is_shadowed = shadowed.as_ref().is_some_and(|shadowed| file.path.file_name().is_some_and(|name| real_dir.join(name) == *shadowed));
            if !is_shadowed && !seen.contains(&file.path) {
                seen.push(file.path.clone());
                ancestors.push(file);
            }
        }
        if !dir.pop() {
            break;
        }
    }
    ancestors.reverse();
    files.extend(ancestors);
    files
}

fn import_target(raw: &str, importing_file: &Path) -> PathBuf {
    let home = || PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let path = if raw == "~" {
        home()
    } else if let Some(rest) = raw.strip_prefix("~/") {
        home().join(rest)
    } else if Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        importing_file.parent().unwrap_or(Path::new("/")).join(raw)
    };
    crate::tools::paths::canonical(&crate::tools::paths::resolve(&path.to_string_lossy(), Path::new("/")))
}

// One context file's text with each import line replaced by the file it names (lean.ts:77-101).
// Input: the file, and the real paths of every context file already in the prompt.
fn expand_imports(file: &ContextFile, in_prompt: &[PathBuf], reads: &mut Reads) -> String {
    let mut imported_here: Vec<PathBuf> = Vec::new();
    let lines: Vec<String> = file
        .content
        .split('\n')
        .map(|line| {
            let trimmed = crate::js::trim(line);
            let Some(raw) = trimmed.strip_prefix('@').filter(|raw| !raw.is_empty() && !raw.contains(crate::js::is_whitespace)) else {
                return line.to_string();
            };
            let target = import_target(raw, &file.path);
            if in_prompt.contains(&target) || imported_here.contains(&target) {
                return line.to_string();
            }
            if target.is_dir() {
                return line.to_string();
            }
            let Some(bytes) = reads.read(&target) else { return line.to_string() };
            imported_here.push(target);
            let mut text = if bytes.len() > IMPORT_MAX_BYTES {
                format!("{}\n[{raw} cut at {IMPORT_MAX_BYTES} bytes; read the file for the rest]", String::from_utf8_lossy(&bytes[..IMPORT_MAX_BYTES]))
            } else {
                String::from_utf8_lossy(&bytes).into_owned()
            };
            if text.ends_with('\n') {
                text.pop();
            }
            text
        })
        .collect();
    lines.join("\n")
}

// The constitution text with its placeholders filled, or None when it's turned off. A file that
// can't be read is an error: a run that thinks it has a constitution and doesn't is worse.
pub fn constitution(model_name: &str) -> Result<Option<String>, String> {
    let text = match (std::env::var(CONSTITUTION_TEXT_ENV), std::env::var(CONSTITUTION_FILE_ENV)) {
        (Ok(text), _) if !text.trim().is_empty() => text,
        (_, Ok(path)) if path == CONSTITUTION_BUILT_IN => BUILT_IN_CONSTITUTION.to_string(),
        (_, Ok(path)) if !path.is_empty() => std::fs::read_to_string(&path).map_err(|error| format!("{CONSTITUTION_FILE_ENV}={path}: {error}"))?,
        _ => return Ok(None),
    };
    if text.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(crate::js::trim(&text.replace("{modelname}", model_name)).to_string()))
}

// Everything the prompt is built from.
pub struct PromptInputs<'a> {
    pub cwd: &'a Path,
    pub agent_dir: &'a Path,
    pub tools: &'a [Arc<dyn Tool>],
    // The agent dir's SYSTEM.md (replaces pi's preamble, tool list and rules) and APPEND_SYSTEM.md.
    pub custom_prompt: Option<String>,
    pub append_prompt: Option<String>,
    pub constitution: Option<String>,
    // AGENTS.md / CLAUDE.md files go in unless --no-context-files.
    pub context_files: bool,
    // The project_context section the session holds, if it has one.
    pub held_context: Option<String>,
}

// pi's rules: one line for shell file work, each tool's guidelines in tool order, then two of its
// own, each once (system-prompt.ts:93-128).
fn rules(tools: &[Arc<dyn Tool>]) -> String {
    let mut rules: Vec<String> = Vec::new();
    let mut add = |rule: &str| {
        let rule = crate::js::trim(rule);
        if !rule.is_empty() && !rules.iter().any(|seen| seen == rule) {
            rules.push(rule.to_string());
        }
    };
    let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
    if names.contains(&"bash") && !names.iter().any(|name| matches!(*name, "grep" | "find" | "ls")) {
        add("Use bash for file operations like ls, rg, find");
    }
    for tool in tools {
        for guideline in tool.prompt_guidelines() {
            add(guideline);
        }
    }
    // People watch a session work: a run of tool calls with no word between them reads as a hang.
    add("Talk to the user as you work: before a run of tool calls say in a sentence what you're doing, after it what you found, and when they write mid-task, answer them first");
    add("Be concise in your responses");
    add("Show file paths clearly when working with files");
    rules.iter().map(|rule| format!("- {rule}")).collect::<Vec<_>>().join("\n")
}

// The prompt's sections, in order, as a session's system message stores them.
pub fn sections(inputs: &PromptInputs) -> IndexMap<String, String> {
    let mut raw: IndexMap<String, String> = IndexMap::new();
    match &inputs.custom_prompt {
        Some(custom) => {
            raw.insert("preamble".into(), custom.clone());
        }
        None => {
            raw.insert("preamble".into(), PREAMBLE.into());
            let listed: Vec<String> =
                inputs.tools.iter().filter_map(|tool| tool.prompt_snippet().map(|snippet| format!("- {}: {snippet}", tool.name()))).collect();
            let list = if listed.is_empty() { "(none)".to_string() } else { listed.join("\n") };
            raw.insert("tools".into(), format!("{list}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project."));
            raw.insert("rules".into(), rules(inputs.tools));
        }
    }
    if let Some(append) = inputs.append_prompt.as_ref().filter(|text| !text.is_empty()) {
        raw.insert("addendum".into(), append.clone());
    }
    let mut reads = Reads::default();
    let project = inputs.context_files.then(|| project_context(inputs.cwd, inputs.agent_dir, &mut reads)).flatten();
    // A file that can't be read with no last good bytes in this process (piccosta just started): a
    // session that holds instructions keeps them, rather than lose that file's.
    let held = inputs.held_context.as_deref().and_then(|held| held.strip_prefix("<project_context>\n")?.strip_suffix("\n</project_context>"));
    let project = if reads.failures.is_empty() && !reads.unsettled { project } else { held.map(String::from).or(project) };
    if let Some(project) = project {
        raw.insert("project_context".into(), project);
    }
    for failure in reads.failures {
        eprintln!("Warning: {failure}");
    }
    raw.insert("cwd".into(), inputs.cwd.to_string_lossy().replace('\\', "/"));

    let mut sections = IndexMap::new();
    for (name, content) in raw {
        let value = if name == "preamble" { content } else { tagged(&name, &content) };
        sections.insert(name, value);
    }
    if let Some(constitution) = &inputs.constitution {
        sections.insert("constitution".into(), constitution.clone());
    }
    sections
}

// A section as the prompt carries it: its text inside a tag named for it.
fn tagged(name: &str, content: &str) -> String {
    format!("<{name}>\n{content}\n</{name}>")
}

// The project_context section's text: every context file for the session, imports expanded, or
// None when there are none.
fn project_context(cwd: &Path, agent_dir: &Path, reads: &mut Reads) -> Option<String> {
    let files = context_files(cwd, agent_dir, reads);
    let in_prompt: Vec<PathBuf> = files.iter().map(|file| crate::tools::paths::canonical(&file.path)).collect();
    let mut parts = vec!["Project-specific instructions and guidelines:".to_string()];
    for file in &files {
        parts.push(format!("<project_instructions path=\"{}\">\n{}\n</project_instructions>", file.path.display(), expand_imports(file, &in_prompt, reads)));
    }
    (!files.is_empty()).then(|| parts.join("\n\n"))
}

// Read a running session's context files again into its sections, so an AGENTS.md edited, added or
// removed since it opened reaches the next request. The section keeps its place, and a new one goes
// where sections() puts it, just before cwd. A file that can't be read, or is still being saved,
// goes in as it last read whole (Reads::read); one that never did leaves the section as it was,
// since dropping instructions that are still there would rewrite the cache now and again once
// they read. The next run reads them again.
pub fn reload_context_files(sections: &mut IndexMap<String, String>, cwd: &Path, agent_dir: &Path) {
    let mut reads = Reads::default();
    let project = project_context(cwd, agent_dir, &mut reads);
    if !reads.failures.is_empty() || reads.unsettled {
        return;
    }
    match project {
        Some(project) => {
            let text = tagged("project_context", &project);
            match sections.get_index_of("project_context") {
                Some(index) => sections[index] = text,
                None => {
                    let before_cwd = sections.get_index_of("cwd").unwrap_or(sections.len());
                    sections.shift_insert(before_cwd, "project_context".into(), text);
                }
            }
        }
        None => {
            sections.shift_remove("project_context");
        }
    }
}

// The agent dir's SYSTEM.md and APPEND_SYSTEM.md, when present (resource-loader.ts:1213-1239).
// pi also reads them from a trusted project's .pi/; piccosta has no project trust, so only the
// agent dir's count.
pub fn custom_prompts(agent_dir: &Path) -> (Option<String>, Option<String>) {
    let read = |name: &str| std::fs::read_to_string(agent_dir.join(name)).ok();
    (read("SYSTEM.md"), read("APPEND_SYSTEM.md"))
}

// What /init asks for (the TUI and ACP both offer it): AGENTS.md, direction.md and ledger.md.
pub const INIT_PROMPT: &str = "Set up this repository's agent docs. Read the code and any existing docs first. Write AGENTS.md: the rules for working in this repo (how to build, run and check it, its conventions, what not to touch), short and specific to this repo, with CLAUDE.md as a symlink to it. Write direction.md: the current goal and what is planned for later. Start ledger.md: an append-only log of what the user asked and what happened, a few lines per session. Update any of these that already exist instead of replacing them, and create no other documents.";
const REVIEW_PROMPT: &str = "Use the subagent tool with the review job to review {target}. Have it report problems it can show, each with the file and line, and change nothing. Then tell me what it found.";
const REVIEW_DEFAULT: &str = "the current changes in this repository (git diff, including untracked files)";

// What /review sends: a review of `target`, or of the working tree's changes when it's empty,
// handed to the review job.
pub fn review_prompt(target: &str) -> String {
    REVIEW_PROMPT.replace("{target}", if target.trim().is_empty() { REVIEW_DEFAULT } else { target.trim() })
}

// The commands every frontend turns into a prompt, with what /help and rpc's get_commands say of
// them.
pub const PROMPT_COMMANDS: [(&str, &str); 2] = [
    ("init", "have the model write AGENTS.md, direction.md and ledger.md"),
    ("review", "hand a review of the changes to the review job"),
];

// A typed `/init` or `/review [what]` as the prompt it stands for; None for anything else.
pub fn command_prompt(text: &str) -> Option<String> {
    let line = text.trim().strip_prefix('/')?;
    let (name, argument) = line.split_once(' ').map_or((line, ""), |(name, rest)| (name, rest.trim()));
    match name {
        "init" => Some(INIT_PROMPT.into()),
        "review" => Some(review_prompt(argument)),
        _ => None,
    }
}
