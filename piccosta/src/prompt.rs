// The system prompt: piccosta's own text, in sections.
//
// The sections, and which AGENTS.md / CLAUDE.md files load, are based on pi 1.4.0 (MIT, see
// LICENSE.md): packages/coding-agent/src/core/system-prompt.ts and resource-loader.ts. None of
// pi's wording is kept. A session's system message stores the sections, rendered joined by blank
// lines; each one but the preamble is wrapped in a tag of its own name, so a later change can name
// what it replaces. In order: a short preamble, the tools (each tool's own line), the rules (each
// tool's guidelines and piccosta's own), the addendum (APPEND_SYSTEM.md, --append-system-prompt,
// a manager's part), the project's context files, the cwd, and the constitution when it's on.
//
// - A line in a context file that is only `@some/file.md` is replaced by that file, the way
//   Claude Code does. Paths are relative to the importing file, `~/` is home. One level deep: an
//   imported file's own imports stay as written. A file never imports itself or a context file
//   already in the prompt, and an import that can't be read keeps its `@` line, so nothing
//   disappears silently.
// - The constitution (prompts/constitution.md) goes last, as plain text after the tagged sections,
//   once, the same every request, so after the first call it comes from the prompt cache.

use crate::tools::Tool;
use indexmap::IndexMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// The opening line, when no SYSTEM.md replaces it.
const PREAMBLE: &str = "You are a coding agent in piccosta. You work in the user's project: read its code, run its commands and change its files.";
// The context files a directory can hold, first one found wins (resource-loader.ts:185).
const CONTEXT_FILE_NAMES: [&str; 5] = ["AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"];
// The largest import inlined: pi's read cap. Anything the model couldn't read in one read call
// doesn't belong in every request either.
const IMPORT_MAX_BYTES: usize = crate::tools::truncate::DEFAULT_MAX_BYTES;

// How recently a context file may have been written for its new text to be taken: a save that
// truncates the file and then writes it is mid-way for milliseconds, and a read then would put half
// a file in the prompt and the whole one in the next, two rewrites for one edit. Two seconds is far
// past any such save and well short of the time to the next message.
const SAVE_SETTLE: std::time::Duration = std::time::Duration::from_secs(2);

// The constitution is off unless asked for: benchmarks gave mixed results with it, and it is about
// 2,200 tokens on every request (2026-11-02). PICCOSTA_CONSTITUTION holds the text itself (how the
// bench passes it into a sandbox), PICCOSTA_CONSTITUTION_FILE a path to it, or "built-in" for
// prompts/constitution.md, the text this binary was built with. The `constitution` setting does the
// same where no variable is set: true for the built-in text, a path for a file.
const CONSTITUTION_TEXT_ENV: &str = "PICCOSTA_CONSTITUTION";
const CONSTITUTION_FILE_ENV: &str = "PICCOSTA_CONSTITUTION_FILE";
const BUILT_IN_CONSTITUTION: &str = include_str!("../prompts/constitution.md");
const CONSTITUTION_BUILT_IN: &str = "built-in";

pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
}

// What reading the context files ran into: the files that couldn't be read, and whether one was
// written so recently that it may be mid-save.
#[derive(Default)]
struct Reads {
    failures: Vec<String>,
    unsettled: bool,
}

impl Reads {
    // A file's bytes, or None when it's not there (only a not-found says so) or can't be read (a
    // failure). One written under SAVE_SETTLE ago is read, and marks the reads unsettled.
    fn read(&mut self, path: &Path) -> Option<Vec<u8>> {
        match std::fs::metadata(path).and_then(|metadata| Ok((metadata.modified()?, std::fs::read(path)?))) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Ok((written, bytes)) => {
                self.unsettled |= written.elapsed().is_ok_and(|age| age < SAVE_SETTLE);
                Some(bytes)
            }
            Err(error) => {
                self.failures.push(format!("Could not read {}: {error}", path.display()));
                None
            }
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
    // Real paths, so two links to one file load it once.
    let mut seen = Vec::new();
    if let Some(global) = read_context_file(agent_dir, reads) {
        seen.push(crate::tools::paths::canonical(&global.path));
        files.push(global);
    }
    let shadowed = shadowed_context_file(cwd, reads);
    let mut ancestors = Vec::new();
    let mut dir = cwd.to_path_buf();
    loop {
        if let Some(file) = read_context_file(&dir, reads) {
            let real_dir = crate::tools::paths::canonical(&dir);
            let is_shadowed = shadowed.as_ref().is_some_and(|shadowed| file.path.file_name().is_some_and(|name| real_dir.join(name) == *shadowed));
            let real = crate::tools::paths::canonical(&file.path);
            if !is_shadowed && !seen.contains(&real) {
                seen.push(real);
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
    let path = crate::tools::paths::expand_home(raw);
    let path = if path.is_absolute() { path } else { importing_file.parent().unwrap_or(Path::new("/")).join(path) };
    crate::tools::paths::canonical(&crate::tools::paths::resolve(&path.to_string_lossy(), Path::new("/")))
}

// One context file's text with each import line replaced by the file it names (lean.ts:77-101).
// Input: the file, the real paths of every context file already in the prompt, and of every file
// already imported (by this file or an earlier one), which isn't imported again.
fn expand_imports(file: &ContextFile, in_prompt: &[PathBuf], imported: &mut Vec<PathBuf>, reads: &mut Reads) -> String {
    let lines: Vec<String> = file
        .content
        .split('\n')
        .map(|line| {
            let trimmed = crate::js::trim(line);
            let Some(raw) = trimmed.strip_prefix('@').filter(|raw| !raw.is_empty() && !raw.contains(crate::js::is_whitespace)) else {
                return line.to_string();
            };
            let target = import_target(raw, &file.path);
            if in_prompt.contains(&target) || imported.contains(&target) {
                return line.to_string();
            }
            if target.is_dir() {
                return line.to_string();
            }
            let Some(bytes) = reads.read(&target) else { return line.to_string() };
            imported.push(target);
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
pub fn constitution(model_name: &str, setting: Option<&serde_json::Value>) -> Result<Option<String>, String> {
    let file = std::env::var(CONSTITUTION_FILE_ENV).ok().filter(|path| !path.is_empty()).or_else(|| match setting {
        Some(serde_json::Value::Bool(true)) => Some(CONSTITUTION_BUILT_IN.to_string()),
        Some(serde_json::Value::String(path)) if !path.is_empty() => Some(path.clone()),
        _ => None,
    });
    let text = match (std::env::var(CONSTITUTION_TEXT_ENV), file) {
        (Ok(text), _) if !text.trim().is_empty() => text,
        (_, Some(path)) if path == CONSTITUTION_BUILT_IN => BUILT_IN_CONSTITUTION.to_string(),
        (_, Some(path)) => std::fs::read_to_string(&path).map_err(|error| format!("constitution {path}: {error}"))?,
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
    // Another agent owns its tools. Keep our instructions, not descriptions of tools it can't call.
    pub external_tools: bool,
    // The agent dir's SYSTEM.md (replaces the preamble, tool list and rules) and APPEND_SYSTEM.md.
    pub custom_prompt: Option<String>,
    pub append_prompt: Option<String>,
    pub constitution: Option<String>,
    // AGENTS.md / CLAUDE.md files go in unless --no-context-files.
    pub context_files: bool,
    // The project_context section the session holds, if it has one.
    pub held_context: Option<String>,
}

// The rules: each tool's guidelines in tool order, then piccosta's own, each once.
fn rules(tools: &[Arc<dyn Tool>]) -> String {
    let mut rules: Vec<&str> = Vec::new();
    // People watch a session work: a run of tool calls with no word between them reads as a hang.
    let talk = "Talk to the user as you work: before a run of tool calls say in a sentence what you're doing, after it what you found, and when they write mid-task, answer them first";
    // What Claude Code's own prompt asks and piccosta sends instead of it: read before changing,
    // check by running, finish the task, and treat what others report as unchecked.
    let thorough = "Read the code and files a change touches before you make it, and don't guess at what you haven't read. Check your work by running it (build it, run the command, look at the output), and say what you checked and what you didn't. Finish the task: don't stop at the first answer that looks right, and when something fails, find out why. What a tool, subagent or other session reports is unchecked until you've checked it";
    // Late in long sessions models write notes, reports and plans nobody asked for, and leave them.
    let files = "Don't create documents (notes, reports, plans, summaries) unless the user asked for one; put what you found in your reply. Delete files you made for your own use before you finish";
    for rule in tools.iter().flat_map(|tool| tool.prompt_guidelines().iter().copied()).chain([thorough, talk, files]) {
        if !rules.contains(&rule) {
            rules.push(rule);
        }
    }
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
            if !inputs.external_tools { raw.insert("tools".into(), list); }
            raw.insert("rules".into(), rules(if inputs.external_tools { &[] } else { inputs.tools }));
        }
    }
    if let Some(append) = inputs.append_prompt.as_ref().filter(|text| !text.is_empty()) {
        raw.insert("addendum".into(), append.clone());
    }
    let mut reads = Reads::default();
    let project = inputs.context_files.then(|| project_context(&expanded(inputs.cwd, inputs.agent_dir, &mut reads))).flatten();
    // A file that can't be read, or may be mid-save: a session that holds instructions keeps them
    // as they were, rather than lose that file's or take half of it.
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

// Every context file for the session (path, text), imports expanded.
fn expanded(cwd: &Path, agent_dir: &Path, reads: &mut Reads) -> Vec<(PathBuf, String)> {
    let files = context_files(cwd, agent_dir, reads);
    let in_prompt: Vec<PathBuf> = files.iter().map(|file| crate::tools::paths::canonical(&file.path)).collect();
    let mut imported = Vec::new();
    files.iter().map(|file| (file.path.clone(), expand_imports(file, &in_prompt, &mut imported, reads))).collect()
}

// The project_context section's text for those files, or None when there are none.
fn project_context(files: &[(PathBuf, String)]) -> Option<String> {
    let mut parts = vec!["The user's instructions for this project, from its AGENTS.md and CLAUDE.md files, outermost first:".to_string()];
    for (path, text) in files {
        parts.push(format!("<project_instructions path=\"{}\">\n{text}\n{FILE_END}", path.display()));
    }
    (!files.is_empty()).then(|| parts.join("\n\n"))
}

// A running session's project_context section as its context files are now (agent.rs
// prompt_update): Some(None) when there are none, and None when one can't be read whole now (a
// failure, or written in the last SAVE_SETTLE, maybe mid-save).
pub fn context_now(cwd: &Path, agent_dir: &Path) -> Option<Option<String>> {
    let mut reads = Reads::default();
    let files = expanded(cwd, agent_dir, &mut reads);
    (reads.failures.is_empty() && !reads.unsettled).then(|| project_context(&files).map(|project| tagged("project_context", &project)))
}

// How each context file's part of the project_context section ends.
pub const FILE_END: &str = "</project_instructions>";

// The instructions a project_context section holds: each file's path and text, below the header
// line, which is piccosta's wording. "" for none.
pub fn instructions(section: Option<&str>) -> &str {
    section.and_then(|text| text.split_once("\n\n")).map_or("", |(_, files)| files)
}

// Put a project_context section in a prompt's sections, or take it out (None). It keeps its place,
// and a new one goes where sections() puts it, just before cwd.
pub fn set_project_context(sections: &mut IndexMap<String, String>, project: Option<String>) {
    match (project, sections.get_index_of("project_context")) {
        (Some(text), Some(index)) => sections[index] = text,
        (Some(text), None) => {
            let before_cwd = sections.get_index_of("cwd").unwrap_or(sections.len());
            sections.shift_insert(before_cwd, "project_context".into(), text);
        }
        (None, _) => {
            sections.shift_remove("project_context");
        }
    }
}

// The agent dir's SYSTEM.md and APPEND_SYSTEM.md, when present (resource-loader.ts:1213-1239).
// Only the agent dir's count: piccosta has no project trust to let a repo replace its prompt.
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
