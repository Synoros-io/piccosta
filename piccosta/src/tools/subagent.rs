// Subagents: a task handed to another agent, on a model the job picks, not the model handing it.
//
// Each job has a list of allowed models, not one model per tier, and the model handing the work
// over doesn't choose among them. So the
// model calling `subagent` names a job and a task, and the job's own list says which models may
// do it: the first one with a login runs it, and when that run fails the next one takes it over.
// The lists are the user's; nothing the calling model sends can change them.
//
// A job is a role file: `<agent dir>/agents/<name>.md` when there is one, else the built-in of
// that name (piccosta/agents/). Frontmatter keys: description, models (provider/id, comma
// separated, in the order to try them; pi-subagents' single `model` also works; none means the
// calling session's own model), or `use` (a kind of work on the model scale, scale.rs, whose models
// and levels it takes, in the order usage allows at the time), thinking, tools (comma separated; every tool when absent) and
// disabled. The body is appended to the subagent's system prompt.
//
// A run is a child piccosta in print mode, started through bash's runner, so it is a shell command
// like any other the session runs: its output streams to the UI, an abort kills it, one still going
// at the soft limit becomes a background job whose message arrives when it ends, and print mode
// waits for it. The child's session goes in a `subagents` folder next to the parent's, out of the
// parent folder's list. A subagent gets no subagent tool of its own (CHILD_ENV).

use super::{Tool, ToolContext, ToolFuture};
use crate::auth::{credential_for, CredentialPaths};
use crate::message::ToolDefinition;
use crate::model::ModelRegistry;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// The built-in jobs, a monitor among them; their models come from the model scale. The advisor
// (research design or rescue) ships disabled, since it shouldn't be called on its own.
const BUILTIN: [(&str, &str); 6] = [
    ("search", include_str!("../../agents/search.md")),
    ("scout", include_str!("../../agents/scout.md")),
    ("worker", include_str!("../../agents/worker.md")),
    ("reviewer", include_str!("../../agents/reviewer.md")),
    ("monitor", include_str!("../../agents/monitor.md")),
    ("advisor", include_str!("../../agents/advisor.md")),
];
// Set in a subagent's environment (to its job's name); a session that has it offers no subagent tool.
pub const CHILD_ENV: &str = "PICCOSTA_SUBAGENT";
// The tool's name, as the model calls it.
pub const NAME: &str = "subagent";
// Linux's cap on one argument (MAX_ARG_STRLEN), which the whole command `bash -c` runs must fit under.
const MAX_COMMAND_BYTES: usize = 128 * 1024;

struct Job {
    name: String,
    description: String,
    models: Vec<String>,
    // A kind of work on the model scale, whose models it runs on instead.
    scale_use: Option<String>,
    thinking: Option<String>,
    tools: Option<String>,
    prompt: String,
}

pub struct SubagentTool {
    jobs: Vec<Job>,
    agent_dir: PathBuf,
}

// Input: a job's name and its role file. Output: the job, or None when it is disabled or the file
// has no frontmatter.
fn parse(name: &str, text: &str) -> Option<Job> {
    let rest = text.strip_prefix("---\n")?;
    let (head, body) = rest.split_once("\n---\n").or_else(|| rest.strip_suffix("\n---").map(|head| (head, "")))?;
    let fields: HashMap<&str, &str> = head.lines().filter_map(|line| line.split_once(':')).map(|(key, value)| (key.trim(), value.trim())).collect();
    if fields.get("disabled") == Some(&"true") {
        return None;
    }
    let list = |key: &str| -> Vec<String> { fields.get(key).into_iter().flat_map(|value| value.split(',')).map(str::trim).filter(|item| !item.is_empty()).map(String::from).collect() };
    let models = if fields.contains_key("models") { list("models") } else { list("model") };
    Some(Job {
        name: name.to_string(),
        description: fields.get("description").unwrap_or(&"").to_string(),
        models,
        scale_use: fields.get("use").filter(|name| !name.is_empty()).map(|name| name.to_string()),
        thinking: fields.get("thinking").filter(|level| !level.is_empty()).map(|level| level.to_string()),
        tools: fields.get("tools").map(|tools| tools.to_string()),
        prompt: body.trim().to_string(),
    })
}

// Text as one single-quoted shell word.
fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

impl SubagentTool {
    // The jobs for an agent dir: each built-in one unless a role file of its name replaces it, then
    // the agent dir's other role files by name. None when no job is enabled (no tool then).
    pub fn load(agent_dir: &Path) -> Option<SubagentTool> {
        let mut files: Vec<(String, String)> = std::fs::read_dir(agent_dir.join("agents"))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let name = path.file_stem()?.to_str()?.to_string();
                (path.extension()? == "md").then(|| Some((name, std::fs::read_to_string(&path).ok()?)))?
            })
            .collect();
        files.sort();
        let mut roles: Vec<(String, String)> = BUILTIN.iter().map(|(name, text)| (name.to_string(), text.to_string())).collect();
        for (name, text) in files {
            match roles.iter_mut().find(|(builtin, _)| *builtin == name) {
                Some(role) => role.1 = text,
                None => roles.push((name, text)),
            }
        }
        let jobs: Vec<Job> = roles.iter().filter_map(|(name, text)| parse(name, text)).collect();
        (!jobs.is_empty()).then(|| SubagentTool { jobs, agent_dir: agent_dir.to_path_buf() })
    }

    // The job's models that are in the registry and have a login, in its order (the calling
    // session's own model for a job that lists none), and a line for each one left out and why.
    async fn available(&self, job: &Job, context: &ToolContext) -> Result<(Vec<String>, Vec<String>), String> {
        let models = match &job.scale_use {
            Some(name) => {
                let settings = crate::config::Settings::load(&self.agent_dir, &context.cwd);
                crate::scale::order(&self.agent_dir, &settings, &crate::scale::find(&settings, name)?)
            }
            None => job.models.clone(),
        };
        if models.is_empty() {
            return Ok((vec![format!("{}/{}", context.provider, context.model_id)], Vec::new()));
        }
        let registry = ModelRegistry::load(&self.agent_dir.join("models.json"))?;
        let paths = CredentialPaths::from_environment(&self.agent_dir);
        let (mut found, mut skipped) = (Vec::new(), Vec::new());
        // A scale model carries its level (provider/id:level), which --model passes along.
        for name in &models {
            let (id, level) = crate::scale::split(name);
            let level = level.map_or(String::new(), |level| format!(":{level}"));
            let Ok(model) = registry.resolve(id) else {
                skipped.push(format!("[subagent {}: skipped {name}, which is not in models.json]", job.name));
                continue;
            };
            match credential_for(&model.provider, registry.provider_config(&model.provider), &paths).await {
                Ok(_) => found.push(format!("{}/{}{level}", model.provider, model.id)),
                Err(_) => skipped.push(format!("[subagent {}: skipped {name}, which has no login]", job.name)),
            }
        }
        if found.is_empty() {
            return Err(format!("None of the {} job's models can run: {}. Log in to one, or change its models in {}.", job.name, skipped.join(" "), self.agent_dir.join("agents").join(format!("{}.md", job.name)).display()));
        }
        Ok((found, skipped))
    }

    // Input: the job, the task, the models to try in order and the lines saying which were left
    // out. Output: the shell command that prints those lines, then runs the task on the first
    // model and, each time a run fails, on the next. Its first line names the job and the task,
    // which is what a background job's message shows of it.
    fn command(&self, job: &Job, task: &str, (models, skipped): (Vec<String>, Vec<String>), context: &ToolContext) -> Result<String, String> {
        let exe = std::env::current_exe().map_err(|error| format!("could not find piccosta's own binary: {error}"))?;
        // A binary replaced while it runs (a rebuild) reads as "<path> (deleted)"; the new one at
        // that path is the one to start.
        let exe = exe.to_string_lossy().trim_end_matches(" (deleted)").to_string();
        let mut flags = vec![match &context.session_file {
            Some(file) => format!("--session-dir {}", quote(&file.parent().unwrap_or(Path::new(".")).join("subagents").to_string_lossy())),
            None => "--no-session".to_string(),
        }];
        flags.extend(job.thinking.as_ref().map(|level| format!("--thinking {}", quote(level))));
        flags.extend(job.tools.as_ref().map(|tools| format!("--tools {}", quote(tools))));
        if !job.prompt.is_empty() {
            flags.push(format!("--append-system-prompt {}", quote(&job.prompt)));
        }
        let flags = flags.join(" ");
        let runs: Vec<String> = models.iter().map(|model| format!("printf '%s' \"$task\" | {CHILD_ENV}={} {} -p --model {} {flags}", quote(&job.name), quote(&exe), quote(model))).collect();
        let notes: String = skipped.iter().map(|line| format!("printf '%s\\n' {}\n", quote(line))).collect();
        let command = format!("# subagent {}: {}\ntask={}\n{notes}{}", job.name, task.lines().next().unwrap_or_default(), quote(task), runs.join(" ||\n"));
        if command.len() > MAX_COMMAND_BYTES {
            return Err(format!("The task is too long to hand over ({} bytes, the limit is about {MAX_COMMAND_BYTES}). Write it to a file and give the subagent the path.", task.len()));
        }
        Ok(command)
    }
}

impl SubagentTool {
    // Each job in a line: its name, what it's for, and the models it runs on (the tool's
    // description and the TUI's /agents both show this).
    pub fn describe(&self) -> Vec<String> {
        self.jobs
            .iter()
            .map(|job| {
                let models = match (&job.scale_use, job.models.is_empty()) {
                    (Some(name), _) => format!("the scale's {name} models"),
                    (None, true) => "your own model".to_string(),
                    (None, false) => job.models.join(", else "),
                };
                format!("{}: {} Runs on {models}.", job.name, job.description)
            })
            .collect()
    }
}

impl Tool for SubagentTool {
    fn name(&self) -> &'static str {
        NAME
    }

    fn definition(&self) -> ToolDefinition {
        let names: Vec<&str> = self.jobs.iter().map(|job| job.name.as_str()).collect();
        let jobs: String = self.describe().iter().map(|line| format!("\n- {line}")).collect();
        ToolDefinition {
            name: NAME.into(),
            description: format!(
                "Hand a task to a subagent: an agent with a fresh context that works in this directory and returns its final answer. You choose the job; the user chose the models each job runs on, and the first one available is used. The subagent knows only the task you give it, so include the paths, what you already know and what you want back. One that runs long moves to the background, and you get a message when it ends.\nJobs:{jobs}"
            ),
            parameters: json!({
                "type": "object",
                "required": ["agent", "task"],
                "properties": {
                    "agent": {"type": "string", "enum": names, "description": "The job"},
                    "task": {"type": "string", "description": "The task, with everything the subagent needs to do it"}
                }
            }),
            constrained_sampling: Some(json!({"type": "json_schema", "strict": "prefer"})),
        }
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("Hand a task to a subagent job, which runs on the model the user set for it")
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let name = arguments.get("agent").and_then(Value::as_str).unwrap_or_default();
            let task = arguments.get("task").and_then(Value::as_str).unwrap_or_default();
            let Some(job) = self.jobs.iter().find(|job| job.name == name) else {
                let names: Vec<&str> = self.jobs.iter().map(|job| job.name.as_str()).collect();
                return Err(format!("No subagent job \"{name}\". Jobs: {}", names.join(", ")));
            };
            let models = self.available(job, context).await?;
            super::bash::run_own(self.command(job, task, models, context)?, context).await
        })
    }
}

// The program name a piccosta process has in /proc/<pid>/comm.
const PROCESS_NAME: &str = "piccosta";
// The variable bash sets to the session that ran a command, which everything it starts inherits.
const SESSION_ENV: &str = "PI_SESSION_ID";
// Where /proc/<pid>/stat keeps the start time: field 22 (proc_pid_stat(5)), counted among the
// fields after the name's ')', which start at field 3.
const STAT_STARTTIME_AFTER_NAME: usize = 22 - 3;

// A piccosta run a session's tools started: a subagent job, or `piccosta -p` run by hand.
pub struct RunningAgent {
    pub job: Option<String>,
    pub model: Option<String>,
    pub running_for: std::time::Duration,
    // Its session file, which it holds open (and locked) while it runs; None for one kept in memory.
    pub session: Option<PathBuf>,
}

// The piccosta processes started from these sessions (by id), found through /proc in one walk: those
// whose environment names one of them as the session that started them. The environment carries
// over to what a command leaves running with `&` after its shell exits, which the process tree
// doesn't. Output: each session's agents. Where there's no /proc there are none.
pub fn running_under(session_ids: &[&str]) -> std::collections::HashMap<String, Vec<RunningAgent>> {
    let mut found: std::collections::HashMap<String, Vec<RunningAgent>> = std::collections::HashMap::new();
    let wanted: Vec<String> = session_ids.iter().map(|id| format!("{SESSION_ENV}={id}")).collect();
    let job_prefix = format!("{CHILD_ENV}=");
    let Some(uptime) = std::fs::read_to_string("/proc/uptime").ok().and_then(|text| text.split_whitespace().next()?.parse::<f64>().ok()) else { return found };
    // SAFETY: sysconf only reads a configuration value.
    let ticks_per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    let Ok(entries) = std::fs::read_dir("/proc") else { return found };
    let agents = entries.flatten().filter_map(|entry| {
            let dir = entry.path();
            entry.file_name().to_str()?.parse::<u32>().ok()?;
            if std::fs::read_to_string(dir.join("comm")).ok()?.trim_end() != PROCESS_NAME {
                return None;
            }
            let environ = std::fs::read(dir.join("environ")).ok()?;
            let variables: Vec<&[u8]> = environ.split(|byte| *byte == 0).collect();
            // A piccosta's reaper (tools/reaper.rs) shares its environment but is no agent.
            let reaper = format!("{}=", super::reaper::REAPER_ENV);
            let started_by = wanted.iter().find(|wanted| variables.contains(&wanted.as_bytes()))?;
            if variables.iter().any(|variable| variable.starts_with(reaper.as_bytes())) {
                return None;
            }
            let job = variables.iter().find_map(|variable| variable.strip_prefix(job_prefix.as_bytes())).map(|name| String::from_utf8_lossy(name).into_owned());
            let command_line = std::fs::read(dir.join("cmdline")).ok()?;
            let arguments: Vec<String> = command_line.split(|byte| *byte == 0).map(|argument| String::from_utf8_lossy(argument).into_owned()).collect();
            // Its model, or the kind of work it was started for (a manager's lane, --use exploration).
            let value_of = |flag: &str| arguments.iter().position(|argument| argument == flag).and_then(|index| arguments.get(index + 1)).cloned();
            let model = value_of("--model").or_else(|| value_of("--use"));
            let stat = std::fs::read_to_string(dir.join("stat")).ok()?;
            let started_ticks: f64 = stat.rsplit_once(')')?.1.split_whitespace().nth(STAT_STARTTIME_AFTER_NAME)?.parse().ok()?;
            let running_for = std::time::Duration::from_secs_f64((uptime - started_ticks / ticks_per_second).max(0.0));
            let open_files = std::fs::read_dir(dir.join("fd")).into_iter().flatten().flatten();
            let session = open_files.filter_map(|fd| std::fs::read_link(fd.path()).ok()).find(|path| path.extension().is_some_and(|extension| extension == "jsonl"));
            Some((started_by[SESSION_ENV.len() + 1..].to_string(), RunningAgent { job, model, running_for, session }))
        });
    for (session_id, agent) in agents {
        found.entry(session_id).or_default().push(agent);
    }
    found
}

// The tools a job's subagent runs with, for an approval card: its own list, or "all tools".
pub fn tools_of(job: &str) -> String {
    let jobs = SubagentTool::load(&crate::config::agent_dir()).map(|tool| tool.jobs).unwrap_or_default();
    jobs.into_iter().find(|candidate| candidate.name == job).and_then(|job| job.tools).unwrap_or_else(|| "all tools".into())
}
