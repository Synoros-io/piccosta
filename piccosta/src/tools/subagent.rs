// Subagents: a task handed to another agent, on a model the job picks unless the call names one.
//
// Each job has a list of allowed models, not one model per tier, and the model handing the work
// over doesn't choose among them. So the
// model calling `subagent` names a job and a task, and the job's own list says which models may
// do it: the first one with a login runs it, and when that run fails the next one takes it over.
// The lists are the user's; a call can name a model and a level for its own task instead.
//
// The jobs are the model scale's kinds of work (scale.rs): each says what it's for, its models and
// levels (in the order usage allows at the time; none means the calling session's own model), its
// tools (all when it lists none), whether its commands may change files (readOnly: bash runs in
// a sandbox that can't, readonly.rs) and its instructions, appended to the subagent's system prompt.
//
// A run is a child piccosta in print mode, started through bash's runner, so it is a shell command
// like any other the session runs: its output streams to the UI, an abort kills it, one still going
// at the soft limit becomes a background job whose message arrives when it ends, and print mode
// waits for it. The child's session goes in a `subagents` folder next to the parent's, out of the
// parent folder's list. A subagent gets no subagent tool of its own (CHILD_ENV), and its session
// records its job as its kind of work, so it has the job's tools and readOnly when it's reopened
// too (host.rs tool_list). A manager's lane hands work only to the jobs its manager granted it, and
// search (manager.rs granted).

use super::{Tool, ToolContext, ToolFuture};
use crate::auth::{credential_for, CredentialPaths};
use crate::message::ToolDefinition;
use crate::model::ModelRegistry;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

// Set in a subagent's environment (to its job's name); a session that has it offers no subagent tool.
// What a model is told when it takes over a subagent's session from one that failed.
const CARRY_ON: &str = "The previous model stopped with an error before finishing. Carry on with the task from where it got to.";
pub const CHILD_ENV: &str = "PICCOSTA_SUBAGENT";
// The tool's name, as the model calls it.
pub const NAME: &str = "subagent";
// Linux's cap on one argument (MAX_ARG_STRLEN), which the whole command `bash -c` runs must fit under.
const MAX_COMMAND_BYTES: usize = 128 * 1024;

struct Job {
    name: String,
    description: String,
    tools: Option<String>,
    read_only: bool,
    prompt: String,
}

pub struct SubagentTool {
    jobs: Vec<Job>,
    agent_dir: PathBuf,
    // Whether these are only the jobs a manager granted (a lane's), not every one.
    granted: bool,
}

// Text as one single-quoted shell word.
fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

impl SubagentTool {
    // The scale's jobs for an agent dir, only those `granted` when it names some. None when none
    // is on (no tool then).
    pub fn load(agent_dir: &Path, granted: Option<&[String]>) -> Option<SubagentTool> {
        let strings = |entry: &Map<String, Value>, key: &str| -> Vec<String> { entry.get(key).and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect() };
        let text = |entry: &Map<String, Value>, key: &str| entry.get(key).and_then(Value::as_str).filter(|value| !value.is_empty()).map(str::to_string);
        let jobs: Vec<Job> = crate::scale::jobs(agent_dir)
            .into_iter()
            .filter(|(name, _)| granted.is_none_or(|jobs| jobs.contains(name)))
            .map(|(name, entry)| Job {
                description: text(&entry, "for").unwrap_or_default(),
                tools: Some(strings(&entry, "tools").join(",")).filter(|tools| !tools.is_empty()),
                read_only: entry.get("readOnly").and_then(Value::as_bool) == Some(true),
                prompt: text(&entry, "instructions").unwrap_or_default(),
                name,
            })
            .collect();
        (!jobs.is_empty()).then(|| SubagentTool { jobs, agent_dir: agent_dir.to_path_buf(), granted: granted.is_some() })
    }

    // The job's models that a connected account serves and that have a login, in its order, and a
    // line for each one left out and why. A job that lists none, or none of whose models can run,
    // runs on the calling session's own model.
    async fn available(&self, job: &Job, context: &ToolContext<'_>) -> Result<(Vec<String>, Vec<String>), String> {
        let models = crate::scale::order(&self.agent_dir, &crate::scale::find(&self.agent_dir, &job.name, false)?);
        let own = format!("{}/{}", context.provider, context.model_id);
        if models.is_empty() {
            return Ok((vec![own], Vec::new()));
        }
        let registry = ModelRegistry::load(&self.agent_dir.join("models.json"))?;
        let paths = CredentialPaths::from_environment(&self.agent_dir);
        let (mut found, mut skipped) = (Vec::new(), Vec::new());
        // A scale model carries its level (provider/id:level), which --model passes along.
        for name in &models {
            let (id, level) = crate::scale::split(name);
            let level = level.map_or(String::new(), |level| format!(":{level}"));
            let Ok(model) = registry.resolve(id) else {
                skipped.push(format!("[subagent {}: skipped {name}, which no connected account serves]", job.name));
                continue;
            };
            if model.interactive_only {
                skipped.push(format!("[subagent {}: skipped {name}, interactive-only provider]", job.name));
                continue;
            }
            match credential_for(&model.provider, registry.provider_config(&model.provider), &paths).await {
                Ok(_) => found.push(format!("{}/{}{level}", model.provider, model.id)),
                Err(_) => skipped.push(format!("[subagent {}: skipped {name}, which has no login]", job.name)),
            }
        }
        if found.is_empty() {
            skipped.push(format!("[subagent {}: none of its models can run, so it runs on {own}; /scale edit changes them]", job.name));
            found.push(own);
        }
        Ok((found, skipped))
    }

    // Input: the job, the task, the models to try in order and the lines saying which were left
    // out. Output: the shell command that prints those lines, then runs the task on the first
    // model and, each time a run fails, goes on in the same session on the next, with what the
    // failed one did in its context, saying so; a run that saved nothing starts the task afresh.
    // Its first line names the job and the task, which is what a background job's message shows.
    fn command(&self, job: &Job, task: &str, message: &str, (models, skipped): (Vec<String>, Vec<String>), context: &ToolContext) -> Result<String, String> {
        let exe = std::env::current_exe().map_err(|error| format!("could not find piccosta's own binary: {error}"))?;
        // A binary replaced while it runs (a rebuild) reads as "<path> (deleted)"; the new one at
        // that path is the one to start.
        let exe = exe.to_string_lossy().trim_end_matches(" (deleted)").to_string();
        // The subagent's one session file, which each model after a failure carries on.
        let file = context.session_file.as_ref().map(|parent| {
            let stamp = crate::clock::iso_from_ms(crate::clock::now_ms()).replace([':', '.'], "-");
            parent.parent().unwrap_or(Path::new(".")).join("subagents").join(format!("{stamp}_{}.jsonl", uuid::Uuid::now_v7()))
        });
        let mut flags = vec![match &file {
            Some(file) => format!("--session {}", quote(&file.to_string_lossy())),
            None => "--no-session".to_string(),
        }];
        if !job.prompt.is_empty() {
            flags.push(format!("--append-system-prompt {}", quote(&job.prompt)));
        }
        let flags = flags.join(" ");
        let run = |model: &String| format!("{CHILD_ENV}={} {} -p --model {} {flags}", quote(&job.name), quote(&exe), quote(model));
        let mut runs = vec![format!("printf '%s' \"$task\" | {}", run(&models[0]))];
        for pair in models.windows(2) {
            let note = quote(&format!("[subagent {}: {} failed, so {} carries on]", job.name, pair[0], pair[1]));
            let input = match &file {
                Some(file) => format!("{{ [ -s {} ] && printf '%s' \"$carry\" || printf '%s' \"$task\"; }}", quote(&file.to_string_lossy())),
                None => "printf '%s' \"$task\"".to_string(),
            };
            runs.push(format!("{{ printf '%s\\n' {note}; {input} | {}; }}", run(&pair[1])));
        }
        let notes: String = skipped.iter().map(|line| format!("printf '%s\\n' {}\n", quote(line))).collect();
        let command = format!("# subagent {}: {}\ntask={}\ncarry={}\n{notes}{}", job.name, task.lines().next().unwrap_or_default(), quote(message), quote(CARRY_ON), runs.join(" ||\n"));
        if command.len() > MAX_COMMAND_BYTES {
            return Err(format!("The task is too long to hand over ({} bytes, the limit is about {MAX_COMMAND_BYTES}). Write it to a file and give the subagent the path.", message.len()));
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
                let models = crate::scale::find(&self.agent_dir, &job.name, false).map(|entry| crate::scale::available(&self.agent_dir, &entry).join(", else ")).unwrap_or_default();
                let models = if models.is_empty() { "your own model".to_string() } else { models };
                let owns: Vec<String> = crate::scale::owned(&self.agent_dir).into_iter().filter(|(_, kind)| *kind == job.name).map(|(tool, _)| tool).collect();
                let owns = if owns.is_empty() { String::new() } else { format!(" Only this job has {}: hand it that work.", owns.join(", ")) };
                let read_only = if job.read_only { " Changes no files." } else { "" };
                format!("{}: {}. Runs on {models}.{read_only}{owns}", job.name, job.description)
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
                "Hand a task to a subagent: an agent with a fresh context that works in this directory. It runs in the background: you get its final answer as a message when it ends, and meanwhile you carry on or end your turn. Its answer is a claim: check what it built or found before you rely on it or pass it on. You choose the job; the user chose the models each job runs on, and the first one available is used. model (provider/id) and thinking, when given, replace the job's own. Pick a job only when its description fits the task; when none does, do the task yourself rather than taking the nearest. The subagent also gets your discussion with the user, word for word (discussed chooses where it starts); the task should say what to do and what you want back, not retell the discussion.\nJobs:{jobs}"
            ),
            parameters: json!({
                "type": "object",
                "required": ["agent", "task"],
                "properties": {
                    "agent": {"type": "string", "enum": names, "description": "The job"},
                    "task": {"type": "string", "description": "What to do and what you want back; the discussion is copied separately"},
                    "discussed": {"type": "string", "description": "The first words of the user's message where this task was first discussed; piccosta copies the discussion from there, word for word, to the subagent"},
                    "model": {"type": "string", "description": "provider/id; else the job's own"},
                    "thinking": {"type": "string", "enum": crate::provider::THINKING_LEVELS, "description": "The level; else the job's own"}
                }
            }),
            constrained_sampling: Some(json!({"type": "json_schema", "strict": "prefer"})),
        }
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("Hand a task to a subagent job, which runs on the model the user set for it")
    }

    // Who's who among the agents and how to start one, since agents started assistants and managers
    // of their own, ran agents through bash, and handed off long work as one-shot subagents.
    fn prompt_guidelines(&self) -> &'static [&'static str] {
        &[
            "The assistant and the project managers are the user's: the user makes them. Never start or stop one unless the user asks",
            "Start other agents only with the subagent tool (a manager also has sessions), never through bash: no piccosta, claude or codex commands",
            "A subagent does one task with a clear end and answers once; it can't be messaged meanwhile and starts no agents itself. Long work that needs steering is a manager's lane (sessions start), not a subagent",
            "In a subagent's task, say what to do, where to look, what's already settled, what to send back and when to stop",
            "You answer for what you start: read a subagent's answer before you pass it on, and check the claims you repeat",
        ]
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let name = arguments.get("agent").and_then(Value::as_str).unwrap_or_default();
            let task = arguments.get("task").and_then(Value::as_str).unwrap_or_default();
            let Some(job) = self.jobs.iter().find(|job| job.name == name) else {
                let names: Vec<&str> = self.jobs.iter().map(|job| job.name.as_str()).collect();
                let granted = if self.granted { " Your manager granted this session only these: do the rest of the work yourself." } else { "" };
                return Err(format!("No subagent job \"{name}\". Jobs: {}.{granted}", names.join(", ")));
            };
            // The model named for this task, else the job's own, each at the level named, if any.
            let (models, skipped) = match arguments.get("model").and_then(Value::as_str).filter(|model| !model.is_empty()) {
                Some(named) => {
                    let registry = ModelRegistry::load(&self.agent_dir.join("models.json"))?;
                    let model = registry.resolve(named).map_err(|_| format!("No connected account serves the model \"{named}\"; /model lists the ones that are"))?;
                    (vec![format!("{}/{}", model.provider, model.id)], Vec::new())
                }
                None => self.available(job, context).await?,
            };
            let registry = ModelRegistry::load(&self.agent_dir.join("models.json"))?;
            for name in &models {
                if registry.resolve(crate::scale::split(name).0)?.interactive_only {
                    return Err("Interactive-only providers cannot run subagents, including inherited and fallback models (interactiveOnly in models.json).".into());
                }
            }
            let models = match arguments.get("thinking").and_then(Value::as_str) {
                Some(level) => models.iter().map(|model| format!("{}:{level}", crate::scale::split(model).0)).collect(),
                None => models,
            };
            // Straight to the background: a subagent runs for minutes, and the session stays free
            // to answer meanwhile; its answer comes back as the job's message.
            let message = crate::session::delegation_message(context, arguments.get("discussed").and_then(Value::as_str), task)?;
            super::bash::run_in_background(self.command(job, task, &message, (models, skipped), context)?, context).await
        })
    }
}

// The variable bash sets to the session that ran a command, which everything it starts inherits.
pub const SESSION_ENV: &str = "PICCOSTA_SESSION_ID";

// The tools a job's subagent runs with, for an approval card: its own list, or "all tools".
pub fn tools_of(job: &str) -> String {
    let jobs = SubagentTool::load(&crate::config::agent_dir(), None).map(|tool| tool.jobs).unwrap_or_default();
    let Some(job) = jobs.into_iter().find(|candidate| candidate.name == job) else { return "all tools".into() };
    let tools = job.tools.unwrap_or_else(|| "all tools".into());
    if job.read_only { format!("{tools}, changing no files") } else { tools }
}
