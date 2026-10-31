// Opening sessions to run, for every frontend: print and json mode open one, the terminal UI
// keeps several live at once. A session is a file plus the agent that runs it; what an agent runs
// with (model, prompt sections, tools, settings) is resolved here the same way for all of them.

use crate::agent::{Agent, AgentConfig, EventSink};
use crate::message::AgentMessage;
use crate::model::{Model, ModelRegistry};
use crate::session::Session;
use crate::tools::ToolShared;
use crate::{auth, config, prompt, session, tools};
use indexmap::IndexMap;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;

// The pi release piccosta is based on and speaks the protocols of (session files, json and rpc
// mode).
pub const PI_VERSION: &str = "1.4.0";
// How many characters of a session's first prompt make its title in a list.
pub const TITLE_CHARS: usize = 80;

// Where piccosta runs: the directory, the agent dir and its settings, and this directory's
// session folder.
pub struct Host {
    pub cwd: PathBuf,
    pub agent_dir: PathBuf,
    pub settings: config::Settings,
    pub session_dir: PathBuf,
    // Whether AGENTS.md and CLAUDE.md go into the prompt (--no-context-files turns them off).
    pub context_files: bool,
    // Whether the model gets piccosta's tools (--no-tools turns them off), and which (--tools).
    pub tools: bool,
    pub tool_names: Option<Vec<String>>,
    pub exclude_tools: Vec<String>,
    // --append-system-prompt's text, in place of the agent dir's APPEND_SYSTEM.md.
    pub append_prompt: Option<String>,
    // --yolo: no approvals, whatever the frontend asks for.
    pub yolo: bool,
}

// What a live session's task is asked to do, one at a time.
pub enum Command {
    Prompt(String, Vec<crate::message::Content>),
    Model(String),
    Thinking(String),
    Name(String),
    Compact(Option<String>),
    AutoCompaction(bool),
    AutoRetry(bool),
    // Work with partners: after each of the user's prompts, these models (provider/id), one after
    // another, each take a turn checking the work so far, then the session model takes it back for
    // the user (a pair with one, a trio with two); none works alone again.
    Pair(Vec<String>),
}

// The message that opens the partner's turn, shown in the conversation (Jack, 2026-10-30: "they
// review eachotehres outputs in one user session, a three part conversation with the user in the
// loop").
pub const PAIR_TYPE: &str = "piccosta-pair";
// The session entry that records the partners, so a reopened session still works with them.
const PAIR_SETTING: &str = "piccosta-pair-partner";

// The partners a session last recorded (none when it works alone). The first recordings named one
// `partner`; later ones list `partners`.
pub fn pair_partners(session: &Session) -> Vec<String> {
    let recorded = session.path_to_leaf().into_iter().rev().find_map(|entry| match entry {
        session::SessionEntry::Custom { custom_type, data, .. } if custom_type == PAIR_SETTING => Some(data.clone().unwrap_or_default()),
        _ => None,
    });
    let Some(recorded) = recorded else { return Vec::new() };
    match (recorded.get("partners").and_then(Value::as_array), recorded.get("partner").and_then(Value::as_str)) {
        (Some(list), _) => list.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        (None, Some(one)) => vec![one.to_string()],
        _ => Vec::new(),
    }
}

const PAIR_NOTE: &str = "{partner}: you are working with {others} in this session, and the user reads you all. {lead} has just answered the user's last message above{checked}. Check the work against what the user asked: verify what it claims, fix what is wrong, finish what was left undone, and say plainly what you changed, or that it holds up. Then stop; {next} speaks next.";

// What the task reports after each command (and after a turn a background job started, marked
// `notice`): how it went, and the model and thinking level the session has now.
pub struct Done {
    pub result: Result<(), String>,
    pub notice: bool,
    pub model: Model,
    pub thinking: String,
}

// A session file as a list shows it.
#[derive(Clone)]
pub struct SessionSummary {
    pub path: PathBuf,
    pub id: String,
    pub cwd: PathBuf,
    pub title: String,
    pub modified: std::time::SystemTime,
}

impl Host {
    pub fn load() -> Host {
        Host::at(std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")))
    }

    // The host for a directory other than the process's own (an ACP client names one per session).
    pub fn at(cwd: PathBuf) -> Host {
        let agent_dir = config::agent_dir();
        let settings = config::Settings::load(&agent_dir, &cwd);
        let session_dir = session::default_session_dir(&agent_dir, &cwd);
        Host { cwd, agent_dir, settings, session_dir, context_files: true, tools: true, tool_names: None, exclude_tools: Vec::new(), append_prompt: None, yolo: false }
    }

    // The host as a command line sets it up: --no-context-files, --no-tools, --tools,
    // --session-dir, --append-system-prompt (each one a file's contents when it names a file, as
    // pi's resolvePromptInput reads it, else the text itself).
    pub fn for_args(args: &crate::cli::Args) -> Host {
        Host::for_args_at(args, std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")))
    }

    // The same for another directory (an ACP session's).
    pub fn for_args_at(args: &crate::cli::Args, cwd: PathBuf) -> Host {
        let host = Host::at(cwd);
        let session_dir = args.session_dir.as_ref().map(PathBuf::from).unwrap_or(host.session_dir);
        let read = |input: &String| std::fs::read_to_string(input).map(|text| text.trim_start_matches('\u{feff}').to_string()).unwrap_or_else(|_| input.clone());
        let append_prompt = (!args.append_prompt.is_empty()).then(|| args.append_prompt.iter().map(read).collect::<Vec<_>>().join("\n\n"));
        Host { context_files: !args.no_context_files, tools: !args.no_tools, tool_names: args.tools.clone(), exclude_tools: args.exclude_tools.clone(), append_prompt, session_dir, yolo: args.yolo, ..host }
    }

    pub fn registry(&self) -> Result<ModelRegistry, String> {
        ModelRegistry::load(&self.agent_dir.join("models.json"))
    }

    pub fn show_settings_errors(&self) {
        for error in &self.settings.load_errors {
            eprintln!("Warning: Invalid settings file {error}");
        }
    }

    // The models whose provider has a credential (pi's getAvailableSnapshot), in the registry's order.
    pub async fn usable_models(&self) -> Result<Vec<Model>, String> {
        let registry = self.registry()?;
        let paths = auth::CredentialPaths::from_environment(&self.agent_dir);
        let mut usable: std::collections::HashMap<String, bool> = std::collections::HashMap::new();
        let mut models = Vec::new();
        for model in registry.models() {
            if !usable.contains_key(&model.provider) {
                let found = auth::credential_for(&model.provider, registry.provider_config(&model.provider), &paths).await.is_ok();
                usable.insert(model.provider.clone(), found);
            }
            if usable[&model.provider] {
                models.push(model.clone());
            }
        }
        Ok(models)
    }

    // A session by path, this directory's latest (`latest`), or a new one, kept in memory only
    // when `persist` is false.
    pub fn session(&self, path: Option<&Path>, latest: bool, persist: bool) -> Result<Session, String> {
        let latest = if latest { self.sessions(1).into_iter().next().map(|summary| summary.path) } else { None };
        match path.map(Path::to_path_buf).or(latest) {
            Some(path) => Session::open(&path, &self.cwd).map_err(|error| error.to_string()),
            None => Ok(Session::create(&self.cwd, persist.then_some(self.session_dir.as_path()))),
        }
    }

    // Up to `limit` of this directory's sessions, newest first: files whose header names this cwd
    // (pi's continueRecent), each titled by its name or its first prompt. Files are opened newest
    // first and only until the limit is met.
    pub fn sessions(&self, limit: usize) -> Vec<SessionSummary> {
        sessions_in(&self.session_dir, Some(&self.cwd), limit)
    }

    // The opening session a command line asks for: a new one in memory only (--no-session wins,
    // as in pi), a fork of one (--fork), one named (--session, --resume), this directory's latest
    // (-c), or a new one. `--resume` with no name is the TUI's picker, so it opens a new one here.
    pub fn opening(&self, args: &crate::cli::Args) -> Result<Session, String> {
        if args.no_session {
            return Ok(Session::create(&self.cwd, None));
        }
        if let Some(source) = &args.fork {
            let source = Session::open(&self.find_session(source)?, &self.cwd).map_err(|error| error.to_string())?;
            let leaf = source.leaf_id().map(str::to_string);
            return source.branch(leaf.as_deref(), &self.session_dir).map_err(|error| error.to_string());
        }
        let named = args.session.iter().chain(args.resume.iter().filter(|name| !name.is_empty())).next();
        let path = named.map(|name| self.find_session(name)).transpose()?;
        self.session(path.as_deref(), args.continue_recent, true)
    }

    // A session named on the command line. One that looks like a path (it has a `/` or ends in
    // .jsonl) is that file, made on first message when it doesn't exist yet. Anything else is an
    // id, an id's start or a session's name: this directory's sessions first, then every
    // project's (pi's resolveSessionPath; Claude Code resumes by name). Output: the file, or an
    // error saying nothing matched.
    pub fn find_session(&self, wanted: &str) -> Result<PathBuf, String> {
        if wanted.contains('/') || wanted.ends_with(".jsonl") {
            return Ok(self.cwd.join(wanted));
        }
        let matches = |summary: &SessionSummary| summary.id.starts_with(wanted) || summary.title == wanted;
        let everywhere = || {
            let folders = std::fs::read_dir(self.agent_dir.join("sessions")).into_iter().flatten().flatten();
            folders.flat_map(|folder| sessions_in(&folder.path(), None, usize::MAX))
        };
        let found = self.sessions(usize::MAX).into_iter().find(matches).or_else(|| everywhere().find(matches));
        found.map(|summary| summary.path).ok_or_else(|| format!("No session matches \"{wanted}\" (an id, the start of one, or a name)"))
    }

    // The model's tools: the built-in ones and the subagent tool (not in a subagent, which can't
    // hand its task on), cut to --tools when given.
    fn tool_list(&self) -> Vec<Arc<dyn tools::Tool>> {
        if !self.tools {
            return Vec::new();
        }
        let mut list = tools::builtin();
        if std::env::var_os(tools::subagent::CHILD_ENV).is_none() {
            list.extend(tools::subagent::SubagentTool::load(&self.agent_dir).map(|tool| Arc::new(tool) as Arc<dyn tools::Tool>));
        }
        let wanted = |tool: &Arc<dyn tools::Tool>| self.tool_names.as_ref().is_none_or(|names| names.iter().any(|name| name == tool.name()));
        list.into_iter().filter(|tool| wanted(tool) && !self.exclude_tools.iter().any(|name| name == tool.name())).collect()
    }

    // The prompt sections for a model; its constitution can differ by model.
    pub fn sections(&self, model: &Model, cwd: &Path) -> Result<IndexMap<String, String>, String> {
        let constitution = prompt::constitution(&self.settings, &model.name)?;
        let (custom_prompt, discovered_append) = prompt::custom_prompts(&self.agent_dir);
        let append_prompt = self.append_prompt.clone().or(discovered_append);
        let tool_list = self.tool_list();
        Ok(prompt::sections(&prompt::PromptInputs { cwd, agent_dir: &self.agent_dir, tools: &tool_list, custom_prompt, append_prompt, constitution, context_files: self.context_files }))
    }

    // A session ready to run. Input: the model and thinking level asked for, if any (else what
    // the session last used, else the level that model was last set to, else the settings'
    // defaults), where events go, and whether a run
    // waits for background jobs before it ends (print and json mode). Output: the agent and the
    // tools' shared state (background jobs, web results), which the caller shuts down at the end.
    pub fn open(&self, mut session: Session, model: Option<&str>, thinking: Option<&str>, emit: EventSink, hold_for_jobs: bool) -> Result<(Agent, Arc<ToolShared>), String> {
        session.claim()?;
        let registry = self.registry()?;
        let resumed = session.context();
        let wanted = model
            .map(str::to_string)
            .or_else(|| resumed.model.as_ref().map(|(provider, id)| format!("{provider}/{id}")))
            .or_else(|| self.settings.default_model());
        let wanted = wanted.ok_or_else(|| format!("no model; pass --model or set defaultModel in {}", self.agent_dir.join("settings.json").display()))?;
        // A session whose model has left the catalog opens on the default model rather than not at
        // all (pi's "Could not restore model"); a model asked for by name still has to exist.
        let model = match (registry.resolve(&wanted), model, self.settings.default_model()) {
            (Ok(found), ..) => found.clone(),
            (Err(_), None, Some(default)) if default != wanted => registry.resolve(&default)?.clone(),
            (Err(error), ..) => return Err(error),
        };
        let has_thinking = resumed.messages.iter().any(|message| matches!(message, AgentMessage::Assistant(_)));
        // The level asked for, else the one this model was last set to by hand (in any session, so
        // a reopened one follows the latest choice, as Claude Code's effort does), else what the
        // session last ran at, else the default.
        let remembered = thinking.map(str::to_string).or_else(|| self.remembered_thinking_level(&model));
        let thinking_level = remembered.unwrap_or_else(|| if has_thinking { resumed.thinking_level.clone() } else { self.settings.default_thinking_level() });
        let cwd = PathBuf::from(&session.header().cwd);
        let sections = self.sections(&model, &cwd)?;

        let (notice_sender, notices) = tokio::sync::mpsc::unbounded_channel();
        let credentials = auth::CredentialPaths::from_environment(&self.agent_dir);
        let tool_shared = Arc::new(ToolShared::new(&self.settings, &self.agent_dir, notice_sender, &registry, credentials.clone()));
        // A reopened session's stored web results, so its responseIds still page.
        tool_shared.web.store.restore(session.path_to_leaf());
        let compaction = self.settings.compaction(&model.provider, &model.id);
        let config = AgentConfig {
            model,
            registry,
            credentials,
            thinking_level,
            tools: self.tool_list(),
            tool_shared: tool_shared.clone(),
            cwd,
            sections,
            retry: self.settings.retry(),
            compaction,
            transport: self.settings.transport(),
            notices,
            gate: None,
            hold_for_jobs,
        };
        let agent = Agent::new(session, config, emit).map_err(|error| format!("could not write the session: {error}"))?;
        Ok((agent, tool_shared))
    }
}

impl Host {
    // The thinking level `model` was last set to by hand, in any session.
    pub fn remembered_thinking_level(&self, model: &Model) -> Option<String> {
        config::remembered_thinking_level(&self.agent_dir, &format!("{}/{}", model.provider, model.id))
    }

    // A live session's task: its commands one at a time, and (with `wake`) a background job's
    // finish message as a turn of its own when it comes while the session is idle; without, it
    // waits for the next prompt (rpc mode: T3 Code stops a pi that starts a run unasked). Steering and aborts don't wait
    // in this line; they go through the agent's handle while a prompt runs. The task ends when
    // the commands' sender is dropped, after stopping the session's jobs and web state. Runs on a
    // LocalSet: the agent's future isn't Send (rustc can't prove it through the providers'
    // higher-ranked lifetimes).
    pub fn spawn(self: Arc<Host>, mut agent: Agent, shared: Arc<ToolShared>, wake: bool, mut commands: UnboundedReceiver<Command>, mut done: impl FnMut(Done) + 'static) -> tokio::task::JoinHandle<()> {
        tokio::task::spawn_local(async move {
            let mut partners = pair_partners(&agent.session);
            loop {
                let (result, notice) = tokio::select! {
                    command = commands.recv() => match command {
                        Some(command) if !commands.is_closed() => (self.obey(&mut agent, command, &mut partners).await, false),
                        _ => break,
                    },
                    Some(notice) = agent.notice(), if wake => {
                        if tools::jobs::stopped_by_user(&notice) {
                            agent.hold_notice(notice);
                            continue;
                        }
                        (agent.take_notice(notice).await, true)
                    }
                };
                done(Done { result, notice, model: agent.model().clone(), thinking: agent.thinking_level().to_string() });
            }
            shared.jobs.shutdown("quit");
            shared.web.shutdown();
        })
    }

    // Carry out one command. `partners` are the models that check the session model's work.
    async fn obey(&self, agent: &mut Agent, command: Command, partners: &mut Vec<String>) -> Result<(), String> {
        let written = |result: std::io::Result<()>| result.map_err(|error| format!("could not write the session: {error}"));
        match command {
            // With partners, a prompt the session model answered (not stopped) goes round them next.
            Command::Prompt(text, images) => {
                agent.prompt_with_images(&text, images).await?;
                self.partner_turns(agent, partners).await
            }
            // Each named by its full provider/id, so any model the registry has can join.
            Command::Pair(models) => {
                let registry = self.registry()?;
                let models = models.iter().map(|name| registry.resolve(name).map(|model| format!("{}/{}", model.provider, model.id))).collect::<Result<Vec<_>, _>>()?;
                written(agent.session.append_custom(PAIR_SETTING, Some(serde_json::json!({ "partners": models }))).map(drop))?;
                *partners = models;
                Ok(())
            }
            Command::Compact(instructions) => agent.compact(instructions.as_deref()).await,
            Command::Thinking(level) => written(agent.set_thinking_level(&level)),
            Command::Name(name) => written(agent.set_name(&name)),
            Command::AutoCompaction(enabled) => {
                agent.set_auto_compaction(enabled);
                Ok(())
            }
            Command::AutoRetry(enabled) => {
                agent.set_auto_retry(enabled);
                Ok(())
            }
            Command::Model(name) => self.switch_model(agent, &name),
        }
    }

    // Put the session on another model (provider/id). It brings the level it was last set to;
    // one never set keeps the session's.
    fn switch_model(&self, agent: &mut Agent, name: &str) -> Result<(), String> {
        let written = |result: std::io::Result<()>| result.map_err(|error| format!("could not write the session: {error}"));
        let model = self.registry()?.resolve(name)?.clone();
        let sections = self.sections(&model, Path::new(&agent.session.header().cwd))?;
        let compaction = self.settings.compaction(&model.provider, &model.id);
        let level = self.remembered_thinking_level(&model).unwrap_or_else(|| agent.thinking_level().to_string());
        written(agent.set_model(model, sections, compaction))?;
        written(agent.set_thinking_level(&level))
    }

    // The partners' turns, in order, after the session model answered: each takes the session for
    // one run, opened by a note saying whose work to check and who checked it already, and the
    // session model takes it back for the user's next message, also when a turn failed. A turn the
    // user stops ends the round.
    pub async fn partner_turns(&self, agent: &mut Agent, partners: &[String]) -> Result<(), String> {
        let lead = format!("{}/{}", agent.model().provider, agent.model().id);
        let mut result = Ok(());
        for (index, partner) in partners.iter().enumerate().filter(|(_, partner)| **partner != lead) {
            if agent.aborted() || result.is_err() {
                break;
            }
            let others: Vec<&str> = std::iter::once(lead.as_str()).chain(partners.iter().map(String::as_str).filter(|other| other != partner)).collect();
            let checked = if index == 0 { String::new() } else { format!(", and {} checked it since", partners[..index].join(" and ")) };
            let next = partners.get(index + 1).map_or("the user", String::as_str);
            let text = PAIR_NOTE.replace("{partner}", partner).replace("{others}", &others.join(" and ")).replace("{lead}", &lead).replace("{checked}", &checked).replace("{next}", next);
            if let Err(error) = self.switch_model(agent, partner) {
                result = Err(error);
                break;
            }
            let details = serde_json::json!({ "partner": partner, "lead": lead });
            let note = crate::message::CustomMessage { custom_type: PAIR_TYPE.into(), content: crate::message::TextOrBlocks::Text(text), display: true, details: Some(details), timestamp: crate::clock::now_ms() };
            result = agent.take_notice(note).await;
        }
        self.switch_model(agent, &lead)?;
        result
    }
}

// Session files in one directory, optionally restricted to their recorded cwd. A closed
// session may have been renamed, so its header supplies both the id and the directory it ran in.
pub fn sessions_in(dir: &Path, cwd: Option<&Path>, limit: usize) -> Vec<SessionSummary> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|extension| extension == "jsonl"))
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect();
    files.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    files.into_iter().filter_map(|(modified, path)| {
        let session = Session::open(&path, cwd.unwrap_or(dir)).ok()?;
        let recorded = Path::new(&session.header().cwd);
        cwd.is_none_or(|cwd| recorded == cwd).then(|| SessionSummary {
            title: title(&session), id: session.id().to_string(), cwd: recorded.to_path_buf(), path, modified,
        })
    }).take(limit).collect()
}

// The name the session was last given (pi's session_info entries).
pub fn name(session: &Session) -> Option<String> {
    session.entries().iter().rev().find_map(|entry| match entry {
        session::SessionEntry::SessionInfo { name: Some(name), .. } => Some(name.clone()),
        _ => None,
    })
}

// A session's title: its last name, else the start of its first prompt, else "new session".
pub fn title(session: &Session) -> String {
    let named = name(session);
    let first_prompt = || {
        session.entries().iter().find_map(|entry| match entry {
            session::SessionEntry::Message { message: AgentMessage::User(user), .. } => Some(user.content.text()),
            _ => None,
        })
    };
    let text = named.or_else(first_prompt).unwrap_or_else(|| "new session".into());
    let line = text.lines().find(|line| !line.trim().is_empty()).unwrap_or("").trim();
    line.chars().take(TITLE_CHARS).collect()
}
