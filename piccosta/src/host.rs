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
    // Work with other models from the next prompt on (none: alone again).
    Team(Team),
}

// The models a session works with besides its own (provider/id, in the order they speak after
// it), and how. A pair answers back and forth in the one chat until a model says it's done or the
// round limit, then the user speaks; a trio is the session's model answering, the other checking
// that once, then the user.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Team {
    pub models: Vec<String>,
    pub pair: bool,
}

// The note that opens each model's turn with a team: what the model reads, and in the conversation
// one line saying whose turn it is.
pub const PAIR_TYPE: &str = "piccosta-pair";
// The session entry that records the team, so a reopened session still works with it.
pub const PAIR_SETTING: &str = "piccosta-pair-partner";
// How many rounds (every model speaking once) a pair goes before it's the user's turn without
// either saying it's done; `pairRounds` in settings changes it. Three is a guess, not measured:
// it lets each model answer the other's corrections twice, and each turn rereads the whole chat.
const PAIR_ROUNDS: u64 = 3;
// What a model in a pair ends its reply with, alone on the last line, when the work is done and it
// has nothing to add. A line of its own is easy to ask for and to find, unlike a phrase in prose;
// the brackets keep a plain closing "Done." from ending it by accident.
const DONE_LINE: &str = "[done]";

// The team a session last recorded (none when it works alone). Recordings before pairs answered
// back and forth (one `partner`, or a list of `partners` without `pair`) open as trios.
pub fn team(session: &Session) -> Team {
    let recorded = session.path_to_leaf().into_iter().rev().find_map(|entry| match entry {
        session::SessionEntry::Custom { custom_type, data, .. } if custom_type == PAIR_SETTING => Some(data.clone().unwrap_or_default()),
        _ => None,
    });
    let Some(recorded) = recorded else { return Team::default() };
    let models = match (recorded.get("partners").and_then(Value::as_array), recorded.get("partner").and_then(Value::as_str)) {
        (Some(list), _) => list.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        (None, Some(one)) => vec![one.to_string()],
        _ => Vec::new(),
    };
    Team { models, pair: recorded.get("pair").and_then(Value::as_bool).unwrap_or(false) }
}

// Record a session's team, each model by its full provider/id so any model the registry has can
// join, whichever provider it's on. Output: the team as recorded.
pub fn record_team(registry: &ModelRegistry, session: &mut Session, team: &Team) -> Result<Team, String> {
    let models = team.models.iter().map(|name| registry.resolve(name).map(|model| format!("{}/{}", model.provider, model.id))).collect::<Result<Vec<_>, _>>()?;
    let data = serde_json::json!({ "partners": models, "pair": team.pair });
    session.append_custom(PAIR_SETTING, Some(data)).map_err(|error| format!("could not write the session: {error}"))?;
    Ok(Team { models, pair: team.pair })
}

// The note opening turn `turn` of an exchange: turn 0 is the session model's answer to the user's
// message, each turn after is the next speaker's reply to the one before. `speakers` are the
// session's model then its team's; `last` says the user speaks after this turn. It tells the model
// who it's working with, what this turn is for, and how the exchange ends.
fn team_note(team: &Team, speakers: &[String], turn: usize, rounds: u64, last: bool) -> crate::message::CustomMessage {
    let speaker = &speakers[turn % speakers.len()];
    let previous = turn.checked_sub(1).map(|before| speakers[before % speakers.len()].clone());
    let others: Vec<&str> = speakers.iter().map(String::as_str).filter(|other| other != speaker).collect();
    let next = if last { "the user" } else { &speakers[(turn + 1) % speakers.len()] };
    let (kind, how, check) = if team.pair {
        let how = format!("You take turns in this chat, back and forth, without waiting for the user. When the work is done and you have nothing to add, end your reply with a line that says only {DONE_LINE}; that, or {rounds} rounds each, hands it back to the user.");
        ("pair", how, "Respond to it: check it against what the user asked, fix what is wrong, finish or build on what was left, and say plainly what you changed.")
    } else {
        ("trio", "One of you answers the user, the other checks that answer, then the user speaks.".to_string(), "Check the work against what the user asked: verify what it claims, fix what is wrong, finish what was left undone, and say plainly what you changed, or that it holds up.")
    };
    let task = match &previous {
        Some(previous) => format!("{previous} has just replied above. {check}"),
        None => "Answer the user's message below.".to_string(),
    };
    let text = format!("{speaker}: you're in a {kind} with {}, and the user reads you all. {how} {task} Then stop; {next} speaks next.", others.join(" and "));
    let details = serde_json::json!({ "mode": kind, "speaker": speaker, "after": previous });
    crate::message::CustomMessage { custom_type: PAIR_TYPE.into(), content: crate::message::TextOrBlocks::Text(text), display: true, details: Some(details), timestamp: crate::clock::now_ms() }
}

// Whether an exchange stops before the next turn: the last reply failed or was stopped, or (in a
// pair, once the first answer has been answered) its last line is the done line, give or take
// markdown around it.
fn exchange_over(session: &Session, done_counts: bool) -> bool {
    let last_reply = session.path_to_leaf().into_iter().rev().find_map(|entry| match entry {
        session::SessionEntry::Message { message: AgentMessage::Assistant(reply), .. } => Some(reply),
        _ => None,
    });
    let Some(reply) = last_reply else { return true };
    if matches!(reply.stop_reason, crate::message::StopReason::Error | crate::message::StopReason::Aborted) {
        return true;
    }
    let text: String = reply.content.iter().filter_map(|block| if let crate::message::Content::Text { text, .. } = block { Some(text.as_str()) } else { None }).collect();
    let last_line = text.lines().rev().map(|line| line.trim_matches(|character: char| character.is_whitespace() || "*_`.".contains(character))).find(|line| !line.is_empty());
    done_counts && last_line.is_some_and(|line| line.eq_ignore_ascii_case(DONE_LINE))
}

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
    // Someone said it's done (session.rs, closed sessions).
    pub closed: bool,
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

    // The model's tools: the built-in ones, the subagent tool (not in a subagent, which can't hand
    // its task on) and, for a manager (`role`, in `cwd`), its own; cut to --tools when given.
    // A session that can hand work on leaves the tools a kind of work owns to that kind's job, so
    // searching (say) runs on the scale's search models; a manager keeps them all, and so does a
    // subagent, which has no one to hand them to. --tools naming one keeps it. A session started
    // for a kind of work (a subagent's job among them) gets exactly the tools its kind lists, owned
    // or not, else all of them; and none that write when it's read-only, since edit and write run
    // here, outside its commands' sandbox (work). --exclude-tools still takes any away.
    fn tool_list(&self, session: &Session) -> Vec<Arc<dyn tools::Tool>> {
        let (cwd, role) = (Path::new(&session.header().cwd), crate::manager::role(session));
        let (work_tools, work_read_only) = self.work(session);
        if !self.tools {
            return Vec::new();
        }
        let mut list = tools::builtin();
        let subagents = std::env::var_os(tools::subagent::CHILD_ENV).is_none().then(|| tools::subagent::SubagentTool::load(&self.agent_dir)).flatten();
        let owned: Vec<String> = match (&subagents, role, crate::manager::work(session)) {
            (Some(_), None, None) => crate::scale::owned(&self.agent_dir).into_iter().map(|(tool, _)| tool).collect(),
            _ => Vec::new(),
        };
        list.extend(subagents.map(|tool| Arc::new(tool) as Arc<dyn tools::Tool>));
        if let Some(role) = role {
            list.extend(crate::manager::tools(&self.agent_dir, &self.settings, role, cwd));
        }
        let named = |tool: &Arc<dyn tools::Tool>| self.tool_names.as_ref().is_some_and(|names| names.iter().any(|name| name == tool.name()));
        let wanted = |tool: &Arc<dyn tools::Tool>| match &work_tools {
            // A kind's list, narrowed further by an explicit --tools.
            Some(names) => names.iter().any(|name| name == tool.name()) && (self.tool_names.is_none() || named(tool)),
            None => (self.tool_names.is_none() && !owned.iter().any(|name| name == tool.name())) || named(tool),
        };
        let writes = |tool: &Arc<dyn tools::Tool>| work_read_only && matches!(tool.name(), "edit" | "write");
        list.into_iter().filter(|tool| wanted(tool) && !writes(tool) && !self.exclude_tools.iter().any(|name| name == tool.name())).collect()
    }

    // The kind of work a session was started for (the sessions tool's `use`, manager.rs work): its
    // tools (None for all) and whether it's read-only. A session started for none gets (None,
    // false); one whose kind has since left the scale is read-only, so it doesn't gain rights.
    fn work(&self, session: &Session) -> (Option<Vec<String>>, bool) {
        let Some(kind) = crate::manager::work(session) else { return (None, false) };
        let Some((_, entry)) = crate::scale::jobs(&self.agent_dir).into_iter().find(|(name, _)| *name == kind) else { return (None, true) };
        let tools: Vec<String> = entry.get("tools").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect();
        (Some(tools).filter(|tools| !tools.is_empty()), entry.get("readOnly").and_then(Value::as_bool) == Some(true))
    }

    // The prompt sections for a model (its constitution can differ by model), with a manager's
    // part when the session is one. The instructions the session holds stand in when a context
    // file can't be read (prompt::sections).
    pub fn sections(&self, model: &Model, session: &Session) -> Result<IndexMap<String, String>, String> {
        let (cwd, role) = (Path::new(&session.header().cwd), crate::manager::role(session));
        let messages = crate::message::to_model_messages(&session.context().messages);
        let held_context = crate::provider::transcript::current_system_message(&messages).and_then(|system| system.sections?.shift_remove("project_context")?);
        let constitution = prompt::constitution(&model.name)?;
        let (custom_prompt, discovered_append) = prompt::custom_prompts(&self.agent_dir);
        let append_prompt = self.append_prompt.clone().or(discovered_append);
        let manager = role.map(|role| crate::manager::prompt(&self.settings, role, cwd));
        let append_prompt = if manager.is_some() { Some([manager, append_prompt].into_iter().flatten().collect::<Vec<_>>().join("\n\n")) } else { append_prompt };
        let tool_list = self.tool_list(session);
        Ok(prompt::sections(&prompt::PromptInputs { cwd, agent_dir: &self.agent_dir, tools: &tool_list, custom_prompt, append_prompt, constitution, context_files: self.context_files, held_context }))
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
            .or_else(|| self.settings.default_model())
            .or_else(|| registry.models().first().map(|first| format!("{}/{}", first.provider, first.id)));
        let wanted = wanted.ok_or_else(|| format!("no model: no connected account serves one; log in to one, or pass --model or set defaultModel in {}", self.agent_dir.join("settings.json").display()))?;
        // A session whose model is no longer offered opens on the default model, and with that gone
        // too, on the first model a connected account serves, rather than not at all (the status
        // line shows which). A model asked for by name still has to exist.
        let fallback = || self.settings.default_model().and_then(|default| registry.resolve(&default).ok()).or(registry.models().first());
        let model = match (registry.resolve(&wanted), model) {
            (Ok(found), _) => found.clone(),
            (Err(error), Some(_)) => return Err(error),
            (Err(error), None) => fallback().ok_or(error)?.clone(),
        };
        let has_thinking = resumed.messages.iter().any(|message| matches!(message, AgentMessage::Assistant(_)));
        // The level asked for, else the one this model was last set to by hand (in any session, so
        // a reopened one follows the latest choice, as Claude Code's effort does), else what the
        // session last ran at, else the default.
        let remembered = thinking.map(str::to_string).or_else(|| self.remembered_thinking_level(&model));
        let thinking_level = remembered.unwrap_or_else(|| if has_thinking { resumed.thinking_level.clone() } else { self.settings.default_thinking_level() });
        let cwd = PathBuf::from(&session.header().cwd);
        let role = crate::manager::role(&session);
        // Managers are the user's: an agent's command (a test, a lane) doesn't run one in the
        // user's own agent dir, whether by `piccosta assistant`, a print run or the terminal UI.
        // A scratch agent dir (PICCOSTA_AGENT_DIR) is for testing them.
        if role.is_some() {
            crate::manager::users_only()?;
        }
        let sections = self.sections(&model, &session)?;

        let (notice_sender, notices) = tokio::sync::mpsc::unbounded_channel();
        let credentials = auth::CredentialPaths::from_environment(&self.agent_dir);
        let mut tool_shared = ToolShared::new(&self.settings, &self.agent_dir, notice_sender, &registry, credentials.clone());
        if role.is_some() {
            tool_shared.jobs.soft_limit = crate::manager::SOFT_LIMIT;
        }
        tool_shared.read_only |= self.work(&session).1;
        let tool_shared = Arc::new(tool_shared);
        // A reopened session's stored web results, so its responseIds still page.
        tool_shared.web.store.restore(session.path_to_leaf());
        let compaction = self.settings.compaction(&model.provider, &model.id);
        let manager_home = role.map(|role| crate::manager::home_for(&self.agent_dir, role, &cwd));
        let config = AgentConfig {
            model,
            registry,
            credentials,
            thinking_level,
            tools: self.tool_list(&session),
            tool_shared: tool_shared.clone(),
            cwd,
            sections,
            context_files: self.context_files,
            retry: self.settings.retry(),
            compaction,
            transport: self.settings.transport(),
            notices,
            gate: None,
            hold_for_jobs,
            manager_home,
        };
        let agent = Agent::new(session, config, emit).map_err(|error| format!("could not write the session: {error}"))?;
        Ok((agent, tool_shared))
    }
}

impl Host {
    // The model and level a command line asks for: --model and --thinking as given, else what the
    // scale picks for --use (its level unless --thinking says otherwise).
    pub fn launch(&self, args: &crate::cli::Args) -> Result<(Option<String>, Option<String>), String> {
        match (crate::cli::model_name(args), &args.use_scale) {
            (None, Some(name)) => {
                let (model, level) = crate::scale::pick(&self.agent_dir, name)?;
                Ok((Some(model), args.thinking.clone().or(level)))
            }
            (model, _) => Ok((model, args.thinking.clone())),
        }
    }

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
            let mut team = team(&agent.session);
            loop {
                let (result, notice) = tokio::select! {
                    command = commands.recv() => match command {
                        Some(command) if !commands.is_closed() => (self.obey(&mut agent, command, &mut team).await, false),
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
            // The session is closed: a Claude query waiting for its next message goes too.
            crate::provider::claude::close(agent.session.id());
            shared.jobs.shutdown("quit");
            shared.web.shutdown();
        })
    }

    // Carry out one command. `team` is the models the session works with.
    async fn obey(&self, agent: &mut Agent, command: Command, team: &mut Team) -> Result<(), String> {
        let written = |result: std::io::Result<()>| result.map_err(|error| format!("could not write the session: {error}"));
        match command {
            Command::Prompt(text, images) => self.answer(agent, team, &text, images).await,
            Command::Team(asked) => {
                *team = record_team(&self.registry()?, &mut agent.session, &asked)?;
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
        let sections = self.sections(&model, &agent.session)?;
        let compaction = self.settings.compaction(&model.provider, &model.id);
        let level = self.remembered_thinking_level(&model).unwrap_or_else(|| agent.thinking_level().to_string());
        written(agent.set_model(model, sections, compaction))?;
        written(agent.set_thinking_level(&level))
    }

    // Answer the user's message, with the session's team when it has one. Alone it's one run. In
    // a trio the session model answers, then each other model checks once. In a pair they take
    // turns until one ends its reply with the done line, or the round limit. Each turn is the
    // speaker's model on the session for one run, opened by its note; the session model takes it
    // back for the user's next message. A failed or stopped turn ends the exchange.
    pub async fn answer(&self, agent: &mut Agent, team: &Team, text: &str, images: Vec<crate::message::Content>) -> Result<(), String> {
        let lead = format!("{}/{}", agent.model().provider, agent.model().id);
        let speakers: Vec<String> = std::iter::once(lead.clone()).chain(team.models.iter().filter(|model| **model != lead).cloned()).collect();
        let rounds = self.settings.get("pairRounds").and_then(Value::as_u64).unwrap_or(PAIR_ROUNDS).max(1);
        let turns = match speakers.len() {
            1 => 1,
            count if team.pair => rounds as usize * count,
            count => count,
        };
        if turns > 1 {
            agent.hold_notice(team_note(team, &speakers, 0, rounds, false));
        }
        agent.prompt_with_images(text, images).await?;
        let mut result = Ok(());
        for turn in 1..turns {
            if agent.aborted() || exchange_over(&agent.session, team.pair && turn > 1) {
                break;
            }
            result = self.switch_model(agent, &speakers[turn % speakers.len()]);
            if result.is_err() {
                break;
            }
            result = agent.take_notice(team_note(team, &speakers, turn, rounds, turn + 1 == turns)).await;
            if result.is_err() {
                break;
            }
        }
        if turns > 1 {
            self.switch_model(agent, &lead)?;
        }
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
    let closed = session::closed_sessions(&config::agent_dir());
    files.into_iter().filter_map(|(modified, path)| {
        let session = Session::open(&path, cwd.unwrap_or(dir)).ok()?;
        let recorded = Path::new(&session.header().cwd);
        cwd.is_none_or(|cwd| recorded == cwd).then(|| SessionSummary {
            title: title(&session), id: session.id().to_string(), cwd: recorded.to_path_buf(), closed: closed.contains_key(&crate::tools::paths::canonical(&path)), path, modified,
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
