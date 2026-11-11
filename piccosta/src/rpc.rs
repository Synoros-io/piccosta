// pi's rpc mode (pi 1.4.0 modes/rpc/rpc-mode.ts and rpc-types.ts, MIT): commands as JSON lines on
// stdin, responses and the session's events as JSON lines on stdout, so a frontend that drives pi
// this way (T3 Code's pi provider) drives piccosta. A response is {id, type: "response", command,
// success, data | error}, without an id when the command had none; events are json mode's. One
// session is served at a time; new_session, switch_session, fork and clone replace it. piccosta
// has no extensions: it sends extension_ui_request only to ask approval (confirm) and to tell the
// user something (notify), and reads extension_ui_response only for its confirms.

use crate::agent::{Agent, Event, EventSink, Gate, Handle};
use crate::host::Host;
use crate::runtime::{Command, Done};
use crate::message::{AgentMessage, Content};
use crate::model::Model;
use crate::session::{Session, SessionEntry};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

// How long a session being replaced, or the last one at exit, gets to stop its run.
const STOP_WAIT: Duration = Duration::from_secs(5);
// T3 Code's approval setting, which its pi extension's tool_call hook reads
// (mcpExtensionSource.ts:317-334 in pingdotgg/t3code). piccosta runs no JavaScript extensions, so
// rpc mode asks for approval itself, the same way: for every tool but the read-only ones (and
// edit and write under auto-accept-edits), an extension_ui_request confirm titled "Allow <tool>?"
// with the arguments as JSON, cut to APPROVAL_ARGUMENT_CHARS. The read-only ones are the tools
// whose read_only() says so, the same as ACP's.
const RUNTIME_MODE_ENV: &str = "T3_PI_RUNTIME_MODE";
const APPROVAL_ARGUMENT_CHARS: usize = 4000;
const BUSY: &str = "Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message.";
const ACP_QUEUE_UNSUPPORTED: &str = "ACP agents do not support RPC steering or follow-ups. Submit a new prompt after the current turn finishes.";

enum Input {
    Line(String),
    Closed,
    // From the task of the session with this generation; older sessions' are dropped.
    Event(u64, Box<Event>),
    Done(u64, Box<Done>),
    // A tool call waiting for the user's approval.
    Ask { title: String, message: String, reply: tokio::sync::oneshot::Sender<bool> },
    Choose(crate::provider::Interaction, tokio::sync::oneshot::Sender<Result<Value, String>>),
}

// The session being served, and what this loop knows of it while its task has the agent.
struct Live {
    generation: u64,
    handle: Handle,
    // Its tools' shared state: a steering message moves the running commands to the background.
    shared: Arc<crate::tools::ToolShared>,
    commands: UnboundedSender<Command>,
    task: Option<tokio::task::JoinHandle<Agent>>,
    id: String,
    file: Option<PathBuf>,
    // With no file (--no-session, which T3 Code uses for titles and other generated text): the
    // session as this loop has seen it, from the messages the run reported, since there is nothing
    // on disk to read.
    memory: Option<Session>,
    name: Option<String>,
    model: Model,
    thinking: String,
    auto_compaction: bool,
    streaming: bool,
    // Owned completions ahead of the accepted prompt, and its API until it starts.
    pending_prompt: Option<(usize, String)>,
    compacting: bool,
    // One per command sent to the task, oldest first: the response it still owes, if any.
    owed: VecDeque<Option<(Option<Value>, &'static str)>>,
    // The last compaction's outcome, for compact's response.
    compaction: Option<Result<Value, String>>,
    // abort commands waiting for the run to stop.
    aborts: Vec<Option<Value>>,
}

impl Live {
    // The executing turn wins, then its pre-start snapshot; only idle controls use the picker.
    fn acp_turn(&self) -> bool {
        self.handle.running_api().as_deref().or(self.pending_prompt.as_ref().map(|(_, api)| api.as_str())).unwrap_or(&self.model.api) == crate::provider::acp_client::API
    }

    // One owned FIFO slot just completed. Older commands advance the position; only this
    // prompt's own slot settles it when no agent_start/agent_settled events were emitted.
    fn command_finished(&mut self) {
        let Some((commands_ahead, _)) = &mut self.pending_prompt else { return };
        if *commands_ahead > 0 { *commands_ahead -= 1; }
        else { self.pending_prompt = None; self.streaming = false; }
    }
}

struct Rpc {
    host: Arc<Host>,
    inputs: UnboundedSender<Input>,
    live: Live,
    generations: u64,
    steering_mode: String,
    follow_up_mode: String,
    // Approvals asked and not answered, by request id.
    asks: HashMap<String, tokio::sync::oneshot::Sender<bool>>,
    // False under --no-session: new sessions stay in memory too.
    persist: bool,
    choices: HashMap<String, (crate::provider::Interaction, tokio::sync::oneshot::Sender<Result<Value, String>>)>,
}

enum Reply {
    Data(Option<Value>),
    Later,
}

pub async fn run(args: crate::cli::Args) -> Result<ExitCode, String> {
    tokio::task::LocalSet::new().run_until(serve(args)).await
}

async fn serve(args: crate::cli::Args) -> Result<ExitCode, String> {
    let host = Arc::new(Host::for_args(&args));
    host.show_settings_errors();
    crate::scale::watch();
    let (inputs, mut received) = unbounded_channel();
    // stdin on a plain thread, as print mode reads it (main.rs read_piped_stdin says why).
    let lines = inputs.clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else { break };
            if lines.send(Input::Line(line)).is_err() {
                return;
            }
        }
        let _ = lines.send(Input::Closed);
    });
    let session = host.opening(&args)?;
    let (model, thinking, note) = host.launch(&args)?;
    if let Some(listing) = host.listing(Some(&session), model.as_deref()) {
        listing.await;
    }
    let live = start(&host, &inputs, 0, session, model.as_deref(), thinking.as_deref())?;
    // A kind with none of its models served: the frontend shows which model it runs on, as pi's
    // notify (an extension_ui_request no response answers).
    if let Some(note) = note {
        let note = crate::scale::fell_back(&note, &format!("{}/{}", live.model.provider, live.model.id));
        write_line(&json!({"type": "extension_ui_request", "id": uuid::Uuid::new_v4().to_string(), "method": "notify", "message": note, "notifyType": "warning"}));
    }
    let mut rpc = Rpc { host, inputs, live, generations: 1, steering_mode: "one-at-a-time".into(), follow_up_mode: "one-at-a-time".into(), asks: HashMap::new(), persist: !args.no_session, choices: HashMap::new() };
    if let Some(name) = args.name.clone().filter(|name| !name.trim().is_empty()) {
        rpc.live.name = Some(name.clone());
        rpc.send(Command::Name(name), None);
    }
    let termination = crate::proc::termination(false);
    tokio::pin!(termination);
    let code = loop {
        let input = tokio::select! {
            input = received.recv() => input,
            code = &mut termination => break code,
        };
        match input {
            Some(Input::Line(line)) => rpc.line(&line).await,
            Some(Input::Event(generation, event)) if generation == rpc.live.generation => rpc.event(*event),
            Some(Input::Done(generation, done)) if generation == rpc.live.generation => rpc.done(*done),
            Some(Input::Ask { title, message, reply }) => rpc.ask(title, message, reply),
            Some(Input::Choose(request, reply)) => rpc.choose(request, reply),
            Some(Input::Closed) | None => break 0,
            Some(_) => {}
        }
    };
    let live = rpc.live;
    drop(live.commands);
    if let Some(task) = live.task {
        let _ = tokio::time::timeout(STOP_WAIT, task).await;
    }
    crate::provider::shutdown().await;
    Ok(ExitCode::from(code))
}

// Start serving a session: its agent on a task whose events and reports come back as inputs.
fn start(host: &Arc<Host>, inputs: &UnboundedSender<Input>, generation: u64, session: Session, model: Option<&str>, thinking: Option<&str>) -> Result<Live, String> {
    let events = inputs.clone();
    let emit: EventSink = Box::new(move |event: &Event| {
        let _ = events.send(Input::Event(generation, Box::new(event.clone())));
    });
    // A prompt stays open until the background jobs it started end, so a job's message reaches the
    // model; T3 Code stops a pi that starts a turn on its own.
    let (mut agent, shared) = host.open(session, model, thinking, emit, true)?;
    // On disk at once: the loop answers get_entries from the file, and T3 Code passes the path to
    // other processes before the first prompt.
    agent.session.persist_now().map_err(|error| format!("could not write the session: {error}"))?;
    agent.set_gate(approval_gate(inputs, host.yolo));
    let questions = inputs.clone();
    agent.set_interaction(Arc::new(move |request| {
        let questions = questions.clone();
        Box::pin(async move {
            let (reply, answer) = tokio::sync::oneshot::channel();
            questions.send(Input::Choose(request, reply)).map_err(|_| "Frontend disconnected")?;
            answer.await.map_err(|_| "Frontend cancelled the question")?
        })
    }));
    let reports = inputs.clone();
    let (handle, id, file, name) = (agent.handle(), agent.session.id().to_string(), agent.session.file().map(PathBuf::from), crate::host::name(&agent.session));
    let (memory, model, thinking, auto_compaction) = (agent.session.file().is_none().then(|| agent.session.clone()), agent.model().clone(), agent.thinking_level().to_string(), agent.auto_compaction());
    let (commands, task) = crate::runtime::spawn(host.clone(), agent, shared.clone(), false, false, move |done| {
        let _ = reports.send(Input::Done(generation, Box::new(done)));
    });
    let task = Some(task);
    let (streaming, compacting, owed, compaction, aborts) = (false, false, VecDeque::new(), None, Vec::new());
    Ok(Live { generation, handle, shared, commands, task, id, file, memory, name, model, thinking, auto_compaction, streaming, pending_prompt: None, compacting, owed, compaction, aborts })
}

// Ordinary tools follow T3 Code's mode. A plugin's own question still needs this call's yes.
fn approval_policy(yolo: bool) -> (bool, bool) {
    let mode = std::env::var(RUNTIME_MODE_ENV).ok();
    (yolo || mode.as_deref().is_none_or(|mode| mode == "full-access"), mode.as_deref() == Some("auto-accept-edits"))
}

fn approval_gate(inputs: &UnboundedSender<Input>, yolo: bool) -> Gate {
    let (full, edits_free) = approval_policy(yolo);
    let inputs = inputs.clone();
    std::rc::Rc::new(move |_call_id: String, name: String, arguments: Map<String, Value>, question: Option<String>, read_only: bool| {
        let inputs = inputs.clone();
        Box::pin(async move {
            if question.is_none() && (full || read_only || (edits_free && matches!(name.as_str(), "edit" | "write"))) { return Ok(()); }
            let arguments = Value::Object(arguments).to_string();
            let message: String = arguments.chars().take(APPROVAL_ARGUMENT_CHARS).collect();
            let (reply, answer) = tokio::sync::oneshot::channel();
            let _ = inputs.send(Input::Ask { title: question.unwrap_or_else(|| format!("Allow {name}?")), message, reply });
            match answer.await {
                Ok(true) => Ok(()),
                _ => Err(format!("{name} was declined in T3 Code.")),
            }
        })
    })
}

pub(crate) fn write_line(value: &Value) {
    let mut stdout = std::io::stdout().lock();
    let line = format!("{value}\n");
    if stdout.write_all(line.as_bytes()).and_then(|()| stdout.flush()).is_err() {
        // The frontend went away: nobody is left to serve.
        std::process::exit(1);
    }
}

fn respond(id: Option<Value>, command: &str, result: Result<Option<Value>, String>) {
    let mut response = Map::new();
    if let Some(id) = id {
        response.insert("id".into(), id);
    }
    response.insert("type".into(), "response".into());
    response.insert("command".into(), command.into());
    match result {
        Ok(data) => {
            response.insert("success".into(), true.into());
            if let Some(data) = data {
                response.insert("data".into(), data);
            }
        }
        Err(error) => {
            response.insert("success".into(), false.into());
            response.insert("error".into(), error.into());
        }
    }
    write_line(&Value::Object(response));
}

impl Rpc {
    async fn line(&mut self, line: &str) {
        let command: Value = match serde_json::from_str(line) {
            Ok(command) => command,
            Err(error) => return respond(None, "parse", Err(format!("Failed to parse command: {error}"))),
        };
        let id = command.get("id").cloned();
        let kind = command.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        if kind == "extension_ui_response" {
            if let Some((request, reply)) = command.get("id").and_then(Value::as_str).and_then(|id| self.choices.remove(id)) {
                let _ = reply.send(Ok(request.answer_label(command["value"].as_str())));
                return;
            }
            // An approval's answer: confirmed true allows the call; false or cancelled refuses it.
            if let Some(reply) = command.get("id").and_then(Value::as_str).and_then(|id| self.asks.remove(id)) {
                let _ = reply.send(command.get("confirmed").and_then(Value::as_bool).unwrap_or(false));
            }
            return;
        }
        match self.command(&kind, &command, id.clone()).await {
            Ok(Reply::Data(data)) => respond(id, &kind, Ok(data)),
            Ok(Reply::Later) => {}
            Err(error) => respond(id, &kind, Err(error)),
        }
    }

    // Hand a command to the session's task; `owed` is the response it answers when done.
    fn send(&mut self, command: Command, owed: Option<(Option<Value>, &'static str)>) {
        if self.live.commands.send(command).is_ok() {
            self.live.owed.push_back(owed);
        } else if let Some((id, name)) = owed {
            respond(id, name, Err("the session stopped".into()));
        }
    }

    async fn command(&mut self, kind: &str, command: &Value, id: Option<Value>) -> Result<Reply, String> {
        let text = |key: &str| command.get(key).and_then(Value::as_str).unwrap_or("").to_string();
        let live = &mut self.live;
        let data = match kind {
            "prompt" => {
                let message = text("message");
                let message = crate::prompt::command_prompt(&message).unwrap_or(message);
                if live.streaming {
                    if live.acp_turn() && matches!(command["streamingBehavior"].as_str(), Some("steer" | "followUp")) { return Err(ACP_QUEUE_UNSUPPORTED.into()); }
                    match command.get("streamingBehavior").and_then(Value::as_str) {
                        Some("steer") => {
                            live.handle.steer(message);
                            live.shared.jobs.detach_all();
                        }
                        Some("followUp") => live.handle.follow_up(message),
                        _ => return Err(BUSY.into()),
                    }
                    self.announce_queue();
                    return Ok(Reply::Data(Some(json!({"disposition": "queued"}))));
                }
                let images: Vec<Content> = command.get("images").and_then(Value::as_array).into_iter().flatten().filter_map(|image| serde_json::from_value(image.clone()).ok()).collect();
                // The response goes before the run's first event, as pi's preflight answers.
                respond(id, "prompt", Ok(Some(json!({"disposition": "started"}))));
                live.streaming = true;
                live.pending_prompt = Some((live.owed.len(), live.model.api.clone()));
                self.send(Command::Prompt(message, images), None);
                return Ok(Reply::Later);
            }
            "steer" | "follow_up" => {
                if live.acp_turn() { return Err(ACP_QUEUE_UNSUPPORTED.into()); }
                let message = text("message");
                if kind == "steer" { live.handle.steer(message) } else { live.handle.follow_up(message) }
                self.announce_queue();
                Some(json!({"disposition": "queued"}))
            }
            "abort" | "abort_retry" => {
                let _ = live.commands.send(Command::Stop(None));
                if live.streaming && kind == "abort" {
                    live.aborts.push(id);
                    return Ok(Reply::Later);
                }
                None
            }
            "abort_bash" => None,
            "clear_queue" => {
                let (steering, follow_up) = live.handle.clear();
                self.announce_queue();
                Some(json!({"steering": steering, "followUp": follow_up}))
            }
            "new_session" => {
                let parent = command.get("parentSession").and_then(Value::as_str).map(str::to_string);
                self.replace(
                    |rpc| {
                        let mut session = Session::create(&rpc.host.cwd, rpc.persist.then_some(rpc.host.session_dir.as_path()));
                        if let Some(parent) = &parent {
                            session.set_parent(parent);
                        }
                        Ok(session)
                    },
                    true,
                )
                .await?;
                Some(json!({"cancelled": false}))
            }
            "switch_session" => {
                let path = PathBuf::from(text("sessionPath"));
                self.replace(|rpc| Session::open(&path, &rpc.host.cwd).map_err(|error| error.to_string()), false).await?;
                Some(json!({"cancelled": false}))
            }
            "fork" => {
                // Checked before anything stops, so a bad id leaves the session running.
                let entry_id = text("entryId");
                let selected = match self.read()?.entry(&entry_id) {
                    Some(SessionEntry::Message { message: AgentMessage::User(user), .. }) => user.content.text(),
                    _ => return Err("Invalid entry ID for forking".into()),
                };
                self.replace(
                    |rpc| {
                        let current = rpc.read()?;
                        let parent = current.entry(&entry_id).and_then(SessionEntry::parent_id).map(str::to_string);
                        current.branch(parent.as_deref(), &rpc.host.session_dir).map_err(|error| error.to_string())
                    },
                    true,
                )
                .await?;
                Some(json!({"text": selected, "cancelled": false}))
            }
            "clone" => {
                if self.read()?.leaf_id().is_none() {
                    return Err("Cannot clone session: no current entry selected".into());
                }
                self.replace(
                    |rpc| {
                        let current = rpc.read()?;
                        let leaf = current.leaf_id().map(str::to_string);
                        current.branch(leaf.as_deref(), &rpc.host.session_dir).map_err(|error| error.to_string())
                    },
                    true,
                )
                .await?;
                Some(json!({"cancelled": false}))
            }
            "get_state" => Some(self.state()?),
            "set_model" => {
                let (provider, model_id) = (text("provider"), text("modelId"));
                let registry = self.host.registry()?;
                let model = registry.models().iter().find(|model| model.provider == provider && model.id == model_id);
                let model = model.ok_or(format!("Model not found: {provider}/{model_id}"))?.clone();
                self.live.model = model.clone();
                self.send(Command::Model(format!("{provider}/{model_id}")), None);
                // The level the task switches to with the model (its own, else this one), told now.
                let level = self.host.remembered_thinking_level(&model).unwrap_or_else(|| self.live.thinking.clone());
                let level = crate::provider::clamp_thinking_level(&model, &level).to_string();
                if level != self.live.thinking {
                    write_line(&json!({"type": "thinking_level_changed", "level": level}));
                    self.live.thinking = level;
                }
                Some(serde_json::to_value(model).unwrap_or_default())
            }
            // piccosta has no scoped model list to cycle through.
            "cycle_model" => Some(Value::Null),
            "get_available_models" => Some(json!({"models": self.host.usable_models().await?})),
            "set_thinking_level" => {
                self.set_thinking(&text("level"), true);
                None
            }
            "cycle_thinking_level" => {
                let levels = crate::provider::thinking_levels(&self.live.model);
                if levels.len() <= 1 {
                    Some(Value::Null)
                } else {
                    let position = levels.iter().position(|level| *level == self.live.thinking).unwrap_or(0);
                    let next = levels[(position + 1) % levels.len()].to_string();
                    self.set_thinking(&next, true);
                    Some(json!({"level": next}))
                }
            }
            "get_available_thinking_levels" => Some(json!({"levels": crate::provider::thinking_levels(&self.live.model)})),
            "set_steering_mode" => {
                self.steering_mode = text("mode");
                self.live.handle.set_steering_mode(&self.steering_mode);
                None
            }
            "set_follow_up_mode" => {
                self.follow_up_mode = text("mode");
                self.live.handle.set_follow_up_mode(&self.follow_up_mode);
                None
            }
            "compact" => {
                self.live.compaction = None;
                let instructions = command.get("customInstructions").and_then(Value::as_str).map(str::to_string);
                self.send(Command::Compact(instructions), Some((id, "compact")));
                return Ok(Reply::Later);
            }
            "set_auto_compaction" | "set_auto_retry" => {
                let enabled = command.get("enabled").and_then(Value::as_bool).unwrap_or(true);
                if kind == "set_auto_compaction" {
                    self.live.auto_compaction = enabled;
                    self.send(Command::AutoCompaction(enabled), None);
                } else {
                    self.send(Command::AutoRetry(enabled), None);
                }
                None
            }
            "get_session_stats" => Some(self.stats()?),
            "get_entries" => {
                let session = self.read()?;
                let mut entries: Vec<&SessionEntry> = session.entries().iter().collect();
                if let Some(since) = command.get("since").and_then(Value::as_str) {
                    let found = entries.iter().position(|entry| entry.id() == since).ok_or(format!("Entry not found: {since}"))?;
                    entries.drain(..=found);
                }
                let entries: Vec<Value> = entries.into_iter().map(entry_json).collect();
                Some(json!({"entries": entries, "leafId": session.leaf_id()}))
            }
            "get_fork_messages" => {
                let session = self.read()?;
                let messages: Vec<Value> = session
                    .path_to_leaf()
                    .into_iter()
                    .filter_map(|entry| match entry {
                        SessionEntry::Message { base, message: AgentMessage::User(user) } => Some(json!({"entryId": base.id, "text": user.content.text()})),
                        _ => None,
                    })
                    .collect();
                Some(json!({"messages": messages}))
            }
            "get_last_assistant_text" => {
                let messages = self.read()?.context().messages;
                let text = messages.iter().rev().find_map(|message| match message {
                    AgentMessage::Assistant(assistant) => {
                        let text: Vec<&str> = assistant.content.iter().filter_map(|block| if let Content::Text { text, .. } = block { Some(text.as_str()) } else { None }).collect();
                        Some(text.join(""))
                    }
                    _ => None,
                });
                Some(json!({"text": text}))
            }
            "set_session_name" => {
                let name = text("name").trim().to_string();
                if name.is_empty() {
                    return Err("Session name cannot be empty".into());
                }
                self.live.name = Some(name.clone());
                write_line(&json!({"type": "session_info_changed", "name": name}));
                self.send(Command::Name(name), None);
                None
            }
            "get_messages" => Some(json!({"messages": self.read()?.context().messages})),
            // pi's RpcSlashCommand: piccosta's prompt commands, as built-in prompt templates.
            "get_commands" => {
                let commands: Vec<Value> = crate::prompt::PROMPT_COMMANDS
                    .iter()
                    .map(|(name, description)| json!({"name": name, "description": description, "source": "prompt", "sourceInfo": {"path": format!("builtin:{name}"), "source": "builtin", "scope": "user", "origin": "top-level"}}))
                    .collect();
                Some(json!({"commands": commands}))
            }
            "bash" | "export_html" | "get_tree" => return Err(format!("{kind} isn't supported by piccosta's rpc mode")),
            _ => return Err(format!("Unknown command: {kind}")),
        };
        Ok(Reply::Data(data))
    }

    // pi's queue_update, after the queue changed here.
    fn announce_queue(&self) {
        let (steering, follow_up) = self.live.handle.queued();
        write_line(&json!({"type": "queue_update", "steering": steering, "followUp": follow_up}));
    }

    // A thinking level for the session, clamped to its model; thinking_level_changed when that
    // changes it, as pi says so. The task records it after any run in progress. A level the client
    // picked is saved for the model's next sessions; a model switch's own isn't (`picked`).
    fn set_thinking(&mut self, level: &str, picked: bool) {
        let level = crate::provider::clamp_thinking_level(&self.live.model, level).to_string();
        if picked {
            let model = format!("{}/{}", self.live.model.provider, self.live.model.id);
            if let Err(error) = crate::config::remember_thinking_level(&self.host.agent_dir, &model, Some(&level)) {
                eprintln!("piccosta: {error}");
            }
        }
        if level != self.live.thinking {
            write_line(&json!({"type": "thinking_level_changed", "level": level}));
            self.live.thinking = level.clone();
        }
        self.send(Command::Thinking(level), None);
    }

    fn event(&mut self, event: Event) {
        let live = &mut self.live;
        match &event {
            Event::AgentStart => { live.streaming = true; live.pending_prompt = None; }
            Event::AgentSettled { .. } => {
                live.streaming = false;
                live.pending_prompt = None;
                // An approval still open when the run ended refuses its call.
                self.asks.clear();
            }
            Event::MessageEnd { message } => {
                if let Some(memory) = &mut live.memory {
                    let _ = memory.append_message(message.clone());
                }
            }
            Event::CompactionStart { .. } => live.compacting = true,
            Event::CompactionEnd { result, error_message, aborted, .. } => {
                live.compacting = false;
                live.compaction = Some(match (result, error_message) {
                    (Some(result), _) => Ok(result.clone()),
                    (None, Some(error)) => Err(error.strip_prefix("Compaction failed: ").unwrap_or(error).to_string()),
                    (None, None) => Err(if *aborted { "Compaction cancelled" } else { "Nothing to compact" }.into()),
                });
            }
            _ => {}
        }
        write_line(&serde_json::to_value(&event).unwrap_or_default());
        // A run's aborts are answered once it has settled, after agent_settled, as pi answers.
        if matches!(event, Event::AgentSettled { .. }) {
            for id in self.live.aborts.drain(..) {
                respond(id, "abort", Ok(None));
            }
        }
    }

    fn done(&mut self, done: Done) {
        self.live.model = done.model;
        self.live.thinking = done.thinking;
        if !done.frontend {
            return;
        }
        let Some(owed) = self.live.owed.pop_front() else { return };
        self.live.command_finished();
        let Some((id, name)) = owed else { return };
        let result = match done.result {
            Err(error) => Err(error),
            Ok(()) if name == "compact" => self.live.compaction.take().unwrap_or_else(|| Err("Nothing to compact".into())).map(Some),
            Ok(()) => Ok(None),
        };
        respond(id, name, result);
    }

    // Serve another session in place of this one. The old one stops first (its run aborted, its
    // task ended), so a switch to the same file opens it with everything the old agent wrote.
    // `session` is built from the file after that. `keep_model` carries the current model and
    // thinking level over (new, fork, clone); a switched-to session runs with its own. If the new
    // one can't start, the old file is served again.
    async fn replace(&mut self, session: impl FnOnce(&Rpc) -> Result<Session, String>, keep_model: bool) -> Result<(), String> {
        let (closed, _) = unbounded_channel();
        drop(std::mem::replace(&mut self.live.commands, closed));
        if let Some(task) = self.live.task.take() {
            let _ = tokio::time::timeout(STOP_WAIT, task).await;
        }
        for (id, name) in std::mem::take(&mut self.live.owed).into_iter().flatten() {
            respond(id, name, Err("the session was replaced".into()));
        }
        for id in std::mem::take(&mut self.live.aborts) {
            respond(id, "abort", Ok(None));
        }
        self.asks.clear();
        let model = format!("{}/{}", self.live.model.provider, self.live.model.id);
        let thinking = self.live.thinking.clone();
        let (model, thinking) = if keep_model { (Some(model.as_str()), Some(thinking.as_str())) } else { (None, None) };
        let generation = self.generations;
        self.generations += 1;
        // A switched-to session's model may still be being listed; there's no session to serve
        // meanwhile, so what comes in waits its turn.
        let session = session(self);
        if let Some(listing) = session.as_ref().ok().and_then(|session| self.host.listing(Some(session), model)) {
            listing.await;
        }
        let started = session.and_then(|session| start(&self.host, &self.inputs, generation, session, model, thinking));
        match started {
            Ok(live) => {
                live.handle.set_steering_mode(&self.steering_mode);
                live.handle.set_follow_up_mode(&self.follow_up_mode);
                self.live = live;
                Ok(())
            }
            Err(error) => {
                let old = self.read()?;
                self.live = start(&self.host, &self.inputs, generation, old, model, thinking)?;
                self.live.handle.set_steering_mode(&self.steering_mode);
                self.live.handle.set_follow_up_mode(&self.follow_up_mode);
                Err(error)
            }
        }
    }

    // Ask the frontend to approve a tool call (pi's ctx.ui.confirm as an extension_ui_request).
    fn ask(&mut self, title: String, message: String, reply: tokio::sync::oneshot::Sender<bool>) {
        let id = uuid::Uuid::new_v4().to_string();
        write_line(&json!({"type": "extension_ui_request", "id": id, "method": "confirm", "title": title, "message": message}));
        self.asks.insert(id, reply);
    }

    fn choose(&mut self, request: crate::provider::Interaction, reply: tokio::sync::oneshot::Sender<Result<Value, String>>) {
        if reply.is_closed() { return; }
        self.choices.retain(|_, (_, reply)| !reply.is_closed());
        let (full, edits_free) = approval_policy(self.host.yolo);
        let automatic = full || (edits_free && request.params.pointer("/toolCall/kind").and_then(Value::as_str) == Some("edit"));
        if let Some(answer) = request.auto_allow(automatic) { let _ = reply.send(Ok(answer)); return; }
        let options = request.labels();
        if options.is_empty() { let _ = reply.send(Err("This frontend does not support elicitation forms".into())); return; }
        let id = uuid::Uuid::new_v4().to_string();
        write_line(&json!({"type": "extension_ui_request", "id": id, "method": "select", "title": request.title(), "options": options}));
        self.choices.insert(id, (request, reply));
    }

    // The session as its file holds it now (the task has the agent), or as this loop has seen it
    // when there is no file; a file not written yet is empty.
    fn read(&self) -> Result<Session, String> {
        match (&self.live.file, &self.live.memory) {
            (Some(file), _) if file.exists() => Session::open(file, &self.host.cwd).map_err(|error| error.to_string()),
            (_, Some(memory)) => Ok(memory.clone()),
            _ => Ok(Session::create(&self.host.cwd, None)),
        }
    }

    // pi's RpcSessionState.
    fn state(&self) -> Result<Value, String> {
        let live = &self.live;
        let (steering, follow_up) = live.handle.queued();
        let mut state = Map::new();
        state.insert("model".into(), serde_json::to_value(&live.model).unwrap_or_default());
        state.insert("thinkingLevel".into(), live.thinking.clone().into());
        state.insert("isStreaming".into(), live.streaming.into());
        state.insert("isCompacting".into(), live.compacting.into());
        state.insert("steeringMode".into(), self.steering_mode.clone().into());
        state.insert("followUpMode".into(), self.follow_up_mode.clone().into());
        if let Some(file) = &live.file {
            state.insert("sessionFile".into(), file.to_string_lossy().into_owned().into());
        }
        state.insert("sessionId".into(), live.id.clone().into());
        if let Some(name) = &live.name {
            state.insert("sessionName".into(), name.clone().into());
        }
        state.insert("autoCompactionEnabled".into(), live.auto_compaction.into());
        state.insert("messageCount".into(), self.read()?.context().messages.len().into());
        state.insert("pendingMessageCount".into(), (steering.len() + follow_up.len()).into());
        Ok(Value::Object(state))
    }

    // pi's SessionStats (agent-session.ts getSessionStats), with the context and the session's
    // file and id.
    fn stats(&self) -> Result<Value, String> {
        let session = self.read()?;
        let mut stats = session_stats(&session);
        let window = self.live.model.context_window;
        stats.insert("contextUsage".into(), if window > 0 { context_usage(&session, window) } else { Value::Null });
        stats.insert("sessionFile".into(), json!(self.live.file));
        stats.insert("sessionId".into(), json!(self.live.id));
        stats.insert("planUsage".into(), crate::scale::plan_usage(&self.host.agent_dir, &self.live.model.provider));
        Ok(Value::Object(stats))
    }
}

// pi's SessionStats counts (agent-session.ts getSessionStats): messages and tool calls over every
// entry, and tokens and cost summed from responses, tool results, usage entries and compactions.
// The TUI's /usage shows the same numbers.
pub fn session_stats(session: &Session) -> serde_json::Map<String, Value> {
    let mut stats = crate::session::index::Statistics::default();
    for entry in session.entries() { stats.add(entry); }
    stats.value()
}

// pi's ContextUsage (agent-session.ts getContextUsage): the projected context's size, or null
// tokens after a compaction until a response measures the new context.
fn context_usage(session: &Session, window: u64) -> Value {
    let branch = session.path_to_leaf();
    let projection = session.projection();
    if let Some(at) = branch.iter().rposition(|entry| matches!(entry, SessionEntry::Compaction { .. })) {
        let measured: std::collections::HashSet<&str> = projection
            .iter()
            .filter(|entry| entry.messages.iter().any(|message| crate::compaction::assistant_usage(message).is_some()))
            .map(|entry| entry.source.id())
            .collect();
        if !branch[at + 1..].iter().any(|entry| measured.contains(entry.id())) {
            return json!({"tokens": null, "contextWindow": window, "percent": null});
        }
    }
    let tokens = crate::compaction::estimate_projected_context_tokens(&projection, &branch).tokens;
    json!({"tokens": tokens, "contextWindow": window, "percent": tokens as f64 / window as f64 * 100.0})
}

// An entry as its line in the session file.
fn entry_json(entry: &SessionEntry) -> Value {
    match entry {
        SessionEntry::Unknown { raw, .. } => raw.clone(),
        known => serde_json::to_value(known).unwrap_or_default(),
    }
}
