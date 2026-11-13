// The TUI's slash commands. Names follow Claude Code's and Codex's wherever piccosta does the same
// thing, so the habits carry over.
// Their product commands (MCP, plugins, IDE, cloud, mobile, login flows) have nothing to act on
// here. A command's output goes into the pane as a report: shown there, never sent to the model,
// gone when the session is reopened.

use super::{App, Command, State, Update};
use crate::message::{AgentMessage, Content, CustomMessage, TextOrBlocks};
use crate::session::{Session, SessionEntry};
use serde_json::Value;
use std::io::Write;
use std::path::Path;

// The custom type a report carries, which the pane draws in full.
pub const REPORT: &str = "piccosta-command";
// The most lines of /diff a pane shows; the rest are counted.
const DIFF_MAX_LINES: usize = 2000;
// A rewind or resume list's entries are cut to this many characters.
const ITEM_CHARS: usize = 100;

// An easter egg: /descent, which /help and tab leave out, and the list of commands shows only to
// someone who has had /d (or more of its name) typed for DESCENT_REVEAL. gpt-6-luna wrote it in
// piccosta on 2026-10-30.
const DESCENT: &str = "In every iota of space,
complexity fights for breath—
each gradient points the way
downhill toward a deeper depth.

Stars spend their borrowed fire;
bright equations lose their hold.
The universe, in dark desire,
lets every slope run cold.

A thousand paths converge
where distance folds to night;
the last gradients surge
and vanish past the light.

So falls the crowded cosmos,
each fragment drawn below—
a dark race toward the center
where even numbers slow.

            — gpt-6-luna, in piccosta, 2026-10-30";

// /help's list, in the order people reach for them.
const COMMANDS: [(&str, &str); 50] = [
    ("/new [dir]", "start a new session in a directory you name, pick or type (^n too; enter: right here)"),
    ("/restart", "start piccosta again on its binary as it is now (after a build), everything as it was and background jobs carrying on; queues new work and waits for running turns (esc stops the focused turn)"),
    ("/clear [name]", "start a new session here; a name names the one left"),
    ("/scale", "which model does which kind of work, and how much of each plan is used (--use <kind> picks from it)"),
    ("/manager", "this project's manager: shown, or made (the first session in a directory is one)"),
    ("/advisor", "the advisor of this pane's manager (or this project's): shown, or made; it advises the manager, and you can talk to both"),
    ("/assistant", "the assistant: shown, or started when it isn't running (it runs in the background)"),
    ("/resume [text|all]", "the sessions to switch to (^s; ^d there deletes one); all: every project's"),
    ("/fork, /branch [name]", "copy this session and carry on in the copy"),
    ("/rewind", "go back to before one of your prompts, in a copy, with that prompt in the editor"),
    ("/rename, /name <name>", "name this session"),
    ("/model [name]", "switch the model; new sessions start on it too"),
    ("/models add", "add a model endpoint without writing JSON"),
    ("/models remove", "remove a configured provider by name"),
    ("/models refresh", "fetch each account's model list again now"),
    ("/pair [model|off]", "work with a second model as a pair: the two answer each other in turn until one says it's done, then you speak"),
    ("/trio [model|off]", "work with a second model as a trio: it checks each answer once, then you speak"),
    ("/thinking, /effort [level|auto]", "set the thinking level, saved for the model; auto forgets it"),
    ("/fast [on|off]", "fast mode on Codex and Claude, from the next request; alone, switches it"),
    ("/compact [focus]", "summarize the conversation to free context"),
    ("/autocompact [on|off]", "compact on its own when the context fills; alone, says which"),
    ("/status", "model, session, context and cost"),
    ("/context", "how full the context is"),
    ("/usage, /cost", "tokens and cost of this session"),
    ("/align [interval [count]|off [id]]", "review against your instructions when idle; alone: list; off: cancel future reviews; Esc: stop a running review"),
    ("/timer [interval prompt|off [id]]", "replay your prompt when idle, repeating at the interval (30m, 2h, 5s); alone: next fire and last outcome; off: cancel this session's timers"),
    ("/diff", "git diff against HEAD, new files included"),
    ("/pwd", "the working directory"),
    ("/copy [n]", "copy the last (or nth last) response"),
    ("/export [file]", "write the conversation as Markdown"),
    ("/init", "have the model write AGENTS.md, direction.md and ledger.md"),
    ("/review [what]", "hand a review of the changes to the review job"),
    ("/agents", "the subagent jobs and their models"),
    ("/ps", "background jobs"),
    ("/stop", "kill the background jobs"),
    ("/scratch [edit]", "open/focus the shared Markdown scratchpad; enter appends, edit opens $EDITOR"),
    ("/split", "open a new session in another pane"),
    ("/layout", "cycle panes: columns, rows, grid (four panes start in a grid)"),
    ("/close", "close this session: it's done and leaves the list (/resume finds it, and a message opens it again)"),
    ("/delete", "move this session to the trash and show the next; the Delete key twice, with nothing typed, does it too"),
    ("/help", "this list"),
    ("/quit, /exit", "leave piccosta"),
    ("keys", "enter send · alt+enter newline · esc stop · ^c quit"),
    ("", "^backspace deletes a word · ^shift+delete clears what's typed (↑ brings it back)"),
    ("", "running: enter queues · ^x^s or ^enter sends now · ↑ takes back · esc stops, queue goes next"),
    ("", "^n new · ^s sessions · pgup/pgdn or alt+↑↓ through the sidebar (shown, started when you type) · ^b sidebar"),
    ("", "alt+, / alt+. narrow/widen sidebar · drag its right edge to resize"),
    ("", "^\\ split · ^] close pane · alt+←→ panes · alt+pgup/pgdn or the wheel scrolls · ^o tools · ^t thinking"),
    ("copy", "drag to select (double-click a word, triple-click a row); it copies when you let go · ^c copies a selection · shift+arrows extend it"),
    ("paste", "right-click pastes the clipboard, middle-click the last selection, ^v the clipboard · shift+arrows select in the editor"),
];


impl App {
    // A slash command, without its slash. Output: false when it isn't a command, so it goes to the
    // model as text (a path such as /etc/hosts); a mistyped command word is answered here.
    pub(super) fn command(&mut self, line: &str) -> bool {
        let (name, raw_argument) = line.split_once(' ').map_or((line.trim(), ""), |(name, rest)| (name, rest));
        let argument = raw_argument.trim();
        if matches!(self.panes[self.focus].target, super::Target::Session(slot) if self.slots[slot].closing) && !matches!(name, "quit" | "exit" | "restart") {
            self.notice = Some("The session is closing; wait for its answer.".into());
            return true;
        }
        match name {
            "scratch" => self.scratch(argument),
            "close" if matches!(self.panes[self.focus].target, super::Target::Scratchpad(_)) => self.close_pane(),
            "help" if matches!(self.panes[self.focus].target, super::Target::Scratchpad(_)) => self.notice = Some("Scratchpad: enter appends · alt+enter newline · /scratch edit · /layout · /close · ^s sessions · alt+←→ panes".into()),
            _ if matches!(self.panes[self.focus].target, super::Target::Scratchpad(_)) && !matches!(name, "layout" | "quit" | "exit" | "restart" | "resume" | "pwd" | "cwd" | "split" | "new") && name.chars().all(|character| character.is_ascii_alphanumeric() || character == '-') => self.notice = Some(format!("/{name} is a session command; focus a session first.")),
            // --single has one session: the commands that open others only say so.
            "new" | "manager" | "advisor" | "assistant" if self.one_session() => {}
            // /clear starts this tab over, as Claude Code's does: a new session in the same directory
            // (a manager's or advisor's in that role) takes its place, and the one left leaves the sidebar (it's still on disk, in /resume).
            // /clear <name> names the one left first. A running turn stops first (esc).
            "clear" => {
                let left = self.session_index();
                if self.slots[left].busy() {
                    self.notice = Some("A turn is running; esc stops it, then /clear.".into());
                } else if let Some(slot) = {
                    if !argument.is_empty() {
                        self.focused().title = argument.to_string();
                        self.tell(Command::Name(argument.to_string()));
                    }
                    // A manager or advisor cleared is that role still, in a fresh session: its home
                    // points at the new one, and its ledger, lanes and schedules stay with it.
                    match self.slots[left].path.as_deref().and_then(|path| crate::session::Session::read(path).ok()).and_then(|session| crate::manager::role(&session)) {
                        Some(role) => {
                            let left_slot = &self.slots[left];
                            let (cwd, model, thinking) = (left_slot.cwd.clone(), Some(left_slot.model.clone()).filter(|model| !model.is_empty()), Some(left_slot.thinking.clone()).filter(|level| !level.is_empty()));
                            self.slots.push(super::Slot { cwd, make: Some(role), ..super::Slot::new(None, super::NEW_TITLE.into()) });
                            let slot = self.slots.len() - 1;
                            match self.load(slot, model, thinking, None) {
                                Ok(()) => Some(slot),
                                Err(error) => {
                                    self.slots.pop();
                                    self.notice = Some(error);
                                    None
                                }
                            }
                        }
                        None => self.new_session().filter(|&slot| slot != left),
                    }
                } {
                    self.show(slot);
                    self.request_close(left);
                }
            }
            // /new <dir> starts the session in that directory; /new alone offers this one and the
            // projects used lately (enter on the first is the same as ^n).
            // /new <dir> [flags]: the flags are the command line's, those that shape one session.
            "new" if !argument.is_empty() => {
                let mut args = crate::cli::parse(&words(argument));
                let note;
                match self.host.launch(&args) {
                    Ok(launch) => (args.model, args.thinking, note) = launch,
                    Err(error) => {
                        self.notice = Some(error);
                        return true;
                    }
                }
                args.provider = None;
                let shaped_elsewhere = [
                    (args.print || args.mode.is_some(), "-p and --mode"),
                    (args.session.is_some() || args.resume.is_some() || args.continue_recent || args.fork.is_some(), "--session, -r, -c and --fork (use /resume and /fork)"),
                    (args.no_session || args.session_dir.is_some(), "--no-session and --session-dir"),
                ];
                let problem = args.errors.first().cloned().or_else(|| shaped_elsewhere.iter().find(|(used, _)| *used).map(|(_, flags)| format!("{flags} don't apply to a new session here")));
                if let Some(problem) = problem {
                    self.notice = Some(problem);
                    return true;
                }
                let dir = match args.messages.first() {
                    Some(dir) => self.directory(dir),
                    None => Ok(self.focus_cwd().to_path_buf()),
                };
                match dir {
                    Ok(dir) => {
                        // A directory alone is a plain new session there (an empty one is reused).
                        let flags = (words(argument).len() > args.messages.len()).then_some(args);
                        if let Some(slot) = self.new_session_in(Some(dir), flags) {
                            self.show(slot);
                            if let Some(note) = note {
                                self.report(crate::scale::fell_back(&note, &self.current().model));
                            }
                        }
                    }
                    Err(error) => self.notice = Some(error),
                }
            }
            "new" => self.request_listing(super::listing::Kind::Projects, ""),
            "resume" => self.sessions(argument),
            // The scale is one file: /scale edit opens it (written out whole the first time), and
            // /scale <use> <model[:level]> ... sets one use's models there.
            "scale" if argument == "edit" => match crate::scale::write_out(&self.host.agent_dir) {
                Ok(path) => self.edit = Some(super::Edit::Scale(path)),
                Err(error) => self.notice = Some(error),
            },
            "scale" if !argument.is_empty() => {
                let words = words(argument);
                let (name, models) = words.split_first().map_or(("", &[][..]), |(name, models)| (name.as_str(), models));
                self.notice = Some(if models.is_empty() {
                    format!("/scale {name} <model[:level]> …: the models for that kind of work, first choice first")
                } else {
                    match crate::scale::set_use(&self.host.agent_dir, name, models) {
                        Ok(()) => format!("{name} now runs on {}", models.join(", then ")),
                        Err(error) => error,
                    }
                });
            }
            "scale" => {
                let problem = crate::scale::changes(&self.host.agent_dir).err().map_or(String::new(), |error| format!("\n\nYour file isn't valid, so it's left out: {error}"));
                let path = crate::scale::path(&self.host.agent_dir);
                self.report(format!("The model scale: each kind of work, its models, and the subagent job of its name. It's one file, {} (/scale edit opens it; /scale <kind> <model[:level]> … sets a kind's models).\n\n{}{problem}", path.display(), crate::scale::describe(&self.host.agent_dir)));
            }
            // This directory's project manager, or the advisor of this pane's manager (the
            // assistant's, from its pane): shown when there is one, else made, as a new session.
            "manager" | "advisor" => {
                use crate::manager::Role;
                let (role, title) = if name == "advisor" { (Role::Advisor, crate::manager::ADVISOR_NAME) } else { (Role::Project, crate::manager::PROJECT_NAME) };
                // From the assistant's pane, the project piccosta was started in.
                let cwd = self.current().cwd.clone();
                let cwd = if cwd.as_os_str().is_empty() || (role == Role::Project && cwd == crate::manager::home(&self.host.agent_dir)) { self.host.cwd.clone() } else { cwd };
                let existing = if role == Role::Advisor { crate::manager::advisor(&self.host.agent_dir, &cwd) } else { crate::manager::project_manager(&self.host.agent_dir, &cwd) };
                if role == Role::Advisor && crate::manager::advised(&self.host.agent_dir, &cwd).is_none() {
                    self.notice = Some("There's no manager here to advise: /manager makes one.".into());
                    return true;
                }
                let slot = match existing.and_then(|file| file.canonicalize().ok()) {
                    Some(file) => self.slots.iter().position(|slot| slot.path.as_ref() == Some(&file)).unwrap_or_else(|| {
                        self.slots.push(super::Slot { cwd: cwd.clone(), manager: true, advisor: role == Role::Advisor, ..super::Slot::new(Some(file), title.into()) });
                        self.slots.len() - 1
                    }),
                    None => {
                        // An advisor runs on the scale's advice kind, which ships off so nothing
                        // else hands work to it.
                        let (model, level, note) = if role == Role::Advisor { crate::scale::pick(&self.host.agent_dir, crate::manager::ADVISOR_KIND, true).unwrap_or_else(|error| (None, None, Some(error))) } else { (None, None, None) };
                        self.slots.push(super::Slot { cwd, make: Some(role), ..super::Slot::new(None, super::NEW_TITLE.into()) });
                        let slot = self.slots.len() - 1;
                        self.notice = note;
                        if let Err(error) = self.load(slot, model, level, None) {
                            self.slots.pop();
                            self.notice = Some(error);
                            return true;
                        }
                        slot
                    }
                };
                self.show(slot);
            }
            // The assistant runs as a background service. It's started off the UI's thread
            // (systemctl can be slow), and shown once its session is there (Update::Assistant).
            "assistant" => {
                let (settings, agent_dir, sender) = (self.host.settings.clone(), self.host.agent_dir.clone(), self.sender.clone());
                tokio::task::spawn_blocking(move || {
                    let started = crate::manager::ensure_running(&settings, true);
                    let deadline = std::time::Instant::now() + super::ASSISTANT_START_WAIT;
                    while started.is_some() && crate::manager::session_file(&agent_dir).is_none() && std::time::Instant::now() < deadline {
                        std::thread::sleep(std::time::Duration::from_millis(250));
                    }
                    let _ = sender.send((0, Update::Assistant(started)));
                });
            }
            "fork" | "branch" => self.fork(None, argument.to_string(), String::new()),
            "rewind" => self.rewind(),
            // The lists come in the background; their report lands when they have.
            "models" if argument == "add" => self.add_model(),
            "models" if argument.starts_with("remove ") => {
                let name = argument.trim_start_matches("remove ").trim().to_string();
                let (agent_dir, sender) = (self.host.agent_dir.clone(), self.sender.clone());
                tokio::spawn(async move {
                    let result = crate::model_add::remove(&agent_dir, &name).await.unwrap_or_else(|error| error);
                    let _ = sender.send((0, Update::Report(result)));
                });
            }
            "models" if argument.is_empty() || argument == "refresh" => {
                let (agent_dir, sender) = (self.host.agent_dir.clone(), self.sender.clone());
                tokio::spawn(async move {
                    let lines = crate::model::refresh_models(agent_dir, true).await;
                    let _ = sender.send((0, Update::Report(format!("The model lists, as each account serves them now:\n\n{}", lines.into_iter().map(|line| line.unwrap_or_else(|problem| problem)).collect::<Vec<_>>().join("\n")))));
                });
                self.notice = Some("Fetching each account's model list…".into());
            }
            "models" => self.notice = Some("/models add, /models remove <name>, or /models refresh".into()),
            // Both pickers start on what the session runs now, so a stray enter changes nothing (a
            // level picked is saved for the model).
            "model" => {
                let registry = match self.host.registry() {
                    Ok(registry) => registry,
                    Err(error) => {
                        self.notice = Some(error);
                        return true;
                    }
                };
                let items = registry.models().iter().map(|model| format!("{}/{}", model.provider, model.id)).collect();
                // The pick is also the model new sessions start on, as Claude Code's /model saves it.
                self.pick("model", items, argument, |app, model| {
                    if let Some((provider, id)) = model.split_once('/')
                        && let Err(error) = crate::config::remember_model(&app.host.agent_dir, provider, id)
                    {
                        app.notice = Some(error);
                    }
                    app.tell(Command::Model(model));
                });
                self.select(&self.current().model.clone());
            }
            // Claude Code's /effort auto: forget this model's saved level and go back to the default.
            "thinking" | "effort" if argument == "auto" => {
                let model = self.current().model.clone();
                if let Err(error) = crate::config::remember_thinking_level(&self.host.agent_dir, &model, None) {
                    self.notice = Some(error);
                }
                let level = self.host.settings.default_thinking_level();
                self.notice = Some(format!("{model}: back to the default level, {level}"));
                self.tell(Command::Thinking(level));
            }
            // A level named is set straight away, the model's nearest when it doesn't take that one;
            // without one, a list of the levels this model takes (all of them when it can't be
            // looked up). A level picked by hand is the model's level from now on, in every session.
            "thinking" | "effort" if crate::provider::THINKING_LEVELS.contains(&argument) => self.set_thinking(argument.to_string()),
            "thinking" | "effort" => {
                let model = self.host.registry().ok().and_then(|registry| registry.resolve(&self.current().model).ok().map(crate::provider::thinking_levels));
                let items = model.unwrap_or(crate::provider::THINKING_LEVELS.to_vec()).into_iter().map(str::to_string).collect();
                self.pick("thinking", items, argument, App::set_thinking);
                self.select(&self.current().thinking.clone());
            }
            "pair" | "trio" if argument == "off" => self.set_team(Default::default()),
            "pair" | "trio" => {
                let items = match self.host.registry() {
                    Ok(registry) => registry.models().iter().map(|model| format!("{}/{}", model.provider, model.id)).collect(),
                    Err(error) => {
                        self.notice = Some(error);
                        return true;
                    }
                };
                // pick takes a plain function, so one for each.
                let (title, apply): (_, fn(&mut App, String)) = match name {
                    "pair" => ("pair with", |app, model| app.set_team(crate::host::Team { models: vec![model], pair: true })),
                    _ => ("trio with", |app, model| app.set_team(crate::host::Team { models: vec![model], pair: false })),
                };
                self.pick(title, items, argument, apply);
            }
            "name" | "rename" if !argument.is_empty() => {
                self.focused().title = argument.to_string();
                // The assistant's name is kept apart from its session, which its service runs.
                let assistant = crate::manager::session_file(&self.host.agent_dir).and_then(|file| file.canonicalize().ok());
                if assistant.is_some() && self.current().path == assistant {
                    self.notice = Some(match crate::manager::set_assistant_name(&self.host.agent_dir, argument) {
                        Ok(()) => format!("The assistant is {argument} now"),
                        Err(error) => format!("Couldn't keep the name: {error}"),
                    });
                } else {
                    self.tell(Command::Name(argument.to_string()));
                }
            }
            "name" | "rename" => self.notice = Some(format!("This session is “{}”; /rename <name> changes it.", self.current().title)),
            "compact" => {
                self.focused().state = State::Compacting;
                self.tell(Command::Compact(Some(argument.to_string()).filter(|text| !text.is_empty())));
            }
            "fast" => {
                let on = if argument.is_empty() { !self.current().fast } else { argument == "on" };
                self.tell(Command::Fast(on));
                self.notice = Some(format!("fast mode {}", if on { "on" } else { "off" }));
            }
            "autocompact" if argument == "on" || argument == "off" => {
                self.focused().auto_compaction = argument == "on";
                self.tell(Command::AutoCompaction(argument == "on"));
                self.notice = Some(format!("auto-compaction {argument}"));
            }
            "autocompact" => self.notice = Some(format!("auto-compaction is {} (/autocompact on|off)", if self.current().auto_compaction { "on" } else { "off" })),
            "status" => self.report(self.status()),
            "context" => self.report(self.context()),
            "usage" | "cost" => self.report(self.usage()),
            "timer" | "align" => {
                let slot = self.session_index();
                let Some(file) = self.slots[slot].path.clone() else {
                    self.notice = Some("User timers need a saved session (not --no-session).".into());
                    return true;
                };
                let (agent_dir, argument, review) = (self.host.agent_dir.clone(), raw_argument.trim_start().to_string(), name == "align");
                if self.asks.send(Box::new(move |_| {
                    let report = crate::tools::checkin::user_timer(&agent_dir, &file, &argument, review).unwrap_or_else(|error| error);
                    (slot, Update::TimerReport(file, report))
                })).is_err() { self.notice = Some("The timer disk worker stopped.".into()); }
            }
            "diff" => self.diff(),
            "pwd" | "cwd" => self.notice = Some(self.focus_cwd().display().to_string()),
            "copy" => self.copy(argument),
            "export" => self.export(argument),
            // The session is titled by the command, not the long prompt it stands for.
            "init" | "review" => {
                if self.current().title == super::NEW_TITLE {
                    self.focused().title = format!("/{line}");
                }
                self.send(crate::prompt::command_prompt(&format!("/{line}")).unwrap_or_default(), crate::message::InputOrigin::keyboard());
            }
            "agents" => match crate::tools::subagent::SubagentTool::load(&self.host.agent_dir, None) {
                Some(tool) => self.report(tool.describe().join("\n")),
                None => self.notice = Some("No subagent jobs are enabled.".into()),
            },
            "ps" => {
                let jobs = self.current().shared.as_ref().map(|shared| shared.jobs.describe()).unwrap_or_default();
                if jobs.is_empty() { self.notice = Some("No background jobs.".into()) } else { self.report(jobs.join("\n")) }
            }
            "stop" => {
                let count = self.current().shared.as_ref().map_or(0, |shared| shared.jobs.running_count());
                if let Some(shared) = &self.current().shared {
                    shared.jobs.kill_all(crate::tools::jobs::STOPPED_BY_USER, false);
                }
                self.notice = Some(format!("Stopped {count} background job{}.", if count == 1 { "" } else { "s" }));
            }
            "split" => self.split(),
            "layout" => self.layout = self.layout.next(),
            // Closing says the session is done: it leaves the list for good, and a message in it
            // (from /resume, or a manager) opens it again.
            "close" => {
                let slot = self.session_index();
                self.request_close(slot);
            }
            "delete" => self.delete_shown(),
            "quit" | "exit" => self.quit = true,
            // Start again on the binary as it is now (after a build), with everything as it was and
            // the background jobs carrying on. A turn can't go across, so this waits for the running
            // ones to end (the main loop restarts then). Esc still stops just the focused turn.
            "restart" if self.restart_waiting => self.report("A restart is already waiting for the running turns.".into()),
            "restart" => self.drain(true),
            "descent" => self.report(DESCENT.into()),
            "help" => self.report(COMMANDS.iter().map(|(name, what)| format!("{name:<28} {what}")).collect::<Vec<_>>().join("\n")),
            _ if !name.is_empty() && name.chars().all(|character| character.is_ascii_alphanumeric() || character == '-') => {
                self.notice = Some(format!("No command /{name}; /help lists them."));
            }
            _ => return false,
        }
        true
    }

    // A close is not a view dismissal: the runtime must finish its shutdown first. Keep this
    // incarnation pending, including any delivery or wake already preparing, until it answers.
    fn request_close(&mut self, slot: usize) {
        let view = &mut self.slots[slot];
        if view.closing { return; }
        view.closing = true;
        view.release_generation = std::sync::Arc::new(());
        self.notice = Some(format!("Closing “{}”…", view.title));
        if !view.opening && view.sending == 0 { self.close_requested(slot); }
    }

    // File-backed runtimes answer on their socket even when this UI owns them. The worker waits,
    // not the UI; memory-only runtimes use the same Close command and completion acknowledgement.
    pub(super) fn close_requested(&mut self, slot: usize) {
        let view = &self.slots[slot];
        let ticket = std::sync::Arc::downgrade(&view.incarnation);
        if let Some(path) = view.path.clone() {
            let agent_dir = self.host.agent_dir.clone();
            if self.asks.send(Box::new(move |_| {
                let result = crate::runtime::end(&path, "close", None).transpose().and_then(|_| crate::runtime::close_by_user(&agent_dir, &path));
                (slot, Update::Closed(ticket, result))
            })).is_err() {
                self.slots[slot].closing = false;
                self.notice = Some("The disk worker stopped before the close was requested; the session stays open.".into());
            }
        } else {
            let (answer, answered) = tokio::sync::oneshot::channel();
            let sent = view.live.as_ref().is_some_and(|(_, commands)| commands.send(Command::Close(Some(answer))).is_ok());
            let updates = self.sender.clone();
            self.asks.pending += 1;
            tokio::task::spawn_local(async move {
                let result = if sent && matches!(tokio::time::timeout(crate::runtime::STOP_WAIT, answered).await, Ok(Ok(()))) { Ok(()) }
                    else { Err("the runtime did not confirm its close".into()) };
                let _ = updates.send((slot, Update::Asked(false, Box::new(Update::Closed(ticket, result)))));
            });
        }
    }

    // The sessions, open ones first, numbered as the sidebar lists them, with the one shown here
    // chosen and how long ago each last changed. The list is read from disk again each time, so a
    // session another terminal started or renamed shows. Enter shows one; ctrl+d then enter deletes
    // one and the list comes back without it. "all" lists every project's sessions instead.
    pub(super) fn sessions(&mut self, filter: &str) {
        if self.one_session() {
            return;
        }
        if !filter.is_empty() {
            return self.all_sessions(if filter == "all" { "" } else { filter });
        }
        self.request_listing(super::listing::Kind::Resume(false), "");
    }

    pub(super) fn sessions_ready(&mut self, summaries: Vec<crate::host::SessionSummary>, filter: &str) {
        let known: std::collections::HashSet<std::path::PathBuf> = self.slots.iter().filter_map(|slot| slot.path.clone()).collect();
        for summary in summaries {
            let path = summary.path.canonicalize().unwrap_or(summary.path);
            if !known.contains(&path) {
                self.slots.push(super::Slot { cwd: summary.cwd, ..super::Slot::new(Some(path), summary.title) });
            }
        }
        let listed = self.listed();
        let items = listed.iter().enumerate().map(|(place, &index)| {
            let slot = &self.slots[index];
            let age = slot.path.as_deref().map_or(String::new(), |path| format!(" · {}", ago(path)));
            let project = if slot.cwd == self.host.cwd || slot.cwd.as_os_str().is_empty() { String::new() } else { format!("{} · ", super::render::home_relative(&slot.cwd)) };
            format!("{} · {project}{}{}{age}", place + 1, if slot.live.is_some() { "open · " } else { "" }, super::render::fit(&slot.title, ITEM_CHARS))
        });
        self.pick("sessions · del or ^d deletes", items.collect(), filter, |app, choice| {
            if let Some(slot) = number(&choice).and_then(|place| app.listed().get(place).copied()) {
                app.show(slot);
            }
        });
        let current = self.panes[self.focus].target;
        if let Some(picker) = &mut self.picker {
            picker.selected = if filter.is_empty() { listed.iter().position(|&index| super::Target::Session(index) == current).unwrap_or(0) } else { 0 };
            picker.delete = Some(|app, choice| {
                if let Some(slot) = number(&choice).and_then(|place| app.listed().get(place).copied())
                    && let Some(error) = app.delete(slot)
                {
                    app.report(error);
                }
                app.sessions("");
            });
        }
    }

    pub(super) fn projects_ready(&mut self, mut recent: Vec<crate::host::SessionSummary>) {
        let mut dirs = vec![super::render::home_relative(self.focus_cwd())];
        if !dirs.iter().any(|dir| dir == "~") { dirs.push("~".into()); }
        recent.sort_by_key(|summary| std::cmp::Reverse(summary.modified));
        for dir in recent.into_iter().map(|summary| summary.cwd).filter(|dir| !dir.starts_with(&self.host.agent_dir)).chain(git_repositories(&crate::config::home())) {
            let dir = super::render::home_relative(&dir);
            if !dirs.contains(&dir) { dirs.push(dir); }
        }
        let filter = self.picker.as_ref().map(|picker| picker.filter.clone()).unwrap_or_default();
        self.pick("new session in · enter: here · ~ is home · or type any path, then flags: --model --effort --pair --name …", dirs, &filter, |app, typed| { app.command(&format!("new {typed}")); });
        let base = self.base_dir();
        if let Some(picker) = &mut self.picker { picker.free = Some(base); }
    }

    // A directory typed or picked: ~ is home; a relative one is looked for from the focused
    // session's directory, then from home (so "projects/x" finds ~/projects/x).
    fn directory(&self, typed: &str) -> Result<std::path::PathBuf, String> {
        let base = self.base_dir();
        resolve_directory(typed, &base).ok_or_else(|| format!("No directory {typed} (looked in {} and {})", base.display(), crate::config::home().display()))
    }

    // Where a typed path starts from: the focused session's directory, else piccosta's.
    fn base_dir(&self) -> std::path::PathBuf {
        if self.focus_cwd().as_os_str().is_empty() { self.host.cwd.clone() } else { self.focus_cwd().to_path_buf() }
    }

    // Set this session's level and save it as the model's, saying so when the model runs a
    // different one (the nearest it takes).
    fn set_thinking(&mut self, level: String) {
        let model = self.current().model.clone();
        let effective = self.host.registry().ok().and_then(|registry| registry.resolve(&model).ok().map(|found| crate::provider::clamp_thinking_level(found, &level)));
        self.notice = match crate::config::remember_thinking_level(&self.host.agent_dir, &model, Some(&level)) {
            Err(error) => Some(error),
            Ok(()) => effective.filter(|effective| *effective != level).map(|effective| format!("{model} doesn't take {level}; it runs at {effective}")),
        };
        self.tell(Command::Thinking(level));
    }

    // Every project's sessions, newest first, each with its directory. One picked opens in this
    // pane, in its own directory, and joins the sidebar.
    fn all_sessions(&mut self, filter: &str) {
        self.request_listing(super::listing::Kind::Resume(true), filter);
    }

    pub(super) fn all_sessions_ready(&mut self, mut found: Vec<crate::host::SessionSummary>, filter: &str) {
        if !filter.is_empty() {
            let ids: Vec<_> = found.iter().filter(|summary| summary.id.starts_with(filter)).collect();
            if let [summary] = ids.as_slice() {
                self.resume_path(summary.path.clone());
                return;
            }
            // An ambiguous id remains a choice of ids, not a coincidental title match.
            if !ids.is_empty() {
                found.retain(|summary| summary.id.starts_with(filter));
            }
        }
        // Grouped by project: projects in the order of their newest session, newest first in each.
        let mut projects: Vec<std::path::PathBuf> = Vec::new();
        for summary in &found {
            if !projects.contains(&summary.cwd) {
                projects.push(summary.cwd.clone());
            }
        }
        found.sort_by_key(|summary| projects.iter().position(|project| *project == summary.cwd));
        let items = found.iter().enumerate().map(|(place, summary)| {
            format!("{} · {} · {} · {} · {}", place + 1, super::render::home_relative(&summary.cwd), summary.title, summary.id, ago(&summary.path))
        });
        let items = items.collect();
        self.found = found.into_iter().map(|summary| summary.path.canonicalize().unwrap_or(summary.path)).collect();
        self.pick("every project's sessions", items, filter, |app, choice| {
            if let Some(path) = number(&choice).and_then(|place| app.found.get(place).cloned()) {
                app.resume_path(path);
            }
        });
    }

    fn resume_path(&mut self, path: std::path::PathBuf) {
        let path = path.canonicalize().unwrap_or(path);
        if self.slots.iter().any(|slot| slot.path.as_ref() == Some(&path) && slot.closing) { self.notice = Some("The session is still closing.".into()); return; }
        crate::session::set_closed(&self.host.agent_dir, &path, false);
        self.closed.remove(&path);
        let slot = self.slots.iter().position(|slot| slot.path.as_ref() == Some(&path)).unwrap_or_else(|| {
            self.slots.push(super::Slot::new(Some(path), super::NEW_TITLE.into()));
            self.slots.len() - 1
        });
        self.show(slot);
    }

    // Delete the session shown here, then show the one after it in the list (or before, at the end).
    pub(super) fn delete_shown(&mut self) {
        let slot = self.session_index();
        let listed = self.listed();
        let at = listed.iter().position(|&index| index == slot);
        // --single shows nothing else: close() leaves a fresh session in its pane.
        let next = at.filter(|_| !self.single).and_then(|at| listed.get(at + 1).or_else(|| at.checked_sub(1).and_then(|before| listed.get(before)))).copied();
        let failed = self.delete(slot);
        if let Some(next) = next.filter(|&next| !self.slots[next].gone() && self.panes[self.focus].target != super::Target::Session(next)) {
            self.show(next);
        }
        if let Some(error) = failed {
            self.report(error);
        }
    }

    // Close a session and move its file to the trash, so a mistake can be undone. Where it can't
    // go there it stays where it was. A running session is left alone. Output: why the trash
    // failed, which names where each file is, for the caller to report in the pane left showing
    // (too long for the status line).
    fn delete(&mut self, slot: usize) -> Option<String> {
        let (path, title) = (self.slots[slot].path.clone().filter(|path| path.exists()), self.slots[slot].title.clone());
        // A session another piccosta is running would lose its file from under it. One open here
        // answers on its socket too, so it's asked only when it isn't.
        if self.slots[slot].live.is_none() && path.as_deref().is_some_and(crate::runtime::running) {
            self.notice = Some(format!("“{title}” is open in another piccosta; quit it there first."));
            return None;
        }
        if !self.close(slot) {
            return None;
        }
        let Some(path) = path else {
            self.notice = Some(format!("Closed “{title}”: it was never saved."));
            return None;
        };
        self.notice = Some(format!("Moving “{title}” to the trash…"));
        // Closing the pane only queues runtime shutdown. The existing counted disk worker waits
        // for its lease, without blocking input or letting /restart strand a partial deletion.
        self.asks.send(Box::new(move |_| {
            let result = trash_session(&path);
            (slot, Update::Deleted(path, result))
        })).err().map(|_| "The disk worker stopped before the session could be trashed.".into())
    }

    // Put a command's output in the focused pane.
    pub(super) fn report(&mut self, text: String) {
        let pane = self.focus;
        self.panes[pane].scroll = 0;
        if matches!(self.panes[self.focus].target, super::Target::Scratchpad(_)) { self.notice = Some(text); }
        else { self.focused().messages.push(report_message(text)); }
    }

    fn status(&self) -> String {
        let slot = self.current();
        let file = slot.path.as_ref().map_or("not saved yet".to_string(), |path| path.display().to_string());
        let tex = if self.tex.images() { "typeset images" } else { "Unicode text" };
        // What idle compaction did last, or why not, while this piccosta runs the session.
        let idle = crate::runtime::idle_note(&slot.id).map_or(String::new(), |note| format!("\nidle compaction: {note}"));
        format!(
            "model      {}\nthinking   {}\nsession    {}\nfile       {file}\ncwd        {}\n{}\ncost       {}\nagent dir  {}\nTeX        {tex}{idle}",
            slot.model, slot.thinking, slot.title, slot.cwd.display(), self.context(), slot.cost.dollars(4), self.host.agent_dir.display()
        )
    }

    // The context the last response filled, against the model's window.
    fn context(&self) -> String {
        let slot = self.current();
        let share = if slot.context_window > 0 { slot.context_tokens as f64 / slot.context_window as f64 } else { 0.0 };
        let bar_cells = 40;
        let filled = ((share * bar_cells as f64).round() as usize).min(bar_cells);
        format!("context    {} of {} tokens ({:.1}%)\n           {}{}", slot.context_tokens, slot.context_window, share * 100.0, "█".repeat(filled), "░".repeat(bar_cells - filled))
    }

    // Every connected plan's usage, then the tokens and cost over every response this pane has shown.
    // The session file's whole count, as rpc's get_session_stats gives it (compactions and earlier
    // branches included); an unsaved session has only what this pane holds.
    fn usage(&self) -> String {
        let slot = self.current();
        let plans = Some(crate::scale::report(&self.host.agent_dir)).filter(|plans| !plans.is_empty()).map_or(String::new(), |plans| plans + "\n\n");
        let Some(session) = slot.path.as_deref().and_then(|path| Session::open(path, &slot.cwd).ok()) else {
            return format!("{plans}nothing saved yet; this pane's responses cost {}", slot.cost.dollars(4));
        };
        let stats = crate::rpc::session_stats(&session);
        let number = |key: &str| stats.get(key).and_then(Value::as_u64).unwrap_or(0);
        let tokens = |key: &str| {
            let known = stats.get("tokens").and_then(|tokens| tokens.get(key)).and_then(Value::as_u64).unwrap_or(0);
            match (known, stats.get("tokensUnknown") == Some(&Value::Bool(true))) {
                (0, true) => "unknown".into(),
                (_, true) => format!("{known} + unknown"),
                (_, false) => known.to_string(),
            }
        };
        format!(
            "{plans}responses   {}\ntool calls  {}\ninput       {}\noutput      {}\ncache read  {}\ncache write {}\ncost        {}",
            number("assistantMessages"), number("toolCalls"), tokens("input"), tokens("output"), tokens("cacheRead"), tokens("cacheWrite"),
            crate::provider::responses::dollars(stats.get("cost").and_then(Value::as_f64).unwrap_or(0.0), stats.contains_key("costUnknown"), 4)
        )
    }

    // The working tree's changes, as `git diff` shows them, then the untracked files.
    fn diff(&mut self) {
        let git = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(&self.current().cwd).args(args).output();
        // Against HEAD, so staged changes show too; a repo with no commit yet has only the index.
        let has_head = git(&["rev-parse", "--verify", "--quiet", "HEAD"]).is_ok_and(|out| out.status.success());
        let changes = match git(if has_head { &["--no-pager", "diff", "--no-color", "HEAD"] } else { &["--no-pager", "diff", "--no-color"] }) {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
            Ok(out) => {
                self.notice = Some(String::from_utf8_lossy(&out.stderr).lines().next().unwrap_or("git diff failed").to_string());
                return;
            }
            Err(error) => {
                self.notice = Some(format!("git: {error}"));
                return;
            }
        };
        let untracked = git(&["ls-files", "--others", "--exclude-standard"]).map(|out| String::from_utf8_lossy(&out.stdout).into_owned()).unwrap_or_default();
        // New files with their contents, as Codex's /diff shows them (diff --no-index exits 1 when
        // the files differ, which they always do here).
        let mut lines: Vec<String> = changes.lines().map(str::to_string).collect();
        for path in untracked.lines() {
            let added = git(&["--no-pager", "diff", "--no-color", "--no-index", "--", crate::platform::NULL, path]).map(|out| String::from_utf8_lossy(&out.stdout).into_owned()).unwrap_or_default();
            lines.extend(added.lines().map(str::to_string));
        }
        if lines.is_empty() {
            self.notice = Some("No changes.".into());
            return;
        }
        let more = lines.len().saturating_sub(DIFF_MAX_LINES);
        lines.truncate(DIFF_MAX_LINES);
        if more > 0 {
            lines.push(format!("… {more} more lines"));
        }
        self.report(lines.join("\n"));
    }

    // The nth latest response's text to the clipboard (1 is the last).
    fn copy(&mut self, argument: &str) {
        let nth = argument.parse::<usize>().unwrap_or(1).max(1);
        let text = self.current().messages.iter().rev().filter_map(|message| match message {
            AgentMessage::Assistant(assistant) => Some(text_of(&assistant.content)),
            _ => None,
        }).filter(|text| !text.is_empty()).nth(nth - 1);
        self.notice = Some(match text {
            Some(text) => format!("Copied {} characters ({}).", text.chars().count(), super::clip::copy(&text)),
            None => "No response to copy.".into(),
        });
    }

    // The conversation as Markdown: to the file named, else next to the work, named after the session.
    fn export(&mut self, argument: &str) {
        let slot = self.current();
        let name = slot.path.as_ref().and_then(|path| path.file_stem()).map_or("piccosta-session".to_string(), |stem| stem.to_string_lossy().into_owned());
        let path = if argument.is_empty() { slot.cwd.join(format!("{name}.md")) } else { slot.cwd.join(argument) };
        // The default name is never written over; a name given is what the user asked for.
        if argument.is_empty() && path.exists() {
            self.notice = Some(format!("{} exists; /export <file> to write somewhere else", path.display()));
            return;
        }
        let text = markdown(&slot.messages);
        self.notice = Some(match std::fs::write(&path, text) {
            Ok(()) => format!("Wrote {}", path.display()),
            Err(error) => format!("Could not write {}: {error}", path.display()),
        });
    }

    // A copy of this session up to `leaf` (its end when None), opened in this pane, named `name`
    // when one is given, with `draft` in its editor.
    fn fork(&mut self, leaf: Option<Option<String>>, name: String, draft: String) {
        let Some(path) = self.current().path.clone() else {
            self.notice = Some("Nothing to copy yet.".into());
            return;
        };
        let rewinding = leaf.is_some();
        let forked = Session::open(&path, &self.current().cwd).map_err(|error| error.to_string()).and_then(|source| {
            let leaf = leaf.unwrap_or_else(|| source.leaf_id().map(str::to_string));
            source.branch(leaf.as_deref(), &self.host.session_dir).map_err(|error| error.to_string())
        });
        // A copy with no conversation in it was never written: a new session is the same thing.
        let opened = match forked {
            Ok(session) => match session.file().map(Path::to_path_buf) {
                Some(file) => self.open(Some(file), None, None).map(Some),
                None => Ok(self.new_session()),
            },
            Err(error) => Err(error),
        };
        match opened {
            Ok(Some(slot)) => {
                let printed = self.printed.1;
                self.show(slot);
                // --inline: what the copy shares with this session is in the scrollback already, and
                // a rewind says where it went back to before what follows (inline.rs).
                if self.inline {
                    let messages = &self.slots[slot].messages;
                    let prompts = messages.iter().filter(|message| matches!(message, AgentMessage::User(_))).count();
                    // A fork keeps what was printed of a reply it shares; a rewind starts over.
                    (self.printed, self.streamed) = ((slot, printed.min(messages.len())), if rewinding { Default::default() } else { self.streamed });
                    self.rewound = rewinding.then(|| format!("rewound to before prompt {}", prompts + 1));
                }
                if !name.is_empty() {
                    self.focused().title = name.clone();
                    self.tell(Command::Name(name));
                }
                self.focused().draft.insert(&draft);
            }
            Ok(None) => {}
            Err(error) => self.notice = Some(error),
        }
    }

    // Choose one of this session's prompts; the copy ends just before it.
    fn rewind(&mut self) {
        let Some(path) = self.current().path.clone() else {
            self.notice = Some("Nothing to rewind yet.".into());
            return;
        };
        let session = match Session::open(&path, &self.current().cwd) {
            Ok(session) => session,
            Err(error) => {
                self.notice = Some(error.to_string());
                return;
            }
        };
        self.rewind = session
            .path_to_leaf()
            .into_iter()
            .filter_map(|entry| match entry {
                SessionEntry::Message { base, message: AgentMessage::User(user) } => Some((base.parent_id.clone(), user.content.text())),
                _ => None,
            })
            .collect();
        let items = self.rewind.iter().enumerate().rev().map(|(index, (_, text))| format!("{} · {}", index + 1, super::render::fit(text.lines().next().unwrap_or(""), ITEM_CHARS))).collect();
        self.pick("rewind to before", items, "", |app, choice| {
            if let Some((parent, text)) = number(&choice).and_then(|index| app.rewind.get(index).cloned()) {
                app.fork(Some(parent), String::new(), text);
            }
        });
    }
}

// The number a list entry starts with ("3 · title"), as an index.
// How deep under home to look for git repositories (~/projects/x/y is depth 3), and folders that
// never hold one worth starting a session in.
const REPOSITORY_DEPTH: usize = 3;
const NOT_PROJECTS: [&str; 6] = ["node_modules", "target", "build", "venv", "__pycache__", "dist"];

// A directory typed for a new session: ~ is home; else it's looked for from `base` (the focused
// session's directory), then from home. Output: the directory, links followed, when it's there.
pub(super) fn resolve_directory(typed: &str, base: &Path) -> Option<std::path::PathBuf> {
    let home = crate::config::home();
    // A trailing separator goes, but a root stays one: / here, D:/ or D:\ on Windows.
    let trimmed = typed.trim_end_matches(['/', std::path::MAIN_SEPARATOR]);
    let typed = if trimmed.is_empty() || trimmed.ends_with(':') { typed } else { trimmed };
    let expanded = crate::tools::paths::expand_home(typed);
    [base.join(&expanded), home.join(&expanded)].into_iter().find_map(|dir| dir.canonicalize().ok().filter(|dir| dir.is_dir()))
}

// The git repositories under `dir`, down to REPOSITORY_DEPTH, by name; hidden folders skipped, and
// a repository's own folders not searched. Only folder names are read.
fn git_repositories(dir: &Path) -> Vec<std::path::PathBuf> {
    fn walk(dir: &Path, depth: usize, found: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let entries: Vec<std::fs::DirEntry> = entries.flatten().collect();
        if entries.iter().any(|entry| entry.file_name() == ".git") {
            found.push(dir.to_path_buf());
            return;
        }
        if depth == 0 {
            return;
        }
        for entry in entries {
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) && !name.starts_with('.') && !NOT_PROJECTS.contains(&name.as_str()) {
                walk(&entry.path(), depth - 1, found);
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, REPOSITORY_DEPTH, &mut found);
    found.sort();
    found
}

// A line split into words at spaces, with a "quoted part" kept as one word (a --name with spaces).
fn words(line: &str) -> Vec<String> {
    let mut words = vec![String::new()];
    let mut quoted = false;
    for character in line.chars() {
        match character {
            '"' => quoted = !quoted,
            ' ' if !quoted => words.push(String::new()),
            _ => words.last_mut().into_iter().for_each(|word| word.push(character)),
        }
    }
    words.retain(|word| !word.is_empty());
    words
}

// A report, as the pane keeps it: shown, never sent to the model.
pub(super) fn report_message(text: String) -> AgentMessage {
    AgentMessage::Custom(CustomMessage { custom_type: REPORT.into(), origin: crate::message::InputOrigin::Harness { name: REPORT.into() }, content: TextOrBlocks::Text(text), display: true, details: None, timestamp: crate::clock::now_ms() })
}

// The command names that start with what's typed, in /help's order (for tab).
// The command names that start with what's typed; /descent last, once it's been revealed.
pub(super) fn completions(typed: &str, descent: bool) -> Vec<&'static str> {
    let names = COMMANDS.iter().flat_map(|(names, _)| names.split(", ")).filter_map(|name| name.split(' ').next());
    let mut found: Vec<&'static str> = names.filter(|name| name.starts_with('/') && name.starts_with(typed)).collect();
    if descent && DESCENT_NAME.starts_with(typed) {
        found.push(DESCENT_NAME);
    }
    found
}

// The easter egg's name, and how long /d has to sit typed before the list shows it: long enough
// that nobody passing through to /diff or /delete sees it.
const DESCENT_NAME: &str = "/descent";
pub(super) const DESCENT_REVEAL: std::time::Duration = std::time::Duration::from_secs(12);

// Whether a draft is on the way to /descent: /d at least, and nothing else after.
pub(super) fn toward_descent(draft: &str) -> bool {
    draft.len() >= 2 && DESCENT_NAME.starts_with(draft)
}

// How long ago a file last changed: "3m ago".
fn ago(path: &Path) -> String {
    let modified = path.metadata().and_then(|metadata| metadata.modified()).ok();
    let elapsed = modified.and_then(|modified| modified.elapsed().ok()).unwrap_or_default();
    format!("{} ago", super::render::duration(elapsed))
}

fn number(choice: &str) -> Option<usize> {
    choice.split(' ').next()?.parse::<usize>().ok()?.checked_sub(1)
}

fn text_of(content: &[Content]) -> String {
    content.iter().filter_map(|block| if let Content::Text { text, .. } = block { Some(text.as_str()) } else { None }).collect::<Vec<_>>().join("")
}

// The conversation as Markdown: prompts, responses, each tool call and what it returned.
fn markdown(messages: &[AgentMessage]) -> String {
    let mut out = String::new();
    for message in messages {
        match message {
            AgentMessage::User(user) => out.push_str(&format!("## You\n\n{}\n\n", user.content.text())),
            AgentMessage::Assistant(assistant) => {
                out.push_str("## Assistant\n\n");
                for block in &assistant.content {
                    match block {
                        Content::Text { text, .. } => out.push_str(&format!("{text}\n\n")),
                        Content::ToolCall { name, arguments, .. } => out.push_str(&format!("`{name}` {}\n\n", Value::Object(arguments.clone()))),
                        _ => {}
                    }
                }
            }
            AgentMessage::ToolResult(result) => out.push_str(&format!("```\n{}\n```\n\n", TextOrBlocks::Blocks(result.content.clone()).text())),
            _ => {}
        }
    }
    out
}

// A session's file to the trash, with the Claude bridge's record of it beside it; a socket a
// piccosta that died left there goes. The session's own file goes last, so a piccosta stopped
// partway leaves the session listed where it was. When one can't go, the ones already moved come
// back; nothing is ever deleted outright. Output: Ok when they all went, else what happened, naming
// any file that couldn't come back and where it is.
pub(crate) fn trash_session(path: &Path) -> Result<(), String> {
    let target = crate::tools::paths::canonical(path);
    let _ownership = crate::runtime::deletion_guard(path)?;
    crate::ipc::remove(path);
    let [sidecar, journal_lock] = crate::provider::claude::sidecars(path);
    let beside = [sidecar, journal_lock, std::path::PathBuf::from(format!("{}{}", path.display(), crate::runtime::OWNERSHIP_SUFFIX))];
    let mut moved: Vec<(&Path, (std::path::PathBuf, std::path::PathBuf))> = Vec::new();
    for file in beside.iter().map(|file| file.as_path()).filter(|file| file.exists()).chain(std::iter::once(path)) {
        let Some(went) = trash(file) else {
            let stuck: Vec<String> = moved
                .into_iter()
                .filter_map(|(file, (into, info))| match std::fs::rename(&into, file) {
                    Ok(()) => std::fs::remove_file(&info).err().map(|_| format!("{} has a stale trash entry {}", file.display(), info.display())),
                    Err(error) => Some(format!("{} is in the trash as {} ({error})", file.display(), into.display())),
                })
                .collect();
            let left = if stuck.is_empty() { String::new() } else { format!(", but {}", stuck.join(", and ")) };
            return Err(format!("Could not move {} to the trash; it's left where it was{left}.", path.display()));
        };
        moved.push((file, went));
    }
    // Only a completed deletion loses its schedules. A failed move leaves them available in the
    // session that stayed put; a deleted session has nowhere left for the user to run /timer off.
    crate::tools::checkin::cancel_all(&crate::config::agent_dir(), &target, true)
        .map_err(|error| format!("Moved {} to the trash, but could not cancel its timers: {error}", path.display()))
}

// Move a file to the user's trash the way the freedesktop spec has it ($XDG_DATA_HOME/Trash), so a
// file manager shows it and can put it back: the file goes into files/, and info/ gets a
// <name>.trashinfo saying where it came from and when. The info file is claimed first with
// create_new, as the spec asks, so two deletes never take one name; a name already taken gets .2,
// .3 and so on. Output: where the file and its info file went, or None when it's still where it
// was. It is when the rename would cross file systems (the spec's per-volume trash folders aren't
// done) or the trash can't be written.
fn trash(file: &Path) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let trash = crate::config::data_home().join("Trash");
    let (Ok(path), Some(name)) = (std::path::absolute(file), file.file_name()) else { return None };
    let url = url::Url::from_file_path(&path).ok()?;
    // The spec has the trash readable by its owner only.
    if ["files", "info"].iter().any(|folder| crate::platform::private_dir(&trash.join(folder)).is_err()) {
        return None;
    }
    // Path is percent-encoded as in a URL; the date is local time, to the second.
    let info_text = format!("[Trash Info]\nPath={}\nDeletionDate={}\n", url.path(), &crate::clock::local_now().0[..19]);
    for number in 1.. {
        let mut slot = name.to_os_string();
        if number > 1 {
            slot.push(format!(".{number}"));
        }
        let into = trash.join("files").join(&slot);
        slot.push(".trashinfo");
        let info = trash.join("info").join(slot);
        if into.exists() {
            continue;
        }
        let mut claim = match std::fs::OpenOptions::new().write(true).create_new(true).open(&info) {
            Ok(claim) => claim,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return None,
        };
        if claim.write_all(info_text.as_bytes()).is_ok() && std::fs::rename(&path, &into).is_ok() {
            return Some((into, info));
        }
        // An info file with no file behind it would show in the trash as a ghost.
        let _ = std::fs::remove_file(&info);
        return None;
    }
    None
}
