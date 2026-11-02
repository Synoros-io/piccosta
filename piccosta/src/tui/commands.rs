// The TUI's slash commands. Names follow Claude Code's and Codex's wherever piccosta does the same
// thing, so the habits carry over.
// Their product commands (MCP, plugins, IDE, cloud, mobile, login flows) have nothing to act on
// here. A command's output goes into the pane as a report: shown there, never sent to the model,
// gone when the session is reopened.

use super::{App, Command, State};
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
const COMMANDS: [(&str, &str); 39] = [
    ("/new [dir]", "start a new session in a directory you name, pick or type (^n too; enter: right here)"),
    ("/restart", "start piccosta again on its binary as it is now (after a build), everything as it was and background jobs carrying on; waits for running turns (esc cancels)"),
    ("/clear [name]", "start a new session here; a name names the one left"),
    ("/scale", "which model does which kind of work, and how much of each plan is used (--use <kind> picks from it)"),
    ("/manager", "this project's manager: shown, or made (the first session in a directory is one)"),
    ("/resume [text|all]", "the sessions to switch to (^s; ^d there deletes one); all: every project's"),
    ("/fork, /branch [name]", "copy this session and carry on in the copy"),
    ("/rewind", "go back to before one of your prompts, in a copy, with that prompt in the editor"),
    ("/rename, /name <name>", "name this session"),
    ("/model [name]", "switch the model; new sessions start on it too"),
    ("/pair [model|off]", "work with a second model as a pair: the two answer each other in turn until one says it's done, then you speak"),
    ("/trio [model|off]", "work with a second model as a trio: it checks each answer once, then you speak"),
    ("/thinking, /effort [level|auto]", "set the thinking level, saved for the model; auto forgets it"),
    ("/compact [focus]", "summarize the conversation to free context"),
    ("/autocompact [on|off]", "compact on its own when the context fills; alone, says which"),
    ("/status", "model, session, context and cost"),
    ("/context", "how full the context is"),
    ("/usage, /cost", "tokens and cost of this session"),
    ("/diff", "git diff against HEAD, new files included"),
    ("/pwd", "the working directory"),
    ("/copy [n]", "copy the last (or nth last) response"),
    ("/export [file]", "write the conversation as Markdown"),
    ("/init", "have the model write AGENTS.md, direction.md and ledger.md"),
    ("/review [what]", "hand a review of the changes to the review job"),
    ("/agents", "the subagent jobs and their models"),
    ("/ps", "background jobs"),
    ("/stop", "kill the background jobs"),
    ("/split", "open a new session in another pane"),
    ("/close", "close this session: it's done and leaves the list (/resume finds it, and a message opens it again)"),
    ("/delete", "move this session to the trash and show the next; the Delete key twice, with nothing typed, does it too"),
    ("/help", "this list"),
    ("/quit, /exit", "leave piccosta"),
    ("keys", "enter send · alt+enter newline · esc stop · ^c quit"),
    ("", "^backspace deletes a word · ^shift+delete clears what's typed (↑ brings it back)"),
    ("", "running: enter queues · ^x^s or ^enter sends now · ↑ takes back · esc stops, queue goes next"),
    ("", "^n new · ^s sessions · alt+↑↓ through the sidebar (shown, started when you type) · ^b sidebar"),
    ("", "^\\ split · ^] close pane · alt+←→ panes · ^o tools · ^t thinking"),
    ("copy", "drag to select (double-click a word, triple-click a row); it copies when you let go · ^c copies a selection · shift+arrows extend it"),
    ("paste", "right-click pastes the clipboard, middle-click the last selection, ^v the clipboard · shift+arrows select in the editor"),
];


impl App {
    // A slash command, without its slash. Output: false when it isn't a command, so it goes to the
    // model as text (a path such as /etc/hosts); a mistyped command word is answered here.
    pub(super) fn command(&mut self, line: &str) -> bool {
        let (name, argument) = line.split_once(' ').map_or((line, ""), |(name, rest)| (name, rest.trim()));
        match name {
            // /clear <name> names the session being left, as Claude Code's does.
            // /clear <name> names the session being left, as Claude Code's does.
            "clear" => {
                if !argument.is_empty() {
                    self.focused().title = argument.to_string();
                    self.tell(Command::Name(argument.to_string()));
                }
                if let Some(slot) = self.new_session() {
                    self.show(slot);
                }
            }
            // /new <dir> starts the session in that directory; /new alone offers this one and the
            // projects used lately (enter on the first is the same as ^n).
            // /new <dir> [flags]: the flags are the command line's, those that shape one session.
            "new" if !argument.is_empty() => {
                let mut args = crate::cli::parse(&words(argument));
                match self.host.launch(&args) {
                    Ok(launch) => (args.model, args.thinking) = launch,
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
                    None => Ok(self.current().cwd.clone()),
                };
                match dir {
                    Ok(dir) => {
                        // A directory alone is a plain new session there (an empty one is reused).
                        let flags = (words(argument).len() > args.messages.len()).then_some(args);
                        if let Some(slot) = self.new_session_in(Some(dir), flags) {
                            self.show(slot);
                        }
                    }
                    Err(error) => self.notice = Some(error),
                }
            }
            "new" => {
                // Here first, then home, for a session outside every project.
                let mut dirs = vec![super::render::home_relative(&self.current().cwd)];
                if !dirs.iter().any(|dir| dir == "~") {
                    dirs.push("~".into());
                }
                let folders = std::fs::read_dir(self.host.agent_dir.join("sessions")).into_iter().flatten().flatten();
                let mut recent: Vec<crate::host::SessionSummary> = folders.flat_map(|folder| crate::host::sessions_in(&folder.path(), None, 1)).filter(|summary| summary.cwd.is_dir()).collect();
                recent.sort_by_key(|summary| std::cmp::Reverse(summary.modified));
                // Then the projects worked in lately, then every git repository near home.
                let repos = git_repositories(&std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()));
                // piccosta's own folders (the assistant's home) aren't projects.
                for dir in recent.into_iter().map(|summary| summary.cwd).filter(|dir| !dir.starts_with(&self.host.agent_dir)).chain(repos) {
                    let dir = super::render::home_relative(&dir);
                    if !dirs.contains(&dir) {
                        dirs.push(dir);
                    }
                }
                self.pick("new session in · enter: here · ~ is home · or type any path, then flags: --model --effort --pair --name …", dirs, "", |app, typed| {
                    app.command(&format!("new {typed}"));
                });
                let base = self.base_dir();
                if let Some(picker) = &mut self.picker {
                    picker.free = Some(base);
                }
            }
            "resume" => self.sessions(argument),
            // The scale is one file: /scale edit opens it (written out whole the first time), and
            // /scale <use> <model[:level]> ... sets one use's models there.
            "scale" if argument == "edit" => match crate::scale::write_out(&self.host.agent_dir) {
                Ok(path) => self.edit = Some(path),
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
            // This directory's project manager: shown when there is one, else made, as a new session.
            "manager" => {
                // From the assistant's pane, the project piccosta was started in.
                let cwd = self.current().cwd.clone();
                let cwd = if cwd.as_os_str().is_empty() || cwd == crate::manager::home(&self.host.agent_dir) { self.host.cwd.clone() } else { cwd };
                let existing = crate::manager::project_manager(&self.host.agent_dir, &cwd).and_then(|file| file.canonicalize().ok());
                let slot = match existing {
                    Some(file) => self.slots.iter().position(|slot| slot.path.as_ref() == Some(&file)).unwrap_or_else(|| {
                        self.slots.push(super::Slot { cwd: cwd.clone(), manager: true, ..super::Slot::new(Some(file), crate::manager::PROJECT_NAME.into()) });
                        self.slots.len() - 1
                    }),
                    None => {
                        self.slots.push(super::Slot { cwd, make_manager: true, ..super::Slot::new(None, super::NEW_TITLE.into()) });
                        let slot = self.slots.len() - 1;
                        if let Err(error) = self.load(slot, None, None, None) {
                            self.slots.pop();
                            self.notice = Some(error);
                            return true;
                        }
                        slot
                    }
                };
                self.show(slot);
            }
            "fork" | "branch" => self.fork(None, argument.to_string(), String::new()),
            "rewind" => self.rewind(),
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
            "autocompact" if argument == "on" || argument == "off" => {
                self.focused().auto_compaction = argument == "on";
                self.tell(Command::AutoCompaction(argument == "on"));
                self.notice = Some(format!("auto-compaction {argument}"));
            }
            "autocompact" => self.notice = Some(format!("auto-compaction is {} (/autocompact on|off)", if self.current().auto_compaction { "on" } else { "off" })),
            "status" => self.report(self.status()),
            "context" => self.report(self.context()),
            "usage" | "cost" => self.report(self.usage()),
            "diff" => self.diff(),
            "pwd" | "cwd" => self.notice = Some(self.current().cwd.display().to_string()),
            "copy" => self.copy(argument),
            "export" => self.export(argument),
            // The session is titled by the command, not the long prompt it stands for.
            "init" | "review" => {
                if self.current().title == super::NEW_TITLE {
                    self.focused().title = format!("/{line}");
                }
                self.send(crate::prompt::command_prompt(&format!("/{line}")).unwrap_or_default());
            }
            "agents" => match crate::tools::subagent::SubagentTool::load(&self.host.agent_dir) {
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
            // Closing says the session is done: it leaves the list for good, and a message in it
            // (from /resume, or a manager) opens it again.
            "close" => {
                let slot = self.panes[self.focus].slot;
                if self.close(slot)
                    && let Some(path) = self.slots[slot].path.take()
                {
                    crate::session::set_closed(&self.host.agent_dir, &path, true);
                }
            }
            "delete" => self.delete_shown(),
            "quit" | "exit" => self.quit = true,
            // Start again on the binary as it is now (after a build), with everything as it was and
            // the background jobs carrying on. A turn can't go across, so this waits for the running
            // ones to end (the main loop restarts then); asked again while waiting, it doesn't.
            "restart" if self.restart_waiting => {
                self.restart_waiting = false;
                self.notice = Some("Not restarting.".into());
            }
            "restart" => {
                self.restart_waiting = true;
                let running = self.slots.iter().filter(|slot| slot.busy()).count();
                if running > 0 {
                    self.notice = Some(format!("Restarting when {running} running turn{} end{} (esc or /restart: don't).", if running == 1 { "" } else { "s" }, if running == 1 { "s" } else { "" }));
                }
            }
            "descent" => self.report(DESCENT.into()),
            "help" => self.report(COMMANDS.iter().map(|(name, what)| format!("{name:<28} {what}")).collect::<Vec<_>>().join("\n")),
            _ if !name.is_empty() && name.chars().all(|character| character.is_ascii_alphanumeric() || character == '-') => {
                self.notice = Some(format!("No command /{name}; /help lists them."));
            }
            _ => return false,
        }
        true
    }

    // The sessions, open ones first, numbered as the sidebar lists them, with the one shown here
    // chosen and how long ago each last changed. The list is read from disk again each time, so a
    // session another terminal started or renamed shows. Enter shows one; ctrl+d then enter deletes
    // one and the list comes back without it. "all" lists every project's sessions instead.
    pub(super) fn sessions(&mut self, filter: &str) {
        if filter == "all" {
            return self.all_sessions();
        }
        let known: std::collections::HashSet<std::path::PathBuf> = self.slots.iter().filter_map(|slot| slot.path.clone()).collect();
        for summary in self.host.sessions(super::PICKER_SESSIONS) {
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
        let current = self.panes[self.focus].slot;
        if let Some(picker) = &mut self.picker {
            picker.selected = if filter.is_empty() { listed.iter().position(|&index| index == current).unwrap_or(0) } else { 0 };
            picker.delete = Some(|app, choice| {
                if let Some(slot) = number(&choice).and_then(|place| app.listed().get(place).copied()) {
                    app.delete(slot);
                }
                app.sessions("");
            });
        }
    }

    // A directory typed or picked: ~ is home; a relative one is looked for from the focused
    // session's directory, then from home (so "projects/x" finds ~/projects/x).
    fn directory(&self, typed: &str) -> Result<std::path::PathBuf, String> {
        let base = self.base_dir();
        resolve_directory(typed, &base).ok_or_else(|| format!("No directory {typed} (looked in {} and {})", base.display(), std::env::var("HOME").unwrap_or_default()))
    }

    // Where a typed path starts from: the focused session's directory, else piccosta's.
    fn base_dir(&self) -> std::path::PathBuf {
        if self.current().cwd.as_os_str().is_empty() { self.host.cwd.clone() } else { self.current().cwd.clone() }
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
    fn all_sessions(&mut self) {
        let folders = std::fs::read_dir(self.host.agent_dir.join("sessions")).into_iter().flatten().flatten();
        let mut found: Vec<crate::host::SessionSummary> = folders.flat_map(|folder| crate::host::sessions_in(&folder.path(), None, super::PICKER_SESSIONS)).collect();
        found.sort_by_key(|summary| std::cmp::Reverse(summary.modified));
        // Grouped by project: projects in the order of their newest session, newest first in each.
        let mut projects: Vec<std::path::PathBuf> = Vec::new();
        for summary in &found {
            if !projects.contains(&summary.cwd) {
                projects.push(summary.cwd.clone());
            }
        }
        found.sort_by_key(|summary| projects.iter().position(|project| *project == summary.cwd));
        let items = found.iter().enumerate().map(|(place, summary)| {
            format!("{} · {} · {} · {}", place + 1, super::render::home_relative(&summary.cwd), super::render::fit(&summary.title, ITEM_CHARS), ago(&summary.path))
        });
        let items = items.collect();
        self.found = found.into_iter().map(|summary| summary.path.canonicalize().unwrap_or(summary.path)).collect();
        self.pick("every project's sessions", items, "", |app, choice| {
            let Some(path) = number(&choice).and_then(|place| app.found.get(place).cloned()) else { return };
            let slot = app.slots.iter().position(|slot| slot.path.as_ref() == Some(&path)).unwrap_or_else(|| {
                app.slots.push(super::Slot::new(Some(path), super::NEW_TITLE.into()));
                app.slots.len() - 1
            });
            app.show(slot);
        });
    }

    // Delete the session shown here, then show the one after it in the list (or before, at the end).
    pub(super) fn delete_shown(&mut self) {
        let slot = self.panes[self.focus].slot;
        let listed = self.listed();
        let at = listed.iter().position(|&index| index == slot);
        let next = at.and_then(|at| listed.get(at + 1).or_else(|| at.checked_sub(1).and_then(|before| listed.get(before)))).copied();
        self.delete(slot);
        if let Some(next) = next.filter(|&next| !self.slots[next].gone() && self.panes[self.focus].slot != next) {
            self.show(next);
        }
    }

    // Close a session and move its file to the trash, so a mistake can be undone, or remove it
    // where it can't go there. A running session is left alone.
    fn delete(&mut self, slot: usize) {
        let (path, title) = (self.slots[slot].path.clone().filter(|path| path.exists()), self.slots[slot].title.clone());
        // A session another piccosta is running would lose its file from under it. One open here
        // is held by this piccosta's own lock, which reads as held from a second look.
        if self.slots[slot].live.is_none() && path.as_deref().is_some_and(crate::session::held_elsewhere) {
            self.notice = Some(format!("“{title}” is open in another piccosta; quit it there first."));
            return;
        }
        if !self.close(slot) {
            return;
        }
        self.slots[slot].path = None;
        let Some(path) = path else {
            self.notice = Some(format!("Closed “{title}”: it was never saved."));
            return;
        };
        self.notice = Some(match trash_session(&path) {
            Some(true) => format!("Moved “{title}” to the trash."),
            Some(false) => format!("Deleted “{title}”."),
            None => format!("Could not delete {}.", path.display()),
        });
    }

    // Put a command's output in the focused pane.
    pub(super) fn report(&mut self, text: String) {
        let pane = self.focus;
        self.panes[pane].scroll = 0;
        self.focused().messages.push(report_message(text));
    }

    fn status(&self) -> String {
        let slot = self.current();
        let file = slot.path.as_ref().map_or("not saved yet".to_string(), |path| path.display().to_string());
        let tex = if self.tex.images() { "typeset images" } else { "Unicode text" };
        format!(
            "model      {}\nthinking   {}\nsession    {}\nfile       {file}\ncwd        {}\n{}\ncost       ${:.4}\nagent dir  {}\nTeX        {tex}",
            slot.model, slot.thinking, slot.title, slot.cwd.display(), self.context(), slot.cost, self.host.agent_dir.display()
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

    // Tokens and cost over every response this pane has shown.
    // The session file's whole count, as rpc's get_session_stats gives it (compactions and earlier
    // branches included); an unsaved session has only what this pane holds.
    fn usage(&self) -> String {
        let slot = self.current();
        let Some(session) = slot.path.as_deref().and_then(|path| Session::open(path, &slot.cwd).ok()) else {
            return format!("nothing saved yet; this pane's responses cost ${:.4}", slot.cost);
        };
        let stats = crate::rpc::session_stats(&session);
        let number = |key: &str| stats.get(key).and_then(Value::as_u64).unwrap_or(0);
        let tokens = |key: &str| stats.get("tokens").and_then(|tokens| tokens.get(key)).and_then(Value::as_u64).unwrap_or(0);
        format!(
            "responses   {}\ntool calls  {}\ninput       {}\noutput      {}\ncache read  {}\ncache write {}\ncost        ${:.4}",
            number("assistantMessages"), number("toolCalls"), tokens("input"), tokens("output"), tokens("cacheRead"), tokens("cacheWrite"),
            stats.get("cost").and_then(Value::as_f64).unwrap_or(0.0)
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
            let added = git(&["--no-pager", "diff", "--no-color", "--no-index", "--", "/dev/null", path]).map(|out| String::from_utf8_lossy(&out.stdout).into_owned()).unwrap_or_default();
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
                self.show(slot);
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
    let home = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default());
    // A trailing slash goes, but / stays the root.
    let trimmed = typed.trim_end_matches('/');
    let typed = if trimmed.is_empty() { typed } else { trimmed };
    let expanded = match typed.strip_prefix('~') {
        Some(rest) => home.join(rest.trim_start_matches('/')),
        None => std::path::PathBuf::from(typed),
    };
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
    AgentMessage::Custom(CustomMessage { custom_type: REPORT.into(), content: TextOrBlocks::Text(text), display: true, details: None, timestamp: crate::clock::now_ms() })
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

// Move a file to the user's trash the way the freedesktop spec has it ($XDG_DATA_HOME/Trash), so a
// file manager shows it and can put it back: the file goes into files/, and info/ gets a
// <name>.trashinfo saying where it came from and when. The info file is claimed first with
// create_new, as the spec asks, so two deletes never take one name; a name already taken gets .2,
// .3 and so on. Output: whether the file is in the trash. It isn't when the rename would cross file
// systems (the spec's per-volume trash folders aren't done), and the caller deletes it instead.
// A session's file to the trash, with the files beside it (the Claude bridge's record of it, its
// inbox), each deleted instead where it can't go there. Output: Some(true) when the session's file
// is in the trash, Some(false) when it was deleted, None when it's still there.
pub(crate) fn trash_session(path: &Path) -> Option<bool> {
    for suffix in [crate::provider::claude::SIDECAR_SUFFIX, ".inbox"] {
        let beside = std::path::PathBuf::from(format!("{}{suffix}", path.display()));
        if beside.exists() && !trash(&beside) {
            let _ = std::fs::remove_file(&beside);
        }
    }
    let trashed = trash(path);
    (trashed || std::fs::remove_file(path).is_ok()).then_some(trashed)
}

fn trash(file: &Path) -> bool {
    use std::os::unix::fs::DirBuilderExt;
    let trash = crate::config::data_home().join("Trash");
    let (Ok(path), Some(name)) = (std::path::absolute(file), file.file_name()) else { return false };
    let Ok(url) = url::Url::from_file_path(&path) else { return false };
    // The spec has the trash readable by its owner only.
    if ["files", "info"].iter().any(|folder| std::fs::DirBuilder::new().recursive(true).mode(0o700).create(trash.join(folder)).is_err()) {
        return false;
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
            Err(_) => return false,
        };
        if claim.write_all(info_text.as_bytes()).is_ok() && std::fs::rename(&path, &into).is_ok() {
            return true;
        }
        // An info file with no file behind it would show in the trash as a ghost.
        let _ = std::fs::remove_file(&info);
        return false;
    }
    false
}
