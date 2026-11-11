// Disk work for the TUI's one asker thread. A followed tree never crosses to the UI; only
// prepared transcript changes do. Slot tickets expire on replacement, not on focus or typing.
use super::{render, Slot};
use crate::{host::Host, message::AgentMessage, session::{Follower, Session}};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::SystemTime;

#[derive(Default)]
pub(super) struct Transcript {
    pub messages: Vec<AgentMessage>,
    pub tools: HashMap<String, render::ToolView>,
    pub call_owner: HashMap<String, usize>,
    pub cache: HashMap<usize, (u16, render::Flags, render::Rows)>,
    pub context_tokens: u64,
    pub cost: crate::provider::responses::Tally,
}

impl Transcript {
    pub fn touch(&mut self, call_id: &str) {
        if let Some(index) = self.call_owner.get(call_id) { self.cache.remove(index); }
    }

    pub fn remember(&mut self, message: AgentMessage) {
        match &message {
            AgentMessage::Assistant(reply) => {
                for block in &reply.content {
                    if let crate::message::Content::ToolCall { id, .. } = block { self.call_owner.insert(id.clone(), self.messages.len()); }
                }
                if let Some(before) = self.messages.iter().rposition(|message| matches!(message, AgentMessage::Assistant(_))) { self.cache.remove(&before); }
                let usage = &reply.usage;
                let used = crate::provider::estimate::context_tokens(usage);
                if used > 0 { self.context_tokens = used; }
                self.cost.add(usage, &reply.provider);
            }
            AgentMessage::ToolResult(result) => {
                self.tools.insert(result.tool_call_id.clone(), render::ToolView {
                    text: crate::message::TextOrBlocks::Blocks(result.content.clone()).text(),
                    diff: result.details.as_ref().and_then(|details| details.get("diff")).and_then(Value::as_str).map(str::to_string),
                    is_error: result.is_error, running: false,
                });
                self.touch(&result.tool_call_id);
            }
            _ => {}
        }
        self.messages.push(message);
    }

    fn from_messages(messages: impl IntoIterator<Item = AgentMessage>) -> Self {
        let mut transcript = Self::default();
        for message in messages { transcript.remember(message); }
        transcript
    }

    pub fn append(&mut self, mut delta: Self) {
        if delta.messages.iter().any(|message| matches!(message, AgentMessage::Assistant(_)))
            && let Some(before) = self.messages.iter().rposition(|message| matches!(message, AgentMessage::Assistant(_))) { self.cache.remove(&before); }
        for call in delta.tools.keys() { self.touch(call); }
        let offset = self.messages.len();
        self.call_owner.extend(delta.call_owner.into_iter().map(|(id, index)| (id, index + offset)));
        self.tools.extend(delta.tools);
        self.messages.append(&mut delta.messages);
        (self.context_tokens, self.cost) = (delta.context_tokens, delta.cost);
    }
}

pub(super) struct Followed {
    title: String,
    id: String,
    cwd: PathBuf,
    model: String,
    thinking: String,
    context_window: u64,
    fast: bool,
    role: Option<crate::manager::Role>,
    team: crate::host::Team,
    pub transcript: Transcript,
    pub replace: bool,
}

impl Followed {
    fn metadata(host: &Host, session: &Session, transcript: Transcript, replace: bool) -> Self {
        let (model, thinking) = session.state();
        let model = model.map_or(String::new(), |(provider, id)| format!("{provider}/{id}"));
        let found = host.registry().ok().and_then(|registry| registry.resolve(&model).ok().map(|found| (found.context_window, crate::agent::service_tier(session, found))));
        Self {
            title: crate::manager::title(&host.agent_dir, session), id: session.id().to_string(), cwd: session.header().cwd.clone().into(), model, thinking,
            context_window: found.as_ref().map_or(0, |found| found.0), fast: found.and_then(|found| found.1).as_deref() == Some(crate::provider::FAST_TIER),
            role: crate::manager::role(session), team: crate::host::team(session), transcript, replace,
        }
    }

    pub fn replacement(host: &Host, session: &Session) -> Self {
        let projection = session.projection();
        let tokens = crate::compaction::estimate_projected_context_tokens(&projection, &session.path_to_leaf()).tokens;
        let mut transcript = Transcript::from_messages(projection.into_iter().flat_map(|entry| entry.messages));
        transcript.context_tokens = tokens;
        Self::metadata(host, session, transcript, true)
    }

    // Output: the retired transcript, to dispose of back on the worker. Applying a replacement
    // swaps ready state; neither replaying nor freeing a large conversation holds up input.
    pub fn apply(self, slot: &mut Slot) -> Option<Transcript> {
        if slot.parent.is_none() { slot.title = self.title; }
        slot.id = self.id;
        slot.cwd = self.cwd;
        slot.model = self.model;
        slot.thinking = self.thinking;
        slot.context_window = self.context_window;
        slot.fast = self.fast;
        slot.manager = self.role.is_some();
        slot.advisor = self.role == Some(crate::manager::Role::Advisor);
        slot.team = self.team;
        if self.replace { Some(std::mem::replace(&mut slot.transcript, self.transcript)) }
        else { slot.transcript.append(self.transcript); None }
    }
}

struct Following {
    ticket: Weak<()>,
    reader: Follower,
    tokens: u64,
    cost: crate::provider::responses::Tally,
}

#[derive(Default)]
pub(super) struct Worker {
    followers: HashMap<usize, Following>,
    folder_times: HashMap<PathBuf, SystemTime>,
}

impl Worker {
    pub fn prune(&mut self) { self.followers.retain(|_, following| following.ticket.strong_count() != 0); }

    pub fn follow(&mut self, slot: usize, ticket: Weak<()>, path: &Path, host: &Host) -> Result<Option<Followed>, String> {
        let following = self.followers.get_mut(&slot).filter(|following| following.ticket.ptr_eq(&ticket));
        let Some(following) = following else {
            let reader = Follower::open(path).map_err(|error| error.to_string())?;
            return Ok(Some(self.start(slot, ticket, reader, host)));
        };
        let before = following.reader.session.entries().len();
        let Some(mut replace) = following.reader.advance().map_err(|error| error.to_string())? else { return Ok(None) };
        // A new system snapshot changes the folded prompt, so use the full projection for it too.
        replace |= following.reader.session.entries()[before.min(following.reader.session.entries().len())..].iter().any(|entry| matches!(entry, crate::session::SessionEntry::Message { message: AgentMessage::System(_), .. }));
        let result = if replace { Followed::replacement(host, &following.reader.session) } else {
            let messages = following.reader.session.entries()[before..].iter().flat_map(crate::session::entry_context_messages);
            let mut transcript = Transcript::from_messages(messages);
            for message in &transcript.messages {
                if let Some(usage) = crate::compaction::assistant_usage(message) { following.tokens = crate::provider::estimate::context_tokens(usage); }
                else { following.tokens += crate::compaction::estimate_tokens(message); }
                if let AgentMessage::Assistant(reply) = message { following.cost.add(&reply.usage, &reply.provider); }
            }
            (transcript.context_tokens, transcript.cost) = (following.tokens, following.cost);
            Followed::metadata(host, &following.reader.session, transcript, false)
        };
        (following.tokens, following.cost) = (result.transcript.context_tokens, result.transcript.cost);
        Ok(Some(result))
    }

    pub fn start(&mut self, slot: usize, ticket: Weak<()>, reader: Follower, host: &Host) -> Followed {
        let result = Followed::replacement(host, &reader.session);
        self.followers.insert(slot, Following { ticket, reader, tokens: result.transcript.context_tokens, cost: result.transcript.cost });
        result
    }

    pub fn scan(&mut self, request: ScanRequest) -> Scan {
        let ScanRequest { host, mut files, folders, known, tickets, mut projects, started, discover, agents, closed_before } = request;
        for folder in folders {
            for entry in std::fs::read_dir(folder).into_iter().flatten().flatten() {
                if let Some(file) = entry.path().to_str().and_then(|file| file.strip_suffix(crate::runtime::SOCKET_SUFFIX)) { files.insert(file.into()); }
            }
        }
        let mut models = HashMap::new();
        let mut index = None;
        let statuses = files.iter().map(|file| {
            let status = crate::runtime::status(file);
            if status.is_some() && !known.contains(file) {
                let model = index.get_or_insert_with(|| crate::session::index::Index::open(&host.agent_dir)).metadata(file).ok().and_then(|metadata| metadata.1).map_or(String::new(), |(_, id)| id);
                models.insert(file.clone(), model);
            }
            (file.clone(), status)
        }).collect();
        let modified = known.iter().map(|file| (file.clone(), file.metadata().ok().and_then(|metadata| metadata.modified().ok()))).collect();
        let closed = crate::session::closed_sessions(&host.agent_dir);
        let mut summaries = Vec::new();
        if discover {
            for entry in std::fs::read_dir(host.agent_dir.join("sessions")).into_iter().flatten().flatten() {
                let folder = entry.path();
                let Some(modified) = entry.metadata().ok().and_then(|metadata| metadata.modified().ok()) else { continue };
                if self.folder_times.get(&folder) == Some(&modified) { continue; }
                for (modified, path) in crate::session::index::newest(crate::session::index::in_folder(&folder)) {
                    let canonical = crate::tools::paths::canonical(&path);
                    if modified < started || known.contains(&canonical) || closed.contains_key(&canonical) { continue; }
                    if let Ok(summary) = index.get_or_insert_with(|| crate::session::index::Index::open(&host.agent_dir)).summary(&path, modified, false)
                        && summary.cwd.is_dir() { summaries.push(summary); }
                }
                self.folder_times.insert(folder, modified);
            }
        }
        projects.extend(summaries.iter().map(|summary| summary.cwd.clone()));
        let managers = if discover || !projects.is_empty() { managers(&host, projects) } else { Vec::new() };
        Scan { statuses, models, modified, tickets, closed, closed_before, summaries, managers, agents, index }
    }
}

pub(super) type Managers = Vec<(PathBuf, PathBuf, String, bool)>;
pub(super) fn managers(host: &Host, projects: BTreeSet<PathBuf>) -> Managers {
    let managers = projects.into_iter().filter_map(|cwd| crate::manager::project_manager(&host.agent_dir, &cwd).map(|path| (cwd, path, crate::manager::PROJECT_NAME.to_string())))
        .chain(crate::manager::session_file(&host.agent_dir).map(|path| (crate::manager::home(&host.agent_dir), path, crate::manager::assistant_name(&host.agent_dir))));
    managers.flat_map(|(cwd, path, title)| {
        let advisor = crate::manager::advisor(&host.agent_dir, &cwd).map(|file| (cwd.clone(), file, crate::manager::ADVISOR_NAME.to_string(), true));
        [Some((cwd, path, title, false)), advisor]
    }).flatten().map(|(cwd, path, title, advisor)| (cwd, crate::tools::paths::canonical(&path), title, advisor)).collect()
}

pub(super) struct ScanRequest {
    pub host: Arc<Host>,
    pub files: BTreeSet<PathBuf>,
    pub folders: BTreeSet<PathBuf>,
    pub known: BTreeSet<PathBuf>,
    pub tickets: HashMap<PathBuf, Weak<()>>,
    pub projects: BTreeSet<PathBuf>,
    pub started: SystemTime,
    pub discover: bool,
    pub agents: bool,
    pub closed_before: HashMap<PathBuf, i64>,
}

pub(super) struct Scan {
    pub statuses: BTreeMap<PathBuf, Option<Value>>,
    pub models: HashMap<PathBuf, String>,
    pub modified: HashMap<PathBuf, Option<SystemTime>>,
    pub tickets: HashMap<PathBuf, Weak<()>>,
    pub closed: HashMap<PathBuf, i64>,
    pub closed_before: HashMap<PathBuf, i64>,
    pub summaries: Vec<crate::host::SessionSummary>,
    pub managers: Managers,
    pub agents: bool,
    pub index: Option<crate::session::index::Index>,
}
