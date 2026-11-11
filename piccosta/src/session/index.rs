// A disposable index of session prefixes. Catalog reads never load accounting sections; each
// section has its own checksum so fresh processes can seek straight to the facts they need.
// Statistics retain pi 1.4.0's SessionStats semantics (MIT; see LICENSE.md).
use super::{Cursor, Follower, OpenError, Record, Session, SessionEntry, SessionHeader};
use crate::{
    host::SessionSummary,
    manager::Role,
    message::{AgentMessage, Content, Usage},
    provider::responses::Tally,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

// Bump when either the stored representation or either accounting reducer changes.
const VERSION: u32 = 2;
const FILE: &str = "session-index";

#[derive(Clone, Serialize, Deserialize)]
struct Span {
    offset: u64,
    length: usize,
    hash: [u8; 32],
}
#[derive(Clone, Serialize, Deserialize)]
struct Catalog {
    header: SessionHeader,
    cursor: Option<Cursor>,
    name: Option<String>,
    prompt: Option<String>,
    role: Option<Role>,
    detail: Option<Span>,
}
impl Catalog {
    fn new(header: SessionHeader) -> Self {
        Self { header, cursor: None, name: None, prompt: None, role: None, detail: None }
    }
    fn add(&mut self, entry: &SessionEntry) {
        match entry {
            SessionEntry::SessionInfo { name: Some(name), .. } => self.name = Some(crate::host::title_text(name)),
            SessionEntry::Message { message: AgentMessage::User(user), .. } if self.prompt.is_none() => {
                self.prompt = Some(crate::host::prompt_title(user))
            }
            _ => {}
        }
        if self.role.is_none() {
            self.role = crate::manager::entry_role(entry);
        }
    }
    fn summary(&self, path: &Path, modified: SystemTime, closed: bool) -> SessionSummary {
        SessionSummary {
            path: path.into(),
            modified,
            closed,
            id: self.header.id.clone(),
            cwd: self.header.cwd.clone().into(),
            title: self.name.as_ref().or(self.prompt.as_ref()).cloned().unwrap_or_else(|| "new session".into()),
        }
    }
}

// Keep every entry identity, including ones without cost. The manager deduplicates before it
// changes provider provenance; a copied assistant must not settle or rename another file's work.
#[derive(Serialize, Deserialize)]
struct Fact {
    id: String,
    parent: Option<String>,
    timestamp: String,
    model: Option<(String, String)>,
    assistant: Option<String>,
    spent: Vec<(Usage, Option<String>, Option<i64>)>,
}
#[derive(Default, Serialize, Deserialize)]
struct Detail {
    cursor: Option<Cursor>,
    facts: Vec<Fact>,
    misses: Vec<(i64, Value)>,
    stats: Statistics,
}
impl Detail {
    fn add(&mut self, entry: &SessionEntry) {
        self.stats.add(entry);
        let mut fact = Fact {
            id: entry.id().into(),
            parent: entry.parent_id().map(str::to_string),
            timestamp: entry.base().timestamp.clone(),
            model: None,
            assistant: None,
            spent: Vec::new(),
        };
        let at = crate::clock::ms_from_iso(&fact.timestamp);
        match entry {
            SessionEntry::ModelChange { provider, model_id, .. } => fact.model = Some((provider.clone(), model_id.clone())),
            SessionEntry::Message { message: AgentMessage::Assistant(answer), .. } => {
                fact.model = Some((answer.provider.clone(), answer.model.clone()));
                fact.assistant = Some(answer.provider.clone());
                fact.spent.push((answer.usage.clone(), Some(answer.provider.clone()), Some(answer.timestamp)));
            }
            SessionEntry::Usage { provider, usage, .. } => fact.spent.push((usage.clone(), Some(provider.clone()), at)),
            SessionEntry::Compaction { usage: Some(usage), .. } => fact.spent.push((usage.clone(), None, at)),
            SessionEntry::Message { message: AgentMessage::ToolResult(result), .. } => {
                for spent in result.details.as_ref().and_then(|details| details.get("usage")).and_then(Value::as_array).into_iter().flatten() {
                    if let (Some(usage), Some(provider)) =
                        (spent.get("usage").and_then(|usage| serde_json::from_value(usage.clone()).ok()), spent.get("provider").and_then(Value::as_str))
                    {
                        fact.spent.push((usage, Some(provider.into()), Some(result.timestamp)));
                    }
                }
            }
            SessionEntry::Custom { custom_type, data: Some(miss), .. } if custom_type == crate::cache::MISS_TYPE => {
                if let Some(at) = at {
                    self.misses.push((at, miss.clone()));
                }
            }
            _ => {}
        }
        self.facts.push(fact);
    }
    fn model(&self) -> Option<(String, String)> {
        let entries: HashMap<_, _> = self.facts.iter().map(|fact| (fact.id.as_str(), fact)).collect();
        let mut current = self.facts.last();
        // Malformed imported parent cycles must not wedge a catalog request.
        for _ in 0..self.facts.len() {
            let fact = current?;
            if let Some(model) = &fact.model {
                return Some(model.clone());
            }
            current = fact.parent.as_deref().and_then(|parent| entries.get(parent).copied());
        }
        None
    }
}

#[derive(Default, Serialize, Deserialize)]
pub(crate) struct Statistics {
    users: u64,
    assistants: u64,
    calls: u64,
    results: u64,
    messages: u64,
    tokens: [u64; 5],
    tokens_unknown: bool,
    cost: Tally,
    provider: String,
}
impl Statistics {
    fn spent(&mut self, usage: &Usage, provider: &str) {
        self.tokens_unknown |= usage.tokens_unknown;
        if !usage.tokens_unknown {
            for (total, part) in self.tokens.iter_mut().zip([usage.input, usage.output, usage.cache_read, usage.cache_write, usage.total_tokens]) {
                *total += part;
            }
        }
        self.cost.add(usage, provider);
    }
    pub fn add(&mut self, entry: &SessionEntry) {
        match entry {
            SessionEntry::ModelChange { provider, .. } => self.provider.clone_from(provider),
            SessionEntry::Usage { provider, usage, .. } => self.spent(usage, provider),
            SessionEntry::Compaction { usage: Some(usage), .. } | SessionEntry::BranchSummary { usage: Some(usage), .. } => {
                self.spent(usage, &self.provider.clone())
            }
            SessionEntry::Message { message, .. } => {
                self.messages += 1;
                match message {
                    AgentMessage::User(_) => self.users += 1,
                    AgentMessage::Assistant(answer) => {
                        self.assistants += 1;
                        self.calls += answer.content.iter().filter(|part| matches!(part, Content::ToolCall { .. })).count() as u64;
                        self.spent(&answer.usage, &answer.provider);
                    }
                    AgentMessage::ToolResult(result) => {
                        self.results += 1;
                        if let Some(usage) = &result.usage {
                            self.spent(usage, "");
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    pub fn value(&self) -> serde_json::Map<String, Value> {
        let [input, output, cache_read, cache_write, total] = self.tokens;
        let mut value = json!({"userMessages":self.users,"assistantMessages":self.assistants,"toolCalls":self.calls,"toolResults":self.results,"totalMessages":self.messages,
            "tokens":{"input":input,"output":output,"cacheRead":cache_read,"cacheWrite":cache_write,"total":total},"cost":self.cost.total}).as_object().unwrap().clone();
        if self.tokens_unknown {
            value.insert("tokensUnknown".into(), true.into());
        }
        if self.cost.unknown() {
            value.insert("costUnknown".into(), true.into());
        }
        value
    }
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    version: u32,
    hash: [u8; 32],
    catalog: BTreeMap<PathBuf, Catalog>,
}

pub type Metadata = (Option<Role>, Option<(String, String)>, serde_json::Map<String, Value>);

pub struct Index {
    agent_dir: PathBuf,
    catalog: BTreeMap<PathBuf, Catalog>,
    source: Option<fs::File>,
    base: u64,
    details: HashMap<PathBuf, Detail>,
    dirty: bool,
}
impl Index {
    pub fn open(agent_dir: &Path) -> Self {
        let mut index = Self { agent_dir: agent_dir.into(), catalog: BTreeMap::new(), source: None, base: 0, details: HashMap::new(), dirty: false };
        let loaded = (|| -> Option<_> {
            let mut input = io::BufReader::new(super::open_reader(&agent_dir.join(FILE)).ok()?);
            let mut bytes = Vec::new();
            let base = input.read_until(b'\n', &mut bytes).ok()? as u64;
            let envelope: Envelope = serde_json::from_slice(&bytes).ok()?;
            if envelope.version != VERSION || hash(&serde_json::to_vec(&envelope.catalog).ok()?) != envelope.hash {
                return None;
            }
            let length = input.get_ref().metadata().ok()?.len();
            if envelope.catalog.values().any(|record| {
                record.header.kind != "session"
                    || record.header.version.unwrap_or(1) < super::CURRENT_SESSION_VERSION
                    || record.cursor.as_ref().is_some_and(|cursor| !cursor.valid())
                    || record.detail.as_ref().is_some_and(|span| {
                        base.checked_add(span.offset).and_then(|offset| offset.checked_add(span.length as u64)).is_none_or(|end| end > length)
                    })
            }) {
                return None;
            }
            Some((envelope.catalog, input.into_inner(), base))
        })();
        if let Some((catalog, source, base)) = loaded {
            index.catalog = catalog;
            index.source = Some(source);
            index.base = base;
        }
        index
    }

    pub fn summary(&mut self, path: &Path, modified: SystemTime, closed: bool) -> Result<SessionSummary, OpenError> {
        self.refresh(path, false)?;
        Ok(self.catalog[path].summary(path, modified, closed))
    }

    fn detail(&mut self, path: &Path) -> &mut Detail {
        if !self.details.contains_key(path) {
            let detail = self
                .catalog
                .get(path)
                .and_then(|record| record.detail.as_ref())
                .and_then(|span| self.section(span).ok())
                .and_then(|bytes| serde_json::from_slice::<Detail>(&bytes).ok())
                .filter(|detail| detail.cursor.as_ref().is_none_or(Cursor::valid))
                .unwrap_or_default();
            self.details.insert(path.into(), detail);
        }
        self.details.get_mut(path).unwrap()
    }

    fn refresh(&mut self, path: &Path, rich: bool) -> Result<(), OpenError> {
        let previous = if rich { self.detail(path).cursor.clone() } else { self.catalog.get(path).and_then(|record| record.cursor.clone()) };
        let mut changed = false;
        let cursor = Cursor::scan(
            path,
            previous.as_ref(),
            |line| rich || line.contains("user") || line.contains("session_info") || line.contains("piccosta-manager") || line.contains("\\u"),
            |record| {
                changed = true;
                match record {
                    Record::Header(header) => {
                        self.catalog.insert(path.into(), Catalog::new(header));
                        if rich {
                            self.details.insert(path.into(), Detail::default());
                        }
                    }
                    Record::Entry(entry) => {
                        self.catalog.get_mut(path).unwrap().add(&entry);
                        if rich {
                            self.details.get_mut(path).unwrap().add(&entry);
                        }
                    }
                }
            },
        )?;
        if rich {
            self.details.get_mut(path).unwrap().cursor = Some(cursor.clone());
            // A rich refresh also supplies the catalog. Never advance a rich cursor merely
            // because a title-only scan consumed bytes it did not account for.
        }
        self.catalog.get_mut(path).unwrap().cursor = Some(cursor);
        self.dirty |= changed
            || previous.as_ref().is_none_or(|old| {
                old.consumed != self.catalog[path].cursor.as_ref().unwrap().consumed || old.modified != self.catalog[path].cursor.as_ref().unwrap().modified
            });
        Ok(())
    }

    pub fn role(&mut self, path: &Path) -> Result<Option<Role>, OpenError> {
        self.refresh(path, false)?;
        Ok(self.catalog[path].role)
    }

    pub fn metadata(&mut self, path: &Path) -> Result<Metadata, OpenError> {
        self.refresh(path, true)?;
        Ok((self.catalog[path].role, self.details[path].model(), self.details[path].stats.value()))
    }

    // The selected transcript is read exactly once. Its cursor and entries seed both sections,
    // and the caller uses this same Session for authorization and projected messages.
    pub fn read(&mut self, path: &Path) -> Result<Session, OpenError> {
        let reader = Follower::open(path)?;
        // A title scan may be ahead of accounting. Only the detail cursor proves these facts
        // already describe this prefix; a missing or corrupt section still needs seeding.
        if self.detail(path).cursor.as_ref() == Some(&reader.cursor) { return Ok(reader.session); }
        let mut catalog = Catalog::new(reader.session.header().clone());
        let mut detail = Detail::default();
        for entry in reader.session.entries() {
            catalog.add(entry);
            detail.add(entry);
        }
        catalog.cursor = Some(reader.cursor.clone());
        detail.cursor = Some(reader.cursor);
        self.catalog.insert(path.into(), catalog);
        self.details.insert(path.into(), detail);
        self.dirty = true;
        Ok(reader.session)
    }

    fn section(&self, span: &Span) -> io::Result<Vec<u8>> {
        let mut file = self.source.as_ref().ok_or_else(|| io::Error::other("missing index section"))?.try_clone()?;
        let bytes = super::read_at(&mut file, self.base + span.offset, span.length)?;
        if bytes.len() != span.length || hash(&bytes) != span.hash {
            return Err(io::Error::other("corrupt index section"));
        }
        Ok(bytes)
    }

    // Cache progress is expendable. A unique temporary file and rename permit concurrent
    // readers/writers without locks or merge retries; the last complete snapshot wins.
    pub fn save(&mut self) {
        self.dirty |= self.catalog.keys().any(|path| path.try_exists().is_ok_and(|exists| !exists));
        if !self.dirty {
            return;
        }
        if let Err(error) = self.publish() {
            eprintln!("piccosta session index: {error}");
        }
    }
    fn publish(&mut self) -> io::Result<()> {
        let mut payloads = Vec::new();
        let mut offset = 0;
        for (path, catalog) in &self.catalog {
            if !path.is_file() {
                continue;
            }
            let bytes = if let Some(detail) = self.details.get(path) {
                Some(serde_json::to_vec(detail)?)
            } else {
                catalog.detail.as_ref().and_then(|span| self.section(span).ok())
            };
            payloads.push((path.clone(), catalog.clone(), bytes));
        }
        let mut catalog = BTreeMap::new();
        for (path, record, bytes) in &mut payloads {
            record.detail = bytes.as_ref().map(|bytes| {
                let span = Span { offset, length: bytes.len(), hash: hash(bytes) };
                offset += bytes.len() as u64;
                span
            });
            catalog.insert(path.clone(), record.clone());
        }
        let envelope = Envelope { version: VERSION, hash: hash(&serde_json::to_vec(&catalog)?), catalog };
        crate::platform::private_dir(&self.agent_dir)?;
        let path = self.agent_dir.join(FILE);
        let partial = self.agent_dir.join(format!("{FILE}-{}.partial", uuid::Uuid::new_v4()));
        let result = (|| {
            let mut output = crate::platform::private_file(&mut fs::OpenOptions::new()).create_new(true).write(true).open(&partial)?;
            output.write_all(&serde_json::to_vec(&envelope)?)?;
            output.write_all(b"\n")?;
            for (_, _, bytes) in payloads {
                if let Some(bytes) = bytes {
                    output.write_all(&bytes)?;
                }
            }
            output.sync_all()?;
            fs::rename(&partial, path)
        })();
        let _ = fs::remove_file(partial);
        // Reload before another publication: the old section offsets belong to the old handle.
        if result.is_ok() {
            *self = Self::open(&self.agent_dir);
        }
        result
    }

    pub fn usage(&mut self, files: &[PathBuf], now: i64, day_ms: i64) -> BTreeMap<String, (Tally, Tally)> {
        let mut totals = BTreeMap::<String, (Tally, Tally)>::new();
        let mut counted = std::collections::HashSet::new();
        for file in files {
            if self.refresh(file, true).is_err() {
                continue;
            }
            let (mut tallies, mut provider) = (BTreeMap::<String, (Tally, Tally)>::new(), String::new());
            for fact in &self.details[file].facts {
                if !counted.insert((fact.id.clone(), fact.timestamp.clone())) {
                    continue;
                }
                if let Some(answer) = &fact.assistant {
                    provider.clone_from(answer);
                }
                for (usage, spent_by, when) in &fact.spent {
                    let spent_by = spent_by.as_ref().unwrap_or(&provider);
                    let (day, week) = tallies.entry(spent_by.clone()).or_default();
                    let when = when.unwrap_or(now);
                    if now - when < 7 * day_ms {
                        week.add(usage, spent_by);
                    }
                    if now - when < day_ms {
                        day.add(usage, spent_by);
                    }
                }
            }
            for (provider, (day, week)) in tallies {
                let total = totals.entry(provider).or_default();
                total.0.merge(day);
                total.1.merge(week);
            }
        }
        totals
    }
    pub fn misses(&mut self, files: &[PathBuf], since: i64) -> Vec<(i64, String, Value)> {
        let mut misses = Vec::new();
        for path in files {
            if self.refresh(path, true).is_err() {
                continue;
            }
            let project = Path::new(&self.catalog[path].header.cwd).file_name().unwrap_or_default().to_string_lossy().into_owned();
            misses.extend(self.details[path].misses.iter().filter(|(at, _)| *at >= since).map(|(at, miss)| (*at, project.clone(), miss.clone())));
        }
        misses
    }
}
fn hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

// Directory order is deliberately retained for usage's fork ownership. Presentation callers
// sort candidates before asking for titles, rather than refreshing each folder's entire list.
pub fn files(agent_dir: &Path, children: bool) -> Vec<PathBuf> {
    folders(agent_dir)
        .into_iter()
        .flat_map(|folder| if children { vec![folder.join("subagents"), folder] } else { vec![folder] })
        .flat_map(|folder| in_folder(&folder))
        .collect()
}
pub fn folders(agent_dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(agent_dir.join("sessions")).into_iter().flatten().flatten().filter(|entry| entry.path().is_dir()).map(|entry| entry.path()).collect()
}
pub fn in_folder(folder: &Path) -> Vec<PathBuf> {
    fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "jsonl"))
        .collect()
}
pub fn newest(files: Vec<PathBuf>) -> Vec<(SystemTime, PathBuf)> {
    let mut files: Vec<_> = files.into_iter().filter_map(|path| Some((path.metadata().ok()?.modified().ok()?, path))).collect();
    files.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    files
}
pub fn header(path: &Path) -> Result<SessionHeader, OpenError> {
    let file = super::open_reader(path)?;
    let length = file.metadata()?.len();
    let mut input = io::BufReader::new(file.take(length));
    let mut bytes = Vec::new();
    while input.read_until(b'\n', &mut bytes)? != 0 {
        if let Some(value) = super::parse_line(&String::from_utf8_lossy(&bytes)) {
            return super::parse_header(path, value);
        }
        bytes.clear();
    }
    Err(OpenError::NotASession(path.into()))
}

// Full ids need neither a title scan nor the snapshot. A canonical filename cannot outrank a
// newer imported copy, and a duplicate outside the caller's scope cannot hide one inside it.
pub fn exact(files: &[PathBuf], wanted: &str, cwd: Option<&Path>) -> Option<(PathBuf, SessionHeader)> {
    newest(files.to_vec()).into_iter().find_map(|(_, path)| {
        let header = header(&path).ok()?;
        (header.id == wanted && cwd.is_none_or(|cwd| Path::new(&header.cwd) == cwd)).then_some((path, header))
    })
}
