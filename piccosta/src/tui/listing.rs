// Broad catalogs run a file at a time on the existing asker. Each continuation goes to the
// back of its queue, so following, runtime status and wake preparation can pass a cold picker.
use super::{App, Update, follow};
use crate::{
    host::{Host, SessionSummary},
    session::index::{self, Index},
};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    path::PathBuf,
    sync::{Arc, Weak},
    time::SystemTime,
};

pub(super) const LOADING_TITLE: &str = "loading sessions · esc cancels";

#[derive(Clone)]
pub(super) enum Kind {
    Sidebar,
    Resume(bool),
    Projects,
}
pub(super) struct Listing {
    pub ticket: Weak<()>,
    pub kind: Kind,
    pub summaries: Vec<SessionSummary>,
    pub managers: follow::Managers,
    pub done: bool,
    host: Arc<Host>,
    index: Option<Index>,
    files: VecDeque<(SystemTime, PathBuf)>,
    taken: HashMap<PathBuf, usize>,
    resolved: HashMap<PathBuf, PathBuf>,
    local: PathBuf,
    known: BTreeSet<PathBuf>,
    closed: HashMap<PathBuf, i64>,
    others: usize,
    projects: BTreeSet<PathBuf>,
}
impl Listing {
    fn new(host: Arc<Host>, kind: Kind, ticket: Weak<()>, known: BTreeSet<PathBuf>, projects: BTreeSet<PathBuf>) -> Self {
        let local = crate::manager::managed_project(&host.agent_dir, &host.cwd).map_or(host.cwd.as_path(), |(folder, _)| folder).to_path_buf();
        let files = index::newest(index::files(&host.agent_dir, false).into_iter().chain(index::in_folder(&host.session_dir)).collect::<BTreeSet<_>>().into_iter().collect()).into();
        Self {
            index: Some(Index::open(&host.agent_dir)),
            closed: crate::session::closed_sessions(&host.agent_dir),
            host,
            kind,
            ticket,
            files,
            taken: HashMap::new(),
            resolved: HashMap::new(),
            local,
            known,
            summaries: Vec::new(),
            managers: follow::Managers::default(),
            done: false,
            others: 0,
            projects,
        }
    }
    fn step(&mut self) {
        if self.ticket.strong_count() == 0 {
            self.done = true;
            return;
        }
        let Some((modified, path)) = self.files.pop_front() else {
            self.done = true;
            if matches!(self.kind, Kind::Sidebar) {
                self.projects.extend(self.summaries.iter().map(|summary| summary.cwd.clone()));
                self.managers = follow::managers(&self.host, std::mem::take(&mut self.projects));
            }
            return;
        };
        let canonical = crate::tools::paths::canonical(&path);
        let sidebar = matches!(self.kind, Kind::Sidebar);
        if sidebar && (self.known.contains(&canonical) || self.closed.contains_key(&canonical)) {
            return;
        }
        let Ok(header) = index::header(&path) else { return };
        let cwd = PathBuf::from(&header.cwd);
        let project = self.resolved.entry(cwd.clone()).or_insert_with(|| {
            crate::manager::managed_project(&self.host.agent_dir, &cwd).map_or(cwd.as_path(), |(folder, _)| folder).to_path_buf()
        });
        let other = *project != self.local;
        let limit = match self.kind {
            Kind::Sidebar if other => super::RECENT_PER_OTHER_PROJECT,
            Kind::Sidebar => super::RECENT_SESSIONS,
            Kind::Resume(false) if other => return,
            Kind::Resume(false) => super::PICKER_SESSIONS,
            Kind::Resume(true) => usize::MAX,
            Kind::Projects => 1,
        };
        let taken = self.taken.get(project).copied().unwrap_or(0);
        if taken >= limit || (sidebar && other && taken == 0 && self.others >= super::OTHER_PROJECTS)
            || ((matches!(self.kind, Kind::Projects) || (sidebar && other)) && !project.is_dir()) { return; }
        let summary = if matches!(self.kind, Kind::Projects) {
            SessionSummary { path, modified, cwd: project.clone(), title: String::new(), id: header.id, closed: false }
        } else {
            let Ok(summary) = self.index.as_mut().unwrap().summary(&path, modified, self.closed.contains_key(&canonical)) else { return };
            summary
        };
        if sidebar && other && taken == 0 { self.others += 1; }
        self.taken.insert(project.clone(), taken + 1);
        self.summaries.push(summary);
    }
}

impl App {
    pub(super) fn request_listing(&mut self, kind: Kind, filter: &str) {
        let place = usize::from(!matches!(kind, Kind::Sidebar));
        if place == 1 {
            self.pick(LOADING_TITLE, Vec::new(), filter, |_, _| {});
        }
        self.listing_tickets[place] = Arc::new(());
        let (host, ticket, known) =
            (self.host.clone(), Arc::downgrade(&self.listing_tickets[place]), self.slots.iter().filter_map(|slot| slot.path.clone()).collect());
        let projects = std::iter::once(self.host.cwd.clone()).chain(self.slots.iter().map(|slot| slot.cwd.clone())).collect();
        let _ = self.asks.send(Box::new(move |_| {
            let mut listing = Box::new(Listing::new(host, kind, ticket, known, projects));
            listing.step();
            (0, Update::Listed(listing))
        }));
    }
    pub(super) fn listed_result(&mut self, mut listing: Box<Listing>) {
        let place = usize::from(!matches!(listing.kind, Kind::Sidebar));
        if !listing.ticket.ptr_eq(&Arc::downgrade(&self.listing_tickets[place])) || (place == 1 && self.picker.is_none()) || self.quit || self.restart_waiting {
            self.retire(listing);
        } else if !listing.done {
            let _ = self.asks.send(Box::new(move |_| {
                listing.step();
                (0, Update::Listed(listing))
            }));
        } else {
            if let Some(index) = listing.index.take() {
                self.persist_index(index);
            }
            let summaries = std::mem::take(&mut listing.summaries);
            match &listing.kind {
                Kind::Sidebar => {
                    self.list_sessions(summaries);
                    self.list_managers(std::mem::take(&mut listing.managers));
                }
                Kind::Resume(global) => {
                    let filter = self.picker.take().unwrap().filter;
                    if *global {
                        self.all_sessions_ready(summaries, &filter);
                    } else {
                        self.sessions_ready(summaries, &filter);
                    }
                }
                Kind::Projects => self.projects_ready(summaries),
            }
            self.retire(listing);
        }
    }
}
