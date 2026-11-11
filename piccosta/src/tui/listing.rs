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
struct Group {
    cwd: Option<PathBuf>,
    files: VecDeque<(SystemTime, PathBuf)>,
    limit: usize,
    taken: usize,
    other: bool,
}
pub(super) struct Listing {
    pub ticket: Weak<()>,
    pub kind: Kind,
    pub summaries: Vec<SessionSummary>,
    pub managers: follow::Managers,
    pub done: bool,
    host: Arc<Host>,
    index: Option<Index>,
    groups: VecDeque<Group>,
    known: BTreeSet<PathBuf>,
    closed: HashMap<PathBuf, i64>,
    others: usize,
    projects: BTreeSet<PathBuf>,
}
impl Listing {
    fn new(host: Arc<Host>, kind: Kind, ticket: Weak<()>, known: BTreeSet<PathBuf>, projects: BTreeSet<PathBuf>) -> Self {
        let group = |folder: PathBuf, cwd, limit, other| Group { cwd, files: index::newest(index::in_folder(&folder)).into(), limit, taken: 0, other };
        let groups = match &kind {
            Kind::Sidebar => {
                let folders = index::newest(index::folders(&host.agent_dir).into_iter().filter(|folder| *folder != host.session_dir).collect());
                std::iter::once(group(host.session_dir.clone(), Some(host.cwd.clone()), super::RECENT_SESSIONS, false))
                    .chain(folders.into_iter().map(|(_, folder)| group(folder, None, super::RECENT_PER_OTHER_PROJECT, true)))
                    .collect()
            }
            Kind::Resume(false) => VecDeque::from([group(host.session_dir.clone(), Some(host.cwd.clone()), super::PICKER_SESSIONS, false)]),
            Kind::Resume(true) => VecDeque::from([Group {
                cwd: None,
                files: index::newest(index::files(&host.agent_dir, false)).into(),
                limit: usize::MAX,
                taken: 0,
                other: false,
            }]),
            Kind::Projects => index::folders(&host.agent_dir).into_iter().map(|folder| group(folder, None, 1, false)).collect(),
        };
        Self {
            index: Some(Index::open(&host.agent_dir)),
            closed: crate::session::closed_sessions(&host.agent_dir),
            host,
            kind,
            ticket,
            groups,
            known,
            summaries: Vec::new(),
            managers: Vec::new(),
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
        while self
            .groups
            .front()
            .is_some_and(|group| group.files.is_empty() || group.taken == group.limit || (group.other && self.others >= super::OTHER_PROJECTS))
        {
            let group = self.groups.pop_front().unwrap();
            if group.other && group.taken > 0 {
                self.others += 1;
            }
        }
        let Some(group) = self.groups.front_mut() else {
            self.done = true;
            if matches!(self.kind, Kind::Sidebar) {
                self.projects.extend(self.summaries.iter().map(|summary| summary.cwd.clone()));
                self.managers = follow::managers(&self.host, std::mem::take(&mut self.projects));
            }
            return;
        };
        let (modified, path) = group.files.pop_front().unwrap();
        let canonical = crate::tools::paths::canonical(&path);
        let sidebar = matches!(self.kind, Kind::Sidebar);
        if sidebar && (self.known.contains(&canonical) || self.closed.contains_key(&canonical)) {
            return;
        }
        let Ok(header) = index::header(&path) else { return };
        let cwd = PathBuf::from(&header.cwd);
        if group.cwd.as_ref().is_some_and(|expected| *expected != cwd) {
            return;
        }
        if (matches!(self.kind, Kind::Projects) || (sidebar && group.other)) && !cwd.is_dir() {
            group.files.clear();
            return;
        }
        let summary = if matches!(self.kind, Kind::Projects) {
            SessionSummary { path, modified, cwd, title: String::new(), id: header.id, closed: false }
        } else {
            let Ok(summary) = self.index.as_mut().unwrap().summary(&path, modified, self.closed.contains_key(&canonical)) else { return };
            summary
        };
        group.taken += 1;
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
