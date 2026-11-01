// One file, one writer at a time. A model often sends several tool calls in one turn and they run
// in parallel; two edits to the same file must not read the same old contents and then overwrite
// each other. Edits and writes to the same real file wait in line, first come first served, while
// different files go ahead in parallel.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/tools/file-mutation-queue.ts.
// tokio's Mutex already queues its waiters in order, so a lock per file is the whole queue.

use super::paths;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct MutationQueue {
    files: Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>,
}

// Held while a tool changes one file. Dropping it lets the next one in line go, and forgets the
// file once nobody else is waiting on it.
pub struct MutationGuard<'a> {
    queue: &'a MutationQueue,
    key: PathBuf,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl MutationQueue {
    // Input: the absolute path about to change. Output: a guard, once every earlier change to the
    // same real file has finished.
    pub async fn lock(&self, path: &Path) -> MutationGuard<'_> {
        let key = paths::canonical(path);
        let file_lock = {
            let mut files = self.files.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            files.entry(key.clone()).or_default().clone()
        };
        let guard = file_lock.lock_owned().await;
        MutationGuard { queue: self, key, guard: Some(guard) }
    }
}

impl Drop for MutationGuard<'_> {
    fn drop(&mut self) {
        drop(self.guard.take());
        let mut files = self.queue.files.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // With our guard gone, a count of one is the map's own reference: nobody is waiting.
        if files.get(&self.key).is_some_and(|lock| Arc::strong_count(lock) == 1) {
            files.remove(&self.key);
        }
    }
}
