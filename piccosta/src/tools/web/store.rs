// Where web_search and fetch_content keep what they found, so get_search_content can page
// through it by responseId, in this run and after the session is reopened.
//
// Based on pi-web-access 0.44.0 (MIT, see LICENSE.md): storage.ts. The record
// shapes, the "web-search-results" session entries and the web-search-cache files are the same,
// so a session written by the TypeScript piccosta restores here and the other way round.
// - A search result goes into the session whole.
// - A fetch result's pages go to <agent dir>/web-search-cache/<id>.json (mode 0600), and the
//   session entry keeps only each URL's title, error and length plus a pointer to that file.
// - Fetched pages expire after an hour; the cache holds at most 128 files and 128 MB.

use super::extract::Extracted;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

// storage.ts:9-16.
pub const SESSION_ENTRY_TYPE: &str = "web-search-results";
const CACHE_TTL_MS: i64 = 60 * 60 * 1000;
const FETCH_CACHE_DIR: &str = "web-search-cache";
const FETCH_CACHE_VERSION: i64 = 1;
const MAX_METADATA_TEXT: usize = 8192;
const CACHE_MAX_ENTRIES: usize = 128;
const CACHE_MAX_BYTES: u64 = 128 * 1024 * 1024;

// A responseId: the time in base 36 then six random base-36 characters (storage.ts generateId).
pub fn generate_id() -> String {
    let mut millis = crate::clock::now_ms() as u64;
    let digits = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut time = Vec::new();
    while millis > 0 {
        time.push(digits[(millis % 36) as usize]);
        millis /= 36;
    }
    time.reverse();
    let random: String = uuid::Uuid::new_v4().as_bytes().iter().take(6).map(|byte| digits[(*byte % 36) as usize] as char).collect();
    format!("{}{random}", String::from_utf8_lossy(&time))
}

fn is_cache_id(id: &str) -> bool {
    !id.is_empty() && id.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn truncate_metadata(text: &str) -> String {
    if crate::js::utf16_len(text) <= MAX_METADATA_TEXT {
        return text.to_string();
    }
    let units: Vec<u16> = text.encode_utf16().take(MAX_METADATA_TEXT).collect();
    format!("{}...", String::from_utf16_lossy(&units))
}

// What the session entry says about each fetched URL.
fn url_metadata(urls: &[Extracted]) -> Vec<Value> {
    urls.iter()
        .map(|url| {
            let mut meta = Map::new();
            meta.insert("url".into(), json!(truncate_metadata(&url.url)));
            meta.insert("title".into(), json!(truncate_metadata(&url.title)));
            meta.insert("error".into(), url.error.as_deref().map(truncate_metadata).map(Value::from).unwrap_or(Value::Null));
            meta.insert("contentLength".into(), json!(crate::js::utf16_len(&url.content)));
            if let Some(mime) = &url.mime_type {
                meta.insert("mimeType".into(), json!(truncate_metadata(mime)));
            }
            if let Some(status) = url.status {
                meta.insert("status".into(), json!(status));
            }
            Value::Object(meta)
        })
        .collect()
}

// A stored fetch whose pages are gone: every URL keeps its title and says why (storage.ts
// unavailableFetchData).
fn unavailable(record: &Value, reason: &str) -> Value {
    let metadata: Vec<Value> = match record.get("urlMetadata").and_then(Value::as_array) {
        Some(metadata) => metadata.clone(),
        None => record.get("urls").and_then(Value::as_array).map(|urls| {
            let extracted: Vec<Extracted> = urls.iter().filter_map(|url| serde_json::from_value(url.clone()).ok()).collect();
            url_metadata(&extracted)
        }).unwrap_or_default(),
    };
    let urls: Vec<Value> = metadata
        .iter()
        .map(|meta| {
            let mut url = Map::new();
            url.insert("url".into(), meta.get("url").cloned().unwrap_or(json!("")));
            url.insert("title".into(), meta.get("title").cloned().unwrap_or(json!("")));
            url.insert("content".into(), json!(""));
            url.insert("error".into(), json!(reason));
            for key in ["mimeType", "status"] {
                if let Some(value) = meta.get(key) {
                    url.insert(key.into(), value.clone());
                }
            }
            Value::Object(url)
        })
        .collect();
    let mut out = record.as_object().cloned().unwrap_or_default();
    out.insert("urls".into(), Value::Array(urls));
    Value::Object(out)
}

fn timestamp_of(record: &Value) -> i64 {
    record.get("timestamp").and_then(Value::as_f64).unwrap_or(0.0) as i64
}

fn modified_ms(meta: &fs::Metadata) -> i64 {
    meta.modified().ok().and_then(|time| time.duration_since(UNIX_EPOCH).ok()).map(|age| age.as_millis() as i64).unwrap_or(0)
}

// Stored results for one session, in memory, backed by the session file and the cache dir.
pub struct Store {
    results: Mutex<HashMap<String, Value>>,
    cache_dir: PathBuf,
    // Entries to append to the session, drained by the agent after each tool batch.
    pending_entries: Mutex<Vec<Value>>,
}

impl Store {
    pub fn new(agent_dir: &Path) -> Store {
        Store { results: Mutex::new(HashMap::new()), cache_dir: agent_dir.join(FETCH_CACHE_DIR), pending_entries: Mutex::new(Vec::new()) }
    }

    // Session entries written since the last call, oldest first.
    pub fn take_entries(&self) -> Vec<Value> {
        std::mem::take(&mut *self.pending_entries.lock().unwrap())
    }

    fn publish(&self, entry: Value) {
        self.pending_entries.lock().unwrap().push(entry);
    }

    // A search's results: kept whole, in memory and in the session.
    pub fn store_search(&self, id: &str, queries: Value) {
        let record = json!({"id": id, "type": "search", "timestamp": crate::clock::now_ms(), "queries": queries});
        self.results.lock().unwrap().insert(id.to_string(), record.clone());
        self.publish(record);
    }

    // A fetch's pages: in memory whole, on disk in the cache, in the session as metadata
    // (storage.ts storeFetchedContentResult).
    pub fn store_fetch(&self, id: &str, urls: &[Extracted]) {
        let now = crate::clock::now_ms();
        self.expire_in_memory(now);
        let full = json!({"id": id, "type": "fetch", "timestamp": now, "urls": urls});
        let (memory, session) = match self.write_cache(id, &full) {
            Ok(reference) => {
                let mut memory = full.as_object().cloned().unwrap_or_default();
                memory.insert("fetchCache".into(), reference.clone());
                memory.insert("urlMetadata".into(), Value::Array(url_metadata(urls)));
                let session = json!({"id": id, "type": "fetch", "timestamp": now, "urlMetadata": url_metadata(urls), "fetchCache": reference});
                (Value::Object(memory), session)
            }
            Err(error) => {
                let message = truncate_metadata(&format!("Failed to write fetched content cache: {error}"));
                let mut memory = full.as_object().cloned().unwrap_or_default();
                memory.insert("fetchCacheError".into(), json!(message));
                let session = json!({"id": id, "type": "fetch", "timestamp": now, "urlMetadata": url_metadata(urls), "fetchCacheError": message});
                (Value::Object(memory), session)
            }
        };
        self.results.lock().unwrap().insert(id.to_string(), memory);
        self.publish(session);
    }

    fn expire_in_memory(&self, now: i64) {
        let mut results = self.results.lock().unwrap();
        for record in results.values_mut() {
            if record.get("type").and_then(Value::as_str) == Some("fetch") && now - timestamp_of(record) >= CACHE_TTL_MS {
                *record = unavailable(record, "Cached fetched content is missing or expired");
            }
        }
    }

    fn ensure_cache_dir(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.cache_dir)?;
        crate::platform::permissions(&self.cache_dir, 0o700)
    }

    // Write <id>.json beside its siblings by a temp file and rename, after making room for it.
    fn write_cache(&self, id: &str, full: &Value) -> Result<Value, String> {
        if !is_cache_id(id) {
            return Err(format!("Invalid fetched content cache id: {id}"));
        }
        let serialized = serde_json::to_string(full).map_err(|error| error.to_string())?;
        if serialized.len() as u64 > CACHE_MAX_BYTES {
            return Err(format!("Fetched content cache entry exceeds {CACHE_MAX_BYTES} bytes"));
        }
        self.ensure_cache_dir().map_err(|error| error.to_string())?;
        let key = format!("{id}.json");
        self.prune(crate::clock::now_ms(), Some(&key), Some(serialized.len() as u64));
        let temp_name = format!("{key}.{}.{}.{}.tmp", std::process::id(), crate::clock::now_ms(), uuid::Uuid::new_v4().simple());
        let temp_path = self.cache_dir.join(&temp_name);
        let written = (|| -> std::io::Result<()> {
            let mut file = crate::platform::private_file(fs::OpenOptions::new().write(true).create_new(true)).open(&temp_path)?;
            file.write_all(serialized.as_bytes())?;
            file.sync_all()?;
            fs::rename(&temp_path, self.cache_dir.join(&key))
        })();
        if let Err(error) = written {
            let _ = fs::remove_file(&temp_path);
            return Err(error.to_string());
        }
        self.prune(crate::clock::now_ms(), Some(&key), None);
        Ok(json!({"version": FETCH_CACHE_VERSION, "key": key, "storedAt": crate::clock::now_ms()}))
    }

    // Drop expired cache files, then the oldest until the cache is within its limits, counting a
    // file about to be written (storage.ts pruneFetchCache).
    fn prune(&self, now: i64, keep: Option<&str>, reserve: Option<u64>) {
        let Ok(entries) = fs::read_dir(&self.cache_dir) else { return };
        let mut files: Vec<(String, u64, i64)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_cache = name.ends_with(".json") && is_cache_id(name.trim_end_matches(".json"));
            let is_temp = name.ends_with(".tmp") && name.contains(".json.");
            if !is_cache && !is_temp {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            let modified = modified_ms(&meta);
            if now - modified >= CACHE_TTL_MS {
                let _ = fs::remove_file(entry.path());
                continue;
            }
            if is_cache {
                files.push((name, meta.len(), modified));
            }
        }
        files.sort_by(|a, b| a.2.cmp(&b.2).then(a.0.cmp(&b.0)));
        let usage = |files: &[(String, u64, i64)]| {
            let replaced = keep.and_then(|key| files.iter().find(|file| file.0 == key)).map(|file| file.1);
            let entries = files.len() + usize::from(reserve.is_some() && replaced.is_none());
            let bytes = files.iter().map(|file| file.1).sum::<u64>() + reserve.map(|bytes| bytes - replaced.unwrap_or(0).min(bytes)).unwrap_or(0);
            (entries, bytes)
        };
        while {
            let (entries, bytes) = usage(&files);
            entries > CACHE_MAX_ENTRIES || bytes > CACHE_MAX_BYTES
        } {
            let Some(index) = files.iter().position(|file| Some(file.0.as_str()) != keep) else { break };
            let (name, _, _) = files.remove(index);
            let _ = fs::remove_file(self.cache_dir.join(name));
        }
    }

    // A stored result by id, a fetch's pages loaded from the cache when only the pointer is in
    // memory (storage.ts getResult, readCachedFetchData).
    pub fn get(&self, id: &str) -> Option<Value> {
        let record = self.results.lock().unwrap().get(id).cloned()?;
        let loaded = self.load_fetch(&record);
        if loaded != record {
            self.results.lock().unwrap().insert(id.to_string(), loaded.clone());
        }
        Some(loaded)
    }

    fn load_fetch(&self, record: &Value) -> Value {
        if record.get("type").and_then(Value::as_str) != Some("fetch") {
            return record.clone();
        }
        if crate::clock::now_ms() - timestamp_of(record) >= CACHE_TTL_MS {
            return unavailable(record, "Cached fetched content is missing or expired");
        }
        if record.get("urls").and_then(Value::as_array).is_some() {
            return record.clone();
        }
        let Some(reference) = record.get("fetchCache") else {
            let reason = record.get("fetchCacheError").and_then(Value::as_str).unwrap_or("Cached fetched content is unavailable");
            return unavailable(record, reason);
        };
        let key = reference.get("key").and_then(Value::as_str).unwrap_or("");
        if !key.ends_with(".json") || !is_cache_id(key.trim_end_matches(".json")) {
            return unavailable(record, "Cached fetched content is missing or expired");
        }
        let text = match fs::read_to_string(self.cache_dir.join(key)) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return unavailable(record, "Cached fetched content is missing or expired"),
            Err(error) => return unavailable(record, &format!("Cached fetched content could not be read: {error}")),
        };
        let parsed: Value = match serde_json::from_str(&text) {
            Ok(parsed) => parsed,
            Err(_) => return unavailable(record, "Cached fetched content is invalid"),
        };
        if parsed.get("type").and_then(Value::as_str) != Some("fetch") || parsed.get("id") != record.get("id") || parsed.get("urls").and_then(Value::as_array).is_none() {
            return unavailable(record, "Cached fetched content is invalid");
        }
        let mut loaded = parsed.as_object().cloned().unwrap_or_default();
        loaded.insert("fetchCache".into(), reference.clone());
        if let Some(metadata) = record.get("urlMetadata") {
            loaded.insert("urlMetadata".into(), metadata.clone());
        }
        Value::Object(loaded)
    }

    // Load a reopened session's results: every "web-search-results" entry on its branch younger
    // than an hour (storage.ts loadSessionResults).
    pub fn restore<'a>(&self, entries: impl IntoIterator<Item = &'a crate::session::SessionEntry>) {
        let now = crate::clock::now_ms();
        self.prune(now, None, None);
        let mut results = self.results.lock().unwrap();
        for entry in entries {
            let crate::session::SessionEntry::Custom { custom_type, data: Some(data), .. } = entry else { continue };
            if custom_type != SESSION_ENTRY_TYPE {
                continue;
            }
            let Some(id) = data.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) else { continue };
            let kind = data.get("type").and_then(Value::as_str).unwrap_or("");
            if !matches!(kind, "search" | "fetch") || data.get("timestamp").and_then(Value::as_f64).is_none() || now - timestamp_of(data) >= CACHE_TTL_MS {
                continue;
            }
            if kind == "search" && data.get("queries").and_then(Value::as_array).is_none() {
                continue;
            }
            results.insert(id.to_string(), data.clone());
        }
    }
}
