// Adding a server uses the same probe and file edits in the terminal and on the command line.
// We keep the user's JSON values, including overrides we don't know about, under an OS lock.
use crate::{auth, config, model};
use serde_json::{json, Value};
use std::{fs, io::{self, IsTerminal, Write}, path::Path};

pub const APIS: [&str; 4] = ["openai-completions", "openai-responses", "anthropic-messages", "google-generative-ai"];

#[derive(Default)]
pub struct Addition {
    pub base: String,
    pub name: String,
    pub api: String,
    pub key: String,
    pub literal: bool,
    pub models: Vec<String>,
    pub manual: bool,
}

pub enum Probe {
    Ready,
    KeyNeeded,
}

impl Addition {
    // A URL without credentials or a query: those belong in the credential store, not models.json.
    pub fn new(base: &str) -> Result<Self, String> {
        let url = url::Url::parse(base.trim()).map_err(|_| "Enter an http or https base URL")?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() || !url.username().is_empty() || url.password().is_some() || url.query().is_some() || url.fragment().is_some() {
            return Err("Use an http or https base URL without credentials, a query or a fragment".into());
        }
        let name: String = url.host_str().unwrap_or_default().chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '-' { ch } else { '-' }).collect();
        Ok(Self { base: url.as_str().trim_end_matches('/').into(), name, api: APIS[0].into(), ..Self::default() })
    }

    // Probe only the given origin. No response body or credential is included in an error.
    pub async fn probe(&mut self, agent_dir: &Path) -> Result<Probe, String> {
        let settings = config::Settings::load(agent_dir, None);
        let client = model::list_client(&settings).redirect(reqwest::redirect::Policy::none()).build().map_err(|_| "Couldn't create the model-list client")?;
        let key = if self.literal { self.key.clone() } else { auth::resolve_configured_key(&self.key).await? };
        let mut bases = vec![self.base.clone()];
        if !url::Url::parse(&self.base).map_err(|_| "Invalid base URL")?.path().split('/').any(|part| part == "v1") {
            bases.push(format!("{}/v1", self.base));
        }
        let (mut needs_key, mut failure) = (false, None);
        for base in bases {
            let mut request = client.get(format!("{base}/models"));
            if !key.is_empty() {
                request = request.bearer_auth(&key).header("x-api-key", &key).header("x-goog-api-key", &key).header("anthropic-version", crate::provider::anthropic_messages::ANTHROPIC_VERSION);
            }
            let response = request.send().await.map_err(|_| "Couldn't reach the server's model list")?;
            if matches!(response.status().as_u16(), 401 | 403) {
                needs_key = true;
                continue;
            }
            if response.status() == reqwest::StatusCode::NOT_FOUND || response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED {
                continue;
            }
            if !response.status().is_success() {
                failure = Some(format!("Model-list probe returned HTTP {}", response.status().as_u16()));
                continue;
            }
            let body = response.json::<Value>().await.unwrap_or(Value::Null);
            if let Some(list) = model::model_list(&body) {
                self.models = list.iter().filter_map(model::listed_id).map(str::to_string).collect();
                self.models.sort();
                self.models.dedup();
                if body["models"].is_array() {
                    self.api = "google-generative-ai".into();
                } else if list.iter().any(|row| row["type"] == "model" && row.get("display_name").is_some()) {
                    self.api = "anthropic-messages".into();
                }
                self.base = base;
                self.manual = self.models.is_empty();
                return Ok(Probe::Ready);
            }
        }
        if needs_key {
            return Ok(Probe::KeyNeeded);
        }
        if let Some(error) = failure {
            return Err(error);
        }
        self.manual = true;
        Ok(Probe::Ready)
    }

    // A new provider only: overwriting an account or an existing provider needs an explicit remove.
    pub async fn save(&self, agent_dir: &Path) -> Result<String, String> {
        valid_name(&self.name)?;
        if !APIS.contains(&self.api.as_str()) {
            return Err(format!("API must be one of {}", APIS.join(", ")));
        }
        if self.models.is_empty() || self.models.iter().any(|id| id.trim().is_empty()) {
            return Err("Enter at least one model id".into());
        }
        let path = agent_dir.join("models.json");
        let _lock = lock_models(&path).await?;
        let mut file = read_models(&path)?;
        let providers = file.entry("providers").or_insert_with(|| json!({})).as_object_mut().ok_or("models.json providers is not an object")?;
        if providers.contains_key(&self.name) {
            return Err(format!("Provider {} already exists; remove it first or choose another name", self.name));
        }
        let base = if self.api == "anthropic-messages" { self.base.trim_end_matches("/v1") } else { &self.base };
        let mut provider = json!({"baseUrl": base, "api": self.api});
        if !self.key.is_empty() {
            if !self.literal && self.key.starts_with(['$', '!']) {
                provider["apiKey"] = self.key.clone().into();
            } else {
                edit_key(agent_dir, &self.name, Some((&self.key, base))).await?;
                provider["apiKey"] = "@stored".into();
            }
        }
        // Keep discovered lists refreshable; a server with no list gets its one explicit model.
        if self.manual {
            provider["models"] = self.models.iter().map(|id| json!({"id": id})).collect();
        }
        providers.insert(self.name.clone(), provider);
        auth::write_private_json(&path, &Value::Object(file)).map_err(|error| error.to_string())?;
        if !self.manual {
            // The probe is the same refresh without a second network trip. Make it pickable now.
            let config: model::ProviderConfig = serde_json::from_value(json!({"baseUrl": base, "api": self.api})).map_err(|error| error.to_string())?;
            model::keep_added_list(agent_dir, &self.name, &self.models, &config)?;
        }
        Ok(format!("Added {}: {} models ({})", self.name, self.models.len(), self.api))
    }
}

fn valid_name(name: &str) -> Result<(), String> {
    if name.is_empty() || !name.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')) || matches!(name, "." | "..") || auth::account(name).is_some() {
        return Err("Use a new provider name with letters, numbers, dots, hyphens or underscores".into());
    }
    Ok(())
}

fn read_models(path: &Path) -> Result<serde_json::Map<String, Value>, String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Default::default()),
        Err(error) => return Err(error.to_string()),
    };
    serde_json::from_str::<Value>(&model::strip_json_comments(text.trim_start_matches('\u{feff}'))).map_err(|error| error.to_string())?.as_object().cloned().ok_or_else(|| "models.json is not an object".into())
}

// Poll without spinning, noticing a released lock within 50 ms. A minute allows a paused writer
// to resume, but isn't permission to hang forever; cancellation drops the file immediately.
const MODELS_LOCK_RETRY: std::time::Duration = std::time::Duration::from_millis(50);
const MODELS_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

// The lock file stays put: renaming models.json must not change which inode writers lock.
async fn lock_models(path: &Path) -> Result<fs::File, String> {
    let path = path.with_extension("json.lock");
    if let Some(dir) = path.parent() { crate::platform::private_dir(dir).map_err(|error| error.to_string())?; }
    let file = crate::platform::private_file(fs::OpenOptions::new().read(true).write(true).create(true).truncate(false)).open(path).map_err(|error| error.to_string())?;
    tokio::time::timeout(MODELS_LOCK_TIMEOUT, async {
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(file),
                Err(fs::TryLockError::WouldBlock) => tokio::time::sleep(MODELS_LOCK_RETRY).await,
                Err(fs::TryLockError::Error(error)) => return Err(error.to_string()),
            }
        }
    }).await.map_err(|_| "Timed out waiting for models.json lock".to_string())?
}

async fn edit_key(agent_dir: &Path, name: &str, key: Option<(&str, &str)>) -> Result<(), String> {
    let path = auth::CredentialPaths::from_environment(agent_dir).login_dir.join("auth.json");
    let _lock = auth::FileLock::acquire(&path).await.map_err(|error| error.to_string())?;
    let mut file = config::read_json_object(&path)?;
    if let Some((key, base)) = key {
        let origin = url::Url::parse(base).map_err(|_| "Invalid base URL")?.origin().ascii_serialization();
        file.insert(name.into(), json!({"type": "api_key", "key": key, "literal": true, "origin": origin}));
    } else if file.remove(name).is_none() {
        return Ok(());
    }
    auth::write_private_json(&path, &Value::Object(file)).map_err(|error| error.to_string())
}

pub async fn remove(agent_dir: &Path, name: &str) -> Result<String, String> {
    valid_name(name)?;
    let path = agent_dir.join("models.json");
    let _lock = lock_models(&path).await?;
    let mut file = read_models(&path)?;
    let removed = file.get_mut("providers").and_then(Value::as_object_mut).and_then(|providers| providers.remove(name)).ok_or_else(|| format!("No configured provider {name}"))?;
    auth::write_private_json(&path, &Value::Object(file)).map_err(|error| error.to_string())?;
    if removed["apiKey"] == "@stored" {
        edit_key(agent_dir, name, None).await?;
    }
    let copy = agent_dir.join(model::LISTS_DIR).join(format!("{name}.json"));
    match fs::remove_file(copy) {
        Ok(()) => {},
        Err(error) if error.kind() == io::ErrorKind::NotFound => {},
        Err(error) => return Err(error.to_string()),
    }
    Ok(format!("Removed {name}"))
}

// Ordinary CLI answers use stdin; pasted secrets use raw terminal input, never echo or history.
fn ask(label: &str, secret: bool) -> Result<String, String> {
    if !io::stdin().is_terminal() {
        return Err(format!("{label}: use the corresponding flag, or run in a terminal"));
    }
    eprint!("{label}: ");
    io::stderr().flush().map_err(|error| error.to_string())?;
    if !secret {
        let mut line = String::new();
        io::stdin().read_line(&mut line).map_err(|error| error.to_string())?;
        return Ok(line.trim().into());
    }
    crossterm::terminal::enable_raw_mode().map_err(|error| error.to_string())?;
    let result = (|| {
        use crossterm::event::{read, Event, KeyCode, KeyModifiers};
        let mut key = String::new();
        loop {
            match read().map_err(|error| error.to_string())? {
                Event::Key(event) => match event.code {
                    KeyCode::Enter => return Ok(key),
                    KeyCode::Esc => return Err("Cancelled".into()),
                    KeyCode::Char('c') if event.modifiers.contains(KeyModifiers::CONTROL) => return Err("Cancelled".into()),
                    KeyCode::Backspace => { key.pop(); },
                    KeyCode::Char(ch) => key.push(ch),
                    _ => {},
                },
                Event::Paste(text) => key.push_str(text.trim()),
                _ => {},
            }
        }
    })();
    let _ = crossterm::terminal::disable_raw_mode();
    eprintln!();
    result
}

pub async fn cli(arguments: &[String]) -> Result<(), String> {
    const USAGE: &str = "Usage: piccosta models add <url> [--name n] [--api a] [--key-env VAR | --key-command CMD | --key-stdin], or piccosta models remove <name>";
    let agent_dir = config::agent_dir();
    if arguments.first().is_some_and(|arg| arg == "remove") && arguments.len() == 2 {
        println!("{}", remove(&agent_dir, &arguments[1]).await?);
        return Ok(());
    }
    if arguments.first().is_none_or(|arg| arg != "add") || arguments.len() < 2 {
        return Err(USAGE.into());
    }
    let mut addition = Addition::new(&arguments[1])?;
    let (mut api, mut key_stdin, mut key_given) = (None, false, false);
    let mut flags = arguments[2..].iter();
    while let Some(flag) = flags.next() {
        if flag == "--key-stdin" {
            if key_given { return Err("Choose only one key source".into()); }
            key_stdin = true;
            key_given = true;
            continue;
        }
        let value = flags.next().ok_or(USAGE)?;
        match flag.as_str() {
            "--name" => addition.name = value.clone(),
            "--api" if APIS.contains(&value.as_str()) => api = Some(value.clone()),
            "--key-env" | "--key-command" => {
                if key_given || value.trim().is_empty() { return Err("Choose one nonempty key source".into()); }
                key_given = true;
                addition.key = format!("{}{value}", if flag == "--key-env" { "$" } else { "!" });
            }
            _ => return Err(USAGE.into()),
        }
    }
    if key_stdin {
        if io::stdin().is_terminal() { return Err("--key-stdin requires piped input, so the key is never echoed".into()); }
        io::stdin().read_line(&mut addition.key).map_err(|error| error.to_string())?;
        addition.key = addition.key.trim().into();
        addition.literal = true;
        if addition.key.is_empty() { return Err("The key on stdin is empty".into()); }
    }
    if matches!(addition.probe(&agent_dir).await?, Probe::KeyNeeded) {
        if key_given { return Err("The server rejected the key (401/403)".into()); }
        addition.key = ask("Key (hidden; paste, $VAR or !command)", true)?;
        if addition.key.is_empty() || matches!(addition.probe(&agent_dir).await?, Probe::KeyNeeded) {
            return Err("The server rejected the key (401/403)".into());
        }
    }
    if let Some(api) = api { addition.api = api; }
    if addition.manual {
        addition.models = vec![ask("No model list; model id", false)?];
    }
    println!("Models: {}", addition.models.join(", "));
    println!("{}", addition.save(&agent_dir).await?);
    Ok(())
}
