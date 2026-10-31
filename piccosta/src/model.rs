// Models: the catalog of what each provider offers, the user's models.json on top of it, and
// turning a name like "openai-codex/gpt-6-luna" into one model.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/types.ts (the Model shape) and the
// merge rules of packages/coding-agent/src/core/provider-composer.ts and model-config.ts.
//
// pi ships 42 provider catalogs and refreshes them from pi.dev in the background. piccosta needs
// three, so those are compiled in from catalog/<provider>.json and nothing is fetched at run time.
// To refresh one, fetch it and replace the file:
//   curl -s "https://pi.dev/api/models/providers/<provider>?types=chat" | jq . > catalog/<provider>.json
// (last fetched 2026-10-29).

use crate::json_number;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs;
use std::path::Path;

// The compiled-in catalogs: (provider id, JSON list of models).
const BUILT_IN_CATALOGS: [(&str, &str); 3] = [
    ("openai-codex", include_str!("../catalog/openai-codex.json")),
    ("opencode-go", include_str!("../catalog/opencode-go.json")),
    // The Claude models the bridge serves under its own provider name.
    ("anthropic", include_str!("../catalog/anthropic.json")),
];

// What a model declared in models.json gets when it says nothing else and no catalog model shares
// its id (provider-composer.ts:232-247).
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
const DEFAULT_MAX_TOKENS: u64 = 16_384;

// Dollars per million tokens. `tiers` hold higher prices that apply past an input size.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    #[serde(serialize_with = "json_number::serialize", default)]
    pub input: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub output: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub cache_read: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub cache_write: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<CostTier>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CostTier {
    pub input_tokens_above: u64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub input: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub output: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub cache_read: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub cache_write: f64,
}

// One chat model. `compat` holds the provider quirks the request builders read (developer role,
// strict mode, thinking format, ...); it stays JSON because each wire protocol reads its own keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    #[serde(default)]
    pub reasoning: bool,
    // "text", and "image" when it takes images.
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    // pi's thinking levels (off, minimal, low, medium, high, xhigh, max) to the provider's own
    // effort names. A null value marks a level the model doesn't support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<IndexMap<String, Option<String>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
}

impl Model {
    // A compat flag, false when the model doesn't set it.
    pub fn compat_flag(&self, key: &str) -> bool {
        self.compat.as_ref().and_then(|compat| compat.get(key)).and_then(Value::as_bool).unwrap_or(false)
    }

    // A compat string value, such as `thinkingFormat`.
    pub fn compat_str(&self, key: &str) -> Option<&str> {
        self.compat.as_ref().and_then(|compat| compat.get(key)).and_then(Value::as_str)
    }

    // The provider's effort name for a pi thinking level. Input: a level such as "medium".
    // Output: Some(name) when the model reasons and maps the level (a level absent from the map
    // passes through unchanged), None when the model doesn't reason or marks the level null.
    pub fn provider_thinking_level(&self, level: &str) -> Option<String> {
        if !self.reasoning {
            return None;
        }
        match self.thinking_level_map.as_ref().and_then(|map| map.get(level)) {
            Some(Some(name)) => Some(name.clone()),
            Some(None) => None,
            None => Some(level.to_string()),
        }
    }
}

// models.json: per-provider settings over the catalog, and models of the user's own
// (model-config.ts:19-87). Only the fields piccosta acts on are read; others are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelsFile {
    #[serde(default)]
    providers: IndexMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    pub base_url: Option<String>,
    // The key, or `$VAR`, or `!command` whose output is the key. See auth.rs.
    pub api_key: Option<String>,
    pub api: Option<String>,
    pub headers: Option<IndexMap<String, String>>,
    pub compat: Option<Map<String, Value>>,
    #[serde(default)]
    pub models: Vec<Map<String, Value>>,
    // Per-model field overrides, applied last.
    #[serde(default)]
    pub model_overrides: IndexMap<String, Map<String, Value>>,
}

// Every model piccosta knows, and each provider's models.json settings.
pub struct ModelRegistry {
    models: Vec<Model>,
    providers: IndexMap<String, ProviderConfig>,
}

// Merge `overlay`'s keys into `base`, one level deep for objects (so a compat override changes
// one flag without dropping the others), replacing everything else.
fn merge_object(base: &mut Map<String, Value>, overlay: &Map<String, Value>) {
    for (key, value) in overlay {
        match (base.get_mut(key), value) {
            (Some(Value::Object(existing)), Value::Object(update)) => {
                for (inner_key, inner_value) in update {
                    existing.insert(inner_key.clone(), inner_value.clone());
                }
            }
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

impl ModelRegistry {
    // The catalog with models.json on top. Input: the path of models.json, which may not exist.
    // Output: the registry, or an error naming the file when it exists but isn't valid. The file may
    // have // comments and a BOM, as pi allows (model-config.ts:140).
    pub fn load(models_json: &Path) -> Result<ModelRegistry, String> {
        let mut models_by_provider: IndexMap<String, Vec<Map<String, Value>>> = IndexMap::new();
        for (provider, json) in BUILT_IN_CATALOGS {
            let list: Vec<Map<String, Value>> =
                serde_json::from_str(json).map_err(|error| format!("built-in catalog {provider}: {error}"))?;
            models_by_provider.insert(provider.to_string(), list);
        }
        // Claude runs only through the bridge, under its own provider name.
        if let Some(anthropic) = models_by_provider.shift_remove("anthropic") {
            models_by_provider.insert(crate::provider::claude::PROVIDER.to_string(), crate::provider::claude::catalog_models(anthropic));
        }

        let file: ModelsFile = match fs::read_to_string(models_json) {
            Ok(text) => {
                let cleaned = strip_json_comments(text.trim_start_matches('\u{feff}'));
                serde_json::from_str(&cleaned)
                    .map_err(|error| format!("Invalid models.json: {error}\n\nFile: {}", models_json.display()))?
            }
            Err(_) => ModelsFile::default(),
        };

        for (provider, config) in &file.providers {
            let list = models_by_provider.entry(provider.clone()).or_default();
            // Provider-wide settings reach every catalog model of that provider.
            for model in list.iter_mut() {
                if let Some(base_url) = &config.base_url {
                    model.insert("baseUrl".into(), Value::String(base_url.clone()));
                }
                if let Some(compat) = &config.compat {
                    let mut overlay = Map::new();
                    overlay.insert("compat".into(), Value::Object(compat.clone()));
                    merge_object(model, &overlay);
                }
            }
            // The user's own models: replace a catalog model with the same id, field by field, or
            // join the list. A new model starts from the provider's settings and pi's defaults.
            for declared in &config.models {
                let Some(id) = declared.get("id").and_then(Value::as_str) else { continue };
                if let Some(existing) = list.iter_mut().find(|model| model.get("id").and_then(Value::as_str) == Some(id)) {
                    if let Some(api) = &config.api {
                        existing.insert("api".into(), Value::String(api.clone()));
                    }
                    merge_object(existing, declared);
                    continue;
                }
                let mut model = Map::new();
                model.insert("id".into(), Value::String(id.to_string()));
                model.insert("name".into(), Value::String(id.to_string()));
                model.insert("provider".into(), Value::String(provider.clone()));
                if let Some(api) = &config.api {
                    model.insert("api".into(), Value::String(api.clone()));
                }
                if let Some(base_url) = &config.base_url {
                    model.insert("baseUrl".into(), Value::String(base_url.clone()));
                }
                model.insert("input".into(), serde_json::json!(["text"]));
                model.insert("contextWindow".into(), DEFAULT_CONTEXT_WINDOW.into());
                model.insert("maxTokens".into(), DEFAULT_MAX_TOKENS.into());
                if let Some(compat) = &config.compat {
                    model.insert("compat".into(), Value::Object(compat.clone()));
                }
                merge_object(&mut model, declared);
                list.push(model);
            }
            for (id, overrides) in &config.model_overrides {
                if let Some(model) = list.iter_mut().find(|model| model.get("id").and_then(Value::as_str) == Some(id)) {
                    merge_object(model, overrides);
                }
            }
            if let Some(headers) = &config.headers {
                for model in list.iter_mut() {
                    let mut merged = serde_json::to_value(headers).unwrap_or_default();
                    if let (Some(merged), Some(own)) = (merged.as_object_mut(), model.get("headers").and_then(Value::as_object)) {
                        merged.extend(own.clone());
                    }
                    model.insert("headers".into(), merged);
                }
            }
        }

        let mut models = Vec::new();
        for (provider, list) in models_by_provider {
            for mut raw in list {
                raw.entry("provider").or_insert(Value::String(provider.clone()));
                let id = raw.get("id").and_then(Value::as_str).unwrap_or("?").to_string();
                let model: Model = serde_json::from_value(Value::Object(raw))
                    .map_err(|error| format!("model {provider}/{id}: {error}"))?;
                models.push(model);
            }
        }
        Ok(ModelRegistry { models, providers: file.providers })
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    // A provider's models.json settings, if it has any.
    pub fn provider_config(&self, provider: &str) -> Option<&ProviderConfig> {
        self.providers.get(provider)
    }

    // Find one model by name. Input: "provider/id", or a bare id. Output: the model, or an error
    // listing what matched when a bare id is ambiguous or nothing matches. Matching is exact and
    // case-insensitive; pi's substring and fuzzy matching are left out, since a near-miss model name
    // should fail loudly rather than pick something else.
    pub fn resolve(&self, name: &str) -> Result<&Model, String> {
        if let Some((provider, id)) = name.split_once('/')
            && let Some(model) = self
                .models
                .iter()
                .find(|model| model.provider.eq_ignore_ascii_case(provider) && model.id.eq_ignore_ascii_case(id))
            {
                return Ok(model);
            }
        let matches: Vec<&Model> = self.models.iter().filter(|model| model.id.eq_ignore_ascii_case(name)).collect();
        match matches.as_slice() {
            [one] => Ok(one),
            [] => Err(format!("Model \"{name}\" not found. Use provider/id, e.g. openai-codex/gpt-6-luna.")),
            several => Err(format!(
                "Model \"{name}\" is offered by several providers; say which: {}",
                several.iter().map(|model| format!("{}/{}", model.provider, model.id)).collect::<Vec<_>>().join(", ")
            )),
        }
    }
}

// Remove // line comments and /* */ block comments from JSON text, leaving strings alone. Input:
// JSON with comments. Output: plain JSON.
fn strip_json_comments(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    let mut in_string = false;
    while let Some(character) = characters.next() {
        if in_string {
            output.push(character);
            if character == '\\' {
                if let Some(escaped) = characters.next() {
                    output.push(escaped);
                }
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match (character, characters.peek()) {
            ('"', _) => {
                in_string = true;
                output.push(character);
            }
            ('/', Some('/')) => {
                for skipped in characters.by_ref() {
                    if skipped == '\n' {
                        output.push('\n');
                        break;
                    }
                }
            }
            ('/', Some('*')) => {
                characters.next();
                let mut previous = '\0';
                for skipped in characters.by_ref() {
                    if previous == '*' && skipped == '/' {
                        break;
                    }
                    previous = skipped;
                }
            }
            _ => output.push(character),
        }
    }
    output
}
