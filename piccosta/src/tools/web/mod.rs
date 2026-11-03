// The web tools: web_search, fetch_content, and get_search_content for paging through what they
// stored.
//
// Based on pi-web-access 0.44.0 (MIT, see LICENSE.md): the tool definitions in
// index.ts and their execution in web-tool-core.ts. What the model reads is pi-web-access's word
// for word: the output layout, the responseId and continuation lines, every error message. What
// it can ask for is trimmed to what piccosta does: no provider choice, curator workflow, proxy,
// video frames, cookie auth, answer mode or redirect policy (and their parameters and sentences
// are gone from the descriptions). One behaviour changed: includeContent fetches the pages before
// web_search returns, as pi-web-access does in hosts without background notices, instead of
// fetching in the background and sending a message later.

pub mod extract;
pub mod find;
pub mod github;
pub mod net;
pub mod openai_search;
pub mod search;
pub mod store;

use super::{Tool, ToolContext, ToolFuture, ToolOutput, ToolUpdate};
use crate::config::WebSettings;
use crate::message::{Content, ToolDefinition};
use extract::{Extracted, Mode};
use find::FindMode;
use serde_json::{json, Map, Value};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;

// URLs fetched at once across the session (extract.ts:34), and queries searched at once per call
// (web-tool-core.ts:142).
const FETCH_CONCURRENCY: usize = 3;
const SEARCH_QUERY_CONCURRENCY: usize = 3;
// A fetch page's end moves back to a line break when one is this close (initialContentSlice).
const LINE_BREAK_FRACTION: f64 = 0.8;
const STORED_CONTENT_SOURCES: &str = "web_search or fetch_content";

// What every web tool call in a session shares.
pub struct WebShared {
    pub settings: WebSettings,
    client: OnceLock<Result<net::WebClient, String>>,
    pub fetch_slots: Semaphore,
    pub store: store::Store,
    pub clones: github::Clones,
    // The models web_search may ask when no SearXNG is configured, or why there are none.
    pub search_targets: Result<Vec<openai_search::Target>, String>,
    pub credentials: crate::auth::CredentialPaths,
}

impl WebShared {
    // Input: the web settings, the agent dir, the model registry the search model comes from, and
    // where credentials live.
    pub fn new(settings: WebSettings, agent_dir: &Path, registry: &crate::model::ModelRegistry, credentials: crate::auth::CredentialPaths) -> WebShared {
        let clones = github::Clones::new(agent_dir, settings.github_clone_path.as_deref());
        let search_targets = openai_search::targets(registry, settings.search_model.as_deref(), agent_dir);
        WebShared { settings, client: OnceLock::new(), fetch_slots: Semaphore::new(FETCH_CONCURRENCY), store: store::Store::new(agent_dir), clones, search_targets, credentials }
    }

    // When piccosta exits: this process's GitHub clones go.
    pub fn shutdown(&self) {
        self.clones.shutdown();
    }

    pub fn client(&self) -> Result<&net::WebClient, String> {
        self.client.get_or_init(|| net::WebClient::new(&self.settings)).as_ref().map_err(Clone::clone)
    }
}

pub fn tools() -> Vec<Arc<dyn Tool>> {
    vec![Arc::new(WebSearchTool), Arc::new(FetchContentTool), Arc::new(GetSearchContentTool)]
}

fn text_output(text: String, details: Value, is_error: bool) -> ToolOutput {
    ToolOutput { content: vec![Content::text(text)], details: Some(details), is_error }
}

fn send_update(context: &ToolContext, text: String, details: Value) {
    if let Some(updates) = &context.updates {
        let _ = updates.send(ToolUpdate { content: vec![Content::text(text)], details: Some(details) });
    }
}

// How an argument is quoted back in a message (web-tool-core.ts formatInputValue).
fn format_input_value(value: &Value) -> String {
    crate::js::stringify(value)
}

fn utf16(text: &str) -> Vec<u16> {
    text.encode_utf16().collect()
}

fn from_utf16(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

// A page end that doesn't split a surrogate pair: back one unit, or past the pair when backing
// up would leave the page empty (web-tool-core.ts pageEnd).
fn page_end(content: &[u16], offset: usize, end: usize) -> usize {
    if end <= offset || end >= content.len() {
        return end;
    }
    let (high, low) = (content[end - 1], content[end]);
    if !(0xd800..=0xdbff).contains(&high) || !(0xdc00..=0xdfff).contains(&low) {
        return end;
    }
    if end - 1 > offset { end - 1 } else { end + 1 }
}

// A slice from `offset` with the next-slice line appended, shrunk until both fit in `max_chars`
// (web-tool-core.ts sliceWithContinuation).
fn slice_with_continuation(content: &[u16], offset: usize, limit: usize, max_chars: usize, continuation_at: impl Fn(usize) -> String) -> (usize, String) {
    let mut end = page_end(content, offset, offset + limit.min(content.len() - offset));
    let mut continuation = String::new();
    while end < content.len() {
        continuation = continuation_at(end);
        let overflow = (end - offset + crate::js::utf16_len(&continuation)) as isize - max_chars as isize;
        if overflow <= 0 {
            break;
        }
        end = page_end(content, offset, end.saturating_sub(overflow as usize));
    }
    let text = from_utf16(&content[offset..end]) + if end < content.len() { continuation.as_str() } else { "" };
    (end, text)
}

fn line_count(text: &str) -> usize {
    if text.is_empty() { 0 } else { text.split('\n').count() }
}

// ---------------------------------------------------------------------------------------------
// web_search

pub struct WebSearchTool;

struct QueryData {
    query: String,
    answer: String,
    results: Vec<search::SearchResult>,
    error: Option<String>,
    provider: Option<&'static str>,
}

impl QueryData {
    fn providers(&self) -> Vec<&'static str> {
        self.provider.into_iter().collect()
    }

    fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert("query".into(), json!(self.query));
        object.insert("answer".into(), json!(self.answer));
        object.insert("results".into(), Value::Array(self.results.iter().map(|r| json!({"title": r.title, "url": r.url, "snippet": r.snippet})).collect()));
        object.insert("error".into(), self.error.clone().map(Value::from).unwrap_or(Value::Null));
        if let Some(provider) = self.provider {
            object.insert("provider".into(), json!(provider));
            object.insert("providers".into(), json!(self.providers()));
        }
        Value::Object(object)
    }
}

// A string that is a JSON array of strings searches each one; models sometimes put the list in
// `query` (web-tool-core.ts expandQueryString).
fn expand_query_string(query: &str) -> Vec<String> {
    let trimmed = crate::js::trim(query);
    if trimmed.starts_with('[') && trimmed.ends_with(']')
        && let Ok(Value::Array(items)) = serde_json::from_str::<Value>(trimmed)
        && items.iter().all(Value::is_string)
    {
        return items.iter().filter_map(Value::as_str).map(|item| crate::js::trim(item).to_string()).filter(|item| !item.is_empty()).collect();
    }
    vec![query.to_string()]
}

fn format_search_summary(results: &[search::SearchResult], answer: &str) -> String {
    if results.is_empty() {
        return if answer.is_empty() { "No results found.".to_string() } else { format!("{answer}\n\n---\n\n**Sources:**\nNo sources returned.") };
    }
    let mut output = if answer.is_empty() { String::new() } else { format!("{answer}\n\n---\n\n**Sources:**\n") };
    output += &results.iter().enumerate().map(|(i, r)| format!("{}. {}\n   {}", i + 1, r.title, r.url)).collect::<Vec<_>>().join("\n\n");
    output
}

// Keep the search text and its guidance under the output cap, cutting the text and saying so
// (web-tool-core.ts boundSearchPresentation). Output: (text, truncated, original, returned, omitted).
fn bound_search_presentation(text: &str, guidance: &str, truncation_guidance: &str, max_chars: usize) -> (String, bool, usize, usize, usize) {
    let text_length = crate::js::utf16_len(text);
    let full = format!("{text}{guidance}");
    if crate::js::utf16_len(&full) <= max_chars {
        return (full, false, text_length, text_length, 0);
    }
    let marker = format!("\n\n---\n[Output truncated.]{truncation_guidance}");
    let prefix_length = max_chars.saturating_sub(crate::js::utf16_len(&marker));
    let units = utf16(text);
    let bounded = format!("{}{marker}", from_utf16(&units[..prefix_length.min(units.len())]));
    (bounded, true, text_length, prefix_length, text_length.saturating_sub(prefix_length))
}

impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "web_search".into(),
            description: "Search the web and return source-linked results. Full results are stored for retrieval with get_search_content using the returned responseId. For research, prefer queries with 2-4 varied angles over a single query.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Single search query. For research tasks, prefer 'queries' with multiple varied angles instead."},
                    "queries": {"type": "array", "items": {"type": "string"}, "description": "Multiple queries searched concurrently (up to three at a time), each returning source-linked search results. Prefer this for research \u{2014} vary phrasing, scope, and angle across 2-4 queries to maximize coverage. Good: ['React vs Vue performance benchmarks 2026', 'React vs Vue developer experience comparison', 'React ecosystem size vs Vue ecosystem']. Bad: ['React vs Vue', 'React vs Vue comparison', 'React vs Vue review'] (too similar, redundant results)."},
                    "numResults": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Results per query (default: 5, max: 20)"},
                    "includeContent": {"type": "boolean", "description": "Also fetch each result's full page content"},
                    "recencyFilter": {"type": "string", "enum": ["day", "week", "month", "year"], "description": "Filter by recency"},
                    "domainFilter": {"type": "array", "items": {"type": "string"}, "description": "Limit to domains (prefix with - to exclude)"}
                }
            }),
            constrained_sampling: None,
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("search the web; two to four differently worded `queries` find more than one")
    }

    // A list sent as a JSON string becomes the list (tool-arguments.ts parseStringifiedArrays).
    fn prepare_arguments(&self, mut arguments: Map<String, Value>) -> Map<String, Value> {
        for key in ["queries", "domainFilter"] {
            if let Some(Value::String(text)) = arguments.get(key)
                && crate::js::trim(text).starts_with('[')
                && let Ok(Value::Array(items)) = serde_json::from_str::<Value>(text)
                && items.iter().all(Value::is_string)
            {
                arguments.insert(key.to_string(), Value::Array(items));
            }
        }
        arguments
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let web = &context.shared.web;
            let raw: Vec<String> = match arguments.get("queries") {
                Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).map(str::to_string).collect(),
                _ => arguments.get("query").and_then(Value::as_str).map(expand_query_string).unwrap_or_default(),
            };
            let query_list: Vec<String> = raw.iter().map(|query| crate::js::trim(query).to_string()).filter(|query| !query.is_empty()).collect();
            if query_list.is_empty() {
                return Ok(text_output("Error: No query provided. Use 'query' or 'queries' parameter.".into(), json!({"error": "No query provided"}), false));
            }
            let recency = arguments.get("recencyFilter").and_then(Value::as_str).filter(|r| matches!(*r, "day" | "week" | "month" | "year"));
            let domain_filter: Vec<String> = arguments.get("domainFilter").and_then(Value::as_array).map(|items| items.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
            let include_content = arguments.get("includeContent").and_then(Value::as_bool).unwrap_or(false);
            let options = search::SearchOptions { num_results: arguments.get("numResults").and_then(Value::as_f64), recency, domain_filter: &domain_filter };

            let completed = std::sync::atomic::AtomicUsize::new(0);
            let slots = Semaphore::new(SEARCH_QUERY_CONCURRENCY);
            let total = query_list.len();
            let jobs = query_list.iter().map(|query| {
                let (slots, completed, options) = (&slots, &completed, &options);
                async move {
                    let _slot = slots.acquire().await;
                    let done = completed.load(std::sync::atomic::Ordering::SeqCst);
                    send_update(context, format!("Searching \"{query}\" ({done}/{total} complete)..."), json!({"phase": "search", "progress": done as f64 / total as f64, "currentQuery": query}));
                    let found = tokio::select! {
                        found = search::search(web.settings.searxng_url.as_deref(), &web.search_targets, &web.credentials, &context.session_id, query, options) => found,
                        () = context.cancel.cancelled() => Err("Aborted".to_string()),
                    };
                    let done = completed.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    send_update(context, format!("Completed {done}/{total} searches."), json!({"phase": "search", "progress": done as f64 / total as f64, "currentQuery": query}));
                    match found {
                        Ok(response) => QueryData { query: query.clone(), answer: response.answer, results: response.results, error: None, provider: Some(response.provider) },
                        Err(error) => QueryData { query: query.clone(), answer: String::new(), results: Vec::new(), error: Some(error), provider: None },
                    }
                }
            });
            let results: Vec<QueryData> = futures_util::future::join_all(jobs).await;
            if context.cancel.is_cancelled() {
                return Err("Operation aborted".to_string());
            }
            let mut all_urls: Vec<String> = Vec::new();
            for result in &results {
                for found in &result.results {
                    if !all_urls.contains(&found.url) {
                        all_urls.push(found.url.clone());
                    }
                }
            }
            let inline_content = if include_content && !all_urls.is_empty() { Some(extract::fetch_all(&all_urls, Mode::Readable, false, web, context.image_limits, &context.cancel).await) } else { None };
            if context.cancel.is_cancelled() {
                return Err("Operation aborted".to_string());
            }
            Ok(build_search_return(web, &query_list, &results, inline_content, include_content))
        })
    }
}

// build_search_return from stored-search JSON (pi-web-access's QueryResultData shape), for
// checking the output against pi-web-access's on the same results.
pub fn build_search_return_json(web: &WebShared, query_list: &[String], results: &Value, inline_content: Option<Vec<Extracted>>, include_content: bool) -> ToolOutput {
    let results: Vec<QueryData> = results
        .as_array()
        .into_iter()
        .flatten()
        .map(|r| QueryData {
            query: r["query"].as_str().unwrap_or("").to_string(),
            answer: r["answer"].as_str().unwrap_or("").to_string(),
            results: r["results"].as_array().into_iter().flatten().map(|x| search::SearchResult { title: x["title"].as_str().unwrap_or("").into(), url: x["url"].as_str().unwrap_or("").into(), snippet: x["snippet"].as_str().unwrap_or("").into() }).collect(),
            error: r["error"].as_str().map(str::to_string),
            provider: match r["provider"].as_str() { Some("searxng") => Some("searxng"), Some("openai") => Some("openai"), _ => None },
        })
        .collect();
    build_search_return(web, query_list, &results, inline_content, include_content)
}

// The search's text, its stored ids and its details (web-tool-core.ts buildSearchReturn).
fn build_search_return(web: &WebShared, query_list: &[String], results: &[QueryData], inline_content: Option<Vec<Extracted>>, include_content: bool) -> ToolOutput {
    let max_chars = web.settings.max_inline_content_chars;
    let successful = results.iter().filter(|r| r.error.is_none()).count();
    let total_results: usize = results.iter().map(|r| r.results.len()).sum();
    let provider_names: Vec<String> = results.iter().map(|r| { let joined = r.providers().join(", "); if joined.is_empty() { "unknown".to_string() } else { joined } }).collect();
    let mut output = if results.len() == 1 {
        format!("**Provider:** {}\n\n", provider_names[0])
    } else {
        format!("**Providers used:** {}\n\n", provider_names.iter().enumerate().map(|(i, name)| format!("Query {}: {name}", i + 1)).collect::<Vec<_>>().join("; "))
    };
    for result in results {
        if query_list.len() > 1 {
            output += &format!("## Query: \"{}\"\n\n", result.query);
        }
        match &result.error {
            Some(error) => output += &format!("Error: {error}\n\n"),
            None => output += &(format_search_summary(&result.results, &result.answer) + "\n\n"),
        }
    }

    let fetch_id = inline_content.as_ref().filter(|content| !content.is_empty()).map(|content| {
        let id = store::generate_id();
        let stripped: Vec<Extracted> = content.iter().map(|page| Extracted { image: None, ..page.clone() }).collect();
        web.store.store_fetch(&id, &stripped);
        id
    });
    let search_id = store::generate_id();
    web.store.store_search(&search_id, Value::Array(results.iter().map(QueryData::to_json).collect()));

    let guidance = |_for_truncation: bool| {
        let mut value = String::new();
        if let (Some(id), Some(content)) = (&fetch_id, &inline_content) {
            value += &format!("\n---\nFull content for {} sources is ready as responseId \"{id}\". ", content.len());
            value += &format!("Use get_search_content({{ responseId: \"{id}\", urlIndex: 0, offset: 0, limit: {max_chars} }}) to retrieve the first bounded page.");
        }
        value += &format!("\n---\nFull search results are stored as responseId \"{search_id}\". ");
        let repeat = if results.len() > 1 { format!("; repeat with queryIndex 1 through {}", results.len() - 1) } else { String::new() };
        value += &format!("Use get_search_content({{ responseId: \"{search_id}\", queryIndex: 0, offset: 0, limit: {max_chars} }}) to retrieve the first bounded page{repeat}.");
        value
    };
    let trimmed = crate::js::trim(&output).to_string();
    let (text, truncated, original, returned, omitted) = bound_search_presentation(&trimmed, &guidance(false), &guidance(true), max_chars);
    let details = json!({
        "queries": query_list,
        "queryCount": query_list.len(),
        "successfulQueries": successful,
        "totalResults": total_results,
        "includeContent": include_content,
        "fetchId": fetch_id,
        "searchId": search_id,
        "queryProviders": results.iter().map(|r| json!({"query": r.query, "providers": r.providers()})).collect::<Vec<_>>(),
        "truncated": truncated,
        "originalChars": original,
        "returnedChars": returned,
        "omittedChars": omitted,
    });
    text_output(text, details, false)
}

// ---------------------------------------------------------------------------------------------
// fetch_content

pub struct FetchContentTool;

const FETCH_MODES: &str = "readable (default): extract readable content as markdown; raw: return the exact textual body using direct HTTP only";

impl Tool for FetchContentTool {
    fn name(&self) -> &'static str {
        "fetch_content"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_content".into(),
            description: format!("Fetch URL(s). Available modes: {FETCH_MODES}. Direct image URLs return resized image content. Supports GitHub repositories and PDFs. Full original content is stored for retrieval with get_search_content."),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Single URL to fetch"},
                    "urls": {"type": "array", "items": {"type": "string"}, "description": "Multiple URLs (parallel)"},
                    "forceClone": {"type": "boolean", "description": "Force cloning large GitHub repositories that exceed the size threshold"},
                    "mode": {"type": "string", "enum": ["readable", "raw"], "description": format!("Fetch mode. {FETCH_MODES}.")}
                }
            }),
            constrained_sampling: None,
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("get what a URL holds: a page, an image, a GitHub repo or a PDF")
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let web = &context.shared.web;
            // urls wins when it has any; each trimmed, empties dropped, repeats dropped
            // (fetch-params.ts normalizeFetchContentParams).
            let mut url_list: Vec<String> = Vec::new();
            for url in arguments.get("urls").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                let trimmed = crate::js::trim(url).to_string();
                if !trimmed.is_empty() && !url_list.contains(&trimmed) {
                    url_list.push(trimmed);
                }
            }
            if url_list.is_empty()
                && let Some(url) = arguments.get("url").and_then(Value::as_str).map(crate::js::trim).filter(|url| !url.is_empty())
            {
                url_list.push(url.to_string());
            }
            let mode = if arguments.get("mode").and_then(Value::as_str) == Some("raw") { Mode::Raw } else { Mode::Readable };
            let mode_name = if mode == Mode::Raw { "raw" } else { "readable" };
            let force_clone = arguments.get("forceClone").and_then(Value::as_bool).unwrap_or(false);
            if mode == Mode::Raw && force_clone {
                return Ok(text_output("Error: mode raw cannot be combined with forceClone.".into(), json!({"error": "Incompatible raw mode options"}), false));
            }
            if url_list.is_empty() {
                return Ok(text_output("Error: No URL provided. Use the 'url' parameter, or 'urls' for parallel fetches.".into(), json!({"error": "No URL provided"}), false));
            }
            send_update(context, format!("Fetching {} URL(s)...", url_list.len()), json!({"phase": "fetch", "progress": 0}));
            let results = extract::fetch_all(&url_list, mode, force_clone, web, context.image_limits, &context.cancel).await;
            if context.cancel.is_cancelled() {
                return Err("Operation aborted".to_string());
            }
            let successful = results.iter().filter(|r| r.error.is_none()).count();
            let total_chars: usize = results.iter().map(|r| crate::js::utf16_len(&r.content)).sum();
            let response_id = store::generate_id();
            let stripped: Vec<Extracted> = results.iter().map(|page| Extracted { image: None, ..page.clone() }).collect();
            web.store.store_fetch(&response_id, &stripped);

            if url_list.len() == 1 {
                let result = &results[0];
                if let Some(error) = &result.error {
                    return Ok(text_output(format!("Error: {error}"), json!({"urls": url_list, "urlCount": 1, "successful": 0, "error": error, "responseId": response_id}), true));
                }
                Ok(single_fetch_output(result, &url_list, &response_id, mode_name, web.settings.max_inline_content_chars))
            } else {
                let mut output = "## Fetched URLs\n\n".to_string();
                for result in &results {
                    output += &match &result.error {
                        Some(error) => format!("- {}: Error - {error}\n", result.url),
                        None => format!("- {} ({} chars)\n", if result.title.is_empty() { &result.url } else { &result.title }, crate::js::utf16_len(&result.content)),
                    };
                }
                output += &format!("\n---\nUse get_search_content({{ responseId: \"{response_id}\", urlIndex: 0 }}) to retrieve bounded content slices.");
                let details = json!({"urls": url_list, "urlCount": url_list.len(), "successful": successful, "totalChars": total_chars, "responseId": response_id});
                Ok(text_output(output, details, successful == 0))
            }
        })
    }
}

// One fetched URL shown inline up to the cap, cut at a line break when one is near the end,
// with where to read on (web-tool-core.ts:1027-1105).
fn single_fetch_output(result: &Extracted, url_list: &[String], response_id: &str, mode: &str, max_chars: usize) -> ToolOutput {
    let content = utf16(&result.content);
    let full_length = content.len();
    let mut end = page_end(&content, 0, full_length.min(max_chars));
    if end < full_length {
        let newline = u16::from(b'\n');
        let search_to = end.min(full_length - 1);
        if let Some(line_break) = (0..=search_to).rev().find(|at| content[*at] == newline)
            && line_break >= (max_chars as f64 * LINE_BREAK_FRACTION).floor() as usize
        {
            end = line_break + 1;
        }
    }
    let shown = from_utf16(&content[..end]);
    let truncated = end < full_length;
    let (total_bytes, total_lines, shown_bytes, shown_lines) = (result.content.len(), line_count(&result.content), shown.len(), line_count(&shown));
    let mut output = shown;
    if truncated {
        output += &format!("\n\n---\nShowing {end} of {full_length} chars, {shown_bytes} of {total_bytes} bytes, and {shown_lines} of {total_lines} lines. ");
        output += &format!("Use get_search_content({{ responseId: \"{response_id}\", urlIndex: 0, offset: {end} }}) for the next slice.");
    }
    let mut blocks = Vec::new();
    if let Some((data, mime_type)) = &result.image {
        blocks.push(Content::Image { data: data.clone(), mime_type: mime_type.clone() });
    }
    blocks.push(Content::text(output));
    if !truncated {
        blocks.push(Content::text(format!("\n---\nFull content is stored as responseId \"{response_id}\". Use get_search_content({{ responseId: \"{response_id}\", urlIndex: 0, offset: 0 }}) to retrieve it.")));
    }
    let image_count = usize::from(result.image.is_some());
    let mut details = json!({
        "urls": url_list,
        "urlCount": 1,
        "successful": 1,
        "totalChars": full_length,
        "title": result.title,
        "responseId": response_id,
        "truncated": truncated,
        "hasImage": image_count > 0,
        "imageCount": image_count,
        "mode": mode,
    });
    if let Some(mime) = &result.mime_type {
        details["mimeType"] = json!(mime);
    }
    if let Some(status) = result.status {
        details["status"] = json!(status);
    }
    for (key, value) in [("totalBytes", total_bytes), ("totalLines", total_lines), ("shownBytes", shown_bytes), ("shownLines", shown_lines)] {
        details[key] = json!(value);
    }
    ToolOutput { content: blocks, details: Some(details), is_error: false }
}

// ---------------------------------------------------------------------------------------------
// get_search_content

pub struct GetSearchContentTool;

impl Tool for GetSearchContentTool {
    fn name(&self) -> &'static str {
        "get_search_content"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_search_content".into(),
            description: format!("Retrieve bounded pages of full stored search results or fetched content, or find matching passages, from a previous {STORED_CONTENT_SOURCES} call."),
            parameters: json!({
                "type": "object",
                "required": ["responseId"],
                "properties": {
                    "responseId": {"type": "string", "description": format!("The responseId from {STORED_CONTENT_SOURCES}")},
                    "query": {"type": "string", "description": "Get content for this query (web_search)"},
                    "queryIndex": {"type": "integer", "minimum": 0, "description": "Get content for query at index"},
                    "url": {"type": "string", "description": "Get content for this URL"},
                    "urlIndex": {"type": "integer", "minimum": 0, "description": "Get content for URL at index"},
                    "offset": {"type": "integer", "minimum": 0, "description": "Character offset in stored search or fetched URL content (default 0). Ignored when findText is supplied."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": GET_CONTENT_LIMIT_MAX, "description": "Requested maximum stored-content characters (default and max use maxInlineContentChars). Search-page continuation guidance shares the global output cap and may reduce returnedChars. Ignored when findText is supplied."},
                    "findText": {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 500}, {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 500}, "minItems": 1, "maxItems": 10}], "description": "Text or texts to find in the selected stored content. When supplied, offset and limit are ignored."},
                    "findMode": {"type": "string", "enum": ["exact", "case-insensitive", "fuzzy"], "description": "Matching mode for findText (default: case-insensitive). Requires findText."}
                }
            }),
            constrained_sampling: None,
        }
    }

    fn read_only(&self) -> bool {
        true
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("read more of a stored web_search or fetch_content result by its responseId; findText finds passages in it")
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move { Ok(get_search_content(&context.shared.web, arguments)) })
    }
}

// The schema's ceiling for limit: the default output cap. A session that raises
// web.maxInlineContentChars can still ask for up to its own cap; validation uses this schema.
const GET_CONTENT_LIMIT_MAX: usize = 30_000;

// An index argument as a position. Validation hands numbers on as floats, so 0 arrives as 0.0.
fn as_index(value: &Value) -> Option<usize> {
    value.as_f64().filter(|n| n.fract() == 0.0 && *n >= 0.0).map(|n| n as usize)
}

fn find_queries(value: &Value) -> Result<Vec<String>, String> {
    let items: Vec<&str> = match value {
        Value::String(text) => vec![text.as_str()],
        Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    let queries: Vec<String> = items.iter().map(|query| crate::js::trim(query).to_string()).filter(|query| !query.is_empty()).collect();
    if queries.is_empty() {
        return Err("findText must contain at least one non-empty string".to_string());
    }
    Ok(queries)
}

fn find_details(found: &find::Found) -> Vec<(&'static str, Value)> {
    vec![
        ("matchCount", json!(found.match_count)),
        ("returnedMatches", json!(found.returned_matches)),
        ("queryResults", Value::Array(found.query_results.iter().map(|(query, count)| json!({"query": query, "matchCount": count})).collect())),
    ]
}

fn with_entries(mut details: Value, entries: Vec<(&'static str, Value)>) -> Value {
    for (key, value) in entries {
        details[key] = value;
    }
    details
}

// Offset and limit, checked the way pi-web-access checks them; `what` names the content in the
// message ("query \"x\"" or "URL \"y\"").
fn check_window(offset: &Value, limit: &Value, max_chars: usize, what: &str) -> Result<(usize, usize), ToolOutput> {
    let Some(offset_value) = as_index(offset) else {
        return Err(text_output(format!("Invalid offset: received {} for {what}; offset must be a non-negative integer. Use 0 or a larger integer.", format_input_value(offset)), json!({"error": "Invalid offset", "offset": offset}), false));
    };
    let limit_value = as_index(limit).filter(|n| *n > 0 && *n <= max_chars);
    let Some(limit_value) = limit_value else {
        return Err(text_output(format!("Invalid limit: received {} for {what}; limit must be an integer from 1 to {max_chars}. Use a value in that range.", format_input_value(limit)), json!({"error": "Invalid limit", "limit": limit, "maxLimit": max_chars}), false));
    };
    Ok((offset_value, limit_value))
}

// get_search_content's whole answer for one call's arguments; public so the output can be checked
// against pi-web-access's on the same stored data.
pub fn get_search_content(web: &WebShared, mut arguments: Map<String, Value>) -> ToolOutput {
    let max_chars = web.settings.max_inline_content_chars;
    let response_id = arguments.get("responseId").cloned().unwrap_or(Value::Null);
    let id_text = format_input_value(&response_id);
    let find_mode = match arguments.get("findMode").and_then(Value::as_str) {
        Some("exact") => Some(FindMode::Exact),
        Some("fuzzy") => Some(FindMode::Fuzzy),
        Some("case-insensitive") => Some(FindMode::CaseInsensitive),
        _ => None,
    };
    for key in ["query", "url"] {
        if arguments.get(key).and_then(Value::as_str).is_some_and(|text| crate::js::trim(text).is_empty()) {
            arguments.remove(key);
        }
    }
    let find_text = arguments.get("findText").cloned();
    if find_text.is_some() {
        arguments.remove("offset");
        arguments.remove("limit");
    }
    if let (Some(mode), None) = (arguments.get("findMode"), &find_text) {
        return text_output(format!("findMode {} requires findText; provide findText or omit findMode.", format_input_value(mode)), json!({"error": "findMode requires findText"}), false);
    }
    let mode = find_mode.unwrap_or(FindMode::CaseInsensitive);
    let Some(data) = response_id.as_str().and_then(|id| web.store.get(id)) else {
        return text_output(format!("Error: No stored results for responseId {id_text}. Use a responseId returned by {STORED_CONTENT_SOURCES}."), json!({"error": "Not found", "responseId": response_id}), false);
    };
    let offset = arguments.get("offset").cloned().unwrap_or(json!(0));
    let limit = arguments.get("limit").cloned().unwrap_or(json!(max_chars));

    match data.get("type").and_then(Value::as_str) {
        Some("search") if data.get("queries").and_then(Value::as_array).is_some() => {
            let queries = data["queries"].as_array().unwrap();
            let query_of = |q: &Value| q.get("query").and_then(Value::as_str).unwrap_or("").to_string();
            let (index, query_data) = if let Some(query) = arguments.get("query") {
                match queries.iter().position(|q| q.get("query") == Some(query)) {
                    Some(index) => (index, &queries[index]),
                    None => {
                        let available = queries.iter().map(|q| format!("\"{}\"", query_of(q))).collect::<Vec<_>>().join(", ");
                        let available = if available.is_empty() { "none".to_string() } else { available };
                        let shown = format_input_value(query);
                        return text_output(format!("Query {shown} was not found for responseId {id_text}. Received query={shown}. Available queries: {available}. Use one of the available queries or queryIndex."), json!({"error": "Query not found"}), false);
                    }
                }
            } else if let Some(query_index) = arguments.get("queryIndex") {
                let available = queries.iter().enumerate().map(|(i, q)| format!("{i}: \"{}\"", query_of(q))).collect::<Vec<_>>().join(", ");
                let available = if available.is_empty() { "none".to_string() } else { available };
                match as_index(query_index).and_then(|i| queries.get(i).map(|q| (i, q))) {
                    Some(found) => found,
                    None => {
                        let shown = format_input_value(query_index);
                        return text_output(format!("Query index {shown} is out of range for responseId {id_text}. Received queryIndex={shown}; valid indexes are 0-{}. Available queries: {available}. Use one of the available indexes.", queries.len() as isize - 1), json!({"error": "Index out of range"}), false);
                    }
                }
            } else {
                let available = queries.iter().enumerate().map(|(i, q)| format!("{i}: \"{}\"", query_of(q))).collect::<Vec<_>>().join(", ");
                let available = if available.is_empty() { "none".to_string() } else { available };
                return text_output(format!("Specify query or queryIndex for responseId {id_text}. Available queries: {available}."), json!({"error": "No query specified"}), false);
            };
            let query = query_of(query_data);
            let query_shown = format_input_value(&json!(query));
            if let Some(error) = query_data.get("error").and_then(Value::as_str) {
                return text_output(format!("Error retrieving query {query_shown} from responseId {id_text}: {error}. Check the stored search result and retry with another query or queryIndex if needed."), json!({"error": error, "query": query}), false);
            }
            let full = format_full_results(query_data);
            let result_count = query_data.get("results").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
            let content_length = crate::js::utf16_len(&full);
            if let Some(find_value) = &find_text {
                return match find_queries(find_value) {
                    Ok(texts) => {
                        let found = find::find_content(&full, &texts, mode);
                        let details = with_entries(json!({"responseId": response_id, "query": query, "resultCount": result_count, "contentLength": content_length, "findMode": mode.name()}), find_details(&found));
                        text_output(found.text, details, false)
                    }
                    Err(error) => text_output(format!("Unable to find {} in query {query_shown} for responseId {id_text}: {error}. Check findText and use a supported findMode.", format_input_value(find_value)), json!({"error": error, "query": query}), false),
                };
            }
            let (offset, limit) = match check_window(&offset, &limit, max_chars, &format!("query {query_shown}")) {
                Ok(window) => window,
                Err(output) => return output,
            };
            if offset > content_length {
                return text_output(format!("Offset {offset} is out of range for query {query_shown} in responseId {id_text}. Received offset {offset}; valid range is 0-{content_length}. Use an offset within that range."), json!({"error": "Offset out of range", "offset": offset, "contentLength": content_length}), false);
            }
            let units = utf16(&full);
            let (end, text) = slice_with_continuation(&units, offset, limit, max_chars, |end| {
                format!("\n\n---\nShowing chars {offset}-{end} of {content_length}. Use get_search_content({{ responseId: \"{}\", queryIndex: {index}, offset: {end}, limit: {limit} }}) for the next slice.", response_id.as_str().unwrap_or(""))
            });
            let has_more = end < content_length;
            let details = json!({"responseId": response_id, "query": query, "resultCount": result_count, "contentLength": content_length, "offset": offset, "limit": limit, "returnedChars": end - offset, "nextOffset": if has_more { json!(end) } else { Value::Null }, "truncated": has_more});
            text_output(text, details, false)
        }
        Some("fetch") if data.get("urls").and_then(Value::as_array).is_some() => {
            let urls: Vec<Extracted> = data["urls"].as_array().unwrap().iter().map(|url| serde_json::from_value(url.clone()).unwrap_or_default()).collect();
            let listing = |numbered: bool| {
                let lines: Vec<String> = urls.iter().enumerate().map(|(i, u)| if numbered { format!("{i}: {}", u.url) } else { u.url.clone() }).collect();
                if lines.is_empty() { "  none".to_string() } else { lines.join("\n  ") }
            };
            let (index, url_data) = if let Some(url) = arguments.get("url") {
                match urls.iter().position(|u| Some(u.url.as_str()) == url.as_str()) {
                    Some(index) => (index, &urls[index]),
                    None => {
                        let shown = format_input_value(url);
                        return text_output(format!("URL {shown} was not found for responseId {id_text}. Received url={shown}. Available URLs:\n  {}\nUse one of the available URLs or urlIndex.", listing(false)), json!({"error": "URL not found"}), false);
                    }
                }
            } else if let Some(url_index) = arguments.get("urlIndex") {
                match as_index(url_index).and_then(|i| urls.get(i).map(|u| (i, u))) {
                    Some(found) => found,
                    None => {
                        let shown = format_input_value(url_index);
                        return text_output(format!("URL index {shown} is out of range for responseId {id_text}. Received urlIndex={shown}; valid indexes are 0-{}. Available URLs:\n  {}\nUse one of the available indexes.", urls.len() as isize - 1, listing(true)), json!({"error": "Index out of range"}), false);
                    }
                }
            } else if urls.len() == 1 {
                (0, &urls[0])
            } else {
                return text_output(format!("Specify url or urlIndex for responseId {id_text}. Available URLs:\n  {}", listing(true)), json!({"error": "No URL specified"}), false);
            };
            let url_shown = format_input_value(&json!(url_data.url));
            if let Some(error) = &url_data.error {
                return text_output(format!("Error retrieving URL {url_shown} from responseId {id_text}: {error}. Check the stored fetch result and retry with another URL or urlIndex if needed."), json!({"error": error, "url": url_data.url}), false);
            }
            let heading = if url_data.title.is_empty() { &url_data.url } else { &url_data.title };
            let units = utf16(&url_data.content);
            let content_length = units.len();
            if let Some(find_value) = &find_text {
                return match find_queries(find_value) {
                    Ok(texts) => {
                        let found = find::find_content(&url_data.content, &texts, mode);
                        let details = with_entries(json!({"url": url_data.url, "title": url_data.title, "contentLength": content_length, "findMode": mode.name()}), find_details(&found));
                        text_output(format!("# {heading}\n\n{}", found.text), details, false)
                    }
                    Err(error) => text_output(format!("Unable to find {} in URL {url_shown} for responseId {id_text}: {error}. Check findText and use a supported findMode.", format_input_value(find_value)), json!({"error": error, "url": url_data.url}), false),
                };
            }
            let (offset, limit) = match check_window(&offset, &limit, max_chars, &format!("URL {url_shown}")) {
                Ok(window) => window,
                Err(output) => return output,
            };
            if offset > content_length {
                return text_output(format!("Offset {offset} is out of range for URL {url_shown} in responseId {id_text}. Received offset {offset}; valid range is 0-{content_length}. Use an offset within that range."), json!({"error": "Offset out of range", "offset": offset, "contentLength": content_length}), false);
            }
            let end = page_end(&units, offset, (offset + limit).min(content_length));
            let slice = from_utf16(&units[offset..end]);
            let has_more = end < content_length;
            let mut text = format!("# {heading}\n\n{slice}");
            if has_more || offset > 0 {
                text += &format!("\n\n---\nShowing chars {offset}-{end} of {content_length}.");
                if has_more {
                    text += &format!(" Use get_search_content({{ responseId: \"{}\", urlIndex: {index}, offset: {end}, limit: {limit} }}) for the next slice.", response_id.as_str().unwrap_or(""));
                }
            }
            let details = json!({"url": url_data.url, "title": url_data.title, "contentLength": content_length, "offset": offset, "limit": limit, "returnedChars": end - offset, "nextOffset": if has_more { json!(end) } else { Value::Null }, "truncated": has_more});
            text_output(text, details, false)
        }
        kind => {
            let kind = kind.map(Value::from).unwrap_or(Value::Null);
            text_output(format!("Invalid stored data for responseId {id_text}: received type {}. Use a responseId returned by {STORED_CONTENT_SOURCES}.", format_input_value(&kind)), json!({"error": "Invalid data"}), false)
        }
    }
}

// One query's stored results as get_search_content pages them (web-tool-core.ts formatFullResults).
fn format_full_results(query_data: &Value) -> String {
    let query = query_data.get("query").and_then(Value::as_str).unwrap_or("");
    let mut output = format!("## Results for: \"{query}\"\n\n");
    let providers: Vec<String> = match query_data.get("providers").and_then(Value::as_array) {
        Some(list) => list.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        None => query_data.get("provider").and_then(Value::as_str).map(|p| vec![p.to_string()]).unwrap_or_default(),
    };
    if !providers.is_empty() {
        output += &format!("**Provider{}:** {}\n\n", if providers.len() == 1 { "" } else { "s" }, providers.join(", "));
    }
    if let Some(answer) = query_data.get("answer").and_then(Value::as_str).filter(|a| !a.is_empty()) {
        output += &format!("{answer}\n\n---\n\n");
    }
    for result in query_data.get("results").and_then(Value::as_array).into_iter().flatten() {
        let title = result.get("title").and_then(Value::as_str).unwrap_or("");
        let url = result.get("url").and_then(Value::as_str).unwrap_or("");
        let snippet = result.get("snippet").and_then(Value::as_str).unwrap_or("");
        output += &format!("### {title}\n{url}{}\n\n", if snippet.is_empty() { String::new() } else { format!("\n\n{snippet}") });
    }
    output
}
