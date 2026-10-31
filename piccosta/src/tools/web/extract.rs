// Turning a URL into text a model can read: a page's article as Markdown, a text or Markdown body
// as it is, a PDF's text, an image resized to send.
//
// Based on pi-web-access 0.44.0 (MIT, see LICENSE.md): extract.ts extractContent
// and extractViaHttp, and pdf-extract.ts's local path. The error texts and the order of checks are
// pi-web-access's. Loosely based on it, on purpose:
// - One way to fetch: direct HTTP. pi-web-access falls back through a dozen hosted services
//   (Jina, Firecrawl, Kagi, ...), each a key or a third party seeing every URL; the failure text
//   here keeps only the fallback that exists, web_search.
// - Readability is dom_smoothie (a port of the same Mozilla Readability) and Markdown is htmd
//   (turndown's rules). The Next.js RSC and Defuddle second passes are left out.
// - A PDF's text comes back as the content, paged by get_search_content like any page.
//   pi-web-access writes it to a file under /tmp and returns the path.
// - Video, YouTube, browser cookies, proxies and the answer mode are left out. GitHub repo URLs
//   go to github.rs first.

use super::net::{self, WebClient};
use super::WebShared;
use crate::tools::image;
use regex::Regex;
use reqwest::header::{HeaderMap, HeaderName, CONTENT_LENGTH, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use url::Url;

// extract.ts:69-71: below this many characters an extraction counts as having found nothing.
const MIN_USEFUL_CONTENT: usize = 500;
// How many <meta http-equiv="refresh"> pages are followed for one URL, the same budget as HTTP
// redirects (ssrf-protection.ts:6).
const MAX_META_REFRESH_HOPS: usize = 5;
// A response larger than this is refused, unless it's a PDF (extract.ts:1328, `?? 5`).
const DEFAULT_MAX_RESPONSE_MB: u64 = 5;
const SUPPORTED_IMAGE_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/webp", "image/gif"];
// Errors that end a fetch: retrying elsewhere can't help (extract.ts:69, 142-158).
const NON_RECOVERABLE_ERRORS: [&str; 4] = ["Unsupported content type", "Response too large", "PDF extraction is disabled", "Image fetching is disabled"];
const BROWSER_ACCEPT: &str = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8";
// Servers that negotiate Markdown (Cloudflare's Markdown for Agents, Mintlify, Vercel) send it
// straight; everyone else sees BROWSER_ACCEPT's preferences (extract.ts:1262-1266).
const MARKDOWN_FIRST_ACCEPT: &str = "text/markdown,text/html;q=0.9,application/xhtml+xml;q=0.9,application/xml;q=0.8,image/avif;q=0.9,image/webp;q=0.9,image/apng;q=0.9,*/*;q=0.7";
// The HTML spec's JavaScript MIME types; any other <script type> is a data block (extract.ts:1154-1159).
const JAVASCRIPT_MIME_TYPES: [&str; 16] = [
    "application/ecmascript", "application/javascript", "application/x-ecmascript", "application/x-javascript",
    "text/ecmascript", "text/javascript", "text/javascript1.0", "text/javascript1.1", "text/javascript1.2",
    "text/javascript1.3", "text/javascript1.4", "text/javascript1.5", "text/jscript", "text/livescript",
    "text/x-ecmascript", "text/x-javascript",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Readable,
    Raw,
}

// One fetched URL as fetch_content stores it (extract.ts ExtractedContent). The image is never
// stored: pi-web-access strips thumbnails before storing.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Extracted {
    pub url: String,
    pub title: String,
    pub content: String,
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(skip)]
    pub image: Option<(String, String)>,
}

impl Extracted {
    fn failed(url: &str, error: impl Into<String>) -> Extracted {
        Extracted { url: url.to_string(), error: Some(error.into()), ..Default::default() }
    }
}

fn is_non_recoverable(error: &str) -> bool {
    NON_RECOVERABLE_ERRORS.iter().any(|prefix| error.starts_with(prefix))
        || error.starts_with("Blocked internal ")
        || error.starts_with("Blocked hostname by fetch_content domain policy")
        || error.starts_with("Hostname not allowed by fetch_content domain policy")
        || error.starts_with("Too many redirects fetching ")
        || error == "Only HTTP and HTTPS URLs can be fetched remotely"
        || error == "URL must include a hostname"
        || error.starts_with("Failed to resolve ")
}

// Input: one URL as the model gave it. Output: what fetch_content stores for it; failures are in
// `error`, never an Err (extract.ts:548-1141, the direct-HTTP route).
pub async fn extract_content(url: &str, mode: Mode, force_clone: bool, web: &WebShared, cancel: &CancellationToken) -> Extracted {
    if cancel.is_cancelled() {
        return Extracted::failed(url, "Aborted");
    }
    let Ok(parsed) = Url::parse(url) else {
        return Extracted::failed(url, "Invalid URL");
    };
    let client = match web.client() {
        Ok(client) => client,
        Err(error) => return Extracted::failed(url, error),
    };
    if let Err(error) = client.validate(&parsed) {
        return Extracted::failed(url, error);
    }
    // A GitHub repo, tree or blob URL is a clone (raw mode always fetches the page itself).
    if mode != Mode::Raw
        && let Some(info) = super::github::parse_github_url(url)
        && let Some(found) = tokio::select! {
            found = super::github::extract_github(url, &info, force_clone, web, cancel) => found,
            () = cancel.cancelled() => None,
        }
    {
        return found;
    }
    if cancel.is_cancelled() {
        return Extracted::failed(url, "Aborted");
    }
    let timeout = Duration::from_millis(web.settings.fetch_timeout_ms);
    let result = extract_via_http(url, &parsed, mode, mode != Mode::Raw, timeout, web, client, cancel, 0).await;
    if mode == Mode::Raw || result.error.is_none() {
        return result;
    }
    let error = result.error.clone().unwrap_or_default();
    if is_non_recoverable(&error) || error == "Aborted" {
        return result;
    }
    // A 404 or 410 means no other route could find it either; send the model to search instead
    // (extract.ts:294-312).
    if let Some(status @ (404 | 410)) = result.status {
        let guidance = [
            error,
            String::new(),
            format!("The origin server says this page does not exist (HTTP {status}), so extraction providers cannot retrieve it."),
            "The page may have moved or been renamed. Use web_search to find the current URL, then retry fetch_content with it.".to_string(),
        ];
        return Extracted { error: Some(guidance.join("\n")), ..result };
    }
    let guidance = [error, String::new(), "Fallback options:".to_string(), "  • Use web_search to find content about this topic".to_string()];
    Extracted { error: Some(guidance.join("\n")), ..result }
}

// The request's headers (extract.ts:1288-1300).
fn request_headers(user_agent: &str, prefer_markdown: bool) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let mut set = |name: &'static str, value: &str| {
        headers.insert(HeaderName::from_static(name), net::header_value(value));
    };
    set("user-agent", user_agent);
    set("accept", if prefer_markdown { MARKDOWN_FIRST_ACCEPT } else { BROWSER_ACCEPT });
    set("accept-language", "en-US,en;q=0.9");
    set("cache-control", "no-cache");
    set("sec-fetch-dest", "document");
    set("sec-fetch-mode", "navigate");
    set("sec-fetch-site", "none");
    set("sec-fetch-user", "?1");
    set("upgrade-insecure-requests", "1");
    headers
}

// extract.ts:1201-1212.
fn is_text_content_type(content_type: &str) -> bool {
    let mime = mime_of(content_type);
    mime.starts_with("text/")
        || matches!(mime.as_str(), "application/json" | "application/ld+json" | "application/xml" | "application/xhtml+xml" | "application/javascript" | "application/x-javascript")
        || mime.ends_with("+json")
        || mime.ends_with("+xml")
}

fn mime_of(content_type: &str) -> String {
    content_type.split(';').next().unwrap_or("").trim().to_lowercase()
}

// A PDF by its Content-Type or its path (pdf-extract.ts isPDF).
fn is_pdf(url: &Url, content_type: &str) -> bool {
    content_type.contains("application/pdf") || url.path().to_lowercase().ends_with(".pdf")
}

// A text body's title: its first # or ## heading, else the URL's last path segment, else the URL
// (extract.ts:1607-1616).
fn text_title(text: &str, url: &str) -> String {
    static HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^#{1,2}\s+(.+)").unwrap());
    if let Some(found) = HEADING.captures(text) {
        let cleaned = found[1].replace('*', "");
        let cleaned = crate::js::trim(&cleaned);
        if !cleaned.is_empty() {
            return cleaned.to_string();
        }
    }
    last_path_segment(url).unwrap_or_else(|| url.to_string())
}

fn last_path_segment(url: &str) -> Option<String> {
    let parsed = Url::parse(url).ok()?;
    let segment = parsed.path().rsplit('/').next().unwrap_or("");
    (!segment.is_empty()).then(|| segment.to_string())
}

// Cloudflare's interstitial served with a 200 (extract.ts:1143-1150).
fn is_cloudflare_challenge(status: u16, headers: &HeaderMap, text: &str, is_html: bool) -> bool {
    static CHALLENGE_TITLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)<title>\s*Just a moment\.\.\.\s*</title>").unwrap());
    if status != 200 {
        return false;
    }
    if headers.get("cf-mitigated").and_then(|value| value.to_str().ok()) == Some("challenge") {
        return true;
    }
    is_html && CHALLENGE_TITLE.is_match(text) && text.contains("window._cf_chl_opt") && text.contains("/cdn-cgi/challenge-platform/")
}

fn is_executable_script_type(script_type: Option<&str>) -> bool {
    let mime = script_type.unwrap_or("").split(';').next().unwrap_or("").trim().to_lowercase();
    mime.is_empty() || mime == "module" || JAVASCRIPT_MIME_TYPES.contains(&mime.as_str())
}

static SCRIPT_BLOCK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<script[\s\S]*?</script>").unwrap());
static STYLE_BLOCK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<style[\s\S]*?</style>").unwrap());
static ANY_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static SCRIPT_OPEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)<script\b([^>]*)>").unwrap());
static SCRIPT_TYPE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?i)\btype\s*=\s*["']?([^"'\s>]+)"#).unwrap());

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_space = false;
    for c in text.chars() {
        if crate::js::is_whitespace(c) {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    crate::js::trim(&out).to_string()
}

fn executable_script_count(html: &str) -> usize {
    SCRIPT_OPEN
        .captures_iter(html)
        .filter(|found| is_executable_script_type(SCRIPT_TYPE.captures(&found[1]).map(|t| t.get(1).unwrap().as_str())))
        .count()
}

// Under 500 characters of body text and more than three scripts (extract.ts:1166-1184).
fn is_likely_js_rendered(html: &str) -> bool {
    static BODY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<body[^>]*>([\s\S]*?)</body>").unwrap());
    let Some(body) = BODY.captures(html) else { return false };
    let without_scripts = SCRIPT_BLOCK.replace_all(&body[1], "");
    let without_styles = STYLE_BLOCK.replace_all(&without_scripts, "");
    let text = collapse_whitespace(&ANY_TAG.replace_all(&without_styles, ""));
    crate::js::utf16_len(&text) < 500 && executable_script_count(html) > 3
}

// The page's visible text length, for saying how incomplete an extraction looks
// (extract.ts:1538-1548, there from a fresh DOM; here from the markup with script, style,
// noscript and template removed).
fn visible_text_chars(html: &str) -> usize {
    static BODY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<body[^>]*>([\s\S]*)</body>").unwrap());
    static HIDDEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<(script|style|noscript|template)\b[\s\S]*?</(script|style|noscript|template)>").unwrap());
    let body = BODY.captures(html).map(|found| found[1].to_string()).unwrap_or_default();
    let shown = HIDDEN.replace_all(&body, "");
    crate::js::utf16_len(&collapse_whitespace(&decode_entities(&ANY_TAG.replace_all(&shown, ""))))
}

// The few character references a <title> or visible text needs decoded to count right.
fn decode_entities(text: &str) -> String {
    static ENTITY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"&(#[0-9]+|#[xX][0-9a-fA-F]+|amp|lt|gt|quot|apos|nbsp);").unwrap());
    ENTITY
        .replace_all(text, |found: &regex::Captures| {
            let name = &found[1];
            let decoded = match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                _ if name.starts_with("#x") || name.starts_with("#X") => u32::from_str_radix(&name[2..], 16).ok().and_then(char::from_u32),
                _ => name[1..].parse::<u32>().ok().and_then(char::from_u32),
            };
            decoded.map(String::from).unwrap_or_else(|| found[0].to_string())
        })
        .into_owned()
}

// Where a page's <meta http-equiv="refresh" content="0; url=..."> sends the browser, resolved
// against the page's URL. pi-web-access doesn't follow these; a moved page that leaves one behind
// (blog.rust-lang.org does) otherwise extracts as empty.
fn meta_refresh_target(html: &str, base: &Url) -> Option<Url> {
    static META: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?i)<meta\b[^>]*http-equiv\s*=\s*["']?refresh["']?[^>]*>"#).unwrap());
    static CONTENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?i)content\s*=\s*(?:"([^"]*)"|'([^']*)')"#).unwrap());
    static TARGET: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?i)^\s*[0-9.]*\s*[;,]\s*url\s*=\s*['"]?([^'"]+)"#).unwrap());
    let tag = META.find(html)?;
    let content = CONTENT.captures(tag.as_str())?;
    let value = content.get(1).or_else(|| content.get(2))?.as_str();
    let target = TARGET.captures(value)?.get(1)?.as_str().trim();
    base.join(target).ok()
}

// The document's <title>, whitespace collapsed (what document.title gives).
fn document_title(html: &str) -> String {
    static TITLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<title[^>]*>([\s\S]*?)</title>").unwrap());
    TITLE.captures(html).map(|found| collapse_whitespace(&decode_entities(&found[1]))).unwrap_or_default()
}

// Readability's article as Markdown, the way pi-web-access gets it (Readability, then turndown
// with ATX headings and fenced code). Output: (article title, markdown), or None when
// Readability finds no article.
fn readable_markdown(html: &str, url: &str) -> Option<(String, String)> {
    let mut readability = dom_smoothie::Readability::new(html, Some(url), None).ok()?;
    let article = readability.parse().ok()?;
    let converter = htmd::HtmlToMarkdown::builder().skip_tags(vec!["script", "style"]).build();
    let markdown = converter.convert(&article.content).ok()?;
    Some((article.title.trim().to_string(), markdown))
}

// Inline data: URIs in extracted Markdown would otherwise reach the model and the cache as
// base64 (extract.ts:1627-1631, data-uri-sanitize.ts); each becomes a short marker.
fn sanitize_data_uris(text: &str) -> String {
    static DATA_URI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bdata:([a-z0-9.+-]+/[a-z0-9.+-]+)?((?:;[a-z0-9=._+-]+)*),([^\s)\]'\x22<>]*)").unwrap());
    DATA_URI
        .replace_all(text, |found: &regex::Captures| {
            let mime = found.get(1).map(|m| m.as_str().to_lowercase()).unwrap_or_else(|| "text/plain".to_string());
            let encoding = if found[2].to_lowercase().contains(";base64") { "base64" } else { "percent" };
            format!("[inline data URI omitted; mime={mime}; encoding={encoding}; encodedBytes={}]", found[3].len())
        })
        .into_owned()
}

// PDF text strings: UTF-16BE behind a BOM, else PDFDocEncoding (Latin-1 for what titles use).
fn decode_pdf_text(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes[0] == 0xfe && bytes[1] == 0xff {
        let units: Vec<u16> = bytes[2..].chunks(2).filter(|pair| pair.len() == 2).map(|pair| u16::from_be_bytes([pair[0], pair[1]])).collect();
        return String::from_utf16_lossy(&units);
    }
    bytes.iter().map(|byte| *byte as char).collect()
}

fn pdf_info_title(document: &pdf_oxide::PdfDocument) -> Option<String> {
    let info = document.trailer().as_dict()?.get("Info")?.clone();
    let info = document.resolve_object(&info).ok()?;
    let title = info.as_dict()?.get("Title")?.clone();
    let title = document.resolve_object(&title).ok()?;
    let text = decode_pdf_text(title.as_string()?);
    let trimmed = crate::js::trim(&text);
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

// A title from a PDF's URL: the file name without .pdf, arxiv ids as "arxiv-<id>", separators as
// spaces (pdf-extract.ts extractTitleFromURL).
fn pdf_title_from_url(url: &str) -> String {
    static ARXIV: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"/(?:pdf|abs)/(\d+\.\d+)").unwrap());
    static SEPARATORS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[_-]+").unwrap());
    let Ok(parsed) = Url::parse(url) else { return "document".to_string() };
    let path = parsed.path();
    let base = path.rsplit('/').next().unwrap_or("");
    let mut name = base.strip_suffix(".pdf").unwrap_or(base).to_string();
    if parsed.host_str().unwrap_or("").contains("arxiv.org")
        && let Some(found) = ARXIV.captures(path)
    {
        name = format!("arxiv-{}", &found[1]);
    }
    let name = collapse_whitespace(&SEPARATORS.replace_all(&name, " "));
    if name.is_empty() { "document".to_string() } else { name }
}

// A PDF's pages as Markdown (pdf-extract.ts 262-371): a title, the source, the page count, then
// each page's text with a page marker between pages.
fn pdf_markdown(bytes: Vec<u8>, url: &str, max_pages: usize) -> Result<(String, String), String> {
    let document = pdf_oxide::PdfDocument::from_bytes(bytes).map_err(|error| error.to_string())?;
    let page_count = document.page_count().map_err(|error| error.to_string())?;
    let title = pdf_info_title(&document).unwrap_or_else(|| pdf_title_from_url(url));
    let pages_to_extract = page_count.min(max_pages);
    let truncated = page_count > max_pages;
    let mut body: Vec<String> = Vec::new();
    for index in 0..pages_to_extract {
        let text = document.extract_text(index).unwrap_or_default();
        let text = collapse_whitespace(&text);
        if text.is_empty() {
            continue;
        }
        if !body.is_empty() {
            body.push(String::new());
            body.push(format!("<!-- Page {} -->", index + 1));
            body.push(String::new());
        }
        body.push(text);
    }
    let mut lines = vec![format!("# {title}"), String::new(), format!("> Source: {url}")];
    if page_count > 0 {
        lines.push(format!("> Pages: {page_count}{}", if truncated { format!(" (extracted first {pages_to_extract})") } else { String::new() }));
    }
    lines.extend([String::new(), "---".to_string(), String::new()]);
    if !body.is_empty() {
        lines.push(body.join("\n"));
    }
    if truncated {
        lines.extend([String::new(), "---".to_string(), String::new(), format!("*[Truncated: Only first {pages_to_extract} of {page_count} pages extracted]*")]);
    }
    Ok((title, lines.join("\n")))
}

fn too_large_text(max_bytes: usize) -> String {
    format!("Response too large ({}MB)", (max_bytes as f64 / 1024.0 / 1024.0).round() as u64)
}

// One direct HTTP fetch and its extraction (extract.ts:1268-1605). The deadline covers the whole
// fetch, body included; running past it gives pi-web-access's "The operation was aborted."
#[allow(clippy::too_many_arguments)]
fn extract_via_http<'a>(
    url: &'a str,
    parsed: &'a Url,
    mode: Mode,
    prefer_markdown: bool,
    timeout: Duration,
    web: &'a WebShared,
    client: &'a WebClient,
    cancel: &'a CancellationToken,
    meta_hops: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Extracted> + Send + 'a>> {
    Box::pin(async move {
        let started = Instant::now();
        tokio::select! {
            result = tokio::time::timeout(timeout, fetch_and_extract(url, parsed, mode, prefer_markdown, timeout, started, web, client, cancel, meta_hops)) => {
                result.unwrap_or_else(|_| Extracted::failed(url, "The operation was aborted."))
            }
            () = cancel.cancelled() => Extracted::failed(url, "Aborted"),
        }
    })
}

#[allow(clippy::too_many_arguments)]
async fn fetch_and_extract(
    url: &str,
    parsed: &Url,
    mode: Mode,
    prefer_markdown: bool,
    timeout: Duration,
    started: Instant,
    web: &WebShared,
    client: &WebClient,
    cancel: &CancellationToken,
    meta_hops: usize,
) -> Extracted {
    let headers = request_headers(&client.user_agent, prefer_markdown);
    let (response, final_url) = match client.get(parsed, &headers).await {
        Ok(found) => found,
        Err(error) => return Extracted::failed(url, error),
    };
    let status = response.status();
    let code = status.as_u16();
    if !status.is_success() && mode != Mode::Raw {
        return Extracted { status: Some(code), ..Extracted::failed(url, format!("HTTP {code}: {}", status.canonical_reason().unwrap_or(""))) };
    }
    let response_headers = response.headers().clone();
    let content_type = response_headers.get(CONTENT_TYPE).and_then(|value| value.to_str().ok()).unwrap_or("").to_string();
    let mime = mime_of(&content_type);
    let pdf = is_pdf(parsed, &content_type);
    let max_response_bytes = (if pdf { web.settings.pdf_max_size_mb } else { DEFAULT_MAX_RESPONSE_MB } * 1024 * 1024) as usize;
    let pdf_too_large = || format!("PDF exceeds configured pdf.maxSizeMB limit ({} MB)", web.settings.pdf_max_size_mb);
    if let Some(length) = response_headers.get(CONTENT_LENGTH).and_then(|value| value.to_str().ok()).and_then(|value| value.trim().parse::<u64>().ok())
        && length as usize > max_response_bytes
    {
        let error = if pdf { pdf_too_large() } else { format!("Response too large ({}MB)", (length as f64 / 1024.0 / 1024.0).round() as u64) };
        return Extracted::failed(url, error);
    }
    let with_status = |extracted: Extracted| Extracted { mime_type: Some(mime.clone()).filter(|m| !m.is_empty()), status: Some(code), ..extracted };

    if mode == Mode::Raw {
        if !is_text_content_type(&content_type) {
            let hint = if pdf { "; retry without mode for Markdown" } else { "" };
            let shown = if mime.is_empty() { "missing" } else { mime.as_str() };
            return with_status(Extracted::failed(url, format!("Unsupported content type in raw mode: {shown}{hint}")));
        }
        let bytes = match net::read_limited(response, max_response_bytes, || too_large_text(max_response_bytes)).await {
            Ok(bytes) => bytes,
            Err(error) => return Extracted::failed(url, error),
        };
        let text = net::decode_text(&bytes, &content_type, false);
        return with_status(Extracted { url: url.to_string(), title: text_title(&text, url), content: text, ..Default::default() });
    }

    if SUPPORTED_IMAGE_TYPES.contains(&mime.as_str()) {
        let bytes = match net::read_limited(response, max_response_bytes, || too_large_text(max_response_bytes)).await {
            Ok(bytes) => bytes,
            Err(error) => return with_status(Extracted::failed(url, error)),
        };
        let image_type = mime.clone();
        let prepared = tokio::task::spawn_blocking(move || image::prepare(&bytes, &image_type)).await;
        let Ok(Ok(prepared)) = prepared else {
            return with_status(Extracted::failed(url, format!("Could not decode image: {mime}")));
        };
        let title = last_path_segment(final_url.as_str()).unwrap_or_else(|| url.to_string());
        return Extracted {
            url: url.to_string(),
            title,
            content: format!("Image fetched ({}×{}, {})", prepared.width, prepared.height, prepared.mime_type),
            error: None,
            mime_type: Some(prepared.mime_type.clone()),
            status: Some(code),
            image: Some((prepared.data, prepared.mime_type)),
        };
    }

    if pdf {
        let bytes = match net::read_limited(response, max_response_bytes, pdf_too_large).await {
            Ok(bytes) => bytes,
            Err(error) => return Extracted::failed(url, error),
        };
        if cancel.is_cancelled() {
            return Extracted::failed(url, "Aborted");
        }
        let source = url.to_string();
        let max_pages = web.settings.pdf_max_pages;
        return match tokio::task::spawn_blocking(move || pdf_markdown(bytes, &source, max_pages)).await {
            Ok(Ok((title, content))) => Extracted { url: url.to_string(), title, content, ..Default::default() },
            Ok(Err(error)) => Extracted::failed(url, format!("PDF extraction failed: {error}")),
            Err(error) => Extracted::failed(url, format!("PDF extraction failed: {error}")),
        };
    }

    if ["application/octet-stream", "image/", "audio/", "video/", "application/zip"].iter().any(|kind| content_type.contains(kind)) {
        return Extracted::failed(url, format!("Unsupported content type: {}", content_type.split(';').next().unwrap_or("")));
    }

    let bytes = match net::read_limited(response, max_response_bytes, || too_large_text(max_response_bytes)).await {
        Ok(bytes) => bytes,
        Err(error) => return Extracted::failed(url, error),
    };
    let is_html = content_type.contains("text/html") || content_type.contains("application/xhtml+xml");
    let text = net::decode_text(&bytes, &content_type, is_html);
    if is_cloudflare_challenge(code, &response_headers, &text, is_html) {
        return Extracted { status: Some(code), ..Extracted::failed(url, format!("HTTP {code}: Blocked by Cloudflare challenge page")) };
    }

    if !is_html {
        let title = text_title(&text, url);
        if mime != "text/markdown" && mime != "text/x-markdown" {
            return Extracted { url: url.to_string(), title, content: text, ..Default::default() };
        }
        let markdown = Extracted { url: url.to_string(), title, content: text, ..Default::default() };
        if !prefer_markdown || crate::js::utf16_len(crate::js::trim(&markdown.content)) >= MIN_USEFUL_CONTENT {
            return markdown;
        }
        // Short negotiated Markdown: ask again with the browser's Accept, and let that answer win
        // unless it came back empty where the Markdown didn't (extract.ts:1452-1467).
        let remaining = timeout.saturating_sub(started.elapsed()).max(Duration::from_millis(1));
        let normal = extract_via_http(url, parsed, mode, false, remaining, web, client, cancel, meta_hops).await;
        let normal_error = normal.error.clone().unwrap_or_default();
        if cancel.is_cancelled() || is_non_recoverable(&normal_error) || matches!(normal.status, Some(404 | 410)) || normal_error.contains("abort") || normal_error == "Aborted" {
            return normal;
        }
        if !crate::js::trim(&normal.content).is_empty() || crate::js::trim(&markdown.content).is_empty() {
            return normal;
        }
        let retry_note = normal.error.as_ref().map(|error| format!("\nBrowser retry failed: {error}")).unwrap_or_default();
        return Extracted { error: Some(format!("Extracted content appears incomplete{retry_note}")), ..Extracted { mime_type: normal.mime_type, status: normal.status, ..markdown } };
    }

    // A page with almost nothing on it that refreshes elsewhere: fetch where it points.
    if meta_hops < MAX_META_REFRESH_HOPS
        && let Some(target) = meta_refresh_target(&text, &final_url)
        && visible_text_chars(&text) < MIN_USEFUL_CONTENT
    {
        if let Err(error) = client.validate(&target) {
            return Extracted::failed(url, error);
        }
        let remaining = timeout.saturating_sub(started.elapsed()).max(Duration::from_millis(1));
        return extract_via_http(url, &target, mode, prefer_markdown, remaining, web, client, cancel, meta_hops + 1).await;
    }
    let title_of_document = document_title(&text);
    let html = text.clone();
    let base = final_url.to_string();
    let article = tokio::task::spawn_blocking(move || readable_markdown(&html, &base)).await.ok().flatten();
    let Some((article_title, markdown)) = article else {
        let error = if is_likely_js_rendered(&text) {
            "Page appears to be JavaScript-rendered (content loads dynamically)"
        } else {
            "Could not extract readable content from HTML structure"
        };
        return Extracted { url: url.to_string(), title: title_of_document, error: Some(error.to_string()), ..Default::default() };
    };
    let title = if article_title.is_empty() { title_of_document } else { article_title };
    let markdown = sanitize_data_uris(&markdown);
    if crate::js::utf16_len(&markdown) >= MIN_USEFUL_CONTENT {
        return Extracted { url: url.to_string(), title, content: markdown, ..Default::default() };
    }
    // Too little came out. Say why when we can tell: a page that runs scripts may fill itself in
    // later, a static page this short is simply short (extract.ts:1526-1556).
    let error = if is_likely_js_rendered(&text) {
        Some("Page appears to be JavaScript-rendered (content loads dynamically)".to_string())
    } else {
        let has_script = executable_script_count(&text) > 0;
        let visible = visible_text_chars(&text);
        let extracted = crate::js::utf16_len(crate::js::trim(&markdown));
        let counts = format!("Extracted content appears incomplete: extracted {extracted} characters from a page with {visible} characters of visible text");
        if visible >= MIN_USEFUL_CONTENT {
            Some(counts)
        } else if has_script {
            Some(format!("Extracted content appears incomplete: the page has only {visible} characters of visible text and runs scripts, so content may load dynamically"))
        } else if extracted > 0 {
            None
        } else {
            Some(counts)
        }
    };
    Extracted { url: url.to_string(), title, content: markdown, error, ..Default::default() }
}

// Every URL of one call, three at a time across the whole session (extract.ts:1618-1633), in the
// order given.
pub async fn fetch_all(urls: &[String], mode: Mode, force_clone: bool, web: &WebShared, cancel: &CancellationToken) -> Vec<Extracted> {
    let jobs = urls.iter().map(|url| async move {
        let Ok(_slot) = web.fetch_slots.acquire().await else {
            return Extracted::failed(url, "Aborted");
        };
        extract_content(url, mode, force_clone, web, cancel).await
    });
    futures_util::future::join_all(jobs).await
}
