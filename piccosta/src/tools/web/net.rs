// Fetching a URL a model handed us without letting it reach this machine's own network.
//
// Based on pi-web-access 0.44.0 (MIT, see LICENSE.md): ssrf-protection.ts, and
// extract.ts's response readers (1186-1259). The refusal messages are pi-web-access's. What
// changed, on purpose:
// - The private-address check runs inside the DNS resolver, so the address that gets checked is
//   the one that gets connected to. pi-web-access looks a name up, checks the answer, then lets
//   fetch look it up again, and a rebinding DNS server can answer differently the second time.
// - Proxies from the environment are never used. Node's fetch ignores them too, so pi-web-access
//   went direct by default; through a proxy the name is resolved where we can't check it.
// - The User-Agent says what it is. pi-web-access's default calls itself OpenAI's and xAI's
//   fetchers ("OpenAI File Downloader, XaiImageApiFetch/1.0"); `web.fetchContent.userAgent`
//   still overrides ours.
// - An HTML page whose Content-Type names no charset is decoded by its <meta charset>, the way a
//   browser does, instead of always as UTF-8.

use crate::config::WebSettings;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::header::{HeaderMap, HeaderValue, LOCATION};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use url::Url;

// ssrf-protection.ts:6-7.
const MAX_REDIRECTS: usize = 5;
const REDIRECT_STATUSES: [u16; 5] = [301, 302, 303, 307, 308];
// What a fetch says it is, unless settings say otherwise.
const DEFAULT_USER_AGENT: &str = concat!("Mozilla/5.0 (compatible; piccosta/", env!("CARGO_PKG_VERSION"), ")");
// Bytes of an HTML page searched for a <meta charset> (the HTML spec's prescan window).
const CHARSET_PRESCAN_BYTES: usize = 1024;

// An address range the private-address check lets through, parsed from "a.b.c.d/n" or an address.
#[derive(Debug, Clone, Copy)]
struct Cidr {
    network: IpAddr,
    prefix: u8,
}

fn parse_cidr(raw: &str) -> Option<Cidr> {
    let (address, prefix) = match raw.split_once('/') {
        Some((address, prefix)) => (address, Some(prefix.parse::<u8>().ok()?)),
        None => (raw, None),
    };
    let network: IpAddr = address.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
    let max = if network.is_ipv4() { 32 } else { 128 };
    let prefix = prefix.unwrap_or(max);
    (prefix <= max).then_some(Cidr { network, prefix })
}

// An address as 16 bytes, IPv4 in its mapped form, with the matching prefix shift for a range.
fn in_range(address: IpAddr, range: &Cidr) -> bool {
    let (bytes, network, prefix): (Vec<u8>, Vec<u8>, u32) = match (address, range.network) {
        (IpAddr::V4(a), IpAddr::V4(n)) => (a.octets().to_vec(), n.octets().to_vec(), range.prefix as u32),
        (IpAddr::V6(a), IpAddr::V6(n)) => (a.octets().to_vec(), n.octets().to_vec(), range.prefix as u32),
        _ => return false,
    };
    let full_bytes = (prefix / 8) as usize;
    if bytes[..full_bytes] != network[..full_bytes] {
        return false;
    }
    let rest = prefix % 8;
    rest == 0 || (bytes[full_bytes] >> (8 - rest)) == (network[full_bytes] >> (8 - rest))
}

// 198.18.0.0/15, where TUN and fake-IP proxies put their synthetic answers.
fn is_fake_ip_proxy_address(address: Ipv4Addr) -> bool {
    let [a, b, _, _] = address.octets();
    a == 198 && (b == 18 || b == 19)
}

// ssrf-protection.ts:416-429: this host, private networks, carrier NAT, link-local, the fake-IP
// range, multicast and everything above it.
fn is_blocked_ipv4(address: Ipv4Addr) -> bool {
    let [a, b, _, _] = address.octets();
    a == 0
        || a == 10
        || a == 127
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 168)
        || is_fake_ip_proxy_address(address)
        || a >= 224
}

// ssrf-protection.ts:431-448: unspecified, loopback, unique-local, link-local, and an IPv4-mapped
// address by its IPv4 rules.
fn is_blocked_ipv6(address: Ipv6Addr) -> bool {
    let groups = address.segments();
    let first = groups[0];
    if groups.iter().all(|group| *group == 0) {
        return true;
    }
    if groups[..7].iter().all(|group| *group == 0) && groups[7] == 1 {
        return true;
    }
    if (first & 0xfe00) == 0xfc00 || (first & 0xffc0) == 0xfe80 {
        return true;
    }
    if groups[..5].iter().all(|group| *group == 0) && groups[5] == 0xffff {
        let [high, low] = [groups[6], groups[7]];
        return is_blocked_ipv4(Ipv4Addr::new((high >> 8) as u8, high as u8, (low >> 8) as u8, low as u8));
    }
    false
}

// Input: an address a hostname resolved to (or the hostname itself, when it's an IP literal).
// Output: the refusal when it's internal and no configured range allows it (ssrf-protection.ts:393-409).
fn check_address(address: IpAddr, hostname: &str, allow: &[Cidr]) -> Result<(), String> {
    if allow.iter().any(|range| in_range(address, range)) {
        return Ok(());
    }
    match address {
        IpAddr::V4(v4) if is_blocked_ipv4(v4) => {
            let hint = if is_fake_ip_proxy_address(v4) {
                ". This address is in 198.18.0.0/15, commonly used by TUN/fake-IP proxies. If that matches your setup, configure web.ssrf.allowRanges with [\"198.18.0.0/15\"] in settings.json."
            } else {
                ""
            };
            Err(format!("Blocked internal address for {hostname}: {v4}{hint}"))
        }
        IpAddr::V6(v6) if is_blocked_ipv6(v6) => Err(format!("Blocked internal address for {hostname}: {v6}")),
        _ => Ok(()),
    }
}

// A refusal raised inside the resolver, found again in reqwest's error chain by error_text.
#[derive(Debug)]
struct Refusal(String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refusal {}

// The resolver every web fetch connects through: the system's getaddrinfo, then the
// private-address check on every answer.
struct GuardedResolver {
    allow: Arc<Vec<Cidr>>,
}

impl Resolve for GuardedResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let allow = self.allow.clone();
        let host = name.as_str().to_string();
        Box::pin(async move {
            let lookup_host = host.clone();
            let found = tokio::task::spawn_blocking(move || (lookup_host.as_str(), 0).to_socket_addrs().map(|addrs| addrs.collect::<Vec<SocketAddr>>()))
                .await
                .map_err(|error| Box::new(Refusal(format!("Failed to resolve {host}: {error}"))) as Box<dyn std::error::Error + Send + Sync>)?;
            let addresses = found.map_err(|error| Box::new(Refusal(format!("Failed to resolve {host}: {error}"))) as Box<dyn std::error::Error + Send + Sync>)?;
            if addresses.is_empty() {
                return Err(Box::new(Refusal(format!("Failed to resolve {host}: no addresses returned"))) as Box<dyn std::error::Error + Send + Sync>);
            }
            for address in &addresses {
                check_address(address.ip(), &host, &allow).map_err(|refusal| Box::new(Refusal(refusal)) as Box<dyn std::error::Error + Send + Sync>)?;
            }
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

// A host lower-cased, without IPv6 brackets or a trailing dot (ssrf-protection.ts:304-306).
fn normalize_hostname(hostname: &str) -> String {
    hostname.trim_start_matches('[').trim_end_matches(']').trim_end_matches('.').to_lowercase()
}

fn domain_matches(hostname: &str, entry: &str) -> bool {
    hostname == entry || hostname.ends_with(&format!(".{entry}"))
}

// The web client: one per session, shared by every fetch_content and web_search page fetch.
pub struct WebClient {
    client: reqwest::Client,
    allow: Arc<Vec<Cidr>>,
    domain_allow: Vec<String>,
    domain_deny: Vec<String>,
    pub user_agent: String,
}

impl WebClient {
    pub fn new(settings: &WebSettings) -> Result<WebClient, String> {
        let allow: Arc<Vec<Cidr>> = Arc::new(settings.allow_ranges.iter().filter_map(|raw| parse_cidr(raw)).collect());
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .dns_resolver(Arc::new(GuardedResolver { allow: allow.clone() }))
            .build()
            .map_err(|error| error.to_string())?;
        let normalize = |entries: &[String]| entries.iter().map(|entry| entry.trim().trim_start_matches("*.").trim_end_matches('.').to_lowercase()).filter(|entry| !entry.is_empty()).collect();
        Ok(WebClient {
            client,
            allow,
            domain_allow: normalize(&settings.domain_allow),
            domain_deny: normalize(&settings.domain_deny),
            user_agent: settings.user_agent.clone().unwrap_or_else(|| DEFAULT_USER_AGENT.to_string()),
        })
    }

    // Input: a URL about to be requested. Output: why it may not be, checked before any DNS
    // (ssrf-protection.ts:216-265). A hostname's addresses are checked when it's resolved.
    pub fn validate(&self, url: &Url) -> Result<(), String> {
        if url.scheme() != "http" && url.scheme() != "https" {
            return Err("Only HTTP and HTTPS URLs can be fetched remotely".to_string());
        }
        let hostname = normalize_hostname(url.host_str().unwrap_or(""));
        if hostname.is_empty() {
            return Err("URL must include a hostname".to_string());
        }
        if hostname == "localhost" || hostname.ends_with(".localhost") {
            return Err(format!("Blocked internal hostname: {hostname}"));
        }
        if self.domain_deny.iter().any(|entry| domain_matches(&hostname, entry)) {
            return Err(format!("Blocked hostname by fetch_content domain policy: {hostname}"));
        }
        if !self.domain_allow.is_empty() && !self.domain_allow.iter().any(|entry| domain_matches(&hostname, entry)) {
            return Err(format!("Hostname not allowed by fetch_content domain policy: {hostname}"));
        }
        if let Ok(address) = hostname.parse::<IpAddr>() {
            check_address(address, &hostname, &self.allow)?;
        }
        Ok(())
    }

    // GET a URL, following up to five redirects and checking every hop (ssrf-protection.ts:267-302).
    // Output: the final response and the URL it came from.
    pub async fn get(&self, url: &Url, headers: &HeaderMap) -> Result<(reqwest::Response, Url), String> {
        let mut current = url.clone();
        self.validate(&current)?;
        for redirects in 0..=MAX_REDIRECTS {
            let response = self.client.get(current.clone()).headers(headers.clone()).send().await.map_err(|error| error_text(&error))?;
            if !REDIRECT_STATUSES.contains(&response.status().as_u16()) {
                return Ok((response, current));
            }
            let Some(location) = response.headers().get(LOCATION).and_then(|value| value.to_str().ok()) else {
                return Ok((response, current));
            };
            if redirects == MAX_REDIRECTS {
                break;
            }
            let next = current.join(location).map_err(|_| "Invalid URL".to_string())?;
            self.validate(&next)?;
            current = next;
        }
        Err(format!("Too many redirects fetching {current}"))
    }
}

// What a failed request says: a resolver refusal word for word, anything else as Node's fetch
// would put it ("fetch failed") with the innermost cause.
pub fn error_text(error: &reqwest::Error) -> String {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    let mut innermost = error.to_string();
    while let Some(current) = source {
        if let Some(refusal) = current.downcast_ref::<Refusal>() {
            return refusal.0.clone();
        }
        innermost = current.to_string();
        source = current.source();
    }
    if error.is_timeout() { "The operation was aborted.".to_string() } else { format!("fetch failed: {innermost}") }
}

// Read a body, stopping once it passes `max_bytes` (extract.ts:1214-1251). Input: the response,
// the cap, and the error message for going over it.
pub async fn read_limited(mut response: reqwest::Response, max_bytes: usize, too_large: impl Fn() -> String) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| error_text(&error))? {
        if body.len() + chunk.len() > max_bytes {
            return Err(too_large());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn charset_param(content_type: &str) -> Option<&str> {
    let lower = content_type.to_ascii_lowercase();
    let at = lower.find("charset")?;
    let rest = content_type[at + "charset".len()..].trim_start();
    let rest = rest.strip_prefix('=')?.trim_start().trim_start_matches(['"', '\'']);
    let end = rest.find(|c: char| c == ';' || c == '"' || c == '\'' || c.is_whitespace()).unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

// <meta charset="x"> or <meta http-equiv="Content-Type" content="...; charset=x"> in a page's
// first kilobyte.
fn meta_charset(bytes: &[u8]) -> Option<String> {
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(CHARSET_PRESCAN_BYTES)]).to_ascii_lowercase();
    let mut from = 0;
    while let Some(at) = head[from..].find("<meta") {
        let start = from + at;
        let end = head[start..].find('>').map(|offset| start + offset).unwrap_or(head.len());
        let tag = &head[start..end];
        if let Some(charset) = charset_param(tag) {
            return Some(charset.to_string());
        }
        from = end;
    }
    None
}

// Bytes to text by the Content-Type's charset (TextDecoder, extract.ts:1191-1199), else for HTML
// by its <meta charset>, else UTF-8. A matching BOM is dropped and bad bytes become U+FFFD.
pub fn decode_text(bytes: &[u8], content_type: &str, is_html: bool) -> String {
    let label = charset_param(content_type).map(str::to_string).or_else(|| if is_html { meta_charset(bytes) } else { None });
    let encoding = label.and_then(|label| encoding_rs::Encoding::for_label(label.as_bytes())).unwrap_or(encoding_rs::UTF_8);
    encoding.decode_with_bom_removal(bytes).0.into_owned()
}

pub fn header_value(text: &str) -> HeaderValue {
    HeaderValue::from_str(text).unwrap_or_else(|_| HeaderValue::from_static(""))
}
