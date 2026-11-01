// Server-sent events: the streaming format both Responses endpoints answer in.
//
// Frames end at a blank line. A frame's `data:` lines, trimmed and joined, are one JSON event;
// `event:` lines and comments are ignored, since every payload carries its own `type`. The end of
// the body closes a last frame that lacked its blank line. `[DONE]` ends the stream, as the
// Completions SDK iterator does (openai-completions.ts:558).

use futures_util::StreamExt;
use serde_json::Value;
use std::time::Duration;

// Read a response body's events and hand each to `handle`, until `handle` says the stream is done
// (returns true), the body ends, or an error. Input: the response, how long the body may go quiet,
// and a name for error messages. Output: an error naming what went wrong, otherwise Ok.
pub async fn read_events(
    response: reqwest::Response,
    idle_timeout: Duration,
    name: &str,
    mut handle: impl FnMut(Value) -> Result<bool, String>,
) -> Result<(), String> {
    let mut body = response.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();
    let mut after_cr = false;
    loop {
        let chunk = tokio::time::timeout(idle_timeout, body.next())
            .await
            .map_err(|_| format!("{name} stream idle for {} s", idle_timeout.as_secs()))?;
        let finished = match chunk {
            Some(Ok(bytes)) => {
                // CR, LF and CRLF all end a line. Keep this across chunks so a split CRLF
                // doesn't become two line endings and dispatch a frame too early.
                for byte in bytes {
                    if byte != b'\n' || !after_cr {
                        buffer.push(if byte == b'\r' { b'\n' } else { byte });
                    }
                    after_cr = byte == b'\r';
                }
                false
            }
            Some(Err(error)) => return Err(super::http_error(error)),
            None => {
                if !String::from_utf8_lossy(&buffer).trim().is_empty() {
                    buffer.extend_from_slice(b"\n\n");
                }
                true
            }
        };
        while let Some(end) = buffer.windows(2).position(|pair| pair == b"\n\n") {
            let frame = String::from_utf8_lossy(&buffer[..end]).into_owned();
            buffer.drain(..end + 2);
            let data: Vec<&str> = frame.lines().filter_map(|line| line.strip_prefix("data:")).map(str::trim).collect();
            if data.is_empty() {
                continue;
            }
            let data = data.join("\n");
            let data = data.trim();
            if data == "[DONE]" {
                return Ok(());
            }
            if data.is_empty() {
                continue;
            }
            let event: Value = serde_json::from_str(data).map_err(|error| format!("Invalid {name} SSE JSON: {error}"))?;
            if handle(event)? {
                return Ok(());
            }
        }
        if finished {
            return Ok(());
        }
    }
}
