// A command's output as it streams in, kept to a bounded size: a decoded tail for what the model
// sees, line and byte counts for the truncation notice, and the raw bytes, in memory while they're
// small and in a spill file once they pass pi's limits, so the model can read all of it later.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/tools/output-accumulator.ts.
// pi spills to $TMPDIR and never deletes the file; here the spill file goes where the caller says
// (the session's job directory) and the caller removes it when the session ends.

use super::truncate::{self, TruncatedBy, Truncation};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct Snapshot {
    pub content: String,
    pub truncation: Truncation,
    pub full_output_path: Option<PathBuf>,
}

pub struct OutputAccumulator {
    max_lines: usize,
    max_bytes: usize,
    max_rolling_bytes: usize,
    // TextDecoder's UTF-8: a character split between two reads waits for the rest, anything
    // invalid becomes U+FFFD, and a leading BOM goes.
    decoder: encoding_rs::Decoder,
    // Raw output until it spills; empty after.
    raw: Vec<u8>,
    // Called once, when the output first needs a spill file, so a command that stays small never
    // creates the directory it would go in.
    choose_spill_path: Option<Box<dyn FnOnce() -> PathBuf + Send>>,
    spill_path: PathBuf,
    spill: Option<std::fs::File>,
    spilled: bool,
    error: Option<std::io::Error>,
    tail_text: String,
    tail_starts_at_line_boundary: bool,
    total_raw_bytes: usize,
    total_decoded_bytes: usize,
    completed_lines: usize,
    total_lines: usize,
    current_line_bytes: usize,
    has_open_line: bool,
}

impl OutputAccumulator {
    // Input: the line and byte limits, and what picks the spill file if the output passes them.
    pub fn new(max_lines: usize, max_bytes: usize, choose_spill_path: Box<dyn FnOnce() -> PathBuf + Send>) -> OutputAccumulator {
        OutputAccumulator {
            max_lines,
            max_bytes,
            max_rolling_bytes: (max_bytes * 2).max(1),
            decoder: encoding_rs::UTF_8.new_decoder_with_bom_removal(),
            raw: Vec::new(),
            choose_spill_path: Some(choose_spill_path),
            spill_path: PathBuf::new(),
            spill: None,
            spilled: false,
            error: None,
            tail_text: String::new(),
            tail_starts_at_line_boundary: true,
            total_raw_bytes: 0,
            total_decoded_bytes: 0,
            completed_lines: 0,
            total_lines: 0,
            current_line_bytes: 0,
            has_open_line: false,
        }
    }

    pub fn append(&mut self, data: &[u8]) {
        self.total_raw_bytes += data.len();
        self.decode(data, false);
        if self.spill.is_some() || self.should_spill() {
            self.ensure_spill();
            if self.error.is_none() && let Some(file) = self.spill.as_mut() && let Err(error) = file.write_all(data) {
                self.error = Some(error);
            }
        } else {
            self.raw.extend_from_slice(data);
        }
    }

    pub fn finish(&mut self) {
        self.decode(&[], true);
        if self.should_spill() {
            self.ensure_spill();
        }
        self.flush_spill();
    }

    // What the model would see now: the tail cut to the limits, and the counts for the notice. A
    // truncated snapshot names its spill file only when all writes succeeded.
    pub fn snapshot(&mut self) -> Snapshot {
        let tail = truncate::truncate_tail(self.snapshot_text(), self.max_lines, self.max_bytes);
        let truncated = self.total_lines > self.max_lines || self.total_decoded_bytes > self.max_bytes;
        let truncated_by = truncated.then(|| {
            tail.truncated_by.unwrap_or(if self.total_decoded_bytes > self.max_bytes { TruncatedBy::Bytes } else { TruncatedBy::Lines })
        });
        let truncation = Truncation {
            truncated,
            truncated_by,
            total_lines: self.total_lines,
            total_bytes: self.total_decoded_bytes,
            max_lines: self.max_lines,
            max_bytes: self.max_bytes,
            ..tail
        };
        if truncated {
            self.ensure_spill();
        }
        Snapshot { content: truncation.content.clone(), truncation, full_output_path: self.spill_path().map(Path::to_path_buf) }
    }

    pub fn last_line_bytes(&self) -> usize {
        self.current_line_bytes
    }

    // Everything so far, raw, for when a command moves to the background and its log starts:
    // the bytes in memory, or the spill file's path holding them. Borrow them so a failed job
    // handoff can keep collecting output in the foreground.
    pub fn raw(&mut self) -> Result<RawOutput<'_>, String> {
        self.flush_spill();
        self.check_error()?;
        if self.spilled {
            Ok(RawOutput::File(&self.spill_path))
        } else {
            Ok(RawOutput::Memory(&self.raw))
        }
    }

    pub fn check_error(&self) -> Result<(), String> {
        match &self.error {
            Some(error) => Err(format!("Could not save full output to {}: {error}", self.spill_path.display())),
            None => Ok(()),
        }
    }

    fn flush_spill(&mut self) {
        if let Some(file) = self.spill.as_mut() && let Err(error) = file.flush() {
            self.error.get_or_insert(error);
        }
    }

    fn decode(&mut self, data: &[u8], last: bool) {
        let mut text = String::with_capacity(self.decoder.max_utf8_buffer_length(data.len()).unwrap_or(data.len() * 3 + 4));
        let _ = self.decoder.decode_to_string(data, &mut text, last);
        self.append_decoded(&text);
    }

    fn append_decoded(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.total_decoded_bytes += text.len();
        self.tail_text.push_str(text);
        if self.tail_text.len() > self.max_rolling_bytes * 2 {
            self.trim_tail();
        }
        let newlines = text.bytes().filter(|byte| *byte == b'\n').count();
        if newlines == 0 {
            self.current_line_bytes += text.len();
            self.has_open_line = true;
        } else {
            self.completed_lines += newlines;
            let after = &text[text.rfind('\n').map_or(0, |at| at + 1)..];
            self.current_line_bytes = after.len();
            self.has_open_line = !after.is_empty();
        }
        self.total_lines = self.completed_lines + usize::from(self.has_open_line);
    }

    // Keep the last max_rolling_bytes of the decoded tail, starting on a character boundary, and
    // remember whether that start is also the start of a line.
    fn trim_tail(&mut self) {
        if self.tail_text.len() <= self.max_rolling_bytes {
            return;
        }
        let mut start = self.tail_text.len() - self.max_rolling_bytes;
        while !self.tail_text.is_char_boundary(start) {
            start += 1;
        }
        if start > 0 {
            self.tail_starts_at_line_boundary = self.tail_text.as_bytes()[start - 1] == b'\n';
        }
        self.tail_text.drain(..start);
    }

    // The tail without a partial first line, when trimming may have cut into one.
    fn snapshot_text(&self) -> &str {
        if self.tail_starts_at_line_boundary {
            return &self.tail_text;
        }
        match self.tail_text.find('\n') {
            Some(at) => &self.tail_text[at + 1..],
            None => &self.tail_text,
        }
    }

    fn should_spill(&self) -> bool {
        self.total_raw_bytes > self.max_bytes || self.total_decoded_bytes > self.max_bytes || self.total_lines > self.max_lines
    }

    fn ensure_spill(&mut self) {
        if self.spilled || self.error.is_some() {
            return;
        }
        if let Some(choose) = self.choose_spill_path.take() {
            self.spill_path = choose();
        }
        // Only the user may read it (output can carry private data), and create_new never follows
        // a link someone else put at the path (output-files.ts:12-33).
        use std::os::unix::fs::OpenOptionsExt;
        let file = (|| {
            if let Some(parent) = self.spill_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&self.spill_path)?;
            file.write_all(&self.raw)?;
            Ok(file)
        })();
        match file {
            Ok(file) => {
                self.raw.clear();
                self.spilled = true;
                self.spill = Some(file);
            }
            Err(error) => self.error = Some(error),
        }
    }

    pub fn spill_path(&self) -> Option<&Path> {
        (self.spilled && self.error.is_none()).then_some(self.spill_path.as_path())
    }
}

pub enum RawOutput<'a> {
    Memory(&'a [u8]),
    File(&'a Path),
}
