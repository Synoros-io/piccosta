// get_search_content's findText: the passages of stored content that mention what the model is
// looking for, each with some context, kept under a size bound.
//
// Based on pi-web-access 0.44.0 (MIT, see LICENSE.md): content-find.ts, line
// for line. Every offset and length is in UTF-16 code units, as JavaScript counts, so match
// positions, context windows and the output bound come out the same as pi-web-access's.

use regex::Regex;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

// Characters of context on each side of a match, and the most text one find returns.
const CONTEXT_CHARS: usize = 400;
const MAX_OUTPUT_CHARS: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FindMode {
    Exact,
    CaseInsensitive,
    Fuzzy,
}

impl FindMode {
    pub fn name(self) -> &'static str {
        match self {
            FindMode::Exact => "exact",
            FindMode::CaseInsensitive => "case-insensitive",
            FindMode::Fuzzy => "fuzzy",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Match {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Range {
    start: usize,
    end: usize,
}

pub struct Found {
    pub text: String,
    pub match_count: usize,
    pub returned_matches: usize,
    // Each query and how many times it matched.
    pub query_results: Vec<(String, usize)>,
}

fn units(text: &str) -> Vec<u16> {
    text.encode_utf16().collect()
}

// indexOf over UTF-16 code units, from `from`.
fn index_of(haystack: &[u16], needle: &[u16], from: usize) -> Option<usize> {
    if needle.is_empty() {
        return (from <= haystack.len()).then_some(from);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    (from..=haystack.len() - needle.len()).find(|start| haystack[*start..*start + needle.len()] == *needle)
}

// NFD, diacritics removed, lower-cased (content-find.ts normalize).
fn normalize(value: &str) -> String {
    static DIACRITIC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\p{Diacritic}").unwrap());
    let decomposed: String = value.nfd().collect();
    DIACRITIC.replace_all(&decomposed, "").to_lowercase()
}

// Whether two strings are within `maximum` edits, comparing UTF-16 code units.
fn edit_distance_within(left: &[u16], right: &[u16], maximum: usize) -> bool {
    if left.len().abs_diff(right.len()) > maximum {
        return false;
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for i in 1..=left.len() {
        let mut current = vec![i];
        let mut row_minimum = i;
        for j in 1..=right.len() {
            let value = (previous[j] + 1).min(current[j - 1] + 1).min(previous[j - 1] + usize::from(left[i - 1] != right[j - 1]));
            current.push(value);
            row_minimum = row_minimum.min(value);
        }
        if row_minimum > maximum {
            return false;
        }
        previous = current;
    }
    previous[right.len()] <= maximum
}

fn literal_matches(text: &str, query: &str, case_insensitive: bool) -> Vec<Match> {
    // Like JavaScript, the search runs on the lower-cased text and the positions are used on the
    // original; the match is the query's own length long.
    let haystack = units(&if case_insensitive { text.to_lowercase() } else { text.to_string() });
    let needle = units(&if case_insensitive { query.to_lowercase() } else { query.to_string() });
    let query_length = units(query).len();
    let mut matches = Vec::new();
    let mut found = index_of(&haystack, &needle, 0);
    while let Some(start) = found {
        matches.push(Match { start, end: start + query_length });
        found = index_of(&haystack, &needle, start + needle.len().max(1));
    }
    matches
}

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\p{L}\p{N}]+").unwrap());

// The UTF-16 offset of every byte offset that starts a char in `text`, for mapping regex matches.
fn utf16_offsets(text: &str) -> Vec<usize> {
    let mut offsets = vec![0; text.len() + 1];
    let mut unit = 0;
    for (byte, c) in text.char_indices() {
        offsets[byte] = unit;
        unit += c.len_utf16();
    }
    offsets[text.len()] = unit;
    offsets
}

fn token_limit(query_token: &[u16]) -> usize {
    if query_token.len() >= 9 {
        2
    } else if query_token.len() >= 5 {
        1
    } else {
        0
    }
}

// Paragraphs (runs of non-empty lines) that hold at least 60% of the query's words, each within a
// small edit distance; the match is the paragraph's first such word (content-find.ts fuzzyMatches).
fn fuzzy_matches(text: &str, query: &str) -> Vec<Match> {
    let query_tokens: Vec<Vec<u16>> = WORD.find_iter(&normalize(query)).map(|token| units(token.as_str())).collect();
    if query_tokens.is_empty() {
        return Vec::new();
    }
    let offsets = utf16_offsets(text);
    let mut matches = Vec::new();
    // The regex /[^\n]+(?:\n(?!\n)[^\n]+)*/g, walked by hand: a paragraph is consecutive
    // non-empty lines.
    let mut paragraphs: Vec<(usize, usize)> = Vec::new();
    let mut line_start = 0;
    let mut open: Option<(usize, usize)> = None;
    for (at, _) in text.match_indices('\n').chain(std::iter::once((text.len(), ""))) {
        let line = (line_start, at);
        if line.1 > line.0 {
            open = Some(match open {
                Some((start, _)) => (start, line.1),
                None => line,
            });
        } else if let Some(paragraph) = open.take() {
            paragraphs.push(paragraph);
        }
        line_start = at + 1;
    }
    if let Some(paragraph) = open {
        paragraphs.push(paragraph);
    }
    for (start_byte, end_byte) in paragraphs {
        let paragraph = &text[start_byte..end_byte];
        if crate::js::trim(paragraph).is_empty() {
            continue;
        }
        let tokens: Vec<(usize, &str)> = WORD.find_iter(paragraph).map(|token| (token.start(), token.as_str())).collect();
        let normalized: Vec<Vec<u16>> = tokens.iter().map(|(_, token)| units(&normalize(token))).collect();
        let close = |query_token: &Vec<u16>, candidate: &Vec<u16>| edit_distance_within(query_token, candidate, token_limit(query_token));
        let matched: Vec<&Vec<u16>> = query_tokens.iter().filter(|query_token| normalized.iter().any(|candidate| close(query_token, candidate))).collect();
        let required = if query_tokens.len() == 1 { 1 } else { (query_tokens.len() as f64 * 0.6).ceil() as usize };
        if matched.len() < required {
            continue;
        }
        let Some(first) = tokens.iter().zip(&normalized).find(|(_, candidate)| matched.iter().any(|query_token| close(query_token, candidate))) else {
            continue;
        };
        let start = offsets[start_byte + first.0 .0];
        matches.push(Match { start, end: start + first.0 .1.encode_utf16().count() });
    }
    matches
}

fn merge_ranges(ranges: &[Range]) -> Vec<Range> {
    let mut sorted = ranges.to_vec();
    sorted.sort_by(|left, right| left.start.cmp(&right.start).then(left.end.cmp(&right.end)));
    let mut merged: Vec<Range> = Vec::new();
    for range in sorted {
        match merged.last_mut() {
            Some(previous) if range.start <= previous.end => previous.end = previous.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

fn context_ranges(text_length: usize, matches: &[Match]) -> Vec<Range> {
    let ranges: Vec<Range> = matches
        .iter()
        .map(|m| Range { start: m.start.saturating_sub(CONTEXT_CHARS), end: (m.end + CONTEXT_CHARS).min(text_length) })
        .collect();
    merge_ranges(&ranges)
}

fn lower_bound(values: &[usize], target: usize) -> usize {
    values.partition_point(|value| *value < target)
}

fn upper_bound(values: &[usize], target: usize) -> usize {
    values.partition_point(|value| *value <= target)
}

fn is_js_whitespace_unit(unit: u16) -> bool {
    char::from_u32(unit as u32).is_some_and(crate::js::is_whitespace)
}

struct QueryHits {
    query: String,
    id: String,
    order: usize,
    matches: Vec<Match>,
    starts: Vec<usize>,
    ends: Vec<usize>,
}

struct Count<'a> {
    hits: &'a QueryHits,
    count: usize,
    first_start: usize,
}

struct Finder<'a> {
    text: &'a [u16],
    total_matches: usize,
    matching: Vec<QueryHits>,
    heading: String,
    legend: String,
    missing_notice: String,
    whitespace_starts: Vec<usize>,
    whitespace_ends: Vec<usize>,
    whitespace_savings: Vec<usize>,
}

impl Finder<'_> {
    // The length a range's snippet has once whitespace runs are one space and the ends trimmed.
    fn normalized_length(&self, mut start: usize, mut end: usize) -> usize {
        let first_run = upper_bound(&self.whitespace_starts, start) as isize - 1;
        if first_run >= 0 && self.whitespace_ends[first_run as usize] > start {
            start = self.whitespace_ends[first_run as usize];
        }
        if start >= end {
            return 0;
        }
        let last_run = upper_bound(&self.whitespace_starts, end - 1) as isize - 1;
        if last_run >= 0 && self.whitespace_ends[last_run as usize] >= end {
            end = self.whitespace_starts[last_run as usize];
        }
        if start >= end {
            return 0;
        }
        let first = lower_bound(&self.whitespace_starts, start);
        let last = upper_bound(&self.whitespace_ends, end);
        end - start - (self.whitespace_savings[last] - self.whitespace_savings[first])
    }

    // The queries with matches inside a range, by first match then query order.
    fn range_counts(&self, range: Range) -> Vec<Count<'_>> {
        let mut counts: Vec<Count> = self
            .matching
            .iter()
            .filter_map(|hits| {
                let first = lower_bound(&hits.starts, range.start);
                let last = upper_bound(&hits.ends, range.end);
                (last > first).then(|| Count { hits, count: last - first, first_start: hits.starts[first] })
            })
            .collect();
        counts.sort_by(|left, right| left.first_start.cmp(&right.first_start).then(left.hits.order.cmp(&right.hits.order)));
        counts
    }

    fn summary_lines(&self, returned: usize, omitted: &[String], lines_length: &mut usize, sections: Option<&mut Vec<String>>) {
        let mut tail: Vec<String> = Vec::new();
        if !self.missing_notice.is_empty() {
            tail.push(self.missing_notice.clone());
        }
        if !omitted.is_empty() {
            tail.push(format!("No representative excerpt: {}.", omitted.join(", ")));
        }
        if returned < self.total_matches {
            tail.push(format!("Showing {returned} of {} matches.", self.total_matches));
        }
        for line in &tail {
            *lines_length += 2 + crate::js::utf16_len(line);
        }
        if let Some(sections) = sections {
            sections.extend(tail);
        }
    }

    // The length formatRanges would produce, without building it (content-find.ts measure).
    fn measure(&self, ranges: &[Range], overflow: bool, omitted: &[String]) -> (usize, usize) {
        let mut length = crate::js::utf16_len(&self.heading);
        let mut returned = 0;
        if overflow && !self.legend.is_empty() {
            length += 2 + crate::js::utf16_len(&self.legend);
        }
        for (index, range) in ranges.iter().enumerate() {
            let counts = self.range_counts(*range);
            if counts.is_empty() {
                continue;
            }
            let labels_length = counts.iter().fold(2 * (counts.len() - 1), |total, count| {
                total + if overflow { count.hits.id.len() } else { crate::js::utf16_len(&count.hits.query) + 2 } + 2 + count.count.to_string().len()
            });
            let snippet_length = self.normalized_length(range.start, range.end) + usize::from(range.start > 0) + usize::from(range.end < self.text.len());
            length += 2 + (index + 1).to_string().len() + 2 + labels_length + 1 + snippet_length;
            returned += counts.iter().map(|count| count.count).sum::<usize>();
        }
        self.summary_lines(returned, omitted, &mut length, None);
        (length, returned)
    }

    fn format(&self, ranges: &[Range], overflow: bool, omitted: &[String], query_results: Vec<(String, usize)>) -> Found {
        let mut sections = vec![self.heading.clone()];
        let mut returned = 0;
        if overflow && !self.legend.is_empty() {
            sections.push(self.legend.clone());
        }
        for (index, range) in ranges.iter().enumerate() {
            let contained = self.range_counts(*range);
            if contained.is_empty() {
                continue;
            }
            let prefix = if range.start > 0 { "…" } else { "" };
            let suffix = if range.end < self.text.len() { "…" } else { "" };
            // Lowercasing can expand a character, leaving offsets past the original text. JS's
            // slice clamps those offsets, including a start beyond the end.
            let start = range.start.min(self.text.len());
            let end = range.end.min(self.text.len()).max(start);
            let raw = String::from_utf16_lossy(&self.text[start..end]);
            let mut collapsed = String::with_capacity(raw.len());
            let mut in_space = false;
            for c in raw.chars() {
                if crate::js::is_whitespace(c) {
                    if !in_space {
                        collapsed.push(' ');
                    }
                    in_space = true;
                } else {
                    collapsed.push(c);
                    in_space = false;
                }
            }
            let snippet = format!("{prefix}{}{suffix}", crate::js::trim(&collapsed));
            let counts: Vec<String> = contained
                .iter()
                .map(|count| format!("{} ×{}", if overflow { count.hits.id.clone() } else { format!("\"{}\"", count.hits.query) }, count.count))
                .collect();
            sections.push(format!("{}. {}\n{snippet}", index + 1, counts.join(", ")));
            returned += contained.iter().map(|count| count.count).sum::<usize>();
        }
        let mut ignored = 0;
        self.summary_lines(returned, omitted, &mut ignored, Some(&mut sections));
        Found { text: sections.join("\n\n"), match_count: self.total_matches, returned_matches: returned, query_results }
    }
}

// Input: stored content, the texts to find (already trimmed, at least one), and the mode.
// Output: the formatted excerpts and the counts (content-find.ts findContent).
pub fn find_content(text: &str, queries: &[String], mode: FindMode) -> Found {
    let mut normalized_queries: Vec<String> = Vec::new();
    for query in queries {
        let trimmed = crate::js::trim(query).to_string();
        if !trimmed.is_empty() && !normalized_queries.contains(&trimmed) {
            normalized_queries.push(trimmed);
        }
    }
    let occurrences: Vec<(String, Vec<Match>)> = normalized_queries
        .iter()
        .map(|query| {
            let matches = if mode == FindMode::Fuzzy { fuzzy_matches(text, query) } else { literal_matches(text, query, mode == FindMode::CaseInsensitive) };
            (query.clone(), matches)
        })
        .collect();
    let matches: Vec<Match> = occurrences.iter().flat_map(|(_, found)| found.iter().copied()).collect();
    let query_results: Vec<(String, usize)> = occurrences.iter().map(|(query, found)| (query.clone(), found.len())).collect();
    let heading = if matches.is_empty() { format!("Text matches ({}): no matches", mode.name()) } else { format!("Text matches ({})", mode.name()) };
    let missing: Vec<String> = query_results.iter().filter(|(_, count)| *count == 0).map(|(query, _)| format!("\"{query}\"")).collect();
    let matching: Vec<QueryHits> = occurrences
        .iter()
        .filter(|(_, found)| !found.is_empty())
        .enumerate()
        .map(|(index, (query, found))| QueryHits {
            query: query.clone(),
            id: format!("Q{}", index + 1),
            order: index,
            matches: found.clone(),
            starts: found.iter().map(|m| m.start).collect(),
            ends: found.iter().map(|m| m.end).collect(),
        })
        .collect();
    let text_units = units(text);
    let mut whitespace_starts = Vec::new();
    let mut whitespace_ends = Vec::new();
    let mut at = 0;
    while at < text_units.len() {
        if is_js_whitespace_unit(text_units[at]) {
            let start = at;
            while at < text_units.len() && is_js_whitespace_unit(text_units[at]) {
                at += 1;
            }
            whitespace_starts.push(start);
            whitespace_ends.push(at);
        } else {
            at += 1;
        }
    }
    let mut whitespace_savings = vec![0];
    for (start, end) in whitespace_starts.iter().zip(&whitespace_ends) {
        let last = *whitespace_savings.last().unwrap();
        whitespace_savings.push(last + end - start - 1);
    }
    let legend = if matching.is_empty() {
        String::new()
    } else {
        format!("Queries: {}", matching.iter().map(|hits| format!("{} = \"{}\"", hits.id, hits.query)).collect::<Vec<_>>().join(", "))
    };
    let finder = Finder {
        text: &text_units,
        total_matches: matches.len(),
        matching,
        heading,
        legend,
        missing_notice: if missing.is_empty() { String::new() } else { format!("No matches: {}", missing.join(", ")) },
        whitespace_starts,
        whitespace_ends,
        whitespace_savings,
    };

    let full_ranges = context_ranges(text_units.len(), &matches);
    let (full_length, full_returned) = finder.measure(&full_ranges, false, &[]);
    if full_returned == matches.len() && full_length <= MAX_OUTPUT_CHARS {
        return finder.format(&full_ranges, false, &[], query_results);
    }

    // Over the bound: give each query one witness occurrence, cheapest first, then widen witnesses
    // with context and add bare occurrences while the bound allows.
    let mut ranges: Vec<Range> = Vec::new();
    let mut omitted: Vec<String> = finder.matching.iter().map(|hits| hits.id.clone()).collect();
    if finder.measure(&ranges, true, &omitted).0 > MAX_OUTPUT_CHARS {
        let text = format!(
            "{}\n\nUnable to format bounded excerpts: query metadata exceeds {MAX_OUTPUT_CHARS} characters.\n\nShowing 0 of {} matches.",
            finder.heading,
            matches.len()
        );
        return Found { text, match_count: matches.len(), returned_matches: 0, query_results };
    }
    let mut witnesses: Vec<Match> = Vec::new();
    for hits in &finder.matching {
        let proposed_omitted: Vec<String> = omitted.iter().filter(|id| **id != hits.id).cloned().collect();
        let current_length = finder.measure(&ranges, true, &omitted).0 as isize;
        let mut selected: Option<(Match, Vec<Range>, isize)> = None;
        for witness in &hits.matches {
            let mut with_witness = ranges.clone();
            with_witness.push(Range { start: witness.start, end: witness.end });
            let proposed = merge_ranges(&with_witness);
            let length = finder.measure(&proposed, true, &proposed_omitted).0;
            let cost = length as isize - current_length;
            let better = match &selected {
                None => true,
                Some((chosen, _, chosen_cost)) => {
                    cost < *chosen_cost || (cost == *chosen_cost && (witness.start < chosen.start || (witness.start == chosen.start && witness.end < chosen.end)))
                }
            };
            if length <= MAX_OUTPUT_CHARS && better {
                selected = Some((*witness, proposed, cost));
            }
        }
        if let Some((witness, proposed, _)) = selected {
            ranges = proposed;
            omitted = proposed_omitted;
            witnesses.push(witness);
        }
    }
    for witness in &witnesses {
        let mut widened = ranges.clone();
        widened.push(Range { start: witness.start.saturating_sub(CONTEXT_CHARS), end: (witness.end + CONTEXT_CHARS).min(text_units.len()) });
        let proposed = merge_ranges(&widened);
        if finder.measure(&proposed, true, &omitted).0 <= MAX_OUTPUT_CHARS {
            ranges = proposed;
        }
    }
    let mut every = ranges.clone();
    every.extend(matches.iter().map(|m| Range { start: m.start, end: m.end }));
    let all_occurrences = merge_ranges(&every);
    if finder.measure(&all_occurrences, true, &omitted).0 <= MAX_OUTPUT_CHARS {
        ranges = all_occurrences;
    }
    finder.format(&ranges, true, &omitted, query_results)
}
