// The byte-level BPE tokenizer of tokenizer.json, as Hugging Face's tokenizers runs it for LFM3: split the text
// with the pre-tokenizer's pattern, map each piece's bytes to GPT-2's printable characters, then take the piece
// whole if the vocabulary has it (ignore_merges) or merge it up pair by pair in merge-rank order.

use crate::tensor::Result;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;

pub struct Tokenizer {
    ids: HashMap<String, u32>,
    // id → piece, kept sparse so that a file naming a few large ids costs only those pieces
    pieces: HashMap<u32, String>,
    // (left, right) → (rank, merged id)
    merges: HashMap<(u32, u32), (usize, u32)>,
    // added tokens decode to their text as is; the special ones not at all
    added: HashMap<u32, (String, bool)>,
    split: Regex,
    byte_char: Vec<char>,
    char_byte: HashMap<char, u8>,
}

impl Tokenizer {
    // `vocabulary` is the model's embedding rows (each one a real row of the checkpoint): every id must be one
    // of them.
    pub fn load(json: &str, vocabulary: usize) -> Result<Tokenizer> {
        let config: Value = serde_json::from_str(json).map_err(|e| format!("tokenizer.json: {e}"))?;
        let model = &config["model"];
        if model["type"] != "BPE" || model["ignore_merges"] != true {
            return Err("tokenizer.json is not the byte-level BPE LFM3 uses".into());
        }
        let id = |value: &Value| {
            value
                .as_u64()
                .filter(|&id| id < vocabulary as u64)
                .map(|id| id as u32)
                .ok_or(format!("tokenizer.json: token id {value} is not below the model's {vocabulary} embeddings"))
        };
        let mut ids: HashMap<String, u32> = HashMap::new();
        for (piece, value) in model["vocab"].as_object().ok_or("tokenizer.json has no vocabulary")? {
            ids.insert(piece.clone(), id(value)?);
        }
        let mut added = HashMap::new();
        for token in config["added_tokens"].as_array().into_iter().flatten() {
            let (id, text) = (id(&token["id"])?, token["content"].as_str().unwrap_or(""));
            ids.insert(text.to_string(), id);
            added.insert(id, (text.to_string(), token["special"] == true));
        }
        let pieces = ids.iter().map(|(piece, &id)| (id, piece.clone())).collect();
        let mut merges = HashMap::new();
        for (rank, pair) in model["merges"].as_array().ok_or("tokenizer.json has no merges")?.iter().enumerate() {
            // merges are ["left", "right"] pairs, or "left right" strings in older files
            let (left, right) = match pair {
                Value::Array(parts) => match &parts[..] {
                    [Value::String(left), Value::String(right)] => (left.as_str(), right.as_str()),
                    _ => return Err(format!("tokenizer.json: merge {rank} is {pair}, not two strings")),
                },
                Value::String(text) => text.split_once(' ').unwrap_or((text, "")),
                _ => continue,
            };
            if let (Some(&a), Some(&b), Some(&merged)) =
                (ids.get(left), ids.get(right), ids.get(&format!("{left}{right}")))
            {
                merges.entry((a, b)).or_insert((rank, merged));
            }
        }
        // The pre-tokenizer pattern without its one lookahead, `\s+(?!\S)`, which the regex crate cannot run;
        // `encode` gives that branch's behaviour back by hand.
        let pattern = config["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
            .as_str()
            .ok_or("tokenizer.json has no split pattern")?
            .replace(r"\s+(?!\S)|", "");
        // GPT-2's byte-to-character map: printable Latin-1 bytes stand for themselves, the rest move to 256 and up
        let printable = |b: u8| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
        let mut next = 256;
        let byte_char: Vec<char> = (0..=255u8)
            .map(|b| {
                if printable(b) {
                    b as char
                } else {
                    next += 1;
                    char::from_u32(next - 1).unwrap()
                }
            })
            .collect();
        let char_byte = byte_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let split = Regex::new(&pattern).map_err(|e| format!("tokenizer pattern: {e}"))?;
        Ok(Tokenizer { ids, pieces, merges, added, split, byte_char, char_byte })
    }

    // The id of one exact token, such as "<|im_start|>".
    pub fn id(&self, token: &str) -> Result<u32> {
        self.ids.get(token).copied().ok_or(format!("the tokenizer has no {token}"))
    }

    // The ids of plain text. Added-token markup in the text is not interpreted: it is encoded as the
    // characters it is made of, so text cannot open or close turns. More than `max` ids is refused as soon as
    // the encoding passes it, and so is a pattern that matches nothing, which would never move on.
    pub fn encode(&self, text: &str, max: usize) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut at = 0;
        while let Some(found) = self.split.find_at(text, at) {
            let piece = found.as_str();
            if piece.is_empty() {
                return Err(format!("the tokenizer's split pattern matches nothing at byte {at}"));
            }
            // `\s+(?!\S)` takes a run of spaces (with no line break, which an earlier branch takes) up to but not
            // including the last space before a word, so that space starts the word's own piece.
            let mut end = found.end();
            if end < text.len()
                && piece.chars().all(char::is_whitespace)
                && !piece.ends_with(['\r', '\n'])
                && let Some((last, _)) = piece.char_indices().last().filter(|&(last, _)| last > 0)
            {
                end = found.start() + last;
            }
            self.encode_piece(&text[found.start()..end], &mut out);
            if out.len() > max {
                return Err(format!("the text is over the budget of {max} tokens"));
            }
            at = end;
        }
        Ok(out)
    }

    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let mapped: String = piece.bytes().map(|b| self.byte_char[b as usize]).collect();
        if let Some(&id) = self.ids.get(&mapped) {
            out.push(id);
            return;
        }
        let mut parts: Vec<u32> = mapped.chars().filter_map(|c| self.ids.get(&c.to_string()).copied()).collect();
        // merge the adjacent pair with the lowest rank (the leftmost on ties) until no pair has a merge
        loop {
            let best = (0..parts.len().saturating_sub(1))
                .filter_map(|i| self.merges.get(&(parts[i], parts[i + 1])).map(|&(rank, merged)| (rank, i, merged)))
                .min();
            let Some((_, i, merged)) = best else { break };
            parts[i] = merged;
            parts.remove(i + 1);
        }
        out.extend(parts);
    }

    // The text of `ids` without special tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            match self.added.get(&id) {
                Some((_, true)) => {}
                Some((text, false)) => bytes.extend(text.bytes()),
                // the language model's head is wider than the tokenizer's vocabulary; ids past it have no text
                None => bytes.extend(
                    self.pieces.get(&id).into_iter().flat_map(|p| p.chars()).filter_map(|c| self.char_byte.get(&c)),
                ),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
