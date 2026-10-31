// The few JavaScript behaviors that show up in text a model reads, done the way JavaScript does
// them, so piccosta's messages come out with the same characters pi's do.
//
// - to_fixed: Number.prototype.toFixed. Rust's `{:.1}` rounds an exact tie to even (1.25 -> "1.2"),
//   JavaScript rounds it up ("1.3"). Sizes like 1280 bytes = 1.25 KB hit that tie.
// - number_string: String(n), which drops ".0" and switches to exponent form outside 1e-6..1e21.
// - whitespace: JavaScript's \s and trimEnd, which include U+FEFF and leave out U+0085, unlike
//   Rust's char::is_whitespace.
// - stringify: JSON.stringify, which writes 1.0 as `1` and puts integer-like object keys first.

use serde_json::{Map, Number, Value};

// Input: a finite number and how many digits after the point. Output: x.toFixed(digits).
// Works from the exact decimal value of the double, as the spec does: Rust prints every digit
// exactly when asked for enough of them (1100 covers the smallest subnormal), and we round that
// digit string half up by hand.
pub fn to_fixed(x: f64, digits: usize) -> String {
    if !x.is_finite() {
        return number_string(x);
    }
    if x.abs() >= 1e21 {
        return number_string(x);
    }
    let negative = x < 0.0;
    let exact = format!("{:.1100}", x.abs());
    let (whole, fraction) = exact.split_once('.').unwrap_or((&exact, ""));
    let mut kept: Vec<u8> = whole.bytes().chain(fraction.bytes().take(digits)).collect();
    let round_up = fraction.as_bytes().get(digits).is_some_and(|digit| *digit >= b'5');
    if round_up {
        // Carry from the last kept digit leftward; a carry out of the top adds a leading 1.
        let mut index = kept.len();
        loop {
            if index == 0 {
                kept.insert(0, b'1');
                break;
            }
            index -= 1;
            if kept[index] == b'9' {
                kept[index] = b'0';
            } else {
                kept[index] += 1;
                break;
            }
        }
    }
    let split = kept.len() - digits;
    let (whole_digits, fraction_digits) = kept.split_at(split);
    let mut out = String::new();
    // toFixed keeps the sign of a negative number even when it rounds to zero, except for -0.
    if negative && x != 0.0 {
        out.push('-');
    }
    out.push_str(std::str::from_utf8(whole_digits).unwrap_or("0"));
    if digits > 0 {
        out.push('.');
        out.push_str(std::str::from_utf8(fraction_digits).unwrap_or(""));
    }
    out
}

// Input: a number. Output: String(n) as JavaScript writes it (Number::toString, ECMA-262 6.1.6.1.20).
// Rust's `{:e}` gives the same shortest round-trip digits JavaScript picks; this only places the
// decimal point and exponent the way JavaScript does.
pub fn number_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let scientific = format!("{:e}", x.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let exponent: i64 = exponent.parse().unwrap_or(0);
    // n is where the decimal point goes: the value is 0.<digits> x 10^n.
    let n = exponent + 1;
    let k = digits.len() as i64;
    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let exponent_sign = if n - 1 < 0 { "-" } else { "+" };
        let rest = if k == 1 { String::new() } else { format!(".{}", &digits[1..]) };
        format!("{}{rest}e{exponent_sign}{}", &digits[..1], (n - 1).abs())
    };
    format!("{sign}{body}")
}

// JavaScript's WhiteSpace and LineTerminator characters: what \s matches and trim() removes.
pub fn is_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}'
            | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}'
    )
}

pub fn trim_end(text: &str) -> &str {
    text.trim_end_matches(is_whitespace)
}

pub fn trim(text: &str) -> &str {
    text.trim_matches(is_whitespace)
}

// A string's length as JavaScript counts it, in UTF-16 code units.
pub fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

// Input: a string a model sent where a number was expected. Output: Number(text), or None when
// JavaScript would give NaN. Handles what Number() takes: surrounding whitespace, the empty string
// (0), a sign, decimals and exponents, 0x/0o/0b integers, and Infinity.
pub fn parse_number(text: &str) -> Option<f64> {
    let text = trim(text);
    if text.is_empty() {
        return Some(0.0);
    }
    let radix = |prefix: &str, radix: u32| {
        text.strip_prefix(prefix)
            .or_else(|| text.strip_prefix(&prefix.to_uppercase()))
            .filter(|digits| !digits.is_empty())
            .map(|digits| u128::from_str_radix(digits, radix).ok().map(|value| value as f64))
    };
    for (prefix, base) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if let Some(parsed) = radix(prefix, base) {
            return parsed;
        }
    }
    let unsigned = text.strip_prefix(['+', '-']).unwrap_or(text);
    if unsigned == "Infinity" {
        return Some(if text.starts_with('-') { f64::NEG_INFINITY } else { f64::INFINITY });
    }
    // Rust's parser also takes "inf", "nan" and "infinity", which Number() doesn't; only digits,
    // a point and an exponent get through to it.
    let valid = !unsigned.is_empty()
        && unsigned.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'.' | b'e' | b'E' | b'+' | b'-'))
        && unsigned.bytes().any(|byte| byte.is_ascii_digit());
    if !valid {
        return None;
    }
    text.parse::<f64>().ok()
}

// The largest whole number an f64 holds exactly (2^53). A JSON integer past it is a double in
// JavaScript, so it prints the way that double does.
const LARGEST_EXACT_INTEGER: u64 = 9_007_199_254_740_992;

// A key JavaScript treats as an array index: the canonical decimal form of an integer below
// 2^32 - 1. Objects list these keys first, in ascending order (ECMA-262 OrdinaryOwnPropertyKeys).
fn array_index(key: &str) -> Option<u32> {
    let index: u32 = key.parse().ok()?;
    (index < u32::MAX && index.to_string() == key).then_some(index)
}

// Input: an object. Output: its entries in the order Object.entries and JSON.stringify walk them:
// array-index keys ascending, then every other key in insertion order.
pub fn object_entries(object: &Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut indexed: Vec<(u32, (&String, &Value))> = Vec::new();
    let mut named = Vec::new();
    for entry in object {
        match array_index(entry.0) {
            Some(index) => indexed.push((index, entry)),
            None => named.push(entry),
        }
    }
    indexed.sort_by_key(|(index, _)| *index);
    indexed.into_iter().map(|(_, entry)| entry).chain(named).collect()
}

fn number_json(number: &Number) -> String {
    let exact = |magnitude: u64| magnitude <= LARGEST_EXACT_INTEGER;
    if let Some(integer) = number.as_i64().filter(|integer| exact(integer.unsigned_abs())) {
        return integer.to_string();
    }
    if let Some(integer) = number.as_u64().filter(|integer| exact(*integer)) {
        return integer.to_string();
    }
    number_string(number.as_f64().unwrap_or(0.0))
}

fn write_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number_json(number)),
        // serde_json escapes a string exactly as JSON.stringify does: \" \\ \b \f \n \r \t, other
        // control characters as lowercase \u00XX, everything else as is.
        Value::String(text) => out.push_str(&serde_json::to_string(text).unwrap_or_default()),
        Value::Array(items) => {
            out.push('[');
            for (position, item) in items.iter().enumerate() {
                if position > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Object(object) => {
            out.push('{');
            for (position, (key, item)) in object_entries(object).into_iter().enumerate() {
                if position > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                write_json(item, out);
            }
            out.push('}');
        }
    }
}

// Input: a JSON value. Output: JSON.stringify(value), no indent, character for character.
pub fn stringify(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}
