// English text for speech, not display. The checkpoint often skips digits or reads punctuation
// literally; spell those out before tokenizing. Plain prose takes the borrowed, unchanged path.
use regex::{Captures, Regex};
use std::{borrow::Cow, sync::LazyLock};

// English place values and irregular words; these aren't model or deployment settings.
const SMALL: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES: [(u64, &str); 6] = [
    (1_000_000_000_000_000_000, "quintillion"),
    (1_000_000_000_000_000, "quadrillion"),
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];
const ORDINALS: [(&str, &str); 13] = [
    ("one", "first"),
    ("two", "second"),
    ("three", "third"),
    ("four", "fourth"),
    ("five", "fifth"),
    ("six", "sixth"),
    ("seven", "seventh"),
    ("eight", "eighth"),
    ("nine", "ninth"),
    ("ten", "tenth"),
    ("eleven", "eleventh"),
    ("twelve", "twelfth"),
    ("zero", "zeroth"),
];
const UNITS: [(&str, &str, &str); 25] = [
    ("kg", "kilogram", "kilograms"),
    ("km", "kilometer", "kilometers"),
    ("cm", "centimeter", "centimeters"),
    ("mm", "millimeter", "millimeters"),
    ("m", "meter", "meters"),
    ("g", "gram", "grams"),
    ("ft", "foot", "feet"),
    ("lb", "pound", "pounds"),
    ("lbs", "pound", "pounds"),
    ("ml", "milliliter", "milliliters"),
    ("mL", "milliliter", "milliliters"),
    ("L", "liter", "liters"),
    ("mph", "mile per hour", "miles per hour"),
    ("kph", "kilometer per hour", "kilometers per hour"),
    ("Hz", "hertz", "hertz"),
    ("kHz", "kilohertz", "kilohertz"),
    ("MHz", "megahertz", "megahertz"),
    ("GB", "gigabyte", "gigabytes"),
    ("MB", "megabyte", "megabytes"),
    ("s", "second", "seconds"),
    ("ms", "millisecond", "milliseconds"),
    ("°C", "degree Celsius", "degrees Celsius"),
    ("°F", "degree Fahrenheit", "degrees Fahrenheit"),
    ("%", "percent", "percent"),
    ("°", "degree", "degrees"),
];

// A single pass keeps generated words from being interpreted again. Units are accepted only
// after a number, so ordinary words and abbreviations in prose are left alone.
static TOKENS: LazyLock<Regex> = LazyLock::new(|| {
    let units = UNITS
        .iter()
        .map(|(unit, _, _)| {
            format!("{}{}", regex::escape(unit), if unit.ends_with(char::is_alphabetic) { r"\b" } else { "" })
        })
        .collect::<Vec<_>>()
        .join("|");
    Regex::new(&format!(r"(?P<time>\b[0-9]{{1,2}}:[0-9]{{2}})(?:\s*(?P<ampm>(?i:am|pm))\b)?|(?P<currency>[$£€])?(?P<number>[+-]?(?:[0-9]{{1,3}}(?:,[0-9]{{3}})+|[0-9]+)(?:\.[0-9]+)?|[+-]?\.[0-9]+)(?P<ordinal>st\b|nd\b|rd\b|th\b)?(?:\s*(?P<unit>{units}))?|(?P<symbol>[\[\](){{}}&+%=<>#*/°$£€–—])"))
        .expect("English speech tokens")
});

// Nonnegative integer to American English. Large integers that don't fit u64 are read digit by
// digit rather than rounded through floating point.
fn integer(text: &str) -> String {
    fn words(value: u64) -> String {
        if value < 20 {
            return SMALL[value as usize].into();
        }
        let (head, rest) = if value < 100 {
            (TENS[(value / 10) as usize].to_string(), value % 10)
        } else if value < 1_000 {
            (format!("{} hundred", SMALL[(value / 100) as usize]), value % 100)
        } else {
            let &(scale, name) = SCALES.iter().find(|(scale, _)| value >= *scale).unwrap();
            (format!("{} {name}", words(value / scale)), value % scale)
        };
        if rest == 0 { head } else { format!("{head} {}", words(rest)) }
    }
    text.parse().map(words).unwrap_or_else(|_| digits(text))
}

fn digits(text: &str) -> String {
    text.bytes().filter(u8::is_ascii_digit).map(|digit| SMALL[(digit - b'0') as usize]).collect::<Vec<_>>().join(" ")
}

fn number(text: &str, year: bool) -> String {
    let clean = text.replace(',', "");
    let (sign, clean) = match clean.strip_prefix('-') {
        Some(rest) => ("minus ", rest),
        None => (if clean.starts_with('+') { "plus " } else { "" }, clean.trim_start_matches('+')),
    };
    let clean = if clean.is_empty() { "0" } else { clean };
    let words = if let Some((whole, fraction)) = clean.split_once('.') {
        format!("{} point {}", integer(if whole.is_empty() { "0" } else { whole }), digits(fraction))
    } else if year
        && !text.contains(',')
        && clean.len() == 4
        && ("1000"..="2099").contains(&clean)
        && !clean.starts_with("200")
    {
        let (century, rest) = clean.split_at(2);
        format!(
            "{} {}",
            integer(century),
            if rest == "00" {
                "hundred".into()
            } else if rest.starts_with('0') {
                format!("oh {}", integer(rest))
            } else {
                integer(rest)
            }
        )
    } else {
        integer(clean)
    };
    format!("{sign}{words}")
}

fn token(c: &Captures<'_>) -> String {
    if let Some(time) = c.name("time") {
        let (hour, minute) = time.as_str().split_once(':').unwrap();
        let minute = if minute == "00" {
            "o'clock".into()
        } else if minute.starts_with('0') {
            format!("oh {}", integer(minute))
        } else {
            integer(minute)
        };
        let period = c
            .name("ampm")
            .map_or("", |period| if period.as_str().eq_ignore_ascii_case("am") { " A M" } else { " P M" });
        return format!("{} {minute}{period}", integer(hour));
    }
    if let Some(n) = c.name("number") {
        let text = n.as_str();
        if let Some(currency) = c.name("currency") {
            let (single, plural, cent) = match currency.as_str() {
                "£" => ("pound", "pounds", "penny"),
                "€" => ("euro", "euros", "cent"),
                _ => ("dollar", "dollars", "cent"),
            };
            let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
            let unit = if whole.replace(',', "").parse::<f64>().is_ok_and(|value| value.abs() == 1.0) {
                single
            } else {
                plural
            };
            if fraction.len() <= 2 {
                let cents = format!("{fraction:0<2}");
                let tail = if cents == "00" {
                    String::new()
                } else {
                    format!(
                        " and {} {}",
                        integer(&cents),
                        if cents == "01" {
                            cent
                        } else if cent == "penny" {
                            "pence"
                        } else {
                            "cents"
                        }
                    )
                };
                return format!("{} {unit}{tail}", number(whole, false));
            }
            return format!("{} {plural}", number(text, false));
        }
        let mut words = number(text, c.name("unit").is_none() && c.name("ordinal").is_none());
        if c.name("ordinal").is_some() {
            let start = words.rfind(' ').map_or(0, |index| index + 1);
            let last = &words[start..];
            let ordinal =
                ORDINALS.iter().find(|(word, _)| *word == last).map(|(_, word)| word.to_string()).unwrap_or_else(
                    || last.strip_suffix('y').map_or_else(|| format!("{last}th"), |stem| format!("{stem}ieth")),
                );
            words.replace_range(start.., &ordinal);
        }
        if let Some(unit) = c.name("unit") {
            let (_, single, plural) = UNITS.iter().find(|(name, _, _)| *name == unit.as_str()).unwrap();
            words.push_str(&format!(" {}", if text.parse::<f64>() == Ok(1.0) { single } else { plural }));
        }
        return words;
    }
    match c.name("symbol").unwrap().as_str() {
        "&" => " and ",
        "+" => " plus ",
        "%" => " percent ",
        "=" => " equals ",
        "<" => " less than ",
        ">" => " greater than ",
        "#" => " number ",
        "*" => " times ",
        "/" => " per ",
        "°" => " degrees ",
        "$" => " dollars ",
        "£" => " pounds ",
        "€" => " euros ",
        _ => ", ",
    }
    .into()
}

// Input: a display line. Output: the same bytes for plain prose, otherwise English speech text.
// Don't squeeze whitespace here: unchanged text must keep exactly the same tokenizer input.
pub fn english(text: &str) -> Cow<'_, str> {
    TOKENS.replace_all(text, |c: &Captures<'_>| {
        let matched = c.get(0).unwrap();
        let before = text[..matched.start()].chars().next_back();
        let after = text[matched.end()..].chars().next();
        let mut words = token(c);
        if before.is_some_and(char::is_alphanumeric) && words.starts_with(char::is_alphanumeric) {
            words.insert(0, ' ');
        }
        if after.is_some_and(char::is_alphanumeric) && words.ends_with(char::is_alphanumeric) {
            words.push(' ');
        }
        words
    })
}
