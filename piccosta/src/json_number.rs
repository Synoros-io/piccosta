// JSON numbers written the way JavaScript writes them, where it matters to us.
//
// pi wrote every number with JSON.stringify, and JavaScript has one number type, so a whole number
// like a 936 ms duration or a $0 cost comes out as `936` and `0`. serde_json writes an f64 as
// `936.0`. Fields we keep as f64 (costs, durations) go through `serialize` below so whole values
// keep the integer form old session files already use. Parsing needs nothing special: serde reads
// `936` and `936.0` into an f64 the same way.

use serde::Serializer;

// The largest whole number an f64 holds exactly (2^53). Past it, printing as an integer could
// claim a precision the value doesn't have, so those values keep the float form.
const LARGEST_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;

// Write one f64. Input: the value. Output: an integer when the value is whole and exact,
// otherwise serde_json's float form. NaN and infinity can't occur in JSON we read, and
// serde_json writes them as null, which is what JSON.stringify does too.
pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if value.fract() == 0.0 && value.abs() <= LARGEST_EXACT_INTEGER {
        serializer.serialize_i64(*value as i64)
    } else {
        serializer.serialize_f64(*value)
    }
}

// The same for an optional field; callers pair it with `skip_serializing_if = "Option::is_none"`,
// so only the Some case reaches here.
pub fn serialize_option<S: Serializer>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error> {
    match value {
        Some(number) => serialize(number, serializer),
        None => serializer.serialize_none(),
    }
}
