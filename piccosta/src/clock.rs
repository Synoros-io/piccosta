// Time as session files write it: epoch milliseconds inside messages, and ISO 8601 strings in UTC
// with milliseconds ("2026-10-29T12:31:25.335Z", what JavaScript's toISOString gives) on entries
// and headers. Done by hand so no date crate is needed for two conversions.

use std::time::{SystemTime, UNIX_EPOCH};

const MILLISECONDS_PER_SECOND: i64 = 1_000;
const SECONDS_PER_DAY: i64 = 86_400;
const MILLISECONDS_PER_DAY: i64 = SECONDS_PER_DAY * MILLISECONDS_PER_SECOND;
// Days from 0000-03-01 to 1970-01-01 in the proleptic Gregorian calendar. The civil-date
// conversions below count from March 1 so a leap day falls at the end of the year (the method
// is Howard Hinnant's "days from civil").
const DAYS_FROM_YEAR_ZERO_TO_EPOCH: i64 = 719_468;
const DAYS_PER_400_YEARS: i64 = 146_097;

// The current time in epoch milliseconds. A clock set before 1970 reads as 0.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

// Calendar date of a day count since 1970-01-01. Output: (year, month 1-12, day 1-31).
fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let days = days_since_epoch + DAYS_FROM_YEAR_ZERO_TO_EPOCH;
    let era = days.div_euclid(DAYS_PER_400_YEARS);
    let day_of_era = days.rem_euclid(DAYS_PER_400_YEARS);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 { month_from_march + 3 } else { month_from_march - 9 };
    let year = year_of_era + era * 400 + if month <= 2 { 1 } else { 0 };
    (year, month, day)
}

// Day count since 1970-01-01 of a calendar date, the inverse of `civil_from_days`.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let month_from_march = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_from_march + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * DAYS_PER_400_YEARS + day_of_era - DAYS_FROM_YEAR_ZERO_TO_EPOCH
}

// An epoch-ms time as toISOString writes it. Input: epoch milliseconds. Output:
// "YYYY-MM-DDTHH:MM:SS.mmmZ".
pub fn iso_from_ms(epoch_ms: i64) -> String {
    let days = epoch_ms.div_euclid(MILLISECONDS_PER_DAY);
    let ms_of_day = epoch_ms.rem_euclid(MILLISECONDS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let hour = ms_of_day / 3_600_000;
    let minute = ms_of_day / 60_000 % 60;
    let second = ms_of_day / 1_000 % 60;
    let millisecond = ms_of_day % 1_000;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millisecond:03}Z")
}

// Parse a toISOString time back to epoch ms. Input: an ISO 8601 UTC string; the fraction may be
// missing or of any length (only milliseconds are kept). Output: None when the text isn't that
// shape. Entries pi wrote always carry the exact toISOString form.
pub fn ms_from_iso(text: &str) -> Option<i64> {
    let text = text.strip_suffix('Z')?;
    let (date, time) = text.split_once('T')?;
    let mut date_parts = date.splitn(3, '-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: i64 = date_parts.next()?.parse().ok()?;
    let day: i64 = date_parts.next()?.parse().ok()?;
    let (clock, fraction) = time.split_once('.').unwrap_or((time, "0"));
    let mut clock_parts = clock.splitn(3, ':');
    let hour: i64 = clock_parts.next()?.parse().ok()?;
    let minute: i64 = clock_parts.next()?.parse().ok()?;
    let second: i64 = clock_parts.next()?.parse().ok()?;
    // The first three digits of the fraction are the milliseconds; pad a shorter fraction.
    let millis_text: String = fraction.chars().chain("000".chars()).take(3).collect();
    let millisecond: i64 = millis_text.parse().ok()?;
    let days = days_from_civil(year, month, day);
    Some(days * MILLISECONDS_PER_DAY + ((hour * 60 + minute) * 60 + second) * MILLISECONDS_PER_SECOND + millisecond)
}
