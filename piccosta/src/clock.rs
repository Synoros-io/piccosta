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

// The local time now, in the C library's time zone (TZ, else /etc/localtime). Output: the time as
// "YYYY-MM-DDTHH:MM:SS.mmm" with no zone, and the zone's offset from UTC in seconds.
pub fn local_now() -> (String, i64) {
    // SAFETY: localtime_r only fills the struct it's given.
    #[cfg(unix)]
    let offset_seconds = unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut local: libc::tm = std::mem::zeroed();
        libc::localtime_r(&now, &mut local);
        local.tm_gmtoff
    };
    #[cfg(windows)]
    let offset_seconds = unsafe {
        use windows_sys::Win32::System::{Time::*, SystemServices::{TIME_ZONE_ID_DAYLIGHT, TIME_ZONE_ID_STANDARD}};
        let mut zone: DYNAMIC_TIME_ZONE_INFORMATION = std::mem::zeroed();
        let state = GetDynamicTimeZoneInformation(&mut zone);
        // Windows' bias is UTC minus local, in minutes; Unix's offset is local minus UTC.
        let seasonal = match state { TIME_ZONE_ID_DAYLIGHT => zone.DaylightBias, TIME_ZONE_ID_STANDARD => zone.StandardBias, _ => 0 };
        -i64::from(zone.Bias + seasonal) * 60
    };
    let local = iso_from_ms(now_ms() + offset_seconds * MILLISECONDS_PER_SECOND);
    (local.trim_end_matches('Z').to_string(), offset_seconds)
}

// An original message time in the machine's local zone, with the offset at that time, not today's
// (daylight saving may have changed). Missing session times are zero; don't give them a date.
pub fn dated(epoch_ms: i64) -> String {
    if epoch_ms <= 0 { return "date unknown".into(); }
    #[cfg(unix)]
    let offset = unsafe {
        let seconds: libc::time_t = epoch_ms.div_euclid(MILLISECONDS_PER_SECOND) as _;
        let mut local: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&seconds, &mut local).is_null() { return "date unknown".into(); }
        local.tm_gmtoff
    };
    #[cfg(windows)]
    let offset = unsafe {
        use windows_sys::Win32::{Foundation::{FILETIME, SYSTEMTIME}, System::Time::*};
        // Windows FILETIME counts 100 ns ticks since 1601; Unix counts milliseconds since 1970.
        const UNIX_EPOCH_TICKS: u64 = 116_444_736_000_000_000;
        const TICKS_PER_MILLISECOND: u64 = 10_000;
        let ticks = epoch_ms as u64 * TICKS_PER_MILLISECOND + UNIX_EPOCH_TICKS;
        let utc = FILETIME { dwLowDateTime: ticks as u32, dwHighDateTime: (ticks >> 32) as u32 };
        let mut universal: SYSTEMTIME = std::mem::zeroed();
        let mut local: SYSTEMTIME = std::mem::zeroed();
        let mut shifted: FILETIME = std::mem::zeroed();
        if FileTimeToSystemTime(&utc, &mut universal) == 0
            || SystemTimeToTzSpecificLocalTimeEx(std::ptr::null(), &universal, &mut local) == 0
            || SystemTimeToFileTime(&local, &mut shifted) == 0 { return "date unknown".into(); }
        let shifted = (u64::from(shifted.dwHighDateTime) << 32) | u64::from(shifted.dwLowDateTime);
        (i128::from(shifted) - i128::from(ticks)) as i64 / (TICKS_PER_MILLISECOND as i64 * MILLISECONDS_PER_SECOND)
    };
    let offset = offset as i64;
    let local = iso_from_ms(epoch_ms + offset * MILLISECONDS_PER_SECOND);
    let minutes = offset.unsigned_abs() / 60;
    format!("{}{sign}{:02}:{:02}", local.trim_end_matches('Z'), minutes / 60, minutes % 60, sign = if offset < 0 { '-' } else { '+' })
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
