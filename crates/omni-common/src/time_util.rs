use std::time::{SystemTime, UNIX_EPOCH};

/// Return the current time as an ISO 8601 UTC timestamp (e.g., `2026-04-05T12:34:56Z`).
pub fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let (year, month, day) = days_to_date(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Return the current time of day as `HH:MM:SS`.
pub fn time_of_day_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Bound the year walk. iso_now() uses the machine clock (not attacker input),
    // but a clock set absurdly far in the future would otherwise make this loop
    // run for billions of iterations on every stats write. Clamp to year 9999;
    // every real timestamp is unaffected (the value is identical below 9999).
    const MAX_DAYS: u64 = 2_932_896; // days from 1970-01-01 to 9999-12-31
    let mut y = 1970;
    let mut remaining = days.min(MAX_DAYS);
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }
    let months = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 1;
    for days_in_month in &months {
        if remaining < *days_in_month {
            break;
        }
        remaining -= days_in_month;
        m += 1;
    }
    (y, m, remaining + 1)
}

fn is_leap(y: u64) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_dates_are_unaffected_by_the_clamp() {
        // 2026-05-28 is day 20601 since the epoch.
        assert_eq!(days_to_date(20601), (2026, 5, 28));
        assert_eq!(days_to_date(0), (1970, 1, 1));
        // 2000-02-29 (leap day) is day 11016.
        assert_eq!(days_to_date(11016), (2000, 2, 29));
    }

    #[test]
    fn far_future_day_count_is_bounded_not_spinning() {
        // A clock set absurdly far in the future must clamp to 9999-12-31 rather
        // than walking billions of year iterations.
        assert_eq!(days_to_date(u64::MAX), (9999, 12, 31));
        assert_eq!(days_to_date(1_000_000_000_000), (9999, 12, 31));
    }
}
