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
	let mut y = 1970;
	let mut remaining = days;
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
