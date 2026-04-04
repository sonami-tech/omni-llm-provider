use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::models::normalize_model_name;
use crate::subprocess::ndjson::ResultMessage;

// ── redb table definitions (persistent across restarts) ───────────

const TOTAL_REQUESTS: TableDefinition<&str, u64> = TableDefinition::new("total_requests");
const REQUESTS_BY_MODEL: TableDefinition<&str, u64> = TableDefinition::new("requests_by_model");
const TOTAL_ERRORS: TableDefinition<&str, u64> = TableDefinition::new("total_errors");
const TOKENS_BY_MODEL: TableDefinition<&str, &[u8]> = TableDefinition::new("tokens_by_model");
const TOTAL_KEY: &str = "total";

const MAX_LATENCY_SAMPLES: usize = 100;
const MAX_RECENT_ERRORS: usize = 50;

// ── Serializable token stats for redb storage ─────────────────────

#[derive(Serialize, Deserialize, Default, Clone)]
struct TokenStats {
	input_tokens: u64,
	output_tokens: u64,
	cache_read_input_tokens: u64,
	cache_creation_input_tokens: u64,
}

// ── Volatile types ────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct ErrorRecord {
	pub timestamp: String,
	pub model: String,
	pub message: String,
}

// ── Stats snapshot for JSON/HTML rendering ─────────────────────────

#[derive(Serialize)]
pub struct StatsSnapshot {
	pub uptime_seconds: u64,
	pub total_requests: u64,
	pub active_requests: u64,
	pub errors: u64,
	pub total_input_tokens: u64,
	pub total_output_tokens: u64,
	pub total_cache_read_input_tokens: u64,
	pub total_cache_creation_input_tokens: u64,
	pub models: HashMap<String, ModelStats>,
	pub recent_errors: Vec<ErrorRecord>,
}

#[derive(Serialize)]
pub struct ModelStats {
	pub requests: u64,
	pub avg_ttft_ms: f64,
	pub avg_duration_ms: f64,
	pub input_tokens: u64,
	pub output_tokens: u64,
	pub cache_read_input_tokens: u64,
	pub cache_creation_input_tokens: u64,
}

// ── Main Stats struct ─────────────────────────────────────────────

pub struct Stats {
	db: Database,
	active_requests: AtomicU64,
	started_at: Instant,
	recent_errors: Mutex<VecDeque<ErrorRecord>>,
	ttft_samples: Mutex<HashMap<String, VecDeque<f64>>>,
	duration_samples: Mutex<HashMap<String, VecDeque<f64>>>,
}

impl Stats {
	pub fn open(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
		if let Some(parent) = path.as_ref().parent() {
			std::fs::create_dir_all(parent)?;
		}
		let db = Database::create(path)?;

		// Create tables on first access.
		let write_txn = db.begin_write()?;
		let _ = write_txn.open_table(TOTAL_REQUESTS);
		let _ = write_txn.open_table(REQUESTS_BY_MODEL);
		let _ = write_txn.open_table(TOTAL_ERRORS);
		let _ = write_txn.open_table(TOKENS_BY_MODEL);
		write_txn.commit()?;

		Ok(Self {
			db,
			active_requests: AtomicU64::new(0),
			started_at: Instant::now(),
			recent_errors: Mutex::new(VecDeque::new()),
			ttft_samples: Mutex::new(HashMap::new()),
			duration_samples: Mutex::new(HashMap::new()),
		})
	}

	/// Increment persistent request count (total + per-model).
	pub fn record_request(&self, model: &str) {
		if let Ok(write_txn) = self.db.begin_write() {
			if let Ok(mut table) = write_txn.open_table(TOTAL_REQUESTS) {
				let current = table.get(TOTAL_KEY).ok().flatten().map(|v| v.value()).unwrap_or(0);
				let _ = table.insert(TOTAL_KEY, current + 1);
			}
			if let Ok(mut table) = write_txn.open_table(REQUESTS_BY_MODEL) {
				let current = table.get(model).ok().flatten().map(|v| v.value()).unwrap_or(0);
				let _ = table.insert(model, current + 1);
			}
			let _ = write_txn.commit();
		}
	}

	/// Record successful completion metrics.
	pub fn record_completion(
		&self,
		model: &str,
		ttft_ms: Option<f64>,
		duration_ms: f64,
		result: &ResultMessage,
	) {
		// In-memory TTFT and duration samples.
		if let Some(ttft) = ttft_ms
			&& let Ok(mut samples) = self.ttft_samples.lock()
		{
			push_sample(samples.entry(model.to_string()).or_default(), ttft);
		}
		if let Ok(mut samples) = self.duration_samples.lock() {
			push_sample(samples.entry(model.to_string()).or_default(), duration_ms);
		}

		// Persistent token stats.
		if let Some(mu) = &result.model_usage
			&& let Ok(write_txn) = self.db.begin_write()
		{
			if let Ok(mut table) = write_txn.open_table(TOKENS_BY_MODEL) {
				for (raw_name, usage) in mu {
					let normalized = normalize_model_name(raw_name);
					let mut stats: TokenStats = table
						.get(normalized.as_ref())
						.ok()
						.flatten()
						.and_then(|v| serde_json::from_slice(v.value()).ok())
						.unwrap_or_default();

					stats.input_tokens += usage.input_tokens.unwrap_or(0);
					stats.output_tokens += usage.output_tokens.unwrap_or(0);
					stats.cache_read_input_tokens +=
						usage.cache_read_input_tokens.unwrap_or(0);
					stats.cache_creation_input_tokens +=
						usage.cache_creation_input_tokens.unwrap_or(0);

					if let Ok(bytes) = serde_json::to_vec(&stats) {
						let _ = table.insert(normalized.as_ref(), bytes.as_slice());
					}
				}
			}
			let _ = write_txn.commit();
		}
	}

	/// Record an error (persistent count + in-memory recent).
	pub fn record_error(&self, model: &str, message: &str) {
		if let Ok(write_txn) = self.db.begin_write() {
			if let Ok(mut table) = write_txn.open_table(TOTAL_ERRORS) {
				let current = table.get(TOTAL_KEY).ok().flatten().map(|v| v.value()).unwrap_or(0);
				let _ = table.insert(TOTAL_KEY, current + 1);
			}
			let _ = write_txn.commit();
		}

		if let Ok(mut errors) = self.recent_errors.lock() {
			if errors.len() >= MAX_RECENT_ERRORS {
				errors.pop_front();
			}
			errors.push_back(ErrorRecord {
				timestamp: chrono_now(),
				model: model.to_string(),
				message: message.to_string(),
			});
		}
	}

	pub fn increment_active(&self) {
		self.active_requests.fetch_add(1, Ordering::Relaxed);
	}

	pub fn decrement_active(&self) {
		self.active_requests.fetch_sub(1, Ordering::Relaxed);
	}

	pub fn active_count(&self) -> u64 {
		self.active_requests.load(Ordering::Relaxed)
	}

	pub fn uptime_secs(&self) -> u64 {
		self.started_at.elapsed().as_secs()
	}

	/// Take a snapshot of all stats for JSON/HTML rendering.
	pub fn snapshot(&self) -> StatsSnapshot {
		let uptime_seconds = self.started_at.elapsed().as_secs();
		let active_requests = self.active_count();

		// Read persistent data.
		let mut total_requests = 0u64;
		let mut total_errors = 0u64;
		let mut requests_by_model: HashMap<String, u64> = HashMap::new();
		let mut tokens_by_model: HashMap<String, TokenStats> = HashMap::new();

		if let Ok(read_txn) = self.db.begin_read() {
			if let Ok(table) = read_txn.open_table(TOTAL_REQUESTS)
				&& let Ok(Some(v)) = table.get(TOTAL_KEY)
			{
				total_requests = v.value();
			}
			if let Ok(table) = read_txn.open_table(TOTAL_ERRORS)
				&& let Ok(Some(v)) = table.get(TOTAL_KEY)
			{
				total_errors = v.value();
			}
			if let Ok(table) = read_txn.open_table(REQUESTS_BY_MODEL)
				&& let Ok(iter) = table.iter()
			{
				for entry in iter.flatten() {
					let (k, v) = entry;
					requests_by_model.insert(k.value().to_string(), v.value());
				}
			}
			if let Ok(table) = read_txn.open_table(TOKENS_BY_MODEL)
				&& let Ok(iter) = table.iter()
			{
				for entry in iter.flatten() {
					let (k, v) = entry;
					if let Ok(stats) = serde_json::from_slice::<TokenStats>(v.value()) {
						tokens_by_model.insert(k.value().to_string(), stats);
					}
				}
			}
		}

		// Combine into per-model stats.
		let mut all_model_names: std::collections::HashSet<String> = requests_by_model.keys().cloned().collect();
		for k in tokens_by_model.keys() {
			all_model_names.insert(k.clone());
		}

		let ttft = self.ttft_samples.lock().unwrap_or_else(|e| e.into_inner());
		let dur = self.duration_samples.lock().unwrap_or_else(|e| e.into_inner());

		let mut models = HashMap::new();
		let mut total_input = 0u64;
		let mut total_output = 0u64;
		let mut total_cache_read = 0u64;
		let mut total_cache_creation = 0u64;

		for name in &all_model_names {
			let ts = tokens_by_model.get(name).cloned().unwrap_or_default();
			total_input += ts.input_tokens;
			total_output += ts.output_tokens;
			total_cache_read += ts.cache_read_input_tokens;
			total_cache_creation += ts.cache_creation_input_tokens;

			let avg_ttft = ttft.get(name).map(avg).unwrap_or(0.0);
			let avg_duration = dur.get(name).map(avg).unwrap_or(0.0);

			models.insert(
				name.clone(),
				ModelStats {
					requests: *requests_by_model.get(name).unwrap_or(&0),
					avg_ttft_ms: (avg_ttft * 10.0).round() / 10.0,
					avg_duration_ms: (avg_duration * 10.0).round() / 10.0,
					input_tokens: ts.input_tokens,
					output_tokens: ts.output_tokens,
					cache_read_input_tokens: ts.cache_read_input_tokens,
					cache_creation_input_tokens: ts.cache_creation_input_tokens,
				},
			);
		}

		let recent_errors = self
			.recent_errors
			.lock()
			.unwrap_or_else(|e| e.into_inner())
			.iter()
			.rev()
			.take(10)
			.cloned()
			.collect();

		StatsSnapshot {
			uptime_seconds,
			total_requests,
			active_requests,
			errors: total_errors,
			total_input_tokens: total_input,
			total_output_tokens: total_output,
			total_cache_read_input_tokens: total_cache_read,
			total_cache_creation_input_tokens: total_cache_creation,
			models,
			recent_errors,
		}
	}
}

/// Active request guard — decrements on drop.
pub struct ActiveRequestGuard<'a> {
	stats: &'a Stats,
}

impl<'a> ActiveRequestGuard<'a> {
	pub fn new(stats: &'a Stats) -> Self {
		stats.increment_active();
		Self { stats }
	}
}

impl Drop for ActiveRequestGuard<'_> {
	fn drop(&mut self) {
		self.stats.decrement_active();
	}
}

fn push_sample(buf: &mut VecDeque<f64>, value: f64) {
	if buf.len() >= MAX_LATENCY_SAMPLES {
		buf.pop_front();
	}
	buf.push_back(value);
}

fn avg(samples: &VecDeque<f64>) -> f64 {
	if samples.is_empty() {
		0.0
	} else {
		samples.iter().sum::<f64>() / samples.len() as f64
	}
}

fn chrono_now() -> String {
	use std::time::{SystemTime, UNIX_EPOCH};
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
