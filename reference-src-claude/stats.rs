use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Sender, SyncSender, sync_channel};
use std::time::Instant;

use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

// ── redb table definitions (persistent across restarts) ───────────

const TOTAL_REQUESTS: TableDefinition<&str, u64> = TableDefinition::new("total_requests");
const REQUESTS_BY_MODEL: TableDefinition<&str, u64> = TableDefinition::new("requests_by_model");
const TOTAL_ERRORS: TableDefinition<&str, u64> = TableDefinition::new("total_errors");
const TOKENS_BY_MODEL: TableDefinition<&str, &[u8]> = TableDefinition::new("tokens_by_model");
const REQUESTS_BY_KEY: TableDefinition<&str, u64> = TableDefinition::new("requests_by_key");
const LAST_REQUEST_AT: TableDefinition<&str, &str> = TableDefinition::new("last_request_at");
const TOTAL_KEY: &str = "total";

const MAX_RECENT_ERRORS: usize = 50;

/// Per-model rolling window for TTFT / duration averages. Bounds the in-memory
/// sample deques so a long-running process does not grow them without limit.
const MAX_SAMPLES: usize = 1000;

/// Token counts to fold into per-model persistent stats. Mirrors the fields of
/// the Anthropic usage object; kept as a plain struct so `stats` does not depend
/// on the translation layer's types.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
	pub input_tokens: u64,
	pub output_tokens: u64,
	pub cache_read_input_tokens: u64,
	pub cache_creation_input_tokens: u64,
}

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
	pub last_request_at: Option<String>,
	pub models: HashMap<String, ModelStats>,
	pub api_keys: HashMap<String, u64>,
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

// ── Persistent-write operations sent to the writer thread ─────────

/// A persistent (redb) mutation. The blocking write is performed on a dedicated
/// writer thread so request handlers never touch the disk on a Tokio worker.
enum StatOp {
	Request {
		model: String,
		api_key_id: Option<String>,
	},
	Error,
	Response {
		model: String,
		usage: TokenUsage,
	},
	/// Test/shutdown aid: reply once all prior ops have been applied.
	Flush(Sender<()>),
}

// ── Main Stats struct ─────────────────────────────────────────────

pub struct Stats {
	/// Shared with the writer thread (writes) and snapshot readers (reads).
	/// redb permits concurrent read txns alongside the single writer.
	db: std::sync::Arc<Database>,
	/// Sends persistent-write ops to the writer thread. A bounded channel applies
	/// backpressure under extreme load instead of growing memory without limit.
	writer: SyncSender<StatOp>,
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
		let db = std::sync::Arc::new(Database::create(path)?);

		// Create tables on first access.
		let write_txn = db.begin_write()?;
		let _ = write_txn.open_table(TOTAL_REQUESTS);
		let _ = write_txn.open_table(REQUESTS_BY_MODEL);
		let _ = write_txn.open_table(TOTAL_ERRORS);
		let _ = write_txn.open_table(TOKENS_BY_MODEL);
		let _ = write_txn.open_table(REQUESTS_BY_KEY);
		let _ = write_txn.open_table(LAST_REQUEST_AT);
		write_txn.commit()?;

		// Persistent writes run on a dedicated OS thread fed by a bounded channel.
		// Request handlers only enqueue (cheap, non-blocking) so a slow fsync can
		// never stall a Tokio worker, and redb's single-writer model maps cleanly
		// onto one serial writer. The bounded queue applies backpressure under
		// extreme load rather than growing unbounded.
		let (writer, rx) = sync_channel::<StatOp>(4096);
		let writer_db = db.clone();
		std::thread::Builder::new()
			.name("ccp-stats-writer".into())
			.spawn(move || writer_loop(&writer_db, rx))?;

		Ok(Self {
			db,
			writer,
			active_requests: AtomicU64::new(0),
			started_at: Instant::now(),
			recent_errors: Mutex::new(VecDeque::new()),
			ttft_samples: Mutex::new(HashMap::new()),
			duration_samples: Mutex::new(HashMap::new()),
		})
	}

	/// Enqueue a persistent-write op for the writer thread. Dropped (with a warn)
	/// only if the writer has gone away or the bounded queue is full — telemetry
	/// must never block or fail a request.
	fn enqueue(&self, op: StatOp) {
		if let Err(e) = self.writer.try_send(op) {
			tracing::warn!("stats: dropping persistent write (writer queue full or closed): {e}");
		}
	}

	/// Increment persistent request count (total + per-model + per-key).
	pub fn record_request(&self, model: &str, api_key_id: Option<&str>) {
		self.enqueue(StatOp::Request {
			model: model.to_string(),
			api_key_id: api_key_id.map(str::to_string),
		});
	}

	/// Record an error (persistent count + in-memory recent).
	pub fn record_error(&self, model: &str, message: &str) {
		self.enqueue(StatOp::Error);

		if let Ok(mut errors) = self.recent_errors.lock() {
			if errors.len() >= MAX_RECENT_ERRORS {
				errors.pop_front();
			}
			errors.push_back(ErrorRecord {
				timestamp: crate::time_util::iso_now(),
				model: model.to_string(),
				message: message.to_string(),
			});
		}
	}

	/// Record a successful completion: fold token usage into the persistent
	/// per-model totals and push duration / TTFT samples into the bounded
	/// rolling windows used for the dashboard averages.
	///
	/// `ttft_ms` is the time to first streamed token (None for non-streaming
	/// responses, where TTFT is not meaningful). `duration_ms` is the total
	/// wall-clock time for the request.
	pub fn record_response(
		&self,
		model: &str,
		usage: TokenUsage,
		ttft_ms: Option<f64>,
		duration_ms: f64,
	) {
		self.enqueue(StatOp::Response {
			model: model.to_string(),
			usage,
		});

		push_sample(&self.duration_samples, model, duration_ms);
		if let Some(ttft) = ttft_ms {
			push_sample(&self.ttft_samples, model, ttft);
		}
	}

	/// Block (up to `FLUSH_TIMEOUT`) until all persistent-write ops enqueued
	/// before this call have been applied by the writer thread. Gives a
	/// deterministic drain point for tests (read-after-write) and a bounded flush
	/// hook for graceful shutdown — a wedged disk must not hang shutdown
	/// indefinitely. The in-memory windows are already updated synchronously, so
	/// this only concerns the redb-backed counters.
	pub fn flush(&self) {
		const FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
		let (tx, rx) = std::sync::mpsc::channel();
		// `try_send` (non-blocking): if the bounded queue is momentarily full we
		// skip this flush rather than block the caller — the worst case is a few
		// un-drained counters at shutdown, never a hang.
		if self.writer.try_send(StatOp::Flush(tx)).is_ok() {
			let _ = rx.recv_timeout(FLUSH_TIMEOUT);
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

		// Read persistent data in a single transaction.
		let mut total_requests = 0u64;
		let mut total_errors = 0u64;
		let mut last_request_at: Option<String> = None;
		let mut requests_by_model: HashMap<String, u64> = HashMap::new();
		let mut tokens_by_model: HashMap<String, TokenStats> = HashMap::new();
		let mut api_keys: HashMap<String, u64> = HashMap::new();

		let read = self.db.begin_read();
		if read.is_err() {
			tracing::warn!("stats: failed to open redb read txn; snapshot may read as empty");
		}
		if let Ok(read_txn) = read {
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
			if let Ok(table) = read_txn.open_table(LAST_REQUEST_AT)
				&& let Ok(Some(v)) = table.get(TOTAL_KEY)
			{
				last_request_at = Some(v.value().to_string());
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
			if let Ok(table) = read_txn.open_table(REQUESTS_BY_KEY)
				&& let Ok(iter) = table.iter()
			{
				for entry in iter.flatten() {
					let (k, v) = entry;
					api_keys.insert(k.value().to_string(), v.value());
				}
			}
		}

		// Combine into per-model stats.
		let mut all_model_names: HashSet<String> = requests_by_model.keys().cloned().collect();
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
			total_input = total_input.saturating_add(ts.input_tokens);
			total_output = total_output.saturating_add(ts.output_tokens);
			total_cache_read = total_cache_read.saturating_add(ts.cache_read_input_tokens);
			total_cache_creation =
				total_cache_creation.saturating_add(ts.cache_creation_input_tokens);

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
			last_request_at,
			total_input_tokens: total_input,
			total_output_tokens: total_output,
			total_cache_read_input_tokens: total_cache_read,
			total_cache_creation_input_tokens: total_cache_creation,
			models,
			api_keys,
			recent_errors,
		}
	}
}

impl Drop for Stats {
	fn drop(&mut self) {
		// Drain any persistent-write ops still queued so a graceful shutdown does
		// not lose recently-recorded counters. The writer thread is a daemon that
		// exits with the process; flushing the queue is what matters here.
		self.flush();
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

/// Owned active-request guard — like [`ActiveRequestGuard`] but holds an
/// `Arc<Stats>` instead of a borrow, so it can be created before a request's
/// async setup and `move`d into a spawned task that outlives the handler. Used
/// by the streaming path so the active count covers the full request lifetime
/// (setup through stream completion), not just one or the other.
pub struct OwnedActiveRequestGuard {
	stats: std::sync::Arc<Stats>,
}

impl OwnedActiveRequestGuard {
	pub fn new(stats: std::sync::Arc<Stats>) -> Self {
		stats.increment_active();
		Self { stats }
	}
}

impl Drop for OwnedActiveRequestGuard {
	fn drop(&mut self) {
		self.stats.decrement_active();
	}
}

fn avg(samples: &VecDeque<f64>) -> f64 {
	if samples.is_empty() {
		0.0
	} else {
		samples.iter().sum::<f64>() / samples.len() as f64
	}
}

/// Writer-thread loop: drains persistent-write ops and applies each in its own
/// redb transaction. Runs on a dedicated OS thread (not the Tokio runtime), so a
/// slow fsync blocks only this thread, never a request handler. Exits when all
/// `Stats` senders are dropped (channel closed).
fn writer_loop(db: &Database, rx: std::sync::mpsc::Receiver<StatOp>) {
	// Coalesce all currently-queued ops into a single transaction per drain
	// cycle: block for the first op, then greedily pull everything already in the
	// channel. Under load this collapses many per-request fsyncs into one commit;
	// when idle it is exactly one op per commit. Flush replies are deferred until
	// after the commit so a flush still guarantees prior writes are persisted.
	while let Ok(first) = rx.recv() {
		let mut writes: Vec<StatOp> = Vec::new();
		let mut flush_replies: Vec<Sender<()>> = Vec::new();
		let mut sort = |op| match op {
			StatOp::Flush(reply) => flush_replies.push(reply),
			other => writes.push(other),
		};
		sort(first);
		while let Ok(op) = rx.try_recv() {
			sort(op);
		}

		if !writes.is_empty() {
			match db.begin_write() {
				Ok(write_txn) => {
					for op in &writes {
						apply_op_in_txn(&write_txn, op);
					}
					if let Err(e) = write_txn.commit() {
						tracing::warn!("stats: failed to commit redb write txn: {e}");
					}
				}
				Err(e) => tracing::warn!("stats: failed to open redb write txn: {e}"),
			}
		}
		// Replies after commit: a flush observed by the caller means every op
		// enqueued before it has been persisted.
		for reply in flush_replies {
			let _ = reply.send(());
		}
	}
}

/// Apply one persistent-write op within an existing write transaction. All
/// counter increments use `saturating_add` so a corrupted/extreme counter cannot
/// panic (debug) or wrap to zero (release).
fn apply_op_in_txn(write_txn: &redb::WriteTransaction, op: &StatOp) {
	match op {
		StatOp::Request { model, api_key_id } => {
			if let Ok(mut table) = write_txn.open_table(TOTAL_REQUESTS) {
				let current = table.get(TOTAL_KEY).ok().flatten().map(|v| v.value()).unwrap_or(0);
				let _ = table.insert(TOTAL_KEY, current.saturating_add(1));
			}
			if let Ok(mut table) = write_txn.open_table(REQUESTS_BY_MODEL) {
				let current = table.get(model.as_str()).ok().flatten().map(|v| v.value()).unwrap_or(0);
				let _ = table.insert(model.as_str(), current.saturating_add(1));
			}
			if let Some(key_id) = api_key_id
				&& let Ok(mut table) = write_txn.open_table(REQUESTS_BY_KEY)
			{
				let current = table.get(key_id.as_str()).ok().flatten().map(|v| v.value()).unwrap_or(0);
				let _ = table.insert(key_id.as_str(), current.saturating_add(1));
			}
			if let Ok(mut table) = write_txn.open_table(LAST_REQUEST_AT) {
				let now = crate::time_util::iso_now();
				let _ = table.insert(TOTAL_KEY, now.as_str());
			}
		}
		StatOp::Error => {
			if let Ok(mut table) = write_txn.open_table(TOTAL_ERRORS) {
				let current = table.get(TOTAL_KEY).ok().flatten().map(|v| v.value()).unwrap_or(0);
				let _ = table.insert(TOTAL_KEY, current.saturating_add(1));
			}
		}
		StatOp::Response { model, usage } => {
			if let Ok(mut table) = write_txn.open_table(TOKENS_BY_MODEL) {
				let mut stats: TokenStats = table
					.get(model.as_str())
					.ok()
					.flatten()
					.and_then(|v| serde_json::from_slice(v.value()).ok())
					.unwrap_or_default();
				stats.input_tokens = stats.input_tokens.saturating_add(usage.input_tokens);
				stats.output_tokens = stats.output_tokens.saturating_add(usage.output_tokens);
				stats.cache_read_input_tokens = stats
					.cache_read_input_tokens
					.saturating_add(usage.cache_read_input_tokens);
				stats.cache_creation_input_tokens = stats
					.cache_creation_input_tokens
					.saturating_add(usage.cache_creation_input_tokens);
				if let Ok(bytes) = serde_json::to_vec(&stats) {
					let _ = table.insert(model.as_str(), bytes.as_slice());
				}
			}
		}
		StatOp::Flush(_) => {} // separated out in writer_loop before reaching here
	}
}

/// Push a timing sample into a per-model bounded rolling window, evicting the
/// oldest sample once the window is full.
fn push_sample(samples: &Mutex<HashMap<String, VecDeque<f64>>>, model: &str, value: f64) {
	let mut map = samples.lock().unwrap_or_else(|e| e.into_inner());
	let window = map.entry(model.to_string()).or_default();
	if window.len() >= MAX_SAMPLES {
		window.pop_front();
	}
	window.push_back(value);
}

#[cfg(test)]
mod tests {
	use super::*;

	fn temp_stats() -> (Stats, std::path::PathBuf) {
		let path = std::env::temp_dir()
			.join(format!("ccp-stats-test-{}.redb", uuid::Uuid::new_v4()));
		(Stats::open(&path).unwrap(), path)
	}

	#[test]
	fn record_response_persists_tokens_and_timing_into_snapshot() {
		// Intent: the /stats dashboard must reflect real token + latency data.
		// Before this wiring these were always zero regardless of traffic.
		let (stats, path) = temp_stats();
		stats.record_request("claude-sonnet-4-6", Some("k1"));
		stats.record_response(
			"claude-sonnet-4-6",
			TokenUsage {
				input_tokens: 100,
				output_tokens: 40,
				cache_read_input_tokens: 10,
				cache_creation_input_tokens: 5,
			},
			Some(123.0),
			456.0,
		);
		// A second response accumulates and averages timing.
		stats.record_response(
			"claude-sonnet-4-6",
			TokenUsage { input_tokens: 50, output_tokens: 20, ..Default::default() },
			Some(223.0),
			544.0,
		);

		// Persistent writes are applied asynchronously on the writer thread; wait
		// for them to drain before reading the redb-backed snapshot.
		stats.flush();
		let snap = stats.snapshot();
		assert_eq!(snap.total_input_tokens, 150);
		assert_eq!(snap.total_output_tokens, 60);
		assert_eq!(snap.total_cache_read_input_tokens, 10);
		assert_eq!(snap.total_cache_creation_input_tokens, 5);

		let m = snap.models.get("claude-sonnet-4-6").expect("model present");
		assert_eq!(m.input_tokens, 150);
		assert_eq!(m.output_tokens, 60);
		// Averages of the two samples: (123+223)/2=173, (456+544)/2=500.
		assert_eq!(m.avg_ttft_ms, 173.0);
		assert_eq!(m.avg_duration_ms, 500.0);

		let _ = std::fs::remove_file(path);
	}

	#[test]
	fn record_error_counts_and_keeps_recent() {
		let (stats, path) = temp_stats();
		stats.record_error("claude-haiku-4-5", "boom");
		stats.record_error("claude-haiku-4-5", "kaboom");
		stats.flush();
		let snap = stats.snapshot();
		assert_eq!(snap.errors, 2);
		// recent_errors is newest-first in the snapshot.
		assert_eq!(snap.recent_errors.len(), 2);
		assert_eq!(snap.recent_errors[0].message, "kaboom");
		let _ = std::fs::remove_file(path);
	}

	#[test]
	fn timing_samples_are_bounded() {
		let (stats, path) = temp_stats();
		for i in 0..(MAX_SAMPLES + 50) {
			push_sample(&stats.duration_samples, "m", i as f64);
		}
		let map = stats.duration_samples.lock().unwrap();
		assert_eq!(map.get("m").unwrap().len(), MAX_SAMPLES);
		drop(map);
		let _ = std::fs::remove_file(path);
	}

	#[test]
	fn batched_ops_all_persist_through_one_flush() {
		// Many ops enqueued back-to-back are coalesced by the writer into one (or
		// few) transactions; after a single flush every one must be reflected.
		let (stats, path) = temp_stats();
		for _ in 0..25 {
			stats.record_request("m", Some("k1"));
			stats.record_response("m", TokenUsage { input_tokens: 2, output_tokens: 1, ..Default::default() }, None, 1.0);
		}
		stats.record_error("m", "boom");
		stats.flush();

		let snap = stats.snapshot();
		assert_eq!(snap.total_requests, 25);
		assert_eq!(snap.errors, 1);
		assert_eq!(snap.total_input_tokens, 50);
		assert_eq!(snap.total_output_tokens, 25);
		assert_eq!(*snap.api_keys.get("k1").unwrap(), 25);
		let _ = std::fs::remove_file(path);
	}
}
