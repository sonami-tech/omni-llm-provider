// Persistent, provider-agnostic request/usage statistics.
//
// Durable counters (total/per-model/per-key request counts, total errors,
// per-model token usage, last-request timestamp) live in a redb file the app
// owns. Volatile, bounded series that only matter while the process is alive
// (recent error messages, latency samples used to compute rolling averages)
// live in an in-memory mutex-guarded buffer. `snapshot()` joins both into the
// serializable shape the GET /stats endpoint serves.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use chrono::Utc;
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::error::AppError;

const TOTAL_REQUESTS: TableDefinition<&str, u64> = TableDefinition::new("total_requests");
const REQUESTS_BY_MODEL: TableDefinition<&str, u64> = TableDefinition::new("requests_by_model");
const TOTAL_ERRORS: TableDefinition<&str, u64> = TableDefinition::new("total_errors");
const TOKENS_BY_MODEL: TableDefinition<&str, &[u8]> = TableDefinition::new("tokens_by_model");
const REQUESTS_BY_KEY: TableDefinition<&str, u64> = TableDefinition::new("requests_by_key");
const LAST_REQUEST_AT: TableDefinition<&str, &str> = TableDefinition::new("last_request_at");
const TOTAL_KEY: &str = "total";

// The recent-errors ring buffer is a debugging aid, not an audit log: it must
// not grow without bound as the process serves traffic for days.
const MAX_RECENT_ERRORS: usize = 50;
// Rolling latency averages are computed over a bounded window of the most
// recent samples per model so memory stays flat under sustained load.
const MAX_SAMPLES: usize = 1000;

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

// Persisted per-model token tally. Serialized to bytes in TOKENS_BY_MODEL so a
// single value carries the full token breakdown for a model across restarts.
#[derive(Serialize, Deserialize, Default, Clone)]
struct TokenStats {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
}

#[derive(Serialize, Clone)]
pub struct ErrorRecord {
    pub timestamp: String,
    pub model: String,
    pub message: String,
}

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
    /// Process-scoped request count since this process launched (not durable).
    pub requests_since_launch: u64,
    /// Process-scoped token sum (input+output+cache_*) since this process launched.
    pub tokens_since_launch: u64,
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

// Volatile per-model latency window. Bounded to MAX_SAMPLES so averages reflect
// recent behavior without unbounded growth.
#[derive(Default)]
struct LatencyWindow {
    ttft_ms: VecDeque<f64>,
    duration_ms: VecDeque<f64>,
}

impl LatencyWindow {
    fn push_ttft(&mut self, v: f64) {
        push_bounded(&mut self.ttft_ms, v, MAX_SAMPLES);
    }
    fn push_duration(&mut self, v: f64) {
        push_bounded(&mut self.duration_ms, v, MAX_SAMPLES);
    }
    fn avg_ttft(&self) -> f64 {
        avg(&self.ttft_ms)
    }
    fn avg_duration(&self) -> f64 {
        avg(&self.duration_ms)
    }
}

// In-memory state that is intentionally not persisted: a debugging ring buffer
// of recent errors and rolling latency windows per model.
#[derive(Default)]
struct Volatile {
    recent_errors: VecDeque<ErrorRecord>,
    latency: HashMap<String, LatencyWindow>,
}

pub struct Stats {
    db: Database,
    volatile: Mutex<Volatile>,
    active: AtomicU64,
    start: Instant,
    /// Process-scoped counters (reset on restart). Cheap for status lines.
    requests_since_launch: AtomicU64,
    tokens_since_launch: AtomicU64,
}

impl Stats {
    /// Open (creating if absent) the stats database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let db = Database::create(path)?;
        // Create the tables up front so reads in snapshot() before any write
        // succeed instead of erroring on a missing table.
        let wtx = db.begin_write()?;
        {
            wtx.open_table(TOTAL_REQUESTS)?;
            wtx.open_table(REQUESTS_BY_MODEL)?;
            wtx.open_table(REQUESTS_BY_KEY)?;
            wtx.open_table(TOKENS_BY_MODEL)?;
            wtx.open_table(TOTAL_ERRORS)?;
            wtx.open_table(LAST_REQUEST_AT)?;
        }
        wtx.commit()?;
        Ok(Stats {
            db,
            volatile: Mutex::new(Volatile::default()),
            active: AtomicU64::new(0),
            start: Instant::now(),
            requests_since_launch: AtomicU64::new(0),
            tokens_since_launch: AtomicU64::new(0),
        })
    }

    /// Process-scoped totals since this process launched (not durable, not clearable).
    /// There is no stats-clear API; when one exists, a since-clear window can be added.
    pub fn since_launch_totals(&self) -> (u64, u64) {
        (
            self.requests_since_launch.load(Ordering::Relaxed),
            self.tokens_since_launch.load(Ordering::Relaxed),
        )
    }

    /// Record an inbound request: bumps the global counter, the per-model and
    /// per-key counters, and stamps the last-request time. Per-key attribution
    /// is what lets the dashboard show which API key drives load.
    pub fn record_request(&self, model: &str, key: Option<&str>) {
        self.requests_since_launch.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = self.record_request_inner(model, key) {
            tracing::warn!("stats: record_request failed: {e}");
        }
    }

    fn record_request_inner(&self, model: &str, key: Option<&str>) -> Result<(), AppError> {
        let now = Utc::now().to_rfc3339();
        let wtx = self.db.begin_write().map_err(db_err)?;
        {
            increment(&wtx, TOTAL_REQUESTS, TOTAL_KEY, 1)?;
            increment(&wtx, REQUESTS_BY_MODEL, model, 1)?;
            if let Some(k) = key {
                increment(&wtx, REQUESTS_BY_KEY, k, 1)?;
            }
            let mut t = wtx.open_table(LAST_REQUEST_AT).map_err(db_err)?;
            t.insert(TOTAL_KEY, now.as_str()).map_err(db_err)?;
        }
        wtx.commit().map_err(db_err)?;
        Ok(())
    }

    /// Record a completed response: accumulates token usage for the model and
    /// feeds the rolling latency window (ttft + total duration).
    pub fn record_response(&self, model: &str, usage: TokenUsage, ttft: Option<f64>, dur: f64) {
        let tok = usage
            .input_tokens
            .saturating_add(usage.output_tokens)
            .saturating_add(usage.cache_read_input_tokens)
            .saturating_add(usage.cache_creation_input_tokens);
        self.tokens_since_launch
            .fetch_add(tok, Ordering::Relaxed);
        if let Err(e) = self.record_response_inner(model, usage) {
            tracing::warn!("stats: record_response failed: {e}");
        }
        // Latency samples are volatile; never let a poisoned lock take down the
        // request path.
        if let Ok(mut v) = self.volatile.lock() {
            let w = v.latency.entry(model.to_string()).or_default();
            if let Some(t) = ttft {
                w.push_ttft(t);
            }
            w.push_duration(dur);
        }
    }

    fn record_response_inner(&self, model: &str, usage: TokenUsage) -> Result<(), AppError> {
        let wtx = self.db.begin_write().map_err(db_err)?;
        {
            let mut t = wtx.open_table(TOKENS_BY_MODEL).map_err(db_err)?;
            let mut stats: TokenStats = match t.get(model).map_err(db_err)? {
                Some(bytes) => serde_json::from_slice(bytes.value())
                    .map_err(|e| AppError::ServerError(format!("stats decode: {e}")))?,
                None => TokenStats::default(),
            };
            stats.input_tokens += usage.input_tokens;
            stats.output_tokens += usage.output_tokens;
            stats.cache_read_input_tokens += usage.cache_read_input_tokens;
            stats.cache_creation_input_tokens += usage.cache_creation_input_tokens;
            let encoded = serde_json::to_vec(&stats)
                .map_err(|e| AppError::ServerError(format!("stats encode: {e}")))?;
            t.insert(model, encoded.as_slice()).map_err(db_err)?;
        }
        wtx.commit().map_err(db_err)?;
        Ok(())
    }

    /// Record an error: bumps the durable error counter and pushes the message
    /// onto the bounded in-memory recent-errors ring (newest at the front).
    pub fn record_error(&self, model: &str, msg: &str) {
        if let Err(e) = self.bump_total_errors() {
            tracing::warn!("stats: record_error failed: {e}");
        }
        if let Ok(mut v) = self.volatile.lock() {
            v.recent_errors.push_front(ErrorRecord {
                timestamp: Utc::now().to_rfc3339(),
                model: model.to_string(),
                message: msg.to_string(),
            });
            while v.recent_errors.len() > MAX_RECENT_ERRORS {
                v.recent_errors.pop_back();
            }
        }
    }

    fn bump_total_errors(&self) -> Result<(), AppError> {
        let wtx = self.db.begin_write().map_err(db_err)?;
        increment(&wtx, TOTAL_ERRORS, TOTAL_KEY, 1)?;
        wtx.commit().map_err(db_err)?;
        Ok(())
    }

    /// Build the full serializable snapshot served by GET /stats. Joins durable
    /// redb counters with the volatile latency windows and recent-errors ring.
    pub fn snapshot(&self) -> StatsSnapshot {
        self.snapshot_inner().unwrap_or_else(|e| {
            tracing::warn!("stats: snapshot read failed, returning empty: {e}");
            self.empty_snapshot()
        })
    }

    fn snapshot_inner(&self) -> Result<StatsSnapshot, AppError> {
        let rtx = self.db.begin_read().map_err(db_err)?;

        let total_requests = read_one(&rtx, TOTAL_REQUESTS, TOTAL_KEY)?;
        let errors = read_one(&rtx, TOTAL_ERRORS, TOTAL_KEY)?;
        let last_request_at = {
            let t = rtx.open_table(LAST_REQUEST_AT).map_err(db_err)?;
            t.get(TOTAL_KEY)
                .map_err(db_err)?
                .map(|v| v.value().to_string())
        };

        let mut api_keys = HashMap::new();
        {
            let t = rtx.open_table(REQUESTS_BY_KEY).map_err(db_err)?;
            for row in t.iter().map_err(db_err)? {
                let (k, v) = row.map_err(db_err)?;
                api_keys.insert(k.value().to_string(), v.value());
            }
        }

        let mut req_by_model: HashMap<String, u64> = HashMap::new();
        {
            let t = rtx.open_table(REQUESTS_BY_MODEL).map_err(db_err)?;
            for row in t.iter().map_err(db_err)? {
                let (k, v) = row.map_err(db_err)?;
                req_by_model.insert(k.value().to_string(), v.value());
            }
        }

        let mut tokens_by_model: HashMap<String, TokenStats> = HashMap::new();
        {
            let t = rtx.open_table(TOKENS_BY_MODEL).map_err(db_err)?;
            for row in t.iter().map_err(db_err)? {
                let (k, v) = row.map_err(db_err)?;
                let ts: TokenStats = serde_json::from_slice(v.value())
                    .map_err(|e| AppError::ServerError(format!("stats decode: {e}")))?;
                tokens_by_model.insert(k.value().to_string(), ts);
            }
        }

        // Snapshot the volatile latency windows so the held lock is brief.
        let (recent_errors, latency_avgs) = {
            let v = self
                .volatile
                .lock()
                .map_err(|_| AppError::ServerError("stats lock poisoned".into()))?;
            let recent: Vec<ErrorRecord> = v.recent_errors.iter().cloned().collect();
            let avgs: HashMap<String, (f64, f64)> = v
                .latency
                .iter()
                .map(|(m, w)| (m.clone(), (w.avg_ttft(), w.avg_duration())))
                .collect();
            (recent, avgs)
        };

        // Union of every model seen via requests, tokens, or latency samples.
        let mut model_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        model_names.extend(req_by_model.keys().cloned());
        model_names.extend(tokens_by_model.keys().cloned());
        model_names.extend(latency_avgs.keys().cloned());

        let mut models = HashMap::new();
        let mut total_input = 0u64;
        let mut total_output = 0u64;
        let mut total_cache_read = 0u64;
        let mut total_cache_creation = 0u64;
        for name in model_names {
            let toks = tokens_by_model.get(&name).cloned().unwrap_or_default();
            let (avg_ttft_ms, avg_duration_ms) =
                latency_avgs.get(&name).copied().unwrap_or((0.0, 0.0));
            total_input += toks.input_tokens;
            total_output += toks.output_tokens;
            total_cache_read += toks.cache_read_input_tokens;
            total_cache_creation += toks.cache_creation_input_tokens;
            models.insert(
                name.clone(),
                ModelStats {
                    requests: req_by_model.get(&name).copied().unwrap_or(0),
                    avg_ttft_ms,
                    avg_duration_ms,
                    input_tokens: toks.input_tokens,
                    output_tokens: toks.output_tokens,
                    cache_read_input_tokens: toks.cache_read_input_tokens,
                    cache_creation_input_tokens: toks.cache_creation_input_tokens,
                },
            );
        }

        let (requests_since_launch, tokens_since_launch) = self.since_launch_totals();
        Ok(StatsSnapshot {
            uptime_seconds: self.start.elapsed().as_secs(),
            total_requests,
            active_requests: self.active.load(Ordering::Relaxed),
            errors,
            total_input_tokens: total_input,
            total_output_tokens: total_output,
            total_cache_read_input_tokens: total_cache_read,
            total_cache_creation_input_tokens: total_cache_creation,
            requests_since_launch,
            tokens_since_launch,
            last_request_at,
            models,
            api_keys,
            recent_errors,
        })
    }

    fn empty_snapshot(&self) -> StatsSnapshot {
        let (requests_since_launch, tokens_since_launch) = self.since_launch_totals();
        StatsSnapshot {
            uptime_seconds: self.start.elapsed().as_secs(),
            total_requests: 0,
            active_requests: self.active.load(Ordering::Relaxed),
            errors: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_read_input_tokens: 0,
            total_cache_creation_input_tokens: 0,
            requests_since_launch,
            tokens_since_launch,
            last_request_at: None,
            models: HashMap::new(),
            api_keys: HashMap::new(),
            recent_errors: Vec::new(),
        }
    }

    /// Human-readable plain text (ckb-mcp style). Optional version banner.
    pub fn format_human(&self, version: Option<&str>) -> String {
        self.snapshot().format_human(version)
    }

    /// Pretty-printed JSON of the full snapshot.
    pub fn format_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.snapshot())
    }
}

impl StatsSnapshot {
    /// Render a curl-friendly plain-text summary (default GET /stats body).
    ///
    /// Layout (top → bottom): live process state → this-process window →
    /// durable lifetime totals (traffic, then tokens with cache ratio under
    /// cache counts) → per-model / per-key breakdowns → recent errors.
    pub fn format_human(&self, version: Option<&str>) -> String {
        let mut out = String::new();
        match version {
            Some(v) => {
                let header = format!("Omni LLM Provider Stats (v{v})");
                out.push_str(&header);
                out.push('\n');
                out.push_str(&"=".repeat(header.len()));
                out.push_str("\n\n");
            }
            None => {
                out.push_str("Omni LLM Provider Stats\n");
                out.push_str("======================\n\n");
            }
        }

        // 1. Live process state (is the server up and busy right now?)
        out.push_str("Process\n");
        out.push_str("-------\n");
        out.push_str(&format!(
            "  Uptime:          {}\n",
            format_uptime(self.uptime_seconds)
        ));
        out.push_str(&format!(
            "  Active now:      {}\n",
            format_commas(self.active_requests)
        ));
        out.push_str(&format!(
            "  Last request:    {}\n\n",
            self.last_request_at.as_deref().unwrap_or("never")
        ));

        // 2. This process only (resets on restart; short window operators watch)
        out.push_str("Since launch (this process)\n");
        out.push_str("---------------------------\n");
        out.push_str(&format!(
            "  Requests:        {}\n",
            format_commas(self.requests_since_launch)
        ));
        out.push_str(&format!(
            "  Tokens:          {}\n\n",
            format_commas(self.tokens_since_launch)
        ));

        // 3. Durable lifetime totals (survive restarts)
        let total_tok = self
            .total_input_tokens
            .saturating_add(self.total_output_tokens)
            .saturating_add(self.total_cache_read_input_tokens)
            .saturating_add(self.total_cache_creation_input_tokens);

        out.push_str("All time (durable)\n");
        out.push_str("------------------\n");
        out.push_str("  Traffic\n");
        out.push_str(&format!(
            "    Requests:      {}\n",
            format_commas(self.total_requests)
        ));
        out.push_str(&format!(
            "    Errors:        {}\n",
            format_commas(self.errors)
        ));
        if self.total_requests > 0 {
            let rate = (self.errors as f64 / self.total_requests as f64) * 100.0;
            out.push_str(&format!("    Error rate:    {rate:.1}%\n"));
        }
        out.push_str("  Tokens\n");
        out.push_str(&format!(
            "    Input:         {}\n",
            format_commas(self.total_input_tokens)
        ));
        out.push_str(&format!(
            "    Output:        {}\n",
            format_commas(self.total_output_tokens)
        ));
        out.push_str(&format!(
            "    Cache read:    {}\n",
            format_commas(self.total_cache_read_input_tokens)
        ));
        out.push_str(&format!(
            "    Cache create:  {}\n",
            format_commas(self.total_cache_creation_input_tokens)
        ));
        // Ratio sits under the cache counts it is derived from.
        if total_tok > 0 {
            let ratio =
                (self.total_cache_read_input_tokens as f64 / total_tok as f64) * 100.0;
            out.push_str(&format!("    Cache read ratio: {ratio:.1}%\n"));
        }
        out.push_str(&format!(
            "    Total:         {}\n\n",
            format_commas(total_tok)
        ));

        // 4. Where load goes — models busiest first
        if !self.models.is_empty() {
            out.push_str("Per-model\n");
            out.push_str("---------\n");
            let mut names: Vec<&String> = self.models.keys().collect();
            names.sort_by(|a, b| {
                let ra = self.models[*a].requests;
                let rb = self.models[*b].requests;
                rb.cmp(&ra).then_with(|| a.cmp(b))
            });
            for name in names {
                let m = &self.models[name];
                out.push_str(&format!("  {name}\n"));
                out.push_str(&format!(
                    "    Requests:     {}\n",
                    format_commas(m.requests)
                ));
                out.push_str(&format!(
                    "    Latency:      avg_ttft_ms={:.1}  avg_duration_ms={:.1}\n",
                    m.avg_ttft_ms, m.avg_duration_ms
                ));
                out.push_str(&format!(
                    "    Tokens:       in={}  out={}\n",
                    format_commas(m.input_tokens),
                    format_commas(m.output_tokens)
                ));
                out.push_str(&format!(
                    "    Cache:        read={}  create={}\n",
                    format_commas(m.cache_read_input_tokens),
                    format_commas(m.cache_creation_input_tokens)
                ));
            }
            out.push('\n');
        }

        // 5. Who is calling
        if !self.api_keys.is_empty() {
            out.push_str("API keys\n");
            out.push_str("--------\n");
            let mut keys: Vec<(&String, u64)> =
                self.api_keys.iter().map(|(k, v)| (k, *v)).collect();
            keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
            for (key, count) in keys {
                out.push_str(&format!(
                    "  {} - {} requests\n",
                    key,
                    format_commas(count)
                ));
            }
            out.push('\n');
        }

        // 6. Diagnostics last
        if !self.recent_errors.is_empty() {
            out.push_str("Recent errors\n");
            out.push_str("-------------\n");
            for (i, err) in self.recent_errors.iter().take(20).enumerate() {
                out.push_str(&format!(
                    "  {}. [{}] {} - {}\n",
                    i + 1,
                    err.timestamp,
                    err.model,
                    err.message
                ));
            }
        }

        out
    }
}

fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if days > 0 {
        format!("{days}d {hours}h {minutes}m {seconds}s")
    } else if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

/// Group thousands with commas for human-readable counters (e.g. 1234567 -> "1,234,567").
fn format_commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let first = bytes.len() % 3;
    if first > 0 {
        out.push_str(&s[..first]);
    }
    let mut i = first;
    while i < bytes.len() {
        if !out.is_empty() {
            out.push(',');
        }
        out.push_str(&s[i..i + 3]);
        i += 3;
    }
    out
}

fn db_err(e: impl std::fmt::Display) -> AppError {
    AppError::ServerError(format!("stats db: {e}"))
}

// Read-modify-write a single u64 counter row by `delta` inside an open write tx.
fn increment(
    wtx: &redb::WriteTransaction,
    table: TableDefinition<&'static str, u64>,
    key: &str,
    delta: u64,
) -> Result<(), AppError> {
    let mut t = wtx.open_table(table).map_err(db_err)?;
    let current = t.get(key).map_err(db_err)?.map(|v| v.value()).unwrap_or(0);
    t.insert(key, current + delta).map_err(db_err)?;
    Ok(())
}

fn read_one(
    rtx: &redb::ReadTransaction,
    table: TableDefinition<&'static str, u64>,
    key: &str,
) -> Result<u64, AppError> {
    let t = rtx.open_table(table).map_err(db_err)?;
    Ok(t.get(key).map_err(db_err)?.map(|v| v.value()).unwrap_or(0))
}

fn push_bounded(buf: &mut VecDeque<f64>, v: f64, max: usize) {
    buf.push_back(v);
    while buf.len() > max {
        buf.pop_front();
    }
}

fn avg(buf: &VecDeque<f64>) -> f64 {
    if buf.is_empty() {
        0.0
    } else {
        buf.iter().sum::<f64>() / buf.len() as f64
    }
}

pub struct ActiveRequestGuard<'a> {
    stats: &'a Stats,
}
impl<'a> ActiveRequestGuard<'a> {
    pub fn new(s: &'a Stats) -> Self {
        s.active.fetch_add(1, Ordering::Relaxed);
        Self { stats: s }
    }
}
impl Drop for ActiveRequestGuard<'_> {
    fn drop(&mut self) {
        self.stats.active.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Each test owns a unique temp file (clearly labeled test data) so parallel
    // `cargo test` runs never collide on the same redb path.
    fn temp_db_path() -> PathBuf {
        std::env::temp_dir().join(format!("omni-stats-test-{}.redb", uuid::Uuid::new_v4()))
    }

    struct TempDb(PathBuf);
    impl Drop for TempDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn token_usage_defaults_and_fields() {
        let u = TokenUsage::default();
        assert_eq!(u.input_tokens, 0);
        let u2 = TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_input_tokens: 2,
            cache_creation_input_tokens: 1,
        };
        assert_eq!(u2.input_tokens, 10);
    }

    // The core product requirement: requests, responses and errors must be
    // attributed per-model and per-key and survive into the snapshot with exact
    // counts (this is what the /stats dashboard renders).
    #[test]
    fn records_persist_with_exact_per_model_and_per_key_counts() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        let stats = Stats::open(&path).expect("open temp stats db");

        // Two requests, same model, two distinct keys.
        stats.record_request("m", Some("keyA"));
        stats.record_request("m", Some("keyB"));
        stats.record_response(
            "m",
            TokenUsage {
                input_tokens: 7,
                output_tokens: 3,
                cache_read_input_tokens: 1,
                cache_creation_input_tokens: 2,
            },
            Some(120.0),
            300.0,
        );
        stats.record_error("m", "rate limit");

        let snap = stats.snapshot();

        assert_eq!(snap.total_requests, 2, "both requests counted globally");
        assert_eq!(
            snap.models["m"].requests, 2,
            "both requests attributed to model m"
        );
        assert_eq!(snap.api_keys.len(), 2, "two distinct keys tracked");
        assert_eq!(snap.api_keys["keyA"], 1);
        assert_eq!(snap.api_keys["keyB"], 1);

        // Response tokens accumulate against the model and the totals.
        assert_eq!(snap.models["m"].input_tokens, 7);
        assert_eq!(snap.models["m"].output_tokens, 3);
        assert_eq!(snap.total_input_tokens, 7);
        assert_eq!(snap.total_output_tokens, 3);
        assert_eq!(snap.total_cache_read_input_tokens, 1);
        assert_eq!(snap.total_cache_creation_input_tokens, 2);
        // Process-scoped since-launch: 2 requests, all token kinds summed once.
        assert_eq!(snap.requests_since_launch, 2);
        assert_eq!(snap.tokens_since_launch, 7 + 3 + 1 + 2);
        assert_eq!(stats.since_launch_totals(), (2, 13));

        // Error counter and recent-errors ring both reflect the one error.
        assert_eq!(snap.errors, 1);
        assert_eq!(snap.recent_errors.len(), 1);
        assert_eq!(snap.recent_errors[0].message, "rate limit");

        // last_request_at was stamped (RFC3339, non-empty).
        assert!(snap.last_request_at.is_some());
        assert!(!snap.last_request_at.unwrap().is_empty());
    }

    // Token usage for the same model must accumulate across calls, not overwrite
    // (the redb value is a read-modify-write of TokenStats).
    #[test]
    fn token_usage_accumulates_across_responses() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        let stats = Stats::open(&path).expect("open");

        stats.record_response(
            "m",
            TokenUsage {
                input_tokens: 10,
                output_tokens: 4,
                ..Default::default()
            },
            None,
            10.0,
        );
        stats.record_response(
            "m",
            TokenUsage {
                input_tokens: 5,
                output_tokens: 1,
                ..Default::default()
            },
            None,
            20.0,
        );

        let snap = stats.snapshot();
        assert_eq!(snap.models["m"].input_tokens, 15, "input tokens summed");
        assert_eq!(snap.models["m"].output_tokens, 5, "output tokens summed");
        assert_eq!(snap.total_input_tokens, 15);
    }

    // The recent-errors ring is a debugging aid that must never grow without
    // bound: it caps at MAX_RECENT_ERRORS and keeps the most recent entries.
    #[test]
    fn recent_errors_are_bounded_and_keep_newest() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        let stats = Stats::open(&path).expect("open");

        let n = MAX_RECENT_ERRORS + 5;
        for i in 0..n {
            stats.record_error("m", &format!("err{i}"));
        }

        let snap = stats.snapshot();
        // Every error still counted durably, even those evicted from the ring.
        assert_eq!(snap.errors, n as u64);
        // Ring is capped.
        assert_eq!(snap.recent_errors.len(), MAX_RECENT_ERRORS);
        // Newest is at the front; the oldest few were evicted.
        assert_eq!(snap.recent_errors[0].message, format!("err{}", n - 1));
        assert_eq!(
            snap.recent_errors.last().unwrap().message,
            format!("err{}", n - MAX_RECENT_ERRORS)
        );
    }

    // Counters must survive reopening the database file (durability is the whole
    // point of using redb rather than in-memory counters).
    #[test]
    fn counters_persist_across_reopen() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        {
            let stats = Stats::open(&path).expect("open");
            stats.record_request("m", Some("k"));
            stats.record_request("m", Some("k"));
        } // db dropped/closed here

        let reopened = Stats::open(&path).expect("reopen");
        let snap = reopened.snapshot();
        assert_eq!(snap.total_requests, 2, "request count survived reopen");
        assert_eq!(snap.models["m"].requests, 2);
        assert_eq!(snap.api_keys["k"], 2, "per-key count survived reopen");
        // recent_errors is volatile and intentionally empty after reopen.
        assert!(snap.recent_errors.is_empty());
    }

    // Latency averages are computed over a bounded window and reflect samples.
    #[test]
    fn latency_averages_reflect_samples_and_serialize() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        let stats = Stats::open(&path).expect("open");

        stats.record_response("m", TokenUsage::default(), Some(100.0), 200.0);
        stats.record_response("m", TokenUsage::default(), Some(300.0), 400.0);

        let snap = stats.snapshot();
        assert_eq!(snap.models["m"].avg_ttft_ms, 200.0);
        assert_eq!(snap.models["m"].avg_duration_ms, 300.0);

        // Snapshot is the wire shape for GET /stats?format=json; it must serialize.
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("uptime_seconds"));
        assert!(json.contains("avg_ttft_ms"));
    }

    // Human text is the default /stats body: readable summary with since-launch window.
    #[test]
    fn format_human_includes_totals_and_since_launch() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        let stats = Stats::open(&path).expect("open");
        stats.record_request("m", Some("k1"));
        stats.record_response(
            "m",
            TokenUsage {
                input_tokens: 10,
                output_tokens: 4,
                cache_read_input_tokens: 1,
                cache_creation_input_tokens: 0,
            },
            Some(50.0),
            100.0,
        );
        let text = stats.format_human(Some("9.9.9"));
        assert!(text.contains("Omni LLM Provider Stats (v9.9.9)"), "{text}");
        assert!(text.contains("Process"), "{text}");
        assert!(text.contains("Since launch"), "{text}");
        assert!(text.contains("All time (durable)"), "{text}");
        // Section order: Process → Since launch → All time → Per-model
        let process_i = text.find("Process\n").expect("Process section");
        let launch_i = text.find("Since launch").expect("Since launch section");
        let all_i = text.find("All time (durable)").expect("All time section");
        let model_i = text.find("Per-model").expect("Per-model section");
        assert!(process_i < launch_i && launch_i < all_i && all_i < model_i, "{text}");
        // Cache ratio sits after cache counts, before total.
        let cache_read = text.find("Cache read:").expect("cache read");
        let cache_create = text.find("Cache create:").expect("cache create");
        let cache_ratio = text.find("Cache read ratio:").expect("cache ratio");
        let total_line = text.find("    Total:").expect("token total");
        assert!(
            cache_read < cache_create && cache_create < cache_ratio && cache_ratio < total_line,
            "token block order wrong: {text}"
        );
        assert!(text.contains("Tokens:          15"), "{text}");
        assert!(text.contains("  m\n"), "{text}");
        let pretty = stats.format_json().expect("json");
        assert!(pretty.contains("\"total_requests\": 1"), "{pretty}");
        assert!(pretty.contains("requests_since_launch"), "{pretty}");
    }

    #[test]
    fn format_commas_groups_thousands() {
        assert_eq!(format_commas(0), "0");
        assert_eq!(format_commas(12), "12");
        assert_eq!(format_commas(999), "999");
        assert_eq!(format_commas(1_000), "1,000");
        assert_eq!(format_commas(12_345), "12,345");
        assert_eq!(format_commas(1_234_567), "1,234,567");
        assert_eq!(format_commas(1_000_000_000), "1,000,000,000");
    }

    // Large cumulative counters in human text use comma grouping (JSON stays raw).
    #[test]
    fn format_human_comma_formats_large_counts() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        let stats = Stats::open(&path).expect("open");
        // Bump process-scoped token total into thousands via one response.
        stats.record_request("m", None);
        stats.record_response(
            "m",
            TokenUsage {
                input_tokens: 1_234_567,
                output_tokens: 8_901,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
            None,
            1.0,
        );
        let text = stats.format_human(None);
        assert!(
            text.contains("1,234,567"),
            "input tokens should be comma-grouped: {text}"
        );
        assert!(
            text.contains("8,901"),
            "output tokens should be comma-grouped: {text}"
        );
        // 1_234_567 + 8_901 = 1_243_468
        assert!(
            text.contains("1,243,468"),
            "since-launch / sum tokens should be comma-grouped: {text}"
        );
        // JSON remains unformatted integers for machines.
        let pretty = stats.format_json().expect("json");
        assert!(pretty.contains("1234567"), "{pretty}");
        assert!(!pretty.contains("1,234,567"), "{pretty}");
    }

    // Since-launch counters are process-scoped: they advance with record_* but
    // do not survive reopen (unlike durable redb totals).
    #[test]
    fn since_launch_counters_reset_on_reopen() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        {
            let stats = Stats::open(&path).expect("open");
            stats.record_request("m", None);
            stats.record_response(
                "m",
                TokenUsage {
                    input_tokens: 4,
                    output_tokens: 1,
                    ..Default::default()
                },
                None,
                10.0,
            );
            assert_eq!(stats.since_launch_totals(), (1, 5));
        }
        let reopened = Stats::open(&path).expect("reopen");
        let snap = reopened.snapshot();
        assert_eq!(snap.total_requests, 1, "durable request count survived");
        assert_eq!(snap.total_input_tokens, 4, "durable tokens survived");
        assert_eq!(
            reopened.since_launch_totals(),
            (0, 0),
            "process-scoped counters start at zero after relaunch"
        );
        assert_eq!(snap.requests_since_launch, 0);
        assert_eq!(snap.tokens_since_launch, 0);
    }

    // ActiveRequestGuard tracks in-flight requests: increment on construction,
    // decrement on drop, surfaced as active_requests in the snapshot.
    #[test]
    fn active_request_guard_tracks_in_flight() {
        let path = temp_db_path();
        let _cleanup = TempDb(path.clone());
        let stats = Stats::open(&path).expect("open");

        assert_eq!(stats.snapshot().active_requests, 0);
        {
            let _g1 = ActiveRequestGuard::new(&stats);
            assert_eq!(stats.snapshot().active_requests, 1);
            {
                let _g2 = ActiveRequestGuard::new(&stats);
                assert_eq!(stats.snapshot().active_requests, 2);
            }
            assert_eq!(stats.snapshot().active_requests, 1);
        }
        assert_eq!(stats.snapshot().active_requests, 0);
    }
}
