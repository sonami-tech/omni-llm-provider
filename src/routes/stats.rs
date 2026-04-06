use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;

use crate::AppState;

pub async fn stats_json_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
	Json(state.stats.snapshot())
}

pub async fn stats_html_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
	let snap = state.stats.snapshot();

	let uptime = format_duration(snap.uptime_seconds);
	let last_request = snap.last_request_at.as_deref().unwrap_or("–");

	let error_rate = if snap.total_requests > 0 {
		(snap.errors as f64 / snap.total_requests as f64) * 100.0
	} else {
		0.0
	};

	// Per-model rows.
	let mut model_rows = String::new();
	let mut model_names: Vec<&String> = snap.models.keys().collect();
	model_names.sort();
	for name in &model_names {
		let m = &snap.models[*name];
		model_rows.push_str(&format!(
			"<tr><td>{}</td><td>{}</td><td>{:.1}</td><td>{:.1}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
			name, m.requests, m.avg_ttft_ms, m.avg_duration_ms,
			m.input_tokens, m.output_tokens,
			m.cache_read_input_tokens, m.cache_creation_input_tokens,
		));
	}

	// Recent errors.
	let mut error_rows = String::new();
	for err in &snap.recent_errors {
		error_rows.push_str(&format!(
			"<tr><td>{}</td><td>{}</td><td>{}</td></tr>\n",
			html_escape(&err.timestamp),
			html_escape(&err.model),
			html_escape(&err.message),
		));
	}

	// Per-key rows.
	let mut key_rows = String::new();
	let mut key_names: Vec<&String> = snap.api_keys.keys().collect();
	key_names.sort();
	for name in &key_names {
		let count = snap.api_keys[*name];
		key_rows.push_str(&format!(
			"<tr><td>{}</td><td>{}</td></tr>\n",
			html_escape(name),
			count,
		));
	}

	// Cache efficiency.
	let total_cache_input =
		snap.total_cache_read_input_tokens + snap.total_cache_creation_input_tokens + snap.total_input_tokens;
	let cache_ratio = if total_cache_input > 0 {
		(snap.total_cache_read_input_tokens as f64 / total_cache_input as f64) * 100.0
	} else {
		0.0
	};

	let html = format!(
		r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>Claude Code Provider</title>
<style>
body {{ font-family: monospace; margin: 2em; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }}
h2 {{ color: #0f3460; border-bottom: 1px solid #333; padding-bottom: 4px; }}
.cards {{ display: flex; gap: 1.5em; margin: 1em 0; flex-wrap: wrap; }}
.card {{ background: #16213e; padding: 1em 1.5em; border-radius: 8px; min-width: 120px; }}
.card .value {{ font-size: 1.8em; font-weight: bold; color: #e94560; }}
.card .label {{ color: #888; font-size: 0.85em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: right; }}
th {{ background: #0f3460; }}
td:first-child, th:first-child {{ text-align: left; }}
tr:nth-child(even) {{ background: #16213e; }}
.empty {{ color: #666; font-style: italic; }}
</style>
</head>
<body>
<h1>Claude Code Provider v{version}</h1>
<p>Uptime: {uptime}</p>

<div class="cards">
<div class="card"><div class="value">{total_req}</div><div class="label">Total Requests</div></div>
<div class="card"><div class="value">{active}</div><div class="label">Active</div></div>
<div class="card"><div class="value">{errors}</div><div class="label">Errors</div></div>
<div class="card"><div class="value">{error_rate:.1}%</div><div class="label">Error Rate</div></div>
<div class="card"><div class="value">{cache_ratio:.1}%</div><div class="label">Cache Read Ratio</div></div>
<div class="card"><div class="value">{last_request}</div><div class="label">Last Request</div></div>
</div>

<h2>Per-Model Statistics</h2>
{model_table}

<h2>API Key Usage</h2>
{key_table}

<h2>Recent Errors</h2>
{error_table}

</body>
</html>"#,
		version = env!("CARGO_PKG_VERSION"),
		uptime = uptime,
		total_req = snap.total_requests,
		active = snap.active_requests,
		errors = snap.errors,
		error_rate = error_rate,
		cache_ratio = cache_ratio,
		last_request = last_request,
		model_table = if model_rows.is_empty() {
			"<p class=\"empty\">No requests yet.</p>".to_string()
		} else {
			format!(
				"<table><tr><th>Model</th><th>Requests</th><th>Avg TTFT (ms)</th><th>Avg Duration (ms)</th><th>Input Tokens</th><th>Output Tokens</th><th>Cache Read</th><th>Cache Creation</th></tr>\n{}</table>",
				model_rows
			)
		},
		key_table = if key_rows.is_empty() {
			"<p class=\"empty\">No API key usage recorded.</p>".to_string()
		} else {
			format!(
				"<table><tr><th>Key</th><th>Requests</th></tr>\n{}</table>",
				key_rows
			)
		},
		error_table = if error_rows.is_empty() {
			"<p class=\"empty\">No errors.</p>".to_string()
		} else {
			format!(
				"<table><tr><th>Timestamp</th><th>Model</th><th>Message</th></tr>\n{}</table>",
				error_rows
			)
		},
	);

	([(header::CONTENT_TYPE, "text/html")], html)
}

fn format_duration(secs: u64) -> String {
	let h = secs / 3600;
	let m = (secs % 3600) / 60;
	let s = secs % 60;
	if h > 0 {
		format!("{}h {}m {}s", h, m, s)
	} else if m > 0 {
		format!("{}m {}s", m, s)
	} else {
		format!("{}s", s)
	}
}

fn html_escape(s: &str) -> String {
	s.replace('&', "&amp;")
		.replace('<', "&lt;")
		.replace('>', "&gt;")
		.replace('"', "&quot;")
}
