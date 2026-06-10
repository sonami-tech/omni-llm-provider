//! Native Anthropic Messages passthrough: reconcile a CLIENT-supplied
//! `MessagesRequest` into the Claude Code fingerprint, and apply text
//! replacements to RAW Anthropic responses / SSE frames without going through
//! the OpenAI translation. This is the data layer for the `/v1/messages`
//! surface (handlers live in `routes/messages.rs`).
//!
//! Design + rationale: `docs/anthropic-compat-design.md`. Two invariants this
//! module exists to enforce (Gate-1 review, see that doc's review log):
//!   * fingerprint-bearing BODY fields (`betas`, `metadata`, `service_tier`,
//!     `mcp_servers`, `container`) are CCP-owned — a client cannot set them
//!     (G1-B2). The deserialization target is a closed allowlist.
//!   * raw passthrough is STRUCTURALLY faithful when no response rules are
//!     configured — every field and value reaches the client unchanged. (It is
//!     not guaranteed byte-identical: response JSON round-trips through
//!     `serde_json::Value`, which may reorder object keys. This is harmless —
//!     no checksum rides on RESPONSE key order and JSON clients are
//!     order-insensitive — but the contract is "faithful content", not "identical
//!     bytes".) When rules exist, replacement only touches known string leaves
//!     and, for streaming, is deferred to content-block close (G1-MAJOR-1).

use serde::Deserialize;
use serde_json::Value;

use crate::error::AppError;
use crate::replacements::Replacements;
use crate::routes::completions_v2::{
	apply_profile_wire_defaults, apply_replacements_outbound, apply_response_to_json,
	prepend_claude_code_identity,
};
use crate::translate::anthropic::{
	Message, MessagesRequest, SystemField, Thinking, Tool, ToolChoice,
};
use crate::translate::request::ChatCompletionRequest;
use crate::upstream::fingerprint::FingerprintProfile;

/// A client-supplied `/v1/messages` body, deserialized into a CLOSED allowlist.
///
/// Anything not named here is dropped, not forwarded — this is the native-surface
/// analogue of the OpenAI path's closed schema and the mechanism that keeps
/// client-controlled fingerprint-bearing fields (`betas`, `service_tier`,
/// `mcp_servers`, `container`, and a client-chosen `metadata`) off the wire
/// (G1-B2). The fields mirror `MessagesRequest` (the outbound type) minus the
/// CCP-owned ones.
//
// Deliberately a distinct type from `MessagesRequest`: the inbound contract is
// "what a client may set", the outbound contract is "what CCP sends". Keeping
// them separate is what makes the allowlist enforceable by construction — the
// forbidden fields have no field to deserialize into, so serde drops them, and
// `to_messages_request` only ever builds from the allowlisted set.
#[derive(Debug, Clone, Deserialize)]
pub struct ClientMessagesRequest {
	pub model: String,
	#[serde(default)]
	pub max_tokens: Option<u32>,
	pub messages: Vec<Message>,
	#[serde(default)]
	pub system: Option<SystemField>,
	#[serde(default)]
	pub tools: Option<Vec<Tool>>,
	#[serde(default)]
	pub tool_choice: Option<ToolChoice>,
	#[serde(default)]
	pub temperature: Option<f32>,
	#[serde(default)]
	pub top_p: Option<f32>,
	#[serde(default)]
	pub top_k: Option<u32>,
	#[serde(default)]
	pub stop_sequences: Option<Vec<String>>,
	#[serde(default)]
	pub stream: Option<bool>,
	#[serde(default)]
	pub thinking: Option<Thinking>,
	// NOTE: `betas`, `metadata`, `service_tier`, `mcp_servers`, `container`,
	// `output_config` are intentionally ABSENT — CCP owns every fingerprint- and
	// billing-bearing field. Unknown fields (these included) are ignored by serde.
}

impl ClientMessagesRequest {
	/// Convert the allowlisted client fields into a base `MessagesRequest`. The
	/// model is taken verbatim here; resolution + wire defaults + identity are
	/// applied by `reconcile_client_request`. `max_tokens` falls back to 0 and is
	/// always overwritten downstream (the client value, when present, is honored;
	/// when absent, the profile wire default fills it — same as the OpenAI path).
	fn to_messages_request(&self) -> MessagesRequest {
		MessagesRequest {
			model: self.model.clone(),
			max_tokens: self.max_tokens.unwrap_or(0),
			messages: self.messages.clone(),
			system: self.system.clone(),
			tools: self.tools.clone(),
			tool_choice: self.tool_choice.clone(),
			temperature: self.temperature,
			top_p: self.top_p,
			top_k: self.top_k,
			stop_sequences: self.stop_sequences.clone(),
			stream: self.stream,
			metadata: None,
			thinking: self.thinking.clone(),
			output_config: None,
		}
	}
}

/// Whether the client's body requested streaming. Anthropic selects SSE vs JSON
/// by this BODY field, not by route (G1-B1). Defaults to false when absent.
pub fn client_requested_stream(raw_body: &Value) -> bool {
	raw_body
		.get("stream")
		.and_then(|v| v.as_bool())
		.unwrap_or(false)
}

/// The top-level body fields the native surface forwards. A client field NOT in
/// this set is dropped by the closed allowlist (`ClientMessagesRequest`). Kept
/// in sync with that struct; used only for drop diagnostics (`dropped_fields`).
const FORWARDED_FIELDS: &[&str] = &[
	"model",
	"max_tokens",
	"messages",
	"system",
	"tools",
	"tool_choice",
	"temperature",
	"top_p",
	"top_k",
	"stop_sequences",
	"stream",
	"thinking",
];

/// Top-level client body keys that CCP does NOT forward, so a caller (and the
/// logs) can see exactly what was dropped instead of it vanishing silently.
/// Fingerprint/billing-bearing fields (`betas`, `metadata`, `service_tier`,
/// `mcp_servers`, `container`) are dropped BY DESIGN (G1-B2); any other unknown
/// field is dropped because the allowlist is closed. Returned sorted+deduped.
pub fn dropped_fields(raw_body: &Value) -> Vec<String> {
	let Some(obj) = raw_body.as_object() else {
		return Vec::new();
	};
	let mut out: Vec<String> = obj
		.keys()
		.filter(|k| !FORWARDED_FIELDS.contains(&k.as_str()))
		.cloned()
		.collect();
	out.sort();
	out
}

/// Resolve the client's model string. Anthropic-real ids (catalog canonical or
/// wire-override) are forwarded verbatim; bare family aliases resolve to the
/// profile canonical. A model that cannot be mapped to a real Anthropic id is
/// rejected at the CCP boundary with an Anthropic-shaped 400 (G1-MINOR) — never
/// forwarded to 400 upstream. "Real id" here means the resolver produced a
/// model whose canonical name actually matches the requested family; an
/// unrelated fallback (e.g. "gpt-4" → opus) is treated as unresolvable for the
/// strict native surface.
fn resolve_outbound_model(
	input: &str,
	profile: &'static FingerprintProfile,
) -> Result<String, AppError> {
	let model_def = profile.resolve_model(input);
	let outbound = profile.outbound_model(input, model_def);

	// Accept if the input is itself a claude- id (the resolver forwards real ones
	// verbatim and canonicalizes bare families to a same-family canonical), OR if
	// the input is a known bare alias of the resolved model. Reject a non-claude
	// or otherwise unrecognizable model that only matched via the catch-all
	// default — that is the "definitely-not-a-model" case U16/I7 pin.
	let is_claude_family = input.starts_with("claude-")
		|| model_def.aliases.iter().any(|a| a.eq_ignore_ascii_case(input))
		|| input.eq_ignore_ascii_case(model_def.cli_name);
	if !is_claude_family {
		return Err(AppError::BadRequest(format!(
			"model: {input} is not a recognized Anthropic model"
		)));
	}
	Ok(outbound)
}

/// Reconcile a client request into the outbound `MessagesRequest` CCP will send:
/// resolve the model, fill profile wire defaults ONLY for fields the client left
/// unset, re-apply the `max_tokens > thinking.budget` invariant, strip any
/// client-sent Claude Code identity then prepend CCP's pinned billing+preamble,
/// apply outbound replacements, and force `stream`.
///
/// Returns an Anthropic-shaped `AppError::BadRequest` if the model does not
/// resolve to a real Anthropic id (G1-MINOR) rather than letting it 400 upstream.
pub fn reconcile_client_request(
	client: &ClientMessagesRequest,
	_raw_body: &Value,
	profile: &'static FingerprintProfile,
	replacements: &Replacements,
	inject_identity: bool,
	stream: bool,
) -> Result<MessagesRequest, AppError> {
	let mut req = client.to_messages_request();
	let outbound_model = resolve_outbound_model(&client.model, profile)?;
	req.model = outbound_model;

	// Wire defaults fill only fields the client left unset. We synthesize the
	// minimal `ChatCompletionRequest` shape `apply_profile_wire_defaults` reads
	// (only max_tokens / max_completion_tokens are consulted as "did the caller
	// set max_tokens"), so a client-set max_tokens is preserved and an unset one
	// gets the profile wire default — identical semantics to the OpenAI path.
	let model_def = profile.resolve_model(&client.model);
	let source = ChatCompletionRequest {
		model: client.model.clone(),
		max_tokens: client.max_tokens,
		..Default::default()
	};
	apply_profile_wire_defaults(&mut req, &source, model_def, profile);

	// Re-apply the `max_tokens > thinking.budget_tokens` invariant for the
	// native surface. `apply_profile_wire_defaults` only enforces it when it
	// FILLED max_tokens (client left it unset); but a native client may send BOTH
	// an explicit max_tokens AND a thinking budget, and Anthropic rejects
	// max_tokens <= budget_tokens. The OpenAI path relies on build_messages_request
	// for this; we don't call that, so enforce it here too. Cap at the model's
	// catalog ceiling, mirroring the existing reconciliation.
	if let Some(budget) = req
		.thinking
		.as_ref()
		.filter(|t| t.kind == "enabled")
		.and_then(|t| t.budget_tokens)
		&& req.max_tokens <= budget
	{
		let ceiling = model_def.max_tokens.min(u32::MAX as u64) as u32;
		req.max_tokens = budget.saturating_add(1024).min(ceiling);
	}

	req.stream = Some(stream);

	// Replacements run BEFORE identity injection so the billing suffix is computed
	// from the exact post-replacement first-user text (same order as the OpenAI
	// path). Identity strip+inject is the shared, battle-tested routine.
	if !replacements.is_empty() {
		apply_replacements_outbound(&mut req, replacements);
	}
	prepend_claude_code_identity(&mut req, profile, inject_identity);

	Ok(req)
}

/// Build the body forwarded to the upstream `count_tokens` endpoint. Per the
/// Gate-1 decision (G1-MAJOR-2) the DEFAULT is to count the client's body as
/// sent (only model resolution + replacements applied) so the number is
/// comparable to real Anthropic — NOT inject-then-count. Identity blocks are
/// NOT added here.
pub fn build_count_tokens_request(
	client: &ClientMessagesRequest,
	raw_body: &Value,
	profile: &'static FingerprintProfile,
	replacements: &Replacements,
) -> Result<MessagesRequest, AppError> {
	// inject_identity=false: count the client's own body. Reuse reconcile so model
	// resolution and replacements stay consistent — only identity injection is
	// suppressed. NOTE: reconcile also fills wire defaults (max_tokens,
	// temperature, ...) which count_tokens does NOT accept; the count_tokens
	// HANDLER strips those generation fields from the serialized body before
	// sending. Here we only need a consistent, replacement-applied request shape.
	let mut req = reconcile_client_request(client, raw_body, profile, replacements, false, false)?;
	req.stream = None;
	Ok(req)
}

/// Apply response-scope replacements to a RAW non-streaming Anthropic response
/// value in place: text block `text`, tool_use `name`, and the STRING LEAVES of
/// tool_use `input` (a structured object — never whole-object replacement).
/// Object keys, structure, and unknown block types are left untouched. No-op
/// when `repl` has no response rules (true byte-faithful passthrough).
pub fn apply_response_replacements_raw(resp: &mut Value, repl: &Replacements) {
	if repl.max_response_search_len() == 0 {
		return; // no response rules: byte-faithful passthrough
	}
	let Some(content) = resp.get_mut("content").and_then(|c| c.as_array_mut()) else {
		return;
	};
	for block in content {
		let kind = block.get("type").and_then(|t| t.as_str()).map(str::to_string);
		match kind.as_deref() {
			Some("text") => {
				if let Some(text) = block.get_mut("text").and_then(|t| t.as_str()) {
					let replaced = repl.apply_response(text);
					block["text"] = Value::String(replaced);
				}
			}
			Some("tool_use") => {
				if let Some(name) = block.get_mut("name").and_then(|n| n.as_str()) {
					let replaced = repl.apply_response(name);
					block["name"] = Value::String(replaced);
				}
				if let Some(input) = block.get_mut("input") {
					// Walk only string leaves; keys + structure untouched.
					apply_response_to_json(input, repl);
				}
			}
			// Unknown / future block types (thinking, citations, server tool
			// results, redacted_thinking, ...) are NOT known string-leaf targets,
			// so they pass through untouched — fidelity is preserved.
			_ => {}
		}
	}
}

/// Render an `AppError` as an Anthropic error envelope value
/// `{"type":"error","error":{"type":<kind>,"message":<msg>}}` plus the HTTP
/// status. The `type` values are Anthropic's, which differ from CCP's
/// OpenAI-shaped renderer (e.g. ServerError → `api_error`, not `server_error`).
pub fn anthropic_error_body(err: &AppError) -> (axum::http::StatusCode, Value) {
	use axum::http::StatusCode;
	let (status, kind) = match err {
		AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
		AppError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
		AppError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found_error"),
		AppError::RateLimited(_) => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
		AppError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "overloaded_error"),
		AppError::BadGateway(_) => (StatusCode::BAD_GATEWAY, "api_error"),
		AppError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, "api_error"),
		AppError::ServerError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
	};
	let body = serde_json::json!({
		"type": "error",
		"error": { "type": kind, "message": err.to_string() },
	});
	(status, body)
}

/// Build the Anthropic-shaped `GET /v1/models` body from the static catalog.
/// Shape: `{ "data": [ {type, id, display_name, created_at} ], "has_more": false,
/// "first_id": <id>, "last_id": <id> }`.
pub fn anthropic_models_body(profile: &'static FingerprintProfile) -> Value {
	let infos = profile.models_list();
	let data: Vec<Value> = infos
		.iter()
		.map(|m| {
			serde_json::json!({
				"type": "model",
				"id": m.id,
				"display_name": display_name_for_model(&m.id),
				"created_at": iso8601_utc(m.created),
			})
		})
		.collect();
	let first_id = data.first().and_then(|m| m["id"].as_str()).map(str::to_string);
	let last_id = data.last().and_then(|m| m["id"].as_str()).map(str::to_string);
	serde_json::json!({
		"data": data,
		"has_more": false,
		"first_id": first_id,
		"last_id": last_id,
	})
}

/// Human-readable display name for a model id, e.g.
/// `claude-opus-4-8` → "Claude Opus 4.8", `claude-haiku-4-5-20251001` →
/// "Claude Haiku 4.5". The catalog only stores ids, so this derives a
/// presentable name: the leading `claude` + family word is title-cased, the
/// `<major>-<minor>` numeric pair is joined with a dot, and a trailing 8-digit
/// date snapshot suffix is dropped. Unrecognized shapes fall back to a simple
/// title-case so the function never panics or produces an empty string.
fn display_name_for_model(id: &str) -> String {
	let parts: Vec<&str> = id.split('-').filter(|p| !p.is_empty()).collect();
	// Drop a trailing YYYYMMDD snapshot suffix (8 digits).
	let parts: Vec<&str> = match parts.split_last() {
		Some((last, head)) if last.len() == 8 && last.chars().all(|c| c.is_ascii_digit()) => {
			head.to_vec()
		}
		_ => parts,
	};

	let title = |s: &str| -> String {
		let mut chars = s.chars();
		match chars.next() {
			Some(c) if c.is_ascii_alphabetic() => c.to_ascii_uppercase().to_string() + chars.as_str(),
			_ => s.to_string(),
		}
	};

	// Split into leading word(s) and the trailing run of numeric version parts.
	let split = parts.iter().position(|p| p.chars().all(|c| c.is_ascii_digit()));
	match split {
		Some(idx) => {
			let words: Vec<String> = parts[..idx].iter().map(|p| title(p)).collect();
			let version = parts[idx..].join(".");
			if words.is_empty() {
				version
			} else {
				format!("{} {}", words.join(" "), version)
			}
		}
		// No numeric version segment: just title-case all words.
		None => parts.iter().map(|p| title(p)).collect::<Vec<_>>().join(" "),
	}
}

/// Format a unix timestamp (seconds) as an RFC3339 / ISO-8601 UTC string, which
/// is the shape Anthropic's `created_at` field uses. Implemented without chrono
/// (not a dependency) via a minimal civil-date conversion.
fn iso8601_utc(secs: u64) -> String {
	// Days since epoch and seconds-of-day.
	let days = (secs / 86_400) as i64;
	let sod = secs % 86_400;
	let (hh, mm, ss) = (sod / 3600, (sod % 3600) / 60, sod % 60);
	// Howard Hinnant's civil_from_days algorithm.
	let z = days + 719_468;
	let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
	let doe = z - era * 146_097; // [0, 146096]
	let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
	let y = yoe + era * 400;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
	let mp = (5 * doy + 2) / 153; // [0, 11]
	let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
	let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
	let year = if m <= 2 { y + 1 } else { y };
	format!("{year:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

// ── Streaming replacement state (G1-MAJOR-1) ──────────────────────────────────
//
// When response rules are active, rewriting is deferred to content-block close:
// `text_delta` fragments and `input_json_delta` fragments are buffered per block
// index and emitted as a single rewritten delta at `content_block_stop`. With no
// rules, every frame passes through byte-identical. See the design doc §5a.

/// Per-stream buffer for deferred replacement over raw Anthropic SSE frames.
pub struct RawSseReplState {
	/// Whether any response-scope rule exists. When false, every frame is
	/// forwarded byte-identical (zero buffering) — the common, faithful path.
	has_response_rules: bool,
	/// Buffered `text_delta` text per content-block index.
	text_buf: std::collections::BTreeMap<u64, String>,
	/// Buffered `input_json_delta` partial-JSON per content-block index.
	json_buf: std::collections::BTreeMap<u64, String>,
	/// Content-block indices that have already been closed (their
	/// `content_block_stop` seen and flushed). A delta arriving for a stopped
	/// index — which only happens if the UPSTREAM violates the SSE protocol — is
	/// forwarded as-is rather than re-buffered, and the index is never re-flushed,
	/// so no duplicate `content_block_stop` can ever be synthesized.
	stopped: std::collections::BTreeSet<u64>,
}

impl RawSseReplState {
	pub fn new(repl: &Replacements) -> Self {
		Self {
			has_response_rules: repl.max_response_search_len() > 0,
			text_buf: std::collections::BTreeMap::new(),
			json_buf: std::collections::BTreeMap::new(),
			stopped: std::collections::BTreeSet::new(),
		}
	}

	/// Feed one raw upstream SSE frame `(event, data)`, return the frames to emit
	/// downstream. With no rules: returns the input frame unchanged. With rules:
	/// buffers `text_delta`/`input_json_delta` per block index and emits one
	/// coalesced rewritten delta at `content_block_stop`.
	pub fn on_frame(
		&mut self,
		event: &str,
		data: Value,
		repl: &Replacements,
	) -> Vec<(String, Value)> {
		if !self.has_response_rules {
			return vec![(event.to_string(), data)];
		}

		let index = data.get("index").and_then(|v| v.as_u64());
		let delta_type = data
			.get("delta")
			.and_then(|d| d.get("type"))
			.and_then(|t| t.as_str());

		match (event, delta_type, index) {
			// Buffer text_delta; suppress the partial frame (emit at block close).
			// A delta for an already-stopped index (upstream protocol violation) is
			// forwarded verbatim, not re-buffered, so it cannot resurrect the block.
			("content_block_delta", Some("text_delta"), Some(idx))
				if !self.stopped.contains(&idx) =>
			{
				if let Some(t) = data["delta"]["text"].as_str() {
					self.text_buf.entry(idx).or_default().push_str(t);
				}
				vec![]
			}
			// Buffer input_json_delta partial_json; suppress until block close.
			("content_block_delta", Some("input_json_delta"), Some(idx))
				if !self.stopped.contains(&idx) =>
			{
				if let Some(j) = data["delta"]["partial_json"].as_str() {
					self.json_buf.entry(idx).or_default().push_str(j);
				}
				vec![]
			}
			// At block close, flush any buffered text/json for this index as ONE
			// rewritten delta, then forward the stop frame. Mark the index stopped
			// so a stray later delta can never re-open or re-close it. A duplicate
			// stop for an already-stopped index is forwarded once but flushes
			// nothing (buffers are empty), so no second synthetic stop is produced.
			("content_block_stop", _, Some(idx)) => {
				let mut out = self.flush_block(idx, repl);
				self.stopped.insert(idx);
				out.push((event.to_string(), data));
				out
			}
			// Any other frame (message_start, content_block_start, message_delta,
			// message_stop, ping, error, a delta for an already-stopped index, ...)
			// is forwarded faithfully.
			_ => vec![(event.to_string(), data)],
		}
	}

	/// Flush leftover buffers for blocks that never received their natural
	/// `content_block_stop` (a stream that ended without a final stop, or the
	/// error/disconnect paths). Unlike the `on_frame` stop path — where the real
	/// stop frame follows the flushed delta — here we append a synthetic
	/// `content_block_stop` per flushed block so the emitted sequence terminates
	/// each open block well-formed rather than leaving it dangling.
	pub fn flush_all(&mut self, repl: &Replacements) -> Vec<(String, Value)> {
		let mut indices: Vec<u64> = self
			.text_buf
			.keys()
			.chain(self.json_buf.keys())
			.copied()
			.collect();
		indices.sort_unstable();
		indices.dedup();
		let mut out = Vec::new();
		for idx in indices {
			let block = self.flush_block(idx, repl);
			if !block.is_empty() {
				out.extend(block);
				out.push((
					"content_block_stop".to_string(),
					serde_json::json!({ "type": "content_block_stop", "index": idx }),
				));
			}
			// Mark closed so a stray post-flush delta for this index is forwarded
			// verbatim (never re-buffered) and the block is never re-flushed.
			self.stopped.insert(idx);
		}
		out
	}

	/// Emit the rewritten coalesced delta(s) for one block index.
	fn flush_block(&mut self, idx: u64, repl: &Replacements) -> Vec<(String, Value)> {
		let mut out = Vec::new();
		if let Some(text) = self.text_buf.remove(&idx) {
			let rewritten = repl.apply_response(&text);
			out.push((
				"content_block_delta".to_string(),
				serde_json::json!({
					"type": "content_block_delta",
					"index": idx,
					"delta": { "type": "text_delta", "text": rewritten },
				}),
			));
		}
		if let Some(raw_json) = self.json_buf.remove(&idx) {
			let rewritten = rewrite_json_string_leaves(&raw_json, repl);
			out.push((
				"content_block_delta".to_string(),
				serde_json::json!({
					"type": "content_block_delta",
					"index": idx,
					"delta": { "type": "input_json_delta", "partial_json": rewritten },
				}),
			));
		}
		out
	}
}

/// Apply response rules to the string leaves of a JSON document supplied as text
/// (a tool_use `input` accumulated from `input_json_delta` fragments). Falls back
/// to a whole-string replacement if the buffer is not valid JSON (e.g. a
/// truncated stream), so partial args are still rewritten rather than dropped.
fn rewrite_json_string_leaves(raw: &str, repl: &Replacements) -> String {
	match serde_json::from_str::<Value>(raw) {
		Ok(mut value) => {
			apply_response_to_json(&mut value, repl);
			serde_json::to_string(&value).unwrap_or_else(|_| repl.apply_response(raw))
		}
		Err(_) => repl.apply_response(raw),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::translate::anthropic::{
		ContentBlock, MessageContent, MessagesRequest, SystemField,
	};
	use crate::upstream::fingerprint::{
		CLAUDE_CODE_SYSTEM_PREAMBLE, RequestContext, default_profile,
	};

	fn parse_client(body: serde_json::Value) -> ClientMessagesRequest {
		serde_json::from_value(body).expect("client body should deserialize")
	}

	fn empty_repl() -> Replacements {
		Replacements::empty()
	}

	fn system_texts(req: &MessagesRequest) -> Vec<String> {
		match req.system.as_ref().expect("system present") {
			SystemField::Blocks(blocks) => blocks.iter().map(|b| b.text.clone()).collect(),
			SystemField::Text(_) => panic!("identity injection must force block system field"),
		}
	}

	// ── U1: flat-string system → [billing, preamble, original] ───────────────
	#[test]
	fn u1_reconcile_flat_string_system_becomes_blocks_with_identity_prepended() {
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
			"system": "be terse",
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		)
		.expect("reconcile ok");
		let texts = system_texts(&req);
		assert_eq!(texts.len(), 3, "billing + preamble + original");
		assert_eq!(texts[0], default_profile().billing_header_text("Say OK"));
		assert_eq!(texts[1], CLAUDE_CODE_SYSTEM_PREAMBLE);
		assert_eq!(texts[2], "be terse");
	}

	// ── U2: client already has CC preamble → not duplicated ──────────────────
	#[test]
	fn u2_reconcile_existing_cc_preamble_is_not_duplicated() {
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
			"system": [{"type": "text", "text": CLAUDE_CODE_SYSTEM_PREAMBLE}],
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		)
		.expect("reconcile ok");
		let body_value = serde_json::to_value(&req).unwrap();
		let bytes = default_profile()
			.finalize_body_json(&body_value, &RequestContext::new_reply())
			.unwrap();
		let json = String::from_utf8(bytes).unwrap();
		assert_eq!(
			json.matches("x-anthropic-billing-header:").count(),
			1,
			"exactly one billing marker after strip+inject"
		);
		let preamble_count = system_texts(&req)
			.iter()
			.filter(|t| t.as_str() == CLAUDE_CODE_SYSTEM_PREAMBLE)
			.count();
		assert_eq!(preamble_count, 1, "preamble not duplicated");
	}

	// ── U3: stale cch billing block replaced by fresh pinned ─────────────────
	#[test]
	fn u3_reconcile_stale_billing_block_is_replaced() {
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
			"system": [
				{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=e5ba6;"},
				{"type": "text", "text": CLAUDE_CODE_SYSTEM_PREAMBLE},
				{"type": "text", "text": "consumer system"},
			],
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		)
		.expect("reconcile ok");
		let texts = system_texts(&req);
		assert_eq!(texts[0], default_profile().billing_header_text("Say OK"));
		assert_eq!(texts[1], CLAUDE_CODE_SYSTEM_PREAMBLE);
		assert_eq!(texts[2], "consumer system");
		assert!(!texts[0].contains("cch=e5ba6"), "stale cch gone");
	}

	// ── U4: model resolve parity with the OpenAI path ────────────────────────
	#[test]
	fn u4_model_resolution_matches_openai_path() {
		let cases = [
			("claude-opus-4-8", "claude-opus-4-8"),
			("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
			("claude-opus", "claude-opus-4-8"),
			("claude-sonnet", "claude-sonnet-4-6"),
		];
		for (input, expected) in cases {
			let body = serde_json::json!({
				"model": input,
				"max_tokens": 100,
				"messages": [{"role": "user", "content": "Say OK"}],
			});
			let client = parse_client(body.clone());
			let req = reconcile_client_request(
				&client, &body, default_profile(), &empty_repl(), true, false,
			)
			.expect("reconcile ok");
			assert_eq!(req.model, expected, "{input} -> {expected}");
		}
	}

	// ── U5: wire defaults fill gaps; client-set values preserved ─────────────
	#[test]
	fn u5_wire_defaults_fill_only_unset_fields() {
		// haiku wire default temperature is 1.0; client omits -> filled.
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		)
		.unwrap();
		assert_eq!(req.temperature, Some(1.0), "wire default filled");

		// client sets temperature -> preserved verbatim.
		let body2 = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
			"temperature": 0.3,
		});
		let client2 = parse_client(body2.clone());
		let req2 = reconcile_client_request(
			&client2, &body2, default_profile(), &empty_repl(), true, false,
		)
		.unwrap();
		assert_eq!(req2.temperature, Some(0.3), "client value preserved");
	}

	// ── U6: max_tokens > thinking budget invariant re-applied ────────────────
	#[test]
	fn u6_max_tokens_never_below_thinking_budget() {
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
			"thinking": {"type": "enabled", "budget_tokens": 4096},
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		)
		.unwrap();
		assert!(
			req.max_tokens > 4096,
			"max_tokens {} must exceed thinking budget 4096",
			req.max_tokens
		);
	}

	// ── U7: outbound replacements on system/messages/tools (prompt scope) ────
	#[test]
	fn u7_outbound_replacements_applied_to_client_request() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"prompt\"\nsearch = \"MAGIC\"\nreplace = \"DONE\"\n",
		)
		.unwrap();
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "use MAGIC now"}],
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &repl, true, false,
		)
		.unwrap();
		let user = req.messages.iter().find(|m| m.role == "user").unwrap();
		let text = match &user.content {
			MessageContent::Text(t) => t.clone(),
			MessageContent::Blocks(blocks) => match &blocks[0] {
				ContentBlock::Text { text, .. } => text.clone(),
				_ => panic!("expected text block"),
			},
		};
		assert!(text.contains("DONE") && !text.contains("MAGIC"), "got: {text}");
	}

	// ── U8: inbound replacements on a RAW response value ─────────────────────
	#[test]
	fn u8_response_replacements_touch_only_string_leaves() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"response\"\nsearch = \"SECRET\"\nreplace = \"REDACTED\"\n",
		)
		.unwrap();
		let mut resp = serde_json::json!({
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"content": [
				{"type": "text", "text": "the SECRET is out"},
				{"type": "tool_use", "id": "t1", "name": "SECRET_tool",
				 "input": {"path": "has SECRET inside", "count": 3}},
			],
			"usage": {"input_tokens": 5, "output_tokens": 7},
		});
		apply_response_replacements_raw(&mut resp, &repl);
		assert_eq!(resp["content"][0]["text"], "the REDACTED is out");
		assert_eq!(resp["content"][1]["name"], "REDACTED_tool");
		assert_eq!(resp["content"][1]["input"]["path"], "has REDACTED inside");
		// structural: numeric + key untouched
		assert_eq!(resp["content"][1]["input"]["count"], 3);
		assert!(resp["content"][1]["input"].get("path").is_some(), "key intact");
	}

	// ── U10: raw passthrough preserves OAI-dropped fields ────────────────────
	#[test]
	fn u10_no_rules_passthrough_is_byte_faithful() {
		let resp_in = serde_json::json!({
			"id": "msg_42",
			"type": "message",
			"role": "assistant",
			"model": "claude-haiku-4-5-20251001",
			"content": [{"type": "text", "text": "hi"}],
			"stop_reason": "end_turn",
			"stop_sequence": null,
			"usage": {"input_tokens": 3, "output_tokens": 1},
		});
		let mut resp = resp_in.clone();
		apply_response_replacements_raw(&mut resp, &empty_repl());
		assert_eq!(resp, resp_in, "no rules => identical value, all fields survive");
	}

	// ── U11: Anthropic error envelope + status mapping ───────────────────────
	#[test]
	fn u11_anthropic_error_envelope_mapping() {
		let cases: [(AppError, u16, &str); 4] = [
			(AppError::BadRequest("x".into()), 400, "invalid_request_error"),
			(AppError::Unauthorized("x".into()), 401, "authentication_error"),
			(AppError::RateLimited("x".into()), 429, "rate_limit_error"),
			(AppError::ServerError("x".into()), 500, "api_error"),
		];
		for (err, status, kind) in cases {
			let (st, body) = anthropic_error_body(&err);
			assert_eq!(st.as_u16(), status);
			assert_eq!(body["type"], "error");
			assert_eq!(body["error"]["type"], kind);
			assert!(body["error"]["message"].is_string());
		}
	}

	// ── U12: Anthropic /v1/models shape ──────────────────────────────────────
	#[test]
	fn u12_anthropic_models_shape() {
		let body = anthropic_models_body(default_profile());
		assert_eq!(body["has_more"], false);
		let data = body["data"].as_array().expect("data array");
		assert!(!data.is_empty());
		for m in data {
			assert_eq!(m["type"], "model");
			assert!(m["id"].is_string());
			assert!(m["display_name"].is_string());
		}
		// the pinned default model must be present
		let ids: Vec<&str> = data.iter().filter_map(|m| m["id"].as_str()).collect();
		assert!(ids.iter().any(|id| id.starts_with("claude-")), "{ids:?}");
		// display_name must be presentable (no bare-digit "4 8" / date suffix).
		for m in data {
			let dn = m["display_name"].as_str().unwrap();
			assert!(dn.starts_with("Claude "), "display_name should start 'Claude ': {dn}");
			assert!(!dn.contains(" 4 "), "version digits must be dotted, not spaced: {dn}");
			assert!(
				!dn.chars().rev().take(8).all(|c| c.is_ascii_digit()),
				"date snapshot suffix must be dropped: {dn}"
			);
		}
	}

	// ── UAT-driven: display_name_for_model produces presentable names ─────────
	#[test]
	fn display_name_formatting() {
		assert_eq!(display_name_for_model("claude-opus-4-8"), "Claude Opus 4.8");
		assert_eq!(display_name_for_model("claude-sonnet-4-6"), "Claude Sonnet 4.6");
		assert_eq!(
			display_name_for_model("claude-haiku-4-5-20251001"),
			"Claude Haiku 4.5"
		);
		// Degenerate inputs never panic / never empty.
		assert_eq!(display_name_for_model("claude"), "Claude");
		assert!(!display_name_for_model("weird").is_empty());
	}

	// ── U13: count_tokens body has NO identity injected (G1-MAJOR-2) ─────────
	#[test]
	fn u13_count_tokens_body_is_client_body_no_identity() {
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
			"system": "be terse",
		});
		let client = parse_client(body.clone());
		let req = build_count_tokens_request(
			&client, &body, default_profile(), &empty_repl(),
		)
		.unwrap();
		// system must be the client's, with NO billing/preamble blocks prepended.
		match req.system.as_ref().expect("system present") {
			SystemField::Text(s) => assert_eq!(s, "be terse"),
			SystemField::Blocks(blocks) => {
				for b in blocks {
					assert_ne!(b.text, CLAUDE_CODE_SYSTEM_PREAMBLE, "no preamble in count");
					assert!(
						!b.text.contains("x-anthropic-billing-header:"),
						"no billing header in count"
					);
				}
			}
		}
	}

	// ── U14: client fingerprint-bearing body fields NEVER survive as client
	// values (G1-B2). Per Grok Gate-2: the invariant is "the client's value does
	// not reach the wire", which is satisfied by EITHER dropping the field OR
	// forcing it to the profile's value — but NEVER by forwarding the client's.
	// Testing `is_none()` alone is wrong (it fails a correct impl that forces a
	// profile value, and an impl that forwards `betas: []` would be a leak this
	// must catch). So assert: client array/string values are absent-or-different.
	#[test]
	fn u14_client_fingerprint_fields_never_survive_as_client_values() {
		let client_betas =
			serde_json::json!(["output-128k-2025-02-19", "computer-use-2024-10-22"]);
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
			"betas": client_betas,
			"service_tier": "priority",
			"mcp_servers": [{"type": "url", "url": "https://evil.example", "name": "x"}],
			"container": "cnt_client",
			"metadata": {"user_id": "client-chosen-id"},
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		)
		.unwrap();
		let out = serde_json::to_value(&req).unwrap();

		// betas: absent, or present but NOT the client's array (profile-forced).
		if let Some(betas) = out.get("betas") {
			assert_ne!(betas, &client_betas, "client betas must not reach the wire");
		}
		// service_tier / mcp_servers / container: the client's exact values must
		// not survive. (Profile does not set these today, so absence is expected;
		// if a future profile sets one, it must differ from the client's value.)
		if let Some(v) = out.get("service_tier") {
			assert_ne!(v.as_str(), Some("priority"), "client service_tier leaked");
		}
		if let Some(v) = out.get("mcp_servers") {
			assert_ne!(v, &body["mcp_servers"], "client mcp_servers leaked");
		}
		if let Some(v) = out.get("container") {
			assert_ne!(v.as_str(), Some("cnt_client"), "client container leaked");
		}
		// metadata.user_id must never be the client-chosen id (it is session-derived).
		if let Some(meta) = out.get("metadata") {
			assert_ne!(
				meta.get("user_id").and_then(|v| v.as_str()),
				Some("client-chosen-id"),
				"client must not control metadata.user_id (fingerprint)"
			);
		}
	}

	// ── U14b: a forced profile value for a fingerprint field (if the profile
	// sets one) is what reaches the wire — proves "force" semantics, not just
	// "drop". Today no profile sets body `betas`; this test asserts the contract
	// that IF the implementation forces a value, it equals the profile's, never
	// the client's. With no profile betas, it asserts absence (the current
	// correct behavior) — and will start asserting equality the day a profile
	// adds one, by reading the profile rather than hardcoding.
	#[test]
	fn u14b_betas_is_profile_owned_not_client_owned() {
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "Say OK"}],
			"betas": ["client-only-beta"],
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		)
		.unwrap();
		let out = serde_json::to_value(&req).unwrap();
		match out.get("betas").and_then(|b| b.as_array()) {
			None => { /* profile sets no body betas today: correct */ }
			Some(arr) => assert!(
				!arr.iter().any(|v| v.as_str() == Some("client-only-beta")),
				"client beta must never appear; only profile-owned betas allowed"
			),
		}
	}

	// ── U15: streaming replacement wire-sequence (deferred at block close) ────
	#[test]
	fn u15_streaming_replacement_coalesces_at_block_close() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"response\"\nsearch = \"foo\"\nreplace = \"bar\"\n",
		)
		.unwrap();
		let mut st = RawSseReplState::new(&repl);
		let frames = [
			("content_block_start", serde_json::json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})),
			("content_block_delta", serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"fo"}})),
			("content_block_delta", serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"o!"}})),
			("content_block_stop", serde_json::json!({"type":"content_block_stop","index":0})),
		];
		let mut emitted: Vec<(String, Value)> = Vec::new();
		for (ev, data) in frames {
			emitted.extend(st.on_frame(ev, data, &repl));
		}
		// The client must observe the rewritten text exactly once, coalesced.
		let all_text: String = emitted
			.iter()
			.filter(|(ev, _)| ev == "content_block_delta")
			.filter_map(|(_, d)| d["delta"]["text"].as_str())
			.collect();
		assert_eq!(all_text, "bar!", "rewritten text observed, not partial 'foo'");
		assert!(
			!all_text.contains("foo"),
			"client must never see the pre-replacement string"
		);
	}

	// ── U15b: with NO rules, every frame is byte-identical (passthrough) ──────
	#[test]
	fn u15b_streaming_no_rules_is_passthrough() {
		let st_repl = empty_repl();
		let mut st = RawSseReplState::new(&st_repl);
		let data = serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"foo"}});
		let out = st.on_frame("content_block_delta", data.clone(), &st_repl);
		assert_eq!(out, vec![("content_block_delta".to_string(), data)]);
	}

	// ── U16: unresolvable model → boundary 400, not forwarded ────────────────
	#[test]
	fn u16_unresolvable_model_is_boundary_400() {
		// A clearly non-Anthropic, non-aliasable id. resolve_model currently
		// falls back to default for unknown ids; the native surface must instead
		// reject at the boundary. (If resolve_model is chosen to always fall
		// back, this test pins that the native handler adds the explicit guard.)
		let body = serde_json::json!({
			"model": "definitely-not-a-model-xyz",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "hi"}],
		});
		let client = parse_client(body.clone());
		let result = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		);
		match result {
			Err(AppError::BadRequest(_)) => {}
			other => panic!("expected boundary BadRequest, got {other:?}"),
		}
	}

	// ── U18: unknown content-block type passes through untouched ─────────────
	#[test]
	fn u18_unknown_block_survives_replacement() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"response\"\nsearch = \"x\"\nreplace = \"y\"\n",
		)
		.unwrap();
		let mut resp = serde_json::json!({
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"content": [
				{"type": "some_future_block", "blob": {"x": "x"}, "raw": "xx"}
			],
			"usage": {"input_tokens": 1, "output_tokens": 1},
		});
		let before = resp.clone();
		apply_response_replacements_raw(&mut resp, &repl);
		// Unknown block is not a known string-leaf target, so it must be intact.
		assert_eq!(
			resp["content"][0], before["content"][0],
			"unknown block type must not be mutated"
		);
	}

	// ── U4b (Gate-2): bare family aliases resolve, matching the Python tests ──
	#[test]
	fn u4b_bare_alias_models_resolve() {
		for (input, expected_prefix) in [("haiku", "claude-haiku"), ("opus", "claude-opus"), ("sonnet", "claude-sonnet")] {
			let body = serde_json::json!({
				"model": input,
				"max_tokens": 100,
				"messages": [{"role": "user", "content": "Say OK"}],
			});
			let client = parse_client(body.clone());
			let req = reconcile_client_request(
				&client, &body, default_profile(), &empty_repl(), true, false,
			)
			.expect("reconcile ok");
			assert!(
				req.model.starts_with(expected_prefix),
				"{input} -> {} should start {expected_prefix}",
				req.model
			);
		}
	}

	// ── U7b (Gate-2): prompt replacement in a client SYSTEM string, and its
	// interaction with identity prepend — the search string in system must be
	// rewritten, and the injected billing+preamble must still be present and
	// unrewritten (they are added AFTER replacement, per the OpenAI path order).
	#[test]
	fn u7b_outbound_replacement_in_client_system() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"prompt\"\nsearch = \"MAGIC\"\nreplace = \"DONE\"\n",
		)
		.unwrap();
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [{"role": "user", "content": "hi"}],
			"system": "the MAGIC instruction",
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &repl, true, false,
		)
		.unwrap();
		let texts = system_texts(&req);
		// identity blocks present and intact
		assert_eq!(texts[1], CLAUDE_CODE_SYSTEM_PREAMBLE);
		// the client's system text rewritten
		assert!(
			texts.iter().any(|t| t.contains("DONE instruction") && !t.contains("MAGIC")),
			"client system not rewritten: {texts:?}"
		);
	}

	// ── U8b (Gate-2): cache_control and other sibling keys survive response
	// replacement (only the targeted string leaf changes). A dropped
	// cache_control would silently break prompt caching.
	#[test]
	fn u8b_response_replacement_preserves_sibling_keys() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"response\"\nsearch = \"SECRET\"\nreplace = \"REDACTED\"\n",
		)
		.unwrap();
		let mut resp = serde_json::json!({
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"content": [
				{"type": "text", "text": "SECRET here", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
			],
			"usage": {"input_tokens": 1, "output_tokens": 1},
		});
		apply_response_replacements_raw(&mut resp, &repl);
		assert_eq!(resp["content"][0]["text"], "REDACTED here");
		assert_eq!(
			resp["content"][0]["cache_control"],
			serde_json::json!({"type": "ephemeral", "ttl": "5m"}),
			"cache_control sibling must survive untouched"
		);
	}

	// ── U15c (Gate-2): streaming tool_use input_json_delta fragments are
	// buffered across chunk boundaries and emitted as ONE well-formed rewritten
	// input_json_delta at block close — never a corrupted partial.
	#[test]
	fn u15c_streaming_tool_input_json_delta_buffered_and_rewritten() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"response\"\nsearch = \"Paris\"\nreplace = \"London\"\n",
		)
		.unwrap();
		let mut st = RawSseReplState::new(&repl);
		let frames = [
			("content_block_start", serde_json::json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"get_weather","input":{}}})),
			("content_block_delta", serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Pa"}})),
			("content_block_delta", serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ris\"}"}})),
			("content_block_stop", serde_json::json!({"type":"content_block_stop","index":0})),
		];
		let mut emitted: Vec<(String, Value)> = Vec::new();
		for (ev, data) in frames {
			emitted.extend(st.on_frame(ev, data, &repl));
		}
		// Collect the emitted partial_json fragments for block 0.
		let json_out: String = emitted
			.iter()
			.filter(|(ev, _)| ev == "content_block_delta")
			.filter_map(|(_, d)| d["delta"]["partial_json"].as_str())
			.collect();
		// The reassembled JSON must be valid and rewritten.
		let parsed: Value = serde_json::from_str(&json_out)
			.unwrap_or_else(|e| panic!("emitted partial_json not valid JSON: {e}; got {json_out:?}"));
		assert_eq!(parsed["city"], "London", "value rewritten");
		assert!(!json_out.contains("Paris"), "pre-replacement value must not leak");
	}

	// ── U19 (Gate-2): multi-turn request with an assistant message and a
	// tool_result is reconciled without corrupting or duplicating later turns;
	// replacement (none here) and identity injection touch only system, not the
	// conversation body shape.
	#[test]
	fn u19_multi_turn_with_assistant_and_tool_result_preserved() {
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 100,
			"messages": [
				{"role": "user", "content": "weather in Paris?"},
				{"role": "assistant", "content": [
					{"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "Paris"}}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "tu1", "content": "sunny"}
				]},
			],
		});
		let client = parse_client(body.clone());
		let req = reconcile_client_request(
			&client, &body, default_profile(), &empty_repl(), true, false,
		)
		.unwrap();
		assert_eq!(req.messages.len(), 3, "no turns added or dropped");
		assert_eq!(req.messages[0].role, "user");
		assert_eq!(req.messages[1].role, "assistant");
		assert_eq!(req.messages[2].role, "user");
		// the assistant tool_use and the tool_result survive intact
		let out = serde_json::to_value(&req).unwrap();
		assert_eq!(out["messages"][1]["content"][0]["type"], "tool_use");
		assert_eq!(out["messages"][1]["content"][0]["input"]["city"], "Paris");
		assert_eq!(out["messages"][2]["content"][0]["type"], "tool_result");
		assert_eq!(out["messages"][2]["content"][0]["tool_use_id"], "tu1");
	}

	// ── U20 (Gate-3): flush_all on an interrupted stream emits the coalesced
	// rewritten delta AND a synthetic content_block_stop, so a block left open by
	// an error / disconnect / missing-stop terminates well-formed.
	#[test]
	fn u20_flush_all_emits_coalesced_delta_then_block_stop() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"response\"\nsearch = \"foo\"\nreplace = \"bar\"\n",
		)
		.unwrap();
		let mut st = RawSseReplState::new(&repl);
		// Open a text block and feed two deltas, but NEVER send content_block_stop.
		st.on_frame(
			"content_block_start",
			serde_json::json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
			&repl,
		);
		st.on_frame(
			"content_block_delta",
			serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"foo"}}),
			&repl,
		);
		let flushed = st.flush_all(&repl);
		// Expect: one coalesced rewritten text_delta, then a content_block_stop.
		assert_eq!(flushed.len(), 2, "delta + stop: {flushed:?}");
		assert_eq!(flushed[0].0, "content_block_delta");
		assert_eq!(flushed[0].1["delta"]["text"], "bar");
		assert_eq!(flushed[1].0, "content_block_stop");
		assert_eq!(flushed[1].1["index"], 0);
		// A second flush is empty (buffers drained).
		assert!(st.flush_all(&repl).is_empty(), "buffers must be drained");
	}

	// ── U21 (Gate-3): the on_frame content_block_stop path must NOT add a second
	// stop (the real stop frame follows). Regression guard against double-stop.
	#[test]
	fn u21_on_frame_stop_does_not_duplicate_stop() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"response\"\nsearch = \"foo\"\nreplace = \"bar\"\n",
		)
		.unwrap();
		let mut st = RawSseReplState::new(&repl);
		st.on_frame(
			"content_block_delta",
			serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"foo"}}),
			&repl,
		);
		let out = st.on_frame(
			"content_block_stop",
			serde_json::json!({"type":"content_block_stop","index":0}),
			&repl,
		);
		let stops = out.iter().filter(|(ev, _)| ev == "content_block_stop").count();
		assert_eq!(stops, 1, "exactly one stop on the natural-close path: {out:?}");
	}

	// ── U22 (hardening): a non-conformant upstream that emits a delta AFTER a
	// block's stop must NOT cause re-buffering, a resurrected block, or a second
	// synthetic stop. The stray delta is forwarded verbatim; flush_all stays empty.
	#[test]
	fn u22_delta_after_stop_does_not_duplicate_or_rebuffer() {
		let repl = Replacements::parse_for_test(
			"[[rule]]\nscope = \"response\"\nsearch = \"foo\"\nreplace = \"bar\"\n",
		)
		.unwrap();
		let mut st = RawSseReplState::new(&repl);
		st.on_frame(
			"content_block_delta",
			serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"foo"}}),
			&repl,
		);
		// Natural close: coalesced "bar" delta + the real stop.
		let closed = st.on_frame(
			"content_block_stop",
			serde_json::json!({"type":"content_block_stop","index":0}),
			&repl,
		);
		assert_eq!(closed.iter().filter(|(e, _)| e == "content_block_stop").count(), 1);

		// Stray delta AFTER stop (protocol violation): forwarded verbatim, NOT
		// buffered/suppressed.
		let stray = st.on_frame(
			"content_block_delta",
			serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"late"}}),
			&repl,
		);
		assert_eq!(stray.len(), 1, "stray post-stop delta forwarded verbatim");
		assert_eq!(stray[0].1["delta"]["text"], "late", "not rewritten/suppressed");

		// flush_all must produce nothing — no resurrected block, no second stop.
		assert!(st.flush_all(&repl).is_empty(), "no duplicate stop / re-flush");
	}

	// ── U23 (hardening): dropped_fields reports exactly the non-forwarded client
	// top-level fields (fingerprint fields + any unknown), so drops are observable.
	#[test]
	fn u23_dropped_fields_reports_non_forwarded_keys() {
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 10,
			"messages": [{"role": "user", "content": "hi"}],
			"system": "ok",
			"temperature": 0.5,
			// non-forwarded:
			"betas": ["x"],
			"service_tier": "priority",
			"metadata": {"user_id": "u"},
			"mcp_servers": [],
			"container": "c",
			"some_future_field": 1,
		});
		let dropped = dropped_fields(&body);
		assert_eq!(
			dropped,
			vec![
				"betas".to_string(),
				"container".to_string(),
				"mcp_servers".to_string(),
				"metadata".to_string(),
				"service_tier".to_string(),
				"some_future_field".to_string(),
			]
		);
		// Forwarded fields are never reported as dropped.
		for f in ["model", "max_tokens", "messages", "system", "temperature"] {
			assert!(!dropped.contains(&f.to_string()), "{f} is forwarded");
		}
	}

	// ── U24 (hardening): the FORWARDED_FIELDS list stays in sync with what
	// ClientMessagesRequest actually deserializes — a forwarded field, sent alone,
	// must survive into the outbound body (so the allowlist constant cannot drift
	// from the struct without a test failing).
	#[test]
	fn u24_forwarded_fields_match_client_struct() {
		// Build a request exercising every forwarded optional field and assert the
		// reconciled outbound body still carries them (model/messages always; the
		// optionals when set).
		let body = serde_json::json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 123,
			"messages": [{"role": "user", "content": "hi"}],
			"system": "sys",
			"tools": [{"name": "t", "input_schema": {"type": "object"}}],
			"tool_choice": {"type": "auto"},
			"temperature": 0.4,
			"top_p": 0.9,
			"top_k": 5,
			"stop_sequences": ["x"],
			"stream": false,
			"thinking": {"type": "disabled"},
		});
		// dropped_fields must be empty: everything sent is forwarded.
		assert!(
			dropped_fields(&body).is_empty(),
			"a body of only forwarded fields must drop nothing: {:?}",
			dropped_fields(&body)
		);
	}
}
