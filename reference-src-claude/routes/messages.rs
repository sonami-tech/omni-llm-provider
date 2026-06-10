//! Native Anthropic Messages surface: `POST /v1/messages` (stream + non-stream
//! selected by the `stream` BODY field), `POST /v1/messages/count_tokens`, and
//! `GET /v1/models` in Anthropic shape.
//!
//! This bypasses the OpenAI translation on both ends: the client's own
//! Anthropic body is reconciled into the Claude Code fingerprint (strip+inject
//! identity, profile wire defaults, model resolution, replacements) and the
//! upstream Anthropic response / SSE is forwarded faithfully. See
//! `docs/anthropic-compat-design.md` and `translate::anthropic_passthrough`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{Instrument, error, info, warn};

use crate::AppState;
use crate::error::AppError;
use crate::routes::completions_v2::{derive_session_uuid, map_upstream_err};
use crate::translate::anthropic_passthrough::{
	ClientMessagesRequest, RawSseReplState, anthropic_error_body, apply_response_replacements_raw,
	build_count_tokens_request, client_requested_stream, reconcile_client_request,
};
use crate::upstream::credentials::Credentials;
use crate::upstream::fingerprint::RequestContext;

/// `POST /v1/messages` — dispatches to streaming or non-streaming on the parsed
/// `stream` body field (Anthropic uses one endpoint; the body selects the mode).
pub async fn messages_handler(
	State(state): State<Arc<AppState>>,
	request: axum::http::Request<axum::body::Body>,
) -> Response {
	match messages_handler_inner(state, request).await {
		Ok(resp) => resp,
		Err(err) => anthropic_error_response(&err),
	}
}

async fn messages_handler_inner(
	state: Arc<AppState>,
	request: axum::http::Request<axum::body::Body>,
) -> Result<Response, AppError> {
	let session_header = request
		.headers()
		.get("x-session-id")
		.and_then(|v| v.to_str().ok())
		.map(str::to_string);
	let body = axum::body::to_bytes(request.into_body(), crate::MAX_BODY_SIZE)
		.await
		.map_err(|e| AppError::BadRequest(format!("Failed to read body: {e}")))?;

	let raw_body: serde_json::Value = serde_json::from_slice(&body)
		.map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;
	let client: ClientMessagesRequest = serde_json::from_value(raw_body.clone())
		.map_err(|e| AppError::BadRequest(format!("Invalid Anthropic request: {e}")))?;

	let uuid_str = uuid::Uuid::new_v4().to_string();
	let request_id = uuid_str[..8].to_string();
	let session_id = session_header.unwrap_or_else(|| format!("anth:{request_id}"));
	let stream = client_requested_stream(&raw_body);

	// Surface (do not hide) any client top-level fields CCP does not forward:
	// fingerprint/billing fields dropped by design, plus any unknown field the
	// closed allowlist omits. Observable in logs rather than silently vanishing.
	let dropped = crate::translate::anthropic_passthrough::dropped_fields(&raw_body);
	if !dropped.is_empty() {
		warn!(?dropped, "anthropic request: dropped non-forwarded client body fields");
	}

	if let Some(ref log) = state.conversation_log {
		let raw = std::str::from_utf8(&body)
			.map(str::to_string)
			.unwrap_or_else(|_| String::from_utf8_lossy(&body).into_owned());
		log.log(&session_id, &request_id, ">>>", "Inbound Anthropic body", &raw);
	}

	let model_def = state.fingerprint_profile.resolve_model(&client.model);
	state
		.stats
		.record_request(model_def.canonical, None);

	let span = tracing::info_span!(
		"request",
		request_id = %request_id,
		session_id = %session_id,
		model = %client.model,
		stream = %stream,
		surface = "anthropic",
	);

	async move {
		if stream {
			handle_messages_streaming(state, client, raw_body, request_id, session_id).await
		} else {
			handle_messages_non_streaming(state, client, raw_body, request_id, session_id).await
		}
	}
	.instrument(span)
	.await
}

async fn handle_messages_non_streaming(
	state: Arc<AppState>,
	client: ClientMessagesRequest,
	raw_body: serde_json::Value,
	request_id: String,
	session_id: String,
) -> Result<Response, AppError> {
	let _active = crate::stats::ActiveRequestGuard::new(&state.stats);
	let start = std::time::Instant::now();

	let anth_req = reconcile_client_request(
		&client,
		&raw_body,
		state.fingerprint_profile,
		state.replacements.as_ref(),
		!state.config.no_preamble,
		false,
	)?;
	let model_canonical = state.fingerprint_profile.resolve_model(&client.model).canonical;

	let creds = load_creds(&state, model_canonical).await?;
	let ctx = RequestContext::new_reply()
		.with_session(derive_session_uuid(&session_id))
		.with_model(anth_req.model.clone());

	let body = serde_json::to_value(&anth_req).map_err(|e| {
		AppError::ServerError(format!("failed to serialize anthropic request: {e}"))
	})?;

	if let Some(ref log) = state.conversation_log
		&& let Ok(bytes) = state.fingerprint_profile.finalize_body_json(&body, &ctx)
	{
		log.log(
			&session_id,
			&request_id,
			">>>",
			"Anthropic request",
			&String::from_utf8_lossy(&bytes),
		);
	}

	let mut resp_value = match state.upstream.send_messages_json(&creds, &ctx, &body).await {
		Ok(v) => v,
		Err(e) => {
			let app_err = map_upstream_err(e);
			state.stats.record_error(model_canonical, &app_err.to_string());
			return Err(app_err);
		}
	};

	// Record usage from the raw response (same numbers the typed path recorded).
	let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
	record_usage_from_value(&state, model_canonical, &resp_value, None, duration_ms);

	// Inbound replacements on the raw response (no-op without response rules).
	apply_response_replacements_raw(&mut resp_value, state.replacements.as_ref());

	if let Some(ref log) = state.conversation_log
		&& let Ok(text) = serde_json::to_string(&resp_value)
	{
		log.log(&session_id, &request_id, "<<<", "Anthropic response", &text);
	}

	info!(duration_ms = duration_ms.round() as u64, "anthropic non-streaming finished");
	let headers = [(
		header::HeaderName::from_static("x-request-id"),
		request_id_header(&request_id),
	)];
	Ok((headers, axum::Json(resp_value)).into_response())
}

async fn handle_messages_streaming(
	state: Arc<AppState>,
	client: ClientMessagesRequest,
	raw_body: serde_json::Value,
	request_id: String,
	session_id: String,
) -> Result<Response, AppError> {
	let start = std::time::Instant::now();
	let active = crate::stats::OwnedActiveRequestGuard::new(state.stats.clone());

	let anth_req = reconcile_client_request(
		&client,
		&raw_body,
		state.fingerprint_profile,
		state.replacements.as_ref(),
		!state.config.no_preamble,
		true,
	)?;
	let model_canonical = state.fingerprint_profile.resolve_model(&client.model).canonical;

	let creds = load_creds(&state, model_canonical).await?;
	let ctx = RequestContext::new_reply()
		.with_session(derive_session_uuid(&session_id))
		.with_model(anth_req.model.clone());

	let body = serde_json::to_value(&anth_req).map_err(|e| {
		AppError::ServerError(format!("failed to serialize anthropic request: {e}"))
	})?;

	if let Some(ref log) = state.conversation_log
		&& let Ok(bytes) = state.fingerprint_profile.finalize_body_json(&body, &ctx)
	{
		log.log(
			&session_id,
			&request_id,
			">>>",
			"Anthropic streaming request",
			&String::from_utf8_lossy(&bytes),
		);
	}

	let upstream_stream = state
		.upstream
		.send_messages_stream_raw(&creds, &ctx, &body)
		.await
		.map_err(|e| {
			let app_err = map_upstream_err(e);
			state.stats.record_error(model_canonical, &app_err.to_string());
			app_err
		})?;

	let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
	let replacements = state.replacements.clone();
	let stats = state.stats.clone();
	let span = tracing::Span::current();

	tokio::spawn(
		async move {
			let _active = active;
			let mut repl_state = RawSseReplState::new(&replacements);
			let mut usage = crate::stats::TokenUsage::default();
			let mut ttft_ms: Option<f64> = None;
			// Flush headers immediately.
			let _ = tx.send(Ok(Event::default().comment("ok"))).await;

			let mut stream = upstream_stream;
			// Records usage + timing on EVERY exit path (clean end, upstream error,
			// or client disconnect) so aggregates are never silently undercounted.
			// `disconnected` short-circuits trailing sends once the client is gone.
			let mut disconnected = false;

			while let Some(item) = stream.next().await {
				match item {
					Ok(frame) => {
						accumulate_stream_usage(&frame, &mut usage);
						// TTFT is sampled on the UPSTREAM content delta, not the
						// emitted frame: with replacement rules active, the emitted
						// delta is deferred to block close, so keying off emitted
						// frames would measure time-to-block-close, not first token.
						if ttft_ms.is_none() && is_upstream_content_delta(&frame) {
							ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
						}
						for (event, data) in repl_state.on_frame(&frame.event, frame.data, &replacements)
						{
							if send_raw_frame(&tx, &event, &data).await.is_err() {
								disconnected = true;
								break;
							}
						}
						if disconnected {
							break;
						}
					}
					Err(e) => {
						// Flush buffered (rewritten) deltas + close their blocks
						// before the error frame, so a client mid tool-call sees the
						// completed arguments first.
						for (event, data) in repl_state.flush_all(&replacements) {
							if send_raw_frame(&tx, &event, &data).await.is_err() {
								disconnected = true;
								break;
							}
						}
						let msg = truncate_for_sse(&upstream_error_message(&e));
						warn!(error = %msg, "anthropic raw stream upstream error");
						stats.record_error(model_canonical, &msg);
						if !disconnected {
							let err_data = serde_json::json!({
								"type": "error",
								"error": { "type": "api_error", "message": msg },
							});
							let _ = send_raw_frame(&tx, "error", &err_data).await;
						}
						break;
					}
				}
			}

			// Flush trailing buffered deltas (clean stream that ended without a
			// final stop, or a disconnect after a partial block). Skipped when the
			// client is already gone.
			if !disconnected {
				for (event, data) in repl_state.flush_all(&replacements) {
					if send_raw_frame(&tx, &event, &data).await.is_err() {
						break;
					}
				}
			}

			// Stats are recorded on ALL paths, including client disconnect (the
			// upstream work + billing already happened; the counts are real).
			let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
			stats.record_response(model_canonical, usage, ttft_ms, duration_ms);
			info!(disconnected, "anthropic streaming finished");
		}
		.instrument(span),
	);

	let stream = ReceiverStream::new(rx);
	let sse = Sse::new(stream).keep_alive(KeepAlive::default());
	let mut response = sse.into_response();
	response.headers_mut().insert(
		header::HeaderName::from_static("x-request-id"),
		request_id_header(&request_id),
	);
	Ok(response)
}

/// `POST /v1/messages/count_tokens`. Counts the client body as-sent (no identity
/// injection), per the native-surface decision.
pub async fn count_tokens_handler(
	State(state): State<Arc<AppState>>,
	request: axum::http::Request<axum::body::Body>,
) -> Response {
	match count_tokens_inner(state, request).await {
		Ok(resp) => resp,
		Err(err) => anthropic_error_response(&err),
	}
}

async fn count_tokens_inner(
	state: Arc<AppState>,
	request: axum::http::Request<axum::body::Body>,
) -> Result<Response, AppError> {
	let session_header = request
		.headers()
		.get("x-session-id")
		.and_then(|v| v.to_str().ok())
		.map(str::to_string);
	let body = axum::body::to_bytes(request.into_body(), crate::MAX_BODY_SIZE)
		.await
		.map_err(|e| AppError::BadRequest(format!("Failed to read body: {e}")))?;
	let raw_body: serde_json::Value = serde_json::from_slice(&body)
		.map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;
	let client: ClientMessagesRequest = serde_json::from_value(raw_body.clone())
		.map_err(|e| AppError::BadRequest(format!("Invalid Anthropic request: {e}")))?;

	let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
	let session_id = session_header.unwrap_or_else(|| format!("anth-ct:{request_id}"));

	let count_req = build_count_tokens_request(
		&client,
		&raw_body,
		state.fingerprint_profile,
		state.replacements.as_ref(),
	)?;
	let model_canonical = state.fingerprint_profile.resolve_model(&client.model).canonical;
	let creds = load_creds(&state, model_canonical).await?;
	let ctx = RequestContext::new_reply()
		.with_session(derive_session_uuid(&session_id))
		.with_model(count_req.model.clone());

	let mut count_body = serde_json::to_value(&count_req).map_err(|e| {
		AppError::ServerError(format!("failed to serialize count_tokens request: {e}"))
	})?;
	// Anthropic's count_tokens endpoint accepts only the structural fields that
	// affect the token count (model, messages, system, tools, tool_choice,
	// thinking). Sampling / output / control fields are NOT part of its schema and
	// are rejected with a 400 ("Extra inputs are not permitted"). `MessagesRequest`
	// carries them (and CCP fills wire defaults), so strip them from the count body.
	if let Some(obj) = count_body.as_object_mut() {
		for k in [
			"max_tokens",
			"temperature",
			"top_p",
			"top_k",
			"stop_sequences",
			"stream",
			"output_config",
			"metadata",
		] {
			obj.remove(k);
		}
	}

	if let Some(ref log) = state.conversation_log
		&& let Ok(text) = serde_json::to_string(&count_body)
	{
		log.log(&session_id, &request_id, ">>>", "Anthropic count_tokens", &text);
	}

	let value = state
		.upstream
		.count_tokens(&creds, &ctx, &count_body)
		.await
		.map_err(|e| {
			let app_err = map_upstream_err(e);
			state.stats.record_error(model_canonical, &app_err.to_string());
			app_err
		})?;

	Ok(axum::Json(value).into_response())
}

// ── helpers ───────────────────────────────────────────────────────────────────

async fn load_creds(state: &Arc<AppState>, model_canonical: &str) -> Result<Credentials, AppError> {
	let creds = Credentials::load_fresh_async(&Credentials::default_path())
		.await
		.map_err(|e| {
			let app_err = map_upstream_err(e);
			state.stats.record_error(model_canonical, &app_err.to_string());
			app_err
		})?;
	creds.check_expired().map_err(|e| {
		let app_err = map_upstream_err(e);
		state.stats.record_error(model_canonical, &app_err.to_string());
		app_err
	})?;
	Ok(creds)
}

fn request_id_header(request_id: &str) -> header::HeaderValue {
	header::HeaderValue::from_str(request_id)
		.unwrap_or_else(|_| header::HeaderValue::from_static("unknown"))
}

/// Render an `AppError` as an Anthropic-shaped error HTTP response.
fn anthropic_error_response(err: &AppError) -> Response {
	let (status, body) = anthropic_error_body(err);
	(status, axum::Json(body)).into_response()
}

/// Serialize a raw `(event, data)` SSE frame and send it. Anthropic SSE carries
/// both the `event:` line and the `data:` JSON; axum's `Event` sets both.
async fn send_raw_frame(
	tx: &mpsc::Sender<Result<Event, Infallible>>,
	event: &str,
	data: &serde_json::Value,
) -> Result<(), ()> {
	// A `serde_json::Value` round-trips to a string in practice, but on the
	// near-impossible failure we end the stream (Err) rather than silently drop a
	// frame and leave the client with a gapped stream and no terminal signal.
	let payload = match serde_json::to_string(data) {
		Ok(s) => s,
		Err(e) => {
			error!("anthropic sse frame serialize: {e}");
			return Err(());
		}
	};
	let ev = Event::default().event(event).data(payload);
	tx.send(Ok(ev)).await.map_err(|_| ())
}

fn record_usage_from_value(
	state: &Arc<AppState>,
	model_canonical: &str,
	resp: &serde_json::Value,
	ttft_ms: Option<f64>,
	duration_ms: f64,
) {
	let usage = resp.get("usage");
	let get = |k: &str| usage.and_then(|u| u.get(k)).and_then(|v| v.as_u64()).unwrap_or(0);
	state.stats.record_response(
		model_canonical,
		crate::stats::TokenUsage {
			input_tokens: get("input_tokens"),
			output_tokens: get("output_tokens"),
			cache_read_input_tokens: get("cache_read_input_tokens"),
			cache_creation_input_tokens: get("cache_creation_input_tokens"),
		},
		ttft_ms,
		duration_ms,
	);
}

/// Pull token counts out of streaming `message_start` / `message_delta` frames,
/// including cache tokens (which appear in `message_start.message.usage`), so
/// streaming stats match the non-streaming path rather than reporting 0 cache.
fn accumulate_stream_usage(
	frame: &crate::upstream::stream::RawFrame,
	usage: &mut crate::stats::TokenUsage,
) {
	let read = |u: &serde_json::Value, k: &str| u.get(k).and_then(|v| v.as_u64());
	match frame.event.as_str() {
		"message_start" => {
			if let Some(u) = frame.data.get("message").and_then(|m| m.get("usage")) {
				if let Some(v) = read(u, "input_tokens") {
					usage.input_tokens = v;
				}
				if let Some(v) = read(u, "output_tokens") {
					usage.output_tokens = v;
				}
				if let Some(v) = read(u, "cache_read_input_tokens") {
					usage.cache_read_input_tokens = v;
				}
				if let Some(v) = read(u, "cache_creation_input_tokens") {
					usage.cache_creation_input_tokens = v;
				}
			}
		}
		"message_delta" => {
			// `message_delta.usage` reports the running output_tokens (and may
			// update cache_* on some responses); take the latest values.
			if let Some(u) = frame.data.get("usage") {
				if let Some(v) = read(u, "output_tokens") {
					usage.output_tokens = v;
				}
				if let Some(v) = read(u, "cache_read_input_tokens") {
					usage.cache_read_input_tokens = v;
				}
				if let Some(v) = read(u, "cache_creation_input_tokens") {
					usage.cache_creation_input_tokens = v;
				}
			}
		}
		_ => {}
	}
}

/// Whether an UPSTREAM frame is a content delta (text or tool-input), used to
/// time TTFT at the first real token regardless of replacement buffering.
fn is_upstream_content_delta(frame: &crate::upstream::stream::RawFrame) -> bool {
	frame.event == "content_block_delta"
		&& frame
			.data
			.get("delta")
			.and_then(|d| d.get("type"))
			.and_then(|t| t.as_str())
			.is_some_and(|t| t == "text_delta" || t == "input_json_delta")
}

/// Extract a human-readable message from an upstream error.
fn upstream_error_message(e: &crate::upstream::errors::UpstreamError) -> String {
	match e {
		crate::upstream::errors::UpstreamError::Anthropic {
			parsed: Some(p), ..
		} => p.error.message.clone(),
		_ => e.to_string(),
	}
}

/// Bound a message surfaced in an SSE error frame, mirroring the HTTP error
/// path's truncation so an unbounded upstream string is never echoed verbatim.
fn truncate_for_sse(s: &str) -> String {
	const MAX: usize = 500;
	if s.chars().count() <= MAX {
		return s.to_string();
	}
	let mut out: String = s.chars().take(MAX).collect();
	out.push_str("… (truncated)");
	out
}
