//! v2 completions handler — talks directly to api.anthropic.com via the
//! upstream client. Native Anthropic Messages format upstream, OAI-completions
//! format inbound. No subprocess.
//!
//! Phase 2: non-streaming only. Streaming is added in Phase 3.

use std::convert::Infallible;
use std::sync::Arc;

use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{Instrument, error, info, warn};

use crate::AppState;
use crate::error::AppError;
use crate::models::ModelDef;
use crate::translate::anthropic::MessagesResponse;
use crate::translate::build::build_messages_request;
use crate::translate::from_anthropic::build_oai_response;
use crate::translate::request::ChatCompletionRequest;
use crate::translate::to_oai_stream::OaiStreamConverter;
use crate::upstream::credentials::Credentials;
use crate::upstream::errors::UpstreamError;
use crate::upstream::fingerprint::RequestContext;

pub async fn handle_non_streaming_v2(
	state: Arc<AppState>,
	request: ChatCompletionRequest,
	model_def: &'static ModelDef,
	request_id: String,
	chat_id: String,
	created: u64,
	session_id: String,
	conv_log: Option<Arc<crate::conversation_log::ConversationLog>>,
) -> Result<Response, AppError> {
	let _active = crate::stats::ActiveRequestGuard::new(&state.stats);
	let start = std::time::Instant::now();

	// Build the Anthropic request body.
	let mut anth_req = build_messages_request(&request, model_def, !state.config.no_preamble)?;
	// Force non-streaming on the upstream call.
	anth_req.stream = Some(false);

	// Apply replacements to all outbound text. Cheap pass over the body.
	if !state.replacements.is_empty() {
		apply_replacements_outbound(&mut anth_req, state.replacements.as_ref());
	}

	if let Some(ref log) = conv_log {
		match serde_json::to_string(&anth_req) {
			Ok(json) => log.log(&request_id, ">>>", "Anthropic request", &json),
			Err(_) => {}
		}
	}

	let creds_path = Credentials::default_path();
	let creds = Credentials::load_fresh(&creds_path).map_err(map_upstream_err)?;
	if let Err(e) = creds.check_expired() {
		return Err(map_upstream_err(e));
	}

	// Build per-request context. Session ID is derived from CCP's resolved
	// session id so multi-turn requests share an identifier.
	let session_uuid = derive_session_uuid(&session_id);
	let ctx = RequestContext::new_reply().with_session(session_uuid);

	let body = serde_json::to_value(&anth_req).map_err(|e| {
		AppError::ServerError(format!("failed to serialize anthropic request: {e}"))
	})?;

	let resp_value = state
		.upstream
		.send_messages_json(&creds, &ctx, &body)
		.await
		.map_err(map_upstream_err)?;

	let resp: MessagesResponse = serde_json::from_value(resp_value.clone()).map_err(|e| {
		AppError::ServerError(format!("anthropic response decode: {e} (raw: {})", resp_value))
	})?;

	let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

	tracing::debug!(
		input_tokens = resp.usage.input_tokens,
		output_tokens = resp.usage.output_tokens,
		model = %resp.model,
		"v2 completion usage"
	);

	let mut oai_response = build_oai_response(&resp, &chat_id, created, &request.model);

	// Apply replacements to the assistant text content if present.
	if !state.replacements.is_empty() {
		apply_replacements_inbound(&mut oai_response, state.replacements.as_ref());
	}

	if let Some(ref log) = conv_log {
		if let Ok(text) = serde_json::to_string(&oai_response) {
			log.log(&request_id, "<<<", "OAI response", &text);
		}
	}

	let finish_reason = oai_response
		.get("choices")
		.and_then(|c| c.as_array())
		.and_then(|arr| arr.first())
		.and_then(|c| c.get("finish_reason"))
		.and_then(|f| f.as_str())
		.unwrap_or("stop")
		.to_string();

	info!(
		duration_ms = duration_ms.round() as u64,
		finish_reason = %finish_reason,
		"v2 non-streaming completion finished"
	);

	let headers = [(
		header::HeaderName::from_static("x-request-id"),
		header::HeaderValue::from_str(&request_id).unwrap_or_else(|_| header::HeaderValue::from_static("unknown")),
	)];
	Ok((headers, axum::Json(oai_response)).into_response())
}

pub async fn handle_streaming_v2(
	state: Arc<AppState>,
	request: ChatCompletionRequest,
	model_def: &'static ModelDef,
	request_id: String,
	chat_id: String,
	created: u64,
	session_id: String,
	conv_log: Option<Arc<crate::conversation_log::ConversationLog>>,
) -> Result<Response, AppError> {
	let _active = crate::stats::ActiveRequestGuard::new(&state.stats);

	let mut anth_req = build_messages_request(&request, model_def, !state.config.no_preamble)?;
	anth_req.stream = Some(true);

	if !state.replacements.is_empty() {
		apply_replacements_outbound(&mut anth_req, state.replacements.as_ref());
	}

	if let Some(ref log) = conv_log {
		if let Ok(json) = serde_json::to_string(&anth_req) {
			log.log(&request_id, ">>>", "Anthropic streaming request", &json);
		}
	}

	let creds_path = Credentials::default_path();
	let creds = Credentials::load_fresh(&creds_path).map_err(map_upstream_err)?;
	creds.check_expired().map_err(map_upstream_err)?;

	let session_uuid = derive_session_uuid(&session_id);
	let ctx = RequestContext::new_reply().with_session(session_uuid);

	let body = serde_json::to_value(&anth_req).map_err(|e| {
		AppError::ServerError(format!("failed to serialize anthropic request: {e}"))
	})?;

	let upstream_stream = state
		.upstream
		.send_messages_stream(&creds, &ctx, &body)
		.await
		.map_err(map_upstream_err)?;

	let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

	let conv_request_id = request_id.clone();
	let conv_log_for_task = conv_log.clone();
	let replacements = state.replacements.clone();
	let requested_model = request.model.clone();
	let span = tracing::Span::current();

	tokio::spawn(
		async move {
			let mut converter = OaiStreamConverter::new(chat_id, created, requested_model);
			// Initial :ok comment to flush headers immediately.
			let _ = tx.send(Ok(Event::default().comment("ok"))).await;

			let mut stream = upstream_stream;
			let mut accumulated_text = String::new();
			let mut errored = false;

			while let Some(item) = stream.next().await {
				match item {
					Ok(event) => {
						let chunks = converter.on_event(event);
						for chunk in chunks {
							// Apply replacements only to plain text content deltas.
							let chunk = if !replacements.is_empty() {
								apply_stream_replacements(chunk, &replacements, &mut accumulated_text)
							} else {
								if let Some(text) = extract_content_delta(&chunk) {
									accumulated_text.push_str(text);
								}
								chunk
							};
							match serde_json::to_string(&chunk) {
								Ok(s) => {
									if tx.send(Ok(Event::default().data(s))).await.is_err() {
										return;
									}
								}
								Err(e) => {
									error!("v2 stream chunk serialize: {e}");
								}
							}
						}
					}
					Err(e) => {
						errored = true;
						let msg = match &e {
							UpstreamError::Anthropic { parsed: Some(p), .. } => p.error.message.clone(),
							_ => e.to_string(),
						};
						warn!(error = %msg, "v2 stream upstream error");
						let payload = serde_json::json!({
							"error": {
								"type": "upstream_error",
								"message": msg,
							}
						});
						if let Ok(s) = serde_json::to_string(&payload) {
							let _ = tx.send(Ok(Event::default().data(s))).await;
						}
						break;
					}
				}
			}

			if !errored {
				for chunk in converter.finalize_if_needed() {
					if let Ok(s) = serde_json::to_string(&chunk) {
						let _ = tx.send(Ok(Event::default().data(s))).await;
					}
				}
			}

			let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

			if let Some(log) = conv_log_for_task {
				log.log(
					&conv_request_id,
					"<<<",
					"Streaming response (text accumulator)",
					&accumulated_text,
				);
			}

			info!("v2 streaming completion finished");
		}
		.instrument(span),
	);

	let stream = ReceiverStream::new(rx);
	let sse = Sse::new(stream).keep_alive(KeepAlive::default());
	Ok(sse.into_response())
}

/// Pull a `delta.content` text fragment from an outbound chunk if present.
fn extract_content_delta(chunk: &serde_json::Value) -> Option<&str> {
	chunk
		.get("choices")?
		.as_array()?
		.first()?
		.get("delta")?
		.get("content")?
		.as_str()
}

/// Apply replacements to a chunk's `delta.content` and `delta.tool_calls[].function.name`
/// if present, returning the transformed chunk. Also accumulates text into
/// `accumulator` for logging. Tool-call argument fragments are NOT rewritten —
/// the rename strings could straddle chunk boundaries and partial-fragment
/// replacement would corrupt arguments.
fn apply_stream_replacements(
	mut chunk: serde_json::Value,
	repl: &crate::replacements::Replacements,
	accumulator: &mut String,
) -> serde_json::Value {
	if let Some(text) = extract_content_delta(&chunk).map(str::to_string) {
		let replaced = repl.apply_response(&text);
		accumulator.push_str(&replaced);
		if let Some(delta) = chunk
			.get_mut("choices")
			.and_then(|c| c.as_array_mut())
			.and_then(|arr| arr.first_mut())
			.and_then(|c| c.get_mut("delta"))
		{
			delta["content"] = serde_json::Value::String(replaced);
		}
	}
	if let Some(tool_calls) = chunk
		.get_mut("choices")
		.and_then(|c| c.as_array_mut())
		.and_then(|arr| arr.first_mut())
		.and_then(|c| c.get_mut("delta"))
		.and_then(|d| d.get_mut("tool_calls"))
		.and_then(|t| t.as_array_mut())
	{
		for call in tool_calls {
			if let Some(function) = call.get_mut("function").and_then(|f| f.as_object_mut()) {
				if let Some(name) = function.get_mut("name") {
					if let Some(s) = name.as_str() {
						*name = serde_json::Value::String(repl.apply_response(s));
					}
				}
			}
		}
	}
	chunk
}

fn map_upstream_err(e: UpstreamError) -> AppError {
	let surface = e.surface_status();
	let mut msg = match &e {
		UpstreamError::Anthropic { parsed: Some(p), .. } => p.error.message.clone(),
		_ => e.to_string(),
	};
	// Anthropic frequently returns the literal string "Error" as the
	// message; surface enough context for operators.
	if msg == "Error" {
		if let UpstreamError::Anthropic { parsed: Some(p), status, .. } = &e {
			msg = format!("upstream {} ({}): {}", status, p.error.kind, p.error.message);
		} else if let UpstreamError::Anthropic { status, body, .. } = &e {
			msg = format!("upstream {}: {}", status, body);
		}
	}
	match surface {
		429 => AppError::RateLimited(msg),
		401 | 403 => AppError::Unauthorized(msg),
		400..=499 => AppError::BadRequest(msg),
		504 => AppError::Timeout(msg),
		_ => AppError::ServerError(msg),
	}
}

fn derive_session_uuid(session_id: &str) -> uuid::Uuid {
	// Prefer parsing the inbound session id as a UUID; fall back to v5 hash
	// in DNS namespace so the same session id always maps to the same UUID.
	if let Ok(u) = uuid::Uuid::parse_str(session_id) {
		return u;
	}
	uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, session_id.as_bytes())
}

fn apply_replacements_outbound(
	req: &mut crate::translate::anthropic::MessagesRequest,
	repl: &crate::replacements::Replacements,
) {
	use crate::translate::anthropic::{MessageContent, SystemField};
	match &mut req.system {
		Some(SystemField::Text(s)) => *s = repl.apply_prompt(s),
		Some(SystemField::Blocks(blocks)) => {
			for b in blocks {
				b.text = repl.apply_prompt(&b.text);
			}
		}
		None => {}
	}
	for m in &mut req.messages {
		match &mut m.content {
			MessageContent::Text(s) => *s = repl.apply_prompt(s),
			MessageContent::Blocks(blocks) => {
				for b in blocks {
					apply_prompt_to_content_block(b, repl);
				}
			}
		}
	}
	if let Some(tools) = req.tools.as_mut() {
		for t in tools {
			t.name = repl.apply_prompt(&t.name);
			if let Some(d) = t.description.as_mut() {
				*d = repl.apply_prompt(d);
			}
			apply_prompt_to_json(&mut t.input_schema, repl);
		}
	}
}

fn apply_prompt_to_content_block(
	block: &mut crate::translate::anthropic::ContentBlock,
	repl: &crate::replacements::Replacements,
) {
	use crate::translate::anthropic::{ContentBlock, ToolResultContent};
	match block {
		ContentBlock::Text { text, .. } => *text = repl.apply_prompt(text),
		ContentBlock::ToolUse { name, input, .. } => {
			*name = repl.apply_prompt(name);
			apply_prompt_to_json(input, repl);
		}
		ContentBlock::ToolResult { content, .. } => match content {
			Some(ToolResultContent::Text(t)) => *t = repl.apply_prompt(t),
			Some(ToolResultContent::Blocks(blocks)) => {
				for b in blocks {
					apply_prompt_to_content_block(b, repl);
				}
			}
			None => {}
		},
		ContentBlock::Thinking { .. } | ContentBlock::Image { .. } => {}
	}
}

fn apply_prompt_to_json(
	value: &mut serde_json::Value,
	repl: &crate::replacements::Replacements,
) {
	match value {
		serde_json::Value::String(s) => *s = repl.apply_prompt(s),
		serde_json::Value::Array(arr) => {
			for v in arr {
				apply_prompt_to_json(v, repl);
			}
		}
		serde_json::Value::Object(obj) => {
			for (_, v) in obj.iter_mut() {
				apply_prompt_to_json(v, repl);
			}
		}
		_ => {}
	}
}

fn apply_replacements_inbound(
	resp: &mut serde_json::Value,
	repl: &crate::replacements::Replacements,
) {
	let Some(choices) = resp.get_mut("choices").and_then(|c| c.as_array_mut()) else {
		return;
	};
	for c in choices {
		let Some(message) = c.get_mut("message").and_then(|m| m.as_object_mut()) else {
			continue;
		};
		if let Some(content) = message.get_mut("content") {
			if let Some(s) = content.as_str() {
				let replaced = repl.apply_response(s);
				*content = serde_json::Value::String(replaced);
			}
		}
		if let Some(tool_calls) = message.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
			for call in tool_calls {
				if let Some(function) = call.get_mut("function").and_then(|f| f.as_object_mut()) {
					if let Some(name) = function.get_mut("name") {
						if let Some(s) = name.as_str() {
							*name = serde_json::Value::String(repl.apply_response(s));
						}
					}
					if let Some(args) = function.get_mut("arguments") {
						if let Some(s) = args.as_str() {
							*args = serde_json::Value::String(repl.apply_response(s));
						}
					}
				}
			}
		}
	}
}
