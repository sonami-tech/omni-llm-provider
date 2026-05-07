use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{Instrument, error, info, warn};

use crate::AppState;
use crate::auth::ApiKeyId;
use crate::error::AppError;
use crate::models::{normalize_model_name, resolve_model, validate_effort};
use crate::subprocess::SubprocessEvent;
use crate::subprocess::manager::spawn_managed;
use crate::translate::request::{
	ChatCompletionRequest, build_cli_args, build_prompt_and_system, compose_system_prompt,
	validate_request,
};
use crate::translate::response::{build_response, build_tool_call_response, extract_usage};
use crate::translate::stream;
use crate::translate::tools::{
	ParsedResponse, ResolvedToolChoice, build_tool_prompt_prefix, malformed_tool_call_message,
	parse_tool_response, resolve_tool_choice, to_response_tool_calls,
};

pub async fn completions_handler(
	State(state): State<Arc<AppState>>,
	request: axum::http::Request<axum::body::Body>,
) -> Result<Response, AppError> {
	let api_key_id = request.extensions().get::<ApiKeyId>().map(|k| k.0.clone());
	let body = axum::body::to_bytes(request.into_body(), crate::MAX_BODY_SIZE)
		.await
		.map_err(|e| AppError::BadRequest(format!("Failed to read body: {}", e)))?;
	// Parse body manually for OpenAI-format error on bad JSON.
	let request: ChatCompletionRequest = serde_json::from_slice(&body)
		.map_err(|e| AppError::BadRequest(format!("Invalid JSON: {}", e)))?;

	// Generate request ID: first 8 hex chars of UUID v4.
	let uuid_str = uuid::Uuid::new_v4().to_string();
	let request_id = &uuid_str[..8];
	let chat_id = format!("chatcmpl-{request_id}");

	let span = tracing::info_span!(
		"request",
		request_id = %request_id,
		model = %request.model,
		stream = %request.stream,
	);

	// Run the rest of the handler instrumented with the request span so the
	// request_id is attached to every log line, including those emitted from
	// detached tasks (subprocess driver, streaming converter) that explicitly
	// adopt this span via `.instrument(Span::current())`.
	async move {

	// Validate.
	validate_request(&request)?;
	let model_def = resolve_model(&request.model);
	let effort = validate_effort(request.reasoning_effort.as_deref())?;

	// Resolve tool passthrough.
	let resolved_choice = resolve_tool_choice(&request.tool_choice);
	let tools_active = !state.config.no_tool_passthrough
		&& request.tools.as_ref().is_some_and(|t| !t.is_empty())
		&& !matches!(resolved_choice, ResolvedToolChoice::None);

	// Build prompt and system prompt.
	let (mut prompt, mut system_prompt) = build_prompt_and_system(&request.messages)?;

	if tools_active {
		let tools = request.tools.as_ref().unwrap();
		let prefix = build_tool_prompt_prefix(tools, &resolved_choice);
		prompt = format!("{}{}", prefix, prompt);
	}

	if !state.replacements.is_empty() {
		prompt = state.replacements.apply_prompt(&prompt);
		system_prompt = system_prompt.map(|sp| state.replacements.apply_prompt(&sp));
	}

	// Always set --system-prompt so the CLI's built-in agentic prompt is
	// replaced (not appended to). Otherwise it dominates and triggers
	// tool-call retry loops in single-shot proxy mode.
	let final_system_prompt = compose_system_prompt(system_prompt.as_deref());

	// System prompt is passed as a CLI argument; check against the Linux
	// MAX_ARG_STRLEN limit (~128KB per argument). The main prompt is piped
	// via stdin, so it has no kernel size limit.
	const MAX_ARG_LEN: usize = 128_000;
	if final_system_prompt.len() > MAX_ARG_LEN {
		return Err(AppError::BadRequest(format!(
			"System prompt too large ({} bytes, max {} bytes)",
			final_system_prompt.len(),
			MAX_ARG_LEN
		)));
	}

	let cli_args = build_cli_args(model_def, Some(&final_system_prompt), effort);

	let created = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_secs();

	// Record stats.
	state.stats.record_request(model_def.canonical, api_key_id.as_deref());

	info!(key = api_key_id.as_deref().unwrap_or("-"), "Chat completion request");

	if let Some(ref log) = state.conversation_log {
		log.log(request_id, ">>>", "Prompt", &prompt);
		if let Some(ref sp) = system_prompt {
			log.log(request_id, ">>>", "System", sp);
		}
	}

	let conv_log = state.conversation_log.clone();
	let request_id = request_id.to_string();
	if request.stream {
		handle_streaming(state, request_id, chat_id, created, model_def.canonical, cli_args, prompt, conv_log, tools_active).await
	} else {
		handle_non_streaming(state, request_id, chat_id, created, model_def.canonical, cli_args, prompt, conv_log, tools_active)
			.await
	}

	}
	.instrument(span)
	.await
}

async fn handle_non_streaming(
	state: Arc<AppState>,
	request_id: String,
	chat_id: String,
	created: u64,
	requested_model: &'static str,
	cli_args: Vec<String>,
	prompt: String,
	conv_log: Option<Arc<crate::conversation_log::ConversationLog>>,
	tools_active: bool,
) -> Result<Response, AppError> {
	let _active = crate::stats::ActiveRequestGuard::new(&state.stats);
	let start = std::time::Instant::now();
	let (tx, mut rx) = mpsc::channel::<SubprocessEvent>(64);

	spawn_managed(
		state.config.clone(),
		state.semaphore.clone(),
		Duration::from_secs(state.config.queue_timeout),
		request_id.to_string(),
		cli_args,
		prompt,
		tx,
	)
	.await?;

	// Collect all events.
	let mut content = String::new();
	let mut model = requested_model.to_string();
	let mut result_msg = None;
	let mut error_msg = None;
	let mut ttft_ms: Option<f64> = None;

	while let Some(event) = rx.recv().await {
		match event {
			SubprocessEvent::Model(m) => {
				model = normalize_model_name(&m).into_owned();
			}
			SubprocessEvent::ContentDelta(text) => {
				if ttft_ms.is_none() {
					ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
				}
				content.push_str(&text);
			}
			SubprocessEvent::Result(r) => {
				result_msg = Some(r);
				// Result is terminal: stop reading so the request returns
				// promptly even if the subprocess task is still finishing
				// teardown. Closing the receiver also signals the sender
				// side to wind down cleanly.
				rx.close();
				break;
			}
			SubprocessEvent::Error(e) => {
				error_msg = Some(e);
				rx.close();
				break;
			}
		}
	}

	let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

	// Check for subprocess errors.
	if let Some(err) = error_msg {
		state.stats.record_error(&model, &err);
		if err.contains("Inactivity timeout") {
			return Err(AppError::Timeout(err));
		}
		return Err(AppError::ServerError(err));
	}

	let result = result_msg.ok_or_else(|| {
		let msg = "Process exited without producing a result".to_string();
		state.stats.record_error(&model, &msg);
		AppError::ServerError(msg)
	})?;

	// Check is_error FIRST (can be true even with subtype "success").
	if result.is_error.unwrap_or(false) {
		let msg = result.result.clone().unwrap_or_else(|| {
			format!(
				"CLI returned an error with no message (subtype: {})",
				result.subtype.as_deref().unwrap_or("none")
			)
		});
		state.stats.record_error(&model, &msg);
		return Err(AppError::ServerError(msg));
	}

	// Record successful completion.
	state.stats.record_completion(&model, ttft_ms, duration_ms, &result);

	// Fall back to result.result if no deltas were collected.
	if content.is_empty() {
		content = result.result.clone().unwrap_or_default();
	}

	if !state.replacements.is_empty() {
		content = state.replacements.apply_response(&content);
	}

	// Parse for tool calls if tools are active.
	if tools_active {
		match parse_tool_response(&content) {
			ParsedResponse::ToolCalls(calls) => {
				let tool_calls = to_response_tool_calls(calls);

				if let Some(ref log) = conv_log {
					log.log(&request_id, "<<<", "Tool calls", &content);
				}

				let n = tool_calls.len();
				let response =
					build_tool_call_response(&chat_id, created, &model, tool_calls, &result);

				log_completion_finished("tool_calls", n, false, duration_ms.round() as u64);

				let headers = [(
					header::HeaderName::from_static("x-request-id"),
					header::HeaderValue::from_str(&request_id).unwrap(),
				)];

				return Ok((headers, axum::Json(response)).into_response());
			}
			ParsedResponse::MalformedToolCall(err) => {
				warn!(error = %err, "Malformed tool call attempt detected");
				if let Some(ref log) = conv_log {
					log.log(&request_id, "<<<", "Malformed tool call", &content);
				}
				content = malformed_tool_call_message(&err);
			}
			ParsedResponse::Text => {}
		}
	}

	if let Some(ref log) = conv_log {
		log.log(&request_id, "<<<", "Response", &content);
	}

	let response = build_response(&chat_id, created, &model, &content, &result);

	log_completion_finished("stop", 0, false, duration_ms.round() as u64);

	let headers = [(
		header::HeaderName::from_static("x-request-id"),
		header::HeaderValue::from_str(&request_id).unwrap(),
	)];

	Ok((headers, axum::Json(response)).into_response())
}

async fn handle_streaming(
	state: Arc<AppState>,
	request_id: String,
	chat_id: String,
	created: u64,
	requested_model: &'static str,
	cli_args: Vec<String>,
	prompt: String,
	conv_log: Option<Arc<crate::conversation_log::ConversationLog>>,
	tools_active: bool,
) -> Result<Response, AppError> {
	let (sub_tx, mut sub_rx) = mpsc::channel::<SubprocessEvent>(64);
	let (sse_tx, sse_rx) = mpsc::channel::<Result<Event, Infallible>>(64);

	spawn_managed(
		state.config.clone(),
		state.semaphore.clone(),
		Duration::from_secs(state.config.queue_timeout),
		request_id.to_string(),
		cli_args,
		prompt,
		sub_tx,
	)
	.await?;

	// Converter task: SubprocessEvent → SSE Event. Inherit the request span
	// so all logs from this detached task carry the request_id.
	let conv_chat_id = chat_id.clone();
	let conv_request_id = request_id.clone();
	let conv_model = requested_model.to_string();
	let replacements = state.replacements.clone();
	let stats = state.stats.clone();
	let converter_span = tracing::Span::current();
	tokio::spawn(async move {
		let _active = crate::stats::ActiveRequestGuard::new(&stats);
		let start = std::time::Instant::now();
		let mut model = conv_model;
		let mut is_first = true;
		let mut content_sent = false;
		let mut ttft_ms: Option<f64> = None;
		let mut full_content = if conv_log.is_some() || tools_active {
			Some(String::new())
		} else {
			None
		};

		// Initial :ok comment.
		let _ = sse_tx.send(Ok(Event::default().comment("ok"))).await;

		while let Some(event) = sub_rx.recv().await {
			match event {
				SubprocessEvent::Model(m) => {
					model = normalize_model_name(&m).into_owned();
				}
				SubprocessEvent::ContentDelta(text) => {
					if ttft_ms.is_none() {
						ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
					}
					content_sent = true;

					if tools_active {
						// Buffer mode: collect all content, parse for tool calls later.
						if let Some(ref mut buf) = full_content {
							buf.push_str(&text);
						}
					} else {
						// Stream mode: emit chunks immediately.
						let text = if !replacements.is_empty() {
							replacements.apply_response(&text)
						} else {
							text
						};
						if let Some(ref mut buf) = full_content {
							buf.push_str(&text);
						}
						let chunk = stream::content_chunk(
							&conv_chat_id,
							created,
							&model,
							&text,
							is_first,
						);
						is_first = false;

						match serde_json::to_string(&chunk) {
							Ok(json) => {
								if sse_tx.send(Ok(Event::default().data(json))).await.is_err() {
									return;
								}
							}
							Err(e) => {
								error!("Failed to serialize chunk: {}", e);
							}
						}
					}
				}
				SubprocessEvent::Result(result) => {
					let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

					if result.is_error.unwrap_or(false) {
						let msg = result.result.clone().unwrap_or_else(|| {
							format!(
								"CLI returned an error with no message (subtype: {})",
								result.subtype.as_deref().unwrap_or("none")
							)
						});
						stats.record_error(&model, &msg);
						let error_data = stream::error_event_data(&msg);
						let _ = sse_tx.send(Ok(Event::default().data(error_data))).await;
						if content_sent {
							let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
						}
						return;
					}

					stats.record_completion(&model, ttft_ms, duration_ms, &result);

					// Track what we ultimately returned so we can emit a single
					// structured end-of-request log line after [DONE].
					let mut finish_reason: &'static str = "stop";
					let mut n_tool_calls: usize = 0;

					// Emit buffered tool calls or text if in tool mode.
					if tools_active {
						if let Some(ref mut buf) = full_content {
							if !replacements.is_empty() {
								*buf = replacements.apply_response(buf);
							}

							match parse_tool_response(buf) {
								ParsedResponse::ToolCalls(calls) => {
									let tool_calls = to_response_tool_calls(calls);
									n_tool_calls = tool_calls.len();
									finish_reason = "tool_calls";

									if let Some(log) = &conv_log {
										log.log(&conv_request_id, "<<<", "Tool calls", buf);
									}

									for chunk in stream::tool_call_chunks(
										&conv_chat_id,
										created,
										&model,
										&tool_calls,
									) {
										if let Ok(json) = serde_json::to_string(&chunk) {
											let _ = sse_tx
												.send(Ok(Event::default().data(json)))
												.await;
										}
									}

									let finish = stream::finish_chunk(
										&conv_chat_id,
										created,
										&model,
										"tool_calls",
									);
									if let Ok(json) = serde_json::to_string(&finish) {
										let _ = sse_tx
											.send(Ok(Event::default().data(json)))
											.await;
									}
								}
								ParsedResponse::MalformedToolCall(err) => {
									warn!(error = %err, "Malformed tool call attempt detected");
									let message = malformed_tool_call_message(&err);
									emit_text_finish(
										&sse_tx,
										&conv_chat_id,
										created,
										&model,
										&message,
										&conv_log,
										&conv_request_id,
										"Malformed tool call",
									)
									.await;
								}
								ParsedResponse::Text => {
									emit_text_finish(
										&sse_tx,
										&conv_chat_id,
										created,
										&model,
										buf,
										&conv_log,
										&conv_request_id,
										"Response",
									)
									.await;
								}
							}
						}
					} else {
						if let (Some(log), Some(buf)) = (&conv_log, &full_content) {
							log.log(&conv_request_id, "<<<", "Response", buf);
						}

						let finish = stream::finish_chunk(&conv_chat_id, created, &model, "stop");
						if let Ok(json) = serde_json::to_string(&finish) {
							let _ = sse_tx.send(Ok(Event::default().data(json))).await;
						}
					}

					// Emit usage chunk.
					if let Some(usage) = extract_usage(&result) {
						let usage_c =
							stream::usage_chunk(&conv_chat_id, created, &model, usage);
						if let Ok(json) = serde_json::to_string(&usage_c) {
							let _ = sse_tx.send(Ok(Event::default().data(json))).await;
						}
					}

					let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;

					log_completion_finished(finish_reason, n_tool_calls, true, duration_ms.round() as u64);
				}
				SubprocessEvent::Error(msg) => {
					stats.record_error(&model, &msg);
					let error_data = stream::error_event_data(&msg);
					let _ = sse_tx.send(Ok(Event::default().data(error_data))).await;
					if content_sent {
						let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
					}
					return;
				}
			}
		}
	}.instrument(converter_span));

	let sse_stream = ReceiverStream::new(sse_rx);
	let sse = Sse::new(sse_stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));

	let headers = [
		(
			header::HeaderName::from_static("x-request-id"),
			header::HeaderValue::from_str(&request_id).unwrap(),
		),
		(
			header::CACHE_CONTROL,
			header::HeaderValue::from_static("no-cache"),
		),
	];

	Ok((headers, sse).into_response())
}

#[allow(clippy::too_many_arguments)]
async fn emit_text_finish(
	sse_tx: &mpsc::Sender<Result<Event, Infallible>>,
	chat_id: &str,
	created: u64,
	model: &str,
	text: &str,
	conv_log: &Option<Arc<crate::conversation_log::ConversationLog>>,
	request_id: &str,
	log_label: &str,
) {
	if let Some(log) = conv_log {
		log.log(request_id, "<<<", log_label, text);
	}

	let chunk = stream::content_chunk(chat_id, created, model, text, true);
	if let Ok(json) = serde_json::to_string(&chunk) {
		let _ = sse_tx.send(Ok(Event::default().data(json))).await;
	}

	let finish = stream::finish_chunk(chat_id, created, model, "stop");
	if let Ok(json) = serde_json::to_string(&finish) {
		let _ = sse_tx.send(Ok(Event::default().data(json))).await;
	}
}

/// Single structured end-of-request log line for successful responses.
/// Operators read this to tell at a glance whether the client will continue
/// the conversation (`tool_calls`) or whether the request was terminal
/// (`stop`). Malformed tool-call attempts fall back to a synthetic text reply
/// and are reported here as `stop` (see the separate `warn!` for the
/// malformed event itself). Errors are not reported via this log.
fn log_completion_finished(
	finish_reason: &str,
	tool_call_count: usize,
	streaming: bool,
	duration_ms: u64,
) {
	info!(
		finish_reason = finish_reason,
		tool_call_count = tool_call_count,
		streaming = streaming,
		duration_ms = duration_ms,
		"Chat completion finished"
	);
}
