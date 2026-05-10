use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::Response;
use tracing::{Instrument, info};

use crate::AppState;
use crate::auth::ApiKeyId;
use crate::error::AppError;
use crate::models::{resolve_model, validate_effort};
use crate::session::resolve_session_id;
use crate::translate::request::{ChatCompletionRequest, validate_request};

pub async fn completions_handler(
	State(state): State<Arc<AppState>>,
	request: axum::http::Request<axum::body::Body>,
) -> Result<Response, AppError> {
	let api_key_id = request.extensions().get::<ApiKeyId>().map(|k| k.0.clone());
	// Extract correlation header before the body consumes the request.
	let session_header = request
		.headers()
		.get("x-session-id")
		.and_then(|v| v.to_str().ok())
		.map(str::to_string);
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

	// Log the raw inbound body for end-to-end diagnostics. Falls back to a
	// utf8-lossy rendering if the body isn't valid UTF-8.
	if let Some(ref log) = state.conversation_log {
		let raw = std::str::from_utf8(&body)
			.map(str::to_string)
			.unwrap_or_else(|_| String::from_utf8_lossy(&body).into_owned());
		log.log(request_id, ">>>", "Inbound OAI body", &raw);
	}

	let session_id =
		resolve_session_id(session_header.as_deref(), &request, api_key_id.as_deref());

	let span = tracing::info_span!(
		"request",
		request_id = %request_id,
		session_id = %session_id,
		model = %request.model,
		stream = %request.stream,
	);

	// Run the rest of the handler instrumented with the request span so the
	// request_id is attached to every log line, including those emitted from
	// detached streaming tasks that explicitly adopt this span via
	// `.instrument(Span::current())`.
	async move {
	// Validate.
	validate_request(&request)?;
	let model_def = resolve_model(&request.model);
	validate_effort(request.reasoning_effort.as_deref())?;

	let created = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_secs();

	// Record stats.
	state.stats.record_request(model_def.canonical, api_key_id.as_deref());

	info!(key = api_key_id.as_deref().unwrap_or("-"), "Chat completion request");

	let conv_log = state.conversation_log.clone();
	let request_id = request_id.to_string();
	if request.stream {
		crate::routes::completions_v2::handle_streaming_v2(
			state.clone(),
			request,
			model_def,
			request_id,
			chat_id,
			created,
			session_id,
			conv_log,
		)
		.await
	} else {
		crate::routes::completions_v2::handle_non_streaming_v2(
			state.clone(),
			request,
			model_def,
			request_id,
			chat_id,
			created,
			session_id,
			conv_log,
		)
		.await
	}

	}
	.instrument(span)
	.await
}
