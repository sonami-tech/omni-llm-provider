use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;

use crate::AppState;

/// `GET /v1/models`. Serves both SDKs on one path: an Anthropic-shaped catalog
/// when the request carries an `anthropic-version` header (the Anthropic SDK
/// always sends it), otherwise the OpenAI-shaped list. This lets both
/// `openai.models.list()` and `anthropic.models.list()` work unmodified.
pub async fn models_handler(
	State(state): State<Arc<AppState>>,
	headers: HeaderMap,
) -> impl IntoResponse {
	if headers.contains_key("anthropic-version") {
		return Json(crate::translate::anthropic_passthrough::anthropic_models_body(
			state.fingerprint_profile,
		));
	}
	Json(serde_json::json!({
		"object": "list",
		"data": state.fingerprint_profile.models_list(),
	}))
}
