use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use axum::response::IntoResponse;

use crate::AppState;

pub async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
	Json(serde_json::json!({
		"object": "list",
		"data": state.fingerprint_profile.models_list(),
	}))
}
