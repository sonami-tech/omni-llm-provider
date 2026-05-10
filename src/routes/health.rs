use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;

use crate::AppState;

pub async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
	Json(serde_json::json!({
		"status": "ok",
		"uptime_seconds": state.stats.uptime_secs(),
		"active_requests": state.stats.active_count(),
	}))
}
