use axum::Json;
use axum::response::IntoResponse;

pub async fn models_handler() -> impl IntoResponse {
	Json(serde_json::json!({
		"object": "list",
		"data": crate::models::models_list(),
	}))
}
