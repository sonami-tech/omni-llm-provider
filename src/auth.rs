use std::collections::HashSet;
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::error::AppError;

/// Identifies which API key was used for a request.
#[derive(Clone, Debug)]
pub struct ApiKeyId(pub String);

/// Auth middleware. If `valid_keys` is empty, all requests pass through.
pub async fn auth_layer(
	valid_keys: Arc<HashSet<String>>,
	mut req: Request<Body>,
	next: Next,
) -> Response {
	if valid_keys.is_empty() {
		return next.run(req).await;
	}

	let key = req
		.headers()
		.get("authorization")
		.and_then(|v| v.to_str().ok())
		.and_then(|v| v.strip_prefix("Bearer "))
		.map(str::trim);

	match key {
		Some(k) if valid_keys.contains(k) => {
			let id = key_id(k);
			req.extensions_mut().insert(ApiKeyId(id));
			next.run(req).await
		}
		Some(_) => AppError::Unauthorized("Invalid API key".into()).into_response(),
		None => AppError::Unauthorized(
			"Missing API key. Include 'Authorization: Bearer <key>' header.".into(),
		)
		.into_response(),
	}
}

/// Generate a short identifier for a key (first 4 + last 4, or just last 4 for short keys).
fn key_id(key: &str) -> String {
	if key.len() < 12 {
		let suffix_len = key.len().min(4);
		return format!("...{}", &key[key.len() - suffix_len..]);
	}
	format!("{}...{}", &key[..4], &key[key.len() - 4..])
}
