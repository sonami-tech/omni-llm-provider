use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[derive(Debug, thiserror::Error)]
pub enum AppError {
	#[error("{0}")]
	Unauthorized(String),
	#[error("{0}")]
	BadRequest(String),
	#[error("{0}")]
	NotFound(String),
	#[error("{0}")]
	ServerError(String),
	#[error("{0}")]
	Timeout(String),
	#[error("{0}")]
	ServiceUnavailable(String),
	#[error("{0}")]
	BadGateway(String),
	#[error("{0}")]
	RateLimited(String),
}

impl IntoResponse for AppError {
	fn into_response(self) -> Response {
		let (status, error_type) = match &self {
			AppError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
			AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
			AppError::NotFound(_) => (StatusCode::NOT_FOUND, "invalid_request_error"),
			AppError::ServerError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
			AppError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, "server_error"),
			AppError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "server_error"),
			AppError::BadGateway(_) => (StatusCode::BAD_GATEWAY, "server_error"),
			AppError::RateLimited(_) => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
		};
		let body = serde_json::json!({
			"error": {
				"message": self.to_string(),
				"type": error_type,
				"code": null,
			}
		});
		let mut resp = (status, Json(body)).into_response();
		if matches!(self, AppError::ServiceUnavailable(_)) {
			resp.headers_mut()
				.insert("retry-after", "30".parse().unwrap());
		}
		resp
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use axum::body::to_bytes;

	async fn response_body(resp: Response) -> (StatusCode, serde_json::Value) {
		let status = resp.status();
		let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
		let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
		(status, body)
	}

	#[tokio::test]
	async fn bad_request_returns_400() {
		let err = AppError::BadRequest("Missing model".into());
		let (status, body) = response_body(err.into_response()).await;
		assert_eq!(status, StatusCode::BAD_REQUEST);
		assert_eq!(body["error"]["type"], "invalid_request_error");
		assert_eq!(body["error"]["message"], "Missing model");
		assert!(body["error"]["code"].is_null());
	}

	#[tokio::test]
	async fn not_found_returns_404() {
		let err = AppError::NotFound("No such endpoint".into());
		let (status, body) = response_body(err.into_response()).await;
		assert_eq!(status, StatusCode::NOT_FOUND);
		assert_eq!(body["error"]["type"], "invalid_request_error");
	}

	#[tokio::test]
	async fn server_error_returns_500() {
		let err = AppError::ServerError("Internal failure".into());
		let (status, body) = response_body(err.into_response()).await;
		assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
		assert_eq!(body["error"]["type"], "server_error");
	}

	#[tokio::test]
	async fn timeout_returns_504() {
		let err = AppError::Timeout("Inactivity timeout".into());
		let (status, body) = response_body(err.into_response()).await;
		assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
		assert_eq!(body["error"]["type"], "server_error");
	}

	#[tokio::test]
	async fn service_unavailable_returns_503_with_retry_after() {
		let err = AppError::ServiceUnavailable("Queue full".into());
		let resp = err.into_response();
		assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
		assert_eq!(resp.headers().get("retry-after").unwrap(), "30");
	}
}
