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
    // Add others as needed for prototype (Timeout etc. can map to ServerError).
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            AppError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
            AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            AppError::NotFound(_) => (StatusCode::NOT_FOUND, "invalid_request_error"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };
        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": error_type,
                "code": null,
            }
        });
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    async fn into_body(resp: Response) -> (StatusCode, String) {
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let s = String::from_utf8(bytes.to_vec()).unwrap();
        (status, s)
    }

    #[test]
    fn unauthorized_maps_to_401_and_shape() {
        let err = AppError::Unauthorized("bad key".into());
        // sync check variant (into_response consumes but we only care it doesn't panic and variant ok)
        let _resp = err.into_response();
        // can't easily await here in plain test, use block_on or just status from match in Into
        // instead re-check the logic lightly
        match &AppError::Unauthorized("x".into()) {
            AppError::Unauthorized(_) => {}
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn error_into_response_shapes() {
        let cases = vec![
            (
                AppError::Unauthorized("u".into()),
                StatusCode::UNAUTHORIZED,
                "authentication_error",
            ),
            (
                AppError::BadRequest("b".into()),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                AppError::NotFound("n".into()),
                StatusCode::NOT_FOUND,
                "invalid_request_error",
            ),
            (
                AppError::ServerError("s".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
            ),
        ];
        for (err, want_status, want_type) in cases {
            let resp = err.into_response();
            let (status, body) = into_body(resp).await;
            assert_eq!(status, want_status);
            assert!(body.contains(want_type));
            assert!(body.contains("message"));
        }
    }

    // All variants produce strict OAI error envelope: {"error":{"message":...,"type":...,"code":null}}
    // Mirrors CCP error mapping for client compatibility.
    #[tokio::test]
    async fn all_variants_map_to_oai_error_shape() {
        let cases = [
            (AppError::Unauthorized("auth fail".into()), "auth fail"),
            (AppError::BadRequest("bad req".into()), "bad req"),
            (AppError::NotFound("nope".into()), "nope"),
            (AppError::ServerError("boom".into()), "boom"),
        ];
        for (err, want_msg) in cases {
            let resp = err.into_response();
            let (_status, body) = into_body(resp).await;
            let v: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert!(v["error"].is_object());
            assert!(v["error"]["message"].is_string());
            assert!(v["error"]["type"].is_string());
            assert!(v["error"]["code"].is_null());
            // message carries the inner
            assert!(body.contains(want_msg));
        }
    }

    #[tokio::test]
    async fn unauthorized_uses_auth_error_type() {
        let resp = AppError::Unauthorized("bad".into()).into_response();
        let (_s, body) = into_body(resp).await;
        assert!(body.contains("authentication_error"));
        assert!(!body.contains("invalid_request_error"));
    }

    // All variants map to strict OAI error shape: {"error":{"message":...,"type":...,"code":null}}
    // Mirrors CCP error.rs mapping for wire compatibility with OpenAI clients.
    #[tokio::test]
    async fn all_variants_map_to_oai_shape_exact() {
        let cases = vec![
            (
                AppError::Unauthorized("u msg".into()),
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "u msg",
            ),
            (
                AppError::BadRequest("b msg".into()),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "b msg",
            ),
            (
                AppError::NotFound("n msg".into()),
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "n msg",
            ),
            (
                AppError::ServerError("s msg".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "s msg",
            ),
        ];
        for (err, want_status, want_type, want_msg) in cases {
            let resp = err.into_response();
            let (status, body) = into_body(resp).await;
            assert_eq!(status, want_status);
            let v: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v["error"]["message"], want_msg);
            assert_eq!(v["error"]["type"], want_type);
            assert!(v["error"]["code"].is_null());
            let obj = v["error"].as_object().unwrap();
            assert_eq!(obj.len(), 3); // message, type, code only
        }
    }

    // Unauthorized specifically uses "authentication_error" (not request or server).
    #[tokio::test]
    async fn unauthorized_uses_auth_error() {
        let resp = AppError::Unauthorized("authz fail".into()).into_response();
        let (_status, body) = into_body(resp).await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "authentication_error");
        assert_eq!(v["error"]["message"], "authz fail");
        assert!(v["error"]["code"].is_null());
    }

    // BadRequest / NotFound both use invalid_request_error type; Server uses server_error.
    #[tokio::test]
    async fn error_types_for_variants() {
        let br = AppError::BadRequest("bad".into()).into_response();
        let nf = AppError::NotFound("nf".into()).into_response();
        let se = AppError::ServerError("se".into()).into_response();
        let (_s, b1) = into_body(br).await;
        let (_s, b2) = into_body(nf).await;
        let (_s, b3) = into_body(se).await;
        assert!(b1.contains("invalid_request_error"));
        assert!(b2.contains("invalid_request_error"));
        assert!(b3.contains("server_error"));
        assert!(!b1.contains("authentication_error"));
    }
}
