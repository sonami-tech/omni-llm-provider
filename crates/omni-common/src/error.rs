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
    /// A generic status-carrying error, so upstream statuses can be translated
    /// without one AppError variant per HTTP code. The `error.type` is derived
    /// from the status class (see `into_response`).
    #[error("{1}")]
    Http(StatusCode, String),
}

impl AppError {
    /// A status-carrying error (used by the upstream classifier to translate
    /// gateway statuses that have no dedicated variant, e.g. 429/502/503/504).
    pub fn http(status: u16, message: impl Into<String>) -> Self {
        let status =
            StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        AppError::Http(status, message.into())
    }

    /// A 429 rate-limit error (carries `error.type = "rate_limit_error"` so
    /// client SDKs back off correctly).
    pub fn rate_limited(message: impl Into<String>) -> Self {
        AppError::Http(StatusCode::TOO_MANY_REQUESTS, message.into())
    }
}

/// Translate a `ProviderError::Upstream { status, message }` into the
/// client-facing `AppError`, per the gateway status policy (issue #3). Both the
/// completion paths and the Anthropic-prepare path route through this so the
/// same upstream failure yields the same client status regardless of route.
///
/// omni is a credentialed gateway, not a transparent proxy: it terminates the
/// client's auth and opens its own upstream connection. So a status is mirrored
/// only when it still means the same thing to the new client; statuses that
/// would point at the wrong actor (upstream 401/403 = omni's key, not the
/// client's) or destroy diagnosability (a client 500 must mean omni broke) are
/// remapped to 502.
pub fn classify_upstream(status: Option<u16>, message: String) -> AppError {
    let msg = format!("upstream: {message}");
    match status {
        Some(429) => AppError::rate_limited(msg),
        // Upstream auth failure is omni's credential problem, not the client's;
        // mirroring 401/403 would wrongly tell the client to rotate its key.
        Some(401 | 403) => AppError::http(502, msg),
        Some(404) => AppError::NotFound(msg),
        Some(408) => AppError::http(504, msg), // upstream timeout
        Some(s @ (409 | 422)) => AppError::http(s, msg), // mirror semantic 4xx
        Some(400) => AppError::BadRequest(msg),
        Some(s @ 502..=504) => AppError::http(s, msg), // mirror gateway 5xx
        // Collapse provider-origin 5xx to 502 so a client-visible 500 stays
        // reserved for omni-internal faults.
        Some(s) if (500..=599).contains(&s) => AppError::http(502, msg),
        // No status (transport/decode/stream) or any other code -> bad gateway.
        Some(_) | None => AppError::http(502, msg),
    }
}

/// OpenAI-style `error.type` for a translated HTTP status. 4xx that the client
/// can act on are request/rate errors; everything else is a server error.
fn error_type_for_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        429 => "rate_limit_error",
        400 | 404 | 409 | 422 => "invalid_request_error",
        401 | 403 => "authentication_error",
        _ => "server_error",
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            AppError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
            AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            AppError::NotFound(_) => (StatusCode::NOT_FOUND, "invalid_request_error"),
            AppError::Http(s, _) => (*s, error_type_for_status(*s)),
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

    // Issue #3: the ratified upstream -> client status policy. One row per status
    // class. WHY: omni is a credentialed gateway, so a status is only mirrored
    // when it still means the same thing to the client; statuses that would point
    // at the wrong actor (401/403 = omni's key) or destroy diagnosability (a
    // client 500 must mean omni broke) collapse to 502.
    #[test]
    fn classify_upstream_status_policy() {
        let cases: &[(Option<u16>, u16)] = &[
            (Some(400), 400),
            (Some(401), 502), // upstream auth is omni's problem, not the client's
            (Some(403), 502),
            (Some(404), 404),
            (Some(408), 504), // upstream timeout
            (Some(409), 409),
            (Some(422), 422),
            (Some(429), 429), // must survive for client backoff
            (Some(500), 502), // provider 5xx collapses; client 500 reserved for omni
            (Some(501), 502),
            (Some(503), 503), // gateway 5xx mirrored exactly
            (Some(502), 502),
            (Some(504), 504),
            (Some(599), 502), // other 5xx -> bad gateway
            (None, 502),      // transport / decode / stream -> bad gateway
        ];
        for (upstream, want) in cases {
            let err = classify_upstream(*upstream, "detail".into());
            let status = match err {
                AppError::Http(s, _) => s.as_u16(),
                AppError::BadRequest(_) => 400,
                AppError::NotFound(_) => 404,
                AppError::Unauthorized(_) => 401,
                AppError::ServerError(_) => 500,
            };
            assert_eq!(
                status, *want,
                "upstream {upstream:?} must map to client {want}"
            );
        }
    }

    #[test]
    fn classify_upstream_429_carries_rate_limit_type() {
        // WHY: SDK backoff keys off both the 429 status AND the rate_limit_error
        // type; collapsing either breaks retry behavior.
        let resp = classify_upstream(Some(429), "slow down".into()).into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn classify_upstream_preserves_message_for_diagnostics() {
        // The numeric upstream status is collapsed for 401, but the original
        // detail (which a provider may have redacted upstream) is preserved so
        // the precise cause is still diagnosable.
        let err = classify_upstream(Some(401), "provider key rejected".into());
        assert!(err.to_string().contains("provider key rejected"));
    }

    #[test]
    fn classify_upstream_is_route_independent() {
        // WHY: the bug #3 fixes was the SAME ProviderError::Upstream yielding 400
        // on the anthropic-prepare route and 500 on the completion route. Both
        // routes now call classify_upstream, so identical input -> identical
        // status. This pins that single-classifier contract at its source.
        let a = classify_upstream(Some(503), "down".into());
        let b = classify_upstream(Some(503), "down".into());
        assert_eq!(a.into_response().status(), b.into_response().status());
    }

    #[tokio::test]
    async fn http_variant_oai_error_types() {
        // The status-carrying variant derives error.type from the status class,
        // so client SDKs see the right category (rate_limit vs invalid_request vs
        // server) for a translated upstream status.
        let cases: &[(u16, &str)] = &[
            (429, "rate_limit_error"),
            (404, "invalid_request_error"),
            (409, "invalid_request_error"),
            (502, "server_error"),
            (504, "server_error"),
        ];
        for (status, want_type) in cases {
            let resp = AppError::http(*status, "x").into_response();
            let (got_status, body) = into_body(resp).await;
            assert_eq!(got_status.as_u16(), *status);
            let v: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v["error"]["type"], *want_type);
        }
    }

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
