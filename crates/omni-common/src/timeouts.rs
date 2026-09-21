//! Outbound HTTP timeouts shared by every provider.
//!
//! Two families, kept as separate constants:
//! - **upstream**: inference POST (chat / messages / responses), including the
//!   streamed body. `UPSTREAM_HEADERS_TIMEOUT` bounds only the pre-body attempt.
//! - **oauth**: token-endpoint POSTs only (Claude / Grok / Codex refresh).
//!
//! Provider crates must use these names. Do not restate the seconds at the
//! `Client::builder` site. Fingerprint wire values such as Claude's
//! `x-stainless-timeout: 600` are not ours; they stay in the fingerprint
//! module.

use std::time::Duration;

/// Total time for one upstream inference request, including streamed body.
///
/// Matches xAI's documented reasoning-stream timeout (3600s) and grok-shell's
/// live `inference_idle_timeout_secs`. This is a total cap, not an idle gap.
pub const UPSTREAM_REQUEST_TIMEOUT: Duration = Duration::from_secs(3600);

/// TCP/TLS connect budget for an upstream inference request.
pub const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// Budget for one pre-body attempt: stream POST, its 401-retry POST, Claude
/// stream-open POST, Codex models preflight, or a WebSocket connect plus the
/// `response.create` send. Not time-to-first-token and not a generation cap.
pub const UPSTREAM_HEADERS_TIMEOUT: Duration = Duration::from_secs(60);

/// Stop reading a non-success upstream body after this many bytes.
pub const UPSTREAM_ERROR_BODY_PREFIX: usize = 8 * 1024;

/// Give up a non-success body read after this long and keep the HTTP status.
pub const UPSTREAM_ERROR_BODY_READ_TIMEOUT: Duration = Duration::from_secs(1);

/// Total time for an OAuth token refresh POST. Not an inference timeout.
pub const OAUTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Bound one pre-body attempt. Does not map the timeout to an HTTP status.
pub async fn timeout_upstream_headers<T>(
    fut: impl std::future::Future<Output = T>,
) -> Result<T, tokio::time::error::Elapsed> {
    tokio::time::timeout(UPSTREAM_HEADERS_TIMEOUT, fut).await
}

/// Map a headers-wait result. `Elapsed` is HTTP 504. The inner error is the
/// caller's: a reqwest timeout stays whatever `map_err` returns and is not
/// rewritten to 504.
pub fn map_upstream_headers_wait<T, E>(
    waited: Result<Result<T, E>, tokio::time::error::Elapsed>,
    map_err: impl FnOnce(E) -> omni_core::ProviderError,
) -> Result<T, omni_core::ProviderError> {
    match waited {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(map_err(err)),
        Err(_) => Err(omni_core::ProviderError::upstream_status(
            504,
            "upstream headers timeout",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_request_timeout_is_3600s() {
        assert_eq!(UPSTREAM_REQUEST_TIMEOUT, Duration::from_secs(3600));
    }

    #[test]
    fn families_stay_distinct() {
        assert!(UPSTREAM_REQUEST_TIMEOUT > UPSTREAM_CONNECT_TIMEOUT);
        assert!(UPSTREAM_REQUEST_TIMEOUT > OAUTH_REQUEST_TIMEOUT);
        assert!(UPSTREAM_HEADERS_TIMEOUT > UPSTREAM_CONNECT_TIMEOUT);
        assert!(UPSTREAM_REQUEST_TIMEOUT > UPSTREAM_HEADERS_TIMEOUT);
        assert_eq!(UPSTREAM_CONNECT_TIMEOUT, Duration::from_secs(15));
        assert_eq!(UPSTREAM_HEADERS_TIMEOUT, Duration::from_secs(60));
        assert_eq!(OAUTH_REQUEST_TIMEOUT, Duration::from_secs(30));
        assert_eq!(UPSTREAM_ERROR_BODY_PREFIX, 8 * 1024);
        assert_eq!(UPSTREAM_ERROR_BODY_READ_TIMEOUT, Duration::from_secs(1));
    }

    #[tokio::test]
    async fn headers_wait_elapsed_maps_to_504_inner_error_does_not() {
        let elapsed = tokio::time::timeout(Duration::from_millis(1), std::future::pending::<()>())
            .await
            .expect_err("pending future must elapse");
        let waited: Result<Result<(), &str>, _> = Err(elapsed);
        let err = super::map_upstream_headers_wait(waited, |_| {
            panic!("transport mapper must not run on Elapsed")
        })
        .expect_err("elapsed is an error");
        match err {
            omni_core::ProviderError::Upstream {
                status: Some(504),
                message,
            } => assert!(message.contains("upstream headers timeout"), "{message}"),
            other => panic!("expected 504, got {other:?}"),
        }

        let inner = super::map_upstream_headers_wait(
            Ok(Err::<(), &str>("connect timeout")),
            omni_core::ProviderError::upstream,
        )
        .expect_err("inner error");
        match inner {
            omni_core::ProviderError::Upstream {
                status: None,
                message,
            } => assert!(message.contains("connect timeout"), "{message}"),
            other => panic!("inner error must stay status-less, got {other:?}"),
        }
    }
}
