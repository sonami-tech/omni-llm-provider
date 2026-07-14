//! Read OAuth subscription credentials from `~/.claude/.credentials.json`.
//!
//! Locked design: never cache. Always re-read per request so CLI or Omni write-back
//! is picked up on the next call.
//!
//! OAuth refresh is **on by default**: Omni may proactively refresh a near-expired
//! or expired OAuth access token via the Claude token endpoint and **atomically
//! write back** the rotated tokens to the same credentials path.
//! Disable/enable globally or per provider via `OMNI_OAUTH_REFRESH`,
//! `OMNI_NO_OAUTH_REFRESH`, `OMNI_CLAUDE_OAUTH_REFRESH`, or matching omni CLI
//! flags (see `omni_common::oauth_refresh`).
//!
//! Ported from reference-src-claude/upstream/credentials.rs .
//! All credential handling for the OAuth gate stays isolated here.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{info, warn};

use crate::upstream::errors::UpstreamError;

/// Captured Claude OAuth token endpoint (claude-cli 2.1.205).
pub const CLAUDE_OAUTH_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
/// Fixed public client_id from live capture.
pub const CLAUDE_OAUTH_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
/// Scope string from live capture (order matters for wire parity).
pub const CLAUDE_OAUTH_SCOPE: &str =
    "user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
/// User-Agent from live token-grant capture.
pub const CLAUDE_OAUTH_USER_AGENT: &str = "axios/1.15.2";

/// Refresh when access token expires within this many milliseconds (5 minutes).
const NEAR_EXPIRY_SKEW_MS: i64 = 5 * 60 * 1000;

/// On-disk shape of `~/.claude/.credentials.json` (subset used for load).
#[derive(Debug, Clone, Deserialize)]
struct CredentialsFile {
    #[serde(rename = "claudeAiOauth")]
    claude_ai_oauth: ClaudeAiOauth,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaudeAiOauth {
    #[serde(rename = "accessToken")]
    access_token: String,
    #[serde(rename = "refreshToken")]
    refresh_token: Option<String>,
    #[serde(rename = "expiresAt")]
    expires_at: Option<i64>,
    #[serde(rename = "subscriptionType")]
    subscription_type: Option<String>,
}

/// Parsed OAuth grant response (token endpoint).
#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeTokenGrant {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: Option<i64>,
    pub refresh_token_expires_in: Option<i64>,
}

/// In-memory parsed credentials.
#[derive(Debug, Clone)]
pub struct Credentials {
    pub access_token: String,
    /// Unix epoch milliseconds. None if absent in file.
    pub expires_at_ms: Option<i64>,
    pub subscription_type: Option<String>,
}

impl Credentials {
    pub(crate) fn placeholder_for_custom_gateway() -> Self {
        Self {
            access_token: "custom-gateway-placeholder".into(),
            expires_at_ms: None,
            subscription_type: None,
        }
    }

    /// Default location: `~/.claude/.credentials.json`. Override via $CLAUDE_CREDENTIALS_PATH.
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("CLAUDE_CREDENTIALS_PATH") {
            return PathBuf::from(p);
        }
        match dirs::home_dir() {
            Some(home) => home.join(".claude").join(".credentials.json"),
            None => PathBuf::from(".claude/.credentials.json"),
        }
    }

    /// Read and parse the credentials file synchronously. Never cached. Use this
    /// only outside the async request path (e.g. startup validation); request
    /// handlers should use [`Credentials::load_fresh_async`] to avoid blocking a
    /// Tokio worker on the file read.
    pub fn load_fresh(path: &Path) -> Result<Self, UpstreamError> {
        let bytes = std::fs::read(path).map_err(UpstreamError::CredentialsRead)?;
        Self::from_bytes(&bytes)
    }

    /// Read and parse the credentials file without blocking the async executor.
    /// When OAuth refresh is enabled and the access token is expired or near
    /// expiry, attempts an in-place refresh before returning.
    pub async fn load_fresh_async(path: &Path) -> Result<Self, UpstreamError> {
        Self::load_fresh_async_with_refresh(path, RefreshTrigger::ProactiveNearExpiry).await
    }

    /// Force an OAuth refresh (if enabled and a refresh token is present), then
    /// re-read. Used on the 401-once retry path so a server-rejected access token
    /// can be rotated even when wall-clock expiry still looks valid.
    pub async fn load_fresh_async_force_refresh(path: &Path) -> Result<Self, UpstreamError> {
        Self::load_fresh_async_with_refresh(path, RefreshTrigger::Force).await
    }

    async fn load_fresh_async_with_refresh(
        path: &Path,
        trigger: RefreshTrigger,
    ) -> Result<Self, UpstreamError> {
        let bytes = tokio::fs::read(path)
            .await
            .map_err(UpstreamError::CredentialsRead)?;
        let creds = Self::from_bytes(&bytes)?;

        if !oauth_refresh_enabled() {
            return Ok(creds);
        }

        let should = match trigger {
            RefreshTrigger::ProactiveNearExpiry => needs_refresh(creds.expires_at_ms),
            RefreshTrigger::Force => true,
        };
        if !should {
            return Ok(creds);
        }

        match refresh_oauth_inplace(path, &oauth_token_url()).await {
            Ok(()) => {
                let bytes = tokio::fs::read(path)
                    .await
                    .map_err(UpstreamError::CredentialsRead)?;
                Self::from_bytes(&bytes)
            }
            Err(e) => {
                // If the access token is still usable, prefer it over hard-failing the request.
                if creds.check_expired().is_ok() {
                    warn!(error = %e, "claude OAuth refresh failed; using existing access token");
                    Ok(creds)
                } else {
                    Err(e)
                }
            }
        }
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, UpstreamError> {
        let parsed: CredentialsFile =
            serde_json::from_slice(bytes).map_err(UpstreamError::CredentialsParse)?;
        if parsed.claude_ai_oauth.access_token.is_empty() {
            return Err(UpstreamError::CredentialsMissingToken);
        }
        Ok(Credentials {
            access_token: parsed.claude_ai_oauth.access_token,
            expires_at_ms: parsed.claude_ai_oauth.expires_at,
            subscription_type: parsed.claude_ai_oauth.subscription_type,
        })
    }

    /// Surface a clear error if the token is past its expiry. When OAuth refresh
    /// is enabled, callers should attempt refresh before treating this as terminal.
    pub fn check_expired(&self) -> Result<(), UpstreamError> {
        if let Some(exp) = self.expires_at_ms {
            let now_ms = chrono::Utc::now().timestamp_millis();
            if now_ms >= exp {
                return Err(UpstreamError::TokenExpired);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum RefreshTrigger {
    ProactiveNearExpiry,
    Force,
}

/// Gate for in-Omni Claude OAuth refresh.
///
/// Default **on**. See [`omni_common::oauth_refresh_enabled_for`] for global and
/// per-provider env / CLI controls.
pub fn oauth_refresh_enabled() -> bool {
    omni_common::oauth_refresh_enabled_for(omni_common::OAuthRefreshProvider::Claude)
}

/// Effective Claude token endpoint. Production uses the captured host; tests may
/// set `OMNI_CLAUDE_OAUTH_TOKEN_URL` to a wiremock URL.
pub fn oauth_token_url() -> String {
    std::env::var("OMNI_CLAUDE_OAUTH_TOKEN_URL")
        .unwrap_or_else(|_| CLAUDE_OAUTH_TOKEN_URL.to_string())
}

/// True when access-token expiry is missing-as-ok, or within the near-expiry skew.
pub fn needs_refresh(expires_at_ms: Option<i64>) -> bool {
    match expires_at_ms {
        None => false,
        Some(exp) => {
            let now_ms = chrono::Utc::now().timestamp_millis();
            now_ms + NEAR_EXPIRY_SKEW_MS >= exp
        }
    }
}

/// Build the JSON body for the Claude refresh_token grant (pure; hermetic tests assert this).
pub fn build_refresh_request_body(refresh_token: &str) -> Value {
    json!({
        "client_id": CLAUDE_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": CLAUDE_OAUTH_SCOPE,
    })
}

/// Apply a successful grant onto the on-disk credentials JSON (pure).
///
/// Updates `claudeAiOauth.accessToken`, rotates `refreshToken` when the grant
/// returns one, and rewrites absolute ms expiry fields. Preserves sibling fields
/// (scopes, subscriptionType, rateLimitTier, etc.).
pub fn apply_grant_to_credentials_json(
    file: &mut Value,
    grant: &ClaudeTokenGrant,
    now_ms: i64,
) -> Result<(), UpstreamError> {
    let oauth = file
        .get_mut("claudeAiOauth")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| {
            UpstreamError::Decode("credentials.json missing claudeAiOauth object".into())
        })?;

    oauth.insert(
        "accessToken".into(),
        Value::String(grant.access_token.clone()),
    );

    if let Some(rt) = grant.refresh_token.as_ref().filter(|s| !s.is_empty()) {
        oauth.insert("refreshToken".into(), Value::String(rt.clone()));
    }

    if let Some(expires_in) = grant.expires_in {
        let expires_at = now_ms.saturating_add(expires_in.saturating_mul(1000));
        oauth.insert("expiresAt".into(), json!(expires_at));
    }

    if let Some(rt_expires_in) = grant.refresh_token_expires_in {
        let rt_expires_at = now_ms.saturating_add(rt_expires_in.saturating_mul(1000));
        oauth.insert("refreshTokenExpiresAt".into(), json!(rt_expires_at));
    }

    Ok(())
}

/// Atomically replace `path` with `bytes` via temp file + rename.
/// Preserves the prior file mode when possible so RTs stay non-world-readable.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), UpstreamError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(
        ".{}.omni-oauth-{}-{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("credentials.json"),
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&tmp, bytes).map_err(UpstreamError::CredentialsRead)?;
    if let Ok(meta) = std::fs::metadata(path) {
        let _ = std::fs::set_permissions(&tmp, meta.permissions());
    } else {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
        }
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(UpstreamError::CredentialsRead(e));
    }
    Ok(())
}

/// Extract refresh_token from raw credentials bytes.
fn refresh_token_from_bytes(bytes: &[u8]) -> Result<String, UpstreamError> {
    let parsed: CredentialsFile =
        serde_json::from_slice(bytes).map_err(UpstreamError::CredentialsParse)?;
    parsed
        .claude_ai_oauth
        .refresh_token
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            UpstreamError::Decode(
                "credentials.json has no refreshToken; cannot OAuth-refresh (re-login with claude CLI)"
                    .into(),
            )
        })
}

/// POST the refresh grant and atomically write back to `path`.
///
/// `token_url` is injectable so hermetic tests can point at wiremock.
pub async fn refresh_oauth_inplace(path: &Path, token_url: &str) -> Result<(), UpstreamError> {
    info!(path = %path.display(), "claude OAuth refresh starting");
    let bytes = tokio::fs::read(path)
        .await
        .map_err(UpstreamError::CredentialsRead)?;
    let refresh_token = refresh_token_from_bytes(&bytes)?;
    let body = build_refresh_request_body(&refresh_token);

    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(UpstreamError::Transport)?;

    let resp = client
        .post(token_url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::USER_AGENT, CLAUDE_OAUTH_USER_AGENT)
        .json(&body)
        .send()
        .await
        .map_err(UpstreamError::Transport)?;

    let status = resp.status();
    let resp_bytes = resp.bytes().await.map_err(UpstreamError::Transport)?;
    if !status.is_success() {
        let body_txt = String::from_utf8_lossy(&resp_bytes);
        return Err(UpstreamError::Decode(format!(
            "Claude OAuth token refresh failed HTTP {status}: {body_txt}"
        )));
    }

    let grant: ClaudeTokenGrant = serde_json::from_slice(&resp_bytes)
        .map_err(|e| UpstreamError::Decode(format!("Claude OAuth token response parse: {e}")))?;
    if grant.access_token.is_empty() {
        return Err(UpstreamError::Decode(
            "Claude OAuth token response missing access_token".into(),
        ));
    }
    let new_rt = grant
        .refresh_token
        .as_ref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            UpstreamError::Decode(
                "Claude OAuth token response missing refresh_token; refusing write-back that would strand RT"
                    .into(),
            )
        })?;

    // CAS against concurrent writers: re-read and refuse to clobber a different RT.
    let latest_bytes = tokio::fs::read(path)
        .await
        .map_err(UpstreamError::CredentialsRead)?;
    let mut latest: Value =
        serde_json::from_slice(&latest_bytes).map_err(UpstreamError::CredentialsParse)?;
    let disk_rt = latest
        .pointer("/claudeAiOauth/refreshToken")
        .and_then(|v| v.as_str());
    match disk_rt {
        Some(rt) if rt == refresh_token || rt == new_rt => {}
        Some(_) => {
            return Err(UpstreamError::Decode(
                "credentials.json refreshToken changed during OAuth refresh (concurrent writer); not clobbering"
                    .into(),
            ));
        }
        None => {}
    }

    let now_ms = chrono::Utc::now().timestamp_millis();
    apply_grant_to_credentials_json(&mut latest, &grant, now_ms)?;
    let out = serde_json::to_vec_pretty(&latest)
        .map_err(|e| UpstreamError::Decode(format!("serialize refreshed credentials: {e}")))?;
    atomic_write(path, &out)?;
    info!(path = %path.display(), "claude OAuth refresh ok");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    fn write_temp_creds(content: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "ccp_creds_test_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn load_fresh_parses_valid_oauth_file() {
        let path = write_temp_creds(
            r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-abc123","expiresAt":9999999999999,"subscriptionType":"max"}}"#,
        );
        let c = Credentials::load_fresh(&path).unwrap();
        assert!(c.access_token.starts_with("sk-ant-oat01-"));
        assert_eq!(c.subscription_type.as_deref(), Some("max"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_fresh_rejects_missing_token() {
        let path = write_temp_creds(r#"{"claudeAiOauth":{"accessToken":"","expiresAt":null}}"#);
        let res = Credentials::load_fresh(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(res, Err(UpstreamError::CredentialsMissingToken)));
    }

    #[test]
    fn load_fresh_rejects_malformed() {
        let path = write_temp_creds(r#"not json"#);
        let res = Credentials::load_fresh(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(res, Err(UpstreamError::CredentialsParse(_))));
    }

    #[test]
    fn check_expired_detects_future_vs_past() {
        let c = Credentials {
            access_token: "t".into(),
            expires_at_ms: Some(chrono::Utc::now().timestamp_millis() + 10_000),
            subscription_type: None,
        };
        assert!(c.check_expired().is_ok());

        let c2 = Credentials {
            access_token: "t".into(),
            expires_at_ms: Some(chrono::Utc::now().timestamp_millis() - 10_000),
            subscription_type: None,
        };
        assert!(matches!(
            c2.check_expired(),
            Err(UpstreamError::TokenExpired)
        ));
    }

    #[test]
    fn build_refresh_request_body_matches_capture() {
        // WHY: wire parity with claude-cli token grant; wrong client_id/scope burns or rejects.
        let body = build_refresh_request_body("rt-test-token");
        assert_eq!(body["client_id"], CLAUDE_OAUTH_CLIENT_ID);
        assert_eq!(body["grant_type"], "refresh_token");
        assert_eq!(body["refresh_token"], "rt-test-token");
        assert_eq!(body["scope"], CLAUDE_OAUTH_SCOPE);
    }

    #[test]
    fn apply_grant_rotates_refresh_token_and_expiry() {
        // WHY: isolation-copy theory test proved RT rotation must land on disk or main login dies.
        let mut file = json!({
            "claudeAiOauth": {
                "accessToken": "old-at",
                "refreshToken": "old-rt",
                "expiresAt": 1000,
                "refreshTokenExpiresAt": 2000,
                "subscriptionType": "max",
                "scopes": ["user:profile"]
            }
        });
        let grant = ClaudeTokenGrant {
            access_token: "new-at".into(),
            refresh_token: Some("new-rt".into()),
            expires_in: Some(28800),
            refresh_token_expires_in: Some(2_160_000),
        };
        let now_ms = 1_700_000_000_000i64;
        apply_grant_to_credentials_json(&mut file, &grant, now_ms).unwrap();
        let oauth = &file["claudeAiOauth"];
        assert_eq!(oauth["accessToken"], "new-at");
        assert_eq!(oauth["refreshToken"], "new-rt");
        assert_eq!(oauth["expiresAt"], now_ms + 28800 * 1000);
        assert_eq!(oauth["refreshTokenExpiresAt"], now_ms + 2_160_000 * 1000);
        assert_eq!(oauth["subscriptionType"], "max");
        assert_eq!(oauth["scopes"][0], "user:profile");
    }

    #[test]
    fn atomic_write_replaces_file_contents() {
        let path = write_temp_creds(r#"{"old":true}"#);
        atomic_write(&path, br#"{"new":true}"#).unwrap();
        let got = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(got.contains("\"new\""));
        assert!(!got.contains("old"));
    }

    #[test]
    fn needs_refresh_respects_skew() {
        let future = chrono::Utc::now().timestamp_millis() + 60 * 60 * 1000;
        assert!(!needs_refresh(Some(future)));
        let near = chrono::Utc::now().timestamp_millis() + 60 * 1000;
        assert!(needs_refresh(Some(near)));
        let past = chrono::Utc::now().timestamp_millis() - 1000;
        assert!(needs_refresh(Some(past)));
        assert!(!needs_refresh(None));
    }

    #[tokio::test]
    async fn refresh_oauth_inplace_posts_capture_shape_and_writes_rotated_rt() {
        // WHY: shipped refresh path must hit the real grant shape and persist rotated RT.
        use wiremock::matchers::{body_partial_json, header, method, path as url_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _guard = env_lock();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(url_path("/v1/oauth/token"))
            .and(header("content-type", "application/json"))
            .and(header("user-agent", CLAUDE_OAUTH_USER_AGENT))
            .and(body_partial_json(json!({
                "client_id": CLAUDE_OAUTH_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": "rt-old",
                "scope": CLAUDE_OAUTH_SCOPE,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "sk-ant-oat01-new",
                "refresh_token": "rt-new-rotated",
                "token_type": "Bearer",
                "expires_in": 28800,
                "refresh_token_expires_in": 2160000,
            })))
            .mount(&server)
            .await;

        let path = write_temp_creds(
            r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-old","refreshToken":"rt-old","expiresAt":1000,"subscriptionType":"max","scopes":["user:profile"]}}"#,
        );
        let token_url = format!("{}/v1/oauth/token", server.uri());
        refresh_oauth_inplace(&path, &token_url).await.unwrap();

        let on_disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(on_disk["claudeAiOauth"]["accessToken"], "sk-ant-oat01-new");
        assert_eq!(on_disk["claudeAiOauth"]["refreshToken"], "rt-new-rotated");
        assert!(
            on_disk["claudeAiOauth"]["expiresAt"].as_i64().unwrap()
                > chrono::Utc::now().timestamp_millis()
        );
        assert_eq!(on_disk["claudeAiOauth"]["subscriptionType"], "max");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn load_fresh_async_refreshes_when_flag_on_and_expired() {
        // WHY: request path must drive shipped load → grant → write-back → new access token.
        use wiremock::matchers::{method, path as url_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _guard = env_lock();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(url_path("/v1/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "sk-ant-oat01-after",
                "refresh_token": "rt-after",
                "expires_in": 28800,
                "refresh_token_expires_in": 2160000,
            })))
            .mount(&server)
            .await;

        let path = write_temp_creds(
            r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-before","refreshToken":"rt-before","expiresAt":1000,"subscriptionType":"max"}}"#,
        );
        let token_url = format!("{}/v1/oauth/token", server.uri());
        let old_flag = std::env::var_os("OMNI_OAUTH_REFRESH");
        let old_url = std::env::var_os("OMNI_CLAUDE_OAUTH_TOKEN_URL");
        unsafe {
            std::env::set_var("OMNI_OAUTH_REFRESH", "1");
            std::env::set_var("OMNI_CLAUDE_OAUTH_TOKEN_URL", &token_url);
        }
        let c = Credentials::load_fresh_async(&path).await.unwrap();
        unsafe {
            match old_flag {
                Some(v) => std::env::set_var("OMNI_OAUTH_REFRESH", v),
                None => std::env::remove_var("OMNI_OAUTH_REFRESH"),
            }
            match old_url {
                Some(v) => std::env::set_var("OMNI_CLAUDE_OAUTH_TOKEN_URL", v),
                None => std::env::remove_var("OMNI_CLAUDE_OAUTH_TOKEN_URL"),
            }
        }
        let on_disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(c.access_token, "sk-ant-oat01-after");
        assert_eq!(on_disk["claudeAiOauth"]["refreshToken"], "rt-after");
        assert!(c.check_expired().is_ok());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn load_fresh_async_does_not_refresh_when_flag_off() {
        let _guard = env_lock();
        let path = write_temp_creds(
            r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-stale","refreshToken":"rt-stale","expiresAt":1000,"subscriptionType":"max"}}"#,
        );
        let old = std::env::var_os("OMNI_OAUTH_REFRESH");
        unsafe {
            std::env::set_var("OMNI_OAUTH_REFRESH", "0");
        }
        let c = Credentials::load_fresh_async(&path).await.unwrap();
        unsafe {
            match old {
                Some(v) => std::env::set_var("OMNI_OAUTH_REFRESH", v),
                None => std::env::remove_var("OMNI_OAUTH_REFRESH"),
            }
        }
        let on_disk = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(c.access_token, "sk-ant-oat01-stale");
        assert!(on_disk.contains("rt-stale"));
        assert!(on_disk.contains("sk-ant-oat01-stale"));
    }
}
