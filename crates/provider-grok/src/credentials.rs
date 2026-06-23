//! Grok / xAI credentials loading, modeled exactly after the Claude Code Provider technique.
//!
//! Locked design (same as CCP): never cache. Always re-read per request.
//! This picks up any background refresh or key rotation the user may have performed
//! (e.g. the Grok CLI refreshing its login, or an updated console key).
//!
//! ## Sources and precedence (highest first), all read fresh per request:
//! 1. `$XAI_CREDENTIALS_PATH` (explicit override) — parsed as either shape below.
//! 2. `~/.xai/.credentials.json` — a simple static-key file (deliberate console key).
//! 3. `~/.grok/auth.json` — the Grok CLI's own login file (OIDC). This is the primary
//!    auto-detect path: just as omni reads `~/.claude/.credentials.json` (the Claude
//!    CLI's file), it reads the Grok CLI's file so an existing `grok` login Just Works.
//!
//! Explicit beats ambient: a usable static-key file you created on purpose wins over
//! the CLI's auto-managed OIDC login. If the ambient static-key file exists but has
//! no usable key, we fall through to the CLI login instead of letting a stale file
//! break an otherwise valid `grok` login.
//!
//! ## On-disk shapes
//! Static key (`~/.xai/.credentials.json` or `$XAI_CREDENTIALS_PATH`):
//! ```json
//! { "apiKey": "xai-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" }
//! ```
//! (a top-level `xaiApiKey` alias is also tolerated.)
//!
//! Grok CLI OIDC (`~/.grok/auth.json`), keyed by `https://auth.x.ai::<client-id>`:
//! ```json
//! { "https://auth.x.ai::<id>": { "key": "<JWT>", "auth_mode": "oidc",
//!     "refresh_token": "...", "expires_at": "2026-06-10T22:20:22.000000Z" } }
//! ```
//! The `key` JWT is a Bearer that authenticates `api.x.ai/v1` directly. We read it
//! READ-ONLY: we never write this file or consume its single-use `refresh_token`;
//! on expiry we surface a clear error so the user re-authenticates via the Grok CLI.

use std::path::{Path, PathBuf};

use serde::Deserialize;

// Local error for credentials (kept small; mapped by callers to ProviderError or AppError).
#[derive(Debug, thiserror::Error)]
pub enum GrokCredentialsError {
    #[error("read: {0}")]
    Read(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(#[from] serde_json::Error),
    #[error(
        "no Grok credentials found (tried $XAI_CREDENTIALS_PATH, ~/.xai/.credentials.json, ~/.grok/auth.json)"
    )]
    NoSource,
    #[error("credentials file present but held no usable key")]
    MissingToken,
    #[error("token expired")]
    Expired,
}

/// On-disk shape of the simple static-key file.
#[derive(Debug, Clone, Deserialize)]
struct StaticKeyFile {
    #[serde(rename = "apiKey")]
    api_key: Option<String>,
    #[serde(rename = "xaiApiKey")]
    xai_api_key: Option<String>,
}

/// On-disk shape of one entry inside the Grok CLI's `~/.grok/auth.json`. The file is a
/// JSON object whose single value (keyed by `https://auth.x.ai::<client-id>`) has this shape.
#[derive(Debug, Clone, Deserialize)]
struct GrokCliEntry {
    /// The JWT access token used as the Bearer.
    key: Option<String>,
    /// "oidc" for the CLI's OIDC login.
    auth_mode: Option<String>,
    /// ISO 8601 expiry of the access token (e.g. "2026-06-10T22:20:22.000000Z").
    expires_at: Option<String>,
    /// The OIDC subject (a uuid) the Grok CLI sends as `x-grok-user-id` in
    /// conservative mode. Absent on static-key files (None there).
    user_id: Option<String>,
}

/// In-memory parsed Grok credentials.
#[derive(Debug, Clone)]
pub struct GrokCredentials {
    pub api_key: String,
    /// Access-token expiry in Unix epoch milliseconds. `None` for static API keys
    /// (which do not expire) or when the OIDC entry omits/!parses `expires_at`.
    pub expires_at_ms: Option<i64>,
    /// The OIDC `user_id` (a uuid) used as `x-grok-user-id` in conservative mode.
    /// `None` for static API keys (the CLI omits the header when unavailable).
    pub user_id: Option<String>,
}

impl GrokCredentials {
    /// The static-key default location: `~/.xai/.credentials.json`. Override via
    /// `$XAI_CREDENTIALS_PATH`. Mirrors the CCP `default_path` + env-override logic.
    pub fn default_path() -> PathBuf {
        if let Some(p) = std::env::var_os("XAI_CREDENTIALS_PATH") {
            return PathBuf::from(p);
        }
        match dirs::home_dir() {
            Some(home) => home.join(".xai").join(".credentials.json"),
            None => PathBuf::from(".xai/.credentials.json"),
        }
    }

    /// The Grok CLI's own login file: `~/.grok/auth.json`. This is the file the
    /// `grok` CLI writes on login; omni reads it so an existing login Just Works,
    /// exactly as it reads `~/.claude/.credentials.json` for Claude.
    pub fn grok_cli_path() -> Option<PathBuf> {
        dirs::home_dir().map(|home| home.join(".grok").join("auth.json"))
    }

    /// Resolve credentials fresh from disk (never cached). The operator provides creds in one of
    /// three places, checked in order:
    /// 1. `$XAI_CREDENTIALS_PATH` — an explicit file path the operator chose. If set, it is the
    ///    only source; a failure to load it surfaces directly.
    /// 2. `~/.xai/.credentials.json` — a static-key file.
    /// 3. `~/.grok/auth.json` — the Grok CLI's own login file (auto-detected).
    pub async fn load_resolved_async() -> Result<Self, GrokCredentialsError> {
        if let Some(p) = std::env::var_os("XAI_CREDENTIALS_PATH") {
            return Self::load_fresh_async(Path::new(&p)).await;
        }

        let mut ambient_static_error = None;
        if let Some(home) = dirs::home_dir() {
            let static_path = home.join(".xai").join(".credentials.json");
            if tokio::fs::try_exists(&static_path).await.unwrap_or(false) {
                match Self::load_fresh_async(&static_path).await {
                    Ok(creds) => return Ok(creds),
                    Err(GrokCredentialsError::MissingToken) => {
                        ambient_static_error = Some(GrokCredentialsError::MissingToken);
                    }
                    Err(err) => return Err(err),
                }
            }
        }

        if let Some(cli_path) = Self::grok_cli_path()
            && tokio::fs::try_exists(&cli_path).await.unwrap_or(false)
        {
            return Self::load_fresh_async(&cli_path).await;
        }

        Err(ambient_static_error.unwrap_or(GrokCredentialsError::NoSource))
    }

    /// Read and parse a specific credentials file synchronously. Never cached.
    /// Parses either supported shape (static-key or Grok CLI OIDC). Use outside the
    /// async hot path (e.g. startup); handlers should prefer [`load_resolved_async`].
    pub fn load_fresh(path: &Path) -> Result<Self, GrokCredentialsError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Async read+parse of a specific file, not blocking the Tokio worker. Never cached.
    pub async fn load_fresh_async(path: &Path) -> Result<Self, GrokCredentialsError> {
        let bytes = tokio::fs::read(path).await?;
        Self::from_bytes(&bytes)
    }

    /// Parse either credential shape from raw bytes: the Grok CLI OIDC file (an object whose entry
    /// carries `auth_mode:"oidc"` + a `key` JWT) or the simple static-key file (`{"apiKey": "..."}`).
    /// The static shape has no `auth_mode:"oidc"` entry, so the two are distinguishable.
    fn from_bytes(bytes: &[u8]) -> Result<Self, GrokCredentialsError> {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;

        if let Some(creds) = Self::try_grok_cli(&value) {
            return Ok(creds);
        }

        // Fall back to the static-key shape.
        let parsed: StaticKeyFile = serde_json::from_value(value)?;
        let key = parsed
            .api_key
            .or(parsed.xai_api_key)
            .filter(|k| !k.trim().is_empty())
            .ok_or(GrokCredentialsError::MissingToken)?;
        Ok(GrokCredentials {
            api_key: key,
            expires_at_ms: None,
            user_id: None,
        })
    }

    /// Recognize the Grok CLI's `~/.grok/auth.json` shape: a JSON object whose entry carries
    /// `auth_mode:"oidc"` and a non-empty `key` (the JWT). Reads the JWT and its expiry.
    fn try_grok_cli(value: &serde_json::Value) -> Option<GrokCredentials> {
        let obj = value.as_object()?;
        for entry_val in obj.values() {
            let entry: GrokCliEntry = match serde_json::from_value(entry_val.clone()) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let is_oidc = entry.auth_mode.as_deref() == Some("oidc");
            let key = entry.key.filter(|k| !k.trim().is_empty());
            if let (true, Some(key)) = (is_oidc, key) {
                return Some(GrokCredentials {
                    api_key: key,
                    expires_at_ms: entry.expires_at.as_deref().and_then(parse_iso8601_to_ms),
                    user_id: entry.user_id.filter(|id| !id.trim().is_empty()),
                });
            }
        }
        None
    }

    /// Surface a clear error if an OIDC access token is past its expiry. Static keys
    /// (no `expires_at`) are always OK. Callers should warn-but-continue (the upstream
    /// will 401 if the token is truly dead) and tell the user to re-run the Grok CLI
    /// login; we never refresh or rewrite the CLI's file ourselves.
    pub fn check_expired(&self) -> Result<(), GrokCredentialsError> {
        if let Some(exp) = self.expires_at_ms {
            let now_ms = chrono::Utc::now().timestamp_millis();
            if now_ms >= exp {
                return Err(GrokCredentialsError::Expired);
            }
        }
        Ok(())
    }
}

/// Parse an ISO 8601 / RFC 3339 timestamp (e.g. "2026-06-10T22:20:22.000000Z") to Unix
/// epoch milliseconds. Returns None on any parse failure so a malformed `expires_at`
/// degrades to "no known expiry" rather than erroring the whole credential load.
fn parse_iso8601_to_ms(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use crate::GROK_ENV_LOCK as ENV_LOCK;

    fn temp_credentials_path() -> PathBuf {
        std::env::temp_dir().join(format!("omni-creds-test-{}.json", uuid::Uuid::new_v4()))
    }

    fn write_temp_creds(path: &Path, content: &str) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
    }

    #[test]
    fn load_fresh_file_not_exist_returns_read_error() {
        let p = temp_credentials_path(); // ensure absent
        let _ = std::fs::remove_file(&p);
        let res = GrokCredentials::load_fresh(&p);
        assert!(res.is_err());
        match res.unwrap_err() {
            GrokCredentialsError::Read(_) => {}
            other => panic!("expected Read, got {other:?}"),
        }
    }

    #[test]
    fn load_fresh_bad_json_returns_parse_error() {
        let p = temp_credentials_path();
        write_temp_creds(&p, "{ not json }");
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(res.is_err());
        match res.unwrap_err() {
            GrokCredentialsError::Parse(_) => {}
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn load_fresh_missing_key_returns_missing_token() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"foo": "bar"}"#);
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(
            res.unwrap_err(),
            GrokCredentialsError::MissingToken
        ));
    }

    #[test]
    fn default_path_respects_env_override_xai_credentials_path() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let p = temp_credentials_path();
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        // SAFETY: test-only mutation of process env for override verification; serialized by ENV_LOCK.
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", p.to_str().unwrap());
        }
        let got = GrokCredentials::default_path();
        // restore
        unsafe {
            if let Some(v) = old {
                std::env::set_var("XAI_CREDENTIALS_PATH", v);
            } else {
                std::env::remove_var("XAI_CREDENTIALS_PATH");
            }
        }
        assert_eq!(got, p);
    }

    #[tokio::test]
    async fn load_fresh_async_works() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "xai-test-123"}"#);
        let c = GrokCredentials::load_fresh_async(&p).await.unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "xai-test-123");
    }

    #[test]
    fn check_expired_is_noop_for_api_keys() {
        let c = GrokCredentials {
            api_key: "xai-foo".into(),
            expires_at_ms: None,
            user_id: None,
        };
        assert!(c.check_expired().is_ok());
    }

    #[test]
    fn parse_accepts_xai_api_key_alias() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"xaiApiKey": "xai-alias-456"}"#);
        let c = GrokCredentials::load_fresh(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "xai-alias-456");
    }

    // Full coverage per spec: file not exist read err (already basic), ensure exact err variant + temp fixture.
    #[test]
    fn file_not_exist_read_err() {
        let p = temp_credentials_path();
        let _ = std::fs::remove_file(&p);
        let res = GrokCredentials::load_fresh(&p);
        assert!(res.is_err());
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::Read(_)));
    }

    // Bad json parse err using temp fixture.
    #[test]
    fn bad_json_parse_err() {
        let p = temp_credentials_path();
        write_temp_creds(&p, "this is not { json at all");
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::Parse(_)));
    }

    // Missing key (neither apiKey nor alias) -> MissingToken.
    #[test]
    fn missing_key_missing_token() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"other": "stuff"}"#);
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(
            res.unwrap_err(),
            GrokCredentialsError::MissingToken
        ));
    }

    // Env override XAI_CREDENTIALS_PATH (already tested; re-exercise with async load too).
    #[test]
    fn env_override_xai_credentials_path() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "env-override-key"}"#);
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", p.to_str().unwrap());
        }
        let got = GrokCredentials::default_path();
        let c = GrokCredentials::load_fresh(&got).unwrap();
        unsafe {
            if let Some(v) = old {
                std::env::set_var("XAI_CREDENTIALS_PATH", v);
            } else {
                std::env::remove_var("XAI_CREDENTIALS_PATH");
            }
        }
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "env-override-key");
    }

    // Async load + check_expired noop for api keys + parse alias already; add temp fixture async variant.
    #[tokio::test]
    async fn async_load_and_check_expired_noop() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "async-xai-999"}"#);
        let c = GrokCredentials::load_fresh_async(&p).await.unwrap();
        assert!(c.check_expired().is_ok());
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "async-xai-999");
    }

    // Additional: empty value for key is treated as missing (after trim).
    #[test]
    fn empty_key_value_is_missing_token() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "   "} "#);
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(
            res.unwrap_err(),
            GrokCredentialsError::MissingToken
        ));
    }

    // ---- Grok CLI OIDC file (~/.grok/auth.json) ----

    /// A minimal but realistic ~/.grok/auth.json: object keyed by the auth URL+client,
    /// value carries the JWT `key`, `auth_mode:"oidc"`, and an ISO 8601 `expires_at`.
    fn grok_cli_json(key: &str, expires_at: &str) -> String {
        format!(
            r#"{{ "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {{
                "key": "{key}",
                "auth_mode": "oidc",
                "refresh_token": "rt-single-use-rotating",
                "expires_at": "{expires_at}",
                "user_id": "11111111-2222-3333-4444-555555555555"
            }} }}"#
        )
    }

    #[test]
    fn parses_grok_cli_oidc_file_with_expiry() {
        // WHY: this is the Grok CLI's real on-disk shape. omni must read the JWT from it
        // (the whole point of auto-detecting an existing `grok` login) and capture the
        // expiry so check_expired can later warn. A future-dated token must be Ok.
        let p = temp_credentials_path();
        write_temp_creds(
            &p,
            &grok_cli_json("jwt-abc.def.ghi", "2999-01-01T00:00:00Z"),
        );
        let c = GrokCredentials::load_fresh(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "jwt-abc.def.ghi");
        assert!(c.expires_at_ms.is_some(), "OIDC expiry must be captured");
        assert!(c.check_expired().is_ok(), "future-dated token must be Ok");
        assert_eq!(
            c.user_id.as_deref(),
            Some("11111111-2222-3333-4444-555555555555"),
            "OIDC user_id must be captured for x-grok-user-id"
        );
    }

    #[test]
    fn oidc_captures_user_id_static_key_omits_it() {
        // WHY: conservative mode sends `x-grok-user-id: <user_id>`, sourced ONLY
        // from the OIDC file's `user_id`. A static-key file has no subject, so the
        // header (and thus user_id) must be None there - the CLI omits the header
        // when the id is unavailable, and we must mirror that exactly.
        let oidc = temp_credentials_path();
        write_temp_creds(&oidc, &grok_cli_json("jwt-uid", "2999-01-01T00:00:00Z"));
        let c = GrokCredentials::load_fresh(&oidc).unwrap();
        let _ = std::fs::remove_file(&oidc);
        assert_eq!(
            c.user_id.as_deref(),
            Some("11111111-2222-3333-4444-555555555555")
        );

        let static_p = temp_credentials_path();
        write_temp_creds(&static_p, r#"{"apiKey": "xai-static-no-uid"}"#);
        let s = GrokCredentials::load_fresh(&static_p).unwrap();
        let _ = std::fs::remove_file(&static_p);
        assert!(
            s.user_id.is_none(),
            "static-key file has no OIDC subject; user_id must be None"
        );
    }

    #[test]
    fn grok_cli_oidc_past_expiry_reports_expired() {
        // WHY: an expired OIDC token must surface as Err from check_expired so the provider
        // warns the user to re-run the Grok CLI login. The token value is still returned
        // (callers warn-but-continue and let the upstream make the final call).
        let p = temp_credentials_path();
        write_temp_creds(&p, &grok_cli_json("jwt-old", "2000-01-01T00:00:00Z"));
        let c = GrokCredentials::load_fresh(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "jwt-old");
        assert!(matches!(
            c.check_expired(),
            Err(GrokCredentialsError::Expired)
        ));
    }

    #[test]
    fn grok_cli_entry_without_oidc_mode_is_not_accepted() {
        // WHY: we only treat an entry marked auth_mode:"oidc" as the CLI login. An entry with a
        // different mode is not a credential we understand; since the file is also not a static
        // -key shape, the load yields MissingToken rather than silently using the stray key.
        let p = temp_credentials_path();
        write_temp_creds(
            &p,
            r#"{ "https://auth.x.ai::x": { "key": "jwt-x", "auth_mode": "password" } }"#,
        );
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(
            res.unwrap_err(),
            GrokCredentialsError::MissingToken
        ));
    }

    #[test]
    fn from_bytes_accepts_either_shape_via_explicit_path() {
        // WHY: $XAI_CREDENTIALS_PATH may point at EITHER file shape. Prove load_fresh parses
        // both, so the explicit override is not locked to one format.
        let static_p = temp_credentials_path();
        write_temp_creds(&static_p, r#"{"apiKey": "xai-static"}"#);
        let s = GrokCredentials::load_fresh(&static_p).unwrap();
        let _ = std::fs::remove_file(&static_p);
        assert_eq!(s.api_key, "xai-static");
        assert!(s.expires_at_ms.is_none());

        let oidc_p = temp_credentials_path();
        write_temp_creds(&oidc_p, &grok_cli_json("jwt-oidc", "2999-01-01T00:00:00Z"));
        let o = GrokCredentials::load_fresh(&oidc_p).unwrap();
        let _ = std::fs::remove_file(&oidc_p);
        assert_eq!(o.api_key, "jwt-oidc");
        assert!(o.expires_at_ms.is_some());
    }

    #[test]
    fn malformed_expires_at_degrades_to_no_expiry() {
        // WHY: a garbled expires_at must not error the whole load; it degrades to "no known
        // expiry" (the token is still usable; the upstream will 401 if truly dead).
        let p = temp_credentials_path();
        write_temp_creds(&p, &grok_cli_json("jwt-bad-exp", "not-a-timestamp"));
        let c = GrokCredentials::load_fresh(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "jwt-bad-exp");
        assert!(c.expires_at_ms.is_none());
        assert!(c.check_expired().is_ok());
    }

    // ---- precedence chain (load_resolved_async) ----

    #[tokio::test]
    // ENV_LOCK is held across the load_resolved_async().await because XAI_CREDENTIALS_PATH
    // must stay set while the loader reads it. Safe: #[tokio::test] uses a current-thread
    // runtime, so the task never migrates threads while the std Mutex guard is held.
    #[allow(clippy::await_holding_lock)]
    async fn resolved_honors_explicit_env_override_first() {
        // WHY: $XAI_CREDENTIALS_PATH is the highest-precedence source. When set, it is the only
        // source consulted, so an OIDC file there resolves even if it is the explicit path.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let p = temp_credentials_path();
        write_temp_creds(&p, &grok_cli_json("jwt-via-env", "2999-01-01T00:00:00Z"));
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", p.to_str().unwrap());
        }
        let res = GrokCredentials::load_resolved_async().await;
        unsafe {
            match old {
                Some(v) => std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
        }
        let _ = std::fs::remove_file(&p);
        let c = res.expect("env-pointed creds must resolve");
        assert_eq!(c.api_key, "jwt-via-env");
    }

    #[tokio::test]
    // See resolved_honors_explicit_env_override_first: env lock held across await is safe
    // on the current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn resolved_env_override_pointing_at_missing_file_errors_not_falls_through() {
        // WHY: an explicit override is deliberate. If it points at a missing/broken file we must
        // surface that loudly (a Read error) rather than silently falling through to the home
        // files, which would mask a misconfiguration.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let missing = temp_credentials_path();
        let _ = std::fs::remove_file(&missing);
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", missing.to_str().unwrap());
        }
        let res = GrokCredentials::load_resolved_async().await;
        unsafe {
            match old {
                Some(v) => std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
        }
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::Read(_)));
    }

    #[tokio::test]
    // See resolved_honors_explicit_env_override_first: env lock held across await is safe
    // on the current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn ambient_static_missing_token_falls_through_to_grok_cli_login() {
        // WHY: a stale ~/.xai/.credentials.json from an abandoned static-key attempt must
        // not break the common path where the user has a valid Grok CLI login in
        // ~/.grok/auth.json. Only the explicit XAI_CREDENTIALS_PATH override remains hard.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let home = std::env::temp_dir().join(format!("omni-grok-home-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(home.join(".xai")).unwrap();
        std::fs::create_dir_all(home.join(".grok")).unwrap();
        write_temp_creds(&home.join(".xai/.credentials.json"), r#"{"apiKey":" "}"#);
        write_temp_creds(
            &home.join(".grok/auth.json"),
            &grok_cli_json("jwt-from-cli", "2999-01-01T00:00:00Z"),
        );

        let old_home = std::env::var("HOME").ok();
        let old_path = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("HOME", &home);
            std::env::remove_var("XAI_CREDENTIALS_PATH");
        }
        let res = GrokCredentials::load_resolved_async().await;
        unsafe {
            match old_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match old_path {
                Some(v) => std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
        }
        let _ = std::fs::remove_dir_all(&home);

        let creds = res.expect("stale ambient static file must fall through to CLI login");
        assert_eq!(creds.api_key, "jwt-from-cli");
    }
}
