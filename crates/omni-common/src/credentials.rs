//! Grok / xAI credentials loading, modeled exactly after the Claude Code Provider technique.
//!
//! Locked design (same as CCP): never cache. Always re-read per request. 
//! This picks up any background refresh or key rotation the user may have performed
//! (e.g. via xAI console or a login helper that writes the file).
//!
//! Default location: `~/.xai/.credentials.json` (or $XAI_CREDENTIALS_PATH to override).
//! The on-disk shape is intentionally simple and documented so users/tools can create it.
//!
//! Example file:
//! {
//!   "apiKey": "xai-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
//! }
//!
//! We also tolerate a top-level "xaiApiKey" for compatibility with some third-party
//! tools that store xAI OAuth or key material.

use std::path::{Path, PathBuf};

use serde::Deserialize;

// Local error for credentials (kept small; mapped by callers to ProviderError or AppError).
#[derive(Debug, thiserror::Error)]
pub enum GrokCredentialsError {
    #[error("read: {0}")]
    Read(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("missing apiKey (or xaiApiKey) in credentials file")]
    MissingToken,
    #[error("token expired")]
    Expired,
}

/// On-disk shape of the Grok credentials file.
#[derive(Debug, Clone, Deserialize)]
struct CredentialsFile {
    #[serde(rename = "apiKey")]
    api_key: Option<String>,
    #[serde(rename = "xaiApiKey")]
    xai_api_key: Option<String>,
}

/// In-memory parsed Grok credentials.
#[derive(Debug, Clone)]
pub struct GrokCredentials {
    pub api_key: String,
}

impl GrokCredentials {
    /// Default location: `~/.xai/.credentials.json`. Override via $XAI_CREDENTIALS_PATH.
    /// This mirrors the CCP `default_path` + CLAUDE_CREDENTIALS_PATH logic exactly.
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("XAI_CREDENTIALS_PATH") {
            return PathBuf::from(p);
        }
        match dirs::home_dir() {
            Some(home) => home.join(".xai").join(".credentials.json"),
            None => PathBuf::from(".xai/.credentials.json"),
        }
    }

    /// Read and parse the credentials file synchronously. Never cached.
    /// Use outside the async hot path (e.g. startup); handlers should prefer the async variant.
    pub fn load_fresh(path: &Path) -> Result<Self, GrokCredentialsError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Async version that does not block the Tokio worker.
    /// Never cached. Same "fresh per request" contract as CCP.
    pub async fn load_fresh_async(path: &Path) -> Result<Self, GrokCredentialsError> {
        let bytes = tokio::fs::read(path).await?;
        Self::from_bytes(&bytes)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, GrokCredentialsError> {
        let parsed: CredentialsFile = serde_json::from_slice(bytes)?;

        let key = parsed
            .api_key
            .or(parsed.xai_api_key)
            .filter(|k| !k.trim().is_empty())
            .ok_or(GrokCredentialsError::MissingToken)?;

        Ok(GrokCredentials { api_key: key })
    }

    /// For Grok API keys there is typically no expiry (unlike Claude OAuth tokens),
    /// but we keep the hook for symmetry with CCP and future OAuth support.
    pub fn check_expired(&self) -> Result<(), GrokCredentialsError> {
        // No-op for standard API keys. If we later support time-limited tokens
        // (e.g. from xAI OAuth in third-party tools), implement here and callers
        // will re-read from disk (exactly like CCP).
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::MissingToken));
    }

    #[test]
    fn default_path_respects_env_override_xai_credentials_path() {
        let p = temp_credentials_path();
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        // SAFETY: test-only mutation of process env for override verification; single threaded test context.
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", p.to_str().unwrap());
        }
        let got = GrokCredentials::default_path();
        // restore
        unsafe {
            if let Some(v) = old { std::env::set_var("XAI_CREDENTIALS_PATH", v); } else { let _ = std::env::remove_var("XAI_CREDENTIALS_PATH"); }
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
        let c = GrokCredentials { api_key: "xai-foo".into() };
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
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::MissingToken));
    }

    // Env override XAI_CREDENTIALS_PATH (already tested; re-exercise with async load too).
    #[test]
    fn env_override_xai_credentials_path() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "env-override-key"}"#);
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe { std::env::set_var("XAI_CREDENTIALS_PATH", p.to_str().unwrap()); }
        let got = GrokCredentials::default_path();
        let c = GrokCredentials::load_fresh(&got).unwrap();
        unsafe {
            if let Some(v) = old { std::env::set_var("XAI_CREDENTIALS_PATH", v); } else { let _ = std::env::remove_var("XAI_CREDENTIALS_PATH"); }
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
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::MissingToken));
    }
}
