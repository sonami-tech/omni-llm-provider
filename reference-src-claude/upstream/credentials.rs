//! Read OAuth subscription credentials from `~/.claude/.credentials.json`.
//!
//! Locked design: never cache. Always re-read per request. claude CLI's
//! background refresh keeps the file current. We just read what's there.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::errors::UpstreamError;

/// On-disk shape of `~/.claude/.credentials.json`.
#[derive(Debug, Clone, Deserialize)]
struct CredentialsFile {
	#[serde(rename = "claudeAiOauth")]
	claude_ai_oauth: ClaudeAiOauth,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaudeAiOauth {
	#[serde(rename = "accessToken")]
	access_token: String,
	#[serde(rename = "expiresAt")]
	expires_at: Option<i64>,
	#[serde(rename = "subscriptionType")]
	subscription_type: Option<String>,
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
	/// Performs the file read via `tokio::fs` so a slow/stalled filesystem cannot
	/// stall a Tokio worker thread. Never cached.
	pub async fn load_fresh_async(path: &Path) -> Result<Self, UpstreamError> {
		let bytes = tokio::fs::read(path)
			.await
			.map_err(UpstreamError::CredentialsRead)?;
		Self::from_bytes(&bytes)
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

	/// Surface a clear error if the token is past its expiry. Caller should
	/// re-read from disk before treating this as terminal — claude CLI may
	/// have just refreshed.
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
