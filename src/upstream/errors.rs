//! Upstream error types and retry classification.

use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UpstreamError {
	#[error("credentials.json read failed: {0}")]
	CredentialsRead(#[source] std::io::Error),

	#[error("credentials.json malformed: {0}")]
	CredentialsParse(#[source] serde_json::Error),

	#[error("credentials.json missing accessToken")]
	CredentialsMissingToken,

	#[error("OAuth token expired (per credentials.json expiresAt). Run `claude` once to refresh.")]
	TokenExpired,

	#[error("HTTP transport: {0}")]
	Transport(#[from] reqwest::Error),

	#[error("Anthropic returned HTTP {status}: {body}")]
	Anthropic { status: u16, body: String, parsed: Option<AnthropicErrorBody> },

	#[error("response decode: {0}")]
	Decode(String),
}

impl UpstreamError {
	/// Classify whether the operation should be retried.
	///
	/// 429 and 5xx are transient. 401/403/400 are terminal (no retry helps).
	/// Network errors are transient.
	pub fn is_transient(&self) -> bool {
		match self {
			UpstreamError::Transport(e) => e.is_timeout() || e.is_connect() || e.is_request(),
			UpstreamError::Anthropic { status, .. } => *status >= 500 || *status == 429,
			_ => false,
		}
	}

	/// Map to an HTTP status to surface to the consumer.
	pub fn surface_status(&self) -> u16 {
		match self {
			UpstreamError::Anthropic { status, .. } => *status,
			UpstreamError::TokenExpired | UpstreamError::CredentialsMissingToken => 401,
			UpstreamError::CredentialsRead(_) | UpstreamError::CredentialsParse(_) => 500,
			UpstreamError::Transport(e) if e.is_timeout() => 504,
			UpstreamError::Transport(_) => 502,
			UpstreamError::Decode(_) => 502,
		}
	}
}

/// Anthropic standard error envelope: `{"type":"error","error":{"type":..,"message":..}}`.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorBody {
	#[serde(rename = "type")]
	pub kind: String,
	pub error: AnthropicErrorDetail,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorDetail {
	#[serde(rename = "type")]
	pub kind: String,
	pub message: String,
}

pub fn parse_anthropic_error(body: &str) -> Option<AnthropicErrorBody> {
	serde_json::from_str(body).ok()
}
