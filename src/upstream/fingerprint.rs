//! Build the outbound header set for api.anthropic.com requests, mimicking
//! the claude CLI wire fingerprint.
//!
//! Baseline captured 2026-05-10 from claude CLI 2.1.138 SDK 0.93.0 against
//! `claude --print --model claude-haiku-4-5 ...`. See
//! `tools/fingerprint/BASELINE_HEADERS.md` for the source-of-truth notes.

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use uuid::Uuid;

use super::credentials::Credentials;

/// CCP claims to be claude CLI 2.1.138. Update when we re-baseline.
pub const CLAUDE_CLI_VERSION: &str = "2.1.138";
pub const STAINLESS_PACKAGE_VERSION: &str = "0.93.0";
pub const STAINLESS_RUNTIME_VERSION: &str = "v24.3.0";
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default beta-header set, matching the captured "user reply" flow (most
/// permissive). Includes claude-code-20250219 (turns on Claude Code-mode
/// behavior including OAuth-only-models eligibility) and oauth-2025-04-20
/// (Bearer-token acceptance).
pub const DEFAULT_BETA: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,advisor-tool-2026-03-01,extended-cache-ttl-2025-04-11";

/// What kind of request this is — controls minor header variations.
#[derive(Debug, Clone, Copy)]
pub enum RequestKind {
	/// A user-facing reply request. Default beta list.
	Reply,
}

/// Per-call ephemeral context. Session ID stays stable across a logical
/// "session"; client_request_id is regenerated per HTTP call.
#[derive(Debug, Clone)]
pub struct RequestContext {
	pub session_id: Uuid,
	pub client_request_id: Uuid,
	pub retry_count: u32,
	pub kind: RequestKind,
}

impl RequestContext {
	pub fn new_reply() -> Self {
		Self {
			session_id: Uuid::new_v4(),
			client_request_id: Uuid::new_v4(),
			retry_count: 0,
			kind: RequestKind::Reply,
		}
	}

	pub fn with_session(mut self, session_id: Uuid) -> Self {
		self.session_id = session_id;
		self
	}

	pub fn next_attempt(&mut self) {
		self.retry_count += 1;
		self.client_request_id = Uuid::new_v4();
	}
}

/// Build the full outbound header set for a Messages call.
///
/// Header names are emitted lowercase because HTTP/2 requires lowercase and
/// HTTP/1.1 is case-insensitive. Anthropic does not appear to care about case.
pub fn build_headers(creds: &Credentials, ctx: &RequestContext) -> HeaderMap {
	let mut h = HeaderMap::new();

	insert(&mut h, "accept", "application/json");

	let bearer = format!("Bearer {}", creds.access_token);
	insert(&mut h, "authorization", &bearer);

	insert(&mut h, "content-type", "application/json");

	let ua = format!("claude-cli/{} (external, sdk-cli)", CLAUDE_CLI_VERSION);
	insert(&mut h, "user-agent", &ua);

	insert(&mut h, "x-claude-code-session-id", &ctx.session_id.to_string());

	insert(&mut h, "x-stainless-arch", "x64");
	insert(&mut h, "x-stainless-lang", "js");
	insert(&mut h, "x-stainless-os", "Linux");
	insert(&mut h, "x-stainless-package-version", STAINLESS_PACKAGE_VERSION);
	insert(&mut h, "x-stainless-retry-count", &ctx.retry_count.to_string());
	insert(&mut h, "x-stainless-runtime", "node");
	insert(&mut h, "x-stainless-runtime-version", STAINLESS_RUNTIME_VERSION);
	insert(&mut h, "x-stainless-timeout", "600");

	let beta = match ctx.kind {
		RequestKind::Reply => DEFAULT_BETA,
	};
	insert(&mut h, "anthropic-beta", beta);

	insert(&mut h, "anthropic-dangerous-direct-browser-access", "true");
	insert(&mut h, "anthropic-version", ANTHROPIC_VERSION);
	insert(&mut h, "x-app", "cli");
	insert(&mut h, "x-client-request-id", &ctx.client_request_id.to_string());

	h
}

fn insert(h: &mut HeaderMap, name: &'static str, value: &str) {
	let n = HeaderName::from_static(name);
	if let Ok(v) = HeaderValue::from_str(value) {
		h.insert(n, v);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn fixture_creds() -> Credentials {
		Credentials {
			access_token: "sk-ant-oat01-test-token".into(),
			expires_at_ms: None,
			subscription_type: Some("max".into()),
		}
	}

	#[test]
	fn header_set_matches_claude_baseline() {
		// Lock in the header NAMES we send. Values are mostly captured
		// constants; the dynamic parts are user-agent (versioned), session id,
		// retry count, and client request id.
		//
		// If this test fails, re-run a baseline capture with mitmproxy and
		// update either the constants or the assertion.
		let creds = fixture_creds();
		let ctx = RequestContext::new_reply();
		let h = build_headers(&creds, &ctx);

		let expected_names = [
			"accept",
			"authorization",
			"content-type",
			"user-agent",
			"x-claude-code-session-id",
			"x-stainless-arch",
			"x-stainless-lang",
			"x-stainless-os",
			"x-stainless-package-version",
			"x-stainless-retry-count",
			"x-stainless-runtime",
			"x-stainless-runtime-version",
			"x-stainless-timeout",
			"anthropic-beta",
			"anthropic-dangerous-direct-browser-access",
			"anthropic-version",
			"x-app",
			"x-client-request-id",
		];
		for name in expected_names {
			assert!(h.contains_key(name), "missing header `{name}`");
		}

		// Spot-check critical static values.
		assert_eq!(h.get("anthropic-version").unwrap(), "2023-06-01");
		assert_eq!(h.get("anthropic-dangerous-direct-browser-access").unwrap(), "true");
		assert_eq!(h.get("x-app").unwrap(), "cli");
		assert_eq!(h.get("x-stainless-arch").unwrap(), "x64");
		assert_eq!(h.get("x-stainless-lang").unwrap(), "js");
		assert_eq!(h.get("x-stainless-os").unwrap(), "Linux");
		assert_eq!(h.get("x-stainless-runtime").unwrap(), "node");

		// Beta list must include the two load-bearing tokens.
		let beta = h.get("anthropic-beta").unwrap().to_str().unwrap();
		assert!(
			beta.contains("oauth-2025-04-20"),
			"beta list missing oauth-2025-04-20: {beta}"
		);
		assert!(
			beta.contains("claude-code-20250219"),
			"beta list missing claude-code-20250219: {beta}"
		);
		assert!(
			beta.contains("interleaved-thinking-2025-05-14"),
			"beta list missing interleaved-thinking-2025-05-14: {beta}"
		);

		// Authorization is Bearer-shaped with the OAuth prefix.
		let auth = h.get("authorization").unwrap().to_str().unwrap();
		assert!(auth.starts_with("Bearer sk-ant-oat01-"));

		// User-Agent claims to be claude-cli, including the version.
		let ua = h.get("user-agent").unwrap().to_str().unwrap();
		assert!(ua.contains(CLAUDE_CLI_VERSION), "user-agent: {ua}");
		assert!(ua.contains("sdk-cli"), "user-agent: {ua}");
	}

	#[test]
	fn next_attempt_increments_retry_count_and_rotates_request_id() {
		let mut ctx = RequestContext::new_reply();
		let first_id = ctx.client_request_id;
		ctx.next_attempt();
		assert_eq!(ctx.retry_count, 1);
		assert_ne!(ctx.client_request_id, first_id);
	}
}
