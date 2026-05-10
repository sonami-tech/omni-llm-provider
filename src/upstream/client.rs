//! HTTP client to api.anthropic.com.
//!
//! Phase 1 scope: build a reqwest client with rustls + http2, send a JSON
//! body to /v1/messages?beta=true with the fingerprint headers, return the
//! parsed JSON response or a typed error.
//!
//! Streaming is added in Phase 3.

use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use futures_util::Stream;
use reqwest::Client;
use serde_json::Value;
use tracing::{debug, warn};

use super::credentials::Credentials;
use super::errors::{UpstreamError, parse_anthropic_error};
use super::fingerprint::{RequestContext, build_headers};
use super::stream::{EventStream, StreamEvent};

/// Retry budget for 5xx + transient transport errors. Per locked decision #2,
/// 429 is NEVER retried; it's surfaced as-is to the consumer. 401 is retried
/// once after a fresh credentials.json re-read.
const MAX_RETRIES: u32 = 2;

const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages?beta=true";

#[derive(Clone)]
pub struct UpstreamClient {
	http: Client,
}

impl UpstreamClient {
	pub fn new() -> Result<Self, UpstreamError> {
		let http = Client::builder()
			.use_rustls_tls()
			.http2_prior_knowledge()
			.http2_keep_alive_interval(Some(Duration::from_secs(30)))
			.tcp_keepalive(Duration::from_secs(60))
			.connect_timeout(Duration::from_secs(15))
			.timeout(Duration::from_secs(600))
			.pool_idle_timeout(Some(Duration::from_secs(90)))
			.pool_max_idle_per_host(8)
			.build()
			.map_err(UpstreamError::Transport)?;
		Ok(Self { http })
	}

	/// Send a non-streaming JSON body to /v1/messages with retry on 5xx and
	/// transient errors. Returns the parsed JSON response on 2xx, or a typed
	/// UpstreamError otherwise. 429 is NOT retried.
	pub async fn send_messages_json(
		&self,
		creds: &Credentials,
		ctx: &RequestContext,
		body: &Value,
	) -> Result<Value, UpstreamError> {
		let mut creds_owned = creds.clone();
		let mut ctx_owned = ctx.clone();
		let mut refreshed_credentials = false;

		for attempt in 0..=MAX_RETRIES {
			let result = self.send_messages_json_once(&creds_owned, &ctx_owned, body).await;
			match result {
				Ok(v) => return Ok(v),
				Err(e) => {
					let status = match &e {
						UpstreamError::Anthropic { status, .. } => Some(*status),
						_ => None,
					};

					// 401: re-read credentials once, then retry.
					if status == Some(401) && !refreshed_credentials {
						refreshed_credentials = true;
						match Credentials::load_fresh(&Credentials::default_path()) {
							Ok(fresh) => {
								warn!("upstream 401, re-reading credentials.json and retrying");
								creds_owned = fresh;
								ctx_owned.next_attempt();
								continue;
							}
							Err(_) => return Err(e),
						}
					}

					// 429: pass-through, no retry.
					if status == Some(429) {
						return Err(e);
					}

					// Other terminal: pass-through.
					if !e.is_transient() {
						return Err(e);
					}

					if attempt >= MAX_RETRIES {
						return Err(e);
					}

					let backoff = Duration::from_millis(250u64.saturating_mul(1u64 << attempt));
					debug!(attempt = attempt + 1, ?backoff, "transient upstream error, retrying");
					tokio::time::sleep(backoff).await;
					ctx_owned.next_attempt();
				}
			}
		}
		// Loop exits via return; this is unreachable.
		Err(UpstreamError::Decode("retry loop exhausted".into()))
	}

	async fn send_messages_json_once(
		&self,
		creds: &Credentials,
		ctx: &RequestContext,
		body: &Value,
	) -> Result<Value, UpstreamError> {
		let headers = build_headers(creds, ctx);

		let resp = self
			.http
			.post(ANTHROPIC_MESSAGES_URL)
			.headers(headers)
			.json(body)
			.send()
			.await?;

		let status = resp.status();
		let bytes = resp.bytes().await?;

		if status.is_success() {
			serde_json::from_slice(&bytes)
				.map_err(|e| UpstreamError::Decode(format!("response json parse: {e}")))
		} else {
			let body_str = String::from_utf8_lossy(&bytes).into_owned();
			let parsed = parse_anthropic_error(&body_str);
			Err(UpstreamError::Anthropic {
				status: status.as_u16(),
				body: body_str,
				parsed,
			})
		}
	}

	/// Send a streaming Messages request with retry on the *initial response*
	/// only. Once the byte stream begins flowing we cannot retry without
	/// re-emitting bytes the consumer has already seen. 429 is pass-through.
	pub async fn send_messages_stream(
		&self,
		creds: &Credentials,
		ctx: &RequestContext,
		body: &Value,
	) -> Result<impl Stream<Item = Result<StreamEvent, UpstreamError>> + Send + 'static + use<>, UpstreamError> {
		let mut creds_owned = creds.clone();
		let mut ctx_owned = ctx.clone();
		let mut refreshed_credentials = false;

		for attempt in 0..=MAX_RETRIES {
			match self.send_messages_stream_once(&creds_owned, &ctx_owned, body).await {
				Ok(s) => return Ok(s),
				Err(e) => {
					let status = match &e {
						UpstreamError::Anthropic { status, .. } => Some(*status),
						_ => None,
					};
					if status == Some(401) && !refreshed_credentials {
						refreshed_credentials = true;
						if let Ok(fresh) = Credentials::load_fresh(&Credentials::default_path()) {
							warn!("upstream 401 on stream open, re-reading credentials.json");
							creds_owned = fresh;
							ctx_owned.next_attempt();
							continue;
						}
						return Err(e);
					}
					if status == Some(429) || !e.is_transient() || attempt >= MAX_RETRIES {
						return Err(e);
					}
					let backoff = Duration::from_millis(250u64.saturating_mul(1u64 << attempt));
					debug!(attempt = attempt + 1, ?backoff, "transient stream-open error, retrying");
					tokio::time::sleep(backoff).await;
					ctx_owned.next_attempt();
				}
			}
		}
		Err(UpstreamError::Decode("retry loop exhausted".into()))
	}

	async fn send_messages_stream_once(
		&self,
		creds: &Credentials,
		ctx: &RequestContext,
		body: &Value,
	) -> Result<EventStream<BoxByteStream>, UpstreamError> {
		let headers = build_headers(creds, ctx);
		let resp = self
			.http
			.post(ANTHROPIC_MESSAGES_URL)
			.headers(headers)
			.json(body)
			.send()
			.await?;

		let status = resp.status();
		if !status.is_success() {
			let bytes = resp.bytes().await?;
			let body_str = String::from_utf8_lossy(&bytes).into_owned();
			let parsed = parse_anthropic_error(&body_str);
			return Err(UpstreamError::Anthropic {
				status: status.as_u16(),
				body: body_str,
				parsed,
			});
		}

		let boxed: BoxByteStream = Box::pin(resp.bytes_stream());
		Ok(EventStream::new(boxed))
	}
}

type BoxByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	/// Phase 1 gate. Hits api.anthropic.com with subscription OAuth from
	/// `~/.claude/.credentials.json`. Marked `#[ignore]` so the default
	/// `cargo test` doesn't burn rate limit. Run explicitly:
	///
	/// ```sh
	/// cargo test -p claude-code-provider --bin claude-code-provider \
	///     upstream::client::tests::live_anthropic_hello_world \
	///     -- --ignored --nocapture
	/// ```
	#[tokio::test]
	#[ignore]
	async fn live_anthropic_hello_world() {
		let path = Credentials::default_path();
		assert!(
			path.exists(),
			"credentials file not found at {:?}; cannot run live test",
			path
		);
		let creds = Credentials::load_fresh(&path).expect("load creds");
		creds.check_expired().expect("token not expired");

		let client = UpstreamClient::new().expect("build client");
		let ctx = RequestContext::new_reply();

		let body = json!({
			"model": "claude-haiku-4-5",
			"max_tokens": 32,
			"system": "You are a helpful assistant.",
			"messages": [
				{ "role": "user", "content": "Say OK" }
			]
		});

		let resp = client
			.send_messages_json(&creds, &ctx, &body)
			.await
			.expect("anthropic call succeeds");

		let content = resp
			.get("content")
			.and_then(|v| v.as_array())
			.expect("content array");
		assert!(!content.is_empty(), "non-empty content");

		let text: String = content
			.iter()
			.filter_map(|b| b.get("text").and_then(|t| t.as_str()))
			.collect();
		assert!(!text.is_empty(), "got some text; full resp: {:?}", resp);

		eprintln!("Phase 1 gate PASS — text: {:?}", text);
	}
}
