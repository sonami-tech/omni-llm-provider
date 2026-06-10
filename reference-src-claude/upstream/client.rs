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
use super::fingerprint::{FingerprintProfile, RequestContext, build_headers, default_profile};
use super::stream::{EventStream, RawEventStream, RawFrame, StreamEvent};

/// Retry budget for 5xx + transient transport errors. Per locked decision #2,
/// 429 is NEVER retried; it's surfaced as-is to the consumer. 401 is retried
/// once after a fresh credentials.json re-read.
const MAX_RETRIES: u32 = 2;

const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages?beta=true";
const ANTHROPIC_COUNT_TOKENS_URL: &str =
	"https://api.anthropic.com/v1/messages/count_tokens?beta=true";

#[derive(Clone)]
pub struct UpstreamClient {
    http: Client,
    profile: &'static FingerprintProfile,
}

impl UpstreamClient {
    pub fn new() -> Result<Self, UpstreamError> {
        Self::new_with_profile(default_profile())
    }

    pub fn new_with_profile(profile: &'static FingerprintProfile) -> Result<Self, UpstreamError> {
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
        Ok(Self { http, profile })
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
            let body_bytes = self
                .profile
                .finalize_body_json(body, &ctx_owned)
                .map_err(|e| UpstreamError::Decode(format!("request json serialize: {e}")))?;
            let result = self
                .send_messages_json_once(&creds_owned, &ctx_owned, body_bytes)
                .await;
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
                        match Credentials::load_fresh_async(&Credentials::default_path()).await {
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
                    debug!(
                        attempt = attempt + 1,
                        ?backoff,
                        "transient upstream error, retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    ctx_owned.next_attempt();
                }
            }
        }
        // Loop exits via return; this is unreachable.
        Err(UpstreamError::Decode("retry loop exhausted".into()))
    }

    /// Send a `count_tokens` request. Same fingerprint headers + 401-refresh
    /// retry as the messages path, but posts to the count_tokens endpoint and
    /// returns the parsed `{"input_tokens": N}` JSON. 429 is NOT retried.
    ///
    /// NOTE: the body is sent VERBATIM (not run through `finalize_body_json`):
    /// count_tokens counts the client body as-sent and carries no injected
    /// billing-header placeholder, so there is no cch to finalize. This matches
    /// the native-surface decision to count the client's own body.
    pub async fn count_tokens(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: &Value,
    ) -> Result<Value, UpstreamError> {
        let mut creds_owned = creds.clone();
        let mut ctx_owned = ctx.clone();
        let mut refreshed_credentials = false;

        for attempt in 0..=MAX_RETRIES {
            let body_bytes = serde_json::to_vec(body)
                .map_err(|e| UpstreamError::Decode(format!("count_tokens serialize: {e}")))?;
            let result = self
                .post_json_once(ANTHROPIC_COUNT_TOKENS_URL, &creds_owned, &ctx_owned, body_bytes)
                .await;
            match result {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let status = match &e {
                        UpstreamError::Anthropic { status, .. } => Some(*status),
                        _ => None,
                    };
                    if status == Some(401) && !refreshed_credentials {
                        refreshed_credentials = true;
                        match Credentials::load_fresh_async(&Credentials::default_path()).await {
                            Ok(fresh) => {
                                warn!("count_tokens 401, re-reading credentials.json and retrying");
                                creds_owned = fresh;
                                ctx_owned.next_attempt();
                                continue;
                            }
                            Err(_) => return Err(e),
                        }
                    }
                    if status == Some(429) || !e.is_transient() || attempt >= MAX_RETRIES {
                        return Err(e);
                    }
                    let backoff = Duration::from_millis(250u64.saturating_mul(1u64 << attempt));
                    tokio::time::sleep(backoff).await;
                    ctx_owned.next_attempt();
                }
            }
        }
        Err(UpstreamError::Decode("count_tokens retry loop exhausted".into()))
    }

    async fn send_messages_json_once(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: Vec<u8>,
    ) -> Result<Value, UpstreamError> {
        self.post_json_once(ANTHROPIC_MESSAGES_URL, creds, ctx, body).await
    }

    /// POST a finalized body to a JSON endpoint and parse the 2xx body, or map a
    /// non-2xx into a typed `UpstreamError::Anthropic`. Shared by the messages
    /// and count_tokens non-streaming paths.
    async fn post_json_once(
        &self,
        url: &str,
        creds: &Credentials,
        ctx: &RequestContext,
        body: Vec<u8>,
    ) -> Result<Value, UpstreamError> {
        let headers = build_headers(creds, ctx, self.profile);

        let resp = self
            .http
            .post(url)
            .headers(headers)
            .body(body)
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
    ) -> Result<
        impl Stream<Item = Result<StreamEvent, UpstreamError>> + Send + 'static + use<>,
        UpstreamError,
    > {
        let mut creds_owned = creds.clone();
        let mut ctx_owned = ctx.clone();
        let mut refreshed_credentials = false;

        for attempt in 0..=MAX_RETRIES {
            let body_bytes = self
                .profile
                .finalize_body_json(body, &ctx_owned)
                .map_err(|e| UpstreamError::Decode(format!("request json serialize: {e}")))?;
            match self
                .send_messages_stream_once(&creds_owned, &ctx_owned, body_bytes)
                .await
            {
                Ok(s) => return Ok(s),
                Err(e) => {
                    let status = match &e {
                        UpstreamError::Anthropic { status, .. } => Some(*status),
                        _ => None,
                    };
                    if status == Some(401) && !refreshed_credentials {
                        refreshed_credentials = true;
                        if let Ok(fresh) =
                            Credentials::load_fresh_async(&Credentials::default_path()).await
                        {
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
                    debug!(
                        attempt = attempt + 1,
                        ?backoff,
                        "transient stream-open error, retrying"
                    );
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
        body: Vec<u8>,
    ) -> Result<EventStream<BoxByteStream>, UpstreamError> {
        let boxed = self.open_message_byte_stream(creds, ctx, body).await?;
        Ok(EventStream::new(boxed))
    }

    /// Streaming variant for the NATIVE Anthropic surface: yields faithful raw
    /// SSE frames (`RawFrame { event, data }`) instead of the lossy typed
    /// `StreamEvent`, so the handler can forward Anthropic's wire format
    /// byte-faithfully. Same fingerprint headers, body finalization, and
    /// stream-open retry semantics as `send_messages_stream`.
    pub async fn send_messages_stream_raw(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: &Value,
    ) -> Result<
        impl Stream<Item = Result<RawFrame, UpstreamError>> + Send + 'static + use<>,
        UpstreamError,
    > {
        let mut creds_owned = creds.clone();
        let mut ctx_owned = ctx.clone();
        let mut refreshed_credentials = false;

        for attempt in 0..=MAX_RETRIES {
            let body_bytes = self
                .profile
                .finalize_body_json(body, &ctx_owned)
                .map_err(|e| UpstreamError::Decode(format!("request json serialize: {e}")))?;
            match self
                .open_message_byte_stream(&creds_owned, &ctx_owned, body_bytes)
                .await
            {
                Ok(boxed) => return Ok(RawEventStream::new(boxed)),
                Err(e) => {
                    let status = match &e {
                        UpstreamError::Anthropic { status, .. } => Some(*status),
                        _ => None,
                    };
                    if status == Some(401) && !refreshed_credentials {
                        refreshed_credentials = true;
                        if let Ok(fresh) =
                            Credentials::load_fresh_async(&Credentials::default_path()).await
                        {
                            warn!("upstream 401 on raw stream open, re-reading credentials.json");
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
                    tokio::time::sleep(backoff).await;
                    ctx_owned.next_attempt();
                }
            }
        }
        Err(UpstreamError::Decode("retry loop exhausted".into()))
    }

    /// Open the messages SSE byte stream, mapping a non-2xx response to a typed
    /// error. Shared by the typed and raw streaming paths.
    async fn open_message_byte_stream(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: Vec<u8>,
    ) -> Result<BoxByteStream, UpstreamError> {
        let headers = build_headers(creds, ctx, self.profile);
        let resp = self
            .http
            .post(ANTHROPIC_MESSAGES_URL)
            .headers(headers)
            .body(body)
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
        Ok(boxed)
    }
}

type BoxByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
