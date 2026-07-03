//! provider-claude
//!
//! **All** Claude Code / Anthropic Max specific logic that must remain isolated
//! to protect the fingerprint invariant.
//!
//! This crate owns:
//! - Fingerprint profiles, cch checksum computation + injection, billing headers
//! - `~/.claude/.credentials.json` fresh-read + 401 retry refresh
//! - Exact Anthropic Messages wire types (serde order is load-bearing for cch)
//! - Identity injection (billing marker + canonical "You are Claude Code..." preamble)
//! - Wire defaults, per-model beta lists, model catalog + resolution (claude-specific)
//! - Translation adapters CanonicalRequest <-> MessagesRequest (using above)
//! - The UpstreamClient that applies the headers + finalize_body_json
//!
//! It depends on omni-common (only for Replacements + Stats concepts) and
//! omni-core (Canonical* + LlmProvider trait) for the shared contract.
//!
//! NOTHING Claude-specific (cch, betas, preamble, profiles, CLAUDE_CODE_* consts,
//! billing suffix, xxh64, stainless header values, etc.) is allowed in omni-* crates.
//!
//! The original invariant is preserved by porting the reference-src-claude
//! logic with its unit-test pins (header names/values, beta per-model lists,
//! billing suffix vectors, cch snapshots) intact.

pub mod anthropic_passthrough;
pub mod credentials;
pub mod fingerprint;
pub mod models;
pub mod translate;
pub mod upstream;

pub use fingerprint::{
    CLAUDE_CODE_SYSTEM_PREAMBLE, FingerprintProfile, RequestContext, RequestKind, default_profile,
    resolve_profile, valid_profile_selectors,
};
pub use upstream::{UpstreamClient, UpstreamError};

use async_trait::async_trait;
use futures_util::StreamExt;
use omni_common::Replacements;
use omni_common::env_nonempty;
use omni_core::{
    CanonicalRequest, CanonicalResponse, CanonicalStream, CanonicalStreamEvent, CanonicalUsage,
    LlmProvider, ProviderError,
};
use tracing::debug;

use crate::upstream::stream::{BlockDelta, BlockStart, StreamEvent};

/// Stateful converter from Anthropic typed [`StreamEvent`]s to the
/// provider-agnostic [`CanonicalStreamEvent`] vocabulary.
///
/// Anthropic addresses content by `content_block` index and signals tool calls
/// via a `content_block_start` (carrying id + name) followed by
/// `input_json_delta` fragments. The canonical `ToolCallDelta` instead carries a
/// stable per-tool `index` plus id/name on its first delta. This converter maps
/// Anthropic block indexes to dense canonical tool indexes and remembers the
/// stop reason / token usage so it can emit a terminal `Finish` (and a `Usage`)
/// when the message stops. Mirrors the event handling in the working reference's
/// to_oai_stream converter, retargeted to canonical events.
#[derive(Default)]
struct ClaudeStreamConverter {
    /// Anthropic content_block index -> canonical tool-call index (tool-use blocks only).
    tool_block_to_index: std::collections::HashMap<u32, u32>,
    next_tool_index: u32,
    stop_reason: Option<String>,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    usage_emitted: bool,
    /// Set once a terminal `Finish` has been emitted by ANY path (`finish_events`
    /// for `message_stop`, or the `Error` arm). The `send_stream` EOF guard checks
    /// this so a stream that already produced its terminal `Finish` does not get a
    /// SECOND one appended at end-of-byte-stream.
    finished: bool,
}

impl ClaudeStreamConverter {
    fn on_event(&mut self, event: StreamEvent) -> Vec<CanonicalStreamEvent> {
        match event {
            StreamEvent::MessageStart {
                id,
                input_tokens,
                output_tokens,
                ..
            } => {
                let out = vec![CanonicalStreamEvent::ResponseMetadata(
                    omni_core::CanonicalResponseMetadata {
                        id: Some(id),
                        ..Default::default()
                    },
                )];
                if input_tokens.is_some() {
                    self.input_tokens = input_tokens;
                }
                if output_tokens.is_some() {
                    self.output_tokens = output_tokens;
                }
                out
            }
            StreamEvent::ContentBlockStart {
                index,
                block: BlockStart::ToolUse { id, name },
            } => {
                let tool_index = self.next_tool_index;
                self.next_tool_index += 1;
                self.tool_block_to_index.insert(index, tool_index);
                vec![CanonicalStreamEvent::ToolCallDelta {
                    index: tool_index,
                    id: Some(id),
                    name: Some(name),
                    arguments_delta: String::new(),
                }]
            }
            // Text / Thinking / Other block starts carry no canonical payload.
            StreamEvent::ContentBlockStart { .. } => vec![],
            StreamEvent::ContentBlockDelta { index, delta } => match delta {
                BlockDelta::Text(s) => vec![CanonicalStreamEvent::TextDelta(s)],
                BlockDelta::InputJson(s) => {
                    // Arguments fragment for the tool call opened at this block.
                    match self.tool_block_to_index.get(&index) {
                        Some(&tool_index) => vec![CanonicalStreamEvent::ToolCallDelta {
                            index: tool_index,
                            id: None,
                            name: None,
                            arguments_delta: s,
                        }],
                        None => vec![],
                    }
                }
                BlockDelta::Thinking(s) => vec![CanonicalStreamEvent::ReasoningDelta(s)],
                BlockDelta::ThinkingSignature(s) => {
                    vec![CanonicalStreamEvent::ReasoningSignatureDelta(s)]
                }
                BlockDelta::Other => vec![],
            },
            StreamEvent::MessageDelta {
                stop_reason,
                output_tokens,
                ..
            } => {
                if let Some(r) = stop_reason {
                    self.stop_reason = Some(map_stop_reason(&r));
                }
                if output_tokens.is_some() {
                    self.output_tokens = output_tokens;
                }
                vec![]
            }
            StreamEvent::MessageStop => self.finish_events(),
            StreamEvent::Error { kind, message } => {
                // Surface as a terminal error so the consumer stops cleanly. This
                // IS a terminal Finish, so mark `finished` to stop the send_stream
                // EOF guard from appending a second (success) Finish if the
                // upstream then closes the byte stream.
                self.finished = true;
                vec![CanonicalStreamEvent::Finish {
                    finish_reason: Some(format!("error: {kind}: {message}")),
                }]
            }
            // Ping / ContentBlockStop / Unknown carry nothing canonical.
            _ => vec![],
        }
    }

    /// Emit the trailing Usage (once) + terminal Finish for message_stop / EOF.
    fn finish_events(&mut self) -> Vec<CanonicalStreamEvent> {
        let mut out = Vec::new();
        if !self.usage_emitted && (self.input_tokens.is_some() || self.output_tokens.is_some()) {
            self.usage_emitted = true;
            out.push(CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: self.input_tokens.unwrap_or(0) as u64,
                output_tokens: self.output_tokens.unwrap_or(0) as u64,
                ..Default::default()
            }));
        }
        out.push(CanonicalStreamEvent::Finish {
            finish_reason: self.stop_reason.clone().or(Some("stop".into())),
        });
        self.finished = true;
        out
    }
}

/// Map Anthropic stop_reason to the OpenAI-style finish_reason vocabulary so the
/// canonical stream matches the non-stream path's mapping.
fn map_stop_reason(anth: &str) -> String {
    match anth {
        "end_turn" | "stop_sequence" => "stop".into(),
        "max_tokens" => "length".into(),
        "tool_use" => "tool_calls".into(),
        other => other.into(),
    }
}

/// The Claude provider. Holds the fingerprint profile (for the invariant)
/// and a reusable upstream client.
/// Replacements are exercised via empty() hook (real caller can load from
/// omni-common and pass/apply earlier; here we demonstrate the seam inside
/// the claude path for tool-name masking etc that the gate cares about).
#[derive(Clone)]
pub struct ClaudeProvider {
    profile: &'static FingerprintProfile,
    client: UpstreamClient,
}

impl ClaudeProvider {
    /// Construct using the default (latest) fingerprint profile.
    /// Reads no credentials at construction time (per-request fresh read).
    pub fn new() -> Result<Self, ProviderError> {
        Self::new_with_profile(default_profile())
    }

    pub fn detected() -> bool {
        env_nonempty("OMNI_CLAUDE_BASE_URL").is_some()
            || env_nonempty("ANTHROPIC_BASE_URL").is_some()
            || credentials::Credentials::default_path().is_file()
    }

    pub fn new_with_profile(profile: &'static FingerprintProfile) -> Result<Self, ProviderError> {
        let client = UpstreamClient::new_with_profile(profile).map_err(|e| {
            ProviderError::Other(anyhow::Error::msg(format!("upstream client: {e}")))
        })?;
        Ok(Self { profile, client })
    }

    /// Construct a provider whose upstream client targets an explicit base URL
    /// instead of `api.anthropic.com`. Used by hermetic tests (and, potentially,
    /// router-level integration tests in the bin crates) to point the real
    /// request-build -> HTTP -> response-parse path at a local mock server. The
    /// HTTP client, headers, body finalization, and parsing are all the production
    /// code; only the host changes. Mirrors `GrokProvider::new_for_test`'s public
    /// base-URL seam so the two providers are testable the same way.
    pub fn new_for_test_with_base(
        profile: &'static FingerprintProfile,
        base_url: impl Into<String>,
    ) -> Result<Self, ProviderError> {
        let client = UpstreamClient::new_with_profile_and_base(profile, base_url).map_err(|e| {
            ProviderError::Other(anyhow::Error::msg(format!("upstream client: {e}")))
        })?;
        Ok(Self { profile, client })
    }

    /// Construct a Claude provider for an explicit custom Anthropic-compatible
    /// gateway. The gateway auth configuration replaces the default Claude Code
    /// OAuth Authorization header so a local `~/.claude/.credentials.json` token
    /// cannot be leaked to the custom endpoint.
    pub fn new_for_custom_gateway(
        profile: &'static FingerprintProfile,
        base_url: impl Into<String>,
        authorization_bearer: Option<String>,
        headers: Vec<(String, String)>,
    ) -> Result<Self, ProviderError> {
        let client = UpstreamClient::new_with_profile_base_and_auth(
            profile,
            base_url,
            upstream::ClaudeAuthConfig::Custom {
                authorization_bearer,
                headers,
                authorization_bearer_env: None,
                api_key_env: None,
                custom_headers_env: None,
            },
        )
        .map_err(|e| ProviderError::Other(anyhow::Error::msg(format!("upstream client: {e}"))))?;
        Ok(Self { profile, client })
    }

    /// Construct a custom gateway provider whose auth material is read from env
    /// on each request. This is the Omni runtime path: rotating
    /// `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, or `ANTHROPIC_CUSTOM_HEADERS`
    /// does not require rebuilding the provider.
    pub fn new_for_custom_gateway_env(
        profile: &'static FingerprintProfile,
        base_url: impl Into<String>,
        authorization_bearer_env: Option<String>,
        api_key_env: Option<String>,
        custom_headers_env: Option<String>,
    ) -> Result<Self, ProviderError> {
        let client = UpstreamClient::new_with_profile_base_and_auth(
            profile,
            base_url,
            upstream::ClaudeAuthConfig::Custom {
                authorization_bearer: None,
                headers: Vec::new(),
                authorization_bearer_env,
                api_key_env,
                custom_headers_env,
            },
        )
        .map_err(|e| ProviderError::Other(anyhow::Error::msg(format!("upstream client: {e}"))))?;
        Ok(Self { profile, client })
    }

    /// For tests / alternate profiles (e.g. pinned older for rebaseline).
    #[cfg(test)]
    pub fn new_for_test(profile: &'static FingerprintProfile) -> Self {
        // Use a dummy client; real tests avoid the network path or use integration.
        // We construct via the real new but ignore; for mapper tests we don't need.
        // For send tests that must not hit net, callers should not call send.
        let client = UpstreamClient::new_with_profile(profile).expect("test profile client");
        Self { profile, client }
    }

    pub fn profile(&self) -> &'static FingerprintProfile {
        self.profile
    }

    /// Whether to inject Anthropic top-level automatic caching for this request.
    /// This is only ever called from `send`/`send_stream`, which serve the
    /// OpenAI-compatible inbound door exclusively (the native Anthropic passthrough
    /// never reaches here). That door has no Claude Code fingerprint contract to
    /// preserve, so caching is injected automatically and unconditionally — the
    /// only escape is the `OMNI_CLAUDE_NO_AUTO_CACHE` opt-out.
    fn supports_auto_cache(&self) -> bool {
        !auto_cache_disabled()
    }

    pub(crate) async fn credentials_for_request(
        &self,
    ) -> Result<credentials::Credentials, ProviderError> {
        if self.client.uses_custom_auth() {
            Ok(credentials::Credentials::placeholder_for_custom_gateway())
        } else {
            // No creds in scope here (this IS the failure to load them), so the
            // default (1-arg) mapper is used: empty exact-secret list, still
            // prefix-scrubs.
            credentials::Credentials::load_fresh_async(&credentials::Credentials::default_path())
                .await
                .map_err(map_upstream_err)
        }
    }
}

/// Opt-out for Anthropic automatic caching on the OpenAI-inbound Claude path.
/// Caching is ON by default there (it has no fingerprint contract and injecting
/// the marker is harmless). Set `OMNI_CLAUDE_NO_AUTO_CACHE` to a truthy value
/// (`1`/`true`/`yes`/`on`, case-insensitive) to disable injection.
fn auto_cache_disabled() -> bool {
    matches!(
        env_nonempty("OMNI_CLAUDE_NO_AUTO_CACHE")
            .map(|v| v.to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

impl Default for ClaudeProvider {
    fn default() -> Self {
        Self::new().expect("default claude profile must be usable at construction")
    }
}

/// Prefix scrubber for Claude secrets in an upstream error body. Delegates to
/// the shared [`omni_common::responses_upstream::redact_prefixed_secrets`] with
/// Claude's marker set: `sk-` also covers Claude OAuth tokens
/// (`sk-ant-oat01-...`) and custom-gateway `sk-...` keys since both start with
/// `sk-`, and `eyJ` covers JWT bearers. No xAI (`xai-`) keys reach the Anthropic
/// path, so that marker is intentionally omitted.
fn redact(input: &str) -> String {
    omni_common::responses_upstream::redact_prefixed_secrets(input, &["sk-", "eyJ"])
}

/// Layered redactor for Claude error bodies: the prefix scrubber [`redact`]
/// PLUS the EXACT resolved bearer/token strings captured in `secrets`. The exact
/// list catches a token that has no known prefix; the prefix scrubber catches a
/// known-prefix secret even when creds are not in scope. `Default` yields an
/// empty secret list (still prefix-redacts). Mirrors `GrokErrorRedactor`.
#[derive(Clone, Debug, Default)]
struct ClaudeErrorRedactor {
    secrets: Vec<String>,
}

impl ClaudeErrorRedactor {
    /// Capture the exact secrets from the resolved credentials: both the full
    /// `Bearer <token>` header value and the bare token, so either form in an
    /// upstream error body is scrubbed regardless of prefix. The custom-gateway
    /// placeholder sentinel and empty tokens are skipped (no real secret).
    /// Longest-first so a longer secret is replaced before a substring of it.
    fn for_credentials(creds: &credentials::Credentials) -> Self {
        let mut secrets = Vec::new();
        let token = creds.access_token.trim();
        if !token.is_empty() && token != "custom-gateway-placeholder" {
            secrets.push(format!("Bearer {token}"));
            secrets.push(token.to_string());
        }
        secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
        secrets.dedup();
        Self { secrets }
    }

    fn redact(&self, input: &str) -> String {
        let mut out = redact(input);
        for secret in &self.secrets {
            out = out.replace(secret, "<redacted>");
        }
        out
    }
}

/// 1-arg form for callers that have no resolved credentials in scope (the
/// `anthropic_passthrough` paths). Uses a default redactor: empty exact-secret
/// list, but still prefix-scrubs `sk-`/`eyJ` from the upstream body.
fn map_upstream_err(e: UpstreamError) -> ProviderError {
    map_upstream_err_with(e, &ClaudeErrorRedactor::default())
}

/// Redaction-aware form for the credentialed send / send_stream paths: scrubs
/// the EXACT bearer/token captured in `redactor` plus the prefix markers.
fn map_upstream_err_with(e: UpstreamError, redactor: &ClaudeErrorRedactor) -> ProviderError {
    match e {
        UpstreamError::TokenExpired | UpstreamError::CredentialsMissingToken => {
            ProviderError::Auth(e.to_string())
        }
        UpstreamError::CredentialsRead(_) | UpstreamError::CredentialsParse(_) => {
            ProviderError::Auth(e.to_string())
        }
        UpstreamError::Anthropic { status, body, .. } => {
            let body = redactor.redact(&body);
            ProviderError::upstream_status(status, format!("anthropic {status}: {body}"))
        }
        UpstreamError::Transport(_) | UpstreamError::Decode(_) => {
            ProviderError::upstream(redactor.redact(&e.to_string()))
        }
    }
}

#[async_trait]
impl LlmProvider for ClaudeProvider {
    fn id(&self) -> &'static str {
        "claude"
    }

    async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError> {
        debug!(
            provider = "claude",
            model = %req.model,
            n_msgs = req.messages.len(),
            n_tools = req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            has_reasoning = req.reasoning.is_some(),
            "sending via claude (fingerprint profile {})",
            self.profile.name
        );

        // 1. Replacements hook (omni-common) on prompt surface. Must happen
        //    BEFORE identity so the billing suffix (derived from first user
        //    text) is computed on the post-replacement bytes.
        let repl = Replacements::empty();

        // 2. Build the exact wire request (model resolution inside profile),
        //    apply replacements, THEN identity injection (claude-only).
        let anth_req = translate::prepare_anthropic_request(
            &req,
            self.profile,
            &repl,
            true, // always inject for the gate (callers that want --no-preamble use a different path)
            self.supports_auto_cache(),
        )
        // Request-shaping failures (unrepresentable content, malformed tool args,
        // bad image URL, unsupported role) are the client's fault -> 400, not 500.
        .map_err(ProviderError::BadRequest)?;

        // 3. Serialize for finalize (cch lives in the billing text inside system).
        let body_val = serde_json::to_value(&anth_req).map_err(|e| {
            ProviderError::Other(anyhow::Error::msg(format!("anth serialize: {e}")))
        })?;

        let ctx = RequestContext::new_reply().with_model(anth_req.model.clone());

        // Fresh creds read for the default Claude Code path. Custom gateways
        // own auth and must not require or leak the local OAuth file.
        let creds = self.credentials_for_request().await?;
        let redactor = ClaudeErrorRedactor::for_credentials(&creds);

        // 4. Send (this does finalize_body_json which patches the 5-hex cch,
        //    builds the full header set with per-profile betas / stainless / ua,
        //    and does the 401-once refresh).
        let raw_resp = self
            .client
            .send_messages_json(&creds, &ctx, &body_val)
            .await
            .map_err(|e| map_upstream_err_with(e, &redactor))?;

        // 5. Parse into our response type (non-stream path).
        let anth_resp: translate::MessagesResponse = serde_json::from_value(raw_resp)
            .map_err(|e| ProviderError::upstream(format!("decode anth response: {e}")))?;

        // 6. Map back to canonical + apply response-scope replacements hook.
        let canon = translate::build_canonical_response(&anth_resp, &req.model, &repl);

        debug!(
            model = %canon.model,
            finish = ?canon.finish_reason,
            cache_read = canon.usage.cache_read,
            cache_creation = canon.usage.cache_creation,
            "claude response mapped to canonical"
        );

        Ok(canon)
    }

    async fn send_stream(&self, req: CanonicalRequest) -> Result<CanonicalStream, ProviderError> {
        debug!(
            provider = "claude",
            model = %req.model,
            n_msgs = req.messages.len(),
            "streaming via claude (fingerprint profile {})",
            self.profile.name
        );

        // Same outbound build as send(): replacements -> exact wire request ->
        // identity injection. Streaming only flips the `stream` flag, so the
        // fingerprint body (betas, cch, preamble, wire defaults) is identical to
        // the non-stream path. Build, serialize, then set stream=true on the
        // JSON value (the typed builder set Some(false)).
        let repl = Replacements::empty();
        let anth_req = translate::prepare_anthropic_request(
            &req,
            self.profile,
            &repl,
            true,
            self.supports_auto_cache(),
        )
        // Request-shaping failures are client faults -> 400, not 500.
        .map_err(ProviderError::BadRequest)?;
        let mut body_val = serde_json::to_value(&anth_req).map_err(|e| {
            ProviderError::Other(anyhow::Error::msg(format!("anth serialize: {e}")))
        })?;
        body_val["stream"] = serde_json::Value::Bool(true);

        let ctx = RequestContext::new_reply().with_model(anth_req.model.clone());

        let creds = self.credentials_for_request().await?;
        // Build the redactor BEFORE the stream closure: `creds` is borrowed by the
        // open call below and must not be captured by the (potentially long-lived)
        // closure. Clone the redactor into the closure for per-event redaction.
        let redactor = ClaudeErrorRedactor::for_credentials(&creds);

        // Open the upstream SSE stream (full retry / 401-refresh semantics live
        // in send_messages_stream). Yields typed Anthropic StreamEvents.
        let upstream = self
            .client
            .send_messages_stream(&creds, &ctx, &body_val)
            .await
            .map_err(|e| map_upstream_err_with(e, &redactor))?;

        // Map each Anthropic event to zero-or-more canonical events, flattening
        // into a single ordered canonical stream. The converter is stateful, so
        // it lives inside the async stream closure.
        let stream_redactor = redactor.clone();
        let canonical = async_stream::try_stream! {
            let mut conv = ClaudeStreamConverter::default();
            futures_util::pin_mut!(upstream);
            while let Some(item) = upstream.next().await {
                let event = item.map_err(|e| map_upstream_err_with(e, &stream_redactor))?;
                for canon_event in conv.on_event(event) {
                    yield canon_event;
                }
            }
            // If the upstream ended without an explicit message_stop, still emit
            // a terminal Finish so consumers always see stream completion. A
            // stream that DID see message_stop already emitted its Finish via
            // `finish_events` (which sets `finished`), so this guard prevents a
            // duplicate terminal chunk on the normal completed path.
            if !conv.finished {
                for tail in conv.finish_events() {
                    yield tail;
                }
            }
        };

        Ok(Box::pin(canonical))
    }
}

// Free fn for any legacy direct use (matches provider-grok).
pub fn provider_id() -> &'static str {
    "claude"
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::{
        CanonicalBlock, CanonicalContent,
        CanonicalMessage, /*CanonicalTool, CanonicalToolChoice*/
    };

    #[test]
    fn provider_id_and_construction() {
        assert_eq!(provider_id(), "claude");
        let p = ClaudeProvider::new().expect("default profile constructs");
        assert_eq!(p.id(), "claude");
        assert_eq!(p.profile().name, "cc-2.1.197-sdk-cli");
    }

    #[test]
    fn detected_accepts_omni_base_url_without_native_creds() {
        let old_omni = std::env::var_os("OMNI_CLAUDE_BASE_URL");
        let old_legacy = std::env::var_os("ANTHROPIC_BASE_URL");
        unsafe {
            std::env::set_var("OMNI_CLAUDE_BASE_URL", "https://claude-proxy.example.com");
            std::env::remove_var("ANTHROPIC_BASE_URL");
        }
        let detected = ClaudeProvider::detected();
        unsafe {
            match old_omni {
                Some(value) => std::env::set_var("OMNI_CLAUDE_BASE_URL", value),
                None => std::env::remove_var("OMNI_CLAUDE_BASE_URL"),
            }
            match old_legacy {
                Some(value) => std::env::set_var("ANTHROPIC_BASE_URL", value),
                None => std::env::remove_var("ANTHROPIC_BASE_URL"),
            }
        }
        assert!(detected);
    }

    #[test]
    fn auto_cache_on_by_default_with_opt_out() {
        // WHY: `send`/`send_stream` serve ONLY the OpenAI-inbound door, which has
        // no Claude Code fingerprint contract, so caching is injected automatically
        // with no flag to remember. The single escape is OMNI_CLAUDE_NO_AUTO_CACHE.
        // This locks in: on by default, opt-out disables, non-truthy is ignored,
        // and the opt-out applies regardless of first-party vs custom gateway.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let old = std::env::var_os("OMNI_CLAUDE_NO_AUTO_CACHE");
        let first_party = ClaudeProvider::new().unwrap();
        let custom = ClaudeProvider::new_for_custom_gateway(
            default_profile(),
            "https://claude-proxy.example.com",
            Some("sk-ant-oat01-example".into()),
            vec![],
        )
        .unwrap();

        // 1. Unset -> ON by default (both provider kinds; the door, not the
        // backend, decides — a custom Claude gateway still has no fingerprint).
        unsafe { std::env::remove_var("OMNI_CLAUDE_NO_AUTO_CACHE") };
        assert!(
            !auto_cache_disabled(),
            "on by default when opt-out is unset"
        );
        assert!(
            first_party.supports_auto_cache(),
            "first-party caches by default"
        );
        assert!(
            custom.supports_auto_cache(),
            "custom gateway caches by default"
        );

        // 2. Opt-out truthy -> injection disabled everywhere.
        unsafe { std::env::set_var("OMNI_CLAUDE_NO_AUTO_CACHE", "1") };
        assert!(auto_cache_disabled(), "\"1\" opts out");
        assert!(
            !first_party.supports_auto_cache(),
            "opt-out disables first-party"
        );
        assert!(
            !custom.supports_auto_cache(),
            "opt-out disables custom gateway"
        );

        // 3. Non-truthy value is ignored -> stays ON (fail-open on typos).
        unsafe { std::env::set_var("OMNI_CLAUDE_NO_AUTO_CACHE", "maybe") };
        assert!(
            !auto_cache_disabled(),
            "non-truthy opt-out value is ignored"
        );

        unsafe {
            match old {
                Some(v) => std::env::set_var("OMNI_CLAUDE_NO_AUTO_CACHE", v),
                None => std::env::remove_var("OMNI_CLAUDE_NO_AUTO_CACHE"),
            }
        }
    }

    #[test]
    fn resolve_via_profile_still_claude_specific() {
        let p = ClaudeProvider::new().unwrap();
        let m = p.profile().resolve_model("sonnet").unwrap();
        assert_eq!(m.canonical, "claude-sonnet-4-6");
    }

    #[test]
    fn mapper_roundtrip_via_prepare_identity() {
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("test identity".into()),
            }],
            tools: Some(vec![omni_core::CanonicalTool {
                name: "do_x".into(),
                description: Some("do the x".into()),
                parameters: serde_json::json!({"type":"object"}),
            }]),
            tool_choice: Some(omni_core::CanonicalToolChoice::Auto),
            ..Default::default()
        };
        let profile = default_profile();
        let repl = Replacements::empty();
        let anth = translate::prepare_anthropic_request(&req, profile, &repl, true, false).unwrap();
        // system has the two identity blocks
        let sys = anth.system.expect("system after identity");
        match sys {
            translate::SystemField::Blocks(blocks) => {
                assert!(blocks.len() >= 2);
                assert!(crate::fingerprint::is_claude_code_billing_header(
                    &blocks[0].text
                ));
                assert_eq!(blocks[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE);
            }
            _ => panic!("blocks form required for identity"),
        }
        assert_eq!(anth.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn from_response_maps_usage_and_finish() {
        let raw = serde_json::json!({
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [ {"type": "text", "text": "ok"} ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 3, "output_tokens": 1 }
        });
        let resp: translate::MessagesResponse = serde_json::from_value(raw).unwrap();
        let canon = translate::build_canonical_response(&resp, "haiku", &Replacements::empty());
        assert_eq!(canon.id.as_deref(), Some("msg_abc"));
        assert_eq!(canon.content, "ok");
        assert_eq!(canon.finish_reason.as_deref(), Some("stop"));
        assert_eq!(canon.usage.input_tokens, 3);
        assert_eq!(canon.usage.output_tokens, 1);
    }

    // --- additional comprehensive tests for claude (from port) covering shared, parity, mocked send ---

    fn sample_req(text: &str) -> CanonicalRequest {
        CanonicalRequest {
            model: "claude-sonnet-4-5".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text(text.into()),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn stream_converter_maps_anthropic_events_to_canonical_sequence() {
        // WHY: send_stream's correctness reduces to this converter. The HTTP path
        // just feeds it upstream StreamEvents. This pins the contract the binary
        // SSE framing (F8) depends on: text deltas pass through in order; a
        // tool-use block start opens a ToolCallDelta (id+name) and its
        // input_json deltas append arguments under the SAME canonical index;
        // usage + a mapped finish_reason terminate the stream exactly once.
        use crate::upstream::stream::{BlockDelta, BlockStart, StreamEvent};
        let mut conv = ClaudeStreamConverter::default();
        let mut out: Vec<CanonicalStreamEvent> = Vec::new();

        out.extend(conv.on_event(StreamEvent::MessageStart {
            id: "msg_1".into(),
            model: "claude-haiku-4-5-20251001".into(),
            input_tokens: Some(11),
            output_tokens: Some(0),
        }));
        out.extend(conv.on_event(StreamEvent::ContentBlockStart {
            index: 0,
            block: BlockStart::Text,
        }));
        out.extend(conv.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::Text("Hello".into()),
        }));
        out.extend(conv.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::Text(" world".into()),
        }));
        // Tool-use block at Anthropic index 1 -> canonical tool index 0.
        out.extend(conv.on_event(StreamEvent::ContentBlockStart {
            index: 1,
            block: BlockStart::ToolUse {
                id: "toolu_9".into(),
                name: "get_weather".into(),
            },
        }));
        out.extend(conv.on_event(StreamEvent::ContentBlockDelta {
            index: 1,
            delta: BlockDelta::InputJson("{\"city\":".into()),
        }));
        out.extend(conv.on_event(StreamEvent::ContentBlockDelta {
            index: 1,
            delta: BlockDelta::InputJson("\"SF\"}".into()),
        }));
        out.extend(conv.on_event(StreamEvent::MessageDelta {
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            output_tokens: Some(7),
        }));
        out.extend(conv.on_event(StreamEvent::MessageStop));

        assert_eq!(
            out[0],
            CanonicalStreamEvent::ResponseMetadata(omni_core::CanonicalResponseMetadata {
                id: Some("msg_1".into()),
                ..Default::default()
            })
        );
        // Text deltas, in order.
        assert_eq!(out[1], CanonicalStreamEvent::TextDelta("Hello".into()));
        assert_eq!(out[2], CanonicalStreamEvent::TextDelta(" world".into()));
        // Tool-call open carries id+name at canonical index 0.
        assert_eq!(
            out[3],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("toolu_9".into()),
                name: Some("get_weather".into()),
                arguments_delta: String::new(),
            }
        );
        // Argument fragments append under the SAME index, id/name now None.
        assert_eq!(
            out[4],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "{\"city\":".into(),
            }
        );
        assert_eq!(
            out[5],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "\"SF\"}".into(),
            }
        );
        // Usage then terminal Finish with mapped reason (tool_use -> tool_calls).
        assert_eq!(
            out[6],
            CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 11,
                output_tokens: 7,
                ..Default::default()
            })
        );
        assert_eq!(
            out[7],
            CanonicalStreamEvent::Finish {
                finish_reason: Some("tool_calls".into())
            }
        );
        assert_eq!(out.len(), 8, "no extra events emitted");
    }

    #[test]
    fn stream_converter_error_event_marks_finished_for_single_finish() {
        // WHY: an Anthropic `error` SSE frame emits a terminal Finish directly
        // (not via finish_events). If `finished` were not set here, a subsequent
        // end-of-byte-stream would make send_stream's EOF guard append a SECOND,
        // contradictory success Finish. This pins that the error path is also a
        // single-Finish terminal, matching the message_stop path.
        use crate::upstream::stream::StreamEvent;
        let mut conv = ClaudeStreamConverter::default();
        let out = conv.on_event(StreamEvent::Error {
            kind: "overloaded_error".into(),
            message: "upstream busy".into(),
        });
        assert_eq!(out.len(), 1);
        assert!(
            matches!(out[0], CanonicalStreamEvent::Finish { .. }),
            "error event yields a terminal Finish"
        );
        assert!(
            conv.finished,
            "error path must mark finished so the EOF guard does not double-Finish"
        );
        // The EOF guard would now be a no-op: finish_events is only re-run when
        // !finished, so no second Finish is appended.
    }

    #[test]
    fn stream_converter_maps_stop_reasons_to_oai_vocabulary() {
        // WHY: the canonical finish_reason must match the non-stream path's
        // mapping so OAI clients see consistent values regardless of stream mode.
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("tool_use"), "tool_calls");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
        assert_eq!(map_stop_reason("weird_future"), "weird_future");
    }

    #[test]
    fn claude_additional_ctors_and_shared_repl() {
        let p = ClaudeProvider::new().expect("ctor");
        assert_eq!(p.id(), "claude");
        let repl =
            Replacements::parse(r#"rule = [ { scope = "prompt", search = "x", replace = "y" } ]"#)
                .unwrap();
        assert_eq!(repl.apply_prompt("x"), "y");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_send_exercises_full_fingerprint_path() {
        // Live test: exercises the complete path end-to-end against the real
        // Anthropic upstream (canonical -> translate -> identity/cch finalize ->
        // headers + body -> real call -> from_anth -> canonical).
        //
        // Guarded so `cargo test` stays hermetic and green even on credentialed
        // developer machines. Credential presence alone is not enough: set
        // OMNI_LIVE_TESTS=1 to spend provider quota. The byte-exact wire pins
        // (fingerprint, cch, translate) are asserted by offline unit tests above;
        // this test only proves live wiring when explicitly requested.
        //
        // Holds CREDS_ENV_LOCK across the gate + send so it cannot race a hermetic
        // wiremock test that points CLAUDE_CREDENTIALS_PATH at a temp creds file
        // (which would otherwise change what `default_path()` resolves to mid-call).
        // Mirrors the Grok live test's CRED_ENV_LOCK discipline.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        if !omni_common::test_support::live_tests_enabled() {
            eprintln!(
                "skipping claude_send_exercises_full_fingerprint_path: set OMNI_LIVE_TESTS=1"
            );
            return;
        }
        // Also skip when CLAUDE_CREDENTIALS_PATH is set: that means a hermetic test
        // (or an operator) is pointing the loader at a throwaway dummy creds file,
        // which must not be trusted for a real network call. Mirrors Grok's
        // live_grok_key() bailing when XAI_CREDENTIALS_PATH is set; this prevents a
        // live api.anthropic.com call with a dummy bearer if env cleanup ever failed.
        if std::env::var_os("CLAUDE_CREDENTIALS_PATH").is_some() {
            eprintln!(
                "skipping claude_send_exercises_full_fingerprint_path: CLAUDE_CREDENTIALS_PATH override is set"
            );
            return;
        }
        if !crate::credentials::Credentials::default_path().exists() {
            eprintln!(
                "skipping claude_send_exercises_full_fingerprint_path: no Claude credentials at {}",
                crate::credentials::Credentials::default_path().display()
            );
            return;
        }
        let p = ClaudeProvider::new().expect("ctor");
        let resp = p
            .send(sample_req("port test"))
            .await
            .expect("claude provider send must succeed with creds");
        assert!(
            resp.model.contains("sonnet"),
            "resolved sonnet alias should yield sonnet-family model, got {}",
            resp.model
        );
        assert!(!resp.content.is_empty() || resp.tool_calls.is_empty()); // basic shape
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_auto_cache_creates_then_reads_across_turns_live() {
        // Live test (#1 prefix-stability + #2 above-floor engagement): proves the
        // OpenAI-inbound auto-cache marker ACTUALLY caches against the real
        // Anthropic upstream, not just that we emit the right bytes offline.
        //
        // WHY this is the only test that can catch the feature silently no-op'ing:
        // the gateway injects a billing suffix (Sha256 of message[0]) and a
        // version-pinned identity preamble into the cacheable prefix. If either
        // varied byte-for-byte between turns, every turn would MISS and we'd pay
        // full input price with caching "on". Offline tests pin the wire bytes but
        // cannot observe whether Anthropic's server-side content hash matched across
        // two real turns. This does.
        //
        // Construction that makes the assertion meaningful:
        //   - Turn 2's prefix is a strict superset of Turn 1's: same big user
        //     message, then an assistant reply, then a new user message. So Turn 1's
        //     exact prefix bytes are a prefix of Turn 2's request. A cache READ on
        //     Turn 2 is therefore only possible if that shared prefix (including our
        //     injected billing + preamble) was byte-identical across the two builds.
        //   - The shared message is padded far past the model floor (Sonnet = 1024
        //     tokens). Below the floor Anthropic silently declines to cache and
        //     cache_creation would be 0 -> so requiring cache_creation > 0 on Turn 1
        //     simultaneously proves we cleared the floor (#2) and that the marker is
        //     live. Padding is deterministic (fixed string), so no clock/nonce leaks
        //     into the prefix.
        //
        // Guarded exactly like claude_send_exercises_full_fingerprint_path: opt-in
        // via OMNI_LIVE_TESTS=1, real creds required, CLAUDE_CREDENTIALS_PATH
        // override forces a skip (a dummy creds file must never drive a real call),
        // and CREDS_ENV_LOCK is held across gate+sends so a concurrent hermetic test
        // cannot repoint default_path() mid-call.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        if !omni_common::test_support::live_tests_enabled() {
            eprintln!(
                "skipping claude_auto_cache_creates_then_reads_across_turns_live: set OMNI_LIVE_TESTS=1"
            );
            return;
        }
        if std::env::var_os("CLAUDE_CREDENTIALS_PATH").is_some() {
            eprintln!(
                "skipping claude_auto_cache_creates_then_reads_across_turns_live: CLAUDE_CREDENTIALS_PATH override is set"
            );
            return;
        }
        if !crate::credentials::Credentials::default_path().exists() {
            eprintln!(
                "skipping claude_auto_cache_creates_then_reads_across_turns_live: no Claude credentials at {}",
                crate::credentials::Credentials::default_path().display()
            );
            return;
        }
        // Also require auto-cache to be ON (its default). If a developer exported the
        // opt-out in their shell, this test cannot observe caching -> skip loudly
        // rather than fail spuriously.
        if auto_cache_disabled() {
            eprintln!(
                "skipping claude_auto_cache_creates_then_reads_across_turns_live: OMNI_CLAUDE_NO_AUTO_CACHE is set (auto-cache off)"
            );
            return;
        }

        // Deterministic padding well past the 1024-token Sonnet floor. ~4 chars/token
        // => ~12k chars targets ~3k tokens, comfortably clearing the floor with
        // margin for tokenizer variance. Fixed content: identical bytes every turn.
        let big_context = format!(
            "You are reviewing the following reference material. \
             Answer only from it. Reference block repeated for length:\n{}",
            "The quick brown fox jumps over the lazy dog near the riverbank. ".repeat(200)
        );

        let p = ClaudeProvider::new().expect("ctor");

        // ---- Turn 1: single user message carrying the big shared context. ----
        let turn1 = CanonicalRequest {
            model: "claude-sonnet-4-5".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text(big_context.clone()),
            }],
            ..Default::default()
        };
        let r1 = p
            .send(turn1)
            .await
            .expect("turn 1 send must succeed with creds");

        // Turn 1 must CREATE cache. Non-zero creation proves two things at once:
        // the prefix cleared the model floor (#2), and the top-level marker is live.
        assert!(
            r1.usage.cache_creation > 0,
            "turn 1 must create cache (proves marker live + prefix above floor); got creation={} read={}",
            r1.usage.cache_creation,
            r1.usage.cache_read
        );

        // ---- Turn 2: same big context, plus the assistant reply, plus a follow-up.
        // Turn 1's exact prefix bytes are a prefix of this request. ----
        let assistant_reply = if r1.content.is_empty() {
            "Understood.".to_string()
        } else {
            r1.content.clone()
        };
        let turn2 = CanonicalRequest {
            model: "claude-sonnet-4-5".into(),
            messages: vec![
                CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text(big_context.clone()),
                },
                CanonicalMessage {
                    role: "assistant".into(),
                    content: CanonicalContent::Text(assistant_reply),
                },
                CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("Briefly, what did I ask you to do?".into()),
                },
            ],
            ..Default::default()
        };
        let r2 = p
            .send(turn2)
            .await
            .expect("turn 2 send must succeed with creds");

        // THE assertion (#1): Turn 2 reads the cache Turn 1 wrote. This is only
        // possible if the shared prefix - INCLUDING our injected billing suffix and
        // identity preamble - was byte-identical across both builds. A wobble in
        // either would drop cache_read to 0 and fail here.
        assert!(
            r2.usage.cache_read > 0,
            "turn 2 must READ cache written by turn 1 (proves prefix byte-stable across turns); \
             got read={} creation={}. read==0 means the injected prefix (billing/preamble) drifted.",
            r2.usage.cache_read,
            r2.usage.cache_creation
        );

        // Cross-check the hit is substantive, not a degenerate few-token match: the
        // bytes cached on turn 1 should be reflected in turn 2's read. Allow slack
        // for Anthropic block-boundary rounding, but require the read to be within a
        // sane fraction of what turn 1 created rather than a trivial prefix.
        let created = r1.usage.cache_creation;
        assert!(
            r2.usage.cache_read >= created / 2,
            "turn 2 cache_read ({}) should cover most of turn 1's cached prefix ({}); a tiny read \
             suggests only a fragment matched, i.e. partial prefix instability",
            r2.usage.cache_read,
            created
        );
    }

    #[test]
    fn claude_parity_note() {
        // Full parity with grok via common trait + core canonical + replacements is verified in:
        // omni-core::tests, provider-grok::tests, omni router tests, and common replacements/stats units.
    }

    #[test]
    fn new_for_test_allows_other_profiles() {
        let p154 =
            ClaudeProvider::new_for_test(crate::fingerprint::resolve_profile("2.1.154").unwrap());
        assert_eq!(p154.profile().name, "cc-2.1.154-sdk-cli");
    }

    #[test]
    fn identity_invariants_cch_billing_system0_and_preamble_exact() {
        let req = sample_req("inv");
        let profile = default_profile();
        let repl = Replacements::empty();
        let anth = translate::prepare_anthropic_request(&req, profile, &repl, true, false).unwrap();
        let sys = match anth.system.expect("sys") {
            translate::SystemField::Blocks(b) => b,
            _ => panic!(),
        };
        assert!(sys.len() >= 2);
        assert!(crate::fingerprint::is_claude_code_billing_header(
            &sys[0].text
        ));
        assert_eq!(sys[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE);
    }

    #[test]
    fn prompt_repl_before_identity_gate_for_billing() {
        let repl =
            Replacements::parse(r#"rule = [ { scope = "prompt", search = "Q", replace = "Z" } ]"#)
                .unwrap();
        let mut req = sample_req("ask Q");
        // force model that hits haiku path etc
        req.model = "haiku".into();
        let profile = default_profile();
        let anth = translate::prepare_anthropic_request(&req, profile, &repl, true, false).unwrap();
        let blocks = match anth.system.unwrap() {
            translate::SystemField::Blocks(b) => b,
            _ => panic!(),
        };
        // first user post-repl is used for suffix
        assert!(blocks[0].text.contains("cc_version="));
        match &anth.messages[0].content {
            translate::MessageContent::Text(t) => assert_eq!(t, "ask Z"),
            _ => panic!(),
        }
    }

    #[test]
    fn model_resolution_all_cases_via_provider_profile() {
        let p = ClaudeProvider::new().unwrap();
        let prof = p.profile();
        // canonical -> resolves
        assert_eq!(
            prof.resolve_model("claude-sonnet-4-6").unwrap().canonical,
            "claude-sonnet-4-6"
        );
        // alias -> resolves
        assert_eq!(
            prof.resolve_model("sonnet").unwrap().canonical,
            "claude-sonnet-4-6"
        );
        // substring -> NO LONGER resolves (matcher deleted): passes through raw
        assert!(prof.resolve_model("foo-sonnet-bar").is_none());
        // cli alias -> resolves
        assert_eq!(
            prof.resolve_model("opus").unwrap().canonical,
            "claude-opus-4-8"
        );
        // unknown -> NO LONGER defaults: passes through raw
        assert!(prof.resolve_model("weird").is_none());
        // exact dated canonical -> resolves
        assert_eq!(
            prof.resolve_model("claude-haiku-4-5-20251001")
                .unwrap()
                .canonical,
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn prepare_injects_identity_even_for_no_preamble_caller_paths() {
        // The provider always passes true for gate; other paths (e.g. --no-preamble
        // debug) use false. Here assert the seam.
        let req = sample_req("x");
        let profile = default_profile();
        let repl = Replacements::empty();
        let with_id =
            translate::prepare_anthropic_request(&req, profile, &repl, true, false).unwrap();
        let without =
            translate::prepare_anthropic_request(&req, profile, &repl, false, false).unwrap();
        assert!(with_id.system.is_some());
        assert!(without.system.is_none());
    }

    #[test]
    fn build_canonical_response_usage_from_raw_includes_cache() {
        let raw = serde_json::json!({
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [ {"type": "text", "text": "u"} ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 7, "output_tokens": 2, "cache_creation_input_tokens": 1, "cache_read_input_tokens": 3 }
        });
        let resp: translate::MessagesResponse = serde_json::from_value(raw).unwrap();
        let canon = translate::build_canonical_response(&resp, "haiku", &Replacements::empty());
        assert_eq!(canon.id.as_deref(), Some("m"));
        assert_eq!(canon.usage.input_tokens, 7);
        assert_eq!(canon.usage.cache_creation, 1);
        assert_eq!(canon.usage.cache_read, 3);
    }

    // ── Hermetic wiremock round-trip tests ────────────────────────────────────
    //
    // These close the CI coverage gap left by the live test above: they prove the
    // full request-build -> HTTP -> response-parse round-trip OFFLINE, against a
    // local mock standing in for api.anthropic.com. The provider is pointed at the
    // mock via the production base-URL seam (`new_for_test_with_base`); everything
    // else (the rustls + http2_prior_knowledge client, the fingerprint headers, the
    // body finalize, the typed response/stream decoders) is the real production
    // path. wiremock serves cleartext HTTP/2, so the production transport config
    // reaches it unchanged.
    //
    // Both the gate AND the env mutation are serialized by CREDS_ENV_LOCK so they
    // cannot race the live test (which also takes it) or each other.

    use std::sync::Mutex as StdMutex;
    use wiremock::matchers::{body_partial_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Serializes every test that reads or mutates `CLAUDE_CREDENTIALS_PATH`
    /// (the hermetic round-trip tests and the live `claude_send_*` test), so the
    /// env override one test sets can never be observed by another mid-call.
    static CREDS_ENV_LOCK: StdMutex<()> = StdMutex::new(());

    /// RAII guard: writes a temp creds file, points `CLAUDE_CREDENTIALS_PATH` at
    /// it, and restores the prior env value + removes the file on drop. The token
    /// is a non-empty dummy (the only gate `from_bytes` enforces) with a far-future
    /// `expiresAt` for realism. The caller must hold CREDS_ENV_LOCK for the guard's
    /// whole lifetime so the env mutation is not observed by a concurrent test.
    struct TempCreds {
        path: std::path::PathBuf,
        /// Prior value as an OsString so a non-UTF-8 prior path is restored exactly
        /// (var() would treat it as absent and wrongly remove the var on drop).
        prev: Option<std::ffi::OsString>,
    }

    impl TempCreds {
        fn dummy_token() -> &'static str {
            "sk-ant-oat01-dummy"
        }

        fn install(tag: &str) -> Self {
            // Far-future expiry (year ~2065) so any expiry-aware code sees a live
            // token; the send path does not actually gate on it, but realism costs
            // nothing here.
            let body = format!(
                r#"{{"claudeAiOauth":{{"accessToken":"{}","expiresAt":3000000000000,"subscriptionType":"max"}}}}"#,
                Self::dummy_token()
            );
            let path = std::env::temp_dir().join(format!(
                "claude-creds-{}-{}.json",
                tag,
                std::process::id()
            ));
            std::fs::write(&path, body).expect("write temp claude creds");
            let prev = std::env::var_os("CLAUDE_CREDENTIALS_PATH");
            // SAFETY (edition 2024): single-threaded mutation while holding
            // CREDS_ENV_LOCK; no other thread reads the env concurrently.
            unsafe {
                std::env::set_var("CLAUDE_CREDENTIALS_PATH", &path);
            }
            Self { path, prev }
        }
    }

    impl Drop for TempCreds {
        fn drop(&mut self) {
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var("CLAUDE_CREDENTIALS_PATH", v),
                    None => std::env::remove_var("CLAUDE_CREDENTIALS_PATH"),
                }
            }
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// The minimal-but-complete Anthropic non-stream response body. Includes every
    /// field `MessagesResponse` requires (id/type/role/model/content/usage); the
    /// caller overrides `content` + `stop_reason` for the tool-call variant.
    fn anth_text_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "msg_hermetic_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [ {"type": "text", "text": "Hello"} ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 10, "output_tokens": 2 }
        })
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_nonstream_success_roundtrip_via_wiremock() {
        // WHY: proves a successful non-stream completion end to end offline - the
        // request leaves with the right bearer + anthropic-version + path/query, and
        // the real response decoder maps the Anthropic body to canonical (content,
        // end_turn -> stop, usage). This is exactly what CI could not prove before:
        // a green round-trip with no network and no real creds.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempCreds::install("nonstream-ok");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header(
                "authorization",
                format!("Bearer {}", TempCreds::dummy_token()).as_str(),
            ))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anth_text_response_json()))
            .expect(1)
            .mount(&server)
            .await;

        let p = ClaudeProvider::new_for_test_with_base(default_profile(), server.uri())
            .expect("test provider against mock base");
        let resp = p
            .send(sample_req("hi"))
            .await
            .expect("hermetic claude send must succeed");

        assert_eq!(resp.content, "Hello");
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 2);
        assert!(resp.tool_calls.is_empty());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_send_non_text_system_block_is_client_bad_request() {
        // WHY (issue #2 + Rule 9): a non-text block in a system message is a
        // client-input fault. The translate-level test only proves the builder
        // returns Err; this proves the CLIENT-FACING outcome: `send` must surface
        // ProviderError::BadRequest (-> HTTP 400), NOT Other/Upstream (-> 500/502).
        // It would fail if prepare errors were mapped back through Other.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempCreds::install("non-text-system");

        let server = MockServer::start().await;
        // No mock expectation: validation must fail BEFORE any upstream call.
        let p = ClaudeProvider::new_for_test_with_base(default_profile(), server.uri())
            .expect("test provider against mock base");

        let req = CanonicalRequest {
            model: "claude-sonnet-4-5".into(),
            messages: vec![
                CanonicalMessage {
                    role: "system".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::Image {
                        source: omni_core::CanonicalImageSource::Url {
                            url: "https://example.com/x.png".into(),
                        },
                    }]),
                },
                CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                },
            ],
            ..Default::default()
        };

        let err = p.send(req).await.expect_err("non-text system must reject");
        assert!(
            matches!(err, ProviderError::BadRequest(_)),
            "non-text system block must surface as a client BadRequest (400), got {err:?}"
        );
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            0,
            "validation must fail before any upstream request"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_custom_gateway_without_auth_does_not_use_default_oauth_token() {
        // WHY: custom Anthropic-compatible gateways can be arbitrary URLs. A
        // local Claude Code OAuth token must not be sent there unless custom
        // gateway auth explicitly says so, and a no-auth gateway must not need
        // the local OAuth file to exist.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let creds = TempCreds::install("custom-no-auth");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anth_text_response_json()))
            .expect(1)
            .mount(&server)
            .await;

        let p =
            ClaudeProvider::new_for_custom_gateway(default_profile(), server.uri(), None, vec![])
                .expect("custom gateway provider");
        drop(creds);
        let resp = p
            .send(sample_req("custom no auth"))
            .await
            .expect("custom gateway send");

        assert_eq!(resp.content, "Hello");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            !requests[0].headers.contains_key("authorization"),
            "custom Claude gateway no-auth must not receive Claude OAuth Authorization"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_custom_gateway_auth_overrides_default_oauth_token() {
        // WHY: explicit gateway auth owns the upstream Authorization header and
        // must replace the Claude Code bearer read from CLAUDE_CREDENTIALS_PATH.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempCreds::install("custom-auth");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header("authorization", "Bearer custom-claude-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anth_text_response_json()))
            .expect(1)
            .mount(&server)
            .await;

        let p = ClaudeProvider::new_for_custom_gateway(
            default_profile(),
            server.uri(),
            Some("custom-claude-token".into()),
            vec![],
        )
        .expect("custom gateway provider");
        let resp = p
            .send(sample_req("custom auth"))
            .await
            .expect("custom gateway send");

        assert_eq!(resp.content, "Hello");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_custom_gateway_x_api_key_header_does_not_need_oauth_file() {
        // WHY: Claude Code gateways may use ANTHROPIC_API_KEY-style x-api-key
        // auth. Custom headers must be enough, without falling back to the
        // subscription OAuth file.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let creds = TempCreds::install("custom-api-key");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header("x-api-key", "custom-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anth_text_response_json()))
            .expect(1)
            .mount(&server)
            .await;

        let p = ClaudeProvider::new_for_custom_gateway(
            default_profile(),
            server.uri(),
            None,
            vec![("x-api-key".into(), "custom-api-key".into())],
        )
        .expect("custom gateway provider");
        drop(creds);
        let resp = p
            .send(sample_req("custom x-api-key"))
            .await
            .expect("custom gateway send");

        assert_eq!(resp.content, "Hello");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_nonstream_tool_call_roundtrip_via_wiremock() {
        // WHY: tool calls take a different decode branch (tool_use content block ->
        // canonical tool_calls, stop_reason tool_use -> tool_calls). Pins that the
        // wire round-trip surfaces the tool name, id, and JSON arguments intact.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempCreds::install("nonstream-tool");

        let body = serde_json::json!({
            "id": "msg_hermetic_tool",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "SF"}}
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": { "input_tokens": 12, "output_tokens": 5 }
        });

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header(
                "authorization",
                format!("Bearer {}", TempCreds::dummy_token()).as_str(),
            ))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .expect(1)
            .mount(&server)
            .await;

        let p = ClaudeProvider::new_for_test_with_base(default_profile(), server.uri())
            .expect("test provider against mock base");
        let resp = p
            .send(sample_req("weather?"))
            .await
            .expect("hermetic claude tool-call send must succeed");

        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(resp.tool_calls.len(), 1);
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.id, "toolu_abc");
        assert_eq!(tc.name, "get_weather");
        // Arguments are the serialized tool input object.
        let args: serde_json::Value = serde_json::from_str(&tc.arguments).expect("args are json");
        assert_eq!(args["city"], "SF");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_streaming_roundtrip_via_wiremock_emits_single_finish() {
        // WHY: proves the streaming round-trip offline AND pins the duplicate-Finish
        // fix. The mock emits a normal completed Anthropic SSE sequence ending in
        // message_stop; the converter emits its terminal Finish there, and the
        // send_stream EOF guard must NOT append a second one. Asserting exactly one
        // Finish is the regression test for that bug.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempCreds::install("stream-ok");

        // data: {json}\n\n frames; the decoder keys on the JSON `type` and ignores
        // the `event:` line. message_start carries the nested `message` object with
        // usage; message_delta carries stop_reason + output_tokens; message_stop
        // closes. Frames are concatenated into one SSE body.
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header(
                "authorization",
                format!("Bearer {}", TempCreds::dummy_token()).as_str(),
            ))
            // send_stream flips the wire body to stream:true; pin it so a regression
            // that stopped requesting a stream can't pass against an SSE-only mock.
            .and(body_partial_json(serde_json::json!({"stream": true})))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .expect(1)
            .mount(&server)
            .await;

        let p = ClaudeProvider::new_for_test_with_base(default_profile(), server.uri())
            .expect("test provider against mock base");
        let stream = p
            .send_stream(sample_req("hi"))
            .await
            .expect("hermetic claude stream must open");
        let events: Vec<CanonicalStreamEvent> =
            stream.map(|r| r.expect("no stream error")).collect().await;

        assert_eq!(
            events[0],
            CanonicalStreamEvent::ResponseMetadata(omni_core::CanonicalResponseMetadata {
                id: Some("msg_s".into()),
                ..Default::default()
            })
        );
        assert_eq!(events[1], CanonicalStreamEvent::TextDelta("Hello".into()));
        // Usage carries the input tokens from message_start and output from message_delta.
        assert_eq!(
            events[2],
            CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 10,
                output_tokens: 3,
                ..Default::default()
            })
        );
        // Exactly ONE terminal Finish (the dup-Finish fix). end_turn -> stop.
        assert_eq!(
            events[3],
            CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into())
            }
        );
        let finishes = events
            .iter()
            .filter(|e| matches!(e, CanonicalStreamEvent::Finish { .. }))
            .count();
        assert_eq!(finishes, 1, "exactly one terminal Finish, got {finishes}");
        assert_eq!(events.len(), 4, "no extra events: {events:?}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_streaming_eof_without_message_stop_emits_single_finish() {
        // WHY: pins the OTHER branch of the send_stream tail guard. When the
        // upstream byte stream closes WITHOUT a message_stop frame (a truncated or
        // abruptly-closed stream), the converter never ran finish_events, so
        // `finished` is false and the EOF guard must synthesize EXACTLY ONE
        // terminal Finish. The happy-path test above pins the message_stop branch;
        // this pins that a regression breaking only the EOF guard (e.g. dropping
        // the synthesized Finish, or re-introducing a double-emit) would be caught.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempCreds::install("stream-eof");

        // Ends after message_delta - NO message_stop frame.
        let sse_body = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_e\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        );

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header(
                "authorization",
                format!("Bearer {}", TempCreds::dummy_token()).as_str(),
            ))
            // send_stream flips the wire body to stream:true; pin it so a regression
            // that stopped requesting a stream can't pass against an SSE-only mock.
            .and(body_partial_json(serde_json::json!({"stream": true})))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .expect(1)
            .mount(&server)
            .await;

        let p = ClaudeProvider::new_for_test_with_base(default_profile(), server.uri())
            .expect("test provider against mock base");
        let stream = p
            .send_stream(sample_req("hi"))
            .await
            .expect("hermetic claude stream must open");
        let events: Vec<CanonicalStreamEvent> =
            stream.map(|r| r.expect("no stream error")).collect().await;

        assert_eq!(
            events[0],
            CanonicalStreamEvent::ResponseMetadata(omni_core::CanonicalResponseMetadata {
                id: Some("msg_e".into()),
                ..Default::default()
            })
        );
        assert_eq!(events[1], CanonicalStreamEvent::TextDelta("Hi".into()));
        // EOF guard synthesizes Usage (tokens were seen) + exactly one Finish.
        let finishes = events
            .iter()
            .filter(|e| matches!(e, CanonicalStreamEvent::Finish { .. }))
            .count();
        assert_eq!(
            finishes, 1,
            "EOF-without-message_stop must still yield exactly one Finish, got {finishes}: {events:?}"
        );
        // The mapped stop_reason from message_delta is carried into the Finish.
        assert_eq!(
            events.last(),
            Some(&CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into())
            })
        );
    }

    #[test]
    fn redactor_scrubs_exact_bearer_and_prefixed_tokens_leaves_text() {
        // WHY: the client-facing error.message must never echo the upstream
        // credential. This pins both redaction layers: the EXACT captured bearer
        // (a token with no recognizable prefix is still scrubbed) AND the prefix
        // scrubber for sk-ant-oat01-.../eyJ... tokens that appear in an error body
        // but were NOT the captured creds, while ordinary prose survives intact.
        let creds = credentials::Credentials {
            // Deliberately prefix-less so only the exact-secret layer can catch it.
            access_token: "plainsecret123".into(),
            expires_at_ms: None,
            subscription_type: None,
        };
        let r = ClaudeErrorRedactor::for_credentials(&creds);

        // A fake Anthropic error body echoing the exact bearer plus other secrets.
        let body = "anthropic 401: {\"error\":{\"message\":\"invalid Bearer plainsecret123 \
            and key sk-ant-oat01-leak999, jwt eyJhbGciOiJI.payload.sig\"}}";
        let out = r.redact(body);

        assert!(
            !out.contains("plainsecret123"),
            "exact bearer token must be scrubbed: {out}"
        );
        assert!(
            !out.contains("sk-ant-oat01-leak999"),
            "sk-ant-oat01- prefixed token must be prefix-scrubbed: {out}"
        );
        assert!(
            !out.contains("eyJhbGciOiJI"),
            "eyJ JWT must be prefix-scrubbed: {out}"
        );
        assert!(
            out.contains("<redacted>"),
            "redaction marker present: {out}"
        );
        // Ordinary words around the secrets survive.
        assert!(out.contains("invalid"), "prose preserved: {out}");
        assert!(out.contains("anthropic 401"), "prose preserved: {out}");
    }

    #[test]
    fn redactor_skips_custom_gateway_placeholder_and_empty() {
        // WHY: the custom-gateway sentinel and empty tokens are not real secrets;
        // capturing them would replace those harmless substrings everywhere. The
        // default redactor (empty secrets) still prefix-scrubs.
        let placeholder = credentials::Credentials::placeholder_for_custom_gateway();
        let r = ClaudeErrorRedactor::for_credentials(&placeholder);
        assert!(r.secrets.is_empty(), "placeholder must not be captured");
        // Prefix scrub still runs.
        assert!(r.redact("key sk-leak end").contains("<redacted>"));
        assert!(!r.redact("key sk-leak end").contains("sk-leak"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn claude_error_body_is_redacted_before_surfacing() {
        // WHY (the fix): Anthropic's raw error body flows into the client-facing
        // ProviderError message. It can echo the bearer the provider just sent.
        // This proves end to end that the surfaced message scrubs the dummy token
        // (which TempCreds makes the provider send) and replaces it with
        // <redacted>, with no network and no real creds.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempCreds::install("error-redact");

        // 401 body deliberately echoes the exact bearer the provider sends.
        let leaky_body = format!(
            "{{\"type\":\"error\",\"error\":{{\"type\":\"authentication_error\",\"message\":\"invalid token Bearer {tok} ({tok})\"}}}}",
            tok = TempCreds::dummy_token()
        );

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_string(leaky_body))
            .mount(&server)
            .await;

        let p = ClaudeProvider::new_for_test_with_base(default_profile(), server.uri())
            .expect("test provider against mock base");
        let err = p
            .send(sample_req("hi"))
            .await
            .expect_err("401 must surface as an error");

        let msg = err.to_string();
        assert!(
            !msg.contains(TempCreds::dummy_token()),
            "surfaced error must not leak the bearer token: {msg}"
        );
        assert!(
            msg.contains("<redacted>"),
            "surfaced error must show the redaction marker: {msg}"
        );
    }

    #[test]
    fn stream_converter_surfaces_thinking_deltas() {
        // WHY: Claude streaming thinking data should be preserved in canonical
        // reasoning events rather than silently dropped.
        let mut conv = ClaudeStreamConverter::default();
        let out = conv.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::Thinking("think".into()),
        });
        assert_eq!(
            out,
            vec![CanonicalStreamEvent::ReasoningDelta("think".into())]
        );

        let out = conv.on_event(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: BlockDelta::ThinkingSignature("sig".into()),
        });
        assert_eq!(
            out,
            vec![CanonicalStreamEvent::ReasoningSignatureDelta("sig".into())]
        );
    }
}
