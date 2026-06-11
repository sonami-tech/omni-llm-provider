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

pub mod credentials;
pub mod fingerprint;
pub mod models;
pub mod translate;
pub mod upstream;

pub use fingerprint::{
    CLAUDE_CODE_SYSTEM_PREAMBLE, FingerprintProfile, RequestContext, RequestKind, default_profile,
    resolve_profile,
};
pub use upstream::{UpstreamClient, UpstreamError};

use async_trait::async_trait;
use futures_util::StreamExt;
use omni_common::Replacements;
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
                input_tokens,
                output_tokens,
                ..
            } => {
                if input_tokens.is_some() {
                    self.input_tokens = input_tokens;
                }
                if output_tokens.is_some() {
                    self.output_tokens = output_tokens;
                }
                vec![]
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
                // Thinking deltas are not surfaced in the canonical text stream today.
                _ => vec![],
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
}

impl Default for ClaudeProvider {
    fn default() -> Self {
        Self::new().expect("default claude profile must be usable at construction")
    }
}

fn map_upstream_err(e: UpstreamError) -> ProviderError {
    match e {
        UpstreamError::TokenExpired | UpstreamError::CredentialsMissingToken => {
            ProviderError::Auth(e.to_string())
        }
        UpstreamError::CredentialsRead(_) | UpstreamError::CredentialsParse(_) => {
            ProviderError::Auth(e.to_string())
        }
        UpstreamError::Anthropic { status, body, .. } => {
            ProviderError::Upstream(format!("anthropic {status}: {body}"))
        }
        UpstreamError::Transport(_) | UpstreamError::Decode(_) => {
            ProviderError::Upstream(e.to_string())
        }
    }
}

#[async_trait]
impl LlmProvider for ClaudeProvider {
    fn id(&self) -> &'static str {
        "claude-code"
    }

    async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError> {
        debug!(
            provider = "claude-code",
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
        )
        .map_err(|e| ProviderError::Other(anyhow::Error::msg(e)))?;

        // 3. Serialize for finalize (cch lives in the billing text inside system).
        let body_val = serde_json::to_value(&anth_req).map_err(|e| {
            ProviderError::Other(anyhow::Error::msg(format!("anth serialize: {e}")))
        })?;

        let ctx = RequestContext::new_reply().with_model(anth_req.model.clone());

        // Fresh creds read (the design: never cached in this process).
        let creds =
            credentials::Credentials::load_fresh_async(&credentials::Credentials::default_path())
                .await
                .map_err(map_upstream_err)?;

        // 4. Send (this does finalize_body_json which patches the 5-hex cch,
        //    builds the full header set with per-profile betas / stainless / ua,
        //    and does the 401-once refresh).
        let raw_resp = self
            .client
            .send_messages_json(&creds, &ctx, &body_val)
            .await
            .map_err(map_upstream_err)?;

        // 5. Parse into our response type (non-stream path).
        let anth_resp: translate::MessagesResponse = serde_json::from_value(raw_resp)
            .map_err(|e| ProviderError::Upstream(format!("decode anth response: {e}")))?;

        // 6. Map back to canonical + apply response-scope replacements hook.
        let canon = translate::build_canonical_response(&anth_resp, &req.model, &repl);

        debug!(model = %canon.model, finish = ?canon.finish_reason, "claude response mapped to canonical");

        Ok(canon)
    }

    async fn send_stream(&self, req: CanonicalRequest) -> Result<CanonicalStream, ProviderError> {
        debug!(
            provider = "claude-code",
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
        let anth_req = translate::prepare_anthropic_request(&req, self.profile, &repl, true)
            .map_err(|e| ProviderError::Other(anyhow::Error::msg(e)))?;
        let mut body_val = serde_json::to_value(&anth_req).map_err(|e| {
            ProviderError::Other(anyhow::Error::msg(format!("anth serialize: {e}")))
        })?;
        body_val["stream"] = serde_json::Value::Bool(true);

        let ctx = RequestContext::new_reply().with_model(anth_req.model.clone());

        let creds =
            credentials::Credentials::load_fresh_async(&credentials::Credentials::default_path())
                .await
                .map_err(map_upstream_err)?;

        // Open the upstream SSE stream (full retry / 401-refresh semantics live
        // in send_messages_stream). Yields typed Anthropic StreamEvents.
        let upstream = self
            .client
            .send_messages_stream(&creds, &ctx, &body_val)
            .await
            .map_err(map_upstream_err)?;

        // Map each Anthropic event to zero-or-more canonical events, flattening
        // into a single ordered canonical stream. The converter is stateful, so
        // it lives inside the async stream closure.
        let canonical = async_stream::try_stream! {
            let mut conv = ClaudeStreamConverter::default();
            futures_util::pin_mut!(upstream);
            while let Some(item) = upstream.next().await {
                let event = item.map_err(map_upstream_err)?;
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
    "claude-code"
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::{CanonicalContent, CanonicalMessage, /*CanonicalTool, CanonicalToolChoice*/};

    #[test]
    fn provider_id_and_construction() {
        assert_eq!(provider_id(), "claude-code");
        let p = ClaudeProvider::new().expect("default profile constructs");
        assert_eq!(p.id(), "claude-code");
        assert_eq!(p.profile().name, "cc-2.1.165-sdk-cli");
    }

    #[test]
    fn resolve_via_profile_still_claude_specific() {
        let p = ClaudeProvider::new().unwrap();
        let m = p.profile().resolve_model("sonnet");
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
        let anth = translate::prepare_anthropic_request(&req, profile, &repl, true).unwrap();
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

        // Text deltas, in order.
        assert_eq!(out[0], CanonicalStreamEvent::TextDelta("Hello".into()));
        assert_eq!(out[1], CanonicalStreamEvent::TextDelta(" world".into()));
        // Tool-call open carries id+name at canonical index 0.
        assert_eq!(
            out[2],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("toolu_9".into()),
                name: Some("get_weather".into()),
                arguments_delta: String::new(),
            }
        );
        // Argument fragments append under the SAME index, id/name now None.
        assert_eq!(
            out[3],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "{\"city\":".into(),
            }
        );
        assert_eq!(
            out[4],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "\"SF\"}".into(),
            }
        );
        // Usage then terminal Finish with mapped reason (tool_use -> tool_calls).
        assert_eq!(
            out[5],
            CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 11,
                output_tokens: 7,
                ..Default::default()
            })
        );
        assert_eq!(
            out[6],
            CanonicalStreamEvent::Finish {
                finish_reason: Some("tool_calls".into())
            }
        );
        assert_eq!(out.len(), 7, "no extra events emitted");
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
        assert_eq!(p.id(), "claude-code");
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
        // Guarded so `cargo test` stays hermetic and green offline: skips cleanly
        // when no Claude OAuth credentials are present, so it never burns Max quota
        // on every run and never fails in a creds-less CI. The byte-exact wire
        // pins (fingerprint, cch, translate) are asserted by the offline unit tests
        // above; this test only proves the live wiring when creds exist.
        //
        // Holds CREDS_ENV_LOCK across the gate + send so it cannot race a hermetic
        // wiremock test that points CLAUDE_CREDENTIALS_PATH at a temp creds file
        // (which would otherwise change what `default_path()` resolves to mid-call).
        // Mirrors the Grok live test's CRED_ENV_LOCK discipline.
        let _guard = CREDS_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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

    #[test]
    fn claude_parity_note() {
        // Full parity with grok via common trait + core canonical + replacements is verified in:
        // omni-core::tests, provider-grok::tests, bin wrapper tests, and common replacements/stats units.
    }

    #[test]
    fn new_for_test_allows_other_profiles() {
        let p154 =
            ClaudeProvider::new_for_test(crate::fingerprint::resolve_profile("2.1.154").unwrap());
        assert_eq!(p154.profile().name, "cc-2.1.154-sdk-cli");
        assert_eq!(p154.profile().default_model, "sonnet");
    }

    #[test]
    fn identity_invariants_cch_billing_system0_and_preamble_exact() {
        let req = sample_req("inv");
        let profile = default_profile();
        let repl = Replacements::empty();
        let anth = translate::prepare_anthropic_request(&req, profile, &repl, true).unwrap();
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
        let anth = translate::prepare_anthropic_request(&req, profile, &repl, true).unwrap();
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
        // canonical
        assert_eq!(
            prof.resolve_model("claude-sonnet-4-6").canonical,
            "claude-sonnet-4-6"
        );
        // alias
        assert_eq!(prof.resolve_model("sonnet").canonical, "claude-sonnet-4-6");
        // substring
        assert_eq!(
            prof.resolve_model("foo-sonnet-bar").canonical,
            "claude-sonnet-4-6"
        );
        // cli
        assert_eq!(prof.resolve_model("opus").canonical, "claude-opus-4-8");
        // default
        assert_eq!(prof.resolve_model("weird").canonical, "claude-opus-4-8");
        // haiku dated
        assert_eq!(
            prof.resolve_model("claude-haiku-4-5-20251001").canonical,
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
        let with_id = translate::prepare_anthropic_request(&req, profile, &repl, true).unwrap();
        let without = translate::prepare_anthropic_request(&req, profile, &repl, false).unwrap();
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
        // WHY: proves a successful non-stream completion end to end offline — the
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

        // Text delta first.
        assert_eq!(events[0], CanonicalStreamEvent::TextDelta("Hello".into()));
        // Usage carries the input tokens from message_start and output from message_delta.
        assert_eq!(
            events[1],
            CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 10,
                output_tokens: 3,
                ..Default::default()
            })
        );
        // Exactly ONE terminal Finish (the dup-Finish fix). end_turn -> stop.
        assert_eq!(
            events[2],
            CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into())
            }
        );
        let finishes = events
            .iter()
            .filter(|e| matches!(e, CanonicalStreamEvent::Finish { .. }))
            .count();
        assert_eq!(finishes, 1, "exactly one terminal Finish, got {finishes}");
        assert_eq!(events.len(), 3, "no extra events: {events:?}");
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

        // Ends after message_delta — NO message_stop frame.
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

        assert_eq!(events[0], CanonicalStreamEvent::TextDelta("Hi".into()));
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
}
