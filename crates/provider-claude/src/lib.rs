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
    default_profile, resolve_profile, FingerprintProfile, RequestContext, RequestKind,
    CLAUDE_CODE_SYSTEM_PREAMBLE,
};
pub use upstream::{UpstreamClient, UpstreamError};

use async_trait::async_trait;
use omni_common::Replacements;
use omni_core::{
    CanonicalRequest, CanonicalResponse, LlmProvider, ProviderError,
};
use tracing::debug;

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
        let client = UpstreamClient::new_with_profile(profile)
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
        let body_val = serde_json::to_value(&anth_req)
            .map_err(|e| ProviderError::Other(anyhow::Error::msg(format!("anth serialize: {e}"))))?;

        let ctx = RequestContext::new_reply().with_model(anth_req.model.clone());

        // Fresh creds read (the design: never cached in this process).
        let creds = credentials::Credentials::load_fresh_async(&credentials::Credentials::default_path())
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
                assert!(crate::fingerprint::is_claude_code_billing_header(&blocks[0].text));
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
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text(text.into()) }],
            ..Default::default()
        }
    }

    #[test]
    fn claude_additional_ctors_and_shared_repl() {
        let p = ClaudeProvider::new().expect("ctor");
        assert_eq!(p.id(), "claude-code");
        let repl = Replacements::parse(r#"rule = [ { scope = "prompt", search = "x", replace = "y" } ]"#).unwrap();
        assert_eq!(repl.apply_prompt("x"), "y");
    }

    #[tokio::test]
    async fn claude_send_exercises_full_fingerprint_path() {
        // If creds present (as in this env), exercises the complete path:
        // canonical -> translate build -> identity prepend -> finalize cch (profile) ->
        // upstream headers (betas, stainless, ua, session) + body -> real Anth call -> from_anth -> canonical.
        // This is the "live but deterministic shape" test for the port (the unit pins for exact bytes live in fingerprint + translate).
        let p = ClaudeProvider::new().expect("ctor");
        let resp = p.send(sample_req("port test")).await.expect("claude provider send must succeed with creds");
        assert!(resp.model.contains("sonnet"), "resolved sonnet alias should yield sonnet-family model, got {}", resp.model);
        assert!(!resp.content.is_empty() || resp.tool_calls.is_empty()); // basic shape
    }

    #[test]
    fn claude_parity_note() {
        // Full parity with grok via common trait + core canonical + replacements is verified in:
        // omni-core::tests, provider-grok::tests, bin wrapper tests, and common replacements/stats units.
    }

    #[test]
    fn new_for_test_allows_other_profiles() {
        let p154 = ClaudeProvider::new_for_test(crate::fingerprint::resolve_profile("2.1.154").unwrap());
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
        assert!(crate::fingerprint::is_claude_code_billing_header(&sys[0].text));
        assert_eq!(sys[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE);
    }

    #[test]
    fn prompt_repl_before_identity_gate_for_billing() {
        let repl = Replacements::parse(r#"rule = [ { scope = "prompt", search = "Q", replace = "Z" } ]"#).unwrap();
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
        assert_eq!(prof.resolve_model("claude-sonnet-4-6").canonical, "claude-sonnet-4-6");
        // alias
        assert_eq!(prof.resolve_model("sonnet").canonical, "claude-sonnet-4-6");
        // substring
        assert_eq!(prof.resolve_model("foo-sonnet-bar").canonical, "claude-sonnet-4-6");
        // cli
        assert_eq!(prof.resolve_model("opus").canonical, "claude-opus-4-8");
        // default
        assert_eq!(prof.resolve_model("weird").canonical, "claude-opus-4-8");
        // haiku dated
        assert_eq!(prof.resolve_model("claude-haiku-4-5-20251001").canonical, "claude-haiku-4-5-20251001");
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
}

