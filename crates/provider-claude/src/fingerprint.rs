//! Build the outbound header set for api.anthropic.com requests, mimicking
//! the claude CLI wire fingerprint.
//!
//! **CRITICAL INVARIANT (Claude Code fingerprint exactness):**
//! The single active Claude Code pin must reproduce that version's wire
//! fingerprint **byte-for-byte** - the version string, `anthropic-beta` flags,
//! stainless versions, the `x-anthropic-billing-header` no-cch shape, the
//! `cc_version` suffix, the model catalog, wire defaults, and identity
//! preamble injection. This exactness is the entire point of provider-claude:
//! an inexact fingerprint is eventually rejected by Anthropic's subscription
//! OAuth gate. "Close" is a failure, not a partial success.
//!
//! All code that contributes to the serialized request body or the header
//! set for /v1/messages lives ONLY in this crate (fingerprint + translate
//! wire types + identity prepend). It never leaks into omni-common or
//! omni-core.
//!
//! Active baseline: Claude Code 2.1.259 (captured 2026-09-03). Single pin only
//! (issue #12). Billing ends at `cc_entrypoint=sdk-cli;` with no `cch=` field.
//! Historical checksum rewrite lives in `docs/providers/claude/CCH_ALGORITHM.md`.
//!
//! Ported/adapted directly from reference-src-claude/upstream/fingerprint.rs
//! (the authoritative source for the invariant).

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use ring::digest;
use uuid::Uuid;

use crate::credentials::Credentials;
use crate::models::{
    MODEL_CATALOG, ModelDef, ModelInfo, models_list_from_catalog, resolve_model_in_catalog,
};

/// Static identity Omni claims on the wire. These values must move together
/// when re-baselining against a new Claude Code release.
#[derive(Debug, Clone, Copy)]
pub struct FingerprintProfile {
    pub name: &'static str,
    pub claude_cli_version: &'static str,
    pub stainless_package_version: &'static str,
    pub stainless_runtime_version: &'static str,
    pub entrypoint: &'static str,
    pub beta_reply: &'static str,
    pub model_beta_overrides: &'static [ModelBetaOverride],
    pub system_preamble: &'static str,
    pub models: &'static [ModelDef],
    pub preserve_explicit_model: bool,
    pub wire_defaults: WireDefaults,
    pub model_wire_overrides: &'static [ModelWireOverride],
    billing: BillingScheme,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelBetaOverride {
    pub model: &'static str,
    pub beta_reply: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct WireDefaults {
    pub max_tokens: u32,
    pub opus_max_tokens: u32,
    pub temperature: Option<f32>,
    pub output_effort: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelWireOverride {
    pub model: &'static str,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub output_effort: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
struct BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm,
    seed: &'static str,
    sample_indices: &'static [usize],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BillingSuffixAlgorithm {
    Sha256Utf16SampleV1,
}

impl FingerprintProfile {
    pub fn user_agent(&self) -> String {
        format!(
            "claude-cli/{} (external, {})",
            self.claude_cli_version, self.entrypoint
        )
    }

    pub fn resolve_model(&self, input: &str) -> Option<&'static ModelDef> {
        resolve_model_in_catalog(input, self.models)
    }

    pub fn outbound_model(&self, input: &str, model: &ModelDef) -> String {
        if self.preserve_explicit_model && self.is_explicit_claude_model(input) {
            input.to_string()
        } else {
            model.canonical.to_string()
        }
    }

    /// Whether `input` is a real, Anthropic-acceptable Claude model id that
    /// should be forwarded verbatim (an explicit version pin) rather than
    /// resolved to the profile canonical.
    fn is_explicit_claude_model(&self, input: &str) -> bool {
        if !input.starts_with("claude-") {
            return false;
        }
        self.models.iter().any(|model| input == model.canonical)
            || self
                .model_wire_overrides
                .iter()
                .any(|override_| override_.model == input)
    }

    pub fn beta_reply_for_model(&self, model: &str) -> &'static str {
        self.model_beta_overrides
            .iter()
            .find(|override_| override_.model == model)
            .map(|override_| override_.beta_reply)
            .unwrap_or(self.beta_reply)
    }

    pub fn wire_defaults_for_model(&self, model: &str) -> WireDefaults {
        if let Some(override_) = self
            .model_wire_overrides
            .iter()
            .find(|override_| override_.model == model)
        {
            return WireDefaults {
                max_tokens: override_.max_tokens,
                opus_max_tokens: override_.max_tokens,
                temperature: override_.temperature,
                output_effort: override_.output_effort,
            };
        }
        if model.contains("opus") {
            return WireDefaults {
                max_tokens: self.wire_defaults.opus_max_tokens,
                ..self.wire_defaults
            };
        }
        self.wire_defaults
    }

    pub fn models_list(&self) -> Vec<ModelInfo> {
        models_list_from_catalog(self.models)
    }

    pub fn billing_header_text(&self, first_user_text: &str) -> String {
        // Live pin (2.1.186+): no trailing cch field; header ends at cc_entrypoint.
        format!(
            "x-anthropic-billing-header: cc_version={}.{}; cc_entrypoint={};",
            self.claude_cli_version,
            self.billing_suffix(first_user_text),
            self.entrypoint,
        )
    }

    pub fn finalize_body_json(
        &self,
        body: &serde_json::Value,
        _ctx: &RequestContext,
    ) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(body)
    }

    fn billing_suffix(&self, first_user_text: &str) -> String {
        match self.billing.suffix_algorithm {
            BillingSuffixAlgorithm::Sha256Utf16SampleV1 => claude_code_version_suffix_v1(
                first_user_text,
                self.claude_cli_version,
                self.billing.seed,
                self.billing.sample_indices,
            ),
        }
    }
}

const BILLING_SUFFIX_SEED_V1: &str = "59cf53e54c78";
const BILLING_SUFFIX_INDICES_V1: [usize; 3] = [4, 7, 20];
// Live pin: the version suffix is still computed, but the billing header
// carries no cch field and the body is serialized as-is. Captured 2026-09-03
// against Claude Code 2.1.259: the header ends at `cc_entrypoint=sdk-cli;`.
const BILLING_SCHEME_V1_NO_CCH: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
};

pub const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic's OAuth-subscription gate expects this canonical Claude Code
/// identifier in the system block array after the billing marker.
///
/// Verified empirically 2026-05-10: any other prefix, suffix, casing, or
/// preceding whitespace fails. Only block-array form allows additional
/// content; flat-string form must equal this sentence verbatim.
pub const CLAUDE_CODE_SYSTEM_PREAMBLE: &str =
    "You are a Claude agent, built on Anthropic's Claude Agent SDK.";

/// Active-pin default beta list (default-model resolution to opus).
pub const BETA_DEFAULT: &str = "claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,extended-cache-ttl-2025-04-11";
/// Active-pin explicit opus beta list.
pub const BETA_OPUS: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,extended-cache-ttl-2025-04-11";
/// Active-pin sonnet beta list (matches mid-conversation, no fallback-credit).
pub const BETA_SONNET: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,extended-cache-ttl-2025-04-11";
/// Active-pin haiku beta list.
pub const BETA_HAIKU: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11";
/// Active-pin fable beta list (Fable 5.1 and explicit Fable 5; same membership as opus).
pub const BETA_FABLE: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,extended-cache-ttl-2025-04-11";

const MODEL_BETA_OVERRIDES: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-fable-5-1",
        beta_reply: BETA_FABLE,
    },
    ModelBetaOverride {
        model: "claude-fable-5",
        beta_reply: BETA_FABLE,
    },
    ModelBetaOverride {
        model: "fable",
        beta_reply: BETA_FABLE,
    },
    ModelBetaOverride {
        model: "claude-opus-5",
        beta_reply: BETA_OPUS,
    },
    ModelBetaOverride {
        model: "claude-sonnet-5",
        beta_reply: BETA_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_HAIKU,
    },
];

// Active-pin wire: fable-5.1 + fable-5 + opus-5 + sonnet-5 64k/no-temp/high;
// haiku 32k/no-temp/no-effort.
const MODEL_WIRE_OVERRIDES: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-fable-5-1",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-fable-5",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-5",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-sonnet-5",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-haiku-4-5",
        max_tokens: 32_000,
        temperature: None,
        output_effort: None,
    },
    ModelWireOverride {
        model: "claude-haiku-4-5-20251001",
        max_tokens: 32_000,
        temperature: None,
        output_effort: None,
    },
];

/// Active-pin wire defaults: temperature omitted; non-override fallback 32k/high-effort.
pub const WIRE_DEFAULTS: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: None,
    output_effort: Some("high"),
};

pub const DEFAULT_PROFILE_NAME: &str = "cc-2.1.259-sdk-cli";

// Captured 2026-09-03 against installed Claude Code 2.1.259 via the shared
// tools.capture framework (mitmproxy reverse proxy + real claude CLI, clean tmpfs
// HOME), for default, explicit opus, sonnet, haiku, and fable. This is the sole
// active pin (issue #12). Catalog, per-model betas, stainless, preamble, and
// header set match 2.1.257. Drift is the CLI version string. No cch field.
// Captured cc_version=2.1.259.cc8 for prompt "Say OK".
pub const PROFILE_CLAUDE_2_1_259_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: DEFAULT_PROFILE_NAME,
    claude_cli_version: "2.1.259",
    stainless_package_version: "0.112.1",
    stainless_runtime_version: "v26.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: MODEL_CATALOG,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS,
    model_wire_overrides: MODEL_WIRE_OVERRIDES,
    billing: BILLING_SCHEME_V1_NO_CCH,
};

pub fn default_profile() -> &'static FingerprintProfile {
    &PROFILE_CLAUDE_2_1_259_SDK_CLI
}

pub fn is_claude_code_billing_header(text: &str) -> bool {
    // Two accepted shapes:
    //   <= 2.1.175: ...; cc_entrypoint=<ep>; cch=<checksum>;
    //   >= 2.1.186: ...; cc_entrypoint=<ep>;     (no trailing cch field)
    text.starts_with("x-anthropic-billing-header: cc_version=") && text.contains("; cc_entrypoint=")
}

/// What kind of request this is - controls minor header variations.
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
    pub model: Option<String>,
}

impl RequestContext {
    pub fn new_reply() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            client_request_id: Uuid::new_v4(),
            retry_count: 0,
            kind: RequestKind::Reply,
            model: None,
        }
    }

    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = session_id;
        self
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
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
pub fn build_headers(
    creds: &Credentials,
    ctx: &RequestContext,
    profile: &FingerprintProfile,
) -> HeaderMap {
    build_headers_with_profile(creds, ctx, profile)
}

fn build_headers_with_profile(
    creds: &Credentials,
    ctx: &RequestContext,
    profile: &FingerprintProfile,
) -> HeaderMap {
    let mut h = HeaderMap::new();

    insert(&mut h, "accept", "application/json");

    let bearer = format!("Bearer {}", creds.access_token);
    insert(&mut h, "authorization", &bearer);

    insert(&mut h, "content-type", "application/json");

    insert(&mut h, "user-agent", &profile.user_agent());

    insert(
        &mut h,
        "x-claude-code-session-id",
        &ctx.session_id.to_string(),
    );

    insert(&mut h, "x-stainless-arch", "x64");
    insert(&mut h, "x-stainless-lang", "js");
    insert(&mut h, "x-stainless-os", "Linux");
    insert(
        &mut h,
        "x-stainless-package-version",
        profile.stainless_package_version,
    );
    insert(
        &mut h,
        "x-stainless-retry-count",
        &ctx.retry_count.to_string(),
    );
    insert(&mut h, "x-stainless-runtime", "node");
    insert(
        &mut h,
        "x-stainless-runtime-version",
        profile.stainless_runtime_version,
    );
    insert(&mut h, "x-stainless-timeout", "600");

    let beta = match ctx.kind {
        RequestKind::Reply => ctx
            .model
            .as_deref()
            .map(|model| profile.beta_reply_for_model(model))
            .unwrap_or(profile.beta_reply),
    };
    insert(&mut h, "anthropic-beta", beta);

    insert(&mut h, "anthropic-dangerous-direct-browser-access", "true");
    insert(&mut h, "anthropic-version", ANTHROPIC_VERSION);
    insert(&mut h, "x-app", "cli");
    // 2.1.259 capture does not send x-client-request-id.

    h
}

/// Claude Code's body marker appends a three-hex-character suffix to the CLI
/// version. The sampled positions are JavaScript string indices, so non-BMP
/// characters count as two UTF-16 code units. Claude Code joins the sampled
/// one-code-unit strings before hashing, so sampled surrogate halves can pair
/// with each other exactly as a JavaScript string would during UTF-8 encoding.
#[cfg(test)]
pub fn claude_code_version_suffix(first_user_text: &str, claude_cli_version: &str) -> String {
    claude_code_version_suffix_v1(
        first_user_text,
        claude_cli_version,
        BILLING_SUFFIX_SEED_V1,
        &BILLING_SUFFIX_INDICES_V1,
    )
}

fn claude_code_version_suffix_v1(
    first_user_text: &str,
    claude_cli_version: &str,
    seed: &str,
    sample_indices: &[usize],
) -> String {
    let mut input = Vec::new();
    input.extend_from_slice(seed.as_bytes());
    let code_units: Vec<u16> = first_user_text.encode_utf16().collect();
    let mut sampled_units = Vec::with_capacity(sample_indices.len());
    for index in sample_indices {
        if let Some(unit) = code_units.get(*index) {
            sampled_units.push(*unit);
        } else {
            sampled_units.push(b'0' as u16);
        }
    }
    append_javascript_utf8(&mut input, &sampled_units);
    input.extend_from_slice(claude_cli_version.as_bytes());

    let digest = digest::digest(&digest::SHA256, &input);
    let mut suffix = String::with_capacity(3);
    for byte in digest.as_ref().iter().take(2) {
        suffix.push_str(&format!("{byte:02x}"));
    }
    suffix.truncate(3);
    suffix
}

fn append_javascript_utf8(out: &mut Vec<u8>, units: &[u16]) {
    let mut idx = 0;
    while idx < units.len() {
        let unit = units[idx];
        let scalar = if (0xd800..=0xdbff).contains(&unit) {
            if let Some(low) = units.get(idx + 1) {
                if (0xdc00..=0xdfff).contains(low) {
                    idx += 2;
                    0x10000 + (((unit as u32 - 0xd800) << 10) | (*low as u32 - 0xdc00))
                } else {
                    idx += 1;
                    char::REPLACEMENT_CHARACTER as u32
                }
            } else {
                idx += 1;
                char::REPLACEMENT_CHARACTER as u32
            }
        } else if (0xdc00..=0xdfff).contains(&unit) {
            idx += 1;
            char::REPLACEMENT_CHARACTER as u32
        } else {
            idx += 1;
            unit as u32
        };

        let ch = char::from_u32(scalar).unwrap_or(char::REPLACEMENT_CHARACTER);
        let mut buf = [0; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
    }
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
        // WHY: active pin must keep the captured Claude Code header name set;
        // missing or extra headers fail the OAuth subscription gate.
        assert_profile_header_set_matches_baseline(default_profile());
    }

    #[test]
    fn active_pin_uses_captured_beta_list_per_model() {
        // WHY: per-model anthropic-beta bytes are load-bearing. A silent swap of
        // sonnet onto the default (context-1m) list would be an inexact fingerprint.
        let profile = default_profile();
        let creds = fixture_creds();
        let cases = [
            ("claude-fable-5-1", BETA_FABLE),
            ("claude-fable-5", BETA_FABLE),
            ("claude-opus-5", BETA_OPUS),
            ("claude-sonnet-5", BETA_SONNET),
            ("claude-haiku-4-5", BETA_HAIKU),
            ("claude-haiku-4-5-20251001", BETA_HAIKU),
        ];
        for (model, expected_beta) in cases {
            let ctx = RequestContext::new_reply().with_model(model.to_string());
            let beta = build_headers(&creds, &ctx, profile)
                .get("anthropic-beta")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            assert_eq!(beta, expected_beta, "unexpected beta list for {model}");
        }
    }

    #[test]
    fn bare_aliases_resolve_to_correct_per_model_beta_on_active_pin() {
        // WHY: bare alias ("sonnet"/"haiku"/"opus") must get that model's captured
        // beta, not DEFAULT. Requires outbound_model canonicalize before headers.
        let profile = default_profile();
        let creds = fixture_creds();
        let cases = [
            ("fable", BETA_FABLE, false),
            ("opus", BETA_OPUS, false),
            ("sonnet", BETA_SONNET, false),
            ("haiku", BETA_HAIKU, false),
        ];
        for (alias, expected_beta, has_context_1m) in cases {
            let model_def = profile.resolve_model(alias).unwrap();
            let outbound = profile.outbound_model(alias, model_def);
            let ctx = RequestContext::new_reply().with_model(outbound);
            let beta = build_headers(&creds, &ctx, profile)
                .get("anthropic-beta")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            assert_eq!(beta, expected_beta, "alias {alias} got the wrong beta list");
            assert_eq!(
                beta.contains("context-1m"),
                has_context_1m,
                "alias {alias} context-1m presence mismatch"
            );
        }
    }

    fn assert_profile_header_set_matches_baseline(profile: &FingerprintProfile) {
        // WHY: active-pin header names AND static values must match capture.
        // Name-set equality catches missing/extra headers; value locks catch silent
        // fingerprint drift that still returns HTTP 200.
        let creds = fixture_creds();
        let ctx = RequestContext::new_reply();
        let h = build_headers(&creds, &ctx, profile);

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
        ];
        let mut names: Vec<&str> = h.keys().map(|k| k.as_str()).collect();
        names.sort();
        let mut expected = expected_names.to_vec();
        expected.sort();
        assert_eq!(
            names, expected,
            "header name set drifted on {}",
            profile.name
        );

        assert_eq!(h.get("anthropic-version").unwrap(), "2023-06-01");
        assert_eq!(
            h.get("anthropic-dangerous-direct-browser-access").unwrap(),
            "true"
        );
        assert_eq!(h.get("x-app").unwrap(), "cli");
        assert_eq!(h.get("x-stainless-arch").unwrap(), "x64");
        assert_eq!(h.get("x-stainless-lang").unwrap(), "js");
        assert_eq!(h.get("x-stainless-os").unwrap(), "Linux");
        assert_eq!(h.get("x-stainless-runtime").unwrap(), "node");
        assert_eq!(
            h.get("x-stainless-package-version")
                .unwrap()
                .to_str()
                .unwrap(),
            profile.stainless_package_version,
            "x-stainless-package-version drifted on {}",
            profile.name
        );
        assert_eq!(
            h.get("x-stainless-runtime-version")
                .unwrap()
                .to_str()
                .unwrap(),
            profile.stainless_runtime_version,
            "x-stainless-runtime-version drifted on {}",
            profile.name
        );

        // No-model reply path: exact default beta bytes (order included).
        let beta = h.get("anthropic-beta").unwrap().to_str().unwrap();
        assert_eq!(
            beta, profile.beta_reply,
            "default-reply beta drifted from profile.beta_reply on {}",
            profile.name
        );
        assert!(
            beta.contains("oauth-2025-04-20"),
            "beta list missing oauth-2025-04-20: {beta}"
        );
        assert!(
            beta.contains("claude-code-20250219"),
            "beta list missing claude-code-20250219: {beta}"
        );

        let auth = h.get("authorization").unwrap().to_str().unwrap();
        assert!(auth.starts_with("Bearer sk-ant-oat01-"));
        let ua = h.get("user-agent").unwrap().to_str().unwrap();
        assert_eq!(
            ua,
            profile.user_agent(),
            "user-agent drifted from profile.user_agent() on {}",
            profile.name
        );
    }

    #[test]
    fn next_attempt_increments_retry_count_and_rotates_request_id() {
        let mut ctx = RequestContext::new_reply();
        let first_id = ctx.client_request_id;
        ctx.next_attempt();
        assert_eq!(ctx.retry_count, 1);
        assert_ne!(ctx.client_request_id, first_id);
    }

    #[test]
    fn active_pin_matches_refreshed_claude_code_baseline() {
        // WHY: issue #12 ships exactly one pin. These bytes are the gate: UA,
        // stainless, catalog ids, and per-model beta lists must match capture.
        let profile = default_profile();
        assert_eq!(profile.name, "cc-2.1.259-sdk-cli");
        assert_eq!(profile.claude_cli_version, "2.1.259");
        assert_eq!(profile.stainless_package_version, "0.112.1");
        assert_eq!(profile.stainless_runtime_version, "v26.3.0");
        assert_eq!(
            profile.user_agent(),
            "claude-cli/2.1.259 (external, sdk-cli)"
        );
        assert_eq!(
            profile.resolve_model("fable").unwrap().canonical,
            "claude-fable-5-1"
        );
        assert_eq!(
            profile.resolve_model("opus").unwrap().canonical,
            "claude-opus-5"
        );
        assert_eq!(
            profile.resolve_model("sonnet").unwrap().canonical,
            "claude-sonnet-5"
        );
        assert_eq!(
            profile.resolve_model("haiku").unwrap().canonical,
            "claude-haiku-4-5-20251001"
        );
        assert!(profile.beta_reply.contains("fallback-credit-2026-06-01"));
        assert_eq!(profile.beta_reply_for_model("claude-opus-5"), BETA_OPUS);
        assert_eq!(profile.beta_reply_for_model("claude-sonnet-5"), BETA_SONNET);
        // Wire golden: fable/opus/sonnet 64k no-temp high; haiku 32k no-temp no-effort.
        let fable_w = profile.wire_defaults_for_model("claude-fable-5-1");
        assert_eq!(fable_w.max_tokens, 64_000);
        assert_eq!(fable_w.temperature, None);
        assert_eq!(fable_w.output_effort, Some("high"));
        let opus_w = profile.wire_defaults_for_model("claude-opus-5");
        assert_eq!(opus_w.max_tokens, 64_000);
        assert_eq!(opus_w.temperature, None);
        assert_eq!(opus_w.output_effort, Some("high"));
        let sonnet_w = profile.wire_defaults_for_model("claude-sonnet-5");
        assert_eq!(sonnet_w.max_tokens, 64_000);
        assert_eq!(sonnet_w.temperature, None);
        assert_eq!(sonnet_w.output_effort, Some("high"));
        let haiku_w = profile.wire_defaults_for_model("claude-haiku-4-5");
        assert_eq!(haiku_w.max_tokens, 32_000);
        assert_eq!(haiku_w.temperature, None);
        assert_eq!(haiku_w.output_effort, None);
        assert!(
            !profile.billing_header_text("Say OK").contains("cch="),
            "active pin must use no-cch billing header"
        );
    }

    #[test]
    fn active_pin_catalog_names_are_unique() {
        let profile = default_profile();
        assert!(!profile.name.is_empty());
        assert!(!profile.claude_cli_version.is_empty());
        assert!(!profile.models.is_empty());
        assert!(crate::models::catalog_contains_unique_names(profile.models));
    }

    #[test]
    fn billing_suffix_matches_claude_code_probe() {
        // Historical suffix vectors lock the algorithm across past versions.
        // Active pin: 2.1.259 / "Say OK" -> cc8; header has no cch field.
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.142"), "73b");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.150"), "5bd");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.154"), "cea");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.161"), "d2b");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.162"), "b87");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.165"), "492");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.175"), "174");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.186"), "a80");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.197"), "c8e");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.207"), "aa4");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.211"), "08c");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.220"), "01b");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.221"), "116");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.228"), "a3a");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.232"), "1d9");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.257"), "27e");
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.259"), "cc8");
        assert_eq!(
            default_profile().billing_header_text("Say OK"),
            "x-anthropic-billing-header: cc_version=2.1.259.cc8; cc_entrypoint=sdk-cli;"
        );
    }

    #[test]
    fn billing_cch_stays_on_known_safe_sentinel() {
        let profile = default_profile();
        let header = profile.billing_header_text("Say OK");
        assert!(
            !header.contains("cch="),
            "active pin must omit cch: {header}"
        );
        assert!(
            header.ends_with("; cc_entrypoint=sdk-cli;"),
            "active pin unexpected header tail: {header}"
        );
    }

    #[test]
    fn omni_serialized_body_stays_no_cch() {
        let profile = default_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Say OK"}
                    ]
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                },
                {
                    "type": "text",
                    "text": profile.system_preamble,
                }
            ],
            "stream": false
        });
        let json = String::from_utf8(profile.finalize_body_json(&body, &ctx).unwrap()).unwrap();

        assert!(
            json.contains("cc_entrypoint=sdk-cli;"),
            "active body missing entrypoint terminator"
        );
        assert!(
            !json.contains("cch="),
            "active pin body unexpectedly contains a cch field: {json}"
        );
        assert_eq!(json.as_bytes(), serde_json::to_vec(&body).unwrap());
    }

    #[test]
    fn finalized_body_is_deterministic() {
        let profile = default_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                }
            ],
            "messages": []
        });

        assert_eq!(
            profile.finalize_body_json(&body, &ctx).unwrap(),
            profile.finalize_body_json(&body, &ctx).unwrap()
        );
    }

    #[test]
    fn finalized_body_without_billing_sentinel_is_unchanged() {
        let profile = default_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Say OK"}
                    ]
                }
            ]
        });
        let expected = serde_json::to_vec(&body).unwrap();
        assert_eq!(profile.finalize_body_json(&body, &ctx).unwrap(), expected);
    }

    #[test]
    fn finalized_body_preserves_stale_inbound_cch_bytes() {
        // WHY: finalize_body_json serializes only. Stale inbound cch= in a body
        // is not rewritten here; identity injection replaces billing headers
        // before serialize.
        let profile = default_profile();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=abcde;",
                }
            ],
            "messages": []
        });
        let expected = serde_json::to_vec(&body).unwrap();
        assert_eq!(profile.finalize_body_json(&body, &ctx).unwrap(), expected);
    }

    #[test]
    fn billing_header_detector_accepts_real_nonzero_cch() {
        assert!(is_claude_code_billing_header(
            "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=e5ba6;"
        ));
    }

    #[test]
    fn billing_suffix_uses_zero_for_missing_positions() {
        assert_eq!(claude_code_version_suffix("", "2.1.142"), "1aa");
        assert_eq!(claude_code_version_suffix("abc", "2.1.142"), "1aa");
    }

    #[test]
    fn billing_suffix_uses_utf16_code_units() {
        assert_eq!(
            claude_code_version_suffix("abc😀efghijklmnopqrstuv", "2.1.142"),
            "db0"
        );
    }

    #[test]
    fn billing_suffix_treats_sampled_surrogates_like_javascript_string_indices() {
        assert_eq!(claude_code_version_suffix("abcd😀😀", "2.1.142"), "052");
    }

    #[test]
    fn billing_header_text_end_to_end_matches_suffix_oracle_for_varied_first_text() {
        let profile = default_profile();
        let ver = profile.claude_cli_version;
        let inputs = [
            "",
            "Say OK",
            "abc",
            "0123456789abcdefghijuvwxyz",
            "héllo wörld with nön-ascii café 99",
            "abc😀efghijklmnopqrstuv",
            "abcd😀😀",
        ];
        for input in inputs {
            let expected_suffix = claude_code_version_suffix(input, ver);
            assert_eq!(expected_suffix.len(), 3, "suffix len for {input:?}");
            assert!(
                expected_suffix.chars().all(|c| c.is_ascii_hexdigit()),
                "suffix not hex for {input:?}: {expected_suffix}"
            );
            let header = profile.billing_header_text(input);
            assert_eq!(
                header,
                format!(
                    "x-anthropic-billing-header: cc_version={ver}.{expected_suffix}; \
                     cc_entrypoint=sdk-cli;"
                ),
                "billing_header_text diverged from suffix oracle for first_user_text {input:?}"
            );
        }
    }
}
