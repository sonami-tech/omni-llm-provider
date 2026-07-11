//! Build the outbound header set for api.anthropic.com requests, mimicking
//! the claude CLI wire fingerprint.
//!
//! **CRITICAL INVARIANT (Claude Code fingerprint exactness):**
//! For every Claude Code version this crate supports, it MUST reproduce that
//! version's wire fingerprint **byte-for-byte** - the version string,
//! `anthropic-beta` flags, stainless versions, the `x-anthropic-billing-header`
//! cch checksum, the model catalog, wire defaults, and identity preamble
//! injection. This exactness is the entire point of provider-claude: an
//! inexact fingerprint is eventually rejected by Anthropic's subscription
//! OAuth gate. "Close" is a failure, not a partial success.
//!
//! All code that contributes to the serialized request body or the header
//! set for /v1/messages lives ONLY in this crate (fingerprint + translate
//! wire types + identity prepend). It never leaks into omni-common or
//! omni-core.
//!
//! Active baseline captured 2026-06-12 against claude CLI 2.1.175
//! SDK 0.94.0. This profile adds Fable 5, xhigh Opus/Fable effort, and the
//! recovered 2.1.175 cch transform that removes model string values and
//! max_tokens fields before hashing.
//!
//! Ported/adapted directly from reference-src-claude/upstream/fingerprint.rs
//! (the authoritative source for the invariant).

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use ring::digest;
use uuid::Uuid;

use crate::credentials::Credentials;
use crate::models::{
    CATALOG_CC_2_1_142, CATALOG_CC_2_1_150, CATALOG_CC_2_1_154, CATALOG_CC_2_1_158,
    CATALOG_CC_2_1_175, CATALOG_CC_2_1_207, ModelDef, ModelInfo, models_list_from_catalog,
    resolve_model_in_catalog,
};

/// Static identity Omni claims on the wire. These values must move together
/// when re-baselining against a new Claude Code release.
#[derive(Debug, Clone, Copy)]
pub struct FingerprintProfile {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
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
    cch: BillingCchMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BillingSuffixAlgorithm {
    Sha256Utf16SampleV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BillingCchMode {
    #[allow(dead_code)]
    Static(&'static str),
    FinalBodyChecksum,
    FinalBodyChecksumSkipModelsAndMaxTokens,
    /// No `cch=` segment in the billing header at all. Observed on Claude Code
    /// 2.1.186: the header ends at `cc_entrypoint=<entrypoint>;` with no
    /// trailing checksum field. The body is sent unmodified.
    None,
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
        if matches!(self.billing.cch, BillingCchMode::None) {
            // 2.1.186+: no trailing cch field; header ends at cc_entrypoint.
            return format!(
                "x-anthropic-billing-header: cc_version={}.{}; cc_entrypoint={};",
                self.claude_cli_version,
                self.billing_suffix(first_user_text),
                self.entrypoint,
            );
        }
        format!(
            "x-anthropic-billing-header: cc_version={}.{}; cc_entrypoint={}; cch={};",
            self.claude_cli_version,
            self.billing_suffix(first_user_text),
            self.entrypoint,
            self.billing.cch.placeholder()
        )
    }

    pub fn finalize_body_json(
        &self,
        body: &serde_json::Value,
        ctx: &RequestContext,
    ) -> Result<Vec<u8>, serde_json::Error> {
        let bytes = serde_json::to_vec(body)?;
        Ok(self.finalize_body_bytes(bytes, ctx))
    }

    fn finalize_body_bytes(&self, bytes: Vec<u8>, _ctx: &RequestContext) -> Vec<u8> {
        match self.billing.cch {
            BillingCchMode::Static(_) | BillingCchMode::None => bytes,
            BillingCchMode::FinalBodyChecksum => {
                self.finalize_body_cch_checksum(bytes, claude_code_cch_checksum)
            }
            BillingCchMode::FinalBodyChecksumSkipModelsAndMaxTokens => self
                .finalize_body_cch_checksum(
                    bytes,
                    claude_code_cch_checksum_skip_models_and_max_tokens,
                ),
        }
    }

    fn finalize_body_cch_checksum(
        &self,
        mut bytes: Vec<u8>,
        checksum_fn: fn(&[u8]) -> u64,
    ) -> Vec<u8> {
        let Some(offset) = self.find_billing_cch_placeholder(&bytes) else {
            return bytes;
        };
        let checksum = checksum_fn(&bytes);
        let replacement = format!("{checksum:05x}");
        debug_assert_eq!(replacement.len(), 5);
        bytes[offset..offset + 5].copy_from_slice(replacement.as_bytes());
        bytes
    }

    fn find_billing_cch_placeholder(&self, bytes: &[u8]) -> Option<usize> {
        let system_start = find_subslice(bytes, br#""system":"#)?;
        let prefix = format!(
            "x-anthropic-billing-header: cc_version={}.",
            self.claude_cli_version
        );
        let tail = format!("; cc_entrypoint={}; cch=00000;", self.entrypoint);
        let search = &bytes[system_start..];
        let mut cursor = 0;
        while cursor < search.len() {
            let prefix_rel = find_subslice(&search[cursor..], prefix.as_bytes())?;
            let prefix_pos = system_start + cursor + prefix_rel;
            let suffix_pos = prefix_pos + prefix.len() + 3;
            let suffix_end = suffix_pos + tail.len();
            if suffix_end <= bytes.len()
                && bytes[prefix_pos + prefix.len()..suffix_pos]
                    .iter()
                    .all(u8::is_ascii_hexdigit)
                && bytes[suffix_pos..suffix_end] == *tail.as_bytes()
            {
                let cch_rel = tail.find("00000").expect("tail contains cch placeholder");
                return Some(suffix_pos + cch_rel);
            }
            cursor += prefix_rel + prefix.len();
        }
        None
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

impl BillingCchMode {
    fn placeholder(self) -> &'static str {
        match self {
            BillingCchMode::Static(value) => value,
            BillingCchMode::FinalBodyChecksum
            | BillingCchMode::FinalBodyChecksumSkipModelsAndMaxTokens => "00000",
            // No cch segment is emitted; the placeholder is never used.
            BillingCchMode::None => "",
        }
    }
}

const BILLING_SUFFIX_SEED_V1: &str = "59cf53e54c78";
const BILLING_SUFFIX_INDICES_V1: [usize; 3] = [4, 7, 20];
#[allow(dead_code)]
const BILLING_SCHEME_V1_CCH_00000: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
    cch: BillingCchMode::Static("00000"),
};
const BILLING_SCHEME_V1_CCH_XXH64_BODY: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
    cch: BillingCchMode::FinalBodyChecksum,
};
const BILLING_SCHEME_V1_CCH_XXH64_SKIP_MODELS_AND_MAX_TOKENS: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
    cch: BillingCchMode::FinalBodyChecksumSkipModelsAndMaxTokens,
};
// 2.1.186: the version suffix is still computed (cc_version=...a80), but the
// billing header carries no cch field and the body is not rewritten. Verified
// 2026-06-22 against two independent live captures (mitmproxy reverse proxy and
// the drift checker's capture server): the header ends at `cc_entrypoint=sdk-cli;`.
const BILLING_SCHEME_V1_NO_CCH: BillingScheme = BillingScheme {
    suffix_algorithm: BillingSuffixAlgorithm::Sha256Utf16SampleV1,
    seed: BILLING_SUFFIX_SEED_V1,
    sample_indices: &BILLING_SUFFIX_INDICES_V1,
    cch: BillingCchMode::None,
};

pub const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic's OAuth-subscription gate expects this canonical Claude Code
/// identifier in the system block array after the billing marker.
///
/// Verified empirically 2026-05-10: any other prefix, suffix, casing, or
/// preceding whitespace fails. Only block-array form allows additional
/// content; flat-string form must equal this sentence verbatim.
pub const CLAUDE_CODE_SYSTEM_PREAMBLE: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";

/// Default beta-header set, matching the captured "user reply" flow (most
/// permissive). Includes claude-code-20250219 (turns on Claude Code-mode
/// behavior including OAuth-only-models eligibility) and oauth-2025-04-20
/// (Bearer-token acceptance).
pub const DEFAULT_BETA: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,advisor-tool-2026-03-01,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_154_DEFAULT: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_154_SONNET: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_154_HAIKU: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11";

/// 2.1.158 betas captured 2026-05-30 via mitmproxy reverse proxy + real claude CLI 2.1.158.
/// DEFAULT includes the new context-1m-2025-08-07 (observed on default-model resolution to opus).
/// SONNET and HAIKU match the per-model variants observed (haiku beta order differs).
pub const BETA_CC_2_1_158_DEFAULT: &str = "claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_158_SONNET: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_158_HAIKU: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11";

/// 2.1.161 betas captured 2026-06-03 via mitmproxy reverse proxy + real claude
/// CLI 2.1.161 (default/opus, sonnet, haiku). Byte-identical to the 2.1.158
/// per-model lists - aliased rather than re-typed so a future drift is a single
/// edit. Confirmed from live traffic, not carried forward blind.
pub const BETA_CC_2_1_161_DEFAULT: &str = BETA_CC_2_1_158_DEFAULT;
pub const BETA_CC_2_1_161_SONNET: &str = BETA_CC_2_1_158_SONNET;
pub const BETA_CC_2_1_161_HAIKU: &str = BETA_CC_2_1_158_HAIKU;

/// 2.1.162 betas captured 2026-06-04 via mitmproxy reverse proxy + real claude
/// CLI 2.1.162 (default/opus, sonnet, haiku), driven from a clean CWD. Confirmed
/// byte-identical to the 2.1.158/2.1.161 per-model lists from live traffic, not
/// carried forward blind - aliased so a future drift is a single edit.
pub const BETA_CC_2_1_162_DEFAULT: &str = BETA_CC_2_1_158_DEFAULT;
pub const BETA_CC_2_1_162_SONNET: &str = BETA_CC_2_1_158_SONNET;
pub const BETA_CC_2_1_162_HAIKU: &str = BETA_CC_2_1_158_HAIKU;

/// 2.1.165 betas captured 2026-06-05 via mitmproxy reverse proxy + real claude
/// CLI 2.1.165 (default/opus, sonnet, haiku), driven from a clean CWD. Confirmed
/// byte-identical to the 2.1.158/2.1.161/2.1.162 per-model lists from live
/// traffic, not carried forward blind - aliased so a future drift is a single
/// edit.
pub const BETA_CC_2_1_165_DEFAULT: &str = BETA_CC_2_1_158_DEFAULT;
pub const BETA_CC_2_1_165_SONNET: &str = BETA_CC_2_1_158_SONNET;
pub const BETA_CC_2_1_165_HAIKU: &str = BETA_CC_2_1_158_HAIKU;

/// 2.1.175 betas captured 2026-06-12 with local fake-server probes. Default
/// model resolution to Opus carries context-1m, but explicit Opus does not.
/// Fable adds the fallback-credit beta.
pub const BETA_CC_2_1_175_DEFAULT: &str = BETA_CC_2_1_158_DEFAULT;
pub const BETA_CC_2_1_175_OPUS: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_175_SONNET: &str = BETA_CC_2_1_158_SONNET;
pub const BETA_CC_2_1_175_HAIKU: &str = BETA_CC_2_1_158_HAIKU;
pub const BETA_CC_2_1_175_FABLE: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11";

/// 2.1.186 betas captured 2026-06-22 via the shared `tools.capture` framework
/// (mitmproxy reverse proxy + real claude CLI 2.1.186, clean tmpfs HOME), for
/// default/opus, explicit opus, sonnet, and haiku. Versus 2.1.175 every model
/// gains `thinking-token-count-2026-05-13` and drops `afk-mode-2026-01-31`;
/// sonnet additionally gains `mid-conversation-system-2026-04-07`. Default-model
/// resolution to Opus still carries context-1m; explicit Opus does not. Fable was
/// unavailable on the capture account (Fable Mythos access gate), so its list is
/// carried forward from 2.1.175 with the same thinking-token-count/afk-mode delta
/// applied to stay internally consistent (account gate, not a client change).
pub const BETA_CC_2_1_186_DEFAULT: &str = "claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_186_OPUS: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_186_SONNET: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_186_HAIKU: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11";
pub const BETA_CC_2_1_186_FABLE: &str = "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,extended-cache-ttl-2025-04-11";

const MODEL_BETA_OVERRIDES_CC_2_1_154: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-opus-4-8",
        beta_reply: BETA_CC_2_1_154_DEFAULT,
    },
    ModelBetaOverride {
        model: "claude-opus-4-7",
        beta_reply: BETA_CC_2_1_154_SONNET,
    },
    ModelBetaOverride {
        model: "claude-opus-4-6",
        beta_reply: BETA_CC_2_1_154_SONNET,
    },
    ModelBetaOverride {
        model: "claude-sonnet-4-6",
        beta_reply: BETA_CC_2_1_154_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_CC_2_1_154_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_CC_2_1_154_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_CC_2_1_154_HAIKU,
    },
];

// The opus-4-7 / opus-4-6 rows are off-catalog for 2.1.158 (not in
// CATALOG_CC_2_1_158); real 2.1.158 never emits them, so there is no captured
// fingerprint to match - they are carry-forward fallbacks for consumers that
// explicitly request those ids, mapped to the closest captured beta (SONNET).
const MODEL_BETA_OVERRIDES_CC_2_1_158: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-opus-4-8",
        beta_reply: BETA_CC_2_1_158_DEFAULT,
    },
    ModelBetaOverride {
        model: "claude-opus-4-7",
        beta_reply: BETA_CC_2_1_158_SONNET,
    },
    ModelBetaOverride {
        model: "claude-opus-4-6",
        beta_reply: BETA_CC_2_1_158_SONNET,
    },
    ModelBetaOverride {
        model: "claude-sonnet-4-6",
        beta_reply: BETA_CC_2_1_158_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_CC_2_1_158_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_CC_2_1_158_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_CC_2_1_158_HAIKU,
    },
];

// 2.1.161 per-model beta overrides. Same shape and (captured-identical) values
// as 2.1.158; opus-4-7/4-6 remain off-catalog carry-forward fallbacks mapped to
// the closest captured beta (SONNET).
const MODEL_BETA_OVERRIDES_CC_2_1_161: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-opus-4-8",
        beta_reply: BETA_CC_2_1_161_DEFAULT,
    },
    ModelBetaOverride {
        model: "claude-opus-4-7",
        beta_reply: BETA_CC_2_1_161_SONNET,
    },
    ModelBetaOverride {
        model: "claude-opus-4-6",
        beta_reply: BETA_CC_2_1_161_SONNET,
    },
    ModelBetaOverride {
        model: "claude-sonnet-4-6",
        beta_reply: BETA_CC_2_1_161_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_CC_2_1_161_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_CC_2_1_161_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_CC_2_1_161_HAIKU,
    },
];

// 2.1.162 per-model beta overrides. Same shape and (captured-identical 2026-06-04)
// values as 2.1.158/2.1.161; opus-4-7/4-6 remain off-catalog carry-forward
// fallbacks mapped to the closest captured beta (SONNET).
const MODEL_BETA_OVERRIDES_CC_2_1_162: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-opus-4-8",
        beta_reply: BETA_CC_2_1_162_DEFAULT,
    },
    ModelBetaOverride {
        model: "claude-opus-4-7",
        beta_reply: BETA_CC_2_1_162_SONNET,
    },
    ModelBetaOverride {
        model: "claude-opus-4-6",
        beta_reply: BETA_CC_2_1_162_SONNET,
    },
    ModelBetaOverride {
        model: "claude-sonnet-4-6",
        beta_reply: BETA_CC_2_1_162_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_CC_2_1_162_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_CC_2_1_162_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_CC_2_1_162_HAIKU,
    },
];

// 2.1.165 per-model beta overrides. Same shape and (captured-identical 2026-06-05)
// values as 2.1.158/2.1.161/2.1.162; opus-4-7/4-6 remain off-catalog carry-forward
// fallbacks mapped to the closest captured beta (SONNET).
const MODEL_BETA_OVERRIDES_CC_2_1_165: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-opus-4-8",
        beta_reply: BETA_CC_2_1_165_DEFAULT,
    },
    ModelBetaOverride {
        model: "claude-opus-4-7",
        beta_reply: BETA_CC_2_1_165_SONNET,
    },
    ModelBetaOverride {
        model: "claude-opus-4-6",
        beta_reply: BETA_CC_2_1_165_SONNET,
    },
    ModelBetaOverride {
        model: "claude-sonnet-4-6",
        beta_reply: BETA_CC_2_1_165_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_CC_2_1_165_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_CC_2_1_165_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_CC_2_1_165_HAIKU,
    },
];

const MODEL_BETA_OVERRIDES_CC_2_1_175: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-fable-5",
        beta_reply: BETA_CC_2_1_175_FABLE,
    },
    ModelBetaOverride {
        model: "claude-opus-4-8",
        beta_reply: BETA_CC_2_1_175_OPUS,
    },
    ModelBetaOverride {
        model: "claude-sonnet-4-6",
        beta_reply: BETA_CC_2_1_175_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_CC_2_1_175_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_CC_2_1_175_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_CC_2_1_175_HAIKU,
    },
];

const MODEL_BETA_OVERRIDES_CC_2_1_186: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-fable-5",
        beta_reply: BETA_CC_2_1_186_FABLE,
    },
    ModelBetaOverride {
        model: "claude-opus-4-8",
        beta_reply: BETA_CC_2_1_186_OPUS,
    },
    ModelBetaOverride {
        model: "claude-sonnet-4-6",
        beta_reply: BETA_CC_2_1_186_SONNET,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_CC_2_1_186_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_CC_2_1_186_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_CC_2_1_186_HAIKU,
    },
];

// 2.1.207 per-model betas captured 2026-07-11. Default/opus/haiku lists match
// 2.1.186. Sonnet (now `claude-sonnet-5`) gains mid-conversation-system and
// matches the explicit-opus list (no context-1m). Fable uncaptured; carry 186.
const MODEL_BETA_OVERRIDES_CC_2_1_207: &[ModelBetaOverride] = &[
    ModelBetaOverride {
        model: "claude-fable-5",
        beta_reply: BETA_CC_2_1_186_FABLE,
    },
    ModelBetaOverride {
        model: "claude-opus-4-8",
        beta_reply: BETA_CC_2_1_186_OPUS,
    },
    ModelBetaOverride {
        model: "claude-sonnet-5",
        beta_reply: BETA_CC_2_1_186_OPUS,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5",
        beta_reply: BETA_CC_2_1_186_HAIKU,
    },
    ModelBetaOverride {
        model: "claude-haiku-4-5-20251001",
        beta_reply: BETA_CC_2_1_186_HAIKU,
    },
    ModelBetaOverride {
        model: "haiku",
        beta_reply: BETA_CC_2_1_186_HAIKU,
    },
];

// 2.1.207 wire overrides: opus+sonnet-5 are 64k/no-temp/high-effort; haiku is
// 32k/no-temp/no-effort. Fable uncaptured; carry 175 xhigh.
const MODEL_WIRE_OVERRIDES_CC_2_1_207: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-fable-5",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("xhigh"),
    },
    ModelWireOverride {
        model: "claude-opus-4-8",
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

const MODEL_WIRE_OVERRIDES_CC_2_1_154: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-opus-4-8",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-7",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-6",
        max_tokens: 64_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-sonnet-4-6",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-haiku-4-5",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
    ModelWireOverride {
        model: "claude-haiku-4-5-20251001",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
];

const MODEL_WIRE_OVERRIDES_CC_2_1_158: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-opus-4-8",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-7",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-6",
        max_tokens: 64_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-sonnet-4-6",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-haiku-4-5",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
    ModelWireOverride {
        model: "claude-haiku-4-5-20251001",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
];

// 2.1.161 per-model wire overrides. Captured 2026-06-03 and byte-identical to
// 2.1.158: opus 64k/no-temp/high-effort, sonnet & haiku 32k/temp=1, haiku no
// effort. Re-typed (not aliased) to keep each profile's wire surface explicit.
const MODEL_WIRE_OVERRIDES_CC_2_1_161: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-opus-4-8",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-7",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-6",
        max_tokens: 64_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-sonnet-4-6",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-haiku-4-5",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
    ModelWireOverride {
        model: "claude-haiku-4-5-20251001",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
];

// 2.1.162 per-model wire overrides. Captured 2026-06-04 (clean-CWD mitmproxy) and
// byte-identical to 2.1.158/2.1.161: opus 64k/no-temp/high-effort, sonnet & haiku
// 32k/temp=1, haiku no effort. The profile `output_effort` serializes to the wire
// `output_config.effort` object (confirmed: real bodies carry
// `output_config:{"effort":"high"}` on opus+sonnet, none on haiku). Re-typed (not
// aliased) to keep each profile's wire surface explicit.
const MODEL_WIRE_OVERRIDES_CC_2_1_162: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-opus-4-8",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-7",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-6",
        max_tokens: 64_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-sonnet-4-6",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-haiku-4-5",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
    ModelWireOverride {
        model: "claude-haiku-4-5-20251001",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
];

// 2.1.165 per-model wire overrides. Captured 2026-06-05 (clean-CWD mitmproxy) and
// byte-identical to 2.1.158/2.1.161/2.1.162: opus 64k/no-temp/high-effort, sonnet
// & haiku 32k/temp=1, haiku no effort. The profile `output_effort` serializes to
// the wire `output_config.effort` object (confirmed: real bodies carry
// `output_config:{"effort":"high"}` on opus+sonnet, none on haiku). Re-typed (not
// aliased) to keep each profile's wire surface explicit.
const MODEL_WIRE_OVERRIDES_CC_2_1_165: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-opus-4-8",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-7",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-opus-4-6",
        max_tokens: 64_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-sonnet-4-6",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-haiku-4-5",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
    ModelWireOverride {
        model: "claude-haiku-4-5-20251001",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
];

const MODEL_WIRE_OVERRIDES_CC_2_1_175: &[ModelWireOverride] = &[
    ModelWireOverride {
        model: "claude-fable-5",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("xhigh"),
    },
    ModelWireOverride {
        model: "claude-opus-4-8",
        max_tokens: 64_000,
        temperature: None,
        output_effort: Some("xhigh"),
    },
    ModelWireOverride {
        model: "claude-sonnet-4-6",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: Some("high"),
    },
    ModelWireOverride {
        model: "claude-haiku-4-5",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
    ModelWireOverride {
        model: "claude-haiku-4-5-20251001",
        max_tokens: 32_000,
        temperature: Some(1.0),
        output_effort: None,
    },
];

pub const WIRE_DEFAULTS_LEGACY: WireDefaults = WireDefaults {
    max_tokens: 64_000,
    opus_max_tokens: 128_000,
    temperature: None,
    output_effort: None,
};

pub const WIRE_DEFAULTS_CC_2_1_154: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: Some(1.0),
    output_effort: Some("high"),
};

pub const WIRE_DEFAULTS_CC_2_1_158: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: Some(1.0),
    output_effort: Some("high"),
};

// 2.1.161 wire defaults - captured 2026-06-03, identical to 2.1.158.
pub const WIRE_DEFAULTS_CC_2_1_161: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: Some(1.0),
    output_effort: Some("high"),
};

// 2.1.162 wire defaults - captured 2026-06-04 (clean-CWD mitmproxy), identical to
// 2.1.158/2.1.161.
pub const WIRE_DEFAULTS_CC_2_1_162: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: Some(1.0),
    output_effort: Some("high"),
};

// 2.1.165 wire defaults - captured 2026-06-05 (clean-CWD mitmproxy), identical to
// 2.1.158/2.1.161/2.1.162.
pub const WIRE_DEFAULTS_CC_2_1_165: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: Some(1.0),
    output_effort: Some("high"),
};

pub const WIRE_DEFAULTS_CC_2_1_175: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: Some(1.0),
    output_effort: Some("high"),
};

// 2.1.186 wire defaults - captured 2026-06-22 (shared tools.capture, real claude
// CLI 2.1.186): opus 64k/no-temp/high-effort, sonnet & haiku 32k/temp=1 (haiku no
// effort). Byte-identical to 2.1.175; the per-model overrides carry the exact
// values, this struct only supplies the non-opus fallback.
pub const WIRE_DEFAULTS_CC_2_1_186: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: Some(1.0),
    output_effort: Some("high"),
};

// 2.1.207 wire defaults - captured 2026-07-11: temperature omitted on all
// captured models; non-override fallback is 32k/high-effort.
pub const WIRE_DEFAULTS_CC_2_1_207: WireDefaults = WireDefaults {
    max_tokens: 32_000,
    opus_max_tokens: 64_000,
    temperature: None,
    output_effort: Some("high"),
};

pub const DEFAULT_PROFILE_NAME: &str = "cc-2.1.207-sdk-cli";
pub const LATEST_PROFILE_ALIAS: &str = "latest";

pub const PROFILE_CLAUDE_2_1_142_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.142-sdk-cli",
    aliases: &["2.1.142"],
    claude_cli_version: "2.1.142",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: DEFAULT_BETA,
    model_beta_overrides: &[],
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_142,
    preserve_explicit_model: false,
    wire_defaults: WIRE_DEFAULTS_LEGACY,
    model_wire_overrides: &[],
    billing: BILLING_SCHEME_V1_CCH_XXH64_BODY,
};

pub const PROFILE_CLAUDE_2_1_150_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.150-sdk-cli",
    aliases: &["2.1.150"],
    claude_cli_version: "2.1.150",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: DEFAULT_BETA,
    model_beta_overrides: &[],
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_150,
    preserve_explicit_model: false,
    wire_defaults: WIRE_DEFAULTS_LEGACY,
    model_wire_overrides: &[],
    billing: BILLING_SCHEME_V1_CCH_XXH64_BODY,
};

pub const PROFILE_CLAUDE_2_1_154_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.154-sdk-cli",
    aliases: &["2.1.154"],
    claude_cli_version: "2.1.154",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_154_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_154,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_154,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_154,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_154,
    billing: BILLING_SCHEME_V1_CCH_XXH64_BODY,
};

pub const PROFILE_CLAUDE_2_1_158_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.158-sdk-cli",
    aliases: &["2.1.158"],
    claude_cli_version: "2.1.158",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_158_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_158,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_158,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_158,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_158,
    billing: BILLING_SCHEME_V1_CCH_XXH64_BODY,
};

pub const PROFILE_CLAUDE_2_1_161_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.161-sdk-cli",
    aliases: &["2.1.161"],
    claude_cli_version: "2.1.161",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_161_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_161,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_158,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_161,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_161,
    billing: BILLING_SCHEME_V1_CCH_XXH64_BODY,
};

// Captured 2026-06-04 (clean-CWD mitmproxy, real claude CLI 2.1.162). A pure
// version bump from 2.1.161: per-model beta lists, stainless versions, wire
// defaults, default-model resolution (opus -> claude-opus-4-8), and the cch
// algorithm (xxh64 seed 0x4d659218e32a3268, drift checker "matches pinned
// algorithm") are all live-confirmed unchanged; only the version string and the
// version-derived cc_version suffix moved (d2b -> b87 for "Say OK"). The model
// catalog is carried forward from 2.1.158 (reused, as 161 does): the capture ran
// with CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1, which suppresses the startup
// /v1/models GET, so the catalog is not freshly GET-enumerable here - but all
// three pinned ids (opus-4-8/sonnet-4-6/haiku-4-5) were confirmed accepted in
// real bodies and 2.1.162 carries no model rename. `preserve_explicit_model` is
// set deliberately to true, matching 2.1.161 (not defaulted).
pub const PROFILE_CLAUDE_2_1_162_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.162-sdk-cli",
    aliases: &["2.1.162"],
    claude_cli_version: "2.1.162",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_162_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_162,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_158,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_162,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_162,
    billing: BILLING_SCHEME_V1_CCH_XXH64_BODY,
};

// Captured 2026-06-05 (clean-CWD mitmproxy, real claude CLI 2.1.165). A pure
// version bump from 2.1.162: per-model beta lists (opus/default, sonnet, haiku
// all byte-identical to the 2.1.158 reference), stainless versions (0.94.0 /
// v24.3.0), wire defaults (opus 64k/no-temp/high-effort; sonnet & haiku
// 32k/temp=1; haiku no effort), default-model resolution (opus ->
// claude-opus-4-8), and the cch algorithm (xxh64 seed 0x4d659218e32a3268, drift
// checker "matches pinned algorithm: b5d33") are all live-confirmed unchanged;
// only the version string and the version-derived cc_version suffix moved. The
// model catalog is carried forward from 2.1.158 (reused, as 161/162 do): the
// capture ran with CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1, which suppresses
// the startup /v1/models GET, so the catalog is not freshly GET-enumerable here -
// but all three pinned ids (opus-4-8/sonnet-4-6/haiku-4-5) were confirmed
// accepted in real bodies and 2.1.165 carries no model rename.
// `preserve_explicit_model` is set deliberately to true, matching 2.1.162 (not
// defaulted).
pub const PROFILE_CLAUDE_2_1_165_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.165-sdk-cli",
    aliases: &["2.1.165"],
    claude_cli_version: "2.1.165",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_165_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_165,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_158,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_165,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_165,
    billing: BILLING_SCHEME_V1_CCH_XXH64_BODY,
};

// Captured 2026-06-12 against installed Claude Code 2.1.175. Headers keep SDK
// 0.94.0 / Node v24.3.0. Body defaults changed: Opus now emits
// output_config.effort=xhigh, Fable is in the catalog with fallback-credit beta,
// and the cch input omits model values plus the max_tokens field.
pub const PROFILE_CLAUDE_2_1_175_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.175-sdk-cli",
    aliases: &["2.1.175"],
    claude_cli_version: "2.1.175",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_175_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_175,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_175,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_175,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_175,
    billing: BILLING_SCHEME_V1_CCH_XXH64_SKIP_MODELS_AND_MAX_TOKENS,
};

// Captured 2026-06-22 against installed Claude Code 2.1.186 via the shared
// tools.capture framework (mitmproxy reverse proxy + real claude CLI, clean tmpfs
// HOME), for default/opus, explicit opus, sonnet, and haiku. Headers keep SDK
// 0.94.0 / Node v24.3.0 and anthropic-version 2023-06-01. The cc_version suffix
// algorithm is UNCHANGED: the existing Sha256Utf16SampleV1 suffix reproduces the
// captured cc_version=2.1.186.a80 exactly (verified against the live header).
// TWO substantive drifts vs 2.1.175:
//   1. The billing header DROPS the trailing `cch=` field entirely - it now ends
//      at `cc_entrypoint=sdk-cli;` with no checksum and no body rewrite. Confirmed
//      byte-for-byte by two independent live captures (mitmproxy + drift checker
//      capture server). Hence billing = BILLING_SCHEME_V1_NO_CCH.
//   2. Per-model betas: every model gains thinking-token-count-2026-05-13 and
//      drops afk-mode-2026-01-31; sonnet also gains mid-conversation-system.
// Wire defaults, catalog, default model (opus -> claude-opus-4-8), and
// preserve_explicit_model carry forward from 2.1.175 (all live-confirmed).
pub const PROFILE_CLAUDE_2_1_186_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.186-sdk-cli",
    aliases: &["2.1.186"],
    claude_cli_version: "2.1.186",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_186_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_186,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_175,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_186,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_175,
    billing: BILLING_SCHEME_V1_NO_CCH,
};

// Captured 2026-07-01 against installed Claude Code 2.1.197 via the shared
// tools.capture framework (mitmproxy reverse proxy + real claude CLI, clean tmpfs
// HOME), for default/opus, explicit opus, sonnet, and haiku. Retained as a
// selectable older profile (no longer `latest`). ONLY TWO fields drifted vs
// 2.1.186: claude_cli_version and stainless_runtime_version (v26.3.0).
pub const PROFILE_CLAUDE_2_1_197_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: "cc-2.1.197-sdk-cli",
    aliases: &["2.1.197"],
    claude_cli_version: "2.1.197",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v26.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_186_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_186,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_175,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_186,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_175,
    billing: BILLING_SCHEME_V1_NO_CCH,
};

// Captured 2026-07-11 against installed Claude Code 2.1.207 via the shared
// tools.capture framework (mitmproxy reverse proxy + real claude CLI, clean tmpfs
// HOME), for default/opus, explicit opus, sonnet, and haiku. This is now the
// default `latest` profile. Drift vs 2.1.197:
//   1. claude_cli_version 2.1.197 -> 2.1.207 (UA + billing cc_version).
//   2. Sonnet catalog id claude-sonnet-4-6 -> claude-sonnet-5; sonnet wire is
//      now 64k/no-temp/effort=high (was 32k/temp=1).
//   3. Opus output_config.effort is high (captured), not the older xhigh pin.
//   4. Haiku temperature omitted (was temp=1).
//   5. Sonnet beta list matches explicit opus (gains mid-conversation-system).
// SDK package stays 0.94.0, runtime stays v26.3.0, anthropic-version stays
// 2023-06-01. No cch field; cc_version suffix algorithm unchanged
// (captured cc_version=2.1.207.aa4 for prompt "Say OK").
pub const PROFILE_CLAUDE_2_1_207_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: DEFAULT_PROFILE_NAME,
    aliases: &["2.1.207"],
    claude_cli_version: "2.1.207",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v26.3.0",
    entrypoint: "sdk-cli",
    beta_reply: BETA_CC_2_1_186_DEFAULT,
    model_beta_overrides: MODEL_BETA_OVERRIDES_CC_2_1_207,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_207,
    preserve_explicit_model: true,
    wire_defaults: WIRE_DEFAULTS_CC_2_1_207,
    model_wire_overrides: MODEL_WIRE_OVERRIDES_CC_2_1_207,
    billing: BILLING_SCHEME_V1_NO_CCH,
};

pub static FINGERPRINT_PROFILES: &[FingerprintProfile] = &[
    PROFILE_CLAUDE_2_1_207_SDK_CLI,
    PROFILE_CLAUDE_2_1_197_SDK_CLI,
    PROFILE_CLAUDE_2_1_186_SDK_CLI,
    PROFILE_CLAUDE_2_1_175_SDK_CLI,
    PROFILE_CLAUDE_2_1_165_SDK_CLI,
    PROFILE_CLAUDE_2_1_162_SDK_CLI,
    PROFILE_CLAUDE_2_1_161_SDK_CLI,
    PROFILE_CLAUDE_2_1_158_SDK_CLI,
    PROFILE_CLAUDE_2_1_154_SDK_CLI,
    PROFILE_CLAUDE_2_1_150_SDK_CLI,
    PROFILE_CLAUDE_2_1_142_SDK_CLI,
];

pub fn default_profile() -> &'static FingerprintProfile {
    resolve_profile(DEFAULT_PROFILE_NAME).expect("default fingerprint profile must exist")
}

pub fn resolve_profile(selector: &str) -> Option<&'static FingerprintProfile> {
    let selector = selector.trim();
    let selector = if selector.is_empty() || selector == LATEST_PROFILE_ALIAS {
        DEFAULT_PROFILE_NAME
    } else {
        selector
    };

    FINGERPRINT_PROFILES
        .iter()
        .find(|profile| profile.name == selector || profile.aliases.contains(&selector))
}

pub fn valid_profile_selectors() -> String {
    let mut selectors = vec![LATEST_PROFILE_ALIAS.to_string()];
    for profile in FINGERPRINT_PROFILES {
        selectors.push(profile.name.to_string());
        for alias in profile.aliases {
            selectors.push((*alias).to_string());
        }
    }
    selectors.join(", ")
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
    insert(
        &mut h,
        "x-client-request-id",
        &ctx.client_request_id.to_string(),
    );

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

const CCH_XXH64_SEED: u64 = 0x4d659218e32a3268;
const XXH64_PRIME1: u64 = 11_400_714_785_074_694_791;
const XXH64_PRIME2: u64 = 14_029_467_366_897_019_727;
const XXH64_PRIME3: u64 = 1_609_587_929_392_839_161;
const XXH64_PRIME4: u64 = 9_650_029_242_287_828_579;
const XXH64_PRIME5: u64 = 2_870_177_450_012_600_261;

fn claude_code_cch_checksum(bytes: &[u8]) -> u64 {
    xxh64(bytes, CCH_XXH64_SEED) & 0xfffff
}

fn claude_code_cch_checksum_skip_models_and_max_tokens(bytes: &[u8]) -> u64 {
    xxh64(
        &body_for_cch_skip_models_and_max_tokens(bytes),
        CCH_XXH64_SEED,
    ) & 0xfffff
}

fn body_for_cch_skip_models_and_max_tokens(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    while let Some((range_start, range_end)) = find_next_max_tokens_range(bytes, cursor) {
        append_with_model_values_removed(&mut out, &bytes[cursor..range_start]);
        cursor = range_end;
    }
    append_with_model_values_removed(&mut out, &bytes[cursor..]);
    out
}

fn find_next_max_tokens_range(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let found = find_subslice(&bytes[start..], br#""max_tokens":"#)? + start;
    let value_start = found + br#""max_tokens":"#.len();
    let mut value_end = value_start;
    while value_end < bytes.len() && bytes[value_end].is_ascii_digit() {
        value_end += 1;
    }
    if value_end == value_start {
        return Some((found, value_end));
    }
    if found > start && bytes[found - 1] == b',' {
        Some((found - 1, value_end))
    } else if value_end < bytes.len() && bytes[value_end] == b',' {
        Some((found, value_end + 1))
    } else {
        Some((found, value_end))
    }
}

fn append_with_model_values_removed(out: &mut Vec<u8>, bytes: &[u8]) {
    let mut cursor = 0;
    while let Some(rel) = find_subslice(&bytes[cursor..], br#""model":""#) {
        let start = cursor + rel;
        let value_start = start + br#""model":""#.len();
        let Some(value_end) = find_json_string_end(bytes, value_start) else {
            break;
        };
        out.extend_from_slice(&bytes[cursor..value_start]);
        cursor = value_end;
    }
    out.extend_from_slice(&bytes[cursor..]);
}

fn find_json_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut idx = start;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\\' => idx = idx.saturating_add(2),
            b'"' => return Some(idx),
            _ => idx += 1,
        }
    }
    None
}

fn xxh64(bytes: &[u8], seed: u64) -> u64 {
    let mut offset = 0;
    let mut h64;

    if bytes.len() >= 32 {
        let mut v1 = seed.wrapping_add(XXH64_PRIME1).wrapping_add(XXH64_PRIME2);
        let mut v2 = seed.wrapping_add(XXH64_PRIME2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(XXH64_PRIME1);

        while offset <= bytes.len() - 32 {
            v1 = xxh64_round(v1, read_u64_le(bytes, offset));
            v2 = xxh64_round(v2, read_u64_le(bytes, offset + 8));
            v3 = xxh64_round(v3, read_u64_le(bytes, offset + 16));
            v4 = xxh64_round(v4, read_u64_le(bytes, offset + 24));
            offset += 32;
        }

        h64 = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed.wrapping_add(XXH64_PRIME5);
    }

    h64 = h64.wrapping_add(bytes.len() as u64);

    while offset + 8 <= bytes.len() {
        let k1 = xxh64_round(0, read_u64_le(bytes, offset));
        h64 ^= k1;
        h64 = h64
            .rotate_left(27)
            .wrapping_mul(XXH64_PRIME1)
            .wrapping_add(XXH64_PRIME4);
        offset += 8;
    }

    if offset + 4 <= bytes.len() {
        h64 ^= (read_u32_le(bytes, offset) as u64).wrapping_mul(XXH64_PRIME1);
        h64 = h64
            .rotate_left(23)
            .wrapping_mul(XXH64_PRIME2)
            .wrapping_add(XXH64_PRIME3);
        offset += 4;
    }

    while offset < bytes.len() {
        h64 ^= (bytes[offset] as u64).wrapping_mul(XXH64_PRIME5);
        h64 = h64.rotate_left(11).wrapping_mul(XXH64_PRIME1);
        offset += 1;
    }

    xxh64_avalanche(h64)
}

fn xxh64_round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(XXH64_PRIME2))
        .rotate_left(31)
        .wrapping_mul(XXH64_PRIME1)
}

fn xxh64_merge_round(acc: u64, val: u64) -> u64 {
    let mut acc = acc ^ xxh64_round(0, val);
    acc = acc.wrapping_mul(XXH64_PRIME1).wrapping_add(XXH64_PRIME4);
    acc
}

fn xxh64_avalanche(mut h64: u64) -> u64 {
    h64 ^= h64 >> 33;
    h64 = h64.wrapping_mul(XXH64_PRIME2);
    h64 ^= h64 >> 29;
    h64 = h64.wrapping_mul(XXH64_PRIME3);
    h64 ^= h64 >> 32;
    h64
}

fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("xxh64 chunk length must be 8"),
    )
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("xxh64 chunk length must be 4"),
    )
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
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
        for profile in FINGERPRINT_PROFILES {
            assert_profile_header_set_matches_baseline(profile);
        }
    }

    #[test]
    fn claude_2_1_154_uses_captured_beta_list_per_model() {
        let profile = resolve_profile("2.1.154").unwrap();
        let creds = fixture_creds();
        let cases = [
            ("claude-opus-4-8", BETA_CC_2_1_154_DEFAULT),
            ("claude-opus-4-7", BETA_CC_2_1_154_SONNET),
            ("claude-opus-4-6", BETA_CC_2_1_154_SONNET),
            ("claude-sonnet-4-6", BETA_CC_2_1_154_SONNET),
            ("claude-haiku-4-5", BETA_CC_2_1_154_HAIKU),
            ("claude-haiku-4-5-20251001", BETA_CC_2_1_154_HAIKU),
        ];

        for (model, expected_beta) in cases {
            let ctx = RequestContext::new_reply().with_model(model.to_string());
            let headers = build_headers(&creds, &ctx, profile);
            assert_eq!(
                headers.get("anthropic-beta").unwrap().to_str().unwrap(),
                expected_beta,
                "unexpected beta list for {model}"
            );
        }
    }

    #[test]
    fn claude_2_1_158_uses_captured_beta_list_per_model() {
        // Locks the 2.1.158 per-model anthropic-beta header against the values
        // captured from real claude CLI 2.1.158 traffic (2026-05-30). The
        // load-bearing fact: opus-4-8 carries the new `context-1m-2025-08-07`
        // flag (DEFAULT beta) while sonnet/haiku do NOT - both byte-confirmed
        // against live capture.
        let profile = resolve_profile("2.1.158").unwrap();
        let creds = fixture_creds();
        // All six pinned model ids, including the off-catalog opus-4-7/4-6
        // carry-forward fallbacks - every override row gets a full-string lock,
        // not just the three on-catalog models.
        let cases = [
            ("claude-opus-4-8", BETA_CC_2_1_158_DEFAULT),
            ("claude-opus-4-7", BETA_CC_2_1_158_SONNET),
            ("claude-opus-4-6", BETA_CC_2_1_158_SONNET),
            ("claude-sonnet-4-6", BETA_CC_2_1_158_SONNET),
            ("claude-haiku-4-5", BETA_CC_2_1_158_HAIKU),
            ("claude-haiku-4-5-20251001", BETA_CC_2_1_158_HAIKU),
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
    fn claude_2_1_161_uses_captured_beta_list_per_model() {
        // Independent by-name full-string lock for the NEWEST DEFAULT profile.
        // 2.1.161's beta constants are aliases of 2.1.158's, so this asserts the
        // *resolved* per-model header equals the captured value rather than
        // trusting the alias wiring. Without this, a future edit that repoints
        // 2.1.161 betas (or breaks an override row) would only be caught for the
        // 3 aliases exercised by the default_profile() test. Mirrors the 154/158
        // by-name tests so all three live profiles have parity coverage.
        let profile = resolve_profile("2.1.161").unwrap();
        let creds = fixture_creds();
        let cases = [
            ("claude-opus-4-8", BETA_CC_2_1_161_DEFAULT),
            ("claude-opus-4-7", BETA_CC_2_1_161_SONNET),
            ("claude-opus-4-6", BETA_CC_2_1_161_SONNET),
            ("claude-sonnet-4-6", BETA_CC_2_1_161_SONNET),
            ("claude-haiku-4-5", BETA_CC_2_1_161_HAIKU),
            ("claude-haiku-4-5-20251001", BETA_CC_2_1_161_HAIKU),
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
    fn claude_2_1_162_uses_captured_beta_list_per_model() {
        // Independent by-name full-string lock for the NEWEST DEFAULT profile
        // (2.1.162). Its beta constants are aliases of 2.1.158's, so this asserts
        // the *resolved* per-model header equals the value captured from real
        // claude CLI 2.1.162 traffic (2026-06-04, clean-CWD mitmproxy) rather than
        // trusting the alias wiring. Without this, a future edit that repoints
        // 2.1.162 betas (or breaks an override row) would only be caught for the
        // aliases exercised by the default_profile() test. Mirrors the 154/158/161
        // by-name tests so every live profile has full per-model parity coverage.
        let profile = resolve_profile("2.1.162").unwrap();
        let creds = fixture_creds();
        let cases = [
            ("claude-opus-4-8", BETA_CC_2_1_162_DEFAULT),
            ("claude-opus-4-7", BETA_CC_2_1_162_SONNET),
            ("claude-opus-4-6", BETA_CC_2_1_162_SONNET),
            ("claude-sonnet-4-6", BETA_CC_2_1_162_SONNET),
            ("claude-haiku-4-5", BETA_CC_2_1_162_HAIKU),
            ("claude-haiku-4-5-20251001", BETA_CC_2_1_162_HAIKU),
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
    fn claude_2_1_165_uses_captured_beta_list_per_model() {
        // Independent by-name full-string lock for the NEWEST DEFAULT profile
        // (2.1.165). Its beta constants are aliases of 2.1.158's, so this asserts
        // the *resolved* per-model header equals the value captured from real
        // claude CLI 2.1.165 traffic (2026-06-05, clean-CWD mitmproxy) rather than
        // trusting the alias wiring. Without this, a future edit that repoints
        // 2.1.165 betas (or breaks an override row) would only be caught for the
        // aliases exercised by the default_profile() test. Mirrors the
        // 154/158/161/162 by-name tests so every live profile has full per-model
        // parity coverage.
        let profile = resolve_profile("2.1.165").unwrap();
        let creds = fixture_creds();
        let cases = [
            ("claude-opus-4-8", BETA_CC_2_1_165_DEFAULT),
            ("claude-opus-4-7", BETA_CC_2_1_165_SONNET),
            ("claude-opus-4-6", BETA_CC_2_1_165_SONNET),
            ("claude-sonnet-4-6", BETA_CC_2_1_165_SONNET),
            ("claude-haiku-4-5", BETA_CC_2_1_165_HAIKU),
            ("claude-haiku-4-5-20251001", BETA_CC_2_1_165_HAIKU),
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
    fn claude_2_1_175_uses_captured_beta_list_per_model() {
        let profile = resolve_profile("2.1.175").unwrap();
        let creds = fixture_creds();
        let cases = [
            ("claude-fable-5", BETA_CC_2_1_175_FABLE),
            ("claude-opus-4-8", BETA_CC_2_1_175_OPUS),
            ("claude-sonnet-4-6", BETA_CC_2_1_175_SONNET),
            ("claude-haiku-4-5", BETA_CC_2_1_175_HAIKU),
            ("claude-haiku-4-5-20251001", BETA_CC_2_1_175_HAIKU),
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
    fn bare_aliases_resolve_to_correct_per_model_beta_on_default_profile() {
        // Regression guard for the FULL handler path: a client sending a bare
        // alias ("sonnet"/"haiku"/"opus") must get that model's captured beta,
        // NOT the profile DEFAULT beta. This only holds because outbound_model()
        // canonicalizes the alias *before* the header is built; if that ordering
        // ever regresses, a `sonnet` request would silently leak the opus DEFAULT
        // beta (with context-1m) - an inexact fingerprint. Covers the alias gap
        // the canonical-only test above does not exercise.
        let profile = default_profile();
        let creds = fixture_creds();
        let cases = [
            ("fable", BETA_CC_2_1_186_FABLE, false),
            ("opus", BETA_CC_2_1_186_OPUS, false),
            // 2.1.207: sonnet-5 beta matches explicit opus (mid-conversation-system).
            ("sonnet", BETA_CC_2_1_186_OPUS, false),
            ("haiku", BETA_CC_2_1_186_HAIKU, false),
        ];
        for (alias, expected_beta, has_context_1m) in cases {
            // Mirror the handler: resolve + canonicalize before building headers.
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
        // Lock in the header NAMES we send. Values are mostly captured
        // constants; the dynamic parts are user-agent (versioned), session id,
        // retry count, and client request id.
        //
        // If this test fails, re-run a baseline capture with mitmproxy and
        // update either the constants or the assertion.
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
            "x-client-request-id",
        ];
        for name in expected_names {
            assert!(h.contains_key(name), "missing header `{name}`");
        }

        // Spot-check critical static values.
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

        // EXACT per-profile values, not just "contains". A wrong fingerprint is
        // only caught LATER by Anthropic (a 200 today does not prove exactness),
        // so the unit suite must pin the bytes itself. The stainless versions
        // and User-Agent are version-coupled and a prime silent-drift vector on
        // a rebaseline (carry-forward leaves a stale value that still 200s).
        assert_eq!(
            h.get("x-stainless-package-version")
                .unwrap()
                .to_str()
                .unwrap(),
            profile.stainless_package_version,
            "x-stainless-package-version drifted from profile field on {}",
            profile.name
        );
        assert_eq!(
            h.get("x-stainless-runtime-version")
                .unwrap()
                .to_str()
                .unwrap(),
            profile.stainless_runtime_version,
            "x-stainless-runtime-version drifted from profile field on {}",
            profile.name
        );

        // The default-reply anthropic-beta header must be the profile's EXACT
        // captured string, byte-for-byte (order included), not merely contain a
        // few tokens - beta-flag order and membership are part of the
        // fingerprint. Per-model variants are locked separately by the
        // claude_2_1_<ver>_uses_captured_beta_list_per_model tests; this asserts
        // the no-model reply path for every registered profile.
        let beta = h.get("anthropic-beta").unwrap().to_str().unwrap();
        assert_eq!(
            beta, profile.beta_reply,
            "default-reply beta drifted from profile.beta_reply on {}",
            profile.name
        );
        // The two load-bearing tokens must always be present in any profile's
        // default reply (kept as an explicit invariant beyond the equality
        // above, so a bad future constant fails with a pointed message).
        assert!(
            beta.contains("oauth-2025-04-20"),
            "beta list missing oauth-2025-04-20: {beta}"
        );
        assert!(
            beta.contains("claude-code-20250219"),
            "beta list missing claude-code-20250219: {beta}"
        );

        // Authorization is Bearer-shaped with the OAuth prefix.
        let auth = h.get("authorization").unwrap().to_str().unwrap();
        assert!(auth.starts_with("Bearer sk-ant-oat01-"));

        // User-Agent must equal the profile's exact UA string (version +
        // entrypoint baked in), not just contain the version.
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
    fn default_profile_matches_refreshed_claude_code_baseline() {
        let profile = default_profile();
        assert_eq!(profile.name, "cc-2.1.207-sdk-cli");
        assert_eq!(profile.claude_cli_version, "2.1.207");
        assert_eq!(profile.stainless_package_version, "0.94.0");
        assert_eq!(profile.stainless_runtime_version, "v26.3.0");
        assert_eq!(
            profile.user_agent(),
            "claude-cli/2.1.207 (external, sdk-cli)"
        );
        assert_eq!(
            profile.resolve_model("fable").unwrap().canonical,
            "claude-fable-5"
        );
        assert_eq!(
            profile.resolve_model("opus").unwrap().canonical,
            "claude-opus-4-8"
        );
        assert_eq!(
            profile.resolve_model("sonnet").unwrap().canonical,
            "claude-sonnet-5"
        );
        assert_eq!(
            profile.resolve_model("haiku").unwrap().canonical,
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn profile_registry_resolves_known_selectors() {
        assert_eq!(
            resolve_profile("latest").unwrap().name,
            "cc-2.1.207-sdk-cli"
        );
        assert_eq!(
            resolve_profile("cc-2.1.197-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.197"
        );
        assert_eq!(
            resolve_profile("cc-2.1.186-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.186"
        );
        assert_eq!(
            resolve_profile("2.1.186").unwrap().name,
            "cc-2.1.186-sdk-cli"
        );
        // 2.1.175 is retained for back-compat (no longer the default).
        assert_eq!(
            resolve_profile("cc-2.1.175-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.175"
        );
        assert_eq!(
            resolve_profile("2.1.175").unwrap().name,
            "cc-2.1.175-sdk-cli"
        );
        assert_eq!(
            resolve_profile("cc-2.1.165-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.165"
        );
        assert_eq!(
            resolve_profile("2.1.165").unwrap().name,
            "cc-2.1.165-sdk-cli"
        );
        // 2.1.162 is retained for back-compat (no longer the default).
        assert_eq!(
            resolve_profile("cc-2.1.162-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.162"
        );
        assert_eq!(
            resolve_profile("2.1.162").unwrap().name,
            "cc-2.1.162-sdk-cli"
        );
        assert_eq!(
            resolve_profile("cc-2.1.161-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.161"
        );
        assert_eq!(
            resolve_profile("cc-2.1.158-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.158"
        );
        assert_eq!(
            resolve_profile("2.1.154").unwrap().name,
            "cc-2.1.154-sdk-cli"
        );
        assert_eq!(
            resolve_profile("cc-2.1.150-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.150"
        );
        assert_eq!(
            resolve_profile("2.1.150").unwrap().name,
            "cc-2.1.150-sdk-cli"
        );
        assert_eq!(
            resolve_profile("cc-2.1.142-sdk-cli")
                .unwrap()
                .claude_cli_version,
            "2.1.142"
        );
        assert!(resolve_profile("2.1.138").is_none());
        assert!(resolve_profile("2.0.0").is_none());
    }

    #[test]
    fn profile_registry_names_are_unique() {
        for (idx, profile) in FINGERPRINT_PROFILES.iter().enumerate() {
            assert!(!profile.name.is_empty());
            assert_ne!(profile.name, LATEST_PROFILE_ALIAS);
            assert!(!profile.claude_cli_version.is_empty());
            assert!(!profile.stainless_package_version.is_empty());
            assert!(!profile.aliases.contains(&LATEST_PROFILE_ALIAS));
            assert!(!profile.models.is_empty());
            assert!(crate::models::catalog_contains_unique_names(profile.models));
            for other in FINGERPRINT_PROFILES.iter().skip(idx + 1) {
                assert_ne!(profile.name, other.name);
                for alias in profile.aliases {
                    assert_ne!(*alias, other.name);
                    assert!(!other.aliases.contains(alias));
                }
            }
        }
    }

    #[test]
    fn billing_suffix_matches_claude_code_probe() {
        // Captured from Claude Code 2.1.142 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-05-15.
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.142"), "73b");
        // Captured from Claude Code 2.1.150 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-05-25.
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.150"), "5bd");
        // Captured from Claude Code 2.1.154 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-05-28.
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.154"), "cea");
        // Captured from Claude Code 2.1.161 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-06-03 (live mitmproxy capture: the real
        // billing header read `cc_version=2.1.161.d2b`).
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.161"), "d2b");
        // Captured from Claude Code 2.1.162 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-06-04 (live clean-CWD mitmproxy capture: the
        // real billing header read `cc_version=2.1.162.b87`).
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.162"), "b87");
        // Captured from Claude Code 2.1.165 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-06-05 (live clean-CWD mitmproxy capture: the
        // real billing header read `cc_version=2.1.165.492`).
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.165"), "492");
        // Captured from Claude Code 2.1.175 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-06-12.
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.175"), "174");
        // Captured from Claude Code 2.1.186 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-06-22 (live shared-capture mitmproxy run:
        // the real billing header read `cc_version=2.1.186.a80`).
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.186"), "a80");
        // Captured from Claude Code 2.1.197 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-07-01 (live shared-capture mitmproxy run:
        // the real billing header read `cc_version=2.1.197.c8e`).
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.197"), "c8e");
        // Captured from Claude Code 2.1.207 with CLAUDE_CODE_ENTRYPOINT=sdk-cli
        // and prompt "Say OK" on 2026-07-11 (live shared-capture mitmproxy run:
        // the real billing header read `cc_version=2.1.207.aa4`).
        assert_eq!(claude_code_version_suffix("Say OK", "2.1.207"), "aa4");
        // 2.1.207 (the default) emits NO cch field - the header ends at the
        // entrypoint. Verified byte-for-byte from the live capture.
        assert_eq!(
            default_profile().billing_header_text("Say OK"),
            "x-anthropic-billing-header: cc_version=2.1.207.aa4; cc_entrypoint=sdk-cli;"
        );
        // The 2.1.175 profile still emits the cch placeholder form.
        assert_eq!(
            resolve_profile("2.1.175")
                .unwrap()
                .billing_header_text("Say OK"),
            "x-anthropic-billing-header: cc_version=2.1.175.174; cc_entrypoint=sdk-cli; cch=00000;"
        );
    }

    #[test]
    fn billing_cch_stays_on_known_safe_sentinel() {
        for profile in FINGERPRINT_PROFILES {
            let header = profile.billing_header_text("Say OK");
            // A profile either carries the safe sentinel placeholder (rewritten by
            // finalize_body) or carries no cch field at all (2.1.186+). What is
            // never allowed is a baked-in real (non-sentinel) cch value.
            if header.contains("cch=") {
                assert!(
                    header.contains("cch=00000;"),
                    "profile {} emitted unexpected cch value: {header}",
                    profile.name
                );
            } else {
                assert!(
                    header.ends_with("; cc_entrypoint=sdk-cli;"),
                    "profile {} omitted cch but has an unexpected header tail: {header}",
                    profile.name
                );
            }
        }
    }

    #[test]
    fn finalized_body_writes_profile_checksum() {
        // Exercises the cch REWRITE mechanism, which only the cch-emitting
        // profiles use. The default (2.1.186) has no cch; pin a representative
        // cch profile so this guards the rewrite path itself.
        let profile = resolve_profile("2.1.175").unwrap();
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
        let placeholder = serde_json::to_vec(&body).unwrap();
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();
        let expected = format!(
            "{:05x}",
            claude_code_cch_checksum_skip_models_and_max_tokens(&placeholder)
        );

        assert!(json.contains(&format!("cch={expected};")));
        assert_eq!(json.len(), placeholder.len());
        assert!(!json.contains("cch=00000;"));
    }

    #[test]
    fn omni_serialized_body_cch_snapshot_stays_stable() {
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

        // The default (2.1.186) emits NO cch: the body must carry the bare
        // entrypoint terminator and no cch field anywhere.
        assert!(
            json.contains("cc_entrypoint=sdk-cli;"),
            "default body missing entrypoint terminator"
        );
        assert!(
            !json.contains("cch="),
            "default (2.1.186) body unexpectedly contains a cch field: {json}"
        );

        // Regression guard for the cch REWRITE algorithm on the 2.1.175 profile:
        // the snapshot value is re-derived from the rebuilt binary, not hand-edited,
        // and moves whenever the embedded cc_version suffix or cch algorithm changes.
        let cch_profile = resolve_profile("2.1.175").unwrap();
        let cch_body = serde_json::json!({
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
                    "text": cch_profile.billing_header_text("Say OK"),
                },
                {
                    "type": "text",
                    "text": cch_profile.system_preamble,
                }
            ],
            "stream": false
        });
        let cch_json =
            String::from_utf8(cch_profile.finalize_body_json(&cch_body, &ctx).unwrap()).unwrap();
        let marker = "cc_entrypoint=sdk-cli; cch=";
        let idx = cch_json
            .find(marker)
            .expect("snapshot body missing cch marker");
        let got = &cch_json[idx + marker.len()..idx + marker.len() + 5];
        assert_eq!(
            got, "527d7",
            "2.1.175 snapshot cch changed (re-derive literal)"
        );
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
    fn finalized_body_does_not_rewrite_user_text_sentinel() {
        // The "don't clobber a user-supplied cch=00000" guard only applies to the
        // cch REWRITE path. Pin a cch-emitting profile (2.1.175); the default
        // (2.1.186) performs no rewrite at all.
        let profile = resolve_profile("2.1.175").unwrap();
        let ctx = RequestContext::new_reply();
        let user_text = "leave user cch=00000 untouched";
        let body = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text}
                    ]
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                }
            ]
        });
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();

        assert!(json.contains(user_text));
        assert_eq!(json.matches("cch=00000").count(), 1);
        assert!(json.contains(
            "x-anthropic-billing-header: cc_version=2.1.175.174; cc_entrypoint=sdk-cli; cch="
        ));
        assert!(!json.contains("cc_entrypoint=sdk-cli; cch=00000;"));
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
    fn finalized_body_preserves_non_sentinel_cch() {
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
    fn finalized_body_rewrites_only_first_billing_sentinel() {
        // "Rewrite only the first sentinel" is a property of the cch REWRITE path.
        // Pin a cch-emitting profile (2.1.175); the default (2.1.186) has no cch.
        let profile = resolve_profile("2.1.175").unwrap();
        let ctx = RequestContext::new_reply();
        let body = serde_json::json!({
            "system": [
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                },
                {
                    "type": "text",
                    "text": profile.billing_header_text("Say OK"),
                }
            ],
            "messages": []
        });
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();

        assert_eq!(json.matches("x-anthropic-billing-header:").count(), 2);
        assert_eq!(json.matches("cc_entrypoint=sdk-cli; cch=00000;").count(), 1);
    }

    #[test]
    fn static_cch_mode_preserves_sentinel() {
        let profile = FingerprintProfile {
            name: "test-static",
            aliases: &[],
            claude_cli_version: "2.1.142",
            stainless_package_version: "0.94.0",
            stainless_runtime_version: "v24.3.0",
            entrypoint: "sdk-cli",
            beta_reply: DEFAULT_BETA,
            model_beta_overrides: &[],
            system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
            models: CATALOG_CC_2_1_142,
            preserve_explicit_model: false,
            wire_defaults: WIRE_DEFAULTS_LEGACY,
            model_wire_overrides: &[],
            billing: BILLING_SCHEME_V1_CCH_00000,
        };
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
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();

        assert_eq!(json, serde_json::to_string(&body).unwrap());
        assert!(json.contains("cch=00000;"));
    }

    #[test]
    fn cch_checksum_matches_recovered_claude_code_captures() {
        // Self-consistency over small recovered bodies from the original provider reference; proves
        // the xxh64 impl + sentinel rewrite produces the embedded cch that real
        // Claude Code emitted for those exact normalized shapes.
        let cases = [
            (
                "3bc55",
                r#"{"model":"claude-haiku-4-5","messages":[{"role":"user","content":[{"type":"text","text":"Say OK"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=3bc55;"}],"max_tokens":1,"stream":true}"#,
            ),
            (
                "06b67",
                r#"{"model":"claude-haiku-4-5","messages":[{"role":"user","content":[{"type":"text","text":"Say OK"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=06b67;"}],"max_tokens":2,"stream":true}"#,
            ),
            (
                "9bce0",
                r#"{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"factor"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=9bce0;"},{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}],"stream":true}"#,
            ),
            (
                "4dc19",
                r#"{"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=4dc19;"},{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}],"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"factor"}]}],"stream":true}"#,
            ),
            (
                "7afbb",
                r#"{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"factor"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=7afbb;"},{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=00000;"},{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}],"stream":true}"#,
            ),
            (
                "c159b",
                r#"{"model":"claude-haiku-4-5-20251001","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"WATCHPOINT_MARKER_CCP_CCH_7a9d3f41"}]}],"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=c159b;"},{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"type":"ephemeral"}}],"stream":true}"#,
            ),
        ];

        for (expected, final_body) in cases {
            let placeholder_body =
                final_body.replacen(&format!("cch={expected};"), "cch=00000;", 1);
            assert_eq!(
                format!(
                    "{:05x}",
                    claude_code_cch_checksum(placeholder_body.as_bytes())
                ),
                expected
            );
        }
    }

    #[test]
    fn cch_matches_real_2_1_162_clean_room_capture_vectors() {
        // Breaks the self-consistency circularity: these are REAL Claude Code
        // 2.1.162 request bodies (one per pinned model), captured in clean room
        // and committed under tools/providers/claude/fingerprint/vectors/.
        // Asserts our xxh64 + finalize reproduces the real cch over full body
        // shapes including metadata/thinking/tools/cache_control.
        let vectors = [
            (
                "claude-haiku-4-5",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.162-claude-haiku-4-5.json"
                ),
            ),
            (
                "claude-sonnet-4-6",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.162-claude-sonnet-4-6.json"
                ),
            ),
            (
                "claude-opus-4-8",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.162-claude-opus-4-8.json"
                ),
            ),
        ];
        for (model, body) in vectors {
            let marker = "cc_entrypoint=sdk-cli; cch=";
            let idx = body
                .find(marker)
                .unwrap_or_else(|| panic!("no billing cch marker in {model} vector"));
            let start = idx + marker.len();
            let embedded = &body[start..start + 5];
            let placeholder_body = body.replacen(
                &format!("{marker}{embedded};"),
                &format!("{marker}00000;"),
                1,
            );
            assert_ne!(
                placeholder_body, body,
                "{model}: cch substitution was a no-op"
            );
            assert_eq!(
                format!(
                    "{:05x}",
                    claude_code_cch_checksum(placeholder_body.as_bytes())
                ),
                embedded,
                "cch != real Claude Code 2.1.162 cch for the {model} capture vector"
            );
        }
    }

    #[test]
    fn cch_matches_real_2_1_165_clean_room_capture_vectors() {
        // Sibling coverage for the older 2.1.165 profile. Local capture vectors
        // ensure our impl matches live Claude Code wire cch over rich shapes,
        // not just our synthetic bodies.
        let vectors = [
            (
                "claude-haiku-4-5",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.165-claude-haiku-4-5.json"
                ),
            ),
            (
                "claude-sonnet-4-6",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.165-claude-sonnet-4-6.json"
                ),
            ),
            (
                "claude-opus-4-8",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.165-claude-opus-4-8.json"
                ),
            ),
        ];
        for (model, body) in vectors {
            let marker = "cc_entrypoint=sdk-cli; cch=";
            let idx = body
                .find(marker)
                .unwrap_or_else(|| panic!("no billing cch marker in {model} vector"));
            let start = idx + marker.len();
            let embedded = &body[start..start + 5];
            let placeholder_body = body.replacen(
                &format!("{marker}{embedded};"),
                &format!("{marker}00000;"),
                1,
            );
            assert_ne!(
                placeholder_body, body,
                "{model}: cch substitution was a no-op"
            );
            assert_eq!(
                format!(
                    "{:05x}",
                    claude_code_cch_checksum(placeholder_body.as_bytes())
                ),
                embedded,
                "cch != real Claude Code 2.1.165 cch for the {model} capture vector"
            );
        }
    }

    #[test]
    fn cch_matches_real_2_1_175_clean_room_capture_vectors() {
        let vectors = [
            (
                "claude-fable-5",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.175-claude-fable-5.json"
                ),
            ),
            (
                "claude-haiku-4-5",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.175-claude-haiku-4-5.json"
                ),
            ),
            (
                "claude-sonnet-4-6",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.175-claude-sonnet-4-6.json"
                ),
            ),
            (
                "claude-opus-4-8",
                include_str!(
                    "../../../tools/providers/claude/fingerprint/vectors/vector-2.1.175-claude-opus-4-8.json"
                ),
            ),
        ];
        for (model, body) in vectors {
            let marker = "cc_entrypoint=sdk-cli; cch=";
            let idx = body
                .find(marker)
                .unwrap_or_else(|| panic!("no billing cch marker in {model} vector"));
            let start = idx + marker.len();
            let embedded = &body[start..start + 5];
            let placeholder_body = body.replacen(
                &format!("{marker}{embedded};"),
                &format!("{marker}00000;"),
                1,
            );
            assert_ne!(
                placeholder_body, body,
                "{model}: cch substitution was a no-op"
            );
            assert_eq!(
                format!(
                    "{:05x}",
                    claude_code_cch_checksum_skip_models_and_max_tokens(
                        placeholder_body.as_bytes()
                    )
                ),
                embedded,
                "cch != real Claude Code 2.1.175 cch for the {model} capture vector"
            );
        }
    }

    #[test]
    fn xxh64_matches_independent_small_input_vectors() {
        // Cross-checked against Bun.hash.xxHash64 with the recovered seed.
        assert_eq!(xxh64(b"", CCH_XXH64_SEED), 0xb8b30e7de65b46c5);
        assert_eq!(xxh64(b"abc", CCH_XXH64_SEED), 0xdfc4f4d6913699b6);
        assert_eq!(xxh64(b"hello", CCH_XXH64_SEED), 0xfc8105d2d40e53f1);
        assert_eq!(
            xxh64(b"123456789abcdef", CCH_XXH64_SEED),
            0xd491c6f888304d64
        );
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
        // Cross-checked against Node's JS string indexing/crypto behavior.
        assert_eq!(
            claude_code_version_suffix("abc😀efghijklmnopqrstuv", "2.1.142"),
            "db0"
        );
    }

    #[test]
    fn billing_suffix_treats_sampled_surrogates_like_javascript_string_indices() {
        // Cross-checked against Node's `s[i]` behavior.
        assert_eq!(claude_code_version_suffix("abcd😀😀", "2.1.142"), "052");
    }

    #[test]
    fn billing_header_text_end_to_end_matches_suffix_oracle_for_varied_first_text() {
        // END-TO-END of suffix on the billing_header_text path (the one used
        // for real first-user text before identity). Covers empty/short/emoji
        // etc. Mirrors the original provider reference.
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
            // The default (2.1.186) emits the No-CCH header shape; the suffix
            // itself is what this oracle is guarding across varied first-user text.
            assert_eq!(
                header,
                format!(
                    "x-anthropic-billing-header: cc_version={ver}.{expected_suffix}; \
                     cc_entrypoint=sdk-cli;"
                ),
                "billing_header_text diverged from suffix oracle for first_user_text {input:?}"
            );
        }

        // Same oracle on a cch-emitting profile (2.1.175) to lock the cch-tail
        // shape too.
        let cch_profile = resolve_profile("2.1.175").unwrap();
        let cver = cch_profile.claude_cli_version;
        for input in inputs {
            let expected_suffix = claude_code_version_suffix(input, cver);
            let header = cch_profile.billing_header_text(input);
            assert_eq!(
                header,
                format!(
                    "x-anthropic-billing-header: cc_version={cver}.{expected_suffix}; \
                     cc_entrypoint=sdk-cli; cch=00000;"
                ),
                "cch billing_header_text diverged from suffix oracle for {input:?}"
            );
        }
    }
}
