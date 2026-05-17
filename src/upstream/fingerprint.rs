//! Build the outbound header set for api.anthropic.com requests, mimicking
//! the claude CLI wire fingerprint.
//!
//! Active baseline captured 2026-05-15 against claude CLI 2.1.142
//! SDK 0.94.0. See
//! `tools/fingerprint/BASELINE_HEADERS.md` for the source-of-truth notes.

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use ring::digest;
use uuid::Uuid;

use crate::models::{
    CATALOG_CC_2_1_142, ModelDef, ModelInfo, models_list_from_catalog, resolve_model_in_catalog,
};

use super::credentials::Credentials;

/// Static identity CCP claims on the wire. These values must move together
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
    pub system_preamble: &'static str,
    pub models: &'static [ModelDef],
    pub default_model: &'static str,
    billing: BillingScheme,
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
    Static(&'static str),
    FinalBodyChecksum,
}

impl FingerprintProfile {
    pub fn user_agent(&self) -> String {
        format!(
            "claude-cli/{} (external, {})",
            self.claude_cli_version, self.entrypoint
        )
    }

    pub fn resolve_model(&self, input: &str) -> &'static ModelDef {
        resolve_model_in_catalog(input, self.models, self.default_model)
    }

    pub fn models_list(&self) -> Vec<ModelInfo> {
        models_list_from_catalog(self.models)
    }

    pub fn billing_header_text(&self, first_user_text: &str) -> String {
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
        // Claude Code 2.1.142's cch is a pure body-byte hash. Keep the
        // context in this API so future pinned profiles can add ctx-sensitive
        // behavior without changing call sites.
        match self.billing.cch {
            BillingCchMode::Static(_) => bytes,
            BillingCchMode::FinalBodyChecksum => self.finalize_body_cch_checksum(bytes),
        }
    }

    fn finalize_body_cch_checksum(&self, mut bytes: Vec<u8>) -> Vec<u8> {
        let Some(offset) = self.find_billing_cch_placeholder(&bytes) else {
            return bytes;
        };
        let checksum = claude_code_cch_checksum(&bytes);
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
            let Some(prefix_rel) = find_subslice(&search[cursor..], prefix.as_bytes()) else {
                return None;
            };
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
            BillingCchMode::FinalBodyChecksum => "00000",
        }
    }
}

const BILLING_SUFFIX_SEED_V1: &str = "59cf53e54c78";
const BILLING_SUFFIX_INDICES_V1: [usize; 3] = [4, 7, 20];
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

pub const DEFAULT_PROFILE_NAME: &str = "cc-2.1.142-sdk-cli";
pub const LATEST_PROFILE_ALIAS: &str = "latest";

pub const PROFILE_CLAUDE_2_1_142_SDK_CLI: FingerprintProfile = FingerprintProfile {
    name: DEFAULT_PROFILE_NAME,
    aliases: &["2.1.142"],
    claude_cli_version: "2.1.142",
    stainless_package_version: "0.94.0",
    stainless_runtime_version: "v24.3.0",
    entrypoint: "sdk-cli",
    beta_reply: DEFAULT_BETA,
    system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
    models: CATALOG_CC_2_1_142,
    default_model: "sonnet",
    billing: BILLING_SCHEME_V1_CCH_XXH64_BODY,
};

pub static FINGERPRINT_PROFILES: &[FingerprintProfile] = &[PROFILE_CLAUDE_2_1_142_SDK_CLI];

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

    FINGERPRINT_PROFILES.iter().find(|profile| {
        profile.name == selector || profile.aliases.iter().any(|alias| *alias == selector)
    })
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
    text.starts_with("x-anthropic-billing-header: cc_version=")
        && text.contains("; cc_entrypoint=")
        && text.contains("; cch=")
}

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
        RequestKind::Reply => profile.beta_reply,
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
        assert!(ua.contains(profile.claude_cli_version), "user-agent: {ua}");
        assert!(ua.contains(profile.entrypoint), "user-agent: {ua}");
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
        assert_eq!(profile.name, "cc-2.1.142-sdk-cli");
        assert_eq!(profile.claude_cli_version, "2.1.142");
        assert_eq!(profile.stainless_package_version, "0.94.0");
        assert_eq!(profile.stainless_runtime_version, "v24.3.0");
        assert_eq!(
            profile.user_agent(),
            "claude-cli/2.1.142 (external, sdk-cli)"
        );
        assert_eq!(profile.default_model, "sonnet");
        assert_eq!(profile.resolve_model("opus").canonical, "claude-opus-4-7");
        assert_eq!(
            profile.resolve_model("sonnet").canonical,
            "claude-sonnet-4-6"
        );
        assert_eq!(
            profile.resolve_model("haiku").canonical,
            "claude-haiku-4-5"
        );
    }

    #[test]
    fn profile_registry_resolves_known_selectors() {
        assert_eq!(
            resolve_profile("latest").unwrap().name,
            "cc-2.1.142-sdk-cli"
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
            assert!(profile.models.iter().any(|model| {
                model.cli_name == profile.default_model || model.canonical == profile.default_model
            }));
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
        assert_eq!(
            default_profile().billing_header_text("Say OK"),
            "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=00000;"
        );
    }

    #[test]
    fn billing_cch_stays_on_known_safe_sentinel() {
        for profile in FINGERPRINT_PROFILES {
            let header = profile.billing_header_text("Say OK");
            assert!(
                header.contains("cch=00000;"),
                "profile {} emitted unexpected cch value: {header}",
                profile.name
            );
        }
    }

	#[test]
	fn finalized_body_writes_profile_checksum() {
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
        let placeholder = serde_json::to_vec(&body).unwrap();
        let bytes = profile.finalize_body_json(&body, &ctx).unwrap();
        let json = String::from_utf8(bytes).unwrap();
        let expected = format!("{:05x}", claude_code_cch_checksum(&placeholder));

        assert!(json.contains(&format!("cch={expected};")));
		assert_eq!(json.len(), placeholder.len());
		assert!(!json.contains("cch=00000;"));
	}

	#[test]
	fn ccp_serialized_body_cch_snapshot_stays_stable() {
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

		assert!(json.contains("cc_entrypoint=sdk-cli; cch=37cb4;"));
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
        let profile = default_profile();
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
            "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch="
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
        let profile = default_profile();
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
            system_preamble: CLAUDE_CODE_SYSTEM_PREAMBLE,
            models: CATALOG_CC_2_1_142,
            default_model: "sonnet",
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
        // Cross-checked against Node's `s[i]` behavior. Sampling can make
        // surrogate halves from different original code points adjacent;
        // JavaScript encodes that joined string as a valid surrogate pair.
        assert_eq!(claude_code_version_suffix("abcd😀😀", "2.1.142"), "052");
    }
}
