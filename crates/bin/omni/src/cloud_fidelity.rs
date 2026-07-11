//! Opt-in strict cloud-fidelity helpers and always-on dual-credential detection.
//!
//! Pure functions only: header shape checks for Anthropic paths, and OpenAI
//! chat token-cap field rules for known model families. Wired from `main` auth
//! middleware and `/v1/chat/completions`.

use axum::http::HeaderMap;

/// Which Anthropic client auth header shape is required under strict mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum AnthropicAuthScheme {
    /// Stock Anthropic API key clients: `x-api-key` only.
    #[default]
    #[value(name = "api-key")]
    ApiKey,
    /// OAuth / Bearer clients: `Authorization: Bearer` only.
    #[value(name = "oauth")]
    Oauth,
}

/// OpenAI chat token-cap family for strict body validation.
///
/// Classification keys on **OpenAI-compatible wire model ids** on the chat
/// completions surface (id shape), not on which backend provider will serve the
/// request. That matches cloud-contract testing of the OpenAI request form.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TokenCapFamily {
    /// o1 / o3 / o4 family: require `max_completion_tokens`, reject `max_tokens`.
    Reasoning,
    /// gpt-* chat family: require `max_tokens`, reject `max_completion_tokens`.
    Plain,
    /// Unknown / non-OpenAI ids: skip strict token-cap enforcement (fail-open).
    Unknown,
}

/// Parsed Anthropic-path credential presence from request headers.
///
/// Scans **all** `authorization` and `x-api-key` values (not just the first).
/// Bearer scheme matching is case-insensitive per HTTP auth scheme rules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AnthropicCredPresence {
    /// First non-empty Bearer token (case-insensitive scheme), if any.
    pub bearer_token: Option<String>,
    /// First non-empty `x-api-key` value, if any.
    pub x_api_key: Option<String>,
    /// True when both Bearer material and `x-api-key` material appear.
    pub dual: bool,
    /// True when more than one distinct non-empty Bearer token appears.
    pub ambiguous_bearer: bool,
    /// True when more than one distinct non-empty `x-api-key` appears.
    pub ambiguous_x_api_key: bool,
    /// True when a non-empty `Authorization` value is present that is not a
    /// valid Bearer credential (e.g. `Basic …`, bare tokens, empty scheme,
    /// Bearer with internal whitespace).
    pub non_bearer_authorization: bool,
    /// True when an `x-api-key` header value could not be decoded as UTF-8
    /// (still credential material for dual / strict shape checks).
    pub opaque_x_api_key: bool,
    /// True when an Authorization value used the Bearer scheme but the token
    /// was empty/malformed (still Bearer-scheme material for dual detection).
    pub malformed_bearer: bool,
}

impl AnthropicCredPresence {
    pub fn has_bearer(&self) -> bool {
        self.bearer_token.is_some()
    }

    pub fn has_x_api_key(&self) -> bool {
        self.x_api_key.is_some() || self.opaque_x_api_key
    }

    /// Any credential-shaped material was presented (including malformed Authorization).
    pub fn any_credential_material(&self) -> bool {
        self.has_bearer()
            || self.has_x_api_key()
            || self.non_bearer_authorization
            || self.ambiguous_bearer
            || self.ambiguous_x_api_key
            || self.malformed_bearer
    }
}

/// Parse a single Authorization header value as a Bearer token.
///
/// Scheme match is case-insensitive. The token is the remainder after the first
/// whitespace and must be non-empty with **no internal whitespace** (RFC 6750
/// b64token-style single token; rejects `Bearer a b` and comma-stuffed multi-creds
/// that include spaces).
pub fn parse_bearer_value(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let mut parts = value.splitn(2, char::is_whitespace);
    let scheme = parts.next()?;
    let rest = parts.next()?.trim();
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    if rest.is_empty() || rest.chars().any(char::is_whitespace) {
        return None;
    }
    Some(rest)
}

/// True when the value looks like a Bearer scheme (case-insensitive) even if the
/// token part is missing/malformed. Used to flag shape errors under strict mode.
pub fn is_bearer_scheme_prefix(value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return false;
    }
    let scheme = value.split(char::is_whitespace).next().unwrap_or("");
    scheme.eq_ignore_ascii_case("bearer")
}

/// Best-effort Bearer detection on raw header bytes when UTF-8 decode fails.
/// Matches optional leading spaces, case-insensitive `Bearer`, then whitespace
/// or end (so `Bearer` / `Bearer <opaque>` both count as Bearer-scheme material).
fn authorization_bytes_look_like_bearer(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    const BEARER: &[u8] = b"bearer";
    if bytes.len().saturating_sub(i) < BEARER.len() {
        return false;
    }
    let scheme = &bytes[i..i + BEARER.len()];
    if !scheme.eq_ignore_ascii_case(BEARER) {
        return false;
    }
    let after = i + BEARER.len();
    after == bytes.len() || bytes[after].is_ascii_whitespace()
}

/// Collect all Anthropic-relevant credential material from headers.
pub fn anthropic_cred_presence(headers: &HeaderMap) -> AnthropicCredPresence {
    let mut presence = AnthropicCredPresence::default();
    let mut bearer_seen: Vec<String> = Vec::new();
    let mut xkey_seen: Vec<String> = Vec::new();

    for val in headers.get_all("authorization") {
        let Ok(raw) = val.to_str() else {
            // Non-UTF-8 Authorization: still inspect ASCII Bearer prefix so dual
            // with x-api-key cannot be bypassed by opaque bytes after "Bearer ".
            presence.non_bearer_authorization = true;
            if authorization_bytes_look_like_bearer(val.as_bytes()) {
                presence.malformed_bearer = true;
            }
            continue;
        };
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        if let Some(tok) = parse_bearer_value(raw) {
            let tok = tok.to_string();
            if !bearer_seen.iter().any(|t| t == &tok) {
                bearer_seen.push(tok);
            }
        } else {
            // Includes Basic, bare tokens, and malformed Bearer (empty / spaced token).
            presence.non_bearer_authorization = true;
            if is_bearer_scheme_prefix(raw) {
                presence.malformed_bearer = true;
            }
        }
    }

    for val in headers.get_all("x-api-key") {
        let Ok(raw) = val.to_str() else {
            // Non-UTF-8 values are still credential material (must not drop dual detection).
            presence.opaque_x_api_key = true;
            continue;
        };
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let key = raw.to_string();
        if !xkey_seen.iter().any(|k| k == &key) {
            xkey_seen.push(key);
        }
    }

    if bearer_seen.len() > 1 {
        presence.ambiguous_bearer = true;
    }
    if xkey_seen.len() > 1 {
        presence.ambiguous_x_api_key = true;
    }
    presence.bearer_token = bearer_seen.into_iter().next();
    presence.x_api_key = xkey_seen.into_iter().next();

    // Dual: any Bearer-scheme material (valid or malformed) + any x-api-key material
    // (UTF-8 or opaque non-UTF-8). Must not drop dual on decode/shape edge cases.
    let has_bearer_material = presence.bearer_token.is_some() || presence.malformed_bearer;
    let has_x_material = presence.x_api_key.is_some() || presence.opaque_x_api_key;
    presence.dual = has_bearer_material && has_x_material;

    // Mixed opaque + UTF-8 (or valid + malformed Bearer) is multi-value ambiguity.
    if presence.opaque_x_api_key && presence.x_api_key.is_some() {
        presence.ambiguous_x_api_key = true;
    }
    if presence.malformed_bearer && presence.bearer_token.is_some() {
        presence.ambiguous_bearer = true;
    }
    presence
}

/// Under strict mode, when any credential material is presented, enforce the
/// configured single-header scheme. Dual / multi-value ambiguity is handled
/// separately before this when possible.
pub fn strict_anthropic_scheme_ok(
    scheme: AnthropicAuthScheme,
    presence: &AnthropicCredPresence,
) -> Result<(), String> {
    if presence.ambiguous_bearer || presence.ambiguous_x_api_key {
        return Err(
            "Ambiguous credentials: multiple distinct Authorization Bearer or x-api-key values were provided. Send exactly one."
                .into(),
        );
    }
    if presence.dual {
        return Err(
            "Ambiguous credentials: both non-empty x-api-key and Authorization Bearer headers were provided. Send exactly one."
                .into(),
        );
    }
    if !presence.any_credential_material() {
        return Ok(());
    }
    if presence.non_bearer_authorization {
        return Err(
            "Strict cloud fidelity: Authorization must use the Bearer scheme (case-insensitive) with a non-empty token, or be omitted."
                .into(),
        );
    }
    match scheme {
        AnthropicAuthScheme::ApiKey => {
            if presence.has_x_api_key() && !presence.has_bearer() {
                Ok(())
            } else if presence.has_bearer() {
                Err(
                    "Strict cloud fidelity (api-key): provide credentials via x-api-key only, not Authorization Bearer."
                        .into(),
                )
            } else {
                // Credential material without either parsed form should already
                // have been rejected as non_bearer_authorization.
                Err(
                    "Strict cloud fidelity (api-key): provide credentials via x-api-key only."
                        .into(),
                )
            }
        }
        AnthropicAuthScheme::Oauth => {
            if presence.has_bearer() && !presence.has_x_api_key() {
                Ok(())
            } else if presence.has_x_api_key() {
                Err(
                    "Strict cloud fidelity (oauth): provide credentials via Authorization Bearer only, not x-api-key."
                        .into(),
                )
            } else {
                Err(
                    "Strict cloud fidelity (oauth): provide credentials via Authorization Bearer only."
                        .into(),
                )
            }
        }
    }
}

/// Classify a resolved OpenAI-facing model id for strict token-cap rules.
///
/// Reasoning: whole-segment `o1` / `o3` / `o4` or those prefixes with `-`
/// (e.g. `o1-mini`). Does not match mid-string (`gpt-4o` is plain).
/// Plain: `gpt-` prefix. Everything else is Unknown (fail-open).
pub fn classify_openai_token_cap_family(model: &str) -> TokenCapFamily {
    let model = model.trim();
    if is_openai_reasoning_model(model) {
        return TokenCapFamily::Reasoning;
    }
    if model.starts_with("gpt-") {
        return TokenCapFamily::Plain;
    }
    TokenCapFamily::Unknown
}

fn is_openai_reasoning_model(model: &str) -> bool {
    matches!(model, "o1" | "o3" | "o4")
        || model.starts_with("o1-")
        || model.starts_with("o3-")
        || model.starts_with("o4-")
}

/// Enforce strict OpenAI token-cap field shape for a known family.
pub fn validate_strict_token_caps(
    family: TokenCapFamily,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
) -> Result<(), String> {
    match family {
        TokenCapFamily::Unknown => Ok(()),
        TokenCapFamily::Reasoning => {
            if max_tokens.is_some() {
                return Err(
                    "Strict cloud fidelity: reasoning models (o1/o3/o4) require max_completion_tokens and must not set max_tokens."
                        .into(),
                );
            }
            if max_completion_tokens.is_none() {
                return Err(
                    "Strict cloud fidelity: reasoning models (o1/o3/o4) require max_completion_tokens."
                        .into(),
                );
            }
            Ok(())
        }
        TokenCapFamily::Plain => {
            if max_completion_tokens.is_some() {
                return Err(
                    "Strict cloud fidelity: plain chat models (gpt-*) require max_tokens and must not set max_completion_tokens."
                        .into(),
                );
            }
            if max_tokens.is_none() {
                return Err(
                    "Strict cloud fidelity: plain chat models (gpt-*) require max_tokens.".into(),
                );
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (k, v) in pairs {
            map.append(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        map
    }

    #[test]
    fn dual_credentials_both_nonempty() {
        let h = headers(&[
            ("x-api-key", "key-a"),
            ("authorization", "Bearer token-b"),
        ]);
        assert!(anthropic_cred_presence(&h).dual);
    }

    #[test]
    fn dual_credentials_case_insensitive_bearer() {
        let h = headers(&[
            ("x-api-key", "key-a"),
            ("authorization", "bearer token-b"),
        ]);
        assert!(anthropic_cred_presence(&h).dual);
        let h2 = headers(&[
            ("x-api-key", "key-a"),
            ("authorization", "BEARER token-b"),
        ]);
        assert!(anthropic_cred_presence(&h2).dual);
    }

    #[test]
    fn dual_credentials_false_for_single_or_empty() {
        assert!(!anthropic_cred_presence(&headers(&[("x-api-key", "key-a")])).dual);
        assert!(!anthropic_cred_presence(&headers(&[("authorization", "Bearer token-b")])).dual);
        // Empty Bearer token is still Bearer-scheme material: dual with x-api-key.
        assert!(anthropic_cred_presence(&headers(&[
            ("x-api-key", "key-a"),
            ("authorization", "Bearer "),
        ]))
        .dual);
        // Whitespace-only x-api-key is not material; Bearer alone is not dual.
        assert!(!anthropic_cred_presence(&headers(&[
            ("x-api-key", "   "),
            ("authorization", "Bearer token-b"),
        ]))
        .dual);
        assert!(!anthropic_cred_presence(&headers(&[])).dual);
        // Non-Bearer Authorization is not Bearer-scheme material (not dual).
        assert!(!anthropic_cred_presence(&headers(&[
            ("x-api-key", "key-a"),
            ("authorization", "Basic abc"),
        ]))
        .dual);
    }

    #[test]
    fn dual_credentials_scans_all_authorization_values() {
        // First Authorization is Basic; second is Bearer. Must still dual with x-api-key.
        let h = headers(&[
            ("authorization", "Basic abc"),
            ("authorization", "Bearer token-b"),
            ("x-api-key", "key-a"),
        ]);
        let p = anthropic_cred_presence(&h);
        assert_eq!(p.bearer_token.as_deref(), Some("token-b"));
        assert!(p.non_bearer_authorization);
        assert!(p.dual);
    }

    #[test]
    fn dual_credentials_scans_all_x_api_key_values() {
        let h = headers(&[
            ("x-api-key", "   "),
            ("x-api-key", "key-a"),
            ("authorization", "Bearer token-b"),
        ]);
        assert!(anthropic_cred_presence(&h).dual);
    }

    #[test]
    fn parse_bearer_value_case_and_whitespace() {
        assert_eq!(parse_bearer_value("Bearer tok"), Some("tok"));
        assert_eq!(parse_bearer_value("bearer tok"), Some("tok"));
        assert_eq!(parse_bearer_value("  BEARER   tok  "), Some("tok"));
        assert_eq!(parse_bearer_value("Basic abc"), None);
        assert_eq!(parse_bearer_value("Bearer "), None);
        assert_eq!(parse_bearer_value(""), None);
        // Multi-token remainder is malformed (not a single b64token-style value).
        assert_eq!(parse_bearer_value("Bearer good evil"), None);
        assert_eq!(parse_bearer_value("Bearer good, Bearer evil"), None);
    }

    #[test]
    fn dual_with_malformed_bearer_and_x_api_key() {
        let h = headers(&[
            ("x-api-key", "key-a"),
            ("authorization", "Bearer good evil"),
        ]);
        let p = anthropic_cred_presence(&h);
        assert!(p.malformed_bearer);
        assert!(p.dual);
    }

    #[test]
    fn dual_with_opaque_x_api_key_and_bearer() {
        // Non-UTF-8 x-api-key must still count as credential material.
        let mut map = HeaderMap::new();
        map.append(
            axum::http::HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer token-b"),
        );
        map.append(
            axum::http::HeaderName::from_static("x-api-key"),
            HeaderValue::from_bytes(&[0xff, 0xfe, 0xfd]).unwrap(),
        );
        let p = anthropic_cred_presence(&map);
        assert!(p.opaque_x_api_key);
        assert!(p.dual);
    }

    #[test]
    fn mixed_opaque_and_utf8_x_api_key_is_ambiguous() {
        let mut map = HeaderMap::new();
        map.append(
            axum::http::HeaderName::from_static("x-api-key"),
            HeaderValue::from_static("key-a"),
        );
        map.append(
            axum::http::HeaderName::from_static("x-api-key"),
            HeaderValue::from_bytes(&[0xff, 0xfe, 0xfd]).unwrap(),
        );
        let p = anthropic_cred_presence(&map);
        assert!(p.opaque_x_api_key);
        assert_eq!(p.x_api_key.as_deref(), Some("key-a"));
        assert!(p.ambiguous_x_api_key);
        let err = strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, &p).unwrap_err();
        assert!(err.to_lowercase().contains("ambiguous"), "{err}");
    }

    #[test]
    fn dual_with_opaque_bearer_bytes_and_x_api_key() {
        // Non-UTF-8 Authorization that still has ASCII Bearer prefix + x-api-key.
        let mut raw = b"Bearer ".to_vec();
        raw.extend_from_slice(&[0xff, 0xfe, 0xfd]);
        let mut map = HeaderMap::new();
        map.append(
            axum::http::HeaderName::from_static("authorization"),
            HeaderValue::from_bytes(&raw).unwrap(),
        );
        map.append(
            axum::http::HeaderName::from_static("x-api-key"),
            HeaderValue::from_static("key-a"),
        );
        let p = anthropic_cred_presence(&map);
        assert!(p.malformed_bearer);
        assert!(p.dual);
    }

    #[test]
    fn strict_scheme_api_key_accepts_x_api_key_only() {
        let ok = AnthropicCredPresence {
            x_api_key: Some("k".into()),
            ..Default::default()
        };
        assert!(strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, &ok).is_ok());
        let bearer_only = AnthropicCredPresence {
            bearer_token: Some("t".into()),
            ..Default::default()
        };
        let err =
            strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, &bearer_only).unwrap_err();
        assert!(err.contains("api-key"), "{err}");
        assert!(err.contains("x-api-key"), "{err}");
        assert!(strict_anthropic_scheme_ok(
            AnthropicAuthScheme::ApiKey,
            &AnthropicCredPresence::default()
        )
        .is_ok());
    }

    #[test]
    fn strict_scheme_rejects_non_bearer_authorization() {
        let p = AnthropicCredPresence {
            non_bearer_authorization: true,
            ..Default::default()
        };
        let err = strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, &p).unwrap_err();
        assert!(err.to_lowercase().contains("bearer"), "{err}");
        let err2 = strict_anthropic_scheme_ok(AnthropicAuthScheme::Oauth, &p).unwrap_err();
        assert!(err2.to_lowercase().contains("bearer"), "{err2}");
    }

    #[test]
    fn strict_scheme_oauth_accepts_bearer_only() {
        let ok = AnthropicCredPresence {
            bearer_token: Some("t".into()),
            ..Default::default()
        };
        assert!(strict_anthropic_scheme_ok(AnthropicAuthScheme::Oauth, &ok).is_ok());
        let x_only = AnthropicCredPresence {
            x_api_key: Some("k".into()),
            ..Default::default()
        };
        let err = strict_anthropic_scheme_ok(AnthropicAuthScheme::Oauth, &x_only).unwrap_err();
        assert!(err.contains("oauth"), "{err}");
        assert!(err.contains("Bearer"), "{err}");
    }

    #[test]
    fn strict_scheme_rejects_dual_if_reached() {
        let dual = AnthropicCredPresence {
            bearer_token: Some("t".into()),
            x_api_key: Some("k".into()),
            dual: true,
            ..Default::default()
        };
        let err = strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, &dual).unwrap_err();
        assert!(err.to_lowercase().contains("ambiguous"), "{err}");
    }

    #[test]
    fn classify_reasoning_and_plain() {
        for id in ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4", "o4-mini"] {
            assert_eq!(
                classify_openai_token_cap_family(id),
                TokenCapFamily::Reasoning,
                "{id}"
            );
        }
        for id in ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4o-mini"] {
            assert_eq!(
                classify_openai_token_cap_family(id),
                TokenCapFamily::Plain,
                "{id}"
            );
        }
        // Must not treat gpt-4o as reasoning (o mid-string).
        assert_ne!(
            classify_openai_token_cap_family("gpt-4o"),
            TokenCapFamily::Reasoning
        );
        // Fail-open unknowns.
        for id in ["claude-sonnet-4-6", "grok-4.3", "sonnet", "o10-mini", "xo1-mini"] {
            assert_eq!(
                classify_openai_token_cap_family(id),
                TokenCapFamily::Unknown,
                "{id}"
            );
        }
    }

    #[test]
    fn validate_token_caps_reasoning() {
        assert!(validate_strict_token_caps(TokenCapFamily::Reasoning, None, Some(64)).is_ok());
        assert!(
            validate_strict_token_caps(TokenCapFamily::Reasoning, Some(64), None)
                .unwrap_err()
                .contains("max_completion_tokens")
        );
        assert!(
            validate_strict_token_caps(TokenCapFamily::Reasoning, Some(64), Some(64))
                .unwrap_err()
                .contains("max_tokens")
        );
        assert!(
            validate_strict_token_caps(TokenCapFamily::Reasoning, None, None)
                .unwrap_err()
                .contains("max_completion_tokens")
        );
    }

    #[test]
    fn validate_token_caps_plain() {
        assert!(validate_strict_token_caps(TokenCapFamily::Plain, Some(64), None).is_ok());
        assert!(
            validate_strict_token_caps(TokenCapFamily::Plain, None, Some(64))
                .unwrap_err()
                .contains("max_tokens")
        );
        assert!(
            validate_strict_token_caps(TokenCapFamily::Plain, Some(64), Some(64))
                .unwrap_err()
                .contains("max_completion_tokens")
        );
        assert!(
            validate_strict_token_caps(TokenCapFamily::Plain, None, None)
                .unwrap_err()
                .contains("max_tokens")
        );
    }

    #[test]
    fn validate_token_caps_unknown_is_noop() {
        assert!(validate_strict_token_caps(TokenCapFamily::Unknown, None, None).is_ok());
        assert!(validate_strict_token_caps(TokenCapFamily::Unknown, Some(1), Some(2)).is_ok());
    }
}
