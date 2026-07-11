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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TokenCapFamily {
    /// o1 / o3 / o4 family: require `max_completion_tokens`, reject `max_tokens`.
    Reasoning,
    /// gpt-* chat family: require `max_tokens`, reject `max_completion_tokens`.
    Plain,
    /// Unknown / non-OpenAI ids: skip strict token-cap enforcement (fail-open).
    Unknown,
}

/// Extract non-empty Bearer token (trimmed), if present.
pub fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Extract non-empty `x-api-key` value (trimmed), if present.
pub fn x_api_key(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Both non-empty `x-api-key` and non-empty `Authorization: Bearer` present.
/// Always-on reject for Anthropic paths (independent of strict mode / key set).
pub fn dual_anthropic_credentials(headers: &HeaderMap) -> bool {
    x_api_key(headers).is_some() && bearer_token(headers).is_some()
}

/// Under strict mode, when any credential is presented, enforce the configured
/// single-header scheme. Dual credentials are rejected separately before this.
pub fn strict_anthropic_scheme_ok(
    scheme: AnthropicAuthScheme,
    has_x_api_key: bool,
    has_bearer: bool,
) -> Result<(), String> {
    if !has_x_api_key && !has_bearer {
        return Ok(());
    }
    if has_x_api_key && has_bearer {
        return Err(
            "Ambiguous credentials: both non-empty x-api-key and Authorization Bearer headers were provided. Send exactly one."
                .into(),
        );
    }
    match scheme {
        AnthropicAuthScheme::ApiKey => {
            if has_x_api_key {
                Ok(())
            } else {
                Err(
                    "Strict cloud fidelity (api-key): provide credentials via x-api-key only, not Authorization Bearer."
                        .into(),
                )
            }
        }
        AnthropicAuthScheme::Oauth => {
            if has_bearer {
                Ok(())
            } else {
                Err(
                    "Strict cloud fidelity (oauth): provide credentials via Authorization Bearer only, not x-api-key."
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
        assert!(dual_anthropic_credentials(&h));
    }

    #[test]
    fn dual_credentials_false_for_single_or_empty() {
        assert!(!dual_anthropic_credentials(&headers(&[("x-api-key", "key-a")])));
        assert!(!dual_anthropic_credentials(&headers(&[
            ("authorization", "Bearer token-b")
        ])));
        assert!(!dual_anthropic_credentials(&headers(&[
            ("x-api-key", "key-a"),
            ("authorization", "Bearer "),
        ])));
        assert!(!dual_anthropic_credentials(&headers(&[
            ("x-api-key", "   "),
            ("authorization", "Bearer token-b"),
        ])));
        assert!(!dual_anthropic_credentials(&headers(&[])));
        // Non-Bearer Authorization does not count as Bearer credential.
        assert!(!dual_anthropic_credentials(&headers(&[
            ("x-api-key", "key-a"),
            ("authorization", "Basic abc"),
        ])));
    }

    #[test]
    fn strict_scheme_api_key_accepts_x_api_key_only() {
        assert!(strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, true, false).is_ok());
        let err = strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, false, true).unwrap_err();
        assert!(err.contains("api-key"), "{err}");
        assert!(err.contains("x-api-key"), "{err}");
        assert!(strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, false, false).is_ok());
    }

    #[test]
    fn strict_scheme_oauth_accepts_bearer_only() {
        assert!(strict_anthropic_scheme_ok(AnthropicAuthScheme::Oauth, false, true).is_ok());
        let err = strict_anthropic_scheme_ok(AnthropicAuthScheme::Oauth, true, false).unwrap_err();
        assert!(err.contains("oauth"), "{err}");
        assert!(err.contains("Bearer"), "{err}");
        assert!(strict_anthropic_scheme_ok(AnthropicAuthScheme::Oauth, false, false).is_ok());
    }

    #[test]
    fn strict_scheme_rejects_dual_if_reached() {
        let err = strict_anthropic_scheme_ok(AnthropicAuthScheme::ApiKey, true, true).unwrap_err();
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
        assert!(
            validate_strict_token_caps(TokenCapFamily::Reasoning, None, Some(64)).is_ok()
        );
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
        assert!(
            validate_strict_token_caps(TokenCapFamily::Unknown, Some(1), Some(2)).is_ok()
        );
    }
}
