//! Shared environment-variable and custom-header helpers.
//!
//! These are used by the binary and every provider to read optional
//! configuration from the environment. Keeping one copy avoids the drift that
//! four hand-maintained duplicates invite.

use axum::http::{HeaderName, HeaderValue};

/// Read an environment variable, trim surrounding whitespace, and treat an
/// empty (or all-whitespace) value as absent. Returns `None` when the variable
/// is unset or blank.
pub fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Parse a `Name: value` custom-header block, one header per line. Blank lines
/// are skipped; each header's name and value are trimmed and validated as a
/// legal HTTP header name/value. Returns a human-readable error string on the
/// first malformed line so callers can wrap it in their own error type.
///
/// Providers that omitted parse-time validation still validated at header
/// insertion, so requiring it here is behavior-preserving and fails a bad
/// header one step earlier.
pub fn parse_custom_headers(raw: &str) -> Result<Vec<(String, String)>, String> {
    let mut headers = Vec::new();
    for line in raw.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| "custom header must be formatted as `Name: value`".to_string())?;
        let name = name.trim();
        let value = value.trim();
        if name.is_empty() || value.is_empty() {
            return Err("custom header name and value must both be non-empty".into());
        }
        HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| format!("invalid custom header name `{name}`"))?;
        HeaderValue::from_str(value)
            .map_err(|_| format!("invalid custom header value for `{name}`"))?;
        headers.push((name.to_string(), value.to_string()));
    }
    Ok(headers)
}

/// Read a custom-header env var (via [`env_nonempty`]) and parse it with
/// [`parse_custom_headers`]. An unset or blank var yields an empty list. The
/// error is returned as a string so each caller maps it to its own error type.
pub fn headers_from_env(env_name: &str) -> Result<Vec<(String, String)>, String> {
    let Some(raw) = env_nonempty(env_name) else {
        return Ok(Vec::new());
    };
    parse_custom_headers(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_custom_headers_accepts_valid_lines_and_skips_blanks() {
        // WHY: operators supply one `Name: value` per line; surrounding blank
        // lines and padding are cosmetic and must not produce phantom headers.
        let parsed = parse_custom_headers("X-One: a\n\n  X-Two :  b  \n").unwrap();
        assert_eq!(
            parsed,
            vec![
                ("X-One".to_string(), "a".to_string()),
                ("X-Two".to_string(), "b".to_string()),
            ]
        );
    }

    #[test]
    fn parse_custom_headers_rejects_missing_colon() {
        // WHY: a line with no `:` is a config typo; failing loudly beats
        // silently dropping the intended header.
        let err = parse_custom_headers("X-One a").unwrap_err();
        assert!(err.contains("Name: value"), "{err}");
    }

    #[test]
    fn parse_custom_headers_rejects_blank_name_or_value() {
        // WHY: an empty name or value cannot form a usable header; reject it at
        // parse time rather than emitting a degenerate header.
        assert!(parse_custom_headers(": v").is_err());
        assert!(parse_custom_headers("X-One:").is_err());
    }

    #[test]
    fn parse_custom_headers_rejects_illegal_header_name() {
        // WHY: the value must be a legal HTTP header; validating here fails a bad
        // header one step before insertion, with a clear message.
        let err = parse_custom_headers("Bad Name: v").unwrap_err();
        assert!(err.contains("invalid custom header name"), "{err}");
    }
}
