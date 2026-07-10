//! Shared operational request-complete status text for bin handlers.
//!
//! Emits a single `info!` twin next to terminal `record_*` sites, plus a
//! separate conversation-log finish renderer from the same snapshot. Field
//! rules and finish_reason total order are locked by the status-text plan.

use std::fmt::Write as _;

use tracing::{debug, info};

/// Control-flow flags that select `finish_reason` without inventing values.
#[derive(Debug, Clone, Default)]
pub struct FinishSite {
    /// True once a body/stream path that could produce Finish was entered.
    pub entered_body: bool,
    /// True only when the incomplete `record_error` arm ran.
    pub incomplete: bool,
    /// Site has no finish concept (e.g. count_tokens success would use this).
    pub no_finish_concept: bool,
    /// Last observed finish/stop reason, already mapped to plain vocabulary.
    pub finish_latch: Option<String>,
}

/// Outcome mirrors stats: `record_response` → Ok, `record_error` → Error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Error,
}

/// Snapshot gathered once next to a terminal `record_*` call.
#[derive(Debug, Clone)]
pub struct RequestCompleteParams {
    /// Exact model string passed into the adjacent `record_*` call.
    pub model: String,
    pub finish: FinishSite,
    pub duration_ms: f64,
    pub outcome: Outcome,
    /// `Some` only when usage was observed (including true zero).
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read: Option<u64>,
    pub cache_creation: Option<u64>,
    pub ttft_ms: Option<f64>,
    /// Only when client raw model differs from the stats model argument.
    pub requested_model: Option<String>,
    /// Raw error text; redacted fail-closed before emission.
    pub error: Option<String>,
}

impl RequestCompleteParams {
    pub fn ok(model: impl Into<String>, finish: FinishSite, duration_ms: f64) -> Self {
        Self {
            model: model.into(),
            finish,
            duration_ms,
            outcome: Outcome::Ok,
            input_tokens: None,
            output_tokens: None,
            cache_read: None,
            cache_creation: None,
            ttft_ms: None,
            requested_model: None,
            error: None,
        }
    }

    pub fn error(model: impl Into<String>, finish: FinishSite, duration_ms: f64) -> Self {
        Self {
            model: model.into(),
            finish,
            duration_ms,
            outcome: Outcome::Error,
            input_tokens: None,
            output_tokens: None,
            cache_read: None,
            cache_creation: None,
            ttft_ms: None,
            requested_model: None,
            error: None,
        }
    }

    pub fn with_tokens(mut self, input: Option<u64>, output: Option<u64>) -> Self {
        self.input_tokens = input;
        self.output_tokens = output;
        self
    }

    #[allow(dead_code)] // API for call sites that observe cache presence.
    pub fn with_cache(mut self, read: Option<u64>, creation: Option<u64>) -> Self {
        self.cache_read = read;
        self.cache_creation = creation;
        self
    }

    pub fn with_ttft_ms(mut self, ttft_ms: Option<f64>) -> Self {
        self.ttft_ms = ttft_ms;
        self
    }

    pub fn with_requested_model(mut self, requested: Option<String>) -> Self {
        self.requested_model = requested;
        self
    }

    pub fn with_error(mut self, error: Option<String>) -> Self {
        self.error = error;
        self
    }
}

/// Resolve finish_reason with the plan's total order (first match wins).
pub fn resolve_finish_reason(finish: &FinishSite, outcome: Outcome) -> String {
    let latch = finish
        .finish_latch
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    // 1. incomplete arm
    if finish.incomplete {
        return match latch {
            Some(plain) => plain.to_string(),
            None => "incomplete".to_string(),
        };
    }
    // 2. finish latch / response finish_reason present
    if let Some(plain) = latch {
        return plain.to_string();
    }
    // 3. never entered body/stream path that could produce Finish
    if !finish.entered_body {
        return "none".to_string();
    }
    // 4. no_finish_concept + success
    if finish.no_finish_concept && outcome == Outcome::Ok {
        return "n/a".to_string();
    }
    // 5. success on finish-capable site with empty latch
    if outcome == Outcome::Ok {
        return "unknown".to_string();
    }
    // 6. error with empty latch after body entry, or pure transport
    "none".to_string()
}

/// Emit `requested_model=` only when client raw differs from stats model arg.
pub fn requested_model_if_differs(stats_model: &str, client_raw: &str) -> Option<String> {
    if client_raw != stats_model {
        Some(client_raw.to_string())
    } else {
        None
    }
}

/// Map Anthropic wire `stop_reason` to the plain OAI-style vocabulary used on
/// the canonical path. Unknown values pass through as plain strings.
pub fn map_anthropic_stop_reason(anth: &str) -> String {
    match anth {
        "end_turn" | "stop_sequence" | "pause_turn" => "stop".into(),
        "max_tokens" => "length".into(),
        "tool_use" => "tool_calls".into(),
        "refusal" => "content_filter".into(),
        other => other.into(),
    }
}

/// Fail-closed error redactor for optional `error=` on complete lines.
///
/// Redact known secret shapes first, then truncate. If residual text still
/// looks secret-bearing, omit the field entirely.
pub fn redact_error_for_log(raw: &str) -> Option<String> {
    let mut s = raw.to_string();

    // URL query secrets: ?token=...&api_key=...
    s = redact_url_query_secrets(&s);
    // Auth-like header values
    s = redact_auth_headers(&s);
    // JSON body secret fields
    s = redact_json_secret_fields(&s);
    // Bearer / basic / sk- style tokens when still present as bare values
    s = redact_bare_secrets(&s);

    if still_secret_shaped(&s) {
        return None;
    }

    Some(truncate_chars(&s, 200))
}

fn redact_url_query_secrets(input: &str) -> String {
    // Match query/fragment secret keys case-insensitively.
    let keys = [
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "key",
        "secret",
        "password",
        "passwd",
        "authorization",
        "auth",
        "client_secret",
        "x-api-key",
    ];
    let mut out = input.to_string();
    for key in keys {
        // key=value until & or whitespace or end
        let lower = out.to_ascii_lowercase();
        let needle = format!("{key}=");
        let mut search_from = 0;
        while let Some(rel) = lower[search_from..].find(&needle) {
            let start = search_from + rel;
            // require start-of-string, ? or & before key
            if start > 0 {
                let prev = out.as_bytes()[start - 1];
                if prev != b'?' && prev != b'&' && prev != b' ' && prev != b'\t' {
                    search_from = start + needle.len();
                    continue;
                }
            }
            let val_start = start + needle.len();
            let val_end = out[val_start..]
                .find(|c: char| c == '&' || c == ' ' || c == '\t' || c == '"' || c == '\'')
                .map(|i| val_start + i)
                .unwrap_or(out.len());
            out.replace_range(val_start..val_end, "REDACTED");
            // restart from after replacement (length changed)
            break;
        }
        // multi-pass until stable for this key
        let mut guard = 0;
        loop {
            let before = out.clone();
            let lower = out.to_ascii_lowercase();
            let needle = format!("{key}=");
            if let Some(rel) = lower.find(&needle) {
                let start = rel;
                if start > 0 {
                    let prev = out.as_bytes()[start - 1];
                    if prev != b'?' && prev != b'&' && prev != b' ' && prev != b'\t' {
                        // not a query key; stop trying this key to avoid loops
                        break;
                    }
                }
                let val_start = start + needle.len();
                if out[val_start..].starts_with("REDACTED") {
                    break;
                }
                let val_end = out[val_start..]
                    .find(|c: char| c == '&' || c == ' ' || c == '\t' || c == '"' || c == '\'')
                    .map(|i| val_start + i)
                    .unwrap_or(out.len());
                out.replace_range(val_start..val_end, "REDACTED");
            } else {
                break;
            }
            if out == before {
                break;
            }
            guard += 1;
            if guard > 16 {
                break;
            }
        }
    }
    out
}

fn redact_auth_headers(input: &str) -> String {
    // Authorization: Bearer xxx / Basic xxx / raw header-like — redact the full
    // credential material (scheme + token), not only the first whitespace token.
    let markers = [
        "authorization:",
        "x-api-key:",
        "api-key:",
        "x-auth-token:",
    ];
    let mut out = input.to_string();
    for marker in markers {
        let mut guard = 0;
        loop {
            let lower = out.to_ascii_lowercase();
            let Some(idx) = lower.find(marker) else {
                break;
            };
            let after = idx + marker.len();
            // Skip whitespace after the colon, then take the rest of the
            // credential run (until quote/comma/newline/end). Include scheme
            // words like Bearer/Basic so the token cannot trail as bare text.
            let value_part = out[after..].trim_start();
            let ws = out[after..].len() - value_part.len();
            let value_start = after + ws;
            let value_end = value_part
                .find(|c: char| c == '"' || c == '\'' || c == ',' || c == '\n' || c == '\r')
                .map(|i| value_start + i)
                .unwrap_or(out.len());
            // Also stop at a trailing English word boundary when value continues
            // with " leaked" style prose: keep space-separated residual after
            // the credential by redacting scheme+token only (two tokens max).
            let value_slice = &out[value_start..value_end];
            let cred_end = {
                let mut parts = value_slice.split_whitespace();
                let first = parts.next().unwrap_or("");
                let second = parts.next();
                match second {
                    Some(tok)
                        if first.eq_ignore_ascii_case("bearer")
                            || first.eq_ignore_ascii_case("basic") =>
                    {
                        // scheme + token
                        value_start
                            + first.len()
                            + value_slice[first.len()..]
                                .find(tok)
                                .map(|i| i + tok.len())
                                .unwrap_or(first.len())
                    }
                    _ => value_start + first.len(),
                }
            };
            if out[value_start..cred_end].starts_with("REDACTED") {
                break;
            }
            out.replace_range(value_start..cred_end, "REDACTED");
            guard += 1;
            if guard > 16 {
                break;
            }
        }
    }
    out
}

fn redact_json_secret_fields(input: &str) -> String {
    let keys = [
        "api_key",
        "apiKey",
        "access_token",
        "accessToken",
        "refresh_token",
        "authorization",
        "password",
        "secret",
        "client_secret",
        "token",
    ];
    let mut out = input.to_string();
    for key in keys {
        // "key":"value" or "key": "value"
        let patterns = [
            format!("\"{key}\""),
            format!("'{key}'"),
        ];
        for pat in patterns {
            let mut guard = 0;
            while let Some(idx) = out.find(&pat) {
                let after_key = idx + pat.len();
                let rest = &out[after_key..];
                let Some(colon_rel) = rest.find(':') else {
                    break;
                };
                let after_colon = after_key + colon_rel + 1;
                let value_part = out[after_colon..].trim_start();
                let ws = out[after_colon..].len() - value_part.len();
                let value_start = after_colon + ws;
                if value_part.starts_with('"') {
                    if let Some(end_rel) = value_part[1..].find('"') {
                        let value_end = value_start + 1 + end_rel + 1;
                        out.replace_range(value_start..value_end, "\"REDACTED\"");
                    } else {
                        break;
                    }
                } else {
                    break;
                }
                guard += 1;
                if guard > 16 {
                    break;
                }
            }
        }
    }
    out
}

fn redact_bare_secrets(input: &str) -> String {
    let mut out = input.to_string();
    // Bearer <token>
    out = replace_prefixed_token(&out, "Bearer ");
    out = replace_prefixed_token(&out, "bearer ");
    out = replace_prefixed_token(&out, "Basic ");
    out = replace_prefixed_token(&out, "basic ");
    // sk-... OpenAI-style keys
    out = replace_sk_tokens(&out);
    out
}

fn replace_prefixed_token(input: &str, prefix: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find(prefix) {
        out.push_str(&rest[..idx + prefix.len()]);
        let after = &rest[idx + prefix.len()..];
        let end = after
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == ',' || c == '}')
            .unwrap_or(after.len());
        if end > 0 {
            out.push_str("REDACTED");
            rest = &after[end..];
        } else {
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn replace_sk_tokens(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 3 <= bytes.len() && &bytes[i..i + 3] == b"sk-" {
            // ensure start boundary
            let boundary_ok = i == 0
                || !bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_' && bytes[i - 1] != b'-';
            if boundary_ok {
                let mut j = i + 3;
                while j < bytes.len()
                    && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' || bytes[j] == b'-')
                {
                    j += 1;
                }
                if j - i >= 12 {
                    out.push_str("sk-REDACTED");
                    i = j;
                    continue;
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn still_secret_shaped(s: &str) -> bool {
    // Residual high-entropy secret indicators after redaction.
    if s.contains("sk-") && s.contains("sk-REDACTED") {
        // redacted form is fine
    } else if s.contains("sk-") {
        // unredacted sk- left
        return true;
    }
    // Long base64-ish token after auth keywords that we failed to scrub
    let lower = s.to_ascii_lowercase();
    for marker in ["authorization=", "api_key=", "access_token=", "password="] {
        if let Some(idx) = lower.find(marker) {
            let after = &s[idx + marker.len()..];
            let token: String = after
                .chars()
                .take_while(|c| !c.is_whitespace() && *c != '&' && *c != '"' && *c != '\'')
                .collect();
            if token.len() >= 16 && token != "REDACTED" && !token.starts_with("REDACTED") {
                return true;
            }
        }
    }
    false
}

fn truncate_chars(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(3)).collect();
    out.push_str("...");
    out
}

/// Format the complete line for tests / conv-log. Never panics.
pub fn format_request_complete_fields(params: &RequestCompleteParams) -> String {
    let finish_reason = resolve_finish_reason(&params.finish, params.outcome);
    let outcome = match params.outcome {
        Outcome::Ok => "ok",
        Outcome::Error => "error",
    };
    let mut out = String::new();
    let _ = write!(
        out,
        "model={} finish_reason={} duration_ms={:.1} outcome={}",
        params.model, finish_reason, params.duration_ms, outcome
    );
    if let Some(n) = params.input_tokens {
        let _ = write!(out, " input_tokens={n}");
    }
    if let Some(n) = params.output_tokens {
        let _ = write!(out, " output_tokens={n}");
    }
    if let Some(n) = params.cache_read {
        let _ = write!(out, " cache_read={n}");
    }
    if let Some(n) = params.cache_creation {
        let _ = write!(out, " cache_creation={n}");
    }
    if let Some(t) = params.ttft_ms {
        let _ = write!(out, " ttft_ms={t:.1}");
    }
    if let Some(ref rm) = params.requested_model {
        let _ = write!(out, " requested_model={rm}");
    }
    if let Some(ref raw_err) = params.error {
        if let Some(redacted) = redact_error_for_log(raw_err) {
            let _ = write!(out, " error={redacted}");
        }
    }
    out
}

/// Emit the operational request-complete `info!` line. Never panics.
///
/// Required fields are structured (so `finish_reason` keeps log-color cues).
/// Optional token/cache/ttft/error keys are appended to the message only when
/// present, never as Debug `Some(...)`.
pub fn log_request_complete(params: &RequestCompleteParams) {
    let finish_reason = resolve_finish_reason(&params.finish, params.outcome);
    let outcome = match params.outcome {
        Outcome::Ok => "ok",
        Outcome::Error => "error",
    };
    let duration = format!("{:.1}", params.duration_ms);

    let mut optional = String::new();
    if let Some(n) = params.input_tokens {
        let _ = write!(optional, " input_tokens={n}");
    }
    if let Some(n) = params.output_tokens {
        let _ = write!(optional, " output_tokens={n}");
    }
    if let Some(n) = params.cache_read {
        let _ = write!(optional, " cache_read={n}");
    }
    if let Some(n) = params.cache_creation {
        let _ = write!(optional, " cache_creation={n}");
    }
    if let Some(t) = params.ttft_ms {
        let _ = write!(optional, " ttft_ms={t:.1}");
    }
    if let Some(ref rm) = params.requested_model {
        let _ = write!(optional, " requested_model={rm}");
    }
    if let Some(ref raw_err) = params.error {
        if let Some(redacted) = redact_error_for_log(raw_err) {
            let _ = write!(optional, " error={redacted}");
        }
    }

    // Build the message string so optional keys do not become a separate
    // structured field named `optional`.
    let message = format!("request_complete{optional}");
    info!(
        model = %params.model,
        finish_reason = %finish_reason,
        duration_ms = %duration,
        outcome = %outcome,
        "{message}",
        message = message.as_str()
    );
}

/// Separate conversation-log finish renderer from the same snapshot.
pub fn format_conv_log_finish_summary(params: &RequestCompleteParams) -> String {
    format_request_complete_fields(params)
}

/// Thin debug breadcrumb at request-span open. Model often unknown here.
pub fn log_request_start(model: Option<&str>) {
    match model {
        Some(m) if !m.is_empty() => {
            debug!(model = %m, "request_start");
        }
        _ => {
            debug!("request_start");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_reason_total_order_incomplete_empty_latch() {
        // WHY: incomplete is a control-flow arm, not inferred from emptiness.
        let finish = FinishSite {
            entered_body: true,
            incomplete: true,
            no_finish_concept: false,
            finish_latch: None,
        };
        assert_eq!(
            resolve_finish_reason(&finish, Outcome::Error),
            "incomplete"
        );
    }

    #[test]
    fn finish_reason_incomplete_with_latch_keeps_plain() {
        // WHY: Finish-then-error (incomplete arm with prior latch) must not
        // clobber the already-observed terminal reason.
        let finish = FinishSite {
            entered_body: true,
            incomplete: true,
            no_finish_concept: false,
            finish_latch: Some("stop".into()),
        };
        assert_eq!(resolve_finish_reason(&finish, Outcome::Error), "stop");
    }

    #[test]
    fn finish_reason_plain_latch_wins() {
        let finish = FinishSite {
            entered_body: true,
            incomplete: false,
            no_finish_concept: false,
            finish_latch: Some("tool_calls".into()),
        };
        assert_eq!(
            resolve_finish_reason(&finish, Outcome::Ok),
            "tool_calls"
        );
    }

    #[test]
    fn finish_reason_pre_body_is_none() {
        let finish = FinishSite {
            entered_body: false,
            incomplete: false,
            no_finish_concept: false,
            finish_latch: None,
        };
        assert_eq!(resolve_finish_reason(&finish, Outcome::Error), "none");
    }

    #[test]
    fn finish_reason_no_finish_concept_is_na_on_ok() {
        let finish = FinishSite {
            entered_body: true,
            incomplete: false,
            no_finish_concept: true,
            finish_latch: None,
        };
        assert_eq!(resolve_finish_reason(&finish, Outcome::Ok), "n/a");
    }

    #[test]
    fn finish_reason_success_empty_latch_is_unknown() {
        // WHY: finish-capable success without a reason must not look like a
        // pre-send failure (`none`) or incomplete.
        let finish = FinishSite {
            entered_body: true,
            incomplete: false,
            no_finish_concept: false,
            finish_latch: None,
        };
        assert_eq!(resolve_finish_reason(&finish, Outcome::Ok), "unknown");
    }

    #[test]
    fn finish_reason_error_empty_latch_after_body_is_none() {
        let finish = FinishSite {
            entered_body: true,
            incomplete: false,
            no_finish_concept: false,
            finish_latch: None,
        };
        assert_eq!(resolve_finish_reason(&finish, Outcome::Error), "none");
    }

    #[test]
    fn format_omits_token_keys_when_usage_missing() {
        // WHY: never print 0 to mean unknown; omit keys entirely.
        let params = RequestCompleteParams::ok(
            "claude:m",
            FinishSite {
                entered_body: true,
                finish_latch: Some("stop".into()),
                ..Default::default()
            },
            12.5,
        );
        let s = format_request_complete_fields(&params);
        assert!(s.contains("finish_reason=stop"));
        assert!(s.contains("outcome=ok"));
        assert!(!s.contains("input_tokens"));
        assert!(!s.contains("output_tokens"));
        assert!(!s.contains("cache_read"));
        assert!(!s.contains("Some("));
    }

    #[test]
    fn format_emits_true_zero_tokens_when_observed() {
        let params = RequestCompleteParams::ok(
            "m",
            FinishSite {
                entered_body: true,
                finish_latch: Some("stop".into()),
                ..Default::default()
            },
            1.0,
        )
        .with_tokens(Some(0), Some(0));
        let s = format_request_complete_fields(&params);
        assert!(s.contains("input_tokens=0"));
        assert!(s.contains("output_tokens=0"));
        assert!(!s.contains("Some("));
    }

    #[test]
    fn redact_url_query_secrets() {
        let raw = "upstream https://api.example.com/v1?api_key=supersecret&x=1 failed";
        let redacted = redact_error_for_log(raw).expect("safe after redaction");
        assert!(!redacted.contains("supersecret"), "{redacted}");
        assert!(redacted.contains("REDACTED"), "{redacted}");
    }

    #[test]
    fn redact_auth_like_headers() {
        let raw = "header Authorization: Bearer abcdefghijklmnop leaked";
        let redacted = redact_error_for_log(raw).expect("safe after redaction");
        assert!(!redacted.contains("abcdefghijklmnop"), "{redacted}");
        assert!(redacted.contains("REDACTED"), "{redacted}");
    }

    #[test]
    fn redact_json_body_fragments() {
        let raw = r#"upstream body {"api_key":"sekrit-value-here","ok":true}"#;
        let redacted = redact_error_for_log(raw).expect("safe after redaction");
        assert!(!redacted.contains("sekrit-value-here"), "{redacted}");
        assert!(redacted.contains("REDACTED"), "{redacted}");
    }

    #[test]
    fn redact_fail_closed_on_residual_secret() {
        // A bare high-entropy token next to an auth key that the scrubber
        // cannot confidently rewrite must omit error= entirely.
        let raw = "password=this_is_a_long_secret_value_xx";
        // Our URL query redactor should catch password= after treating it
        // carefully; if something slips, fail-closed returns None.
        match redact_error_for_log(raw) {
            Some(s) => assert!(!s.contains("this_is_a_long_secret_value_xx"), "{s}"),
            None => {}
        }
    }

    #[test]
    fn requested_model_only_when_differs() {
        assert_eq!(
            requested_model_if_differs("claude:sonnet", "sonnet"),
            Some("sonnet".into())
        );
        assert_eq!(requested_model_if_differs("m", "m"), None);
    }

    #[test]
    fn map_anthropic_stop_reason_vocabulary() {
        assert_eq!(map_anthropic_stop_reason("end_turn"), "stop");
        assert_eq!(map_anthropic_stop_reason("tool_use"), "tool_calls");
        assert_eq!(map_anthropic_stop_reason("max_tokens"), "length");
        assert_eq!(map_anthropic_stop_reason("weird"), "weird");
    }
}
