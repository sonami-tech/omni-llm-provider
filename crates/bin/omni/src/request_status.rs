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

/// When only plain `u64` usage fields exist (no Option presence), all-zero is
/// ambiguous: it may mean "usage missing" (`CanonicalUsage::default()`, Grok
/// `from_xai` with `usage: None`). Emit `None` for both when every field is 0;
/// otherwise emit `Some` for input/output (true zeros on a sibling field are OK
/// once another field proves usage was present).
pub fn tokens_from_plain_usage(
    input: u64,
    output: u64,
    cache_read: u64,
    cache_creation: u64,
) -> (Option<u64>, Option<u64>) {
    if input == 0 && output == 0 && cache_read == 0 && cache_creation == 0 {
        (None, None)
    } else {
        (Some(input), Some(output))
    }
}

/// Fail-closed error redactor for optional `error=` on complete lines.
///
/// Redact known secret shapes first, sanitize control characters for single-line
/// logs, then truncate. If residual text still looks secret-bearing, omit the
/// field entirely.
pub fn redact_error_for_log(raw: &str) -> Option<String> {
    // Unescape nested JSON dumps (e.g. body {\"api_key\":\"…\"}) so secret-key
    // matchers see plain "key" forms. Cosmetic for non-secret text; required for
    // fail-closed redaction of escaped-in-string JSON secrets.
    let mut s = raw.replace("\\\"", "\"").replace("\\'", "'");

    // URL query secrets: ?token=...&api_key=...
    s = redact_url_query_secrets(&s);
    // Auth-like header values
    s = redact_auth_headers(&s);
    // JSON body secret fields
    s = redact_json_secret_fields(&s);
    // Bearer / basic / sk- / xai- style tokens when still present as bare values
    s = redact_bare_secrets(&s);

    if still_secret_shaped(&s) {
        return None;
    }

    // After redaction, force a single log line (no embedded newlines/tabs).
    s = sanitize_control_chars(&s);
    Some(truncate_chars(&s, 200))
}

/// Replace ASCII controls (0x00-0x1F, DEL) with spaces and collapse whitespace runs.
fn sanitize_control_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        let is_ctrl = c.is_control() || (c as u32) < 0x20;
        if is_ctrl || c == ' ' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out
}

/// Query/fragment secret keys (lowercase). Shared by redaction and residual checks.
const URL_QUERY_SECRET_KEYS: &[&str] = &[
    "api_key",
    "apikey",
    "access_token",
    "accesstoken", // camelCase OAuth forms (matched case-insensitively)
    "refresh_token",
    "refreshtoken",
    "id_token",
    "idtoken",
    "token",
    "key",
    "secret",
    "password",
    "passwd",
    "authorization",
    "auth",
    "client_secret",
    "clientsecret",
    "x-api-key",
];

fn is_query_key_boundary(prev: u8) -> bool {
    // `?` / `&` start query pairs; `#` starts a fragment (same key=value form).
    // `\n`/`\r` so multi-line error bodies still match keys before control-char collapse.
    prev == b'?'
        || prev == b'&'
        || prev == b'#'
        || prev == b' '
        || prev == b'\t'
        || prev == b'\n'
        || prev == b'\r'
}

/// Sanitize client-controlled model strings for status text: strip controls and
/// scrub known secret shapes (e.g. pasted sk-/xai- keys as the model field).
fn sanitize_model_for_status(raw: &str) -> String {
    let cleaned = sanitize_control_chars(raw);
    match redact_error_for_log(&cleaned) {
        Some(s) => s,
        None => {
            // Known secret shape residual: do not emit the raw secret as model=.
            if still_secret_shaped(&cleaned) {
                "<redacted>".to_string()
            } else {
                cleaned
            }
        }
    }
}

/// True only when `token` is exactly `REDACTED` (not `REDACTEDsuffix` glued on).
fn is_exact_redacted_placeholder(token: &str) -> bool {
    token == "REDACTED"
}

/// True when the value at `s` is the exact placeholder followed by end or a
/// delimiter — not `REDACTED` glued to more secret material.
fn is_exact_redacted_value_at(s: &str) -> bool {
    if !s.starts_with("REDACTED") {
        return false;
    }
    let rest = &s["REDACTED".len()..];
    rest.is_empty()
        || rest.starts_with(|c: char| {
            c == '&'
                || c == ' '
                || c == '\t'
                || c == '"'
                || c == '\''
                || c == ','
                || c == '\n'
                || c == '\r'
        })
}

fn redact_url_query_secrets(input: &str) -> String {
    // Walk every occurrence of each secret key; advance past already-REDACTED
    // values so later duplicates (e.g. ?api_key=SEC1&api_key=SEC2) are scrubbed.
    let mut out = input.to_string();
    for key in URL_QUERY_SECRET_KEYS {
        let needle = format!("{key}=");
        let mut search_from = 0;
        let mut guard = 0;
        loop {
            let lower = out.to_ascii_lowercase();
            let Some(rel) = lower[search_from..].find(&needle) else {
                break;
            };
            let start = search_from + rel;
            // require start-of-string, ? or & (or whitespace) before key
            if start > 0 {
                let prev = out.as_bytes()[start - 1];
                if !is_query_key_boundary(prev) {
                    search_from = start + needle.len();
                    continue;
                }
            }
            let val_start = start + needle.len();
            if is_exact_redacted_value_at(&out[val_start..]) {
                // Already scrubbed (exact placeholder); advance past it.
                search_from = val_start + "REDACTED".len();
                continue;
            }
            let val_end = out[val_start..]
                .find(|c: char| c == '&' || c == ' ' || c == '\t' || c == '"' || c == '\'')
                .map(|i| val_start + i)
                .unwrap_or(out.len());
            out.replace_range(val_start..val_end, "REDACTED");
            search_from = val_start + "REDACTED".len();
            guard += 1;
            if guard > 64 {
                break;
            }
        }
    }
    out
}

fn redact_auth_headers(input: &str) -> String {
    // Authorization: / x-api-key: / etc. — for ANY scheme (Bearer, Basic, Token,
    // ApiKey, raw token), redact the full credential material in one shot from
    // the first non-ws after the colon through end of the credential run
    // (quote/comma/newline). Do not leave trailing scheme or token fragments.
    let markers = ["authorization:", "x-api-key:", "api-key:", "x-auth-token:"];
    let mut out = input.to_string();
    for marker in markers {
        let mut guard = 0;
        let mut search_from = 0;
        loop {
            let lower = out.to_ascii_lowercase();
            let Some(rel) = lower[search_from..].find(marker) else {
                break;
            };
            let idx = search_from + rel;
            let after = idx + marker.len();
            let value_part = out[after..].trim_start();
            let ws = out[after..].len() - value_part.len();
            let value_start = after + ws;
            if value_part.is_empty() {
                break;
            }
            if is_exact_redacted_value_at(value_part) {
                // Already scrubbed (exact placeholder); advance past it.
                search_from = value_start + "REDACTED".len();
                continue;
            }
            let value_end = value_part
                .find(|c: char| c == '"' || c == '\'' || c == ',' || c == '\n' || c == '\r')
                .map(|i| value_start + i)
                .unwrap_or(out.len());
            out.replace_range(value_start..value_end, "REDACTED");
            search_from = value_start + "REDACTED".len();
            guard += 1;
            if guard > 16 {
                break;
            }
        }
    }
    out
}

/// JSON object keys that carry secret material (lowercase forms only).
/// Matching is case-insensitive so API_KEY, Authorization, ApiKey, X-API-Key,
/// accessToken, etc. all hit the same entries.
const JSON_SECRET_KEYS: &[&str] = &[
    "apikey",
    "api_key",
    "x-api-key",
    "x_api_key",
    "access_token",
    "accesstoken",
    "refresh_token",
    "refreshtoken",
    "client_secret",
    "clientsecret",
    "authorization",
    "password",
    "secret",
    "token",
];

/// End index for an unterminated quote-started JSON secret value.
/// Consumes through end of input, or a hard line boundary if present.
fn unterminated_json_value_end(value_part: &str, value_start: usize) -> usize {
    value_part
        .find(|c: char| c == '\n' || c == '\r')
        .map(|i| value_start + i)
        .unwrap_or(value_start + value_part.len())
}

/// Relative index of the first unescaped `quote` in `inner` (content after the
/// opening quote). Treats `\\` and `\"` (or `\'`) so a quote after an odd-length
/// backslash run is escaped; even-length runs leave the quote as closer.
/// Returns `None` when no unescaped closer exists.
fn find_unescaped_quote_end(inner: &str, quote: u8) -> Option<usize> {
    let bytes = inner.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            // Consume backslash + following byte (if any) as one escape unit.
            i = i.saturating_add(2);
            continue;
        }
        if bytes[i] == quote {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// True when `after_close` begins with secret-shaped material (alphanumeric /
/// `_` / `-`). Used to fail-closed on partial redaction like `"REDACTED"suffix`.
fn residual_after_redacted_close(after_close: &str) -> bool {
    after_close
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn redact_json_secret_fields(input: &str) -> String {
    // Walk every occurrence case-insensitively; advance past already-REDACTED
    // values so later duplicates (e.g. "api_key":"a","API_KEY":"b") are scrubbed.
    // Scan a lowercased copy for key positions; redact in the original string
    // (ASCII keys preserve byte offsets under to_ascii_lowercase).
    let mut out = input.to_string();
    for key in JSON_SECRET_KEYS {
        for quote in ['"', '\''] {
            let needle = format!("{quote}{key}{quote}");
            let mut search_from = 0;
            let mut guard = 0;
            loop {
                let lower = out.to_ascii_lowercase();
                let Some(rel) = lower[search_from..].find(&needle) else {
                    break;
                };
                let idx = search_from + rel;
                let after_key = idx + needle.len();
                let rest = &out[after_key..];
                let Some(colon_rel) = rest.find(':') else {
                    search_from = after_key;
                    continue;
                };
                let after_colon = after_key + colon_rel + 1;
                let value_part = out[after_colon..].trim_start();
                let ws = out[after_colon..].len() - value_part.len();
                let value_start = after_colon + ws;
                if value_part.starts_with("\"REDACTED\"") || value_part.starts_with("'REDACTED'") {
                    // Already scrubbed; advance past this occurrence. Residual
                    // alphanumerics glued after the closer are caught by
                    // still_secret_shaped (fail-closed omit).
                    search_from = value_start + "\"REDACTED\"".len();
                    continue;
                }
                if value_part.starts_with('"') {
                    if let Some(end_rel) = find_unescaped_quote_end(&value_part[1..], b'"') {
                        let value_end = value_start + 1 + end_rel + 1;
                        out.replace_range(value_start..value_end, "\"REDACTED\"");
                        search_from = value_start + "\"REDACTED\"".len();
                    } else {
                        // Unterminated string after a secret key: fail-closed —
                        // redact from opening quote through end (or hard boundary).
                        let value_end = unterminated_json_value_end(value_part, value_start);
                        out.replace_range(value_start..value_end, "\"REDACTED\"");
                        search_from = value_start + "\"REDACTED\"".len();
                    }
                } else if value_part.starts_with('\'') {
                    if let Some(end_rel) = find_unescaped_quote_end(&value_part[1..], b'\'') {
                        let value_end = value_start + 1 + end_rel + 1;
                        out.replace_range(value_start..value_end, "\"REDACTED\"");
                        search_from = value_start + "\"REDACTED\"".len();
                    } else {
                        let value_end = unterminated_json_value_end(value_part, value_start);
                        out.replace_range(value_start..value_end, "\"REDACTED\"");
                        search_from = value_start + "\"REDACTED\"".len();
                    }
                } else {
                    // Non-string JSON value after a secret key (number/bool/null
                    // or bare token). Fail-closed: scrub the run rather than
                    // leave e.g. {"api_key":123456789012345678} live.
                    let bare_end = value_part
                        .find(|c: char| {
                            c == ','
                                || c == '}'
                                || c == ']'
                                || c == ' '
                                || c == '\t'
                                || c == '\n'
                                || c == '\r'
                        })
                        .map(|i| value_start + i)
                        .unwrap_or(out.len());
                    if bare_end > value_start {
                        out.replace_range(value_start..bare_end, "REDACTED");
                        search_from = value_start + "REDACTED".len();
                    } else {
                        search_from = after_key;
                    }
                    continue;
                }
                guard += 1;
                if guard > 64 {
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
    // sk-... OpenAI-style keys and xai-... xAI keys
    out = replace_dashed_key_tokens(&out, b"sk-", "sk-REDACTED", 12);
    out = replace_dashed_key_tokens(&out, b"xai-", "xai-REDACTED", 12);
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

/// Redact bare `sk-…` / `xai-…` keys: start boundary + min total length.
///
/// Walks by UTF-8 char boundaries so multi-byte characters (e.g. `…`) are
/// preserved in the passthrough path; never uses `bytes[i] as char`.
fn replace_dashed_key_tokens(
    input: &str,
    prefix: &[u8],
    replacement: &str,
    min_total_len: usize,
) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + prefix.len() <= bytes.len()
            && input.is_char_boundary(i)
            && &bytes[i..i + prefix.len()] == prefix
        {
            let boundary_ok = i == 0
                || (!bytes[i - 1].is_ascii_alphanumeric()
                    && bytes[i - 1] != b'_'
                    && bytes[i - 1] != b'-');
            if boundary_ok {
                let mut j = i + prefix.len();
                while j < bytes.len()
                    && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' || bytes[j] == b'-')
                {
                    j += 1;
                }
                if j - i >= min_total_len {
                    out.push_str(replacement);
                    i = j;
                    continue;
                }
            }
        }
        // Advance one UTF-8 character so multi-byte sequences stay intact.
        let ch = input[i..].chars().next().expect("i is a char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Minimum residual query/JSON value length that still looks like a secret.
const RESIDUAL_SECRET_MIN_LEN: usize = 8;

fn still_secret_shaped(s: &str) -> bool {
    // Residual unredacted sk- / xai- key material (not the REDACTED form).
    if has_unredacted_dashed_key(s, "sk-", "sk-REDACTED") {
        return true;
    }
    if has_unredacted_dashed_key(s, "xai-", "xai-REDACTED") {
        return true;
    }
    // Bare JWT-looking blobs (base64url header starting with eyJ, min ~20 chars).
    if has_bare_jwt_blob(s) {
        return true;
    }
    let lower = s.to_ascii_lowercase();
    // Residual query-key=value for every secret key (not just a short subset).
    // Walk all occurrences so a later unredacted value after REDACTED is caught.
    for key in URL_QUERY_SECRET_KEYS {
        let needle = format!("{key}=");
        let mut search_from = 0;
        while let Some(rel) = lower[search_from..].find(&needle) {
            let start = search_from + rel;
            if start > 0 {
                let prev = s.as_bytes()[start - 1];
                if !is_query_key_boundary(prev) {
                    search_from = start + needle.len();
                    continue;
                }
            }
            let after = &s[start + needle.len()..];
            let token: String = after
                .chars()
                .take_while(|c| !c.is_whitespace() && *c != '&' && *c != '"' && *c != '\'')
                .collect();
            // Exact REDACTED only — REDACTEDgluedsecret is still secret-shaped.
            if token.len() >= RESIDUAL_SECRET_MIN_LEN && !is_exact_redacted_placeholder(&token) {
                return true;
            }
            search_from = start + needle.len() + token.len().max(1);
        }
    }
    // Residual JSON "key":"value" for secret keys (case-insensitive; non-REDACTED).
    // Scan lowercased copy for key positions; inspect values in the original.
    // Closing quotes respect escapes so \" cannot truncate a live secret value.
    for key in JSON_SECRET_KEYS {
        for quote in ['"', '\''] {
            let pat = format!("{quote}{key}{quote}");
            let mut search_from = 0;
            while let Some(rel) = lower[search_from..].find(&pat) {
                let idx = search_from + rel;
                let after_key = idx + pat.len();
                let rest = &s[after_key..];
                let Some(colon_rel) = rest.find(':') else {
                    search_from = after_key;
                    continue;
                };
                let after_colon = after_key + colon_rel + 1;
                let value_part = s[after_colon..].trim_start();
                if value_part.starts_with('"') {
                    if let Some(end_rel) = find_unescaped_quote_end(&value_part[1..], b'"') {
                        let inner = &value_part[1..1 + end_rel];
                        if inner != "REDACTED"
                            && !inner.is_empty()
                            && inner.len() >= RESIDUAL_SECRET_MIN_LEN
                        {
                            return true;
                        }
                        // Partial scrub: `"REDACTED"secret_suffix…` (naive closer
                        // left live material glued after the REDACTED closer).
                        if inner == "REDACTED"
                            && residual_after_redacted_close(&value_part[1 + end_rel + 1..])
                        {
                            return true;
                        }
                        search_from = after_colon
                            + (s[after_colon..].len() - value_part.len())
                            + 1
                            + end_rel
                            + 1;
                        continue;
                    } else {
                        // Unterminated "value after secret key → fail-closed.
                        return true;
                    }
                } else if value_part.starts_with('\'') {
                    if let Some(end_rel) = find_unescaped_quote_end(&value_part[1..], b'\'') {
                        let inner = &value_part[1..1 + end_rel];
                        if inner != "REDACTED"
                            && !inner.is_empty()
                            && inner.len() >= RESIDUAL_SECRET_MIN_LEN
                        {
                            return true;
                        }
                        if inner == "REDACTED"
                            && residual_after_redacted_close(&value_part[1 + end_rel + 1..])
                        {
                            return true;
                        }
                        search_from = after_colon
                            + (s[after_colon..].len() - value_part.len())
                            + 1
                            + end_rel
                            + 1;
                        continue;
                    } else {
                        // Unterminated 'value after secret key → fail-closed.
                        return true;
                    }
                } else if !value_part.is_empty() {
                    // Non-string value after secret key — fail-closed unless exact REDACTED.
                    let bare: String = value_part
                        .chars()
                        .take_while(|c| {
                            *c != ','
                                && *c != '}'
                                && *c != ']'
                                && *c != ' '
                                && *c != '\t'
                                && *c != '\n'
                                && *c != '\r'
                        })
                        .collect();
                    if !bare.is_empty()
                        && bare.len() >= RESIDUAL_SECRET_MIN_LEN
                        && !is_exact_redacted_placeholder(&bare)
                    {
                        return true;
                    }
                    search_from = after_colon
                        + (s[after_colon..].len() - value_part.len())
                        + bare.len().max(1);
                    continue;
                }
                search_from = after_key;
            }
        }
    }
    // Colon-form header residuals: authorization: / x-api-key: with non-REDACTED material
    for marker in ["authorization:", "x-api-key:", "api-key:", "x-auth-token:"] {
        let mut search_from = 0;
        while let Some(rel) = lower[search_from..].find(marker) {
            let idx = search_from + rel;
            let after = s[idx + marker.len()..].trim_start();
            let token: String = after
                .chars()
                .take_while(|c| *c != '"' && *c != '\'' && *c != ',' && *c != '\n' && *c != '\r')
                .collect();
            let token = token.trim();
            // Exact REDACTED only — REDACTEDgluedsecret is still secret-shaped.
            if !token.is_empty() && !is_exact_redacted_placeholder(token) {
                return true;
            }
            search_from = idx + marker.len() + token.len().max(1);
        }
    }
    false
}

fn has_unredacted_dashed_key(s: &str, prefix: &str, redacted_form: &str) -> bool {
    let mut rest = s;
    while let Some(idx) = rest.find(prefix) {
        let at = &rest[idx..];
        if at.starts_with(redacted_form) {
            rest = &rest[idx + redacted_form.len()..];
            continue;
        }
        // Unredacted prefix: require enough key-shaped material after prefix.
        let after_prefix = &at[prefix.len()..];
        let key_len = after_prefix
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .count();
        if prefix.len() + key_len >= 12 {
            return true;
        }
        rest = &rest[idx + prefix.len()..];
    }
    false
}

fn has_bare_jwt_blob(s: &str) -> bool {
    // JWT headers are base64url(`{"…}`) → typically start with `eyJ`.
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 3 <= bytes.len() {
        if &bytes[i..i + 3] == b"eyJ" {
            let boundary_ok = i == 0
                || (!bytes[i - 1].is_ascii_alphanumeric()
                    && bytes[i - 1] != b'_'
                    && bytes[i - 1] != b'-'
                    && bytes[i - 1] != b'.');
            if boundary_ok {
                let mut j = i;
                while j < bytes.len()
                    && (bytes[j].is_ascii_alphanumeric()
                        || bytes[j] == b'_'
                        || bytes[j] == b'-'
                        || bytes[j] == b'.'
                        || bytes[j] == b'+'
                        || bytes[j] == b'/'
                        || bytes[j] == b'=')
                {
                    j += 1;
                }
                if j - i >= 20 {
                    return true;
                }
            }
        }
        i += 1;
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

/// Sanitize `finish_reason` for log / conv-log fields.
///
/// Always strips control characters (single-line), then runs the fail-closed
/// known-shape redactor (not only `error:`-prefixed values). Upstream/OpenAI-
/// compatible finish strings can carry secret-shaped content; if residual still
/// looks secret-bearing, collapse to a bounded token (`error` or `unknown`).
fn sanitize_finish_reason(finish_reason: &str) -> String {
    let s = sanitize_control_chars(finish_reason);
    match redact_error_for_log(&s) {
        Some(redacted) => {
            if still_secret_shaped(&redacted) {
                if s.starts_with("error:") {
                    "error".to_string()
                } else {
                    "unknown".to_string()
                }
            } else {
                redacted
            }
        }
        None => {
            if s.starts_with("error:") {
                "error".to_string()
            } else {
                "unknown".to_string()
            }
        }
    }
}

/// Format the complete line for tests / conv-log. Never panics.
///
/// Client-controlled text (`model`, `requested_model`, and error-prefixed
/// `finish_reason`) is control-char sanitized so embedded newlines cannot forge
/// extra log lines (unlike `sanitize_message`, which preserves `\n` for banners).
pub fn format_request_complete_fields(params: &RequestCompleteParams) -> String {
    let finish_reason =
        sanitize_finish_reason(&resolve_finish_reason(&params.finish, params.outcome));
    let outcome = match params.outcome {
        Outcome::Ok => "ok",
        Outcome::Error => "error",
    };
    // model can be raw client text on bad_request paths; scrub controls + secrets.
    let model = sanitize_model_for_status(&params.model);
    let mut out = String::new();
    let _ = write!(
        out,
        "model={} finish_reason={} duration_ms={:.1} outcome={}",
        model, finish_reason, params.duration_ms, outcome
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
        let rm = sanitize_model_for_status(rm);
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
/// present, never as Debug `Some(...)`. Client text in the message path is
/// control-char sanitized so `sanitize_message` cannot re-emit forged newlines.
pub fn log_request_complete(params: &RequestCompleteParams) {
    let finish_reason =
        sanitize_finish_reason(&resolve_finish_reason(&params.finish, params.outcome));
    let outcome = match params.outcome {
        Outcome::Ok => "ok",
        Outcome::Error => "error",
    };
    let duration = format!("{:.1}", params.duration_ms);
    // model may be raw client text (bad_request); scrub controls + secrets.
    let model = sanitize_model_for_status(&params.model);

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
        let rm = sanitize_model_for_status(rm);
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
        model = %model,
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
        assert_eq!(resolve_finish_reason(&finish, Outcome::Error), "incomplete");
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
        assert_eq!(resolve_finish_reason(&finish, Outcome::Ok), "tool_calls");
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
        // WHY: when presence is known (stream Usage event), true zeros print.
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
    fn plain_usage_all_zero_omits_token_keys() {
        // WHY: CanonicalUsage defaults missing usage to 0; never print fake zeros.
        let (inp, out) = tokens_from_plain_usage(0, 0, 0, 0);
        assert_eq!((inp, out), (None, None));
        let params = RequestCompleteParams::ok(
            "m",
            FinishSite {
                entered_body: true,
                finish_latch: Some("stop".into()),
                ..Default::default()
            },
            1.0,
        )
        .with_tokens(inp, out);
        let s = format_request_complete_fields(&params);
        assert!(!s.contains("input_tokens"), "{s}");
        assert!(!s.contains("output_tokens"), "{s}");
    }

    #[test]
    fn plain_usage_nonzero_sibling_keeps_true_zero() {
        // WHY: a non-zero field proves usage was present; sibling zeros are real.
        let (inp, out) = tokens_from_plain_usage(0, 5, 0, 0);
        assert_eq!((inp, out), (Some(0), Some(5)));
        let (inp2, out2) = tokens_from_plain_usage(0, 0, 3, 0);
        assert_eq!((inp2, out2), (Some(0), Some(0)));
    }

    #[test]
    fn redact_url_query_secrets() {
        let raw = "upstream https://api.example.com/v1?api_key=supersecret&x=1 failed";
        let redacted = redact_error_for_log(raw).expect("safe after redaction");
        assert!(!redacted.contains("supersecret"), "{redacted}");
        assert!(redacted.contains("REDACTED"), "{redacted}");
    }

    #[test]
    fn redact_repeated_url_query_secrets() {
        // WHY: multi-occurrence query secrets must all scrub; early break left
        // later api_key= values unredacted.
        let raw = "upstream https://api.example.com/v1?api_key=SEC1&api_key=SEC2 failed";
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(!s.contains("SEC1"), "{s}");
                assert!(!s.contains("SEC2"), "{s}");
                assert!(s.contains("REDACTED"), "{s}");
            }
            None => {} // fail-closed omit also acceptable
        }
        // Direct unit: both values become REDACTED even before residual check.
        let scrubbed = super::redact_url_query_secrets(raw);
        assert!(!scrubbed.contains("SEC1"), "{scrubbed}");
        assert!(!scrubbed.contains("SEC2"), "{scrubbed}");
    }

    #[test]
    fn redact_repeated_json_secret_fields() {
        // WHY: repeated JSON secret keys must all scrub; re-finding the first
        // already-REDACTED value must not stall before later keys.
        let raw = r#"upstream body {"api_key":"sekrit-aaa","api_key":"sekrit-bbb"}"#;
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(!s.contains("sekrit-aaa"), "{s}");
                assert!(!s.contains("sekrit-bbb"), "{s}");
                assert!(s.contains("REDACTED"), "{s}");
            }
            None => {} // fail-closed omit also acceptable
        }
        let scrubbed = super::redact_json_secret_fields(raw);
        assert!(!scrubbed.contains("sekrit-aaa"), "{scrubbed}");
        assert!(!scrubbed.contains("sekrit-bbb"), "{scrubbed}");
    }

    #[test]
    fn still_secret_shaped_catches_residual_query_and_json() {
        // WHY: residual detector must cover the same key list as the redactor.
        assert!(still_secret_shaped("token=this_is_a_long_token_xx"));
        assert!(still_secret_shaped("secret=this_is_a_long_secret_x"));
        assert!(still_secret_shaped("client_secret=longclientsecretval"));
        assert!(still_secret_shaped("refresh_token=longrefreshtokenval"));
        assert!(still_secret_shaped("id_token=longidtokenvaluehere"));
        // First already REDACTED, second still live.
        assert!(still_secret_shaped(
            "?api_key=REDACTED&api_key=stillsecretvalue"
        ));
        assert!(still_secret_shaped(r#"{"api_key":"stillsecretvalue"}"#));
        assert!(still_secret_shaped(r#"{"apikey":"opaque_secret_123456"}"#));
        assert!(still_secret_shaped("#access_token=opaque_secret_123456"));
        assert!(!still_secret_shaped("?api_key=REDACTED&x=1"));
        assert!(!still_secret_shaped(r#"{"api_key":"REDACTED"}"#));
    }

    #[test]
    fn redact_newline_delimited_query_secret_or_fail_closed() {
        // WHY: multi-line errors place api_key= after \n; boundary must include newline
        // before control-char collapse turns it into a same-line key=value.
        let raw = "failed\napi_key=opaque_secret_123456";
        match redact_error_for_log(raw) {
            Some(s) => assert!(!s.contains("opaque_secret_123456"), "{s}"),
            None => {}
        }
    }

    #[test]
    fn model_field_scrubs_known_secret_shapes() {
        // WHY: bad_request can put a pasted sk- key into model=; scrub it.
        let params = RequestCompleteParams::error(
            "sk-abcdefghijklmnopqrstuv",
            FinishSite {
                entered_body: false,
                incomplete: false,
                no_finish_concept: false,
                finish_latch: None,
            },
            0.0,
        );
        let line = format_request_complete_fields(&params);
        assert!(!line.contains("sk-abcdefghijklmnopqrstuv"), "{line}");
        assert!(line.contains("model="), "{line}");
    }

    #[test]
    fn redact_escaped_json_in_error_string_or_fail_closed() {
        // WHY: nested JSON dumps in error text use \"key\" form; redactors that
        // only match literal "key" would fail-open on opaque secrets.
        let raw = r#"body {\"api_key\":\"opaque_secret_123456\"}"#;
        match redact_error_for_log(raw) {
            Some(s) => assert!(!s.contains("opaque_secret_123456"), "{s}"),
            None => {}
        }
    }

    #[test]
    fn redact_json_non_string_secret_value_or_fail_closed() {
        // WHY: bare number/bool after a secret key must not pass through as
        // error= material (fail-open if only quoted strings are scrubbed).
        let raw = r#"body {"api_key":123456789012345678}"#;
        match redact_error_for_log(raw) {
            Some(s) => assert!(!s.contains("123456789012345678"), "{s}"),
            None => {}
        }
        assert!(still_secret_shaped(r#"{"api_key":123456789012345678}"#));
    }

    #[test]
    fn redacted_prefix_glued_secret_is_not_treated_as_clean() {
        // WHY: skipping any token that merely starts_with("REDACTED") fail-opens
        // on api_key=REDACTEDactualsecret… — only the exact placeholder is clean.
        let glued = "api_key=REDACTEDactualsecretvalue12345";
        assert!(
            still_secret_shaped(glued),
            "residual must flag glued secret"
        );
        match redact_error_for_log(glued) {
            Some(s) => assert!(
                !s.contains("actualsecretvalue12345"),
                "must scrub glued secret, got {s}"
            ),
            None => {} // fail-closed also OK
        }
        assert!(still_secret_shaped(
            "authorization: REDACTEDactualsecretvalue12345"
        ));
    }

    #[test]
    fn redact_json_apikey_key_or_fail_closed() {
        // WHY: bare "apikey" (no underscore) is a common JSON secret field name.
        let raw = r#"upstream body {"apikey":"opaque_secret_123456"}"#;
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(!s.contains("opaque_secret_123456"), "{s}");
                assert!(s.contains("REDACTED"), "{s}");
            }
            None => {} // fail-closed omit also acceptable
        }
    }

    #[test]
    fn redact_json_secret_keys_case_insensitive() {
        // WHY: API_KEY / Authorization / X-API-Key / ApiKey must not slip past
        // exact-case matching and leak into error= (or residual fail-closed).
        for raw in [
            r#"upstream body {"API_KEY":"opaque_secret_123456"}"#,
            r#"upstream body {"Authorization":"secretvaluehere123"}"#,
            r#"upstream body {"X-API-Key":"opaque_secret_123456"}"#,
            r#"upstream body {"ApiKey":"opaque_secret_123456"}"#,
            r#"upstream body {"accessToken":"opaque_secret_123456"}"#,
        ] {
            match redact_error_for_log(raw) {
                Some(s) => {
                    assert!(!s.contains("opaque_secret_123456"), "{s}");
                    assert!(!s.contains("secretvaluehere123"), "{s}");
                    assert!(s.contains("REDACTED"), "{s}");
                }
                None => {} // fail-closed omit also acceptable
            }
            // Direct redactor must rewrite the value (not only residual omit).
            let scrubbed = super::redact_json_secret_fields(raw);
            assert!(!scrubbed.contains("opaque_secret_123456"), "{scrubbed}");
            assert!(!scrubbed.contains("secretvaluehere123"), "{scrubbed}");
            assert!(scrubbed.contains("REDACTED"), "{scrubbed}");
        }
        // Residual detector must catch unredacted case variants.
        assert!(still_secret_shaped(r#"{"API_KEY":"opaque_secret_123456"}"#));
        assert!(still_secret_shaped(
            r#"{"Authorization":"secretvaluehere123"}"#
        ));
        assert!(!still_secret_shaped(r#"{"API_KEY":"REDACTED"}"#));
    }

    #[test]
    fn redact_fragment_access_token_or_fail_closed() {
        // WHY: fragment-bound keys (#access_token=…) must redact like query pairs.
        let raw = "redirect https://example.com/cb#access_token=opaque_secret_123456 failed";
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(!s.contains("opaque_secret_123456"), "{s}");
                assert!(s.contains("REDACTED"), "{s}");
            }
            None => {} // fail-closed omit also acceptable
        }
        let scrubbed = super::redact_url_query_secrets(raw);
        assert!(!scrubbed.contains("opaque_secret_123456"), "{scrubbed}");
    }

    #[test]
    fn redact_dashed_key_preserves_utf8() {
        // WHY: byte-as-char passthrough corrupted multi-byte UTF-8 (e.g. …).
        let raw = "upstream … failed with sk-abcdefghijklmnopqrstuv";
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(s.contains('…') || s.contains("…"), "{s}");
                assert!(!s.contains("abcdefghijklmnopqrstuv"), "{s}");
                assert!(s.contains("sk-REDACTED") || s.contains("REDACTED"), "{s}");
            }
            None => {
                // Fail-closed is fine for residual secrets, but direct redactor
                // must keep UTF-8 when rewriting sk- material.
                let scrubbed = super::replace_dashed_key_tokens(raw, b"sk-", "sk-REDACTED", 12);
                assert!(scrubbed.contains('…'), "{scrubbed}");
                assert!(scrubbed.contains("sk-REDACTED"), "{scrubbed}");
                assert!(!scrubbed.contains("abcdefghijklmnopqrstuv"), "{scrubbed}");
            }
        }
        let scrubbed = super::replace_dashed_key_tokens(raw, b"sk-", "sk-REDACTED", 12);
        assert!(scrubbed.contains('…'), "{scrubbed}");
        assert!(scrubbed.contains("sk-REDACTED"), "{scrubbed}");
    }

    #[test]
    fn redact_auth_like_headers() {
        let raw = "header Authorization: Bearer abcdefghijklmnop leaked";
        let redacted = redact_error_for_log(raw).expect("safe after redaction");
        assert!(!redacted.contains("abcdefghijklmnop"), "{redacted}");
        assert!(redacted.contains("REDACTED"), "{redacted}");
    }

    #[test]
    fn redact_authorization_token_scheme_fully() {
        // WHY: non-Bearer schemes must not leave trailing credential material.
        let raw = "upstream Authorization: Token ghp_abcdefghijklmnopqrstuv failed";
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(!s.contains("ghp_"), "{s}");
                assert!(!s.contains("abcdefghijklmnopqrstuv"), "{s}");
            }
            None => {} // fail-closed also acceptable
        }
    }

    #[test]
    fn redact_xai_key_or_fail_closed() {
        let raw = "auth failed with xai-abcdefghijklmnopqrstuvwxyz012345";
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(!s.contains("abcdefghijklmnopqrstuvwxyz012345"), "{s}");
                assert!(s.contains("xai-REDACTED") || s.contains("REDACTED"), "{s}");
            }
            None => {} // residual still secret-shaped → omit error=
        }
    }

    #[test]
    fn redact_eyj_jwt_fail_closed_or_omit() {
        // WHY: bare JWT blobs must not appear on complete lines.
        let raw = "token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig leaked";
        match redact_error_for_log(raw) {
            Some(s) => assert!(!s.contains("eyJ"), "{s}"),
            None => {} // fail-closed preferred for residual JWT shape
        }
        // Explicit residual path: still_secret_shaped must catch bare eyJ.
        assert!(still_secret_shaped(
            "token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        ));
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
    fn error_field_strips_newlines_single_line() {
        // WHY: error= must not forge multi-line log events.
        let params = RequestCompleteParams::error(
            "m",
            FinishSite {
                entered_body: false,
                ..Default::default()
            },
            1.0,
        )
        .with_error(Some("upstream failed:\nbad\r\tstatus".into()));
        let s = format_request_complete_fields(&params);
        assert!(!s.contains('\n'), "{s}");
        assert!(!s.contains('\r'), "{s}");
        assert!(!s.contains('\t'), "{s}");
        assert!(s.lines().count() == 1, "{s}");
        assert!(s.contains("error="), "{s}");
        assert!(s.contains("upstream failed"), "{s}");
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
    fn requested_model_and_model_strip_control_chars_single_line() {
        // WHY: client-controlled model text is appended into the message path;
        // sanitize_message preserves newlines for banners, so controls must be
        // stripped here or a model like "x\nEVIL" forges a second log line.
        let evil_rm = "client-model\nEVIL_FORGED_LINE";
        let evil_model = "stats-key\nEVIL_MODEL_LINE";
        let params = RequestCompleteParams::ok(
            evil_model,
            FinishSite {
                entered_body: true,
                finish_latch: Some("stop".into()),
                ..Default::default()
            },
            1.0,
        )
        .with_requested_model(Some(evil_rm.into()));
        let s = format_request_complete_fields(&params);
        assert_eq!(s.lines().count(), 1, "multi-line status: {s}");
        assert!(!s.contains('\n'), "{s}");
        assert!(!s.contains('\r'), "{s}");
        assert!(
            !s.contains("EVIL_FORGED_LINE\n") && !s.starts_with("EVIL"),
            "{s}"
        );
        // Content preserved without controls (spaces collapse runs).
        assert!(
            s.contains("requested_model=client-model EVIL_FORGED_LINE"),
            "{s}"
        );
        assert!(s.contains("model=stats-key EVIL_MODEL_LINE"), "{s}");
        // Bad-request path: raw requested model used as model key alone.
        let bad = RequestCompleteParams::error(
            "raw\nEVIL_ONLY_MODEL",
            FinishSite {
                entered_body: false,
                ..Default::default()
            },
            0.0,
        );
        let s2 = format_request_complete_fields(&bad);
        assert_eq!(s2.lines().count(), 1, "{s2}");
        assert!(!s2.contains('\n'), "{s2}");
        assert!(s2.contains("model=raw EVIL_ONLY_MODEL"), "{s2}");
    }

    #[test]
    fn map_anthropic_stop_reason_vocabulary() {
        assert_eq!(map_anthropic_stop_reason("end_turn"), "stop");
        assert_eq!(map_anthropic_stop_reason("tool_use"), "tool_calls");
        assert_eq!(map_anthropic_stop_reason("max_tokens"), "length");
        assert_eq!(map_anthropic_stop_reason("weird"), "weird");
    }

    #[test]
    fn redact_unterminated_json_secret_or_fail_closed() {
        // WHY: quote-started JSON secret values without a closing quote must not
        // skip past the key and leak the residual secret into error= logs.
        let raw = r#"body {"api_key":"opaque_secret_123456"#;
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(!s.contains("opaque_secret_123456"), "{s}");
                assert!(s.contains("REDACTED"), "{s}");
            }
            None => {} // fail-closed omit also acceptable
        }
        // Direct redactor must rewrite (not skip) the unterminated value.
        let scrubbed = super::redact_json_secret_fields(raw);
        assert!(!scrubbed.contains("opaque_secret_123456"), "{scrubbed}");
        assert!(scrubbed.contains("REDACTED"), "{scrubbed}");
        // Residual detector must catch unterminated secret-key:"value forms.
        assert!(still_secret_shaped(raw));
        assert!(still_secret_shaped(r#"{"API_KEY":"opaque_secret_123456"#));
    }

    #[test]
    fn redact_json_escaped_quote_secret_or_fail_closed() {
        // WHY: naive .find('"') stops at escaped \", truncating redaction and
        // leaving secret_suffix after "REDACTED"; residual must not treat that
        // as clean and leak the suffix into error= logs.
        let raw = r#"body {"api_key":"prefix\"secret_suffix_long_enough"}"#;
        match redact_error_for_log(raw) {
            Some(s) => {
                assert!(!s.contains("secret_suffix"), "{s}");
                assert!(!s.contains("prefix"), "{s}");
                assert!(s.contains("REDACTED"), "{s}");
            }
            None => {} // fail-closed omit also acceptable
        }
        // Direct redactor must consume through the true unescaped closer.
        let scrubbed = super::redact_json_secret_fields(raw);
        assert!(!scrubbed.contains("secret_suffix"), "{scrubbed}");
        assert!(!scrubbed.contains("prefix"), "{scrubbed}");
        assert!(scrubbed.contains("REDACTED"), "{scrubbed}");
        // Residual fail-closed on partial scrub: "REDACTED" + glued alphanumerics.
        assert!(still_secret_shaped(
            r#"{"api_key":"REDACTED"secret_suffix_long_enough"}"#
        ));
        // Clean REDACTED remains non-secret-shaped.
        assert!(!still_secret_shaped(r#"{"api_key":"REDACTED"}"#));
        // Even-length backslash run (\\") ends the string; odd (\\\") escapes.
        let even = super::redact_json_secret_fields(r#"{"api_key":"ab\\"}"#);
        assert!(!even.contains("ab\\"), "{even}");
        assert!(even.contains("REDACTED"), "{even}");
    }

    #[test]
    fn finish_reason_control_chars_single_line() {
        // WHY: finish_reason (latch plain text) must not forge multi-line status
        // or conv-log lines via embedded newlines/tabs.
        let params = RequestCompleteParams::ok(
            "m",
            FinishSite {
                entered_body: true,
                finish_latch: Some("stop\nEVIL_FORGED_LINE".into()),
                ..Default::default()
            },
            1.0,
        );
        let s = format_request_complete_fields(&params);
        assert_eq!(s.lines().count(), 1, "multi-line status: {s}");
        assert!(!s.contains('\n'), "{s}");
        assert!(!s.contains('\r'), "{s}");
        assert!(s.contains("finish_reason=stop EVIL_FORGED_LINE"), "{s}");
        let conv = format_conv_log_finish_summary(&params);
        assert_eq!(conv.lines().count(), 1, "multi-line conv: {conv}");
        assert!(!conv.contains('\n'), "{conv}");
    }

    #[test]
    fn finish_reason_error_prefix_redacts_secrets_or_collapses() {
        // WHY: error:-prefixed finish latch may embed upstream detail with secrets;
        // must redact or collapse to plain `error` so secrets never appear.
        let secret = "secretvaluehere12345";
        let params = RequestCompleteParams::error(
            "m",
            FinishSite {
                entered_body: true,
                finish_latch: Some(format!("error: kind: api_key={secret}")),
                ..Default::default()
            },
            1.0,
        );
        let s = format_request_complete_fields(&params);
        assert!(!s.contains(secret), "{s}");
        // Either redacted in place or collapsed to the bounded token `error`.
        assert!(
            s.contains("finish_reason=error") || s.contains("REDACTED"),
            "expected redacted or collapsed finish_reason: {s}"
        );
        let conv = format_conv_log_finish_summary(&params);
        assert!(!conv.contains(secret), "{conv}");
    }

    #[test]
    fn query_camelcase_oauth_keys_redact_or_fail_closed() {
        // WHY: JSON redactor already knows camelCase OAuth keys; query/residual
        // must match so accessToken= secrets cannot bypass fail-closed checks.
        for raw in [
            "accessToken=opaque_secret_123456",
            "refreshToken=opaque_secret_123456",
            "clientSecret=opaque_secret_123456",
        ] {
            match redact_error_for_log(raw) {
                Some(s) => assert!(!s.contains("opaque_secret_123456"), "{s} from {raw}"),
                None => {}
            }
        }
    }

    #[test]
    fn finish_reason_scrubs_non_error_prefix_secret_shapes() {
        // WHY: raw upstream finish values (not only error:) can be secret-shaped;
        // sanitize_finish_reason must not pass them through unredacted.
        let secret = "sk-abcdefghijklmnopqrstuv";
        let params = RequestCompleteParams::ok(
            "m",
            FinishSite {
                entered_body: true,
                finish_latch: Some(secret.into()),
                ..Default::default()
            },
            1.0,
        );
        let s = format_request_complete_fields(&params);
        assert!(!s.contains(secret), "{s}");
        assert!(
            s.contains("finish_reason=unknown")
                || s.contains("REDACTED")
                || s.contains("sk-REDACTED"),
            "{s}"
        );
    }
}
