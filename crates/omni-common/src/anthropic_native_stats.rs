//! Edge-facing Anthropic Messages stats helpers for the native path.
//!
//! These parse Anthropic JSON shapes only (usage objects, content deltas). They
//! do not touch fingerprints, stale-cch handling, or provider credentials. Used
//! by the thin edge when relaying `AnthropicNativeSurface` streams/responses.

use serde_json::Value;

use crate::stats::TokenUsage;

/// Extract token usage from a non-stream Anthropic Messages response JSON.
pub fn token_usage_from_anthropic_response(resp: &Value) -> TokenUsage {
    let usage = resp.get("usage");
    let get = |k: &str| {
        usage
            .and_then(|u| u.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    TokenUsage {
        input_tokens: get("input_tokens"),
        output_tokens: get("output_tokens"),
        cache_read_input_tokens: get("cache_read_input_tokens"),
        cache_creation_input_tokens: get("cache_creation_input_tokens"),
    }
}

/// Fold usage fields from one Anthropic SSE frame into a running total.
pub fn accumulate_anthropic_stream_usage(event: &str, data: &Value, usage: &mut TokenUsage) {
    let read = |u: &Value, k: &str| u.get(k).and_then(|v| v.as_u64());
    match event {
        "message_start" => {
            if let Some(u) = data.get("message").and_then(|m| m.get("usage")) {
                if let Some(v) = read(u, "input_tokens") {
                    usage.input_tokens = v;
                }
                if let Some(v) = read(u, "output_tokens") {
                    usage.output_tokens = v;
                }
                if let Some(v) = read(u, "cache_read_input_tokens") {
                    usage.cache_read_input_tokens = v;
                }
                if let Some(v) = read(u, "cache_creation_input_tokens") {
                    usage.cache_creation_input_tokens = v;
                }
            }
        }
        "message_delta" => {
            if let Some(u) = data.get("usage") {
                if let Some(v) = read(u, "output_tokens") {
                    usage.output_tokens = v;
                }
                if let Some(v) = read(u, "cache_read_input_tokens") {
                    usage.cache_read_input_tokens = v;
                }
                if let Some(v) = read(u, "cache_creation_input_tokens") {
                    usage.cache_creation_input_tokens = v;
                }
            }
        }
        _ => {}
    }
}

/// True when this frame is a content-producing delta (text or tool input JSON).
pub fn is_anthropic_content_delta(event: &str, data: &Value) -> bool {
    event == "content_block_delta"
        && data
            .get("delta")
            .and_then(|d| d.get("type"))
            .and_then(|t| t.as_str())
            .is_some_and(|t| t == "text_delta" || t == "input_json_delta")
}
