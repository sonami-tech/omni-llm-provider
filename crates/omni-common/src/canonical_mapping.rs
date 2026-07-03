//! Small leaf helpers shared between the Chat Completions (`http`) and Responses
//! (`responses`) `from_canonical` conversions. Only genuinely identical
//! output-direction helpers live here; the per-format message-walking routines
//! stay in their own modules because the Chat and Responses wire shapes differ.

use serde_json::{Map, Value};

use omni_core::CanonicalResponse;

/// Build a `*_tokens_details` object from `(key, count)` pairs, omitting any
/// zero count and returning `None` when nothing remains. Both wire formats emit
/// the same detail shape, so this is shared verbatim.
pub(crate) fn usage_detail_json(fields: &[(&str, u64)]) -> Option<Value> {
    let mut details = Map::new();
    for (key, value) in fields {
        if *value != 0 {
            details.insert((*key).to_string(), Value::from(*value));
        }
    }
    (!details.is_empty()).then_some(Value::Object(details))
}

/// Build the `provider_metadata` object common to both formats: provider name,
/// raw provider payload, source count, model reasoning, and optionally the
/// annotation array. Returns `None` when nothing was populated.
///
/// `include_annotations` is explicit because the two formats differ on exactly
/// this point: the Chat response carries `annotations` in provider metadata, the
/// Responses shape does not. Passing it at the call site keeps that divergence
/// visible rather than hidden behind a shared helper.
pub(crate) fn provider_metadata_json(
    canon: &CanonicalResponse,
    include_annotations: bool,
) -> Option<Value> {
    let mut metadata = Map::new();
    if let Some(meta) = &canon.metadata {
        if let Some(provider) = &meta.provider {
            metadata.insert("provider".into(), Value::String(provider.clone()));
        }
        if let Some(raw) = &meta.raw {
            metadata.insert("raw".into(), raw.clone());
        }
    }
    if canon.usage.num_sources_used != 0 {
        metadata.insert(
            "num_sources_used".into(),
            Value::from(canon.usage.num_sources_used),
        );
    }
    if include_annotations && !canon.annotations.is_empty() {
        metadata.insert(
            "annotations".into(),
            Value::Array(canon.annotations.clone()),
        );
    }
    if !canon.reasoning.is_empty() {
        metadata.insert("reasoning".into(), serde_json::json!(canon.reasoning));
    }
    (!metadata.is_empty()).then_some(Value::Object(metadata))
}
