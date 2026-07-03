//! Native Anthropic Messages passthrough helpers for Omni's Claude-only inbound
//! `/v1/messages` surface.
//!
//! This module deliberately lives in `provider-claude`: the allowlist, model
//! resolution, identity injection, wire defaults, and raw SSE handling all touch
//! Claude Code fingerprint behavior that must not move into shared crates.

use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;

use futures_util::{Stream, StreamExt};
use omni_common::Replacements;
use serde::Deserialize;
use serde_json::Value;

use crate::fingerprint::{FingerprintProfile, RequestContext};
use crate::translate::{
    Message, MessagesRequest, SystemField, Thinking, Tool, ToolChoice, finalize_claude_wire_request,
};
use crate::upstream::RawFrame;
use crate::{ClaudeProvider, ProviderError, UpstreamError};

/// A client-supplied `/v1/messages` body, deserialized into a closed allowlist.
///
/// Fields that alter the Claude Code fingerprint or billing context, such as
/// `betas`, `metadata`, `service_tier`, `mcp_servers`, `container`, and
/// `output_config`, are intentionally absent and never forwarded.
#[derive(Debug, Clone, Deserialize)]
pub struct ClientMessagesRequest {
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub system: Option<SystemField>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub thinking: Option<Thinking>,
}

impl ClientMessagesRequest {
    fn to_messages_request(&self) -> MessagesRequest {
        MessagesRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens.unwrap_or(0),
            messages: self.messages.clone(),
            system: self.system.clone(),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences.clone(),
            stream: self.stream,
            metadata: None,
            thinking: self.thinking.clone(),
            output_config: None,
            // Native passthrough never injects a gateway-owned top-level marker
            // (deferred: see PR1 scope). The client's own block-level markers
            // ride inside messages/system/tools and are preserved untouched.
            cache_control: None,
        }
    }
}

/// One request to the native Anthropic surface after parsing and model routing.
#[derive(Debug, Clone)]
pub struct PreparedAnthropicRequest {
    pub requested_model: String,
    pub model_canonical: String,
    pub outbound_model: String,
    pub stream: bool,
    pub dropped_fields: Vec<String>,
    body: Value,
}

impl PreparedAnthropicRequest {
    pub fn body(&self) -> &Value {
        &self.body
    }
}

pub type RawFrameStream =
    Pin<Box<dyn Stream<Item = Result<RawFrame, ProviderError>> + Send + 'static>>;

/// Whether the client's body requested streaming. Anthropic selects SSE vs JSON
/// by this body field, defaulting to false when omitted.
pub fn client_requested_stream(raw_body: &Value) -> bool {
    raw_body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

const FORWARDED_FIELDS: &[&str] = &[
    "model",
    "max_tokens",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "stream",
    "thinking",
];

/// Top-level body keys not forwarded by the closed allowlist. Returned sorted
/// for stable diagnostics.
pub fn dropped_fields(raw_body: &Value) -> Vec<String> {
    let Some(obj) = raw_body.as_object() else {
        return Vec::new();
    };
    let mut out: Vec<String> = obj
        .keys()
        .filter(|k| !FORWARDED_FIELDS.contains(&k.as_str()))
        .cloned()
        .collect();
    out.sort();
    out
}

/// Prepare a client Anthropic request for the upstream Claude Code fingerprint
/// path. This parses the closed allowlist, resolves the model to Claude, fills
/// wire defaults, applies prompt replacements, and injects the Claude Code
/// identity when requested.
pub fn prepare_client_messages_request(
    raw_body: Value,
    profile: &'static FingerprintProfile,
    replacements: &Replacements,
    inject_identity: bool,
) -> Result<PreparedAnthropicRequest, ProviderError> {
    let client: ClientMessagesRequest = serde_json::from_value(raw_body.clone())
        .map_err(|e| ProviderError::BadRequest(format!("invalid Anthropic request: {e}")))?;
    let stream = client_requested_stream(&raw_body);
    let dropped = dropped_fields(&raw_body);
    let mut req =
        reconcile_client_request(&client, profile, replacements, inject_identity, stream)?;
    let outbound_model = req.model.clone();
    let model_canonical = profile
        .resolve_model(&client.model)
        .map(|d| d.canonical.to_string())
        .unwrap_or_else(|| client.model.clone());
    let body = serde_json::to_value(&mut req)
        .map_err(|e| ProviderError::Other(anyhow::Error::msg(format!("anth serialize: {e}"))))?;

    Ok(PreparedAnthropicRequest {
        requested_model: client.model,
        model_canonical,
        outbound_model,
        stream,
        dropped_fields: dropped,
        body,
    })
}

/// Prepare a client body for `/v1/messages/count_tokens`.
///
/// Count tokens intentionally does not inject Claude Code identity. Sampling and
/// output-control fields are stripped because Anthropic rejects them on that
/// endpoint.
pub fn prepare_count_tokens_request(
    raw_body: Value,
    profile: &'static FingerprintProfile,
    replacements: &Replacements,
) -> Result<PreparedAnthropicRequest, ProviderError> {
    let client: ClientMessagesRequest = serde_json::from_value(raw_body.clone())
        .map_err(|e| ProviderError::BadRequest(format!("invalid Anthropic request: {e}")))?;
    let dropped = dropped_fields(&raw_body);
    let mut req = reconcile_client_request(&client, profile, replacements, false, false)?;
    req.stream = None;
    let outbound_model = req.model.clone();
    let model_canonical = profile
        .resolve_model(&client.model)
        .map(|d| d.canonical.to_string())
        .unwrap_or_else(|| client.model.clone());
    let mut body = serde_json::to_value(&mut req)
        .map_err(|e| ProviderError::Other(anyhow::Error::msg(format!("anth serialize: {e}"))))?;
    if let Some(obj) = body.as_object_mut() {
        for key in [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "stream",
            "output_config",
            "metadata",
        ] {
            obj.remove(key);
        }
    }

    Ok(PreparedAnthropicRequest {
        requested_model: client.model,
        model_canonical,
        outbound_model,
        stream: false,
        dropped_fields: dropped,
        body,
    })
}

fn reconcile_client_request(
    client: &ClientMessagesRequest,
    profile: &'static FingerprintProfile,
    replacements: &Replacements,
    inject_identity: bool,
    stream: bool,
) -> Result<MessagesRequest, ProviderError> {
    let mut req = client.to_messages_request();
    // Pure pass-through: resolve exact canonical/alias, otherwise forward the id
    // raw (no strict-family reject). The shared forge tail applies the model id,
    // this door's real prompt replacements, the thinking-budget bump, wire
    // defaults, and identity - in that order.
    let resolved = profile.resolve_model(&client.model);
    finalize_claude_wire_request(
        &mut req,
        &client.model,
        resolved,
        profile,
        replacements,
        inject_identity,
        // Door 2 (native passthrough) does not inject a gateway-owned auto-cache
        // marker in PR1: deciding this correctly needs the position-vs-content-hash
        // question resolved (our identity prepend shifts client block indices), and
        // whether to forward a client's own top-level cache_control is a separate
        // allowlist change. Deferred -> always false here.
        false,
    );
    // `stream` is owned by this door, not the tail: a native body that omitted
    // stream must serialize `"stream": false`, not null. count_tokens clears it
    // afterward.
    req.stream = Some(stream);
    Ok(req)
}

// Shared with the forge tail in translate.rs (finalize_claude_wire_request):
// Door 2 applies its prompt replacements through the tail, which calls this.
pub(crate) fn apply_prompt_replacements(req: &mut MessagesRequest, replacements: &Replacements) {
    if replacements.is_empty() {
        return;
    }
    if let Some(system) = req.system.as_mut() {
        match system {
            SystemField::Text(text) => {
                *text = replacements.apply_prompt(text);
            }
            SystemField::Blocks(blocks) => {
                for block in blocks {
                    block.text = replacements.apply_prompt(&block.text);
                }
            }
        }
    }
    for message in &mut req.messages {
        apply_prompt_to_message_content(&mut message.content, replacements);
    }
    if let Some(tools) = req.tools.as_mut() {
        for tool in tools {
            tool.name = replacements.apply_prompt(&tool.name);
            if let Some(description) = tool.description.as_mut() {
                *description = replacements.apply_prompt(description);
            }
            apply_response_to_json(&mut tool.input_schema, replacements, true);
        }
    }
    if let Some(ToolChoice::Tool { name, .. }) = req.tool_choice.as_mut() {
        *name = replacements.apply_prompt(name);
    }
}

fn apply_prompt_to_message_content(
    content: &mut crate::translate::MessageContent,
    replacements: &Replacements,
) {
    match content {
        crate::translate::MessageContent::Text(text) => {
            *text = replacements.apply_prompt(text);
        }
        crate::translate::MessageContent::Blocks(blocks) => {
            for block in blocks {
                match block {
                    crate::translate::ContentBlock::Text { text, .. } => {
                        *text = replacements.apply_prompt(text);
                    }
                    crate::translate::ContentBlock::ToolUse { name, input, .. } => {
                        *name = replacements.apply_prompt(name);
                        apply_response_to_json(input, replacements, true);
                    }
                    crate::translate::ContentBlock::ToolResult {
                        content: Some(crate::translate::ToolResultContent::Text(text)),
                        ..
                    } => {
                        *text = replacements.apply_prompt(text);
                    }
                    crate::translate::ContentBlock::ToolResult {
                        content: Some(crate::translate::ToolResultContent::Blocks(blocks)),
                        ..
                    } => {
                        for block in blocks {
                            if let crate::translate::ContentBlock::Text { text, .. } = block {
                                *text = replacements.apply_prompt(text);
                            }
                        }
                    }
                    crate::translate::ContentBlock::Thinking { thinking, .. } => {
                        *thinking = replacements.apply_prompt(thinking);
                    }
                    crate::translate::ContentBlock::Image { .. }
                    | crate::translate::ContentBlock::ToolResult { content: None, .. } => {}
                }
            }
        }
    }
}

/// Apply response-scope replacements to raw Anthropic response content while
/// preserving unknown fields and block types.
pub fn apply_response_replacements_raw(resp: &mut Value, repl: &Replacements) {
    if repl.max_response_search_len() == 0 {
        return;
    }
    let Some(content) = resp.get_mut("content").and_then(|c| c.as_array_mut()) else {
        return;
    };
    for block in content {
        let kind = block
            .get("type")
            .and_then(|t| t.as_str())
            .map(str::to_string);
        match kind.as_deref() {
            Some("text") => {
                if let Some(text) = block.get_mut("text").and_then(|t| t.as_str()) {
                    block["text"] = Value::String(repl.apply_response(text));
                }
            }
            Some("tool_use") => {
                if let Some(name) = block.get_mut("name").and_then(|n| n.as_str()) {
                    block["name"] = Value::String(repl.apply_response(name));
                }
                if let Some(input) = block.get_mut("input") {
                    apply_response_to_json(input, repl, false);
                }
            }
            _ => {}
        }
    }
}

fn apply_response_to_json(value: &mut Value, repl: &Replacements, prompt_scope: bool) {
    match value {
        Value::String(s) => {
            *s = if prompt_scope {
                repl.apply_prompt(s)
            } else {
                repl.apply_response(s)
            };
        }
        Value::Array(arr) => {
            for item in arr {
                apply_response_to_json(item, repl, prompt_scope);
            }
        }
        Value::Object(obj) => {
            for item in obj.values_mut() {
                apply_response_to_json(item, repl, prompt_scope);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// Per-stream buffer for deferred response replacements over raw Anthropic SSE
/// frames. With no response rules, frames pass through unchanged.
pub struct RawSseReplState {
    has_response_rules: bool,
    text_buf: BTreeMap<u64, String>,
    json_buf: BTreeMap<u64, String>,
    stopped: BTreeSet<u64>,
}

impl RawSseReplState {
    pub fn new(repl: &Replacements) -> Self {
        Self {
            has_response_rules: repl.max_response_search_len() > 0,
            text_buf: BTreeMap::new(),
            json_buf: BTreeMap::new(),
            stopped: BTreeSet::new(),
        }
    }

    pub fn on_frame(
        &mut self,
        event: &str,
        data: Value,
        repl: &Replacements,
    ) -> Vec<(String, Value)> {
        if !self.has_response_rules {
            return vec![(event.to_string(), data)];
        }

        let index = data.get("index").and_then(|v| v.as_u64());
        let delta_type = data
            .get("delta")
            .and_then(|d| d.get("type"))
            .and_then(|t| t.as_str());

        match (event, delta_type, index) {
            ("content_block_delta", Some("text_delta"), Some(idx))
                if !self.stopped.contains(&idx) =>
            {
                if let Some(text) = data["delta"]["text"].as_str() {
                    self.text_buf.entry(idx).or_default().push_str(text);
                }
                vec![]
            }
            ("content_block_delta", Some("input_json_delta"), Some(idx))
                if !self.stopped.contains(&idx) =>
            {
                if let Some(json) = data["delta"]["partial_json"].as_str() {
                    self.json_buf.entry(idx).or_default().push_str(json);
                }
                vec![]
            }
            ("content_block_stop", _, Some(idx)) => {
                let mut out = self.flush_block(idx, repl);
                self.stopped.insert(idx);
                out.push((event.to_string(), data));
                out
            }
            _ => vec![(event.to_string(), data)],
        }
    }

    pub fn flush_all(&mut self, repl: &Replacements) -> Vec<(String, Value)> {
        let mut indices: Vec<u64> = self
            .text_buf
            .keys()
            .chain(self.json_buf.keys())
            .copied()
            .collect();
        indices.sort_unstable();
        indices.dedup();

        let mut out = Vec::new();
        for idx in indices {
            let frames = self.flush_block(idx, repl);
            if !frames.is_empty() {
                out.extend(frames);
                out.push((
                    "content_block_stop".to_string(),
                    serde_json::json!({"type": "content_block_stop", "index": idx}),
                ));
            }
            self.stopped.insert(idx);
        }
        out
    }

    fn flush_block(&mut self, idx: u64, repl: &Replacements) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        if let Some(text) = self.text_buf.remove(&idx) {
            out.push((
                "content_block_delta".to_string(),
                serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": repl.apply_response(&text)},
                }),
            ));
        }
        if let Some(raw_json) = self.json_buf.remove(&idx) {
            out.push((
                "content_block_delta".to_string(),
                serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": rewrite_json_string_leaves(&raw_json, repl),
                    },
                }),
            ));
        }
        out
    }
}

fn rewrite_json_string_leaves(raw: &str, repl: &Replacements) -> String {
    match serde_json::from_str::<Value>(raw) {
        Ok(mut value) => {
            apply_response_to_json(&mut value, repl, false);
            serde_json::to_string(&value).unwrap_or_else(|_| repl.apply_response(raw))
        }
        Err(_) => repl.apply_response(raw),
    }
}

pub fn token_usage_from_response(resp: &Value) -> omni_common::TokenUsage {
    let usage = resp.get("usage");
    let get = |k: &str| {
        usage
            .and_then(|u| u.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    omni_common::TokenUsage {
        input_tokens: get("input_tokens"),
        output_tokens: get("output_tokens"),
        cache_read_input_tokens: get("cache_read_input_tokens"),
        cache_creation_input_tokens: get("cache_creation_input_tokens"),
    }
}

pub fn accumulate_stream_usage(frame: &RawFrame, usage: &mut omni_common::TokenUsage) {
    let read = |u: &Value, k: &str| u.get(k).and_then(|v| v.as_u64());
    match frame.event.as_str() {
        "message_start" => {
            if let Some(u) = frame.data.get("message").and_then(|m| m.get("usage")) {
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
            if let Some(u) = frame.data.get("usage") {
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

pub fn is_upstream_content_delta(frame: &RawFrame) -> bool {
    frame.event == "content_block_delta"
        && frame
            .data
            .get("delta")
            .and_then(|d| d.get("type"))
            .and_then(|t| t.as_str())
            .is_some_and(|t| t == "text_delta" || t == "input_json_delta")
}

impl ClaudeProvider {
    pub fn prepare_anthropic_messages(
        &self,
        raw_body: Value,
        replacements: &Replacements,
        inject_identity: bool,
    ) -> Result<PreparedAnthropicRequest, ProviderError> {
        prepare_client_messages_request(raw_body, self.profile, replacements, inject_identity)
    }

    pub fn prepare_anthropic_count_tokens(
        &self,
        raw_body: Value,
        replacements: &Replacements,
    ) -> Result<PreparedAnthropicRequest, ProviderError> {
        prepare_count_tokens_request(raw_body, self.profile, replacements)
    }

    pub async fn send_anthropic_messages_json(
        &self,
        body: &Value,
        ctx: &RequestContext,
    ) -> Result<Value, ProviderError> {
        let creds = self.credentials_for_request().await?;
        let mut value = self
            .client
            .send_messages_json(&creds, ctx, body)
            .await
            .map_err(super::map_upstream_err)?;
        apply_response_replacements_raw(&mut value, &Replacements::empty());
        Ok(value)
    }

    pub async fn send_anthropic_messages_stream(
        &self,
        body: &Value,
        ctx: &RequestContext,
    ) -> Result<RawFrameStream, ProviderError> {
        let creds = self.credentials_for_request().await?;
        let stream = self
            .client
            .send_messages_stream_raw(&creds, ctx, body)
            .await
            .map_err(super::map_upstream_err)?;
        Ok(Box::pin(
            stream.map(|item| item.map_err(super::map_upstream_err)),
        ))
    }

    pub async fn send_anthropic_count_tokens(
        &self,
        body: &Value,
        ctx: &RequestContext,
    ) -> Result<Value, ProviderError> {
        let creds = self.credentials_for_request().await?;
        self.client
            .count_tokens(&creds, ctx, body)
            .await
            .map_err(super::map_upstream_err)
    }
}

#[allow(dead_code)]
fn _assert_upstream_error_send_sync(_: UpstreamError) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CLAUDE_CODE_SYSTEM_PREAMBLE;
    use crate::fingerprint::{RequestContext, default_profile};

    fn empty_repl() -> Replacements {
        Replacements::empty()
    }

    fn parse_client(body: Value) -> ClientMessagesRequest {
        serde_json::from_value(body).expect("client body parses")
    }

    fn system_texts(req: &MessagesRequest) -> Vec<String> {
        match req.system.as_ref().expect("system present") {
            SystemField::Blocks(blocks) => blocks.iter().map(|b| b.text.clone()).collect(),
            SystemField::Text(_) => panic!("identity injection must force system blocks"),
        }
    }

    #[test]
    fn reconcile_flat_system_prepends_identity() {
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "system": "be terse"
        });
        let client = parse_client(body);
        let req = reconcile_client_request(&client, default_profile(), &empty_repl(), true, false)
            .expect("reconcile ok");
        let texts = system_texts(&req);
        assert_eq!(texts.len(), 3);
        assert_eq!(texts[0], default_profile().billing_header_text("Say OK"));
        assert_eq!(texts[1], CLAUDE_CODE_SYSTEM_PREAMBLE);
        assert_eq!(texts[2], "be terse");
    }

    #[test]
    fn reconcile_strips_existing_identity_before_injecting() {
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=abcde;"},
                {"type": "text", "text": CLAUDE_CODE_SYSTEM_PREAMBLE},
                {"type": "text", "text": "consumer system"}
            ]
        });
        let client = parse_client(body);
        let req = reconcile_client_request(&client, default_profile(), &empty_repl(), true, false)
            .expect("reconcile ok");
        let texts = system_texts(&req);
        assert_eq!(texts[0], default_profile().billing_header_text("Say OK"));
        assert_eq!(texts[1], CLAUDE_CODE_SYSTEM_PREAMBLE);
        assert_eq!(texts[2], "consumer system");
        assert_eq!(
            texts
                .iter()
                .filter(|text| text.contains("x-anthropic-billing-header:"))
                .count(),
            1
        );
    }

    #[test]
    fn model_resolution_resolves_known_and_passes_through_the_rest() {
        // WHY: /v1/messages is now pure pass-through. Exact canonical/alias
        // resolve to the canonical; everything else (family long-forms, non-Claude
        // ids) forwards RAW instead of being rewritten or rejected. The former
        // strict-family reject is deliberately gone (owner-decided pass-through).
        for (input, expected) in [
            // exact canonical -> canonical
            ("claude-opus-4-8", "claude-opus-4-8"),
            // short alias -> canonical
            ("sonnet", "claude-sonnet-4-6"),
            // family long-form: NOT a catalog alias -> forwards raw
            ("claude-sonnet", "claude-sonnet"),
            // non-Claude id: no reject -> forwards raw (Anthropic will 400 it)
            ("grok-4.3", "grok-4.3"),
        ] {
            let body = serde_json::json!({
                "model": input,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say OK"}]
            });
            let client = parse_client(body);
            let req =
                reconcile_client_request(&client, default_profile(), &empty_repl(), false, false)
                    .expect("reconcile ok (pass-through never rejects on model)");
            assert_eq!(req.model, expected, "input {input:?}");
        }
    }

    #[test]
    fn prepare_surfaces_client_validation_as_bad_request() {
        // WHY (Rule 9): a body that does not match the Anthropic request schema
        // (max_tokens as a string) is a CLIENT fault -> serde failure -> it must
        // stay a ProviderError::BadRequest (-> 400), not regress to an Upstream
        // 502 via classify_upstream. This is the surviving half of issue #3's
        // guard; the bare-unrecognized-model 400 was intentionally dropped when
        // /v1/messages became pure pass-through (see the pass-through note below).
        let malformed = serde_json::json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": "not-a-number",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let err =
            prepare_client_messages_request(malformed, default_profile(), &empty_repl(), true)
                .expect_err("malformed body must reject");
        assert!(
            matches!(err, ProviderError::BadRequest(_)),
            "malformed body must be a client BadRequest, got {err:?}"
        );
    }

    #[test]
    fn prepare_passes_through_unrecognized_model_without_local_reject() {
        // WHY: dropping resolve_strict_claude_model means an unrecognized (non-
        // Claude) model on /v1/messages is no longer 400'd locally; it forwards
        // RAW to Anthropic (which returns its own 400). This is the deliberate
        // pass-through decision (issue #3's bare-model guard removed on purpose).
        let unknown_model = serde_json::json!({
            "model": "grok-4.3",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let prepared =
            prepare_client_messages_request(unknown_model, default_profile(), &empty_repl(), true)
                .expect("pass-through must not reject an unrecognized model");
        // Forwarded verbatim, not rewritten to a Claude canonical.
        assert_eq!(prepared.outbound_model, "grok-4.3");
        assert_eq!(prepared.requested_model, "grok-4.3");
    }

    #[test]
    fn wire_defaults_fill_only_unset_values() {
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let req = reconcile_client_request(
            &parse_client(body),
            default_profile(),
            &empty_repl(),
            false,
            false,
        )
        .expect("reconcile ok");
        assert!(req.max_tokens > 0);
        assert_eq!(req.temperature, Some(1.0));

        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 123,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let req = reconcile_client_request(
            &parse_client(body),
            default_profile(),
            &empty_repl(),
            false,
            false,
        )
        .expect("reconcile ok");
        assert_eq!(req.max_tokens, 123);
        assert_eq!(req.temperature, Some(0.3));
    }

    #[test]
    fn thinking_budget_forces_larger_max_tokens() {
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096}
        });
        let req = reconcile_client_request(
            &parse_client(body),
            default_profile(),
            &empty_repl(),
            false,
            false,
        )
        .expect("reconcile ok");
        assert!(req.max_tokens > 4096);
    }

    #[test]
    fn door2_thinking_bump_converges_onto_door1_algorithm() {
        // WHY (forge-tail unification, intended CHANGE): Door 2 now runs the
        // thinking-budget bump BEFORE wire defaults (via the shared tail), same
        // as Door 1. Previously Door 2 filled the wire default (32000) first, so a
        // budget BELOW 32000 with an omitted max_tokens emitted 32000. Now it
        // emits budget+1024 = 17408, converging onto Door 1's result. Both are
        // valid (> budget). This test locks the convergence so a regression that
        // reordered the tail (wire-defaults-first) would fail here.
        let body = serde_json::json!({
            "model": "sonnet",
            "messages": [{"role": "user", "content": "Say OK"}],
            "thinking": {"type": "enabled", "budget_tokens": 16384}
        });
        let req = reconcile_client_request(
            &parse_client(body),
            default_profile(),
            &empty_repl(),
            false,
            false,
        )
        .expect("reconcile ok");
        assert_eq!(
            req.max_tokens, 17408,
            "budget 16384 + omitted max_tokens must bump to budget+1024, not the wire default 32000"
        );
    }

    #[test]
    fn door2_prompt_replacements_apply_before_identity_suffix() {
        // WHY (forge-tail unification): Door 2 passes its REAL replacements to the
        // shared tail, which must apply them BEFORE identity injection so the
        // billing suffix is computed over the REPLACED first-user text. If the tail
        // dropped replacements or ran them after identity, the suffix would be
        // computed over the original text and drift from the wire body.
        let repl = Replacements::parse(
            r#"rule = [{ scope = "prompt", search = "PLAIN", replace = "OK" }]"#,
        )
        .unwrap();
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "PLAIN"}]
        });
        let client = parse_client(body);
        let req = reconcile_client_request(&client, default_profile(), &repl, true, false)
            .expect("reconcile ok");
        // The first user message text was replaced PLAIN -> OK.
        let texts = system_texts(&req);
        // Billing header (texts[0]) must be computed over the REPLACED text "OK",
        // not the original "PLAIN".
        assert_eq!(texts[0], default_profile().billing_header_text("OK"));
        assert_ne!(texts[0], default_profile().billing_header_text("PLAIN"));
    }

    #[test]
    fn closed_allowlist_drops_fingerprint_fields() {
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "betas": ["client-beta"],
            "metadata": {"user_id": "client"},
            "service_tier": "auto",
            "mcp_servers": [],
            "container": {"id": "c1"},
            "output_config": {"effort": "max"}
        });
        let dropped = dropped_fields(&body);
        assert_eq!(
            dropped,
            vec![
                "betas",
                "container",
                "mcp_servers",
                "metadata",
                "output_config",
                "service_tier"
            ]
        );
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), true)
                .expect("prepare ok");
        let wire = prepared.body();
        assert!(wire.get("metadata").is_none());
        assert!(wire.get("betas").is_none());
        assert!(wire.get("service_tier").is_none());
    }

    #[test]
    fn count_tokens_omits_identity_and_sampling_fields() {
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "temperature": 0.2,
            "top_p": 0.9,
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let prepared = prepare_count_tokens_request(body, default_profile(), &empty_repl())
            .expect("prepare ok");
        let wire = prepared.body();
        assert!(wire.get("max_tokens").is_none());
        assert!(wire.get("temperature").is_none());
        assert!(wire.get("top_p").is_none());
        assert!(
            wire.get("system").is_none(),
            "no identity is injected for count_tokens"
        );
    }

    #[test]
    fn response_replacements_touch_known_raw_leaves_only() {
        let repl = Replacements::parse(
            r#"rule = [{ scope = "response", search = "SECRET", replace = "OK" }]"#,
        )
        .unwrap();
        let mut resp = serde_json::json!({
            "id": "msg_1",
            "content": [
                {"type": "text", "text": "SECRET text"},
                {"type": "tool_use", "id": "toolu_1", "name": "SECRET_tool", "input": {"arg": "SECRET arg", "n": 1}},
                {"type": "thinking", "thinking": "SECRET hidden"}
            ],
            "other": "SECRET untouched"
        });
        apply_response_replacements_raw(&mut resp, &repl);
        assert_eq!(resp["content"][0]["text"], "OK text");
        assert_eq!(resp["content"][1]["name"], "OK_tool");
        assert_eq!(resp["content"][1]["input"]["arg"], "OK arg");
        assert_eq!(resp["content"][2]["thinking"], "SECRET hidden");
        assert_eq!(resp["other"], "SECRET untouched");
    }

    #[test]
    fn raw_sse_replacements_buffer_until_block_stop() {
        let repl = Replacements::parse(
            r#"rule = [{ scope = "response", search = "SECRET", replace = "OK" }]"#,
        )
        .unwrap();
        let mut state = RawSseReplState::new(&repl);
        let out = state.on_frame(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "SEC"}
            }),
            &repl,
        );
        assert!(out.is_empty());
        let out = state.on_frame(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "RET"}
            }),
            &repl,
        );
        assert!(out.is_empty());
        let out = state.on_frame(
            "content_block_stop",
            serde_json::json!({"type": "content_block_stop", "index": 0}),
            &repl,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "content_block_delta");
        assert_eq!(out[0].1["delta"]["text"], "OK");
        assert_eq!(out[1].0, "content_block_stop");
    }

    #[test]
    fn raw_sse_no_response_rules_passthrough() {
        let repl = Replacements::empty();
        let mut state = RawSseReplState::new(&repl);
        let frame = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"}
        });
        let out = state.on_frame("content_block_delta", frame.clone(), &repl);
        assert_eq!(out, vec![("content_block_delta".to_string(), frame)]);
    }

    #[test]
    fn prepared_body_finalizes_cch_after_identity() {
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), true)
                .expect("prepare ok");
        let bytes = default_profile()
            .finalize_body_json(
                prepared.body(),
                &RequestContext::new_reply().with_model(prepared.outbound_model.clone()),
            )
            .expect("finalize ok");
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("x-anthropic-billing-header:"));
        assert!(!text.contains("cch=00000"));
    }
}
