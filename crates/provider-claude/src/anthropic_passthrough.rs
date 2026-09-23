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
    CacheControl, Message, MessagesRequest, OutputConfig, SystemField, Thinking, Tool, ToolChoice,
    finalize_claude_wire_request,
};
use crate::upstream::RawFrame;
use crate::{ClaudeProvider, ProviderError};

/// A client-supplied `/v1/messages` body, deserialized into a closed allowlist.
///
/// Fingerprint/billing-owned fields (`betas`, `metadata`, `service_tier`,
/// `mcp_servers`, `container`) stay absent and are never forwarded.
///
/// `output_config` is client-intent: when the client sets it, Door-2 passes it
/// through. Capture/pin defaults fill only when the client left it unset
/// (precedence client > pin default > absent; issues #20 / #22 / #27).
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
    /// Native Anthropic effort surface. Client-set values are honored; pin
    /// defaults apply only when this is `None` (issue #22).
    #[serde(default)]
    pub output_config: Option<OutputConfig>,
    /// Official Anthropic automatic cache marker. Same-provider passthrough
    /// forwards it unchanged; Door 2 does not inject a gateway marker.
    #[serde(default)]
    pub cache_control: Option<CacheControl>,
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
            // Client > pin default > absent (issue #22). Pin fill happens later
            // in finalize_claude_wire_request → apply_profile_wire_defaults.
            output_config: self.output_config.clone(),
            // Client top-level automatic cache_control is official Anthropic.
            // Forward it. Do not inject a gateway-owned marker on this door.
            cache_control: self.cache_control.clone(),
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
    "output_config",
    "cache_control",
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
    reject_native_anthropic_cache_dialect(&raw_body)?;
    let client: ClientMessagesRequest = serde_json::from_value(raw_body.clone())
        .map_err(|e| ProviderError::BadRequest(format!("invalid Anthropic request: {e}")))?;
    validate_native_tools(&client)?;
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
    reject_native_anthropic_cache_dialect(&raw_body)?;
    let client: ClientMessagesRequest = serde_json::from_value(raw_body.clone())
        .map_err(|e| ProviderError::BadRequest(format!("invalid Anthropic request: {e}")))?;
    validate_native_tools(&client)?;
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

fn reject_native_anthropic_cache_dialect(raw_body: &Value) -> Result<(), ProviderError> {
    omni_common::cache::reject_anthropic_prompt_cache_key(raw_body)
        .map_err(ProviderError::BadRequest)
}

fn validate_native_tools(client: &ClientMessagesRequest) -> Result<(), ProviderError> {
    if let Some(tools) = &client.tools {
        for tool in tools {
            omni_core::validate_tool_schema(&tool.input_schema, tool.strict.unwrap_or(false), true)
                .map_err(ProviderError::BadRequest)?;
            omni_core::validate_provider_tool_shape(&tool.input_schema)
                .map_err(ProviderError::BadRequest)?;
        }
    }
    Ok(())
}

fn reconcile_client_request(
    client: &ClientMessagesRequest,
    profile: &'static FingerprintProfile,
    replacements: &Replacements,
    inject_identity: bool,
    stream: bool,
) -> Result<MessagesRequest, ProviderError> {
    let mut req = client.to_messages_request();
    if let Some(tools) = req.tools.as_mut() {
        for tool in tools {
            tool.strict =
                omni_core::claude_strict(&tool.input_schema, tool.strict.unwrap_or(false))
                    .then_some(true);
        }
    }
    // Pure pass-through: resolve exact canonical/alias, otherwise forward the id
    // raw (no strict-family reject). The shared forge tail applies the model id,
    // this door's real prompt replacements, wire defaults, and identity - in
    // that order. Thinking budget never auto-bumps max_tokens (issue #19).
    let resolved = profile.resolve_model(&client.model);
    finalize_claude_wire_request(
        &mut req,
        &client.model,
        resolved,
        profile,
        replacements,
        inject_identity,
        // Door 2 does not inject a gateway-owned auto-cache marker. Client
        // top-level cache_control is already on `req` from the allowlist.
        // Do not apply the translation last-4 slot cap on this path.
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
                    | crate::translate::ContentBlock::Document { .. }
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

    fn request_context_for_session(session_key: &str, outbound_model: &str) -> RequestContext {
        let session = if let Ok(uuid) = uuid::Uuid::parse_str(session_key) {
            uuid
        } else {
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, session_key.as_bytes())
        };
        RequestContext::new_reply()
            .with_session(session)
            .with_model(outbound_model.to_string())
    }
}

/// Object-safe native Anthropic surface for the thin edge (no FingerprintProfile
/// or passthrough imports required at the call site).
#[async_trait::async_trait]
impl omni_core::AnthropicNativeSurface for ClaudeProvider {
    fn prepare_messages(
        &self,
        raw_body: Value,
        inject_identity: bool,
    ) -> Result<omni_core::PreparedAnthropicNative, ProviderError> {
        let prepared =
            self.prepare_anthropic_messages(raw_body, &Replacements::empty(), inject_identity)?;
        let body = prepared.body().clone();
        Ok(omni_core::PreparedAnthropicNative::new(
            prepared.requested_model,
            prepared.model_canonical,
            prepared.outbound_model,
            prepared.stream,
            prepared.dropped_fields,
            body,
        ))
    }

    fn prepare_count_tokens(
        &self,
        raw_body: Value,
    ) -> Result<omni_core::PreparedAnthropicNative, ProviderError> {
        let prepared = self.prepare_anthropic_count_tokens(raw_body, &Replacements::empty())?;
        let body = prepared.body().clone();
        Ok(omni_core::PreparedAnthropicNative::new(
            prepared.requested_model,
            prepared.model_canonical,
            prepared.outbound_model,
            prepared.stream,
            prepared.dropped_fields,
            body,
        ))
    }

    async fn send_messages_json(
        &self,
        body: &Value,
        session_key: &str,
        outbound_model: &str,
    ) -> Result<Value, ProviderError> {
        let ctx = Self::request_context_for_session(session_key, outbound_model);
        self.send_anthropic_messages_json(body, &ctx).await
    }

    async fn send_messages_stream(
        &self,
        body: &Value,
        session_key: &str,
        outbound_model: &str,
    ) -> Result<omni_core::NativeAnthropicSseStream, ProviderError> {
        let ctx = Self::request_context_for_session(session_key, outbound_model);
        let stream = self.send_anthropic_messages_stream(body, &ctx).await?;
        Ok(Box::pin(stream.map(|item| {
            item.map(|frame| omni_core::NativeAnthropicSseFrame {
                event: frame.event,
                data: frame.data,
            })
        })))
    }

    async fn send_count_tokens(
        &self,
        body: &Value,
        session_key: &str,
        outbound_model: &str,
    ) -> Result<Value, ProviderError> {
        let ctx = Self::request_context_for_session(session_key, outbound_model);
        self.send_anthropic_count_tokens(body, &ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CLAUDE_CODE_SYSTEM_PREAMBLE;
    use crate::UpstreamError;
    use crate::fingerprint::{RequestContext, default_profile};

    fn empty_repl() -> Replacements {
        Replacements::empty()
    }

    #[test]
    fn upstream_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<UpstreamError>();
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
    fn native_ingress_preserves_document_metadata() {
        let body = serde_json::json!({
            "model": "claude-sonnet-5", "max_tokens": 100,
            "messages": [{ "role": "user", "content": [{
                "type": "document",
                "source": { "type": "url", "url": "https://example.com/a.pdf" },
                "title": "Invoice A", "context": "Financial record",
                "citations": { "enabled": true },
                "cache_control": { "type": "ephemeral" }
            }] }]
        });
        let wire = serde_json::to_value(parse_client(body.clone()).to_messages_request()).unwrap();
        for key in ["source", "title", "context", "citations", "cache_control"] {
            assert_eq!(
                wire["messages"][0]["content"][0][key], body["messages"][0]["content"][0][key],
                "{key}"
            );
        }
    }

    #[test]
    fn native_ingress_preserves_all_anthropic_tool_choice_modes() {
        for choice in [
            serde_json::json!({"type": "auto", "disable_parallel_tool_use": true}),
            serde_json::json!({"type": "any", "disable_parallel_tool_use": true}),
            serde_json::json!({"type": "tool", "name": "read", "disable_parallel_tool_use": true}),
            serde_json::json!({"type": "none"}),
        ] {
            let expected_type = choice["type"].as_str().unwrap();
            let body = serde_json::json!({
                "model": "claude-sonnet-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "use tools"}],
                "tools": [{"name": "read", "input_schema": {"type": "object"}}],
                "tool_choice": choice
            });
            let wire = serde_json::to_value(parse_client(body).to_messages_request())
                .expect("request serializes");
            assert_eq!(wire["tool_choice"]["type"], expected_type);
            if expected_type == "tool" {
                assert_eq!(wire["tool_choice"]["name"], "read");
            }
            if expected_type != "none" {
                assert_eq!(wire["tool_choice"]["disable_parallel_tool_use"], true);
            }
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
            ("claude-opus-5-5", "claude-opus-5-5"),
            // short alias -> canonical
            ("sonnet", "claude-sonnet-5"),
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
        // 2.1.207 haiku wire omits temperature when the client did not set one.
        assert_eq!(req.temperature, None);

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
    fn thinking_budget_does_not_raise_client_max_tokens() {
        // WHY (issue #19): passthrough. Client max_tokens is sent as given even
        // when thinking.budget_tokens is larger. Omni must not auto-bump.
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
        assert_eq!(
            req.max_tokens, 100,
            "client max_tokens must not be raised for thinking budget"
        );
        assert_eq!(
            req.thinking.as_ref().and_then(|t| t.budget_tokens),
            Some(4096)
        );
    }

    #[test]
    fn door2_omitted_max_tokens_uses_wire_default_not_thinking_bump() {
        // WHY (issue #19): when the client omits max_tokens, only fingerprint
        // wire defaults fill it. Thinking budget must not replace that with
        // budget+1024.
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
            req.max_tokens, 64_000,
            "omitted max_tokens must use wire default, not budget+1024"
        );
        assert_eq!(
            req.thinking.as_ref().and_then(|t| t.budget_tokens),
            Some(16384)
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
    fn door2_forwards_client_cache_control_and_injects_no_gateway_marker() {
        // WHY: Door 2 does not inject a gateway-owned top-level marker. It does
        // forward the client's own block-level cache_control. Identity prepend
        // shifts the client block behind billing+preamble, but the marker stays.
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "system": [{
                "type": "text",
                "text": "client cached system",
                "cache_control": {"type": "ephemeral"}
            }]
        });
        let client = parse_client(body);
        let req = reconcile_client_request(&client, default_profile(), &empty_repl(), true, false)
            .expect("reconcile ok");

        // (a) No gateway-owned top-level marker on the native door.
        assert!(
            req.cache_control.is_none(),
            "Door 2 must not inject a top-level auto-cache marker; that is Door 1 only"
        );

        // (b) The client's own system-block marker is preserved. After identity
        // injection the system is blocks: [billing, preamble, ...client blocks].
        let SystemField::Blocks(blocks) = req.system.as_ref().expect("system present") else {
            panic!("identity injection must force system blocks");
        };
        let marked: Vec<_> = blocks
            .iter()
            .filter(|b| b.cache_control.is_some())
            .collect();
        assert_eq!(
            marked.len(),
            1,
            "exactly the client's one cached system block should carry a marker (identity blocks carry none)"
        );
        assert_eq!(
            marked[0].text, "client cached system",
            "the preserved marker must ride the client's block, not a gateway block"
        );
        assert_eq!(
            marked[0].cache_control.as_ref().map(|c| c.kind.as_str()),
            Some("ephemeral"),
            "client's ephemeral marker must survive reconciliation unchanged"
        );
    }

    #[test]
    fn door2_forwards_top_level_automatic_cache_control_unchanged() {
        // Native entry: prepare_client_messages_request. Same-provider
        // passthrough must send official automatic cache_control as given.
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), true)
                .expect("prepare ok");
        let wire = prepared.body();
        assert_eq!(
            wire.get("cache_control")
                .and_then(|c| c.get("type"))
                .and_then(|t| t.as_str()),
            Some("ephemeral")
        );
        assert_eq!(
            wire.get("cache_control")
                .and_then(|c| c.get("ttl"))
                .and_then(|t| t.as_str()),
            Some("1h")
        );
        assert!(
            !prepared.dropped_fields.iter().any(|k| k == "cache_control"),
            "cache_control must be forwarded, not dropped: {:?}",
            prepared.dropped_fields
        );
    }

    #[test]
    fn door2_rejects_anthropic_body_prompt_cache_key() {
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "prompt_cache_key": "sess-1",
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let err = prepare_client_messages_request(body, default_profile(), &empty_repl(), true)
            .expect_err("prompt_cache_key is not an Anthropic field");
        match err {
            ProviderError::BadRequest(msg) => {
                assert!(msg.contains("prompt_cache_key"), "{msg}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn door2_rejects_invalid_cache_control_type_and_ttl() {
        let bad_type = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "cache_control": {"type": "persistent"},
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let err = prepare_client_messages_request(bad_type, default_profile(), &empty_repl(), true)
            .expect_err("invalid type must 400");
        match err {
            ProviderError::BadRequest(msg) => {
                assert!(msg.contains("cache_control"), "{msg}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }

        let bad_ttl = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "cache_control": {"type": "ephemeral", "ttl": "30m"},
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let err = prepare_client_messages_request(bad_ttl, default_profile(), &empty_repl(), true)
            .expect_err("invalid ttl must 400");
        match err {
            ProviderError::BadRequest(msg) => {
                assert!(msg.contains("cache_control"), "{msg}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn door2_does_not_apply_last_four_slot_cap() {
        // Translation caps at 4 including automatic. Native passthrough must
        // not rewrite the client body.
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "cache_control": {"type": "ephemeral"},
            "system": [
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "d", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "e", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), true)
                .expect("prepare ok");
        let wire = prepared.body();
        assert!(wire.get("cache_control").is_some());
        let system = wire["system"].as_array().expect("system blocks");
        let marked = system
            .iter()
            .filter(|b| b.get("cache_control").is_some())
            .count();
        assert_eq!(
            marked, 5,
            "native passthrough must keep all five client marks"
        );
    }

    #[test]
    fn closed_allowlist_drops_fingerprint_fields() {
        // WHY (issue #22): output_config is client intent, not a fingerprint-
        // owned strip. Only true fingerprint/billing fields stay dropped.
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
        // Client output_config survives prepare (honor client on Door-2).
        assert_eq!(
            wire.get("output_config")
                .and_then(|o| o.get("effort"))
                .and_then(|e| e.as_str()),
            Some("max")
        );
    }

    #[test]
    fn door2_client_output_config_effort_preserved_over_pin() {
        // WHY (issue #22): Door-2 must not strip client output_config for
        // fingerprint fidelity. Client effort wins over the Sonnet pin default.
        let body = serde_json::json!({
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "output_config": {"effort": "low"}
        });
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), false)
                .expect("prepare ok");
        let wire = prepared.body();
        assert_eq!(
            wire.get("output_config")
                .and_then(|o| o.get("effort"))
                .and_then(|e| e.as_str()),
            Some("low"),
            "client output_config.effort must win over the Sonnet pin default"
        );
        // Also via reconcile (same prepare path guts).
        let body = serde_json::json!({
            "model": "sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "output_config": {"effort": "max"}
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
            req.output_config.as_ref().and_then(|o| o.effort.as_deref()),
            Some("max"),
            "client max must not be replaced by sonnet pin high"
        );
    }

    #[test]
    fn door2_absent_output_config_gets_pin_default() {
        // WHY (issue #22): pin/capture default applies only when the client left
        // output_config unset. Same precedence as Door-1 / OpenAI-compat chat.
        let body = serde_json::json!({
            "model": "sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}]
        });
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), false)
                .expect("prepare ok");
        assert_eq!(
            prepared
                .body()
                .get("output_config")
                .and_then(|o| o.get("effort"))
                .and_then(|e| e.as_str()),
            Some("high"),
            "sonnet pin default effort when client omitted output_config"
        );

        let body = serde_json::json!({
            "model": "claude-sonnet-5",
            "max_tokens": 100,
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
        assert_eq!(
            req.output_config.as_ref().and_then(|o| o.effort.as_deref()),
            Some("high"),
            "Sonnet pin default when client omitted output_config"
        );

        // Haiku pin has no output_effort: stay absent (do not invent).
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
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
        assert!(
            req.output_config.is_none(),
            "haiku pin has no output_effort: {:?}",
            req.output_config
        );
    }

    #[test]
    fn door2_client_output_config_on_haiku_is_not_stripped() {
        // WHY (issue #22): even when the pin has no effort surface, a client
        // that set output_config is honored (pass-through). Upstream may 400;
        // silent strip of client intent is the bug this issue closes.
        let body = serde_json::json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "output_config": {"effort": "high"}
        });
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), false)
                .expect("prepare ok");
        assert_eq!(
            prepared
                .body()
                .get("output_config")
                .and_then(|o| o.get("effort"))
                .and_then(|e| e.as_str()),
            Some("high"),
            "client-set output_config on haiku must not be fingerprint-stripped"
        );
    }

    #[test]
    fn door2_output_config_preserves_format_and_format_only() {
        // WHY (issue #22 panel): OutputConfig must not silently drop non-effort
        // members (e.g. format) or 400 on format-only objects. Client object
        // is pass-through; pin fills only when the whole field is absent.
        let body = serde_json::json!({
            "model": "sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "output_config": {
                "effort": "low",
                "format": {"type": "json_schema", "schema": {"type": "object"}}
            }
        });
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), false)
                .expect("prepare ok");
        let oc = prepared.body().get("output_config").expect("output_config");
        assert_eq!(oc.get("effort").and_then(|e| e.as_str()), Some("low"));
        assert_eq!(
            oc.get("format")
                .and_then(|f| f.get("type"))
                .and_then(|t| t.as_str()),
            Some("json_schema"),
            "format must survive round-trip (not effort-only strip)"
        );

        let body = serde_json::json!({
            "model": "sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say OK"}],
            "output_config": {
                "format": {"type": "json_object"}
            }
        });
        let prepared =
            prepare_client_messages_request(body, default_profile(), &empty_repl(), false)
                .expect("format-only must not fail deserialize");
        let oc = prepared.body().get("output_config").expect("output_config");
        assert!(
            oc.get("effort").is_none(),
            "format-only must not invent effort: {oc:?}"
        );
        assert_eq!(
            oc.get("format")
                .and_then(|f| f.get("type"))
                .and_then(|t| t.as_str()),
            Some("json_object")
        );
        // Client set output_config (format-only): pin must not inject effort.
        assert_ne!(
            oc.get("effort").and_then(|e| e.as_str()),
            Some("high"),
            "pin must not merge effort onto client-set format-only output_config"
        );
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
    fn prepared_body_has_billing_marker_and_no_cch_sentinel() {
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

#[cfg(test)]
mod issue_51_tool_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_anthropic_preserves_root_ref() {
        let schema = json!({"$defs":{"args":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},"$ref":"#/$defs/args"});
        let raw = json!({"model":"sonnet","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","input_schema":schema}]});
        let wire = prepare_client_messages_request(
            raw,
            crate::fingerprint::default_profile(),
            &Replacements::empty(),
            false,
        )
        .unwrap();
        assert_eq!(wire.body()["tools"][0]["input_schema"], schema);
    }

    #[test]
    fn native_anthropic_strict_passthrough_and_invalid_input() {
        let make = |schema: Value, strict: Value| json!({"model":"sonnet","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","input_schema":schema,"strict":strict}]});
        let schema = json!({"type":"object","additionalProperties":false,"properties":{"x":{"type":"string"}}});
        let wire = prepare_client_messages_request(
            make(schema.clone(), json!(true)),
            crate::fingerprint::default_profile(),
            &Replacements::empty(),
            false,
        )
        .unwrap();
        assert_eq!(wire.body()["tools"][0]["strict"], true);
        assert_eq!(wire.body()["tools"][0]["input_schema"], schema);
        let loose = json!({"type":"object","properties":{"x":{"allOf":[{"type":"string"}]}}});
        let wire = prepare_client_messages_request(
            make(loose.clone(), json!(true)),
            crate::fingerprint::default_profile(),
            &Replacements::empty(),
            false,
        )
        .unwrap();
        assert!(wire.body()["tools"][0].get("strict").is_none());
        assert_eq!(wire.body()["tools"][0]["input_schema"], loose);
        assert!(matches!(
            prepare_count_tokens_request(
                make(json!({"anyOf":[{}]}), json!(false)),
                crate::fingerprint::default_profile(),
                &Replacements::empty()
            ),
            Err(ProviderError::BadRequest(_))
        ));
        for bad in [
            json!({"anyOf":[{}]}),
            json!({"type":"object","properties":{"x":{"allOf":[]}}}),
        ] {
            assert!(matches!(
                prepare_client_messages_request(
                    make(bad, json!(false)),
                    crate::fingerprint::default_profile(),
                    &Replacements::empty(),
                    false
                ),
                Err(ProviderError::BadRequest(_))
            ));
        }
    }
}
