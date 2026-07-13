//! Anthropic Messages request/response translation through the canonical model.
//!
//! Used by the dual-mode `/v1/messages` path for **Grok** and **Codex** only.
//! Claude stays on native passthrough and never calls these mappers.
//!
//! Contracts: `docs/anthropic-frontend-multi-backend-plan.md` §5–§6.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_util::{Stream, StreamExt};
use serde_json::{Map, Value};
use tracing::debug;

use omni_core::{
    CanonicalBlock, CanonicalContent, CanonicalImageSource, CanonicalMessage, CanonicalReasoning,
    CanonicalRequest, CanonicalResponse, CanonicalStream, CanonicalStreamEvent, CanonicalTool,
    CanonicalToolChoice, CanonicalUsage,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Client-facing request translation failure (maps to Anthropic 400).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct AnthropicMapError(pub String);

impl AnthropicMapError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

/// Upstream / protocol failure on the response side (maps to 502 / SSE error).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct AnthropicProtocolError(pub String);

impl AnthropicProtocolError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

// ---------------------------------------------------------------------------
// is_error wire encoding (Grok/Codex OpenAI-compat tool messages)
// ---------------------------------------------------------------------------

/// Encode a canonical tool result for OpenAI-style `role:tool` / function_call_output.
///
/// Normative (plan §5.2):
/// - `is_error && content.empty` → `"error"`
/// - `is_error && !content.empty` → `"ERROR: " + content`
/// - `!is_error` → content as-is
pub fn encode_tool_result_content(content: &str, is_error: bool) -> String {
    if is_error {
        if content.is_empty() {
            "error".to_string()
        } else {
            format!("ERROR: {content}")
        }
    } else {
        content.to_string()
    }
}

// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------

/// Parse JSON object bytes and reject **duplicate top-level keys**.
pub fn parse_anthropic_object_no_dup_keys(bytes: &[u8]) -> Result<Value, AnthropicMapError> {
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let value = deserialize_value_no_dup_top_level(&mut de)
        .map_err(|e| AnthropicMapError::new(format!("invalid JSON: {e}")))?;
    if !value.is_object() {
        return Err(AnthropicMapError::new("request body must be a JSON object"));
    }
    Ok(value)
}

/// Peek the `model` string from raw body bytes without mutating the body.
/// Returns `None` if model is missing or not a string.
pub fn peek_model_string(bytes: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(bytes).ok()?;
    value
        .get("model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Deserialize a JSON value, rejecting duplicate keys only at the top-level object.
fn deserialize_value_no_dup_top_level<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, MapAccess, Visitor};

    struct TopLevelVisitor;

    impl<'de> Visitor<'de> for TopLevelVisitor {
        type Value = Value;

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a JSON object")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut out = Map::new();
            let mut seen = HashSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if !seen.insert(key.clone()) {
                    return Err(de::Error::custom(format!("duplicate top-level key: {key}")));
                }
                // Nested values: normal serde_json (duplicates only matter top-level).
                let value: Value = map.next_value()?;
                out.insert(key, value);
            }
            Ok(Value::Object(out))
        }
    }

    deserializer.deserialize_map(TopLevelVisitor)
}

// ---------------------------------------------------------------------------
// anthropic_to_canonical
// ---------------------------------------------------------------------------

/// Translate an Anthropic Messages request body into a [`CanonicalRequest`].
///
/// `provider_id` selects provider-specific extras allowlist behavior
/// (`stop_sequences`, `parallel_tool_calls`). Only `"grok"` and `"codex"` are
/// intended call sites.
pub fn anthropic_to_canonical(
    body: &Value,
    provider_id: &str,
) -> Result<CanonicalRequest, AnthropicMapError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| AnthropicMapError::new("model is required"))?
        .to_string();

    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| AnthropicMapError::new("max_tokens is required"))?;
    let max_tokens = u32::try_from(max_tokens)
        .map_err(|_| AnthropicMapError::new("max_tokens out of range"))?;

    let messages_val = body
        .get("messages")
        .ok_or_else(|| AnthropicMapError::new("messages is required"))?;
    let messages_arr = messages_val
        .as_array()
        .ok_or_else(|| AnthropicMapError::new("messages must be an array"))?;
    if messages_arr.is_empty() {
        return Err(AnthropicMapError::new("messages must not be empty"));
    }

    // Validation order (plan §5.1):
    // (1) parse → (2) prefill → (3) drop thinking → (4) strip empty →
    // (5) last must be user → (6) tool pairing
    // Prefill check is on raw messages before thinking strip.
    if let Some(last) = messages_arr.last() {
        if last.get("role").and_then(|r| r.as_str()) == Some("assistant") {
            return Err(AnthropicMapError::new(
                "trailing assistant prefill is not supported on translated Anthropic path",
            ));
        }
    }

    let mut messages: Vec<CanonicalMessage> = Vec::new();

    // Top-level system → one system message
    if let Some(system) = body.get("system") {
        if let Some(text) = system_to_text(system)? {
            if !text.is_empty() {
                messages.push(CanonicalMessage {
                    role: "system".into(),
                    content: CanonicalContent::Text(text),
                });
            }
        }
    }

    // Parse conversation messages with thinking drop + empty strip inline.
    let mut raw_msgs: Vec<RawMsg> = Vec::new();
    for (i, m) in messages_arr.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| AnthropicMapError::new(format!("messages[{i}]: role is required")))?;
        if role == "system" {
            return Err(AnthropicMapError::new(
                "mid-conversation role \"system\" is not supported; use top-level system",
            ));
        }
        if role != "user" && role != "assistant" {
            return Err(AnthropicMapError::new(format!(
                "messages[{i}]: unsupported role \"{role}\""
            )));
        }
        let content = m
            .get("content")
            .ok_or_else(|| AnthropicMapError::new(format!("messages[{i}]: content is required")))?;
        let blocks = parse_message_blocks(role, content, i)?;
        raw_msgs.push(RawMsg {
            role: role.to_string(),
            blocks,
        });
    }

    // Drop thinking / redacted_thinking; strip empty; fail on same-role adjacency after drop.
    let mut normalized: Vec<RawMsg> = Vec::new();
    for mut msg in raw_msgs {
        msg.blocks.retain(|b| {
            !matches!(
                b,
                RawBlock::Thinking | RawBlock::RedactedThinking
            )
        });
        if msg.blocks.is_empty() {
            continue;
        }
        if let Some(prev) = normalized.last() {
            if prev.role == msg.role {
                return Err(AnthropicMapError::new(format!(
                    "after dropping thinking blocks, adjacent same-role messages (\"{}\") are not supported",
                    msg.role
                )));
            }
        }
        normalized.push(msg);
    }

    if normalized.is_empty() {
        return Err(AnthropicMapError::new(
            "conversation is empty after removing thinking blocks",
        ));
    }
    if normalized.last().map(|m| m.role.as_str()) != Some("user") {
        return Err(AnthropicMapError::new(
            "last message must be a user message after normalization",
        ));
    }

    // Tool-history rewrite + pairing (§5.2)
    let rewritten = rewrite_tool_history(normalized)?;
    messages.extend(rewritten);

    // Tools
    let tools = map_tools(body.get("tools"))?;
    let tool_choice = map_tool_choice(body.get("tool_choice"), provider_id)?;

    // Sampling
    let temperature = body
        .get("temperature")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let top_p = body.get("top_p").and_then(|v| v.as_f64()).map(|f| f as f32);
    // top_k: drop (documented lossy)

    // metadata → session only
    let mut metadata = HashMap::new();
    if let Some(uid) = body
        .get("metadata")
        .and_then(|m| m.get("user_id"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        metadata.insert("user_id".into(), uid.to_string());
    }

    // thinking request config → CanonicalReasoning
    let reasoning = map_thinking_config(body.get("thinking"))?;

    // provider extras: stop_sequences, parallel_tool_calls
    let provider_extras = build_provider_extras(body, provider_id);

    // stream is handler control only — not placed in canonical

    Ok(CanonicalRequest {
        model,
        messages,
        tools,
        tool_choice,
        max_tokens: Some(max_tokens),
        temperature,
        top_p,
        reasoning,
        metadata,
        provider_extras,
    })
}

struct RawMsg {
    role: String,
    blocks: Vec<RawBlock>,
}

enum RawBlock {
    Text(String),
    Image { source: CanonicalImageSource },
    ToolUse {
        id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    Thinking,
    RedactedThinking,
}

fn system_to_text(system: &Value) -> Result<Option<String>, AnthropicMapError> {
    match system {
        Value::String(s) => Ok(Some(s.clone())),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for (i, b) in blocks.iter().enumerate() {
                let ty = b
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                match ty {
                    "text" => {
                        parts.push(
                            b.get("text")
                                .and_then(|t| t.as_str())
                                .unwrap_or("")
                                .to_string(),
                        );
                    }
                    "" if b.get("text").is_some() => {
                        parts.push(b.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string());
                    }
                    other => {
                        return Err(AnthropicMapError::new(format!(
                            "system[{i}]: unsupported block type \"{other}\""
                        )));
                    }
                }
            }
            Ok(Some(parts.join("\n")))
        }
        Value::Null => Ok(None),
        _ => Err(AnthropicMapError::new(
            "system must be a string or array of text blocks",
        )),
    }
}

fn parse_message_blocks(
    role: &str,
    content: &Value,
    msg_idx: usize,
) -> Result<Vec<RawBlock>, AnthropicMapError> {
    match content {
        Value::String(s) => Ok(vec![RawBlock::Text(s.clone())]),
        Value::Array(arr) => {
            let mut blocks = Vec::with_capacity(arr.len());
            for (bi, b) in arr.iter().enumerate() {
                let ty = b
                    .get("type")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| {
                        AnthropicMapError::new(format!(
                            "messages[{msg_idx}].content[{bi}]: type is required"
                        ))
                    })?;
                match ty {
                    "text" => {
                        let text = b
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();
                        blocks.push(RawBlock::Text(text));
                    }
                    "image" => {
                        if role == "assistant" {
                            return Err(AnthropicMapError::new(format!(
                                "messages[{msg_idx}]: assistant images are not supported"
                            )));
                        }
                        if role != "user" {
                            return Err(AnthropicMapError::new(format!(
                                "messages[{msg_idx}]: image blocks only allowed on user messages"
                            )));
                        }
                        let source = b.get("source").ok_or_else(|| {
                            AnthropicMapError::new(format!(
                                "messages[{msg_idx}].content[{bi}]: image missing source"
                            ))
                        })?;
                        let src_ty = source
                            .get("type")
                            .and_then(|t| t.as_str())
                            .ok_or_else(|| {
                                AnthropicMapError::new(format!(
                                    "messages[{msg_idx}].content[{bi}]: image source.type required"
                                ))
                            })?;
                        let img = match src_ty {
                            "base64" => {
                                let media_type = source
                                    .get("media_type")
                                    .and_then(|m| m.as_str())
                                    .filter(|s| !s.is_empty())
                                    .ok_or_else(|| {
                                        AnthropicMapError::new(format!(
                                            "messages[{msg_idx}].content[{bi}]: image base64 missing media_type"
                                        ))
                                    })?;
                                let data = source
                                    .get("data")
                                    .and_then(|d| d.as_str())
                                    .filter(|s| !s.is_empty())
                                    .ok_or_else(|| {
                                        AnthropicMapError::new(format!(
                                            "messages[{msg_idx}].content[{bi}]: image base64 missing data"
                                        ))
                                    })?;
                                CanonicalImageSource::Base64 {
                                    media_type: media_type.to_string(),
                                    data: data.to_string(),
                                }
                            }
                            "url" => {
                                let url = source
                                    .get("url")
                                    .and_then(|u| u.as_str())
                                    .filter(|s| !s.trim().is_empty())
                                    .ok_or_else(|| {
                                        AnthropicMapError::new(format!(
                                            "messages[{msg_idx}].content[{bi}]: image url missing url"
                                        ))
                                    })?;
                                CanonicalImageSource::Url {
                                    url: url.to_string(),
                                }
                            }
                            other => {
                                return Err(AnthropicMapError::new(format!(
                                    "messages[{msg_idx}].content[{bi}]: unsupported image source.type \"{other}\""
                                )));
                            }
                        };
                        blocks.push(RawBlock::Image { source: img });
                    }
                    "tool_use" => {
                        if role != "assistant" {
                            return Err(AnthropicMapError::new(format!(
                                "messages[{msg_idx}]: tool_use only allowed on assistant messages"
                            )));
                        }
                        let id = b
                            .get("id")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                AnthropicMapError::new(format!(
                                    "messages[{msg_idx}].content[{bi}]: tool_use.id required and non-empty"
                                ))
                            })?
                            .to_string();
                        let name = b
                            .get("name")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                AnthropicMapError::new(format!(
                                    "messages[{msg_idx}].content[{bi}]: tool_use.name required and non-empty"
                                ))
                            })?
                            .to_string();
                        let input = b.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
                        if !input.is_object() {
                            return Err(AnthropicMapError::new(format!(
                                "messages[{msg_idx}].content[{bi}]: tool_use.input must be an object"
                            )));
                        }
                        let arguments = serde_json::to_string(&input).map_err(|e| {
                            AnthropicMapError::new(format!(
                                "messages[{msg_idx}].content[{bi}]: tool_use.input serialize: {e}"
                            ))
                        })?;
                        blocks.push(RawBlock::ToolUse {
                            id,
                            name,
                            arguments,
                        });
                    }
                    "tool_result" => {
                        if role != "user" {
                            return Err(AnthropicMapError::new(format!(
                                "messages[{msg_idx}]: tool_result only allowed on user messages"
                            )));
                        }
                        let tool_use_id = b
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                AnthropicMapError::new(format!(
                                    "messages[{msg_idx}].content[{bi}]: tool_result.tool_use_id required"
                                ))
                            })?
                            .to_string();
                        let content_str = tool_result_content(b.get("content"), msg_idx, bi)?;
                        let is_error = b
                            .get("is_error")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        blocks.push(RawBlock::ToolResult {
                            tool_use_id,
                            content: content_str,
                            is_error,
                        });
                    }
                    "thinking" => blocks.push(RawBlock::Thinking),
                    "redacted_thinking" => blocks.push(RawBlock::RedactedThinking),
                    "document" => {
                        return Err(AnthropicMapError::new(format!(
                            "messages[{msg_idx}].content[{bi}]: document blocks are not supported"
                        )));
                    }
                    other => {
                        return Err(AnthropicMapError::new(format!(
                            "messages[{msg_idx}].content[{bi}]: unknown content block type \"{other}\""
                        )));
                    }
                }
            }
            Ok(blocks)
        }
        _ => Err(AnthropicMapError::new(format!(
            "messages[{msg_idx}]: content must be a string or array of blocks"
        ))),
    }
}

fn tool_result_content(
    content: Option<&Value>,
    msg_idx: usize,
    bi: usize,
) -> Result<String, AnthropicMapError> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(arr)) => {
            let mut parts = Vec::new();
            for (ci, c) in arr.iter().enumerate() {
                let ty = c.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                match ty {
                    "text" => {
                        parts.push(
                            c.get("text")
                                .and_then(|t| t.as_str())
                                .unwrap_or("")
                                .to_string(),
                        );
                    }
                    "image" => {
                        return Err(AnthropicMapError::new(format!(
                            "messages[{msg_idx}].content[{bi}]: tool_result image content is not supported"
                        )));
                    }
                    other => {
                        return Err(AnthropicMapError::new(format!(
                            "messages[{msg_idx}].content[{bi}].content[{ci}]: unsupported type \"{other}\""
                        )));
                    }
                }
            }
            Ok(parts.join("\n"))
        }
        Some(_) => Err(AnthropicMapError::new(format!(
            "messages[{msg_idx}].content[{bi}]: tool_result.content must be string or text array"
        ))),
    }
}

/// Rewrite Anthropic tool history into canonical tool-role messages.
///
/// After any assistant turn with ≥1 tool_use, the immediately next user message
/// must resolve every outstanding id exactly once. Tool results emit first, then
/// trailing user text/images (documented lossy reorder).
fn rewrite_tool_history(msgs: Vec<RawMsg>) -> Result<Vec<CanonicalMessage>, AnthropicMapError> {
    let mut out: Vec<CanonicalMessage> = Vec::new();
    let mut i = 0;
    while i < msgs.len() {
        let msg = &msgs[i];
        if msg.role == "assistant" {
            // Validate unique tool_use ids within this turn
            let mut ids_in_turn = HashSet::new();
            let mut tool_uses = Vec::new();
            let mut text_parts = Vec::new();
            for b in &msg.blocks {
                match b {
                    RawBlock::Text(t) => {
                        if !t.is_empty() {
                            text_parts.push(t.clone());
                        }
                    }
                    RawBlock::ToolUse {
                        id,
                        name,
                        arguments,
                    } => {
                        if !ids_in_turn.insert(id.clone()) {
                            return Err(AnthropicMapError::new(format!(
                                "duplicate tool_use.id \"{id}\" in assistant turn"
                            )));
                        }
                        tool_uses.push((id.clone(), name.clone(), arguments.clone()));
                    }
                    RawBlock::Image { .. } => {
                        return Err(AnthropicMapError::new(
                            "assistant images are not supported",
                        ));
                    }
                    RawBlock::ToolResult { .. } => {
                        return Err(AnthropicMapError::new(
                            "tool_result is not allowed on assistant messages",
                        ));
                    }
                    RawBlock::Thinking | RawBlock::RedactedThinking => {}
                }
            }

            // Build assistant message
            if tool_uses.is_empty() {
                let text = text_parts.join("");
                out.push(CanonicalMessage {
                    role: "assistant".into(),
                    content: CanonicalContent::Text(text),
                });
            } else {
                let mut blocks = Vec::new();
                let text = text_parts.join("");
                if !text.is_empty() {
                    blocks.push(CanonicalBlock::Text(text));
                }
                for (id, name, arguments) in &tool_uses {
                    blocks.push(CanonicalBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                }
                out.push(CanonicalMessage {
                    role: "assistant".into(),
                    content: CanonicalContent::Blocks(blocks),
                });

                // Next message must resolve all ids
                i += 1;
                if i >= msgs.len() {
                    return Err(AnthropicMapError::new(
                        "assistant tool_use must be followed by a user message with tool_result blocks",
                    ));
                }
                let next = &msgs[i];
                if next.role != "user" {
                    return Err(AnthropicMapError::new(
                        "assistant tool_use must be immediately followed by a user message resolving tool results",
                    ));
                }
                let mut outstanding: HashSet<String> = tool_uses.iter().map(|(id, _, _)| id.clone()).collect();
                let mut seen_results: HashSet<String> = HashSet::new();
                let mut result_msgs = Vec::new();
                let mut trailing: Vec<CanonicalBlock> = Vec::new();

                for b in &next.blocks {
                    match b {
                        RawBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            if !outstanding.contains(tool_use_id) {
                                return Err(AnthropicMapError::new(format!(
                                    "tool_result for unknown or already-resolved id \"{tool_use_id}\""
                                )));
                            }
                            if !seen_results.insert(tool_use_id.clone()) {
                                return Err(AnthropicMapError::new(format!(
                                    "duplicate tool_result for id \"{tool_use_id}\""
                                )));
                            }
                            outstanding.remove(tool_use_id);
                            result_msgs.push(CanonicalMessage {
                                role: "tool".into(),
                                content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                                    tool_use_id: tool_use_id.clone(),
                                    content: content.clone(),
                                    is_error: *is_error,
                                }]),
                            });
                        }
                        RawBlock::Text(t) => {
                            if !t.is_empty() {
                                trailing.push(CanonicalBlock::Text(t.clone()));
                            }
                        }
                        RawBlock::Image { source } => {
                            trailing.push(CanonicalBlock::Image {
                                source: source.clone(),
                            });
                        }
                        RawBlock::ToolUse { .. } => {
                            return Err(AnthropicMapError::new(
                                "tool_use is not allowed on user messages",
                            ));
                        }
                        RawBlock::Thinking | RawBlock::RedactedThinking => {}
                    }
                }
                if !outstanding.is_empty() {
                    let mut missing: Vec<_> = outstanding.into_iter().collect();
                    missing.sort();
                    return Err(AnthropicMapError::new(format!(
                        "user message missing tool_result for id(s): {}",
                        missing.join(", ")
                    )));
                }
                out.extend(result_msgs);
                if !trailing.is_empty() {
                    let content = if trailing.len() == 1 {
                        match &trailing[0] {
                            CanonicalBlock::Text(t) => CanonicalContent::Text(t.clone()),
                            _ => CanonicalContent::Blocks(trailing),
                        }
                    } else {
                        CanonicalContent::Blocks(trailing)
                    };
                    out.push(CanonicalMessage {
                        role: "user".into(),
                        content,
                    });
                }
            }
        } else {
            // User message without being a tool-result resolver for a prior tool_use.
            // Unpaired tool_result → 400.
            let mut trailing = Vec::new();
            for b in &msg.blocks {
                match b {
                    RawBlock::ToolResult { tool_use_id, .. } => {
                        return Err(AnthropicMapError::new(format!(
                            "unpaired tool_result for id \"{tool_use_id}\" (no preceding assistant tool_use)"
                        )));
                    }
                    RawBlock::Text(t) => {
                        if !t.is_empty() {
                            trailing.push(CanonicalBlock::Text(t.clone()));
                        }
                    }
                    RawBlock::Image { source } => {
                        trailing.push(CanonicalBlock::Image {
                            source: source.clone(),
                        });
                    }
                    RawBlock::ToolUse { .. } => {
                        return Err(AnthropicMapError::new(
                            "tool_use is not allowed on user messages",
                        ));
                    }
                    RawBlock::Thinking | RawBlock::RedactedThinking => {}
                }
            }
            if trailing.is_empty() {
                // empty user after strip — skip (already stripped earlier for thinking-only)
                out.push(CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text(String::new()),
                });
            } else if trailing.len() == 1 {
                match &trailing[0] {
                    CanonicalBlock::Text(t) => {
                        out.push(CanonicalMessage {
                            role: "user".into(),
                            content: CanonicalContent::Text(t.clone()),
                        });
                    }
                    _ => {
                        out.push(CanonicalMessage {
                            role: "user".into(),
                            content: CanonicalContent::Blocks(trailing),
                        });
                    }
                }
            } else {
                out.push(CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Blocks(trailing),
                });
            }
        }
        i += 1;
    }
    Ok(out)
}

fn map_tools(tools: Option<&Value>) -> Result<Option<Vec<CanonicalTool>>, AnthropicMapError> {
    let Some(tools) = tools else {
        return Ok(None);
    };
    let arr = tools
        .as_array()
        .ok_or_else(|| AnthropicMapError::new("tools must be an array"))?;
    if arr.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(arr.len());
    for (i, t) in arr.iter().enumerate() {
        // Hosted/server tools (computer use, etc.) lack plain name + input_schema
        let name = t
            .get("name")
            .and_then(|n| n.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                AnthropicMapError::new(format!(
                    "tools[{i}]: only function tools with name + input_schema are supported"
                ))
            })?;
        let schema = t.get("input_schema").ok_or_else(|| {
            AnthropicMapError::new(format!(
                "tools[{i}]: input_schema is required (hosted/server tools are not supported)"
            ))
        })?;
        if !schema.is_object() {
            return Err(AnthropicMapError::new(format!(
                "tools[{i}]: input_schema must be an object"
            )));
        }
        let mut parameters = schema.clone();
        // Ensure object properties default {}
        if parameters.get("type").and_then(|t| t.as_str()) == Some("object")
            && parameters.get("properties").is_none()
        {
            parameters
                .as_object_mut()
                .unwrap()
                .insert("properties".into(), serde_json::json!({}));
        }
        let description = t
            .get("description")
            .and_then(|d| d.as_str())
            .map(|s| s.to_string());
        out.push(CanonicalTool {
            name: name.to_string(),
            description,
            parameters,
        });
    }
    Ok(Some(out))
}

fn map_tool_choice(
    choice: Option<&Value>,
    provider_id: &str,
) -> Result<Option<CanonicalToolChoice>, AnthropicMapError> {
    let Some(choice) = choice else {
        return Ok(None);
    };
    let ty = choice
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| AnthropicMapError::new("tool_choice.type is required"))?;
    let mapped = match ty {
        "auto" => CanonicalToolChoice::Auto,
        "any" => CanonicalToolChoice::Required,
        "none" => CanonicalToolChoice::None,
        "tool" => {
            let name = choice
                .get("name")
                .and_then(|n| n.as_str())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    AnthropicMapError::new("tool_choice.type=tool requires non-empty name")
                })?;
            CanonicalToolChoice::Specific {
                name: name.to_string(),
            }
        }
        other => {
            return Err(AnthropicMapError::new(format!(
                "unsupported tool_choice.type \"{other}\""
            )));
        }
    };
    // disable_parallel_tool_use is handled in build_provider_extras
    let _ = provider_id;
    Ok(Some(mapped))
}

fn map_thinking_config(
    thinking: Option<&Value>,
) -> Result<Option<CanonicalReasoning>, AnthropicMapError> {
    let Some(thinking) = thinking else {
        return Ok(None);
    };
    if thinking.is_null() {
        return Ok(None);
    }
    let ty = thinking.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
        "enabled" | "adaptive" | "" => {
            let budget = thinking
                .get("budget_tokens")
                .and_then(|v| v.as_u64())
                .and_then(|v| u32::try_from(v).ok());
            let effort = thinking
                .get("effort")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            if budget.is_none() && effort.is_none() && ty.is_empty() {
                return Ok(None);
            }
            Ok(Some(CanonicalReasoning {
                effort,
                budget_tokens: budget,
            }))
        }
        "disabled" => Ok(None),
        other => Err(AnthropicMapError::new(format!(
            "unsupported thinking.type \"{other}\""
        ))),
    }
}

fn build_provider_extras(body: &Value, provider_id: &str) -> Option<Value> {
    let mut extras = Map::new();

    // stop_sequences → stop (Grok); drop + log for Codex
    if let Some(stops) = body.get("stop_sequences").and_then(|v| v.as_array()) {
        if !stops.is_empty() {
            if provider_id == "grok" {
                extras.insert("stop".into(), Value::Array(stops.clone()));
            } else if provider_id == "codex" {
                debug!(
                    count = stops.len(),
                    "dropping stop_sequences for codex (not mapped on translated Anthropic path)"
                );
            }
        }
    }

    // tool_choice.disable_parallel_tool_use → parallel_tool_calls: false
    if let Some(tc) = body.get("tool_choice") {
        if let Some(true) = tc
            .get("disable_parallel_tool_use")
            .and_then(|v| v.as_bool())
        {
            if provider_id == "grok" || provider_id == "codex" {
                extras.insert("parallel_tool_calls".into(), Value::Bool(false));
            } else {
                debug!("dropping disable_parallel_tool_use for provider {provider_id}");
            }
        }
    }

    if extras.is_empty() {
        None
    } else {
        Some(Value::Object(extras))
    }
}

// ---------------------------------------------------------------------------
// canonical_to_anthropic (non-stream)
// ---------------------------------------------------------------------------

/// Finish classification outcome for Anthropic stop_reason mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicFinishClass {
    Success { stop_reason: &'static str },
    Error { message: String },
}

/// Classify finish reason + emitted content per plan §6.1 priority order.
pub fn classify_anthropic_finish(
    finish_reason: Option<&str>,
    tool_count: usize,
    text_empty: bool,
) -> AnthropicFinishClass {
    let fr = finish_reason.unwrap_or("stop");
    let fr_l = fr.to_ascii_lowercase();

    // Priority 1: error class
    if matches!(
        fr_l.as_str(),
        "content_filter"
            | "refusal"
            | "error"
            | "cancel"
            | "cancelled"
            | "incomplete"
    ) {
        return AnthropicFinishClass::Error {
            message: format!("upstream finish_reason={fr}"),
        };
    }

    // Priority 2: length → max_tokens
    if fr_l == "length" || fr_l == "max_tokens" {
        return AnthropicFinishClass::Success {
            stop_reason: "max_tokens",
        };
    }

    // Priority 3: tool_use
    let toolish = fr_l == "tool_calls" || fr_l == "function_call" || fr_l == "tool_use";
    if toolish {
        if tool_count >= 1 {
            return AnthropicFinishClass::Success {
                stop_reason: "tool_use",
            };
        }
        return AnthropicFinishClass::Error {
            message: "finish_reason indicates tool_calls but no complete tool_use blocks".into(),
        };
    }
    if tool_count >= 1 {
        // ordinary stop + tools → tool_use
        return AnthropicFinishClass::Success {
            stop_reason: "tool_use",
        };
    }

    // Priority 4: ordinary stop
    let _ = text_empty;
    AnthropicFinishClass::Success {
        stop_reason: "end_turn",
    }
}

/// Parse tool-call arguments for Anthropic `input` object.
/// Empty/missing → `{}`; non-empty must be a JSON object or protocol error.
pub fn parse_tool_arguments_object(args: &str) -> Result<Value, AnthropicProtocolError> {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return Ok(serde_json::json!({}));
    }
    let value: Value = serde_json::from_str(trimmed).map_err(|e| {
        AnthropicProtocolError::new(format!("tool arguments are not valid JSON: {e}"))
    })?;
    if !value.is_object() {
        return Err(AnthropicProtocolError::new(
            "tool arguments must be a JSON object",
        ));
    }
    Ok(value)
}

/// Convert a [`CanonicalResponse`] to an Anthropic Messages JSON body.
pub fn canonical_to_anthropic(
    canon: &CanonicalResponse,
    model: &str,
) -> Result<Value, AnthropicProtocolError> {
    // No thinking blocks on translated wire.
    let mut content: Vec<Value> = Vec::new();

    if !canon.content.is_empty() {
        content.push(serde_json::json!({
            "type": "text",
            "text": canon.content,
        }));
    }

    for tc in &canon.tool_calls {
        if tc.id.is_empty() || tc.name.is_empty() {
            return Err(AnthropicProtocolError::new(
                "incomplete tool call (missing id or name)",
            ));
        }
        let input = parse_tool_arguments_object(&tc.arguments)?;
        content.push(serde_json::json!({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": input,
        }));
    }

    let tool_count = canon.tool_calls.len();
    let text_empty = canon.content.is_empty();

    // Refusal / content_filter from structured fields
    if let Some(ref refusal) = canon.refusal {
        if !refusal.is_empty() {
            return Err(AnthropicProtocolError::new(format!("refusal: {refusal}")));
        }
    }

    let class = classify_anthropic_finish(
        canon.finish_reason.as_deref(),
        tool_count,
        text_empty,
    );

    match class {
        AnthropicFinishClass::Error { message } => {
            Err(AnthropicProtocolError::new(message))
        }
        AnthropicFinishClass::Success { stop_reason } => {
            if content.is_empty() {
                // Empty success → one empty text block
                content.push(serde_json::json!({
                    "type": "text",
                    "text": "",
                }));
            }
            let id = canon
                .id
                .clone()
                .or_else(|| {
                    canon
                        .metadata
                        .as_ref()
                        .and_then(|m| m.id.clone())
                })
                .unwrap_or_else(|| format!("msg_{}", uuid_simple()));

            let mut usage = serde_json::json!({
                "input_tokens": canon.usage.input_tokens,
                "output_tokens": canon.usage.output_tokens,
            });
            if canon.usage.cache_read > 0 {
                usage["cache_read_input_tokens"] = serde_json::json!(canon.usage.cache_read);
            }
            if canon.usage.cache_creation > 0 {
                usage["cache_creation_input_tokens"] =
                    serde_json::json!(canon.usage.cache_creation);
            }

            Ok(serde_json::json!({
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": null,
                "usage": usage,
            }))
        }
    }
}

fn uuid_simple() -> String {
    // Lightweight id without pulling uuid if not needed; use timestamp+random-ish.
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{t:x}")
}

// ---------------------------------------------------------------------------
// SSE framer (plan §6.2)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ToolAccum {
    id: String,
    name: String,
    args: String,
    anth_index: Option<usize>,
    started: bool,
}

/// Frame a canonical stream as Anthropic Messages SSE events.
///
/// Success terminalization only after success-class Finish **and** clean stream
/// end. Bare EOF without Finish → SSE `error` (no `message_stop`).
pub fn sse_from_canonical_stream_anthropic(
    stream: CanonicalStream,
    model: String,
    message_id: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let body = anthropic_sse_body(stream, model, message_id);
    Sse::new(crate::span_stream::SpannedStream::current(Box::pin(body)))
}

fn anthropic_sse_body(
    mut stream: CanonicalStream,
    model: String,
    message_id: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let mut next_index: usize = 0;
        let mut open_text: Option<usize> = None;
        let mut tools: HashMap<u32, ToolAccum> = HashMap::new();
        let mut tool_order: Vec<u32> = Vec::new();
        let mut saw_tool_use = false;
        let mut usage = CanonicalUsage::default();
        let mut pending_finish: Option<String> = None;
        let mut text_buf = String::new();
        let mut tools_phase = false; // true once we start emitting tools

        // Buffer tool deltas until text phase ends (wire order: text then tools).
        let mut buffered_tool_events: Vec<(u32, Option<String>, Option<String>, String)> = Vec::new();

        // Emit message_start first (handler must open upstream before framing).
        {
            let start = serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            });
            yield Ok(Event::default().event("message_start").data(start.to_string()));
        }

        let mut stream_error: Option<String> = None;
        let mut saw_finish = false;

        while let Some(item) = stream.next().await {
            match item {
                Ok(CanonicalStreamEvent::TextDelta(text)) => {
                    if pending_finish.is_some() {
                        continue; // ignore content after Finish
                    }
                    if tools_phase || saw_tool_use {
                        // Late text after tools started: drop + debug log
                        debug!("dropping late TextDelta after tool_use started on Anthropic translate path");
                        continue;
                    }
                    // If we have buffered tools waiting, still in text phase.
                    if open_text.is_none() {
                        let idx = next_index;
                        next_index += 1;
                        open_text = Some(idx);
                        let start = serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": { "type": "text", "text": "" }
                        });
                        yield Ok(Event::default().event("content_block_start").data(start.to_string()));
                    }
                    let idx = open_text.unwrap();
                    text_buf.push_str(&text);
                    let delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "text_delta", "text": text }
                    });
                    yield Ok(Event::default().event("content_block_delta").data(delta.to_string()));
                }
                Ok(CanonicalStreamEvent::ToolCallDelta { index, id, name, arguments_delta }) => {
                    if pending_finish.is_some() {
                        continue;
                    }
                    // Buffer until text phase ends; flush on Finish or when we decide.
                    buffered_tool_events.push((index, id, name, arguments_delta));
                }
                Ok(CanonicalStreamEvent::Usage(u)) => {
                    // Replace with latest full snapshot — never sum
                    usage = u;
                }
                Ok(CanonicalStreamEvent::Finish { finish_reason }) => {
                    saw_finish = true;
                    // Flush buffered tools (text-then-tools wire order) then classify.
                    {
                        // close open text first
                        if let Some(idx) = open_text.take() {
                            let stop = serde_json::json!({
                                "type": "content_block_stop",
                                "index": idx
                            });
                            yield Ok(Event::default().event("content_block_stop").data(stop.to_string()));
                        }
                        tools_phase = true;
                        for (index, id, name, arguments_delta) in buffered_tool_events.drain(..) {
                            let entry = tools.entry(index).or_insert_with(|| {
                                tool_order.push(index);
                                ToolAccum {
                                    id: String::new(),
                                    name: String::new(),
                                    args: String::new(),
                                    anth_index: None,
                                    started: false,
                                }
                            });
                            if let Some(i) = id {
                                if !entry.id.is_empty() && entry.id != i {
                                    stream_error = Some(format!(
                                        "conflicting tool call id for index {index}"
                                    ));
                                    break;
                                }
                                entry.id = i;
                            }
                            if let Some(n) = name {
                                if !entry.name.is_empty() && entry.name != n {
                                    stream_error = Some(format!(
                                        "conflicting tool call name for index {index}"
                                    ));
                                    break;
                                }
                                entry.name = n;
                            }
                            entry.args.push_str(&arguments_delta);
                        }
                        if stream_error.is_some() {
                            break;
                        }
                        // Check duplicate ids across tools
                        let mut seen_ids = HashSet::new();
                        for idx in &tool_order {
                            if let Some(t) = tools.get(idx) {
                                if !t.id.is_empty() && !seen_ids.insert(t.id.clone()) {
                                    stream_error = Some(format!(
                                        "duplicate tool call id \"{}\"",
                                        t.id
                                    ));
                                    break;
                                }
                            }
                        }
                        if stream_error.is_some() {
                            break;
                        }
                        // Emit tools sequentially
                        for idx in tool_order.clone() {
                            let Some(t) = tools.get_mut(&idx) else { continue };
                            if t.id.is_empty() || t.name.is_empty() {
                                stream_error = Some(
                                    "incomplete tool call (missing id or name)".into(),
                                );
                                break;
                            }
                            // validate args
                            if let Err(e) = parse_tool_arguments_object(&t.args) {
                                stream_error = Some(e.0);
                                break;
                            }
                            let anth_idx = next_index;
                            next_index += 1;
                            t.anth_index = Some(anth_idx);
                            t.started = true;
                            saw_tool_use = true;
                            let start = serde_json::json!({
                                "type": "content_block_start",
                                "index": anth_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": t.id,
                                    "name": t.name,
                                    "input": {}
                                }
                            });
                            yield Ok(Event::default().event("content_block_start").data(start.to_string()));
                            if !t.args.is_empty() {
                                let delta = serde_json::json!({
                                    "type": "content_block_delta",
                                    "index": anth_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": t.args
                                    }
                                });
                                yield Ok(Event::default().event("content_block_delta").data(delta.to_string()));
                            }
                            let stop = serde_json::json!({
                                "type": "content_block_stop",
                                "index": anth_idx
                            });
                            yield Ok(Event::default().event("content_block_stop").data(stop.to_string()));
                        }
                    }

                    if stream_error.is_some() {
                        break;
                    }

                    let complete_tools = tools.values().filter(|t| t.started).count();
                    let text_empty = text_buf.is_empty();
                    let class = classify_anthropic_finish(
                        finish_reason.as_deref(),
                        complete_tools,
                        text_empty,
                    );
                    match class {
                        AnthropicFinishClass::Error { message } => {
                            stream_error = Some(message);
                            break;
                        }
                        AnthropicFinishClass::Success { stop_reason } => {
                            pending_finish = Some(stop_reason.to_string());
                        }
                    }
                }
                Ok(CanonicalStreamEvent::RefusalDelta(text)) => {
                    if !text.is_empty() {
                        stream_error = Some(format!("refusal: {text}"));
                        break;
                    }
                }
                Ok(CanonicalStreamEvent::ReasoningDelta(_))
                | Ok(CanonicalStreamEvent::ReasoningSignatureDelta(_))
                | Ok(CanonicalStreamEvent::OutputAnnotations(_))
                | Ok(CanonicalStreamEvent::ResponseMetadata(_)) => {
                    // No thinking on translated wire
                }
                Err(e) => {
                    stream_error = Some(e.to_string());
                    break;
                }
            }
        }

        // Stream ended
        if stream_error.is_none() && !saw_finish {
            // Bare EOF without Finish → error
            stream_error = Some(
                "upstream stream ended without finish (truncated or incomplete)".into(),
            );
        }

        if let Some(msg) = stream_error {
            // Error path: best-effort content_block_stop; SSE error; no message_stop
            if let Some(idx) = open_text.take() {
                let stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": idx
                });
                yield Ok(Event::default().event("content_block_stop").data(stop.to_string()));
            }
            let err = serde_json::json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": msg
                }
            });
            yield Ok(Event::default().event("error").data(err.to_string()));
        } else if let Some(stop_reason) = pending_finish.take() {
            // Successful terminalization (stream ended cleanly after Finish)
            if let Some(idx) = open_text.take() {
                let stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": idx
                });
                yield Ok(Event::default().event("content_block_stop").data(stop.to_string()));
            }
            // Empty content success: emit empty text block if nothing emitted
            if next_index == 0 {
                let start = serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                });
                yield Ok(Event::default().event("content_block_start").data(start.to_string()));
                let stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": 0
                });
                yield Ok(Event::default().event("content_block_stop").data(stop.to_string()));
            }
            let mut usage_json = serde_json::json!({
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            });
            if usage.cache_read > 0 {
                usage_json["cache_read_input_tokens"] = serde_json::json!(usage.cache_read);
            }
            if usage.cache_creation > 0 {
                usage_json["cache_creation_input_tokens"] =
                    serde_json::json!(usage.cache_creation);
            }
            let delta = serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": null
                },
                "usage": usage_json
            });
            yield Ok(Event::default().event("message_delta").data(delta.to_string()));
            let stop = serde_json::json!({ "type": "message_stop" });
            yield Ok(Event::default().event("message_stop").data(stop.to_string()));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::CanonicalToolCall;
    use serde_json::json;

    fn basic_user(text: &str) -> Value {
        json!({
            "model": "grok-3",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": text}]
        })
    }

    #[test]
    fn text_and_sampling_map() {
        let body = json!({
            "model": "grok-3",
            "max_tokens": 256,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let canon = anthropic_to_canonical(&body, "grok").unwrap();
        assert_eq!(canon.model, "grok-3");
        assert_eq!(canon.max_tokens, Some(256));
        assert_eq!(canon.temperature, Some(0.5));
        assert_eq!(canon.top_p, Some(0.9));
        assert_eq!(canon.messages.len(), 1);
        assert_eq!(canon.messages[0].role, "user");
        assert_eq!(canon.messages[0].content.as_text(), "hi");
    }

    #[test]
    fn system_string_and_array_join() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "system": "sys",
            "messages": [{"role": "user", "content": "u"}]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        assert_eq!(c.messages[0].role, "system");
        assert_eq!(c.messages[0].content.as_text(), "sys");

        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "system": [{"type":"text","text":"a"},{"type":"text","text":"b"}],
            "messages": [{"role": "user", "content": "u"}]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        assert_eq!(c.messages[0].content.as_text(), "a\nb");
    }

    #[test]
    fn max_tokens_required() {
        let body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "u"}]
        });
        let err = anthropic_to_canonical(&body, "grok").unwrap_err();
        assert!(err.0.contains("max_tokens"), "{err}");
    }

    #[test]
    fn prefill_rejected() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "partial"}
            ]
        });
        let err = anthropic_to_canonical(&body, "grok").unwrap_err();
        assert!(err.0.contains("prefill"), "{err}");
    }

    #[test]
    fn mid_system_rejected() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "system", "content": "nope"},
                {"role": "user", "content": "u2"}
            ]
        });
        let err = anthropic_to_canonical(&body, "grok").unwrap_err();
        assert!(err.0.contains("system"), "{err}");
    }

    #[test]
    fn thinking_dropped_empty_strip() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "secret"},
                    {"type": "text", "text": "answer"}
                ]},
                {"role": "user", "content": "ok"}
            ]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        assert_eq!(c.messages[1].role, "assistant");
        assert_eq!(c.messages[1].content.as_text(), "answer");
    }

    #[test]
    fn thinking_drop_adjacency_fails() {
        // user → assistant(thinking only) → user  becomes user→user after drop
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "t"}]},
                {"role": "user", "content": "u2"}
            ]
        });
        let err = anthropic_to_canonical(&body, "grok").unwrap_err();
        assert!(err.0.contains("adjacent") || err.0.contains("same-role"), "{err}");
    }

    #[test]
    fn tools_and_specific_tool_choice() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "tools": [{
                "name": "get_weather",
                "description": "weather",
                "input_schema": {"type": "object"}
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "weather?"}]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        let tools = c.tools.unwrap();
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].parameters["properties"], json!({}));
        match c.tool_choice {
            Some(CanonicalToolChoice::Specific { name }) => assert_eq!(name, "get_weather"),
            other => panic!("expected Specific, got {other:?}"),
        }
    }

    #[test]
    fn hosted_tool_rejected() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "tools": [{"type": "computer_20241022", "name": "computer"}],
            "messages": [{"role": "user", "content": "u"}]
        });
        // missing input_schema
        let err = anthropic_to_canonical(&body, "grok").unwrap_err();
        assert!(err.0.contains("input_schema") || err.0.contains("function"), "{err}");
    }

    #[test]
    fn tool_history_fanout_and_is_error() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "use tools"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                    {"type": "tool_use", "id": "t2", "name": "b", "input": {"x": 1}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok1"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "fail", "is_error": true},
                    {"type": "text", "text": "thanks"}
                ]}
            ]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        // user, assistant, tool, tool, trailing user
        assert_eq!(c.messages.len(), 5);
        assert_eq!(c.messages[0].role, "user");
        assert_eq!(c.messages[1].role, "assistant");
        assert_eq!(c.messages[2].role, "tool");
        assert_eq!(c.messages[3].role, "tool");
        assert_eq!(c.messages[4].role, "user");
        assert_eq!(c.messages[4].content.as_text(), "thanks");

        match &c.messages[2].content {
            CanonicalContent::Blocks(b) => match &b[0] {
                CanonicalBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    assert_eq!(tool_use_id, "t1");
                    assert_eq!(content, "ok1");
                    assert!(!is_error);
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
        match &c.messages[3].content {
            CanonicalContent::Blocks(b) => match &b[0] {
                CanonicalBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    assert_eq!(tool_use_id, "t2");
                    assert_eq!(content, "fail");
                    assert!(*is_error);
                    assert_eq!(
                        encode_tool_result_content(content, *is_error),
                        "ERROR: fail"
                    );
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unpaired_tool_result_rejected() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "x", "content": "nope"}]
            }]
        });
        let err = anthropic_to_canonical(&body, "grok").unwrap_err();
        assert!(err.0.contains("unpaired"), "{err}");
    }

    #[test]
    fn missing_tool_result_rejected() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "a", "input": {}}
                ]},
                {"role": "user", "content": "forgot tool result"}
            ]
        });
        let err = anthropic_to_canonical(&body, "grok").unwrap_err();
        assert!(err.0.contains("missing tool_result") || err.0.contains("t1"), "{err}");
    }

    #[test]
    fn stop_sequences_grok_vs_codex() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "u"}]
        });
        let g = anthropic_to_canonical(&body, "grok").unwrap();
        assert_eq!(g.provider_extras.unwrap()["stop"][0], "END");

        let c = anthropic_to_canonical(&body, "codex").unwrap();
        assert!(c.provider_extras.is_none());
    }

    #[test]
    fn parallel_disable_maps_extra() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
            "messages": [{"role": "user", "content": "u"}]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        assert_eq!(c.provider_extras.unwrap()["parallel_tool_calls"], false);
    }

    #[test]
    fn image_base64_and_url() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abc"
                }},
                {"type": "image", "source": {
                    "type": "url",
                    "url": "https://example.com/i.png"
                }}
            ]}]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        match &c.messages[0].content {
            CanonicalContent::Blocks(blocks) => {
                assert!(matches!(blocks[0], CanonicalBlock::Text(_)));
                assert!(matches!(
                    &blocks[1],
                    CanonicalBlock::Image {
                        source: CanonicalImageSource::Base64 { media_type, data }
                    } if media_type == "image/png" && data == "abc"
                ));
                assert!(matches!(
                    &blocks[2],
                    CanonicalBlock::Image {
                        source: CanonicalImageSource::Url { url }
                    } if url == "https://example.com/i.png"
                ));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn document_block_rejected() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "url", "url": "x"}}
            ]}]
        });
        let err = anthropic_to_canonical(&body, "grok").unwrap_err();
        assert!(err.0.contains("document"), "{err}");
    }

    #[test]
    fn duplicate_top_level_keys_rejected() {
        let bytes = br#"{"model":"a","model":"b","max_tokens":1,"messages":[{"role":"user","content":"u"}]}"#;
        let err = parse_anthropic_object_no_dup_keys(bytes).unwrap_err();
        assert!(err.0.contains("duplicate"), "{err}");
    }

    #[test]
    fn encode_tool_result_content_matrix() {
        assert_eq!(encode_tool_result_content("", false), "");
        assert_eq!(encode_tool_result_content("ok", false), "ok");
        assert_eq!(encode_tool_result_content("", true), "error");
        assert_eq!(encode_tool_result_content("boom", true), "ERROR: boom");
    }

    #[test]
    fn tool_id_round_trip_loop() {
        // Response tool_use id → next request tool_result → same id in canonical
        let id = "call_exact_abc";
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": id, "name": "fn", "input": {"q": 1}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": id, "content": "result"}
                ]}
            ]
        });
        let c = anthropic_to_canonical(&body, "codex").unwrap();
        match &c.messages[1].content {
            CanonicalContent::Blocks(b) => match &b[0] {
                CanonicalBlock::ToolUse { id: got, .. } => assert_eq!(got, id),
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
        match &c.messages[2].content {
            CanonicalContent::Blocks(b) => match &b[0] {
                CanonicalBlock::ToolResult { tool_use_id, .. } => {
                    assert_eq!(tool_use_id, id);
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn thinking_config_maps_reasoning() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "u"}]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        let r = c.reasoning.unwrap();
        assert_eq!(r.budget_tokens, Some(1024));
    }

    #[test]
    fn metadata_user_id_session_only() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "metadata": {"user_id": "u-1", "other": "ignored"},
            "messages": [{"role": "user", "content": "u"}]
        });
        let c = anthropic_to_canonical(&body, "grok").unwrap();
        assert_eq!(c.metadata.get("user_id").map(String::as_str), Some("u-1"));
        assert!(!c.metadata.contains_key("other"));
    }

    // --- Response mapper tests ---

    #[test]
    fn nonstream_text_response() {
        let canon = CanonicalResponse {
            model: "grok-3".into(),
            content: "hello".into(),
            finish_reason: Some("stop".into()),
            usage: CanonicalUsage {
                input_tokens: 3,
                output_tokens: 1,
                ..Default::default()
            },
            id: Some("msg_1".into()),
            ..Default::default()
        };
        let v = canonical_to_anthropic(&canon, "grok-3").unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["stop_sequence"], Value::Null);
        assert_eq!(v["content"][0]["text"], "hello");
        assert_eq!(v["usage"]["input_tokens"], 3);
    }

    #[test]
    fn nonstream_tool_only() {
        let canon = CanonicalResponse {
            model: "m".into(),
            content: String::new(),
            tool_calls: vec![CanonicalToolCall {
                id: "toolu_1".into(),
                name: "fn".into(),
                arguments: r#"{"a":1}"#.into(),
            }],
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        };
        let v = canonical_to_anthropic(&canon, "m").unwrap();
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["content"][0]["type"], "tool_use");
        assert_eq!(v["content"][0]["id"], "toolu_1");
        assert_eq!(v["content"][0]["input"]["a"], 1);
    }

    #[test]
    fn nonstream_empty_args_become_object() {
        let canon = CanonicalResponse {
            model: "m".into(),
            tool_calls: vec![CanonicalToolCall {
                id: "t1".into(),
                name: "fn".into(),
                arguments: String::new(),
            }],
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        };
        let v = canonical_to_anthropic(&canon, "m").unwrap();
        assert_eq!(v["content"][0]["input"], json!({}));
    }

    #[test]
    fn nonstream_invalid_args_error() {
        let canon = CanonicalResponse {
            model: "m".into(),
            tool_calls: vec![CanonicalToolCall {
                id: "t1".into(),
                name: "fn".into(),
                arguments: "not-json".into(),
            }],
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        };
        let err = canonical_to_anthropic(&canon, "m").unwrap_err();
        assert!(err.0.contains("JSON") || err.0.contains("arguments"), "{err}");
    }

    #[test]
    fn nonstream_tool_calls_finish_zero_tools_error() {
        let canon = CanonicalResponse {
            model: "m".into(),
            content: "x".into(),
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        };
        let err = canonical_to_anthropic(&canon, "m").unwrap_err();
        assert!(err.0.contains("tool"), "{err}");
    }

    #[test]
    fn finish_classification_priority() {
        assert!(matches!(
            classify_anthropic_finish(Some("content_filter"), 1, false),
            AnthropicFinishClass::Error { .. }
        ));
        assert_eq!(
            classify_anthropic_finish(Some("length"), 0, true),
            AnthropicFinishClass::Success {
                stop_reason: "max_tokens"
            }
        );
        assert_eq!(
            classify_anthropic_finish(Some("stop"), 2, false),
            AnthropicFinishClass::Success {
                stop_reason: "tool_use"
            }
        );
        assert_eq!(
            classify_anthropic_finish(Some("stop"), 0, false),
            AnthropicFinishClass::Success {
                stop_reason: "end_turn"
            }
        );
    }

    #[test]
    fn basic_user_helper_smoke() {
        let _ = basic_user("hi");
    }

    #[tokio::test]
    async fn sse_text_then_stop() {
        use futures_util::stream;
        use omni_core::ProviderError;

        let events = vec![
            Ok::<_, ProviderError>(CanonicalStreamEvent::TextDelta("hi".into())),
            Ok(CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 1,
                output_tokens: 1,
                ..Default::default()
            })),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into()),
            }),
        ];
        let stream: CanonicalStream = Box::pin(stream::iter(events));
        let mut s = std::pin::pin!(anthropic_sse_body(stream, "m".into(), "msg_x".into()));
        let mut frames = Vec::new();
        while let Some(Ok(ev)) = s.next().await {
            frames.push(format!("{ev:?}"));
        }
        let joined = frames.join("\n");
        assert!(joined.contains("message_start") || frames.len() >= 3, "{joined}");
        assert!(
            joined.contains("message_stop") || joined.contains("message_delta"),
            "{joined}"
        );
    }

    #[tokio::test]
    async fn sse_eof_without_finish_is_error() {
        use futures_util::stream;
        use omni_core::ProviderError;

        let events: Vec<Result<CanonicalStreamEvent, ProviderError>> = vec![Ok(
            CanonicalStreamEvent::TextDelta("hi".into()),
        )];
        let stream: CanonicalStream = Box::pin(stream::iter(events));
        let mut s = std::pin::pin!(anthropic_sse_body(stream, "m".into(), "msg_y".into()));
        let mut saw_error = false;
        let mut saw_stop = false;
        while let Some(Ok(ev)) = s.next().await {
            let dbg = format!("{ev:?}");
            if dbg.contains("error") || dbg.contains("api_error") {
                saw_error = true;
            }
            if dbg.contains("message_stop") {
                saw_stop = true;
            }
        }
        assert!(saw_error, "expected SSE error on bare EOF");
        assert!(!saw_stop, "must not emit message_stop on error path");
    }
}
