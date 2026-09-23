//! Shared OpenAI-compatible HTTP surface: request/response types, canonical
//! conversion, and SSE streaming framing.
//!
//! The `omni` server speaks this OpenAI-compatible wire shape and delegates to
//! provider crates through `LlmProvider`. This module is the single source of
//! truth for request/response translation and SSE framing.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use omni_core::{
    CanonicalBlock, CanonicalContent, CanonicalImageSource, CanonicalMessage, CanonicalReasoning,
    CanonicalRequest, CanonicalResponse, CanonicalStream, CanonicalStreamEvent, CanonicalTool,
    CanonicalToolCall, CanonicalToolChoice,
};

use crate::cache::{parse_openai_cache_intent, parse_prompt_cache_breakpoint};
use crate::canonical_mapping::{provider_metadata_json, usage_detail_json};

/// OpenAI-compatible chat completion request (text messages, tools, and core
/// sampling). Unknown fields are captured in `extras` so a client request never
/// fails to deserialize on an unrecognized key.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Tool (function) definitions the model may call. OpenAI's nested shape:
    /// `{type:"function", function:{name, description, parameters}}`.
    #[serde(default)]
    pub tools: Option<Vec<ChatTool>>,
    /// How the model should choose among `tools`: a string mode
    /// (`"auto"`/`"required"`/`"none"`) or a forced function selection.
    #[serde(default)]
    pub tool_choice: Option<ChatToolChoice>,
    #[serde(flatten)]
    pub extras: serde_json::Value,
}

/// One message in a chat request. Beyond `role`/`content`, an *assistant* turn
/// can carry `tool_calls` (the model's prior tool requests) and a *tool* turn
/// carries the result keyed by `tool_call_id` - both required so multi-turn
/// tool conversations can be fed back through the proxy.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<ChatMessageContent>,
    /// Present on assistant messages that called tools (OpenAI echoes these
    /// back into the next request's history).
    #[serde(default)]
    pub tool_calls: Option<Vec<ChatToolCallReq>>,
    /// Present on `role:"tool"` messages: the id of the tool call this result
    /// answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

impl From<&str> for ChatMessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for ChatMessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<ChatImageUrl>,
    /// Official Chat Completions breakpoint on a supported part.
    #[serde(default)]
    pub prompt_cache_breakpoint: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatImageUrl {
    pub url: String,
}

/// A tool definition in OpenAI's nested form. Only `function` tools are
/// supported (the canonical layer and both backends model function tools).
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatToolFunction,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

/// `tool_choice`: a bare mode string, a forced function selection, or an
/// allowed subset of the declared tools.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChatToolChoice {
    /// `"auto" | "required" | "none"`
    Mode(String),
    /// `{type:"function", function:{name}}`
    Function {
        #[serde(rename = "type")]
        kind: String,
        function: ChatToolChoiceFunction,
    },
    /// `{type:"allowed_tools",allowed_tools:{mode,tools:[...]}}`
    Allowed {
        #[serde(rename = "type")]
        kind: String,
        allowed_tools: ChatAllowedTools,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatAllowedTools {
    pub mode: String,
    pub tools: Vec<ChatAllowedTool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatAllowedTool {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub function: Option<ChatToolChoiceFunction>,
}

/// Minimal OpenAI-compatible chat completion response (non-streaming).
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub function: ChatFunctionCall,
}

/// Request-side counterpart of [`ChatToolCall`]: an assistant tool call echoed
/// back by the client in a follow-up turn. Separate from the response type
/// because it deserializes a client-supplied `type` string.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatToolCallReq {
    pub id: String,
    #[serde(rename = "type", default = "default_function_kind")]
    pub kind: String,
    pub function: ChatFunctionCallReq,
}

fn default_function_kind() -> String {
    "function".into()
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatFunctionCallReq {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Default)]
pub struct ChatUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<Value>,
}

/// Top-level chat fields Omni consumes (or ignores) as gateway/transport
/// metadata rather than forwarding as provider extras.
///
/// - `user`: session attribution for the gateway, not an upstream model field.
/// - `stream_options`: OpenAI chat streaming transport (`include_usage`); the
///   Responses/Codex/Claude paths do not accept this key.
const GATEWAY_ONLY_EXTRAS: &[&str] = &["user", "stream_options"];

/// Max length for a free-string `reasoning_effort` at the edge (issue #20).
/// Soft UI vocab is discovery-only; the edge does not close the name set.
pub const MAX_REASONING_EFFORT_LEN: usize = 32;

/// Top-level request fields that Omni consumes as gateway metadata rather than
/// forwarding as provider extras.
pub fn gateway_only_extra_keys() -> &'static [&'static str] {
    GATEWAY_ONLY_EXTRAS
}

/// Lexical hygiene for free-string reasoning effort at chat and Responses edges.
///
/// Accepts any non-empty, bounded, safe-charset name (`A-Z a-z 0-9 _ -`),
/// including `xhigh` and other unknown values. No global allowlist: adapters
/// map or fail loud. Catalog advertised lists are discovery-only and never
/// gate this path (issue #20).
pub fn validate_reasoning_effort_lexical(effort: &str) -> Result<(), String> {
    if effort.is_empty() {
        return Err("Invalid reasoning_effort: empty string".into());
    }
    if effort.len() > MAX_REASONING_EFFORT_LEN {
        return Err(format!(
            "Invalid reasoning_effort: length {} exceeds max {MAX_REASONING_EFFORT_LEN}",
            effort.len()
        ));
    }
    if !effort
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "Invalid reasoning_effort: '{effort}' has unsafe characters (allowed: A-Z a-z 0-9 _ -)"
        ));
    }
    Ok(())
}

/// Parse a chat effort string into a value for `CanonicalReasoning.effort`.
/// Explicit `"none"` is preserved so providers that support disable (Codex)
/// can emit it; Claude/Grok adapters treat `"none"` as omit.
fn parse_chat_effort_value(effort: &str) -> Result<String, String> {
    validate_reasoning_effort_lexical(effort)?;
    Ok(effort.to_string())
}

/// Read an optional string effort from a JSON value. JSON `null` is absent.
fn effort_string_from_value(value: &Value, field: &str) -> Result<Option<String>, String> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        _ => Err(format!("{field} must be a string")),
    }
}

/// Lift chat reasoning fields out of flattened extras into
/// [`CanonicalReasoning`], and return the remaining provider extras.
///
/// OpenAI chat clients send top-level `reasoning_effort`. Nested
/// `reasoning.effort` (Responses-style) is also accepted. Lifted keys are
/// stripped from provider extras so edge allowlists never see them as
/// unsupported passthrough fields. Top-level `reasoning_effort` wins when both
/// are present. JSON `null` is treated as absent. Explicit `"none"` is kept as
/// `CanonicalReasoning.effort` so providers can disable (Codex) or omit
/// (Claude/Grok). Nested sibling keys under `reasoning` (e.g. `summary`) remain
/// in extras for allowlist handling (fail loud if unsupported).
fn chat_reasoning_and_extras(
    extras: &Value,
) -> Result<(Option<CanonicalReasoning>, Option<Value>), String> {
    let Some(obj) = extras.as_object() else {
        return Ok((None, None));
    };

    // Top-level OpenAI chat shape first so a *present string* wins without
    // nested validation blocking it. JSON null is true absence (does not
    // suppress a nested reasoning.effort). Explicit `"none"` is preserved.
    let mut effort: Option<String> = None;
    if let Some(effort_val) = obj.get("reasoning_effort") {
        if let Some(raw) = effort_string_from_value(effort_val, "reasoning_effort")? {
            effort = Some(parse_chat_effort_value(&raw)?);
        }
        // null: leave effort as None (absent); key is still stripped below.
    }

    // What (if anything) to leave under provider_extras["reasoning"].
    let mut reasoning_extra: Option<Value> = None;

    if let Some(reasoning_val) = obj.get("reasoning") {
        match reasoning_val {
            Value::Null => {
                // Strip explicit null; do not leave it as an extra.
                reasoning_extra = None;
            }
            Value::Object(nested) if nested.is_empty() => {
                // Empty object is absent, not an unsupported extra.
                reasoning_extra = None;
            }
            Value::Object(nested) => {
                if nested.contains_key("effort") {
                    // Lift effort only when top-level did not already set a value
                    // (including explicit "none", which wins over nested).
                    if effort.is_none() {
                        let effort_val = &nested["effort"];
                        if let Some(raw) = effort_string_from_value(effort_val, "reasoning.effort")?
                        {
                            effort = Some(parse_chat_effort_value(&raw)?);
                        }
                        // nested effort null: absent; strip with siblings below.
                    }
                    // Preserve non-effort siblings for allowlist (fail loud).
                    let siblings: Map<_, _> = nested
                        .iter()
                        .filter(|(k, _)| k.as_str() != "effort")
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    if !siblings.is_empty() {
                        reasoning_extra = Some(Value::Object(siblings));
                    }
                } else {
                    // No effort key: leave whole object as extra (fail loud).
                    reasoning_extra = Some(reasoning_val.clone());
                }
            }
            other => {
                // Non-object: leave as extra so allowlist fails loud by name.
                reasoning_extra = Some(other.clone());
            }
        }
    }

    let reasoning = effort.map(|e| CanonicalReasoning {
        effort: Some(e),
        budget_tokens: None,
    });

    let mut filtered = Map::new();
    for (key, value) in obj {
        let k = key.as_str();
        if gateway_only_extra_keys().contains(&k)
            || k == "reasoning_effort"
            || k == "reasoning"
            || crate::cache::OPENAI_CACHE_EXTRA_KEYS.contains(&k)
        {
            continue;
        }
        filtered.insert(key.clone(), value.clone());
    }
    if let Some(rem) = reasoning_extra {
        filtered.insert("reasoning".into(), rem);
    }

    let provider_extras = if filtered.is_empty() {
        None
    } else {
        Some(Value::Object(filtered))
    };

    Ok((reasoning, provider_extras))
}

/// Convert an OpenAI request into a `CanonicalRequest`. The `model` field is the
/// caller-supplied value; the `omni` router overwrites it with the
/// prefix-stripped model before delegating when needed.
///
/// Fallible because a malformed tool surface (non-function tool, unknown
/// `tool_choice` mode, a tool-result message missing its `tool_call_id`) is
/// rejected by name as a 400 rather than silently dropped - the same contract
/// the Responses protocol enforces.
pub fn to_canonical(req: &ChatCompletionRequest) -> Result<CanonicalRequest, String> {
    to_canonical_with_headers(req, None)
}

/// Chat Completions inbound parse, including optional `x-grok-conv-id`.
///
/// Header only becomes routing identity. Body `prompt_cache_key` wins when both
/// are present. The header is never forwarded alongside the body key.
pub fn to_canonical_with_headers(
    req: &ChatCompletionRequest,
    grok_conv_id: Option<&str>,
) -> Result<CanonicalRequest, String> {
    let messages: Vec<CanonicalMessage> = req
        .messages
        .iter()
        .map(chat_message_to_canonical)
        .collect::<Result<_, _>>()?;

    // Nested function tools -> canonical tools (non-function tools rejected).
    let mut tools = match req.tools.as_ref() {
        Some(ts) if !ts.is_empty() => {
            let mut out = Vec::with_capacity(ts.len());
            for t in ts {
                if t.kind != "function" {
                    return Err(format!("unsupported tool type: {}", t.kind));
                }
                let parameters = t
                    .function
                    .parameters
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({}));
                omni_core::validate_tool_schema(
                    &parameters,
                    t.function.strict.unwrap_or(false),
                    false,
                )?;
                out.push(CanonicalTool {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters,
                    strict: t.function.strict.unwrap_or(false),
                    cache: None,
                });
            }
            Some(out)
        }
        _ => None,
    };

    // tool_choice: standard modes and one forced function map directly. An
    // allowed subset is enforced by filtering tools before provider dispatch.
    let tool_choice = match req.tool_choice.as_ref() {
        Some(ChatToolChoice::Mode(mode)) => match mode.as_str() {
            "auto" => Some(CanonicalToolChoice::Auto),
            "required" => Some(CanonicalToolChoice::Required),
            "none" => Some(CanonicalToolChoice::None),
            other => return Err(format!("unsupported tool_choice mode: {other}")),
        },
        Some(ChatToolChoice::Function { kind, function }) => {
            if kind != "function" {
                return Err(format!("unsupported tool_choice type: {kind}"));
            }
            Some(CanonicalToolChoice::Specific {
                name: function.name.clone(),
            })
        }
        Some(ChatToolChoice::Allowed {
            kind,
            allowed_tools,
        }) => {
            if kind != "allowed_tools" {
                return Err(format!("unsupported tool_choice type: {kind}"));
            }
            let mut names = Vec::with_capacity(allowed_tools.tools.len());
            for tool in &allowed_tools.tools {
                if tool.kind != "function" {
                    return Err(format!("unsupported allowed tool type: {}", tool.kind));
                }
                let name = tool
                    .function
                    .as_ref()
                    .map(|function| function.name.as_str())
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| "allowed function tool requires a non-empty name".to_string())?;
                names.push(name.to_string());
            }
            restrict_tools_to_allowed(&mut tools, &allowed_tools.mode, &names)?
        }
        None => None,
    };

    let (reasoning, provider_extras) = chat_reasoning_and_extras(&req.extras)?;
    let cache = parse_openai_cache_intent(&req.extras, grok_conv_id)?;

    Ok(CanonicalRequest {
        model: req.model.clone(),
        messages,
        tools,
        tool_choice,
        // OpenAI's max_completion_tokens supersedes the legacy max_tokens.
        max_tokens: req.max_completion_tokens.or(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        reasoning,
        metadata: Default::default(),
        cache,
        provider_extras,
    })
}

fn restrict_tools_to_allowed(
    tools: &mut Option<Vec<CanonicalTool>>,
    mode: &str,
    allowed_names: &[String],
) -> Result<Option<CanonicalToolChoice>, String> {
    let choice = match mode {
        "auto" => CanonicalToolChoice::Auto,
        "required" => CanonicalToolChoice::Required,
        other => return Err(format!("unsupported allowed_tools mode: {other}")),
    };

    for name in allowed_names {
        if !tools
            .as_ref()
            .is_some_and(|declared| declared.iter().any(|tool| tool.name == name.as_str()))
        {
            return Err(format!("allowed tool is not declared: {name}"));
        }
    }

    if allowed_names.is_empty() {
        if mode == "required" {
            return Err("allowed_tools mode required needs at least one tool".into());
        }
        *tools = None;
        return Ok(None);
    }

    if let Some(declared) = tools.as_mut() {
        declared.retain(|tool| allowed_names.iter().any(|name| name == &tool.name));
    }
    Ok(Some(choice))
}

/// Convert one chat message into a canonical message. Plain text and
/// tool-bearing turns both map here; tool turns become `Blocks` so the call/
/// result linkage survives into the canonical layer.
fn chat_message_to_canonical(m: &ChatMessage) -> Result<CanonicalMessage, String> {
    // tool_calls are only valid on the assistant role (OpenAI contract).
    // Checked up front so a non-assistant message carrying tool_calls is
    // rejected by name rather than having them silently dropped (e.g. a
    // role:"tool" message must not also carry tool_calls).
    let has_tool_calls = m.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());
    if has_tool_calls && m.role != "assistant" {
        return Err(format!(
            "tool_calls are only valid on an assistant message, not role \"{}\"",
            m.role
        ));
    }

    // A `role:"tool"` message carries a tool result keyed by tool_call_id.
    if m.role == "tool" {
        let id = m
            .tool_call_id
            .clone()
            .ok_or_else(|| "tool message missing tool_call_id".to_string())?;
        let cache = chat_content_cache_mark(&m.content)?;
        return Ok(CanonicalMessage {
            role: "tool".into(),
            content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                tool_use_id: id,
                content: chat_content_text(&m.content)?,
                is_error: false,
                cache,
            }]),
        });
    }

    // An assistant message may interleave text with tool calls.
    if let Some(tool_calls) = m.tool_calls.as_ref().filter(|tc| !tc.is_empty()) {
        let mut blocks: Vec<CanonicalBlock> = Vec::new();
        match chat_content_to_canonical(&m.content)? {
            CanonicalContent::Text(text) if !text.is_empty() => {
                blocks.push(CanonicalBlock::text(text));
            }
            CanonicalContent::Text(_) => {}
            CanonicalContent::Blocks(content_blocks) => blocks.extend(content_blocks),
        }
        for tc in tool_calls {
            if tc.kind != "function" {
                return Err(format!("unsupported tool_call type: {}", tc.kind));
            }
            blocks.push(CanonicalBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
                cache: None,
            });
        }
        return Ok(CanonicalMessage {
            role: m.role.clone(),
            content: CanonicalContent::Blocks(blocks),
        });
    }

    Ok(CanonicalMessage {
        role: m.role.clone(),
        content: chat_content_to_canonical(&m.content)?,
    })
}

fn chat_content_text(content: &Option<ChatMessageContent>) -> Result<String, String> {
    match content {
        Some(ChatMessageContent::Text(text)) => Ok(text.clone()),
        Some(ChatMessageContent::Parts(parts)) => {
            let mut fragments = Vec::new();
            for part in parts {
                match part.kind.as_str() {
                    "text" => fragments.push(part.text.clone().unwrap_or_default()),
                    "image_url" => {
                        return Err("image_url content is not supported on tool messages".into());
                    }
                    other => return Err(format!("unsupported content part type: {other}")),
                }
            }
            Ok(fragments.join("\n"))
        }
        None => Ok(String::new()),
    }
}

fn chat_content_cache_mark(
    content: &Option<ChatMessageContent>,
) -> Result<Option<omni_core::CanonicalCacheMark>, String> {
    let Some(ChatMessageContent::Parts(parts)) = content else {
        return Ok(None);
    };
    let mut found = None;
    for part in parts {
        if let Some(mark) = parse_prompt_cache_breakpoint(part.prompt_cache_breakpoint.as_ref())? {
            found = Some(mark);
        }
    }
    Ok(found)
}

fn chat_content_to_canonical(
    content: &Option<ChatMessageContent>,
) -> Result<CanonicalContent, String> {
    match content {
        Some(ChatMessageContent::Text(text)) => Ok(CanonicalContent::Text(text.clone())),
        Some(ChatMessageContent::Parts(parts)) => {
            let mut blocks = Vec::with_capacity(parts.len());
            let mut has_image = false;
            let mut has_mark = false;
            for part in parts {
                let cache = parse_prompt_cache_breakpoint(part.prompt_cache_breakpoint.as_ref())?;
                if cache.is_some() {
                    has_mark = true;
                }
                match part.kind.as_str() {
                    "text" => blocks.push(CanonicalBlock::Text {
                        text: part.text.clone().unwrap_or_default(),
                        cache,
                    }),
                    "image_url" => {
                        let url = part
                            .image_url
                            .as_ref()
                            .ok_or_else(|| "image_url content part missing image_url".to_string())?
                            .url
                            .as_str();
                        blocks.push(CanonicalBlock::Image {
                            source: CanonicalImageSource::from_image_url(url)?,
                            cache,
                        });
                        has_image = true;
                    }
                    other => return Err(format!("unsupported content part type: {other}")),
                }
            }
            if has_image || has_mark {
                Ok(CanonicalContent::Blocks(blocks))
            } else {
                Ok(CanonicalContent::Text(
                    blocks
                        .into_iter()
                        .filter_map(|block| match block {
                            CanonicalBlock::Text { text, .. } => Some(text),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                ))
            }
        }
        None => Ok(CanonicalContent::Text(String::new())),
    }
}

/// Convert a `CanonicalResponse` into the OpenAI non-streaming response shape.
/// `requested_model` is echoed back verbatim (with the client's prefix, if any)
/// for client UX.
pub fn from_canonical(
    canon: CanonicalResponse,
    requested_model: String,
    chat_id: String,
    created: u64,
) -> ChatCompletionResponse {
    let metadata = canon.metadata.clone();
    let provider_metadata = provider_metadata_json(&canon, true);
    let tool_calls: Vec<ChatToolCall> = canon
        .tool_calls
        .into_iter()
        .map(|tc: CanonicalToolCall| ChatToolCall {
            id: tc.id,
            type_: "function",
            function: ChatFunctionCall {
                name: tc.name,
                arguments: tc.arguments,
            },
        })
        .collect();

    let has_tools = !tool_calls.is_empty();
    let content = if canon.content.is_empty() && has_tools {
        None
    } else {
        Some(canon.content)
    };

    let finish = canon.finish_reason.or_else(|| {
        if has_tools {
            Some("tool_calls".to_string())
        } else {
            Some("stop".to_string())
        }
    });

    let total = canon.usage.input_tokens + canon.usage.output_tokens;

    ChatCompletionResponse {
        id: chat_id,
        object: "chat.completion",
        created,
        model: requested_model,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content,
                refusal: canon.refusal,
                tool_calls,
            },
            finish_reason: finish,
        }],
        usage: chat_usage_from_canonical(&canon.usage, total),
        system_fingerprint: metadata
            .as_ref()
            .and_then(|metadata| metadata.system_fingerprint.clone()),
        service_tier: metadata
            .as_ref()
            .and_then(|metadata| metadata.service_tier.clone()),
        provider_metadata,
    }
}

fn chat_usage_from_canonical(usage: &omni_core::CanonicalUsage, total: u64) -> ChatUsage {
    let has_split_audio = usage.input_audio_tokens != 0 || usage.output_audio_tokens != 0;
    let prompt_audio_tokens = if has_split_audio {
        usage.input_audio_tokens
    } else {
        usage.audio_tokens
    };
    let prompt_details = usage_detail_json(&[
        ("cached_tokens", usage.cache_read),
        ("audio_tokens", prompt_audio_tokens),
        ("image_tokens", usage.image_tokens),
    ]);
    let completion_details = usage_detail_json(&[
        ("reasoning_tokens", usage.reasoning_tokens),
        ("audio_tokens", usage.output_audio_tokens),
        (
            "accepted_prediction_tokens",
            usage.accepted_prediction_tokens,
        ),
        (
            "rejected_prediction_tokens",
            usage.rejected_prediction_tokens,
        ),
    ]);
    ChatUsage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: total,
        prompt_tokens_details: prompt_details,
        completion_tokens_details: completion_details,
    }
}

/// Seconds since the Unix epoch, for the `created` field. Falls back to 0 on a
/// clock before the epoch (cannot happen in practice).
pub fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Build an OpenAI `chat.completion.chunk` JSON value carrying a content delta.
fn chunk_content(chat_id: &str, created: u64, model: &str, content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "content": content },
            "finish_reason": serde_json::Value::Null,
        }],
    })
}

/// Build a `chat.completion.chunk` carrying a tool-call delta fragment.
fn chunk_tool_call(
    chat_id: &str,
    created: u64,
    model: &str,
    index: u32,
    id: Option<&str>,
    name: Option<&str>,
    arguments: &str,
) -> serde_json::Value {
    let mut function = serde_json::Map::new();
    if let Some(n) = name {
        function.insert("name".into(), serde_json::Value::String(n.to_string()));
    }
    function.insert(
        "arguments".into(),
        serde_json::Value::String(arguments.to_string()),
    );
    let mut tool_call = serde_json::Map::new();
    tool_call.insert("index".into(), serde_json::json!(index));
    if let Some(i) = id {
        tool_call.insert("id".into(), serde_json::Value::String(i.to_string()));
        tool_call.insert("type".into(), serde_json::Value::String("function".into()));
    }
    tool_call.insert("function".into(), serde_json::Value::Object(function));
    serde_json::json!({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "tool_calls": [tool_call] },
            "finish_reason": serde_json::Value::Null,
        }],
    })
}

/// OpenAI Chat Completions finish_reason values. Anything else is illegal on
/// the wire (`error` is not in this set).
pub fn is_allowed_chat_finish_reason(reason: &str) -> bool {
    matches!(
        reason,
        "stop" | "length" | "tool_calls" | "content_filter" | "function_call"
    )
}

fn is_chat_error_finish_reason(reason: &str) -> bool {
    reason == "error" || reason.starts_with("error:")
}

/// Client-facing Chat mid-stream error. Fixed text: upstream decode payloads
/// are not fully redacted, so they stay in server logs only.
const CHAT_STREAM_ERROR_DATA: &str =
    r#"{"error":{"message":"upstream stream error","type":"server_error","code":null}}"#;

fn chat_stream_error_event() -> Event {
    Event::default().event("error").data(CHAT_STREAM_ERROR_DATA)
}

/// Whether `Content-Type` is the SSE media type.
///
/// A missing header is not SSE. Parameters after `;` are ignored. The compare
/// is the media type, not a raw prefix (`text/event-streamfoo` is not SSE).
pub fn is_sse_content_type(header: Option<&str>) -> bool {
    let Some(header) = header else {
        return false;
    };
    let media = header.split(';').next().unwrap_or("").trim();
    media.eq_ignore_ascii_case("text/event-stream")
}

/// Build the terminal `chat.completion.chunk` carrying the finish reason.
fn chunk_finish(chat_id: &str, created: u64, model: &str, reason: &str) -> serde_json::Value {
    serde_json::json!({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": reason,
        }],
    })
}

/// Frame a canonical event stream as an OpenAI-compatible SSE response.
///
/// Each [`CanonicalStreamEvent`] becomes one `data: {chunk}` SSE event:
/// - `TextDelta` -> a content-delta chunk
/// - `ToolCallDelta` -> a tool-call-delta chunk
/// - `Finish` -> a finish-reason chunk when the reason is in the OpenAI set
///   (default "stop" when none). An `error` / `error:` reason is a named
///   `event: error`, not a chunk. Any other label is coerced to `stop`.
/// - `Usage` is not emitted as a chunk (OpenAI streams omit usage by default).
///
/// The stream is always terminated by a literal `data: [DONE]` event, matching
/// the OpenAI streaming protocol. A stream `Err`, or an end with no `Finish`
/// and no `Err`, is a named `event: error` with a fixed message, then `[DONE]`.
/// Polling stops after the first `Finish` or `Err`.
pub fn sse_from_canonical_stream(
    stream: CanonicalStream,
    requested_model: String,
    chat_id: String,
    created: u64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Keep the caller's request span entered while this body is polled, so the
    // mid-stream error log below (and any other log emitted here) retains the
    // request's correlation fields. The inner `stream` may already be span-aware,
    // but this generator adds its OWN logs outside that stream's poll, so it must
    // be spanned too.
    let body = async_stream_chunks(stream, requested_model, chat_id, created);
    Sse::new(crate::span_stream::SpannedStream::current(Box::pin(body)))
}

fn async_stream_chunks(
    mut stream: CanonicalStream,
    requested_model: String,
    chat_id: String,
    created: u64,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let mut finished = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(CanonicalStreamEvent::TextDelta(text)) => {
                    let v = chunk_content(&chat_id, created, &requested_model, &text);
                    yield Ok(Event::default().data(v.to_string()));
                }
                Ok(CanonicalStreamEvent::RefusalDelta(text)) => {
                    let v = chunk_content(&chat_id, created, &requested_model, &text);
                    yield Ok(Event::default().data(v.to_string()));
                }
                Ok(CanonicalStreamEvent::ReasoningDelta(_)) |
                Ok(CanonicalStreamEvent::ReasoningSignatureDelta(_)) |
                Ok(CanonicalStreamEvent::OutputAnnotations(_)) => {
                    // Chat Completions has no standard reasoning-delta field.
                }
                Ok(CanonicalStreamEvent::ToolCallDelta { index, id, name, arguments_delta }) => {
                    let v = chunk_tool_call(
                        &chat_id, created, &requested_model,
                        index, id.as_deref(), name.as_deref(), &arguments_delta,
                    );
                    yield Ok(Event::default().data(v.to_string()));
                }
                Ok(CanonicalStreamEvent::Usage(_)) => {
                    // OpenAI streams omit usage unless stream_options.include_usage;
                    // not requested at this layer, so usage events are dropped.
                }
                Ok(CanonicalStreamEvent::ResponseMetadata(_)) => {
                    // Chat Completions has no response-id metadata event.
                }
                Ok(CanonicalStreamEvent::Finish { finish_reason }) => {
                    finished = true;
                    match finish_reason.as_deref() {
                        Some(reason) if is_chat_error_finish_reason(reason) => {
                            tracing::warn!(
                                finish_reason = reason,
                                "chat stream error finish"
                            );
                            yield Ok(chat_stream_error_event());
                        }
                        Some(reason) if is_allowed_chat_finish_reason(reason) => {
                            let v = chunk_finish(&chat_id, created, &requested_model, reason);
                            yield Ok(Event::default().data(v.to_string()));
                        }
                        Some(reason) => {
                            tracing::warn!(
                                finish_reason = reason,
                                "coercing unknown chat finish_reason to stop"
                            );
                            let v = chunk_finish(&chat_id, created, &requested_model, "stop");
                            yield Ok(Event::default().data(v.to_string()));
                        }
                        None => {
                            let v = chunk_finish(&chat_id, created, &requested_model, "stop");
                            yield Ok(Event::default().data(v.to_string()));
                        }
                    }
                    break;
                }
                Err(e) => {
                    finished = true;
                    tracing::warn!(error = %e, "canonical stream error mid-flight");
                    yield Ok(chat_stream_error_event());
                    break;
                }
            }
        }
        if !finished {
            // No Finish and no Err. Do not synthesize a successful stop.
            tracing::warn!("chat stream ended without Finish");
            yield Ok(chat_stream_error_event());
        }
        // OpenAI streaming sentinel.
        yield Ok(Event::default().data("[DONE]"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::CanonicalUsage;

    fn sample_oai_req() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: Some(10),
            max_completion_tokens: None,
            temperature: Some(0.5),
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        }
    }

    #[test]
    fn to_canonical_maps_text_and_sampling() {
        // WHY: the canonical request is the contract every provider consumes;
        // dropping a message or a sampling field silently changes behavior.
        let canon = to_canonical(&sample_oai_req()).unwrap();
        assert_eq!(canon.messages.len(), 1);
        match &canon.messages[0].content {
            CanonicalContent::Text(t) => assert_eq!(t, "hi"),
            CanonicalContent::Blocks(_) => panic!("unexpected blocks content"),
        }
        assert_eq!(canon.max_tokens, Some(10));
        assert_eq!(canon.temperature, Some(0.5));
    }

    #[test]
    fn max_completion_tokens_supersedes_max_tokens() {
        // WHY: OpenAI deprecated max_tokens in favor of max_completion_tokens;
        // when both are present the newer field must win or clients that send
        // both get the wrong limit.
        let mut req = sample_oai_req();
        req.max_tokens = Some(10);
        req.max_completion_tokens = Some(99);
        assert_eq!(to_canonical(&req).unwrap().max_tokens, Some(99));
    }

    #[test]
    fn to_canonical_preserves_provider_extras_but_not_gateway_user() {
        // WHY: OpenAI-compatible clients use top-level extension fields for
        // provider features, but `user` is gateway/session metadata in Omni and
        // must not be treated as a provider passthrough field.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "response_format":{"type":"json_object"},"user":"session-user"}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).expect("chat request should convert");
        let extras = canon
            .provider_extras
            .expect("provider extras should be preserved");
        assert_eq!(extras["response_format"]["type"], "json_object");
        assert!(extras.get("user").is_none());
    }

    #[test]
    fn to_canonical_lifts_prompt_cache_key_out_of_extras() {
        // WHY: prompt_cache_key is a cache-routing identity, not a provider extra.
        // Leaving it in extras 400s every current extras allowlist (Claude has
        // none; Grok/Codex lists omit it). It must become CanonicalRequest.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "prompt_cache_key":"sess-1","response_format":{"type":"json_object"}}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).expect("chat request should convert");
        assert_eq!(canon.cache_routing_identity(), Some("sess-1"));
        let extras = canon
            .provider_extras
            .expect("unrelated extras should remain");
        assert_eq!(extras["response_format"]["type"], "json_object");
        assert!(
            extras.get("prompt_cache_key").is_none(),
            "prompt_cache_key must not become a provider extra: {extras}"
        );
    }

    #[test]
    fn to_canonical_rejects_non_string_prompt_cache_key() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "prompt_cache_key":{"id":"sess-1"}}"#,
        )
        .unwrap();
        let err = to_canonical(&req).expect_err("non-string prompt_cache_key must 400");
        assert!(
            err.contains("prompt_cache_key must be a string"),
            "error must name the field: {err}"
        );
    }

    #[test]
    fn to_canonical_empty_prompt_cache_key_is_absent() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "prompt_cache_key":"  "}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).expect("whitespace key is absent, not an error");
        assert!(canon.cache.is_none());
    }

    #[test]
    fn chat_header_only_becomes_routing_identity() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .unwrap();
        let canon = to_canonical_with_headers(&req, Some("hdr-1")).unwrap();
        assert_eq!(canon.cache_routing_identity(), Some("hdr-1"));
        assert!(canon.provider_extras.is_none());
    }

    #[test]
    fn chat_body_key_wins_over_header() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "prompt_cache_key":"body-1"}"#,
        )
        .unwrap();
        let canon = to_canonical_with_headers(&req, Some("hdr-1")).unwrap();
        assert_eq!(canon.cache_routing_identity(), Some("body-1"));
    }

    #[test]
    fn chat_user_and_safety_identifier_are_not_cache_identity() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "user":"alice","safety_identifier":"sid-1"}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        assert!(canon.cache.is_none());
        assert_ne!(canon.cache_routing_identity(), Some("alice"));
        assert_ne!(canon.cache_routing_identity(), Some("sid-1"));
    }

    #[test]
    fn chat_prompt_cache_breakpoint_keeps_marked_text_as_blocks() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"prefix","prompt_cache_breakpoint":{"mode":"explicit"}},
                {"type":"text","text":"suffix"}
            ]}]}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        match &canon.messages[0].content {
            CanonicalContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(blocks[0].cache_mark().is_some());
                assert!(blocks[1].cache_mark().is_none());
            }
            other => panic!("marked text must not collapse to a string: {other:?}"),
        }
    }

    #[test]
    fn chat_invalid_prompt_cache_options_ttl_is_400() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "prompt_cache_options":{"ttl":"1h"}}"#,
        )
        .unwrap();
        let err = to_canonical(&req).expect_err("OpenAI ttl 1h is not legal");
        assert!(err.contains("prompt_cache_options.ttl"), "{err}");
    }

    #[test]
    fn to_canonical_strips_stream_options_as_gateway_only() {
        // WHY: chat clients send stream_options (include_usage) as transport
        // metadata. Codex/Responses reject it as an unsupported provider extra;
        // strip at the chat→canonical boundary instead of failing the request.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "stream":true,"stream_options":{"include_usage":true},
                "response_format":{"type":"json_object"}}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).expect("chat request should convert");
        let extras = canon
            .provider_extras
            .expect("non-gateway extras should remain");
        assert_eq!(extras["response_format"]["type"], "json_object");
        assert!(
            extras.get("stream_options").is_none(),
            "stream_options must not become a provider extra: {extras}"
        );
    }

    #[test]
    fn to_canonical_maps_reasoning_effort_and_strips_from_extras() {
        // WHY: OpenAI chat clients send top-level reasoning_effort. Mapping it
        // into provider_extras causes edge allowlists to 400 (issue #16). It
        // must become CanonicalReasoning.effort and leave provider_extras.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":"high","response_format":{"type":"json_object"}}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).expect("chat request should convert");
        assert_eq!(
            canon.reasoning.expect("reasoning mapped").effort.as_deref(),
            Some("high")
        );
        let extras = canon
            .provider_extras
            .expect("unrelated extras should remain");
        assert_eq!(extras["response_format"]["type"], "json_object");
        assert!(
            extras.get("reasoning_effort").is_none(),
            "reasoning_effort must not become a provider extra: {extras}"
        );
    }

    #[test]
    fn to_canonical_maps_nested_reasoning_effort_shape() {
        // WHY: some clients send Responses-style reasoning:{effort} on chat.
        // Accept and lift so it is not rejected as an unsupported extra.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning":{"effort":"medium"}}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).expect("nested reasoning converts");
        assert_eq!(
            canon.reasoning.expect("reasoning mapped").effort.as_deref(),
            Some("medium")
        );
        assert!(
            canon.provider_extras.is_none(),
            "nested reasoning must be stripped from extras, got {:?}",
            canon.provider_extras
        );
    }

    #[test]
    fn to_canonical_top_level_reasoning_effort_wins_over_nested() {
        // WHY: when both shapes appear, the OpenAI chat top-level field is the
        // primary contract and must win.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":"high","reasoning":{"effort":"low"}}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        assert_eq!(
            canon.reasoning.expect("reasoning mapped").effort.as_deref(),
            Some("high")
        );
        assert!(canon.provider_extras.is_none());
    }

    #[test]
    fn to_canonical_accepts_xhigh_and_unknown_effort_names() {
        // WHY (issue #20): chat must not hard-allowlist effort names. Real
        // upstream levels (Anthropic/OpenAI `xhigh`) and other well-formed
        // free strings lift into CanonicalReasoning; adapters map or fail loud.
        for effort in ["xhigh", "ultra", "extreme"] {
            let req: ChatCompletionRequest = serde_json::from_str(&format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],
                    "reasoning_effort":"{effort}"}}"#
            ))
            .unwrap();
            let canon = to_canonical(&req).unwrap_or_else(|e| {
                panic!("well-formed effort {effort:?} must accept at chat edge: {e}")
            });
            assert_eq!(
                canon.reasoning.expect("reasoning mapped").effort.as_deref(),
                Some(effort)
            );
        }
    }

    #[test]
    fn to_canonical_rejects_lexically_invalid_reasoning_effort() {
        // WHY (issue #20): edge keeps lexical hygiene only (empty, over-long,
        // unsafe charset). No closed valid-values list.
        let empty: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":""}"#,
        )
        .unwrap();
        let err = to_canonical(&empty).expect_err("empty effort must reject");
        assert!(
            err.contains("reasoning_effort") && err.contains("empty"),
            "error must name empty: {err}"
        );

        let bad_chars: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":"hi there"}"#,
        )
        .unwrap();
        let err = to_canonical(&bad_chars).expect_err("unsafe charset must reject");
        assert!(
            err.contains("reasoning_effort") && err.contains("unsafe"),
            "error must name charset: {err}"
        );

        let too_long = "a".repeat(MAX_REASONING_EFFORT_LEN + 1);
        let long: ChatCompletionRequest = serde_json::from_str(&format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],
                "reasoning_effort":"{too_long}"}}"#
        ))
        .unwrap();
        let err = to_canonical(&long).expect_err("over-long effort must reject");
        assert!(
            err.contains("reasoning_effort") && err.contains("max"),
            "error must name length bound: {err}"
        );
    }

    #[test]
    fn to_canonical_accepts_exact_max_reasoning_effort_length() {
        // WHY (issue #29): MAX+1 rejects; exact-MAX must accept on the real
        // chat entry path (to_canonical), not a reimplemented validator.
        let exact = "a".repeat(MAX_REASONING_EFFORT_LEN);
        let req: ChatCompletionRequest = serde_json::from_str(&format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],
                "reasoning_effort":"{exact}"}}"#
        ))
        .unwrap();
        let canon = to_canonical(&req)
            .unwrap_or_else(|e| panic!("exact-MAX length effort must accept on chat path: {e}"));
        assert_eq!(
            canon.reasoning.expect("reasoning mapped").effort.as_deref(),
            Some(exact.as_str())
        );
    }

    #[test]
    fn to_canonical_rejects_non_string_reasoning_effort() {
        // WHY: a non-string effort is malformed input; reject by name.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":3}"#,
        )
        .unwrap();
        let err = to_canonical(&req).expect_err("non-string effort must reject");
        assert!(
            err.contains("reasoning_effort") && err.contains("string"),
            "error must name the type requirement: {err}"
        );
    }

    #[test]
    fn to_canonical_none_effort_is_explicit_disable() {
        // WHY: preserve effort:"none" so Codex can emit reasoning.effort:none.
        // Claude/Grok adapters omit "none"; collapsing to reasoning:None would
        // make chat disable indistinguishable from "field absent".
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":"none"}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        assert_eq!(
            canon.reasoning.expect("none preserved").effort.as_deref(),
            Some("none")
        );
        assert!(
            canon.provider_extras.is_none(),
            "none must not leave reasoning_effort as an extra"
        );
    }

    #[test]
    fn to_canonical_null_reasoning_effort_is_absent() {
        // WHY: clients often send explicit null for unset fields. Null must not
        // 400 as a type error or become an unsupported extra.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":null,"reasoning":null}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        assert!(canon.reasoning.is_none());
        assert!(
            canon.provider_extras.is_none(),
            "null reasoning fields must be stripped: {:?}",
            canon.provider_extras
        );
    }

    #[test]
    fn to_canonical_null_top_level_does_not_block_nested_effort() {
        // WHY: null means absent, not disable. A nested effort must still lift
        // when top-level is JSON null (common SDK "unset" shape).
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":null,"reasoning":{"effort":"high"}}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        assert_eq!(
            canon
                .reasoning
                .expect("nested effort lifted")
                .effort
                .as_deref(),
            Some("high")
        );
        assert!(canon.provider_extras.is_none());
    }

    #[test]
    fn to_canonical_empty_reasoning_object_is_absent() {
        // WHY: `{}` is not a useful extra; treating it as unsupported is noise.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning":{}}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        assert!(canon.reasoning.is_none());
        assert!(canon.provider_extras.is_none());
    }

    #[test]
    fn to_canonical_preserves_nested_reasoning_siblings_in_extras() {
        // WHY: lifting effort must not silently drop sibling keys (summary).
        // Leave them in extras so the allowlist can fail loud if unsupported.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning":{"effort":"high","summary":"auto"}}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        assert_eq!(
            canon.reasoning.expect("effort lifted").effort.as_deref(),
            Some("high")
        );
        let extras = canon.provider_extras.expect("siblings remain as extras");
        assert!(
            extras
                .get("reasoning")
                .and_then(|r| r.get("effort"))
                .is_none()
        );
        assert_eq!(extras["reasoning"]["summary"], "auto");
    }

    #[test]
    fn to_canonical_accepts_minimal_effort() {
        // WHY: OpenAI reasoning models document "minimal"; rejecting it at the
        // chat boundary forces clients into opaque upstream failures.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":"minimal"}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).unwrap();
        assert_eq!(
            canon.reasoning.expect("minimal mapped").effort.as_deref(),
            Some("minimal")
        );
    }

    #[test]
    fn to_canonical_maps_chat_text_and_image_parts_in_order() {
        // WHY: modern OpenAI-compatible clients send typed content arrays for
        // multimodal prompts. Image URL and data URL parts must survive in the
        // same order as adjacent text instead of flattening to plain text.
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"look"},
                {"type":"image_url","image_url":{"url":"https://example.com/cat.png"}},
                {"type":"text","text":"again"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}
            ]}]}"#,
        )
        .unwrap();
        let canon = to_canonical(&req).expect("multimodal chat content converts");
        match &canon.messages[0].content {
            CanonicalContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 4);
                assert!(matches!(&blocks[0], CanonicalBlock::Text { text, .. } if text == "look"));
                assert!(matches!(
                    &blocks[1],
                    CanonicalBlock::Image {
                        source: omni_core::CanonicalImageSource::Url { url }, .. } if url == "https://example.com/cat.png"
                ));
                assert!(matches!(&blocks[2], CanonicalBlock::Text { text, .. } if text == "again"));
                assert!(matches!(
                    &blocks[3],
                    CanonicalBlock::Image {
                        source: omni_core::CanonicalImageSource::Base64 { media_type, data }, .. } if media_type == "image/png" && data == "aGVsbG8="
                ));
            }
            CanonicalContent::Text(_) => panic!("image content must become blocks"),
        }
    }

    #[test]
    fn to_canonical_rejects_unknown_chat_content_part() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"user","content":[
                {"type":"input_audio","audio":"..."}
            ]}]}"#,
        )
        .unwrap();
        let err = to_canonical(&req).expect_err("unsupported content part must reject");
        assert!(err.contains("input_audio"), "error must name part: {err}");
    }

    fn req_from_json(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("chat request json")
    }

    #[test]
    fn to_canonical_maps_tools_and_tool_choice_modes() {
        // WHY: a client that declares tools must have them reach the provider;
        // before this wiring to_canonical hardcoded tools:None and the model
        // never saw the tools. Each tool_choice mode must map to its canonical
        // equivalent, and "none" must KEEP the tools visible (not drop them).
        let req = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object"}}}],
                "tool_choice":"auto"}"#,
        );
        let canon = to_canonical(&req).unwrap();
        let tools = canon.tools.as_ref().expect("tools must reach canonical");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert!(matches!(canon.tool_choice, Some(CanonicalToolChoice::Auto)));

        for (mode, want) in [
            ("required", CanonicalToolChoice::Required),
            ("none", CanonicalToolChoice::None),
        ] {
            let req = req_from_json(&format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],
                    "tools":[{{"type":"function","function":{{"name":"f"}}}}],
                    "tool_choice":"{mode}"}}"#,
            ));
            let canon = to_canonical(&req).unwrap();
            assert!(
                canon.tools.is_some(),
                "tool_choice {mode} must keep tools visible"
            );
            assert_eq!(
                std::mem::discriminant(canon.tool_choice.as_ref().unwrap()),
                std::mem::discriminant(&want),
                "tool_choice {mode} mapped wrong"
            );
        }
    }

    #[test]
    fn to_canonical_filters_chat_allowed_tools_for_both_modes() {
        for (mode, required) in [("auto", false), ("required", true)] {
            let req = req_from_json(&format!(
                r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}],
                    "tools":[
                        {{"type":"function","function":{{"name":"read"}}}},
                        {{"type":"function","function":{{"name":"write"}}}},
                        {{"type":"function","function":{{"name":"inspect"}}}}
                    ],
                    "tool_choice":{{"type":"allowed_tools","allowed_tools":{{
                        "mode":"{mode}","tools":[
                            {{"type":"function","function":{{"name":"read"}}}},
                            {{"type":"function","function":{{"name":"inspect"}}}}
                        ]
                    }}}}}}"#,
            ));
            let canon = to_canonical(&req).expect("allowed tools must convert");
            let names = canon
                .tools
                .as_ref()
                .expect("allowed tools remain")
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(names, ["read", "inspect"]);
            assert!(
                matches!(
                    &canon.tool_choice,
                    Some(CanonicalToolChoice::Required) if required
                ) || matches!(
                    &canon.tool_choice,
                    Some(CanonicalToolChoice::Auto) if !required
                )
            );
        }
    }

    #[test]
    fn chat_allowed_tools_rejects_invalid_or_unsafe_subsets() {
        let undeclared = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"read"}}],
                "tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"auto","tools":[
                    {"type":"function","function":{"name":"write"}}
                ]}}}"#,
        );
        let err = to_canonical(&undeclared).expect_err("undeclared tool must reject");
        assert!(err.contains("write"), "error must name the tool: {err}");

        let bad_mode = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"read"}}],
                "tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"sometimes","tools":[
                    {"type":"function","function":{"name":"read"}}
                ]}}}"#,
        );
        let err = to_canonical(&bad_mode).expect_err("bad mode must reject");
        assert!(err.contains("sometimes"), "error must name the mode: {err}");

        let empty_required = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"read"}}],
                "tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[]}}}"#,
        );
        let err = to_canonical(&empty_required).expect_err("empty required subset must reject");
        assert!(err.contains("at least one tool"), "unexpected error: {err}");

        let empty_auto = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"read"}}],
                "tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"auto","tools":[]}}}"#,
        );
        let canon = to_canonical(&empty_auto).expect("empty auto subset disables all tools");
        assert!(canon.tools.is_none());
        assert!(canon.tool_choice.is_none());
    }

    #[test]
    fn to_canonical_maps_specific_tool_choice() {
        // WHY: a forced function call ({type:function,function:{name}}) must map
        // to Specific so the provider can require exactly that tool.
        let req = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"f"}}],
                "tool_choice":{"type":"function","function":{"name":"f"}}}"#,
        );
        match to_canonical(&req).unwrap().tool_choice {
            Some(CanonicalToolChoice::Specific { name }) => assert_eq!(name, "f"),
            other => panic!("expected Specific, got {other:?}"),
        }
    }

    #[test]
    fn to_canonical_maps_assistant_tool_calls_to_blocks() {
        // WHY: a multi-turn tool conversation feeds the assistant's prior
        // tool_calls back in the next request; they must survive into canonical
        // ToolUse blocks (keyed by id) rather than being dropped, or the upstream
        // loses the call it is answering.
        let req = req_from_json(
            r#"{"model":"m","messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]}
            ]}"#,
        );
        let canon = to_canonical(&req).unwrap();
        match &canon.messages[1].content {
            CanonicalContent::Blocks(blocks) => match &blocks[0] {
                CanonicalBlock::ToolUse {
                    id,
                    name,
                    arguments,
                    ..
                } => {
                    assert_eq!(id, "call_1");
                    assert_eq!(name, "get_weather");
                    assert!(arguments.contains("SF"));
                }
                other => panic!("expected ToolUse, got {other:?}"),
            },
            other => panic!("assistant tool_calls must become Blocks, got {other:?}"),
        }
    }

    #[test]
    fn to_canonical_maps_tool_role_to_tool_result_block() {
        // WHY: the tool result the client feeds back (role:"tool", keyed by
        // tool_call_id) must become a ToolResult block linked to its call, or the
        // model cannot tie the result to the request it made.
        let req = req_from_json(
            r#"{"model":"m","messages":[
                {"role":"tool","tool_call_id":"call_1","content":"72F"}]}"#,
        );
        let canon = to_canonical(&req).unwrap();
        match &canon.messages[0].content {
            CanonicalContent::Blocks(blocks) => match &blocks[0] {
                CanonicalBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    ..
                } => {
                    assert_eq!(tool_use_id, "call_1");
                    assert_eq!(content, "72F");
                    assert!(!is_error);
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            other => panic!("tool role must become Blocks, got {other:?}"),
        }
    }

    #[test]
    fn to_canonical_rejects_malformed_tool_surfaces_by_name() {
        // WHY: malformed tool input must fail LOUDLY (a 400 naming the offender)
        // rather than being silently dropped, so a broken integration is
        // debuggable instead of producing wrong answers.
        // Non-function tool type.
        let req = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"retrieval","function":{"name":"f"}}]}"#,
        );
        let err = to_canonical(&req).expect_err("non-function tool must reject");
        assert!(err.contains("retrieval"), "error must name the type: {err}");

        // tool message missing tool_call_id.
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"tool","content":"42"}]}"#);
        let err = to_canonical(&req).expect_err("tool msg without id must reject");
        assert!(
            err.contains("tool_call_id"),
            "error must name the missing field: {err}"
        );

        // Unknown tool_choice mode.
        let req = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"f"}}],
                "tool_choice":"bogus"}"#,
        );
        let err = to_canonical(&req).expect_err("unknown mode must reject");
        assert!(err.contains("bogus"), "error must name the mode: {err}");
    }

    #[test]
    fn to_canonical_rejects_non_function_tool_choice_type() {
        // WHY: a forced tool_choice object must select a `function`. A non-function
        // type (e.g. "retrieval") is a shape we do not support; coercing it to
        // Specific would force a tool the model cannot dispatch. Reject loudly,
        // naming the offending type, instead of silently mistranslating it.
        let req = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"f"}}],
                "tool_choice":{"type":"retrieval","function":{"name":"f"}}}"#,
        );
        let err = to_canonical(&req).expect_err("non-function tool_choice type must reject");
        assert!(
            err.contains("retrieval") || err.contains("type"),
            "error must name the bad tool_choice type: {err}"
        );
    }

    #[test]
    fn to_canonical_rejects_tool_calls_on_non_assistant_role() {
        // WHY: tool_calls are only valid on an assistant message (OpenAI
        // contract). A user/other role carrying tool_calls is a malformed history
        // shape; forwarding it would feed a backend an invalid turn. Reject it,
        // naming the role/field, rather than silently accepting.
        let req = req_from_json(
            r#"{"model":"m","messages":[
                {"role":"user","content":"hi","tool_calls":[
                    {"id":"x","type":"function","function":{"name":"f","arguments":"{}"}}]}
            ]}"#,
        );
        let err = to_canonical(&req).expect_err("tool_calls on non-assistant must reject");
        assert!(
            err.contains("assistant") || err.contains("tool_calls"),
            "error must explain tool_calls are assistant-only: {err}"
        );

        // A role:"tool" message carrying tool_calls must ALSO reject -- the
        // tool-result branch is reached first, so the assistant-only check has
        // to run BEFORE it or the tool_calls would be silently dropped.
        let req = req_from_json(
            r#"{"model":"m","messages":[
                {"role":"tool","tool_call_id":"c1","content":"42","tool_calls":[
                    {"id":"x","type":"function","function":{"name":"f","arguments":"{}"}}]}
            ]}"#,
        );
        let err = to_canonical(&req).expect_err("tool_calls on tool role must reject");
        assert!(
            err.contains("assistant") || err.contains("tool_calls"),
            "tool role with tool_calls must reject, not drop them: {err}"
        );
    }

    #[test]
    fn from_canonical_text_response_shape() {
        // WHY: pins the OpenAI response envelope (object tag, echoed model,
        // usage totals, default stop reason) that every client parses.
        let canon = CanonicalResponse {
            model: "backend-model".into(),
            content: "hello".into(),
            tool_calls: vec![],
            finish_reason: None,
            usage: CanonicalUsage {
                input_tokens: 3,
                output_tokens: 4,
                ..Default::default()
            },
            id: None,
            refusal: None,
            ..Default::default()
        };
        let oai = from_canonical(
            canon,
            "prefix:backend-model".into(),
            "chatcmpl-1".into(),
            123,
        );
        assert_eq!(oai.object, "chat.completion");
        assert_eq!(oai.model, "prefix:backend-model");
        assert_eq!(oai.choices[0].message.content.as_deref(), Some("hello"));
        assert_eq!(oai.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(oai.usage.prompt_tokens, 3);
        assert_eq!(oai.usage.completion_tokens, 4);
        assert_eq!(oai.usage.total_tokens, 7);
    }

    #[test]
    fn from_canonical_preserves_refusal_field() {
        // WHY: OpenAI-compatible chat responses can carry refusal separately
        // from text. Providers that expose canonical refusal must not lose it
        // when served through the Chat Completions surface.
        let canon = CanonicalResponse {
            model: "backend-model".into(),
            content: String::new(),
            refusal: Some("policy".into()),
            tool_calls: vec![],
            finish_reason: Some("content_filter".into()),
            usage: CanonicalUsage::default(),
            id: None,
            ..Default::default()
        };
        let oai = from_canonical(canon, "m".into(), "id".into(), 0);
        let v = serde_json::to_value(&oai).unwrap();
        assert_eq!(v["choices"][0]["message"]["refusal"], "policy");
        assert_eq!(v["choices"][0]["finish_reason"], "content_filter");
    }

    #[test]
    fn from_canonical_tool_calls_finish_reason() {
        // WHY: when the model returns tool calls and no text, content must be
        // null and finish_reason must be tool_calls per the OpenAI contract.
        let canon = CanonicalResponse {
            model: "m".into(),
            content: String::new(),
            tool_calls: vec![CanonicalToolCall {
                id: "call_1".into(),
                name: "f".into(),
                arguments: "{}".into(),
            }],
            finish_reason: None,
            usage: CanonicalUsage::default(),
            id: None,
            refusal: None,
            ..Default::default()
        };
        let oai = from_canonical(canon, "m".into(), "id".into(), 0);
        assert!(oai.choices[0].message.content.is_none());
        assert_eq!(oai.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(oai.choices[0].message.tool_calls.len(), 1);
    }

    #[test]
    fn from_canonical_emits_rich_usage_and_provider_metadata_when_present() {
        // WHY: rich provider details are opt-in extension fields. They should be
        // visible when available without changing minimal responses.
        let canon = CanonicalResponse {
            model: "m".into(),
            content: "ok".into(),
            usage: CanonicalUsage {
                input_tokens: 5,
                output_tokens: 2,
                cache_read: 1,
                reasoning_tokens: 9,
                input_audio_tokens: 5,
                output_audio_tokens: 6,
                num_sources_used: 3,
                ..Default::default()
            },
            metadata: Some(omni_core::CanonicalResponseMetadata {
                system_fingerprint: Some("fp_1".into()),
                service_tier: Some("priority".into()),
                provider: Some("grok".into()),
                ..Default::default()
            }),
            annotations: vec![serde_json::json!({"type":"url_citation","url":"https://e.test"})],
            ..Default::default()
        };
        let v = serde_json::to_value(from_canonical(canon, "m".into(), "id".into(), 1)).unwrap();
        assert_eq!(v["system_fingerprint"], "fp_1");
        assert_eq!(v["service_tier"], "priority");
        assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 1);
        assert_eq!(v["usage"]["prompt_tokens_details"]["audio_tokens"], 5);
        assert_eq!(
            v["usage"]["completion_tokens_details"]["reasoning_tokens"],
            9
        );
        assert_eq!(v["usage"]["completion_tokens_details"]["audio_tokens"], 6);
        assert_eq!(v["provider_metadata"]["provider"], "grok");
        assert_eq!(v["provider_metadata"]["num_sources_used"], 3);
        assert_eq!(
            v["provider_metadata"]["annotations"][0]["url"],
            "https://e.test"
        );
    }

    #[tokio::test]
    async fn sse_frames_canonical_stream_with_done_terminator() {
        // WHY: the streaming HTTP contract is "one data: chunk per event,
        // terminated by data: [DONE]". A client that does not see [DONE] hangs;
        // a missing finish chunk breaks finish_reason handling. This drives the
        // exact framing the binaries expose.
        use axum::response::IntoResponse;
        use omni_core::ProviderError;

        let events: Vec<Result<CanonicalStreamEvent, ProviderError>> = vec![
            Ok(CanonicalStreamEvent::TextDelta("Hel".into())),
            Ok(CanonicalStreamEvent::TextDelta("lo".into())),
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_1".into()),
                name: Some("f".into()),
                arguments_delta: "{}".into(),
            }),
            Ok(CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 1,
                output_tokens: 2,
                ..Default::default()
            })),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("tool_calls".into()),
            }),
        ];
        let canon: CanonicalStream = Box::pin(futures_util::stream::iter(events));
        let sse = sse_from_canonical_stream(canon, "m".into(), "chatcmpl-x".into(), 7);

        // Render the SSE body to bytes and assert the wire content.
        let resp = sse.into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            text.contains("\"content\":\"Hel\""),
            "first text delta framed"
        );
        assert!(
            text.contains("\"content\":\"lo\""),
            "second text delta framed"
        );
        assert!(
            text.contains("\"tool_calls\"") && text.contains("call_1"),
            "tool-call delta framed"
        );
        assert!(
            text.contains("\"finish_reason\":\"tool_calls\""),
            "finish chunk carries mapped reason"
        );
        assert!(
            text.trim_end().ends_with("[DONE]"),
            "stream terminated by [DONE]"
        );
        // Usage events are intentionally not framed as chunks at this layer.
        assert!(
            !text.contains("\"usage\""),
            "usage not emitted in default stream"
        );
    }

    async fn render_sse(
        events: Vec<Result<CanonicalStreamEvent, omni_core::ProviderError>>,
    ) -> String {
        use axum::response::IntoResponse;
        let canon: CanonicalStream = Box::pin(futures_util::stream::iter(events));
        let sse = sse_from_canonical_stream(canon, "m".into(), "chatcmpl-x".into(), 7);
        let resp = sse.into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        String::from_utf8(body.to_vec()).unwrap()
    }

    #[test]
    fn sse_content_type_is_media_type_not_raw_prefix() {
        assert!(is_sse_content_type(Some("text/event-stream")));
        assert!(is_sse_content_type(Some(
            "text/event-stream; charset=utf-8"
        )));
        assert!(is_sse_content_type(Some("Text/Event-Stream")));
        assert!(!is_sse_content_type(None));
        assert!(!is_sse_content_type(Some("application/json")));
        assert!(!is_sse_content_type(Some("text/event-streamfoo")));
        assert!(!is_sse_content_type(Some("")));
    }

    #[test]
    fn allowed_chat_finish_reasons_are_the_openai_set() {
        for reason in [
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "function_call",
        ] {
            assert!(is_allowed_chat_finish_reason(reason), "{reason}");
        }
        assert!(!is_allowed_chat_finish_reason("error"));
        assert!(!is_allowed_chat_finish_reason("error: overloaded"));
        assert!(!is_allowed_chat_finish_reason(
            "model_context_window_exceeded"
        ));
    }

    #[tokio::test]
    async fn sse_mid_stream_err_is_named_error_event_not_illegal_finish() {
        use omni_core::ProviderError;
        let text = render_sse(vec![
            Ok(CanonicalStreamEvent::TextDelta("partial".into())),
            Err(ProviderError::upstream("boom secret")),
        ])
        .await;
        assert!(text.contains("event: error"), "{text}");
        assert!(text.contains("upstream stream error"), "{text}");
        assert!(text.contains("[DONE]"), "{text}");
        assert!(!text.contains("finish_reason\":\"error"), "{text}");
        assert!(!text.contains("boom secret"), "{text}");
        assert!(!text.contains("\"finish_reason\":\"stop\""), "{text}");
    }

    #[tokio::test]
    async fn sse_eof_without_finish_is_error_event_not_synthesized_stop() {
        let text = render_sse(vec![Ok(CanonicalStreamEvent::TextDelta("partial".into()))]).await;
        assert!(text.contains("event: error"), "{text}");
        assert!(text.contains("[DONE]"), "{text}");
        assert!(!text.contains("\"finish_reason\":\"stop\""), "{text}");
        assert!(!text.contains("finish_reason\":\"error"), "{text}");
    }

    #[tokio::test]
    async fn sse_unknown_finish_reason_coerces_to_stop() {
        let text = render_sse(vec![Ok(CanonicalStreamEvent::Finish {
            finish_reason: Some("model_context_window_exceeded".into()),
        })])
        .await;
        assert!(text.contains("\"finish_reason\":\"stop\""), "{text}");
        assert!(!text.contains("model_context_window_exceeded"), "{text}");
        assert!(!text.contains("event: error"), "{text}");
    }

    #[tokio::test]
    async fn sse_error_prefix_finish_is_fixed_error_event() {
        let text = render_sse(vec![Ok(CanonicalStreamEvent::Finish {
            finish_reason: Some("error: overloaded".into()),
        })])
        .await;
        assert!(text.contains("event: error"), "{text}");
        assert!(text.contains("upstream stream error"), "{text}");
        assert!(!text.contains("overloaded"), "{text}");
        assert!(!text.contains("finish_reason\":\"error"), "{text}");
        assert!(text.trim_end().ends_with("[DONE]"), "{text}");
    }

    #[tokio::test]
    async fn sse_stops_polling_after_first_finish() {
        let text = render_sse(vec![
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into()),
            }),
            Err(omni_core::ProviderError::upstream("late")),
        ])
        .await;
        assert!(text.contains("\"finish_reason\":\"stop\""), "{text}");
        assert!(!text.contains("event: error"), "{text}");
        assert!(!text.contains("late"), "{text}");
        assert_eq!(text.matches("[DONE]").count(), 1, "{text}");
    }
}

#[cfg(test)]
mod issue_51_tool_tests {
    use super::*;
    use serde_json::{Value, json};

    fn parse(schema: Value, strict: Value) -> Result<CanonicalRequest, String> {
        let mut function = json!({"name":"f", "parameters":schema});
        function["strict"] = strict;
        let req: ChatCompletionRequest = serde_json::from_value(json!({"model":"sonnet","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":function}]})).unwrap();
        to_canonical(&req)
    }

    #[test]
    fn chat_keeps_schema_and_strict_null_is_false() {
        let schema = json!({"type":"object","properties":{"b":{"type":"string"}},"oneOf":[{"properties":{"a":{"type":"number"}}}]});
        for flag in [Value::Null, json!(false)] {
            let canon = parse(schema.clone(), flag).unwrap();
            let tool = &canon.tools.unwrap()[0];
            assert_eq!(tool.parameters, schema);
            assert!(!tool.strict);
        }
        assert!(
            parse(
                json!({"type":"object","properties":{"x":{"anyOf":[{}]}}}),
                json!(true)
            )
            .unwrap()
            .tools
            .unwrap()[0]
                .strict
        );
    }

    #[test]
    fn chat_rejects_invalid_combinators_before_dispatch() {
        for schema in [
            json!({"oneOf":[]}),
            json!({"properties":{"x":{"allOf":null}}}),
            json!({"type":"object","oneOf":[{}]}),
        ] {
            assert!(parse(schema, json!(true)).is_err());
        }
        assert!(parse(json!({"type":"object","oneOf":[]}), Value::Null).is_err());
        assert!(
            parse(
                json!({"type":"object","properties":{"x":{"oneOf":[{}]}}}),
                json!(true)
            )
            .is_err()
        );
    }
}
