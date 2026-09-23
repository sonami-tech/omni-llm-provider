//! OpenAI Responses API protocol surface (`POST /v1/responses`).
//!
//! Second inbound protocol next to Chat Completions (`http.rs`): parses the
//! Responses request shape (`input` items, `instructions`, flattened function
//! tools, `reasoning.effort`), converts to/from the canonical types, and frames
//! a canonical stream as Responses SSE events (`response.created`,
//! `response.output_text.delta`, ..., `response.completed`) with increasing
//! `sequence_number`s and NO `[DONE]` sentinel (that is a Chat Completions
//! convention only).
//!
//! Scope: text input (string or message items with `input_text`/`output_text`
//! parts), image input (`input_image` with URL or base64 data URL), file input
//! (`input_file`), function tools, and full multi-turn tool conversations
//! (`function_call` / `function_call_output` items round-trip through canonical
//! tool blocks), non-stream and stream. Input shapes the canonical layer still
//! cannot represent (e.g. audio parts, non-function tools) are rejected
//! loudly with a clear error instead of degraded silently.
//!
//! The types below are the wire contract. The conversion and SSE-framing
//! functions are pinned by tests in this module.

use std::convert::Infallible;
use std::pin::Pin;

use axum::response::sse::{Event, Sse};
use futures_util::Stream;
use serde::{Deserialize, Serialize};

use crate::cache::{
    parse_openai_cache_intent, parse_prompt_cache_breakpoint, strip_openai_cache_keys,
};
use crate::canonical_mapping::{provider_metadata_json, usage_detail_json};
use crate::http::{gateway_only_extra_keys, validate_reasoning_effort_lexical};

use omni_core::{
    CanonicalBlock, CanonicalContent, CanonicalFileSource, CanonicalImageSource, CanonicalMessage,
    CanonicalReasoning, CanonicalRequest, CanonicalResponse, CanonicalResponseMetadata,
    CanonicalStream, CanonicalStreamEvent, CanonicalTool, CanonicalToolChoice,
};

/// The boxed SSE event stream produced by the Responses framer. Boxed so the
/// signature stays stable while the implementation evolves.
pub type ResponsesSseStream = Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

// ── Request (Deserialize) ─────────────────────────────────────────

/// `POST /v1/responses` request body (supported subset). Unknown fields are
/// captured in `extras` so a request never fails to parse on an
/// unrecognized key.
#[derive(Debug, Deserialize, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default)]
    pub instructions: Option<ResponsesInstructions>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub reasoning: Option<ResponsesReasoning>,
    #[serde(default)]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default)]
    pub tool_choice: Option<ResponsesToolChoice>,
    #[serde(flatten)]
    pub extras: serde_json::Value,
}

/// `input` is either a bare string (one user message) or a list of items.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

/// One entry of an `input` array. Message items carry `role` + `content`
/// (`type` defaults to "message" when absent). Tool-conversation items use
/// `type:"function_call"` (the assistant's prior tool call) and
/// `type:"function_call_output"` (the result fed back), keyed by `call_id`.
#[derive(Debug, Deserialize, Serialize)]
pub struct ResponsesInputItem {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<ResponsesInputContent>,
    /// `function_call` / `function_call_output`: links a call to its result.
    #[serde(default)]
    pub call_id: Option<String>,
    /// `function_call`: the called tool's name.
    #[serde(default)]
    pub name: Option<String>,
    /// `function_call`: the raw JSON arguments string.
    #[serde(default)]
    pub arguments: Option<String>,
    /// `function_call_output`: the tool's result text.
    #[serde(default)]
    pub output: Option<String>,
}

/// Message content: a bare string or typed parts.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponsesInputContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

/// Top-level Responses `instructions`: a string, or typed parts. Breakpoints
/// on this field are forbidden (put marked instructions in a developer
/// `input_text` part).
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponsesInstructions {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

/// One typed content part. Supports text, images and files.
/// Other media part types are rejected by name.
#[derive(Debug, Deserialize, Serialize)]
pub struct ResponsesContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<String>,
    #[serde(default)]
    pub file_id: Option<String>,
    #[serde(default)]
    pub file_url: Option<String>,
    #[serde(default)]
    pub file_data: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub prompt_cache_breakpoint: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResponsesReasoning {
    #[serde(default)]
    pub effort: Option<String>,
}

/// Responses tools are FLATTENED function definitions (unlike Chat
/// Completions' nested `function` object): `{type:"function", name, ...}`.
#[derive(Debug, Deserialize, Serialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    /// "auto" | "required" | "none"
    Mode(String),
    /// `{type:"function", name:"..."}`
    Function {
        #[serde(rename = "type")]
        kind: String,
        name: String,
    },
    /// `{type:"allowed_tools",mode,tools:[...]}`
    Allowed {
        #[serde(rename = "type")]
        kind: String,
        mode: String,
        tools: Vec<ResponsesAllowedTool>,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResponsesAllowedTool {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
}

// ── Response (Serialize) ──────────────────────────────────────────

/// Non-streaming `POST /v1/responses` response envelope (supported subset).
#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str, // always "response"
    pub created_at: u64,
    /// "completed" | "incomplete" | "failed"
    pub status: String,
    pub model: String,
    pub output: Vec<ResponsesOutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponsesError>,
    pub usage: ResponsesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        status: String,
        role: &'static str, // "assistant"
        content: Vec<ResponsesOutputContent>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: String,
    },
}

#[derive(Debug, Serialize)]
pub struct ResponsesOutputContent {
    #[serde(rename = "type")]
    pub kind: &'static str, // "output_text" | "refusal"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    pub annotations: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct IncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponsesError {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub code: Option<String>,
}

/// Responses usage uses input/output naming (NOT prompt/completion like Chat).
#[derive(Debug, Serialize, Default)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<serde_json::Value>,
}

// ── Conversions + SSE framing ─────────────────────────────────────

/// Convert a Responses request into a `CanonicalRequest`.
///
/// Mapping contract (pinned by tests):
/// - string `input` -> a single "user" message
/// - `instructions` -> a leading "system" message
/// - message items: role preserved ("developer" maps to "system"); string
///   content used verbatim; multiple text parts joined with "\n"
/// - `function_call` item -> an assistant message with a ToolUse block;
///   `function_call_output` item -> a "tool" message with a ToolResult block
///   (both keyed by `call_id`), so multi-turn tool loops round-trip
/// - `max_output_tokens`/`temperature`/`top_p` -> canonical sampling
/// - `reasoning.effort` -> canonical reasoning effort
/// - unknown top-level Responses fields -> canonical provider_extras, where
///   each backend can allowlist the fields it natively supports
/// - flattened function tools -> canonical tools; `tool_choice` "auto" ->
///   Auto, "required" -> Required, "none" -> None (tools stay visible, model
///   must not call), `{type:"function",name}` -> Specific
/// - unsupported shapes (unknown item types, unsupported media parts,
///   non-function tools, message items without a role) -> Err naming the offender
pub fn responses_to_canonical(req: &ResponsesRequest) -> Result<CanonicalRequest, String> {
    let mut messages: Vec<CanonicalMessage> = Vec::new();

    // `instructions` is the Responses system prompt: a leading system message.
    // Breakpoints on this field are forbidden.
    if let Some(instructions) = req.instructions.as_ref() {
        messages.push(CanonicalMessage {
            role: "system".into(),
            content: responses_instructions_to_canonical(instructions)?,
        });
    }

    match &req.input {
        ResponsesInput::Text(text) => {
            messages.push(CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text(text.clone()),
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                match item.kind.as_deref() {
                    // The assistant's prior tool call -> a ToolUse block.
                    Some("function_call") => {
                        let call_id = item
                            .call_id
                            .clone()
                            .ok_or_else(|| "function_call item missing call_id".to_string())?;
                        let name = item
                            .name
                            .clone()
                            .ok_or_else(|| "function_call item missing name".to_string())?;
                        messages.push(CanonicalMessage {
                            role: "assistant".into(),
                            content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolUse {
                                id: call_id,
                                name,
                                arguments: item.arguments.clone().unwrap_or_default(),
                                cache: None,
                            }]),
                        });
                    }
                    // A tool result fed back -> a ToolResult block.
                    Some("function_call_output") => {
                        let call_id = item.call_id.clone().ok_or_else(|| {
                            "function_call_output item missing call_id".to_string()
                        })?;
                        messages.push(CanonicalMessage {
                            role: "tool".into(),
                            content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                                tool_use_id: call_id,
                                content: item.output.clone().unwrap_or_default(),
                                is_error: false,
                                cache: None,
                            }]),
                        });
                    }
                    // A message item (the default when `type` is absent).
                    None | Some("message") => {
                        let role = item
                            .role
                            .as_deref()
                            .ok_or_else(|| "message input item missing role".to_string())?;
                        let role = if role == "developer" { "system" } else { role };
                        let content = responses_content_to_canonical(item.content.as_ref())?;
                        messages.push(CanonicalMessage {
                            role: role.to_string(),
                            content,
                        });
                    }
                    Some(other) => {
                        return Err(format!("unsupported input item type: {other}"));
                    }
                }
            }
        }
    }

    // Flattened function tools -> canonical tools (non-function tools rejected).
    let mut tools = match req.tools.as_ref() {
        Some(ts) if !ts.is_empty() => {
            let mut out = Vec::with_capacity(ts.len());
            for t in ts {
                if t.kind != "function" {
                    return Err(format!("unsupported tool type: {}", t.kind));
                }
                let parameters = t
                    .parameters
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({}));
                omni_core::validate_tool_schema(&parameters, t.strict.unwrap_or(false), false)?;
                out.push(CanonicalTool {
                    name: t.name.clone().unwrap_or_default(),
                    description: t.description.clone(),
                    parameters,
                    strict: t.strict.unwrap_or(false),
                    cache: None,
                });
            }
            Some(out)
        }
        _ => None,
    };

    // Standard modes and one forced function map directly. An allowed subset
    // is enforced by filtering tools before provider dispatch.
    let tool_choice = match req.tool_choice.as_ref() {
        Some(ResponsesToolChoice::Mode(mode)) => match mode.as_str() {
            "auto" => Some(CanonicalToolChoice::Auto),
            "required" => Some(CanonicalToolChoice::Required),
            "none" => Some(CanonicalToolChoice::None),
            other => return Err(format!("unsupported tool_choice mode: {other}")),
        },
        Some(ResponsesToolChoice::Function { kind, name }) => {
            if kind != "function" {
                return Err(format!("unsupported tool_choice type: {kind}"));
            }
            Some(CanonicalToolChoice::Specific { name: name.clone() })
        }
        Some(ResponsesToolChoice::Allowed {
            kind,
            mode,
            tools: allowed_tools,
        }) => {
            if kind != "allowed_tools" {
                return Err(format!("unsupported tool_choice type: {kind}"));
            }
            let mut names = Vec::with_capacity(allowed_tools.len());
            for tool in allowed_tools {
                if tool.kind != "function" {
                    return Err(format!("unsupported allowed tool type: {}", tool.kind));
                }
                let name = tool
                    .name
                    .as_deref()
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| "allowed function tool requires a non-empty name".to_string())?;
                names.push(name.to_string());
            }
            restrict_tools_to_allowed(&mut tools, mode, &names)?
        }
        None => None,
    };

    let reasoning = match req.reasoning.as_ref() {
        None => None,
        Some(r) => {
            // Free-string effort with lexical hygiene only (issue #20 parity
            // with chat). No closed allowlist at the edge.
            if let Some(effort) = r.effort.as_deref() {
                validate_reasoning_effort_lexical(effort)?;
            }
            Some(CanonicalReasoning {
                effort: r.effort.clone(),
                budget_tokens: None,
            })
        }
    };

    let cache = parse_openai_cache_intent(&req.extras, None)?;
    let provider_extras = req.extras.as_object().and_then(|extras| {
        let filtered = extras
            .iter()
            .filter(|(key, _)| !gateway_only_extra_keys().contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<serde_json::Map<_, _>>();
        let filtered = strip_openai_cache_keys(&filtered);
        if filtered.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(filtered))
        }
    });

    Ok(CanonicalRequest {
        model: req.model.clone(),
        messages,
        tools,
        tool_choice,
        max_tokens: req.max_output_tokens,
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

fn responses_instructions_to_canonical(
    instructions: &ResponsesInstructions,
) -> Result<CanonicalContent, String> {
    match instructions {
        ResponsesInstructions::Text(text) => Ok(CanonicalContent::Text(text.clone())),
        ResponsesInstructions::Parts(parts) => {
            for part in parts {
                if parse_prompt_cache_breakpoint(part.prompt_cache_breakpoint.as_ref())?.is_some() {
                    return Err(
                        "prompt_cache_breakpoint is not allowed on top-level instructions".into(),
                    );
                }
            }
            let mut fragments = Vec::new();
            for part in parts {
                match part.kind.as_str() {
                    "input_text" | "output_text" => {
                        fragments.push(part.text.clone().unwrap_or_default());
                    }
                    other => {
                        return Err(format!("unsupported instructions part type: {other}"));
                    }
                }
            }
            Ok(CanonicalContent::Text(fragments.join("\n")))
        }
    }
}

fn responses_content_to_canonical(
    content: Option<&ResponsesInputContent>,
) -> Result<CanonicalContent, String> {
    match content {
        Some(ResponsesInputContent::Text(text)) => Ok(CanonicalContent::Text(text.clone())),
        Some(ResponsesInputContent::Parts(parts)) => {
            let mut text_fragments = Vec::new();
            let mut blocks = Vec::with_capacity(parts.len());
            let mut has_media = false;
            let mut has_mark = false;
            for part in parts {
                let cache = parse_prompt_cache_breakpoint(part.prompt_cache_breakpoint.as_ref())?;
                if cache.is_some() {
                    has_mark = true;
                }
                match part.kind.as_str() {
                    "input_text" | "output_text" => {
                        let text = part.text.clone().unwrap_or_default();
                        text_fragments.push(text.clone());
                        blocks.push(CanonicalBlock::Text { text, cache });
                    }
                    "input_image" => {
                        let image_url = part.image_url.as_deref().ok_or_else(|| {
                            "input_image content part missing image_url".to_string()
                        })?;
                        blocks.push(CanonicalBlock::Image {
                            source: CanonicalImageSource::from_image_url(image_url)?,
                            cache,
                        });
                        has_media = true;
                    }
                    "input_file" => {
                        let source = CanonicalFileSource::from_parts(
                            part.file_id.as_deref(),
                            part.file_url.as_deref(),
                            part.file_data.as_deref(),
                        )?;
                        if part
                            .filename
                            .as_ref()
                            .is_some_and(|name| name.trim().is_empty())
                        {
                            return Err("filename must not be empty".into());
                        }
                        if let Some(detail) = part.detail.as_deref() {
                            if !matches!(detail, "auto" | "low" | "high") {
                                return Err(format!("unsupported input_file detail: {detail}"));
                            }
                        }
                        blocks.push(CanonicalBlock::File {
                            source,
                            filename: part.filename.clone(),
                            detail: part.detail.clone(),
                            cache,
                        });
                        has_media = true;
                    }
                    other => return Err(format!("unsupported content part type: {other}")),
                }
            }
            if has_media || has_mark {
                Ok(CanonicalContent::Blocks(blocks))
            } else {
                Ok(CanonicalContent::Text(text_fragments.join("\n")))
            }
        }
        None => Ok(CanonicalContent::Text(String::new())),
    }
}

/// Convert a `CanonicalResponse` into the Responses envelope.
///
/// Contract (pinned by tests): one assistant Message item (output_text part)
/// when content is non-empty, then one FunctionCall item per canonical tool
/// call (call_id = canonical id, ids prefixed "msg_"/"fc_"); finish_reason
/// "length" -> incomplete/max_output_tokens, "content_filter" ->
/// incomplete/content_filter, everything else -> completed; usage totals filled.
pub fn responses_from_canonical(
    canon: CanonicalResponse,
    requested_model: String,
    response_id: String,
    created_at: u64,
) -> ResponsesResponse {
    let response_id = canon.id.clone().unwrap_or(response_id);
    let metadata = canon.metadata.clone();
    let provider_metadata = provider_metadata_json(&canon, false);
    let (status, incomplete_details, error) =
        if responses_error_reason(canon.finish_reason.as_deref()).is_some() {
            (
                "failed".to_string(),
                None,
                Some(ResponsesError {
                    message: "canonical response error".into(),
                    kind: "server_error",
                    code: None,
                }),
            )
        } else if let Some(reason) = responses_incomplete_reason(canon.finish_reason.as_deref()) {
            (
                "incomplete".to_string(),
                Some(IncompleteDetails {
                    reason: reason.into(),
                }),
                None,
            )
        } else {
            ("completed".to_string(), None, None)
        };

    let mut output: Vec<ResponsesOutputItem> = Vec::new();

    // A message item is emitted only when there is assistant text; a tool-only
    // turn carries no empty message (mirrors Chat's null-content contract).
    if !canon.content.is_empty() || canon.refusal.is_some() {
        let mut content = Vec::new();
        if !canon.content.is_empty() {
            content.push(ResponsesOutputContent {
                kind: "output_text",
                text: Some(canon.content),
                refusal: None,
                annotations: canon.annotations.clone(),
            });
        }
        if let Some(refusal) = canon.refusal {
            content.push(ResponsesOutputContent {
                kind: "refusal",
                text: None,
                refusal: Some(refusal),
                annotations: Vec::new(),
            });
        }
        output.push(ResponsesOutputItem::Message {
            id: format!("msg_{response_id}"),
            status: status.clone(),
            role: "assistant",
            content,
        });
    }

    for tc in canon.tool_calls {
        output.push(ResponsesOutputItem::FunctionCall {
            id: format!("fc_{}", tc.id),
            call_id: tc.id,
            name: tc.name,
            arguments: tc.arguments,
            status: status.clone(),
        });
    }

    let total = canon.usage.input_tokens + canon.usage.output_tokens;
    let usage = responses_usage_from_canonical(&canon.usage, total);

    ResponsesResponse {
        id: response_id,
        object: "response",
        created_at,
        status,
        model: requested_model,
        output,
        incomplete_details,
        error,
        usage,
        service_tier: metadata
            .as_ref()
            .and_then(|metadata| metadata.service_tier.clone()),
        system_fingerprint: metadata
            .as_ref()
            .and_then(|metadata| metadata.system_fingerprint.clone()),
        provider_metadata,
    }
}

fn responses_usage_from_canonical(usage: &omni_core::CanonicalUsage, total: u64) -> ResponsesUsage {
    let has_split_audio = usage.input_audio_tokens != 0 || usage.output_audio_tokens != 0;
    let input_audio_tokens = if has_split_audio {
        usage.input_audio_tokens
    } else {
        usage.audio_tokens
    };
    ResponsesUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: total,
        input_tokens_details: usage_detail_json(&[
            ("cached_tokens", usage.cache_read),
            ("audio_tokens", input_audio_tokens),
            ("image_tokens", usage.image_tokens),
        ]),
        output_tokens_details: usage_detail_json(&[
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
        ]),
    }
}

/// Frame a canonical event stream as Responses SSE.
///
/// Contract (pinned by tests): every SSE event has an `event:` name matching
/// the JSON `type` and a strictly-increasing `sequence_number`. Sequence:
/// `response.created` first; each output item gets one stable `output_index`
/// when first seen; text items are announced before their first text delta;
/// tool calls open with `response.output_item.added` (function_call, carrying
/// the name) followed by `response.function_call_arguments.delta` events;
/// terminal event is `response.completed` (carrying the aggregated output +
/// usage + status "completed"), or `response.incomplete` on a "length" finish,
/// or `response.failed` on a stream error. NO `data: [DONE]` sentinel.
pub fn sse_from_canonical_stream_responses(
    stream: CanonicalStream,
    requested_model: String,
    response_id: String,
    created_at: u64,
) -> Sse<ResponsesSseStream> {
    // Keep the caller's request span entered while this body is polled, so the
    // mid-stream error logs in `responses_sse_events` retain the request's
    // correlation fields (the inner stream's own poll-span does not cover logs
    // this outer generator emits). See omni_common::span_stream::SpannedStream.
    let body = responses_sse_events(stream, requested_model, response_id, created_at);
    let spanned = crate::span_stream::SpannedStream::current(Box::pin(body));
    Sse::new(Box::pin(spanned) as ResponsesSseStream)
}

/// One SSE frame: `event:`-named with the JSON `type` and a stamped
/// `sequence_number`. Centralizes the naming so the event name and the payload
/// `type` can never drift apart (SDK clients match on both).
fn responses_event(seq: &mut u64, name: &str, mut payload: serde_json::Value) -> Event {
    payload["type"] = serde_json::Value::String(name.to_string());
    payload["sequence_number"] = serde_json::json!(*seq);
    *seq += 1;
    Event::default().event(name).data(payload.to_string())
}

/// Stream-wide constants shared by every Responses SSE event (the response id,
/// the creation timestamp, and the echoed model). Built once per stream and
/// threaded by reference so the envelope/event helpers stay small.
#[derive(Clone)]
struct StreamMeta {
    response_id: String,
    created_at: u64,
    model: String,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
    provider_metadata: Option<serde_json::Value>,
}

fn apply_stream_metadata(meta: &mut StreamMeta, metadata: CanonicalResponseMetadata) {
    if let Some(id) = metadata.id {
        meta.response_id = id;
    }
    if metadata.system_fingerprint.is_some() {
        meta.system_fingerprint = metadata.system_fingerprint;
    }
    if metadata.service_tier.is_some() {
        meta.service_tier = metadata.service_tier;
    }
    let mut provider_metadata = meta
        .provider_metadata
        .take()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    if let Some(provider) = metadata.provider {
        provider_metadata.insert("provider".into(), serde_json::Value::String(provider));
    }
    if let Some(raw) = metadata.raw {
        provider_metadata.insert("raw".into(), raw);
    }
    if provider_metadata.is_empty() {
        meta.provider_metadata = None;
    } else {
        meta.provider_metadata = Some(serde_json::Value::Object(provider_metadata));
    }
}

enum ResponsesStreamOutputItem {
    Message,
    Tool(u32),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResponsesContentChannel {
    Text,
    Refusal,
}

struct ResponsesStreamToolCall {
    canonical_index: u32,
    output_index: u32,
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
    emitted_open: bool,
    emitted_arguments_len: usize,
}

struct ResponsesStreamSnapshot<'a> {
    text: &'a str,
    refusal: &'a str,
    message_content_order: &'a [ResponsesContentChannel],
    tool_calls: &'a [ResponsesStreamToolCall],
    output_order: &'a [ResponsesStreamOutputItem],
    usage: &'a omni_core::CanonicalUsage,
    annotations: &'a [serde_json::Value],
}

/// Build the terminal response envelope embedded in `response.completed` /
/// `response.incomplete` / `response.failed`, carrying the aggregated output
/// and usage so a streaming client ends with the same shape as non-streaming.
fn responses_stream_envelope(
    meta: &StreamMeta,
    status: &str,
    snapshot: &ResponsesStreamSnapshot<'_>,
    incomplete_reason: Option<&str>,
) -> serde_json::Value {
    let mut output: Vec<serde_json::Value> = Vec::new();
    for item in snapshot.output_order {
        match item {
            ResponsesStreamOutputItem::Message
                if !snapshot.text.is_empty() || !snapshot.refusal.is_empty() =>
            {
                let content = responses_message_content_json(
                    snapshot.text,
                    snapshot.refusal,
                    snapshot.message_content_order,
                    snapshot.annotations,
                );
                output.push(serde_json::json!({
                    "type": "message",
                    "id": format!("msg_{}", meta.response_id),
                    "status": status,
                    "role": "assistant",
                    "content": content,
                }));
            }
            ResponsesStreamOutputItem::Tool(canonical_index) => {
                if let Some(call) = snapshot
                    .tool_calls
                    .iter()
                    .find(|call| call.canonical_index == *canonical_index)
                    && let (Some(call_id), Some(name)) = (&call.call_id, &call.name)
                {
                    output.push(serde_json::json!({
                        "type": "function_call",
                        "id": format!("fc_{call_id}"),
                        "call_id": call_id,
                        "name": name,
                        "arguments": call.arguments,
                        "status": status,
                    }));
                }
            }
            ResponsesStreamOutputItem::Message => {}
        }
    }
    let total = snapshot.usage.input_tokens + snapshot.usage.output_tokens;
    let has_split_audio =
        snapshot.usage.input_audio_tokens != 0 || snapshot.usage.output_audio_tokens != 0;
    let input_audio_tokens = if has_split_audio {
        snapshot.usage.input_audio_tokens
    } else {
        snapshot.usage.audio_tokens
    };
    let mut usage = serde_json::json!({
        "input_tokens": snapshot.usage.input_tokens,
        "output_tokens": snapshot.usage.output_tokens,
        "total_tokens": total,
    });
    if let Some(details) = usage_detail_json(&[
        ("cached_tokens", snapshot.usage.cache_read),
        ("audio_tokens", input_audio_tokens),
        ("image_tokens", snapshot.usage.image_tokens),
    ]) {
        usage["input_tokens_details"] = details;
    }
    if let Some(details) = usage_detail_json(&[
        ("reasoning_tokens", snapshot.usage.reasoning_tokens),
        ("audio_tokens", snapshot.usage.output_audio_tokens),
        (
            "accepted_prediction_tokens",
            snapshot.usage.accepted_prediction_tokens,
        ),
        (
            "rejected_prediction_tokens",
            snapshot.usage.rejected_prediction_tokens,
        ),
    ]) {
        usage["output_tokens_details"] = details;
    }
    let mut envelope = serde_json::json!({
        "id": meta.response_id,
        "object": "response",
        "created_at": meta.created_at,
        "status": status,
        "model": meta.model,
        "output": output,
        "usage": usage,
    });
    if let Some(reason) = incomplete_reason {
        envelope["incomplete_details"] = serde_json::json!({ "reason": reason });
    }
    if let Some(system_fingerprint) = &meta.system_fingerprint {
        envelope["system_fingerprint"] = serde_json::json!(system_fingerprint);
    }
    if let Some(service_tier) = &meta.service_tier {
        envelope["service_tier"] = serde_json::json!(service_tier);
    }
    if let Some(provider_metadata) = &meta.provider_metadata {
        envelope["provider_metadata"] = provider_metadata.clone();
    }
    envelope
}

fn responses_sse_events(
    mut stream: CanonicalStream,
    requested_model: String,
    response_id: String,
    created_at: u64,
) -> impl Stream<Item = Result<Event, Infallible>> {
    use futures_util::StreamExt;
    let mut meta = StreamMeta {
        response_id,
        created_at,
        model: requested_model,
        system_fingerprint: None,
        service_tier: None,
        provider_metadata: None,
    };
    async_stream::stream! {
        let mut seq: u64 = 0;
        let mut pending_item = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(CanonicalStreamEvent::ResponseMetadata(metadata)) => {
                    apply_stream_metadata(&mut meta, metadata);
                }
                other => {
                    pending_item = Some(other);
                    break;
                }
            }
        }

        // response.created opens the stream (status in_progress, empty output).
        let mut created_env = serde_json::json!({
            "id": meta.response_id,
            "object": "response",
            "created_at": meta.created_at,
            "status": "in_progress",
            "model": meta.model,
            "output": [],
        });
        if let Some(system_fingerprint) = &meta.system_fingerprint {
            created_env["system_fingerprint"] = serde_json::json!(system_fingerprint);
        }
        if let Some(service_tier) = &meta.service_tier {
            created_env["service_tier"] = serde_json::json!(service_tier);
        }
        if let Some(provider_metadata) = &meta.provider_metadata {
            created_env["provider_metadata"] = provider_metadata.clone();
        }
        yield Ok(responses_event(&mut seq, "response.created", serde_json::json!({
            "response": created_env,
        })));

        // Aggregated state for the terminal envelope.
        let mut text = String::new();
        let mut refusal = String::new();
        let mut next_output_index: u32 = 0;
        let mut message_output_index: Option<u32> = None;
        let mut message_content_order: Vec<ResponsesContentChannel> = Vec::new();
        let mut output_order: Vec<ResponsesStreamOutputItem> = Vec::new();
        // Tool calls in arrival order, indexed by their canonical stream index.
        let mut tool_calls: Vec<ResponsesStreamToolCall> = Vec::new();
        let mut usage = omni_core::CanonicalUsage::default();
        let mut annotations: Vec<serde_json::Value> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut error_message: Option<String> = None;
        let mut saw_finish = false;

        loop {
            let item = if let Some(item) = pending_item.take() {
                item
            } else if let Some(item) = stream.next().await {
                item
            } else {
                break;
            };
            match item {
                Ok(CanonicalStreamEvent::ResponseMetadata(metadata)) => {
                    apply_stream_metadata(&mut meta, metadata);
                }
                Ok(CanonicalStreamEvent::TextDelta(delta)) => {
                    if delta.is_empty() {
                        continue;
                    }
                    let output_index = if let Some(output_index) = message_output_index {
                        output_index
                    } else {
                        let output_index = next_output_index;
                        next_output_index += 1;
                        message_output_index = Some(output_index);
                        output_order.push(ResponsesStreamOutputItem::Message);
                        let item = serde_json::json!({
                            "type": "message",
                            "id": format!("msg_{}", meta.response_id),
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        });
                        yield Ok(responses_event(&mut seq, "response.output_item.added", serde_json::json!({
                            "output_index": output_index,
                            "item": item,
                        })));
                        output_index
                    };
                    let (content_index, content_added) = content_index_for(
                        &mut message_content_order,
                        ResponsesContentChannel::Text,
                    );
                    if content_added {
                        yield Ok(responses_event(&mut seq, "response.content_part.added", serde_json::json!({
                            "output_index": output_index,
                            "content_index": content_index,
                            "item_id": format!("msg_{}", meta.response_id),
                            "part": responses_content_part_json(ResponsesContentChannel::Text, "", &[]),
                        })));
                    }
                    text.push_str(&delta);
                    yield Ok(responses_event(&mut seq, "response.output_text.delta", serde_json::json!({
                        "output_index": output_index,
                        "content_index": content_index,
                        "item_id": format!("msg_{}", meta.response_id),
                        "delta": delta,
                    })));
                }
                Ok(CanonicalStreamEvent::RefusalDelta(delta)) => {
                    if delta.is_empty() {
                        continue;
                    }
                    let output_index = if let Some(output_index) = message_output_index {
                        output_index
                    } else {
                        let output_index = next_output_index;
                        next_output_index += 1;
                        message_output_index = Some(output_index);
                        output_order.push(ResponsesStreamOutputItem::Message);
                        let item = serde_json::json!({
                            "type": "message",
                            "id": format!("msg_{}", meta.response_id),
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        });
                        yield Ok(responses_event(&mut seq, "response.output_item.added", serde_json::json!({
                            "output_index": output_index,
                            "item": item,
                        })));
                        output_index
                    };
                    let (content_index, content_added) = content_index_for(
                        &mut message_content_order,
                        ResponsesContentChannel::Refusal,
                    );
                    if content_added {
                        yield Ok(responses_event(&mut seq, "response.content_part.added", serde_json::json!({
                            "output_index": output_index,
                            "content_index": content_index,
                            "item_id": format!("msg_{}", meta.response_id),
                            "part": responses_content_part_json(ResponsesContentChannel::Refusal, "", &[]),
                        })));
                    }
                    refusal.push_str(&delta);
                    yield Ok(responses_event(&mut seq, "response.refusal.delta", serde_json::json!({
                        "output_index": output_index,
                        "content_index": content_index,
                        "item_id": format!("msg_{}", meta.response_id),
                        "delta": delta,
                    })));
                }
                Ok(CanonicalStreamEvent::ReasoningDelta(_)) |
                Ok(CanonicalStreamEvent::ReasoningSignatureDelta(_)) => {
                    // Responses SSE has model-specific reasoning events; Omni
                    // does not synthesize them from provider-specific deltas yet.
                }
                Ok(CanonicalStreamEvent::OutputAnnotations(new_annotations)) => {
                    annotations.extend(new_annotations);
                }
                Ok(CanonicalStreamEvent::ToolCallDelta { index, id, name, arguments_delta }) => {
                    let call_pos = if let Some(pos) = tool_calls
                        .iter_mut()
                        .position(|call| call.canonical_index == index)
                    {
                        let slot = &mut tool_calls[pos];
                        if slot.call_id.is_none() && let Some(id) = id {
                            slot.call_id = Some(id);
                        }
                        if slot.name.is_none() && let Some(name) = name {
                            slot.name = Some(name);
                        }
                        pos
                    } else {
                        let output_index = next_output_index;
                        next_output_index += 1;
                        tool_calls.push(ResponsesStreamToolCall {
                            canonical_index: index,
                            output_index,
                            call_id: id,
                            name,
                            arguments: String::new(),
                            emitted_open: false,
                            emitted_arguments_len: 0,
                        });
                        output_order.push(ResponsesStreamOutputItem::Tool(index));
                        tool_calls.len() - 1
                    };
                    if !arguments_delta.is_empty() {
                        tool_calls[call_pos].arguments.push_str(&arguments_delta);
                    }
                    if !tool_calls[call_pos].emitted_open
                        && let (Some(call_id), Some(call_name)) = (
                            tool_calls[call_pos].call_id.clone(),
                            tool_calls[call_pos].name.clone(),
                        )
                    {
                        tool_calls[call_pos].emitted_open = true;
                        let output_index = tool_calls[call_pos].output_index;
                        let item = serde_json::json!({
                            "type": "function_call",
                            "id": format!("fc_{call_id}"),
                            "call_id": call_id,
                            "name": call_name,
                            "arguments": "",
                            "status": "in_progress",
                        });
                        yield Ok(responses_event(&mut seq, "response.output_item.added", serde_json::json!({
                            "output_index": output_index,
                            "item": item,
                        })));
                    }
                    let slot = &mut tool_calls[call_pos];
                    if slot.emitted_open && slot.emitted_arguments_len < slot.arguments.len() {
                        let delta = slot.arguments[slot.emitted_arguments_len..].to_string();
                        slot.emitted_arguments_len = slot.arguments.len();
                        let item_id = format!("fc_{}", slot.call_id.as_deref().unwrap_or("call_unknown"));
                        yield Ok(responses_event(&mut seq, "response.function_call_arguments.delta", serde_json::json!({
                            "output_index": slot.output_index,
                            "item_id": item_id,
                            "delta": delta,
                        })));
                    }
                }
                Ok(CanonicalStreamEvent::Usage(u)) => {
                    usage = u;
                }
                Ok(CanonicalStreamEvent::Finish { finish_reason: fr }) => {
                    finish_reason = fr;
                    saw_finish = true;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "canonical stream error mid-flight (responses)");
                    error_message = Some(e.to_string());
                    break;
                }
            }
        }

        if !saw_finish && error_message.is_none() {
            error_message = Some("canonical stream ended before Finish event".into());
        }

        let snapshot = ResponsesStreamSnapshot {
            text: &text,
            refusal: &refusal,
            message_content_order: &message_content_order,
            tool_calls: &tool_calls,
            output_order: &output_order,
            usage: &usage,
            annotations: &annotations,
        };
        if let Some(message) = error_message {
            tracing::warn!(error = %message, "canonical stream error mid-flight (responses)");
            if let Some(output_index) = message_output_index {
                for event in emit_responses_text_done_events(
                    &mut seq,
                    &meta,
                    output_index,
                    &text,
                    &refusal,
                    &message_content_order,
                    &annotations,
                ) {
                    yield Ok(event);
                }
            }
            for event in emit_responses_tool_done_events(&mut seq, &tool_calls) {
                yield Ok(event);
            }
            yield Ok(emit_responses_failed_event(&mut seq, &meta, &snapshot));
        } else if responses_error_reason(finish_reason.as_deref()).is_some() {
            if let Some(output_index) = message_output_index {
                for event in emit_responses_text_done_events(
                    &mut seq,
                    &meta,
                    output_index,
                    &text,
                    &refusal,
                    &message_content_order,
                    &annotations,
                ) {
                    yield Ok(event);
                }
            }
            for event in emit_responses_tool_done_events(&mut seq, &tool_calls) {
                yield Ok(event);
            }
            yield Ok(emit_responses_failed_event(&mut seq, &meta, &snapshot));
        } else if let Some(reason) = responses_incomplete_reason(finish_reason.as_deref()) {
            if let Some(output_index) = message_output_index {
                for event in emit_responses_text_done_events(
                    &mut seq,
                    &meta,
                    output_index,
                    &text,
                    &refusal,
                    &message_content_order,
                    &annotations,
                ) {
                    yield Ok(event);
                }
            }
            for event in emit_responses_tool_done_events(&mut seq, &tool_calls) {
                yield Ok(event);
            }
            let env = responses_stream_envelope(&meta, "incomplete", &snapshot, Some(reason));
            yield Ok(responses_event(
                &mut seq,
                "response.incomplete",
                serde_json::json!({
                    "response": env,
                }),
            ));
        } else {
            if let Some(output_index) = message_output_index {
                for event in emit_responses_text_done_events(
                    &mut seq,
                    &meta,
                    output_index,
                    &text,
                    &refusal,
                    &message_content_order,
                    &annotations,
                ) {
                    yield Ok(event);
                }
            }
            for event in emit_responses_tool_done_events(&mut seq, &tool_calls) {
                yield Ok(event);
            }
            let env = responses_stream_envelope(&meta, "completed", &snapshot, None);
            yield Ok(responses_event(
                &mut seq,
                "response.completed",
                serde_json::json!({
                    "response": env,
                }),
            ));
        }
        // NB: no [DONE] sentinel - that is Chat Completions framing only.
    }
}

fn emit_responses_text_done_events(
    seq: &mut u64,
    meta: &StreamMeta,
    output_index: u32,
    text: &str,
    refusal: &str,
    message_content_order: &[ResponsesContentChannel],
    annotations: &[serde_json::Value],
) -> Vec<Event> {
    let mut events = Vec::new();
    if !text.is_empty() {
        let content_index =
            content_index(message_content_order, ResponsesContentChannel::Text).unwrap_or(0);
        events.push(responses_event(
            seq,
            "response.output_text.done",
            serde_json::json!({
                "output_index": output_index,
                "content_index": content_index,
                "item_id": format!("msg_{}", meta.response_id),
                "text": text,
            }),
        ));
        events.push(responses_event(
            seq,
            "response.content_part.done",
            serde_json::json!({
                "output_index": output_index,
                "content_index": content_index,
                "item_id": format!("msg_{}", meta.response_id),
                "part": responses_content_part_json(ResponsesContentChannel::Text, text, annotations),
            }),
        ));
    }
    if !refusal.is_empty() {
        let content_index =
            content_index(message_content_order, ResponsesContentChannel::Refusal).unwrap_or(0);
        events.push(responses_event(
            seq,
            "response.refusal.done",
            serde_json::json!({
                "output_index": output_index,
                "content_index": content_index,
                "item_id": format!("msg_{}", meta.response_id),
                "refusal": refusal,
            }),
        ));
        events.push(responses_event(
            seq,
            "response.content_part.done",
            serde_json::json!({
                "output_index": output_index,
                "content_index": content_index,
                "item_id": format!("msg_{}", meta.response_id),
                "part": responses_content_part_json(ResponsesContentChannel::Refusal, refusal, &[]),
            }),
        ));
    }
    if !text.is_empty() || !refusal.is_empty() {
        events.push(responses_event(
            seq,
            "response.output_item.done",
            serde_json::json!({
                "output_index": output_index,
                "item": {
                    "type": "message",
                    "id": format!("msg_{}", meta.response_id),
                    "status": "completed",
                    "role": "assistant",
                    "content": responses_message_content_json(text, refusal, message_content_order, annotations),
                },
            }),
        ));
    }
    events
}

fn emit_responses_tool_done_events(
    seq: &mut u64,
    tool_calls: &[ResponsesStreamToolCall],
) -> Vec<Event> {
    let mut events = Vec::new();
    for call in tool_calls.iter().filter(|call| call.emitted_open) {
        let Some(call_id) = call.call_id.as_deref() else {
            continue;
        };
        let Some(name) = call.name.as_deref() else {
            continue;
        };
        events.push(responses_event(
            seq,
            "response.function_call_arguments.done",
            serde_json::json!({
                "output_index": call.output_index,
                "item_id": format!("fc_{call_id}"),
                "arguments": call.arguments,
            }),
        ));
        events.push(responses_event(
            seq,
            "response.output_item.done",
            serde_json::json!({
                "output_index": call.output_index,
                "item": {
                    "type": "function_call",
                    "id": format!("fc_{call_id}"),
                    "call_id": call_id,
                    "name": name,
                    "arguments": call.arguments,
                    "status": "completed",
                },
            }),
        ));
    }
    events
}

fn responses_message_content_json(
    text: &str,
    refusal: &str,
    message_content_order: &[ResponsesContentChannel],
    annotations: &[serde_json::Value],
) -> Vec<serde_json::Value> {
    let mut content = Vec::new();
    let mut content_order = message_content_order.to_vec();
    if !text.is_empty() && !content_order.contains(&ResponsesContentChannel::Text) {
        content_order.push(ResponsesContentChannel::Text);
    }
    if !refusal.is_empty() && !content_order.contains(&ResponsesContentChannel::Refusal) {
        content_order.push(ResponsesContentChannel::Refusal);
    }
    for channel in content_order {
        match channel {
            ResponsesContentChannel::Text if !text.is_empty() => {
                content.push(serde_json::json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": annotations,
                }));
            }
            ResponsesContentChannel::Refusal if !refusal.is_empty() => {
                content.push(serde_json::json!({
                    "type": "refusal",
                    "refusal": refusal,
                    "annotations": [],
                }));
            }
            _ => {}
        }
    }
    content
}

fn responses_content_part_json(
    channel: ResponsesContentChannel,
    value: &str,
    annotations: &[serde_json::Value],
) -> serde_json::Value {
    match channel {
        ResponsesContentChannel::Text => serde_json::json!({
            "type": "output_text",
            "text": value,
            "annotations": annotations,
        }),
        ResponsesContentChannel::Refusal => serde_json::json!({
            "type": "refusal",
            "refusal": value,
            "annotations": [],
        }),
    }
}

fn emit_responses_failed_event(
    seq: &mut u64,
    meta: &StreamMeta,
    snapshot: &ResponsesStreamSnapshot<'_>,
) -> Event {
    let mut env = responses_stream_envelope(meta, "failed", snapshot, None);
    let error = serde_json::json!({
        "message": "canonical stream error",
        "type": "server_error",
        "code": serde_json::Value::Null,
    });
    env["error"] = error.clone();
    responses_event(
        seq,
        "response.failed",
        serde_json::json!({
            "response": env,
            "error": error,
        }),
    )
}

fn content_index_for(
    order: &mut Vec<ResponsesContentChannel>,
    channel: ResponsesContentChannel,
) -> (u32, bool) {
    if let Some(index) = content_index(order, channel) {
        return (index, false);
    }
    order.push(channel);
    ((order.len() - 1) as u32, true)
}

fn content_index(
    order: &[ResponsesContentChannel],
    channel: ResponsesContentChannel,
) -> Option<u32> {
    order
        .iter()
        .position(|existing| *existing == channel)
        .map(|index| index as u32)
}

fn responses_incomplete_reason(finish_reason: Option<&str>) -> Option<&str> {
    match finish_reason {
        Some("length") => Some("max_output_tokens"),
        Some("content_filter") => Some("content_filter"),
        _ => None,
    }
}

fn responses_error_reason(finish_reason: Option<&str>) -> Option<&str> {
    match finish_reason {
        Some(reason) if reason == "error" || reason.starts_with("error:") => Some(reason),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::MAX_REASONING_EFFORT_LEN;
    use omni_core::{
        CanonicalContent, CanonicalImageSource, CanonicalResponseMetadata, CanonicalStreamEvent,
        CanonicalToolCall, CanonicalToolChoice, CanonicalUsage, ProviderError,
    };

    // ---- helpers ----

    fn parse(json: &str) -> ResponsesRequest {
        serde_json::from_str(json).expect("request json should deserialize")
    }

    fn text_of(content: &CanonicalContent) -> &str {
        match content {
            CanonicalContent::Text(t) => t,
            CanonicalContent::Blocks(_) => panic!("unexpected blocks content"),
        }
    }

    /// Render an SSE response body to a string for wire-level assertions.
    async fn sse_body_to_string(sse: Sse<ResponsesSseStream>) -> String {
        use axum::response::IntoResponse;
        let resp = sse.into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("collect sse body");
        String::from_utf8(body.to_vec()).expect("sse body utf8")
    }

    /// Extract the JSON payloads of all `data:` lines, in order.
    fn data_payloads(body: &str) -> Vec<serde_json::Value> {
        body.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .map(|d| serde_json::from_str(d).expect("each data line is JSON"))
            .collect()
    }

    // ---- request deserialization (wire contract) ----

    #[test]
    fn request_parses_string_input() {
        // WHY: the simplest SDK call is input as a bare string; it must parse
        // with stream defaulting to false and unknown fields tolerated.
        let req = parse(r#"{"model":"m","input":"hi","store":false,"metadata":{"a":"b"}}"#);
        assert_eq!(req.model, "m");
        assert!(!req.stream);
        match &req.input {
            ResponsesInput::Text(t) => assert_eq!(t, "hi"),
            other => panic!("expected Text input, got {other:?}"),
        }
    }

    #[test]
    fn request_parses_message_items_with_string_and_part_content() {
        // WHY: SDKs send both the shorthand {role, content:"..."} and the
        // explicit {type:"message", role, content:[{type:"input_text",...}]}
        // forms; both must deserialize into message items.
        let req = parse(
            r#"{"model":"m","input":[
                {"role":"user","content":"plain"},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"prev"}]}
            ]}"#,
        );
        let items = match &req.input {
            ResponsesInput::Items(v) => v,
            other => panic!("expected Items, got {other:?}"),
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].role.as_deref(), Some("user"));
        assert!(matches!(
            items[0].content,
            Some(ResponsesInputContent::Text(_))
        ));
        assert_eq!(items[1].kind.as_deref(), Some("message"));
        match items[1].content.as_ref().expect("content") {
            ResponsesInputContent::Parts(parts) => {
                assert_eq!(parts[0].kind, "output_text");
                assert_eq!(parts[0].text.as_deref(), Some("prev"));
            }
            other => panic!("expected Parts, got {other:?}"),
        }
    }

    #[test]
    fn request_parses_flattened_tools_and_tool_choice_forms() {
        // WHY: Responses tools are FLAT function defs (name at top level), not
        // Chat Completions' nested {function:{...}} shape; tool_choice comes as
        // either a mode string or a {type:"function",name} object.
        let req = parse(
            r#"{"model":"m","input":"q",
                "tools":[{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object"}}],
                "tool_choice":{"type":"function","name":"get_weather"}}"#,
        );
        let tools = req.tools.as_ref().expect("tools parsed");
        assert_eq!(tools[0].kind, "function");
        assert_eq!(tools[0].name.as_deref(), Some("get_weather"));
        match req.tool_choice.as_ref().expect("tool_choice parsed") {
            ResponsesToolChoice::Function { kind, name } => {
                assert_eq!(kind, "function");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected Function choice, got {other:?}"),
        }

        let req2 = parse(r#"{"model":"m","input":"q","tool_choice":"auto"}"#);
        assert!(matches!(
            req2.tool_choice,
            Some(ResponsesToolChoice::Mode(ref s)) if s == "auto"
        ));
    }

    // ---- responses_to_canonical ----

    #[test]
    fn to_canonical_string_input_becomes_user_message() {
        let req = parse(r#"{"model":"grok-3","input":"hello there"}"#);
        let canon = responses_to_canonical(&req).expect("string input converts");
        assert_eq!(canon.model, "grok-3");
        assert_eq!(canon.messages.len(), 1);
        assert_eq!(canon.messages[0].role, "user");
        assert_eq!(text_of(&canon.messages[0].content), "hello there");
    }

    #[test]
    fn to_canonical_instructions_become_leading_system_message() {
        // WHY: `instructions` is the Responses equivalent of a system prompt;
        // providers consume it as a leading system-role canonical message.
        let req = parse(r#"{"model":"m","input":"q","instructions":"be terse"}"#);
        let canon = responses_to_canonical(&req).unwrap();
        assert_eq!(canon.messages.len(), 2);
        assert_eq!(canon.messages[0].role, "system");
        assert_eq!(text_of(&canon.messages[0].content), "be terse");
        assert_eq!(canon.messages[1].role, "user");
    }

    #[test]
    fn to_canonical_joins_multiple_text_parts_with_newline() {
        // WHY: canonical content is flat text; multi-part text input must be
        // joined deterministically (newline) so providers see stable prompts.
        let req = parse(
            r#"{"model":"m","input":[{"role":"user","content":[
                {"type":"input_text","text":"line one"},
                {"type":"input_text","text":"line two"}
            ]}]}"#,
        );
        let canon = responses_to_canonical(&req).unwrap();
        assert_eq!(text_of(&canon.messages[0].content), "line one\nline two");
    }

    #[test]
    fn to_canonical_preserves_input_image_parts_in_order() {
        // WHY: Responses clients send `input_image` alongside text. Canonical
        // must preserve the order so providers receive the intended prompt.
        let req = parse(
            r#"{"model":"m","input":[{"role":"user","content":[
                {"type":"input_text","text":"first"},
                {"type":"input_image","image_url":"https://example.com/a.png"},
                {"type":"input_text","text":"second"},
                {"type":"input_image","image_url":"data:image/jpeg;base64,abcd"}
            ]}]}"#,
        );
        let canon = responses_to_canonical(&req).unwrap();
        match &canon.messages[0].content {
            CanonicalContent::Blocks(blocks) => {
                assert!(matches!(&blocks[0], CanonicalBlock::Text { text, .. } if text == "first"));
                assert!(matches!(
                    &blocks[1],
                    CanonicalBlock::Image {
                        source: CanonicalImageSource::Url { url }, .. } if url == "https://example.com/a.png"
                ));
                assert!(
                    matches!(&blocks[2], CanonicalBlock::Text { text, .. } if text == "second")
                );
                assert!(matches!(
                    &blocks[3],
                    CanonicalBlock::Image {
                        source: CanonicalImageSource::Base64 { media_type, data }, .. } if media_type == "image/jpeg" && data == "abcd"
                ));
            }
            CanonicalContent::Text(_) => panic!("image parts must produce blocks"),
        }
    }

    #[test]
    fn to_canonical_rejects_unsupported_media_parts() {
        let req = parse(
            r#"{"model":"m","input":[{"role":"user","content":[
                {"type":"input_audio","input_audio":{"data":"..."}}
            ]}]}"#,
        );
        let err = responses_to_canonical(&req).expect_err("audio remains unsupported");
        assert!(err.contains("input_audio"), "error must name part: {err}");
    }

    #[test]
    fn to_canonical_maps_sampling_and_max_output_tokens() {
        // WHY: max_output_tokens is the Responses name for the output cap; it
        // must land in canonical max_tokens or providers use wire defaults.
        let req = parse(
            r#"{"model":"m","input":"q","max_output_tokens":77,"temperature":0.25,"top_p":0.9}"#,
        );
        let canon = responses_to_canonical(&req).unwrap();
        assert_eq!(canon.max_tokens, Some(77));
        assert_eq!(canon.temperature, Some(0.25));
        assert_eq!(canon.top_p, Some(0.9));
    }

    #[test]
    fn to_canonical_maps_reasoning_effort() {
        let req = parse(r#"{"model":"m","input":"q","reasoning":{"effort":"high"}}"#);
        let canon = responses_to_canonical(&req).unwrap();
        assert_eq!(
            canon.reasoning.expect("reasoning mapped").effort.as_deref(),
            Some("high")
        );
    }

    #[test]
    fn to_canonical_accepts_free_string_effort_including_xhigh() {
        // WHY (issue #20): Responses and chat must agree on free-string effort
        // acceptance. No closed valid-values reject at the edge.
        for effort in ["xhigh", "ultra", "minimal"] {
            let req = parse(&format!(
                r#"{{"model":"m","input":"q","reasoning":{{"effort":"{effort}"}}}}"#
            ));
            let canon = responses_to_canonical(&req)
                .unwrap_or_else(|e| panic!("Responses free effort {effort:?} must accept: {e}"));
            assert_eq!(
                canon.reasoning.expect("reasoning mapped").effort.as_deref(),
                Some(effort)
            );
        }
    }

    #[test]
    fn to_canonical_rejects_lexically_invalid_effort() {
        // WHY (issue #20 / #23): lexical hygiene only, shared with chat
        // (empty, over-long, unsafe charset). No closed valid-values list.
        let empty = parse(r#"{"model":"m","input":"q","reasoning":{"effort":""}}"#);
        let err = responses_to_canonical(&empty).expect_err("empty effort must reject");
        assert!(
            err.contains("reasoning_effort") && err.contains("empty"),
            "error must name empty: {err}"
        );

        let bad_chars = parse(r#"{"model":"m","input":"q","reasoning":{"effort":"hi there"}}"#);
        let err = responses_to_canonical(&bad_chars).expect_err("unsafe charset must reject");
        assert!(
            err.contains("reasoning_effort") && err.contains("unsafe"),
            "error must name charset: {err}"
        );

        let too_long = "a".repeat(MAX_REASONING_EFFORT_LEN + 1);
        let long = parse(&format!(
            r#"{{"model":"m","input":"q","reasoning":{{"effort":"{too_long}"}}}}"#
        ));
        let err = responses_to_canonical(&long).expect_err("over-long effort must reject");
        assert!(
            err.contains("reasoning_effort") && err.contains("max"),
            "error must name length bound: {err}"
        );
    }

    #[test]
    fn to_canonical_accepts_exact_max_reasoning_effort_length() {
        // WHY (issue #29): MAX+1 rejects; exact-MAX must accept on the real
        // Responses entry path (responses_to_canonical), not a reimplemented
        // validator. Parity with chat to_canonical accept bound.
        let exact = "a".repeat(MAX_REASONING_EFFORT_LEN);
        let req = parse(&format!(
            r#"{{"model":"m","input":"q","reasoning":{{"effort":"{exact}"}}}}"#
        ));
        let canon = responses_to_canonical(&req).unwrap_or_else(|e| {
            panic!("exact-MAX length effort must accept on Responses path: {e}")
        });
        assert_eq!(
            canon.reasoning.expect("reasoning mapped").effort.as_deref(),
            Some(exact.as_str())
        );
    }

    #[test]
    fn to_canonical_preserves_top_level_extras_for_provider() {
        // WHY: Responses-native Codex features such as previous_response_id are
        // top-level fields with no canonical equivalent. The inbound adapter
        // must preserve them so each provider can forward its supported subset.
        // `user` is gateway/session metadata, not provider passthrough.
        let req = parse(
            r#"{"model":"m","input":"q","previous_response_id":"resp_prev","store":false,
                "metadata":{"trace":"abc"},"parallel_tool_calls":true,
                "service_tier":"priority","user":"session-user"}"#,
        );
        let canon = responses_to_canonical(&req).unwrap();
        let extras = canon
            .provider_extras
            .expect("responses extras should survive conversion");
        assert_eq!(extras["previous_response_id"], "resp_prev");
        assert_eq!(extras["store"], false);
        assert_eq!(extras["metadata"]["trace"], "abc");
        assert_eq!(extras["parallel_tool_calls"], true);
        assert_eq!(extras["service_tier"], "priority");
        assert!(extras.get("user").is_none());
    }

    #[test]
    fn to_canonical_lifts_prompt_cache_key_out_of_extras() {
        // WHY: prompt_cache_key is the OpenAI/xAI cache-routing identity. If it
        // stays in extras, Codex/Grok allowlists 400 it and Claude rejects all
        // extras. Lift it onto CanonicalRequest instead.
        let req = parse(r#"{"model":"m","input":"q","prompt_cache_key":"sess-1","store":false}"#);
        let canon = responses_to_canonical(&req).unwrap();
        assert_eq!(canon.cache_routing_identity(), Some("sess-1"));
        let extras = canon
            .provider_extras
            .expect("unrelated extras should remain");
        assert_eq!(extras["store"], false);
        assert!(
            extras.get("prompt_cache_key").is_none(),
            "prompt_cache_key must not become a provider extra: {extras}"
        );
    }

    #[test]
    fn responses_instructions_breakpoint_is_400() {
        let req = parse(
            r#"{"model":"m","input":"q","instructions":[
                {"type":"input_text","text":"sys","prompt_cache_breakpoint":{"mode":"explicit"}}
            ]}"#,
        );
        let err = responses_to_canonical(&req).expect_err("instructions breakpoint must 400");
        assert!(
            err.contains("instructions"),
            "error must name instructions: {err}"
        );
    }

    #[test]
    fn responses_input_breakpoint_stays_on_blocks() {
        let req = parse(
            r#"{"model":"m","input":[{"role":"user","content":[
                {"type":"input_text","text":"prefix","prompt_cache_breakpoint":{"mode":"explicit"}},
                {"type":"input_text","text":"suffix"}
            ]}]}"#,
        );
        let canon = responses_to_canonical(&req).unwrap();
        match &canon.messages[0].content {
            CanonicalContent::Blocks(blocks) => {
                assert!(blocks[0].cache_mark().is_some());
                assert!(blocks[1].cache_mark().is_none());
            }
            other => panic!("marked input_text must not collapse: {other:?}"),
        }
    }

    #[test]
    fn to_canonical_rejects_non_string_prompt_cache_key() {
        let req = parse(r#"{"model":"m","input":"q","prompt_cache_key":1}"#);
        let err = responses_to_canonical(&req).expect_err("non-string key must 400");
        assert!(
            err.contains("prompt_cache_key must be a string"),
            "error must name the field: {err}"
        );
    }

    #[test]
    fn to_canonical_maps_function_tools_and_specific_choice() {
        let req = parse(
            r#"{"model":"m","input":"q",
                "tools":[{"type":"function","name":"add","description":"adds","parameters":{"type":"object","properties":{"a":{"type":"number"}}}}],
                "tool_choice":{"type":"function","name":"add"}}"#,
        );
        let canon = responses_to_canonical(&req).unwrap();
        let tools = canon.tools.expect("tools mapped");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "add");
        assert_eq!(tools[0].description.as_deref(), Some("adds"));
        assert_eq!(tools[0].parameters["type"], "object");
        assert!(matches!(
            canon.tool_choice,
            Some(CanonicalToolChoice::Specific { ref name }) if name == "add"
        ));
    }

    #[test]
    fn to_canonical_tool_choice_modes_map_including_none_keeps_tools_visible() {
        // WHY: OpenAI `tool_choice:"none"` means "tools stay visible to the
        // model, but it must not call any of them" -- NOT "remove the tools".
        // Dropping the tool schemas would lose visibility/token accounting and
        // diverge from the spec, so "none" maps to CanonicalToolChoice::None
        // while the tool list is preserved.
        let auto = parse(
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"f"}],"tool_choice":"auto"}"#,
        );
        assert!(matches!(
            responses_to_canonical(&auto).unwrap().tool_choice,
            Some(CanonicalToolChoice::Auto)
        ));

        let required = parse(
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"f"}],"tool_choice":"required"}"#,
        );
        assert!(matches!(
            responses_to_canonical(&required).unwrap().tool_choice,
            Some(CanonicalToolChoice::Required)
        ));

        let none = parse(
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"f"}],"tool_choice":"none"}"#,
        );
        let canon = responses_to_canonical(&none).unwrap();
        assert!(
            canon.tools.is_some(),
            "tool_choice none must keep the tools visible to the model"
        );
        assert!(
            matches!(canon.tool_choice, Some(CanonicalToolChoice::None)),
            "tool_choice none must map to CanonicalToolChoice::None, got {:?}",
            canon.tool_choice
        );
    }

    #[test]
    fn to_canonical_filters_responses_allowed_tools_for_both_modes() {
        for (mode, required) in [("auto", false), ("required", true)] {
            let req = parse(&format!(
                r#"{{"model":"m","input":"q","tools":[
                    {{"type":"function","name":"read"}},
                    {{"type":"function","name":"write"}},
                    {{"type":"function","name":"inspect"}}
                ],"tool_choice":{{"type":"allowed_tools","mode":"{mode}","tools":[
                    {{"type":"function","name":"read"}},
                    {{"type":"function","name":"inspect"}}
                ]}}}}"#,
            ));
            let canon = responses_to_canonical(&req).expect("allowed tools must convert");
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
    fn responses_allowed_tools_rejects_invalid_or_unsafe_subsets() {
        let undeclared = parse(
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"read"}],
                "tool_choice":{"type":"allowed_tools","mode":"auto","tools":[
                    {"type":"function","name":"write"}
                ]}}"#,
        );
        let err = responses_to_canonical(&undeclared).expect_err("undeclared tool must reject");
        assert!(err.contains("write"), "error must name the tool: {err}");

        let bad_mode = parse(
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"read"}],
                "tool_choice":{"type":"allowed_tools","mode":"sometimes","tools":[
                    {"type":"function","name":"read"}
                ]}}"#,
        );
        let err = responses_to_canonical(&bad_mode).expect_err("bad mode must reject");
        assert!(err.contains("sometimes"), "error must name the mode: {err}");

        let empty_required = parse(
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"read"}],
                "tool_choice":{"type":"allowed_tools","mode":"required","tools":[]}}"#,
        );
        let err =
            responses_to_canonical(&empty_required).expect_err("empty required subset must reject");
        assert!(err.contains("at least one tool"), "unexpected error: {err}");

        let empty_auto = parse(
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"read"}],
                "tool_choice":{"type":"allowed_tools","mode":"auto","tools":[]}}"#,
        );
        let canon =
            responses_to_canonical(&empty_auto).expect("empty auto subset disables all tools");
        assert!(canon.tools.is_none());
        assert!(canon.tool_choice.is_none());
    }

    #[test]
    fn to_canonical_developer_role_maps_to_system() {
        // WHY: Responses uses "developer" where older surfaces use "system";
        // providers only understand the system convention.
        let req = parse(r#"{"model":"m","input":[{"role":"developer","content":"rule"}]}"#);
        let canon = responses_to_canonical(&req).unwrap();
        assert_eq!(canon.messages[0].role, "system");
    }

    #[test]
    fn to_canonical_maps_tool_conversation_items_to_blocks() {
        // WHY: a multi-turn tool conversation (the assistant's function_call
        // followed by the caller's function_call_output) must survive into the
        // canonical layer as ToolUse / ToolResult blocks keyed by the same
        // call_id, so the linkage reaches the upstream intact rather than being
        // dropped or rejected. This is the behavior that replaced the old v1
        // "reject function_call_output" rule once canonical gained tool blocks.
        let req = parse(
            r#"{"model":"m","input":[
                {"type":"message","role":"user","content":"weather in SF?"},
                {"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"SF\"}"},
                {"type":"function_call_output","call_id":"c1","output":"72F"}
            ]}"#,
        );
        let canon = responses_to_canonical(&req).expect("tool items must convert");
        assert_eq!(canon.messages.len(), 3);
        // The assistant function_call -> a ToolUse block keyed by call_id.
        match &canon.messages[1].content {
            CanonicalContent::Blocks(blocks) => match &blocks[0] {
                CanonicalBlock::ToolUse {
                    id,
                    name,
                    arguments,
                    ..
                } => {
                    assert_eq!(id, "c1");
                    assert_eq!(name, "get_weather");
                    assert!(arguments.contains("SF"));
                }
                other => panic!("expected ToolUse block, got {other:?}"),
            },
            other => panic!("function_call must become Blocks, got {other:?}"),
        }
        // The function_call_output -> a ToolResult block keyed by the same id.
        match &canon.messages[2].content {
            CanonicalContent::Blocks(blocks) => match &blocks[0] {
                CanonicalBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    assert_eq!(tool_use_id, "c1");
                    assert_eq!(content, "72F");
                }
                other => panic!("expected ToolResult block, got {other:?}"),
            },
            other => panic!("function_call_output must become Blocks, got {other:?}"),
        }
    }

    #[test]
    fn responses_files_keep_order_and_cache_marks() {
        let req = parse(
            r#"{"model":"m","input":[{"role":"user","content":[
            {"type":"input_text","text":"first"},
            {"type":"input_file","file_url":"https://example.com/a.pdf","prompt_cache_breakpoint":{"mode":"explicit"}},
            {"type":"input_file","file_id":"file-1"},
            {"type":"input_file","filename":"b.pdf","file_data":"data:application/pdf;base64,abcd","detail":"high"},
            {"type":"input_text","text":"last"}
        ]}]}"#,
        );
        let canon = responses_to_canonical(&req).unwrap();
        let CanonicalContent::Blocks(blocks) = &canon.messages[0].content else {
            panic!("file must keep blocks")
        };
        assert_eq!(blocks.len(), 5);
        assert!(matches!(&blocks[0], CanonicalBlock::Text { text, .. } if text == "first"));
        assert!(
            matches!(&blocks[1], CanonicalBlock::File { source: omni_core::CanonicalFileSource::Url { file_url }, .. } if file_url == "https://example.com/a.pdf")
        );
        assert!(blocks[1].cache_mark().is_some());
        assert!(canon.has_cache_marks());
        assert!(
            matches!(&blocks[2], CanonicalBlock::File { source: omni_core::CanonicalFileSource::Id { file_id }, .. } if file_id == "file-1")
        );
        assert!(
            matches!(&blocks[3], CanonicalBlock::File { source: omni_core::CanonicalFileSource::Data { file_data }, filename: Some(filename), detail: Some(detail), .. } if file_data == "data:application/pdf;base64,abcd" && filename == "b.pdf" && detail == "high")
        );
        assert!(matches!(&blocks[4], CanonicalBlock::Text { text, .. } if text == "last"));
    }

    #[test]
    fn responses_files_reject_invalid_sources_and_detail() {
        for (part, field) in [
            (r#"{"type":"input_file"}"#, "exactly one"),
            (
                r#"{"type":"input_file","file_id":"file-1","file_url":"https://example.com/a.pdf"}"#,
                "exactly one",
            ),
            (r#"{"type":"input_file","file_id":" "}"#, "file_id"),
            (r#"{"type":"input_file","file_data":""}"#, "file_data"),
            (
                r#"{"type":"input_file","file_url":"http://example.com/a.pdf"}"#,
                "file_url",
            ),
            (
                r#"{"type":"input_file","file_id":"file-1","detail":"ultra"}"#,
                "detail",
            ),
        ] {
            let req = parse(&format!(
                r#"{{"model":"m","input":[{{"role":"user","content":[{part}]}}]}}"#
            ));
            let err = responses_to_canonical(&req).expect_err("invalid file must fail");
            assert!(err.contains(field), "{part}: {err}");
        }
    }

    #[test]
    fn to_canonical_rejects_non_function_tools() {
        let req = parse(r#"{"model":"m","input":"q","tools":[{"type":"web_search"}]}"#);
        let err = responses_to_canonical(&req).expect_err("must reject");
        assert!(
            err.contains("web_search"),
            "error must name the unsupported tool type, got: {err}"
        );
    }

    #[test]
    fn responses_rejects_non_function_tool_choice_type() {
        // WHY: a forced tool_choice must select a `function`. A non-function type
        // (e.g. "retrieval") is unsupported; coercing it to Specific would force a
        // tool the model cannot dispatch. Reject loudly, naming the type, rather
        // than silently mistranslating it. Mirrors the Chat protocol's contract.
        let req = parse(
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"f"}],
                "tool_choice":{"type":"retrieval","name":"f"}}"#,
        );
        let err = responses_to_canonical(&req).expect_err("non-function tool_choice must reject");
        assert!(
            err.contains("retrieval") || err.contains("type"),
            "error must name the bad tool_choice type, got: {err}"
        );
    }

    #[test]
    fn to_canonical_rejects_message_item_without_role() {
        let req = parse(r#"{"model":"m","input":[{"content":"orphan"}]}"#);
        let err = responses_to_canonical(&req).expect_err("must reject");
        assert!(
            err.to_lowercase().contains("role"),
            "error must mention the missing role, got: {err}"
        );
    }

    // ---- responses_from_canonical ----

    fn canon_resp(content: &str, tool_calls: Vec<CanonicalToolCall>) -> CanonicalResponse {
        CanonicalResponse {
            model: "backend-model".into(),
            content: content.into(),
            tool_calls,
            finish_reason: Some("stop".into()),
            usage: CanonicalUsage {
                input_tokens: 11,
                output_tokens: 7,
                ..Default::default()
            },
            id: None,
            refusal: None,
            ..Default::default()
        }
    }

    #[test]
    fn from_canonical_text_only_shape() {
        // WHY: pins the exact Responses envelope clients parse: object tag,
        // created_at naming, a message output item with an output_text part
        // (empty annotations), completed status, and input/output usage naming.
        let resp = responses_from_canonical(
            canon_resp("hello", vec![]),
            "grok:grok-3".into(),
            "resp_test1".into(),
            1234,
        );
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["id"], "resp_test1");
        assert_eq!(v["object"], "response");
        assert_eq!(v["created_at"], 1234);
        assert_eq!(v["status"], "completed");
        assert_eq!(v["model"], "grok:grok-3");
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][0]["role"], "assistant");
        assert_eq!(v["output"][0]["status"], "completed");
        assert_eq!(v["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(v["output"][0]["content"][0]["text"], "hello");
        assert!(
            v["output"][0]["content"][0]["annotations"]
                .as_array()
                .expect("annotations array")
                .is_empty()
        );
        assert!(
            v["output"][0]["id"]
                .as_str()
                .expect("message item id")
                .starts_with("msg_")
        );
        assert_eq!(v["usage"]["input_tokens"], 11);
        assert_eq!(v["usage"]["output_tokens"], 7);
        assert_eq!(v["usage"]["total_tokens"], 18);
        assert!(
            v.get("incomplete_details").is_none(),
            "completed response omits incomplete_details"
        );
    }

    #[test]
    fn from_canonical_tool_calls_become_function_call_items() {
        // WHY: function calls are first-class output items in Responses (not
        // nested in a message like Chat); call_id must round-trip the canonical
        // tool-call id so clients can pair function_call_output later.
        let tc = CanonicalToolCall {
            id: "call_abc".into(),
            name: "get_weather".into(),
            arguments: r#"{"city":"SF"}"#.into(),
        };
        let resp = responses_from_canonical(
            canon_resp("with text", vec![tc]),
            "m".into(),
            "resp_t2".into(),
            5,
        );
        let v = serde_json::to_value(&resp).unwrap();
        // Message first (text present), then the function call.
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][1]["type"], "function_call");
        assert_eq!(v["output"][1]["call_id"], "call_abc");
        assert_eq!(v["output"][1]["name"], "get_weather");
        assert_eq!(v["output"][1]["arguments"], r#"{"city":"SF"}"#);
        assert_eq!(v["output"][1]["status"], "completed");
        assert!(
            v["output"][1]["id"]
                .as_str()
                .expect("fc item id")
                .starts_with("fc_")
        );

        // Empty text + tools: no empty message item is emitted.
        let tc2 = CanonicalToolCall {
            id: "call_x".into(),
            name: "f".into(),
            arguments: "{}".into(),
        };
        let only_tools =
            responses_from_canonical(canon_resp("", vec![tc2]), "m".into(), "r".into(), 0);
        let v2 = serde_json::to_value(&only_tools).unwrap();
        assert_eq!(v2["output"].as_array().unwrap().len(), 1);
        assert_eq!(v2["output"][0]["type"], "function_call");
    }

    #[test]
    fn from_canonical_length_finish_marks_incomplete() {
        // WHY: clients rely on status=incomplete + the max_output_tokens reason
        // to detect truncation; mapping finish_reason "length" anywhere else
        // would hide truncated outputs.
        let mut canon = canon_resp("partial", vec![]);
        canon.finish_reason = Some("length".into());
        let resp = responses_from_canonical(canon, "m".into(), "r".into(), 0);
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["status"], "incomplete");
        assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[test]
    fn from_canonical_content_filter_marks_incomplete() {
        // WHY: policy stops are incomplete for Responses clients, but distinct
        // from max-token truncation.
        let mut canon = canon_resp("blocked", vec![]);
        canon.finish_reason = Some("content_filter".into());
        let resp = responses_from_canonical(canon, "m".into(), "r".into(), 0);
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["status"], "incomplete");
        assert_eq!(v["incomplete_details"]["reason"], "content_filter");
    }

    #[test]
    fn from_canonical_error_finish_marks_failed() {
        // WHY: provider adapters may encode native errors as finish reasons.
        // Responses clients must receive a failed envelope, not completed.
        let mut canon = canon_resp("partial", vec![]);
        canon.finish_reason = Some("error: overloaded_error: retry".into());
        let resp = responses_from_canonical(canon, "m".into(), "r".into(), 0);
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["status"], "failed");
        assert_eq!(v["error"]["type"], "server_error");
        assert_eq!(v["error"]["message"], "canonical response error");
    }

    #[test]
    fn from_canonical_preserves_native_response_id_and_refusal_part() {
        // WHY: Codex Responses continuations use the backend response id, and
        // refusal is a distinct Responses content part rather than output_text.
        let mut canon = canon_resp("", vec![]);
        canon.id = Some("resp_backend".into());
        canon.refusal = Some("No thanks".into());
        let resp = responses_from_canonical(canon, "m".into(), "resp_synth".into(), 0);
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["id"], "resp_backend");
        assert_eq!(v["output"][0]["id"], "msg_resp_backend");
        assert_eq!(v["output"][0]["content"][0]["type"], "refusal");
        assert_eq!(v["output"][0]["content"][0]["refusal"], "No thanks");
        assert!(v["output"][0]["content"][0].get("text").is_none());
    }

    #[test]
    fn from_canonical_preserves_annotations_usage_details_and_metadata() {
        // WHY: Responses clients consume citations/annotations and token-detail
        // accounting from this envelope; these were previously dropped.
        let canon = CanonicalResponse {
            model: "m".into(),
            content: "answer".into(),
            usage: CanonicalUsage {
                input_tokens: 10,
                output_tokens: 4,
                cache_read: 2,
                reasoning_tokens: 8,
                input_audio_tokens: 5,
                output_audio_tokens: 6,
                num_sources_used: 1,
                ..Default::default()
            },
            annotations: vec![serde_json::json!({"type":"url_citation","url":"https://e.test"})],
            metadata: Some(CanonicalResponseMetadata {
                service_tier: Some("default".into()),
                system_fingerprint: Some("fp_resp".into()),
                provider: Some("codex".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let v = serde_json::to_value(responses_from_canonical(
            canon,
            "m".into(),
            "resp_x".into(),
            0,
        ))
        .unwrap();
        assert_eq!(
            v["output"][0]["content"][0]["annotations"][0]["url"],
            "https://e.test"
        );
        assert_eq!(v["usage"]["input_tokens_details"]["cached_tokens"], 2);
        assert_eq!(v["usage"]["input_tokens_details"]["audio_tokens"], 5);
        assert_eq!(v["usage"]["output_tokens_details"]["reasoning_tokens"], 8);
        assert_eq!(v["usage"]["output_tokens_details"]["audio_tokens"], 6);
        assert_eq!(v["service_tier"], "default");
        assert_eq!(v["system_fingerprint"], "fp_resp");
        assert_eq!(v["provider_metadata"]["provider"], "codex");
        assert_eq!(v["provider_metadata"]["num_sources_used"], 1);
    }

    // ---- SSE streaming framing ----

    fn canonical_stream(
        events: Vec<Result<CanonicalStreamEvent, ProviderError>>,
    ) -> CanonicalStream {
        Box::pin(futures_util::stream::iter(events))
    }

    fn happy_stream() -> CanonicalStream {
        canonical_stream(vec![
            Ok(CanonicalStreamEvent::TextDelta("Hel".into())),
            Ok(CanonicalStreamEvent::TextDelta("lo".into())),
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_9".into()),
                name: Some("get_weather".into()),
                arguments_delta: String::new(),
            }),
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: r#"{"city":"SF"}"#.into(),
            }),
            Ok(CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 3,
                output_tokens: 9,
                ..Default::default()
            })),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("tool_calls".into()),
            }),
        ])
    }

    #[tokio::test]
    async fn sse_responses_stream_frames_events_in_order_with_completed() {
        // WHY: this is the Responses streaming wire contract SDK clients parse.
        // Each SSE event carries an `event:` name matching its JSON `type`;
        // the stream opens with response.created, message items are announced
        // before their text deltas, function calls are announced (with name)
        // before their argument deltas, and the terminal response.completed
        // carries the aggregated output + usage. There is NO [DONE] sentinel
        // (that is Chat Completions framing; emitting it here breaks parsers).
        let sse = sse_from_canonical_stream_responses(
            happy_stream(),
            "grok:grok-3".into(),
            "resp_s1".into(),
            42,
        );
        let body = sse_body_to_string(sse).await;

        assert!(
            body.contains("event: response.created"),
            "stream must open with response.created: {body}"
        );
        assert!(body.contains("event: response.output_item.added"));
        assert!(body.contains("event: response.content_part.added"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.output_text.done"));
        assert!(body.contains("event: response.content_part.done"));
        assert!(body.contains("event: response.output_item.done"));
        assert!(body.contains("event: response.function_call_arguments.delta"));
        assert!(body.contains("event: response.function_call_arguments.done"));
        assert!(body.contains("event: response.completed"));
        assert!(
            !body.contains("[DONE]"),
            "Responses streams must not emit the Chat [DONE] sentinel"
        );

        let payloads = data_payloads(&body);
        // created is first, completed is last.
        assert_eq!(payloads.first().unwrap()["type"], "response.created");
        assert_eq!(
            payloads.first().unwrap()["response"]["status"],
            "in_progress"
        );
        let last = payloads.last().unwrap();
        assert_eq!(last["type"], "response.completed");
        assert_eq!(last["response"]["status"], "completed");
        assert_eq!(last["response"]["id"], "resp_s1");
        assert_eq!(last["response"]["usage"]["input_tokens"], 3);
        assert_eq!(last["response"]["usage"]["output_tokens"], 9);
        assert_eq!(last["response"]["usage"]["total_tokens"], 12);

        // Aggregated output in the completed envelope: full text + the call.
        let out = last["response"]["output"].as_array().expect("output array");
        assert!(out.iter().any(|i| i["type"] == "message"
            && i["content"][0]["type"] == "output_text"
            && i["content"][0]["text"] == "Hello"));
        assert!(out.iter().any(|i| i["type"] == "function_call"
            && i["name"] == "get_weather"
            && i["call_id"] == "call_9"
            && i["arguments"] == r#"{"city":"SF"}"#));

        // Ordering: message item announced before its first text delta; the
        // function_call item (with name) announced before its argument delta.
        let msg_added = payloads
            .iter()
            .position(|p| {
                p["type"] == "response.output_item.added" && p["item"]["type"] == "message"
            })
            .expect("message output_item.added present");
        let content_added = payloads
            .iter()
            .position(|p| p["type"] == "response.content_part.added")
            .expect("content_part.added present");
        let first_text_delta = payloads
            .iter()
            .position(|p| p["type"] == "response.output_text.delta")
            .expect("text delta present");
        let fc_added = payloads
            .iter()
            .position(|p| {
                p["type"] == "response.output_item.added"
                    && p["item"]["type"] == "function_call"
                    && p["item"]["name"] == "get_weather"
            })
            .expect("function_call output_item.added present");
        let first_args_delta = payloads
            .iter()
            .position(|p| p["type"] == "response.function_call_arguments.delta")
            .expect("arguments delta present");
        assert!(
            msg_added < first_text_delta,
            "message announced before delta"
        );
        assert!(
            msg_added < content_added && content_added < first_text_delta,
            "content part announced between message item and text delta"
        );
        assert!(
            fc_added < first_args_delta,
            "call announced before arguments"
        );

        // The two text deltas arrive in order with the right fragments.
        let deltas: Vec<&str> = payloads
            .iter()
            .filter(|p| p["type"] == "response.output_text.delta")
            .map(|p| p["delta"].as_str().expect("delta string"))
            .collect();
        assert_eq!(deltas, vec!["Hel", "lo"]);
        let text_done = payloads
            .iter()
            .find(|p| p["type"] == "response.output_text.done")
            .expect("output_text done");
        assert_eq!(text_done["text"], "Hello");
        let text_part_done = payloads
            .iter()
            .find(|p| {
                p["type"] == "response.content_part.done" && p["part"]["type"] == "output_text"
            })
            .expect("output_text content part done");
        assert_eq!(text_part_done["part"]["text"], "Hello");
        let args_done = payloads
            .iter()
            .find(|p| p["type"] == "response.function_call_arguments.done")
            .expect("function arguments done");
        assert_eq!(args_done["arguments"], r#"{"city":"SF"}"#);
        let done_items: Vec<&serde_json::Value> = payloads
            .iter()
            .filter(|p| p["type"] == "response.output_item.done")
            .collect();
        assert_eq!(done_items.len(), 2);
        assert_eq!(done_items[0]["item"]["type"], "message");
        assert_eq!(done_items[1]["item"]["type"], "function_call");
        let last_done = payloads
            .iter()
            .rposition(|p| p["type"] == "response.output_item.done")
            .expect("last done event");
        let completed = payloads
            .iter()
            .position(|p| p["type"] == "response.completed")
            .expect("completed event");
        assert!(last_done < completed, "item done events precede completed");
    }

    #[tokio::test]
    async fn sse_responses_tool_first_keeps_stable_output_indexes() {
        // WHY: upstream providers may emit tool calls before text. Responses
        // clients key deltas by output_index, so indexes must be reserved once
        // and never recomputed from later text state.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_first".into()),
                name: Some("lookup".into()),
                arguments_delta: String::new(),
            }),
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: r#"{"q":"sf"}"#.into(),
            }),
            Ok(CanonicalStreamEvent::TextDelta("done".into())),
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: r#","unit":"f"}"#.into(),
            }),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("tool_calls".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_tf".into(), 0);
        let payloads = data_payloads(&sse_body_to_string(sse).await);

        let tool_added = payloads
            .iter()
            .find(|p| {
                p["type"] == "response.output_item.added" && p["item"]["type"] == "function_call"
            })
            .expect("tool added");
        assert_eq!(tool_added["output_index"], 0);
        let text_added = payloads
            .iter()
            .find(|p| p["type"] == "response.output_item.added" && p["item"]["type"] == "message")
            .expect("message added");
        assert_eq!(text_added["output_index"], 1);
        let arg_indexes: Vec<u64> = payloads
            .iter()
            .filter(|p| p["type"] == "response.function_call_arguments.delta")
            .map(|p| p["output_index"].as_u64().expect("arg output_index"))
            .collect();
        assert_eq!(arg_indexes, vec![0, 0]);
        let text_delta = payloads
            .iter()
            .find(|p| p["type"] == "response.output_text.delta")
            .expect("text delta");
        assert_eq!(text_delta["output_index"], 1);

        let last = payloads.last().unwrap();
        assert_eq!(last["response"]["output"][0]["type"], "function_call");
        assert_eq!(last["response"]["output"][1]["type"], "message");
    }

    #[tokio::test]
    async fn sse_responses_preserves_metadata_annotations_and_usage_details() {
        // WHY: streaming Responses terminal envelopes should expose the same
        // additive rich fields as non-streaming responses when providers emit
        // them during canonical stream parsing.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::ResponseMetadata(
                CanonicalResponseMetadata {
                    id: Some("resp_backend".into()),
                    system_fingerprint: Some("fp_stream".into()),
                    service_tier: Some("priority".into()),
                    provider: Some("codex".into()),
                    ..Default::default()
                },
            )),
            Ok(CanonicalStreamEvent::TextDelta("Hello".into())),
            Ok(CanonicalStreamEvent::OutputAnnotations(vec![
                serde_json::json!({"type":"url_citation","url":"https://e.test"}),
            ])),
            Ok(CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 9,
                output_tokens: 3,
                cache_read: 2,
                reasoning_tokens: 4,
                input_audio_tokens: 5,
                output_audio_tokens: 6,
                ..Default::default()
            })),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_synth".into(), 0);
        let payloads = data_payloads(&sse_body_to_string(sse).await);

        let created = payloads
            .iter()
            .find(|p| p["type"] == "response.created")
            .expect("created event");
        assert_eq!(created["response"]["id"], "resp_backend");
        assert_eq!(created["response"]["system_fingerprint"], "fp_stream");
        assert_eq!(created["response"]["service_tier"], "priority");
        assert_eq!(
            created["response"]["provider_metadata"]["provider"],
            "codex"
        );

        let text_part_done = payloads
            .iter()
            .find(|p| {
                p["type"] == "response.content_part.done" && p["part"]["type"] == "output_text"
            })
            .expect("text content part done");
        assert_eq!(
            text_part_done["part"]["annotations"][0]["url"],
            "https://e.test"
        );

        let completed = payloads
            .iter()
            .find(|p| p["type"] == "response.completed")
            .expect("completed event");
        assert_eq!(completed["response"]["system_fingerprint"], "fp_stream");
        assert_eq!(completed["response"]["service_tier"], "priority");
        assert_eq!(
            completed["response"]["provider_metadata"]["provider"],
            "codex"
        );
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["annotations"][0]["url"],
            "https://e.test"
        );
        assert_eq!(
            completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            2
        );
        assert_eq!(
            completed["response"]["usage"]["input_tokens_details"]["audio_tokens"],
            5
        );
        assert_eq!(
            completed["response"]["usage"]["output_tokens_details"]["reasoning_tokens"],
            4
        );
        assert_eq!(
            completed["response"]["usage"]["output_tokens_details"]["audio_tokens"],
            6
        );
    }

    #[tokio::test]
    async fn sse_responses_late_tool_metadata_opens_once_with_stable_item_id() {
        // WHY: provider streams should open tool calls with metadata, but this
        // framer is shared. If late metadata reaches it, live SSE must not emit
        // a placeholder function_call item or switch item_id mid-stream.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: String::new(),
            }),
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: r#"{"q":"sf"}"#.into(),
            }),
            Ok(CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_real".into()),
                name: Some("lookup".into()),
                arguments_delta: String::new(),
            }),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("tool_calls".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_late".into(), 0);
        let payloads = data_payloads(&sse_body_to_string(sse).await);
        let tool_added = payloads
            .iter()
            .filter(|p| {
                p["type"] == "response.output_item.added" && p["item"]["type"] == "function_call"
            })
            .collect::<Vec<_>>();
        assert_eq!(tool_added.len(), 1);
        assert_eq!(tool_added[0]["item"]["id"], "fc_call_real");
        assert_eq!(tool_added[0]["item"]["call_id"], "call_real");
        assert_eq!(tool_added[0]["item"]["name"], "lookup");
        let arg_deltas = payloads
            .iter()
            .filter(|p| p["type"] == "response.function_call_arguments.delta")
            .collect::<Vec<_>>();
        assert_eq!(arg_deltas.len(), 1);
        assert_eq!(arg_deltas[0]["item_id"], "fc_call_real");
        assert_eq!(arg_deltas[0]["delta"], r#"{"q":"sf"}"#);
        let last = payloads.last().unwrap();
        assert_eq!(last["response"]["output"][0]["call_id"], "call_real");
        assert_eq!(last["response"]["output"][0]["name"], "lookup");
        assert_eq!(last["response"]["output"][0]["arguments"], r#"{"q":"sf"}"#);
    }

    #[tokio::test]
    async fn sse_responses_sequence_numbers_strictly_increase() {
        // WHY: SDKs use sequence_number to detect missed/reordered events; it
        // must be present on every event and strictly increasing.
        let sse =
            sse_from_canonical_stream_responses(happy_stream(), "m".into(), "resp_seq".into(), 0);
        let body = sse_body_to_string(sse).await;
        let payloads = data_payloads(&body);
        assert!(payloads.len() >= 4, "expected several events");
        let seqs: Vec<u64> = payloads
            .iter()
            .map(|p| {
                p["sequence_number"]
                    .as_u64()
                    .expect("every event carries sequence_number")
            })
            .collect();
        for pair in seqs.windows(2) {
            assert!(
                pair[1] > pair[0],
                "sequence numbers must strictly increase: {seqs:?}"
            );
        }
    }

    #[tokio::test]
    async fn sse_responses_uses_provider_response_id_before_created() {
        // WHY: stateful Responses clients pass the returned id as
        // previous_response_id, so native provider ids must replace Omni's
        // synthetic id before response.created.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::ResponseMetadata(
                CanonicalResponseMetadata {
                    id: Some("resp_backend".into()),
                    ..Default::default()
                },
            )),
            Ok(CanonicalStreamEvent::TextDelta("ok".into())),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_synth".into(), 0);
        let payloads = data_payloads(&sse_body_to_string(sse).await);
        assert_eq!(payloads[0]["type"], "response.created");
        assert_eq!(payloads[0]["response"]["id"], "resp_backend");
        let text_delta = payloads
            .iter()
            .find(|p| p["type"] == "response.output_text.delta")
            .expect("text delta");
        assert_eq!(text_delta["item_id"], "msg_resp_backend");
        assert_eq!(payloads.last().unwrap()["response"]["id"], "resp_backend");
    }

    #[tokio::test]
    async fn sse_responses_refusal_delta_remains_refusal_in_terminal_envelope() {
        // WHY: Responses-native clients need refusal semantics on the stream
        // and in the final envelope, not only flattened text.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::RefusalDelta("No".into())),
            Ok(CanonicalStreamEvent::RefusalDelta(" thanks".into())),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("content_filter".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_ref".into(), 0);
        let payloads = data_payloads(&sse_body_to_string(sse).await);
        let refusal_deltas: Vec<&str> = payloads
            .iter()
            .filter(|p| p["type"] == "response.refusal.delta")
            .map(|p| p["delta"].as_str().expect("refusal delta"))
            .collect();
        assert_eq!(refusal_deltas, vec!["No", " thanks"]);
        let refusal_done = payloads
            .iter()
            .find(|p| p["type"] == "response.refusal.done")
            .expect("refusal done");
        assert_eq!(refusal_done["content_index"], 0);
        assert_eq!(refusal_done["refusal"], "No thanks");
        let last = payloads.last().unwrap();
        assert_eq!(last["type"], "response.incomplete");
        assert_eq!(
            last["response"]["output"][0]["content"][0]["type"],
            "refusal"
        );
        assert_eq!(
            last["response"]["output"][0]["content"][0]["refusal"],
            "No thanks"
        );
    }

    #[tokio::test]
    async fn sse_responses_text_and_refusal_use_distinct_content_indexes() {
        // WHY: Responses message content parts are separately indexed. If text
        // and refusal both use content_index 0, clients cannot reconcile the
        // delta/done events with the terminal message content array.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::TextDelta("Allowed".into())),
            Ok(CanonicalStreamEvent::RefusalDelta("Nope".into())),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("content_filter".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_mix".into(), 0);
        let payloads = data_payloads(&sse_body_to_string(sse).await);

        let text_delta = payloads
            .iter()
            .find(|p| p["type"] == "response.output_text.delta")
            .expect("text delta");
        assert_eq!(text_delta["content_index"], 0);
        let text_part_added = payloads
            .iter()
            .find(|p| {
                p["type"] == "response.content_part.added" && p["part"]["type"] == "output_text"
            })
            .expect("text part added");
        assert_eq!(text_part_added["content_index"], 0);
        let refusal_delta = payloads
            .iter()
            .find(|p| p["type"] == "response.refusal.delta")
            .expect("refusal delta");
        assert_eq!(refusal_delta["content_index"], 1);
        let refusal_part_added = payloads
            .iter()
            .find(|p| p["type"] == "response.content_part.added" && p["part"]["type"] == "refusal")
            .expect("refusal part added");
        assert_eq!(refusal_part_added["content_index"], 1);
        let text_done = payloads
            .iter()
            .find(|p| p["type"] == "response.output_text.done")
            .expect("text done");
        assert_eq!(text_done["content_index"], 0);
        let refusal_done = payloads
            .iter()
            .find(|p| p["type"] == "response.refusal.done")
            .expect("refusal done");
        assert_eq!(refusal_done["content_index"], 1);
        let refusal_part_done = payloads
            .iter()
            .find(|p| p["type"] == "response.content_part.done" && p["part"]["type"] == "refusal")
            .expect("refusal part done");
        assert_eq!(refusal_part_done["content_index"], 1);

        let content = payloads.last().unwrap()["response"]["output"][0]["content"]
            .as_array()
            .expect("terminal message content");
        assert_eq!(content[0]["type"], "output_text");
        assert_eq!(content[0]["text"], "Allowed");
        assert_eq!(content[1]["type"], "refusal");
        assert_eq!(content[1]["refusal"], "Nope");
    }

    #[tokio::test]
    async fn sse_responses_error_midstream_emits_failed() {
        // WHY: a client must learn the stream died; a silent hang or a fake
        // completed status would corrupt downstream state. The terminal event
        // for an errored stream is response.failed with status failed.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::TextDelta("par".into())),
            Err(ProviderError::upstream("boom mid-stream")),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_e".into(), 0);
        let body = sse_body_to_string(sse).await;
        assert!(
            body.contains("event: response.failed"),
            "errored stream must terminate with response.failed: {body}"
        );
        let payloads = data_payloads(&body);
        let last = payloads.last().unwrap();
        assert_eq!(last["type"], "response.failed");
        assert_eq!(last["response"]["status"], "failed");
        assert_eq!(
            last["response"]["error"]["message"],
            "canonical stream error"
        );
        assert_eq!(last["response"]["error"]["type"], "server_error");
        assert!(!body.contains("[DONE]"));
    }

    #[tokio::test]
    async fn sse_responses_missing_finish_emits_failed() {
        // WHY: the canonical streaming contract requires an explicit Finish.
        // EOF without it is a truncated upstream stream, not a successful
        // Responses completion.
        let stream = canonical_stream(vec![Ok(CanonicalStreamEvent::TextDelta("partial".into()))]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_eof".into(), 0);
        let payloads = data_payloads(&sse_body_to_string(sse).await);
        let last = payloads.last().unwrap();
        assert_eq!(last["type"], "response.failed");
        assert_eq!(last["response"]["status"], "failed");
        assert_eq!(
            last["response"]["error"]["message"],
            "canonical stream error"
        );
        assert!(
            last["response"]["error"]["message"]
                .as_str()
                .expect("error message")
                .contains("canonical stream error"),
            "client-visible error should be generic: {last}"
        );
    }

    #[tokio::test]
    async fn sse_responses_error_finish_emits_failed_not_completed() {
        // WHY: some provider adapters encode native stream error events as a
        // terminal Finish reason. Responses clients must see failure, not a
        // successful completion with partial text.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::TextDelta("partial".into())),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("error: overloaded_error: retry".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_err".into(), 0);
        let body = sse_body_to_string(sse).await;
        assert!(
            !body.contains("event: response.completed"),
            "error finish must not emit completed: {body}"
        );
        let payloads = data_payloads(&body);
        let last = payloads.last().unwrap();
        assert_eq!(last["type"], "response.failed");
        assert_eq!(last["response"]["status"], "failed");
        assert_eq!(
            last["response"]["error"]["message"],
            "canonical stream error"
        );
    }

    #[tokio::test]
    async fn sse_responses_length_finish_emits_incomplete() {
        // WHY: truncation must surface in streaming exactly like non-streaming
        // (status incomplete + max_output_tokens reason), or streaming clients
        // silently treat truncated output as complete.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::TextDelta("trunc".into())),
            Ok(CanonicalStreamEvent::Usage(CanonicalUsage {
                input_tokens: 1,
                output_tokens: 2,
                ..Default::default()
            })),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("length".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_l".into(), 0);
        let body = sse_body_to_string(sse).await;
        assert!(
            body.contains("event: response.incomplete"),
            "length finish must terminate with response.incomplete: {body}"
        );
        let payloads = data_payloads(&body);
        let last = payloads.last().unwrap();
        assert_eq!(last["type"], "response.incomplete");
        assert_eq!(last["response"]["status"], "incomplete");
        assert_eq!(
            last["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    #[tokio::test]
    async fn sse_responses_content_filter_finish_emits_incomplete_reason() {
        // WHY: not every incomplete stream is a max-token truncation; preserving
        // content_filter lets clients distinguish policy stop from length stop.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::TextDelta("blocked".into())),
            Ok(CanonicalStreamEvent::Finish {
                finish_reason: Some("content_filter".into()),
            }),
        ]);
        let sse = sse_from_canonical_stream_responses(stream, "m".into(), "resp_cf".into(), 0);
        let payloads = data_payloads(&sse_body_to_string(sse).await);
        let last = payloads.last().unwrap();
        assert_eq!(last["type"], "response.incomplete");
        assert_eq!(last["response"]["status"], "incomplete");
        assert_eq!(
            last["response"]["incomplete_details"]["reason"],
            "content_filter"
        );
    }
}

#[cfg(test)]
mod issue_51_tool_tests {
    use super::*;
    use serde_json::{Value, json};

    fn parse(schema: Value, strict: Value) -> Result<omni_core::CanonicalRequest, String> {
        let req: ResponsesRequest = serde_json::from_value(json!({"model":"gpt","input":"hi", "tools":[{"type":"function","name":"f","parameters":schema,"strict":strict}]})).unwrap();
        responses_to_canonical(&req)
    }

    #[test]
    fn responses_keep_schema_and_validate_strict() {
        let schema = json!({"type":"object","properties":{"x":{"anyOf":[{"type":"string"}]}}});
        assert_eq!(
            parse(schema.clone(), json!(true)).unwrap().tools.unwrap()[0].parameters,
            schema
        );
        assert!(
            !parse(json!({"anyOf":[true]}), Value::Null)
                .unwrap()
                .tools
                .unwrap()[0]
                .strict
        );
        for bad in [
            json!({"type":"object","anyOf":[{}]}),
            json!({"type":"object","properties":{"x":{"allOf":[{}]}}}),
        ] {
            assert!(parse(bad, json!(true)).is_err());
        }
        assert!(parse(json!({"properties":{"x":{"anyOf":[]}}}), Value::Null).is_err());
    }
}
