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
//! parts), function tools, and full multi-turn tool conversations
//! (`function_call` / `function_call_output` items round-trip through canonical
//! tool blocks), non-stream and stream. Input shapes the canonical layer still
//! cannot represent (e.g. `input_image` parts, non-function tools) are rejected
//! loudly with a clear error instead of degraded silently.
//!
//! The types below are the wire contract. The conversion and SSE-framing
//! functions are pinned by tests in this module.

use std::convert::Infallible;
use std::pin::Pin;

use axum::response::sse::{Event, Sse};
use futures_util::Stream;
use serde::{Deserialize, Serialize};

use omni_core::{
    CanonicalBlock, CanonicalContent, CanonicalMessage, CanonicalReasoning, CanonicalRequest,
    CanonicalResponse, CanonicalStream, CanonicalStreamEvent, CanonicalTool, CanonicalToolChoice,
};

/// The boxed SSE event stream produced by the Responses framer. Boxed so the
/// signature stays stable while the implementation evolves.
pub type ResponsesSseStream = Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

// ── Request (Deserialize) ─────────────────────────────────────────

/// `POST /v1/responses` request body (supported subset). Unknown fields are
/// captured in `extras` so a request never fails to parse on an
/// unrecognized key.
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default)]
    pub instructions: Option<String>,
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
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

/// One entry of an `input` array. Message items carry `role` + `content`
/// (`type` defaults to "message" when absent). Tool-conversation items use
/// `type:"function_call"` (the assistant's prior tool call) and
/// `type:"function_call_output"` (the result fed back), keyed by `call_id`.
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInputContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

/// One typed content part. Only `input_text` / `output_text` are supported in
/// v1; anything else (e.g. `input_image`) is rejected by name.
#[derive(Debug, Deserialize)]
pub struct ResponsesContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesReasoning {
    #[serde(default)]
    pub effort: Option<String>,
}

/// Responses tools are FLATTENED function definitions (unlike Chat
/// Completions' nested `function` object): `{type:"function", name, ...}`.
#[derive(Debug, Deserialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
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
    pub usage: ResponsesUsage,
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
    pub kind: &'static str, // "output_text"
    pub text: String,
    pub annotations: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct IncompleteDetails {
    pub reason: String,
}

/// Responses usage uses input/output naming (NOT prompt/completion like Chat).
#[derive(Debug, Serialize, Default)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
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
/// - flattened function tools -> canonical tools; `tool_choice` "auto" ->
///   Auto, "required" -> Required, "none" -> None (tools stay visible, model
///   must not call), `{type:"function",name}` -> Specific
/// - unsupported shapes (unknown item types, non-text parts like `input_image`,
///   non-function tools, message items without a role) -> Err naming the offender
pub fn responses_to_canonical(req: &ResponsesRequest) -> Result<CanonicalRequest, String> {
    let mut messages: Vec<CanonicalMessage> = Vec::new();

    // `instructions` is the Responses system prompt: a leading system message.
    if let Some(instructions) = req.instructions.as_ref() {
        messages.push(CanonicalMessage {
            role: "system".into(),
            content: CanonicalContent::Text(instructions.clone()),
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
                        let text = match item.content.as_ref() {
                            Some(ResponsesInputContent::Text(t)) => t.clone(),
                            Some(ResponsesInputContent::Parts(parts)) => {
                                let mut fragments = Vec::with_capacity(parts.len());
                                for part in parts {
                                    if part.kind != "input_text" && part.kind != "output_text" {
                                        return Err(format!(
                                            "unsupported content part type: {}",
                                            part.kind
                                        ));
                                    }
                                    fragments.push(part.text.clone().unwrap_or_default());
                                }
                                fragments.join("\n")
                            }
                            None => String::new(),
                        };
                        messages.push(CanonicalMessage {
                            role: role.to_string(),
                            content: CanonicalContent::Text(text),
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
    let tools = match req.tools.as_ref() {
        Some(ts) if !ts.is_empty() => {
            let mut out = Vec::with_capacity(ts.len());
            for t in ts {
                if t.kind != "function" {
                    return Err(format!("unsupported tool type: {}", t.kind));
                }
                out.push(CanonicalTool {
                    name: t.name.clone().unwrap_or_default(),
                    description: t.description.clone(),
                    parameters: t
                        .parameters
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({})),
                });
            }
            Some(out)
        }
        _ => None,
    };

    // tool_choice: "auto"->Auto, "required"->Required, "none"->None (tools stay
    // visible but the model must not call them), {function,name}->Specific.
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
        None => None,
    };

    let reasoning = req.reasoning.as_ref().map(|r| CanonicalReasoning {
        effort: r.effort.clone(),
        budget_tokens: None,
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
        provider_extras: None,
    })
}

/// Convert a `CanonicalResponse` into the Responses envelope.
///
/// Contract (pinned by tests): one assistant Message item (output_text part)
/// when content is non-empty, then one FunctionCall item per canonical tool
/// call (call_id = canonical id, ids prefixed "msg_"/"fc_"); finish_reason
/// "length" -> status "incomplete" with incomplete_details.reason
/// "max_output_tokens", everything else -> "completed"; usage totals filled.
pub fn responses_from_canonical(
    canon: CanonicalResponse,
    requested_model: String,
    response_id: String,
    created_at: u64,
) -> ResponsesResponse {
    let length_finish = canon.finish_reason.as_deref() == Some("length");
    let (status, incomplete_details) = if length_finish {
        (
            "incomplete".to_string(),
            Some(IncompleteDetails {
                reason: "max_output_tokens".into(),
            }),
        )
    } else {
        ("completed".to_string(), None)
    };

    let mut output: Vec<ResponsesOutputItem> = Vec::new();

    // A message item is emitted only when there is assistant text; a tool-only
    // turn carries no empty message (mirrors Chat's null-content contract).
    if !canon.content.is_empty() {
        output.push(ResponsesOutputItem::Message {
            id: format!("msg_{response_id}"),
            status: status.clone(),
            role: "assistant",
            content: vec![ResponsesOutputContent {
                kind: "output_text",
                text: canon.content,
                annotations: Vec::new(),
            }],
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

    ResponsesResponse {
        id: response_id,
        object: "response",
        created_at,
        status,
        model: requested_model,
        output,
        incomplete_details,
        usage: ResponsesUsage {
            input_tokens: canon.usage.input_tokens,
            output_tokens: canon.usage.output_tokens,
            total_tokens: total,
        },
    }
}

/// Frame a canonical event stream as Responses SSE.
///
/// Contract (pinned by tests): every SSE event has an `event:` name matching
/// the JSON `type` and a strictly-increasing `sequence_number`. Sequence:
/// `response.created` first; `response.output_item.added` (message) before the
/// first `response.output_text.delta`; tool calls open with
/// `response.output_item.added` (function_call, carrying the name) followed by
/// `response.function_call_arguments.delta` events; terminal event is
/// `response.completed` (carrying the aggregated output + usage + status
/// "completed"), or `response.incomplete` on a "length" finish, or
/// `response.failed` on a stream error. NO `data: [DONE]` sentinel.
pub fn sse_from_canonical_stream_responses(
    stream: CanonicalStream,
    requested_model: String,
    response_id: String,
    created_at: u64,
) -> Sse<ResponsesSseStream> {
    let body = responses_sse_events(stream, requested_model, response_id, created_at);
    Sse::new(Box::pin(body) as ResponsesSseStream)
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
struct StreamMeta {
    response_id: String,
    created_at: u64,
    model: String,
}

/// Build the terminal response envelope embedded in `response.completed` /
/// `response.incomplete` / `response.failed`, carrying the aggregated output
/// and usage so a streaming client ends with the same shape as non-streaming.
fn responses_stream_envelope(
    meta: &StreamMeta,
    status: &str,
    text: &str,
    tool_calls: &[(String, String, String)], // (call_id, name, arguments)
    usage: &CanonicalUsageAccum,
    incomplete_reason: Option<&str>,
) -> serde_json::Value {
    let mut output: Vec<serde_json::Value> = Vec::new();
    if !text.is_empty() {
        output.push(serde_json::json!({
            "type": "message",
            "id": format!("msg_{}", meta.response_id),
            "status": status,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }));
    }
    for (call_id, name, arguments) in tool_calls {
        output.push(serde_json::json!({
            "type": "function_call",
            "id": format!("fc_{call_id}"),
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": status,
        }));
    }
    let mut envelope = serde_json::json!({
        "id": meta.response_id,
        "object": "response",
        "created_at": meta.created_at,
        "status": status,
        "model": meta.model,
        "output": output,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        },
    });
    if let Some(reason) = incomplete_reason {
        envelope["incomplete_details"] = serde_json::json!({ "reason": reason });
    }
    envelope
}

#[derive(Default)]
struct CanonicalUsageAccum {
    input_tokens: u64,
    output_tokens: u64,
}

fn responses_sse_events(
    mut stream: CanonicalStream,
    requested_model: String,
    response_id: String,
    created_at: u64,
) -> impl Stream<Item = Result<Event, Infallible>> {
    use futures_util::StreamExt;
    let meta = StreamMeta {
        response_id,
        created_at,
        model: requested_model,
    };
    async_stream::stream! {
        let mut seq: u64 = 0;

        // response.created opens the stream (status in_progress, empty output).
        let created_env = serde_json::json!({
            "id": meta.response_id,
            "object": "response",
            "created_at": meta.created_at,
            "status": "in_progress",
            "model": meta.model,
            "output": [],
        });
        yield Ok(responses_event(&mut seq, "response.created", serde_json::json!({
            "response": created_env,
        })));

        // Aggregated state for the terminal envelope.
        let mut text = String::new();
        let mut message_added = false;
        // Tool calls in arrival order, indexed by their canonical stream index.
        let mut tool_calls: Vec<(u32, String, String, String)> = Vec::new(); // (index, call_id, name, args)
        let mut usage = CanonicalUsageAccum::default();
        let mut finish_reason: Option<String> = None;
        let mut errored = false;

        while let Some(item) = stream.next().await {
            match item {
                Ok(CanonicalStreamEvent::TextDelta(delta)) => {
                    if !message_added {
                        message_added = true;
                        let item = serde_json::json!({
                            "type": "message",
                            "id": format!("msg_{}", meta.response_id),
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        });
                        yield Ok(responses_event(&mut seq, "response.output_item.added", serde_json::json!({
                            "item": item,
                        })));
                    }
                    text.push_str(&delta);
                    yield Ok(responses_event(&mut seq, "response.output_text.delta", serde_json::json!({
                        "delta": delta,
                    })));
                }
                Ok(CanonicalStreamEvent::ToolCallDelta { index, id, name, arguments_delta }) => {
                    // First delta for an index opens the function_call item.
                    if !tool_calls.iter().any(|(i, ..)| *i == index) {
                        let call_id = id.clone().unwrap_or_else(|| format!("call_{index}"));
                        let call_name = name.clone().unwrap_or_default();
                        tool_calls.push((index, call_id.clone(), call_name.clone(), String::new()));
                        let item = serde_json::json!({
                            "type": "function_call",
                            "id": format!("fc_{call_id}"),
                            "call_id": call_id,
                            "name": call_name,
                            "arguments": "",
                            "status": "in_progress",
                        });
                        yield Ok(responses_event(&mut seq, "response.output_item.added", serde_json::json!({
                            "item": item,
                        })));
                    }
                    if !arguments_delta.is_empty() {
                        if let Some(slot) = tool_calls.iter_mut().find(|(i, ..)| *i == index) {
                            slot.3.push_str(&arguments_delta);
                        }
                        yield Ok(responses_event(&mut seq, "response.function_call_arguments.delta", serde_json::json!({
                            "delta": arguments_delta,
                        })));
                    }
                }
                Ok(CanonicalStreamEvent::Usage(u)) => {
                    usage.input_tokens = u.input_tokens;
                    usage.output_tokens = u.output_tokens;
                }
                Ok(CanonicalStreamEvent::Finish { finish_reason: fr }) => {
                    finish_reason = fr;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "canonical stream error mid-flight (responses)");
                    errored = true;
                    break;
                }
            }
        }

        let calls: Vec<(String, String, String)> = tool_calls
            .into_iter()
            .map(|(_, call_id, name, args)| (call_id, name, args))
            .collect();

        if errored {
            let env = responses_stream_envelope(
                &meta, "failed", &text, &calls, &usage, None,
            );
            yield Ok(responses_event(&mut seq, "response.failed", serde_json::json!({
                "response": env,
            })));
        } else if finish_reason.as_deref() == Some("length") {
            let env = responses_stream_envelope(
                &meta, "incomplete", &text, &calls, &usage, Some("max_output_tokens"),
            );
            yield Ok(responses_event(&mut seq, "response.incomplete", serde_json::json!({
                "response": env,
            })));
        } else {
            let env = responses_stream_envelope(
                &meta, "completed", &text, &calls, &usage, None,
            );
            yield Ok(responses_event(&mut seq, "response.completed", serde_json::json!({
                "response": env,
            })));
        }
        // NB: no [DONE] sentinel — that is Chat Completions framing only.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::{
        CanonicalContent, CanonicalStreamEvent, CanonicalToolCall, CanonicalToolChoice,
        CanonicalUsage, ProviderError,
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
    fn to_canonical_rejects_non_text_content_parts() {
        let req = parse(
            r#"{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":"http://x/y.png"}]}]}"#,
        );
        let err = responses_to_canonical(&req).expect_err("must reject");
        assert!(
            err.contains("input_image"),
            "error must name the unsupported part type, got: {err}"
        );
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
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.function_call_arguments.delta"));
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
    async fn sse_responses_error_midstream_emits_failed() {
        // WHY: a client must learn the stream died; a silent hang or a fake
        // completed status would corrupt downstream state. The terminal event
        // for an errored stream is response.failed with status failed.
        let stream = canonical_stream(vec![
            Ok(CanonicalStreamEvent::TextDelta("par".into())),
            Err(ProviderError::Upstream("boom mid-stream".into())),
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
        assert!(!body.contains("[DONE]"));
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
}
