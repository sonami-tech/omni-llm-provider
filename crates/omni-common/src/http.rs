//! Shared OpenAI-compatible HTTP surface: request/response types, canonical
//! conversion, and SSE streaming framing.
//!
//! All three binaries (omni, omni-claude, omni-grok) speak the same
//! OpenAI-compatible wire shape and delegate to `LlmProvider`. This module is
//! the single source of truth for that translation so the binaries do not
//! triplicate it. The aggregator (`omni`) adds prefix routing on top; the
//! single-backend binaries (`omni-claude`, `omni-grok`) use these helpers
//! directly against their one provider.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use omni_core::{
    CanonicalContent, CanonicalMessage, CanonicalRequest, CanonicalResponse, CanonicalStream,
    CanonicalStreamEvent, CanonicalToolCall,
};

/// Minimal OpenAI-compatible chat completion request (the supported subset:
/// text messages + core sampling). Unknown fields are captured in `extras` so a
/// client request never fails to deserialize on an unrecognized key.
#[derive(Debug, Deserialize)]
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
    #[serde(flatten)]
    pub extras: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
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
}

/// Convert an OpenAI request into a `CanonicalRequest`. The `model` field is the
/// caller-supplied value; single-backend binaries pass it through, the
/// aggregator overwrites it with the prefix-stripped model before delegating.
pub fn to_canonical(req: &ChatCompletionRequest) -> CanonicalRequest {
    let messages: Vec<CanonicalMessage> = req
        .messages
        .iter()
        .map(|m| CanonicalMessage {
            role: m.role.clone(),
            content: CanonicalContent::Text(m.content.clone().unwrap_or_default()),
        })
        .collect();

    CanonicalRequest {
        model: req.model.clone(),
        messages,
        tools: None,
        tool_choice: None,
        // OpenAI's max_completion_tokens supersedes the legacy max_tokens.
        max_tokens: req.max_completion_tokens.or(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        reasoning: None,
        metadata: Default::default(),
        provider_extras: None,
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
                tool_calls,
            },
            finish_reason: finish,
        }],
        usage: ChatUsage {
            prompt_tokens: canon.usage.input_tokens,
            completion_tokens: canon.usage.output_tokens,
            total_tokens: total,
        },
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
/// - `Finish` -> a finish-reason chunk (default "stop" when none)
/// - `Usage` is not emitted as a chunk (OpenAI streams omit usage by default).
///
/// The stream is always terminated by a literal `data: [DONE]` event, matching
/// the OpenAI streaming protocol. An error in the underlying stream is mapped to
/// a finish chunk with reason "error" followed by `[DONE]` so the consumer
/// always sees a clean termination.
pub fn sse_from_canonical_stream(
    stream: CanonicalStream,
    requested_model: String,
    chat_id: String,
    created: u64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let body = async_stream_chunks(stream, requested_model, chat_id, created);
    Sse::new(body)
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
                Ok(CanonicalStreamEvent::Finish { finish_reason }) => {
                    finished = true;
                    let reason = finish_reason.unwrap_or_else(|| "stop".to_string());
                    let v = chunk_finish(&chat_id, created, &requested_model, &reason);
                    yield Ok(Event::default().data(v.to_string()));
                }
                Err(e) => {
                    finished = true;
                    let v = chunk_finish(&chat_id, created, &requested_model, "error");
                    tracing::warn!(error = %e, "canonical stream error mid-flight");
                    yield Ok(Event::default().data(v.to_string()));
                }
            }
        }
        if !finished {
            // Upstream ended without an explicit Finish; synthesize one so the
            // client still sees a terminal chunk before [DONE].
            let v = chunk_finish(&chat_id, created, &requested_model, "stop");
            yield Ok(Event::default().data(v.to_string()));
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
            }],
            stream: false,
            max_tokens: Some(10),
            max_completion_tokens: None,
            temperature: Some(0.5),
            top_p: None,
            extras: serde_json::Value::Null,
        }
    }

    #[test]
    fn to_canonical_maps_text_and_sampling() {
        // WHY: the canonical request is the contract every provider consumes;
        // dropping a message or a sampling field silently changes behavior.
        let canon = to_canonical(&sample_oai_req());
        assert_eq!(canon.messages.len(), 1);
        match &canon.messages[0].content {
            CanonicalContent::Text(t) => assert_eq!(t, "hi"),
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
        assert_eq!(to_canonical(&req).max_tokens, Some(99));
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
        };
        let oai = from_canonical(canon, "m".into(), "id".into(), 0);
        assert!(oai.choices[0].message.content.is_none());
        assert_eq!(oai.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(oai.choices[0].message.tool_calls.len(), 1);
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
}
