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
    CanonicalBlock, CanonicalContent, CanonicalMessage, CanonicalRequest, CanonicalResponse,
    CanonicalStream, CanonicalStreamEvent, CanonicalTool, CanonicalToolCall, CanonicalToolChoice,
};

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
    pub content: Option<String>,
    /// Present on assistant messages that called tools (OpenAI echoes these
    /// back into the next request's history).
    #[serde(default)]
    pub tool_calls: Option<Vec<ChatToolCallReq>>,
    /// Present on `role:"tool"` messages: the id of the tool call this result
    /// answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
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
}

/// `tool_choice`: either a bare mode string or a forced function selection.
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
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatToolChoiceFunction {
    pub name: String,
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
}

const GATEWAY_ONLY_EXTRAS: &[&str] = &["user"];

/// Top-level request fields that Omni consumes as gateway metadata rather than
/// forwarding as provider extras.
pub fn gateway_only_extra_keys() -> &'static [&'static str] {
    GATEWAY_ONLY_EXTRAS
}

fn provider_extras_from_flattened(extras: &Value) -> Option<Value> {
    let filtered = extras.as_object().and_then(|extras| {
        let filtered = extras
            .iter()
            .filter(|(key, _)| !gateway_only_extra_keys().contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<Map<_, _>>();
        if filtered.is_empty() {
            None
        } else {
            Some(filtered)
        }
    })?;
    Some(Value::Object(filtered))
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
    let messages: Vec<CanonicalMessage> = req
        .messages
        .iter()
        .map(chat_message_to_canonical)
        .collect::<Result<_, _>>()?;

    // Nested function tools -> canonical tools (non-function tools rejected).
    let tools = match req.tools.as_ref() {
        Some(ts) if !ts.is_empty() => {
            let mut out = Vec::with_capacity(ts.len());
            for t in ts {
                if t.kind != "function" {
                    return Err(format!("unsupported tool type: {}", t.kind));
                }
                out.push(CanonicalTool {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t
                        .function
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
    // visible), {function,name}->Specific.
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
        None => None,
    };

    Ok(CanonicalRequest {
        model: req.model.clone(),
        messages,
        tools,
        tool_choice,
        // OpenAI's max_completion_tokens supersedes the legacy max_tokens.
        max_tokens: req.max_completion_tokens.or(req.max_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        reasoning: None,
        metadata: Default::default(),
        provider_extras: provider_extras_from_flattened(&req.extras),
    })
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
        return Ok(CanonicalMessage {
            role: "tool".into(),
            content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                tool_use_id: id,
                content: m.content.clone().unwrap_or_default(),
                is_error: false,
            }]),
        });
    }

    // An assistant message may interleave text with tool calls.
    if let Some(tool_calls) = m.tool_calls.as_ref().filter(|tc| !tc.is_empty()) {
        let mut blocks: Vec<CanonicalBlock> = Vec::new();
        if let Some(text) = m.content.as_ref().filter(|t| !t.is_empty()) {
            blocks.push(CanonicalBlock::Text(text.clone()));
        }
        for tc in tool_calls {
            if tc.kind != "function" {
                return Err(format!("unsupported tool_call type: {}", tc.kind));
            }
            blocks.push(CanonicalBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
            });
        }
        return Ok(CanonicalMessage {
            role: m.role.clone(),
            content: CanonicalContent::Blocks(blocks),
        });
    }

    Ok(CanonicalMessage {
        role: m.role.clone(),
        content: CanonicalContent::Text(m.content.clone().unwrap_or_default()),
    })
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
                refusal: canon.refusal,
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
                Ok(CanonicalStreamEvent::RefusalDelta(text)) => {
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
                Ok(CanonicalStreamEvent::ResponseMetadata(_)) => {
                    // Chat Completions has no response-id metadata event.
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
