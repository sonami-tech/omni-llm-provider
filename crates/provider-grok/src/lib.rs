//! provider-grok
//!
//! Grok / xAI provider implementation.
//!
//! Uses omni-core canonical types (CanonicalRequest / CanonicalResponse + LlmProvider trait).
//! Makes real HTTP calls to https://api.x.ai/v1/chat/completions (primary OpenAI-compatible surface).
//! Auth: XAI_API_KEY via Bearer token (env var or passed to ctor).
//!
//! ## Headers / wire notes (research findings, 2026-06)
//! - **Standard, no special gates**: `Authorization: Bearer $XAI_API_KEY`, `Content-Type: application/json`.
//!   No xai-*- headers, no cch checksum, no OAuth subscription gate, no identity preamble, no per-version
//!   fingerprint profiles (unlike Claude Code provider). xAI accepts standard OpenAI SDK clients pointed at
//!   base_url="https://api.x.ai/v1".
//! - API keys are typically prefixed `xai-...` (but the wire does not enforce or inspect the prefix; any valid
//!   bearer from https://console.x.ai works).
//! - Primary focus (per requirements): /v1/chat/completions for chat + tools + streaming compat.
//! - Also exposes /v1/responses (different shape: "input" instead of "messages", "reasoning":{"effort":...},
//!   output blocks). We deliberately use chat.completions for OpenAI-compat clients and canonical mapping.
//! - reasoning_effort: for chat.completions surface, top-level string "reasoning_effort": "none"|"low"|"medium"|"high"
//!   (default "low" on supported models like grok-4.3). In Responses it is nested "reasoning":{"effort":...}.
//!   CanonicalReasoning.effort is mapped to the chat.completions form. Some models reject presence_penalty etc
//!   when reasoning is active.
//! - Tools: full function calling (standard OpenAI tool schema). Built-in server-side tools (web_search, x_search,
//!   code_execution, collections_search, mcp) are also supported by xAI; they can be passed via provider_extras
//!   or as special tool entries (e.g. {"type":"web_search"}). Custom tools use {"type":"function", "function":{...}}.
//!   search_parameters (legacy) is deprecated in favor of tools.
//! - Streaming: SSE on ?stream=true (or "stream":true in body), deltas for content + tool_calls (incremental args).
//!   Not yet exposed via the LlmProvider trait (trait only has non-stream send today); wire is ready.
//! - Usage: prompt_tokens / completion_tokens + details (cached_tokens in prompt, reasoning_tokens in completion).
//!   We map the main counters; reasoning_tokens are part of billing but surfaced in provider_extras on future
//!   CanonicalResponse extensions if needed.
//! - Other xAI extensions passed via CanonicalRequest.provider_extras (e.g. service_tier, search_parameters,
//!   deferred, parallel_tool_calls, response_format for json_schema, etc.). Merged at top level of the wire body.
//!   Official values for service_tier: "default" | "priority" (affects scheduling/billing per docs.x.ai). No
//!   other "gate" headers (e.g. no xai-*, no special enterprise tokens on wire for basic use). Built-in tools
//!   like web_search are passed as top-level tool objects with {"type": "web_search", ...options...} (or via
//!   provider_extras["tools"] in some flows); /responses surface exists for stateful/agentic use cases but
//!   chat/completions remains the primary for canonical OpenAI-compat.
//! - No Replacements or Stats are *required* inside the provider (they are cross-cutting and applied in
//!   frontends/bin layers per omni design). However this crate depends on omni-common and lightly exercises
//!   Replacements::empty() + apply paths inside the mappers as a hook demonstration. In a fuller integration
//!   the ctor would accept `Arc<Replacements>` (and/or Stats handle) from omni-common and apply prompt-scope
//!   rules to message texts/tool surfaces before serialization, and response-scope rules to returned content +
//!   tool names/arguments after. See omni-common::replacements and omni-common::stats.
//!
//! Production-quality prototype: typed wire structs, robust error mapping to ProviderError, timeouts suitable
//! for long reasoning traces (5min), tracing, basic id synthesis for tool_calls, support for tools +
//! reasoning_effort + provider_extras + all core sampling params. Unit tests cover the (de)serialization mappers
//! with no network.

use async_trait::async_trait;
use futures_util::StreamExt;
use omni_common::{GrokCredentials, Replacements};
use omni_core::{
    CanonicalBlock, CanonicalContent, CanonicalReasoning, CanonicalRequest, CanonicalResponse,
    CanonicalStream, CanonicalStreamEvent, CanonicalToolCall, CanonicalToolChoice, CanonicalUsage,
    LlmProvider, ProviderError,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{debug, error, warn};

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

/// The Grok / xAI provider. Holds a reqwest client.
/// Credentials are loaded fresh per request using the same technique as the
/// original Claude Code Provider (see omni-common::credentials::GrokCredentials
/// and docs/grok-gate.md).
///
/// The loader looks for $XAI_CREDENTIALS_PATH or ~/.xai/.credentials.json and
/// re-reads on every send (never cached). This picks up key rotations or
/// refreshes without restarting the process — exactly like CCP does for
/// ~/.claude/.credentials.json.
#[derive(Debug)]
pub struct GrokProvider {
    client: Client,
    // Stored key is only for explicit ctor / test helpers.
    // Normal path always prefers fresh load from the credentials file.
    api_key: Option<String>,
    base_url: String,
}

impl GrokProvider {
    /// Create a provider (client only).
    /// Key is not required here; the normal send path will load fresh from the
    /// credentials file (or fall back to XAI_API_KEY env for compatibility).
    /// Use `new(Some(key))` only for explicit/testing scenarios.
    ///
    /// The client is configured with a long timeout (reasoning models can think for minutes)
    /// and a descriptive User-Agent.
    pub fn new(api_key: Option<String>) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent("omni-grok/0.1 (+https://github.com/omni-llm-provider; rust-reqwest)")
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| {
                ProviderError::Other(anyhow::Error::msg(format!(
                    "failed to build http client: {}",
                    e
                )))
            })?;

        // Honor XAI_API_KEY when no explicit key is passed. The module docs and
        // the binaries' startup messages both advertise XAI_API_KEY as a valid
        // credential source, and `resolve_api_key` falls back to this ctor key
        // when the creds file is absent — but that fallback is only reachable if
        // we actually capture the env var here. Without this, setting only
        // XAI_API_KEY (the documented env path) yields an auth error.
        let api_key = api_key.or_else(|| {
            std::env::var("XAI_API_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty())
        });

        Ok(Self {
            client,
            api_key,
            base_url: DEFAULT_BASE_URL.to_owned(),
        })
    }

    /// Override the base URL (useful for tests or proxies). Chainable.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Test-only constructor (no env, custom client possible in future).
    /// Not under cfg(test) so bin integration tests and other dependents can construct
    /// a mock instance for routing/wrapper tests (while production still uses new()).
    pub fn new_for_test(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: Some(api_key.into()),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }

    /// Returns the configured upstream base (without trailing slash).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Resolve the effective bearer key the same way for every request: load the credentials file
    /// fresh (never cached, so key rotations are picked up), warning-but-continuing if it reports
    /// expired, and falling back to the ctor/env key only when the file load fails. Returns an
    /// `Auth` error when neither source yields a key. Shared by `send` and `send_stream` so the two
    /// paths cannot drift in how they authenticate.
    async fn resolve_api_key(&self) -> Result<String, ProviderError> {
        match GrokCredentials::load_fresh_async(&GrokCredentials::default_path()).await {
            Ok(creds) => {
                if let Err(e) = creds.check_expired() {
                    warn!(error = %e, "grok creds reported expired (continuing with file key)");
                }
                Ok(creds.api_key)
            }
            Err(e) => {
                // Fallback to the key we were constructed with (explicit or XAI_API_KEY env)
                // for backward compatibility with existing Grok users who only set the env var.
                if let Some(k) = &self.api_key {
                    debug!(error = %e, "no grok creds file (or load failed); falling back to ctor/env key");
                    Ok(k.clone())
                } else {
                    Err(ProviderError::Auth(format!(
                        "failed to load Grok credentials from file and no explicit/env key: {}",
                        e
                    )))
                }
            }
        }
    }
}

/// Map a CanonicalRequest (after light replacements hook) to the JSON body for xAI /v1/chat/completions.
/// OpenAI-compatible shape + xAI extensions (reasoning_effort, provider_extras passthrough).
fn to_xai_chat_request(req: &CanonicalRequest, repl: &Replacements) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    for m in &req.messages {
        match &m.content {
            CanonicalContent::Text(t) => {
                messages.push(json!({ "role": m.role, "content": repl.apply_prompt(t) }));
            }
            CanonicalContent::Blocks(blocks) => {
                // OpenAI/xAI wire shape for multi-turn tools: each tool RESULT is
                // its own `role:"tool"` message; the text + tool CALLS in the
                // turn go in one message for `m.role`. A block message may mix
                // both (e.g. an assistant turn plus its results), so emit every
                // block rather than dropping siblings.
                let mut text = String::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                for b in blocks {
                    match b {
                        CanonicalBlock::Text(t) => text.push_str(&repl.apply_prompt(t)),
                        CanonicalBlock::ToolUse {
                            id,
                            name,
                            arguments,
                        } => {
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": { "name": name, "arguments": arguments }
                            }));
                        }
                        CanonicalBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": repl.apply_prompt(content)
                            }));
                        }
                    }
                }
                // Emit the role's own message only when it carries text or tool
                // calls (a message of pure tool results adds nothing here). When
                // there are tool_calls but no text, content is null per the
                // OpenAI contract; omit an empty tool_calls array otherwise.
                if !text.is_empty() || !tool_calls.is_empty() {
                    let mut msg = serde_json::Map::new();
                    msg.insert("role".into(), json!(m.role));
                    msg.insert(
                        "content".into(),
                        if text.is_empty() {
                            Value::Null
                        } else {
                            json!(text)
                        },
                    );
                    if !tool_calls.is_empty() {
                        msg.insert("tool_calls".into(), json!(tool_calls));
                    }
                    messages.push(Value::Object(msg));
                }
            }
        }
    }

    let tools: Option<Vec<Value>> = req.tools.as_ref().map(|ts| {
        ts.iter()
            .map(|t| {
                let desc = t.description.as_ref().map(|d| repl.apply_prompt(d));
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,  // note: if tool-name masking rules exist they were applied upstream or will be via repl on name too if caller chose
                        "description": desc,
                        "parameters": t.parameters.clone()
                    }
                })
            })
            .collect()
    });

    let mut body = json!({
        "model": req.model,
        "messages": messages,
        "stream": false,
    });

    if let Some(ts) = tools {
        body["tools"] = json!(ts);
    }

    if let Some(tc) = &req.tool_choice {
        let v = match tc {
            CanonicalToolChoice::Auto => json!("auto"),
            CanonicalToolChoice::Required => json!("required"),
            CanonicalToolChoice::Specific { name } => {
                json!({"type": "function", "function": {"name": name}})
            }
            CanonicalToolChoice::None => json!("none"),
        };
        body["tool_choice"] = v;
    }

    if let Some(mt) = req.max_tokens {
        // xAI prefers max_completion_tokens (does not count internal reasoning/function tokens)
        body["max_completion_tokens"] = json!(mt);
    }
    if let Some(t) = req.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(p) = req.top_p {
        body["top_p"] = json!(p);
    }

    // Map canonical reasoning -> xAI chat.completions form (top level for this surface)
    if let Some(CanonicalReasoning {
        effort: Some(eff), ..
    }) = &req.reasoning
        && !eff.is_empty()
    {
        body["reasoning_effort"] = json!(eff);
    }

    // Passthrough any xAI-specific (search_parameters, service_tier, response_format, parallel_tool_calls, etc.)
    // Extras win on collision (caller responsibility).
    if let Some(extras) = &req.provider_extras
        && let Some(obj) = extras.as_object()
    {
        for (k, v) in obj {
            body[k] = v.clone();
        }
    }

    // Light hook demonstration for omni-common replacements on the *structured* prompt surface.
    // (Real rules for tool names etc. are typically applied by the frontend before producing CanonicalRequest,
    // or the provider ctor would be given a live Replacements instance instead of always empty().)
    body
}

/// Map a CanonicalRequest to the JSON body for a *streaming* xAI /v1/chat/completions call.
/// Reuses `to_xai_chat_request` (identical message/tool/sampling mapping + replacements hook) and
/// then flips `stream` to true. `stream_options.include_usage` asks xAI to emit one final chunk
/// carrying the `usage` object (otherwise streamed responses omit token accounting entirely), which
/// the parser turns into a terminal `CanonicalStreamEvent::Usage`.
fn to_xai_chat_stream_request(req: &CanonicalRequest, repl: &Replacements) -> Value {
    let mut body = to_xai_chat_request(req, repl);
    body["stream"] = json!(true);
    body["stream_options"] = json!({ "include_usage": true });
    body
}

/// Internal typed response shapes (subset of xAI chat.completions response for robust mapping).
/// Many fields are parsed for wire fidelity / future use (e.g. service_tier, fingerprints, detailed token breakdowns)
/// but not yet surfaced in CanonicalResponse; allow(dead_code) keeps the compiler clean per project rules
/// while we keep the full shapes (not a minimal projection).
#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct XaiChatCompletion {
    id: Option<String>,
    object: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    choices: Option<Vec<XaiChoice>>,
    usage: Option<XaiUsage>,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct XaiChoice {
    index: Option<i32>,
    message: Option<XaiAssistantMessage>,
    finish_reason: Option<String>,
    logprobs: Option<Value>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct XaiAssistantMessage {
    role: Option<String>,
    content: Option<String>,
    refusal: Option<Value>,
    tool_calls: Option<Vec<XaiToolCall>>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct XaiToolCall {
    id: Option<String>,
    #[serde(rename = "type")]
    type_: Option<String>,
    function: Option<XaiFunctionCall>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct XaiFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct XaiUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    prompt_tokens_details: Option<XaiPromptDetails>,
    completion_tokens_details: Option<XaiCompletionDetails>,
    num_sources_used: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct XaiPromptDetails {
    cached_tokens: Option<u64>,
    text_tokens: Option<u64>,
    audio_tokens: Option<u64>,
    image_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct XaiCompletionDetails {
    reasoning_tokens: Option<u64>,
    audio_tokens: Option<u64>,
    accepted_prediction_tokens: Option<u64>,
    rejected_prediction_tokens: Option<u64>,
}

/// Map xAI chat completion JSON response to canonical form. Applies inbound replacements hook on text + tool surfaces.
fn from_xai_chat_response(raw: XaiChatCompletion, repl: &Replacements) -> CanonicalResponse {
    let model = raw.model.unwrap_or_else(|| "unknown".to_string());

    let (content, tool_calls, finish_reason) =
        if let Some(ch) = raw.choices.and_then(|mut c| c.drain(..).next()) {
            let msg = ch.message.unwrap_or_default();
            let raw_content = msg.content.unwrap_or_default();
            let content = repl.apply_response(&raw_content);

            let tcs: Vec<CanonicalToolCall> = msg
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .enumerate()
                .map(|(i, tc)| {
                    let func = tc.function.unwrap_or_default();
                    let raw_name = func.name.unwrap_or_default();
                    let raw_args = func.arguments.unwrap_or_default();
                    CanonicalToolCall {
                        // xAI (OpenAI-compat) normally supplies stable ids like "call_xxx"; synthesize if absent.
                        id: tc.id.unwrap_or_else(|| format!("call_{}_{}", i, raw_name)),
                        name: repl.apply_response(&raw_name),
                        arguments: repl.apply_response(&raw_args),
                    }
                })
                .collect();

            (content, tcs, ch.finish_reason)
        } else {
            (String::new(), vec![], None)
        };

    let usage = if let Some(u) = raw.usage {
        CanonicalUsage {
            input_tokens: u.prompt_tokens.unwrap_or(0),
            output_tokens: u.completion_tokens.unwrap_or(0),
            cache_read: u
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0),
            cache_creation: 0,
        }
    } else {
        CanonicalUsage::default()
    };

    CanonicalResponse {
        model,
        content,
        tool_calls,
        finish_reason,
        usage,
        // (no provider_extras field on CanonicalResponse today; system_fingerprint etc. logged at debug if needed)
    }
}

// --- Streaming (SSE) wire shapes + parsing -------------------------------------------------
//
// xAI streams OpenAI-style Server-Sent Events: each event is a line `data: {json}` (one chat
// completion *chunk*), and the stream terminates with the sentinel `data: [DONE]`. A chunk's
// `choices[0].delta` carries incremental `content` and/or `tool_calls`; `finish_reason` becomes
// non-null on the chunk that closes generation. With `stream_options.include_usage` xAI appends a
// trailing chunk whose `choices` is empty but which carries the cumulative `usage`.
//
// JUDGMENT CALL (tool_call delta shape): xAI follows OpenAI's incremental tool-call convention.
// The first chunk for a given tool call sets `index`, `id`, and `function.name`; subsequent chunks
// for the same `index` carry only `function.arguments` fragments (and null id/name). We map each
// raw tool-call delta straight onto `CanonicalStreamEvent::ToolCallDelta { index, id, name,
// arguments_delta }` without accumulating, because the canonical contract documents exactly this
// incremental shape (consumers concatenate by index). `index` is required by canonical; if a chunk
// ever omits it we default to 0 (single tool call), which matches the common non-parallel case.

#[derive(Debug, Deserialize)]
struct XaiStreamChunk {
    choices: Option<Vec<XaiStreamChoice>>,
    usage: Option<XaiUsage>,
}

#[derive(Debug, Deserialize)]
struct XaiStreamChoice {
    delta: Option<XaiStreamDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct XaiStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<XaiStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct XaiStreamToolCall {
    index: Option<u32>,
    id: Option<String>,
    function: Option<XaiStreamFunction>,
}

#[derive(Debug, Deserialize, Default)]
struct XaiStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

/// Parse the JSON payload of a single SSE `data:` frame (the `[DONE]` sentinel is handled by the
/// caller, not here) into zero or more canonical stream events, in emission order.
///
/// One chunk can legitimately yield several events: a `content` text delta, one `ToolCallDelta`
/// per entry in `tool_calls`, and/or a `Usage` event from the trailing include_usage chunk. A
/// non-null `finish_reason` is *not* emitted here; the caller remembers it and emits a single
/// terminal `Finish` at `[DONE]` (per the canonical contract: exactly one terminal Finish).
///
/// Returns a Vec rather than the single-event `Option` sketched in the task because a chunk is
/// genuinely multi-event; an empty Vec means "nothing to surface from this frame" (e.g. a role-only
/// opening delta). A malformed JSON frame yields a single `Err(Upstream(..))` so the stream fails
/// loud instead of silently dropping data.
fn parse_grok_sse_frame(data: &str) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
    let chunk: XaiStreamChunk = match serde_json::from_str(data) {
        Ok(c) => c,
        Err(e) => {
            return vec![Err(ProviderError::Upstream(format!(
                "failed to decode xAI stream chunk: {e}: {data}"
            )))];
        }
    };

    let mut events: Vec<Result<CanonicalStreamEvent, ProviderError>> = Vec::new();

    if let Some(choice) = chunk.choices.and_then(|mut c| c.drain(..).next()) {
        if let Some(delta) = choice.delta {
            if let Some(text) = delta.content
                && !text.is_empty()
            {
                events.push(Ok(CanonicalStreamEvent::TextDelta(text)));
            }
            if let Some(tcs) = delta.tool_calls {
                for tc in tcs {
                    let func = tc.function.unwrap_or_default();
                    events.push(Ok(CanonicalStreamEvent::ToolCallDelta {
                        index: tc.index.unwrap_or(0),
                        id: tc.id,
                        name: func.name,
                        arguments_delta: func.arguments.unwrap_or_default(),
                    }));
                }
            }
        }
        // finish_reason is remembered by the driver and emitted once at [DONE]; not surfaced here.
        let _ = choice.finish_reason;
    }

    if let Some(u) = chunk.usage {
        events.push(Ok(CanonicalStreamEvent::Usage(CanonicalUsage {
            input_tokens: u.prompt_tokens.unwrap_or(0),
            output_tokens: u.completion_tokens.unwrap_or(0),
            cache_read: u
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0),
            cache_creation: 0,
        })));
    }

    events
}

/// Extract the `finish_reason` from a single SSE `data:` frame, if present and non-null.
/// The driver records the last one seen and emits it in the terminal `Finish` at `[DONE]`.
fn finish_reason_from_frame(data: &str) -> Option<String> {
    let chunk: XaiStreamChunk = serde_json::from_str(data).ok()?;
    chunk
        .choices
        .and_then(|mut c| c.drain(..).next())
        .and_then(|ch| ch.finish_reason)
}

/// Incremental SSE line buffer. Bytes arrive from `reqwest::bytes_stream()` in arbitrary chunks; a
/// single `data: {json}` line (and the JSON inside it) can be split across two byte chunks, so we
/// accumulate into a `String` and only hand back *complete* lines (those terminated by `\n`). Any
/// trailing partial line stays buffered for the next byte chunk.
#[derive(Default)]
struct SseBuffer {
    buf: String,
}

impl SseBuffer {
    /// Feed a UTF-8 string slice of freshly received bytes; returns each complete line (newline
    /// stripped, including the trailing `\r` from CRLF framing) now available. A line not yet
    /// terminated by `\n` is retained internally until more bytes arrive.
    fn push(&mut self, s: &str) -> Vec<String> {
        self.buf.push_str(s);
        let mut lines = Vec::new();
        while let Some(nl) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=nl).collect();
            // Strip the trailing \n and any \r (SSE uses CRLF or LF).
            lines.push(line.trim_end_matches(['\r', '\n']).to_string());
        }
        lines
    }
}

/// Classify a single complete SSE line. Returns `None` for blank lines, comments (`:` prefix), and
/// non-`data:` fields (e.g. `event:`), which carry nothing the canonical stream needs.
enum SseLine {
    Done,
    Data(String),
    Ignore,
}

fn classify_sse_line(line: &str) -> SseLine {
    let trimmed = line.trim_end();
    if trimmed.is_empty() || trimmed.starts_with(':') {
        return SseLine::Ignore;
    }
    if let Some(payload) = trimmed.strip_prefix("data:") {
        let payload = payload.trim();
        if payload == "[DONE]" {
            return SseLine::Done;
        }
        return SseLine::Data(payload.to_string());
    }
    SseLine::Ignore
}

#[async_trait]
impl LlmProvider for GrokProvider {
    fn id(&self) -> &'static str {
        "grok"
    }

    async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError> {
        debug!(
            provider = "grok",
            model = %req.model,
            n_msgs = req.messages.len(),
            n_tools = req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            has_reasoning = req.reasoning.is_some(),
            "sending to xAI"
        );

        // Hook point (using omni-common): replacements applied at prompt boundary inside the provider.
        // In real deployment the Replacements would be loaded once in the binary and injected here.
        let repl = Replacements::empty();
        let body = to_xai_chat_request(&req, &repl);

        let url = format!("{}/chat/completions", self.base_url);
        debug!(%url, "POST xAI chat completions");

        // Fresh credentials load (the "Grok gate" technique, copied from CCP).
        // See docs/grok-gate.md and omni-common::credentials::GrokCredentials.
        let effective_key = self.resolve_api_key().await?;

        let http_resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", effective_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Upstream(format!("network error calling xAI: {}", e)))?;

        let status = http_resp.status();
        if !status.is_success() {
            let err_body = http_resp
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());
            error!(%status, body = %err_body, "xAI upstream error");
            return Err(ProviderError::Upstream(format!(
                "xAI {}: {}",
                status, err_body
            )));
        }

        let raw: XaiChatCompletion = http_resp.json().await.map_err(|e| {
            ProviderError::Upstream(format!("failed to decode xAI response: {}", e))
        })?;

        debug!(
            model = %raw.model.as_deref().unwrap_or("unknown"),
            choices = raw.choices.as_ref().map(|c| c.len()).unwrap_or(0),
            "xAI response received"
        );

        let canon = from_xai_chat_response(raw, &repl);

        // Final inbound hook demonstration (response scope from omni-common).
        // (content + tool names/args already processed in from_... using the same repl)

        // If the caller supplied provider_extras that requested something we can surface, it would live here.
        // For now the canonical shape is the contract.

        Ok(canon)
    }

    /// Native SSE streaming against xAI /v1/chat/completions.
    ///
    /// Overrides the trait default (which buffers a whole `send`) so callers get incremental
    /// deltas as xAI emits them. The HTTP request is issued *inside* the returned stream (via
    /// `async_stream::stream!`) so the call site gets the stream immediately and any upstream
    /// failure surfaces as the first `Err` item rather than from the `send_stream` call itself.
    async fn send_stream(&self, req: CanonicalRequest) -> Result<CanonicalStream, ProviderError> {
        debug!(
            provider = "grok",
            model = %req.model,
            n_msgs = req.messages.len(),
            n_tools = req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            has_reasoning = req.reasoning.is_some(),
            "streaming to xAI"
        );

        // Same prompt-scope replacements seam as send() (Replacements::empty() hook).
        let repl = Replacements::empty();
        let body = to_xai_chat_stream_request(&req, &repl);
        let url = format!("{}/chat/completions", self.base_url);

        // Resolve the key fresh exactly like send() (shared helper; never cached).
        let effective_key = self.resolve_api_key().await?;
        let client = self.client.clone();

        let stream = async_stream::stream! {
            let send_result = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", effective_key))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .json(&body)
                .send()
                .await;

            let http_resp = match send_result {
                Ok(r) => r,
                Err(e) => {
                    yield Err(ProviderError::Upstream(format!("network error calling xAI: {}", e)));
                    return;
                }
            };

            let status = http_resp.status();
            if !status.is_success() {
                // Read the error body first, same as the non-stream path.
                let err_body = http_resp
                    .text()
                    .await
                    .unwrap_or_else(|_| "<no body>".to_string());
                error!(%status, body = %err_body, "xAI upstream stream error");
                yield Err(ProviderError::Upstream(format!("xAI {}: {}", status, err_body)));
                return;
            }

            // Consume the raw byte stream, reframing into SSE lines (a JSON object may span
            // multiple byte chunks; SseBuffer holds partial lines across chunk boundaries) and
            // mapping each `data:` frame to canonical events. The last non-null finish_reason is
            // remembered and emitted once as the terminal Finish at `data: [DONE]`.
            let mut bytes = http_resp.bytes_stream();
            let mut sse = SseBuffer::default();
            let mut finish_reason: Option<String> = None;
            let mut done = false;

            while let Some(chunk) = bytes.next().await {
                let chunk = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(ProviderError::Upstream(format!("xAI stream read error: {}", e)));
                        return;
                    }
                };
                // xAI SSE payloads are UTF-8; tolerate any split multi-byte sequence by lossy decode
                // (frame boundaries are at `\n`, so SseBuffer only releases complete lines anyway).
                let text = String::from_utf8_lossy(&chunk);
                for line in sse.push(&text) {
                    match classify_sse_line(&line) {
                        SseLine::Ignore => {}
                        SseLine::Done => {
                            done = true;
                            break;
                        }
                        SseLine::Data(payload) => {
                            if let Some(fr) = finish_reason_from_frame(&payload) {
                                finish_reason = Some(fr);
                            }
                            for ev in parse_grok_sse_frame(&payload) {
                                let is_err = ev.is_err();
                                yield ev;
                                if is_err {
                                    return;
                                }
                            }
                        }
                    }
                }
                if done {
                    break;
                }
            }

            // Terminal Finish (exactly one), carrying the remembered finish_reason. Emitted even if
            // the upstream closed without an explicit `[DONE]` so consumers always see a terminator.
            yield Ok(CanonicalStreamEvent::Finish { finish_reason });
        };

        Ok(Box::pin(stream))
    }
}

// Keep the original free fn for any legacy direct callers (returns the provider id).
pub fn provider_id() -> &'static str {
    "grok"
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::{
        CanonicalContent, CanonicalMessage, CanonicalReasoning, CanonicalTool, CanonicalToolChoice,
    };
    use serde_json::json;

    fn empty_repl() -> Replacements {
        Replacements::empty()
    }

    // Serialize credential tests that mutate process env (XAI_CREDENTIALS_PATH) to avoid
    // races when cargo runs tests in parallel (default >1 threads). Other tests unaffected.
    static CRED_ENV_LOCK: ::std::sync::Mutex<()> = ::std::sync::Mutex::new(());

    #[test]
    fn test_to_xai_basic() {
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![
                CanonicalMessage {
                    role: "system".into(),
                    content: CanonicalContent::Text("You are Grok.".into()),
                },
                CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("Hi".into()),
                },
            ],
            tools: None,
            tool_choice: None,
            max_tokens: Some(128),
            temperature: Some(0.5),
            top_p: None,
            reasoning: Some(CanonicalReasoning {
                effort: Some("high".into()),
                budget_tokens: None,
            }),
            metadata: Default::default(),
            provider_extras: Some(json!({"service_tier": "priority"})),
        };

        let body = to_xai_chat_request(&req, &empty_repl());
        assert_eq!(body["model"], "grok-4.3");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(body["max_completion_tokens"], 128);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["service_tier"], "priority");
        assert_eq!(body["stream"], false);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_to_xai_tools_and_choice() {
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("use tool".into()),
            }],
            tools: Some(vec![CanonicalTool {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                parameters: json!({"type":"object","properties":{}}),
            }]),
            tool_choice: Some(CanonicalToolChoice::Specific {
                name: "get_weather".into(),
            }),
            max_tokens: None,
            temperature: None,
            top_p: None,
            reasoning: None,
            metadata: Default::default(),
            provider_extras: None,
        };

        let body = to_xai_chat_request(&req, &empty_repl());
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(body["tool_choice"]["function"]["name"], "get_weather");
    }

    #[test]
    fn mixed_block_message_emits_tool_result_and_assistant_without_dropping() {
        // WHY: an assistant turn that mixes Text with a ToolUse must keep BOTH:
        // the text becomes the assistant message `content` and the call becomes a
        // `tool_calls` entry. A prior bug dropped the Text sibling when a block
        // message also produced a tool message, silently losing the model's
        // reasoning/answer. A following tool result is its own `role:"tool"`
        // message keyed by tool_call_id.
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![
                CanonicalMessage {
                    role: "assistant".into(),
                    content: CanonicalContent::Blocks(vec![
                        CanonicalBlock::Text("thinking".into()),
                        CanonicalBlock::ToolUse {
                            id: "c1".into(),
                            name: "f".into(),
                            arguments: "{}".into(),
                        },
                    ]),
                },
                CanonicalMessage {
                    role: "tool".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                        tool_use_id: "c1".into(),
                        content: "R".into(),
                        is_error: false,
                    }]),
                },
            ],
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &empty_repl());
        let messages = body["messages"].as_array().unwrap();

        // The assistant message keeps its Text sibling as `content` AND carries
        // the tool call (the sibling is NOT dropped).
        let assistant = messages
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("assistant message must be present");
        assert_eq!(
            assistant["content"], "thinking",
            "the Text sibling must survive as the assistant content"
        );
        let calls = assistant["tool_calls"]
            .as_array()
            .expect("assistant must carry tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "c1");

        // The tool result is a separate role:"tool" message keyed by id.
        let tool_msg = messages
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("tool result message must be present");
        assert_eq!(tool_msg["tool_call_id"], "c1");
        assert_eq!(tool_msg["content"], "R");
    }

    #[test]
    fn plain_assistant_block_message_omits_empty_tool_calls_key() {
        // WHY: a plain assistant block message (no tool calls) must NOT emit an
        // empty `tool_calls` array; the OpenAI contract only includes the key
        // when the assistant actually called tools, and an empty array can be
        // rejected upstream.
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![CanonicalMessage {
                role: "assistant".into(),
                content: CanonicalContent::Blocks(vec![CanonicalBlock::Text("hi".into())]),
            }],
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &empty_repl());
        let messages = body["messages"].as_array().unwrap();
        let assistant = messages
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("assistant message must be present");
        assert_eq!(assistant["content"], "hi");
        assert!(
            assistant.get("tool_calls").is_none(),
            "no tool_calls key when the assistant called no tools"
        );
    }

    #[test]
    fn test_from_xai_basic() {
        let raw = XaiChatCompletion {
            model: Some("grok-4.3".into()),
            choices: Some(vec![XaiChoice {
                message: Some(XaiAssistantMessage {
                    content: Some("Hello from Grok".into()),
                    tool_calls: None,
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
                ..Default::default()
            }]),
            usage: Some(XaiUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(5),
                prompt_tokens_details: Some(XaiPromptDetails {
                    cached_tokens: Some(3),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        let canon = from_xai_chat_response(raw, &empty_repl());
        assert_eq!(canon.model, "grok-4.3");
        assert_eq!(canon.content, "Hello from Grok");
        assert!(canon.tool_calls.is_empty());
        assert_eq!(canon.finish_reason.as_deref(), Some("stop"));
        assert_eq!(canon.usage.input_tokens, 10);
        assert_eq!(canon.usage.output_tokens, 5);
        assert_eq!(canon.usage.cache_read, 3);
    }

    #[test]
    fn test_from_xai_tool_calls_and_repl() {
        // Demonstrate inbound replacement hook (response scope)
        let repl = Replacements::parse(
            r#"rule = [ { scope = "response", search = "get_weather", replace = "get_weather_masked" } ]"#,
        )
        .unwrap();

        let raw = XaiChatCompletion {
            model: Some("grok-beta".into()),
            choices: Some(vec![XaiChoice {
                message: Some(XaiAssistantMessage {
                    content: Some("I will call it".into()),
                    tool_calls: Some(vec![XaiToolCall {
                        id: Some("call_123".into()),
                        function: Some(XaiFunctionCall {
                            name: Some("get_weather".into()),
                            arguments: Some(r#"{"city":"sf"}"#.into()),
                        }),
                        ..Default::default()
                    }]),
                    ..Default::default()
                }),
                finish_reason: Some("tool_calls".into()),
                ..Default::default()
            }]),
            usage: None,
            ..Default::default()
        };

        let canon = from_xai_chat_response(raw, &repl);
        assert_eq!(canon.content, "I will call it");
        assert_eq!(canon.tool_calls.len(), 1);
        assert_eq!(canon.tool_calls[0].id, "call_123");
        assert_eq!(canon.tool_calls[0].name, "get_weather_masked");
        assert_eq!(canon.tool_calls[0].arguments, r#"{"city":"sf"}"#); // args not masked by this rule
        assert_eq!(canon.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn test_provider_id_and_ctor() {
        assert_eq!(provider_id(), "grok");
        // new(None) now succeeds (the "same technique as CCP": fresh file load with env override
        // happens inside send(), not at construction time). This lets the binary start without
        // the key and pick it up (or pick up a rotated key) on the first request.
        let p = GrokProvider::new(None)
            .expect("new without key must succeed after credential extraction");
        assert_eq!(p.id(), "grok");

        let p2 = GrokProvider::new_for_test("xai-test-123", "https://api.x.ai/v1");
        assert_eq!(p2.id(), "grok");
        assert_eq!(p2.base_url(), "https://api.x.ai/v1");
    }

    #[test]
    fn test_replacements_hook_in_request() {
        let repl = Replacements::parse(
            r#"rule = [ { scope = "prompt", search = "secret", replace = "REDACTED" } ]"#,
        )
        .unwrap();
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("tell secret".into()),
            }],
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &repl);
        let msg0 = &body["messages"][0];
        assert_eq!(msg0["content"], "tell REDACTED");
    }

    // --- additional comprehensive mapper + integration coverage ---

    #[test]
    fn test_to_xai_with_tools_and_extras_and_reasoning() {
        let req = CanonicalRequest {
            model: "grok-4".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("search".into()),
            }],
            tools: Some(vec![CanonicalTool {
                name: "web".into(),
                description: Some("web".into()),
                parameters: serde_json::json!({"type":"object"}),
            }]),
            tool_choice: Some(CanonicalToolChoice::Auto),
            max_tokens: Some(256),
            temperature: None,
            top_p: Some(0.9),
            reasoning: Some(CanonicalReasoning {
                effort: Some("medium".into()),
                budget_tokens: Some(100),
            }),
            metadata: Default::default(),
            provider_extras: Some(serde_json::json!({"service_tier": "standard"})),
        };
        let body = to_xai_chat_request(&req, &empty_repl());
        assert_eq!(body["model"], "grok-4");
        assert!(body.get("tools").is_some());
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["max_completion_tokens"], 256);
        let top_p = body["top_p"].as_f64().unwrap();
        assert!(
            (top_p - 0.9).abs() < 1e-6,
            "top_p float json approx: {}",
            top_p
        );
        assert_eq!(body["reasoning_effort"], "medium");
        assert_eq!(body["service_tier"], "standard");
    }

    #[test]
    fn test_from_xai_with_details_and_refusal() {
        let raw = XaiChatCompletion {
            model: Some("grok-3".into()),
            choices: Some(vec![XaiChoice {
                message: Some(XaiAssistantMessage {
                    content: Some("ok".into()),
                    refusal: Some(serde_json::json!("policy")),
                    tool_calls: None,
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
                ..Default::default()
            }]),
            usage: Some(XaiUsage {
                prompt_tokens: Some(2),
                completion_tokens: Some(1),
                prompt_tokens_details: Some(XaiPromptDetails {
                    text_tokens: Some(2),
                    ..Default::default()
                }),
                completion_tokens_details: Some(XaiCompletionDetails {
                    reasoning_tokens: Some(10),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let canon = from_xai_chat_response(raw, &empty_repl());
        assert_eq!(canon.content, "ok");
        assert_eq!(canon.usage.input_tokens, 2);
        // note: reasoning_tokens not mapped into core usage yet
    }

    #[tokio::test]
    async fn test_send_mocked_upstream_error() {
        // Use impossible port as "mock" for upstream failure (no extra crates, exercises error path + ProviderError)
        let p = GrokProvider::new_for_test("xai-dummy", "http://127.0.0.1:1");
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };
        let err = p.send(req).await.unwrap_err();
        match err {
            ProviderError::Upstream(s) => {
                assert!(s.contains("error calling xAI") || s.contains("connection"))
            }
            _ => panic!("expected Upstream error for mocked bad port"),
        }
    }

    #[tokio::test]
    async fn test_send_real_if_key_present() {
        // "real if key, but mocked" -- when XAI_API_KEY in env, exercises full send path against live xAI (uses real key, real net)
        // otherwise returns early (mocked/no-op for CI without secrets)
        // Snapshot the key UNDER CRED_ENV_LOCK so this live test cannot read a
        // transiently-mutated env: the credential tests set/clear XAI_API_KEY +
        // XAI_CREDENTIALS_PATH while holding this lock, and a concurrent read
        // here could otherwise catch a blanked key and fail auth. We take the
        // lock only to copy the key into a local, then drop it before the
        // network await (no lock held across .await; no needless serialization).
        let key = {
            let _guard = CRED_ENV_LOCK.lock().expect("cred lock for live read");
            match std::env::var("XAI_API_KEY") {
                Ok(k) if !k.trim().is_empty() => k,
                _ => {
                    eprintln!(
                        "skipping real grok send test (no XAI_API_KEY set; using mocked behavior)"
                    );
                    return;
                }
            }
        };
        let p = GrokProvider::new(Some(key)).expect("ctor with explicit key");
        let req = CanonicalRequest {
            model: "grok-3-mini".into(), // use a generally available lightweight model for the real probe
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("Reply with the single word: PONG".into()),
            }],
            max_tokens: Some(8),
            ..Default::default()
        };
        let resp = p
            .send(req)
            .await
            .expect("live xAI call must succeed with valid key");
        assert!(!resp.content.trim().is_empty());
        // model may be resolved/echoed by upstream
        assert!(
            resp.usage.input_tokens > 0 || resp.usage.output_tokens > 0 || !resp.content.is_empty()
        );
    }

    // ============================================================
    // EXPANDED SUITE: 10+ new tests for to_xai / from_xai coverage,
    // credentials (file/env/bad), headers (via reqwest builder assert),
    // tool roundtrips + built-in, replacements full, error cases,
    // passthroughs, usage/refusal/citations variants, etc.
    // Mirrors CCP style: mapper unit pins + mocked upstream + conditional real.
    // Uses only existing patterns (bad-port mock, new_for_test, temp fs for creds, no new deps).
    // ============================================================

    #[test]
    fn test_to_xai_all_sampling_combos() {
        // cover all supported sampling + reasoning + max in one + combos
        let base = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };

        // temp only
        let mut r = base.clone();
        r.temperature = Some(0.2);
        r.max_tokens = Some(64);
        let b = to_xai_chat_request(&r, &empty_repl());
        let t = b["temperature"].as_f64().unwrap();
        assert!((t - 0.2).abs() < 1e-6, "temp float json: {}", t);
        assert_eq!(b["max_completion_tokens"], 64);

        // top_p only
        let mut r = base.clone();
        r.top_p = Some(0.95);
        let b = to_xai_chat_request(&r, &empty_repl());
        let tp = b["top_p"].as_f64().unwrap();
        assert!((tp - 0.95).abs() < 1e-6, "top_p float json approx: {}", tp);

        // reasoning only (no sampling)
        let mut r = base.clone();
        r.reasoning = Some(CanonicalReasoning {
            effort: Some("low".into()),
            budget_tokens: Some(50),
        });
        let b = to_xai_chat_request(&r, &empty_repl());
        assert_eq!(b["reasoning_effort"], "low");

        // all together
        let mut r = base.clone();
        r.temperature = Some(1.0);
        r.top_p = Some(1.0);
        r.max_tokens = Some(10);
        r.reasoning = Some(CanonicalReasoning {
            effort: Some("high".into()),
            budget_tokens: None,
        });
        r.provider_extras = Some(json!({"service_tier": "priority"}));
        let b = to_xai_chat_request(&r, &empty_repl());
        assert_eq!(b["temperature"], 1.0);
        assert_eq!(b["max_completion_tokens"], 10);
        assert_eq!(b["reasoning_effort"], "high");
        assert_eq!(b["service_tier"], "priority");
    }

    #[test]
    fn test_to_xai_parallel_tool_calls_response_format_user_seed_stop_n() {
        // these come via provider_extras passthrough (canonical has limited native sampling)
        let req = CanonicalRequest {
            model: "grok-4".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("x".into()),
            }],
            provider_extras: Some(json!({
                "parallel_tool_calls": true,
                "response_format": {"type": "json_object"},
                "user": "u123",
                "seed": 42,
                "stop": ["END"],
                "n": 2
            })),
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &empty_repl());
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["user"], "u123");
        assert_eq!(body["seed"], 42);
        assert_eq!(body["stop"][0], "END");
        assert_eq!(body["n"], 2);
    }

    #[test]
    fn test_to_xai_responses_shape_not_used() {
        // deliberate: we target chat/completions (messages+stream), not /responses (input+reasoning.effort)
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &empty_repl());
        assert!(body.get("input").is_none(), "no responses 'input' shape");
        assert!(body.get("messages").is_some());
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn test_to_xai_built_in_web_search_and_tool_roundtrip() {
        // built-in via extras (overwrites or provides the tools array); function tools via canonical
        let req = CanonicalRequest {
            model: "grok-4".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("search web".into()),
            }],
            tools: Some(vec![CanonicalTool {
                name: "get_weather".into(),
                description: Some("weather fn".into()),
                parameters: json!({"type":"object"}),
            }]),
            tool_choice: Some(CanonicalToolChoice::Auto),
            provider_extras: Some(json!({
                "tools": [
                    {"type": "web_search", "search_parameters": {"max_results": 5}}
                ]
            })),
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &empty_repl());
        // extras "tools" wins (last write)
        let tools = &body["tools"];
        assert!(tools.is_array());
        assert_eq!(tools[0]["type"], "web_search");
        // function tools were set first but overwritten; for mixed the caller uses extras for builtins
    }

    #[test]
    fn test_to_xai_tool_choice_required() {
        let req = CanonicalRequest {
            model: "grok-beta".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("do it".into()),
            }],
            tools: Some(vec![CanonicalTool {
                name: "calc".into(),
                description: None,
                parameters: json!({}),
            }]),
            tool_choice: Some(CanonicalToolChoice::Required),
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &empty_repl());
        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn test_to_xai_replacements_full_prompt_on_tools() {
        let repl = Replacements::parse(
            r#"rule = [
                { scope = "prompt", search = "SECRET", replace = "REDACTED" },
                { scope = "prompt", search = "weather tool", replace = "wx tool" }
            ]"#,
        )
        .unwrap();
        let req = CanonicalRequest {
            model: "grok-4".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("tell SECRET".into()),
            }],
            tools: Some(vec![CanonicalTool {
                name: "get_wx".into(),
                description: Some("the weather tool here".into()),
                parameters: json!({"type":"object"}),
            }]),
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &repl);
        assert_eq!(body["messages"][0]["content"], "tell REDACTED");
        // desc gets prompt apply (name currently does not per mapper)
        assert_eq!(
            body["tools"][0]["function"]["description"],
            "the wx tool here"
        );
    }

    #[test]
    fn test_from_xai_usage_more_details_and_citations_tolerated() {
        // extra fields like citations (from web_search etc) are tolerated (no deny_unknown); details mapped where possible
        let raw_json = json!({
            "model": "grok-4.3",
            "choices": [{"message": {"content": "searched", "citations": ["https://x.ai/1", "https://x.ai/2"] }, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 7,
                "total_tokens": 27,
                "prompt_tokens_details": {"cached_tokens": 5, "text_tokens": 15},
                "completion_tokens_details": {"reasoning_tokens": 3},
                "num_sources_used": 2
            }
        });
        let raw: XaiChatCompletion = serde_json::from_value(raw_json).unwrap();
        let canon = from_xai_chat_response(raw, &empty_repl());
        assert_eq!(canon.content, "searched");
        assert_eq!(canon.usage.input_tokens, 20);
        assert_eq!(canon.usage.output_tokens, 7);
        assert_eq!(canon.usage.cache_read, 5);
        // citations / num_sources / reasoning_tokens not yet lifted into canonical; tolerated here
    }

    #[test]
    fn test_from_xai_refusal_variants() {
        // string refusal, null, object, absent -- explicit to avoid move issues
        {
            let raw = XaiChatCompletion {
                model: Some("grok".into()),
                choices: Some(vec![XaiChoice {
                    message: Some(XaiAssistantMessage {
                        content: Some("fallback".into()),
                        refusal: Some(json!("policy violation")),
                        tool_calls: None,
                        ..Default::default()
                    }),
                    finish_reason: Some("stop".into()),
                    ..Default::default()
                }]),
                usage: None,
                ..Default::default()
            };
            let _ = from_xai_chat_response(raw, &empty_repl());
        }
        {
            let raw = XaiChatCompletion {
                model: Some("grok".into()),
                choices: Some(vec![XaiChoice {
                    message: Some(XaiAssistantMessage {
                        content: Some("fallback".into()),
                        refusal: None,
                        tool_calls: None,
                        ..Default::default()
                    }),
                    finish_reason: Some("stop".into()),
                    ..Default::default()
                }]),
                usage: None,
                ..Default::default()
            };
            let _ = from_xai_chat_response(raw, &empty_repl());
        }
        {
            let raw = XaiChatCompletion {
                model: Some("grok".into()),
                choices: Some(vec![XaiChoice {
                    message: Some(XaiAssistantMessage {
                        content: Some("fallback".into()),
                        refusal: Some(json!({"type":"other"})),
                        tool_calls: None,
                        ..Default::default()
                    }),
                    finish_reason: Some("stop".into()),
                    ..Default::default()
                }]),
                usage: None,
                ..Default::default()
            };
            let _ = from_xai_chat_response(raw, &empty_repl());
        }
    }

    #[test]
    fn test_from_xai_tool_args_repl_and_output_files_tolerated() {
        let repl = Replacements::parse(
            r#"rule = [ { scope = "response", search = "sf", replace = "SAN_FRANCISCO" } ]"#,
        )
        .unwrap();
        let raw_json = json!({
            "model": "grok",
            "choices": [{
                "message": {
                    "content": "calling",
                    "tool_calls": [{"id":"c1", "type":"function", "function": {"name": "geo", "arguments": "{\"city\":\"sf\"}" }}],
                    "output_files": [{"id":"f1"}]  // tolerated extra
                },
                "finish_reason": "tool_calls"
            }]
        });
        let raw: XaiChatCompletion = serde_json::from_value(raw_json).unwrap();
        let canon = from_xai_chat_response(raw, &repl);
        assert_eq!(canon.tool_calls.len(), 1);
        assert_eq!(canon.tool_calls[0].name, "geo"); // name rule not matching
        assert_eq!(
            canon.tool_calls[0].arguments,
            "{\"city\":\"SAN_FRANCISCO\"}"
        ); // args get response repl
        assert_eq!(canon.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[tokio::test]
    // Holds the env-serialization Mutex across the send().await on purpose: the
    // XAI_CREDENTIALS_PATH env var must stay set while send() reads it fresh.
    // Safe here because #[tokio::test] uses a current-thread runtime, so the task
    // never migrates threads while the guard is held.
    #[allow(clippy::await_holding_lock)]
    async fn test_credentials_file_load_in_send() {
        // prove send path does fresh file load: write dummy creds file, point env, use new(None) so no ctor fallback,
        // hit bad-port upstream -> must be network err (not Auth) proving load succeeded and key taken from file.
        let _guard = CRED_ENV_LOCK.lock().expect("cred lock for env mutate");
        let tmp = ::std::env::temp_dir()
            .join(format!("xai-creds-grok-test-{}.json", ::std::process::id()));
        ::std::fs::write(&tmp, r#"{"apiKey": "xai-from-file-dummy-for-load-test"}"#).unwrap();
        let old = ::std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            ::std::env::set_var("XAI_CREDENTIALS_PATH", tmp.to_str().unwrap());
        }
        let p = GrokProvider::new(None)
            .expect("new(None) succeeds")
            .with_base_url("http://127.0.0.1:1");
        let req = CanonicalRequest {
            model: "grok-3-mini".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("cred file test".into()),
            }],
            ..Default::default()
        };
        let err = p.send(req).await.unwrap_err();
        unsafe {
            if let Some(v) = old {
                ::std::env::set_var("XAI_CREDENTIALS_PATH", v);
            } else {
                ::std::env::remove_var("XAI_CREDENTIALS_PATH");
            }
        }
        match err {
            ProviderError::Upstream(s) => assert!(
                s.contains("error calling xAI") || s.contains("connection"),
                "expected net err after file load: {}",
                s
            ),
            other => panic!(
                "expected Upstream after successful file load, got {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    // See test_credentials_file_load_in_send: env lock held across await is safe
    // on the current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn test_credentials_bad_file_no_key_gives_auth_error() {
        let _guard = CRED_ENV_LOCK.lock().expect("cred lock for env mutate");
        let non = format!("/tmp/xai-no-such-creds-{}.json", ::std::process::id());
        let _ = ::std::fs::remove_file(&non);
        let old = ::std::env::var("XAI_CREDENTIALS_PATH").ok();
        // This test asserts the genuine no-key path; it must clear any ambient
        // XAI_API_KEY (an operator running live has one set) so new(None) has no
        // env fallback to capture. Restored below.
        let old_key = ::std::env::var("XAI_API_KEY").ok();
        unsafe {
            ::std::env::set_var("XAI_CREDENTIALS_PATH", &non);
            ::std::env::remove_var("XAI_API_KEY");
        }
        let p = GrokProvider::new(None).expect("ctor ok"); // no ctor key, no env key
        let req = CanonicalRequest {
            model: "grok-3-mini".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("x".into()),
            }],
            ..Default::default()
        };
        let err = p.send(req).await.unwrap_err();
        unsafe {
            if let Some(v) = old {
                ::std::env::set_var("XAI_CREDENTIALS_PATH", v);
            } else {
                ::std::env::remove_var("XAI_CREDENTIALS_PATH");
            }
            if let Some(v) = old_key {
                ::std::env::set_var("XAI_API_KEY", v);
            }
        }
        match err {
            ProviderError::Auth(s) => assert!(
                s.contains("failed to load Grok credentials from file and no explicit/env key"),
                "got: {}",
                s
            ),
            other => panic!(
                "expected Auth for missing creds + no fallback, got {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    // See test_credentials_file_load_in_send: env lock held across await is safe
    // on the current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn test_xai_credentials_path_and_ctor_fallback() {
        // XAI_CREDENTIALS_PATH honored; when load fails, ctor key (simulating env) is fallback
        let _guard = CRED_ENV_LOCK.lock().expect("cred lock for env mutate");
        let non = format!("/tmp/xai-no-creds-fallback-{}.json", ::std::process::id());
        let _ = ::std::fs::remove_file(&non);
        let old = ::std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            ::std::env::set_var("XAI_CREDENTIALS_PATH", &non);
        }
        let p = GrokProvider::new_for_test("xai-ctor-fallback-key", "http://127.0.0.1:1");
        let req = CanonicalRequest {
            model: "grok-3-mini".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("fallback".into()),
            }],
            ..Default::default()
        };
        let err = p.send(req).await.unwrap_err();
        unsafe {
            if let Some(v) = old {
                ::std::env::set_var("XAI_CREDENTIALS_PATH", v);
            } else {
                ::std::env::remove_var("XAI_CREDENTIALS_PATH");
            }
        }
        // load failed -> used ctor key -> still network err (not the "no key" Auth)
        match err {
            ProviderError::Upstream(s) => {
                assert!(s.contains("error calling xAI") || s.contains("connection"))
            }
            other => panic!(
                "expected fallback to ctor key then upstream, got {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    // See test_credentials_file_load_in_send: env lock held across await is safe
    // on the current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn test_send_401_on_bad_key_forced_via_creds_path() {
        // Always exercises 401 path from xAI (no secret needed): force non-existent creds file so load fails,
        // ctor supplies a deliberately invalid key, real base_url -> 401 upstream error.
        let _guard = CRED_ENV_LOCK.lock().expect("cred lock for env mutate");
        let non = format!("/tmp/xai-badkey-401-{}.json", ::std::process::id());
        let _ = ::std::fs::remove_file(&non);
        let old = ::std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            ::std::env::set_var("XAI_CREDENTIALS_PATH", &non);
        }
        let p = GrokProvider::new_for_test(
            "xai-DEFINITELY-INVALID-KEY-FOR-TEST-401-XYZ",
            "https://api.x.ai/v1",
        );
        let req = CanonicalRequest {
            model: "grok-3-mini".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("auth test".into()),
            }],
            max_tokens: Some(4),
            ..Default::default()
        };
        let err = p.send(req).await.unwrap_err();
        unsafe {
            if let Some(v) = old {
                ::std::env::set_var("XAI_CREDENTIALS_PATH", v);
            } else {
                ::std::env::remove_var("XAI_CREDENTIALS_PATH");
            }
        }
        match err {
            ProviderError::Upstream(s) => {
                assert!(
                    s.contains("401")
                        || s.contains("xAI 401")
                        || s.to_lowercase().contains("invalid")
                        || s.to_lowercase().contains("auth"),
                    "bad key 401: {}",
                    s
                );
            }
            other => panic!("expected 401 Upstream for bad key, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_send_400_and_real_error_cases_conditional() {
        // When real key present, exercise a 4xx error path (bad model) in addition to success path.
        // Snapshot the key UNDER CRED_ENV_LOCK (see test_send_real_if_key_present) so this live
        // test cannot read a transiently-blanked env while a credential test mutates it.
        let key = {
            let _guard = CRED_ENV_LOCK.lock().expect("cred lock for live read");
            match ::std::env::var("XAI_API_KEY") {
                Ok(k) if !k.trim().is_empty() => k,
                _ => {
                    eprintln!(
                        "skipping 400 error case (no XAI_API_KEY); 401 path covered unconditionally"
                    );
                    return;
                }
            }
        };
        // success part (similar to existing)
        let p_ok = GrokProvider::new(Some(key.clone())).expect("ctor");
        let req_ok = CanonicalRequest {
            model: "grok-3-mini".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("PING".into()),
            }],
            max_tokens: Some(5),
            ..Default::default()
        };
        let ok = p_ok.send(req_ok).await.expect("success with good key");
        assert!(!ok.content.trim().is_empty() || ok.usage.input_tokens > 0);

        // error part: invalid model -> expect 4xx (400/404) from xAI
        let p_bad = GrokProvider::new(Some(key)).expect("ctor2");
        let req_bad = CanonicalRequest {
            model: "this-model-does-not-exist-xyz-999".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            max_tokens: Some(1),
            ..Default::default()
        };
        let err = p_bad.send(req_bad).await.unwrap_err();
        match err {
            ProviderError::Upstream(s) => {
                assert!(
                    s.contains("400") || s.contains("404") || s.contains("model"),
                    "expected 4xx model error: {}",
                    s
                );
            }
            other => panic!("expected upstream 4xx, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_headers_bearer_and_json_via_reqwest_builder() {
        // Assert the exact headers/body shape used inside send() (no extra deps; mirrors send code).
        // We build a request the same way the provider does (Authorization Bearer + Content-Type json).
        let effective_key = "xai-header-test-KEY-987";
        let body = json!({
            "model": "grok-4.3",
            "messages": [{"role":"user","content":"h"}],
            "stream": false
        });
        let url = "https://api.x.ai/v1/chat/completions";
        // replicate the builder steps from send (client is private but builder logic is the test target)
        let built = Client::new()
            .post(url)
            .header("Authorization", format!("Bearer {}", effective_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .build()
            .expect("build req");
        let headers = built.headers();
        assert_eq!(
            headers.get("authorization").unwrap().to_str().unwrap(),
            "Bearer xai-header-test-KEY-987"
        );
        assert_eq!(
            headers.get("content-type").unwrap().to_str().unwrap(),
            "application/json"
        );
        // body would be sent as the json
        assert!(built.body().is_some());
    }

    #[test]
    fn test_tool_roundtrip_to_from_and_streaming_note() {
        // full roundtrip mapper for a tool-using turn (to body shape + from response shape)
        let req = CanonicalRequest {
            model: "grok-4".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("call tool please".into()),
            }],
            tools: Some(vec![CanonicalTool {
                name: "adder".into(),
                description: Some("add two nums".into()),
                parameters: json!({"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}),
            }]),
            tool_choice: Some(CanonicalToolChoice::Specific {
                name: "adder".into(),
            }),
            ..Default::default()
        };
        let body = to_xai_chat_request(&req, &empty_repl());
        assert_eq!(body["tools"][0]["function"]["name"], "adder");
        assert_eq!(body["tool_choice"]["function"]["name"], "adder");

        // corresponding response from xai
        let raw = XaiChatCompletion {
            model: Some("grok-4".into()),
            choices: Some(vec![XaiChoice {
                message: Some(XaiAssistantMessage {
                    content: Some("".into()),
                    tool_calls: Some(vec![XaiToolCall {
                        id: Some("call_abc".into()),
                        function: Some(XaiFunctionCall {
                            name: Some("adder".into()),
                            arguments: Some(r#"{"a":2,"b":3}"#.into()),
                        }),
                        ..Default::default()
                    }]),
                    ..Default::default()
                }),
                finish_reason: Some("tool_calls".into()),
                ..Default::default()
            }]),
            usage: Some(XaiUsage {
                prompt_tokens: Some(8),
                completion_tokens: Some(2),
                ..Default::default()
            }),
            ..Default::default()
        };
        let canon = from_xai_chat_response(raw, &empty_repl());
        assert_eq!(canon.tool_calls.len(), 1);
        assert_eq!(canon.tool_calls[0].name, "adder");
        assert_eq!(canon.tool_calls[0].arguments, r#"{"a":2,"b":3}"#);
        assert_eq!(canon.finish_reason.as_deref(), Some("tool_calls"));

        // Streaming is now implemented via LlmProvider::send_stream: the stream
        // request builder flips stream:true and asks for usage, and the SSE parser
        // maps content/tool_call deltas to canonical events (see the dedicated
        // SSE parser test). Pin the builder flags here so the non-stream tool path
        // above and the stream path stay distinct.
        let stream_body = to_xai_chat_stream_request(&req, &empty_repl());
        assert_eq!(stream_body["stream"], true);
        assert_eq!(stream_body["stream_options"]["include_usage"], true);
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn test_replacements_interaction_response_on_args_and_full() {
        // inbound repl applies to content + tool name + arguments (full surface)
        let repl = Replacements::parse(
            r#"rule = [
                { scope = "response", search = "adder", replace = "sum" },
                { scope = "response", search = "2", replace = "TWO" }
            ]"#,
        )
        .unwrap();
        let raw = XaiChatCompletion {
            model: Some("grok".into()),
            choices: Some(vec![XaiChoice {
                message: Some(XaiAssistantMessage {
                    content: Some("result ready".into()),
                    tool_calls: Some(vec![XaiToolCall {
                        id: Some("c9".into()),
                        function: Some(XaiFunctionCall {
                            name: Some("adder".into()),
                            arguments: Some(r#"{"x":2,"y":2}"#.into()),
                        }),
                        ..Default::default()
                    }]),
                    ..Default::default()
                }),
                finish_reason: Some("tool_calls".into()),
                ..Default::default()
            }]),
            usage: None,
            ..Default::default()
        };
        let canon = from_xai_chat_response(raw, &repl);
        assert_eq!(canon.content, "result ready");
        assert_eq!(canon.tool_calls[0].name, "sum");
        assert_eq!(canon.tool_calls[0].arguments, r#"{"x":TWO,"y":TWO}"#); // both instances replaced
    }

    #[test]
    fn test_creds_check_expired_direct() {
        // Explicit unit for "check_expired" in the creds requirements list.
        // Intent: send() always calls it after fresh load (warn+continue on err); for xAI API keys it is always Ok (no-op today, future-proofs for any oauth-style tokens exactly as in CCP).
        // Construction works because GrokCredentials pub fields (see its tests); exercised indirectly by all file-load send tests.
        let c = GrokCredentials {
            api_key: "xai-foo-bar-123".into(),
        };
        assert!(
            c.check_expired().is_ok(),
            "grok creds check_expired must be non-fatal noop for standard keys"
        );
    }

    #[tokio::test]
    // Env lock held across the send().await (like the other credential tests):
    // both XAI_CREDENTIALS_PATH and XAI_API_KEY must stay fixed through the
    // request, and the lock serializes against other env mutators. Safe on the
    // current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn test_new_none_captures_xai_api_key_env_when_no_creds_file() {
        // WHY: the module docs and both binaries' startup messages advertise
        // XAI_API_KEY as a valid credential source, and resolve_api_key falls
        // back to the ctor key when the creds file is absent. That fallback is
        // only reachable if new(None) actually captures the env var. This pins
        // the documented contract: with only XAI_API_KEY set (no creds file),
        // the request authenticates with that key and reaches the upstream
        // (here a dead port, so it fails with a NETWORK error -- proving the key
        // was resolved, not an Auth error that would mean the key was dropped).
        let _guard = CRED_ENV_LOCK.lock().expect("cred lock for env mutate");
        // Point the creds path at a definitely-absent file so the file load
        // fails and the ctor/env fallback is the only key source.
        let missing = ::std::env::temp_dir().join(format!(
            "xai-creds-absent-{}-{}.json",
            ::std::process::id(),
            "envkeytest"
        ));
        let old_path = ::std::env::var("XAI_CREDENTIALS_PATH").ok();
        let old_key = ::std::env::var("XAI_API_KEY").ok();
        unsafe {
            ::std::env::set_var("XAI_CREDENTIALS_PATH", missing.to_str().unwrap());
            ::std::env::set_var("XAI_API_KEY", "xai-env-key-dummy-for-capture-test");
        }
        let p = GrokProvider::new(None)
            .expect("new(None) succeeds")
            .with_base_url("http://127.0.0.1:1");
        let req = CanonicalRequest {
            model: "grok-3-mini".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("env key capture test".into()),
            }],
            ..Default::default()
        };
        let err = p.send(req).await.unwrap_err();
        unsafe {
            match old_path {
                Some(v) => ::std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => ::std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
            match old_key {
                Some(v) => ::std::env::set_var("XAI_API_KEY", v),
                None => ::std::env::remove_var("XAI_API_KEY"),
            }
        }
        match err {
            // A network error means the env key WAS resolved and we got as far as
            // dialing the (dead) upstream. An Auth error here would mean the key
            // was never captured -- the bug this test guards against.
            ProviderError::Upstream(s) => assert!(
                s.contains("error calling xAI")
                    || s.contains("connection")
                    || s.contains("network"),
                "expected a network error after env-key resolution, got: {s}"
            ),
            ProviderError::Auth(s) => {
                panic!("XAI_API_KEY must be captured by new(None); got an Auth error instead: {s}")
            }
            other => panic!("expected Upstream network error, got {other:?}"),
        }
    }

    #[test]
    fn test_from_xai_no_choices_no_usage_tolerated() {
        // Edge for from_xai robustness (more from_xai coverage): partial/empty responses from wire must not panic; defaults to 0 usage, empty content/tools.
        // Why: xAI (and OpenAI compat) can return such in certain error/early-finish/tool-only or rate cases; canonical must stay usable.
        let raw = XaiChatCompletion {
            model: Some("grok-3-mini".into()),
            choices: Some(vec![]),
            usage: None,
            ..Default::default()
        };
        let canon = from_xai_chat_response(raw, &empty_repl());
        assert_eq!(canon.model, "grok-3-mini");
        assert!(canon.content.is_empty());
        assert!(canon.tool_calls.is_empty());
        assert_eq!(canon.usage.input_tokens, 0);
        assert_eq!(canon.usage.output_tokens, 0);
        assert!(canon.finish_reason.is_none());
    }

    // ============================================================
    // STREAMING (native SSE) tests
    // ============================================================

    #[test]
    fn test_stream_builder_sets_stream_and_usage_flag() {
        // WHY: the streaming path depends on two wire facts. (1) `stream: true` is what makes xAI
        // emit SSE instead of one JSON body; the non-stream builder MUST stay `false` (the existing
        // non-stream assertions and callers rely on that). (2) `stream_options.include_usage: true`
        // is the ONLY way xAI appends a final chunk carrying `usage` for a streamed response; without
        // it the parser would never see a Usage event and token accounting for streams would be lost.
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };
        let stream_body = to_xai_chat_stream_request(&req, &empty_repl());
        assert_eq!(stream_body["stream"], true);
        assert_eq!(stream_body["stream_options"]["include_usage"], true);

        // The non-stream builder is unchanged: still stream: false, no stream_options.
        let plain_body = to_xai_chat_request(&req, &empty_repl());
        assert_eq!(plain_body["stream"], false);
        assert!(plain_body.get("stream_options").is_none());
    }

    /// Drive the *production* SSE logic over a sequence of raw byte chunks exactly the way
    /// `send_stream` does: `SseBuffer` reframes bytes into complete lines (holding partial frames
    /// across chunk boundaries), `classify_sse_line` finds `data:`/`[DONE]`, `finish_reason_from_frame`
    /// remembers the reason, `parse_grok_sse_frame` produces events, and a single terminal `Finish`
    /// is appended at `[DONE]`. This is the identical pipeline the HTTP path runs, just fed from
    /// in-memory chunks so it needs no network or creds.
    fn drive_sse(chunks: &[&[u8]]) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let mut sse = SseBuffer::default();
        let mut out = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut done = false;
        for chunk in chunks {
            let text = String::from_utf8_lossy(chunk);
            for line in sse.push(&text) {
                match classify_sse_line(&line) {
                    SseLine::Ignore => {}
                    SseLine::Done => {
                        done = true;
                        break;
                    }
                    SseLine::Data(payload) => {
                        if let Some(fr) = finish_reason_from_frame(&payload) {
                            finish_reason = Some(fr);
                        }
                        out.extend(parse_grok_sse_frame(&payload));
                    }
                }
            }
            if done {
                break;
            }
        }
        out.push(Ok(CanonicalStreamEvent::Finish { finish_reason }));
        out
    }

    #[test]
    fn test_sse_parser_buffers_split_frames_and_orders_events() {
        // WHY: this is the load-bearing guarantee the HTTP streaming path depends on. reqwest's
        // bytes_stream() yields bytes in arbitrary splits, so a single `data: {json}` frame (and the
        // JSON inside it) WILL sometimes arrive across two network reads. If the parser did not buffer
        // partial lines, that JSON would fail to decode and we would either drop a delta or fail the
        // stream. We also pin event ORDER (text deltas in arrival order, then the tool-call delta,
        // then the trailing usage chunk, then exactly one terminal Finish with the right reason)
        // because downstream framing concatenates text by arrival and tool args by index; reordering
        // or a missing/duplicated Finish would corrupt the reconstructed assistant turn.

        // A complete content frame...
        let c0: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";
        // ...then a SECOND content frame deliberately CUT in half across two chunks to prove buffering.
        let c1a: &[u8] = b"data: {\"choices\":[{\"delta\":{\"con";
        let c1b: &[u8] = b"tent\":\" world\"}}]}\n\n";
        // A tool-call delta frame (first delta for index 0 carries id + name + an args fragment).
        let c2: &[u8] = b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n";
        // The closing content/finish frame (finish_reason non-null; remembered, not emitted yet).
        let c3: &[u8] =
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n";
        // The include_usage trailer (empty choices, carries usage) then the [DONE] sentinel.
        let c4: &[u8] = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\ndata: [DONE]\n\n";

        let events = drive_sse(&[c0, c1a, c1b, c2, c3, c4]);
        let events: Vec<CanonicalStreamEvent> = events
            .into_iter()
            .map(|r| r.expect("no parse errors expected for well-formed frames"))
            .collect();

        assert_eq!(
            events,
            vec![
                CanonicalStreamEvent::TextDelta("Hello".into()),
                // proves the split " world" frame was reassembled, not lost.
                CanonicalStreamEvent::TextDelta(" world".into()),
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call_abc".into()),
                    name: Some("get_weather".into()),
                    arguments_delta: "{\"city\":".into(),
                },
                CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: 11,
                    output_tokens: 7,
                    cache_read: 0,
                    cache_creation: 0,
                }),
                // exactly one terminal Finish, carrying the reason seen on the finish frame.
                CanonicalStreamEvent::Finish {
                    finish_reason: Some("tool_calls".into()),
                },
            ]
        );
    }

    #[test]
    fn test_sse_parser_malformed_frame_yields_upstream_error() {
        // WHY: a corrupt frame must fail loud (Err) so the stream surfaces the problem rather than
        // silently swallowing data the consumer is counting on. The driver stops the stream on the
        // first error (mirrors send_stream returning after yielding the Err).
        let bad: &[u8] = b"data: {not json}\n\n";
        let events = drive_sse(&[bad]);
        match &events[0] {
            Err(ProviderError::Upstream(s)) => {
                assert!(s.contains("decode xAI stream chunk"), "got: {s}")
            }
            other => panic!("expected Upstream error for malformed frame, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_send_stream_upstream_error_is_first_item() {
        // WHY: send_stream issues the HTTP call inside the stream, so a connection failure must
        // surface as the FIRST yielded Err (not a panic, not an empty stream). Uses an impossible
        // port as the "mock" upstream (same pattern as test_send_mocked_upstream_error). No creds: the
        // ctor key is the fallback, so we never hit the Auth branch.
        let p = GrokProvider::new_for_test("xai-dummy", "http://127.0.0.1:1");
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };
        let mut stream = p
            .send_stream(req)
            .await
            .expect("send_stream returns the stream eagerly");
        let first = stream
            .next()
            .await
            .expect("stream must yield at least one item");
        match first {
            Err(ProviderError::Upstream(s)) => {
                assert!(
                    s.contains("network error calling xAI") || s.contains("connection"),
                    "got: {s}"
                )
            }
            other => panic!("expected leading Upstream error for bad port, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_send_stream_real_if_creds_present() {
        // Guarded live test: only runs when creds are available (XAI_API_KEY env or
        // ~/.xai/.credentials.json). Otherwise skips so the offline suite stays green. Mirrors the
        // existing guarded-live pattern (test_send_real_if_key_present).
        // Snapshot creds-presence + key UNDER CRED_ENV_LOCK (see the note in
        // test_send_real_if_key_present) so this live test cannot race the
        // env-mutating credential tests. Lock dropped before the network await.
        let key_opt = {
            let _guard = CRED_ENV_LOCK.lock().expect("cred lock for live read");
            if !has_grok_creds() {
                eprintln!(
                    "skipping real grok stream test (no XAI_API_KEY and no ~/.xai/.credentials.json)"
                );
                return;
            }
            std::env::var("XAI_API_KEY").ok()
        };
        let p = GrokProvider::new(key_opt).expect("ctor for live stream test");
        let req = CanonicalRequest {
            model: "grok-3-mini".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("Reply with the single word: PONG".into()),
            }],
            max_tokens: Some(8),
            ..Default::default()
        };
        let mut stream = p.send_stream(req).await.expect("live stream open");
        let mut text = String::new();
        let mut saw_finish = false;
        while let Some(ev) = stream.next().await {
            match ev.expect("live stream event") {
                CanonicalStreamEvent::TextDelta(t) => text.push_str(&t),
                CanonicalStreamEvent::Finish { .. } => saw_finish = true,
                _ => {}
            }
        }
        assert!(saw_finish, "live stream must terminate with a Finish");
        assert!(
            !text.trim().is_empty(),
            "live stream must produce some text"
        );
    }

    /// True when real, human-provisioned xAI credentials are reachable for a live test: either the
    /// `XAI_API_KEY` env var, or a credentials file at the canonical `~/.xai/.credentials.json` home
    /// path. We deliberately read the *home* path directly rather than `GrokCredentials::default_path()`
    /// because that honors `$XAI_CREDENTIALS_PATH`, which the credential tests in this module set to a
    /// throwaway dummy-key file under the `CRED_ENV_LOCK`; keying the live guard off it would race those
    /// tests and trigger a real (failing) network call with a junk key. Requiring `XAI_CREDENTIALS_PATH`
    /// to be unset before trusting the file makes this fully race-immune and keeps the offline suite green.
    fn has_grok_creds() -> bool {
        if std::env::var("XAI_API_KEY")
            .map(|k| !k.trim().is_empty())
            .unwrap_or(false)
        {
            return true;
        }
        if std::env::var_os("XAI_CREDENTIALS_PATH").is_some() {
            return false;
        }
        match std::env::var_os("HOME") {
            Some(home) => ::std::path::Path::new(&home)
                .join(".xai")
                .join(".credentials.json")
                .exists(),
            None => false,
        }
    }
}
