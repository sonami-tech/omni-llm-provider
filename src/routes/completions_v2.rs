//! v2 completions handler — talks directly to api.anthropic.com via the
//! upstream client. Native Anthropic Messages format upstream, OAI-completions
//! format inbound. No subprocess.
//!
//! Phase 2: non-streaming only. Streaming is added in Phase 3.

use std::convert::Infallible;
use std::sync::Arc;

use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{Instrument, error, info, warn};

use crate::AppState;
use crate::error::AppError;
use crate::models::ModelDef;
use crate::translate::anthropic::{
    ContentBlock, MessageContent, MessagesRequest, MessagesResponse, OutputConfig, SystemBlock,
    SystemField,
};
use crate::translate::build::build_messages_request;
use crate::translate::from_anthropic::build_oai_response;
use crate::translate::request::ChatCompletionRequest;
use crate::translate::to_oai_stream::OaiStreamConverter;
use crate::upstream::credentials::Credentials;
use crate::upstream::errors::UpstreamError;
use crate::upstream::fingerprint::{
    FINGERPRINT_PROFILES, FingerprintProfile, RequestContext, is_claude_code_billing_header,
};

fn request_id_header(request_id: &str) -> header::HeaderValue {
    header::HeaderValue::from_str(request_id)
        .unwrap_or_else(|_| header::HeaderValue::from_static("unknown"))
}

// Request context is threaded as discrete params rather than a struct; the
// handler is the single call site (from completions::completions_handler).
#[allow(clippy::too_many_arguments)]
pub async fn handle_non_streaming_v2(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
    model_def: &'static ModelDef,
    request_id: String,
    chat_id: String,
    created: u64,
    session_id: String,
    conv_log: Option<Arc<crate::conversation_log::ConversationLog>>,
) -> Result<Response, AppError> {
    let _active = crate::stats::ActiveRequestGuard::new(&state.stats);
    let start = std::time::Instant::now();

    let anth_req = build_outbound_messages_request(
        &request,
        model_def,
        false,
        state.replacements.as_ref(),
        state.fingerprint_profile,
        !state.config.no_preamble,
    )?;

    let creds_path = Credentials::default_path();
    let creds = Credentials::load_fresh(&creds_path).map_err(map_upstream_err)?;
    if let Err(e) = creds.check_expired() {
        return Err(map_upstream_err(e));
    }

    // Build per-request context. Session ID is derived from CCP's resolved
    // session id so multi-turn requests share an identifier.
    let session_uuid = derive_session_uuid(&session_id);
    let ctx = RequestContext::new_reply()
        .with_session(session_uuid)
        .with_model(anth_req.model.clone());

    let body = serde_json::to_value(&anth_req).map_err(|e| {
        AppError::ServerError(format!("failed to serialize anthropic request: {e}"))
    })?;

    if let Some(ref log) = conv_log {
        // This is the pre-retry context; update this logging path if a future
        // cch algorithm uses per-attempt request context fields.
        if let Ok(bytes) = state.fingerprint_profile.finalize_body_json(&body, &ctx) {
            log.log(
                &session_id,
                &request_id,
                ">>>",
                "Anthropic request",
                &String::from_utf8_lossy(&bytes),
            );
        }
    }

    let resp_value = match state.upstream.send_messages_json(&creds, &ctx, &body).await {
        Ok(v) => v,
        Err(e) => {
            let app_err = map_upstream_err(e);
            state.stats.record_error(model_def.canonical, &app_err.to_string());
            return Err(app_err);
        }
    };

    let resp: MessagesResponse = serde_json::from_value(resp_value.clone()).map_err(|e| {
        AppError::ServerError(format!(
            "anthropic response decode: {e} (raw: {})",
            resp_value
        ))
    })?;

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Record per-model token usage and duration. Keyed by canonical model so it
    // joins the request counts recorded in completions::completions_handler.
    // TTFT is None: it is only meaningful for streaming responses.
    state.stats.record_response(
        model_def.canonical,
        crate::stats::TokenUsage {
            input_tokens: resp.usage.input_tokens as u64,
            output_tokens: resp.usage.output_tokens as u64,
            cache_read_input_tokens: resp.usage.cache_read_input_tokens.unwrap_or(0) as u64,
            cache_creation_input_tokens: resp.usage.cache_creation_input_tokens.unwrap_or(0) as u64,
        },
        None,
        duration_ms,
    );

    tracing::debug!(
        input_tokens = resp.usage.input_tokens,
        output_tokens = resp.usage.output_tokens,
        model = %resp.model,
        "v2 completion usage"
    );

    let mut oai_response = build_oai_response(&resp, &chat_id, created, &anth_req.model);

    // Apply replacements to the assistant text content if present.
    if !state.replacements.is_empty() {
        apply_replacements_inbound(&mut oai_response, state.replacements.as_ref());
    }

    if let Some(ref log) = conv_log
        && let Ok(text) = serde_json::to_string(&oai_response) {
            log.log(&session_id, &request_id, "<<<", "OAI response", &text);
        }

    let finish_reason = oai_response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        .unwrap_or("stop")
        .to_string();

    info!(
        duration_ms = duration_ms.round() as u64,
        finish_reason = %finish_reason,
        "v2 non-streaming completion finished"
    );

    let headers = [(
        header::HeaderName::from_static("x-request-id"),
        request_id_header(&request_id),
    )];
    Ok((headers, axum::Json(oai_response)).into_response())
}

#[allow(clippy::too_many_arguments)]
pub async fn handle_streaming_v2(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
    model_def: &'static ModelDef,
    request_id: String,
    chat_id: String,
    created: u64,
    session_id: String,
    conv_log: Option<Arc<crate::conversation_log::ConversationLog>>,
) -> Result<Response, AppError> {
    let start = std::time::Instant::now();
    // Owned guard: counts this streaming request as active from setup onward,
    // then moves into the spawned task so the count spans the whole stream
    // lifetime (setup + body), and decrements on every task exit including
    // early returns and setup failures.
    let active = crate::stats::OwnedActiveRequestGuard::new(state.stats.clone());

    let anth_req = build_outbound_messages_request(
        &request,
        model_def,
        true,
        state.replacements.as_ref(),
        state.fingerprint_profile,
        !state.config.no_preamble,
    )?;

    let creds_path = Credentials::default_path();
    let creds = Credentials::load_fresh(&creds_path).map_err(map_upstream_err)?;
    creds.check_expired().map_err(map_upstream_err)?;

    let session_uuid = derive_session_uuid(&session_id);
    let ctx = RequestContext::new_reply()
        .with_session(session_uuid)
        .with_model(anth_req.model.clone());

    let body = serde_json::to_value(&anth_req).map_err(|e| {
        AppError::ServerError(format!("failed to serialize anthropic request: {e}"))
    })?;

    if let Some(ref log) = conv_log {
        // This is the pre-retry context; update this logging path if a future
        // cch algorithm uses per-attempt request context fields.
        if let Ok(bytes) = state.fingerprint_profile.finalize_body_json(&body, &ctx) {
            log.log(
                &session_id,
                &request_id,
                ">>>",
                "Anthropic streaming request",
                &String::from_utf8_lossy(&bytes),
            );
        }
    }

    let upstream_stream = state
        .upstream
        .send_messages_stream(&creds, &ctx, &body)
        .await
        .map_err(map_upstream_err)?;

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    let conv_request_id = request_id.clone();
    let conv_session_id = session_id.clone();
    let conv_log_for_task = conv_log.clone();
    let replacements = state.replacements.clone();
    let requested_model = anth_req.model.clone();
    let stats = state.stats.clone();
    let model_canonical = model_def.canonical;
    let span = tracing::Span::current();

    tokio::spawn(
        async move {
            // Keep the active-request guard (created before setup) alive for
            // the whole task so the active count spans setup + stream body and
            // decrements on every exit path (including early returns).
            let _active = active;
            let mut ttft_ms: Option<f64> = None;
            let mut converter =
                OaiStreamConverter::new(chat_id.clone(), created, requested_model.clone());
            let mut stream_state = StreamReplState::new(chat_id, created, requested_model);
            // Initial :ok comment to flush headers immediately.
            let _ = tx.send(Ok(Event::default().comment("ok"))).await;

            let mut stream = upstream_stream;
            let mut errored = false;

            while let Some(item) = stream.next().await {
                match item {
                    Ok(event) => {
                        let chunks = converter.on_event(event);
                        for chunk in chunks {
                            let chunks_to_emit = if !replacements.is_empty() {
                                stream_state.process(chunk, &replacements)
                            } else {
                                if let Some(text) = extract_content_delta(&chunk) {
                                    stream_state.accumulator.push_str(text);
                                }
                                vec![chunk]
                            };
                            for c in chunks_to_emit {
                                if ttft_ms.is_none() && chunk_carries_content(&c) {
                                    ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                                }
                                match serde_json::to_string(&c) {
                                    Ok(s) => {
                                        if tx.send(Ok(Event::default().data(s))).await.is_err() {
                                            // Client disconnected: no consumer
                                            // remains, so skip the trailing
                                            // flush / [DONE] / stats recording.
                                            // The ActiveRequestGuard still
                                            // decrements on this return (Drop),
                                            // and the request was already counted
                                            // by record_request at dispatch.
                                            return;
                                        }
                                    }
                                    Err(e) => {
                                        error!("v2 stream chunk serialize: {e}");
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        errored = true;
                        let msg = match &e {
                            UpstreamError::Anthropic {
                                parsed: Some(p), ..
                            } => p.error.message.clone(),
                            _ => e.to_string(),
                        };
                        warn!(error = %msg, "v2 stream upstream error");
                        stats.record_error(model_canonical, &msg);

                        // Order matters: flush any buffered (rewritten) tool-call
                        // arg fragments BEFORE the error frame so a client that
                        // was mid tool-call sees the completed arguments first,
                        // not after an error object.
                        if !replacements.is_empty() {
                            for chunk in stream_state.flush(&replacements) {
                                if let Ok(s) = serde_json::to_string(&chunk) {
                                    let _ = tx.send(Ok(Event::default().data(s))).await;
                                }
                            }
                        }

                        // Emit a single terminal chunk that carries both the
                        // error detail AND choices[0].finish_reason:"error", so
                        // clients keying off finish_reason see a clean stream end
                        // instead of a truncated stream. Routing the error
                        // through the converter reuses the canonical chunk shape
                        // and marks the converter finished (no double-finish).
                        for chunk in converter.on_event(
                            crate::upstream::stream::StreamEvent::Error {
                                kind: "upstream_error".into(),
                                message: msg,
                            },
                        ) {
                            if let Ok(s) = serde_json::to_string(&chunk) {
                                let _ = tx.send(Ok(Event::default().data(s))).await;
                            }
                        }
                        break;
                    }
                }
            }

            // Clean-finish path: flush buffered tool-call arg fragments that
            // didn't see a finish_reason (e.g. client cancellation), then emit
            // the terminal chunk if message_stop never arrived. The error path
            // above has already flushed and finalized.
            if !errored {
                if !replacements.is_empty() {
                    for chunk in stream_state.flush(&replacements) {
                        if let Ok(s) = serde_json::to_string(&chunk) {
                            let _ = tx.send(Ok(Event::default().data(s))).await;
                        }
                    }
                }
                for chunk in converter.finalize_if_needed() {
                    if let Ok(s) = serde_json::to_string(&chunk) {
                        let _ = tx.send(Ok(Event::default().data(s))).await;
                    }
                }
            }

            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

            // Record token usage + timing after the client has its terminal
            // frame so the stats write never adds to perceived latency. On the
            // error path the counts may be partial, which is still real usage.
            let (input_tokens, output_tokens) = converter.token_usage();
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            stats.record_response(
                model_canonical,
                crate::stats::TokenUsage {
                    input_tokens: input_tokens as u64,
                    output_tokens: output_tokens as u64,
                    ..Default::default()
                },
                ttft_ms,
                duration_ms,
            );

            if let Some(log) = conv_log_for_task {
                log.log(
                    &conv_session_id,
                    &conv_request_id,
                    "<<<",
                    "Streaming response (text accumulator)",
                    &stream_state.accumulator,
                );
            }

            info!("v2 streaming completion finished");
        }
        .instrument(span),
    );

    let stream = ReceiverStream::new(rx);
    let sse = Sse::new(stream).keep_alive(KeepAlive::default());
    let mut response = sse.into_response();
    response.headers_mut().insert(
        header::HeaderName::from_static("x-request-id"),
        request_id_header(&request_id),
    );
    Ok(response)
}

/// Whether an outbound chunk carries actual generated content (a non-empty text
/// delta or any tool-call delta), as opposed to the role-only opener or a
/// finish/usage trailer. Used to time TTFT at the first real token.
fn chunk_carries_content(chunk: &serde_json::Value) -> bool {
    let Some(delta) = chunk
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("delta"))
    else {
        return false;
    };
    let has_text = delta
        .get("content")
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty());
    let has_tool_calls = delta.get("tool_calls").is_some();
    has_text || has_tool_calls
}

/// Pull a `delta.content` text fragment from an outbound chunk if present.
fn extract_content_delta(chunk: &serde_json::Value) -> Option<&str> {
    chunk
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")?
        .as_str()
}

/// Streaming-replacement state. Buffers tool-call `function.arguments`
/// fragments per `tool_calls[i].index` so the response rewrite can run against
/// the complete argument JSON instead of partial fragments — partial-fragment
/// replacement would corrupt cases where a rename string straddles a chunk
/// boundary (e.g. `claudec` + `odecodetransit`).
struct StreamReplState {
    /// Rewritten text content accumulator for end-of-stream logging.
    pub accumulator: String,
    /// Raw (pre-rewrite) per-tool-call argument buffers, keyed by index.
    tool_args: std::collections::BTreeMap<u64, String>,
    /// Static OAI chunk header fields used when emitting synthetic flush chunks.
    chat_id: String,
    created: u64,
    model: String,
}

impl StreamReplState {
    fn new(chat_id: String, created: u64, model: String) -> Self {
        Self {
            accumulator: String::new(),
            tool_args: std::collections::BTreeMap::new(),
            chat_id,
            created,
            model,
        }
    }

    /// Process a chunk. Returns the chunks to emit downstream (typically just
    /// the rewritten chunk; when `finish_reason` arrives, synthetic flush
    /// chunks carrying the rewritten tool-call args are emitted first).
    fn process(
        &mut self,
        mut chunk: serde_json::Value,
        repl: &crate::replacements::Replacements,
    ) -> Vec<serde_json::Value> {
        if let Some(text) = extract_content_delta(&chunk).map(str::to_string) {
            let replaced = repl.apply_response(&text);
            self.accumulator.push_str(&replaced);
            if let Some(delta) = chunk
                .get_mut("choices")
                .and_then(|c| c.as_array_mut())
                .and_then(|arr| arr.first_mut())
                .and_then(|c| c.get_mut("delta"))
            {
                delta["content"] = serde_json::Value::String(replaced);
            }
        }

        if let Some(tool_calls) = chunk
            .get_mut("choices")
            .and_then(|c| c.as_array_mut())
            .and_then(|arr| arr.first_mut())
            .and_then(|c| c.get_mut("delta"))
            .and_then(|d| d.get_mut("tool_calls"))
            .and_then(|t| t.as_array_mut())
        {
            for call in tool_calls {
                let index = call.get("index").and_then(|v| v.as_u64());
                if let Some(function) = call.get_mut("function").and_then(|f| f.as_object_mut()) {
                    if let Some(name) = function.get_mut("name")
                        && let Some(s) = name.as_str() {
                            *name = serde_json::Value::String(repl.apply_response(s));
                        }
                    if let Some(idx) = index
                        && let Some(args) = function.get("arguments").and_then(|v| v.as_str()) {
                            self.tool_args.entry(idx).or_default().push_str(args);
                            function.remove("arguments");
                        }
                }
            }
        }

        let has_finish = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("finish_reason"))
            .map(|v| !v.is_null())
            .unwrap_or(false);

        let mut out: Vec<serde_json::Value> = Vec::new();
        if has_finish {
            out.extend(self.drain_flushed(repl));
        }
        out.push(chunk);
        out
    }

    /// Flush any leftover buffered args (called after the upstream loop exits,
    /// covering streams that ended without a `finish_reason`).
    fn flush(&mut self, repl: &crate::replacements::Replacements) -> Vec<serde_json::Value> {
        self.drain_flushed(repl)
    }

    fn drain_flushed(
        &mut self,
        repl: &crate::replacements::Replacements,
    ) -> Vec<serde_json::Value> {
        std::mem::take(&mut self.tool_args)
            .into_iter()
            .map(|(index, raw)| {
                let rewritten = repl.apply_response(&raw);
                serde_json::json!({
                    "id": self.chat_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "function": {
                                    "arguments": rewritten,
                                }
                            }]
                        },
                        "finish_reason": null
                    }]
                })
            })
            .collect()
    }
}

fn map_upstream_err(e: UpstreamError) -> AppError {
    let surface = e.surface_status();
    let mut msg = match &e {
        UpstreamError::Anthropic {
            parsed: Some(p), ..
        } => p.error.message.clone(),
        _ => e.to_string(),
    };
    // Anthropic frequently returns the literal string "Error" as the
    // message; surface enough context for operators.
    if msg == "Error" {
        if let UpstreamError::Anthropic {
            parsed: Some(p),
            status,
            ..
        } = &e
        {
            msg = format!(
                "upstream {} ({}): {}",
                status, p.error.kind, p.error.message
            );
        } else if let UpstreamError::Anthropic { status, body, .. } = &e {
            // Bound the raw upstream body before surfacing it to the client;
            // the full body is preserved server-side via tracing on the error
            // paths. Avoids forwarding an unbounded upstream blob downstream.
            msg = format!("upstream {}: {}", status, truncate_for_client(body));
        }
    }
    match surface {
        429 => AppError::RateLimited(msg),
        401 | 403 => AppError::Unauthorized(msg),
        400..=499 => AppError::BadRequest(msg),
        503 => AppError::ServiceUnavailable(msg),
        502 => AppError::BadGateway(msg),
        504 => AppError::Timeout(msg),
        // Includes 500 and any other 5xx Anthropic status we don't special-case.
        _ => AppError::ServerError(msg),
    }
}

/// Cap a string surfaced to API clients so an unbounded upstream error body is
/// not echoed verbatim. Operates on char boundaries.
fn truncate_for_client(s: &str) -> String {
    const MAX: usize = 500;
    if s.chars().count() <= MAX {
        return s.to_string();
    }
    let mut out: String = s.chars().take(MAX).collect();
    out.push_str("… (truncated)");
    out
}

fn derive_session_uuid(session_id: &str) -> uuid::Uuid {
    // Prefer parsing the inbound session id as a UUID; fall back to v5 hash
    // in DNS namespace so the same session id always maps to the same UUID.
    if let Ok(u) = uuid::Uuid::parse_str(session_id) {
        return u;
    }
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, session_id.as_bytes())
}

fn build_outbound_messages_request(
    request: &ChatCompletionRequest,
    model_def: &ModelDef,
    stream: bool,
    replacements: &crate::replacements::Replacements,
    profile: &FingerprintProfile,
    inject_identity: bool,
) -> Result<MessagesRequest, AppError> {
    let mut anth_req = build_messages_request(request, model_def)?;
    anth_req.model = profile.outbound_model(&request.model, model_def);
    apply_profile_wire_defaults(&mut anth_req, request, model_def, profile);
    anth_req.stream = Some(stream);

    // Apply replacements before identity injection: Claude Code's dynamic
    // suffix is computed from the exact post-replacement text we send.
    if !replacements.is_empty() {
        apply_replacements_outbound(&mut anth_req, replacements);
    }
    prepend_claude_code_identity(&mut anth_req, profile, inject_identity);

    Ok(anth_req)
}

fn apply_profile_wire_defaults(
    req: &mut MessagesRequest,
    source: &ChatCompletionRequest,
    model_def: &ModelDef,
    profile: &FingerprintProfile,
) {
    if source.max_tokens.is_none() && source.max_completion_tokens.is_none() {
        req.max_tokens = profile.wire_defaults_for_model(&req.model).max_tokens;
        // The wire-default override can drop max_tokens below the thinking
        // budget that build_messages_request already reconciled against the
        // model's own ceiling (e.g. reasoning_effort:"max" -> budget 32768 on a
        // 32k-default model). Anthropic rejects max_tokens <= budget_tokens, so
        // re-apply the invariant here. The wire default is a default, not a cap:
        // an explicit thinking budget must win over it. Cap at the model's
        // catalog ceiling (not the wire default — that is what caused the bug),
        // mirroring build_messages_request's reconciliation.
        if let Some(budget) = req
            .thinking
            .as_ref()
            .filter(|t| t.kind == "enabled")
            .and_then(|t| t.budget_tokens)
            && req.max_tokens <= budget
        {
            let ceiling = model_def.max_tokens.min(u32::MAX as u64) as u32;
            req.max_tokens = budget.saturating_add(1024).min(ceiling);
        }
    }
    let wire_defaults = profile.wire_defaults_for_model(&req.model);
    if req.temperature.is_none() {
        req.temperature = wire_defaults.temperature;
    }
    if req.output_config.is_none()
        && let Some(effort) = wire_defaults.output_effort {
            req.output_config = Some(OutputConfig {
                effort: effort.to_string(),
            });
        }
}

fn prepend_claude_code_identity(
    req: &mut MessagesRequest,
    profile: &FingerprintProfile,
    inject_identity: bool,
) {
    if !inject_identity {
        return;
    }

    let first_user_text = first_user_text_for_billing(req).unwrap_or("");
    let billing = SystemBlock {
        kind: "text".into(),
        text: profile.billing_header_text(first_user_text),
        cache_control: None,
    };
    let preamble = SystemBlock {
        kind: "text".into(),
        text: profile.system_preamble.to_string(),
        cache_control: None,
    };

    let existing_blocks = match req.system.take() {
        None => Vec::new(),
        Some(SystemField::Text(s)) => {
            if s.is_empty() {
                Vec::new()
            } else {
                vec![SystemBlock {
                    kind: "text".into(),
                    text: s,
                    cache_control: None,
                }]
            }
        }
        Some(SystemField::Blocks(blocks)) => blocks,
    };
    let existing_blocks = strip_existing_claude_identity(existing_blocks);
    let mut blocks = Vec::with_capacity(existing_blocks.len() + 2);
    blocks.push(billing);
    blocks.push(preamble);
    blocks.extend(existing_blocks);
    req.system = Some(SystemField::Blocks(blocks));
}

fn strip_existing_claude_identity(blocks: Vec<SystemBlock>) -> Vec<SystemBlock> {
    // Remove stale identity blocks anywhere in the system array so chained
    // proxies or callers that already injected Claude identity do not leave
    // conflicting billing markers behind the fresh pinned marker.
    blocks
        .into_iter()
        .filter(|b| {
            !is_claude_code_billing_header(&b.text) && !is_claude_code_system_preamble(&b.text)
        })
        .collect()
}

fn is_claude_code_system_preamble(text: &str) -> bool {
    FINGERPRINT_PROFILES
        .iter()
        .any(|profile| profile.system_preamble == text)
}

fn first_user_text_for_billing(req: &MessagesRequest) -> Option<&str> {
    let first_user = req.messages.iter().find(|m| m.role == "user")?;
    match &first_user.content {
        MessageContent::Text(text) => Some(text.as_str()),
        MessageContent::Blocks(blocks) => blocks.iter().find_map(|block| match block {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        }),
    }
}

fn apply_replacements_outbound(
    req: &mut crate::translate::anthropic::MessagesRequest,
    repl: &crate::replacements::Replacements,
) {
    match &mut req.system {
        Some(SystemField::Text(s)) => *s = repl.apply_prompt(s),
        Some(SystemField::Blocks(blocks)) => {
            for b in blocks {
                b.text = repl.apply_prompt(&b.text);
            }
        }
        None => {}
    }
    for m in &mut req.messages {
        match &mut m.content {
            MessageContent::Text(s) => *s = repl.apply_prompt(s),
            MessageContent::Blocks(blocks) => {
                for b in blocks {
                    apply_prompt_to_content_block(b, repl);
                }
            }
        }
    }
    if let Some(tools) = req.tools.as_mut() {
        for t in tools {
            t.name = repl.apply_prompt(&t.name);
            if let Some(d) = t.description.as_mut() {
                *d = repl.apply_prompt(d);
            }
            apply_prompt_to_json(&mut t.input_schema, repl);
        }
    }
}

fn apply_prompt_to_content_block(
    block: &mut crate::translate::anthropic::ContentBlock,
    repl: &crate::replacements::Replacements,
) {
    use crate::translate::anthropic::{ContentBlock, ToolResultContent};
    match block {
        ContentBlock::Text { text, .. } => *text = repl.apply_prompt(text),
        ContentBlock::ToolUse { name, input, .. } => {
            *name = repl.apply_prompt(name);
            apply_prompt_to_json(input, repl);
        }
        ContentBlock::ToolResult { content, .. } => match content {
            Some(ToolResultContent::Text(t)) => *t = repl.apply_prompt(t),
            Some(ToolResultContent::Blocks(blocks)) => {
                for b in blocks {
                    apply_prompt_to_content_block(b, repl);
                }
            }
            None => {}
        },
        ContentBlock::Thinking { .. } | ContentBlock::Image { .. } => {}
    }
}

fn apply_prompt_to_json(value: &mut serde_json::Value, repl: &crate::replacements::Replacements) {
    match value {
        serde_json::Value::String(s) => *s = repl.apply_prompt(s),
        serde_json::Value::Array(arr) => {
            for v in arr {
                apply_prompt_to_json(v, repl);
            }
        }
        serde_json::Value::Object(obj) => {
            for (_, v) in obj.iter_mut() {
                apply_prompt_to_json(v, repl);
            }
        }
        _ => {}
    }
}

fn apply_replacements_inbound(
    resp: &mut serde_json::Value,
    repl: &crate::replacements::Replacements,
) {
    let Some(choices) = resp.get_mut("choices").and_then(|c| c.as_array_mut()) else {
        return;
    };
    for c in choices {
        let Some(message) = c.get_mut("message").and_then(|m| m.as_object_mut()) else {
            continue;
        };
        if let Some(content) = message.get_mut("content")
            && let Some(s) = content.as_str() {
                let replaced = repl.apply_response(s);
                *content = serde_json::Value::String(replaced);
            }
        if let Some(tool_calls) = message.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
            for call in tool_calls {
                if let Some(function) = call.get_mut("function").and_then(|f| f.as_object_mut()) {
                    if let Some(name) = function.get_mut("name")
                        && let Some(s) = name.as_str() {
                            *name = serde_json::Value::String(repl.apply_response(s));
                        }
                    if let Some(args) = function.get_mut("arguments")
                        && let Some(s) = args.as_str() {
                            *args = serde_json::Value::String(repl.apply_response(s));
                        }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::anthropic::{CacheControl, ImageSource};
    use crate::translate::request::{ChatMessage, MessageContent as OaiMessageContent};
    use crate::upstream::fingerprint::{CLAUDE_CODE_SYSTEM_PREAMBLE, default_profile};

    fn haiku_model() -> &'static ModelDef {
        default_profile().resolve_model("claude-haiku-4-5")
    }

    fn chat_request(model: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(OaiMessageContent::Text("Say OK".into())),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn anthropic_request(user_text: &str) -> MessagesRequest {
        let mut req = chat_request("claude-haiku-4-5");
        req.messages[0].content = Some(OaiMessageContent::Text(user_text.into()));
        build_messages_request(&req, haiku_model()).unwrap()
    }

    fn system_texts(req: &MessagesRequest) -> Vec<&str> {
        match req.system.as_ref().unwrap() {
            SystemField::Blocks(blocks) => blocks.iter().map(|b| b.text.as_str()).collect(),
            SystemField::Text(_) => panic!("identity injection should force block system field"),
        }
    }

    #[test]
    fn identity_blocks_are_prepended_with_dynamic_billing_marker() {
        let mut req = anthropic_request("Say OK");
        req.system = Some(SystemField::Text("consumer system".into()));

        prepend_claude_code_identity(&mut req, default_profile(), true);

        let texts = system_texts(&req);
        assert_eq!(
            texts,
            vec![
                default_profile().billing_header_text("Say OK").as_str(),
                CLAUDE_CODE_SYSTEM_PREAMBLE,
                "consumer system"
            ]
        );
    }

    #[test]
    fn identity_injection_is_disabled_by_no_preamble() {
        let mut req = anthropic_request("Say OK");
        req.system = Some(SystemField::Text("consumer system".into()));

        prepend_claude_code_identity(&mut req, default_profile(), false);

        match req.system.as_ref().unwrap() {
            SystemField::Text(s) => assert_eq!(s, "consumer system"),
            SystemField::Blocks(_) => panic!("disabled identity injection should preserve system"),
        }
    }

    #[test]
    fn identity_dedupes_flat_canonical_preamble() {
        let mut req = anthropic_request("Say OK");
        req.system = Some(SystemField::Text(CLAUDE_CODE_SYSTEM_PREAMBLE.into()));

        prepend_claude_code_identity(&mut req, default_profile(), true);

        let texts = system_texts(&req);
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0], default_profile().billing_header_text("Say OK"));
        assert_eq!(texts[1], CLAUDE_CODE_SYSTEM_PREAMBLE);
    }

    #[test]
    fn existing_billing_marker_with_real_cch_is_replaced() {
        let mut req = anthropic_request("Say OK");
        req.system = Some(SystemField::Blocks(vec![
			SystemBlock {
				kind: "text".into(),
				text: "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch=e5ba6;".into(),
				cache_control: None,
			},
			SystemBlock {
				kind: "text".into(),
				text: CLAUDE_CODE_SYSTEM_PREAMBLE.into(),
				cache_control: None,
			},
			SystemBlock {
				kind: "text".into(),
				text: "consumer system".into(),
				cache_control: Some(CacheControl {
					kind: "ephemeral".into(),
					ttl: Some("5m".into()),
				}),
			},
		]));

        prepend_claude_code_identity(&mut req, default_profile(), true);

        let blocks = match req.system.as_ref().unwrap() {
            SystemField::Blocks(blocks) => blocks,
            SystemField::Text(_) => panic!("identity injection should force block system field"),
        };
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            blocks[0].text,
            default_profile().billing_header_text("Say OK")
        );
        assert_eq!(blocks[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE);
        assert_eq!(blocks[2].text, "consumer system");
        assert!(blocks[2].cache_control.is_some());
    }

    #[test]
    fn first_user_text_uses_first_text_block_only() {
        let mut req = anthropic_request("unused");
        req.messages = vec![crate::translate::anthropic::Message {
            role: "user".into(),
            content: MessageContent::Blocks(vec![
                ContentBlock::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/image.png".into(),
                    },
                },
                ContentBlock::Text {
                    text: "Say OK".into(),
                    cache_control: None,
                },
                ContentBlock::Text {
                    text: "ignored".into(),
                    cache_control: None,
                },
            ]),
        }];

        assert_eq!(first_user_text_for_billing(&req), Some("Say OK"));
    }

    #[test]
    fn first_user_text_does_not_fall_through_when_first_user_has_no_text() {
        let req = MessagesRequest {
            model: "claude-haiku-4-5".into(),
            max_tokens: 10,
            messages: vec![
                crate::translate::anthropic::Message {
                    role: "user".into(),
                    content: MessageContent::Blocks(vec![ContentBlock::Image {
                        source: ImageSource::Url {
                            url: "https://example.com/image.png".into(),
                        },
                    }]),
                },
                crate::translate::anthropic::Message {
                    role: "user".into(),
                    content: MessageContent::Text("ignored".into()),
                },
            ],
            system: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: Some(false),
            metadata: None,
            thinking: None,
            output_config: None,
        };

        assert_eq!(first_user_text_for_billing(&req), None);
    }

    #[test]
    fn outbound_build_applies_replacements_before_identity_suffix() {
        let req = ChatCompletionRequest {
            model: "claude-haiku-4-5".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(OaiMessageContent::Text("hellzz".into())),
                ..Default::default()
            }],
            ..Default::default()
        };
        let replacements = crate::replacements::Replacements::parse_for_test(
            r#"
			[[rule]]
			scope = "prompt"
			search = "hellzz"
			replace = "worldX"
		"#,
        )
        .unwrap();

        let anth_req = build_outbound_messages_request(
            &req,
            haiku_model(),
            false,
            &replacements,
            default_profile(),
            true,
        )
        .unwrap();
        let texts = system_texts(&anth_req);
        assert_eq!(texts[0], default_profile().billing_header_text("worldX"));
        match &anth_req.messages[0].content {
            MessageContent::Blocks(blocks) => match &blocks[0] {
                ContentBlock::Text { text, .. } => assert_eq!(text, "worldX"),
                _ => panic!("expected text block"),
            },
            MessageContent::Text(text) => assert_eq!(text, "worldX"),
        }
    }

    #[test]
    fn outbound_build_streaming_and_non_streaming_are_parity_paths() {
        let req = chat_request("claude-haiku-4-5");
        let replacements = crate::replacements::Replacements::empty();

        let mut non_stream = build_outbound_messages_request(
            &req,
            haiku_model(),
            false,
            &replacements,
            default_profile(),
            true,
        )
        .unwrap();
        let mut stream = build_outbound_messages_request(
            &req,
            haiku_model(),
            true,
            &replacements,
            default_profile(),
            true,
        )
        .unwrap();
        non_stream.stream = None;
        stream.stream = None;

        assert_eq!(
            serde_json::to_value(&non_stream).unwrap(),
            serde_json::to_value(&stream).unwrap()
        );
    }

    #[test]
    fn claude_2_1_154_short_alias_body_defaults_match_capture() {
        let replacements = crate::replacements::Replacements::empty();
        let cases = [
            ("opus", "claude-opus-4-8", 64_000, None, Some("high")),
            ("sonnet", "claude-sonnet-4-6", 32_000, Some(1.0), Some("high")),
            (
                "haiku",
                "claude-haiku-4-5-20251001",
                32_000,
                Some(1.0),
                None,
            ),
        ];

        for (input, expected_model, expected_max, expected_temperature, expected_effort) in cases {
            let req = chat_request(input);
            let model_def = default_profile().resolve_model(input);
            let anth_req = build_outbound_messages_request(
                &req,
                model_def,
                true,
                &replacements,
                default_profile(),
                true,
            )
            .unwrap();

            assert_eq!(anth_req.model, expected_model);
            assert_eq!(anth_req.max_tokens, expected_max);
            assert_eq!(anth_req.temperature, expected_temperature);
            assert_eq!(
                anth_req.output_config.as_ref().map(|config| config.effort.as_str()),
                expected_effort
            );
        }
    }

    #[test]
    fn claude_2_1_154_real_versioned_models_are_preserved_verbatim() {
        // Only real, Anthropic-acceptable versioned ids are forwarded verbatim:
        // exact catalog canonicals and model_wire_overrides keys.
        let replacements = crate::replacements::Replacements::empty();
        let cases = [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
        ];

        for input in cases {
            let req = chat_request(input);
            let model_def = default_profile().resolve_model(input);
            let anth_req = build_outbound_messages_request(
                &req,
                model_def,
                true,
                &replacements,
                default_profile(),
                true,
            )
            .unwrap();
            assert_eq!(anth_req.model, input, "{input} must be sent verbatim");
        }
    }

    #[test]
    fn claude_2_1_154_bare_family_names_resolve_to_canonical() {
        // Regression: bare family names and fake dated forms are NOT valid
        // Anthropic model ids. They must resolve to a real canonical, not be
        // forwarded verbatim (which produced live 400s: "model: claude-sonnet").
        let replacements = crate::replacements::Replacements::empty();
        let cases = [
            ("claude-opus", "claude-opus-4-8"),
            ("claude-sonnet", "claude-sonnet-4-6"),
            ("claude-haiku", "claude-haiku-4-5-20251001"),
            ("claude-sonnet-4-6-20260101", "claude-sonnet-4-6"),
            ("claude-opus-4-6-20260101", "claude-opus-4-8"),
        ];

        for (input, expected) in cases {
            let req = chat_request(input);
            let model_def = default_profile().resolve_model(input);
            let anth_req = build_outbound_messages_request(
                &req,
                model_def,
                true,
                &replacements,
                default_profile(),
                true,
            )
            .unwrap();
            assert_eq!(
                anth_req.model, expected,
                "{input} must resolve to canonical {expected}, not be sent verbatim"
            );
        }
    }

    #[test]
    fn claude_2_1_154_explicit_model_body_defaults_match_capture() {
        let replacements = crate::replacements::Replacements::empty();
        let cases = [
            ("claude-opus-4-8", 64_000, None, Some("high")),
            ("claude-opus-4-7", 64_000, None, Some("high")),
            ("claude-opus-4-6", 64_000, Some(1.0), Some("high")),
            ("claude-sonnet-4-6", 32_000, Some(1.0), Some("high")),
            ("claude-haiku-4-5", 32_000, Some(1.0), None),
            ("claude-haiku-4-5-20251001", 32_000, Some(1.0), None),
        ];

        for (input, expected_max, expected_temperature, expected_effort) in cases {
            let req = chat_request(input);
            let model_def = default_profile().resolve_model(input);
            let anth_req = build_outbound_messages_request(
                &req,
                model_def,
                true,
                &replacements,
                default_profile(),
                true,
            )
            .unwrap();

            assert_eq!(anth_req.model, input);
            assert_eq!(anth_req.max_tokens, expected_max);
            assert_eq!(anth_req.temperature, expected_temperature);
            assert_eq!(
                anth_req.output_config.as_ref().map(|config| config.effort.as_str()),
                expected_effort
            );
        }
    }

    #[test]
    fn wire_default_max_tokens_never_drops_below_thinking_budget() {
        // Regression: reasoning_effort:"max" on a 32k-default model (haiku,
        // sonnet) must not emit max_tokens <= thinking.budget_tokens, which
        // Anthropic rejects with a 400. opus has a 64k default and is already
        // above the 32768 "max" budget.
        let replacements = crate::replacements::Replacements::empty();
        for model in ["haiku", "sonnet", "opus"] {
            let mut req = chat_request(model);
            req.reasoning_effort = Some("max".into());
            let model_def = default_profile().resolve_model(model);
            let anth_req = build_outbound_messages_request(
                &req,
                model_def,
                true,
                &replacements,
                default_profile(),
                true,
            )
            .unwrap();
            let budget = anth_req
                .thinking
                .as_ref()
                .and_then(|t| t.budget_tokens)
                .expect("reasoning_effort:max enables a thinking budget");
            assert!(
                anth_req.max_tokens > budget,
                "{model}: max_tokens {} must exceed thinking budget {budget}",
                anth_req.max_tokens
            );
        }
    }

    #[test]
    fn explicit_max_tokens_is_respected_even_with_thinking() {
        // When the caller sets max_tokens explicitly, we do NOT apply the wire
        // default at all, so the reconciliation branch must not fire. The
        // caller owns the value (build_messages_request still bumps it above
        // the budget if needed, but that is the caller-set path).
        let replacements = crate::replacements::Replacements::empty();
        let mut req = chat_request("haiku");
        req.reasoning_effort = Some("low".into()); // budget 1024
        req.max_tokens = Some(2048);
        let model_def = default_profile().resolve_model("haiku");
        let anth_req = build_outbound_messages_request(
            &req,
            model_def,
            true,
            &replacements,
            default_profile(),
            true,
        )
        .unwrap();
        assert_eq!(anth_req.max_tokens, 2048);
    }

    #[test]
    fn claude_2_1_154_unknown_non_claude_model_still_falls_back_to_sonnet() {
        let req = chat_request("gpt-4");
        let model_def = default_profile().resolve_model(&req.model);
        let anth_req = build_outbound_messages_request(
            &req,
            model_def,
            true,
            &crate::replacements::Replacements::empty(),
            default_profile(),
            true,
        )
        .unwrap();

        assert_eq!(anth_req.model, "claude-sonnet-4-6");
    }

    #[test]
    fn outbound_build_has_exactly_one_billing_cch_marker() {
        let req = chat_request("claude-haiku-4-5");
        let anth_req = build_outbound_messages_request(
            &req,
            haiku_model(),
            false,
            &crate::replacements::Replacements::empty(),
            default_profile(),
            true,
        )
        .unwrap();
        let body = serde_json::to_value(&anth_req).unwrap();
        let bytes = default_profile()
            .finalize_body_json(&body, &RequestContext::new_reply())
            .unwrap();
        let json = String::from_utf8(bytes).unwrap();

        assert_eq!(json.matches("x-anthropic-billing-header:").count(), 1);
        assert_eq!(json.matches("cc_entrypoint=sdk-cli; cch=00000;").count(), 0);
		assert!(json.contains(
			"x-anthropic-billing-header: cc_version=2.1.154.cea; cc_entrypoint=sdk-cli; cch="
		));
    }

    #[test]
    fn identity_injection_works_for_every_profile() {
        for profile in FINGERPRINT_PROFILES {
            let mut req = anthropic_request("Say OK");
            req.system = Some(SystemField::Text("consumer system".into()));

            prepend_claude_code_identity(&mut req, profile, true);

            let texts = system_texts(&req);
            assert_eq!(texts[0], profile.billing_header_text("Say OK"));
            assert_eq!(texts[1], profile.system_preamble);
            assert_eq!(texts[2], "consumer system");
        }
    }
}
