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
    ContentBlock, MessageContent, MessagesRequest, MessagesResponse, SystemBlock, SystemField,
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
    let ctx = RequestContext::new_reply().with_session(session_uuid);

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

    let resp_value = state
        .upstream
        .send_messages_json(&creds, &ctx, &body)
        .await
        .map_err(map_upstream_err)?;

    let resp: MessagesResponse = serde_json::from_value(resp_value.clone()).map_err(|e| {
        AppError::ServerError(format!(
            "anthropic response decode: {e} (raw: {})",
            resp_value
        ))
    })?;

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    tracing::debug!(
        input_tokens = resp.usage.input_tokens,
        output_tokens = resp.usage.output_tokens,
        model = %resp.model,
        "v2 completion usage"
    );

    let mut oai_response = build_oai_response(&resp, &chat_id, created, model_def.canonical);

    // Apply replacements to the assistant text content if present.
    if !state.replacements.is_empty() {
        apply_replacements_inbound(&mut oai_response, state.replacements.as_ref());
    }

    if let Some(ref log) = conv_log {
        if let Ok(text) = serde_json::to_string(&oai_response) {
            log.log(&session_id, &request_id, "<<<", "OAI response", &text);
        }
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
    let _active = crate::stats::ActiveRequestGuard::new(&state.stats);

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
    let ctx = RequestContext::new_reply().with_session(session_uuid);

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
    let requested_model = model_def.canonical.to_string();
    let span = tracing::Span::current();

    tokio::spawn(
        async move {
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
                                match serde_json::to_string(&c) {
                                    Ok(s) => {
                                        if tx.send(Ok(Event::default().data(s))).await.is_err() {
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
                        let payload = serde_json::json!({
                            "error": {
                                "type": "upstream_error",
                                "message": msg,
                            }
                        });
                        if let Ok(s) = serde_json::to_string(&payload) {
                            let _ = tx.send(Ok(Event::default().data(s))).await;
                        }
                        break;
                    }
                }
            }

            // Flush any buffered tool-call arg fragments that didn't see a
            // finish_reason (e.g., upstream error or client cancellation),
            // rewritten so partial state still gets through correctly.
            if !replacements.is_empty() {
                for chunk in stream_state.flush(&replacements) {
                    if let Ok(s) = serde_json::to_string(&chunk) {
                        let _ = tx.send(Ok(Event::default().data(s))).await;
                    }
                }
            }

            if !errored {
                for chunk in converter.finalize_if_needed() {
                    if let Ok(s) = serde_json::to_string(&chunk) {
                        let _ = tx.send(Ok(Event::default().data(s))).await;
                    }
                }
            }

            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

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
                    if let Some(name) = function.get_mut("name") {
                        if let Some(s) = name.as_str() {
                            *name = serde_json::Value::String(repl.apply_response(s));
                        }
                    }
                    if let Some(idx) = index {
                        if let Some(args) = function.get("arguments").and_then(|v| v.as_str()) {
                            self.tool_args.entry(idx).or_default().push_str(args);
                            function.remove("arguments");
                        }
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
            msg = format!("upstream {}: {}", status, body);
        }
    }
    match surface {
        429 => AppError::RateLimited(msg),
        401 | 403 => AppError::Unauthorized(msg),
        400..=499 => AppError::BadRequest(msg),
        504 => AppError::Timeout(msg),
        _ => AppError::ServerError(msg),
    }
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
    anth_req.stream = Some(stream);

    // Apply replacements before identity injection: Claude Code's dynamic
    // suffix is computed from the exact post-replacement text we send.
    if !replacements.is_empty() {
        apply_replacements_outbound(&mut anth_req, replacements);
    }
    prepend_claude_code_identity(&mut anth_req, profile, inject_identity);

    Ok(anth_req)
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
        if let Some(content) = message.get_mut("content") {
            if let Some(s) = content.as_str() {
                let replaced = repl.apply_response(s);
                *content = serde_json::Value::String(replaced);
            }
        }
        if let Some(tool_calls) = message.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
            for call in tool_calls {
                if let Some(function) = call.get_mut("function").and_then(|f| f.as_object_mut()) {
                    if let Some(name) = function.get_mut("name") {
                        if let Some(s) = name.as_str() {
                            *name = serde_json::Value::String(repl.apply_response(s));
                        }
                    }
                    if let Some(args) = function.get_mut("arguments") {
                        if let Some(s) = args.as_str() {
                            *args = serde_json::Value::String(repl.apply_response(s));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::resolve_model;
    use crate::translate::anthropic::{CacheControl, ImageSource};
    use crate::translate::request::{ChatMessage, MessageContent as OaiMessageContent};
    use crate::upstream::fingerprint::{CLAUDE_CODE_SYSTEM_PREAMBLE, default_profile};

    fn haiku_model() -> &'static ModelDef {
        resolve_model("claude-haiku-4-5")
    }

    fn anthropic_request(user_text: &str) -> MessagesRequest {
        let req = ChatCompletionRequest {
            model: "claude-haiku-4-5".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(OaiMessageContent::Text(user_text.into())),
                ..Default::default()
            }],
            ..Default::default()
        };
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
        let req = ChatCompletionRequest {
            model: "claude-haiku-4-5".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(OaiMessageContent::Text("Say OK".into())),
                ..Default::default()
            }],
            ..Default::default()
        };
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
    fn outbound_build_has_exactly_one_billing_cch_marker() {
        let req = ChatCompletionRequest {
            model: "claude-haiku-4-5".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(OaiMessageContent::Text("Say OK".into())),
                ..Default::default()
            }],
            ..Default::default()
        };
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
            "x-anthropic-billing-header: cc_version=2.1.142.73b; cc_entrypoint=sdk-cli; cch="
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
