//! OpenAI-Responses *upstream* protocol machinery (shared, provider-neutral).
//!
//! This is the single source of truth for the highest-wire-risk surface: the
//! pure SSE framing + the streaming event parser + the non-stream
//! response->canonical mapper for backends that speak the OpenAI Responses wire
//! (`response.created`, `response.output_text.delta`, ..., `response.completed`,
//! with NO `[DONE]` sentinel - that is a Chat Completions convention only).
//!
//! It was extracted verbatim from provider-codex so that any provider talking
//! the same wire at a different host (e.g. Grok CLI path) can reuse
//! the exact same parsing, guaranteeing wire parity. To stay decoupled from any
//! one provider, the three Codex-specific couplings are parameterized:
//!   1. the canonical metadata `provider` tag -> [`response_to_canonical`] takes
//!      a `provider_tag: &str`,
//!   2. error-string redaction -> abstracted behind the [`ErrorRedactor`] trait,
//!   3. the literal "codex" substrings in error messages -> reworded to
//!      provider-neutral text ("Responses stream ...").
//!
//! Request BODY builders stay per-provider; only the response/stream protocol
//! lives here.

use std::collections::HashMap;

use omni_core::{
    CanonicalResponse, CanonicalResponseMetadata, CanonicalStreamEvent, CanonicalToolCall,
    CanonicalUsage, ProviderError,
};
use serde_json::Value;

/// Hard caps on SSE framing to bound memory against a hostile or broken
/// upstream. A single line may not exceed [`MAX_SSE_LINE_BYTES`]; the
/// accumulated `data:` payload of one event may not exceed
/// [`MAX_SSE_EVENT_BYTES`]. These match the original Codex values exactly so
/// the framing behavior is identical across providers.
pub const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;
pub const MAX_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;

/// Redacts secrets out of error strings before they are surfaced to a caller.
///
/// The parser and the non-stream mapper run upstream payloads through this
/// before wrapping them in [`ProviderError::Upstream`], so credentials that an
/// upstream may echo back in an error never leak. Each provider supplies its
/// own implementation (it knows which header/query secrets to scrub); the
/// shared protocol code only needs the `redact` operation.
///
/// The bounds (`Clone + Default + Debug`) match how [`ResponsesStreamParser`]
/// uses the redactor: it stores it in a field, is `#[derive(Default)]`, and
/// clones it per stream.
pub trait ErrorRedactor: Clone + Default + std::fmt::Debug {
    /// Return `input` with any known secrets replaced. Must be lossless with
    /// respect to non-secret content (callers rely on the redacted text still
    /// describing the error).
    fn redact(&self, input: &str) -> String;
}

/// Prefix scrubber for secrets in an upstream error body. Scans for each marker
/// prefix and replaces from the marker to the next delimiter (whitespace /
/// quote / comma) with `<redacted>`. This catches known-prefix secrets even when
/// no resolved credentials are in scope; providers layer their exact captured
/// secrets on top for tokens that carry no known prefix.
///
/// Each provider passes its own marker set: Grok and Codex scrub
/// `["sk-", "xai-", "eyJ"]`; Claude scrubs `["sk-", "eyJ"]` (no xAI keys reach
/// the Anthropic path, and `sk-` already covers Claude OAuth `sk-ant-oat01-...`
/// and custom-gateway `sk-...` keys). `eyJ` covers JWT bearers.
pub fn redact_prefixed_secrets(input: &str, markers: &[&str]) -> String {
    let mut out = input.to_string();
    for marker in markers {
        while let Some(pos) = out.find(marker) {
            let end = out[pos..]
                .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == ',')
                .map(|i| pos + i)
                .unwrap_or(out.len());
            out.replace_range(pos..end, "<redacted>");
        }
    }
    out
}

/// Map a non-stream OpenAI-Responses payload to a [`CanonicalResponse`].
///
/// `fallback_model` is used when the payload omits `model`. `provider_tag` is
/// stamped into [`CanonicalResponseMetadata::provider`] so the caller can tell
/// which backend produced the response. `error_redactor` scrubs secrets from
/// the error body when the upstream reports `status == "failed"`.
pub fn response_to_canonical(
    value: &Value,
    fallback_model: &str,
    provider_tag: &str,
    error_redactor: &impl ErrorRedactor,
) -> Result<CanonicalResponse, ProviderError> {
    if value.get("status").and_then(|v| v.as_str()) == Some("failed") {
        return Err(ProviderError::upstream(
            error_redactor.redact(&value.to_string()),
        ));
    }

    let model = value
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_model)
        .to_string();
    let response_id = value.get("id").and_then(|v| v.as_str()).map(str::to_string);
    let mut content = String::new();
    let mut refusal = String::new();
    let mut tool_calls = Vec::new();
    let mut annotations = Vec::new();
    if let Some(items) = value.get("output").and_then(|v| v.as_array()) {
        for item in items {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                content.push_str(text);
                                if let Some(part_annotations) =
                                    part.get("annotations").and_then(|v| v.as_array())
                                {
                                    annotations.extend(part_annotations.iter().cloned());
                                }
                            } else if let Some(refusal_text) =
                                part.get("refusal").and_then(|v| v.as_str())
                            {
                                refusal.push_str(refusal_text);
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("call_unknown")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    tool_calls.push(CanonicalToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                _ => {}
            }
        }
    }
    if content.is_empty()
        && let Some(text) = value.get("output_text").and_then(|v| v.as_str())
    {
        content.push_str(text);
    }

    let usage = response_usage(value).unwrap_or_default();
    let finish_reason = match response_status(value) {
        Some("incomplete") => Some(response_incomplete_reason(value).to_string()),
        _ if !tool_calls.is_empty() => Some("tool_calls".to_string()),
        _ => Some("stop".to_string()),
    };

    Ok(CanonicalResponse {
        model,
        content,
        refusal: if refusal.is_empty() {
            None
        } else {
            Some(refusal)
        },
        tool_calls,
        finish_reason,
        usage,
        id: response_id.clone(),
        annotations,
        metadata: Some(CanonicalResponseMetadata {
            id: response_id,
            system_fingerprint: value
                .get("system_fingerprint")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            service_tier: value
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            provider: Some(provider_tag.to_string()),
            raw: None,
        }),
        reasoning: Vec::new(),
    })
}

/// Incremental SSE framer for the Responses wire.
///
/// Feeds raw response bytes (which may split a UTF-8 char, a line, or an event
/// across chunks) and yields complete [`ResponsesSseEvent`]s. Handles `\n`,
/// `\r\n`, and bare `\r` line endings, ignores comment (`:`) lines, and rejects
/// lines/events that exceed the byte caps. Call [`finish`](Self::finish) at end
/// of stream to flush any buffered trailing event.
#[derive(Debug, Default)]
pub struct ResponsesSseBuffer {
    line: Vec<u8>,
    last_was_cr: bool,
    event: Option<String>,
    data: Vec<String>,
    event_bytes: usize,
}

/// One framed SSE event: an optional `event:` name and the joined `data:` body.
#[derive(Debug)]
pub struct ResponsesSseEvent {
    pub event: Option<String>,
    pub data: String,
}

impl ResponsesSseBuffer {
    /// Push a chunk of raw bytes, returning any events completed by it.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<ResponsesSseEvent>, String> {
        let mut events = Vec::new();
        for line in self.complete_lines(bytes)? {
            self.process_line(line, &mut events)?;
        }
        Ok(events)
    }

    fn complete_lines(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        let mut lines = Vec::new();
        for byte in bytes {
            if self.last_was_cr {
                self.last_was_cr = false;
                if *byte == b'\n' {
                    continue;
                }
            }
            match *byte {
                b'\n' => lines.push(self.take_line()?),
                b'\r' => {
                    lines.push(self.take_line()?);
                    self.last_was_cr = true;
                }
                byte => {
                    self.line.push(byte);
                    if self.line.len() > MAX_SSE_LINE_BYTES {
                        return Err(format!(
                            "Responses stream line exceeded {} bytes",
                            MAX_SSE_LINE_BYTES
                        ));
                    }
                }
            }
        }
        Ok(lines)
    }

    fn take_line(&mut self) -> Result<String, String> {
        String::from_utf8(std::mem::take(&mut self.line))
            .map_err(|e| format!("Responses stream line was not UTF-8: {e}"))
    }

    fn process_line(
        &mut self,
        line: String,
        events: &mut Vec<ResponsesSseEvent>,
    ) -> Result<(), String> {
        if line.is_empty() {
            if let Some(event) = self.take_event() {
                events.push(event);
            }
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }
        if let Some(value) = line.strip_prefix("event:") {
            self.event = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            let value = value.trim_start();
            self.event_bytes = self.event_bytes.saturating_add(value.len());
            if self.event_bytes > MAX_SSE_EVENT_BYTES {
                return Err(format!(
                    "Responses stream event exceeded {} bytes",
                    MAX_SSE_EVENT_BYTES
                ));
            }
            self.data.push(value.to_string());
        }
        Ok(())
    }

    /// Flush a trailing event that was not terminated by a blank line.
    pub fn finish(&mut self) -> Result<Option<ResponsesSseEvent>, String> {
        if !self.line.is_empty() {
            let line = self.take_line()?;
            let mut events = Vec::new();
            self.process_line(line, &mut events)?;
            if let Some(event) = events.into_iter().next() {
                return Ok(Some(event));
            }
        }
        Ok(self.take_event())
    }

    fn take_event(&mut self) -> Option<ResponsesSseEvent> {
        if self.event.is_none() && self.data.is_empty() {
            return None;
        }
        let event = ResponsesSseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
        };
        self.event_bytes = 0;
        Some(event)
    }
}

#[derive(Debug, Clone, Default)]
struct StreamToolCall {
    id: Option<String>,
    name: Option<String>,
    emitted_open: bool,
    arguments: String,
    emitted_arguments_len: usize,
    canonical_index: u32,
}

/// Stateful parser that turns framed Responses SSE events into canonical
/// stream events.
///
/// Construct with [`new`](Self::new), then feed each [`ResponsesSseEvent`] to
/// [`handle_event`](Self::handle_event). The parser accumulates text/refusal
/// deltas, assembles parallel function-call arguments (absorbing gateways that
/// repeat the full arguments), and on a terminal `response.completed` /
/// `response.incomplete` emits any remaining content, a [`Usage`] event, and a
/// single [`Finish`]. A `[DONE]` sentinel or `response.failed` / `error` event
/// is surfaced as a redacted [`ProviderError::Upstream`].
///
/// [`Usage`]: CanonicalStreamEvent::Usage
/// [`Finish`]: CanonicalStreamEvent::Finish
#[derive(Debug, Default)]
pub struct ResponsesStreamParser<R: ErrorRedactor> {
    tool_calls: HashMap<u32, StreamToolCall>,
    next_tool_index: u32,
    saw_tool_call: bool,
    emitted_text: HashMap<(u32, &'static str), String>,
    completed: bool,
    provider_tag: String,
    error_redactor: R,
}

impl<R: ErrorRedactor> ResponsesStreamParser<R> {
    /// Create a parser stamping `provider_tag` into emitted response metadata
    /// and using `error_redactor` to scrub surfaced errors.
    pub fn new(provider_tag: &str, error_redactor: R) -> Self {
        Self {
            provider_tag: provider_tag.to_string(),
            error_redactor,
            ..Default::default()
        }
    }

    /// Whether a terminal event (completed/incomplete) has been observed.
    pub fn completed(&self) -> bool {
        self.completed
    }

    fn redact(&self, input: &str) -> String {
        self.error_redactor.redact(input)
    }

    /// Parse one framed SSE event into zero or more canonical stream events.
    pub fn handle_event(
        &mut self,
        event: ResponsesSseEvent,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let event_type = event.event.as_deref().unwrap_or_default();
        if event.data.trim() == "[DONE]" {
            return vec![Err(ProviderError::upstream(
                "Responses stream sent Chat [DONE] sentinel without a terminal response event",
            ))];
        }
        let value: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(e) => {
                return vec![Err(ProviderError::upstream(self.redact(&format!(
                    "decode Responses stream event {event_type}: {e}: {}",
                    event.data
                ))))];
            }
        };
        let kind = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or(event_type);
        match kind {
            "response.created" => self.handle_response_metadata(&value),
            "response.output_text.delta" | "response.refusal.delta" => self
                .emit_text_delta(
                    response_output_index(&value),
                    if kind == "response.refusal.delta" {
                        "refusal"
                    } else {
                        "text"
                    },
                    value
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default(),
                )
                .into_iter()
                .map(Ok)
                .collect(),
            "response.output_text.done" => self.handle_text_done(&value, "text"),
            "response.refusal.done" => self.handle_text_done(&value, "refusal"),
            "response.output_item.added" => self.handle_output_item_added(&value),
            "response.function_call_arguments.delta" => self.handle_function_args_delta(&value),
            "response.function_call_arguments.done" => self.handle_function_args_done(&value),
            "response.output_item.done" => self.handle_output_item_done(&value),
            "response.completed" => self.handle_completed(&value),
            "response.incomplete" => self.handle_incomplete(&value),
            "response.failed" | "error" => {
                vec![Err(ProviderError::upstream(
                    self.redact(&value.to_string()),
                ))]
            }
            _ => Vec::new(),
        }
    }

    fn handle_response_metadata(
        &self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let id = value
            .get("response")
            .and_then(|v| v.get("id"))
            .or_else(|| value.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let response = response_payload(value);
        let metadata = CanonicalResponseMetadata {
            id,
            system_fingerprint: response
                .get("system_fingerprint")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            service_tier: response
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            provider: Some(self.provider_tag.clone()),
            raw: None,
        };
        if metadata.id.is_none()
            && metadata.system_fingerprint.is_none()
            && metadata.service_tier.is_none()
        {
            Vec::new()
        } else {
            vec![Ok(CanonicalStreamEvent::ResponseMetadata(metadata))]
        }
    }

    fn emit_text_delta(
        &mut self,
        output_index: u32,
        channel: &'static str,
        delta: &str,
    ) -> Vec<CanonicalStreamEvent> {
        if delta.is_empty() {
            return Vec::new();
        }
        self.emitted_text
            .entry((output_index, channel))
            .or_default()
            .push_str(delta);
        let delta = delta.to_string();
        if channel == "refusal" {
            vec![CanonicalStreamEvent::RefusalDelta(delta)]
        } else {
            vec![CanonicalStreamEvent::TextDelta(delta)]
        }
    }

    fn handle_text_done(
        &mut self,
        value: &Value,
        field: &'static str,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let final_text = value
            .get(field)
            .or_else(|| value.get("text"))
            .or_else(|| value.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let output_index = response_output_index(value);
        self.emit_final_text(output_index, field, final_text)
    }

    fn emit_final_text(
        &mut self,
        output_index: u32,
        field: &'static str,
        final_text: &str,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        if final_text.is_empty() {
            return Vec::new();
        }
        let emitted = self
            .emitted_text
            .get(&(output_index, field))
            .map(String::as_str)
            .unwrap_or_default();
        if !final_text.starts_with(emitted) {
            return vec![Err(ProviderError::upstream(self.redact(&format!(
                "Responses stream {field}.done text did not extend prior text deltas"
            ))))];
        }
        let suffix = &final_text[emitted.len()..];
        self.emit_text_delta(output_index, field, suffix)
            .into_iter()
            .map(Ok)
            .collect()
    }

    fn handle_output_item_added(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let Some(item) = value.get("item") else {
            return Vec::new();
        };
        if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
            return Vec::new();
        }
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let canonical_index = self.ensure_tool_call(output_index);
        let call = self.tool_calls.entry(output_index).or_default();
        call.id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        call.name = item
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        self.saw_tool_call = true;
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) {
            if let Some(err) = self.append_tool_arguments_from_full(output_index, arguments) {
                return vec![Err(err)];
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect::<Vec<_>>()
    }

    fn handle_function_args_delta(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let delta = value
            .get("delta")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        self.saw_tool_call = true;
        let canonical_index = self.ensure_tool_call(output_index);
        if !delta.is_empty() {
            let already = self
                .tool_calls
                .get(&output_index)
                .map(|call| call.arguments.clone())
                .unwrap_or_default();
            if !already.is_empty() && delta == already {
                // Some Responses-compatible gateways repeat the full arguments
                // as the first delta after announcing them on output_item.added.
            } else if delta.starts_with(&already) && delta.len() > already.len() {
                self.append_tool_arguments(output_index, &delta[already.len()..]);
            } else {
                self.append_tool_arguments(output_index, &delta);
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn handle_function_args_done(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let Some(arguments) = value.get("arguments").and_then(|v| v.as_str()) else {
            return Vec::new();
        };
        let already = self
            .tool_calls
            .get(&output_index)
            .map(|call| call.arguments.clone())
            .unwrap_or_default();
        if arguments == already {
            return Vec::new();
        }
        if arguments.len() <= already.len() || !arguments.starts_with(&already) {
            return vec![Err(ProviderError::upstream(self.redact(
				"Responses stream function_call_arguments.done arguments did not extend prior argument deltas",
			)))];
        }
        let delta = arguments[already.len()..].to_string();
        let canonical_index = self.ensure_tool_call(output_index);
        self.append_tool_arguments(output_index, &delta);
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn handle_output_item_done(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let Some(item) = value.get("item") else {
            return Vec::new();
        };
        if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
            return Vec::new();
        }
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let canonical_index = self.ensure_tool_call(output_index);
        {
            let call = self.tool_calls.entry(output_index).or_default();
            if call.id.is_none() {
                call.id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
            if call.name.is_none() {
                call.name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
        }
        self.saw_tool_call = true;
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) {
            if let Some(err) = self.append_tool_arguments_from_full(output_index, arguments) {
                return vec![Err(err)];
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn handle_completed(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        if response_status(value) == Some("failed") {
            return vec![Err(ProviderError::upstream(
                self.redact(&value.to_string()),
            ))];
        }
        self.completed = true;
        let mut events = self.handle_response_metadata(value);
        events.extend(self.emit_terminal_output(value));
        if let Some(usage) = response_usage(value) {
            events.push(Ok(CanonicalStreamEvent::Usage(usage)));
        }
        events.push(Ok(CanonicalStreamEvent::Finish {
            finish_reason: self.finish_reason(),
        }));
        events
    }

    fn handle_incomplete(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        self.completed = true;
        let mut events = self.handle_response_metadata(value);
        events.extend(self.emit_terminal_output(value));
        if let Some(usage) = response_usage(value) {
            events.push(Ok(CanonicalStreamEvent::Usage(usage)));
        }
        events.push(Ok(CanonicalStreamEvent::Finish {
            finish_reason: Some(response_incomplete_reason(value).to_string()),
        }));
        events
    }

    fn emit_terminal_output(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let mut events = Vec::new();
        let Some(items) = response_payload(value)
            .get("output")
            .and_then(|v| v.as_array())
        else {
            return events;
        };
        for (position, item) in items.iter().enumerate() {
            let output_index = item
                .get("output_index")
                .and_then(|v| v.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(position as u32);
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            events.extend(self.emit_terminal_content_part(output_index, part));
                        }
                    }
                }
                Some("function_call") => {
                    events.extend(self.emit_terminal_function_call(output_index, item));
                }
                _ => {}
            }
        }
        events
    }

    fn emit_terminal_content_part(
        &mut self,
        output_index: u32,
        part: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let kind = part.get("type").and_then(|v| v.as_str());
        if kind == Some("refusal") || part.get("refusal").is_some() {
            let final_text = part
                .get("refusal")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            return self.emit_final_text(output_index, "refusal", final_text);
        }
        if kind == Some("output_text") || part.get("text").is_some() {
            let final_text = part
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let mut events = self.emit_final_text(output_index, "text", final_text);
            if let Some(annotations) = part.get("annotations").and_then(|v| v.as_array())
                && !annotations.is_empty()
            {
                events.push(Ok(CanonicalStreamEvent::OutputAnnotations(
                    annotations.to_vec(),
                )));
            }
            return events;
        }
        Vec::new()
    }

    fn emit_terminal_function_call(
        &mut self,
        output_index: u32,
        item: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let canonical_index = self.ensure_tool_call(output_index);
        {
            let call = self.tool_calls.entry(output_index).or_default();
            if call.id.is_none() {
                call.id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
            if call.name.is_none() {
                call.name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
        }
        self.saw_tool_call = true;
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) {
            if let Some(err) = self.append_tool_arguments_from_full(output_index, arguments) {
                return vec![Err(err)];
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn ensure_tool_call(&mut self, output_index: u32) -> u32 {
        if let Some(call) = self.tool_calls.get(&output_index) {
            return call.canonical_index;
        }
        let canonical_index = self.next_tool_index;
        self.next_tool_index += 1;
        self.tool_calls.insert(
            output_index,
            StreamToolCall {
                canonical_index,
                ..Default::default()
            },
        );
        canonical_index
    }

    fn emit_tool_open(&mut self, output_index: u32) -> Vec<CanonicalStreamEvent> {
        let canonical_index = self.ensure_tool_call(output_index);
        let call = self.tool_calls.entry(output_index).or_default();
        if call.emitted_open {
            return Vec::new();
        }
        call.emitted_open = true;
        vec![CanonicalStreamEvent::ToolCallDelta {
            index: canonical_index,
            id: call.id.clone(),
            name: call.name.clone(),
            arguments_delta: String::new(),
        }]
    }

    fn emit_tool_open_if_ready(&mut self, output_index: u32) -> Vec<CanonicalStreamEvent> {
        let Some(call) = self.tool_calls.get(&output_index) else {
            return Vec::new();
        };
        if call.emitted_open || call.id.is_none() || call.name.is_none() {
            return Vec::new();
        }
        self.emit_tool_open(output_index)
    }

    fn append_tool_arguments(&mut self, output_index: u32, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(call) = self.tool_calls.get_mut(&output_index) {
            call.arguments.push_str(delta);
        }
    }

    fn append_tool_arguments_from_full(
        &mut self,
        output_index: u32,
        arguments: &str,
    ) -> Option<ProviderError> {
        if arguments.is_empty() {
            return None;
        }
        let already = self
            .tool_calls
            .get(&output_index)
            .map(|call| call.arguments.clone())
            .unwrap_or_default();
        if arguments == already {
            return None;
        }
        if arguments.len() > already.len() && arguments.starts_with(&already) {
            self.append_tool_arguments(output_index, &arguments[already.len()..]);
            return None;
        }
        Some(ProviderError::upstream(self.redact(
			"Responses stream terminal function_call arguments did not extend prior argument deltas",
		)))
    }

    fn emit_pending_tool_args(
        &mut self,
        output_index: u32,
        canonical_index: u32,
    ) -> Vec<CanonicalStreamEvent> {
        let Some(call) = self.tool_calls.get_mut(&output_index) else {
            return Vec::new();
        };
        if !call.emitted_open || call.emitted_arguments_len >= call.arguments.len() {
            return Vec::new();
        }
        let delta = call.arguments[call.emitted_arguments_len..].to_string();
        call.emitted_arguments_len = call.arguments.len();
        vec![CanonicalStreamEvent::ToolCallDelta {
            index: canonical_index,
            id: None,
            name: None,
            arguments_delta: delta,
        }]
    }

    fn finish_reason(&self) -> Option<String> {
        Some(if self.saw_tool_call {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        })
    }
}

/// Extract token usage from a Responses payload (looking under `response.usage`
/// first, then top-level `usage`). Returns `None` when no usage is present.
pub fn response_usage(value: &Value) -> Option<CanonicalUsage> {
    let usage = value
        .get("response")
        .and_then(|v| v.get("usage"))
        .or_else(|| value.get("usage"))?;
    let input_audio_tokens = usage
        .get("input_tokens_details")
        .and_then(|v| v.get("audio_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_audio_tokens = usage
        .get("output_tokens_details")
        .and_then(|v| v.get("audio_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    Some(CanonicalUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        output_tokens: usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_read: usage
            .get("input_tokens_details")
            .and_then(|v| v.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_creation: 0,
        reasoning_tokens: usage
            .get("output_tokens_details")
            .and_then(|v| v.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        audio_tokens: input_audio_tokens + output_audio_tokens,
        input_audio_tokens,
        output_audio_tokens,
        accepted_prediction_tokens: usage
            .get("output_tokens_details")
            .and_then(|v| v.get("accepted_prediction_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        rejected_prediction_tokens: usage
            .get("output_tokens_details")
            .and_then(|v| v.get("rejected_prediction_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        ..CanonicalUsage::default()
    })
}

/// Read the response `status` ("completed", "incomplete", "failed", ...),
/// looking under a `response` envelope first.
pub fn response_status(value: &Value) -> Option<&str> {
    response_payload(value)
        .get("status")
        .and_then(|v| v.as_str())
}

/// Unwrap the `response` envelope when present, else return `value` itself.
/// Terminal stream events nest the payload under `response`; the non-stream
/// body is the payload directly.
pub fn response_payload(value: &Value) -> &Value {
    value.get("response").unwrap_or(value)
}

fn response_output_index(value: &Value) -> u32 {
    value
        .get("output_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

/// Map an `incomplete` response to its canonical finish reason. The wire
/// `max_output_tokens` reason is normalized to the canonical `length`; any
/// other reason (e.g. `content_filter`) is preserved verbatim.
pub fn response_incomplete_reason(value: &Value) -> &str {
    let reason = value
        .get("response")
        .and_then(|v| v.get("incomplete_details"))
        .and_then(|v| v.get("reason"))
        .or_else(|| {
            value
                .get("incomplete_details")
                .and_then(|v| v.get("reason"))
        })
        .and_then(|v| v.as_str())
        .unwrap_or("max_output_tokens");
    if reason == "max_output_tokens" {
        "length"
    } else {
        reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TAG: &str = "test-provider";

    /// A redactor that scrubs a single known token. Used to prove the parser
    /// and mapper still run errors through redaction (the wire-safety
    /// guarantee), without pulling in any provider-specific secret detection.
    #[derive(Clone, Debug, Default)]
    struct TokenRedactor;

    impl ErrorRedactor for TokenRedactor {
        fn redact(&self, input: &str) -> String {
            input.replace("sk-secret", "<redacted>")
        }
    }

    fn parser() -> ResponsesStreamParser<TokenRedactor> {
        ResponsesStreamParser::new(TAG, TokenRedactor)
    }

    #[test]
    fn redact_prefixed_secrets_honors_marker_set_and_delimiter() {
        // WHY: providers pass different marker sets (Claude omits `xai-`). A
        // marker NOT in the set must survive verbatim, or Claude would scrub
        // substrings it never intended to. Present markers must scrub from the
        // prefix up to the next delimiter (space/quote/comma), no further.
        let markers = ["sk-", "eyJ"];
        let out = redact_prefixed_secrets(
            r#"{"a":"sk-leak","b":"xai-keep","c":"eyJtok end"}"#,
            &markers,
        );
        assert!(!out.contains("sk-leak"), "prefixed secret leaked: {out}");
        assert!(!out.contains("eyJtok"), "jwt bearer leaked: {out}");
        assert!(
            out.contains("xai-keep"),
            "marker not in set must be untouched: {out}"
        );
        // Non-secret structure around the scrubbed spans is preserved.
        assert!(out.contains("<redacted>"));
        assert!(out.contains(r#""b":"xai-keep""#));
    }

    /// Feed a single SSE event (by name + JSON data) through a fresh framer and
    /// parser, returning the canonical events. WHY: exercises the real framing
    /// + parse path, not a hand-built event, so the two stay in lock-step.
    fn run_event(
        parser: &mut ResponsesStreamParser<TokenRedactor>,
        event_name: &str,
        data: Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let mut buffer = ResponsesSseBuffer::default();
        let chunk = format!("event: {event_name}\ndata: {data}\n\n");
        let framed = buffer.push(chunk.as_bytes()).expect("frame event");
        let mut out = Vec::new();
        for event in framed {
            out.extend(parser.handle_event(event));
        }
        out
    }

    fn ok_events(
        results: Vec<Result<CanonicalStreamEvent, ProviderError>>,
    ) -> Vec<CanonicalStreamEvent> {
        results.into_iter().map(|r| r.expect("ok event")).collect()
    }

    // WHY: text deltas are the most common path; they must surface as ordered
    // TextDelta events so the framing layer can reconstruct the assistant
    // message. Mirrors codex send_stream_maps_responses_text_usage_and_finish.
    #[test]
    fn text_deltas_accumulate_into_text_events() {
        let mut parser = parser();
        let a = ok_events(run_event(
            &mut parser,
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","delta":"Hel"}),
        ));
        let b = ok_events(run_event(
            &mut parser,
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","delta":"lo"}),
        ));
        assert_eq!(a, vec![CanonicalStreamEvent::TextDelta("Hel".into())]);
        assert_eq!(b, vec![CanonicalStreamEvent::TextDelta("lo".into())]);
    }

    // WHY: some upstreams send only the final output_text.done with no prior
    // deltas; we must still emit the full text once. Mirrors codex
    // send_stream_uses_output_text_done_when_no_deltas.
    #[test]
    fn output_text_done_without_deltas_emits_full_text() {
        let mut parser = parser();
        let events = ok_events(run_event(
            &mut parser,
            "response.output_text.done",
            json!({"type":"response.output_text.done","text":"complete"}),
        ));
        assert_eq!(
            events,
            vec![CanonicalStreamEvent::TextDelta("complete".into())]
        );
    }

    // WHY: when deltas precede a .done that repeats the full text, only the
    // missing suffix may be emitted, never a duplicate. Mirrors codex
    // send_stream_emits_output_text_done_suffix_without_duplicate.
    #[test]
    fn output_text_done_emits_only_missing_suffix() {
        let mut parser = parser();
        let mut text = String::new();
        for event in ok_events(run_event(
            &mut parser,
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","delta":"Hel"}),
        )) {
            if let CanonicalStreamEvent::TextDelta(d) = event {
                text.push_str(&d);
            }
        }
        for event in ok_events(run_event(
            &mut parser,
            "response.output_text.done",
            json!({"type":"response.output_text.done","text":"Hello"}),
        )) {
            if let CanonicalStreamEvent::TextDelta(d) = event {
                text.push_str(&d);
            }
        }
        assert_eq!(text, "Hello");
    }

    // WHY: a function call announced via output_item.added then streamed via
    // argument deltas must assemble into one tool call: an opening
    // ToolCallDelta carrying id+name, then argument-only deltas in order, and a
    // terminal Finish reason of "tool_calls". This is the core multi-turn-tools
    // contract. Mirrors codex send_stream_maps_responses_tool_call_deltas.
    #[test]
    fn function_call_argument_deltas_assemble_into_tool_call() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
            &mut parser,
            "response.output_item.added",
            json!({"type":"response.output_item.added","output_index":0,
				"item":{"type":"function_call","call_id":"call_1","name":"lookup","arguments":""}}),
        )));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\""}),
		)));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":":\"sf\"}"}),
		)));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.completed",
			json!({"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":6}}}),
		)));
        assert_eq!(
            events,
            vec![
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call_1".into()),
                    name: Some("lookup".into()),
                    arguments_delta: String::new(),
                },
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments_delta: "{\"q\"".into(),
                },
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments_delta: ":\"sf\"}".into(),
                },
                CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: 5,
                    output_tokens: 6,
                    ..Default::default()
                }),
                CanonicalStreamEvent::Finish {
                    finish_reason: Some("tool_calls".into())
                }
            ]
        );
    }

    // WHY: certain Responses gateways repeat the FULL arguments as the first
    // delta right after announcing them on output_item.added. The parser must
    // absorb that repeat and emit the arguments exactly once, or downstream
    // JSON would be doubled and unparseable. Mirrors codex
    // send_stream_does_not_duplicate_arguments_repeated_after_item_added.
    #[test]
    fn arguments_repeated_after_item_added_are_absorbed() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
			&mut parser,
			"response.output_item.added",
			json!({"type":"response.output_item.added","output_index":0,
				"item":{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"sf\"}"}}),
		)));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"sf\"}"}),
		)));
        let arguments: String = events
            .into_iter()
            .filter_map(|event| match event {
                CanonicalStreamEvent::ToolCallDelta {
                    arguments_delta, ..
                } if !arguments_delta.is_empty() => Some(arguments_delta),
                _ => None,
            })
            .collect();
        assert_eq!(arguments, r#"{"q":"sf"}"#);
    }

    // WHY: argument deltas may arrive BEFORE the output_item.added that carries
    // id+name; the opening ToolCallDelta must be withheld until id+name exist,
    // then the buffered args flushed. Otherwise a tool call with no name would
    // reach the client. Mirrors codex
    // send_stream_buffers_tool_arguments_until_metadata_arrives.
    #[test]
    fn tool_arguments_buffer_until_metadata_arrives() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":3,"delta":"{\"q\""}),
		)));
        events.extend(ok_events(run_event(
			&mut parser,
			"response.function_call_arguments.delta",
			json!({"type":"response.function_call_arguments.delta","output_index":3,"delta":":\"sf\"}"}),
		)));
        events.extend(ok_events(run_event(
            &mut parser,
            "response.output_item.added",
            json!({"type":"response.output_item.added","output_index":3,
				"item":{"type":"function_call","call_id":"call_late","name":"lookup","arguments":""}}),
        )));
        assert_eq!(
            events[0],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_late".into()),
                name: Some("lookup".into()),
                arguments_delta: String::new(),
            }
        );
        assert_eq!(
            events[1],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "{\"q\":\"sf\"}".into(),
            }
        );
    }

    // WHY: sparse upstream output_index values (e.g. 2) must map to a dense,
    // zero-based canonical tool index so clients see contiguous tool calls.
    // Mirrors codex send_stream_maps_sparse_response_output_indexes_to_dense_tool_indexes.
    #[test]
    fn sparse_output_indexes_map_to_dense_tool_indexes() {
        let mut parser = parser();
        let mut events = Vec::new();
        events.extend(ok_events(run_event(
            &mut parser,
            "response.output_item.added",
            json!({"type":"response.output_item.added","output_index":2,
				"item":{"type":"function_call","call_id":"call_sparse","name":"lookup","arguments":""}}),
        )));
        events.extend(ok_events(run_event(
            &mut parser,
            "response.function_call_arguments.delta",
            json!({"type":"response.function_call_arguments.delta","output_index":2,"delta":"{}"}),
        )));
        assert_eq!(
            events[0],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_sparse".into()),
                name: Some("lookup".into()),
                arguments_delta: String::new(),
            }
        );
        assert_eq!(
            events[1],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "{}".into(),
            }
        );
    }

    // WHY: response.completed must close the stream with a Usage event (so the
    // framing layer can report tokens) followed by exactly one Finish. Mirrors
    // the tail of codex send_stream_maps_responses_text_usage_and_finish.
    #[test]
    fn completed_emits_usage_then_finish() {
        let mut parser = parser();
        let events = ok_events(run_event(
            &mut parser,
            "response.completed",
            json!({"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":3,"output_tokens":4}}}),
        ));
        assert_eq!(
            events,
            vec![
                CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: 3,
                    output_tokens: 4,
                    ..Default::default()
                }),
                CanonicalStreamEvent::Finish {
                    finish_reason: Some("stop".into())
                }
            ]
        );
        assert!(parser.completed());
    }

    // WHY: an incomplete response must preserve the wire reason as the finish
    // reason (here content_filter) rather than collapsing to "stop", so the
    // caller learns WHY generation stopped. Mirrors codex
    // send_stream_preserves_incomplete_content_filter_reason.
    #[test]
    fn incomplete_preserves_reason_as_finish_reason() {
        let mut parser = parser();
        let events = ok_events(run_event(
            &mut parser,
            "response.incomplete",
            json!({"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"content_filter"}}}),
        ));
        assert_eq!(
            events.last().unwrap(),
            &CanonicalStreamEvent::Finish {
                finish_reason: Some("content_filter".into())
            }
        );
    }

    // WHY: response.failed (and the bare `error` event) must become a
    // ProviderError::Upstream, AND the error body must be redacted so a leaked
    // upstream credential is never surfaced. Mirrors codex
    // send_stream_redacts_failed_response_event.
    #[test]
    fn failed_event_becomes_redacted_upstream_error() {
        let mut parser = parser();
        let results = run_event(
            &mut parser,
            "response.failed",
            json!({"type":"response.failed","response":{"status":"failed"},"error":{"message":"bad sk-secret token"}}),
        );
        let err = results.into_iter().next().unwrap().unwrap_err().to_string();
        assert!(matches!(
            ProviderError::upstream(String::new()),
            ProviderError::Upstream { .. }
        ));
        assert!(!err.contains("sk-secret"), "leaked secret: {err}");
        assert!(err.contains("<redacted>"), "redaction missing: {err}");
    }

    // WHY: completed-with-status-failed is a second failure shape (the failure
    // rides inside the completed envelope) and must also redact and error.
    // Mirrors codex send_stream_treats_completed_failed_status_as_error.
    #[test]
    fn completed_with_failed_status_is_redacted_error() {
        let mut parser = parser();
        let results = run_event(
            &mut parser,
            "response.completed",
            json!({"type":"response.completed","response":{"status":"failed","error":"boom sk-secret"}}),
        );
        let err = results.into_iter().next().unwrap().unwrap_err().to_string();
        assert!(!err.contains("sk-secret"), "leaked secret: {err}");
        assert!(err.contains("<redacted>"));
    }

    // WHY: the Responses wire has NO [DONE] sentinel (that is Chat Completions
    // only). Receiving one means a mislabeled/Chat stream reached the Responses
    // parser and must be rejected loudly, not silently treated as success.
    // Mirrors codex send_stream_rejects_chat_done_sentinel_on_responses_wire.
    #[test]
    fn done_sentinel_is_rejected_as_error() {
        let mut parser = parser();
        let mut buffer = ResponsesSseBuffer::default();
        let framed = buffer.push(b"data: [DONE]\n\n").expect("frame");
        let mut results = Vec::new();
        for event in framed {
            results.extend(parser.handle_event(event));
        }
        let err = results.into_iter().next().unwrap().unwrap_err().to_string();
        assert!(err.contains("[DONE] sentinel"), "{err}");
    }

    // WHY: response bytes can split a multi-byte UTF-8 char across network
    // chunks; the framer must reassemble before decoding so text is never
    // corrupted. Mirrors codex send_stream_preserves_split_utf8_lines.
    #[test]
    fn buffer_reassembles_utf8_split_across_chunks() {
        let mut parser = parser();
        let mut buffer = ResponsesSseBuffer::default();
        let line =
			b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"h\xc3\xa9llo \xf0\x9f\x8c\x8e\"}\n\n";
        let split = 40;
        let mut results = Vec::new();
        for event in buffer.push(&line[..split]).expect("first chunk") {
            results.extend(parser.handle_event(event));
        }
        for event in buffer.push(&line[split..]).expect("second chunk") {
            results.extend(parser.handle_event(event));
        }
        let events = ok_events(results);
        assert_eq!(
            events[0],
            CanonicalStreamEvent::TextDelta("héllo 🌎".into())
        );
    }

    // WHY: SSE permits bare \r line endings; the framer must treat \r, \n, and
    // \r\n identically so a CR-only stream still parses. Mirrors codex
    // send_stream_accepts_bare_cr_sse_line_endings.
    #[test]
    fn buffer_accepts_bare_cr_line_endings() {
        let mut parser = parser();
        let mut buffer = ResponsesSseBuffer::default();
        let body = "event: response.output_text.delta\r\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\r\
\r\
event: response.completed\r\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\r\
\r";
        let mut results = Vec::new();
        for event in buffer.push(body.as_bytes()).expect("frame") {
            results.extend(parser.handle_event(event));
        }
        if let Some(event) = buffer.finish().expect("finish") {
            results.extend(parser.handle_event(event));
        }
        let events = ok_events(results);
        assert_eq!(events[0], CanonicalStreamEvent::TextDelta("ok".into()));
        assert_eq!(
            events.last().unwrap(),
            &CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into())
            }
        );
    }

    // WHY: an oversized accumulated event must be rejected to bound memory
    // against a hostile/broken upstream. Mirrors codex
    // send_stream_rejects_oversized_sse_event.
    #[test]
    fn buffer_rejects_oversized_event() {
        let mut buffer = ResponsesSseBuffer::default();
        let line = format!("data: {}\n", "x".repeat(1024));
        let body = line.repeat((MAX_SSE_EVENT_BYTES / 1024) + 2);
        let err = buffer.push(body.as_bytes()).unwrap_err();
        assert!(err.contains("event exceeded"), "{err}");
    }

    // WHY: an oversized single line must likewise be rejected before it is
    // buffered without bound.
    #[test]
    fn buffer_rejects_oversized_line() {
        let mut buffer = ResponsesSseBuffer::default();
        let body = "x".repeat(MAX_SSE_LINE_BYTES + 1);
        let err = buffer.push(body.as_bytes()).unwrap_err();
        assert!(err.contains("line exceeded"), "{err}");
    }

    // WHY: the non-stream mapper is the second entry point onto this wire. It
    // must extract id/model/content/tool_calls/usage and stamp the caller's
    // provider_tag (not a hardcoded "codex") and finish reason. Mirrors codex
    // responses_output_maps_to_canonical.
    #[test]
    fn response_to_canonical_maps_full_payload_and_tags_provider() {
        let value = json!({
            "id": "resp_backend",
            "model": "gpt-5.5",
            "service_tier": "default",
            "system_fingerprint": "fp_x",
            "status": "completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"hello","annotations":[{"type":"url_citation","url":"https://e.test"}]}]},
                {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "input_tokens_details": {"cached_tokens": 1, "audio_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 6, "audio_tokens": 7}
            }
        });
        let resp = response_to_canonical(&value, "fallback", TAG, &TokenRedactor).unwrap();
        assert_eq!(resp.id.as_deref(), Some("resp_backend"));
        assert_eq!(resp.model, "gpt-5.5");
        assert_eq!(resp.content, "hello");
        assert!(resp.refusal.is_none());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_1");
        assert_eq!(resp.tool_calls[0].name, "lookup");
        assert_eq!(resp.tool_calls[0].arguments, "{}");
        assert_eq!(resp.usage.input_tokens, 3);
        assert_eq!(resp.usage.cache_read, 1);
        assert_eq!(resp.usage.reasoning_tokens, 6);
        assert_eq!(resp.usage.audio_tokens, 12);
        assert_eq!(resp.usage.input_audio_tokens, 5);
        assert_eq!(resp.usage.output_audio_tokens, 7);
        assert_eq!(resp.annotations[0]["url"], "https://e.test");
        let meta = resp.metadata.as_ref().unwrap();
        assert_eq!(meta.service_tier.as_deref(), Some("default"));
        assert_eq!(meta.system_fingerprint.as_deref(), Some("fp_x"));
        assert_eq!(meta.provider.as_deref(), Some(TAG));
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
    }

    // WHY: a refusal + incomplete (content_filter) non-stream payload must map
    // refusal to the refusal field and preserve the incomplete reason as the
    // finish reason. Mirrors codex
    // responses_output_maps_refusal_and_content_filter_to_canonical.
    #[test]
    fn response_to_canonical_maps_refusal_and_incomplete_reason() {
        let value = json!({
            "model": "gpt-5.5",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [
                {"type":"message","content":[{"type":"refusal","refusal":"No thanks"}]}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let resp = response_to_canonical(&value, "fallback", TAG, &TokenRedactor).unwrap();
        assert_eq!(resp.content, "");
        assert_eq!(resp.refusal.as_deref(), Some("No thanks"));
        assert_eq!(resp.finish_reason.as_deref(), Some("content_filter"));
    }

    // WHY: a non-stream payload with status "failed" must become a redacted
    // upstream error, never a successful CanonicalResponse.
    #[test]
    fn response_to_canonical_failed_is_redacted_error() {
        let value = json!({"status":"failed","error":"boom sk-secret"});
        let err = response_to_canonical(&value, "fallback", TAG, &TokenRedactor)
            .unwrap_err()
            .to_string();
        assert!(!err.contains("sk-secret"), "leaked secret: {err}");
        assert!(err.contains("<redacted>"));
    }

    // WHY: incomplete with the wire reason max_output_tokens must normalize to
    // the canonical "length" finish reason.
    #[test]
    fn incomplete_reason_normalizes_max_output_tokens_to_length() {
        let value = json!({"response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}});
        assert_eq!(response_incomplete_reason(&value), "length");
    }
}
