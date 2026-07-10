//! Upstream HTTPS layer to api.anthropic.com + supporting types.
//!
//! v2 talks directly to Anthropic Messages over HTTPS, mimicking the claude
//! CLI wire fingerprint. No subprocess. Credentials come from
//! `~/.claude/.credentials.json`, re-read per request.
//!
//! Port/adapted from reference-src-claude/upstream/* .
//! The client, cch finalization, 401 refresh, and header construction
//! are load-bearing for the fingerprint invariant and live ONLY here.

pub mod errors {
    //! Upstream error types and retry classification.
    use serde::Deserialize;
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum UpstreamError {
        #[error("credentials.json read failed: {0}")]
        CredentialsRead(#[source] std::io::Error),

        #[error("credentials.json malformed: {0}")]
        CredentialsParse(#[source] serde_json::Error),

        #[error("credentials.json missing accessToken")]
        CredentialsMissingToken,

        #[error(
            "OAuth token expired (per credentials.json expiresAt). Enable OMNI_OAUTH_REFRESH=1 for in-process refresh, or run `claude` once to refresh."
        )]
        TokenExpired,

        #[error("HTTP transport: {0}")]
        Transport(#[from] reqwest::Error),

        #[error("Anthropic returned HTTP {status}: {body}")]
        Anthropic {
            status: u16,
            body: String,
            parsed: Option<AnthropicErrorBody>,
        },

        #[error("response decode: {0}")]
        Decode(String),
    }

    impl UpstreamError {
        /// Classify whether the operation should be retried.
        ///
        /// 5xx and transient network errors are retryable. 429 is NOT retried — it
        /// is surfaced as-is per the locked rate-limit-passthrough decision (the
        /// retry loops also guard 429 explicitly before consulting this, so the
        /// invariant holds in one place here). 401/403/400 are terminal.
        pub fn is_transient(&self) -> bool {
            match self {
                UpstreamError::Transport(e) => e.is_timeout() || e.is_connect() || e.is_request(),
                UpstreamError::Anthropic { status, .. } => *status >= 500,
                _ => false,
            }
        }

        /// Map to an HTTP status to surface to the consumer.
        pub fn surface_status(&self) -> u16 {
            match self {
                UpstreamError::Anthropic { status, .. } => *status,
                UpstreamError::TokenExpired | UpstreamError::CredentialsMissingToken => 401,
                UpstreamError::CredentialsRead(_) | UpstreamError::CredentialsParse(_) => 500,
                UpstreamError::Transport(e) if e.is_timeout() => 504,
                UpstreamError::Transport(_) => 502,
                UpstreamError::Decode(_) => 502,
            }
        }
    }

    /// Anthropic standard error envelope: `{"type":"error","error":{"type":..,"message":..}}`.
    #[derive(Debug, Clone, Deserialize)]
    pub struct AnthropicErrorBody {
        #[serde(rename = "type")]
        pub kind: String,
        pub error: AnthropicErrorDetail,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct AnthropicErrorDetail {
        #[serde(rename = "type")]
        pub kind: String,
        pub message: String,
    }

    pub fn parse_anthropic_error(body: &str) -> Option<AnthropicErrorBody> {
        serde_json::from_str(body).ok()
    }
}

#[cfg(test)]
mod error_tests {
    use super::errors::*;

    #[test]
    fn surface_status_maps_auth_to_401() {
        assert_eq!(UpstreamError::TokenExpired.surface_status(), 401);
        assert_eq!(UpstreamError::CredentialsMissingToken.surface_status(), 401);
        assert_eq!(
            UpstreamError::CredentialsRead(std::io::Error::new(std::io::ErrorKind::NotFound, "no"))
                .surface_status(),
            500
        );
    }

    #[test]
    fn surface_status_for_anthropic_and_transient() {
        let a = UpstreamError::Anthropic {
            status: 429,
            body: "{}".into(),
            parsed: None,
        };
        assert_eq!(a.surface_status(), 429);
        // Transport construction is private in reqwest; is_transient for
        // transport is exercised via real paths and the 5xx/timeout branches
        // in client code. Surface for timeout is 504 per impl.
    }

    #[test]
    fn is_transient_only_5xx_and_some_transport() {
        let anth_5 = UpstreamError::Anthropic {
            status: 503,
            body: "".into(),
            parsed: None,
        };
        assert!(anth_5.is_transient());
        let anth_4 = UpstreamError::Anthropic {
            status: 400,
            body: "".into(),
            parsed: None,
        };
        assert!(!anth_4.is_transient());
        assert!(!UpstreamError::TokenExpired.is_transient());
    }

    #[test]
    fn parse_anthropic_error_roundtrips_standard() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#;
        let p = parse_anthropic_error(body).unwrap();
        assert_eq!(p.kind, "error");
        assert_eq!(p.error.kind, "invalid_request_error");
        assert_eq!(p.error.message, "bad");
    }
}

pub use errors::UpstreamError;

pub mod stream {
    //! Anthropic Messages SSE event stream parser.
    //!
    //! (Full port of reference-src-claude/upstream/stream.rs for client compatibility.
    //! The LlmProvider non-stream path only exercises the json send, but client
    //! code and type signatures require the stream types to be present.)

    use bytes::Bytes;
    use futures_util::Stream;
    use serde::Deserialize;
    use serde_json::Value;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use super::errors::UpstreamError;

    /// Typed Anthropic stream event.
    #[derive(Debug, Clone)]
    pub enum StreamEvent {
        MessageStart {
            id: String,
            model: String,
            input_tokens: Option<u32>,
            output_tokens: Option<u32>,
            cache_read_input_tokens: Option<u32>,
            cache_creation_input_tokens: Option<u32>,
        },
        ContentBlockStart {
            index: u32,
            block: BlockStart,
        },
        ContentBlockDelta {
            index: u32,
            delta: BlockDelta,
        },
        ContentBlockStop {
            index: u32,
        },
        MessageDelta {
            stop_reason: Option<String>,
            stop_sequence: Option<String>,
            output_tokens: Option<u32>,
        },
        MessageStop,
        Ping,
        Error {
            kind: String,
            message: String,
        },
        /// Unknown event — surfaced for forward-compat. Caller may ignore.
        Unknown(String, Value),
    }

    #[derive(Debug, Clone)]
    pub enum BlockStart {
        Text,
        ToolUse { id: String, name: String },
        Thinking,
        Other(String),
    }

    #[derive(Debug, Clone)]
    pub enum BlockDelta {
        Text(String),
        InputJson(String),
        Thinking(String),
        ThinkingSignature(String),
        Other,
    }

    // ── Wire types ────────────────────────────────────────────────────

    #[derive(Debug, Deserialize)]
    struct AnyEvent {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        index: Option<u32>,
        #[serde(default)]
        message: Option<MessageStartInner>,
        #[serde(default)]
        content_block: Option<Value>,
        #[serde(default)]
        delta: Option<Value>,
        #[serde(default)]
        usage: Option<UsageDelta>,
        #[serde(default)]
        error: Option<ErrorInner>,
    }

    #[derive(Debug, Deserialize)]
    struct MessageStartInner {
        id: String,
        model: String,
        #[serde(default)]
        usage: Option<UsageDelta>,
    }

    #[derive(Debug, Deserialize, Default)]
    struct UsageDelta {
        #[serde(default)]
        input_tokens: Option<u32>,
        #[serde(default)]
        output_tokens: Option<u32>,
        #[serde(default)]
        cache_read_input_tokens: Option<u32>,
        #[serde(default)]
        cache_creation_input_tokens: Option<u32>,
    }

    #[derive(Debug, Deserialize)]
    struct ErrorInner {
        #[serde(rename = "type")]
        kind: String,
        message: String,
    }

    fn parse_event_data(json_str: &str) -> Result<StreamEvent, UpstreamError> {
        let raw_value: Value = serde_json::from_str(json_str).map_err(|e| {
            tracing::debug!("sse data parse failed: {e}; data: {json_str}");
            UpstreamError::Decode(format!("sse data parse: {e}"))
        })?;
        let any: AnyEvent = serde_json::from_value(raw_value.clone())
            .map_err(|e| UpstreamError::Decode(format!("sse event shape: {e}")))?;

        let event = match any.kind.as_str() {
            "message_start" => {
                let inner = any
                    .message
                    .ok_or_else(|| UpstreamError::Decode("message_start missing message".into()))?;
                let usage = any.usage.or(inner.usage);
                StreamEvent::MessageStart {
                    id: inner.id,
                    model: inner.model,
                    input_tokens: usage.as_ref().and_then(|u| u.input_tokens),
                    output_tokens: usage.as_ref().and_then(|u| u.output_tokens),
                    cache_read_input_tokens: usage.as_ref().and_then(|u| u.cache_read_input_tokens),
                    cache_creation_input_tokens: usage.and_then(|u| u.cache_creation_input_tokens),
                }
            }
            "content_block_start" => {
                let idx = any.index.unwrap_or(0);
                let block_val = any.content_block.ok_or_else(|| {
                    UpstreamError::Decode("content_block_start missing content_block".into())
                })?;
                let block = parse_block_start(&block_val);
                StreamEvent::ContentBlockStart { index: idx, block }
            }
            "content_block_delta" => {
                let idx = any.index.unwrap_or(0);
                let delta_val = any.delta.ok_or_else(|| {
                    UpstreamError::Decode("content_block_delta missing delta".into())
                })?;
                StreamEvent::ContentBlockDelta {
                    index: idx,
                    delta: parse_block_delta(&delta_val),
                }
            }
            "content_block_stop" => StreamEvent::ContentBlockStop {
                index: any.index.unwrap_or(0),
            },
            "message_delta" => {
                let (stop_reason, stop_sequence) = any
                    .delta
                    .as_ref()
                    .map(|d| {
                        (
                            d.get("stop_reason")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            d.get("stop_sequence")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                        )
                    })
                    .unwrap_or((None, None));
                StreamEvent::MessageDelta {
                    stop_reason,
                    stop_sequence,
                    output_tokens: any.usage.and_then(|u| u.output_tokens),
                }
            }
            "message_stop" => StreamEvent::MessageStop,
            "ping" => StreamEvent::Ping,
            "error" => {
                let err = any.error.ok_or_else(|| {
                    UpstreamError::Decode("error event missing error inner".into())
                })?;
                StreamEvent::Error {
                    kind: err.kind,
                    message: err.message,
                }
            }
            other => StreamEvent::Unknown(other.to_string(), raw_value),
        };
        Ok(event)
    }

    fn parse_block_start(v: &Value) -> BlockStart {
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "text" => BlockStart::Text,
            "tool_use" => BlockStart::ToolUse {
                id: v
                    .get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                name: v
                    .get("name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            },
            "thinking" => BlockStart::Thinking,
            other => BlockStart::Other(other.to_string()),
        }
    }

    fn parse_block_delta(v: &Value) -> BlockDelta {
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "text_delta" => BlockDelta::Text(
                v.get("text")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            ),
            "input_json_delta" => BlockDelta::InputJson(
                v.get("partial_json")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            ),
            "thinking_delta" => BlockDelta::Thinking(
                v.get("thinking")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            ),
            "signature_delta" => BlockDelta::ThinkingSignature(
                v.get("signature")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            ),
            _ => BlockDelta::Other,
        }
    }

    // ── SSE byte-stream → event-stream ────────────────────────────────

    /// Hard cap on the SSE reassembly buffer. A well-behaved Anthropic stream
    /// emits an event-terminating blank line frequently, so this is only reached
    /// if the upstream (or an intermediary) sends an unbounded run of bytes with
    /// no event boundary. We fail the stream rather than grow memory without limit.
    const MAX_EVENT_BUFFER: usize = 16 * 1024 * 1024;

    /// Stream adapter: byte stream → typed event stream.
    pub struct EventStream<S> {
        inner: S,
        buf: Vec<u8>,
        closed: bool,
    }

    impl<S> EventStream<S> {
        pub fn new(inner: S) -> Self {
            Self {
                inner,
                buf: Vec::with_capacity(4096),
                closed: false,
            }
        }
    }

    impl<S> Stream for EventStream<S>
    where
        S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    {
        type Item = Result<StreamEvent, UpstreamError>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            loop {
                // Try to extract a full event from the buffer.
                if let Some(event_bytes) = take_event(&mut self.buf) {
                    match parse_event_block(&event_bytes) {
                        Ok(Some(ev)) => return Poll::Ready(Some(Ok(ev))),
                        Ok(None) => continue, // empty / non-data block — keep going
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }

                if self.closed {
                    return Poll::Ready(None);
                }

                // Need more bytes from upstream.
                match Pin::new(&mut self.inner).poll_next(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(None) => {
                        self.closed = true;
                        if self.buf.is_empty() {
                            return Poll::Ready(None);
                        }
                        // Try to flush whatever's left as a final event.
                        let final_bytes = std::mem::take(&mut self.buf);
                        match parse_event_block(&final_bytes) {
                            Ok(Some(ev)) => return Poll::Ready(Some(Ok(ev))),
                            _ => return Poll::Ready(None),
                        }
                    }
                    Poll::Ready(Some(Err(e))) => {
                        return Poll::Ready(Some(Err(UpstreamError::Transport(e))));
                    }
                    Poll::Ready(Some(Ok(chunk))) => {
                        self.buf.extend_from_slice(&chunk);
                        if self.buf.len() > MAX_EVENT_BUFFER {
                            self.closed = true;
                            return Poll::Ready(Some(Err(UpstreamError::Decode(format!(
                                "SSE reassembly buffer exceeded {MAX_EVENT_BUFFER} bytes without an event boundary"
                            )))));
                        }
                    }
                }
            }
        }
    }

    /// Extract one complete event-block (terminated by `\n\n`) from buf and remove
    /// it. Returns None if no complete event is yet present.
    fn take_event(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
        // Find the first occurrence of either `\n\n` or `\r\n\r\n`.
        let mut end = None;
        let mut sep_len = 0;
        for i in 0..buf.len() {
            if buf[i] == b'\n' && i > 0 && buf[i - 1] == b'\n' {
                end = Some(i + 1);
                sep_len = 2;
                break;
            }
            if i >= 3
                && buf[i] == b'\n'
                && buf[i - 1] == b'\r'
                && buf[i - 2] == b'\n'
                && buf[i - 3] == b'\r'
            {
                end = Some(i + 1);
                sep_len = 4;
                break;
            }
        }
        let end = end?;
        let event_end = end - sep_len;
        let event_bytes = buf[..event_end].to_vec();
        buf.drain(..end);
        Some(event_bytes)
    }

    /// Parse a complete event block (one or more lines, no trailing `\n\n`) into
    /// an event. Returns `Ok(None)` if the block contains no `data:` line (e.g. a
    /// pure comment block).
    fn parse_event_block(bytes: &[u8]) -> Result<Option<StreamEvent>, UpstreamError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| UpstreamError::Decode(format!("sse not utf-8: {e}")))?;
        let mut data_lines: Vec<&str> = Vec::new();
        for line in text.split('\n') {
            let line = line.trim_end_matches('\r');
            if line.starts_with(":") || line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("data: ") {
                data_lines.push(rest);
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest);
            }
            // `event:` and `id:` lines are noted in raw form but we use the
            // `type` field inside the JSON for dispatch, so we ignore them.
        }
        if data_lines.is_empty() {
            return Ok(None);
        }
        let joined = data_lines.join("\n");
        let event = parse_event_data(&joined)?;
        Ok(Some(event))
    }

    // ── Raw (faithful) frame stream ───────────────────────────────────────────────

    /// One raw SSE frame: the `event:` name and the parsed-but-untyped `data:` JSON.
    #[derive(Debug, Clone)]
    pub struct RawFrame {
        pub event: String,
        pub data: Value,
    }

    /// SSE stream that yields faithful raw frames for the native Anthropic surface.
    pub struct RawEventStream<S> {
        inner: S,
        buf: Vec<u8>,
        closed: bool,
    }

    impl<S> RawEventStream<S> {
        pub fn new(inner: S) -> Self {
            Self {
                inner,
                buf: Vec::with_capacity(4096),
                closed: false,
            }
        }
    }

    impl<S> Stream for RawEventStream<S>
    where
        S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    {
        type Item = Result<RawFrame, UpstreamError>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            loop {
                if let Some(event_bytes) = take_event(&mut self.buf) {
                    match parse_raw_frame(&event_bytes) {
                        Ok(Some(frame)) => return Poll::Ready(Some(Ok(frame))),
                        Ok(None) => continue, // comment / no data line — skip
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }

                if self.closed {
                    return Poll::Ready(None);
                }

                match Pin::new(&mut self.inner).poll_next(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(None) => {
                        self.closed = true;
                        if self.buf.is_empty() {
                            return Poll::Ready(None);
                        }
                        let final_bytes = std::mem::take(&mut self.buf);
                        match parse_raw_frame(&final_bytes) {
                            Ok(Some(frame)) => return Poll::Ready(Some(Ok(frame))),
                            _ => return Poll::Ready(None),
                        }
                    }
                    Poll::Ready(Some(Err(e))) => {
                        return Poll::Ready(Some(Err(UpstreamError::Transport(e))));
                    }
                    Poll::Ready(Some(Ok(chunk))) => {
                        self.buf.extend_from_slice(&chunk);
                        if self.buf.len() > MAX_EVENT_BUFFER {
                            self.closed = true;
                            return Poll::Ready(Some(Err(UpstreamError::Decode(format!(
                                "SSE reassembly buffer exceeded {MAX_EVENT_BUFFER} bytes without an event boundary"
                            )))));
                        }
                    }
                }
            }
        }
    }

    /// Parse a complete event block into a `RawFrame`, preserving the full data JSON.
    /// Returns `Ok(None)` for a block with no `data:` line (e.g. a pure comment).
    fn parse_raw_frame(bytes: &[u8]) -> Result<Option<RawFrame>, UpstreamError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| UpstreamError::Decode(format!("sse not utf-8: {e}")))?;
        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<&str> = Vec::new();
        for line in text.split('\n') {
            let line = line.trim_end_matches('\r');
            if line.starts_with(':') || line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("event:") {
                event_name = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
            }
        }
        if data_lines.is_empty() {
            return Ok(None);
        }
        let joined = data_lines.join("\n");
        let data: Value = serde_json::from_str(&joined)
            .map_err(|e| UpstreamError::Decode(format!("sse data not json: {e}")))?;
        let event = event_name
            .or_else(|| {
                data.get("type")
                    .and_then(|t| t.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "message".to_string());
        Ok(Some(RawFrame { event, data }))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_message_start() {
            // WHY the cache_* fields are here: Anthropic reports cache read/creation
            // token counts in the message_start usage object on a streamed request.
            // The parser must carry them through; a regression that dropped them
            // (as an earlier version did) would silently zero the cache usage we
            // report to streaming clients. This asserts they survive the parse.
            let json = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":7,"cache_creation_input_tokens":3}}}"#;
            let ev = parse_event_data(json).unwrap();
            match ev {
                StreamEvent::MessageStart {
                    id,
                    model,
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                } => {
                    assert_eq!(id, "msg_1");
                    assert_eq!(model, "claude-haiku-4-5");
                    assert_eq!(input_tokens, Some(10));
                    assert_eq!(output_tokens, Some(1));
                    assert_eq!(cache_read_input_tokens, Some(7));
                    assert_eq!(cache_creation_input_tokens, Some(3));
                }
                _ => panic!("wrong variant"),
            }
        }

        #[test]
        fn parse_text_delta() {
            let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
            let ev = parse_event_data(json).unwrap();
            match ev {
                StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: BlockDelta::Text(s),
                } => assert_eq!(s, "hi"),
                _ => panic!("wrong variant: {:?}", ev),
            }
        }

        #[test]
        fn take_event_splits_on_double_newline() {
            let mut buf = b"event: msg\ndata: hello\n\nevent: next\n".to_vec();
            let first = take_event(&mut buf).expect("first event");
            assert_eq!(&first, b"event: msg\ndata: hello");
            assert_eq!(buf, b"event: next\n");
            assert!(take_event(&mut buf).is_none());
        }

        #[test]
        fn parse_tool_use_block_start() {
            // Covers tool call start in stream (name/ id for tool use blocks); part of
            // streaming surface that must roundtrip for native passthrough parity.
            let json = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_x","name":"foo","input":{}}}"#;
            let ev = parse_event_data(json).unwrap();
            match ev {
                StreamEvent::ContentBlockStart {
                    index: 1,
                    block: BlockStart::ToolUse { id, name },
                } => {
                    assert_eq!(id, "toolu_x");
                    assert_eq!(name, "foo");
                }
                _ => panic!("wrong variant: {:?}", ev),
            }
        }

        #[test]
        fn parse_input_json_delta_accumulates_tool_args() {
            // Tool arg streaming uses partial_json deltas; the RawSse / typed path
            // must surface them without loss for downstream accumulation (defer full
            // object to stop). Mirrors CCP streaming repl/arg buffer tests intent.
            let json = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}"#;
            let ev = parse_event_data(json).unwrap();
            match ev {
                StreamEvent::ContentBlockDelta {
                    index: 1,
                    delta: BlockDelta::InputJson(s),
                } => assert_eq!(s, "{\"a\":1}"),
                _ => panic!("wrong: {:?}", ev),
            }
        }

        #[test]
        fn parse_raw_frame_yields_event_and_full_data() {
            // Raw SSE path (for native Anthropic passthrough) preserves the full
            // data JSON and event name; used for byte-faithful forward of deltas,
            // tool args, usage etc without the typed lossy map.
            let text = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n";
            let bytes = text.as_bytes().to_vec();
            // simulate via the block parser (take_event would split)
            let frame = parse_raw_frame(&bytes[0..bytes.len() - 2])
                .unwrap()
                .unwrap();
            assert_eq!(frame.event, "content_block_delta");
            assert_eq!(frame.data["delta"]["text"], "hi");
        }
    }
}

pub use stream::{BlockDelta, BlockStart, EventStream, RawEventStream, RawFrame, StreamEvent};

use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use futures_util::Stream;
use reqwest::{Client, header::HeaderMap};
use serde_json::Value;
use tracing::{debug, warn};

use omni_common::{headers_from_env, parse_custom_headers};

use crate::credentials::Credentials;
use crate::fingerprint::{FingerprintProfile, RequestContext, build_headers, default_profile};

/// Retry budget for 5xx + transient transport errors. Per locked decision #2,
/// 429 is NEVER retried; it's surfaced as-is to the consumer. 401 is retried
/// once after a fresh credentials.json re-read.
const MAX_RETRIES: u32 = 2;

const DEFAULT_ANTHROPIC_BASE: &str = "https://api.anthropic.com";

#[derive(Clone)]
pub struct UpstreamClient {
    http: Client,
    profile: &'static FingerprintProfile,
    /// Upstream base URL (scheme + host[:port]), no trailing slash and no path.
    /// Production uses [`DEFAULT_ANTHROPIC_BASE`]; tests point it at a local mock
    /// server. The request paths (`/v1/messages?beta=true` etc.) are appended per
    /// call, so the exact wire path is identical for production and tests.
    base: String,
    auth: ClaudeAuthConfig,
}

#[derive(Clone, Debug)]
pub(crate) enum ClaudeAuthConfig {
    Default,
    Custom {
        authorization_bearer: Option<String>,
        headers: Vec<(String, String)>,
        authorization_bearer_env: Option<String>,
        api_key_env: Option<String>,
        custom_headers_env: Option<String>,
    },
}

impl ClaudeAuthConfig {
    pub(crate) fn is_custom(&self) -> bool {
        matches!(self, Self::Custom { .. })
    }
}

impl UpstreamClient {
    pub fn new() -> Result<Self, UpstreamError> {
        Self::new_with_profile(default_profile())
    }

    pub fn new_with_profile(profile: &'static FingerprintProfile) -> Result<Self, UpstreamError> {
        Self::new_with_profile_and_base(profile, DEFAULT_ANTHROPIC_BASE)
    }

    /// Construct against an explicit base URL, reusing the production HTTP client
    /// builder verbatim. The builder pins `http2_prior_knowledge`, so an
    /// `http://` test base reaches a wiremock server over cleartext HTTP/2 (h2c),
    /// which wiremock serves via hyper-util's protocol auto-detection. The
    /// transport config is therefore the SAME one production uses, not a test
    /// stand-in. `pub(crate)` so `ClaudeProvider::new_for_test_with_base` can
    /// forward to it without widening the public surface further than needed.
    pub(crate) fn new_with_profile_and_base(
        profile: &'static FingerprintProfile,
        base: impl Into<String>,
    ) -> Result<Self, UpstreamError> {
        Self::new_with_profile_base_and_auth(profile, base, ClaudeAuthConfig::Default)
    }

    pub(crate) fn new_with_profile_base_and_auth(
        profile: &'static FingerprintProfile,
        base: impl Into<String>,
        auth: ClaudeAuthConfig,
    ) -> Result<Self, UpstreamError> {
        validate_auth_config(&auth).map_err(UpstreamError::Decode)?;
        let http = Client::builder()
            .use_rustls_tls()
            .http2_prior_knowledge()
            .http2_keep_alive_interval(Some(Duration::from_secs(30)))
            .tcp_keepalive(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(600))
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .pool_max_idle_per_host(8)
            .build()
            .map_err(UpstreamError::Transport)?;
        let base = base.into().trim_end_matches('/').to_string();
        Ok(Self {
            http,
            profile,
            base,
            auth,
        })
    }

    /// The `/v1/messages` URL for this client's base, preserving the `?beta=true`
    /// query the production endpoint requires. Lives in one place so the
    /// non-stream and streaming POST sites cannot drift.
    fn messages_url(&self) -> String {
        format!("{}/v1/messages?beta=true", self.base)
    }

    /// The `/v1/messages/count_tokens` URL for this client's base.
    fn count_tokens_url(&self) -> String {
        format!("{}/v1/messages/count_tokens?beta=true", self.base)
    }

    pub(crate) fn uses_custom_auth(&self) -> bool {
        self.auth.is_custom()
    }

    /// Send a non-streaming JSON body to /v1/messages with retry on 5xx and
    /// transient errors. Returns the parsed JSON response on 2xx, or a typed
    /// UpstreamError otherwise. 429 is NOT retried.
    pub async fn send_messages_json(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: &Value,
    ) -> Result<Value, UpstreamError> {
        let mut creds_owned = creds.clone();
        let mut ctx_owned = ctx.clone();
        let mut refreshed_credentials = false;

        for attempt in 0..=MAX_RETRIES {
            let body_bytes = self
                .profile
                .finalize_body_json(body, &ctx_owned)
                .map_err(|e| UpstreamError::Decode(format!("request json serialize: {e}")))?;
            let result = self
                .send_messages_json_once(&creds_owned, &ctx_owned, body_bytes)
                .await;
            match result {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let status = match &e {
                        UpstreamError::Anthropic { status, .. } => Some(*status),
                        _ => None,
                    };

                    // 401 on the default Claude Code path: re-read credentials
                    // (and force OAuth refresh when OMNI_OAUTH_REFRESH is on).
                    // Custom gateways own their auth, so never fall back to the
                    // local OAuth file.
                    if status == Some(401) && !refreshed_credentials && !self.auth.is_custom() {
                        refreshed_credentials = true;
                        let path = Credentials::default_path();
                        match Credentials::load_fresh_async_force_refresh(&path).await {
                            Ok(fresh) => {
                                warn!(
                                    "upstream 401, re-reading credentials.json (oauth refresh if enabled) and retrying"
                                );
                                creds_owned = fresh;
                                ctx_owned.next_attempt();
                                continue;
                            }
                            Err(_) => return Err(e),
                        }
                    }

                    // 429: pass-through, no retry.
                    if status == Some(429) {
                        return Err(e);
                    }

                    // Other terminal: pass-through.
                    if !e.is_transient() {
                        return Err(e);
                    }

                    if attempt >= MAX_RETRIES {
                        return Err(e);
                    }

                    let backoff = Duration::from_millis(250u64.saturating_mul(1u64 << attempt));
                    debug!(
                        attempt = attempt + 1,
                        ?backoff,
                        "transient upstream error, retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    ctx_owned.next_attempt();
                }
            }
        }
        // Loop exits via return; this is unreachable.
        Err(UpstreamError::Decode("retry loop exhausted".into()))
    }

    /// Send a `count_tokens` request. Same fingerprint headers + 401-refresh
    /// retry as the messages path, but posts to the count_tokens endpoint and
    /// returns the parsed `{"input_tokens": N}` JSON. 429 is NOT retried.
    ///
    /// NOTE: the body is sent VERBATIM (not run through `finalize_body_json`):
    /// count_tokens counts the client body as-sent and carries no injected
    /// billing-header placeholder, so there is no cch to finalize. This matches
    /// the native-surface decision to count the client's own body.
    pub async fn count_tokens(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: &Value,
    ) -> Result<Value, UpstreamError> {
        let mut creds_owned = creds.clone();
        let mut ctx_owned = ctx.clone();
        let mut refreshed_credentials = false;

        for attempt in 0..=MAX_RETRIES {
            let body_bytes = serde_json::to_vec(body)
                .map_err(|e| UpstreamError::Decode(format!("count_tokens serialize: {e}")))?;
            let result = self
                .post_json_once(
                    &self.count_tokens_url(),
                    &creds_owned,
                    &ctx_owned,
                    body_bytes,
                )
                .await;
            match result {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let status = match &e {
                        UpstreamError::Anthropic { status, .. } => Some(*status),
                        _ => None,
                    };
                    if status == Some(401) && !refreshed_credentials && !self.auth.is_custom() {
                        refreshed_credentials = true;
                        let path = Credentials::default_path();
                        match Credentials::load_fresh_async_force_refresh(&path).await {
                            Ok(fresh) => {
                                warn!(
                                    "count_tokens 401, re-reading credentials.json (oauth refresh if enabled) and retrying"
                                );
                                creds_owned = fresh;
                                ctx_owned.next_attempt();
                                continue;
                            }
                            Err(_) => return Err(e),
                        }
                    }
                    if status == Some(429) || !e.is_transient() || attempt >= MAX_RETRIES {
                        return Err(e);
                    }
                    let backoff = Duration::from_millis(250u64.saturating_mul(1u64 << attempt));
                    tokio::time::sleep(backoff).await;
                    ctx_owned.next_attempt();
                }
            }
        }
        Err(UpstreamError::Decode(
            "count_tokens retry loop exhausted".into(),
        ))
    }

    async fn send_messages_json_once(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: Vec<u8>,
    ) -> Result<Value, UpstreamError> {
        self.post_json_once(&self.messages_url(), creds, ctx, body)
            .await
    }

    /// POST a finalized body to a JSON endpoint and parse the 2xx body, or map a
    /// non-2xx into a typed `UpstreamError::Anthropic`. Shared by the messages
    /// and count_tokens non-streaming paths.
    async fn post_json_once(
        &self,
        url: &str,
        creds: &Credentials,
        ctx: &RequestContext,
        body: Vec<u8>,
    ) -> Result<Value, UpstreamError> {
        let headers = self.build_headers(creds, ctx)?;

        let resp = self
            .http
            .post(url)
            .headers(headers)
            .body(body)
            .send()
            .await?;

        let status = resp.status();
        let bytes = resp.bytes().await?;

        if status.is_success() {
            serde_json::from_slice(&bytes)
                .map_err(|e| UpstreamError::Decode(format!("response json parse: {e}")))
        } else {
            let body_str = String::from_utf8_lossy(&bytes).into_owned();
            let parsed = errors::parse_anthropic_error(&body_str);
            Err(UpstreamError::Anthropic {
                status: status.as_u16(),
                body: body_str,
                parsed,
            })
        }
    }

    /// Send a streaming Messages request with retry on the *initial response*
    /// only. Once the byte stream begins flowing we cannot retry without
    /// re-emitting bytes the consumer has already seen. 429 is pass-through.
    pub async fn send_messages_stream(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: &Value,
    ) -> Result<
        impl Stream<Item = Result<StreamEvent, UpstreamError>> + Send + 'static + use<>,
        UpstreamError,
    > {
        let mut creds_owned = creds.clone();
        let mut ctx_owned = ctx.clone();
        let mut refreshed_credentials = false;

        for attempt in 0..=MAX_RETRIES {
            let body_bytes = self
                .profile
                .finalize_body_json(body, &ctx_owned)
                .map_err(|e| UpstreamError::Decode(format!("request json serialize: {e}")))?;
            match self
                .send_messages_stream_once(&creds_owned, &ctx_owned, body_bytes)
                .await
            {
                Ok(s) => return Ok(s),
                Err(e) => {
                    let status = match &e {
                        UpstreamError::Anthropic { status, .. } => Some(*status),
                        _ => None,
                    };
                    if status == Some(401) && !refreshed_credentials && !self.auth.is_custom() {
                        refreshed_credentials = true;
                        let path = Credentials::default_path();
                        if let Ok(fresh) =
                            Credentials::load_fresh_async_force_refresh(&path).await
                        {
                            warn!(
                                "upstream 401 on stream open, re-reading credentials.json (oauth refresh if enabled)"
                            );
                            creds_owned = fresh;
                            ctx_owned.next_attempt();
                            continue;
                        }
                        return Err(e);
                    }
                    if status == Some(429) || !e.is_transient() || attempt >= MAX_RETRIES {
                        return Err(e);
                    }
                    let backoff = Duration::from_millis(250u64.saturating_mul(1u64 << attempt));
                    debug!(
                        attempt = attempt + 1,
                        ?backoff,
                        "transient stream-open error, retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    ctx_owned.next_attempt();
                }
            }
        }
        Err(UpstreamError::Decode("retry loop exhausted".into()))
    }

    async fn send_messages_stream_once(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: Vec<u8>,
    ) -> Result<EventStream<BoxByteStream>, UpstreamError> {
        let boxed = self.open_message_byte_stream(creds, ctx, body).await?;
        Ok(EventStream::new(boxed))
    }

    /// Streaming variant for the NATIVE Anthropic surface: yields faithful raw
    /// SSE frames (`RawFrame { event, data }`) instead of the lossy typed
    /// `StreamEvent`, so the handler can forward Anthropic's wire format
    /// byte-faithfully. Same fingerprint headers, body finalization, and
    /// stream-open retry semantics as `send_messages_stream`.
    pub async fn send_messages_stream_raw(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: &Value,
    ) -> Result<
        impl Stream<Item = Result<RawFrame, UpstreamError>> + Send + 'static + use<>,
        UpstreamError,
    > {
        let mut creds_owned = creds.clone();
        let mut ctx_owned = ctx.clone();
        let mut refreshed_credentials = false;

        for attempt in 0..=MAX_RETRIES {
            let body_bytes = self
                .profile
                .finalize_body_json(body, &ctx_owned)
                .map_err(|e| UpstreamError::Decode(format!("request json serialize: {e}")))?;
            match self
                .open_message_byte_stream(&creds_owned, &ctx_owned, body_bytes)
                .await
            {
                Ok(boxed) => return Ok(RawEventStream::new(boxed)),
                Err(e) => {
                    let status = match &e {
                        UpstreamError::Anthropic { status, .. } => Some(*status),
                        _ => None,
                    };
                    if status == Some(401) && !refreshed_credentials && !self.auth.is_custom() {
                        refreshed_credentials = true;
                        let path = Credentials::default_path();
                        if let Ok(fresh) =
                            Credentials::load_fresh_async_force_refresh(&path).await
                        {
                            warn!(
                                "upstream 401 on raw stream open, re-reading credentials.json (oauth refresh if enabled)"
                            );
                            creds_owned = fresh;
                            ctx_owned.next_attempt();
                            continue;
                        }
                        return Err(e);
                    }
                    if status == Some(429) || !e.is_transient() || attempt >= MAX_RETRIES {
                        return Err(e);
                    }
                    let backoff = Duration::from_millis(250u64.saturating_mul(1u64 << attempt));
                    tokio::time::sleep(backoff).await;
                    ctx_owned.next_attempt();
                }
            }
        }
        Err(UpstreamError::Decode("retry loop exhausted".into()))
    }

    /// Open the messages SSE byte stream, mapping a non-2xx response to a typed
    /// error. Shared by the typed and raw streaming paths.
    async fn open_message_byte_stream(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
        body: Vec<u8>,
    ) -> Result<BoxByteStream, UpstreamError> {
        let headers = self.build_headers(creds, ctx)?;
        let resp = self
            .http
            .post(self.messages_url())
            .headers(headers)
            .body(body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let bytes = resp.bytes().await?;
            let body_str = String::from_utf8_lossy(&bytes).into_owned();
            let parsed = errors::parse_anthropic_error(&body_str);
            return Err(UpstreamError::Anthropic {
                status: status.as_u16(),
                body: body_str,
                parsed,
            });
        }

        let boxed: BoxByteStream = Box::pin(resp.bytes_stream());
        Ok(boxed)
    }

    fn build_headers(
        &self,
        creds: &Credentials,
        ctx: &RequestContext,
    ) -> Result<HeaderMap, UpstreamError> {
        let mut headers = build_headers(creds, ctx, self.profile);
        match &self.auth {
            ClaudeAuthConfig::Default => {}
            ClaudeAuthConfig::Custom {
                authorization_bearer,
                headers: custom_headers,
                authorization_bearer_env,
                api_key_env,
                custom_headers_env,
            } => {
                headers.remove("authorization");
                for (name, value) in custom_headers {
                    insert_custom_header(&mut headers, name, value);
                }
                if let Some(env_name) = custom_headers_env {
                    for (name, value) in
                        headers_from_env(env_name).map_err(UpstreamError::Decode)?
                    {
                        insert_custom_header(&mut headers, &name, &value);
                    }
                }
                let env_bearer = authorization_bearer_env.as_ref().and_then(|env_name| {
                    std::env::var(env_name)
                        .ok()
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty())
                });
                let token = authorization_bearer
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                    .cloned()
                    .or(env_bearer);
                if let Some(token) = token {
                    insert_authorization_bearer(&mut headers, &token);
                } else if !custom_headers
                    .iter()
                    .any(|(name, _)| name.eq_ignore_ascii_case("x-api-key"))
                    && let Some(api_key) = api_key_env.as_ref().and_then(|env_name| {
                        std::env::var(env_name)
                            .ok()
                            .map(|value| value.trim().to_string())
                            .filter(|value| !value.is_empty())
                    })
                {
                    insert_custom_header(&mut headers, "x-api-key", &api_key);
                }
            }
        }
        Ok(headers)
    }
}

fn validate_auth_config(auth: &ClaudeAuthConfig) -> Result<(), String> {
    let ClaudeAuthConfig::Custom {
        headers,
        custom_headers_env,
        ..
    } = auth
    else {
        return Ok(());
    };
    for (name, value) in headers {
        validate_custom_header(name, value)?;
    }
    if let Some(env_name) = custom_headers_env
        && let Ok(raw) = std::env::var(env_name)
    {
        for (name, value) in parse_custom_headers(&raw)? {
            validate_custom_header(&name, &value)?;
        }
    }
    Ok(())
}

fn validate_custom_header(name: &str, value: &str) -> Result<(), String> {
    reqwest::header::HeaderName::from_bytes(name.as_bytes())
        .map_err(|_| format!("invalid custom header name `{name}`"))?;
    reqwest::header::HeaderValue::from_str(value)
        .map_err(|_| format!("invalid custom header value for `{name}`"))?;
    Ok(())
}

fn insert_custom_header(headers: &mut HeaderMap, name: &str, value: &str) {
    if let (Ok(name), Ok(value)) = (
        reqwest::header::HeaderName::from_bytes(name.as_bytes()),
        reqwest::header::HeaderValue::from_str(value),
    ) {
        headers.insert(name, value);
    }
}

fn insert_authorization_bearer(headers: &mut HeaderMap, token: &str) {
    if let Ok(value) = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")) {
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
}

type BoxByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
