//! Anthropic Messages SSE event stream parser.
//!
//! Anthropic's streaming endpoint emits Server-Sent Events:
//!
//! ```text
//! event: message_start
//! data: {"type":"message_start", "message": {...}}
//!
//! event: content_block_start
//! data: {"type":"content_block_start", "index":0, "content_block": {"type":"text","text":""}}
//!
//! ...
//! ```
//!
//! Each block ends with a blank line. We parse line-by-line, accumulate
//! `data:` lines, and emit a typed event on each blank line.

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
}

#[derive(Debug, Deserialize)]
struct ErrorInner {
	#[serde(rename = "type")]
	kind: String,
	message: String,
}

fn parse_event_data(json_str: &str) -> Result<StreamEvent, UpstreamError> {
	let raw_value: Value = serde_json::from_str(json_str).map_err(|e| {
		// Keep the raw SSE payload out of the client-visible error message; log
		// it server-side instead so a malformed upstream frame is not reflected
		// back to the consumer.
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
				output_tokens: usage.and_then(|u| u.output_tokens),
			}
		}
		"content_block_start" => {
			let idx = any.index.unwrap_or(0);
			let block_val = any
				.content_block
				.ok_or_else(|| UpstreamError::Decode("content_block_start missing content_block".into()))?;
			let block = parse_block_start(&block_val);
			StreamEvent::ContentBlockStart { index: idx, block }
		}
		"content_block_delta" => {
			let idx = any.index.unwrap_or(0);
			let delta_val = any
				.delta
				.ok_or_else(|| UpstreamError::Decode("content_block_delta missing delta".into()))?;
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
						d.get("stop_reason").and_then(|v| v.as_str()).map(str::to_string),
						d.get("stop_sequence").and_then(|v| v.as_str()).map(str::to_string),
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
			id: v.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string(),
			name: v.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
		},
		"thinking" => BlockStart::Thinking,
		other => BlockStart::Other(other.to_string()),
	}
}

fn parse_block_delta(v: &Value) -> BlockDelta {
	let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
	match kind {
		"text_delta" => BlockDelta::Text(
			v.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string(),
		),
		"input_json_delta" => BlockDelta::InputJson(
			v.get("partial_json")
				.and_then(|x| x.as_str())
				.unwrap_or("")
				.to_string(),
		),
		"thinking_delta" => BlockDelta::Thinking(
			v.get("thinking").and_then(|x| x.as_str()).unwrap_or("").to_string(),
		),
		"signature_delta" => BlockDelta::ThinkingSignature(
			v.get("signature").and_then(|x| x.as_str()).unwrap_or("").to_string(),
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
//
// `EventStream` above parses SSE into the lossy typed `StreamEvent`, which the
// OpenAI translation path consumes. The NATIVE Anthropic surface must instead
// forward frames byte-faithfully, so it needs the SSE `event:` name plus the
// FULL `data:` JSON value (not a typed subset). `RawEventStream` reuses the same
// `\n\n` framing (`take_event`) but yields `RawFrame { event, data }` with the
// data as the original `serde_json::Value`. Re-serializing that Value to the
// client is faithful for response SSE (no checksum rides on response key order).

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
/// The `event:` name, when absent, falls back to the data's `type` field (Anthropic
/// always sets both, but the SSE spec allows event-less frames).
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
		.or_else(|| data.get("type").and_then(|t| t.as_str()).map(str::to_string))
		.unwrap_or_else(|| "message".to_string());
	Ok(Some(RawFrame { event, data }))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parse_message_start() {
		let json = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#;
		let ev = parse_event_data(json).unwrap();
		match ev {
			StreamEvent::MessageStart {
				id,
				model,
				input_tokens,
				output_tokens,
			} => {
				assert_eq!(id, "msg_1");
				assert_eq!(model, "claude-haiku-4-5");
				assert_eq!(input_tokens, Some(10));
				assert_eq!(output_tokens, Some(1));
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
	fn parse_tool_use_block_start() {
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
	fn take_event_splits_on_double_newline() {
		let mut buf = b"event: msg\ndata: hello\n\nevent: next\n".to_vec();
		let first = take_event(&mut buf).expect("first event");
		assert_eq!(&first, b"event: msg\ndata: hello");
		assert_eq!(buf, b"event: next\n");
		assert!(take_event(&mut buf).is_none());
	}
}
