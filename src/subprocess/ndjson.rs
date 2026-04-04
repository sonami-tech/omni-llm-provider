use std::collections::HashMap;

use serde::Deserialize;

use crate::subprocess::process::SubprocessEvent;

// ── CLI NDJSON types (single-layer parse) ─────────────────────────

/// Top-level NDJSON message from the Claude CLI. Tagged on "type".
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClaudeCliMessage {
	#[serde(rename = "system")]
	System { subtype: Option<String> },
	#[serde(rename = "assistant")]
	Assistant {
		message: Option<AssistantInner>,
	},
	#[serde(rename = "stream_event")]
	StreamEvent { event: StreamEventInner },
	#[serde(rename = "result")]
	Result(ResultMessage),
	#[serde(other)]
	Unknown,
}

/// Inner event from a stream_event wrapper. Tagged on event.type.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEventInner {
	#[serde(rename = "content_block_delta")]
	ContentBlockDelta { delta: Delta },
	#[serde(rename = "message_start")]
	MessageStart {
		message: Option<MessageStartInfo>,
	},
	#[serde(rename = "message_delta")]
	MessageDelta {},
	#[serde(other)]
	Other,
}

#[derive(Debug, Deserialize)]
pub struct AssistantInner {
	pub model: Option<String>,
	// Content exists but we do NOT extract it — spec says use stream_event deltas.
}

#[derive(Debug, Deserialize)]
pub struct Delta {
	#[serde(rename = "type")]
	pub delta_type: Option<String>,
	pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MessageStartInfo {
	pub model: Option<String>,
}

// ── Result message types ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ResultMessage {
	pub subtype: Option<String>,
	pub is_error: Option<bool>,
	pub result: Option<String>,
	pub duration_ms: Option<u64>,
	pub duration_api_ms: Option<u64>,
	pub num_turns: Option<u64>,
	pub total_cost_usd: Option<f64>,
	pub usage: Option<FlatUsage>,
	#[serde(rename = "modelUsage")]
	pub model_usage: Option<HashMap<String, ModelUsage>>,
}

#[derive(Debug, Deserialize)]
pub struct FlatUsage {
	pub input_tokens: Option<u64>,
	pub output_tokens: Option<u64>,
	pub cache_creation_input_tokens: Option<u64>,
	pub cache_read_input_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelUsage {
	pub input_tokens: Option<u64>,
	pub output_tokens: Option<u64>,
	pub cache_read_input_tokens: Option<u64>,
	pub cache_creation_input_tokens: Option<u64>,
	#[serde(rename = "costUSD")]
	pub cost_usd: Option<f64>,
	pub context_window: Option<u64>,
	pub max_output_tokens: Option<u64>,
}

// ── Parsing functions ─────────────────────────────────────────────

/// Parse a single NDJSON line. Returns None for empty/unparseable lines.
pub fn parse_line(line: &str) -> Option<ClaudeCliMessage> {
	if line.trim().is_empty() {
		return None;
	}
	match serde_json::from_str::<ClaudeCliMessage>(line) {
		Ok(msg) => Some(msg),
		Err(e) => {
			tracing::debug!(error = %e, "Unparseable NDJSON line");
			None
		}
	}
}

/// Convert a parsed CLI message into zero or more SubprocessEvents.
pub fn process_message(msg: ClaudeCliMessage) -> Vec<SubprocessEvent> {
	match msg {
		ClaudeCliMessage::System { subtype, .. } => {
			if subtype.as_deref() == Some("api_retry") {
				tracing::warn!("CLI is retrying API request");
			}
			vec![]
		}
		ClaudeCliMessage::Assistant { message, .. } => {
			// Extract model name only — do NOT extract content from assistant messages.
			if let Some(inner) = message
				&& let Some(model) = inner.model
			{
				return vec![SubprocessEvent::Model(model)];
			}
			vec![]
		}
		ClaudeCliMessage::StreamEvent { event } => match event {
			StreamEventInner::ContentBlockDelta { delta } => {
				if delta.delta_type.as_deref() == Some("text_delta")
					&& let Some(text) = delta.text
					&& !text.is_empty()
				{
					return vec![SubprocessEvent::ContentDelta(text)];
				}
				// Skip thinking_delta, input_json_delta, etc.
				vec![]
			}
			StreamEventInner::MessageStart { message } => {
				if let Some(info) = message
					&& let Some(model) = info.model
				{
					return vec![SubprocessEvent::Model(model)];
				}
				vec![]
			}
			StreamEventInner::MessageDelta { .. } | StreamEventInner::Other => vec![],
		},
		ClaudeCliMessage::Result(result) => {
			vec![SubprocessEvent::Result(Box::new(result))]
		}
		ClaudeCliMessage::Unknown => vec![],
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	// ── parse_line ────────────────────────────────────────────

	#[test]
	fn parse_empty_line_returns_none() {
		assert!(parse_line("").is_none());
		assert!(parse_line("  ").is_none());
	}

	#[test]
	fn parse_invalid_json_returns_none() {
		assert!(parse_line("not json").is_none());
	}

	#[test]
	fn parse_unknown_type_returns_unknown() {
		let msg = parse_line(r#"{"type":"rate_limit_event","data":123}"#).unwrap();
		assert!(matches!(msg, ClaudeCliMessage::Unknown));
	}

	// ── System messages ───────────────────────────────────────

	#[test]
	fn parse_system_init() {
		let line = r#"{"type":"system","subtype":"init","model":"claude-sonnet-4-6","tools":[]}"#;
		let msg = parse_line(line).unwrap();
		match msg {
			ClaudeCliMessage::System { subtype } => {
				assert_eq!(subtype.unwrap(), "init");
			}
			other => panic!("Expected System, got {:?}", other),
		}
	}

	#[test]
	fn system_init_produces_no_events() {
		let msg = ClaudeCliMessage::System {
			subtype: Some("init".into()),
		};
		assert!(process_message(msg).is_empty());
	}

	// ── Assistant messages ────────────────────────────────────

	#[test]
	fn parse_assistant_with_model() {
		let line = r#"{"type":"assistant","message":{"model":"claude-sonnet-4-6","content":[{"type":"text","text":"hello"}]}}"#;
		let msg = parse_line(line).unwrap();
		let events = process_message(msg);
		assert_eq!(events.len(), 1);
		match &events[0] {
			SubprocessEvent::Model(m) => assert_eq!(m, "claude-sonnet-4-6"),
			other => panic!("Expected Model, got {:?}", other),
		}
	}

	#[test]
	fn assistant_does_not_extract_content() {
		// Even though content is present, we only extract the model.
		let line = r#"{"type":"assistant","message":{"model":"claude-sonnet-4-6","content":[{"type":"text","text":"hello world"}]}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert_eq!(events.len(), 1);
		assert!(matches!(&events[0], SubprocessEvent::Model(_)));
	}

	#[test]
	fn parse_assistant_no_message() {
		let line = r#"{"type":"assistant"}"#;
		let events = process_message(parse_line(line).unwrap());
		assert!(events.is_empty());
	}

	// ── Stream events (wrapped in stream_event envelope) ──────

	#[test]
	fn parse_stream_event_text_delta() {
		let line = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert_eq!(events.len(), 1);
		match &events[0] {
			SubprocessEvent::ContentDelta(t) => assert_eq!(t, "Hello"),
			other => panic!("Expected ContentDelta, got {:?}", other),
		}
	}

	#[test]
	fn stream_event_empty_text_skipped() {
		let line = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert!(events.is_empty());
	}

	#[test]
	fn stream_event_thinking_delta_skipped() {
		let line = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reasoning..."}}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert!(events.is_empty());
	}

	#[test]
	fn parse_stream_event_message_start_with_model() {
		let line = r#"{"type":"stream_event","event":{"type":"message_start","message":{"model":"claude-opus-4-6","id":"msg_123","role":"assistant","content":[]}}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert_eq!(events.len(), 1);
		match &events[0] {
			SubprocessEvent::Model(m) => assert_eq!(m, "claude-opus-4-6"),
			other => panic!("Expected Model, got {:?}", other),
		}
	}

	#[test]
	fn stream_event_content_block_start_ignored() {
		let line = r#"{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert!(events.is_empty());
	}

	#[test]
	fn stream_event_content_block_stop_ignored() {
		let line =
			r#"{"type":"stream_event","event":{"type":"content_block_stop","index":0}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert!(events.is_empty());
	}

	#[test]
	fn stream_event_message_delta_ignored() {
		let line = r#"{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"}}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert!(events.is_empty());
	}

	#[test]
	fn stream_event_message_stop_ignored() {
		let line = r#"{"type":"stream_event","event":{"type":"message_stop"}}"#;
		let events = process_message(parse_line(line).unwrap());
		assert!(events.is_empty());
	}

	// ── Result messages ───────────────────────────────────────

	#[test]
	fn parse_result_success() {
		let line = r#"{
			"type": "result",
			"subtype": "success",
			"is_error": false,
			"result": "Hello world.",
			"duration_ms": 1776,
			"duration_api_ms": 1729,
			"num_turns": 1,
			"total_cost_usd": 0.017064,
			"usage": {
				"input_tokens": 3,
				"output_tokens": 5,
				"cache_creation_input_tokens": 4528,
				"cache_read_input_tokens": 0
			},
			"modelUsage": {
				"claude-sonnet-4-6": {
					"inputTokens": 3,
					"outputTokens": 5,
					"cacheReadInputTokens": 0,
					"cacheCreationInputTokens": 4528,
					"costUSD": 0.017064,
					"contextWindow": 200000,
					"maxOutputTokens": 32000
				}
			}
		}"#;
		let msg = parse_line(line).unwrap();
		let events = process_message(msg);
		assert_eq!(events.len(), 1);
		match &events[0] {
			SubprocessEvent::Result(r) => {
				assert_eq!(r.subtype.as_deref(), Some("success"));
				assert_eq!(r.is_error, Some(false));
				assert_eq!(r.result.as_deref(), Some("Hello world."));
				// Check modelUsage camelCase parsing.
				let mu = r.model_usage.as_ref().unwrap();
				let usage = &mu["claude-sonnet-4-6"];
				assert_eq!(usage.input_tokens, Some(3));
				assert_eq!(usage.output_tokens, Some(5));
				assert_eq!(usage.cache_read_input_tokens, Some(0));
				assert_eq!(usage.cache_creation_input_tokens, Some(4528));
				assert_eq!(usage.cost_usd, Some(0.017064));
				assert_eq!(usage.context_window, Some(200000));
				assert_eq!(usage.max_output_tokens, Some(32000));
				// Check flat usage.
				let flat = r.usage.as_ref().unwrap();
				assert_eq!(flat.input_tokens, Some(3));
				assert_eq!(flat.output_tokens, Some(5));
			}
			other => panic!("Expected Result, got {:?}", other),
		}
	}

	#[test]
	fn parse_result_with_is_error_true() {
		let line = r#"{"type":"result","subtype":"success","is_error":true,"result":"There's an issue with the selected model.","modelUsage":{}}"#;
		let msg = parse_line(line).unwrap();
		match msg {
			ClaudeCliMessage::Result(r) => {
				assert_eq!(r.is_error, Some(true));
				assert_eq!(r.subtype.as_deref(), Some("success"));
			}
			other => panic!("Expected Result, got {:?}", other),
		}
	}

	#[test]
	fn parse_result_minimal() {
		let line = r#"{"type":"result"}"#;
		let msg = parse_line(line).unwrap();
		match msg {
			ClaudeCliMessage::Result(r) => {
				assert!(r.result.is_none());
				assert!(r.is_error.is_none());
				assert!(r.model_usage.is_none());
			}
			other => panic!("Expected Result, got {:?}", other),
		}
	}

	// ── Unknown types ─────────────────────────────────────────

	#[test]
	fn unknown_type_produces_no_events() {
		let events = process_message(ClaudeCliMessage::Unknown);
		assert!(events.is_empty());
	}

	#[test]
	fn user_type_parsed_as_unknown() {
		let line = r#"{"type":"user","message":"tool result"}"#;
		let msg = parse_line(line).unwrap();
		assert!(matches!(msg, ClaudeCliMessage::Unknown));
	}
}
