//! Translate Anthropic Messages non-streaming responses into OpenAI
//! ChatCompletion responses.

use serde_json::{Map, Value, json};

use crate::translate::anthropic::{MessagesResponse, ResponseContentBlock};

pub fn build_oai_response(
	resp: &MessagesResponse,
	chat_id: &str,
	created: u64,
	requested_model: &str,
) -> Value {
	let mut text_parts: Vec<String> = Vec::new();
	let mut reasoning_text_parts: Vec<String> = Vec::new();
	let mut reasoning_blocks: Vec<Value> = Vec::new();
	let mut tool_calls: Vec<Value> = Vec::new();

	for (i, block) in resp.content.iter().enumerate() {
		match block {
			ResponseContentBlock::Text { text } => {
				text_parts.push(text.clone());
			}
			ResponseContentBlock::ToolUse { id, name, input } => {
				tool_calls.push(json!({
					"index": i,
					"id": id,
					"type": "function",
					"function": {
						"name": name,
						"arguments": serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
					},
				}));
			}
			ResponseContentBlock::Thinking { thinking, signature } => {
				if !thinking.is_empty() {
					reasoning_text_parts.push(thinking.clone());
				}
				if let Some(sig) = signature.as_ref().filter(|s| !s.is_empty()) {
					let mut entry = Map::new();
					entry.insert("type".into(), Value::String("thinking".into()));
					entry.insert("thinking".into(), Value::String(thinking.clone()));
					entry.insert("signature".into(), Value::String(sig.clone()));
					reasoning_blocks.push(Value::Object(entry));
				}
			}
			ResponseContentBlock::Other => {}
		}
	}

	let finish_reason = map_stop_reason(resp.stop_reason.as_deref(), !tool_calls.is_empty());

	let mut message = Map::new();
	message.insert("role".into(), Value::String("assistant".into()));
	if !text_parts.is_empty() || tool_calls.is_empty() {
		message.insert("content".into(), Value::String(text_parts.join("")));
	} else {
		message.insert("content".into(), Value::Null);
	}
	if !reasoning_text_parts.is_empty() {
		message.insert(
			"reasoning_content".into(),
			Value::String(reasoning_text_parts.join("")),
		);
	}
	if !reasoning_blocks.is_empty() {
		message.insert(
			"reasoning_content_blocks".into(),
			Value::Array(reasoning_blocks),
		);
	}
	if !tool_calls.is_empty() {
		message.insert("tool_calls".into(), Value::Array(tool_calls));
	}

	let mut usage = Map::new();
	usage.insert(
		"prompt_tokens".into(),
		Value::Number(resp.usage.input_tokens.into()),
	);
	usage.insert(
		"completion_tokens".into(),
		Value::Number(resp.usage.output_tokens.into()),
	);
	let total = resp.usage.input_tokens + resp.usage.output_tokens;
	usage.insert("total_tokens".into(), Value::Number(total.into()));
	if let Some(c) = resp.usage.cache_creation_input_tokens {
		usage.insert("cache_write_tokens".into(), Value::Number(c.into()));
	}
	if let Some(c) = resp.usage.cache_read_input_tokens {
		usage.insert("cache_read_tokens".into(), Value::Number(c.into()));
	}

	json!({
		"id": chat_id,
		"object": "chat.completion",
		"created": created,
		"model": requested_model,
		"system_fingerprint": resp.id,
		"choices": [
			{
				"index": 0,
				"message": Value::Object(message),
				"finish_reason": finish_reason,
			}
		],
		"usage": Value::Object(usage),
	})
}

fn map_stop_reason(reason: Option<&str>, has_tool_calls: bool) -> &'static str {
	match reason {
		Some("end_turn") => "stop",
		Some("max_tokens") => "length",
		Some("stop_sequence") => "stop",
		Some("tool_use") => "tool_calls",
		Some("pause_turn") => "stop",
		Some("refusal") => "content_filter",
		_ if has_tool_calls => "tool_calls",
		_ => "stop",
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::translate::anthropic::{MessagesResponse, ResponseContentBlock, Usage};

	fn response_with(content: Vec<ResponseContentBlock>) -> MessagesResponse {
		MessagesResponse {
			id: "msg-test".into(),
			kind: "message".into(),
			role: "assistant".into(),
			model: "claude-test".into(),
			content,
			stop_reason: Some("end_turn".into()),
			stop_sequence: None,
			usage: Usage::default(),
		}
	}

	#[test]
	fn signed_thinking_emits_string_and_signed_blocks() {
		let resp = response_with(vec![ResponseContentBlock::Thinking {
			thinking: "private chain".into(),
			signature: Some("sig-123".into()),
		}]);

		let out = build_oai_response(&resp, "chatcmpl-test", 123, "claude-test");
		let message = &out["choices"][0]["message"];
		assert_eq!(message["reasoning_content"], "private chain");
		assert_eq!(
			message["reasoning_content_blocks"],
			json!([{
				"type": "thinking",
				"thinking": "private chain",
				"signature": "sig-123",
			}])
		);
	}

	#[test]
	fn unsigned_thinking_emits_display_string_only() {
		let resp = response_with(vec![ResponseContentBlock::Thinking {
			thinking: "display only".into(),
			signature: None,
		}]);

		let out = build_oai_response(&resp, "chatcmpl-test", 123, "claude-test");
		let message = &out["choices"][0]["message"];
		assert_eq!(message["reasoning_content"], "display only");
		assert!(message.get("reasoning_content_blocks").is_none());
	}
}
