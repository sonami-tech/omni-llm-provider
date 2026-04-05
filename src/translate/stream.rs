use serde::Serialize;

use crate::translate::response::Usage;

// ── OpenAI streaming chunk types ──────────────────────────────────

#[derive(Serialize)]
pub struct ChatCompletionChunk {
	pub id: String,
	pub object: String,
	pub created: u64,
	pub model: String,
	pub system_fingerprint: Option<()>, // Always null.
	pub choices: Vec<ChunkChoice>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub usage: Option<Usage>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
	pub index: u32,
	pub delta: ChunkDelta,
	pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct ChunkDelta {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub role: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub content: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Serialize)]
pub struct ChunkToolCall {
	pub index: u32,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub id: Option<String>,
	#[serde(rename = "type")]
	#[serde(skip_serializing_if = "Option::is_none")]
	pub call_type: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub function: Option<ChunkFunctionCall>,
}

#[derive(Serialize)]
pub struct ChunkFunctionCall {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub name: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub arguments: Option<String>,
}

fn base_chunk(
	id: &str,
	created: u64,
	model: &str,
	choices: Vec<ChunkChoice>,
	usage: Option<Usage>,
) -> ChatCompletionChunk {
	ChatCompletionChunk {
		id: id.to_string(),
		object: "chat.completion.chunk".to_string(),
		created,
		model: model.to_string(),
		system_fingerprint: None,
		choices,
		usage,
	}
}

/// Create a content delta chunk. First chunk includes role: "assistant".
pub fn content_chunk(
	id: &str,
	created: u64,
	model: &str,
	text: &str,
	is_first: bool,
) -> ChatCompletionChunk {
	base_chunk(
		id,
		created,
		model,
		vec![ChunkChoice {
			index: 0,
			delta: ChunkDelta {
				role: if is_first {
					Some("assistant".to_string())
				} else {
					None
				},
				content: Some(text.to_string()),
				tool_calls: None,
			},
			finish_reason: None,
		}],
		None,
	)
}

/// Create the finish chunk with the given finish_reason.
pub fn finish_chunk(id: &str, created: u64, model: &str, reason: &str) -> ChatCompletionChunk {
	base_chunk(
		id,
		created,
		model,
		vec![ChunkChoice {
			index: 0,
			delta: ChunkDelta {
				role: None,
				content: None,
				tool_calls: None,
			},
			finish_reason: Some(reason.to_string()),
		}],
		None,
	)
}

/// Create streaming chunks for tool calls. Returns one chunk per tool call.
pub fn tool_call_chunks(
	id: &str,
	created: u64,
	model: &str,
	tool_calls: &[crate::translate::response::ResponseToolCall],
) -> Vec<ChatCompletionChunk> {
	tool_calls
		.iter()
		.enumerate()
		.map(|(i, tc)| {
			base_chunk(
				id,
				created,
				model,
				vec![ChunkChoice {
					index: 0,
					delta: ChunkDelta {
						role: if i == 0 {
							Some("assistant".to_string())
						} else {
							None
						},
						content: None,
						tool_calls: Some(vec![ChunkToolCall {
							index: i as u32,
							id: Some(tc.id.clone()),
							call_type: Some("function".to_string()),
							function: Some(ChunkFunctionCall {
								name: Some(tc.function.name.clone()),
								arguments: Some(tc.function.arguments.clone()),
							}),
						}]),
					},
					finish_reason: None,
				}],
				None,
			)
		})
		.collect()
}

/// Create the usage chunk (empty choices, usage populated).
pub fn usage_chunk(id: &str, created: u64, model: &str, usage: Usage) -> ChatCompletionChunk {
	base_chunk(id, created, model, vec![], Some(usage))
}

/// Serialize an error event for SSE streaming.
pub fn error_event_data(message: &str) -> String {
	serde_json::json!({
		"error": {
			"message": message,
			"type": "server_error",
			"code": null,
		}
	})
	.to_string()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn content_chunk_first_has_role() {
		let chunk = content_chunk("chatcmpl-abc", 1000, "claude-sonnet-4-6", "Hello", true);
		assert_eq!(chunk.choices[0].delta.role, Some("assistant".into()));
		assert_eq!(chunk.choices[0].delta.content, Some("Hello".into()));
		assert!(chunk.choices[0].finish_reason.is_none());
		assert!(chunk.usage.is_none());
	}

	#[test]
	fn content_chunk_subsequent_no_role() {
		let chunk = content_chunk("chatcmpl-abc", 1000, "claude-sonnet-4-6", "world", false);
		assert!(chunk.choices[0].delta.role.is_none());
		assert_eq!(chunk.choices[0].delta.content, Some("world".into()));
	}

	#[test]
	fn finish_chunk_has_stop() {
		let chunk = finish_chunk("chatcmpl-abc", 1000, "claude-sonnet-4-6", "stop");
		assert_eq!(
			chunk.choices[0].finish_reason,
			Some("stop".into())
		);
		assert!(chunk.choices[0].delta.content.is_none());
		assert!(chunk.choices[0].delta.role.is_none());
	}

	#[test]
	fn usage_chunk_has_empty_choices() {
		let u = Usage {
			prompt_tokens: 10,
			completion_tokens: 5,
			total_tokens: 15,
		};
		let chunk = usage_chunk("chatcmpl-abc", 1000, "claude-sonnet-4-6", u);
		assert!(chunk.choices.is_empty());
		assert!(chunk.usage.is_some());
		assert_eq!(chunk.usage.unwrap().total_tokens, 15);
	}

	#[test]
	fn error_event_data_format() {
		let data = error_event_data("something broke");
		let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
		assert_eq!(parsed["error"]["message"], "something broke");
		assert_eq!(parsed["error"]["type"], "server_error");
		assert!(parsed["error"]["code"].is_null());
	}

	#[test]
	fn system_fingerprint_is_null_in_chunks() {
		let chunk = content_chunk("id", 0, "model", "text", true);
		let json = serde_json::to_value(&chunk).unwrap();
		assert!(json["system_fingerprint"].is_null());
	}

	#[test]
	fn serialization_format() {
		// Usage should be absent (not null) when None on chunks.
		let chunk = content_chunk("id", 0, "model", "text", false);
		let json = serde_json::to_value(&chunk).unwrap();
		assert!(!json.as_object().unwrap().contains_key("usage"));
		// finish_reason should be present as null on content chunks.
		assert!(json["choices"][0]["finish_reason"].is_null());
		assert!(json["choices"][0].as_object().unwrap().contains_key("finish_reason"));
		// Role and content should be absent (not null) when None on delta.
		let chunk = finish_chunk("id", 0, "model", "stop");
		let json = serde_json::to_value(&chunk).unwrap();
		let delta = &json["choices"][0]["delta"];
		assert!(!delta.as_object().unwrap().contains_key("role"));
		assert!(!delta.as_object().unwrap().contains_key("content"));
		// finish_reason should be present as "stop" on finish chunk.
		assert_eq!(json["choices"][0]["finish_reason"], "stop");
	}
}
