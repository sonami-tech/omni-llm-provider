use serde::Serialize;

use crate::subprocess::ndjson::ResultMessage;

// ── OpenAI response types (non-streaming) ─────────────────────────

#[derive(Serialize)]
pub struct ChatCompletionResponse {
	pub id: String,
	pub object: String,
	pub created: u64,
	pub model: String,
	pub system_fingerprint: Option<()>, // Always null.
	pub choices: Vec<Choice>,
	pub usage: Option<Usage>,
}

#[derive(Serialize)]
pub struct Choice {
	pub index: u32,
	pub message: ResponseMessage,
	pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ResponseMessage {
	pub role: String,
	pub content: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Serialize, Clone)]
pub struct ResponseToolCall {
	pub id: String,
	#[serde(rename = "type")]
	pub call_type: String,
	pub function: ResponseFunctionCall,
}

#[derive(Serialize, Clone)]
pub struct ResponseFunctionCall {
	pub name: String,
	pub arguments: String,
}

#[derive(Serialize, Clone)]
pub struct Usage {
	pub prompt_tokens: u64,
	pub completion_tokens: u64,
	pub total_tokens: u64,
}

/// Extract usage from a ResultMessage. Prefers modelUsage (camelCase), falls back to flat usage.
pub fn extract_usage(result: &ResultMessage) -> Option<Usage> {
	// Try modelUsage first (sum all entries).
	if let Some(mu) = &result.model_usage
		&& !mu.is_empty()
	{
		let mut input = 0u64;
		let mut output = 0u64;
		for u in mu.values() {
			input += u.input_tokens.unwrap_or(0);
			output += u.output_tokens.unwrap_or(0);
		}
		return Some(Usage {
			prompt_tokens: input,
			completion_tokens: output,
			total_tokens: input + output,
		});
	}

	// Fall back to flat usage.
	if let Some(flat) = &result.usage {
		let input = flat.input_tokens.unwrap_or(0);
		let output = flat.output_tokens.unwrap_or(0);
		return Some(Usage {
			prompt_tokens: input,
			completion_tokens: output,
			total_tokens: input + output,
		});
	}

	None
}

/// Build a non-streaming ChatCompletionResponse.
pub fn build_response(
	chat_id: &str,
	created: u64,
	model: &str,
	content: &str,
	result: &ResultMessage,
) -> ChatCompletionResponse {
	ChatCompletionResponse {
		id: chat_id.to_string(),
		object: "chat.completion".to_string(),
		created,
		model: model.to_string(),
		system_fingerprint: None,
		choices: vec![Choice {
			index: 0,
			message: ResponseMessage {
				role: "assistant".to_string(),
				content: Some(content.to_string()),
				tool_calls: None,
			},
			finish_reason: "stop".to_string(),
		}],
		usage: extract_usage(result),
	}
}

/// Build a non-streaming response containing tool calls.
pub fn build_tool_call_response(
	chat_id: &str,
	created: u64,
	model: &str,
	tool_calls: Vec<ResponseToolCall>,
	result: &ResultMessage,
) -> ChatCompletionResponse {
	ChatCompletionResponse {
		id: chat_id.to_string(),
		object: "chat.completion".to_string(),
		created,
		model: model.to_string(),
		system_fingerprint: None,
		choices: vec![Choice {
			index: 0,
			message: ResponseMessage {
				role: "assistant".to_string(),
				content: None,
				tool_calls: Some(tool_calls),
			},
			finish_reason: "tool_calls".to_string(),
		}],
		usage: extract_usage(result),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::subprocess::ndjson::{FlatUsage, ModelUsage, ResultMessage};
	use std::collections::HashMap;

	#[test]
	fn extract_usage_from_model_usage() {
		let mut mu = HashMap::new();
		mu.insert(
			"claude-sonnet-4-6".to_string(),
			ModelUsage {
				input_tokens: Some(100),
				output_tokens: Some(50),
				cache_read_input_tokens: Some(10),
				cache_creation_input_tokens: Some(5),
				cost_usd: None,
				context_window: None,
				max_output_tokens: None,
			},
		);
		let result = ResultMessage {
			subtype: None,
			is_error: None,
			result: None,
			duration_ms: None,
			duration_api_ms: None,
			num_turns: None,
			total_cost_usd: None,
			usage: None,
			model_usage: Some(mu),
		};
		let usage = extract_usage(&result).unwrap();
		assert_eq!(usage.prompt_tokens, 100);
		assert_eq!(usage.completion_tokens, 50);
		assert_eq!(usage.total_tokens, 150);
	}

	#[test]
	fn extract_usage_falls_back_to_flat() {
		let result = ResultMessage {
			subtype: None,
			is_error: None,
			result: None,
			duration_ms: None,
			duration_api_ms: None,
			num_turns: None,
			total_cost_usd: None,
			usage: Some(FlatUsage {
				input_tokens: Some(200),
				output_tokens: Some(100),
				cache_creation_input_tokens: None,
				cache_read_input_tokens: None,
			}),
			model_usage: None,
		};
		let usage = extract_usage(&result).unwrap();
		assert_eq!(usage.prompt_tokens, 200);
		assert_eq!(usage.completion_tokens, 100);
	}

	#[test]
	fn extract_usage_none_when_empty() {
		let result = ResultMessage {
			subtype: None,
			is_error: None,
			result: None,
			duration_ms: None,
			duration_api_ms: None,
			num_turns: None,
			total_cost_usd: None,
			usage: None,
			model_usage: None,
		};
		assert!(extract_usage(&result).is_none());
	}

	#[test]
	fn extract_usage_sums_multiple_models() {
		let mut mu = HashMap::new();
		mu.insert(
			"claude-opus-4-6".to_string(),
			ModelUsage {
				input_tokens: Some(100),
				output_tokens: Some(50),
				cache_read_input_tokens: None,
				cache_creation_input_tokens: None,
				cost_usd: None,
				context_window: None,
				max_output_tokens: None,
			},
		);
		mu.insert(
			"claude-sonnet-4-6".to_string(),
			ModelUsage {
				input_tokens: Some(200),
				output_tokens: Some(100),
				cache_read_input_tokens: None,
				cache_creation_input_tokens: None,
				cost_usd: None,
				context_window: None,
				max_output_tokens: None,
			},
		);
		let result = ResultMessage {
			subtype: None,
			is_error: None,
			result: None,
			duration_ms: None,
			duration_api_ms: None,
			num_turns: None,
			total_cost_usd: None,
			usage: None,
			model_usage: Some(mu),
		};
		let usage = extract_usage(&result).unwrap();
		assert_eq!(usage.prompt_tokens, 300);
		assert_eq!(usage.completion_tokens, 150);
		assert_eq!(usage.total_tokens, 450);
	}

	#[test]
	fn build_response_basic() {
		let result = ResultMessage {
			subtype: Some("success".into()),
			is_error: Some(false),
			result: Some("Hello".into()),
			duration_ms: None,
			duration_api_ms: None,
			num_turns: None,
			total_cost_usd: None,
			usage: None,
			model_usage: None,
		};
		let resp = build_response("chatcmpl-abc123", 1000, "claude-sonnet-4-6", "Hello", &result);
		assert_eq!(resp.id, "chatcmpl-abc123");
		assert_eq!(resp.object, "chat.completion");
		assert_eq!(resp.model, "claude-sonnet-4-6");
		assert!(resp.system_fingerprint.is_none());
		assert_eq!(resp.choices.len(), 1);
		assert_eq!(resp.choices[0].message.role, "assistant");
		assert_eq!(resp.choices[0].message.content, Some("Hello".into()));
		assert_eq!(resp.choices[0].finish_reason, "stop");
	}

	#[test]
	fn system_fingerprint_serializes_as_null() {
		let result = ResultMessage {
			subtype: None,
			is_error: None,
			result: None,
			duration_ms: None,
			duration_api_ms: None,
			num_turns: None,
			total_cost_usd: None,
			usage: None,
			model_usage: None,
		};
		let resp = build_response("chatcmpl-x", 0, "model", "", &result);
		let json = serde_json::to_value(&resp).unwrap();
		assert!(json["system_fingerprint"].is_null());
	}
}
