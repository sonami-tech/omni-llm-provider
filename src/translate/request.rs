use serde::Deserialize;

use crate::error::AppError;

// ── OpenAI request types ──────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
pub struct ChatCompletionRequest {
	pub model: String,
	pub messages: Vec<ChatMessage>,
	#[serde(default)]
	pub stream: bool,
	pub reasoning_effort: Option<String>,
	#[serde(default)]
	pub tools: Option<Vec<ToolDefinition>>,
	#[serde(default)]
	pub tool_choice: Option<ToolChoice>,
	/// OpenAI `parallel_tool_calls`. When false, CCP sets Anthropic
	/// `disable_parallel_tool_use: true` on the tool_choice.
	#[serde(default)]
	pub parallel_tool_calls: Option<bool>,
	/// OpenAI's optional end-user identifier; used as a session-id fallback
	/// when the `x-session-id` header is absent. See `routes::completions`.
	#[serde(default)]
	pub user: Option<String>,

	// ── Sampling/output controls (OAI-compat, v2 forwards to Anthropic) ──
	#[serde(default)]
	pub max_tokens: Option<u32>,
	#[serde(default)]
	pub max_completion_tokens: Option<u32>,
	#[serde(default)]
	pub temperature: Option<f32>,
	#[serde(default)]
	pub top_p: Option<f32>,
	#[serde(default)]
	pub top_k: Option<u32>,
	#[serde(default)]
	pub stop: Option<StopSpec>,
	#[serde(default)]
	pub metadata: Option<serde_json::Value>,
	#[serde(default)]
	pub n: Option<u32>,
	#[serde(default)]
	pub response_format: Option<serde_json::Value>,
	#[serde(default)]
	pub seed: Option<i64>,
	#[serde(default)]
	pub presence_penalty: Option<f32>,
	#[serde(default)]
	pub frequency_penalty: Option<f32>,
	/// Anthropic-style thinking pass-through. If consumer sets this, it
	/// takes precedence over `reasoning_effort`.
	#[serde(default)]
	pub thinking: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
	One(String),
	Many(Vec<String>),
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatMessage {
	pub role: String,
	#[serde(default)]
	pub content: Option<MessageContent>,
	#[serde(default)]
	pub tool_calls: Option<Vec<RequestToolCall>>,
	#[serde(default)]
	pub tool_call_id: Option<String>,
	/// OAI-compatible reasoning text on assistant turns. This is display-only
	/// for Anthropic replay because strings do not carry thinking signatures.
	#[serde(default)]
	#[allow(dead_code)]
	pub reasoning_content: Option<serde_json::Value>,
	/// CCP extension: signed Anthropic thinking-block carryover.
	/// Only blocks with a non-empty `signature` can be safely replayed.
	#[serde(default)]
	pub reasoning_content_blocks: Option<serde_json::Value>,
}

// ── Tool types ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ToolDefinition {
	#[serde(rename = "type")]
	#[allow(dead_code)]
	pub tool_type: Option<String>,
	pub function: FunctionDefinition,
}

#[derive(Debug, Deserialize)]
pub struct FunctionDefinition {
	pub name: String,
	pub description: Option<String>,
	pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
	Mode(String),
	Specific {
		#[serde(rename = "type")]
		#[allow(dead_code)]
		choice_type: String,
		function: ToolChoiceFunction,
	},
}

#[derive(Debug, Deserialize)]
pub struct ToolChoiceFunction {
	pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct RequestToolCall {
	#[allow(dead_code)]
	pub id: String,
	#[serde(rename = "type")]
	#[allow(dead_code)]
	pub call_type: Option<String>,
	pub function: RequestFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct RequestFunctionCall {
	pub name: String,
	pub arguments: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
	Text(String),
	Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
	#[serde(rename = "type")]
	pub part_type: Option<String>,
	pub text: Option<String>,
	/// OpenAI vision input: `{"type":"image_url","image_url":{"url":"..."}}`.
	pub image_url: Option<ImageUrl>,
}

/// OpenAI `image_url` part payload. `url` is either a public http(s) URL or a
/// base64 data URL (`data:image/png;base64,...`). `detail` is accepted but
/// ignored (Anthropic has no equivalent knob).
#[derive(Debug, Deserialize)]
pub struct ImageUrl {
	pub url: String,
	#[serde(default)]
	#[allow(dead_code)]
	pub detail: Option<String>,
}

// ── Text extraction ───────────────────────────────────────────────

/// Extract text from a message's content field.
pub fn extract_text(content: &Option<MessageContent>) -> String {
	match content {
		None => String::new(),
		Some(MessageContent::Text(s)) => s.clone(),
		Some(MessageContent::Parts(parts)) => {
			let mut texts = Vec::new();
			for part in parts {
				match part.part_type.as_deref() {
					Some("text") => {
						if let Some(text) = &part.text {
							texts.push(text.as_str());
						}
					}
					Some(other) => {
						tracing::warn!(content_type = %other, "Non-text content block ignored");
					}
					None => {}
				}
			}
			texts.join("")
		}
	}
}

/// Validate the incoming request.
pub fn validate_request(req: &ChatCompletionRequest) -> Result<(), AppError> {
	if req.model.is_empty() {
		return Err(AppError::BadRequest("model field is required".into()));
	}
	if req.messages.is_empty() {
		return Err(AppError::BadRequest(
			"messages is required and must be a non-empty array".into(),
		));
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	fn msg(role: &str, content: &str) -> ChatMessage {
		ChatMessage {
			role: role.into(),
			content: Some(MessageContent::Text(content.into())),
			..Default::default()
		}
	}

	// ── extract_text ──────────────────────────────────────────

	#[test]
	fn extract_text_none_returns_empty() {
		assert_eq!(extract_text(&None), "");
	}

	#[test]
	fn extract_text_plain_string() {
		let content = Some(MessageContent::Text("hello".into()));
		assert_eq!(extract_text(&content), "hello");
	}

	#[test]
	fn extract_text_multipart() {
		let content = Some(MessageContent::Parts(vec![
			ContentPart {
				part_type: Some("text".into()),
				text: Some("Hello ".into()),
				image_url: None,
			},
			ContentPart {
				part_type: Some("text".into()),
				text: Some("world".into()),
				image_url: None,
			},
			ContentPart {
				part_type: Some("image_url".into()),
				text: None,
				image_url: Some(ImageUrl {
					url: "data:image/png;base64,AAAA".into(),
					detail: None,
				}),
			},
		]));
		assert_eq!(extract_text(&content), "Hello world");
	}

	// ── validate_request ──────────────────────────────────────

	#[test]
	fn validate_missing_model() {
		let req = ChatCompletionRequest {
			model: String::new(),
			messages: vec![msg("user", "hi")],
			..Default::default()
		};
		assert!(validate_request(&req).is_err());
	}

	#[test]
	fn validate_empty_messages() {
		let req = ChatCompletionRequest {
			model: "sonnet".into(),
			..Default::default()
		};
		assert!(validate_request(&req).is_err());
	}

	#[test]
	fn validate_valid_request() {
		let req = ChatCompletionRequest {
			model: "sonnet".into(),
			messages: vec![msg("user", "hi")],
			..Default::default()
		};
		assert!(validate_request(&req).is_ok());
	}
}
