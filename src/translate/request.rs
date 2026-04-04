use serde::Deserialize;

use crate::error::AppError;
use crate::models::ModelDef;

// ── OpenAI request types ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
	pub model: String,
	pub messages: Vec<ChatMessage>,
	#[serde(default)]
	pub stream: bool,
	pub reasoning_effort: Option<String>,
	// All other OpenAI fields (max_tokens, temperature, top_p, stop, etc.)
	// are accepted and silently ignored — no #[serde(deny_unknown_fields)].
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
	pub role: String,
	#[serde(default)]
	pub content: Option<MessageContent>,
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

// ── Prompt construction ───────────────────────────────────────────

/// Separate messages into a conversation prompt and optional system prompt.
/// System messages are concatenated for --append-system-prompt.
/// Non-system messages are formatted for the positional prompt argument.
pub fn build_prompt_and_system(
	messages: &[ChatMessage],
) -> Result<(String, Option<String>), AppError> {
	let mut system_parts: Vec<String> = Vec::new();
	let mut prompt_parts: Vec<String> = Vec::new();
	let mut has_user_message = false;

	for msg in messages {
		let text = extract_text(&msg.content);
		match msg.role.as_str() {
			"system" | "developer" => {
				if !text.is_empty() {
					system_parts.push(text);
				}
			}
			"assistant" => {
				prompt_parts.push(format!(
					"<previous_response>\n{}\n</previous_response>",
					text
				));
			}
			_ => {
				// user, tool, function, and other roles treated as user messages.
				if !text.is_empty() {
					has_user_message = true;
				}
				prompt_parts.push(text);
			}
		}
	}

	if !has_user_message {
		return Err(AppError::BadRequest(
			"No user messages found after filtering".into(),
		));
	}

	let prompt = prompt_parts.join("\n\n");
	let system_prompt = if system_parts.is_empty() {
		None
	} else {
		Some(system_parts.join("\n"))
	};

	Ok((prompt, system_prompt))
}

/// Build CLI arguments for the claude subprocess.
pub fn build_cli_args(
	model_def: &ModelDef,
	prompt: &str,
	system_prompt: Option<&str>,
	effort: Option<&str>,
) -> Vec<String> {
	let mut args = vec![
		"-p".to_string(),
		"--verbose".to_string(),
		"--output-format".to_string(),
		"stream-json".to_string(),
		"--include-partial-messages".to_string(),
		"--tools".to_string(),
		String::new(), // Empty string — disables all built-in tools.
		"--model".to_string(),
		model_def.cli_name.to_string(),
		"--no-session-persistence".to_string(),
	];

	if let Some(sp) = system_prompt {
		args.push("--append-system-prompt".to_string());
		args.push(sp.to_string());
	}

	if let Some(e) = effort {
		args.push("--effort".to_string());
		args.push(e.to_string());
	}

	// Prompt as the final positional argument.
	args.push(prompt.to_string());

	args
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
			},
			ContentPart {
				part_type: Some("text".into()),
				text: Some("world".into()),
			},
			ContentPart {
				part_type: Some("image_url".into()),
				text: None,
			},
		]));
		assert_eq!(extract_text(&content), "Hello world");
	}

	// ── build_prompt_and_system ───────────────────────────────

	#[test]
	fn single_user_message() {
		let messages = vec![msg("user", "Hello")];
		let (prompt, system) = build_prompt_and_system(&messages).unwrap();
		assert_eq!(prompt, "Hello");
		assert!(system.is_none());
	}

	#[test]
	fn system_message_extracted() {
		let messages = vec![msg("system", "Be helpful"), msg("user", "Hi")];
		let (prompt, system) = build_prompt_and_system(&messages).unwrap();
		assert_eq!(prompt, "Hi");
		assert_eq!(system.unwrap(), "Be helpful");
	}

	#[test]
	fn assistant_message_wrapped() {
		let messages = vec![
			msg("user", "Hi"),
			msg("assistant", "Hello!"),
			msg("user", "How are you?"),
		];
		let (prompt, _) = build_prompt_and_system(&messages).unwrap();
		assert!(prompt.contains("<previous_response>\nHello!\n</previous_response>"));
		assert!(prompt.contains("How are you?"));
	}

	#[test]
	fn multi_turn_conversation() {
		let messages = vec![
			msg("system", "You are a helper."),
			msg("user", "Question 1"),
			msg("assistant", "Answer 1"),
			msg("user", "Question 2"),
		];
		let (prompt, system) = build_prompt_and_system(&messages).unwrap();
		assert_eq!(system.unwrap(), "You are a helper.");
		assert!(prompt.contains("Question 1"));
		assert!(prompt.contains("<previous_response>\nAnswer 1\n</previous_response>"));
		assert!(prompt.contains("Question 2"));
	}

	#[test]
	fn unknown_role_treated_as_user() {
		let messages = vec![msg("tool", "tool output"), msg("user", "continue")];
		let (prompt, _) = build_prompt_and_system(&messages).unwrap();
		assert!(prompt.contains("tool output"));
	}

	#[test]
	fn no_user_messages_returns_error() {
		let messages = vec![msg("system", "Be helpful")];
		assert!(build_prompt_and_system(&messages).is_err());
	}

	#[test]
	fn empty_messages_returns_error() {
		let messages: Vec<ChatMessage> = vec![];
		// This would be caught by validate_request, but build_prompt_and_system
		// returns error for no user messages too.
		assert!(build_prompt_and_system(&messages).is_err());
	}

	// ── build_cli_args ────────────────────────────────────────

	#[test]
	fn cli_args_basic() {
		let model_def = &crate::models::MODELS[1]; // sonnet
		let args = build_cli_args(model_def, "Hello", None, None);
		assert!(args.contains(&"-p".to_string()));
		assert!(args.contains(&"--verbose".to_string()));
		assert!(args.contains(&"--output-format".to_string()));
		assert!(args.contains(&"stream-json".to_string()));
		assert!(args.contains(&"--include-partial-messages".to_string()));
		assert!(args.contains(&"--tools".to_string()));
		assert!(args.contains(&String::new())); // empty string for --tools
		assert!(args.contains(&"--model".to_string()));
		assert!(args.contains(&"sonnet".to_string()));
		assert!(args.contains(&"--no-session-persistence".to_string()));
		assert!(!args.contains(&"--append-system-prompt".to_string()));
		assert!(!args.contains(&"--effort".to_string()));
		assert_eq!(args.last().unwrap(), "Hello");
	}

	#[test]
	fn cli_args_with_system_and_effort() {
		let model_def = &crate::models::MODELS[0]; // opus
		let args = build_cli_args(model_def, "Hello", Some("Be concise"), Some("high"));
		assert!(args.contains(&"--append-system-prompt".to_string()));
		assert!(args.contains(&"Be concise".to_string()));
		assert!(args.contains(&"--effort".to_string()));
		assert!(args.contains(&"high".to_string()));
		assert!(args.contains(&"opus".to_string()));
	}

	#[test]
	fn tools_empty_string_is_separate_arg() {
		let model_def = &crate::models::MODELS[1];
		let args = build_cli_args(model_def, "test", None, None);
		let tools_idx = args.iter().position(|a| a == "--tools").unwrap();
		assert_eq!(args[tools_idx + 1], "");
	}

	// ── validate_request ──────────────────────────────────────

	#[test]
	fn validate_missing_model() {
		let req = ChatCompletionRequest {
			model: String::new(),
			messages: vec![msg("user", "hi")],
			stream: false,
			reasoning_effort: None,
		};
		assert!(validate_request(&req).is_err());
	}

	#[test]
	fn validate_empty_messages() {
		let req = ChatCompletionRequest {
			model: "sonnet".into(),
			messages: vec![],
			stream: false,
			reasoning_effort: None,
		};
		assert!(validate_request(&req).is_err());
	}

	#[test]
	fn validate_valid_request() {
		let req = ChatCompletionRequest {
			model: "sonnet".into(),
			messages: vec![msg("user", "hi")],
			stream: false,
			reasoning_effort: None,
		};
		assert!(validate_request(&req).is_ok());
	}
}
