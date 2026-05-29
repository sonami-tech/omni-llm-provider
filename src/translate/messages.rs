//! Reshape OpenAI-completions message arrays into Anthropic Messages format.
//!
//! Rules:
//! - Leading `system`/`developer` messages are extracted into the top-level
//!   `system` field as a block array.
//! - `user` messages map directly. Multi-part content is translated block by
//!   block (text → text, image_url → image).
//! - `assistant` messages with `tool_calls` produce both text + `tool_use`
//!   content blocks. Order: thinking blocks (if any) first, then text, then
//!   tool_use blocks. (Per Anthropic's invariant that thinking precedes
//!   text/tool_use in an assistant turn that follows a tool_result.)
//! - `tool` role messages become `user` messages with a `tool_result`
//!   content block, keyed by `tool_call_id` → `tool_use_id`.

use serde_json::Value;

use crate::error::AppError;
use crate::translate::anthropic::{
	ContentBlock, ImageSource, Message, MessageContent, SystemBlock, SystemField,
	ToolResultContent,
};
use crate::translate::request::{
	ChatMessage, ContentPart, MessageContent as OaiMessageContent, RequestToolCall,
};

/// Output of message reshaping.
pub struct Reshaped {
	pub system: Option<SystemField>,
	pub messages: Vec<Message>,
}

pub fn reshape(messages: &[ChatMessage]) -> Result<Reshaped, AppError> {
	if messages.is_empty() {
		return Err(AppError::BadRequest("messages must be non-empty".into()));
	}

	// Extract leading system/developer messages.
	let mut system_blocks: Vec<SystemBlock> = Vec::new();
	let mut idx = 0;
	while idx < messages.len() {
		let m = &messages[idx];
		match m.role.as_str() {
			"system" | "developer" => {
				let text = extract_text(&m.content);
				if !text.is_empty() {
					system_blocks.push(SystemBlock {
						kind: "text".into(),
						text,
						cache_control: None,
					});
				}
				idx += 1;
			}
			_ => break,
		}
	}

	let mut out = Vec::with_capacity(messages.len() - idx);
	let mut has_user = false;

	while idx < messages.len() {
		let m = &messages[idx];
		match m.role.as_str() {
			"system" | "developer" => {
				// Mid-conversation system message: fold into a user message
				// with a clear marker. Anthropic doesn't have mid-stream
				// system messages.
				let text = extract_text(&m.content);
				if !text.is_empty() {
					out.push(Message {
						role: "user".into(),
						content: MessageContent::Blocks(vec![ContentBlock::Text {
							text: format!("[system message]\n{}", text),
							cache_control: None,
						}]),
					});
					has_user = true;
				}
			}
			"user" => {
				let blocks = translate_user_content(&m.content)?;
				if !blocks.is_empty() {
					out.push(Message {
						role: "user".into(),
						content: MessageContent::Blocks(blocks),
					});
					has_user = true;
				}
			}
			"assistant" => {
				let blocks = translate_assistant(m)?;
				if !blocks.is_empty() {
					out.push(Message {
						role: "assistant".into(),
						content: MessageContent::Blocks(blocks),
					});
				}
			}
			"tool" | "function" => {
				// Anthropic spec: tool_result content blocks live inside a
				// `user` message. Coalesce consecutive tool messages.
				let mut blocks: Vec<ContentBlock> = Vec::new();
				blocks.extend(tool_or_user_block(m)?);
				let mut peek = idx + 1;
				while peek < messages.len() {
					let next = &messages[peek];
					if matches!(next.role.as_str(), "tool" | "function") {
						blocks.extend(tool_or_user_block(next)?);
						peek += 1;
					} else {
						break;
					}
				}
				// Only emit a user turn if the coalesced block list is
				// non-empty; an all-empty tool batch would otherwise produce
				// a user message with zero content blocks (Anthropic rejects).
				if !blocks.is_empty() {
					out.push(Message {
						role: "user".into(),
						content: MessageContent::Blocks(blocks),
					});
					has_user = true;
				}
				idx = peek;
				continue;
			}
			other => {
				return Err(AppError::BadRequest(format!(
					"unsupported message role: {}",
					other
				)));
			}
		}
		idx += 1;
	}

	if !has_user {
		return Err(AppError::BadRequest(
			"messages must include at least one user/tool message".into(),
		));
	}

	let system = if system_blocks.is_empty() {
		None
	} else {
		Some(SystemField::Blocks(system_blocks))
	};

	Ok(Reshaped {
		system,
		messages: out,
	})
}

fn translate_user_content(content: &Option<OaiMessageContent>) -> Result<Vec<ContentBlock>, AppError> {
	match content {
		None => Ok(vec![]),
		Some(OaiMessageContent::Text(s)) => {
			if s.is_empty() {
				Ok(vec![])
			} else {
				Ok(vec![ContentBlock::Text {
					text: s.clone(),
					cache_control: None,
				}])
			}
		}
		Some(OaiMessageContent::Parts(parts)) => translate_parts(parts),
	}
}

fn translate_parts(parts: &[ContentPart]) -> Result<Vec<ContentBlock>, AppError> {
	let mut out = Vec::with_capacity(parts.len());
	for part in parts {
		match part.part_type.as_deref() {
			Some("text") | None => {
				if let Some(text) = &part.text
					&& !text.is_empty() {
						out.push(ContentBlock::Text {
							text: text.clone(),
							cache_control: None,
						});
					}
			}
			Some("image_url") => {
				// OAI image_url shape: {type:"image_url", image_url:{url:"data:image/png;base64,..."}}
				// The current OAI deserialization (ContentPart) only captures `text`,
				// so image_url passthrough requires a richer ContentPart. Defer.
				tracing::warn!("image_url content not yet supported in v2 — skipping");
			}
			Some(other) => {
				tracing::warn!(content_type = %other, "unknown content part type — skipping");
			}
		}
	}
	Ok(out)
}

fn translate_assistant(m: &ChatMessage) -> Result<Vec<ContentBlock>, AppError> {
	let mut blocks: Vec<ContentBlock> = Vec::new();

	// Thinking blocks first (Anthropic invariant when signed thinking precedes
	// a tool_use turn). Plain reasoning_content is display text in OAI clients;
	// replaying it as Anthropic thinking would be unsigned and rejected.
	if let Some(reasoning) = &m.reasoning_content_blocks {
		blocks.extend(reasoning_to_blocks(reasoning));
	}

	// Text content.
	let text = extract_text(&m.content);
	if !text.is_empty() {
		blocks.push(ContentBlock::Text {
			text,
			cache_control: None,
		});
	}

	// Tool calls become tool_use blocks.
	if let Some(calls) = &m.tool_calls {
		for c in calls {
			blocks.push(tool_use_block(c)?);
		}
	}

	Ok(blocks)
}

fn reasoning_to_blocks(value: &Value) -> Vec<ContentBlock> {
	match value {
		Value::Array(arr) => arr.iter().filter_map(signed_reasoning_block).collect(),
		_ => vec![],
	}
}

fn signed_reasoning_block(value: &Value) -> Option<ContentBlock> {
	let obj = value.as_object()?;
	if obj.get("type").and_then(|x| x.as_str()) != Some("thinking") {
		return None;
	}
	let thinking = obj.get("thinking").and_then(|x| x.as_str())?;
	if thinking.is_empty() {
		return None;
	}
	let signature = obj
		.get("signature")
		.and_then(|x| x.as_str())
		.filter(|s| !s.is_empty())?;
	Some(ContentBlock::Thinking {
		thinking: thinking.to_string(),
		signature: Some(signature.to_string()),
	})
}

fn tool_use_block(call: &RequestToolCall) -> Result<ContentBlock, AppError> {
	let input = if call.function.arguments.trim().is_empty() {
		Value::Object(Default::default())
	} else {
		serde_json::from_str(&call.function.arguments).map_err(|e| {
			AppError::BadRequest(format!(
				"tool_call.function.arguments not valid JSON: {} (input: {:?})",
				e, call.function.arguments
			))
		})?
	};
	Ok(ContentBlock::ToolUse {
		id: call.id.clone(),
		name: call.function.name.clone(),
		input,
	})
}

fn tool_result_block(m: &ChatMessage) -> Result<ContentBlock, AppError> {
	let tool_use_id = m
		.tool_call_id
		.clone()
		.ok_or_else(|| AppError::BadRequest("tool message missing tool_call_id".into()))?;
	let text = extract_text(&m.content);
	Ok(ContentBlock::ToolResult {
		tool_use_id,
		content: if text.is_empty() {
			None
		} else {
			Some(ToolResultContent::Text(text))
		},
		is_error: None,
	})
}

/// Translate a single `tool`/`function` message into zero or one content
/// blocks. A message with a `tool_call_id` becomes a `tool_result`; one
/// without is treated as plain user text. Empty text yields no block so the
/// caller never emits an empty (Anthropic-rejected) text block.
fn tool_or_user_block(m: &ChatMessage) -> Result<Vec<ContentBlock>, AppError> {
	if m.tool_call_id.is_some() {
		Ok(vec![tool_result_block(m)?])
	} else {
		let text = extract_text(&m.content);
		if text.is_empty() {
			Ok(vec![])
		} else {
			Ok(vec![ContentBlock::Text {
				text,
				cache_control: None,
			}])
		}
	}
}

fn extract_text(content: &Option<OaiMessageContent>) -> String {
	match content {
		None => String::new(),
		Some(OaiMessageContent::Text(s)) => s.clone(),
		Some(OaiMessageContent::Parts(parts)) => parts
			.iter()
			.filter(|p| matches!(p.part_type.as_deref(), Some("text") | None))
			.filter_map(|p| p.text.clone())
			.collect::<Vec<_>>()
			.join(""),
	}
}

#[allow(dead_code)]
fn _unused_image_marker() -> ImageSource {
	// Placeholder to keep ImageSource import live until image support is added.
	ImageSource::Url { url: String::new() }
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	fn assistant_with_reasoning(
		reasoning_content: Option<Value>,
		reasoning_content_blocks: Option<Value>,
	) -> ChatMessage {
		ChatMessage {
			role: "assistant".into(),
			reasoning_content,
			reasoning_content_blocks,
			..Default::default()
		}
	}

	fn user_msg(text: &str) -> ChatMessage {
		ChatMessage {
			role: "user".into(),
			content: Some(crate::translate::request::MessageContent::Text(text.into())),
			..Default::default()
		}
	}

	#[test]
	fn empty_tool_message_does_not_emit_empty_text_block() {
		// A tool/function message with no tool_call_id and empty content must
		// not produce a {"type":"text","text":""} block (Anthropic rejects
		// empty text blocks).
		let msgs = vec![
			user_msg("hi"),
			ChatMessage { role: "tool".into(), content: Some(crate::translate::request::MessageContent::Text(String::new())), ..Default::default() },
		];
		let reshaped = reshape(&msgs).unwrap();
		// The empty tool batch yields no user turn; only the original user msg.
		assert_eq!(reshaped.messages.len(), 1);
		for m in &reshaped.messages {
			if let MessageContent::Blocks(blocks) = &m.content {
				for b in blocks {
					if let ContentBlock::Text { text, .. } = b {
						assert!(!text.is_empty(), "no empty text block may be emitted");
					}
				}
			}
		}
	}

	#[test]
	fn tool_result_with_id_is_still_emitted_when_empty() {
		// A real tool_result (has tool_call_id) with empty content is valid:
		// it becomes a tool_result block with content:None, NOT dropped.
		let msgs = vec![
			user_msg("hi"),
			ChatMessage { role: "tool".into(), tool_call_id: Some("t1".into()), content: Some(crate::translate::request::MessageContent::Text(String::new())), ..Default::default() },
		];
		let reshaped = reshape(&msgs).unwrap();
		assert_eq!(reshaped.messages.len(), 2);
		match &reshaped.messages[1].content {
			MessageContent::Blocks(blocks) => {
				assert_eq!(blocks.len(), 1);
				assert!(matches!(blocks[0], ContentBlock::ToolResult { content: None, .. }));
			}
			_ => panic!("expected blocks"),
		}
	}

	#[test]
	fn reasoning_string_is_not_forwarded_as_unsigned_thinking() {
		let msg = assistant_with_reasoning(Some(json!("[object Object]")), None);

		let blocks = translate_assistant(&msg).expect("assistant translation");
		assert!(blocks.is_empty());
	}

	#[test]
	fn unsigned_reasoning_blocks_are_dropped() {
		let msg = assistant_with_reasoning(
			Some(json!([{
				"type": "thinking",
				"thinking": "no usable signature",
			}])),
			None,
		);

		let blocks = translate_assistant(&msg).expect("assistant translation");
		assert!(blocks.is_empty());
	}

	#[test]
	fn legacy_reasoning_content_arrays_are_display_only() {
		let msg = assistant_with_reasoning(
			Some(json!([{
				"type": "thinking",
				"thinking": "safe replay",
				"signature": "sig-123",
			}])),
			None,
		);

		let blocks = translate_assistant(&msg).expect("assistant translation");
		assert!(blocks.is_empty());
	}

	#[test]
	fn reasoning_content_blocks_extension_is_forwarded() {
		let msg = assistant_with_reasoning(
			Some(json!("display only")),
			Some(json!([{
				"type": "thinking",
				"thinking": "extension replay",
				"signature": "sig-456",
			}])),
		);

		let blocks = translate_assistant(&msg).expect("assistant translation");
		assert_eq!(blocks.len(), 1);
		match &blocks[0] {
			ContentBlock::Thinking {
				thinking,
				signature,
			} => {
				assert_eq!(thinking, "extension replay");
				assert_eq!(signature.as_deref(), Some("sig-456"));
			}
			other => panic!("unexpected block: {other:?}"),
		}
	}
}
