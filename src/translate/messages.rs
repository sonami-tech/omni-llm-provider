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
				let mut blocks = vec![tool_or_user_block(m)?];
				let mut peek = idx + 1;
				while peek < messages.len() {
					let next = &messages[peek];
					if matches!(next.role.as_str(), "tool" | "function") {
						blocks.push(tool_or_user_block(next)?);
						peek += 1;
					} else {
						break;
					}
				}
				out.push(Message {
					role: "user".into(),
					content: MessageContent::Blocks(blocks),
				});
				has_user = true;
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
				if let Some(text) = &part.text {
					if !text.is_empty() {
						out.push(ContentBlock::Text {
							text: text.clone(),
							cache_control: None,
						});
					}
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

	// Thinking blocks first (Anthropic invariant when reasoning_content
	// precedes a tool_use turn).
	if let Some(reasoning) = &m.reasoning_content {
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
		Value::String(s) if !s.is_empty() => vec![ContentBlock::Thinking {
			thinking: s.clone(),
			signature: None,
		}],
		Value::Array(arr) => arr
			.iter()
			.filter_map(|v| {
				let obj = v.as_object()?;
				let thinking = obj.get("thinking").and_then(|x| x.as_str())?.to_string();
				let signature = obj
					.get("signature")
					.and_then(|x| x.as_str())
					.map(str::to_string);
				Some(ContentBlock::Thinking { thinking, signature })
			})
			.collect(),
		_ => vec![],
	}
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

fn tool_or_user_block(m: &ChatMessage) -> Result<ContentBlock, AppError> {
	if m.tool_call_id.is_some() {
		tool_result_block(m)
	} else {
		let text = extract_text(&m.content);
		Ok(ContentBlock::Text {
			text,
			cache_control: None,
		})
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
