//! Convert a stream of Anthropic SSE events into OpenAI chat.completion.chunk
//! JSON values, in order. Stateful: tracks block indexes, whether the
//! assistant-role chunk has been emitted, and tool-use → OAI tool_call index
//! mapping.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::upstream::stream::{BlockDelta, BlockStart, StreamEvent};

/// What kind of block a particular Anthropic content_block_index is.
#[derive(Debug, Clone, Copy)]
enum TrackedBlock {
	Text,
	ToolUse { oai_index: u32 },
	Thinking,
	Other,
}

#[derive(Debug, Default)]
struct ThinkingBuffer {
	thinking: String,
	signature: Option<String>,
}

pub struct OaiStreamConverter {
	pub chat_id: String,
	pub created: u64,
	/// Model name to echo back to the consumer. Set from message_start, or
	/// from the originally requested model if message_start has been seen
	/// but specified a dated form the caller wants normalized.
	pub model: String,
	pub requested_model: String,
	role_emitted: bool,
	finish_emitted: bool,
	blocks: HashMap<u32, TrackedBlock>,
	thinking_buffers: HashMap<u32, ThinkingBuffer>,
	next_oai_tool_index: u32,
	stop_reason: Option<String>,
	input_tokens: Option<u32>,
	output_tokens: Option<u32>,
}

impl OaiStreamConverter {
	pub fn new(chat_id: String, created: u64, requested_model: String) -> Self {
		Self {
			chat_id,
			created,
			model: requested_model.clone(),
			requested_model,
			role_emitted: false,
			finish_emitted: false,
			blocks: HashMap::new(),
			thinking_buffers: HashMap::new(),
			next_oai_tool_index: 0,
			stop_reason: None,
			input_tokens: None,
			output_tokens: None,
		}
	}

	/// Translate a single upstream event into zero or more outbound chunks.
	/// The caller serializes each Value to JSON and emits as `data: <json>\n\n`,
	/// then emits `data: [DONE]\n\n` after the stream ends.
	pub fn on_event(&mut self, event: StreamEvent) -> Vec<Value> {
		match event {
			StreamEvent::MessageStart {
				id: _,
				model: _,
				input_tokens,
				output_tokens,
			} => {
				if input_tokens.is_some() {
					self.input_tokens = input_tokens;
				}
				if output_tokens.is_some() {
					self.output_tokens = output_tokens;
				}
				// Don't switch model name to dated form — keep what the
				// consumer requested. This avoids confusing OAI clients that
				// echo the model back.
				if !self.role_emitted {
					self.role_emitted = true;
					return vec![self.role_chunk()];
				}
				vec![]
			}
			StreamEvent::ContentBlockStart { index, block } => match block {
				BlockStart::Text => {
					self.blocks.insert(index, TrackedBlock::Text);
					vec![]
				}
				BlockStart::ToolUse { id, name } => {
					let oai_index = self.next_oai_tool_index;
					self.next_oai_tool_index += 1;
					self.blocks
						.insert(index, TrackedBlock::ToolUse { oai_index });
					let mut chunks = Vec::new();
					if !self.role_emitted {
						self.role_emitted = true;
						chunks.push(self.role_chunk());
					}
					chunks.push(self.tool_call_open_chunk(oai_index, &id, &name));
					chunks
				}
				BlockStart::Thinking => {
					self.blocks.insert(index, TrackedBlock::Thinking);
					self.thinking_buffers
						.insert(index, ThinkingBuffer::default());
					vec![]
				}
				BlockStart::Other(_) => {
					self.blocks.insert(index, TrackedBlock::Other);
					vec![]
				}
			},
			StreamEvent::ContentBlockDelta { index, delta } => {
				let tracked = self.blocks.get(&index).copied().unwrap_or(TrackedBlock::Other);
				match (tracked, delta) {
					(TrackedBlock::Text, BlockDelta::Text(s)) => {
						let mut chunks = Vec::new();
						if !self.role_emitted {
							self.role_emitted = true;
							chunks.push(self.role_chunk());
						}
						chunks.push(self.text_delta_chunk(&s));
						chunks
					}
					(TrackedBlock::ToolUse { oai_index }, BlockDelta::InputJson(s)) => {
						vec![self.tool_call_args_chunk(oai_index, &s)]
					}
					(TrackedBlock::Thinking, BlockDelta::Thinking(s)) => {
						self.thinking_buffers
							.entry(index)
							.or_default()
							.thinking
							.push_str(&s);
						vec![]
					}
					(TrackedBlock::Thinking, BlockDelta::ThinkingSignature(s)) => {
						self.thinking_buffers
							.entry(index)
							.or_default()
							.signature = Some(s);
						vec![]
					}
					_ => vec![],
				}
			}
			StreamEvent::ContentBlockStop { index } => {
				match self.blocks.remove(&index) {
					Some(TrackedBlock::Thinking) => self.flush_thinking_buffer(index),
					_ => vec![],
				}
			}
			StreamEvent::MessageDelta {
				stop_reason,
				output_tokens,
				..
			} => {
				if let Some(r) = stop_reason {
					self.stop_reason = Some(r);
				}
				if output_tokens.is_some() {
					self.output_tokens = output_tokens;
				}
				vec![]
			}
			StreamEvent::MessageStop => self.finish_chunks(),
			StreamEvent::Ping => vec![],
			StreamEvent::Error { kind, message } => {
				vec![json!({
					"id": self.chat_id,
					"object": "chat.completion.chunk",
					"created": self.created,
					"model": self.requested_model,
					"choices": [{
						"index": 0,
						"delta": {},
						"finish_reason": "error",
					}],
					"error": {
						"type": kind,
						"message": message,
					}
				})]
			}
			StreamEvent::Unknown(..) => vec![],
		}
	}

	/// If the upstream stream ended without a message_stop event we still
	/// need to emit the final chunk. Caller invokes this once after the
	/// stream is exhausted.
	pub fn finalize_if_needed(&mut self) -> Vec<Value> {
		if self.finish_emitted {
			vec![]
		} else {
			self.finish_chunks()
		}
	}

	fn role_chunk(&self) -> Value {
		json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {"role": "assistant", "content": ""},
				"finish_reason": null,
			}]
		})
	}

	fn text_delta_chunk(&self, text: &str) -> Value {
		json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {"content": text},
				"finish_reason": null,
			}]
		})
	}

	fn reasoning_content_chunk(&self, thinking: String, signature: String) -> Value {
		let reasoning_part = json!({
			"type": "thinking",
			"thinking": thinking.clone(),
			"signature": signature,
		});

		json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {
					"reasoning_content": thinking,
					"reasoning_content_blocks": [reasoning_part],
				},
				"finish_reason": null,
			}]
		})
	}

	fn flush_thinking_buffer(&mut self, index: u32) -> Vec<Value> {
		let Some(buffer) = self.thinking_buffers.remove(&index) else {
			return vec![];
		};
		match (buffer.thinking.is_empty(), buffer.signature) {
			(false, Some(signature)) if !signature.is_empty() => {
				vec![self.reasoning_content_chunk(buffer.thinking, signature)]
			}
			_ => vec![],
		}
	}

	fn flush_all_thinking_buffers(&mut self) -> Vec<Value> {
		let mut indexes: Vec<u32> = self.thinking_buffers.keys().copied().collect();
		indexes.sort_unstable();
		indexes
			.into_iter()
			.flat_map(|index| self.flush_thinking_buffer(index))
			.collect()
	}

	fn tool_call_open_chunk(&self, oai_index: u32, id: &str, name: &str) -> Value {
		json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {
					"tool_calls": [{
						"index": oai_index,
						"id": id,
						"type": "function",
						"function": {
							"name": name,
							"arguments": "",
						}
					}]
				},
				"finish_reason": null,
			}]
		})
	}

	fn tool_call_args_chunk(&self, oai_index: u32, partial: &str) -> Value {
		json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {
					"tool_calls": [{
						"index": oai_index,
						"function": {
							"arguments": partial,
						}
					}]
				},
				"finish_reason": null,
			}]
		})
	}

	fn finish_chunks(&mut self) -> Vec<Value> {
		if self.finish_emitted {
			return vec![];
		}
		self.finish_emitted = true;
		let mut chunks = self.flush_all_thinking_buffers();
		let has_tool_calls = self.next_oai_tool_index > 0;
		let finish_reason = match self.stop_reason.as_deref() {
			Some("end_turn") => "stop",
			Some("max_tokens") => "length",
			Some("stop_sequence") => "stop",
			Some("tool_use") => "tool_calls",
			Some("pause_turn") => "stop",
			Some("refusal") => "content_filter",
			_ if has_tool_calls => "tool_calls",
			_ => "stop",
		};
		let mut chunk = json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {},
				"finish_reason": finish_reason,
			}]
		});
		if let Some(usage) = self.usage_value() {
			chunk["usage"] = usage;
		}
		chunks.push(chunk);
		chunks
	}

	fn usage_value(&self) -> Option<Value> {
		let prompt_tokens = self.input_tokens?;
		let completion_tokens = self.output_tokens.unwrap_or(0);
		Some(json!({
			"prompt_tokens": prompt_tokens,
			"completion_tokens": completion_tokens,
			"total_tokens": prompt_tokens + completion_tokens,
		}))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn converter() -> OaiStreamConverter {
		OaiStreamConverter::new("chatcmpl-test".into(), 123, "claude-test".into())
	}

	#[test]
	fn buffers_thinking_until_content_block_stop() {
		let mut converter = converter();

		assert!(converter
			.on_event(StreamEvent::ContentBlockStart {
				index: 3,
				block: BlockStart::Thinking,
			})
			.is_empty());
		assert!(converter
			.on_event(StreamEvent::ContentBlockDelta {
				index: 3,
				delta: BlockDelta::Thinking("first ".into()),
			})
			.is_empty());
		assert!(converter
			.on_event(StreamEvent::ContentBlockDelta {
				index: 3,
				delta: BlockDelta::Thinking("second".into()),
			})
			.is_empty());
		assert!(converter
			.on_event(StreamEvent::ContentBlockDelta {
				index: 3,
				delta: BlockDelta::ThinkingSignature("sig-123".into()),
			})
			.is_empty());

		let chunks = converter.on_event(StreamEvent::ContentBlockStop { index: 3 });
		assert_eq!(chunks.len(), 1);
		assert_eq!(
			chunks[0]["choices"][0]["delta"]["reasoning_content"],
			"first second"
		);
		assert_eq!(
			chunks[0]["choices"][0]["delta"]["reasoning_content_blocks"],
			json!([{
				"type": "thinking",
				"thinking": "first second",
				"signature": "sig-123",
			}])
		);
		assert!(chunks[0]["choices"][0]["delta"]
			.get("reasoning_signature")
			.is_none());
	}

	#[test]
	fn finalization_drops_unclosed_thinking_without_signature() {
		let mut converter = converter();

		converter.on_event(StreamEvent::ContentBlockStart {
			index: 1,
			block: BlockStart::Thinking,
		});
		converter.on_event(StreamEvent::ContentBlockDelta {
			index: 1,
			delta: BlockDelta::Thinking("partial".into()),
		});

		let chunks = converter.finalize_if_needed();
		assert_eq!(chunks.len(), 1);
		assert!(chunks[0]["choices"][0]["delta"]
			.get("reasoning_content")
			.is_none());
		assert_eq!(chunks[0]["choices"][0]["finish_reason"], "stop");
	}
}
