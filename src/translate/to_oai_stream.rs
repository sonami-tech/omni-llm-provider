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
	next_oai_tool_index: u32,
	stop_reason: Option<String>,
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
			next_oai_tool_index: 0,
			stop_reason: None,
		}
	}

	/// Translate a single upstream event into zero or more outbound chunks.
	/// The caller serializes each Value to JSON and emits as `data: <json>\n\n`,
	/// then emits `data: [DONE]\n\n` after the stream ends.
	pub fn on_event(&mut self, event: StreamEvent) -> Vec<Value> {
		match event {
			StreamEvent::MessageStart { id: _, model: _ } => {
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
						vec![self.reasoning_delta_chunk(&s)]
					}
					(TrackedBlock::Thinking, BlockDelta::ThinkingSignature(s)) => {
						vec![self.reasoning_signature_chunk(&s)]
					}
					_ => vec![],
				}
			}
			StreamEvent::ContentBlockStop { .. } => vec![],
			StreamEvent::MessageDelta {
				stop_reason,
				output_tokens: _,
				..
			} => {
				if let Some(r) = stop_reason {
					self.stop_reason = Some(r);
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

	fn reasoning_delta_chunk(&self, text: &str) -> Value {
		json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {"reasoning_content": text},
				"finish_reason": null,
			}]
		})
	}

	fn reasoning_signature_chunk(&self, sig: &str) -> Value {
		// Surface signature as an extension field — OpenAI clients ignore
		// unknown delta fields.
		json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {"reasoning_signature": sig},
				"finish_reason": null,
			}]
		})
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
		vec![json!({
			"id": self.chat_id,
			"object": "chat.completion.chunk",
			"created": self.created,
			"model": self.requested_model,
			"choices": [{
				"index": 0,
				"delta": {},
				"finish_reason": finish_reason,
			}]
		})]
	}
}
