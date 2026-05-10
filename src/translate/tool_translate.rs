//! Translate OpenAI tool definitions and tool_choice into Anthropic
//! Messages-API native shapes.
//!
//! Replaces the v1 approach of inlining tools as a `<tool_definitions>` text
//! block in the user message. v2 uses Anthropic's first-class `tools[]` and
//! `tool_choice` fields — the bug fix that motivates the entire rewrite.

use serde_json::{Value, json};

use crate::error::AppError;
use crate::translate::anthropic::{Tool, ToolChoice};
use crate::translate::request::{ToolChoice as OaiToolChoice, ToolDefinition};

pub fn translate_tools(tools: &[ToolDefinition]) -> Result<Vec<Tool>, AppError> {
	let mut out = Vec::with_capacity(tools.len());
	for t in tools {
		let func = &t.function;
		// Anthropic requires an `input_schema`. If the OAI side omitted
		// parameters, default to an open object schema.
		let input_schema = func
			.parameters
			.clone()
			.unwrap_or_else(|| json!({"type": "object", "properties": {}}));
		out.push(Tool {
			name: func.name.clone(),
			description: func.description.clone(),
			input_schema,
			cache_control: None,
		});
	}
	Ok(out)
}

pub fn translate_tool_choice(choice: &Option<OaiToolChoice>) -> Option<ToolChoice> {
	match choice {
		None => None, // omit field; Anthropic defaults to auto
		Some(OaiToolChoice::Mode(s)) => match s.as_str() {
			"auto" => Some(ToolChoice::Auto {
				disable_parallel_tool_use: None,
			}),
			"none" => Some(ToolChoice::None {}),
			"required" | "any" => Some(ToolChoice::Any {
				disable_parallel_tool_use: None,
			}),
			_ => None,
		},
		Some(OaiToolChoice::Specific { function, .. }) => Some(ToolChoice::Tool {
			name: function.name.clone(),
			disable_parallel_tool_use: None,
		}),
	}
}

#[allow(dead_code)]
fn _unused_value_marker() -> Value {
	Value::Null
}
