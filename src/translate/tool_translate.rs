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

/// Translate an OpenAI tool_choice into the Anthropic shape. `disable_parallel`
/// carries OpenAI `parallel_tool_calls: false` onto the modes where Anthropic
/// accepts `disable_parallel_tool_use` (auto / any / specific tool).
pub fn translate_tool_choice(
	choice: &Option<OaiToolChoice>,
	disable_parallel: Option<bool>,
) -> Option<ToolChoice> {
	// Anthropic only accepts the flag as `true` (false is the default and is
	// rejected by some API versions); send Some(true) or omit.
	let dptu = if disable_parallel == Some(true) { Some(true) } else { None };
	match choice {
		// No explicit choice: Anthropic defaults to auto. Normally we omit the
		// field, but if the caller asked to disable parallel tool use we must
		// emit an explicit `auto` carrying the flag, since there is nowhere
		// else to attach it.
		None => dptu.map(|_| ToolChoice::Auto {
			disable_parallel_tool_use: dptu,
		}),
		Some(OaiToolChoice::Mode(s)) => match s.as_str() {
			"auto" => Some(ToolChoice::Auto {
				disable_parallel_tool_use: dptu,
			}),
			"none" => Some(ToolChoice::None {}),
			"required" | "any" => Some(ToolChoice::Any {
				disable_parallel_tool_use: dptu,
			}),
			_ => None,
		},
		Some(OaiToolChoice::Specific { function, .. }) => Some(ToolChoice::Tool {
			name: function.name.clone(),
			disable_parallel_tool_use: dptu,
		}),
	}
}

#[allow(dead_code)]
fn _unused_value_marker() -> Value {
	Value::Null
}
