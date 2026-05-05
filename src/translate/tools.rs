use serde::Deserialize;

use crate::translate::request::{FunctionDefinition, ToolChoice, ToolDefinition};
use crate::translate::response::{ResponseFunctionCall, ResponseToolCall};

// ── Tool choice resolution ───────────────────────────────────────

pub enum ResolvedToolChoice {
	Auto,
	None,
	Required,
	Specific(String),
}

pub fn resolve_tool_choice(choice: &Option<ToolChoice>) -> ResolvedToolChoice {
	match choice {
		None => ResolvedToolChoice::Auto,
		Some(ToolChoice::Mode(s)) => match s.as_str() {
			"none" => ResolvedToolChoice::None,
			"required" => ResolvedToolChoice::Required,
			_ => ResolvedToolChoice::Auto,
		},
		Some(ToolChoice::Specific { function, .. }) => {
			ResolvedToolChoice::Specific(function.name.clone())
		}
	}
}

// ── Prompt building ──────────────────────────────────────────────

/// Build the tool dispatch prefix to prepend to the user message.
pub fn build_tool_prompt_prefix(
	tools: &[ToolDefinition],
	tool_choice: &ResolvedToolChoice,
) -> String {
	let mut prefix = String::from(
		"<tool_definitions>\n\
		 You have access to the following tools. To call a tool, respond with ONLY a JSON array \
		 of tool call objects. Each object must have \"name\" (string) and \"arguments\" (object) \
		 fields.\n\n\
		 CRITICAL: You must choose ONE response type per message:\n\
		 - EITHER a JSON tool call array with NO surrounding text\n\
		 - OR normal text with NO embedded JSON tool calls\n\
		 NEVER mix text and tool calls in the same response.\n\n",
	);

	for tool in tools {
		format_tool_definition(&mut prefix, &tool.function);
	}

	prefix.push_str("</tool_definitions>\n\n");

	prefix.push_str(
		"<tool_response_format>\n\
		 When calling tools, output ONLY a JSON array (no markdown, no explanation, no code fences):\n\
		 [{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}]\n\n\
		 You may call multiple tools at once by including multiple objects in the array.\n\
		 If no tool is needed, respond with normal text.\n\
		 </tool_response_format>\n",
	);

	match tool_choice {
		ResolvedToolChoice::Required => {
			prefix.push_str(
				"\n<tool_constraint>\n\
				 You should use one of the available tools to respond.\n\
				 </tool_constraint>\n",
			);
		}
		ResolvedToolChoice::Specific(name) => {
			prefix.push_str(&format!(
				"\n<tool_constraint>\n\
				 You must call the \"{}\" tool.\n\
				 </tool_constraint>\n",
				name
			));
		}
		_ => {}
	}

	prefix.push('\n');
	prefix
}

fn format_tool_definition(buf: &mut String, func: &FunctionDefinition) {
	buf.push_str(&format!("- {}", func.name));
	if let Some(desc) = &func.description {
		buf.push_str(&format!(": {}", desc));
	}
	buf.push('\n');
	if let Some(params) = &func.parameters {
		buf.push_str(&format!("  Parameters: {}\n", params));
	}
	buf.push('\n');
}

// ── Response parsing ─────────────────────────────────────────────

pub enum ParsedResponse {
	Text,
	ToolCalls(Vec<ParsedToolCall>),
	/// The response looks like a tool-call attempt (shape matches `[{"name":...}]`
	/// or `{"name":...}`) but failed to parse as valid JSON. Carries the parser
	/// error so the caller can surface a self-correcting message back to the model.
	MalformedToolCall(String),
}

pub struct ParsedToolCall {
	pub name: String,
	pub arguments: serde_json::Value,
}

#[derive(Deserialize)]
struct ToolCallCandidate {
	name: String,
	arguments: serde_json::Value,
}

/// Detect whether a response *looks* like a tool-call attempt, even if the JSON
/// is malformed. Conservative: only matches when the entire trimmed/fence-stripped
/// response begins like a tool-call array or object and ends with the matching
/// close bracket. Avoids false positives on legitimate prose containing JSON snippets.
fn looks_like_tool_call_attempt(stripped: &str) -> bool {
	let s = stripped.trim();
	if s.is_empty() {
		return false;
	}

	// Array form: starts with `[`, optional ws, `{`, optional ws, `"name"`; ends with `}]`.
	if s.starts_with('[') && s.ends_with("}]") {
		let after_brace = s[1..].trim_start().strip_prefix('{').unwrap_or("").trim_start();
		if after_brace.starts_with("\"name\"") {
			return true;
		}
	}

	// Object form: starts with `{`, optional ws, `"name"`; ends with `}`.
	if s.starts_with('{') && s.ends_with('}') {
		let after = s[1..].trim_start();
		if after.starts_with("\"name\"") {
			return true;
		}
	}

	false
}

/// Parse a model response to detect tool call JSON or plain text.
/// The prompt instructs the model to respond with ONLY a JSON array (no surrounding text),
/// but models sometimes mix prose and tool calls anyway. The parser handles both clean
/// and mixed responses as a defensive fallback.
///
/// If the response *shape* matches a tool-call attempt but the JSON is malformed
/// (e.g., unescaped quotes inside string values), returns `MalformedToolCall` with
/// the parser error so the caller can surface a self-correcting message to the model.
pub fn parse_tool_response(raw: &str) -> ParsedResponse {
	let stripped = strip_code_fences(raw.trim());

	let mut all_calls = Vec::new();
	for json_str in extract_json_arrays(stripped) {
		if let Ok(candidates) = serde_json::from_str::<Vec<ToolCallCandidate>>(json_str) {
			if !candidates.is_empty() && candidates.iter().all(|c| validate_candidate(c)) {
				all_calls.extend(candidates.into_iter().map(|c| ParsedToolCall {
					name: c.name,
					arguments: c.arguments,
				}));
			}
		}
	}
	if !all_calls.is_empty() {
		return ParsedResponse::ToolCalls(all_calls);
	}

	// Try single object: {"name": "...", "arguments": {...}}
	if let Ok(candidate) = serde_json::from_str::<ToolCallCandidate>(stripped) {
		if validate_candidate(&candidate) {
			return ParsedResponse::ToolCalls(vec![ParsedToolCall {
				name: candidate.name,
				arguments: candidate.arguments,
			}]);
		}
	}

	// Nothing parsed cleanly. If the shape looks like a tool-call attempt,
	// surface the parser error so the model can self-correct.
	if looks_like_tool_call_attempt(stripped) {
		let err = serde_json::from_str::<serde_json::Value>(stripped)
			.err()
			.map(|e| e.to_string())
			.unwrap_or_else(|| "JSON did not satisfy tool-call schema".to_string());
		return ParsedResponse::MalformedToolCall(err);
	}

	ParsedResponse::Text
}

/// Build a self-correcting assistant message returned when a tool-call attempt
/// failed to parse. Wrapped in sentinel tags so harnesses can catch it
/// programmatically; the body is plain English so the model itself recognizes
/// the problem and retries with corrected JSON on the next turn.
pub fn malformed_tool_call_message(parser_error: &str) -> String {
	format!(
		"[CCP_TOOL_CALL_PARSE_ERROR]\n\
		 Your previous response was detected as a tool call attempt, but the JSON could not be parsed.\n\
		 Parser error: {}\n\
		 Common cause: unescaped double-quote characters (\") inside string values. \
		 Inside a JSON string, every \" must be written as \\\". Backslashes must be \\\\.\n\
		 Example — INVALID:  [{{\"name\": \"exec\", \"arguments\": {{\"command\": \"echo \"hi\"\"}}}}]\n\
		 Example — VALID:    [{{\"name\": \"exec\", \"arguments\": {{\"command\": \"echo \\\"hi\\\"\"}}}}]\n\
		 Please retry with valid JSON.\n\
		 [/CCP_TOOL_CALL_PARSE_ERROR]",
		parser_error
	)
}

fn validate_candidate(c: &ToolCallCandidate) -> bool {
	!c.name.is_empty() && c.arguments.is_object()
}

/// Strip markdown code fences from text.
fn strip_code_fences(text: &str) -> &str {
	let trimmed = text.trim();

	// Check for ```json or ``` opening fence.
	let after_fence = if trimmed.starts_with("```json") {
		&trimmed[7..]
	} else if trimmed.starts_with("```") {
		&trimmed[3..]
	} else {
		return trimmed;
	};

	// Skip the newline after the opening fence.
	let after_fence = after_fence.strip_prefix('\n').unwrap_or(after_fence);

	// Find the closing fence.
	if let Some(end) = after_fence.rfind("```") {
		after_fence[..end].trim()
	} else {
		after_fence.trim()
	}
}

/// Extract all balanced JSON array substrings from text.
/// Tracks bracket nesting and string literals to find each top-level `[...]`.
fn extract_json_arrays(text: &str) -> Vec<&str> {
	let mut results = Vec::new();
	let bytes = text.as_bytes();
	let mut i = 0;

	while i < bytes.len() {
		if bytes[i] == b'[' {
			if let Some(end) = find_balanced_close(bytes, i) {
				let candidate = &text[i..=end];
				if candidate.contains("\"name\"") {
					results.push(candidate);
				}
				i = end + 1;
				continue;
			} else {
				// Unbalanced bracket — skip to end to avoid O(n^2) rescanning.
				break;
			}
		}
		i += 1;
	}

	results
}

/// Find the matching `]` for a `[` at `start`, respecting nesting and JSON strings.
fn find_balanced_close(bytes: &[u8], start: usize) -> Option<usize> {
	let mut depth = 0i32;
	let mut in_string = false;
	let mut i = start;

	while i < bytes.len() {
		let b = bytes[i];

		if in_string {
			if b == b'\\' {
				// Skip escaped character. Safe for UTF-8: all JSON escape sequences
				// after `\` are ASCII, and continuation bytes (0x80-0xBF) cannot
				// collide with the delimiters we track (`"`, `[`, `]`).
				i += 1;
			} else if b == b'"' {
				in_string = false;
			}
		} else {
			match b {
				b'"' => in_string = true,
				b'[' => depth += 1,
				b']' => {
					depth -= 1;
					if depth == 0 {
						return Some(i);
					}
				}
				_ => {}
			}
		}

		i += 1;
	}

	None
}

/// Generate a unique tool call ID.
pub fn generate_tool_call_id() -> String {
	let uuid = uuid::Uuid::new_v4().simple().to_string();
	format!("call_{}", &uuid[..24])
}

/// Convert parsed tool calls to OpenAI response format.
pub fn to_response_tool_calls(calls: Vec<ParsedToolCall>) -> Vec<ResponseToolCall> {
	calls
		.into_iter()
		.map(|c| ResponseToolCall {
			id: generate_tool_call_id(),
			call_type: "function".to_string(),
			function: ResponseFunctionCall {
				name: c.name,
				arguments: serde_json::to_string(&c.arguments).unwrap_or_default(),
			},
		})
		.collect()
}

// ── Multi-turn formatting ────────────────────────────────────────

use crate::translate::request::{ChatMessage, RequestToolCall};

/// Format an assistant message with tool_calls for the prompt.
pub fn format_assistant_tool_calls(tool_calls: &[RequestToolCall]) -> String {
	let calls: Vec<String> = tool_calls
		.iter()
		.map(|tc| format!("{}({})", tc.function.name, tc.function.arguments))
		.collect();
	format!("[Tool calls: {}]", calls.join(", "))
}

/// Format a tool result message for the prompt.
pub fn format_tool_result(msg: &ChatMessage) -> String {
	let content = crate::translate::request::extract_text(&msg.content);
	let call_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
	format!(
		"<tool_result call_id=\"{}\">\n{}\n</tool_result>",
		call_id, content
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn parsed_variant_name(p: &ParsedResponse) -> &'static str {
		match p {
			ParsedResponse::Text => "Text",
			ParsedResponse::ToolCalls(_) => "ToolCalls",
			ParsedResponse::MalformedToolCall(_) => "MalformedToolCall",
		}
	}

	// ── parse_tool_response ──────────────────────────────────

	#[test]
	fn parse_raw_json_array() {
		let raw = r#"[{"name": "get_weather", "arguments": {"location": "London"}}]"#;
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert_eq!(calls[0].name, "get_weather");
				assert_eq!(calls[0].arguments["location"], "London");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_fenced_json() {
		let raw = "```json\n[{\"name\": \"search\", \"arguments\": {\"q\": \"test\"}}]\n```";
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert_eq!(calls[0].name, "search");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_fenced_no_lang_tag() {
		let raw = "```\n[{\"name\": \"f\", \"arguments\": {}}]\n```";
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert_eq!(calls[0].name, "f");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_single_object() {
		let raw = r#"{"name": "get_time", "arguments": {}}"#;
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert_eq!(calls[0].name, "get_time");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_multiple_tool_calls() {
		let raw = r#"[{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]"#;
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 2);
				assert_eq!(calls[0].name, "a");
				assert_eq!(calls[1].name, "b");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_nested_arguments() {
		let raw = r#"[{"name": "create_event", "arguments": {"title": "Meeting", "attendees": ["Alice", "Bob"], "location": {"name": "Room A", "address": "123 St"}}}]"#;
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert!(calls[0].arguments["attendees"].is_array());
				assert!(calls[0].arguments["location"].is_object());
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_plain_text() {
		let raw = "The capital of France is Paris.";
		assert!(matches!(parse_tool_response(raw), ParsedResponse::Text));
	}

	#[test]
	fn parse_malformed_json_garbage_inside_returns_malformed() {
		// Shape matches a tool-call attempt but the JSON body is broken.
		let raw = "[{\"name\": \"f\", \"arguments\": {broken}}]";
		assert!(matches!(
			parse_tool_response(raw),
			ParsedResponse::MalformedToolCall(_)
		));
	}

	#[test]
	fn parse_malformed_unescaped_quotes_returns_malformed() {
		// The real-world OpenClaw failure case: unescaped " inside string value.
		let raw = r#"[{"name":"exec","arguments":{"command":"echo "hi""}}]"#;
		match parse_tool_response(raw) {
			ParsedResponse::MalformedToolCall(err) => {
				assert!(!err.is_empty());
			}
			_ => panic!("Expected MalformedToolCall for unescaped quotes"),
		}
	}

	#[test]
	fn parse_malformed_object_form_returns_malformed() {
		let raw = r#"{"name":"exec","arguments":{"command":"echo "hi""}}"#;
		assert!(matches!(
			parse_tool_response(raw),
			ParsedResponse::MalformedToolCall(_)
		));
	}

	#[test]
	fn parse_prose_with_broken_brackets_returns_text() {
		// Doesn't match the tool-call shape — should be Text, not Malformed.
		let raw = "Here is some text [not a tool call] and more text.";
		assert!(matches!(parse_tool_response(raw), ParsedResponse::Text));
	}

	#[test]
	fn malformed_message_contains_sentinels_and_guidance() {
		let msg = malformed_tool_call_message("test parser error");
		assert!(msg.contains("[CCP_TOOL_CALL_PARSE_ERROR]"));
		assert!(msg.contains("[/CCP_TOOL_CALL_PARSE_ERROR]"));
		assert!(msg.contains("test parser error"));
		assert!(msg.contains("\\\""));
	}

	#[test]
	fn parse_empty_name_rejected_as_malformed() {
		// Shape matches a tool-call attempt but the schema fails validation
		// (empty name). Surface as MalformedToolCall so the model self-corrects.
		let raw = r#"[{"name": "", "arguments": {}}]"#;
		assert!(matches!(
			parse_tool_response(raw),
			ParsedResponse::MalformedToolCall(_)
		));
	}

	#[test]
	fn parse_arguments_not_object_rejected_as_malformed() {
		// Same as above: shape matches, schema fails (arguments is a string).
		let raw = r#"[{"name": "f", "arguments": "not an object"}]"#;
		assert!(matches!(
			parse_tool_response(raw),
			ParsedResponse::MalformedToolCall(_)
		));
	}

	#[test]
	fn parse_json_with_trailing_text() {
		let raw = "[{\"name\": \"f\", \"arguments\": {\"x\": 1}}]\n\nI'm Claude, an AI assistant...";
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert_eq!(calls[0].name, "f");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_fenced_json_with_trailing_text() {
		let raw =
			"```json\n[{\"name\": \"f\", \"arguments\": {}}]\n```\n\nNote: I don't have tools.";
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_tool_calls_embedded_in_prose() {
		let raw = r#"Let me find those.[{"name": "exec", "arguments": {"command": "ls", "timeout": 10}}]  The latest error is a 403."#;
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert_eq!(calls[0].name, "exec");
				assert_eq!(calls[0].arguments["command"], "ls");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_multiple_tool_call_arrays_in_prose() {
		let raw = r#"Some text [{"name": "a", "arguments": {"x": 1}}] middle text [{"name": "b", "arguments": {"y": 2}}] end"#;
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 2);
				assert_eq!(calls[0].name, "a");
				assert_eq!(calls[1].name, "b");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn parse_tool_calls_with_nested_arrays_in_args() {
		let raw = r#"Here: [{"name": "f", "arguments": {"items": ["a", "b"]}}] done"#;
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert_eq!(calls[0].name, "f");
				assert!(calls[0].arguments["items"].is_array());
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	#[test]
	fn plain_text_with_brackets_not_tool_calls() {
		let raw = "The array [1, 2, 3] contains numbers.";
		assert!(matches!(parse_tool_response(raw), ParsedResponse::Text));
	}

	#[test]
	fn escaped_quotes_in_arguments() {
		let raw = r#"[{"name": "exec", "arguments": {"command": "echo \"hello\""}}]"#;
		match parse_tool_response(raw) {
			ParsedResponse::ToolCalls(calls) => {
				assert_eq!(calls.len(), 1);
				assert_eq!(calls[0].name, "exec");
			}
			other => panic!("Expected ToolCalls, got {}", parsed_variant_name(&other)),
		}
	}

	// ── strip_code_fences ────────────────────────────────────

	#[test]
	fn strip_json_fence() {
		assert_eq!(strip_code_fences("```json\n[1,2,3]\n```"), "[1,2,3]");
	}

	#[test]
	fn strip_plain_fence() {
		assert_eq!(strip_code_fences("```\nhello\n```"), "hello");
	}

	#[test]
	fn strip_no_fence() {
		assert_eq!(strip_code_fences("plain text"), "plain text");
	}

	#[test]
	fn strip_fence_with_whitespace() {
		assert_eq!(
			strip_code_fences("  ```json\n  [1]  \n```  "),
			"[1]"
		);
	}

	// ── resolve_tool_choice ──────────────────────────────────

	#[test]
	fn resolve_none_is_auto() {
		assert!(matches!(resolve_tool_choice(&None), ResolvedToolChoice::Auto));
	}

	#[test]
	fn resolve_auto_string() {
		let choice = Some(ToolChoice::Mode("auto".into()));
		assert!(matches!(resolve_tool_choice(&choice), ResolvedToolChoice::Auto));
	}

	#[test]
	fn resolve_none_string() {
		let choice = Some(ToolChoice::Mode("none".into()));
		assert!(matches!(resolve_tool_choice(&choice), ResolvedToolChoice::None));
	}

	#[test]
	fn resolve_required_string() {
		let choice = Some(ToolChoice::Mode("required".into()));
		assert!(matches!(
			resolve_tool_choice(&choice),
			ResolvedToolChoice::Required
		));
	}

	#[test]
	fn resolve_specific_function() {
		let choice = Some(ToolChoice::Specific {
			choice_type: "function".into(),
			function: crate::translate::request::ToolChoiceFunction {
				name: "get_weather".into(),
			},
		});
		match resolve_tool_choice(&choice) {
			ResolvedToolChoice::Specific(name) => assert_eq!(name, "get_weather"),
			_ => panic!("Expected Specific"),
		}
	}

	// ── generate_tool_call_id ────────────────────────────────

	#[test]
	fn tool_call_id_format() {
		let id = generate_tool_call_id();
		assert!(id.starts_with("call_"));
		assert_eq!(id.len(), 29); // "call_" + 24 hex chars
	}

	// ── build_tool_prompt_prefix ─────────────────────────────

	#[test]
	fn prefix_contains_tool_name() {
		let tools = vec![ToolDefinition {
			tool_type: Some("function".into()),
			function: FunctionDefinition {
				name: "get_weather".into(),
				description: Some("Get weather".into()),
				parameters: None,
			},
		}];
		let prefix = build_tool_prompt_prefix(&tools, &ResolvedToolChoice::Auto);
		assert!(prefix.contains("get_weather"));
		assert!(prefix.contains("Get weather"));
		assert!(!prefix.contains("<tool_constraint>"));
	}

	#[test]
	fn prefix_required_has_constraint() {
		let tools = vec![ToolDefinition {
			tool_type: Some("function".into()),
			function: FunctionDefinition {
				name: "f".into(),
				description: None,
				parameters: None,
			},
		}];
		let prefix = build_tool_prompt_prefix(&tools, &ResolvedToolChoice::Required);
		assert!(prefix.contains("<tool_constraint>"));
		assert!(prefix.contains("should use one of the available tools"));
	}

	#[test]
	fn prefix_specific_has_function_name() {
		let tools = vec![ToolDefinition {
			tool_type: Some("function".into()),
			function: FunctionDefinition {
				name: "search".into(),
				description: None,
				parameters: None,
			},
		}];
		let prefix =
			build_tool_prompt_prefix(&tools, &ResolvedToolChoice::Specific("search".into()));
		assert!(prefix.contains("must call the \"search\" tool"));
	}
}
