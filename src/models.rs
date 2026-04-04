use std::borrow::Cow;

use serde::Serialize;

use crate::error::AppError;

#[derive(Debug)]
pub struct ModelDef {
	pub canonical: &'static str,
	pub cli_name: &'static str,
	pub aliases: &'static [&'static str],
	pub context_window: u64,
	pub max_tokens: u64,
}

pub static MODELS: &[ModelDef] = &[
	ModelDef {
		canonical: "claude-opus-4-6",
		cli_name: "opus",
		aliases: &["opus", "claude-opus"],
		context_window: 1_000_000,
		max_tokens: 128_000,
	},
	ModelDef {
		canonical: "claude-sonnet-4-6",
		cli_name: "sonnet",
		aliases: &["sonnet", "claude-sonnet"],
		context_window: 1_000_000,
		max_tokens: 64_000,
	},
	ModelDef {
		canonical: "claude-haiku-4-5",
		cli_name: "haiku",
		aliases: &["haiku", "claude-haiku"],
		context_window: 200_000,
		max_tokens: 64_000,
	},
];

/// Match a model string by substring against known model families.
fn match_by_substring(input: &str) -> Option<&'static ModelDef> {
	for m in MODELS {
		if input.contains(m.cli_name) {
			return Some(m);
		}
	}
	None
}

/// Resolve an input model string to a ModelDef.
/// Tries exact canonical match, then alias match, then substring fallback.
/// Falls back to sonnet with a warning.
pub fn resolve_model(input: &str) -> &'static ModelDef {
	// Exact canonical match.
	for m in MODELS {
		if m.canonical == input {
			return m;
		}
	}

	// Alias match.
	for m in MODELS {
		for alias in m.aliases {
			if *alias == input {
				return m;
			}
		}
	}

	if let Some(m) = match_by_substring(input) {
		return m;
	}

	tracing::warn!(model = %input, "Unrecognized model, falling back to sonnet");
	&MODELS[1]
}

/// Normalize a raw CLI model string to a canonical name.
/// Uses substring matching. Falls back to the raw string as-is.
pub fn normalize_model_name(raw: &str) -> Cow<'static, str> {
	match match_by_substring(raw) {
		Some(m) => Cow::Borrowed(m.canonical),
		None => Cow::Owned(raw.to_string()),
	}
}

/// Validate the reasoning_effort field.
/// Returns the effort string to pass to --effort, or None if the flag should be omitted.
pub fn validate_effort(effort: Option<&str>) -> Result<Option<&'static str>, AppError> {
	match effort {
		None => Ok(None),
		Some("none") => Ok(None),
		Some("low") => Ok(Some("low")),
		Some("medium") => Ok(Some("medium")),
		Some("high") => Ok(Some("high")),
		Some("max") => Ok(Some("max")),
		Some(other) => Err(AppError::BadRequest(format!(
			"Invalid reasoning_effort: '{}'. Valid values: none, low, medium, high, max",
			other
		))),
	}
}

#[derive(Serialize)]
pub struct ModelInfo {
	pub id: String,
	pub object: &'static str,
	pub created: u64,
	pub owned_by: &'static str,
	pub context_window: u64,
	pub max_tokens: u64,
}

/// Return the model list for GET /v1/models.
pub fn models_list() -> Vec<ModelInfo> {
	MODELS
		.iter()
		.map(|m| ModelInfo {
			id: m.canonical.to_string(),
			object: "model",
			created: 0,
			owned_by: "anthropic",
			context_window: m.context_window,
			max_tokens: m.max_tokens,
		})
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn resolve_canonical_names() {
		assert_eq!(resolve_model("claude-opus-4-6").canonical, "claude-opus-4-6");
		assert_eq!(
			resolve_model("claude-sonnet-4-6").canonical,
			"claude-sonnet-4-6"
		);
		assert_eq!(
			resolve_model("claude-haiku-4-5").canonical,
			"claude-haiku-4-5"
		);
	}

	#[test]
	fn resolve_short_aliases() {
		assert_eq!(resolve_model("opus").canonical, "claude-opus-4-6");
		assert_eq!(resolve_model("sonnet").canonical, "claude-sonnet-4-6");
		assert_eq!(resolve_model("haiku").canonical, "claude-haiku-4-5");
	}

	#[test]
	fn resolve_claude_prefix_aliases() {
		assert_eq!(resolve_model("claude-opus").canonical, "claude-opus-4-6");
		assert_eq!(
			resolve_model("claude-sonnet").canonical,
			"claude-sonnet-4-6"
		);
		assert_eq!(resolve_model("claude-haiku").canonical, "claude-haiku-4-5");
	}

	#[test]
	fn resolve_date_suffixed_via_substring() {
		assert_eq!(
			resolve_model("claude-opus-4-6-20260101").canonical,
			"claude-opus-4-6"
		);
		assert_eq!(
			resolve_model("claude-sonnet-4-6-20260101").canonical,
			"claude-sonnet-4-6"
		);
		assert_eq!(
			resolve_model("claude-haiku-4-5-20251001").canonical,
			"claude-haiku-4-5"
		);
	}

	#[test]
	fn resolve_unknown_falls_back_to_sonnet() {
		assert_eq!(resolve_model("gpt-4").canonical, "claude-sonnet-4-6");
		assert_eq!(resolve_model("unknown").canonical, "claude-sonnet-4-6");
		assert_eq!(resolve_model("").canonical, "claude-sonnet-4-6");
	}

	#[test]
	fn normalize_model_name_substring() {
		assert_eq!(
			normalize_model_name("claude-opus-4-6-20260101").as_ref(),
			"claude-opus-4-6"
		);
		assert_eq!(
			normalize_model_name("claude-sonnet-4-6").as_ref(),
			"claude-sonnet-4-6"
		);
		assert_eq!(
			normalize_model_name("claude-haiku-4-5-20251001").as_ref(),
			"claude-haiku-4-5"
		);
	}

	#[test]
	fn normalize_model_name_unknown_returns_raw() {
		assert_eq!(normalize_model_name("gpt-4").as_ref(), "gpt-4");
	}

	#[test]
	fn validate_effort_valid_values() {
		assert_eq!(validate_effort(None).unwrap(), None);
		assert_eq!(validate_effort(Some("none")).unwrap(), None);
		assert_eq!(validate_effort(Some("low")).unwrap(), Some("low"));
		assert_eq!(validate_effort(Some("medium")).unwrap(), Some("medium"));
		assert_eq!(validate_effort(Some("high")).unwrap(), Some("high"));
		assert_eq!(validate_effort(Some("max")).unwrap(), Some("max"));
	}

	#[test]
	fn validate_effort_invalid() {
		assert!(validate_effort(Some("extreme")).is_err());
		assert!(validate_effort(Some("")).is_err());
	}

	#[test]
	fn models_list_returns_three() {
		let list = models_list();
		assert_eq!(list.len(), 3);
		assert_eq!(list[0].id, "claude-opus-4-6");
		assert_eq!(list[1].id, "claude-sonnet-4-6");
		assert_eq!(list[2].id, "claude-haiku-4-5");
		assert_eq!(list[0].context_window, 1_000_000);
		assert_eq!(list[2].max_tokens, 64_000);
	}
}
