use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Scope {
	Prompt,
	Response,
	Both,
}

#[derive(Deserialize)]
struct RawRule {
	scope: Scope,
	search: String,
	replace: String,
}

#[derive(Deserialize)]
struct RulesFile {
	rule: Vec<RawRule>,
}

#[derive(Debug)]
struct Rule {
	search: String,
	replace: String,
}

#[derive(Debug)]
pub struct Replacements {
	prompt_rules: Vec<Rule>,
	response_rules: Vec<Rule>,
	/// Number of rules in the source file (before Both-scope expansion).
	file_rule_count: usize,
}

impl Replacements {
	pub fn empty() -> Self {
		Self {
			prompt_rules: Vec::new(),
			response_rules: Vec::new(),
			file_rule_count: 0,
		}
	}

	/// Load replacement rules from a TOML file.
	pub fn load(path: &Path) -> Result<Self, String> {
		let contents = std::fs::read_to_string(path)
			.map_err(|e| format!("Failed to read replacement rules file {:?}: {}", path, e))?;

		Self::parse(&contents)
			.map_err(|e| format!("Failed to parse replacement rules file {:?}: {}", path, e))
	}

	/// Parse replacement rules from a TOML string.
	fn parse(toml_str: &str) -> Result<Self, String> {
		let file: RulesFile =
			toml::from_str(toml_str).map_err(|e| format!("{}", e))?;

		let file_rule_count = file.rule.len();
		let mut prompt_rules = Vec::new();
		let mut response_rules = Vec::new();

		for (i, raw) in file.rule.into_iter().enumerate() {
			if raw.search.is_empty() {
				return Err(format!("Rule {} has empty search string", i + 1));
			}

			let rule = Rule {
				search: raw.search,
				replace: raw.replace,
			};

			match raw.scope {
				Scope::Prompt => prompt_rules.push(rule),
				Scope::Response => response_rules.push(rule),
				Scope::Both => {
					response_rules.push(Rule {
						search: rule.search.clone(),
						replace: rule.replace.clone(),
					});
					prompt_rules.push(rule);
				}
			}
		}

		Ok(Self {
			prompt_rules,
			response_rules,
			file_rule_count,
		})
	}

	pub fn is_empty(&self) -> bool {
		self.prompt_rules.is_empty() && self.response_rules.is_empty()
	}

	/// Number of rules as they appear in the source file.
	pub fn count(&self) -> usize {
		self.file_rule_count
	}

	pub fn apply_prompt(&self, text: &str) -> String {
		apply_rules(text, &self.prompt_rules)
	}

	pub fn apply_response(&self, text: &str) -> String {
		apply_rules(text, &self.response_rules)
	}
}

fn apply_rules(text: &str, rules: &[Rule]) -> String {
	if rules.is_empty() {
		return text.to_string();
	}
	let mut result = text.to_string();
	for rule in rules {
		result = result.replace(&rule.search, &rule.replace);
	}
	result
}

#[cfg(test)]
mod tests {
	use super::*;

	fn load_from_str(toml_str: &str) -> Result<Replacements, String> {
		Replacements::parse(toml_str)
	}

	#[test]
	fn parse_all_scopes() {
		let r = load_from_str(r#"
			[[rule]]
			scope = "prompt"
			search = "foo"
			replace = "bar"

			[[rule]]
			scope = "response"
			search = "baz"
			replace = "qux"

			[[rule]]
			scope = "both"
			search = "old"
			replace = "new"
		"#)
		.unwrap();
		assert_eq!(r.prompt_rules.len(), 2); // prompt + both
		assert_eq!(r.response_rules.len(), 2); // response + both
		assert_eq!(r.count(), 3); // 3 rules in the file
	}

	#[test]
	fn reject_empty_search() {
		let result = load_from_str(r#"
			[[rule]]
			scope = "prompt"
			search = ""
			replace = "bar"
		"#);
		assert!(result.is_err());
		assert!(result.unwrap_err().contains("empty search"));
	}

	#[test]
	fn reject_invalid_scope() {
		let result = load_from_str(r#"
			[[rule]]
			scope = "invalid"
			search = "foo"
			replace = "bar"
		"#);
		assert!(result.is_err());
	}

	#[test]
	fn prompt_rules_dont_apply_to_response() {
		let r = load_from_str(r#"
			[[rule]]
			scope = "prompt"
			search = "secret"
			replace = "REDACTED"
		"#)
		.unwrap();
		assert_eq!(r.apply_prompt("my secret"), "my REDACTED");
		assert_eq!(r.apply_response("my secret"), "my secret");
	}

	#[test]
	fn response_rules_dont_apply_to_prompt() {
		let r = load_from_str(r#"
			[[rule]]
			scope = "response"
			search = "hello"
			replace = "goodbye"
		"#)
		.unwrap();
		assert_eq!(r.apply_prompt("hello world"), "hello world");
		assert_eq!(r.apply_response("hello world"), "goodbye world");
	}

	#[test]
	fn both_scope_applies_everywhere() {
		let r = load_from_str(r#"
			[[rule]]
			scope = "both"
			search = "old"
			replace = "new"
		"#)
		.unwrap();
		assert_eq!(r.apply_prompt("old value"), "new value");
		assert_eq!(r.apply_response("old value"), "new value");
	}

	#[test]
	fn rules_applied_in_order() {
		let r = load_from_str(r#"
			[[rule]]
			scope = "prompt"
			search = "foo"
			replace = "bar"

			[[rule]]
			scope = "prompt"
			search = "bar"
			replace = "baz"
		"#)
		.unwrap();
		assert_eq!(r.apply_prompt("foo"), "baz");
	}

	#[test]
	fn empty_replacement_is_deletion() {
		let r = load_from_str(r#"
			[[rule]]
			scope = "response"
			search = "remove me"
			replace = ""
		"#)
		.unwrap();
		assert_eq!(r.apply_response("please remove me now"), "please  now");
	}

	#[test]
	fn empty_replacements_is_noop() {
		let r = Replacements::empty();
		assert!(r.is_empty());
		assert_eq!(r.apply_prompt("hello"), "hello");
		assert_eq!(r.apply_response("hello"), "hello");
	}

	#[test]
	fn colons_in_search_and_replace() {
		let r = load_from_str(r#"
			[[rule]]
			scope = "both"
			search = "http://old.example.com:8080"
			replace = "https://new.example.com:443"
		"#)
		.unwrap();
		assert_eq!(
			r.apply_prompt("visit http://old.example.com:8080/api"),
			"visit https://new.example.com:443/api"
		);
	}
}
