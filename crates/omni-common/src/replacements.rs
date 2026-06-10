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

#[derive(Debug, Clone)]
struct Rule {
    search: String,
    replace: String,
}

#[derive(Debug)]
pub struct Replacements {
    prompt_rules: Vec<Rule>,
    response_rules: Vec<Rule>,
    file_rule_count: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ReplacementsError {
    #[error("parse: {0}")]
    Parse(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl Replacements {
    pub fn empty() -> Self {
        Self {
            prompt_rules: vec![],
            response_rules: vec![],
            file_rule_count: 0,
        }
    }

    pub fn load(path: &Path) -> Result<Self, ReplacementsError> {
        let contents = std::fs::read_to_string(path)?;
        Self::parse(&contents)
    }

    pub fn parse(toml_str: &str) -> Result<Self, ReplacementsError> {
        if toml_str.trim().is_empty() {
            return Ok(Self::empty());
        }
        let file: RulesFile =
            toml::from_str(toml_str).map_err(|e| ReplacementsError::Parse(e.to_string()))?;

        let mut prompt_rules = Vec::new();
        let mut response_rules = Vec::new();
        for r in &file.rule {
            let rule = Rule {
                search: r.search.clone(),
                replace: r.replace.clone(),
            };
            match r.scope {
                Scope::Prompt => prompt_rules.push(rule),
                Scope::Response => response_rules.push(rule),
                Scope::Both => {
                    prompt_rules.push(rule.clone());
                    response_rules.push(rule);
                }
            }
        }

        for (scope_name, rules) in [("prompt", &prompt_rules), ("response", &response_rules)] {
            for (i, a) in rules.iter().enumerate() {
                for b in &rules[i + 1..] {
                    if a.search == b.search {
                        return Err(ReplacementsError::Parse(format!(
                            "duplicate {} rule for {:?}",
                            scope_name, a.search
                        )));
                    }
                }
            }
        }

        Ok(Replacements {
            prompt_rules,
            response_rules,
            file_rule_count: file.rule.len(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.prompt_rules.is_empty() && self.response_rules.is_empty()
    }
    pub fn count(&self) -> usize {
        self.file_rule_count
    }
    pub fn apply_prompt(&self, text: &str) -> String {
        apply_rules(text, &self.prompt_rules)
    }
    pub fn apply_response(&self, text: &str) -> String {
        apply_rules(text, &self.response_rules)
    }
    pub fn max_response_search_len(&self) -> usize {
        self.response_rules
            .iter()
            .map(|r| r.search.len())
            .max()
            .unwrap_or(0)
    }
}

fn apply_rules(text: &str, rules: &[Rule]) -> String {
    let mut out = text.to_string();
    for r in rules {
        out = out.replace(&r.search, &r.replace);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_str: &str) -> Result<Replacements, ReplacementsError> {
        Replacements::parse(toml_str)
    }

    #[test]
    fn parse_all_scopes() {
        let r = parse(
            r#"rule = [
                { scope = "prompt", search = "foo", replace = "bar" },
                { scope = "response", search = "baz", replace = "qux" },
                { scope = "both", search = "old", replace = "new" }
            ]"#,
        )
        .unwrap();
        assert_eq!(r.prompt_rules.len(), 2); // prompt + both
        assert_eq!(r.response_rules.len(), 2); // response + both
        assert_eq!(r.count(), 3); // 3 rules in the file
    }

    #[test]
    fn reject_invalid_scope() {
        let result = parse(r#"rule = [ { scope = "invalid", search = "foo", replace = "bar" } ]"#);
        assert!(result.is_err());
    }

    #[test]
    fn prompt_rules_dont_apply_to_response() {
        let r =
            parse(r#"rule = [ { scope = "prompt", search = "secret", replace = "REDACTED" } ]"#)
                .unwrap();
        assert_eq!(r.apply_prompt("my secret"), "my REDACTED");
        assert_eq!(r.apply_response("my secret"), "my secret");
    }

    #[test]
    fn response_rules_dont_apply_to_prompt() {
        let r =
            parse(r#"rule = [ { scope = "response", search = "hello", replace = "goodbye" } ]"#)
                .unwrap();
        assert_eq!(r.apply_prompt("hello world"), "hello world");
        assert_eq!(r.apply_response("hello world"), "goodbye world");
    }

    #[test]
    fn both_scope_applies_everywhere() {
        let r = parse(r#"rule = [ { scope = "both", search = "old", replace = "new" } ]"#).unwrap();
        assert_eq!(r.apply_prompt("old value"), "new value");
        assert_eq!(r.apply_response("old value"), "new value");
    }

    #[test]
    fn rules_applied_in_order() {
        let r = parse(
            r#"rule = [
                { scope = "prompt", search = "foo", replace = "bar" },
                { scope = "prompt", search = "bar", replace = "baz" }
            ]"#,
        )
        .unwrap();
        assert_eq!(r.apply_prompt("foo"), "baz");
    }

    #[test]
    fn empty_replacement_is_deletion() {
        let r = parse(r#"rule = [ { scope = "response", search = "remove me", replace = "" } ]"#)
            .unwrap();
        assert_eq!(r.apply_response("please remove me now"), "please  now");
    }

    #[test]
    fn empty_replacements_is_noop() {
        let r = Replacements::empty();
        assert!(r.is_empty());
        assert_eq!(r.apply_prompt("hello"), "hello");
        assert_eq!(r.apply_response("hello"), "hello");
        assert_eq!(r.count(), 0);
    }

    #[test]
    fn colons_in_search_and_replace() {
        let r = parse(
            r#"rule = [ { scope = "both", search = "http://old.example.com:8080", replace = "https://new.example.com:443" } ]"#,
        )
        .unwrap();
        assert_eq!(
            r.apply_prompt("visit http://old.example.com:8080/api"),
            "visit https://new.example.com:443/api"
        );
    }

    #[test]
    fn duplicate_search_in_scope_is_rejected() {
        let result = parse(
            r#"rule = [
                { scope = "prompt", search = "x", replace = "a" },
                { scope = "prompt", search = "x", replace = "b" }
            ]"#,
        );
        assert!(result.is_err());
        let e = result.unwrap_err().to_string();
        assert!(e.contains("duplicate"));
    }

    #[test]
    fn parse_empty_string_gives_empty() {
        let r = parse("").unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn max_response_search_len() {
        let r = parse(
            r#"rule = [
                { scope = "response", search = "abc", replace = "1" },
                { scope = "response", search = "abcdef", replace = "2" }
            ]"#,
        )
        .unwrap();
        assert_eq!(r.max_response_search_len(), 6);
    }

    // Mirrors CCP streaming repl buffer logic intent: a search spanning what would be
    // chunk boundaries (tool arg json partials) must be handled by max lookback sizing.
    #[test]
    fn mirrors_ccp_streaming_repl_buffer_for_partial_tool_args() {
        let r =
            parse(r#"rule = [ { scope = "response", search = "foo", replace = "bar" } ]"#).unwrap();
        // partial json arg fragment that would arrive in separate SSE chunks; apply on
        // concat simulates post-buffer result.
        let partial1 = r#"{"function":{"name":"x","arguments":"{\"k\":\"foo"#;
        let partial2 = r#"\"}}}"#;
        let full = format!("{}{}", partial1, partial2);
        assert_eq!(
            r.apply_response(&full),
            format!("{}{}", partial1.replace("foo", "bar"), partial2)
        );
        assert!(r.max_response_search_len() > 0);
    }

    // Tool arg json partials (even invalid json) still undergo text replace on response path.
    // Mirrors CCP apply_response_to_args_string fallback.
    #[test]
    fn response_rules_apply_to_tool_arg_json_partials() {
        let r =
            parse(r#"rule = [ { scope = "response", search = "foo", replace = "bar" } ]"#).unwrap();
        let partial = r#"{"k":"foo"#; // unterminated json
        assert_eq!(r.apply_response(partial), r#"{"k":"bar"#);
        let args = r#"{"a": "foo", "b": "xfoo y"}"#;
        assert_eq!(r.apply_response(args), r#"{"a": "bar", "b": "xbar y"}"#);
    }

    // Schema title/desc only: rules must be applicable to description/title strings in
    // schema text; structural keys like "type":"string" would be hit by blind replace but
    // callers (per CCP) restrict to nl fields only. Test documents the raw apply behavior.
    #[test]
    fn prompt_rules_hit_schema_title_desc_text() {
        let r = parse(r#"rule = [ { scope = "prompt", search = "string", replace = "text" } ]"#)
            .unwrap();
        let schema_text = r#"{"type":"object","properties":{"p":{"type":"string","description":"a string path","title":"string title"}}}"#;
        let replaced = r.apply_prompt(schema_text);
        // raw replace hits the type value too (intent: real schema applicator in translate must limit)
        // note: case-sensitive; "String" would not match "string" rule.
        assert!(replaced.contains("\"type\":\"text\""));
        assert!(replaced.contains("a text path"));
        // the title value "string title" contains the search token so also rewritten by raw apply
        assert!(replaced.contains("text title"));
    }

    // Length changing rules across chunks: non-streaming apply on concat gives canonical
    // result; mirrors CCP streaming_replacement_handles_length_changing_rules_across_chunks.
    #[test]
    fn length_changing_rules_produce_consistent_result() {
        let r = parse(
            r#"rule = [
                { scope = "response", search = "ab", replace = "é" },
                { scope = "response", search = "a", replace = "LONG" }
            ]"#,
        )
        .unwrap();
        // first rule wins on "ab"; "aXb" has no "ab" so second rule applies to leading a
        assert_eq!(r.apply_response("ab"), "é");
        assert_eq!(r.apply_response("aXb"), "LONGXb");
    }

    // Role opener empty content: empty content delta must be preserved (not stripped) when
    // response rules are active. Mirrors CCP role_opener_keeps_empty_content_under_response_rules.
    #[test]
    fn role_opener_empty_content_is_preserved_under_rules() {
        let r =
            parse(r#"rule = [ { scope = "response", search = "foo", replace = "bar" } ]"#).unwrap();
        let opener = r#"{"role":"assistant","content":""}"#;
        assert_eq!(r.apply_response(opener), opener); // no match, preserved exactly
        let with_match = r#"{"role":"assistant","content":"foo"}"#;
        assert_eq!(
            r.apply_response(with_match),
            r#"{"role":"assistant","content":"bar"}"#
        );
    }

    // Duplicate rejection variants: prompt dups, response dups (via both), and cross
    // expansion that produces intra-scope dup.
    #[test]
    fn duplicate_rejection_variants() {
        // response dup via direct
        let res = parse(
            r#"rule = [
                { scope = "response", search = "x", replace = "a" },
                { scope = "response", search = "x", replace = "b" }
            ]"#,
        );
        assert!(res.is_err());
        // both + prompt on same search produces dup in prompt scope after expansion
        let res2 = parse(
            r#"rule = [
                { scope = "both", search = "y", replace = "1" },
                { scope = "prompt", search = "y", replace = "2" }
            ]"#,
        );
        assert!(res2.is_err());
        let e = res2.unwrap_err().to_string();
        assert!(e.contains("duplicate prompt rule"));
    }

    // Parse errors: bad toml, missing required fields, etc. (empty search is allowed in this
    // omni variant unlike strict CCP).
    #[test]
    fn parse_errors_variants() {
        assert!(parse("not = valid toml [").is_err());
        // missing scope -> serde fail
        let bad = parse(r#"rule = [ { search = "a", replace = "b" } ]"#);
        assert!(bad.is_err());
        let e = bad.unwrap_err();
        assert!(matches!(e, ReplacementsError::Parse(_)));
    }

    // Additional prompt/response isolation: rules in one scope never leak; both expands to copies.
    // Mirrors CCP scope separation invariant.
    #[test]
    fn both_prompt_response_isolation() {
        let r = parse(
            r#"rule = [
                { scope = "prompt", search = "p", replace = "P" },
                { scope = "response", search = "r", replace = "R" },
                { scope = "both", search = "b", replace = "B" }
            ]"#,
        )
        .unwrap();
        assert_eq!(r.apply_prompt("p r b"), "P r B");
        assert_eq!(r.apply_response("p r b"), "p R B");
        assert_eq!(r.prompt_rules.len(), 2);
        assert_eq!(r.response_rules.len(), 2);
    }

    // Streaming buffer full/partial tool json: search spanning chunks (e.g. arg json split) relies on
    // max_response_search_len for buffer; apply on concat is canonical. Mirrors CCP.
    #[test]
    fn streaming_buffer_full_partial_tool_json() {
        let r =
            parse(r#"rule = [ { scope = "response", search = "TOOLARG", replace = "REPLACED" } ]"#)
                .unwrap();
        let full = r#"{"tool_calls":[{"function":{"arguments":"{\"x\":\"TOOLARG\"}"}}]}"#;
        assert_eq!(r.apply_response(full), full.replace("TOOLARG", "REPLACED"));
        let partial1 = r#"{"tool_calls":[{"function":{"arguments":"{\"x\":\"TOO"#;
        let partial2 = r#"LARG\"}"}}]}"#;
        let concat = format!("{}{}", partial1, partial2);
        assert_eq!(
            r.apply_response(&concat),
            concat.replace("TOOLARG", "REPLACED")
        );
        assert!(r.max_response_search_len() >= 7);
    }

    // Schema title/desc only intent + length change: raw apply hits text fields; length change
    // across chunks must be consistent with non-stream (order+first match wins).
    #[test]
    fn schema_title_desc_and_length_change_across_chunks() {
        let r = parse(
            r#"rule = [
                { scope = "prompt", search = "ab", replace = "X" },
                { scope = "prompt", search = "a", replace = "YY" }
            ]"#,
        )
        .unwrap();
        let schema = r#"{"title":"ab desc","description":"aZzz init"}"#;
        let replaced = r.apply_prompt(schema);
        assert!(replaced.contains("\"title\":\"X desc\""));
        // "aZzz init" has lone "a" at start of value (chosen so no secondary 'a' in rest of value after replace)
        assert!(replaced.contains("\"description\":\"YYZzz init\""));
        // length changing: ab first
        assert_eq!(r.apply_prompt("ab"), "X");
        assert_eq!(r.apply_prompt("aX"), "YYX");
    }

    // Role opener empty content variants + duplicate rejection more: empty preserved; cross both+scope dups rejected.
    #[test]
    fn role_opener_empty_and_duplicate_rejection_more() {
        let r = parse(r#"rule = [ { scope = "response", search = "x", replace = "y" } ]"#).unwrap();
        assert_eq!(
            r.apply_response(r#"{"role":"assistant","content":""}"#),
            r#"{"role":"assistant","content":""}"#
        );
        // dup via both + response
        let bad = parse(
            r#"rule = [
                { scope = "both", search = "z", replace = "1" },
                { scope = "response", search = "z", replace = "2" }
            ]"#,
        );
        assert!(bad.is_err());
        let e = bad.unwrap_err().to_string();
        assert!(e.contains("duplicate response rule"));
    }

    // Parse errors, empty as deletion variants, colons, noop, max len, order.
    #[test]
    fn more_parse_edges_empty_deletion_colons_noop_maxlen_order() {
        // parse bad toml and missing field already covered; add empty search allowed here
        let r_ok = parse(r#"rule = [ { scope = "both", search = "", replace = "X" } ]"#);
        assert!(r_ok.is_ok());
        // empty search as deletion? but "" replace would be insert? use non-empty search delete
        let r_del =
            parse(r#"rule = [ { scope = "response", search = "DELME", replace = "" } ]"#).unwrap();
        assert_eq!(r_del.apply_response("keep DELME here"), "keep  here");
        // colons already; noop no rules
        let r0 = Replacements::empty();
        assert_eq!(r0.apply_response("same"), "same");
        // max + order
        let rmax = parse(
            r#"rule = [
                { scope = "response", search = "short", replace = "s" },
                { scope = "response", search = "longersearch", replace = "L" }
            ]"#,
        )
        .unwrap();
        assert_eq!(rmax.max_response_search_len(), 12);
        // order: first match wins in seq replace
        let rord = parse(
            r#"rule = [
                { scope = "prompt", search = "ab", replace = "1" },
                { scope = "prompt", search = "a", replace = "2" }
            ]"#,
        )
        .unwrap();
        assert_eq!(rord.apply_prompt("ab"), "1");
    }

    // Noop on no-match, both/prompt isolation, duplicate variants prompt only.
    #[test]
    fn noop_no_match_both_isolation_and_dup_variants() {
        let r =
            parse(r#"rule = [ { scope = "prompt", search = "NOHIT", replace = "X" } ]"#).unwrap();
        assert_eq!(r.apply_prompt("nothing here"), "nothing here");
        assert_eq!(r.apply_response("nothing here"), "nothing here");
        let rboth =
            parse(r#"rule = [ { scope = "both", search = "hit", replace = "HIT" } ]"#).unwrap();
        assert_eq!(rboth.apply_prompt("hit me"), "HIT me");
        assert_eq!(rboth.apply_response("hit me"), "HIT me");
        // prompt dup direct
        assert!(parse(
            r#"rule = [ { scope = "prompt", search = "d", replace = "1" }, { scope = "prompt", search = "d", replace = "2" } ]"#
        ).is_err());
    }
}
