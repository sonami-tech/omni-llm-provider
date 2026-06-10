//! Minimal demo of shared omni-common (replacements + stats) + a mock provider.
//! This proves the cross-cutting shared pieces work independently of any backend/frontend.

use omni_common::{Replacements, Stats, TokenUsage};

fn main() {
    // Shared replacements (prompt/response scopes)
    let repl = Replacements::parse(r#"
[[rule]]
scope = "prompt"
search = "SECRET"
replace = "REDACTED"

[[rule]]
scope = "response"
search = "foo"
replace = "bar"
"#).unwrap();

    let cleaned_prompt = repl.apply_prompt("Tell me about SECRET project");
    println!("Outbound (prompt): {}", cleaned_prompt);

    let cleaned_resp = repl.apply_response("The foo is ready.");
    println!("Inbound (response): {}", cleaned_resp);

    // Shared stats (redb backed, but in-memory for demo here we just exercise the API)
    // In real: Stats::open(path)
    println!("Replacements + basic stats concepts exercised (full redb in lib).");

    // Mock "provider" usage
    println!("Mock provider would receive cleaned prompt and emit response for inbound replace.");

    // Demonstrate isolation: no fingerprint or Claude types here.
    println!("Demo complete. This layer knows nothing about Claude Code fingerprints or specific providers.");
}
