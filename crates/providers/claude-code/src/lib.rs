//! providers/claude-code
//!
//! The Claude Code / Anthropic Max provider.
//!
//! **CRITICAL ISOLATION RULE (for this monorepo investigation):**
//! - All fingerprint, cch, credentials, identity injection, per-profile wire defaults,
//!   Anthropic Messages wire types, and re-baselining concerns live ONLY here.
//! - Nothing in omni-common or omni-core may depend on types from this crate.
//! - This crate may depend on omni-common (for stats, replacements, auth concepts)
//!   and omni-core (for Canonical* types).
//!
//! This is the "heavy" provider. Most of the original claude-code-provider logic
//! (upstream/fingerprint, credentials, translate, gate layers) would move here
//! or be re-implemented against the canonical types.

// Placeholder to demonstrate the boundary.
// In a real port we would have:
// - mod fingerprint; (the entire current fingerprint.rs + profiles)
// - mod credentials;
// - mod client;
// - mod translate_to_anthropic;
// - mod gate; // prepend_identity, apply_wire_defaults, finalize_cch
// - impl LlmProvider for ClaudeCodeProvider { ... }

pub struct ClaudeCodeProvider {
    // Will hold profile, upstream client (internal to this crate), replacements handle, etc.
}

impl ClaudeCodeProvider {
    pub fn new(/* config, profile selector, etc. */) -> Self {
        // ...
        Self {}
    }
}
