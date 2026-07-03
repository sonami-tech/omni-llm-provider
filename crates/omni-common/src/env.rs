//! Shared environment-variable and custom-header helpers.
//!
//! These are used by the binary and every provider to read optional
//! configuration from the environment. Keeping one copy avoids the drift that
//! four hand-maintained duplicates invite.

/// Read an environment variable, trim surrounding whitespace, and treat an
/// empty (or all-whitespace) value as absent. Returns `None` when the variable
/// is unset or blank.
pub fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
