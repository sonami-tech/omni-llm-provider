//! Shared OAuth-refresh enable gate for Claude / Codex / Grok.
//!
//! Default is **on**. Operators can force on/off globally or per provider via
//! env (or the omni CLI, which writes the same env keys at startup):
//!
//! - Global: `OMNI_OAUTH_REFRESH=0|1`, `OMNI_NO_OAUTH_REFRESH=1` (force off)
//! - Claude: `OMNI_CLAUDE_OAUTH_REFRESH=0|1`
//! - Codex:  `OMNI_CODEX_OAUTH_REFRESH=0|1`
//! - Grok:   `OMNI_GROK_OAUTH_REFRESH=0|1`
//!
//! Resolution: provider env if set, else global, else on. Provider always beats
//! global (so `OMNI_OAUTH_REFRESH=0` + `OMNI_CODEX_OAUTH_REFRESH=1` refreshes
//! only Codex).

/// Which provider's refresh gate to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthRefreshProvider {
    Claude,
    Codex,
    Grok,
}

impl OAuthRefreshProvider {
    /// Env key that overrides the global gate for this provider.
    pub const fn env_key(self) -> &'static str {
        match self {
            Self::Claude => "OMNI_CLAUDE_OAUTH_REFRESH",
            Self::Codex => "OMNI_CODEX_OAUTH_REFRESH",
            Self::Grok => "OMNI_GROK_OAUTH_REFRESH",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Grok => "grok",
        }
    }
}

/// Whether in-process OAuth refresh is enabled for `provider`.
///
/// See module docs for precedence.
pub fn oauth_refresh_enabled_for(provider: OAuthRefreshProvider) -> bool {
    if let Some(v) = parse_bool_env(provider.env_key()) {
        return v;
    }
    global_oauth_refresh_enabled()
}

/// Global gate only (no per-provider override). Used by tests and diagnostics.
pub fn global_oauth_refresh_enabled() -> bool {
    if env_flag_truthy("OMNI_NO_OAUTH_REFRESH") {
        return false;
    }
    match parse_bool_env("OMNI_OAUTH_REFRESH") {
        None => true,
        Some(v) => v,
    }
}

/// One-line effective policy for startup logs: `claude=on codex=on grok=off`.
///
/// Always reflects resolved env (after CLI has written flags into the process).
pub fn oauth_refresh_policy_summary() -> String {
    fn slot(p: OAuthRefreshProvider) -> String {
        let state = if oauth_refresh_enabled_for(p) {
            "on"
        } else {
            "off"
        };
        format!("{}={state}", p.as_str())
    }
    format!(
        "{} {} {}",
        slot(OAuthRefreshProvider::Claude),
        slot(OAuthRefreshProvider::Codex),
        slot(OAuthRefreshProvider::Grok),
    )
}

/// Parse a tri-state bool env: unset → `None`, falsey → `Some(false)`, else `Some(true)`.
///
/// Falsey: `0` / `false` / `off` / `no` (case-insensitive).
/// Truthy: any other non-empty value (matches historical `OMNI_OAUTH_REFRESH` behavior).
pub fn parse_bool_env(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    let lower = raw.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }
    Some(!matches!(lower.as_str(), "0" | "false" | "off" | "no"))
}

fn env_flag_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        keys: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl EnvGuard {
        fn capture(keys: &[&'static str]) -> Self {
            let keys = keys
                .iter()
                .map(|&k| (k, std::env::var_os(k)))
                .collect::<Vec<_>>();
            for &(k, _) in &keys {
                unsafe {
                    std::env::remove_var(k);
                }
            }
            Self { keys }
        }

        fn set(&self, key: &str, value: &str) {
            unsafe {
                std::env::set_var(key, value);
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, old) in self.keys.drain(..) {
                unsafe {
                    match old {
                        Some(v) => std::env::set_var(k, v),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }
    }

    const ALL_KEYS: &[&str] = &[
        "OMNI_OAUTH_REFRESH",
        "OMNI_NO_OAUTH_REFRESH",
        "OMNI_CLAUDE_OAUTH_REFRESH",
        "OMNI_CODEX_OAUTH_REFRESH",
        "OMNI_GROK_OAUTH_REFRESH",
    ];

    #[test]
    fn default_is_on_for_all_providers() {
        // WHY: refresh is opt-out; a fresh process must refresh every provider.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _env = EnvGuard::capture(ALL_KEYS);
        assert!(oauth_refresh_enabled_for(OAuthRefreshProvider::Claude));
        assert!(oauth_refresh_enabled_for(OAuthRefreshProvider::Codex));
        assert!(oauth_refresh_enabled_for(OAuthRefreshProvider::Grok));
    }

    #[test]
    fn global_off_disables_all_without_provider_override() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let env = EnvGuard::capture(ALL_KEYS);
        env.set("OMNI_OAUTH_REFRESH", "0");
        assert!(!oauth_refresh_enabled_for(OAuthRefreshProvider::Claude));
        assert!(!oauth_refresh_enabled_for(OAuthRefreshProvider::Codex));
        assert!(!oauth_refresh_enabled_for(OAuthRefreshProvider::Grok));
    }

    #[test]
    fn no_oauth_refresh_alias_disables_global() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let env = EnvGuard::capture(ALL_KEYS);
        env.set("OMNI_NO_OAUTH_REFRESH", "1");
        assert!(!global_oauth_refresh_enabled());
        assert!(!oauth_refresh_enabled_for(OAuthRefreshProvider::Codex));
    }

    #[test]
    fn provider_on_beats_global_off() {
        // WHY: "only codex refreshes" = global off + codex on.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let env = EnvGuard::capture(ALL_KEYS);
        env.set("OMNI_OAUTH_REFRESH", "0");
        env.set("OMNI_CODEX_OAUTH_REFRESH", "1");
        assert!(!oauth_refresh_enabled_for(OAuthRefreshProvider::Claude));
        assert!(oauth_refresh_enabled_for(OAuthRefreshProvider::Codex));
        assert!(!oauth_refresh_enabled_for(OAuthRefreshProvider::Grok));
    }

    #[test]
    fn provider_off_beats_global_on() {
        // WHY: "all but codex" = default on + codex off.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let env = EnvGuard::capture(ALL_KEYS);
        env.set("OMNI_CODEX_OAUTH_REFRESH", "0");
        assert!(oauth_refresh_enabled_for(OAuthRefreshProvider::Claude));
        assert!(!oauth_refresh_enabled_for(OAuthRefreshProvider::Codex));
        assert!(oauth_refresh_enabled_for(OAuthRefreshProvider::Grok));
    }

    #[test]
    fn parse_bool_env_falsey_and_truthy() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let env = EnvGuard::capture(&["OMNI_TEST_BOOL"]);
        assert_eq!(parse_bool_env("OMNI_TEST_BOOL"), None);
        env.set("OMNI_TEST_BOOL", "0");
        assert_eq!(parse_bool_env("OMNI_TEST_BOOL"), Some(false));
        env.set("OMNI_TEST_BOOL", "false");
        assert_eq!(parse_bool_env("OMNI_TEST_BOOL"), Some(false));
        env.set("OMNI_TEST_BOOL", "1");
        assert_eq!(parse_bool_env("OMNI_TEST_BOOL"), Some(true));
        env.set("OMNI_TEST_BOOL", "yes");
        assert_eq!(parse_bool_env("OMNI_TEST_BOOL"), Some(true));
    }

    #[test]
    fn policy_summary_reflects_effective_state() {
        // WHY: startup must show resolved on/off, not only which flags were passed.
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let env = EnvGuard::capture(ALL_KEYS);
        assert_eq!(
            oauth_refresh_policy_summary(),
            "claude=on codex=on grok=on"
        );
        env.set("OMNI_OAUTH_REFRESH", "0");
        env.set("OMNI_CODEX_OAUTH_REFRESH", "1");
        assert_eq!(
            oauth_refresh_policy_summary(),
            "claude=off codex=on grok=off"
        );
    }
}
