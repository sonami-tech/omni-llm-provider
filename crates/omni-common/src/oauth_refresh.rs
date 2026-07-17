//! Shared OAuth-refresh gate and cross-provider recovery coordination.
//!
//! ## Enable gate
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
//!
//! ## Recovery shape
//!
//! Providers run a count-bounded loop (see [`MAX_CREDENTIAL_RECOVERY_TURNS`]):
//! re-read disk → use AT if still good → else refresh under lock → always
//! re-read again. Refresh posts are serialized with an in-process single-flight
//! mutex per credential path plus a sibling `.lock` file flock (same naming as
//! Grok CLI's `auth.json.lock`) so Omni and vendor CLIs do not double-spend RTs.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use tokio::sync::Mutex as AsyncMutex;

/// Refresh when the access token expires within this many seconds (15 minutes).
pub const NEAR_EXPIRY_SKEW_SECS: i64 = 15 * 60;
/// Same skew in milliseconds for providers that store absolute ms expiry.
pub const NEAR_EXPIRY_SKEW_MS: i64 = NEAR_EXPIRY_SKEW_SECS * 1000;
/// Max full read → (optional refresh) → re-read turns per credential load.
pub const MAX_CREDENTIAL_RECOVERY_TURNS: u32 = 3;

/// True when an IdP error body indicates the refresh token was already spent
/// or revoked (peer may have written a fresh AT/RT to disk).
pub fn looks_like_refresh_token_spent(body: &str) -> bool {
    let b = body.to_ascii_lowercase();
    b.contains("refresh_token_reused")
        || b.contains("invalid_grant")
        || b.contains("refresh token has been revoked")
        || b.contains("token has been revoked")
        || b.contains("token_revoked")
}

/// Sibling lock path matching common CLI style: `auth.json` → `auth.json.lock`.
pub fn credential_lock_path(cred_path: &Path) -> PathBuf {
    let mut name = cred_path
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| std::ffi::OsString::from("credentials"));
    name.push(".lock");
    match cred_path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(name),
        _ => PathBuf::from(name),
    }
}

struct FlockGuard {
    file: File,
}

impl Drop for FlockGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

fn acquire_flock_blocking(lock_path: &Path) -> io::Result<FlockGuard> {
    if let Some(parent) = lock_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(not(unix))]
    {
        // Best-effort: exclusive create is not portable here; process mutex still applies.
        let _ = &file;
    }
    Ok(FlockGuard { file })
}

fn process_mutex_for(path: &Path) -> Arc<AsyncMutex<()>> {
    static MAP: OnceLock<StdMutex<HashMap<String, Arc<AsyncMutex<()>>>>> = OnceLock::new();
    let map = MAP.get_or_init(|| StdMutex::new(HashMap::new()));
    let key = path.to_string_lossy().into_owned();
    let mut guard = map.lock().unwrap_or_else(|p| p.into_inner());
    guard
        .entry(key)
        .or_insert_with(|| Arc::new(AsyncMutex::new(())))
        .clone()
}

/// Run `f` while holding the in-process single-flight mutex and path flock for
/// `cred_path`. Blocking flock is taken off the async worker via `spawn_blocking`.
pub async fn with_oauth_refresh_lock<F, Fut, T>(cred_path: &Path, f: F) -> io::Result<T>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    let process = process_mutex_for(cred_path);
    let _process_guard = process.lock().await;

    let lock_path = credential_lock_path(cred_path);
    let flock = tokio::task::spawn_blocking(move || acquire_flock_blocking(&lock_path))
        .await
        .map_err(io::Error::other)??;

    let out = f().await;
    drop(flock);
    Ok(out)
}

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
    parse_bool_env("OMNI_OAUTH_REFRESH").unwrap_or(true)
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
        assert_eq!(oauth_refresh_policy_summary(), "claude=on codex=on grok=on");
        env.set("OMNI_OAUTH_REFRESH", "0");
        env.set("OMNI_CODEX_OAUTH_REFRESH", "1");
        assert_eq!(
            oauth_refresh_policy_summary(),
            "claude=off codex=on grok=off"
        );
    }

    #[test]
    fn credential_lock_path_matches_cli_style() {
        // WHY: Grok uses auth.json.lock; we must share the same sibling name.
        let p = Path::new("/home/u/.grok/auth.json");
        assert_eq!(
            credential_lock_path(p),
            PathBuf::from("/home/u/.grok/auth.json.lock")
        );
        assert_eq!(
            credential_lock_path(Path::new("/h/.claude/.credentials.json")),
            PathBuf::from("/h/.claude/.credentials.json.lock")
        );
    }

    #[test]
    fn looks_like_refresh_token_spent_detects_known_idp_errors() {
        assert!(looks_like_refresh_token_spent("refresh_token_reused"));
        assert!(looks_like_refresh_token_spent(
            r#"{"error":"invalid_grant","error_description":"Refresh token has been revoked"}"#
        ));
        assert!(!looks_like_refresh_token_spent("rate_limit_exceeded"));
    }

    #[tokio::test]
    async fn with_oauth_refresh_lock_serializes_concurrent_callers() {
        // WHY: concurrent near-expiry loads must not all POST; single-flight is the fix.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dir = std::env::temp_dir().join(format!(
            "omni-oauth-lock-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cred = dir.join("auth.json");
        std::fs::write(&cred, b"{}").unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let path = cred.clone();
            let counter = Arc::clone(&counter);
            handles.push(tokio::spawn(async move {
                with_oauth_refresh_lock(&path, || async {
                    let n = counter.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    // While one holder runs, others must not enter (counter stays n until done).
                    assert_eq!(counter.load(Ordering::SeqCst), n + 1);
                })
                .await
                .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 8);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
