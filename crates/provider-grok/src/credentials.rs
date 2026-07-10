//! Grok / xAI credentials loading, modeled exactly after the Claude Code Provider technique.
//!
//! Locked design (same as CCP): never cache. Always re-read per request.
//! This picks up any background refresh or key rotation the user (or Omni write-back)
//! may have performed (e.g. the Grok CLI refreshing its login, or an updated console key).
//!
//! ## Sources and precedence (highest first), all read fresh per request:
//! 1. `$XAI_CREDENTIALS_PATH` (explicit override) — parsed as either shape below.
//! 2. `~/.xai/.credentials.json` — a simple static-key file (deliberate console key).
//! 3. `~/.grok/auth.json` — the Grok CLI's own login file (OIDC). This is the primary
//!    auto-detect path: just as omni reads `~/.claude/.credentials.json` (the Claude
//!    CLI's file), it reads the Grok CLI's file so an existing `grok` login Just Works.
//!
//! Explicit beats ambient: a usable static-key file you created on purpose wins over
//! the CLI's auto-managed OIDC login. If the ambient static-key file exists but has
//! no usable key, we fall through to the CLI login instead of letting a stale file
//! break an otherwise valid `grok` login.
//!
//! ## On-disk shapes
//! Static key (`~/.xai/.credentials.json` or `$XAI_CREDENTIALS_PATH`):
//! ```json
//! { "apiKey": "xai-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" }
//! ```
//! (a top-level `xaiApiKey` alias is also tolerated.) Static keys are never refreshed.
//!
//! Grok CLI OIDC (`~/.grok/auth.json`), keyed by `https://auth.x.ai::<client-id>`:
//! ```json
//! { "https://auth.x.ai::<id>": { "key": "<JWT>", "auth_mode": "oidc",
//!     "refresh_token": "...", "expires_at": "2026-06-10T22:20:22.000000Z" } }
//! ```
//! The `key` JWT is a Bearer that authenticates `api.x.ai/v1` directly.
//!
//! When `OMNI_OAUTH_REFRESH` is truthy, Omni may proactively refresh a near-expired
//! OIDC access token via `POST https://auth.x.ai/oauth2/token` (form-urlencoded,
//! public client) and **atomically write back** the rotated `refresh_token` to the
//! same path. RTs rotate with a short grace window then revoke — never leave the old
//! RT on disk after a successful grant. When the flag is off (default), Omni only
//! re-reads the file and surfaces expiry (CLI still owns refresh).

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use tracing::warn;

/// Captured Grok OIDC token endpoint (grok-shell 0.2.93).
pub const GROK_OAUTH_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
/// User-Agent template prefix; full UA is `grok-shell/<ver> (linux; x86_64)`.
pub const GROK_OAUTH_USER_AGENT: &str = "grok-shell/0.2.93 (linux; x86_64)";

/// Refresh when access token expires within this many milliseconds (5 minutes).
const NEAR_EXPIRY_SKEW_MS: i64 = 5 * 60 * 1000;

// Local error for credentials (kept small; mapped by callers to ProviderError or AppError).
#[derive(Debug, thiserror::Error)]
pub enum GrokCredentialsError {
    #[error("read: {0}")]
    Read(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(#[from] serde_json::Error),
    #[error(
        "no Grok credentials found (tried $XAI_CREDENTIALS_PATH, ~/.xai/.credentials.json, ~/.grok/auth.json)"
    )]
    NoSource,
    #[error("credentials file present but held no usable key")]
    MissingToken,
    #[error(
        "token expired (enable OMNI_OAUTH_REFRESH=1 for in-process refresh, or re-login with the Grok CLI)"
    )]
    Expired,
    #[error("oauth refresh: {0}")]
    Refresh(String),
}

/// On-disk shape of the simple static-key file.
#[derive(Debug, Clone, Deserialize)]
struct StaticKeyFile {
    #[serde(rename = "apiKey")]
    api_key: Option<String>,
    #[serde(rename = "xaiApiKey")]
    xai_api_key: Option<String>,
}

/// On-disk shape of one entry inside the Grok CLI's `~/.grok/auth.json`. The file is a
/// JSON object whose single value (keyed by `https://auth.x.ai::<client-id>`) has this shape.
#[derive(Debug, Clone, Deserialize)]
struct GrokCliEntry {
    /// The JWT access token used as the Bearer.
    key: Option<String>,
    /// "oidc" for the CLI's OIDC login.
    auth_mode: Option<String>,
    /// ISO 8601 expiry of the access token (e.g. "2026-06-10T22:20:22.000000Z").
    expires_at: Option<String>,
    /// The OIDC subject (a uuid) the Grok CLI sends as `x-grok-user-id` in
    /// conservative mode. Absent on static-key files (None there).
    user_id: Option<String>,
    /// Rotating refresh token (grace-then-revoke; must be persisted after use).
    refresh_token: Option<String>,
    /// OIDC client id (also appears in the object key).
    oidc_client_id: Option<String>,
    /// Principal id for the form grant (`principal_type=User`).
    principal_id: Option<String>,
}

/// Material needed to POST a refresh_token grant for one OIDC entry.
#[derive(Debug, Clone)]
pub struct GrokOidcRefreshMaterial {
    pub entry_key: String,
    pub refresh_token: String,
    pub client_id: String,
    pub principal_id: String,
}

/// Parsed OIDC token grant response.
#[derive(Debug, Clone, Deserialize)]
pub struct GrokTokenGrant {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_in: Option<i64>,
}

/// In-memory parsed Grok credentials.
#[derive(Debug, Clone)]
pub struct GrokCredentials {
    pub api_key: String,
    /// Access-token expiry in Unix epoch milliseconds. `None` for static API keys
    /// (which do not expire) or when the OIDC entry omits/!parses `expires_at`.
    pub expires_at_ms: Option<i64>,
    /// The OIDC `user_id` (a uuid) used as `x-grok-user-id` in conservative mode.
    /// `None` for static API keys (the CLI omits the header when unavailable).
    pub user_id: Option<String>,
}

impl GrokCredentials {
    /// The static-key default location: `~/.xai/.credentials.json`. Override via
    /// `$XAI_CREDENTIALS_PATH`. Mirrors the CCP `default_path` + env-override logic.
    pub fn default_path() -> PathBuf {
        if let Some(p) = std::env::var_os("XAI_CREDENTIALS_PATH") {
            return PathBuf::from(p);
        }
        match dirs::home_dir() {
            Some(home) => home.join(".xai").join(".credentials.json"),
            None => PathBuf::from(".xai/.credentials.json"),
        }
    }

    /// The Grok CLI's own login file: `~/.grok/auth.json`. This is the file the
    /// `grok` CLI writes on login; omni reads it so an existing login Just Works,
    /// exactly as it reads `~/.claude/.credentials.json` for Claude.
    pub fn grok_cli_path() -> Option<PathBuf> {
        dirs::home_dir().map(|home| home.join(".grok").join("auth.json"))
    }

    /// Resolve credentials fresh from disk (never cached). The operator provides creds in one of
    /// three places, checked in order:
    /// 1. `$XAI_CREDENTIALS_PATH` — an explicit file path the operator chose. If set, it is the
    ///    only source; a failure to load it surfaces directly.
    /// 2. `~/.xai/.credentials.json` — a static-key file.
    /// 3. `~/.grok/auth.json` — the Grok CLI's own login file (auto-detected).
    pub async fn load_resolved_async() -> Result<Self, GrokCredentialsError> {
        if let Some(p) = std::env::var_os("XAI_CREDENTIALS_PATH") {
            return Self::load_fresh_async(Path::new(&p)).await;
        }

        let mut ambient_static_error = None;
        if let Some(home) = dirs::home_dir() {
            let static_path = home.join(".xai").join(".credentials.json");
            if tokio::fs::try_exists(&static_path).await.unwrap_or(false) {
                match Self::load_fresh_async(&static_path).await {
                    Ok(creds) => return Ok(creds),
                    Err(GrokCredentialsError::MissingToken) => {
                        ambient_static_error = Some(GrokCredentialsError::MissingToken);
                    }
                    Err(err) => return Err(err),
                }
            }
        }

        if let Some(cli_path) = Self::grok_cli_path()
            && tokio::fs::try_exists(&cli_path).await.unwrap_or(false)
        {
            return Self::load_fresh_async(&cli_path).await;
        }

        Err(ambient_static_error.unwrap_or(GrokCredentialsError::NoSource))
    }

    /// Resolve credentials for CONSERVATIVE (grok-shell CLI parity) mode.
    ///
    /// Differs from [`load_resolved_async`] in precedence ONLY: it prefers the Grok
    /// CLI OIDC login (`~/.grok/auth.json`) over the static-key file. The
    /// conservative wire sends `x-grok-user-id`, which is carried ONLY by the OIDC
    /// entry; a static key has no subject, so preferring it would drift from the
    /// grok-shell fingerprint (Authorization with no matching user_id). The chat
    /// path keeps its documented "explicit static beats ambient OIDC" rule via
    /// [`load_resolved_async`]; this is a conservative-specific override and does
    /// NOT change that.
    ///
    /// Order (highest first), all read fresh:
    /// 1. `$XAI_CREDENTIALS_PATH` — explicit operator override, still wins (a
    ///    deliberate choice; may be either shape).
    /// 2. `~/.grok/auth.json` — the Grok CLI OIDC login (carries `user_id`).
    /// 3. `~/.xai/.credentials.json` — static-key fallback (no `user_id`), only
    ///    when the CLI login is absent/unusable.
    pub async fn load_resolved_conservative_async() -> Result<Self, GrokCredentialsError> {
        if let Some(p) = std::env::var_os("XAI_CREDENTIALS_PATH") {
            return Self::load_fresh_async(Path::new(&p)).await;
        }

        let mut ambient_cli_error = None;
        if let Some(cli_path) = Self::grok_cli_path()
            && tokio::fs::try_exists(&cli_path).await.unwrap_or(false)
        {
            match Self::load_fresh_async(&cli_path).await {
                Ok(creds) => return Ok(creds),
                // A present-but-unusable CLI login (no usable key) falls through to
                // the static-key file rather than breaking an otherwise valid setup.
                Err(GrokCredentialsError::MissingToken) => {
                    ambient_cli_error = Some(GrokCredentialsError::MissingToken);
                }
                Err(err) => return Err(err),
            }
        }

        if let Some(home) = dirs::home_dir() {
            let static_path = home.join(".xai").join(".credentials.json");
            if tokio::fs::try_exists(&static_path).await.unwrap_or(false) {
                return Self::load_fresh_async(&static_path).await;
            }
        }

        Err(ambient_cli_error.unwrap_or(GrokCredentialsError::NoSource))
    }

    /// Read and parse a specific credentials file synchronously. Never cached.
    /// Parses either supported shape (static-key or Grok CLI OIDC). Use outside the
    /// async hot path (e.g. startup); handlers should prefer [`load_resolved_async`].
    pub fn load_fresh(path: &Path) -> Result<Self, GrokCredentialsError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Async read+parse of a specific file, not blocking the Tokio worker. Never cached.
    /// When `OMNI_OAUTH_REFRESH` is on and this path holds a near-expired OIDC entry,
    /// refreshes in-place then re-reads.
    pub async fn load_fresh_async(path: &Path) -> Result<Self, GrokCredentialsError> {
        Self::load_fresh_async_with_refresh(path, false).await
    }

    /// Force OAuth refresh (if enabled) then re-read — for 401-once paths.
    pub async fn load_fresh_async_force_refresh(
        path: &Path,
    ) -> Result<Self, GrokCredentialsError> {
        Self::load_fresh_async_with_refresh(path, true).await
    }

    async fn load_fresh_async_with_refresh(
        path: &Path,
        force: bool,
    ) -> Result<Self, GrokCredentialsError> {
        let bytes = tokio::fs::read(path).await?;
        let creds = Self::from_bytes(&bytes)?;

        if !oauth_refresh_enabled() {
            return Ok(creds);
        }
        // Static keys never refresh.
        if creds.expires_at_ms.is_none() && !force {
            return Ok(creds);
        }
        let should = force || needs_refresh(creds.expires_at_ms);
        if !should {
            return Ok(creds);
        }
        // Only OIDC files carry refresh material.
        if extract_oidc_refresh_material(&bytes).is_err() {
            return Ok(creds);
        }

        match refresh_oauth_inplace(path, &oauth_token_url()).await {
            Ok(()) => {
                let bytes = tokio::fs::read(path).await?;
                Self::from_bytes(&bytes)
            }
            Err(e) => {
                if creds.check_expired().is_ok() {
                    warn!(error = %e, "grok OAuth refresh failed; using existing access token");
                    Ok(creds)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Parse either credential shape from raw bytes: the Grok CLI OIDC file (an object whose entry
    /// carries `auth_mode:"oidc"` + a `key` JWT) or the simple static-key file (`{"apiKey": "..."}`).
    /// The static shape has no `auth_mode:"oidc"` entry, so the two are distinguishable.
    fn from_bytes(bytes: &[u8]) -> Result<Self, GrokCredentialsError> {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;

        if let Some(creds) = Self::try_grok_cli(&value) {
            return Ok(creds);
        }

        // Fall back to the static-key shape.
        let parsed: StaticKeyFile = serde_json::from_value(value)?;
        let key = parsed
            .api_key
            .or(parsed.xai_api_key)
            .filter(|k| !k.trim().is_empty())
            .ok_or(GrokCredentialsError::MissingToken)?;
        Ok(GrokCredentials {
            api_key: key,
            expires_at_ms: None,
            user_id: None,
        })
    }

    /// Recognize the Grok CLI's `~/.grok/auth.json` shape: a JSON object whose entry carries
    /// `auth_mode:"oidc"` and a non-empty `key` (the JWT). Reads the JWT and its expiry.
    fn try_grok_cli(value: &serde_json::Value) -> Option<GrokCredentials> {
        let obj = value.as_object()?;
        for entry_val in obj.values() {
            let entry: GrokCliEntry = match serde_json::from_value(entry_val.clone()) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let is_oidc = entry.auth_mode.as_deref() == Some("oidc");
            let key = entry.key.filter(|k| !k.trim().is_empty());
            if let (true, Some(key)) = (is_oidc, key) {
                return Some(GrokCredentials {
                    api_key: key,
                    expires_at_ms: entry.expires_at.as_deref().and_then(parse_iso8601_to_ms),
                    user_id: entry.user_id.filter(|id| !id.trim().is_empty()),
                });
            }
        }
        None
    }

    /// Surface a clear error if an OIDC access token is past its expiry. Static keys
    /// (no `expires_at`) are always OK. When `OMNI_OAUTH_REFRESH` is enabled, callers
    /// should attempt refresh before treating this as terminal.
    pub fn check_expired(&self) -> Result<(), GrokCredentialsError> {
        if let Some(exp) = self.expires_at_ms {
            let now_ms = chrono::Utc::now().timestamp_millis();
            if now_ms >= exp {
                return Err(GrokCredentialsError::Expired);
            }
        }
        Ok(())
    }
}

/// Parse an ISO 8601 / RFC 3339 timestamp (e.g. "2026-06-10T22:20:22.000000Z") to Unix
/// epoch milliseconds. Returns None on any parse failure so a malformed `expires_at`
/// degrades to "no known expiry" rather than erroring the whole credential load.
fn parse_iso8601_to_ms(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

/// Opt-in gate for in-Omni OAuth refresh (shared name across providers).
pub fn oauth_refresh_enabled() -> bool {
    matches!(
        std::env::var("OMNI_OAUTH_REFRESH")
            .ok()
            .map(|v| v.to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Effective Grok token endpoint. Tests may set `OMNI_GROK_OAUTH_TOKEN_URL`.
pub fn oauth_token_url() -> String {
    std::env::var("OMNI_GROK_OAUTH_TOKEN_URL").unwrap_or_else(|_| GROK_OAUTH_TOKEN_URL.to_string())
}

pub fn needs_refresh(expires_at_ms: Option<i64>) -> bool {
    match expires_at_ms {
        None => false,
        Some(exp) => {
            let now_ms = chrono::Utc::now().timestamp_millis();
            now_ms + NEAR_EXPIRY_SKEW_MS >= exp
        }
    }
}

/// Build form-urlencoded body for the Grok refresh_token grant (pure).
pub fn build_refresh_form_body(material: &GrokOidcRefreshMaterial) -> String {
    form_encode(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", &material.refresh_token),
        ("client_id", &material.client_id),
        ("principal_type", "User"),
        ("principal_id", &material.principal_id),
    ])
}

/// application/x-www-form-urlencoded encoding (no extra deps).
fn form_encode(pairs: &[(&str, &str)]) -> String {
    let mut out = String::new();
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push('&');
        }
        out.push_str(&form_quote(k));
        out.push('=');
        out.push_str(&form_quote(v));
    }
    out
}

fn form_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push_str(&format!("{b:02X}"));
            }
        }
    }
    out
}

/// Apply grant onto the OIDC entry inside auth.json (pure).
pub fn apply_grant_to_auth_json(
    file: &mut Value,
    entry_key: &str,
    grant: &GrokTokenGrant,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<(), GrokCredentialsError> {
    let entry = file
        .get_mut(entry_key)
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| {
            GrokCredentialsError::Refresh(format!("auth.json missing entry {entry_key}"))
        })?;

    entry.insert("key".into(), Value::String(grant.access_token.clone()));
    if let Some(rt) = grant.refresh_token.as_ref().filter(|s| !s.is_empty()) {
        entry.insert("refresh_token".into(), Value::String(rt.clone()));
    }
    if let Some(expires_in) = grant.expires_in {
        let expires_at = now + chrono::Duration::seconds(expires_in);
        // CLI writes ISO-8601 with fractional seconds and Z.
        let formatted = expires_at.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string();
        entry.insert("expires_at".into(), Value::String(formatted));
    }
    Ok(())
}

/// Atomically replace `path` with `bytes` via temp file + rename.
/// Preserves the prior file mode when possible so RTs stay non-world-readable.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), GrokCredentialsError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(
        ".{}.omni-oauth-{}-{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("auth.json"),
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&tmp, bytes)?;
    if let Ok(meta) = std::fs::metadata(path) {
        let _ = std::fs::set_permissions(&tmp, meta.permissions());
    } else {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
        }
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

/// Pull refresh material from a Grok CLI auth.json document.
pub fn extract_oidc_refresh_material(
    bytes: &[u8],
) -> Result<GrokOidcRefreshMaterial, GrokCredentialsError> {
    let value: Value = serde_json::from_slice(bytes)?;
    let obj = value
        .as_object()
        .ok_or(GrokCredentialsError::MissingToken)?;
    for (entry_key, entry_val) in obj {
        let entry: GrokCliEntry = match serde_json::from_value(entry_val.clone()) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.auth_mode.as_deref() != Some("oidc") {
            continue;
        }
        let refresh_token = entry
            .refresh_token
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                GrokCredentialsError::Refresh(
                    "OIDC entry has no refresh_token; re-login with the Grok CLI".into(),
                )
            })?;
        // client_id: prefer field, else parse from key `https://auth.x.ai::<id>`.
        let client_id = entry
            .oidc_client_id
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                entry_key
                    .rsplit("::")
                    .next()
                    .map(|s| s.to_string())
                    .filter(|s| !s.is_empty())
            })
            .ok_or_else(|| {
                GrokCredentialsError::Refresh("OIDC entry missing client_id".into())
            })?;
        let principal_id = entry
            .principal_id
            .filter(|s| !s.trim().is_empty())
            .or_else(|| entry.user_id.filter(|s| !s.trim().is_empty()))
            .ok_or_else(|| {
                GrokCredentialsError::Refresh("OIDC entry missing principal_id/user_id".into())
            })?;
        return Ok(GrokOidcRefreshMaterial {
            entry_key: entry_key.clone(),
            refresh_token,
            client_id,
            principal_id,
        });
    }
    Err(GrokCredentialsError::MissingToken)
}

/// POST the form grant and atomically write back to `path`.
pub async fn refresh_oauth_inplace(
    path: &Path,
    token_url: &str,
) -> Result<(), GrokCredentialsError> {
    let bytes = tokio::fs::read(path).await?;
    let material = extract_oidc_refresh_material(&bytes)?;
    let form = build_refresh_form_body(&material);

    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| GrokCredentialsError::Refresh(e.to_string()))?;

    let resp = client
        .post(token_url)
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .header(reqwest::header::USER_AGENT, GROK_OAUTH_USER_AGENT)
        .body(form)
        .send()
        .await
        .map_err(|e| GrokCredentialsError::Refresh(e.to_string()))?;

    let status = resp.status();
    let resp_bytes = resp
        .bytes()
        .await
        .map_err(|e| GrokCredentialsError::Refresh(e.to_string()))?;
    if !status.is_success() {
        let body_txt = String::from_utf8_lossy(&resp_bytes);
        return Err(GrokCredentialsError::Refresh(format!(
            "HTTP {status}: {body_txt}"
        )));
    }

    let grant: GrokTokenGrant = serde_json::from_slice(&resp_bytes).map_err(|e| {
        GrokCredentialsError::Refresh(format!("token response parse: {e}"))
    })?;
    if grant.access_token.is_empty() {
        return Err(GrokCredentialsError::Refresh(
            "token response missing access_token".into(),
        ));
    }
    let new_rt = grant
        .refresh_token
        .as_ref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            GrokCredentialsError::Refresh(
                "token response missing refresh_token; refusing write-back that would strand RT"
                    .into(),
            )
        })?;

    // CAS against concurrent writers: re-read and refuse to clobber a different RT.
    let latest_bytes = tokio::fs::read(path).await?;
    let mut latest: Value = serde_json::from_slice(&latest_bytes)?;
    let disk_rt = latest
        .get(&material.entry_key)
        .and_then(|e| e.get("refresh_token"))
        .and_then(|v| v.as_str());
    match disk_rt {
        Some(rt) if rt == material.refresh_token || rt == new_rt => {}
        Some(_) => {
            return Err(GrokCredentialsError::Refresh(
                "auth.json refresh_token changed during OAuth refresh (concurrent writer); not clobbering"
                    .into(),
            ));
        }
        None => {}
    }

    let now = chrono::Utc::now();
    apply_grant_to_auth_json(&mut latest, &material.entry_key, &grant, now)?;
    let out = serde_json::to_vec_pretty(&latest)
        .map_err(|e| GrokCredentialsError::Refresh(format!("serialize auth.json: {e}")))?;
    atomic_write(path, &out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use serde_json::json;

    use crate::GROK_ENV_LOCK as ENV_LOCK;

    fn temp_credentials_path() -> PathBuf {
        std::env::temp_dir().join(format!("omni-creds-test-{}.json", uuid::Uuid::new_v4()))
    }

    fn write_temp_creds(path: &Path, content: &str) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
    }

    #[test]
    fn load_fresh_file_not_exist_returns_read_error() {
        let p = temp_credentials_path(); // ensure absent
        let _ = std::fs::remove_file(&p);
        let res = GrokCredentials::load_fresh(&p);
        assert!(res.is_err());
        match res.unwrap_err() {
            GrokCredentialsError::Read(_) => {}
            other => panic!("expected Read, got {other:?}"),
        }
    }

    #[test]
    fn load_fresh_bad_json_returns_parse_error() {
        let p = temp_credentials_path();
        write_temp_creds(&p, "{ not json }");
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(res.is_err());
        match res.unwrap_err() {
            GrokCredentialsError::Parse(_) => {}
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn load_fresh_missing_key_returns_missing_token() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"foo": "bar"}"#);
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(
            res.unwrap_err(),
            GrokCredentialsError::MissingToken
        ));
    }

    #[test]
    fn default_path_respects_env_override_xai_credentials_path() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let p = temp_credentials_path();
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        // SAFETY: test-only mutation of process env for override verification; serialized by ENV_LOCK.
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", p.to_str().unwrap());
        }
        let got = GrokCredentials::default_path();
        // restore
        unsafe {
            if let Some(v) = old {
                std::env::set_var("XAI_CREDENTIALS_PATH", v);
            } else {
                std::env::remove_var("XAI_CREDENTIALS_PATH");
            }
        }
        assert_eq!(got, p);
    }

    #[tokio::test]
    async fn load_fresh_async_works() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "xai-test-123"}"#);
        let c = GrokCredentials::load_fresh_async(&p).await.unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "xai-test-123");
    }

    #[test]
    fn check_expired_is_noop_for_api_keys() {
        let c = GrokCredentials {
            api_key: "xai-foo".into(),
            expires_at_ms: None,
            user_id: None,
        };
        assert!(c.check_expired().is_ok());
    }

    #[test]
    fn parse_accepts_xai_api_key_alias() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"xaiApiKey": "xai-alias-456"}"#);
        let c = GrokCredentials::load_fresh(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "xai-alias-456");
    }

    // Full coverage per spec: file not exist read err (already basic), ensure exact err variant + temp fixture.
    #[test]
    fn file_not_exist_read_err() {
        let p = temp_credentials_path();
        let _ = std::fs::remove_file(&p);
        let res = GrokCredentials::load_fresh(&p);
        assert!(res.is_err());
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::Read(_)));
    }

    // Bad json parse err using temp fixture.
    #[test]
    fn bad_json_parse_err() {
        let p = temp_credentials_path();
        write_temp_creds(&p, "this is not { json at all");
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::Parse(_)));
    }

    // Missing key (neither apiKey nor alias) -> MissingToken.
    #[test]
    fn missing_key_missing_token() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"other": "stuff"}"#);
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(
            res.unwrap_err(),
            GrokCredentialsError::MissingToken
        ));
    }

    // Env override XAI_CREDENTIALS_PATH (already tested; re-exercise with async load too).
    #[test]
    fn env_override_xai_credentials_path() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "env-override-key"}"#);
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", p.to_str().unwrap());
        }
        let got = GrokCredentials::default_path();
        let c = GrokCredentials::load_fresh(&got).unwrap();
        unsafe {
            if let Some(v) = old {
                std::env::set_var("XAI_CREDENTIALS_PATH", v);
            } else {
                std::env::remove_var("XAI_CREDENTIALS_PATH");
            }
        }
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "env-override-key");
    }

    // Async load + check_expired noop for api keys + parse alias already; add temp fixture async variant.
    #[tokio::test]
    async fn async_load_and_check_expired_noop() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "async-xai-999"}"#);
        let c = GrokCredentials::load_fresh_async(&p).await.unwrap();
        assert!(c.check_expired().is_ok());
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "async-xai-999");
    }

    // Additional: empty value for key is treated as missing (after trim).
    #[test]
    fn empty_key_value_is_missing_token() {
        let p = temp_credentials_path();
        write_temp_creds(&p, r#"{"apiKey": "   "} "#);
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(
            res.unwrap_err(),
            GrokCredentialsError::MissingToken
        ));
    }

    // ---- Grok CLI OIDC file (~/.grok/auth.json) ----

    /// A minimal but realistic ~/.grok/auth.json: object keyed by the auth URL+client,
    /// value carries the JWT `key`, `auth_mode:"oidc"`, rotating `refresh_token`,
    /// and an ISO 8601 `expires_at`.
    fn grok_cli_json(key: &str, expires_at: &str) -> String {
        format!(
            r#"{{ "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {{
                "key": "{key}",
                "auth_mode": "oidc",
                "refresh_token": "rt-rotating-grace",
                "oidc_client_id": "b1a00492-073a-47ea-816f-4c329264a828",
                "principal_id": "11111111-2222-3333-4444-555555555555",
                "expires_at": "{expires_at}",
                "user_id": "11111111-2222-3333-4444-555555555555"
            }} }}"#
        )
    }

    #[test]
    fn parses_grok_cli_oidc_file_with_expiry() {
        // WHY: this is the Grok CLI's real on-disk shape. omni must read the JWT from it
        // (the whole point of auto-detecting an existing `grok` login) and capture the
        // expiry so check_expired can later warn. A future-dated token must be Ok.
        let p = temp_credentials_path();
        write_temp_creds(
            &p,
            &grok_cli_json("jwt-abc.def.ghi", "2999-01-01T00:00:00Z"),
        );
        let c = GrokCredentials::load_fresh(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "jwt-abc.def.ghi");
        assert!(c.expires_at_ms.is_some(), "OIDC expiry must be captured");
        assert!(c.check_expired().is_ok(), "future-dated token must be Ok");
        assert_eq!(
            c.user_id.as_deref(),
            Some("11111111-2222-3333-4444-555555555555"),
            "OIDC user_id must be captured for x-grok-user-id"
        );
    }

    #[test]
    fn oidc_captures_user_id_static_key_omits_it() {
        // WHY: conservative mode sends `x-grok-user-id: <user_id>`, sourced ONLY
        // from the OIDC file's `user_id`. A static-key file has no subject, so the
        // header (and thus user_id) must be None there - the CLI omits the header
        // when the id is unavailable, and we must mirror that exactly.
        let oidc = temp_credentials_path();
        write_temp_creds(&oidc, &grok_cli_json("jwt-uid", "2999-01-01T00:00:00Z"));
        let c = GrokCredentials::load_fresh(&oidc).unwrap();
        let _ = std::fs::remove_file(&oidc);
        assert_eq!(
            c.user_id.as_deref(),
            Some("11111111-2222-3333-4444-555555555555")
        );

        let static_p = temp_credentials_path();
        write_temp_creds(&static_p, r#"{"apiKey": "xai-static-no-uid"}"#);
        let s = GrokCredentials::load_fresh(&static_p).unwrap();
        let _ = std::fs::remove_file(&static_p);
        assert!(
            s.user_id.is_none(),
            "static-key file has no OIDC subject; user_id must be None"
        );
    }

    #[test]
    fn grok_cli_oidc_past_expiry_reports_expired() {
        // WHY: an expired OIDC token must surface as Err from check_expired so the provider
        // warns the user to re-run the Grok CLI login. The token value is still returned
        // (callers warn-but-continue and let the upstream make the final call).
        let p = temp_credentials_path();
        write_temp_creds(&p, &grok_cli_json("jwt-old", "2000-01-01T00:00:00Z"));
        let c = GrokCredentials::load_fresh(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "jwt-old");
        assert!(matches!(
            c.check_expired(),
            Err(GrokCredentialsError::Expired)
        ));
    }

    #[test]
    fn grok_cli_entry_without_oidc_mode_is_not_accepted() {
        // WHY: we only treat an entry marked auth_mode:"oidc" as the CLI login. An entry with a
        // different mode is not a credential we understand; since the file is also not a static
        // -key shape, the load yields MissingToken rather than silently using the stray key.
        let p = temp_credentials_path();
        write_temp_creds(
            &p,
            r#"{ "https://auth.x.ai::x": { "key": "jwt-x", "auth_mode": "password" } }"#,
        );
        let res = GrokCredentials::load_fresh(&p);
        let _ = std::fs::remove_file(&p);
        assert!(matches!(
            res.unwrap_err(),
            GrokCredentialsError::MissingToken
        ));
    }

    #[test]
    fn from_bytes_accepts_either_shape_via_explicit_path() {
        // WHY: $XAI_CREDENTIALS_PATH may point at EITHER file shape. Prove load_fresh parses
        // both, so the explicit override is not locked to one format.
        let static_p = temp_credentials_path();
        write_temp_creds(&static_p, r#"{"apiKey": "xai-static"}"#);
        let s = GrokCredentials::load_fresh(&static_p).unwrap();
        let _ = std::fs::remove_file(&static_p);
        assert_eq!(s.api_key, "xai-static");
        assert!(s.expires_at_ms.is_none());

        let oidc_p = temp_credentials_path();
        write_temp_creds(&oidc_p, &grok_cli_json("jwt-oidc", "2999-01-01T00:00:00Z"));
        let o = GrokCredentials::load_fresh(&oidc_p).unwrap();
        let _ = std::fs::remove_file(&oidc_p);
        assert_eq!(o.api_key, "jwt-oidc");
        assert!(o.expires_at_ms.is_some());
    }

    #[test]
    fn malformed_expires_at_degrades_to_no_expiry() {
        // WHY: a garbled expires_at must not error the whole load; it degrades to "no known
        // expiry" (the token is still usable; the upstream will 401 if truly dead).
        let p = temp_credentials_path();
        write_temp_creds(&p, &grok_cli_json("jwt-bad-exp", "not-a-timestamp"));
        let c = GrokCredentials::load_fresh(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(c.api_key, "jwt-bad-exp");
        assert!(c.expires_at_ms.is_none());
        assert!(c.check_expired().is_ok());
    }

    // ---- precedence chain (load_resolved_async) ----

    #[tokio::test]
    // ENV_LOCK is held across the load_resolved_async().await because XAI_CREDENTIALS_PATH
    // must stay set while the loader reads it. Safe: #[tokio::test] uses a current-thread
    // runtime, so the task never migrates threads while the std Mutex guard is held.
    #[allow(clippy::await_holding_lock)]
    async fn resolved_honors_explicit_env_override_first() {
        // WHY: $XAI_CREDENTIALS_PATH is the highest-precedence source. When set, it is the only
        // source consulted, so an OIDC file there resolves even if it is the explicit path.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let p = temp_credentials_path();
        write_temp_creds(&p, &grok_cli_json("jwt-via-env", "2999-01-01T00:00:00Z"));
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", p.to_str().unwrap());
        }
        let res = GrokCredentials::load_resolved_async().await;
        unsafe {
            match old {
                Some(v) => std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
        }
        let _ = std::fs::remove_file(&p);
        let c = res.expect("env-pointed creds must resolve");
        assert_eq!(c.api_key, "jwt-via-env");
    }

    #[tokio::test]
    // See resolved_honors_explicit_env_override_first: env lock held across await is safe
    // on the current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn resolved_env_override_pointing_at_missing_file_errors_not_falls_through() {
        // WHY: an explicit override is deliberate. If it points at a missing/broken file we must
        // surface that loudly (a Read error) rather than silently falling through to the home
        // files, which would mask a misconfiguration.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let missing = temp_credentials_path();
        let _ = std::fs::remove_file(&missing);
        let old = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", missing.to_str().unwrap());
        }
        let res = GrokCredentials::load_resolved_async().await;
        unsafe {
            match old {
                Some(v) => std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
        }
        assert!(matches!(res.unwrap_err(), GrokCredentialsError::Read(_)));
    }

    #[tokio::test]
    // See resolved_honors_explicit_env_override_first: env lock held across await is safe
    // on the current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn ambient_static_missing_token_falls_through_to_grok_cli_login() {
        // WHY: a stale ~/.xai/.credentials.json from an abandoned static-key attempt must
        // not break the common path where the user has a valid Grok CLI login in
        // ~/.grok/auth.json. Only the explicit XAI_CREDENTIALS_PATH override remains hard.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let home = std::env::temp_dir().join(format!("omni-grok-home-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(home.join(".xai")).unwrap();
        std::fs::create_dir_all(home.join(".grok")).unwrap();
        write_temp_creds(&home.join(".xai/.credentials.json"), r#"{"apiKey":" "}"#);
        write_temp_creds(
            &home.join(".grok/auth.json"),
            &grok_cli_json("jwt-from-cli", "2999-01-01T00:00:00Z"),
        );

        let old_home = std::env::var("HOME").ok();
        let old_path = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("HOME", &home);
            std::env::remove_var("XAI_CREDENTIALS_PATH");
        }
        let res = GrokCredentials::load_resolved_async().await;
        unsafe {
            match old_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match old_path {
                Some(v) => std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
        }
        let _ = std::fs::remove_dir_all(&home);

        let creds = res.expect("stale ambient static file must fall through to CLI login");
        assert_eq!(creds.api_key, "jwt-from-cli");
    }

    #[tokio::test]
    // ENV_LOCK held across the await: HOME / XAI_CREDENTIALS_PATH must stay fixed
    // through the resolve; safe on the current-thread test runtime.
    #[allow(clippy::await_holding_lock)]
    async fn conservative_prefers_oidc_over_static_for_user_id() {
        // WHY (Finding 3 regression): conservative mode must prefer the Grok CLI
        // OIDC login over a static key so `x-grok-user-id` is present and the
        // grok-shell fingerprint is exact. With BOTH files present, the
        // conservative resolver returns the OIDC creds (carrying user_id), whereas
        // the chat resolver (asserted alongside) still prefers the static key. This
        // pins the deliberate per-mode precedence divergence in both directions.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let home = std::env::temp_dir().join(format!("omni-grok-cons-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(home.join(".xai")).unwrap();
        std::fs::create_dir_all(home.join(".grok")).unwrap();
        write_temp_creds(
            &home.join(".xai/.credentials.json"),
            r#"{"apiKey": "xai-static-no-uid"}"#,
        );
        write_temp_creds(
            &home.join(".grok/auth.json"),
            &grok_cli_json("jwt-oidc-with-uid", "2999-01-01T00:00:00Z"),
        );

        let old_home = std::env::var("HOME").ok();
        let old_path = std::env::var("XAI_CREDENTIALS_PATH").ok();
        unsafe {
            std::env::set_var("HOME", &home);
            std::env::remove_var("XAI_CREDENTIALS_PATH");
        }
        let conservative = GrokCredentials::load_resolved_conservative_async().await;
        // The chat path (unchanged) must still prefer the static key here.
        let chat = GrokCredentials::load_resolved_async().await;
        unsafe {
            match old_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match old_path {
                Some(v) => std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
        }
        let _ = std::fs::remove_dir_all(&home);

        let conservative = conservative.expect("conservative resolve must succeed");
        assert_eq!(
            conservative.api_key, "jwt-oidc-with-uid",
            "conservative must prefer the OIDC CLI login"
        );
        assert_eq!(
            conservative.user_id.as_deref(),
            Some("11111111-2222-3333-4444-555555555555"),
            "OIDC user_id must be present for x-grok-user-id parity"
        );

        let chat = chat.expect("chat resolve must succeed");
        assert_eq!(
            chat.api_key, "xai-static-no-uid",
            "chat path precedence is UNCHANGED: explicit static beats ambient OIDC"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn conservative_honors_explicit_env_override_then_falls_to_static() {
        // WHY: $XAI_CREDENTIALS_PATH stays the highest-precedence explicit override
        // even in conservative mode (a deliberate operator choice); and when no CLI
        // OIDC file exists, conservative falls back to the static key rather than
        // erroring. Two cases in one lock window.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());

        // Case A: explicit env override wins (points at a static file).
        let env_file = temp_credentials_path();
        write_temp_creds(&env_file, r#"{"apiKey": "xai-env-explicit"}"#);
        let old_path = std::env::var("XAI_CREDENTIALS_PATH").ok();
        let old_home = std::env::var("HOME").ok();
        unsafe {
            std::env::set_var("XAI_CREDENTIALS_PATH", env_file.to_str().unwrap());
        }
        let via_env = GrokCredentials::load_resolved_conservative_async().await;

        // Case B: no env override, only a static key present (no ~/.grok) -> static.
        let home = std::env::temp_dir().join(format!("omni-grok-consb-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(home.join(".xai")).unwrap();
        write_temp_creds(
            &home.join(".xai/.credentials.json"),
            r#"{"apiKey": "xai-static-only"}"#,
        );
        unsafe {
            std::env::remove_var("XAI_CREDENTIALS_PATH");
            std::env::set_var("HOME", &home);
        }
        let via_static = GrokCredentials::load_resolved_conservative_async().await;

        unsafe {
            match old_path {
                Some(v) => std::env::set_var("XAI_CREDENTIALS_PATH", v),
                None => std::env::remove_var("XAI_CREDENTIALS_PATH"),
            }
            match old_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
        let _ = std::fs::remove_file(&env_file);
        let _ = std::fs::remove_dir_all(&home);

        assert_eq!(via_env.unwrap().api_key, "xai-env-explicit");
        assert_eq!(
            via_static.unwrap().api_key,
            "xai-static-only",
            "conservative must fall back to the static key when no CLI OIDC file exists"
        );
    }

    // ---- OAuth refresh (OMNI_OAUTH_REFRESH) ----

    #[test]
    fn build_refresh_form_body_matches_capture() {
        // WHY: wire parity with grok-shell form grant; wrong fields yield invalid_grant.
        let material = GrokOidcRefreshMaterial {
            entry_key: "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828".into(),
            refresh_token: "rt-abc".into(),
            client_id: "b1a00492-073a-47ea-816f-4c329264a828".into(),
            principal_id: "11111111-2222-3333-4444-555555555555".into(),
        };
        let form = build_refresh_form_body(&material);
        assert!(form.contains("grant_type=refresh_token"));
        assert!(form.contains("refresh_token=rt-abc"));
        assert!(form.contains("client_id=b1a00492-073a-47ea-816f-4c329264a828"));
        assert!(form.contains("principal_type=User"));
        assert!(form.contains("principal_id=11111111-2222-3333-4444-555555555555"));
        assert!(!form.contains('{'), "must be form-urlencoded, not JSON");
    }

    #[test]
    fn apply_grant_rotates_refresh_token_on_disk_shape() {
        // WHY: rotation-then-revoke — old RT must not remain after success.
        let mut file: Value = serde_json::from_str(&grok_cli_json("old-jwt", "2000-01-01T00:00:00Z"))
            .unwrap();
        let entry_key = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828";
        let grant = GrokTokenGrant {
            access_token: "new-jwt".into(),
            refresh_token: Some("rt-new-rotated".into()),
            expires_in: Some(21600),
        };
        let now = chrono::DateTime::parse_from_rfc3339("2026-07-09T12:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        apply_grant_to_auth_json(&mut file, entry_key, &grant, now).unwrap();
        let entry = &file[entry_key];
        assert_eq!(entry["key"], "new-jwt");
        assert_eq!(entry["refresh_token"], "rt-new-rotated");
        assert!(
            entry["expires_at"]
                .as_str()
                .unwrap()
                .starts_with("2026-07-09T18:00:00")
        );
        assert_eq!(entry["auth_mode"], "oidc");
        assert_eq!(
            entry["user_id"],
            "11111111-2222-3333-4444-555555555555"
        );
    }

    #[test]
    fn static_key_file_has_no_oidc_refresh_material() {
        // WHY: static API keys must never enter the OAuth refresh path.
        let bytes = br#"{"apiKey": "xai-static-key"}"#;
        assert!(matches!(
            extract_oidc_refresh_material(bytes),
            Err(GrokCredentialsError::MissingToken)
        ));
    }

    #[tokio::test]
    async fn refresh_oauth_inplace_posts_form_and_writes_rotated_rt() {
        // WHY: shipped Grok refresh must use form-urlencoded grant + persist rotated RT.
        use wiremock::matchers::{header, method, path as url_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(url_path("/oauth2/token"))
            .and(header(
                "content-type",
                "application/x-www-form-urlencoded",
            ))
            .and(header("user-agent", GROK_OAUTH_USER_AGENT))
            .and(wiremock::matchers::body_string_contains(
                "grant_type=refresh_token",
            ))
            .and(wiremock::matchers::body_string_contains(
                "refresh_token=rt-rotating-grace",
            ))
            .and(wiremock::matchers::body_string_contains(
                "principal_type=User",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "jwt-new-access",
                "token_type": "Bearer",
                "expires_in": 21600,
                "refresh_token": "rt-new-from-server",
                "scope": "openid profile email offline_access grok-cli:access api:access"
            })))
            .mount(&server)
            .await;

        let path = temp_credentials_path();
        write_temp_creds(&path, &grok_cli_json("jwt-old", "2000-01-01T00:00:00Z"));
        let token_url = format!("{}/oauth2/token", server.uri());
        refresh_oauth_inplace(&path, &token_url).await.unwrap();

        let on_disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        let entry = &on_disk["https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"];
        assert_eq!(entry["key"], "jwt-new-access");
        assert_eq!(entry["refresh_token"], "rt-new-from-server");
        assert_ne!(entry["expires_at"], "2000-01-01T00:00:00Z");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn load_fresh_async_refreshes_oidc_when_flag_on_and_expired() {
        use wiremock::matchers::{method, path as url_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(url_path("/oauth2/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "jwt-after-load",
                "refresh_token": "rt-after-load",
                "expires_in": 21600
            })))
            .mount(&server)
            .await;

        let path = temp_credentials_path();
        write_temp_creds(&path, &grok_cli_json("jwt-before", "2000-01-01T00:00:00Z"));
        let token_url = format!("{}/oauth2/token", server.uri());
        let old_flag = std::env::var_os("OMNI_OAUTH_REFRESH");
        let old_url = std::env::var_os("OMNI_GROK_OAUTH_TOKEN_URL");
        unsafe {
            std::env::set_var("OMNI_OAUTH_REFRESH", "1");
            std::env::set_var("OMNI_GROK_OAUTH_TOKEN_URL", &token_url);
        }
        let c = GrokCredentials::load_fresh_async(&path).await.unwrap();
        unsafe {
            match old_flag {
                Some(v) => std::env::set_var("OMNI_OAUTH_REFRESH", v),
                None => std::env::remove_var("OMNI_OAUTH_REFRESH"),
            }
            match old_url {
                Some(v) => std::env::set_var("OMNI_GROK_OAUTH_TOKEN_URL", v),
                None => std::env::remove_var("OMNI_GROK_OAUTH_TOKEN_URL"),
            }
        }
        let on_disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(c.api_key, "jwt-after-load");
        assert_eq!(
            on_disk["https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"]["refresh_token"],
            "rt-after-load"
        );
        assert!(c.check_expired().is_ok());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn static_key_load_never_attempts_refresh() {
        // WHY: static API-key credential path must remain untouched by OAuth refresh.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let path = temp_credentials_path();
        write_temp_creds(&path, r#"{"apiKey": "xai-never-refresh"}"#);
        let old = std::env::var_os("OMNI_OAUTH_REFRESH");
        unsafe {
            std::env::set_var("OMNI_OAUTH_REFRESH", "1");
        }
        let c = GrokCredentials::load_fresh_async(&path).await.unwrap();
        unsafe {
            match old {
                Some(v) => std::env::set_var("OMNI_OAUTH_REFRESH", v),
                None => std::env::remove_var("OMNI_OAUTH_REFRESH"),
            }
        }
        let on_disk = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(c.api_key, "xai-never-refresh");
        assert_eq!(on_disk, r#"{"apiKey": "xai-never-refresh"}"#);
    }
}
