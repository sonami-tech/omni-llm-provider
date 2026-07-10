//! In-place ChatGPT OAuth refresh for `~/.codex/auth.json`.
//!
//! OAuth refresh is **on by default**: Omni may refresh a near-expired
//! `tokens.access_token` (JWT `exp`) via the captured OpenAI auth endpoint and
//! atomically write back `tokens.*` + `last_refresh`. Disable with
//! `OMNI_OAUTH_REFRESH=0` (or `false`/`off`/`no`), `OMNI_NO_OAUTH_REFRESH=1`, or
//! the omni CLI flag `--no-oauth-refresh`. Static `OPENAI_API_KEY` entries are
//! never refreshed.
//!
//! Wire contract: live capture codex 0.144.1 — see
//! `/home/username/oauth-credential-renewal-handoff.md`.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{Value, json};
use tracing::warn;

/// Captured ChatGPT OAuth token endpoint.
pub const CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
/// Fixed public client_id confirmed in live request body.
pub const CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
/// Originator header from capture.
pub const CODEX_OAUTH_ORIGINATOR: &str = "codex_exec";
/// User-Agent template matching capture (`codex_exec/<ver> …`).
pub const CODEX_OAUTH_USER_AGENT: &str = "codex_exec/0.144.1 (linux; x86_64) unknown";

/// Refresh when JWT exp is within this many seconds (5 minutes).
const NEAR_EXPIRY_SKEW_SECS: i64 = 5 * 60;

#[derive(Debug, Clone, Deserialize)]
pub struct CodexTokenGrant {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub id_token: Option<String>,
    /// Present on the wire; access lifetime is taken from JWT `exp` after write-back.
    #[allow(dead_code)]
    pub expires_in: Option<i64>,
}

/// Gate for in-Omni OAuth refresh (shared name across providers).
/// Default **on**. Disable with `OMNI_OAUTH_REFRESH=0`/`false`/`off`/`no`,
/// `OMNI_NO_OAUTH_REFRESH=1`/`true`/`yes`/`on`, or CLI `--no-oauth-refresh`.
pub fn oauth_refresh_enabled() -> bool {
    if env_flag_truthy("OMNI_NO_OAUTH_REFRESH") {
        return false;
    }
    match std::env::var("OMNI_OAUTH_REFRESH")
        .ok()
        .map(|v| v.to_ascii_lowercase())
        .as_deref()
    {
        None => true,
        Some("0" | "false" | "off" | "no") => false,
        Some(_) => true,
    }
}

fn env_flag_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .map(|v| v.to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Effective token endpoint. Tests may set `OMNI_CODEX_OAUTH_TOKEN_URL`.
pub fn oauth_token_url() -> String {
    std::env::var("OMNI_CODEX_OAUTH_TOKEN_URL")
        .unwrap_or_else(|_| CODEX_OAUTH_TOKEN_URL.to_string())
}

/// Build JSON body for the Codex refresh_token grant (pure).
pub fn build_refresh_request_body(refresh_token: &str) -> Value {
    json!({
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })
}

/// Decode JWT `exp` (seconds since epoch) without verifying signature.
pub fn jwt_exp_secs(jwt: &str) -> Option<i64> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = b64url_decode(payload_b64)?;
    let value: Value = serde_json::from_slice(&payload).ok()?;
    value.get("exp")?.as_i64()
}

fn b64url_decode(input: &str) -> Option<Vec<u8>> {
    // Minimal base64url decode (no padding required).
    let mut s = input.replace('-', "+").replace('_', "/");
    while !s.len().is_multiple_of(4) {
        s.push('=');
    }
    // std has no base64; use a tiny decoder or serde via manual.
    base64_decode(&s)
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 256] = &{
        let mut t = [0xffu8; 256];
        let mut i = 0u8;
        while i < 26 {
            t[(b'A' + i) as usize] = i;
            t[(b'a' + i) as usize] = 26 + i;
            i += 1;
        }
        i = 0;
        while i < 10 {
            t[(b'0' + i) as usize] = 52 + i;
            i += 1;
        }
        t[b'+' as usize] = 62;
        t[b'/' as usize] = 63;
        t
    };

    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in bytes {
        if b == b'=' {
            break;
        }
        let v = TABLE[b as usize];
        if v == 0xff {
            continue;
        }
        buf = (buf << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

pub fn needs_refresh_access_token(access_token: &str) -> bool {
    match jwt_exp_secs(access_token) {
        None => false,
        Some(exp) => {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            now + NEAR_EXPIRY_SKEW_SECS >= exp
        }
    }
}

/// Apply grant onto auth.json Value (pure). Preserves account_id and sibling keys.
pub fn apply_grant_to_auth_json(
    file: &mut Value,
    grant: &CodexTokenGrant,
    now_rfc3339: &str,
) -> Result<(), String> {
    let tokens = file
        .get_mut("tokens")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| "auth.json missing tokens object".to_string())?;

    tokens.insert(
        "access_token".into(),
        Value::String(grant.access_token.clone()),
    );
    if let Some(rt) = grant.refresh_token.as_ref().filter(|s| !s.is_empty()) {
        tokens.insert("refresh_token".into(), Value::String(rt.clone()));
    }
    if let Some(id) = grant.id_token.as_ref().filter(|s| !s.is_empty()) {
        tokens.insert("id_token".into(), Value::String(id.clone()));
    }
    file.as_object_mut()
        .ok_or_else(|| "auth.json root must be object".to_string())?
        .insert("last_refresh".into(), Value::String(now_rfc3339.to_string()));
    Ok(())
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(
        ".{}.omni-oauth-{}-{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("auth.json"),
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
    // Preserve prior mode so rotated RTs do not become world-readable under umask 022.
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
        return Err(e.to_string());
    }
    Ok(())
}

fn refresh_token_from_auth(file: &Value) -> Option<String> {
    file.get("tokens")
        .and_then(|t| t.get("refresh_token"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
}

fn access_token_from_auth(file: &Value) -> Option<String> {
    file.get("tokens")
        .and_then(|t| t.get("access_token"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
}

/// If refresh is enabled and the ChatGPT access JWT is near expiry, refresh in-place.
///
/// - No-ops for static API-key-only files (no `tokens.refresh_token`).
/// - When `skip_if_static_api_key` is true (REST/env auth paths), also no-ops if a
///   top-level `OPENAI_API_KEY` is present — that key wins over OAuth tokens, so
///   rotating the RT would burn OAuth without helping the request.
/// - ChatGPT conservative auth sets `skip_if_static_api_key=false` so tokens still
///   refresh even when a sibling API key exists in the same file.
pub async fn maybe_refresh_auth_json(
    path: &Path,
    force: bool,
    skip_if_static_api_key: bool,
) -> Result<(), String> {
    if !oauth_refresh_enabled() {
        return Ok(());
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| e.to_string())?;
    let file: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if skip_if_static_api_key
        && file
            .get("OPENAI_API_KEY")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.trim().is_empty())
    {
        return Ok(());
    }
    let Some(access) = access_token_from_auth(&file) else {
        return Ok(());
    };
    if !force && !needs_refresh_access_token(&access) {
        return Ok(());
    }
    if refresh_token_from_auth(&file).is_none() {
        return Ok(());
    }
    refresh_oauth_inplace(path, &oauth_token_url()).await
}

/// POST the JSON grant and atomically write back to `path`.
pub async fn refresh_oauth_inplace(path: &Path, token_url: &str) -> Result<(), String> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| e.to_string())?;
    let file: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let refresh_token = refresh_token_from_auth(&file)
        .ok_or_else(|| "auth.json has no tokens.refresh_token".to_string())?;
    let body = build_refresh_request_body(&refresh_token);

    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;

    let resp = client
        .post(token_url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header("originator", CODEX_OAUTH_ORIGINATOR)
        .header(reqwest::header::USER_AGENT, CODEX_OAUTH_USER_AGENT)
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let status = resp.status();
    let resp_bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        let body_txt = String::from_utf8_lossy(&resp_bytes);
        return Err(format!("Codex OAuth refresh HTTP {status}: {body_txt}"));
    }

    let grant: CodexTokenGrant =
        serde_json::from_slice(&resp_bytes).map_err(|e| format!("token response parse: {e}"))?;
    if grant.access_token.is_empty() {
        return Err("token response missing access_token".into());
    }
    let new_rt = grant
        .refresh_token
        .as_ref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "token response missing refresh_token; refusing write-back that would strand RT".to_string()
        })?;

    // CAS: if another writer rotated the RT while we were on the network, do not
    // clobber their write with our snapshot of the pre-refresh file.
    let latest_bytes = tokio::fs::read(path)
        .await
        .map_err(|e| e.to_string())?;
    let mut latest: Value =
        serde_json::from_slice(&latest_bytes).map_err(|e| e.to_string())?;
    match refresh_token_from_auth(&latest).as_deref() {
        Some(disk_rt) if disk_rt == refresh_token || disk_rt == new_rt => {
            // Our RT still current, or peer already wrote the same grant.
        }
        Some(_) => {
            return Err(
                "auth.json refresh_token changed during OAuth refresh (concurrent writer); not clobbering"
                    .into(),
            );
        }
        None => {}
    }

    let now = chrono::Utc::now().to_rfc3339();
    apply_grant_to_auth_json(&mut latest, &grant, &now)?;
    let out = serde_json::to_vec_pretty(&latest).map_err(|e| e.to_string())?;
    atomic_write(path, &out)?;
    Ok(())
}

/// Best-effort proactive refresh for REST paths that prefer `OPENAI_API_KEY` when present.
pub async fn ensure_fresh_chatgpt_tokens(path: &Path) {
    if let Err(e) = maybe_refresh_auth_json(path, false, true).await {
        warn!(error = %e, path = %path.display(), "codex OAuth refresh failed");
    }
}

/// Proactive refresh for ChatGPT OAuth token consumers (conservative mode).
/// Still refreshes tokens even when a sibling `OPENAI_API_KEY` exists.
pub async fn ensure_fresh_chatgpt_oauth_tokens(path: &Path) {
    if let Err(e) = maybe_refresh_auth_json(path, false, false).await {
        warn!(error = %e, path = %path.display(), "codex OAuth refresh failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    fn write_temp_auth(content: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "omni-codex-auth-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&p, content).unwrap();
        p
    }

    /// Build a minimal unsigned JWT with the given exp claim.
    fn fake_jwt(exp: i64) -> String {
        let header = b64url_encode(br#"{"alg":"none","typ":"JWT"}"#);
        let payload = b64url_encode(format!(r#"{{"exp":{exp}}}"#).as_bytes());
        format!("{header}.{payload}.sig")
    }

    fn b64url_encode(data: &[u8]) -> String {
        const CHARS: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
            let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(CHARS[((n >> 18) & 63) as usize] as char);
            out.push(CHARS[((n >> 12) & 63) as usize] as char);
            if chunk.len() > 1 {
                out.push(CHARS[((n >> 6) & 63) as usize] as char);
            }
            if chunk.len() > 2 {
                out.push(CHARS[(n & 63) as usize] as char);
            }
        }
        out.replace('+', "-").replace('/', "_").replace('=', "")
    }

    #[test]
    fn build_refresh_request_body_matches_capture() {
        // WHY: client_id was confirmed in live request body; inventing it would fail auth.
        let body = build_refresh_request_body("rt-codex");
        assert_eq!(body["client_id"], CODEX_OAUTH_CLIENT_ID);
        assert_eq!(body["grant_type"], "refresh_token");
        assert_eq!(body["refresh_token"], "rt-codex");
    }

    #[test]
    fn jwt_exp_secs_reads_payload() {
        let exp = 1_700_000_000i64;
        let jwt = fake_jwt(exp);
        assert_eq!(jwt_exp_secs(&jwt), Some(exp));
    }

    #[test]
    fn needs_refresh_when_exp_past() {
        let jwt = fake_jwt(1_000);
        assert!(needs_refresh_access_token(&jwt));
        let future = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            + 3600;
        assert!(!needs_refresh_access_token(&fake_jwt(future)));
    }

    #[test]
    fn apply_grant_updates_tokens_and_last_refresh() {
        let mut file = json!({
            "tokens": {
                "access_token": "old-at",
                "refresh_token": "old-rt",
                "id_token": "old-id",
                "account_id": "acct-1"
            },
            "last_refresh": "2000-01-01T00:00:00Z"
        });
        let grant = CodexTokenGrant {
            access_token: "new-at".into(),
            refresh_token: Some("new-rt".into()),
            id_token: Some("new-id".into()),
            expires_in: Some(864000),
        };
        apply_grant_to_auth_json(&mut file, &grant, "2026-07-09T12:00:00+00:00").unwrap();
        assert_eq!(file["tokens"]["access_token"], "new-at");
        assert_eq!(file["tokens"]["refresh_token"], "new-rt");
        assert_eq!(file["tokens"]["id_token"], "new-id");
        assert_eq!(file["tokens"]["account_id"], "acct-1");
        assert_eq!(file["last_refresh"], "2026-07-09T12:00:00+00:00");
    }

    #[tokio::test]
    async fn refresh_oauth_inplace_posts_json_and_writes_tokens() {
        // WHY: shipped Codex path must match capture headers/body and persist rotated RT.
        use wiremock::matchers::{body_partial_json, header, method, path as url_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _guard = env_lock();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(url_path("/oauth/token"))
            .and(header("content-type", "application/json"))
            .and(header("originator", CODEX_OAUTH_ORIGINATOR))
            .and(header("user-agent", CODEX_OAUTH_USER_AGENT))
            .and(body_partial_json(json!({
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": "rt-old",
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": fake_jwt(9_999_999_999),
                "refresh_token": "rt-new",
                "id_token": "id-new",
                "token_type": "Bearer",
                "expires_in": 864000
            })))
            .mount(&server)
            .await;

        let path = write_temp_auth(&format!(
            r#"{{"tokens":{{"access_token":"{}","refresh_token":"rt-old","id_token":"id-old","account_id":"acct-9"}},"last_refresh":"2000-01-01T00:00:00Z"}}"#,
            fake_jwt(1_000)
        ));
        let token_url = format!("{}/oauth/token", server.uri());
        refresh_oauth_inplace(&path, &token_url).await.unwrap();

        let on_disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(on_disk["tokens"]["refresh_token"], "rt-new");
        assert_eq!(on_disk["tokens"]["id_token"], "id-new");
        assert_eq!(on_disk["tokens"]["account_id"], "acct-9");
        assert_ne!(on_disk["last_refresh"], "2000-01-01T00:00:00Z");
        assert!(
            jwt_exp_secs(on_disk["tokens"]["access_token"].as_str().unwrap()).unwrap()
                > 1_000_000_000
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn maybe_refresh_skips_api_key_only_file() {
        // WHY: OPENAI_API_KEY-only auth.json must not hit the token endpoint.
        let _guard = env_lock();
        let path = write_temp_auth(r#"{"OPENAI_API_KEY":"sk-static-only"}"#);
        let old = std::env::var_os("OMNI_OAUTH_REFRESH");
        unsafe {
            std::env::set_var("OMNI_OAUTH_REFRESH", "1");
        }
        maybe_refresh_auth_json(&path, true, true).await.unwrap();
        unsafe {
            match old {
                Some(v) => std::env::set_var("OMNI_OAUTH_REFRESH", v),
                None => std::env::remove_var("OMNI_OAUTH_REFRESH"),
            }
        }
        let on_disk = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(on_disk, r#"{"OPENAI_API_KEY":"sk-static-only"}"#);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn maybe_refresh_skips_oauth_when_static_api_key_wins() {
        // WHY: mixed auth.json with OPENAI_API_KEY must not burn OAuth RT on REST path.
        use wiremock::matchers::{method, path as url_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _guard = env_lock();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(url_path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "should-not-be-used",
                "refresh_token": "rt-should-not-rotate",
                "expires_in": 864000
            })))
            .expect(0)
            .mount(&server)
            .await;

        let path = write_temp_auth(&format!(
            r#"{{"OPENAI_API_KEY":"sk-wins","tokens":{{"access_token":"{}","refresh_token":"rt-old","account_id":"a1"}}}}"#,
            fake_jwt(1_000)
        ));
        let token_url = format!("{}/oauth/token", server.uri());
        let old_flag = std::env::var_os("OMNI_OAUTH_REFRESH");
        let old_url = std::env::var_os("OMNI_CODEX_OAUTH_TOKEN_URL");
        unsafe {
            std::env::set_var("OMNI_OAUTH_REFRESH", "1");
            std::env::set_var("OMNI_CODEX_OAUTH_TOKEN_URL", &token_url);
        }
        maybe_refresh_auth_json(&path, false, true).await.unwrap();
        unsafe {
            match old_flag {
                Some(v) => std::env::set_var("OMNI_OAUTH_REFRESH", v),
                None => std::env::remove_var("OMNI_OAUTH_REFRESH"),
            }
            match old_url {
                Some(v) => std::env::set_var("OMNI_CODEX_OAUTH_TOKEN_URL", v),
                None => std::env::remove_var("OMNI_CODEX_OAUTH_TOKEN_URL"),
            }
        }
        let on_disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(on_disk["tokens"]["refresh_token"], "rt-old");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn maybe_refresh_via_flag_and_inject_url() {
        use wiremock::matchers::{method, path as url_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let _guard = env_lock();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(url_path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": fake_jwt(9_999_999_999),
                "refresh_token": "rt-via-maybe",
                "id_token": "id-via-maybe",
                "expires_in": 864000
            })))
            .mount(&server)
            .await;

        let path = write_temp_auth(&format!(
            r#"{{"tokens":{{"access_token":"{}","refresh_token":"rt-old","account_id":"a1"}}}}"#,
            fake_jwt(1_000)
        ));
        let token_url = format!("{}/oauth/token", server.uri());
        let old_flag = std::env::var_os("OMNI_OAUTH_REFRESH");
        let old_url = std::env::var_os("OMNI_CODEX_OAUTH_TOKEN_URL");
        unsafe {
            std::env::set_var("OMNI_OAUTH_REFRESH", "1");
            std::env::set_var("OMNI_CODEX_OAUTH_TOKEN_URL", &token_url);
        }
        maybe_refresh_auth_json(&path, false, false).await.unwrap();
        unsafe {
            match old_flag {
                Some(v) => std::env::set_var("OMNI_OAUTH_REFRESH", v),
                None => std::env::remove_var("OMNI_OAUTH_REFRESH"),
            }
            match old_url {
                Some(v) => std::env::set_var("OMNI_CODEX_OAUTH_TOKEN_URL", v),
                None => std::env::remove_var("OMNI_CODEX_OAUTH_TOKEN_URL"),
            }
        }
        let on_disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(on_disk["tokens"]["refresh_token"], "rt-via-maybe");
    }
}
