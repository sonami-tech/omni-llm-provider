//! omni-grok: single-backend OpenAI-compatible proxy for xAI Grok models.
//!
//! Locked to one provider (`provider_grok::GrokProvider`). Speaks the shared
//! OpenAI-compatible wire shape from `omni_common::http` and delegates every
//! request to the canonical `LlmProvider` surface. No prefix routing (that lives
//! in the `omni` aggregator); this binary always talks to Grok.
//!
//! ## Surfaces
//! - GET  /health                -> "ok"
//! - GET  /                       -> banner
//! - GET  /v1/models , /models    -> static Grok catalog in OpenAI list shape
//! - GET  /stats                  -> request/usage snapshot JSON
//! - POST /v1/chat/completions    -> non-stream (JSON) or stream (SSE) completion
//!
//! ## Shared crate reuse
//! - omni_common::http: ChatCompletionRequest, to_canonical, from_canonical,
//!   sse_from_canonical_stream, unix_now_secs (single source of truth for the
//!   OpenAI translation; never re-implemented here).
//! - omni_common::auth_layer + AppError: consistent OAI-shaped auth + errors.
//! - omni_common::Stats: durable request/usage stats served at /stats.
//! - omni_core: CanonicalRequest/Response + LlmProvider trait (the delegation
//!   contract). provider_grok: the one real backend.
//!
//! ## Credentials
//! `GrokProvider::new(None)` requires creds at startup (XAI_API_KEY env or a
//! credentials file at $XAI_CREDENTIALS_PATH / ~/.xai/.credentials.json). Creds
//! are re-loaded fresh per request inside the provider, never cached or printed
//! here.
//!
//! Build: cargo build -p omni-grok
//! Run:   XAI_API_KEY=... cargo run -p omni-grok -- --port 18402

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use axum::{
    Router,
    extract::{Json, State},
    middleware,
    response::IntoResponse,
    routing::{get, post},
};
use clap::Parser;
use tracing::{info, warn};
use uuid::Uuid;

use omni_common::{
    AppError, ChatCompletionRequest, Stats, TokenUsage, auth_layer, from_canonical,
    responses_from_canonical, responses_to_canonical, sse_from_canonical_stream,
    sse_from_canonical_stream_responses, to_canonical, unix_now_secs,
};
use omni_core::{LlmProvider, ProviderError};
use provider_grok::GrokProvider;

/// Static Grok model catalog exposed via /v1/models and /models.
const GROK_MODELS: &[&str] = &["grok-3", "grok-4", "grok-4.3"];

/// CLI for the single-backend Grok proxy.
#[derive(Parser, Debug)]
#[command(
    name = "omni-grok",
    about = "OpenAI-compatible proxy for xAI Grok models"
)]
struct Cli {
    /// Listen port.
    #[arg(long, env = "OMNI_GROK_PORT", default_value_t = 18402)]
    port: u16,

    /// Disable API key auth. When omitted, auth is still off unless OMNI_API_KEYS
    /// is set (empty key set => allow all, matching omni-common::auth_layer).
    #[arg(long, env = "OMNI_GROK_NO_AUTH")]
    no_auth: bool,
}

/// Shared handler state: the one Grok provider plus optional stats.
///
/// Stats is `Option` because serving must never fail just because the stats DB
/// could not be opened; a failure logs and continues with stats disabled.
#[derive(Clone)]
struct AppState {
    provider: Arc<dyn LlmProvider>,
    stats: Option<Arc<Stats>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,omni_grok=debug,provider_grok=debug")
        .init();

    let cli = Cli::parse();

    // Provider init requires creds at startup (fails fast). Map the failure to a
    // clear, actionable message; never print the creds themselves.
    let provider: Arc<dyn LlmProvider> = Arc::new(GrokProvider::new(None).context(
        "failed to init grok provider: set XAI_API_KEY or provide ~/.xai/.credentials.json",
    )?);
    info!("grok provider initialized");

    // Stats DB lives at a local temp/data path the app owns. A failure to open
    // must not prevent serving, so we log and continue without stats.
    let stats_path = std::env::temp_dir().join("omni-grok-stats.redb");
    let stats: Option<Arc<Stats>> = match Stats::open(&stats_path) {
        Ok(s) => {
            info!(path = %stats_path.display(), "stats db opened");
            Some(Arc::new(s))
        }
        Err(e) => {
            warn!(error = %e, path = %stats_path.display(), "failed to open stats db; continuing without stats");
            None
        }
    };

    let state = Arc::new(AppState { provider, stats });

    // Auth keys: empty set => allow-all (see omni_common::auth_layer). When not
    // --no-auth, a non-empty OMNI_API_KEYS enables key checking.
    let auth_keys: Arc<HashSet<String>> = if cli.no_auth {
        Arc::new(HashSet::new())
    } else {
        Arc::new(parse_api_keys(
            std::env::var("OMNI_API_KEYS").ok().as_deref(),
        ))
    };
    info!(auth_enabled = !auth_keys.is_empty(), "auth layer ready");

    let app = build_router(state, auth_keys);

    let addr: SocketAddr = format!("127.0.0.1:{}", cli.port).parse()?;
    info!("omni-grok listening on http://{}", addr);
    info!("  try:    curl http://{}/health", addr);
    info!("  models: curl http://{}/v1/models", addr);

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app)
        .await
        .context("server error")?;

    Ok(())
}

/// Parse OMNI_API_KEYS (comma-separated) into a key set; empty/absent => empty.
fn parse_api_keys(raw: Option<&str>) -> HashSet<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Build the full router, layering the shared auth middleware. Extracted so the
/// subprocess-free in-process tests can construct the exact production surface.
fn build_router(state: Arc<AppState>, auth_keys: Arc<HashSet<String>>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/", get(root_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/models", get(models_handler))
        .route("/models", get(models_handler))
        .route("/stats", get(stats_handler))
        .with_state(state)
        .layer(middleware::from_fn(move |req, next| {
            auth_layer(auth_keys.clone(), req, next)
        }))
}

/// GET /health
async fn health_handler() -> impl IntoResponse {
    "ok"
}

/// GET /
async fn root_handler() -> impl IntoResponse {
    "omni-grok - OpenAI compatible for xAI Grok models"
}

/// GET /v1/models and /models. Static Grok catalog in OpenAI list shape.
async fn models_handler() -> impl IntoResponse {
    let data: Vec<serde_json::Value> = GROK_MODELS
        .iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": "grok",
            })
        })
        .collect();
    Json(serde_json::json!({ "object": "list", "data": data }))
}

/// GET /stats. Snapshot of durable request/usage counters as JSON.
async fn stats_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match &state.stats {
        Some(s) => {
            Json(serde_json::to_value(s.snapshot()).unwrap_or_else(|_| serde_json::json!({})))
                .into_response()
        }
        None => Json(serde_json::json!({ "stats": "unavailable" })).into_response(),
    }
}

/// POST /v1/chat/completions. Non-stream returns JSON; stream returns SSE.
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, AppError> {
    let requested_model = body.model.clone();
    let canon = to_canonical(&body).map_err(AppError::BadRequest)?;

    if let Some(s) = &state.stats {
        s.record_request(&requested_model, None);
    }

    let chat_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_now_secs();

    if body.stream {
        // Streaming path: open the upstream stream and frame it as OpenAI SSE.
        let stream = state
            .provider
            .send_stream(canon)
            .await
            .map_err(|e| map_provider_err(&state, &requested_model, e))?;
        let sse = sse_from_canonical_stream(stream, requested_model, chat_id, created);
        Ok(sse.into_response())
    } else {
        // Non-streaming path: full response, recorded into stats.
        let start = Instant::now();
        let canon_resp = state
            .provider
            .send(canon)
            .await
            .map_err(|e| map_provider_err(&state, &requested_model, e))?;

        if let Some(s) = &state.stats {
            let usage = TokenUsage {
                input_tokens: canon_resp.usage.input_tokens,
                output_tokens: canon_resp.usage.output_tokens,
                cache_read_input_tokens: canon_resp.usage.cache_read,
                cache_creation_input_tokens: canon_resp.usage.cache_creation,
            };
            s.record_response(&requested_model, usage, None, start.elapsed().as_secs_f64());
        }

        let oai = from_canonical(canon_resp, requested_model, chat_id, created);
        Ok(Json(oai).into_response())
    }
}

/// Map a `ProviderError` to an `AppError`, recording the error in stats first.
fn map_provider_err(state: &AppState, model: &str, e: ProviderError) -> AppError {
    if let Some(s) = &state.stats {
        s.record_error(model, &e.to_string());
    }
    match e {
        ProviderError::Auth(msg) => AppError::Unauthorized(msg),
        ProviderError::Upstream(msg) => AppError::ServerError(format!("upstream: {}", msg)),
        ProviderError::Other(a) => AppError::ServerError(a.to_string()),
    }
}

/// Handler for POST /v1/responses (OpenAI Responses API protocol).
///
/// Contract (pinned by the responses tests below + omni-common::responses):
/// unsupported input shapes map to BadRequest; non-stream returns the
/// Responses envelope; stream:true returns Responses SSE (response.created
/// ... response.completed, no [DONE]); requests/errors recorded in stats
/// exactly like chat completions.
async fn responses_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<omni_common::ResponsesRequest>,
) -> Result<axum::response::Response, AppError> {
    let requested_model = body.model.clone();
    let stream = body.stream;

    // Record the inbound request FIRST (mirroring the chat handler's record-then
    // -work order) so even a request later rejected for unsupported input counts
    // as inbound traffic; an unrecorded protocol would skew per-model accounting.
    if let Some(s) = &state.stats {
        s.record_request(&requested_model, None);
    }

    // Unsupported input shapes are rejected loudly as a 400 naming the offender.
    let canon = responses_to_canonical(&body).map_err(AppError::BadRequest)?;

    let response_id = format!("resp_{}", Uuid::new_v4());
    let created_at = unix_now_secs();

    if stream {
        let stream = state
            .provider
            .send_stream(canon)
            .await
            .map_err(|e| map_provider_err(&state, &requested_model, e))?;
        let sse =
            sse_from_canonical_stream_responses(stream, requested_model, response_id, created_at);
        Ok(sse.into_response())
    } else {
        let start = Instant::now();
        let canon_resp = state
            .provider
            .send(canon)
            .await
            .map_err(|e| map_provider_err(&state, &requested_model, e))?;

        if let Some(s) = &state.stats {
            let usage = TokenUsage {
                input_tokens: canon_resp.usage.input_tokens,
                output_tokens: canon_resp.usage.output_tokens,
                cache_read_input_tokens: canon_resp.usage.cache_read,
                cache_creation_input_tokens: canon_resp.usage.cache_creation,
            };
            s.record_response(&requested_model, usage, None, start.elapsed().as_secs_f64());
        }

        let resp = responses_from_canonical(canon_resp, requested_model, response_id, created_at);
        Ok(Json(resp).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::Duration;

    // A clearly-labeled dummy key that lets GrokProvider::new(None) succeed at
    // startup WITHOUT real creds. It never reaches the upstream in the hermetic
    // tests (those endpoints do not call xAI), so no real network/auth happens.
    const TEST_DUMMY_KEY: &str = "xai-dummy-test-key";

    // ---- shared subprocess test helpers (every test uses its OWN free port) ----

    /// Bind 127.0.0.1:0 to get a free ephemeral port, then drop the listener so
    /// the spawned binary can claim it. WHY random per test: the previous stub
    /// hardcoded 18322 and two subprocess tests collided on it, so one always
    /// failed to read a health body. Random ports make the suite collision-free
    /// and re-runnable in parallel.
    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("bind ephemeral port")
            .local_addr()
            .unwrap()
            .port()
    }

    /// Resolve the standalone `omni-grok` binary, building it on demand. WHY
    /// build-on-demand: `cargo test -p omni-grok` only compiles the unit-test
    /// harness, NOT the standalone binary, and unit tests (unlike integration
    /// tests) do not get a `CARGO_BIN_EXE_*` env var. Building here (and locating
    /// the real artifact via cargo's JSON output, which honors CARGO_TARGET_DIR and
    /// builds the dev profile) makes the subprocess suite self-contained and green
    /// offline regardless of invocation order, and never spawns a stale binary.
    /// Cached so the build runs once per test process. Kept in sync with
    /// omni::omni_bin_path.
    fn bin_path() -> std::path::PathBuf {
        // cargo normalizes '-' to '_' in the injected env var name.
        if let Ok(p) = std::env::var("CARGO_BIN_EXE_omni_grok") {
            return std::path::PathBuf::from(p);
        }
        static BIN: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        BIN.get_or_init(|| omni_common::test_support::build_workspace_bin("omni-grok"))
            .clone()
    }

    /// Poll /health (optionally with an auth header) until it returns "ok".
    fn wait_for_health(port: u16, bearer: Option<&str>, timeout: Duration) -> bool {
        let start = Instant::now();
        let url = format!("http://127.0.0.1:{}/health", port);
        let header = bearer.map(|b| format!("Authorization: Bearer {}", b));
        while start.elapsed() < timeout {
            let mut args: Vec<&str> = vec!["-s", "--max-time", "1"];
            if let Some(h) = &header {
                args.push("-H");
                args.push(h);
            }
            args.push(&url);
            if let Ok(out) = Command::new("curl").args(&args).output()
                && out.status.success()
                && String::from_utf8_lossy(&out.stdout).trim() == "ok"
            {
                return true;
            }
            thread::sleep(Duration::from_millis(100));
        }
        false
    }

    fn http_code(args: &[&str]) -> String {
        let out = Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}"])
            .args(args)
            .output()
            .expect("curl");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// True when real xAI creds are reachable for the live-conditional test.
    /// Race-safe: when XAI_CREDENTIALS_PATH is set (some other test points it at
    /// a throwaway dummy file) we treat creds as ABSENT, and we read the home
    /// path directly rather than the env-overridable default. Mirrors the guard
    /// in provider-grok so the offline suite stays green and never fires a real
    /// network call with a junk key.
    fn has_grok_creds() -> bool {
        if std::env::var("XAI_API_KEY")
            .map(|k| !k.trim().is_empty())
            .unwrap_or(false)
        {
            return true;
        }
        if std::env::var_os("XAI_CREDENTIALS_PATH").is_some() {
            return false;
        }
        match std::env::var_os("HOME") {
            Some(home) => std::path::Path::new(&home)
                .join(".xai")
                .join(".credentials.json")
                .exists(),
            None => false,
        }
    }

    // ---- in-process handler tests (no network, no subprocess) ----
    //
    // These call the handlers directly (constructing the axum extractors) and
    // inspect the rendered Response. WHY direct calls instead of a tower oneshot:
    // it keeps the dependency surface to what the workspace already provides (no
    // tower dev-dep) while still exercising the real handlers + error mapping.

    use axum::http::StatusCode;

    /// A Grok test provider pointed at a bad port: lets us exercise the handlers
    /// + error mapping + the stream-vs-nonstream branch without touching xAI.
    fn test_state() -> Arc<AppState> {
        Arc::new(AppState {
            provider: Arc::new(GrokProvider::new_for_test("k", "http://127.0.0.1:1")),
            stats: None,
        })
    }

    async fn render(resp: axum::response::Response) -> (StatusCode, String, String) {
        let status = resp.status();
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, ct, String::from_utf8_lossy(&bytes).to_string())
    }

    #[tokio::test]
    async fn models_handler_lists_grok_catalog() {
        // WHY: clients discover available models here; the static catalog must
        // surface the grok ids in OpenAI list shape.
        let (status, _ct, body) = render(models_handler().await.into_response()).await;
        assert_eq!(status, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["object"], "list");
        let ids: Vec<String> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap().to_string())
            .collect();
        assert!(ids.contains(&"grok-3".to_string()));
        assert!(ids.contains(&"grok-4.3".to_string()));
    }

    #[tokio::test]
    async fn stats_handler_returns_json_snapshot() {
        // WHY: /stats must always parse as JSON even when the DB is disabled, so
        // a dashboard never chokes on the response.
        let (status, _ct, body) =
            render(stats_handler(State(test_state())).await.into_response()).await;
        assert_eq!(status, StatusCode::OK);
        let _v: serde_json::Value = serde_json::from_str(&body).expect("stats is json");
    }

    #[tokio::test]
    async fn nonstream_upstream_error_maps_to_500() {
        // WHY: the bad-port test provider yields a network Upstream error; the
        // handler must surface that as a 5xx server error (not a panic / 400),
        // proving the ProviderError -> AppError mapping on the non-stream path.
        let body = serde_json::json!({
            "model": "grok-3",
            "messages": [{"role": "user", "content": "ping"}],
        });
        let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let resp = chat_completions_handler(State(test_state()), Json(req))
            .await
            .into_response();
        let (status, _ct, text) = render(resp).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            text.contains("upstream") || text.contains("network"),
            "got: {text}"
        );
    }

    #[tokio::test]
    async fn stream_true_returns_sse_not_400() {
        // WHY: streaming is supported (unlike the aggregator). With stream:true
        // the response must be an SSE response framed from the canonical stream,
        // NOT a 400 rejection. The bad-port upstream surfaces inside the SSE body
        // as an error/finish chunk, so the HTTP status is 200 with an SSE
        // content-type and the body ends in the [DONE] sentinel.
        let body = serde_json::json!({
            "model": "grok-3",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": true,
        });
        let req: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let resp = chat_completions_handler(State(test_state()), Json(req))
            .await
            .into_response();
        let (status, ct, text) = render(resp).await;
        assert_eq!(status, StatusCode::OK);
        assert!(ct.contains("text/event-stream"), "expected SSE, got {ct}");
        assert!(
            text.contains("[DONE]"),
            "SSE must terminate with [DONE]: {text}"
        );
    }

    // ---- subprocess + curl tests (full HTTP stack, RANDOM free port, kill child) ----

    /// Spawn the binary on its OWN free port with the dummy key so the provider
    /// starts without real creds. The caller MUST kill the returned child.
    fn spawn_hermetic(port: u16, extra_env: &[(&str, &str)], no_auth: bool) -> std::process::Child {
        let mut cmd = Command::new(bin_path());
        cmd.env("XAI_API_KEY", TEST_DUMMY_KEY)
            .arg("--port")
            .arg(port.to_string());
        if no_auth {
            cmd.arg("--no-auth");
        }
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        cmd.stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn omni-grok")
    }

    #[test]
    fn subprocess_health_models_stats_on_random_port() {
        // WHY: exercises the full layered HTTP stack (health, models, stats) via
        // a real process on a RANDOM free port. The dummy key lets the provider
        // start; none of these endpoints call xAI, so the test is hermetic.
        let port = free_port();
        let mut child = spawn_hermetic(port, &[], true);
        assert!(
            wait_for_health(port, None, Duration::from_secs(8)),
            "omni-grok did not become healthy on {port}"
        );

        // /v1/models -> JSON list containing grok ids.
        let out = Command::new("curl")
            .args(["-s", &format!("http://127.0.0.1:{}/v1/models", port)])
            .output()
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("models json");
        assert_eq!(v["object"], "list");
        let ids: Vec<String> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str().map(str::to_string))
            .collect();
        assert!(
            ids.iter().any(|id| id == "grok-3"),
            "grok ids present: {ids:?}"
        );

        // /stats parses as JSON.
        let out = Command::new("curl")
            .args(["-s", &format!("http://127.0.0.1:{}/stats", port)])
            .output()
            .unwrap();
        let _v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stats json");

        let _ = child.kill();
        let _ = child.wait(); // reap the child so no zombie is left behind
    }

    #[test]
    fn subprocess_auth_401_without_key_200_with_key() {
        // WHY: proves the auth middleware end-to-end through the real binary. With
        // OMNI_API_KEYS set and no --no-auth, a header-less request is 401 and a
        // good-key request is 200. Own free port; child is killed.
        let port = free_port();
        let mut child = spawn_hermetic(port, &[("OMNI_API_KEYS", "secret123,other")], false);
        assert!(
            wait_for_health(port, Some("secret123"), Duration::from_secs(8)),
            "protected omni-grok did not become ready on {port}"
        );

        let url = format!("http://127.0.0.1:{}/health", port);
        // no header -> 401
        assert_eq!(http_code(&[&url]), "401");
        // good key -> 200
        assert_eq!(
            http_code(&["-H", "Authorization: Bearer secret123", &url]),
            "200"
        );
        // bad key -> 401
        assert_eq!(
            http_code(&["-H", "Authorization: Bearer wrong", &url]),
            "401"
        );

        let _ = child.kill();
        let _ = child.wait(); // reap the child so no zombie is left behind
    }

    #[test]
    fn live_conditional_real_completion_nonstream_and_stream() {
        // WHY: when real creds are present, prove a genuine non-stream AND stream
        // completion round-trips through the binary. Skips cleanly (and loudly)
        // when creds are absent so the offline suite stays green. NOTE: we do NOT
        // inject the dummy key here, because real creds must be used; the guard is
        // race-safe (see has_grok_creds).
        if !has_grok_creds() {
            eprintln!(
                "skipping live grok completion test (no XAI_API_KEY and no ~/.xai/.credentials.json)"
            );
            return;
        }
        let port = free_port();
        // Pass through the real env key if set; otherwise the provider loads the
        // credentials file fresh. Do not override with the dummy.
        let mut cmd = Command::new(bin_path());
        cmd.arg("--no-auth").arg("--port").arg(port.to_string());
        if let Ok(k) = std::env::var("XAI_API_KEY") {
            cmd.env("XAI_API_KEY", k);
        }
        let mut child = cmd
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn omni-grok (live)");
        assert!(
            wait_for_health(port, None, Duration::from_secs(8)),
            "live omni-grok did not become healthy on {port}"
        );

        // Non-stream real completion.
        let out = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"grok-3","messages":[{"role":"user","content":"Reply with the single word PONG"}],"max_tokens":8}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).unwrap_or(serde_json::json!({}));
        assert!(
            v.get("choices").is_some(),
            "live non-stream completion should yield choices, got: {v}"
        );

        // Stream real completion: SSE body must terminate with [DONE].
        let out = Command::new("curl")
            .args([
                "-sN",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"grok-3","messages":[{"role":"user","content":"Reply with the single word PONG"}],"max_tokens":8,"stream":true}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            text.contains("[DONE]"),
            "live stream must end with [DONE]: {text}"
        );

        let _ = child.kill();
        let _ = child.wait(); // reap the child so no zombie is left behind
    }

    // ---- OpenAI Responses protocol (/v1/responses): TDD contract tests ----

    #[tokio::test]
    async fn responses_unsupported_input_is_bad_request() {
        // WHY: input shapes the canonical layer still cannot represent (an
        // `input_image` content part) are rejected loudly as an OAI-shaped 400
        // naming the offender, instead of silently mangling the request.
        // (function_call / function_call_output items ARE now supported and
        // round-trip through canonical tool blocks.)
        let state = test_state();
        let req: omni_common::ResponsesRequest = serde_json::from_str(
            r#"{"model":"grok-3","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"x"}]}]}"#,
        )
        .expect("responses request json");
        let res = responses_handler(State(state), Json(req)).await;
        match res {
            Err(AppError::BadRequest(msg)) => assert!(
                msg.contains("input_image"),
                "400 must name the unsupported content part type: {msg}"
            ),
            other => panic!("expected BadRequest for unsupported input, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn responses_nonstream_upstream_error_maps_to_500() {
        // WHY: proves the Responses handler delegates to the provider with the
        // same error classification as chat completions: a dead upstream is a
        // ServerError (5xx), never a parse-level 400.
        let state = test_state();
        let req: omni_common::ResponsesRequest =
            serde_json::from_str(r#"{"model":"grok-3","input":"ping"}"#).unwrap();
        let res = responses_handler(State(state), Json(req)).await;
        match res {
            Err(AppError::ServerError(msg)) => {
                assert!(msg.contains("upstream") || msg.contains("network"))
            }
            other => panic!("expected upstream ServerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn responses_stream_returns_sse_with_failed_event_on_dead_upstream() {
        // WHY: grok's send_stream defers the HTTP call into the stream body, so
        // stream:true must yield an SSE response even for a dead upstream, whose
        // terminal Responses event is response.failed. Pins hermetically that
        // streaming is routed (never a 400) and that stream errors surface in
        // Responses framing, not the Chat [DONE] convention.
        let state = test_state();
        let req: omni_common::ResponsesRequest =
            serde_json::from_str(r#"{"model":"grok-3","input":"ping","stream":true}"#).unwrap();
        let res = responses_handler(State(state), Json(req)).await;
        let resp = match res {
            Ok(r) => r,
            Err(e) => panic!("stream must not be rejected; got error {e:?}"),
        };
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("text/event-stream"),
            "Responses stream must be SSE, got content-type {ct:?}"
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body_bytes);
        assert!(
            body.contains("event: response.failed"),
            "dead upstream must terminate the stream with response.failed: {body}"
        );
        assert!(
            !body.contains("[DONE]"),
            "Responses SSE has no [DONE] sentinel"
        );
    }

    #[tokio::test]
    async fn responses_route_registered_returns_400_not_404_for_bad_input() {
        // WHY: the route must be wired into the PRODUCTION router. A
        // parseable-but-unsupported body is rejected at parse level (hermetic,
        // no upstream call); a 404 means /v1/responses is not registered.
        use tower::ServiceExt;
        let app = build_router(test_state(), Arc::new(std::collections::HashSet::new()));
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"model":"grok-3","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"x"}]}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::BAD_REQUEST,
            "POST /v1/responses must exist and reject unsupported input with 400 (404 = not registered)"
        );
    }

    #[test]
    fn responses_live_subprocess_roundtrip_conditional() {
        // WHY: end-to-end live proof over real HTTP that the standalone binary
        // serves the Responses protocol: envelope for non-stream, Responses SSE
        // events for stream. Skips cleanly offline.
        if !has_grok_creds() {
            eprintln!("skipping responses live subprocess test: no grok creds");
            return;
        }
        let port = free_port();
        let mut child = Command::new(bin_path())
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn omni-grok");
        assert!(wait_for_health(port, None, Duration::from_secs(6)));

        let out = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"grok-3","input":"Reply with the single word PONG","max_output_tokens":16}"#,
                &format!("http://127.0.0.1:{}/v1/responses", port),
            ])
            .output()
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("responses json");
        assert_eq!(v["object"], "response", "live body: {v}");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert!(
            !v["output"][0]["content"][0]["text"]
                .as_str()
                .unwrap_or("")
                .is_empty()
        );

        let out2 = Command::new("curl")
            .args([
                "-sN",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"grok-3","input":"Reply with the single word PONG","max_output_tokens":16,"stream":true}"#,
                &format!("http://127.0.0.1:{}/v1/responses", port),
            ])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out2.stdout);
        let _ = child.kill();
        let _ = child.wait();
        assert!(text.contains("event: response.created"));
        assert!(text.contains("event: response.output_text.delta"));
        assert!(text.contains("event: response.completed"));
        assert!(!text.contains("[DONE]"));
    }
}
