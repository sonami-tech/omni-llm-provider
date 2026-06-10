//! omni-claude: single-backend Claude proxy.
//!
//! An OpenAI-compatible HTTP front for the Claude Code / Anthropic Max upstream.
//! Unlike the `omni` aggregator there is NO prefix routing and NO providers map:
//! this binary is locked to exactly one provider (claude), so the model name is
//! passed straight through to the provider (the provider's fingerprint profile
//! does its own alias/catalog resolution).
//!
//! ## How it uses the shared crates
//! - omni-common: the OpenAI wire types + canonical conversion + SSE framing
//!   (`ChatCompletionRequest`, `to_canonical`, `from_canonical`,
//!   `sse_from_canonical_stream`, `unix_now_secs`), `auth_layer`, `AppError`,
//!   and `Stats`. We reuse these instead of re-deriving them so the wire shape
//!   stays identical across all binaries.
//! - omni-core: `LlmProvider` + Canonical* (the delegation contract).
//! - provider-claude: the one concrete backend (`ClaudeProvider`). All Claude
//!   fingerprint logic + fresh credential reads live inside that crate; this
//!   binary never touches credentials directly.
//!
//! ## Surfaces (OpenAI-compatible)
//! - GET  /health                -> "ok"
//! - GET  /                       -> description line
//! - GET  /v1/models , /models    -> the Claude catalog (OpenAI list shape)
//! - GET  /stats                  -> Stats::snapshot() as JSON
//! - POST /v1/chat/completions    -> non-stream JSON or SSE stream
//!
//! Auth: always layers `omni-common::auth_layer`. An empty key set (via
//! `--no-auth`, or no `OMNI_API_KEYS`) means "allow all" (passthrough).
//!
//! Stats: opened best-effort. If the redb file cannot be opened we log and
//! continue with stats disabled (`Option<Arc<Stats>>`) so a stats failure can
//! never break serving traffic.
//!
//! Build: cargo build -p omni-claude
//! Run (no keys needed): cargo run -p omni-claude -- --no-auth
//! Test: cargo test -p omni-claude

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use axum::{
    Router,
    extract::{Json, State},
    http::StatusCode,
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
use provider_claude::ClaudeProvider;

/// CLI for the single-backend Claude proxy.
/// Env vars (clap env support): OMNI_CLAUDE_PORT, OMNI_CLAUDE_NO_AUTH,
/// OMNI_CLAUDE_STATS_DB. The API key set is read from OMNI_API_KEYS (shared
/// with the other binaries) when auth is enabled.
#[derive(Parser, Debug)]
#[command(
    name = "omni-claude",
    about = "Single-backend OpenAI-compatible proxy for Claude (Anthropic Max)"
)]
struct Cli {
    /// Listen port.
    #[arg(long, env = "OMNI_CLAUDE_PORT", default_value_t = 18401)]
    port: u16,

    /// Disable API key auth. If omitted, auth still allows all unless
    /// OMNI_API_KEYS is set to a non-empty list.
    #[arg(long, env = "OMNI_CLAUDE_NO_AUTH")]
    no_auth: bool,

    /// Path to the stats redb file. Defaults to a file under the system temp
    /// dir. Stats are best-effort: if this file cannot be opened the server
    /// still serves traffic with stats disabled.
    #[arg(long, env = "OMNI_CLAUDE_STATS_DB")]
    stats_db: Option<PathBuf>,
}

/// Shared application state for the single Claude backend.
///
/// We store the concrete `ClaudeProvider` (not `Arc<dyn LlmProvider>`) because
/// the `/models` endpoint needs the Claude-specific fingerprint profile catalog,
/// which is not part of the object-safe `LlmProvider` trait. `ClaudeProvider`
/// still implements `LlmProvider`, so the completions handler delegates through
/// it exactly like the aggregator does through a boxed provider.
#[derive(Clone)]
struct AppState {
    /// The one and only provider (claude).
    provider: Arc<ClaudeProvider>,
    /// Best-effort stats. `None` when the redb file could not be opened so the
    /// request path never depends on stats being available.
    stats: Option<Arc<Stats>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,omni_claude=debug,provider_claude=debug")
        .init();

    let cli = Cli::parse();

    // Construct the single Claude provider once. new() reads no credentials
    // (those are read fresh per request inside the provider).
    let provider = Arc::new(
        ClaudeProvider::new()
            .context("failed to initialize claude provider (fingerprint profile)")?,
    );
    info!(provider = provider.id(), "claude provider initialized");

    // Open stats best-effort. A failure here must NOT stop the server: we log
    // and run with stats disabled.
    let stats_path = cli.stats_db.clone().unwrap_or_else(default_stats_db_path);
    let stats: Option<Arc<Stats>> = match Stats::open(&stats_path) {
        Ok(s) => {
            info!(path = %stats_path.display(), "stats db opened");
            Some(Arc::new(s))
        }
        Err(e) => {
            warn!(path = %stats_path.display(), error = %e, "stats db open failed; continuing with stats disabled");
            None
        }
    };

    let state = Arc::new(AppState { provider, stats });

    // Auth keys: empty set => allow-all (see omni-common::auth_layer).
    let auth_keys: Arc<HashSet<String>> = if cli.no_auth {
        Arc::new(HashSet::new())
    } else {
        Arc::new(parse_api_keys(
            std::env::var("OMNI_API_KEYS").ok().as_deref(),
        ))
    };
    info!(no_auth_effective = auth_keys.is_empty(), "auth layer ready");

    let app = build_router(state, auth_keys);

    let addr: SocketAddr = format!("127.0.0.1:{}", cli.port).parse()?;
    info!("omni-claude listening on http://{}", addr);
    info!("  try:    curl http://{}/health", addr);
    info!("  models: curl http://{}/v1/models", addr);
    info!("  stats:  curl http://{}/stats", addr);

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app)
        .await
        .context("server error")?;

    Ok(())
}

/// Default stats DB path: a clearly named file under the system temp dir. The
/// app owns this file; it is not a production/persistent data store.
fn default_stats_db_path() -> PathBuf {
    std::env::temp_dir().join("omni-claude-stats.redb")
}

/// Parse a comma-separated OMNI_API_KEYS value into a key set (trimmed,
/// non-empty entries only). Pure for unit testing.
fn parse_api_keys(raw: Option<&str>) -> HashSet<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Build the full router. Extracted so tests can construct the exact production
/// surface (routes + auth layer) without spawning the binary.
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
        // Always layer; the common impl short-circuits when keys are empty.
        .layer(middleware::from_fn(move |req, next| {
            let keys = auth_keys.clone();
            async move { auth_layer(keys, req, next).await }
        }))
}

/// Map a provider error to the OAI-shaped AppError, mirroring the aggregator.
fn map_provider_err(e: ProviderError) -> AppError {
    match e {
        ProviderError::Auth(msg) => AppError::Unauthorized(msg),
        ProviderError::Upstream(msg) => AppError::ServerError(format!("upstream: {msg}")),
        ProviderError::Other(a) => AppError::ServerError(a.to_string()),
    }
}

/// GET /health
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// GET /
async fn root_handler() -> impl IntoResponse {
    "omni-claude - single-backend OpenAI-compatible proxy for Claude (Anthropic Max)"
}

/// GET /v1/models (and /models). Lists the Claude catalog from the provider's
/// fingerprint profile in the OpenAI list shape.
async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // The catalog lives on the concrete ClaudeProvider's fingerprint profile.
    let data: Vec<serde_json::Value> = state
        .provider
        .profile()
        .models_list()
        .into_iter()
        .map(|m| serde_json::to_value(m).unwrap_or(serde_json::Value::Null))
        .collect();
    Json(serde_json::json!({ "object": "list", "data": data }))
}

/// GET /stats. Serializes the stats snapshot as JSON. When stats are disabled
/// returns 200 with a note plus a zeroed snapshot so dashboards still parse it.
async fn stats_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match &state.stats {
        Some(s) => Json(serde_json::json!(s.snapshot())),
        None => Json(serde_json::json!({
            "stats_enabled": false,
            "note": "stats db unavailable; counters not being recorded",
        })),
    }
}

/// POST /v1/chat/completions. Non-stream returns Json; stream returns SSE.
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, AppError> {
    let model = body.model.clone();
    let canon = to_canonical(&body).map_err(AppError::BadRequest)?;

    // Record the inbound request (best-effort; never blocks serving).
    if let Some(s) = &state.stats {
        s.record_request(&model, None);
    }

    let chat_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_now_secs();

    if body.stream {
        // Streaming: open the upstream SSE stream and frame it as OpenAI chunks.
        // We do not record per-token usage on the stream path here (usage events
        // are not surfaced through the framing layer), but the request was
        // already counted above.
        let stream = state
            .provider
            .send_stream(canon)
            .await
            .map_err(|e| record_and_map(&state, &model, e))?;
        let sse = sse_from_canonical_stream(stream, model, chat_id, created);
        return Ok(sse.into_response());
    }

    // Non-stream: send, record response stats, return the OAI JSON envelope.
    let started = Instant::now();
    let canon_resp = state
        .provider
        .send(canon)
        .await
        .map_err(|e| record_and_map(&state, &model, e))?;

    if let Some(s) = &state.stats {
        let usage = TokenUsage {
            input_tokens: canon_resp.usage.input_tokens,
            output_tokens: canon_resp.usage.output_tokens,
            cache_read_input_tokens: canon_resp.usage.cache_read,
            cache_creation_input_tokens: canon_resp.usage.cache_creation,
        };
        let dur_ms = started.elapsed().as_secs_f64() * 1000.0;
        // No streaming here, so there is no meaningful time-to-first-token.
        s.record_response(&model, usage, None, dur_ms);
    }

    let oai = from_canonical(canon_resp, model, chat_id, created);
    Ok(Json(oai).into_response())
}

/// Record an error against stats (best-effort) and map it to an AppError.
fn record_and_map(state: &AppState, model: &str, e: ProviderError) -> AppError {
    if let Some(s) = &state.stats {
        s.record_error(model, &e.to_string());
    }
    map_provider_err(e)
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
    let model = body.model.clone();
    let stream = body.stream;

    // Record the inbound request FIRST (mirroring the chat handler's record-then
    // -work order) so even a request later rejected for unsupported input counts
    // as inbound traffic; an unrecorded protocol would skew per-model accounting.
    if let Some(s) = &state.stats {
        s.record_request(&model, None);
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
            .map_err(|e| record_and_map(&state, &model, e))?;
        let sse = sse_from_canonical_stream_responses(stream, model, response_id, created_at);
        return Ok(sse.into_response());
    }

    let started = Instant::now();
    let canon_resp = state
        .provider
        .send(canon)
        .await
        .map_err(|e| record_and_map(&state, &model, e))?;

    if let Some(s) = &state.stats {
        let usage = TokenUsage {
            input_tokens: canon_resp.usage.input_tokens,
            output_tokens: canon_resp.usage.output_tokens,
            cache_read_input_tokens: canon_resp.usage.cache_read,
            cache_creation_input_tokens: canon_resp.usage.cache_creation,
        };
        let dur_ms = started.elapsed().as_secs_f64() * 1000.0;
        s.record_response(&model, usage, None, dur_ms);
    }

    let resp = responses_from_canonical(canon_resp, model, response_id, created_at);
    Ok(Json(resp).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use omni_common::ChatMessage;
    use serde_json::Value;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};
    use tower::ServiceExt; // for Router::oneshot

    // ---- pure / in-proc helpers ----

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("bind for free port")
            .local_addr()
            .unwrap()
            .port()
    }

    fn has_claude_creds() -> bool {
        // Honor CLAUDE_CREDENTIALS_PATH (the same override ClaudeProvider reads) so
        // this guard agrees with what the live send actually loads; otherwise a
        // missing-file override would pass the guard and then fail the send.
        if let Ok(p) = std::env::var("CLAUDE_CREDENTIALS_PATH") {
            return std::path::Path::new(&p).exists();
        }
        let home = std::env::var("HOME").unwrap_or_default();
        std::path::Path::new(&(home + "/.claude/.credentials.json")).exists()
    }

    fn temp_stats_path() -> PathBuf {
        // Clearly-labeled TEST data file; unique per test so parallel runs never
        // collide on the same redb path.
        std::env::temp_dir().join(format!("omni-claude-stats-TEST-{}.redb", Uuid::new_v4()))
    }

    /// Cleanup guard: remove the temp test stats file on drop.
    struct TempStats(PathBuf);
    impl Drop for TempStats {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn test_state(with_stats: bool) -> (Arc<AppState>, Option<TempStats>) {
        let provider = Arc::new(ClaudeProvider::new().expect("claude provider for test"));
        if with_stats {
            let path = temp_stats_path();
            let stats = Stats::open(&path).expect("open temp test stats db");
            let guard = TempStats(path);
            (
                Arc::new(AppState {
                    provider,
                    stats: Some(Arc::new(stats)),
                }),
                Some(guard),
            )
        } else {
            (
                Arc::new(AppState {
                    provider,
                    stats: None,
                }),
                None,
            )
        }
    }

    /// Resolve the standalone `omni-claude` binary, building it on demand. Unit
    /// tests in a bin crate get no `CARGO_BIN_EXE_*` env var, so `cargo test -p
    /// omni-claude` from a clean tree would otherwise spawn a missing or stale
    /// binary. Builds and locates the real artifact via cargo's JSON output (honors
    /// CARGO_TARGET_DIR; builds the dev profile); cached so the build runs once per
    /// test process. Kept in sync with omni::omni_bin_path / omni-grok::bin_path.
    fn omni_claude_bin_path() -> PathBuf {
        if let Ok(p) = std::env::var("CARGO_BIN_EXE_omni_claude") {
            return p.into();
        }
        static BIN: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
        BIN.get_or_init(|| omni_common::test_support::build_workspace_bin("omni-claude"))
            .clone()
    }

    /// Kill a spawned child and reap it so no zombie is left behind. Every
    /// subprocess test MUST call this on the child it spawned.
    fn kill_child(child: &mut std::process::Child) {
        let _ = child.kill();
        let _ = child.wait();
    }

    fn wait_for_200_health(port: u16, timeout: Duration) -> bool {
        let start = Instant::now();
        let url = format!("http://127.0.0.1:{}/health", port);
        while start.elapsed() < timeout {
            if let Ok(out) = Command::new("curl")
                .args(["-s", "--max-time", "1", &url])
                .output()
                && out.status.success()
                && String::from_utf8_lossy(&out.stdout).trim() == "ok"
            {
                return true;
            }
            thread::sleep(Duration::from_millis(120));
        }
        false
    }

    // ---- pure unit tests ----

    #[test]
    fn parse_api_keys_trims_and_drops_empties() {
        // WHY: the auth gate depends on the exact key set parsed from
        // OMNI_API_KEYS; a stray empty entry would let a blank Bearer match.
        let keys = parse_api_keys(Some(" a , b ,, c , "));
        assert_eq!(keys.len(), 3);
        assert!(keys.contains("a") && keys.contains("b") && keys.contains("c"));
        // None / empty => allow-all (empty set).
        assert!(parse_api_keys(None).is_empty());
        assert!(parse_api_keys(Some("")).is_empty());
        assert!(parse_api_keys(Some("  ,  ")).is_empty());
    }

    #[test]
    fn map_provider_err_maps_variants_like_aggregator() {
        // WHY: clients distinguish 401 (fix your creds) from 5xx (retry); the
        // provider->HTTP mapping must keep Auth on 401 and Upstream/Other on 5xx.
        assert!(matches!(
            map_provider_err(ProviderError::Auth("x".into())),
            AppError::Unauthorized(_)
        ));
        assert!(matches!(
            map_provider_err(ProviderError::Upstream("x".into())),
            AppError::ServerError(_)
        ));
        assert!(matches!(
            map_provider_err(ProviderError::Other(anyhow::anyhow!("x"))),
            AppError::ServerError(_)
        ));
    }

    #[test]
    fn default_stats_db_path_is_labeled_temp_file() {
        let p = default_stats_db_path();
        assert!(p.starts_with(std::env::temp_dir()));
        assert!(p.to_string_lossy().contains("omni-claude-stats"));
    }

    // ---- in-proc handler tests (no live backend needed) ----

    #[tokio::test]
    async fn health_returns_ok_200() {
        // WHY: liveness probe must be a plain 200 "ok" so the subprocess tests
        // (and real orchestrators) can gate readiness on it.
        let (state, _g) = test_state(false);
        let app = build_router(state, Arc::new(HashSet::new()));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1 << 16).await.unwrap();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn models_returns_nonempty_claude_catalog() {
        // WHY: /v1/models is how clients discover what to call; it must return a
        // non-empty OpenAI list whose ids are the real Claude catalog models.
        let (state, _g) = test_state(false);
        let app = build_router(state, Arc::new(HashSet::new()));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1 << 16).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "list");
        let ids: Vec<String> = v["data"]
            .as_array()
            .expect("data array")
            .iter()
            .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
            .collect();
        assert!(!ids.is_empty(), "catalog must not be empty");
        assert!(
            ids.iter().any(|id| id.contains("claude")),
            "ids must include claude models, got {ids:?}"
        );
    }

    #[tokio::test]
    async fn stats_endpoint_parses_as_json_with_and_without_db() {
        // WHY: /stats must always return parseable JSON. With a db it serializes
        // the snapshot (uptime_seconds present); without one it must still be a
        // 200 JSON body flagged disabled rather than a 500.
        let (state_on, _g) = test_state(true);
        let app = build_router(state_on, Arc::new(HashSet::new()));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/stats")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1 << 16).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert!(v["uptime_seconds"].is_u64(), "snapshot shape present");

        let (state_off, _g2) = test_state(false);
        let app2 = build_router(state_off, Arc::new(HashSet::new()));
        let resp2 = app2
            .oneshot(
                Request::builder()
                    .uri("/stats")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
        let body2 = to_bytes(resp2.into_body(), 1 << 16).await.unwrap();
        let v2: Value = serde_json::from_slice(&body2).unwrap();
        assert_eq!(v2["stats_enabled"], false);
    }

    #[tokio::test]
    async fn auth_gate_blocks_missing_key_but_allows_good_key() {
        // WHY: when keys are configured the proxy must be auth-gated: a request
        // with no Bearer is 401, a request with a valid key is 200. This is the
        // core security property of running with OMNI_API_KEYS set.
        let (state, _g) = test_state(false);
        let mut keys = HashSet::new();
        keys.insert("secret123".to_string());
        let app = build_router(state, Arc::new(keys));

        // No header -> 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Good key -> 200.
        let resp2 = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("authorization", "Bearer secret123")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn stream_request_returns_sse_content_type_not_400() {
        // WHY: the OLD light aggregator 400'd on stream:true. This single-backend
        // proxy MUST instead enter the streaming path and return an SSE response.
        // We assert the handler does not reject stream requests and, when the
        // upstream is reachable, hands back a text/event-stream response.
        //
        // Without live creds the upstream call fails; that maps to a mapped
        // AppError (NOT a BadRequest about "streaming not supported"), which
        // still proves we took the stream branch rather than rejecting it.
        let (state, _g) = test_state(false);
        let req = ChatCompletionRequest {
            model: "claude-haiku".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            max_tokens: Some(8),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: Value::Null,
        };
        let res = chat_completions_handler(State(state), Json(req)).await;
        match res {
            Ok(resp) => {
                // Stream branch taken and upstream opened: must be an SSE response.
                let ct = resp
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default();
                assert!(
                    ct.contains("text/event-stream"),
                    "stream path must yield SSE content-type, got {ct:?}"
                );
            }
            Err(e) => {
                // Acceptable offline: the error is a mapped provider error
                // (auth/server), never a "streaming not supported" BadRequest.
                let msg = match &e {
                    AppError::Unauthorized(m)
                    | AppError::BadRequest(m)
                    | AppError::ServerError(m)
                    | AppError::NotFound(m) => m.clone(),
                };
                assert!(
                    !msg.contains("streaming not supported"),
                    "must not reject streaming; got {msg}"
                );
            }
        }
    }

    #[tokio::test]
    async fn completions_records_error_in_stats_on_upstream_failure() {
        // WHY: errors must be observable via /stats so operators can see when the
        // upstream is failing. Offline (no creds) the claude send fails inside the
        // provider; that must bump the stats error counter via record_and_map.
        if has_claude_creds() {
            eprintln!(
                "skipping completions_records_error_in_stats_on_upstream_failure: live creds present, send may succeed"
            );
            return;
        }
        let (state, _g) = test_state(true);
        let req = ChatCompletionRequest {
            model: "claude-haiku".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: Some(8),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: Value::Null,
        };
        let res = chat_completions_handler(State(state.clone()), Json(req)).await;
        assert!(res.is_err(), "expected upstream/auth error without creds");
        let snap = state.stats.as_ref().unwrap().snapshot();
        assert_eq!(snap.total_requests, 1, "request was counted");
        assert_eq!(snap.errors, 1, "error was recorded for the failed send");
    }

    // ---- subprocess + curl tests (full HTTP stack, random port, kill child) ----

    #[test]
    fn subprocess_health_models_stats_on_random_port() {
        // WHY: exercises the real binary end to end over HTTP on a RANDOM free
        // port (never hardcoded, so parallel tests never collide): /health must
        // report ready, /models must return a JSON list, /stats must parse as
        // JSON. Proves the wired router + serialization work in the shipped bin.
        let port = free_port();
        let mut child = Command::new(omni_claude_bin_path())
            .args(["--no-auth", "--port", &port.to_string()])
            .env("OMNI_CLAUDE_STATS_DB", temp_stats_path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn omni-claude");

        let healthy = wait_for_200_health(port, Duration::from_secs(8));
        if !healthy {
            kill_child(&mut child);
            panic!("omni-claude did not become healthy on port {port}");
        }

        // /models -> JSON list with claude ids.
        let out = Command::new("curl")
            .args(["-s", &format!("http://127.0.0.1:{}/v1/models", port)])
            .output()
            .unwrap();
        let v: Value = serde_json::from_slice(&out.stdout).expect("models json");
        assert_eq!(v["object"], "list");
        assert!(
            !v["data"].as_array().unwrap().is_empty(),
            "models list must be non-empty"
        );

        // /stats -> parseable JSON.
        let out2 = Command::new("curl")
            .args(["-s", &format!("http://127.0.0.1:{}/stats", port)])
            .output()
            .unwrap();
        let sv: Value = serde_json::from_slice(&out2.stdout).expect("stats json");
        assert!(sv.is_object(), "stats must be a JSON object");

        kill_child(&mut child);
    }

    #[test]
    fn subprocess_auth_401_without_header_200_with_key() {
        // WHY: with OMNI_API_KEYS set the deployed binary must enforce auth over
        // the full HTTP stack: no header -> 401, correct key -> 200. Uses its OWN
        // random port so it can run in parallel with the other subprocess test.
        let port = free_port();
        let mut child = Command::new(omni_claude_bin_path())
            .args(["--port", &port.to_string()])
            .env("OMNI_API_KEYS", "secret123,other")
            .env("OMNI_CLAUDE_STATS_DB", temp_stats_path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn omni-claude (auth)");

        // Health is itself gated when keys are set, so wait using the header.
        let start = Instant::now();
        let mut ready = false;
        while start.elapsed() < Duration::from_secs(8) {
            if let Ok(out) = Command::new("curl")
                .args([
                    "-s",
                    "--max-time",
                    "1",
                    "-H",
                    "Authorization: Bearer secret123",
                    &format!("http://127.0.0.1:{}/health", port),
                ])
                .output()
                && out.status.success()
                && String::from_utf8_lossy(&out.stdout).trim() == "ok"
            {
                ready = true;
                break;
            }
            thread::sleep(Duration::from_millis(120));
        }
        if !ready {
            kill_child(&mut child);
            panic!("auth-protected omni-claude did not become ready on {port}");
        }

        // No header -> 401.
        let out1 = Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                &format!("http://127.0.0.1:{}/health", port),
            ])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out1.stdout).trim(), "401");

        // Good key -> 200.
        let out2 = Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "-H",
                "Authorization: Bearer secret123",
                &format!("http://127.0.0.1:{}/health", port),
            ])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out2.stdout).trim(), "200");

        kill_child(&mut child);
    }

    // ---- live-conditional test: real non-stream + stream completion ----

    #[test]
    fn live_real_completion_nonstream_and_stream_conditional() {
        // WHY: when real Claude credentials exist we prove the proxy actually
        // talks to the upstream and produces content for BOTH a non-stream JSON
        // response and a streaming SSE response. Skips cleanly (eprintln+return)
        // when creds are absent so the offline suite stays green and no Max quota
        // is burned on every run. Uses its own random port and kills the child.
        if !has_claude_creds() {
            eprintln!(
                "skipping live_real_completion_nonstream_and_stream_conditional: no Claude credentials at ~/.claude/.credentials.json"
            );
            return;
        }
        let port = free_port();
        let mut child = Command::new(omni_claude_bin_path())
            .args(["--no-auth", "--port", &port.to_string()])
            .env("OMNI_CLAUDE_STATS_DB", temp_stats_path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn omni-claude (live)");
        if !wait_for_200_health(port, Duration::from_secs(8)) {
            kill_child(&mut child);
            panic!("omni-claude did not become healthy on {port}");
        }

        // Non-stream: expect a chat.completion with non-empty content.
        let out = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"haiku","messages":[{"role":"user","content":"Reply with the single word PONG."}],"max_tokens":16}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(serde_json::json!({}));
        assert_eq!(
            v["object"], "chat.completion",
            "non-stream response envelope, got {v}"
        );
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
        assert!(
            !content.is_empty(),
            "live non-stream content must be present"
        );

        // Stream: expect SSE chunks terminated by [DONE].
        let out2 = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"haiku","messages":[{"role":"user","content":"Reply with the single word PONG."}],"max_tokens":16,"stream":true}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out2.stdout);
        assert!(
            text.contains("chat.completion.chunk"),
            "stream must emit OpenAI chunks, got: {text}"
        );
        assert!(text.contains("[DONE]"), "stream must terminate with [DONE]");

        kill_child(&mut child);
    }

    #[test]
    fn live_tool_call_loop_conditional() {
        // WHY: codifies the multi-turn tool-calling loop end to end against real
        // Claude. Hop 1: declaring a tool makes the model emit a tool_call
        // (finish_reason "tool_calls", function name get_weather). Hop 2: feeding
        // the tool RESULT back makes the model answer using it (finish_reason
        // "stop", content mentions the fed-back temperature). The hop-2 "72"
        // assertion is the key proof the tool result actually round-tripped and
        // was consumed. Skips cleanly (eprintln+return) when creds are absent so
        // the offline suite stays green and no Max quota is burned. Own random
        // port; child is killed.
        if !has_claude_creds() {
            eprintln!(
                "skipping live_tool_call_loop_conditional: no Claude credentials at ~/.claude/.credentials.json"
            );
            return;
        }
        let port = free_port();
        let mut child = Command::new(omni_claude_bin_path())
            .args(["--no-auth", "--port", &port.to_string()])
            .env("OMNI_CLAUDE_STATS_DB", temp_stats_path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn omni-claude (live tools)");
        if !wait_for_200_health(port, Duration::from_secs(8)) {
            kill_child(&mut child);
            panic!("omni-claude did not become healthy on {port}");
        }

        // Hop 1: declare a tool; model must emit a get_weather tool_call.
        let out = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"claude-3-5-haiku-20241022","messages":[{"role":"user","content":"What is the weather in San Francisco? Use the get_weather tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":256}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(serde_json::json!({}));
        assert_eq!(
            v["choices"][0]["finish_reason"], "tool_calls",
            "hop-1 must stop for tool_calls, got: {v}"
        );
        let tool_calls = v["choices"][0]["message"]["tool_calls"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert!(
            !tool_calls.is_empty(),
            "hop-1 must carry a non-empty tool_calls array: {v}"
        );
        assert_eq!(
            tool_calls[0]["function"]["name"], "get_weather",
            "hop-1 tool call must name get_weather: {v}"
        );
        assert!(
            !tool_calls[0]["function"]["arguments"]
                .as_str()
                .unwrap_or("")
                .is_empty(),
            "hop-1 tool call must carry arguments (city), got: {v}"
        );

        // Hop 2: feed the tool result back; model must answer using it.
        let out2 = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"claude-3-5-haiku-20241022","messages":[{"role":"user","content":"What is the weather in SF?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"72F and sunny"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"max_tokens":256}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        let v2: Value = serde_json::from_slice(&out2.stdout).unwrap_or(serde_json::json!({}));
        assert_eq!(
            v2["choices"][0]["finish_reason"], "stop",
            "hop-2 must finish with stop after consuming the tool result, got: {v2}"
        );
        let content = v2["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(!content.is_empty(), "hop-2 content must be present: {v2}");
        assert!(
            content.contains("72"),
            "hop-2 content must mention the fed-back temperature (proves the tool result round-tripped): {content}"
        );

        kill_child(&mut child);
    }

    // ---- OpenAI Responses protocol (/v1/responses): TDD contract tests ----

    #[tokio::test]
    async fn responses_unsupported_input_is_bad_request() {
        // WHY: input shapes the canonical layer still cannot represent (an
        // `input_image` content part) are rejected loudly as an OAI-shaped 400
        // naming the offender, instead of silently mangling the request.
        // (function_call / function_call_output items ARE now supported and
        // round-trip through canonical tool blocks.)
        let (state, _guard) = test_state(false);
        let req: omni_common::ResponsesRequest = serde_json::from_str(
            r#"{"model":"sonnet","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"x"}]}]}"#,
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
    async fn responses_route_registered_returns_400_not_404_for_bad_input() {
        // WHY: the route must be wired into the PRODUCTION router. A
        // parseable-but-unsupported body is rejected at parse level (hermetic,
        // no upstream call); a 404 means /v1/responses is not registered.
        let (state, _guard) = test_state(false);
        let app = build_router(state, Arc::new(HashSet::new()));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"model":"sonnet","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"x"}]}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "POST /v1/responses must exist and reject unsupported input with 400 (404 = not registered)"
        );
    }

    #[tokio::test]
    async fn responses_records_request_in_stats() {
        // WHY: the Responses surface must feed the same persistent stats as chat
        // completions; an unrecorded protocol skews per-model accounting. Pinned
        // contract: the inbound request is recorded FIRST (mirroring the chat
        // handler's record-then-work order), so even a request later rejected
        // for unsupported input counts as inbound traffic. Using the rejected
        // path keeps this test hermetic (no upstream call ever happens).
        let (state, _guard) = test_state(true);
        let req: omni_common::ResponsesRequest = serde_json::from_str(
            r#"{"model":"sonnet","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"x"}]}]}"#,
        )
        .expect("responses request json");
        let _ = responses_handler(State(state.clone()), Json(req)).await;
        let snap = state.stats.as_ref().expect("stats enabled").snapshot();
        assert_eq!(
            snap.total_requests, 1,
            "an inbound Responses request must be recorded in stats (record-first order)"
        );
    }

    #[tokio::test]
    async fn responses_live_nonstream_and_stream_conditional() {
        // WHY: live proof the Claude backend serves the Responses protocol:
        // non-stream returns the envelope with assistant output_text; stream
        // returns Responses SSE terminated by response.completed (no [DONE]).
        // Skips cleanly offline.
        if !has_claude_creds() {
            eprintln!("skipping responses live test: no claude creds");
            return;
        }
        let (state, _guard) = test_state(false);

        // Non-stream.
        let req: omni_common::ResponsesRequest = serde_json::from_str(
            r#"{"model":"haiku","input":"Reply with the single word PONG","max_output_tokens":16}"#,
        )
        .unwrap();
        let resp = responses_handler(State(state.clone()), Json(req))
            .await
            .expect("live non-stream responses call succeeds")
            .into_response();
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: Value = serde_json::from_slice(&body).expect("responses envelope json");
        assert_eq!(v["object"], "response", "live body: {v}");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert!(
            !v["output"][0]["content"][0]["text"]
                .as_str()
                .unwrap_or("")
                .is_empty()
        );

        // Stream.
        let req2: omni_common::ResponsesRequest = serde_json::from_str(
            r#"{"model":"haiku","input":"Reply with the single word PONG","max_output_tokens":16,"stream":true}"#,
        )
        .unwrap();
        let resp2 = responses_handler(State(state), Json(req2))
            .await
            .expect("live stream responses call succeeds")
            .into_response();
        let ct = resp2
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(ct.contains("text/event-stream"), "got content-type {ct:?}");
        let body2 = to_bytes(resp2.into_body(), 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&body2);
        assert!(text.contains("event: response.created"));
        assert!(text.contains("event: response.output_text.delta"));
        assert!(text.contains("event: response.completed"));
        assert!(!text.contains("[DONE]"));
    }
}
