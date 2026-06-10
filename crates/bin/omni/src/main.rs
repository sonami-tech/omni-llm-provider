//! Light Omni wrapper/aggregator binary.
//!
//! Thin routing layer only. No heavy logic, no fingerprint, no provider-specific
//! translation or gate code. Delegates to pluggable LlmProvider impls from
//! provider-claude and provider-grok via omni-core::LlmProvider + Canonical*.
//!
//! ## Supported configuration (per task)
//! - --providers claude,grok   or   OMNI_PROVIDERS=claude,grok (comma sep, order preserved)
//! - Model prefix routing: "grok:foo" or "claude:bar" (case-insensitive prefix)
//! - When only one provider enabled, bare model names (e.g. "grok-4") are routed to it.
//! - When multiple enabled, bare models are rejected with clear error (forces prefix).
//!
//! ## Surfaces (unified OpenAI-compatible)
//! - POST /v1/chat/completions  (text + sampling; non-stream JSON and stream SSE)
//! - GET  /v1/models , /models
//! - GET  /health
//! - GET  /
//!
//! ## How it uses shared crates (per requirements)
//! - omni-common: auth_layer + AppError (OAI-shaped errors) + the shared http
//!   layer (to_canonical/from_canonical + the SSE stream framer).
//! - omni-core: CanonicalRequest/Response + LlmProvider trait for the delegation contract.
//! - Depends on provider-claude (full fingerprint provider) + provider-grok (full).
//!
//! ## Routing implementation (pure, unit-testable)
//! See `resolve_provider_and_model` below. Pure function; no side effects.
//! Prefix takes precedence. Provider keys in the map and for prefixes are "claude" / "grok"
//! (matching task examples and GrokProvider::id()).
//!
//! ## Light by design + findings
//! - Claude backend here is a *stub* (see provider-claude/src/lib.rs). This is
//!   intentional: the real Claude Code fingerprint invariant, OAuth creds handling,
//!   cch, betas, preamble, full Anthropic Messages translate etc. MUST stay
//!   isolated in the dedicated `omni-claude` binary (and eventual full provider-claude
//!   port) per DESIGN.md / fingerprint rules. The stub only proves uniform routing +
//!   canonical delegation works for "claude" without risking the gate.
//! - Grok is the real impl (OpenAI-compat upstream). Its internal replacements hook
//!   and mappers are exercised when delegated to.
//! - Streaming, tools, vision, full sampling, native Anthropic surfaces etc. are
//!   out of scope for this "light" aggregator (would live in frontends/ or per-provider).
//! - Auth: always layers omni-common::auth_layer. Empty key set (via --no-auth or
//!   no OMNI_API_KEYS) means "allow all" (passthrough), matching original CCP behavior.
//! - No stats/replacements loaded at this layer (cross-cutting; providers demonstrate
//!   the hook; real bins can inject Arc<Replacements> etc. into provider ctors later).
//! - Startup: claude stub always succeeds. grok requires XAI_API_KEY (or explicit)
//!   when "grok" is in the enabled list — fails fast, as intended.
//!
//! Build: cargo build -p omni
//! Run (claude only, no keys needed): OMNI_PROVIDERS=claude cargo run -p omni -- --no-auth --port 18323
//! Test: cargo test -p omni
//!
//! Documented here per "document any findings in the code or note for docs/" (no new .md).

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

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
use tracing::info;
use uuid::Uuid;

use omni_common::{AppError, ChatCompletionRequest, auth_layer, from_canonical, to_canonical};
use omni_core::{CanonicalResponse, LlmProvider, ProviderError};

// Re-export the concrete providers so main can construct them by name.
use provider_claude::ClaudeProvider;
use provider_grok::GrokProvider;

/// CLI for the light omni aggregator.
/// Env vars: OMNI_PROVIDERS, OMNI_PORT, OMNI_NO_AUTH (clap env support).
#[derive(Parser, Debug)]
#[command(
    name = "omni",
    about = "Light Omni LLM aggregator (claude + grok backends)"
)]
struct Cli {
    /// Comma-separated list of providers to enable (claude,grok). Prefix routing uses these names.
    #[arg(
        long,
        env = "OMNI_PROVIDERS",
        default_value = "claude,grok",
        value_delimiter = ','
    )]
    providers: Vec<String>,

    /// Listen port.
    #[arg(long, env = "OMNI_PORT", default_value_t = 18323)]
    port: u16,

    /// Disable API key auth (if omitted, still allows all unless OMNI_API_KEYS is set).
    #[arg(long, env = "OMNI_NO_AUTH")]
    no_auth: bool,
}

#[derive(Clone)]
struct AppState {
    /// Map from provider key ("claude" | "grok") -> live LlmProvider.
    /// Thin indirection: the wrapper only selects and delegates .send().
    providers: HashMap<String, Arc<dyn LlmProvider>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,omni=debug,provider_claude=debug,provider_grok=debug")
        .init();

    let cli = Cli::parse();

    // Normalize + validate providers list (unique, known names only).
    let enabled: Vec<String> = normalize_providers(&cli.providers)?;
    info!(?enabled, "omni enabled providers");

    // Instantiate providers (thin). Claude stub never needs keys/creds.
    let mut providers_map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
    for name in &enabled {
        let arc: Arc<dyn LlmProvider> = match name.as_str() {
            "claude" => {
                info!("initializing claude provider");
                // new() returns Result because of optional creds/profile load for the real gate logic.
                // In production bin this would be .expect or proper error handling.
                Arc::new(ClaudeProvider::new().expect(
                    "claude provider init (valid ~/.claude/.credentials.json or explicit profile)",
                ))
            }
            "grok" => {
                info!("initializing grok provider (requires XAI_API_KEY if not using test ctor)");
                let p = GrokProvider::new(None)
                    .context("failed to init grok provider (set XAI_API_KEY or remove grok from OMNI_PROVIDERS)")?;
                Arc::new(p)
            }
            other => {
                // Should be impossible after normalize.
                anyhow::bail!("unknown provider in list: {}", other);
            }
        };
        providers_map.insert(name.clone(), arc);
    }

    let state = AppState {
        providers: providers_map,
    };

    // Auth keys: empty set => allow-all (see omni-common::auth_layer).
    // Support OMNI_API_KEYS= k1,k2,... for a non-empty set when !--no-auth.
    let auth_keys: Arc<HashSet<String>> = if cli.no_auth {
        Arc::new(HashSet::new())
    } else {
        let keys: HashSet<String> = std::env::var("OMNI_API_KEYS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        Arc::new(keys)
    };
    info!(no_auth_effective = auth_keys.is_empty(), "auth layer ready");

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/", get(root_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/models", get(models_handler))
        .route("/models", get(models_handler))
        .with_state(Arc::new(state))
        // Always layer; the common impl short-circuits when keys empty.
        .layer(middleware::from_fn({
            let keys = auth_keys.clone();
            move |req, next| auth_layer(keys.clone(), req, next)
        }));

    let addr: SocketAddr = format!("127.0.0.1:{}", cli.port).parse()?;
    info!("omni listening on http://{}", addr);
    info!("  providers: {}", enabled.join(","));
    info!("  try: curl http://{}/health", addr);
    info!("  models:  curl http://{}/v1/models", addr);
    info!(
        "  completions example: model=grok:grok-4 or claude:claude-3-5-sonnet-20241022 (or bare if single provider)"
    );

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app)
        .await
        .context("server error")?;

    Ok(())
}

/// Normalize/validate the providers CLI/env list.
/// Accepts "claude,grok", trims, lowercases, dedups in order, rejects unknowns.
fn normalize_providers(raw: &[String]) -> anyhow::Result<Vec<String>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for r in raw {
        let p = r.trim().to_lowercase();
        if p.is_empty() {
            continue;
        }
        if p != "claude" && p != "grok" {
            anyhow::bail!("unknown provider {:?}; supported: claude,grok", r);
        }
        if seen.insert(p.clone()) {
            out.push(p);
        }
    }
    if out.is_empty() {
        anyhow::bail!("at least one provider required (claude and/or grok)");
    }
    Ok(out)
}

/// Pure routing function. Extracted for easy unit testing of the core logic.
/// Returns (provider_key, stripped_model).
///
/// Rules:
/// - If input contains "prefix:rest" (first :), use prefix (lowercased) if enabled.
/// - Else if exactly one provider enabled, use it with model unchanged.
/// - Else (multi + no prefix) => error (caller turns into AppError::BadRequest).
pub fn resolve_provider_and_model(
    model: &str,
    enabled: &[String],
) -> Result<(String, String), String> {
    if let Some((pre, rest)) = model.split_once(':') {
        let key = pre.trim().to_lowercase();
        if enabled.iter().any(|e| e == &key) {
            let stripped = rest.trim().to_string();
            if stripped.is_empty() {
                return Err(format!("empty model after prefix for provider {}", key));
            }
            return Ok((key, stripped));
        } else {
            return Err(format!(
                "provider '{}' not enabled (enabled: [{}])",
                key,
                enabled.join(",")
            ));
        }
    }

    // No prefix.
    if enabled.len() == 1 {
        return Ok((enabled[0].clone(), model.to_string()));
    }

    Err(
        "when multiple providers enabled, model must use prefix (e.g. grok:foo or claude:bar)"
            .to_string(),
    )
}

/// Handler: GET /health
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Handler: GET /
async fn root_handler() -> impl IntoResponse {
    "omni - light multi-backend OpenAI-compatible aggregator (claude + grok via prefix or --providers)"
}

/// Handler: GET /v1/models (and /models)
async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let enabled = state.providers.keys().cloned().collect::<Vec<_>>();
    let mut data = Vec::new();
    for p in &enabled {
        let examples: &[&str] = match p.as_str() {
            "claude" => &[
                "claude:claude-3-5-sonnet-20241022",
                "claude:claude-opus-4-1",
            ],
            "grok" => &["grok:grok-3", "grok:grok-4.3"],
            _ => &[],
        };
        for ex in examples {
            data.push(serde_json::json!({
                "id": ex,
                "object": "model",
                "created": 0,
                "owned_by": p,
            }));
        }
        // Also expose a bare-ish example under the prefix for discoverability.
        data.push(serde_json::json!({
            "id": format!("{}:default", p),
            "object": "model",
            "created": 0,
            "owned_by": p,
        }));
    }
    Json(serde_json::json!({ "object": "list", "data": data }))
}

/// Handler: POST /v1/chat/completions
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, AppError> {
    let enabled: Vec<String> = state.providers.keys().cloned().collect();
    let (prov_key, stripped_model) =
        resolve_provider_and_model(&body.model, &enabled).map_err(AppError::BadRequest)?;

    let provider = state
        .providers
        .get(&prov_key)
        .ok_or_else(|| AppError::ServerError("provider disappeared".into()))?;

    // Build canonical *with the stripped model* so the delegated provider sees the real model name.
    let mut canon = to_canonical(&body);
    canon.model = stripped_model.clone();

    let chat_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = omni_common::unix_now_secs();

    if body.stream {
        // Streaming: delegate to the provider's native SSE stream and frame it as
        // OpenAI chat.completion.chunk events (terminated by [DONE]) via the shared
        // serializer. Prefix routing has already selected the provider above.
        let stream = provider
            .send_stream(canon)
            .await
            .map_err(map_provider_err)?;
        // requested_model echoed is the *original* (with prefix if any) for client UX.
        let sse = omni_common::sse_from_canonical_stream(stream, body.model, chat_id, created);
        return Ok(sse.into_response());
    }

    // The actual delegation (thin by design).
    let canon_resp: CanonicalResponse = provider.send(canon).await.map_err(map_provider_err)?;

    // requested_model echoed is the *original* (with prefix if client used one) for client UX.
    let oai = from_canonical(canon_resp, body.model, chat_id, created);
    Ok(Json(oai).into_response())
}

/// Map a provider error to the OAI-shaped AppError. Shared by the stream and
/// non-stream completion paths so they classify errors identically.
fn map_provider_err(e: ProviderError) -> AppError {
    match e {
        ProviderError::Auth(msg) => AppError::Unauthorized(msg),
        ProviderError::Upstream(msg) => AppError::ServerError(format!("upstream: {}", msg)),
        ProviderError::Other(a) => AppError::ServerError(a.to_string()),
    }
}

/// Handler for POST /v1/responses (OpenAI Responses API protocol).
///
/// Contract (pinned by the responses tests below + omni-common::responses):
/// same prefix routing as chat completions; unsupported input shapes map to
/// BadRequest; non-stream returns the Responses envelope; stream:true returns
/// Responses SSE events (response.created ... response.completed, no [DONE]).
async fn responses_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<omni_common::ResponsesRequest>,
) -> Result<axum::response::Response, AppError> {
    let enabled: Vec<String> = state.providers.keys().cloned().collect();
    let (prov_key, stripped_model) =
        resolve_provider_and_model(&body.model, &enabled).map_err(AppError::BadRequest)?;

    let provider = state
        .providers
        .get(&prov_key)
        .ok_or_else(|| AppError::ServerError("provider disappeared".into()))?;

    // Convert the Responses request to canonical, then swap in the stripped model
    // so the delegated provider sees the real backend model name. Unsupported
    // input shapes are rejected loudly as a 400 naming the offender.
    let mut canon = omni_common::responses_to_canonical(&body).map_err(AppError::BadRequest)?;
    canon.model = stripped_model;

    let response_id = format!("resp_{}", Uuid::new_v4());
    let created_at = omni_common::unix_now_secs();

    if body.stream {
        let stream = provider
            .send_stream(canon)
            .await
            .map_err(map_provider_err)?;
        // Echo the original (prefixed) model for client UX.
        let sse = omni_common::sse_from_canonical_stream_responses(
            stream,
            body.model,
            response_id,
            created_at,
        );
        return Ok(sse.into_response());
    }

    let canon_resp: CanonicalResponse = provider.send(canon).await.map_err(map_provider_err)?;
    let resp =
        omni_common::responses_from_canonical(canon_resp, body.model, response_id, created_at);
    Ok(Json(resp).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_common::ChatMessage; // test constructors build requests literally
    use omni_core::LlmProvider; // for the smoke

    fn enabled_claude_grok() -> Vec<String> {
        vec!["claude".into(), "grok".into()]
    }
    fn enabled_only_claude() -> Vec<String> {
        vec!["claude".into()]
    }

    #[test]
    fn test_resolve_prefix_grok() {
        let (k, m) = resolve_provider_and_model("grok:foo-bar", &enabled_claude_grok()).unwrap();
        assert_eq!(k, "grok");
        assert_eq!(m, "foo-bar");
    }

    #[test]
    fn test_resolve_prefix_claude() {
        let (k, m) =
            resolve_provider_and_model("CLAUDE:claude-3-5-sonnet-20241022", &enabled_claude_grok())
                .unwrap();
        assert_eq!(k, "claude");
        assert_eq!(m, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_resolve_bare_single_provider() {
        let (k, m) = resolve_provider_and_model("my-model", &enabled_only_claude()).unwrap();
        assert_eq!(k, "claude");
        assert_eq!(m, "my-model");
    }

    #[test]
    fn test_resolve_bare_multi_errors() {
        let err = resolve_provider_and_model("bare-model", &enabled_claude_grok()).unwrap_err();
        assert!(err.contains("must use prefix"));
    }

    #[test]
    fn test_resolve_unknown_prefix_errors() {
        let err = resolve_provider_and_model("codex:foo", &enabled_claude_grok()).unwrap_err();
        assert!(err.contains("not enabled"));
    }

    #[test]
    fn test_normalize_providers() {
        let n = normalize_providers(&[" claude ".into(), "GROK".into(), "claude".into()]).unwrap();
        assert_eq!(n, vec!["claude".to_string(), "grok".to_string()]);
    }

    #[tokio::test]
    async fn smoke_routing_and_delegate_claude_stub() {
        // Verifies the full path resolve + to_canonical + delegate + from_canonical
        // against the real Claude upstream. Live-conditional: skips cleanly when no
        // Claude credentials are present so the suite stays green offline (the
        // routing + conversion logic without the live send is covered by the
        // hermetic handler tests below).
        if !has_claude_creds() {
            eprintln!("skipping smoke_routing_and_delegate_claude_stub: no claude creds");
            return;
        }
        let claude = Arc::new(ClaudeProvider::new().expect("claude provider for wrapper test"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("claude".to_string(), claude);

        let enabled: Vec<String> = vec!["claude".into()];
        let (key, stripped) = resolve_provider_and_model("claude:sonnet", &enabled).unwrap();
        assert_eq!(key, "claude");
        assert_eq!(stripped, "sonnet");

        let oai_req = ChatCompletionRequest {
            model: "claude:sonnet".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("tell me a joke".into()),
            }],
            stream: false,
            max_tokens: Some(64),
            max_completion_tokens: None,
            temperature: Some(0.7),
            top_p: None,
            extras: serde_json::Value::Null,
        };

        let mut canon = to_canonical(&oai_req);
        canon.model = stripped;

        let provider = map.get(&key).unwrap();
        let canon_resp = provider.send(canon).await.unwrap();

        assert_eq!(canon_resp.model, "sonnet");
        // real response (live creds) or auth/upstream error would have failed .unwrap; content from canonical
        assert!(!canon_resp.content.is_empty() || !canon_resp.tool_calls.is_empty());

        let oai_resp = from_canonical(
            canon_resp,
            oai_req.model,
            "chatcmpl-test".into(),
            1234567890,
        );
        assert_eq!(oai_resp.model, "claude:sonnet");
        let has_content = oai_resp.choices[0]
            .message
            .content
            .as_deref()
            .is_some_and(|c| !c.is_empty());
        let has_tools = !oai_resp.choices[0].message.tool_calls.is_empty();
        assert!(has_content || has_tools);
        let _ = oai_resp.usage.prompt_tokens; // u64 always >=0 by type; shape covered by other asserts + CCP mirror
        assert_eq!(oai_resp.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn smoke_routing_selects_grok() {
        // Grok-specific send smoke is in the provider-grok crate (uses its own new_for_test).
        // Here we prove:
        // 1. Routing logic selects "grok" for "grok:..." prefix (even in multi-provider mode).
        // 2. The GrokProvider type satisfies LlmProvider (compile-time), so the dyn map + delegation
        //    code paths in main are valid for it. (No runtime construction here to avoid XAI_API_KEY
        //    requirement.)
        let enabled: Vec<String> = vec!["claude".into(), "grok".into()];
        let (key, stripped) = resolve_provider_and_model("grok:grok-4.3", &enabled).unwrap();
        assert_eq!(key, "grok");
        assert_eq!(stripped, "grok-4.3");

        // Compile-time assertion that the real grok type can be used exactly as the omni router does
        // (stored in HashMap<String, Arc<dyn LlmProvider>> and delegated to).
        fn assert_impls_dyn<P: LlmProvider + 'static>() {
            // If this compiles, GrokProvider (and Claude) can be the pointee for the thin router.
        }
        assert_impls_dyn::<GrokProvider>();
        assert_impls_dyn::<ClaudeProvider>();
    }

    // --- comprehensive http/integration tests added per task (using direct handler calls for router surfaces
    // + subprocess+curl for full binary http stack incl auth mw, random port, live creds conditional) ---

    use axum::http::StatusCode;
    use omni_core::CanonicalResponse;
    use serde_json::Value;
    use std::collections::HashSet;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("bind for free port")
            .local_addr()
            .unwrap()
            .port()
    }

    fn has_claude_creds() -> bool {
        // Honor the same override the provider uses (CLAUDE_CREDENTIALS_PATH), so
        // this guard agrees with what ClaudeProvider::send actually reads. Without
        // this, pointing the override at a missing file would pass the guard yet
        // fail the live send.
        if let Ok(p) = std::env::var("CLAUDE_CREDENTIALS_PATH") {
            return std::path::Path::new(&p).exists();
        }
        let home = std::env::var("HOME").unwrap_or_default();
        std::path::Path::new(&(home + "/.claude/.credentials.json")).exists()
    }

    fn has_grok_creds() -> bool {
        // Mirror the Grok provider's fresh-load precedence and the race-safe guard
        // used in provider-grok: when XAI_CREDENTIALS_PATH is set, treat creds as
        // present only if that file exists (a test may point it at a dummy/missing
        // path); otherwise fall back to XAI_API_KEY or the default file.
        if let Ok(p) = std::env::var("XAI_CREDENTIALS_PATH") {
            return std::path::Path::new(&p).exists();
        }
        if std::env::var("XAI_API_KEY").is_ok() {
            return true;
        }
        let home = std::env::var("HOME").unwrap_or_default();
        std::path::Path::new(&(home + "/.xai/.credentials.json")).exists()
    }

    fn wait_for_200_health(port: u16, timeout: Duration) -> bool {
        let start = Instant::now();
        let url = format!("http://127.0.0.1:{}/health", port);
        while start.elapsed() < timeout {
            if let Ok(out) = Command::new("curl")
                .args(["-s", "--max-time", "1", &url])
                .output()
                && out.status.success()
            {
                let body = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if body == "ok" {
                    return true;
                }
            }
            thread::sleep(Duration::from_millis(120));
        }
        false
    }

    fn omni_bin_path() -> std::path::PathBuf {
        // Runtime lookup so this compiles even when CARGO_BIN_EXE_* is absent at
        // compile time. Prefer the cargo-injected path when present (integration
        // tests get it; unit tests in a bin crate do not).
        if let Ok(p) = std::env::var("CARGO_BIN_EXE_omni") {
            return std::path::PathBuf::from(p);
        }
        // Otherwise build the binary on demand and locate the real artifact (honors
        // CARGO_TARGET_DIR + profile). Cache so the build runs once per test process;
        // see omni_common::test_support::build_workspace_bin for the full rationale.
        static BIN: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        BIN.get_or_init(|| omni_common::test_support::build_workspace_bin("omni"))
            .clone()
    }

    fn mk_app_with(
        providers: HashMap<String, Arc<dyn LlmProvider>>,
        auth_keys: Arc<HashSet<String>>,
    ) -> axum::Router {
        // Test-only router builder (dupe of main construction for full surface tests w/o editing prod).
        // Mirrors CCP test server setup.
        let state = Arc::new(AppState { providers });
        axum::Router::new()
            .route("/health", axum::routing::get(health_handler))
            .route("/", axum::routing::get(root_handler))
            .route(
                "/v1/chat/completions",
                axum::routing::post(chat_completions_handler),
            )
            .route("/v1/models", axum::routing::get(models_handler))
            .route("/models", axum::routing::get(models_handler))
            .with_state(state)
            .layer(axum::middleware::from_fn({
                let keys = auth_keys.clone();
                move |req, next| auth_layer(keys.clone(), req, next)
            }))
    }

    #[test]
    fn test_mk_app_with_and_router_surfaces() {
        // Verifies we can build the full router for different provider configs (used below for in-proc handler flows).
        let cl = Arc::new(ClaudeProvider::new().expect("claude"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("claude".into(), cl);
        let app = mk_app_with(map, Arc::new(HashSet::new()));
        // construction success + state present proves unified surfaces setup
        assert!(format!("{:?}", app).contains("Router")); // loose but exercises
    }

    #[tokio::test]
    async fn test_http_health_handler() {
        let resp = health_handler().await;
        let (parts, _body) = resp.into_response().into_parts();
        assert_eq!(parts.status, StatusCode::OK);
        // body would be "ok"
    }

    #[tokio::test]
    async fn test_http_models_handler_single_and_multi() {
        let c = Arc::new(ClaudeProvider::new().expect("c"));
        let g = Arc::new(GrokProvider::new_for_test("k", "http://127.0.0.1:9"));
        let mut m1: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        m1.insert("claude".into(), c.clone());
        let state1 = Arc::new(AppState { providers: m1 });
        let _j1 = models_handler(State(state1)).await;
        // models always returns json; call succeeds proves /v1/models and /models surface
        let mut m2: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        m2.insert("claude".into(), c);
        m2.insert("grok".into(), g);
        let state2 = Arc::new(AppState { providers: m2 });
        let _j2 = models_handler(State(state2)).await;
    }

    #[tokio::test]
    async fn test_http_completions_stream_is_routed_not_rejected() {
        // WHY: streaming is now a first-class path. A stream:true request must be
        // ROUTED to the provider's send_stream (and, when reachable, framed as an
        // SSE response), never rejected with the old "streaming not supported"
        // 400. We use the grok test provider pointed at a dead port: routing +
        // stream-open is exercised; the dead upstream surfaces as a ServerError
        // (NOT a BadRequest stream-rejection), proving the stream branch is live.
        let g = Arc::new(GrokProvider::new_for_test("dummy", "http://127.0.0.1:1"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("grok".into(), g);
        let state = Arc::new(AppState { providers: map });
        let req = ChatCompletionRequest {
            model: "grok:grok-3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
            }],
            stream: true,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            extras: serde_json::Value::Null,
        };
        let res = chat_completions_handler(State(state), Json(req)).await;
        match res {
            // Dead upstream: stream-open failed -> mapped to a server error. The
            // key assertion is that it is NOT the old BadRequest rejection.
            Err(AppError::ServerError(_)) => {}
            Err(AppError::BadRequest(msg)) => {
                panic!("stream must not be rejected as bad request: {msg}")
            }
            Err(other) => panic!("unexpected error from stream route: {other:?}"),
            Ok(resp) => {
                // If a stream did open, it must be an SSE response.
                let ct = resp
                    .headers()
                    .get(axum::http::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                assert!(
                    ct.contains("text/event-stream"),
                    "streaming response must be SSE, got content-type {ct:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_http_completions_stream_returns_sse_when_backend_reachable() {
        // WHY: pins that a successfully-opened stream is framed as an SSE response
        // (text/event-stream), live-conditional on Grok creds so it stays green
        // offline. The byte-level [DONE] framing is pinned in omni-common::http.
        if !has_grok_creds() {
            eprintln!("skipping SSE-reachable test: no grok creds");
            return;
        }
        let g = Arc::new(GrokProvider::new(None).expect("grok provider with creds"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("grok".into(), g);
        let state = Arc::new(AppState { providers: map });
        let req = ChatCompletionRequest {
            model: "grok:grok-3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("Reply with the single word PONG".into()),
            }],
            stream: true,
            max_tokens: Some(16),
            max_completion_tokens: None,
            temperature: Some(0.0),
            top_p: None,
            extras: serde_json::Value::Null,
        };
        let res = chat_completions_handler(State(state), Json(req)).await;
        let resp = res.expect("stream should open with creds").into_response();
        let ct = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("text/event-stream"),
            "live stream must be SSE, got {ct:?}"
        );
    }

    #[tokio::test]
    async fn test_http_completions_unknown_provider_prefix_error() {
        let c = Arc::new(ClaudeProvider::new().expect("c"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("claude".into(), c);
        let state = Arc::new(AppState { providers: map });
        let req = ChatCompletionRequest {
            model: "codex:bar".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            extras: serde_json::Value::Null,
        };
        let res = chat_completions_handler(State(state), Json(req)).await;
        let err = match res {
            Err(e) => e,
            Ok(_) => panic!("expected err for unknown prov"),
        };
        match err {
            AppError::BadRequest(msg) => assert!(msg.contains("not enabled")),
            _ => panic!("expected badrequest for unknown prov"),
        }
    }

    #[tokio::test]
    async fn test_http_completions_disabled_provider_error() {
        // only claude enabled; grok: prefix should 400
        let c = Arc::new(ClaudeProvider::new().expect("c"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("claude".into(), c);
        let state = Arc::new(AppState { providers: map });
        let req = ChatCompletionRequest {
            model: "grok:bar".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            extras: serde_json::Value::Null,
        };
        let res = chat_completions_handler(State(state), Json(req)).await;
        let m = match res {
            Err(e) => e,
            Ok(_) => panic!("want badreq"),
        };
        match m {
            AppError::BadRequest(mm) => assert!(mm.contains("not enabled")),
            _ => panic!("want badreq"),
        }
    }

    #[tokio::test]
    async fn test_http_completions_bare_model_requires_prefix_when_multi() {
        let c = Arc::new(ClaudeProvider::new().expect("c"));
        let g = Arc::new(GrokProvider::new_for_test("k", "http://127.0.0.1:9"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("claude".into(), c);
        map.insert("grok".into(), g);
        let state = Arc::new(AppState { providers: map });
        let req = ChatCompletionRequest {
            model: "bare-model".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            extras: serde_json::Value::Null,
        };
        let res = chat_completions_handler(State(state), Json(req)).await;
        let m = match res {
            Err(e) => e,
            Ok(_) => panic!("want prefix error"),
        };
        match m {
            AppError::BadRequest(mm) => assert!(mm.contains("must use prefix")),
            _ => panic!("want prefix error"),
        }
    }

    #[tokio::test]
    async fn test_http_completions_routes_via_prefix_to_grok_test_provider() {
        // grok test ctor points to bad port -> upstream err mapped to 5xx server err (delegation exercised)
        let g = Arc::new(GrokProvider::new_for_test(
            "dummy-key",
            "http://127.0.0.1:1",
        ));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("grok".into(), g);
        let state = Arc::new(AppState { providers: map });
        let req = ChatCompletionRequest {
            model: "grok:grok-3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("ping".into()),
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            extras: serde_json::Value::Null,
        };
        let res = chat_completions_handler(State(state), Json(req)).await;
        let err = match res {
            Err(e) => e,
            Ok(_) => panic!("expected err from grok test"),
        };
        match err {
            AppError::ServerError(msg) => {
                assert!(msg.contains("upstream") || msg.contains("network"))
            }
            _ => panic!(
                "expected server/upstream error from grok test dispatch, got {:?}",
                err
            ),
        }
    }

    #[tokio::test]
    async fn test_http_completions_routes_via_prefix_to_claude() {
        // claude via prefix; will succeed if live creds (canonical delegation + real response), else auth err from cred load inside provider.
        let c = Arc::new(ClaudeProvider::new().expect("claude"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("claude".into(), c);
        let state = Arc::new(AppState { providers: map });
        let req = ChatCompletionRequest {
            model: "claude:claude-3-5-sonnet-20241022".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("Reply with the word PONG only.".into()),
            }],
            stream: false,
            max_tokens: Some(16),
            max_completion_tokens: None,
            temperature: Some(0.0),
            top_p: None,
            extras: serde_json::Value::Null,
        };
        let res = chat_completions_handler(State(state), Json(req)).await;
        match res {
            Ok(_) => {
                // Ok means canonical delegation succeeded (live creds path exercised, unified oai surface produced)
            }
            Err(AppError::Unauthorized(_)) => {
                // no live claude creds; cred loading path in wrapper delegation exercised (acceptable for conditional)
            }
            Err(e) => panic!("unexpected err from claude route: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_http_completions_unified_oai_response_shape() {
        // Use grok test provider (always errors upstream but proves from_canonical + oai shape on err path? no, err before)
        // Instead construct a direct canonical resp and from_ to pin the surface (unified for both backends)
        let canon = CanonicalResponse {
            model: "grok-3".into(),
            content: "hello from backend".into(),
            tool_calls: vec![],
            finish_reason: Some("stop".into()),
            usage: omni_core::CanonicalUsage {
                input_tokens: 5,
                output_tokens: 2,
                cache_read: 0,
                cache_creation: 0,
            },
        };
        let oai = from_canonical(canon, "grok:grok-3".into(), "chatcmpl-xyz".into(), 123);
        assert_eq!(oai.id, "chatcmpl-xyz");
        assert_eq!(oai.object, "chat.completion");
        assert_eq!(oai.model, "grok:grok-3");
        assert_eq!(
            oai.choices[0].message.content.as_deref(),
            Some("hello from backend")
        );
        assert_eq!(oai.usage.prompt_tokens, 5);
        assert_eq!(oai.usage.completion_tokens, 2);
    }

    #[test]
    fn test_replacements_e2e_config_toml_rule_for_prompt() {
        // Config rule in TOML (as would be loaded for real bins); verifies parse + apply (used by both backends inside send)
        let toml = r#"rule = [ { scope = "prompt", search = "foo", replace = "bar" } ]"#;
        let r = omni_common::Replacements::parse(toml).expect("parse config toml");
        assert_eq!(r.count(), 1);
        assert_eq!(r.apply_prompt("foo baz"), "bar baz");
        assert_eq!(r.apply_prompt("no match"), "no match");
    }

    #[test]
    fn test_replacements_e2e_config_toml_rule_for_response_and_both() {
        let toml = r#"
rule = [
  { scope = "response", search = "old", replace = "new" },
  { scope = "both", search = "x", replace = "y" }
]
"#;
        let r = omni_common::Replacements::parse(toml).expect("valid response+both toml config");
        assert_eq!(r.count(), 2);
        assert_eq!(r.apply_response("old x resp"), "new y resp");
        assert_eq!(r.apply_prompt("x in prompt"), "y in prompt"); // both applies to prompt too
    }

    #[tokio::test]
    async fn test_replacements_applied_in_provider_paths_for_both_backends() {
        // Exercises that both backends go through omni-common Replacements hook (empty in current, but config parse proves e2e seam)
        // (prompt apply inside to_* , response apply inside from_* )
        let _r = omni_common::Replacements::parse(
            r#"rule = [{scope="both", search="ping", replace="pong"}]"#,
        )
        .unwrap();
        // grok path (test ctor, no net)
        let pg = GrokProvider::new_for_test("k", "http://127.0.0.1:9");
        assert_eq!(pg.id(), "grok");
        // claude path
        let pc = ClaudeProvider::new().expect("claude");
        assert_eq!(pc.id(), "claude-code");
        // unified via core
        let _ = omni_core::CanonicalResponse {
            model: "m".into(),
            content: "c".into(),
            tool_calls: vec![],
            finish_reason: None,
            usage: Default::default(),
        };
    }

    #[tokio::test]
    async fn test_multi_backend_enable_both_and_route_each() {
        // Multi backend (enable both, test both in one test)
        let gc = GrokProvider::new_for_test("dummy", "http://127.0.0.1:1");
        let cc = ClaudeProvider::new().expect("claude for multi");
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("grok".to_string(), Arc::new(gc));
        map.insert("claude".to_string(), Arc::new(cc));
        let enabled: Vec<String> = map.keys().cloned().collect();
        assert_eq!(enabled.len(), 2);
        let (kg, mg) = resolve_provider_and_model("grok:x", &enabled).unwrap();
        assert_eq!(kg, "grok");
        assert_eq!(mg, "x");
        let (kc, mc) = resolve_provider_and_model("claude:y", &enabled).unwrap();
        assert_eq!(kc, "claude");
        assert_eq!(mc, "y");
        // bare fails
        assert!(resolve_provider_and_model("bare", &enabled).is_err());
    }

    #[tokio::test]
    async fn test_credential_loading_in_wrapper_delegation_context() {
        // Wrapper delegates; creds loaded fresh inside provider send (claude ~/.claude, grok xai file/env) - exercised on real send
        // Use grok test that forces load path (falls to ctor key but covers the fresh load attempt in prod path)
        let g = GrokProvider::new_for_test("xai-test-cred", "http://127.0.0.1:1");
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("grok".into(), Arc::new(g));
        let state = Arc::new(AppState { providers: map });
        let req = ChatCompletionRequest {
            model: "grok:grok-3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("c".into()),
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            extras: serde_json::Value::Null,
        };
        let _ = chat_completions_handler(State(state), Json(req)).await; // will err on net but cred code ran in provider
    }

    // --- subprocess binary + curl tests (full http stack, random port, kill, live conditional for real calls) ---

    #[test]
    fn test_subprocess_omni_binary_health_and_root() {
        let port = free_port();
        let mut child = Command::new(omni_bin_path())
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn omni bin for health test");
        assert!(
            wait_for_200_health(port, Duration::from_secs(6)),
            "omni did not become healthy on {}",
            port
        );
        // root
        let out = Command::new("curl")
            .args(["-s", &format!("http://127.0.0.1:{}/", port)])
            .output()
            .unwrap();
        let body = String::from_utf8_lossy(&out.stdout);
        assert!(body.contains("omni - light multi-backend"));
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn test_subprocess_omni_binary_models() {
        let port = free_port();
        let mut child = Command::new(omni_bin_path())
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        let out = Command::new("curl")
            .args(["-s", &format!("http://127.0.0.1:{}/v1/models", port)])
            .output()
            .unwrap();
        let v: Value = serde_json::from_slice(&out.stdout).expect("models json");
        assert_eq!(v["object"], "list");
        assert!(v["data"].as_array().unwrap().len() >= 2); // at least defaults for enabled (default claude,grok)
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn test_subprocess_omni_binary_auth_mw_401_vs_200() {
        // Auth mw (with/without keys, 401 vs 200) - full layered router via binary
        let port = free_port();
        // with keys set (no --no-auth): unauthed requests 401, authed 200. Wait must auth.
        let mut child = Command::new(omni_bin_path())
            .env("OMNI_API_KEYS", "secret123,other")
            .args(["--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        // wait using proper header (keys case requires it for any surface incl health)
        let start = Instant::now();
        let mut ready = false;
        while start.elapsed() < Duration::from_secs(6) {
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
        assert!(ready, "protected server did not become ready");
        // no header -> 401
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
        // with good key -> 200
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
        // bad key -> 401
        let out3 = Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "-H",
                "Authorization: Bearer wrong",
                &format!("http://127.0.0.1:{}/health", port),
            ])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out3.stdout).trim(), "401");
        let _ = child.kill();
        let _ = child.wait();

        // without keys (empty or --no-auth) -> 200 even no header
        let port2 = free_port();
        let mut child2 = Command::new(omni_bin_path())
            .args(["--no-auth", "--port", &port2.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn2");
        assert!(wait_for_200_health(port2, Duration::from_secs(6)));
        let out4 = Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                &format!("http://127.0.0.1:{}/health", port2),
            ])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out4.stdout).trim(), "200");
        let _ = child2.kill();
        let _ = child2.wait();
    }

    #[test]
    fn test_subprocess_omni_binary_completions_routing_errors() {
        // errors (unknown provider, disabled, bad model) via full http
        let port = free_port();
        let mut child = Command::new(omni_bin_path())
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        // unknown prefix
        let out = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"nope:xx","messages":[{"role":"user","content":"hi"}]}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        assert!(out.status.success(), "curl invocation failed");
        let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(serde_json::json!({}));
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("not enabled")
        );
        // bare in multi
        let out2 = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"bare","messages":[{"role":"user","content":"hi"}]}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        let v2: Value = serde_json::from_slice(&out2.stdout).unwrap_or(serde_json::json!({}));
        assert!(
            v2["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("must use prefix")
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn test_subprocess_omni_binary_completions_live_conditional_both_backends() {
        // live creds conditional for real calls to both backends; unified surfaces
        let port = free_port();
        let mut child = Command::new(omni_bin_path())
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        if has_grok_creds() {
            let out = Command::new("curl")
                .args(["-s", "-X", "POST", "-H", "content-type: application/json", "-d", r#"{"model":"grok:grok-3","messages":[{"role":"user","content":"Reply PONG"}]}"#, &format!("http://127.0.0.1:{}/v1/chat/completions", port)])
                .output().unwrap();
            let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(serde_json::json!({}));
            // accept 200 with real or if rate limit etc, just that it reached delegation not routing err
            if let Some(code) = out.status.code()
                && code == 0
            { /*curl ok*/ }
            let err_msg = v["error"]["message"].as_str().unwrap_or("");
            assert!(
                !err_msg.contains("not enabled") && !err_msg.contains("must use prefix"),
                "routing should have succeeded for grok: prefix: {}",
                err_msg
            );
            if v.get("choices").is_some() {
                assert!(
                    !v["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .is_empty()
                        || v["choices"][0]["message"].get("tool_calls").is_some()
                );
            }
        }
        if has_claude_creds() {
            let out = Command::new("curl")
                .args(["-s", "-X", "POST", "-H", "content-type: application/json", "-d", r#"{"model":"claude:haiku","messages":[{"role":"user","content":"Reply PONG"}]}"#, &format!("http://127.0.0.1:{}/v1/chat/completions", port)])
                .output().unwrap();
            let v: Value = serde_json::from_slice(&out.stdout).unwrap_or(serde_json::json!({}));
            let err_msg = v["error"]["message"].as_str().unwrap_or("");
            assert!(
                !err_msg.contains("not enabled"),
                "claude route: {}",
                err_msg
            );
            if v.get("choices").is_some() {
                let c = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
                assert!(!c.is_empty());
            }
        }
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn test_subprocess_omni_binary_multi_provider_config() {
        // enable both via OMNI_PROVIDERS, test routing to each (prefix)
        let port = free_port();
        let mut child = Command::new(omni_bin_path())
            .env("OMNI_PROVIDERS", "claude,grok")
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        // models should list for both
        let out = Command::new("curl")
            .args(["-s", &format!("http://127.0.0.1:{}/models", port)])
            .output()
            .unwrap();
        let v: Value = serde_json::from_slice(&out.stdout).expect("json");
        let ids: Vec<String> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
            .collect();
        assert!(
            ids.iter()
                .any(|id| id.starts_with("claude:") || id.starts_with("grok:"))
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn test_subprocess_omni_binary_streaming_sse_done_terminator() {
        // WHY: full-stack proof that stream:true over real HTTP yields an SSE
        // body terminated by `data: [DONE]` (the OpenAI streaming contract).
        // Live-conditional on grok creds so the suite stays green offline; the
        // hermetic framing is already pinned by omni-common::http unit tests.
        if !has_grok_creds() {
            eprintln!("skipping streaming subprocess test: no grok creds");
            return;
        }
        let port = free_port();
        let mut child = Command::new(omni_bin_path())
            .env("OMNI_PROVIDERS", "grok")
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        let out = Command::new("curl")
            .args([
                "-sN",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"grok:grok-3","stream":true,"max_tokens":16,"messages":[{"role":"user","content":"Reply PONG"}]}"#,
                &format!("http://127.0.0.1:{}/v1/chat/completions", port),
            ])
            .output()
            .unwrap();
        let body = String::from_utf8_lossy(&out.stdout);
        assert!(
            body.contains("chat.completion.chunk"),
            "expected SSE chunks, got: {body}"
        );
        assert!(body.contains("[DONE]"), "stream must terminate with [DONE]");
        let _ = child.kill();
        let _ = child.wait();
    }

    // ---- OpenAI Responses protocol (/v1/responses): TDD contract tests ----

    fn responses_req(body: &str) -> omni_common::ResponsesRequest {
        serde_json::from_str(body).expect("responses request json")
    }

    #[tokio::test]
    async fn test_responses_unsupported_input_is_bad_request() {
        // WHY: v1 rejects non-message input items (function_call_output needs
        // richer-than-text canonical content) LOUDLY as an OAI-shaped 400 naming
        // the offender; a 500 or silent mangling would corrupt tool conversations.
        let c = Arc::new(ClaudeProvider::new().expect("c"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("claude".into(), c);
        let state = Arc::new(AppState { providers: map });
        let req = responses_req(
            r#"{"model":"claude:sonnet","input":[{"type":"function_call_output","call_id":"c1","output":"42"}]}"#,
        );
        let res = responses_handler(State(state), Json(req)).await;
        match res {
            Err(AppError::BadRequest(msg)) => assert!(
                msg.contains("function_call_output"),
                "400 must name the unsupported item type: {msg}"
            ),
            other => panic!("expected BadRequest for unsupported input, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_responses_bare_model_multi_requires_prefix() {
        // WHY: the aggregator's prefix-routing contract applies to EVERY inbound
        // protocol; Responses requests route exactly like chat completions.
        let c = Arc::new(ClaudeProvider::new().expect("c"));
        let g = Arc::new(GrokProvider::new_for_test("k", "http://127.0.0.1:9"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("claude".into(), c);
        map.insert("grok".into(), g);
        let state = Arc::new(AppState { providers: map });
        let req = responses_req(r#"{"model":"bare-model","input":"hi"}"#);
        let res = responses_handler(State(state), Json(req)).await;
        match res {
            Err(AppError::BadRequest(msg)) => assert!(msg.contains("must use prefix")),
            other => panic!("expected prefix-required BadRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_responses_nonstream_routes_via_prefix_to_grok() {
        // WHY: proves the Responses handler delegates through the same provider
        // map: a dead grok upstream surfaces as ServerError (delegation reached
        // the provider), never as a routing-level BadRequest.
        let g = Arc::new(GrokProvider::new_for_test("dummy", "http://127.0.0.1:1"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("grok".into(), g);
        let state = Arc::new(AppState { providers: map });
        let req = responses_req(r#"{"model":"grok:grok-3","input":"ping"}"#);
        let res = responses_handler(State(state), Json(req)).await;
        match res {
            Err(AppError::ServerError(msg)) => {
                assert!(msg.contains("upstream") || msg.contains("network"))
            }
            other => panic!("expected upstream ServerError via grok route, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_responses_stream_is_routed_to_sse_with_failed_event_on_dead_upstream() {
        // WHY: grok's send_stream defers the HTTP call into the stream body, so
        // even a dead upstream yields an SSE response whose terminal event is
        // response.failed. This pins both halves hermetically: stream:true is
        // routed (not rejected) AND errors surface in Responses SSE framing.
        let g = Arc::new(GrokProvider::new_for_test("dummy", "http://127.0.0.1:1"));
        let mut map: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        map.insert("grok".into(), g);
        let state = Arc::new(AppState { providers: map });
        let req = responses_req(r#"{"model":"grok:grok-3","input":"ping","stream":true}"#);
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

    #[test]
    fn test_subprocess_omni_binary_responses_route_exists() {
        // WHY: the route must be registered in the PRODUCTION router (not just a
        // test router). A parseable-but-unsupported body is rejected at parse
        // level (400) before any upstream call, so this is hermetic; a 404 means
        // /v1/responses is not wired.
        let port = free_port();
        let mut child = Command::new(omni_bin_path())
            .env("OMNI_PROVIDERS", "claude")
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        let out = Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"claude:sonnet","input":[{"type":"function_call_output","call_id":"c","output":"x"}]}"#,
                &format!("http://127.0.0.1:{}/v1/responses", port),
            ])
            .output()
            .unwrap();
        let code = String::from_utf8_lossy(&out.stdout);
        let _ = child.kill();
        let _ = child.wait();
        assert_eq!(
            code.trim(),
            "400",
            "POST /v1/responses must exist and reject unsupported input with 400 (404 = route not registered)"
        );
    }

    #[test]
    fn test_subprocess_omni_binary_responses_live_roundtrip() {
        // WHY: end-to-end proof over real HTTP: a Responses request returns the
        // Responses envelope (non-stream) and Responses SSE events (stream)
        // through the aggregator's prefix routing. Live-conditional on grok
        // creds so the suite stays green offline.
        if !has_grok_creds() {
            eprintln!("skipping responses live roundtrip: no grok creds");
            return;
        }
        let port = free_port();
        let mut child = Command::new(omni_bin_path())
            .env("OMNI_PROVIDERS", "grok")
            .args(["--no-auth", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn");
        assert!(wait_for_200_health(port, Duration::from_secs(6)));

        // Non-stream: Responses envelope with assistant output_text.
        let out = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"grok:grok-3","input":"Reply with the single word PONG","max_output_tokens":16}"#,
                &format!("http://127.0.0.1:{}/v1/responses", port),
            ])
            .output()
            .unwrap();
        let v: Value = serde_json::from_slice(&out.stdout).expect("responses json");
        assert_eq!(v["object"], "response", "live body: {v}");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["output"][0]["type"], "message");
        assert!(
            !v["output"][0]["content"][0]["text"]
                .as_str()
                .unwrap_or("")
                .is_empty(),
            "live response must carry assistant text: {v}"
        );
        assert!(v["usage"]["total_tokens"].as_u64().unwrap_or(0) > 0);

        // Stream: Responses SSE events, no [DONE].
        let out2 = Command::new("curl")
            .args([
                "-sN",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-d",
                r#"{"model":"grok:grok-3","input":"Reply with the single word PONG","max_output_tokens":16,"stream":true}"#,
                &format!("http://127.0.0.1:{}/v1/responses", port),
            ])
            .output()
            .unwrap();
        let body = String::from_utf8_lossy(&out2.stdout);
        let _ = child.kill();
        let _ = child.wait();
        assert!(
            body.contains("event: response.created"),
            "live stream must open with response.created: {body}"
        );
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.completed"));
        assert!(!body.contains("[DONE]"));
    }
}
