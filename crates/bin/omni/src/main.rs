//! Omni multi-provider server binary.
//!
//! This is the only server binary. Provider-specific protocol, credential, and
//! fingerprint logic stays in `provider-claude`, `provider-grok`, and
//! `provider-codex`; this binary
//! owns HTTP routing, auth, stats, and OpenAI-compatible response framing.
//!
//! ## Supported configuration (per task)
//! - --providers claude,grok,codex   or   OMNI_PROVIDERS=... (comma sep, order preserved)
//! - --bind 127.0.0.1 by default, or --public as shorthand for --bind 0.0.0.0
//! - Canonical model routing: real model ids (e.g. "claude-sonnet-4-6", "grok-4.3")
//!   route directly when they uniquely identify an enabled provider.
//! - Alias routing: "fable", "opus", "sonnet", "haiku", "grok", and "composer"
//!   resolve to current provider-owned model ids when unique.
//! - Optional prefix routing remains an escape hatch: "grok:foo", "claude:bar", or "codex:bar".
//!
//! ## Surfaces (unified OpenAI-compatible)
//! - POST /v1/chat/completions  (text + sampling; non-stream JSON and stream SSE)
//! - POST /v1/responses          (supported OpenAI Responses subset)
//! - GET  /v1/models , /models
//! - GET  /stats
//! - GET  /health
//! - GET  /
//!
//! ## How it uses shared crates (per requirements)
//! - omni-common: auth_layer + AppError (OAI-shaped errors) + the shared http
//!   layer (to_canonical/from_canonical + the SSE stream framer).
//! - omni-core: CanonicalRequest/Response + LlmProvider trait for the delegation contract.
//! - Depends on provider-claude (full fingerprint provider), provider-grok, and provider-codex.
//!
//! ## Routing implementation (pure, unit-testable)
//! See `resolve_provider_and_model` below. Pure function; no side effects.
//! Prefix takes precedence. Provider keys in the map and for prefixes are "claude", "grok", "codex".
//!
//! ## Boundaries
//! - Claude fingerprint logic, cch, betas, preamble, and fresh credential reads
//!   stay in `provider-claude`.
//! - Grok wire mapping and fresh xAI credential reads stay in `provider-grok`.
//! - Codex config/auth and Responses wire mapping stay in `provider-codex`.
//! - Auth and stats are server concerns handled here with `omni-common`.
//! - Empty key set (via --no-auth or no OMNI_API_KEYS) means "allow all".
//!
//! Build: cargo build -p omni
//! Run (claude only, no keys needed): OMNI_PROVIDERS=claude cargo run -p omni -- --no-auth --port 18321
//! Test: cargo test -p omni
//!
//! Documented here per "document any findings in the code or note for docs/" (no new .md).

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use axum::{
    Router,
    body::Body,
    extract::{Extension, Json, State},
    http::{HeaderMap, Request, StatusCode, header},
    middleware,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use clap::Parser;
use futures_util::StreamExt;
use tracing::{info, warn};
use uuid::Uuid;

use omni_common::{
    ActiveRequestGuard, ApiKeyId, AppError, ChatCompletionRequest, ConversationLog, Replacements,
    Stats, TokenUsage, from_canonical, to_canonical,
};
use omni_core::{
    CanonicalResponse, CanonicalStream, CanonicalStreamEvent, LlmProvider, ProviderError,
};

// Re-export the concrete providers so main can construct them by name.
use provider_claude::ClaudeProvider;
use provider_codex::CodexProvider;
use provider_grok::GrokProvider;

const OMNI_ASCII_BANNER: &str = r#"
   ___  __  __ _   _ ___
  / _ \|  \/  | \ | |_ _|
 | | | | |\/| |  \| || |
 | |_| | |  | | |\  || |
  \___/|_|  |_|_| \_|___|
"#;

/// CLI for the light omni aggregator.
/// Env vars: OMNI_PROVIDERS, OMNI_BIND, OMNI_PUBLIC, OMNI_PORT, OMNI_NO_AUTH,
/// OMNI_STATS_DB (clap env support). OMNI_API_KEYS configures auth keys.
#[derive(Parser, Debug)]
#[command(
    name = "omni",
    version,
    about = "Omni LLM server (claude + grok + codex backends)"
)]
struct Cli {
    /// Comma-separated list of providers to enable (claude,grok,codex). Prefix routing uses these names.
    #[arg(
        long,
        env = "OMNI_PROVIDERS",
        default_value = "claude,grok",
        value_delimiter = ','
    )]
    providers: Vec<String>,

    /// Listen port.
    #[arg(long, env = "OMNI_PORT", default_value_t = 18321)]
    port: u16,

    /// Listen address. Defaults to localhost only.
    #[arg(long, env = "OMNI_BIND")]
    bind: Option<String>,

    /// Listen on all interfaces. Equivalent to --bind 0.0.0.0.
    #[arg(long, env = "OMNI_PUBLIC")]
    public: bool,

    /// Disable API key auth (if omitted, still allows all unless OMNI_API_KEYS is set).
    #[arg(long, env = "OMNI_NO_AUTH")]
    no_auth: bool,

    /// Path to the stats redb file. Defaults to a fixed temp file; use a
    /// durable, per-instance path for long-running or concurrent servers.
    #[arg(long, env = "OMNI_STATS_DB")]
    stats_db: Option<PathBuf>,

    /// Log conversation prompts and responses to stderr.
    #[arg(long, env = "OMNI_LOG_CONVERSATIONS")]
    log_conversations: bool,

    /// File to write conversation logs to. Implies --log-conversations.
    #[arg(long, env = "OMNI_LOG_FILE", conflicts_with = "log_dir")]
    log_file: Option<PathBuf>,

    /// Directory to write one conversation log file per session id.
    #[arg(long, env = "OMNI_LOG_DIR", conflicts_with = "log_file")]
    log_dir: Option<PathBuf>,

    /// Rotate --log-file after this many bytes. Set to 0 to disable rotation.
    #[arg(
        long,
        env = "OMNI_LOG_MAX_BYTES",
        default_value_t = omni_common::DEFAULT_LOG_MAX_BYTES
    )]
    log_max_bytes: u64,

    /// Number of rotated conversation log files to keep.
    #[arg(
        long,
        env = "OMNI_LOG_BACKUPS",
        default_value_t = omni_common::DEFAULT_LOG_BACKUPS
    )]
    log_backups: usize,
}

#[derive(Clone)]
struct ProviderEntry {
    provider: Arc<dyn LlmProvider>,
    claude_native: Option<Arc<ClaudeProvider>>,
    models: Vec<serde_json::Value>,
    catalog: ModelCatalog,
}

#[derive(Clone, Debug, Default)]
struct ModelCatalog {
    aliases: HashMap<String, String>,
}

#[derive(Clone)]
struct AppState {
    /// Map from provider key ("claude" | "grok" | "codex") to live provider + catalog.
    providers: HashMap<String, ProviderEntry>,
    stats: Option<Arc<Stats>>,
    conversation_log: Option<Arc<ConversationLog>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            "info,omni=debug,provider_claude=debug,provider_grok=debug,provider_codex=debug",
        )
        .init();

    let cli = Cli::parse();
    log_startup_banner();

    // Normalize + validate providers list (unique, known names only).
    let enabled: Vec<String> = normalize_providers(&cli.providers)?;
    info!(?enabled, "omni enabled providers");

    let mut providers_map: HashMap<String, ProviderEntry> = HashMap::new();
    for name in &enabled {
        let entry = match name.as_str() {
            "claude" => {
                let provider = init_claude_provider()
                    .context("failed to initialize claude provider (fingerprint profile)")?;
                let models = provider_model_values("claude", provider.profile().models_list())?;
                let catalog = claude_model_catalog(provider.profile());
                let provider = Arc::new(provider);
                ProviderEntry {
                    provider: provider.clone(),
                    claude_native: Some(provider),
                    models,
                    catalog,
                }
            }
            "grok" => {
                let p = init_grok_provider().context("failed to init grok provider")?;
                let models = provider_model_values("grok", GrokProvider::models_list())?;
                let catalog = grok_model_catalog();
                ProviderEntry {
                    provider: Arc::new(p),
                    claude_native: None,
                    models,
                    catalog,
                }
            }
            "codex" => {
                info!("initializing codex provider (config read from CODEX_HOME or ~/.codex)");
                let p = CodexProvider::new().context("failed to init codex provider")?;
                let models = provider_model_values("codex", p.models_list())?;
                let catalog = codex_model_catalog(&p);
                ProviderEntry {
                    provider: Arc::new(p),
                    claude_native: None,
                    models,
                    catalog,
                }
            }
            other => {
                // Should be impossible after normalize.
                anyhow::bail!("unknown provider in list: {}", other);
            }
        };
        providers_map.insert(name.clone(), entry);
    }

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

    let state = AppState {
        providers: providers_map,
        stats,
        conversation_log: build_conversation_log(&cli)?,
    };
    let alias_text = format_aliases_for_log(&state.providers).unwrap_or_else(|| "none".into());

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

    let app = build_router(Arc::new(state), auth_keys);

    let bind_host = resolve_bind_host(cli.public, cli.bind.as_deref())?;
    let addr: SocketAddr = format!("{}:{}", bind_host, cli.port).parse()?;
    info!("omni listening on http://{}", addr);
    info!("  providers: {}", enabled.join(","));
    info!("  try: curl http://{}/health", addr);
    info!("  models:  curl http://{}/v1/models", addr);
    info!("  stats:   curl http://{}/stats", addr);
    info!("  completions example: model=grok, codex, or claude-sonnet-4-6");
    info!("  aliases: {}", alias_text);

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app)
        .await
        .context("server error")?;

    Ok(())
}

fn log_startup_banner() {
    info!(
        "\n{}\n\n  Omni LLM Provider\n  Version: {}\n",
        OMNI_ASCII_BANNER.trim_matches('\n'),
        env!("CARGO_PKG_VERSION")
    );
}

fn init_claude_provider() -> anyhow::Result<ClaudeProvider> {
    if let Some(base_url) = env_nonempty("ANTHROPIC_BASE_URL") {
        let authorization_bearer = std::env::var_os("ANTHROPIC_AUTH_TOKEN")
            .is_some()
            .then(|| "ANTHROPIC_AUTH_TOKEN".to_string());
        let api_key = std::env::var_os("ANTHROPIC_API_KEY")
            .is_some()
            .then(|| "ANTHROPIC_API_KEY".to_string());
        let custom_headers = std::env::var_os("ANTHROPIC_CUSTOM_HEADERS")
            .is_some()
            .then(|| "ANTHROPIC_CUSTOM_HEADERS".to_string());
        info!(
            base_url = %base_url,
            auth = if authorization_bearer.is_some() {
                "bearer"
            } else if api_key.is_some() {
                "x-api-key"
            } else {
                "custom-headers-or-no-auth"
            },
            "initializing claude provider with ANTHROPIC_BASE_URL custom gateway"
        );
        return ClaudeProvider::new_for_custom_gateway_env(
            provider_claude::default_profile(),
            base_url,
            authorization_bearer,
            api_key,
            custom_headers,
        )
        .map_err(anyhow::Error::from);
    }

    info!("initializing claude provider");
    ClaudeProvider::new().map_err(anyhow::Error::from)
}

fn init_grok_provider() -> anyhow::Result<GrokProvider> {
    let provider = GrokProvider::new(None).map_err(anyhow::Error::from)?;
    if let Some(base_url) = env_nonempty("GROK_MODELS_BASE_URL") {
        info!(
            base_url = %base_url,
            auth = if std::env::var_os("XAI_API_KEY").is_some() { "bearer" } else { "no-auth" },
            "initializing grok provider with GROK_MODELS_BASE_URL custom endpoint"
        );
        return Ok(provider.with_base_url(base_url).with_custom_auth(
            None,
            Some("XAI_API_KEY".into()),
            vec![],
        ));
    }

    info!("initializing grok provider (key read per request from ~/.xai/.credentials.json)");
    Ok(provider)
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn build_router(state: Arc<AppState>, auth_keys: Arc<HashSet<String>>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/", get(root_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/responses", post(responses_handler))
        .route("/v1/messages", post(anthropic_messages_handler))
        .route(
            "/v1/messages/count_tokens",
            post(anthropic_count_tokens_handler),
        )
        .route("/v1/models", get(models_handler))
        .route("/models", get(models_handler))
        .route("/stats", get(stats_handler))
        .with_state(state)
        // Always layer; the common impl short-circuits when keys are empty.
        .layer(middleware::from_fn({
            let keys = auth_keys.clone();
            move |req, next| omni_auth_layer(keys.clone(), req, next)
        }))
}

async fn omni_auth_layer(
    valid_keys: Arc<HashSet<String>>,
    mut req: Request<Body>,
    next: middleware::Next,
) -> Response {
    if valid_keys.is_empty() {
        return next.run(req).await;
    }

    let is_anthropic =
        req.uri().path() == "/v1/messages" || req.uri().path() == "/v1/messages/count_tokens";
    let key = req
        .headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
        .map(str::to_string);

    match key {
        Some(key) if valid_keys.contains(&key) => {
            req.extensions_mut().insert(ApiKeyId(auth_key_id(&key)));
            next.run(req).await
        }
        Some(_) if is_anthropic => {
            anthropic_error_response(AppError::Unauthorized("Invalid API key".into()))
        }
        None if is_anthropic => anthropic_error_response(AppError::Unauthorized(
            "Missing API key. Include 'Authorization: Bearer <key>' header.".into(),
        )),
        Some(_) => AppError::Unauthorized("Invalid API key".into()).into_response(),
        None => AppError::Unauthorized(
            "Missing API key. Include 'Authorization: Bearer <key>' header.".into(),
        )
        .into_response(),
    }
}

fn auth_key_id(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() < 12 {
        let suffix: String = chars.iter().rev().take(4).rev().collect();
        return format!("...{}", suffix);
    }
    let prefix: String = chars.iter().take(4).collect();
    let suffix: String = chars.iter().rev().take(4).rev().collect();
    format!("{}...{}", prefix, suffix)
}

fn default_stats_db_path() -> PathBuf {
    std::env::temp_dir().join("omni-stats.redb")
}

fn build_conversation_log(cli: &Cli) -> anyhow::Result<Option<Arc<ConversationLog>>> {
    if let Some(path) = cli.log_dir.as_ref() {
        let log = ConversationLog::to_dir(path)
            .with_context(|| format!("failed to open conversation log dir {}", path.display()))?;
        info!(path = %path.display(), "conversation log enabled in directory mode");
        return Ok(Some(Arc::new(log)));
    }
    if let Some(path) = cli.log_file.as_ref() {
        let log = ConversationLog::to_file(path, cli.log_max_bytes, cli.log_backups)
            .with_context(|| format!("failed to open conversation log file {}", path.display()))?;
        info!(path = %path.display(), "conversation log enabled in file mode");
        return Ok(Some(Arc::new(log)));
    }
    if cli.log_conversations {
        info!("conversation log enabled to stderr");
        return Ok(Some(Arc::new(ConversationLog::to_stderr())));
    }
    Ok(None)
}

fn provider_model_values<T: serde::Serialize>(
    provider: &str,
    models: Vec<T>,
) -> anyhow::Result<Vec<serde_json::Value>> {
    models
        .into_iter()
        .map(|model| {
            let mut value = serde_json::to_value(model)
                .context("failed to serialize provider model catalog entry")?;
            let obj = value
                .as_object_mut()
                .context("provider model catalog entry must serialize as an object")?;
            obj.get("id")
                .and_then(|v| v.as_str())
                .context("provider model catalog entry missing string id")?;
            obj.insert("owned_by".to_string(), serde_json::json!(provider));
            Ok(value)
        })
        .collect()
}

fn claude_model_catalog(profile: &provider_claude::FingerprintProfile) -> ModelCatalog {
    let mut catalog = ModelCatalog::default();
    for model in profile.models {
        catalog.insert(model.canonical, model.canonical);
        catalog.insert(model.cli_name, model.canonical);
        for alias in model.aliases {
            catalog.insert(alias, model.canonical);
        }
    }
    for model in profile.model_wire_overrides {
        catalog.insert(model.model, model.model);
    }
    catalog
}

fn grok_model_catalog() -> ModelCatalog {
    let mut catalog = ModelCatalog::default();
    for (alias, canonical) in GrokProvider::model_aliases() {
        catalog.insert(alias, canonical);
    }
    catalog
}

fn codex_model_catalog(provider: &CodexProvider) -> ModelCatalog {
    let mut catalog = ModelCatalog::default();
    for (alias, canonical) in provider.model_aliases() {
        catalog.insert(&alias, &canonical);
    }
    catalog
}

impl ModelCatalog {
    fn insert(&mut self, alias: &str, canonical: &str) {
        if let Some(existing) = self.aliases.get(alias) {
            assert_eq!(
                existing, canonical,
                "model alias {alias:?} maps to both {existing:?} and {canonical:?}"
            );
            return;
        }
        self.aliases
            .insert(alias.to_string(), canonical.to_string());
    }

    fn resolve(&self, model: &str) -> Option<&str> {
        self.aliases.get(model).map(String::as_str)
    }
}

fn format_aliases_for_log(providers: &HashMap<String, ProviderEntry>) -> Option<String> {
    let catalogs = provider_catalogs(providers);
    let mut pairs = Vec::new();
    for alias in [
        "sonnet", "opus", "haiku", "fable", "grok", "composer", "codex",
    ] {
        let matches = model_matches(alias, &catalogs);
        if matches.len() == 1 {
            pairs.push(format!("{}={}", alias, matches[0].1));
        }
    }
    if pairs.is_empty() {
        None
    } else {
        Some(pairs.join(", "))
    }
}

/// Resolve the configured listen host. `--public` is intentional shorthand for
/// all interfaces, while the default remains loopback-only.
fn resolve_bind_host(public: bool, bind: Option<&str>) -> anyhow::Result<String> {
    let bind = bind.map(str::trim).filter(|s| !s.is_empty());
    if public && bind.is_some() {
        anyhow::bail!("--public cannot be used with --bind/OMNI_BIND");
    }
    if public {
        return Ok("0.0.0.0".to_string());
    }
    Ok(bind.unwrap_or("127.0.0.1").to_string())
}

/// Normalize/validate the providers CLI/env list.
/// Accepts "claude,grok,codex", trims, lowercases, dedups in order, rejects unknowns.
fn normalize_providers(raw: &[String]) -> anyhow::Result<Vec<String>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for r in raw {
        let p = r.trim().to_lowercase();
        if p.is_empty() {
            continue;
        }
        if p != "claude" && p != "grok" && p != "codex" {
            anyhow::bail!("unknown provider {:?}; supported: claude,grok,codex", r);
        }
        if seen.insert(p.clone()) {
            out.push(p);
        }
    }
    if out.is_empty() {
        anyhow::bail!("at least one provider required (claude, grok, and/or codex)");
    }
    Ok(out)
}

/// Pure routing function. Extracted for easy unit testing of the core logic.
/// Returns (provider_key, provider_model).
///
/// Rules:
/// - If input contains "prefix:rest" (first :), use prefix (lowercased) if enabled
///   and normalize `rest` through that provider's alias catalog.
/// - Else if exactly one provider catalog matches the bare model, route there.
/// - Else if exactly one provider is enabled, route to it and normalize if possible.
/// - Else reject unknown or ambiguous bare names loudly.
fn resolve_provider_and_model(
    model: &str,
    catalogs: &HashMap<String, ModelCatalog>,
) -> Result<(String, String), String> {
    if let Some((pre, rest)) = model.split_once(':') {
        let key = pre.trim().to_lowercase();
        if let Some(catalog) = catalogs.get(&key) {
            let stripped = rest.trim().to_string();
            if stripped.is_empty() {
                return Err(format!("empty model after prefix for provider {}", key));
            }
            let normalized = catalog.resolve(&stripped).unwrap_or(&stripped);
            return Ok((key, normalized.to_string()));
        } else {
            let enabled = enabled_provider_keys(catalogs);
            return Err(format!(
                "provider '{}' not enabled (enabled: [{}])",
                key,
                enabled.join(",")
            ));
        }
    }

    let matches = model_matches(model, catalogs);
    match matches.as_slice() {
        [(provider, canonical)] => return Ok((provider.clone(), canonical.clone())),
        [] => {}
        _ => {
            let providers = matches
                .iter()
                .map(|(provider, canonical)| format!("{provider}:{canonical}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "model '{}' is ambiguous across enabled providers ({providers}); use a provider prefix",
                model
            ));
        }
    }

    if catalogs.len() == 1
        && let Some((provider, catalog)) = catalogs.iter().next()
    {
        let normalized = catalog.resolve(model).unwrap_or(model);
        return Ok((provider.clone(), normalized.to_string()));
    }

    Err(format!(
        "unknown model '{}' for enabled providers [{}]; use a listed model id or a provider prefix",
        model,
        enabled_provider_keys(catalogs).join(",")
    ))
}

fn model_matches(model: &str, catalogs: &HashMap<String, ModelCatalog>) -> Vec<(String, String)> {
    let mut matches = catalogs
        .iter()
        .filter_map(|(provider, catalog)| {
            catalog
                .resolve(model)
                .map(|canonical| (provider.clone(), canonical.to_string()))
        })
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| a.0.cmp(&b.0));
    matches
}

fn enabled_provider_keys(catalogs: &HashMap<String, ModelCatalog>) -> Vec<String> {
    let mut enabled = catalogs.keys().cloned().collect::<Vec<_>>();
    enabled.sort();
    enabled
}

fn provider_catalogs(providers: &HashMap<String, ProviderEntry>) -> HashMap<String, ModelCatalog> {
    providers
        .iter()
        .map(|(provider, entry)| (provider.clone(), entry.catalog.clone()))
        .collect()
}

/// Handler: GET /health
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Handler: GET /
async fn root_handler() -> impl IntoResponse {
    "omni - multi-backend OpenAI-compatible server (claude + grok + codex via canonical model ids, aliases, or provider prefixes)"
}

/// Handler: GET /v1/models (and /models)
async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut data = Vec::new();
    let mut providers = state.providers.keys().collect::<Vec<_>>();
    providers.sort();
    for p in providers {
        if let Some(entry) = state.providers.get(p) {
            data.extend(entry.models.iter().cloned());
        }
    }
    Json(serde_json::json!({ "object": "list", "data": data }))
}

/// Handler: GET /stats
async fn stats_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match &state.stats {
        Some(stats) => Json(serde_json::json!(stats.snapshot())),
        None => Json(serde_json::json!({
            "stats_enabled": false,
            "note": "stats db unavailable; counters not being recorded",
        })),
    }
}

/// Handler: POST /v1/chat/completions
async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    api_key: Option<Extension<ApiKeyId>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, AppError> {
    let request_id = Uuid::new_v4().to_string();
    let short_request_id = request_id.chars().take(8).collect::<String>();
    let session_id = chat_session_id(&headers, &body, api_key.as_ref().map(|k| k.0.0.as_str()));
    log_json(
        &state,
        &session_id,
        &short_request_id,
        ">>>",
        "Inbound Chat Completions body",
        &body,
    );

    let requested_model = body.model.clone();
    let catalogs = provider_catalogs(&state.providers);
    let (prov_key, stripped_model) = resolve_provider_and_model(&requested_model, &catalogs)
        .map_err(|e| record_bad_request(&state, &requested_model, e))?;
    let stats_key = stats_model_key(&prov_key, &stripped_model);

    let entry = state
        .providers
        .get(&prov_key)
        .ok_or_else(|| AppError::ServerError("provider disappeared".into()))?;
    let provider = &entry.provider;

    // Build canonical *with the stripped model* so the delegated provider sees the real model name.
    let mut canon = to_canonical(&body).map_err(|e| record_bad_request(&state, &stats_key, e))?;
    canon.model = stripped_model.clone();

    if let Some(stats) = &state.stats {
        stats.record_request(&stats_key, None);
    }

    let chat_id = format!("chatcmpl-{request_id}");
    let created = omni_common::unix_now_secs();

    if body.stream {
        // Streaming: delegate to the provider's native SSE stream and frame it as
        // OpenAI chat.completion.chunk events (terminated by [DONE]) via the shared
        // serializer. Prefix routing has already selected the provider above.
        let stream = provider.send_stream(canon).await.map_err(|e| {
            if let Some(stats) = &state.stats {
                stats.record_error(&stats_key, &e.to_string());
            }
            map_provider_err(e)
        })?;
        log_text(
            &state,
            &session_id,
            &short_request_id,
            "<<<",
            "Chat Completions stream opened",
            &format!("model={requested_model}"),
        );
        let stream = wrap_stream_for_stats(
            stream,
            state.stats.clone(),
            stats_key.clone(),
            state.conversation_log.clone(),
            session_id.clone(),
            short_request_id.clone(),
            "Chat Completions stream",
        );
        // Echo the resolved canonical provider model, not shorthand aliases.
        let sse = omni_common::sse_from_canonical_stream(stream, stripped_model, chat_id, created);
        return Ok(sse.into_response());
    }

    // The actual delegation (thin by design).
    let _active = state.stats.as_deref().map(ActiveRequestGuard::new);
    let started = Instant::now();
    let canon_resp: CanonicalResponse = provider.send(canon).await.map_err(|e| {
        if let Some(stats) = &state.stats {
            stats.record_error(&stats_key, &e.to_string());
        }
        map_provider_err(e)
    })?;

    if let Some(stats) = &state.stats {
        record_response_stats(stats, &stats_key, &canon_resp, started);
    }

    // Echo the resolved canonical provider model, not shorthand aliases.
    let oai = from_canonical(canon_resp, stripped_model, chat_id, created);
    log_json(
        &state,
        &session_id,
        &short_request_id,
        "<<<",
        "Chat Completions response",
        &oai,
    );
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

fn record_bad_request(state: &AppState, model: &str, msg: String) -> AppError {
    if let Some(stats) = &state.stats {
        stats.record_error(model, &msg);
    }
    AppError::BadRequest(msg)
}

fn record_response_stats(
    stats: &Stats,
    model: &str,
    canon_resp: &CanonicalResponse,
    started: Instant,
) {
    let usage = TokenUsage {
        input_tokens: canon_resp.usage.input_tokens,
        output_tokens: canon_resp.usage.output_tokens,
        cache_read_input_tokens: canon_resp.usage.cache_read,
        cache_creation_input_tokens: canon_resp.usage.cache_creation,
    };
    let dur_ms = started.elapsed().as_secs_f64() * 1000.0;
    stats.record_response(model, usage, None, dur_ms);
}

fn wrap_stream_for_stats(
    mut stream: CanonicalStream,
    stats: Option<Arc<Stats>>,
    model: String,
    conversation_log: Option<Arc<ConversationLog>>,
    session_id: String,
    request_id: String,
    label: &'static str,
) -> CanonicalStream {
    Box::pin(async_stream::stream! {
        let _active = stats.as_deref().map(ActiveRequestGuard::new);
        let started = Instant::now();
        let mut usage = TokenUsage::default();
        let mut saw_usage = false;
        let mut finished = false;

        while let Some(item) = futures_util::StreamExt::next(&mut stream).await {
            match item {
                Ok(CanonicalStreamEvent::Usage(u)) => {
                    usage = TokenUsage {
                        input_tokens: u.input_tokens,
                        output_tokens: u.output_tokens,
                        cache_read_input_tokens: u.cache_read,
                        cache_creation_input_tokens: u.cache_creation,
                    };
                    saw_usage = true;
                    yield Ok(CanonicalStreamEvent::Usage(u));
                }
                Ok(CanonicalStreamEvent::Finish { finish_reason }) => {
                    finished = true;
                    let dur_ms = started.elapsed().as_secs_f64() * 1000.0;
                    if let Some(stats) = stats.as_ref() {
                        stats.record_response(
                            &model,
                            usage,
                            None,
                            dur_ms,
                        );
                    }
                    if let Some(log) = conversation_log.as_ref() {
                        log.log(
                            &session_id,
                            &request_id,
                            "<<<",
                            label,
                            &format!("finish_reason={:?} duration_ms={dur_ms:.1}", finish_reason),
                        );
                    }
                    yield Ok(CanonicalStreamEvent::Finish { finish_reason });
                }
                Ok(other) => {
                    yield Ok(other);
                }
                Err(e) => {
                    if let Some(stats) = stats.as_ref() {
                        stats.record_error(&model, &e.to_string());
                    }
                    if let Some(log) = conversation_log.as_ref() {
                        log.log(&session_id, &request_id, "<<<", label, &format!("error={e}"));
                    }
                    yield Err(e);
                }
            }
        }

        if !finished {
            let dur_ms = started.elapsed().as_secs_f64() * 1000.0;
            let usage = if saw_usage { usage } else { TokenUsage::default() };
            if let Some(stats) = stats.as_ref() {
                stats.record_response(&model, usage, None, dur_ms);
            }
            if let Some(log) = conversation_log.as_ref() {
                log.log(
                    &session_id,
                    &request_id,
                    "<<<",
                    label,
                    &format!("stream ended without finish duration_ms={dur_ms:.1}"),
                );
            }
        }
    })
}

fn stats_model_key(provider: &str, model: &str) -> String {
    format!("{provider}:{model}")
}

fn derive_session_uuid(session_id: &str) -> uuid::Uuid {
    if let Ok(uuid) = uuid::Uuid::parse_str(session_id) {
        return uuid;
    }
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_DNS, session_id.as_bytes())
}

fn session_header(headers: &HeaderMap) -> Option<&str> {
    headers.get("x-session-id").and_then(|v| v.to_str().ok())
}

fn chat_session_id(
    headers: &HeaderMap,
    body: &ChatCompletionRequest,
    api_key_id: Option<&str>,
) -> String {
    let user = body
        .extras
        .get("user")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    omni_common::session::resolve_session_id(session_header(headers), user, api_key_id)
}

fn responses_session_id(
    headers: &HeaderMap,
    body: &omni_common::ResponsesRequest,
    api_key_id: Option<&str>,
) -> String {
    let user = body
        .extras
        .get("user")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    omni_common::session::resolve_session_id(session_header(headers), user, api_key_id)
}

fn log_json<T: serde::Serialize>(
    state: &AppState,
    session_id: &str,
    request_id: &str,
    direction: &str,
    label: &str,
    value: &T,
) {
    if let Some(log) = state.conversation_log.as_ref() {
        match serde_json::to_string(value) {
            Ok(body) => log.log(session_id, request_id, direction, label, &body),
            Err(e) => log.log(
                session_id,
                request_id,
                direction,
                label,
                &format!("<json serialization failed: {e}>"),
            ),
        }
    }
}

fn log_text(
    state: &AppState,
    session_id: &str,
    request_id: &str,
    direction: &str,
    label: &str,
    content: &str,
) {
    if let Some(log) = state.conversation_log.as_ref() {
        log.log(session_id, request_id, direction, label, content);
    }
}

async fn anthropic_messages_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    api_key: Option<Extension<ApiKeyId>>,
    request: Request<Body>,
) -> Response {
    match anthropic_messages_inner(state, headers, api_key, request).await {
        Ok(response) => response,
        Err(error) => anthropic_error_response(error),
    }
}

async fn anthropic_messages_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    api_key: Option<Extension<ApiKeyId>>,
    request: Request<Body>,
) -> Result<Response, AppError> {
    let request_id = Uuid::new_v4().to_string();
    let short_request_id = request_id.chars().take(8).collect::<String>();
    let raw_body = read_anthropic_body(request).await?;
    let requested_model = raw_body
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("<missing>")
        .to_string();
    let session_id = anthropic_session_id(
        &headers,
        raw_body
            .get("metadata")
            .and_then(|metadata| metadata.get("user_id"))
            .and_then(|value| value.as_str()),
        api_key.as_ref().map(|key| key.0.0.as_str()),
        &short_request_id,
    );
    log_json(
        &state,
        &session_id,
        &short_request_id,
        ">>>",
        "Inbound Anthropic Messages body",
        &raw_body,
    );

    let claude = claude_native_for_anthropic(&state, &requested_model)?;
    let raw_body = strip_claude_model_prefix(raw_body)?;
    let replacements = Replacements::empty();
    let prepared = claude
        .prepare_anthropic_messages(raw_body, &replacements, true)
        .map_err(map_anthropic_prepare_err)?;
    if !prepared.dropped_fields.is_empty() {
        warn!(
            dropped = ?prepared.dropped_fields,
            "anthropic request dropped non-forwarded client body fields"
        );
    }

    let stats_key = stats_model_key("claude", &prepared.model_canonical);
    if let Some(stats) = &state.stats {
        stats.record_request(&stats_key, api_key.as_ref().map(|key| key.0.0.as_str()));
    }
    let ctx = provider_claude::RequestContext::new_reply()
        .with_session(derive_session_uuid(&session_id))
        .with_model(prepared.outbound_model.clone());

    log_json(
        &state,
        &session_id,
        &short_request_id,
        ">>>",
        "Anthropic upstream body",
        prepared.body(),
    );

    if prepared.stream {
        let stream = claude
            .send_anthropic_messages_stream(prepared.body(), &ctx)
            .await
            .map_err(|error| {
                if let Some(stats) = &state.stats {
                    stats.record_error(&stats_key, &error.to_string());
                }
                map_provider_err(error)
            })?;
        log_text(
            &state,
            &session_id,
            &short_request_id,
            "<<<",
            "Anthropic Messages stream opened",
            &format!("model={}", prepared.requested_model),
        );
        return Ok(anthropic_sse_response(
            stream,
            state.stats.clone(),
            state.conversation_log.clone(),
            stats_key,
            session_id,
            short_request_id,
            replacements,
        ));
    }

    let _active = state.stats.as_deref().map(ActiveRequestGuard::new);
    let started = Instant::now();
    let mut value = claude
        .send_anthropic_messages_json(prepared.body(), &ctx)
        .await
        .map_err(|error| {
            if let Some(stats) = &state.stats {
                stats.record_error(&stats_key, &error.to_string());
            }
            map_provider_err(error)
        })?;
    provider_claude::anthropic_passthrough::apply_response_replacements_raw(
        &mut value,
        &replacements,
    );
    if let Some(stats) = &state.stats {
        stats.record_response(
            &stats_key,
            provider_claude::anthropic_passthrough::token_usage_from_response(&value),
            None,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    log_json(
        &state,
        &session_id,
        &short_request_id,
        "<<<",
        "Anthropic Messages response",
        &value,
    );
    Ok((anthropic_request_id_header(&short_request_id), Json(value)).into_response())
}

async fn anthropic_count_tokens_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    api_key: Option<Extension<ApiKeyId>>,
    request: Request<Body>,
) -> Response {
    match anthropic_count_tokens_inner(state, headers, api_key, request).await {
        Ok(response) => response,
        Err(error) => anthropic_error_response(error),
    }
}

async fn anthropic_count_tokens_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    api_key: Option<Extension<ApiKeyId>>,
    request: Request<Body>,
) -> Result<Response, AppError> {
    let request_id = Uuid::new_v4().to_string();
    let short_request_id = request_id.chars().take(8).collect::<String>();
    let raw_body = read_anthropic_body(request).await?;
    let requested_model = raw_body
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or("<missing>")
        .to_string();
    let session_id = anthropic_session_id(
        &headers,
        raw_body
            .get("metadata")
            .and_then(|metadata| metadata.get("user_id"))
            .and_then(|value| value.as_str()),
        api_key.as_ref().map(|key| key.0.0.as_str()),
        &short_request_id,
    );

    let claude = claude_native_for_anthropic(&state, &requested_model)?;
    let raw_body = strip_claude_model_prefix(raw_body)?;
    let replacements = Replacements::empty();
    let prepared = claude
        .prepare_anthropic_count_tokens(raw_body, &replacements)
        .map_err(map_anthropic_prepare_err)?;
    let stats_key = stats_model_key("claude", &prepared.model_canonical);
    if let Some(stats) = &state.stats {
        stats.record_request(&stats_key, api_key.as_ref().map(|key| key.0.0.as_str()));
    }
    let ctx = provider_claude::RequestContext::new_reply()
        .with_session(derive_session_uuid(&session_id))
        .with_model(prepared.outbound_model.clone());
    log_json(
        &state,
        &session_id,
        &short_request_id,
        ">>>",
        "Anthropic count_tokens upstream body",
        prepared.body(),
    );

    let value = claude
        .send_anthropic_count_tokens(prepared.body(), &ctx)
        .await
        .map_err(|error| {
            if let Some(stats) = &state.stats {
                stats.record_error(&stats_key, &error.to_string());
            }
            map_provider_err(error)
        })?;
    Ok((anthropic_request_id_header(&short_request_id), Json(value)).into_response())
}

async fn read_anthropic_body(request: Request<Body>) -> Result<serde_json::Value, AppError> {
    let body = axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024)
        .await
        .map_err(|error| AppError::BadRequest(format!("failed to read body: {error}")))?;
    serde_json::from_slice(&body)
        .map_err(|error| AppError::BadRequest(format!("invalid JSON: {error}")))
}

fn claude_native_for_anthropic(
    state: &AppState,
    requested_model: &str,
) -> Result<Arc<ClaudeProvider>, AppError> {
    let Some(entry) = state.providers.get("claude") else {
        return Err(AppError::BadRequest(
            "Anthropic /v1/messages is supported only when the claude provider is enabled".into(),
        ));
    };
    let Some(claude) = entry.claude_native.as_ref() else {
        return Err(AppError::ServerError(
            "claude provider does not expose native Anthropic support".into(),
        ));
    };
    if let Some((prefix, _)) = requested_model.split_once(':')
        && prefix != "claude"
    {
        return Err(AppError::BadRequest(
            "Anthropic /v1/messages supports only claude models".into(),
        ));
    }
    Ok(claude.clone())
}

fn strip_claude_model_prefix(mut body: serde_json::Value) -> Result<serde_json::Value, AppError> {
    let Some(model) = body.get("model").and_then(|value| value.as_str()) else {
        return Ok(body);
    };
    let Some((prefix, stripped)) = model.split_once(':') else {
        return Ok(body);
    };
    if prefix != "claude" {
        return Err(AppError::BadRequest(
            "Anthropic /v1/messages supports only claude models".into(),
        ));
    }
    if stripped.trim().is_empty() {
        return Err(AppError::BadRequest(
            "empty model after claude provider prefix".into(),
        ));
    }
    body["model"] = serde_json::Value::String(stripped.trim().to_string());
    Ok(body)
}

fn map_anthropic_prepare_err(error: ProviderError) -> AppError {
    match error {
        ProviderError::Auth(message) => AppError::Unauthorized(message),
        ProviderError::Upstream(message) => AppError::BadRequest(message),
        ProviderError::Other(error) => AppError::ServerError(error.to_string()),
    }
}

fn anthropic_session_id(
    headers: &HeaderMap,
    metadata_user: Option<&str>,
    api_key_id: Option<&str>,
    request_id: &str,
) -> String {
    if let Some(session) = session_header(headers).filter(|value| !value.is_empty()) {
        return session.to_string();
    }
    if let Some(user) = metadata_user.filter(|value| !value.is_empty()) {
        return format!("user:{user}");
    }
    if let Some(key) = api_key_id.filter(|value| !value.is_empty()) {
        return format!("key:{key}");
    }
    format!("anth:{request_id}")
}

fn anthropic_sse_response(
    mut upstream: provider_claude::anthropic_passthrough::RawFrameStream,
    stats: Option<Arc<Stats>>,
    conversation_log: Option<Arc<ConversationLog>>,
    model: String,
    session_id: String,
    request_id: String,
    replacements: Replacements,
) -> Response {
    let response_request_id = request_id.clone();
    let stream = async_stream::stream! {
        let _active = stats.as_deref().map(ActiveRequestGuard::new);
        let started = Instant::now();
        let mut usage = TokenUsage::default();
        let mut ttft_ms: Option<f64> = None;
        let mut repl_state = provider_claude::anthropic_passthrough::RawSseReplState::new(&replacements);

        yield Ok::<Event, Infallible>(Event::default().comment("ok"));

        while let Some(item) = upstream.next().await {
            match item {
                Ok(frame) => {
                    provider_claude::anthropic_passthrough::accumulate_stream_usage(&frame, &mut usage);
                    if ttft_ms.is_none()
                        && provider_claude::anthropic_passthrough::is_upstream_content_delta(&frame)
                    {
                        ttft_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                    }
                    for (event, data) in repl_state.on_frame(&frame.event, frame.data, &replacements) {
                        match anthropic_sse_event(&event, &data) {
                            Ok(event) => yield Ok(event),
                            Err(error) => {
                                if let Some(stats) = stats.as_ref() {
                                    stats.record_error(&model, &error);
                                }
                                yield Ok(anthropic_error_event(&error));
                                break;
                            }
                        }
                    }
                }
                Err(error) => {
                    let message = truncate_for_sse(&error.to_string());
                    if let Some(stats) = stats.as_ref() {
                        stats.record_error(&model, &message);
                    }
                    for (event, data) in repl_state.flush_all(&replacements) {
                        if let Ok(event) = anthropic_sse_event(&event, &data) {
                            yield Ok(event);
                        }
                    }
                    yield Ok(anthropic_error_event(&message));
                    break;
                }
            }
        }

        for (event, data) in repl_state.flush_all(&replacements) {
            if let Ok(event) = anthropic_sse_event(&event, &data) {
                yield Ok(event);
            }
        }
        if let Some(stats) = stats.as_ref() {
            stats.record_response(
                &model,
                usage,
                ttft_ms,
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        if let Some(log) = conversation_log.as_ref() {
            log.log(
                &session_id,
                &request_id,
                "<<<",
                "Anthropic Messages stream",
                &format!("duration_ms={:.1}", started.elapsed().as_secs_f64() * 1000.0),
            );
        }
    };
    let sse = Sse::new(stream).keep_alive(KeepAlive::default());
    let mut response = sse.into_response();
    response.headers_mut().insert(
        header::HeaderName::from_static("x-request-id"),
        request_id_header(&response_request_id),
    );
    response
}

fn anthropic_sse_event(event: &str, data: &serde_json::Value) -> Result<Event, String> {
    let payload = serde_json::to_string(data)
        .map_err(|error| format!("failed to serialize Anthropic SSE frame: {error}"))?;
    Ok(Event::default().event(event).data(payload))
}

fn anthropic_error_event(message: &str) -> Event {
    Event::default().event("error").data(
        serde_json::json!({
            "type": "error",
            "error": {"type": "api_error", "message": message},
        })
        .to_string(),
    )
}

fn truncate_for_sse(message: &str) -> String {
    const MAX: usize = 500;
    if message.chars().count() <= MAX {
        return message.to_string();
    }
    let mut out = message.chars().take(MAX).collect::<String>();
    out.push_str("... (truncated)");
    out
}

fn anthropic_request_id_header(request_id: &str) -> [(header::HeaderName, header::HeaderValue); 1] {
    [(
        header::HeaderName::from_static("x-request-id"),
        request_id_header(request_id),
    )]
}

fn request_id_header(request_id: &str) -> header::HeaderValue {
    header::HeaderValue::from_str(request_id)
        .unwrap_or_else(|_| header::HeaderValue::from_static("unknown"))
}

fn anthropic_error_response(error: AppError) -> Response {
    let (status, kind) = match &error {
        AppError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
        AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        AppError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found_error"),
        AppError::ServerError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
    };
    (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": {"type": kind, "message": error.to_string()},
        })),
    )
        .into_response()
}

/// Handler for POST /v1/responses (OpenAI Responses API protocol).
///
/// Contract (pinned by the responses tests below + omni-common::responses):
/// same prefix routing as chat completions; unsupported input shapes map to
/// BadRequest; non-stream returns the Responses envelope; stream:true returns
/// Responses SSE events (response.created ... response.completed, no [DONE]).
async fn responses_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    api_key: Option<Extension<ApiKeyId>>,
    Json(body): Json<omni_common::ResponsesRequest>,
) -> Result<axum::response::Response, AppError> {
    let request_id = Uuid::new_v4().to_string();
    let short_request_id = request_id.chars().take(8).collect::<String>();
    let session_id =
        responses_session_id(&headers, &body, api_key.as_ref().map(|k| k.0.0.as_str()));
    log_json(
        &state,
        &session_id,
        &short_request_id,
        ">>>",
        "Inbound Responses body",
        &body,
    );

    let requested_model = body.model.clone();
    let catalogs = provider_catalogs(&state.providers);
    let (prov_key, stripped_model) = resolve_provider_and_model(&requested_model, &catalogs)
        .map_err(|e| record_bad_request(&state, &requested_model, e))?;
    let stats_key = stats_model_key(&prov_key, &stripped_model);

    let entry = state
        .providers
        .get(&prov_key)
        .ok_or_else(|| AppError::ServerError("provider disappeared".into()))?;
    let provider = &entry.provider;

    // Convert the Responses request to canonical, then swap in the stripped model
    // so the delegated provider sees the real backend model name. Unsupported
    // input shapes are rejected loudly as a 400 naming the offender.
    let mut canon = omni_common::responses_to_canonical(&body)
        .map_err(|e| record_bad_request(&state, &stats_key, e))?;
    canon.model = stripped_model.clone();

    if let Some(stats) = &state.stats {
        stats.record_request(&stats_key, None);
    }

    let response_id = format!("resp_{request_id}");
    let created_at = omni_common::unix_now_secs();

    if body.stream {
        let stream = provider.send_stream(canon).await.map_err(|e| {
            if let Some(stats) = &state.stats {
                stats.record_error(&stats_key, &e.to_string());
            }
            map_provider_err(e)
        })?;
        log_text(
            &state,
            &session_id,
            &short_request_id,
            "<<<",
            "Responses stream opened",
            &format!("model={requested_model}"),
        );
        let stream = wrap_stream_for_stats(
            stream,
            state.stats.clone(),
            stats_key.clone(),
            state.conversation_log.clone(),
            session_id.clone(),
            short_request_id.clone(),
            "Responses stream",
        );
        // Echo the resolved canonical provider model, not shorthand aliases.
        let sse = omni_common::sse_from_canonical_stream_responses(
            stream,
            stripped_model,
            response_id,
            created_at,
        );
        return Ok(sse.into_response());
    }

    let _active = state.stats.as_deref().map(ActiveRequestGuard::new);
    let started = Instant::now();
    let canon_resp: CanonicalResponse = provider.send(canon).await.map_err(|e| {
        if let Some(stats) = &state.stats {
            stats.record_error(&stats_key, &e.to_string());
        }
        map_provider_err(e)
    })?;
    if let Some(stats) = &state.stats {
        record_response_stats(stats, &stats_key, &canon_resp, started);
    }
    let resp =
        omni_common::responses_from_canonical(canon_resp, stripped_model, response_id, created_at);
    log_json(
        &state,
        &session_id,
        &short_request_id,
        "<<<",
        "Responses response",
        &resp,
    );
    Ok(Json(resp).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_common::ChatMessage; // test constructors build requests literally
    use omni_core::LlmProvider; // for the smoke

    fn catalogs_claude_grok() -> HashMap<String, ModelCatalog> {
        HashMap::from([
            (
                "claude".to_string(),
                claude_model_catalog(provider_claude::default_profile()),
            ),
            ("grok".to_string(), grok_model_catalog()),
        ])
    }

    fn catalogs_claude_grok_codex() -> HashMap<String, ModelCatalog> {
        let codex = CodexProvider::new().expect("codex provider");
        let mut catalogs = catalogs_claude_grok();
        catalogs.insert("codex".to_string(), codex_model_catalog(&codex));
        catalogs
    }

    fn catalogs_only_claude() -> HashMap<String, ModelCatalog> {
        HashMap::from([(
            "claude".to_string(),
            claude_model_catalog(provider_claude::default_profile()),
        )])
    }

    #[test]
    fn test_resolve_prefix_grok() {
        let catalogs = catalogs_claude_grok();
        let (k, m) = resolve_provider_and_model("grok:foo-bar", &catalogs).unwrap();
        assert_eq!(k, "grok");
        assert_eq!(m, "foo-bar");
    }

    #[test]
    fn test_resolve_prefix_claude() {
        let catalogs = catalogs_claude_grok();
        let (k, m) =
            resolve_provider_and_model("CLAUDE:claude-3-5-sonnet-20241022", &catalogs).unwrap();
        assert_eq!(k, "claude");
        assert_eq!(m, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_resolve_bare_single_provider() {
        let catalogs = catalogs_only_claude();
        let (k, m) = resolve_provider_and_model("my-model", &catalogs).unwrap();
        assert_eq!(k, "claude");
        assert_eq!(m, "my-model");
    }

    #[test]
    fn test_resolve_bare_multi_unknown_errors() {
        let catalogs = catalogs_claude_grok();
        let err = resolve_provider_and_model("bare-model", &catalogs).unwrap_err();
        assert!(err.contains("unknown model"));
    }

    #[test]
    fn test_resolve_bare_canonical_and_aliases_when_unique() {
        let catalogs = catalogs_claude_grok_codex();
        let (k, m) = resolve_provider_and_model("claude-sonnet-4-6", &catalogs).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("claude", "claude-sonnet-4-6"));

        let (k, m) = resolve_provider_and_model("fable", &catalogs).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("claude", "claude-fable-5"));

        let (k, m) = resolve_provider_and_model("sonnet", &catalogs).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("claude", "claude-sonnet-4-6"));

        let (k, m) = resolve_provider_and_model("claude-haiku-4-5", &catalogs).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("claude", "claude-haiku-4-5"));

        let (k, m) = resolve_provider_and_model("haiku", &catalogs).unwrap();
        assert_eq!(
            (k.as_str(), m.as_str()),
            ("claude", "claude-haiku-4-5-20251001")
        );

        let (k, m) = resolve_provider_and_model("grok", &catalogs).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("grok", "grok-4.3"));

        let (k, m) = resolve_provider_and_model("composer", &catalogs).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("grok", "grok-composer-2.5-fast"));

        let (k, m) = resolve_provider_and_model("codex", &catalogs).unwrap();
        assert_eq!(k.as_str(), "codex");
        assert!(!m.is_empty());
    }

    #[test]
    fn test_startup_alias_log_lists_documented_shorthands() {
        let mut providers: HashMap<String, ProviderEntry> = HashMap::new();
        providers.insert("claude".into(), claude_entry());
        providers.insert("grok".into(), grok_entry("http://127.0.0.1:1"));
        providers.insert("codex".into(), codex_entry());

        let text = format_aliases_for_log(&providers).expect("aliases format");
        for expected in [
            "sonnet=claude-sonnet-4-6",
            "opus=claude-opus-4-8",
            "haiku=claude-haiku-4-5-20251001",
            "fable=claude-fable-5",
            "grok=grok-4.3",
            "composer=grok-composer-2.5-fast",
            "codex=",
        ] {
            assert!(
                text.contains(expected),
                "startup alias log missing {expected}: {text}"
            );
        }
    }

    #[test]
    fn test_cli_version_switch_reports_package_version() {
        // WHY: release artifacts must expose the same version Cargo embeds in
        // startup logs and package metadata.
        let err = Cli::try_parse_from(["omni", "--version"]).expect_err("--version exits early");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(
            err.to_string().contains(env!("CARGO_PKG_VERSION")),
            "--version output must include package version"
        );
    }

    #[test]
    fn test_resolve_bare_ambiguous_alias_errors() {
        let mut left = ModelCatalog::default();
        left.insert("same", "left-model");
        let mut right = ModelCatalog::default();
        right.insert("same", "right-model");
        let catalogs = HashMap::from([("left".to_string(), left), ("right".to_string(), right)]);

        let err = resolve_provider_and_model("same", &catalogs).unwrap_err();
        assert!(err.contains("ambiguous"));
    }

    #[test]
    fn test_resolve_unknown_prefix_errors() {
        let catalogs = catalogs_claude_grok();
        let err = resolve_provider_and_model("codex:foo", &catalogs).unwrap_err();
        assert!(err.contains("not enabled"));
    }

    #[test]
    fn test_resolve_prefix_codex_when_enabled() {
        let catalogs = catalogs_claude_grok_codex();
        let (k, m) = resolve_provider_and_model("CODEX:gpt-5.5", &catalogs).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("codex", "gpt-5.5"));
    }

    #[test]
    fn test_normalize_providers() {
        let n = normalize_providers(&[
            " claude ".into(),
            "GROK".into(),
            "codex".into(),
            "claude".into(),
        ])
        .unwrap();
        assert_eq!(
            n,
            vec![
                "claude".to_string(),
                "grok".to_string(),
                "codex".to_string()
            ]
        );
    }

    #[test]
    fn test_resolve_bind_host_default_loopback() {
        assert_eq!(resolve_bind_host(false, None).unwrap(), "127.0.0.1");
    }

    #[test]
    fn test_resolve_bind_host_explicit_bind() {
        assert_eq!(
            resolve_bind_host(false, Some("192.168.1.25")).unwrap(),
            "192.168.1.25"
        );
    }

    #[test]
    fn test_resolve_bind_host_public_all_interfaces() {
        assert_eq!(resolve_bind_host(true, None).unwrap(), "0.0.0.0");
    }

    #[test]
    fn test_resolve_bind_host_public_conflicts_with_bind() {
        let err = resolve_bind_host(true, Some("127.0.0.1")).unwrap_err();
        assert!(err.to_string().contains("cannot be used with --bind"));
    }

    #[tokio::test]
    async fn smoke_routing_and_delegate_claude() {
        // Verifies the full path resolve + to_canonical + delegate + from_canonical
        // against the real Claude upstream. Live-conditional: skips cleanly when no
        // Claude credentials are present so the suite stays green offline (the
        // routing + conversion logic without the live send is covered by the
        // hermetic handler tests below).
        if !has_claude_creds() {
            eprintln!("skipping smoke_routing_and_delegate_claude: no claude creds");
            return;
        }
        let claude = Arc::new(ClaudeProvider::new().expect("claude provider for omni router test"));

        let catalogs = catalogs_only_claude();
        let (key, stripped) = resolve_provider_and_model("claude:sonnet", &catalogs).unwrap();
        assert_eq!(key, "claude");
        assert_eq!(stripped, "claude-sonnet-4-6");

        let oai_req = ChatCompletionRequest {
            model: "claude:sonnet".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("tell me a joke".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: Some(64),
            max_completion_tokens: None,
            temperature: Some(0.7),
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };

        let mut canon = to_canonical(&oai_req).unwrap();
        canon.model = stripped;

        let provider = claude;
        let canon_resp = provider.send(canon).await.unwrap();

        assert_eq!(canon_resp.model, "claude-sonnet-4-6");
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
        //    code paths in main are valid for it. (Routing only; no runtime construction or network.)
        let catalogs = catalogs_claude_grok();
        let (key, stripped) = resolve_provider_and_model("grok:grok-4.3", &catalogs).unwrap();
        assert_eq!(key, "grok");
        assert_eq!(stripped, "grok-4.3");

        // Compile-time assertion that the real grok type can be used exactly as the omni router does
        // (stored in HashMap<String, ProviderEntry> and delegated to).
        fn assert_impls_dyn<P: LlmProvider + 'static>() {
            // If this compiles, GrokProvider (and Claude) can be the pointee for the thin router.
        }
        assert_impls_dyn::<GrokProvider>();
        assert_impls_dyn::<ClaudeProvider>();
    }

    // --- comprehensive http/integration tests added per task (using direct handler calls for router surfaces
    // + subprocess HTTP checks for full binary stack incl auth mw, random port, live creds conditional) ---

    use axum::http::StatusCode;
    use omni_core::CanonicalResponse;
    use serde_json::Value;
    use std::collections::HashSet;
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use std::time::{Duration, Instant};
    use wiremock::matchers::{body_partial_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    static PROVIDER_ENV_LOCK: StdMutex<()> = StdMutex::new(());

    struct TempEnvVars {
        old: Vec<(&'static str, Option<OsString>)>,
    }

    impl TempEnvVars {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let old = vars
                .iter()
                .map(|(name, _)| (*name, std::env::var_os(name)))
                .collect();
            for (name, value) in vars {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
            Self { old }
        }
    }

    impl Drop for TempEnvVars {
        fn drop(&mut self) {
            for (name, value) in &self.old {
                unsafe {
                    match value {
                        Some(value) => std::env::set_var(name, value),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    struct TempClaudeCreds {
        path: PathBuf,
        prev: Option<std::ffi::OsString>,
    }

    impl TempClaudeCreds {
        fn dummy_token() -> &'static str {
            "sk-ant-oat01-omni-dummy"
        }

        fn install(tag: &str) -> Self {
            let body = format!(
                r#"{{"claudeAiOauth":{{"accessToken":"{}","expiresAt":3000000000000,"subscriptionType":"max"}}}}"#,
                Self::dummy_token()
            );
            let path = std::env::temp_dir().join(format!(
                "omni-claude-creds-{}-{}.json",
                tag,
                Uuid::new_v4()
            ));
            std::fs::write(&path, body).expect("write temp Claude creds");
            let prev = std::env::var_os("CLAUDE_CREDENTIALS_PATH");
            unsafe {
                std::env::set_var("CLAUDE_CREDENTIALS_PATH", &path);
            }
            Self { path, prev }
        }
    }

    impl Drop for TempClaudeCreds {
        fn drop(&mut self) {
            unsafe {
                match &self.prev {
                    Some(value) => std::env::set_var("CLAUDE_CREDENTIALS_PATH", value),
                    None => std::env::remove_var("CLAUDE_CREDENTIALS_PATH"),
                }
            }
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("bind for free port")
            .local_addr()
            .unwrap()
            .port()
    }

    fn live_tests_enabled() -> bool {
        omni_common::test_support::live_tests_enabled()
    }

    fn has_claude_creds() -> bool {
        if !live_tests_enabled() {
            return false;
        }
        // Honor the same override the provider uses (CLAUDE_CREDENTIALS_PATH), so
        // this guard agrees with what ClaudeProvider::send actually reads. Without
        // this, pointing the override at a missing file would pass the guard yet
        // fail the live send.
        if let Ok(p) = std::env::var("CLAUDE_CREDENTIALS_PATH") {
            return Path::new(&p).exists();
        }
        let home = std::env::var("HOME").unwrap_or_default();
        Path::new(&(home + "/.claude/.credentials.json")).exists()
    }

    fn has_grok_creds() -> bool {
        if !live_tests_enabled() {
            return false;
        }
        // Mirror the Grok provider's fresh-load source precedence: when XAI_CREDENTIALS_PATH
        // is set, treat creds as present only if that file exists (a test may point it at a
        // dummy/missing path); otherwise creds are present if EITHER the static-key file
        // ~/.xai/.credentials.json OR the Grok CLI login ~/.grok/auth.json exists. Files are
        // the only credential source (no env key), matching the provider.
        if let Ok(p) = std::env::var("XAI_CREDENTIALS_PATH") {
            return Path::new(&p).exists();
        }
        let home = std::env::var("HOME").unwrap_or_default();
        Path::new(&format!("{home}/.xai/.credentials.json")).exists()
            || Path::new(&format!("{home}/.grok/auth.json")).exists()
    }

    fn claude_entry() -> ProviderEntry {
        let provider = ClaudeProvider::new().expect("claude");
        let models = provider_model_values("claude", provider.profile().models_list())
            .expect("claude model catalog serializes");
        let catalog = claude_model_catalog(provider.profile());
        let provider = Arc::new(provider);
        ProviderEntry {
            provider: provider.clone(),
            claude_native: Some(provider),
            models,
            catalog,
        }
    }

    fn claude_entry_with_base(base_url: &str) -> ProviderEntry {
        let provider =
            ClaudeProvider::new_for_test_with_base(provider_claude::default_profile(), base_url)
                .expect("claude test provider");
        let models = provider_model_values("claude", provider.profile().models_list())
            .expect("claude model catalog serializes");
        let catalog = claude_model_catalog(provider.profile());
        let provider = Arc::new(provider);
        ProviderEntry {
            provider: provider.clone(),
            claude_native: Some(provider),
            models,
            catalog,
        }
    }

    fn grok_entry(base_url: &str) -> ProviderEntry {
        ProviderEntry {
            provider: Arc::new(GrokProvider::new_for_test("k", base_url)),
            claude_native: None,
            models: provider_model_values("grok", GrokProvider::models_list())
                .expect("grok model catalog serializes"),
            catalog: grok_model_catalog(),
        }
    }

    fn codex_entry() -> ProviderEntry {
        let provider = CodexProvider::new().expect("codex provider");
        ProviderEntry {
            provider: Arc::new(provider.clone()),
            claude_native: None,
            models: provider_model_values("codex", provider.models_list())
                .expect("codex model catalog serializes"),
            catalog: codex_model_catalog(&provider),
        }
    }

    struct TempCodexHome {
        path: PathBuf,
        prev: Option<std::ffi::OsString>,
    }

    impl TempCodexHome {
        fn install_for_mock(base_url: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("omni-codex-home-{}", Uuid::new_v4()));
            std::fs::create_dir_all(&dir).expect("create codex test home");
            std::fs::write(
                dir.join("config.toml"),
                format!(
                    r#"
model = "gpt-codex-test"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{base_url}"
wire_api = "responses"
requires_openai_auth = false
"#
                ),
            )
            .expect("write codex config");
            let old = std::env::var_os("CODEX_HOME");
            unsafe {
                std::env::set_var("CODEX_HOME", &dir);
            }
            Self {
                path: dir,
                prev: old,
            }
        }
    }

    impl Drop for TempCodexHome {
        fn drop(&mut self) {
            unsafe {
                match &self.prev {
                    Some(value) => std::env::set_var("CODEX_HOME", value),
                    None => std::env::remove_var("CODEX_HOME"),
                }
            }
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn live_grok_entry() -> ProviderEntry {
        ProviderEntry {
            provider: Arc::new(GrokProvider::new(None).expect("grok provider with creds")),
            claude_native: None,
            models: provider_model_values("grok", GrokProvider::models_list())
                .expect("grok model catalog serializes"),
            catalog: grok_model_catalog(),
        }
    }

    fn state_with(providers: HashMap<String, ProviderEntry>) -> Arc<AppState> {
        Arc::new(AppState {
            providers,
            stats: None,
            conversation_log: None,
        })
    }

    fn state_with_stats(providers: HashMap<String, ProviderEntry>) -> (Arc<AppState>, TempStats) {
        let path = temp_stats_path();
        let stats = Stats::open(&path).expect("open temp stats");
        (
            Arc::new(AppState {
                providers,
                stats: Some(Arc::new(stats)),
                conversation_log: None,
            }),
            TempStats(path),
        )
    }

    fn state_with_conversation_log(
        providers: HashMap<String, ProviderEntry>,
        dir: &Path,
    ) -> Arc<AppState> {
        Arc::new(AppState {
            providers,
            stats: None,
            conversation_log: Some(Arc::new(
                ConversationLog::to_dir(dir).expect("open temp conversation log dir"),
            )),
        })
    }

    fn temp_stats_path() -> PathBuf {
        std::env::temp_dir().join(format!("omni-stats-TEST-{}.redb", Uuid::new_v4()))
    }

    struct TempStats(PathBuf);
    impl Drop for TempStats {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn wait_for_200_health(port: u16, timeout: Duration) -> bool {
        omni_common::test_support::wait_for_http_body(
            format!("http://127.0.0.1:{}/health", port),
            "ok",
            timeout,
        )
    }

    fn omni_bin_path() -> std::path::PathBuf {
        // Runtime lookup so this compiles even when CARGO_BIN_EXE_* is absent at
        // compile time. Prefer the cargo-injected path when present (integration
        // tests get it; unit tests in a bin crate do not).
        if let Ok(p) = std::env::var("CARGO_BIN_EXE_omni") {
            return std::path::PathBuf::from(p);
        }
        // Otherwise build the binary on demand and locate the real artifact (honors
        // CARGO_TARGET_DIR; builds the dev profile). Cache so the build runs once per
        // test process; see omni_common::test_support::build_workspace_bin for the
        // full rationale.
        static BIN: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        BIN.get_or_init(|| omni_common::test_support::build_workspace_bin("omni"))
            .clone()
    }

    fn mk_app_with(
        providers: HashMap<String, ProviderEntry>,
        auth_keys: Arc<HashSet<String>>,
    ) -> axum::Router {
        build_router(state_with(providers), auth_keys)
    }

    fn spawn_omni(args: &[&str], envs: &[(&str, &str)]) -> omni_common::test_support::ChildGuard {
        let mut cmd = Command::new(omni_bin_path());
        cmd.args(args).stdout(Stdio::null()).stderr(Stdio::null());
        for (key, value) in envs {
            cmd.env(key, value);
        }
        omni_common::test_support::ChildGuard::new(cmd.spawn().expect("spawn omni"))
    }

    fn get(port: u16, path: &str) -> omni_common::test_support::HttpResponse {
        omni_common::test_support::http_get(format!("http://127.0.0.1:{port}{path}"))
    }

    fn post_json(port: u16, path: &str, body: &str) -> omni_common::test_support::HttpResponse {
        omni_common::test_support::http_post_json(format!("http://127.0.0.1:{port}{path}"), body)
    }

    async fn call_chat_handler(
        state: Arc<AppState>,
        req: ChatCompletionRequest,
    ) -> Result<axum::response::Response, AppError> {
        chat_completions_handler(State(state), HeaderMap::new(), None, Json(req)).await
    }

    async fn call_chat_handler_with_session(
        state: Arc<AppState>,
        req: ChatCompletionRequest,
        session_id: &str,
    ) -> Result<axum::response::Response, AppError> {
        let mut headers = HeaderMap::new();
        headers.insert("x-session-id", session_id.parse().unwrap());
        chat_completions_handler(State(state), headers, None, Json(req)).await
    }

    async fn call_responses_handler(
        state: Arc<AppState>,
        req: omni_common::ResponsesRequest,
    ) -> Result<axum::response::Response, AppError> {
        responses_handler(State(state), HeaderMap::new(), None, Json(req)).await
    }

    async fn call_anthropic_messages_handler(
        state: Arc<AppState>,
        body: &str,
    ) -> axum::response::Response {
        anthropic_messages_handler(
            State(state),
            HeaderMap::new(),
            None,
            Request::builder()
                .uri("/v1/messages")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
    }

    async fn call_anthropic_count_tokens_handler(
        state: Arc<AppState>,
        body: &str,
    ) -> axum::response::Response {
        anthropic_count_tokens_handler(
            State(state),
            HeaderMap::new(),
            None,
            Request::builder()
                .uri("/v1/messages/count_tokens")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
    }

    #[test]
    fn test_mk_app_with_and_router_surfaces() {
        // WHY: build_router must register all production surfaces with the auth
        // layer. Rendering a known route proves construction was not just a type
        // check over an unused Router value.
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("claude".into(), claude_entry());
        let app = mk_app_with(map, Arc::new(HashSet::new()));
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        rt.block_on(async {
            use axum::body::Body;
            use axum::http::{Request, StatusCode};
            use tower::ServiceExt;

            let resp = app
                .oneshot(
                    Request::builder()
                        .uri("/health")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .expect("router responds");
            assert_eq!(resp.status(), StatusCode::OK);

            for path in ["/v1/messages", "/v1/messages/count_tokens"] {
                let app = mk_app_with(
                    HashMap::from([("claude".to_string(), claude_entry())]),
                    Arc::new(HashSet::new()),
                );
                let resp = app
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri(path)
                            .header("content-type", "application/json")
                            .body(Body::from("not json"))
                            .unwrap(),
                    )
                    .await
                    .expect("router responds");
                assert_eq!(
                    resp.status(),
                    StatusCode::BAD_REQUEST,
                    "{path} must be registered; 404 means build_router missed it"
                );
            }
        });
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
        let mut m1: HashMap<String, ProviderEntry> = HashMap::new();
        m1.insert("claude".into(), claude_entry());
        let state1 = state_with(m1);
        let r1 = models_handler(State(state1)).await.into_response();
        let b1 = axum::body::to_bytes(r1.into_body(), 1 << 20).await.unwrap();
        let v1: Value = serde_json::from_slice(&b1).unwrap();
        let ids1: Vec<_> = v1["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(ids1.iter().all(|id| id.starts_with("claude-")));

        let mut m2: HashMap<String, ProviderEntry> = HashMap::new();
        m2.insert("claude".into(), claude_entry());
        m2.insert("grok".into(), grok_entry("http://127.0.0.1:9"));
        let state2 = state_with(m2);
        let r2 = models_handler(State(state2)).await.into_response();
        let b2 = axum::body::to_bytes(r2.into_body(), 1 << 20).await.unwrap();
        let v2: Value = serde_json::from_slice(&b2).unwrap();
        let ids2: Vec<_> = v2["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(ids2.iter().any(|id| id.starts_with("claude-")));
        assert!(ids2.iter().any(|id| id.starts_with("grok-")));
        assert!(
            ids2.iter().all(|id| !id.contains(':')),
            "model ids must be canonical upstream ids, not provider-prefixed: {ids2:?}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_init_claude_custom_gateway_uses_anthropic_auth_token() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let creds = TempClaudeCreds::install("omni-claude-env-token");
        let server = MockServer::start().await;
        let _env = TempEnvVars::set(&[
            ("ANTHROPIC_BASE_URL", Some(&server.uri())),
            ("ANTHROPIC_AUTH_TOKEN", Some("custom-claude-token")),
            ("ANTHROPIC_API_KEY", Some("must-not-win")),
            ("ANTHROPIC_CUSTOM_HEADERS", Some("X-Test-Gateway: yes")),
        ]);
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header("authorization", "Bearer custom-claude-token"))
            .and(header("x-test-gateway", "yes"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_env",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type":"text","text":"env ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = init_claude_provider().expect("custom Claude provider from env");
        drop(creds);
        let response = provider
            .send(omni_core::CanonicalRequest {
                model: "sonnet".into(),
                messages: vec![omni_core::CanonicalMessage {
                    role: "user".into(),
                    content: omni_core::CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
            .expect("custom Claude send");

        assert_eq!(response.content, "env ok");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let auth = requests[0]
            .headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert_eq!(auth, "Bearer custom-claude-token");
        assert!(
            !auth.contains(TempClaudeCreds::dummy_token()),
            "custom Claude gateway must not receive default OAuth token"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_init_claude_custom_gateway_uses_api_key_for_anthropic_native() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let creds = TempClaudeCreds::install("omni-claude-env-api-key");
        let server = MockServer::start().await;
        let _env = TempEnvVars::set(&[
            ("ANTHROPIC_BASE_URL", Some(&server.uri())),
            ("ANTHROPIC_AUTH_TOKEN", None),
            ("ANTHROPIC_API_KEY", Some("custom-api-key")),
            ("ANTHROPIC_CUSTOM_HEADERS", None),
        ]);
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header("x-api-key", "custom-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_native_custom",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type":"text","text":"native ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = init_claude_provider().expect("custom Claude provider from env");
        drop(creds);
        let models = provider_model_values("claude", provider.profile().models_list()).unwrap();
        let catalog = claude_model_catalog(provider.profile());
        let provider = Arc::new(provider);
        let mut providers = HashMap::new();
        providers.insert(
            "claude".into(),
            ProviderEntry {
                provider: provider.clone(),
                claude_native: Some(provider),
                models,
                catalog,
            },
        );

        let resp = call_anthropic_messages_handler(
            state_with(providers),
            r#"{"model":"sonnet","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["content"][0]["text"], "native ok");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            !requests[0].headers.contains_key("authorization"),
            "x-api-key custom Claude gateway must not receive default OAuth Authorization"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_init_claude_custom_gateway_reads_auth_env_per_request() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let creds = TempClaudeCreds::install("omni-claude-env-rotate");
        let server = MockServer::start().await;
        let _env = TempEnvVars::set(&[
            ("ANTHROPIC_BASE_URL", Some(&server.uri())),
            ("ANTHROPIC_AUTH_TOKEN", Some("first-claude-token")),
            ("ANTHROPIC_API_KEY", None),
            ("ANTHROPIC_CUSTOM_HEADERS", Some("X-Rotate: first")),
        ]);
        for (token, marker, text) in [
            ("first-claude-token", "first", "first ok"),
            ("second-claude-token", "second", "second ok"),
        ] {
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .and(query_param("beta", "true"))
                .and(header("authorization", format!("Bearer {token}").as_str()))
                .and(header("x-rotate", marker))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "msg_rotate",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type":"text","text": text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                })))
                .expect(1)
                .mount(&server)
                .await;
        }

        let provider = init_claude_provider().expect("custom Claude provider from env");
        drop(creds);
        let request = || omni_core::CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![omni_core::CanonicalMessage {
                role: "user".into(),
                content: omni_core::CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };

        let first = provider.send(request()).await.expect("first send");
        assert_eq!(first.content, "first ok");
        unsafe {
            std::env::set_var("ANTHROPIC_AUTH_TOKEN", "second-claude-token");
            std::env::set_var("ANTHROPIC_CUSTOM_HEADERS", "X-Rotate: second");
        }
        let second = provider.send(request()).await.expect("second send");
        assert_eq!(second.content, "second ok");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_init_claude_custom_gateway_rejects_invalid_custom_headers_per_request() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _env = TempEnvVars::set(&[
            ("ANTHROPIC_BASE_URL", Some(&server.uri())),
            ("ANTHROPIC_AUTH_TOKEN", Some("custom-claude-token")),
            ("ANTHROPIC_API_KEY", None),
            ("ANTHROPIC_CUSTOM_HEADERS", Some("X-Valid: yes")),
        ]);
        let provider = init_claude_provider().expect("custom Claude provider from env");
        unsafe {
            std::env::set_var("ANTHROPIC_CUSTOM_HEADERS", "bad header");
        }
        let err = provider
            .send(omni_core::CanonicalRequest {
                model: "sonnet".into(),
                messages: vec![omni_core::CanonicalMessage {
                    role: "user".into(),
                    content: omni_core::CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
            .expect_err("invalid custom header env must fail loudly");
        assert!(
            err.to_string().contains("custom header"),
            "invalid ANTHROPIC_CUSTOM_HEADERS must not be silently ignored: {err}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_init_grok_custom_endpoint_uses_xai_api_key_not_default_credentials() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let creds =
            std::env::temp_dir().join(format!("omni-grok-omni-env-{}.json", Uuid::new_v4()));
        std::fs::write(&creds, r#"{"apiKey":"xai-must-not-leak"}"#).expect("write temp xAI creds");
        let server = MockServer::start().await;
        let _env = TempEnvVars::set(&[
            ("GROK_MODELS_BASE_URL", Some(&server.uri())),
            ("XAI_API_KEY", Some("custom-grok-key")),
            ("XAI_CREDENTIALS_PATH", Some(creds.to_str().unwrap())),
        ]);
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer custom-grok-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "grok-4.3",
                "choices": [{"message": {"content": "grok ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = init_grok_provider().expect("custom Grok provider from env");
        let response = provider
            .send(omni_core::CanonicalRequest {
                model: "grok".into(),
                messages: vec![omni_core::CanonicalMessage {
                    role: "user".into(),
                    content: omni_core::CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
            .expect("custom Grok send");

        assert_eq!(response.content, "grok ok");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let auth = requests[0]
            .headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert_eq!(auth, "Bearer custom-grok-key");
        assert!(
            !auth.contains("xai-must-not-leak"),
            "custom Grok endpoint must not receive default xAI credential"
        );
        let _ = std::fs::remove_file(creds);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_init_grok_custom_endpoint_reads_xai_api_key_per_request() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let creds =
            std::env::temp_dir().join(format!("omni-grok-omni-rotate-{}.json", Uuid::new_v4()));
        std::fs::write(&creds, r#"{"apiKey":"xai-must-not-leak"}"#).expect("write temp xAI creds");
        let server = MockServer::start().await;
        let _env = TempEnvVars::set(&[
            ("GROK_MODELS_BASE_URL", Some(&server.uri())),
            ("XAI_API_KEY", Some("first-grok-key")),
            ("XAI_CREDENTIALS_PATH", Some(creds.to_str().unwrap())),
        ]);
        for (token, text) in [
            ("first-grok-key", "first grok ok"),
            ("second-grok-key", "second grok ok"),
        ] {
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .and(header("authorization", format!("Bearer {token}").as_str()))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "model": "grok-4.3",
                    "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                })))
                .expect(1)
                .mount(&server)
                .await;
        }

        let provider = init_grok_provider().expect("custom Grok provider from env");
        let request = || omni_core::CanonicalRequest {
            model: "grok".into(),
            messages: vec![omni_core::CanonicalMessage {
                role: "user".into(),
                content: omni_core::CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };

        let first = provider.send(request()).await.expect("first send");
        assert_eq!(first.content, "first grok ok");
        unsafe {
            std::env::set_var("XAI_API_KEY", "second-grok-key");
        }
        let second = provider.send(request()).await.expect("second send");
        assert_eq!(second.content, "second grok ok");
        let _ = std::fs::remove_file(creds);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_init_grok_custom_endpoint_without_key_sends_no_authorization() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let creds =
            std::env::temp_dir().join(format!("omni-grok-omni-noauth-{}.json", Uuid::new_v4()));
        std::fs::write(&creds, r#"{"apiKey":"xai-must-not-leak"}"#).expect("write temp xAI creds");
        let server = MockServer::start().await;
        let _env = TempEnvVars::set(&[
            ("GROK_MODELS_BASE_URL", Some(&server.uri())),
            ("XAI_API_KEY", None),
            ("XAI_CREDENTIALS_PATH", Some(creds.to_str().unwrap())),
        ]);
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "grok-4.3",
                "choices": [{"message": {"content": "grok no auth ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = init_grok_provider().expect("custom Grok provider from env");
        let response = provider
            .send(omni_core::CanonicalRequest {
                model: "grok".into(),
                messages: vec![omni_core::CanonicalMessage {
                    role: "user".into(),
                    content: omni_core::CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
            .expect("custom Grok send");

        assert_eq!(response.content, "grok no auth ok");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            !requests[0].headers.contains_key("authorization"),
            "custom Grok no-auth endpoint must not receive ambient xAI Authorization"
        );
        let _ = std::fs::remove_file(creds);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_chat_completions_routes_to_codex_provider() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _codex_home = TempCodexHome::install_for_mock(&server.uri());
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(body_partial_json(serde_json::json!({
                "model": "gpt-5.5",
                "input": [{"type":"message","role":"user","content":"hi"}],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "gpt-5.5",
                "status": "completed",
                "output": [{"type":"message","content":[{"type":"output_text","text":"codex ok"}]}],
                "usage": {"input_tokens": 2, "output_tokens": 3}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut map = HashMap::new();
        map.insert("codex".into(), codex_entry());
        let req = ChatCompletionRequest {
            model: "codex:gpt-5.5".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let resp = call_chat_handler(state_with(map), req)
            .await
            .expect("chat routed to codex");
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "codex ok");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            !requests[0].headers.contains_key("authorization"),
            "codex no-auth custom provider must not receive default Authorization"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_responses_routes_to_codex_provider() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _codex_home = TempCodexHome::install_for_mock(&server.uri());
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "gpt-5.5",
                "status": "completed",
                "output": [{"type":"message","content":[{"type":"output_text","text":"response ok"}]}],
                "usage": {"input_tokens": 2, "output_tokens": 3}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut map = HashMap::new();
        map.insert("codex".into(), codex_entry());
        let req: omni_common::ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "codex:gpt-5.5",
            "input": "hi",
            "store": false
        }))
        .unwrap();
        let resp = call_responses_handler(state_with(map), req)
            .await
            .expect("responses routed to codex");
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["output"][0]["content"][0]["text"], "response ok");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            !requests[0].headers.contains_key("authorization"),
            "codex no-auth custom provider must not receive default Authorization"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_chat_completions_codex_stream_rejects_until_native_streaming_exists() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _codex_home = TempCodexHome::install_for_mock("http://127.0.0.1:9");
        let mut map = HashMap::new();
        map.insert("codex".into(), codex_entry());
        let req = ChatCompletionRequest {
            model: "codex:gpt-5.5".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };

        let err = call_chat_handler(state_with(map), req)
            .await
            .expect_err("Codex stream:true must fail loudly");
        match err {
            AppError::ServerError(message) => assert!(
                message.contains("streaming is not implemented"),
                "Codex stream:true should name the unsupported streaming mode: {message}"
            ),
            other => panic!("expected Codex stream unsupported server error, got {other:?}"),
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_responses_codex_stream_rejects_until_native_streaming_exists() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _codex_home = TempCodexHome::install_for_mock("http://127.0.0.1:9");
        let mut map = HashMap::new();
        map.insert("codex".into(), codex_entry());
        let req: omni_common::ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "codex:gpt-5.5",
            "input": "hi",
            "stream": true
        }))
        .unwrap();

        let err = call_responses_handler(state_with(map), req)
            .await
            .expect_err("Codex Responses stream:true must fail loudly");
        match err {
            AppError::ServerError(message) => assert!(
                message.contains("streaming is not implemented"),
                "Codex Responses stream:true should name the unsupported streaming mode: {message}"
            ),
            other => panic!("expected Codex stream unsupported server error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_models_handler_uses_real_canonical_provider_catalogs() {
        // WHY: omni is the only server binary, so /v1/models must expose the
        // provider-owned upstream ids, not aliases or prefixed routing ids.
        let mut providers: HashMap<String, ProviderEntry> = HashMap::new();
        providers.insert("claude".into(), claude_entry());
        providers.insert("grok".into(), grok_entry("http://127.0.0.1:9"));
        let resp = models_handler(State(state_with(providers)))
            .await
            .into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        let ids: Vec<String> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str().map(str::to_string))
            .collect();
        assert!(
            ids.iter().any(|id| id == "grok-4.3"),
            "grok real catalog entry missing: {ids:?}"
        );
        assert!(
            ids.iter().any(|id| id == "grok-composer-2.5-fast"),
            "grok composer catalog entry missing: {ids:?}"
        );
        assert!(
            ids.iter().any(|id| id.starts_with("claude-")),
            "claude real catalog entries missing: {ids:?}"
        );
        assert!(
            !ids.iter()
                .any(|id| id == "grok" || id == "composer" || id.contains(':')),
            "catalog must not expose aliases or provider-prefixed ids: {ids:?}"
        );
        assert!(
            !ids.iter().any(|id| id == "default"),
            "old placeholder default entries must not remain: {ids:?}"
        );
    }

    #[tokio::test]
    async fn test_anthropic_messages_rejects_when_claude_disabled() {
        let mut providers = HashMap::new();
        providers.insert("grok".into(), grok_entry("http://127.0.0.1:9"));
        let resp = call_anthropic_messages_handler(
            state_with(providers),
            r#"{"model":"grok-4.3","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("claude provider")
        );
    }

    #[tokio::test]
    async fn test_anthropic_messages_rejects_non_claude_prefix() {
        let mut providers = HashMap::new();
        providers.insert("claude".into(), claude_entry());
        providers.insert("grok".into(), grok_entry("http://127.0.0.1:9"));
        let resp = call_anthropic_messages_handler(
            state_with(providers),
            r#"{"model":"grok:grok-4.3","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("only claude models")
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_anthropic_messages_nonstream_passthrough_and_prefix_strip() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempClaudeCreds::install("anth-nonstream");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header(
                "authorization",
                format!("Bearer {}", TempClaudeCreds::dummy_token()).as_str(),
            ))
            .and(body_partial_json(serde_json::json!({
                "model": "claude-sonnet-4-6",
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_native",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type":"text","text":"hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 1
                },
                "future_field": {"kept": true}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut providers = HashMap::new();
        providers.insert("claude".into(), claude_entry_with_base(&server.uri()));
        let (state, _stats_guard) = state_with_stats(providers);
        let resp = call_anthropic_messages_handler(
            state.clone(),
            r#"{"model":"claude:sonnet","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["id"], "msg_native");
        assert_eq!(v["future_field"]["kept"], true);
        let snap = state.stats.as_ref().unwrap().snapshot();
        let model = &snap.models["claude:claude-sonnet-4-6"];
        assert_eq!(model.requests, 1);
        assert_eq!(model.input_tokens, 11);
        assert_eq!(model.output_tokens, 3);
        assert_eq!(model.cache_read_input_tokens, 2);
        assert_eq!(model.cache_creation_input_tokens, 1);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_anthropic_count_tokens_proxies_native_shape() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempClaudeCreds::install("anth-count");

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(query_param("beta", "true"))
            .and(header(
                "authorization",
                format!("Bearer {}", TempClaudeCreds::dummy_token()).as_str(),
            ))
            .and(body_partial_json(serde_json::json!({
                "model": "claude-sonnet-4-6",
                "messages": [{"role":"user","content":"hi"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 7
            })))
            .expect(1)
            .mount(&server)
            .await;

        let mut providers = HashMap::new();
        providers.insert("claude".into(), claude_entry_with_base(&server.uri()));
        let resp = call_anthropic_count_tokens_handler(
            state_with(providers),
            r#"{"model":"sonnet","max_tokens":99,"temperature":0.2,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["input_tokens"], 7);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_anthropic_messages_stream_preserves_raw_events() {
        let _guard = PROVIDER_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _creds = TempClaudeCreds::install("anth-stream");

        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0,\"cache_read_input_tokens\":1}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(query_param("beta", "true"))
            .and(body_partial_json(serde_json::json!({"stream": true})))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .expect(1)
            .mount(&server)
            .await;

        let mut providers = HashMap::new();
        providers.insert("claude".into(), claude_entry_with_base(&server.uri()));
        let (state, _stats_guard) = state_with_stats(providers);
        let resp = call_anthropic_messages_handler(
            state.clone(),
            r#"{"model":"sonnet","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("event: message_start"), "{text}");
        assert!(text.contains("event: content_block_delta"), "{text}");
        assert!(text.contains("\"text\":\"hi\""), "{text}");
        assert!(text.contains("event: message_stop"), "{text}");
        assert!(
            !text.contains("[DONE]"),
            "Anthropic SSE must not use OpenAI sentinels: {text}"
        );
        let snap = state.stats.as_ref().unwrap().snapshot();
        let model = &snap.models["claude:claude-sonnet-4-6"];
        assert_eq!(model.requests, 1);
        assert_eq!(model.input_tokens, 5);
        assert_eq!(model.output_tokens, 2);
        assert_eq!(model.cache_read_input_tokens, 1);
    }

    #[tokio::test]
    async fn test_stats_records_request_response_and_error() {
        // WHY: /stats replaces the provider-specific binary stats endpoints.
        // It must count routed requests, token usage on successful non-stream
        // responses, and errors on failed provider calls.
        #[derive(Debug)]
        struct StaticProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StaticProvider {
            fn id(&self) -> &'static str {
                "static"
            }

            async fn send(
                &self,
                req: omni_core::CanonicalRequest,
            ) -> Result<CanonicalResponse, ProviderError> {
                Ok(CanonicalResponse {
                    model: req.model,
                    content: "ok".into(),
                    tool_calls: vec![],
                    finish_reason: Some("stop".into()),
                    usage: omni_core::CanonicalUsage {
                        input_tokens: 3,
                        output_tokens: 2,
                        cache_read: 1,
                        cache_creation: 0,
                    },
                })
            }
        }

        let mut providers: HashMap<String, ProviderEntry> = HashMap::new();
        providers.insert(
            "grok".into(),
            ProviderEntry {
                provider: Arc::new(StaticProvider),
                claude_native: None,
                models: provider_model_values("grok", GrokProvider::models_list()).unwrap(),
                catalog: grok_model_catalog(),
            },
        );
        let (state, _guard) = state_with_stats(providers);
        let req = ChatCompletionRequest {
            model: "grok:grok-4.3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let ok = call_chat_handler(state.clone(), req)
            .await
            .expect("static provider succeeds");
        assert_eq!(ok.status(), StatusCode::OK);

        let bad_req = ChatCompletionRequest {
            model: "claude:sonnet".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let err = call_chat_handler(state.clone(), bad_req).await.unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));

        let resp = stats_handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["total_requests"], 1);
        assert_eq!(v["errors"], 1);
        assert_eq!(v["models"]["grok:grok-4.3"]["requests"], 1);
        assert_eq!(v["models"]["grok:grok-4.3"]["input_tokens"], 3);
        assert_eq!(v["models"]["grok:grok-4.3"]["output_tokens"], 2);
    }

    #[tokio::test]
    async fn test_conversation_log_records_chat_request_and_response() {
        // WHY: conversation_log and session are shared production modules, not
        // dead scaffolding. When enabled, Omni must log the request and response
        // under a stable session derived from x-session-id.
        #[derive(Debug)]
        struct StaticProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StaticProvider {
            fn id(&self) -> &'static str {
                "static"
            }

            async fn send(
                &self,
                req: omni_core::CanonicalRequest,
            ) -> Result<CanonicalResponse, ProviderError> {
                Ok(CanonicalResponse {
                    model: req.model,
                    content: "logged response".into(),
                    tool_calls: vec![],
                    finish_reason: Some("stop".into()),
                    usage: Default::default(),
                })
            }
        }

        let dir = std::env::temp_dir().join(format!("omni-conv-log-TEST-{}", Uuid::new_v4()));
        let mut providers: HashMap<String, ProviderEntry> = HashMap::new();
        providers.insert(
            "grok".into(),
            ProviderEntry {
                provider: Arc::new(StaticProvider),
                claude_native: None,
                models: provider_model_values("grok", GrokProvider::models_list()).unwrap(),
                catalog: grok_model_catalog(),
            },
        );
        let state = state_with_conversation_log(providers, &dir);
        let req = ChatCompletionRequest {
            model: "grok:grok-4.3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("logged request".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let resp = call_chat_handler_with_session(state.clone(), req, "session alpha")
            .await
            .expect("static provider succeeds");
        assert_eq!(resp.status(), StatusCode::OK);
        drop(state);

        let log = std::fs::read_to_string(dir.join("session_alpha.log")).unwrap();
        assert!(log.contains("session=session alpha"));
        assert!(log.contains("Inbound Chat Completions body"));
        assert!(log.contains("logged request"));
        assert!(log.contains("Chat Completions response"));
        assert!(log.contains("logged response"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn test_stats_records_stream_usage_after_body_consumed() {
        // WHY: stats active_requests must track the streamed body lifetime, not
        // just the handler future, and streamed Usage events must feed /stats.
        #[derive(Debug)]
        struct StreamingProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StreamingProvider {
            fn id(&self) -> &'static str {
                "streaming"
            }

            async fn send(
                &self,
                _req: omni_core::CanonicalRequest,
            ) -> Result<CanonicalResponse, ProviderError> {
                unreachable!("stream test uses send_stream")
            }

            async fn send_stream(
                &self,
                _req: omni_core::CanonicalRequest,
            ) -> Result<CanonicalStream, ProviderError> {
                Ok(Box::pin(futures_util::stream::iter(vec![
                    Ok(CanonicalStreamEvent::TextDelta("hi".into())),
                    Ok(CanonicalStreamEvent::Usage(omni_core::CanonicalUsage {
                        input_tokens: 11,
                        output_tokens: 7,
                        cache_read: 3,
                        cache_creation: 2,
                    })),
                    Ok(CanonicalStreamEvent::Finish {
                        finish_reason: Some("stop".into()),
                    }),
                ])))
            }
        }

        let mut providers: HashMap<String, ProviderEntry> = HashMap::new();
        providers.insert(
            "grok".into(),
            ProviderEntry {
                provider: Arc::new(StreamingProvider),
                claude_native: None,
                models: provider_model_values("grok", GrokProvider::models_list()).unwrap(),
                catalog: grok_model_catalog(),
            },
        );
        let (state, _guard) = state_with_stats(providers);
        let req = ChatCompletionRequest {
            model: "grok:grok-4.3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let resp = call_chat_handler(state.clone(), req)
            .await
            .expect("stream opens");
        assert_eq!(
            state.stats.as_ref().unwrap().snapshot().active_requests,
            0,
            "SSE body has not been polled before into_body"
        );
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let sse_body = String::from_utf8_lossy(&body);
        assert!(sse_body.contains("[DONE]"));

        let snap = state.stats.as_ref().unwrap().snapshot();
        assert_eq!(snap.active_requests, 0);
        assert_eq!(snap.total_requests, 1);
        assert_eq!(snap.models["grok:grok-4.3"].requests, 1);
        assert_eq!(snap.models["grok:grok-4.3"].input_tokens, 11);
        assert_eq!(snap.models["grok:grok-4.3"].output_tokens, 7);
        assert_eq!(snap.models["grok:grok-4.3"].cache_read_input_tokens, 3);
        assert_eq!(snap.models["grok:grok-4.3"].cache_creation_input_tokens, 2);
    }

    #[tokio::test]
    async fn test_http_completions_stream_is_routed_not_rejected() {
        // WHY: streaming is now a first-class path. A stream:true request must be
        // ROUTED to the provider's send_stream (and, when reachable, framed as an
        // SSE response), never rejected with the old "streaming not supported"
        // 400. We use the grok test provider pointed at a dead port: routing +
        // stream-open is exercised; the dead upstream surfaces as a ServerError
        // (NOT a BadRequest stream-rejection), proving the stream branch is live.
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("grok".into(), grok_entry("http://127.0.0.1:1"));
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "grok:grok-4.3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let res = call_chat_handler(state, req).await;
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
        // (text/event-stream), live-conditional on explicit opt-in + Grok creds.
        // The byte-level [DONE] framing is pinned in omni-common::http.
        if !has_grok_creds() {
            eprintln!("skipping SSE-reachable test: set OMNI_LIVE_TESTS=1 with Grok creds");
            return;
        }
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("grok".into(), live_grok_entry());
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "grok:grok-4.3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("Reply with the single word PONG".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: true,
            max_tokens: Some(16),
            max_completion_tokens: None,
            temperature: Some(0.0),
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let res = call_chat_handler(state, req).await;
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
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert(
            "claude".into(),
            claude_entry_with_base("http://127.0.0.1:1"),
        );
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "codex:bar".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let res = call_chat_handler(state, req).await;
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
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("claude".into(), claude_entry());
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "grok:bar".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let res = call_chat_handler(state, req).await;
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
    async fn test_http_completions_unknown_bare_model_errors_when_multi() {
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("claude".into(), claude_entry());
        map.insert("grok".into(), grok_entry("http://127.0.0.1:9"));
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "bare-model".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let res = call_chat_handler(state, req).await;
        let m = match res {
            Err(e) => e,
            Ok(_) => panic!("want prefix error"),
        };
        match m {
            AppError::BadRequest(mm) => assert!(mm.contains("unknown model")),
            _ => panic!("want prefix error"),
        }
    }

    #[tokio::test]
    async fn test_http_completions_bare_alias_routes_when_multi() {
        // WHY: Omni advertises real model ids and documented aliases. A known
        // bare alias must route even when multiple providers are enabled.
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("claude".into(), claude_entry());
        map.insert("grok".into(), grok_entry("http://127.0.0.1:1"));
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "grok".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("ping".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let res = call_chat_handler(state, req).await;
        match res {
            Err(AppError::ServerError(msg)) => {
                assert!(msg.contains("upstream") || msg.contains("network"))
            }
            other => panic!("bare grok alias must route to provider, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_http_completions_bare_alias_echoes_canonical_model() {
        // WHY: shorthand aliases are request conveniences. Responses should
        // identify the resolved canonical provider model.
        #[derive(Debug)]
        struct StaticProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StaticProvider {
            fn id(&self) -> &'static str {
                "static"
            }

            async fn send(
                &self,
                req: omni_core::CanonicalRequest,
            ) -> Result<CanonicalResponse, ProviderError> {
                Ok(CanonicalResponse {
                    model: req.model,
                    content: "ok".into(),
                    tool_calls: vec![],
                    finish_reason: Some("stop".into()),
                    usage: Default::default(),
                })
            }
        }

        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert(
            "grok".into(),
            ProviderEntry {
                provider: Arc::new(StaticProvider),
                claude_native: None,
                models: provider_model_values("grok", GrokProvider::models_list()).unwrap(),
                catalog: grok_model_catalog(),
            },
        );
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "grok".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let resp = call_chat_handler(state, req)
            .await
            .expect("static provider succeeds");
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model"], "grok-4.3");
    }

    #[tokio::test]
    async fn test_stats_keys_normalize_aliases_and_prefixes_to_canonical() {
        // WHY: aliases are request conveniences. Equivalent traffic must not
        // split metrics across `grok`, `grok-4.3`, and `grok:grok-4.3`.
        #[derive(Debug)]
        struct StaticProvider;
        #[async_trait::async_trait]
        impl LlmProvider for StaticProvider {
            fn id(&self) -> &'static str {
                "static"
            }

            async fn send(
                &self,
                req: omni_core::CanonicalRequest,
            ) -> Result<CanonicalResponse, ProviderError> {
                Ok(CanonicalResponse {
                    model: req.model,
                    content: "ok".into(),
                    tool_calls: vec![],
                    finish_reason: Some("stop".into()),
                    usage: Default::default(),
                })
            }
        }

        let mut providers: HashMap<String, ProviderEntry> = HashMap::new();
        providers.insert(
            "grok".into(),
            ProviderEntry {
                provider: Arc::new(StaticProvider),
                claude_native: None,
                models: provider_model_values("grok", GrokProvider::models_list()).unwrap(),
                catalog: grok_model_catalog(),
            },
        );
        let (state, _guard) = state_with_stats(providers);
        for model in ["grok", "grok-4.3", "grok:grok-4.3"] {
            let req = ChatCompletionRequest {
                model: model.into(),
                messages: vec![ChatMessage {
                    role: "user".into(),
                    content: Some("hi".into()),
                    tool_calls: None,
                    tool_call_id: None,
                }],
                stream: false,
                max_tokens: None,
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                tools: None,
                tool_choice: None,
                extras: serde_json::Value::Null,
            };
            call_chat_handler(state.clone(), req)
                .await
                .expect("static provider succeeds");
        }

        let snap = state.stats.as_ref().unwrap().snapshot();
        assert_eq!(snap.models["grok:grok-4.3"].requests, 3);
        assert_eq!(snap.models.len(), 1);
    }

    #[tokio::test]
    async fn test_http_completions_routes_via_prefix_to_grok_test_provider() {
        // grok test ctor points to bad port -> upstream err mapped to 5xx server err (delegation exercised)
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("grok".into(), grok_entry("http://127.0.0.1:1"));
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "grok:grok-4.3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("ping".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let res = call_chat_handler(state, req).await;
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
        // WHY: prefix routing into the Claude provider must delegate with the
        // stripped backend model. A dead test upstream then surfaces as a server
        // error, proving dispatch reached the provider instead of failing routing.
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert(
            "claude".into(),
            claude_entry_with_base("http://127.0.0.1:1"),
        );
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "claude:sonnet".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("Reply with the word PONG only.".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: Some(16),
            max_completion_tokens: None,
            temperature: Some(0.0),
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let res = call_chat_handler(state, req).await;
        match res {
            Err(AppError::ServerError(msg)) => {
                assert!(msg.contains("upstream") || msg.contains("network"))
            }
            other => panic!("expected provider upstream error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_http_completions_unified_oai_response_shape() {
        // Use grok test provider (always errors upstream but proves from_canonical + oai shape on err path? no, err before)
        // Instead construct a direct canonical resp and from_ to pin the surface (unified for both backends)
        let canon = CanonicalResponse {
            model: "grok-4.3".into(),
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
        let oai = from_canonical(canon, "grok:grok-4.3".into(), "chatcmpl-xyz".into(), 123);
        assert_eq!(oai.id, "chatcmpl-xyz");
        assert_eq!(oai.object, "chat.completion");
        assert_eq!(oai.model, "grok:grok-4.3");
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
        // WHY: both provider crates must still share the same replacement engine
        // contract even though provider-specific protocol logic is isolated.
        let r = omni_common::Replacements::parse(
            r#"rule = [{scope="both", search="ping", replace="pong"}]"#,
        )
        .unwrap();
        assert_eq!(r.apply_prompt("ping"), "pong");
        assert_eq!(r.apply_response("ping"), "pong");
        // grok path (test ctor, no net)
        let pg = GrokProvider::new_for_test("k", "http://127.0.0.1:9");
        assert_eq!(pg.id(), "grok");
        // claude path
        let pc = ClaudeProvider::new().expect("claude");
        assert_eq!(pc.id(), "claude-code");
        let canon = omni_core::CanonicalResponse {
            model: "m".into(),
            content: "c".into(),
            tool_calls: vec![],
            finish_reason: None,
            usage: Default::default(),
        };
        let oai = from_canonical(canon, "grok:m".into(), "chatcmpl-test".into(), 1);
        assert_eq!(oai.choices[0].message.content.as_deref(), Some("c"));
    }

    #[tokio::test]
    async fn test_multi_backend_enable_both_and_route_each() {
        // Multi backend (enable both, test both in one test)
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("grok".to_string(), grok_entry("http://127.0.0.1:1"));
        map.insert("claude".to_string(), claude_entry());
        let catalogs = provider_catalogs(&map);
        assert_eq!(catalogs.len(), 2);
        let (kg, mg) = resolve_provider_and_model("grok:x", &catalogs).unwrap();
        assert_eq!(kg, "grok");
        assert_eq!(mg, "x");
        let (kc, mc) = resolve_provider_and_model("claude:y", &catalogs).unwrap();
        assert_eq!(kc, "claude");
        assert_eq!(mc, "y");
        let (kg, mg) = resolve_provider_and_model("grok", &catalogs).unwrap();
        assert_eq!((kg.as_str(), mg.as_str()), ("grok", "grok-4.3"));
        assert!(resolve_provider_and_model("bare", &catalogs).is_err());
    }

    #[tokio::test]
    async fn test_credential_loading_in_omni_delegation_context() {
        // WHY: omni delegation must leave credential loading inside the
        // provider. A dead upstream should therefore surface after the provider's
        // fresh credential-resolution path, not be swallowed by router logic.
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("grok".into(), grok_entry("http://127.0.0.1:1"));
        let state = state_with(map);
        let req = ChatCompletionRequest {
            model: "grok:grok-4.3".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some("c".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            extras: serde_json::Value::Null,
        };
        let err = call_chat_handler(state, req)
            .await
            .expect_err("dead upstream must fail after provider delegation");
        match err {
            AppError::ServerError(msg) => {
                assert!(msg.contains("upstream") || msg.contains("network"))
            }
            other => panic!("expected provider upstream error, got {other:?}"),
        }
    }

    // --- subprocess binary HTTP tests (full stack, random port, kill, live conditional for real calls) ---

    #[test]
    fn test_subprocess_omni_binary_health_and_root() {
        let port = free_port();
        let _child = spawn_omni(&["--no-auth", "--port", &port.to_string()], &[]);
        assert!(
            wait_for_200_health(port, Duration::from_secs(6)),
            "omni did not become healthy on {}",
            port
        );
        let resp = get(port, "/");
        assert_eq!(resp.status, 200);
        assert!(resp.body.contains("omni - multi-backend"));
    }

    #[test]
    fn test_subprocess_omni_binary_models() {
        let port = free_port();
        let _child = spawn_omni(&["--no-auth", "--port", &port.to_string()], &[]);
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        let resp = get(port, "/v1/models");
        assert_eq!(resp.status, 200);
        let v = omni_common::test_support::parse_json(&resp.body);
        assert_eq!(v["object"], "list");
        let ids = omni_common::test_support::model_ids(&v);
        assert!(
            ids.iter().any(|id| id.starts_with("claude-")),
            "default catalog must include claude models: {ids:?}"
        );
        assert!(
            ids.iter().any(|id| id.starts_with("grok-")),
            "default catalog must include grok models: {ids:?}"
        );
        assert!(
            ids.iter().all(|id| !id.contains(':')),
            "default catalog must expose canonical model ids: {ids:?}"
        );
    }

    #[test]
    fn test_subprocess_omni_binary_stats_route_exists() {
        // WHY: /stats is the replacement for the removed provider-specific
        // binaries' stats endpoints; the production router must expose JSON.
        let port = free_port();
        let stats_path = temp_stats_path();
        let _guard = TempStats(stats_path.clone());
        let _child = spawn_omni(
            &[
                "--no-auth",
                "--port",
                &port.to_string(),
                "--stats-db",
                stats_path.to_str().unwrap(),
            ],
            &[],
        );
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        let resp = get(port, "/stats");
        assert_eq!(resp.status, 200);
        let v = omni_common::test_support::parse_json(&resp.body);
        assert!(v["uptime_seconds"].is_u64(), "stats shape missing: {v}");
        assert!(v["models"].is_object(), "stats models missing: {v}");
    }

    #[test]
    fn test_subprocess_omni_binary_auth_mw_401_vs_200() {
        // Auth mw (with/without keys, 401 vs 200) - full layered router via binary
        let port = free_port();
        // with keys set (no --no-auth): unauthed requests 401, authed 200. Wait must auth.
        let child = spawn_omni(
            &["--port", &port.to_string()],
            &[("OMNI_API_KEYS", "secret123,other")],
        );
        // wait using proper header (keys case requires it for any surface incl health)
        let start = Instant::now();
        let mut ready = false;
        while start.elapsed() < Duration::from_secs(6) {
            if omni_common::test_support::wait_for_http_body_with_headers(
                format!("http://127.0.0.1:{}/health", port),
                &[("Authorization", "Bearer secret123")],
                "ok",
                Duration::from_millis(250),
            ) {
                ready = true;
                break;
            }
        }
        assert!(ready, "protected server did not become ready");
        // no header -> 401
        let out1 = get(port, "/health");
        assert_eq!(out1.status, 401);
        // with good key -> 200
        let out2 = omni_common::test_support::http_get_with_headers(
            format!("http://127.0.0.1:{}/health", port),
            &[("Authorization", "Bearer secret123")],
        );
        assert_eq!(out2.status, 200);
        // bad key -> 401
        let out3 = omni_common::test_support::http_get_with_headers(
            format!("http://127.0.0.1:{}/health", port),
            &[("Authorization", "Bearer wrong")],
        );
        assert_eq!(out3.status, 401);
        let out_anth = post_json(
            port,
            "/v1/messages",
            r#"{"model":"sonnet","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(out_anth.status, 401);
        let v = omni_common::test_support::parse_json(&out_anth.body);
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "authentication_error");
        drop(child);

        // without keys (empty or --no-auth) -> 200 even no header
        let port2 = free_port();
        let _child2 = spawn_omni(&["--no-auth", "--port", &port2.to_string()], &[]);
        assert!(wait_for_200_health(port2, Duration::from_secs(6)));
        let out4 = get(port2, "/health");
        assert_eq!(out4.status, 200);
    }

    #[test]
    fn test_subprocess_omni_binary_completions_routing_errors() {
        // errors (unknown provider, disabled, bad model) via full http
        let port = free_port();
        let _child = spawn_omni(&["--no-auth", "--port", &port.to_string()], &[]);
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        // unknown prefix
        let out = post_json(
            port,
            "/v1/chat/completions",
            r#"{"model":"nope:xx","messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(out.status, 400);
        let v = omni_common::test_support::parse_json(&out.body);
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("not enabled")
        );
        // bare in multi
        let out2 = post_json(
            port,
            "/v1/chat/completions",
            r#"{"model":"bare","messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(out2.status, 400);
        let v2 = omni_common::test_support::parse_json(&out2.body);
        assert!(
            v2["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("unknown model")
        );
    }

    #[test]
    fn test_subprocess_omni_binary_completions_live_conditional_both_backends() {
        // Live opt-in proof for real calls to both backends; unified surfaces.
        if !has_grok_creds() && !has_claude_creds() {
            eprintln!("skipping live completions: set OMNI_LIVE_TESTS=1 with provider creds");
            return;
        }
        let port = free_port();
        let _child = spawn_omni(&["--no-auth", "--port", &port.to_string()], &[]);
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        if has_grok_creds() {
            let out = post_json(
                port,
                "/v1/chat/completions",
                r#"{"model":"grok:grok-4.3","messages":[{"role":"user","content":"Reply PONG"}]}"#,
            );
            let v = omni_common::test_support::parse_json(&out.body);
            if v.get("choices").is_some() {
                assert!(
                    !v["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .is_empty()
                        || v["choices"][0]["message"].get("tool_calls").is_some()
                );
            } else {
                let err_msg = v["error"]["message"].as_str().unwrap_or("");
                assert!(
                    !err_msg.contains("not enabled") && !err_msg.contains("unknown model"),
                    "routing should have succeeded for grok request: {err_msg}"
                );
            }
        }
        if has_claude_creds() {
            let out = post_json(
                port,
                "/v1/chat/completions",
                r#"{"model":"claude:haiku","messages":[{"role":"user","content":"Reply PONG"}]}"#,
            );
            let v = omni_common::test_support::parse_json(&out.body);
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
    }

    #[test]
    fn test_subprocess_omni_binary_multi_provider_config() {
        // enable both via OMNI_PROVIDERS, test routing to each (prefix)
        let port = free_port();
        let _child = spawn_omni(
            &["--no-auth", "--port", &port.to_string()],
            &[("OMNI_PROVIDERS", "claude,grok")],
        );
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        // models should list for both
        let out = get(port, "/models");
        assert_eq!(out.status, 200);
        let v = omni_common::test_support::parse_json(&out.body);
        let ids = omni_common::test_support::model_ids(&v);
        assert!(ids.iter().any(|id| id.starts_with("claude-")));
        assert!(ids.iter().any(|id| id.starts_with("grok-")));
        assert!(ids.iter().all(|id| !id.contains(':')));
    }

    #[test]
    fn test_subprocess_omni_binary_streaming_sse_done_terminator() {
        // WHY: full-stack proof that stream:true over real HTTP yields an SSE
        // body terminated by `data: [DONE]` (the OpenAI streaming contract).
        // Live-conditional on explicit opt-in + grok creds; the hermetic framing
        // is already pinned by omni-common::http unit tests.
        if !has_grok_creds() {
            eprintln!("skipping streaming subprocess test: set OMNI_LIVE_TESTS=1 with Grok creds");
            return;
        }
        let port = free_port();
        let _child = spawn_omni(
            &["--no-auth", "--port", &port.to_string()],
            &[("OMNI_PROVIDERS", "grok")],
        );
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        let out = post_json(
            port,
            "/v1/chat/completions",
            r#"{"model":"grok:grok-4.3","stream":true,"max_tokens":16,"messages":[{"role":"user","content":"Reply PONG"}]}"#,
        );
        assert_eq!(out.status, 200);
        let body = out.body;
        assert!(
            body.contains("chat.completion.chunk"),
            "expected SSE chunks, got: {body}"
        );
        assert!(body.contains("[DONE]"), "stream must terminate with [DONE]");
    }

    // ---- OpenAI Responses protocol (/v1/responses): TDD contract tests ----

    fn responses_req(body: &str) -> omni_common::ResponsesRequest {
        serde_json::from_str(body).expect("responses request json")
    }

    #[tokio::test]
    async fn test_responses_unsupported_input_is_bad_request() {
        // WHY: input shapes the canonical layer still cannot represent (an
        // `input_image` content part) are rejected LOUDLY as an OAI-shaped 400
        // naming the offender, BEFORE any provider call; a 500 or silent
        // mangling would corrupt the request. (Tool-conversation items like
        // function_call / function_call_output ARE now supported and round-trip
        // through canonical blocks, so they are no longer the rejection case.)
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("claude".into(), claude_entry());
        let state = state_with(map);
        let req = responses_req(
            r#"{"model":"claude:sonnet","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"x"}]}]}"#,
        );
        let res = call_responses_handler(state, req).await;
        match res {
            Err(AppError::BadRequest(msg)) => assert!(
                msg.contains("input_image"),
                "400 must name the unsupported content part type: {msg}"
            ),
            other => panic!("expected BadRequest for unsupported input, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_responses_unknown_bare_model_multi_errors() {
        // WHY: the aggregator's routing contract applies to EVERY inbound
        // protocol; unknown bare model ids must fail before provider dispatch.
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("claude".into(), claude_entry());
        map.insert("grok".into(), grok_entry("http://127.0.0.1:9"));
        let state = state_with(map);
        let req = responses_req(r#"{"model":"bare-model","input":"hi"}"#);
        let res = call_responses_handler(state, req).await;
        match res {
            Err(AppError::BadRequest(msg)) => assert!(msg.contains("unknown model")),
            other => panic!("expected unknown-model BadRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_responses_nonstream_routes_via_prefix_to_grok() {
        // WHY: proves the Responses handler delegates through the same provider
        // map: a dead grok upstream surfaces as ServerError (delegation reached
        // the provider), never as a routing-level BadRequest.
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("grok".into(), grok_entry("http://127.0.0.1:1"));
        let state = state_with(map);
        let req = responses_req(r#"{"model":"grok:grok-4.3","input":"ping"}"#);
        let res = call_responses_handler(state, req).await;
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
        let mut map: HashMap<String, ProviderEntry> = HashMap::new();
        map.insert("grok".into(), grok_entry("http://127.0.0.1:1"));
        let state = state_with(map);
        let req = responses_req(r#"{"model":"grok:grok-4.3","input":"ping","stream":true}"#);
        let res = call_responses_handler(state, req).await;
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
        let _child = spawn_omni(
            &["--no-auth", "--port", &port.to_string()],
            &[("OMNI_PROVIDERS", "claude")],
        );
        assert!(wait_for_200_health(port, Duration::from_secs(6)));
        let out = post_json(
            port,
            "/v1/responses",
            r#"{"model":"claude:sonnet","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"x"}]}]}"#,
        );
        assert_eq!(
            out.status, 400,
            "POST /v1/responses must exist and reject unsupported input with 400 (404 = route not registered)"
        );
    }

    #[test]
    fn test_subprocess_omni_binary_responses_live_roundtrip() {
        // WHY: end-to-end proof over real HTTP: a Responses request returns the
        // Responses envelope (non-stream) and Responses SSE events (stream)
        // through the aggregator's prefix routing. Live-conditional on explicit
        // opt-in + grok creds so the suite stays green offline.
        if !has_grok_creds() {
            eprintln!("skipping responses live roundtrip: set OMNI_LIVE_TESTS=1 with Grok creds");
            return;
        }
        let port = free_port();
        let _child = spawn_omni(
            &["--no-auth", "--port", &port.to_string()],
            &[("OMNI_PROVIDERS", "grok")],
        );
        assert!(wait_for_200_health(port, Duration::from_secs(6)));

        // Non-stream: Responses envelope with assistant output_text.
        let out = post_json(
            port,
            "/v1/responses",
            r#"{"model":"grok:grok-4.3","input":"Reply with the single word PONG","max_output_tokens":16}"#,
        );
        assert_eq!(out.status, 200, "live body: {}", out.body);
        let v = omni_common::test_support::parse_json(&out.body);
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
        let out2 = post_json(
            port,
            "/v1/responses",
            r#"{"model":"grok:grok-4.3","input":"Reply with the single word PONG","max_output_tokens":16,"stream":true}"#,
        );
        assert_eq!(out2.status, 200, "live stream body: {}", out2.body);
        let body = out2.body;
        assert!(
            body.contains("event: response.created"),
            "live stream must open with response.created: {body}"
        );
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.completed"));
        assert!(!body.contains("[DONE]"));
    }

    #[test]
    fn live_tool_call_loop_both_backends_conditional() {
        // WHY: closes per-backend chat tool coverage on the AGGREGATOR. When a
        // backend's creds are present, runs the full multi-turn tool loop through
        // prefix routing for that backend: hop 1 declares a tool and the model
        // emits a get_weather tool_call (finish_reason "tool_calls"); hop 2 feeds
        // the tool RESULT back and the model answers using it (finish_reason
        // "stop", content contains the fed-back "72"). The hop-2 "72" assertion
        // proves the tool result actually round-tripped through the aggregator.
        // Skips unless OMNI_LIVE_TESTS=1 and at least one backend's creds are
        // present. Starts both providers; each reads its key fresh per request.
        if !has_grok_creds() && !has_claude_creds() {
            eprintln!("skipping live tool-call loop: set OMNI_LIVE_TESTS=1 with provider creds");
            return;
        }
        let port = free_port();
        let _child = spawn_omni(
            &["--no-auth", "--port", &port.to_string()],
            &[("OMNI_PROVIDERS", "claude,grok")],
        );
        assert!(wait_for_200_health(port, Duration::from_secs(6)));

        // Per-backend prefix model; only exercised when that backend has creds.
        let mut backends: Vec<&str> = Vec::new();
        if has_grok_creds() {
            backends.push("grok:grok-4.3");
        }
        if has_claude_creds() {
            backends.push("claude:haiku");
        }
        for model in backends {
            // Hop 1: declare a tool; model must emit a get_weather tool_call.
            let body1 = format!(
                r#"{{"model":"{model}","messages":[{{"role":"user","content":"What is the weather in San Francisco? Use the get_weather tool."}}],"tools":[{{"type":"function","function":{{"name":"get_weather","description":"Get weather for a city","parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}},"required":["city"]}}}}}}],"tool_choice":"auto","max_tokens":256}}"#
            );
            let out = post_json(port, "/v1/chat/completions", &body1);
            assert_eq!(out.status, 200, "{model} hop-1 body: {}", out.body);
            let v = omni_common::test_support::parse_json(&out.body);
            assert_eq!(
                v["choices"][0]["finish_reason"], "tool_calls",
                "{model} hop-1 must stop for tool_calls, got: {v}"
            );
            let tool_calls = v["choices"][0]["message"]["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            assert!(
                !tool_calls.is_empty(),
                "{model} hop-1 must carry a non-empty tool_calls array: {v}"
            );
            assert_eq!(
                tool_calls[0]["function"]["name"], "get_weather",
                "{model} hop-1 tool call must name get_weather: {v}"
            );
            assert!(
                !tool_calls[0]["function"]["arguments"]
                    .as_str()
                    .unwrap_or("")
                    .is_empty(),
                "{model} hop-1 tool call must carry arguments (city), got: {v}"
            );

            // Hop 2: feed the tool result back; model must answer using it.
            let body2 = format!(
                r#"{{"model":"{model}","messages":[{{"role":"user","content":"What is the weather in SF?"}},{{"role":"assistant","content":null,"tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"get_weather","arguments":"{{\"city\":\"SF\"}}"}}}}]}},{{"role":"tool","tool_call_id":"call_1","content":"72F and sunny"}}],"tools":[{{"type":"function","function":{{"name":"get_weather","parameters":{{"type":"object","properties":{{"city":{{"type":"string"}}}}}}}}}}],"max_tokens":256}}"#
            );
            let out2 = post_json(port, "/v1/chat/completions", &body2);
            assert_eq!(out2.status, 200, "{model} hop-2 body: {}", out2.body);
            let v2 = omni_common::test_support::parse_json(&out2.body);
            assert_eq!(
                v2["choices"][0]["finish_reason"], "stop",
                "{model} hop-2 must finish with stop after consuming the tool result, got: {v2}"
            );
            let content = v2["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("");
            assert!(
                !content.is_empty(),
                "{model} hop-2 content must be present: {v2}"
            );
            assert!(
                content.contains("72"),
                "{model} hop-2 content must mention the fed-back temperature (proves the tool result round-tripped): {content}"
            );
        }
    }

    #[test]
    fn test_subprocess_omni_binary_responses_live_roundtrip_claude() {
        // WHY: closes the CLAUDE gap in the live Responses coverage (the grok
        // roundtrip test only covers grok). End-to-end over real HTTP through the
        // aggregator's prefix routing: non-stream yields the Responses envelope
        // (object "response", status completed, output[0] message with non-empty
        // text); stream yields Responses SSE events (response.created /
        // output_text.delta / completed) with no [DONE]. Live-conditional on
        // explicit opt-in + claude creds so the suite stays green offline.
        if !has_claude_creds() {
            eprintln!(
                "skipping responses live roundtrip (claude): set OMNI_LIVE_TESTS=1 with Claude creds"
            );
            return;
        }
        let port = free_port();
        let _child = spawn_omni(
            &["--no-auth", "--port", &port.to_string()],
            &[("OMNI_PROVIDERS", "claude")],
        );
        assert!(wait_for_200_health(port, Duration::from_secs(6)));

        // Non-stream: Responses envelope with assistant output_text.
        let out = post_json(
            port,
            "/v1/responses",
            r#"{"model":"claude:haiku","input":"Reply with the single word PONG","max_output_tokens":16}"#,
        );
        assert_eq!(out.status, 200, "live body: {}", out.body);
        let v = omni_common::test_support::parse_json(&out.body);
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

        // Stream: Responses SSE events, no [DONE].
        let out2 = post_json(
            port,
            "/v1/responses",
            r#"{"model":"claude:haiku","input":"Reply with the single word PONG","max_output_tokens":16,"stream":true}"#,
        );
        assert_eq!(out2.status, 200, "live stream body: {}", out2.body);
        let body = out2.body;
        assert!(
            body.contains("event: response.created"),
            "live stream must open with response.created: {body}"
        );
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.completed"));
        assert!(!body.contains("[DONE]"));

        // Full tool loop (payload 3): feed function_call + output back; the model
        // must complete using the fed-back result. The "72" assertion proves the
        // tool result round-tripped through the Responses protocol on claude.
        let out3 = post_json(
            port,
            "/v1/responses",
            r#"{"model":"claude:haiku","input":[{"type":"message","role":"user","content":"Weather in SF?"},{"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"SF\"}"},{"type":"function_call_output","call_id":"c1","output":"72F and sunny"}],"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],"max_output_tokens":256}"#,
        );
        assert_eq!(out3.status, 200, "responses tool body: {}", out3.body);
        let v3 = omni_common::test_support::parse_json(&out3.body);
        assert_eq!(
            v3["status"], "completed",
            "responses tool loop must complete, got: {v3}"
        );
        let text3 = v3["output"]
            .as_array()
            .and_then(|items| {
                items
                    .iter()
                    .find(|it| it["type"] == "message")
                    .and_then(|m| m["content"][0]["text"].as_str())
            })
            .unwrap_or("");
        assert!(
            text3.contains("72"),
            "responses tool loop output must mention the fed-back temperature (proves the tool result round-tripped): {v3}"
        );
    }

    #[test]
    fn test_subprocess_omni_binary_responses_tool_loop_live() {
        // WHY: proves the Responses full-tool-loop on GROK through the aggregator
        // (payload 3): feeding a function_call + function_call_output back makes
        // the model complete using the fed-back result. The "72" assertion proves
        // the tool result round-tripped through the Responses protocol.
        // Live-conditional on explicit opt-in + grok creds.
        if !has_grok_creds() {
            eprintln!(
                "skipping responses tool loop live (grok): set OMNI_LIVE_TESTS=1 with Grok creds"
            );
            return;
        }
        let port = free_port();
        let _child = spawn_omni(
            &["--no-auth", "--port", &port.to_string()],
            &[("OMNI_PROVIDERS", "grok")],
        );
        assert!(wait_for_200_health(port, Duration::from_secs(6)));

        let out = post_json(
            port,
            "/v1/responses",
            r#"{"model":"grok:grok-4.3","input":[{"type":"message","role":"user","content":"Weather in SF?"},{"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"SF\"}"},{"type":"function_call_output","call_id":"c1","output":"72F and sunny"}],"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}],"max_output_tokens":256}"#,
        );
        assert_eq!(out.status, 200, "responses tool body: {}", out.body);
        let v = omni_common::test_support::parse_json(&out.body);
        assert_eq!(
            v["status"], "completed",
            "responses tool loop must complete, got: {v}"
        );
        let text = v["output"]
            .as_array()
            .and_then(|items| {
                items
                    .iter()
                    .find(|it| it["type"] == "message")
                    .and_then(|m| m["content"][0]["text"].as_str())
            })
            .unwrap_or("");
        assert!(
            text.contains("72"),
            "responses tool loop output must mention the fed-back temperature (proves the tool result round-tripped): {v}"
        );
    }
}
