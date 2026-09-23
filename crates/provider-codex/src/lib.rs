//! provider-codex
//!
//! Codex configuration backed provider. This crate intentionally reads Codex's
//! own `CODEX_HOME` / `~/.codex` config and auth state instead of inventing a
//! parallel Omni-only setup.
//!
//! Transport is inferred from that state (no operator mode flag): ChatGPT OAuth
//! on the default OpenAI-shaped config uses the CLI-parity ChatGPT WebSocket
//! path; API keys and custom gateways use the configured REST Responses path.
//!
//! ChatGPT OAuth tokens in `auth.json` are refreshed in-place by default
//! (see [`oauth_refresh`]); control with global/per-provider env or CLI
//! (`OMNI_OAUTH_REFRESH`, `OMNI_CODEX_OAUTH_REFRESH`, `--no-oauth-refresh-codex`,
//! etc.). Static API keys are never refreshed. On the default CLI-file path,
//! upstream 401 force-refreshes once and replays the inference request once
//! (issue #31). Custom/override endpoints do not 401-replay via those files.

pub mod bootstrap;
mod oauth_refresh;

pub use bootstrap::{
    PROVIDER_ID, bootstrap, detected, detection_source, extras_allowed, from_provider,
};

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use omni_common::responses_upstream::{
    self, ErrorRedactor, ResponsesSseBuffer, ResponsesSseEvent, ResponsesStreamParser,
};
use omni_common::{
    UPSTREAM_CONNECT_TIMEOUT, UPSTREAM_ERROR_BODY_PREFIX, UPSTREAM_ERROR_BODY_READ_TIMEOUT,
    UPSTREAM_REQUEST_TIMEOUT, env_nonempty, headers_from_env, is_sse_content_type,
    map_upstream_headers_wait, timeout_upstream_headers,
};
use omni_core::{
    CanonicalBlock, CanonicalCacheMark, CanonicalCacheMode, CanonicalCacheRetention,
    CanonicalCacheTtl, CanonicalContent, CanonicalMessage, CanonicalReasoning, CanonicalRequest,
    CanonicalResponse, CanonicalStream, CanonicalStreamEvent, CanonicalToolCall,
    CanonicalToolChoice, CatalogModel, LlmProvider, ProviderError, clamp_duration, clamp_retention,
};
use reqwest::header::HeaderMap;
use reqwest::{Client, Url, header};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http;
use tokio_tungstenite::tungstenite::protocol::Message;
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const CONSERVATIVE_CHATGPT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const CONSERVATIVE_OPENAI_BETA: &str = "responses_websockets=2026-02-06";
const CONSERVATIVE_ORIGINATOR: &str = "codex_exec";
const CONSERVATIVE_BETA_FEATURES: &str = "remote_compaction_v2";
const CONSERVATIVE_CLIENT_REQUEST_ID: &str = "00000000-0000-4000-8000-000000000000";
const CONSERVATIVE_WINDOW_ID: &str = "00000000-0000-4000-8000-000000000000:0";
const DEFAULT_CODEX_MODEL: &str = "gpt-6-astra";
const DEFAULT_AUTH_COMMAND_TIMEOUT_MS: u64 = 5_000;

// Codex catalog from `codex debug models --bundled` on CLI 0.156.1 (2026-09-23).
// Visibility=list slugs only, in dump order: gpt-6-astra (default, priority 1),
// gpt-6-sol, gpt-6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gpt-5.5.
// The dump has no alias field, so short names stay on the previous targets:
// `gpt` / `astra` map to astra; `sol` maps to gpt-5.6-sol; `luna` / `mini` /
// `gpt-mini` map to gpt-5.6-luna. Hidden slugs stay out of the caller catalog.
// If that listing cannot be read, rebaseline must fail; do not keep a previous
// pin's catalog. Custom Responses base_url does not change this source.
/// Pinned Codex CLI version (UA + header fingerprint). Single live pin.
pub const CODEX_VERSION: &str = "0.156.1";

/// Model catalog for the active pin.
const CODEX_CATALOG: &[CatalogModel] = &[
    CatalogModel::new("gpt-6-astra", &["gpt", "astra"]),
    CatalogModel::new("gpt-6-sol", &[]),
    CatalogModel::new("gpt-6-luna", &[]),
    CatalogModel::new("gpt-5.6-sol", &["sol"]),
    CatalogModel::new("gpt-5.6-terra", &["terra"]),
    CatalogModel::new("gpt-5.6-luna", &["luna", "mini", "gpt-mini"]),
    CatalogModel::new("gpt-5.5", &[]),
];

#[derive(Debug, Clone, Serialize)]
pub struct CodexModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Clone)]
pub struct CodexProvider {
    client: Client,
    /// Pinned Codex CLI version string (for UA/headers).
    version: &'static str,
}

impl CodexProvider {
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent(format!("omni/{} provider-codex", env!("CARGO_PKG_VERSION")))
            .connect_timeout(UPSTREAM_CONNECT_TIMEOUT)
            .timeout(UPSTREAM_REQUEST_TIMEOUT)
            .build()
            .map_err(|e| ProviderError::Other(anyhow::anyhow!("http client: {e}")))?;
        Ok(Self {
            client,
            version: CODEX_VERSION,
        })
    }

    /// The model catalog for the active pin.
    ///
    /// Path selection (ChatGPT WebSocket vs REST) is inferred from
    /// `CODEX_HOME` config/auth, not from a catalog mode flag.
    fn active_catalog(&self) -> &'static [CatalogModel] {
        CODEX_CATALOG
    }

    /// Shipped Codex CLI pin version string.
    pub fn pinned_version() -> &'static str {
        CODEX_VERSION
    }

    /// The model id used for an actual request. This stays config-driven (the
    /// installed Codex picks its model from `~/.codex/config.toml`); the catalog
    /// above is only what `/v1/models` advertises.
    pub fn current_model(&self) -> Result<String, ProviderError> {
        Ok(CodexRequestConfig::load()?.model)
    }

    pub fn detected() -> bool {
        CodexRequestConfig::detected()
    }

    /// Operator-facing one-liner: resolved home, auth winner, transport, model.
    ///
    /// Discovery only (no secrets). Used at omni startup so logs name the real
    /// `CODEX_HOME` / files rather than "CODEX_HOME or ~/.codex".
    ///
    /// Returns `Err` when config/home cannot be loaded (startup should fail hard).
    pub fn startup_source_summary() -> Result<String, ProviderError> {
        let config = CodexRequestConfig::load()?;
        let home = display_path(&config.home);
        let config_path = config.home.join("config.toml");
        let auth_path = config.home.join("auth.json");
        let config_state = if config_path.is_file() {
            "present"
        } else {
            "missing"
        };
        let auth = describe_codex_auth_source(&config, &auth_path);
        let path = if config.should_use_chatgpt_ws() {
            "chatgpt-ws"
        } else if omni_codex_override_present() {
            "omni-override-rest"
        } else {
            "rest"
        };
        Ok(format!(
            "home={home} config={config_state} {auth} path={path} model={} base_url={}",
            config.model, config.base_url
        ))
    }

    /// Short reason codex is auto-detectable (file/env), for startup logs.
    ///
    /// Prefers concrete home files over ambient env keys when both are present,
    /// so operators see the home that will actually be used.
    pub fn detection_source() -> Option<String> {
        if omni_codex_override_present() {
            return Some("OMNI_CODEX_BASE_URL set".into());
        }
        let home = codex_home();
        // Explicit invalid CODEX_HOME is still a detection signal for diagnostics.
        if std::env::var_os("CODEX_HOME").is_some() && !home.is_dir() {
            return Some(format!("CODEX_HOME={} (missing)", home.display()));
        }
        let config = home.join("config.toml");
        let auth = home.join("auth.json");
        if config.is_file() {
            return Some(format!("home={} config=present", display_path(&home)));
        }
        if auth.is_file() {
            return Some(format!("home={} auth=present", display_path(&home)));
        }
        for key in ["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"] {
            if env_nonempty(key).is_some() {
                return Some(format!("env:{key} set"));
            }
        }
        None
    }

    /// `/v1/models` advertises the active version catalog plus the actually
    /// configured model if it is not already in the catalog.
    ///
    /// Codex's model is config-driven (`~/.codex/config.toml` or an `OMNI_CODEX_*`
    /// override), so whatever the operator configured must appear here even if it
    /// is outside the verified catalog - that is the id requests will actually
    /// use. The configured model is listed first.
    pub fn models_list(&self) -> Vec<CodexModelInfo> {
        let mut ids: Vec<String> = Vec::new();
        if let Ok(configured) = self.current_model() {
            if !configured.is_empty() {
                ids.push(configured);
            }
        }
        for m in self.active_catalog() {
            if !ids.iter().any(|id| id == m.id) {
                ids.push(m.id.to_string());
            }
        }
        ids.into_iter()
            .map(|id| CodexModelInfo {
                id,
                object: "model",
                created: 0,
                owned_by: "codex",
            })
            .collect()
    }

    pub fn model_aliases(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        // The configured model keeps the `gpt` shorthand pointing at whatever is
        // actually in use.
        if let Ok(configured) = self.current_model() {
            if !configured.is_empty() {
                out.push(("gpt".to_string(), configured.clone()));
                out.push((configured.clone(), configured));
            }
        }
        // Catalog aliases (id->id plus each alias->id), skipping dups.
        for m in self.active_catalog() {
            for pair in std::iter::once((m.id.to_string(), m.id.to_string())).chain(
                m.aliases
                    .iter()
                    .map(|alias| (alias.to_string(), m.id.to_string())),
            ) {
                if !out.iter().any(|(a, _)| a == &pair.0) {
                    out.push(pair);
                }
            }
        }
        out
    }

    async fn send_responses_once(
        &self,
        config: &CodexRequestConfig,
        req: &CanonicalRequest,
    ) -> Result<CanonicalResponse, ProviderError> {
        let url = config.responses_url()?;
        let mut headers = config.headers().await?;
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        let error_redactor = CodexErrorRedactor::from_request(&url, &headers);

        let body = codex_responses_body(req, false)?;
        let resp = self
            .client
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                ProviderError::upstream(error_redactor.redact(&format!("codex network error: {e}")))
            })?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| {
            ProviderError::upstream(
                error_redactor.redact(&format!("codex response read error: {e}")),
            )
        })?;
        if !status.is_success() {
            return Err(ProviderError::upstream_status(
                status.as_u16(),
                error_redactor.redact(&format!(
                    "codex HTTP {status}: {}",
                    String::from_utf8_lossy(&bytes)
                )),
            ));
        }

        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| ProviderError::upstream(format!("decode codex response: {e}")))?;
        responses_upstream::response_to_canonical(&value, &req.model, "codex", &error_redactor)
    }

    async fn send_conservative_ws(
        &self,
        req: CanonicalRequest,
        config: CodexRequestConfig,
    ) -> Result<CanonicalResponse, ProviderError> {
        let model = req.model.clone();
        let mut stream = self.send_stream_conservative_ws(req, config).await?;
        collect_canonical_stream(&mut stream, &model).await
    }

    async fn send_stream_conservative_ws(
        &self,
        req: CanonicalRequest,
        config: CodexRequestConfig,
    ) -> Result<CanonicalStream, ProviderError> {
        let mut auth = config.chatgpt_auth().await?;
        let models_url = config.conservative_models_url(self.version)?;
        let mut headers = conservative_codex_headers(self.version, &auth)?;
        let mut redactor = CodexErrorRedactor::for_secrets([auth.access_token.clone()]);

        let mut preflight = map_upstream_headers_wait(
            timeout_upstream_headers(
                self.client
                    .get(models_url.clone())
                    .headers(headers.clone())
                    .send(),
            )
            .await,
            |e| {
                ProviderError::upstream(
                    redactor.redact(&format!("codex conservative models preflight error: {e}")),
                )
            },
        )?;
        if preflight.status().as_u16() == 401
            && config.uses_default_cli_file_auth()
            && config.force_refresh_cli_oauth().await.is_ok()
        {
            auth = config.chatgpt_auth().await?;
            headers = conservative_codex_headers(self.version, &auth)?;
            redactor = CodexErrorRedactor::for_secrets([auth.access_token.clone()]);
            preflight = map_upstream_headers_wait(
                timeout_upstream_headers(
                    self.client.get(models_url).headers(headers.clone()).send(),
                )
                .await,
                |e| {
                    ProviderError::upstream(
                        redactor.redact(&format!("codex conservative models preflight error: {e}")),
                    )
                },
            )?;
        }
        let status = preflight.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let prefix = read_capped_error_body(preflight).await;
            return Err(ProviderError::upstream_status(
                status_code,
                redactor.redact(&format!(
                    "codex conservative models preflight HTTP {status}: {prefix}"
                )),
            ));
        }
        // Success body is unused. Do not unbounded-read it.
        drop(preflight);

        let ws_url = config.conservative_responses_ws_url()?;
        let ws_request = conservative_ws_request(&ws_url, self.version, &auth)?;
        let body = codex_response_create_body(&req)?;
        let open_redactor = redactor.clone();
        let mut ws = match timeout_upstream_headers(async move {
            let (mut ws, _) = connect_async(ws_request)
                .await
                .map_err(|e| map_ws_open_error(e, &open_redactor))?;
            let frame = serde_json::to_string(&body).map_err(|e| {
                ProviderError::upstream(format!("encode codex conservative response.create: {e}"))
            })?;
            ws.send(Message::Text(frame.into())).await.map_err(|e| {
                ProviderError::upstream(
                    open_redactor.redact(&format!("codex conservative websocket send error: {e}")),
                )
            })?;
            Ok::<_, ProviderError>(ws)
        })
        .await
        {
            Ok(Ok(ws)) => ws,
            Ok(Err(err)) => return Err(err),
            Err(_) => {
                return Err(ProviderError::upstream_status(
                    504,
                    "upstream headers timeout",
                ));
            }
        };

        let stream = async_stream::stream! {
            let mut parser = ResponsesStreamParser::new("codex", redactor.clone());
            let mut saw_event = false;
            let mut finished = false;

            while let Some(message) = ws.next().await {
                let message = match message {
                    Ok(message) => message,
                    Err(e) => {
                        yield Err(ProviderError::upstream(redactor.redact(&format!(
                            "codex conservative websocket read error: {e}"
                        ))));
                        return;
                    }
                };
                match message {
                    Message::Text(text) => {
                        saw_event = true;
                        let event = ResponsesSseEvent {
                            event: None,
                            data: text.to_string(),
                        };
                        for parsed in parser.handle_event(event) {
                            match parsed {
                                Ok(CanonicalStreamEvent::Finish { .. }) => {
                                    finished = true;
                                    yield parsed;
                                }
                                Err(_) => {
                                    yield parsed;
                                    return;
                                }
                                Ok(other) => {
                                    yield Ok(other);
                                }
                            }
                        }
                        if finished {
                            let _ = ws.close(None).await;
                            break;
                        }
                    }
                    Message::Binary(_) => {
                        yield Err(ProviderError::upstream(
                            "codex conservative websocket returned a binary frame",
                        ));
                        return;
                    }
                    Message::Close(_) => {
                        break;
                    }
                    Message::Ping(payload) => {
                        if let Err(e) = ws.send(Message::Pong(payload)).await {
                            yield Err(ProviderError::upstream(redactor.redact(&format!(
                                "codex conservative websocket pong error: {e}"
                            ))));
                            return;
                        }
                    }
                    Message::Pong(_) | Message::Frame(_) => {}
                }
            }

            if !finished {
                let message = if saw_event {
                    "codex conservative websocket ended before a terminal response event"
                } else {
                    "codex conservative websocket ended without any response events"
                };
                yield Err(ProviderError::upstream(redactor.redact(message)));
            }
        };

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl LlmProvider for CodexProvider {
    fn id(&self) -> &'static str {
        "codex"
    }

    async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError> {
        let config = CodexRequestConfig::load()?;
        // ChatGPT login + default Codex OpenAI-shaped config → CLI-parity WS path.
        // Custom gateway / API-key-only / override env → configured REST path.
        if config.should_use_chatgpt_ws() {
            return self.send_conservative_ws(req, config).await;
        }
        if config.wire_api != WireApi::Responses {
            return Err(ProviderError::upstream(format!(
                "unsupported Codex wire_api {}; only responses is supported",
                config.wire_api.as_str()
            )));
        }

        match self.send_responses_once(&config, &req).await {
            Err(e) if is_upstream_status(&e, 401) && config.rest_401_replay_allowed() => {
                match config.force_refresh_cli_auth().await {
                    Ok(()) => self.send_responses_once(&config, &req).await,
                    Err(_) => Err(e),
                }
            }
            other => other,
        }
    }

    async fn send_stream(&self, req: CanonicalRequest) -> Result<CanonicalStream, ProviderError> {
        let config = CodexRequestConfig::load()?;
        if config.should_use_chatgpt_ws() {
            return self.send_stream_conservative_ws(req, config).await;
        }
        if config.wire_api != WireApi::Responses {
            return Err(ProviderError::upstream(format!(
                "unsupported Codex wire_api {}; only responses is supported",
                config.wire_api.as_str()
            )));
        }

        let url = config.responses_url()?;
        let mut headers = config.headers().await?;
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            header::ACCEPT,
            header::HeaderValue::from_static("text/event-stream"),
        );
        let body = codex_responses_body(&req, true)?;
        let mut error_redactor = CodexErrorRedactor::from_request(&url, &headers);
        let client = self.client.clone();
        let can_replay = config.rest_401_replay_allowed();
        let replay_config = config.clone();

        let mut http_resp = map_upstream_headers_wait(
            timeout_upstream_headers(
                client
                    .post(url.clone())
                    .headers(headers.clone())
                    .json(&body)
                    .send(),
            )
            .await,
            |e| {
                ProviderError::upstream(error_redactor.redact(&format!("codex network error: {e}")))
            },
        )?;

        if can_replay
            && http_resp.status().as_u16() == 401
            && replay_config.force_refresh_cli_auth().await.is_ok()
        {
            match replay_config.headers().await {
                Ok(mut next_headers) => {
                    next_headers.insert(
                        header::CONTENT_TYPE,
                        header::HeaderValue::from_static("application/json"),
                    );
                    next_headers.insert(
                        header::ACCEPT,
                        header::HeaderValue::from_static("text/event-stream"),
                    );
                    headers = next_headers;
                    error_redactor = CodexErrorRedactor::from_request(&url, &headers);
                    http_resp = map_upstream_headers_wait(
                        timeout_upstream_headers(
                            client
                                .post(url.clone())
                                .headers(headers.clone())
                                .json(&body)
                                .send(),
                        )
                        .await,
                        |e| {
                            ProviderError::upstream(
                                error_redactor.redact(&format!("codex network error: {e}")),
                            )
                        },
                    )?;
                }
                Err(e) => return Err(e),
            }
        }

        let status = http_resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let err_body = error_redactor.redact(&read_capped_error_body(http_resp).await);
            return Err(ProviderError::upstream_status(
                status_code,
                format!("codex HTTP {status}: {err_body}"),
            ));
        }

        let content_type = http_resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok());
        if !is_sse_content_type(content_type) {
            let got = content_type.unwrap_or("missing");
            return Err(ProviderError::upstream(format!(
                "codex stream expected text/event-stream, got {got}"
            )));
        }

        let stream = async_stream::stream! {
            let mut bytes = http_resp.bytes_stream();
            let mut sse = ResponsesSseBuffer::default();
            let mut parser = ResponsesStreamParser::new("codex", error_redactor.clone());
            let mut finished = false;
            let mut saw_event = false;

            while let Some(chunk) = bytes.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        yield Err(ProviderError::upstream(error_redactor.redact(&format!("codex stream read error: {e}"))));
                        return;
                    }
                };
                let events = match sse.push(&chunk) {
                    Ok(events) => events,
                    Err(e) => {
                        yield Err(ProviderError::upstream(e));
                        return;
                    }
                };
                for event in events {
                    saw_event = true;
                    for parsed in parser.handle_event(event) {
                        match parsed {
                            Ok(CanonicalStreamEvent::Finish { .. }) => {
                                finished = true;
                                yield parsed;
                            }
                            Err(_) => {
                                yield parsed;
                                return;
                            }
                            Ok(other) => {
                                yield Ok(other);
                            }
                        }
                    }
                    if finished {
                        break;
                    }
                }
                if finished {
                    break;
                }
            }

            if !finished {
                match sse.finish() {
                    Ok(Some(event)) => {
                        saw_event = true;
                        for parsed in parser.handle_event(event) {
                            match parsed {
                                Ok(CanonicalStreamEvent::Finish { .. }) => {
                                    finished = true;
                                    yield parsed;
                                }
                                Err(_) => {
                                    yield parsed;
                                    return;
                                }
                                Ok(other) => {
                                    yield Ok(other);
                                }
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        yield Err(ProviderError::upstream(e));
                        return;
                    }
                }
            }

            if !finished {
                let message = if saw_event {
                    "codex stream ended before a terminal response event"
                } else {
                    "codex stream ended without any SSE events"
                };
                yield Err(ProviderError::upstream(error_redactor.redact(message)));
            }
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireApi {
    Responses,
    Other,
}

impl WireApi {
    fn parse(value: Option<&str>) -> Self {
        match value.unwrap_or("responses") {
            "responses" => Self::Responses,
            _ => Self::Other,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::Other => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
struct CodexRequestConfig {
    home: PathBuf,
    model: String,
    base_url: String,
    wire_api: WireApi,
    requires_openai_auth: bool,
    env_key: Option<String>,
    experimental_bearer_token: Option<String>,
    auth_command: Option<AuthCommand>,
    http_headers: BTreeMap<String, String>,
    env_http_headers: BTreeMap<String, String>,
    query_params: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct ChatGptAuth {
    access_token: String,
    account_id: String,
}

#[derive(Debug, Clone)]
struct AuthCommand {
    command: String,
    args: Vec<String>,
    timeout_ms: u64,
}

impl CodexRequestConfig {
    fn load() -> Result<Self, ProviderError> {
        let home = codex_home();
        let native = Self::load_from_home(&home);
        if omni_codex_override_present() {
            return Self::load_omni_override(native.ok().as_ref(), home);
        }
        native
    }

    fn detected() -> bool {
        if omni_codex_override_present() {
            return true;
        }
        if ["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"]
            .iter()
            .any(|key| {
                std::env::var(key)
                    .ok()
                    .is_some_and(|value| !value.trim().is_empty())
            })
        {
            return true;
        }
        let home = codex_home();
        home.join("config.toml").is_file() || home.join("auth.json").is_file()
    }

    fn load_omni_override(
        native: Option<&CodexRequestConfig>,
        home: PathBuf,
    ) -> Result<Self, ProviderError> {
        let base_url = env_nonempty("OMNI_CODEX_BASE_URL").ok_or_else(|| {
            ProviderError::Auth("OMNI_CODEX_BASE_URL is required for Codex override".into())
        })?;
        let model = env_nonempty("OMNI_CODEX_MODEL")
            .or_else(|| native.map(|cfg| cfg.model.clone()))
            .unwrap_or_else(|| DEFAULT_CODEX_MODEL.to_string());
        let wire_api = WireApi::parse(env_nonempty("OMNI_CODEX_WIRE_API").as_deref());
        let env_key = if env_nonempty("OMNI_CODEX_AUTH_TOKEN").is_some() {
            Some("OMNI_CODEX_AUTH_TOKEN".to_string())
        } else if env_nonempty("OMNI_CODEX_API_KEY").is_some() {
            Some("OMNI_CODEX_API_KEY".to_string())
        } else {
            None
        };
        let env_http_headers = if std::env::var_os("OMNI_CODEX_CUSTOM_HEADERS").is_some() {
            BTreeMap::from([(
                "__omni_custom_headers__".to_string(),
                "OMNI_CODEX_CUSTOM_HEADERS".to_string(),
            )])
        } else {
            BTreeMap::new()
        };
        Ok(Self {
            home,
            model,
            base_url,
            wire_api,
            requires_openai_auth: false,
            env_key,
            experimental_bearer_token: None,
            auth_command: None,
            http_headers: BTreeMap::new(),
            env_http_headers,
            query_params: BTreeMap::new(),
        })
    }

    fn load_from_home(home: &Path) -> Result<Self, ProviderError> {
        // Explicit CODEX_HOME must resolve to a real directory (invalid paths
        // used to fall through to empty defaults and start a broken provider).
        let codex_home_explicit = std::env::var_os("CODEX_HOME").is_some();
        if codex_home_explicit {
            if !home.exists() {
                return Err(ProviderError::Auth(format!(
                    "CODEX_HOME={} does not exist",
                    home.display()
                )));
            }
            if !home.is_dir() {
                return Err(ProviderError::Auth(format!(
                    "CODEX_HOME={} is not a directory",
                    home.display()
                )));
            }
        }

        let config_path = home.join("config.toml");
        let raw = match std::fs::read_to_string(&config_path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                return Err(ProviderError::Auth(format!(
                    "Codex config.toml at {}: {e}",
                    config_path.display()
                )));
            }
        };
        let value: toml::Value = if raw.trim().is_empty() {
            toml::Value::Table(Default::default())
        } else {
            raw.parse::<toml::Value>().map_err(|e| {
                ProviderError::Auth(format!(
                    "Codex config parse failed at {}: {e}",
                    config_path.display()
                ))
            })?
        };

        let model = toml_str(&value, &["model"])
            .unwrap_or(DEFAULT_CODEX_MODEL)
            .to_string();
        let provider_id = toml_str(&value, &["model_provider"])
            .unwrap_or("openai")
            .to_string();

        let providers = value.get("model_providers").and_then(|v| v.as_table());
        if provider_id == "openai"
            && providers
                .and_then(|table| table.get("openai"))
                .and_then(|v| v.as_table())
                .is_some()
        {
            return Err(ProviderError::Auth(
                "Codex config uses reserved [model_providers.openai]; use openai_base_url or a non-reserved provider id"
                    .into(),
            ));
        }

        let provider = providers
            .and_then(|table| table.get(&provider_id))
            .or_else(|| {
                providers.and_then(|table| {
                    table
                        .iter()
                        .find(|(key, _)| key.eq_ignore_ascii_case(&provider_id))
                        .map(|(_, value)| value)
                })
            })
            .and_then(|v| v.as_table());

        let built_in_openai = provider_id == "openai" && provider.is_none();
        let base_url = if built_in_openai {
            toml_str(&value, &["openai_base_url"])
                .unwrap_or(DEFAULT_OPENAI_BASE_URL)
                .to_string()
        } else {
            provider
                .and_then(|p| p.get("base_url"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ProviderError::Auth(format!(
                        "Codex model_provider {provider_id:?} missing [model_providers.{provider_id}].base_url"
                    ))
                })?
                .to_string()
        };

        let wire_api = WireApi::parse(
            provider
                .and_then(|p| p.get("wire_api"))
                .and_then(|v| v.as_str()),
        );
        let requires_openai_auth = provider
            .and_then(|p| p.get("requires_openai_auth"))
            .and_then(|v| v.as_bool())
            .unwrap_or(built_in_openai);
        let env_key = provider
            .and_then(|p| p.get("env_key"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let experimental_bearer_token = provider
            .and_then(|p| p.get("experimental_bearer_token"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let auth_command = provider
            .and_then(|p| p.get("auth"))
            .and_then(|v| v.as_table())
            .and_then(parse_auth_command);
        let http_headers = provider
            .and_then(|p| p.get("http_headers"))
            .and_then(string_map)
            .unwrap_or_default();
        let env_http_headers = provider
            .and_then(|p| p.get("env_http_headers"))
            .and_then(string_map)
            .unwrap_or_default();
        let query_params = provider
            .and_then(|p| p.get("query_params"))
            .and_then(string_map)
            .unwrap_or_default();

        Ok(Self {
            home: home.to_path_buf(),
            model,
            base_url,
            wire_api,
            requires_openai_auth,
            env_key,
            experimental_bearer_token,
            auth_command,
            http_headers,
            env_http_headers,
            query_params,
        })
    }

    fn responses_url(&self) -> Result<Url, ProviderError> {
        let mut url = Url::parse(&self.base_url)
            .map_err(|e| ProviderError::Auth(format!("invalid Codex base_url: {e}")))?;
        let path = url.path().trim_end_matches('/');
        let next_path = if path.is_empty() || path == "/" || path == "/v1" {
            "/v1/responses".to_string()
        } else if path.ends_with("/v1") {
            format!("{path}/responses")
        } else {
            format!("{path}/v1/responses")
        };
        url.set_path(&next_path);
        {
            let mut pairs = url.query_pairs_mut();
            for (key, value) in &self.query_params {
                pairs.append_pair(key, value);
            }
        }
        Ok(url)
    }

    /// Config shape matches the real Codex CLI's default OpenAI provider (no
    /// custom base_url, headers, or auth hooks). ChatGPT WS still requires OAuth
    /// tokens; see [`Self::should_use_chatgpt_ws`].
    fn conservative_ws_eligible(&self) -> bool {
        !omni_codex_override_present()
            && self.wire_api == WireApi::Responses
            && Url::parse(&self.base_url)
                .ok()
                .is_some_and(|url| is_default_openai_base(url.as_str()))
            && self.http_headers.is_empty()
            && self.env_http_headers.is_empty()
            && self.query_params.is_empty()
            && self.auth_command.is_none()
            && self.experimental_bearer_token.is_none()
            && self.env_key.is_none()
            && self.requires_openai_auth
    }

    /// Whether this request should use the ChatGPT backend WebSocket path.
    ///
    /// Mirrors Claude/Grok: infer from credentials + config, no operator mode
    /// flag. Default OpenAI-shaped Codex config with ChatGPT OAuth tokens → WS;
    /// API-key-only or custom gateway → REST.
    fn should_use_chatgpt_ws(&self) -> bool {
        self.conservative_ws_eligible() && self.has_chatgpt_oauth_tokens()
    }

    /// Default CLI-file OpenAI auth (not OMNI override, not custom env/command).
    /// Only this path may 401-replay via `~/.codex/auth.json`.
    fn uses_default_cli_file_auth(&self) -> bool {
        !omni_codex_override_present()
            && self.auth_command.is_none()
            && self.experimental_bearer_token.is_none()
            && self.env_key.is_none()
            && self.requires_openai_auth
    }

    /// REST 401 replay only when this request's bearer comes from auth.json.
    /// Env API keys win over file OAuth; do not rotate leftover ChatGPT tokens
    /// for a 401 that used `CODEX_API_KEY` / `OPENAI_API_KEY` / `CODEX_ACCESS_TOKEN`.
    fn rest_401_replay_allowed(&self) -> bool {
        self.uses_default_cli_file_auth() && !env_openai_key_present()
    }

    async fn force_refresh_cli_auth(&self) -> Result<(), String> {
        if env_openai_key_present() {
            return Err("env API key owns this request; not rotating CLI OAuth".into());
        }
        oauth_refresh::force_refresh_chatgpt_tokens(&self.home.join("auth.json")).await
    }

    async fn force_refresh_cli_oauth(&self) -> Result<(), String> {
        oauth_refresh::force_refresh_chatgpt_oauth_tokens(&self.home.join("auth.json")).await
    }

    /// True when `auth.json` has a non-empty ChatGPT `access_token` + `account_id`.
    fn has_chatgpt_oauth_tokens(&self) -> bool {
        let auth_path = self.home.join("auth.json");
        let Ok(bytes) = std::fs::read(&auth_path) else {
            return false;
        };
        let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
            return false;
        };
        let Some(tokens) = value.get("tokens").and_then(|v| v.as_object()) else {
            return false;
        };
        let access = tokens
            .get("access_token")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.trim().is_empty());
        let account = tokens
            .get("account_id")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.trim().is_empty());
        access && account
    }

    fn conservative_models_url(&self, version: &str) -> Result<Url, ProviderError> {
        let mut url = Url::parse(&conservative_chatgpt_base_url())
            .map_err(|e| ProviderError::Auth(format!("invalid Codex conservative URL: {e}")))?;
        url.set_path("/backend-api/codex/models");
        url.query_pairs_mut().append_pair("client_version", version);
        Ok(url)
    }

    fn conservative_responses_ws_url(&self) -> Result<String, ProviderError> {
        let mut url = Url::parse(&conservative_chatgpt_ws_base_url())
            .map_err(|e| ProviderError::Auth(format!("invalid Codex conservative URL: {e}")))?;
        let scheme = match url.scheme() {
            "http" => "ws",
            "https" => "wss",
            other => {
                return Err(ProviderError::Auth(format!(
                    "unsupported Codex conservative scheme {other:?}"
                )));
            }
        };
        url.set_scheme(scheme).map_err(|_| {
            ProviderError::Auth("invalid Codex conservative WebSocket scheme".into())
        })?;
        url.set_path("/backend-api/codex/responses");
        url.set_query(None);
        Ok(url.to_string())
    }

    async fn headers(&self) -> Result<header::HeaderMap, ProviderError> {
        let mut headers = header::HeaderMap::new();
        for (name, value) in &self.http_headers {
            insert_header(&mut headers, name, value)?;
        }
        for (name, env_name) in &self.env_http_headers {
            if name == "__omni_custom_headers__" {
                for (header_name, header_value) in
                    headers_from_env(env_name).map_err(ProviderError::Auth)?
                {
                    insert_header(&mut headers, &header_name, &header_value)?;
                }
            } else if let Ok(value) = std::env::var(env_name)
                && !value.is_empty()
            {
                insert_header(&mut headers, name, &value)?;
            }
        }
        if let Some(token) = self.auth_token().await? {
            let value = header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|_| ProviderError::Auth("invalid Codex bearer token".into()))?;
            headers.insert(header::AUTHORIZATION, value);
        }
        Ok(headers)
    }

    async fn chatgpt_auth(&self) -> Result<ChatGptAuth, ProviderError> {
        let auth_path = self.home.join("auth.json");
        // Always consider OAuth tokens (even if a sibling OPENAI_API_KEY exists).
        oauth_refresh::ensure_fresh_chatgpt_oauth_tokens(&auth_path)
            .await
            .map_err(|e| {
                ProviderError::Auth(format!("Codex OAuth credential recovery failed: {e}"))
            })?;
        let bytes = tokio::fs::read(&auth_path).await.map_err(|e| {
            ProviderError::Auth(format!(
                "Codex conservative mode requires ChatGPT OAuth auth.json: failed to read auth.json: {e}"
            ))
        })?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| ProviderError::Auth(format!("Codex auth.json malformed: {e}")))?;
        let tokens = value
            .get("tokens")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                ProviderError::Auth(
                    "Codex conservative mode requires auth.json tokens from ChatGPT login".into(),
                )
            })?;
        let access_token = tokens
            .get("access_token")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                ProviderError::Auth(
                    "Codex conservative mode requires tokens.access_token in auth.json".into(),
                )
            })?
            .to_string();
        let account_id = tokens
            .get("account_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                ProviderError::Auth(
                    "Codex conservative mode requires tokens.account_id in auth.json".into(),
                )
            })?
            .to_string();
        Ok(ChatGptAuth {
            access_token,
            account_id,
        })
    }

    async fn auth_token(&self) -> Result<Option<String>, ProviderError> {
        if let Some(command) = &self.auth_command {
            return command.run(&self.home).await.map(Some);
        }
        if let Some(token) = self
            .experimental_bearer_token
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            return Ok(Some(token.to_string()));
        }
        if self.requires_openai_auth {
            return self.openai_auth_token().await.map(Some);
        }
        if let Some(env_key) = &self.env_key {
            let token = std::env::var(env_key)
                .map_err(|_| ProviderError::Auth(format!("Codex env_key {env_key} is not set")))?;
            if token.trim().is_empty() {
                return Err(ProviderError::Auth(format!(
                    "Codex env_key {env_key} is empty"
                )));
            }
            return Ok(Some(token));
        }
        // Custom gateway (requires_openai_auth=false) with no explicit source: mirror the real
        // Codex CLI, which still falls back to auth.json's OPENAI_API_KEY (then tokens.access_token,
        // then env) rather than sending no credential. Non-failing: None when nothing usable exists.
        Ok(self.openai_auth_token_fallback().await)
    }

    /// Optional auth.json-first credential lookup for the `requires_openai_auth=false` fallback.
    /// Order (auth.json prioritized over env, matching the observed CLI winner): auth.json
    /// `OPENAI_API_KEY`, auth.json `tokens.access_token`, then env `CODEX_API_KEY` /
    /// `OPENAI_API_KEY` / `CODEX_ACCESS_TOKEN`. Returns `None` when no source yields a token
    /// (never errors, unlike the strict `openai_auth_token`).
    async fn openai_auth_token_fallback(&self) -> Option<String> {
        let auth_path = self.home.join("auth.json");
        // Best-effort: fallback path still prefers whatever is on disk if recovery fails.
        let _ = oauth_refresh::ensure_fresh_chatgpt_tokens(&auth_path).await;
        if let Ok(bytes) = tokio::fs::read(&auth_path).await
            && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
        {
            if let Some(token) = value
                .get("OPENAI_API_KEY")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
            {
                return Some(token.to_string());
            }
            if let Some(token) = value
                .get("tokens")
                .and_then(|v| v.get("access_token"))
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
            {
                return Some(token.to_string());
            }
        }
        for env_key in ["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"] {
            if let Ok(token) = std::env::var(env_key)
                && !token.trim().is_empty()
            {
                return Some(token);
            }
        }
        None
    }

    async fn openai_auth_token(&self) -> Result<String, ProviderError> {
        for env_key in ["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"] {
            if let Ok(token) = std::env::var(env_key)
                && !token.trim().is_empty()
            {
                return Ok(token);
            }
        }
        let auth_path = self.home.join("auth.json");
        oauth_refresh::ensure_fresh_chatgpt_tokens(&auth_path)
            .await
            .map_err(|e| {
                ProviderError::Auth(format!("Codex OAuth credential recovery failed: {e}"))
            })?;
        let bytes = tokio::fs::read(&auth_path).await.map_err(|e| {
            ProviderError::Auth(format!(
                "Codex OpenAI auth unavailable: failed to read auth.json: {e}"
            ))
        })?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| ProviderError::Auth(format!("Codex auth.json malformed: {e}")))?;
        if let Some(token) = value
            .get("OPENAI_API_KEY")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
        {
            return Ok(token.to_string());
        }
        if let Some(token) = value
            .get("tokens")
            .and_then(|v| v.get("access_token"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
        {
            return Ok(token.to_string());
        }
        Err(ProviderError::Auth(
            "Codex OpenAI auth unavailable: auth.json held no usable token".into(),
        ))
    }
}

fn omni_codex_override_present() -> bool {
    env_nonempty("OMNI_CODEX_BASE_URL").is_some()
}

fn env_openai_key_present() -> bool {
    ["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"]
        .iter()
        .any(|key| env_nonempty(key).is_some())
}

fn is_default_openai_base(base_url: &str) -> bool {
    Url::parse(base_url)
        .ok()
        .is_some_and(|url| url.as_str().trim_end_matches('/') == DEFAULT_OPENAI_BASE_URL)
}

fn conservative_chatgpt_base_url() -> String {
    #[cfg(test)]
    if let Some(base) = env_nonempty("OMNI_CODEX_CONSERVATIVE_BASE_URL_FOR_TEST") {
        return base.trim_end_matches('/').to_string();
    }
    CONSERVATIVE_CHATGPT_BASE_URL.to_string()
}

fn conservative_chatgpt_ws_base_url() -> String {
    #[cfg(test)]
    if let Some(base) = env_nonempty("OMNI_CODEX_CONSERVATIVE_WS_BASE_URL_FOR_TEST") {
        return base.trim_end_matches('/').to_string();
    }
    conservative_chatgpt_base_url()
}

impl AuthCommand {
    async fn run(&self, home: &Path) -> Result<String, ProviderError> {
        let mut child = Command::new(&self.command)
            .args(&self.args)
            .current_dir(home)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| ProviderError::Auth(format!("Codex auth command failed to start: {e}")))?;
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.shutdown().await;
        }
        let output = timeout(
            Duration::from_millis(self.timeout_ms),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| ProviderError::Auth("Codex auth command timed out".into()))?
        .map_err(|e| ProviderError::Auth(format!("Codex auth command failed: {e}")))?;
        if !output.status.success() {
            return Err(ProviderError::Auth(format!(
                "Codex auth command exited with status {}",
                output.status
            )));
        }
        let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if token.is_empty() {
            return Err(ProviderError::Auth(
                "Codex auth command returned an empty token".into(),
            ));
        }
        Ok(token)
    }
}

fn codex_home() -> PathBuf {
    if let Some(value) = std::env::var_os("CODEX_HOME") {
        return PathBuf::from(value);
    }
    dirs::home_dir()
        .map(|home| home.join(".codex"))
        .unwrap_or_else(|| PathBuf::from(".codex"))
}

fn display_path(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

/// Auth source that the current config will prefer (no secret values).
fn describe_codex_auth_source(config: &CodexRequestConfig, auth_path: &Path) -> String {
    if let Some(env_key) = &config.env_key {
        let state = if env_nonempty(env_key).is_some() {
            "set"
        } else {
            "unset"
        };
        return format!("auth=env:{env_key} ({state})");
    }
    if config.experimental_bearer_token.is_some() {
        return "auth=experimental_bearer_token (set)".into();
    }
    if config.auth_command.is_some() {
        return "auth=auth_command".into();
    }

    // ChatGPT WS path uses tokens.* only (not env sk- keys).
    if config.should_use_chatgpt_ws() {
        return describe_codex_auth_json(auth_path, true);
    }

    // Built-in OpenAI REST: env keys beat auth.json (mirrors openai_auth_token).
    if config.requires_openai_auth {
        for env_key in ["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"] {
            if env_nonempty(env_key).is_some() {
                return format!("auth=env:{env_key} (set)");
            }
        }
        return describe_codex_auth_json(auth_path, false);
    }

    // Custom gateway fallback: auth.json then env (mirrors openai_auth_token_fallback).
    let from_file = describe_codex_auth_json(auth_path, false);
    if !from_file.contains("missing") && !from_file.contains("no usable token") {
        return from_file;
    }
    for env_key in ["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"] {
        if env_nonempty(env_key).is_some() {
            return format!("auth=env:{env_key} (set)");
        }
    }
    from_file
}

fn describe_codex_auth_json(auth_path: &Path, prefer_oauth: bool) -> String {
    if !auth_path.is_file() {
        return format!("auth={} (missing)", display_path(auth_path));
    }
    let path = display_path(auth_path);
    match std::fs::read(auth_path)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
    {
        Some(value) => {
            let has_oauth = value
                .get("tokens")
                .and_then(|t| t.get("access_token"))
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.trim().is_empty())
                && value
                    .get("tokens")
                    .and_then(|t| t.get("account_id"))
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| !s.trim().is_empty());
            let has_api_key = value
                .get("OPENAI_API_KEY")
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.trim().is_empty());
            if prefer_oauth && has_oauth {
                format!("auth={path} (chatgpt-oauth, present)")
            } else if has_api_key {
                format!("auth={path} (OPENAI_API_KEY, present)")
            } else if has_oauth {
                format!("auth={path} (chatgpt-oauth, present)")
            } else {
                format!("auth={path} (present, no usable token)")
            }
        }
        None => format!("auth={path} (present, unreadable)"),
    }
}

fn toml_str<'a>(value: &'a toml::Value, path: &[&str]) -> Option<&'a str> {
    let mut cur = value;
    for key in path {
        cur = cur.get(*key)?;
    }
    cur.as_str()
}

fn string_map(value: &toml::Value) -> Option<BTreeMap<String, String>> {
    let table = value.as_table()?;
    let mut out = BTreeMap::new();
    for (key, value) in table {
        if let Some(s) = value.as_str() {
            out.insert(key.clone(), s.to_string());
        } else {
            out.insert(key.clone(), value.to_string());
        }
    }
    Some(out)
}

fn parse_auth_command(table: &toml::map::Map<String, toml::Value>) -> Option<AuthCommand> {
    let command = table.get("command")?.as_str()?.to_string();
    let args = table
        .get("args")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let timeout_ms = table
        .get("timeout_ms")
        .and_then(|v| v.as_integer())
        .and_then(|v| u64::try_from(v).ok())
        .unwrap_or(DEFAULT_AUTH_COMMAND_TIMEOUT_MS);
    Some(AuthCommand {
        command,
        args,
        timeout_ms,
    })
}

fn conservative_user_agent(version: &str) -> String {
    format!("codex_exec/{version} (Ubuntu 24.4.0; x86_64) unknown (codex_exec; {version})")
}

fn conservative_codex_headers(
    version: &str,
    auth: &ChatGptAuth,
) -> Result<header::HeaderMap, ProviderError> {
    let mut headers = header::HeaderMap::new();
    insert_header(&mut headers, "version", version)?;
    let bearer = format!("Bearer {}", auth.access_token);
    insert_header(&mut headers, "authorization", &bearer)?;
    if let Some(value) = headers.get_mut(header::AUTHORIZATION) {
        value.set_sensitive(true);
    }
    insert_header(&mut headers, "chatgpt-account-id", &auth.account_id)?;
    insert_header(&mut headers, "accept", "*/*")?;
    insert_header(&mut headers, "originator", CONSERVATIVE_ORIGINATOR)?;
    insert_header(
        &mut headers,
        "user-agent",
        &conservative_user_agent(version),
    )?;
    Ok(headers)
}

fn conservative_turn_metadata() -> String {
    json!({
        "installation_id": "omni-llm-provider",
        "session_id": CONSERVATIVE_CLIENT_REQUEST_ID,
        "thread_id": CONSERVATIVE_CLIENT_REQUEST_ID,
        "turn_id": "",
        "window_id": CONSERVATIVE_WINDOW_ID,
        "request_kind": "prewarm",
        "thread_source": "user",
        "sandbox": "none",
        "workspaces": {},
    })
    .to_string()
}

fn conservative_ws_request(
    ws_url: &str,
    version: &str,
    auth: &ChatGptAuth,
) -> Result<http::Request<()>, ProviderError> {
    let mut request = ws_url
        .into_client_request()
        .map_err(|e| ProviderError::Auth(format!("invalid Codex WebSocket request: {e}")))?;
    let headers = request.headers_mut();
    headers.insert(
        "chatgpt-account-id",
        http_header_value("chatgpt-account-id", &auth.account_id)?,
    );
    let mut bearer = http_header_value("authorization", &format!("Bearer {}", auth.access_token))?;
    bearer.set_sensitive(true);
    headers.insert(http::header::AUTHORIZATION, bearer);
    headers.insert(
        http::header::USER_AGENT,
        http_header_value("user-agent", &conservative_user_agent(version))?,
    );
    headers.insert(
        "originator",
        http::HeaderValue::from_static(CONSERVATIVE_ORIGINATOR),
    );
    headers.insert(
        "openai-beta",
        http::HeaderValue::from_static(CONSERVATIVE_OPENAI_BETA),
    );
    headers.insert("version", http_header_value("version", version)?);
    headers.insert(
        "x-codex-beta-features",
        http::HeaderValue::from_static(CONSERVATIVE_BETA_FEATURES),
    );
    headers.insert(
        "x-client-request-id",
        http::HeaderValue::from_static(CONSERVATIVE_CLIENT_REQUEST_ID),
    );
    headers.insert(
        "session-id",
        http::HeaderValue::from_static(CONSERVATIVE_CLIENT_REQUEST_ID),
    );
    headers.insert(
        "thread-id",
        http::HeaderValue::from_static(CONSERVATIVE_CLIENT_REQUEST_ID),
    );
    headers.insert(
        "x-codex-window-id",
        http::HeaderValue::from_static(CONSERVATIVE_WINDOW_ID),
    );
    headers.insert(
        "x-codex-turn-metadata",
        http_header_value("x-codex-turn-metadata", &conservative_turn_metadata())?,
    );
    Ok(request)
}

fn http_header_value(name: &str, value: &str) -> Result<http::HeaderValue, ProviderError> {
    http::HeaderValue::from_str(value)
        .map_err(|_| ProviderError::Auth(format!("invalid Codex header value for {name}")))
}

fn codex_response_create_body(req: &CanonicalRequest) -> Result<Value, ProviderError> {
    let mut body = codex_responses_body(req, true)?;
    body["type"] = Value::String("response.create".into());
    if let Some(obj) = body.as_object_mut() {
        // WebSocket frames are the stream; REST `stream: true` is not used.
        obj.remove("stream");
        // ChatGPT's codex backend rejects max_output_tokens (HTTP 400). The
        // public OpenAI Responses REST path still accepts it via
        // `codex_responses_body`; only the WS response.create shape strips it.
        obj.remove("max_output_tokens");
    }
    Ok(body)
}

fn insert_header(
    headers: &mut header::HeaderMap,
    name: &str,
    value: &str,
) -> Result<(), ProviderError> {
    let name = header::HeaderName::from_bytes(name.as_bytes())
        .map_err(|_| ProviderError::Auth(format!("invalid Codex header name {name:?}")))?;
    let value = header::HeaderValue::from_str(value)
        .map_err(|_| ProviderError::Auth(format!("invalid Codex header value for {name}")))?;
    headers.insert(name, value);
    Ok(())
}

async fn collect_canonical_stream(
    stream: &mut CanonicalStream,
    model: &str,
) -> Result<CanonicalResponse, ProviderError> {
    let mut content = String::new();
    let mut refusal = String::new();
    let mut tool_slots: BTreeMap<u32, CanonicalToolCall> = BTreeMap::new();
    let mut usage = Default::default();
    let mut id = None;
    let mut metadata = None;
    let mut annotations = Vec::new();
    let mut finish_reason = None;
    let mut saw_finish = false;

    while let Some(event) = stream.next().await {
        match event? {
            CanonicalStreamEvent::ResponseMetadata(meta) => {
                if let Some(response_id) = meta.id.clone() {
                    id = Some(response_id);
                }
                metadata = Some(meta);
            }
            CanonicalStreamEvent::TextDelta(delta) => content.push_str(&delta),
            CanonicalStreamEvent::RefusalDelta(delta) => refusal.push_str(&delta),
            CanonicalStreamEvent::ReasoningDelta(_)
            | CanonicalStreamEvent::ReasoningSignatureDelta(_) => {}
            CanonicalStreamEvent::OutputAnnotations(new_annotations) => {
                annotations.extend(new_annotations);
            }
            CanonicalStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                let slot = tool_slots
                    .entry(index)
                    .or_insert_with(|| CanonicalToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: String::new(),
                    });
                if let Some(id) = id {
                    slot.id = id;
                }
                if let Some(name) = name {
                    slot.name = name;
                }
                slot.arguments.push_str(&arguments_delta);
            }
            CanonicalStreamEvent::Usage(u) => usage = u,
            CanonicalStreamEvent::Finish { finish_reason: fr } => {
                finish_reason = fr;
                saw_finish = true;
            }
        }
    }

    if !saw_finish {
        return Err(ProviderError::upstream(
            "codex conservative websocket ended before a terminal response event",
        ));
    }

    let tool_calls = tool_slots.into_values().collect::<Vec<_>>();
    Ok(CanonicalResponse {
        model: model.to_string(),
        content,
        refusal: (!refusal.is_empty()).then_some(refusal),
        finish_reason,
        usage,
        id,
        annotations,
        metadata,
        tool_calls,
        reasoning: Vec::new(),
    })
}

fn codex_responses_body(req: &CanonicalRequest, stream: bool) -> Result<Value, ProviderError> {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for message in &req.messages {
        if message.role == "system" || message.role == "developer" {
            if message.content.has_cache_marks()
                || matches!(&message.content, CanonicalContent::Blocks(blocks) if blocks.iter().any(|block| matches!(block, CanonicalBlock::File { .. } | CanonicalBlock::Image { .. })))
            {
                // Marked instructions cannot live on the string `instructions`
                // field. Emit a developer input item so breakpoints survive.
                let mut marked = message.clone();
                marked.role = "developer".into();
                append_message_items(&marked, &mut input);
            } else {
                instructions.push(message.content.as_text());
            }
            continue;
        }
        append_message_items(message, &mut input);
    }

    let mut body = json!({
        "model": req.model,
        "input": input,
        "stream": stream,
    });
    if !instructions.is_empty() {
        body["instructions"] = Value::String(instructions.join("\n\n"));
    }
    if let Some(max_tokens) = req.max_tokens {
        body["max_output_tokens"] = json!(max_tokens);
    }
    if let Some(temperature) = req.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = req.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(CanonicalReasoning { effort, .. }) = &req.reasoning
        && let Some(effort) = effort.as_deref()
        && let Some(effort) = codex_reasoning_effort(effort, &req.model)?
    {
        body["reasoning"] = json!({ "effort": effort });
    }
    if let Some(tools) = &req.tools
        && !tools.is_empty()
    {
        body["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": omni_core::codex_strict(&tool.parameters, tool.strict),
                    })
                })
                .collect(),
        );
    }
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = match choice {
            CanonicalToolChoice::Auto => json!("auto"),
            CanonicalToolChoice::Required => json!("required"),
            CanonicalToolChoice::None => json!("none"),
            CanonicalToolChoice::Specific { name } => json!({"type": "function", "name": name}),
        };
    }
    apply_codex_cache(&mut body, req);
    if let Some(extras) = &req.provider_extras
        && let Some(obj) = extras.as_object()
    {
        // Chat `response_format` is a Codex allowlist extra, but Responses
        // rejects that key. Map it to `text.format` before the copy loop so it
        // never lands on the upstream body (issue #32).
        let mapped_format = match obj.get("response_format") {
            Some(Value::Null) | None => None,
            Some(value) => Some(map_chat_response_format(value)?),
        };
        for (key, value) in obj {
            if key == "response_format" {
                continue;
            }
            if !codex_extra_allowed(key) {
                return Err(ProviderError::Other(anyhow::anyhow!(
                    "unsupported provider extra for codex: {key}"
                )));
            }
            body[key] = value.clone();
        }
        if let Some(format) = mapped_format {
            apply_mapped_text_format(&mut body, format)?;
        }
    }
    Ok(body)
}

/// Chat Completions `response_format` → Responses `text.format`.
///
/// `json_object` maps 1:1. `json_schema` unwraps the nested
/// `json_schema.{name,schema}` object and keeps `description`/`strict` when
/// present. Other shapes fail loud as a named 400.
fn map_chat_response_format(value: &Value) -> Result<Value, ProviderError> {
    let obj = value
        .as_object()
        .ok_or_else(|| ProviderError::BadRequest("response_format must be an object".into()))?;
    let typ = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ProviderError::BadRequest("response_format.type is required".into()))?;
    match typ {
        "text" => {
            reject_unknown_keys(obj, &["type"], "response_format")?;
            Ok(json!({ "type": "text" }))
        }
        "json_object" => {
            reject_unknown_keys(obj, &["type"], "response_format")?;
            Ok(json!({ "type": "json_object" }))
        }
        "json_schema" => {
            reject_unknown_keys(obj, &["type", "json_schema"], "response_format")?;
            let schema_obj = obj
                .get("json_schema")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    ProviderError::BadRequest(
                        "response_format.json_schema must be an object".into(),
                    )
                })?;
            reject_unknown_keys(
                schema_obj,
                &["name", "schema", "description", "strict"],
                "response_format.json_schema",
            )?;
            let name = schema_obj.get("name").ok_or_else(|| {
                ProviderError::BadRequest("response_format.json_schema.name is required".into())
            })?;
            if !name.is_string() {
                return Err(ProviderError::BadRequest(
                    "response_format.json_schema.name must be a string".into(),
                ));
            }
            let schema = schema_obj.get("schema").ok_or_else(|| {
                ProviderError::BadRequest("response_format.json_schema.schema is required".into())
            })?;
            if !schema.is_object() {
                return Err(ProviderError::BadRequest(
                    "response_format.json_schema.schema must be an object".into(),
                ));
            }
            let mut format = json!({
                "type": "json_schema",
                "name": name,
                "schema": schema,
            });
            if let Some(description) = schema_obj
                .get("description")
                .filter(|value| !value.is_null())
            {
                if !description.is_string() {
                    return Err(ProviderError::BadRequest(
                        "response_format.json_schema.description must be a string".into(),
                    ));
                }
                format["description"] = description.clone();
            }
            if let Some(strict) = schema_obj.get("strict").filter(|value| !value.is_null()) {
                if !strict.is_boolean() {
                    return Err(ProviderError::BadRequest(
                        "response_format.json_schema.strict must be a boolean".into(),
                    ));
                }
                format["strict"] = strict.clone();
            }
            Ok(format)
        }
        other => Err(ProviderError::BadRequest(format!(
            "response_format type {other} is unmappable"
        ))),
    }
}

fn reject_unknown_keys(
    obj: &serde_json::Map<String, Value>,
    allowed: &[&str],
    prefix: &str,
) -> Result<(), ProviderError> {
    if let Some(key) = obj.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(ProviderError::BadRequest(format!(
            "{prefix} has unknown field {key}"
        )));
    }
    Ok(())
}

/// Merge a mapped Chat `response_format` into Responses `text.format`.
///
/// Absent `text` creates `{format}`. Existing `text` without `format` keeps
/// siblings such as `verbosity`. An equivalent `text.format` is accepted; any
/// other value is a named 400.
fn apply_mapped_text_format(body: &mut Value, format: Value) -> Result<(), ProviderError> {
    if body.get("text").is_none() {
        body["text"] = json!({ "format": format });
        return Ok(());
    }
    let Some(text) = body.get("text").and_then(Value::as_object) else {
        return Err(ProviderError::BadRequest(
            "response_format cannot merge into non-object text".into(),
        ));
    };
    if let Some(existing) = text.get("format") {
        if existing == &format {
            return Ok(());
        }
        return Err(ProviderError::BadRequest(
            "response_format conflicts with text.format".into(),
        ));
    }
    body["text"]["format"] = format;
    Ok(())
}

/// Map canonical effort onto OpenAI Responses vocabulary used by Codex.
///
/// Documented aliases: `max`→`high`. Empty omits the field. Explicit `"none"`
/// still emits so Codex can disable reasoning (unlike Grok). First-class
/// `xhigh` is kept (OpenAI/Codex wire). Unmappable values (e.g. `ultra`) fail
/// loud — never silent omit (issue #20).
fn codex_reasoning_effort(
    effort: &str,
    model: &str,
) -> Result<Option<&'static str>, ProviderError> {
    const SUPPORTED: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh"];
    match effort {
        "" => Ok(None),
        "none" => Ok(Some("none")),
        "minimal" => Ok(Some("minimal")),
        "low" => Ok(Some("low")),
        "medium" => Ok(Some("medium")),
        "high" | "max" => Ok(Some("high")),
        "xhigh" => Ok(Some("xhigh")),
        other => Err(ProviderError::unsupported_reasoning_effort(
            "codex",
            Some(model),
            "responses",
            other,
            SUPPORTED,
        )),
    }
}

fn apply_codex_cache(body: &mut Value, req: &CanonicalRequest) {
    if let Some(key) = req
        .cache_routing_identity()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        body["prompt_cache_key"] = json!(key);
    }
    let uses_ttl = codex_model_uses_ttl(&req.model);
    let cache = req.cache.as_ref();
    if uses_ttl {
        let mut options = serde_json::Map::new();
        if let Some(mode) = cache.and_then(|c| c.mode) {
            options.insert(
                "mode".into(),
                json!(match mode {
                    CanonicalCacheMode::Implicit => "implicit",
                    CanonicalCacheMode::Explicit => "explicit",
                }),
            );
        }
        if let Some(secs) = cache.and_then(|c| c.requested_duration_secs()) {
            if let Some(ttl) = clamp_duration(secs, CanonicalCacheTtl::CODEX_GPT56) {
                options.insert("ttl".into(), json!(ttl.as_str()));
            }
        }
        if !options.is_empty() {
            body["prompt_cache_options"] = Value::Object(options);
        }
    } else if let Some(cache) = cache {
        if let Some(ret) = cache.legacy_retention {
            body["prompt_cache_retention"] = json!(ret.as_str());
        } else if let Some(secs) = cache.requested_duration_secs()
            && let Some(ret) = clamp_retention(
                secs,
                [
                    CanonicalCacheRetention::InMemory,
                    CanonicalCacheRetention::TwentyFourHours,
                ],
            )
        {
            body["prompt_cache_retention"] = json!(ret.as_str());
        }
    }
}

fn codex_model_uses_ttl(model: &str) -> bool {
    let resolved = CODEX_CATALOG
        .iter()
        .find(|entry| entry.matches(model))
        .map(|entry| entry.id)
        .unwrap_or(model);
    is_gpt56_or_later(resolved)
}

fn is_gpt56_or_later(model: &str) -> bool {
    let id = model.to_ascii_lowercase();
    let Some(rest) = id.strip_prefix("gpt-") else {
        return false;
    };
    let ver = rest.split('-').next().unwrap_or(rest);
    let mut parts = ver.split('.');
    let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    major > 5 || (major == 5 && minor >= 6)
}

fn openai_breakpoint_part(mut part: Value, cache: Option<&CanonicalCacheMark>) -> Value {
    if cache.is_some() {
        part["prompt_cache_breakpoint"] = json!({ "mode": "explicit" });
    }
    part
}

fn append_message_items(message: &CanonicalMessage, input: &mut Vec<Value>) {
    match &message.content {
        CanonicalContent::Text(text) => {
            input.push(json!({
                "type": "message",
                "role": message.role,
                "content": text,
            }));
        }
        CanonicalContent::Blocks(blocks) => {
            let mut text = String::new();
            let mut content_parts: Vec<Value> = Vec::new();
            let mut has_image = false;
            let mut has_breakpoint = false;
            for block in blocks {
                match block {
                    CanonicalBlock::Text { text: t, cache } => {
                        text.push_str(t);
                        if !t.is_empty() || cache.is_some() {
                            if cache.is_some() {
                                has_breakpoint = true;
                            }
                            content_parts.push(openai_breakpoint_part(
                                json!({
                                    "type": "input_text",
                                    "text": t,
                                }),
                                cache.as_ref(),
                            ));
                        }
                    }
                    CanonicalBlock::Image { source, cache } => {
                        has_image = true;
                        if cache.is_some() {
                            has_breakpoint = true;
                        }
                        content_parts.push(openai_breakpoint_part(
                            json!({
                                "type": "input_image",
                                "image_url": source.as_image_url(),
                            }),
                            cache.as_ref(),
                        ));
                    }
                    CanonicalBlock::File {
                        source,
                        filename,
                        detail,
                        cache,
                    } => {
                        has_image = true;
                        if cache.is_some() {
                            has_breakpoint = true;
                        }
                        let mut part = source.as_responses_part();
                        if let Some(filename) = filename {
                            part["filename"] = json!(filename);
                        }
                        if let Some(detail) = detail {
                            part["detail"] = json!(detail);
                        }
                        content_parts.push(openai_breakpoint_part(part, cache.as_ref()));
                    }
                    CanonicalBlock::ToolUse {
                        id,
                        name,
                        arguments,
                        ..
                    } => {
                        flush_codex_message(
                            input,
                            &message.role,
                            &mut text,
                            &mut content_parts,
                            &mut has_image,
                            &mut has_breakpoint,
                        );
                        input.push(json!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": arguments,
                        }));
                    }
                    CanonicalBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                        ..
                    } => {
                        flush_codex_message(
                            input,
                            &message.role,
                            &mut text,
                            &mut content_parts,
                            &mut has_image,
                            &mut has_breakpoint,
                        );
                        let wire = omni_common::encode_tool_result_content(content, *is_error);
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": tool_use_id,
                            "output": wire,
                        }));
                    }
                }
            }
            flush_codex_message(
                input,
                &message.role,
                &mut text,
                &mut content_parts,
                &mut has_image,
                &mut has_breakpoint,
            );
        }
    }
}

fn flush_codex_message(
    input: &mut Vec<Value>,
    role: &str,
    text: &mut String,
    content_parts: &mut Vec<Value>,
    has_image: &mut bool,
    has_breakpoint: &mut bool,
) {
    if *has_image && content_parts.is_empty() {
        text.clear();
        *has_image = false;
        *has_breakpoint = false;
        return;
    }
    if !*has_image && !*has_breakpoint && text.is_empty() {
        content_parts.clear();
        return;
    }
    let content = if *has_image || *has_breakpoint {
        Value::Array(std::mem::take(content_parts))
    } else {
        content_parts.clear();
        Value::String(std::mem::take(text))
    };
    input.push(json!({
        "type": "message",
        "role": role,
        "content": content,
    }));
    text.clear();
    *has_image = false;
    *has_breakpoint = false;
}

pub fn codex_extra_allowed(key: &str) -> bool {
    matches!(
        key,
        "store"
            | "previous_response_id"
            | "metadata"
            | "parallel_tool_calls"
            | "service_tier"
            | "response_format"
            | "text"
    )
}

#[derive(Debug, Clone, Default)]
struct CodexErrorRedactor {
    secrets: Vec<String>,
}

impl CodexErrorRedactor {
    fn for_secrets(secrets: impl IntoIterator<Item = String>) -> Self {
        let mut secrets = secrets
            .into_iter()
            .filter(|secret| !secret.trim().is_empty())
            .collect::<Vec<_>>();
        secrets.sort_by_key(|value| std::cmp::Reverse(value.len()));
        secrets.dedup();
        Self { secrets }
    }

    fn from_request(url: &Url, headers: &HeaderMap) -> Self {
        let mut secrets = Vec::new();
        for (name, value) in headers {
            let name = name.as_str();
            if let Ok(value) = value.to_str() {
                collect_secret_candidates(name, value, &mut secrets);
            }
        }
        for (name, value) in url.query_pairs() {
            collect_secret_candidates(&name, &value, &mut secrets);
        }
        secrets.sort_by_key(|value| std::cmp::Reverse(value.len()));
        secrets.dedup();
        Self { secrets }
    }

    fn redact(&self, input: &str) -> String {
        let mut out = redact(input);
        for secret in &self.secrets {
            out = out.replace(secret, "<redacted>");
        }
        out
    }
}

impl ErrorRedactor for CodexErrorRedactor {
    fn redact(&self, input: &str) -> String {
        // Delegate to the inherent method (named via the type path so it can never
        // resolve back to this trait method, which would recurse infinitely).
        CodexErrorRedactor::redact(self, input)
    }
}

fn collect_secret_candidates(name: &str, value: &str, out: &mut Vec<String>) {
    let value = value.trim();
    if value.is_empty() || !is_sensitive_name(name) {
        return;
    }
    out.push(value.to_string());
    if name.eq_ignore_ascii_case("authorization") {
        if let Some((_, token)) = value.split_once(' ') {
            let token = token.trim();
            if !token.is_empty() {
                out.push(token.to_string());
            }
        }
    }
}

fn is_sensitive_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    ["auth", "token", "key", "secret", "password", "credential"]
        .iter()
        .any(|needle| name.contains(needle))
}

fn redact(input: &str) -> String {
    responses_upstream::redact_prefixed_secrets(input, &["sk-", "xai-", "eyJ"])
}

fn map_ws_open_error(
    err: tokio_tungstenite::tungstenite::Error,
    redactor: &CodexErrorRedactor,
) -> ProviderError {
    match err {
        tokio_tungstenite::tungstenite::Error::Http(response) => {
            let status = response.status().as_u16();
            let body = match response.body() {
                Some(bytes) => {
                    let bytes = &bytes[..bytes.len().min(UPSTREAM_ERROR_BODY_PREFIX)];
                    String::from_utf8_lossy(bytes).into_owned()
                }
                None => String::new(),
            };
            ProviderError::upstream_status(
                status,
                redactor.redact(&format!(
                    "codex conservative websocket handshake HTTP {status}: {body}"
                )),
            )
        }
        other => ProviderError::upstream(redactor.redact(&format!(
            "codex conservative websocket connect error: {other}"
        ))),
    }
}

async fn read_capped_error_body(mut resp: reqwest::Response) -> String {
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(UPSTREAM_ERROR_BODY_READ_TIMEOUT, async {
        while buf.len() < UPSTREAM_ERROR_BODY_PREFIX {
            match resp.chunk().await {
                Ok(Some(chunk)) => {
                    let room = UPSTREAM_ERROR_BODY_PREFIX - buf.len();
                    let n = chunk.len().min(room);
                    buf.extend_from_slice(&chunk[..n]);
                    if n < chunk.len() {
                        break;
                    }
                }
                Ok(None) | Err(_) => break,
            }
        }
    })
    .await;
    String::from_utf8_lossy(&buf).into_owned()
}

fn is_upstream_status(err: &ProviderError, want: u16) -> bool {
    matches!(
        err,
        ProviderError::Upstream {
            status: Some(status),
            ..
        } if *status == want
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_common::responses_upstream::MAX_SSE_EVENT_BYTES;
    use omni_core::{CanonicalResponseMetadata, CanonicalTool, CanonicalUsage};
    use std::sync::Mutex;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_tungstenite::accept_hdr_async;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    async fn collect_stream_events(
        provider: &CodexProvider,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let mut stream = provider
            .send_stream(CanonicalRequest {
                model: "gpt-custom".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
            .expect("Codex streaming should open");
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event);
        }
        events
    }

    struct TempCodexHome {
        path: PathBuf,
        old_home: Option<std::ffi::OsString>,
        old_codex_api_key: Option<std::ffi::OsString>,
        old_openai: Option<std::ffi::OsString>,
        old_codex_access_token: Option<std::ffi::OsString>,
        old_custom: Option<std::ffi::OsString>,
        old_custom_header: Option<std::ffi::OsString>,
        old_omni_base_url: Option<std::ffi::OsString>,
        old_omni_model: Option<std::ffi::OsString>,
        old_omni_wire_api: Option<std::ffi::OsString>,
        old_omni_auth_token: Option<std::ffi::OsString>,
        old_omni_api_key: Option<std::ffi::OsString>,
        old_omni_custom_headers: Option<std::ffi::OsString>,
        old_conservative_base_url: Option<std::ffi::OsString>,
        old_conservative_ws_base_url: Option<std::ffi::OsString>,
    }

    impl TempCodexHome {
        fn new(config: &str, auth: Option<&str>) -> Self {
            let path = std::env::temp_dir().join(format!(
                "omni-codex-home-{}",
                chrono::Utc::now().timestamp_nanos_opt().unwrap()
            ));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("config.toml"), config).unwrap();
            if let Some(auth) = auth {
                std::fs::write(path.join("auth.json"), auth).unwrap();
            }
            let old_home = std::env::var_os("CODEX_HOME");
            let old_codex_api_key = std::env::var_os("CODEX_API_KEY");
            let old_openai = std::env::var_os("OPENAI_API_KEY");
            let old_codex_access_token = std::env::var_os("CODEX_ACCESS_TOKEN");
            let old_custom = std::env::var_os("CUSTOM_CODEX_KEY");
            let old_custom_header = std::env::var_os("CUSTOM_CODEX_HEADER");
            let old_omni_base_url = std::env::var_os("OMNI_CODEX_BASE_URL");
            let old_omni_model = std::env::var_os("OMNI_CODEX_MODEL");
            let old_omni_wire_api = std::env::var_os("OMNI_CODEX_WIRE_API");
            let old_omni_auth_token = std::env::var_os("OMNI_CODEX_AUTH_TOKEN");
            let old_omni_api_key = std::env::var_os("OMNI_CODEX_API_KEY");
            let old_omni_custom_headers = std::env::var_os("OMNI_CODEX_CUSTOM_HEADERS");
            let old_conservative_base_url =
                std::env::var_os("OMNI_CODEX_CONSERVATIVE_BASE_URL_FOR_TEST");
            let old_conservative_ws_base_url =
                std::env::var_os("OMNI_CODEX_CONSERVATIVE_WS_BASE_URL_FOR_TEST");
            unsafe {
                std::env::set_var("CODEX_HOME", &path);
                std::env::remove_var("CODEX_API_KEY");
                std::env::remove_var("OPENAI_API_KEY");
                std::env::remove_var("CODEX_ACCESS_TOKEN");
                std::env::remove_var("CUSTOM_CODEX_KEY");
                std::env::remove_var("CUSTOM_CODEX_HEADER");
                std::env::remove_var("OMNI_CODEX_BASE_URL");
                std::env::remove_var("OMNI_CODEX_MODEL");
                std::env::remove_var("OMNI_CODEX_WIRE_API");
                std::env::remove_var("OMNI_CODEX_AUTH_TOKEN");
                std::env::remove_var("OMNI_CODEX_API_KEY");
                std::env::remove_var("OMNI_CODEX_CUSTOM_HEADERS");
                std::env::remove_var("OMNI_CODEX_CONSERVATIVE_BASE_URL_FOR_TEST");
                std::env::remove_var("OMNI_CODEX_CONSERVATIVE_WS_BASE_URL_FOR_TEST");
            }
            Self {
                path,
                old_home,
                old_codex_api_key,
                old_openai,
                old_codex_access_token,
                old_custom,
                old_custom_header,
                old_omni_base_url,
                old_omni_model,
                old_omni_wire_api,
                old_omni_auth_token,
                old_omni_api_key,
                old_omni_custom_headers,
                old_conservative_base_url,
                old_conservative_ws_base_url,
            }
        }
    }

    impl Drop for TempCodexHome {
        fn drop(&mut self) {
            unsafe {
                match &self.old_home {
                    Some(v) => std::env::set_var("CODEX_HOME", v),
                    None => std::env::remove_var("CODEX_HOME"),
                }
                match &self.old_codex_api_key {
                    Some(v) => std::env::set_var("CODEX_API_KEY", v),
                    None => std::env::remove_var("CODEX_API_KEY"),
                }
                match &self.old_openai {
                    Some(v) => std::env::set_var("OPENAI_API_KEY", v),
                    None => std::env::remove_var("OPENAI_API_KEY"),
                }
                match &self.old_codex_access_token {
                    Some(v) => std::env::set_var("CODEX_ACCESS_TOKEN", v),
                    None => std::env::remove_var("CODEX_ACCESS_TOKEN"),
                }
                match &self.old_custom {
                    Some(v) => std::env::set_var("CUSTOM_CODEX_KEY", v),
                    None => std::env::remove_var("CUSTOM_CODEX_KEY"),
                }
                match &self.old_custom_header {
                    Some(v) => std::env::set_var("CUSTOM_CODEX_HEADER", v),
                    None => std::env::remove_var("CUSTOM_CODEX_HEADER"),
                }
                match &self.old_omni_base_url {
                    Some(v) => std::env::set_var("OMNI_CODEX_BASE_URL", v),
                    None => std::env::remove_var("OMNI_CODEX_BASE_URL"),
                }
                match &self.old_omni_model {
                    Some(v) => std::env::set_var("OMNI_CODEX_MODEL", v),
                    None => std::env::remove_var("OMNI_CODEX_MODEL"),
                }
                match &self.old_omni_wire_api {
                    Some(v) => std::env::set_var("OMNI_CODEX_WIRE_API", v),
                    None => std::env::remove_var("OMNI_CODEX_WIRE_API"),
                }
                match &self.old_omni_auth_token {
                    Some(v) => std::env::set_var("OMNI_CODEX_AUTH_TOKEN", v),
                    None => std::env::remove_var("OMNI_CODEX_AUTH_TOKEN"),
                }
                match &self.old_omni_api_key {
                    Some(v) => std::env::set_var("OMNI_CODEX_API_KEY", v),
                    None => std::env::remove_var("OMNI_CODEX_API_KEY"),
                }
                match &self.old_omni_custom_headers {
                    Some(v) => std::env::set_var("OMNI_CODEX_CUSTOM_HEADERS", v),
                    None => std::env::remove_var("OMNI_CODEX_CUSTOM_HEADERS"),
                }
                match &self.old_conservative_base_url {
                    Some(v) => std::env::set_var("OMNI_CODEX_CONSERVATIVE_BASE_URL_FOR_TEST", v),
                    None => std::env::remove_var("OMNI_CODEX_CONSERVATIVE_BASE_URL_FOR_TEST"),
                }
                match &self.old_conservative_ws_base_url {
                    Some(v) => std::env::set_var("OMNI_CODEX_CONSERVATIVE_WS_BASE_URL_FOR_TEST", v),
                    None => std::env::remove_var("OMNI_CODEX_CONSERVATIVE_WS_BASE_URL_FOR_TEST"),
                }
            }
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    const CUSTOM_PROVIDER_CONFIG: &str = r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com"
wire_api = "responses"
requires_openai_auth = false
"#;

    // Issue #1: a custom gateway (requires_openai_auth=false) with no explicit source must mirror
    // the real Codex CLI and fall back to auth.json's OPENAI_API_KEY, prioritized over env.
    #[test]
    fn custom_provider_falls_back_to_auth_json_over_env() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            CUSTOM_PROVIDER_CONFIG,
            Some(r#"{"OPENAI_API_KEY":"sk-from-auth-json"}"#),
        );
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-from-env-should-lose");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer sk-from-auth-json",
            "auth.json OPENAI_API_KEY must win over env (matches the CLI's observed winner)"
        );
    }

    #[test]
    fn custom_provider_falls_back_to_env_when_no_auth_json() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(CUSTOM_PROVIDER_CONFIG, None);
        unsafe {
            std::env::remove_var("CODEX_API_KEY");
            std::env::remove_var("CODEX_ACCESS_TOKEN");
            std::env::set_var("OPENAI_API_KEY", "sk-from-env-fallback");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer sk-from-env-fallback",
            "env OPENAI_API_KEY is the fallback when auth.json holds no usable token"
        );
    }

    #[test]
    fn custom_provider_no_auth_source_sends_no_header() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(CUSTOM_PROVIDER_CONFIG, None);
        unsafe {
            std::env::remove_var("CODEX_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("CODEX_ACCESS_TOKEN");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert!(
            !headers.contains_key(header::AUTHORIZATION),
            "no usable source must yield no Authorization header (Ok(None)), not an error"
        );
    }

    #[test]
    fn omni_codex_override_uses_only_omni_auth_and_headers() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-native"
"#,
            Some(r#"{"OPENAI_API_KEY":"sk-native-must-not-leak"}"#),
        );
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-env-must-not-leak");
            std::env::set_var("OMNI_CODEX_BASE_URL", "https://omni-codex.example.com");
            std::env::set_var("OMNI_CODEX_MODEL", "gpt-override");
            std::env::set_var("OMNI_CODEX_AUTH_TOKEN", "omni-token");
            std::env::set_var("OMNI_CODEX_API_KEY", "omni-api-key-must-not-win");
            std::env::set_var(
                "OMNI_CODEX_CUSTOM_HEADERS",
                "X-Omni: yes\nAuthorization: Bearer header-must-not-win",
            );
        }

        assert!(CodexProvider::detected());
        let cfg = CodexRequestConfig::load().unwrap();
        assert_eq!(cfg.model, "gpt-override");
        assert_eq!(cfg.base_url, "https://omni-codex.example.com");
        assert!(!cfg.requires_openai_auth);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer omni-token"
        );
        assert_eq!(headers.get("x-omni").unwrap(), "yes");
        assert!(
            !headers
                .get(header::AUTHORIZATION)
                .unwrap()
                .to_str()
                .unwrap()
                .contains("sk-"),
            "OMNI_CODEX_BASE_URL must not inherit native OpenAI/Codex credentials"
        );
    }

    #[test]
    fn omni_codex_empty_auth_token_falls_through_to_api_key() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new("", None);
        unsafe {
            std::env::set_var("OMNI_CODEX_BASE_URL", "https://omni-codex.example.com");
            std::env::set_var("OMNI_CODEX_AUTH_TOKEN", " ");
            std::env::set_var("OMNI_CODEX_API_KEY", "api-key-token");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer api-key-token"
        );
    }

    #[test]
    fn load_fails_when_codex_home_does_not_exist() {
        // WHY: an invalid/missing CODEX_HOME must not start as empty defaults;
        // operators need a hard config error at launch.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let missing = std::env::temp_dir().join(format!(
            "omni-codex-missing-home-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Ensure absent.
        let _ = std::fs::remove_dir_all(&missing);
        let old_home = std::env::var_os("CODEX_HOME");
        let old_override = std::env::var_os("OMNI_CODEX_BASE_URL");
        unsafe {
            std::env::set_var("CODEX_HOME", &missing);
            std::env::remove_var("OMNI_CODEX_BASE_URL");
        }
        let err = CodexRequestConfig::load().expect_err("missing CODEX_HOME must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("does not exist"),
            "expected missing-home error, got: {msg}"
        );
        assert!(
            CodexProvider::startup_source_summary().is_err(),
            "startup summary must fail when CODEX_HOME is invalid"
        );
        unsafe {
            match old_home {
                Some(v) => std::env::set_var("CODEX_HOME", v),
                None => std::env::remove_var("CODEX_HOME"),
            }
            match old_override {
                Some(v) => std::env::set_var("OMNI_CODEX_BASE_URL", v),
                None => std::env::remove_var("OMNI_CODEX_BASE_URL"),
            }
        }
    }

    #[test]
    fn detected_accepts_openai_api_key() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new("", None);
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-openai-detected");
        }

        assert!(CodexProvider::detected());
    }

    #[test]
    fn omni_codex_override_feeds_models_and_aliases() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new("", None);
        unsafe {
            std::env::set_var("OMNI_CODEX_BASE_URL", "https://omni-codex.example.com");
            std::env::set_var("OMNI_CODEX_MODEL", "gpt-override");
        }

        let provider = CodexProvider::new().unwrap();
        let models = provider.models_list();
        assert_eq!(models[0].id, "gpt-override");
        let aliases = provider.model_aliases();
        // WHY: the legacy `codex` shorthand was pruned; only `gpt` maps to the
        // configured model now. Asserting `codex` is absent locks in the removal
        // so a re-added alias can't silently reintroduce the retired shorthand.
        assert!(aliases.contains(&("gpt".into(), "gpt-override".into())));
        assert!(!aliases.iter().any(|(a, _)| a == "codex"));
    }

    #[test]
    fn active_catalog_advertises_verified_models_alongside_configured() {
        // WHY: /v1/models must surface the verified single-pin catalog (gpt-6
        // astra/sol/luna, the 5.6 family, and still-listed 5.5) AND the
        // actually-configured model.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new("", None);
        // Clear any OMNI override from a prior test in the shared process.
        unsafe {
            std::env::remove_var("OMNI_CODEX_BASE_URL");
            std::env::remove_var("OMNI_CODEX_MODEL");
        }
        let provider = CodexProvider::new().unwrap();
        let ids: Vec<String> = provider.models_list().into_iter().map(|m| m.id).collect();
        assert!(ids.iter().any(|id| id == "gpt-6-astra"), "ids: {ids:?}");
        assert!(ids.iter().any(|id| id == "gpt-6-sol"), "ids: {ids:?}");
        assert!(ids.iter().any(|id| id == "gpt-6-luna"), "ids: {ids:?}");
        assert!(ids.iter().any(|id| id == "gpt-5.6-sol"), "ids: {ids:?}");
        assert!(ids.iter().any(|id| id == "gpt-5.6-terra"), "ids: {ids:?}");
        assert!(ids.iter().any(|id| id == "gpt-5.6-luna"), "ids: {ids:?}");
        assert!(ids.iter().any(|id| id == "gpt-5.5"), "ids: {ids:?}");
        assert!(!ids.iter().any(|id| id == "gpt-5.2"), "ids: {ids:?}");
        assert!(
            !ids.iter()
                .any(|id| id == "gpt-5.4-mini" || id == "codex-auto-review"),
            "hidden/internal slugs must stay out of the caller catalog: {ids:?}"
        );
    }

    #[test]
    fn active_pin_is_single_catalog_version() {
        // WHY: issue #12 ships one pin only. Catalog and UA version must stay
        // locked to the verified Codex CLI release so wire headers cannot drift.
        // Hold ENV_LOCK so models_list() (reads CODEX_HOME) does not race other
        // tests that mutate env.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new("", None);
        assert_eq!(CodexProvider::pinned_version(), "0.156.1");
        let provider = CodexProvider::new().unwrap();
        assert_eq!(provider.version, "0.156.1");
        let ids: Vec<_> = provider.active_catalog().iter().map(|m| m.id).collect();
        assert_eq!(
            ids,
            [
                "gpt-6-astra",
                "gpt-6-sol",
                "gpt-6-luna",
                "gpt-5.6-sol",
                "gpt-5.6-terra",
                "gpt-5.6-luna",
                "gpt-5.5"
            ]
        );
    }

    #[test]
    fn capitalized_custom_provider_id_matches_current_codex_config_style() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-5.5"
model_provider = "OpenAI"
[model_providers.OpenAI]
base_url = "https://share-ai.example.com"
wire_api = "responses"
requires_openai_auth = false
"#,
            None,
        );
        let cfg = CodexRequestConfig::load().unwrap();
        assert_eq!(cfg.base_url, "https://share-ai.example.com");
        assert!(!cfg.requires_openai_auth);
    }

    #[test]
    fn custom_provider_env_key_overrides_openai_auth() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com/v1"
wire_api = "responses"
env_key = "CUSTOM_CODEX_KEY"
"#,
            Some(r#"{"OPENAI_API_KEY":"sk-auth-json"}"#),
        );
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-openai-env");
            std::env::set_var("CUSTOM_CODEX_KEY", "custom-token");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer custom-token"
        );
    }

    #[test]
    fn requires_openai_auth_uses_openai_auth_and_ignores_env_key() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com/v1"
wire_api = "responses"
requires_openai_auth = true
env_key = "CUSTOM_CODEX_KEY"
"#,
            Some(r#"{"OPENAI_API_KEY":"sk-auth-json"}"#),
        );
        unsafe {
            std::env::set_var("CUSTOM_CODEX_KEY", "custom-token");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer sk-auth-json",
            "requires_openai_auth must use Codex/OpenAI auth instead of custom env_key"
        );
    }

    #[test]
    fn auth_command_overrides_env_key_and_static_token() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com/v1"
wire_api = "responses"
env_key = "CUSTOM_CODEX_KEY"
experimental_bearer_token = "static-token"
[model_providers.proxy.auth]
command = "/bin/sh"
args = ["-c", "printf cmd-token"]
timeout_ms = 1000
"#,
            None,
        );
        unsafe {
            std::env::set_var("CUSTOM_CODEX_KEY", "env-token");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer cmd-token",
            "command-backed auth owns Authorization when configured"
        );
    }

    #[test]
    fn auth_command_empty_stdout_is_an_auth_error() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com/v1"
wire_api = "responses"
[model_providers.proxy.auth]
command = "/bin/sh"
args = ["-c", "true"]
timeout_ms = 1000
"#,
            None,
        );

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt.block_on(cfg.headers()).unwrap_err();
        assert!(
            err.to_string().contains("empty token"),
            "empty auth command output should fail loudly: {err}"
        );
    }

    #[test]
    fn custom_headers_and_env_headers_are_applied() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // No auth.json and no env credential: isolates the header behavior from the issue #1
        // auth.json fallback so this test asserts only that static + env headers are applied.
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com/v1"
wire_api = "responses"
requires_openai_auth = false
http_headers = { "X-Static" = "static-value" }
env_http_headers = { "X-Dynamic" = "CUSTOM_CODEX_HEADER" }
"#,
            None,
        );
        unsafe {
            std::env::remove_var("CODEX_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("CODEX_ACCESS_TOKEN");
            std::env::set_var("CUSTOM_CODEX_HEADER", "dynamic-value");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(headers.get("x-static").unwrap(), "static-value");
        assert_eq!(headers.get("x-dynamic").unwrap(), "dynamic-value");
    }

    #[test]
    fn builtin_openai_uses_openai_auth_json() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-5.5"
"#,
            Some(r#"{"OPENAI_API_KEY":"sk-from-auth-json"}"#),
        );

        let cfg = CodexRequestConfig::load().unwrap();
        assert!(cfg.requires_openai_auth);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(
            headers.get(header::AUTHORIZATION).unwrap(),
            "Bearer sk-from-auth-json"
        );
    }

    #[test]
    fn url_join_does_not_duplicate_v1() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com/v1"
wire_api = "responses"
query_params = { api-version = "2026-01-01" }
"#,
            None,
        );
        let cfg = CodexRequestConfig::load().unwrap();
        assert_eq!(
            cfg.responses_url().unwrap().as_str(),
            "https://proxy.example.com/v1/responses?api-version=2026-01-01"
        );
    }

    #[test]
    fn canonical_maps_to_responses_body_with_tools_reasoning_and_extras() {
        let req = CanonicalRequest {
            model: "gpt-5.5".into(),
            messages: vec![
                CanonicalMessage {
                    role: "system".into(),
                    content: CanonicalContent::Text("sys".into()),
                },
                CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                },
                CanonicalMessage {
                    role: "assistant".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolUse {
                        id: "call_1".into(),
                        name: "lookup".into(),
                        arguments: "{\"q\":\"x\"}".into(),
                        cache: None,
                    }]),
                },
            ],
            tools: Some(vec![CanonicalTool {
                name: "lookup".into(),
                description: Some("Lookup".into()),
                parameters: json!({"type":"object"}),
                strict: false,
                cache: None,
            }]),
            tool_choice: Some(CanonicalToolChoice::Specific {
                name: "lookup".into(),
            }),
            max_tokens: Some(50),
            temperature: Some(0.2),
            top_p: None,
            reasoning: Some(CanonicalReasoning {
                effort: Some("high".into()),
                budget_tokens: None,
            }),
            metadata: Default::default(),
            cache: Some(omni_core::CanonicalCacheIntent {
                routing_identity: Some("sess-1".into()),
                ..Default::default()
            }),
            provider_extras: Some(json!({
                "store": false,
                "previous_response_id": "resp_1",
                "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}}},
                "text": {"format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}}}
            })),
        };
        let body = codex_responses_body(&req, false).unwrap();
        assert_eq!(body["instructions"], "sys");
        assert_eq!(body["input"].as_array().unwrap().len(), 2);
        assert_eq!(body["tools"][0]["name"], "lookup");
        assert_eq!(body["tool_choice"]["name"], "lookup");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["prompt_cache_key"], "sess-1");
        assert_eq!(body["store"], false);
        assert_eq!(body["previous_response_id"], "resp_1");
        assert!(
            body.get("response_format").is_none(),
            "chat response_format must not be copied onto Responses: {body}"
        );
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["name"], "out");
        assert_eq!(body["text"]["format"]["schema"], json!({"type": "object"}));
    }

    #[test]
    fn anthropic_1h_clamps_to_codex_gpt56_30m() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "messages": [{"role": "user", "content": "u"}]
        });
        let mut canon = omni_common::anthropic_to_canonical(&body, "codex").unwrap();
        canon.model = "gpt-5.6-sol".into();
        let rest = codex_responses_body(&canon, false).unwrap();
        let ws = codex_response_create_body(&canon).unwrap();
        assert_eq!(rest["prompt_cache_options"]["ttl"], "30m");
        assert_eq!(ws["prompt_cache_options"]["ttl"], "30m");
        assert_eq!(
            rest.get("prompt_cache_options"),
            ws.get("prompt_cache_options")
        );
    }

    #[test]
    fn anthropic_5m_omits_codex_gpt56_ttl() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "messages": [{"role": "user", "content": "u"}]
        });
        let mut canon = omni_common::anthropic_to_canonical(&body, "codex").unwrap();
        canon.model = "gpt-5.6-sol".into();
        let rest = codex_responses_body(&canon, false).unwrap();
        assert!(
            rest.get("prompt_cache_options")
                .and_then(|v| v.get("ttl"))
                .is_none(),
            "5m has no Codex GPT-5.6+ value ≤ 5m: {rest}"
        );
        let ws = codex_response_create_body(&canon).unwrap();
        assert_eq!(
            rest.get("prompt_cache_options"),
            ws.get("prompt_cache_options")
        );
    }

    #[test]
    fn cache_control_breakpoint_becomes_codex_prompt_cache_breakpoint() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "suffix"}
            ]}]
        });
        let mut canon = omni_common::anthropic_to_canonical(&body, "codex").unwrap();
        canon.model = "gpt-5.6-sol".into();
        let rest = codex_responses_body(&canon, false).unwrap();
        let parts = rest["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["prompt_cache_breakpoint"]["mode"], "explicit");
        assert!(parts[1].get("prompt_cache_breakpoint").is_none());
        let ws = codex_response_create_body(&canon).unwrap();
        assert_eq!(rest["input"], ws["input"]);
    }

    #[test]
    fn responses_developer_breakpoint_survives_on_codex_wire() {
        let req: omni_common::ResponsesRequest = serde_json::from_str(
            r#"{"model":"gpt-5.6-sol","input":[
                {"role":"developer","content":[
                    {"type":"input_text","text":"sys","prompt_cache_breakpoint":{"mode":"explicit"}}
                ]},
                {"role":"user","content":"hi"}
            ]}"#,
        )
        .unwrap();
        let canon = omni_common::responses_to_canonical(&req).unwrap();
        let rest = codex_responses_body(&canon, false).unwrap();
        assert!(
            rest.get("instructions").is_none(),
            "marked developer content must not collapse into instructions: {rest}"
        );
        let input = rest["input"].as_array().unwrap();
        assert_eq!(input[0]["role"], "developer");
        let parts = input[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "sys");
        assert_eq!(parts[0]["prompt_cache_breakpoint"]["mode"], "explicit");
        let ws = codex_response_create_body(&canon).unwrap();
        assert_eq!(rest["input"], ws["input"]);
        assert_eq!(rest.get("instructions"), ws.get("instructions"));
    }

    #[test]
    fn anthropic_marked_system_survives_on_codex_wire() {
        let body = json!({
            "model": "m",
            "max_tokens": 10,
            "system": [{
                "type": "text",
                "text": "stable prefix",
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{"role": "user", "content": "u"}]
        });
        let mut canon = omni_common::anthropic_to_canonical(&body, "codex").unwrap();
        canon.model = "gpt-5.6-sol".into();
        let rest = codex_responses_body(&canon, false).unwrap();
        assert!(
            rest.get("instructions").is_none(),
            "marked Anthropic system must not collapse into instructions: {rest}"
        );
        let input = rest["input"].as_array().unwrap();
        assert_eq!(input[0]["role"], "developer");
        let parts = input[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "stable prefix");
        assert_eq!(parts[0]["prompt_cache_breakpoint"]["mode"], "explicit");
        let ws = codex_response_create_body(&canon).unwrap();
        assert_eq!(rest["input"], ws["input"]);
    }

    #[test]
    fn conservative_headers_match_captured_codex_0142_names_and_values() {
        let auth = ChatGptAuth {
            access_token: "eyJ-fake-oauth".into(),
            account_id: "11111111-2222-3333-4444-555555555555".into(),
        };
        let headers = conservative_codex_headers("0.156.1", &auth).unwrap();
        assert_eq!(headers.get("version").unwrap(), "0.156.1");
        assert_eq!(
            headers.get("authorization").unwrap(),
            "Bearer eyJ-fake-oauth"
        );
        assert_eq!(
            headers.get("chatgpt-account-id").unwrap(),
            "11111111-2222-3333-4444-555555555555"
        );
        assert_eq!(headers.get("accept").unwrap(), "*/*");
        assert_eq!(headers.get("originator").unwrap(), "codex_exec");
        assert_eq!(
            headers.get("user-agent").unwrap(),
            "codex_exec/0.156.1 (Ubuntu 24.4.0; x86_64) unknown (codex_exec; 0.156.1)"
        );

        let request = conservative_ws_request(
            "ws://127.0.0.1/backend-api/codex/responses",
            "0.156.1",
            &auth,
        )
        .unwrap();
        let ws_headers = request.headers();
        assert_eq!(request.uri().path(), "/backend-api/codex/responses");
        assert_eq!(
            ws_headers.get("openai-beta").unwrap(),
            CONSERVATIVE_OPENAI_BETA
        );
        assert_eq!(
            ws_headers.get("x-codex-beta-features").unwrap(),
            CONSERVATIVE_BETA_FEATURES
        );
        assert_eq!(ws_headers.get("originator").unwrap(), "codex_exec");
        assert_eq!(ws_headers.get("version").unwrap(), "0.156.1");
        assert_eq!(
            ws_headers.get("x-client-request-id").unwrap(),
            CONSERVATIVE_CLIENT_REQUEST_ID
        );
        assert_eq!(
            ws_headers.get("session-id").unwrap(),
            CONSERVATIVE_CLIENT_REQUEST_ID
        );
        assert_eq!(
            ws_headers.get("thread-id").unwrap(),
            CONSERVATIVE_CLIENT_REQUEST_ID
        );
        let metadata: Value = serde_json::from_str(
            ws_headers
                .get("x-codex-turn-metadata")
                .unwrap()
                .to_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["request_kind"], "prewarm");
        assert_eq!(metadata["thread_source"], "user");
        assert_eq!(metadata["workspaces"], json!({}));
    }

    #[test]
    fn conservative_response_create_wraps_existing_codex_body_shape() {
        let req = CanonicalRequest {
            model: "gpt-5.5".into(),
            messages: vec![
                CanonicalMessage {
                    role: "system".into(),
                    content: CanonicalContent::Text("sys".into()),
                },
                CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                },
            ],
            max_tokens: Some(256),
            reasoning: Some(CanonicalReasoning {
                effort: Some("high".into()),
                budget_tokens: None,
            }),
            provider_extras: Some(json!({"store": false, "metadata": {"source": "test"}})),
            ..Default::default()
        };
        let body = codex_response_create_body(&req).unwrap();
        assert_eq!(body["type"], "response.create");
        assert_eq!(body["model"], "gpt-5.5");
        assert_eq!(body["instructions"], "sys");
        assert_eq!(body["input"][0]["content"], "hi");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["store"], false);
        assert!(
            body.get("stream").is_none(),
            "WebSocket response.create frames are text messages, not REST stream:true bodies"
        );
        assert!(
            body.get("max_output_tokens").is_none(),
            "ChatGPT codex WS rejects max_output_tokens; must not appear on response.create: {body}"
        );
        // REST body still carries the cap for OpenAI api.openai.com Responses.
        let rest = codex_responses_body(&req, false).unwrap();
        assert_eq!(rest["max_output_tokens"], 256);
    }

    #[test]
    fn codex_unmappable_effort_fails_loud_not_silent_omit() {
        // WHY (issue #20): unmappable explicit client efforts must not be
        // silently dropped from the Responses body; fail with structured error.
        // OpenAI/Codex wire includes xhigh; ultra is the unmappable example.
        let base = CanonicalRequest {
            model: "gpt-5.5".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };
        for bad in ["ultra", "extreme"] {
            let mut req = base.clone();
            req.reasoning = Some(CanonicalReasoning {
                effort: Some(bad.into()),
                budget_tokens: None,
            });
            let err =
                codex_responses_body(&req, false).expect_err("unmappable effort must fail loud");
            match err {
                ProviderError::BadRequest(msg) => {
                    assert!(
                        msg.contains("unsupported reasoning_effort")
                            && msg.contains("provider=codex")
                            && msg.contains(bad)
                            && msg.contains("supported=["),
                        "structured unsupported-effort error: {msg}"
                    );
                }
                other => panic!("expected BadRequest, got {other:?}"),
            }
        }
        // Documented alias max→high still maps; xhigh is first-class OpenAI.
        let mut req = base.clone();
        req.reasoning = Some(CanonicalReasoning {
            effort: Some("max".into()),
            budget_tokens: None,
        });
        let body = codex_responses_body(&req, false).unwrap();
        assert_eq!(body["reasoning"]["effort"], "high");
        req.reasoning = Some(CanonicalReasoning {
            effort: Some("xhigh".into()),
            budget_tokens: None,
        });
        let body = codex_responses_body(&req, false).unwrap();
        assert_eq!(body["reasoning"]["effort"], "xhigh");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn conservative_auth_requires_chatgpt_oauth_token_and_account_id() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"model = "gpt-5.5""#,
            Some(
                r#"{"OPENAI_API_KEY":"sk-rest","tokens":{"access_token":"eyJ-oauth","account_id":"acct-1"}}"#,
            ),
        );
        let cfg = CodexRequestConfig::load().unwrap();
        let auth = cfg.chatgpt_auth().await.unwrap();
        assert_eq!(auth.access_token, "eyJ-oauth");
        assert_eq!(auth.account_id, "acct-1");
    }

    #[test]
    fn custom_codex_config_is_not_conservative_ws_eligible() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com/v1"
wire_api = "responses"
requires_openai_auth = false
"#,
            None,
        );
        let cfg = CodexRequestConfig::load().unwrap();
        assert!(
            !cfg.conservative_ws_eligible(),
            "custom provider auth/base_url must stay on the configured REST path"
        );
    }

    #[allow(clippy::result_large_err)]
    fn assert_codex_ws_request(
        req: &http::Request<()>,
        response: http::Response<()>,
    ) -> Result<http::Response<()>, http::Response<Option<String>>> {
        let headers = req.headers();
        assert_eq!(req.uri().path(), "/backend-api/codex/responses");
        assert_eq!(
            headers.get("authorization").unwrap().to_str().unwrap(),
            "Bearer eyJ-test-oauth"
        );
        assert_eq!(
            headers.get("chatgpt-account-id").unwrap().to_str().unwrap(),
            "acct-test"
        );
        assert_eq!(
            headers.get("user-agent").unwrap().to_str().unwrap(),
            "codex_exec/0.156.1 (Ubuntu 24.4.0; x86_64) unknown (codex_exec; 0.156.1)"
        );
        assert_eq!(headers.get("originator").unwrap(), "codex_exec");
        assert_eq!(
            headers.get("openai-beta").unwrap(),
            CONSERVATIVE_OPENAI_BETA
        );
        assert_eq!(headers.get("version").unwrap(), "0.156.1");
        assert_eq!(
            headers.get("x-codex-beta-features").unwrap(),
            CONSERVATIVE_BETA_FEATURES
        );
        Ok(response)
    }

    async fn spawn_codex_ws_server() -> (String, oneshot::Receiver<Value>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = accept_hdr_async(stream, assert_codex_ws_request)
                .await
                .unwrap();
            let frame = ws.next().await.unwrap().unwrap();
            let body: Value = serde_json::from_str(frame.to_text().unwrap()).unwrap();
            tx.send(body).unwrap();
            ws.send(Message::Text(
                json!({
                    "type": "codex.rate_limits",
                    "rate_limits": {"allowed": true}
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                json!({
                    "type": "response.created",
                    "response": {
                        "id": "resp_ws",
                        "status": "in_progress",
                        "model": "gpt-5.5"
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                json!({
                    "type": "response.output_text.delta",
                    "delta": "Hel",
                    "output_index": 0,
                    "content_index": 0
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                json!({
                    "type": "response.output_text.delta",
                    "delta": "lo",
                    "output_index": 0,
                    "content_index": 0
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                json!({
                    "type": "response.completed",
                    "response": {
                        "id": "resp_ws",
                        "status": "completed",
                        "model": "gpt-5.5",
                        "usage": {"input_tokens": 3, "output_tokens": 4}
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        });
        (format!("http://{addr}"), rx)
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn conservative_send_uses_models_preflight_ws_and_shared_parser() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ws_base_url, body_rx) = spawn_codex_ws_server().await;
        let _home = TempCodexHome::new(
            r#"
model = "gpt-5.5"
"#,
            Some(
                r#"{"OPENAI_API_KEY":"sk-rest","tokens":{"access_token":"eyJ-test-oauth","account_id":"acct-test"}}"#,
            ),
        );
        let models_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/backend-api/codex/models"))
            .and(header("authorization", "Bearer eyJ-test-oauth"))
            .and(header("chatgpt-account-id", "acct-test"))
            .and(header("originator", "codex_exec"))
            .and(header("version", "0.156.1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "models": [{"slug": "gpt-5.5", "prefer_websockets": true}]
            })))
            .expect(1)
            .mount(&models_server)
            .await;
        unsafe {
            std::env::set_var(
                "OMNI_CODEX_CONSERVATIVE_BASE_URL_FOR_TEST",
                models_server.uri(),
            );
            std::env::set_var("OMNI_CODEX_CONSERVATIVE_WS_BASE_URL_FOR_TEST", &ws_base_url);
        }

        // No mode flag: ChatGPT OAuth tokens + default OpenAI-shaped config
        // auto-select the CLI-parity WebSocket path.
        let provider = CodexProvider::new().unwrap();
        let resp = provider
            .send(CanonicalRequest {
                model: "gpt-5.5".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(resp.content, "Hello");
        assert_eq!(resp.id.as_deref(), Some("resp_ws"));
        assert_eq!(resp.usage.input_tokens, 3);
        assert_eq!(resp.usage.output_tokens, 4);
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            resp.metadata
                .as_ref()
                .and_then(|meta| meta.provider.as_deref()),
            Some("codex")
        );

        let body = body_rx.await.unwrap();
        assert_eq!(body["type"], "response.create");
        assert_eq!(body["model"], "gpt-5.5");
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"], "hi");
        assert!(body.get("stream").is_none());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn conservative_ws_handshake_http_503_is_send_stream_err() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let body = b"overloaded";
            let resp = format!(
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.write_all(body).await;
        });
        let _home = TempCodexHome::new(
            r#"
model = "gpt-5.5"
"#,
            Some(
                r#"{"OPENAI_API_KEY":"sk-rest","tokens":{"access_token":"eyJ-test-oauth","account_id":"acct-test"}}"#,
            ),
        );
        let models_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/backend-api/codex/models"))
            .respond_with(ResponseTemplate::new(200).set_body_string("unused"))
            .expect(1)
            .mount(&models_server)
            .await;
        let ws_base = format!("http://{addr}");
        let _env = EnvRestore::set(&[
            (
                "OMNI_CODEX_CONSERVATIVE_BASE_URL_FOR_TEST",
                Some(models_server.uri()),
            ),
            (
                "OMNI_CODEX_CONSERVATIVE_WS_BASE_URL_FOR_TEST",
                Some(ws_base),
            ),
        ]);
        let provider = CodexProvider::new().unwrap();
        let err = match provider
            .send_stream(CanonicalRequest {
                model: "gpt-5.5".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
        {
            Err(err) => err,
            Ok(_) => panic!("handshake 503 is Err from send_stream"),
        };
        assert!(
            matches!(
                err,
                ProviderError::Upstream {
                    status: Some(503),
                    ..
                }
            ),
            "expected upstream 503, got {err:?}"
        );
    }

    struct EnvRestore {
        old: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl EnvRestore {
        fn set(vars: &[(&'static str, Option<String>)]) -> Self {
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

    impl Drop for EnvRestore {
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

    #[test]
    fn should_use_chatgpt_ws_when_oauth_tokens_and_default_config() {
        // WHY: ChatGPT login must not require an operator mode flag; path follows
        // credentials + config like Claude/Grok.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"model = "gpt-5.5""#,
            Some(r#"{"tokens":{"access_token":"eyJ-oauth","account_id":"acct-1"}}"#),
        );
        let cfg = CodexRequestConfig::load().unwrap();
        assert!(cfg.should_use_chatgpt_ws());
        let summary = CodexProvider::startup_source_summary().unwrap();
        assert!(
            summary.contains("path=chatgpt-ws") && summary.contains("chatgpt-oauth"),
            "startup summary must name the resolved chatgpt path: {summary}"
        );
        assert!(
            summary.contains("home=") && summary.contains("model=gpt-5.5"),
            "startup summary must include home and model: {summary}"
        );
    }

    #[test]
    fn should_not_use_chatgpt_ws_for_api_key_only_auth() {
        // WHY: platform sk- auth must stay on REST api.openai.com, not ChatGPT WS.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"model = "gpt-5.5""#,
            Some(r#"{"OPENAI_API_KEY":"sk-rest-only"}"#),
        );
        let cfg = CodexRequestConfig::load().unwrap();
        assert!(cfg.conservative_ws_eligible());
        assert!(!cfg.should_use_chatgpt_ws());
    }

    #[test]
    fn should_not_use_chatgpt_ws_for_custom_gateway() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com/v1"
wire_api = "responses"
requires_openai_auth = false
"#,
            Some(r#"{"tokens":{"access_token":"eyJ-oauth","account_id":"acct-1"}}"#),
        );
        let cfg = CodexRequestConfig::load().unwrap();
        assert!(!cfg.should_use_chatgpt_ws());
    }

    #[test]
    fn canonical_image_blocks_map_to_responses_input_image_parts() {
        // WHY: Codex/OpenAI Responses accepts image input as typed content
        // parts; base64 canonical sources must reconstruct a data URL.
        let req = CanonicalRequest {
            model: "gpt-5.5".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Blocks(vec![
                    CanonicalBlock::Text {
                        text: "look".into(),
                        cache: None,
                    },
                    CanonicalBlock::Image {
                        source: omni_core::CanonicalImageSource::Url {
                            url: "https://example.com/a.png".into(),
                        },
                        cache: None,
                    },
                    CanonicalBlock::Image {
                        source: omni_core::CanonicalImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: "abcd".into(),
                        },
                        cache: None,
                    },
                ]),
            }],
            ..Default::default()
        };
        let body = codex_responses_body(&req, false).unwrap();
        let content = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "look");
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "https://example.com/a.png");
        assert_eq!(content[2]["image_url"], "data:image/png;base64,abcd");
    }

    #[test]
    fn canonical_files_map_to_responses_with_breakpoint() {
        let req = CanonicalRequest {
            model: "gpt-5.5".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Blocks(vec![
                    CanonicalBlock::Text {
                        text: "read".into(),
                        cache: None,
                    },
                    CanonicalBlock::File {
                        source: omni_core::CanonicalFileSource::Url {
                            file_url: "https://example.com/a.pdf".into(),
                        },
                        filename: None,
                        detail: None,
                        cache: Some(omni_core::CanonicalCacheMark::breakpoint()),
                    },
                    CanonicalBlock::File {
                        source: omni_core::CanonicalFileSource::Data {
                            file_data: "data:application/pdf;base64,abcd".into(),
                        },
                        filename: Some("b.pdf".into()),
                        detail: Some("high".into()),
                        cache: None,
                    },
                ]),
            }],
            ..Default::default()
        };
        let body = codex_responses_body(&req, false).unwrap();
        let parts = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0], json!({ "type": "input_text", "text": "read" }));
        assert_eq!(
            parts[1],
            json!({ "type": "input_file", "file_url": "https://example.com/a.pdf", "prompt_cache_breakpoint": {"mode": "explicit"} })
        );
        assert_eq!(
            parts[2],
            json!({ "type": "input_file", "file_data": "data:application/pdf;base64,abcd", "filename": "b.pdf", "detail": "high" })
        );
    }

    fn chat_req_with_extras(extras: Value) -> CanonicalRequest {
        CanonicalRequest {
            model: "gpt-5.5".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            provider_extras: Some(extras),
            ..Default::default()
        }
    }

    #[test]
    fn response_format_text_maps_to_text_format() {
        // WHY (issue #32): Chat type=text maps 1:1 onto Responses text.format.
        let body = codex_responses_body(
            &chat_req_with_extras(json!({
                "response_format": {"type": "text"}
            })),
            false,
        )
        .unwrap();
        assert_eq!(body["text"]["format"]["type"], "text");
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn response_format_json_object_maps_to_text_format() {
        // WHY (issue #32): Chat json_object must become Responses text.format
        // and must not appear as a top-level Responses key.
        let body = codex_responses_body(
            &chat_req_with_extras(json!({
                "response_format": {"type": "json_object"}
            })),
            false,
        )
        .unwrap();
        assert_eq!(body["text"]["format"]["type"], "json_object");
        assert!(
            body.get("response_format").is_none(),
            "response_format must not remain on the Responses body: {body}"
        );
    }

    #[test]
    fn response_format_json_schema_unwraps_to_flat_text_format() {
        // WHY (issue #32): Chat nests name/schema/description/strict under
        // json_schema; Responses text.format is flat.
        let body = codex_responses_body(
            &chat_req_with_extras(json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "out",
                        "schema": {"type": "object", "properties": {"n": {"type": "integer"}}},
                        "description": "an answer",
                        "strict": true
                    }
                }
            })),
            false,
        )
        .unwrap();
        let format = &body["text"]["format"];
        assert_eq!(format["type"], "json_schema");
        assert_eq!(format["name"], "out");
        assert_eq!(
            format["schema"],
            json!({"type": "object", "properties": {"n": {"type": "integer"}}})
        );
        assert_eq!(format["description"], "an answer");
        assert_eq!(format["strict"], true);
        assert!(
            format.get("json_schema").is_none(),
            "Responses format must be flat, not nested under json_schema: {format}"
        );
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn response_format_null_optional_schema_fields_are_omitted() {
        // WHY (issue #32): Chat/Responses allow description/strict to be null;
        // treat null as absent instead of a named 400.
        let body = codex_responses_body(
            &chat_req_with_extras(json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "out",
                        "schema": {"type": "object"},
                        "description": null,
                        "strict": null
                    }
                }
            })),
            false,
        )
        .unwrap();
        let format = &body["text"]["format"];
        assert_eq!(format["type"], "json_schema");
        assert_eq!(format["name"], "out");
        assert!(format.get("description").is_none());
        assert!(format.get("strict").is_none());
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn response_format_null_is_treated_as_absent() {
        // WHY (issue #32): JSON null means the extra is absent. Do not invent
        // text.format and do not copy the Chat key onto Responses.
        let body = codex_responses_body(
            &chat_req_with_extras(json!({
                "store": false,
                "response_format": null
            })),
            false,
        )
        .unwrap();
        assert_eq!(body["store"], false);
        assert!(body.get("response_format").is_none());
        assert!(body.get("text").is_none());
    }

    #[test]
    fn inbound_text_only_is_unchanged() {
        // WHY (issue #32): Responses clients already send text; mapping must
        // not invent or rewrite format when response_format is absent.
        let text = json!({"format": {"type": "json_object"}, "verbosity": "low"});
        let body =
            codex_responses_body(&chat_req_with_extras(json!({ "text": text })), false).unwrap();
        assert_eq!(body["text"], text);
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn response_format_merges_with_text_verbosity() {
        // WHY (issue #32): existing text siblings such as verbosity stay when
        // response_format supplies the missing format.
        let body = codex_responses_body(
            &chat_req_with_extras(json!({
                "response_format": {"type": "json_object"},
                "text": {"verbosity": "high"}
            })),
            false,
        )
        .unwrap();
        assert_eq!(body["text"]["verbosity"], "high");
        assert_eq!(body["text"]["format"]["type"], "json_object");
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn equivalent_response_format_and_text_format_are_accepted() {
        // WHY (issue #32): dual fields with the same structured-output intent
        // must succeed; only the Responses name remains.
        let body = codex_responses_body(
            &chat_req_with_extras(json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "out", "schema": {"type": "object"}}
                },
                "text": {"format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}}}
            })),
            false,
        )
        .unwrap();
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["name"], "out");
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn conflicting_response_format_and_text_format_fails_loud() {
        // WHY (issue #32): conflicting structured-output fields must not pick
        // a winner; named HTTP 400.
        let err = codex_responses_body(
            &chat_req_with_extras(json!({
                "response_format": {"type": "json_object"},
                "text": {"format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}}}
            })),
            false,
        )
        .expect_err("conflicting structured output must fail loud");
        match err {
            ProviderError::BadRequest(msg) => {
                assert!(
                    msg.contains("response_format") && msg.contains("text.format"),
                    "conflict error must name both fields: {msg}"
                );
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn malformed_response_format_fails_loud() {
        // WHY (issue #32): unmappable or malformed Chat shapes must 400 with
        // a named error instead of copying the Chat key upstream.
        let cases = [
            json!("json_object"),
            json!({"type": "json_schema"}),
            json!({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}),
            json!({"type": "json_schema", "json_schema": {"name": 1, "schema": {"type": "object"}}}),
            json!({"type": "json_object", "extra": true}),
            json!({"type": "xml"}),
        ];
        for value in cases {
            let err = codex_responses_body(
                &chat_req_with_extras(json!({ "response_format": value })),
                false,
            )
            .expect_err("malformed response_format must fail loud");
            match err {
                ProviderError::BadRequest(msg) => {
                    assert!(
                        msg.contains("response_format"),
                        "malformed error must name response_format: {msg}"
                    );
                }
                other => panic!("expected BadRequest, got {other:?}"),
            }
        }
    }

    #[test]
    fn canonical_rejects_unsupported_responses_extras() {
        // WHY: provider extras that Codex cannot forward must fail loudly
        // instead of being silently dropped and making client requests look
        // accepted when they were not honored.
        let req = CanonicalRequest {
            model: "gpt-5.5".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            provider_extras: Some(json!({
                "store": false,
                "not_allowed": "drop"
            })),
            ..Default::default()
        };
        let err =
            codex_responses_body(&req, false).expect_err("unsupported Codex extra must reject");
        assert!(
            err.to_string().contains("not_allowed"),
            "error must name the unsupported provider extra: {err}"
        );
    }

    #[test]
    fn responses_output_maps_to_canonical() {
        let value = json!({
            "id": "resp_backend",
            "model": "gpt-5.5",
            "service_tier": "default",
            "system_fingerprint": "fp_codex",
            "status": "completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"hello","annotations":[{"type":"url_citation","url":"https://e.test"}]}]},
                {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "input_tokens_details": {"cached_tokens": 1, "audio_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 6, "audio_tokens": 7}
            }
        });
        let resp = responses_upstream::response_to_canonical(
            &value,
            "fallback",
            "codex",
            &CodexErrorRedactor::default(),
        )
        .unwrap();
        assert_eq!(resp.id.as_deref(), Some("resp_backend"));
        assert_eq!(resp.model, "gpt-5.5");
        assert_eq!(resp.content, "hello");
        assert!(resp.refusal.is_none());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_1");
        assert_eq!(resp.tool_calls[0].name, "lookup");
        assert_eq!(resp.tool_calls[0].arguments, "{}");
        assert_eq!(resp.usage.input_tokens, 3);
        assert_eq!(resp.usage.cache_read, 1);
        assert_eq!(resp.usage.reasoning_tokens, 6);
        assert_eq!(resp.usage.audio_tokens, 12);
        assert_eq!(resp.usage.input_audio_tokens, 5);
        assert_eq!(resp.usage.output_audio_tokens, 7);
        assert_eq!(resp.annotations[0]["url"], "https://e.test");
        assert_eq!(
            resp.metadata
                .as_ref()
                .and_then(|meta| meta.service_tier.as_deref()),
            Some("default")
        );
        assert_eq!(
            resp.metadata
                .as_ref()
                .and_then(|meta| meta.system_fingerprint.as_deref()),
            Some("fp_codex")
        );
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn responses_output_maps_refusal_and_content_filter_to_canonical() {
        let value = json!({
            "model": "gpt-5.5",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [
                {"type":"message","content":[{"type":"refusal","refusal":"No thanks"}]}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let resp = responses_upstream::response_to_canonical(
            &value,
            "fallback",
            "codex",
            &CodexErrorRedactor::default(),
        )
        .unwrap();
        assert_eq!(resp.content, "");
        assert_eq!(resp.refusal.as_deref(), Some("No thanks"));
        assert_eq!(resp.finish_reason.as_deref(), Some("content_filter"));
    }

    #[test]
    fn upstream_errors_are_redacted() {
        let redacted = redact(r#"{"error":"bad sk-one sk-two xai-one xai-two eyJone eyJtwo"}"#);
        for secret in ["sk-one", "sk-two", "xai-one", "xai-two", "eyJone", "eyJtwo"] {
            assert!(
                !redacted.contains(secret),
                "redacted body leaked {secret}: {redacted}"
            );
        }
        assert!(redacted.contains("<redacted>"));
    }

    // Issue #1: a custom gateway with requires_openai_auth=false and a key only in auth.json must
    // send that key as Bearer, matching the real Codex CLI (which the gateway requires; otherwise 401).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_uses_custom_provider_auth_json_fallback_header() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            Some(r#"{"OPENAI_API_KEY":"sk-from-auth-json"}"#),
        );
        unsafe {
            std::env::remove_var("CODEX_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("CODEX_ACCESS_TOKEN");
        }
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "gpt-custom",
                "status": "completed",
                "output": [{"type":"message","content":[{"type":"output_text","text":"ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let resp = provider
            .send(CanonicalRequest {
                model: "gpt-custom".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(resp.content, "ok");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].headers.get("authorization").unwrap(),
            "Bearer sk-from-auth-json",
            "custom no-auth provider must fall back to auth.json's OPENAI_API_KEY"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_uses_custom_provider_env_auth_header() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
env_key = "CUSTOM_CODEX_KEY"
"#,
                server.uri()
            ),
            Some(r#"{"OPENAI_API_KEY":"sk-must-not-leak"}"#),
        );
        unsafe {
            std::env::set_var("CUSTOM_CODEX_KEY", "codex-custom-token");
        }
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer codex-custom-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "gpt-custom",
                "status": "completed",
                "output": [{"type":"message","content":[{"type":"output_text","text":"ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let resp = provider
            .send(CanonicalRequest {
                model: "gpt-custom".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(resp.content, "ok");
    }

    fn issue31_base_req() -> CanonicalRequest {
        CanonicalRequest {
            model: "gpt-5.5".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        }
    }

    fn issue31_oauth_auth(access: &str) -> String {
        format!(
            r#"{{"tokens":{{"access_token":"{access}","refresh_token":"rt-old","account_id":"acct-1"}}}}"#
        )
    }

    struct CodexOauthUrlGuard {
        flag: Option<std::ffi::OsString>,
        url: Option<std::ffi::OsString>,
    }

    impl CodexOauthUrlGuard {
        fn install(token_url: &str) -> Self {
            let flag = std::env::var_os("OMNI_OAUTH_REFRESH");
            let url = std::env::var_os("OMNI_CODEX_OAUTH_TOKEN_URL");
            unsafe {
                std::env::set_var("OMNI_OAUTH_REFRESH", "1");
                std::env::set_var("OMNI_CODEX_OAUTH_TOKEN_URL", token_url);
            }
            Self { flag, url }
        }
    }

    impl Drop for CodexOauthUrlGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.flag {
                    Some(v) => std::env::set_var("OMNI_OAUTH_REFRESH", v),
                    None => std::env::remove_var("OMNI_OAUTH_REFRESH"),
                }
                match &self.url {
                    Some(v) => std::env::set_var("OMNI_CODEX_OAUTH_TOKEN_URL", v),
                    None => std::env::remove_var("OMNI_CODEX_OAUTH_TOKEN_URL"),
                }
            }
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn default_path_401_force_refresh_replays_nonstream() {
        // WHY: issue #31. Default CLI-file Codex send must force-refresh once
        // on upstream 401 and replay the same inference request.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let token_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "codex-after-401",
                "refresh_token": "rt-after-401",
                "id_token": "id-after-401",
                "expires_in": 864000
            })))
            .expect(1)
            .mount(&token_server)
            .await;

        let infer = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer codex-stale"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {"message": "invalid token"}
            })))
            .expect(1)
            .mount(&infer)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer codex-after-401"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "gpt-5.5",
                "status": "completed",
                "output": [{"type":"message","content":[{"type":"output_text","text":"recovered"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&infer)
            .await;

        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-5.5"
openai_base_url = "{}"
"#,
                infer.uri()
            ),
            Some(&issue31_oauth_auth("codex-stale")),
        );
        let token_url = format!("{}/oauth/token", token_server.uri());
        let _oauth = CodexOauthUrlGuard::install(&token_url);
        let provider = CodexProvider::new().unwrap();
        let resp = provider
            .send(issue31_base_req())
            .await
            .expect("401 recovery must replay send");
        assert_eq!(resp.content, "recovered");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn default_path_401_force_refresh_replays_stream_open() {
        // WHY: issue #31. Stream-open 401 on the default CLI path must
        // force-refresh once and reopen. No mid-stream replay.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let token_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "codex-stream-after-401",
                "refresh_token": "rt-stream-after-401",
                "id_token": "id-stream-after-401",
                "expires_in": 864000
            })))
            .expect(1)
            .mount(&token_server)
            .await;

        let infer = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer codex-stream-stale"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": {"message": "invalid token"}
            })))
            .expect(1)
            .mount(&infer)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer codex-stream-after-401"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&infer)
            .await;

        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-5.5"
openai_base_url = "{}"
"#,
                infer.uri()
            ),
            Some(&issue31_oauth_auth("codex-stream-stale")),
        );
        let token_url = format!("{}/oauth/token", token_server.uri());
        let _oauth = CodexOauthUrlGuard::install(&token_url);
        let provider = CodexProvider::new().unwrap();
        let mut stream = provider
            .send_stream(issue31_base_req())
            .await
            .expect("stream opens");
        let mut saw_text = false;
        while let Some(event) = stream.next().await {
            let event = event.expect("401 recovery must reopen stream");
            if matches!(event, CanonicalStreamEvent::TextDelta(ref t) if t == "ok") {
                saw_text = true;
            }
        }
        assert!(saw_text, "expected recovered stream text");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn default_path_429_and_5xx_pass_through_once() {
        // WHY: issue #31. 429 and 5xx must pass through after one upstream
        // attempt. Codex must not treat them as credential recovery.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let token_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "should-not-refresh",
                "refresh_token": "should-not-rotate",
                "expires_in": 864000
            })))
            .expect(0)
            .mount(&token_server)
            .await;
        let token_url = format!("{}/oauth/token", token_server.uri());
        let _oauth = CodexOauthUrlGuard::install(&token_url);

        for status in [429_u16, 503] {
            let infer = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/responses"))
                .respond_with(ResponseTemplate::new(status).set_body_string("upstream busy"))
                .expect(1)
                .mount(&infer)
                .await;
            let _home = TempCodexHome::new(
                &format!(
                    r#"
model = "gpt-5.5"
openai_base_url = "{}"
"#,
                    infer.uri()
                ),
                Some(&issue31_oauth_auth("codex-live-no-retry")),
            );
            let provider = CodexProvider::new().unwrap();
            let err = provider
                .send(issue31_base_req())
                .await
                .expect_err("upstream status must pass through");
            assert!(
                matches!(
                    err,
                    ProviderError::Upstream {
                        status: Some(got),
                        ..
                    } if got == status
                ),
                "expected upstream {status}, got {err:?}"
            );
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn omni_override_401_does_not_replay_via_cli_file() {
        // WHY: issue #31. Custom/override endpoints must not 401-replay via
        // default CLI auth.json.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let token_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "must-not-use",
                "refresh_token": "must-not-rotate",
                "expires_in": 864000
            })))
            .expect(0)
            .mount(&token_server)
            .await;

        let infer = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(401).set_body_string("nope"))
            .expect(1)
            .mount(&infer)
            .await;

        let _home = TempCodexHome::new(
            r#"
model = "gpt-5.5"
"#,
            Some(&issue31_oauth_auth("codex-must-not-leak")),
        );
        unsafe {
            std::env::set_var("OMNI_CODEX_BASE_URL", infer.uri());
            std::env::set_var("OMNI_CODEX_AUTH_TOKEN", "omni-override-token");
        }
        let token_url = format!("{}/oauth/token", token_server.uri());
        let _oauth = CodexOauthUrlGuard::install(&token_url);
        let provider = CodexProvider::new().unwrap();
        let err = provider
            .send(issue31_base_req())
            .await
            .expect_err("override 401 must pass through");
        assert!(
            matches!(
                err,
                ProviderError::Upstream {
                    status: Some(401),
                    ..
                }
            ),
            "expected 401, got {err:?}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn env_api_key_401_does_not_rotate_leftover_oauth() {
        // WHY: issue #31 review. REST bearer from CODEX_API_KEY/OPENAI_API_KEY
        // must not force-refresh leftover ChatGPT OAuth in auth.json on 401.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let token_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": "must-not-rotate",
                "refresh_token": "must-not-rotate",
                "expires_in": 864000
            })))
            .expect(0)
            .mount(&token_server)
            .await;

        let infer = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer sk-env-owns-request"))
            .respond_with(ResponseTemplate::new(401).set_body_string("nope"))
            .expect(1)
            .mount(&infer)
            .await;

        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-5.5"
openai_base_url = "{}"
"#,
                infer.uri()
            ),
            Some(&issue31_oauth_auth("codex-leftover-oauth")),
        );
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-env-owns-request");
        }
        let token_url = format!("{}/oauth/token", token_server.uri());
        let _oauth = CodexOauthUrlGuard::install(&token_url);
        let provider = CodexProvider::new().unwrap();
        let err = provider
            .send(issue31_base_req())
            .await
            .expect_err("env-key 401 must pass through once");
        assert!(
            matches!(
                err,
                ProviderError::Upstream {
                    status: Some(401),
                    ..
                }
            ),
            "expected 401, got {err:?}"
        );
        let on_disk: Value =
            serde_json::from_slice(&std::fs::read(_home.path.join("auth.json")).unwrap()).unwrap();
        assert_eq!(
            on_disk["tokens"]["refresh_token"], "rt-old",
            "leftover OAuth RT must stay unrotated"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_maps_responses_text_usage_and_finish() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n\
event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":4}}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            vec![
                CanonicalStreamEvent::TextDelta("Hel".into()),
                CanonicalStreamEvent::TextDelta("lo".into()),
                CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: 3,
                    output_tokens: 4,
                    cache_read: 0,
                    cache_creation: 0,
                    ..Default::default()
                }),
                CanonicalStreamEvent::Finish {
                    finish_reason: Some("stop".into())
                }
            ]
        );
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(
            requests[0].headers.get("accept").unwrap(),
            "text/event-stream"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_uses_output_text_done_when_no_deltas() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_text.done\n\
data: {\"type\":\"response.output_text.done\",\"text\":\"complete\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            events[0],
            CanonicalStreamEvent::TextDelta("complete".into())
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_emits_output_text_done_suffix_without_duplicate() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n\
event: response.output_text.done\n\
data: {\"type\":\"response.output_text.done\",\"text\":\"Hello\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let text = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .filter_map(|event| match event {
                CanonicalStreamEvent::TextDelta(delta) => Some(delta),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "Hello");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_recovers_terminal_only_text_and_response_id() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_backend\",\"status\":\"completed\",\"service_tier\":\"default\",\"system_fingerprint\":\"fp_stream\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"terminal only\",\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://e.test\"}]}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            events[0],
            CanonicalStreamEvent::ResponseMetadata(CanonicalResponseMetadata {
                id: Some("resp_backend".into()),
                system_fingerprint: Some("fp_stream".into()),
                service_tier: Some("default".into()),
                provider: Some("codex".into()),
                ..Default::default()
            })
        );
        assert_eq!(
            events[1],
            CanonicalStreamEvent::TextDelta("terminal only".into())
        );
        assert_eq!(
            events[2],
            CanonicalStreamEvent::OutputAnnotations(vec![json!({
                "type": "url_citation",
                "url": "https://e.test",
            })])
        );
        assert_eq!(
            events.last().unwrap(),
            &CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into())
            }
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_recovers_terminal_only_refusal_and_tool_call() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.incomplete\n\
data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"},\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"No thanks\"}]},{\"type\":\"function_call\",\"call_id\":\"call_terminal\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"sf\\\"}\"}]}}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert!(events.contains(&CanonicalStreamEvent::RefusalDelta("No thanks".into())));
        assert!(events.contains(&CanonicalStreamEvent::ToolCallDelta {
            index: 0,
            id: Some("call_terminal".into()),
            name: Some("lookup".into()),
            arguments_delta: String::new(),
        }));
        assert!(events.contains(&CanonicalStreamEvent::ToolCallDelta {
            index: 0,
            id: None,
            name: None,
            arguments_delta: r#"{"q":"sf"}"#.into(),
        }));
        assert_eq!(
            events.last().unwrap(),
            &CanonicalStreamEvent::Finish {
                finish_reason: Some("content_filter".into())
            }
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_rejects_terminal_function_args_that_conflict_with_deltas() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_item.added\n\
data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n\
event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"q\\\":\\\"sf\\\"}\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"nyc\\\"}\"}]}}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider).await;
        let err = events[2].as_ref().unwrap_err().to_string();
        assert!(err.contains("terminal function_call arguments"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_rejects_output_item_done_args_that_conflict_with_deltas() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_item.added\n\
data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n\
event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"q\\\":\\\"sf\\\"}\"}\n\n\
event: response.output_item.done\n\
data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"nyc\\\"}\"}}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider).await;
        let err = events[2].as_ref().unwrap_err().to_string();
        assert!(err.contains("terminal function_call arguments"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_rejects_output_text_done_that_conflicts_with_deltas() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"A\"}\n\n\
event: response.output_text.done\n\
data: {\"type\":\"response.output_text.done\",\"text\":\"B\"}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider).await;
        let err = events[1].as_ref().unwrap_err().to_string();
        assert!(err.contains("did not extend prior text deltas"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_rejects_function_args_done_that_conflicts_with_deltas() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_item.added\n\
data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n\
event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"q\\\":\\\"sf\\\"}\"}\n\n\
event: response.function_call_arguments.done\n\
data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{\\\"q\\\":\\\"nyc\\\"}\"}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider).await;
        let err = events[2].as_ref().unwrap_err().to_string();
        assert!(
            err.contains("did not extend prior argument deltas"),
            "{err}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_maps_refusal_delta_and_done_to_text() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.refusal.delta\n\
data: {\"type\":\"response.refusal.delta\",\"delta\":\"No\"}\n\n\
event: response.refusal.done\n\
data: {\"type\":\"response.refusal.done\",\"refusal\":\"No thanks\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let text = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .filter_map(|event| match event {
                CanonicalStreamEvent::RefusalDelta(delta) => Some(delta),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "No thanks");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_tracks_text_and_refusal_done_independently() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Partial\"}\n\n\
event: response.refusal.delta\n\
data: {\"type\":\"response.refusal.delta\",\"output_index\":1,\"delta\":\"No\"}\n\n\
event: response.refusal.done\n\
data: {\"type\":\"response.refusal.done\",\"output_index\":1,\"refusal\":\"No thanks\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert!(events.contains(&CanonicalStreamEvent::TextDelta("Partial".into())));
        let refusal = events
            .iter()
            .filter_map(|event| match event {
                CanonicalStreamEvent::RefusalDelta(delta) => Some(delta.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(refusal, "No thanks");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_maps_responses_tool_call_deltas() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.output_item.added\n\
data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n\
event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"q\\\"\"}\n\n\
event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\":\\\"sf\\\"}\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":6}}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            vec![
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call_1".into()),
                    name: Some("lookup".into()),
                    arguments_delta: String::new(),
                },
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments_delta: "{\"q\"".into(),
                },
                CanonicalStreamEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments_delta: ":\"sf\"}".into(),
                },
                CanonicalStreamEvent::Usage(CanonicalUsage {
                    input_tokens: 5,
                    output_tokens: 6,
                    cache_read: 0,
                    cache_creation: 0,
                    ..Default::default()
                }),
                CanonicalStreamEvent::Finish {
                    finish_reason: Some("tool_calls".into())
                }
            ]
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_buffers_tool_arguments_until_metadata_arrives() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":3,\"delta\":\"{\\\"q\\\"\"}\n\n\
event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":3,\"delta\":\":\\\"sf\\\"}\"}\n\n\
event: response.output_item.added\n\
data: {\"type\":\"response.output_item.added\",\"output_index\":3,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_late\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            events[0],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_late".into()),
                name: Some("lookup".into()),
                arguments_delta: String::new(),
            }
        );
        assert_eq!(
            events[1],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "{\"q\":\"sf\"}".into(),
            }
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_does_not_duplicate_arguments_repeated_after_item_added() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.output_item.added\n\
data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{\\\"q\\\":\\\"sf\\\"}\"}}\n\n\
event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"q\\\":\\\"sf\\\"}\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let arguments = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .filter_map(|event| match event {
                CanonicalStreamEvent::ToolCallDelta {
                    arguments_delta, ..
                } if !arguments_delta.is_empty() => Some(arguments_delta),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(arguments, r#"{"q":"sf"}"#);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_redacts_failed_response_event() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.failed\n\
data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"},\"error\":{\"message\":\"bad sk-one sk-two xai-one eyJone\"}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let err = collect_stream_events(&provider)
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(!err.contains("sk-one"));
        assert!(!err.contains("sk-two"));
        assert!(!err.contains("xai-one"));
        assert!(!err.contains("eyJone"));
        assert!(err.contains("<redacted>"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_redacts_custom_bearer_from_sse_errors() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
env_key = "CUSTOM_CODEX_KEY"
"#,
                server.uri()
            ),
            None,
        );
        unsafe {
            std::env::set_var("CUSTOM_CODEX_KEY", "opaque-custom-token");
        }
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer opaque-custom-token"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.failed\n\
data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"},\"error\":{\"message\":\"bad opaque-custom-token\"}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let err = collect_stream_events(&provider)
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(!err.contains("opaque-custom-token"), "{err}");
        assert!(err.contains("<redacted>"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_redacts_sensitive_custom_headers_and_query_params() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false

[model_providers.proxy.http_headers]
X-API-Key = "opaque-header-secret"

[model_providers.proxy.query_params]
api_token = "opaque-query-secret"
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.failed\n\
data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"},\"error\":{\"message\":\"bad opaque-header-secret opaque-query-secret\"}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let err = collect_stream_events(&provider)
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(!err.contains("opaque-header-secret"), "{err}");
        assert!(!err.contains("opaque-query-secret"), "{err}");
        assert!(err.contains("<redacted>"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_preserves_split_utf8_lines() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"héllo 🌎\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            events[0],
            CanonicalStreamEvent::TextDelta("héllo 🌎".into())
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_accepts_bare_cr_sse_line_endings() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_text.delta\r\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\r\
\r\
event: response.completed\r\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\r\
\r",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(events[0], CanonicalStreamEvent::TextDelta("ok".into()));
        assert_eq!(
            events.last().unwrap(),
            &CanonicalStreamEvent::Finish {
                finish_reason: Some("stop".into())
            }
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_rejects_oversized_sse_event() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        let line = format!("data: {}\n", "x".repeat(1024));
        let body = line.repeat((MAX_SSE_EVENT_BYTES / 1024) + 2);
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let err = collect_stream_events(&provider)
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(err.contains("stream event exceeded"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_errors_on_missing_terminal_event() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}",
                "text/event-stream",
            ))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider).await;
        assert_eq!(
            events[0].as_ref().unwrap(),
            &CanonicalStreamEvent::TextDelta("partial".into())
        );
        let err = events[1].as_ref().unwrap_err().to_string();
        assert!(err.contains("terminal response event"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_rejects_chat_done_sentinel_on_responses_wire() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let err = collect_stream_events(&provider)
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(err.contains("[DONE] sentinel"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_rejects_non_sse_content_type() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status":"completed"})))
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let err = match provider
            .send_stream(CanonicalRequest {
                model: "gpt-custom".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await
        {
            Err(err) => err.to_string(),
            Ok(_) => panic!("non-SSE content-type is a pre-body error"),
        };
        assert!(err.contains("expected text/event-stream"), "{err}");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_treats_completed_failed_status_as_error() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"failed\",\"error\":{\"message\":\"bad sk-terminal\"}}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let err = collect_stream_events(&provider)
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap_err()
            .to_string();
        assert!(!err.contains("sk-terminal"));
        assert!(err.contains("<redacted>"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_preserves_incomplete_content_filter_reason() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.incomplete\n\
data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"}}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            events.last().unwrap(),
            &CanonicalStreamEvent::Finish {
                finish_reason: Some("content_filter".into())
            }
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_stream_maps_sparse_response_output_indexes_to_dense_tool_indexes() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let server = MockServer::start().await;
        let _home = TempCodexHome::new(
            &format!(
                r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "{}"
wire_api = "responses"
requires_openai_auth = false
"#,
                server.uri()
            ),
            None,
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    "event: response.output_item.added\n\
data: {\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_sparse\",\"name\":\"lookup\",\"arguments\":\"\"}}\n\n\
event: response.function_call_arguments.delta\n\
data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":2,\"delta\":\"{}\"}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
                    "text/event-stream",
                ),
            )
            .expect(1)
            .mount(&server)
            .await;

        let provider = CodexProvider::new().unwrap();
        let events = collect_stream_events(&provider)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>();
        assert_eq!(
            events[0],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: Some("call_sparse".into()),
                name: Some("lookup".into()),
                arguments_delta: String::new(),
            }
        );
        assert_eq!(
            events[1],
            CanonicalStreamEvent::ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments_delta: "{}".into(),
            }
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn codex_prompt_cache_key_reads_across_turns_live() {
        // Live opt-in: two turns with the same prompt_cache_key and a stable
        // prefix. Codex reports cached_tokens as cache_read on a hit. Model is
        // whatever the installed Codex config currently uses.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        if !omni_common::test_support::live_tests_enabled() {
            eprintln!(
                "skipping codex_prompt_cache_key_reads_across_turns_live: set OMNI_LIVE_TESTS=1"
            );
            return;
        }
        if !CodexProvider::detected() {
            eprintln!(
                "skipping codex_prompt_cache_key_reads_across_turns_live: Codex not detected"
            );
            return;
        }
        let p = match CodexProvider::new() {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "skipping codex_prompt_cache_key_reads_across_turns_live: ctor failed: {e}"
                );
                return;
            }
        };
        let model = match p.current_model() {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "skipping codex_prompt_cache_key_reads_across_turns_live: no current model: {e}"
                );
                return;
            }
        };
        let cache_key = Some("omni-live-codex-cache".to_string());
        let big_context = format!(
            "You are reviewing the following reference material. \
             Answer only from it. Reference block repeated for length:\n{}",
            "The quick brown fox jumps over the lazy dog near the riverbank. ".repeat(600)
        );

        let turn1 = CanonicalRequest {
            model: model.clone(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text(big_context.clone()),
            }],
            cache: Some(omni_core::CanonicalCacheIntent {
                routing_identity: cache_key.clone(),
                ..Default::default()
            }),
            max_tokens: Some(16),
            ..Default::default()
        };
        let r1 = p
            .send(turn1)
            .await
            .expect("turn 1 send must succeed with creds");
        let assistant_reply = if r1.content.is_empty() {
            "Understood.".to_string()
        } else {
            r1.content.clone()
        };
        let turn2 = CanonicalRequest {
            model,
            messages: vec![
                CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text(big_context),
                },
                CanonicalMessage {
                    role: "assistant".into(),
                    content: CanonicalContent::Text(assistant_reply),
                },
                CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("Reply with one word: ok".into()),
                },
            ],
            cache: Some(omni_core::CanonicalCacheIntent {
                routing_identity: cache_key,
                ..Default::default()
            }),
            max_tokens: Some(16),
            ..Default::default()
        };
        let r2 = p
            .send(turn2)
            .await
            .expect("turn 2 send must succeed with creds");
        assert!(
            r2.usage.cache_read > 0,
            "turn 2 must READ Codex prompt cache (prompt_cache_key routing + stable prefix); \
             got read={} input={}",
            r2.usage.cache_read,
            r2.usage.input_tokens
        );
    }
}

#[cfg(test)]
mod issue_51_tool_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn codex_copies_invalid_provider_schema_and_emits_boolean_strict() {
        for (schema, expected) in [
            (
                json!({"type":"object","additionalProperties":false,"properties":{"x":{"type":"string"}},"required":["x"]}),
                true,
            ),
            (
                json!({"type":"object","properties":{"x":{"type":"string"}}}),
                false,
            ),
            (json!({"type":"string"}), false),
            (
                json!({"anyOf":[{"type":"object"},{"type":"string"}]}),
                false,
            ),
            (json!({"oneOf":[{}],"allOf":[{}]}), false),
        ] {
            let req: omni_common::ChatCompletionRequest = serde_json::from_value(json!({"model":"gpt","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":schema,"strict":false}}]})).unwrap();
            let mut canon = omni_common::to_canonical(&req).unwrap();
            canon.tools.as_mut().unwrap()[0].strict = true;
            let wire = codex_responses_body(&canon, false).unwrap();
            assert_eq!(wire["tools"][0]["parameters"], schema);
            assert_eq!(wire["tools"][0]["strict"], expected);
        }
    }
}
