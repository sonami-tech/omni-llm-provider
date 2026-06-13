//! provider-codex
//!
//! Codex configuration backed provider. This crate intentionally reads Codex's
//! own `CODEX_HOME` / `~/.codex` config and auth state instead of inventing a
//! parallel Omni-only setup.

use async_trait::async_trait;
use futures_util::StreamExt;
use omni_core::{
    CanonicalBlock, CanonicalContent, CanonicalMessage, CanonicalReasoning, CanonicalRequest,
    CanonicalResponse, CanonicalResponseMetadata, CanonicalStream, CanonicalStreamEvent,
    CanonicalToolChoice, CanonicalUsage, LlmProvider, ProviderError,
};
use reqwest::header::HeaderMap;
use reqwest::{Client, Url, header};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_CODEX_MODEL: &str = "gpt-5.5";
const DEFAULT_AUTH_COMMAND_TIMEOUT_MS: u64 = 5_000;
const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;
const MAX_SSE_EVENT_BYTES: usize = 8 * 1024 * 1024;

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
}

impl CodexProvider {
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent(format!("omni/{} provider-codex", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(|e| ProviderError::Other(anyhow::anyhow!("http client: {e}")))?;
        Ok(Self { client })
    }

    pub fn current_model(&self) -> Result<String, ProviderError> {
        Ok(CodexRequestConfig::load()?.model)
    }

    pub fn detected() -> bool {
        CodexRequestConfig::detected()
    }

    pub fn models_list(&self) -> Vec<CodexModelInfo> {
        let id = self
            .current_model()
            .unwrap_or_else(|_| DEFAULT_CODEX_MODEL.to_string());
        vec![CodexModelInfo {
            id,
            object: "model",
            created: 0,
            owned_by: "codex",
        }]
    }

    pub fn model_aliases(&self) -> Vec<(String, String)> {
        let model = self
            .current_model()
            .unwrap_or_else(|_| DEFAULT_CODEX_MODEL.to_string());
        vec![
            ("codex".to_string(), model.clone()),
            ("gpt".to_string(), model.clone()),
            (model.clone(), model),
        ]
    }
}

#[async_trait]
impl LlmProvider for CodexProvider {
    fn id(&self) -> &'static str {
        "codex"
    }

    async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError> {
        let config = CodexRequestConfig::load()?;
        if config.wire_api != WireApi::Responses {
            return Err(ProviderError::Upstream(format!(
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
        let error_redactor = CodexErrorRedactor::from_request(&url, &headers);

        let body = codex_responses_body(&req, false)?;
        let resp = self
            .client
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                ProviderError::Upstream(error_redactor.redact(&format!("codex network error: {e}")))
            })?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| {
            ProviderError::Upstream(
                error_redactor.redact(&format!("codex response read error: {e}")),
            )
        })?;
        if !status.is_success() {
            return Err(ProviderError::Upstream(error_redactor.redact(&format!(
                "codex HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
            ))));
        }

        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| ProviderError::Upstream(format!("decode codex response: {e}")))?;
        codex_response_to_canonical(&value, &req.model, &error_redactor)
    }

    async fn send_stream(&self, req: CanonicalRequest) -> Result<CanonicalStream, ProviderError> {
        let config = CodexRequestConfig::load()?;
        if config.wire_api != WireApi::Responses {
            return Err(ProviderError::Upstream(format!(
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
        let error_redactor = CodexErrorRedactor::from_request(&url, &headers);
        let client = self.client.clone();

        let stream = async_stream::stream! {
            let send_result = client
                .post(url)
                .headers(headers)
                .json(&body)
                .send()
                .await;

            let http_resp = match send_result {
                Ok(resp) => resp,
                Err(e) => {
                    yield Err(ProviderError::Upstream(error_redactor.redact(&format!("codex network error: {e}"))));
                    return;
                }
            };

            let status = http_resp.status();
            if !status.is_success() {
                let err_body = error_redactor.redact(
                    &http_resp
                        .text()
                        .await
                        .unwrap_or_else(|_| "<no body>".to_string()),
                );
                yield Err(ProviderError::Upstream(error_redactor.redact(&format!("codex HTTP {status}: {err_body}"))));
                return;
            }

            if let Some(content_type) = http_resp
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                && !content_type
                    .to_ascii_lowercase()
                    .starts_with("text/event-stream")
            {
                yield Err(ProviderError::Upstream(format!(
                    "codex stream expected text/event-stream, got {content_type}"
                )));
                return;
            }

            let mut bytes = http_resp.bytes_stream();
            let mut sse = ResponsesSseBuffer::default();
            let mut parser = ResponsesStreamParser::new(error_redactor.clone());
            let mut finished = false;
            let mut saw_event = false;

            while let Some(chunk) = bytes.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        yield Err(ProviderError::Upstream(error_redactor.redact(&format!("codex stream read error: {e}"))));
                        return;
                    }
                };
                let events = match sse.push(&chunk) {
                    Ok(events) => events,
                    Err(e) => {
                        yield Err(ProviderError::Upstream(e));
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
                        yield Err(ProviderError::Upstream(e));
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
                yield Err(ProviderError::Upstream(error_redactor.redact(message)));
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
        let config_path = home.join("config.toml");
        let raw = std::fs::read_to_string(&config_path).unwrap_or_default();
        let value: toml::Value = if raw.trim().is_empty() {
            toml::Value::Table(Default::default())
        } else {
            raw.parse::<toml::Value>()
                .map_err(|e| ProviderError::Auth(format!("Codex config parse failed: {e}")))?
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

    async fn headers(&self) -> Result<header::HeaderMap, ProviderError> {
        let mut headers = header::HeaderMap::new();
        for (name, value) in &self.http_headers {
            insert_header(&mut headers, name, value)?;
        }
        for (name, env_name) in &self.env_http_headers {
            if name == "__omni_custom_headers__" {
                for (header_name, header_value) in headers_from_env(env_name)? {
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
            return self.openai_auth_token().map(Some);
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
        Ok(None)
    }

    fn openai_auth_token(&self) -> Result<String, ProviderError> {
        for env_key in ["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"] {
            if let Ok(token) = std::env::var(env_key)
                && !token.trim().is_empty()
            {
                return Ok(token);
            }
        }
        let auth_path = self.home.join("auth.json");
        let bytes = std::fs::read(&auth_path).map_err(|e| {
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

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn headers_from_env(env_name: &str) -> Result<Vec<(String, String)>, ProviderError> {
    let Some(raw) = env_nonempty(env_name) else {
        return Ok(Vec::new());
    };
    parse_custom_headers(&raw).map_err(ProviderError::Auth)
}

fn parse_custom_headers(raw: &str) -> Result<Vec<(String, String)>, String> {
    let mut headers = Vec::new();
    for line in raw.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| "custom header must be formatted as `Name: value`".to_string())?;
        let name = name.trim();
        let value = value.trim();
        if name.is_empty() || value.is_empty() {
            return Err("custom header name and value must both be non-empty".into());
        }
        headers.push((name.to_string(), value.to_string()));
    }
    Ok(headers)
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

fn codex_responses_body(req: &CanonicalRequest, stream: bool) -> Result<Value, ProviderError> {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for message in &req.messages {
        if message.role == "system" || message.role == "developer" {
            instructions.push(message.content.as_text());
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
        && let Some(effort) = effort
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
    if let Some(extras) = &req.provider_extras
        && let Some(obj) = extras.as_object()
    {
        for (key, value) in obj {
            if codex_extra_allowed(key) {
                body[key] = value.clone();
            }
        }
    }
    Ok(body)
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
            for block in blocks {
                match block {
                    CanonicalBlock::Text(t) => text.push_str(t),
                    CanonicalBlock::ToolUse {
                        id,
                        name,
                        arguments,
                    } => {
                        if !text.is_empty() {
                            input.push(json!({
                                "type": "message",
                                "role": message.role,
                                "content": std::mem::take(&mut text),
                            }));
                        }
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
                        ..
                    } => {
                        if !text.is_empty() {
                            input.push(json!({
                                "type": "message",
                                "role": message.role,
                                "content": std::mem::take(&mut text),
                            }));
                        }
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": tool_use_id,
                            "output": content,
                        }));
                    }
                }
            }
            if !text.is_empty() {
                input.push(json!({
                    "type": "message",
                    "role": message.role,
                    "content": text,
                }));
            }
        }
    }
}

fn codex_extra_allowed(key: &str) -> bool {
    matches!(
        key,
        "store" | "previous_response_id" | "metadata" | "parallel_tool_calls" | "service_tier"
    )
}

fn codex_response_to_canonical(
    value: &Value,
    fallback_model: &str,
    error_redactor: &CodexErrorRedactor,
) -> Result<CanonicalResponse, ProviderError> {
    if value.get("status").and_then(|v| v.as_str()) == Some("failed") {
        return Err(ProviderError::Upstream(
            error_redactor.redact(&value.to_string()),
        ));
    }

    let model = value
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_model)
        .to_string();
    let response_id = value.get("id").and_then(|v| v.as_str()).map(str::to_string);
    let mut content = String::new();
    let mut refusal = String::new();
    let mut tool_calls = Vec::new();
    if let Some(items) = value.get("output").and_then(|v| v.as_array()) {
        for item in items {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                content.push_str(text);
                            } else if let Some(refusal_text) =
                                part.get("refusal").and_then(|v| v.as_str())
                            {
                                refusal.push_str(refusal_text);
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("call_unknown")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    tool_calls.push(omni_core::CanonicalToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                _ => {}
            }
        }
    }
    if content.is_empty()
        && let Some(text) = value.get("output_text").and_then(|v| v.as_str())
    {
        content.push_str(text);
    }

    let usage = value.get("usage").unwrap_or(&Value::Null);
    let input_tokens = usage
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let finish_reason = match response_status(value) {
        Some("incomplete") => Some(response_incomplete_reason(value).to_string()),
        _ if !tool_calls.is_empty() => Some("tool_calls".to_string()),
        _ => Some("stop".to_string()),
    };

    Ok(CanonicalResponse {
        model,
        content,
        refusal: if refusal.is_empty() {
            None
        } else {
            Some(refusal)
        },
        tool_calls,
        finish_reason,
        usage: CanonicalUsage {
            input_tokens,
            output_tokens,
            cache_read: 0,
            cache_creation: 0,
        },
        id: response_id,
    })
}

#[derive(Debug, Default)]
struct ResponsesSseBuffer {
    line: Vec<u8>,
    last_was_cr: bool,
    event: Option<String>,
    data: Vec<String>,
    event_bytes: usize,
}

#[derive(Debug)]
struct ResponsesSseEvent {
    event: Option<String>,
    data: String,
}

impl ResponsesSseBuffer {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<ResponsesSseEvent>, String> {
        let mut events = Vec::new();
        for line in self.complete_lines(bytes)? {
            self.process_line(line, &mut events)?;
        }
        Ok(events)
    }

    fn complete_lines(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        let mut lines = Vec::new();
        for byte in bytes {
            if self.last_was_cr {
                self.last_was_cr = false;
                if *byte == b'\n' {
                    continue;
                }
            }
            match *byte {
                b'\n' => lines.push(self.take_line()?),
                b'\r' => {
                    lines.push(self.take_line()?);
                    self.last_was_cr = true;
                }
                byte => {
                    self.line.push(byte);
                    if self.line.len() > MAX_SSE_LINE_BYTES {
                        return Err(format!(
                            "codex stream line exceeded {} bytes",
                            MAX_SSE_LINE_BYTES
                        ));
                    }
                }
            }
        }
        Ok(lines)
    }

    fn take_line(&mut self) -> Result<String, String> {
        String::from_utf8(std::mem::take(&mut self.line))
            .map_err(|e| format!("codex stream line was not UTF-8: {e}"))
    }

    fn process_line(
        &mut self,
        line: String,
        events: &mut Vec<ResponsesSseEvent>,
    ) -> Result<(), String> {
        if line.is_empty() {
            if let Some(event) = self.take_event() {
                events.push(event);
            }
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }
        if let Some(value) = line.strip_prefix("event:") {
            self.event = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            let value = value.trim_start();
            self.event_bytes = self.event_bytes.saturating_add(value.len());
            if self.event_bytes > MAX_SSE_EVENT_BYTES {
                return Err(format!(
                    "codex stream event exceeded {} bytes",
                    MAX_SSE_EVENT_BYTES
                ));
            }
            self.data.push(value.to_string());
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<ResponsesSseEvent>, String> {
        if !self.line.is_empty() {
            let line = self.take_line()?;
            let mut events = Vec::new();
            self.process_line(line, &mut events)?;
            if let Some(event) = events.into_iter().next() {
                return Ok(Some(event));
            }
        }
        Ok(self.take_event())
    }

    fn take_event(&mut self) -> Option<ResponsesSseEvent> {
        if self.event.is_none() && self.data.is_empty() {
            return None;
        }
        let event = ResponsesSseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data).join("\n"),
        };
        self.event_bytes = 0;
        Some(event)
    }
}

#[derive(Debug, Clone, Default)]
struct StreamToolCall {
    id: Option<String>,
    name: Option<String>,
    emitted_open: bool,
    arguments: String,
    emitted_arguments_len: usize,
    canonical_index: u32,
}

#[derive(Debug, Default)]
struct ResponsesStreamParser {
    tool_calls: HashMap<u32, StreamToolCall>,
    next_tool_index: u32,
    saw_tool_call: bool,
    emitted_text: HashMap<(u32, &'static str), String>,
    completed: bool,
    error_redactor: CodexErrorRedactor,
}

impl ResponsesStreamParser {
    fn new(error_redactor: CodexErrorRedactor) -> Self {
        Self {
            error_redactor,
            ..Default::default()
        }
    }

    fn redact(&self, input: &str) -> String {
        self.error_redactor.redact(input)
    }

    fn handle_event(
        &mut self,
        event: ResponsesSseEvent,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let event_type = event.event.as_deref().unwrap_or_default();
        if event.data.trim() == "[DONE]" {
            return vec![Err(ProviderError::Upstream(
                "codex Responses stream sent Chat [DONE] sentinel without a terminal response event"
                    .into(),
            ))];
        }
        let value: Value = match serde_json::from_str(&event.data) {
            Ok(value) => value,
            Err(e) => {
                return vec![Err(ProviderError::Upstream(self.redact(&format!(
                    "decode codex stream event {event_type}: {e}: {}",
                    event.data
                ))))];
            }
        };
        let kind = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or(event_type);
        match kind {
            "response.created" => self.handle_response_metadata(&value),
            "response.output_text.delta" | "response.refusal.delta" => self
                .emit_text_delta(
                    response_output_index(&value),
                    if kind == "response.refusal.delta" {
                        "refusal"
                    } else {
                        "text"
                    },
                    value
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default(),
                )
                .into_iter()
                .map(Ok)
                .collect(),
            "response.output_text.done" => self.handle_text_done(&value, "text"),
            "response.refusal.done" => self.handle_text_done(&value, "refusal"),
            "response.output_item.added" => self.handle_output_item_added(&value),
            "response.function_call_arguments.delta" => self.handle_function_args_delta(&value),
            "response.function_call_arguments.done" => self.handle_function_args_done(&value),
            "response.output_item.done" => self.handle_output_item_done(&value),
            "response.completed" => self.handle_completed(&value),
            "response.incomplete" => self.handle_incomplete(&value),
            "response.failed" | "error" => {
                vec![Err(ProviderError::Upstream(
                    self.redact(&value.to_string()),
                ))]
            }
            _ => Vec::new(),
        }
    }

    fn handle_response_metadata(
        &self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let id = value
            .get("response")
            .and_then(|v| v.get("id"))
            .or_else(|| value.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        id.map(|id| {
            vec![Ok(CanonicalStreamEvent::ResponseMetadata(
                CanonicalResponseMetadata { id: Some(id) },
            ))]
        })
        .unwrap_or_default()
    }

    fn emit_text_delta(
        &mut self,
        output_index: u32,
        channel: &'static str,
        delta: &str,
    ) -> Vec<CanonicalStreamEvent> {
        if delta.is_empty() {
            return Vec::new();
        }
        self.emitted_text
            .entry((output_index, channel))
            .or_default()
            .push_str(delta);
        let delta = delta.to_string();
        if channel == "refusal" {
            vec![CanonicalStreamEvent::RefusalDelta(delta)]
        } else {
            vec![CanonicalStreamEvent::TextDelta(delta)]
        }
    }

    fn handle_text_done(
        &mut self,
        value: &Value,
        field: &'static str,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let final_text = value
            .get(field)
            .or_else(|| value.get("text"))
            .or_else(|| value.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let output_index = response_output_index(value);
        self.emit_final_text(output_index, field, final_text)
    }

    fn emit_final_text(
        &mut self,
        output_index: u32,
        field: &'static str,
        final_text: &str,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        if final_text.is_empty() {
            return Vec::new();
        }
        let emitted = self
            .emitted_text
            .get(&(output_index, field))
            .map(String::as_str)
            .unwrap_or_default();
        if !final_text.starts_with(emitted) {
            return vec![Err(ProviderError::Upstream(self.redact(&format!(
                "codex stream {field}.done text did not extend prior text deltas"
            ))))];
        }
        let suffix = &final_text[emitted.len()..];
        self.emit_text_delta(output_index, field, suffix)
            .into_iter()
            .map(Ok)
            .collect()
    }

    fn handle_output_item_added(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let Some(item) = value.get("item") else {
            return Vec::new();
        };
        if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
            return Vec::new();
        }
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let canonical_index = self.ensure_tool_call(output_index);
        let call = self.tool_calls.entry(output_index).or_default();
        call.id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        call.name = item
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        self.saw_tool_call = true;
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) {
            if let Some(err) = self.append_tool_arguments_from_full(output_index, arguments) {
                return vec![Err(err)];
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect::<Vec<_>>()
    }

    fn handle_function_args_delta(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let delta = value
            .get("delta")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        self.saw_tool_call = true;
        let canonical_index = self.ensure_tool_call(output_index);
        if !delta.is_empty() {
            let already = self
                .tool_calls
                .get(&output_index)
                .map(|call| call.arguments.clone())
                .unwrap_or_default();
            if !already.is_empty() && delta == already {
                // Some Responses-compatible gateways repeat the full arguments
                // as the first delta after announcing them on output_item.added.
            } else if delta.starts_with(&already) && delta.len() > already.len() {
                self.append_tool_arguments(output_index, &delta[already.len()..]);
            } else {
                self.append_tool_arguments(output_index, &delta);
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn handle_function_args_done(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let Some(arguments) = value.get("arguments").and_then(|v| v.as_str()) else {
            return Vec::new();
        };
        let already = self
            .tool_calls
            .get(&output_index)
            .map(|call| call.arguments.clone())
            .unwrap_or_default();
        if arguments == already {
            return Vec::new();
        }
        if arguments.len() <= already.len() || !arguments.starts_with(&already) {
            return vec![Err(ProviderError::Upstream(self.redact(
                "codex stream function_call_arguments.done arguments did not extend prior argument deltas",
            )))];
        }
        let delta = arguments[already.len()..].to_string();
        let canonical_index = self.ensure_tool_call(output_index);
        self.append_tool_arguments(output_index, &delta);
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn handle_output_item_done(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let Some(item) = value.get("item") else {
            return Vec::new();
        };
        if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
            return Vec::new();
        }
        let output_index = value
            .get("output_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let canonical_index = self.ensure_tool_call(output_index);
        {
            let call = self.tool_calls.entry(output_index).or_default();
            if call.id.is_none() {
                call.id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
            if call.name.is_none() {
                call.name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
        }
        self.saw_tool_call = true;
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) {
            if let Some(err) = self.append_tool_arguments_from_full(output_index, arguments) {
                return vec![Err(err)];
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn handle_completed(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        if response_status(value) == Some("failed") {
            return vec![Err(ProviderError::Upstream(
                self.redact(&value.to_string()),
            ))];
        }
        self.completed = true;
        let mut events = self.handle_response_metadata(value);
        events.extend(self.emit_terminal_output(value));
        if let Some(usage) = response_usage(value) {
            events.push(Ok(CanonicalStreamEvent::Usage(usage)));
        }
        events.push(Ok(CanonicalStreamEvent::Finish {
            finish_reason: self.finish_reason(),
        }));
        events
    }

    fn handle_incomplete(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        self.completed = true;
        let mut events = self.handle_response_metadata(value);
        events.extend(self.emit_terminal_output(value));
        if let Some(usage) = response_usage(value) {
            events.push(Ok(CanonicalStreamEvent::Usage(usage)));
        }
        events.push(Ok(CanonicalStreamEvent::Finish {
            finish_reason: Some(response_incomplete_reason(value).to_string()),
        }));
        events
    }

    fn emit_terminal_output(
        &mut self,
        value: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let mut events = Vec::new();
        let Some(items) = response_payload(value)
            .get("output")
            .and_then(|v| v.as_array())
        else {
            return events;
        };
        for (position, item) in items.iter().enumerate() {
            let output_index = item
                .get("output_index")
                .and_then(|v| v.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(position as u32);
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            events.extend(self.emit_terminal_content_part(output_index, part));
                        }
                    }
                }
                Some("function_call") => {
                    events.extend(self.emit_terminal_function_call(output_index, item));
                }
                _ => {}
            }
        }
        events
    }

    fn emit_terminal_content_part(
        &mut self,
        output_index: u32,
        part: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let kind = part.get("type").and_then(|v| v.as_str());
        if kind == Some("refusal") || part.get("refusal").is_some() {
            let final_text = part
                .get("refusal")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            return self.emit_final_text(output_index, "refusal", final_text);
        }
        if kind == Some("output_text") || part.get("text").is_some() {
            let final_text = part
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            return self.emit_final_text(output_index, "text", final_text);
        }
        Vec::new()
    }

    fn emit_terminal_function_call(
        &mut self,
        output_index: u32,
        item: &Value,
    ) -> Vec<Result<CanonicalStreamEvent, ProviderError>> {
        let canonical_index = self.ensure_tool_call(output_index);
        {
            let call = self.tool_calls.entry(output_index).or_default();
            if call.id.is_none() {
                call.id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
            if call.name.is_none() {
                call.name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
            }
        }
        self.saw_tool_call = true;
        if let Some(arguments) = item.get("arguments").and_then(|v| v.as_str()) {
            if let Some(err) = self.append_tool_arguments_from_full(output_index, arguments) {
                return vec![Err(err)];
            }
        }
        let mut events = self.emit_tool_open_if_ready(output_index);
        events.extend(self.emit_pending_tool_args(output_index, canonical_index));
        events.into_iter().map(Ok).collect()
    }

    fn ensure_tool_call(&mut self, output_index: u32) -> u32 {
        if let Some(call) = self.tool_calls.get(&output_index) {
            return call.canonical_index;
        }
        let canonical_index = self.next_tool_index;
        self.next_tool_index += 1;
        self.tool_calls.insert(
            output_index,
            StreamToolCall {
                canonical_index,
                ..Default::default()
            },
        );
        canonical_index
    }

    fn emit_tool_open(&mut self, output_index: u32) -> Vec<CanonicalStreamEvent> {
        let canonical_index = self.ensure_tool_call(output_index);
        let call = self.tool_calls.entry(output_index).or_default();
        if call.emitted_open {
            return Vec::new();
        }
        call.emitted_open = true;
        vec![CanonicalStreamEvent::ToolCallDelta {
            index: canonical_index,
            id: call.id.clone(),
            name: call.name.clone(),
            arguments_delta: String::new(),
        }]
    }

    fn emit_tool_open_if_ready(&mut self, output_index: u32) -> Vec<CanonicalStreamEvent> {
        let Some(call) = self.tool_calls.get(&output_index) else {
            return Vec::new();
        };
        if call.emitted_open || call.id.is_none() || call.name.is_none() {
            return Vec::new();
        }
        self.emit_tool_open(output_index)
    }

    fn append_tool_arguments(&mut self, output_index: u32, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if let Some(call) = self.tool_calls.get_mut(&output_index) {
            call.arguments.push_str(delta);
        }
    }

    fn append_tool_arguments_from_full(
        &mut self,
        output_index: u32,
        arguments: &str,
    ) -> Option<ProviderError> {
        if arguments.is_empty() {
            return None;
        }
        let already = self
            .tool_calls
            .get(&output_index)
            .map(|call| call.arguments.clone())
            .unwrap_or_default();
        if arguments == already {
            return None;
        }
        if arguments.len() > already.len() && arguments.starts_with(&already) {
            self.append_tool_arguments(output_index, &arguments[already.len()..]);
            return None;
        }
        Some(ProviderError::Upstream(self.redact(
            "codex stream terminal function_call arguments did not extend prior argument deltas",
        )))
    }

    fn emit_pending_tool_args(
        &mut self,
        output_index: u32,
        canonical_index: u32,
    ) -> Vec<CanonicalStreamEvent> {
        let Some(call) = self.tool_calls.get_mut(&output_index) else {
            return Vec::new();
        };
        if !call.emitted_open || call.emitted_arguments_len >= call.arguments.len() {
            return Vec::new();
        }
        let delta = call.arguments[call.emitted_arguments_len..].to_string();
        call.emitted_arguments_len = call.arguments.len();
        vec![CanonicalStreamEvent::ToolCallDelta {
            index: canonical_index,
            id: None,
            name: None,
            arguments_delta: delta,
        }]
    }

    fn finish_reason(&self) -> Option<String> {
        Some(if self.saw_tool_call {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        })
    }
}

fn response_usage(value: &Value) -> Option<CanonicalUsage> {
    let usage = value
        .get("response")
        .and_then(|v| v.get("usage"))
        .or_else(|| value.get("usage"))?;
    Some(CanonicalUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        output_tokens: usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_read: 0,
        cache_creation: 0,
    })
}

fn response_status(value: &Value) -> Option<&str> {
    response_payload(value)
        .get("status")
        .and_then(|v| v.as_str())
}

fn response_payload(value: &Value) -> &Value {
    value.get("response").unwrap_or(value)
}

fn response_output_index(value: &Value) -> u32 {
    value
        .get("output_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

fn response_incomplete_reason(value: &Value) -> &str {
    let reason = value
        .get("response")
        .and_then(|v| v.get("incomplete_details"))
        .and_then(|v| v.get("reason"))
        .or_else(|| {
            value
                .get("incomplete_details")
                .and_then(|v| v.get("reason"))
        })
        .and_then(|v| v.as_str())
        .unwrap_or("max_output_tokens");
    if reason == "max_output_tokens" {
        "length"
    } else {
        reason
    }
}

#[derive(Debug, Clone, Default)]
struct CodexErrorRedactor {
    secrets: Vec<String>,
}

impl CodexErrorRedactor {
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
    let mut out = input.to_string();
    for marker in ["sk-", "xai-", "eyJ"] {
        while let Some(pos) = out.find(marker) {
            let end = out[pos..]
                .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == ',')
                .map(|i| pos + i)
                .unwrap_or(out.len());
            out.replace_range(pos..end, "<redacted>");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::CanonicalTool;
    use std::sync::Mutex;
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
            }
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn custom_provider_without_auth_does_not_use_auth_json_or_env() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = TempCodexHome::new(
            r#"
model = "gpt-custom"
model_provider = "proxy"
[model_providers.proxy]
base_url = "https://proxy.example.com"
wire_api = "responses"
requires_openai_auth = false
"#,
            Some(
                r#"{"OPENAI_API_KEY":"sk-should-not-leak","tokens":{"access_token":"eyJshould-not-leak"}}"#,
            ),
        );
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-env-should-not-leak");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert!(
            !headers.contains_key(header::AUTHORIZATION),
            "custom provider no-auth must not inherit OpenAI auth"
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
        assert!(aliases.contains(&("codex".into(), "gpt-override".into())));
        assert!(aliases.contains(&("gpt".into(), "gpt-override".into())));
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
    fn custom_headers_and_env_headers_are_applied_without_auth_fallback() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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
            Some(r#"{"OPENAI_API_KEY":"sk-should-not-leak"}"#),
        );
        unsafe {
            std::env::set_var("CUSTOM_CODEX_HEADER", "dynamic-value");
        }

        let cfg = CodexRequestConfig::load().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let headers = rt.block_on(cfg.headers()).unwrap();
        assert_eq!(headers.get("x-static").unwrap(), "static-value");
        assert_eq!(headers.get("x-dynamic").unwrap(), "dynamic-value");
        assert!(
            !headers.contains_key(header::AUTHORIZATION),
            "custom headers must not imply OpenAI auth fallback"
        );
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
                    }]),
                },
            ],
            tools: Some(vec![CanonicalTool {
                name: "lookup".into(),
                description: Some("Lookup".into()),
                parameters: json!({"type":"object"}),
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
            provider_extras: Some(json!({
                "store": false,
                "previous_response_id": "resp_1",
                "not_allowed": "drop"
            })),
        };
        let body = codex_responses_body(&req, false).unwrap();
        assert_eq!(body["instructions"], "sys");
        assert_eq!(body["input"].as_array().unwrap().len(), 2);
        assert_eq!(body["tools"][0]["name"], "lookup");
        assert_eq!(body["tool_choice"]["name"], "lookup");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["store"], false);
        assert_eq!(body["previous_response_id"], "resp_1");
        assert!(body.get("not_allowed").is_none());
    }

    #[test]
    fn responses_output_maps_to_canonical() {
        let value = json!({
            "id": "resp_backend",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"hello"}]},
                {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4}
        });
        let resp = codex_response_to_canonical(&value, "fallback", &CodexErrorRedactor::default())
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
        let resp = codex_response_to_canonical(&value, "fallback", &CodexErrorRedactor::default())
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

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn send_uses_custom_provider_without_auth_header() {
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
            Some(r#"{"OPENAI_API_KEY":"sk-must-not-leak"}"#),
        );
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
        assert!(
            !requests[0].headers.contains_key("authorization"),
            "custom no-auth provider must not send Authorization"
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
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_backend\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"terminal only\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
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
                id: Some("resp_backend".into())
            })
        );
        assert_eq!(
            events[1],
            CanonicalStreamEvent::TextDelta("terminal only".into())
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
        let err = collect_stream_events(&provider)
            .await
            .into_iter()
            .next()
            .unwrap()
            .unwrap_err()
            .to_string();
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
}
