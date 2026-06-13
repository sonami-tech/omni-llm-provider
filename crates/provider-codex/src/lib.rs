//! provider-codex
//!
//! Codex configuration backed provider. This crate intentionally reads Codex's
//! own `CODEX_HOME` / `~/.codex` config and auth state instead of inventing a
//! parallel Omni-only setup.

use async_trait::async_trait;
use omni_core::{
    CanonicalBlock, CanonicalContent, CanonicalMessage, CanonicalReasoning, CanonicalRequest,
    CanonicalResponse, CanonicalStream, CanonicalToolChoice, CanonicalUsage, LlmProvider,
    ProviderError,
};
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

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_CODEX_MODEL: &str = "gpt-5.5";
const DEFAULT_AUTH_COMMAND_TIMEOUT_MS: u64 = 5_000;

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
        vec![("codex".to_string(), model.clone()), (model.clone(), model)]
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

        let body = codex_responses_body(&req)?;
        let resp = self
            .client
            .post(url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Upstream(redact(&format!("codex network error: {e}"))))?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| {
            ProviderError::Upstream(redact(&format!("codex response read error: {e}")))
        })?;
        if !status.is_success() {
            return Err(ProviderError::Upstream(redact(&format!(
                "codex HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
            ))));
        }

        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| ProviderError::Upstream(format!("decode codex response: {e}")))?;
        codex_response_to_canonical(&value, &req.model)
    }

    async fn send_stream(&self, _req: CanonicalRequest) -> Result<CanonicalStream, ProviderError> {
        Err(ProviderError::Upstream(
            "Codex streaming is not implemented; retry without stream:true".into(),
        ))
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
        Self::load_from_home(&home)
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
            if let Ok(value) = std::env::var(env_name)
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

fn codex_responses_body(req: &CanonicalRequest) -> Result<Value, ProviderError> {
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
        "stream": false,
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
) -> Result<CanonicalResponse, ProviderError> {
    if value.get("status").and_then(|v| v.as_str()) == Some("failed") {
        return Err(ProviderError::Upstream(redact(&value.to_string())));
    }

    let model = value
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(fallback_model)
        .to_string();
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    if let Some(items) = value.get("output").and_then(|v| v.as_array()) {
        for item in items {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                content.push_str(text);
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
    let finish_reason = match value.get("status").and_then(|v| v.as_str()) {
        Some("incomplete") => Some("length".to_string()),
        _ if !tool_calls.is_empty() => Some("tool_calls".to_string()),
        _ => Some("stop".to_string()),
    };

    Ok(CanonicalResponse {
        model,
        content,
        tool_calls,
        finish_reason,
        usage: CanonicalUsage {
            input_tokens,
            output_tokens,
            cache_read: 0,
            cache_creation: 0,
        },
    })
}

fn redact(input: &str) -> String {
    let mut out = input.to_string();
    for marker in ["sk-", "xai-", "eyJ"] {
        if let Some(pos) = out.find(marker) {
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

    struct TempCodexHome {
        path: PathBuf,
        old_home: Option<std::ffi::OsString>,
        old_codex_api_key: Option<std::ffi::OsString>,
        old_openai: Option<std::ffi::OsString>,
        old_codex_access_token: Option<std::ffi::OsString>,
        old_custom: Option<std::ffi::OsString>,
        old_custom_header: Option<std::ffi::OsString>,
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
            unsafe {
                std::env::set_var("CODEX_HOME", &path);
                std::env::remove_var("CODEX_API_KEY");
                std::env::remove_var("OPENAI_API_KEY");
                std::env::remove_var("CODEX_ACCESS_TOKEN");
                std::env::remove_var("CUSTOM_CODEX_KEY");
                std::env::remove_var("CUSTOM_CODEX_HEADER");
            }
            Self {
                path,
                old_home,
                old_codex_api_key,
                old_openai,
                old_codex_access_token,
                old_custom,
                old_custom_header,
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
        let body = codex_responses_body(&req).unwrap();
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
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {"type":"message","content":[{"type":"output_text","text":"hello"}]},
                {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4}
        });
        let resp = codex_response_to_canonical(&value, "fallback").unwrap();
        assert_eq!(resp.model, "gpt-5.5");
        assert_eq!(resp.content, "hello");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_1");
        assert_eq!(resp.tool_calls[0].name, "lookup");
        assert_eq!(resp.tool_calls[0].arguments, "{}");
        assert_eq!(resp.usage.input_tokens, 3);
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn upstream_errors_are_redacted() {
        let redacted = redact(r#"{"error":"bad sk-test xai-test eyJtoken"}"#);
        assert!(!redacted.contains("sk-test"));
        assert!(!redacted.contains("xai-test"));
        assert!(!redacted.contains("eyJtoken"));
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
    async fn send_stream_rejects_until_native_responses_sse_is_implemented() {
        let provider = CodexProvider::new().unwrap();
        let result = provider
            .send_stream(CanonicalRequest {
                model: "gpt-custom".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            })
            .await;
        let err = match result {
            Ok(_) => panic!("Codex stream:true must not succeed before native SSE is implemented"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("streaming is not implemented"),
            "Codex stream:true must fail loudly instead of using buffered pseudo-streaming: {err}"
        );
    }
}
