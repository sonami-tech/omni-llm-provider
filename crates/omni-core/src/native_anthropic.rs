//! Optional native Anthropic Messages surface.
//!
//! Providers that speak Anthropic natively (Claude) expose this capability so the
//! edge can dispatch raw Messages / count_tokens without importing fingerprint
//! profiles, passthrough helpers, or concrete provider types.
//!
//! `LlmProvider` remains canonical-only. This trait is a separate optional
//! capability on the uniform provider entry.

use std::pin::Pin;

use async_trait::async_trait;
use futures_core::Stream;
use serde_json::Value;

use crate::ProviderError;

/// One prepared native Anthropic request after model routing and wire shaping.
///
/// The body is opaque to the edge: only the provider that prepared it should
/// send it. Metadata fields are for stats / logging / model keys.
#[derive(Debug, Clone)]
pub struct PreparedAnthropicNative {
    pub requested_model: String,
    pub model_canonical: String,
    pub outbound_model: String,
    pub stream: bool,
    pub dropped_fields: Vec<String>,
    body: Value,
}

impl PreparedAnthropicNative {
    pub fn new(
        requested_model: String,
        model_canonical: String,
        outbound_model: String,
        stream: bool,
        dropped_fields: Vec<String>,
        body: Value,
    ) -> Self {
        Self {
            requested_model,
            model_canonical,
            outbound_model,
            stream,
            dropped_fields,
            body,
        }
    }

    pub fn body(&self) -> &Value {
        &self.body
    }
}

/// One SSE frame from a native Anthropic Messages stream (event name + JSON data).
#[derive(Debug, Clone)]
pub struct NativeAnthropicSseFrame {
    pub event: String,
    pub data: Value,
}

/// Object-safe stream of native Anthropic SSE frames.
pub type NativeAnthropicSseStream =
    Pin<Box<dyn Stream<Item = Result<NativeAnthropicSseFrame, ProviderError>> + Send + 'static>>;

/// Optional capability: raw Anthropic Messages / count_tokens.
///
/// Implementors own fingerprint, stale-cch handling, credentials, and wire
/// defaults. The edge only prepares body identity (model strip), then calls
/// these methods.
#[async_trait]
pub trait AnthropicNativeSurface: Send + Sync {
    /// Parse + shape a client `/v1/messages` body for upstream (identity injection
    /// and wire defaults applied inside the implementor).
    fn prepare_messages(
        &self,
        raw_body: Value,
        inject_identity: bool,
    ) -> Result<PreparedAnthropicNative, ProviderError>;

    /// Parse + shape a client `/v1/messages/count_tokens` body.
    fn prepare_count_tokens(
        &self,
        raw_body: Value,
    ) -> Result<PreparedAnthropicNative, ProviderError>;

    /// Non-streaming Messages send. `session_key` is an opaque correlation string
    /// (gateway session id); the implementor maps it to its own request context.
    async fn send_messages_json(
        &self,
        body: &Value,
        session_key: &str,
        outbound_model: &str,
    ) -> Result<Value, ProviderError>;

    /// Streaming Messages send (raw Anthropic SSE frames).
    async fn send_messages_stream(
        &self,
        body: &Value,
        session_key: &str,
        outbound_model: &str,
    ) -> Result<NativeAnthropicSseStream, ProviderError>;

    /// count_tokens send.
    async fn send_count_tokens(
        &self,
        body: &Value,
        session_key: &str,
        outbound_model: &str,
    ) -> Result<Value, ProviderError>;
}
