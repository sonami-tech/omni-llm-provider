use crate::canonical::{
    CanonicalRequest, CanonicalResponse, CanonicalStream, CanonicalStreamEvent,
};
use crate::version::ProviderVersion;
use async_trait::async_trait;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn id(&self) -> &'static str;

    /// The provider's own version catalog, newest-first (index 0 is the newest).
    ///
    /// Each version carries its conservative and extended model lists. Core uses
    /// this to resolve a [`crate::version::VersionSelector`] without ever
    /// hardcoding a provider's models. The default is empty: a provider that does
    /// not track an upstream client version opts out of version selection.
    fn versions(&self) -> &'static [ProviderVersion] {
        &[]
    }

    /// Non-streaming send: returns the full response once the upstream completes.
    async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError>;

    /// Streaming send: returns a stream of [`CanonicalStreamEvent`]s.
    ///
    /// The default implementation adapts [`send`](Self::send) into a single-shot
    /// stream (run to completion, then emit the buffered content/tool-calls/usage
    /// followed by a terminal `Finish`). Providers with native server-sent-event
    /// support (Claude, Grok) override this to pass deltas through incrementally.
    /// This keeps the trait object-safe and lets any provider participate in the
    /// streaming HTTP path even before it has a native implementation.
    async fn send_stream(&self, req: CanonicalRequest) -> Result<CanonicalStream, ProviderError> {
        let resp = self.send(req).await?;
        let mut events: Vec<Result<CanonicalStreamEvent, ProviderError>> = Vec::new();
        if !resp.content.is_empty() {
            events.push(Ok(CanonicalStreamEvent::TextDelta(resp.content)));
        }
        for (i, tc) in resp.tool_calls.into_iter().enumerate() {
            events.push(Ok(CanonicalStreamEvent::ToolCallDelta {
                index: i as u32,
                id: Some(tc.id),
                name: Some(tc.name),
                arguments_delta: tc.arguments,
            }));
        }
        events.push(Ok(CanonicalStreamEvent::Usage(resp.usage)));
        events.push(Ok(CanonicalStreamEvent::Finish {
            finish_reason: resp.finish_reason,
        }));
        Ok(Box::pin(futures_util_stream(events)))
    }
    // count_tokens etc. added as needed.
}

/// Build a `Send` stream from a finite vector of events without pulling in the
/// full `futures-util` dependency at this layer. Each poll yields the next item.
fn futures_util_stream(
    items: Vec<Result<CanonicalStreamEvent, ProviderError>>,
) -> impl futures_core::Stream<Item = Result<CanonicalStreamEvent, ProviderError>> + Send {
    struct VecStream(std::vec::IntoIter<Result<CanonicalStreamEvent, ProviderError>>);
    impl futures_core::Stream for VecStream {
        type Item = Result<CanonicalStreamEvent, ProviderError>;
        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Ready(self.0.next())
        }
    }
    VecStream(items.into_iter())
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("auth: {0}")]
    Auth(String),
    /// A client-input validation failure detected while preparing the request
    /// (malformed body, unrecognized model, unrepresentable content). This is the
    /// caller's fault, not an upstream failure, so the edge surfaces it as a 400
    /// rather than routing it through the upstream classifier.
    #[error("bad request: {0}")]
    BadRequest(String),
    /// An upstream provider/gateway failure. `status` carries the upstream HTTP
    /// status when the error originated at an HTTP-response boundary (so the
    /// edge can translate it deliberately); it is `None` for transport, decode,
    /// stream-framing, and other non-HTTP failures. `message` is the (already
    /// redacted, where applicable) detail; `status` is never redacted.
    #[error("upstream: {message}")]
    Upstream {
        status: Option<u16>,
        message: String,
    },
    #[error("other: {0}")]
    Other(#[from] anyhow::Error),
}

impl ProviderError {
    /// Construct an upstream error with no known HTTP status (transport, decode,
    /// stream-framing, or other non-HTTP failures). The edge classifier maps
    /// these to a 502 Bad Gateway.
    pub fn upstream(message: impl Into<String>) -> Self {
        Self::Upstream {
            status: None,
            message: message.into(),
        }
    }

    /// Construct an upstream error carrying the upstream HTTP status, so the edge
    /// can translate it per the gateway status policy instead of collapsing it.
    pub fn upstream_status(status: u16, message: impl Into<String>) -> Self {
        Self::Upstream {
            status: Some(status),
            message: message.into(),
        }
    }
}
