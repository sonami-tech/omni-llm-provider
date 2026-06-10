use crate::canonical::{CanonicalRequest, CanonicalResponse};
use async_trait::async_trait;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn id(&self) -> &'static str;
    async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError>;
    // streaming + count_tokens etc. added as needed.
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("auth: {0}")]
    Auth(String),
    #[error("upstream: {0}")]
    Upstream(String),
    #[error("other: {0}")]
    Other(#[from] anyhow::Error),
}
