//! omni-common
//! Shared, provider- and frontend-agnostic infrastructure.
//! Extracted/adapted from the original claude-code-provider common pieces.

pub mod auth;
pub mod conversation_log;
pub mod credentials;
pub mod error;
pub mod http;
pub mod replacements;
pub mod session;
pub mod stats;
pub mod time_util;

pub use auth::{ApiKeyId, auth_layer};
pub use credentials::GrokCredentials;
pub use error::AppError;
pub use http::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, from_canonical,
    sse_from_canonical_stream, to_canonical, unix_now_secs,
};
pub use replacements::{Replacements, ReplacementsError};
pub use stats::{Stats, StatsSnapshot, TokenUsage};
