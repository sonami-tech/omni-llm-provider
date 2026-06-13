//! omni-common
//! Shared, provider- and frontend-agnostic infrastructure.
//! Extracted/adapted from the original claude-code-provider common pieces.

pub mod auth;
pub mod conversation_log;
pub mod error;
pub mod http;
pub mod replacements;
pub mod responses;
pub mod session;
pub mod stats;
#[cfg(feature = "test-support")]
pub mod test_support;
pub mod time_util;

pub use auth::{ApiKeyId, auth_layer};
pub use conversation_log::{ConversationLog, DEFAULT_LOG_BACKUPS, DEFAULT_LOG_MAX_BYTES};
pub use error::AppError;
pub use http::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, from_canonical,
    sse_from_canonical_stream, to_canonical, unix_now_secs,
};
pub use replacements::{Replacements, ReplacementsError};
pub use responses::{
    ResponsesRequest, ResponsesResponse, responses_from_canonical, responses_to_canonical,
    sse_from_canonical_stream_responses,
};
pub use stats::{ActiveRequestGuard, Stats, StatsSnapshot, TokenUsage};
