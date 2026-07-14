//! omni-common
//! Shared, provider- and frontend-agnostic infrastructure.
//! Extracted/adapted from the original claude-code-provider common pieces.

pub mod anthropic;
pub mod auth;
pub mod canonical_mapping;
pub mod conversation_log;
pub mod env;
pub mod error;
pub mod http;
pub mod oauth_refresh;
pub mod replacements;
pub mod responses;
pub mod responses_upstream;
pub mod session;
pub mod span_stream;
pub mod stats;
#[cfg(feature = "test-support")]
pub mod test_support;

pub use anthropic::{
    AnthropicMapError, AnthropicProtocolError, anthropic_to_canonical, canonical_to_anthropic,
    encode_tool_result_content, parse_anthropic_object_no_dup_keys, peek_model_string,
    sse_from_canonical_stream_anthropic,
};
pub use auth::{ApiKeyId, auth_layer};
pub use conversation_log::{ConversationLog, DEFAULT_LOG_BACKUPS, DEFAULT_LOG_MAX_BYTES};
pub use env::{env_nonempty, headers_from_env, parse_custom_headers};
pub use error::{AppError, classify_upstream};
pub use http::{
    ChatCompletionRequest, ChatCompletionResponse, ChatContentPart, ChatImageUrl, ChatMessage,
    ChatMessageContent, from_canonical, sse_from_canonical_stream, to_canonical, unix_now_secs,
};
pub use oauth_refresh::{
    OAuthRefreshProvider, global_oauth_refresh_enabled, oauth_refresh_enabled_for,
    oauth_refresh_policy_summary, parse_bool_env,
};
pub use replacements::{Replacements, ReplacementsError};
pub use responses::{
    ResponsesRequest, ResponsesResponse, responses_from_canonical, responses_to_canonical,
    sse_from_canonical_stream_responses,
};
pub use stats::{ActiveRequestGuard, Stats, StatsSnapshot, TokenUsage};
