//! omni-common
//! Shared, provider- and frontend-agnostic infrastructure.
//! Extracted/adapted from the original claude-code-provider common pieces.

pub mod anthropic;
pub mod anthropic_native_stats;
pub mod auth;
pub mod cache;
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
pub mod timeouts;

pub use anthropic::{
    AnthropicMapError, AnthropicProtocolError, anthropic_to_canonical, canonical_to_anthropic,
    encode_tool_result_content, parse_anthropic_object_no_dup_keys, peek_model_string,
    sse_from_canonical_stream_anthropic,
};
pub use anthropic_native_stats::{
    accumulate_anthropic_stream_usage, is_anthropic_content_delta,
    token_usage_from_anthropic_response,
};
pub use auth::{ApiKeyId, auth_layer};
pub use conversation_log::{ConversationLog, DEFAULT_LOG_BACKUPS, DEFAULT_LOG_MAX_BYTES};
pub use env::{env_nonempty, headers_from_env, parse_custom_headers};
pub use error::{AppError, classify_upstream};
pub use http::{
    ChatCompletionRequest, ChatCompletionResponse, ChatContentPart, ChatImageUrl, ChatMessage,
    ChatMessageContent, MAX_REASONING_EFFORT_LEN, from_canonical, is_allowed_chat_finish_reason,
    is_sse_content_type, sse_from_canonical_stream, to_canonical, to_canonical_with_headers,
    unix_now_secs, validate_reasoning_effort_lexical,
};
pub use oauth_refresh::{
    MAX_CREDENTIAL_RECOVERY_TURNS, NEAR_EXPIRY_SKEW_MS, NEAR_EXPIRY_SKEW_SECS,
    OAuthRefreshProvider, credential_lock_path, global_oauth_refresh_enabled,
    looks_like_refresh_token_spent, oauth_refresh_enabled_for, oauth_refresh_policy_summary,
    parse_bool_env, with_oauth_refresh_lock,
};
pub use replacements::{Replacements, ReplacementsError};
pub use responses::{
    ResponsesRequest, ResponsesResponse, responses_from_canonical, responses_to_canonical,
    sse_from_canonical_stream_responses,
};
pub use stats::{ActiveRequestGuard, Stats, StatsSnapshot, TokenUsage};
pub use timeouts::{
    OAUTH_REQUEST_TIMEOUT, UPSTREAM_CONNECT_TIMEOUT, UPSTREAM_ERROR_BODY_PREFIX,
    UPSTREAM_ERROR_BODY_READ_TIMEOUT, UPSTREAM_HEADERS_TIMEOUT, UPSTREAM_REQUEST_TIMEOUT,
    map_upstream_headers_wait, timeout_upstream_headers,
};
