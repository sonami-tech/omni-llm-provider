//! Upstream HTTPS layer to api.anthropic.com.
//!
//! v2 talks directly to Anthropic Messages over HTTPS, mimicking the claude
//! CLI wire fingerprint. No subprocess. Credentials come from
//! `~/.claude/.credentials.json`, re-read per request.

pub mod client;
pub mod credentials;
pub mod errors;
pub mod fingerprint;
pub mod stream;

pub use client::UpstreamClient;
