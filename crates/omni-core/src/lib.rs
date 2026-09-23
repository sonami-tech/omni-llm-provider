//! omni-core
//! Canonical types and traits for pluggable frontends and providers.
//! The "connect anything to anything" glue. Minimal and stable.

pub mod bootstrap;
pub mod cache;
pub mod canonical;
pub mod native_anthropic;
pub mod tool_schema;
pub mod traits;
pub mod version;

pub use bootstrap::*;
pub use cache::*;
pub use canonical::*;
pub use native_anthropic::*;
pub use tool_schema::*;
pub use traits::*;
pub use version::*;
