//! omni-core
//! Canonical types and traits for pluggable frontends and providers.
//! The "connect anything to anything" glue. Minimal and stable.

pub mod canonical;
pub mod traits;
pub mod version;

pub use canonical::*;
pub use traits::*;
pub use version::*;
