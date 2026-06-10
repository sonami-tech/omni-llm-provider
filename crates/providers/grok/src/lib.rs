//! providers/grok
//!
//! xAI / Grok provider.
//!
//! Expected to be much lighter than claude-code because xAI's API is already
//! largely OpenAI-compatible. Minimal or no special outbound mutation,
//! standard API key auth, simpler (or direct) translation from canonical.

pub struct GrokProvider {
    // api key, base url, model mapping, etc.
}

impl GrokProvider {
    pub fn new(/* ... */) -> Self {
        Self {}
    }
}
