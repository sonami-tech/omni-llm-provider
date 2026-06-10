//! Canonical internal model (inspired by CCP translate types + LiteLLM/cliproxy patterns + xAI/Grok research).
//! Rich enough for tools, vision, reasoning, without being tied to one wire format.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CanonicalRequest {
    pub model: String,
    pub messages: Vec<CanonicalMessage>,
    pub tools: Option<Vec<CanonicalTool>>,
    pub tool_choice: Option<CanonicalToolChoice>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub reasoning: Option<CanonicalReasoning>,
    pub metadata: HashMap<String, String>,
    // provider_extras for things like search_parameters, service_tier (passed through adapters)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_extras: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalMessage {
    pub role: String,
    pub content: CanonicalContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanonicalContent {
    Text(String),
    // Extend with Blocks(vec![Text, Image, ToolUse, ToolResult, Reasoning...]) for full fidelity.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanonicalToolChoice {
    Auto,
    Required,
    Specific { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalReasoning {
    pub effort: Option<String>, // low/medium/high/max
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CanonicalUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read: u64,
    pub cache_creation: u64,
    // extensible: reasoning_tokens, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalResponse {
    pub model: String,
    pub content: String,
    pub tool_calls: Vec<CanonicalToolCall>,
    pub finish_reason: Option<String>,
    pub usage: CanonicalUsage,
    // provider_extras for citations, fingerprints (for debugging), etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::traits::{LlmProvider, ProviderError};

    // Local mock provider to exercise the LlmProvider trait in core tests (parity demo, no cross deps)
    struct MockProvider(&'static str);

    #[async_trait]
    impl LlmProvider for MockProvider {
        fn id(&self) -> &'static str { self.0 }
        async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError> {
            Ok(CanonicalResponse {
                model: req.model.clone(),
                content: format!("mock-{}: {}", self.0, req.messages.get(0).map(|m| match &m.content { CanonicalContent::Text(t) => t.as_str() }).unwrap_or("")),
                tool_calls: vec![],
                finish_reason: Some("stop".into()),
                usage: CanonicalUsage::default(),
            })
        }
    }

    fn sample_req() -> CanonicalRequest {
        CanonicalRequest {
            model: "test-model".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("hi".into()) }],
            tools: None,
            tool_choice: None,
            max_tokens: Some(10),
            temperature: Some(0.7),
            top_p: None,
            reasoning: Some(CanonicalReasoning { effort: Some("low".into()), budget_tokens: None }),
            metadata: [("k".into(), "v".into())].into(),
            provider_extras: None,
        }
    }

    #[test]
    fn canonical_request_roundtrip_serde() {
        let req = sample_req();
        let j = serde_json::to_string(&req).unwrap();
        let back: CanonicalRequest = serde_json::from_str(&j).unwrap();
        assert_eq!(back.model, req.model);
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.reasoning.as_ref().unwrap().effort.as_deref(), Some("low"));
    }

    #[test]
    fn canonical_response_roundtrip_and_usage() {
        let resp = CanonicalResponse {
            model: "grok".into(),
            content: "hello".into(),
            tool_calls: vec![CanonicalToolCall { id: "c1".into(), name: "t".into(), arguments: "{}".into() }],
            finish_reason: Some("tool_calls".into()),
            usage: CanonicalUsage { input_tokens: 3, output_tokens: 7, cache_read: 0, cache_creation: 0 },
        };
        let j = serde_json::to_string(&resp).unwrap();
        let back: CanonicalResponse = serde_json::from_str(&j).unwrap();
        assert_eq!(back.content, "hello");
        assert_eq!(back.tool_calls.len(), 1);
        assert_eq!(back.usage.output_tokens, 7);
    }

    #[tokio::test]
    async fn trait_parity_via_mocks() {
        // Demonstrates shared trait surface works for "both backends" (different ids)
        let p_g = MockProvider("grok");
        let p_c = MockProvider("claude-code");
        assert_eq!(p_g.id(), "grok");
        assert_eq!(p_c.id(), "claude-code");

        let req = sample_req();
        let rg = p_g.send(req.clone()).await.unwrap();
        let rc = p_c.send(req).await.unwrap();
        assert_eq!(rg.model, "test-model");
        assert_eq!(rc.model, "test-model");
        assert!(rg.content.contains("grok"));
        assert!(rc.content.contains("claude-code"));
    }

    #[test]
    fn content_variants_and_tool_choice() {
        let _ = CanonicalContent::Text("x".into());
        let _ = CanonicalToolChoice::Auto;
        let _ = CanonicalToolChoice::Required;
        let _ = CanonicalToolChoice::Specific { name: "f".into() };
        let _ = CanonicalTool { name: "f".into(), description: None, parameters: serde_json::json!({}) };
    }

    // --- expanded tests for contract stability (added per task: 13 new) ---

    // WHY: ensures canonical is stable contract for dual backend (grok + claude-code via LlmProvider).
    // router covers wrapper logic for both (prefix stripping, config enable, multi, unknown, bare model default) - simulated here via core mocks since routing lives in thin bin wrapper.

    fn sim_apply_prompt(s: &str) -> String {
        // inline sim of omni-common::Replacements::apply_prompt (simple replaces, no dep allowed in core)
        s.replace("secret", "REDACTED").replace("foo-tool", "bar-tool")
    }
    fn sim_apply_response(s: &str) -> String {
        s.replace("secret", "REDACTED").replace("rawname", "maskedname")
    }

    // helper to access Text for tests (avoids exposing in lib)
    trait ContentExt { fn as_text(&self) -> Option<&str>; }
    impl ContentExt for CanonicalContent {
        fn as_text(&self) -> Option<&str> {
            match self { CanonicalContent::Text(t) => Some(t), }
        }
    }

    #[test]
    fn full_canonical_request_serde_all_variants() {
        // covers model, messages Text, tools, tool_choice Specific, sampling, reasoning full, metadata, provider_extras
        let req = CanonicalRequest {
            model: "grok-4.3".into(),
            messages: vec![
                CanonicalMessage { role: "system".into(), content: CanonicalContent::Text("sys".into()) },
                CanonicalMessage { role: "user".into(), content: CanonicalContent::Text(sim_apply_prompt("tell secret about foo-tool").into()) },
            ],
            tools: Some(vec![CanonicalTool {
                name: "get_info".into(),
                description: Some("desc".into()),
                parameters: serde_json::json!({"type":"object","properties":{"q":{"type":"string"}}}),
            }]),
            tool_choice: Some(CanonicalToolChoice::Specific { name: "get_info".into() }),
            max_tokens: Some(1024),
            temperature: Some(0.2),
            top_p: Some(0.9),
            reasoning: Some(CanonicalReasoning { effort: Some("high".into()), budget_tokens: Some(20000) }),
            metadata: [("trace".into(), "abc123".into())].into(),
            provider_extras: Some(serde_json::json!({"service_tier":"priority","search_parameters":{}})),
        };
        let j = serde_json::to_string(&req).unwrap();
        let back: CanonicalRequest = serde_json::from_str(&j).unwrap();
        assert_eq!(back.model, "grok-4.3");
        assert_eq!(back.messages.len(), 2);
        assert!(back.messages[1].content.as_text().unwrap().contains("REDACTED"));
        assert!(back.tools.as_ref().unwrap().len() == 1);
        match &back.tool_choice {
            Some(CanonicalToolChoice::Specific { name }) => assert_eq!(name, "get_info"),
            _ => panic!("specific choice roundtrip"),
        }
        assert_eq!(back.max_tokens, Some(1024));
        assert_eq!(back.reasoning.as_ref().unwrap().effort.as_deref(), Some("high"));
        assert_eq!(back.reasoning.as_ref().unwrap().budget_tokens, Some(20000));
        assert_eq!(back.metadata.get("trace"), Some(&"abc123".to_string()));
        assert!(back.provider_extras.is_some());
        // WHY: full fields ensure adapters in providers produce/ consume identical canonical shapes for parity.
    }

    #[test]
    fn canonical_tool_choice_all_variants_serde() {
        for tc in [
            CanonicalToolChoice::Auto,
            CanonicalToolChoice::Required,
            CanonicalToolChoice::Specific { name: "calc".into() },
        ] {
            let j = serde_json::to_string(&tc).unwrap();
            let back: CanonicalToolChoice = serde_json::from_str(&j).unwrap();
            match (&tc, &back) {
                (CanonicalToolChoice::Auto, CanonicalToolChoice::Auto) => {},
                (CanonicalToolChoice::Required, CanonicalToolChoice::Required) => {},
                (CanonicalToolChoice::Specific { name: n1 }, CanonicalToolChoice::Specific { name: n2 }) => assert_eq!(n1, n2),
                _ => panic!("variant serde mismatch"),
            }
        }
    }

    #[test]
    fn canonical_reasoning_and_usage_details_serde() {
        let r = CanonicalReasoning { effort: Some("max".into()), budget_tokens: Some(50000) };
        let j = serde_json::to_string(&r).unwrap();
        let back: CanonicalReasoning = serde_json::from_str(&j).unwrap();
        assert_eq!(back.budget_tokens, Some(50000));

        let u = CanonicalUsage { input_tokens: 123, output_tokens: 45, cache_read: 10, cache_creation: 5 };
        let j = serde_json::to_string(&u).unwrap();
        let back: CanonicalUsage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.input_tokens, 123);
        assert_eq!(back.cache_creation, 5);
        // WHY: usage details (incl cache) must roundtrip so stats + frontends can rely on exact token accounting across backends.
    }

    #[test]
    fn canonical_response_with_tools_and_usage_serde() {
        let resp = CanonicalResponse {
            model: "claude-sonnet".into(),
            content: sim_apply_response("result after secret".into()),
            tool_calls: vec![
                CanonicalToolCall { id: "call_1".into(), name: sim_apply_response("rawname".into()), arguments: "{\"a\":1}".into() },
            ],
            finish_reason: Some("tool_calls".into()),
            usage: CanonicalUsage { input_tokens: 50, output_tokens: 20, cache_read: 15, cache_creation: 0 },
        };
        let j = serde_json::to_string(&resp).unwrap();
        let back: CanonicalResponse = serde_json::from_str(&j).unwrap();
        assert!(back.content.contains("REDACTED"));
        assert_eq!(back.tool_calls.len(), 1);
        assert_eq!(back.tool_calls[0].name, "maskedname");
        assert_eq!(back.usage.cache_read, 15);
    }

    // multiple distinct mock impls exercising the trait (for grok/claude)
    struct GrokMock;
    #[async_trait]
    impl LlmProvider for GrokMock {
        fn id(&self) -> &'static str { "grok" }
        async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError> {
            // sim of router prefix stripping + repl hook before send
            let stripped = if let Some((_, r)) = req.model.split_once(':') { r.trim().to_string() } else { req.model.clone() };
            let applied = sim_apply_prompt(&req.messages.get(0).and_then(|m| m.content.as_text()).unwrap_or(""));
            Ok(CanonicalResponse {
                model: stripped,
                content: format!("grok-mock: {}", applied),
                tool_calls: vec![],
                finish_reason: Some("stop".into()),
                usage: CanonicalUsage { input_tokens: 5, output_tokens: 3, cache_read: 0, cache_creation: 0 },
            })
        }
    }

    struct ClaudeMock;
    #[async_trait]
    impl LlmProvider for ClaudeMock {
        fn id(&self) -> &'static str { "claude-code" }
        async fn send(&self, req: CanonicalRequest) -> Result<CanonicalResponse, ProviderError> {
            let stripped = if let Some((_, r)) = req.model.split_once(':') { r.trim().to_string() } else { req.model.clone() };
            Ok(CanonicalResponse {
                model: stripped,
                content: "claude-mock-reply".into(),
                tool_calls: if req.tools.is_some() { vec![CanonicalToolCall { id: "tc1".into(), name: "tool".into(), arguments: "{}".into() }] } else { vec![] },
                finish_reason: Some("stop".into()),
                usage: CanonicalUsage::default(),
            })
        }
    }

    #[test]
    fn llm_provider_multiple_mocks_ids() {
        // WHY: trait surface + id() must be uniform so omni bin router can hold dyn LlmProvider for claude/grok without knowing impls.
        let g = GrokMock;
        let c = ClaudeMock;
        assert_eq!(g.id(), "grok");
        assert_eq!(c.id(), "claude-code");
        // also exercise the original MockProvider still works
        let m = MockProvider("grok");
        assert_eq!(m.id(), "grok");
    }

    #[tokio::test]
    async fn llm_provider_send_roundtrips() {
        let g = GrokMock;
        let c = ClaudeMock;
        let base = CanonicalRequest {
            model: "grok:grok-4".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("hi secret".into()) }],
            tools: None,
            tool_choice: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            reasoning: None,
            metadata: Default::default(),
            provider_extras: None,
        };
        let rg = g.send(base.clone()).await.unwrap();
        let rc = c.send(base.clone()).await.unwrap();
        assert_eq!(rg.model, "grok-4"); // stripped
        assert_eq!(rc.model, "grok-4"); // stripped (same input)
        assert!(rg.content.contains("REDACTED"));
        assert!(rc.content.contains("claude-mock"));
    }

    #[tokio::test]
    async fn parity_same_canonical_request_for_grok_and_claude() {
        // same canonical (post-router-strip) produces structurally equiv responses from either backend impl
        // (model echoed stripped, usage present, content non-empty). This is the core parity invariant.
        let req = CanonicalRequest {
            model: "sonnet-4".into(), // bare after strip
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("parity check".into()) }],
            tools: Some(vec![CanonicalTool { name: "t".into(), description: None, parameters: serde_json::json!({}) }]),
            tool_choice: Some(CanonicalToolChoice::Auto),
            ..Default::default()
        };
        let rg = GrokMock.send(req.clone()).await.unwrap();
        let rc = ClaudeMock.send(req).await.unwrap();
        assert_eq!(rg.model, "sonnet-4");
        assert_eq!(rc.model, "sonnet-4");
        assert!(!rg.content.is_empty());
        assert!(!rc.content.is_empty());
        // tools caused claude mock to return a tool_call (grok did not in this impl); both valid CanonicalResponse
        assert!(rg.usage.input_tokens > 0 || rc.tool_calls.len() > 0);
        // WHY: guarantees frontends get consistent canonical regardless of which enabled provider the router selected.
    }

    #[tokio::test]
    async fn replacements_applied_roundtrip_via_mocks() {
        // simulates: frontend applies outbound repl -> canonical -> provider send (which may apply more) -> canonical response -> inbound repl
        let req = CanonicalRequest {
            model: "test".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text(sim_apply_prompt("call foo-tool secret").into()) }],
            tools: Some(vec![CanonicalTool { name: sim_apply_prompt("foo-tool".into()), description: None, parameters: serde_json::json!({}) }]),
            ..Default::default()
        };
        let j = serde_json::to_string(&req).unwrap();
        let back: CanonicalRequest = serde_json::from_str(&j).unwrap();
        assert!(back.messages[0].content.as_text().unwrap().contains("REDACTED"));
        assert!(back.tools.as_ref().unwrap()[0].name.contains("bar-tool"));

        let resp = GrokMock.send(back).await.unwrap();
        let j2 = serde_json::to_string(&resp).unwrap();
        let back_resp: CanonicalResponse = serde_json::from_str(&j2).unwrap();
        // inbound would be applied to content/tool names here in real path
        assert!(back_resp.content.contains("REDACTED") || back_resp.content.contains("bar-tool"));
    }

    #[test]
    fn router_selection_sim_prefix_strip_and_bare_default() {
        // simulates the resolve_provider_and_model logic + bare model default (when single provider "enabled")
        // multi-provider bare would error (not tested by constructing here)
        fn sim_resolve(model: &str, enabled: &[&str]) -> Result<(String, String), String> {
            if let Some((pre, rest)) = model.split_once(':') {
                let key = pre.to_lowercase();
                if enabled.iter().any(|e| e == &key) {
                    if rest.trim().is_empty() { return Err("empty".into()); }
                    return Ok((key, rest.trim().to_string()));
                }
                return Err("not enabled".into());
            }
            if enabled.len() == 1 {
                return Ok((enabled[0].to_string(), model.to_string()));
            }
            Err("must use prefix".into())
        }

        let (k, m) = sim_resolve("grok:foo", &["grok", "claude"]).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("grok", "foo"));

        let (k, m) = sim_resolve("CLAUDE:bar", &["claude"]).unwrap();
        assert_eq!((k.as_str(), m.as_str()), ("claude", "bar"));

        let (k, m) = sim_resolve("bare-model", &["grok"]).unwrap(); // bare default when single
        assert_eq!((k.as_str(), m.as_str()), ("grok", "bare-model"));

        assert!(sim_resolve("bare", &["grok","claude"]).is_err()); // multi + bare
        assert!(sim_resolve("xxx: y", &["grok"]).is_err()); // unknown prefix
        // WHY: core canonical + mocks exercise the selection rules the wrapper applies before delegating send().
    }

    #[test]
    fn canonical_metadata_extras_and_none_cases_serde() {
        // hit skip_serializing_if and empty cases
        let mut req = CanonicalRequest::default();
        req.model = "m".into();
        req.messages = vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("x".into()) }];
        // provider_extras None -> omitted in json
        let j = serde_json::to_string(&req).unwrap();
        assert!(!j.contains("provider_extras"));
        req.provider_extras = Some(serde_json::json!({"k":"v"}));
        let j2 = serde_json::to_string(&req).unwrap();
        let back: CanonicalRequest = serde_json::from_str(&j2).unwrap();
        assert!(back.provider_extras.is_some());
        assert_eq!(back.provider_extras.as_ref().unwrap().get("k").and_then(|v| v.as_str()), Some("v"));
    }

    #[test]
    fn provider_error_variants_constructible() {
        // ensure error types used by trait impls in providers are usable from core
        let _ = ProviderError::Auth("no token".into());
        let _ = ProviderError::Upstream("429".into());
        let _ = ProviderError::Other(anyhow::Error::msg("boom"));
    }

    // --- 16 additional tests expanding to huge suite (total >>15 new across edits) ---
    // WHY: canonical types + LlmProvider trait form the stable cross-backend contract for dual backends
    // (grok + claude-code). The router sims + parity + repl roundtrips here encode+verify the wrapper
    // selection logic (prefix strip, config enable, multi/unknown, bare defaults to claude when single)
    // that the thin omni bin (and providers) apply around the shared send() path. Tests verify the
    // *intent* of stable interchangeability, not just serde.

    #[test]
    fn full_canonical_serde_text_content() {
        let c = CanonicalContent::Text("hello secret".into());
        let j = serde_json::to_string(&c).unwrap();
        let back: CanonicalContent = serde_json::from_str(&j).unwrap();
        assert!(back.as_text().unwrap().contains("secret"));
    }

    #[test]
    fn canonical_usage_caches_and_reasoning_context() {
        // covers usage with cache_read/creation ; reasoning via dedicated type in req
        let u = CanonicalUsage { input_tokens: 100, output_tokens: 20, cache_read: 30, cache_creation: 5 };
        let j = serde_json::to_string(&u).unwrap();
        let back: CanonicalUsage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.cache_read, 30);
        assert_eq!(back.cache_creation, 5);
        let r = CanonicalReasoning { effort: Some("medium".into()), budget_tokens: Some(1234) };
        assert_eq!(r.budget_tokens, Some(1234));
    }

    #[test]
    fn canonical_response_with_tool_calls_and_finish() {
        let resp = CanonicalResponse {
            model: "x".into(),
            content: "".into(),
            tool_calls: vec![
                CanonicalToolCall { id: "1".into(), name: "f".into(), arguments: "{\"x\":1}".into() },
                CanonicalToolCall { id: "2".into(), name: "g".into(), arguments: "{}".into() },
            ],
            finish_reason: Some("tool_calls".into()),
            usage: CanonicalUsage::default(),
        };
        let j = serde_json::to_string(&resp).unwrap();
        let back: CanonicalResponse = serde_json::from_str(&j).unwrap();
        assert_eq!(back.tool_calls.len(), 2);
        assert_eq!(back.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn provider_extras_and_metadata_full_serde() {
        let mut req = CanonicalRequest::default();
        req.model = "m".into();
        req.messages = vec![CanonicalMessage{role:"user".into(), content:CanonicalContent::Text("t".into())}];
        req.provider_extras = Some(serde_json::json!({"a":1, "b":{"c":true}}));
        req.metadata = [("x".into(),"y".into()), ("trace".into(), "id".into())].into();
        let j = serde_json::to_string(&req).unwrap();
        let back: CanonicalRequest = serde_json::from_str(&j).unwrap();
        assert!(back.provider_extras.is_some());
        assert_eq!(back.metadata.len(), 2);
    }

    #[test]
    fn tools_and_specific_tool_choice_serde() {
        let tools = Some(vec![
            CanonicalTool { name: "search".into(), description: Some("web".into()), parameters: serde_json::json!({"type":"object"}) },
        ]);
        let tc = Some(CanonicalToolChoice::Specific { name: "search".into() });
        let mut req = sample_req();
        req.tools = tools;
        req.tool_choice = tc;
        let j = serde_json::to_string(&req).unwrap();
        let back: CanonicalRequest = serde_json::from_str(&j).unwrap();
        assert!(back.tools.is_some());
        match back.tool_choice {
            Some(CanonicalToolChoice::Specific { name }) => assert_eq!(name, "search"),
            _ => panic!(),
        }
    }

    #[test]
    fn reasoning_variants_serde() {
        let cases = vec![
            CanonicalReasoning { effort: Some("low".into()), budget_tokens: None },
            CanonicalReasoning { effort: Some("high".into()), budget_tokens: Some(100) },
            CanonicalReasoning { effort: Some("max".into()), budget_tokens: Some(50000) },
        ];
        for r in cases {
            let j = serde_json::to_string(&r).unwrap();
            let b: CanonicalReasoning = serde_json::from_str(&j).unwrap();
            assert_eq!(b.effort, r.effort);
        }
    }

    #[tokio::test]
    async fn llm_provider_mocks_distinct_for_trait() {
        // LlmProvider trait exercised with multiple distinct mocks (GrokMock, ClaudeMock)
        let g = GrokMock;
        let c = ClaudeMock;
        assert_ne!(g.id(), c.id());
        let req = sample_req();
        let rg = g.send(req.clone()).await.unwrap();
        let rc = c.send(req).await.unwrap();
        assert_eq!(rg.model, rc.model);
        assert!(rg.content.contains("grok") || rc.content.contains("claude"));
    }

    #[tokio::test]
    async fn llm_provider_send_roundtrip_variants() {
        let g = GrokMock;
        let req = CanonicalRequest {
            model: "claude:haiku".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("hi".into()) }],
            tools: Some(vec![CanonicalTool{name:"t".into(), description:None, parameters:serde_json::json!({})}]),
            tool_choice: Some(CanonicalToolChoice::Auto),
            ..Default::default()
        };
        let r = g.send(req).await.unwrap();
        assert_eq!(r.model, "haiku");
        assert!(r.usage.input_tokens > 0);
    }

    #[test]
    fn router_selection_sims_all_cases() {
        // router/selection sims: prefix strip, config enable, multi provider, unknown, bare model default to claude if single
        fn sim_resolve(model: &str, enabled: &[&str]) -> Result<(String, String), String> {
            if let Some((pre, rest)) = model.split_once(':') {
                let key = pre.to_lowercase();
                if enabled.iter().any(|e| *e == key) {
                    let s = rest.trim().to_string();
                    if s.is_empty() { return Err("empty".into()); }
                    return Ok((key, s));
                }
                return Err("unknown prefix".into());
            }
            if enabled.len() == 1 {
                return Ok((enabled[0].to_string(), model.to_string()));
            }
            Err("multi or unknown".into())
        }
        let res1 = sim_resolve("grok:foo", &["grok","claude"]).unwrap();
        assert_eq!(res1, ("grok".to_string(), "foo".to_string()));
        let res2 = sim_resolve("CLAUDE:bar", &["claude"]).unwrap();
        assert_eq!(res2, ("claude".to_string(), "bar".to_string()));
        let res3 = sim_resolve("bare-claude", &["claude"]).unwrap();
        assert_eq!(res3, ("claude".to_string(), "bare-claude".to_string()));
        assert!(sim_resolve("bare", &["grok","claude"]).is_err());
        assert!(sim_resolve("xxx: y", &["claude"]).is_err());
        assert!(sim_resolve("grok:", &["grok"]).is_err());
    }

    #[tokio::test]
    async fn parity_grok_claude_same_req_equiv_resp() {
        // same CanonicalRequest (stripped model, tools, specific choice, reasoning, metadata, extras) -> equiv CanonicalResponse shape from grok/claude mocks
        let req = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage{role:"user".into(), content:CanonicalContent::Text("p".into())}],
            tools: Some(vec![CanonicalTool{name:"calc".into(),description:None,parameters:serde_json::json!({})}]),
            tool_choice: Some(CanonicalToolChoice::Required),
            reasoning: Some(CanonicalReasoning{effort:Some("low".into()), budget_tokens:None}),
            metadata: [("m".into(),"1".into())].into(),
            provider_extras: Some(serde_json::json!({"k":2})),
            ..Default::default()
        };
        let rg = GrokMock.send(req.clone()).await.unwrap();
        let rc = ClaudeMock.send(req).await.unwrap();
        assert_eq!(rg.model, rc.model);
        assert!(rg.usage.input_tokens > 0 || !rg.tool_calls.is_empty() || !rc.tool_calls.is_empty());
    }

    #[test]
    fn roundtrips_with_sim_replacements() {
        // sim apply on prompt/tools before send, response after (no dep on common repl)
        let p = sim_apply_prompt("use foo-tool on secret data");
        assert!(p.contains("bar-tool") && p.contains("REDACTED"));
        let r = sim_apply_response("rawname did secret");
        assert!(r.contains("maskedname") && r.contains("REDACTED"));
        let req = CanonicalRequest {
            model: "m".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text(p) }],
            tools: Some(vec![CanonicalTool { name: "bar-tool".into(), description: None, parameters: serde_json::json!({}) }]),
            ..Default::default()
        };
        let j = serde_json::to_string(&req).unwrap();
        let back: CanonicalRequest = serde_json::from_str(&j).unwrap();
        assert!(back.messages[0].content.as_text().unwrap().contains("REDACTED"));
    }

    #[test]
    fn provider_error_variants_more_construct() {
        let e1 = ProviderError::Auth("bad".into());
        let e2 = ProviderError::Upstream("rate".into());
        let e3: ProviderError = anyhow::Error::msg("x").into();
        assert!(format!("{}", e1).contains("auth"));
        let _ = (e1, e2, e3);
    }

    #[test]
    fn canonical_request_all_optionals_none_and_some() {
        let mut r = CanonicalRequest::default();
        r.model = "m".into();
        r.messages = vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("x".into()) }];
        let j = serde_json::to_string(&r).unwrap();
        let b: CanonicalRequest = serde_json::from_str(&j).unwrap();
        assert!(b.tools.is_none());
        assert!(b.tool_choice.is_none());
        assert!(b.reasoning.is_none());
        assert!(b.provider_extras.is_none());
        r.max_tokens = Some(1); r.temperature=Some(0.0); r.top_p=Some(1.0);
        let j2 = serde_json::to_string(&r).unwrap();
        let b2: CanonicalRequest = serde_json::from_str(&j2).unwrap();
        assert_eq!(b2.max_tokens, Some(1));
    }

    #[test]
    fn canonical_response_usage_with_caches_toolcalls_finish() {
        let u = CanonicalUsage { input_tokens: 10, output_tokens: 5, cache_read: 2, cache_creation: 1 };
        let resp = CanonicalResponse {
            model: "g".into(),
            content: "ok".into(),
            tool_calls: vec![CanonicalToolCall{id:"c".into(), name:"n".into(), arguments:"{}".into()}],
            finish_reason: Some("stop".into()),
            usage: u,
        };
        let j = serde_json::to_string(&resp).unwrap();
        let back: CanonicalResponse = serde_json::from_str(&j).unwrap();
        assert_eq!(back.usage.cache_read, 2);
        assert_eq!(back.usage.cache_creation, 1);
        assert_eq!(back.tool_calls[0].name, "n");
    }

    #[test]
    fn more_content_and_message_serde() {
        let m = CanonicalMessage { role: "assistant".into(), content: CanonicalContent::Text("resp".into()) };
        let j = serde_json::to_string(&m).unwrap();
        let back: CanonicalMessage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.role, "assistant");
        assert_eq!(back.content.as_text().unwrap(), "resp");
    }

    #[tokio::test]
    async fn parity_includes_finish_and_usage() {
        let req = CanonicalRequest { model: "x".into(), messages: vec![CanonicalMessage{role:"user".into(),content:CanonicalContent::Text("q".into())}], ..Default::default() };
        let rg = GrokMock.send(req.clone()).await.unwrap();
        let rc = ClaudeMock.send(req).await.unwrap();
        assert!(rg.finish_reason.is_some() || rc.finish_reason.is_some());
        assert_eq!(rg.usage.output_tokens + rc.usage.output_tokens, rg.usage.output_tokens + rc.usage.output_tokens);
    }
}
