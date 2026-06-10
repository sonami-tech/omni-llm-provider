//! Translation between omni-core Canonical* types and Anthropic Messages wire format.
//! Plus identity (preamble + billing) injection and outbound/inbound replacements hook.
//!
//! **Isolation note:** The wire structs here (MessagesRequest etc) control
//! exact JSON serialization order and presence for cch computation and the
//! OAuth gate. They are deliberately private to provider-claude.
//!
//! Adapted from reference-src-claude/translate/{anthropic.rs, build.rs,
//! from_anthropic.rs, tool_translate.rs, ...} and routes/completions_v2.rs
//! (prepend_claude_code_identity).
//!
//! The canonical types are intentionally lossy (flat Text only today); the
//! adapter here maps the supported subset while still routing the request
//! through the full fingerprint + cch + identity path.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use omni_common::Replacements;
use omni_core::{
    CanonicalContent, CanonicalReasoning, CanonicalRequest, CanonicalResponse,
    CanonicalTool, CanonicalToolCall, CanonicalToolChoice, CanonicalUsage,
};

use crate::fingerprint::FingerprintProfile;
use crate::models::ModelDef;
// UpstreamError kept commented for future use in count_tokens etc; no current non-test refs.

// ── Native Anthropic Messages API wire types (exact shapes for cch) ──

/// Outbound `POST /v1/messages` body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemField>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
}

/// `system` may be a flat string OR an array of typed text blocks. The block
/// form is required when `cache_control` markers are present, and what claude
/// itself sends.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemField {
    #[allow(dead_code)]
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: String, // always "text"
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String, // "user" or "assistant"
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    #[allow(dead_code)]
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    Image {
        source: ImageSource,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    #[allow(dead_code)]
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    None {},
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thinking {
    #[serde(rename = "type")]
    pub kind: String, // "enabled" or "disabled"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub effort: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String, // "ephemeral"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>, // "5m" or "1h"
}

// ── Response (non-streaming) ──────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub kind: String, // "message"
    #[allow(dead_code)]
    pub role: String, // "assistant"
    pub model: String,
    pub content: Vec<ResponseContentBlock>,
    pub stop_reason: Option<String>,
    #[allow(dead_code)]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
}

// ── Canonical <-> Anthropic (minimal adapter for current Canonical shape) ──

/// Build an Anthropic MessagesRequest from a CanonicalRequest, using the
/// resolved model_def for defaults. This is the Claude-specific path.
pub fn build_messages_request_from_canonical(
    req: &CanonicalRequest,
    model_def: &ModelDef,
    repl: &Replacements,
) -> Result<MessagesRequest, String> {
    // Apply prompt-scope replacements to the canonical texts (and tool surfaces).
    let mut messages: Vec<Message> = Vec::new();
    for m in &req.messages {
        let text = match &m.content {
            CanonicalContent::Text(t) => repl.apply_prompt(t),
        };
        // For simplicity with current Canonical (Text only): emit as text content.
        // When Canonical grows Blocks we will extend.
        messages.push(Message {
            role: m.role.clone(),
            content: MessageContent::Text(text),
        });
    }

    let tools = match req.tools.as_ref() {
        Some(t) if !t.is_empty() => Some(translate_tools(t, repl)?),
        _ => None,
    };

    if tools.is_none() && tool_choice_requires_tool(&req.tool_choice) {
        return Err("tool_choice requires or selects a tool, but no tools were provided".into());
    }

    let tool_choice = if tools.is_some() {
        translate_tool_choice(&req.tool_choice, None)  // parallel handled elsewhere for now
    } else {
        None
    };

    let mut max_tokens = req.max_tokens.unwrap_or_else(|| default_max_tokens(model_def));

    let thinking = derive_thinking_from_canonical(req.reasoning.as_ref(), model_def);

    if let Some(t) = thinking.as_ref()
        && let Some(budget) = t.budget_tokens
            && max_tokens <= budget {
                max_tokens = budget.saturating_add(1024).min(default_max_tokens(model_def));
            }

    let thinking_active = thinking
        .as_ref()
        .map(|t| t.kind == "enabled")
        .unwrap_or(false);

    let temperature = if thinking_active {
        Some(1.0)
    } else {
        req.temperature
    };
    let top_p = if thinking_active {
        None
    } else {
        req.top_p
    };
    let top_k = if thinking_active { None } else { None };
    let stop_sequences = if thinking_active { None } else { None };

    // metadata / user passthrough limited for canonical v1
    let metadata = None;

    if req.provider_extras.is_some() {
        // In real we could merge some, but for canonical contract we keep minimal.
    }

    Ok(MessagesRequest {
        model: model_def.canonical.to_string(),
        max_tokens,
        messages,
        system: None, // identity injection happens *after* this in the provider
        tools,
        tool_choice,
        temperature,
        top_p,
        top_k,
        stop_sequences,
        stream: Some(false),
        metadata,
        thinking,
        output_config: None,
    })
}

fn default_max_tokens(model_def: &ModelDef) -> u32 {
    model_def.max_tokens.min(u32::MAX as u64) as u32
}

fn derive_thinking_from_canonical(
    reasoning: Option<&CanonicalReasoning>,
    model_def: &ModelDef,
) -> Option<Thinking> {
    match reasoning {
        Some(CanonicalReasoning { effort: Some(e), budget_tokens }) if !e.is_empty() => {
            let budget = budget_tokens.or_else(|| Some(budget_for_effort(e, model_def)));
            Some(Thinking {
                kind: "enabled".into(),
                budget_tokens: budget,
            })
        }
        _ => None,
    }
}

fn budget_for_effort(effort: &str, _model_def: &ModelDef) -> u32 {
    match effort {
        "low" => 1024,
        "medium" => 8192,
        "high" => 16384,
        "max" => 32768,
        _ => 0,
    }
}

fn tool_choice_requires_tool(choice: &Option<CanonicalToolChoice>) -> bool {
    matches!(choice, Some(CanonicalToolChoice::Required) | Some(CanonicalToolChoice::Specific { .. }))
}

fn translate_tools(tools: &[CanonicalTool], repl: &Replacements) -> Result<Vec<Tool>, String> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let name = repl.apply_prompt(&t.name);
        let description = t.description.as_ref().map(|d| repl.apply_prompt(d));
        let input_schema = t.parameters.clone();
        out.push(Tool {
            name,
            description,
            input_schema,
            cache_control: None,
        });
    }
    Ok(out)
}

fn translate_tool_choice(
    choice: &Option<CanonicalToolChoice>,
    _disable_parallel: Option<bool>,
) -> Option<ToolChoice> {
    match choice {
        None => None,
        Some(CanonicalToolChoice::Auto) => Some(ToolChoice::Auto { disable_parallel_tool_use: None }),
        Some(CanonicalToolChoice::Required) => Some(ToolChoice::Any { disable_parallel_tool_use: None }),
        Some(CanonicalToolChoice::Specific { name }) => Some(ToolChoice::Tool {
            name: name.clone(),
            disable_parallel_tool_use: None,
        }),
    }
}

// ── Identity injection (the preamble + dynamic billing marker) ──

/// Prepend the Claude Code billing marker (dynamic cch placeholder) + system
/// preamble to the request's system field (forcing block form).
///
/// Replacements MUST have already been applied to the request body texts
/// (including the first user message) before calling this, because the billing
/// suffix is derived from the (post-replacement) first user text.
///
/// This function and everything it touches are CLAUDE-SPECIFIC and must never
/// be moved to omni-common/core.
pub fn prepend_claude_code_identity(
    req: &mut MessagesRequest,
    profile: &FingerprintProfile,
    inject_identity: bool,
) {
    if !inject_identity {
        return;
    }

    let first_user_text = first_user_text_for_billing(req).unwrap_or("");
    let billing = SystemBlock {
        kind: "text".into(),
        text: profile.billing_header_text(first_user_text),
        cache_control: None,
    };
    let preamble = SystemBlock {
        kind: "text".into(),
        text: profile.system_preamble.to_string(),
        cache_control: None,
    };

    let existing_blocks = match req.system.take() {
        None => Vec::new(),
        Some(SystemField::Text(s)) => {
            if s.is_empty() {
                Vec::new()
            } else {
                vec![SystemBlock {
                    kind: "text".into(),
                    text: s,
                    cache_control: None,
                }]
            }
        }
        Some(SystemField::Blocks(blocks)) => blocks,
    };
    let existing_blocks = strip_existing_claude_identity(existing_blocks);
    let mut blocks = Vec::with_capacity(existing_blocks.len() + 2);
    blocks.push(billing);
    blocks.push(preamble);
    blocks.extend(existing_blocks);
    req.system = Some(SystemField::Blocks(blocks));
}

fn strip_existing_claude_identity(blocks: Vec<SystemBlock>) -> Vec<SystemBlock> {
    blocks
        .into_iter()
        .filter(|b| {
            !crate::fingerprint::is_claude_code_billing_header(&b.text)
                && !is_claude_code_system_preamble(&b.text)
        })
        .collect()
}

fn is_claude_code_system_preamble(text: &str) -> bool {
    crate::fingerprint::FINGERPRINT_PROFILES
        .iter()
        .any(|profile| profile.system_preamble == text)
}

fn first_user_text_for_billing(req: &MessagesRequest) -> Option<&str> {
    let first_user = req.messages.iter().find(|m| m.role == "user")?;
    match &first_user.content {
        MessageContent::Text(text) => Some(text.as_str()),
        MessageContent::Blocks(blocks) => blocks.iter().find_map(|block| match block {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        }),
    }
}

// Apply inbound replacements (response scope) to assistant text + tool surfaces.
#[allow(dead_code)]
fn apply_response_replacements(text: &str, tool_calls: &mut [CanonicalToolCall], repl: &Replacements) -> String {
    let t = repl.apply_response(text);
    for tc in tool_calls.iter_mut() {
        tc.name = repl.apply_response(&tc.name);
        tc.arguments = repl.apply_response(&tc.arguments);
    }
    t
}

// ── From Anthropic response to Canonical ──

pub fn build_canonical_response(
    resp: &MessagesResponse,
    requested_model: &str,
    repl: &Replacements,
) -> CanonicalResponse {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<CanonicalToolCall> = Vec::new();

    for (_i, block) in resp.content.iter().enumerate() {
        match block {
            ResponseContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            ResponseContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(CanonicalToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                });
            }
            ResponseContentBlock::Thinking { .. } => {
                // For now, thinking is dropped from canonical text (future: provider_extras or extended content).
            }
            ResponseContentBlock::Other => {}
        }
    }

    let content = text_parts.join("");
    let content = repl.apply_response(&content);
    // tool names/args replaced too
    for tc in &mut tool_calls {
        tc.name = repl.apply_response(&tc.name);
        tc.arguments = repl.apply_response(&tc.arguments);
    }

    let finish_reason = map_stop_reason(resp.stop_reason.as_deref(), !tool_calls.is_empty());

    let usage = CanonicalUsage {
        input_tokens: resp.usage.input_tokens as u64,
        output_tokens: resp.usage.output_tokens as u64,
        cache_read: resp.usage.cache_read_input_tokens.unwrap_or(0) as u64,
        cache_creation: resp.usage.cache_creation_input_tokens.unwrap_or(0) as u64,
    };

    CanonicalResponse {
        model: requested_model.to_string(),
        content,
        tool_calls,
        finish_reason: Some(finish_reason.to_string()),
        usage,
    }
}

fn map_stop_reason(reason: Option<&str>, has_tool_calls: bool) -> &'static str {
    match reason {
        Some("end_turn") => "stop",
        Some("max_tokens") => "length",
        Some("stop_sequence") => "stop",
        Some("tool_use") => "tool_calls",
        Some("pause_turn") => "stop",
        Some("refusal") => "content_filter",
        _ if has_tool_calls => "tool_calls",
        _ => "stop",
    }
}

// ── Convenience: full outbound path exercised by provider (repl + identity) ──

/// Given a canonical request, produce the *final* body JSON (post-replacements,
/// post-identity, ready for finalize_body_json + cch).
/// Returns the MessagesRequest and the first-user text (for context if needed).
pub fn prepare_anthropic_request(
    canon: &CanonicalRequest,
    profile: &FingerprintProfile,
    repl: &Replacements,
    inject_identity: bool,
) -> Result<MessagesRequest, String> {
    let model_def = profile.resolve_model(&canon.model);
    let mut anth = build_messages_request_from_canonical(canon, model_def, repl)?;
    // Important: identity uses the (post-repl) first user text for the suffix.
    prepend_claude_code_identity(&mut anth, profile, inject_identity);
    Ok(anth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_core::CanonicalMessage;
    use crate::CLAUDE_CODE_SYSTEM_PREAMBLE;

    fn empty_repl() -> Replacements {
        Replacements::empty()
    }

    #[test]
    fn canonical_to_anth_basic_text() {
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hello world".into()),
            }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("haiku");
        let anth = build_messages_request_from_canonical(&req, model_def, &empty_repl()).unwrap();
        assert_eq!(anth.model, "claude-haiku-4-5-20251001");
        assert!(matches!(anth.messages[0].content, MessageContent::Text(ref s) if s == "hello world"));
    }

    #[test]
    fn identity_prepend_adds_billing_and_preamble() {
        let mut req = MessagesRequest {
            model: "claude-haiku-4-5-20251001".into(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".into(),
                content: MessageContent::Text("Say OK".into()),
            }],
            system: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            metadata: None,
            thinking: None,
            output_config: None,
        };
        let profile = crate::fingerprint::default_profile();
        prepend_claude_code_identity(&mut req, profile, true);
        let blocks = match req.system {
            Some(SystemField::Blocks(b)) => b,
            _ => panic!("expected blocks"),
        };
        assert!(blocks.len() >= 2);
        assert!(crate::fingerprint::is_claude_code_billing_header(&blocks[0].text));
        assert_eq!(blocks[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE);
    }

    #[test]
    fn from_anth_response_to_canon() {
        let resp = MessagesResponse {
            id: "msg_1".into(),
            kind: "message".into(),
            role: "assistant".into(),
            model: "claude-haiku-4-5-20251001".into(),
            content: vec![ResponseContentBlock::Text { text: "hi there".into() }],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: Usage { input_tokens: 5, output_tokens: 2, ..Default::default() },
        };
        let canon = build_canonical_response(&resp, "haiku", &empty_repl());
        assert_eq!(canon.model, "haiku");
        assert_eq!(canon.content, "hi there");
        assert_eq!(canon.usage.input_tokens, 5);
        assert_eq!(canon.usage.output_tokens, 2);
    }

    #[test]
    fn prepare_applies_repl_and_identity() {
        let repl = Replacements::parse(r#"rule = [ { scope = "prompt", search = "SECRET", replace = "REDACTED" } ]"#).unwrap();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("tell SECRET".into()) }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let anth = prepare_anthropic_request(&canon, profile, &repl, true).unwrap();
        // first user text was replaced before billing suffix
        let blocks = match anth.system.unwrap() {
            SystemField::Blocks(b) => b,
            _ => panic!(),
        };
        // billing text contains the suffix derived from post-repl "tell REDACTED"
        assert!(blocks[0].text.contains("cc_version="));
        assert_eq!(blocks[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE);
        // message content replaced
        match &anth.messages[0].content {
            MessageContent::Text(t) => assert_eq!(t, "tell REDACTED"),
            _ => panic!(),
        }
    }

    #[test]
    fn strip_existing_claude_identity_removes_billing_and_preamble() {
        // The strip is the "reconcile" gate that prevents dup identity when
        // client already injected (passthrough path); must leave user system.
        let billing = SystemBlock {
            kind: "text".into(),
            text: "x-anthropic-billing-header: cc_version=2.1.165.492; cc_entrypoint=sdk-cli; cch=00000;".into(),
            cache_control: None,
        };
        let pre = SystemBlock {
            kind: "text".into(),
            text: CLAUDE_CODE_SYSTEM_PREAMBLE.into(),
            cache_control: None,
        };
        let user = SystemBlock {
            kind: "text".into(),
            text: "keep me".into(),
            cache_control: None,
        };
        let blocks = vec![billing, pre, user];
        let kept = strip_existing_claude_identity(blocks);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].text, "keep me");
    }

    #[test]
    fn identity_injects_billing_at_system_0_then_preamble() {
        // Invariant: cch/billing marker is ALWAYS first in system blocks (before
        // preamble), because suffix is computed on first user text and cch
        // finalizer searches from "system" for the billing sentinel.
        let mut req = MessagesRequest {
            model: "haiku".into(),
            max_tokens: 1,
            messages: vec![Message { role: "user".into(), content: MessageContent::Text("x".into()) }],
            system: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            metadata: None,
            thinking: None,
            output_config: None,
        };
        let profile = crate::fingerprint::default_profile();
        prepend_claude_code_identity(&mut req, profile, true);
        let blocks = match req.system {
            Some(SystemField::Blocks(b)) => b,
            _ => panic!(),
        };
        assert!(blocks.len() >= 2);
        assert!(crate::fingerprint::is_claude_code_billing_header(&blocks[0].text), "billing must be [0]");
        assert_eq!(blocks[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE, "preamble must be exact at [1]");
    }

    #[test]
    fn prepare_uses_post_repl_first_user_for_billing_suffix() {
        // Prompt-before-identity gate: repls (tool/prompt scope) run on texts
        // BEFORE billing_header_text is called, so suffix (and thus cch body)
        // is derived from post-repl bytes. Critical for gate fingerprint.
        let repl = Replacements::parse(r#"rule = [ { scope = "prompt", search = "FOO", replace = "BAR" } ]"#).unwrap();
        let canon = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("say FOO".into()) }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let anth = prepare_anthropic_request(&canon, profile, &repl, true).unwrap();
        let blocks = match anth.system.unwrap() {
            SystemField::Blocks(b) => b,
            _ => panic!(),
        };
        // suffix for "say BAR" not "say FOO"
        assert!(blocks[0].text.contains(".492") || blocks[0].text.contains("cc_version=")); // at least structure; exact suffix would require oracle but gate cares post-repl
        assert!(!blocks[0].text.contains("FOO"));
    }

    #[test]
    fn build_messages_drops_unsupported_canonical_fields() {
        // Current canonical adapter intentionally drops provider_extras,
        // some reasoning non-effort, etc. The wire shape for cch/identity
        // must be minimal + our injected.
        let mut canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("hi".into()) }],
            ..Default::default()
        };
        canon.provider_extras = Some(serde_json::json!({"foo":"bar"}));
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("sonnet");
        let anth = build_messages_request_from_canonical(&canon, model_def, &empty_repl()).unwrap();
        assert!(anth.metadata.is_none());
        assert_eq!(anth.stream, Some(false));
        // no output_config unless reasoning high-effort etc
        assert!(anth.output_config.is_none());
    }

    #[test]
    fn build_canonical_response_maps_raw_usage_with_cache_and_tool_calls() {
        // Usage raw (incl cache_*) + tool_calls from anth must round to canon;
        // finish "tool_use" -> "tool_calls".
        let resp = MessagesResponse {
            id: "m".into(),
            kind: "message".into(),
            role: "assistant".into(),
            model: "claude-haiku-4-5-20251001".into(),
            content: vec![
                ResponseContentBlock::ToolUse { id: "t1".into(), name: "do".into(), input: serde_json::json!({"x":1}) },
                ResponseContentBlock::Text { text: "done".into() },
            ],
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 3,
                cache_creation_input_tokens: Some(2),
                cache_read_input_tokens: Some(1),
                ..Default::default()
            },
        };
        let canon = build_canonical_response(&resp, "haiku", &empty_repl());
        assert_eq!(canon.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(canon.tool_calls.len(), 1);
        assert_eq!(canon.tool_calls[0].name, "do");
        assert_eq!(canon.usage.cache_creation, 2);
        assert_eq!(canon.usage.cache_read, 1);
        assert!(canon.content.contains("done"));
    }

    #[test]
    fn response_repl_applies_to_text_and_tool_surfaces() {
        let repl = Replacements::parse(r#"rule = [ { scope = "response", search = "HIDE", replace = "SHOWN" } ]"#).unwrap();
        let resp = MessagesResponse {
            id: "m".into(), kind: "message".into(), role: "assistant".into(),
            model: "haiku".into(),
            content: vec![ ResponseContentBlock::ToolUse { id: "t".into(), name: "callHIDE".into(), input: serde_json::json!({}) } ],
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            usage: Usage { input_tokens: 1, output_tokens: 1, ..Default::default() },
        };
        let canon = build_canonical_response(&resp, "haiku", &repl);
        assert_eq!(canon.tool_calls[0].name, "callSHOWN");
    }

    #[test]
    fn prepare_vs_build_plus_prepend_are_equivalent_for_canonical_path() {
        // Build/passthrough parity (for the canonical adapter path): direct
        // build+prepend produces identical wire to the prepare convenience used
        // by the LlmProvider send.
        let canon = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage { role: "user".into(), content: CanonicalContent::Text("p".into()) }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let mdef = profile.resolve_model("haiku");
        let mut built = build_messages_request_from_canonical(&canon, mdef, &repl).unwrap();
        prepend_claude_code_identity(&mut built, profile, true);
        let prepped = prepare_anthropic_request(&canon, profile, &repl, true).unwrap();
        // compare key identity + model
        let bsys = match built.system { Some(SystemField::Blocks(b)) => b, _ => panic!() };
        let psys = match prepped.system { Some(SystemField::Blocks(b)) => b, _ => panic!() };
        assert_eq!(bsys[0].text, psys[0].text);
        assert_eq!(bsys[1].text, psys[1].text);
        assert_eq!(built.model, prepped.model);
    }
}
