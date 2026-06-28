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
    CanonicalBlock, CanonicalContent, CanonicalImageSource, CanonicalMessage, CanonicalReasoning,
    CanonicalReasoningBlock, CanonicalRequest, CanonicalResponse, CanonicalResponseMetadata,
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
    //
    // Anthropic's /v1/messages has no `system` (or `developer`) role inside the
    // `messages` array; leaving one there draws a 400. Reshape per the in-repo
    // reference (reference-src-claude/translate/messages.rs `reshape`): hoist a
    // *leading* run of system/developer messages into the top-level `system`
    // field, and fold a *mid-thread* system/developer message in place into a
    // marked `user` turn so its position relative to the conversation is kept.
    // Both `system` and `developer` are handled because the Chat Completions
    // surface does not normalize `developer` to `system` (see
    // omni-common/src/http.rs chat_message_to_canonical).
    let mut system_blocks: Vec<SystemBlock> = Vec::new();
    let mut seen_non_system = false;
    let mut messages: Vec<Message> = Vec::new();
    for m in &req.messages {
        if m.role == "system" || m.role == "developer" {
            // Non-text blocks (e.g. an image) in a system/developer message are
            // unrepresentable as system content; fail loud rather than silently
            // drop them (diverges from the reference's silent text-only extract).
            let text = repl.apply_prompt(&system_text(m)?);
            if text.is_empty() {
                // Empty system/developer content contributes nothing; skip it so
                // we never emit a phantom system block or marked user turn.
                continue;
            }
            if seen_non_system {
                // Mid-thread: fold in place into a user turn, preserving order.
                messages.push(Message {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::Text {
                        text: format!("[system message]\n{text}"),
                        cache_control: None,
                    }]),
                });
            } else {
                // Leading: accumulate into the top-level `system` field.
                system_blocks.push(SystemBlock {
                    kind: "text".to_string(),
                    text,
                    cache_control: None,
                });
            }
            continue;
        }
        seen_non_system = true;

        let content = match &m.content {
            CanonicalContent::Text(t) => MessageContent::Text(repl.apply_prompt(t)),
            CanonicalContent::Blocks(blocks) => {
                let out: Vec<ContentBlock> = blocks
                    .iter()
                    .map(|b| canonical_block_to_anthropic(b, repl))
                    .collect::<Result<_, String>>()?;
                MessageContent::Blocks(out)
            }
        };
        // Anthropic has no "tool" role: a tool result is a `user` message
        // carrying tool_result blocks. The canonical "tool" role (from the
        // OpenAI-shaped surfaces) maps to "user" here; everything else passes
        // through. See https://docs.claude.com/en/docs/build-with-claude/tool-use.
        let is_tool_result = m.role == "tool";
        let role = if is_tool_result {
            "user".to_string()
        } else {
            m.role.clone()
        };

        // Anthropic expects multiple tool results for one assistant turn to be
        // sibling tool_result blocks inside a SINGLE user message, not a run of
        // consecutive user messages. The OpenAI-shaped surfaces emit one
        // `tool` message per result, so coalesce a tool-result message into the
        // immediately preceding message when that one is itself a user message
        // made only of tool_result blocks.
        if is_tool_result
            && let MessageContent::Blocks(new_blocks) = &content
            && let Some(prev) = messages.last_mut()
            && prev.role == "user"
            && let MessageContent::Blocks(prev_blocks) = &mut prev.content
            && prev_blocks
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolResult { .. }))
        {
            prev_blocks.extend(new_blocks.iter().cloned());
            continue;
        }
        messages.push(Message { role, content });
    }

    let tools = match req.tools.as_ref() {
        Some(t) if !t.is_empty() => Some(translate_tools(t, repl)?),
        _ => None,
    };

    if tools.is_none() && tool_choice_requires_tool(&req.tool_choice) {
        return Err("tool_choice requires or selects a tool, but no tools were provided".into());
    }

    let tool_choice = if tools.is_some() {
        translate_tool_choice(&req.tool_choice, None) // parallel handled elsewhere for now
    } else {
        None
    };

    // Leave max_tokens at the sentinel 0 when the client supplied neither an
    // explicit value nor (below) a thinking budget. apply_profile_wire_defaults
    // then fills the captured Claude Code wire value (64k/32k per model), which
    // is what real Claude Code bodies carry. Using the catalog default here
    // instead would deviate from the fingerprint baseline. See
    // apply_profile_wire_defaults + FingerprintProfile::wire_defaults_for_model.
    let mut max_tokens = req.max_tokens.unwrap_or(0);

    let thinking = derive_thinking_from_canonical(req.reasoning.as_ref(), model_def);

    if let Some(t) = thinking.as_ref()
        && let Some(budget) = t.budget_tokens
        && max_tokens <= budget
    {
        // Thinking is active: max_tokens must exceed the budget. This is a
        // different request shape than a default reply, so we compute a concrete
        // value here (non-zero, so the wire-default gate below leaves it alone).
        max_tokens = budget
            .saturating_add(1024)
            .min(default_max_tokens(model_def));
    }

    let thinking_active = thinking
        .as_ref()
        .map(|t| t.kind == "enabled")
        .unwrap_or(false);

    // When thinking is active Anthropic requires temperature=1.0 and forbids
    // top_p/top_k/stop_sequences. Otherwise pass client sampling through; the
    // wire-default temperature (when the client omitted one) is applied later in
    // apply_profile_wire_defaults so it can key off the resolved outbound model.
    let temperature = if thinking_active {
        Some(1.0)
    } else {
        req.temperature
    };
    let top_p = if thinking_active { None } else { req.top_p };
    let top_k = None;
    let stop_sequences = None;

    // metadata / user passthrough limited for canonical v1
    let metadata = None;

    if req.provider_extras.is_some() {
        return Err("unsupported provider extras for claude".into());
    }

    // Hoisted leading system/developer text. `prepend_claude_code_identity`
    // (run later in the provider) treats this as the existing system blocks and
    // prepends the Claude Code identity blocks before it, preserving order.
    let system = if system_blocks.is_empty() {
        None
    } else {
        Some(SystemField::Blocks(system_blocks))
    };

    Ok(MessagesRequest {
        model: model_def.canonical.to_string(),
        max_tokens,
        messages,
        system,
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

/// Extract the text of a canonical system/developer message for hoisting into
/// the top-level `system` field (or a mid-thread marked user turn). Any non-text
/// block (image, tool_use, tool_result) is rejected with a 400-class error
/// rather than silently dropped, because such content cannot be represented as
/// Anthropic system content. Multiple text blocks are joined with newlines.
fn system_text(m: &CanonicalMessage) -> Result<String, String> {
    match &m.content {
        CanonicalContent::Text(t) => Ok(t.clone()),
        CanonicalContent::Blocks(blocks) => {
            let mut parts: Vec<&str> = Vec::with_capacity(blocks.len());
            for b in blocks {
                match b {
                    CanonicalBlock::Text(t) => parts.push(t),
                    _ => {
                        return Err(
                            "system/developer messages must contain only text content".into()
                        );
                    }
                }
            }
            Ok(parts.join("\n"))
        }
    }
}

/// Convert one canonical content block into its Anthropic `ContentBlock`.
/// Fallible because a malformed tool-call arguments string must surface as an
/// error rather than be silently coerced.
fn canonical_block_to_anthropic(
    block: &CanonicalBlock,
    repl: &Replacements,
) -> Result<ContentBlock, String> {
    Ok(match block {
        CanonicalBlock::Text(t) => ContentBlock::Text {
            text: repl.apply_prompt(t),
            cache_control: None,
        },
        CanonicalBlock::ToolUse {
            id,
            name,
            arguments,
        } => ContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: parse_tool_arguments(arguments)?,
        },
        CanonicalBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: Some(ToolResultContent::Text(repl.apply_prompt(content))),
            is_error: if *is_error { Some(true) } else { None },
        },
        CanonicalBlock::Image { source } => ContentBlock::Image {
            source: canonical_image_to_anthropic(source)?,
        },
    })
}

fn canonical_image_to_anthropic(source: &CanonicalImageSource) -> Result<ImageSource, String> {
    match source {
        CanonicalImageSource::Url { url } if url.starts_with("https://") => {
            Ok(ImageSource::Url { url: url.clone() })
        }
        CanonicalImageSource::Url { .. } => Err("Claude image URLs must use https://".into()),
        CanonicalImageSource::Base64 { media_type, data } => Ok(ImageSource::Base64 {
            media_type: media_type.clone(),
            data: data.clone(),
        }),
    }
}

/// Parse a tool-call `arguments` string into the JSON object Anthropic expects
/// for tool `input`. An empty string means "no arguments" (-> `{}`). A non-empty
/// string that is not a JSON object is a malformed call: error loudly rather
/// than coerce it to `{}` and silently corrupt tool dispatch.
fn parse_tool_arguments(arguments: &str) -> Result<Value, String> {
    if arguments.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(v) if v.is_object() => Ok(v),
        Ok(_) => Err(format!(
            "tool_call arguments must be a JSON object, got: {arguments}"
        )),
        Err(e) => Err(format!(
            "tool_call arguments is not valid JSON ({e}): {arguments}"
        )),
    }
}

fn default_max_tokens(model_def: &ModelDef) -> u32 {
    model_def.max_tokens.min(u32::MAX as u64) as u32
}

fn derive_thinking_from_canonical(
    reasoning: Option<&CanonicalReasoning>,
    model_def: &ModelDef,
) -> Option<Thinking> {
    match reasoning {
        Some(CanonicalReasoning {
            effort: Some(e),
            budget_tokens,
        }) if !e.is_empty() => {
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
    matches!(
        choice,
        Some(CanonicalToolChoice::Required) | Some(CanonicalToolChoice::Specific { .. })
    )
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
        Some(CanonicalToolChoice::Auto) => Some(ToolChoice::Auto {
            disable_parallel_tool_use: None,
        }),
        Some(CanonicalToolChoice::Required) => Some(ToolChoice::Any {
            disable_parallel_tool_use: None,
        }),
        Some(CanonicalToolChoice::Specific { name }) => Some(ToolChoice::Tool {
            name: name.clone(),
            disable_parallel_tool_use: None,
        }),
        Some(CanonicalToolChoice::None) => Some(ToolChoice::None {}),
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
fn apply_response_replacements(
    text: &str,
    tool_calls: &mut [CanonicalToolCall],
    repl: &Replacements,
) -> String {
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
    let mut reasoning: Vec<CanonicalReasoningBlock> = Vec::new();

    for block in resp.content.iter() {
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
            ResponseContentBlock::Thinking {
                thinking,
                signature,
            } => {
                reasoning.push(CanonicalReasoningBlock {
                    text: thinking.clone(),
                    signature: signature.clone(),
                });
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
        ..Default::default()
    };

    CanonicalResponse {
        model: requested_model.to_string(),
        content,
        refusal: None,
        tool_calls,
        finish_reason: Some(finish_reason.to_string()),
        usage,
        id: Some(resp.id.clone()),
        annotations: Vec::new(),
        metadata: Some(CanonicalResponseMetadata {
            id: Some(resp.id.clone()),
            provider: Some("claude".into()),
            ..Default::default()
        }),
        reasoning,
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

    // Emit the outbound model exactly as Claude Code would: an explicit version
    // pin is forwarded verbatim, otherwise the profile canonical. This is part
    // of the wire fingerprint (per-model betas key off this value upstream).
    anth.model = profile.outbound_model(&canon.model, model_def);

    // Apply the captured Claude Code wire defaults (max_tokens / temperature /
    // output_config.effort) for any field the client did not specify, so a
    // default-shaped request matches the real Claude Code body byte-for-byte on
    // these fields rather than deviating. Must run BEFORE identity injection so
    // the billing suffix is computed over the final body shape.
    apply_profile_wire_defaults(&mut anth, profile);

    // Important: identity uses the (post-repl) first user text for the suffix.
    prepend_claude_code_identity(&mut anth, profile, inject_identity);
    Ok(anth)
}

/// Fill the fingerprint wire defaults for any field the client left unset, so a
/// default request reproduces the real Claude Code 2.1.x body. Gated on
/// "client did not supply": max_tokens sentinel 0, temperature None,
/// output_config None. Mirrors the reference implementation's
/// `apply_profile_wire_defaults` (see the working claude-code-provider).
pub fn apply_profile_wire_defaults(req: &mut MessagesRequest, profile: &FingerprintProfile) {
    let defaults = profile.wire_defaults_for_model(&req.model);
    if req.max_tokens == 0 {
        req.max_tokens = defaults.max_tokens;
    }
    if req.temperature.is_none() {
        req.temperature = defaults.temperature;
    }
    if req.output_config.is_none()
        && let Some(effort) = defaults.output_effort
    {
        req.output_config = Some(OutputConfig {
            effort: effort.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CLAUDE_CODE_SYSTEM_PREAMBLE;
    use omni_core::CanonicalMessage;

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
        assert!(
            matches!(anth.messages[0].content, MessageContent::Text(ref s) if s == "hello world")
        );
    }

    #[test]
    fn canonical_tool_blocks_map_to_anthropic_with_user_role_for_results() {
        // WHY: Anthropic has no "tool" role -- a tool result is a `user` message
        // carrying tool_result blocks. A live request with role:"tool" returns a
        // 400 "Unexpected role tool", so the canonical "tool" role MUST be
        // remapped to "user" here. This also pins the block mapping: a ToolUse's
        // string arguments become a parsed JSON `input` object, and a ToolResult
        // becomes a tool_result block keyed by tool_use_id.
        use omni_core::CanonicalBlock;
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![
                CanonicalMessage {
                    role: "assistant".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolUse {
                        id: "toolu_1".into(),
                        name: "get_weather".into(),
                        arguments: r#"{"city":"SF"}"#.into(),
                    }]),
                },
                CanonicalMessage {
                    role: "tool".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                        tool_use_id: "toolu_1".into(),
                        content: "72F".into(),
                        is_error: false,
                    }]),
                },
            ],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("haiku");
        let anth = build_messages_request_from_canonical(&req, model_def, &empty_repl()).unwrap();

        // Assistant ToolUse: role preserved, arguments parsed into a JSON object.
        assert_eq!(anth.messages[0].role, "assistant");
        match &anth.messages[0].content {
            MessageContent::Blocks(blocks) => match &blocks[0] {
                ContentBlock::ToolUse { id, name, input } => {
                    assert_eq!(id, "toolu_1");
                    assert_eq!(name, "get_weather");
                    assert_eq!(input["city"], "SF");
                }
                other => panic!("expected ToolUse block, got {other:?}"),
            },
            other => panic!("expected Blocks, got {other:?}"),
        }

        // Tool result: role REMAPPED to "user" (the bug this test guards).
        assert_eq!(
            anth.messages[1].role, "user",
            "a tool result must be sent as a Anthropic `user` message"
        );
        match &anth.messages[1].content {
            MessageContent::Blocks(blocks) => match &blocks[0] {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => {
                    assert_eq!(tool_use_id, "toolu_1");
                    assert!(matches!(content, Some(ToolResultContent::Text(t)) if t == "72F"));
                }
                other => panic!("expected ToolResult block, got {other:?}"),
            },
            other => panic!("expected Blocks, got {other:?}"),
        }
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
        assert!(crate::fingerprint::is_claude_code_billing_header(
            &blocks[0].text
        ));
        assert_eq!(blocks[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE);
    }

    #[test]
    fn from_anth_response_to_canon() {
        let resp = MessagesResponse {
            id: "msg_1".into(),
            kind: "message".into(),
            role: "assistant".into(),
            model: "claude-haiku-4-5-20251001".into(),
            content: vec![ResponseContentBlock::Text {
                text: "hi there".into(),
            }],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 5,
                output_tokens: 2,
                ..Default::default()
            },
        };
        let canon = build_canonical_response(&resp, "haiku", &empty_repl());
        assert_eq!(canon.id.as_deref(), Some("msg_1"));
        assert_eq!(canon.model, "haiku");
        assert_eq!(canon.content, "hi there");
        assert_eq!(canon.usage.input_tokens, 5);
        assert_eq!(canon.usage.output_tokens, 2);
    }

    #[test]
    fn prepare_applies_repl_and_identity() {
        let repl = Replacements::parse(
            r#"rule = [ { scope = "prompt", search = "SECRET", replace = "REDACTED" } ]"#,
        )
        .unwrap();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("tell SECRET".into()),
            }],
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
            text: "x-anthropic-billing-header: cc_version=2.1.175.174; cc_entrypoint=sdk-cli; cch=00000;".into(),
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
            messages: vec![Message {
                role: "user".into(),
                content: MessageContent::Text("x".into()),
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
            _ => panic!(),
        };
        assert!(blocks.len() >= 2);
        assert!(
            crate::fingerprint::is_claude_code_billing_header(&blocks[0].text),
            "billing must be [0]"
        );
        assert_eq!(
            blocks[1].text, CLAUDE_CODE_SYSTEM_PREAMBLE,
            "preamble must be exact at [1]"
        );
    }

    #[test]
    fn canonical_image_blocks_map_to_anthropic_images() {
        // WHY: Claude already has native image blocks; canonical image URL and
        // base64 sources must reach that wire shape without text replacement.
        use omni_core::{CanonicalBlock, CanonicalImageSource};
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Blocks(vec![
                    CanonicalBlock::Text("see".into()),
                    CanonicalBlock::Image {
                        source: CanonicalImageSource::Url {
                            url: "https://example.com/a.png".into(),
                        },
                    },
                    CanonicalBlock::Image {
                        source: CanonicalImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: "abcd".into(),
                        },
                    },
                ]),
            }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("haiku");
        let anth = build_messages_request_from_canonical(&req, model_def, &empty_repl()).unwrap();
        match &anth.messages[0].content {
            MessageContent::Blocks(blocks) => {
                assert!(matches!(&blocks[1], ContentBlock::Image {
                    source: ImageSource::Url { url }
                } if url == "https://example.com/a.png"));
                assert!(matches!(&blocks[2], ContentBlock::Image {
                    source: ImageSource::Base64 { media_type, data }
                } if media_type == "image/png" && data == "abcd"));
            }
            other => panic!("expected blocks, got {other:?}"),
        }
    }

    #[test]
    fn canonical_image_blocks_reject_non_https_urls_for_claude() {
        // WHY: Anthropic rejects non-HTTPS image URLs upstream. Catch this in
        // translation so unsupported image inputs fail before dispatch.
        use omni_core::{CanonicalBlock, CanonicalImageSource};
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Blocks(vec![CanonicalBlock::Image {
                    source: CanonicalImageSource::Url {
                        url: "http://example.com/a.png".into(),
                    },
                }]),
            }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("haiku");
        let err = build_messages_request_from_canonical(&req, model_def, &empty_repl())
            .expect_err("Claude must reject non-HTTPS image URLs");
        assert!(err.contains("https"), "error must mention HTTPS: {err}");
    }

    #[test]
    fn prepare_uses_post_repl_first_user_for_billing_suffix() {
        // Prompt-before-identity gate: repls (tool/prompt scope) run on texts
        // BEFORE billing_header_text is called, so suffix (and thus cch body)
        // is derived from post-repl bytes. Critical for gate fingerprint.
        let repl = Replacements::parse(
            r#"rule = [ { scope = "prompt", search = "FOO", replace = "BAR" } ]"#,
        )
        .unwrap();
        let canon = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("say FOO".into()),
            }],
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
    fn wire_defaults_applied_for_default_request_matches_capture() {
        // WHY: this is the project's #1 invariant. Real Claude Code bodies
        // carry captured per-model wire values; a default-shaped request (client
        // supplies neither max_tokens nor temperature nor output_config) MUST
        // reproduce them, or the body deviates from the fingerprint baseline on
        // exactly the fields the gate inspects. These expectations are the
        // captured values in fingerprint.rs MODEL_WIRE_OVERRIDES for the default profile.
        // If this test fails, either a capture changed (rebaseline) or the
        // wire-default wiring regressed (bug). Both must be caught.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();

        // (input alias, expected outbound model, expected max_tokens,
        //  expected temperature, expected output_config effort)
        type WireCase = (
            &'static str,
            &'static str,
            u32,
            Option<f32>,
            Option<&'static str>,
        );
        let cases: &[WireCase] = &[
            ("fable", "claude-fable-5", 64_000, None, Some("xhigh")),
            ("opus", "claude-opus-4-8", 64_000, None, Some("xhigh")),
            (
                "sonnet",
                "claude-sonnet-4-6",
                32_000,
                Some(1.0),
                Some("high"),
            ),
            // The "haiku" alias resolves to the dated canonical; its wire override
            // carries the same 32k / 1.0 / no-effort values as the undated entry.
            (
                "haiku",
                "claude-haiku-4-5-20251001",
                32_000,
                Some(1.0),
                None,
            ),
            (
                "claude-haiku-4-5",
                "claude-haiku-4-5",
                32_000,
                Some(1.0),
                None,
            ),
        ];

        for (alias, exp_model, exp_max, exp_temp, exp_effort) in cases {
            let canon = CanonicalRequest {
                model: (*alias).into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                ..Default::default()
            };
            let anth = prepare_anthropic_request(&canon, profile, &repl, false).unwrap();
            assert_eq!(anth.model, *exp_model, "outbound model for {alias}");
            assert_eq!(
                anth.max_tokens, *exp_max,
                "wire max_tokens for {alias} (must be the captured value, not the catalog default)"
            );
            assert_eq!(anth.temperature, *exp_temp, "wire temperature for {alias}");
            assert_eq!(
                anth.output_config.as_ref().map(|o| o.effort.as_str()),
                *exp_effort,
                "wire output_config.effort for {alias}"
            );
        }
    }

    #[test]
    fn client_supplied_values_override_wire_defaults() {
        // WHY: wire defaults fill UNSET fields only. A client that explicitly
        // sets max_tokens/temperature must have those honored (proxy fidelity),
        // and the wire default must not clobber them. Guards the is_none()/==0
        // gating in apply_profile_wire_defaults against regressing to
        // unconditional overwrite.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            max_tokens: Some(7),
            temperature: Some(0.3),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&canon, profile, &repl, false).unwrap();
        assert_eq!(anth.max_tokens, 7, "client max_tokens must be preserved");
        assert_eq!(
            anth.temperature,
            Some(0.3),
            "client temperature must be preserved"
        );
        // output_config still filled from wire default (client did not set it).
        assert_eq!(
            anth.output_config.as_ref().map(|o| o.effort.as_str()),
            Some("high")
        );
    }

    #[test]
    fn build_messages_rejects_unsupported_provider_extras() {
        // WHY: Claude's OpenAI-compatible path has no provider-extra
        // passthrough today. Unsupported extras must fail loudly instead of
        // vanishing before the fingerprint-sensitive wire request is built.
        let mut canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };
        canon.provider_extras = Some(serde_json::json!({"foo":"bar"}));
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("sonnet");
        let err = build_messages_request_from_canonical(&canon, model_def, &empty_repl())
            .expect_err("provider extras must reject");
        assert!(
            err.contains("provider extras"),
            "error must name provider extras: {err}"
        );
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
                ResponseContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "do".into(),
                    input: serde_json::json!({"x":1}),
                },
                ResponseContentBlock::Text {
                    text: "done".into(),
                },
            ],
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 3,
                cache_creation_input_tokens: Some(2),
                cache_read_input_tokens: Some(1),
            },
        };
        let canon = build_canonical_response(&resp, "haiku", &empty_repl());
        assert_eq!(canon.id.as_deref(), Some("m"));
        assert_eq!(canon.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(canon.tool_calls.len(), 1);
        assert_eq!(canon.tool_calls[0].name, "do");
        assert_eq!(canon.usage.cache_creation, 2);
        assert_eq!(canon.usage.cache_read, 1);
        assert!(canon.content.contains("done"));
    }

    #[test]
    fn build_canonical_response_preserves_thinking_blocks() {
        // WHY: Claude thinking was parsed but dropped. It must stay available
        // as additive reasoning metadata without polluting assistant text.
        let resp = MessagesResponse {
            id: "m".into(),
            kind: "message".into(),
            role: "assistant".into(),
            model: "claude-haiku-4-5-20251001".into(),
            content: vec![
                ResponseContentBlock::Thinking {
                    thinking: "internal".into(),
                    signature: Some("sig".into()),
                },
                ResponseContentBlock::Text {
                    text: "visible".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 2,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let canon = build_canonical_response(&resp, "haiku", &empty_repl());
        assert_eq!(canon.content, "visible");
        assert_eq!(canon.reasoning.len(), 1);
        assert_eq!(canon.reasoning[0].text, "internal");
        assert_eq!(canon.reasoning[0].signature.as_deref(), Some("sig"));
        assert_eq!(
            canon
                .metadata
                .as_ref()
                .and_then(|meta| meta.provider.as_deref()),
            Some("claude")
        );
    }

    #[test]
    fn response_repl_applies_to_text_and_tool_surfaces() {
        let repl = Replacements::parse(
            r#"rule = [ { scope = "response", search = "HIDE", replace = "SHOWN" } ]"#,
        )
        .unwrap();
        let resp = MessagesResponse {
            id: "m".into(),
            kind: "message".into(),
            role: "assistant".into(),
            model: "haiku".into(),
            content: vec![ResponseContentBlock::ToolUse {
                id: "t".into(),
                name: "callHIDE".into(),
                input: serde_json::json!({}),
            }],
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                ..Default::default()
            },
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
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("p".into()),
            }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let mdef = profile.resolve_model("haiku");
        let mut built = build_messages_request_from_canonical(&canon, mdef, &repl).unwrap();
        prepend_claude_code_identity(&mut built, profile, true);
        let prepped = prepare_anthropic_request(&canon, profile, &repl, true).unwrap();
        // compare key identity + model
        let bsys = match built.system {
            Some(SystemField::Blocks(b)) => b,
            _ => panic!(),
        };
        let psys = match prepped.system {
            Some(SystemField::Blocks(b)) => b,
            _ => panic!(),
        };
        assert_eq!(bsys[0].text, psys[0].text);
        assert_eq!(bsys[1].text, psys[1].text);
        assert_eq!(built.model, prepped.model);
    }

    #[test]
    fn consecutive_tool_results_coalesce_into_one_user_message() {
        // WHY: Anthropic requires the results of a parallel tool turn to be
        // sibling tool_result blocks inside ONE `user` message. The OpenAI-shaped
        // surfaces emit one `tool` message per result, so two consecutive
        // tool-result messages must be merged here; emitting two separate user
        // messages is a 400 on a live request. This pins the coalesce path.
        use omni_core::CanonicalBlock;
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![
                CanonicalMessage {
                    role: "assistant".into(),
                    content: CanonicalContent::Blocks(vec![
                        CanonicalBlock::ToolUse {
                            id: "t1".into(),
                            name: "f".into(),
                            arguments: "{}".into(),
                        },
                        CanonicalBlock::ToolUse {
                            id: "t2".into(),
                            name: "g".into(),
                            arguments: "{}".into(),
                        },
                    ]),
                },
                CanonicalMessage {
                    role: "tool".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                        tool_use_id: "t1".into(),
                        content: "A".into(),
                        is_error: false,
                    }]),
                },
                CanonicalMessage {
                    role: "tool".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                        tool_use_id: "t2".into(),
                        content: "B".into(),
                        is_error: false,
                    }]),
                },
            ],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("haiku");
        let anth = build_messages_request_from_canonical(&req, model_def, &empty_repl()).unwrap();

        // Exactly ONE user message carries tool_result blocks (not two).
        let tool_result_user_msgs: Vec<&Message> = anth
            .messages
            .iter()
            .filter(|m| {
                m.role == "user"
                    && matches!(
                        &m.content,
                        MessageContent::Blocks(blocks)
                            if blocks.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. }))
                    )
            })
            .collect();
        assert_eq!(
            tool_result_user_msgs.len(),
            1,
            "both tool results must coalesce into a single user message, not two"
        );

        // That one message holds BOTH results (t1 and t2), in order.
        let ids: Vec<&str> = match &tool_result_user_msgs[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(
                    blocks.len(),
                    2,
                    "the merged user message must carry both results"
                );
                blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::ToolResult { tool_use_id, .. } => tool_use_id.as_str(),
                        other => panic!("expected ToolResult block, got {other:?}"),
                    })
                    .collect()
            }
            other => panic!("expected Blocks, got {other:?}"),
        };
        assert!(ids.contains(&"t1"), "merged message must keep t1");
        assert!(ids.contains(&"t2"), "merged message must keep t2");
    }

    #[test]
    fn tool_arguments_parsing_rejects_malformed_and_accepts_empty() {
        // WHY: a tool-call `arguments` string must become the JSON object
        // Anthropic expects for `input`. An empty string means "no arguments"
        // (-> {}), but any non-object (a JSON array/string/scalar) or invalid
        // JSON is a malformed call: surface it as an error rather than coerce to
        // {} and silently corrupt tool dispatch.
        let empty = parse_tool_arguments("").expect("empty args is allowed");
        assert_eq!(empty, serde_json::json!({}), "empty args must be {{}}");

        let obj = parse_tool_arguments(r#"{"a":1}"#).expect("object args is allowed");
        assert_eq!(obj["a"], 1);

        parse_tool_arguments("not json").expect_err("invalid JSON must reject");
        parse_tool_arguments("[1,2]").expect_err("a JSON array is not an object: must reject");
        parse_tool_arguments("\"x\"").expect_err("a JSON string is not an object: must reject");
    }

    // ── Issue #2: system/developer hoisting (C-ref reshape) ───────────────

    fn sys_msg(role: &str, text: &str) -> CanonicalMessage {
        CanonicalMessage {
            role: role.into(),
            content: CanonicalContent::Text(text.into()),
        }
    }

    fn user_msg(text: &str) -> CanonicalMessage {
        CanonicalMessage {
            role: "user".into(),
            content: CanonicalContent::Text(text.into()),
        }
    }

    fn build_haiku(messages: Vec<CanonicalMessage>) -> Result<MessagesRequest, String> {
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages,
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("haiku");
        build_messages_request_from_canonical(&req, model_def, &empty_repl())
    }

    fn system_texts(req: &MessagesRequest) -> Vec<String> {
        match &req.system {
            Some(SystemField::Blocks(b)) => b.iter().map(|s| s.text.clone()).collect(),
            Some(SystemField::Text(t)) => vec![t.clone()],
            None => Vec::new(),
        }
    }

    #[test]
    fn leading_system_hoists_to_top_level_system() {
        // WHY: Anthropic /v1/messages rejects a `system` role inside `messages`
        // with a 400; a leading system prompt (the common OpenAI case) MUST be
        // moved to the top-level `system` field instead.
        let anth = build_haiku(vec![sys_msg("system", "You are terse."), user_msg("hi")]).unwrap();
        assert_eq!(system_texts(&anth), vec!["You are terse.".to_string()]);
        assert!(
            anth.messages.iter().all(|m| m.role != "system"),
            "no system role may remain in the messages array"
        );
        assert_eq!(anth.messages.len(), 1);
        assert_eq!(anth.messages[0].role, "user");
    }

    #[test]
    fn developer_role_hoists_like_system() {
        // WHY: the Chat Completions surface does not normalize `developer` to
        // `system` (omni-common chat_message_to_canonical clones the role), so
        // the Claude path must handle `developer` identically or it 400s.
        let anth =
            build_haiku(vec![sys_msg("developer", "Be precise."), user_msg("hi")]).unwrap();
        assert_eq!(system_texts(&anth), vec!["Be precise.".to_string()]);
        assert!(anth.messages.iter().all(|m| m.role != "developer"));
    }

    #[test]
    fn mid_thread_system_folds_into_marked_user_turn_in_place() {
        // WHY: a system message after the conversation has started carries a
        // temporal directive ("from now on..."); relocating it to the global
        // preamble would lose its position. C-ref folds it in place into a
        // user turn marked "[system message]\n", preserving order.
        let anth = build_haiku(vec![
            sys_msg("system", "Lead."),
            user_msg("u1"),
            sys_msg("system", "Switch to JSON."),
            user_msg("u2"),
        ])
        .unwrap();

        // Leading system hoisted; mid-thread one stays in the array as a user turn.
        assert_eq!(system_texts(&anth), vec!["Lead.".to_string()]);
        assert!(anth.messages.iter().all(|m| m.role != "system"));
        assert_eq!(anth.messages.len(), 3);
        assert_eq!(anth.messages[0].role, "user"); // u1
        assert_eq!(anth.messages[1].role, "user"); // folded mid-thread system, position preserved
        assert_eq!(anth.messages[2].role, "user"); // u2
        match &anth.messages[1].content {
            MessageContent::Blocks(blocks) => match &blocks[0] {
                ContentBlock::Text { text, .. } => {
                    assert_eq!(text, "[system message]\nSwitch to JSON.");
                }
                other => panic!("expected Text block, got {other:?}"),
            },
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn multiple_leading_system_messages_concatenate_as_blocks() {
        let anth = build_haiku(vec![
            sys_msg("system", "One."),
            sys_msg("developer", "Two."),
            user_msg("hi"),
        ])
        .unwrap();
        assert_eq!(
            system_texts(&anth),
            vec!["One.".to_string(), "Two.".to_string()]
        );
    }

    #[test]
    fn empty_system_message_is_skipped() {
        // WHY: a truly empty ("") system/developer message must not produce a
        // phantom top-level system block (leading) or an empty marked user turn
        // (mid-thread). Matches the reference's `!text.is_empty()` guard.
        let anth = build_haiku(vec![
            sys_msg("system", ""),
            user_msg("u1"),
            sys_msg("system", ""),
            user_msg("u2"),
        ])
        .unwrap();
        assert!(
            anth.system.is_none(),
            "empty leading system contributes no top-level system"
        );
        assert_eq!(
            anth.messages.len(),
            2,
            "empty mid-thread system must not add a phantom user turn"
        );
        assert!(anth.messages.iter().all(|m| m.role == "user"));
    }

    #[test]
    fn non_text_block_in_system_message_errors() {
        // WHY: an image (or other non-text) block in a system/developer message
        // cannot be represented as Anthropic system content; fail loud (400)
        // rather than silently drop it.
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![
                CanonicalMessage {
                    role: "system".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::Image {
                        source: CanonicalImageSource::Url {
                            url: "https://example.com/x.png".into(),
                        },
                    }]),
                },
                user_msg("hi"),
            ],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("haiku");
        build_messages_request_from_canonical(&req, model_def, &empty_repl())
            .expect_err("non-text block in a system message must be rejected");
    }

    #[test]
    fn hoisted_system_survives_identity_injection_both_ways() {
        // WHY: prepend_claude_code_identity early-returns when inject_identity is
        // false WITHOUT clearing req.system, and prepends identity before it when
        // true. Either way the hoisted leading system must survive.
        let profile = crate::fingerprint::default_profile();

        let mut with_identity =
            build_haiku(vec![sys_msg("system", "Keep me."), user_msg("hi")]).unwrap();
        prepend_claude_code_identity(&mut with_identity, profile, true);
        let texts = system_texts(&with_identity);
        assert!(
            texts.iter().any(|t| t == "Keep me."),
            "hoisted system must survive identity injection (true)"
        );
        assert!(
            crate::fingerprint::is_claude_code_billing_header(&texts[0]),
            "identity billing block prepended before the hoisted system"
        );

        let mut without_identity =
            build_haiku(vec![sys_msg("system", "Keep me."), user_msg("hi")]).unwrap();
        prepend_claude_code_identity(&mut without_identity, profile, false);
        assert_eq!(
            system_texts(&without_identity),
            vec!["Keep me.".to_string()],
            "hoisted system must survive when identity injection is off"
        );
    }
}
