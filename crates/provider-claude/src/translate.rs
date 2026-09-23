//! Translation between omni-core Canonical* types and Anthropic Messages wire format.
//! Plus identity (preamble + billing) injection and outbound/inbound replacements hook.
//!
//! **Isolation note:** The wire structs here (MessagesRequest etc) control
//! exact JSON field presence and shape for the OAuth gate. They are
//! deliberately private to provider-claude.
//!
//! Adapted from reference-src-claude/translate/{anthropic.rs, build.rs,
//! from_anthropic.rs, tool_translate.rs, ...} and routes/completions_v2.rs
//! (prepend_claude_code_identity).
//!
//! The canonical types are intentionally lossy (flat Text only today); the
//! adapter here maps the supported subset while still routing the request
//! through the full fingerprint + identity path.

use std::collections::BTreeMap;

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use omni_common::Replacements;
use omni_core::{
    CanonicalBlock, CanonicalCacheMark, CanonicalCacheMode, CanonicalCacheTtl, CanonicalContent,
    CanonicalFileSource, CanonicalImageSource, CanonicalMessage, CanonicalReasoning,
    CanonicalReasoningBlock, CanonicalRequest, CanonicalResponse, CanonicalResponseMetadata,
    CanonicalTool, CanonicalToolCall, CanonicalToolChoice, CanonicalUsage, ProviderError,
    clamp_duration,
};

use crate::anthropic_passthrough::apply_prompt_replacements;
use crate::fingerprint::FingerprintProfile;
use crate::models::ModelDef;
// UpstreamError kept commented for future use in count_tokens etc; no current non-test refs.

// ── Native Anthropic Messages API wire types (exact shapes for the gate) ──

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

    /// Top-level Anthropic *automatic* caching marker. When set, the server
    /// places one cache breakpoint on the last cacheable block and moves it
    /// forward as the conversation grows (the documented analog of OpenAI's
    /// server-side caching). Serialized LAST so it never lands inside the
    /// cached prefix. Set when the builder copies client `cache.automatic`,
    /// when Door 2 forwards the client's official top-level `cache_control`,
    /// or when first-party OpenAI inbound injects gateway auto-cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
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
    pub strict: Option<bool>,
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

/// Anthropic `output_config` object.
///
/// `effort` is the Claude Code pin / Door-1 surface. Other members (e.g.
/// `format` for structured outputs) are preserved via flatten so Door-2 native
/// pass-through does not silently strip client intent (issue #22).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Unknown / future Anthropic keys under `output_config` (e.g. `format`).
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String, // "ephemeral"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>, // "5m" or "1h"
}

impl<'de> Deserialize<'de> for CacheControl {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(rename = "type")]
            kind: String,
            #[serde(default)]
            ttl: Option<String>,
        }
        let raw = Raw::deserialize(deserializer)?;
        if raw.kind != "ephemeral" {
            return Err(de::Error::custom(format!(
                "cache_control.type must be \"ephemeral\", got {:?}",
                raw.kind
            )));
        }
        match raw.ttl.as_deref() {
            None => {}
            Some("5m") | Some("1h") => {}
            Some(other) => {
                return Err(de::Error::custom(format!(
                    "cache_control.ttl must be \"5m\" or \"1h\", got {other:?}"
                )));
            }
        }
        Ok(Self {
            kind: raw.kind,
            ttl: raw.ttl,
        })
    }
}

// ── Response (non-streaming) ──────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ResponseContentBlock>,
    pub stop_reason: Option<String>,
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
    model_def: Option<&ModelDef>,
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
    let request_ttl = claude_request_ttl(req);
    let mut system_blocks: Vec<SystemBlock> = Vec::new();
    let mut seen_non_system = false;
    let mut messages: Vec<Message> = Vec::new();
    for m in &req.messages {
        if m.role == "system" || m.role == "developer" {
            // Non-text blocks (e.g. an image) in a system/developer message are
            // unrepresentable as system content; fail loud rather than silently
            // drop them (diverges from the reference's silent text-only extract).
            let hoisted = hoist_system_blocks(m, repl, request_ttl)?;
            if hoisted.is_empty() {
                continue;
            }
            if seen_non_system {
                // Mid-thread: one user message holding every hoisted block, each
                // keeping its cache_control. Splitting per block would add extra
                // user turns and break adjacent-role / cache-slot layout.
                messages.push(Message {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(
                        hoisted
                            .into_iter()
                            .map(|block| ContentBlock::Text {
                                text: format!("[system message]\n{}", block.text),
                                cache_control: block.cache_control,
                            })
                            .collect(),
                    ),
                });
            } else {
                system_blocks.extend(hoisted);
            }
            continue;
        }
        seen_non_system = true;

        let content = match &m.content {
            CanonicalContent::Text(t) => MessageContent::Text(repl.apply_prompt(t)),
            CanonicalContent::Blocks(blocks) => {
                let out: Vec<ContentBlock> = blocks
                    .iter()
                    .map(|b| canonical_block_to_anthropic(b, repl, request_ttl))
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
        Some(t) if !t.is_empty() => Some(translate_tools(t, repl, request_ttl)?),
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

    // Leave max_tokens at the sentinel 0 when the client omitted it. The
    // provider-level forge tail (finalize_claude_wire_request) runs
    // apply_profile_wire_defaults, so the sentinel must survive this build
    // untouched. Filling a concrete value here would pre-empt the tail and
    // deviate from the fingerprint baseline. Thinking budget never auto-bumps
    // max_tokens (issue #19).
    let max_tokens = req.max_tokens.unwrap_or(0);

    let thinking = derive_thinking_from_canonical(req.reasoning.as_ref());

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

    // output_config.effort from client is applied in prepare_anthropic_request
    // (needs profile wire/beta support). Leave None here so pin defaults still
    // fill only when client effort is absent.
    let output_config = None;

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
        // Known model -> its canonical id; unknown -> the raw requested model.
        // In the provider path this is immediately overwritten by
        // outbound_model (which re-decides verbatim-vs-canonical), so this only
        // matters to the direct test callers of this builder.
        model: model_def
            .map(|d| d.canonical.to_string())
            .unwrap_or_else(|| req.model.clone()),
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
        output_config,
        // Copy client automatic cache_control when present. Gateway auto-cache
        // injection still happens later in prepare_anthropic_request /
        // finalize_claude_wire_request when should_inject_gateway_auto_cache
        // and supports_auto_cache both allow it.
        cache_control: req
            .cache
            .as_ref()
            .and_then(|c| c.automatic.as_ref())
            .and_then(|mark| claude_cache_control(Some(mark), request_ttl)),
    })
}

/// Extract the text of a canonical system/developer message for hoisting into
/// the top-level `system` field (or a mid-thread marked user turn). Any non-text
/// block (image, tool_use, tool_result) is rejected with a 400-class error
/// rather than silently dropped, because such content cannot be represented as
/// Anthropic system content. Multiple text blocks are joined with newlines.
fn hoist_system_blocks(
    m: &CanonicalMessage,
    repl: &Replacements,
    request_ttl: Option<CanonicalCacheTtl>,
) -> Result<Vec<SystemBlock>, String> {
    match &m.content {
        CanonicalContent::Text(t) => {
            if t.is_empty() {
                Ok(Vec::new())
            } else {
                Ok(vec![SystemBlock {
                    kind: "text".into(),
                    text: repl.apply_prompt(t),
                    cache_control: None,
                }])
            }
        }
        CanonicalContent::Blocks(blocks) => {
            let has_mark = blocks.iter().any(|b| b.cache_mark().is_some());
            if !has_mark {
                let text = system_text(m)?;
                if text.is_empty() {
                    return Ok(Vec::new());
                }
                return Ok(vec![SystemBlock {
                    kind: "text".into(),
                    text: repl.apply_prompt(&text),
                    cache_control: None,
                }]);
            }
            let mut out = Vec::new();
            for b in blocks {
                match b {
                    CanonicalBlock::Text { text, cache } => {
                        out.push(SystemBlock {
                            kind: "text".into(),
                            text: repl.apply_prompt(text),
                            cache_control: claude_cache_control(cache.as_ref(), request_ttl),
                        });
                    }
                    _ => {
                        return Err(
                            "system/developer messages must contain only text content".into()
                        );
                    }
                }
            }
            Ok(out)
        }
    }
}

fn claude_request_ttl(req: &CanonicalRequest) -> Option<CanonicalCacheTtl> {
    let cache = req.cache.as_ref()?;
    if let Some(ttl) = cache.ttl {
        return if CanonicalCacheTtl::CLAUDE.contains(&ttl) {
            Some(ttl)
        } else {
            ttl.clamp_down(&CanonicalCacheTtl::CLAUDE)
        };
    }
    cache
        .legacy_retention
        .and_then(|ret| clamp_duration(ret.duration_secs(), CanonicalCacheTtl::CLAUDE))
}

fn claude_cache_control(
    mark: Option<&CanonicalCacheMark>,
    request_ttl: Option<CanonicalCacheTtl>,
) -> Option<CacheControl> {
    let mark = mark?;
    let ttl = mark
        .ttl
        .map(|ttl| {
            if CanonicalCacheTtl::CLAUDE.contains(&ttl) {
                Some(ttl)
            } else {
                ttl.clamp_down(&CanonicalCacheTtl::CLAUDE)
            }
        })
        .unwrap_or(request_ttl);
    Some(CacheControl {
        kind: "ephemeral".into(),
        ttl: ttl.map(|t| t.as_str().to_string()),
    })
}

fn should_inject_gateway_auto_cache(canon: &CanonicalRequest) -> bool {
    if canon.has_cache_marks() {
        return false;
    }
    match &canon.cache {
        Some(cache) if cache.automatic.is_some() => false,
        Some(cache) if cache.mode == Some(CanonicalCacheMode::Explicit) => false,
        _ => true,
    }
}

fn cap_claude_cache_slots(req: &mut MessagesRequest) {
    const MAX_SLOTS: usize = 4;
    let mut remaining_drop = claude_cache_slot_count(req).saturating_sub(MAX_SLOTS);
    if remaining_drop == 0 {
        return;
    }
    if let Some(tools) = req.tools.as_mut() {
        for tool in tools {
            if remaining_drop == 0 {
                return;
            }
            if tool.cache_control.is_some() {
                tool.cache_control = None;
                remaining_drop -= 1;
            }
        }
    }
    if let Some(SystemField::Blocks(blocks)) = req.system.as_mut() {
        for block in blocks {
            if remaining_drop == 0 {
                return;
            }
            if block.cache_control.is_some() {
                block.cache_control = None;
                remaining_drop -= 1;
            }
        }
    }
    for message in &mut req.messages {
        let MessageContent::Blocks(blocks) = &mut message.content else {
            continue;
        };
        for block in blocks {
            if remaining_drop == 0 {
                return;
            }
            let cache_control = match block {
                ContentBlock::Text { cache_control, .. }
                | ContentBlock::Image { cache_control, .. }
                | ContentBlock::Document { cache_control, .. }
                | ContentBlock::ToolUse { cache_control, .. }
                | ContentBlock::ToolResult { cache_control, .. } => cache_control,
                ContentBlock::Thinking { .. } => continue,
            };
            if cache_control.is_some() {
                *cache_control = None;
                remaining_drop -= 1;
            }
        }
    }
    if remaining_drop > 0 && req.cache_control.is_some() {
        req.cache_control = None;
    }
}

fn claude_cache_slot_count(req: &MessagesRequest) -> usize {
    let mut n = 0;
    if let Some(tools) = req.tools.as_ref() {
        n += tools.iter().filter(|t| t.cache_control.is_some()).count();
    }
    if let Some(SystemField::Blocks(blocks)) = req.system.as_ref() {
        n += blocks.iter().filter(|b| b.cache_control.is_some()).count();
    }
    for message in &req.messages {
        let MessageContent::Blocks(blocks) = &message.content else {
            continue;
        };
        for block in blocks {
            let marked = match block {
                ContentBlock::Text { cache_control, .. }
                | ContentBlock::Image { cache_control, .. }
                | ContentBlock::Document { cache_control, .. }
                | ContentBlock::ToolUse { cache_control, .. }
                | ContentBlock::ToolResult { cache_control, .. } => cache_control.is_some(),
                ContentBlock::Thinking { .. } => false,
            };
            if marked {
                n += 1;
            }
        }
    }
    if req.cache_control.is_some() {
        n += 1;
    }
    n
}

fn system_text(m: &CanonicalMessage) -> Result<String, String> {
    match &m.content {
        CanonicalContent::Text(t) => Ok(t.clone()),
        CanonicalContent::Blocks(blocks) => {
            let mut parts: Vec<&str> = Vec::with_capacity(blocks.len());
            for b in blocks {
                match b {
                    CanonicalBlock::Text { text: t, .. } => parts.push(t),
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
    request_ttl: Option<CanonicalCacheTtl>,
) -> Result<ContentBlock, String> {
    Ok(match block {
        CanonicalBlock::Text { text: t, cache } => ContentBlock::Text {
            text: repl.apply_prompt(t),
            cache_control: claude_cache_control(cache.as_ref(), request_ttl),
        },
        CanonicalBlock::ToolUse {
            id,
            name,
            arguments,
            cache,
        } => ContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: parse_tool_arguments(arguments)?,
            cache_control: claude_cache_control(cache.as_ref(), request_ttl),
        },
        CanonicalBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
            cache,
        } => ContentBlock::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: Some(ToolResultContent::Text(repl.apply_prompt(content))),
            is_error: if *is_error { Some(true) } else { None },
            cache_control: claude_cache_control(cache.as_ref(), request_ttl),
        },
        CanonicalBlock::Image { source, cache } => ContentBlock::Image {
            source: canonical_image_to_anthropic(source)?,
            cache_control: claude_cache_control(cache.as_ref(), request_ttl),
        },
        CanonicalBlock::File {
            source,
            filename,
            detail,
            cache,
        } => {
            if detail.is_some() {
                return Err("Claude documents cannot honor input_file detail".into());
            }
            let source = match source {
                CanonicalFileSource::Url { file_url } => DocumentSource::Url {
                    url: file_url.clone(),
                },
                CanonicalFileSource::Data { file_data } => {
                    let data = if let Some(encoded) = file_data.strip_prefix("data:") {
                        let (media_type, data) = encoded
                            .split_once(";base64,")
                            .ok_or("Claude file_data requires base64 data")?;
                        if media_type != "application/pdf" || data.is_empty() {
                            return Err(
                                "Claude file_data requires application/pdf base64 data".into()
                            );
                        }
                        data
                    } else {
                        file_data.as_str()
                    };
                    if !file_data.starts_with("data:")
                        && filename
                            .as_deref()
                            .is_none_or(|name| !name.to_ascii_lowercase().ends_with(".pdf"))
                    {
                        return Err(
                            "Claude file_data requires a PDF filename or application/pdf data URL"
                                .into(),
                        );
                    }
                    DocumentSource::Base64 {
                        media_type: "application/pdf".into(),
                        data: data.into(),
                    }
                }
                CanonicalFileSource::Id { .. } => {
                    return Err(
                        "Claude cannot use an OpenAI file_id; use file_url or PDF file_data".into(),
                    );
                }
            };
            ContentBlock::Document {
                source,
                title: filename.clone(),
                context: None,
                citations: None,
                cache_control: claude_cache_control(cache.as_ref(), request_ttl),
            }
        }
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

fn derive_thinking_from_canonical(reasoning: Option<&CanonicalReasoning>) -> Option<Thinking> {
    match reasoning {
        Some(CanonicalReasoning {
            effort: Some(e),
            budget_tokens,
        }) if !e.is_empty() && e != "none" => {
            let budget = budget_tokens.unwrap_or_else(|| budget_for_effort(e));
            // Anthropic rejects budget 0. Unknown efforts that map to 0 disable
            // thinking rather than building an invalid body.
            if budget == 0 {
                return None;
            }
            Some(Thinking {
                kind: "enabled".into(),
                budget_tokens: Some(budget),
            })
        }
        _ => None,
    }
}

fn budget_for_effort(effort: &str) -> u32 {
    match effort {
        // OpenAI "minimal" is below "low"; Claude has no lower rung, so map to low.
        "minimal" | "low" => 1024,
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

fn translate_tools(
    tools: &[CanonicalTool],
    repl: &Replacements,
    request_ttl: Option<CanonicalCacheTtl>,
) -> Result<Vec<Tool>, String> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let name = repl.apply_prompt(&t.name);
        let description = t.description.as_ref().map(|d| repl.apply_prompt(d));
        let input_schema = omni_core::provider_tool_schema(&t.parameters, false)?;
        let strict = omni_core::claude_strict(&input_schema, t.strict).then_some(true);
        out.push(Tool {
            name,
            description,
            input_schema,
            strict,
            cache_control: claude_cache_control(t.cache.as_ref(), request_ttl),
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

/// Prepend the Claude Code billing marker (dynamic `cc_version` suffix) + system
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
    text == crate::fingerprint::CLAUDE_CODE_SYSTEM_PREAMBLE
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
/// post-identity, ready for finalize_body_json).
/// Returns the MessagesRequest and the first-user text (for context if needed).
pub fn prepare_anthropic_request(
    canon: &CanonicalRequest,
    profile: &FingerprintProfile,
    repl: &Replacements,
    inject_identity: bool,
    supports_auto_cache: bool,
) -> Result<MessagesRequest, ProviderError> {
    let resolved = profile.resolve_model(&canon.model);
    // Door 1 applies prompt replacements during the canonical build (repl is
    // threaded into build_messages_request_from_canonical), so the shared tail
    // must NOT re-apply them or the billing suffix would be computed over
    // double-replaced text. Pass an empty Replacements to the tail.
    // Shaping failures (unrepresentable content, bad tools/images) are client
    // faults -> BadRequest. Effort boundary already returns ProviderError
    // (issue #30: no String unwrap/re-wrap).
    let mut anth = build_messages_request_from_canonical(canon, resolved, repl)
        .map_err(ProviderError::BadRequest)?;

    // Issue #20: client effort -> output_config.effort before wire defaults so
    // pin cannot overwrite. Only on models that support the effort surface
    // (pin output_effort or effort beta). Else keep thinking-budget path; if
    // that also cannot express the value, fail loud (no silent omit).
    apply_client_effort_to_output_config(&mut anth, canon, profile)?;

    // Prefer output_config.effort when set from client. Legacy thinking budgets
    // only when the client supplied an explicit budget_tokens (or the model has
    // no effort surface, so build left thinking as the expression path).
    // Effort-derived thinking + output_config together can 400 on modern
    // Anthropic models that want adaptive effort, not manual enabled thinking.
    if anth.output_config.is_some()
        && canon
            .reasoning
            .as_ref()
            .and_then(|r| r.budget_tokens)
            .is_none()
        && anth.thinking.is_some()
    {
        anth.thinking = None;
        // build forced temperature=1.0 / top_p=None while thinking was active;
        // restore client sampling so pin defaults can fill unset fields.
        anth.temperature = canon.temperature;
        anth.top_p = canon.top_p;
    }

    let inject_auto = supports_auto_cache && should_inject_gateway_auto_cache(canon);
    finalize_claude_wire_request(
        &mut anth,
        &canon.model,
        resolved,
        profile,
        &Replacements::empty(),
        inject_identity,
        inject_auto,
    );
    if inject_auto && anth.cache_control.is_some() {
        if let Some(ttl) = claude_request_ttl(canon) {
            if let Some(cc) = anth.cache_control.as_mut() {
                cc.ttl = Some(ttl.as_str().to_string());
            }
        }
    }
    cap_claude_cache_slots(&mut anth);
    // Explicit client "none"/empty means omit effort, not "use pin default".
    if let Some(effort) = canon.reasoning.as_ref().and_then(|r| r.effort.as_deref())
        && (effort.is_empty() || effort == "none")
    {
        anth.output_config = None;
    }
    Ok(anth)
}

/// Whether this model’s fingerprint pin exposes `output_config.effort`.
fn model_supports_output_effort(profile: &FingerprintProfile, model: &str) -> bool {
    let defaults = profile.wire_defaults_for_model(model);
    defaults.output_effort.is_some()
        || profile
            .beta_reply_for_model(model)
            .split(',')
            .any(|b| b == "effort-2025-11-24")
}

/// Map OpenAI-centric effort aliases onto Anthropic `output_config.effort`.
/// Free strings (incl. `xhigh`) pass through. Documented aliases: `minimal`→
/// `low`. `max` is kept as Anthropic vocabulary (not clamped to high).
fn claude_output_effort_value(effort: &str) -> &str {
    match effort {
        "minimal" => "low",
        other => other,
    }
}

/// Apply explicit client reasoning effort onto `output_config` when the model
/// can express it. Runs before pin wire defaults (issue #20 precedence).
fn apply_client_effort_to_output_config(
    anth: &mut MessagesRequest,
    canon: &CanonicalRequest,
    profile: &FingerprintProfile,
) -> Result<(), ProviderError> {
    let Some(effort) = canon.reasoning.as_ref().and_then(|r| r.effort.as_deref()) else {
        return Ok(());
    };
    if effort.is_empty() || effort == "none" {
        return Ok(());
    }
    // Use the builder’s resolved model id (canonical when known).
    let model = anth.model.as_str();
    if model_supports_output_effort(profile, model) {
        anth.output_config = Some(OutputConfig {
            effort: Some(claude_output_effort_value(effort).to_string()),
            extra: BTreeMap::new(),
        });
        return Ok(());
    }
    // No output_config surface (e.g. Haiku pin). Thinking budget may still
    // express known ladder values; unmappable efforts must not silently vanish.
    // Shared structured shape (issue #25); thinking-budget-only models still
    // fail loud for values with no budget mapping.
    if budget_for_effort(effort) == 0 {
        return Err(ProviderError::unsupported_reasoning_effort(
            "claude",
            Some(model),
            "messages",
            effort,
            &["low", "medium", "high", "max"],
        ));
    }
    Ok(())
}

/// The single provider-level "forge tail" shared by both Claude-bound doors
/// (OpenAI->Anthropic translate and native /v1/messages). It finalizes an
/// already-built MessagesRequest into the exact outbound wire shape.
///
/// Ordering is load-bearing and MUST NOT be reordered:
/// 1. outbound model id (verbatim pin vs profile canonical)
/// 2. prompt replacements (before identity, so the billing suffix sees final text)
/// 3. wire defaults (fills only still-unset fields)
/// 4. identity injection (the billing suffix is derived from the
///    post-replacement first user text)
/// 5. auto-cache marker (LAST: a pure top-level appendage; it does not touch the
///    system/message prefix identity computed by step 4)
///
/// Intentionally does **not** raise `max_tokens` when a thinking budget would
/// prefer more room (issue #19). Client `max_tokens` is sent as given; if the
/// client omitted it, wire defaults fill the fingerprint capture value only.
/// Upstream may reject `max_tokens <= thinking.budget_tokens`; that is a
/// client/request concern, not a gateway auto-bump.
///
/// `supports_auto_cache` gates step 5: it is only true on first-party Anthropic
/// routes with auto-caching enabled (Bedrock/Vertex and custom gateways pass
/// false). See the caller in lib.rs for the gate.
///
/// `stream` is intentionally NOT handled here; each door owns it (a native
/// request forces `stream: false` when omitted, which the tail must not clobber).
pub fn finalize_claude_wire_request(
    req: &mut MessagesRequest,
    input: &str,
    resolved: Option<&ModelDef>,
    profile: &FingerprintProfile,
    replacements: &Replacements,
    inject_identity: bool,
    supports_auto_cache: bool,
) {
    // 1. Emit the outbound model exactly as Claude Code would: an explicit
    // version pin is forwarded verbatim, an alias/canonical maps to the profile
    // canonical, and a truly unknown id passes through raw. Part of the wire
    // fingerprint (per-model betas key off this value upstream).
    req.model = match resolved {
        Some(def) => profile.outbound_model(input, def),
        None => input.to_string(),
    };

    // 2. Prompt-scope replacements. Door 1 passes an empty set (it already
    // applied them in the build); Door 2 passes its real replacements here.
    apply_prompt_replacements(req, replacements);

    // 3. Fill the captured Claude Code wire defaults for any field the client
    // left unset (max_tokens sentinel 0, temperature None, output_config None).
    // NOTE: this is NOT a no-op wrapper - it also re-defaults an explicit
    // max_tokens:0, intentionally. Client non-none effort is already on
    // output_config from Door-1 build and is not overwritten here (issue #20).
    // Explicit "none" is cleared after this tail in prepare_anthropic_request.
    // Thinking budget does not influence max_tokens here (issue #19).
    apply_profile_wire_defaults(req, profile);

    // 4. Identity: runs after wire defaults; the billing suffix is derived from
    // the (post-replacement) first user text.
    prepend_claude_code_identity(req, profile, inject_identity);

    // 5. Auto-cache marker LAST. A single top-level `cache_control` puts Anthropic
    // in automatic-caching mode: the server anchors one breakpoint on the last
    // cacheable block and advances it as the conversation grows. This is a pure
    // sibling field (serialized after everything else), so it does NOT alter the
    // tools->system->messages prefix that identity just finalized; the sha-stable
    // billing header at system[0] means the cached prefix is byte-stable across
    // turns as long as the client keeps the first user message identical.
    // Gated to first-party routes only (Bedrock/Vertex do not support automatic
    // caching); the per-model minimum-token floor makes a sub-floor prefix a
    // silent zero-cost no-op, so no size gate is needed here.
    if supports_auto_cache {
        req.cache_control = Some(CacheControl {
            kind: "ephemeral".into(),
            ttl: None,
        });
    }
}

/// Fill the fingerprint wire defaults for any field the client left unset, so a
/// default request reproduces the real Claude Code 2.1.x body. Gated on
/// "client did not supply": max_tokens sentinel 0, temperature None,
/// output_config None. Mirrors the reference implementation's
/// `apply_profile_wire_defaults` (see the working claude-code-provider).
///
/// Client non-none effort must already be on `output_config` before this runs
/// so pin defaults cannot overwrite it (issue #20).
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
            effort: Some(effort.to_string()),
            extra: BTreeMap::new(),
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
                        cache: None,
                    }]),
                },
                CanonicalMessage {
                    role: "tool".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                        tool_use_id: "toolu_1".into(),
                        content: "72F".into(),
                        is_error: false,
                        cache: None,
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
                ContentBlock::ToolUse {
                    id, name, input, ..
                } => {
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
            cache_control: None,
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
            model: "claude-haiku-4-5-20251001".into(),
            content: vec![ResponseContentBlock::Text {
                text: "hi there".into(),
            }],
            stop_reason: Some("end_turn".into()),

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
        let anth = prepare_anthropic_request(&canon, profile, &repl, true, false).unwrap();
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
    fn auto_cache_true_emits_top_level_ephemeral_marker() {
        // WHY: this is the PR's whole behavior — a single top-level cache_control
        // marker puts Anthropic in automatic-caching mode. It MUST serialize as a
        // sibling of model/system/messages (NOT inside a content block; a block
        // marker is a different, manual mode), with exactly {"type":"ephemeral"}
        // and no ttl (automatic caching uses the default 5m TTL).
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hello".into()),
            }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let anth = prepare_anthropic_request(&canon, profile, &empty_repl(), true, true).unwrap();
        assert_eq!(
            anth.cache_control.as_ref().map(|c| c.kind.as_str()),
            Some("ephemeral"),
            "top-level cache_control must be ephemeral when auto-cache is on"
        );
        assert!(
            anth.cache_control.as_ref().is_some_and(|c| c.ttl.is_none()),
            "automatic caching uses the default TTL (no ttl field on the wire)"
        );
        // The marker is a TOP-LEVEL sibling; the wire form is {"type":"ephemeral"}.
        let val = serde_json::to_value(&anth).unwrap();
        assert_eq!(
            val.get("cache_control")
                .and_then(|c| c.get("type"))
                .and_then(|t| t.as_str()),
            Some("ephemeral"),
            "cache_control must be a top-level sibling serialized as {{\"type\":\"ephemeral\"}}"
        );
        // And it must NOT have leaked into any content block or the system field
        // (those are the manual-mode placements this PR deliberately does not use).
        for m in &anth.messages {
            if let MessageContent::Blocks(blocks) = &m.content {
                for b in blocks {
                    if let ContentBlock::Text { cache_control, .. } = b {
                        assert!(cache_control.is_none(), "no block-level marker in PR1");
                    }
                }
            }
        }
    }

    #[test]
    fn auto_cache_false_is_byte_identical_to_baseline() {
        // WHY: the feature ships dark. With auto-cache off the outbound body must
        // be byte-for-byte identical to today's captured Claude Code baseline, so
        // the default path keeps fingerprint parity. Proven by serializing the
        // SAME request with the flag off vs on and asserting the off form has no
        // cache_control key at all while the on form differs by exactly that key.
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hello".into()),
            }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let off = prepare_anthropic_request(&canon, profile, &empty_repl(), true, false).unwrap();
        let on = prepare_anthropic_request(&canon, profile, &empty_repl(), true, true).unwrap();
        assert!(off.cache_control.is_none(), "off => no marker");

        let off_bytes = serde_json::to_vec(&off).unwrap();
        assert!(
            !String::from_utf8_lossy(&off_bytes).contains("cache_control"),
            "off body must not contain the cache_control key (skip_serializing_if, not null)"
        );
        // The ONLY difference between off and on is the added top-level key: clear
        // it on the `on` form and the two serialize identically.
        let mut on_cleared = on;
        on_cleared.cache_control = None;
        assert_eq!(
            serde_json::to_vec(&on_cleared).unwrap(),
            off_bytes,
            "on and off must differ by exactly the cache_control key, nothing else"
        );
    }

    #[test]
    fn auto_cache_prefix_is_byte_stable_across_turns() {
        // WHY: automatic caching only pays off if the cached prefix is byte-stable
        // turn over turn. The billing header at system[0] is derived from the FIRST
        // user message; this test locks in that a later conversation turn (same
        // first message, more messages appended) reproduces byte-identical
        // system[0] (billing) and system[1] (preamble). If the sha-billing suffix
        // ever became per-request (a timestamp/nonce leak), this fails and the
        // cache would silently never hit.
        let profile = crate::fingerprint::default_profile();
        let mk = |msgs: Vec<CanonicalMessage>| {
            let canon = CanonicalRequest {
                model: "sonnet".into(),
                messages: msgs,
                ..Default::default()
            };
            let anth =
                prepare_anthropic_request(&canon, profile, &empty_repl(), true, true).unwrap();
            match anth.system.unwrap() {
                SystemField::Blocks(b) => b,
                _ => panic!("expected system blocks"),
            }
        };
        let user = |t: &str| CanonicalMessage {
            role: "user".into(),
            content: CanonicalContent::Text(t.into()),
        };
        let assistant = |t: &str| CanonicalMessage {
            role: "assistant".into(),
            content: CanonicalContent::Text(t.into()),
        };
        // Turn 1: single user message.
        let turn1 = mk(vec![user("what is the capital of France?")]);
        // Turn 3: SAME first user message, plus an assistant reply and a follow-up.
        let turn3 = mk(vec![
            user("what is the capital of France?"),
            assistant("Paris."),
            user("and of Germany?"),
        ]);
        assert_eq!(
            turn1[0].text, turn3[0].text,
            "billing header (system[0]) must be byte-identical across turns of the same conversation"
        );
        assert_eq!(
            turn1[1].text, turn3[1].text,
            "system preamble (system[1]) must be byte-identical across turns"
        );
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
        // Invariant: the billing marker is ALWAYS first in system blocks and the
        // preamble comes right after it. That is the captured Claude Code order
        // the OAuth gate expects (see CLAUDE_CODE_SYSTEM_PREAMBLE).
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
            cache_control: None,
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
                    CanonicalBlock::Text {
                        text: "see".into(),
                        cache: None,
                    },
                    CanonicalBlock::Image {
                        source: CanonicalImageSource::Url {
                            url: "https://example.com/a.png".into(),
                        },
                        cache: None,
                    },
                    CanonicalBlock::Image {
                        source: CanonicalImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: "abcd".into(),
                        },
                        cache: None,
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
                    source: ImageSource::Url { url },
                    ..
                } if url == "https://example.com/a.png"));
                assert!(matches!(&blocks[2], ContentBlock::Image {
                    source: ImageSource::Base64 { media_type, data },
                    ..
                } if media_type == "image/png" && data == "abcd"));
            }
            other => panic!("expected blocks, got {other:?}"),
        }
    }

    #[test]
    fn canonical_files_map_to_claude_documents_with_cache_control() {
        use omni_core::{CanonicalBlock, CanonicalFileSource};
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Blocks(vec![
                    CanonicalBlock::File {
                        source: CanonicalFileSource::Url {
                            file_url: "https://example.com/a.pdf".into(),
                        },
                        filename: None,
                        detail: None,
                        cache: Some(omni_core::CanonicalCacheMark::breakpoint()),
                    },
                    CanonicalBlock::File {
                        source: CanonicalFileSource::Data {
                            file_data: "data:application/pdf;base64,abcd".into(),
                        },
                        filename: Some("a.pdf".into()),
                        detail: None,
                        cache: None,
                    },
                    CanonicalBlock::File {
                        source: CanonicalFileSource::Data {
                            file_data: concat!("data:application/", "pdf;base64,YWJjZA==").into(),
                        },
                        filename: None,
                        detail: None,
                        cache: None,
                    },
                ]),
            }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let anth = build_messages_request_from_canonical(
            &req,
            profile.resolve_model("haiku"),
            &empty_repl(),
        )
        .unwrap();
        let wire = serde_json::to_value(&anth.messages[0]).unwrap();
        assert_eq!(wire["content"][0]["type"], "document");
        assert_eq!(
            wire["content"][0]["source"],
            serde_json::json!({"type":"url","url":"https://example.com/a.pdf"})
        );
        assert_eq!(wire["content"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(wire["content"][1]["title"], "a.pdf");
        assert_eq!(wire["content"][2]["source"]["type"], "base64");
        assert_eq!(wire["content"][2]["source"]["data"], "YWJjZA==");
        assert_eq!(
            wire["content"][1]["source"],
            serde_json::json!({"type":"base64","media_type":"application/pdf","data":"abcd"})
        );
    }

    #[test]
    fn pdf_data_url_accepts_filename_without_pdf_extension() {
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Blocks(vec![CanonicalBlock::File {
                    source: CanonicalFileSource::Data {
                        file_data: concat!("data:application/", "pdf;base64,YWJjZA==").into(),
                    },
                    filename: Some("invoice".into()),
                    detail: None,
                    cache: None,
                }]),
            }],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let anth = build_messages_request_from_canonical(
            &req,
            profile.resolve_model("haiku"),
            &empty_repl(),
        )
        .unwrap();
        let wire = serde_json::to_value(&anth.messages[0]).unwrap();
        assert_eq!(wire["content"][0]["title"], "invoice");
        assert_eq!(wire["content"][0]["source"]["data"], "YWJjZA==");
    }

    #[test]
    fn canonical_files_reject_claude_unmappable_inputs() {
        for (source, filename, detail, field) in [
            (
                CanonicalFileSource::Id {
                    file_id: "file-1".into(),
                },
                None,
                None,
                "file_id",
            ),
            (
                CanonicalFileSource::Data {
                    file_data: "abcd".into(),
                },
                None,
                None,
                "PDF filename",
            ),
            (
                CanonicalFileSource::Url {
                    file_url: "https://example.com/a.pdf".into(),
                },
                None,
                Some("high".into()),
                "detail",
            ),
        ] {
            let req = CanonicalRequest {
                model: "haiku".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::File {
                        source,
                        filename,
                        detail,
                        cache: None,
                    }]),
                }],
                ..Default::default()
            };
            let profile = crate::fingerprint::default_profile();
            let err = build_messages_request_from_canonical(
                &req,
                profile.resolve_model("haiku"),
                &empty_repl(),
            )
            .unwrap_err();
            assert!(err.contains(field), "{err}");
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
                    cache: None,
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
        // BEFORE billing_header_text is called, so the cc_version suffix is
        // derived from post-repl bytes. Critical for gate fingerprint.
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
        let anth = prepare_anthropic_request(&canon, profile, &repl, true, false).unwrap();
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
            ("fable", "claude-fable-5-1", 64_000, None, Some("high")),
            (
                "claude-fable-5-1",
                "claude-fable-5-1",
                64_000,
                None,
                Some("high"),
            ),
            (
                "claude-fable-5",
                "claude-fable-5",
                64_000,
                None,
                Some("high"),
            ),
            ("opus", "claude-opus-5-5", 128_000, None, Some("medium")),
            ("sonnet", "claude-sonnet-5", 64_000, None, Some("high")),
            // The "haiku" alias resolves to the dated canonical; 2.1.220 omits
            // temperature on the wire (no temp=1) and has no output_config.
            ("haiku", "claude-haiku-4-5-20251001", 32_000, None, None),
            ("claude-haiku-4-5", "claude-haiku-4-5", 32_000, None, None),
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
            let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
            assert_eq!(anth.model, *exp_model, "outbound model for {alias}");
            assert_eq!(
                anth.max_tokens, *exp_max,
                "wire max_tokens for {alias} (must be the captured value, not the catalog default)"
            );
            assert_eq!(anth.temperature, *exp_temp, "wire temperature for {alias}");
            assert_eq!(
                anth.output_config
                    .as_ref()
                    .and_then(|o| o.effort.as_deref()),
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
        let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
        assert_eq!(anth.max_tokens, 7, "client max_tokens must be preserved");
        assert_eq!(
            anth.temperature,
            Some(0.3),
            "client temperature must be preserved"
        );
        // output_config still filled from wire default (client did not set it).
        assert_eq!(
            anth.output_config
                .as_ref()
                .and_then(|o| o.effort.as_deref()),
            Some("high")
        );
    }

    #[test]
    fn client_effort_sets_output_config_and_overrides_pin_default() {
        // WHY (issue #20): explicit client effort drives output_config.effort and
        // must not be overwritten by fingerprint pin defaults.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            reasoning: Some(CanonicalReasoning {
                effort: Some("low".into()),
                budget_tokens: None,
            }),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
        assert_eq!(
            anth.output_config
                .as_ref()
                .and_then(|o| o.effort.as_deref()),
            Some("low"),
            "client effort must win over the Sonnet pin default"
        );
    }

    #[test]
    fn client_xhigh_effort_reaches_output_config() {
        // WHY (issue #20): chat-accepted xhigh must land on Anthropic
        // output_config.effort, not be blocked or replaced by pin defaults.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            reasoning: Some(CanonicalReasoning {
                effort: Some("xhigh".into()),
                budget_tokens: None,
            }),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
        assert_eq!(
            anth.output_config
                .as_ref()
                .and_then(|o| o.effort.as_deref()),
            Some("xhigh")
        );
    }

    #[test]
    fn absent_effort_keeps_sonnet_pin_default_high() {
        // WHY (issue #20): pin default applies only when client effort is absent.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
        assert_eq!(
            anth.output_config
                .as_ref()
                .and_then(|o| o.effort.as_deref()),
            Some("high"),
            "Sonnet capture-backed pin default when effort absent"
        );
    }

    #[test]
    fn explicit_none_effort_suppresses_pin_output_config() {
        // WHY (issue #20): client "none" is explicit omit, not absence. Pin must
        // not re-inject output_config.effort.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            reasoning: Some(CanonicalReasoning {
                effort: Some("none".into()),
                budget_tokens: None,
            }),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
        assert!(
            anth.output_config.is_none(),
            "explicit none must not get pin effort: {:?}",
            anth.output_config
        );
    }

    #[test]
    fn haiku_client_effort_does_not_inject_output_config() {
        // WHY (issue #20 review): Haiku pin has no output_effort and no effort
        // beta. Client high/medium must not force output_config (upstream 400 /
        // fingerprint break); thinking budget still expresses known efforts.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            reasoning: Some(CanonicalReasoning {
                effort: Some("high".into()),
                budget_tokens: None,
            }),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
        assert!(
            anth.output_config.is_none(),
            "haiku must not get output_config.effort: {:?}",
            anth.output_config
        );
        assert!(
            anth.thinking.is_some(),
            "haiku high effort still uses thinking budget"
        );
    }

    #[test]
    fn haiku_unmappable_effort_fails_loud() {
        // WHY (issues #20/#25): xhigh has no thinking budget and haiku has no
        // output_config surface; must fail loud with the shared structured
        // unsupported-effort shape (same fields as Grok/Codex), not silent omit
        // or a hand-built string.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            reasoning: Some(CanonicalReasoning {
                effort: Some("xhigh".into()),
                budget_tokens: None,
            }),
            ..Default::default()
        };
        let err = prepare_anthropic_request(&canon, profile, &repl, false, false)
            .expect_err("haiku xhigh must fail loud");
        // End-to-end ProviderError::BadRequest (issue #30); no String unwrap.
        // Real fail path (prepare_anthropic_request → apply_client_effort):
        // shared helper includes model= for the resolved outbound pin id (#28).
        let msg = match &err {
            ProviderError::BadRequest(m) => m.as_str(),
            other => panic!("unsupported effort must be BadRequest, got {other:?}"),
        };
        assert!(
            msg.contains("unsupported reasoning_effort")
                && msg.contains("provider=claude")
                && msg.contains("path=messages")
                && msg.contains("requested=xhigh")
                && msg.contains("model=")
                && msg.contains("supported=["),
            "structured unsupported effort (incl. model=): {msg}"
        );
        // Field order matches ProviderError::unsupported_reasoning_effort:
        // provider, path, requested, optional model, then supported.
        let provider_pos = msg.find("provider=claude").expect("provider field");
        let path_pos = msg.find("path=messages").expect("path field");
        let requested_pos = msg.find("requested=xhigh").expect("requested field");
        let model_pos = msg.find("model=").expect("model field");
        let supported_pos = msg.find("supported=[").expect("supported field");
        assert!(
            provider_pos < path_pos
                && path_pos < requested_pos
                && requested_pos < model_pos
                && model_pos < supported_pos,
            "shared helper field order: {msg}"
        );
    }

    #[test]
    fn client_minimal_effort_maps_to_low_on_output_config() {
        // WHY: OpenAI minimal is below Anthropic's lowest rung; map to low on
        // output_config (adapter-local alias), matching thinking budget mapping.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            reasoning: Some(CanonicalReasoning {
                effort: Some("minimal".into()),
                budget_tokens: None,
            }),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
        assert_eq!(
            anth.output_config
                .as_ref()
                .and_then(|o| o.effort.as_deref()),
            Some("low")
        );
    }

    #[test]
    fn door1_effort_on_output_config_model_skips_thinking_budget() {
        // WHY (issue #20): sonnet supports output_config.effort; client effort
        // drives that knob and must not also emit legacy thinking budgets (can
        // 400 on modern Anthropic). max_tokens comes from the pin (64000).
        // Haiku (no effort surface) still uses the thinking path
        // — see haiku_client_effort_does_not_inject_output_config.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        for effort in ["high", "max"] {
            let canon = CanonicalRequest {
                model: "sonnet".into(),
                messages: vec![CanonicalMessage {
                    role: "user".into(),
                    content: CanonicalContent::Text("hi".into()),
                }],
                reasoning: Some(CanonicalReasoning {
                    effort: Some(effort.into()),
                    budget_tokens: None,
                }),
                ..Default::default()
            };
            let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
            assert_eq!(
                anth.output_config
                    .as_ref()
                    .and_then(|o| o.effort.as_deref()),
                Some(effort),
                "sonnet effort={effort} on output_config"
            );
            assert!(
                anth.thinking.is_none(),
                "sonnet effort={effort} must not also set thinking: {:?}",
                anth.thinking
            );
            assert_eq!(
                anth.max_tokens, 64_000,
                "sonnet effort={effort}: pin max_tokens (no thinking path)"
            );
        }
    }

    #[test]
    fn door1_thinking_budget_does_not_raise_client_max_tokens() {
        // WHY (issue #19): when the client sets max_tokens below a thinking
        // budget, omni must NOT auto-bump max_tokens. Cover both explicit
        // budget_tokens and effort-mapped budget (haiku has no output_config
        // effort surface, so effort maps to thinking).
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();

        let explicit = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            max_tokens: Some(100),
            reasoning: Some(CanonicalReasoning {
                effort: Some("high".into()),
                budget_tokens: Some(4096),
            }),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&explicit, profile, &repl, false, false).unwrap();
        assert_eq!(
            anth.thinking.as_ref().and_then(|t| t.budget_tokens),
            Some(4096),
            "explicit thinking budget must still be present"
        );
        assert_eq!(
            anth.max_tokens, 100,
            "client max_tokens must not be raised for explicit thinking budget"
        );

        // effort high -> budget_for_effort = 16384 on haiku thinking path
        let effort_mapped = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            max_tokens: Some(100),
            reasoning: Some(CanonicalReasoning {
                effort: Some("high".into()),
                budget_tokens: None,
            }),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&effort_mapped, profile, &repl, false, false).unwrap();
        assert_eq!(
            anth.thinking.as_ref().and_then(|t| t.budget_tokens),
            Some(16384),
            "effort-mapped thinking budget must still be present"
        );
        assert_eq!(
            anth.max_tokens, 100,
            "client max_tokens must not be raised for effort-mapped thinking budget"
        );
    }

    #[test]
    fn door1_explicit_budget_tokens_keeps_thinking_with_output_config() {
        // WHY: client-supplied budget_tokens is explicit legacy thinking; keep
        // it even when output_config.effort is also set from effort string.
        let profile = crate::fingerprint::default_profile();
        let repl = empty_repl();
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            reasoning: Some(CanonicalReasoning {
                effort: Some("high".into()),
                budget_tokens: Some(4096),
            }),
            ..Default::default()
        };
        let anth = prepare_anthropic_request(&canon, profile, &repl, false, false).unwrap();
        assert_eq!(
            anth.output_config
                .as_ref()
                .and_then(|o| o.effort.as_deref()),
            Some("high")
        );
        assert_eq!(
            anth.thinking.as_ref().and_then(|t| t.budget_tokens),
            Some(4096)
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
    fn build_messages_ignores_prompt_cache_key_and_does_not_emit_it() {
        // WHY: Claude has no cache-routing key. A Chat/Responses client that
        // sends prompt_cache_key must not 400 (that used to happen when the
        // field sat in extras). The Anthropic wire must not grow an unknown
        // top-level key either.
        let canon = CanonicalRequest {
            model: "sonnet".into(),
            messages: vec![CanonicalMessage {
                role: "user".into(),
                content: CanonicalContent::Text("hi".into()),
            }],
            cache: Some(omni_core::CanonicalCacheIntent {
                routing_identity: Some("sess-1".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("sonnet");
        let anth = build_messages_request_from_canonical(&canon, model_def, &empty_repl())
            .expect("prompt_cache_key is not a provider extra");
        let val = serde_json::to_value(&anth).expect("serialize");
        assert!(
            val.get("prompt_cache_key").is_none(),
            "Claude Anthropic body must not carry prompt_cache_key: {val}"
        );
    }

    #[test]
    fn openai_30m_ttl_clamps_to_claude_5m_never_1h() {
        let req: omni_common::ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"sonnet","messages":[{"role":"user","content":"hi"}],
                "prompt_cache_options":{"ttl":"30m"}}"#,
        )
        .unwrap();
        let canon = omni_common::to_canonical(&req).unwrap();
        let profile = crate::fingerprint::default_profile();
        let anth = prepare_anthropic_request(&canon, profile, &empty_repl(), true, true).unwrap();
        let ttl = anth.cache_control.as_ref().and_then(|c| c.ttl.as_deref());
        assert_eq!(ttl, Some("5m"));
        assert_ne!(ttl, Some("1h"));
    }

    #[test]
    fn explicit_mode_without_breakpoints_does_not_inject_claude_auto_cache() {
        let req: omni_common::ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"sonnet","messages":[{"role":"user","content":"hi"}],
                "prompt_cache_options":{"mode":"explicit"}}"#,
        )
        .unwrap();
        let canon = omni_common::to_canonical(&req).unwrap();
        let profile = crate::fingerprint::default_profile();
        let anth = prepare_anthropic_request(&canon, profile, &empty_repl(), true, true).unwrap();
        assert!(
            anth.cache_control.is_none(),
            "explicit with no breakpoints must not inject auto-cache: {:?}",
            anth.cache_control
        );
    }

    #[test]
    fn claude_keeps_last_four_cache_slots() {
        let req: omni_common::ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"sonnet","messages":[{"role":"user","content":[
                {"type":"text","text":"a","prompt_cache_breakpoint":{"mode":"explicit"}},
                {"type":"text","text":"b","prompt_cache_breakpoint":{"mode":"explicit"}},
                {"type":"text","text":"c","prompt_cache_breakpoint":{"mode":"explicit"}},
                {"type":"text","text":"d","prompt_cache_breakpoint":{"mode":"explicit"}},
                {"type":"text","text":"e","prompt_cache_breakpoint":{"mode":"explicit"}}
            ]}]}"#,
        )
        .unwrap();
        let canon = omni_common::to_canonical(&req).unwrap();
        let profile = crate::fingerprint::default_profile();
        let anth = prepare_anthropic_request(&canon, profile, &empty_repl(), true, true).unwrap();
        let MessageContent::Blocks(blocks) = &anth.messages[0].content else {
            panic!("expected blocks");
        };
        let marked: Vec<_> = blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text {
                    text,
                    cache_control: Some(_),
                } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(marked, vec!["b", "c", "d", "e"]);
        assert!(anth.cache_control.is_none());
    }

    #[test]
    fn build_canonical_response_maps_raw_usage_with_cache_and_tool_calls() {
        // Usage raw (incl cache_*) + tool_calls from anth must round to canon;
        // finish "tool_use" -> "tool_calls".
        let resp = MessagesResponse {
            id: "m".into(),
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
            model: "haiku".into(),
            content: vec![ResponseContentBlock::ToolUse {
                id: "t".into(),
                name: "callHIDE".into(),
                input: serde_json::json!({}),
            }],
            stop_reason: Some("tool_use".into()),

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
        let prepped = prepare_anthropic_request(&canon, profile, &repl, true, false).unwrap();
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
                            cache: None,
                        },
                        CanonicalBlock::ToolUse {
                            id: "t2".into(),
                            name: "g".into(),
                            arguments: "{}".into(),
                            cache: None,
                        },
                    ]),
                },
                CanonicalMessage {
                    role: "tool".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                        tool_use_id: "t1".into(),
                        content: "A".into(),
                        is_error: false,
                        cache: None,
                    }]),
                },
                CanonicalMessage {
                    role: "tool".into(),
                    content: CanonicalContent::Blocks(vec![CanonicalBlock::ToolResult {
                        tool_use_id: "t2".into(),
                        content: "B".into(),
                        is_error: false,
                        cache: None,
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
        let anth = build_haiku(vec![sys_msg("developer", "Be precise."), user_msg("hi")]).unwrap();
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
    fn mid_thread_marked_system_stays_one_user_message() {
        // WHY: a mid-thread system/developer with several cache marks must fold
        // into a single user turn. One user message per block would split the
        // conversation and drop the joint cache layout.
        let req = CanonicalRequest {
            model: "haiku".into(),
            messages: vec![
                user_msg("u1"),
                CanonicalMessage {
                    role: "system".into(),
                    content: CanonicalContent::Blocks(vec![
                        CanonicalBlock::Text {
                            text: "first".into(),
                            cache: Some(omni_core::CanonicalCacheMark::breakpoint()),
                        },
                        CanonicalBlock::Text {
                            text: "second".into(),
                            cache: Some(omni_core::CanonicalCacheMark::breakpoint()),
                        },
                    ]),
                },
                user_msg("u2"),
            ],
            ..Default::default()
        };
        let profile = crate::fingerprint::default_profile();
        let model_def = profile.resolve_model("haiku");
        let anth = build_messages_request_from_canonical(&req, model_def, &empty_repl()).unwrap();
        assert_eq!(anth.messages.len(), 3);
        assert_eq!(anth.messages[1].role, "user");
        match &anth.messages[1].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    ContentBlock::Text {
                        text,
                        cache_control,
                    } => {
                        assert_eq!(text, "[system message]\nfirst");
                        assert!(cache_control.is_some());
                    }
                    other => panic!("expected Text, got {other:?}"),
                }
                match &blocks[1] {
                    ContentBlock::Text {
                        text,
                        cache_control,
                    } => {
                        assert_eq!(text, "[system message]\nsecond");
                        assert!(cache_control.is_some());
                    }
                    other => panic!("expected Text, got {other:?}"),
                }
            }
            other => panic!("expected one Blocks user turn, got {other:?}"),
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
                        cache: None,
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

#[cfg(test)]
mod issue_51_tool_tests {
    use super::*;
    use serde_json::json;

    fn chat(schema: Value, strict: bool) -> Result<Value, ProviderError> {
        let req: omni_common::ChatCompletionRequest = serde_json::from_value(json!({"model":"sonnet","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":schema,"strict":strict}}],"tool_choice":"required"})).unwrap();
        let canon = omni_common::to_canonical(&req).unwrap();
        let wire = prepare_anthropic_request(
            &canon,
            crate::fingerprint::default_profile(),
            &Replacements::empty(),
            false,
            false,
        )?;
        Ok(serde_json::to_value(wire).unwrap())
    }

    #[test]
    fn claude_rejects_ref_branch_instead_of_losing_query() {
        let schema = json!({"$defs":{"args":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},"allOf":[{"$ref":"#/$defs/args"}]});
        let req: omni_common::ChatCompletionRequest = serde_json::from_value(json!({"model":"sonnet","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":schema}}]})).unwrap();
        let canon = omni_common::to_canonical(&req).unwrap();
        assert_eq!(canon.tools.as_ref().unwrap()[0].parameters, schema);
        assert!(
            matches!(prepare_anthropic_request(&canon, crate::fingerprint::default_profile(), &Replacements::empty(), false, false), Err(ProviderError::BadRequest(msg)) if msg.contains("$ref"))
        );
        let dangling = json!({"properties":{"query":{"$ref":"#/allOf/0/$defs/q"}},"required":["query"],"allOf":[{"$defs":{"q":{"type":"string"}}}]});
        assert!(
            matches!(chat(dangling, false), Err(ProviderError::BadRequest(msg)) if msg.contains("$ref"))
        );
        let valid = json!({"$defs":{"q":{"type":"string"}},"allOf":[{"properties":{"query":{"$ref":"#/$defs/q"}},"required":["query"]}]});
        let wire = chat(valid, false).unwrap();
        assert_eq!(
            wire["tools"][0]["input_schema"]["properties"]["query"]["$ref"],
            "#/$defs/q"
        );
        assert_eq!(
            wire["tools"][0]["input_schema"]["required"],
            json!(["query"])
        );
    }

    #[test]
    fn claude_flattens_and_preserves_choice() {
        let schema = json!({"type":"object","properties":{"b":{"type":"string"}},"required":["b"],"oneOf":[{"properties":{"a":{"type":"number"}},"required":["a"]},{"properties":{"c":{"type":"boolean"}},"required":["c"]}]});
        let wire = chat(schema, false).unwrap();
        let tool = &wire["tools"][0];
        assert!(tool.get("strict").is_none());
        assert!(tool["input_schema"].get("oneOf").is_none());
        for key in ["a", "b", "c"] {
            assert!(tool["input_schema"]["properties"].get(key).is_some());
        }
        assert_eq!(tool["input_schema"]["required"], json!(["b"]));
        assert_eq!(wire["tool_choice"]["type"], "any");
        assert!(
            chat(json!({"anyOf":[{"type":"object"},7]}), false)
                .unwrap_err()
                .to_string()
                .contains("branch")
        );
    }

    #[test]
    fn claude_strict_downgrades_without_rewriting() {
        let schema = json!({"type":"object","additionalProperties":false,"properties":{"x":{"anyOf":[{"type":"string"}]}}});
        let wire = chat(schema.clone(), true).unwrap();
        assert_eq!(wire["tools"][0]["strict"], true);
        assert_eq!(wire["tools"][0]["input_schema"], schema);
        let loose = json!({"type":"object","properties":{"x":{"anyOf":[{"type":"string"}]}}});
        let wire = chat(loose.clone(), true).unwrap();
        assert!(wire["tools"][0].get("strict").is_none());
        assert_eq!(wire["tools"][0]["input_schema"], loose);
    }
}
