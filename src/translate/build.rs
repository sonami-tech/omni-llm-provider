//! Top-level OAI ChatCompletionRequest → Anthropic MessagesRequest.

use serde_json::Value;

use crate::error::AppError;
use crate::models::ModelDef;
use crate::translate::anthropic::{MessagesRequest, Metadata, Thinking};
use crate::translate::messages::reshape;
use crate::translate::request::{ChatCompletionRequest, StopSpec};
use crate::translate::tool_translate::{translate_tool_choice, translate_tools};

/// Build an Anthropic MessagesRequest from an OAI ChatCompletionRequest.
pub fn build_messages_request(
    req: &ChatCompletionRequest,
    model_def: &ModelDef,
) -> Result<MessagesRequest, AppError> {
    if req.n.unwrap_or(1) > 1 {
        return Err(AppError::BadRequest(
            "n>1 is not supported (Anthropic Messages does not support multiple completions)"
                .into(),
        ));
    }

    let reshaped = reshape(&req.messages)?;

    let tools = match req.tools.as_ref() {
        Some(t) if !t.is_empty() => Some(translate_tools(t)?),
        _ => None,
    };

    let tool_choice = if tools.is_some() {
        translate_tool_choice(&req.tool_choice)
    } else {
        None
    };

    let stop_sequences = req.stop.as_ref().map(|s| match s {
        StopSpec::One(s) => vec![s.clone()],
        StopSpec::Many(v) => v.clone(),
    });

    let mut max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or_else(|| default_max_tokens(model_def));

    let thinking = derive_thinking(req, model_def);

    // Anthropic requires max_tokens > thinking.budget_tokens. Bump the cap
    // upward if the consumer's max_tokens is smaller than the requested
    // thinking budget.
    if let Some(t) = thinking.as_ref() {
        if let Some(budget) = t.budget_tokens {
            if max_tokens <= budget {
                max_tokens = budget
                    .saturating_add(1024)
                    .min(default_max_tokens(model_def));
            }
        }
    }

    // When thinking is enabled, Anthropic requires temperature=1 and rejects
    // top_p/top_k/stop_sequences. Coerce to compatible values.
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
    } else if req.temperature.is_some() && req.top_p.is_some() {
        tracing::debug!(
            "temperature and top_p both set; dropping top_p for Anthropic compatibility"
        );
        None
    } else {
        req.top_p
    };
    let top_k = if thinking_active { None } else { req.top_k };
    let stop_sequences = if thinking_active {
        None
    } else {
        stop_sequences
    };

    let metadata = req
        .metadata
        .as_ref()
        .and_then(value_to_metadata)
        .or_else(|| {
            req.user.as_ref().map(|u| Metadata {
                user_id: Some(u.clone()),
            })
        });

    if req.response_format.is_some() {
        tracing::warn!("response_format is not supported on Anthropic Messages — dropped");
    }
    if req.seed.is_some() {
        tracing::debug!("seed is not supported on Anthropic Messages — dropped");
    }
    if req.presence_penalty.is_some() || req.frequency_penalty.is_some() {
        tracing::debug!("presence_penalty/frequency_penalty are not supported — dropped");
    }

    Ok(MessagesRequest {
        model: model_def.canonical.to_string(),
        max_tokens,
        messages: reshaped.messages,
        system: reshaped.system,
        tools,
        tool_choice,
        temperature,
        top_p,
        top_k,
        stop_sequences,
        stream: Some(req.stream),
        metadata,
        thinking,
    })
}

fn default_max_tokens(model_def: &ModelDef) -> u32 {
    // Cap at u32::MAX-safe; ModelDef.max_tokens is u64 but Anthropic accepts u32.
    model_def.max_tokens.min(u32::MAX as u64) as u32
}

/// Map either an explicit `thinking` value or the `reasoning_effort` knob to
/// an Anthropic `thinking` field.
fn derive_thinking(req: &ChatCompletionRequest, model_def: &ModelDef) -> Option<Thinking> {
    // Explicit pass-through wins.
    if let Some(v) = &req.thinking {
        if let Some(obj) = v.as_object() {
            let kind = obj
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("enabled")
                .to_string();
            let budget_tokens = obj
                .get("budget_tokens")
                .and_then(|b| b.as_u64())
                .map(|b| b as u32);
            return Some(Thinking {
                kind,
                budget_tokens,
            });
        }
    }

    // Otherwise derive from reasoning_effort.
    match req.reasoning_effort.as_deref() {
        Some("low") => Some(Thinking {
            kind: "enabled".into(),
            budget_tokens: Some(budget_for_effort("low", model_def)),
        }),
        Some("medium") => Some(Thinking {
            kind: "enabled".into(),
            budget_tokens: Some(budget_for_effort("medium", model_def)),
        }),
        Some("high") => Some(Thinking {
            kind: "enabled".into(),
            budget_tokens: Some(budget_for_effort("high", model_def)),
        }),
        Some("max") => Some(Thinking {
            kind: "enabled".into(),
            budget_tokens: Some(budget_for_effort("max", model_def)),
        }),
        _ => None, // omit; Anthropic default is no thinking
    }
}

fn budget_for_effort(effort: &str, _model_def: &ModelDef) -> u32 {
    // Conservative defaults; Phase 4 may revisit by capturing claude itself.
    match effort {
        "low" => 1024,
        "medium" => 8192,
        "high" => 16384,
        "max" => 32768,
        _ => 0,
    }
}

fn value_to_metadata(v: &Value) -> Option<Metadata> {
    let obj = v.as_object()?;
    let user_id = obj.get("user_id").and_then(|x| match x {
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    });
    if user_id.is_none() {
        return None;
    }
    Some(Metadata { user_id })
}
