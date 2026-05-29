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

    // A tool_choice that forces or selects a tool is meaningless without tools.
    // Surface a clear 400 instead of silently dropping the constraint (which
    // would turn "must call a tool" into "free text allowed").
    if tools.is_none() && tool_choice_requires_tool(&req.tool_choice) {
        return Err(AppError::BadRequest(
            "tool_choice requires or selects a tool, but no tools were provided".into(),
        ));
    }

    let tool_choice = if tools.is_some() {
        // OpenAI parallel_tool_calls is the inverse of Anthropic
        // disable_parallel_tool_use: parallel_tool_calls:false => disable:true.
        let disable_parallel = req.parallel_tool_calls.map(|p| !p);
        translate_tool_choice(&req.tool_choice, disable_parallel)
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
    if let Some(t) = thinking.as_ref()
        && let Some(budget) = t.budget_tokens
            && max_tokens <= budget {
                max_tokens = budget
                    .saturating_add(1024)
                    .min(default_max_tokens(model_def));
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
        output_config: None,
    })
}

/// Whether a tool_choice forces or selects a specific tool (and therefore needs
/// at least one tool to be present). `auto`/`none`/unknown modes do not.
fn tool_choice_requires_tool(choice: &Option<crate::translate::request::ToolChoice>) -> bool {
    use crate::translate::request::ToolChoice;
    match choice {
        Some(ToolChoice::Mode(s)) => matches!(s.as_str(), "required" | "any"),
        Some(ToolChoice::Specific { .. }) => true,
        _ => false,
    }
}

fn default_max_tokens(model_def: &ModelDef) -> u32 {
    // Cap at u32::MAX-safe; ModelDef.max_tokens is u64 but Anthropic accepts u32.
    model_def.max_tokens.min(u32::MAX as u64) as u32
}

/// Map either an explicit `thinking` value or the `reasoning_effort` knob to
/// an Anthropic `thinking` field.
fn derive_thinking(req: &ChatCompletionRequest, model_def: &ModelDef) -> Option<Thinking> {
    // Explicit pass-through wins.
    if let Some(v) = &req.thinking
        && let Some(obj) = v.as_object() {
            let kind = obj
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("enabled")
                .to_string();
            let budget_tokens = obj
                .get("budget_tokens")
                .and_then(|b| b.as_u64())
                // Clamp instead of silently truncating a huge JSON value.
                .map(|b| b.min(u32::MAX as u64) as u32);
            return Some(Thinking {
                kind,
                budget_tokens,
            });
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
    let user_id = obj.get("user_id").map(|x| match x {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    });
    user_id.as_ref()?;
    Some(Metadata { user_id })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::anthropic::ToolChoice as AnthToolChoice;
    use crate::translate::request::{
        ChatMessage, FunctionDefinition, MessageContent as OaiMessageContent, ToolChoice,
        ToolChoiceFunction, ToolDefinition,
    };
    use crate::upstream::fingerprint::default_profile;

    fn user_req() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "haiku".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(OaiMessageContent::Text("hi".into())),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn a_tool() -> ToolDefinition {
        ToolDefinition {
            tool_type: Some("function".into()),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: None,
                parameters: None,
            },
        }
    }

    #[test]
    fn forced_tool_choice_without_tools_is_rejected() {
        // F18: "required" with no tools must 400, not silently drop the
        // forced-tool constraint.
        let mut req = user_req();
        req.tool_choice = Some(ToolChoice::Mode("required".into()));
        let model_def = default_profile().resolve_model("haiku");
        let err = build_messages_request(&req, model_def).unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));

        // Specific tool selection with no tools is also rejected.
        let mut req2 = user_req();
        req2.tool_choice = Some(ToolChoice::Specific {
            choice_type: "function".into(),
            function: ToolChoiceFunction { name: "x".into() },
        });
        assert!(build_messages_request(&req2, model_def).is_err());
    }

    #[test]
    fn auto_and_none_tool_choice_without_tools_are_allowed() {
        let model_def = default_profile().resolve_model("haiku");
        for mode in ["auto", "none"] {
            let mut req = user_req();
            req.tool_choice = Some(ToolChoice::Mode(mode.into()));
            let out = build_messages_request(&req, model_def).unwrap();
            // No tools provided -> tool_choice is omitted entirely.
            assert!(out.tool_choice.is_none(), "mode {mode}");
        }
    }

    #[test]
    fn parallel_tool_calls_false_sets_disable_parallel_tool_use() {
        // F18: parallel_tool_calls:false must reach Anthropic as
        // disable_parallel_tool_use:true.
        let mut req = user_req();
        req.tools = Some(vec![a_tool()]);
        req.tool_choice = Some(ToolChoice::Mode("auto".into()));
        req.parallel_tool_calls = Some(false);
        let model_def = default_profile().resolve_model("haiku");
        let out = build_messages_request(&req, model_def).unwrap();
        match out.tool_choice {
            Some(AnthToolChoice::Auto { disable_parallel_tool_use }) => {
                assert_eq!(disable_parallel_tool_use, Some(true));
            }
            other => panic!("expected Auto with disable_parallel_tool_use, got {other:?}"),
        }
    }

    #[test]
    fn parallel_tool_calls_false_without_tool_choice_synthesizes_auto() {
        // Regression: tools present, NO tool_choice, parallel_tool_calls:false.
        // The disable flag has nowhere to attach unless we synthesize an
        // explicit `auto` — otherwise Anthropic defaults to parallel auto and
        // the constraint is silently lost.
        let mut req = user_req();
        req.tools = Some(vec![a_tool()]);
        req.tool_choice = None;
        req.parallel_tool_calls = Some(false);
        let model_def = default_profile().resolve_model("haiku");
        let out = build_messages_request(&req, model_def).unwrap();
        match out.tool_choice {
            Some(AnthToolChoice::Auto { disable_parallel_tool_use }) => {
                assert_eq!(disable_parallel_tool_use, Some(true));
            }
            other => panic!("expected synthesized Auto with disable flag, got {other:?}"),
        }
    }

    #[test]
    fn no_tool_choice_no_disable_omits_tool_choice() {
        // tools present, no tool_choice, parallel_tool_calls unset/true: still
        // omit tool_choice entirely (Anthropic defaults to auto).
        let model_def = default_profile().resolve_model("haiku");
        for ptc in [None, Some(true)] {
            let mut req = user_req();
            req.tools = Some(vec![a_tool()]);
            req.tool_choice = None;
            req.parallel_tool_calls = ptc;
            let out = build_messages_request(&req, model_def).unwrap();
            assert!(out.tool_choice.is_none(), "ptc={ptc:?} should omit tool_choice");
        }
    }

    #[test]
    fn parallel_tool_calls_true_or_unset_omits_disable_flag() {
        let model_def = default_profile().resolve_model("haiku");
        for ptc in [None, Some(true)] {
            let mut req = user_req();
            req.tools = Some(vec![a_tool()]);
            req.tool_choice = Some(ToolChoice::Mode("auto".into()));
            req.parallel_tool_calls = ptc;
            let out = build_messages_request(&req, model_def).unwrap();
            match out.tool_choice {
                Some(AnthToolChoice::Auto { disable_parallel_tool_use }) => {
                    assert_eq!(disable_parallel_tool_use, None, "ptc={ptc:?}");
                }
                other => panic!("expected Auto, got {other:?}"),
            }
        }
    }
}
