//! Request-scoped notes for caller fields that a provider dropped or relaxed
//! before it sent the request upstream.
//!
//! - The Codex ChatGPT WebSocket path drops a caller output cap.
//! - The Claude provider relaxes a forced tool choice (`any` / `tool`) to
//!   `auto`, because current Claude models reject forced choices.
//!
//! The provider records the note on the same task as the HTTP handler, just
//! before it sends the request upstream. A task-local carries it to that
//! handler without changing `LlmProvider`. The handler logs one warning per
//! note and merges a token into `x-omni-dropped`.

use std::cell::RefCell;
use std::future::Future;

pub const DROPPED_OUTPUT_CAP_HEADER: &str = "x-omni-dropped";
pub const DROPPED_OUTPUT_CAP_HEADER_VALUE: &str = "max_output_tokens";
/// `x-omni-dropped` token for a forced tool choice sent as `auto`.
pub const DROPPED_TOOL_CHOICE_HEADER_VALUE: &str = "tool_choice";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DroppedOutputCap {
    pub model: String,
    pub route: &'static str,
    pub requested: u32,
}

/// A forced tool choice that was sent upstream as `auto`. `kind` is the
/// forced Anthropic choice type (`any` or `tool`). It never holds a tool name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedToolChoice {
    pub model: String,
    pub route: &'static str,
    pub kind: &'static str,
}

struct InboundRequest {
    route: &'static str,
    dropped: Option<RecordedDrop>,
    tool_choice: Option<RecordedToolChoice>,
}

struct RecordedDrop {
    model: String,
    requested: u32,
}

struct RecordedToolChoice {
    model: String,
    kind: &'static str,
}

tokio::task_local! {
    static INBOUND_REQUEST: RefCell<InboundRequest>;
}

/// Run `fut` as one inbound request on `route`.
pub async fn with_inbound_route<F>(route: &'static str, fut: F) -> F::Output
where
    F: Future,
{
    INBOUND_REQUEST
        .scope(
            RefCell::new(InboundRequest {
                route,
                dropped: None,
                tool_choice: None,
            }),
            fut,
        )
        .await
}

/// Record that this request's caller output cap was removed.
///
/// The first value wins. A second record does not replace it, so one request
/// cannot log two caps. Outside an inbound scope the header cannot be set;
/// that is logged as an error.
pub fn note_dropped_output_cap(model: &str, requested: u32) {
    let recorded = INBOUND_REQUEST.try_with(|note| {
        let mut note = note.borrow_mut();
        if note.dropped.is_none() {
            note.dropped = Some(RecordedDrop {
                model: model.to_string(),
                requested,
            });
        }
    });
    if recorded.is_err() {
        tracing::error!(
            %model,
            requested,
            "dropped caller output cap with no inbound request scope; response header was not recorded"
        );
    }
}

/// The cap dropped on this inbound request, if any.
pub fn dropped_output_cap() -> Option<DroppedOutputCap> {
    INBOUND_REQUEST
        .try_with(|note| {
            let note = note.borrow();
            note.dropped.as_ref().map(|dropped| DroppedOutputCap {
                model: dropped.model.clone(),
                route: note.route,
                requested: dropped.requested,
            })
        })
        .ok()
        .flatten()
}

/// Record that this request's forced tool choice was sent as `auto`.
///
/// Call it only when the request is about to go upstream, so requests that
/// fail locally do not report it. The first value wins, so one request logs
/// one warning. Outside an inbound scope the header cannot be set; that is
/// logged as an error.
pub fn note_relaxed_tool_choice(model: &str, kind: &'static str) {
    let recorded = INBOUND_REQUEST.try_with(|note| {
        let mut note = note.borrow_mut();
        if note.tool_choice.is_none() {
            note.tool_choice = Some(RecordedToolChoice {
                model: model.to_string(),
                kind,
            });
        }
    });
    if recorded.is_err() {
        tracing::error!(
            %model,
            kind,
            "relaxed forced tool_choice with no inbound request scope; response header was not recorded"
        );
    }
}

/// The forced tool choice relaxed on this inbound request, if any.
pub fn relaxed_tool_choice() -> Option<RelaxedToolChoice> {
    INBOUND_REQUEST
        .try_with(|note| {
            let note = note.borrow();
            note.tool_choice.as_ref().map(|relaxed| RelaxedToolChoice {
                model: relaxed.model.clone(),
                route: note.route,
                kind: relaxed.kind,
            })
        })
        .ok()
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn relaxed_tool_choice_is_request_scoped_and_first_wins() {
        assert_eq!(relaxed_tool_choice(), None);
        let seen = with_inbound_route("/v1/messages", async {
            assert_eq!(relaxed_tool_choice(), None);
            note_relaxed_tool_choice("claude-opus-5-5", "tool");
            note_relaxed_tool_choice("other", "any");
            (relaxed_tool_choice(), dropped_output_cap())
        })
        .await;
        assert_eq!(
            seen,
            (
                Some(RelaxedToolChoice {
                    model: "claude-opus-5-5".into(),
                    route: "/v1/messages",
                    kind: "tool",
                }),
                None
            )
        );
        // A later request on the same task starts clean.
        let next = with_inbound_route("/v1/responses", async { relaxed_tool_choice() }).await;
        assert_eq!(next, None);
    }

    #[tokio::test]
    async fn relaxed_tool_choice_and_output_cap_coexist() {
        let (cap, relaxed) = with_inbound_route("/v1/chat/completions", async {
            note_dropped_output_cap("m", 5);
            note_relaxed_tool_choice("m", "any");
            (dropped_output_cap(), relaxed_tool_choice())
        })
        .await;
        assert_eq!(cap.map(|cap| cap.requested), Some(5));
        assert_eq!(relaxed.map(|relaxed| relaxed.kind), Some("any"));
    }
}
