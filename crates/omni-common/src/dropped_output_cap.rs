//! Request-scoped note that the Codex ChatGPT WebSocket path dropped a caller
//! output cap.
//!
//! `send` / `send_stream` remove the cap before they return, on the same task
//! as the HTTP handler. A task-local carries the model and requested value to
//! that handler without changing `LlmProvider`. The handler logs one warning
//! and sets `x-omni-dropped`.

use std::cell::RefCell;
use std::future::Future;

pub const DROPPED_OUTPUT_CAP_HEADER: &str = "x-omni-dropped";
pub const DROPPED_OUTPUT_CAP_HEADER_VALUE: &str = "max_output_tokens";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DroppedOutputCap {
    pub model: String,
    pub route: &'static str,
    pub requested: u32,
}

struct InboundRequest {
    route: &'static str,
    dropped: Option<RecordedDrop>,
}

struct RecordedDrop {
    model: String,
    requested: u32,
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
