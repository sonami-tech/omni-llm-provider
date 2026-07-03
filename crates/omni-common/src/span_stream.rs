//! A stream adapter that keeps a `tracing` span entered for the duration of each
//! poll, without holding the span guard across `.await` points.
//!
//! SSE response bodies outlive the handler that built them, so any operational
//! log emitted while the body is being polled (including from inside the shared
//! serializers here) would otherwise lose the request's correlation fields. The
//! naive fix -- `let _g = span.enter();` at the top of the generator -- is the
//! documented `tracing` anti-pattern: the guard survives across `.await`, so the
//! span stays entered on the worker thread while the task is suspended and a
//! different task resuming on that thread inherits it. Entering the span per
//! poll (and exiting before the poll returns) is the correct, leak-free shape.
//!
//! Capturing `Span::current()` at construction is ambient-context propagation
//! (like `tracing::Instrument`), not frontend-specific knowledge, so it belongs
//! in this shared crate: it preserves whatever span exists when a serializer is
//! invoked, for every caller.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;

/// Wraps a stream so `span` is entered for each `poll_next` and exited before it
/// returns. `Pin<Box<S>>` is `Unpin`, so no pin projection is needed. `S` may be
/// `?Sized`, so a boxed trait object (`Pin<Box<dyn Stream>>`) works directly.
pub struct SpannedStream<S: ?Sized> {
    span: tracing::Span,
    inner: Pin<Box<S>>,
}

impl<S: ?Sized> SpannedStream<S> {
    /// Wrap `inner`, entering `span` on every poll.
    pub fn new(inner: Pin<Box<S>>, span: tracing::Span) -> Self {
        Self { span, inner }
    }

    /// Wrap `inner` under the span that is current at the call site.
    pub fn current(inner: Pin<Box<S>>) -> Self {
        Self::new(inner, tracing::Span::current())
    }
}

impl<S: Stream + ?Sized> Stream for SpannedStream<S> {
    type Item = <S as Stream>::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let _entered = this.span.enter();
        this.inner.as_mut().poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt as _;

    // Two spanned streams driven concurrently on one thread must not bleed each
    // other's span fields. A guard-across-await adapter would fail this; the
    // per-poll adapter passes. Mirrors the bin crate's end-to-end isolation test
    // at the adapter level.
    #[test]
    fn per_poll_entry_does_not_bleed_across_concurrent_streams() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::layer::SubscriberExt as _;
        use tracing_subscriber::registry::LookupSpan;

        // Captures (probe field, current-span request_id) per event.
        #[derive(Clone, Default)]
        struct Probe(Arc<Mutex<Vec<(String, String)>>>);
        struct CapturedId(String);
        impl<S> tracing_subscriber::Layer<S> for Probe
        where
            S: tracing::Subscriber + for<'a> LookupSpan<'a>,
        {
            fn on_new_span(
                &self,
                attrs: &tracing::span::Attributes<'_>,
                id: &tracing::span::Id,
                ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                struct G(Option<String>);
                impl tracing::field::Visit for G {
                    fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
                        if f.name() == "rid" {
                            self.0 = Some(format!("{v:?}").trim_matches('"').to_string());
                        }
                    }
                }
                let mut g = G(None);
                attrs.record(&mut g);
                if let (Some(rid), Some(span)) = (g.0, ctx.span(id)) {
                    span.extensions_mut().insert(CapturedId(rid));
                }
            }
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                struct G(Option<String>);
                impl tracing::field::Visit for G {
                    fn record_str(&mut self, f: &tracing::field::Field, v: &str) {
                        if f.name() == "probe" {
                            self.0 = Some(v.to_string());
                        }
                    }
                    fn record_debug(
                        &mut self,
                        _f: &tracing::field::Field,
                        _v: &dyn std::fmt::Debug,
                    ) {
                    }
                }
                let mut g = G(None);
                event.record(&mut g);
                let Some(probe) = g.0 else { return };
                let mut rid = String::new();
                if let Some(scope) = ctx.event_scope(event) {
                    for span in scope.from_root() {
                        if let Some(c) = span.extensions().get::<CapturedId>() {
                            rid = c.0.clone();
                        }
                    }
                }
                self.0.lock().unwrap().push((probe, rid));
            }
        }

        fn probe_stream(tag: &'static str) -> Pin<Box<dyn Stream<Item = ()> + Send>> {
            Box::pin(async_stream::stream! {
                for _ in 0..6 {
                    tokio::task::yield_now().await;
                    tracing::info!(probe = tag, "tick");
                    tokio::task::yield_now().await;
                    yield ();
                }
            })
        }

        async fn drive(tag: &'static str, id: &'static str) {
            let span = tracing::info_span!("s", rid = %id);
            let mut s = span.in_scope(|| SpannedStream::current(probe_stream(tag)));
            while s.next().await.is_some() {}
        }

        let probe = Probe::default();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_subscriber::filter::LevelFilter::TRACE)
            .with(probe.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        tracing::subscriber::with_default(subscriber, || {
            rt.block_on(async {
                tokio::join!(drive("A", "aaaa"), drive("B", "bbbb"));
            });
        });

        let seen = probe.0.lock().unwrap();
        assert!(!seen.is_empty(), "probe captured nothing; test is vacuous");
        for (probe_tag, rid) in seen.iter() {
            let expected = match probe_tag.as_str() {
                "A" => "aaaa",
                "B" => "bbbb",
                other => panic!("unexpected probe {other:?}"),
            };
            assert_eq!(rid, expected, "span bled: probe={probe_tag} saw rid={rid}");
        }
    }
}
