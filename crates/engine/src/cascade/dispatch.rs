use super::handler::{CascadeHandler, EventKindId, Lane};
use crate::event::{EventLike, EventRing};
use crate::state::SimState;
use crate::telemetry::{metrics, TelemetrySink};

/// Dense slot count covering all `EventKindId` ordinals — includes the 128+
/// chronicle reservation. `Vec<Vec<Box<dyn CascadeHandler>>>` indexed by
/// `[lane as usize][kind as u8 as usize]`.
const KIND_SLOTS: usize = 256;

/// Maximum number of cascade dispatch passes per `run_fixed_point` call.
/// If handlers keep pushing new events beyond this bound, the cascade is
/// considered non-converging: dev builds panic, release builds log and
/// truncate.
pub const MAX_CASCADE_ITERATIONS: usize = 8;

/// Type signature of a compiler-emitted per-event-kind dispatcher. The
/// dispatcher destructures the triggering event once and fans the call
/// out to every applicable handler (kind-specific + tag-matched).
/// The `V` parameter is the views type threaded through all handlers.
pub type KindDispatcher<E, V> = fn(&E, &mut SimState, &mut V, &mut EventRing<E>);

pub struct CascadeRegistry<E: EventLike, V = ()> {
    table: Vec<Vec<Vec<Box<dyn __object_safe::DynHandler<E, V>>>>>,
    /// Compiler-emitted per-event-kind dispatcher fns. Indexed by
    /// `EventKindId as u8 as usize`; `None` means no dispatcher is
    /// installed for that kind (falls back to per-handler trait-object
    /// dispatch via `table`).
    kind_dispatchers: Vec<Option<KindDispatcher<E, V>>>,
}

/// Object-safe wrapper trait used for the boxed handler table.
/// Not public — internal to the dispatch machinery.
mod __object_safe {
    use crate::event::{EventLike, EventRing};
    use crate::state::SimState;

    pub trait DynHandler<E: EventLike, V>: Send + Sync {
        #[allow(dead_code)]
        fn trigger_kind(&self) -> u8;
        #[allow(dead_code)]
        fn lane_ord(&self) -> u8;
        fn handle_dyn(&self, event: &E, state: &mut SimState, views: &mut V, events: &mut EventRing<E>);
    }

    impl<E: EventLike, H: super::super::handler::CascadeHandler<E>> DynHandler<E, H::Views>
        for H
    where
        H: Send + Sync,
    {
        fn trigger_kind(&self) -> u8 { self.trigger() as u8 }
        fn lane_ord(&self) -> u8 { self.lane() as u8 }
        fn handle_dyn(&self, event: &E, state: &mut SimState, views: &mut H::Views, events: &mut EventRing<E>) {
            self.handle(event, state, views, events);
        }
    }
}

impl<E: EventLike, V> CascadeRegistry<E, V> {
    pub fn new() -> Self {
        Self {
            table: (0..Lane::ALL.len()).map(|_| (0..KIND_SLOTS).map(|_| Vec::new()).collect()).collect(),
            kind_dispatchers: (0..KIND_SLOTS).map(|_| None).collect(),
        }
    }

    /// Install a compiler-emitted per-event-kind dispatcher. Overwrites
    /// any previously installed dispatcher for the same kind — the DSL
    /// emitter produces one dispatcher per event kind, so reinstallation
    /// is idempotent within a single registration call.
    pub fn install_kind(&mut self, kind: EventKindId, dispatcher: KindDispatcher<E, V>) {
        let idx = kind as u8 as usize;
        self.kind_dispatchers[idx] = Some(dispatcher);
    }

    pub fn register<H: CascadeHandler<E, Views = V> + 'static>(&mut self, h: H) {
        let lane = h.lane() as usize;
        let kind = h.trigger() as u8 as usize;
        self.table[lane][kind].push(Box::new(h));
    }

    pub fn dispatch(&self, event: &E, state: &mut SimState, views: &mut V, events: &mut EventRing<E>) {
        let kind = event.kind() as u8 as usize;
        // Prefer the compiler-emitted per-kind dispatcher when installed.
        // It fans out to every applicable handler (kind-specific +
        // tag-matched) inline — no runtime handler-list walk.
        if let Some(dispatcher) = self.kind_dispatchers[kind] {
            dispatcher(event, state, views, events);
        }
        // Legacy trait-object handlers still register via `register`;
        // walk them in lane order after the flat dispatcher.
        for lane in Lane::ALL {
            for handler in &self.table[*lane as usize][kind] {
                handler.handle_dyn(event, state, views, events);
            }
        }
    }

    /// Dispatch any events pushed to `events` that haven't been dispatched yet,
    /// iterating until no new events are emitted, bounded by
    /// `MAX_CASCADE_ITERATIONS`. In dev builds non-convergence panics; in
    /// release it logs and truncates.
    ///
    /// Uses the ring's persistent `dispatched` cursor so multiple calls (e.g.
    /// one per tick) don't re-dispatch past events. Within a single call,
    /// iteration continues as long as handlers push new events, up to the
    /// iteration bound.
    ///
    /// Back-compat wrapper over [`run_fixed_point_tel`] for call sites that
    /// don't have a telemetry sink (typically tests).
    pub fn run_fixed_point(&self, state: &mut SimState, views: &mut V, events: &mut EventRing<E>) {
        self.run_fixed_point_tel(state, views, events, &crate::telemetry::NullSink);
    }

    /// Like [`run_fixed_point`] but also emits the
    /// `metrics::CASCADE_ITERATIONS` histogram metric once per call, counting
    /// the number of dispatch passes taken (0 when the initial ring cursor is
    /// already current — no events to drain). Audit fix HIGH #5.
    pub fn run_fixed_point_tel(
        &self,
        state:     &mut SimState,
        views:     &mut V,
        events:    &mut EventRing<E>,
        telemetry: &dyn TelemetrySink,
    ) {
        let mut processed = events.dispatched();
        let mut iterations: usize = 0;
        for iter in 0..MAX_CASCADE_ITERATIONS {
            let snapshot = events.total_pushed();
            if snapshot == processed {
                events.set_dispatched(processed);
                telemetry.emit_histogram(metrics::CASCADE_ITERATIONS, iterations as f64);
                return;
            }
            iterations = iter + 1;
            for idx in processed..snapshot {
                if let Some(e) = events.get_pushed(idx) {
                    self.dispatch(&e, state, views, events);
                }
            }
            processed = snapshot;
            if iter == MAX_CASCADE_ITERATIONS - 1 {
                // Check again — if handlers emitted MORE events in the last pass,
                // we're about to truncate.
                if events.total_pushed() > processed {
                    #[cfg(debug_assertions)]
                    panic!(
                        "cascade did not converge within {} iterations (tick pushes: {} → {})",
                        MAX_CASCADE_ITERATIONS, processed, events.total_pushed()
                    );
                    #[cfg(not(debug_assertions))]
                    eprintln!(
                        "cascade truncated at {} iterations",
                        MAX_CASCADE_ITERATIONS,
                    );
                }
            }
        }
        events.set_dispatched(events.total_pushed());
        telemetry.emit_histogram(metrics::CASCADE_ITERATIONS, iterations as f64);
    }
}

impl<E: EventLike, V> Default for CascadeRegistry<E, V> {
    fn default() -> Self { Self::new() }
}

// `with_engine_builtins` was deleted along with engine/src/generated/. The
// replacement is compiler-emitted into engine_rules/src/cascade.rs (Task 11
// of Plan B1').
