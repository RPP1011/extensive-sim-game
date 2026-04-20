use super::handler::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
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
pub type KindDispatcher = fn(&Event, &mut SimState, &mut EventRing);

pub struct CascadeRegistry {
    table: Vec<Vec<Vec<Box<dyn CascadeHandler>>>>,
    /// Compiler-emitted per-event-kind dispatcher fns. Indexed by
    /// `EventKindId as u8 as usize`; `None` means no dispatcher is
    /// installed for that kind (falls back to per-handler trait-object
    /// dispatch via `table`).
    kind_dispatchers: Vec<Option<KindDispatcher>>,
}

impl CascadeRegistry {
    pub fn new() -> Self {
        let per_lane: Vec<Vec<Box<dyn CascadeHandler>>> =
            (0..KIND_SLOTS).map(|_| Vec::new()).collect();
        Self {
            table: (0..Lane::ALL.len()).map(|_| per_lane.iter().map(|_| Vec::new()).collect()).collect(),
            kind_dispatchers: (0..KIND_SLOTS).map(|_| None).collect(),
        }
    }

    /// Install a compiler-emitted per-event-kind dispatcher. Overwrites
    /// any previously installed dispatcher for the same kind — the DSL
    /// emitter produces one dispatcher per event kind, so reinstallation
    /// is idempotent within a single `register_engine_builtins` call.
    pub fn install_kind(&mut self, kind: EventKindId, dispatcher: KindDispatcher) {
        let idx = kind as u8 as usize;
        self.kind_dispatchers[idx] = Some(dispatcher);
    }

    /// Convenience constructor that pre-registers engine-defined baseline
    /// handlers (e.g. the opportunity-attack cascade for engagement disengage
    /// from Combat Foundation Task 4). Tests that need a fully empty registry
    /// for isolation should use [`CascadeRegistry::new`].
    pub fn with_engine_builtins() -> Self {
        let mut reg = Self::new();
        reg.register_engine_builtins();
        reg
    }

    /// Register the engine's baseline cascade handlers on an existing registry.
    /// Idempotent only in the sense that calling it twice registers the
    /// handler twice — callers should invoke once, typically via
    /// [`CascadeRegistry::with_engine_builtins`].
    pub fn register_engine_builtins(&mut self) {
        // Compiler-emitted physics handlers (DSL-owned). Covers damage, heal,
        // shield, stun, slow, transfer_gold, modify_standing,
        // opportunity_attack, record_memory, and cast. The matching hand-
        // written legacy handlers were deleted in the same commit that
        // landed their DSL equivalent. 2026-04-19: `CastHandler` migrated
        // — the `emit_physics` compiler now lowers `for ... in
        // abilities.effects(...)` loops and `match` over `EffectOp`
        // variants, so the last hand-written cascade handler with game
        // logic is retired. The ability registry continues to live on
        // `SimState`; the DSL `physics cast` rule reaches it through the
        // `abilities.*` stdlib namespace.
        crate::generated::physics::register(self);
        // Task 139 — event-driven engagement update. The old tick-start
        // tentative-commit loop was retired in favour of two cascade
        // dispatchers keyed on `AgentMoved` / `AgentDied`. The DSL physics
        // emitter can't yet lower the spatial query the mover-scan needs
        // (`query.nearby_agents(...)` is mask-only), so these stay
        // hand-written for now.
        self.install_kind(
            super::EventKindId::AgentMoved,
            crate::engagement::dispatch_agent_moved,
        );
        self.install_kind(
            super::EventKindId::AgentDied,
            crate::engagement::dispatch_agent_died,
        );
    }

    pub fn register<H: CascadeHandler + 'static>(&mut self, h: H) {
        let lane = h.lane() as usize;
        let kind = h.trigger() as u8 as usize;
        self.table[lane][kind].push(Box::new(h));
    }

    pub fn dispatch(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        let kind = EventKindId::from_event(event) as u8 as usize;
        // Prefer the compiler-emitted per-kind dispatcher when installed.
        // It fans out to every applicable handler (kind-specific +
        // tag-matched) inline — no runtime handler-list walk.
        if let Some(dispatcher) = self.kind_dispatchers[kind] {
            dispatcher(event, state, events);
        }
        // Legacy trait-object handlers (`CastHandler`) still register via
        // `register`; walk them in lane order after the flat dispatcher.
        for lane in Lane::ALL {
            for handler in &self.table[*lane as usize][kind] {
                handler.handle(event, state, events);
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
    pub fn run_fixed_point(&self, state: &mut SimState, events: &mut EventRing) {
        self.run_fixed_point_tel(state, events, &crate::telemetry::NullSink);
    }

    /// Like [`run_fixed_point`] but also emits the
    /// `metrics::CASCADE_ITERATIONS` histogram metric once per call, counting
    /// the number of dispatch passes taken (0 when the initial ring cursor is
    /// already current — no events to drain). Audit fix HIGH #5.
    pub fn run_fixed_point_tel(
        &self,
        state:     &mut SimState,
        events:    &mut EventRing,
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
                    self.dispatch(&e, state, events);
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

impl Default for CascadeRegistry {
    fn default() -> Self { Self::new() }
}
