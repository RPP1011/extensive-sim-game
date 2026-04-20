use super::handler::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

/// Dense slot count covering all `EventKindId` ordinals — includes the 128+
/// chronicle reservation. `Vec<Vec<Box<dyn CascadeHandler>>>` indexed by
/// `[lane as usize][kind as u8 as usize]`.
const KIND_SLOTS: usize = 256;

/// Maximum number of cascade dispatch passes per `run_fixed_point` call.
/// If handlers keep pushing new events beyond this bound, the cascade is
/// considered non-converging: dev builds panic, release builds log and
/// truncate.
pub const MAX_CASCADE_ITERATIONS: usize = 8;

pub struct CascadeRegistry {
    table: Vec<Vec<Vec<Box<dyn CascadeHandler>>>>,
}

impl CascadeRegistry {
    pub fn new() -> Self {
        let per_lane: Vec<Vec<Box<dyn CascadeHandler>>> =
            (0..KIND_SLOTS).map(|_| Vec::new()).collect();
        Self {
            table: (0..Lane::ALL.len()).map(|_| per_lane.iter().map(|_| Vec::new()).collect()).collect(),
        }
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
        self.register(crate::ability::expire::OpportunityAttackHandler);
        // Combat Foundation Task 10 — effect fold-ins. These handlers pair up
        // with the `Effect*Applied` events the `CastHandler` emits (Task 9)
        // and are the actual state mutators for the combat EffectOps.
        self.register(crate::ability::DamageHandler);
        self.register(crate::ability::HealHandler);
        self.register(crate::ability::ShieldHandler);
    }

    /// Register the Combat Foundation Task 9 `CastHandler` against an
    /// `AbilityRegistry`. Kept off `register_engine_builtins` because it
    /// requires a built registry — callers construct the registry, wrap
    /// it in an `Arc`, and then register the cast handler once at startup.
    /// Calling this twice registers two handlers that both dispatch
    /// `AgentCast`; tests that need registry isolation should hand out
    /// distinct `Arc`s to distinct `CascadeRegistry`s.
    pub fn register_cast_handler(
        &mut self,
        ability_registry: std::sync::Arc<crate::ability::AbilityRegistry>,
    ) {
        self.register(crate::ability::CastHandler::new(ability_registry));
    }

    pub fn register<H: CascadeHandler + 'static>(&mut self, h: H) {
        let lane = h.lane() as usize;
        let kind = h.trigger() as u8 as usize;
        self.table[lane][kind].push(Box::new(h));
    }

    pub fn dispatch(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        let kind = EventKindId::from_event(event) as u8 as usize;
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
    pub fn run_fixed_point(&self, state: &mut SimState, events: &mut EventRing) {
        let mut processed = events.dispatched();
        for iter in 0..MAX_CASCADE_ITERATIONS {
            let snapshot = events.total_pushed();
            if snapshot == processed {
                events.set_dispatched(processed);
                return;
            }
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
    }
}

impl Default for CascadeRegistry {
    fn default() -> Self { Self::new() }
}
