use crate::sim::{SimEvent, SimState, UnitIntent};

/// Common interface for phase-level AI state machines.
///
/// Every AI phase that manages per-tick intent generation and optional
/// event-driven state updates can implement this trait.  The default
/// `update_from_events` is a no-op so phases that do not consume events
/// (e.g. [`crate::squad::SquadAiState`]) need not override it.
#[allow(dead_code)]
pub trait AiPhase {
    /// Produce one intent per living unit for the current tick.
    fn generate_intents(&mut self, state: &SimState, dt_ms: u32) -> Vec<UnitIntent>;

    /// Incorporate tick events into internal bookkeeping (threat tables,
    /// CC windows, etc.).  Default implementation does nothing.
    fn update_from_events(&mut self, _events: &[SimEvent]) {}
}
