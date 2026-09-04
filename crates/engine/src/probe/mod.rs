//! Probes — scripted smoke tests that drive the tick pipeline with a
//! fixed seed + spawn + tick count, then assert on resulting state +
//! events.
//!
//! A probe is a plain `struct`; [`run_probe`] will spin up a fresh
//! `SimState`, run `p.spawn` to populate agents, tick the pipeline
//! `p.ticks` times, and call `p.assert` with the final state + events.
//!
//! NOTE: `run_probe` is UNIMPLEMENTED; the tick driver lives in
//! `engine_rules::step::step`. Callers of `run_probe` (e.g. `probe_determinism`
//! test) are `#[ignore]`d.
//!
//! See `docs/engine/spec.md` §18.

use crate::event::EventRing;
use crate::state::SimState;

// Use the concrete Event type from engine_data.
use engine_data::events::Event;
type SimEventRing = EventRing<Event>;

/// A scripted probe. Fields are public so callers can construct one
/// inline in a test function.
pub struct Probe {
    /// Reported back in the error message when `assert` returns `Err`.
    pub name: &'static str,
    /// PCG seed for the simulation RNG.
    pub seed: u64,
    /// Spawn callback — populates initial agents before tick 0.
    pub spawn: fn(&mut SimState),
    /// Number of ticks to run through the pipeline.
    pub ticks: u32,
    /// Assertion callback. Return `Ok(())` to pass, `Err(msg)` to fail.
    pub assert: fn(&SimState, &SimEventRing) -> Result<(), String>,
}

/// Default agent-cap for probes. Chosen to leave headroom for announce
/// cascades etc.
pub const DEFAULT_AGENT_CAP: u32 = 256;
/// Default event-ring capacity for probes.
pub const DEFAULT_EVENT_CAP: usize = 4096;

/// Run a probe.
///
/// UNIMPLEMENTED: the tick driver lives in `engine_rules::step::step`.
/// Callers of this function must be `#[ignore]`d.
pub fn run_probe(_p: &Probe) -> Result<(), String> {
    unimplemented!(
        "engine::probe::run_probe: the tick driver is engine_rules::step::step"
    )
}
