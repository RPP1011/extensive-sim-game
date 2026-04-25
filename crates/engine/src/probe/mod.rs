//! Probes — scripted smoke tests that drive the tick pipeline with a
//! fixed seed + spawn + tick count, then assert on resulting state +
//! events.
//!
//! A probe is a plain `struct`; [`run_probe`] spins up a fresh
//! `SimState` at the seed, runs `p.spawn` to populate agents, ticks the
//! pipeline `p.ticks` times via [`crate::step::step`], then calls
//! `p.assert` with the final `(&SimState, &EventRing)`.
//!
//! See `docs/engine/spec.md` §18.

use crate::cascade::CascadeRegistry;
use crate::event::EventRing;
use crate::policy::UtilityBackend;
use crate::state::SimState;
use crate::step::{step, SimScratch};

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
    pub assert: fn(&SimState, &EventRing) -> Result<(), String>,
}

/// Default agent-cap for probes. Chosen to leave headroom for announce
/// cascades etc. Adjust the probe harness (e.g. a richer `ProbeConfig`)
/// if a probe needs more.
pub const DEFAULT_AGENT_CAP: u32 = 256;
/// Default event-ring capacity for probes.
pub const DEFAULT_EVENT_CAP: usize = 4096;

/// Run a probe. Returns `Ok(())` on pass; on failure, returns a string
/// prefixed with `"probe '<name>': "` so the test output identifies
/// which probe broke.
pub fn run_probe(p: &Probe) -> Result<(), String> {
    let mut state = SimState::new(DEFAULT_AGENT_CAP, p.seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(DEFAULT_EVENT_CAP);
    let cascade = CascadeRegistry::new();

    (p.spawn)(&mut state);
    for _ in 0..p.ticks {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    (p.assert)(&state, &events).map_err(|e| format!("probe '{}': {}", p.name, e))
}
