use serde::{Deserialize, Serialize};

use super::types::*;
use super::events::SimEvent;
use super::simulation::step;
use super::metrics::compute_metrics;
use super::determinism::{hash_sim_state, hash_event_log};

pub fn run_replay(
    initial_state: SimState,
    scripted_intents: &[Vec<UnitIntent>],
    ticks: u32,
    dt_ms: u32,
) -> ReplayResult {
    let mut state = initial_state.clone();
    let mut all_events = Vec::new();
    let mut per_tick_state_hashes = Vec::with_capacity(ticks as usize);

    for tick in 0..ticks {
        let intents = scripted_intents
            .get(tick as usize)
            .map_or(&[][..], |v| v.as_slice());
        let (new_state, events) = step(state, intents, dt_ms);
        state = new_state;
        all_events.extend(events);
        per_tick_state_hashes.push(hash_sim_state(&state));
    }

    let event_log_hash = hash_event_log(&all_events);
    let final_state_hash = hash_sim_state(&state);
    let metrics = compute_metrics(
        &initial_state, &state, scripted_intents, &all_events, ticks, dt_ms,
    );

    ReplayResult {
        final_state: state,
        events: all_events,
        event_log_hash,
        final_state_hash,
        per_tick_state_hashes,
        metrics,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    pub final_state: SimState,
    pub events: Vec<SimEvent>,
    pub event_log_hash: u64,
    pub final_state_hash: u64,
    pub per_tick_state_hashes: Vec<u64>,
    pub metrics: SimMetrics,
}
