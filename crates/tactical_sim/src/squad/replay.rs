use std::collections::HashMap;

use crate::sim::{run_replay, step, ReplayResult, SimState, Team, UnitIntent};

use super::intents::generate_intents;
use super::personality::Personality;
use super::state::{SquadAiState, SquadBlackboard};

// ---------------------------------------------------------------------------
// Public helpers for replays and benchmarks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Phase3Run {
    pub replay: ReplayResult,
    pub board_history: Vec<HashMap<Team, SquadBlackboard>>,
}

pub fn sample_phase3_party_state(seed: u64) -> SimState {
    crate::roles::sample_phase2_party_state(seed)
}

pub fn generate_scripted_intents(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
    personalities: HashMap<u32, Personality>,
) -> (Vec<Vec<UnitIntent>>, Vec<HashMap<Team, SquadBlackboard>>) {
    let mut ai = SquadAiState::new(initial, personalities);
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(ticks as usize);
    let mut board_history = Vec::with_capacity(ticks as usize);

    for _ in 0..ticks {
        let intents = generate_intents(&state, &mut ai, dt_ms);
        board_history.push(ai.blackboard_by_team.clone());
        script.push(intents.clone());
        let (new_state, _events) = step(state, &intents, dt_ms);
        state = new_state;
    }

    (script, board_history)
}

/// Convenience: generate scripted intents from a role map (backward compat).
pub fn generate_scripted_intents_from_roles(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
    roles: HashMap<u32, crate::roles::Role>,
) -> (Vec<Vec<UnitIntent>>, Vec<HashMap<Team, SquadBlackboard>>) {
    let personalities: HashMap<u32, Personality> = roles
        .into_iter()
        .map(|(id, role)| (id, super::state::role_to_personality(role)))
        .collect();
    generate_scripted_intents(initial, ticks, dt_ms, personalities)
}

/// Infer personalities for all units and generate scripted intents.
pub fn generate_scripted_intents_inferred(
    initial: &SimState,
    ticks: u32,
    dt_ms: u32,
) -> (Vec<Vec<UnitIntent>>, Vec<HashMap<Team, SquadBlackboard>>) {
    let mut ai = SquadAiState::new_inferred(initial);
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(ticks as usize);
    let mut board_history = Vec::with_capacity(ticks as usize);

    for _ in 0..ticks {
        let intents = generate_intents(&state, &mut ai, dt_ms);
        board_history.push(ai.blackboard_by_team.clone());
        script.push(intents.clone());
        let (new_state, _events) = step(state, &intents, dt_ms);
        state = new_state;
    }

    (script, board_history)
}

pub fn run_phase3_sample(seed: u64, ticks: u32, dt_ms: u32) -> Phase3Run {
    let initial = sample_phase3_party_state(seed);
    let (script, board_history) =
        generate_scripted_intents_inferred(&initial, ticks, dt_ms);
    let replay = run_replay(initial, &script, ticks, dt_ms);
    Phase3Run {
        replay,
        board_history,
    }
}
