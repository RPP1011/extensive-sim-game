//! Quest generation — checked every 50 ticks.
//!
//! Generates new quest requests based on world state. Uses a Poisson-style
//! arrival rate scaled by guild reputation (higher rep = more quests).
//!
//! Ported from `crates/headless_campaign/src/systems/quest_generation.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

//              target_position: (f32, f32), potential_loot: bool }
//              reward_gold, target_position, potential_loot }

/// Base interval in ticks between quest arrivals (before reputation scaling).
const BASE_ARRIVAL_INTERVAL: u64 = 50;

/// Center point for reputation scaling (rep at this value = 1x rate).
const REPUTATION_SCALING_CENTER: f32 = 50.0;

/// Min/max reputation factor to avoid degenerate arrival rates.
const REPUTATION_FACTOR_MIN: f32 = 0.5;
const REPUTATION_FACTOR_MAX: f32 = 3.0;

pub fn compute_quest_generation(state: &WorldState, _out: &mut Vec<WorldDelta>) {
    // Gate: only run every BASE_ARRIVAL_INTERVAL ticks
    if state.tick % BASE_ARRIVAL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Without guild_reputation and request_board on WorldState, we cannot
    // perform the Poisson check or emit quest spawn deltas.
    //
    // The original system:
    //   1. Computes a reputation-scaled arrival probability
    //   2. Rolls against it using the campaign RNG
    //   3. Generates a quest via the narrative grammar walker
    //   4. Pushes the quest request onto the board
    //
    // In the delta architecture, quest generation would need a new delta variant
    // (SpawnQuestRequest) since none of the existing WorldDelta variants cover
    // creating a new quest. The system would:
    //
    //   let rep_factor = (state.guild_reputation / REPUTATION_SCALING_CENTER)
    //       .clamp(REPUTATION_FACTOR_MIN, REPUTATION_FACTOR_MAX);
    //   let mean_interval = (BASE_ARRIVAL_INTERVAL as f32 / rep_factor) as u64;
    //   let threshold = 1.0 / mean_interval.max(1) as f32;
    //
    //   // RNG check (needs deterministic RNG from state)
    //   // if roll >= threshold { return; }
    //
    //   // Generate quest parameters from world state
    //   let threat_level = compute_threat_from_regions(state);
    //   let quest_type = pick_quest_type(state);
    //   let target_pos = pick_target_position(state);
    //   let reward_gold = threat_level * 2.0 + 10.0;
    //   let deadline_tick = state.tick + 500; // ~50 turns to accept
    //
    //   out.push(WorldDelta::SpawnQuestRequest { ... });
    //
    // Until the delta variant and state fields exist, this system is a no-op.
    // The cadence (every 50 ticks) and scaling logic are preserved for when
    // the state is extended.
}

/// Compute a base threat level from regional monster density and threat.
/// Higher regional threat -> harder quests.
fn compute_threat_from_regions(state: &WorldState) -> f32 {
    if state.regions.is_empty() {
        return 10.0;
    }
    let avg_threat: f32 = state.regions.iter().map(|r| r.threat_level).sum::<f32>()
        / state.regions.len() as f32;
    // Scale to quest threat range (5-80)
    (avg_threat * 0.8 + 5.0).clamp(5.0, 80.0)
}

/// Pick a target position from settlement positions, biased toward
/// areas with higher monster density.
fn pick_target_position(state: &WorldState) -> (f32, f32) {
    // Default: pick the first region's approximate center
    // In practice this would use RNG to select weighted by threat.
    if let Some(settlement) = state.settlements.first() {
        settlement.pos
    } else {
        (0.0, 0.0)
    }
}
