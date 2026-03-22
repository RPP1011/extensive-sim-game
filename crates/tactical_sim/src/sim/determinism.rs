use super::types::*;
use super::events::SimEvent;
use super::helpers::to_x100;
use super::simulation::step;
use super::replay::run_replay;

/// Result of a determinism check — compares two replays of the same input.
#[derive(Debug, Clone)]
pub struct DeterminismReport {
    /// True if both runs produced identical results.
    pub is_deterministic: bool,
    /// If divergence occurred, the first tick index where hashes differ.
    pub first_divergent_tick: Option<usize>,
    /// Hash from run A at the divergent tick (if any).
    pub hash_a: Option<u64>,
    /// Hash from run B at the divergent tick (if any).
    pub hash_b: Option<u64>,
}

/// Run the same replay twice and compare per-tick state hashes.
/// Returns a report indicating whether the simulation is fully deterministic
/// for the given initial state and intent script.
pub fn verify_determinism(
    initial_state: &SimState,
    scripted_intents: &[Vec<UnitIntent>],
    ticks: u32,
    dt_ms: u32,
) -> DeterminismReport {
    let result_a = run_replay(initial_state.clone(), scripted_intents, ticks, dt_ms);
    let result_b = run_replay(initial_state.clone(), scripted_intents, ticks, dt_ms);

    // Compare per-tick hashes
    for (i, (ha, hb)) in result_a
        .per_tick_state_hashes
        .iter()
        .zip(result_b.per_tick_state_hashes.iter())
        .enumerate()
    {
        if ha != hb {
            return DeterminismReport {
                is_deterministic: false,
                first_divergent_tick: Some(i),
                hash_a: Some(*ha),
                hash_b: Some(*hb),
            };
        }
    }

    // Also check final aggregates
    if result_a.event_log_hash != result_b.event_log_hash
        || result_a.final_state_hash != result_b.final_state_hash
    {
        return DeterminismReport {
            is_deterministic: false,
            first_divergent_tick: None,
            hash_a: Some(result_a.final_state_hash),
            hash_b: Some(result_b.final_state_hash),
        };
    }

    DeterminismReport {
        is_deterministic: true,
        first_divergent_tick: None,
        hash_a: None,
        hash_b: None,
    }
}

/// Run a replay against a known set of expected per-tick hashes.
/// Returns the first tick where the actual hash diverges from the expected.
pub fn verify_replay_against_hashes(
    initial_state: &SimState,
    scripted_intents: &[Vec<UnitIntent>],
    expected_hashes: &[u64],
    dt_ms: u32,
) -> Option<(usize, u64, u64)> {
    let mut state = initial_state.clone();
    for (tick, expected) in expected_hashes.iter().enumerate() {
        let intents = scripted_intents
            .get(tick)
            .map_or(&[][..], |v| v.as_slice());
        let (new_state, _) = step(state, intents, dt_ms);
        state = new_state;
        let actual = hash_sim_state(&state);
        if actual != *expected {
            return Some((tick, *expected, actual));
        }
    }
    None
}

pub fn hash_event_log(events: &[SimEvent]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0001_0000_01b3;
    let mut hash = FNV_OFFSET;
    for event in events {
        let line = format!("{event:?}");
        for byte in line.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

pub fn hash_sim_state(state: &SimState) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0001_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut entries = state.units.iter()
        .map(|u| {
            format!(
                "id={} team={:?} hp={} max_hp={} pos=({}, {}) cd={} acd={} hcd={} ccd={} ctrl={} cast={:?}",
                u.id, u.team, u.hp, u.max_hp,
                to_x100(u.position.x), to_x100(u.position.y),
                u.cooldown_remaining_ms, u.ability_cooldown_remaining_ms,
                u.heal_cooldown_remaining_ms, u.control_cooldown_remaining_ms,
                u.control_remaining_ms, u.casting
            )
        })
        .collect::<Vec<_>>();
    entries.sort();

    let header = format!("tick={} rng={}", state.tick, state.rng_state);
    for byte in header.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for line in entries {
        for byte in line.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}
