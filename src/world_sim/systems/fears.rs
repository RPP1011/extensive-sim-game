#![allow(unused)]
//! Adventurer fears and phobias system — delta architecture port.
//!
//! NPCs can develop fears from traumatic experiences that affect their
//! effectiveness. Fears have severity (0-100), can be triggered in
//! relevant situations, and can be overcome through successful exposure.
//!
//! Original: `crates/headless_campaign/src/systems/fears.rs`
//! Cadence: every 10 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: fears: Vec<Fear> on Entity/NpcData
//   Fear { fear_type: FearType, severity, acquired_tick, times_triggered, times_overcome }
//   FearType: Darkness, Heights, Water, Undead, Fire, Crowds, Isolation, Authority
// NEEDS STATE: adventurer morale, stress, injury, status on Entity/NpcData
// NEEDS DELTA: AcquireFear { entity_id, fear_type, severity }
// NEEDS DELTA: UpdateFearSeverity { entity_id, fear_type, delta }
// NEEDS DELTA: ConquerFear { entity_id, fear_type }
// NEEDS DELTA: AdjustMorale { entity_id, delta }
// NEEDS DELTA: AdjustStress { entity_id, delta }

/// Cadence gate.
const FEAR_TICK_INTERVAL: u64 = 10;

/// Fear acquisition chances.
const DARKNESS_FEAR_CHANCE: f32 = 0.30;
const MONSTER_FEAR_CHANCE: f32 = 0.20;
const CROWDS_FEAR_CHANCE: f32 = 0.15;
const AUTHORITY_FEAR_CHANCE: f32 = 0.25;
const FEAR_CONTAGION_CHANCE: f32 = 0.05;

/// Overcome threshold and severity reduction.
const OVERCOME_THRESHOLD: u32 = 3;
const OVERCOME_SEVERITY_REDUCTION: f32 = 10.0;
const MENTORSHIP_SEVERITY_REDUCTION: f32 = 5.0;

/// Compute fear deltas: acquisition, effects, overcoming, contagion.
///
/// Since WorldState lacks fear storage on entities, this is a structural
/// placeholder. Fear effects (morale and stress penalties) can be expressed
/// via ApplyStatus once fear-related status effects are defined.
pub fn compute_fears(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % FEAR_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Fear acquisition from traumatic events ---
    for entity in &state.entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }

        let hp_ratio = entity.hp / entity.max_hp.max(1.0);

        // Near-death (hp < 20%) on an active grid → Darkness fear (30%)
        if hp_ratio < 0.2 && entity.grid_id.is_some() {
            let roll = deterministic_roll(state.tick, entity.id);
            if roll < DARKNESS_FEAR_CHANCE {
                // NEEDS DELTA: AcquireFear { entity_id, Darkness, severity: 20+rand*40 }
            }
        }

        // Low hp while fighting → Undead fear (20%)
        if hp_ratio < 0.4 && entity.grid_id.is_some() {
            let roll = deterministic_roll(state.tick, entity.id.wrapping_add(1));
            if roll < MONSTER_FEAR_CHANCE {
                // NEEDS DELTA: AcquireFear { entity_id, Undead, severity: 15+rand*35 }
            }
        }

        // NEEDS STATE: check fears list for existing fears
    }

    // --- Phase 2: Fear effects (morale/stress penalties) ---
    // For each entity with fears relevant to current situation:
    //   out.push(WorldDelta::AdjustMorale { entity_id, delta: -(5 + severity/100 * 10) })
    //   out.push(WorldDelta::AdjustStress { entity_id, delta: 3 + severity/100 * 7 })
    //
    // Expressible via ApplyStatus with Debuff variant:
    //   out.push(WorldDelta::ApplyStatus {
    //       target_id: entity.id,
    //       status: StatusEffect { kind: Debuff { stat: "morale".into(), factor: 0.9 }, ... },
    //   })

    // --- Phase 3: Fear overcoming (successful exposure) ---
    // NEEDS STATE: idle entities with triggered fears and low injury
    //   out.push(WorldDelta::UpdateFearSeverity { entity_id, fear_type, delta: -OVERCOME_SEVERITY_REDUCTION })
    //   If times_overcome >= OVERCOME_THRESHOLD:
    //     out.push(WorldDelta::ConquerFear { entity_id, fear_type })

    // --- Phase 4: Mentorship (fearless entity in same grid reduces severity) ---
    for grid in &state.grids {
        let npc_ids: Vec<u32> = grid
            .entity_ids
            .iter()
            .copied()
            .filter(|&eid| {
                state
                    .entity(eid)
                    .map(|e| e.alive && e.npc.is_some())
                    .unwrap_or(false)
            })
            .collect();

        // NEEDS STATE: check if any NPC has no fears (mentor)
        // For fearful NPCs in same grid:
        //   out.push(WorldDelta::UpdateFearSeverity { entity_id, fear_type, delta: -MENTORSHIP_SEVERITY_REDUCTION })
        let _ = npc_ids;
    }

    // --- Phase 5: Fear contagion within shared grids ---
    // NEEDS STATE: for high-stress NPCs with severe fears (>50), spread to co-located NPCs
    //   Deterministic roll < FEAR_CONTAGION_CHANCE:
    //   out.push(WorldDelta::AcquireFear { entity_id, fear_type, severity: source * 0.5 })
}

fn deterministic_roll(tick: u64, id: u32) -> f32 {
    let h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(id as u64);
    let h = h ^ (h >> 33);
    let h = h.wrapping_mul(0xff51afd7ed558ccd);
    let h = h ^ (h >> 33);
    (h & 0xFFFF) as f32 / 65536.0
}
