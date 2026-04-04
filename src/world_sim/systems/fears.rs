//! Adventurer fears and phobias system — delta architecture port.
//!
//! NPCs can develop fears from traumatic experiences that affect their
//! effectiveness. Fears have severity (0-100), can be triggered in
//! relevant situations, and can be overcome through successful exposure.
//!
//! Original: `crates/headless_campaign/src/systems/fears.rs`
//! Cadence: every 10 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityField, WorldState};
use crate::world_sim::state::entity_hash_f32;

//   Fear { fear_type: FearType, severity, acquired_tick, times_triggered, times_overcome }
//   FearType: Darkness, Heights, Water, Undead, Fire, Crowds, Isolation, Authority

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
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_fears_for_settlement(state, settlement.id, &state.entities[range], out);
    }

    // --- Phase 2: Fear effects (morale penalties) ---
    // Until fear storage exists on NpcData, use entity HP ratio on active grids
    // as a proxy for active fear. Low-HP NPCs in combat suffer morale drain
    // representing fear/panic effects.
    for entity in &state.entities {
        if !entity.alive { continue; }
        let hp_ratio = entity.hp / entity.max_hp.max(1.0);
        let on_combat_grid = entity.grid_id
            .and_then(|gid| state.fidelity_zone(gid))
            .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
            .unwrap_or(false);

        if on_combat_grid && hp_ratio < 0.3 {
            // Near-death fear: significant morale penalty.
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: -3.0,
            });
        } else if on_combat_grid && hp_ratio < 0.5 {
            // Wounded and fighting: moderate fear-induced morale drain.
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: -1.0,
            });
        }
    }

    // --- Phase 3: Fear overcoming (successful exposure) ---
    //   out.push(WorldDelta::UpdateFearSeverity { entity_id, fear_type, delta: -OVERCOME_SEVERITY_REDUCTION })
    //   If times_overcome >= OVERCOME_THRESHOLD:
    //     out.push(WorldDelta::ConquerFear { entity_id, fear_type })

    // --- Phase 4: Mentorship (fearless entity in same grid reduces severity) ---
    for grid in &state.fidelity_zones {
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

        // For fearful NPCs in same grid:
        //   out.push(WorldDelta::UpdateFearSeverity { entity_id, fear_type, delta: -MENTORSHIP_SEVERITY_REDUCTION })
        let _ = npc_ids;
    }

    // --- Phase 5: Fear contagion within shared grids ---
    //   Deterministic roll < FEAR_CONTAGION_CHANCE:
    //   out.push(WorldDelta::AcquireFear { entity_id, fear_type, severity: source * 0.5 })
}

/// Per-settlement variant for parallel dispatch (fear acquisition phase).
pub fn compute_fears_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    _out: &mut Vec<WorldDelta>,
) {
    if state.tick % FEAR_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in entities {
        if !entity.alive || entity.npc.is_none() {
            continue;
        }

        let hp_ratio = entity.hp / entity.max_hp.max(1.0);

        // Near-death (hp < 20%) on an active grid → Darkness fear (30%)
        if hp_ratio < 0.2 && entity.grid_id.is_some() {
            let roll = entity_hash_f32(entity.id, state.tick, 0);
            if roll < DARKNESS_FEAR_CHANCE {
            }
        }

        // Low hp while fighting → Undead fear (20%)
        if hp_ratio < 0.4 && entity.grid_id.is_some() {
            let roll = entity_hash_f32(entity.id, state.tick, 1);
            if roll < MONSTER_FEAR_CHANCE {
            }
        }

    }
}

