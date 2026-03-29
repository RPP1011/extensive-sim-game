#![allow(unused)]
//! Battle resolution — every tick.
//!
//! Advances in-progress battles. Interpolates health ratios toward the predicted
//! outcome over the configured duration. When a battle resolves, emits Damage/Die
//! deltas for casualties.
//!
//! Ported from `crates/headless_campaign/src/systems/battles.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: active_battles: Vec<BattleState> on WorldState
// NEEDS STATE: BattleState { id, quest_id, party_id, enemy_strength, elapsed_ticks,
//              predicted_outcome, party_health_ratio, enemy_health_ratio, status,
//              participant_ids: Vec<u32>, enemy_ids: Vec<u32> }
// NEEDS STATE: BattleStatus enum { Active, Victory, Defeat, Retreat }
// NEEDS STATE: config.battle.default_duration_ticks on WorldState
// NEEDS STATE: config.battle.update_interval_ticks on WorldState
// NEEDS STATE: config.battle.victory_party_damage on WorldState
// NEEDS STATE: config.battle.defeat_enemy_damage on WorldState
// NEEDS DELTA: UpdateBattle { battle_id: u32, elapsed_ticks: u32, party_health_ratio: f32, enemy_health_ratio: f32, status: u8 }

/// Default battle duration in ticks if config is unavailable.
const DEFAULT_DURATION_TICKS: u32 = 20;

/// Fraction of max HP dealt as damage to party on victory.
const VICTORY_PARTY_DAMAGE: f32 = 0.3;

/// Fraction of max HP dealt as damage to enemies on defeat.
const DEFEAT_ENEMY_DAMAGE: f32 = 0.7;

pub fn compute_battles(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // For each entity pair on the same grid with opposing teams, resolve combat
    // by emitting Damage/Die deltas.
    //
    // In the full migration this would iterate `state.active_battles`, but since
    // that field does not yet exist on WorldState we operate on grid-level proximity:
    // any hostile+friendly pair sharing a grid exchanges damage.

    for grid in &state.grids {
        // Collect alive friendlies and hostiles on this grid.
        let mut friendlies: Vec<&crate::world_sim::state::Entity> = Vec::new();
        let mut hostiles: Vec<&crate::world_sim::state::Entity> = Vec::new();

        for &eid in &grid.entity_ids {
            if let Some(e) = state.entity(eid) {
                if !e.alive {
                    continue;
                }
                match e.team {
                    crate::world_sim::state::WorldTeam::Friendly => friendlies.push(e),
                    crate::world_sim::state::WorldTeam::Hostile => hostiles.push(e),
                    crate::world_sim::state::WorldTeam::Neutral => {}
                }
            }
        }

        if friendlies.is_empty() || hostiles.is_empty() {
            continue;
        }

        // --- Simplified oracle-style resolution per tick ---
        // Each hostile deals its attack_damage spread across friendlies.
        // Each friendly deals its attack_damage spread across hostiles.

        // Hostiles attack friendlies
        let friendly_count = friendlies.len() as f32;
        for hostile in &hostiles {
            let dmg_each = hostile.attack_damage / friendly_count;
            for friendly in &friendlies {
                if dmg_each > 0.0 {
                    out.push(WorldDelta::Damage {
                        target_id: friendly.id,
                        amount: dmg_each,
                        source_id: hostile.id,
                    });
                }
            }
        }

        // Friendlies attack hostiles
        let hostile_count = hostiles.len() as f32;
        for friendly in &friendlies {
            let dmg_each = friendly.attack_damage / hostile_count;
            for hostile in &hostiles {
                if dmg_each > 0.0 {
                    out.push(WorldDelta::Damage {
                        target_id: hostile.id,
                        amount: dmg_each,
                        source_id: friendly.id,
                    });
                }
            }
        }

        // Emit Die for any entity whose hp would drop to zero after accumulated
        // damage this tick. We check current HP against expected damage.
        // (The apply phase handles clamping, but we emit Die proactively so that
        //  other systems in the same tick can see it.)
        for friendly in &friendlies {
            let incoming: f32 = hostiles.iter().map(|h| h.attack_damage / friendly_count).sum();
            if friendly.hp - incoming <= 0.0 {
                out.push(WorldDelta::Die { entity_id: friendly.id });
            }
        }
        for hostile in &hostiles {
            let incoming: f32 = friendlies.iter().map(|f| f.attack_damage / hostile_count).sum();
            if hostile.hp - incoming <= 0.0 {
                out.push(WorldDelta::Die { entity_id: hostile.id });
            }
        }
    }
}
