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
        // Only process High-fidelity grids (combat already happening).
        if grid.fidelity != crate::world_sim::fidelity::Fidelity::High {
            continue;
        }

        // Count friendlies/hostiles without allocating.
        let mut friendly_ids = [0u32; 64];
        let mut hostile_ids = [0u32; 64];
        let mut fc = 0usize;
        let mut hc = 0usize;

        for &eid in &grid.entity_ids {
            if let Some(e) = state.entity(eid) {
                if !e.alive { continue; }
                match e.team {
                    crate::world_sim::state::WorldTeam::Friendly => {
                        if fc < 64 { friendly_ids[fc] = eid; fc += 1; }
                    }
                    crate::world_sim::state::WorldTeam::Hostile => {
                        if hc < 64 { hostile_ids[hc] = eid; hc += 1; }
                    }
                    _ => {}
                }
            }
        }

        if fc == 0 || hc == 0 { continue; }

        let friendlies = &friendly_ids[..fc];
        let hostiles = &hostile_ids[..hc];

        // Hostiles attack friendlies
        let fcount = fc as f32;
        let hcount = hc as f32;
        for &hid in hostiles {
            let atk = state.entity(hid).map(|e| e.attack_damage).unwrap_or(0.0);
            let dmg_each = atk / fcount;
            if dmg_each > 0.0 {
                for &fid in friendlies {
                    out.push(WorldDelta::Damage { target_id: fid, amount: dmg_each, source_id: hid });
                }
            }
        }
        // Friendlies attack hostiles
        for &fid in friendlies {
            let atk = state.entity(fid).map(|e| e.attack_damage).unwrap_or(0.0);
            let dmg_each = atk / hcount;
            if dmg_each > 0.0 {
                for &hid in hostiles {
                    out.push(WorldDelta::Damage { target_id: hid, amount: dmg_each, source_id: fid });
                }
            }
        }
    }
}
