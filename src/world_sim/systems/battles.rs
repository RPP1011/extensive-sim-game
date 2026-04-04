//! Battle resolution — every tick.
//!
//! Advances in-progress battles. Interpolates health ratios toward the predicted
//! outcome over the configured duration. When a battle resolves, emits Damage/Die
//! deltas for casualties.
//!
//! Ported from `crates/headless_campaign/src/systems/battles.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::{ActionTags, ChronicleCategory, ChronicleEntry, WorldState, tags};

//              predicted_outcome, party_health_ratio, enemy_health_ratio, status,
//              participant_ids: Vec<u32>, enemy_ids: Vec<u32> }

/// Default battle duration in ticks if config is unavailable.
const DEFAULT_DURATION_TICKS: u32 = 20;

/// Fraction of max HP dealt as damage to party on victory.
const VICTORY_PARTY_DAMAGE: f32 = 0.3;

/// Fraction of max HP dealt as damage to enemies on defeat.
const DEFEAT_ENEMY_DAMAGE: f32 = 0.7;

pub fn compute_battles(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_battles_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
///
/// Finds the grid associated with this settlement and processes combat on it.
pub fn compute_battles_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    let grid_id = match state.settlement(settlement_id).and_then(|s| s.grid_id) {
        Some(gid) => gid,
        None => return,
    };
    let grid = match state.fidelity_zone(grid_id) {
        Some(g) => g,
        None => return,
    };
    compute_battles_for_grid(state, grid, out);
}

fn compute_battles_for_grid(
    state: &WorldState,
    grid: &crate::world_sim::state::FidelityZone,
    out: &mut Vec<WorldDelta>,
) {
    // Only process High-fidelity grids (combat already happening).
    if grid.fidelity != crate::world_sim::fidelity::Fidelity::High {
        return;
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

    if fc == 0 || hc == 0 { return; }

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

                // Chronicle: monster kills a high-level NPC (level >= 20).
                let friendly_hp = state.entity(fid).map(|e| e.hp).unwrap_or(0.0);
                let friendly_level = state.entity(fid).map(|e| e.level).unwrap_or(0);
                if friendly_hp <= dmg_each && friendly_level >= 20 {
                    let npc_name = state.entity(fid).map(|e| entity_display_name(e)).unwrap_or_default();
                    let monster_name = state.entity(hid).map(|e| entity_display_name(e)).unwrap_or_default();
                    let pos = state.entity(fid).map(|e| e.pos).unwrap_or((0.0, 0.0));
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Battle,
                            text: format!("{} was slain by {} near ({:.0}, {:.0})", npc_name, monster_name, pos.0, pos.1),
                            entity_ids: vec![fid, hid],
                        },
                    });
                }

                // Behavior tags: taking damage builds defense/endurance.
                let mut action = ActionTags::empty();
                action.add(tags::DEFENSE, 0.5);
                action.add(tags::ENDURANCE, 0.3);
                if let Some(entity) = state.entity(fid) {
                    let action = crate::world_sim::action_context::with_context(&action, entity, state);
                    out.push(WorldDelta::AddBehaviorTags { entity_id: fid, tags: action.tags, count: action.count });
                }
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

                let hostile_hp = state.entity(hid).map(|e| e.hp).unwrap_or(0.0);
                if hostile_hp - dmg_each <= 0.0 {
                    // Behavior tags: kill grants heavy combat tags.
                    let mut kill_action = ActionTags::empty();
                    kill_action.add(tags::COMBAT, 2.0);
                    out.push(WorldDelta::AddBehaviorTags { entity_id: fid, tags: kill_action.tags, count: kill_action.count });

                    // Chronicle: NPC slew a named monster.
                    let npc_name = state.entity(fid).map(|e| entity_display_name(e)).unwrap_or_default();
                    let monster_name = state.entity(hid).map(|e| entity_display_name(e)).unwrap_or_default();
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Battle,
                            text: format!("{} slew {}", npc_name, monster_name),
                            entity_ids: vec![fid, hid],
                        },
                    });
                }
            }

            // Behavior tags: attacking builds melee/combat.
            let mut action = ActionTags::empty();
            action.add(tags::MELEE, 1.0);
            action.add(tags::COMBAT, 0.5);
            if let Some(entity) = state.entity(fid) {
                let action = crate::world_sim::action_context::with_context(&action, entity, state);
                out.push(WorldDelta::AddBehaviorTags { entity_id: fid, tags: action.tags, count: action.count });
            }

        }
    }
}
