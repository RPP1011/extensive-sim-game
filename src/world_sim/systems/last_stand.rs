#![allow(unused)]
//! Heroic last stand system — checked during battle resolution.
//!
//! When a friendly entity is near death on a grid with hostiles, it may
//! trigger a last stand: a burst of damage against nearby enemies and
//! a small self-heal representing the dramatic turnaround.
//!
//! Ported from `crates/headless_campaign/src/systems/last_stand.rs`.
//!
//! NEEDS STATE: `traits: Vec<String>` on NpcData (for "The Undying" check)
//! NEEDS STATE: `loyalty: f32` on NpcData (loyalty > 80 triggers last stand)
//! NEEDS STATE: `morale: f32` on NpcData (boosted on heroic victory)
//! NEEDS STATE: `history_tags: HashMap<String, u32>` on NpcData (near_death count)
//! NEEDS STATE: `bonds: Vec<(u32, u32, f32)>` on WorldState (bond strength lookup)
//! NEEDS DELTA: UpdateMorale { entity_id, delta } (for morale boost on victory)
//! NEEDS DELTA: UpdateReputation { entity_id, delta } (for guild reputation boost)

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{ActionTags, EntityKind, WorldState, WorldTeam, tags};

/// HP ratio threshold below which a last stand can trigger.
const LAST_STAND_HP_THRESHOLD: f32 = 0.15;

/// Damage multiplier for the last stand burst (based on entity's attack damage).
const LAST_STAND_DAMAGE_MULT: f32 = 3.0;

/// Self-heal amount during last stand (fraction of max HP).
const LAST_STAND_HEAL_FRACTION: f32 = 0.10;

/// Level-based bonus chance per level (additive to base 20% victory chance).
const LEVEL_VICTORY_BONUS: f32 = 0.005;

pub fn compute_last_stand(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_last_stand_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_last_stand_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    _entities: &[crate::world_sim::state::Entity],
    out: &mut Vec<WorldDelta>,
) {
    let grid_id = match state.settlement(settlement_id).and_then(|s| s.grid_id) {
        Some(gid) => gid,
        None => return,
    };
    let grid = match state.grid(grid_id) {
        Some(g) => g,
        None => return,
    };

    if grid.fidelity != crate::world_sim::fidelity::Fidelity::High { return; }

    let mut hostile_ids = [0u32; 64];
    let mut hc = 0usize;
    let mut hero_ids = [0u32; 64];
    let mut hero_atk = [0.0f32; 64];
    let mut hero_maxhp = [0.0f32; 64];
    let mut hrc = 0usize;

    for &eid in &grid.entity_ids {
        if let Some(e) = state.entity(eid) {
            if !e.alive || e.hp <= 0.0 { continue; }
            if e.kind == EntityKind::Monster && hc < 64 {
                hostile_ids[hc] = eid; hc += 1;
            } else if e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly
                && e.max_hp > 0.0 && (e.hp / e.max_hp) <= LAST_STAND_HP_THRESHOLD
                && hrc < 64
            {
                hero_ids[hrc] = eid;
                hero_atk[hrc] = e.attack_damage;
                hero_maxhp[hrc] = e.max_hp;
                hrc += 1;
            }
        }
    }

    if hc == 0 || hrc == 0 { return; }

    for hi in 0..hrc {
        let hero_id = hero_ids[hi];
        let burst_damage = hero_atk[hi] * LAST_STAND_DAMAGE_MULT;
        let dmg_per_hostile = burst_damage / hc as f32;

        for i in 0..hc {
            if dmg_per_hostile > 0.0 {
                out.push(WorldDelta::Damage {
                    target_id: hostile_ids[i],
                    amount: dmg_per_hostile,
                    source_id: hero_id,
                });
            }
        }

        let heal_amount = hero_maxhp[hi] * LAST_STAND_HEAL_FRACTION;
        if heal_amount > 0.0 {
            out.push(WorldDelta::Heal {
                target_id: hero_id,
                amount: heal_amount,
                source_id: hero_id,
            });
        }

        let shield_amount = hero_maxhp[hi] * 0.05;
        if shield_amount > 0.0 {
            out.push(WorldDelta::Shield {
                target_id: hero_id,
                amount: shield_amount,
                source_id: hero_id,
            });
        }

        // Behavior tags: desperate combat earns high resilience/combat.
        let mut action = ActionTags::empty();
        action.add(tags::RESILIENCE, 3.0);
        action.add(tags::COMBAT, 2.0);
        if let Some(entity) = state.entity(hero_id) {
            let action = crate::world_sim::action_context::with_context(&action, entity, state);
            out.push(WorldDelta::AddBehaviorTags { entity_id: hero_id, tags: action.tags, count: action.count });
        }
    }
}
