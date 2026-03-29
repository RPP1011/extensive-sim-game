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
use crate::world_sim::state::{EntityKind, WorldState, WorldTeam};

/// HP ratio threshold below which a last stand can trigger.
const LAST_STAND_HP_THRESHOLD: f32 = 0.15;

/// Damage multiplier for the last stand burst (based on entity's attack damage).
const LAST_STAND_DAMAGE_MULT: f32 = 3.0;

/// Self-heal amount during last stand (fraction of max HP).
const LAST_STAND_HEAL_FRACTION: f32 = 0.10;

/// Level-based bonus chance per level (additive to base 20% victory chance).
const LEVEL_VICTORY_BONUS: f32 = 0.005;

pub fn compute_last_stand(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Last stand triggers are checked every tick during combat.
    // We look for friendly NPCs that are near death on grids with hostiles.

    for grid in &state.grids {
        if grid.fidelity != crate::world_sim::fidelity::Fidelity::High { continue; }

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

        if hc == 0 || hrc == 0 { continue; }

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
        }
    }
}
