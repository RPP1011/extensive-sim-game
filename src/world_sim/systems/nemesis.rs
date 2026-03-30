//! Nemesis system -- persistent enemy champions that grow stronger over time.
//!
//! Every `NEMESIS_CHECK_INTERVAL` ticks, each hostile faction with
//! `military_strength > 30` may designate an alive hostile monster as its
//! nemesis champion.  The nemesis gains stat buffs every `LEVEL_UP_INTERVAL`
//! ticks, making it progressively more dangerous.  When a nemesis dies the
//! nearest friendly NPC receives massive XP and combat behavior tags, and a
//! new nemesis can spawn after a `RESPAWN_COOLDOWN`.
//!
//! Uses only existing delta types:
//!   UpdateEntityField, AddBehaviorTags, RecordChronicle, Damage, Heal.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::{
    ActionTags, ChronicleCategory, ChronicleEntry,
    EntityField, EntityKind, WorldState, WorldTeam,
    tags,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// How often we check for nemesis spawns / deaths (ticks).
const NEMESIS_CHECK_INTERVAL: u64 = 100;

/// How often a nemesis gains a level (ticks).
const LEVEL_UP_INTERVAL: u64 = 500;

/// Maximum nemesis level — prevents absurd growth.
const MAX_NEMESIS_LEVEL: u32 = 50;

/// Ticks after a nemesis death before a new one can spawn.
const RESPAWN_COOLDOWN: u64 = 500;

/// Minimum faction military_strength to spawn a nemesis.
const MIN_MILITARY_STRENGTH: f32 = 30.0;

/// Per-level stat buffs applied to the nemesis each level-up.
const NEMESIS_HP_PER_LEVEL: f32 = 30.0;
const NEMESIS_ATK_PER_LEVEL: f32 = 5.0;
const NEMESIS_ARMOR_PER_LEVEL: f32 = 2.0;

/// Initial stat boost when a monster is first designated as nemesis.
const INITIAL_HP_BUFF: f32 = 100.0;
const INITIAL_ATK_BUFF: f32 = 15.0;
const INITIAL_ARMOR_BUFF: f32 = 5.0;

/// XP reward for the NPC that slays a nemesis.
const NEMESIS_KILL_XP: u32 = 50;

// ---------------------------------------------------------------------------
// Deterministic RNG
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Main compute function
// ---------------------------------------------------------------------------

pub fn compute_nemesis(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Level-up runs on its own cadence independent of the spawn check.
    if state.tick > 0 && state.tick % LEVEL_UP_INTERVAL == 0 {
        nemesis_level_up(state, out);
    }

    if state.tick % NEMESIS_CHECK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Phase 1: detect dead nemeses and reward killers.
    check_nemesis_deaths(state, out);

    // Phase 2: spawn new nemeses for factions that qualify.
    spawn_nemeses(state, out);
}

// ---------------------------------------------------------------------------
// Phase 1 — detect nemesis deaths
// ---------------------------------------------------------------------------

fn check_nemesis_deaths(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // A nemesis is identified by being a high-level (>= 5) hostile monster.
    // When such a monster is found dead, the nearest alive friendly NPC gets
    // the kill reward.
    for entity in &state.entities {
        if entity.kind != EntityKind::Monster { continue; }
        if entity.team != WorldTeam::Hostile { continue; }
        if entity.alive { continue; }
        // Only count high-level monsters as nemeses (level >= 5 indicates
        // it was previously buffed by this system).
        if entity.level < 5 { continue; }

        // Find the nearest alive friendly NPC to credit with the kill.
        let mut best_id: Option<u32> = None;
        let mut best_dist_sq = f32::MAX;
        for npc in &state.entities {
            if npc.kind != EntityKind::Npc || !npc.alive { continue; }
            if npc.team != WorldTeam::Friendly { continue; }
            let dx = npc.pos.0 - entity.pos.0;
            let dy = npc.pos.1 - entity.pos.1;
            let d2 = dx * dx + dy * dy;
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best_id = Some(npc.id);
            }
        }

        if let Some(killer_id) = best_id {
            // Only credit if reasonably close (within 100 units).
            if best_dist_sq > 10000.0 { continue; }

            // Combat behavior tags for the slayer.
            let mut action = ActionTags::empty();
            action.add(tags::COMBAT, 5.0);
            action.add(tags::RESILIENCE, 3.0);
            action.add(tags::LEADERSHIP, 2.0);
            if let Some(killer) = state.entity(killer_id) {
                let action = crate::world_sim::action_context::with_context(&action, killer, state);
                out.push(WorldDelta::AddBehaviorTags {
                    entity_id: killer_id,
                    tags: action.tags,
                    count: action.count,
                });
            }

            // Morale boost for the slayer.
            out.push(WorldDelta::UpdateEntityField {
                entity_id: killer_id,
                field: EntityField::Morale,
                value: 15.0,
            });

            // Chronicle the kill.
            let killer_name = state.entity(killer_id)
                .map(|e| entity_display_name(e))
                .unwrap_or_else(|| format!("Entity #{}", killer_id));
            let nemesis_name = entity_display_name(entity);
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Battle,
                    text: format!(
                        "{} slew nemesis {} (level {}) near ({:.0}, {:.0})",
                        killer_name, nemesis_name, entity.level, entity.pos.0, entity.pos.1,
                    ),
                    entity_ids: vec![killer_id, entity.id],
                },
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2 — spawn new nemeses
// ---------------------------------------------------------------------------

fn spawn_nemeses(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for faction in &state.factions {
        // Only hostile factions with enough military strength.
        if faction.relationship_to_guild >= 0.0 { continue; }
        if faction.military_strength < MIN_MILITARY_STRENGTH { continue; }

        // Cooldown: don't spawn if faction recently lost a nemesis.
        // Use a deterministic roll seeded by faction id + tick to approximate
        // cooldown without storing last-death-tick.  The 500-tick cooldown is
        // modeled by requiring tick to be a multiple of RESPAWN_COOLDOWN
        // *relative to the faction hash*, giving each faction its own phase.
        let phase = (faction.id as u64 * 997) % RESPAWN_COOLDOWN;
        if (state.tick + phase) % RESPAWN_COOLDOWN != 0 { continue; }

        // Check if this faction already has a living nemesis (level >= 5 hostile monster).
        let already_has = state.entities.iter().any(|e| {
            e.kind == EntityKind::Monster
                && e.team == WorldTeam::Hostile
                && e.alive
                && e.level >= 5
        });
        if already_has { continue; }

        // Find a suitable living hostile monster to promote.
        // Prefer higher-level monsters; use deterministic tie-breaking.
        let mut best: Option<(u32, u32)> = None; // (entity_id, level)
        for entity in &state.entities {
            if entity.kind != EntityKind::Monster { continue; }
            if entity.team != WorldTeam::Hostile { continue; }
            if !entity.alive { continue; }
            if entity.level >= 5 { continue; } // already a nemesis

            let score = entity.level;
            match best {
                None => best = Some((entity.id, score)),
                Some((_, prev)) if score > prev => best = Some((entity.id, score)),
                Some((prev_id, prev_score)) if score == prev_score && entity.id < prev_id => {
                    best = Some((entity.id, score));
                }
                _ => {}
            }
        }

        if let Some((monster_id, _)) = best {
            // Buff the monster to nemesis status.
            out.push(WorldDelta::UpdateEntityField {
                entity_id: monster_id,
                field: EntityField::MaxHp,
                value: INITIAL_HP_BUFF,
            });
            out.push(WorldDelta::Heal {
                target_id: monster_id,
                amount: INITIAL_HP_BUFF,
                source_id: 0,
            });
            out.push(WorldDelta::UpdateEntityField {
                entity_id: monster_id,
                field: EntityField::AttackDamage,
                value: INITIAL_ATK_BUFF,
            });
            out.push(WorldDelta::UpdateEntityField {
                entity_id: monster_id,
                field: EntityField::Armor,
                value: INITIAL_ARMOR_BUFF,
            });
            // Set level to 5 to mark it as a nemesis (5 level increments).
            let levels_needed = 5u32.saturating_sub(
                state.entity(monster_id).map(|e| e.level).unwrap_or(0)
            );
            if levels_needed > 0 {
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: monster_id,
                    field: EntityField::Level,
                    value: levels_needed as f32,
                });
            }

            // Chronicle the emergence.
            let monster_name = state.entity(monster_id)
                .map(|e| entity_display_name(e))
                .unwrap_or_else(|| format!("Entity #{}", monster_id));
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Battle,
                    text: format!(
                        "A nemesis champion ({}) has emerged from faction '{}'",
                        monster_name, faction.name,
                    ),
                    entity_ids: vec![monster_id],
                },
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Level-up — periodic stat growth for living nemeses
// ---------------------------------------------------------------------------

fn nemesis_level_up(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for entity in &state.entities {
        if entity.kind != EntityKind::Monster { continue; }
        if entity.team != WorldTeam::Hostile { continue; }
        if !entity.alive { continue; }
        if entity.level < 5 { continue; } // not a nemesis
        if entity.level >= MAX_NEMESIS_LEVEL { continue; } // cap reached

        // Buff stats.
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::MaxHp,
            value: NEMESIS_HP_PER_LEVEL,
        });
        out.push(WorldDelta::Heal {
            target_id: entity.id,
            amount: NEMESIS_HP_PER_LEVEL,
            source_id: 0,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::AttackDamage,
            value: NEMESIS_ATK_PER_LEVEL,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Armor,
            value: NEMESIS_ARMOR_PER_LEVEL,
        });
        out.push(WorldDelta::UpdateEntityField {
            entity_id: entity.id,
            field: EntityField::Level,
            value: 1.0,
        });

        // Chronicle the growth.
        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Battle,
                text: format!(
                    "Nemesis {} grows stronger — now level {}",
                    entity_display_name(entity), entity.level + 1,
                ),
                entity_ids: vec![entity.id],
            },
        });
    }
}
