#![allow(unused)]
//! Mentor-apprentice training — ported from headless_campaign.
//!
//! Mentors (level >= apprentice + 3) at the same settlement train apprentices.
//! Each mentor can train at most 2 apprentices. Cadence is jittered per
//! settlement to spread load.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{EntityKind, WorldState, EntityField};

const BASE_CADENCE: u64 = 3;
const BASE_MENTOR_XP: f32 = 5.0;
const BASE_STAT_GAIN: f32 = 0.1;
const LEVEL_HP_GAIN: f32 = 10.0;
const LEVEL_ATTACK_GAIN: f32 = 1.0;

pub fn compute_mentorship(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Pre-bucket: for each settlement, collect NPC indices sorted by level.
    // Jitter cadence per settlement so they don't all fire on the same tick.
    for settlement in &state.settlements {
        let jitter = settlement.id as u64 % BASE_CADENCE;
        if state.tick % BASE_CADENCE != jitter { continue; }

        // Collect eligible NPC (id, level) pairs at this settlement via group index.
        let mut npcs = [(0u32, 0u32); 128];
        let mut count = 0usize;

        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if entity.npc.is_none() { continue; }

            // Skip NPCs in combat.
            if let Some(gid) = entity.grid_id {
                if let Some(g) = state.grid(gid) {
                    if g.fidelity == crate::world_sim::fidelity::Fidelity::High {
                        continue;
                    }
                }
            }

            if count < 128 {
                npcs[count] = (entity.id, entity.level);
                count += 1;
            }
        }

        if count < 2 { continue; }

        // Sort by level descending (insertion sort — small N, no alloc).
        let slice = &mut npcs[..count];
        for i in 1..slice.len() {
            let mut j = i;
            while j > 0 && slice[j].1 > slice[j - 1].1 {
                slice.swap(j, j - 1);
                j -= 1;
            }
        }

        // Pair mentors with apprentices. Each mentor: max 2.
        let mut mentor_used = [0u8; 128]; // slots used per index
        let mut apprenticed = [false; 128];

        for i in 0..count {
            if mentor_used[i] >= 2 { continue; }
            let (mentor_id, mentor_level) = slice[i];

            for j in (i + 1)..count {
                if apprenticed[j] { continue; }
                let (apprentice_id, apprentice_level) = slice[j];
                if mentor_level < apprentice_level + 3 { continue; }

                apprenticed[j] = true;
                mentor_used[i] += 1;

                // Emit training deltas.
                let xp_gain = BASE_MENTOR_XP * (1.0 + mentor_level as f32 / 20.0);

                out.push(WorldDelta::Heal {
                    target_id: apprentice_id,
                    amount: xp_gain * 0.05,
                    source_id: mentor_id,
                });

                // Stat boost via entity field delta.
                let stat_gain = BASE_STAT_GAIN * (1.0 + (mentor_level as f32 - apprentice_level as f32) * 0.1);
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: apprentice_id,
                    field: EntityField::AttackDamage,
                    value: stat_gain,
                });

                // XP
                out.push(WorldDelta::AddXp {
                    entity_id: apprentice_id,
                    amount: xp_gain as u32,
                });

                // Level-up check (staggered).
                let level_up_cadence = 200 * apprentice_level.max(1) as u64;
                let stagger = (mentor_id as u64 + apprentice_id as u64) % level_up_cadence;
                let effective_tick = state.tick / BASE_CADENCE;
                if effective_tick > 0 && (effective_tick + stagger) % level_up_cadence == 0 {
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: apprentice_id,
                        field: EntityField::Hp,
                        value: LEVEL_HP_GAIN,
                    });
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: apprentice_id,
                        field: EntityField::AttackDamage,
                        value: LEVEL_ATTACK_GAIN,
                    });
                }

                if mentor_used[i] >= 2 { break; }
            }
        }
    }
}
