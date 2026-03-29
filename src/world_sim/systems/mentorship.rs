#![allow(unused)]
//! Mentor-apprentice training — ported from headless_campaign.
//!
//! Every `MENTORSHIP_CADENCE` ticks, mentor NPCs train their apprentices.
//! Both must be alive and not on hostile grids. The apprentice gains stats
//! biased by the mentor's archetype, and the bond between them strengthens.
//! Mentorship ends after `MAX_MENTORSHIP_TICKS` or when the apprentice
//! approaches the mentor's level.
//!
//! Original: `crates/headless_campaign/src/systems/mentorship.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

// NEEDS STATE: mentor_assignments: Vec<MentorAssignment> on WorldState
//   where MentorAssignment { mentor_id: u32, apprentice_id: u32, started_tick: u64, xp_transferred: f32 }
// NEEDS STATE: xp: u32 on NpcData
// NEEDS STATE: archetype/class_tags on NpcData (already exists as class_tags)
// NEEDS DELTA: GrantXp { entity_id: u32, amount: u32, source: String }
// NEEDS DELTA: TransferSkill { from_id: u32, to_id: u32, tag: String }
// NEEDS DELTA: CompleteMentorship { mentor_id: u32, apprentice_id: u32 }
// NEEDS DELTA: UpdateBond { entity_a: u32, entity_b: u32, delta: f32 }

/// How often mentorship ticks (in ticks).
const MENTORSHIP_CADENCE: u64 = 3;

/// Base XP per mentorship tick (scaled by mentor level).
const BASE_MENTOR_XP: f32 = 5.0;

/// Maximum mentorship duration in ticks.
const MAX_MENTORSHIP_TICKS: u64 = 67;

/// Stat gain per mentorship tick (base, before archetype bias).
const BASE_STAT_GAIN: f32 = 0.1;

/// Level-up XP multiplier: threshold = level^2 * this.
const XP_LEVEL_MULT: u32 = 100;

/// HP gain per level-up from mentorship.
const LEVEL_HP_GAIN: f32 = 10.0;
/// Attack gain per level-up from mentorship.
const LEVEL_ATTACK_GAIN: f32 = 1.0;
/// Defense gain per level-up from mentorship.
const LEVEL_DEFENSE_GAIN: f32 = 0.5;

/// Compute mentorship deltas for all NPC pairs.
///
/// Since the delta architecture doesn't have a mentor_assignments list yet,
/// this system identifies mentor-apprentice pairs by convention:
///   - Two NPCs at the same settlement (same home_settlement_id)
///   - Neither on a hostile grid
///   - One has level >= other's level + 3
///   - Both alive
///
/// In practice, mentor_assignments should be tracked in WorldState.
/// This implementation shows the delta-producing logic assuming those
/// assignments exist, using entity pairs as a fallback.
///
/// Mentorship effects are expressed as:
///   - `Heal` on apprentice (small HP boost representing training gains)
///   - `ApplyStatus` Buff on apprentice (archetype-biased stat growth)
///   - Level-up via stat buffs when threshold crossed
pub fn compute_mentorship(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % MENTORSHIP_CADENCE != 0 {
        return;
    }

    // Collect all alive, non-fighting NPCs grouped by home settlement.
    let mut npcs_by_settlement: std::collections::HashMap<u32, Vec<&Entity>> =
        std::collections::HashMap::new();

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Skip NPCs on hostile grids (in combat).
        let on_hostile_grid = entity
            .grid_id
            .and_then(|gid| state.grid(gid))
            .map(|g| g.has_hostiles(state))
            .unwrap_or(false);
        if on_hostile_grid {
            continue;
        }

        if let Some(sid) = npc.home_settlement_id {
            npcs_by_settlement.entry(sid).or_default().push(entity);
        }
    }

    // For each settlement, find mentor-apprentice pairs.
    for (_settlement_id, npcs) in &npcs_by_settlement {
        if npcs.len() < 2 {
            continue;
        }

        // Sort by level descending to pair highest with lowest.
        let mut sorted: Vec<&&Entity> = npcs.iter().collect();
        sorted.sort_by(|a, b| b.level.cmp(&a.level));

        // Pair mentors (high level) with apprentices (low level).
        // Each mentor can train at most 2 apprentices.
        let mut mentor_slots: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        let mut apprenticed: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for i in 0..sorted.len() {
            let mentor = sorted[i];
            let mentor_used = mentor_slots.entry(mentor.id).or_insert(0);
            if *mentor_used >= 2 {
                continue;
            }

            for j in (i + 1)..sorted.len() {
                let apprentice = sorted[j];
                if apprenticed.contains(&apprentice.id) {
                    continue;
                }
                // Mentor must be at least 3 levels higher.
                if mentor.level < apprentice.level + 3 {
                    continue;
                }

                // This pair can train.
                apprenticed.insert(apprentice.id);
                *mentor_slots.entry(mentor.id).or_insert(0) += 1;

                emit_mentorship_deltas(mentor, apprentice, state, out);

                if *mentor_slots.get(&mentor.id).unwrap_or(&0) >= 2 {
                    break;
                }
            }
        }
    }
}

/// Emit deltas for a single mentor-apprentice training tick.
fn emit_mentorship_deltas(
    mentor: &Entity,
    apprentice: &Entity,
    state: &WorldState,
    out: &mut Vec<WorldDelta>,
) {
    let mentor_npc = match &mentor.npc {
        Some(n) => n,
        None => return,
    };

    // XP gain scaled by mentor level.
    let xp_gain = BASE_MENTOR_XP * (1.0 + mentor.level as f32 / 20.0);

    // Small HP boost (representing overall growth from training).
    out.push(WorldDelta::Heal {
        target_id: apprentice.id,
        amount: xp_gain * 0.05, // tiny heal as growth proxy
        source_id: mentor.id,
    });

    // Archetype-biased stat growth via buffs.
    let hash = tick_pair_hash(state.tick, mentor.id, apprentice.id);
    let stat_gain = BASE_STAT_GAIN * (hash_to_f32(hash) + 0.5); // 0.05 to 0.15

    // Determine buff based on mentor's class tags.
    let is_tank = mentor_npc.class_tags.iter().any(|t| {
        matches!(
            t.as_str(),
            "knight" | "warrior" | "paladin" | "berserker" | "guardian" | "tank"
        )
    });
    let is_dps = mentor_npc
        .class_tags
        .iter()
        .any(|t| matches!(t.as_str(), "ranger" | "rogue" | "assassin" | "monk"));
    let is_caster = mentor_npc
        .class_tags
        .iter()
        .any(|t| matches!(t.as_str(), "mage" | "warlock" | "sorcerer" | "necromancer"));
    let is_healer = mentor_npc.class_tags.iter().any(|t| {
        matches!(
            t.as_str(),
            "cleric" | "priest" | "healer" | "shaman" | "druid"
        )
    });

    let (stat_name, factor) = if is_tank {
        // Tanks train defense.
        ("armor", 1.0 + stat_gain / apprentice.armor.max(1.0))
    } else if is_dps {
        // DPS train attack + speed.
        (
            "attack",
            1.0 + stat_gain / apprentice.attack_damage.max(1.0),
        )
    } else if is_caster || is_healer {
        // Casters train magic resist (proxy for ability power).
        (
            "magic_resist",
            1.0 + stat_gain / apprentice.magic_resist.max(1.0),
        )
    } else {
        // Generic: small attack boost.
        (
            "attack",
            1.0 + stat_gain * 0.5 / apprentice.attack_damage.max(1.0),
        )
    };

    out.push(WorldDelta::ApplyStatus {
        target_id: apprentice.id,
        status: crate::world_sim::state::StatusEffect {
            kind: crate::world_sim::state::StatusEffectKind::Buff {
                stat: stat_name.into(),
                factor,
            },
            source_id: mentor.id,
            remaining_ms: (MENTORSHIP_CADENCE as u32) * 200, // lasts until next mentorship tick
        },
    });

    // Level-up check for apprentice.
    // Without mutable XP tracking, approximate: if apprentice has been
    // training long enough at current level, emit level-up stats.
    let level_up_cadence = 200 * apprentice.level.max(1) as u64;
    let effective_tick = state.tick / MENTORSHIP_CADENCE;
    let stagger = (mentor.id as u64 + apprentice.id as u64) % level_up_cadence;

    if effective_tick > 0 && (effective_tick + stagger) % level_up_cadence == 0 {
        // Level-up: permanent stat boosts.
        out.push(WorldDelta::Heal {
            target_id: apprentice.id,
            amount: LEVEL_HP_GAIN,
            source_id: mentor.id,
        });

        out.push(WorldDelta::ApplyStatus {
            target_id: apprentice.id,
            status: crate::world_sim::state::StatusEffect {
                kind: crate::world_sim::state::StatusEffectKind::Buff {
                    stat: "attack".into(),
                    factor: 1.0 + LEVEL_ATTACK_GAIN / apprentice.attack_damage.max(1.0),
                },
                source_id: mentor.id,
                remaining_ms: u32::MAX, // permanent
            },
        });
    }
}

// ---------------------------------------------------------------------------
// Deterministic RNG helpers
// ---------------------------------------------------------------------------

fn tick_pair_hash(tick: u64, id_a: u32, id_b: u32) -> u64 {
    let mut h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(id_a as u64)
        .wrapping_mul(0xff51afd7ed558ccd)
        .wrapping_add(id_b as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

fn hash_to_f32(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}
