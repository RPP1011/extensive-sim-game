#![allow(unused)]
//! Adventurer recruitment — ported from headless_campaign.
//!
//! Every `RECRUITMENT_INTERVAL` ticks, new NPC entities may spawn at
//! settlements based on settlement population and treasury. The original
//! system checked guild reputation and gold; here we use settlement
//! treasury and population as proxies.
//!
//! Original: `crates/headless_campaign/src/systems/recruitment.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};

// NEEDS STATE: guild_reputation: f32 on WorldState (or per-settlement reputation)
// NEEDS STATE: max_adventurers: usize on WorldState (population cap)
// NEEDS DELTA: SpawnEntity { template: EntityTemplate, settlement_id: u32 }

/// How often recruitment checks run (in ticks).
const RECRUITMENT_INTERVAL: u64 = 3000;

/// Minimum settlement treasury to afford recruiting.
const RECRUIT_COST: f32 = 50.0;

/// Maximum NPCs per settlement before recruitment stops.
const MAX_NPCS_PER_SETTLEMENT: usize = 20;

/// Base recruitment chance (scaled by settlement population ratio).
const BASE_RECRUIT_CHANCE: f32 = 0.3;

/// Archetype stat templates: (max_hp, attack_damage, armor, move_speed).
const ARCHETYPES: &[(&str, f32, f32, f32, f32)] = &[
    ("knight", 110.0, 12.0, 18.0, 2.5),
    ("ranger", 75.0, 16.0, 8.0, 3.5),
    ("mage", 55.0, 6.0, 5.0, 2.8),
    ("cleric", 65.0, 5.0, 10.0, 2.8),
    ("rogue", 65.0, 18.0, 6.0, 4.0),
    ("paladin", 100.0, 10.0, 15.0, 2.5),
    ("berserker", 95.0, 22.0, 5.0, 3.0),
    ("bard", 60.0, 7.0, 7.0, 3.2),
    ("druid", 70.0, 8.0, 9.0, 2.8),
    ("monk", 75.0, 14.0, 10.0, 3.8),
    ("guardian", 120.0, 8.0, 20.0, 2.2),
    ("shaman", 65.0, 7.0, 8.0, 2.8),
    ("assassin", 60.0, 20.0, 4.0, 4.2),
];

/// Compute recruitment deltas for each settlement.
///
/// The original system mutated `CampaignState` to create new `Adventurer`
/// structs. In the delta architecture, entity creation requires a
/// `SpawnEntity` delta variant that doesn't exist yet. Instead, we
/// express the economic cost of recruitment as a `UpdateTreasury` delta
/// and note that entity spawning needs a new delta type.
///
/// What this system does today:
///   1. Checks each settlement for recruitment eligibility (treasury, NPC cap).
///   2. Uses deterministic hash for the recruitment roll.
///   3. Deducts recruitment cost from settlement treasury via `UpdateTreasury`.
///   4. Notes that a `SpawnEntity` delta is needed for the actual NPC creation.
pub fn compute_recruitment(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % RECRUITMENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_recruitment_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_recruitment_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % RECRUITMENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // Count alive NPCs in the provided entity slice.
    let npc_count = entities
        .iter()
        .filter(|e| {
            e.alive
                && e.kind == EntityKind::Npc
                && e.npc
                    .as_ref()
                    .map(|n| n.home_settlement_id == Some(settlement_id))
                    .unwrap_or(false)
        })
        .count();

    if npc_count >= MAX_NPCS_PER_SETTLEMENT {
        return;
    }

    if settlement.treasury < RECRUIT_COST {
        return;
    }

    // Recruitment chance scales with how far below the cap we are.
    let vacancy_ratio = 1.0 - (npc_count as f32 / MAX_NPCS_PER_SETTLEMENT as f32);
    let recruit_chance = (BASE_RECRUIT_CHANCE * vacancy_ratio).clamp(0.05, 0.8);

    // Deterministic roll from tick + settlement ID.
    let hash = tick_settlement_hash(state.tick, settlement_id);
    let roll = hash_to_f32(hash);

    if roll > recruit_chance {
        return;
    }

    // Deduct recruitment cost from settlement treasury.
    out.push(WorldDelta::UpdateTreasury {
        location_id: settlement_id,
        delta: -RECRUIT_COST,
    });

    // Select archetype deterministically.
    let archetype_idx = (hash >> 16) as usize % ARCHETYPES.len();
    let (_name, _hp, _atk, _armor, _speed) = ARCHETYPES[archetype_idx];

    // NEEDS DELTA: SpawnEntity { settlement_id, kind, max_hp, attack_damage, armor, move_speed, level, class_tag }
}

// ---------------------------------------------------------------------------
// Deterministic RNG helpers
// ---------------------------------------------------------------------------

fn tick_settlement_hash(tick: u64, settlement_id: u32) -> u64 {
    let mut h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(settlement_id as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

fn hash_to_f32(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}
