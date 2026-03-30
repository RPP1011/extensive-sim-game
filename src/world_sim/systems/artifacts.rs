//! Artifact system — notable NPCs forge or discover named artifacts.
//!
//! When an NPC has 3+ classes and level >= 40, they have a chance (10% per
//! check, gated by tick hash) to forge/discover an artifact. The artifact is
//! recorded as a chronicle entry with category Discovery.
//!
//! Artifact names combine a prefix + material + type, e.g.
//! "Stormbreaker, the Adamantine Greatsword".
//!
//! Max 1 artifact per NPC (checked via chronicle scan).
//!
//! Cadence: every 200 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::*;

const ARTIFACT_INTERVAL: u64 = 500;
const MIN_CLASSES: usize = 4;
const MIN_LEVEL: u32 = 35;

static PREFIXES: &[&str] = &[
    "Dawnfire", "Stormbreaker", "Nightfall", "Ironheart", "Soulkeeper",
    "Whisperwind", "Bloodthorn", "Starweaver", "Frostbane", "Shadowmend",
    "Oathkeeper", "Griefender", "Peacemaker", "Ruinbringer", "Truthseeker",
];

static MATERIALS: &[&str] = &[
    "Adamantine", "Mithril", "Obsidian", "Crystal", "Enchanted",
    "Runed", "Ancient", "Celestial", "Abyssal", "Dragonbone",
];

static TYPES: &[&str] = &[
    "Greatsword", "Bow", "Staff", "Shield", "Amulet",
    "Ring", "Crown", "Tome", "Gauntlets", "Cloak",
];

/// Generate a deterministic artifact name from entity ID and tick.
fn artifact_name(entity_id: u32, tick: u64) -> String {
    let prefix_idx = entity_hash(entity_id, tick, 0xA871) as usize % PREFIXES.len();
    let material_idx = entity_hash(entity_id, tick, 0xA872) as usize % MATERIALS.len();
    let type_idx = entity_hash(entity_id, tick, 0xA873) as usize % TYPES.len();

    format!(
        "{}, the {} {}",
        PREFIXES[prefix_idx], MATERIALS[material_idx], TYPES[type_idx]
    )
}

/// Returns true if the tick hash gates this entity for artifact creation (~3% chance).
fn artifact_chance_passes(entity_id: u32, tick: u64) -> bool {
    entity_hash(entity_id, tick, 0xA870) % 100 < 3
}

/// Check whether this NPC has already forged an artifact (scan chronicle).
fn already_has_artifact(state: &WorldState, entity_id: u32) -> bool {
    state.chronicle.iter().any(|entry| {
        entry.category == ChronicleCategory::Discovery
            && entry.entity_ids.contains(&entity_id)
            && entry.text.contains("forged")
    })
}

pub fn compute_artifacts(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % ARTIFACT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Must have 3+ classes and level >= 40.
        if entity.level < MIN_LEVEL { continue; }
        if npc.classes.len() < MIN_CLASSES { continue; }

        // Max 1 artifact per NPC.
        if already_has_artifact(state, entity.id) {
            continue;
        }

        // 10% chance gated by deterministic hash.
        if !artifact_chance_passes(entity.id, state.tick) {
            continue;
        }

        let npc_name = entity_display_name(entity);
        let artifact_full_name = artifact_name(entity.id, state.tick);

        // Find settlement name for the chronicle entry.
        let settlement_name = npc
            .home_settlement_id
            .and_then(|sid| state.settlement(sid))
            .map(|s| s.name.as_str())
            .unwrap_or("the wilderness");

        let text = format!("{} forged {} at {}", npc_name, artifact_full_name, settlement_name);
        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Discovery,
                text,
                entity_ids: vec![entity.id],
            },
        });
    }
}
