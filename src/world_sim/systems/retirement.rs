//! Adventurer retirement — ported from headless_campaign.
//!
//! Every `RETIREMENT_INTERVAL` ticks, high-level NPC entities with good
//! morale and loyalty may retire, becoming legacy bonuses for their
//! home settlement. Retired NPCs are removed via `Die` delta and their
//! legacy bonus is applied as a settlement treasury/stockpile boost.
//!
//! Original: `crates/headless_campaign/src/systems/retirement.rs`

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, EntityKind, WorldState};
use crate::world_sim::state::{entity_hash, entity_hash_f32};


/// How often retirement checks run (in ticks).
const RETIREMENT_INTERVAL: u64 = 17;

/// Minimum level for retirement eligibility.
const MIN_RETIREMENT_LEVEL: u32 = 10;

/// Minimum HP ratio for retirement (proxy for morale > 70).
const MIN_HP_RATIO_FOR_RETIRE: f32 = 0.7;

/// Auto-retire chance per eligible NPC per check.
const AUTO_RETIRE_CHANCE: f32 = 0.10;

/// Legacy bonus amount per level (base, scaled by level / 10).
const LEGACY_BONUS_PER_LEVEL: f32 = 5.0;

/// Maximum retired NPCs whose bonuses stack at one settlement.
const MAX_SAME_TYPE_STACK: usize = 3;

/// Legacy types determine what bonus a retired adventurer provides.
/// Mapped from archetype/class tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LegacyType {
    /// +XP for training (tanky archetypes).
    TrainingMentor,
    /// -supply consumption (rogue/ranger archetypes).
    Quartermaster,
    /// +diplomacy bonus (mage/cleric archetypes).
    Diplomat,
    /// +combat power (fighter archetypes).
    Strategist,
    /// +recruit quality (social archetypes).
    Recruiter,
    /// +quest rewards (knowledge archetypes).
    Lorekeeper,
}

impl LegacyType {
    /// Select legacy type based on NPC class tags.
    fn from_class_tags(tags: &[String], hash: u64) -> Self {
        let roll = (hash >> 48) as f32 / (1u64 << 16) as f32;

        // Check tags for archetype hints.
        let has_tank = tags.iter().any(|t| {
            matches!(
                t.as_str(),
                "knight" | "paladin" | "guardian" | "tank" | "berserker" | "warrior"
            )
        });
        let has_rogue = tags.iter().any(|t| {
            matches!(
                t.as_str(),
                "rogue" | "assassin" | "monk" | "ranger" | "artificer"
            )
        });
        let has_caster = tags.iter().any(|t| {
            matches!(
                t.as_str(),
                "mage" | "cleric" | "druid" | "shaman" | "warlock" | "necromancer" | "healer"
            )
        });
        let has_social = tags.iter().any(|t| matches!(t.as_str(), "bard"));

        if has_tank {
            if roll < 0.5 {
                LegacyType::Strategist
            } else {
                LegacyType::TrainingMentor
            }
        } else if has_rogue {
            if roll < 0.5 {
                LegacyType::Quartermaster
            } else {
                LegacyType::Recruiter
            }
        } else if has_caster {
            if roll < 0.5 {
                LegacyType::Lorekeeper
            } else {
                LegacyType::Diplomat
            }
        } else if has_social {
            if roll < 0.5 {
                LegacyType::Diplomat
            } else {
                LegacyType::Recruiter
            }
        } else {
            // Fallback: deterministic selection from hash.
            match (hash >> 32) as usize % 6 {
                0 => LegacyType::TrainingMentor,
                1 => LegacyType::Quartermaster,
                2 => LegacyType::Diplomat,
                3 => LegacyType::Strategist,
                4 => LegacyType::Recruiter,
                _ => LegacyType::Lorekeeper,
            }
        }
    }

    /// Settlement bonus amount. Applied as treasury boost.
    fn base_bonus(&self) -> f32 {
        match self {
            LegacyType::TrainingMentor => 0.05, // +5% XP → small treasury boost
            LegacyType::Quartermaster => 0.08,  // -8% supply cost → stockpile boost
            LegacyType::Diplomat => 0.03,       // +3% relations → treasury boost
            LegacyType::Strategist => 0.05,     // +5% combat power → treasury boost
            LegacyType::Recruiter => 0.04,      // +4% recruit quality → treasury boost
            LegacyType::Lorekeeper => 0.06,     // +6% quest rewards → treasury boost
        }
    }
}

/// Compute retirement deltas for all eligible NPC entities.
///
/// An NPC is eligible for retirement when:
///   - Level >= MIN_RETIREMENT_LEVEL
///   - HP ratio >= MIN_HP_RATIO_FOR_RETIRE (proxy for high morale/loyalty)
///   - Not currently on a hostile grid
///
/// Retirement is expressed as:
///   1. `Die` delta to remove the entity from play.
///   2. `UpdateTreasury` delta to grant the legacy bonus to their home settlement.
pub fn compute_retirement(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % RETIREMENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_retirement_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_retirement_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % RETIREMENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for entity in entities {
        if !entity.alive || entity.kind != EntityKind::Npc {
            continue;
        }
        let npc = match &entity.npc {
            Some(n) => n,
            None => continue,
        };

        // Eligibility checks.
        if entity.level < MIN_RETIREMENT_LEVEL {
            continue;
        }

        let hp_ratio = entity.hp / entity.max_hp.max(1.0);
        if hp_ratio < MIN_HP_RATIO_FOR_RETIRE {
            continue;
        }

        // Don't retire during combat.
        let on_hostile_grid = entity
            .grid_id
            .and_then(|gid| state.fidelity_zone(gid))
            .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
            .unwrap_or(false);
        if on_hostile_grid {
            continue;
        }

        // Deterministic retirement roll.
        let hash = entity_hash(entity.id, state.tick, 0) as u64;
        let roll = entity_hash_f32(entity.id, state.tick, 0);
        if roll >= AUTO_RETIRE_CHANCE {
            continue;
        }

        // Select legacy type from class tags.
        let legacy = LegacyType::from_class_tags(&npc.class_tags, hash);

        // Calculate bonus: base * (level / 10).
        let bonus_value =
            legacy.base_bonus() * (entity.level as f32 / 10.0) * LEGACY_BONUS_PER_LEVEL;

        // Remove entity from play (represents retirement, not death).
        out.push(WorldDelta::Die {
            entity_id: entity.id,
        });

        // Grant legacy bonus to home settlement as treasury boost.
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement_id,
            delta: bonus_value,
        });

        // Quartermaster legacy → stockpile bonus (food).
        if legacy == LegacyType::Quartermaster {
            out.push(WorldDelta::UpdateStockpile {
                settlement_id: settlement_id,
                commodity: crate::world_sim::commodity::FOOD,
                delta: bonus_value * 10.0,
            });
        }
    }
}

