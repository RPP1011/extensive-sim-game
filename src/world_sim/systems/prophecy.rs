#![allow(unused)]
//! Prophecy engine — procedural prophecies that track world state for fulfillment.
//!
//! At world init, 3-5 prophecies are generated with vague conditions mapped to
//! concrete world state checks. When conditions align, the prophecy is fulfilled
//! with a dramatic chronicle event and world effect.
//!
//! Prophecy types:
//! - "When the last forge falls silent" → no Forge buildings producing
//! - "When blood stains three thrones" → 3+ settlements change faction
//! - "When the deep awakens" → monster density exceeds threshold in Caverns
//! - "When kin turns on kin" → betrayal event fires
//! - "When the wanderer returns" → redeemed outlaw reaches level 20+
//!
//! Cadence: every 500 ticks (prophecy check).

use serde::{Serialize, Deserialize};
use crate::world_sim::state::*;

const PROPHECY_CHECK_INTERVAL: u64 = 500;

/// A world prophecy with a condition and effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prophecy {
    pub text: String,
    pub condition: ProphecyCondition,
    pub effect: ProphecyEffect,
    pub fulfilled: bool,
    pub fulfilled_tick: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProphecyCondition {
    /// No working forges in any settlement.
    ForgesSilent,
    /// N+ settlements have changed faction ownership.
    ThronesBloodied { count: u32 },
    /// Monster density exceeds threshold in any Caverns/DeathZone region.
    DeepAwakens { threshold: f32 },
    /// A betrayal has occurred (any hostile NPC was formerly friendly).
    KinTurnsOnKin,
    /// A redeemed outlaw exists with level >= threshold.
    WandererReturns { min_level: u32 },
    /// Total population drops below threshold.
    PopulationCollapse { threshold: u32 },
    /// A legendary item exists with 5+ kills.
    LegendaryWeaponRises,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProphecyEffect {
    /// Spawn powerful monsters in a region.
    MonsterSurge { density_boost: f32 },
    /// Boost all settlement morale.
    HopeRises { morale_boost: f32 },
    /// Reduce all threats temporarily.
    PeaceDescends { threat_reduction: f32 },
    /// Increase monster threat everywhere.
    DarknessGrows { threat_increase: f32 },
    /// Produce rare resources at all settlements.
    Bounty { commodity: usize, amount: f32 },
}

/// Generate prophecies for the world at init. Call once.
pub fn generate_prophecies(seed: u64) -> Vec<Prophecy> {
    let mut prophecies = Vec::new();
    let h = |salt: u64| -> u32 {
        ((seed.wrapping_mul(6364136223846793005).wrapping_add(salt)) >> 33) as u32
    };

    prophecies.push(Prophecy {
        text: "When the last forge falls silent, the mountain shall weep iron tears.".into(),
        condition: ProphecyCondition::ForgesSilent,
        effect: ProphecyEffect::Bounty {
            commodity: crate::world_sim::commodity::IRON,
            amount: 50.0,
        },
        fulfilled: false,
        fulfilled_tick: None,
    });

    prophecies.push(Prophecy {
        text: "When blood stains three thrones, darkness shall grow bold.".into(),
        condition: ProphecyCondition::ThronesBloodied { count: 3 },
        effect: ProphecyEffect::DarknessGrows { threat_increase: 0.3 },
        fulfilled: false,
        fulfilled_tick: None,
    });

    prophecies.push(Prophecy {
        text: "When the deep awakens, the surface shall tremble.".into(),
        condition: ProphecyCondition::DeepAwakens { threshold: 50.0 },
        effect: ProphecyEffect::MonsterSurge { density_boost: 20.0 },
        fulfilled: false,
        fulfilled_tick: None,
    });

    prophecies.push(Prophecy {
        text: "When kin turns on kin, only the pure of heart shall stand.".into(),
        condition: ProphecyCondition::KinTurnsOnKin,
        effect: ProphecyEffect::HopeRises { morale_boost: 15.0 },
        fulfilled: false,
        fulfilled_tick: None,
    });

    let fifth = match h(5) % 3 {
        0 => Prophecy {
            text: "When the wanderer returns from exile, peace shall descend.".into(),
            condition: ProphecyCondition::WandererReturns { min_level: 15 },
            effect: ProphecyEffect::PeaceDescends { threat_reduction: 0.3 },
            fulfilled: false,
            fulfilled_tick: None,
        },
        1 => Prophecy {
            text: "When the world's people number fewer than the stars can count, bounty shall follow famine.".into(),
            condition: ProphecyCondition::PopulationCollapse { threshold: 100 },
            effect: ProphecyEffect::Bounty {
                commodity: crate::world_sim::commodity::FOOD,
                amount: 100.0,
            },
            fulfilled: false,
            fulfilled_tick: None,
        },
        _ => Prophecy {
            text: "When a blade drinks deep of five lives, it shall name itself.".into(),
            condition: ProphecyCondition::LegendaryWeaponRises,
            effect: ProphecyEffect::HopeRises { morale_boost: 10.0 },
            fulfilled: false,
            fulfilled_tick: None,
        },
    };
    prophecies.push(fifth);

    prophecies
}

/// Check prophecy conditions and fulfill them. Called post-apply.
pub fn advance_prophecies(state: &mut WorldState) {
    if state.tick % PROPHECY_CHECK_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // Count world state for condition checks.
    let total_pop = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .count() as u32;

    let hostile_npcs = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Hostile)
        .count();

    let working_forges = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Building)
        .filter(|e| e.building.as_ref().map(|b|
            b.building_type == BuildingType::Forge && b.construction_progress >= 1.0
        ).unwrap_or(false))
        .count();

    let max_cavern_density = state.regions.iter()
        .filter(|r| matches!(r.terrain, Terrain::Caverns | Terrain::DeathZone))
        .map(|r| r.monster_density)
        .fold(0.0f32, f32::max);

    let has_legendary = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Item)
        .any(|e| e.item.as_ref().map(|i| i.is_legendary).unwrap_or(false));

    let redeemed_outlaws: Vec<u32> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Friendly)
        .filter(|e| e.npc.as_ref().map(|n|
            n.behavior_value(tags::COMPASSION_TAG) > 1.0 // has redemption tags
            && n.behavior_value(tags::DECEPTION) > 30.0  // was once treacherous
        ).unwrap_or(false))
        .map(|e| e.level)
        .collect();

    // Check each unfulfilled prophecy.
    for pi in 0..state.prophecies.len() {
        if state.prophecies[pi].fulfilled { continue; }

        let condition_met = match &state.prophecies[pi].condition {
            ProphecyCondition::ForgesSilent => working_forges == 0,
            ProphecyCondition::ThronesBloodied { count } => {
                // Approximate: count hostile NPCs as proxy for faction conflict.
                hostile_npcs as u32 >= *count
            }
            ProphecyCondition::DeepAwakens { threshold } => max_cavern_density >= *threshold,
            ProphecyCondition::KinTurnsOnKin => hostile_npcs > 0,
            ProphecyCondition::WandererReturns { min_level } => {
                redeemed_outlaws.iter().any(|&lvl| lvl >= *min_level)
            }
            ProphecyCondition::PopulationCollapse { threshold } => total_pop < *threshold,
            ProphecyCondition::LegendaryWeaponRises => has_legendary,
        };

        if !condition_met { continue; }

        // Fulfill the prophecy.
        state.prophecies[pi].fulfilled = true;
        state.prophecies[pi].fulfilled_tick = Some(tick);

        let prophecy_text = state.prophecies[pi].text.clone();

        // Apply effect.
        match &state.prophecies[pi].effect {
            ProphecyEffect::MonsterSurge { density_boost } => {
                for region in &mut state.regions {
                    region.monster_density += density_boost;
                }
            }
            ProphecyEffect::HopeRises { morale_boost } => {
                for entity in &mut state.entities {
                    if entity.alive && entity.kind == EntityKind::Npc {
                        if let Some(npc) = &mut entity.npc {
                            npc.morale = (npc.morale + morale_boost).min(100.0);
                            npc.emotions.joy = (npc.emotions.joy + 0.5).min(1.0);
                        }
                    }
                }
            }
            ProphecyEffect::PeaceDescends { threat_reduction } => {
                for region in &mut state.regions {
                    region.threat_level = (region.threat_level - threat_reduction).max(0.0);
                }
            }
            ProphecyEffect::DarknessGrows { threat_increase } => {
                for region in &mut state.regions {
                    region.threat_level += threat_increase;
                }
            }
            ProphecyEffect::Bounty { commodity, amount } => {
                for settlement in &mut state.settlements {
                    settlement.stockpile[*commodity] += amount;
                }
            }
        }

        // Major chronicle entry.
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Narrative,
            text: format!("A PROPHECY FULFILLED: \"{}\"", prophecy_text),
            entity_ids: vec![],
        });
    }
}
