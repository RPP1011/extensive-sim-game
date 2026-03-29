//! Adventurer awakening/transformation system — every 1000 ticks.
//!
//! Rare dramatic events where an adventurer undergoes a power transformation,
//! permanently changing their capabilities. Each adventurer can awaken at most
//! once in their lifetime. Awakened adventurers attract attention: reputation
//! stories spread and nemeses take notice.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often the awakening system ticks (in ticks).
const AWAKENING_INTERVAL: u64 = 33;

/// Base chance of awakening when conditions are met (1%).
const AWAKENING_CHANCE: f32 = 0.01;

/// Permanent stat boost multiplier applied to primary stat.
const POWER_BOOST: f32 = 0.25;

/// Morale boost on awakening.
const MORALE_BOOST: f32 = 20.0;

/// Chronicle significance for awakening events.
const CHRONICLE_SIGNIFICANCE: f32 = 10.0;

/// Nemesis interest boost when an adventurer awakens.
const NEMESIS_INTEREST_BOOST: f32 = 5.0;

/// Tick the awakening system. Checks each adventurer for awakening conditions
/// and triggers transformations at a 1% chance when conditions are met.
pub fn tick_awakening(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % AWAKENING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Collect awakened adventurer IDs to enforce max-1-per-adventurer.
    let already_awakened: Vec<u32> = state
        .awakenings
        .iter()
        .map(|a| a.adventurer_id)
        .collect();

    // Collect temple devotion max for DivineFavor check.
    let max_devotion: f32 = state
        .temples
        .iter()
        .map(|t| t.devotion)
        .fold(0.0_f32, f32::max);

    // Collect companion bond levels indexed by owner.
    let companion_bonds: Vec<(u32, f32)> = state
        .companions
        .iter()
        .map(|c| (c.owner_id, c.bond_level))
        .collect();

    // Snapshot adventurer data for evaluation (avoid borrow conflict).
    let candidates: Vec<Candidate> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead && !already_awakened.contains(&a.id))
        .map(|a| {
            let knowledge_tags = a
                .history_tags
                .iter()
                .filter(|(k, _)| {
                    k.contains("knowledge")
                        || k.contains("scholarly")
                        || k.contains("research")
                        || k.contains("arcane")
                        || k.contains("lore")
                })
                .map(|(_, v)| *v)
                .sum::<u32>();

            let espionage_tags = a
                .history_tags
                .iter()
                .filter(|(k, _)| {
                    k.contains("espionage")
                        || k.contains("stealth")
                        || k.contains("infiltration")
                        || k.contains("spy")
                        || k.contains("sabotage")
                })
                .map(|(_, v)| *v)
                .sum::<u32>();

            let near_death_count = a.history_tags.get("near_death").copied().unwrap_or(0);

            let companion_bond = companion_bonds
                .iter()
                .find(|(owner, _)| *owner == a.id)
                .map(|(_, bond)| *bond)
                .unwrap_or(0.0);

            Candidate {
                id: a.id,
                archetype: a.archetype.clone(),
                status: a.status,
                near_death_count,
                knowledge_tags,
                espionage_tags,
                companion_bond,
                level: a.level,
            }
        })
        .collect();

    // Evaluate each candidate for awakening conditions.
    let mut new_awakenings: Vec<(u32, AwakeningType, String, String)> = Vec::new();

    for c in &candidates {
        // Try each awakening type in priority order.
        let awakening = check_battle_rage(c)
            .or_else(|| check_arcane_bloom(c))
            .or_else(|| check_divine_favor(c, max_devotion))
            .or_else(|| check_shadow_merge(c))
            .or_else(|| check_nature_bond(c))
            .or_else(|| check_ancestral_call(c));

        if let Some((awakening_type, ability_name, description)) = awakening {
            // Roll for 1% chance.
            let roll = lcg_f32(&mut state.rng);
            if roll < AWAKENING_CHANCE {
                new_awakenings.push((c.id, awakening_type, ability_name, description));
            }
        }
    }

    // Apply awakenings.
    for (adv_id, awakening_type, ability_name, description) in new_awakenings {
        // Record the awakening.
        state.awakenings.push(Awakening {
            adventurer_id: adv_id,
            awakening_type,
            tick: state.tick,
            power_boost: POWER_BOOST,
            new_ability: ability_name.clone(),
        });

        // Apply stat boost and morale to the adventurer.
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            apply_stat_boost(adv, awakening_type);
            adv.morale = (adv.morale + MORALE_BOOST).min(100.0);

            // Add history tag for tracking.
            *adv.history_tags.entry("awakened".into()).or_default() += 1;
        }

        // Add chronicle entry with significance 10.
        state.chronicle.push(ChronicleEntry {
            tick: state.tick,
            entry_type: ChronicleType::HeroicDeed,
            text: description.clone(),
            participants: vec![adv_id],
            location_id: None,
            faction_id: None,
            significance: CHRONICLE_SIGNIFICANCE,
        });

        // Boost nemesis interest — awakened adventurers draw attention.
        for nemesis in &mut state.nemeses {
            if !nemesis.defeated {
                nemesis.strength += NEMESIS_INTEREST_BOOST;
            }
        }

        // Generate a reputation story about the awakening.
        let story_id = state.next_event_id;
        state.next_event_id += 1;
        state.reputation_stories.push(ReputationStory {
            id: story_id,
            text: format!(
                "An adventurer has undergone a powerful awakening: {}",
                description
            ),
            story_type: StoryType::MysteriousEvent,
            origin_region_id: 0,
            spread_to: vec![0],
            accuracy: 1.0,
            impact: 5.0,
            created_tick: state.tick,
        });

        // Emit world event.
        events.push(WorldEvent::AwakeningTriggered {
            adventurer_id: adv_id,
            awakening_type,
            description,
        });
    }
}

/// Apply a permanent +25% boost to the primary stat for the awakening type.
fn apply_stat_boost(adv: &mut Adventurer, awakening_type: AwakeningType) {
    match awakening_type {
        AwakeningType::BattleRage => {
            adv.stats.attack *= 1.0 + POWER_BOOST;
        }
        AwakeningType::ArcaneBloom => {
            adv.stats.ability_power *= 1.0 + POWER_BOOST;
        }
        AwakeningType::DivineFavor => {
            adv.stats.max_hp *= 1.0 + POWER_BOOST;
        }
        AwakeningType::ShadowMerge => {
            adv.stats.speed *= 1.0 + POWER_BOOST;
        }
        AwakeningType::NatureBond => {
            adv.stats.defense *= 1.0 + POWER_BOOST;
        }
        AwakeningType::AncestralCall => {
            // Ancestral power boosts all stats by a smaller amount.
            let half_boost = POWER_BOOST * 0.5;
            adv.stats.attack *= 1.0 + half_boost;
            adv.stats.defense *= 1.0 + half_boost;
            adv.stats.speed *= 1.0 + half_boost;
        }
    }
}

// ---------------------------------------------------------------------------
// Condition checkers
// ---------------------------------------------------------------------------

struct Candidate {
    id: u32,
    archetype: String,
    status: AdventurerStatus,
    near_death_count: u32,
    knowledge_tags: u32,
    espionage_tags: u32,
    companion_bond: f32,
    level: u32,
}

/// BattleRage: adventurer survived 5+ near-death events, currently in combat.
fn check_battle_rage(c: &Candidate) -> Option<(AwakeningType, String, String)> {
    if c.near_death_count >= 5
        && (c.status == AdventurerStatus::Fighting || c.status == AdventurerStatus::OnMission)
    {
        Some((
            AwakeningType::BattleRage,
            "Berserker's Fury".into(),
            format!(
                "After surviving {} brushes with death, a warrior awakens berserker mode",
                c.near_death_count
            ),
        ))
    } else {
        None
    }
}

/// ArcaneBloom: mage with 10+ knowledge/scholarly tags.
fn check_arcane_bloom(c: &Candidate) -> Option<(AwakeningType, String, String)> {
    let is_mage = c.archetype.contains("mage")
        || c.archetype.contains("wizard")
        || c.archetype.contains("sorcerer")
        || c.archetype.contains("warlock");
    if is_mage && c.knowledge_tags >= 10 {
        Some((
            AwakeningType::ArcaneBloom,
            "Arcane Transcendence".into(),
            "A mage unlocks hidden arcane potential through deep study".into(),
        ))
    } else {
        None
    }
}

/// DivineFavor: temple devotion > 80 for any order.
fn check_divine_favor(
    _c: &Candidate,
    max_devotion: f32,
) -> Option<(AwakeningType, String, String)> {
    if max_devotion > 80.0 {
        Some((
            AwakeningType::DivineFavor,
            "Divine Aegis".into(),
            format!(
                "Temple devotion of {:.0} grants divine blessing to a faithful servant",
                max_devotion
            ),
        ))
    } else {
        None
    }
}

/// ShadowMerge: rogue with 5+ espionage missions.
fn check_shadow_merge(c: &Candidate) -> Option<(AwakeningType, String, String)> {
    let is_rogue = c.archetype.contains("rogue")
        || c.archetype.contains("assassin")
        || c.archetype.contains("thief")
        || c.archetype.contains("ranger");
    if is_rogue && c.espionage_tags >= 5 {
        Some((
            AwakeningType::ShadowMerge,
            "Shadow Step".into(),
            "A rogue becomes one with the shadows after countless covert missions".into(),
        ))
    } else {
        None
    }
}

/// NatureBond: companion bond > 90.
fn check_nature_bond(c: &Candidate) -> Option<(AwakeningType, String, String)> {
    if c.companion_bond > 90.0 {
        Some((
            AwakeningType::NatureBond,
            "Primal Resonance".into(),
            format!(
                "A deep bond with a companion ({:.0}) awakens primal power",
                c.companion_bond
            ),
        ))
    } else {
        None
    }
}

/// AncestralCall: high level (bloodline prestige proxy > 50 → level > 8).
/// Uses adventurer level as a proxy for bloodline prestige since the state
/// doesn't track bloodline directly.
fn check_ancestral_call(c: &Candidate) -> Option<(AwakeningType, String, String)> {
    // Level 8+ acts as the prestige threshold (level is 1-based, max ~15).
    if c.level >= 8 {
        Some((
            AwakeningType::AncestralCall,
            "Ancestral Awakening".into(),
            format!(
                "At level {}, an ancient bloodline power surges forth",
                c.level
            ),
        ))
    } else {
        None
    }
}
