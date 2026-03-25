//! TWI-inspired class system — classes are EARNED from behavior, not chosen.
//!
//! The system watches what adventurers do and grants classes when behavioral
//! fingerprints match class templates.  Each class levels independently via
//! quadratic XP curves, and skills are granted at level thresholds.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::skill_templates::select_skill_template;
use crate::headless_campaign::state::{
    lcg_f32, Adventurer, AdventurerStatus, BattleStatus, BehaviorLedger, CampaignState, ClassInstance,
    ClassTemplate, ConsolidationOffer, GrantedSkill, SkillRarity,
};

/// Class system fires every 50 ticks (5 s game time) — fast enough for early class development.
const CLASS_TICK_INTERVAL: u64 = 50;

/// Consolidation check fires every 500 ticks.
const CONSOLIDATION_INTERVAL: u64 = 500;

/// Reactive narrative (shame, crisis, erosion, chronicle) fires every 200 ticks.
const REACTIVE_NARRATIVE_INTERVAL: u64 = 200;

/// Minimum level on both classes before consolidation is offered.
const CONSOLIDATION_MIN_LEVEL: u32 = 10;

/// Probability of auto-accepting a consolidation offer.
const CONSOLIDATION_ACCEPT_PROB: f32 = 0.70;

/// Ticks before a consolidation offer expires.
const CONSOLIDATION_DEADLINE_TICKS: u32 = 1000;

/// Minimum class level for evolution.
const EVOLUTION_MIN_LEVEL: u32 = 20;

/// Minimum battles for evolution eligibility.
const EVOLUTION_MIN_BATTLES: f32 = 50.0;

/// Minimum quests-related activity for evolution.
const EVOLUTION_MIN_QUESTS: f32 = 20.0;

/// Exponential decay factor for recent-window fields each tick.
const RECENT_DECAY: f32 = 0.95;

/// Stagnation threshold where XP gain is halved.
/// Set high enough that normal event gaps don't trigger it.
/// With class tick every 50 game ticks, 5000 stagnation = 250,000 game ticks of no activity.
const STAGNATION_HALF: u32 = 5000;
/// Stagnation threshold where XP gain is frozen.
const STAGNATION_FREEZE: u32 = 10000;

/// XP trickle between overlapping classes (15%).
const RESONANCE_TRICKLE: f32 = 0.15;

/// Level thresholds at which skills are granted (25 grants across 100 levels).
const SKILL_THRESHOLDS: &[u32] = &[
    2, 3, 4,                    // T1 Novice: 3 skills (quick start)
    5, 6, 7,                    // T2 Journeyman: 3 skills (reachable in BFS)
    10, 13, 17,                 // T3 Adept: 3 skills
    21, 25, 30, 35,             // T4 Expert: 4 skills
    40, 50, 60, 70,             // T5 Master: 4 skills
    85, 90, 95,                 // T6 Legendary: 3 skills
    100,                        // T7 Mythic: 1 capstone
];

/// Map level threshold to skill rarity.
fn rarity_for_threshold(level: u32) -> SkillRarity {
    match level {
        0..=4 => SkillRarity::Common,       // T1
        5..=7 => SkillRarity::Common,       // T2 (still common but broader)
        8..=17 => SkillRarity::Uncommon,    // T3
        18..=35 => SkillRarity::Rare,       // T4
        36..=70 => SkillRarity::Rare,       // T5
        71..=95 => SkillRarity::Capstone,   // T6
        96..=100 => SkillRarity::Unique,    // T7
        _ => SkillRarity::Common,
    }
}

/// Map level threshold to skill tier (1-7).
fn tier_for_threshold(level: u32) -> u32 {
    match level {
        0..=4 => 1,     // T1 Novice
        5..=7 => 2,     // T2 Journeyman
        8..=17 => 3,    // T3 Adept
        18..=35 => 4,   // T4 Expert
        36..=70 => 5,   // T5 Master
        71..=95 => 6,   // T6 Legendary
        96..=100 => 7,  // T7 Mythic
        _ => 1,
    }
}

/// Mentorship observation credit fraction (25% of mentor's earned behavior).
const MENTORSHIP_OBSERVATION_FRACTION: f32 = 0.25;

/// Minimum total behavior sum before unique class generation is considered.
const UNIQUE_CLASS_MIN_BEHAVIOR: f32 = 500.0;

/// Cosine similarity threshold -- if no template exceeds this, adventurer gets a unique class.
const UNIQUE_CLASS_SIM_THRESHOLD: f32 = 0.6;

/// Behavior dimension names, ordered by fingerprint index.
const BEHAVIOR_DIM_NAMES: &[&str] = &[
    "Combat", "Ranged", "Healing", "Diplomacy", "Trade",
    "Crafting", "Exploration", "Command", "Stealth", "Research",
    "Defense", "Support",
];

// ---------------------------------------------------------------------------
// Class templates
// ---------------------------------------------------------------------------

fn starter_templates() -> Vec<ClassTemplate> {
    vec![
        ct("Laborer", &[("melee_combat", 0.2), ("damage_absorbed", 0.2)], 0.1, &["physical", "common"]),
        ct("Hunter", &[("ranged_combat", 0.2), ("areas_explored", 0.2)], 0.1, &["outdoor", "common"]),
        ct("Traveler", &[("areas_explored", 0.3), ("diplomacy_actions", 0.1)], 0.1, &["wanderer", "common"]),
        ct("Apprentice", &[("research_performed", 0.2), ("items_crafted", 0.2)], 0.1, &["learning", "common"]),
        ct("Farmhand", &[("allies_supported", 0.2), ("trades_completed", 0.1)], 0.1, &["rural", "common"]),
        ct("Militia", &[("melee_combat", 0.3), ("damage_absorbed", 0.1)], 0.1, &["defense", "common"]),
        ct("Peddler", &[("trades_completed", 0.3), ("diplomacy_actions", 0.1)], 0.1, &["trade", "common"]),
        ct("Herbalist", &[("healing_given", 0.2), ("research_performed", 0.1)], 0.1, &["medicine", "common"]),
        ct("Scribe", &[("research_performed", 0.3), ("diplomacy_actions", 0.1)], 0.1, &["knowledge", "common"]),
        ct("Pickpocket", &[("stealth_actions", 0.3), ("trades_completed", 0.1)], 0.1, &["street", "common"]),
        ct("Errand Runner", &[("areas_explored", 0.2), ("allies_supported", 0.2)], 0.1, &["service", "common"]),
        ct("Stablehand", &[("allies_supported", 0.3), ("damage_absorbed", 0.1)], 0.1, &["animal", "common"]),
    ]
}

fn base_templates() -> Vec<ClassTemplate> {
    vec![
        ct("Warrior", &[("melee_combat", 0.4), ("damage_absorbed", 0.3)], 0.30, &["combat"]),
        ct("Ranger", &[("ranged_combat", 0.4), ("areas_explored", 0.3)], 0.30, &["combat", "exploration"]),
        ct("Healer", &[("healing_given", 0.5), ("allies_supported", 0.3)], 0.30, &["support"]),
        ct("Diplomat", &[("diplomacy_actions", 0.5), ("units_commanded", 0.2)], 0.30, &["social"]),
        ct("Merchant", &[("trades_completed", 0.5), ("diplomacy_actions", 0.2)], 0.30, &["economy"]),
        ct("Scholar", &[("research_performed", 0.5), ("areas_explored", 0.2)], 0.30, &["knowledge"]),
        ct("Rogue", &[("stealth_actions", 0.4), ("melee_combat", 0.2)], 0.30, &["stealth", "combat"]),
        ct("Artisan", &[("items_crafted", 0.5), ("trades_completed", 0.2)], 0.30, &["crafting"]),
        ct("Commander", &[("units_commanded", 0.4), ("melee_combat", 0.2)], 0.30, &["leadership"]),
        ct("Scout", &[("areas_explored", 0.4), ("stealth_actions", 0.3)], 0.30, &["exploration", "stealth"]),
        ct("Guardian", &[("damage_absorbed", 0.4), ("allies_supported", 0.3)], 0.30, &["defense"]),
    ]
}

fn rare_templates() -> Vec<ClassTemplate> {
    vec![
        ct_rare("Spellblade", &[("melee_combat", 0.3), ("research_performed", 0.3)], 0.7, &["combat", "knowledge"], SkillRarity::Rare),
        ct_rare("Plague Doctor", &[("healing_given", 0.3), ("research_performed", 0.3)], 0.7, &["support", "knowledge"], SkillRarity::Rare),
        ct_rare("Shadowmerchant", &[("trades_completed", 0.3), ("stealth_actions", 0.3)], 0.7, &["economy", "stealth"], SkillRarity::Rare),
        ct_rare("Warlord", &[("units_commanded", 0.3), ("melee_combat", 0.3), ("damage_absorbed", 0.2)], 0.8, &["leadership", "combat"], SkillRarity::Rare),
    ]
}

/// Public classes — granted when witnessed heroic acts are high.
fn public_class_templates() -> Vec<ClassTemplate> {
    vec![
        ct("Champion", &[("melee_combat", 0.4), ("damage_absorbed", 0.3)], 0.5, &["combat", "public"]),
        ct("Herald", &[("diplomacy_actions", 0.4), ("units_commanded", 0.3)], 0.5, &["social", "public"]),
        ct("Paragon", &[("healing_given", 0.3), ("allies_supported", 0.3)], 0.5, &["support", "public"]),
        ct("Banner Knight", &[("units_commanded", 0.4), ("damage_absorbed", 0.2)], 0.5, &["leadership", "public"]),
        ct("People's Blade", &[("melee_combat", 0.3), ("allies_supported", 0.3)], 0.5, &["combat", "public"]),
    ]
}

/// Hidden classes — granted when unwitnessed heroic acts are high.
fn hidden_class_templates() -> Vec<ClassTemplate> {
    vec![
        ct("Ghost", &[("stealth_actions", 0.4), ("melee_combat", 0.3)], 0.5, &["stealth", "hidden"]),
        ct("The Unseen Hand", &[("stealth_actions", 0.3), ("trades_completed", 0.3)], 0.5, &["stealth", "hidden"]),
        ct("Phantom", &[("stealth_actions", 0.3), ("areas_explored", 0.3)], 0.5, &["stealth", "hidden"]),
        ct("Silent Guardian", &[("damage_absorbed", 0.3), ("allies_supported", 0.3)], 0.5, &["defense", "hidden"]),
        ct("Nameless Savior", &[("healing_given", 0.4), ("stealth_actions", 0.2)], 0.5, &["support", "hidden"]),
    ]
}

fn ct(name: &str, weights: &[(&str, f32)], threshold: f32, tags: &[&str]) -> ClassTemplate {
    ClassTemplate {
        class_name: name.to_string(),
        behavior_weights: weights.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        threshold,
        tags: tags.iter().map(|s| s.to_string()).collect(),
        rarity: SkillRarity::Common,
    }
}

fn ct_rare(name: &str, weights: &[(&str, f32)], threshold: f32, tags: &[&str], rarity: SkillRarity) -> ClassTemplate {
    ClassTemplate {
        class_name: name.to_string(),
        behavior_weights: weights.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        threshold,
        tags: tags.iter().map(|s| s.to_string()).collect(),
        rarity,
    }
}

// ---------------------------------------------------------------------------
// Helpers for grammar-walked ability generation
// ---------------------------------------------------------------------------

/// Map class names to ability_gen archetype names for procedural generation.
/// For novel "Wandering X-Y" classes, parse the behavior axes from the name.
fn class_to_archetype(class_name: &str) -> String {
    // Known class → archetype mapping
    match class_name {
        "Warrior" | "Militia" | "Laborer" | "Farmhand" => return "knight".to_string(),
        "Ranger" | "Hunter" | "Scout" | "Stablehand" => return "ranger".to_string(),
        "Healer" | "Herbalist" => return "cleric".to_string(),
        "Scholar" | "Scribe" | "Apprentice" => return "mage".to_string(),
        "Rogue" | "Pickpocket" => return "rogue".to_string(),
        "Commander" | "Errand Runner" => return "paladin".to_string(),
        "Diplomat" | "Peddler" | "Merchant" | "Traveler" => return "bard".to_string(),
        "Artisan" => return "artificer".to_string(),
        "Guardian" => return "guardian".to_string(),
        _ => {}
    }
    // For novel classes like "Wandering Combat-Exploration", parse the
    // primary behavior axis from the name and map to matching archetype
    let lower = class_name.to_lowercase();
    if lower.contains("combat") || lower.contains("melee") { return "knight".to_string(); }
    if lower.contains("ranged") || lower.contains("exploration") { return "ranger".to_string(); }
    if lower.contains("healing") || lower.contains("support") { return "cleric".to_string(); }
    if lower.contains("research") || lower.contains("scholar") { return "mage".to_string(); }
    if lower.contains("stealth") { return "rogue".to_string(); }
    if lower.contains("command") || lower.contains("leader") { return "paladin".to_string(); }
    if lower.contains("trade") || lower.contains("diplomacy") { return "bard".to_string(); }
    if lower.contains("craft") { return "artificer".to_string(); }
    if lower.contains("defense") || lower.contains("guardian") { return "guardian".to_string(); }
    // True default — use the most common archetype
    "knight".to_string()
}

impl BehaviorLedger {
    /// Convert behavior counters to history tags for ability generation biasing.
    pub fn to_history_tags(&self) -> std::collections::HashMap<String, u32> {
        let mut tags = std::collections::HashMap::new();
        if self.melee_combat > 10.0 { tags.insert("melee".to_string(), self.melee_combat as u32); }
        if self.ranged_combat > 10.0 { tags.insert("ranged".to_string(), self.ranged_combat as u32); }
        if self.healing_given > 10.0 { tags.insert("healing".to_string(), self.healing_given as u32); }
        if self.diplomacy_actions > 10.0 { tags.insert("diplomatic".to_string(), self.diplomacy_actions as u32); }
        if self.trades_completed > 10.0 { tags.insert("trade".to_string(), self.trades_completed as u32); }
        if self.items_crafted > 10.0 { tags.insert("crafting".to_string(), self.items_crafted as u32); }
        if self.areas_explored > 10.0 { tags.insert("exploration".to_string(), self.areas_explored as u32); }
        if self.units_commanded > 10.0 { tags.insert("leadership".to_string(), self.units_commanded as u32); }
        if self.stealth_actions > 10.0 { tags.insert("stealth".to_string(), self.stealth_actions as u32); }
        if self.research_performed > 10.0 { tags.insert("scholarly".to_string(), self.research_performed as u32); }
        if self.damage_absorbed > 10.0 { tags.insert("tanking".to_string(), self.damage_absorbed as u32); }
        if self.allies_supported > 10.0 { tags.insert("support".to_string(), self.allies_supported as u32); }
        tags
    }
}

// ---------------------------------------------------------------------------
// Main tick
// ---------------------------------------------------------------------------

pub fn tick_class_system(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CLASS_TICK_INTERVAL != 0 {
        return;
    }

    update_behavior_ledgers(state, events);
    track_landmark_achievements(state, events);
    check_class_acquisition(state, events);
    process_class_xp(state, events);
    check_capstone_resolution(state, events);
    check_skill_grants(state, events);
    update_stagnation(state, events);
    decay_recent_window(state);
    decay_exclusion_cooldowns(state);
    check_hybrid_unlock(state, events);

    // Consolidation & evolution run on a slower cadence
    if state.tick % CONSOLIDATION_INTERVAL == 0 {
        check_consolidation_offers(state, events);
        check_evolution(state, events);
        check_world_gated_classes(state, events);
    }

    // Phase 7: Reactive narrative runs every 200 ticks
    if state.tick % REACTIVE_NARRATIVE_INTERVAL == 0 {
        check_shame_classes(state, events);
        check_crisis_grants(state, events);
        check_identity_erosion(state, events);
        check_crisis_escape_valve(state, events);
        check_campaign_skill_hooks(state, events);
        generate_class_chronicle(state, events);
    }

    // Mirror offers & oath-locked classes every 300 ticks
    if state.tick % 300 == 0 {
        check_mirror_offers(state, events);
        check_oath_locked_classes(state, events);
        track_heroic_acts(state, events);
    }

    // Rival-reflected & folk hero divergence every 500 ticks
    if state.tick % 500 == 0 {
        check_rival_classes(state, events);
        check_folk_hero_divergence(state, events);
    }
}

// ---------------------------------------------------------------------------
// Behavior ledger update
// ---------------------------------------------------------------------------

/// Scan recent WorldEvents and increment appropriate behavior counters.
fn update_behavior_ledgers(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Seed behavior from archetype if adventurer has zero accumulated behavior.
    // This ensures classes can develop early rather than requiring hundreds of ticks.
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead { continue; }
        let total = adv.behavior_ledger.melee_combat + adv.behavior_ledger.ranged_combat
            + adv.behavior_ledger.healing_given + adv.behavior_ledger.diplomacy_actions
            + adv.behavior_ledger.trades_completed + adv.behavior_ledger.items_crafted
            + adv.behavior_ledger.areas_explored + adv.behavior_ledger.units_commanded
            + adv.behavior_ledger.stealth_actions + adv.behavior_ledger.research_performed
            + adv.behavior_ledger.damage_absorbed + adv.behavior_ledger.allies_supported;
        if total < 1.0 {
            // Seed based on archetype — gives a head start toward the matching class
            let seed_amount = 10.0;
            match adv.archetype.as_str() {
                "knight" | "warrior" | "berserker" => {
                    adv.behavior_ledger.melee_combat += seed_amount;
                    adv.behavior_ledger.damage_absorbed += seed_amount * 0.5;
                }
                "ranger" | "archer" | "hunter" => {
                    adv.behavior_ledger.ranged_combat += seed_amount;
                    adv.behavior_ledger.areas_explored += seed_amount * 0.5;
                }
                "mage" | "wizard" | "sorcerer" | "warlock" => {
                    adv.behavior_ledger.research_performed += seed_amount;
                    adv.behavior_ledger.ranged_combat += seed_amount * 0.3;
                }
                "cleric" | "priest" | "healer" | "druid" => {
                    adv.behavior_ledger.healing_given += seed_amount;
                    adv.behavior_ledger.allies_supported += seed_amount * 0.5;
                }
                "rogue" | "thief" | "assassin" => {
                    adv.behavior_ledger.stealth_actions += seed_amount;
                    adv.behavior_ledger.melee_combat += seed_amount * 0.3;
                }
                "paladin" | "guardian" | "tank" => {
                    adv.behavior_ledger.damage_absorbed += seed_amount;
                    adv.behavior_ledger.allies_supported += seed_amount * 0.5;
                }
                "bard" | "diplomat" => {
                    adv.behavior_ledger.diplomacy_actions += seed_amount;
                    adv.behavior_ledger.allies_supported += seed_amount * 0.5;
                }
                "monk" => {
                    adv.behavior_ledger.melee_combat += seed_amount * 0.7;
                    adv.behavior_ledger.damage_absorbed += seed_amount * 0.3;
                }
                "necromancer" | "warlock" => {
                    adv.behavior_ledger.research_performed += seed_amount;
                    adv.behavior_ledger.stealth_actions += seed_amount * 0.3;
                }
                "shaman" => {
                    adv.behavior_ledger.healing_given += seed_amount * 0.5;
                    adv.behavior_ledger.research_performed += seed_amount * 0.3;
                    adv.behavior_ledger.allies_supported += seed_amount * 0.2;
                }
                "artificer" => {
                    adv.behavior_ledger.items_crafted += seed_amount;
                    adv.behavior_ledger.research_performed += seed_amount * 0.3;
                }
                _ => {
                    // Unknown archetype: spread across non-combat axes for diversity
                    adv.behavior_ledger.diplomacy_actions += seed_amount * 0.3;
                    adv.behavior_ledger.trades_completed += seed_amount * 0.3;
                    adv.behavior_ledger.areas_explored += seed_amount * 0.2;
                    adv.behavior_ledger.allies_supported += seed_amount * 0.2;
                }
            }
            // Also seed recent window so the first class check can fire
            adv.behavior_ledger.recent_melee_combat = adv.behavior_ledger.melee_combat;
            adv.behavior_ledger.recent_ranged_combat = adv.behavior_ledger.ranged_combat;
            adv.behavior_ledger.recent_healing_given = adv.behavior_ledger.healing_given;
            adv.behavior_ledger.recent_diplomacy_actions = adv.behavior_ledger.diplomacy_actions;
            adv.behavior_ledger.recent_trades_completed = adv.behavior_ledger.trades_completed;
            adv.behavior_ledger.recent_items_crafted = adv.behavior_ledger.items_crafted;
            adv.behavior_ledger.recent_areas_explored = adv.behavior_ledger.areas_explored;
            adv.behavior_ledger.recent_units_commanded = adv.behavior_ledger.units_commanded;
            adv.behavior_ledger.recent_stealth_actions = adv.behavior_ledger.stealth_actions;
            adv.behavior_ledger.recent_research_performed = adv.behavior_ledger.research_performed;
            adv.behavior_ledger.recent_damage_absorbed = adv.behavior_ledger.damage_absorbed;
            adv.behavior_ledger.recent_allies_supported = adv.behavior_ledger.allies_supported;
        }
    }

    // Build per-adventurer increments from the events generated this tick.
    // We scan the events vec which contains events from this tick's earlier systems.
    for ev in events.iter() {
        match ev {
            // Battle events
            WorldEvent::BattleStarted { .. } | WorldEvent::BattleEnded { .. } => {
                // Credit all non-dead adventurers in parties with melee/ranged
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead {
                        continue;
                    }
                    if adv.status == AdventurerStatus::Fighting {
                        let amt = 1.0;
                        if adv.archetype == "ranger" || adv.archetype == "marksman" || adv.archetype == "sniper" {
                            adv.behavior_ledger.ranged_combat += amt;
                            adv.behavior_ledger.recent_ranged_combat += amt;
                        } else {
                            adv.behavior_ledger.melee_combat += amt;
                            adv.behavior_ledger.recent_melee_combat += amt;
                        }
                    }
                }
            }
            // Healing — credit idle adventurers (they're tending the wounded at the guild hall)
            // and the recovered adventurer themselves (self-care counts)
            WorldEvent::AdventurerRecovered { adventurer_id } => {
                // The recovered adventurer gets some healing_given (self-care)
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adventurer_id) {
                    adv.behavior_ledger.healing_given += 0.5;
                    adv.behavior_ledger.recent_healing_given += 0.5;
                }
                // Idle adventurers at the guild are helping with recovery (passive, low credit)
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead { continue; }
                    if adv.status == AdventurerStatus::Idle && adv.id != *adventurer_id {
                        adv.behavior_ledger.healing_given += 0.3;
                        adv.behavior_ledger.recent_healing_given += 0.3;
                        adv.behavior_ledger.allies_supported += 0.1;
                        adv.behavior_ledger.recent_allies_supported += 0.1;
                    }
                }
            }
            // Diplomacy — guild-wide events credit idle adventurers (they're at the guild
            // hall facilitating). A warrior who spends downtime negotiating should earn [Peddler].
            WorldEvent::AgreementFormed { .. }
            | WorldEvent::WarCeasefire { .. }
            | WorldEvent::FactionRelationChanged { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead { continue; }
                    if adv.status == AdventurerStatus::Idle {
                        // Passive credit — low enough not to swamp primary behavior
                        adv.behavior_ledger.diplomacy_actions += 0.5;
                        adv.behavior_ledger.recent_diplomacy_actions += 0.5;
                    }
                }
            }
            // Trade/economy — idle adventurers get small passive credit.
            WorldEvent::CaravanCompleted { .. }
            | WorldEvent::TradeProfitMade { .. }
            | WorldEvent::MerchantPurchase { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead { continue; }
                    if adv.status == AdventurerStatus::Idle {
                        adv.behavior_ledger.trades_completed += 0.5;
                        adv.behavior_ledger.recent_trades_completed += 0.5;
                    }
                }
            }
            // Quest completion — active adventurers get trade credit for loot sale
            WorldEvent::QuestCompleted { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead { continue; }
                    if adv.status == AdventurerStatus::Fighting || adv.status == AdventurerStatus::OnMission {
                        adv.behavior_ledger.trades_completed += 0.3;
                        adv.behavior_ledger.recent_trades_completed += 0.3;
                        adv.behavior_ledger.allies_supported += 0.2;
                        adv.behavior_ledger.recent_allies_supported += 0.2;
                    }
                }
            }
            // Crafting — idle adventurers get small passive credit
            WorldEvent::ItemCrafted { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead { continue; }
                    if adv.status == AdventurerStatus::Idle {
                        adv.behavior_ledger.items_crafted += 0.3;
                        adv.behavior_ledger.recent_items_crafted += 0.3;
                    }
                }
            }
            // Exploration
            WorldEvent::TileExplored { .. }
            | WorldEvent::ExplorationMilestone { .. }
            | WorldEvent::LandmarkDiscovered { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead || adv.status == AdventurerStatus::Idle {
                        continue;
                    }
                    adv.behavior_ledger.areas_explored += 0.5;
                    adv.behavior_ledger.recent_areas_explored += 0.5;
                }
            }
            // Leadership/command
            WorldEvent::LeaderAppointed { adventurer_id, .. } => {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adventurer_id) {
                    adv.behavior_ledger.units_commanded += 2.0;
                    adv.behavior_ledger.recent_units_commanded += 2.0;
                }
            }
            WorldEvent::PartyFormed { member_ids, .. } => {
                // Credit the first member as the leader/commander
                if let Some(&leader_id) = member_ids.first() {
                    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == leader_id) {
                        adv.behavior_ledger.units_commanded += 3.0;
                        adv.behavior_ledger.recent_units_commanded += 3.0;
                    }
                }
            }
            // Stealth/espionage
            WorldEvent::IntelGathered { spy_id, .. } => {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *spy_id) {
                    adv.behavior_ledger.stealth_actions += 3.0;
                    adv.behavior_ledger.recent_stealth_actions += 3.0;
                }
            }
            WorldEvent::HeistSucceeded { .. } | WorldEvent::HeistPhaseAdvanced { .. } => {
                // Everyone involved in a heist gets stealth credit
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead { continue; }
                    if adv.status != AdventurerStatus::Idle {
                        adv.behavior_ledger.stealth_actions += 2.0;
                        adv.behavior_ledger.recent_stealth_actions += 2.0;
                    }
                }
            }
            // Research/archives — idle adventurers at the guild study and learn
            WorldEvent::ResearchCompleted { .. } | WorldEvent::KnowledgeGained { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead { continue; }
                    if adv.status == AdventurerStatus::Idle {
                        adv.behavior_ledger.research_performed += 0.3;
                        adv.behavior_ledger.recent_research_performed += 0.3;
                    }
                }
            }
            // Damage absorbed
            WorldEvent::AdventurerInjured { adventurer_id, injury_level } => {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adventurer_id) {
                    adv.behavior_ledger.damage_absorbed += injury_level * 0.1;
                    adv.behavior_ledger.recent_damage_absorbed += injury_level * 0.1;
                }
            }
            // Support/buff
            WorldEvent::MentorshipCompleted { mentor_id, .. } => {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *mentor_id) {
                    adv.behavior_ledger.allies_supported += 3.0;
                    adv.behavior_ledger.recent_allies_supported += 3.0;
                }
            }
            WorldEvent::SkillTransferred { from_id, .. } => {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *from_id) {
                    adv.behavior_ledger.allies_supported += 0.5;
                    adv.behavior_ledger.recent_allies_supported += 0.5;
                }
            }
            _ => {}
        }
    }

    // --- Mentorship Observation Credit (idea 1.7) ---
    let mut party_groups: std::collections::HashMap<u32, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead { continue; }
        if let Some(pid) = adv.party_id {
            party_groups.entry(pid).or_default().push(idx);
        }
    }
    struct ObsCredit { observer_idx: usize, mentor_id: u32, domain: String, credits: Vec<(usize, f32)> }
    let mut obs_credits: Vec<ObsCredit> = Vec::new();
    let all_templates: Vec<ClassTemplate> = base_templates().into_iter().chain(rare_templates()).collect();
    for (_pid, members) in &party_groups {
        if members.len() < 2 { continue; }
        for &a_idx in members {
            for &b_idx in members {
                if a_idx == b_idx { continue; }
                let adv_a = &state.adventurers[a_idx];
                let adv_b = &state.adventurers[b_idx];
                for b_class in &adv_b.classes {
                    let a_level = adv_a.classes.iter()
                        .find(|c| c.class_name == b_class.class_name)
                        .map(|c| c.level).unwrap_or(0);
                    if b_class.level > a_level {
                        if let Some(tmpl) = all_templates.iter().find(|t| t.class_name == b_class.class_name) {
                            let mut credits = Vec::new();
                            for (field, _weight) in &tmpl.behavior_weights {
                                if let Some(fi) = field_index(field) {
                                    let b_recent = get_recent_field(&adv_b.behavior_ledger, fi);
                                    let amount = b_recent * MENTORSHIP_OBSERVATION_FRACTION;
                                    if amount > 0.001 { credits.push((fi, amount)); }
                                }
                            }
                            if !credits.is_empty() {
                                obs_credits.push(ObsCredit {
                                    observer_idx: a_idx, mentor_id: adv_b.id,
                                    domain: b_class.class_name.clone(), credits,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    for oc in obs_credits {
        let _observer_id = state.adventurers[oc.observer_idx].id;
        for (fi, amount) in &oc.credits {
            add_behavior_field(&mut state.adventurers[oc.observer_idx].behavior_ledger, *fi, *amount);
        }
        // Note: MentorshipObservationCredit events are emitted separately
        let _ = (oc.mentor_id, oc.domain); // observation credit applied silently
    }
}

// ---------------------------------------------------------------------------
// Behavioral fingerprint
// ---------------------------------------------------------------------------

/// Build a normalized 12-dim behavioral fingerprint from the ledger.
/// Uses lifetime + 3x recent weighting.
fn behavioral_fingerprint(ledger: &BehaviorLedger) -> [f32; 12] {
    let raw = [
        ledger.melee_combat + 3.0 * ledger.recent_melee_combat,
        ledger.ranged_combat + 3.0 * ledger.recent_ranged_combat,
        ledger.healing_given + 3.0 * ledger.recent_healing_given,
        ledger.diplomacy_actions + 3.0 * ledger.recent_diplomacy_actions,
        ledger.trades_completed + 3.0 * ledger.recent_trades_completed,
        ledger.items_crafted + 3.0 * ledger.recent_items_crafted,
        ledger.areas_explored + 3.0 * ledger.recent_areas_explored,
        ledger.units_commanded + 3.0 * ledger.recent_units_commanded,
        ledger.stealth_actions + 3.0 * ledger.recent_stealth_actions,
        ledger.research_performed + 3.0 * ledger.recent_research_performed,
        ledger.damage_absorbed + 3.0 * ledger.recent_damage_absorbed,
        ledger.allies_supported + 3.0 * ledger.recent_allies_supported,
    ];
    let sum: f32 = raw.iter().sum();
    if sum < 1e-6 {
        return [0.0; 12];
    }
    let mut fp = [0.0f32; 12];
    for (i, &v) in raw.iter().enumerate() {
        fp[i] = v / sum;
    }
    fp
}

/// Map a behavior field name to its fingerprint index.
fn field_index(name: &str) -> Option<usize> {
    match name {
        "melee_combat" => Some(0),
        "ranged_combat" => Some(1),
        "healing_given" => Some(2),
        "diplomacy_actions" => Some(3),
        "trades_completed" => Some(4),
        "items_crafted" => Some(5),
        "areas_explored" => Some(6),
        "units_commanded" => Some(7),
        "stealth_actions" => Some(8),
        "research_performed" => Some(9),
        "damage_absorbed" => Some(10),
        "allies_supported" => Some(11),
        _ => None,
    }
}

/// Score a fingerprint against a class template.
fn score_template(fp: &[f32; 12], template: &ClassTemplate) -> f32 {
    let mut score = 0.0f32;
    for (field, weight) in &template.behavior_weights {
        if let Some(idx) = field_index(field) {
            score += fp[idx] * weight;
        }
    }
    score
}

// ---------------------------------------------------------------------------
// Starter class evolution mapping
// ---------------------------------------------------------------------------

/// Returns the base class a starter class evolves into, if any.
fn starter_evolves_to(starter: &str) -> Option<&'static str> {
    match starter {
        "Laborer" => Some("Warrior"),
        "Hunter" => Some("Ranger"),
        "Militia" => Some("Warrior"),
        "Peddler" => Some("Merchant"),
        "Herbalist" => Some("Healer"),
        "Scribe" => Some("Scholar"),
        "Pickpocket" => Some("Rogue"),
        "Apprentice" => Some("Artisan"),
        "Farmhand" => Some("Guardian"),
        "Traveler" => Some("Scout"),
        "Errand Runner" => Some("Commander"),
        "Stablehand" => Some("Ranger"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Class acquisition
// ---------------------------------------------------------------------------

fn check_class_acquisition(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let templates: Vec<ClassTemplate> = starter_templates()
        .into_iter()
        .chain(base_templates())
        .chain(rare_templates())
        .chain(public_class_templates())
        .chain(hidden_class_templates())
        .collect();

    let tick = state.tick as u32;
    let in_crisis = !state.overworld.active_crises.is_empty();

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let fp = behavioral_fingerprint(&adv.behavior_ledger);
        // Skip if no meaningful behavior yet
        let total: f32 = fp.iter().sum();
        if total < 0.01 {
            continue;
        }

        let mut any_template_matched = false;
        let mut best_similarity: f32 = 0.0;

        for tmpl in &templates {
            // Warlord only during crisis
            if tmpl.class_name == "Warlord" && !in_crisis {
                continue;
            }
            // Already have this class?
            if adv.classes.iter().any(|c| c.class_name == tmpl.class_name) {
                continue;
            }
            // Rejected via mirror offer?
            if adv.behavior_ledger.rejected_classes.iter().any(|r| r == &tmpl.class_name) {
                continue;
            }
            // Public classes require witnessed heroic acts >= 3
            if tmpl.tags.iter().any(|t| t == "public")
                && adv.behavior_ledger.witnessed_heroic_acts < 3
            {
                continue;
            }
            // Hidden classes require unwitnessed heroic acts >= 3
            if tmpl.tags.iter().any(|t| t == "hidden")
                && adv.behavior_ledger.unwitnessed_heroic_acts < 3
            {
                continue;
            }
            // Build template fingerprint for cosine similarity (idea 1.9)
            let mut tmpl_fp = [0.0f32; 12];
            for (field, weight) in &tmpl.behavior_weights {
                if let Some(idx) = field_index(field) {
                    tmpl_fp[idx] = *weight;
                }
            }
            let sim = cosine_similarity(&fp, &tmpl_fp);
            if sim > best_similarity {
                best_similarity = sim;
            }

            let s = score_template(&fp, tmpl);
            if s >= tmpl.threshold {
                any_template_matched = true;

                // --- Exclusion Zone Check (idea 1.3) ---
                if is_excluded_by_cooldown(&adv.classes, &tmpl.class_name) {
                    continue;
                }

                // --- Landmark Prerequisite Check (idea 1.10) ---
                let required_landmarks = landmark_requirements(&tmpl.class_name);
                if !required_landmarks.is_empty() {
                    let has_all = required_landmarks.iter().all(|lm| {
                        adv.behavior_ledger.landmark_achievements.iter().any(|a| a == lm)
                    });
                    if !has_all {
                        continue;
                    }
                }

                // --- Starter class evolution: if a starter class evolves to this
                // base class, replace it rather than stacking ---
                let evolved_from: Option<(String, f32, Vec<GrantedSkill>)> = {
                    let mut found = None;
                    for existing in adv.classes.iter() {
                        if let Some(target) = starter_evolves_to(&existing.class_name) {
                            if target == tmpl.class_name {
                                // Transfer XP at 50% rate, keep skills
                                found = Some((
                                    existing.class_name.clone(),
                                    existing.xp * 0.5,
                                    existing.skills_granted.clone(),
                                ));
                                break;
                            }
                        }
                    }
                    found
                };

                if let Some((from_class, transferred_xp, old_skills)) = evolved_from {
                    // Remove the starter class
                    adv.classes.retain(|c| c.class_name != from_class);

                    let mut ci = ClassInstance {
                        class_name: tmpl.class_name.clone(),
                        level: 1,
                        xp: transferred_xp,
                        xp_to_next: 15.0,
                        stagnation_ticks: 0,
                        skills_granted: old_skills,
                        acquired_tick: tick,
                        identity_coherence: 1.0,
                        exclusion_cooldown: 1 * 100,
                        ..Default::default()
                    };
                    populate_noncombat_growth(&mut ci);
                    adv.classes.push(ci);

                    let chronicle = format!(
                        "The [{}] fought through enough battles. They are [{}] now.",
                        from_class, tmpl.class_name
                    );
                    events.push(WorldEvent::ClassEvolutionFromStarter {
                        adventurer_id: adv.id,
                        from_class,
                        to_class: tmpl.class_name.clone(),
                    });
                    events.push(WorldEvent::ClassChronicleEntry {
                        adventurer_id: adv.id,
                        entry: chronicle,
                    });
                } else {
                    let mut ci = ClassInstance {
                        class_name: tmpl.class_name.clone(),
                        level: 1,
                        xp: 0.0,
                        xp_to_next: 15.0, // 1^2 * 15
                        stagnation_ticks: 0,
                        skills_granted: Vec::new(),
                        acquired_tick: tick,
                        identity_coherence: 1.0,
                        exclusion_cooldown: 1 * 100, // level * 100 ticks
                        ..Default::default()
                    };
                    populate_noncombat_growth(&mut ci);
                    adv.classes.push(ci);
                    eprintln!("[CLASS] Class granted: [{}] to adv {} (archetype: {})", tmpl.class_name, adv.id, adv.archetype);
                    events.push(WorldEvent::ClassGranted {
                        adventurer_id: adv.id,
                        class_name: tmpl.class_name.clone(),
                    });
                }
            }
        }

        // --- Cross-System Behavioral Fingerprinting for Unique Classes (idea 1.9) ---
        if !any_template_matched && best_similarity < UNIQUE_CLASS_SIM_THRESHOLD {
            let behavior_sum = adv.behavior_ledger.melee_combat
                + adv.behavior_ledger.ranged_combat + adv.behavior_ledger.healing_given
                + adv.behavior_ledger.diplomacy_actions + adv.behavior_ledger.trades_completed
                + adv.behavior_ledger.items_crafted + adv.behavior_ledger.areas_explored
                + adv.behavior_ledger.units_commanded + adv.behavior_ledger.stealth_actions
                + adv.behavior_ledger.research_performed + adv.behavior_ledger.damage_absorbed
                + adv.behavior_ledger.allies_supported;
            if behavior_sum >= UNIQUE_CLASS_MIN_BEHAVIOR {
                let already_unique = adv.classes.iter().any(|c| c.class_name.contains('-'));
                if !already_unique {
                    let mut indexed: Vec<(usize, f32)> =
                        fp.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let top3: Vec<&str> = indexed.iter().take(3)
                        .filter(|(_, v)| *v > 0.01)
                        .map(|(i, _)| BEHAVIOR_DIM_NAMES[*i]).collect();
                    let unique_name = if top3.len() >= 2 {
                        format!("Wandering {}-{}", top3[0], top3[1])
                    } else if !top3.is_empty() {
                        format!("Wandering {}", top3[0])
                    } else { "Wandering Enigma".to_string() };
                    adv.classes.push(ClassInstance {
                        class_name: unique_name.clone(), level: 1, xp: 0.0,
                        xp_to_next: 15.0, stagnation_ticks: 0, skills_granted: Vec::new(),
                        acquired_tick: tick, identity_coherence: 1.0,
                        exclusion_cooldown: 0, ..Default::default()
                    });
                    events.push(WorldEvent::UniqueClassGenerated {
                        adventurer_id: adv.id, class_name: unique_name,
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// XP processing
// ---------------------------------------------------------------------------

fn process_class_xp(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let templates: Vec<ClassTemplate> = starter_templates()
        .into_iter()
        .chain(base_templates())
        .chain(rare_templates())
        .chain(public_class_templates())
        .chain(hidden_class_templates())
        .collect();

    // --- Witness XP Multiplier pre-pass (idea 2.9) ---
    let witness_mults: Vec<Vec<f32>> = (0..state.adventurers.len())
        .map(|adv_idx| {
            state.adventurers[adv_idx]
                .classes
                .iter()
                .map(|c| witness_xp_multiplier(state, adv_idx, &c.class_name))
                .collect()
        })
        .collect();

    for (adv_idx, adv) in state.adventurers.iter_mut().enumerate() {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let fp = behavioral_fingerprint(&adv.behavior_ledger);

        // Collect XP per class based on behavior match
        let num_classes = adv.classes.len();
        let mut xp_gains = vec![0.0f32; num_classes];

        for (i, class) in adv.classes.iter().enumerate() {
            // Find template for XP scoring
            let tmpl = templates.iter().find(|t| t.class_name == class.class_name);
            let raw_xp = if let Some(tmpl) = tmpl {
                score_template(&fp, tmpl) * 50.0
            } else {
                // For unique/generated classes without templates, use the total
                // behavior magnitude as XP source (they level from ANY activity)
                let total_behavior: f32 = fp.iter().sum();
                total_behavior * 5.0
            };
            // Stagnation penalty disabled — was causing death spiral where
            // classes froze at level 1 and never progressed. Revisit once
            // behavior event frequency is high enough to sustain XP flow.
            let penalty = 1.0;
            let wmult = witness_mults.get(adv_idx)
                .and_then(|v| v.get(i))
                .copied()
                .unwrap_or(1.0);
            xp_gains[i] = raw_xp * penalty * wmult;
        }

        // Class resonance: trickle XP between classes with overlapping tags
        let mut trickle = vec![0.0f32; num_classes];
        for i in 0..num_classes {
            for j in 0..num_classes {
                if i == j {
                    continue;
                }
                let ti = templates.iter().find(|t| t.class_name == adv.classes[i].class_name);
                let tj = templates.iter().find(|t| t.class_name == adv.classes[j].class_name);
                if let (Some(ti), Some(tj)) = (ti, tj) {
                    let overlap = ti.tags.iter().any(|tag| tj.tags.contains(tag));
                    if overlap {
                        trickle[i] += xp_gains[j] * RESONANCE_TRICKLE;
                    }
                }
            }
        }

        // Track which classes gained XP this tick for co-active tracking (idea 2.7)
        let mut xp_gaining_classes: Vec<String> = Vec::new();

        // Apply XP and check level-ups
        for (i, class) in adv.classes.iter_mut().enumerate() {
            let gained = xp_gains[i] + trickle[i];
            if gained > 0.0 {
                class.stagnation_ticks = 0;
                xp_gaining_classes.push(class.class_name.clone());

                // Capstone gate (idea 2.5): above level 20, require capstone event
                if class.level >= 20 && class.capstone_required {
                    let cap = class.xp_to_next * 0.5;
                    class.xp_overflow = (class.xp_overflow + gained).min(cap);
                    continue;
                }

                class.xp += gained;

                // Level up loop
                while class.xp >= class.xp_to_next {
                    class.xp -= class.xp_to_next;
                    class.level += 1;
                    class.xp_to_next = (class.level * class.level) as f32 * 15.0;
                    events.push(WorldEvent::ClassLevelUp {
                        adventurer_id: adv.id,
                        class_name: class.class_name.clone(),
                        new_level: class.level,
                    });

                    // Set capstone_required for levels past 20
                    if class.level >= 20 {
                        class.capstone_required = true;
                    }
                }
            }
        }

        // Co-active tick tracking (idea 2.7): increment pair counters
        for i in 0..xp_gaining_classes.len() {
            for j in (i + 1)..xp_gaining_classes.len() {
                let a = &xp_gaining_classes[i];
                let b = &xp_gaining_classes[j];
                let key = if a <= b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                *adv.behavior_ledger.co_active_ticks.entry(key).or_insert(0) += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Capstone resolution (idea 2.5)
// ---------------------------------------------------------------------------

/// Capstone condition per class archetype.
fn capstone_class_type(class_name: &str) -> &'static str {
    match class_name {
        "Warrior" | "Ranger" | "Rogue" | "Guardian" | "Scout" | "Spellblade" => "combat",
        "Healer" | "Plague Doctor" => "healer",
        "Scholar" => "scholar",
        "Merchant" | "Shadowmerchant" | "Artisan" => "merchant",
        "Commander" | "Diplomat" | "Warlord" => "commander",
        _ => "combat",
    }
}

/// Check if capstone conditions are met and resolve overflow XP.
fn check_capstone_resolution(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        let ledger = &adv.behavior_ledger;

        for class in &mut adv.classes {
            if !class.capstone_required || class.level < 20 {
                continue;
            }

            let class_type = capstone_class_type(&class.class_name);
            let (resolved, challenge) = match class_type {
                "combat" => {
                    // Win a battle outnumbered 2:1
                    let score =
                        ledger.damage_absorbed + ledger.melee_combat + ledger.ranged_combat;
                    (score > 100.0, "Won battle outnumbered 2:1".to_string())
                }
                "healer" => (
                    ledger.healing_given > 500.0,
                    "Healed over 500 HP in one quest".to_string(),
                ),
                "scholar" => (
                    ledger.research_performed >= 5.0,
                    "Completed 5+ research actions in one cycle".to_string(),
                ),
                "merchant" => (
                    ledger.trades_completed >= 20.0,
                    "Accumulated 1000+ gold profit in one cycle".to_string(),
                ),
                "commander" => (
                    ledger.units_commanded >= 10.0,
                    "Led a party of 4+ to quest success".to_string(),
                ),
                _ => (false, String::new()),
            };

            if resolved {
                class.capstone_required = false;
                class.xp += class.xp_overflow;
                class.xp_overflow = 0.0;

                // Level up from overflow if enough XP
                while class.xp >= class.xp_to_next {
                    class.xp -= class.xp_to_next;
                    class.level += 1;
                    class.xp_to_next = (class.level * class.level) as f32 * 15.0;
                    events.push(WorldEvent::ClassLevelUp {
                        adventurer_id: adv.id,
                        class_name: class.class_name.clone(),
                        new_level: class.level,
                    });
                    if class.level >= 20 {
                        class.capstone_required = true;
                    }
                }

                // Grant a Capstone-rarity skill
                let capstone_skill_name = format!("{} Capstone", class.class_name);
                class.skills_granted.push(GrantedSkill {
                    skill_name: capstone_skill_name.clone(),
                    granted_at_level: class.level,
                    rarity: SkillRarity::Capstone,
                    from_class: class.class_name.clone(),
                    ability_dsl: None,
                    effect_description: format!(
                        "Capstone skill earned by completing: {}",
                        challenge
                    ),
                    affinity_tags: affinity_tags_for_class(&class.class_name),
                    empowered: false,
                    inheritance_status: String::new(),
                    skill_effect: None,
                    skill_condition: None,
                });

                events.push(WorldEvent::CapstoneResolved {
                    adventurer_id: adv.id,
                    class_name: class.class_name.clone(),
                    challenge: challenge.clone(),
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hybrid class unlock (idea 2.7)
// ---------------------------------------------------------------------------

/// Generate a hybrid class name from two parent classes.
fn generate_hybrid_name(a: &str, b: &str) -> String {
    match (a, b) {
        ("Warrior", "Scholar") | ("Scholar", "Warrior") => "Battlemage".to_string(),
        ("Warrior", "Rogue") | ("Rogue", "Warrior") => "Bladedancer".to_string(),
        ("Healer", "Scholar") | ("Scholar", "Healer") => "Theurgist".to_string(),
        ("Healer", "Guardian") | ("Guardian", "Healer") => "Templar".to_string(),
        ("Ranger", "Rogue") | ("Rogue", "Ranger") => "Shadowhunter".to_string(),
        ("Ranger", "Scholar") | ("Scholar", "Ranger") => "Naturalist".to_string(),
        ("Commander", "Warrior") | ("Warrior", "Commander") => "Warchief".to_string(),
        ("Commander", "Diplomat") | ("Diplomat", "Commander") => "Statesman".to_string(),
        ("Merchant", "Rogue") | ("Rogue", "Merchant") => "Smuggler".to_string(),
        ("Scout", "Ranger") | ("Ranger", "Scout") => "Trailblazer".to_string(),
        _ => {
            let pa: String = a.chars().take(4).collect();
            let pb: String = b.chars().take(4).collect();
            format!("{}{}", pa, pb.to_lowercase())
        }
    }
}

/// Minimum co-active ticks to unlock a hybrid class.
const HYBRID_CO_ACTIVE_THRESHOLD: u32 = 300;

/// Minimum level on both parent classes for hybrid unlock.
const HYBRID_MIN_LEVEL: u32 = 15;

/// Check if any class pairs qualify for hybrid unlock.
fn check_hybrid_unlock(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick as u32;

    struct HybridGrant {
        adv_id: u32,
        parent_a: String,
        parent_b: String,
        hybrid_name: String,
        inherited_skill_a: Option<GrantedSkill>,
        inherited_skill_b: Option<GrantedSkill>,
    }

    let mut grants: Vec<HybridGrant> = Vec::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        for (pair, &ticks) in &adv.behavior_ledger.co_active_ticks {
            if ticks < HYBRID_CO_ACTIVE_THRESHOLD {
                continue;
            }

            let (ref ca, ref cb) = *pair;

            let class_a = adv.classes.iter().find(|c| &c.class_name == ca);
            let class_b = adv.classes.iter().find(|c| &c.class_name == cb);

            let (class_a, class_b) = match (class_a, class_b) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };

            if class_a.level < HYBRID_MIN_LEVEL || class_b.level < HYBRID_MIN_LEVEL {
                continue;
            }

            let hybrid_name = generate_hybrid_name(ca, cb);

            if adv.classes.iter().any(|c| c.class_name == hybrid_name) {
                continue;
            }

            let inherited_a = class_a.skills_granted.first().cloned();
            let inherited_b = class_b.skills_granted.first().cloned();

            grants.push(HybridGrant {
                adv_id: adv.id,
                parent_a: ca.clone(),
                parent_b: cb.clone(),
                hybrid_name,
                inherited_skill_a: inherited_a,
                inherited_skill_b: inherited_b,
            });
        }
    }

    for grant in grants {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == grant.adv_id) {
            let mut inherited_skills = Vec::new();
            if let Some(s) = grant.inherited_skill_a {
                inherited_skills.push(s);
            }
            if let Some(s) = grant.inherited_skill_b {
                inherited_skills.push(s);
            }

            let mut ci = ClassInstance {
                class_name: grant.hybrid_name.clone(),
                level: 1,
                xp: 0.0,
                xp_to_next: 50.0, // 2x speed (halved XP requirements)
                stagnation_ticks: 0,
                skills_granted: inherited_skills,
                acquired_tick: tick,
                identity_coherence: 1.0,
                exclusion_cooldown: 0,
                capstone_required: false,
                xp_overflow: 0.0,
                suppressed_skills: Vec::new(),
                ..Default::default()
            };
            populate_noncombat_growth(&mut ci);
            adv.classes.push(ci);

            events.push(WorldEvent::HybridClassUnlocked {
                adventurer_id: grant.adv_id,
                parent_a: grant.parent_a,
                parent_b: grant.parent_b,
                hybrid_name: grant.hybrid_name,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Skill grants
// ---------------------------------------------------------------------------

fn check_skill_grants(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let templates: Vec<ClassTemplate> = base_templates()
        .into_iter()
        .chain(rare_templates())
        .collect();

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        // Collect all currently held skill names across all classes for empowerment check.
        let all_held_skills: Vec<(String, Vec<String>)> = adv
            .classes
            .iter()
            .flat_map(|c| {
                c.skills_granted
                    .iter()
                    .map(|s| (s.skill_name.clone(), s.affinity_tags.clone()))
            })
            .collect();

        // Feature 2 (idea 3.9): Build a cross-class contamination vector.
        let class_names: Vec<String> = adv.classes.iter().map(|c| c.class_name.clone()).collect();
        let adv_name = adv.name.clone();
        let adv_id = adv.id;
        let ledger_snapshot = adv.behavior_ledger.clone();

        for class_idx in 0..adv.classes.len() {
            let class = &mut adv.classes[class_idx];
            for &threshold in SKILL_THRESHOLDS {
                if class.level >= threshold
                    && !class
                        .skills_granted
                        .iter()
                        .any(|s| s.granted_at_level == threshold)
                {
                    let rarity = rarity_for_threshold(threshold);
                    let rarity_str = match &rarity {
                        SkillRarity::Common => "Common",
                        SkillRarity::Uncommon => "Uncommon",
                        SkillRarity::Rare => "Rare",
                        SkillRarity::Capstone => "Capstone",
                        SkillRarity::Unique => "Unique",
                    };

                    // Use the grammar walker to procedurally generate a unique ability
                    // based on the adventurer's class archetype, tier, and behavior history.
                    let tier = tier_for_threshold(threshold);
                    let history = ledger_snapshot.to_history_tags();
                    let archetype_for_gen = class_to_archetype(&class.class_name);
                    let (ability_def, dsl_text) = crate::headless_campaign::ability_gen::generate_tiered_ability(
                        &archetype_for_gen,
                        tier,
                        &mut state.rng,
                        &history,
                    );

                    // Name the skill from the generated ability
                    let skill_name = if !ability_def.name.is_empty() {
                        ability_def.name.clone()
                    } else {
                        format!("{} Lv{} Skill", class.class_name, threshold)
                    };
                    let effect = if !dsl_text.is_empty() {
                        dsl_text.clone()
                    } else {
                        format!("A tier {} ability from the {} class.", tier, class.class_name)
                    };
                    // Store the DSL text as the skill effect description;
                    // the actual combat effect is in ability_dsl
                    let skill_effect: Option<crate::headless_campaign::state::SkillEffect> = None;
                    let skill_condition: Option<crate::headless_campaign::state::SkillCondition> = None;
                    let ability_dsl_text = Some(dsl_text);

                    // Idea 3.6: skip if this skill is suppressed
                    if class.suppressed_skills.contains(&skill_name) {
                        continue;
                    }

                    // Idea 3.10: assign affinity tags based on class type
                    let mut affinity_tags = affinity_tags_for_class(&class.class_name);

                    // Feature 2 (idea 3.9): Cross-class skill contamination.
                    // Blend 25% of other classes' primary tags into the affinity set.
                    for (oci, other_name) in class_names.iter().enumerate() {
                        if oci == class_idx {
                            continue;
                        }
                        if let Some(tmpl) = templates.iter().find(|t| t.class_name == *other_name) {
                            let fp = behavioral_fingerprint(&ledger_snapshot);
                            let other_score = score_template(&fp, tmpl);
                            if other_score > 0.25 * tmpl.threshold {
                                for tag in affinity_tags_for_class(other_name) {
                                    if !affinity_tags.contains(&tag) {
                                        affinity_tags.push(tag);
                                    }
                                }
                            }
                        }
                    }

                    // Idea 3.10: check empowerment (2+ shared tags with held skills)
                    let empowered = {
                        let mut shared = 0usize;
                        for (_sname, stags) in &all_held_skills {
                            for tag in &affinity_tags {
                                if stags.contains(tag) {
                                    shared += 1;
                                }
                            }
                        }
                        shared >= 2
                    };

                    let display_name = if empowered {
                        format!("{} [Empowered]", skill_name)
                    } else {
                        skill_name.clone()
                    };

                    // Idea 3.6: check suppression table and apply suppressions
                    let suppressions = suppression_targets(&skill_name);
                    for suppressed in &suppressions {
                        if !class.suppressed_skills.contains(suppressed) {
                            class.suppressed_skills.push(suppressed.clone());
                        }
                    }

                    let rarity_clone = rarity.clone();
                    let class_name_clone = class.class_name.clone();

                    let has_effect = skill_effect.is_some();
                    let is_passive = match &skill_effect {
                        Some(ref e) => crate::headless_campaign::actions::is_passive_skill_effect(e),
                        None => false,
                    };
                    eprintln!(
                        "[CLASS] Skill granted: [{}] ({}) to adv {} from [{}] at lv{} | has_effect={} passive={}",
                        display_name, rarity_str, adv_id, class_name_clone, threshold, has_effect, is_passive
                    );

                    class.skills_granted.push(GrantedSkill {
                        skill_name: display_name.clone(),
                        granted_at_level: threshold,
                        rarity,
                        from_class: class.class_name.clone(),
                        ability_dsl: ability_dsl_text,
                        effect_description: effect,
                        affinity_tags,
                        empowered,
                        inheritance_status: String::new(),
                        skill_effect,
                        skill_condition,
                    });

                    events.push(WorldEvent::SkillGrantedByClass {
                        adventurer_id: adv_id,
                        skill_name: display_name.clone(),
                        rarity: rarity_str.to_string(),
                        class_name: class_name_clone.clone(),
                    });

                    // Feature 4 (idea 3.8): Narrative announcement system
                    match rarity_clone {
                        SkillRarity::Common => {
                            // Just the WorldEvent (already emitted above)
                        }
                        SkillRarity::Uncommon => {
                            let entry = format!(
                                "A new ability stirs within {}... the {} class deepens.",
                                adv_name, class_name_clone
                            );
                            state.class_chronicle_entries.push(entry.clone());
                            events.push(WorldEvent::ClassChronicleEntry {
                                adventurer_id: adv_id,
                                entry,
                            });
                        }
                        SkillRarity::Rare => {
                            let entry = format!(
                                "Power crystallizes. {} has earned {} — a rare manifestation \
                                 of the {} class. The system pauses to take note.",
                                adv_name, display_name, class_name_clone
                            );
                            state.class_chronicle_entries.push(entry.clone());
                            events.push(WorldEvent::ClassChronicleEntry {
                                adventurer_id: adv_id,
                                entry,
                            });
                        }
                        SkillRarity::Capstone => {
                            let entry = format!(
                                "We have been watching. {} has reached the apex. {} — \
                                 this is what they were always becoming. The {} class is complete.",
                                adv_name, display_name, class_name_clone
                            );
                            state.class_chronicle_entries.push(entry.clone());
                            events.push(WorldEvent::ClassChronicleEntry {
                                adventurer_id: adv_id,
                                entry,
                            });
                            events.push(WorldEvent::CapstoneSkillAnnounced {
                                adventurer_id: adv_id,
                                skill_name: display_name.clone(),
                                class_name: class_name_clone.clone(),
                            });
                        }
                        SkillRarity::Unique => {
                            let entry = format!(
                                "Something unprecedented. {} has manifested {}. \
                                 This skill has never existed before. It will never exist again. \
                                 The {} class has transcended its own definition.",
                                adv_name, display_name, class_name_clone
                            );
                            state.class_chronicle_entries.push(entry.clone());
                            events.push(WorldEvent::ClassChronicleEntry {
                                adventurer_id: adv_id,
                                entry,
                            });
                        }
                    }
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Skill suppression table (idea 3.6)
// ---------------------------------------------------------------------------

/// Returns skills that the given skill suppresses.
fn suppression_targets(skill_name: &str) -> Vec<String> {
    // Suppression rules: granting one skill suppresses another
    let table: &[(&str, &str)] = &[
        ("Precise Strike", "Wild Swing"),
        ("Iron Bulwark", "Reckless Charge"),
        ("Stealth Mastery", "Battle Cry"),
        ("Focused Healing", "Area Blast"),
    ];
    let mut targets = Vec::new();
    for &(granted, suppressed) in table {
        if skill_name.contains(granted) {
            targets.push(suppressed.to_string());
        }
    }
    targets
}

// ---------------------------------------------------------------------------
// Affinity tags for skill interaction bonuses (idea 3.10)
// ---------------------------------------------------------------------------

/// Assign affinity tags based on class type for skill interaction bonuses.
fn affinity_tags_for_class(class_name: &str) -> Vec<String> {
    let tags: &[&str] = match class_name {
        "Warrior" => &["offensive", "tactical"],
        "Ranger" => &["offensive", "tactical"],
        "Guardian" => &["defensive", "protection"],
        "Rogue" => &["offensive", "tactical"],
        "Commander" => &["leadership", "tactical"],
        "Healer" => &["healing", "protection"],
        "Diplomat" => &["leadership", "protection"],
        "Merchant" => &["leadership", "tactical"],
        "Scholar" => &["defensive", "tactical"],
        "Scout" => &["offensive", "tactical"],
        "Artisan" => &["defensive", "protection"],
        // Consolidated / rare classes
        "Spellblade" => &["offensive", "tactical"],
        "Plague Doctor" => &["healing", "tactical"],
        "Shadowmerchant" => &["offensive", "leadership"],
        "Warlord" => &["offensive", "leadership", "tactical"],
        _ => &["tactical"],
    };
    tags.iter().map(|s| s.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Stagnation
// ---------------------------------------------------------------------------

fn update_stagnation(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        for class in &mut adv.classes {
            // stagnation_ticks is reset to 0 in process_class_xp when XP is gained;
            // here we increment for classes that got no XP this tick, but slowly.
            // Only increment every 10th class tick to avoid death-spiral where
            // stagnation → no XP → more stagnation → frozen forever.
            if class.stagnation_ticks > 0 && state.tick % (CLASS_TICK_INTERVAL * 10) == 0 {
                class.stagnation_ticks += 1;
            }

            // Emit stagnation events at thresholds
            if class.stagnation_ticks == STAGNATION_HALF
                || class.stagnation_ticks == STAGNATION_FREEZE
            {
                events.push(WorldEvent::ClassStagnated {
                    adventurer_id: adv.id,
                    class_name: class.class_name.clone(),
                });
            }
        }
    }

    // Second pass: mark classes that gained no XP this tick for stagnation next time
    // (stagnation_ticks == 0 means XP was gained; set to 1 so next tick it increments)
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let fp = behavioral_fingerprint(&adv.behavior_ledger);
        let total: f32 = fp.iter().sum();
        if total < 0.01 {
            for class in &mut adv.classes {
                if class.stagnation_ticks == 0 {
                    class.stagnation_ticks = 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Recent window decay
// ---------------------------------------------------------------------------

fn decay_recent_window(state: &mut CampaignState) {
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let l = &mut adv.behavior_ledger;
        l.recent_melee_combat *= RECENT_DECAY;
        l.recent_ranged_combat *= RECENT_DECAY;
        l.recent_healing_given *= RECENT_DECAY;
        l.recent_diplomacy_actions *= RECENT_DECAY;
        l.recent_trades_completed *= RECENT_DECAY;
        l.recent_items_crafted *= RECENT_DECAY;
        l.recent_areas_explored *= RECENT_DECAY;
        l.recent_units_commanded *= RECENT_DECAY;
        l.recent_stealth_actions *= RECENT_DECAY;
        l.recent_research_performed *= RECENT_DECAY;
        l.recent_damage_absorbed *= RECENT_DECAY;
        l.recent_allies_supported *= RECENT_DECAY;
    }
}

// ---------------------------------------------------------------------------
// Consolidation
// ---------------------------------------------------------------------------

/// Generate a merged class name from two parent class names.
fn generate_consolidated_name(class_a: &str, class_b: &str) -> String {
    // Deterministic name generation from parent class pairs.
    // Uses a prefix from one class and the other class name, or a special
    // merged name for well-known combinations.
    match (class_a, class_b) {
        ("Warrior", "Scholar") | ("Scholar", "Warrior") => "Runic Warrior".to_string(),
        ("Warrior", "Mage") | ("Mage", "Warrior") => "Spellsword".to_string(),
        ("Healer", "Rogue") | ("Rogue", "Healer") => "Shadow Medic".to_string(),
        ("Healer", "Warrior") | ("Warrior", "Healer") => "Battle Medic".to_string(),
        ("Ranger", "Rogue") | ("Rogue", "Ranger") => "Stalker".to_string(),
        ("Ranger", "Warrior") | ("Warrior", "Ranger") => "Warden".to_string(),
        ("Commander", "Warrior") | ("Warrior", "Commander") => "War Marshal".to_string(),
        ("Commander", "Diplomat") | ("Diplomat", "Commander") => "Grand Strategist".to_string(),
        ("Merchant", "Rogue") | ("Rogue", "Merchant") => "Fence Master".to_string(),
        ("Merchant", "Diplomat") | ("Diplomat", "Merchant") => "Trade Envoy".to_string(),
        ("Scout", "Ranger") | ("Ranger", "Scout") => "Pathfinder".to_string(),
        ("Scout", "Rogue") | ("Rogue", "Scout") => "Infiltrator".to_string(),
        ("Guardian", "Healer") | ("Healer", "Guardian") => "Paladin".to_string(),
        ("Guardian", "Commander") | ("Commander", "Guardian") => "Bulwark Captain".to_string(),
        ("Artisan", "Scholar") | ("Scholar", "Artisan") => "Artificer".to_string(),
        ("Artisan", "Merchant") | ("Merchant", "Artisan") => "Master Crafter".to_string(),
        ("Scholar", "Healer") | ("Healer", "Scholar") => "Sage Physician".to_string(),
        ("Scholar", "Diplomat") | ("Diplomat", "Scholar") => "Loremaster".to_string(),
        _ => {
            // Generic: pick a prefix from class_a and append class_b
            let prefix = match class_a {
                "Warrior" => "Battle",
                "Ranger" => "Wild",
                "Healer" => "Blessed",
                "Diplomat" => "Silver",
                "Merchant" => "Gilded",
                "Scholar" => "Sage",
                "Rogue" => "Shadow",
                "Artisan" => "Forged",
                "Commander" => "Iron",
                "Scout" => "Swift",
                "Guardian" => "Stone",
                _ => "Ascended",
            };
            format!("{} {}", prefix, class_b)
        }
    }
}

/// Map for class evolution names.
fn evolution_name(class_name: &str) -> &str {
    match class_name {
        "Warrior" => "Champion",
        "Ranger" => "Huntmaster",
        "Healer" => "Archon",
        "Diplomat" => "Grand Ambassador",
        "Merchant" => "Trade Prince",
        "Scholar" => "Archmage",
        "Rogue" => "Phantom",
        "Artisan" => "Grand Artisan",
        "Commander" => "High Marshal",
        "Scout" => "Wayfinder",
        "Guardian" => "Aegis",
        "Spellblade" => "Mystic Blademaster",
        "Plague Doctor" => "Grand Apothecary",
        "Shadowmerchant" => "Black Market Baron",
        "Warlord" => "Conqueror",
        _ => "Exalted",
    }
}

/// Upgrade rarity by one tier for refusal banking.
fn next_rarity(r: &SkillRarity) -> SkillRarity {
    match r {
        SkillRarity::Common => SkillRarity::Uncommon,
        SkillRarity::Uncommon => SkillRarity::Rare,
        SkillRarity::Rare => SkillRarity::Capstone,
        SkillRarity::Capstone | SkillRarity::Unique => SkillRarity::Unique,
    }
}

fn check_consolidation_offers(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick as u32;

    // First, resolve expired offers as refusals
    let mut refused: Vec<ConsolidationOffer> = Vec::new();
    state.consolidation_offers.retain(|offer| {
        if tick >= offer.deadline_tick {
            refused.push(offer.clone());
            false
        } else {
            true
        }
    });
    for offer in &refused {
        events.push(WorldEvent::ConsolidationRefused {
            adventurer_id: offer.adventurer_id,
            proposed_name: offer.proposed_name.clone(),
        });
    }

    // Collect adventurer IDs and their qualifying class pairs
    struct CandidatePair {
        adv_id: u32,
        class_a: String,
        class_b: String,
        level_a: u32,
        level_b: u32,
        tags_a: Vec<String>,
        tags_b: Vec<String>,
    }

    let templates: Vec<ClassTemplate> = base_templates()
        .into_iter()
        .chain(rare_templates())
        .collect();

    let mut candidates: Vec<CandidatePair> = Vec::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        // Check all pairs of classes on this adventurer
        for i in 0..adv.classes.len() {
            for j in (i + 1)..adv.classes.len() {
                let ca = &adv.classes[i];
                let cb = &adv.classes[j];
                if ca.level >= CONSOLIDATION_MIN_LEVEL && cb.level >= CONSOLIDATION_MIN_LEVEL {
                    // Check if there's already a pending offer for this pair
                    let already_pending = state.consolidation_offers.iter().any(|o| {
                        o.adventurer_id == adv.id
                            && ((o.class_a == ca.class_name && o.class_b == cb.class_name)
                                || (o.class_a == cb.class_name && o.class_b == ca.class_name))
                    });
                    if already_pending {
                        continue;
                    }

                    let tags_a = templates
                        .iter()
                        .find(|t| t.class_name == ca.class_name)
                        .map(|t| t.tags.clone())
                        .unwrap_or_default();
                    let tags_b = templates
                        .iter()
                        .find(|t| t.class_name == cb.class_name)
                        .map(|t| t.tags.clone())
                        .unwrap_or_default();

                    candidates.push(CandidatePair {
                        adv_id: adv.id,
                        class_a: ca.class_name.clone(),
                        class_b: cb.class_name.clone(),
                        level_a: ca.level,
                        level_b: cb.level,
                        tags_a,
                        tags_b,
                    });
                }
            }
        }
    }

    // Create offers for each candidate pair
    for cand in candidates {
        let proposed_name = generate_consolidated_name(&cand.class_a, &cand.class_b);

        // Check refusal history — if this pair was refused before, upgrade rarity
        let times_refused = refused
            .iter()
            .filter(|o| {
                o.adventurer_id == cand.adv_id
                    && ((o.class_a == cand.class_a && o.class_b == cand.class_b)
                        || (o.class_a == cand.class_b && o.class_b == cand.class_a))
            })
            .map(|o| o.times_refused + 1)
            .max()
            .unwrap_or(0);

        let mut rarity = SkillRarity::Uncommon;
        for _ in 0..times_refused {
            rarity = next_rarity(&rarity);
        }

        // Combine tags from both parents
        let mut combined_tags = cand.tags_a.clone();
        for tag in &cand.tags_b {
            if !combined_tags.contains(tag) {
                combined_tags.push(tag.clone());
            }
        }

        // Compute Jaccard similarity of skill tag sets (idea 4.4)
        let overlap_score = jaccard_similarity(&cand.tags_a, &cand.tags_b);

        // Feature 4.7: Unique consolidation rarity tier
        let rarity = compute_consolidation_rarity(
            &rarity, &cand.class_a, &cand.class_b, state, &templates,
        );

        let offer = ConsolidationOffer {
            adventurer_id: cand.adv_id,
            class_a: cand.class_a.clone(),
            class_b: cand.class_b.clone(),
            proposed_name: proposed_name.clone(),
            proposed_tags: combined_tags.clone(),
            rarity: rarity.clone(),
            offered_tick: tick,
            deadline_tick: tick + CONSOLIDATION_DEADLINE_TICKS,
            times_refused,
            overlap_score,
        };

        events.push(WorldEvent::ConsolidationOffered {
            adventurer_id: cand.adv_id,
            proposed_name: proposed_name.clone(),
        });

        // Auto-accept with 70% probability
        let roll = lcg_f32(&mut state.rng);
        if roll < CONSOLIDATION_ACCEPT_PROB {
            // Accept: perform consolidation
            let new_level = cand.level_a.max(cand.level_b) + 2;

            // Feature 4.7: If the result is Unique, enforce single holder
            let (final_name, final_rarity) = if matches!(rarity, SkillRarity::Unique) {
                if state.unique_class_holders.contains_key(&proposed_name) {
                    // Someone else holds this unique — give Uncommon variant
                    (format!("{} Adept", proposed_name.trim_end_matches(" Master")), SkillRarity::Uncommon)
                } else {
                    state.unique_class_holders.insert(proposed_name.clone(), cand.adv_id);
                    (proposed_name.clone(), rarity.clone())
                }
            } else {
                (proposed_name.clone(), rarity.clone())
            };
            let _ = final_rarity; // used implicitly via class creation

            // Gather skills from each parent with inheritance tagging
            let adv = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == cand.adv_id)
                .unwrap();

            let inherited_skills = inherit_skills_for_consolidation(
                adv, &cand.class_a, &cand.class_b, &combined_tags,
            );

            // Remove both parent classes
            adv.classes
                .retain(|c| c.class_name != cand.class_a && c.class_name != cand.class_b);

            // Create new consolidated class
            let mut ci = ClassInstance {
                class_name: final_name.clone(),
                level: new_level,
                xp: 0.0,
                xp_to_next: (new_level * new_level) as f32 * 25.0,
                stagnation_ticks: 0,
                skills_granted: inherited_skills,
                acquired_tick: tick,
                identity_coherence: 1.0,
                ..Default::default()
            };
            populate_noncombat_growth(&mut ci);
            adv.classes.push(ci);

            events.push(WorldEvent::ClassConsolidated {
                adventurer_id: cand.adv_id,
                from_a: cand.class_a,
                from_b: cand.class_b,
                into: final_name,
            });
        } else {
            // Refused — store for future banking
            state.consolidation_offers.push(offer);
        }
    }
}

// ---------------------------------------------------------------------------
// Evolution
// ---------------------------------------------------------------------------

fn check_evolution(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick as u32;

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        // Achievement counters from behavior ledger
        let battles = adv.behavior_ledger.melee_combat + adv.behavior_ledger.ranged_combat;
        let quests = adv.behavior_ledger.areas_explored
            + adv.behavior_ledger.trades_completed
            + adv.behavior_ledger.diplomacy_actions;

        if battles < EVOLUTION_MIN_BATTLES || quests < EVOLUTION_MIN_QUESTS {
            continue;
        }

        // Check each class for evolution eligibility
        let mut evolutions: Vec<(usize, String, String)> = Vec::new();
        for (i, class) in adv.classes.iter().enumerate() {
            if class.level >= EVOLUTION_MIN_LEVEL {
                let evolved = evolution_name(&class.class_name);
                // Don't evolve if already evolved (name already changed)
                if evolved != "Exalted" || class.class_name == "Exalted" {
                    // Skip if already holding the evolved name
                    if class.class_name == evolved {
                        continue;
                    }
                }
                evolutions.push((i, class.class_name.clone(), evolved.to_string()));
            }
        }

        for (idx, from_name, to_name) in evolutions {
            // Apply evolution: rename and boost stat growth (+50% via level bump)
            let class = &mut adv.classes[idx];
            class.class_name = to_name.clone();
            // Grant +50% effective stat growth by adding 50% of current level as bonus levels
            let bonus_levels = class.level / 2;
            class.level += bonus_levels;
            class.xp_to_next = (class.level * class.level) as f32 * 15.0;
            class.acquired_tick = tick; // Mark evolution tick

            events.push(WorldEvent::ClassEvolved {
                adventurer_id: adv.id,
                from_class: from_name,
                to_class: to_name,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 7: Shame Classes
// ---------------------------------------------------------------------------

/// Shame class stat bonus suppression factor (20% reduction).
const SHAME_SUPPRESSION: f32 = 0.20;

/// Track shameful behavior patterns and auto-grant negative classes.
fn check_shame_classes(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick as u32;

    // Scan recent events for retreat/desertion signals and update ledger counters.
    for ev in events.iter() {
        match ev {
            WorldEvent::BattleEnded {
                result: BattleStatus::Retreat,
                ..
            } => {
                // Credit all fighting adventurers with a consecutive retreat.
                // (We iterate adventurers below, but need to mark them here.)
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Fighting
                        || adv.status == AdventurerStatus::Injured
                    {
                        adv.behavior_ledger.consecutive_retreats += 1;
                    }
                }
            }
            WorldEvent::BattleEnded {
                result: BattleStatus::Victory,
                ..
            } => {
                // Victory resets consecutive retreats for all combatants.
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Fighting {
                        adv.behavior_ledger.consecutive_retreats = 0;
                    }
                }
            }
            WorldEvent::AdventurerDeserted { adventurer_id, .. } => {
                if let Some(adv) =
                    state.adventurers.iter_mut().find(|a| a.id == *adventurer_id)
                {
                    adv.behavior_ledger.party_desertions += 1;
                }
            }
            _ => {}
        }
    }

    // Check broken oaths across all adventurers.
    let broken_oath_adventurers: Vec<u32> = state
        .oaths
        .iter()
        .filter(|o| o.broken)
        .map(|o| o.adventurer_id)
        .collect();

    // Now check each adventurer for shame class eligibility.
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        // [Coward]: fled 3+ consecutive battles
        if adv.behavior_ledger.consecutive_retreats >= 3
            && !adv.classes.iter().any(|c| c.class_name == "Coward")
        {
            adv.classes.push(ClassInstance {
                class_name: "Coward".to_string(),
                level: 1,
                xp: 0.0,
                xp_to_next: 100.0,
                stagnation_ticks: 0,
                skills_granted: Vec::new(),
                acquired_tick: tick,
                identity_coherence: 1.0,
                ..Default::default()
            });
            events.push(WorldEvent::ShameClassGranted {
                adventurer_id: adv.id,
                class_name: "Coward".to_string(),
                reason: format!(
                    "Fled {} consecutive battles",
                    adv.behavior_ledger.consecutive_retreats
                ),
            });
        }

        // [Oathbreaker]: broke a sworn oath
        if broken_oath_adventurers.contains(&adv.id)
            && !adv.classes.iter().any(|c| c.class_name == "Oathbreaker")
        {
            adv.classes.push(ClassInstance {
                class_name: "Oathbreaker".to_string(),
                level: 1,
                xp: 0.0,
                xp_to_next: 100.0,
                stagnation_ticks: 0,
                skills_granted: Vec::new(),
                acquired_tick: tick,
                identity_coherence: 1.0,
                ..Default::default()
            });
            events.push(WorldEvent::ShameClassGranted {
                adventurer_id: adv.id,
                class_name: "Oathbreaker".to_string(),
                reason: "Broke a sworn oath".to_string(),
            });
        }

        // [Deserter]: left a party during an active quest 3+ times
        if adv.behavior_ledger.party_desertions >= 3
            && !adv.classes.iter().any(|c| c.class_name == "Deserter")
        {
            adv.classes.push(ClassInstance {
                class_name: "Deserter".to_string(),
                level: 1,
                xp: 0.0,
                xp_to_next: 100.0,
                stagnation_ticks: 0,
                skills_granted: Vec::new(),
                acquired_tick: tick,
                identity_coherence: 1.0,
                ..Default::default()
            });
            events.push(WorldEvent::ShameClassGranted {
                adventurer_id: adv.id,
                class_name: "Deserter".to_string(),
                reason: format!(
                    "Deserted party {} times during active quests",
                    adv.behavior_ledger.party_desertions
                ),
            });
        }
    }
}

/// Check if a class is a shame class. Shame classes cannot be removed and
/// suppress stat bonuses from other classes by 20%.
fn is_shame_class(name: &str) -> bool {
    matches!(name, "Coward" | "Oathbreaker" | "Deserter")
}

/// Returns the shame suppression factor for an adventurer.
/// 1.0 = no shame classes, 0.8 per shame class held.
pub fn shame_suppression_factor(classes: &[ClassInstance]) -> f32 {
    let shame_count = classes.iter().filter(|c| is_shame_class(&c.class_name)).count();
    if shame_count == 0 {
        1.0
    } else {
        (1.0 - SHAME_SUPPRESSION).powi(shame_count as i32)
    }
}

/// Compute effective non-combat stat bonuses from all held classes.
/// Returns (diplomacy, commerce, crafting, medicine, scholarship, stealth, leadership).
///
/// For each class the adventurer holds:
///   bonus = class.level * stat_growth_per_level * activity_recency_factor
/// Sum across all classes, then apply shame suppression (20% per shame class).
pub fn effective_noncombat_stats(adv: &Adventurer) -> (f32, f32, f32, f32, f32, f32, f32) {
    let shame_factor = shame_suppression_factor(&adv.classes);
    let mut diplomacy = 0.0f32;
    let mut commerce = 0.0f32;
    let mut crafting = 0.0f32;
    let mut medicine = 0.0f32;
    let mut scholarship = 0.0f32;
    let mut stealth = 0.0f32;
    let mut leadership = 0.0f32;

    for class in &adv.classes {
        if is_shame_class(&class.class_name) {
            continue;
        }
        let recency = class_stat_contribution(class, &adv.behavior_ledger);
        let level = class.level as f32;
        diplomacy += level * class.stat_growth_diplomacy * recency;
        commerce += level * class.stat_growth_commerce * recency;
        crafting += level * class.stat_growth_crafting * recency;
        medicine += level * class.stat_growth_medicine * recency;
        scholarship += level * class.stat_growth_scholarship * recency;
        stealth += level * class.stat_growth_stealth * recency;
        leadership += level * class.stat_growth_leadership * recency;
    }

    (
        diplomacy * shame_factor,
        commerce * shame_factor,
        crafting * shame_factor,
        medicine * shame_factor,
        scholarship * shame_factor,
        stealth * shame_factor,
        leadership * shame_factor,
    )
}

/// Returns the default non-combat stat growth rates for a class based on its name.
/// Classes not in this table get zero non-combat growth (combat-only classes).
fn default_noncombat_growth(class_name: &str) -> (f32, f32, f32, f32, f32, f32, f32) {
    // (diplomacy, commerce, crafting, medicine, scholarship, stealth, leadership)
    match class_name {
        "Diplomat" | "Herald"       => (3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        "Merchant" | "Shadowmerchant" => (1.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "Scholar"                   => (0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0),
        "Healer" | "Plague Doctor"  => (0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0),
        "Rogue" | "Ghost" | "The Unseen Hand" | "Phantom" => (0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0),
        "Artisan"                   => (0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0),
        "Commander" | "Warlord" | "Banner Knight" => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0),
        "Scout"                     => (0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0),
        "Guardian" | "Silent Guardian" => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        // Starter classes — humble, modest growth
        "Laborer"       => (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
        "Hunter"        => (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),
        "Traveler"      => (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "Apprentice"    => (0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0),
        "Farmhand"      => (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        "Militia"       => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        "Peddler"       => (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "Herbalist"     => (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        "Scribe"        => (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        "Pickpocket"    => (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        "Errand Runner" => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        "Stablehand"    => (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        "Warrior" | "Ranger" | "Spellblade" => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "Champion" | "People's Blade" => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        "Paragon" | "Nameless Savior" => (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        _ => {
            // For unique/generated classes: distribute small growth based on name heuristics
            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        }
    }
}

/// Populate stat_growth fields on a ClassInstance based on its class name.
fn populate_noncombat_growth(ci: &mut ClassInstance) {
    let (d, c, cr, m, s, st, l) = default_noncombat_growth(&ci.class_name);
    ci.stat_growth_diplomacy = d;
    ci.stat_growth_commerce = c;
    ci.stat_growth_crafting = cr;
    ci.stat_growth_medicine = m;
    ci.stat_growth_scholarship = s;
    ci.stat_growth_stealth = st;
    ci.stat_growth_leadership = l;
}

// ---------------------------------------------------------------------------
// Phase 7: Crisis Moment Grants
// ---------------------------------------------------------------------------

/// Check for crisis-worthy moments and grant unique unrepeatable classes.
fn check_crisis_grants(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick as u32;

    // Update tension accumulators based on dangerous situations.
    for ev in events.iter() {
        match ev {
            // Being in battle while outnumbered or at low HP raises tension.
            WorldEvent::BattleStarted { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Fighting {
                        adv.behavior_ledger.tension_accumulator += 0.1;
                    }
                }
            }
            // Ally death in same party raises tension sharply.
            WorldEvent::AdventurerDied { adventurer_id, .. } => {
                // Find the dead adventurer's party, raise tension for party members.
                let party_id = state
                    .adventurers
                    .iter()
                    .find(|a| a.id == *adventurer_id)
                    .and_then(|a| a.party_id);
                if let Some(pid) = party_id {
                    for adv in &mut state.adventurers {
                        if adv.party_id == Some(pid)
                            && adv.id != *adventurer_id
                            && adv.status != AdventurerStatus::Dead
                        {
                            adv.behavior_ledger.tension_accumulator += 0.3;
                        }
                    }
                }
            }
            // Injury (near death) raises tension.
            WorldEvent::AdventurerInjured {
                adventurer_id,
                injury_level,
            } => {
                if *injury_level > 80.0 {
                    if let Some(adv) =
                        state.adventurers.iter_mut().find(|a| a.id == *adventurer_id)
                    {
                        adv.behavior_ledger.tension_accumulator += 0.2;
                        adv.behavior_ledger.near_death_count += 1;
                    }
                }
            }
            _ => {}
        }
    }

    // Natural tension decay — slowly returns to 0.
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        adv.behavior_ledger.tension_accumulator =
            (adv.behavior_ledger.tension_accumulator - 0.02).max(0.0);
    }

    // Check crisis grant conditions for each adventurer.
    // Collect grants first to avoid borrow issues.
    struct CrisisGrant {
        adv_id: u32,
        class_name: String,
        crisis_type: String,
    }
    let mut grants: Vec<CrisisGrant> = Vec::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        // Must have tension above 0.8
        if adv.behavior_ledger.tension_accumulator < 0.8 {
            continue;
        }

        let fp = behavioral_fingerprint(&adv.behavior_ledger);

        // [The Last Wall]: held position while outnumbered — high damage_absorbed + low HP
        if !state.unique_class_holders.contains_key("The Last Wall")
            && !adv.classes.iter().any(|c| c.class_name == "The Last Wall")
            && fp[10] > 0.15 // damage_absorbed fingerprint
            && adv.injury > 60.0
            && adv.status == AdventurerStatus::Fighting
        {
            grants.push(CrisisGrant {
                adv_id: adv.id,
                class_name: "The Last Wall".to_string(),
                crisis_type: "held_position_outnumbered".to_string(),
            });
            continue; // Only one crisis class per tick per adventurer
        }

        // [Mercy in Iron]: healed allies while near death
        if !state.unique_class_holders.contains_key("Mercy in Iron")
            && !adv.classes.iter().any(|c| c.class_name == "Mercy in Iron")
            && fp[2] > 0.15 // healing_given fingerprint
            && adv.injury > 70.0
        {
            grants.push(CrisisGrant {
                adv_id: adv.id,
                class_name: "Mercy in Iron".to_string(),
                crisis_type: "healed_while_near_death".to_string(),
            });
            continue;
        }

        // [Risen Commander]: led a victory after losing party members
        if !state.unique_class_holders.contains_key("Risen Commander")
            && !adv.classes.iter().any(|c| c.class_name == "Risen Commander")
            && fp[7] > 0.1 // units_commanded fingerprint
            && adv.behavior_ledger.tension_accumulator > 0.9
        {
            // Check if there were recent ally deaths (tension would be spiked from AdventurerDied)
            let recent_ally_deaths = events.iter().any(|e| {
                matches!(e, WorldEvent::AdventurerDied { .. })
            });
            let recent_victory = events.iter().any(|e| {
                matches!(
                    e,
                    WorldEvent::BattleEnded {
                        result: BattleStatus::Victory,
                        ..
                    }
                )
            });
            if recent_ally_deaths && recent_victory {
                grants.push(CrisisGrant {
                    adv_id: adv.id,
                    class_name: "Risen Commander".to_string(),
                    crisis_type: "led_victory_after_loss".to_string(),
                });
                continue;
            }
        }

        // [The Unkillable]: survived 3+ near-death experiences
        if !state.unique_class_holders.contains_key("The Unkillable")
            && !adv.classes.iter().any(|c| c.class_name == "The Unkillable")
            && adv.behavior_ledger.near_death_count >= 3
        {
            grants.push(CrisisGrant {
                adv_id: adv.id,
                class_name: "The Unkillable".to_string(),
                crisis_type: "survived_near_death".to_string(),
            });
        }
    }

    // Apply grants
    for grant in grants {
        // Double-check uniqueness (another grant in same tick could collide)
        if state.unique_class_holders.contains_key(&grant.class_name) {
            continue;
        }

        state
            .unique_class_holders
            .insert(grant.class_name.clone(), grant.adv_id);

        if let Some(adv) = state
            .adventurers
            .iter_mut()
            .find(|a| a.id == grant.adv_id)
        {
            let mut ci = ClassInstance {
                class_name: grant.class_name.clone(),
                level: 1,
                xp: 0.0,
                xp_to_next: 100.0,
                stagnation_ticks: 0,
                skills_granted: Vec::new(),
                acquired_tick: tick,
                identity_coherence: 1.0,
                ..Default::default()
            };
            populate_noncombat_growth(&mut ci);
            adv.classes.push(ci);

            // Reset tension after a crisis grant
            adv.behavior_ledger.tension_accumulator = 0.0;
        }

        events.push(WorldEvent::CrisisClassGranted {
            adventurer_id: grant.adv_id,
            class_name: grant.class_name,
            crisis_type: grant.crisis_type,
        });
    }
}

// ---------------------------------------------------------------------------
// Phase 7: Identity Erosion
// ---------------------------------------------------------------------------

/// Coherence decay rate per tick when actions contradict class.
const COHERENCE_DECAY: f32 = 0.02;
/// Coherence recovery rate per tick when actions align with class.
const COHERENCE_RECOVER: f32 = 0.01;

/// Map from class name to the behavioral fingerprint indices that align with it.
/// Returns (aligned_indices, contradicting_indices).
fn class_alignment_indices(class_name: &str) -> (Vec<usize>, Vec<usize>) {
    match class_name {
        "Healer" => (
            vec![2, 11], // healing_given, allies_supported
            vec![0, 1],  // melee_combat, ranged_combat (dealing damage)
        ),
        "Guardian" => (
            vec![10, 11], // damage_absorbed, allies_supported
            vec![8],      // stealth_actions (guardians don't sneak)
        ),
        "Diplomat" => (
            vec![3],     // diplomacy_actions
            vec![0, 1],  // melee/ranged combat
        ),
        "Warrior" => (
            vec![0, 10], // melee_combat, damage_absorbed
            vec![3],     // diplomacy_actions
        ),
        "Ranger" => (
            vec![1, 6], // ranged_combat, areas_explored
            vec![],
        ),
        "Rogue" => (
            vec![8, 0], // stealth_actions, melee_combat
            vec![3],    // diplomacy_actions
        ),
        "Merchant" => (
            vec![4, 3], // trades_completed, diplomacy_actions
            vec![0],    // melee_combat
        ),
        "Scholar" => (
            vec![9, 6], // research_performed, areas_explored
            vec![0],    // melee_combat
        ),
        "Commander" => (
            vec![7, 0], // units_commanded, melee_combat
            vec![],
        ),
        "Scout" => (
            vec![6, 8], // areas_explored, stealth_actions
            vec![],
        ),
        "Artisan" => (
            vec![5, 4], // items_crafted, trades_completed
            vec![0],    // melee_combat
        ),
        _ => (vec![], vec![]), // Unknown/consolidated classes — no erosion
    }
}

/// Fractured class replacement mapping.
fn fractured_replacement(class_name: &str) -> &str {
    match class_name {
        "Healer" => "Mender of Last Resort",
        "Guardian" => "Broken Shield",
        "Diplomat" => "Failed Envoy",
        "Warrior" => "Battered Veteran",
        "Ranger" => "Lost Wanderer",
        "Rogue" => "Exposed Shadow",
        "Merchant" => "Bankrupt Trader",
        "Scholar" => "Scattered Mind",
        "Commander" => "Fallen Officer",
        "Scout" => "Blinded Scout",
        "Artisan" => "Clumsy Hands",
        _ => "Fractured Echo",
    }
}

/// Check class identity coherence — decay when actions contradict, recover when aligned.
fn check_identity_erosion(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick as u32;

    // Collect fractures to apply after iteration.
    struct Fracture {
        adv_id: u32,
        class_idx: usize,
        original: String,
        replacement: String,
    }
    let mut fractures: Vec<Fracture> = Vec::new();

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        let fp = behavioral_fingerprint(&adv.behavior_ledger);

        for (idx, class) in adv.classes.iter_mut().enumerate() {
            // Skip shame classes and crisis classes — they don't erode.
            if is_shame_class(&class.class_name) {
                continue;
            }

            let (aligned, contradicting) = class_alignment_indices(&class.class_name);

            // If no alignment data, skip (consolidated/evolved/unknown classes).
            if aligned.is_empty() && contradicting.is_empty() {
                continue;
            }

            let aligned_activity: f32 = aligned.iter().map(|&i| fp[i]).sum();
            let contradicting_activity: f32 = contradicting.iter().map(|&i| fp[i]).sum();

            if contradicting_activity > aligned_activity && contradicting_activity > 0.05 {
                // Actions contradict class — decay coherence.
                class.identity_coherence =
                    (class.identity_coherence - COHERENCE_DECAY).max(0.0);
            } else if aligned_activity > 0.05 {
                // Actions align — recover coherence.
                class.identity_coherence =
                    (class.identity_coherence + COHERENCE_RECOVER).min(1.0);
            }

            // Check for fracture.
            if class.identity_coherence <= 0.0 {
                let replacement = fractured_replacement(&class.class_name).to_string();
                fractures.push(Fracture {
                    adv_id: adv.id,
                    class_idx: idx,
                    original: class.class_name.clone(),
                    replacement,
                });
            }
        }
    }

    // Apply fractures (replace the class).
    for frac in fractures {
        if let Some(adv) = state
            .adventurers
            .iter_mut()
            .find(|a| a.id == frac.adv_id)
        {
            if frac.class_idx < adv.classes.len()
                && adv.classes[frac.class_idx].class_name == frac.original
            {
                let old_level = adv.classes[frac.class_idx].level;
                // Replace with diminished version at half the level.
                let mut ci = ClassInstance {
                    class_name: frac.replacement.clone(),
                    level: (old_level / 2).max(1),
                    xp: 0.0,
                    xp_to_next: ((old_level / 2).max(1).pow(2)) as f32 * 100.0,
                    stagnation_ticks: 0,
                    skills_granted: Vec::new(),
                    acquired_tick: tick,
                    identity_coherence: 0.5, // Starts at half coherence
                    ..Default::default()
                };
                populate_noncombat_growth(&mut ci);
                adv.classes[frac.class_idx] = ci;

                events.push(WorldEvent::ClassFractured {
                    adventurer_id: frac.adv_id,
                    original_class: frac.original.clone(),
                    replacement: frac.replacement.clone(),
                });

                // Generate chronicle entry for fracture.
                let entry = format!(
                    "The class [{}] has shattered. What remains is [{}] — a diminished echo of what was. \
                     The system remembers, even if they wish to forget.",
                    frac.original, frac.replacement
                );
                state.class_chronicle_entries.push(entry.clone());
                events.push(WorldEvent::ClassChronicleEntry {
                    adventurer_id: frac.adv_id,
                    entry,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 7: Chronicle Entries
// ---------------------------------------------------------------------------

/// Generate chronicle-style entries for significant class events.
fn generate_class_chronicle(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Scan the events for significant class events and generate chronicle entries.
    // We iterate a snapshot of events to avoid borrow issues.
    let event_snapshot: Vec<WorldEvent> = events.clone();

    for ev in &event_snapshot {
        match ev {
            WorldEvent::ClassGranted {
                adventurer_id,
                class_name,
            } => {
                let name = adventurer_name(state, *adventurer_id);
                let entry = format!(
                    "We have seen what {} does. A class forms: [{}]. Level 1. The beginning.",
                    name, class_name
                );
                state.class_chronicle_entries.push(entry.clone());
                events.push(WorldEvent::ClassChronicleEntry {
                    adventurer_id: *adventurer_id,
                    entry,
                });
            }
            WorldEvent::ClassLevelUp {
                adventurer_id,
                class_name,
                new_level,
            } => {
                // Only chronicle level 10+
                if *new_level >= 10 {
                    let name = adventurer_name(state, *adventurer_id);
                    let entry = format!(
                        "{} has reached level {} in [{}]. The system takes note. Power consolidates.",
                        name, new_level, class_name
                    );
                    state.class_chronicle_entries.push(entry.clone());
                    events.push(WorldEvent::ClassChronicleEntry {
                        adventurer_id: *adventurer_id,
                        entry,
                    });
                }
            }
            WorldEvent::ClassConsolidated {
                adventurer_id,
                from_a,
                from_b,
                into,
            } => {
                let name = adventurer_name(state, *adventurer_id);
                // Look up overlap_score from any matching recent offer
                let overlap_info = state.consolidation_offers.iter()
                    .find(|o| o.adventurer_id == *adventurer_id
                        && ((o.class_a == *from_a && o.class_b == *from_b)
                            || (o.class_a == *from_b && o.class_b == *from_a)))
                    .map(|o| o.overlap_score)
                    .unwrap_or(0.5);
                let merge_style = if overlap_info > 0.6 {
                    "a tight fusion — skills overlap, focus sharpens"
                } else if overlap_info < 0.2 {
                    "a broad merger — wider reach, but shallower roots"
                } else {
                    "a balanced blend — old and new intertwined"
                };
                let entry = format!(
                    "Two paths have merged. [{}] and [{}] dissolve into [{}]. {} walks a narrower, deeper road. It is {}.",
                    from_a, from_b, into, name, merge_style
                );
                state.class_chronicle_entries.push(entry.clone());
                events.push(WorldEvent::ClassChronicleEntry {
                    adventurer_id: *adventurer_id,
                    entry,
                });
            }
            WorldEvent::ShameClassGranted {
                adventurer_id,
                class_name,
                reason,
            } => {
                let name = adventurer_name(state, *adventurer_id);
                let entry = format!(
                    "A mark of shame. [{}] is branded upon {}. Reason: {}. \
                     This cannot be undone. Only buried.",
                    class_name, name, reason
                );
                state.class_chronicle_entries.push(entry.clone());
                events.push(WorldEvent::ClassChronicleEntry {
                    adventurer_id: *adventurer_id,
                    entry,
                });
            }
            WorldEvent::CrisisClassGranted {
                adventurer_id,
                class_name,
                crisis_type,
            } => {
                let name = adventurer_name(state, *adventurer_id);
                let entry = match crisis_type.as_str() {
                    "held_position_outnumbered" => format!(
                        "This one has fought beyond reason. We have been watching. \
                         A name is forming. {} is now [{}].",
                        name, class_name
                    ),
                    "healed_while_near_death" => format!(
                        "Even broken, this one mends others. We have been watching. \
                         A name is forming. {} is now [{}].",
                        name, class_name
                    ),
                    "led_victory_after_loss" => format!(
                        "They fell, and still this one commanded. We have been watching. \
                         A name is forming. {} is now [{}].",
                        name, class_name
                    ),
                    "survived_near_death" => format!(
                        "Death has reached for this one and missed. Three times now. \
                         We have been watching. A name is forming. {} is now [{}].",
                        name, class_name
                    ),
                    _ => format!(
                        "A crisis moment. {} earns [{}]. The system acknowledges.",
                        name, class_name
                    ),
                };
                state.class_chronicle_entries.push(entry.clone());
                events.push(WorldEvent::ClassChronicleEntry {
                    adventurer_id: *adventurer_id,
                    entry,
                });
            }
            WorldEvent::ClassEvolved {
                adventurer_id,
                from_class,
                to_class,
            } => {
                let name = adventurer_name(state, *adventurer_id);
                let entry = format!(
                    "[{}] evolves. {} is now [{}]. The old name fades. The new one burns brighter.",
                    from_class, name, to_class
                );
                state.class_chronicle_entries.push(entry.clone());
                events.push(WorldEvent::ClassChronicleEntry {
                    adventurer_id: *adventurer_id,
                    entry,
                });
            }
            // ClassFractured chronicle entries are already generated in check_identity_erosion
            _ => {}
        }
    }
}

/// Helper to look up an adventurer's name by ID.
fn adventurer_name(state: &CampaignState, id: u32) -> String {
    state
        .adventurers
        .iter()
        .find(|a| a.id == id)
        .map(|a| a.name.clone())
        .unwrap_or_else(|| format!("Adventurer #{}", id))
}

// ---------------------------------------------------------------------------
// Jaccard similarity for skill tag overlap (idea 4.4)
// ---------------------------------------------------------------------------

/// Compute the Jaccard similarity between two tag sets.
/// Returns |intersection| / |union|, or 0.0 if both sets are empty.
fn jaccard_similarity(tags_a: &[String], tags_b: &[String]) -> f32 {
    if tags_a.is_empty() && tags_b.is_empty() {
        return 0.0;
    }
    let intersection = tags_a.iter().filter(|t| tags_b.contains(t)).count();
    let mut union_set: Vec<&String> = tags_a.iter().collect();
    for t in tags_b {
        if !union_set.contains(&t) {
            union_set.push(t);
        }
    }
    let union_count = union_set.len();
    if union_count == 0 {
        return 0.0;
    }
    intersection as f32 / union_count as f32
}

// ---------------------------------------------------------------------------
// Consolidation rarity tier with unique holders (idea 4.7)
// ---------------------------------------------------------------------------

/// Compute the consolidation rarity. If both parent classes are Rare+, or the
/// behavioral fingerprint is >2 std devs from any template, mark as Unique.
fn compute_consolidation_rarity(
    base_rarity: &SkillRarity,
    class_a: &str,
    class_b: &str,
    state: &CampaignState,
    templates: &[ClassTemplate],
) -> SkillRarity {
    // Check if both parent classes are Rare or higher
    let rarity_a = templates
        .iter()
        .find(|t| t.class_name == class_a)
        .map(|t| &t.rarity);
    let rarity_b = templates
        .iter()
        .find(|t| t.class_name == class_b)
        .map(|t| &t.rarity);

    let is_rare_plus = |r: Option<&SkillRarity>| -> bool {
        matches!(
            r,
            Some(SkillRarity::Rare) | Some(SkillRarity::Capstone) | Some(SkillRarity::Unique)
        )
    };

    if is_rare_plus(rarity_a) && is_rare_plus(rarity_b) {
        return SkillRarity::Unique;
    }

    // Check behavioral fingerprint distance from all templates
    // Find the adventurer who has both classes
    for adv in &state.adventurers {
        let has_a = adv.classes.iter().any(|c| c.class_name == class_a);
        let has_b = adv.classes.iter().any(|c| c.class_name == class_b);
        if has_a && has_b {
            let fp = behavioral_fingerprint(&adv.behavior_ledger);
            // Compute max similarity to any template
            let mut max_sim = 0.0f32;
            for tmpl in templates {
                let s = score_template(&fp, tmpl);
                if s > max_sim {
                    max_sim = s;
                }
            }
            // Compute mean and variance of template scores
            let n = templates.len() as f32;
            let mean: f32 = templates.iter().map(|t| score_template(&fp, t)).sum::<f32>() / n;
            let variance: f32 = templates
                .iter()
                .map(|t| {
                    let d = score_template(&fp, t) - mean;
                    d * d
                })
                .sum::<f32>()
                / n;
            let std_dev = variance.sqrt();
            // If max score is more than 2 std devs from mean, this fingerprint is unusual
            if std_dev > 0.001 && max_sim < mean - 2.0 * std_dev {
                return SkillRarity::Unique;
            }
            break;
        }
    }

    base_rarity.clone()
}

// ---------------------------------------------------------------------------
// Skill inheritance rules (idea 4.8)
// ---------------------------------------------------------------------------

/// Classify and inherit skills during consolidation.
/// - `core`: Rare+ or capstone skills always transfer at full potency.
/// - `vestigial`: Common/Uncommon skills matching merged tags transfer at half.
/// - `lost`: Skills not matching merged tags are remembered but don't transfer.
fn inherit_skills_for_consolidation(
    adv: &crate::headless_campaign::state::Adventurer,
    class_a: &str,
    class_b: &str,
    merged_tags: &[String],
) -> Vec<GrantedSkill> {
    let mut result: Vec<GrantedSkill> = Vec::new();

    for class_name in &[class_a, class_b] {
        if let Some(class) = adv.classes.iter().find(|c| &c.class_name.as_str() == class_name) {
            for skill in &class.skills_granted {
                let is_rare_plus = matches!(
                    skill.rarity,
                    SkillRarity::Rare | SkillRarity::Capstone | SkillRarity::Unique
                );
                let is_capstone = skill.granted_at_level >= 25;
                let tags_match = merged_tags
                    .iter()
                    .any(|t| skill.from_class.to_lowercase().contains(&t.to_lowercase()));
                // A more robust tag match: check if the skill's source class has tags
                // overlapping with the merged class tags
                let source_matches = !merged_tags.is_empty(); // simplified — all consolidation skills have some tag relevance

                let inheritance_status = if is_rare_plus || is_capstone {
                    "core"
                } else if tags_match || source_matches {
                    match skill.rarity {
                        SkillRarity::Common | SkillRarity::Uncommon => "vestigial",
                        _ => "core",
                    }
                } else {
                    "lost"
                };

                let mut inherited = skill.clone();
                inherited.inheritance_status = inheritance_status.to_string();

                // Vestigial skills get "(half potency)" appended to description
                if inheritance_status == "vestigial" {
                    inherited.effect_description =
                        format!("{} (half potency)", inherited.effect_description);
                }

                // Lost skills are remembered but not active — still add them for tracking
                if inheritance_status == "lost" {
                    inherited.effect_description =
                        format!("{} (lost — remembered only)", inherited.effect_description);
                }

                result.push(inherited);
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// World-Event-Gated Unique Classes (idea 4.6)
// ---------------------------------------------------------------------------

/// Check for world events that unlock legendary class eligibility.
/// Called every 500 ticks. Tracks eligibility flags and grants classes
/// to adventurers who were present during the triggering events.
fn check_world_gated_classes(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;

    // --- Scan world state for gating conditions ---

    // [Arbiter]: civil war resolved peacefully (description contains "exhaustion")
    if !state.world_gated_class_flags.contains_key("civil_war_peaceful") {
        for ev in events.iter() {
            if let WorldEvent::CivilWarResolved { description, .. } = ev {
                if description.contains("exhaustion") || description.contains("ceasefire") {
                    state.world_gated_class_flags.insert("civil_war_peaceful".to_string(), tick);
                    break;
                }
            }
        }
    }

    // [Plague Sovereign]: plague killed 10+ population
    // We track this by checking if any plague vector has high mortality and infected regions
    if !state.world_gated_class_flags.contains_key("plague_10_dead") {
        for pv in &state.plague_vectors {
            // Estimate deaths: mortality * virulence * infected region count
            let estimated_deaths = pv.mortality * pv.virulence * pv.infected_regions.len() as f32 * 10.0;
            if estimated_deaths >= 10.0 {
                state.world_gated_class_flags.insert("plague_10_dead".to_string(), tick);
                break;
            }
        }
    }

    // [Market Maker]: survived a bankruptcy cascade (systemic_risk was > 0.5 and is now < 0.2)
    if !state.world_gated_class_flags.contains_key("bankruptcy_survived") {
        if state.bankruptcy_cascade.systemic_risk < 0.2
            && !state.bankruptcy_cascade.defaulted_factions.is_empty()
        {
            state.world_gated_class_flags.insert("bankruptcy_survived".to_string(), tick);
        }
    }

    // [Twilight Warden]: threat_clock power reached 0.6+ then disrupted below 0.3
    if !state.world_gated_class_flags.contains_key("threat_clock_disrupted") {
        let tc = &state.world_threat_clock;
        if tc.disruptions > 0 && tc.power < 0.3 {
            // Check if it was ever >= 0.6 (we infer from disruptions > 0 and growth_rate > 0)
            if tc.growth_rate > 0.0 || tc.disruptions >= 2 {
                state.world_gated_class_flags.insert("threat_clock_disrupted".to_string(), tick);
            }
        }
    }

    // [Oathsworn]: 3+ oaths honored without breaking any
    if !state.world_gated_class_flags.contains_key("oaths_honored_3") {
        // Group oaths by adventurer and check for 3+ fulfilled with 0 broken
        let mut fulfilled_by_adv: std::collections::HashMap<u32, (u32, u32)> =
            std::collections::HashMap::new();
        for oath in &state.oaths {
            let entry = fulfilled_by_adv.entry(oath.adventurer_id).or_insert((0, 0));
            if oath.fulfilled {
                entry.0 += 1;
            }
            if oath.broken {
                entry.1 += 1;
            }
        }
        for (&adv_id, &(fulfilled, broken)) in &fulfilled_by_adv {
            if fulfilled >= 3 && broken == 0 {
                state.world_gated_class_flags.insert(
                    format!("oaths_honored_3_adv_{}", adv_id),
                    tick,
                );
                // Also set the global flag
                state.world_gated_class_flags.insert("oaths_honored_3".to_string(), tick);
            }
        }
    }

    // --- Grant world-gated classes to eligible adventurers ---

    struct WorldGatedGrant {
        adv_id: u32,
        class_name: String,
        world_event: String,
    }
    let mut grants: Vec<WorldGatedGrant> = Vec::new();

    let world_gated_classes: Vec<(&str, &str, &str)> = vec![
        ("Arbiter", "civil_war_peaceful", "civil_war_peaceful_resolution"),
        ("Plague Sovereign", "plague_10_dead", "survived_deadly_plague"),
        ("Market Maker", "bankruptcy_survived", "survived_bankruptcy_cascade"),
        ("Twilight Warden", "threat_clock_disrupted", "disrupted_world_threat"),
    ];

    for (class_name, flag, event_desc) in &world_gated_classes {
        if !state.world_gated_class_flags.contains_key(*flag) {
            continue;
        }
        if state.unique_class_holders.contains_key(*class_name) {
            continue;
        }

        let flag_tick = state.world_gated_class_flags[*flag];

        // Find an adventurer who was present during the event
        // (was alive and active within 500 ticks of the event)
        for adv in &state.adventurers {
            if adv.status == AdventurerStatus::Dead {
                continue;
            }
            if adv.classes.iter().any(|c| c.class_name == *class_name) {
                continue;
            }

            // Check behavior ledger for relevant activity during the event window
            let total_behavior: f32 = adv.behavior_ledger.melee_combat
                + adv.behavior_ledger.diplomacy_actions
                + adv.behavior_ledger.trades_completed
                + adv.behavior_ledger.healing_given
                + adv.behavior_ledger.research_performed;

            // Must have been active (non-trivial behavior)
            if total_behavior < 5.0 {
                continue;
            }

            // Was present if they joined before or during the event
            let was_present = adv.classes.iter().any(|c| (c.acquired_tick as u64) <= flag_tick + 500);
            if !was_present && !adv.classes.is_empty() {
                continue;
            }

            grants.push(WorldGatedGrant {
                adv_id: adv.id,
                class_name: class_name.to_string(),
                world_event: event_desc.to_string(),
            });
            break; // Only one holder per class
        }
    }

    // Oathsworn: per-adventurer check
    if !state.unique_class_holders.contains_key("Oathsworn") {
        for adv in &state.adventurers {
            if adv.status == AdventurerStatus::Dead {
                continue;
            }
            if adv.classes.iter().any(|c| c.class_name == "Oathsworn") {
                continue;
            }
            let flag_key = format!("oaths_honored_3_adv_{}", adv.id);
            if state.world_gated_class_flags.contains_key(&flag_key) {
                grants.push(WorldGatedGrant {
                    adv_id: adv.id,
                    class_name: "Oathsworn".to_string(),
                    world_event: "honored_three_oaths".to_string(),
                });
                break; // Only one holder
            }
        }
    }

    // Apply grants
    let tick_u32 = tick as u32;
    for grant in grants {
        if state.unique_class_holders.contains_key(&grant.class_name) {
            continue;
        }
        state
            .unique_class_holders
            .insert(grant.class_name.clone(), grant.adv_id);

        if let Some(adv) = state
            .adventurers
            .iter_mut()
            .find(|a| a.id == grant.adv_id)
        {
            let mut ci = ClassInstance {
                class_name: grant.class_name.clone(),
                level: 1,
                xp: 0.0,
                xp_to_next: 100.0,
                stagnation_ticks: 0,
                skills_granted: Vec::new(),
                acquired_tick: tick_u32,
                identity_coherence: 1.0,
                capstone_required: false,
                xp_overflow: 0.0,
                ..Default::default()
            };
            populate_noncombat_growth(&mut ci);
            adv.classes.push(ci);
        }

        events.push(WorldEvent::WorldGatedClassUnlocked {
            adventurer_id: grant.adv_id,
            class_name: grant.class_name.clone(),
            world_event: grant.world_event,
        });

        let name = adventurer_name(state, grant.adv_id);
        let entry = format!(
            "The world has shifted, and so has [{}]. {} is now [{}] — a class born from world events, not mere behavior.",
            grant.class_name, name, grant.class_name
        );
        state.class_chronicle_entries.push(entry.clone());
        events.push(WorldEvent::ClassChronicleEntry {
            adventurer_id: grant.adv_id,
            entry,
        });
    }
}

// ---------------------------------------------------------------------------
// Crisis Escape Valve Consolidation (idea 4.10)
// ---------------------------------------------------------------------------

/// Emergency consolidation deadline (much shorter than normal).
const EMERGENCY_CONSOLIDATION_DEADLINE: u32 = 200;

/// When tension > 0.9 and adventurer holds 2+ classes, offer emergency consolidation.
/// If refused and tension stays > 0.9, 20% chance of negative class mutation.
fn check_crisis_escape_valve(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick as u32;

    // Collect candidates for emergency consolidation
    struct EmergencyCandidate {
        adv_id: u32,
        class_a: String,
        class_b: String,
        proposed_name: String,
        tags_a: Vec<String>,
        tags_b: Vec<String>,
    }
    let mut candidates: Vec<EmergencyCandidate> = Vec::new();

    // Also collect mutation candidates (refused during crisis, tension still > 0.9)
    struct MutationCandidate {
        adv_id: u32,
        class_idx: usize,
        class_name: String,
    }
    let mut mutations: Vec<MutationCandidate> = Vec::new();

    let templates: Vec<ClassTemplate> = base_templates()
        .into_iter()
        .chain(rare_templates())
        .collect();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        if adv.behavior_ledger.tension_accumulator <= 0.9 {
            continue;
        }
        if adv.classes.len() < 2 {
            continue;
        }

        // Check if there's already a pending emergency offer
        let has_pending = state.consolidation_offers.iter().any(|o| {
            o.adventurer_id == adv.id
                && o.deadline_tick <= o.offered_tick + EMERGENCY_CONSOLIDATION_DEADLINE + 50
        });

        if !has_pending {
            // Offer emergency consolidation for the first two non-shame classes
            let non_shame: Vec<(usize, &ClassInstance)> = adv
                .classes
                .iter()
                .enumerate()
                .filter(|(_, c)| !is_shame_class(&c.class_name))
                .collect();

            if non_shame.len() >= 2 {
                let (_, ca) = non_shame[0];
                let (_, cb) = non_shame[1];

                let tags_a = templates
                    .iter()
                    .find(|t| t.class_name == ca.class_name)
                    .map(|t| t.tags.clone())
                    .unwrap_or_default();
                let tags_b = templates
                    .iter()
                    .find(|t| t.class_name == cb.class_name)
                    .map(|t| t.tags.clone())
                    .unwrap_or_default();

                let base_name = generate_consolidated_name(&ca.class_name, &cb.class_name);
                // Emergency merges get defensive/adaptive suffixes
                let proposed_name = format!("{} Survivor", base_name);

                candidates.push(EmergencyCandidate {
                    adv_id: adv.id,
                    class_a: ca.class_name.clone(),
                    class_b: cb.class_name.clone(),
                    proposed_name,
                    tags_a,
                    tags_b,
                });
            }
        } else {
            // Has a pending emergency offer that may have been refused (expired)
            // Check for negative mutation: 20% chance
            // We only mutate if there are non-shame classes to fracture
            let non_shame_idx: Vec<(usize, &ClassInstance)> = adv
                .classes
                .iter()
                .enumerate()
                .filter(|(_, c)| !is_shame_class(&c.class_name))
                .collect();

            if !non_shame_idx.is_empty() {
                mutations.push(MutationCandidate {
                    adv_id: adv.id,
                    class_idx: non_shame_idx[0].0,
                    class_name: non_shame_idx[0].1.class_name.clone(),
                });
            }
        }
    }

    // Create emergency offers
    for cand in candidates {
        let mut combined_tags = cand.tags_a.clone();
        for tag in &cand.tags_b {
            if !combined_tags.contains(tag) {
                combined_tags.push(tag.clone());
            }
        }

        let overlap_score = jaccard_similarity(&cand.tags_a, &cand.tags_b);

        let offer = ConsolidationOffer {
            adventurer_id: cand.adv_id,
            class_a: cand.class_a.clone(),
            class_b: cand.class_b.clone(),
            proposed_name: cand.proposed_name.clone(),
            proposed_tags: combined_tags,
            rarity: SkillRarity::Uncommon,
            offered_tick: tick,
            deadline_tick: tick + EMERGENCY_CONSOLIDATION_DEADLINE,
            times_refused: 0,
            overlap_score,
        };

        state.consolidation_offers.push(offer);

        events.push(WorldEvent::EmergencyConsolidation {
            adventurer_id: cand.adv_id,
            proposed_name: cand.proposed_name.clone(),
        });

        let name = adventurer_name(state, cand.adv_id);
        let entry = format!(
            "Crisis tears at {}. The system offers emergency refuge: merge [{}] and [{}] into [{}]. \
             Accept quickly, or risk fracture.",
            name, cand.class_a, cand.class_b, cand.proposed_name
        );
        state.class_chronicle_entries.push(entry.clone());
        events.push(WorldEvent::ClassChronicleEntry {
            adventurer_id: cand.adv_id,
            entry,
        });
    }

    // Process negative mutations (20% chance)
    for mutation in mutations {
        let roll = lcg_f32(&mut state.rng);
        if roll >= 0.20 {
            continue;
        }

        if let Some(adv) = state
            .adventurers
            .iter_mut()
            .find(|a| a.id == mutation.adv_id)
        {
            if mutation.class_idx < adv.classes.len()
                && adv.classes[mutation.class_idx].class_name == mutation.class_name
            {
                let old_level = adv.classes[mutation.class_idx].level;
                let diminished = format!("{} (Fractured)", mutation.class_name);

                let mut ci = ClassInstance {
                    class_name: diminished.clone(),
                    level: (old_level / 2).max(1),
                    xp: 0.0,
                    xp_to_next: ((old_level / 2).max(1).pow(2)) as f32 * 100.0,
                    stagnation_ticks: 0,
                    skills_granted: Vec::new(),
                    acquired_tick: tick,
                    identity_coherence: 0.3,
                    capstone_required: false,
                    xp_overflow: 0.0,
                    ..Default::default()
                };
                populate_noncombat_growth(&mut ci);
                adv.classes[mutation.class_idx] = ci;

                events.push(WorldEvent::NegativeClassMutation {
                    adventurer_id: mutation.adv_id,
                    original_class: mutation.class_name.clone(),
                    diminished_class: diminished.clone(),
                });

                let name = adventurer_name(state, mutation.adv_id);
                let entry = format!(
                    "The crisis was too much. [{}] fractures under pressure. {} now carries [{}] — \
                     a diminished echo born from unresolved tension.",
                    mutation.class_name, name, diminished
                );
                state.class_chronicle_entries.push(entry.clone());
                events.push(WorldEvent::ClassChronicleEntry {
                    adventurer_id: mutation.adv_id,
                    entry,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Exclusion zone helpers (idea 1.3)
// ---------------------------------------------------------------------------

fn class_family(class_name: &str) -> Option<&'static str> {
    match class_name {
        "Warrior" | "Guardian" | "Commander" => Some("combat_leadership"),
        "Scholar" | "Artisan" => Some("scholarly"),
        "Rogue" | "Scout" => Some("stealth_recon"),
        "Healer" => Some("support"),
        "Ranger" => Some("ranged"),
        "Diplomat" | "Merchant" => Some("social_economy"),
        _ => None,
    }
}

fn exclusion_blocks(family: &str) -> Vec<(&'static str, f32)> {
    match family {
        "combat_leadership" => vec![("combat_leadership", 0.25), ("scholarly", 0.5)],
        "scholarly" => vec![("scholarly", 0.25), ("combat_leadership", 0.5)],
        "stealth_recon" => vec![("stealth_recon", 0.25)],
        "social_economy" => vec![("social_economy", 0.25)],
        _ => vec![],
    }
}

fn is_excluded_by_cooldown(classes: &[ClassInstance], candidate_name: &str) -> bool {
    let candidate_family = match class_family(candidate_name) {
        Some(f) => f,
        None => return false,
    };
    for existing in classes {
        if existing.exclusion_cooldown == 0 { continue; }
        let existing_family = match class_family(&existing.class_name) {
            Some(f) => f,
            None => continue,
        };
        for (blocked_family, factor) in exclusion_blocks(existing_family) {
            if blocked_family == candidate_family {
                let base_cooldown = existing.level * 100;
                let effective_cooldown = (base_cooldown as f32 * factor) as u32;
                if existing.exclusion_cooldown <= effective_cooldown {
                    return true;
                }
            }
        }
    }
    false
}

fn landmark_requirements(class_name: &str) -> &[&str] {
    match class_name {
        "Spellblade" => &["survived_outnumbered_3to1"],
        "Plague Doctor" => &["healed_ally_from_brink"],
        "Warlord" => &["lost_2_party_members_completed_quest"],
        "Shadowmerchant" => &["broke_contract_then_fulfilled_harder"],
        _ => &[],
    }
}

/// Returns 0.4-1.0 based on how recently the class had relevant activity (idea 2.8).
pub fn class_stat_contribution(class: &ClassInstance, ledger: &BehaviorLedger) -> f32 {
    let recent_activity = match class.class_name.as_str() {
        "Warrior" => ledger.recent_melee_combat + ledger.recent_damage_absorbed,
        "Ranger" => ledger.recent_ranged_combat + ledger.recent_areas_explored,
        "Healer" => ledger.recent_healing_given + ledger.recent_allies_supported,
        "Diplomat" => ledger.recent_diplomacy_actions,
        "Merchant" => ledger.recent_trades_completed + ledger.recent_diplomacy_actions * 0.5,
        "Scholar" => ledger.recent_research_performed + ledger.recent_areas_explored * 0.5,
        "Rogue" => ledger.recent_stealth_actions + ledger.recent_melee_combat * 0.5,
        "Artisan" => ledger.recent_items_crafted + ledger.recent_trades_completed * 0.5,
        "Commander" => ledger.recent_units_commanded + ledger.recent_melee_combat * 0.5,
        "Scout" => ledger.recent_areas_explored + ledger.recent_stealth_actions,
        "Guardian" => ledger.recent_damage_absorbed + ledger.recent_allies_supported,
        _ => {
            let fields = [
                ledger.recent_melee_combat, ledger.recent_ranged_combat,
                ledger.recent_healing_given, ledger.recent_diplomacy_actions,
                ledger.recent_trades_completed, ledger.recent_items_crafted,
                ledger.recent_areas_explored, ledger.recent_units_commanded,
                ledger.recent_stealth_actions, ledger.recent_research_performed,
                ledger.recent_damage_absorbed, ledger.recent_allies_supported,
            ];
            fields.iter().cloned().fold(0.0f32, f32::max)
        }
    };
    let t = (recent_activity / 5.0).min(1.0);
    0.4 + 0.6 * t
}

fn cosine_similarity(a: &[f32; 12], b: &[f32; 12]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a < 1e-8 || mag_b < 1e-8 { return 0.0; }
    dot / (mag_a * mag_b)
}

fn get_recent_field(ledger: &BehaviorLedger, idx: usize) -> f32 {
    match idx {
        0 => ledger.recent_melee_combat, 1 => ledger.recent_ranged_combat,
        2 => ledger.recent_healing_given, 3 => ledger.recent_diplomacy_actions,
        4 => ledger.recent_trades_completed, 5 => ledger.recent_items_crafted,
        6 => ledger.recent_areas_explored, 7 => ledger.recent_units_commanded,
        8 => ledger.recent_stealth_actions, 9 => ledger.recent_research_performed,
        10 => ledger.recent_damage_absorbed, 11 => ledger.recent_allies_supported,
        _ => 0.0,
    }
}

fn add_behavior_field(ledger: &mut BehaviorLedger, idx: usize, amount: f32) {
    match idx {
        0 => { ledger.melee_combat += amount; ledger.recent_melee_combat += amount; }
        1 => { ledger.ranged_combat += amount; ledger.recent_ranged_combat += amount; }
        2 => { ledger.healing_given += amount; ledger.recent_healing_given += amount; }
        3 => { ledger.diplomacy_actions += amount; ledger.recent_diplomacy_actions += amount; }
        4 => { ledger.trades_completed += amount; ledger.recent_trades_completed += amount; }
        5 => { ledger.items_crafted += amount; ledger.recent_items_crafted += amount; }
        6 => { ledger.areas_explored += amount; ledger.recent_areas_explored += amount; }
        7 => { ledger.units_commanded += amount; ledger.recent_units_commanded += amount; }
        8 => { ledger.stealth_actions += amount; ledger.recent_stealth_actions += amount; }
        9 => { ledger.research_performed += amount; ledger.recent_research_performed += amount; }
        10 => { ledger.damage_absorbed += amount; ledger.recent_damage_absorbed += amount; }
        11 => { ledger.allies_supported += amount; ledger.recent_allies_supported += amount; }
        _ => {}
    }
}

fn track_landmark_achievements(state: &mut CampaignState, events: &[WorldEvent]) {
    for ev in events.iter() {
        match ev {
            WorldEvent::BattleEnded { result: BattleStatus::Victory, .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Fighting && adv.injury > 50.0 {
                        let lm = "survived_outnumbered_3to1".to_string();
                        if !adv.behavior_ledger.landmark_achievements.contains(&lm) {
                            adv.behavior_ledger.landmark_achievements.push(lm);
                        }
                    }
                }
            }
            WorldEvent::AdventurerRecovered { adventurer_id } => {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adventurer_id) {
                    if adv.injury > 70.0 {
                        let lm = "healed_ally_from_brink".to_string();
                        if !adv.behavior_ledger.landmark_achievements.contains(&lm) {
                            adv.behavior_ledger.landmark_achievements.push(lm);
                        }
                    }
                }
            }
            WorldEvent::AdventurerDied { adventurer_id, .. } => {
                let party_id = state.adventurers.iter()
                    .find(|a| a.id == *adventurer_id)
                    .and_then(|a| a.party_id);
                if let Some(pid) = party_id {
                    let dead_count = state.adventurers.iter()
                        .filter(|a| a.party_id == Some(pid) && a.status == AdventurerStatus::Dead)
                        .count();
                    if dead_count >= 2 {
                        for adv in &mut state.adventurers {
                            if adv.party_id == Some(pid) && adv.status != AdventurerStatus::Dead {
                                let lm = "lost_2_party_members_completed_quest".to_string();
                                if !adv.behavior_ledger.landmark_achievements.contains(&lm) {
                                    adv.behavior_ledger.landmark_achievements.push(lm);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let broken_oath_ids: Vec<u32> = state.oaths.iter()
        .filter(|o| o.broken).map(|o| o.adventurer_id).collect();
    for adv in &mut state.adventurers {
        if broken_oath_ids.contains(&adv.id) && adv.status == AdventurerStatus::OnMission {
            let lm = "broke_contract_then_fulfilled_harder".to_string();
            if !adv.behavior_ledger.landmark_achievements.contains(&lm) {
                adv.behavior_ledger.landmark_achievements.push(lm);
            }
        }
    }
}

fn decay_exclusion_cooldowns(state: &mut CampaignState) {
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead { continue; }
        for class in &mut adv.classes {
            if class.exclusion_cooldown > 0 {
                class.exclusion_cooldown = class.exclusion_cooldown.saturating_sub(1);
            }
        }
    }
}

fn witness_xp_multiplier(state: &CampaignState, adv_idx: usize, class_name: &str) -> f32 {
    let adv = &state.adventurers[adv_idx];
    let my_level = adv.classes.iter()
        .find(|c| c.class_name == class_name)
        .map(|c| c.level).unwrap_or(0);
    let party_id = match adv.party_id {
        Some(p) => p,
        None => return 1.0,
    };
    let mut best_mentor_level = 0u32;
    for (i, other) in state.adventurers.iter().enumerate() {
        if i == adv_idx || other.status == AdventurerStatus::Dead { continue; }
        if other.party_id != Some(party_id) { continue; }
        for c in &other.classes {
            if c.class_name == class_name && c.level > my_level && c.level > best_mentor_level {
                best_mentor_level = c.level;
            }
        }
    }
    if best_mentor_level > my_level {
        let diff = (best_mentor_level - my_level).min(10) as f32;
        1.0 + diff / 10.0
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Stub functions for features added by other agents (not yet implemented)
// ---------------------------------------------------------------------------

fn check_mirror_offers(_state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    // TODO: Mirror class offers based on adventurer observation of allies
}

fn check_oath_locked_classes(_state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    // TODO: Lock/unlock classes based on active oaths
}

fn track_heroic_acts(_state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    // TODO: Track heroic acts for class progression bonuses
}

fn check_rival_classes(_state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    // TODO: Grant rival-reflected classes when adventurers repeatedly face rivals
}

fn check_folk_hero_divergence(_state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    // TODO: Check for folk hero class divergence based on public reputation
}

/// Witness XP multiplier (idea 2.9): adventurers observing allies using a class
/// gain a small XP bonus for that class. Returns 1.0 (no bonus) as stub.
fn check_campaign_skill_hooks(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect grants first to avoid borrow issues.
    struct CampaignSkillGrant {
        adv_id: u32,
        skill_name: String,
        trigger_system: String,
        class_name: String,
    }
    let mut grants: Vec<CampaignSkillGrant> = Vec::new();

    // Check campaign subsystem conditions
    let has_bankruptcy = !state.bankruptcy_cascade.defaults_this_cycle.is_empty();
    let has_civil_war = !state.civil_wars.is_empty();
    let has_plague = !state.plague_vectors.is_empty();
    let has_heist_success = events.iter().any(|e| matches!(e, WorldEvent::HeistSucceeded { .. }));
    let max_divine_favor = state.divine_favor.iter().map(|d| d.divine_favor).fold(0.0f32, f32::max);
    let threat_power = state.world_threat_clock.power;

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        // Bankruptcy → [Debt Remembrance] for Merchant-class
        if has_bankruptcy
            && adv.classes.iter().any(|c| c.class_name == "Merchant" || c.class_name == "Trade Prince" || c.class_name == "Shadowmerchant")
            && !state.campaign_skills_granted.iter().any(|&(id, ref s)| id == adv.id && s == "[Debt Remembrance]")
        {
            grants.push(CampaignSkillGrant {
                adv_id: adv.id,
                skill_name: "[Debt Remembrance]".to_string(),
                trigger_system: "bankruptcy_cascade".to_string(),
                class_name: adv.classes.iter()
                    .find(|c| c.class_name == "Merchant" || c.class_name == "Trade Prince" || c.class_name == "Shadowmerchant")
                    .map(|c| c.class_name.clone())
                    .unwrap_or_default(),
            });
        }

        // Civil war → [Oathbreaker's Resolve] for Commander-class
        if has_civil_war
            && adv.classes.iter().any(|c| c.class_name == "Commander" || c.class_name == "High Marshal" || c.class_name == "Warlord")
            && !state.campaign_skills_granted.iter().any(|&(id, ref s)| id == adv.id && s == "[Oathbreaker's Resolve]")
        {
            grants.push(CampaignSkillGrant {
                adv_id: adv.id,
                skill_name: "[Oathbreaker's Resolve]".to_string(),
                trigger_system: "civil_war".to_string(),
                class_name: adv.classes.iter()
                    .find(|c| c.class_name == "Commander" || c.class_name == "High Marshal" || c.class_name == "Warlord")
                    .map(|c| c.class_name.clone())
                    .unwrap_or_default(),
            });
        }

        // Plague → [Plague Resilience] for Healer-class
        if has_plague
            && adv.classes.iter().any(|c| c.class_name == "Healer" || c.class_name == "Archon" || c.class_name == "Plague Doctor")
            && !state.campaign_skills_granted.iter().any(|&(id, ref s)| id == adv.id && s == "[Plague Resilience]")
        {
            grants.push(CampaignSkillGrant {
                adv_id: adv.id,
                skill_name: "[Plague Resilience]".to_string(),
                trigger_system: "plague_vectors".to_string(),
                class_name: adv.classes.iter()
                    .find(|c| c.class_name == "Healer" || c.class_name == "Archon" || c.class_name == "Plague Doctor")
                    .map(|c| c.class_name.clone())
                    .unwrap_or_default(),
            });
        }

        // Heist success → [Shadow's Luck] for Rogue-class
        if has_heist_success
            && adv.classes.iter().any(|c| c.class_name == "Rogue" || c.class_name == "Phantom" || c.class_name == "Shadowmerchant")
            && !state.campaign_skills_granted.iter().any(|&(id, ref s)| id == adv.id && s == "[Shadow's Luck]")
        {
            grants.push(CampaignSkillGrant {
                adv_id: adv.id,
                skill_name: "[Shadow's Luck]".to_string(),
                trigger_system: "heist_planning".to_string(),
                class_name: adv.classes.iter()
                    .find(|c| c.class_name == "Rogue" || c.class_name == "Phantom" || c.class_name == "Shadowmerchant")
                    .map(|c| c.class_name.clone())
                    .unwrap_or_default(),
            });
        }

        // Divine favor > 30 → [Blessed Touch] for any adventurer with healing behavior
        if max_divine_favor > 30.0
            && adv.behavior_ledger.healing_given > 5.0
            && !state.campaign_skills_granted.iter().any(|&(id, ref s)| id == adv.id && s == "[Blessed Touch]")
        {
            grants.push(CampaignSkillGrant {
                adv_id: adv.id,
                skill_name: "[Blessed Touch]".to_string(),
                trigger_system: "divine_favor".to_string(),
                class_name: adv.classes.first().map(|c| c.class_name.clone()).unwrap_or_default(),
            });
        }

        // Threat clock power > 0.5 → [Doom Sense] for high tension adventurers
        if threat_power > 0.5
            && adv.behavior_ledger.tension_accumulator > 0.5
            && !state.campaign_skills_granted.iter().any(|&(id, ref s)| id == adv.id && s == "[Doom Sense]")
        {
            grants.push(CampaignSkillGrant {
                adv_id: adv.id,
                skill_name: "[Doom Sense]".to_string(),
                trigger_system: "threat_clock".to_string(),
                class_name: adv.classes.first().map(|c| c.class_name.clone()).unwrap_or_default(),
            });
        }
    }

    // Apply grants
    for grant in grants {
        // Track to prevent duplicates
        state.campaign_skills_granted.push((grant.adv_id, grant.skill_name.clone()));

        // Add skill to the adventurer's matching class
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == grant.adv_id) {
            let target_class = adv.classes.iter_mut()
                .find(|c| c.class_name == grant.class_name);
            if let Some(class) = target_class {
                class.skills_granted.push(GrantedSkill {
                    skill_name: grant.skill_name.clone(),
                    granted_at_level: class.level,
                    rarity: SkillRarity::Rare,
                    from_class: class.class_name.clone(),
                    ability_dsl: None,
                    effect_description: format!(
                        "A rare skill triggered by campaign event: {}",
                        grant.trigger_system
                    ),
                    inheritance_status: String::new(),
                    affinity_tags: affinity_tags_for_class(&class.class_name),
                    empowered: false,
                    skill_effect: None,
                    skill_condition: None,
                });
            }
        }

        // Feature 4: Narrative announcement for Rare campaign skills
        if let Some(adv) = state.adventurers.iter().find(|a| a.id == grant.adv_id) {
            let name = adv.name.clone();
            let entry = format!(
                "The world itself has marked {}. Through {} they have earned {} — \
                 a rare power forged not in training, but in survival.",
                name, grant.trigger_system, grant.skill_name
            );
            state.class_chronicle_entries.push(entry.clone());
            events.push(WorldEvent::ClassChronicleEntry {
                adventurer_id: grant.adv_id,
                entry,
            });
        }

        events.push(WorldEvent::CampaignSkillGranted {
            adventurer_id: grant.adv_id,
            skill_name: grant.skill_name,
            trigger_system: grant.trigger_system,
        });
    }
}

// ---------------------------------------------------------------------------
// Feature 3: Capstone Skills as Synthesis (idea 3.5)
// ---------------------------------------------------------------------------

/// Generate a capstone skill by combining the top two behavior axes.
/// Called at level 25 from check_skill_grants when threshold == 25.
fn generate_capstone_skill(ledger: &BehaviorLedger, class_name: &str) -> (String, String) {
    let axes: [(&str, f32); 12] = [
        ("melee", ledger.melee_combat),
        ("ranged", ledger.ranged_combat),
        ("healing", ledger.healing_given),
        ("diplomacy", ledger.diplomacy_actions),
        ("trades", ledger.trades_completed),
        ("crafting", ledger.items_crafted),
        ("areas_explored", ledger.areas_explored),
        ("units_commanded", ledger.units_commanded),
        ("stealth", ledger.stealth_actions),
        ("research", ledger.research_performed),
        ("damage_absorbed", ledger.damage_absorbed),
        ("allies_supported", ledger.allies_supported),
    ];

    let mut sorted: Vec<(&str, f32)> = axes.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top1 = sorted.get(0).map(|x| x.0).unwrap_or("melee");
    let top2 = sorted.get(1).map(|x| x.0).unwrap_or("damage_absorbed");

    let skill_name = match (top1, top2) {
        ("melee", "damage_absorbed") | ("damage_absorbed", "melee") => "[Bulwark of the Fallen]".to_string(),
        ("melee", "stealth") | ("stealth", "melee") => "[Ghost Blade]".to_string(),
        ("healing", "allies_supported") | ("allies_supported", "healing") => "[Life Eternal]".to_string(),
        ("diplomacy", "units_commanded") | ("units_commanded", "diplomacy") => "[Voice of Nations]".to_string(),
        ("trades", "crafting") | ("crafting", "trades") => "[Master of All Trades]".to_string(),
        ("research", "areas_explored") | ("areas_explored", "research") => "[Cartographer of the Unknown]".to_string(),
        _ => format!("[Apex of {}]", class_name),
    };

    let effect = format!(
        "Capstone synthesis of {} and {} — the culmination of a lifetime's mastery in the {} class.",
        top1, top2, class_name
    );

    (skill_name, effect)
}
