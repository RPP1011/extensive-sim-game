//! TWI-inspired class system — classes are EARNED from behavior, not chosen.
//!
//! The system watches what adventurers do and grants classes when behavioral
//! fingerprints match class templates.  Each class levels independently via
//! quadratic XP curves, and skills are granted at level thresholds.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    lcg_f32, AdventurerStatus, BehaviorLedger, CampaignState, ClassInstance, ClassTemplate,
    ConsolidationOffer, GrantedSkill, SkillRarity,
};

/// Class system fires every 100 ticks (10 s game time).
const CLASS_TICK_INTERVAL: u64 = 100;

/// Consolidation check fires every 500 ticks.
const CONSOLIDATION_INTERVAL: u64 = 500;

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
const STAGNATION_HALF: u32 = 500;
/// Stagnation threshold where XP gain is frozen.
const STAGNATION_FREEZE: u32 = 1000;

/// XP trickle between overlapping classes (15%).
const RESONANCE_TRICKLE: f32 = 0.15;

/// Level thresholds at which skills are granted.
const SKILL_THRESHOLDS: &[u32] = &[3, 5, 7, 10, 15, 20, 25];

// ---------------------------------------------------------------------------
// Class templates
// ---------------------------------------------------------------------------

fn base_templates() -> Vec<ClassTemplate> {
    vec![
        ct("Warrior", &[("melee_combat", 0.4), ("damage_absorbed", 0.3)], 0.5, &["combat"]),
        ct("Ranger", &[("ranged_combat", 0.4), ("areas_explored", 0.3)], 0.5, &["combat", "exploration"]),
        ct("Healer", &[("healing_given", 0.5), ("allies_supported", 0.3)], 0.5, &["support"]),
        ct("Diplomat", &[("diplomacy_actions", 0.5), ("units_commanded", 0.2)], 0.5, &["social"]),
        ct("Merchant", &[("trades_completed", 0.5), ("diplomacy_actions", 0.2)], 0.5, &["economy"]),
        ct("Scholar", &[("research_performed", 0.5), ("areas_explored", 0.2)], 0.5, &["knowledge"]),
        ct("Rogue", &[("stealth_actions", 0.4), ("melee_combat", 0.2)], 0.5, &["stealth", "combat"]),
        ct("Artisan", &[("items_crafted", 0.5), ("trades_completed", 0.2)], 0.5, &["crafting"]),
        ct("Commander", &[("units_commanded", 0.4), ("melee_combat", 0.2)], 0.5, &["leadership"]),
        ct("Scout", &[("areas_explored", 0.4), ("stealth_actions", 0.3)], 0.5, &["exploration", "stealth"]),
        ct("Guardian", &[("damage_absorbed", 0.4), ("allies_supported", 0.3)], 0.5, &["defense"]),
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
    check_class_acquisition(state, events);
    process_class_xp(state, events);
    check_skill_grants(state, events);
    update_stagnation(state, events);
    decay_recent_window(state);

    // Consolidation & evolution run on a slower cadence
    if state.tick % CONSOLIDATION_INTERVAL == 0 {
        check_consolidation_offers(state, events);
        check_evolution(state, events);
    }
}

// ---------------------------------------------------------------------------
// Behavior ledger update
// ---------------------------------------------------------------------------

/// Scan recent WorldEvents and increment appropriate behavior counters.
fn update_behavior_ledgers(state: &mut CampaignState, events: &[WorldEvent]) {
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
            // Healing
            WorldEvent::AdventurerRecovered { adventurer_id } => {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *adventurer_id) {
                    // The healer is whoever supported this recovery; credit the adventurer
                    adv.behavior_ledger.healing_given += 0.5;
                    adv.behavior_ledger.recent_healing_given += 0.5;
                }
            }
            // Diplomacy
            WorldEvent::AgreementFormed { .. }
            | WorldEvent::WarCeasefire { .. }
            | WorldEvent::FactionRelationChanged { .. } => {
                // Credit adventurers on diplomatic missions (Diplomat archetype or idle)
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead {
                        continue;
                    }
                    if adv.archetype == "diplomat" || adv.archetype == "bard" {
                        adv.behavior_ledger.diplomacy_actions += 1.0;
                        adv.behavior_ledger.recent_diplomacy_actions += 1.0;
                    }
                }
            }
            // Trade/economy
            WorldEvent::CaravanCompleted { .. }
            | WorldEvent::TradeProfitMade { .. }
            | WorldEvent::MerchantPurchase { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead {
                        continue;
                    }
                    if adv.archetype == "merchant" {
                        adv.behavior_ledger.trades_completed += 1.0;
                        adv.behavior_ledger.recent_trades_completed += 1.0;
                    }
                }
            }
            // Crafting
            WorldEvent::ItemCrafted { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead {
                        continue;
                    }
                    if adv.archetype == "artisan" || adv.archetype == "blacksmith" {
                        adv.behavior_ledger.items_crafted += 1.0;
                        adv.behavior_ledger.recent_items_crafted += 1.0;
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
                        adv.behavior_ledger.units_commanded += 1.0;
                        adv.behavior_ledger.recent_units_commanded += 1.0;
                    }
                }
            }
            // Stealth/espionage
            WorldEvent::IntelGathered { spy_id, .. } => {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *spy_id) {
                    adv.behavior_ledger.stealth_actions += 1.0;
                    adv.behavior_ledger.recent_stealth_actions += 1.0;
                }
            }
            WorldEvent::HeistSucceeded { .. } | WorldEvent::HeistPhaseAdvanced { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead {
                        continue;
                    }
                    if adv.archetype == "rogue" || adv.archetype == "thief" || adv.archetype == "assassin" {
                        adv.behavior_ledger.stealth_actions += 1.0;
                        adv.behavior_ledger.recent_stealth_actions += 1.0;
                    }
                }
            }
            // Research/archives
            WorldEvent::ResearchCompleted { .. } | WorldEvent::KnowledgeGained { .. } => {
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Dead {
                        continue;
                    }
                    if adv.archetype == "mage" || adv.archetype == "scholar" || adv.archetype == "sage" {
                        adv.behavior_ledger.research_performed += 1.0;
                        adv.behavior_ledger.recent_research_performed += 1.0;
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
                    adv.behavior_ledger.allies_supported += 1.0;
                    adv.behavior_ledger.recent_allies_supported += 1.0;
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
// Class acquisition
// ---------------------------------------------------------------------------

fn check_class_acquisition(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let templates: Vec<ClassTemplate> = base_templates()
        .into_iter()
        .chain(rare_templates())
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

        for tmpl in &templates {
            // Warlord only during crisis
            if tmpl.class_name == "Warlord" && !in_crisis {
                continue;
            }
            // Already have this class?
            if adv.classes.iter().any(|c| c.class_name == tmpl.class_name) {
                continue;
            }
            let s = score_template(&fp, tmpl);
            if s >= tmpl.threshold {
                adv.classes.push(ClassInstance {
                    class_name: tmpl.class_name.clone(),
                    level: 1,
                    xp: 0.0,
                    xp_to_next: 100.0, // 1^2 * 100
                    stagnation_ticks: 0,
                    skills_granted: Vec::new(),
                    acquired_tick: tick,
                    identity_coherence: 1.0,
                });
                events.push(WorldEvent::ClassGranted {
                    adventurer_id: adv.id,
                    class_name: tmpl.class_name.clone(),
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// XP processing
// ---------------------------------------------------------------------------

fn process_class_xp(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let templates: Vec<ClassTemplate> = base_templates()
        .into_iter()
        .chain(rare_templates())
        .collect();

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let fp = behavioral_fingerprint(&adv.behavior_ledger);

        // Collect XP per class based on behavior match
        let num_classes = adv.classes.len();
        let mut xp_gains = vec![0.0f32; num_classes];

        for (i, class) in adv.classes.iter().enumerate() {
            // Find template
            let tmpl = templates.iter().find(|t| t.class_name == class.class_name);
            if let Some(tmpl) = tmpl {
                let raw_xp = score_template(&fp, tmpl) * 10.0;
                // Apply stagnation penalty
                let penalty = if class.stagnation_ticks >= STAGNATION_FREEZE {
                    0.0
                } else if class.stagnation_ticks >= STAGNATION_HALF {
                    0.5
                } else {
                    1.0
                };
                xp_gains[i] = raw_xp * penalty;
            }
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

        // Apply XP and check level-ups
        for (i, class) in adv.classes.iter_mut().enumerate() {
            let gained = xp_gains[i] + trickle[i];
            if gained > 0.0 {
                class.xp += gained;
                class.stagnation_ticks = 0;

                // Level up loop
                while class.xp >= class.xp_to_next {
                    class.xp -= class.xp_to_next;
                    class.level += 1;
                    class.xp_to_next = (class.level * class.level) as f32 * 100.0;
                    events.push(WorldEvent::ClassLevelUp {
                        adventurer_id: adv.id,
                        class_name: class.class_name.clone(),
                        new_level: class.level,
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Skill grants
// ---------------------------------------------------------------------------

fn check_skill_grants(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        for class in &mut adv.classes {
            for &threshold in SKILL_THRESHOLDS {
                if class.level >= threshold
                    && !class
                        .skills_granted
                        .iter()
                        .any(|s| s.granted_at_level == threshold)
                {
                    let rarity = match threshold {
                        3 | 5 => SkillRarity::Common,
                        7 | 10 => SkillRarity::Uncommon,
                        15 | 20 => SkillRarity::Rare,
                        25 => SkillRarity::Capstone,
                        _ => SkillRarity::Common,
                    };
                    let rarity_str = match &rarity {
                        SkillRarity::Common => "Common",
                        SkillRarity::Uncommon => "Uncommon",
                        SkillRarity::Rare => "Rare",
                        SkillRarity::Capstone => "Capstone",
                        SkillRarity::Unique => "Unique",
                    };
                    let skill_name = format!(
                        "{} {} Lv{}",
                        class.class_name, rarity_str, threshold
                    );
                    let effect = format!(
                        "A {} skill earned at level {} of the {} class.",
                        rarity_str.to_lowercase(),
                        threshold,
                        class.class_name
                    );

                    class.skills_granted.push(GrantedSkill {
                        skill_name: skill_name.clone(),
                        granted_at_level: threshold,
                        rarity,
                        from_class: class.class_name.clone(),
                        ability_dsl: None,
                        effect_description: effect,
                    });

                    events.push(WorldEvent::SkillGrantedByClass {
                        adventurer_id: adv.id,
                        skill_name,
                        rarity: rarity_str.to_string(),
                        class_name: class.class_name.clone(),
                    });
                }
            }
        }
    }
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
            // here we increment for classes that got no XP this tick.
            if class.stagnation_ticks > 0 {
                class.stagnation_ticks += 1;
            } else {
                // Will be set to 1 next tick if no XP is gained
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

            // Gather top 2 skills from each parent before removing them
            let adv = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == cand.adv_id)
                .unwrap();

            let mut inherited_skills: Vec<GrantedSkill> = Vec::new();

            // Collect skills from class_a (top 2 by level granted)
            if let Some(ca) = adv.classes.iter().find(|c| c.class_name == cand.class_a) {
                let mut skills: Vec<&GrantedSkill> = ca.skills_granted.iter().collect();
                skills.sort_by(|a, b| b.granted_at_level.cmp(&a.granted_at_level));
                for s in skills.into_iter().take(2) {
                    inherited_skills.push(s.clone());
                }
            }
            // Collect skills from class_b (top 2 by level granted)
            if let Some(cb) = adv.classes.iter().find(|c| c.class_name == cand.class_b) {
                let mut skills: Vec<&GrantedSkill> = cb.skills_granted.iter().collect();
                skills.sort_by(|a, b| b.granted_at_level.cmp(&a.granted_at_level));
                for s in skills.into_iter().take(2) {
                    inherited_skills.push(s.clone());
                }
            }

            // Remove both parent classes
            adv.classes
                .retain(|c| c.class_name != cand.class_a && c.class_name != cand.class_b);

            // Create new consolidated class
            adv.classes.push(ClassInstance {
                class_name: proposed_name.clone(),
                level: new_level,
                xp: 0.0,
                xp_to_next: (new_level * new_level) as f32 * 100.0,
                stagnation_ticks: 0,
                skills_granted: inherited_skills,
                acquired_tick: tick,
                identity_coherence: 1.0,
            });

            events.push(WorldEvent::ClassConsolidated {
                adventurer_id: cand.adv_id,
                from_a: cand.class_a,
                from_b: cand.class_b,
                into: proposed_name,
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
            class.xp_to_next = (class.level * class.level) as f32 * 100.0;
            class.acquired_tick = tick; // Mark evolution tick

            events.push(WorldEvent::ClassEvolved {
                adventurer_id: adv.id,
                from_class: from_name,
                to_class: to_name,
            });
        }
    }
}
