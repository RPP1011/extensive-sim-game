//! Adventurer backstory generation system.
//!
//! Generates deterministic backstories for adventurers at recruitment time,
//! using the campaign's LCG RNG to pick birthplace, motivation, flaw, and
//! a personal quest hook drawn from world state (factions and regions).

use serde::{Deserialize, Serialize};

use crate::headless_campaign::state::*;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A generated backstory attached to an adventurer at recruitment time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Backstory {
    /// Region name from the world template.
    pub birthplace: String,
    /// Core motivation driving the adventurer.
    pub motivation: BackstoryMotivation,
    /// Character flaw that colours decision-making.
    pub flaw: BackstoryFlaw,
    /// Short narrative hook tied to their backstory.
    pub personal_quest_hook: String,
    /// Faction this adventurer has a personal grievance with.
    pub rival_faction_id: Option<usize>,
    /// Region this adventurer hails from.
    pub hometown_region_id: Option<usize>,
}

/// Core motivation — weighted by archetype at generation time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackstoryMotivation {
    Revenge,
    Glory,
    Redemption,
    Greed,
    Duty,
    Curiosity,
    Survival,
    Legacy,
}

impl BackstoryMotivation {
    pub const ALL: [BackstoryMotivation; 8] = [
        Self::Revenge,
        Self::Glory,
        Self::Redemption,
        Self::Greed,
        Self::Duty,
        Self::Curiosity,
        Self::Survival,
        Self::Legacy,
    ];
}

/// Character flaw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackstoryFlaw {
    Reckless,
    Distrustful,
    Greedy,
    Proud,
    Haunted,
    Impulsive,
    Stubborn,
    Cowardly,
}

impl BackstoryFlaw {
    pub const ALL: [BackstoryFlaw; 8] = [
        Self::Reckless,
        Self::Distrustful,
        Self::Greedy,
        Self::Proud,
        Self::Haunted,
        Self::Impulsive,
        Self::Stubborn,
        Self::Cowardly,
    ];
}

// ---------------------------------------------------------------------------
// Archetype → motivation weights
// ---------------------------------------------------------------------------

/// Returns unnormalized weight table (len 8) for motivation sampling.
fn motivation_weights(archetype: &str) -> [f32; 8] {
    // Order: Revenge, Glory, Redemption, Greed, Duty, Curiosity, Survival, Legacy
    match archetype {
        // Warriors lean Glory / Duty
        "knight" | "paladin" | "guardian" | "tank" =>
            [1.0, 4.0, 2.0, 0.5, 4.0, 0.5, 1.0, 2.0],
        // Berserkers lean Revenge / Survival
        "berserker" =>
            [4.0, 3.0, 1.0, 1.0, 1.0, 0.5, 3.0, 1.0],
        // Rogues lean Greed / Curiosity
        "rogue" | "assassin" =>
            [1.0, 1.0, 1.0, 4.0, 0.5, 3.0, 2.0, 0.5],
        // Mages lean Curiosity / Legacy
        "mage" | "warlock" | "necromancer" | "artificer" =>
            [0.5, 1.0, 1.0, 1.0, 0.5, 4.0, 1.0, 4.0],
        // Healers lean Duty / Redemption
        "cleric" | "druid" | "shaman" =>
            [0.5, 1.0, 3.0, 0.5, 4.0, 1.0, 1.0, 2.0],
        // Support / hybrid
        "bard" =>
            [1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0],
        "ranger" =>
            [1.0, 1.5, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0],
        "monk" =>
            [0.5, 1.0, 3.0, 0.5, 3.0, 2.0, 1.0, 2.0],
        _ =>
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
}

// ---------------------------------------------------------------------------
// Quest hook templates
// ---------------------------------------------------------------------------

/// Revenge templates.
const REVENGE_TEMPLATES: &[&str] = &[
    "{name} seeks vengeance against {faction} for the destruction of {region}",
    "{name} hunts the {faction} warlord who murdered their mentor in {region}",
    "{name} swore a blood oath against {faction} after losing everything in {region}",
];

/// Glory templates.
const GLORY_TEMPLATES: &[&str] = &[
    "{name} left {region} to prove their worth on the battlefield",
    "{name} seeks to become the greatest champion {region} has ever known",
    "{name} craves the glory that eluded their family in {region}",
];

/// Redemption templates.
const REDEMPTION_TEMPLATES: &[&str] = &[
    "{name} hopes to atone for past failures in {region}",
    "{name} carries the guilt of {region}'s fall and seeks to make amends",
    "{name} was exiled from {region} and seeks a second chance",
];

/// Greed templates.
const GREED_TEMPLATES: &[&str] = &[
    "{name} heard rumours of ancient treasure hidden near {region}",
    "{name} left {region} chasing fortunes the honest life never provided",
    "{name} plans to buy back their family's lands in {region} — at any cost",
];

/// Duty templates.
const DUTY_TEMPLATES: &[&str] = &[
    "{name} serves {region} out of unwavering loyalty to its people",
    "{name} was charged by the elders of {region} to defend the realm",
    "{name} carries orders from {region}'s last standing commander",
];

/// Curiosity templates.
const CURIOSITY_TEMPLATES: &[&str] = &[
    "{name} left {region} to unravel the mysteries of the wider world",
    "{name} follows ancient texts that point to secrets beyond {region}",
    "{name} is drawn by strange phenomena that have spread from {region}",
];

/// Survival templates.
const SURVIVAL_TEMPLATES: &[&str] = &[
    "{name} fled {region} when {faction} overran their homeland",
    "{name} has no home since {region} fell — the guild is all that remains",
    "{name} fights because returning to {region} means certain death",
];

/// Legacy templates.
const LEGACY_TEMPLATES: &[&str] = &[
    "{name} carries the banner of a forgotten order that once protected {region}",
    "{name} aims to restore the legacy of {region}'s fallen heroes",
    "{name} follows in the footsteps of their ancestors who once ruled {region}",
];

fn templates_for(motivation: BackstoryMotivation) -> &'static [&'static str] {
    match motivation {
        BackstoryMotivation::Revenge => REVENGE_TEMPLATES,
        BackstoryMotivation::Glory => GLORY_TEMPLATES,
        BackstoryMotivation::Redemption => REDEMPTION_TEMPLATES,
        BackstoryMotivation::Greed => GREED_TEMPLATES,
        BackstoryMotivation::Duty => DUTY_TEMPLATES,
        BackstoryMotivation::Curiosity => CURIOSITY_TEMPLATES,
        BackstoryMotivation::Survival => SURVIVAL_TEMPLATES,
        BackstoryMotivation::Legacy => LEGACY_TEMPLATES,
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Generate a backstory for an adventurer using the campaign's deterministic RNG.
///
/// Reads world regions and factions from `state` to produce contextual hooks.
/// Mutates only `rng` (passed separately to avoid borrow conflicts).
pub fn generate_backstory(
    name: &str,
    archetype: &str,
    regions: &[Region],
    factions: &[FactionState],
    rng: &mut u64,
) -> Backstory {
    // --- Pick birthplace region ---
    let hometown_region_id = if regions.is_empty() {
        None
    } else {
        Some((lcg_next(rng) as usize) % regions.len())
    };
    let birthplace = hometown_region_id
        .and_then(|id| regions.get(id))
        .map(|r| r.name.clone())
        .unwrap_or_else(|| "the hinterlands".into());

    // --- Sample motivation (weighted by archetype) ---
    let weights = motivation_weights(archetype);
    let motivation = weighted_pick(&BackstoryMotivation::ALL, &weights, rng);

    // --- Sample flaw (uniform) ---
    let flaw_idx = (lcg_next(rng) as usize) % BackstoryFlaw::ALL.len();
    let flaw = BackstoryFlaw::ALL[flaw_idx];

    // --- Rival faction (30% chance, weighted toward hostile factions) ---
    let rival_faction_id = pick_rival_faction(factions, rng);

    // --- Build quest hook from templates ---
    let templates = templates_for(motivation);
    let tmpl_idx = (lcg_next(rng) as usize) % templates.len();
    let template = templates[tmpl_idx];

    let faction_name = rival_faction_id
        .and_then(|fid| factions.iter().find(|f| f.id == fid))
        .map(|f| f.name.clone())
        .or_else(|| factions.first().map(|f| f.name.clone()))
        .unwrap_or_else(|| "the enemy".into());

    let region_name = &birthplace;
    let personal_quest_hook = template
        .replace("{name}", name)
        .replace("{faction}", &faction_name)
        .replace("{region}", region_name);

    Backstory {
        birthplace,
        motivation,
        flaw,
        personal_quest_hook,
        rival_faction_id,
        hometown_region_id: hometown_region_id.and_then(|idx| regions.get(idx).map(|r| r.id)),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Weighted random pick from a slice, using the campaign LCG.
fn weighted_pick<T: Copy>(items: &[T], weights: &[f32], rng: &mut u64) -> T {
    debug_assert_eq!(items.len(), weights.len());
    let total: f32 = weights.iter().sum();
    let mut roll = lcg_f32(rng) * total;
    for (i, &w) in weights.iter().enumerate() {
        roll -= w;
        if roll <= 0.0 {
            return items[i];
        }
    }
    *items.last().unwrap()
}

/// 30% chance to assign a rival faction, weighted toward hostile/at-war factions.
fn pick_rival_faction(factions: &[FactionState], rng: &mut u64) -> Option<usize> {
    if factions.is_empty() {
        return None;
    }
    let roll = lcg_f32(rng);
    if roll > 0.30 {
        return None;
    }
    // Build weights: hostile/at-war factions are 3× more likely.
    let weights: Vec<f32> = factions
        .iter()
        .map(|f| match f.diplomatic_stance {
            DiplomaticStance::AtWar => 4.0,
            DiplomaticStance::Hostile => 3.0,
            DiplomaticStance::Neutral => 1.0,
            DiplomaticStance::Friendly | DiplomaticStance::Coalition => 0.2,
        })
        .collect();
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut pick = lcg_f32(rng) * total;
    for (i, &w) in weights.iter().enumerate() {
        pick -= w;
        if pick <= 0.0 {
            return Some(factions[i].id);
        }
    }
    Some(factions.last().unwrap().id)
}

// ---------------------------------------------------------------------------
// History tag helpers (called from progression_triggers)
// ---------------------------------------------------------------------------

/// Check if a quest was completed in the adventurer's hometown region and
/// increment the `hometown_quest` history tag. Returns true if incremented.
pub fn check_hometown_quest(
    adventurer: &mut Adventurer,
    quest_region_id: Option<usize>,
) -> bool {
    if let (Some(backstory), Some(quest_rid)) = (&adventurer.backstory, quest_region_id) {
        if backstory.hometown_region_id == Some(quest_rid) {
            let count = adventurer.history_tags.entry("hometown_quest".into()).or_insert(0);
            *count += 1;
            return true;
        }
    }
    false
}

/// Check if a battle involved the adventurer's rival faction and increment
/// the `rival_confrontation` history tag. Returns true if incremented.
pub fn check_rival_confrontation(
    adventurer: &mut Adventurer,
    battle_faction_id: Option<usize>,
) -> bool {
    if let (Some(backstory), Some(bfid)) = (&adventurer.backstory, battle_faction_id) {
        if backstory.rival_faction_id == Some(bfid) {
            let count = adventurer.history_tags.entry("rival_confrontation".into()).or_insert(0);
            *count += 1;
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_regions() -> Vec<Region> {
        vec![
            Region {
                id: 0,
                name: "Ashvale".into(),
                owner_faction_id: 0,
                neighbors: vec![1],
                unrest: 10.0,
                control: 80.0,
                threat_level: 20.0,
                visibility: 0.5,
                population: 500,
                civilian_morale: 50.0,
                tax_rate: 0.1,
                growth_rate: 0.0,
            },
            Region {
                id: 1,
                name: "Frostmere".into(),
                owner_faction_id: 1,
                neighbors: vec![0],
                unrest: 30.0,
                control: 60.0,
                threat_level: 40.0,
                visibility: 0.3,
                population: 400,
                civilian_morale: 40.0,
                tax_rate: 0.15,
                growth_rate: 0.0,
            },
        ]
    }

    fn make_factions() -> Vec<FactionState> {
        vec![
            FactionState {
                id: 0,
                name: "Iron Pact".into(),
                relationship_to_guild: 30.0,
                military_strength: 50.0,
                max_military_strength: 100.0,
                territory_size: 3,
                diplomatic_stance: DiplomaticStance::Neutral,
                coalition_member: false,
                at_war_with: vec![],
                has_guild: false,
                guild_adventurer_count: 0,
                recent_actions: vec![],
            },
            FactionState {
                id: 1,
                name: "Shadow Conclave".into(),
                relationship_to_guild: -40.0,
                military_strength: 70.0,
                max_military_strength: 100.0,
                territory_size: 4,
                diplomatic_stance: DiplomaticStance::Hostile,
                coalition_member: false,
                at_war_with: vec![],
                has_guild: false,
                guild_adventurer_count: 0,
                recent_actions: vec![],
            },
        ]
    }

    #[test]
    fn backstory_generation_deterministic() {
        let regions = make_regions();
        let factions = make_factions();
        let mut rng1 = 42u64;
        let mut rng2 = 42u64;
        let b1 = generate_backstory("Alaric", "knight", &regions, &factions, &mut rng1);
        let b2 = generate_backstory("Alaric", "knight", &regions, &factions, &mut rng2);

        assert_eq!(b1.birthplace, b2.birthplace);
        assert_eq!(b1.motivation, b2.motivation);
        assert_eq!(b1.flaw, b2.flaw);
        assert_eq!(b1.personal_quest_hook, b2.personal_quest_hook);
        assert_eq!(b1.rival_faction_id, b2.rival_faction_id);
        assert_eq!(b1.hometown_region_id, b2.hometown_region_id);
    }

    #[test]
    fn backstory_contains_name_and_region() {
        let regions = make_regions();
        let factions = make_factions();
        let mut rng = 12345u64;
        let b = generate_backstory("Brynn", "rogue", &regions, &factions, &mut rng);

        assert!(b.personal_quest_hook.contains("Brynn"));
        // Birthplace should be one of our regions
        assert!(b.birthplace == "Ashvale" || b.birthplace == "Frostmere");
    }

    #[test]
    fn backstory_empty_world_fallback() {
        let mut rng = 99u64;
        let b = generate_backstory("Cira", "mage", &[], &[], &mut rng);

        assert_eq!(b.birthplace, "the hinterlands");
        assert!(b.rival_faction_id.is_none());
        assert!(b.hometown_region_id.is_none());
        assert!(b.personal_quest_hook.contains("Cira"));
    }

    #[test]
    fn hometown_quest_tag_increments() {
        let regions = make_regions();
        let factions = make_factions();
        let mut rng = 77u64;
        let backstory = generate_backstory("Daven", "knight", &regions, &factions, &mut rng);
        let hometown = backstory.hometown_region_id;

        let mut adv = Adventurer {
            id: 1,
            name: "Daven".into(),
            archetype: "knight".into(),
            level: 1,
            xp: 0,
            stats: AdventurerStats::default(),
            equipment: Equipment::default(),
            traits: vec![],
            status: AdventurerStatus::Idle,
            loyalty: 50.0,
            stress: 0.0,
            fatigue: 0.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 70.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),
            backstory: Some(backstory),
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: crate::headless_campaign::state::DiseaseStatus::Healthy,
        };

        // Quest in a different region should not increment
        let other_region = if hometown == Some(0) { Some(1) } else { Some(0) };
        assert!(!check_hometown_quest(&mut adv, other_region));
        assert!(!adv.history_tags.contains_key("hometown_quest"));

        // Quest in hometown should increment
        assert!(check_hometown_quest(&mut adv, hometown));
        assert_eq!(adv.history_tags["hometown_quest"], 1);

        // Second time
        assert!(check_hometown_quest(&mut adv, hometown));
        assert_eq!(adv.history_tags["hometown_quest"], 2);
    }

    #[test]
    fn rival_confrontation_tag_increments() {
        let regions = make_regions();
        let factions = make_factions();
        // Use a seed that gives a rival faction
        let mut rng = 1u64;
        let mut backstory = None;
        // Try seeds until we get a rival
        for seed in 0..100u64 {
            rng = seed;
            let b = generate_backstory("Elara", "berserker", &regions, &factions, &mut rng);
            if b.rival_faction_id.is_some() {
                backstory = Some(b);
                break;
            }
        }
        let backstory = backstory.expect("should find a seed that produces a rival");
        let rival = backstory.rival_faction_id.unwrap();

        let mut adv = Adventurer {
            id: 2,
            name: "Elara".into(),
            archetype: "berserker".into(),
            level: 3,
            xp: 0,
            stats: AdventurerStats::default(),
            equipment: Equipment::default(),
            traits: vec![],
            status: AdventurerStatus::Idle,
            loyalty: 50.0,
            stress: 0.0,
            fatigue: 0.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 70.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),
            backstory: Some(backstory),
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: crate::headless_campaign::state::DiseaseStatus::Healthy,
        };

        // Wrong faction — no increment
        let wrong_faction = if rival == 0 { 1 } else { 0 };
        assert!(!check_rival_confrontation(&mut adv, Some(wrong_faction)));

        // Right faction — increment
        assert!(check_rival_confrontation(&mut adv, Some(rival)));
        assert_eq!(adv.history_tags["rival_confrontation"], 1);
    }
}
