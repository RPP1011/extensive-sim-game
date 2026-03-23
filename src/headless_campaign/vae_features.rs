//! VAE feature assembler — converts CampaignState + adventurer + trigger into
//! a fixed 124-dim input vector for the grammar-guided content generation VAE.
//!
//! Dimension layout:
//!   [0..39)   Character identity (archetype, level, stats, condition, traits)
//!   [39..57)  Quest history (victories, defeats, fame, type distribution, etc.)
//!   [57..69)  Guild state (size, resources, role distribution)
//!   [69..106) World state (regions, factions, crisis, progress)
//!   [106..124) Trigger context (type, magnitude, output type, tags)

use super::state::*;

/// Total input dimension for the VAE encoder.
pub const VAE_INPUT_DIM: usize = 124;

/// 12 archetype families (collapsing 27+ archetypes).
const NUM_ARCHETYPE_FAMILIES: usize = 12;
/// 16 trait slots (multi-hot).
const NUM_TRAIT_SLOTS: usize = 16;
/// 8 quest type slots (6 types + 2 padding for future).
const NUM_QUEST_TYPE_SLOTS: usize = 8;
/// 5 role buckets for guild composition.
const NUM_ROLE_BUCKETS: usize = 5;
/// Max regions encoded.
const MAX_REGIONS: usize = 5;
/// Max factions encoded.
const MAX_FACTIONS: usize = 3;
/// 8 trigger types.
const NUM_TRIGGER_TYPES: usize = 8;
/// 6 archetype-family tag slots.
const NUM_TAG_SLOTS: usize = 6;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Assemble the 124-dim VAE input vector from campaign state.
///
/// Returns `None` if the adventurer is not found (for guild-wide triggers,
/// pass adv_id=0 and get a zero-filled identity section).
pub fn assemble_input(
    state: &CampaignState,
    adv_id: u32,
    trigger: &str,
    kind: ProgressionKind,
) -> [f32; VAE_INPUT_DIM] {
    let mut v = [0.0f32; VAE_INPUT_DIM];
    let mut offset = 0;

    // --- Character Identity (39 dims) ---
    let adv = state.adventurers.iter().find(|a| a.id == adv_id);

    // Archetype one-hot (12)
    if let Some(a) = adv {
        let family = archetype_family(&a.archetype);
        if family < NUM_ARCHETYPE_FAMILIES {
            v[offset + family] = 1.0;
        }
    }
    offset += NUM_ARCHETYPE_FAMILIES; // 12

    // Level (1)
    v[offset] = adv.map(|a| a.level as f32 / 100.0).unwrap_or(0.0);
    offset += 1; // 13

    // Stats: HP, ATK, DEF, SPD, AP (5)
    if let Some(a) = adv {
        v[offset] = a.stats.max_hp / 500.0;
        v[offset + 1] = a.stats.attack / 100.0;
        v[offset + 2] = a.stats.defense / 100.0;
        v[offset + 3] = a.stats.speed / 100.0;
        v[offset + 4] = a.stats.ability_power / 100.0;
    }
    offset += 5; // 18

    // Condition: stress, fatigue, injury, morale, loyalty (5)
    if let Some(a) = adv {
        v[offset] = a.stress / 100.0;
        v[offset + 1] = a.fatigue / 100.0;
        v[offset + 2] = a.injury / 100.0;
        v[offset + 3] = a.morale / 100.0;
        v[offset + 4] = a.loyalty / 100.0;
    }
    offset += 5; // 23

    // Traits multi-hot (16)
    if let Some(a) = adv {
        for t in &a.traits {
            let idx = trait_index(t);
            if idx < NUM_TRAIT_SLOTS {
                v[offset + idx] = 1.0;
            }
        }
    }
    offset += NUM_TRAIT_SLOTS; // 39

    // --- Quest History (18 dims) ---
    let victories = state.completed_quests.iter()
        .filter(|q| q.result == QuestResult::Victory).count() as f32;
    let defeats = state.completed_quests.iter()
        .filter(|q| q.result == QuestResult::Defeat).count() as f32;
    let total_quests = state.completed_quests.len() as f32;

    v[offset] = log_scale(victories);
    v[offset + 1] = log_scale(defeats);
    offset += 2; // 41

    // Fame (1)
    v[offset] = adv.map(|a| log_scale(a.tier_status.fame)).unwrap_or(0.0);
    offset += 1; // 42

    // Quest type distribution (8 = 6 types + 2 padding)
    if total_quests > 0.0 {
        for q in &state.completed_quests {
            let idx = quest_type_index(q.quest_type);
            if idx < NUM_QUEST_TYPE_SLOTS {
                v[offset + idx] += 1.0 / total_quests;
            }
        }
    }
    offset += NUM_QUEST_TYPE_SLOTS; // 50

    // Win rate (1)
    v[offset] = if total_quests > 0.0 { victories / total_quests } else { 0.5 };
    offset += 1; // 51

    // Total quests completed (1)
    v[offset] = log_scale(total_quests);
    offset += 1; // 52

    // Active quests (1)
    v[offset] = log_scale(state.active_quests.len() as f32);
    offset += 1; // 53

    // Combat/non-combat ratio (1)
    let combat_count = state.completed_quests.iter()
        .filter(|q| q.quest_type == QuestType::Combat).count() as f32;
    v[offset] = if total_quests > 0.0 { combat_count / total_quests } else { 0.5 };
    offset += 1; // 54

    // Win streak (1) — consecutive victories from most recent
    let mut streak = 0.0f32;
    for q in state.completed_quests.iter().rev() {
        if q.result == QuestResult::Victory {
            streak += 1.0;
        } else {
            break;
        }
    }
    v[offset] = log_scale(streak);
    offset += 1; // 55

    // Recency (2) — normalized ticks of last 2 completed quests
    let recent: Vec<f32> = state.completed_quests.iter().rev().take(2)
        .map(|q| q.completed_at_ms as f32 / state.elapsed_ms.max(1) as f32)
        .collect();
    v[offset] = recent.first().copied().unwrap_or(0.0);
    v[offset + 1] = recent.get(1).copied().unwrap_or(0.0);
    offset += 2; // 57

    // --- Guild State (12 dims) ---
    let alive = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead).count() as f32;
    let injured = state.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Injured).count() as f32;
    let fighting = state.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Fighting).count() as f32;

    v[offset] = log_scale(alive);
    v[offset + 1] = injured / alive.max(1.0);
    v[offset + 2] = fighting / alive.max(1.0);
    offset += 3; // 60

    // Gold, supplies, reputation (3)
    v[offset] = log_scale(state.guild.gold.max(0.0));
    v[offset + 1] = log_scale(state.guild.supplies.max(0.0));
    v[offset + 2] = state.guild.reputation / 100.0;
    offset += 3; // 63

    // Mean level (1)
    let mean_level = if alive > 0.0 {
        state.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .map(|a| a.level as f32)
            .sum::<f32>() / alive
    } else {
        1.0
    };
    v[offset] = mean_level / 50.0;
    offset += 1; // 64

    // Role distribution (5): tank/heal/ranged/melee/support
    if alive > 0.0 {
        for a in &state.adventurers {
            if a.status == AdventurerStatus::Dead { continue; }
            let role = archetype_to_role(&a.archetype);
            v[offset + role] += 1.0 / alive;
        }
    }
    offset += NUM_ROLE_BUCKETS; // 69

    // --- World State (37 dims) ---

    // Per-region (5 × 3 = 15): control, unrest, threat
    for (i, region) in state.overworld.regions.iter().take(MAX_REGIONS).enumerate() {
        v[offset + i * 3] = region.control / 100.0;
        v[offset + i * 3 + 1] = region.unrest / 100.0;
        v[offset + i * 3 + 2] = region.threat_level / 100.0;
    }
    offset += MAX_REGIONS * 3; // 84

    // Per-faction (3 × 4 = 12): relation, military, territory, coalition
    for (i, faction) in state.factions.iter().take(MAX_FACTIONS).enumerate() {
        v[offset + i * 4] = (faction.relationship_to_guild + 100.0) / 200.0; // [-100,100] → [0,1]
        v[offset + i * 4 + 1] = log_scale(faction.military_strength);
        // Territory: count regions owned by this faction
        let territory = state.overworld.regions.iter()
            .filter(|r| r.owner_faction_id == faction.id)
            .count() as f32;
        v[offset + i * 4 + 2] = territory / MAX_REGIONS as f32;
        v[offset + i * 4 + 3] = if faction.coalition_member { 1.0 } else { 0.0 };
    }
    offset += MAX_FACTIONS * 4; // 96

    // Campaign progress, global threat (2)
    v[offset] = state.overworld.campaign_progress;
    v[offset + 1] = state.overworld.global_threat_level / 100.0;
    offset += 2; // 98

    // Crisis encoding (8): 5 type one-hot + severity + count + is_active
    for crisis in &state.overworld.active_crises {
        let idx = crisis_type_index(crisis);
        if idx < 5 {
            v[offset + idx] = 1.0;
        }
    }
    // severity = max crisis severity
    let max_severity = state.overworld.active_crises.iter()
        .map(|c| crisis_severity(c))
        .fold(0.0f32, f32::max);
    v[offset + 5] = max_severity;
    v[offset + 6] = log_scale(state.overworld.active_crises.len() as f32);
    v[offset + 7] = if state.overworld.active_crises.is_empty() { 0.0 } else { 1.0 };
    offset += 8; // 106

    // --- Trigger Context (18 dims) ---

    // Trigger type one-hot (8)
    let trigger_type = parse_trigger_type(trigger);
    if trigger_type < NUM_TRIGGER_TYPES {
        v[offset + trigger_type] = 1.0;
    }
    offset += NUM_TRIGGER_TYPES; // 114

    // Trigger magnitude (1)
    v[offset] = parse_trigger_magnitude(trigger) / 100.0;
    offset += 1; // 115

    // Output type one-hot (3): ability, class, quest
    match kind {
        ProgressionKind::Ability => v[offset] = 1.0,
        ProgressionKind::ClassOffer | ProgressionKind::HeroCandidacy => v[offset + 1] = 1.0,
        ProgressionKind::QuestHook => v[offset + 2] = 1.0,
        _ => v[offset] = 1.0, // default to ability
    }
    offset += 3; // 118

    // Available tags multi-hot (6): melee, ranged, arcane, healing, stealth, nature
    if let Some(a) = adv {
        let tags = super::content_prompts::adventurer_tags(a);
        for tag in tags {
            let idx = tag_family_index(tag);
            if idx < NUM_TAG_SLOTS {
                v[offset + idx] = 1.0;
            }
        }
    }
    offset += NUM_TAG_SLOTS; // 124

    debug_assert_eq!(offset, VAE_INPUT_DIM);
    v
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Log-scale normalization: ln(1 + x) / ln(1 + max_expected).
/// Keeps values roughly in [0, 1] for common ranges.
fn log_scale(x: f32) -> f32 {
    (1.0 + x.abs()).ln() / 8.0 // ln(1+2980) ≈ 8.0 covers most game values
}

/// Map 27+ archetypes into 12 families.
fn archetype_family(archetype: &str) -> usize {
    match archetype {
        "knight" | "paladin" | "guardian" | "warden" => 0,    // tank/frontline
        "ranger" | "archer" | "scout" => 1,                    // ranged/nature
        "mage" | "sorcerer" | "enchanter" | "caster" => 2,    // arcane
        "cleric" | "healer" | "druid" => 3,                    // healing
        "rogue" | "assassin" => 4,                              // stealth
        "berserker" | "warrior" | "fighter" => 5,               // melee DPS
        "necromancer" | "warlock" => 6,                         // dark magic
        "bard" | "support" => 7,                                // support
        "monk" => 8,                                            // martial arts
        "shaman" => 9,                                          // nature magic
        "artificer" => 10,                                      // crafting
        "tank" => 11,                                           // pure tank
        _ => 0,                                                 // default to tank family
    }
}

/// Map trait names to a fixed 16-slot vocabulary.
fn trait_index(trait_name: &str) -> usize {
    match trait_name {
        "ambitious" => 0,
        "battle_scarred" => 1,
        "diplomat_born" => 2,
        "explorer" => 3,
        "merchant_mind" => 4,
        "natural_leader" => 5,
        "noble_blood" => 6,
        "peacemaker" => 7,
        "pragmatist" => 8,
        "protector" => 9,
        "wanderer" => 10,
        "cursed_arm" => 11,
        "keen_eye" => 12,
        "shield_wall" => 13,
        "veteran_instinct" => 14,
        _ => 15, // catch-all bucket
    }
}

/// Map QuestType to index (0-5, with padding to 8 slots).
fn quest_type_index(qt: QuestType) -> usize {
    match qt {
        QuestType::Combat => 0,
        QuestType::Exploration => 1,
        QuestType::Diplomatic => 2,
        QuestType::Escort => 3,
        QuestType::Rescue => 4,
        QuestType::Gather => 5,
    }
}

/// Map archetype to role bucket (0-4): tank, heal, ranged, melee, support.
fn archetype_to_role(archetype: &str) -> usize {
    match archetype {
        "knight" | "paladin" | "guardian" | "warden" | "tank" => 0,       // tank
        "cleric" | "healer" | "druid" => 1,                                // heal
        "ranger" | "archer" | "scout" | "mage" | "sorcerer" | "caster"
            | "enchanter" | "necromancer" | "warlock" => 2,                // ranged/caster
        "rogue" | "assassin" | "berserker" | "warrior" | "fighter"
            | "monk" => 3,                                                  // melee DPS
        "bard" | "support" | "shaman" | "artificer" => 4,                 // support
        _ => 3,                                                             // default melee
    }
}

/// Map ActiveCrisis variant to index (0-4).
fn crisis_type_index(crisis: &ActiveCrisis) -> usize {
    match crisis {
        ActiveCrisis::SleepingKing { .. } => 0,
        ActiveCrisis::Breach { .. } => 1,
        ActiveCrisis::Corruption { .. } => 2,
        ActiveCrisis::Unifier { .. } => 3,
        ActiveCrisis::Decline { .. } => 4,
    }
}

/// Extract a severity scalar [0, 1] from a crisis.
fn crisis_severity(crisis: &ActiveCrisis) -> f32 {
    match crisis {
        ActiveCrisis::SleepingKing { champions_arrived, .. } => *champions_arrived as f32 / 5.0,
        ActiveCrisis::Breach { wave_number, .. } => (*wave_number as f32 / 10.0).min(1.0),
        ActiveCrisis::Corruption { corrupted_regions, .. } => {
            (corrupted_regions.len() as f32 / 5.0).min(1.0)
        }
        ActiveCrisis::Unifier { absorbed_factions, .. } => {
            (absorbed_factions.len() as f32 / 3.0).min(1.0)
        }
        ActiveCrisis::Decline { severity, .. } => *severity,
    }
}

/// Parse trigger string prefix into trigger type index (0-7).
fn parse_trigger_type(trigger: &str) -> usize {
    if trigger.starts_with("level_") { 0 }
    else if trigger.starts_with("quests_") { 1 }
    else if trigger.starts_with("victories_") { 2 }
    else if trigger.starts_with("class_fame_") { 3 }
    else if trigger.starts_with("crisis_") { 4 }
    else if trigger.starts_with("hero_") { 5 }
    else if trigger.starts_with("injury_") { 6 }
    else if trigger.starts_with("pc_milestone_") { 7 }
    else { 0 } // default
}

/// Extract numeric magnitude from trigger key.
fn parse_trigger_magnitude(trigger: &str) -> f32 {
    // Try to find the last numeric segment after underscore
    trigger.rsplit('_')
        .next()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0)
}

/// Map tag name to family index (0-5) for the 6-slot tag encoding.
fn tag_family_index(tag: &str) -> usize {
    match tag {
        "melee" | "defense" | "leadership" | "fortification" | "honor" => 0,
        "ranged" | "tracking" | "survival" => 1,
        "arcane" | "elemental" | "ritual" | "knowledge" | "enchantment" => 2,
        "healing" | "divine" | "protection" | "purification" | "restoration" => 3,
        "stealth" | "assassination" | "agility" | "deception" | "sabotage" => 4,
        "nature" => 5,
        _ => 0, // default
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assemble_input_dimensions() {
        let state = CampaignState::default_test_campaign(42);
        // Pick first adventurer if any
        let adv_id = state.adventurers.first().map(|a| a.id).unwrap_or(0);
        let input = assemble_input(&state, adv_id, "level_5", ProgressionKind::Ability);
        assert_eq!(input.len(), VAE_INPUT_DIM);
        // No NaN or Inf
        for (i, &val) in input.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at dim {}", i);
            assert!(!val.is_infinite(), "Inf at dim {}", i);
        }
    }

    #[test]
    fn test_trigger_parsing() {
        assert_eq!(parse_trigger_type("level_10"), 0);
        assert_eq!(parse_trigger_type("quests_15"), 1);
        assert_eq!(parse_trigger_type("class_fame_200"), 3);
        assert_eq!(parse_trigger_type("hero_candidacy"), 5);

        assert_eq!(parse_trigger_magnitude("level_10"), 10.0);
        assert_eq!(parse_trigger_magnitude("class_fame_200"), 200.0);
        assert_eq!(parse_trigger_magnitude("hero_candidacy"), 0.0); // no numeric suffix
    }

    #[test]
    fn test_archetype_family_coverage() {
        // All known archetypes should map to a valid family
        for archetype in [
            "knight", "ranger", "mage", "cleric", "rogue", "paladin",
            "berserker", "necromancer", "bard", "druid", "warlock",
            "monk", "shaman", "artificer", "assassin", "guardian",
            "sorcerer", "warden", "archer", "healer", "warrior",
            "tank", "support", "caster", "fighter", "scout", "enchanter",
        ] {
            let family = archetype_family(archetype);
            assert!(family < NUM_ARCHETYPE_FAMILIES,
                "{} maps to family {} (>= {})", archetype, family, NUM_ARCHETYPE_FAMILIES);
        }
    }
}
