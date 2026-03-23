//! Context assembly for LLM content generation prompts.
//!
//! Builds rich prompts from game state for ability, class, and quest
//! generation. The prompt includes the adventurer's full history,
//! current game state, existing abilities, and the trigger context.

use super::state::*;

// ---------------------------------------------------------------------------
// Adventurer context
// ---------------------------------------------------------------------------

/// Build a rich context string for an adventurer.
pub fn adventurer_context(state: &CampaignState, adv_id: u32) -> String {
    let adv = match state.adventurers.iter().find(|a| a.id == adv_id) {
        Some(a) => a,
        None => return "Unknown adventurer".into(),
    };

    let mut ctx = String::new();

    // Identity
    ctx.push_str(&format!("Name: {}\n", adv.name));
    ctx.push_str(&format!("Archetype: {} (level {})\n", adv.archetype, adv.level));
    if adv.is_player_character {
        ctx.push_str("Role: Player Character (guild leader)\n");
    }

    // Stats
    ctx.push_str(&format!(
        "Stats: HP {:.0}, ATK {:.0}, DEF {:.0}, SPD {:.0}, AP {:.0}\n",
        adv.stats.max_hp, adv.stats.attack, adv.stats.defense,
        adv.stats.speed, adv.stats.ability_power
    ));

    // Condition
    ctx.push_str(&format!(
        "Condition: stress {:.0}, fatigue {:.0}, injury {:.0}, morale {:.0}, loyalty {:.0}\n",
        adv.stress, adv.fatigue, adv.injury, adv.morale, adv.loyalty
    ));

    // Traits
    if !adv.traits.is_empty() {
        ctx.push_str(&format!("Traits: {}\n", adv.traits.join(", ")));
    }

    // Tier status
    let tier_name = match adv.tier_status.tier {
        0 => "Adventurer",
        1 => "Named",
        2 => "Champion",
        3 => "Lord",
        4 => "Hero",
        5 => "Legend",
        _ => "Unknown",
    };
    ctx.push_str(&format!(
        "Tier: {} (fame {:.0}, quests completed {}, party victories {})\n",
        tier_name, adv.tier_status.fame,
        adv.tier_status.quests_completed, adv.tier_status.party_victories
    ));

    // Leadership
    if let Some(ref role) = adv.leadership_role {
        ctx.push_str(&format!("Leadership: {}\n", role.title));
    }

    // Current status
    ctx.push_str(&format!("Status: {:?}\n", adv.status));

    ctx
}

/// Build the list of available tags from an adventurer's archetype.
pub fn adventurer_tags(adv: &Adventurer) -> Vec<&'static str> {
    match adv.archetype.as_str() {
        "ranger" => vec!["ranged", "nature", "stealth", "tracking", "survival"],
        "knight" => vec!["melee", "defense", "leadership", "fortification", "honor"],
        "mage" => vec!["arcane", "elemental", "ritual", "knowledge", "enchantment"],
        "cleric" => vec!["healing", "divine", "protection", "purification", "restoration"],
        "rogue" => vec!["stealth", "assassination", "agility", "deception", "sabotage"],
        _ => vec!["melee", "survival"],
    }
}

// ---------------------------------------------------------------------------
// Quest history context
// ---------------------------------------------------------------------------

/// Summarize an adventurer's recent quest history.
pub fn quest_history_context(state: &CampaignState, adv_id: u32) -> String {
    let mut ctx = String::new();

    // Count quest types completed
    let mut quest_type_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut total_victories = 0u32;
    let mut total_defeats = 0u32;

    for quest in &state.completed_quests {
        // We don't track per-adventurer quest participation yet,
        // so we report guild-wide stats
        let qt = format!("{:?}", quest.quest_type);
        *quest_type_counts.entry(qt).or_default() += 1;
        match quest.result {
            QuestResult::Victory => total_victories += 1,
            QuestResult::Defeat => total_defeats += 1,
            _ => {}
        }
    }

    ctx.push_str(&format!(
        "Quest record: {} victories, {} defeats\n",
        total_victories, total_defeats
    ));

    if !quest_type_counts.is_empty() {
        let mut types: Vec<_> = quest_type_counts.iter().collect();
        types.sort_by(|a, b| b.1.cmp(a.1));
        let type_str: Vec<String> = types.iter()
            .map(|(t, c)| format!("{} {}", c, t))
            .collect();
        ctx.push_str(&format!("Quest types: {}\n", type_str.join(", ")));
    }

    ctx
}

// ---------------------------------------------------------------------------
// Guild context
// ---------------------------------------------------------------------------

/// Summarize current guild state.
pub fn guild_context(state: &CampaignState) -> String {
    let mut ctx = String::new();

    let alive = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count();
    let injured = state.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Injured)
        .count();
    let fighting = state.adventurers.iter()
        .filter(|a| a.status == AdventurerStatus::Fighting)
        .count();

    ctx.push_str(&format!(
        "Guild: {} adventurers ({} injured, {} in combat)\n",
        alive, injured, fighting
    ));
    ctx.push_str(&format!(
        "Resources: {:.0} gold, {:.0} supplies, {:.0} reputation\n",
        state.guild.gold, state.guild.supplies, state.guild.reputation
    ));
    ctx.push_str(&format!(
        "Active quests: {}, Completed: {}\n",
        state.active_quests.len(), state.completed_quests.len()
    ));

    // Roster summary (other adventurers' archetypes and levels)
    let mut roster: Vec<String> = state.adventurers.iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .map(|a| format!("{} (L{} {})", a.name, a.level, a.archetype))
        .collect();
    if !roster.is_empty() {
        ctx.push_str(&format!("Roster: {}\n", roster.join(", ")));
    }

    ctx
}

// ---------------------------------------------------------------------------
// World context
// ---------------------------------------------------------------------------

/// Summarize world state including factions and crises.
pub fn world_context(state: &CampaignState) -> String {
    let mut ctx = String::new();

    // Regions
    ctx.push_str("Regions:\n");
    for r in &state.overworld.regions {
        let owner_name = state.factions.iter()
            .find(|f| f.id == r.owner_faction_id)
            .map(|f| f.name.as_str())
            .unwrap_or("unknown");
        ctx.push_str(&format!(
            "  {} — owner: {}, control: {:.0}, unrest: {:.0}, threat: {:.0}\n",
            r.name, owner_name, r.control, r.unrest, r.threat_level
        ));
    }

    // Factions
    ctx.push_str("Factions:\n");
    for f in &state.factions {
        ctx.push_str(&format!(
            "  {} — stance: {:?}, relation: {:.0}, military: {:.0}, coalition: {}\n",
            f.name, f.diplomatic_stance, f.relationship_to_guild,
            f.military_strength, f.coalition_member
        ));
    }

    // Crises
    if !state.overworld.active_crises.is_empty() {
        ctx.push_str("Active crises:\n");
        for crisis in &state.overworld.active_crises {
            match crisis {
                ActiveCrisis::SleepingKing { champions_arrived, champion_ids, .. } => {
                    ctx.push_str(&format!(
                        "  Sleeping King — {}/{} champions arrived\n",
                        champions_arrived, champion_ids.len()
                    ));
                }
                ActiveCrisis::Breach { wave_number, wave_strength, .. } => {
                    ctx.push_str(&format!(
                        "  Dungeon Breach — wave {}, strength {:.0}\n",
                        wave_number, wave_strength
                    ));
                }
                ActiveCrisis::Corruption { corrupted_regions, .. } => {
                    ctx.push_str(&format!(
                        "  Corruption — {}/{} regions affected\n",
                        corrupted_regions.len(), state.overworld.regions.len()
                    ));
                }
                ActiveCrisis::Unifier { absorbed_factions, .. } => {
                    ctx.push_str(&format!(
                        "  Unifier — {} factions absorbed\n",
                        absorbed_factions.len()
                    ));
                }
                ActiveCrisis::Decline { severity, .. } => {
                    ctx.push_str(&format!(
                        "  Decline — severity {:.1}\n", severity
                    ));
                }
            }
        }
    }

    ctx.push_str(&format!(
        "Campaign progress: {:.0}%, Global threat: {:.0}\n",
        state.overworld.campaign_progress * 100.0,
        state.overworld.global_threat_level
    ));

    ctx
}

// ---------------------------------------------------------------------------
// Player character context
// ---------------------------------------------------------------------------

/// Summarize player character backstory and goal.
pub fn pc_context(state: &CampaignState) -> String {
    let mut ctx = String::new();

    if let Some(ref pc) = state.player_character {
        ctx.push_str(&format!("Player origin: {}\n", pc.origin));
        if !pc.backstory.is_empty() {
            ctx.push_str(&format!("Backstory: {}\n", pc.backstory.join(" → ")));
        }
        if let Some(ref goal) = pc.goal {
            ctx.push_str(&format!(
                "Personal goal: {} — {}{}\n",
                goal.name, goal.description,
                if goal.achieved { " (ACHIEVED)" } else { "" }
            ));
        }
    }

    ctx
}

// ---------------------------------------------------------------------------
// Full generation prompts
// ---------------------------------------------------------------------------

/// Build a complete ability generation prompt.
pub fn ability_prompt(state: &CampaignState, adv_id: u32, trigger: &str) -> String {
    let adv = match state.adventurers.iter().find(|a| a.id == adv_id) {
        Some(a) => a,
        None => return "Generate a generic ability.".into(),
    };

    let tags = adventurer_tags(adv);

    format!(
        "Generate an ability for this adventurer:\n\n\
         {}\n\
         Tags available: {}\n\n\
         {}\n\
         {}\n\
         {}\n\
         {}\n\
         Trigger: {}\n\n\
         The ability should reflect this adventurer's specific experiences, \
         personality, and role in the guild. It should feel earned, not generic.",
        adventurer_context(state, adv_id),
        tags.join(", "),
        quest_history_context(state, adv_id),
        guild_context(state),
        world_context(state),
        pc_context(state),
        trigger,
    )
}

/// Build a complete class generation prompt.
pub fn class_prompt(state: &CampaignState, adv_id: u32, trigger: &str) -> String {
    let adv = match state.adventurers.iter().find(|a| a.id == adv_id) {
        Some(a) => a,
        None => return "Generate a generic class.".into(),
    };

    let tags = adventurer_tags(adv);

    format!(
        "Generate a class specialization for this adventurer:\n\n\
         {}\n\
         Base tags: {}\n\n\
         {}\n\
         {}\n\
         {}\n\
         {}\n\
         Trigger: {}\n\n\
         The class should represent a natural evolution of how this adventurer \
         has been playing. The scaling source should match their role \
         (party-based for squad leaders, faction-based for champions, \
         crisis-based for heroes). Abilities should feel like a culmination \
         of their journey.",
        adventurer_context(state, adv_id),
        tags.join(", "),
        quest_history_context(state, adv_id),
        guild_context(state),
        world_context(state),
        pc_context(state),
        trigger,
    )
}

/// Build a quest hook generation prompt.
pub fn quest_hook_prompt(state: &CampaignState, trigger: &str) -> String {
    format!(
        "Generate a quest hook for this situation:\n\n\
         {}\n\
         {}\n\
         {}\n\
         Trigger: {}\n\n\
         The quest should create a meaningful choice with trade-offs \
         that matter given the current state of the guild and world.",
        guild_context(state),
        world_context(state),
        pc_context(state),
        trigger,
    )
}
