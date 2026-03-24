//! Progression trigger system — detects when content should be generated.
//!
//! Scans adventurers and game state for trigger conditions. When a trigger
//! fires, adds a PendingProgression entry to be presented at the next rest.
//! In the full game, triggers send requests to the LLM immediately.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::content_prompts;
use crate::headless_campaign::llm;
use crate::headless_campaign::state::*;

const TRIGGER_CHECK_INTERVAL: u64 = 50;

pub fn tick_progression_triggers(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TRIGGER_CHECK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    state.ticks_since_rest += TRIGGER_CHECK_INTERVAL;

    // Snapshot adventurer data to avoid borrow conflicts
    let adv_data: Vec<(u32, String, String, u32, Vec<String>, f32, usize, usize, bool)> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .map(|a| (
            a.id, a.name.clone(), a.archetype.clone(), a.level,
            a.traits.clone(), a.tier_status.fame,
            a.tier_status.quests_completed, a.tier_status.party_victories,
            a.is_player_character,
        ))
        .collect();

    let completed_quests = state.completed_quests.len();
    let has_crisis = !state.overworld.active_crises.is_empty();
    let crisis_count = state.overworld.active_crises.len();

    for (adv_id, name, archetype, level, traits, fame, quests, victories, is_pc) in &adv_data {
        // --- Level milestone abilities (every 5 levels) ---
        if *level >= 5 && *level % 5 == 0 {
            let trigger = format!("level_{}", level);
            let content = generate_ability(state, *adv_id, &trigger, archetype, *level);
            try_add_progression(state, *adv_id, ProgressionKind::Ability,
                &trigger,
                &format!("{} reached level {} — new ability available", name, level),
                &content,
            );
        }

        // --- Quest completion abilities (every 5 quests per adventurer) ---
        if *quests > 0 && *quests % 5 == 0 {
            let trigger = format!("quests_{}", quests);
            let content = generate_ability(state, *adv_id, &trigger, archetype, *level);
            try_add_progression(state, *adv_id, ProgressionKind::Ability,
                &trigger,
                &format!("{} completed {} quests — experience yields insight", name, quests),
                &content,
            );
        }

        // --- Battle veteran abilities (every 3 victories) ---
        if *victories > 0 && *victories % 3 == 0 {
            let trigger = format!("victories_{}", victories);
            let content = generate_ability(state, *adv_id, &trigger, archetype, *level);
            try_add_progression(state, *adv_id, ProgressionKind::Ability,
                &trigger,
                &format!("{} won {} battles — combat instincts sharpen", name, victories),
                &content,
            );
        }

        // --- Fame threshold class offers ---
        let fame_thresholds = [30.0, 80.0, 200.0, 500.0, 1000.0, 2000.0];
        for &threshold in &fame_thresholds {
            if *fame >= threshold {
                let trigger = format!("class_fame_{}", threshold as u32);
                let content = generate_class(state, *adv_id, &trigger, archetype, *level, *fame);
                try_add_progression(state, *adv_id, ProgressionKind::ClassOffer,
                    &trigger,
                    &format!("{}'s reputation grows (fame {:.0}) — a new path opens", name, fame),
                    &content,
                );
            }
        }

        // --- Crisis response abilities ---
        if has_crisis && *quests >= 3 {
            let trigger = format!("crisis_response_{}", crisis_count);
            let content = generate_ability(state, *adv_id, &trigger, archetype, *level);
            try_add_progression(state, *adv_id, ProgressionKind::Ability,
                &trigger,
                &format!("{} faces {} active crises — desperation breeds innovation", name, crisis_count),
                &content,
            );
        }

        // --- Hero candidacy ---
        if *fame >= 2000.0 && has_crisis {
            let content = generate_hero(state, *adv_id);
            try_add_progression(state, *adv_id, ProgressionKind::HeroCandidacy,
                "hero_candidacy",
                &format!("{}'s legend grows — the path of the Hero opens", name),
                &content,
            );
        }

        // --- Injury survival ability (survived 80+ injury) ---
        let adv_injury = state.adventurers.iter()
            .find(|a| a.id == *adv_id)
            .map(|a| a.injury)
            .unwrap_or(0.0);
        if adv_injury > 80.0 {
            let content = generate_ability(state, *adv_id, "injury_survivor", archetype, *level);
            try_add_progression(state, *adv_id, ProgressionKind::Ability,
                "injury_survivor",
                &format!("{} fights through grievous wounds — survival instincts awaken", name),
                &content,
            );
        }

        // --- PC-specific triggers ---
        if *is_pc {
            for &milestone in &[10usize, 25, 50] {
                if *quests >= milestone {
                    let trigger = format!("pc_milestone_{}", milestone);
                    let content = generate_ability(state, *adv_id, &trigger, archetype, *level);
                    try_add_progression(state, *adv_id, ProgressionKind::Ability,
                        &trigger,
                        &format!("{}'s journey continues — {} quests shape their destiny", name, milestone),
                        &content,
                    );
                }
            }
        }
    }

    // --- Guild-wide triggers ---

    // Guild milestone abilities (every 25 quests completed)
    if completed_quests > 0 && completed_quests % 25 == 0 {
        try_add_progression(state, 0, ProgressionKind::QuestHook,
            &format!("guild_milestone_{}", completed_quests),
            &format!("The guild has completed {} quests — its reputation precedes it", completed_quests),
            "Guild milestone quest hook",
        );
    }

    // New crisis trigger — collect keys first to avoid borrow conflict
    if has_crisis {
        let crisis_keys: Vec<(String, String)> = state.overworld.active_crises.iter().map(|crisis| {
            let key = match crisis {
                ActiveCrisis::SleepingKing { champions_arrived, .. } =>
                    format!("sleeping_king_champ_{}", champions_arrived),
                ActiveCrisis::Breach { wave_number, .. } =>
                    format!("breach_wave_{}", wave_number),
                ActiveCrisis::Corruption { corrupted_regions, .. } =>
                    format!("corruption_{}", corrupted_regions.len()),
                ActiveCrisis::Unifier { absorbed_factions, .. } =>
                    format!("unifier_{}", absorbed_factions.len()),
                ActiveCrisis::Decline { severity, .. } =>
                    format!("decline_{}", (*severity * 10.0) as u32),
            };
            (format!("crisis_escalation_{}", key), format!("Crisis escalation: {}", key))
        }).collect();

        for (trigger_key, desc) in crisis_keys {
            try_add_progression(state, 0, ProgressionKind::QuestHook,
                &trigger_key, &desc, "Crisis escalation quest hook");
        }
    }

    // Faction relation milestones — collect first
    let faction_triggers: Vec<(String, String)> = state.factions.iter().flat_map(|faction| {
        [50.0f32, 75.0, 90.0].iter().filter_map(move |&threshold| {
            if faction.relationship_to_guild >= threshold {
                Some((
                    format!("faction_{}_{}", faction.id, threshold as u32),
                    format!("{} relations reach {:.0} — new opportunities arise", faction.name, threshold),
                ))
            } else {
                None
            }
        })
    }).collect();

    for (trigger_key, desc) in faction_triggers {
        try_add_progression(state, 0, ProgressionKind::QuestHook,
            &trigger_key, &desc, "Faction milestone quest hook");
    }

    // Rest reminder
    if state.ticks_since_rest > 3000
        && !state.resting
        && !state.pending_progression.is_empty()
    {
        let already_reminded = state.event_log.iter().rev().take(5)
            .any(|e| e.description.contains("should rest"));
        if !already_reminded {
            let n = state.pending_progression.len();
            let id = state.next_event_id;
            state.next_event_id += 1;
            state.event_log.push(CampaignEvent {
                id, tick: state.tick,
                description: format!("Your adventurers should rest. {} progression items await.", n),
            });
        }
    }
}

/// Try to add a progression item if the trigger hasn't already fired.
fn try_add_progression(
    state: &mut CampaignState,
    adv_id: u32,
    kind: ProgressionKind,
    trigger_key: &str,
    description: &str,
    content: &str,
) {
    // Check if this trigger already fired for this adventurer
    let already_exists = state.pending_progression.iter().any(|p| {
        p.trigger == trigger_key && p.adventurer_id == (if adv_id > 0 { Some(adv_id) } else { None })
    });
    if already_exists {
        return;
    }

    state.pending_progression.push(PendingProgression {
        adventurer_id: if adv_id > 0 { Some(adv_id) } else { None },
        kind,
        content: content.to_string(),
        description: description.to_string(),
        generated_at_tick: state.tick,
        trigger: trigger_key.to_string(),
    });
}

// ---------------------------------------------------------------------------
// Content generators — LLM with template fallback
// ---------------------------------------------------------------------------

/// Generate ability content: try VAE first, then LLM, fall back to template.
/// Takes archetype/level from snapshot to avoid stale adventurer lookups.
fn generate_ability(state: &CampaignState, adv_id: u32, trigger: &str, archetype: &str, level: u32) -> String {
    // Try VAE model first (instant, ~0.1ms)
    if let Some(ref vae) = state.vae_model {
        let input = super::super::vae_features::assemble_input(
            state, adv_id, trigger, ProgressionKind::Ability,
        );
        let name = format!("{}_{}", archetype, trigger);
        return vae.generate_ability(&input, &name);
    }

    // Try LLM (slow, ~8s)
    if let (Some(ref llm_cfg), Some(ref store)) = (&state.llm_config, &state.llm_store) {
        let context = content_prompts::ability_prompt(state, adv_id, trigger);
        if let Some(content) = llm::generate_ability(llm_cfg, store, &context) {
            return content;
        }
    }

    generate_ability_template(archetype, level)
}

/// Generate class content: try VAE first, then LLM, fall back to template.
fn generate_class(state: &CampaignState, adv_id: u32, trigger: &str, archetype: &str, level: u32, fame: f32) -> String {
    if let Some(ref vae) = state.vae_model {
        let input = super::super::vae_features::assemble_input(
            state, adv_id, trigger, ProgressionKind::ClassOffer,
        );
        let name = format!("{}Class", archetype.chars().next().unwrap().to_uppercase().to_string() + &archetype[1..]);
        return vae.generate_class(&input, &name);
    }

    if let (Some(ref llm_cfg), Some(ref store)) = (&state.llm_config, &state.llm_store) {
        let context = content_prompts::class_prompt(state, adv_id, trigger);
        if let Some(content) = llm::generate_class(llm_cfg, store, &context) {
            return content;
        }
    }

    generate_class_template(archetype, level, fame)
}

/// Generate hero class content: try LLM first, fall back to template.
fn generate_hero(state: &CampaignState, adv_id: u32) -> String {
    if let (Some(ref llm_cfg), Some(ref store)) = (&state.llm_config, &state.llm_store) {
        let context = content_prompts::class_prompt(state, adv_id, "hero_candidacy");
        if let Some(content) = llm::generate_class(llm_cfg, store, &context) {
            return content;
        }
    }
    generate_hero_class_template()
}

// ---------------------------------------------------------------------------
// Template fallbacks
// ---------------------------------------------------------------------------

fn generate_ability_template(archetype: &str, level: u32) -> String {
    match archetype {
        "ranger" => format!("ability ranger_skill_{} {{ type: passive, trigger: on_ability_used, effect: buff attack 10% 5s, tag: ranged, description: \"Sharpened instincts\" }}", level),
        "knight" => format!("ability knight_skill_{} {{ type: active, cooldown: 12s, effect: buff defense 15% 8s + aura defense +3, tag: defense, description: \"Steadfast resolve\" }}", level),
        "mage" => format!("ability mage_skill_{} {{ type: active, cooldown: 15s, effect: damage 40 + debuff defense 10% 5s, tag: arcane, description: \"Arcane disruption\" }}", level),
        "cleric" => format!("ability cleric_skill_{} {{ type: active, cooldown: 20s, effect: heal 40 + buff max_hp 10% 10s, tag: healing, description: \"Restorative prayer\" }}", level),
        "rogue" => format!("ability rogue_skill_{} {{ type: active, cooldown: 10s, effect: damage 50 + stealth 3s, tag: stealth, description: \"Strike from shadows\" }}", level),
        _ => format!("ability veteran_skill_{} {{ type: passive, trigger: on_damage_taken, effect: buff defense 10% 3s, tag: survival, description: \"Battle hardened\" }}", level),
    }
}

fn generate_class_template(archetype: &str, level: u32, fame: f32) -> String {
    let class_name = match (archetype, fame > 200.0) {
        ("ranger", false) => "Scout",
        ("ranger", true) => "Pathfinder",
        ("knight", false) => "Sentinel",
        ("knight", true) => "Warden",
        ("mage", false) => "Evoker",
        ("mage", true) => "Archmage",
        ("cleric", false) => "Healer",
        ("cleric", true) => "HighPriest",
        ("rogue", false) => "Infiltrator",
        ("rogue", true) => "Shadowmaster",
        (_, false) => "Veteran",
        (_, true) => "Elite",
    };
    format!("class {} {{ stat_growth: +2 attack, +2 defense, +1 speed per level, tags: {}, scaling party_alive_count {{ always: +5% attack }}, abilities {{ level 1: specialization \"Class ability\" }}, requirements: level {}, fame {} }}",
        class_name, archetype, level.saturating_sub(2), fame as u32 / 2)
}

fn generate_hero_class_template() -> String {
    "class Hero { stat_growth: +5 all per level, tags: crisis, leadership, legendary, scaling crisis_severity { when crisis_active: +20% all_stats }, abilities { level 1: rallying_cry \"Allies gain combat bonuses\" }, requirements: fame 2000, active_crisis, consolidates_at: 10 }".into()
}
