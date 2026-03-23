//! Progression trigger system — detects when content should be generated.
//!
//! Runs every 100 ticks. Scans adventurers and game state for trigger
//! conditions. When a trigger fires, adds a PendingProgression entry
//! that will be presented at the next rest event.
//!
//! In the full game, triggers send requests to the LLM immediately.
//! The LLM generates content asynchronously and the result is cached
//! in pending_progression. For now, we use template-based generation.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

const TRIGGER_CHECK_INTERVAL: u64 = 100;

pub fn tick_progression_triggers(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TRIGGER_CHECK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    state.ticks_since_rest += TRIGGER_CHECK_INTERVAL;

    // Check each adventurer for progression triggers
    let adventurer_snapshots: Vec<(u32, String, u32, Vec<String>, f32)> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .map(|a| (a.id, a.archetype.clone(), a.level, a.traits.clone(), a.tier_status.fame))
        .collect();

    for (adv_id, archetype, level, traits, fame) in &adventurer_snapshots {
        // Already has pending progression for this adventurer? Skip.
        if state.pending_progression.iter().any(|p| p.adventurer_id == Some(*adv_id)) {
            continue;
        }

        // Ability trigger: every 5 levels, offer a contextual ability
        if *level > 0 && *level % 5 == 0 {
            let already_offered = state.pending_progression.iter()
                .any(|p| p.adventurer_id == Some(*adv_id) && matches!(p.kind, ProgressionKind::Ability));
            if !already_offered {
                state.pending_progression.push(PendingProgression {
                    adventurer_id: Some(*adv_id),
                    kind: ProgressionKind::Ability,
                    content: generate_template_ability(&archetype, *level),
                    description: format!("New ability available for level {} {}", level, archetype),
                    generated_at_tick: state.tick,
                    trigger: format!("level_{}", level),
                });
            }
        }

        // Class trigger: at fame thresholds, offer a class
        let fame_thresholds = [50.0, 200.0, 500.0, 2000.0];
        for &threshold in &fame_thresholds {
            if *fame >= threshold {
                let trigger_key = format!("class_fame_{}", threshold as u32);
                let already_offered = state.pending_progression.iter()
                    .any(|p| p.adventurer_id == Some(*adv_id) && p.trigger == trigger_key);
                if !already_offered {
                    state.pending_progression.push(PendingProgression {
                        adventurer_id: Some(*adv_id),
                        kind: ProgressionKind::ClassOffer,
                        content: generate_template_class(&archetype, *level, *fame),
                        description: format!("Class specialization available (fame {:.0})", fame),
                        generated_at_tick: state.tick,
                        trigger: trigger_key,
                    });
                }
            }
        }

        // Hero candidacy: fame >= 2000 + active crisis
        if *fame >= 2000.0 && !state.overworld.active_crises.is_empty() {
            let already_offered = state.pending_progression.iter()
                .any(|p| p.adventurer_id == Some(*adv_id) && matches!(p.kind, ProgressionKind::HeroCandidacy));
            if !already_offered {
                state.pending_progression.push(PendingProgression {
                    adventurer_id: Some(*adv_id),
                    kind: ProgressionKind::HeroCandidacy,
                    content: generate_hero_class(),
                    description: "Hero class candidacy available — a defining quest awaits".into(),
                    generated_at_tick: state.tick,
                    trigger: "hero_candidacy".into(),
                });
            }
        }
    }

    // Prompt rest if enough ticks have passed
    if state.ticks_since_rest > 5000 && !state.resting && !state.pending_progression.is_empty() {
        // Add a gentle reminder (don't spam)
        let already_reminded = state.event_log.iter().rev().take(5)
            .any(|e| e.description.contains("should rest"));
        if !already_reminded {
            let id = state.next_event_id;
            state.next_event_id += 1;
            state.event_log.push(CampaignEvent {
                id,
                tick: state.tick,
                description: format!(
                    "Your adventurers should rest soon. {} progression items await.",
                    state.pending_progression.len()
                ),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Template generators (placeholder until LLM is wired)
// ---------------------------------------------------------------------------

fn generate_template_ability(archetype: &str, level: u32) -> String {
    match archetype {
        "ranger" => format!(r#"
ability eagle_eye_{} {{
    type: passive
    trigger: on_ability_used
    effect: +{}% attack for 5s
    tag: ranged
    description: "Keen eyes spot weakness"
}}"#, level, level * 2),
        "knight" => format!(r#"
ability shield_bash_{} {{
    type: active
    cooldown: 10s
    effect: stun 2s + {} damage
    tag: melee
    description: "A devastating shield strike"
}}"#, level, level * 3),
        "mage" => format!(r#"
ability arcane_surge_{} {{
    type: active
    cooldown: 15s
    effect: +{}% ability_power for 8s
    tag: arcane
    description: "Channel raw magical energy"
}}"#, level, level * 3),
        "cleric" => format!(r#"
ability divine_grace_{} {{
    type: active
    cooldown: 20s
    effect: heal {} to all allies
    tag: healing
    description: "A prayer answered with golden light"
}}"#, level, level * 5),
        "rogue" => format!(r#"
ability shadow_strike_{} {{
    type: active
    cooldown: 8s
    effect: {} damage + stealth 3s
    tag: stealth
    description: "Strike from the shadows"
}}"#, level, level * 4),
        _ => format!(r#"
ability veteran_instinct_{} {{
    type: passive
    trigger: on_damage_taken
    effect: +{}% defense for 3s
    tag: survival
    description: "Experience turns pain into focus"
}}"#, level, level),
    }
}

fn generate_template_class(archetype: &str, level: u32, fame: f32) -> String {
    let class_name = match (archetype, fame as u32) {
        ("ranger", 0..=199) => "Scout",
        ("ranger", _) => "Pathfinder",
        ("knight", 0..=199) => "Sentinel",
        ("knight", _) => "Warden",
        ("mage", 0..=199) => "Evoker",
        ("mage", _) => "Archmage",
        ("cleric", 0..=199) => "Healer",
        ("cleric", _) => "High Priest",
        ("rogue", 0..=199) => "Infiltrator",
        ("rogue", _) => "Shadowmaster",
        (_, 0..=199) => "Veteran",
        (_, _) => "Elite",
    };

    format!(r#"class {} {{
    stat_growth: +2 attack, +2 defense, +1 speed per level

    tags: {}, leadership

    scaling party_alive_count {{
        when party_members > 0: +10% attack
        always: aura morale +2
    }}

    abilities {{
        level 1: {} "Specialized combat technique"
    }}

    requirements: level {}, fame {}
}}"#,
        class_name,
        archetype,
        class_name.to_lowercase().replace(' ', "_"),
        level.saturating_sub(2),
        fame as u32 / 2,
    )
}

fn generate_hero_class() -> String {
    r#"class Hero {
    stat_growth: +5 all per level

    tags: crisis, leadership, legendary, inspiration, sacrifice

    scaling crisis_severity {
        when crisis_active: +20% all_stats
        when crisis_severity > 200: escape 1.0
        when crisis_severity > 500: last_stand below 20% hp +100% attack
    }

    abilities {
        level 1: rallying_cry "Nearby allies gain combat bonuses"
        level 3: crisis_sense "Bonus damage against crisis threats"
        level 5: undying_will "Cannot die for 10 ticks when below 20% HP"
        level 10: consolidation "Merge with primary class — all abilities enhanced"
    }

    requirements: fame 2000, active_crisis
    consolidates_at: 10
}"#.to_string()
}
