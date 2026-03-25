//! Quest chain system — multi-part linked quests with escalating rewards.
//!
//! Ticks every 500 ticks. Generates narrative quest chains where completing
//! one step unlocks the next, with increasing threat and reward multipliers.
//! Max 2 active chains at once. Chains require 5+ completed quests to unlock.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    lcg_f32, lcg_next, CampaignState, QuestChain, QuestChainStep, QuestResult,
};

/// Maximum number of active (non-completed, non-failed) chains at any time.
const MAX_ACTIVE_CHAINS: usize = 2;

/// Minimum completed quests before chains can spawn.
const MIN_COMPLETIONS_FOR_CHAINS: usize = 5;

/// Reward multipliers per step index.
const STEP_REWARD_MULTIPLIERS: &[f32] = &[1.0, 1.5, 2.0, 3.0];

/// Chain templates: (name, narrative_theme, steps).
/// Each step: (quest_type, base_threat_boost, description).
const CHAIN_TEMPLATES: &[(&str, &str, &[(&str, f32, &str)])] = &[
    (
        "Revenge Arc",
        "vengeance",
        &[
            ("investigation", 0.1, "Investigate the source of a grievous wrong"),
            ("tracking", 0.2, "Track the perpetrator through hostile territory"),
            ("combat", 0.4, "Confront the villain and bring justice"),
        ],
    ),
    (
        "Treasure Hunt",
        "fortune",
        &[
            ("investigation", 0.05, "Find a cryptic clue pointing to lost treasure"),
            ("exploration", 0.15, "Explore ancient ruins following the trail"),
            ("dungeon", 0.3, "Delve into the treasure vault's depths"),
            ("combat", 0.5, "Defeat the guardian protecting the hoard"),
        ],
    ),
    (
        "Diplomatic Crisis",
        "diplomacy",
        &[
            ("diplomatic", 0.05, "Negotiate a fragile ceasefire between warring parties"),
            ("escort", 0.15, "Deliver critical peace terms through danger"),
            ("diplomatic", 0.25, "Resolve the final terms and forge a lasting accord"),
        ],
    ),
    (
        "Monster Siege",
        "defense",
        &[
            ("scouting", 0.1, "Scout the approaching monster horde"),
            ("construction", 0.2, "Prepare defenses and rally the garrison"),
            ("combat", 0.45, "Defend the settlement against the siege"),
        ],
    ),
    (
        "Ancient Mystery",
        "arcane",
        &[
            ("investigation", 0.05, "Discover fragments of a forgotten legend"),
            ("research", 0.1, "Research the ancient texts for answers"),
            ("exploration", 0.25, "Mount an expedition to the lost site"),
            ("combat", 0.5, "Uncover the revelation and survive its guardian"),
        ],
    ),
];

/// Tick the quest chain system every 500 ticks.
pub fn tick_quest_chains(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 17 != 0 || state.tick == 0 {
        return;
    }

    // Advance active chains based on completed quests
    advance_chains(state, events);

    // Try to generate new chains
    maybe_generate_chain(state, events);
}

/// Check if any active chain step was completed (quest completed with matching type).
fn advance_chains(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect recently completed quests (since last check = 500 ticks)
    let recent_threshold = state.tick.saturating_sub(500);
    let recent_completions: Vec<(String, bool)> = state
        .completed_quests
        .iter()
        .filter(|q| {
            let completed_tick = q.completed_at_ms / 100; // ms to ticks
            completed_tick >= recent_threshold
        })
        .map(|q| {
            let type_str = format!("{:?}", q.quest_type).to_lowercase();
            let success = q.result == QuestResult::Victory;
            (type_str, success)
        })
        .collect();

    if recent_completions.is_empty() {
        return;
    }

    // Process each active chain
    let chain_count = state.quest_chains.len();
    for i in 0..chain_count {
        if state.quest_chains[i].completed || state.quest_chains[i].failed {
            continue;
        }

        let step_idx = state.quest_chains[i].current_step;
        if step_idx >= state.quest_chains[i].steps.len() {
            continue;
        }

        let step_type = state.quest_chains[i].steps[step_idx].quest_type.clone();

        // Check if any recent completion matches the current step type
        let matched = recent_completions.iter().find(|(t, _)| {
            // Fuzzy match: step type is contained in quest type or vice versa
            t.contains(&step_type) || step_type.contains(t.as_str())
        });

        if let Some((_, success)) = matched {
            if *success {
                // Step completed successfully
                state.quest_chains[i].steps[step_idx].completed = true;
                let chain_name = state.quest_chains[i].name.clone();
                let step_desc = state.quest_chains[i].steps[step_idx].description.clone();
                let step_num = step_idx + 1;
                let total_steps = state.quest_chains[i].steps.len();

                events.push(WorldEvent::QuestChainStepCompleted {
                    chain_id: state.quest_chains[i].id,
                    chain_name: chain_name.clone(),
                    step: step_num as u32,
                    total_steps: total_steps as u32,
                    description: step_desc,
                });

                // Apply step reward (gold scaled by multiplier)
                let multiplier = state.quest_chains[i].steps[step_idx].reward_multiplier;
                let step_gold = 50.0 * multiplier;
                state.guild.gold += step_gold;
                state.guild.reputation = (state.guild.reputation + 2.0 * multiplier).min(100.0);

                // Advance to next step or complete chain
                state.quest_chains[i].current_step += 1;
                if state.quest_chains[i].current_step >= total_steps {
                    // Chain completed!
                    state.quest_chains[i].completed = true;
                    complete_chain(state, i, events);
                }
            } else {
                // Step failed
                state.quest_chains[i].failed = true;
                let chain_name = state.quest_chains[i].name.clone();
                let chain_id = state.quest_chains[i].id;

                // Morale and reputation penalty
                for adv in &mut state.adventurers {
                    adv.morale = (adv.morale - 5.0).max(0.0);
                }
                state.guild.reputation = (state.guild.reputation - 5.0).max(0.0);

                events.push(WorldEvent::QuestChainFailed {
                    chain_id,
                    chain_name,
                    step: (step_idx + 1) as u32,
                    reason: "Quest step was not completed successfully".into(),
                });
            }
        }
    }
}

/// Generate a new chain if conditions are met.
fn maybe_generate_chain(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Count active chains
    let active_count = state
        .quest_chains
        .iter()
        .filter(|c| !c.completed && !c.failed)
        .count();
    if active_count >= MAX_ACTIVE_CHAINS {
        return;
    }

    // Require enough quest history
    if state.completed_quests.len() < MIN_COMPLETIONS_FOR_CHAINS {
        return;
    }

    // 30% chance per eligible tick
    let roll = lcg_f32(&mut state.rng);
    if roll >= 0.3 {
        return;
    }

    // Pick a template that hasn't been used recently
    let used_themes: Vec<String> = state
        .quest_chains
        .iter()
        .filter(|c| !c.failed) // Allow retry of failed themes
        .map(|c| c.narrative_theme.clone())
        .collect();

    let template_idx = (lcg_next(&mut state.rng) as usize) % CHAIN_TEMPLATES.len();
    // Try up to all templates to find an unused one
    let mut chosen = None;
    for offset in 0..CHAIN_TEMPLATES.len() {
        let idx = (template_idx + offset) % CHAIN_TEMPLATES.len();
        let (_, theme, _) = CHAIN_TEMPLATES[idx];
        if !used_themes.contains(&theme.to_string()) {
            chosen = Some(idx);
            break;
        }
    }

    let template_idx = match chosen {
        Some(idx) => idx,
        None => return, // All templates already used
    };

    let (name, theme, step_templates) = CHAIN_TEMPLATES[template_idx];

    // Build chain
    let chain_id = state.next_chain_id;
    state.next_chain_id += 1;

    // Base threat scales with guild reputation and completed quests
    let base_threat = 0.3 + (state.completed_quests.len() as f32 * 0.01).min(0.3);

    let steps: Vec<QuestChainStep> = step_templates
        .iter()
        .enumerate()
        .map(|(i, (quest_type, threat_boost, desc))| {
            let multiplier_idx = i.min(STEP_REWARD_MULTIPLIERS.len() - 1);
            QuestChainStep {
                quest_type: quest_type.to_string(),
                threat_level: base_threat + threat_boost,
                description: desc.to_string(),
                reward_multiplier: STEP_REWARD_MULTIPLIERS[multiplier_idx],
                completed: false,
            }
        })
        .collect();

    let chain = QuestChain {
        id: chain_id,
        name: name.to_string(),
        steps,
        current_step: 0,
        started_tick: state.tick,
        completed: false,
        failed: false,
        narrative_theme: theme.to_string(),
    };

    events.push(WorldEvent::QuestChainStarted {
        chain_id,
        chain_name: chain.name.clone(),
        total_steps: chain.steps.len() as u32,
        theme: chain.narrative_theme.clone(),
    });

    state.quest_chains.push(chain);
}

/// Apply major rewards for completing an entire chain.
fn complete_chain(state: &mut CampaignState, chain_idx: usize, events: &mut Vec<WorldEvent>) {
    let chain = &state.quest_chains[chain_idx];
    let chain_id = chain.id;
    let chain_name = chain.name.clone();
    let theme = chain.narrative_theme.clone();
    let total_steps = chain.steps.len() as u32;

    // Major gold reward
    let gold_reward = 200.0 + (total_steps as f32 * 50.0);
    state.guild.gold += gold_reward;

    // Big reputation boost
    state.guild.reputation = (state.guild.reputation + 10.0).min(100.0);

    // Morale boost for all adventurers
    for adv in &mut state.adventurers {
        adv.morale = (adv.morale + 10.0).min(100.0);
    }

    // Add history tags to all living adventurers
    for adv in &mut state.adventurers {
        if adv.status == crate::headless_campaign::state::AdventurerStatus::Dead {
            continue;
        }
        *adv.history_tags
            .entry(format!("chain_{}", theme))
            .or_insert(0) += 1;
        *adv.history_tags.entry("quest_chain".into()).or_insert(0) += 1;
    }

    // Generate an artifact for chain completion
    let artifact_name = match theme.as_str() {
        "vengeance" => "Blade of Retribution",
        "fortune" => "Crown of the Hoard",
        "diplomacy" => "Seal of Accord",
        "defense" => "Shield of the Bastion",
        "arcane" => "Tome of Lost Ages",
        _ => "Chain Relic",
    };

    let artifact = crate::headless_campaign::state::Artifact {
        id: state.next_artifact_id,
        name: artifact_name.to_string(),
        origin_adventurer_name: "Guild".into(),
        origin_deed: format!("Completed the {} quest chain", chain_name),
        slot: "trinket".into(),
        stat_bonuses: crate::headless_campaign::state::ArtifactStats {
            attack: 5.0,
            defense: 5.0,
            hp: 10.0,
            speed: 2.0,
        },
        special_effect: match theme.as_str() {
            "vengeance" => crate::headless_campaign::state::ArtifactEffect::CombatPowerBoost(0.08),
            "fortune" => crate::headless_campaign::state::ArtifactEffect::XpMultiplier(1.15),
            "diplomacy" => crate::headless_campaign::state::ArtifactEffect::FactionInfluence(0.1),
            "defense" => crate::headless_campaign::state::ArtifactEffect::MoraleAura(5.0),
            "arcane" => crate::headless_campaign::state::ArtifactEffect::HealingBoost(0.1),
            _ => crate::headless_campaign::state::ArtifactEffect::None,
        },
        created_tick: state.tick,
        equipped_by: None,
    };
    state.next_artifact_id += 1;
    state.artifacts.push(artifact);

    events.push(WorldEvent::QuestChainCompleted {
        chain_id,
        chain_name: chain_name.clone(),
        total_steps,
        gold_reward,
        artifact_name: artifact_name.to_string(),
    });
}
