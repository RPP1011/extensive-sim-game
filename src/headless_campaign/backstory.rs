//! Backstory loading and character creation wiring.
//!
//! Loads backstory event chains from `assets/backstory/{origin}/` TOML files,
//! converts them to `ChoiceEvent`s, and queues them during the
//! `CharacterCreation` phase. Each choice accumulates effects onto the
//! player character being built.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::actions::WorldEvent;
use super::choice_templates::{instantiate_effect, EffectTemplate, TemplateContext};
use super::state::*;

// ---------------------------------------------------------------------------
// Backstory TOML format
// ---------------------------------------------------------------------------

/// A single backstory event loaded from TOML.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackstoryEvent {
    pub name: String,
    #[serde(default)]
    pub world_template: Option<String>,
    pub sequence: u32,
    pub prompt: String,
    pub options: Vec<BackstoryOption>,
    #[serde(default)]
    pub default_option: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackstoryOption {
    pub label: String,
    pub description: String,
    #[serde(default)]
    pub effects: Vec<EffectTemplate>,
    #[serde(default)]
    pub pc_traits: Vec<PcTraitEntry>,
    #[serde(default)]
    pub pc_stats: Option<PcStatMods>,
    #[serde(default)]
    pub goal: Option<GoalTemplate>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PcTraitEntry {
    #[serde(rename = "trait")]
    pub trait_name: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PcStatMods {
    #[serde(default)]
    pub attack: f32,
    #[serde(default)]
    pub defense: f32,
    #[serde(default)]
    pub speed: f32,
    #[serde(default)]
    pub max_hp: f32,
    #[serde(default)]
    pub ability_power: f32,
    #[serde(default)]
    pub resolve: f32,
    #[serde(default)]
    pub injury: f32,
    #[serde(default)]
    pub loyalty_bonus: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalTemplate {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub conditions: Vec<GoalCondition>,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// A loaded backstory chain (all events for one origin, in sequence order).
#[derive(Clone, Debug)]
pub struct BackstoryChain {
    pub origin_name: String,
    pub events: Vec<BackstoryEvent>,
}

/// Load all backstory chains from a directory.
/// Each subdirectory is an origin, containing numbered TOML event files.
pub fn load_backstory_chains(base_dir: &std::path::Path) -> Vec<BackstoryChain> {
    let mut chains = Vec::new();

    if !base_dir.exists() {
        return chains;
    }

    let entries = match std::fs::read_dir(base_dir) {
        Ok(e) => e,
        Err(_) => return chains,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let origin_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut events = Vec::new();

        if let Ok(files) = std::fs::read_dir(&path) {
            for file_entry in files.flatten() {
                let file_path = file_entry.path();
                if file_path.extension().and_then(|e| e.to_str()) != Some("toml") {
                    continue;
                }

                match std::fs::read_to_string(&file_path) {
                    Ok(content) => match toml::from_str::<BackstoryEvent>(&content) {
                        Ok(event) => events.push(event),
                        Err(e) => {
                            eprintln!("Warning: failed to parse backstory {}: {}",
                                file_path.display(), e);
                        }
                    },
                    Err(e) => {
                        eprintln!("Warning: failed to read {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        // Sort by sequence number
        events.sort_by_key(|e| e.sequence);

        if !events.is_empty() {
            chains.push(BackstoryChain {
                origin_name,
                events,
            });
        }
    }

    chains
}

/// Lazily loaded backstory chains.
static BACKSTORIES: std::sync::OnceLock<Vec<BackstoryChain>> = std::sync::OnceLock::new();

pub fn get_or_load_backstories() -> &'static Vec<BackstoryChain> {
    BACKSTORIES.get_or_init(|| {
        let dir = std::path::Path::new("assets/backstory");
        let chains = load_backstory_chains(dir);
        if !chains.is_empty() {
            eprintln!(
                "Backstories: loaded {} origins ({} total events)",
                chains.len(),
                chains.iter().map(|c| c.events.len()).sum::<usize>()
            );
        }
        chains
    })
}

// ---------------------------------------------------------------------------
// Character creation initialization
// ---------------------------------------------------------------------------

/// Initialize character creation for a campaign.
/// Picks a backstory chain based on seed and world template,
/// creates the initial PC, and queues the first backstory event.
pub fn init_character_creation(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    let backstories = get_or_load_backstories();
    if backstories.is_empty() {
        // No backstory data — skip to starting package selection
        state.phase = CampaignPhase::ChoosingStartingPackage;
        return;
    }

    // Detect world template
    let world_template = detect_world_template(state);

    // Filter chains by world template
    let eligible: Vec<&BackstoryChain> = backstories
        .iter()
        .filter(|chain| {
            chain.events.first().map_or(true, |e| {
                e.world_template.as_ref().map_or(true, |wt| {
                    wt.eq_ignore_ascii_case(&world_template) || wt == "*"
                })
            })
        })
        .collect();

    if eligible.is_empty() {
        state.phase = CampaignPhase::ChoosingStartingPackage;
        return;
    }

    // Pick chain based on seed
    let chain_idx = (state.rng as usize) % eligible.len();
    let chain = eligible[chain_idx];

    // Create initial PC
    let pc_id = state.adventurers.iter().map(|a| a.id).max().unwrap_or(0) + 100;
    let pc = Adventurer {
        id: pc_id,
        name: "Player".into(), // Will be customizable later
        archetype: "knight".into(), // Default, may change from backstory
        level: 1,
        xp: 0,
        stats: AdventurerStats {
            max_hp: 90.0,
            attack: 12.0,
            defense: 10.0,
            speed: 10.0,
            ability_power: 8.0,
        },
        equipment: Equipment::default(),
        traits: Vec::new(),
        status: AdventurerStatus::Idle,
        loyalty: 100.0, // PC is always loyal
        stress: 0.0,
        fatigue: 0.0,
        injury: 0.0,
        resolve: 70.0,
        morale: 90.0,
        party_id: None,
        guild_relationship: 100.0,
        leadership_role: None,
        is_player_character: false,
        faction_id: None,
        rallying_to: None,
                    tier_status: Default::default(),
                    history_tags: Default::default(),
            backstory: None,
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: crate::headless_campaign::state::DiseaseStatus::Healthy,

            mood_state: crate::headless_campaign::state::MoodState::default(),

            fears: Vec::new(),

            personal_goal: None,

            journal: Vec::new(),

            equipped_items: Vec::new(),
            nicknames: Vec::new(),
            secret_past: None,
    };

    state.adventurers.push(pc);
    state.player_character = Some(PlayerCharacter {
        adventurer_id: pc_id,
        name: "Player".into(),
        origin: chain.origin_name.clone(),
        backstory: Vec::new(),
        goal: None,
        alive: true,
        successor_id: None,
    });

    // Store the chain events as a creation queue
    state.creation_event_queue = chain
        .events
        .iter()
        .map(|e| e.clone())
        .collect();

    state.phase = CampaignPhase::CharacterCreation;

    // Queue the first event
    queue_next_creation_event(state, events);
}

/// Queue the next backstory event as a ChoiceEvent.
pub fn queue_next_creation_event(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    if state.creation_event_queue.is_empty() {
        // All backstory events done — transition to starting package
        state.phase = CampaignPhase::ChoosingStartingPackage;
        return;
    }

    let backstory_event = state.creation_event_queue.remove(0);
    let ctx = TemplateContext::new();

    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let options: Vec<ChoiceOption> = backstory_event
        .options
        .iter()
        .map(|opt| {
            let mut effects = Vec::new();

            // Convert template effects
            for eff in &opt.effects {
                if let Some(ce) = instantiate_effect(eff, &ctx) {
                    effects.push(ce);
                }
            }

            // PC trait effects — store as Narrative for now, applied in step.rs
            for trait_entry in &opt.pc_traits {
                effects.push(ChoiceEffect::Narrative(
                    format!("__PC_TRAIT__:{}", trait_entry.trait_name),
                ));
            }

            // PC stat mods — encode as Narrative with parseable format
            if let Some(ref stats) = opt.pc_stats {
                if stats.attack != 0.0 {
                    effects.push(ChoiceEffect::Narrative(format!("__PC_STAT__:attack:{}", stats.attack)));
                }
                if stats.defense != 0.0 {
                    effects.push(ChoiceEffect::Narrative(format!("__PC_STAT__:defense:{}", stats.defense)));
                }
                if stats.speed != 0.0 {
                    effects.push(ChoiceEffect::Narrative(format!("__PC_STAT__:speed:{}", stats.speed)));
                }
                if stats.max_hp != 0.0 {
                    effects.push(ChoiceEffect::Narrative(format!("__PC_STAT__:max_hp:{}", stats.max_hp)));
                }
                if stats.ability_power != 0.0 {
                    effects.push(ChoiceEffect::Narrative(format!("__PC_STAT__:ability_power:{}", stats.ability_power)));
                }
                if stats.resolve != 0.0 {
                    effects.push(ChoiceEffect::Narrative(format!("__PC_STAT__:resolve:{}", stats.resolve)));
                }
                if stats.injury != 0.0 {
                    effects.push(ChoiceEffect::Narrative(format!("__PC_STAT__:injury:{}", stats.injury)));
                }
            }

            // Goal — encode for extraction after choice resolution
            if let Some(ref goal) = opt.goal {
                effects.push(ChoiceEffect::Narrative(
                    format!("__PC_GOAL__:{}:{}", goal.name, goal.description),
                ));
            }

            ChoiceOption {
                label: opt.label.clone(),
                description: opt.description.clone(),
                effects,
            }
        })
        .collect();

    let choice = ChoiceEvent {
        id: choice_id,
        source: ChoiceSource::WorldEvent, // Creation events use WorldEvent source
        prompt: backstory_event.prompt.clone(),
        options,
        default_option: backstory_event.default_option.min(backstory_event.options.len().saturating_sub(1)),
        deadline_ms: None, // No deadline during creation — blocks until chosen
        created_at_ms: 0,
    };

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: choice.prompt.clone(),
        num_options: choice.options.len(),
    });

    state.pending_choices.push(choice);

    if let Some(ref mut pc) = state.player_character {
        pc.backstory.push(backstory_event.name);
    }
}

/// Apply PC-specific effects from a resolved backstory choice.
/// Called from step.rs after a choice is resolved during CharacterCreation.
pub fn apply_creation_effects(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let pc_id = match &state.player_character {
        Some(pc) => pc.adventurer_id,
        None => return,
    };

    // Scan the event log for __PC_TRAIT__, __PC_STAT__, __PC_GOAL__ markers
    // These were placed as Narrative effects during choice resolution
    let mut traits_to_add = Vec::new();
    let mut stat_mods = PcStatMods::default();
    let mut goal = None;

    // Check recent events (last few) for our markers
    for event in state.event_log.iter().rev().take(20) {
        let desc = &event.description;
        if let Some(trait_name) = desc.strip_prefix("__PC_TRAIT__:") {
            traits_to_add.push(trait_name.to_string());
        } else if let Some(stat_str) = desc.strip_prefix("__PC_STAT__:") {
            let parts: Vec<&str> = stat_str.splitn(2, ':').collect();
            if parts.len() == 2 {
                if let Ok(val) = parts[1].parse::<f32>() {
                    match parts[0] {
                        "attack" => stat_mods.attack += val,
                        "defense" => stat_mods.defense += val,
                        "speed" => stat_mods.speed += val,
                        "max_hp" => stat_mods.max_hp += val,
                        "ability_power" => stat_mods.ability_power += val,
                        "resolve" => stat_mods.resolve += val,
                        "injury" => stat_mods.injury += val,
                        _ => {}
                    }
                }
            }
        } else if let Some(goal_str) = desc.strip_prefix("__PC_GOAL__:") {
            let parts: Vec<&str> = goal_str.splitn(2, ':').collect();
            if parts.len() == 2 {
                goal = Some(PersonalGoal {
                    name: parts[0].to_string(),
                    description: parts[1].to_string(),
                    conditions: Vec::new(), // TODO: parse from TOML goal conditions
                    achieved: false,
                    fail_condition: None,
                    failed: false,
                });
            }
        }
    }

    // Apply to PC adventurer
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == pc_id) {
        for trait_name in traits_to_add {
            if !adv.traits.contains(&trait_name) {
                adv.traits.push(trait_name);
            }
        }
        adv.stats.attack += stat_mods.attack;
        adv.stats.defense += stat_mods.defense;
        adv.stats.speed += stat_mods.speed;
        adv.stats.max_hp += stat_mods.max_hp;
        adv.stats.ability_power += stat_mods.ability_power;
        adv.resolve = (adv.resolve + stat_mods.resolve).clamp(0.0, 100.0);
        adv.injury = (adv.injury + stat_mods.injury).clamp(0.0, 100.0);
    }

    // Set goal
    if let Some(g) = goal {
        if let Some(ref mut pc) = state.player_character {
            pc.goal = Some(g);
        }
    }

    // Remove marker events from the log (they're not for display)
    state.event_log.retain(|e| {
        !e.description.starts_with("__PC_TRAIT__:")
            && !e.description.starts_with("__PC_STAT__:")
            && !e.description.starts_with("__PC_GOAL__:")
    });

    // Queue next creation event
    queue_next_creation_event(state, events);
}

/// Detect world template from region names.
fn detect_world_template(state: &CampaignState) -> String {
    if state.overworld.regions.is_empty() {
        return "unknown".into();
    }
    let first = &state.overworld.regions[0].name;
    if first.contains("Greenhollow") { "Frontier".into() }
    else if first.contains("Crown") || first.contains("Landing") { "Civil War".into() }
    else if first.contains("Coral") || first.contains("Tidebreak") { "Archipelago".into() }
    else { "unknown".into() }
}
