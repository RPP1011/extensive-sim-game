use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use crate::game_core::PersonalityArchetype;

/// Aggregated combat stats for one hero across a mission (or room).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombatJournal {
    pub hero_id: u32,
    pub hero_name: String,
    pub hero_archetype: PersonalityArchetype,
    pub hero_backstory: String,
    pub mission_name: String,
    pub outcome: String, // "Victory" / "Defeat" / "InProgress"
    pub damage_dealt: i32,
    pub damage_taken: i32,
    pub heals_given: i32,
    pub kills: u32,
    pub deaths: u32,
    pub abilities_used: Vec<(String, u32)>,
    pub passives_triggered: Vec<(String, u32)>,
    pub near_death_moments: u32,
    pub cc_applied: u32,
    pub shields_given: i32,
    pub duels_fought: u32,
    pub loyalty: f32,
    pub stress: f32,
    pub resolve: f32,
}

/// LFM-generated reward, one of several kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reward_type")]
pub enum ProgressionReward {
    #[serde(rename = "equipment")]
    Equipment {
        name: String,
        slot: String,
        rarity: String,
        #[serde(default)]
        attack_bonus: i32,
        #[serde(default)]
        hp_bonus: i32,
        #[serde(default)]
        speed_bonus: f32,
        #[serde(default = "default_cooldown_mult")]
        cooldown_mult: f32,
        #[serde(default)]
        flavor_text: String,
    },
    #[serde(rename = "stat_boost")]
    StatBoost {
        stat: String,
        amount: f32,
        reason: String,
    },
    #[serde(rename = "ability")]
    Ability {
        ability_name: String,
        toml_content: String,
    },
    #[serde(rename = "trait")]
    Trait {
        #[serde(alias = "trait_name")]
        name: String,
        description: String,
        #[serde(default)]
        passive_toml: Option<String>,
    },
    #[serde(rename = "quest")]
    Quest {
        #[serde(alias = "quest_title")]
        title: String,
        objective: String,
        item_name: String,
        #[serde(default)]
        item_flavor: String,
        reward_on_complete: Box<ProgressionReward>,
    },
}

fn default_cooldown_mult() -> f32 {
    1.0
}

/// Pending reward waiting for unconscious trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingProgression {
    pub hero_id: u32,
    pub reward: ProgressionReward,
    pub narrative_text: String,
    pub source_turn: u32,
}

/// Raw result from LFM subprocess.
#[derive(Debug, Clone)]
pub struct LfmProgressionResult {
    pub hero_id: u32,
    pub reward_json: String,
    pub narrative_text: String,
    pub success: bool,
    pub error: Option<String>,
}

/// Bevy resource tracking the whole narrative progression system.
#[derive(Resource)]
pub struct NarrativeProgressionState {
    pub in_flight: Vec<(u32, Arc<Mutex<Option<LfmProgressionResult>>>)>,
    pub pending: Vec<PendingProgression>,
}

impl Default for NarrativeProgressionState {
    fn default() -> Self {
        Self {
            in_flight: Vec::new(),
            pending: Vec::new(),
        }
    }
}
