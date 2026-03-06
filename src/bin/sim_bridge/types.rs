use std::collections::HashMap;

use bevy_game::ai::personality::PersonalityProfile;
use bevy_game::ai::squad::FormationMode;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Wire protocol types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct InitMessage {
    pub scenario: String,
    #[serde(default = "default_seed")]
    pub seed: Option<u64>,
    #[serde(default = "default_ticks")]
    pub ticks: u32,
    #[serde(default = "default_decision_interval")]
    pub decision_interval: u32,
}

fn default_seed() -> Option<u64> {
    None
}
fn default_ticks() -> u32 {
    320
}
fn default_decision_interval() -> u32 {
    10
}

#[derive(Debug, Deserialize)]
pub struct DecisionMessage {
    #[serde(default)]
    pub personality_updates: HashMap<String, PersonalityProfile>,
    #[serde(default)]
    pub squad_overrides: HashMap<String, SquadOverride>,
}

#[derive(Debug, Deserialize)]
pub struct SquadOverride {
    pub focus_target: Option<u32>,
    pub mode: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CondensedUnit {
    pub id: u32,
    pub team: String,
    pub role: String,
    pub hp: i32,
    pub max_hp: i32,
    pub hp_pct: f32,
    pub position: [f32; 2],
    pub attack_cd_remaining_ms: u32,
    pub ability_cd_remaining_ms: u32,
    pub heal_cd_remaining_ms: u32,
    pub control_cd_remaining_ms: u32,
    pub control_remaining_ms: u32,
    pub is_casting: bool,
}

#[derive(Debug, Serialize)]
pub struct CondensedSquad {
    pub focus_target: Option<u32>,
    pub mode: String,
}

#[derive(Debug, Serialize)]
pub struct CondensedPersonality {
    pub aggression: f32,
    pub risk_tolerance: f32,
    pub discipline: f32,
    pub control_bias: f32,
    pub altruism: f32,
    pub patience: f32,
}

#[derive(Debug, Serialize)]
pub struct CondensedEvent {
    pub kind: String,
    pub tick: u64,
    pub unit_id: Option<u32>,
    pub target_id: Option<u32>,
    pub amount: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct StateMessage {
    pub r#type: String,
    pub tick: u64,
    pub units: Vec<CondensedUnit>,
    pub squads: HashMap<String, CondensedSquad>,
    pub personality: HashMap<String, CondensedPersonality>,
    pub recent_events: Vec<CondensedEvent>,
    pub room_width: f32,
    pub room_depth: f32,
}

#[derive(Debug, Serialize)]
pub struct DoneMessage {
    pub r#type: String,
    pub winner: String,
    pub tick: u64,
    pub hero_hp_total: i32,
    pub enemy_hp_total: i32,
    pub hero_alive: usize,
    pub enemy_alive: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn parse_formation(s: &str) -> FormationMode {
    match s.to_lowercase().as_str() {
        "advance" => FormationMode::Advance,
        "retreat" => FormationMode::Retreat,
        _ => FormationMode::Hold,
    }
}
