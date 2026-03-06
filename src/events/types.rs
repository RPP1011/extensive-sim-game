use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CampaignEventKind {
    MerchantOffer    { supply_cost: u32, item_description: String },
    DeserterIntel    { faction_id: usize, relation_boost: i32, risk_stress: f32 },
    PlagueScare      { region_id: usize, stress_penalty: f32 },
    RivalPartySpotted { region_id: usize, reward_supply: u32 },
    AllyRequest      { faction_id: usize, relation_reward: i32, supply_cost: u32 },
    FactionRumour    { faction_id: usize, description: String },
    AbandonedCache   { supply_reward: u32 },
    StormWarning     { turns_of_slow: u32 },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PendingCampaignEvent {
    pub id: u32,
    pub turn_generated: u32,
    pub kind: CampaignEventKind,
    pub title: String,
    pub description: String,
    /// None = pending, Some(true) = accepted, Some(false) = declined
    pub accepted: Option<bool>,
}

#[derive(Resource, Default, serde::Serialize, serde::Deserialize)]
pub struct CampaignEventQueue {
    pub events: Vec<PendingCampaignEvent>,
    pub next_id: u32,
    pub last_generated_turn: u32,
}
