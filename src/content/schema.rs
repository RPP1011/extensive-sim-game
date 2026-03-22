use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Tagged union of all content data types in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentData {
    HeroTemplate(HeroTemplateContent),
    EnemyTemplate(EnemyTemplateContent),
    Ability(AbilityContent),
    Faction(FactionContent),
    Settlement(SettlementContent),
    Npc(NpcContent),
    Quest(QuestContent),
    Dialogue(DialogueContent),
    Encounter(EncounterContent),
    ScenarioConfig(ScenarioConfigContent),
    // --- Tier 2 content types (Issue #15) ---
    Theme(ThemeContent),
    Region(RegionContent),
    Event(EventContent),
    Item(ItemContent),
    NarrativeArc(NarrativeArcContent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroTemplateContent {
    pub name: String,
    pub hp: f32,
    pub move_speed: f32,
    pub attack_power: f32,
    pub ability_power: f32,
    pub heal_power: f32,
    pub control_power: f32,
    pub attack_range: f32,
    pub attack_cooldown_ms: u32,
    pub abilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnemyTemplateContent {
    pub name: String,
    pub hp: f32,
    pub attack_power: f32,
    pub move_speed: f32,
    pub abilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbilityContent {
    pub name: String,
    pub description: String,
    pub cooldown_ms: u32,
    pub cast_time_ms: u32,
    pub range: f32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionContent {
    pub name: String,
    pub description: String,
    pub color_rgb: [u8; 3],
    pub motto: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementContent {
    pub name: String,
    pub region: String,
    pub population: u32,
    pub description: String,
    pub faction_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcContent {
    pub name: String,
    pub role: String,
    pub faction_id: String,
    pub settlement_id: Option<String>,
    pub dialogue_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestContent {
    pub name: String,
    pub description: String,
    pub objectives: Vec<String>,
    pub reward_description: String,
    pub prerequisite_quest_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueContent {
    pub speaker: String,
    pub lines: Vec<String>,
    pub choices: Vec<DialogueChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueChoice {
    pub text: String,
    pub next_dialogue_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncounterContent {
    pub name: String,
    pub description: String,
    pub enemy_template_ids: Vec<String>,
    pub difficulty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioConfigContent {
    pub name: String,
    pub seed: u64,
    pub hero_count: u32,
    pub enemy_count: u32,
    pub max_ticks: u32,
}

// ---------------------------------------------------------------------------
// Tier 2 content types (Issue #15 — AOT Content Generation Pipeline)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeContent {
    pub name: String,
    pub mood: String,
    pub color_palette: Vec<[u8; 3]>,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionContent {
    pub name: String,
    pub terrain_type: String,
    pub description: String,
    pub settlement_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventContent {
    pub name: String,
    pub description: String,
    pub triggers: Vec<String>,
    pub effects: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemContent {
    pub name: String,
    pub description: String,
    pub rarity: String,
    pub stats: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeArcContent {
    pub name: String,
    pub acts: Vec<String>,
    pub faction_ids: Vec<String>,
}
