use serde::{Deserialize, Serialize};

use super::generation::splitmix64;
use super::overworld_types::OverworldMap;
use super::roster_types::HeroCompanion;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompanionQuestKind {
    Reckoning,
    Homefront,
    RivalOath,
    NarrativeReward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompanionQuestStatus {
    Active,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionQuest {
    pub id: u32,
    pub hero_id: u32,
    pub kind: CompanionQuestKind,
    pub status: CompanionQuestStatus,
    pub title: String,
    pub objective: String,
    pub progress: u32,
    pub target: u32,
    pub issued_turn: u32,
    pub reward_loyalty: f32,
    pub reward_resolve: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionStoryState {
    pub quests: Vec<CompanionQuest>,
    pub next_id: u32,
    pub processed_ledger_len: usize,
    pub notice: String,
}

impl Default for CompanionStoryState {
    fn default() -> Self {
        CompanionStoryState {
            quests: Vec::new(),
            next_id: 1,
            processed_ledger_len: 0,
            notice: "No companion quest updates.".to_string(),
        }
    }
}

pub(crate) fn quest_for_hero(state: &CompanionStoryState, hero_id: u32) -> Option<&CompanionQuest> {
    state
        .quests
        .iter()
        .find(|q| q.hero_id == hero_id && q.status == CompanionQuestStatus::Active)
}

pub(crate) fn quest_for_hero_mut(
    state: &mut CompanionStoryState,
    hero_id: u32,
) -> Option<&mut CompanionQuest> {
    state
        .quests
        .iter_mut()
        .find(|q| q.hero_id == hero_id && q.status == CompanionQuestStatus::Active)
}

pub(crate) fn build_companion_quest(
    hero: &HeroCompanion,
    run_turn: u32,
    seed: u64,
    next_id: u32,
    overworld: &OverworldMap,
) -> CompanionQuest {
    let roll = splitmix64(seed ^ hero.id as u64 ^ run_turn as u64);
    let kind = match roll % 3 {
        0 => CompanionQuestKind::Reckoning,
        1 => CompanionQuestKind::Homefront,
        _ => CompanionQuestKind::RivalOath,
    };
    let region_name = overworld
        .regions
        .iter()
        .find(|r| r.id == hero.origin_region_id)
        .map(|r| r.name.as_str())
        .unwrap_or("the frontier");
    let faction_name = overworld
        .factions
        .get(hero.origin_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("their old banner");

    let (title, objective, target, reward_loyalty, reward_resolve) = match kind {
        CompanionQuestKind::Reckoning => (
            format!("{}: Reckoning Oath", hero.name),
            "Win two assigned missions without a defeat.".to_string(),
            2,
            6.0,
            4.0,
        ),
        CompanionQuestKind::Homefront => (
            format!("{}: Homefront Debt", hero.name),
            format!("Secure one victory tied to {}.", region_name),
            1,
            4.0,
            6.0,
        ),
        CompanionQuestKind::RivalOath => (
            format!("{}: Rival Banner", hero.name),
            format!("Claim two victories to weaken enemies of {}.", faction_name),
            2,
            5.0,
            5.0,
        ),
        CompanionQuestKind::NarrativeReward => (
            format!("{}: Revelation", hero.name),
            "Complete the narrative reward quest.".to_string(),
            1,
            3.0,
            3.0,
        ),
    };

    CompanionQuest {
        id: next_id,
        hero_id: hero.id,
        kind,
        status: CompanionQuestStatus::Active,
        title,
        objective,
        progress: 0,
        target,
        issued_turn: run_turn,
        reward_loyalty,
        reward_resolve,
    }
}
