use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::overworld_types::OverworldMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersonalityArchetype {
    Vanguard,
    Guardian,
    Tactician,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroCompanion {
    pub id: u32,
    pub name: String,
    pub origin_faction_id: usize,
    pub origin_region_id: usize,
    pub backstory: String,
    pub archetype: PersonalityArchetype,
    pub loyalty: f32,
    pub stress: f32,
    pub fatigue: f32,
    pub injury: f32,
    pub resolve: f32,
    pub active: bool,
    pub deserter: bool,
    #[serde(default)]
    pub xp: u32,
    #[serde(default = "default_hero_level")]
    pub level: u32,
    #[serde(default)]
    pub equipment: HeroEquipment,
    #[serde(default)]
    pub traits: Vec<HeroTrait>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroTrait {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub passive_toml: Option<String>,
}

fn default_hero_level() -> u32 {
    1
}

// ---------------------------------------------------------------------------
// Equipment types
// ---------------------------------------------------------------------------

/// All gear slots a hero can equip.  Each slot holds at most one item.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeroEquipment {
    pub weapon:    Option<EquipmentItem>,
    pub offhand:   Option<EquipmentItem>,
    pub chest:     Option<EquipmentItem>,
    pub boots:     Option<EquipmentItem>,
    pub accessory: Option<EquipmentItem>,
}

/// A single piece of equipment that modifies hero combat stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquipmentItem {
    pub name: String,
    pub rarity: ItemRarity,
    /// Added to `attack_damage` in the combat sim.
    pub attack_bonus: i32,
    /// Added to `max_hp` in the combat sim.
    pub hp_bonus: i32,
    /// Added to `move_speed` in the combat sim.
    pub speed_bonus: f32,
    /// Multiplied with `attack_cooldown_ms` in the combat sim (default 1.0 = no change).
    pub cooldown_mult: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemRarity {
    Standard,
    Rare,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecruitCandidate {
    pub id: u32,
    pub codename: String,
    pub origin_faction_id: usize,
    pub origin_region_id: usize,
    pub backstory: String,
    pub archetype: PersonalityArchetype,
    pub resolve: f32,
    pub loyalty_bias: f32,
    pub risk_tolerance: f32,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct CampaignRoster {
    pub heroes: Vec<HeroCompanion>,
    pub recruit_pool: Vec<RecruitCandidate>,
    #[serde(default)]
    pub player_hero_id: Option<u32>,
    pub next_id: u32,
    pub generation_counter: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PartyOrderKind {
    #[default]
    HoldPosition,
    PatrolNearby,
    ReinforceFront,
    RecruitAndTrain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignParty {
    pub id: u32,
    pub name: String,
    pub leader_hero_id: u32,
    pub region_id: usize,
    pub supply: f32,
    pub speed: f32,
    pub is_player_controlled: bool,
    #[serde(default)]
    pub order: PartyOrderKind,
    pub order_target_region_id: Option<usize>,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct CampaignParties {
    pub parties: Vec<CampaignParty>,
    pub selected_party_id: Option<u32>,
    pub next_id: u32,
    pub notice: String,
}

impl Default for CampaignParties {
    fn default() -> Self {
        CampaignParties {
            parties: Vec::new(),
            selected_party_id: None,
            next_id: 1,
            notice: "No active parties.".to_string(),
        }
    }
}

pub(crate) fn best_active_hero_id(roster: &CampaignRoster) -> Option<u32> {
    roster
        .heroes
        .iter()
        .filter(|h| h.active && !h.deserter)
        .max_by(|a, b| {
            let sa = (a.loyalty + a.resolve) - (a.stress + a.fatigue + a.injury);
            let sb = (b.loyalty + b.resolve) - (b.stress + b.fatigue + b.injury);
            sa.total_cmp(&sb).then(a.id.cmp(&b.id))
        })
        .map(|h| h.id)
}

pub fn bootstrap_campaign_parties(
    roster: &CampaignRoster,
    overworld: &OverworldMap,
) -> CampaignParties {
    let mut out = CampaignParties::default();
    if overworld.regions.is_empty() || roster.heroes.is_empty() {
        return out;
    }
    let player_leader = roster
        .player_hero_id
        .or_else(|| best_active_hero_id(roster))
        .or_else(|| roster.heroes.first().map(|h| h.id));
    let Some(player_leader) = player_leader else {
        return out;
    };
    let start_region = overworld
        .current_region
        .min(overworld.regions.len().saturating_sub(1));
    out.parties.push(CampaignParty {
        id: 1,
        name: "Main Company".to_string(),
        leader_hero_id: player_leader,
        region_id: start_region,
        supply: 100.0,
        speed: 1.0,
        is_player_controlled: true,
        order: PartyOrderKind::HoldPosition,
        order_target_region_id: None,
    });
    if let Some(secondary) = roster
        .heroes
        .iter()
        .find(|h| h.id != player_leader && h.active && !h.deserter)
    {
        let ai_region = overworld.regions[start_region]
            .neighbors
            .first()
            .copied()
            .unwrap_or(start_region);
        out.parties.push(CampaignParty {
            id: 2,
            name: "Ranging Band".to_string(),
            leader_hero_id: secondary.id,
            region_id: ai_region,
            supply: 100.0,
            speed: 0.9,
            is_player_controlled: false,
            order: PartyOrderKind::PatrolNearby,
            order_target_region_id: None,
        });
    }
    out.selected_party_id = out
        .parties
        .iter()
        .find(|p| p.is_player_controlled)
        .map(|p| p.id);
    out.next_id = out
        .parties
        .iter()
        .map(|p| p.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    out.notice = format!("{} parties active.", out.parties.len());
    out
}

pub fn normalize_campaign_parties(
    parties: &mut CampaignParties,
    roster: &CampaignRoster,
    overworld: &OverworldMap,
) {
    if overworld.regions.is_empty() {
        parties.parties.clear();
        parties.selected_party_id = None;
        parties.notice = "No overworld regions available for parties.".to_string();
        return;
    }
    parties.parties.retain(|party| {
        roster
            .heroes
            .iter()
            .any(|h| h.id == party.leader_hero_id && h.active && !h.deserter)
    });
    for party in &mut parties.parties {
        party.region_id = party
            .region_id
            .min(overworld.regions.len().saturating_sub(1));
        if party
            .order_target_region_id
            .map(|id| id >= overworld.regions.len())
            .unwrap_or(false)
        {
            party.order_target_region_id = None;
        }
    }
    if parties.parties.is_empty() {
        *parties = bootstrap_campaign_parties(roster, overworld);
        return;
    }
    if !parties.parties.iter().any(|p| p.is_player_controlled) {
        parties.parties[0].is_player_controlled = true;
    }
    if parties
        .parties
        .iter()
        .filter(|p| p.is_player_controlled)
        .count()
        > 1
    {
        let mut seen = false;
        for party in &mut parties.parties {
            if party.is_player_controlled {
                if !seen {
                    seen = true;
                } else {
                    party.is_player_controlled = false;
                }
            }
        }
    }
    let has_selected = parties
        .selected_party_id
        .and_then(|id| parties.parties.iter().find(|p| p.id == id))
        .is_some();
    if !has_selected {
        parties.selected_party_id = parties
            .parties
            .iter()
            .find(|p| p.is_player_controlled)
            .map(|p| p.id)
            .or_else(|| parties.parties.first().map(|p| p.id));
    }
    parties.next_id = parties
        .parties
        .iter()
        .map(|p| p.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
}

pub(crate) fn next_region_step(overworld: &OverworldMap, start: usize, target: usize) -> Option<usize> {
    if start >= overworld.regions.len() || target >= overworld.regions.len() || start == target {
        return None;
    }
    let mut came_from = vec![usize::MAX; overworld.regions.len()];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    came_from[start] = start;
    while let Some(node) = queue.pop_front() {
        if node == target {
            break;
        }
        for n in &overworld.regions[node].neighbors {
            if *n < came_from.len() && came_from[*n] == usize::MAX {
                came_from[*n] = node;
                queue.push_back(*n);
            }
        }
    }
    if came_from[target] == usize::MAX {
        return None;
    }
    let mut walk = target;
    while came_from[walk] != start {
        walk = came_from[walk];
    }
    Some(walk)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsequenceRecord {
    pub turn: u32,
    pub mission_name: String,
    pub result: super::types::MissionResult,
    pub hero_id: Option<u32>,
    pub summary: String,
}

#[derive(Resource, Debug, Clone, Default, Serialize, Deserialize)]
pub struct CampaignLedger {
    pub records: Vec<ConsequenceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignEvent {
    pub turn: u32,
    pub summary: String,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct CampaignEventLog {
    pub entries: Vec<CampaignEvent>,
    pub max_entries: usize,
}

impl Default for CampaignEventLog {
    fn default() -> Self {
        CampaignEventLog {
            entries: Vec::new(),
            max_entries: 120,
        }
    }
}

pub fn push_campaign_event(log: &mut CampaignEventLog, turn: u32, summary: String) {
    log.entries.push(CampaignEvent { turn, summary });
    if log.entries.len() > log.max_entries {
        let overflow = log.entries.len() - log.max_entries;
        log.entries.drain(0..overflow);
    }
}
