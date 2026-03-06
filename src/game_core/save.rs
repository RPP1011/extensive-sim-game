use serde::{Deserialize, Serialize};
use std::fs;

use super::companion::CompanionStoryState;
use super::overworld_types::*;
use super::roster_types::*;
use super::types::*;

// ── Save versioning constants ─────────────────────────────────────────────────

pub const SAVE_VERSION_V1: u32 = 1;
pub const SAVE_VERSION_V2: u32 = 2;
pub const CURRENT_SAVE_VERSION: u32 = 3;

pub fn default_save_version() -> u32 {
    SAVE_VERSION_V1
}

// ── Campaign save data types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CampaignLayerMarker {
    Menu,
    #[default]
    Overworld,
    Region,
    Local,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionTransitionPayload {
    pub region_id: usize,
    pub faction_id: String,
    pub faction_index: usize,
    pub campaign_seed: u64,
    pub region_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CampaignProgressState {
    #[serde(default)]
    pub current_layer: CampaignLayerMarker,
    #[serde(default)]
    pub current_region_id: Option<usize>,
    #[serde(default)]
    pub local_scene_id: Option<String>,
    #[serde(default)]
    pub intro_completed: bool,
    #[serde(default)]
    pub region_payload: Option<RegionTransitionPayload>,
    #[serde(default)]
    pub local_source_region_id: Option<usize>,
}

#[derive(bevy::prelude::Resource, Serialize, Deserialize, Clone, Default)]
pub struct CharacterCreationState {
    #[serde(default)]
    pub selected_faction_id: Option<String>,
    #[serde(default)]
    pub selected_faction_index: Option<usize>,
    #[serde(default)]
    pub selected_backstory_id: Option<String>,
    #[serde(default)]
    pub stat_modifiers: Vec<String>,
    #[serde(default)]
    pub recruit_bias_modifiers: Vec<String>,
    #[serde(default)]
    pub is_confirmed: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CampaignSaveData {
    #[serde(default = "default_save_version")]
    pub save_version: u32,
    pub run_state: RunState,
    #[serde(default)]
    pub mission_map: MissionMap,
    pub attention_state: AttentionState,
    pub overworld_map: OverworldMap,
    pub commander_state: CommanderState,
    pub diplomacy_state: DiplomacyState,
    pub interaction_board: InteractionBoard,
    pub campaign_roster: CampaignRoster,
    #[serde(default)]
    pub campaign_parties: CampaignParties,
    pub campaign_ledger: CampaignLedger,
    #[serde(default)]
    pub campaign_event_log: CampaignEventLog,
    #[serde(default)]
    pub companion_story_state: CompanionStoryState,
    #[serde(default)]
    pub flashpoint_state: FlashpointState,
    #[serde(default)]
    pub character_creation: CharacterCreationState,
    #[serde(default)]
    pub campaign_progress: CampaignProgressState,
    /// Serialized mission entities (replaces the old MissionBoard resource data).
    #[serde(default)]
    pub mission_snapshots: Vec<MissionSnapshot>,
    /// The `id` field of the mission entity that has `ActiveMission`.
    #[serde(default)]
    pub active_mission_id: Option<u32>,
}

// ── Campaign save migration helpers ──────────────────────────────────────────

pub fn derive_region_transition_seed(
    campaign_seed: u64,
    region_id: usize,
    faction_index: usize,
) -> u64 {
    campaign_seed
        ^ ((region_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ ((faction_index as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
}

pub fn validate_region_transition_payload(
    payload: &RegionTransitionPayload,
    overworld: &OverworldMap,
) -> Result<(), String> {
    if overworld.regions.get(payload.region_id).is_none() {
        return Err("selected region is no longer available".to_string());
    }
    if payload.faction_id.trim().is_empty() {
        return Err("missing faction context".to_string());
    }
    if payload.faction_index >= overworld.factions.len() {
        return Err("invalid faction context".to_string());
    }
    if payload.region_seed
        != derive_region_transition_seed(
            payload.campaign_seed,
            payload.region_id,
            payload.faction_index,
        )
    {
        return Err("deterministic seed contract mismatch".to_string());
    }
    Ok(())
}

fn validate_campaign_party_fields(raw: &serde_json::Value) -> Result<(), String> {
    let root = raw
        .as_object()
        .ok_or_else(|| "incompatible save: root payload is not an object".to_string())?;
    let campaign_parties = root
        .get("campaign_parties")
        .ok_or_else(|| "incompatible save: missing campaign_parties".to_string())?
        .as_object()
        .ok_or_else(|| "incompatible save: campaign_parties must be an object".to_string())?;

    for field in ["parties", "selected_party_id", "next_id", "notice"] {
        if !campaign_parties.contains_key(field) {
            return Err(format!(
                "incompatible save: missing campaign_parties.{field}"
            ));
        }
    }

    let parties = campaign_parties
        .get("parties")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            "incompatible save: campaign_parties.parties must be an array".to_string()
        })?;

    for (idx, party) in parties.iter().enumerate() {
        let party_obj = party.as_object().ok_or_else(|| {
            format!("incompatible save: campaign_parties.parties[{idx}] must be an object")
        })?;
        for field in [
            "id",
            "name",
            "leader_hero_id",
            "region_id",
            "supply",
            "speed",
            "is_player_controlled",
            "order",
            "order_target_region_id",
        ] {
            if !party_obj.contains_key(field) {
                return Err(format!(
                    "incompatible save: missing campaign_parties.parties[{idx}].{field}"
                ));
            }
        }
    }

    Ok(())
}

pub fn load_campaign_data(path: &str) -> Result<CampaignSaveData, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let raw: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    validate_campaign_party_fields(&raw)?;
    serde_json::from_value::<CampaignSaveData>(raw).map_err(|e| e.to_string())
}

pub fn migrate_campaign_save_data(mut save: CampaignSaveData) -> Result<CampaignSaveData, String> {
    if save.save_version == 0 {
        save.save_version = SAVE_VERSION_V1;
    }
    match save.save_version {
        SAVE_VERSION_V1 => {
            // v1 -> v2/v3: explicit versioning and event log defaults.
            save.save_version = CURRENT_SAVE_VERSION;
            Ok(save)
        }
        SAVE_VERSION_V2 => {
            // v2 -> v3: campaign progression markers.
            save.save_version = CURRENT_SAVE_VERSION;
            Ok(save)
        }
        CURRENT_SAVE_VERSION => Ok(save),
        other if other > CURRENT_SAVE_VERSION => Err(format!(
            "incompatible save: version {} is newer than supported {}",
            other, CURRENT_SAVE_VERSION
        )),
        other => Err(format!("incompatible save: unsupported version {}", other)),
    }
}

pub fn normalize_loaded_campaign(save: &mut CampaignSaveData) {
    if save.mission_snapshots.is_empty() {
        save.mission_snapshots = default_mission_snapshots();
        save.active_mission_id = None;
    }
    if save.overworld_map.regions.is_empty() {
        save.overworld_map = OverworldMap::default();
    }
    save.overworld_map.current_region = save
        .overworld_map
        .current_region
        .min(save.overworld_map.regions.len().saturating_sub(1));
    save.overworld_map.selected_region = save
        .overworld_map
        .selected_region
        .min(save.overworld_map.regions.len().saturating_sub(1));
    if save.interaction_board.selected >= save.interaction_board.offers.len() {
        save.interaction_board.selected = save.interaction_board.offers.len().saturating_sub(1);
    }
    if save.companion_story_state.processed_ledger_len > save.campaign_ledger.records.len() {
        save.companion_story_state.processed_ledger_len = save.campaign_ledger.records.len();
    }
    if save.campaign_event_log.max_entries == 0 {
        save.campaign_event_log.max_entries = 120;
    }
    if save.campaign_event_log.entries.len() > save.campaign_event_log.max_entries {
        let overflow = save.campaign_event_log.entries.len() - save.campaign_event_log.max_entries;
        save.campaign_event_log.entries.drain(0..overflow);
    }
    let valid_player = save.campaign_roster.player_hero_id.and_then(|id| {
        save.campaign_roster
            .heroes
            .iter()
            .find(|h| h.id == id && h.active && !h.deserter)
            .map(|h| h.id)
    });
    if valid_player.is_none() {
        save.campaign_roster.player_hero_id = save
            .campaign_roster
            .heroes
            .iter()
            .find(|h| h.active && !h.deserter)
            .or_else(|| save.campaign_roster.heroes.first())
            .map(|h| h.id);
    }
    normalize_campaign_parties(
        &mut save.campaign_parties,
        &save.campaign_roster,
        &save.overworld_map,
    );
    if save
        .character_creation
        .selected_faction_index
        .is_some_and(|idx| idx >= save.overworld_map.factions.len())
    {
        save.character_creation.selected_faction_index = None;
        save.character_creation.selected_faction_id = None;
    }
    save.flashpoint_state
        .chains
        .retain(|c| !c.completed && c.stage >= 1 && c.stage <= 3);
    if save
        .campaign_progress
        .current_region_id
        .is_some_and(|idx| idx >= save.overworld_map.regions.len())
    {
        save.campaign_progress.current_region_id = None;
    }
    if let Some(payload) = save.campaign_progress.region_payload.as_ref() {
        if validate_region_transition_payload(payload, &save.overworld_map).is_err() {
            save.campaign_progress.region_payload = None;
        }
    }
    if save.campaign_progress.current_layer == CampaignLayerMarker::Local
        && save.campaign_progress.local_scene_id.is_none()
    {
        save.campaign_progress.local_scene_id = Some("local-eagle-eye-intro".to_string());
    }
}

pub fn validate_and_repair_loaded_campaign(save: &mut CampaignSaveData) -> Vec<String> {
    let mut warnings = Vec::new();

    if save.commander_state.commanders.len() < save.overworld_map.factions.len() {
        warnings.push("Commander roster was incomplete and has been reset.".to_string());
        save.commander_state = CommanderState::default();
    }
    if save.diplomacy_state.relations.len() != save.overworld_map.factions.len()
        || save
            .diplomacy_state
            .relations
            .iter()
            .any(|row| row.len() != save.overworld_map.factions.len())
    {
        warnings.push("Diplomacy matrix was invalid and has been reset.".to_string());
        save.diplomacy_state = DiplomacyState::default();
    }
    if save.diplomacy_state.player_faction_id >= save.overworld_map.factions.len() {
        warnings.push("Player faction id was out-of-range and has been reset.".to_string());
        save.diplomacy_state.player_faction_id = 0;
    }

    for mission in &mut save.mission_snapshots {
        if let Some(bound) = mission.bound_region_id {
            if !save.overworld_map.regions.iter().any(|r| r.id == bound) {
                warnings.push(format!(
                    "Mission '{}' had invalid region binding and was detached.",
                    mission.mission_name
                ));
                mission.bound_region_id = None;
            }
        }
    }
    for region in &mut save.overworld_map.regions {
        if region.owner_faction_id >= save.overworld_map.factions.len() {
            warnings.push(format!(
                "Region '{}' had invalid owner and was reassigned.",
                region.name
            ));
            region.owner_faction_id = 0;
        }
        if region
            .mission_slot
            .map(|s| s >= save.mission_snapshots.len())
            .unwrap_or(false)
        {
            warnings.push(format!(
                "Region '{}' had invalid mission slot and was cleared.",
                region.name
            ));
            region.mission_slot = None;
        }
    }

    for quest in &mut save.companion_story_state.quests {
        if !save
            .campaign_roster
            .heroes
            .iter()
            .any(|h| h.id == quest.hero_id)
        {
            warnings.push(format!(
                "Quest '{}' pointed to missing hero and was marked failed.",
                quest.title
            ));
            quest.status = super::companion::CompanionQuestStatus::Failed;
        }
        if quest.progress > quest.target {
            quest.progress = quest.target;
        }
    }
    if let Some(player_id) = save.campaign_roster.player_hero_id {
        if !save
            .campaign_roster
            .heroes
            .iter()
            .any(|h| h.id == player_id)
        {
            warnings.push("Player hero id was invalid and has been reassigned.".to_string());
            save.campaign_roster.player_hero_id = save
                .campaign_roster
                .heroes
                .iter()
                .find(|h| h.active && !h.deserter)
                .or_else(|| save.campaign_roster.heroes.first())
                .map(|h| h.id);
        }
    }
    for chain in &mut save.flashpoint_state.chains {
        if chain.stage == 0 || chain.stage > 3 {
            warnings.push(format!(
                "Flashpoint chain {} had invalid stage and was reset to stage 1.",
                chain.id
            ));
            chain.stage = 1;
        }
        if chain.region_id >= save.overworld_map.regions.len() {
            warnings.push(format!(
                "Flashpoint chain {} pointed to missing region and was dropped.",
                chain.id
            ));
            chain.completed = true;
        }
    }
    save.flashpoint_state.chains.retain(|c| !c.completed);

    warnings
}

pub fn load_and_prepare_campaign_data(path: &str) -> Result<CampaignSaveData, String> {
    let raw = load_campaign_data(path)?;
    let mut migrated = migrate_campaign_save_data(raw)?;
    normalize_loaded_campaign(&mut migrated);
    let warnings = validate_and_repair_loaded_campaign(&mut migrated);
    if !warnings.is_empty() {
        let turn = migrated.run_state.global_turn;
        for msg in warnings {
            migrated
                .campaign_event_log
                .entries
                .push(CampaignEvent {
                    turn,
                    summary: format!("Save repair: {}", msg),
                });
        }
        if migrated.campaign_event_log.entries.len() > migrated.campaign_event_log.max_entries {
            let overflow =
                migrated.campaign_event_log.entries.len() - migrated.campaign_event_log.max_entries;
            migrated.campaign_event_log.entries.drain(0..overflow);
        }
    }
    Ok(migrated)
}
