use bevy::prelude::*;
use std::fs;

use crate::camera::CameraFocusTransitionState;
use crate::game_core::{
    self, CampaignSaveData, CharacterCreationState,
    MissionData, MissionProgress, MissionTactics,
    load_and_prepare_campaign_data,
};
use crate::region_nav::RegionTargetPickerState;
use crate::ui::save_browser::{
    CampaignSaveIndexState, CampaignSaveNotice, campaign_slot_path, continue_campaign_candidates,
    persist_campaign_save_index, save_slot_label,
    upsert_slot_metadata, CURRENT_SAVE_VERSION,
};
use crate::hub_types::StartMenuState;
use crate::game_core::{HubScreen, HubUiState};

use super::{
    campaign_save_metadata, despawn_all_mission_entities,
    spawn_mission_entities_from_snapshots, snapshot_campaign_progress_from_world,
    apply_campaign_progress_to_world, truncate_for_hud,
};

pub fn save_campaign_data(path: &str, data: &CampaignSaveData) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(data).map_err(|e| e.to_string())?;
    let save_path = std::path::Path::new(path);
    if let Some(parent) = save_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let backup_path = format!("{}.bak", path);
    let tmp_path = format!("{}.tmp", path);

    fs::write(&tmp_path, serialized).map_err(|e| e.to_string())?;
    if save_path.exists() {
        fs::copy(path, &backup_path).map_err(|e| e.to_string())?;
        fs::remove_file(path).map_err(|e| e.to_string())?;
    }
    fs::rename(&tmp_path, path).map_err(|e| e.to_string())
}

pub fn snapshot_campaign_from_world(world: &World) -> CampaignSaveData {
    let mut mission_snapshots = Vec::new();
    let mut active_mission_id = None;
    for entity_ref in world.iter_entities() {
        if let (Some(data), Some(progress), Some(tactics)) = (
            entity_ref.get::<MissionData>(),
            entity_ref.get::<MissionProgress>(),
            entity_ref.get::<MissionTactics>(),
        ) {
            let snapshot = game_core::MissionSnapshot::from_components(data, progress, tactics);
            if entity_ref.get::<game_core::ActiveMission>().is_some() {
                active_mission_id = Some(data.id);
            }
            mission_snapshots.push((data.id, snapshot));
        }
    }
    mission_snapshots.sort_by_key(|(id, _)| *id);
    let mission_snapshots = mission_snapshots.into_iter().map(|(_, s)| s).collect();

    CampaignSaveData {
        save_version: CURRENT_SAVE_VERSION,
        run_state: world.resource::<game_core::RunState>().clone(),
        mission_map: world.resource::<game_core::MissionMap>().clone(),
        attention_state: world.resource::<game_core::AttentionState>().clone(),
        overworld_map: world.resource::<game_core::OverworldMap>().clone(),
        commander_state: world.resource::<game_core::CommanderState>().clone(),
        diplomacy_state: world.resource::<game_core::DiplomacyState>().clone(),
        interaction_board: world.resource::<game_core::InteractionBoard>().clone(),
        campaign_roster: world.resource::<game_core::CampaignRoster>().clone(),
        campaign_parties: world
            .get_resource::<game_core::CampaignParties>()
            .cloned()
            .unwrap_or_default(),
        campaign_ledger: world.resource::<game_core::CampaignLedger>().clone(),
        campaign_event_log: world.resource::<game_core::CampaignEventLog>().clone(),
        companion_story_state: world.resource::<game_core::CompanionStoryState>().clone(),
        flashpoint_state: world
            .get_resource::<game_core::FlashpointState>()
            .cloned()
            .unwrap_or_default(),
        character_creation: world
            .get_resource::<CharacterCreationState>()
            .cloned()
            .unwrap_or_default(),
        campaign_progress: snapshot_campaign_progress_from_world(world),
        mission_snapshots,
        active_mission_id,
    }
}

pub fn apply_loaded_campaign_to_world(world: &mut World, loaded: CampaignSaveData) {
    world.insert_resource(loaded.run_state);
    world.insert_resource(loaded.mission_map);
    world.insert_resource(loaded.attention_state);
    world.insert_resource(loaded.overworld_map);
    world.insert_resource(loaded.commander_state);
    world.insert_resource(loaded.diplomacy_state);
    world.insert_resource(loaded.interaction_board);
    world.insert_resource(loaded.campaign_roster);
    world.insert_resource(loaded.campaign_parties);
    world.insert_resource(loaded.campaign_ledger);
    world.insert_resource(loaded.campaign_event_log);
    world.insert_resource(loaded.companion_story_state);
    world.insert_resource(loaded.flashpoint_state);
    world.insert_resource(loaded.character_creation);
    world.insert_resource(RegionTargetPickerState::default());
    world.insert_resource(CameraFocusTransitionState::default());
    despawn_all_mission_entities(world);
    spawn_mission_entities_from_snapshots(
        world,
        loaded.mission_snapshots,
        loaded.active_mission_id,
    );
}

pub fn save_campaign_to_slot(world: &mut World, slot: u8) -> Result<String, String> {
    let path = campaign_slot_path(slot);
    let data = snapshot_campaign_from_world(world);
    save_campaign_data(path, &data)?;
    let metadata = campaign_save_metadata(save_slot_label(slot), path, &data);
    {
        let mut index_state = world.resource_mut::<CampaignSaveIndexState>();
        upsert_slot_metadata(&mut index_state.index, slot, metadata);
        let _ = persist_campaign_save_index(&index_state.index);
    }
    Ok(format!(
        "Saved slot {} (v{}) t{} -> {}",
        slot, data.save_version, data.run_state.global_turn, path
    ))
}

pub fn load_campaign_from_path_into_world(
    world: &mut World,
    label: &str,
    path: &str,
) -> Result<String, String> {
    let loaded = load_and_prepare_campaign_data(path)?;
    let loaded_turn = loaded.run_state.global_turn;
    let loaded_version = loaded.save_version;
    let loaded_progress = loaded.campaign_progress.clone();
    let loaded_overworld = loaded.overworld_map.clone();
    let loaded_creation = loaded.character_creation.clone();
    apply_loaded_campaign_to_world(world, loaded);
    let resume_message =
        apply_campaign_progress_to_world(world, &loaded_progress, &loaded_overworld, &loaded_creation);
    Ok(format!(
        "Loaded {} (v{}) t{} <- {}. {}",
        label, loaded_version, loaded_turn, path, resume_message
    ))
}

pub fn hub_continue_campaign_requested_system(world: &mut World) {
    let requested = {
        let hub_ui = world.resource::<HubUiState>();
        hub_ui.request_continue_campaign
    };
    if !requested {
        return;
    }
    {
        let mut hub_ui = world.resource_mut::<HubUiState>();
        hub_ui.request_continue_campaign = false;
    }

    let candidates = {
        let index = &world.resource::<CampaignSaveIndexState>().index;
        continue_campaign_candidates(index)
    };
    if candidates.is_empty() {
        let msg = "No compatible campaign saves found.".to_string();
        world.resource_mut::<CampaignSaveNotice>().message = msg.clone();
        world.resource_mut::<StartMenuState>().status = msg;
        return;
    }

    let mut errors = Vec::new();
    let mut attempted = Vec::new();
    for candidate in candidates {
        attempted.push(candidate.label.clone());
        match load_campaign_from_path_into_world(world, &candidate.label, &candidate.path) {
            Ok(msg) => {
                world.resource_mut::<CampaignSaveNotice>().message = msg;
                let mut hub_ui = world.resource_mut::<HubUiState>();
                hub_ui.screen = HubScreen::OverworldMap;
                hub_ui.show_credits = false;
                hub_ui.request_quit = false;
                return;
            }
            Err(err) => {
                errors.push(format!("{} ({})", candidate.label, err));
            }
        }
    }

    let msg = format!(
        "Continue failed. Attempted: {}. Errors: {}",
        attempted.join(", "),
        truncate_for_hud(&errors.join(" | "), 180)
    );
    world.resource_mut::<CampaignSaveNotice>().message = msg.clone();
    world.resource_mut::<StartMenuState>().status = msg;
}
