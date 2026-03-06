mod save_load;
mod initialization;

use bevy::prelude::*;

use crate::game_core::{
    self, AssignedHero, CampaignLayerMarker, CampaignProgressState,
    CampaignSaveData, CharacterCreationState, HubScreen, HubUiState,
    MissionData, RegionTransitionPayload,
    derive_region_transition_seed, validate_region_transition_payload,
};
use crate::local_intro::{
    LocalEagleEyeIntroState, LocalIntroPhase, local_intro_anchor_for_region,
};
use crate::region_nav::RegionLayerTransitionState;
use crate::ui::save_browser::{SaveSlotMetadata, CURRENT_SAVE_VERSION};

// Re-exports from save_load
pub use save_load::{
    save_campaign_data, snapshot_campaign_from_world, save_campaign_to_slot,
    load_campaign_from_path_into_world, hub_continue_campaign_requested_system,
};

// Re-exports from initialization
pub use initialization::{
    hub_new_campaign_requested_system, enter_start_menu,
};

// --- Utility functions kept in mod.rs ---

pub fn unix_now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn campaign_save_metadata(
    slot: String,
    path: &str,
    data: &CampaignSaveData,
) -> SaveSlotMetadata {
    SaveSlotMetadata {
        slot,
        path: path.to_string(),
        save_version: data.save_version,
        compatible: data.save_version <= CURRENT_SAVE_VERSION,
        global_turn: data.run_state.global_turn,
        map_seed: data.overworld_map.map_seed,
        saved_unix_seconds: unix_now_seconds(),
    }
}

pub fn spawn_mission_entities_from_snapshots(
    world: &mut World,
    snapshots: Vec<game_core::MissionSnapshot>,
    active_mission_id: Option<u32>,
) {
    let defaults = if snapshots.is_empty() {
        game_core::default_mission_snapshots()
    } else {
        snapshots
    };
    for (i, snap) in defaults.into_iter().enumerate() {
        let id = {
            let mut board = world.resource_mut::<game_core::MissionBoard>();
            let id = board.next_id;
            board.next_id += 1;
            id
        };
        let is_active = active_mission_id.map_or(i == 0, |aid| aid == id);
        let (data, progress, tactics) = snap.into_components(id);
        let entity = if is_active {
            world
                .spawn((
                    data,
                    progress,
                    tactics,
                    AssignedHero::default(),
                    game_core::ActiveMission,
                ))
                .id()
        } else {
            world
                .spawn((data, progress, tactics, AssignedHero::default()))
                .id()
        };
        world
            .resource_mut::<game_core::MissionBoard>()
            .entities
            .push(entity);
    }
}

pub fn despawn_all_mission_entities(world: &mut World) {
    let entities: Vec<bevy::prelude::Entity> = world
        .iter_entities()
        .filter(|e| e.contains::<MissionData>())
        .map(|e| e.id())
        .collect();
    let mut board = world.resource_mut::<game_core::MissionBoard>();
    board.entities.clear();
    board.next_id = 0;
    drop(board);
    for entity in entities {
        world.despawn(entity);
    }
}

pub fn marker_from_hub_screen(screen: HubScreen) -> CampaignLayerMarker {
    match screen {
        HubScreen::StartMenu | HubScreen::BackstoryCinematic => CampaignLayerMarker::Menu,
        HubScreen::RegionView => CampaignLayerMarker::Region,
        HubScreen::LocalEagleEyeIntro => CampaignLayerMarker::Local,
        _ => CampaignLayerMarker::Overworld,
    }
}

pub fn hub_screen_from_marker(marker: CampaignLayerMarker) -> HubScreen {
    match marker {
        CampaignLayerMarker::Menu => HubScreen::StartMenu,
        CampaignLayerMarker::Overworld => HubScreen::OverworldMap,
        CampaignLayerMarker::Region => HubScreen::RegionView,
        CampaignLayerMarker::Local => HubScreen::LocalEagleEyeIntro,
    }
}

pub fn snapshot_campaign_progress_from_world(world: &World) -> CampaignProgressState {
    let current_layer = world
        .get_resource::<HubUiState>()
        .map(|hub| marker_from_hub_screen(hub.screen))
        .unwrap_or(CampaignLayerMarker::Overworld);
    let current_region_id = world
        .get_resource::<LocalEagleEyeIntroState>()
        .and_then(|local| local.source_region_id)
        .or_else(|| {
            world
                .get_resource::<RegionLayerTransitionState>()
                .and_then(|region| region.active_payload.as_ref().map(|payload| payload.region_id))
        })
        .or_else(|| {
            world
                .get_resource::<game_core::OverworldMap>()
                .map(|overworld| overworld.selected_region)
        });
    let region_payload = world
        .get_resource::<RegionLayerTransitionState>()
        .and_then(|region| region.active_payload.clone());
    let local_source_region_id = world
        .get_resource::<LocalEagleEyeIntroState>()
        .and_then(|local| local.source_region_id);
    let intro_completed = world
        .get_resource::<LocalEagleEyeIntroState>()
        .map(|local| local.intro_completed)
        .unwrap_or(false);
    let local_scene_id = if current_layer == CampaignLayerMarker::Local {
        Some("local-eagle-eye-intro".to_string())
    } else {
        None
    };

    CampaignProgressState {
        current_layer,
        current_region_id,
        local_scene_id,
        intro_completed,
        region_payload,
        local_source_region_id,
    }
}

pub fn derive_region_payload_from_progress(
    progress: &CampaignProgressState,
    overworld: &game_core::OverworldMap,
    character_creation: &CharacterCreationState,
) -> Option<RegionTransitionPayload> {
    if let Some(payload) = progress.region_payload.clone() {
        return Some(payload);
    }
    let region_id = progress.current_region_id?;
    let faction_id = character_creation.selected_faction_id.clone()?;
    let faction_index = character_creation.selected_faction_index?;
    if region_id >= overworld.regions.len() || faction_index >= overworld.factions.len() {
        return None;
    }
    let campaign_seed = overworld.map_seed;
    let region_seed = derive_region_transition_seed(campaign_seed, region_id, faction_index);
    Some(RegionTransitionPayload {
        region_id,
        faction_id,
        faction_index,
        campaign_seed,
        region_seed,
    })
}

pub fn apply_campaign_progress_to_world(
    world: &mut World,
    progress: &CampaignProgressState,
    overworld: &game_core::OverworldMap,
    character_creation: &CharacterCreationState,
) -> String {
    if let Some(mut region_transition) = world.get_resource_mut::<RegionLayerTransitionState>() {
        *region_transition = RegionLayerTransitionState::default();
    }
    if let Some(mut local_intro) = world.get_resource_mut::<LocalEagleEyeIntroState>() {
        *local_intro = LocalEagleEyeIntroState::default();
    }

    let Some(mut hub_ui) = world.get_resource_mut::<HubUiState>() else {
        return "Loaded save data (layer restore unavailable: missing UI state).".to_string();
    };
    hub_ui.screen = hub_screen_from_marker(progress.current_layer);

    match progress.current_layer {
        CampaignLayerMarker::Menu => "Loaded campaign and resumed Start Menu.".to_string(),
        CampaignLayerMarker::Overworld => {
            "Loaded campaign and resumed Overworld Map.".to_string()
        }
        CampaignLayerMarker::Region => {
            let Some(payload) =
                derive_region_payload_from_progress(progress, overworld, character_creation)
            else {
                hub_ui.screen = HubScreen::OverworldMap;
                return "Loaded campaign; region resume context was incomplete, resumed Overworld Map."
                    .to_string();
            };
            if let Err(reason) = validate_region_transition_payload(&payload, overworld) {
                hub_ui.screen = HubScreen::OverworldMap;
                return format!(
                    "Loaded campaign; region resume context was invalid ({reason}), resumed Overworld Map."
                );
            }
            drop(hub_ui);
            if let Some(mut region_transition) = world.get_resource_mut::<RegionLayerTransitionState>()
            {
                region_transition.active_payload = Some(payload.clone());
                region_transition.pending_payload = None;
                region_transition.pending_frames = 0;
                region_transition.interaction_locked = false;
                let region_name = overworld
                    .regions
                    .get(payload.region_id)
                    .map(|region| region.name.as_str())
                    .unwrap_or("Unknown");
                region_transition.status = format!(
                    "Loaded region context: {} (id {}, faction {}, campaign seed {}, region seed {}).",
                    region_name,
                    payload.region_id,
                    payload.faction_id,
                    payload.campaign_seed,
                    payload.region_seed
                );
            }
            "Loaded campaign and resumed Region layer.".to_string()
        }
        CampaignLayerMarker::Local => {
            let Some(payload) =
                derive_region_payload_from_progress(progress, overworld, character_creation)
            else {
                hub_ui.screen = HubScreen::OverworldMap;
                return "Loaded campaign; local resume context was incomplete, resumed Overworld Map."
                    .to_string();
            };
            if let Err(reason) = validate_region_transition_payload(&payload, overworld) {
                hub_ui.screen = HubScreen::OverworldMap;
                return format!(
                    "Loaded campaign; local resume context was invalid ({reason}), resumed Overworld Map."
                );
            }
            let Some(anchor) = local_intro_anchor_for_region(payload.region_id) else {
                hub_ui.screen = HubScreen::RegionView;
                if let Some(mut region_transition) =
                    world.get_resource_mut::<RegionLayerTransitionState>()
                {
                    region_transition.active_payload = Some(payload.clone());
                    region_transition.pending_payload = None;
                    region_transition.pending_frames = 0;
                    region_transition.interaction_locked = false;
                    region_transition.status = format!(
                        "Loaded region context but local anchor was unavailable for region id {}.",
                        payload.region_id
                    );
                }
                return "Loaded campaign; local anchor unavailable, resumed Region layer."
                    .to_string();
            };
            drop(hub_ui);
            if let Some(mut region_transition) = world.get_resource_mut::<RegionLayerTransitionState>()
            {
                region_transition.active_payload = Some(payload.clone());
                region_transition.pending_payload = None;
                region_transition.pending_frames = 0;
                region_transition.interaction_locked = false;
                region_transition.status = format!(
                    "Region context restored for local scene (region id {}).",
                    payload.region_id
                );
            }
            if let Some(mut local_intro) = world.get_resource_mut::<LocalEagleEyeIntroState>() {
                local_intro.source_region_id = progress
                    .local_source_region_id
                    .or(progress.current_region_id)
                    .or(Some(payload.region_id));
                local_intro.anchor = Some(anchor);
                local_intro.phase = if progress.intro_completed {
                    LocalIntroPhase::GameplayControl
                } else {
                    LocalIntroPhase::HiddenInside
                };
                local_intro.phase_frames = 0;
                local_intro.intro_completed = progress.intro_completed;
                local_intro.input_handoff_ready = progress.intro_completed;
                local_intro.status = if progress.intro_completed {
                    "Local intro restored at gameplay control.".to_string()
                } else {
                    "Local intro restored at beginning of intro sequence.".to_string()
                };
            }
            "Loaded campaign and resumed Local layer.".to_string()
        }
    }
}

pub fn format_slot_meta(meta: Option<&SaveSlotMetadata>) -> String {
    match meta {
        Some(m) => format!(
            "{} t{} v{} seed={} ts={} {}",
            m.slot,
            m.global_turn,
            m.save_version,
            m.map_seed,
            m.saved_unix_seconds,
            if m.compatible { "ok" } else { "incompatible" }
        ),
        None => "empty".to_string(),
    }
}

pub fn format_slot_badge(meta: Option<&SaveSlotMetadata>) -> &'static str {
    match meta {
        Some(m) if m.compatible => "[OK]",
        Some(_) => "[NEWER]",
        None => "[EMPTY]",
    }
}

pub fn truncate_for_hud(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in value.chars().take(max_chars) {
        out.push(ch);
    }
    if value.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}
