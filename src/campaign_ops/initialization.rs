use bevy::prelude::*;

use crate::camera::CameraFocusTransitionState;
use crate::game_core::{
    self, CharacterCreationState, HubScreen, HubUiState,
};
use crate::hub_types::{
    CharacterCreationUiState, HubActionQueue, HubMenuState, StartMenuState,
};
use crate::region_nav::RegionTargetPickerState;
use crate::ui::save_browser::{
    CampaignAutosaveState, CampaignSaveNotice, CampaignSavePanelState,
};
use crate::game_loop::StartSceneState;
use crate::ui::settings::SettingsMenuState;

use super::{despawn_all_mission_entities, spawn_mission_entities_from_snapshots};

pub fn new_campaign_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x0A11_CE55_1BAD_C0DE);
    let mut z = nanos.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn initialize_new_campaign_world(world: &mut World) -> u64 {
    let seed = new_campaign_seed();
    let overworld = game_core::OverworldMap::from_seed(seed);
    let roster = game_core::CampaignRoster::default();
    let parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    world.insert_resource(game_core::RunState::default());
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::AttentionState::default());
    world.insert_resource(overworld);
    world.insert_resource(game_core::CommanderState::default());
    world.insert_resource(game_core::DiplomacyState::default());
    world.insert_resource(game_core::InteractionBoard::default());
    world.insert_resource(roster);
    world.insert_resource(parties);
    world.insert_resource(game_core::CampaignLedger::default());
    world.insert_resource(game_core::CampaignEventLog::default());
    world.insert_resource(game_core::CompanionStoryState::default());
    world.insert_resource(game_core::FlashpointState::default());
    world.insert_resource(CharacterCreationState::default());
    world.insert_resource(CharacterCreationUiState::default());
    world.insert_resource(RegionTargetPickerState::default());
    world.insert_resource(CameraFocusTransitionState::default());
    world.insert_resource(CampaignAutosaveState::default());
    world.insert_resource(HubActionQueue::default());
    world.insert_resource(CampaignSaveNotice {
        message: format!("New campaign started (seed {}).", seed),
    });
    world.insert_resource(CampaignSavePanelState::default());
    world.insert_resource(HubMenuState {
        selected: 0,
        notice: "New campaign initialized. Character creation is ready.".to_string(),
    });
    if let Some(mut start_scene) = world.get_resource_mut::<StartSceneState>() {
        start_scene.active = false;
    }
    if let Some(mut settings_menu) = world.get_resource_mut::<SettingsMenuState>() {
        settings_menu.is_open = false;
    }
    despawn_all_mission_entities(world);
    spawn_mission_entities_from_snapshots(world, Vec::new(), None);
    seed
}

pub fn hub_new_campaign_requested_system(world: &mut World) {
    let requested = {
        let hub_ui = world.resource::<HubUiState>();
        hub_ui.request_new_campaign
    };
    if !requested {
        return;
    }
    let seed = initialize_new_campaign_world(world);
    {
        let mut hub_ui = world.resource_mut::<HubUiState>();
        hub_ui.request_new_campaign = false;
        hub_ui.screen = HubScreen::CharacterCreationFaction;
        hub_ui.show_credits = false;
        hub_ui.request_quit = false;
    }
    world.resource_mut::<CampaignSaveNotice>().message = format!(
        "New campaign ready (seed {}). Choose a faction to continue.",
        seed
    );
}

pub fn enter_start_menu(hub_ui: &mut HubUiState, start_menu: &mut StartMenuState) {
    hub_ui.screen = HubScreen::StartMenu;
    start_menu.reset_for_entry();
}
