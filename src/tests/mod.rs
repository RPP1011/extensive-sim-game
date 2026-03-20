use super::*;
use bevy::ecs::schedule::Schedule;
use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::time::{SystemTime, UNIX_EPOCH};

// Imports that were removed from main.rs scope but are needed by tests
use bevy::input::mouse::{MouseMotion, MouseWheel};
use crate::camera::{
    CameraFocusTransitionQueueResult, CameraFocusTrigger, CameraSettings,
};
use crate::ui::settings::{
    OrbitSensitivitySliderFill, OrbitSensitivityLabel,
    ZoomSensitivitySliderFill, ZoomSensitivityLabel,
    InvertOrbitYLabel,
};
use crate::game_core::{
    self, MissionResult,
    CampaignLayerMarker, CampaignProgressState, RegionTransitionPayload,
    derive_region_transition_seed,
    load_campaign_data, load_and_prepare_campaign_data,
    migrate_campaign_save_data, normalize_loaded_campaign,
    validate_and_repair_loaded_campaign,
};
use crate::ui::save_browser::{
    campaign_slot_path, continue_campaign_candidates, upsert_slot_metadata,
    CampaignSaveIndex, SaveSlotMetadata,
    CAMPAIGN_AUTOSAVE_PATH, CAMPAIGN_SAVE_PATH,
    CAMPAIGN_SAVE_SLOT_2_PATH, CAMPAIGN_SAVE_SLOT_3_PATH,
    CURRENT_SAVE_VERSION, SAVE_VERSION_V1,
    build_save_preview, panel_selected_entry,
};
use crate::game_loop::hub_runtime_input_enabled;
use crate::region_nav::{
    RegionTargetPickerState, begin_region_target_picker,
    update_region_target_picker_selection, confirm_region_target_picker,
    party_target_region_label, party_panel_label,
    transfer_direct_command_to_selected,
    build_region_transition_payload, request_enter_selected_region,
    advance_region_layer_transition,
};
use crate::character_select::{
    build_faction_selection_choices, confirm_faction_selection, confirm_backstory_selection,
};
use crate::campaign_ops::{
    snapshot_campaign_from_world, apply_loaded_campaign_to_world, format_slot_badge,
    save_campaign_data,
    load_campaign_from_path_into_world, enter_start_menu,
};
use crate::local_intro::{self, LocalIntroPhase, advance_local_eagle_eye_intro, bootstrap_local_eagle_eye_intro, local_intro_anchor_for_region};
use crate::hub_types::HubAction;
use crate::hub_systems::apply_hub_action;
use crate::campaign_ops::{
    hub_new_campaign_requested_system, hub_continue_campaign_requested_system,
};
use crate::ui::save_browser::campaign_autosave_system;
use crate::ui::settings::{
    update_settings_menu_visual_system, settings_menu_toggle_system,
    settings_menu_slider_input_system, settings_menu_toggle_input_system,
};
use crate::camera::{orbit_camera_controller_system, persist_camera_settings_system};
use crate::game_loop::increment_global_turn;
use crate::scenario_3d::update_mission_hud_system;

mod hub_tests;
mod region_tests;
mod campaign_save_tests;
mod campaign_load_tests;
mod campaign_index_tests;
mod simulation_tests;
mod regression_tests;
mod mission_replay_tests;

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

fn recruit_candidate(
    id: u32,
    archetype: game_core::PersonalityArchetype,
) -> game_core::RecruitCandidate {
    game_core::RecruitCandidate {
        id,
        codename: format!("candidate-{id}"),
        origin_faction_id: 0,
        origin_region_id: 0,
        backstory: "test".to_string(),
        archetype,
        resolve: 50.0,
        loyalty_bias: 50.0,
        risk_tolerance: 50.0,
    }
}

fn valid_character_creation_state() -> CharacterCreationState {
    CharacterCreationState {
        selected_faction_id: Some("faction-0-test".to_string()),
        selected_faction_index: Some(0),
        selected_backstory_id: Some("scout-pathfinder".to_string()),
        stat_modifiers: Vec::new(),
        recruit_bias_modifiers: Vec::new(),
        is_confirmed: true,
    }
}

fn build_layer_resume_test_world() -> World {
    let mut world = World::new();
    world.insert_resource(RunState::default());
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
    world.insert_resource(game_core::AttentionState::default());
    world.insert_resource(game_core::OverworldMap::default());
    world.insert_resource(game_core::CommanderState::default());
    world.insert_resource(game_core::DiplomacyState::default());
    world.insert_resource(game_core::InteractionBoard::default());
    world.insert_resource(game_core::CampaignRoster::default());
    world.insert_resource(game_core::CampaignParties::default());
    world.insert_resource(game_core::CampaignLedger::default());
    world.insert_resource(game_core::CampaignEventLog::default());
    world.insert_resource(game_core::CompanionStoryState::default());
    world.insert_resource(game_core::FlashpointState::default());
    world.insert_resource(HubUiState {
        screen: HubScreen::StartMenu,
        show_credits: false,
        request_quit: false,
        request_new_campaign: false,
        request_continue_campaign: false,
    });
    world.insert_resource(RegionLayerTransitionState::default());
    world.insert_resource(LocalEagleEyeIntroState::default());
    world
}

fn campaign_signature_from_world(world: &World) -> u64 {
    let data = snapshot_campaign_from_world(world);
    let bytes = serde_json::to_vec(&data).expect("serialize campaign");
    bytes
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |acc, b| {
            (acc ^ b as u64).wrapping_mul(0x1000_0000_01b3)
        })
}

fn build_campaign_sim_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            game_core::attention_management_system,
            game_core::overworld_cooldown_system,
            game_core::overworld_sync_from_missions_system,
            game_core::overworld_faction_autonomy_system,
            game_core::simulate_unfocused_missions_system,
            game_core::sync_roster_lore_with_overworld_system,
            game_core::overworld_ai_border_pressure_system,
            game_core::overworld_intel_update_system,
        )
            .chain(),
    );
    schedule.add_systems(
        (
            game_core::update_faction_war_goals_system,
            game_core::pressure_spawn_missions_system,
            game_core::sync_mission_assignments_system,
            game_core::companion_mission_impact_system,
            game_core::companion_state_drift_system,
            game_core::resolve_mission_consequences_system,
            game_core::flashpoint_progression_system,
            game_core::progress_companion_story_quests_system,
            game_core::generate_companion_story_quests_system,
        )
            .chain(),
    );
    schedule.add_systems(
        (
            game_core::companion_recovery_system,
            game_core::generate_commander_intents_system,
            game_core::refresh_interaction_offers_system,
        )
            .chain(),
    );
    schedule
}

fn build_campaign_test_world(seed: u64) -> World {
    let mut world = World::new();
    let overworld = game_core::OverworldMap::from_seed(seed);
    let roster = game_core::CampaignRoster::default();
    let parties = game_core::bootstrap_campaign_parties(&roster, &overworld);
    world.insert_resource(RunState::default());
    world.insert_resource(game_core::MissionMap::default());
    world.insert_resource(game_core::MissionBoard::default());
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
    spawn_mission_entities_from_snapshots(
        &mut world,
        game_core::default_mission_snapshots(),
        None,
    );
    world
}

fn canonical_roundtrip_world(world: &mut World) -> Vec<String> {
    let data = snapshot_campaign_from_world(world);
    let json = serde_json::to_string(&data).expect("save chain serialize");
    let parsed: CampaignSaveData =
        serde_json::from_str(&json).expect("save chain deserialize");
    let mut migrated = migrate_campaign_save_data(parsed).expect("migrate");
    normalize_loaded_campaign(&mut migrated);
    let repairs = validate_and_repair_loaded_campaign(&mut migrated);
    apply_loaded_campaign_to_world(world, migrated);
    repairs
}
