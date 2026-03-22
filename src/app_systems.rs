//! Bevy app system registration — extracted from main() to reduce main.rs size.

use bevy::prelude::*;

use crate::audio;
use crate::progression;
use crate::ascii_viewport::{MissionPaneState, draw_ascii_viewport_system};
use crate::camera::{persist_camera_settings_system, setup_camera};
use crate::events;
use crate::game_core;
use crate::game_loop::{
    exit_after_steps, increment_global_turn, run_if_gameplay_active, run_if_hub_runtime_active,
    run_if_campaign_trace_viewer_active, run_if_mission_execution_active,
    run_if_replay_viewer_active, run_if_simulation_steps_exist,
    start_scene_input_system, turn_pacing_input_system,
};
use crate::hub_outcome::{
    campaign_outcome_check_system, campaign_outcome_ui_system,
    draw_runtime_asset_gen_egui_system, hub_quit_requested_system,
    local_intro_sequence_system, region_layer_transition_system,
};
use crate::hub_systems::{
    hub_apply_action_system, hub_menu_input_system,
};
use crate::backstory_cinematic::{
    backstory_cinematic_bootstrap_system, backstory_cinematic_collect_system,
    backstory_cinematic_playback_system, backstory_cinematic_state_reset_system,
    backstory_cinematic_texture_load_system, backstory_narrative_gen_collect_system,
    backstory_narrative_gen_dispatch_system, draw_backstory_cinematic_egui_system,
};
use crate::campaign_ops::{
    hub_continue_campaign_requested_system, hub_new_campaign_requested_system,
};
use crate::fade::{draw_fade_system, screen_transition_system, update_fade_system};
use crate::mission;
use crate::runtime_assets::{
    draw_runtime_menu_background_egui_system, runtime_asset_gen_bootstrap_system,
    runtime_asset_gen_collect_system, runtime_asset_gen_dispatch_system,
    runtime_asset_preview_update_system, update_region_art_system,
};
use crate::scenario_3d::{
    advance_scenario_3d_replay_system, scenario_playback_slider_input_system,
    scenario_replay_keyboard_controls_system, setup_custom_scenario_scene, setup_mission_hud,
    setup_scenario_playback_ui, update_mission_hud_system, update_scenario_hud_system,
    update_scenario_playback_slider_visual_system,
};
use crate::screenshot_capture::{
    ScreenshotCaptureConfig, ScreenshotCaptureState, ScreenshotMode, screenshot_capture_system,
};
// 3D terrain removed — using ASCII egui viewports instead
use crate::ui::quest_log::{draw_quest_log_system, quest_log_toggle_system};
use crate::ui::save_browser::{
    campaign_autosave_system, campaign_save_load_input_system, campaign_save_panel_input_system,
};
use crate::ui::settings::{
    manual_screenshot_capture_system, screenshot_hotkey_input_system, setup_settings_menu,
    settings_menu_slider_input_system, settings_menu_toggle_input_system,
    settings_menu_toggle_system, update_settings_menu_visual_system,
};
use crate::keybinds::keybind_dispatch_system;
use bevy_game::headless_campaign::trace_viewer::{
    advance_campaign_trace_viewer_system, campaign_trace_viewer_keyboard_system,
    campaign_trace_viewer_transition_system,
};
use crate::ui::tutorial::{draw_tutorial_system, tutorial_toggle_system};

pub fn register_scenario_3d_systems(app: &mut App) {
    app.add_systems(
        Update,
        (
            increment_global_turn,
            scenario_replay_keyboard_controls_system,
            advance_scenario_3d_replay_system,
            update_scenario_hud_system,
            scenario_playback_slider_input_system,
            update_scenario_playback_slider_visual_system,
            exit_after_steps.run_if(run_if_simulation_steps_exist),
        )
            .chain(),
    );
}

pub fn register_hub_systems(app: &mut App) {
    app.add_systems(
        Update,
        (
            game_core::overworld_hub_input_system,
            game_core::flashpoint_intent_input_system,
            game_core::sync_roster_lore_with_overworld_system,
            game_core::sync_campaign_parties_with_roster_system,
            game_core::update_faction_war_goals_system,
            game_core::generate_commander_intents_system,
            game_core::refresh_interaction_offers_system,
            game_core::campaign_party_orders_system,
            game_core::interaction_offer_input_system,
            hub_menu_input_system,
            hub_apply_action_system,
            game_core::progress_companion_story_quests_system,
            game_core::generate_companion_story_quests_system,
            events::campaign_event_generation_system,
            exit_after_steps.run_if(run_if_simulation_steps_exist),
        )
            .chain()
            .run_if(run_if_hub_runtime_active),
    );
    app.add_systems(Update, draw_runtime_menu_background_egui_system);
    app.add_systems(Update, draw_backstory_cinematic_egui_system);
    app.add_systems(Update, crate::hub_ui_draw::draw_hub_egui_system);
    app.add_systems(Update, draw_runtime_asset_gen_egui_system);
    app.add_systems(Update, draw_quest_log_system);
    app.add_systems(Update, events::draw_event_notification_system);
    app.init_resource::<MissionPaneState>();
    app.add_systems(Update, draw_ascii_viewport_system);
    app.add_systems(
        Update,
        (
            backstory_cinematic_state_reset_system,
            runtime_asset_gen_bootstrap_system,
            backstory_cinematic_bootstrap_system,
            backstory_narrative_gen_dispatch_system,
            backstory_narrative_gen_collect_system,
            runtime_asset_gen_collect_system,
            backstory_cinematic_collect_system,
            runtime_asset_gen_dispatch_system,
            runtime_asset_preview_update_system,
            backstory_cinematic_texture_load_system,
            backstory_cinematic_playback_system,
            update_region_art_system,
        )
            .chain(),
    );
    app.add_systems(Update, hub_new_campaign_requested_system);
    app.add_systems(Update, hub_continue_campaign_requested_system);
    app.add_systems(Update, region_layer_transition_system);
    app.add_systems(Update, local_intro_sequence_system);
    app.add_systems(Update, hub_quit_requested_system);
    register_mission_execution_systems(app);
}

fn register_mission_execution_systems(app: &mut App) {
    app.add_systems(Update, mission::execution::mission_scene_transition_system);
    // Replay viewer systems
    app.add_systems(Update, mission::execution::replay_viewer_transition_system);
    app.add_systems(
        Update,
        (
            mission::execution::advance_replay_viewer_system,
            mission::execution::replay_viewer_keyboard_system,
        )
            .run_if(run_if_replay_viewer_active),
    );
    // Campaign trace viewer systems
    app.add_systems(Update, campaign_trace_viewer_transition_system);
    app.add_systems(
        Update,
        (
            advance_campaign_trace_viewer_system,
            campaign_trace_viewer_keyboard_system,
        )
            .run_if(run_if_campaign_trace_viewer_active),
    );
    app.add_systems(
        Update,
        mission::sim_bridge::advance_sim_system
            .run_if(run_if_mission_execution_active),
    );
    app.add_systems(
        Update,
        mission::sim_bridge::apply_audio_sfx_from_sim_events_system
            .after(mission::sim_bridge::advance_sim_system)
            .run_if(run_if_mission_execution_active),
    );
    app.add_systems(
        Update,
        (
            mission::sim_bridge::apply_player_orders_system,
            mission::objectives::check_objective_system,
            mission::room_sequence::advance_room_system,
            mission::execution::mission_outcome_ui_system,
            progression::apply_progression_on_unconscious_system,
        )
            .run_if(run_if_mission_execution_active),
    );
}

pub fn register_default_mission_systems(app: &mut App) {
    app.add_systems(
        Update,
        (
            increment_global_turn,
            game_core::attention_management_system,
            game_core::overworld_cooldown_system,
            game_core::campaign_party_orders_system,
            game_core::focus_input_system,
            game_core::flashpoint_intent_input_system,
            game_core::overworld_sync_from_missions_system,
            game_core::overworld_faction_autonomy_system,
            game_core::turn_management_system,
            game_core::auto_increase_stress,
            game_core::activate_mission_system,
            game_core::mission_map_progression_system,
            game_core::player_command_input_system,
            game_core::focused_attention_intervention_system,
            game_core::hero_ability_system,
            game_core::enemy_ai_system,
            game_core::combat_system,
            game_core::complete_objective_system,
            game_core::end_mission_system,
        )
            .chain()
            .run_if(run_if_gameplay_active),
    );
    app.add_systems(
        Update,
        (
            game_core::simulate_unfocused_missions_system,
            game_core::sync_roster_lore_with_overworld_system,
            game_core::sync_campaign_parties_with_roster_system,
            game_core::overworld_ai_border_pressure_system,
            game_core::overworld_intel_update_system,
            game_core::update_faction_war_goals_system,
            game_core::pressure_spawn_missions_system,
            game_core::generate_commander_intents_system,
            game_core::refresh_interaction_offers_system,
            game_core::sync_mission_assignments_system,
            game_core::companion_mission_impact_system,
            game_core::companion_state_drift_system,
            game_core::resolve_mission_consequences_system,
            progression::dispatch_narrative_progression_system,
            progression::collect_narrative_progression_system,
        )
            .chain()
            .run_if(run_if_gameplay_active),
    );
    app.add_systems(
        Update,
        (
            game_core::flashpoint_progression_system,
            game_core::progress_companion_story_quests_system,
            game_core::generate_companion_story_quests_system,
            game_core::companion_recovery_system,
            game_core::print_game_state,
            exit_after_steps.run_if(run_if_simulation_steps_exist),
        )
            .chain()
            .after(progression::collect_narrative_progression_system)
            .run_if(run_if_gameplay_active),
    );
    app.add_systems(Update, update_mission_hud_system);
}

pub fn register_common_systems(app: &mut App) {
    app.add_systems(Update, keybind_dispatch_system);
    app.add_systems(
        Update,
        (
            campaign_save_load_input_system,
            campaign_save_panel_input_system,
            campaign_autosave_system,
            quest_log_toggle_system,
        ),
    );
    app.add_systems(Update, campaign_outcome_check_system);
    app.add_systems(Update, campaign_outcome_ui_system);
    app.add_systems(Update, tutorial_toggle_system);
    app.add_systems(Update, draw_tutorial_system);
    app.add_systems(Update, update_fade_system);
    app.add_systems(Update, draw_fade_system);
    app.add_systems(Update, screen_transition_system);
}

pub fn register_rendered_input_systems(app: &mut App) {
    app.add_systems(
        Update,
        (
            start_scene_input_system,
            turn_pacing_input_system,
            settings_menu_toggle_system,
            settings_menu_slider_input_system,
            settings_menu_toggle_input_system,
            screenshot_hotkey_input_system,
            manual_screenshot_capture_system,
            persist_camera_settings_system,
            update_settings_menu_visual_system,
        )
            .chain(),
    );
}

pub fn register_startup_systems(
    app: &mut App,
    headless: bool,
    has_scenario: bool,
    hub_mode: bool,
) {
    if headless {
        app.add_systems(Startup, game_core::setup_test_scene_headless);
    } else if has_scenario {
        app.add_systems(
            Startup,
            (
                setup_camera,
                setup_mission_hud,
                setup_custom_scenario_scene,
                setup_scenario_playback_ui,
                setup_settings_menu,
            ),
        );
    } else if hub_mode {
        app.add_systems(
            Startup,
            (setup_camera, setup_settings_menu),
        );
    } else {
        app.add_systems(
            Startup,
            (
                setup_camera,
                setup_mission_hud,
                game_core::setup_test_scene,
                setup_settings_menu,
            ),
        );
    }

    if !headless {
        app.add_systems(Startup, audio::load_audio_assets_system);
    }
    app.add_systems(Update, audio::process_audio_events_system);
    app.add_systems(
        Update,
        audio::combat_music_intensity_system.run_if(run_if_mission_execution_active),
    );
}

pub fn register_screenshot_systems(
    app: &mut App,
    screenshot_dir: Option<String>,
    screenshot_sequence_dir: Option<String>,
    screenshot_hub_stages_dir: Option<String>,
    screenshot_every: u32,
    screenshot_warmup_frames: u32,
    simulation_steps: Option<u32>,
) {
    if let Some(dir) = screenshot_dir {
        app.insert_resource(ScreenshotCaptureConfig {
            mode: ScreenshotMode::Single { dir },
            warmup_frames: screenshot_warmup_frames,
            max_captures: Some(1),
            max_attempts: 90,
        });
        app.init_resource::<ScreenshotCaptureState>();
        app.add_systems(Update, screenshot_capture_system);
    } else if let Some(dir) = screenshot_sequence_dir {
        let max_captures = simulation_steps.map(|steps| {
            let every = screenshot_every.max(1);
            steps.div_ceil(every)
        });
        app.insert_resource(ScreenshotCaptureConfig {
            mode: ScreenshotMode::Sequence {
                dir,
                every: screenshot_every,
            },
            warmup_frames: screenshot_warmup_frames,
            max_captures,
            max_attempts: 0,
        });
        app.init_resource::<ScreenshotCaptureState>();
        app.add_systems(Update, screenshot_capture_system);
    } else if let Some(dir) = screenshot_hub_stages_dir {
        app.insert_resource(ScreenshotCaptureConfig {
            mode: ScreenshotMode::HubStages { dir },
            warmup_frames: screenshot_warmup_frames,
            max_captures: None, // capture all stages in HUB_STAGE_CAPTURE_SEQUENCE
            max_attempts: 180,
        });
        app.init_resource::<ScreenshotCaptureState>();
        app.add_systems(Update, screenshot_capture_system);
    }
}
