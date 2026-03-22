#![allow(dead_code)]

use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use std::env;
use std::fs;

mod ai;
mod audio;
mod camera;
mod events;
mod game_core;
mod mission;
mod scenario;
mod ui;

// ---------------------------------------------------------------------------
// Extracted modules
// ---------------------------------------------------------------------------
mod app_systems;
mod backstory_cinematic;
mod campaign_ops;
mod character_select;
mod cli_args;
mod fade;
mod game_loop;
mod hub_outcome;
mod hub_systems;
mod hub_types;
mod local_intro;
mod region_nav;
mod runtime_assets;
mod scenario_3d;
mod screenshot_capture;
mod simulation_cli;
mod terrain;
mod ui_helpers;
mod hub_ui_draw;
mod keybinds;
mod ascii_viewport;
mod combat_view;
mod content;
mod progression;

use camera::{load_camera_settings, CameraFocusTransitionState, SceneViewBounds};
use game_core::RunState;
use game_core::{CampaignSaveData, HubScreen, HubUiState, CharacterCreationState};
use game_core::load_and_prepare_campaign_data;
use ui::save_browser::{
    load_campaign_save_index_state, CampaignAutosaveState,
    CampaignSaveIndexState, CampaignSaveNotice, CampaignSavePanelState,
};
use ui::settings::{ManualScreenshotState, SettingsMenuState};
use ui::quest_log::QuestLogState;
use game_loop::{SimulationSteps, TurnPacingState, StartSceneState, RuntimeModeState};
use hub_types::{
    HubMenuState, StartMenuState, HubActionQueue, CharacterCreationUiState,
    CampaignOutcomeState, HeroDetailUiState,
};
use scenario_3d::{Scenario3dData, ScenarioReplay, ScenarioPlaybackSpeed};
use region_nav::{RegionLayerTransitionState, RegionTargetPickerState};
use runtime_assets::{RuntimeAssetGenState, RuntimeAssetPreviewState, RegionArtState};
use backstory_cinematic::{BackstoryNarrativeGenState, BackstoryCinematicState};
use campaign_ops::spawn_mission_entities_from_snapshots;
use local_intro::LocalEagleEyeIntroState;
use fade::FadeState;
use ui::tutorial::TutorialState;
use simulation_cli::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    let Some(cli) = cli_args::parse_cli_args(&args) else {
        return;
    };
    let mut app = App::new();

    let headless_mode = cli.headless_mode;
    let simulation_steps = cli.simulation_steps;
    let map_seed = cli.map_seed;
    let campaign_load_path = cli.campaign_load_path;
    let mut run_hub_flag = cli.run_hub_flag;
    let run_dev_mode = cli.run_dev_mode;
    let scenario_3d_path = cli.scenario_3d_path;
    let screenshot_dir = cli.screenshot_dir;
    let screenshot_sequence_dir = cli.screenshot_sequence_dir;
    let screenshot_hub_stages_dir = cli.screenshot_hub_stages_dir;
    let screenshot_every = cli.screenshot_every;
    let screenshot_warmup_frames = cli.screenshot_warmup_frames;
    let horde_3d_flag = cli.horde_3d_flag;
    let horde_3d_hero_win_flag = cli.horde_3d_hero_win_flag;

    if cli.run_phase0_sim { run_phase0_simulation(); return; }
    if cli.run_phase1_sim { run_phase1_simulation(); return; }
    if cli.run_phase2_sim { run_phase2_simulation(); return; }
    if cli.run_phase3_sim { run_phase3_simulation(); return; }
    if cli.run_phase4_sim { run_phase4_simulation(); return; }
    if cli.run_phase5_sim { run_phase5_simulation(); return; }
    if cli.run_phase6_report_flag { run_phase6_report(); return; }
    if cli.run_phase6_viz_flag { run_phase6_visualization(); return; }
    if cli.run_pathing_viz_flag { run_pathing_visualization(); return; }
    if cli.run_pathing_hero_win_viz_flag { run_pathing_hero_win_visualization(); return; }
    if cli.run_phase7_sim { run_phase7_simulation(); return; }
    if cli.run_phase8_sim { run_phase8_simulation(); return; }
    if cli.run_phase9_sim { run_phase9_simulation(); return; }
    if cli.run_pathing_sim { run_pathing_simulation(); return; }
    if cli.run_pathing_hero_win_sim { run_pathing_hero_win_simulation(); return; }
    if cli.run_pathing_hero_hp_ablation { run_pathing_hero_hp_ablation_simulation(); return; }
    if cli.run_viz_index_flag { run_visualization_index(); return; }
    if let Some(path) = cli.scenario_template_path {
        run_write_scenario_template(&path);
        return;
    }
    if let Some(path) = cli.scenario_viz_path {
        let out = cli.scenario_viz_out_path
            .unwrap_or_else(|| "generated/reports/ai_custom_scenario.html".to_string());
        run_custom_scenario_visualization(&path, &out);
        return;
    }

    let horde_flags = (cli.horde_3d_flag as u8) + (cli.horde_3d_hero_win_flag as u8);
    if !headless_mode && !run_hub_flag && scenario_3d_path.is_none() && horde_flags == 0 {
        run_hub_flag = true;
    }
    if horde_flags > 1 {
        eprintln!("Error: use only one of --horde-3d or --horde-3d-hero-win.");
        return;
    }
    if scenario_3d_path.is_some() && horde_flags > 0 {
        eprintln!("Error: choose either --scenario-3d <json> or a --horde-3d mode.");
        return;
    }
    if run_hub_flag && (scenario_3d_path.is_some() || horde_flags > 0) {
        eprintln!("Error: --hub cannot be combined with scenario/horde 3D flags.");
        return;
    }
    let screenshot_mode_count = screenshot_dir.is_some() as u8
        + screenshot_sequence_dir.is_some() as u8
        + screenshot_hub_stages_dir.is_some() as u8;
    if screenshot_mode_count > 1 {
        eprintln!(
            "Error: use only one of --screenshot, --screenshot-sequence, or --screenshot-hub-stages."
        );
        return;
    }

    let scenario_3d_bundle = if let Some(path) = &scenario_3d_path {
        let text = match fs::read_to_string(path) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("Failed to read --scenario-3d file '{}': {}", path, err);
                return;
            }
        };
        let parsed: ai::tooling::CustomScenario = match serde_json::from_str(&text) {
            Ok(value) => value,
            Err(err) => {
                eprintln!(
                    "Invalid scenario json for --scenario-3d '{}': {}",
                    path, err
                );
                return;
            }
        };
        let frames =
            ai::tooling::build_custom_scenario_state_frames(&parsed, ai::core::FIXED_TICK_MS);
        Some((parsed, frames))
    } else if horde_3d_flag {
        Some(scenario_3d::build_horde_3d_bundle(false))
    } else if horde_3d_hero_win_flag {
        Some(scenario_3d::build_horde_3d_bundle(true))
    } else {
        None
    };
    let default_mission_mode = !headless_mode && scenario_3d_bundle.is_none() && !run_hub_flag;
    let start_scene_active = default_mission_mode && simulation_steps.is_none();

    if headless_mode {
        if scenario_3d_bundle.is_some() {
            eprintln!("Error: 3D scenario modes require rendering; remove --headless.");
            return;
        }
        if run_hub_flag {
            eprintln!("Error: --hub requires rendering; remove --headless.");
            return;
        }
        if screenshot_dir.is_some()
            || screenshot_sequence_dir.is_some()
            || screenshot_hub_stages_dir.is_some()
        {
            eprintln!("Error: screenshot capture requires rendering; remove --headless.");
            return;
        }
        app.add_plugins((MinimalPlugins,));
        if simulation_steps.is_none() {
            eprintln!("Error: --headless mode requires --steps <N>.");
            return;
        }
    } else {
        app.add_plugins(DefaultPlugins);
        app.add_plugins(EguiPlugin);
    }

    // Insert all resources
    app.insert_resource(SimulationSteps(simulation_steps))
        .insert_resource(SceneViewBounds::default())
        .insert_resource(load_camera_settings())
        .insert_resource(RuntimeModeState {
            hub_mode: run_hub_flag,
            dev_mode: run_dev_mode,
        })
        .insert_resource(StartSceneState {
            active: start_scene_active,
        })
        .init_resource::<TurnPacingState>()
        .init_resource::<CampaignSaveNotice>()
        .insert_resource(load_campaign_save_index_state())
        .init_resource::<CampaignSavePanelState>()
        .init_resource::<CampaignAutosaveState>()
        .init_resource::<QuestLogState>()
        .init_resource::<SettingsMenuState>()
        .init_resource::<ManualScreenshotState>()
        .init_resource::<StartMenuState>()
        .init_resource::<HubMenuState>()
        .init_resource::<HubUiState>()
        .init_resource::<HeroDetailUiState>()
        .init_resource::<CharacterCreationState>()
        .init_resource::<CharacterCreationUiState>()
        .init_resource::<RuntimeAssetGenState>()
        .init_resource::<RuntimeAssetPreviewState>()
        .init_resource::<RegionArtState>()
        .init_resource::<BackstoryNarrativeGenState>()
        .init_resource::<BackstoryCinematicState>()
        .init_resource::<RegionLayerTransitionState>()
        .init_resource::<LocalEagleEyeIntroState>()
        .init_resource::<RegionTargetPickerState>()
        .init_resource::<CameraFocusTransitionState>()
        .init_resource::<HubActionQueue>()
        .init_resource::<RunState>()
        .init_resource::<game_core::MissionMap>()
        .init_resource::<game_core::MissionBoard>()
        .init_resource::<game_core::AttentionState>()
        .init_resource::<game_core::CommanderState>()
        .init_resource::<game_core::DiplomacyState>()
        .init_resource::<game_core::InteractionBoard>()
        .init_resource::<game_core::CampaignRoster>()
        .init_resource::<game_core::CampaignParties>()
        .init_resource::<game_core::CampaignLedger>()
        .init_resource::<game_core::CampaignEventLog>()
        .init_resource::<game_core::CompanionStoryState>()
        .init_resource::<game_core::FlashpointState>()
        .init_resource::<events::CampaignEventQueue>()
        .init_resource::<CampaignOutcomeState>()
        .init_resource::<mission::vfx::VfxEventQueue>()
        .init_resource::<mission::sim_bridge::SimEventBuffer>()
        .init_resource::<mission::sim_bridge::MissionEventLog>()
        .init_resource::<progression::NarrativeProgressionState>()
        .init_resource::<mission::objectives::MissionObjectiveState>()
        .init_resource::<audio::AudioSettings>()
        .init_resource::<audio::AudioHandles>()
        .init_resource::<audio::AudioEventQueue>()
        .init_resource::<TutorialState>()
        .init_resource::<FadeState>()
        .init_resource::<fade::ScreenTransition>()
        .init_resource::<hub_types::ScreenHistory>()
        .init_resource::<keybinds::KeybindConfig>()
        .init_resource::<keybinds::ActionEvents>()
        .init_resource::<content::ContentRegistry>();

    if let Some(seed) = map_seed {
        app.insert_resource(game_core::OverworldMap::from_seed(seed));
    } else {
        app.init_resource::<game_core::OverworldMap>();
    }

    if let Some(path) = campaign_load_path {
        match load_and_prepare_campaign_data(&path) {
            Ok(loaded) => {
                app.insert_resource(loaded.run_state)
                    .insert_resource(loaded.mission_map)
                    .insert_resource(loaded.attention_state)
                    .insert_resource(loaded.overworld_map)
                    .insert_resource(loaded.commander_state)
                    .insert_resource(loaded.diplomacy_state)
                    .insert_resource(loaded.interaction_board)
                    .insert_resource(loaded.campaign_roster)
                    .insert_resource(loaded.campaign_parties)
                    .insert_resource(loaded.campaign_ledger)
                    .insert_resource(loaded.campaign_event_log)
                    .insert_resource(loaded.companion_story_state);
                app.insert_resource(loaded.character_creation);
                spawn_mission_entities_from_snapshots(
                    &mut app.world,
                    loaded.mission_snapshots,
                    loaded.active_mission_id,
                );
                app.insert_resource(CampaignSaveNotice {
                    message: format!(
                        "Loaded campaign at startup (v{}) <- {}",
                        loaded.save_version, path
                    ),
                });
            }
            Err(err) => {
                eprintln!("Failed to load campaign '{}': {}", path, err);
            }
        }
    }

    if !headless_mode
        && app
            .world
            .resource::<game_core::MissionBoard>()
            .entities
            .is_empty()
    {
        spawn_mission_entities_from_snapshots(&mut app.world, Vec::new(), None);
    }

    // Register mode-specific systems
    if let Some((scenario, frames)) = &scenario_3d_bundle {
        app.insert_resource(SceneViewBounds {
            min_x: scenario.world_min_x,
            max_x: scenario.world_max_x,
            min_z: scenario.world_min_y,
            max_z: scenario.world_max_y,
        })
        .insert_resource(Scenario3dData(scenario.clone()))
        .insert_resource(ScenarioReplay {
            name: scenario.name.clone(),
            events_per_frame: vec![Vec::new(); frames.len()],
            frames: frames.clone(),
            frame_index: 0,
            tick_seconds: ai::core::FIXED_TICK_MS as f32 / 1000.0,
            tick_accumulator: 0.0,
            paused: false,
        })
        .insert_resource(ScenarioPlaybackSpeed {
            value: 1.0,
            min: 0.25,
            max: 3.0,
        });
        app_systems::register_scenario_3d_systems(&mut app);
    } else if run_hub_flag {
        app_systems::register_hub_systems(&mut app);
    } else {
        app_systems::register_default_mission_systems(&mut app);
    }

    app_systems::register_common_systems(&mut app);
    if !headless_mode {
        app_systems::register_rendered_input_systems(&mut app);
    }
    app_systems::register_startup_systems(
        &mut app,
        headless_mode,
        scenario_3d_bundle.is_some(),
        run_hub_flag,
    );
    app_systems::register_screenshot_systems(
        &mut app,
        screenshot_dir,
        screenshot_sequence_dir,
        screenshot_hub_stages_dir,
        screenshot_every,
        screenshot_warmup_frames,
        simulation_steps,
    );

    app.run();
}


#[cfg(test)]
mod tests;
