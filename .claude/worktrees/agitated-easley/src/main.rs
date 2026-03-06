use bevy::app::AppExit;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;

mod ai;
mod game_core;
use game_core::{
    ActiveMission, AssignedHero, MissionData, MissionProgress, MissionResult, MissionTactics,
    RunState,
};

#[derive(Resource)]
struct SimulationSteps(Option<u32>);

#[derive(Component)]
struct MissionHudText;

#[derive(Resource, Clone)]
struct Scenario3dData(ai::tooling::CustomScenario);

#[derive(Resource, Clone, Copy)]
struct SceneViewBounds {
    min_x: f32,
    max_x: f32,
    min_z: f32,
    max_z: f32,
}

impl Default for SceneViewBounds {
    fn default() -> Self {
        Self {
            min_x: -6.0,
            max_x: 6.0,
            min_z: -4.0,
            max_z: 4.0,
        }
    }
}

#[derive(Resource)]
struct ScenarioReplay {
    name: String,
    frames: Vec<ai::core::SimState>,
    frame_index: usize,
    tick_seconds: f32,
    tick_accumulator: f32,
    paused: bool,
}

#[derive(Component)]
struct ScenarioUnitVisual {
    unit_id: u32,
}

#[derive(Resource)]
struct ScenarioPlaybackSpeed {
    value: f32,
    min: f32,
    max: f32,
}

#[derive(Component)]
struct PlaybackSliderTrack;

#[derive(Component)]
struct PlaybackSliderFill;

#[derive(Component)]
struct PlaybackSliderLabel;

#[derive(Component)]
struct OrbitCameraController {
    focus: Vec3,
    radius: f32,
    min_radius: f32,
    max_radius: f32,
    yaw: f32,
    pitch: f32,
}

#[derive(Component)]
struct HubHudText;

#[derive(Resource)]
struct HubMenuState {
    selected: usize,
    notice: String,
}

impl Default for HubMenuState {
    fn default() -> Self {
        Self {
            selected: 0,
            notice: "Welcome back, Commander. The guild is ready.".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HubAction {
    AssembleExpedition,
    ReviewRecruits,
    IntelSweep,
    DispatchRelief,
    LeaveGuild,
}

impl HubAction {
    fn from_selected(selected: usize) -> Self {
        match selected {
            0 => Self::AssembleExpedition,
            1 => Self::ReviewRecruits,
            2 => Self::IntelSweep,
            3 => Self::DispatchRelief,
            _ => Self::LeaveGuild,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::AssembleExpedition => "Assemble Expedition",
            Self::ReviewRecruits => "Review Recruits",
            Self::IntelSweep => "Intel Sweep",
            Self::DispatchRelief => "Dispatch Relief",
            Self::LeaveGuild => "Leave Guild",
        }
    }
}

#[derive(Resource, Default)]
struct HubActionQueue {
    pending: Option<HubAction>,
    actions_taken: u32,
}

const CAMERA_SETTINGS_PATH: &str = "generated/settings/camera_settings.json";
const CAMPAIGN_SAVE_PATH: &str = "generated/saves/campaign_save.json";
const CAMPAIGN_SAVE_SLOT_2_PATH: &str = "generated/saves/campaign_slot_2.json";
const CAMPAIGN_SAVE_SLOT_3_PATH: &str = "generated/saves/campaign_slot_3.json";
const CAMPAIGN_AUTOSAVE_PATH: &str = "generated/saves/campaign_autosave.json";
const CAMPAIGN_SAVE_INDEX_PATH: &str = "generated/saves/campaign_index.json";
const SAVE_VERSION_V1: u32 = 1;
const CURRENT_SAVE_VERSION: u32 = 2;

fn default_save_version() -> u32 {
    SAVE_VERSION_V1
}

#[derive(Resource)]
struct CampaignSaveNotice {
    message: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct SaveSlotMetadata {
    slot: String,
    path: String,
    save_version: u32,
    compatible: bool,
    global_turn: u32,
    map_seed: u64,
    saved_unix_seconds: u64,
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct CampaignSaveIndex {
    slots: Vec<SaveSlotMetadata>,
    autosave: Option<SaveSlotMetadata>,
}

#[derive(Resource, Clone, Default)]
struct CampaignSaveIndexState {
    index: CampaignSaveIndex,
}

#[derive(Resource, Clone)]
struct CampaignSavePanelState {
    open: bool,
    selected: usize,
    pending_load_path: Option<String>,
    pending_label: Option<String>,
    preview: String,
}

impl Default for CampaignSavePanelState {
    fn default() -> Self {
        Self {
            open: false,
            selected: 0,
            pending_load_path: None,
            pending_label: None,
            preview: String::new(),
        }
    }
}

impl Default for CampaignSaveNotice {
    fn default() -> Self {
        Self {
            message: "Save: none (F5/F9 slot1, Shift slot2, Ctrl slot3)".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct CampaignSaveData {
    #[serde(default = "default_save_version")]
    save_version: u32,
    run_state: game_core::RunState,
    #[serde(default)]
    mission_map: game_core::MissionMap,
    attention_state: game_core::AttentionState,
    overworld_map: game_core::OverworldMap,
    commander_state: game_core::CommanderState,
    diplomacy_state: game_core::DiplomacyState,
    interaction_board: game_core::InteractionBoard,
    campaign_roster: game_core::CampaignRoster,
    campaign_ledger: game_core::CampaignLedger,
    #[serde(default)]
    campaign_event_log: game_core::CampaignEventLog,
    #[serde(default)]
    companion_story_state: game_core::CompanionStoryState,
    #[serde(default)]
    flashpoint_state: game_core::FlashpointState,
    /// Serialized mission entities (replaces the old MissionBoard resource data).
    #[serde(default)]
    mission_snapshots: Vec<game_core::MissionSnapshot>,
    /// The `id` field of the mission entity that has `ActiveMission`.
    #[serde(default)]
    active_mission_id: Option<u32>,
}

#[derive(Resource, Clone)]
struct CampaignAutosaveState {
    enabled: bool,
    interval_turns: u32,
    last_autosave_turn: u32,
}

impl Default for CampaignAutosaveState {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_turns: 10,
            last_autosave_turn: 0,
        }
    }
}

#[derive(Resource, Serialize, Deserialize, Clone, Copy)]
struct CameraSettings {
    orbit_sensitivity: f32,
    zoom_sensitivity: f32,
    invert_orbit_y: bool,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            orbit_sensitivity: 1.0,
            zoom_sensitivity: 1.0,
            invert_orbit_y: false,
        }
    }
}

#[derive(Resource)]
struct SettingsMenuState {
    is_open: bool,
}

impl Default for SettingsMenuState {
    fn default() -> Self {
        Self { is_open: false }
    }
}

#[derive(Component)]
struct SettingsMenuRoot;

#[derive(Component)]
struct OrbitSensitivitySliderTrack;

#[derive(Component)]
struct OrbitSensitivitySliderFill;

#[derive(Component)]
struct OrbitSensitivityLabel;

#[derive(Component)]
struct ZoomSensitivitySliderTrack;

#[derive(Component)]
struct ZoomSensitivitySliderFill;

#[derive(Component)]
struct ZoomSensitivityLabel;

#[derive(Component)]
struct InvertOrbitYButton;

#[derive(Component)]
struct InvertOrbitYLabel;

#[derive(Component)]
struct ResetSettingsButton;

fn parse_seed_arg(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u64>().ok()
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut app = App::new();

    let mut headless_mode = false;
    let mut simulation_steps: Option<u32> = None;
    let mut run_phase0_sim = false;
    let mut run_phase1_sim = false;
    let mut run_phase2_sim = false;
    let mut run_phase3_sim = false;
    let mut run_phase4_sim = false;
    let mut run_phase5_sim = false;
    let mut run_phase6_report_flag = false;
    let mut run_phase6_viz_flag = false;
    let mut run_pathing_viz_flag = false;
    let mut run_pathing_hero_win_viz_flag = false;
    let mut run_phase7_sim = false;
    let mut run_phase8_sim = false;
    let mut run_phase9_sim = false;
    let mut run_pathing_sim = false;
    let mut run_pathing_hero_win_sim = false;
    let mut run_pathing_hero_hp_ablation = false;
    let mut run_viz_index_flag = false;
    let mut scenario_template_path: Option<String> = None;
    let mut scenario_viz_path: Option<String> = None;
    let mut scenario_viz_out_path: Option<String> = None;
    let mut scenario_3d_path: Option<String> = None;
    let mut map_seed: Option<u64> = None;
    let mut campaign_load_path: Option<String> = None;
    let mut horde_3d_flag = false;
    let mut horde_3d_hero_win_flag = false;
    let mut run_hub_flag = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--headless" => {
                headless_mode = true;
                i += 1;
            }
            "--steps" => {
                if let Some(steps_str) = args.get(i + 1) {
                    if let Ok(steps) = steps_str.parse::<u32>() {
                        simulation_steps = Some(steps);
                        i += 2;
                    } else {
                        eprintln!("Error: --steps requires a valid number.");
                        return;
                    }
                } else {
                    eprintln!("Error: --steps requires a value.");
                    return;
                }
            }
            "--phase0-sim" => {
                run_phase0_sim = true;
                i += 1;
            }
            "--phase1-sim" => {
                run_phase1_sim = true;
                i += 1;
            }
            "--phase2-sim" => {
                run_phase2_sim = true;
                i += 1;
            }
            "--phase3-sim" => {
                run_phase3_sim = true;
                i += 1;
            }
            "--phase4-sim" => {
                run_phase4_sim = true;
                i += 1;
            }
            "--phase5-sim" => {
                run_phase5_sim = true;
                i += 1;
            }
            "--phase6-report" => {
                run_phase6_report_flag = true;
                i += 1;
            }
            "--phase6-viz" => {
                run_phase6_viz_flag = true;
                i += 1;
            }
            "--pathing-viz" => {
                run_pathing_viz_flag = true;
                i += 1;
            }
            "--pathing-viz-hero-win" => {
                run_pathing_hero_win_viz_flag = true;
                i += 1;
            }
            "--phase7-sim" => {
                run_phase7_sim = true;
                i += 1;
            }
            "--phase8-sim" => {
                run_phase8_sim = true;
                i += 1;
            }
            "--phase9-sim" => {
                run_phase9_sim = true;
                i += 1;
            }
            "--pathing-sim" => {
                run_pathing_sim = true;
                i += 1;
            }
            "--pathing-sim-hero-win" => {
                run_pathing_hero_win_sim = true;
                i += 1;
            }
            "--pathing-sim-hero-hp-ablation" => {
                run_pathing_hero_hp_ablation = true;
                i += 1;
            }
            "--viz-index" => {
                run_viz_index_flag = true;
                i += 1;
            }
            "--scenario-template" => {
                if let Some(path) = args.get(i + 1) {
                    scenario_template_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --scenario-template requires a file path.");
                    return;
                }
            }
            "--scenario-viz" => {
                if let Some(path) = args.get(i + 1) {
                    scenario_viz_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --scenario-viz requires a scenario json path.");
                    return;
                }
            }
            "--scenario-out" => {
                if let Some(path) = args.get(i + 1) {
                    scenario_viz_out_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --scenario-out requires an output html path.");
                    return;
                }
            }
            "--scenario-3d" => {
                if let Some(path) = args.get(i + 1) {
                    scenario_3d_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --scenario-3d requires a scenario json path.");
                    return;
                }
            }
            "--load-campaign" => {
                if let Some(path) = args.get(i + 1) {
                    campaign_load_path = Some(path.clone());
                    i += 2;
                } else {
                    eprintln!("Error: --load-campaign requires a file path.");
                    return;
                }
            }
            "--map-seed" => {
                if let Some(seed_str) = args.get(i + 1) {
                    if let Some(seed) = parse_seed_arg(seed_str) {
                        map_seed = Some(seed);
                        i += 2;
                    } else {
                        eprintln!(
                            "Error: --map-seed requires a valid u64 (decimal or 0x-prefixed hex)."
                        );
                        return;
                    }
                } else {
                    eprintln!("Error: --map-seed requires a value.");
                    return;
                }
            }
            "--horde-3d" => {
                horde_3d_flag = true;
                i += 1;
            }
            "--horde-3d-hero-win" => {
                horde_3d_hero_win_flag = true;
                i += 1;
            }
            "--hub" => {
                run_hub_flag = true;
                i += 1;
            }
            _ => {
                eprintln!("Warning: Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    if run_phase0_sim {
        run_phase0_simulation();
        return;
    }
    if run_phase1_sim {
        run_phase1_simulation();
        return;
    }
    if run_phase2_sim {
        run_phase2_simulation();
        return;
    }
    if run_phase3_sim {
        run_phase3_simulation();
        return;
    }
    if run_phase4_sim {
        run_phase4_simulation();
        return;
    }
    if run_phase5_sim {
        run_phase5_simulation();
        return;
    }
    if run_phase6_report_flag {
        run_phase6_report();
        return;
    }
    if run_phase6_viz_flag {
        run_phase6_visualization();
        return;
    }
    if run_pathing_viz_flag {
        run_pathing_visualization();
        return;
    }
    if run_pathing_hero_win_viz_flag {
        run_pathing_hero_win_visualization();
        return;
    }
    if run_phase7_sim {
        run_phase7_simulation();
        return;
    }
    if run_phase8_sim {
        run_phase8_simulation();
        return;
    }
    if run_phase9_sim {
        run_phase9_simulation();
        return;
    }
    if run_pathing_sim {
        run_pathing_simulation();
        return;
    }
    if run_pathing_hero_win_sim {
        run_pathing_hero_win_simulation();
        return;
    }
    if run_pathing_hero_hp_ablation {
        run_pathing_hero_hp_ablation_simulation();
        return;
    }
    if run_viz_index_flag {
        run_visualization_index();
        return;
    }
    if let Some(path) = scenario_template_path {
        run_write_scenario_template(&path);
        return;
    }
    if let Some(path) = scenario_viz_path {
        let out = scenario_viz_out_path
            .unwrap_or_else(|| "generated/reports/ai_custom_scenario.html".to_string());
        run_custom_scenario_visualization(&path, &out);
        return;
    }

    let horde_flags = (horde_3d_flag as u8) + (horde_3d_hero_win_flag as u8);
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
        Some(build_horde_3d_bundle(false))
    } else if horde_3d_hero_win_flag {
        Some(build_horde_3d_bundle(true))
    } else {
        None
    };

    if headless_mode {
        if scenario_3d_bundle.is_some() {
            eprintln!("Error: 3D scenario modes require rendering; remove --headless.");
            return;
        }
        if run_hub_flag {
            eprintln!("Error: --hub requires rendering; remove --headless.");
            return;
        }
        app.add_plugins((MinimalPlugins,));
        if simulation_steps.is_none() {
            eprintln!("Error: --headless mode requires --steps <N>.");
            return;
        }
    } else {
        app.add_plugins(DefaultPlugins);
    }

    app.insert_resource(SimulationSteps(simulation_steps))
        .insert_resource(SceneViewBounds::default())
        .insert_resource(load_camera_settings())
        .init_resource::<CampaignSaveNotice>()
        .insert_resource(load_campaign_save_index_state())
        .init_resource::<CampaignSavePanelState>()
        .init_resource::<CampaignAutosaveState>()
        .init_resource::<SettingsMenuState>()
        .init_resource::<HubMenuState>()
        .init_resource::<HubActionQueue>()
        .init_resource::<RunState>()
        .init_resource::<game_core::MissionMap>()
        .init_resource::<game_core::MissionBoard>()
        .init_resource::<game_core::AttentionState>()
        .init_resource::<game_core::CommanderState>()
        .init_resource::<game_core::DiplomacyState>()
        .init_resource::<game_core::InteractionBoard>()
        .init_resource::<game_core::CampaignRoster>()
        .init_resource::<game_core::CampaignLedger>()
        .init_resource::<game_core::CampaignEventLog>()
        .init_resource::<game_core::CompanionStoryState>()
        .init_resource::<game_core::FlashpointState>();

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
                    .insert_resource(loaded.campaign_ledger)
                    .insert_resource(loaded.campaign_event_log)
                    .insert_resource(loaded.companion_story_state);
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

    // Spawn mission entities if not already spawned by a load path
    if !headless_mode && app.world.resource::<game_core::MissionBoard>().entities.is_empty() {
        spawn_mission_entities_from_snapshots(&mut app.world, Vec::new(), None);
    }

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
        })
        .add_systems(
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
    } else if run_hub_flag {
        app.add_systems(
            Update,
            (
                game_core::overworld_hub_input_system,
                game_core::flashpoint_intent_input_system,
                game_core::sync_roster_lore_with_overworld_system,
                game_core::update_faction_war_goals_system,
                game_core::generate_commander_intents_system,
                game_core::refresh_interaction_offers_system,
                game_core::interaction_offer_input_system,
                hub_menu_input_system,
                hub_apply_action_system,
                game_core::progress_companion_story_quests_system,
                game_core::generate_companion_story_quests_system,
                update_hub_hud_system,
                exit_after_steps.run_if(run_if_simulation_steps_exist),
            )
                .chain(),
        );
    } else {
        app.add_systems(
            Update,
            (
                increment_global_turn,
                game_core::attention_management_system,
                game_core::overworld_cooldown_system,
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
                .chain(),
        );
        app.add_systems(
            Update,
            (
                game_core::simulate_unfocused_missions_system,
                game_core::sync_roster_lore_with_overworld_system,
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
                game_core::flashpoint_progression_system,
                game_core::progress_companion_story_quests_system,
                game_core::generate_companion_story_quests_system,
                game_core::companion_recovery_system,
                game_core::print_game_state,
                update_mission_hud_system,
                exit_after_steps.run_if(run_if_simulation_steps_exist),
            )
                .chain(),
        );
    }

    app.add_systems(
        Update,
        (
            campaign_save_load_input_system,
            campaign_save_panel_input_system,
            campaign_autosave_system,
        ),
    );

    if !headless_mode {
        app.add_systems(
            Update,
            (
                settings_menu_toggle_system,
                settings_menu_slider_input_system,
                settings_menu_toggle_input_system,
                persist_camera_settings_system,
                update_settings_menu_visual_system,
                orbit_camera_controller_system,
            )
                .chain(),
        );
    }

    if headless_mode {
        app.add_systems(Startup, game_core::setup_test_scene_headless);
    } else {
        if scenario_3d_bundle.is_some() {
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
        } else {
            if run_hub_flag {
                app.add_systems(
                    Startup,
                    (
                        setup_camera,
                        setup_adventurers_guild_scene,
                        setup_hub_hud,
                        setup_settings_menu,
                    ),
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
        }
    }

    app.run();
}

fn setup_camera(mut commands: Commands, bounds: Res<SceneViewBounds>) {
    let center_x = (bounds.min_x + bounds.max_x) * 0.5;
    let center_z = (bounds.min_z + bounds.max_z) * 0.5;
    let span_x = (bounds.max_x - bounds.min_x).max(8.0);
    let span_z = (bounds.max_z - bounds.min_z).max(8.0);
    let span = span_x.max(span_z);
    let focus = Vec3::new(center_x, 0.0, center_z);
    let start = Vec3::new(center_x, span * 0.95, center_z + span * 1.45);
    let offset = start - focus;
    let radius = offset.length().max(0.001);
    let yaw = offset.x.atan2(offset.z);
    let pitch = (offset.y / radius).asin();

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(start).looking_at(focus, Vec3::Y),
            ..default()
        },
        OrbitCameraController {
            focus,
            radius,
            min_radius: 3.0,
            max_radius: span * 6.0,
            yaw,
            pitch,
        },
    ));

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 15000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.1, -0.8, 0.0)),
        ..default()
    });
}

fn orbit_camera_controller_system(
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut mouse_wheel_events: EventReader<MouseWheel>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    bounds: Res<SceneViewBounds>,
    camera_settings: Res<CameraSettings>,
    settings_menu: Res<SettingsMenuState>,
    mut query: Query<(&mut OrbitCameraController, &mut Transform)>,
) {
    const ORBIT_YAW_SENSITIVITY: f32 = 0.0019;
    const ORBIT_PITCH_SENSITIVITY: f32 = 0.0015;
    const ZOOM_SENSITIVITY: f32 = 0.08;
    const MIN_PITCH_RADIANS: f32 = 0.22;
    const MAX_PITCH_RADIANS: f32 = 1.18;
    const MIN_CAMERA_HEIGHT: f32 = 1.1;

    let mut mouse_delta = Vec2::ZERO;
    for event in mouse_motion_events.read() {
        mouse_delta += event.delta;
    }

    let mut scroll_delta = 0.0_f32;
    for event in mouse_wheel_events.read() {
        scroll_delta += event.y;
    }

    let Some(keyboard) = keyboard else {
        return;
    };
    if settings_menu.is_open {
        return;
    }

    for (mut controller, mut transform) in &mut query {
        let mut changed = false;
        let orbit_sens = camera_settings.orbit_sensitivity.clamp(0.2, 2.5);
        let zoom_sens = camera_settings.zoom_sensitivity.clamp(0.2, 2.5);
        let y_invert = if camera_settings.invert_orbit_y {
            1.0
        } else {
            -1.0
        };

        if scroll_delta.abs() > f32::EPSILON {
            controller.radius -= scroll_delta * ZOOM_SENSITIVITY * zoom_sens * controller.radius;
            changed = true;
        }

        if mouse_buttons.pressed(MouseButton::Right) && mouse_delta.length_squared() > 0.0 {
            controller.yaw -= mouse_delta.x * ORBIT_YAW_SENSITIVITY * orbit_sens;
            controller.pitch += y_invert * mouse_delta.y * ORBIT_PITCH_SENSITIVITY * orbit_sens;
            changed = true;
        }

        if mouse_buttons.pressed(MouseButton::Middle) && mouse_delta.length_squared() > 0.0 {
            let right = transform.rotation * Vec3::X;
            let forward = transform.rotation * Vec3::NEG_Z;
            let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
            let pan_scale = 0.006 * controller.radius;
            controller.focus += (-mouse_delta.x * pan_scale) * right;
            controller.focus += (mouse_delta.y * pan_scale) * forward_flat;
            changed = true;
        }

        let mut keyboard_pan = Vec3::ZERO;
        let right = transform.rotation * Vec3::X;
        let forward = transform.rotation * Vec3::NEG_Z;
        let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        if keyboard.pressed(KeyCode::KeyA) {
            keyboard_pan -= right;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            keyboard_pan += right;
        }
        if keyboard.pressed(KeyCode::KeyW) {
            keyboard_pan += forward_flat;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            keyboard_pan -= forward_flat;
        }
        if keyboard_pan.length_squared() > 0.0 {
            let radius = controller.radius;
            controller.focus += keyboard_pan.normalize() * (0.045 * radius);
            changed = true;
        }

        if keyboard.just_pressed(KeyCode::KeyF) {
            controller.focus = Vec3::new(
                (bounds.min_x + bounds.max_x) * 0.5,
                0.0,
                (bounds.min_z + bounds.max_z) * 0.5,
            );
            controller.radius =
                ((bounds.max_x - bounds.min_x).max(bounds.max_z - bounds.min_z) * 1.4).max(6.0);
            controller.pitch = 0.62;
            changed = true;
        }

        controller.pitch = controller.pitch.clamp(MIN_PITCH_RADIANS, MAX_PITCH_RADIANS);
        controller.radius = controller.radius.clamp(
            controller.min_radius,
            controller.max_radius.max(controller.min_radius + 1.0),
        );
        if controller.yaw > std::f32::consts::PI || controller.yaw < -std::f32::consts::PI {
            controller.yaw = controller.yaw.rem_euclid(std::f32::consts::TAU);
            if controller.yaw > std::f32::consts::PI {
                controller.yaw -= std::f32::consts::TAU;
            }
        }
        controller.focus.x = controller
            .focus
            .x
            .clamp(bounds.min_x - 30.0, bounds.max_x + 30.0);
        controller.focus.z = controller
            .focus
            .z
            .clamp(bounds.min_z - 30.0, bounds.max_z + 30.0);

        if !changed {
            continue;
        }

        let cos_pitch = controller.pitch.cos();
        let offset = Vec3::new(
            controller.radius * cos_pitch * controller.yaw.sin(),
            controller.radius * controller.pitch.sin(),
            controller.radius * cos_pitch * controller.yaw.cos(),
        );
        transform.translation = controller.focus + offset;
        if transform.translation.y < MIN_CAMERA_HEIGHT {
            transform.translation.y = MIN_CAMERA_HEIGHT;
        }
        transform.look_at(controller.focus, Vec3::Y);
    }
}

fn setup_settings_menu(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");
    commands
        .spawn((
            SettingsMenuRoot,
            NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    right: Val::Px(20.0),
                    top: Val::Px(20.0),
                    width: Val::Px(320.0),
                    height: Val::Auto,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    padding: UiRect::all(Val::Px(12.0)),
                    row_gap: Val::Px(8.0),
                    display: Display::None,
                    ..default()
                },
                background_color: Color::rgba(0.05, 0.06, 0.08, 0.90).into(),
                ..default()
            },
        ))
        .with_children(|parent| {
            parent.spawn(TextBundle::from_sections([TextSection::new(
                "Settings (Esc)",
                TextStyle {
                    font: font.clone(),
                    font_size: 18.0,
                    color: Color::rgb(0.93, 0.93, 0.97),
                },
            )]));

            parent.spawn((
                OrbitSensitivityLabel,
                TextBundle::from_sections([TextSection::new(
                    "Orbit Sensitivity: 1.00x",
                    TextStyle {
                        font: font.clone(),
                        font_size: 15.0,
                        color: Color::rgb(0.88, 0.88, 0.92),
                    },
                )]),
            ));
            parent
                .spawn((
                    OrbitSensitivitySliderTrack,
                    RelativeCursorPosition::default(),
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(16.0),
                            position_type: PositionType::Relative,
                            ..default()
                        },
                        background_color: Color::rgb(0.20, 0.22, 0.28).into(),
                        ..default()
                    },
                ))
                .with_children(|slider| {
                    slider.spawn((
                        OrbitSensitivitySliderFill,
                        NodeBundle {
                            style: Style {
                                position_type: PositionType::Absolute,
                                left: Val::Px(0.0),
                                top: Val::Px(0.0),
                                bottom: Val::Px(0.0),
                                width: Val::Percent(35.0),
                                ..default()
                            },
                            background_color: Color::rgb(0.30, 0.66, 0.88).into(),
                            ..default()
                        },
                    ));
                });

            parent.spawn((
                ZoomSensitivityLabel,
                TextBundle::from_sections([TextSection::new(
                    "Zoom Sensitivity: 1.00x",
                    TextStyle {
                        font: font.clone(),
                        font_size: 15.0,
                        color: Color::rgb(0.88, 0.88, 0.92),
                    },
                )]),
            ));
            parent
                .spawn((
                    ZoomSensitivitySliderTrack,
                    RelativeCursorPosition::default(),
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(16.0),
                            position_type: PositionType::Relative,
                            ..default()
                        },
                        background_color: Color::rgb(0.20, 0.22, 0.28).into(),
                        ..default()
                    },
                ))
                .with_children(|slider| {
                    slider.spawn((
                        ZoomSensitivitySliderFill,
                        NodeBundle {
                            style: Style {
                                position_type: PositionType::Absolute,
                                left: Val::Px(0.0),
                                top: Val::Px(0.0),
                                bottom: Val::Px(0.0),
                                width: Val::Percent(35.0),
                                ..default()
                            },
                            background_color: Color::rgb(0.26, 0.58, 0.80).into(),
                            ..default()
                        },
                    ));
                });

            parent
                .spawn((
                    InvertOrbitYButton,
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(30.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        background_color: Color::rgb(0.18, 0.20, 0.25).into(),
                        ..default()
                    },
                ))
                .with_children(|button| {
                    button.spawn((
                        InvertOrbitYLabel,
                        TextBundle::from_sections([TextSection::new(
                            "Invert Orbit Y: Off",
                            TextStyle {
                                font: font.clone(),
                                font_size: 15.0,
                                color: Color::rgb(0.92, 0.92, 0.96),
                            },
                        )]),
                    ));
                });

            parent
                .spawn((
                    ResetSettingsButton,
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(30.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        background_color: Color::rgb(0.28, 0.20, 0.18).into(),
                        ..default()
                    },
                ))
                .with_children(|button| {
                    button.spawn(TextBundle::from_sections([TextSection::new(
                        "Reset Camera Defaults",
                        TextStyle {
                            font: font.clone(),
                            font_size: 15.0,
                            color: Color::rgb(0.95, 0.92, 0.90),
                        },
                    )]));
                });
        });
}

fn settings_menu_toggle_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut state: ResMut<SettingsMenuState>,
    mut menu_query: Query<&mut Style, With<SettingsMenuRoot>>,
) {
    let Some(keyboard) = keyboard else {
        return;
    };
    if !keyboard.just_pressed(KeyCode::Escape) {
        return;
    }
    state.is_open = !state.is_open;
    let display = if state.is_open {
        Display::Flex
    } else {
        Display::None
    };
    for mut style in &mut menu_query {
        style.display = display;
    }
}

fn settings_menu_slider_input_system(
    menu_state: Res<SettingsMenuState>,
    mut camera_settings: ResMut<CameraSettings>,
    orbit_track_query: Query<
        (&Interaction, &RelativeCursorPosition),
        With<OrbitSensitivitySliderTrack>,
    >,
    zoom_track_query: Query<
        (&Interaction, &RelativeCursorPosition),
        With<ZoomSensitivitySliderTrack>,
    >,
) {
    if !menu_state.is_open {
        return;
    }

    for (interaction, cursor) in &orbit_track_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(pos) = cursor.normalized else {
            continue;
        };
        let t = pos.x.clamp(0.0, 1.0);
        camera_settings.orbit_sensitivity = 0.2 + (2.5 - 0.2) * t;
    }

    for (interaction, cursor) in &zoom_track_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(pos) = cursor.normalized else {
            continue;
        };
        let t = pos.x.clamp(0.0, 1.0);
        camera_settings.zoom_sensitivity = 0.2 + (2.5 - 0.2) * t;
    }
}

fn settings_menu_toggle_input_system(
    menu_state: Res<SettingsMenuState>,
    mut camera_settings: ResMut<CameraSettings>,
    mut invert_button_query: Query<&Interaction, (With<InvertOrbitYButton>, Changed<Interaction>)>,
    mut reset_button_query: Query<&Interaction, (With<ResetSettingsButton>, Changed<Interaction>)>,
) {
    if !menu_state.is_open {
        return;
    }

    for interaction in &mut invert_button_query {
        if *interaction == Interaction::Pressed {
            camera_settings.invert_orbit_y = !camera_settings.invert_orbit_y;
        }
    }

    for interaction in &mut reset_button_query {
        if *interaction == Interaction::Pressed {
            *camera_settings = CameraSettings::default();
        }
    }
}

fn persist_camera_settings_system(camera_settings: Res<CameraSettings>) {
    if !camera_settings.is_changed() && !camera_settings.is_added() {
        return;
    }
    let serialized = match serde_json::to_string_pretty(&*camera_settings) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("Failed to serialize camera settings: {}", err);
            return;
        }
    };
    if let Some(parent) = std::path::Path::new(CAMERA_SETTINGS_PATH).parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!("Failed to create settings directory: {}", err);
            return;
        }
    }
    if let Err(err) = fs::write(CAMERA_SETTINGS_PATH, serialized) {
        eprintln!("Failed to persist camera settings: {}", err);
    }
}

fn update_settings_menu_visual_system(
    camera_settings: Res<CameraSettings>,
    mut style_sets: ParamSet<(
        Query<&mut Style, With<OrbitSensitivitySliderFill>>,
        Query<&mut Style, With<ZoomSensitivitySliderFill>>,
    )>,
    mut text_sets: ParamSet<(
        Query<&mut Text, With<OrbitSensitivityLabel>>,
        Query<&mut Text, With<ZoomSensitivityLabel>>,
        Query<&mut Text, With<InvertOrbitYLabel>>,
    )>,
) {
    if !camera_settings.is_changed() && !camera_settings.is_added() {
        return;
    }

    let orbit_t = ((camera_settings.orbit_sensitivity - 0.2) / (2.5 - 0.2)).clamp(0.0, 1.0);
    let zoom_t = ((camera_settings.zoom_sensitivity - 0.2) / (2.5 - 0.2)).clamp(0.0, 1.0);

    for mut style in &mut style_sets.p0() {
        style.width = Val::Percent(orbit_t * 100.0);
    }
    for mut style in &mut style_sets.p1() {
        style.width = Val::Percent(zoom_t * 100.0);
    }
    for mut text in &mut text_sets.p0() {
        text.sections[0].value = format!(
            "Orbit Sensitivity: {:.2}x",
            camera_settings.orbit_sensitivity
        );
    }
    for mut text in &mut text_sets.p1() {
        text.sections[0].value =
            format!("Zoom Sensitivity: {:.2}x", camera_settings.zoom_sensitivity);
    }
    for mut text in &mut text_sets.p2() {
        text.sections[0].value = format!(
            "Invert Orbit Y: {}",
            if camera_settings.invert_orbit_y {
                "On"
            } else {
                "Off"
            }
        );
    }
}

fn load_camera_settings() -> CameraSettings {
    let text = match fs::read_to_string(CAMERA_SETTINGS_PATH) {
        Ok(value) => value,
        Err(_) => return CameraSettings::default(),
    };
    let loaded: CameraSettings = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(err) => {
            eprintln!(
                "Invalid camera settings at '{}': {}. Using defaults.",
                CAMERA_SETTINGS_PATH, err
            );
            return CameraSettings::default();
        }
    };
    CameraSettings {
        orbit_sensitivity: loaded.orbit_sensitivity.clamp(0.2, 2.5),
        zoom_sensitivity: loaded.zoom_sensitivity.clamp(0.2, 2.5),
        invert_orbit_y: loaded.invert_orbit_y,
    }
}

fn save_campaign_data(path: &str, data: &CampaignSaveData) -> Result<(), String> {
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

fn unix_now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn save_slot_label(slot: u8) -> String {
    format!("slot{}", slot)
}

fn campaign_save_metadata(slot: String, path: &str, data: &CampaignSaveData) -> SaveSlotMetadata {
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

fn load_campaign_save_index() -> CampaignSaveIndex {
    let text = match fs::read_to_string(CAMPAIGN_SAVE_INDEX_PATH) {
        Ok(value) => value,
        Err(_) => return CampaignSaveIndex::default(),
    };
    serde_json::from_str::<CampaignSaveIndex>(&text).unwrap_or_default()
}

fn persist_campaign_save_index(index: &CampaignSaveIndex) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(index).map_err(|e| e.to_string())?;
    if let Some(parent) = std::path::Path::new(CAMPAIGN_SAVE_INDEX_PATH).parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(CAMPAIGN_SAVE_INDEX_PATH, serialized).map_err(|e| e.to_string())
}

fn upsert_slot_metadata(index: &mut CampaignSaveIndex, slot: u8, metadata: SaveSlotMetadata) {
    let key = save_slot_label(slot);
    if let Some(existing) = index.slots.iter_mut().find(|m| m.slot == key) {
        *existing = metadata;
    } else {
        index.slots.push(metadata);
    }
    index.slots.sort_by(|a, b| a.slot.cmp(&b.slot));
}

fn load_campaign_data(path: &str) -> Result<CampaignSaveData, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str::<CampaignSaveData>(&text).map_err(|e| e.to_string())
}

fn campaign_slot_path(slot: u8) -> &'static str {
    match slot {
        2 => CAMPAIGN_SAVE_SLOT_2_PATH,
        3 => CAMPAIGN_SAVE_SLOT_3_PATH,
        _ => CAMPAIGN_SAVE_PATH,
    }
}

fn spawn_mission_entities_from_snapshots(
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
                .spawn((data, progress, tactics, AssignedHero::default(), game_core::ActiveMission))
                .id()
        } else {
            world.spawn((data, progress, tactics, AssignedHero::default())).id()
        };
        world
            .resource_mut::<game_core::MissionBoard>()
            .entities
            .push(entity);
    }
}

fn despawn_all_mission_entities(world: &mut World) {
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

fn snapshot_campaign_from_world(world: &World) -> CampaignSaveData {
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
        run_state: world.resource::<RunState>().clone(),
        mission_map: world.resource::<game_core::MissionMap>().clone(),
        attention_state: world.resource::<game_core::AttentionState>().clone(),
        overworld_map: world.resource::<game_core::OverworldMap>().clone(),
        commander_state: world.resource::<game_core::CommanderState>().clone(),
        diplomacy_state: world.resource::<game_core::DiplomacyState>().clone(),
        interaction_board: world.resource::<game_core::InteractionBoard>().clone(),
        campaign_roster: world.resource::<game_core::CampaignRoster>().clone(),
        campaign_ledger: world.resource::<game_core::CampaignLedger>().clone(),
        campaign_event_log: world.resource::<game_core::CampaignEventLog>().clone(),
        companion_story_state: world.resource::<game_core::CompanionStoryState>().clone(),
        flashpoint_state: world
            .get_resource::<game_core::FlashpointState>()
            .cloned()
            .unwrap_or_default(),
        mission_snapshots,
        active_mission_id,
    }
}

fn migrate_campaign_save_data(mut save: CampaignSaveData) -> Result<CampaignSaveData, String> {
    if save.save_version == 0 {
        save.save_version = SAVE_VERSION_V1;
    }
    match save.save_version {
        SAVE_VERSION_V1 => {
            // v1 -> v2: explicit versioning and event log defaults.
            save.save_version = CURRENT_SAVE_VERSION;
            Ok(save)
        }
        CURRENT_SAVE_VERSION => Ok(save),
        other if other > CURRENT_SAVE_VERSION => Err(format!(
            "save version {} is newer than supported {}",
            other, CURRENT_SAVE_VERSION
        )),
        other => Err(format!("unsupported save version {}", other)),
    }
}

fn normalize_loaded_campaign(save: &mut CampaignSaveData) {
    if save.mission_snapshots.is_empty() {
        save.mission_snapshots = game_core::default_mission_snapshots();
        save.active_mission_id = None;
    }
    if save.overworld_map.regions.is_empty() {
        save.overworld_map = game_core::OverworldMap::default();
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
    save.flashpoint_state
        .chains
        .retain(|c| !c.completed && c.stage >= 1 && c.stage <= 3);
}

fn validate_and_repair_loaded_campaign(save: &mut CampaignSaveData) -> Vec<String> {
    let mut warnings = Vec::new();

    if save.commander_state.commanders.len() < save.overworld_map.factions.len() {
        warnings.push("Commander roster was incomplete and has been reset.".to_string());
        save.commander_state = game_core::CommanderState::default();
    }
    if save.diplomacy_state.relations.len() != save.overworld_map.factions.len()
        || save
            .diplomacy_state
            .relations
            .iter()
            .any(|row| row.len() != save.overworld_map.factions.len())
    {
        warnings.push("Diplomacy matrix was invalid and has been reset.".to_string());
        save.diplomacy_state = game_core::DiplomacyState::default();
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
            quest.status = game_core::CompanionQuestStatus::Failed;
        }
        if quest.progress > quest.target {
            quest.progress = quest.target;
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

fn load_and_prepare_campaign_data(path: &str) -> Result<CampaignSaveData, String> {
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
                .push(game_core::CampaignEvent {
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

fn load_campaign_save_index_state() -> CampaignSaveIndexState {
    CampaignSaveIndexState {
        index: load_campaign_save_index(),
    }
}

fn format_slot_meta(meta: Option<&SaveSlotMetadata>) -> String {
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

fn format_slot_badge(meta: Option<&SaveSlotMetadata>) -> &'static str {
    match meta {
        Some(m) if m.compatible => "[OK]",
        Some(_) => "[NEWER]",
        None => "[EMPTY]",
    }
}

fn build_save_preview(meta: &SaveSlotMetadata) -> String {
    format!(
        "{} {} | turn={} version={} seed={} timestamp={}",
        format_slot_badge(Some(meta)),
        meta.slot,
        meta.global_turn,
        meta.save_version,
        meta.map_seed,
        meta.saved_unix_seconds
    )
}

fn apply_loaded_campaign_to_world(world: &mut World, loaded: CampaignSaveData) {
    world.insert_resource(loaded.run_state);
    world.insert_resource(loaded.mission_map);
    world.insert_resource(loaded.attention_state);
    world.insert_resource(loaded.overworld_map);
    world.insert_resource(loaded.commander_state);
    world.insert_resource(loaded.diplomacy_state);
    world.insert_resource(loaded.interaction_board);
    world.insert_resource(loaded.campaign_roster);
    world.insert_resource(loaded.campaign_ledger);
    world.insert_resource(loaded.campaign_event_log);
    world.insert_resource(loaded.companion_story_state);
    world.insert_resource(loaded.flashpoint_state);
    despawn_all_mission_entities(world);
    spawn_mission_entities_from_snapshots(world, loaded.mission_snapshots, loaded.active_mission_id);
}

fn save_campaign_to_slot(world: &mut World, slot: u8) -> Result<String, String> {
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

fn load_campaign_from_path_into_world(
    world: &mut World,
    label: &str,
    path: &str,
) -> Result<String, String> {
    let loaded = load_and_prepare_campaign_data(path)?;
    let loaded_turn = loaded.run_state.global_turn;
    let loaded_version = loaded.save_version;
    apply_loaded_campaign_to_world(world, loaded);
    Ok(format!(
        "Loaded {} (v{}) t{} <- {}",
        label, loaded_version, loaded_turn, path
    ))
}

fn panel_selected_entry(
    state: &CampaignSavePanelState,
    index: &CampaignSaveIndex,
) -> (String, String, Option<SaveSlotMetadata>) {
    let slot1 = index.slots.iter().find(|m| m.slot == "slot1").cloned();
    let slot2 = index.slots.iter().find(|m| m.slot == "slot2").cloned();
    let slot3 = index.slots.iter().find(|m| m.slot == "slot3").cloned();
    let autosave = index.autosave.clone();
    let entries = [
        ("slot1".to_string(), CAMPAIGN_SAVE_PATH.to_string(), slot1),
        (
            "slot2".to_string(),
            CAMPAIGN_SAVE_SLOT_2_PATH.to_string(),
            slot2,
        ),
        (
            "slot3".to_string(),
            CAMPAIGN_SAVE_SLOT_3_PATH.to_string(),
            slot3,
        ),
        (
            "autosave".to_string(),
            CAMPAIGN_AUTOSAVE_PATH.to_string(),
            autosave,
        ),
    ];
    let idx = state.selected.min(entries.len().saturating_sub(1));
    entries[idx].clone()
}

fn campaign_save_load_input_system(world: &mut World) {
    let (save_pressed, load_pressed, shift_pressed, ctrl_pressed) = {
        let Some(keyboard) = world.get_resource::<ButtonInput<KeyCode>>() else {
            return;
        };
        (
            keyboard.just_pressed(KeyCode::F5),
            keyboard.just_pressed(KeyCode::F9),
            keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight),
            keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight),
        )
    };
    if !save_pressed && !load_pressed {
        return;
    }
    let slot = if ctrl_pressed {
        3
    } else if shift_pressed {
        2
    } else {
        1
    };
    if save_pressed {
        let message = match save_campaign_to_slot(world, slot) {
            Ok(msg) => msg,
            Err(err) => format!("Save failed: {}", err),
        };
        world.resource_mut::<CampaignSaveNotice>().message = message;
    }

    if load_pressed {
        let message = match load_campaign_from_path_into_world(
            world,
            &format!("slot {}", slot),
            campaign_slot_path(slot),
        ) {
            Ok(msg) => msg,
            Err(err) => format!("Load failed: {}", err),
        };
        world.resource_mut::<CampaignSaveNotice>().message = message;
    }
}

fn campaign_save_panel_input_system(world: &mut World) {
    let (toggle_panel, up, down, save_key, request_load, confirm, cancel) = {
        let Some(keyboard) = world.get_resource::<ButtonInput<KeyCode>>() else {
            return;
        };
        (
            keyboard.just_pressed(KeyCode::F6),
            keyboard.just_pressed(KeyCode::ArrowUp),
            keyboard.just_pressed(KeyCode::ArrowDown),
            keyboard.just_pressed(KeyCode::KeyS),
            keyboard.just_pressed(KeyCode::KeyG),
            keyboard.just_pressed(KeyCode::Enter),
            keyboard.just_pressed(KeyCode::Escape),
        )
    };

    if !toggle_panel && !up && !down && !save_key && !request_load && !confirm && !cancel {
        return;
    }

    if toggle_panel {
        let mut panel = world.resource_mut::<CampaignSavePanelState>();
        panel.open = !panel.open;
        if !panel.open {
            panel.pending_load_path = None;
            panel.pending_label = None;
            panel.preview.clear();
        }
        world.resource_mut::<CampaignSaveNotice>().message = if panel.open {
            "Save panel opened (Up/Down select, S save, G load preview, Enter confirm, Esc cancel)"
                .to_string()
        } else {
            "Save panel closed.".to_string()
        };
        return;
    }

    if !world.resource::<CampaignSavePanelState>().open {
        return;
    }

    {
        let mut panel = world.resource_mut::<CampaignSavePanelState>();
        if up {
            panel.selected = panel.selected.saturating_sub(1);
        }
        if down {
            panel.selected = (panel.selected + 1).min(3);
        }
        if cancel {
            panel.pending_load_path = None;
            panel.pending_label = None;
            panel.preview.clear();
            world.resource_mut::<CampaignSaveNotice>().message =
                "Load confirmation canceled.".to_string();
            return;
        }
    }

    if save_key {
        let selected = world.resource::<CampaignSavePanelState>().selected;
        let message = match selected {
            0 => save_campaign_to_slot(world, 1),
            1 => save_campaign_to_slot(world, 2),
            2 => save_campaign_to_slot(world, 3),
            _ => Err("Autosave slot is read-only.".to_string()),
        }
        .unwrap_or_else(|e| format!("Save failed: {}", e));
        world.resource_mut::<CampaignSaveNotice>().message = message;
        return;
    }

    if request_load {
        let (label, path, meta) = {
            let panel = world.resource::<CampaignSavePanelState>();
            let index = &world.resource::<CampaignSaveIndexState>().index;
            panel_selected_entry(&panel, index)
        };
        if let Some(m) = meta {
            if !m.compatible {
                world.resource_mut::<CampaignSaveNotice>().message = format!(
                    "Cannot load {}: save version {} is newer than supported {}.",
                    label, m.save_version, CURRENT_SAVE_VERSION
                );
                return;
            }
            let mut panel = world.resource_mut::<CampaignSavePanelState>();
            panel.pending_load_path = Some(path.clone());
            panel.pending_label = Some(label.clone());
            panel.preview = build_save_preview(&m);
            world.resource_mut::<CampaignSaveNotice>().message =
                format!("Previewing {}. Press Enter to confirm load.", label);
        } else {
            world.resource_mut::<CampaignSaveNotice>().message =
                format!("No save data found for {}.", label);
        }
        return;
    }

    if confirm {
        let (label, path) = {
            let panel = world.resource::<CampaignSavePanelState>();
            (
                panel
                    .pending_label
                    .clone()
                    .unwrap_or_else(|| "selection".to_string()),
                panel.pending_load_path.clone(),
            )
        };
        let Some(path) = path else {
            world.resource_mut::<CampaignSaveNotice>().message =
                "No load preview active. Press L first.".to_string();
            return;
        };
        let message = match load_campaign_from_path_into_world(world, &label, &path) {
            Ok(msg) => msg,
            Err(err) => format!("Load failed: {}", err),
        };
        {
            let mut panel = world.resource_mut::<CampaignSavePanelState>();
            panel.pending_load_path = None;
            panel.pending_label = None;
            panel.preview.clear();
        }
        world.resource_mut::<CampaignSaveNotice>().message = message;
    }
}

fn campaign_autosave_system(world: &mut World) {
    let (enabled, interval, last_turn) = {
        let state = world.resource::<CampaignAutosaveState>();
        (
            state.enabled,
            state.interval_turns,
            state.last_autosave_turn,
        )
    };
    if !enabled || interval == 0 {
        return;
    }
    let turn = world.resource::<RunState>().global_turn;
    if turn == 0 || turn.saturating_sub(last_turn) < interval {
        return;
    }
    let data = snapshot_campaign_from_world(world);
    let message = match save_campaign_data(CAMPAIGN_AUTOSAVE_PATH, &data) {
        Ok(_) => {
            world
                .resource_mut::<CampaignAutosaveState>()
                .last_autosave_turn = turn;
            let metadata =
                campaign_save_metadata("autosave".to_string(), CAMPAIGN_AUTOSAVE_PATH, &data);
            {
                let mut index_state = world.resource_mut::<CampaignSaveIndexState>();
                index_state.index.autosave = Some(metadata);
                let _ = persist_campaign_save_index(&index_state.index);
            }
            format!("Autosaved t{} -> {}", turn, CAMPAIGN_AUTOSAVE_PATH)
        }
        Err(err) => format!("Autosave failed: {}", err),
    };
    world.resource_mut::<CampaignSaveNotice>().message = message;
}

fn setup_custom_scenario_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    scenario_data: Res<Scenario3dData>,
) {
    let scenario = &scenario_data.0;

    let width = (scenario.world_max_x - scenario.world_min_x).max(2.0);
    let depth = (scenario.world_max_y - scenario.world_min_y).max(2.0);
    let center_x = (scenario.world_min_x + scenario.world_max_x) * 0.5;
    let center_z = (scenario.world_min_y + scenario.world_max_y) * 0.5;

    let ground_mesh = meshes.add(Mesh::from(bevy::math::primitives::Cuboid::new(
        width, 0.2, depth,
    )));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.10, 0.12, 0.16),
        perceptual_roughness: 0.95,
        ..default()
    });
    let obstacle_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.26, 0.29, 0.35),
        perceptual_roughness: 0.9,
        ..default()
    });
    let hero_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.26, 0.62, 0.85),
        perceptual_roughness: 0.55,
        ..default()
    });
    let enemy_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.84, 0.24, 0.24),
        perceptual_roughness: 0.65,
        ..default()
    });

    commands.spawn(PbrBundle {
        mesh: ground_mesh,
        material: ground_material,
        transform: Transform::from_xyz(center_x, -0.1, center_z),
        ..default()
    });

    for obstacle in &scenario.obstacles {
        let obstacle_width = (obstacle.max_x - obstacle.min_x).max(0.2);
        let obstacle_depth = (obstacle.max_y - obstacle.min_y).max(0.2);
        let obstacle_x = (obstacle.min_x + obstacle.max_x) * 0.5;
        let obstacle_z = (obstacle.min_y + obstacle.max_y) * 0.5;
        let obstacle_mesh = meshes.add(Mesh::from(bevy::math::primitives::Cuboid::new(
            obstacle_width,
            1.8,
            obstacle_depth,
        )));
        commands.spawn(PbrBundle {
            mesh: obstacle_mesh,
            material: obstacle_material.clone(),
            transform: Transform::from_xyz(obstacle_x, 0.9, obstacle_z),
            ..default()
        });
    }

    let hero_mesh = meshes.add(Mesh::from(bevy::math::primitives::Capsule3d {
        radius: 0.45,
        half_length: 0.6,
        ..default()
    }));
    let enemy_mesh = meshes.add(Mesh::from(bevy::math::primitives::Cuboid::new(
        1.0, 1.0, 1.0,
    )));

    for unit in &scenario.units {
        let is_hero = unit.team.eq_ignore_ascii_case("hero");
        if is_hero {
            commands.spawn((
                ScenarioUnitVisual { unit_id: unit.id },
                game_core::Hero {
                    name: format!("Hero {}", unit.id),
                },
                game_core::Stress {
                    value: 0.0,
                    max: 100.0,
                },
                game_core::Health {
                    current: unit.hp as f32,
                    max: unit.max_hp as f32,
                },
                game_core::HeroAbilities {
                    focus_fire_cooldown: 0,
                    stabilize_cooldown: 0,
                    sabotage_charge_cooldown: 0,
                },
                PbrBundle {
                    mesh: hero_mesh.clone(),
                    material: hero_material.clone(),
                    transform: Transform::from_xyz(unit.x, 0.7 + unit.elevation, unit.y),
                    ..default()
                },
            ));
        } else {
            commands.spawn((
                ScenarioUnitVisual { unit_id: unit.id },
                game_core::Enemy {
                    name: format!("Enemy {}", unit.id),
                },
                game_core::Health {
                    current: unit.hp as f32,
                    max: unit.max_hp as f32,
                },
                game_core::EnemyAI {
                    base_attack_power: unit.attack_damage as f32,
                    turns_until_attack: 1,
                    attack_interval: 2,
                    enraged_threshold: 0.5,
                },
                PbrBundle {
                    mesh: enemy_mesh.clone(),
                    material: enemy_material.clone(),
                    transform: Transform::from_xyz(unit.x, 0.5 + unit.elevation, unit.y),
                    ..default()
                },
            ));
        }
    }

    commands.spawn(game_core::MissionObjective {
        description: format!("Scenario loaded: {}", scenario.name),
        completed: false,
    });
}

fn setup_adventurers_guild_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let floor = meshes.add(Mesh::from(bevy::math::primitives::Cuboid::new(
        18.0, 0.2, 12.0,
    )));
    let wall_x = meshes.add(Mesh::from(bevy::math::primitives::Cuboid::new(
        0.4, 4.0, 12.0,
    )));
    let wall_z = meshes.add(Mesh::from(bevy::math::primitives::Cuboid::new(
        18.0, 4.0, 0.4,
    )));
    let table = meshes.add(Mesh::from(bevy::math::primitives::Cuboid::new(
        6.0, 0.35, 2.0,
    )));
    let banner = meshes.add(Mesh::from(bevy::math::primitives::Cuboid::new(
        1.8, 2.2, 0.1,
    )));
    let brazier = meshes.add(Mesh::from(bevy::math::primitives::Cylinder::new(0.45, 0.6)));

    let stone = materials.add(StandardMaterial {
        base_color: Color::rgb(0.18, 0.17, 0.16),
        perceptual_roughness: 0.95,
        ..default()
    });
    let timber = materials.add(StandardMaterial {
        base_color: Color::rgb(0.29, 0.20, 0.13),
        perceptual_roughness: 0.88,
        ..default()
    });
    let banner_mat = materials.add(StandardMaterial {
        base_color: Color::rgb(0.12, 0.35, 0.23),
        emissive: Color::rgb(0.02, 0.08, 0.04),
        ..default()
    });
    let brass = materials.add(StandardMaterial {
        base_color: Color::rgb(0.55, 0.44, 0.22),
        metallic: 0.55,
        perceptual_roughness: 0.4,
        ..default()
    });

    commands.spawn(PbrBundle {
        mesh: floor,
        material: stone.clone(),
        transform: Transform::from_xyz(0.0, -0.1, 0.0),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: wall_x.clone(),
        material: stone.clone(),
        transform: Transform::from_xyz(-9.0, 1.9, 0.0),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: wall_x,
        material: stone.clone(),
        transform: Transform::from_xyz(9.0, 1.9, 0.0),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: wall_z.clone(),
        material: stone.clone(),
        transform: Transform::from_xyz(0.0, 1.9, -6.0),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: wall_z,
        material: stone,
        transform: Transform::from_xyz(0.0, 1.9, 6.0),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: table,
        material: timber,
        transform: Transform::from_xyz(0.0, 0.9, 0.0),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: banner.clone(),
        material: banner_mat.clone(),
        transform: Transform::from_xyz(-7.5, 2.2, -2.5),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: banner,
        material: banner_mat,
        transform: Transform::from_xyz(7.5, 2.2, 2.5),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: brazier.clone(),
        material: brass.clone(),
        transform: Transform::from_xyz(-4.0, 0.3, 3.5),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: brazier,
        material: brass,
        transform: Transform::from_xyz(4.0, 0.3, -3.5),
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 8500.0,
            color: Color::rgb(1.0, 0.72, 0.48),
            range: 18.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(-4.0, 1.4, 3.5),
        ..default()
    });
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 8500.0,
            color: Color::rgb(1.0, 0.72, 0.48),
            range: 18.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 1.4, -3.5),
        ..default()
    });
}

fn setup_hub_hud(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::FlexStart,
                padding: UiRect::all(Val::Px(16.0)),
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                HubHudText,
                TextBundle::from_sections([TextSection::new(
                    "Guild Hub initializing...",
                    TextStyle {
                        font,
                        font_size: 20.0,
                        color: Color::rgb(0.93, 0.90, 0.80),
                    },
                )]),
            ));
        });
}

fn apply_hub_action(
    action: HubAction,
    missions: &mut Vec<game_core::MissionSnapshot>,
    attention: &mut game_core::AttentionState,
    roster: &mut game_core::CampaignRoster,
) -> String {
    let spend = |attention: &mut game_core::AttentionState, cost: f32| -> bool {
        if attention.global_energy < cost {
            return false;
        }
        attention.global_energy = (attention.global_energy - cost).max(0.0);
        true
    };

    let in_progress = |m: &game_core::MissionSnapshot| m.result == MissionResult::InProgress;

    match action {
        HubAction::AssembleExpedition => {
            if !spend(attention, 8.0) {
                return "Quartermaster: not enough attention reserve for expedition prep."
                    .to_string();
            }
            for mission in missions.iter_mut().filter(|m| in_progress(m)) {
                mission.turns_remaining = (mission.turns_remaining + 1).min(40);
                mission.reactor_integrity = (mission.reactor_integrity + 2.0).min(100.0);
                mission.alert_level = (mission.alert_level - 2.0).max(0.0);
            }
            "Quartermaster: expedition kits delivered. All active squads gain stability."
                .to_string()
        }
        HubAction::ReviewRecruits => {
            if !spend(attention, 6.0) {
                return "Guild Scribe: review stalled. attention reserve too low.".to_string();
            }
            let signed = game_core::sign_top_recruit(roster);
            let target = missions
                .iter()
                .enumerate()
                .filter(|(_, m)| in_progress(m))
                .max_by(|(_, a), (_, b)| a.alert_level.total_cmp(&b.alert_level))
                .map(|(idx, _)| idx);
            let Some(idx) = target else {
                return "Guild Scribe: no active mission needs reassignment.".to_string();
            };
            let mission = &mut missions[idx];
            mission.tactical_mode = game_core::TacticalMode::Defensive;
            mission.command_cooldown_turns = 0;
            mission.alert_level = (mission.alert_level - 4.0).max(0.0);
            mission.unattended_turns = mission.unattended_turns.saturating_sub(2);
            format!(
                "Guild Scribe: signed '{}' and reassigned '{}' to Defensive doctrine.",
                signed
                    .map(|h| h.name)
                    .unwrap_or_else(|| "no one".to_string()),
                mission.mission_name
            )
        }
        HubAction::IntelSweep => {
            if !spend(attention, 10.0) {
                return "Scouts report: intel sweep delayed. insufficient reserve.".to_string();
            }
            for mission in missions.iter_mut().filter(|m| in_progress(m)) {
                mission.alert_level = (mission.alert_level - 6.0).max(0.0);
                mission.sabotage_progress =
                    (mission.sabotage_progress + 2.0).min(mission.sabotage_goal);
            }
            "Scouts report: enemy routes mapped. alert pressure drops across the board.".to_string()
        }
        HubAction::DispatchRelief => {
            if !spend(attention, 14.0) {
                return "Relief dispatch denied: reserve below safe threshold.".to_string();
            }
            let target = missions
                .iter()
                .enumerate()
                .filter(|(_, m)| in_progress(m))
                .min_by_key(|(_, m)| m.turns_remaining)
                .map(|(idx, _)| idx);
            let Some(idx) = target else {
                return "No active crisis eligible for relief dispatch.".to_string();
            };
            let mission = &mut missions[idx];
            mission.turns_remaining = (mission.turns_remaining + 4).min(45);
            mission.reactor_integrity = (mission.reactor_integrity + 5.0).min(100.0);
            mission.alert_level = (mission.alert_level - 3.0).max(0.0);
            format!(
                "Relief wing dispatched to '{}'. Timer and integrity improved.",
                mission.mission_name
            )
        }
        HubAction::LeaveGuild => "Leaving the guild hall...".to_string(),
    }
}

fn hub_menu_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut hub_menu: ResMut<HubMenuState>,
    mut action_queue: ResMut<HubActionQueue>,
    mut exit_events: EventWriter<AppExit>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };

    if keyboard.just_pressed(KeyCode::ArrowUp) {
        hub_menu.selected = hub_menu.selected.saturating_sub(1);
    }
    if keyboard.just_pressed(KeyCode::ArrowDown) {
        hub_menu.selected = (hub_menu.selected + 1).min(4);
    }
    if keyboard.just_pressed(KeyCode::Enter) || keyboard.just_pressed(KeyCode::Space) {
        let action = HubAction::from_selected(hub_menu.selected);
        if action == HubAction::LeaveGuild {
            exit_events.send(AppExit);
            hub_menu.notice = "Leaving the guild hall...".to_string();
            return;
        }
        action_queue.pending = Some(action);
        hub_menu.notice = format!("Executing '{}'...", action.label());
    }
}

fn hub_apply_action_system(
    mut hub_menu: ResMut<HubMenuState>,
    mut action_queue: ResMut<HubActionQueue>,
    mut mission_query: Query<(bevy::prelude::Entity, &MissionData, &mut MissionProgress, &mut MissionTactics)>,
    mut attention: ResMut<game_core::AttentionState>,
    mut roster: ResMut<game_core::CampaignRoster>,
) {
    let Some(action) = action_queue.pending.take() else {
        return;
    };
    let mut entity_ids: Vec<bevy::prelude::Entity> = mission_query.iter().map(|(e, _, _, _)| e).collect();
    entity_ids.sort_by_key(|e| e.index());
    let mut snapshots: Vec<game_core::MissionSnapshot> = entity_ids
        .iter()
        .filter_map(|e| mission_query.get(*e).ok())
        .map(|(_, data, progress, tactics)| {
            game_core::MissionSnapshot::from_components(data, &progress, &tactics)
        })
        .collect();
    hub_menu.notice = apply_hub_action(action, &mut snapshots, &mut attention, &mut roster);
    action_queue.actions_taken += 1;
    for (entity, new_snap) in entity_ids.iter().zip(snapshots.iter()) {
        if let Ok((_, _, mut progress, mut tactics)) = mission_query.get_mut(*entity) {
            progress.turns_remaining = new_snap.turns_remaining;
            progress.reactor_integrity = new_snap.reactor_integrity;
            progress.alert_level = new_snap.alert_level;
            progress.sabotage_progress = new_snap.sabotage_progress;
            progress.unattended_turns = new_snap.unattended_turns;
            tactics.tactical_mode = new_snap.tactical_mode;
            tactics.command_cooldown_turns = new_snap.command_cooldown_turns;
        }
    }
}

fn update_hub_hud_system(
    hub_menu: Res<HubMenuState>,
    mission_query: Query<(&MissionData, &MissionProgress)>,
    attention: Res<game_core::AttentionState>,
    overworld: Res<game_core::OverworldMap>,
    commanders: Res<game_core::CommanderState>,
    diplomacy: Res<game_core::DiplomacyState>,
    interactions: Res<game_core::InteractionBoard>,
    action_queue: Res<HubActionQueue>,
    roster: Res<game_core::CampaignRoster>,
    ledger: Res<game_core::CampaignLedger>,
    event_log: Res<game_core::CampaignEventLog>,
    story: Res<game_core::CompanionStoryState>,
    save_notice: Res<CampaignSaveNotice>,
    save_index: Res<CampaignSaveIndexState>,
    save_panel: Res<CampaignSavePanelState>,
    mut text_query: Query<&mut Text, With<HubHudText>>,
) {
    if !hub_menu.is_changed()
        && !attention.is_changed()
        && !overworld.is_changed()
        && !commanders.is_changed()
        && !diplomacy.is_changed()
        && !interactions.is_changed()
        && !action_queue.is_changed()
        && !roster.is_changed()
        && !ledger.is_changed()
        && !event_log.is_changed()
        && !story.is_changed()
        && !save_notice.is_changed()
        && !save_index.is_changed()
        && !save_panel.is_changed()
    {
        return;
    }

    let options = [
        "Assemble Expedition",
        "Review Recruits",
        "Intel Sweep",
        "Dispatch Relief",
        "Leave Guild",
    ];
    let mut menu_lines = String::new();
    for (idx, label) in options.iter().enumerate() {
        let marker = if idx == hub_menu.selected { ">" } else { " " };
        menu_lines.push_str(&format!("{} {}\n", marker, label));
    }
    let mut board_lines = String::new();
    for (data, progress) in &mission_query {
        board_lines.push_str(&format!(
            "- {} [{}] t={} prog={:.0} alert={:.0} integ={:.0}\n",
            data.mission_name,
            match progress.result {
                MissionResult::InProgress => "In Progress",
                MissionResult::Victory => "Victory",
                MissionResult::Defeat => "Defeat",
            },
            progress.turns_remaining,
            progress.sabotage_progress,
            progress.alert_level,
            progress.reactor_integrity,
        ));
    }
    let heroes_line = roster
        .heroes
        .iter()
        .take(4)
        .map(|h| {
            format!(
                "{}(L{:.0}/S{:.0}/F{:.0})",
                h.name, h.loyalty, h.stress, h.fatigue
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let recruits_line = roster
        .recruit_pool
        .iter()
        .take(3)
        .map(|r| {
            let faction = overworld
                .factions
                .get(r.origin_faction_id)
                .map(|f| f.name.as_str())
                .unwrap_or("Unknown");
            let region = overworld
                .regions
                .iter()
                .find(|x| x.id == r.origin_region_id)
                .map(|x| x.name.as_str())
                .unwrap_or("Unknown");
            format!(
                "{}({:?}, {} / {})",
                r.codename, r.archetype, faction, region
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let last_consequence = ledger
        .records
        .last()
        .map(|r| {
            format!(
                "t{} {} {:?} hero={:?} | {}",
                r.turn, r.mission_name, r.result, r.hero_id, r.summary
            )
        })
        .unwrap_or_else(|| "none".to_string());
    let story_line = story
        .quests
        .iter()
        .find(|q| q.status == game_core::CompanionQuestStatus::Active)
        .map(|q| {
            let hero = roster
                .heroes
                .iter()
                .find(|h| h.id == q.hero_id)
                .map(|h| h.name.as_str())
                .unwrap_or("Unknown Hero");
            format!(
                "{} | #{} {:?} t{} | {} [{} {}/{}]",
                hero, q.id, q.kind, q.issued_turn, q.title, q.objective, q.progress, q.target
            )
        })
        .unwrap_or_else(|| "none".to_string());
    let event_lines = event_log
        .entries
        .iter()
        .rev()
        .take(4)
        .map(|e| format!("- t{} {}", e.turn, e.summary))
        .collect::<Vec<_>>()
        .join("\n");
    let slot1 = format_slot_meta(save_index.index.slots.iter().find(|m| m.slot == "slot1"));
    let slot2 = format_slot_meta(save_index.index.slots.iter().find(|m| m.slot == "slot2"));
    let slot3 = format_slot_meta(save_index.index.slots.iter().find(|m| m.slot == "slot3"));
    let autosave = format_slot_meta(save_index.index.autosave.as_ref());
    let panel_lines = if save_panel.open {
        let selected = save_panel.selected;
        let rows = [
            (
                "slot1",
                save_index.index.slots.iter().find(|m| m.slot == "slot1"),
            ),
            (
                "slot2",
                save_index.index.slots.iter().find(|m| m.slot == "slot2"),
            ),
            (
                "slot3",
                save_index.index.slots.iter().find(|m| m.slot == "slot3"),
            ),
            ("autosave", save_index.index.autosave.as_ref()),
        ];
        let mut lines = rows
            .iter()
            .enumerate()
            .map(|(idx, (label, meta))| {
                let marker = if idx == selected { ">" } else { " " };
                let badge = format_slot_badge(*meta);
                let body = format_slot_meta(*meta);
                format!("{} {} {} {}", marker, label, badge, body)
            })
            .collect::<Vec<_>>();
        if !save_panel.preview.is_empty() {
            lines.push(format!("Preview: {}", save_panel.preview));
        } else {
            lines.push("Preview: none".to_string());
        }
        lines.join("\n")
    } else {
        "closed (F6 to open)".to_string()
    };
    let region_label = overworld
        .regions
        .get(overworld.current_region)
        .map(|r| r.name.as_str())
        .unwrap_or("Unknown");
    let selected_label = overworld
        .regions
        .get(overworld.selected_region)
        .map(|r| r.name.as_str())
        .unwrap_or("Unknown");
    let map_lines = overworld
        .regions
        .iter()
        .map(|r| {
            let marker = if r.id == overworld.current_region {
                "*"
            } else {
                " "
            };
            let telemetry = if r.intel_level >= 65.0 {
                format!("unrest={:.0} control={:.0}", r.unrest, r.control)
            } else if r.intel_level >= 35.0 {
                let unrest_band = (r.unrest / 10.0).round() * 10.0;
                let control_band = (r.control / 10.0).round() * 10.0;
                format!("unrest~{:.0} control~{:.0}", unrest_band, control_band)
            } else {
                "unrest=? control=?".to_string()
            };
            format!(
                "{} {} [F{}] intel={:.0} {}",
                marker, r.name, r.owner_faction_id, r.intel_level, telemetry
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let faction_lines = overworld
        .factions
        .iter()
        .map(|f| {
            let avg_martial = if f.vassals.is_empty() {
                0.0
            } else {
                f.vassals.iter().map(|v| v.martial).sum::<f32>() / f.vassals.len() as f32
            };
            let managers = f
                .vassals
                .iter()
                .filter(|v| v.post == game_core::VassalPost::ZoneManager)
                .count();
            let roaming = f.vassals.len().saturating_sub(managers);
            let war_goal = f
                .war_goal_faction_id
                .and_then(|id| overworld.factions.get(id).map(|x| x.name.as_str()))
                .unwrap_or("none");
            format!(
                "- {} str={:.0} coh={:.0} goal={} focus={:.0} vassals={} (mgr {} / roam {}) avg_martial={:.0}",
                f.name,
                f.strength,
                f.cohesion,
                war_goal,
                f.war_focus,
                f.vassals.len(),
                managers,
                roaming,
                avg_martial
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let intent_lines = commanders
        .intents
        .iter()
        .map(|i| {
            let f = overworld
                .factions
                .get(i.faction_id)
                .map(|x| x.name.as_str())
                .unwrap_or("Faction");
            let r = overworld
                .regions
                .iter()
                .find(|x| x.id == i.region_id)
                .map(|x| x.name.as_str())
                .unwrap_or("Region");
            let commander = commanders
                .commanders
                .iter()
                .find(|c| c.faction_id == i.faction_id)
                .map(|c| c.name.as_str())
                .unwrap_or("Commander");
            format!(
                "- {} ({}) -> {:?} @ {} (urg {:.0})",
                f, commander, i.kind, r, i.urgency
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let relation_lines = overworld
        .factions
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != diplomacy.player_faction_id)
        .map(|(idx, f)| {
            let rel = diplomacy.relations[diplomacy.player_faction_id][idx];
            format!("- {} relation={}", f.name, rel)
        })
        .collect::<Vec<_>>()
        .join("\n");
    let offer_lines = if interactions.offers.is_empty() {
        "none".to_string()
    } else {
        interactions
            .offers
            .iter()
            .enumerate()
            .map(|(idx, o)| {
                let marker = if idx == interactions.selected {
                    ">"
                } else {
                    " "
                };
                let region = overworld
                    .regions
                    .iter()
                    .find(|r| r.id == o.region_id)
                    .map(|r| r.name.as_str())
                    .unwrap_or("Unknown Region");
                format!(
                    "{} #{} {:?} [{}]: {}",
                    marker, o.id, o.kind, region, o.summary
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    for mut text in &mut text_query {
        text.sections[0].value = format!(
            "Adventurer's Guild - Hub\n\n{}\n{}\n\nAttention Reserve: {:.0}/{:.0}\nActions Taken: {}\nLast Consequence: {}\nStory Quest: {}\n{}\n\nSave Slots\n- {}\n- {}\n- {}\n- autosave: {}\n\nSave Panel (F6)\n{}\n\nRecent Events\n{}\n\nOverworld (seed {}): current={} selected={} (travel CD {})\n{}\n\nFactions\n{}\n\nCommander Intents\n{}\n\nDiplomacy\n{}\n\nOffers (U/O select, Y accept, N decline)\n{}\n\nRoster: {}\nRecruits: {}\n\nMission Board\n{}\nControls: Up/Down select | Enter confirm | J/L pick route | T travel | 1/2/3 flashpoint intent | F5/F9 slot1 | Shift+F5/F9 slot2 | Ctrl+F5/F9 slot3 | Save Panel: Up/Down + S/G + Enter + Esc | Esc settings",
            menu_lines,
            hub_menu.notice,
            attention.global_energy,
            attention.max_energy,
            action_queue.actions_taken,
            last_consequence,
            story_line,
            save_notice.message,
            slot1,
            slot2,
            slot3,
            autosave,
            panel_lines,
            if event_lines.is_empty() { "none" } else { &event_lines },
            overworld.map_seed,
            region_label,
            selected_label,
            overworld.travel_cooldown_turns,
            map_lines,
            faction_lines,
            intent_lines,
            relation_lines,
            offer_lines,
            heroes_line,
            recruits_line,
            board_lines
        );
    }
}

fn setup_scenario_playback_ui(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");
    commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                left: Val::Percent(50.0),
                bottom: Val::Px(20.0),
                margin: UiRect::left(Val::Px(-180.0)),
                width: Val::Px(360.0),
                height: Val::Px(68.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Stretch,
                padding: UiRect::all(Val::Px(10.0)),
                row_gap: Val::Px(8.0),
                ..default()
            },
            background_color: Color::rgba(0.06, 0.07, 0.09, 0.82).into(),
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                PlaybackSliderLabel,
                TextBundle::from_sections([TextSection::new(
                    "Playback Speed: 1.00x",
                    TextStyle {
                        font: font.clone(),
                        font_size: 16.0,
                        color: Color::rgb(0.92, 0.92, 0.95),
                    },
                )]),
            ));

            parent
                .spawn((
                    PlaybackSliderTrack,
                    RelativeCursorPosition::default(),
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(18.0),
                            position_type: PositionType::Relative,
                            ..default()
                        },
                        background_color: Color::rgb(0.22, 0.24, 0.29).into(),
                        ..default()
                    },
                ))
                .with_children(|slider| {
                    slider.spawn((
                        PlaybackSliderFill,
                        NodeBundle {
                            style: Style {
                                position_type: PositionType::Absolute,
                                left: Val::Px(0.0),
                                top: Val::Px(0.0),
                                bottom: Val::Px(0.0),
                                width: Val::Percent(27.0),
                                ..default()
                            },
                            background_color: Color::rgb(0.25, 0.62, 0.86).into(),
                            ..default()
                        },
                    ));
                });
        });
}

fn advance_scenario_3d_replay_system(
    time: Res<Time>,
    mut replay: ResMut<ScenarioReplay>,
    speed: Res<ScenarioPlaybackSpeed>,
    mut unit_query: Query<(
        &ScenarioUnitVisual,
        &mut Transform,
        Option<&mut game_core::Health>,
    )>,
) {
    if replay.paused {
        return;
    }
    if replay.frame_index + 1 >= replay.frames.len() {
        return;
    }

    let effective_speed = speed.value.max(speed.min).min(speed.max);
    replay.tick_accumulator += time.delta_seconds() * effective_speed;
    if replay.tick_accumulator < replay.tick_seconds {
        return;
    }

    while replay.tick_accumulator >= replay.tick_seconds
        && replay.frame_index + 1 < replay.frames.len()
    {
        replay.tick_accumulator -= replay.tick_seconds;
        replay.frame_index += 1;
    }

    let Some(frame) = replay.frames.get(replay.frame_index) else {
        return;
    };
    let units_by_id = frame
        .units
        .iter()
        .map(|u| (u.id, u))
        .collect::<HashMap<u32, &ai::core::UnitState>>();

    for (visual, mut transform, health_opt) in &mut unit_query {
        let Some(unit) = units_by_id.get(&visual.unit_id) else {
            continue;
        };
        transform.translation.x = unit.position.x;
        transform.translation.z = unit.position.y;
        if let Some(mut health) = health_opt {
            health.current = unit.hp as f32;
            health.max = unit.max_hp as f32;
        }
    }
}

fn scenario_replay_keyboard_controls_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut replay: ResMut<ScenarioReplay>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };

    if keyboard.just_pressed(KeyCode::Space) {
        replay.paused = !replay.paused;
    }
    if keyboard.just_pressed(KeyCode::ArrowLeft) {
        replay.frame_index = replay.frame_index.saturating_sub(1);
        replay.tick_accumulator = 0.0;
        replay.paused = true;
    }
    if keyboard.just_pressed(KeyCode::ArrowRight) {
        replay.frame_index = (replay.frame_index + 1).min(replay.frames.len().saturating_sub(1));
        replay.tick_accumulator = 0.0;
        replay.paused = true;
    }
}

fn update_scenario_hud_system(
    replay: Res<ScenarioReplay>,
    mut text_query: Query<&mut Text, With<MissionHudText>>,
) {
    if !replay.is_changed() {
        return;
    }

    let Some(frame) = replay.frames.get(replay.frame_index) else {
        return;
    };
    let hero_alive = frame
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Hero && u.hp > 0)
        .count();
    let enemy_alive = frame
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Enemy && u.hp > 0)
        .count();
    let hero_hp_total = frame
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Hero)
        .map(|u| u.hp.max(0))
        .sum::<i32>();
    let enemy_hp_total = frame
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Enemy)
        .map(|u| u.hp.max(0))
        .sum::<i32>();

    for mut text in &mut text_query {
        text.sections[0].value = format!(
            "Scenario: {}\nTick: {}/{}\nAlive (H/E): {}/{}\nTotal HP (H/E): {}/{}\nPlayback: {}  |  Controls: Space pause/resume, Left prev, Right next",
            replay.name,
            replay.frame_index,
            replay.frames.len().saturating_sub(1),
            hero_alive,
            enemy_alive,
            hero_hp_total,
            enemy_hp_total,
            if replay.paused { "Paused" } else { "Running" },
        );
    }
}

fn scenario_playback_slider_input_system(
    mut speed: ResMut<ScenarioPlaybackSpeed>,
    track_query: Query<(&Interaction, &RelativeCursorPosition), With<PlaybackSliderTrack>>,
) {
    for (interaction, cursor) in &track_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(pos) = cursor.normalized else {
            continue;
        };
        let t = pos.x.clamp(0.0, 1.0);
        speed.value = speed.min + (speed.max - speed.min) * t;
    }
}

fn update_scenario_playback_slider_visual_system(
    speed: Res<ScenarioPlaybackSpeed>,
    mut fill_query: Query<&mut Style, With<PlaybackSliderFill>>,
    mut label_query: Query<&mut Text, With<PlaybackSliderLabel>>,
) {
    if !speed.is_changed() {
        return;
    }

    let denom = (speed.max - speed.min).max(0.001);
    let t = ((speed.value - speed.min) / denom).clamp(0.0, 1.0);
    for mut style in &mut fill_query {
        style.width = Val::Percent(t * 100.0);
    }
    for mut text in &mut label_query {
        text.sections[0].value = format!("Playback Speed: {:.2}x", speed.value);
    }
}

fn setup_mission_hud(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::FlexStart,
                padding: UiRect::all(Val::Px(12.0)),
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                MissionHudText,
                TextBundle::from_sections([TextSection::new(
                    "Mission HUD initializing...",
                    TextStyle {
                        font,
                        font_size: 20.0,
                        color: Color::rgb(0.92, 0.89, 0.74),
                    },
                )]),
            ));
        });
}

fn build_horde_3d_bundle(
    hero_favored: bool,
) -> (ai::tooling::CustomScenario, Vec<ai::core::SimState>) {
    let seed = if hero_favored { 202 } else { 101 };
    let ticks = 420;
    let (initial, script) = if hero_favored {
        ai::advanced::build_horde_chokepoint_hero_favored_script(
            seed,
            ticks,
            ai::core::FIXED_TICK_MS,
        )
    } else {
        ai::advanced::build_horde_chokepoint_script(seed, ticks, ai::core::FIXED_TICK_MS)
    };
    let mut scenario = horde_initial_to_custom_scenario(&initial, seed, ticks, hero_favored);
    scenario.obstacles = horde_chokepoint_obstacles();
    let frames = build_frames_from_script(initial, &script, ai::core::FIXED_TICK_MS);
    (scenario, frames)
}

fn build_frames_from_script(
    mut state: ai::core::SimState,
    script: &[Vec<ai::core::UnitIntent>],
    dt_ms: u32,
) -> Vec<ai::core::SimState> {
    let mut frames = Vec::with_capacity(script.len() + 1);
    frames.push(state.clone());
    for intents in script {
        let (next, _) = ai::core::step(state, intents, dt_ms);
        state = next;
        frames.push(state.clone());
    }
    frames
}

fn horde_chokepoint_obstacles() -> Vec<ai::tooling::ScenarioObstacle> {
    vec![
        ai::tooling::ScenarioObstacle {
            min_x: -0.8,
            max_x: 0.8,
            min_y: -9.5,
            max_y: -1.4,
        },
        ai::tooling::ScenarioObstacle {
            min_x: -0.8,
            max_x: 0.8,
            min_y: 1.4,
            max_y: 9.5,
        },
    ]
}

fn horde_initial_to_custom_scenario(
    initial: &ai::core::SimState,
    seed: u64,
    ticks: u32,
    hero_favored: bool,
) -> ai::tooling::CustomScenario {
    let units = initial
        .units
        .iter()
        .map(|u| ai::tooling::ScenarioUnit {
            id: u.id,
            team: match u.team {
                ai::core::Team::Hero => "Hero".to_string(),
                ai::core::Team::Enemy => "Enemy".to_string(),
            },
            x: u.position.x,
            y: u.position.y,
            elevation: 0.0,
            hp: u.hp,
            max_hp: u.max_hp,
            move_speed: u.move_speed_per_sec,
            attack_damage: u.attack_damage,
            attack_range: u.attack_range,
            ability_damage: u.ability_damage,
            ability_range: u.ability_range,
            heal_amount: u.heal_amount,
            heal_range: u.heal_range,
        })
        .collect::<Vec<_>>();

    ai::tooling::CustomScenario {
        name: if hero_favored {
            "horde_chokepoint_hero_favored".to_string()
        } else {
            "horde_chokepoint".to_string()
        },
        seed,
        ticks,
        world_min_x: -20.0,
        world_max_x: 20.0,
        world_min_y: -10.0,
        world_max_y: 10.0,
        cell_size: 0.7,
        elevation_zones: Vec::new(),
        slope_zones: Vec::new(),
        obstacles: Vec::new(),
        units,
    }
}

fn update_mission_hud_system(
    active_query: Query<(&MissionData, &MissionProgress, &MissionTactics, &AssignedHero), With<ActiveMission>>,
    all_missions_query: Query<(&MissionData, &MissionProgress, Option<&game_core::ActiveMission>)>,
    mission_map: Res<game_core::MissionMap>,
    attention: Option<Res<game_core::AttentionState>>,
    overworld: Option<Res<game_core::OverworldMap>>,
    roster: Option<Res<game_core::CampaignRoster>>,
    ledger: Option<Res<game_core::CampaignLedger>>,
    event_log: Option<Res<game_core::CampaignEventLog>>,
    story: Option<Res<game_core::CompanionStoryState>>,
    save_notice: Option<Res<CampaignSaveNotice>>,
    save_index: Option<Res<CampaignSaveIndexState>>,
    save_panel: Option<Res<CampaignSavePanelState>>,
    mut text_query: Query<&mut Text, With<MissionHudText>>,
) {
    let Ok((active_data, active_progress, _active_tactics, assigned_hero)) = active_query.get_single() else {
        return;
    };

    let Some(current_room) = mission_map.rooms.get(active_progress.room_index) else {
        return;
    };
    let interaction = current_room
        .interaction_nodes
        .first()
        .map(|node| node.verb.as_str())
        .unwrap_or("None");
    let next_room_status = mission_map
        .rooms
        .get(active_progress.room_index + 1)
        .map(|next_room| {
            format!(
                "{:.1} to {}",
                (next_room.sabotage_threshold - active_progress.sabotage_progress).max(0.0),
                next_room.room_name
            )
        })
        .unwrap_or_else(|| "Final room engaged".to_string());
    let result_label = match active_progress.result {
        MissionResult::InProgress => "In Progress",
        MissionResult::Victory => "Victory",
        MissionResult::Defeat => "Defeat",
    };
    let attention_line = if let Some(attn) = attention.as_ref() {
        format!(
            "Attention: {:.0}/{:.0} | Switch CD: {} | Switch Keys: [ ] or Tab",
            attn.global_energy, attn.max_energy, attn.switch_cooldown_turns
        )
    } else {
        "Attention: n/a".to_string()
    };
    let board_lines = {
        let mut out = String::new();
        for (data, progress, is_active) in &all_missions_query {
            let marker = if is_active.is_some() { ">" } else { " " };
            out.push_str(&format!(
                "{} {} [{}] t={} prog={:.0} alert={:.0} u={}\n",
                marker,
                data.mission_name,
                match progress.result {
                    MissionResult::InProgress => "In Progress",
                    MissionResult::Victory => "Victory",
                    MissionResult::Defeat => "Defeat",
                },
                progress.turns_remaining,
                progress.sabotage_progress,
                progress.alert_level,
                progress.unattended_turns
            ));
        }
        out
    };
    let assigned_line = if let Some(roster) = roster.as_ref() {
        let hero_name = assigned_hero
            .hero_id
            .and_then(|id| roster.heroes.iter().find(|h| h.id == id))
            .map(|h| {
                format!(
                    "{} ({:?}) L{:.0}/S{:.0}/F{:.0}",
                    h.name, h.archetype, h.loyalty, h.stress, h.fatigue
                )
            })
            .unwrap_or_else(|| "Unassigned".to_string());
        format!("Assigned Hero: {}", hero_name)
    } else {
        "Assigned Hero: n/a".to_string()
    };
    let quest_line = if let (Some(_roster), Some(story)) = (roster.as_ref(), story.as_ref()) {
        assigned_hero
            .hero_id
            .and_then(|id| {
                story
                    .quests
                    .iter()
                    .find(|q| {
                        q.hero_id == id && q.status == game_core::CompanionQuestStatus::Active
                    })
                    .map(|q| {
                        format!(
                            "Story Quest: #{} {:?} t{} | {} [{} {}/{}]",
                            q.id, q.kind, q.issued_turn, q.title, q.objective, q.progress, q.target
                        )
                    })
            })
            .unwrap_or_else(|| "Story Quest: none".to_string())
    } else {
        "Story Quest: n/a".to_string()
    };
    let consequence_line = ledger
        .as_ref()
        .and_then(|l| l.records.last())
        .map(|r| {
            format!(
                "Last Outcome: t{} {} {:?}",
                r.turn, r.mission_name, r.result
            )
        })
        .unwrap_or_else(|| "Last Outcome: none".to_string());
    let overworld_line = overworld
        .as_ref()
        .and_then(|o| o.regions.get(o.current_region))
        .map(|r| {
            format!(
                "Overworld Region: {} (unrest {:.0}, control {:.0})",
                r.name, r.unrest, r.control
            )
        })
        .unwrap_or_else(|| "Overworld Region: n/a".to_string());
    let save_line = save_notice
        .as_ref()
        .map(|s| s.message.clone())
        .unwrap_or_else(|| "Save: n/a".to_string());
    let event_line = event_log
        .as_ref()
        .and_then(|log| log.entries.last())
        .map(|e| format!("Latest Event: t{} {}", e.turn, e.summary))
        .unwrap_or_else(|| "Latest Event: none".to_string());
    let slot_line = save_index
        .as_ref()
        .and_then(|i| i.index.slots.iter().find(|m| m.slot == "slot1"))
        .map(|m| format!("Slot1: t{} v{} {}", m.global_turn, m.save_version, m.path))
        .unwrap_or_else(|| "Slot1: empty".to_string());
    let panel_line = save_panel
        .as_ref()
        .map(|p| {
            if p.open {
                if p.preview.is_empty() {
                    "Panel: open (no preview)".to_string()
                } else {
                    format!("Panel: open | {}", p.preview)
                }
            } else {
                "Panel: closed (F6)".to_string()
            }
        })
        .unwrap_or_else(|| "Panel: n/a".to_string());

    for mut text in &mut text_query {
        text.sections[0].value = format!(
            "Mission: {} [{}]\nTimer: {}\nSabotage: {:.1}/{:.1} | Alert: {:.1}\nRoom: {} ({:?})\nInteraction: {}\nNext Room: {}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\nMission Queue\n{}\nControls: 1/2/3 flashpoint intent | F5/F9 slot1 | Shift+F5/F9 slot2 | Ctrl+F5/F9 slot3 | F6 panel",
            active_data.mission_name,
            result_label,
            active_progress.turns_remaining,
            active_progress.sabotage_progress,
            active_progress.sabotage_goal,
            active_progress.alert_level,
            current_room.room_name,
            current_room.room_type,
            interaction,
            next_room_status,
            attention_line,
            assigned_line,
            quest_line,
            consequence_line,
            overworld_line,
            save_line,
            event_line,
            slot_line,
            panel_line,
            board_lines
        );
    }
}


fn increment_global_turn(mut run_state: ResMut<RunState>, simulation_steps: Res<SimulationSteps>) {
    if simulation_steps.0.is_none() || simulation_steps.0.unwrap_or(0) > 0 {
        run_state.global_turn += 1;
    }
}

fn exit_after_steps(
    mut app_exit_events: EventWriter<AppExit>,
    mut simulation_steps: ResMut<SimulationSteps>,
) {
    if let Some(steps) = &mut simulation_steps.0 {
        if *steps > 0 {
            *steps -= 1;
        } else {
            println!("Simulation complete after specified steps.");
            app_exit_events.send(AppExit);
        }
    }
}

fn run_if_simulation_steps_exist(simulation_steps: Option<Res<SimulationSteps>>) -> bool {
    simulation_steps.map_or(false, |steps| steps.0.is_some())
}

fn run_phase0_simulation() {
    let ticks = 120;
    let initial = ai::core::sample_duel_state(7);
    let script = ai::core::sample_duel_script(ticks);
    let result = ai::core::run_replay(initial, &script, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 0 deterministic simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Per-tick state hashes: {}",
        result.per_tick_state_hashes.len()
    );
    println!(
        "Winner: {:?}, first death tick: {:?}",
        result.metrics.winner, result.metrics.tick_to_first_death
    );
    println!(
        "Ticks elapsed (metric): {} | Final HP: {:?}",
        result.metrics.ticks_elapsed, result.metrics.final_hp_by_unit
    );
    println!(
        "Seconds elapsed: {:.2} | DPS by unit: {:?}",
        result.metrics.seconds_elapsed, result.metrics.dps_by_unit
    );
    println!(
        "Damage dealt: {:?} | Damage taken: {:?}",
        result.metrics.total_damage_by_unit, result.metrics.damage_taken_by_unit
    );
    println!(
        "Overkill: {} | Reposition events: {}",
        result.metrics.overkill_damage_total, result.metrics.reposition_for_range_events
    );
    println!(
        "Casts started/completed/failed: {}/{}/{} | avg cast delay: {:.2} ms",
        result.metrics.casts_started,
        result.metrics.casts_completed,
        result.metrics.casts_failed_out_of_range,
        result.metrics.avg_cast_delay_ms
    );
    println!(
        "Heals started/completed: {}/{} | healing by unit: {:?}",
        result.metrics.heals_started,
        result.metrics.heals_completed,
        result.metrics.total_healing_by_unit
    );
    println!(
        "Attack intents: {} | executed: {} | blocked cooldown: {} | blocked invalid: {} | dead source: {}",
        result.metrics.attack_intents,
        result.metrics.executed_attack_intents,
        result.metrics.blocked_cooldown_intents,
        result.metrics.blocked_invalid_target_intents,
        result.metrics.dead_source_attack_intents
    );
    println!(
        "Range ticks in/out: {:?}/{:?}",
        result.metrics.in_range_ticks_by_unit, result.metrics.out_of_range_ticks_by_unit
    );
    println!(
        "Movement x100: {:?} | chase ticks: {:?}",
        result.metrics.movement_distance_x100_by_unit, result.metrics.chase_ticks_by_unit
    );
    println!(
        "Focus-fire ticks: {} | max targeters: {} | target switches: {:?}",
        result.metrics.focus_fire_ticks,
        result.metrics.max_targeters_on_single_target,
        result.metrics.target_switches_by_unit
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_phase1_simulation() {
    let ticks = 200;
    let result = ai::utility::run_phase1_sample(11, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 1 utility AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Casts started/completed: {}/{} | avg cast delay: {:.2} ms",
        result.metrics.casts_started,
        result.metrics.casts_completed,
        result.metrics.avg_cast_delay_ms
    );
    println!(
        "Attack intents: {} | executed: {} | blocked cooldown: {} | blocked invalid: {}",
        result.metrics.attack_intents,
        result.metrics.executed_attack_intents,
        result.metrics.blocked_cooldown_intents,
        result.metrics.blocked_invalid_target_intents
    );
    println!(
        "Range ticks in/out: {:?}/{:?}",
        result.metrics.in_range_ticks_by_unit, result.metrics.out_of_range_ticks_by_unit
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_phase2_simulation() {
    let ticks = 260;
    let result = ai::roles::run_phase2_sample(17, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 2 role-contract AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Damage taken: {:?} | Healing by unit: {:?}",
        result.metrics.damage_taken_by_unit, result.metrics.total_healing_by_unit
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_phase3_simulation() {
    let ticks = 280;
    let run = ai::squad::run_phase3_sample(23, ticks, ai::core::FIXED_TICK_MS);
    let result = run.replay;
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 3 squad-blackboard AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    if let Some(last_boards) = run.board_history.last() {
        println!(
            "Final hero board: focus={:?} mode={:?}",
            last_boards
                .get(&ai::core::Team::Hero)
                .map(|b| b.focus_target)
                .flatten(),
            last_boards.get(&ai::core::Team::Hero).map(|b| b.mode)
        );
        println!(
            "Final enemy board: focus={:?} mode={:?}",
            last_boards
                .get(&ai::core::Team::Enemy)
                .map(|b| b.focus_target)
                .flatten(),
            last_boards.get(&ai::core::Team::Enemy).map(|b| b.mode)
        );
    }
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_phase4_simulation() {
    let ticks = 320;
    let run = ai::control::run_phase4_sample(29, ticks, ai::core::FIXED_TICK_MS);
    let result = run.replay;
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 4 CC-reservation AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Reservations observed: {} | control windows observed: {}",
        run.reservation_history
            .iter()
            .filter(|v| !v.is_empty())
            .count(),
        run.control_windows.len()
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_phase5_simulation() {
    let ticks = 320;
    let run = ai::personality::run_phase5_sample(31, ticks, ai::core::FIXED_TICK_MS);
    let result = run.replay;
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 5 personality AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    if let Some(last_modes) = run.mode_history.last() {
        println!("Final modes: {:?}", last_modes);
    }
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_phase7_simulation() {
    let ticks = 320;
    let result = ai::spatial::run_spatial_sample(37, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 7 spatial reasoning simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_phase8_simulation() {
    let ticks = 320;
    let result = ai::tactics::run_tactical_sample(37, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 8 encounter-aware tactics simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_phase9_simulation() {
    let ticks = 320;
    let result = ai::coordination::run_coordination_sample(37, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 9 advanced coordination simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_pathing_simulation() {
    let ticks = 420;
    let result = ai::advanced::run_horde_chokepoint_sample(101, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();
    let hero_alive = result
        .final_state
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Hero && u.hp > 0)
        .count();
    let enemy_alive = result
        .final_state
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Enemy && u.hp > 0)
        .count();

    println!("Pathing horde chokepoint simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!(
        "Alive units: {} (hero={}, enemy={})",
        alive_units, hero_alive, enemy_alive
    );
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | First death tick: {:?}",
        result.metrics.winner, result.metrics.tick_to_first_death
    );
    println!("Final HP: {:?}", result.metrics.final_hp_by_unit);
    println!(
        "Casts completed: {} | Heals completed: {} | Repositions: {}",
        result.metrics.casts_completed,
        result.metrics.heals_completed,
        result.metrics.reposition_for_range_events
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_pathing_hero_win_simulation() {
    let ticks = 420;
    let result =
        ai::advanced::run_horde_chokepoint_hero_favored_sample(202, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();
    let hero_alive = result
        .final_state
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Hero && u.hp > 0)
        .count();
    let enemy_alive = result
        .final_state
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Enemy && u.hp > 0)
        .count();

    println!("Pathing horde chokepoint simulation (hero favored)");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!(
        "Alive units: {} (hero={}, enemy={})",
        alive_units, hero_alive, enemy_alive
    );
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | First death tick: {:?}",
        result.metrics.winner, result.metrics.tick_to_first_death
    );
    println!("Final HP: {:?}", result.metrics.final_hp_by_unit);
    println!(
        "Casts completed: {} | Heals completed: {} | Repositions: {}",
        result.metrics.casts_completed,
        result.metrics.heals_completed,
        result.metrics.reposition_for_range_events
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

fn run_pathing_hero_hp_ablation_simulation() {
    let ticks = 420;
    let scales = [1.0_f32, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
    println!("Pathing horde hero-favored HP ablation");
    println!("Ticks: {}", ticks);
    for scale in scales {
        let result = ai::advanced::run_horde_chokepoint_hero_favored_hp_scaled_sample(
            202,
            ticks,
            ai::core::FIXED_TICK_MS,
            scale,
        );
        let hero_alive = result
            .final_state
            .units
            .iter()
            .filter(|u| u.team == ai::core::Team::Hero && u.hp > 0)
            .count();
        let enemy_alive = result
            .final_state
            .units
            .iter()
            .filter(|u| u.team == ai::core::Team::Enemy && u.hp > 0)
            .count();
        println!(
            "hp_scale={:.2} winner={:?} alive(hero/enemy)={}/{} first_death={:?} event_hash={:016x}",
            scale,
            result.metrics.winner,
            hero_alive,
            enemy_alive,
            result.metrics.tick_to_first_death,
            result.event_log_hash
        );
    }
}

fn run_phase6_report() {
    println!("Phase 6 tooling report");

    println!("\n== Scenario Matrix ==");
    let scenarios = ai::tooling::run_scenario_matrix();
    for s in scenarios {
        println!(
            "{} | winner={} | tfd={:?} | team_ttk={:?} elim={:?} | deaths(h/e)={}/{} | casts={} heals={} | deterministic={} | ehash={:016x} shash={:016x}",
            s.name,
            s.winner,
            s.tick_to_first_death,
            s.team_ttk,
            s.eliminated_team,
            s.hero_deaths,
            s.enemy_deaths,
            s.casts,
            s.heals,
            s.deterministic,
            s.event_hash,
            s.state_hash,
        );
    }

    println!("\n== Decision Debug (first 3 ticks, top-3) ==");
    let debug = ai::tooling::build_phase5_debug(31, 3, 3);
    for tick in debug {
        println!("tick {}", tick.tick);
        for decision in tick.decisions {
            println!(
                "  unit {} mode {:?} chose {:?}",
                decision.unit_id, decision.mode, decision.chosen
            );
            for score in decision.top_k {
                println!(
                    "    score {:>6.2} {:?} reason={}",
                    score.score, score.action, score.reason
                );
            }
        }
    }

    println!("\n== CC Timeline ==");
    println!("{}", ai::tooling::reservation_timeline_summary(29, 320));
    let cc = ai::tooling::analyze_phase4_cc_metrics(29, 320);
    println!(
        "CC metrics: target={:?} windows={} links={} coverage={:.2} overlap={:.2} avg_gap={:.2}",
        cc.primary_target,
        cc.windows,
        cc.links,
        cc.coverage_ratio,
        cc.overlap_ratio,
        cc.avg_gap_ticks
    );

    println!("\n== Tuning Grid (top 5) ==");
    let tuning = ai::tooling::run_personality_grid_tuning();
    for row in tuning.into_iter().take(5) {
        println!(
            "score={} agg={:.2} ctrl={:.2} altru={:.2} winner={} hash={:016x}",
            row.score, row.aggression, row.control_bias, row.altruism, row.winner, row.event_hash
        );
    }
}

fn run_phase6_visualization() {
    let output_path = "generated/reports/ai_phase6_events.html";
    match ai::tooling::export_phase5_event_visualization(output_path, 31, 320) {
        Ok(()) => {
            println!("Phase 6 event visualization written to: {}", output_path);
            println!("Open this file in a browser to explore event filters and timeline.");
        }
        Err(err) => {
            eprintln!("Failed to write phase 6 visualization: {}", err);
        }
    }
}

fn run_pathing_visualization() {
    let output_path = "generated/reports/ai_pathing_chokepoint.html";
    match ai::tooling::export_horde_chokepoint_visualization(output_path, 101, 420) {
        Ok(()) => {
            println!("Pathing visualization written to: {}", output_path);
            println!("Open this file in a browser to inspect chokepoints and flows.");
        }
        Err(err) => {
            eprintln!("Failed to write pathing visualization: {}", err);
        }
    }
}

fn run_pathing_hero_win_visualization() {
    let output_path = "generated/reports/ai_pathing_chokepoint_hero_win.html";
    match ai::tooling::export_horde_chokepoint_hero_favored_visualization(output_path, 202, 420) {
        Ok(()) => {
            println!(
                "Pathing hero-favored visualization written to: {}",
                output_path
            );
            println!("Open this file in a browser to inspect chokepoints and flows.");
        }
        Err(err) => {
            eprintln!(
                "Failed to write hero-favored pathing visualization: {}",
                err
            );
        }
    }
}

fn run_visualization_index() {
    let output_path = "generated/reports/index.html";
    let links = vec![
        (
            "Phase 6 Event Visualization".to_string(),
            "ai_phase6_events.html".to_string(),
        ),
        (
            "Pathing Chokepoint Visualization".to_string(),
            "ai_pathing_chokepoint.html".to_string(),
        ),
        (
            "Custom Scenario Visualization (if generated)".to_string(),
            "ai_custom_scenario.html".to_string(),
        ),
    ];
    match ai::tooling::export_visualization_index(output_path, &links) {
        Ok(()) => {
            println!("Visualization index written to: {}", output_path);
            println!("Open this file in a browser to navigate generated visualizations.");
        }
        Err(err) => {
            eprintln!("Failed to write visualization index: {}", err);
        }
    }
}

fn run_write_scenario_template(path: &str) {
    match ai::tooling::write_custom_scenario_template(path) {
        Ok(()) => {
            println!("Scenario template written to: {}", path);
            println!("Edit the json and run --scenario-viz <path> to visualize it.");
        }
        Err(err) => {
            eprintln!("Failed to write scenario template: {}", err);
        }
    }
}

fn run_custom_scenario_visualization(scenario_path: &str, output_path: &str) {
    match ai::tooling::export_custom_scenario_visualization(scenario_path, output_path) {
        Ok(()) => {
            println!("Custom scenario visualization written to: {}", output_path);
            println!("Source scenario: {}", scenario_path);
        }
        Err(err) => {
            eprintln!("Failed to write custom scenario visualization: {}", err);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::schedule::Schedule;
    use std::panic::{self, AssertUnwindSafe};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parse_seed_arg_supports_decimal_and_hex() {
        assert_eq!(parse_seed_arg("42"), Some(42));
        assert_eq!(parse_seed_arg("0x2a"), Some(42));
        assert_eq!(parse_seed_arg("0X2A"), Some(42));
        assert_eq!(parse_seed_arg("nope"), None);
    }

    #[test]
    fn campaign_save_data_json_roundtrip_preserves_state() {
        let mut run_state = RunState::default();
        run_state.global_turn = 17;
        let mut story = game_core::CompanionStoryState::default();
        story.notice = "test".to_string();
        let mut snapshots = game_core::default_mission_snapshots();
        if !snapshots.is_empty() {
            snapshots[0].alert_level = 77.0;
        }

        let data = CampaignSaveData {
            save_version: CURRENT_SAVE_VERSION,
            run_state,
            mission_map: game_core::MissionMap::default(),
            attention_state: game_core::AttentionState::default(),
            overworld_map: game_core::OverworldMap::default(),
            commander_state: game_core::CommanderState::default(),
            diplomacy_state: game_core::DiplomacyState::default(),
            interaction_board: game_core::InteractionBoard::default(),
            campaign_roster: game_core::CampaignRoster::default(),
            campaign_ledger: game_core::CampaignLedger::default(),
            campaign_event_log: game_core::CampaignEventLog::default(),
            companion_story_state: story,
            flashpoint_state: game_core::FlashpointState::default(),
            mission_snapshots: snapshots,
            active_mission_id: Some(0),
        };

        let json = serde_json::to_string(&data).expect("serialize");
        let decoded: CampaignSaveData = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.run_state.global_turn, 17);
        assert_eq!(decoded.mission_snapshots.first().map(|m| m.alert_level), Some(77.0));
        assert_eq!(decoded.companion_story_state.notice, "test");
    }

    #[test]
    fn campaign_save_file_io_roundtrip_works() {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = format!("/tmp/game-campaign-save-{}.json", id);
        let mut run_state = RunState::default();
        run_state.global_turn = 3;
        let data = CampaignSaveData {
            save_version: CURRENT_SAVE_VERSION,
            run_state,
            mission_map: game_core::MissionMap::default(),
            attention_state: game_core::AttentionState::default(),
            overworld_map: game_core::OverworldMap::default(),
            commander_state: game_core::CommanderState::default(),
            diplomacy_state: game_core::DiplomacyState::default(),
            interaction_board: game_core::InteractionBoard::default(),
            campaign_roster: game_core::CampaignRoster::default(),
            campaign_ledger: game_core::CampaignLedger::default(),
            campaign_event_log: game_core::CampaignEventLog::default(),
            companion_story_state: game_core::CompanionStoryState::default(),
            flashpoint_state: game_core::FlashpointState::default(),
            mission_snapshots: game_core::default_mission_snapshots(),
            active_mission_id: None,
        };

        save_campaign_data(&path, &data).expect("save file");
        let loaded = load_campaign_data(&path).expect("load file");
        assert_eq!(loaded.run_state.global_turn, 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn campaign_slot_path_resolves_expected_files() {
        assert_eq!(campaign_slot_path(1), CAMPAIGN_SAVE_PATH);
        assert_eq!(campaign_slot_path(2), CAMPAIGN_SAVE_SLOT_2_PATH);
        assert_eq!(campaign_slot_path(3), CAMPAIGN_SAVE_SLOT_3_PATH);
        assert_eq!(campaign_slot_path(9), CAMPAIGN_SAVE_PATH);
    }

    #[test]
    fn save_index_upsert_replaces_same_slot() {
        let mut index = CampaignSaveIndex::default();
        let a = SaveSlotMetadata {
            slot: "slot1".to_string(),
            path: "a".to_string(),
            save_version: 2,
            compatible: true,
            global_turn: 2,
            map_seed: 11,
            saved_unix_seconds: 1,
        };
        let b = SaveSlotMetadata {
            slot: "slot1".to_string(),
            path: "b".to_string(),
            save_version: 2,
            compatible: true,
            global_turn: 5,
            map_seed: 12,
            saved_unix_seconds: 2,
        };
        upsert_slot_metadata(&mut index, 1, a);
        upsert_slot_metadata(&mut index, 1, b.clone());
        assert_eq!(index.slots.len(), 1);
        assert_eq!(index.slots[0].path, b.path);
        assert_eq!(index.slots[0].global_turn, b.global_turn);
    }

    #[test]
    fn save_index_file_roundtrip_works() {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = format!("/tmp/game-save-index-{}.json", id);
        let index = CampaignSaveIndex {
            slots: vec![SaveSlotMetadata {
                slot: "slot1".to_string(),
                path: "slot1.json".to_string(),
                save_version: 2,
                compatible: true,
                global_turn: 10,
                map_seed: 42,
                saved_unix_seconds: 99,
            }],
            autosave: None,
        };
        let body = serde_json::to_string_pretty(&index).expect("serialize");
        std::fs::write(&path, body).expect("write");
        let loaded_text = std::fs::read_to_string(&path).expect("read");
        let loaded: CampaignSaveIndex = serde_json::from_str(&loaded_text).expect("parse");
        assert_eq!(loaded.slots.len(), 1);
        assert_eq!(loaded.slots[0].global_turn, 10);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn slot_badge_and_preview_reflect_compatibility() {
        let meta = SaveSlotMetadata {
            slot: "slot1".to_string(),
            path: "slot1.json".to_string(),
            save_version: CURRENT_SAVE_VERSION,
            compatible: true,
            global_turn: 21,
            map_seed: 99,
            saved_unix_seconds: 123,
        };
        assert_eq!(format_slot_badge(Some(&meta)), "[OK]");
        let preview = build_save_preview(&meta);
        assert!(preview.contains("turn=21"));
        assert!(preview.contains("version="));
    }

    #[test]
    fn panel_selected_entry_maps_selected_index() {
        let mut index = CampaignSaveIndex::default();
        upsert_slot_metadata(
            &mut index,
            2,
            SaveSlotMetadata {
                slot: "slot2".to_string(),
                path: CAMPAIGN_SAVE_SLOT_2_PATH.to_string(),
                save_version: CURRENT_SAVE_VERSION,
                compatible: true,
                global_turn: 7,
                map_seed: 2,
                saved_unix_seconds: 1,
            },
        );
        let panel = CampaignSavePanelState {
            open: true,
            selected: 1,
            pending_load_path: None,
            pending_label: None,
            preview: String::new(),
        };
        let (label, path, meta) = panel_selected_entry(&panel, &index);
        assert_eq!(label, "slot2");
        assert_eq!(path, CAMPAIGN_SAVE_SLOT_2_PATH);
        assert!(meta.is_some());
    }

    #[test]
    fn migration_promotes_v1_save_to_current_version() {
        let legacy_json = serde_json::json!({
            "run_state": { "global_turn": 8 },
            "attention_state": game_core::AttentionState::default(),
            "overworld_map": game_core::OverworldMap::default(),
            "commander_state": game_core::CommanderState::default(),
            "diplomacy_state": game_core::DiplomacyState::default(),
            "interaction_board": game_core::InteractionBoard::default(),
            "campaign_roster": game_core::CampaignRoster::default(),
            "campaign_ledger": game_core::CampaignLedger::default(),
            "companion_story_state": game_core::CompanionStoryState::default()
        });
        let parsed: CampaignSaveData = serde_json::from_value(legacy_json).expect("legacy parse");
        assert_eq!(parsed.save_version, SAVE_VERSION_V1);
        let migrated = migrate_campaign_save_data(parsed).expect("migrate");
        assert_eq!(migrated.save_version, CURRENT_SAVE_VERSION);
    }

    #[test]
    fn migration_rejects_newer_unknown_version() {
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
        world.insert_resource(game_core::CampaignLedger::default());
        world.insert_resource(game_core::CampaignEventLog::default());
        world.insert_resource(game_core::CompanionStoryState::default());
        let mut data = snapshot_campaign_from_world(&world);
        data.save_version = CURRENT_SAVE_VERSION + 1;
        assert!(migrate_campaign_save_data(data).is_err());
    }

    #[test]
    fn migration_keeps_current_version_unchanged() {
        let mut world = World::new();
        world.insert_resource(RunState { global_turn: 9 });
        world.insert_resource(game_core::MissionMap::default());
        world.insert_resource(game_core::MissionBoard::default());
        world.insert_resource(game_core::AttentionState::default());
        world.insert_resource(game_core::OverworldMap::default());
        world.insert_resource(game_core::CommanderState::default());
        world.insert_resource(game_core::DiplomacyState::default());
        world.insert_resource(game_core::InteractionBoard::default());
        world.insert_resource(game_core::CampaignRoster::default());
        world.insert_resource(game_core::CampaignLedger::default());
        world.insert_resource(game_core::CampaignEventLog::default());
        world.insert_resource(game_core::CompanionStoryState::default());
        let data = snapshot_campaign_from_world(&world);
        let migrated = migrate_campaign_save_data(data.clone()).expect("migrate");
        assert_eq!(migrated.save_version, CURRENT_SAVE_VERSION);
        assert_eq!(migrated.run_state.global_turn, data.run_state.global_turn);
    }

    #[test]
    fn save_overwrite_creates_backup_file() {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = format!("/tmp/game-campaign-save-backup-{}.json", id);
        let backup_path = format!("{}.bak", path);
        let mut world = World::new();
        world.insert_resource(RunState { global_turn: 1 });
        world.insert_resource(game_core::MissionMap::default());
        world.insert_resource(game_core::MissionBoard::default());
        world.insert_resource(game_core::AttentionState::default());
        world.insert_resource(game_core::OverworldMap::default());
        world.insert_resource(game_core::CommanderState::default());
        world.insert_resource(game_core::DiplomacyState::default());
        world.insert_resource(game_core::InteractionBoard::default());
        world.insert_resource(game_core::CampaignRoster::default());
        world.insert_resource(game_core::CampaignLedger::default());
        world.insert_resource(game_core::CampaignEventLog::default());
        world.insert_resource(game_core::CompanionStoryState::default());

        let mut first = snapshot_campaign_from_world(&world);
        first.run_state.global_turn = 2;
        save_campaign_data(&path, &first).expect("first save");

        let mut second = snapshot_campaign_from_world(&world);
        second.run_state.global_turn = 3;
        save_campaign_data(&path, &second).expect("second save");

        assert!(std::path::Path::new(&backup_path).exists());
        let backup_loaded = load_campaign_data(&backup_path).expect("load backup");
        assert_eq!(backup_loaded.run_state.global_turn, 2);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&backup_path);
    }

    #[test]
    fn validation_repair_clears_invalid_region_binding() {
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
        world.insert_resource(game_core::CampaignLedger::default());
        world.insert_resource(game_core::CampaignEventLog::default());
        world.insert_resource(game_core::CompanionStoryState::default());
        spawn_mission_entities_from_snapshots(
            &mut world,
            game_core::default_mission_snapshots(),
            None,
        );
        let mut data = snapshot_campaign_from_world(&world);
        // Inject an invalid region binding into the first mission
        if let Some(snap) = data.mission_snapshots.first_mut() {
            snap.bound_region_id = Some(9999);
        }
        let warnings = validate_and_repair_loaded_campaign(&mut data);
        assert!(warnings.iter().any(|w| w.contains("invalid region binding")));
    }

    #[test]
    fn snapshot_save_load_pipeline_keeps_current_version() {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let autosave_path = format!("/tmp/game-campaign-autosave-{}.json", id);

        let mut world = World::new();
        world.insert_resource(RunState { global_turn: 10 });
        world.insert_resource(game_core::MissionMap::default());
        world.insert_resource(game_core::MissionBoard::default());
        world.insert_resource(game_core::AttentionState::default());
        world.insert_resource(game_core::OverworldMap::default());
        world.insert_resource(game_core::CommanderState::default());
        world.insert_resource(game_core::DiplomacyState::default());
        world.insert_resource(game_core::InteractionBoard::default());
        world.insert_resource(game_core::CampaignRoster::default());
        world.insert_resource(game_core::CampaignLedger::default());
        world.insert_resource(game_core::CampaignEventLog::default());
        world.insert_resource(game_core::CompanionStoryState::default());
        world.insert_resource(CampaignSaveNotice::default());
        world.insert_resource(CampaignAutosaveState {
            enabled: true,
            interval_turns: 5,
            last_autosave_turn: 0,
        });

        // local wrapper to avoid mutating global autosave path constant
        let data = snapshot_campaign_from_world(&world);
        save_campaign_data(&autosave_path, &data).expect("write autosave sample");
        let loaded = load_and_prepare_campaign_data(&autosave_path).expect("load autosave sample");
        assert_eq!(loaded.save_version, CURRENT_SAVE_VERSION);
        assert_eq!(loaded.run_state.global_turn, 10);
        let _ = std::fs::remove_file(&autosave_path);
    }

    #[test]
    fn autosave_system_updates_last_turn_when_interval_met() {
        let mut world = World::new();
        world.insert_resource(RunState { global_turn: 12 });
        world.insert_resource(game_core::MissionMap::default());
        world.insert_resource(game_core::MissionBoard::default());
        world.insert_resource(game_core::AttentionState::default());
        world.insert_resource(game_core::OverworldMap::default());
        world.insert_resource(game_core::CommanderState::default());
        world.insert_resource(game_core::DiplomacyState::default());
        world.insert_resource(game_core::InteractionBoard::default());
        world.insert_resource(game_core::CampaignRoster::default());
        world.insert_resource(game_core::CampaignLedger::default());
        world.insert_resource(game_core::CampaignEventLog::default());
        world.insert_resource(game_core::CompanionStoryState::default());
        world.insert_resource(CampaignSaveNotice::default());
        world.insert_resource(CampaignSaveIndexState::default());
        world.insert_resource(CampaignAutosaveState {
            enabled: true,
            interval_turns: 10,
            last_autosave_turn: 0,
        });

        campaign_autosave_system(&mut world);
        let state = world.resource::<CampaignAutosaveState>();
        assert_eq!(state.last_autosave_turn, 12);
        let _ = std::fs::remove_file(CAMPAIGN_AUTOSAVE_PATH);
    }

    fn campaign_signature_from_world(world: &World) -> u64 {
        let data = snapshot_campaign_from_world(world);
        let bytes = serde_json::to_vec(&data).expect("serialize campaign");
        bytes.into_iter().fold(0xcbf2_9ce4_8422_2325_u64, |acc, b| {
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
        world.insert_resource(RunState::default());
        world.insert_resource(game_core::MissionMap::default());
        world.insert_resource(game_core::MissionBoard::default());
        world.insert_resource(game_core::AttentionState::default());
        world.insert_resource(game_core::OverworldMap::from_seed(seed));
        world.insert_resource(game_core::CommanderState::default());
        world.insert_resource(game_core::DiplomacyState::default());
        world.insert_resource(game_core::InteractionBoard::default());
        world.insert_resource(game_core::CampaignRoster::default());
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
        let parsed: CampaignSaveData = serde_json::from_str(&json).expect("save chain deserialize");
        let mut migrated = migrate_campaign_save_data(parsed).expect("migrate");
        normalize_loaded_campaign(&mut migrated);
        let repairs = validate_and_repair_loaded_campaign(&mut migrated);
        apply_loaded_campaign_to_world(world, migrated);
        repairs
    }

    #[test]
    fn long_run_save_load_chain_has_no_state_drift_across_seeds() {
        let seeds = [0x11_u64, 0x22_u64, 0x1234_5678_u64];
        for seed in seeds {
            let mut world = build_campaign_test_world(seed);
            let mut schedule = build_campaign_sim_schedule();

            for turn in 1..=80_u32 {
                world.resource_mut::<RunState>().global_turn = turn;
                schedule.run(&mut world);

                let before = campaign_signature_from_world(&world);
                let repairs = canonical_roundtrip_world(&mut world);
                assert!(repairs.is_empty(), "unexpected repairs at turn {turn} seed {seed}");
                let after = campaign_signature_from_world(&world);

                assert_eq!(
                    before, after,
                    "save/load drift at turn {} for seed {} ({} != {})",
                    turn, seed, before, after
                );
            }
        }
    }

    #[test]
    fn repeated_save_migration_roundtrip_keeps_signature_stable() {
        let mut world = build_campaign_test_world(0x00AB_CDEF);
        let repairs = canonical_roundtrip_world(&mut world);
        assert!(repairs.is_empty());
        let baseline = campaign_signature_from_world(&world);
        for _ in 0..40 {
            let repairs = canonical_roundtrip_world(&mut world);
            assert!(repairs.is_empty());
        }
        let final_sig = campaign_signature_from_world(&world);
        assert_eq!(baseline, final_sig);
    }

    #[test]
    fn settings_visual_system_does_not_panic() {
        let mut app = App::new();
        app.insert_resource(CameraSettings {
            orbit_sensitivity: 1.2,
            zoom_sensitivity: 0.9,
            invert_orbit_y: true,
        });
        app.add_systems(Update, update_settings_menu_visual_system);

        app.world
            .spawn((OrbitSensitivitySliderFill, Style::default()));
        app.world
            .spawn((ZoomSensitivitySliderFill, Style::default()));
        app.world.spawn((
            OrbitSensitivityLabel,
            Text::from_section("o", TextStyle::default()),
        ));
        app.world.spawn((
            ZoomSensitivityLabel,
            Text::from_section("z", TextStyle::default()),
        ));
        app.world.spawn((
            InvertOrbitYLabel,
            Text::from_section("i", TextStyle::default()),
        ));

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            app.update();
        }));
        assert!(result.is_ok(), "settings visual system panicked");
    }

    #[test]
    fn non_headless_update_smoke_does_not_panic() {
        let mut app = App::new();

        app.insert_resource(SceneViewBounds::default())
            .insert_resource(CameraSettings::default())
            .insert_resource(SettingsMenuState::default())
            .insert_resource(SimulationSteps(None))
            .insert_resource(ButtonInput::<MouseButton>::default())
            .insert_resource(Events::<MouseMotion>::default())
            .insert_resource(Events::<MouseWheel>::default())
            .init_resource::<RunState>()
            .init_resource::<game_core::MissionMap>()
            .init_resource::<game_core::MissionBoard>();

        app.add_systems(Startup, game_core::setup_test_scene_headless);

        app.add_systems(
            Update,
            (
                increment_global_turn,
                game_core::turn_management_system,
                game_core::auto_increase_stress,
                game_core::activate_mission_system,
                game_core::mission_map_progression_system,
                game_core::player_command_input_system,
                game_core::hero_ability_system,
                game_core::enemy_ai_system,
                game_core::combat_system,
                game_core::complete_objective_system,
                game_core::end_mission_system,
                game_core::print_game_state,
                update_mission_hud_system,
            )
                .chain(),
        );
        app.add_systems(
            Update,
            (
                settings_menu_toggle_system,
                settings_menu_slider_input_system,
                settings_menu_toggle_input_system,
                persist_camera_settings_system,
                update_settings_menu_visual_system,
                orbit_camera_controller_system,
            )
                .chain(),
        );

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            app.update();
        }));
        assert!(
            result.is_ok(),
            "non-headless one-frame update smoke panicked"
        );
    }

    #[test]
    fn hub_assemble_expedition_stabilizes_active_missions() {
        let mut missions = game_core::default_mission_snapshots();
        let mut attention = game_core::AttentionState::default();
        let mut roster = game_core::CampaignRoster::default();

        let notice = apply_hub_action(
            HubAction::AssembleExpedition,
            &mut missions,
            &mut attention,
            &mut roster,
        );

        assert!(notice.contains("Quartermaster"));
        assert!(attention.global_energy < attention.max_energy);
        for mission in &missions {
            if mission.result == MissionResult::InProgress {
                assert!(mission.reactor_integrity >= 92.0);
                assert!(mission.alert_level <= 20.0);
            }
        }
    }

    #[test]
    fn hub_review_recruits_targets_high_alert_mission() {
        let mut missions = game_core::default_mission_snapshots();
        missions[0].alert_level = 12.0;
        missions[1].alert_level = 45.0;
        missions[2].alert_level = 22.0;
        let mut attention = game_core::AttentionState::default();
        let mut roster = game_core::CampaignRoster::default();
        let initial_heroes = roster.heroes.len();

        let notice = apply_hub_action(
            HubAction::ReviewRecruits,
            &mut missions,
            &mut attention,
            &mut roster,
        );

        assert!(notice.contains("signed"));
        assert_eq!(
            missions[1].tactical_mode,
            game_core::TacticalMode::Defensive
        );
        assert!(missions[1].alert_level < 45.0);
        assert_eq!(roster.heroes.len(), initial_heroes + 1);
        assert!(attention.global_energy < attention.max_energy);
    }

    #[test]
    fn hub_action_fails_when_attention_is_insufficient() {
        let mut missions = game_core::default_mission_snapshots();
        let baseline = missions[0].clone();
        let mut attention = game_core::AttentionState::default();
        let mut roster = game_core::CampaignRoster::default();
        attention.global_energy = 2.0;

        let notice = apply_hub_action(
            HubAction::DispatchRelief,
            &mut missions,
            &mut attention,
            &mut roster,
        );

        assert!(notice.contains("denied") || notice.contains("threshold"));
        assert_eq!(attention.global_energy, 2.0);
        assert_eq!(missions[0].turns_remaining, baseline.turns_remaining);
        assert_eq!(
            missions[0].reactor_integrity,
            baseline.reactor_integrity
        );
    }

    #[test]
    fn hub_action_sequence_is_deterministic() {
        let actions = [
            HubAction::AssembleExpedition,
            HubAction::ReviewRecruits,
            HubAction::IntelSweep,
            HubAction::DispatchRelief,
        ];

        let mut missions_a = game_core::default_mission_snapshots();
        let mut attention_a = game_core::AttentionState::default();
        let mut roster_a = game_core::CampaignRoster::default();
        let mut notices_a = Vec::new();
        for action in actions {
            notices_a.push(apply_hub_action(
                action,
                &mut missions_a,
                &mut attention_a,
                &mut roster_a,
            ));
        }

        let mut missions_b = game_core::default_mission_snapshots();
        let mut attention_b = game_core::AttentionState::default();
        let mut roster_b = game_core::CampaignRoster::default();
        let mut notices_b = Vec::new();
        for action in actions {
            notices_b.push(apply_hub_action(
                action,
                &mut missions_b,
                &mut attention_b,
                &mut roster_b,
            ));
        }

        assert_eq!(notices_a, notices_b);
        assert_eq!(attention_a.global_energy, attention_b.global_energy);
        assert_eq!(missions_a.len(), missions_b.len());
        for idx in 0..missions_a.len() {
            assert_eq!(
                missions_a[idx].turns_remaining,
                missions_b[idx].turns_remaining
            );
            assert_eq!(
                missions_a[idx].alert_level,
                missions_b[idx].alert_level
            );
            assert_eq!(
                missions_a[idx].reactor_integrity,
                missions_b[idx].reactor_integrity
            );
        }
        assert_eq!(roster_a.heroes.len(), roster_b.heroes.len());
        let names_a = roster_a
            .heroes
            .iter()
            .map(|h| h.name.clone())
            .collect::<Vec<_>>();
        let names_b = roster_b
            .heroes
            .iter()
            .map(|h| h.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(names_a, names_b);
    }
}
