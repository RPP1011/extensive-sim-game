use bevy::app::AppExit;
use bevy::prelude::*;

use crate::camera::{OrbitCameraController, SceneViewBounds};
use crate::game_core::{self, HubScreen, HubUiState, MissionData, MissionProgress, MissionResult, MissionTactics};
use crate::hub_types::{HubAction, HubActionQueue, HubMenuState, OverworldTerrainVisual};

pub fn sync_hub_scene_visibility_system(
    hub_ui: Res<HubUiState>,
    mut bounds: ResMut<SceneViewBounds>,
    mut terrain_visuals: Query<&mut Visibility, With<OverworldTerrainVisual>>,
    mut cameras: Query<(&mut OrbitCameraController, &mut Transform)>,
    mut last_screen: Local<Option<HubScreen>>,
) {
    let terrain_mode = matches!(hub_ui.screen, HubScreen::OverworldMap | HubScreen::Overworld);
    let target_visibility = if terrain_mode {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut visibility in &mut terrain_visuals {
        *visibility = target_visibility;
    }

    if terrain_mode {
        *bounds = SceneViewBounds {
            min_x: -58.0,
            max_x: 58.0,
            min_z: -58.0,
            max_z: 58.0,
        };
    } else {
        *bounds = SceneViewBounds::default();
    }

    if *last_screen == Some(hub_ui.screen) {
        return;
    }
    *last_screen = Some(hub_ui.screen);

    for (mut controller, mut transform) in &mut cameras {
        if terrain_mode {
            controller.focus = Vec3::new(0.0, 0.0, 0.0);
            controller.radius = 92.0;
            controller.yaw = 0.45;
            controller.pitch = 0.62;
        } else {
            controller.focus = Vec3::new(0.0, 0.0, 0.0);
            controller.radius = 20.0;
            controller.yaw = 0.0;
            controller.pitch = 0.58;
        }
        let horizontal = controller.radius * controller.pitch.cos();
        let offset = Vec3::new(
            horizontal * controller.yaw.sin(),
            controller.radius * controller.pitch.sin(),
            horizontal * controller.yaw.cos(),
        );
        transform.translation = controller.focus + offset;
        transform.look_at(controller.focus, Vec3::Y);
    }
}

pub fn apply_hub_action(
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

pub fn hub_menu_input_system(
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

pub fn hub_apply_action_system(
    mut hub_menu: ResMut<HubMenuState>,
    mut action_queue: ResMut<HubActionQueue>,
    mut mission_query: Query<(
        bevy::prelude::Entity,
        &MissionData,
        &mut MissionProgress,
        &mut MissionTactics,
    )>,
    mut attention: ResMut<game_core::AttentionState>,
    mut roster: ResMut<game_core::CampaignRoster>,
) {
    let Some(action) = action_queue.pending.take() else {
        return;
    };
    let mut entity_ids: Vec<bevy::prelude::Entity> =
        mission_query.iter().map(|(e, _, _, _)| e).collect();
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
