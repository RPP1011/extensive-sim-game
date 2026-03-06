use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;
use std::collections::HashMap;

use crate::ai;
use crate::game_core;
use crate::game_loop::MissionHudText;

use super::types::*;

pub fn advance_scenario_3d_replay_system(
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

pub fn scenario_replay_keyboard_controls_system(
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

pub fn update_scenario_hud_system(
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

pub fn scenario_playback_slider_input_system(
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

pub fn update_scenario_playback_slider_visual_system(
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

pub fn update_mission_hud_system(
    run_state: Res<game_core::RunState>,
    active_query: Query<
        (
            &game_core::MissionData,
            &game_core::MissionProgress,
            &game_core::MissionTactics,
            &game_core::AssignedHero,
        ),
        With<game_core::ActiveMission>,
    >,
    all_missions_query: Query<(
        &game_core::MissionData,
        &game_core::MissionProgress,
        Option<&game_core::ActiveMission>,
    )>,
    mission_map: Res<game_core::MissionMap>,
    attention: Option<Res<game_core::AttentionState>>,
    overworld: Option<Res<game_core::OverworldMap>>,
    roster: Option<Res<game_core::CampaignRoster>>,
    ledger: Option<Res<game_core::CampaignLedger>>,
    event_log: Option<Res<game_core::CampaignEventLog>>,
    story: Option<Res<game_core::CompanionStoryState>>,
    save_notice: Option<Res<crate::ui::save_browser::CampaignSaveNotice>>,
    save_index: Option<Res<crate::ui::save_browser::CampaignSaveIndexState>>,
    save_panel: Option<Res<crate::ui::save_browser::CampaignSavePanelState>>,
    turn_pacing: Option<Res<crate::game_loop::TurnPacingState>>,
    start_scene: Option<Res<crate::game_loop::StartSceneState>>,
    mut text_query: Query<&mut Text, With<MissionHudText>>,
) {
    if start_scene.as_ref().is_some_and(|s| s.active) {
        for mut text in &mut text_query {
            text.sections[0].value = "Adventurer Command\n\nPress Enter to start mission simulation.\nNo turns advance in this start scene.\n\nCamera: RMB orbit, MMB pan, Wheel zoom, F recenter\nSave panel: F6 | Settings: Esc".to_string();
        }
        return;
    }

    let Ok((active_data, active_progress, _active_tactics, assigned_hero)) =
        active_query.get_single()
    else {
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
        game_core::MissionResult::InProgress => "In Progress",
        game_core::MissionResult::Victory => "Victory",
        game_core::MissionResult::Defeat => "Defeat",
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
                    game_core::MissionResult::InProgress => "In Progress",
                    game_core::MissionResult::Victory => "Victory",
                    game_core::MissionResult::Defeat => "Defeat",
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
                let player_tag = if roster.player_hero_id == Some(h.id) {
                    " [PLAYER]"
                } else {
                    ""
                };
                format!(
                    "{}{} ({:?}) L{:.0}/S{:.0}/F{:.0}",
                    h.name, player_tag, h.archetype, h.loyalty, h.stress, h.fatigue
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
    let flow_line = if let Some(pacing) = turn_pacing.as_ref() {
        if pacing.paused {
            format!(
                "Flow: Paused | Step: Enter | Speed: {:.2}s/turn | Turn {}",
                pacing.seconds_per_turn, run_state.global_turn
            )
        } else {
            format!(
                "Flow: Running | Space pause | Speed: {:.2}s/turn | Turn {}",
                pacing.seconds_per_turn, run_state.global_turn
            )
        }
    } else {
        format!("Turn {}", run_state.global_turn)
    };

    for mut text in &mut text_query {
        text.sections[0].value = format!(
            "Mission: {} [{}]\n{}\nTimer: {} | Sabotage: {:.1}/{:.1} | Alert: {:.1}\nRoom: {} ({:?})\nInteraction: {} | Next: {}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\nMission Queue\n{}\nCombat: 1 Balanced | 2 Aggressive | 3 Defensive | B Breach | R Regroup\nFocus: Tab / [ ] switch mission | Camera: RMB orbit, MMB pan, Wheel zoom, F recenter\nSave: F5/F9 slot1 | Shift+F5/F9 slot2 | Ctrl+F5/F9 slot3 | F6 panel",
            active_data.mission_name,
            result_label,
            flow_line,
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
            format!("{} | {}", save_line, slot_line),
            format!("{} | {}", panel_line, event_line),
            board_lines
        );
    }
}
