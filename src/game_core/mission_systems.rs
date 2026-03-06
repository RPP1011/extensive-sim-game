use bevy::prelude::*;

use super::types::*;

pub fn turn_management_system(
    run_state: Res<RunState>,
    mut active_query: Query<(&mut MissionProgress, &mut MissionTactics), With<ActiveMission>>,
) {
    let Ok((mut progress, mut tactics)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    if run_state.global_turn > 0 && progress.turns_remaining > 0 {
        progress.turns_remaining -= 1;
    }
    if tactics.command_cooldown_turns > 0 {
        tactics.command_cooldown_turns -= 1;
    }
}

pub fn auto_increase_stress(mut query: Query<&mut Stress, With<Hero>>, run_state: Res<RunState>) {
    if run_state.global_turn % 10 == 0 && run_state.global_turn > 0 {
        for mut stress in query.iter_mut() {
            stress.value = (stress.value + 5.0).min(stress.max);
            println!(
                "Stress automatically increased for hero! Current stress: {:.1}",
                stress.value
            );
        }
    }
}

pub fn activate_mission_system(
    run_state: Res<RunState>,
    mut active_query: Query<(&MissionData, &mut MissionProgress), With<ActiveMission>>,
) {
    let Ok((data, mut progress)) = active_query.get_single_mut() else {
        return;
    };
    if run_state.global_turn == 5 && !progress.mission_active {
        progress.mission_active = true;
        progress.result = MissionResult::InProgress;
        println!("Mission '{}' is now ACTIVE!", data.mission_name);
    }
}

pub fn mission_map_progression_system(
    run_state: Res<RunState>,
    mission_map: Res<MissionMap>,
    mut active_query: Query<&mut MissionProgress, With<ActiveMission>>,
) {
    let Ok(mut progress) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active || run_state.global_turn == 0 {
        return;
    }

    let next_room_index = progress.room_index + 1;
    let Some(next_room) = mission_map.rooms.get(next_room_index) else {
        return;
    };
    let next_room_name = next_room.room_name.clone();
    let next_room_type = next_room.room_type;
    let next_room_threshold = next_room.sabotage_threshold;

    if progress.sabotage_progress >= next_room_threshold {
        progress.room_index = next_room_index;
        println!(
            "Map progression: entered '{}' ({:?}).",
            next_room_name, next_room_type
        );
    }
}

pub fn player_command_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut active_query: Query<(&MissionProgress, &mut MissionTactics), With<ActiveMission>>,
) {
    let Ok((progress, mut tactics)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    let Some(keyboard) = keyboard_input else {
        return;
    };

    if keyboard.just_pressed(KeyCode::Digit1) {
        tactics.tactical_mode = TacticalMode::Balanced;
        println!("Command: switched tactical mode to BALANCED.");
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        tactics.tactical_mode = TacticalMode::Aggressive;
        println!("Command: switched tactical mode to AGGRESSIVE.");
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        tactics.tactical_mode = TacticalMode::Defensive;
        println!("Command: switched tactical mode to DEFENSIVE.");
    }

    if tactics.command_cooldown_turns == 0 && keyboard.just_pressed(KeyCode::KeyB) {
        tactics.force_sabotage_order = true;
        tactics.command_cooldown_turns = 3;
        println!("Command: BREACH ORDER issued.");
    }
    if tactics.command_cooldown_turns == 0 && keyboard.just_pressed(KeyCode::KeyR) {
        tactics.force_stabilize_order = true;
        tactics.command_cooldown_turns = 3;
        println!("Command: REGROUP ORDER issued.");
    }
}

pub fn hero_ability_system(
    run_state: Res<RunState>,
    mut active_query: Query<(&mut MissionProgress, &mut MissionTactics), With<ActiveMission>>,
    mut hero_query: Query<
        (&mut HeroAbilities, &mut Health, &mut Stress),
        (With<Hero>, Without<Enemy>),
    >,
    mut enemy_query: Query<(&Enemy, &mut Health), (With<Enemy>, Without<Hero>)>,
) {
    let Ok((mut progress, mut tactics)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active || run_state.global_turn == 0 {
        return;
    }

    for (mut abilities, mut hero_health, mut stress) in hero_query.iter_mut() {
        let (focus_damage, stabilize_heal, stabilize_stress_relief, sabotage_gain) =
            match tactics.tactical_mode {
                TacticalMode::Balanced => (12.0, 6.0, 8.0, 18.0),
                TacticalMode::Aggressive => (15.0, 4.0, 6.0, 22.0),
                TacticalMode::Defensive => (9.0, 10.0, 12.0, 14.0),
            };

        if abilities.focus_fire_cooldown > 0 {
            abilities.focus_fire_cooldown -= 1;
        }
        if abilities.stabilize_cooldown > 0 {
            abilities.stabilize_cooldown -= 1;
        }
        if abilities.sabotage_charge_cooldown > 0 {
            abilities.sabotage_charge_cooldown -= 1;
        }

        if abilities.focus_fire_cooldown == 0 {
            let mut best_target: Option<(String, Mut<Health>)> = None;
            for (enemy, enemy_health) in enemy_query.iter_mut() {
                if enemy_health.current <= 0.0 {
                    continue;
                }
                match &best_target {
                    None => best_target = Some((enemy.name.clone(), enemy_health)),
                    Some((_, existing_health))
                        if enemy_health.current > existing_health.current =>
                    {
                        best_target = Some((enemy.name.clone(), enemy_health));
                    }
                    _ => {}
                }
            }

            if let Some((enemy_name, mut target_health)) = best_target {
                target_health.current = (target_health.current - focus_damage).max(0.0);
                abilities.focus_fire_cooldown = 3;
                println!(
                    "Hero ability: Focus Fire hit {} for {:.1} damage.",
                    enemy_name, focus_damage
                );
            }
        }

        let stabilize_threshold = if tactics.tactical_mode == TacticalMode::Defensive {
            14.0
        } else {
            20.0
        };
        if (abilities.stabilize_cooldown == 0 && stress.value >= stabilize_threshold)
            || tactics.force_stabilize_order
        {
            stress.value = (stress.value - stabilize_stress_relief).max(0.0);
            hero_health.current = (hero_health.current + stabilize_heal).min(hero_health.max);
            abilities.stabilize_cooldown = 7;
            println!("Hero ability: Stabilize recovered HP and reduced stress.");
            tactics.force_stabilize_order = false;
        }

        if abilities.sabotage_charge_cooldown == 0 || tactics.force_sabotage_order {
            progress.sabotage_progress =
                (progress.sabotage_progress + sabotage_gain).min(progress.sabotage_goal);
            abilities.sabotage_charge_cooldown = 4;
            println!("Hero ability: Sabotage Charge advanced ritual breach progress.");
            tactics.force_sabotage_order = false;
        }

        if tactics.tactical_mode == TacticalMode::Aggressive {
            stress.value = (stress.value + 1.0).min(stress.max);
        }
    }
}

pub fn enemy_ai_system(
    run_state: Res<RunState>,
    mut active_query: Query<&mut MissionProgress, With<ActiveMission>>,
    mut hero_query: Query<&mut Health, (With<Hero>, Without<Enemy>)>,
    mut enemy_ai_query: Query<(&Enemy, &mut EnemyAI, &Health), (With<Enemy>, Without<Hero>)>,
) {
    let Ok(mut progress) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active || run_state.global_turn == 0 {
        return;
    }

    let enemies_alive = enemy_ai_query
        .iter_mut()
        .any(|(_, _, health)| health.current > 0.0);
    if !enemies_alive {
        return;
    }

    for (enemy, mut ai, enemy_health) in enemy_ai_query.iter_mut() {
        if enemy_health.current <= 0.0 {
            continue;
        }

        if ai.turns_until_attack > 0 {
            ai.turns_until_attack -= 1;
            continue;
        }

        let enraged = (enemy_health.current / enemy_health.max) <= ai.enraged_threshold;
        let mut damage = if enraged {
            ai.base_attack_power * 1.5
        } else {
            ai.base_attack_power
        };
        if progress.alert_level >= 40.0 {
            damage += 1.5;
        }

        for mut hero_health in hero_query.iter_mut() {
            hero_health.current = (hero_health.current - damage).max(0.0);
        }

        progress.alert_level += 6.0;
        progress.sabotage_progress = (progress.sabotage_progress - 3.0).max(0.0);
        progress.reactor_integrity = (progress.reactor_integrity - 2.5).max(0.0);

        ai.turns_until_attack = ai.attack_interval;
        println!(
            "Enemy AI: {} attacked for {:.1} damage (enraged: {}).",
            enemy.name, damage, enraged
        );
    }
}

pub fn combat_system(
    run_state: Res<RunState>,
    active_query: Query<&MissionProgress, With<ActiveMission>>,
    mut hero_stress_query: Query<&mut Stress, With<Hero>>,
    mut enemy_health_query: Query<&mut Health, (With<Enemy>, Without<Hero>)>,
) {
    let Ok(progress) = active_query.get_single() else {
        return;
    };
    if !progress.mission_active || run_state.global_turn == 0 || run_state.global_turn % 5 != 0 {
        return;
    }

    let mut any_enemy_alive_after_attack = false;
    for mut enemy_health in enemy_health_query.iter_mut() {
        if enemy_health.current <= 0.0 {
            continue;
        }
        enemy_health.current = (enemy_health.current - 6.0).max(0.0);
        any_enemy_alive_after_attack |= enemy_health.current > 0.0;
    }

    if any_enemy_alive_after_attack {
        for mut stress in hero_stress_query.iter_mut() {
            stress.value = (stress.value + 2.0).min(stress.max);
        }
    }
}

pub fn complete_objective_system(
    mut objective_query: Query<&mut MissionObjective>,
    active_query: Query<&MissionProgress, With<ActiveMission>>,
) {
    let Ok(progress) = active_query.get_single() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    if progress.sabotage_progress < progress.sabotage_goal {
        return;
    }

    for mut objective in objective_query.iter_mut() {
        if !objective.completed {
            objective.completed = true;
            println!("Objective '{}' COMPLETED!", objective.description);
        }
    }
}

pub fn end_mission_system(
    objective_query: Query<&MissionObjective>,
    hero_health_query: Query<&Health, With<Hero>>,
    mut active_query: Query<(&MissionData, &mut MissionProgress), With<ActiveMission>>,
) {
    let Ok((data, mut progress)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    let hero_alive = hero_health_query.iter().all(|health| health.current > 0.0);
    if !hero_alive {
        progress.mission_active = false;
        progress.result = MissionResult::Defeat;
        println!("Mission '{}' FAILED: hero defeated.", data.mission_name);
        return;
    }

    if progress.reactor_integrity <= 0.0 {
        progress.mission_active = false;
        progress.result = MissionResult::Defeat;
        println!(
            "Mission '{}' FAILED: ward lattice collapsed.",
            data.mission_name
        );
        return;
    }

    let all_objectives_completed = objective_query.iter().all(|objective| objective.completed);
    if all_objectives_completed {
        progress.mission_active = false;
        progress.result = MissionResult::Victory;
        println!(
            "Mission '{}' COMPLETED! Mission is now INACTIVE.",
            data.mission_name
        );
        return;
    }

    if progress.turns_remaining == 0 {
        progress.mission_active = false;
        progress.result = MissionResult::Defeat;
        println!("Mission '{}' FAILED: time expired.", data.mission_name);
    }
}
