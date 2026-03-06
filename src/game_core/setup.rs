use bevy::prelude::*;

use super::types::*;
use super::overworld_types::*;

pub fn setup_test_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let hero_mesh = Mesh::from(bevy::math::primitives::Capsule3d {
        radius: 0.6,
        half_length: 0.8,
        ..default()
    });
    let enemy_mesh = Mesh::from(bevy::math::primitives::Cuboid::new(1.2, 1.2, 1.2));
    let ground_mesh = Mesh::from(bevy::math::primitives::Cuboid::new(12.0, 0.2, 8.0));

    let hero_mesh_handle = meshes.add(hero_mesh);
    let enemy_mesh_handle = meshes.add(enemy_mesh);
    let ground_mesh_handle = meshes.add(ground_mesh);

    let hero_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.70, 0.73, 0.82),
        perceptual_roughness: 0.5,
        metallic: 0.05,
        ..default()
    });
    let enemy_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.85, 0.22, 0.22),
        perceptual_roughness: 0.65,
        metallic: 0.1,
        ..default()
    });
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.12, 0.14, 0.18),
        perceptual_roughness: 0.95,
        ..default()
    });

    commands.spawn(PbrBundle {
        mesh: ground_mesh_handle,
        material: ground_material,
        transform: Transform::from_xyz(0.0, -0.1, 0.0),
        ..default()
    });

    commands.spawn((
        Hero {
            name: "Warden Lyra".to_string(),
        },
        Stress {
            value: 0.0,
            max: 100.0,
        },
        Health {
            current: 100.0,
            max: 100.0,
        },
        HeroAbilities {
            focus_fire_cooldown: 0,
            stabilize_cooldown: 0,
            sabotage_charge_cooldown: 0,
        },
        PbrBundle {
            mesh: hero_mesh_handle,
            material: hero_material,
            transform: Transform::from_xyz(-2.5, 0.85, 0.0),
            ..default()
        },
    ));

    commands.spawn((
        Enemy {
            name: "Crypt Sentinel".to_string(),
        },
        Health {
            current: 40.0,
            max: 40.0,
        },
        EnemyAI {
            base_attack_power: 6.0,
            turns_until_attack: 1,
            attack_interval: 2,
            enraged_threshold: 0.5,
        },
        PbrBundle {
            mesh: enemy_mesh_handle,
            material: enemy_material,
            transform: Transform::from_xyz(2.5, 0.6, 0.0),
            ..default()
        },
    ));

    commands.spawn(MissionObjective {
        description: "Rupture the ritual anchor".to_string(),
        completed: false,
    });
}

pub fn setup_test_scene_headless(mut commands: Commands, mut board: ResMut<MissionBoard>) {
    commands.spawn((
        Hero {
            name: "Warden Lyra".to_string(),
        },
        Stress {
            value: 0.0,
            max: 100.0,
        },
        Health {
            current: 100.0,
            max: 100.0,
        },
        HeroAbilities {
            focus_fire_cooldown: 0,
            stabilize_cooldown: 0,
            sabotage_charge_cooldown: 0,
        },
    ));

    commands.spawn((
        Enemy {
            name: "Crypt Sentinel".to_string(),
        },
        Health {
            current: 40.0,
            max: 40.0,
        },
        EnemyAI {
            base_attack_power: 6.0,
            turns_until_attack: 1,
            attack_interval: 2,
            enraged_threshold: 0.5,
        },
    ));

    commands.spawn(MissionObjective {
        description: "Rupture the ritual anchor".to_string(),
        completed: false,
    });

    // Spawn mission entities for the headless scene.
    for (i, snap) in default_mission_snapshots().into_iter().enumerate() {
        let id = board.next_id;
        board.next_id += 1;
        let (data, progress, tactics) = snap.into_components(id);
        let mut entity_cmd = commands.spawn((data, progress, tactics, AssignedHero::default()));
        if i == 0 {
            entity_cmd.insert(ActiveMission);
        }
    }
}

pub fn print_game_state(
    run_state: Res<RunState>,
    mission_map: Res<MissionMap>,
    active_query: Query<(&MissionData, &MissionProgress, &MissionTactics), With<ActiveMission>>,
    hero_query: Query<(&Hero, &Stress, &Health)>,
    enemy_query: Query<(&Enemy, &Health)>,
    objective_query: Query<&MissionObjective>,
) {
    println!("--- Global Turn: {} ---", run_state.global_turn);
    if let Ok((data, progress, tactics)) = active_query.get_single() {
        println!(
            "Mission: {} (Active: {}, Result: {:?}, Timer: {})",
            data.mission_name, progress.mission_active, progress.result, progress.turns_remaining
        );
        println!(
            "Sabotage: progress {:.1}/{:.1}, reactor {:.1}, alert {:.1}",
            progress.sabotage_progress,
            progress.sabotage_goal,
            progress.reactor_integrity,
            progress.alert_level
        );
        println!(
            "Tactical mode: {:?}, command cooldown: {}",
            tactics.tactical_mode, tactics.command_cooldown_turns
        );
        if let Some(room) = mission_map.rooms.get(progress.room_index) {
            println!(
                "Map: {} | Room: {} [{} / {:?}] | Threat budget: {:.1}",
                mission_map.map_name,
                room.room_name,
                room.room_id,
                room.room_type,
                room.threat_budget
            );
            if let Some(interaction) = room.interaction_nodes.first() {
                println!(
                    "Interaction: {} -> {}",
                    interaction.verb, interaction.description
                );
            }
        }
    }
    for (hero, stress, health) in hero_query.iter() {
        println!(
            "Hero: {}, HP: {:.1}/{:.1}, Stress: {:.1}/{:.1}",
            hero.name, health.current, health.max, stress.value, stress.max
        );
    }
    for (enemy, health) in enemy_query.iter() {
        println!(
            "Enemy: {}, HP: {:.1}/{:.1}",
            enemy.name, health.current, health.max
        );
    }
    for objective in objective_query.iter() {
        println!(
            "Objective: {} (Completed: {})",
            objective.description, objective.completed
        );
    }
    println!("--------------------");
}

pub fn attention_management_system(
    run_state: Res<RunState>,
    mut attention: ResMut<AttentionState>,
) {
    if run_state.global_turn == 0 {
        return;
    }
    if attention.switch_cooldown_turns > 0 {
        attention.switch_cooldown_turns -= 1;
    }
    attention.global_energy =
        (attention.global_energy + attention.regen_per_turn).min(attention.max_energy);
}

pub fn overworld_cooldown_system(run_state: Res<RunState>, mut overworld: ResMut<OverworldMap>) {
    if run_state.global_turn == 0 {
        return;
    }
    if overworld.travel_cooldown_turns > 0 {
        overworld.travel_cooldown_turns -= 1;
    }
}
