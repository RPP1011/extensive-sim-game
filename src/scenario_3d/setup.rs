use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::game_core;
use crate::game_loop::MissionHudText;

use super::types::*;

pub fn setup_custom_scenario_scene(
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

pub fn setup_scenario_playback_ui(mut commands: Commands, asset_server: Res<AssetServer>) {
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

pub fn setup_mission_hud(mut commands: Commands, asset_server: Res<AssetServer>) {
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
                        font_size: 17.0,
                        color: Color::rgb(0.92, 0.89, 0.74),
                    },
                )]),
            ));
        });
}
