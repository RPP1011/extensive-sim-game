//! Champion 3D Model Loader and Animation Controller
//!
//! Loads LoL champion glTF models from `assets/lol_models/` and drives
//! animations based on sim events during replay playback.

use std::collections::HashMap;

use bevy::gltf::Gltf;
use bevy::prelude::*;

use crate::ai::core::{SimEvent, Team};

use super::types::*;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Maps champion name -> ability name -> animation clip name (e.g., "spell1").
/// Loaded from `assets/lol_models/anim_map.json`.
#[derive(Resource, Default)]
pub struct ChampionAnimMap {
    /// champion_name -> { ability_name -> anim_clip_name }
    pub abilities: HashMap<String, HashMap<String, String>>,
    /// champion_name -> default anims (idle, run, attack, death)
    pub defaults: HashMap<String, ChampionDefaultAnims>,
}

#[derive(Debug, Clone)]
pub struct ChampionDefaultAnims {
    pub idle: String,
    pub run: String,
    pub attack: String,
    pub death: String,
}

impl Default for ChampionDefaultAnims {
    fn default() -> Self {
        Self {
            idle: "idle1".into(),
            run: "run".into(),
            attack: "attack1".into(),
            death: "death1".into(),
        }
    }
}

/// Tracks loaded glTF handles per champion name.
#[derive(Resource, Default)]
pub struct ChampionModelHandles {
    pub scenes: HashMap<String, Handle<Scene>>,
    pub gltfs: HashMap<String, Handle<Gltf>>,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marks a spawned champion model entity, linking it to a sim unit.
#[derive(Component)]
pub struct ChampionModel {
    pub unit_id: u32,
    pub champion_name: String,
    pub team: Team,
}

/// Tracks current animation state for a champion model.
#[derive(Component)]
pub struct ChampionAnimState {
    /// Current animation being played (e.g., "idle1", "spell2", "run").
    pub current_anim: String,
    /// Tick when the current ability animation started (to know when to return to idle/run).
    pub anim_started_tick: u64,
    /// Whether this unit is dead.
    pub is_dead: bool,
}

impl Default for ChampionAnimState {
    fn default() -> Self {
        Self {
            current_anim: "idle1".into(),
            anim_started_tick: 0,
            is_dead: false,
        }
    }
}

/// Maps animation clip names (e.g., "spell1") to their Handle<AnimationClip>.
/// Populated after the glTF is loaded.
#[derive(Component, Default)]
pub struct ChampionAnimClips {
    pub clips: HashMap<String, Handle<AnimationClip>>,
}

// ---------------------------------------------------------------------------
// Setup systems
// ---------------------------------------------------------------------------

/// Load the anim_map.json at startup.
pub fn load_anim_map_system(mut anim_map: ResMut<ChampionAnimMap>) {
    let path = "assets/lol_models/anim_map.json";
    let Ok(content) = std::fs::read_to_string(path) else {
        warn!("Could not load animation map from {path}");
        return;
    };

    let Ok(data) = serde_json::from_str::<serde_json::Value>(&content) else {
        warn!("Failed to parse anim_map.json");
        return;
    };

    let Some(map) = data.as_object() else { return };

    for (champion, info) in map {
        let mut ability_map = HashMap::new();
        let mut defaults = ChampionDefaultAnims::default();

        if let Some(abilities) = info.get("abilities").and_then(|v| v.as_object()) {
            for (ability_name, details) in abilities {
                if let Some(anim) = details.get("anim").and_then(|v| v.as_str()) {
                    ability_map.insert(ability_name.clone(), anim.to_string());
                }
            }
        }

        if let Some(idle) = info.get("idle").and_then(|v| v.as_str()) {
            defaults.idle = idle.to_string();
        }
        if let Some(run) = info.get("move").and_then(|v| v.as_str()) {
            defaults.run = run.to_string();
        }
        if let Some(atk) = info.get("basic_attack").and_then(|v| v.as_str()) {
            defaults.attack = atk.to_string();
        }
        if let Some(death) = info.get("death").and_then(|v| v.as_str()) {
            defaults.death = death.to_string();
        }

        anim_map.abilities.insert(champion.clone(), ability_map);
        anim_map.defaults.insert(champion.clone(), defaults);
    }

    info!("Loaded animation map for {} champions", anim_map.abilities.len());
}

/// Load champion glTF models referenced by the scenario.
/// Call this after the scenario data is available.
pub fn load_champion_models_system(
    asset_server: Res<AssetServer>,
    mut handles: ResMut<ChampionModelHandles>,
    replay: Res<ChampionReplayConfig>,
) {
    for champion_name in &replay.champion_names {
        let glb_path = format!("lol_models/{champion_name}.glb");
        if handles.gltfs.contains_key(champion_name) {
            continue;
        }
        let handle: Handle<Gltf> = asset_server.load(&glb_path);
        handles.gltfs.insert(champion_name.clone(), handle);
        info!("Loading champion model: {glb_path}");
    }
}

/// Configuration for which champions are in the replay.
#[derive(Resource, Default)]
pub struct ChampionReplayConfig {
    /// Unique champion names used in this replay.
    pub champion_names: Vec<String>,
    /// unit_id -> champion_name
    pub unit_champions: HashMap<u32, String>,
}

// ---------------------------------------------------------------------------
// Spawn system
// ---------------------------------------------------------------------------

/// Once glTF assets are loaded, spawn champion model entities for each unit.
pub fn spawn_champion_models_system(
    mut commands: Commands,
    gltf_assets: Res<Assets<Gltf>>,
    handles: Res<ChampionModelHandles>,
    replay_config: Res<ChampionReplayConfig>,
    replay: Res<ScenarioReplay>,
    existing: Query<&ChampionModel>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Only spawn once
    if !existing.is_empty() {
        return;
    }

    let Some(frame) = replay.frames.first() else {
        return;
    };

    for unit in &frame.units {
        let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
        let champion_name = replay_config
            .unit_champions
            .get(&unit.id)
            .cloned()
            .unwrap_or_default();

        // Try to spawn glTF model, fall back to primitive shape
        let has_model = if !champion_name.is_empty() {
            if let Some(gltf_handle) = handles.gltfs.get(&champion_name) {
                if let Some(gltf) = gltf_assets.get(gltf_handle) {
                    // Spawn the first scene from the glTF
                    let scene_handle = gltf.scenes.first().cloned();
                    if let Some(scene) = scene_handle {
                        // Build clip name -> handle map from the named animations
                        let mut clip_map = HashMap::new();
                        for (name, handle) in &gltf.named_animations {
                            clip_map.insert(name.clone(), handle.clone());
                        }
                        // Also map by index (Animation0, Animation1, ...)
                        for (i, handle) in gltf.animations.iter().enumerate() {
                            clip_map.entry(format!("Animation{i}")).or_insert_with(|| handle.clone());
                        }

                        commands.spawn((
                            ChampionModel {
                                unit_id: unit.id,
                                champion_name: champion_name.clone(),
                                team: unit.team,
                            },
                            ChampionAnimState::default(),
                            ChampionAnimClips { clips: clip_map },
                            SceneBundle {
                                scene,
                                // LoL models are ~100 units tall; scale to ~1.6m
                                transform: Transform::from_translation(world_pos)
                                    .with_scale(Vec3::splat(0.016)),
                                ..default()
                            },
                        ));
                        true
                    } else {
                        false
                    }
                } else {
                    false // Not loaded yet
                }
            } else {
                false
            }
        } else {
            false
        };

        // Fallback: colored capsule/cube (existing behavior)
        if !has_model {
            let is_hero = unit.team == Team::Hero;
            let color = if is_hero {
                Color::rgb(0.13, 0.33, 1.0)
            } else {
                Color::rgb(1.0, 0.13, 0.13)
            };
            let mat = materials.add(StandardMaterial {
                base_color: color,
                perceptual_roughness: 0.7,
                ..default()
            });
            let mesh = if is_hero {
                meshes.add(Capsule3d {
                    radius: 0.3,
                    half_length: 0.5,
                })
            } else {
                meshes.add(Cuboid::new(1.0, 1.0, 1.0))
            };
            commands.spawn((
                ChampionModel {
                    unit_id: unit.id,
                    champion_name: champion_name.clone(),
                    team: unit.team,
                },
                ChampionAnimState::default(),
                ChampionAnimClips::default(),
                PbrBundle {
                    mesh,
                    material: mat,
                    transform: Transform::from_translation(world_pos + Vec3::Y * 0.8),
                    ..default()
                },
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Animation systems
// ---------------------------------------------------------------------------

/// Drives champion animations based on replay events and unit state.
pub fn drive_champion_animations_system(
    replay: Res<ScenarioReplay>,
    anim_map: Res<ChampionAnimMap>,
    mut model_query: Query<(
        Entity,
        &ChampionModel,
        &mut ChampionAnimState,
        &ChampionAnimClips,
    )>,
    mut anim_players: Query<&mut AnimationPlayer>,
    anim_player_entities: Query<Entity, With<AnimationPlayer>>,
    children_query: Query<&Children>,
) {
    let frame_idx = replay.frame_index;
    let events = replay
        .events_per_frame
        .get(frame_idx)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let current_tick = replay
        .frames
        .get(frame_idx)
        .map(|f| f.tick)
        .unwrap_or(0);

    for (entity, model, mut anim_state, clips) in &mut model_query {
        if clips.clips.is_empty() {
            continue;
        }

        let defaults = anim_map
            .defaults
            .get(&model.champion_name)
            .cloned()
            .unwrap_or_default();
        let ability_anims = anim_map.abilities.get(&model.champion_name);

        let mut new_anim: Option<String> = None;

        for event in events {
            match event {
                SimEvent::AbilityUsed {
                    unit_id,
                    ability_name,
                    ..
                } if *unit_id == model.unit_id => {
                    if let Some(map) = ability_anims {
                        if let Some(anim_name) = map.get(ability_name) {
                            new_anim = Some(anim_name.clone());
                        }
                    }
                }
                SimEvent::DashPerformed { unit_id, .. } if *unit_id == model.unit_id => {
                    if anim_state.current_anim == defaults.idle {
                        new_anim = Some(defaults.run.clone());
                    }
                }
                SimEvent::UnitDied { unit_id, .. } if *unit_id == model.unit_id => {
                    new_anim = Some(defaults.death.clone());
                    anim_state.is_dead = true;
                }
                SimEvent::CastStarted { unit_id, .. } if *unit_id == model.unit_id => {
                    new_anim = Some(defaults.attack.clone());
                }
                _ => {}
            }
        }

        // Movement detection and idle return
        if new_anim.is_none() && !anim_state.is_dead {
            let ability_duration_ticks = 10;
            if anim_state.current_anim.starts_with("spell")
                && current_tick > anim_state.anim_started_tick + ability_duration_ticks
            {
                new_anim = Some(defaults.idle.clone());
            }

            if new_anim.is_none() && frame_idx > 0 {
                if let (Some(prev_frame), Some(curr_frame)) = (
                    replay.frames.get(frame_idx - 1),
                    replay.frames.get(frame_idx),
                ) {
                    let prev_pos = prev_frame.units.iter().find(|u| u.id == model.unit_id).map(|u| u.position);
                    let curr_pos = curr_frame.units.iter().find(|u| u.id == model.unit_id).map(|u| u.position);
                    if let (Some(prev), Some(curr)) = (prev_pos, curr_pos) {
                        let moved = ((curr.x - prev.x).powi(2) + (curr.y - prev.y).powi(2)).sqrt();
                        if moved > 0.05 {
                            if anim_state.current_anim != defaults.run {
                                new_anim = Some(defaults.run.clone());
                            }
                        } else if anim_state.current_anim == defaults.run {
                            new_anim = Some(defaults.idle.clone());
                        }
                    }
                }
            }
        }

        // Apply animation transition
        if let Some(target_anim) = new_anim {
            if target_anim != anim_state.current_anim {
                if let Some(clip_handle) = clips.clips.get(&target_anim) {
                    // Walk entity hierarchy to find the AnimationPlayer
                    if let Some(player_entity) =
                        find_animation_player_entity(entity, &children_query, &anim_player_entities)
                    {
                        if let Ok(mut player) = anim_players.get_mut(player_entity) {
                            player.play_with_transition(
                                clip_handle.clone(),
                                std::time::Duration::from_millis(200),
                            );
                            if target_anim == defaults.idle || target_anim == defaults.run {
                                player.repeat();
                            }
                        }
                    }
                }
                anim_state.current_anim = target_anim;
                anim_state.anim_started_tick = current_tick;
            }
        }
    }
}

/// Recursively search an entity's descendants for an AnimationPlayer.
fn find_animation_player_entity(
    root: Entity,
    children_query: &Query<&Children>,
    anim_players: &Query<Entity, With<AnimationPlayer>>,
) -> Option<Entity> {
    // Check direct children
    if let Ok(children) = children_query.get(root) {
        for &child in children.iter() {
            if anim_players.get(child).is_ok() {
                return Some(child);
            }
            // Recurse
            if let Some(found) = find_animation_player_entity(child, children_query, anim_players) {
                return Some(found);
            }
        }
    }
    None
}

/// Update champion model positions from replay frames.
pub fn update_champion_positions_system(
    replay: Res<ScenarioReplay>,
    mut query: Query<(&ChampionModel, &mut Transform)>,
) {
    let Some(frame) = replay.frames.get(replay.frame_index) else {
        return;
    };

    let units_by_id: HashMap<u32, _> = frame.units.iter().map(|u| (u.id, u)).collect();

    for (model, mut transform) in &mut query {
        if let Some(unit) = units_by_id.get(&model.unit_id) {
            transform.translation.x = unit.position.x;
            transform.translation.z = unit.position.y;

            // Face the direction of movement (rotate Y)
            // We need previous frame to compute heading
        }
    }
}

