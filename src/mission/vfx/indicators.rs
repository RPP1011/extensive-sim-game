use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::ai::effects::StatusKind;
use crate::mission::sim_bridge::MissionSimState;

use super::types::*;

// ---------------------------------------------------------------------------
// Indicator sync systems
// ---------------------------------------------------------------------------

/// Sync-from-state: shield indicator rings for units with shield_hp > 0.
pub fn sync_shield_indicators_system(
    mut commands: Commands,
    sim_state: Res<MissionSimState>,
    query: Query<(Entity, &ShieldIndicator)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cached_mesh: Local<Option<Handle<Mesh>>>,
) {
    let sim = &sim_state.sim;

    let shielded_units: HashSet<u32> = sim
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.shield_hp > 0)
        .map(|u| u.id)
        .collect();

    // Despawn indicators for units no longer shielded.
    for (entity, si) in query.iter() {
        if !shielded_units.contains(&si.unit_id) {
            commands.entity(entity).despawn_recursive();
        }
    }

    let existing_ids: HashSet<u32> = query.iter().map(|(_, si)| si.unit_id).collect();

    let mesh_handle = cached_mesh.get_or_insert_with(|| {
        meshes.add(Torus {
            minor_radius: 0.04,
            major_radius: 0.5,
        })
    }).clone();

    for unit in &sim.units {
        if unit.hp <= 0 || unit.shield_hp <= 0 || existing_ids.contains(&unit.id) {
            continue;
        }
        let color = Color::rgb(0.5, 0.7, 1.0);
        let mat = materials.add(StandardMaterial {
            base_color: color,
            emissive: color.into(),
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: mesh_handle.clone(),
                material: mat,
                transform: Transform::from_translation(
                    Vec3::new(unit.position.x, 0.08, unit.position.y),
                ),
                ..default()
            },
            ShieldIndicator { unit_id: unit.id },
            Name::new("shield_indicator"),
        ));
    }
}

/// Sync-from-state: CC status indicators (stun sphere, root cuboids, etc.).
pub fn sync_status_indicators_system(
    mut commands: Commands,
    sim_state: Res<MissionSimState>,
    query: Query<(Entity, &StatusIndicator)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mesh_cache: Local<HashMap<StatusIndicatorKind, Handle<Mesh>>>,
) {
    let sim = &sim_state.sim;

    let mut active: HashSet<(u32, StatusIndicatorKind)> = HashSet::new();
    for unit in &sim.units {
        if unit.hp <= 0 {
            continue;
        }
        for se in &unit.status_effects {
            if let Some(kind) = status_kind_to_indicator(&se.kind) {
                active.insert((unit.id, kind));
            }
        }
    }

    // Despawn expired indicators.
    for (entity, si) in query.iter() {
        if !active.contains(&(si.unit_id, si.kind)) {
            commands.entity(entity).despawn_recursive();
        }
    }

    let existing: HashSet<(u32, StatusIndicatorKind)> =
        query.iter().map(|(_, si)| (si.unit_id, si.kind)).collect();

    for unit in &sim.units {
        if unit.hp <= 0 {
            continue;
        }
        for se in &unit.status_effects {
            let Some(kind) = status_kind_to_indicator(&se.kind) else {
                continue;
            };
            let key = (unit.id, kind);
            if existing.contains(&key) {
                continue;
            }

            let (color, y_offset, mesh_handle) = match kind {
                StatusIndicatorKind::Stun => {
                    let h = mesh_cache.entry(kind).or_insert_with(|| {
                        meshes.add(Sphere::new(0.1).mesh().ico(1).unwrap())
                    }).clone();
                    (Color::rgb(1.0, 0.9, 0.1), 2.3, h)
                }
                StatusIndicatorKind::Root => {
                    let h = mesh_cache.entry(kind).or_insert_with(|| {
                        meshes.add(Cuboid::new(0.08, 0.3, 0.08))
                    }).clone();
                    (Color::rgb(0.5, 0.3, 0.1), 0.1, h)
                }
                StatusIndicatorKind::Silence => {
                    let h = mesh_cache.entry(kind).or_insert_with(|| {
                        meshes.add(Cuboid::new(0.2, 0.04, 0.2))
                    }).clone();
                    (Color::rgb(0.7, 0.2, 0.9), 2.3, h)
                }
                StatusIndicatorKind::Slow => {
                    let h = mesh_cache.entry(kind).or_insert_with(|| {
                        meshes.add(Torus {
                            minor_radius: 0.03,
                            major_radius: 0.4,
                        })
                    }).clone();
                    (Color::rgb(0.3, 0.9, 1.0), 0.05, h)
                }
                StatusIndicatorKind::Fear => {
                    let h = mesh_cache.entry(kind).or_insert_with(|| {
                        meshes.add(Sphere::new(0.1).mesh().ico(1).unwrap())
                    }).clone();
                    (Color::rgb(1.0, 0.3, 0.1), 2.3, h)
                }
            };

            let mat = materials.add(StandardMaterial {
                base_color: color,
                emissive: color.into(),
                ..default()
            });
            commands.spawn((
                PbrBundle {
                    mesh: mesh_handle,
                    material: mat,
                    transform: Transform::from_translation(
                        Vec3::new(unit.position.x, y_offset, unit.position.y),
                    ),
                    ..default()
                },
                StatusIndicator { unit_id: unit.id, kind },
                Name::new("status_indicator"),
            ));
        }
    }
}

fn status_kind_to_indicator(kind: &StatusKind) -> Option<StatusIndicatorKind> {
    match kind {
        StatusKind::Stun => Some(StatusIndicatorKind::Stun),
        StatusKind::Root => Some(StatusIndicatorKind::Root),
        StatusKind::Silence => Some(StatusIndicatorKind::Silence),
        StatusKind::Slow { .. } => Some(StatusIndicatorKind::Slow),
        StatusKind::Fear { .. } => Some(StatusIndicatorKind::Fear),
        _ => None,
    }
}

/// Sync-from-state: buff/debuff rings at unit feet.
pub fn sync_buff_debuff_rings_system(
    mut commands: Commands,
    sim_state: Res<MissionSimState>,
    query: Query<(Entity, &BuffDebuffRing)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cached_mesh: Local<Option<Handle<Mesh>>>,
) {
    let sim = &sim_state.sim;

    let mut has_buff: HashSet<u32> = HashSet::new();
    let mut has_debuff: HashSet<u32> = HashSet::new();
    for unit in &sim.units {
        if unit.hp <= 0 {
            continue;
        }
        for se in &unit.status_effects {
            match &se.kind {
                StatusKind::Buff { .. } => { has_buff.insert(unit.id); }
                StatusKind::Debuff { .. } => { has_debuff.insert(unit.id); }
                _ => {}
            }
        }
    }

    // Despawn rings no longer needed.
    for (entity, ring) in query.iter() {
        let needed = if ring.is_buff {
            has_buff.contains(&ring.unit_id)
        } else {
            has_debuff.contains(&ring.unit_id)
        };
        if !needed {
            commands.entity(entity).despawn_recursive();
        }
    }

    let existing: HashSet<(u32, bool)> = query
        .iter()
        .map(|(_, r)| (r.unit_id, r.is_buff))
        .collect();

    let mesh_handle = cached_mesh.get_or_insert_with(|| {
        meshes.add(Torus {
            minor_radius: 0.025,
            major_radius: 0.45,
        })
    }).clone();

    for unit in &sim.units {
        if unit.hp <= 0 {
            continue;
        }
        let world_pos = Vec3::new(unit.position.x, 0.04, unit.position.y);

        // Buff ring.
        if has_buff.contains(&unit.id) && !existing.contains(&(unit.id, true)) {
            let color = Color::rgb(0.2, 0.9, 0.2);
            let mat = materials.add(StandardMaterial {
                base_color: color,
                emissive: color.into(),
                ..default()
            });
            commands.spawn((
                PbrBundle {
                    mesh: mesh_handle.clone(),
                    material: mat,
                    transform: Transform::from_translation(world_pos),
                    ..default()
                },
                BuffDebuffRing { unit_id: unit.id, is_buff: true },
                Name::new("buff_ring"),
            ));
        }

        // Debuff ring.
        if has_debuff.contains(&unit.id) && !existing.contains(&(unit.id, false)) {
            let color = Color::rgb(0.9, 0.2, 0.2);
            let mat = materials.add(StandardMaterial {
                base_color: color,
                emissive: color.into(),
                ..default()
            });
            commands.spawn((
                PbrBundle {
                    mesh: mesh_handle.clone(),
                    material: mat,
                    transform: Transform::from_translation(world_pos + Vec3::Y * 0.02),
                    ..default()
                },
                BuffDebuffRing { unit_id: unit.id, is_buff: false },
                Name::new("debuff_ring"),
            ));
        }
    }
}

/// Emits tiny particles for DoT/HoT effects on units.
pub fn emit_dot_hot_particles_system(
    mut commands: Commands,
    time: Res<Time>,
    sim_state: Res<MissionSimState>,
    mut query: Query<(Entity, &mut DotHotParticleTimer)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cached_mesh: Local<Option<Handle<Mesh>>>,
) {
    let delta = time.delta_seconds();
    let sim = &sim_state.sim;

    let mut dot_units: HashSet<u32> = HashSet::new();
    let mut hot_units: HashSet<u32> = HashSet::new();
    for unit in &sim.units {
        if unit.hp <= 0 {
            continue;
        }
        for se in &unit.status_effects {
            match &se.kind {
                StatusKind::Dot { .. } => { dot_units.insert(unit.id); }
                StatusKind::Hot { .. } => { hot_units.insert(unit.id); }
                _ => {}
            }
        }
    }

    // Despawn timers for units no longer affected.
    for (entity, timer) in query.iter() {
        let needed = dot_units.contains(&timer.unit_id) || hot_units.contains(&timer.unit_id);
        if !needed {
            commands.entity(entity).despawn_recursive();
        }
    }

    let existing_ids: HashSet<u32> = query.iter().map(|(_, t)| t.unit_id).collect();

    // Spawn new timers.
    for unit in &sim.units {
        if unit.hp <= 0 {
            continue;
        }
        let has_dot = dot_units.contains(&unit.id);
        let has_hot = hot_units.contains(&unit.id);
        if (has_dot || has_hot) && !existing_ids.contains(&unit.id) {
            commands.spawn((
                SpatialBundle::default(),
                DotHotParticleTimer {
                    unit_id: unit.id,
                    has_dot,
                    has_hot,
                    cooldown: 0.0,
                },
                Name::new("dot_hot_timer"),
            ));
        }
    }

    let mesh_handle = cached_mesh.get_or_insert_with(|| {
        meshes.add(Sphere::new(0.06).mesh().ico(1).unwrap())
    }).clone();

    // Tick existing timers and emit particles.
    for (_, mut timer) in query.iter_mut() {
        timer.cooldown -= delta;
        if timer.cooldown > 0.0 {
            continue;
        }
        timer.cooldown = 0.5;

        timer.has_dot = dot_units.contains(&timer.unit_id);
        timer.has_hot = hot_units.contains(&timer.unit_id);

        let Some(unit) = sim.units.iter().find(|u| u.id == timer.unit_id) else {
            continue;
        };
        let base_pos = Vec3::new(unit.position.x, 1.0, unit.position.y);

        if timer.has_dot {
            let color = Color::rgb(0.9, 0.15, 0.1);
            let mat = materials.add(StandardMaterial {
                base_color: color,
                emissive: color.into(),
                ..default()
            });
            commands.spawn((
                PbrBundle {
                    mesh: mesh_handle.clone(),
                    material: mat,
                    transform: Transform::from_translation(base_pos + Vec3::new(0.15, 0.0, 0.0)),
                    ..default()
                },
                FloatingText {
                    lifetime: 0.4,
                    velocity: Vec3::Y * 0.8,
                },
                Name::new("dot_particle"),
            ));
        }

        if timer.has_hot {
            let color = Color::rgb(0.1, 0.9, 0.3);
            let mat = materials.add(StandardMaterial {
                base_color: color,
                emissive: color.into(),
                ..default()
            });
            commands.spawn((
                PbrBundle {
                    mesh: mesh_handle.clone(),
                    material: mat,
                    transform: Transform::from_translation(base_pos + Vec3::new(-0.15, 0.0, 0.0)),
                    ..default()
                },
                FloatingText {
                    lifetime: 0.4,
                    velocity: Vec3::Y * 0.8,
                },
                Name::new("hot_particle"),
            ));
        }
    }
}
