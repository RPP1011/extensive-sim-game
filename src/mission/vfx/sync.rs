use std::collections::HashSet;

use bevy::prelude::*;

use crate::ai::effects::Area;
use crate::mission::sim_bridge::MissionSimState;
use crate::mission::tag_color::primary_tag_color;

use super::types::*;

// ---------------------------------------------------------------------------
// Sync-from-state systems
// ---------------------------------------------------------------------------

/// Syncs projectile visuals from SimState.projectiles.
pub fn sync_projectile_visuals_system(
    mut commands: Commands,
    sim_state: Res<MissionSimState>,
    mut query: Query<(Entity, &ProjectileVisual, &mut Transform)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cached_mesh: Local<Option<Handle<Mesh>>>,
) {
    let sim = &sim_state.sim;

    let active_proj_keys: HashSet<(u32, u32)> = sim
        .projectiles
        .iter()
        .map(|p| (p.source_id, p.target_id))
        .collect();

    // Despawn visuals whose projectile no longer exists.
    for (entity, pv, _) in query.iter() {
        if !active_proj_keys.contains(&(pv.source_id, pv.target_id)) {
            commands.entity(entity).despawn_recursive();
        }
    }

    // Existing visual keys.
    let existing_keys: HashSet<(u32, u32)> = query
        .iter()
        .map(|(_, pv, _)| (pv.source_id, pv.target_id))
        .collect();

    // Spawn new + update existing.
    let mesh_handle = cached_mesh.get_or_insert_with(|| {
        meshes.add(Sphere::new(0.15).mesh().ico(2).unwrap())
    }).clone();

    for proj in &sim.projectiles {
        let world_pos = Vec3::new(proj.position.x, 0.5, proj.position.y);
        let key = (proj.source_id, proj.target_id);

        if existing_keys.contains(&key) {
            for (_, pv, mut transform) in query.iter_mut() {
                if pv.source_id == proj.source_id && pv.target_id == proj.target_id {
                    transform.translation = world_pos;
                }
            }
        } else {
            let color = if !proj.on_hit.is_empty() {
                primary_tag_color(&proj.on_hit[0].tags)
            } else if !proj.on_arrival.is_empty() {
                primary_tag_color(&proj.on_arrival[0].tags)
            } else {
                Color::rgb(0.7, 0.7, 0.7)
            };
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
                ProjectileVisual {
                    source_id: proj.source_id,
                    target_id: proj.target_id,
                },
                Name::new("projectile_visual"),
            ));
        }
    }
}

/// Syncs zone visuals from SimState.zones.
pub fn sync_zone_visuals_system(
    mut commands: Commands,
    sim_state: Res<MissionSimState>,
    mut query: Query<(Entity, &ZoneVisual, &mut Transform, Option<&ZonePulseEffect>)>,
    vfx_queue: Res<VfxEventQueue>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sim = &sim_state.sim;
    let active_zone_ids: HashSet<u32> = sim.zones.iter().map(|z| z.id).collect();

    // Despawn expired zones.
    for (entity, zv, _, _) in query.iter() {
        if !active_zone_ids.contains(&zv.zone_id) {
            commands.entity(entity).despawn_recursive();
        }
    }

    let existing_ids: HashSet<u32> = query.iter().map(|(_, zv, _, _)| zv.zone_id).collect();

    // Check for pulse events.
    let pulsing_zones: HashSet<u32> = vfx_queue
        .pending
        .iter()
        .filter_map(|e| match e {
            VfxEvent::ZonePulse { zone_id } => Some(*zone_id),
            _ => None,
        })
        .collect();

    for zone in &sim.zones {
        if zone.invisible {
            continue;
        }

        let world_pos = Vec3::new(zone.position.x, 0.05, zone.position.y);
        let radius = match &zone.area {
            Area::Circle { radius } => *radius,
            Area::Ring { outer_radius, .. } => *outer_radius,
            Area::Cone { radius, .. } => *radius,
            Area::Line { length, .. } => *length * 0.5,
            Area::Spread { radius, .. } => *radius,
            _ => 1.0,
        };

        if existing_ids.contains(&zone.id) {
            for (entity, zv, mut transform, _) in query.iter_mut() {
                if zv.zone_id == zone.id {
                    transform.translation = world_pos;
                    if pulsing_zones.contains(&zone.id) {
                        commands.entity(entity).insert(ZonePulseEffect { remaining: 0.2 });
                    }
                }
            }
        } else {
            let color = primary_tag_color(
                &zone.effects.first().map(|e| &e.tags).cloned().unwrap_or_default(),
            );
            let mesh = meshes.add(Torus {
                minor_radius: 0.06,
                major_radius: radius.max(0.3),
            });
            let mat = materials.add(StandardMaterial {
                base_color: color.with_a(0.4),
                emissive: color.into(),
                alpha_mode: AlphaMode::Blend,
                ..default()
            });
            let mut entity_cmds = commands.spawn((
                PbrBundle {
                    mesh,
                    material: mat,
                    transform: Transform::from_translation(world_pos),
                    ..default()
                },
                ZoneVisual { zone_id: zone.id },
                Name::new("zone_visual"),
            ));
            if pulsing_zones.contains(&zone.id) {
                entity_cmds.insert(ZonePulseEffect { remaining: 0.2 });
            }
        }
    }
}

/// Updates zone pulse effects (brief emissive flash).
pub fn update_zone_pulse_system(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut ZonePulseEffect, &mut Transform)>,
) {
    let delta = time.delta_seconds();
    for (entity, mut pulse, mut transform) in query.iter_mut() {
        pulse.remaining -= delta;
        if pulse.remaining <= 0.0 {
            commands.entity(entity).remove::<ZonePulseEffect>();
            transform.scale = Vec3::ONE;
        } else {
            let t = pulse.remaining / 0.2;
            let scale = 1.0 + t * 0.15;
            transform.scale = Vec3::splat(scale);
        }
    }
}

/// Syncs tether visuals from SimState.tethers.
pub fn sync_tether_visuals_system(
    mut commands: Commands,
    sim_state: Res<MissionSimState>,
    mut query: Query<(Entity, &TetherVisual, &mut Transform)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sim = &sim_state.sim;
    let active_tethers: HashSet<(u32, u32)> = sim
        .tethers
        .iter()
        .map(|t| (t.source_id, t.target_id))
        .collect();

    // Despawn expired tethers.
    for (entity, tv, _) in query.iter() {
        if !active_tethers.contains(&(tv.source_id, tv.target_id)) {
            commands.entity(entity).despawn_recursive();
        }
    }

    let existing_keys: HashSet<(u32, u32)> = query
        .iter()
        .map(|(_, tv, _)| (tv.source_id, tv.target_id))
        .collect();

    for tether in &sim.tethers {
        let source_pos = sim.units.iter().find(|u| u.id == tether.source_id);
        let target_pos = sim.units.iter().find(|u| u.id == tether.target_id);
        let (Some(source), Some(target)) = (source_pos, target_pos) else {
            continue;
        };

        let from = Vec3::new(source.position.x, 0.8, source.position.y);
        let to = Vec3::new(target.position.x, 0.8, target.position.y);
        let diff = to - from;
        let distance = diff.length();
        if distance < 0.01 {
            continue;
        }
        let midpoint = (from + to) / 2.0;
        let direction = diff.normalize();
        let rotation = Quat::from_rotation_arc(Vec3::Z, direction);

        let key = (tether.source_id, tether.target_id);
        if existing_keys.contains(&key) {
            for (_, tv, mut transform) in query.iter_mut() {
                if tv.source_id == tether.source_id && tv.target_id == tether.target_id {
                    transform.translation = midpoint;
                    transform.rotation = rotation;
                    transform.scale = Vec3::new(1.0, 1.0, distance / 1.0);
                }
            }
        } else {
            let color = primary_tag_color(
                &tether.tick_effects.first().map(|e| &e.tags).cloned().unwrap_or_default(),
            );
            let mesh = meshes.add(Cuboid::new(0.03, 0.03, 1.0));
            let mat = materials.add(StandardMaterial {
                base_color: color,
                emissive: color.into(),
                ..default()
            });
            commands.spawn((
                PbrBundle {
                    mesh,
                    material: mat,
                    transform: Transform::from_translation(midpoint)
                        .with_rotation(rotation)
                        .with_scale(Vec3::new(1.0, 1.0, distance)),
                    ..default()
                },
                TetherVisual {
                    source_id: tether.source_id,
                    target_id: tether.target_id,
                },
                Name::new("tether_visual"),
            ));
        }
    }
}

/// Updates channel ring position and sinusoidal pulse. Despawns when unit stops channeling.
pub fn update_channel_ring_system(
    mut commands: Commands,
    time: Res<Time>,
    sim_state: Res<MissionSimState>,
    mut query: Query<(Entity, &mut ChannelRing, &mut Transform)>,
) {
    let delta = time.delta_seconds();
    let sim = &sim_state.sim;

    for (entity, mut ring, mut transform) in query.iter_mut() {
        let unit = sim.units.iter().find(|u| u.id == ring.unit_id);
        let still_channeling = unit.map_or(false, |u| u.channeling.is_some());

        if !still_channeling {
            commands.entity(entity).despawn_recursive();
            continue;
        }

        ring.elapsed += delta;

        if let Some(u) = unit {
            transform.translation = Vec3::new(u.position.x, 0.05, u.position.y);
        }

        let pulse = 1.0 + 0.1 * (ring.elapsed * 4.0).sin();
        transform.scale = Vec3::splat(pulse);
    }
}
