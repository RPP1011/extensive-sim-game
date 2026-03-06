use bevy::prelude::*;

use super::types::*;
use crate::mission::unit_vis::UnitVisual;

// ---------------------------------------------------------------------------
// Spawn VFX system (event-driven)
// ---------------------------------------------------------------------------

/// Reads `VfxEventQueue`, drains `pending`, and spawns the appropriate VFX
/// entities for each event.
pub fn spawn_vfx_system(
    mut queue: ResMut<VfxEventQueue>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    unit_query: Query<(Entity, &GlobalTransform), With<UnitVisual>>,
) {
    let events: Vec<VfxEvent> = queue.pending.drain(..).collect();

    for event in events {
        match event {
            VfxEvent::Damage { world_pos, amount, is_crit } => {
                let spawn_pos = world_pos + Vec3::Y * 1.5;
                let color = if is_crit {
                    Color::rgb(1.0, 1.0, 0.1)
                } else {
                    Color::rgb(1.0, 0.15, 0.15)
                };
                let mesh = meshes.add(Cuboid::new(0.18, 0.18, 0.18));
                let mat = materials.add(StandardMaterial {
                    base_color: color,
                    emissive: color.into(),
                    ..default()
                });
                commands.spawn((
                    PbrBundle { mesh, material: mat, transform: Transform::from_translation(spawn_pos), ..default() },
                    FloatingText { lifetime: 0.8, velocity: Vec3::new(0.0, 1.2, 0.0) },
                    Name::new(format!("-{}", amount)),
                ));
            }

            VfxEvent::Heal { world_pos, amount } => {
                let spawn_pos = world_pos + Vec3::Y * 1.5;
                let color = Color::rgb(0.1, 1.0, 0.3);
                let mesh = meshes.add(Cuboid::new(0.18, 0.18, 0.18));
                let mat = materials.add(StandardMaterial {
                    base_color: color, emissive: color.into(), ..default()
                });
                commands.spawn((
                    PbrBundle { mesh, material: mat, transform: Transform::from_translation(spawn_pos), ..default() },
                    FloatingText { lifetime: 0.8, velocity: Vec3::new(0.05, 1.0, 0.0) },
                    Name::new(format!("+{}", amount)),
                ));
            }

            VfxEvent::Death { world_pos } => {
                let mut best_entity: Option<Entity> = None;
                let mut best_dist_sq = f32::MAX;
                for (entity, gtransform) in unit_query.iter() {
                    let pos = gtransform.translation();
                    let dx = pos.x - world_pos.x;
                    let dz = pos.z - world_pos.z;
                    let dist_sq = dx * dx + dz * dz;
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_entity = Some(entity);
                    }
                }
                if let Some(entity) = best_entity {
                    commands.entity(entity).insert(DeathFade { remaining: DEATH_FADE_TOTAL });
                }
            }

            VfxEvent::Control { world_pos } => {
                let mesh = meshes.add(Torus { minor_radius: 0.07, major_radius: 0.6 });
                let color = Color::rgb(1.0, 0.45, 0.0);
                let mat = materials.add(StandardMaterial {
                    base_color: color, emissive: color.into(), ..default()
                });
                commands.spawn((
                    PbrBundle { mesh, material: mat, transform: Transform::from_translation(Vec3::new(world_pos.x, 0.06, world_pos.z)), ..default() },
                    FloatingText { lifetime: 0.5, velocity: Vec3::ZERO },
                    Name::new("control_ring"),
                ));
            }

            VfxEvent::ChannelStart { unit_id } => {
                let mesh = meshes.add(Torus { minor_radius: 0.04, major_radius: 0.5 });
                let color = Color::rgb(0.4, 0.7, 1.0);
                let mat = materials.add(StandardMaterial {
                    base_color: color, emissive: color.into(), ..default()
                });
                commands.spawn((
                    PbrBundle { mesh, material: mat, transform: Transform::from_translation(Vec3::new(0.0, 0.05, 0.0)), ..default() },
                    ChannelRing { unit_id, elapsed: 0.0 },
                    Name::new("channel_ring"),
                ));
            }

            VfxEvent::ChannelEnd { unit_id } => { let _ = unit_id; }

            VfxEvent::ChainFlash { from_pos, to_pos, color } => {
                spawn_beam(&mut commands, &mut meshes, &mut materials, from_pos, to_pos, color, 0.03, 0.15);
            }

            VfxEvent::Trail { from, to, color } => {
                spawn_beam(&mut commands, &mut meshes, &mut materials, from, to, color, 0.04, 0.3);
            }

            VfxEvent::ShieldFlash { world_pos, amount: _ } => {
                let mesh = meshes.add(Torus { minor_radius: 0.05, major_radius: 0.55 });
                let color = Color::rgb(0.5, 0.7, 1.0);
                let mat = materials.add(StandardMaterial {
                    base_color: color, emissive: color.into(), ..default()
                });
                commands.spawn((
                    PbrBundle { mesh, material: mat, transform: Transform::from_translation(Vec3::new(world_pos.x, 0.1, world_pos.z)), ..default() },
                    FloatingText { lifetime: 0.3, velocity: Vec3::Y * 0.5 },
                    Name::new("shield_flash"),
                ));
            }

            VfxEvent::Miss { world_pos } => {
                let spawn_pos = world_pos + Vec3::Y * 1.8;
                let color = Color::rgb(0.6, 0.6, 0.6);
                let mesh = meshes.add(Cuboid::new(0.12, 0.06, 0.12));
                let mat = materials.add(StandardMaterial {
                    base_color: color, emissive: color.into(), ..default()
                });
                commands.spawn((
                    PbrBundle { mesh, material: mat, transform: Transform::from_translation(spawn_pos), ..default() },
                    FloatingText { lifetime: 0.6, velocity: Vec3::new(0.0, 0.8, 0.0) },
                    Name::new("MISS"),
                ));
            }

            VfxEvent::Resist { world_pos } => {
                let spawn_pos = world_pos + Vec3::Y * 1.8;
                let color = Color::rgb(0.7, 0.2, 0.9);
                let mesh = meshes.add(Cuboid::new(0.12, 0.06, 0.12));
                let mat = materials.add(StandardMaterial {
                    base_color: color, emissive: color.into(), ..default()
                });
                commands.spawn((
                    PbrBundle { mesh, material: mat, transform: Transform::from_translation(spawn_pos), ..default() },
                    FloatingText { lifetime: 0.6, velocity: Vec3::new(0.0, 0.8, 0.0) },
                    Name::new("RESIST"),
                ));
            }

            VfxEvent::ZonePulse { zone_id } => { let _ = zone_id; }
        }
    }
}

/// Helper: spawn a thin beam (cuboid) between two points with a short lifetime.
fn spawn_beam(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    from: Vec3,
    to: Vec3,
    color: Color,
    thickness: f32,
    lifetime: f32,
) {
    let diff = to - from;
    let distance = diff.length();
    if distance < 0.01 {
        return;
    }
    let midpoint = (from + to) / 2.0;
    let mesh = meshes.add(Cuboid::new(thickness, thickness, distance));
    let mat = materials.add(StandardMaterial {
        base_color: color, emissive: color.into(), ..default()
    });
    let direction = diff.normalize();
    let rotation = Quat::from_rotation_arc(Vec3::Z, direction);
    commands.spawn((
        PbrBundle { mesh, material: mat, transform: Transform::from_translation(midpoint).with_rotation(rotation), ..default() },
        FloatingText { lifetime, velocity: Vec3::ZERO },
        Name::new("beam"),
    ));
}

// ---------------------------------------------------------------------------
// Update systems
// ---------------------------------------------------------------------------

/// Ticks `FloatingText` down, moves the entity, and despawns it when expired.
pub fn update_floating_text_system(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut FloatingText, &mut Transform)>,
) {
    let delta = time.delta_seconds();
    for (entity, mut ft, mut transform) in query.iter_mut() {
        ft.lifetime -= delta;
        transform.translation += ft.velocity * delta;
        if ft.lifetime <= 0.0 {
            commands.entity(entity).despawn_recursive();
        }
    }
}

/// Ticks `HitFlash` down, adjusts emissive, and removes the component when done.
pub fn update_hit_flash_system(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut HitFlash, &Handle<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let delta = time.delta_seconds();
    for (entity, mut flash, mat_handle) in query.iter_mut() {
        flash.remaining -= delta;
        if let Some(mat) = materials.get_mut(mat_handle) {
            if flash.remaining > 0.0 {
                let intensity = (flash.remaining * 8.0).min(1.0);
                mat.emissive = Color::rgb(intensity, intensity * 0.8, 0.0).into();
            } else {
                mat.emissive = Color::BLACK.into();
                commands.entity(entity).remove::<HitFlash>();
            }
        }
    }
}

/// Ticks `DeathFade` down, shrinks the entity, and despawns it when expired.
pub fn update_death_fade_system(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut DeathFade, &mut Transform)>,
) {
    let delta = time.delta_seconds();
    for (entity, mut fade, mut transform) in query.iter_mut() {
        fade.remaining -= delta;
        if fade.remaining <= 0.0 {
            commands.entity(entity).despawn_recursive();
        } else {
            let scale = (fade.remaining / DEATH_FADE_TOTAL).clamp(0.0, 1.0);
            transform.scale = Vec3::splat(scale);
        }
    }
}
