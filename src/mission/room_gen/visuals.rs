use bevy::prelude::*;

use super::lcg::Lcg;
use super::RoomLayout;

// ---------------------------------------------------------------------------
// Bevy components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct RoomFloor;

#[derive(Component)]
pub struct RoomWall;

#[derive(Component)]
pub struct RoomObstacle;

// ---------------------------------------------------------------------------
// Room visual spawning
// ---------------------------------------------------------------------------

/// Spawn all Bevy PBR meshes for a generated room.
///
/// - Floor: a single `Plane3d` slab; color varies by seed (earthy/stone/mossy/sand).
/// - Perimeter walls: four `Cuboid` meshes, slightly darker version of the floor color.
/// - Obstacle blocks: `Cuboid` meshes, contrasting (darker by 0.15, clamped).
/// - Ramp regions: inclined `Cuboid` approximations.
pub fn spawn_room(
    layout: &RoomLayout,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let w = layout.width;
    let d = layout.depth;
    let cs = layout.nav.cell_size;

    // --- Seed-derived palette ---
    let mut rng = Lcg::new(layout.seed);
    let palette: [(f32, f32, f32); 4] = [
        (0.45, 0.38, 0.28), // Earthy brown
        (0.55, 0.55, 0.58), // Stone grey
        (0.35, 0.48, 0.32), // Mossy green
        (0.68, 0.58, 0.42), // Dusty sand
    ];
    let palette_idx = (rng.next_u64() % 4) as usize;
    let (fr, fg, fb) = palette[palette_idx];

    let wall_darken = 0.12_f32;
    let (wr, wg, wb) = (
        (fr - wall_darken).clamp(0.0, 1.0),
        (fg - wall_darken).clamp(0.0, 1.0),
        (fb - wall_darken).clamp(0.0, 1.0),
    );

    let obs_darken = 0.15_f32;
    let (obsr, obsg, obsb) = (
        (fr - obs_darken).clamp(0.0, 1.0),
        (fg - obs_darken).clamp(0.0, 1.0),
        (fb - obs_darken).clamp(0.0, 1.0),
    );

    // --- Floor ---
    let floor_mesh = meshes.add(Plane3d::new(Vec3::Y));
    let floor_mat = materials.add(StandardMaterial {
        base_color: Color::rgb(fr, fg, fb),
        perceptual_roughness: 0.85,
        metallic: 0.0,
        ..default()
    });
    commands.spawn((
        PbrBundle {
            mesh: floor_mesh,
            material: floor_mat,
            transform: Transform {
                translation: Vec3::new(w / 2.0, 0.0, d / 2.0),
                scale: Vec3::new(w, 1.0, d),
                ..default()
            },
            ..default()
        },
        RoomFloor,
    ));

    // --- Perimeter walls ---
    let wall_mat = materials.add(StandardMaterial {
        base_color: Color::rgb(wr, wg, wb),
        perceptual_roughness: 0.95,
        metallic: 0.0,
        ..default()
    });
    let wall_thick = 1.0_f32;
    let wall_h = 2.0_f32;

    spawn_wall(commands, meshes, wall_mat.clone(), Vec3::new(w / 2.0, wall_h / 2.0, wall_thick / 2.0), Vec3::new(w, wall_h, wall_thick));
    spawn_wall(commands, meshes, wall_mat.clone(), Vec3::new(w / 2.0, wall_h / 2.0, d - wall_thick / 2.0), Vec3::new(w, wall_h, wall_thick));
    spawn_wall(commands, meshes, wall_mat.clone(), Vec3::new(wall_thick / 2.0, wall_h / 2.0, d / 2.0), Vec3::new(wall_thick, wall_h, d));
    spawn_wall(commands, meshes, wall_mat.clone(), Vec3::new(w - wall_thick / 2.0, wall_h / 2.0, d / 2.0), Vec3::new(wall_thick, wall_h, d));

    // --- Interior obstacle blocks ---
    for obs in &layout.obstacles {
        let x0 = obs.col0 as f32 * cs;
        let x1 = (obs.col1 + 1) as f32 * cs;
        let z0 = obs.row0 as f32 * cs;
        let z1 = (obs.row1 + 1) as f32 * cs;

        let obs_w = x1 - x0;
        let obs_d = z1 - z0;
        let obs_h = obs.height;

        let height_factor = (obs_h * 0.02).min(0.05);
        let obs_mat = materials.add(StandardMaterial {
            base_color: Color::rgb(
                (obsr - height_factor).clamp(0.0, 1.0),
                (obsg - height_factor).clamp(0.0, 1.0),
                (obsb - height_factor).clamp(0.0, 1.0),
            ),
            perceptual_roughness: 0.9,
            metallic: 0.0,
            ..default()
        });
        let obs_mesh = meshes.add(Cuboid::new(obs_w, obs_h, obs_d));
        commands.spawn((
            PbrBundle {
                mesh: obs_mesh,
                material: obs_mat,
                transform: Transform::from_translation(Vec3::new(
                    x0 + obs_w / 2.0,
                    obs_h / 2.0,
                    z0 + obs_d / 2.0,
                )),
                ..default()
            },
            RoomObstacle,
        ));
    }

    // --- Ramp regions ---
    let ramp_mat = materials.add(StandardMaterial {
        base_color: Color::rgb(wr, wg, wb),
        perceptual_roughness: 0.88,
        metallic: 0.0,
        ..default()
    });
    for ramp in &layout.ramps {
        let x0 = ramp.col0 as f32 * cs;
        let x1 = (ramp.col1 + 1) as f32 * cs;
        let z0 = ramp.row0 as f32 * cs;
        let z1 = (ramp.row1 + 1) as f32 * cs;

        let ramp_w = x1 - x0;
        let ramp_d = z1 - z0;
        let slab_h = 0.1_f32;

        let angle = (ramp.elevation / ramp_d).atan();
        let centre = Vec3::new(
            x0 + ramp_w / 2.0,
            ramp.elevation / 2.0,
            z0 + ramp_d / 2.0,
        );
        let mut transform = Transform::from_translation(centre);
        transform.rotation = Quat::from_rotation_x(angle);

        let ramp_mesh = meshes.add(Cuboid::new(ramp_w, slab_h, ramp_d));
        commands.spawn((
            PbrBundle {
                mesh: ramp_mesh,
                material: ramp_mat.clone(),
                transform,
                ..default()
            },
            RoomObstacle,
        ));
    }
}

fn spawn_wall(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    material: Handle<StandardMaterial>,
    centre: Vec3,
    size: Vec3,
) {
    let mesh = meshes.add(Cuboid::new(size.x, size.y, size.z));
    commands.spawn((
        PbrBundle {
            mesh,
            material,
            transform: Transform::from_translation(centre),
            ..default()
        },
        RoomWall,
    ));
}
