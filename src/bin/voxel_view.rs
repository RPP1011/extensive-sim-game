//! Standalone voxel terrain viewer with SDF raymarching.
//!
//! Generates a voxel world, computes SDF per chunk, packs into 3D textures,
//! and renders via a raymarching fragment shader on a volume-enclosing box.
//!
//! Usage: cargo run --release --bin voxel_view

use bevy::prelude::*;
use bevy_game::world_sim::voxel::*;
use bevy_game::sdf_renderer::{SdfRendererPlugin, VoxelRenderData, pack_voxel_world};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Voxel SDF Viewer".into(),
                resolution: (1280., 720.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(SdfRendererPlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, orbit_camera)
        .run();
}

fn setup_scene(
    mut commands: Commands,
) {
    // Generate voxel terrain
    let mut world = VoxelWorld::default();
    let seed = 42u64;

    for cz in 0..4 {
        for cy in 0..5 {
            for cx in 0..5 {
                world.generate_chunk(ChunkPos::new(cx, cy, cz), seed);
            }
        }
    }

    info!("Generated {} chunks, {} solid voxels", world.chunk_count(), world.total_solid());

    // Pack into flat arrays for GPU upload
    let (sdf_data, material_data, volume_size, volume_origin) = pack_voxel_world(&world);
    info!("Volume: {}x{}x{}, origin: ({},{},{})",
        volume_size.0, volume_size.1, volume_size.2,
        volume_origin.0, volume_origin.1, volume_origin.2);
    info!("SDF data: {} values, material data: {} values", sdf_data.len(), material_data.len());

    // Insert render data
    commands.insert_resource(VoxelRenderData {
        surfaces: Vec::new(),
        sdf_data,
        material_data,
        volume_size,
        volume_origin,
        dirty: true,
    });

    // Camera
    let center = Vec3::new(
        volume_origin.0 + volume_size.0 as f32 * 0.5,
        volume_origin.2 + volume_size.2 as f32 * 0.8, // above terrain (z→y in Bevy)
        volume_origin.1 + volume_size.1 as f32 * 0.5,
    );

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(center.x - 40.0, center.y + 30.0, center.z - 40.0)
                .looking_at(center, Vec3::Y),
            ..default()
        },
        OrbitCam {
            focus: center,
            radius: 80.0,
            yaw: -0.8,
            pitch: -0.5,
        },
    ));

    // Lighting
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 15000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.3, 0.0)),
        ..default()
    });
}

#[derive(Component)]
struct OrbitCam {
    focus: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
}

fn orbit_camera(
    mut query: Query<(&mut Transform, &mut OrbitCam)>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut scroll: EventReader<bevy::input::mouse::MouseWheel>,
) {
    let Ok((mut transform, mut orbit)) = query.get_single_mut() else { return };

    if mouse.pressed(MouseButton::Left) {
        for ev in motion.read() {
            orbit.yaw -= ev.delta.x * 0.005;
            orbit.pitch -= ev.delta.y * 0.005;
            orbit.pitch = orbit.pitch.clamp(-1.4, -0.1);
        }
    } else {
        motion.clear();
    }

    for ev in scroll.read() {
        orbit.radius -= ev.y * 5.0;
        orbit.radius = orbit.radius.clamp(10.0, 300.0);
    }

    let x = orbit.focus.x + orbit.radius * orbit.pitch.cos() * orbit.yaw.sin();
    let y = orbit.focus.y + orbit.radius * (-orbit.pitch).sin();
    let z = orbit.focus.z + orbit.radius * orbit.pitch.cos() * orbit.yaw.cos();

    *transform = Transform::from_xyz(x, y, z).looking_at(orbit.focus, Vec3::Y);
}
