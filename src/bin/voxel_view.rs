//! Standalone voxel terrain viewer.
//!
//! Generates a voxel world, extracts surface voxels, and renders them
//! in a Bevy window with orbit camera.
//!
//! Usage: cargo run --bin voxel_view [--release]

use bevy::prelude::*;
use bevy_game::world_sim::voxel::*;
use bevy_game::sdf_renderer::{SdfRendererPlugin, VoxelRenderData, extract_surfaces};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Voxel Terrain Viewer".into(),
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

    // Load a 5x5x4 region of chunks around the center
    for cz in 0..4 {
        for cy in 0..5 {
            for cx in 0..5 {
                world.generate_chunk(ChunkPos::new(cx, cy, cz), seed);
            }
        }
    }

    let chunk_count = world.chunk_count();
    let solid_count = world.total_solid();
    info!("Generated {} chunks, {} solid voxels", chunk_count, solid_count);

    // Extract surface voxels (cap at 50K for performance)
    let surfaces = extract_surfaces(&world, 50_000);
    info!("Extracted {} surface voxels", surfaces.len());

    // Insert render data
    commands.insert_resource(VoxelRenderData {
        surfaces,
        dirty: true,
    });

    // Camera — positioned above terrain looking down
    // Terrain is at x=0..80, y=0..80, z=0..64 in voxel space
    // Bevy: x=voxel_x, y=voxel_z (up), z=voxel_y
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(40.0, 60.0, -20.0)
                .looking_at(Vec3::new(40.0, 25.0, 40.0), Vec3::Y),
            ..default()
        },
        OrbitCam {
            focus: Vec3::new(40.0, 25.0, 40.0),
            radius: 80.0,
            yaw: 0.0,
            pitch: -0.6,
        },
    ));

    // Lighting
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 15000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ, -0.8, 0.3, 0.0,
        )),
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 500000.0,
            range: 200.0,
            ..default()
        },
        transform: Transform::from_xyz(40.0, 80.0, 40.0),
        ..default()
    });

    info!("Voxel viewer ready. Drag to orbit, scroll to zoom.");
}

// Simple orbit camera
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

    // Rotate on drag
    if mouse.pressed(MouseButton::Left) {
        for ev in motion.read() {
            orbit.yaw -= ev.delta.x * 0.005;
            orbit.pitch -= ev.delta.y * 0.005;
            orbit.pitch = orbit.pitch.clamp(-1.4, -0.1);
        }
    } else {
        motion.clear();
    }

    // Zoom on scroll
    for ev in scroll.read() {
        orbit.radius -= ev.y * 5.0;
        orbit.radius = orbit.radius.clamp(10.0, 300.0);
    }

    // Update camera position from orbit params
    let x = orbit.focus.x + orbit.radius * orbit.pitch.cos() * orbit.yaw.sin();
    let y = orbit.focus.y + orbit.radius * (-orbit.pitch).sin();
    let z = orbit.focus.z + orbit.radius * orbit.pitch.cos() * orbit.yaw.cos();

    *transform = Transform::from_xyz(x, y, z).looking_at(orbit.focus, Vec3::Y);
}
