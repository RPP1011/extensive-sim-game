//! Voxel terrain renderer — Bevy plugin for visualizing the voxel world.
//!
//! Generates instanced cube meshes for surface voxels (solid adjacent to air),
//! colored by material. Uses Bevy's standard PBR pipeline.
//!
//! Usage: add `SdfRendererPlugin` to the Bevy app, then insert a
//! `VoxelRenderData` resource containing the voxel world to render.

use bevy::prelude::*;

/// Plugin that renders voxel terrain as instanced cubes.
pub struct SdfRendererPlugin;

impl Plugin for SdfRendererPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VoxelRenderState>()
            .add_systems(Update, update_voxel_meshes);
    }
}

/// Resource containing the voxel world data to render.
/// Set this from outside (e.g., from the world sim).
#[derive(Resource, Default)]
pub struct VoxelRenderData {
    /// Surface voxels: (world_x, world_y, world_z, material_id)
    pub surfaces: Vec<(f32, f32, f32, u8)>,
    /// Whether the data changed and meshes need rebuilding.
    pub dirty: bool,
}

/// Internal render state.
#[derive(Resource, Default)]
struct VoxelRenderState {
    /// Entity holding all voxel mesh children.
    root: Option<Entity>,
    /// Last surface count (to detect changes).
    last_count: usize,
}

/// Marker for the voxel root entity.
#[derive(Component)]
struct VoxelRoot;

/// Marker for voxel mesh children.
#[derive(Component)]
struct VoxelMesh;

fn update_voxel_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    render_data: Option<ResMut<VoxelRenderData>>,
    mut render_state: ResMut<VoxelRenderState>,
    root_query: Query<Entity, With<VoxelRoot>>,
    mesh_query: Query<Entity, With<VoxelMesh>>,
) {
    let Some(mut data) = render_data else { return };
    if !data.dirty && data.surfaces.len() == render_state.last_count { return; }
    data.dirty = false;
    render_state.last_count = data.surfaces.len();

    // Despawn old meshes
    for entity in mesh_query.iter() {
        commands.entity(entity).despawn();
    }

    // Get or create root
    let _root = if let Ok(e) = root_query.get_single() {
        e
    } else {
        commands.spawn((VoxelRoot, SpatialBundle::default())).id()
    };

    if data.surfaces.is_empty() { return; }

    // Group surfaces by material for instanced rendering
    let mut groups: std::collections::HashMap<u8, Vec<(f32, f32, f32)>> = std::collections::HashMap::new();
    for &(x, y, z, mat) in &data.surfaces {
        groups.entry(mat).or_default().push((x, y, z));
    }

    let cube = meshes.add(Mesh::from(Cuboid::new(0.95, 0.95, 0.95)));

    for (mat_id, positions) in &groups {
        let color = material_color(*mat_id);
        let mat = materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.85,
            ..default()
        });

        // Spawn each voxel as a child of root
        // For large counts, batched InstancedMesh would be better,
        // but individual PbrBundles work for < 10K voxels.
        for &(x, y, z) in positions {
            commands.spawn((
                PbrBundle {
                    mesh: cube.clone(),
                    material: mat.clone(),
                    // Bevy uses Y-up: voxel x→bevy x, voxel z→bevy y, voxel y→bevy z
                    transform: Transform::from_xyz(x, z, y),
                    ..default()
                },
                VoxelMesh,
            ));
        }
    }

    info!("Voxel renderer: {} surfaces, {} materials", data.surfaces.len(), groups.len());
}

/// Map VoxelMaterial u8 to Bevy Color.
fn material_color(mat_id: u8) -> Color {
    match mat_id {
        1 => Color::rgb(0.45, 0.32, 0.18),   // Dirt
        2 => Color::rgb(0.50, 0.50, 0.50),   // Stone
        3 => Color::rgb(0.35, 0.33, 0.32),   // Granite
        4 => Color::rgb(0.85, 0.78, 0.55),   // Sand
        5 => Color::rgb(0.60, 0.45, 0.30),   // Clay
        6 => Color::rgb(0.55, 0.52, 0.48),   // Gravel
        7 => Color::rgb(0.25, 0.55, 0.18),   // Grass
        8 => Color::rgba(0.15, 0.35, 0.65, 0.6), // Water
        9 => Color::rgb(0.90, 0.30, 0.05),   // Lava
        10 => Color::rgb(0.70, 0.85, 0.95),  // Ice
        11 => Color::rgb(0.90, 0.92, 0.95),  // Snow
        12 => Color::rgb(0.55, 0.40, 0.30),  // IronOre
        13 => Color::rgb(0.60, 0.45, 0.25),  // CopperOre
        14 => Color::rgb(0.80, 0.70, 0.20),  // GoldOre
        15 => Color::rgb(0.15, 0.15, 0.15),  // Coal
        16 => Color::rgb(0.60, 0.30, 0.80),  // Crystal
        17 => Color::rgb(0.50, 0.35, 0.20),  // WoodLog
        18 => Color::rgb(0.65, 0.50, 0.30),  // WoodPlanks
        19 => Color::rgb(0.60, 0.58, 0.55),  // StoneBlock
        20 => Color::rgb(0.55, 0.53, 0.50),  // StoneBrick
        21 => Color::rgb(0.70, 0.60, 0.35),  // Thatch
        22 => Color::rgb(0.50, 0.50, 0.55),  // Iron
        23 => Color::rgba(0.80, 0.85, 0.90, 0.3), // Glass
        24 => Color::rgb(0.35, 0.25, 0.12),  // Farmland
        25 => Color::rgb(0.20, 0.60, 0.10),  // Crop
        _ => Color::rgb(0.80, 0.20, 0.80),   // Unknown = magenta
    }
}

/// Extract surface voxels from a VoxelWorld into VoxelRenderData.
/// Only includes solid voxels adjacent to air (visible surfaces).
pub fn extract_surfaces(
    voxel_world: &crate::world_sim::voxel::VoxelWorld,
    max_surfaces: usize,
) -> Vec<(f32, f32, f32, u8)> {
    use crate::world_sim::voxel::*;

    let mut surfaces = Vec::new();

    for chunk in voxel_world.chunks.values() {
        if surfaces.len() >= max_surfaces { break; }

        let base_x = chunk.pos.x * CHUNK_SIZE as i32;
        let base_y = chunk.pos.y * CHUNK_SIZE as i32;
        let base_z = chunk.pos.z * CHUNK_SIZE as i32;

        for lz in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    if surfaces.len() >= max_surfaces { break; }

                    let voxel = chunk.get(lx, ly, lz);
                    if !voxel.material.is_solid() { continue; }

                    let gx = base_x + lx as i32;
                    let gy = base_y + ly as i32;
                    let gz = base_z + lz as i32;

                    let is_surface = [(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
                        .iter()
                        .any(|&(dx, dy, dz)| {
                            !voxel_world.get_voxel(gx + dx, gy + dy, gz + dz).material.is_solid()
                        });

                    if !is_surface { continue; }

                    let (wx, wy, wz) = voxel_to_world(gx, gy, gz);
                    surfaces.push((wx, wy, wz, voxel.material as u8));
                }
            }
        }
    }

    surfaces
}
