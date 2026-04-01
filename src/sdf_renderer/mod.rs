//! SDF Voxel Renderer — Bevy plugin for raymarched voxel terrain.
//!
//! Renders the voxel world by raymarching through a 3D SDF texture in a
//! custom fragment shader. No mesh generation — a single fullscreen quad
//! does all the work.
//!
//! Usage: add `SdfRendererPlugin`, insert `VoxelRenderData` with chunk SDF data.

use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, ShaderRef, ShaderType, Extent3d, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::texture::ImageSampler;
use bevy::pbr::{MaterialPlugin, MaterialMeshBundle};

/// Plugin that renders voxel terrain via SDF raymarching.
pub struct SdfRendererPlugin;

impl Plugin for SdfRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<SdfRaymarchMaterial>::default())
            .init_resource::<SdfRenderState>()
            .add_systems(Update, update_sdf_material);
    }
}

/// Resource containing voxel world data for rendering.
#[derive(Resource, Default)]
pub struct VoxelRenderData {
    /// Surface voxels: (world_x, world_y, world_z, material_id) — kept for compatibility
    pub surfaces: Vec<(f32, f32, f32, u8)>,
    /// Raw SDF distances packed into a flat array (for 3D texture upload)
    pub sdf_data: Vec<f32>,
    /// Raw material IDs packed into a flat array
    pub material_data: Vec<u8>,
    /// Volume dimensions in voxels
    pub volume_size: (u32, u32, u32),
    /// Volume origin in world space
    pub volume_origin: (f32, f32, f32),
    /// Whether data changed
    pub dirty: bool,
}

/// Internal state tracking
#[derive(Resource, Default)]
struct SdfRenderState {
    quad_entity: Option<Entity>,
    initialized: bool,
}

/// Custom material that does SDF raymarching in the fragment shader.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct SdfRaymarchMaterial {
    /// 3D SDF texture — signed distance field
    #[texture(0, dimension = "3d")]
    #[sampler(1)]
    pub sdf_texture: Handle<Image>,

    /// 3D material ID texture
    #[texture(2, dimension = "3d", sample_type = "u_int")]
    pub material_texture: Handle<Image>,

    /// Volume dimensions and origin
    #[uniform(3)]
    pub volume_info: VolumeInfo,
}

#[derive(Clone, Copy, Default, ShaderType)]
pub struct VolumeInfo {
    pub size: Vec3,
    pub _pad0: f32,
    pub origin: Vec3,
    pub _pad1: f32,
}

impl Material for SdfRaymarchMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/sdf_raymarch_bevy.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
}

fn update_sdf_material(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut sdf_materials: ResMut<Assets<SdfRaymarchMaterial>>,
    render_data: Option<ResMut<VoxelRenderData>>,
    mut render_state: ResMut<SdfRenderState>,
) {
    let Some(mut data) = render_data else { return };
    if !data.dirty { return; }
    data.dirty = false;

    let (w, h, d) = data.volume_size;
    if w == 0 || h == 0 || d == 0 { return; }
    if data.sdf_data.is_empty() || data.material_data.is_empty() { return; }

    info!("Uploading SDF volume: {}x{}x{} ({} voxels)", w, h, d, data.sdf_data.len());

    // Create 3D SDF texture (R32Float)
    let sdf_bytes: Vec<u8> = data.sdf_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut sdf_image = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: d },
        TextureDimension::D3,
        sdf_bytes,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    sdf_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    sdf_image.sampler = ImageSampler::nearest(); // manual trilinear in shader
    let sdf_handle = images.add(sdf_image);

    // Create 3D material texture (R8Uint)
    let mut mat_image = Image::new(
        Extent3d { width: w, height: h, depth_or_array_layers: d },
        TextureDimension::D3,
        data.material_data.clone(),
        TextureFormat::R8Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    mat_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    let mat_handle = images.add(mat_image);

    // Create material
    let material = sdf_materials.add(SdfRaymarchMaterial {
        sdf_texture: sdf_handle,
        material_texture: mat_handle,
        volume_info: VolumeInfo {
            size: Vec3::new(w as f32, h as f32, d as f32),
            origin: Vec3::new(data.volume_origin.0, data.volume_origin.1, data.volume_origin.2),
            ..default()
        },
    });

    // Despawn old quad
    if let Some(entity) = render_state.quad_entity {
        commands.entity(entity).despawn();
    }

    // Spawn a large quad that the camera looks at — the fragment shader does the raymarching.
    // Using a large box that fills the view. The shader ignores the mesh geometry and
    // traces rays from the camera through each fragment.
    let quad = meshes.add(Mesh::from(Cuboid::new(200.0, 200.0, 200.0)));
    let entity = commands.spawn(MaterialMeshBundle {
        mesh: quad,
        material,
        transform: Transform::from_xyz(
            data.volume_origin.0 + w as f32 * 0.5,
            data.volume_origin.2 + d as f32 * 0.5, // z→y in Bevy
            data.volume_origin.1 + h as f32 * 0.5,
        ),
        ..default()
    }).id();

    render_state.quad_entity = Some(entity);
    render_state.initialized = true;
}

/// Extract surface voxels from a VoxelWorld into VoxelRenderData.
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

/// Pack a VoxelWorld into flat arrays for 3D texture upload.
/// Returns (sdf_data, material_data, volume_size, volume_origin).
pub fn pack_voxel_world(
    voxel_world: &crate::world_sim::voxel::VoxelWorld,
) -> (Vec<f32>, Vec<u8>, (u32, u32, u32), (f32, f32, f32)) {
    use crate::world_sim::voxel::*;
    use crate::world_sim::sdf;

    if voxel_world.chunks.is_empty() {
        return (vec![0.0], vec![0], (1, 1, 1), (0.0, 0.0, 0.0));
    }

    // Find chunk bounds
    let min_cx = voxel_world.chunks.keys().map(|c| c.x).min().unwrap();
    let max_cx = voxel_world.chunks.keys().map(|c| c.x).max().unwrap();
    let min_cy = voxel_world.chunks.keys().map(|c| c.y).min().unwrap();
    let max_cy = voxel_world.chunks.keys().map(|c| c.y).max().unwrap();
    let min_cz = voxel_world.chunks.keys().map(|c| c.z).min().unwrap();
    let max_cz = voxel_world.chunks.keys().map(|c| c.z).max().unwrap();

    let nx = ((max_cx - min_cx + 1) as u32) * CHUNK_SIZE as u32;
    let ny = ((max_cy - min_cy + 1) as u32) * CHUNK_SIZE as u32;
    let nz = ((max_cz - min_cz + 1) as u32) * CHUNK_SIZE as u32;
    let total = (nx * ny * nz) as usize;

    let mut sdf_data = vec![8.0f32; total]; // default: far from surface (air)
    let mut mat_data = vec![0u8; total];

    // Pack each chunk into the flat arrays
    for (cp, chunk) in &voxel_world.chunks {
        let ox = ((cp.x - min_cx) as usize) * CHUNK_SIZE;
        let oy = ((cp.y - min_cy) as usize) * CHUNK_SIZE;
        let oz = ((cp.z - min_cz) as usize) * CHUNK_SIZE;

        // Generate SDF for this chunk
        let chunk_sdf = sdf::generate_chunk_sdf(chunk, Some(&|gx, gy, gz| {
            voxel_world.get_voxel(gx, gy, gz)
        }));

        for lz in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let flat_idx = ((oz + lz) * ny as usize + (oy + ly)) * nx as usize + (ox + lx);
                    let local_idx = local_index(lx, ly, lz);
                    sdf_data[flat_idx] = chunk_sdf.distances[local_idx];
                    mat_data[flat_idx] = chunk.voxels[local_idx].material as u8;
                }
            }
        }
    }

    let origin = (
        (min_cx * CHUNK_SIZE as i32) as f32,
        (min_cy * CHUNK_SIZE as i32) as f32,
        (min_cz * CHUNK_SIZE as i32) as f32,
    );

    (sdf_data, mat_data, (nx, ny, nz), origin)
}
