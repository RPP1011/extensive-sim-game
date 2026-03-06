use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;

use crate::game_core;
use crate::hub_types::OverworldTerrainVisual;

pub fn sample_value_noise(seed: u64, x: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = x - x.floor();
    let tz = z - z.floor();
    let hash = |ix: i32, iz: i32| -> f32 {
        let mut v = seed
            ^ ((ix as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ ((iz as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        v ^= v >> 30;
        v = v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        v ^= v >> 27;
        v = v.wrapping_mul(0x94D0_49BB_1331_11EB);
        v ^= v >> 31;
        ((v >> 11) as f32) / ((1u64 << 53) as f32)
    };
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let sx = smooth(tx);
    let sz = smooth(tz);
    let v00 = hash(x0, z0);
    let v10 = hash(x0 + 1, z0);
    let v01 = hash(x0, z0 + 1);
    let v11 = hash(x0 + 1, z0 + 1);
    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sz
}

pub fn sample_overworld_height(seed: u64, x: f32, z: f32) -> f32 {
    let mut amplitude = 1.0;
    let mut frequency = 0.7;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for octave in 0..5 {
        let n = sample_value_noise(
            seed ^ ((octave as u64).wrapping_mul(0xD6E8_FD9A_5A11_6D23)),
            x * frequency,
            z * frequency,
        );
        sum += (n * 2.0 - 1.0) * amplitude;
        norm += amplitude;
        amplitude *= 0.52;
        frequency *= 2.05;
    }
    let base = if norm > 0.0 { sum / norm } else { 0.0 };
    let ridge = (sample_value_noise(seed ^ 0xA53B_9F21, x * 0.28, z * 0.28) * 2.0 - 1.0).abs();
    let basin = sample_value_noise(seed ^ 0x5F4C_3AA1, x * 0.12, z * 0.12);
    base * 6.8 + ridge * 1.9 - basin * 1.2
}

pub fn build_overworld_terrain_mesh(seed: u64, subdivisions: usize, half_extent: f32) -> Mesh {
    let vert_side = subdivisions + 1;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vert_side * vert_side);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(vert_side * vert_side);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(vert_side * vert_side);
    let mut heights = vec![0.0_f32; vert_side * vert_side];
    let idx = |x: usize, z: usize| z * vert_side + x;

    for z in 0..=subdivisions {
        for x in 0..=subdivisions {
            let tx = x as f32 / subdivisions as f32;
            let tz = z as f32 / subdivisions as f32;
            let px = (tx * 2.0 - 1.0) * half_extent;
            let pz = (tz * 2.0 - 1.0) * half_extent;
            let h = sample_overworld_height(seed, px * 0.16, pz * 0.16);
            heights[idx(x, z)] = h;
            positions.push([px, h, pz]);
            uvs.push([tx, tz]);
        }
    }

    for z in 0..=subdivisions {
        for x in 0..=subdivisions {
            let x_l = x.saturating_sub(1);
            let x_r = (x + 1).min(subdivisions);
            let z_d = z.saturating_sub(1);
            let z_u = (z + 1).min(subdivisions);
            let h_l = heights[idx(x_l, z)];
            let h_r = heights[idx(x_r, z)];
            let h_d = heights[idx(x, z_d)];
            let h_u = heights[idx(x, z_u)];
            let n = Vec3::new(h_l - h_r, 2.4, h_d - h_u).normalize_or_zero();
            normals.push([n.x, n.y, n.z]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(subdivisions * subdivisions * 6);
    for z in 0..subdivisions {
        for x in 0..subdivisions {
            let i0 = idx(x, z) as u32;
            let i1 = idx(x + 1, z) as u32;
            let i2 = idx(x, z + 1) as u32;
            let i3 = idx(x + 1, z + 1) as u32;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

pub fn setup_overworld_terrain_scene(
    mut commands: Commands,
    overworld: Res<game_core::OverworldMap>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let terrain_mesh = meshes.add(build_overworld_terrain_mesh(overworld.map_seed, 128, 52.0));
    let terrain_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.22, 0.29, 0.23),
        perceptual_roughness: 0.96,
        metallic: 0.02,
        ..default()
    });
    let water_mesh = meshes.add(Mesh::from(bevy::math::primitives::Plane3d::default()));
    let water_material = materials.add(StandardMaterial {
        base_color: Color::rgba(0.06, 0.16, 0.2, 0.82),
        perceptual_roughness: 0.22,
        metallic: 0.0,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: terrain_mesh,
            material: terrain_material,
            visibility: Visibility::Hidden,
            transform: Transform::from_xyz(0.0, -2.2, 0.0),
            ..default()
        },
        OverworldTerrainVisual,
    ));
    commands.spawn((
        PbrBundle {
            mesh: water_mesh,
            material: water_material,
            visibility: Visibility::Hidden,
            transform: Transform::from_xyz(0.0, -1.3, 0.0).with_scale(Vec3::new(140.0, 1.0, 140.0)),
            ..default()
        },
        OverworldTerrainVisual,
    ));
}
