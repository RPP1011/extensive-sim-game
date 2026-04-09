//! Headless GPU render test for terrain generation.
//!
//! Generates terrain, uploads to GPU, renders a frame, and asserts on pixels.
//! Requires a GPU (Vulkan). Skips gracefully if no GPU available.
//!
//! Run: cargo test --test terrain_render --features app -- --nocapture

#![cfg(feature = "app")]

use std::collections::HashMap;

use game::world_sim::terrain::{generate_continent, CELL_SIZE, MAX_SURFACE_Z};
use game::world_sim::voxel::{VoxelWorld, ChunkPos, CHUNK_SIZE};
use game::world_sim::voxel_bridge::VoxelBridge;

use voxel_engine::camera::FreeCamera;
use voxel_engine::render::VoxelRenderer;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::voxel_gpu;
use voxel_engine::voxel::grid::VoxelGrid;

const RENDER_W: u32 = 320;
const RENDER_H: u32 = 240;

/// Try to create a headless Vulkan context. Returns None if no GPU.
fn try_vulkan() -> Option<VulkanContext> {
    VulkanContext::new().ok()
}

/// Generate terrain chunks around a point and build mega-grids for rendering.
fn generate_test_world() -> (VoxelWorld, Vec<ChunkPos>) {
    let plan = generate_continent(10, 10, 42);
    let mut world = VoxelWorld::default();
    world.region_plan = Some(plan);

    // Find a land cell to generate chunks around
    let plan_ref = world.region_plan.as_ref().unwrap();
    let mut land_col = 5;
    let mut land_row = 5;
    for row in 0..plan_ref.rows {
        for col in 0..plan_ref.cols {
            let cell = plan_ref.get(col, row);
            if cell.height > 0.2 {
                land_col = col;
                land_row = row;
                break;
            }
        }
    }

    // Generate a 6x6x6 block of chunks around the land cell
    let center_cx = (land_col as i32 * CELL_SIZE / CHUNK_SIZE as i32) + 2;
    let center_cy = (land_row as i32 * CELL_SIZE / CHUNK_SIZE as i32) + 2;
    let mut positions = Vec::new();

    // Find approximate surface z
    let height = plan_ref.interpolate_height(
        center_cx as f32 * CHUNK_SIZE as f32,
        center_cy as f32 * CHUNK_SIZE as f32,
    );
    let surface_chunk_z = (height * MAX_SURFACE_Z as f32) as i32 / CHUNK_SIZE as i32;

    for dz in -2..=3 {
        for dy in -2..=3 {
            for dx in -2..=3 {
                let cp = ChunkPos::new(center_cx + dx, center_cy + dy, surface_chunk_z + dz);
                world.generate_chunk(cp, 42);
                positions.push(cp);
            }
        }
    }

    (world, positions)
}

#[test]
fn terrain_renders_not_empty() {
    let ctx = match try_vulkan() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no Vulkan GPU available");
            return;
        }
    };

    let mut alloc = VulkanAllocator::new(&ctx).expect("allocator");
    let mut renderer = VoxelRenderer::new(&ctx, RENDER_W, RENDER_H).expect("renderer");

    // Generate terrain
    let (world, positions) = generate_test_world();
    let bridge = VoxelBridge::new();
    let palette_rgba = bridge.palette_rgba();

    // Build a single mega-grid from all chunks (simple: just pack them into a 96³ grid)
    let grid_size = 6 * CHUNK_SIZE as u32; // 96
    let mut grid = VoxelGrid::new(grid_size, grid_size, grid_size);

    let min_x = positions.iter().map(|p| p.x).min().unwrap();
    let min_y = positions.iter().map(|p| p.y).min().unwrap();
    let min_z = positions.iter().map(|p| p.z).min().unwrap();

    let mut solid_voxels = 0u64;
    for cp in &positions {
        if let Some(chunk) = world.chunks.get(cp) {
            let base_x = ((cp.x - min_x) * CHUNK_SIZE as i32) as u32;
            let base_y = ((cp.y - min_y) * CHUNK_SIZE as i32) as u32;
            let base_z = ((cp.z - min_z) * CHUNK_SIZE as i32) as u32;

            for lz in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let voxel = chunk.get(lx, ly, lz);
                        let mat = voxel.material as u8;
                        if mat != 0 {
                            solid_voxels += 1;
                            // Swap Y↔Z for engine coords
                            grid.set(
                                base_x + lx as u32,
                                base_z + lz as u32,
                                base_y + ly as u32,
                                mat,
                            );
                        }
                    }
                }
            }
        }
    }

    eprintln!("Generated {} solid voxels in {} chunks", solid_voxels, positions.len());
    assert!(solid_voxels > 0, "no solid voxels generated — terrain gen broken");

    // Upload to GPU
    let gpu_tex = voxel_gpu::upload_grid_to_gpu(&ctx, &mut alloc, &grid, palette_rgba)
        .expect("GPU upload");

    // Position camera looking down at the terrain
    let center = grid_size as f32 / 2.0;
    let camera = FreeCamera::new(
        glam::Vec3::new(center, center + 60.0, center - 40.0),
        glam::Vec3::new(center, 0.0, center),
    );

    // Render frame (CPU readback)
    let objects = vec![
        (&gpu_tex, [1.0f32, 1.0, 1.0, 1.0], [0.0f32, 0.0, 0.0], [grid_size as f32, grid_size as f32, grid_size as f32]),
    ];
    let pixels = renderer.render_frame(&ctx, &camera, &objects)
        .expect("render_frame");

    // Assert: frame is not all black
    let total_pixels = pixels.len();
    let black_pixels = pixels.iter().filter(|p| p[0] == 0 && p[1] == 0 && p[2] == 0).count();
    let black_pct = black_pixels as f32 / total_pixels as f32;
    eprintln!("Rendered {}x{} = {} pixels, {:.1}% black",
        RENDER_W, RENDER_H, total_pixels, black_pct * 100.0);
    assert!(black_pct < 0.99, "frame is all black — nothing rendered");

    // Assert: frame has color variation (not a solid fill)
    let mut color_set = std::collections::HashSet::new();
    for p in &pixels {
        color_set.insert((p[0] / 16, p[1] / 16, p[2] / 16)); // quantize to reduce noise
    }
    eprintln!("Unique color buckets: {}", color_set.len());
    assert!(color_set.len() > 3, "frame has no color variation — terrain materials not rendering");

    // Assert: frame contains terrain-like colors (green/brown/gray, not just sky blue)
    let has_green = pixels.iter().any(|p| p[1] as i32 > p[0] as i32 + 20 && p[1] as i32 > p[2] as i32 + 20);
    let has_brown = pixels.iter().any(|p| p[0] > 80 && p[1] > 60 && (p[0] as i32 - p[2] as i32) > 20);
    let has_gray = pixels.iter().any(|p| {
        let avg = (p[0] as i32 + p[1] as i32 + p[2] as i32) / 3;
        (p[0] as i32 - avg).abs() < 15
            && (p[1] as i32 - avg).abs() < 15
            && avg > 50 && avg < 200
    });
    eprintln!("Has green: {}, brown: {}, gray: {}", has_green, has_brown, has_gray);
    // At least one terrain color should be present
    assert!(
        has_green || has_brown || has_gray,
        "frame has no terrain-like colors — only sky?"
    );

    // Save the frame as PNG for manual inspection
    let mut img = image::RgbaImage::new(RENDER_W, RENDER_H);
    for (i, pixel) in pixels.iter().enumerate() {
        let x = (i % RENDER_W as usize) as u32;
        let y = (i / RENDER_W as usize) as u32;
        img.put_pixel(x, y, image::Rgba(*pixel));
    }
    let out_path = "generated/terrain_render_test.png";
    std::fs::create_dir_all("generated").ok();
    img.save(out_path).expect("save PNG");
    eprintln!("Saved test render to {}", out_path);

    // Cleanup
    gpu_tex.destroy(&ctx, &mut alloc);
    renderer.destroy(&ctx);
}

#[test]
fn terrain_has_biome_variation() {
    // Generate two different biome regions and verify they produce different voxel distributions
    let plan = generate_continent(20, 20, 42);

    // Find two cells with different terrain types
    let mut plains_cell = None;
    let mut non_plains_cell = None;
    for row in 0..plan.rows {
        for col in 0..plan.cols {
            let cell = plan.get(col, row);
            if cell.height < 0.15 { continue; } // skip ocean
            match cell.terrain {
                game::world_sim::state::Terrain::Plains if plains_cell.is_none() => {
                    plains_cell = Some((col, row));
                }
                game::world_sim::state::Terrain::Forest
                | game::world_sim::state::Terrain::Desert
                | game::world_sim::state::Terrain::Mountains
                | game::world_sim::state::Terrain::Tundra
                    if non_plains_cell.is_none() =>
                {
                    non_plains_cell = Some((col, row, cell.terrain));
                }
                _ => {}
            }
        }
    }

    let (pc, pr) = plains_cell.expect("no plains cell found");
    let (nc, nr, nt) = non_plains_cell.expect("no non-plains cell found");
    eprintln!("Plains at ({}, {}), {:?} at ({}, {})", pc, pr, nt, nc, nr);

    // Generate a chunk in each biome and compare material distributions
    let mut world = VoxelWorld::default();
    world.region_plan = Some(plan);

    let plains_cx = (pc as i32 * CELL_SIZE / CHUNK_SIZE as i32) + 1;
    let plains_cy = (pr as i32 * CELL_SIZE / CHUNK_SIZE as i32) + 1;
    let other_cx = (nc as i32 * CELL_SIZE / CHUNK_SIZE as i32) + 1;
    let other_cy = (nr as i32 * CELL_SIZE / CHUNK_SIZE as i32) + 1;

    // Generate at a surface-ish Z level
    let surface_z = 3; // low, should intersect surface for most biomes

    world.generate_chunk(ChunkPos::new(plains_cx, plains_cy, surface_z), 42);
    world.generate_chunk(ChunkPos::new(other_cx, other_cy, surface_z), 42);

    let count_materials = |cp: ChunkPos| -> HashMap<u8, usize> {
        let mut counts = HashMap::new();
        if let Some(chunk) = world.chunks.get(&cp) {
            for v in &chunk.voxels {
                *counts.entry(v.material as u8).or_insert(0) += 1;
            }
        }
        counts
    };

    let plains_mats = count_materials(ChunkPos::new(plains_cx, plains_cy, surface_z));
    let other_mats = count_materials(ChunkPos::new(other_cx, other_cy, surface_z));

    eprintln!("Plains materials: {:?}", plains_mats);
    eprintln!("{:?} materials: {:?}", nt, other_mats);

    // They should differ in material composition
    assert_ne!(plains_mats, other_mats, "two different biomes produced identical material distributions");
}
