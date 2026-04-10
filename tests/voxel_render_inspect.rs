//! Multi-biome terrain render inspection.
//!
//! Generates terrain for multiple biomes, renders each from several camera
//! angles, and composites into a contact sheet for visual inspection.
//!
//! Run: cargo test --test voxel_render_inspect --features app -- --nocapture

#![cfg(feature = "app")]


use game::world_sim::terrain::{generate_continent, materialize::surface_height_at, CELL_SIZE};
use game::world_sim::voxel::{VoxelWorld, ChunkPos, CHUNK_SIZE};
use game::world_sim::voxel_bridge::VoxelBridge;
use game::world_sim::state::Terrain;

use voxel_engine::camera::FreeCamera;
use voxel_engine::render::VoxelRenderer;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;
use voxel_engine::vulkan::voxel_gpu;
use voxel_engine::voxel::grid::VoxelGrid;

const TILE_W: u32 = 400;
const TILE_H: u32 = 300;
const CHUNK_RADIUS: i32 = 2; // (2*R)^3 = 64 chunks per biome. At 64³ chunks, 2 = 256m across

fn try_vulkan() -> Option<VulkanContext> {
    VulkanContext::new().ok()
}

/// Find a cell in the plan matching the target terrain.
fn find_biome_cell(plan: &game::world_sim::terrain::region_plan::RegionPlan, target: Terrain) -> Option<(usize, usize)> {
    for row in 0..plan.rows {
        for col in 0..plan.cols {
            let cell = plan.get(col, row);
            if cell.terrain == target && cell.height > 0.2 {
                return Some((col, row));
            }
        }
    }
    None
}

/// Generate chunks around a biome cell and return the world + chunk positions + surface z.
fn generate_biome_chunks(
    plan: &game::world_sim::terrain::region_plan::RegionPlan,
    col: usize,
    row: usize,
    seed: u64,
) -> (VoxelWorld, Vec<ChunkPos>, i32) {
    let mut world = VoxelWorld::default();
    world.region_plan = Some(plan.clone());

    let center_vx = col as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 / 2.0;
    let center_vy = row as f32 * CELL_SIZE as f32 + CELL_SIZE as f32 / 2.0;
    let center_cx = (center_vx / CHUNK_SIZE as f32) as i32;
    let center_cy = (center_vy / CHUNK_SIZE as f32) as i32;
    let surface_z = surface_height_at(center_vx, center_vy, plan, seed);
    let surface_cz = surface_z / CHUNK_SIZE as i32;

    let r = CHUNK_RADIUS;
    let mut positions = Vec::new();
    // Bias downward: more underground chunks so bottom edge isn't visible
    for dz in (-r - 1)..(r) {
        for dy in -r..r {
            for dx in -r..r {
                let cp = ChunkPos::new(center_cx + dx, center_cy + dy, surface_cz + dz);
                world.generate_chunk(cp, seed);
                positions.push(cp);
            }
        }
    }

    (world, positions, surface_z)
}

/// Render a world from a given camera position.
fn render_scene(
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    renderer: &mut VoxelRenderer,
    world: &VoxelWorld,
    positions: &[ChunkPos],
    camera: &FreeCamera,
) -> Vec<[u8; 4]> {
    let bridge = VoxelBridge::new();
    let palette_rgba = bridge.palette_rgba();

    let min_x = positions.iter().map(|p| p.x).min().unwrap();
    let min_y = positions.iter().map(|p| p.y).min().unwrap();
    let min_z = positions.iter().map(|p| p.z).min().unwrap();
    let max_x = positions.iter().map(|p| p.x).max().unwrap();
    let max_y = positions.iter().map(|p| p.y).max().unwrap();
    let max_z = positions.iter().map(|p| p.z).max().unwrap();

    let grid_x = ((max_x - min_x + 1) * CHUNK_SIZE as i32) as u32;
    let grid_y = ((max_y - min_y + 1) * CHUNK_SIZE as i32) as u32;
    let grid_z = ((max_z - min_z + 1) * CHUNK_SIZE as i32) as u32;
    // Engine coords: x, z_up, y — so VoxelGrid dims are (grid_x, grid_z, grid_y)
    let mut grid = VoxelGrid::new(grid_x, grid_z, grid_y);

    for cp in positions {
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

    let gpu_tex = voxel_gpu::upload_grid_to_gpu(ctx, alloc, &grid, palette_rgba)
        .expect("GPU upload");

    let objects = vec![
        (&gpu_tex, [1.0f32, 1.0, 1.0, 1.0], [0.0f32, 0.0, 0.0],
         [grid_x as f32, grid_z as f32, grid_y as f32]),
    ];
    let pixels = renderer.render_frame(ctx, camera, &objects)
        .expect("render_frame");

    gpu_tex.destroy(ctx, alloc);
    pixels
}

struct CameraAngle {
    name: &'static str,
    /// Offset from center, as a fraction of grid_dim. (x, y_up, z)
    eye_offset: (f32, f32, f32),
    /// Look-at offset from center.
    look_offset: (f32, f32, f32),
}

const CAMERAS: &[CameraAngle] = &[
    CameraAngle {
        name: "overhead",
        eye_offset: (0.0, 1.8, 0.0),
        look_offset: (0.0, -0.5, 0.0),
    },
    CameraAngle {
        name: "north45",
        eye_offset: (0.0, 1.0, -1.0),
        look_offset: (0.0, -0.2, 0.3),
    },
    CameraAngle {
        name: "south45",
        eye_offset: (0.0, 1.0, 1.0),
        look_offset: (0.0, -0.2, -0.3),
    },
    CameraAngle {
        name: "ground",
        eye_offset: (-0.35, 0.35, -0.35),
        look_offset: (0.2, -0.05, 0.2),
    },
    CameraAngle {
        name: "closeup",
        eye_offset: (-0.15, 0.35, -0.15),
        look_offset: (0.0, -0.1, 0.0),
    },
];

fn save_png(pixels: &[[u8; 4]], w: u32, h: u32, path: &str) {
    let mut img = image::RgbaImage::new(w, h);
    for (i, pixel) in pixels.iter().enumerate() {
        let x = (i % w as usize) as u32;
        let y = (i / w as usize) as u32;
        img.put_pixel(x, y, image::Rgba(*pixel));
    }
    img.save(path).expect("save PNG");
}

#[test]
fn terrain_gallery() {
    let ctx = match try_vulkan() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no Vulkan GPU available");
            return;
        }
    };

    let mut alloc = VulkanAllocator::new(&ctx).expect("allocator");
    let mut renderer = VoxelRenderer::new(&ctx, TILE_W, TILE_H).expect("renderer");

    let seed = 42u64;
    // Use a large plan to get good biome variety. Bigger grids have more extreme
    // elevation peaks (continent mask is gentler relative to center), producing
    // rarer biomes like Mountains and Tundra.
    let plan = generate_continent(100, 80, seed);

    let biomes = [
        Terrain::Forest,
        Terrain::Desert,
        Terrain::Mountains,
        Terrain::Plains,
        Terrain::Tundra,
        Terrain::Jungle,
        Terrain::Swamp,
        Terrain::Badlands,
    ];

    let mut found_biomes: Vec<(Terrain, usize, usize)> = Vec::new();
    for &biome in &biomes {
        if let Some((col, row)) = find_biome_cell(&plan, biome) {
            found_biomes.push((biome, col, row));
        } else {
            eprintln!("WARN: biome {:?} not found in plan, skipping", biome);
        }
    }

    let num_biomes = found_biomes.len();
    let num_cameras = CAMERAS.len();
    eprintln!("Rendering {} biomes × {} cameras = {} tiles",
        num_biomes, num_cameras, num_biomes * num_cameras);

    std::fs::create_dir_all("generated/render_tests").ok();

    // Render each biome from each camera
    let mut tiles: Vec<Vec<[u8; 4]>> = Vec::new(); // row-major: biome × camera
    let mut labels: Vec<String> = Vec::new();

    for (biome, col, row) in &found_biomes {
        let (world, positions, surface_z) = generate_biome_chunks(&plan, *col, *row, seed);
        let grid_xy = (CHUNK_RADIUS * 2) as f32 * CHUNK_SIZE as f32;
        // Z range is biased: (-r-3)..(r-1) = r*2+2 chunks
        let grid_z = (CHUNK_RADIUS * 2 + 2) as f32 * CHUNK_SIZE as f32;
        // Camera uses horizontal extent for XZ offsets, vertical for Y
        let center_xz = grid_xy / 2.0;
        // Surface sits near the top of the grid (only r-1 chunks above surface)
        let surface_y = grid_z - (CHUNK_RADIUS - 1) as f32 * CHUNK_SIZE as f32;

        eprintln!("  {:?}: col={} row={} surface_z={} chunks={}",
            biome, col, row, surface_z, positions.len());

        for cam in CAMERAS {
            let eye = glam::Vec3::new(
                center_xz + cam.eye_offset.0 * grid_xy,
                surface_y + cam.eye_offset.1 * grid_xy,
                center_xz + cam.eye_offset.2 * grid_xy,
            );
            let look_at = glam::Vec3::new(
                center_xz + cam.look_offset.0 * grid_xy,
                surface_y + cam.look_offset.1 * grid_xy,
                center_xz + cam.look_offset.2 * grid_xy,
            );

            let camera = FreeCamera::new(eye, look_at);
            let pixels = render_scene(&ctx, &mut alloc, &mut renderer, &world, &positions, &camera);

            // Save individual tile
            let filename = format!("generated/render_tests/{:?}_{}.png", biome, cam.name);
            save_png(&pixels, TILE_W, TILE_H, &filename);

            tiles.push(pixels);
            labels.push(format!("{:?} {}", biome, cam.name));
        }
    }

    // Composite into contact sheet
    let sheet_cols = num_cameras as u32;
    let sheet_rows = num_biomes as u32;
    let sheet_w = sheet_cols * TILE_W;
    let sheet_h = sheet_rows * TILE_H;
    let mut sheet = image::RgbaImage::new(sheet_w, sheet_h);

    for (idx, tile_pixels) in tiles.iter().enumerate() {
        let biome_idx = idx / num_cameras;
        let cam_idx = idx % num_cameras;
        let offset_x = cam_idx as u32 * TILE_W;
        let offset_y = biome_idx as u32 * TILE_H;

        for (i, pixel) in tile_pixels.iter().enumerate() {
            let lx = (i % TILE_W as usize) as u32;
            let ly = (i / TILE_W as usize) as u32;
            if offset_x + lx < sheet_w && offset_y + ly < sheet_h {
                sheet.put_pixel(offset_x + lx, offset_y + ly, image::Rgba(*pixel));
            }
        }
    }

    let sheet_path = "generated/render_tests/gallery.png";
    sheet.save(sheet_path).expect("save contact sheet");
    eprintln!("Saved {}x{} contact sheet to {}", sheet_w, sheet_h, sheet_path);
    eprintln!("Columns: {:?}", CAMERAS.iter().map(|c| c.name).collect::<Vec<_>>());
    eprintln!("Rows: {:?}", found_biomes.iter().map(|(b, _, _)| format!("{:?}", b)).collect::<Vec<_>>());

    // Cleanup
    renderer.destroy(&ctx);
}
