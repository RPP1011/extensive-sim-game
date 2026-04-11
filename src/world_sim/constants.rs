//! Spatial constants for the world sim voxel system.
//!
//! All scale-dependent values live here so a voxel size change is a single edit.
//! Current scale: 1 voxel ≈ 2.5cm, 1 NPC ≈ 80 voxels tall (~2m).

// ---------------------------------------------------------------------------
// Voxel grid
// ---------------------------------------------------------------------------

/// Voxels per chunk edge. Chunks are cubic: CHUNK_SIZE³ voxels.
/// At 10cm/voxel, 64³ = 6.4m per chunk edge.
pub const CHUNK_SIZE: usize = 64;

/// Total voxels per chunk.
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// World units per voxel. At 0.10 (10cm), an 18-voxel NPC is 1.8m tall.
/// Targets Trove/Teardown chunky voxel art style.
pub const VOXEL_SCALE: f32 = 0.10;

// ---------------------------------------------------------------------------
// Terrain generation
// ---------------------------------------------------------------------------

/// Voxels per region plan cell (horizontal). One plan cell = one biome region.
pub const CELL_SIZE: i32 = 4096;

/// Maximum surface height in voxels. Mountains peak near this value.
/// At 10cm/voxel, 2000 = 200m elevation range.
pub const MAX_SURFACE_Z: i32 = 2000;

/// Sea level in voxels. Water fills below this in ocean/coast/swamp biomes.
/// At 10cm/voxel, 350 = 35m. Same ~17% ratio of MAX_SURFACE_Z as before.
pub const SEA_LEVEL: i32 = 350;

/// Base Z level where flying islands start generating.
/// At 10cm/voxel, 1600 = 160m.
pub const SKY_BASE_Z: i32 = 1600;

// ---------------------------------------------------------------------------
// Sectors (3D spatial partitioning)
// ---------------------------------------------------------------------------

/// Voxels per sector edge. Sectors group chunks for LOD and activation.
pub const SECTOR_SIZE: i32 = 4096;

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// How many sim chunks per mega-chunk axis. MEGA × CHUNK_SIZE = voxels per mega edge.
/// At 64 chunk size, MEGA=1 means a mega-chunk IS a chunk (64 voxels = 6.4m).
/// Larger CHUNK_SIZE makes mega-chunk grouping unnecessary.
pub const MEGA: i32 = 1;

/// Voxels per mega-chunk edge.
pub const MEGA_VOXELS: u32 = (MEGA as u32) * (CHUNK_SIZE as u32);

/// Maximum distance (voxels) from camera to load/render a mega-chunk.
/// At 10cm/voxel, 768 voxels ≈ 76.8m render distance.
/// Kept short because each chunk is now 64³ = 262K voxels (large upload).
pub const LOAD_RADIUS: f32 = 768.0;

/// Internal render resolution for the voxel raycaster.
/// Kept lower than window size for performance; blit-upscaled to the window.
/// At 480x270 (16:9), the window gets a 2.67x integer-ish upscale which is
/// still crisp but the GPU does ~44% less work per frame vs 640x360.
pub const RENDER_WIDTH: u32 = 480;
pub const RENDER_HEIGHT: u32 = 270;

/// Window dimensions (may differ from render resolution for upscaling).
pub const WINDOW_WIDTH: u32 = 1280;
pub const WINDOW_HEIGHT: u32 = 720;
