# `--world small` Test Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--world small` CLI flag that launches a fixed 9x9x9 chunk forest world with a seeded settlement for NPC behavior iteration.

**Architecture:** A `build_small_world()` function creates a tiny `RegionPlan` + `WorldState` with one settlement and 10 NPCs. The voxel app gets a `world_extent` field that clips chunk loading and camera movement to [0, 9) in chunk coords. Feature density is suppressed in a clearing around the settlement center.

**Tech Stack:** Rust, clap (CLI), existing world_sim + terrain + voxel_app systems.

---

### Task 1: Add `--world` CLI argument

**Files:**
- Modify: `src/bin/xtask/cli/mod.rs:51-120` (WorldSimArgs)
- Modify: `src/bin/xtask/world_sim_cmd.rs:13-42` (run_world_sim)

- [ ] **Step 1: Add the `--world` field to WorldSimArgs**

In `src/bin/xtask/cli/mod.rs`, add after the `render` field (line 119):

```rust
    /// World preset: "small" = fixed 9x9x9 forest test scene (default: infinite)
    #[arg(long)]
    pub world: Option<String>,
```

- [ ] **Step 2: Add the `build_small_world` branch in run_world_sim**

In `src/bin/xtask/world_sim_cmd.rs`, change the state-building block (lines 19-42) to check `args.world` first:

```rust
    let state = if let Some(ref path) = args.load {
        // ... existing load path unchanged ...
    } else if args.world.as_deref() == Some("small") {
        build_small_world(&args)
    } else if args.peaceful {
        build_peaceful_world(&args)
    } else {
        build_world(&args)
    };
```

- [ ] **Step 3: Write the `build_small_world` stub**

Add at the bottom of `src/bin/xtask/world_sim_cmd.rs` (before tests):

```rust
fn build_small_world(args: &WorldSimArgs) -> WorldState {
    use game::world_sim::state::*;
    use game::world_sim::terrain;
    use game::world_sim::voxel::{CHUNK_SIZE, ChunkPos};

    let seed = args.seed;
    let mut state = WorldState::new(seed);

    // 9x9x9 chunks = 576x576x576 voxels. Settlement at center (288, 288).
    let world_size_voxels = 9 * CHUNK_SIZE as i32; // 576
    let center = (world_size_voxels as f32 / 2.0, world_size_voxels as f32 / 2.0);

    // Create a tiny region plan covering just the 9x9 footprint.
    // One cell is CELL_SIZE=4096 voxels, so the whole 576-voxel world fits
    // inside a single cell. We create a 1x1 plan with Forest terrain.
    let plan = terrain::create_small_world_plan(seed, center);
    state.voxel_world.region_plan = Some(plan.clone());

    // Pre-generate all 729 chunks on the CPU.
    for cz in 0..9 {
        for cy in 0..9 {
            for cx in 0..9 {
                let cp = ChunkPos::new(cx, cy, cz);
                state.voxel_world.generate_chunk(cp, seed);
            }
        }
    }

    // Single forest region (for sim systems that read regions).
    state.regions.push(RegionState {
        id: 0,
        name: "Clearbrook Forest".into(),
        terrain: Terrain::Forest,
        pos: center,
        monster_density: 0.0,
        faction_id: None,
        threat_level: 0.0,
        has_river: false,
        has_lake: false,
        is_coastal: false,
        river_connections: vec![],
        dungeon_sites: vec![],
        sub_biome: SubBiome::LightForest,
        neighbors: vec![],
        is_chokepoint: false,
        elevation: 1,
        is_floating: false,
        unrest: 0.0,
        control: 1.0,
    });

    // Settlement at world center.
    let settlement_id = 0u32;
    let surface_z = terrain::surface_height_at(
        center.0, center.1, &plan, seed,
    );
    state.settlements.push(SettlementState::new(
        settlement_id,
        "Clearbrook".into(),
        center,
    ));

    // Spawn 10 NPCs in a ring around the settlement center.
    let archetypes = ["builder", "builder", "woodcutter", "woodcutter",
                      "farmer", "miner", "smith", "hunter", "healer", "merchant"];
    let mut id = 0u32;
    for (i, archetype) in archetypes.iter().enumerate() {
        let angle = (i as f32 / archetypes.len() as f32) * std::f32::consts::TAU;
        let dist = 8.0 + (i as f32) * 2.0;
        let px = center.0 + angle.cos() * dist;
        let py = center.1 + angle.sin() * dist;

        let mut npc = Entity::new_npc(id, (px, py));
        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(settlement_id);
        npc_data.faction_id = None;
        npc_data.gold = 0.0;
        npc_data.morale = 80.0;
        npc_data.archetype = archetype.to_string();
        npc_data.name = game::world_sim::naming::generate_personal_name(id, seed);
        npc.inventory = Some(Inventory::with_capacity(50.0));

        state.entities.push(npc);
        id += 1;
    }

    state.next_id = id;

    eprintln!("[small-world] Generated 9x9x9 forest world with settlement at ({}, {}), surface_z={}",
        center.0, center.1, surface_z);

    state
}
```

- [ ] **Step 4: Build and verify the new flag parses**

Run: `cargo build --bin xtask`
Expected: compiles (build_small_world calls `terrain::create_small_world_plan` which doesn't exist yet — will fail. That's Task 2).

- [ ] **Step 5: Commit**

```bash
git add src/bin/xtask/cli/mod.rs src/bin/xtask/world_sim_cmd.rs
git commit -m "feat: add --world small CLI flag with build_small_world stub"
```

---

### Task 2: Create the small-world region plan

**Files:**
- Modify: `src/world_sim/terrain/mod.rs` (add `create_small_world_plan` + `pub use`)
- Modify: `src/world_sim/terrain/region_plan.rs` (add constructor)

- [ ] **Step 1: Add `create_small_world_plan` to region_plan.rs**

Add at the bottom of `src/world_sim/terrain/region_plan.rs`:

```rust
/// Create a minimal RegionPlan for the `--world small` test scene.
/// The plan is a single cell covering the 576×576 voxel footprint,
/// uniformly Forest/LightForest, with a SettlementPlan at the center.
pub fn create_small_world_plan(seed: u64, center: (f32, f32)) -> RegionPlan {
    use super::super::state::{Terrain, SubBiome, SettlementKind};

    // One cell is CELL_SIZE voxels wide. The 576-voxel world fits in one cell.
    let cell = RegionCell {
        height: 0.3,
        moisture: 0.6,
        temperature: 0.5,
        terrain: Terrain::Forest,
        sub_biome: SubBiome::LightForest,
        settlement: Some(SettlementPlan {
            kind: SettlementKind::Town,
            local_pos: (0.5, 0.5), // center of cell
        }),
        dungeons: vec![],
        has_road: false,
    };

    RegionPlan {
        cols: 1,
        rows: 1,
        cells: vec![cell],
        rivers: vec![],
        roads: vec![],
        seed,
    }
}
```

- [ ] **Step 2: Re-export from terrain/mod.rs**

In `src/world_sim/terrain/mod.rs`, add the public re-export alongside `generate_continent`:

```rust
pub use region_plan::create_small_world_plan;
```

- [ ] **Step 3: Build**

Run: `cargo build --bin xtask`
Expected: compiles successfully (build_small_world can now call `terrain::create_small_world_plan`).

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/terrain/region_plan.rs src/world_sim/terrain/mod.rs
git commit -m "feat: add create_small_world_plan for 9x9x9 forest scene"
```

---

### Task 3: Add world extent + camera clamping

**Files:**
- Modify: `src/world_sim/voxel_app.rs` (AppState fields, update_camera, generate_camera_chunks)

- [ ] **Step 1: Add `world_extent` field to AppState**

In `src/world_sim/voxel_app.rs`, add a field to AppState (near line 110, after `last_spiral_chunk`):

```rust
    /// If set, the world is bounded to chunk coordinates [0, extent) on each
    /// axis. Camera is clamped and no chunks outside this range are generated.
    /// None = infinite world (current default behavior).
    world_extent: Option<i32>,
```

Initialize it to `None` in `AppState::new()` (near line 460):

```rust
            world_extent: None,
```

- [ ] **Step 2: Add camera clamping in update_camera**

At the end of `update_camera` in `src/world_sim/voxel_app.rs` (before the closing `}`), after the `camera.update(...)` call and `camera_version` bump (around line 1542), add:

```rust
        // Clamp camera to world extent if set.
        if let Some(extent) = self.world_extent {
            let max_voxel = (extent * CHUNK_SIZE as i32) as f32;
            let margin = 2.0;
            let eye = self.camera.eye_position();
            // Engine coords: x = sim x, y = sim z (up), z = sim y.
            let clamped_x = eye[0].clamp(margin, max_voxel - margin);
            let clamped_y = eye[1].clamp(margin, max_voxel - margin);
            let clamped_z = eye[2].clamp(margin, max_voxel - margin);
            if clamped_x != eye[0] || clamped_y != eye[1] || clamped_z != eye[2] {
                self.camera.set_position(glam::Vec3::new(clamped_x, clamped_y, clamped_z));
            }
        }
```

- [ ] **Step 3: Clip chunk loading to world extent**

In `generate_camera_chunks` (around line 715-748), wrap the chunk submission in an extent check. Add right after `let cp = ChunkPos::new(...)` and before the `pending_chunk_requests` check:

```rust
                    // Skip chunks outside world extent.
                    if let Some(extent) = self.world_extent {
                        if cp.x < 0 || cp.y < 0 || cp.z < 0
                            || cp.x >= extent || cp.y >= extent || cp.z >= extent
                        {
                            continue;
                        }
                    }
```

- [ ] **Step 4: Build**

Run: `cargo build`
Expected: compiles. `world_extent` defaults to `None` so existing behavior unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/voxel_app.rs
git commit -m "feat: add world_extent field with camera clamp + chunk clip"
```

---

### Task 4: Wire `--world small` through to the voxel app

**Files:**
- Modify: `src/world_sim/voxel_app.rs` (run_with_renderer signature, AppState::new)
- Modify: `src/bin/xtask/world_sim_cmd.rs` (pass world arg)

- [ ] **Step 1: Add `world_preset` parameter to `run_with_renderer`**

In `src/world_sim/voxel_app.rs`, change the signature (line 1725):

```rust
pub fn run_with_renderer(sim: WorldSim, world_preset: Option<&str>) -> Result<()> {
```

Thread it into `WorldSimVoxelApp`:

```rust
    let mut app = WorldSimVoxelApp {
        state: None,
        sim: Some(sim),
        world_preset: world_preset.map(|s| s.to_string()),
    };
```

Add the field to the `WorldSimVoxelApp` struct (around line 1550):

```rust
struct WorldSimVoxelApp {
    state: Option<AppState>,
    sim: Option<WorldSim>,
    world_preset: Option<String>,
}
```

- [ ] **Step 2: Pass world_preset into AppState::new**

In the `resumed` handler where `AppState::new` is called, pass the preset through. Change `AppState::new` to accept an `Option<&str>`:

```rust
fn new(window: Arc<Window>, sim: WorldSim, world_preset: Option<&str>) -> Result<Self> {
```

At the end of `AppState::new`, before `Ok(Self { ... })`, set `world_extent`:

```rust
        let world_extent = match world_preset {
            Some("small") => Some(9),
            _ => None,
        };
```

And include it in the return struct:

```rust
            world_extent,
```

- [ ] **Step 3: When `world_extent` is set, skip the disk loader and submit all chunks**

In `AppState::new`, after the `world_extent` assignment, if it's set, pre-submit all chunks to the GPU pool instead of using the settlement-based pre-gen:

```rust
        // For bounded worlds, submit every chunk to the GPU directly.
        // The disk loader will be clipped to this extent on subsequent frames.
        if let Some(extent) = world_extent {
            let plan = sim.state().voxel_world.region_plan.clone();
            let world_seed = sim.state().rng_state;
            if let Some(ref plan_ref) = plan {
                let mut count = 0usize;
                for cz in 0..extent {
                    for cy in 0..extent {
                        for cx in 0..extent {
                            let cp = crate::world_sim::voxel::ChunkPos::new(cx, cy, cz);
                            // Generate CPU-side if not already present
                            if !sim.state().voxel_world.chunks.contains_key(&cp) {
                                sim.state_mut().voxel_world.generate_chunk(cp, world_seed);
                            }
                            count += 1;
                        }
                    }
                }
                eprintln!("[voxel] Small world: {} chunks ready for GPU upload", count);
            }
        }
```

- [ ] **Step 4: Update the call site in world_sim_cmd.rs**

In `src/bin/xtask/world_sim_cmd.rs`, change the `run_with_renderer` call (line 90):

```rust
        match game::world_sim::voxel_app::run_with_renderer(sim, args.world.as_deref()) {
```

- [ ] **Step 5: Build and verify**

Run: `cargo build --features app --bin xtask`
Expected: compiles.

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/voxel_app.rs src/bin/xtask/world_sim_cmd.rs
git commit -m "feat: wire --world small through to voxel app with world_extent=9"
```

---

### Task 5: Add settlement clearing to feature placement

**Files:**
- Modify: `src/world_sim/terrain/features.rs:661` (add `clearing_center` param, suppress densities)
- Modify: `src/world_sim/terrain/materialize.rs:329` (pass `None` for new param)

The small world pre-generates all 729 chunks on the CPU. Those chunks fit in the 1024-slot GPU pool without eviction, so the GPU feature pass never regenerates them. Only the CPU path needs clearing suppression.

- [ ] **Step 1: Add `clearing_center` parameter to `place_surface_features`**

In `src/world_sim/terrain/features.rs`, change the signature (line 661):

```rust
pub fn place_surface_features(
    chunk: &mut Chunk,
    cp: ChunkPos,
    _terrain: Terrain,
    _sub_biome: SubBiome,
    _surface_z_local: i32,
    seed: u64,
    plan: Option<&RegionPlan>,
    clearing_center: Option<(f32, f32)>,
) {
```

- [ ] **Step 2: Add clearing suppression logic inside the origin loop**

In `place_surface_features`, after the z-range early-out (line 722-724) and before the tree density check (line 726), add:

```rust
            // Settlement clearing: suppress feature density near the
            // clearing center with a smooth linear falloff.
            let density_scale = if let Some((ccx, ccy)) = clearing_center {
                const CLEARING_RADIUS: f32 = 128.0; // ~2 chunks
                const FALLOFF: f32 = 32.0;
                let ddx = vx as f32 - ccx;
                let ddy = vy as f32 - ccy;
                let dist = (ddx * ddx + ddy * ddy).sqrt();
                if dist < CLEARING_RADIUS {
                    0.0
                } else if dist < CLEARING_RADIUS + FALLOFF {
                    (dist - CLEARING_RADIUS) / FALLOFF
                } else {
                    1.0
                }
            } else {
                1.0
            };
```

Then modify the three density checks to multiply by `density_scale`:

```rust
            // Tree placement — modulated by large-scale noise for clustering.
            if params.tree_density * density_scale > 0.0 {
                let td = noise::hash_f32(vx, vy, 0, seed.wrapping_add(TREE_DENSITY_SALT));
                let cluster = noise::fbm_2d(
                    vx as f32 * 0.025,
                    vy as f32 * 0.025,
                    seed.wrapping_add(0xC1C1),
                    2,
                    2.0,
                    0.5,
                );
                let effective_density = params.tree_density * density_scale * (0.3 + cluster * 1.4);
                if td < effective_density {
                    stamp_tree_procedural(
                        chunk, lx, ly, feature_base_z, seed, cell.terrain, cell.sub_biome,
                    );
                }
            }

            // Boulder placement.
            if params.boulder_density * density_scale > 0.0 {
                let bd = noise::hash_f32(vx, vy, 0, seed.wrapping_add(BOULDER_DENSITY_SALT));
                if bd < params.boulder_density * density_scale {
                    stamp_boulder(chunk, lx, ly, feature_base_z, seed);
                }
            }

            // Rock pillar placement (desert/badlands).
            if params.pillar_density * density_scale > 0.0 {
                let pd = noise::hash_f32(vx, vy, 0, seed.wrapping_add(PILLAR_DENSITY_SALT));
                if pd < params.pillar_density * density_scale {
                    stamp_pillar(chunk, lx, ly, feature_base_z, seed, params.pillar_material);
                }
            }
```

- [ ] **Step 3: Update existing callers to pass `None`**

In `src/world_sim/terrain/materialize.rs:329`:

```rust
        features::place_surface_features(&mut chunk, cp, cell.terrain, cell.sub_biome, surface_z_local, seed, Some(plan), None);
```

In `src/world_sim/terrain/features.rs` test helper calls (lines 813, 826):

```rust
        place_surface_features(&mut chunk, cp, Terrain::Forest, SubBiome::DenseForest, surface_local, 42, None, None);
```

```rust
        place_surface_features(&mut chunk, cp, Terrain::Forest, SubBiome::DenseForest, CHUNK_SIZE as i32, 42, None, None);
```

- [ ] **Step 4: Pass the clearing center in `build_small_world`**

Back in `src/bin/xtask/world_sim_cmd.rs`, in `build_small_world`, the CPU chunk generation loop already calls `state.voxel_world.generate_chunk(cp, seed)`, which calls `materialize_chunk`, which calls `place_surface_features` with `None` for clearing. To inject the clearing center, we need `materialize_chunk` to accept it too.

Add `clearing_center: Option<(f32, f32)>` to `materialize_chunk` in `src/world_sim/terrain/materialize.rs`:

```rust
pub fn materialize_chunk(cp: ChunkPos, plan: &RegionPlan, seed: u64, clearing_center: Option<(f32, f32)>) -> Chunk {
```

Thread it through to the `place_surface_features` call (line 329):

```rust
        features::place_surface_features(&mut chunk, cp, cell.terrain, cell.sub_biome, surface_z_local, seed, Some(plan), clearing_center);
```

Update `VoxelWorld::generate_chunk` in `src/world_sim/voxel.rs` to accept and pass it:

```rust
    pub fn generate_chunk(&mut self, cp: ChunkPos, seed: u64, clearing_center: Option<(f32, f32)>) {
        if self.chunks.contains_key(&cp) { return; }
        let chunk = if let Some(ref plan) = self.region_plan {
            crate::world_sim::terrain::materialize_chunk(cp, plan, seed, clearing_center)
        } else {
            self.generate_chunk_legacy(cp, seed)
        };
        self.chunks.insert(cp, chunk);
    }
```

Update all existing callers of `generate_chunk` and `materialize_chunk` to pass `None`.

In `build_small_world`, pass the center:

```rust
                state.voxel_world.generate_chunk(cp, seed, Some(center));
```

- [ ] **Step 5: Build and run tests**

Run: `cargo build && cargo test --lib`
Expected: compiles, all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/terrain/features.rs src/world_sim/terrain/materialize.rs src/world_sim/voxel.rs src/bin/xtask/world_sim_cmd.rs
git commit -m "feat: add settlement clearing suppression to feature placement"
```

---

### Task 6: End-to-end test run

**Files:** None (manual test)

- [ ] **Step 1: Launch the small world**

Run: `cargo run --features app --bin xtask -- world-sim --world small --render`

Expected:
- Terminal prints `[small-world] Generated 9x9x9 forest world...`
- Vulkan window opens showing a forest scene
- Center area is a clearing (fewer/no trees)
- Camera is positioned at the settlement, looking at the clearing
- WASD movement is clamped to the 576³ voxel bounds
- Trees render without dashed artifacts at chunk boundaries

- [ ] **Step 2: Verify NPC presence**

Unpause (Space) and observe sim ticks. NPCs should appear in the settlement area. Builders should begin placing structures after a few sim ticks.

- [ ] **Step 3: Commit any fixes**

If any issues found during testing, fix and commit with descriptive messages.
