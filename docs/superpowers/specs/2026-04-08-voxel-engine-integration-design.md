# Voxel Engine Integration — World Sim Physical Layer

**Date:** 2026-04-08
**Status:** Draft

## Goal

Make the voxel engine `Scene` the authoritative physical layer for the world sim. The world sim reads terrain state from the Scene and writes changes through it. Rendering comes for free. Supports both live windowed mode and trace playback.

## Architecture

```
WorldSim tick → produces deltas (mine, build, move, damage)
      ↓
VoxelBridge syncs deltas into Scene (load_chunk, set_voxel, set_transform, damage_sphere)
      ↓
Scene.tick_sim() → physics, fragment splitting, cleanup
      ↓
Renderer draws the Scene (Vulkan pipeline)
      ↓
Scene events (collapse, fragment) → feed back as deltas into next WorldSim tick
```

### Data ownership

- **`Scene`** owns all voxel data and entity transforms. It is the source of truth for physical state.
- **`VoxelWorld`** becomes a thin wrapper that delegates to the Scene via `VoxelBridge`. All existing call sites (`get_voxel`, `set_voxel`, `surface_height`, `generate_chunk`, etc.) continue to work unchanged.
- **`VoxelBridge`** is the glue layer: it owns the `Scene`, the material palette, and the entity ID ↔ `EntityHandle` mapping.

## Components

### 1. VoxelBridge (`src/world_sim/voxel_bridge.rs`)

Central coordinator. Owns:

- `scene: Scene` — the voxel engine scene
- `palette: MaterialPalette` — maps `VoxelMaterial` u8 → engine palette entries
- `chunk_handles: HashMap<ChunkPos, ChunkHandle>` — loaded chunk tracking
- `entity_handles: HashMap<u32, EntityHandle>` — world sim entity ID → scene entity handle
- `marker_grid: VoxelGrid` — shared 2×2×2 colored cube grid, reused for all entity markers

**Public API:**

```rust
impl VoxelBridge {
    /// Create a new bridge (headless scene, no rendering).
    pub fn new() -> Self;

    // --- Chunk sync (called after each world sim tick) ---

    /// Scan VoxelWorld for dirty chunks, convert to VoxelGrid, push to Scene.
    /// Clears dirty flags on synced chunks.
    pub fn sync_chunks(&mut self, voxel_world: &mut VoxelWorld);

    /// Bulk-load all chunks from VoxelWorld into Scene (initial load / deserialize).
    pub fn load_all_chunks(&mut self, voxel_world: &mut VoxelWorld);

    // --- Entity sync (called after each world sim tick) ---

    /// Sync all alive entities: spawn new markers, update moved positions, despawn dead.
    pub fn sync_entities(&mut self, state: &WorldState);

    /// Remove an entity marker from the scene.
    pub fn remove_entity(&mut self, entity_id: u32);

    // --- Scene operations ---

    /// Advance the scene simulation (physics, cleanup, fragment processing).
    pub fn tick_sim(&mut self);

    /// Drain scene events (collapses, fragments) for feedback into world sim.
    pub fn drain_events(&mut self) -> Vec<SceneEvent>;

    /// Access the underlying Scene (for the App trait / renderer).
    pub fn scene(&self) -> &Scene;
    pub fn scene_mut(&mut self) -> &mut Scene;
}
```

### 2. VoxelWorld as wrapper (`src/world_sim/voxel.rs` refactor)

`VoxelWorld` keeps its public API but its internals change:

**VoxelWorld stays unchanged.** It keeps its `chunks: HashMap<ChunkPos, Chunk>` and all its methods work exactly as before. It remains the sim's source of truth for voxel state.

**VoxelBridge reads from VoxelWorld, not the other way around.** The bridge syncs *outward* from VoxelWorld to the Scene — it doesn't inject itself into VoxelWorld's internals. This avoids lifetime/borrowing complexity entirely.

The sync model is **pull-based on dirty chunks:**
- Each `Chunk` already has a `dirty: bool` flag.
- After each world sim tick, the bridge scans for dirty chunks, converts them to `VoxelGrid`, and pushes to Scene via `load_chunk()`. Then clears dirty flags.
- For single-voxel changes mid-tick (rare), the bridge can batch them.

### 3. Material Palette Mapping (`src/world_sim/voxel_bridge.rs`)

One-time mapping from `VoxelMaterial` (30+ variants) to `PaletteEntry`:

```rust
fn build_palette() -> MaterialPalette {
    let mut p = MaterialPalette::new();
    // Index 0 = Air (default, transparent)
    p.set(VoxelMaterial::Dirt as u8,      PaletteEntry { r: 139, g: 90,  b: 43,  material_type: Dirt, .. });
    p.set(VoxelMaterial::Stone as u8,     PaletteEntry { r: 136, g: 140, b: 141, material_type: Stone, .. });
    p.set(VoxelMaterial::Grass as u8,     PaletteEntry { r: 86,  g: 152, b: 59,  material_type: Foliage, .. });
    p.set(VoxelMaterial::Water as u8,     PaletteEntry { r: 64,  g: 128, b: 200, material_type: Ice, .. }); // translucent
    p.set(VoxelMaterial::WoodLog as u8,   PaletteEntry { r: 160, g: 120, b: 60,  material_type: Wood, .. });
    p.set(VoxelMaterial::WoodPlanks as u8,PaletteEntry { r: 190, g: 150, b: 90,  material_type: Wood, .. });
    p.set(VoxelMaterial::StoneBrick as u8,PaletteEntry { r: 160, g: 160, b: 160, material_type: Stone, .. });
    p.set(VoxelMaterial::Iron as u8,      PaletteEntry { r: 180, g: 180, b: 190, material_type: Metal, .. });
    // ... all 30+ materials
}
```

The `VoxelMaterial` repr(u8) values serve directly as palette indices (0–33), so no lookup table is needed — just cast.

### 4. Entity Markers

Each alive entity in the world sim gets a 2×2×2 voxel cube marker in the Scene:

- **NPCs:** Blue (team 0) or faction-colored
- **Monsters:** Red
- **Buildings:** Brown/gray based on material
- **Resources:** Green

A single shared `VoxelGrid(2,2,2)` filled with palette index 1 is spawned per entity. Color is set via per-entity palette override (or we use a small set of pre-built grids, one per color).

Entity sync happens once per world sim tick: iterate alive entities, update `set_transform()` for moved entities, `spawn()`/`despawn()` for new/dead ones.

### 5. App Loop (`src/world_sim/voxel_app.rs`)

Implements the voxel engine's `App` trait:

```rust
pub struct WorldSimApp {
    bridge: Arc<RwLock<VoxelBridge>>,
    sim: WorldSim,
    camera: FreeCamera,
    ticks_per_frame: u32,      // how many sim ticks per render frame
    paused: bool,
}

impl App for WorldSimApp {
    fn setup(&mut self, scene: &mut Scene) -> Result<()> {
        // Initial chunk loading already done during WorldSim::new()
        // Set up camera at first settlement position, looking down
        Ok(())
    }

    fn tick(&mut self, scene: &mut Scene, dt: f32) {
        if !self.paused {
            for _ in 0..self.ticks_per_frame {
                self.sim.tick();
            }
            self.sync_entities();
        }
    }

    fn on_input(&mut self, scene: &mut Scene, event: &WindowEvent) {
        // WASD + mouse → camera
        // Space → pause/unpause
        // +/- → speed
        // Escape → quit
    }
}
```

### 6. Trace Playback App (`src/world_sim/voxel_app.rs`)

Second `App` impl for trace playback:

```rust
pub struct TracePlaybackApp {
    bridge: VoxelBridge,
    controller: PlaybackController,
    camera: FreeCamera,
}
```

Same camera controls. `PlaybackController` already handles seek/speed/pause. On each frame, reconstruct `WorldState` at current tick, diff chunks against previous frame, sync changes to bridge.

### 7. CLI Integration (`src/bin/xtask/`)

**Live mode:**
```bash
cargo run --bin xtask -- world-sim --render [--entities 2000 --settlements 10 --ticks 5000]
```

New `--render` flag on `WorldSimArgs`. When set, creates a windowed `VoxelBridge` and runs the `WorldSimApp` event loop instead of the headless tick loop.

**Playback mode:**
```bash
cargo run --bin xtask -- visualize --render <trace.json>
```

New `--render` flag on visualize command. Loads trace, creates `TracePlaybackApp`.

## Refactor Scope

### Files modified:
- `src/rendering.rs` — GameRenderer replaced with VoxelBridge re-export or removed
- `src/bin/xtask/world_sim_cmd.rs` — `--render` flag, windowed app path
- `src/bin/xtask/visualize_cmd.rs` — `--render` flag, playback app path
- `src/bin/xtask/cli/mod.rs` — new CLI args

### Files unchanged:
- `src/world_sim/voxel.rs` — VoxelWorld internals completely untouched
- `src/world_sim/runtime.rs` — no changes needed

### Files created:
- `src/world_sim/voxel_bridge.rs` — VoxelBridge struct and material palette
- `src/world_sim/voxel_app.rs` — WorldSimApp and TracePlaybackApp

- All 150+ world sim systems — they call `state.voxel_world.get_voxel()` etc., completely unchanged
- `src/world_sim/visualizer.rs` — TraceFrame generation unchanged, WebSocket path still works
- `src/world_sim/state.rs` — no changes

## Chunk lifecycle

1. **Generation:** `VoxelWorld::generate_chunk()` fills a `Chunk` as before, marks it `dirty = true`.
2. **Sync:** After each world sim tick, `VoxelBridge::sync_chunks(&mut voxel_world)` iterates all chunks. Dirty chunks are converted to `VoxelGrid` and pushed to Scene via `load_chunk()`. Dirty flag is cleared.
3. **Modification:** `set_voxel()` / `mine_voxel()` / `fill_box()` operate on VoxelWorld as before (they already set `dirty = true`). Next sync picks up the changes.
4. **Unloading:** Not in v1 — all chunks stay loaded in both VoxelWorld and Scene.
5. **Initial load:** On startup (or after deserialize), all existing chunks are bulk-synced to the bridge.

## Entity lifecycle

1. **Spawn:** When a new entity appears (birth, recruitment, monster spawn), `bridge.sync_entity()` creates a 2×2×2 marker.
2. **Move:** Each tick, `bridge.sync_entity()` updates transform for any entity whose position changed.
3. **Death:** `bridge.remove_entity()` despawns the marker.
4. **Batch sync:** Single pass over `state.entities` per tick. Track previous positions to skip unchanged entities.

## Camera

- **FreeCamera** from voxel engine — WASD + mouse look
- Initial position: above first settlement, looking down at 45°
- Movement speed: ~50 world units/sec (covers settlement scale quickly)
- No clipping against terrain in v1

## What's deferred (not in this spec)

- `.vox` model loading for entity visualization (buildings, NPCs)
- Terrain LOD / chunk streaming based on camera distance
- Physics feedback (structural collapse → world sim damage events)
- Fluid rendering (water/lava in Scene)
- Shadows from buildings
- Building interior visualization in 3D
- Sound

## Success criteria

1. `world-sim --render` opens a Vulkan window showing terrain chunks with correct materials/colors
2. Entity markers visible at correct positions, colored by type
3. Terrain updates (building construction, mining) appear in real-time
4. Free-fly camera with WASD + mouse
5. Pause/unpause with Space, speed control with +/-
6. `visualize --render <trace>` plays back a recorded trace in the same renderer
7. All existing tests pass (VoxelWorld API unchanged)
8. Headless mode (no `--render`) still works as before
