# Frame Spare-Time Optimizer

**Date:** 2026-04-08

## Goal

Use leftover frame time after sim tick + render to incrementally improve acceleration structures, LOD, lighting, and caching. The longer a player stays in one area, the better everything gets — sharper LOD, baked AO, pre-built SVDAGs, cached nav meshes.

## Frame Budget Model

```
target_ms = 16.6  (60 fps)
frame_work = camera_update + sim_tick + mega_upload + render + present
spare_ms   = target_ms - frame_work - 1.0  (1ms safety margin)
```

After `present()`, drain the job queue until `spare_ms` is exhausted. Each job is one unit of work (typically one mega-chunk) and takes <2ms. If a job exceeds its time estimate, it still completes — the scheduler just skips remaining jobs that frame.

## Job Queue

`BinaryHeap<Job>` sorted by `(priority, distance_to_camera)`. Lower priority number = runs first. Within the same priority, closer to camera runs first.

```rust
struct Job {
    kind: JobKind,
    priority: u8,
    target: MegaPos,          // which mega-chunk (or region)
    distance_sq: f32,         // from camera, for sorting
    estimated_ms: f32,        // rough cost for budget tracking
}
```

Jobs are enqueued by:
- **Sim tick** — dirty chunks produce geometry + mip + SVDAG jobs
- **Camera movement** — frustum change produces cull + occlusion + LOD jobs
- **Timer** — periodic re-check for AO, lighting, nav mesh on nearby megas
- **Eviction** — distant megas produce eviction + disk-cache jobs

Duplicate jobs (same kind + same target) are deduplicated on insert.

## Job Catalog

### Tier 1 — Culling (priority 0-9)

These determine *what* to render. Highest priority because they reduce the draw call list.

**1. Frustum cull rebuild** (priority 0, ~0.1ms)
Rebuild the set of mega-chunks intersecting the camera frustum. Runs every frame the camera moves. Output: `HashSet<MegaPos>` of visible megas. Only visible megas get rendered.

**2. Empty mega skip** (priority 1, ~0.01ms per mega)
Flag mega-chunks that are 100% air. These never get uploaded or drawn. Computed once at load time, invalidated when a voxel is placed in the mega.

**3. Face visibility masks** (priority 5, ~0.2ms per mega)
For each mega-chunk, compute a 6-bit mask: which faces are exposed (neighbor is not fully solid on the touching face). The renderer skips ray entry from masked faces. Recomputed when the mega or any neighbor changes.

**4. Hierarchical occlusion** (priority 8, ~0.5ms per batch)
Rasterize a coarse depth buffer from the nearest N mega-chunks. Test farther megas against it. Occluded megas are skipped for that frame. Re-run when camera moves significantly (>10 units or >5 degree rotation).

**5. Interior cavity detection** (priority 9, ~1ms per mega)
Flood-fill from air voxels on exposed faces. Any sealed air pocket with no path to an exposed face is a hidden cavity. Mark its containing voxels so the renderer skips them. Recomputed when the mega is modified.

### Tier 2 — Geometry Reduction (priority 10-19)

These reduce how much work each draw call does.

**6. Surface shell extraction** (priority 10, ~1ms per mega)
Zero out all interior solid voxels that have no air-adjacent face. A mega that's solid below y=30 with grass on top becomes mostly empty in the 3D texture. The ray-march skips empty regions via mip hierarchy. Re-extract when dirty.

**7. Column height cache** (priority 12, ~0.5ms per mega)
Build a 64x64 array of max solid heights per column within the mega. Stored alongside the mega-chunk. The renderer uses this for early ray termination: rays from above that are past the column height can stop immediately. Rebuilt when dirty.

**8. Greedy mesh build** (priority 18, ~2ms per mega)
Extract visible surface faces and merge coplanar same-material rectangles into large quads. Output: a vertex/index buffer per mega. Requires a separate rasterization render path (not the current DDA ray-march). This is the long-term highest-impact optimization but requires a new pipeline. The job builds the mesh data; a separate system decides when to switch a mega from ray-march to mesh rendering.

### Tier 3 — Spatial Acceleration (priority 20-29)

These make ray traversal and queries faster.

**9. SVDAG compression** (priority 20, ~1.5ms per mega)
Build a sparse voxel directed acyclic graph from the 64^3 grid. The DAG merges identical sub-octants, enabling O(log n) empty/solid region skipping during ray traversal. Stored per-mega. Rebuilt when dirty. The renderer can use SVDAG for distant megas and full grid for near ones.

**10. Mip refinement** (priority 22, ~0.5ms per mega)
Rebuild mip1 (32^3) and mip2 (16^3) textures from the current grid data. Already happens at upload time, but this job re-runs for megas whose voxels changed mid-frame (e.g., building construction) without requiring a full mega rebuild.

**11. BVH for entities** (priority 25, ~0.3ms)
Build a bounding volume hierarchy over entity markers. Used for mouse picking (raycast → entity), frustum culling entities, and spatial queries. Rebuilt when entities move (once per sim tick at most).

**12. Frustum-sorted draw order** (priority 28, ~0.1ms)
Sort the visible mega list front-to-back relative to camera. Enables early-Z rejection in the G-buffer pass (fragments behind already-drawn geometry are discarded). Re-sort when camera moves.

### Tier 4 — LOD and Streaming (priority 30-39)

These manage memory and detail level.

**13. LOD demotion** (priority 30, ~1ms per mega)
Downsample a distant mega-chunk from 64^3 to 32^3 or 16^3. Upload the smaller texture and destroy the full-res one. Criteria: distance > N and mega hasn't been dirty for M frames. LOD levels: full (64^3), half (32^3), quarter (16^3).

**14. LOD promotion** (priority 31, ~1ms per mega)
Camera moved closer — rebuild full 64^3 and re-upload. Higher priority than demotion so approaching terrain sharpens fast.

**15. Priority reorder** (priority 32, ~0.05ms)
Re-sort the upload queue so nearest-to-camera dirty megas process first. Runs after significant camera movement.

**16. VRAM eviction** (priority 35, ~0.1ms per mega)
Destroy GPU textures for mega-chunks beyond load radius * 1.5. The mega is re-dirtied so it reloads when the camera returns.

**17. Disk serialization** (priority 38, ~2ms per mega)
Write evicted mega-chunk voxel data to a disk cache (e.g., `cache/mega_X_Y_Z.bin`). On reload, read from disk instead of regenerating terrain. Avoids redundant terrain generation.

### Tier 5 — Lighting and Visual Quality (priority 40-49)

These improve appearance without affecting correctness.

**18. AO bake** (priority 40, ~1.5ms per mega)
For each surface voxel, sample occlusion from neighboring voxels in a hemisphere. Store as a per-voxel u8 AO value (0=fully occluded, 255=fully open). Packed into a secondary 64^3 texture or baked into the palette alpha. Rebuilt when mega changes.

**19. Light flood fill** (priority 42, ~1ms per mega)
Propagate light levels from sky (top face) and emissive voxels (lava, torches) through air voxels using BFS. Each air voxel gets a light level 0-15. Stored in a secondary texture. Used by the deferred lighting pass for ambient + point light contribution.

**20. Shadow cache** (priority 44, ~1ms per mega)
Pre-render the sun shadow map for static terrain around this mega. Cache the result. Only re-render the shadow map region when a chunk in that area changes. Eliminates per-frame shadow map rendering for static terrain.

**21. Normal smoothing** (priority 46, ~0.5ms per mega)
Average surface normals across adjacent terrain voxels within a 3x3 neighborhood. Store smoothed normals in a secondary texture. Produces less blocky terrain appearance in the deferred lighting pass.

**22. Material edge blend** (priority 48, ~0.5ms per mega)
Detect material transitions (e.g., grass→dirt, stone→sand) on surface voxels. Generate blended palette colors for transition voxels. Reduces hard material boundaries.

### Tier 6 — Physics and Sim Support (priority 50-59)

These improve game interaction quality.

**23. Height map bake** (priority 50, ~0.3ms per mega)
Build a 2D height array for the mega region: `height[x][z] = max solid y`. Makes `surface_height()` O(1) instead of scanning columns. Also useful for entity placement and terrain queries.

**24. Structural graph** (priority 52, ~1ms per mega)
Precompute structural connectivity: which voxel groups are load-bearing, which would collapse if a support is removed. Stored as a graph of connected components with support flags. Enables instant structural collapse calculation when a voxel is damaged.

**25. Collision mesh** (priority 54, ~1ms per mega)
Generate a simplified convex decomposition of the terrain surface. Used for NPC collision, projectile impact, and physics queries. Simpler than the full voxel grid.

**26. Nav mesh bake** (priority 56, ~1.5ms per mega)
Extract walkable surfaces and build a navigation mesh. Stores polygonal walkable regions with adjacency. Used by NPC pathfinding. Rebuilt when terrain changes.

### Tier 7 — Memory Optimization (priority 60-69)

**27. Texture atlas packing** (priority 60, ~0.5ms per batch)
Pack multiple small entity marker textures (2^3 cubes) into a single texture atlas. Reduces draw calls for entities from N to 1.

**28. BC compression** (priority 62, ~1ms per mega)
Convert 3D palette textures to GPU block-compressed format (BC4 for single-channel material index). ~4x less VRAM bandwidth during ray traversal. Only for megas that haven't changed recently.

**29. GPU defrag** (priority 65, ~0.5ms per operation)
After a series of evictions, compact GPU allocations to reduce memory fragmentation. Copies live textures to fill gaps left by destroyed textures.

## Scheduler Implementation

```rust
struct FrameOptimizer {
    queue: BinaryHeap<Job>,
    /// Deduplication: (kind, target) → true if already queued.
    pending: HashSet<(JobKind, MegaPos)>,
    /// Cached results that persist until invalidated.
    cache: OptCache,
    /// Target frame time in ms.
    target_ms: f32,
}

struct OptCache {
    frustum_set: HashSet<MegaPos>,
    face_masks: HashMap<MegaPos, u8>,
    column_heights: HashMap<MegaPos, Vec<u8>>,  // 64*64 = 4096 entries
    ao_textures: HashMap<MegaPos, Vec<u8>>,
    light_levels: HashMap<MegaPos, Vec<u8>>,
    height_maps: HashMap<MegaPos, Vec<i32>>,
    structural_graphs: HashMap<MegaPos, StructuralGraph>,
    nav_meshes: HashMap<MegaPos, NavMesh>,
    lod_levels: HashMap<MegaPos, LodLevel>,
}

impl FrameOptimizer {
    fn run_spare_time(&mut self, state: &AppState, deadline: Instant) {
        let margin = Duration::from_millis(1);
        while Instant::now() + margin < deadline {
            let job = match self.queue.pop() {
                Some(j) => j,
                None => break,
            };
            self.pending.remove(&(job.kind, job.target));
            self.execute(job, state);
        }
    }

    fn on_camera_moved(&mut self, /* ... */) {
        self.enqueue(Job { kind: FrustumCull, priority: 0, .. });
        // Re-prioritize LOD jobs based on new distances.
    }

    fn on_chunk_dirty(&mut self, mp: MegaPos) {
        self.enqueue(Job { kind: SurfaceShell, priority: 10, target: mp, .. });
        self.enqueue(Job { kind: MipRefine, priority: 22, target: mp, .. });
        self.enqueue(Job { kind: FaceMask, priority: 5, target: mp, .. });
        // Invalidate cached AO, lighting, etc. for this mega.
        self.cache.invalidate(mp);
    }
}
```

## Integration Point

In `voxel_app.rs`, after `present()`:

```rust
WindowEvent::RedrawRequested => {
    let frame_start = Instant::now();

    app.update_camera(dt);
    app.tick_sim();
    app.upload_megas()?;
    app.render()?;

    // Spend remaining frame budget on background optimization.
    let deadline = frame_start + Duration::from_micros(16_600);
    app.optimizer.run_spare_time(&app, deadline);

    app.window.request_redraw();
}
```

## Invalidation Rules

| Event | Invalidates |
|-------|-------------|
| Voxel changed in mega M | Shell, face masks, mips, SVDAG, AO, light, shadow, height map, structural graph, collision mesh, nav mesh for M and its 6 neighbors |
| Camera moved >10 units | Frustum cull, occlusion probes, draw order sort |
| Camera moved >50 units | LOD re-evaluation for all loaded megas |
| Entity moved | BVH rebuild |
| Mega evicted | All cached data for that mega |
| Mega loaded | Enqueue shell + face mask + mip + height map (minimum viable set) |

## Implementation Order

Phase 1 (framework + immediate wins):
1. `FrameOptimizer` struct with priority queue and deduplication
2. Frustum culling (job 1)
3. Surface shell extraction (job 6)
4. Face visibility masks (job 3)
5. Column height cache (job 7)
6. Height map bake (job 23)

Phase 2 (spatial acceleration):
7. SVDAG compression (job 9)
8. Mip refinement (job 10)
9. Frustum-sorted draw order (job 12)
10. Empty mega skip (job 2)

Phase 3 (LOD):
11. LOD demotion/promotion (jobs 13-14)
12. Priority reorder (job 15)
13. VRAM eviction improvements (job 16)
14. Disk serialization (job 17)

Phase 4 (visual quality):
15. AO bake (job 18)
16. Light flood fill (job 19)
17. Shadow cache (job 20)
18. Normal smoothing (job 21)

Phase 5 (sim support):
19. Structural graph (job 24)
20. Collision mesh (job 25)
21. Nav mesh bake (job 26)

Phase 6 (memory + advanced):
22. Greedy mesh build (job 8)
23. Texture atlas (job 27)
24. BC compression (job 28)
25. Hierarchical occlusion (job 4)
26. Interior cavity detection (job 5)
