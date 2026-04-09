# Frame Spare-Time Optimizer

**Date:** 2026-04-08 (v2 — rewrite addressing scheduling, parallelism, budget, and pipeline concerns)

## Goal

Use leftover frame time after sim tick + render to incrementally improve acceleration structures, LOD, lighting, and caching. The longer a player stays in one area, the better everything gets.

## Phase 0: Instrument Before Building

Before implementing any of this, instrument the current frame to measure where time goes:

```
[frame] camera=0.01ms  sim=2.1ms  upload=4.3ms  render=8.1ms  present=0.5ms  total=15.0ms  spare=1.6ms
```

If spare time is <1ms consistently, the optimizer is moot — fix the upstream bottleneck first (likely upload or render). The instrumentation stays permanently as a HUD/log line.

## Frame Model

```
┌─────────────────── Frame N ───────────────────┐
│ camera │ sim │ upload │ render │ present │ GATE │ ... idle or vsync ...
│        │     │        │        │         │      │
│◄──────── mandatory frame work ────────►│      │
│                                         │      │
│                                    deadline    │
│                                    check:     │
│                                    submit     │
│                                    new jobs   │
│                                    to pool    │
└───────────────────────────────────────────────┘

Thread pool works continuously across frames.
Completed results are collected at start of next frame.
```

**Mandatory per-frame work** (not spare-time — these are part of frame_work):
- Frustum cull: rebuild visible mega set from camera frustum
- Draw order sort: sort visible megas front-to-back for early-Z
- Collect completed results from thread pool

**Spare-time dispatcher** (runs after present):
- Checks remaining budget: `deadline = frame_start + 16.6ms`
- Does NOT execute jobs — submits them to the thread pool
- Submission is gated by deadline: stop submitting if we're past budget
- Jobs already in-flight on the pool continue regardless of deadline

## Thread Pool Architecture

```rust
struct FrameOptimizer {
    pool: rayon::ThreadPool,                   // N-1 worker threads
    queue: Mutex<BinaryHeap<Job>>,
    pending: Mutex<HashSet<(JobKind, MegaPos)>>,
    in_flight: AtomicUsize,
    results: Mutex<Vec<JobResult>>,            // completed work, collected next frame
    cache: Mutex<OptCache>,
    dirty_coalesce: DirtyCoalescer,
}
```

**Submission** (main thread, after present):
```rust
fn submit_jobs(&self, deadline: Instant) {
    let mut queue = self.queue.lock();
    while Instant::now() < deadline {
        let job = match queue.pop() {
            Some(j) => j,
            None => break,
        };
        self.in_flight.fetch_add(1, Relaxed);
        self.pool.spawn(move || {
            let result = job.execute();  // runs on worker thread
            self.results.lock().push(result);
            self.in_flight.fetch_sub(1, Relaxed);
        });
    }
}
```

**Collection** (main thread, start of next frame, before upload):
```rust
fn collect_results(&mut self) {
    for result in self.results.lock().drain(..) {
        match result {
            JobResult::SurfaceShell { target, grid } => {
                // Stage for GPU upload next upload_megas() call.
                self.staged_uploads.insert(target, grid);
            }
            JobResult::FaceMask { target, mask } => {
                self.cache.face_masks.insert(target, mask);
            }
            // ...
        }
    }
}
```

**CPU prep vs GPU upload**: All spare-time jobs are CPU-only. They produce data (grids, masks, meshes, AO buffers) that gets staged. GPU uploads happen exclusively during the mandatory `upload_megas()` phase using the graphics queue. No transfer queue complexity.

## Job Overrun Protection

Jobs that operate on a single mega-chunk (most of them) must implement cooperative yielding:

```rust
trait IncrementalJob {
    /// Do one unit of work. Returns Continue or Done.
    fn step(&mut self) -> JobStatus;
    /// Approximate progress 0.0-1.0.
    fn progress(&self) -> f32;
}

enum JobStatus {
    Continue,               // More work to do — re-enqueue as continuation
    Done(JobResult),        // Finished
}
```

Large jobs (SVDAG, nav mesh, structural graph, greedy mesh) process one slice or one sub-region per `step()`. A step targets ~0.5ms. If a job's step runs long on a pathological input, the damage is bounded to one worker thread for one step — other workers and the main thread are unaffected.

Jobs that cannot be subdivided (e.g., disk I/O) run on a dedicated low-priority thread, not the compute pool.

## Dirty Coalescing

One voxel edit can invalidate a mega and its 6 neighbors × ~10 job types = 70 jobs. A building collapse touching 20 megas produces 1400 jobs. Without throttling, the queue saturates for seconds.

```rust
struct DirtyCoalescer {
    /// MegaPos → tick when first dirtied.
    dirty_since: HashMap<MegaPos, u64>,
    /// Minimum ticks a mega must stay dirty before jobs are enqueued.
    /// Prevents thrashing during rapid edits (construction, collapse).
    settle_ticks: u64,  // default: 5 ticks = 0.5s
}

impl DirtyCoalescer {
    fn mark_dirty(&mut self, mp: MegaPos, current_tick: u64) {
        self.dirty_since.entry(mp).or_insert(current_tick);
    }

    /// Drain megas that have been dirty long enough. Enqueue jobs for them.
    fn drain_settled(&mut self, current_tick: u64) -> Vec<MegaPos> {
        let mut settled = Vec::new();
        self.dirty_since.retain(|mp, since| {
            if current_tick - *since >= self.settle_ticks {
                settled.push(*mp);
                false
            } else {
                true
            }
        });
        settled
    }
}
```

For each settled mega, enqueue one combined "rebuild" job that does shell + face mask + mip in a single pass, rather than 10 separate jobs. Neighbor invalidation is lazy: neighbors are marked dirty too, but they go through the same settle period.

## Starvation Prevention

Without mitigation, camera movement floods the queue with high-priority jobs and low-priority work (AO, nav mesh, compression) never runs.

**Reserved budget per tier**: Each frame's spare-time budget is split:
- 60% for tier 1-2 (culling + geometry)
- 25% for tier 3-4 (spatial + LOD)
- 15% for tier 5+ (lighting, sim support, memory)

If a tier has no pending work, its budget rolls up to the next tier.

**Age promotion**: Jobs that have been queued for >100 frames get their priority boosted by 10 (lower = higher priority). This guarantees eventual execution even for the lowest-priority work.

## Cache Budget

The `OptCache` has a fixed memory budget (default: 64MB). Each entry has a known size:

| Data | Size per mega | Notes |
|------|--------------|-------|
| AO texture | 256 KB | 64^3 × u8 |
| Light levels | 256 KB | 64^3 × u8 |
| Column heights | 4 KB | 64×64 × u8 |
| Face masks | 1 byte | u8 |
| Height map | 16 KB | 64×64 × i32 |
| Nav mesh | ~10-50 KB | Varies |
| Structural graph | ~5-20 KB | Varies |

At 100 loaded megas with full cache: ~55 MB (fits in 64MB budget).

**Eviction**: LRU by last-access time. When budget is exceeded, evict the least-recently-accessed mega's cache entries (all of them — partial eviction adds complexity for no gain). The mega is re-dirtied so its cache rebuilds if accessed again.

## Job Catalog

### Mandatory Per-Frame (not spare-time)

**Frustum cull**: Test each loaded mega's AABB against the camera frustum. Output: `Vec<MegaPos>` of visible megas passed to the renderer. ~0.1ms for 200 megas.

**Draw order sort**: Sort the visible mega list front-to-back by distance to camera. Enables early-Z rejection in G-buffer pass. ~0.05ms.

### Tier 1 — Culling (priority 0-9)

**1. Hierarchical occlusion** (priority 2, ~0.5ms per batch)
Rasterize a coarse CPU depth buffer from the nearest megas' bounding boxes. Test farther megas against it. Entire underground sections and terrain behind hills get culled. Re-run when camera moves >10 units. This is typically the single biggest win for dense voxel worlds.

**2. Empty mega flag** (priority 4, ~0.01ms per mega)
Flag megas that are 100% air. Computed once at generation, invalidated on voxel placement. Flagged megas bypass frustum testing entirely.

**3. Face visibility masks** (priority 5, ~0.2ms per mega)
6-bit mask per mega: which faces are exposed. The renderer skips ray entry from masked faces. Recomputed when mega or neighbor settles from dirty state.

**4. Interior cavity detection** (priority 9, ~1ms per mega, yielding)
Flood-fill from exposed-face air voxels. Sealed cavities are flagged so the renderer skips their voxels. Yields per slice (one Z-layer per step).

### Tier 2 — Geometry Reduction (priority 10-19)

**5. Surface shell extraction** (priority 10, ~1ms per mega, yielding)
Zero interior solid voxels with no air neighbor. Produces a sparse grid that the ray-march skips via mip hierarchy. Yields per Z-layer.

**6. Column height cache** (priority 12, ~0.5ms per mega)
64×64 max-height array. Enables early ray termination for top-down views. Rays above column height stop immediately.

**7. Mip refinement** (priority 15, ~0.5ms per mega)
Rebuild mip1 (32^3) and mip2 (16^3) from current grid. Runs after surface shell extraction for correct empty-space skipping.

### Tier 3 — LOD and Streaming (priority 20-29)

**8. LOD demotion** (priority 20, ~1ms per mega, yielding)
Downsample distant mega: 64^3 → 32^3 → 16^3. Criteria: distance > 256 units, not dirty for >200 frames.

**9. LOD promotion** (priority 21, ~1ms per mega)
Camera approaching — rebuild full 64^3 from sim chunks. Higher priority than demotion so near terrain sharpens first.

**10. VRAM eviction** (priority 25, ~0.1ms per mega)
Destroy GPU textures beyond load radius × 1.5. Re-dirty for reload.

**11. Disk serialization** (priority 28, ~2ms per mega, dedicated I/O thread)
Write evicted mega data to disk cache. Runs on a separate I/O thread, not the compute pool.

### Tier 4 — Lighting and Visual Quality (priority 30-39)

**12. AO bake** (priority 30, ~1.5ms per mega, yielding)
Per-surface-voxel hemisphere occlusion sampling. Stored as 64^3 u8 texture. Yields per Z-layer.

**13. Light flood fill** (priority 32, ~1ms per mega, yielding)
BFS propagation of light levels from sky + emissive voxels through air. Stored as 64^3 u8 texture. Yields per BFS wavefront.

**14. Shadow cache** (priority 34, ~1ms per mega)
Pre-render sun shadow map region for static terrain. Cache until chunk changes.

**15. Normal smoothing** (priority 36, ~0.5ms per mega)
Average surface normals across 3×3 neighborhood. Less blocky terrain shading.

### Tier 5 — Sim Support (priority 40-49)

**16. Height map bake** (priority 40, ~0.3ms per mega)
2D height array for O(1) `surface_height()`. Also used for entity placement.

**17. Structural graph** (priority 42, ~1ms per mega, yielding)
Connectivity graph for load-bearing analysis. Enables instant collapse detection. Yields per sub-region.

**18. Collision mesh** (priority 44, ~1ms per mega, yielding)
Simplified convex decomposition for physics queries.

**19. Nav mesh bake** (priority 46, ~1.5ms per mega, yielding)
Walkable surface polygons with adjacency for NPC pathfinding.

### Tier 6 — Memory Optimization (priority 50-59)

**20. Texture atlas** (priority 50, ~0.5ms per batch)
Pack entity marker textures into a single atlas. One draw call for all entities.

**21. BC compression** (priority 55, ~1ms per mega)
Block-compress palette textures for ~4× VRAM bandwidth reduction. Only for stable (not recently dirty) megas.

## Hybrid Rendering: Ray-March vs Greedy Mesh

This is the largest architectural decision and deserves dedicated treatment.

**Current state**: All megas render via DDA ray-march through 64^3 3D textures. This is simple and correct but expensive for flat terrain — a grass plain is 64^3 voxels ray-marched when it could be a handful of quads.

**Greedy mesh path**: For megas whose surface is simple (few material types, mostly flat), a greedy mesh is orders of magnitude faster. The mesh renderer uses traditional vertex/index buffer rasterization through the same G-buffer.

**Transition criteria**: A mega switches from ray-march to mesh when:
- Its greedy mesh has been built (background job)
- The mesh quad count is below a threshold (e.g., <2000 quads)
- The mega hasn't been dirty for >50 frames (stable terrain)

**Coexistence**: Both paths write to the same G-buffer (albedo, normal, material). The deferred lighting pass doesn't care which path produced the fragments. In any given frame, some megas are ray-marched and others are rasterized.

**Fallback**: When a meshed mega gets dirtied (construction, mining), it reverts to ray-march immediately. The mesh is invalidated and rebuilt in the background. This ensures edits are always visible instantly.

**Implementation**: Phase 3 of implementation (after core framework and culling are working). Requires a second graphics pipeline (vertex shader + fragment shader writing to G-buffer) alongside the existing ray-march pipeline.

## SVDAG Note

SVDAG at 64^3 (4 octree levels) has questionable ROI. The pointer overhead and traversal indirection may cost more than the mip hierarchy already provides. Do not implement until profiling shows ray traversal is the bottleneck, and even then benchmark against the existing flat mip approach first. If implemented, target 256^3+ volumes (merged groups of megas for distant terrain).

## Implementation Order

**Phase 1 — Instrumentation + Framework**
1. Frame timing instrumentation (measure where time goes)
2. `FrameOptimizer` struct with thread pool, queue, coalescer
3. Mandatory per-frame frustum cull + draw order sort (move out of spare-time)
4. Result collection + staged upload pipeline

**Phase 2 — Culling (biggest wins first)**
5. Hierarchical occlusion culling
6. Face visibility masks
7. Empty mega flags
8. Surface shell extraction

**Phase 3 — Geometry + Hybrid Rendering**
9. Column height cache
10. Mip refinement
11. Greedy mesh builder + rasterization pipeline
12. Hybrid render path (ray-march vs mesh per mega)

**Phase 4 — LOD**
13. LOD demotion/promotion
14. VRAM eviction improvements
15. Disk serialization (I/O thread)

**Phase 5 — Visual Quality**
16. AO bake
17. Light flood fill
18. Shadow cache
19. Normal smoothing

**Phase 6 — Sim Support + Memory**
20. Height map bake
21. Structural graph
22. Nav mesh bake
23. Texture atlas
24. Interior cavity detection
