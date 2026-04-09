# Frame Spare-Time Optimizer

**Date:** 2026-04-08 (v3)

## Goal

Use leftover frame time after sim tick + render to incrementally improve acceleration structures, LOD, lighting, and caching.

## Phase 0: Instrument

Use RenderDoc or equivalent GPU profiling tool to capture frames and identify where time is spent (vertex processing, fragment shading, memory transfers, CPU-side waits). Supplement with CPU-side `Instant` measurements at each stage boundary for the frame log:

```
[frame 1042] camera=0.01ms  sim=2.1ms  upload=4.3ms  render=8.1ms  present=0.5ms  spare=1.6ms  pool_results=3  gpu_util=62%
```

The GPU utilization percentage comes from vendor query extensions (e.g., `VK_KHR_performance_query`) or from RenderDoc's timeline view. If upload dominates, optimize the upload path before building the spare-time system. If render dominates, the culling jobs matter most. If GPU util is low while CPU is busy, we're CPU-bound and the thread pool is the right investment.

This instrumentation is permanent — always available via a debug HUD toggle.

## Frame Model

```
┌──────────────────── Frame N ─────────────────────┐
│ collect │ cull+sort │ sim │ upload │ render │ present │ SUBMIT │
│ results │ (mandatory)│     │        │        │         │ to pool│
└──────────────────────────────────────────────────────────────────┘
```

**Mandatory per-frame** (part of frame_work):
- Collect completed results from pool (swap double-buffer, process unlocked)
- Frustum cull: test loaded mega AABBs against camera frustum
- Draw order sort: front-to-back by distance for early-Z

**Post-present submission** (spare-time):
- Submit jobs to the thread pool until deadline
- Deadline gates submission, not execution — in-flight jobs continue

## Thread Pool

```rust
struct FrameOptimizer {
    pool: rayon::ThreadPool,                       // N-1 workers
    queue: Mutex<BinaryHeap<Job>>,
    pending: Mutex<HashSet<(JobKind, MegaPos)>>,
    generation: AtomicU64,                         // bumped on teleport/major discontinuity
    // Double-buffered results: workers push to back, main thread swaps and drains front.
    results_back: Mutex<Vec<JobResult>>,
    results_front: Vec<JobResult>,                 // only touched by main thread
    dirty: DirtyCoalescer,
    cache: OptCache,                               // only touched by main thread
    tier_budget: TierBudget,
}
```

**Submission** (main thread, after present):
```rust
fn submit_jobs(&self, deadline: Instant) {
    let gen = self.generation.load(Relaxed);
    let mut queue = self.queue.lock();
    let mut budget = self.tier_budget.reset(deadline);

    while Instant::now() < deadline {
        let job = match queue.pop() {
            Some(j) => j,
            None => break,
        };
        if !budget.can_submit(job.tier, job.estimated_ms) {
            // This tier's budget is spent — skip, but don't discard.
            // Put it back for next frame.
            queue.push(job);
            continue;
        }
        budget.charge(job.tier, job.estimated_ms);
        self.pending.lock().remove(&(job.kind, job.target));

        // Clone the Arc for the worker. Results go through channel.
        let results = Arc::clone(&self.results_back);
        let gen_snapshot = gen;
        let generation = Arc::clone(&self.generation_arc);

        self.pool.spawn(move || {
            // Bail if generation changed (player teleported).
            if generation.load(Relaxed) != gen_snapshot { return; }
            let result = job.execute();
            results.lock().push(result);
        });
    }
}
```

`results_back` is `Arc<Mutex<Vec<JobResult>>>` shared with workers. The `FrameOptimizer` itself is owned by the main thread; only the results vec and generation counter are shared via Arc.

**Collection** (main thread, start of frame, before upload):
```rust
fn collect_results(&mut self) {
    // Swap: grab completed results, give workers a fresh empty vec.
    // Lock held only for the swap — O(1), no contention.
    let mut back = self.results_back.lock();
    std::mem::swap(&mut self.results_front, &mut *back);
    drop(back);

    // Process unlocked — workers can push to the new back vec freely.
    for result in self.results_front.drain(..) {
        match result {
            JobResult::SurfaceShell { target, grid } => {
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

## Tier Budget

Submission-time estimated-ms caps per tier, not wall-clock percentages:

```rust
struct TierBudget {
    /// Max estimated ms to submit per tier per frame.
    caps: [f32; 6],
    /// Estimated ms submitted so far this frame.
    spent: [f32; 6],
}

impl TierBudget {
    fn reset(&mut self, _deadline: Instant) -> &mut Self {
        self.spent = [0.0; 6];
        self
    }
    fn can_submit(&self, tier: u8, estimated_ms: f32) -> bool {
        let t = tier as usize;
        self.spent[t] + estimated_ms <= self.caps[t]
    }
    fn charge(&mut self, tier: u8, estimated_ms: f32) {
        self.spent[tier as usize] += estimated_ms;
    }
}
```

Default caps (tuned after instrumentation — these are starting points):
- Tier 0 (culling): 3.0ms
- Tier 1 (geometry): 2.0ms
- Tier 2 (LOD): 1.0ms
- Tier 3 (lighting): 0.5ms
- Tier 4 (sim support): 0.5ms
- Tier 5 (memory): 0.3ms

If a tier has no pending work, its budget is NOT redistributed (avoids starving higher-priority work that arrives next frame). The caps are configurable and should be tuned based on Phase 0 instrumentation data.

## Starvation Prevention

**Age promotion**: Jobs queued for >100 frames get priority boosted by 5 per 100 frames. A tier-4 job (priority 40) reaches tier-2 (priority 20) after 400 frames (~6.6s). Promoted jobs consume the budget of their new tier — this is intentional; if a job has been starving for 6 seconds, it deserves tier-2 treatment.

The math for expected queue depths: with ~200 loaded megas and 20 job types, worst case is 4000 jobs. At ~20 jobs/frame throughput, full drain takes ~200 frames (~3.3s). Age promotion ensures the tail clears within ~10s even under sustained camera movement.

## Cancellation

**Generation counter**: Incremented on teleport, fast-travel, or camera discontinuity (movement >100 units in one frame). Every in-flight job checks the generation at start of `execute()` and bails immediately if stale.

**Queue clearing**: On generation bump, the queue is also cleared and `pending` is reset. Dirty megas around the new position are re-enqueued fresh.

```rust
fn on_teleport(&self) {
    self.generation.fetch_add(1, Relaxed);
    self.queue.lock().clear();
    self.pending.lock().clear();
    // Re-enqueue for new camera position happens in next frame's dirty scan.
}
```

## Dirty Coalescing

Two classes:

**Structural dirty** (voxel changed — must rebuild immediately):
- Surface shell extraction
- Mip refinement
- Face mask update
These are enqueued in the very next frame's submission, no settle delay. They produce the minimum viable visual update so the player sees the edit within 1-2 frames.

**Cosmetic dirty** (appearance/support — can settle):
- AO bake, light flood, shadow cache, normal smoothing
- Nav mesh, structural graph, collision mesh
- LOD re-evaluation
These go through the `DirtyCoalescer` with a settle period of 30 frames (~0.5s). Rapid edits (mining, building) don't trigger expensive cosmetic rebuilds until the dust settles.

```rust
struct DirtyCoalescer {
    cosmetic_dirty: HashMap<MegaPos, u64>,  // MegaPos → frame when first dirtied
    settle_frames: u64,                      // 30
}
```

Structural jobs are separate individual jobs enqueued directly, not combined. Shell is yielding (~1ms), face mask is fast (~0.2ms), mip is fast (~0.5ms). They run independently on the pool. The combined "rebuild" concept from v2 is dropped.

## Cooperative Yielding

Large jobs implement `IncrementalJob`:

```rust
enum StepResult {
    Continue(Box<dyn IncrementalJob + Send>),  // re-enqueue continuation
    Done(JobResult),
}

trait IncrementalJob: Send {
    fn step(&mut self) -> StepResult;
}
```

A step targets ~0.3ms. Jobs that exceed this on pathological input only block one worker thread for one step. The continuation is re-enqueued at the same priority.

Jobs that don't need yielding (face mask, height map, mip) implement `execute()` directly and return `Done`.

## Cache Budget

Fixed 64MB limit. LRU eviction by last-access frame.

| Data | Size per mega |
|------|--------------|
| AO texture | 256 KB |
| Light levels | 256 KB |
| Column heights | 4 KB |
| Face masks | 1 byte |
| Height map | 16 KB |
| Nav mesh | ~10-50 KB |
| Structural graph | ~5-20 KB |

At 100 megas with full cache: ~55 MB. Exceeding 64MB triggers LRU eviction of entire mega cache entries.

## Job Catalog

### Mandatory Per-Frame

**Frustum cull**: AABB test per loaded mega against camera frustum. ~0.1ms for 200 megas.

**Draw order sort**: Front-to-back by camera distance. ~0.05ms.

### Tier 0 — Culling (priority 0-9)

**1. Hierarchical occlusion** (priority 2, ~0.5ms, yielding per depth slice)
Rasterize coarse CPU depth buffer from nearest megas, reject megas behind hills/terrain. Typically the single biggest win. Re-run on camera move >10 units.

**2. Empty mega flag** (priority 4, ~0.01ms per mega)
Flag 100% air megas. Computed at generation, invalidated on voxel placement.

**3. Face visibility masks** (priority 5, ~0.2ms per mega, structural-dirty)
6-bit exposed face mask. Renderer skips ray entry from masked faces.

**4. Interior cavity detection** (priority 9, ~1ms per mega, yielding per Z-layer)
Flood-fill from exposed faces. Sealed cavities flagged for skip.

### Tier 1 — Geometry (priority 10-19)

**5. Surface shell extraction** (priority 10, ~1ms per mega, yielding per Z-layer, structural-dirty)
Zero interior voxels with no air neighbor. Sparse grid for fast mip skip.

**6. Column height cache** (priority 12, ~0.5ms per mega)
64×64 max-height array. Early ray termination above terrain.

**7. Mip refinement** (priority 15, ~0.5ms per mega, structural-dirty)
Rebuild mip1/mip2 from current grid after shell extraction.

### Tier 2 — LOD and Streaming (priority 20-29)

**8. LOD demotion** (priority 20, ~1ms, yielding)
Downsample distant megas: 64³→32³→16³. Criteria: distance >256, stable >200 frames.

**9. LOD promotion** (priority 21, ~1ms)
Camera approaching — rebuild full 64³.

**10. VRAM eviction** (priority 25, ~0.1ms)
Destroy GPU textures beyond load radius × 1.5.

**11. Disk serialization** (priority 28, ~2ms, dedicated I/O thread)
Write evicted data to disk cache. Separate thread, not compute pool.

### Tier 3 — Lighting (priority 30-39, cosmetic-dirty)

**12. AO bake** (priority 30, ~1.5ms, yielding per Z-layer)
Per-surface-voxel hemisphere occlusion. 64³ u8 texture.

**13. Light flood fill** (priority 32, ~1ms, yielding per BFS wavefront)
Sky + emissive light propagation through air. 64³ u8 texture.

**14. Shadow cache** (priority 34, ~1ms)
Baked sun shadow map for static terrain.

**15. Normal smoothing** (priority 36, ~0.5ms)
Averaged surface normals for smoother shading.

### Tier 4 — Sim Support (priority 40-49, cosmetic-dirty)

**16. Height map bake** (priority 40, ~0.3ms)
2D height array for O(1) surface_height().

**17. Structural graph** (priority 42, ~1ms, yielding)
Load-bearing connectivity for collapse detection.

**18. Collision mesh** (priority 44, ~1ms, yielding)
Simplified convex decomposition for physics. Note: NPCs use raw voxel grid queries (`get_voxel`, `surface_height`) for collision until the collision mesh is built. The raw grid is always available and correct — the mesh is an acceleration, not a prerequisite.

**19. Nav mesh bake** (priority 46, ~1.5ms, yielding)
Walkable surfaces with adjacency for pathfinding.

### Tier 5 — Memory (priority 50-59)

**20. Texture atlas** (priority 50, ~0.5ms per batch)
Pack entity markers into one texture.

**21. BC compression** (priority 55, ~1ms per mega)
Block-compress stable megas for ~4× bandwidth reduction.

## Hybrid Rendering: Ray-March vs Greedy Mesh

**Current**: DDA ray-march through 64³ 3D textures.

**Greedy mesh**: Extract visible surface voxels, merge coplanar same-material faces into quads. Orders of magnitude fewer fragments for flat terrain.

**Transition criteria with hysteresis**:
- Ray-march → mesh: quad count < 2000 AND stable > 50 frames AND not meshed-then-reverted in last 200 frames
- Mesh → ray-march: voxel edit in the mega (immediate, for responsiveness)
- The 200-frame cooldown prevents thrash when a mega is near the quad threshold and getting repeated small edits.

**Coexistence**: Both paths write to the same G-buffer. In any frame, some megas are ray-marched and others are rasterized. The deferred lighting pass is agnostic to which path produced the fragments.

**Implementation**: Phase 3 (after core framework and culling). Requires a second graphics pipeline (vertex + fragment shaders writing to G-buffer).

## SVDAG

Deferred. At 64³ (4 octree levels), pointer overhead and traversal indirection likely cost more than the flat mip hierarchy. Do not implement until profiling shows ray traversal is the bottleneck, and benchmark against existing mips first. If the data eventually shows it's worthwhile, it would operate on merged mega groups — but that architecture is out of scope for this spec.

## Implementation Order

**Phase 1 — Instrumentation + Framework**
1. Frame timing instrumentation (CPU stages + GPU profiling via RenderDoc)
2. `FrameOptimizer` struct with thread pool, Arc-shared results, generation counter
3. Mandatory per-frame frustum cull + draw order sort
4. Double-buffered result collection + staged upload pipeline
5. Dirty coalescer (structural vs cosmetic split)
6. Tier budget tracking during submission

**Phase 2 — Culling**
7. Hierarchical occlusion culling
8. Face visibility masks
9. Empty mega flags
10. Surface shell extraction

**Phase 3 — Geometry + Hybrid Rendering**
11. Column height cache
12. Mip refinement
13. Greedy mesh builder + rasterization pipeline
14. Hybrid render path with hysteresis

**Phase 4 — LOD**
15. LOD demotion/promotion
16. VRAM eviction improvements
17. Disk serialization (I/O thread)

**Phase 5 — Visual Quality**
18. AO bake
19. Light flood fill
20. Shadow cache
21. Normal smoothing

**Phase 6 — Sim Support + Memory**
22. Height map bake
23. Structural graph
24. Nav mesh bake
25. Texture atlas
26. Interior cavity detection
