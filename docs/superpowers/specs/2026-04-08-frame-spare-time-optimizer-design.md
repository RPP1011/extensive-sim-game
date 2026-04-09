# Frame Spare-Time Optimizer

**Date:** 2026-04-08 (v4)

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
    pool: rayon::ThreadPool,                          // N-1 workers
    queue: Mutex<BinaryHeap<Job>>,
    pending: Mutex<HashSet<(JobKind, MegaPos)>>,
    generation: Arc<AtomicU64>,                       // Acquire/Release, shared with workers
    results_back: Arc<Mutex<Vec<JobResult>>>,         // workers push here
    results_front: Vec<JobResult>,                    // main thread only
    dirty: DirtyCoalescer,
    cache: OptCache,                                  // main thread only
    tier_budget: TierBudget,
    loaded_megas: HashSet<MegaPos>,                   // main thread tracks what's loaded
}
```

**Submission** (main thread, after present):
```rust
fn submit_jobs(&mut self, spare_ms: f32) {
    let gen = self.generation.load(Acquire);
    let deadline = Instant::now() + Duration::from_secs_f32(spare_ms / 1000.0);

    self.tier_budget.reset(spare_ms);

    // Drain queue into local batch to avoid holding locks in the hot loop.
    let mut batch = Vec::new();
    {
        let mut queue = self.queue.lock();
        let mut pending = self.pending.lock();
        while let Some(job) = queue.pop() {
            pending.remove(&(job.kind, job.target));
            batch.push(job);
            if batch.len() >= 64 { break; } // cap batch size per frame
        }
    }
    // No locks held from here.

    for job in batch {
        if Instant::now() >= deadline { 
            // Re-enqueue unsubmitted jobs.
            let mut queue = self.queue.lock();
            let mut pending = self.pending.lock();
            queue.push(job);
            pending.insert((job.kind, job.target));
            // (remaining batch items also re-enqueued — omitted for brevity)
            break;
        }
        if !self.tier_budget.can_submit(job.tier, job.estimated_ms) {
            // Tier budget spent — re-enqueue.
            let mut queue = self.queue.lock();
            let mut pending = self.pending.lock();
            queue.push(job);
            pending.insert((job.kind, job.target));
            continue;
        }
        self.tier_budget.charge(job.tier, job.estimated_ms);

        let results = Arc::clone(&self.results_back);
        let generation = Arc::clone(&self.generation);
        let gen_snapshot = gen;

        self.pool.spawn(move || {
            // Bail if generation changed (player teleported).
            if generation.load(Acquire) != gen_snapshot { return; }
            let result = job.execute();
            results.lock().push(result);
        });
    }
}
```

**Collection** (main thread, start of frame, before upload):
```rust
fn collect_results(&mut self) {
    // Swap: grab completed results, give workers a fresh vec.
    // Lock held only for the pointer swap — O(1).
    {
        let mut back = self.results_back.lock();
        std::mem::swap(&mut self.results_front, &mut *back);
    }
    // Process unlocked — workers push to the new back vec freely.
    for result in self.results_front.drain(..) {
        // Liveness check: skip results for unloaded megas.
        if !self.loaded_megas.contains(&result.target()) {
            continue;
        }
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

**Backpressure**: `results_back` has a soft capacity check. If `results_back.len() > 256`, workers skip pushing and the result is discarded (the job will be re-dirtied and re-enqueued). This bounds memory growth if the main thread stalls.

## Memory Ordering

The generation counter uses `Acquire` on worker loads and `Release` on the main thread's `fetch_add`. This ensures that when a worker reads a generation value, it also sees the fully constructed job data that was prepared before the generation was published. `Relaxed` is incorrect here — on ARM or high-core-count systems, a worker could see the new generation but stale job fields.

```rust
// Main thread — teleport:
fn on_teleport(&self) {
    self.generation.fetch_add(1, Release);
    self.queue.lock().clear();
    self.pending.lock().clear();
}

// Worker thread — job execution:
fn execute(job: Job, generation: &AtomicU64, gen_snapshot: u64) -> Option<JobResult> {
    if generation.load(Acquire) != gen_snapshot { return None; }
    // Safe to read job data — Release/Acquire guarantees visibility.
    Some(job.run())
}
```

## Tier Budget

Proportional to available spare time, not fixed caps.

```rust
struct TierBudget {
    /// Fraction of spare time allocated to each tier. Must sum to 1.0.
    fractions: [f32; 6],
    /// Computed ms cap per tier for this frame (fractions × spare_ms).
    caps: [f32; 6],
    /// Estimated ms submitted so far this frame.
    spent: [f32; 6],
}

impl TierBudget {
    fn reset(&mut self, spare_ms: f32) {
        self.spent = [0.0; 6];
        // Compute per-tier caps proportional to this frame's spare time.
        for i in 0..6 {
            self.caps[i] = self.fractions[i] * spare_ms;
        }
        // Redistribute from empty tiers: scan for tiers with no pending work
        // and give their budget to the next lower tier.
        // (Caller passes pending-work-per-tier counts for this.)
    }
    fn redistribute(&mut self, has_work: [bool; 6]) {
        // Tiers with no work donate their cap downward.
        let mut surplus = 0.0;
        for i in 0..6 {
            self.caps[i] += surplus;
            surplus = 0.0;
            if !has_work[i] {
                surplus = self.caps[i];
                self.caps[i] = 0.0;
            }
        }
        // Any remaining surplus goes to last tier with work.
        for i in (0..6).rev() {
            if has_work[i] {
                self.caps[i] += surplus;
                break;
            }
        }
    }
    fn can_submit(&self, tier: u8, estimated_ms: f32) -> bool {
        self.spent[tier as usize] + estimated_ms <= self.caps[tier as usize]
    }
    fn charge(&mut self, tier: u8, estimated_ms: f32) {
        self.spent[tier as usize] += estimated_ms;
    }
}
```

Default fractions:
- Tier 0 (culling): 0.35
- Tier 1 (geometry): 0.25
- Tier 2 (LOD): 0.15
- Tier 3 (lighting): 0.10
- Tier 4 (sim support): 0.10
- Tier 5 (memory): 0.05

A frame with 8ms spare gives tier 0 = 2.8ms, tier 1 = 2.0ms, etc. A frame with 0.5ms spare gives tier 0 = 0.175ms (maybe one face mask job). If tier 0 has no work, its 2.8ms redistributes to tier 1.

## Starvation Prevention

**Age promotion**: Jobs queued for >100 frames get priority boosted by 5 per 100 frames. Promoted jobs consume a **dedicated promotion budget** — 10% of spare time — not the target tier's regular budget. This prevents a flood of promoted tier-4 jobs from starving fresh tier-2 work.

```rust
// In TierBudget, a 7th slot for promoted jobs:
promoted_cap: f32,   // 0.10 * spare_ms
promoted_spent: f32,
```

When a promoted job is popped, it checks `promoted_cap` instead of the target tier's cap.

## Cancellation

**Generation counter** (Acquire/Release): Incremented on teleport or camera discontinuity (>100 units in one frame). In-flight workers bail on stale generation. Queue and pending set are cleared.

**Mega unload liveness**: When results are collected, each result's target `MegaPos` is checked against `loaded_megas`. Results for unloaded megas are silently discarded. This handles the case where a mega is unloaded by normal camera drift while a nav mesh bake was in-flight.

## Dirty Coalescing

Two classes:

**Structural dirty** (voxel changed — visible immediately):
- Surface shell extraction
- Mip refinement
- Face mask update

Enqueued next frame, no settle delay. The player sees the edit within 1-2 frames.

**Cosmetic dirty** (appearance/support — can settle):
- AO bake, light flood, shadow cache, normal smoothing
- Nav mesh, structural graph, collision mesh
- LOD re-evaluation

These go through the `DirtyCoalescer` with a settle period of 30 frames (~0.5s). The coalescer tracks **last dirty frame** (not first), so the settle window restarts on each edit:

```rust
struct DirtyCoalescer {
    /// MegaPos → frame of most recent dirty event.
    last_dirty: HashMap<MegaPos, u64>,
    settle_frames: u64,  // 30
}

impl DirtyCoalescer {
    fn mark_dirty(&mut self, mp: MegaPos, current_frame: u64) {
        // Always update to latest frame — settle window restarts.
        self.last_dirty.insert(mp, current_frame);
    }

    fn drain_settled(&mut self, current_frame: u64) -> Vec<MegaPos> {
        let mut settled = Vec::new();
        self.last_dirty.retain(|mp, last| {
            if current_frame - *last >= self.settle_frames {
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

## Cooperative Yielding

Large jobs implement `IncrementalJob`:

```rust
enum StepResult {
    Continue(Box<dyn IncrementalJob + Send>),
    Done(JobResult),
}

trait IncrementalJob: Send {
    fn step(&mut self) -> StepResult;
}
```

A step targets ~0.3ms. Continuations are re-enqueued with `(priority, sequence_number)` ordering. The `BinaryHeap` is ordered by:

1. `priority` (ascending — lower = higher priority)
2. `distance_sq` (ascending — closer to camera first)
3. `sequence_number` (ascending — older jobs first)

This ensures continuations don't livelock: a continuation gets the same `(priority, distance_sq)` as the original job but an incrementing `sequence_number`, so fresh jobs at the same priority and distance interleave fairly.

```rust
impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
            .then(self.distance_sq.partial_cmp(&other.distance_sq).unwrap_or(Ordering::Equal))
            .then(self.sequence.cmp(&other.sequence))
    }
}
```

## Cache Budget

Fixed 64MB limit. LRU eviction by last-access frame.

| Data | Size per mega |
|------|--------------|
| Surface shell grid | ~10-50 KB (sparse, varies by surface area) |
| AO texture | 256 KB |
| Light levels | 256 KB |
| Column heights | 4 KB |
| Face masks | 1 byte |
| Height map | 16 KB |
| Nav mesh | ~10-50 KB |
| Structural graph | ~5-20 KB |

Worst case per mega: ~624 KB (with large shell + nav mesh). At 64MB budget: ~102 megas at full cache. Tight but workable. If profiling shows the shell grids are larger than expected, raise the budget to 96MB or evict shell grids more aggressively (they can be recomputed from the source chunks).

LRU eviction drops an entire mega's cache entries. The mega is re-dirtied (cosmetic) so its cache rebuilds on access.

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

**4. Interior cavity detection** (priority 7, ~1ms per mega, yielding per Z-layer)
Flood-fill from exposed faces. Sealed cavities flagged for skip. Feeds into culling — must be available early.

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
Simplified convex decomposition for physics. Note: NPCs use raw voxel grid queries (`get_voxel`, `surface_height`) for collision at all times. The collision mesh is a fast-path acceleration, not a prerequisite. NPCs never clip through terrain because the raw grid is always available and authoritative.

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
- Ray-march → mesh: quad count < 2000 AND stable > 50 frames AND `cooldown_remaining == 0`
- Mesh → ray-march: voxel edit in the mega (immediate)
- On revert: set `cooldown_remaining = 200` frames before the mega is eligible for mesh again

**Transition frame sequence** (edit to a meshed mega):
1. Frame N: voxel edit happens during sim tick
2. Frame N: `upload_megas()` detects the dirty source chunk, rebuilds the 64³ mega-grid, uploads new 3D texture
3. Frame N: renderer switches this mega from mesh to ray-march path (flag flip, no work)
4. Frame N: the new 3D texture renders via ray-march — correct geometry from this frame onward
5. Frame N+1: structural-dirty jobs enqueued (shell, mip, face mask) for acceleration
6. Frame N+200+: mesh re-evaluation eligible after cooldown

The key: `upload_megas()` already rebuilds the full 3D texture from source chunks on dirty. The ray-march path always has a correct texture. There is no gap where stale data renders.

**Coexistence**: Both paths write to the same G-buffer. The deferred lighting pass is agnostic.

**Implementation**: Phase 3. Requires a second graphics pipeline (vertex + fragment shaders writing to G-buffer).

## SVDAG

Deferred. At 64³ (4 octree levels), pointer overhead and traversal indirection likely cost more than the flat mip hierarchy. Do not implement until profiling shows ray traversal is the bottleneck, and benchmark against existing mips first.

## Implementation Order

**Phase 1 — Instrumentation + Framework**
1. Frame timing instrumentation (CPU stages + GPU profiling via RenderDoc)
2. `FrameOptimizer` struct with thread pool, Arc-shared results, generation counter (Acquire/Release)
3. Mandatory per-frame frustum cull + draw order sort
4. Double-buffered result collection with liveness check + backpressure cap
5. Dirty coalescer (structural immediate, cosmetic settle-after-last-edit)
6. Proportional tier budget with redistribution from empty tiers
7. Promoted-job budget (separate 10% slice)

**Phase 2 — Culling + Core Geometry**
8. Hierarchical occlusion culling
9. Face visibility masks
10. Empty mega flags
11. Interior cavity detection (feeds culling — must be early)
12. Surface shell extraction

**Phase 3 — Geometry + Hybrid Rendering**
13. Column height cache
14. Mip refinement
15. Greedy mesh builder + rasterization pipeline
16. Hybrid render path with hysteresis + cooldown

**Phase 4 — LOD**
17. LOD demotion/promotion
18. VRAM eviction improvements
19. Disk serialization (I/O thread)

**Phase 5 — Visual Quality**
20. AO bake
21. Light flood fill
22. Shadow cache
23. Normal smoothing

**Phase 6 — Sim Support + Memory**
24. Height map bake
25. Structural graph
26. Nav mesh bake
27. Texture atlas
