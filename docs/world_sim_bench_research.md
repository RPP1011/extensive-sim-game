# World Sim Bench — Research Directions

Scratch sheet for what to look into next on this branch. Not a plan — a reading
list + open questions organised by where the time is going right now. When one
of these turns into a concrete plan, promote it to `docs/superpowers/plans/`.

**Branch:** `claude/world-sim-research-doc-*` (formerly the `--world small` +
chunk-pump optimisation line of work).

**Entry points:**
- `cargo run --features app -- world-sim --render --world small` — fixed 9³
  chunk forest, the iteration target for everything in this doc.
- `src/world_sim/voxel_app.rs` — the bench / render loop. `run_batch` is the
  hot outer loop; `run_frame` the per-frame body; `fire_fps_log` the 1 Hz
  telemetry.
- `src/world_sim/constants.rs` — `CHUNK_SIZE=64`, `MEGA=1`, `LOAD_RADIUS=704`,
  render 480×270 blit-upscaled to 1280×720.
- `voxel_engine` crate at `/home/ricky/Projects/voxel_engine` — the Vulkan
  renderer + `terrain_compute` pipeline. *Off-tree dependency; read-only from
  here unless we upstream.*

---

## 1. Where the time actually goes

Before picking experiments, re-derive the breakdown on the branch as it
stands. The `[perf]` line already prints it; what's missing is a non-flaky,
comparable number across commits.

- [ ] Add a `--bench N-seconds` mode to `world-sim --render` that prints a
  single post-run summary (median frame, p95, pool stats, gen_submitted total,
  drained total). Right now we eyeball 1 Hz logs, which makes A/B hard.
- [ ] Dump the same numbers in JSON so a script can plot commit → FPS and
  flag regressions like the ones already logged in commit messages
  (`FRAME_BATCH=16384 stalled`, `PUMP_EVERY=32`, `set_title` 20× regression).
- [ ] Record traces with `perf record --call-graph dwarf` on a fixed 30 s
  small-world run. `perf.data` already exists in the tree from an older run —
  replace it or gitignore it. `samply` or `hotspot` give nicer views than
  `perf report`.
- [ ] Try `tracy` (`tracing-tracy` crate) for a per-frame flame timeline.
  The current EMA perf log aggregates too much to catch a single 5 ms stall.

Open question: the fast path reports ~2 µs per batch of 8192 "frames", i.e.
a virtual ~4 G FPS. **How much of that is a real frame and how much is a
batch-level short-circuit?** Worth documenting the difference in the bench
output so we don't chase ghost gains.

---

## 2. Chunk generation pipeline

This is the real work when the camera actually moves. The branch's recent
commits have been chipping at it (CPU/GPU halo parity, HALO=75 vs 130, LRU
monotonic frame counter, disk-load short-circuits).

### 2.1 CPU vs GPU parity
- `gpu_terrain_parity.rs` test exists — confirm it covers all biome
  combinations we actually generate, not just the default one. Features are
  the most likely place parity drifts (`features.rs:681` comment warns about
  halo lockstep).
- The hash function was ported to u32-only for GLSL compatibility
  (`docs/superpowers/plans/2026-04-09-gpu-terrain-generation.md`). Verify
  determinism tests still pass across GPU vendors — we've only tested on
  the dev machine.

### 2.2 Halo cost
- `HALO=256` voxels for CPU feature placement is what makes pre-generation
  unusable in debug (`build_small_world` skips it explicitly). Is 256
  actually necessary, or does tree-canopy radius + longest river width cap
  at ~130? A smaller halo shrinks each chunk's candidate set ~4× (area).
- Alternative: move feature placement entirely GPU-side and kill the
  CPU halo. Needs a scatter buffer for variable-size feature lists.

### 2.3 Pool behaviour
- `NUM_SLOTS=1024`, `LOAD_RADIUS=704` → 11-chunk horizontal radius × 5 z
  layers ≈ 1900 > 1024. Overflow is intentional (spiral + LRU cycles the
  outer shell), but worth measuring: how often is a visible chunk evicted
  and re-submitted in a 30 s small-world session?
- The commit `make frame_count monotonic to unbreak LRU` suggests LRU
  corners are still load-bearing. Add an invariant test: after N batches on
  a stationary camera, no chunk in the view frustum should have been
  re-submitted.

### 2.4 Streaming pattern
- Today: 8-chunk disk spiral around camera, ignore GPU frustum cull during
  submission. Look at Teardown / Minecraft Bedrock talks for alternatives —
  prioritising by screen-space coverage instead of raw distance tends to win
  when vertical FOV is narrow.

---

## 3. Render loop

### 3.1 The `set_title` mystery (docs/f3c2807)
Removing the 1 Hz `window.set_title` call caused a 20× FPS regression. That's
not a small oversight — the compositor is almost certainly priority-boosting
us for being "active". Worth confirming:
- Does it reproduce on a different WM (gnome-shell vs sway vs kwin)?
- Is there a cleaner way to hint activity (e.g. `wl_surface.frame` callback
  or `XSetWMHints`) that doesn't spam a cosmetic IPC?
- Does it also regress on Windows/macOS if we ever port?

Until we know, keep the `set_title` as-is, but promote the comment block at
`voxel_app.rs:1281-1286` into an architectural note in this doc so we don't
lose the context.

### 3.2 Frame batching ceiling
`FRAME_BATCH=8192` is the current sweet spot. 16384 regressed (recorded in
commit `a3ab58a`). Open questions:
- Why does the non-stable branch double in wall time at 16384? The commit
  blames i-cache, but we haven't actually measured L1i misses
  (`perf stat -e L1-icache-load-misses`).
- Is the "virtual frame" framing even useful past 1 M FPS? We're measuring
  loop overhead, not render work — maybe report `loop_ns/iter` instead of
  fake FPS.

### 3.3 Raycasting cost
`ema_raycast_ms` in the perf log is the per-pixel march. 480×270 = 129 k
rays. Relevant knobs we haven't touched:
- Step size / adaptive DDA termination.
- A coarse pre-pass at 120×67 to kill empty-sky pixels cheaply.
- Reprojection from the previous frame for static cameras (we already
  short-circuit on `camera_version`, but the raycaster still re-runs).

### 3.4 Culling
`last_cull_cam_key` was replaced by `camera_version` for O(1) stability. The
actual cull still walks all loaded megas every frame the camera moves.
Investigate:
- Loose octree / BVH over mega positions. Overkill for 8-radius, probably
  justified if we raise `LOAD_RADIUS`.
- GPU-driven culling: push mega AABBs to a storage buffer and cull in
  compute, matching what the raycaster already reads.

---

## 4. Sim side (not the renderer)

`tick_sim` at `voxel_app.rs:1415` runs `self.sim.tick()` at `SIM_BASE_HZ=10`
with a burst cap of 4 per batch. On the render branch the sim is mostly
invisible — `ema_tick_sim_ms` is tiny because very little happens in
small-world. That makes it a bad benchmark for the *overall* world sim.

- [ ] Second bench mode: `--bench headless`, which drops the renderer and
  measures `sim.tick()` throughput with a richer preset (default `--rich
  --entities 2000`). The render-loop story and the sim-cost story are
  different optimisation targets and they keep getting conflated under
  "world sim bench".
- [ ] The ~140 systems in `src/world_sim/systems/` aren't individually
  profiled. Add a `--features profile-systems` gate (already declared in
  `Cargo.toml`) that wraps each system in a span and dumps a sorted
  top-20 list.
- [ ] Serial-only today. `--parallel` flag exists but I don't know if any
  system actually uses rayon. Quick audit: grep `par_iter` in `systems/`.

---

## 5. World content / test scene

`--world small` (9³ chunks, 1 settlement, 10 NPCs) was added for NPC
iteration. For rendering bench purposes it's almost too small — the whole
world fits in the pool so eviction pressure never triggers.

- [ ] Add `--world medium` (say 32³) that actually forces streaming. Without
  it we can't benchmark the thing that used to dominate frame cost.
- [ ] Add a camera-path recorder / replayer. Today the camera is driven by
  held keys, so batches alternate between stable and non-stable based on
  whether I touched the keyboard. Deterministic camera paths would make
  bench numbers repeatable.

---

## 6. External references worth reading

Bucketed by where they'd apply:

**Voxel rendering & streaming**
- John Lin's voxel engine devlogs (brick-maps, sparse voxel DAGs).
- Douglas Dwyer's 8-bit voxel talk — in particular the argument for
  128³ chunks over 64³ when raycasting dominates.
- Teardown GDC talk on chunk streaming (MagicaVoxel-style, but at scale).
- *Nanite* paper sections on cluster culling — relevant to GPU-driven cull
  even though we're not mesh-based.

**GPU terrain generation**
- Sebastian Lague "Coding Adventure: Terrain Generation" series — the GPU
  dispatch patterns are close to what `terrain_compute` does.
- `wgpu` compute shader examples; we're on `ash` but the dispatch sizing
  lore is shared.

**Large simulation loops**
- Dwarf Fortress dev logs on the ~140-system ordering problem. Ours is
  smaller in count but has the same "which system wakes which" headache.
- Factorio blogs on parallelism (FFFs #176 and #244) — ECS-style data
  layout arguments, most of which apply whether or not we use bevy_ecs.

**Measurement**
- `tracy` Rust integration guide (`tracing-tracy`).
- Brendan Gregg — "Flame Graphs" and "Off-CPU Analysis". The
  `epoll_wait` sleep inside `pump_app_events` is an off-CPU blocker we've
  never characterised.

---

## 7. Experiments I'd try first (ranked)

1. **Headless bench harness.** Without a deterministic, scripted run, every
   other item on this list is unmeasurable. ~half a day.
2. **Remove the CPU halo entirely** (feature placement on GPU only). Biggest
   gen-time win if it lands; needs a scatter buffer design.
3. **128³ chunks instead of 64³.** Halves the pool pressure at the same
   load radius and scales better for the raycaster. Risk: `VoxelGrid`
   texture dimensions, GPU memory. Worth one exploratory spike.
4. **Tracy integration.** Cheap to add, answers five other questions on
   this list once it's in.
5. **Coarse raycast pre-pass.** Low-risk pixel saver, independent of
   everything else.
6. **Camera-path replay.** Bench determinism + regression detection.

Anything below this line is speculative until we have (1) and (4).

---

## 8. Things NOT to touch until measured

- LTO mode (already thin; fat hurt the hot loop at 20 M+ FPS per the
  Cargo.toml comment).
- `FRAME_BATCH` size (8192 is pinned; two earlier attempts regressed).
- `set_title` (see §3.1).
- `Instant::now` inside `run_batch` (removing it 2-3× regressed — it acts
  as an inadvertent memory barrier, commit b835fe1).

Each of these has a commit-message explanation attached; if you need to
revisit, read the commit first so you know what the last person found.
