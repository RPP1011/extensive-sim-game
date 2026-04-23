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

## 6. Literature review — reading list

Grouped by the research question they bear on, with enough bibliographic
handle to look each one up (author, venue, year). Star next to the first
thing to read in each group. When a blog/talk is more useful than the
academic source, I've put the blog first and marked the paper as
"follow-up". Anything I'm unsure about is tagged `[verify]`.

### 6.1 Voxel data structures & raycasting

Question: can we cut per-pixel march cost without rewriting the whole
renderer? What do the state-of-the-art voxel engines actually store?

- ★ Laine & Karras, *"Efficient Sparse Voxel Octrees"*, I3D 2010 / IEEE
  TVCG 2011. Canonical SVO paper; contour voxels + beam optimisation.
  The beam-opt section is directly applicable to our raycaster.
- Kämpe, Sintorn, Assarsson, *"High Resolution Sparse Voxel DAGs"*,
  SIGGRAPH 2013. DAG compression of an SVO — orders of magnitude memory
  reduction for static scenes. Worth reading for the "is our `VoxelGrid`
  3D texture the right representation?" question.
- Villanueva, Marton, Gobbetti, *"Symmetry-aware Sparse Voxel DAGs"*,
  I3D 2016 / CGF 2017. Follow-up to SVDAG.
- Amanatides & Woo, *"A Fast Voxel Traversal Algorithm for Ray Tracing"*,
  Eurographics 1987. DDA traversal primitive. Short, foundational,
  probably already matches what `terrain_compute`'s shader does — useful
  to confirm.
- Revelles, Ureña, Lastra, *"An Efficient Parametric Algorithm for Octree
  Traversal"*, WSCG 2000. Octree variant of the above.
- 0fps blog — Mikola Lysenko, *"Meshing in a Minecraft Game"* series
  (parts 1 & 2, 2012). Greedy meshing reference; relevant if we ever
  consider a mesh path alongside the raycaster.
- John Lin's YouTube voxel-engine devlogs — brick-map layout, LOD
  cross-fades. Informal but practical. `[verify current channel]`
- Dennis Gustafsson, *"Teardown Teardown"*, GDC 2022. Streaming + voxel
  destruction at production scale. The streaming discussion is the part
  that applies to us.
- Douglas Dwyer, *"I spent 100 hours making the world's most beautiful
  voxel engine"* and follow-ups, YouTube 2023–2024. Specifically his
  argument for larger chunks when raycasting dominates.
- Alex Evans, *"Learning from Failure: A Survey of Promising, Unconventional
  and Mostly Abandoned Renderers for 'Dreams PS4'"*, SIGGRAPH 2015
  Advances course. SDF + point-splatting discussion; relevant as an
  alternate data structure path.
- Iñigo Quilez, *iquilezles.org* SDF articles. Reference for analytic
  distance functions if we move any terrain features to SDF evaluation.

### 6.2 GPU-driven culling & streaming

Question: what does the current per-frame mega cull cost, and what's the
ceiling if we move it GPU-side?

- ★ Karis, Stubbe, Wihlidal et al., *"A Deep Dive into Nanite Virtualized
  Geometry"*, SIGGRAPH 2021 Advances course. Section on cluster culling
  + persistent-thread traversal is the template for GPU-driven cull.
  Heavy read; skim the cull sections first.
- Losasso & Hoppe, *"Geometry Clipmaps: Terrain Rendering Using Nested
  Regular Grids"*, SIGGRAPH 2004. LOD via concentric rings; directly
  analogous to our radius-based chunk disk. GPU Gems 2, chapter 2 is
  the hands-on version.
- Mittring (Crytek), *"Advanced Virtual Texture Topics"*, SIGGRAPH 2008.
  Virtual-texture streaming logic; informs how we might prioritise
  chunk submissions by screen-space coverage rather than distance.
- Wihlidal, *"Optimizing the Graphics Pipeline with Compute"*, GDC 2016.
  Frostbite's GPU-driven cull, including the indirect-draw machinery
  we don't use yet.
- Haar & Aaltonen, *"GPU-Driven Rendering Pipelines"*, SIGGRAPH 2015.
  Ubisoft/Assassin's Creed version of the same idea, earlier and more
  accessible than Nanite.

### 6.3 GPU procedural generation

Question: the CPU `HALO=256` feature placement is the biggest
pre-generation cost; what's the right GPU architecture for scatter-style
feature placement?

- ★ Sebastian Lague, *"Coding Adventure: Compute Shaders"* and
  *"Coding Adventure: Terrain Generation"* YouTube series. Hands-on
  dispatch patterns that match `terrain_compute`.
- Ken Perlin, *"Improving Noise"*, SIGGRAPH 2002. Foundational; the hash
  we're using is a PCG-based replacement, but the gradient math is
  Perlin's. Compare against Stefan Gustavson's simplex noise notes for
  whether to switch.
- Gustavson, *"Simplex noise demystified"*, 2005 (PDF, linköping). Free
  and correct reference implementation; useful if we ever swap FBM
  primitives.
- Sean Murray, *"Building Worlds Using Maths(s)"*, GDC 2017 (No Man's
  Sky). Large-scale procedural world pipelines; talks through the
  prioritisation / budgeting problem specifically.
- Juniper / Atomontage blog posts on GPU scatter buffers `[verify]`.
  Less rigorous than academic sources but closer to our actual
  problem shape.
- Jump-Flood Algorithm, Rong & Tan, *"Jump Flooding in GPU with
  Applications to Voronoi Diagram and Distance Transform"*, I3D 2006.
  If we want GPU-side Voronoi for settlement / region placement
  without round-tripping to CPU.

### 6.4 Chunk streaming policies

Question: what's the right priority function when pool size < demand
(the `LOAD_RADIUS=704` overflow case)?

- Carmack, *"id Tech 5 MegaTextures"*, QuakeCon 2007 & 2008 keynotes.
  Disk-to-GPU streaming with priority queues; the original "stream
  only what's visible" architecture at game scale.
- van Waveren, *"The Asynchronous Texture Streaming System of Rage"*,
  2012 (Intel). Implementation details behind MegaTextures.
- Google / NASA tile-pyramid literature (Google Maps, WorldWind). Older
  than voxels but the priority heuristics — screen-space error,
  frustum coverage, predicted motion — translate directly.
- Lengyel, *"Transvoxel Algorithm"*, in *Game Engine Gems 3*, 2012.
  Seam handling between LOD levels; becomes relevant if we ever add a
  second LOD tier.

### 6.5 Determinism & lockstep simulation

Question: how far can we push the "CPU and GPU generate bit-identical
chunks from the same inputs" contract?

- ★ Glenn Fiedler, *gafferongames.com* — *"Deterministic Lockstep"* and
  *"Floating Point Determinism"*. Plain-English foundation; mostly about
  networking but the FP-determinism parts apply 1:1 to our CPU/GPU
  parity contract.
- Paul Tozour, *"The Age of Empires II AI"*, Game Programming Gems 3,
  2002. Early detailed write-up on lockstep determinism in a shipping
  RTS.
- Intel, *"Differences in Floating-Point Arithmetic Between Intel
  Architectures"* (whitepaper). Directly relevant to whether the same
  Rust code produces identical results across AVX-level differences.
- Kahan, *"How Futile are Mindless Assessments of Roundoff in
  Floating-Point Computation?"*, 2006. The grandfather paper; cite when
  someone proposes "just use f64, it's fine".
- GLSL / SPIR-V spec sections on floating-point determinism and the
  `RelaxedPrecision` flag. Worth confirming we're not accidentally
  letting the compiler reorder our hash.

### 6.6 Large-scale agent simulation

Question: how do other ~140-system games schedule work and what do they
do for dependency ordering?

- ★ Tarn Adams, multiple *Dwarf Fortress* talks and interviews on
  Roguelike Celebration / GDC. Informal but the system-ordering rants
  are unusually direct.
- Factorio FFFs — *#176 "Belt optimisations"*, *#204 "10 000 trains"*,
  *#244 "Multithreading"*. Data-oriented layout arguments plus concrete
  profiling numbers.
- Mike Acton, *"Data-Oriented Design and C++"*, CppCon 2014. The talk
  that made DOD canon; applies regardless of ECS choice.
- Richard Fabian, *Data-Oriented Design* (book, 2018, free online at
  dataorienteddesign.com). Long-form version of the same ideas.
- Christian Gyrling, *"Parallelizing the Naughty Dog Engine Using
  Fibers"*, GDC 2015. Fiber-based job system; worth reading even if we
  stick with rayon, because it covers the dependency-DAG scheduling
  problem we're about to hit if more systems parallelise.
- Aras Pranckevičius, *"'Entities' Redux — Data Oriented Tech Stack"*
  and the Unity DOTS talks. Closer to our layout since we're not using
  bevy_ecs either.
- Orkin, *"Three States and a Plan: The AI of F.E.A.R."*, GDC 2006. GOAP
  origin paper; our `src/ai/goap/` implements this. Good to re-read
  when scaling GOAP to hundreds of agents.
- Dave Mark, *"Improving AI Decision Modeling Through Utility Theory"*,
  GDC 2010. Utility-AI counterpoint to GOAP; useful framing when
  deciding which systems should override which.

### 6.7 Profiling & measurement methodology

Question: what tools exist between "`perf record`" and "Tracy" and how
do we actually diagnose a 20× `set_title` regression?

- ★ Brendan Gregg, *Systems Performance* (book, 2nd ed., 2020). The
  reference. Chapters on off-CPU analysis and flame graphs are the ones
  we need now.
- Gregg's blog — *"The PMCs of EC2"*, *"Off-CPU Analysis"*,
  *"Flame Graphs"*. Free, skimmable versions of the book chapters.
- Fabian Giesen, *"A Trip through the Graphics Pipeline 2011"* (blog
  series, ryg). Mental model for what the GPU is actually doing when
  we `vkQueueSubmit`.
- Bartosz Taudul, *Tracy manual* (PDF, shipped with Tracy). Not a paper
  but the authoritative explanation of frame markers, plots, GPU zones.
- Agner Fog, *"Optimizing software in C++"* + microarchitecture manuals.
  For the L1i / branch-predictor questions raised by the `FRAME_BATCH`
  ceiling.
- Denis Bakhvalov, *Performance Analysis and Tuning on Modern CPUs*
  (book, 2020, free online). Practical guide to `perf`, `toplev`,
  top-down methodology. Bridges Gregg (system) and Fog (micro).
- Linux `perf` wiki — *"Tutorial"* and *"Top-down Microarchitecture
  Analysis"* pages. Command-line recipes for the events we'd want.

### 6.8 Compositor / windowing performance

Question: why does dropping `set_title` cause a 20× regression? This
might be niche enough that no paper exists, but there's domain
literature.

- Wayland / wlroots architecture docs — frame-callback + commit
  scheduling. Explains what a compositor expects from a well-behaved
  client.
- Pekka Paalanen, *"Wayland explained, part 1–3"*, 2016. Blog series
  covering frame callbacks specifically.
- X.org EWMH / ICCCM specs — what `_NET_WM_NAME` changes actually
  signal to the WM. If the hypothesis is "compositor priority-boosts
  active clients", the protocol docs will confirm or deny.
- winit issue tracker — search for "priority", "throttle",
  "about_to_wait". Several existing issues touch on event-pump
  scheduling on Linux.

### 6.9 Adjacent but probably out of scope

Listed so we don't rediscover them by accident.

- Neural terrain generation (NVIDIA's *GAN-based* landscape papers,
  *Infinite Nature* etc.). Probably not worth it for our art style;
  procedural gets us there cheaper.
- Euclideon / Atomontage marketing material. Historically over-claims;
  the technically interesting parts are covered by SVDAG papers above.
- Mesh shaders (Vulkan 1.3 / VK_EXT_mesh_shader). Relevant only if we
  add a mesh path; our raycaster side-steps the geometry pipeline
  entirely.

---

### How to actually run the review

1. Pull PDFs for every ★ entry into `docs/references/` (already has
   `elit_part1.pdf`, `elit_part2.pdf` — use that pattern).
2. For each ★, write a 5-line note: problem, technique, applicability
   to our codebase, decision (adopt / adapt / reject / defer).
3. Promote the non-rejected ones into issues tagged `research-adopt`,
   one per paper. Then §7 of this doc can be re-ranked against them.

A single-day pass through the ★ list is the minimum viable literature
review for this branch. Extending to everything else is probably a
week of evenings.

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
