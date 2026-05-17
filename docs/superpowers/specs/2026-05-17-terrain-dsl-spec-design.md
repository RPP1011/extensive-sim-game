# Terrain DSL Spec — Design

**Status:** Design — awaiting user review before plan write-up.
**Date:** 2026-05-17
**Companion plan (TBD):** `docs/superpowers/plans/2026-05-1x-terrain-dsl-*.md` (created next via writing-plans).

## Summary

Add a new top-level `terrain { ... }` block to the `.sim` DSL. Each `.sim`
file optionally declares its terrain — a 3D voxel volume with biome /
material regions, walkable masks, and composable generation primitives
(noise heightfields, AABB shapes, carved caves, prefab refs). The
`dsl_compiler` lowers and emits a deterministic `generate_terrain(seed)`
function per opted-in runtime crate. Generation runs on a worker thread
at world start so the app stays responsive. Output plugs into the
existing `engine::terrain::TerrainQuery` seam via `engine_voxel`.

## Motivation

Today the sim engine has a `TerrainQuery` trait with a `FlatPlane`
default and an `engine_voxel` adapter wrapping the external
`voxel_engine` crate. There is no declarative way to specify terrain
inside a `.sim` file — runtimes hard-code their terrain choice, which
makes per-sim variety expensive and breaks the project's
compiler-first philosophy (P1).

The motivation is three-fold (from brainstorming):

1. **Per-sim terrain in `.sim` files** — each fixture should declare its
   own terrain alongside its agents, rules, events.
2. **Hand-authored terrain assets** — explicit AABBs, prefab `.vox`
   refs, walkable masks. Repeatable test fixtures.
3. **Generation algorithm tuning** — expose noise / region params as
   data, no Rust edits to retune.

## Design

### Architecture & integration

- New top-level `terrain { ... }` block in `.sim`, alongside `agent`,
  `rule`, `event`. Parsed by `dsl_ast`, lowered and emitted by
  `dsl_compiler`.
- Annotated `@cpu_only` per P3 (parser-recognised exception). GPU
  generator emitter is explicit future work.
- Per-runtime `build.rs` consumes the compiled artifact and writes
  `terrain_gen.rs` into `OUT_DIR`, exposing a single function:

  ```rust
  pub fn generate_terrain(world_seed: u64) -> engine_voxel::VoxelTerrain;
  ```

- The runtime calls this once at world start (off-thread; see *Async
  execution*), wraps the result in
  `Arc<dyn TerrainQuery + Send + Sync>`, and installs it into
  `SimState` via the existing setter.
- `.sim` files without a `terrain {}` block keep their current
  behaviour (`FlatPlane` default).

### DSL surface

Fixed grid extent + ordered list of layer primitives. Layers apply in
source order; later layers overwrite earlier ones at conflicting cells.

```text
terrain {
  extent: (128, 128, 64)        // cells along x, y, z
  cell_size: 0.25                // metres per cell (declarative;
                                 // engine_voxel keeps 1-unit cells,
                                 // we map at boundary)
  seed_purpose: 0xBA5E_7E55      // any non-zero u32 tag fed to
                                 // per_agent_u32 for keyed PCG. Must
                                 // be distinct from other purposes
                                 // used in the world. Parser-enforced
                                 // non-zero.

  materials {
    grass { id: 1, walkable: true,  hardness: 1, biome_tag: forest,
            color: 0x4A8B3A }
    stone { id: 2, walkable: false, hardness: 8, biome_tag: rock,
            color: 0x808080 }
    sand  { id: 3, walkable: true,  hardness: 2, biome_tag: desert,
            color: 0xD9C28A, movement_cost: 1.5 }
  }

  layer fill        { material: grass }
  layer heightfield { noise: perlin(octaves: 4, freq: 0.04),
                      amplitude: 12, base_z: 8, fill_below: stone }
  layer carve_caves { noise: worley(freq: 0.08), threshold: 0.35 }
  layer box         { aabb: ((10,10,8)..(20,20,18)), material: stone }
  layer prefab      { file: "assets/prefabs/tower.vox",
                      at: (40, 40, 8) }
  layer walkable_mask { region: ((0,0,0)..(128,128,1)),
                        walkable: false }
}
```

Primitives in v1:

| Primitive | Purpose |
|---|---|
| `fill` | Uniform material across whole extent. |
| `heightfield` | Noise-driven ground surface; fills `[0..h]`. |
| `box` / `sphere` / `cylinder` | Explicit AABB / shape placement. |
| `carve_caves` | Subtract solid cells where noise > threshold. |
| `prefab` | Splat a `.vox` file at a position (no rotation in v1). |
| `region` | Tag an AABB with a biome id (does not change solidity). |
| `walkable_mask` | Force walkable / blocked over a region, independent of solidity. |

Noise sources in v1: `perlin`, `worley`, `value`. Each takes
deterministic params (octaves, freq, lacunarity, gain).

### Material properties

`materials { ... }` block declares per-id property rows:

| Property | Type | Default | Used by |
|---|---|---|---|
| `id` | u8, 1..=255, unique | — | Stored in `VoxelGrid` cell |
| `walkable` | bool | true | `TerrainQuery::walkable` short-circuit |
| `hardness` | u8 | 1 | Future `Harvest` cost / destructibility |
| `biome_tag` | ident | none | Region-of-effect rules |
| `color` | rgb hex | gray | Viewer rendering |
| `movement_cost` | f32 | 1.0 | Future pathing weight |

`id` is the byte stored in each voxel cell (engine_voxel uses non-zero
u8 with low bits as material code today; the compiler enforces
uniqueness and the non-zero range). Materials are immutable for the
sim's lifetime; in-game mutation (burnt-down forest, etc.) rides the
existing event channels, not the materials table.

Materials are stored alongside the grid in the `VoxelTerrain` and
accessed as `terrain.materials.get(cell)`. **Exposing material lookups
to DSL rules is deferred** — v1 rules see only the existing `walkable()`
/ `height_at()` / `line_of_sight()` surface.

### Determinism & seed plumbing

- Seed source: runtime passes `world_seed: u64` into
  `generate_terrain(world_seed)` — same seed used by agent RNG, so
  replay with the same seed is bit-identical.
- Per-layer keying:

  ```rust
  per_agent_u32(world_seed, layer_index as u32, cell_index as u32,
                seed_purpose)
  ```

  - `layer_index` is the source-order index of the layer (1-based).
  - `seed_purpose` is the `terrain { seed_purpose: ... }` value.
- Re-ordering layers re-keys noise (two `heightfield`s in the same file
  don't collide). Changing `seed_purpose` re-rolls terrain without
  touching the world seed.
- No floating-point reduction hazards (P11): every cell is written from
  a deterministic per-cell expression — no atomic-append, no cross-cell
  float reductions, so the P11 sort-then-fold rule is N/A.
- No P5 escape hatches: emitter rejects layer helpers that would need
  `thread_rng`, `SystemTime`, or non-keyed entropy. The existing
  ast-grep rule already covers emitter output.
- Known constraint: `engine`'s PCG uses `ahash`
  (memory `project_engine_pcg_ahash_drift.md`), which is
  toolchain-unstable. Terrain inherits this. **Do not pin
  terrain-output byte-goldens until ahash drift is fixed.**
  Determinism tests in v1 use within-process re-runs, not
  committed byte strings.

### Output format & engine wiring

Each `.sim` with a `terrain {}` block produces one Rust module in the
runtime crate's `OUT_DIR`:

```rust
// OUT_DIR/terrain_gen.rs (one per runtime)
use engine_voxel::{VoxelTerrain, VoxelGrid, MaterialTable};

pub const EXTENT:    (u32, u32, u32) = (128, 128, 64);
pub const CELL_SIZE: f32              = 0.25;

pub static MATERIALS: MaterialTable = MaterialTable::new(&[
    // (id, walkable, hardness, biome_tag, color_rgb, movement_cost)
    (1, true,  1, BIOME_FOREST, 0x4A8B3A, 1.0),
    (2, false, 8, BIOME_ROCK,   0x808080, 1.0),
    (3, true,  2, BIOME_DESERT, 0xD9C28A, 1.5),
]);

pub fn generate_terrain(world_seed: u64) -> VoxelTerrain {
    let mut grid = VoxelGrid::new(EXTENT.0, EXTENT.1, EXTENT.2);
    // layer 1: fill
    layer_fill(&mut grid, /*material=*/ 1);
    // layer 2: heightfield
    layer_heightfield(&mut grid, world_seed, /*layer_idx=*/ 2,
                      /*purpose=*/ 0xBA5E_7E55, /*params*/ &HF_PARAMS_2);
    // ...
    VoxelTerrain::from_parts(grid, &MATERIALS, CELL_SIZE)
}
```

Each layer compiles to a `layer_<kind>` helper with a const param
struct. Helpers live in a new `engine_voxel::terrain_layers` module —
hand-written once, shared across all generated terrains. This is the
only hand-written code added by this slice, and it is library code
(not rule logic), so P1 is not in tension.

Runtime wiring:

```rust
let handle = TerrainGenHandle::spawn(cfg.world_seed,
                                     generated::generate_terrain);
// ... app loop ticks; renders "Generating world..." UI ...
let terrain = handle.block_until_ready()?; // or poll try_take()
sim_state.set_terrain(Arc::new(terrain));
```

`SimState::set_terrain` already exists on the `TerrainQuery` seam in
`engine/src/terrain.rs`. No engine schema-hash bump needed in v1
because no new `SimState` SoA field is added; terrain lives on the
existing `TerrainQuery` Arc slot.

| Path | Status | Purpose |
|---|---|---|
| `crates/dsl_ast/src/terrain.rs` | new | AST nodes for `terrain {}`, layers, materials |
| `crates/dsl_compiler/src/cg/lower/terrain.rs` | new | Lower + validate terrain block |
| `crates/dsl_compiler/src/cg/emit/terrain.rs` | new | Emit `generate_terrain` + layer calls |
| `crates/engine_voxel/src/terrain_layers.rs` | new | Hand-written layer helpers |
| `crates/engine_voxel/src/materials.rs` | new | `MaterialTable` const-friendly struct |
| `crates/engine/src/terrain.rs` | edit | No surface change required for v1; extension point reserved |
| `crates/*_runtime/build.rs` | per-opt-in | Read compiled artifact, write `terrain_gen.rs` |

### Async execution model

Terrain generation runs on a dedicated worker thread, not the main
app/sim thread. App stays responsive; generation failures degrade to
errors, not panics.

```rust
// crates/engine_voxel/src/terrain_gen.rs
pub struct TerrainGenHandle {
    rx:   std::sync::mpsc::Receiver<Result<VoxelTerrain, TerrainGenError>>,
    join: Option<std::thread::JoinHandle<()>>,
    seed: u64,
}

impl TerrainGenHandle {
    pub fn spawn(seed: u64, gen_fn: fn(u64) -> VoxelTerrain) -> Self;
    pub fn try_take(&mut self)
        -> Option<Result<VoxelTerrain, TerrainGenError>>;       // non-blocking
    pub fn block_until_ready(self)
        -> Result<VoxelTerrain, TerrainGenError>;               // for tests
}

pub enum TerrainGenError {
    Panicked(String),
}
```

Runtime flow:

1. App boot → `TerrainGenHandle::spawn(seed, generate_terrain)`. App
   renders UI, loads other assets concurrently.
2. Per frame: `handle.try_take()`. While `None`, display
   "Generating world…".
3. On `Some(Ok(terrain))` → install `Arc<VoxelTerrain>` into
   `SimState`, transition to ticking.
4. On `Some(Err(e))` → surface the error in UI; **app does not
   panic.** Worker wraps generation in `std::panic::catch_unwind`;
   panics become `TerrainGenError::Panicked(msg)`.

Constitution safety:

- **P5:** seed-keyed PCG is thread-of-execution-independent; off-thread
  generation does not change output.
- **P10 (no panic on deterministic path):** generation runs *outside*
  the deterministic tick path. `catch_unwind` degrades a layer bug to
  an error.
- **No tearing:** `TerrainQuery` slot starts as
  `Arc::new(FlatPlane)` and is replaced atomically on terrain ready.
  Ticks that fire before terrain lands see `FlatPlane` (current
  behaviour). Most runtimes will gate tick start on readiness, but
  the engine does not enforce that.

Cancellation: v1 does not support cancelling an in-flight generation.
If the user closes the world before terrain finishes, the worker runs
to completion and the result is dropped. Acceptable since generation
is bounded by extent size (no infinite loops by construction).

## Testing strategy

| Test | Location | What it pins |
|---|---|---|
| Parser fixtures | `crates/dsl_compiler/tests/terrain_parse_*.rs` | Each primitive parses; bad inputs (zero `seed_purpose`, duplicate material id, out-of-bounds AABB) produce specific errors. |
| Lowering | `crates/dsl_compiler/tests/terrain_lower_*.rs` | Layer order preserved; material table resolves; layer indices stable. |
| Emit golden | `crates/dsl_compiler/tests/terrain_emit_golden.rs` | Generated Rust matches a checked-in golden — catches accidental codegen churn. |
| End-to-end determinism | `crates/engine_voxel/tests/terrain_determinism.rs` | Two calls with same seed → byte-identical `VoxelGrid`. Different seeds → different output. Within-process re-runs, not committed byte strings (ahash drift). |
| Threading equivalence | `crates/engine_voxel/tests/terrain_async.rs` | Two separate `TerrainGenHandle::spawn(seed, ...).block_until_ready()` calls with the same seed produce byte-identical output (proves the worker-thread boundary does not perturb determinism). |
| Panic containment | `crates/engine_voxel/tests/terrain_panic_safety.rs` | A layer helper that panics surfaces as `TerrainGenError::Panicked`; test process does not abort. |
| Smoke runtime | one opted-in `*_runtime` (probably `voxel_runtime` or a fresh `terrain_probe_runtime`) | Starts, generates, ticks one step against the produced terrain, exits clean. |

No P3 parity test in v1 — terrain is `@cpu_only`.

## Out of scope (v1)

- **Multi-file terrain inputs.** User explicitly flagged this as a
  follow-up design. Stays single-file in v1.
- **Automatic schema-hash regeneration.** Separate build-infra
  concern; v1 does not require a schema-hash bump.
- **GPU generation emitter.** DSL surface designed so a GPU emitter
  can be added without breaking source files. CPU-only in v1.
- **Material property access from DSL rules.** v1 stores the
  materials table but does not expose `materials.get(cell)` to rules.
- **Run-time-mutable terrain layers.** Existing event channels
  (`PlaceVoxel`, `Harvest`) handle per-tick mutation; the DSL spec is
  initial-state only.
- **Layer rotation, scale, fractal warping.** Future primitive
  extensions.
- **Cancellation of in-flight generation.** Worker runs to completion.
- **Chunked / lazy generation.** v1 generates the full extent eagerly
  on the worker thread. Chunking is a future scaling step.
- **Pinning terrain byte-goldens.** Blocked on the ahash drift
  fix (`project_engine_pcg_ahash_drift.md`).

## Known risks

- **ahash determinism drift.** `engine`'s PCG uses `ahash`, which is
  not stable across rustc versions. Terrain output will drift if the
  toolchain bumps even though the source `.sim` is unchanged.
  Mitigation: within-process re-run tests, not byte goldens. Long-term
  fix is upstream (replace ahash with a stable hash).
- **Worker thread vs `Send` requirements.** `gen_fn: fn(u64) -> ...`
  is a plain fn pointer to avoid `Send`-bound surprises. Closure
  capture is intentionally not supported in v1.
- **engine_voxel u8 cell encoding.** Material `id` overlaps existing
  per-kind material codes used by `PlaceVoxel` / `Harvest` consumers.
  Compiler must reserve / coordinate the id space to avoid collisions
  with the existing kind_hash low-bits convention
  (memory `project_aoe_gotchas.md` for context).

## Constitution touchpoints (for plan AIS)

- **P1 (Compiler-First):** PASS — terrain generation is emitted, not
  hand-written. Layer helpers are library code (P1 scope is rule
  logic), kept under `engine_voxel::terrain_layers`.
- **P2 (Schema-Hash):** N/A in v1 — no new `SimState` SoA fields.
- **P3 (Cross-Backend Parity):** PASS — uses the parser-recognised
  `@cpu_only` exception. GPU emitter is explicit future work.
- **P4 (`EffectOp` Size):** N/A — no new `EffectOp` variants.
- **P5 (Determinism via Keyed PCG):** PASS — all entropy flows
  through `per_agent_u32(world_seed, layer_index, cell_index, seed_purpose)`.
  Known ahash drift inherited; mitigation documented.
- **P6 (Events Are the Mutation Channel):** N/A — terrain init runs
  before tick 0; per-tick mutation continues via existing
  `PlaceVoxel`/`Harvest` events.
- **P7 (Replayability Flagged):** N/A — no new event variants.
- **P8 (AIS Required):** the implementation plan will include the
  full AIS template; this design doc summarises the touchpoints.
- **P10 (No Runtime Panic):** PASS — `catch_unwind` in worker
  thread; terrain init is off the deterministic tick path.
- **P11 (Reduction Determinism):** N/A — no atomic-append, no float
  reductions in generation.
