# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this crate does

Adapter crate bridging `voxel_engine`'s `VoxelGrid`/raycast to the sim engine via the `TerrainQuery` seam (`crates/engine/src/terrain.rs`). `VoxelTerrain` wraps `voxel_engine::voxel::grid::VoxelGrid` (a dense flat `Vec<u8>`, `0` = air) and implements the 3 `TerrainQuery` methods: `height_at`, `walkable`, `line_of_sight` (the last via `voxel_engine::voxel::raycast::ray_cast_grid`). Cells are 1 world-unit cubes; negative/out-of-range coords behave as empty (same defaults as the engine's `FlatPlane`).

Per `docs/superpowers/plans/2026-05-09-voxel-engine-integration.md` (status: **COMPLETE**, all 5 phases shipped, PR train #69–#72 + one more), this crate today implements:

- **Phase A** — CPU-side `VoxelTerrain` + `TerrainQuery` impl.
- **Phase B** — `VoxelTerrain::apply_voxel_chronicle_record` — drains engine chronicle records for `EffectPlaceVoxelApplied` (kind=60) and `EffectHarvestApplied` (kind=59) into grid mutations. Caller resolves `caster_slot → Vec3` and passes the position in (`engine_voxel` has no `engine::SimState` dep by design — see "Caller signature: option (A)" in `src/lib.rs`'s crate doc-comment).
- **Phase C** — `VoxelMirror`, a `wgpu::Buffer` GPU-resident mirror of the grid (one `u32` per cell, chunked dirty-tracking via `BTreeSet<ChunkCoord>` for deterministic upload order). `apply_voxel_chronicle_record_with_mirror` mutates the CPU grid AND marks touched chunks dirty; `VoxelMirror::flush_dirty` pushes dirty chunks to GPU.
- **Phase D/E** (WGSL emit for terrain queries, production fixture opt-in) are described in the plan but their code lives outside this crate (compiler emit + `wave_defense_runtime`) — don't expect to find them under `src/`.

Beyond the 5-phase plan, the crate has grown a few more modules not covered by that doc (read their own header doc-comments for design rationale, they're thorough):

- `region.rs` — `VoxelRegionRegistry`: generational-id (`gen<<32 | slot`) registry of named `VoxelRegion` volumes (`Aabb` or `ChunkSet` bounds), per spec `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md` §6.1.
- `navgrid.rs` — `build_navgrid`: per-region 2D walkability + height index (packed `u32` per cell: walkable bit + height + reserved bits), per spec §7.2.
- `materials.rs` — `MaterialTable`/`MaterialRow`: const per-material property table (walkable, hardness, biome tag, color, movement cost), populated by the DSL emitter (T8) and stored on `VoxelTerrain` via `with_extent_and_materials`.
- `terrain_gen.rs` — `TerrainGenHandle`: worker-thread harness for off-tick terrain generation (`fn` pointer generator, panics caught via `catch_unwind` and surfaced as `TerrainGenError::Panicked`).
- `terrain_layers.rs` — hand-written layer helpers (currently just `layer_fill`) called by emitter-generated `generate_terrain(seed)` functions.

## Commands

```bash
cargo build -p engine_voxel
cargo test -p engine_voxel
cargo test -p engine_voxel --lib <test_name>            # single unit test (src/lib.rs)
cargo test -p engine_voxel --test cpu_gpu_mirror         # GPU divergence pin — needs a real GPU/wgpu device
cargo test -p engine_voxel --test materials
cargo test -p engine_voxel --test terrain_gen_handle
cargo test -p engine_voxel --test terrain_layers_fill
```

## Architecture

- **`TerrainQuery` seam.** `VoxelTerrain` is constructed and wrapped in `Arc<dyn TerrainQuery + Send + Sync>`, then passed as `SimState`'s `terrain` field — same pattern as the default `FlatPlane` impl. No other plumbing needed at call sites.
- **Chronicle → voxel state flow.** The runtime drains one 10-word chronicle record at a time off the engine event ring and calls `apply_voxel_chronicle_record(rec, caster_pos)` (or the `_with_mirror` variant). kind=60 writes a non-zero material byte (`(kind_hash & 0xFF) | 1`, so it never collides with the air sentinel `0`) at `floor(caster_pos)`. kind=59 clears up to `amount` matching-material cells in a deterministic ascending z/y/x walk over a 3-cell radius cube around the caster. Unknown kinds / short slices return `None` (no-op).
- **GPU-resident mirror.** `VoxelMirror::new(gpu, grid)` allocates a `wgpu::Buffer` (`STORAGE | COPY_DST | COPY_SRC`, one `u32` per cell — WGSL has no native u8 storage type) and does a whole-buffer initial upload. Per-tick, `mark_dirty(cell)` records the containing 8³-cell chunk in a `BTreeSet<ChunkCoord>`; `flush_dirty(gpu, grid)` drains the set in ascending order and re-uploads each dirty chunk row-by-row via `Queue::write_buffer`. The `BTreeSet` ordering is load-bearing for P5/P11 determinism (same chronicle drain → same dirty set → same upload sequence → same final GPU state) — don't swap it for a `HashSet`.

## Non-obvious things

- **The real dependency is a pinned `git` dependency, not either local checkout.** `Cargo.toml` declares `voxel_engine = { git = "https://github.com/RPP1011/voxel_engine", rev = "a85f195834e056bf842167974f1b510308318b46", default-features = false }`. This is **not** the sibling checkout at `F:\Game\voxel_engine` (a full Vulkan/ash engine, edition 2024) and **not** `F:\Game\voxel_engine-stub` (an orphaned hand-written API-compatible shim created 2026-07-21 for an unrelated DSL spike, `default = []` but otherwise empty). Neither sibling is referenced by any `[patch]` table or `.cargo/config.toml` in this workspace — verify with `cargo tree -p engine_voxel -i voxel_engine` if in doubt. The crate only touches `voxel::grid` + `voxel::raycast` from the git dep.
- **`Cargo.toml`'s own header comment is stale** — it still describes `voxel_engine` as a "path dep" (leftover from Phase A drafting before the git dep was pinned). Trust the `[dependencies]` block, not the comment above it.
- **Vulkan deps compile unconditionally.** `default-features = false` on the git dep keeps `winit`/`egui` out, but `ash`, `gpu-allocator`, and the `shaderc` build-dep in `voxel_engine` are unconditional — they compile whether or not this crate's code touches Vulkan. Only `voxel::grid`/`voxel::raycast` are used at the Rust level; the Vulkan compile cost is paid regardless. Sim crates that don't depend on `engine_voxel` are unaffected.
- **`wgpu` is pinned to `=26.0.1`** to match `engine`'s exact wgpu version — don't bump it here independently or the dep graph gets two wgpu versions.
- **Determinism audit is a standing obligation, not a one-time check.** The crate doc-comment in `src/lib.rs` documents that `voxel_engine`'s `HashMap`/`HashSet` usage (in `svdag.rs`/`articulation.rs`/`connectivity.rs`) is unreachable from the modules this crate imports — re-audit that claim if the pinned `rev` is ever bumped.
- **`docs/perf/2026-05-09-stress-ceilings.md`** has the measured per-tick mirror upload cost baseline (5.83 µs mean / 325 µs max) from the `wave_defense_runtime` opt-in — reference point if you're evaluating whether mirror upload cost has regressed.
