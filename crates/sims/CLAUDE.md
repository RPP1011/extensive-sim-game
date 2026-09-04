# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

Read the workspace-root `F:\Game\extensive-sim-game\CLAUDE.md` first. This file only adds depth specific to `crates/sims`.

## What this crate is

`sims` is the mega-crate that compiles nearly all of the workspace's `assets/sim/*.sim` DSL fixtures (107 `.sim` files on disk as of this writing) into generated runtimes and hosts their integration/determinism ("pin") tests. Each migrated fixture becomes `sims::<fixture>::GeneratedRuntime` — a GPU-backed struct implementing `engine_play_api::PlayableRuntime` — with zero per-fixture crate boilerplate. This replaced the old one-crate-per-fixture pattern (`crates/*_runtime`) as fixtures migrated over; see `docs/game/compiler_progress.md` and `docs/game/feature_flow.md` for the compiler-first discipline that produces the `.sim` sources these fixtures compile from, and `docs/game/wolves_and_humans.md` for the canonical DSL-port walkthrough (that scenario itself lives in `crates/engine`, not here — this crate is where everything *after* wolves+humans landed).

## Commands

```bash
cargo build -p sims                              # compiles every allow-listed fixture via build.rs
cargo test -p sims                                # runs all pin/exec/smoke tests in crates/sims/tests/
cargo test -p sims <fixture>_pin                  # run one fixture's pin test, e.g.:
cargo test -p sims forest_fire_pin
cargo test -p sims -- --test-threads=1            # serial, for determinism-sensitive runs
```

Most test files are named `tests/<fixture>_pin.rs` (one file per fixture, containing one or more `#[test]` fns — the file itself, not a single fn, is the natural unit to target with `cargo test -p sims <fixture>`). Other suffixes in this crate: `_exec` (drives a `make_playable` runtime end-to-end, e.g. `vampire_survivors_exec.rs`), `_smoke` (minimal compiles-and-runs check), `_bench` (perf, not correctness), `_playable` (exercises the `PlayableRuntime` seam directly). A few tests need a bigger stack because of deep generated dispatch chains — check the file's top comment; `playable_registry.rs`, `predator_prey_playable.rs`, `subkind_seeding_exec.rs`, and `vampire_survivors_exec.rs` currently need:

```bash
RUST_MIN_STACK=33554432 cargo test -p sims --test <that_file>
```

GPU-backed fixtures skip gracefully (`eprintln!` + early `return`, not a failure) when `GeneratedRuntime::try_new` returns `None` because no wgpu adapter is available (headless CI, etc.) — a passing-but-silent test run is expected in that environment, don't chase it as a bug.

## build.rs discovery + the allowlist — precisely

`crates/sims/build.rs` does **not** auto-discover every `.sim` file it finds. It walks `assets/sim/*.sim`, but for each discovered stem it only compiles it if the stem matches an explicit `matches!(stem.as_str(), "a" | "b" | "c" | ...)` allowlist in `build.rs` (currently a single `match` arm list starting around the `fixtures.push(stem)` gate near the top of `main()`). A `.sim` file whose stem is **not** in that list is silently skipped — no compile error, no module emitted, `cargo build -p sims` succeeds either way. As of this writing there are 7 stems under `assets/sim/` that exist on disk but are absent from the list (e.g. `crowd_navigation_min`, `event_kind_filter_probe`, `particle_collision_min`, `predator_prey_min`, `spatial_probe`) — these are either superseded by a listed sibling fixture or not yet migrated; don't assume every `.sim` file you see compiles into this crate.

**To add a new fixture:**
1. Drop `assets/sim/<name>.sim`.
2. Add `"<name>"` as a new `|`-joined arm to the `matches!` list in `crates/sims/build.rs`.
3. `cargo build -p sims` — build.rs re-parses, emits `sims::<name>::GeneratedRuntime` into `OUT_DIR`, and regenerates `sim_modules.rs`.
4. Write `crates/sims/tests/<name>_pin.rs`.

Forgetting step 2 is the single most common mistake: the crate builds fine, but `sims::<name>` doesn't exist and any test referencing it fails to compile with an unresolved-module error that gives no hint the fixture was skipped upstream.

build.rs also re-parses each allow-listed fixture's imports (`dsl_compiler::parse_with_imports`) purely to emit extra `cargo:rerun-if-changed` lines, so edits to shared `stdlib/` files or local-relative imports correctly trigger incremental rebuilds of dependent fixtures.

## Architecture

**`.sim` → `sims::<fixture>::GeneratedRuntime`:** For each allow-listed stem, `build.rs` calls `dsl_compiler::build_helper::emit_namespaced(stem)`, which compiles the `.sim` source (parse → lower → emit) and writes `OUT_DIR/<stem>/generated.rs` + `OUT_DIR/<stem>/runtime_core.rs` (plus `OUT_DIR/<stem>/terrain_gen.rs` when the fixture binds a `voxel_grid`). `build.rs` then synthesizes a single `OUT_DIR/sim_modules.rs` stub containing one `pub mod <stem> { include!(...generated.rs); include!(...runtime_core.rs); }` per fixture, `include!`-ed from `src/lib.rs`. `generated.rs` and `runtime_core.rs` share the same module scope by design — the latter's `dispatch::KernelCache` / `schedule::SCHEDULE` references resolve against modules the former declares. Every `GeneratedRuntime` exposes `try_new(seed: u64, agent_count: u32) -> Option<Self>` (GPU init can fail headless, hence `Option`) and a `step()` that advances one tick.

`build.rs` also emits `sims::make_playable(name, seed, agents) -> Option<Box<dyn engine_play_api::PlayableRuntime>>` and `sims::PLAYABLE_FIXTURES: &[&str]` — a name-keyed registry over the same allow-listed fixture set, letting one generic player binary construct any compiled fixture by string name. See `tests/playable_registry.rs` for the pattern.

**What a pin test checks:** unlike `crates/engine/tests/wolves_and_humans_parity.rs` (which diffs against a checked-in baseline text file for byte-identical event logs), pin tests in this crate do not carry checked-in baselines — none exist under `crates/sims/tests/`. Instead a pin test seeds a `GeneratedRuntime` with a fixed seed, drives a known number of ticks, reads back GPU buffers (host-side SoA columns and/or `@materialized` view storage), and asserts hardcoded expected values (e.g. `tom_probe_decay_pin.rs`: confidence decays exactly 1/tick from a seeded observation, saturates at 0, re-pegs on fresh observation). Several pin tests additionally construct a second fresh runtime with the same seed, replay the same tick sequence, and assert the two runs match (`forest_fire_pin.rs` does this over `@materialized` view-storage buffers) — this is the in-test determinism check (P5: all randomness flows through `per_agent_u32(seed, agent_id, tick, purpose)`), done by direct comparison rather than a golden-file diff. Some pins document a *known, bounded* non-determinism (e.g. `forest_fire_pin.rs`'s tracked `f32` reduction race from parallel `atomicAdd` — non-associative float addition across GPU lanes) and assert a drift tolerance instead of exact equality; read the test's comment block before assuming a failing assertion is a fresh regression.

**Fixtures here vs. legacy per-fixture crates:** `crates/tom_probe_runtime` and `crates/viewer_runtime` are the last surviving crates following the pre-consolidation one-crate-per-fixture pattern (each compiles its own `.sim` independently via its own `build.rs`/`OUT_DIR`). They are out of scope for this file — they get their own `CLAUDE.md` if/when one is written. Note `crates/sims/tests/tom_probe_*_pin.rs` files exist in *this* crate too (ported off `tom_probe_runtime`) via the shared `tests/tom_probe_helpers/mod.rs` shim — don't confuse "a tom_probe-named pin test lives here" with "tom_probe_runtime is part of this crate's build."

## Non-obvious things / pitfalls

- **Generated code is never checked in.** Everything under `OUT_DIR/<fixture>/` is build-time-only (`target/**/out/`), unlike `crates/engine_rules` emission elsewhere in the workspace, which *is* checked in for reviewability. There is nothing to hand-edit or diff-review here beyond the `.sim` source itself — if generated output looks wrong, the fix belongs in `dsl_compiler`'s emitter, never in a patched `OUT_DIR` file (it will be silently overwritten on the next build anyway).
- **The allowlist match arm is the real gate, not the directory scan.** See above — this is the #1 source of "why doesn't `sims::my_fixture` exist" confusion.
- **A fixture with a companion `assets/ability_test/<fixture>/` directory** gets its `.ability` corpus baked in via `include_str!` at compile time and parsed/registered at `try_new()` — no runtime file I/O, but it does mean `try_new()` does real parsing work on first call (relevant if you're chasing where "no wgpu adapter" isn't the reason a test is slow to start).
- **Big fixtures need `RUST_MIN_STACK`** (see Commands) — deep generated dispatch/schedule chains can blow the default thread stack; the failure mode is a stack overflow crash, not a clean panic, so if a test SIGSEGVs / aborts with no assertion message, check for this before assuming a logic bug.
- **`voxel_grid`-bound fixtures pull in `engine_voxel`** (`VoxelTerrain` + `VoxelMirror`) only when `build_helper` detects the binding in the kernel manifest; the `Cargo.toml` dependency is unconditional (cargo needs the path to resolve the import) but voxel-free fixtures don't link any of that code.
- **The `webband_*` fixtures are gone.** They left with the game to its own repo (RPP1011/webband, 2026-07-23); `many_events_ability` is what's left in this corpus covering the >25-user-event `apply_ability` regression they used to also exercise. Don't go looking for `webband_*` under `assets/sim/`.
- **`cargo test -p sims` compiles the entire allow-listed fixture set** even if you only want to run one test — there's no way to build a subset via build.rs. If you're iterating on a single fixture, expect the first `cargo test -p sims <fixture>` after a `.sim` edit to rebuild all 100+ fixtures' generated code (build.rs reruns whenever `assets/sim/` changes, not just the one file you touched), since the crate is a single compilation unit.
