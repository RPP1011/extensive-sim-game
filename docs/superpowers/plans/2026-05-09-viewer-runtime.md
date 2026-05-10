# Viewer Runtime — Wire `voxel_engine`'s Renderer to a Sim Fixture

> Goal: stand up a real-time visual viewer for the deterministic sim by reusing `voxel_engine`'s existing wgpu/Vulkan renderer + `winit` window via its `app::App` harness. Pilot fixture is `wave_defense` (palisades + monsters + settlers makes "looks like a game" the easiest to read). Phased so each PR stands on its own.

## Goal

The sim today is headless: `sim_app` prints a per-tick text summary, runtime crates have console-only `*_app` driver bins, and the only graphics surface is the GPU compute kernels themselves. To make this a *game*, we need a window with a moving picture.

`voxel_engine` (`~/Projects/voxel_engine`, github.com/RPP1011/voxel_engine, pinned at rev `02d21a2`) already has a complete renderer:
- Vulkan rendering via `ash` + `gpu-allocator`
- `winit` + `egui` integration behind the `app-harness` feature
- A `Scene` with entities, transforms, voxel storage, camera
- `app::App` trait — `setup() / tick() / on_input()` lifecycle hooks
- `app::AppConfig.fixed_tick_rate: 10.0` — already 10 Hz, conveniently matches sim's 100ms tick

The integration model is straightforward: write a `viewer_runtime` crate that implements `voxel_engine::app::App`, owns a sim runtime instance, and on each `tick()` reads sim state (agent positions, alive bitmap, voxel terrain) and mirrors it into the renderer's `Scene`. Sim is the source of truth; the renderer is a passive consumer.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `voxel_engine::app::harness::{App, AppConfig}` at `~/Projects/voxel_engine/src/app/harness.rs:1-30` (the lifecycle hook contract)
  - `voxel_engine::scene::scene::Scene` at `~/Projects/voxel_engine/src/scene/scene.rs:332` (entity + voxel mutation API)
  - `engine_voxel::{VoxelTerrain, VoxelMirror}` at `crates/engine_voxel/src/lib.rs` (the existing voxel-engine seam — already pulls voxel_engine as a dep, already pinned to `02d21a2`)
  - `wave_defense_runtime::WaveDefenseState` at `crates/wave_defense_runtime/src/lib.rs:276` (pilot-fixture sim driver; has `read_pos / read_alive / read_score` snapshot methods)
  - `engine::CompiledSim` trait at `crates/engine/src/lib.rs` — `step / tick / agent_count` minimal contract used by `sim_app`
  - `sim_app/src/main.rs` (existing console-only driver — pattern to mirror for the renderer entry)

  Search method: `rg`, `find`, direct `Read`.

- **Decision:** new `crates/viewer_runtime` crate that depends on `voxel_engine` (with `app-harness` feature enabled) and on the chosen sim runtime. Implements `voxel_engine::app::App`. NOT extending `engine_voxel` because that crate is the *data* seam (CPU/GPU voxel grid mirror); the viewer is the *presentation* seam (window + camera + scene rendering). Different concerns, different deps (egui/winit are heavy and shouldn't leak into runtime crates that just want voxel terrain queries).

- **Rule-compiler touchpoints:**
  - DSL inputs edited: NONE (viewer is a presentation layer; sim semantics unchanged).
  - Generated outputs re-emitted: NONE.

- **Hand-written downstream code:** YES, intentionally — the viewer crate is presentation glue, not engine extension. Every `viewer_runtime` LOC is hand-written window/scene/camera code, none of which the dsl_compiler is positioned to emit. The sim itself stays compiler-first.

- **Constitution check:**
  - P1 (Compiler-First Engine Extension): N/A — viewer doesn't add engine rules; it consumes existing engine snapshot APIs.
  - P2 (Schema-Hash on Layout): N/A — no SoA changes.
  - P3 (Cross-Backend Parity): N/A — viewer is presentation-only; sim itself runs unchanged.
  - P4 (`EffectOp` Size Budget): N/A — no EffectOp changes.
  - P5 (Determinism via Keyed PCG): PASS — viewer is read-only WRT sim state; doesn't introduce any randomness on the sim path.
  - P6 (Events Are the Mutation Channel): PASS — viewer reads snapshots, never writes sim state directly.
  - P7 (Replayability Flagged): N/A — no events declared.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task closes with a SHA.
  - P10 (No Runtime Panic): PASS — viewer must `Result`-bubble GPU init failure (no `.expect`), and per-tick `step()` is wrapped in `catch_unwind` (matches `wave_defense_app`'s pattern).
  - P11 (Reduction Determinism): N/A — no atomic reductions added.

- **Runtime gate:** every phase has a runtime test that actually instantiates the viewer (or the relevant subsystem) and asserts an observable post-condition.
  - Phase A: `viewer_runtime::tests::viewer_construction_succeeds_or_skips` — instantiate `ViewerApp::new(WaveDefenseState::new(seed))`, drive 10 frames, assert agent positions reach `Scene` at non-default transforms (or skip cleanly with eprintln if no GPU surface, mirroring `same_seed_same_death_tick`'s pattern).
  - Phase B: `viewer_runtime::tests::voxel_palisade_appears_in_scene` — drive 100 ticks of wave_defense, assert ≥1 voxel cell with `material != 0` reached `Scene`.
  - Phase C: `viewer_runtime::tests::hud_text_reflects_sim_state` — extract HUD overlay's tick string, assert it advances with `state.tick`.
  - Phase D: `viewer_runtime::tests::camera_input_mutates_view_matrix` — synthesize a `WindowEvent::KeyboardInput`, assert `Scene::camera()` view matrix changes.
  - Phase E: `cargo test -p viewer_runtime --features fixture-boids` etc. — one runtime test per supported fixture asserting `setup()` succeeds.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

## Phasing — five PRs

| # | Phase | Files | Deliverable |
|---|---|---|---|
| A | Skeleton + agent spheres | `crates/viewer_runtime/{Cargo.toml,src/lib.rs,src/bin/viewer_app.rs}`, workspace `Cargo.toml` | New crate. Pulls `voxel_engine` with `app-harness`. `ViewerApp<S: CompiledSim>` impls `voxel_engine::app::App`. `setup()` spawns one `Scene` entity per agent (sphere mesh placeholder). `tick()` calls `sim.step()` and updates each entity's transform from `state.read_pos()`. Pilot: wave_defense. |
| B | Voxel terrain sync | `crates/viewer_runtime/src/voxel_sync.rs` (new), `crates/viewer_runtime/src/lib.rs` (wiring) | Subscribe to `engine_voxel::VoxelTerrain`'s dirty set per tick. Mirror dirty cells into `Scene::set_voxel(...)`. Palisades become visible blocks. |
| C | egui HUD | `crates/viewer_runtime/src/hud.rs` (new), `voxel_engine` may need a hook | Overlay top-left: `tick: N · alive_settlers: M · alive_monsters: K · score: S`. Uses `voxel_engine`'s existing egui integration (already wired via `app-harness`). Read sim state via the existing `WaveDefenseState` accessors. |
| D | Free-fly debug camera | `crates/viewer_runtime/src/camera.rs` (new), input handling in `App::on_input()` | WASD + mouse-look free-camera. `voxel_engine` already has a camera; we drive it via input events. No game-specific input yet (no clicks-to-spawn). |
| E | Multi-fixture support | `crates/viewer_runtime/Cargo.toml` (feature-gate runtimes), `viewer_app.rs` (CLI fixture select) | Feature flags `fixture-wave_defense` (default), `fixture-boids`, `fixture-spy_network`, etc. Each fixture supplies a small adapter that maps its agents to scene entities. CLI: `viewer_app --fixture boids`. |

Each phase is one PR. Phase A unblocks B-E. B-E are independent; could parallelize once A lands but keeping sequential is fine — the merge tax doesn't bite (no shared enum/match).

## Test recipe

For each phase: `cargo test -p viewer_runtime --release` with a per-phase test added that drives the relevant code path under the per-fixture default feature. The runtime gate test (above) validates the change actually does what's claimed.

GUI sanity (manual, not CI): `cargo run -p viewer_runtime --bin viewer_app --release` should pop a window showing the wave_defense scene; monsters visibly march toward origin, settlers visibly build palisades around tick 50/100/150/200 (every 50 ticks per `palisade_period`).

## Out of scope (explicitly)

- **Asset pipeline.** Sphere placeholders for agents; cube voxels for terrain. No imported meshes, no textures, no materials beyond color-by-creature-type.
- **Audio.** Defer.
- **Networked / multiplayer.** Single-process, single-window.
- **Player units.** No clickable spawning, no faction control. Camera is observer-only this slice.
- **Save / load through the viewer.** The sim has snapshot/replay; the viewer is just a renderer atop the live sim.
- **Determinism of the renderer.** The sim is deterministic; the renderer just shows what's there. Frame timing variance is fine.
- **Cross-platform.** Linux first (matches dev environment + voxel_engine's primary target). macOS/Windows when there's demand.
- **WebAssembly.** voxel_engine uses raw Vulkan via `ash`; no browser path.

## Cross-cutting risks

1. **Two graphics stacks in one process.** Sim uses wgpu (`=26.0.1`); voxel_engine renderer uses raw Vulkan via `ash`. Two `vk::Instance` objects in one process is fine in theory (Vulkan loader supports it), but worth verifying on first run. If contention surfaces, the cleanest fix is a future slice that ports voxel_engine's renderer to wgpu — bigger lift.

2. **Tick rate alignment.** Sim is 100ms fixed-tick; voxel_engine's `AppConfig.fixed_tick_rate: 10.0` matches that. But the renderer's `tick()` callback fires at the *display* rate (vsync, ~60Hz typically) plus a separate fixed-step loop. We need to call `sim.step()` only on fixed-step ticks, not on every render frame, or sim ticks will outpace wall-clock. Phase A test pin asserts `sim.tick()` advances exactly `N` after `N` calls to `App::tick()` (interpreted as fixed-step ticks).

3. **voxel_engine's own physics.** voxel_engine has `Scene::tick_sim()` for its own physics. We must NOT call it — the sim is the source of truth. Phase A wires `App::tick()` to call only `sim.step()` + transform-mirror, never `scene.tick_sim()`. Phase A test pin assertion: voxel_engine's physics must not move our entities (set agent at pos=X, tick 10 frames, agent must still be at pos=X if sim is also held still).

4. **CI doesn't have a display.** The viewer is a windowed app; CI's lavapipe doesn't have a window surface, so the viewer's full E2E can't run on CI. Phase A's runtime test uses the `Scene::new_headless(...)` constructor (already exists per `scene.rs`) so the viewer can be exercised without a window — only the renderer's swapchain creation is skipped. CI runs the headless variant; the windowed variant is local-only.

## Critical files

- `crates/viewer_runtime/` — entire new crate (5 modules across 5 phases).
- `crates/engine_voxel/Cargo.toml` — *no change*; voxel_engine is already pulled in. The viewer pulls voxel_engine *separately* with the `app-harness` feature; cargo feature-unifies the dep so we don't get two versions.
- workspace `Cargo.toml` — add `crates/viewer_runtime` to `[workspace] members`.
- `~/Projects/voxel_engine/Cargo.toml` — *no change expected*. If the integration surfaces a gap (e.g. needing a public scene mutation API that doesn't exist), file a follow-up against voxel_engine's own repo and pin the rev forward.

## Why path #1 over Bevy or new winit-from-scratch

Per the user's selection 2026-05-09: `voxel_engine` already has a working renderer + the same wgpu version is already in our dep tree (because `engine_voxel` already uses it). Path #2 (new winit + wgpu + egui crate) means building an asset pipeline + scene representation + camera controls from zero (~weeks of work). Path #3 (Bevy) brings ECS that fights the deterministic-sim model + heavyweight dep. Path #1's tradeoff is the tighter coupling to `voxel_engine`'s design — acceptable since both crates are owned by the same person.
