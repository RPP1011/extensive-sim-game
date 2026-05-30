# Generic Player (engine_play) Implementation Plan (Plan B — Wave 1)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** A new `engine_play` crate: a descriptor-driven generic `EngineBridge` (voxel render), a controls→input mapper, and the player loop + `play` binary — all built and tested against Plan 0's sample descriptor JSON + a mock `PlayableRuntime`, with no dependency on the compiler emitting anything yet.

**Architecture:** Port `viewer_runtime::vs::VsBridge` into a generic `EngineBridge` driven by a `RenderDescriptor` + per-frame `AgentView` snapshot (color by field-range, arena floor, follow-cam, ring/beam VFX). A `ControlsMapper` turns held/pressed keys + a `ControlsDescriptor` into `set_input` calls. The loop consumes `dyn PlayableRuntime` + `engine_ui`. Tested headlessly against `engine_play_api`'s sample fixtures and a `MockRuntime` so this whole crate lands in parallel with the compiler (Plan A).

**Tech Stack:** Rust, `voxel_engine` (winit/ash/egui), `engine_ui`, `engine_play_api`. **Depends on Plan 0** (the contract + sample fixtures). NOT on Plan A.

---

## Architectural Impact Statement
- **Existing primitives searched:** `viewer_runtime::vs::VsBridge`/`vs_world_to_voxel`/the VFX code (the logic to generalize); `vs_viewer.rs` event/present loop; `engine_ui::draw` (Plan 2); `engine_play_api` (Plan 0). Method: `Read`.
- **Decision:** NEW crate `engine_play` — the generic player. Presentation/runtime code (P1 governs rules, not viewers); replaces the per-game `vs_*` files (deleted in Plan E).
- **Rule-compiler touchpoints:** none.
- **Hand-written downstream code:** `engine_play/**` — justified presentation/runtime; this is the generic path that *removes* per-game Rust.
- **Constitution check:** P1 PASS (no rule code); P2–P7/P11 N/A; P10 PASS (headless smoke + fallible parse); P8 PASS.
- **Runtime gate:** `bridge_paints_from_descriptor` + `controls_map_to_inputs` + `player_loop_headless` (against MockRuntime + sample JSON) — no panic, expected voxels/inputs.
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: Crate scaffold + MockRuntime
**Files:** Create `crates/engine_play/Cargo.toml`, `src/lib.rs`, `src/mock.rs`; root `Cargo.toml` member.

- [ ] **Step 1:** Cargo.toml deps: `engine_play_api`, `engine_ui` (both path), `voxel_engine = { path="/home/ricky/Projects/voxel_engine", features=["app-harness"] }`, `egui="0.33"`, `winit="0.30"`, `ash="0.38"`, `glam="0.29"`, `anyhow="1"`, `serde_json="1"`. Append member.
- [ ] **Step 2:** `mock.rs` — a `MockRuntime { tick, agents: Vec<AgentView>, render: &'static str, controls: &'static str, ui: &'static str, last_input: Vec<(String,f32)> }` implementing `PlayableRuntime`, returning the sample fixture strings (`include_str!` from `engine_play_api/fixtures/...`) and a couple of fake agents. Used by all tests below. Build: `cargo build -p engine_play`. Commit.

### Task 2: ControlsMapper (pure, testable)
**Files:** Create `crates/engine_play/src/input.rs`.

- [ ] **Step 1 (test first):** `controls_map_to_inputs` — given a `ControlsDescriptor` (parsed from the sample JSON) and a held-key set `{"w","d"}`, `ControlsMapper::resolve(&desc, &held)` returns input writes that, after normalization, set `ctl.move_y≈+1/√2` (before norm `+1`) and `ctl.move_x≈+1/√2`; with `{}` held, all bound fields resolve to their zero/neutral. Assert the (field,value) pairs.
- [ ] **Step 2:** Implement `ControlsMapper::resolve(desc, held) -> Vec<(String,f32)>`: sum each binding's value where its key is held (Hold mode); group by field; normalize the `move_x`/`move_y` pair if both present (unit vector). Press-mode bindings fire once on transition (the loop tracks edges — for the pure test, treat Hold). Run `cargo test -p engine_play input`. Commit.

### Task 3: EngineBridge (descriptor-driven voxel render)
**Files:** Create `crates/engine_play/src/bridge.rs` (port `viewer_runtime/src/vs.rs` logic, generalized).

- [ ] **Step 1:** `EngineBridge::new(ctx, &RenderDescriptor)` builds the flat arena floor sized from `arena_radius`; a palette assigning a material index per `AgentVisual` (color) + per `VfxSpec` color. `refresh(ctx, &[AgentView], tick)`: clear last cells; for each alive agent, pick the first `AgentVisual` whose `when` field-range contains the agent's value (read the named column off `AgentView`) → paint that color; then paint VFX from the descriptor: for each `VfxSpec`, if `tick % period == 0` (Ring) paint a ring at `radius` around the followed agent; `BeamToNearest` paints a beam to the nearest agent matching `target` field-range. (Direct port of the VS bridge's ring/beam code, now data-driven.)
- [ ] **Step 2 (test):** `bridge_paints_from_descriptor` — construct from the sample VS render JSON + a `MockRuntime` snapshot (a player + an enemy); call a CPU-only variant of `refresh` that records painted (cell→material) without GPU (factor the grid-painting out of the GPU upload so it's unit-testable); assert the player cell got the cyan material and a nova ring appears at tick 40. (Keep the GPU `upload` behind the same seam as `VsBridge`.) Run `cargo test -p engine_play bridge`. Commit.

### Task 4: Player loop + `play` binary (against MockRuntime)
**Files:** Create `crates/engine_play/src/player.rs`, `crates/engine_play/src/bin/play.rs`.

- [ ] **Step 1:** `player.rs` — the winit `ApplicationHandler` (ported from `vs_viewer.rs`): owns a `Box<dyn PlayableRuntime>`, an `EngineBridge`, `EguiState`, a `ControlsMapper`, and host `UiData`/level-up/death state generalized from `vs_ui.rs` (the menu/death state machine becomes generic: level via `view_value("xp", player_slot)` if the ui descriptor references it; actions applied via `set_input`). Each frame: resolve controls→`set_input`, step, snapshot, bridge.refresh, egui draw(ui_descriptor model), apply `UiAction` via `set_input`. Camera follows the agent matching the descriptor's `Follow` range.
- [ ] **Step 2:** `bin/play.rs` — for Wave 1, `fn main()` constructs a `MockRuntime` and runs the loop (proves the loop compiles + runs headlessly-skippable). The real registry wire (`sims::make_playable(argv)`) is Plan D.
- [ ] **Step 3 (test):** `player_loop_headless` — construct the player state with `MockRuntime`, drive N synthetic frames (no window: factor the per-frame update — controls resolve, step, snapshot, UiData build, engine_ui::draw in a headless `egui::Context` — into an `update()` separable from window/GPU), assert no panic + that held "d" produced a `ctl.move_x` input on the mock. Run `cargo test -p engine_play`. Commit.

## Self-review note
Everything here is tested against `MockRuntime` + the Plan-0 sample JSON, so this crate lands fully in parallel with Plan A. Plan D swaps `MockRuntime` for `sims::make_playable(name)` in `bin/play.rs` and verifies end-to-end. Keep the per-frame `update()` logic separable from the winit/GPU shell so it stays headlessly testable.
