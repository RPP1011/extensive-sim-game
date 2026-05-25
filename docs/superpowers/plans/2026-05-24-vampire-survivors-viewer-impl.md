# Vampire Survivors — Voxel Viewer (Phase D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Watch `vampire_survivors` run in 3D — render the executing sim (player kiting a DSL-spawned swarm) in the `viewer_runtime` voxel viewer via a parallel `VsViewerApp` + `VsBridge` + `vs_viewer` binary that leaves the existing dungeon_horde viewer untouched.

**Architecture:** A new `crates/viewer_runtime/src/vs.rs` module holds `VsViewerApp` (wraps `sims::vampire_survivors::GeneratedRuntime`, seeds via the existing `seed_initial_state`, steps with `drain_summons`, snapshots agents by mana band) and `VsBridge` (a flat open-arena voxel renderer painting player/enemy/spawner by mana-band color). A new `src/bin/vs_viewer.rs` is the winit window + render loop. The sim-side is gated by a headless smoke test; the Vulkan rendering is verified manually (no headless render path exists). Reuses crate-private helpers (`read_positions`, `read_agent_u32/f32`, `VoxelGrid`, `GpuVoxelTexture`, `VulkanContext`, palette) — a child module can access its crate's private items.

**Tech Stack:** Rust, `viewer_runtime` (winit 0.30 + Vulkan via `voxel_engine`/`ash` for render, wgpu for the sim), `sims::vampire_survivors` + `sims::vampire_survivors_seed` + `sims::summon_alloc`. References: `crates/viewer_runtime/src/lib.rs` (`ViewerApp`/`VoxelBridge`/readback helpers/palette), `crates/viewer_runtime/src/dungeon.rs` (`seed_voxel_dungeon`/`seed_topology`/grid constants), `crates/viewer_runtime/src/bin/viewer_app.rs` (window+loop), `crates/viewer_runtime/tests/smoke.rs` (headless smoke template), `crates/sims/tests/vampire_survivors_exec.rs` (the drain loop pattern).

**Spec:** `docs/superpowers/specs/2026-05-24-vampire-survivors-viewer-design.md` §4 Phase D.

---

## Architectural Impact Statement (P8)

- **Existing primitives searched:** `ViewerApp`/`VoxelBridge`/`material_for`/`read_positions`/`read_agent_u32`/`read_agent_f32` (lib.rs), `seed_voxel_dungeon`/`seed_topology`/grid constants `GRID_X=GRID_Y=96,GRID_Z=8` (dungeon.rs), the winit `ApplicationHandler` loop + `SIM_TICK_PERIOD` (viewer_app.rs), `smoke.rs` (headless construct+step), `sims::vampire_survivors_seed::seed_initial_state` + `sims::summon_alloc::{drain_summons,DrainCtx}`. Method: Explore read of the viewer crate.
- **Decision:** new parallel `VsViewerApp` + `VsBridge` (new `src/vs.rs`) + `vs_viewer` binary. NO changes to `ViewerApp`/dungeon path (zero regression risk). All rendering/runtime glue; no engine or compiler edits, no `.sim` changes.
- **Constitution check:** P1 ✅ — viewer/runtime code, no sim-rule logic. P3/P5/P2 N/A (no new sim behavior, no engine columns, no RNG in the renderer beyond the already-seeded sim). P8 ✅.

## Five gotchas (the design's known traps — every task references these)

1. **`+48` grid-center offset:** the VS sim uses origin-`(0,0)`-centered world coords (spawners at radius 40); the voxel grid is `[0,96)`. Apply `+GRID_X/2` (=48) to x and y **only in the bridge's world→voxel mapping** (leave GPU/sim coords origin-centered). Without it: a blank frame (agents clipped at negative coords).
2. **`drain_summons` after `step()`:** without it, the sim emits summon chronicles but no enemies become alive. Insert it right after `self.app.step()` (in `VsViewerApp::step`).
3. **VS has no `voxel_terrain`/`voxel_mirror`:** do NOT call `seed_voxel_dungeon` or `state.voxel_terrain.*` (won't compile). Paint a flat floor entirely in the CPU `VoxelGrid`.
4. **Mana-based color (not creature_type):** VS's `agent_creature_type_buf` exists but is zero-filled; colors MUST key on `agent_mana_buf` bands (player<1.5, enemy<2.5, spawner else).
5. **Borrow pattern for the drain:** `drain_summons` borrows several `&rt.*` at once — construct `DrainCtx` from `&self.app.state.<field>` refs in one expression (see `vampire_survivors_exec.rs` for the working pattern).

---

## File Structure

- `crates/viewer_runtime/src/vs.rs` — **create**: `VsViewerApp`, `VsBridge`, `material_for_vs`, `seed_vs_floor`.
- `crates/viewer_runtime/src/lib.rs` — **modify**: `pub mod vs;`; if needed, change `read_positions`/`read_agent_u32`/`read_agent_f32`/`VoxelGrid`/`GpuVoxelTexture`/`VulkanContext` from private to `pub(crate)` so `vs.rs` can use them.
- `crates/viewer_runtime/src/bin/vs_viewer.rs` — **create**: winit window + render loop driving `VsViewerApp` + `VsBridge`.
- `crates/viewer_runtime/tests/smoke_vs.rs` — **create**: headless smoke (construct + step + spawn assertion).
- `crates/viewer_runtime/Cargo.toml` — **modify** if `sims` summon_alloc isn't already reachable (it should be; `sims` is already a dep for dungeon_horde).

---

## Task D1: `VsViewerApp` core + headless smoke (sim-side, fully testable)

**Files:** Create `crates/viewer_runtime/src/vs.rs`; modify `crates/viewer_runtime/src/lib.rs`; create `crates/viewer_runtime/tests/smoke_vs.rs`.

- [ ] **Step 1: Read the references**

Read `crates/viewer_runtime/src/lib.rs` for: the `AgentSnapshot` struct, `read_positions`/`read_agent_u32`/`read_agent_f32` signatures, and how `ViewerApp::refresh_snapshot` reads buffers. Read `crates/viewer_runtime/tests/smoke.rs` (the headless template). Read `crates/sims/tests/vampire_survivors_exec.rs` for the `DrainCtx` construction + drain loop.

- [ ] **Step 2: Write `VsViewerApp` (sim-side only, no rendering) in `vs.rs`**

```rust
//! Vampire Survivors voxel-viewer path — parallel to the dungeon_horde ViewerApp.
//! Sim-side here (state, seeding, step+drain, mana-band snapshot); rendering in VsBridge.
use sims::vampire_survivors::GeneratedRuntime;
use sims::vampire_survivors_seed::seed_initial_state;
use sims::summon_alloc::{drain_summons, DrainCtx};

pub const VS_AGENT_COUNT: u32 = 512;

/// A live agent for rendering, classified by mana band.
#[derive(Clone, Copy)]
pub struct VsAgent {
    pub pos: [f32; 3],
    pub hp: f32,
    pub role: VsRole, // Player / Enemy / Spawner, derived from mana band
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VsRole { Player, Enemy, Spawner }

pub fn role_for_mana(mana: f32) -> VsRole {
    if mana < 1.5 { VsRole::Player } else if mana < 2.5 { VsRole::Enemy } else { VsRole::Spawner }
}

pub struct VsViewerApp {
    pub state: GeneratedRuntime,
    pub seed: u64,
    pub agent_count: u32,
    agents: Vec<VsAgent>,
    terminated_at_tick: Option<u64>,
}

impl VsViewerApp {
    pub fn try_new(seed: u64) -> Option<Self> {
        let mut state = GeneratedRuntime::try_new(seed, VS_AGENT_COUNT)?;
        seed_initial_state(&mut state);
        let mut app = Self {
            state, seed, agent_count: VS_AGENT_COUNT,
            agents: Vec::new(), terminated_at_tick: None,
        };
        app.refresh_snapshot();
        Some(app)
    }

    pub fn sim_tick(&self) -> u64 { self.state.tick }
    pub fn agents(&self) -> &[VsAgent] { &self.agents }

    /// One sim tick: step, then drain summons into live enemies, then snapshot.
    pub fn step(&mut self) {
        self.state.step();
        let _ = drain_summons(DrainCtx {
            device: &self.state.gpu.device,
            queue: &self.state.gpu.queue,
            event_ring: &self.state.event_ring,
            agent_alive_buf: &self.state.agent_alive_buf,
            agent_pos_buf: &self.state.agent_pos_buf,
            agent_count: self.state.agent_count,
            seed: self.state.seed,
            tick: self.state.tick,
        });
        self.refresh_snapshot();
        if self.terminated_at_tick.is_none() && !self.agents.iter().any(|a| a.role == VsRole::Player) {
            self.terminated_at_tick = Some(self.state.tick);
        }
    }

    /// Read pos/alive/hp/mana back; build the live-agent list classified by mana band.
    fn refresh_snapshot(&mut self) {
        let n = self.agent_count;
        let alive_buf = self.state.agent_alive_buf.clone();
        let hp_buf = self.state.agent_hp_buf.clone();
        let mana_buf = self.state.agent_mana_buf.clone();
        let positions = crate::read_positions(&mut self.state, n);
        let alive = crate::read_agent_u32(&mut self.state, &alive_buf, n);
        let hps = crate::read_agent_f32(&mut self.state, &hp_buf, n);
        let mana = crate::read_agent_f32(&mut self.state, &mana_buf, n);
        self.agents.clear();
        for i in 0..n as usize {
            if alive[i] == 1 {
                self.agents.push(VsAgent {
                    pos: [positions[i][0], positions[i][1], positions[i][2]],
                    hp: hps[i],
                    role: role_for_mana(mana[i]),
                });
            }
        }
    }
}
```

Add `pub mod vs;` to `crates/viewer_runtime/src/lib.rs`. If `read_positions`/`read_agent_u32`/`read_agent_f32` are private (no `pub`/`pub(crate)`), change them to `pub(crate)` so `crate::read_*` resolves from `vs.rs`. (A child module CAN see ancestor-private items, so this may already work; only escalate visibility if the compiler complains.)

- [ ] **Step 3: Write the headless smoke test**

Create `crates/viewer_runtime/tests/smoke_vs.rs` (mirrors `tests/smoke.rs` skip idiom):

```rust
use viewer_runtime::vs::{VsViewerApp, VsRole};

const SEED: u64 = 0x5_F00D_CAFE_0001;

#[test]
fn vs_viewer_constructs_steps_and_spawns() {
    let mut app = match VsViewerApp::try_new(SEED) {
        Some(a) => a,
        None => { eprintln!("[vs_viewer] skip: no wgpu adapter"); return; }
    };
    // Initial: one player, several spawners, zero live enemies.
    let players = app.agents().iter().filter(|a| a.role == VsRole::Player).count();
    let spawners = app.agents().iter().filter(|a| a.role == VsRole::Spawner).count();
    let enemies0 = app.agents().iter().filter(|a| a.role == VsRole::Enemy).count();
    assert_eq!(players, 1, "exactly one player");
    assert!(spawners >= 1, "at least one spawner; got {spawners}");
    assert_eq!(enemies0, 0, "no live enemies before any wave");

    // Step past a wave period (>30 ticks); enemies should spawn via drain.
    let mut max_enemies = 0;
    for _ in 0..60 { app.step(); max_enemies = max_enemies.max(app.agents().iter().filter(|a| a.role == VsRole::Enemy).count()); }
    assert_eq!(app.sim_tick(), 60, "stepped 60 ticks");
    assert!(max_enemies > 0, "DSL waves should spawn live enemies in the viewer app; got {max_enemies}");
    // No NaN positions.
    assert!(app.agents().iter().all(|a| a.pos.iter().all(|c| c.is_finite())), "no NaN/inf positions");
}

#[test]
fn role_for_mana_bands() {
    use viewer_runtime::vs::role_for_mana;
    assert_eq!(role_for_mana(1.0), VsRole::Player);
    assert_eq!(role_for_mana(2.0), VsRole::Enemy);
    assert_eq!(role_for_mana(3.0), VsRole::Spawner);
}
```

(`VsRole` needs `#[derive(PartialEq, Eq)]` — already in Step 2.)

- [ ] **Step 4: Run the smoke test**

Run: `cargo test -p viewer_runtime --test smoke_vs -- --nocapture`
Expected: PASS on a GPU/lavapipe host (this machine has both) — player=1, spawners≥1, enemies spawn (max>0). Clean skip if no adapter. This re-validates the C3 execution path *through the viewer's app type*, proving the sim-side of the viewer is correct before any Vulkan work.

If `crate::read_positions` etc. don't resolve, fix visibility (Step 2 note). If enemies don't spawn, the drain wiring differs from `vampire_survivors_exec.rs` — diff against it.

- [ ] **Step 5: Commit**

```bash
git add crates/viewer_runtime/src/vs.rs crates/viewer_runtime/src/lib.rs crates/viewer_runtime/tests/smoke_vs.rs
git commit -m "feat(viewer): VsViewerApp sim-side (state+seed+drain+mana snapshot) + headless smoke (Phase D)"
```

---

## Task D2: `VsBridge` — flat-arena voxel render + mana palette

**Files:** Modify `crates/viewer_runtime/src/vs.rs`; modify `crates/viewer_runtime/src/lib.rs` (visibility of `VoxelGrid`/`GpuVoxelTexture`/`VulkanContext`/palette builder if needed).

- [ ] **Step 1: Read the VoxelBridge reference**

Read `VoxelBridge::new` and `VoxelBridge::refresh` in `lib.rs` (the palette construction, `VoxelGrid` painting, agent-splat loop, GPU texture upload/`mark_dirty`, the `VulkanContext`/`VulkanAllocator`/`GpuVoxelTexture` types and `BRIDGE_DIM_*`/`GRID_*` constants). Note the world→voxel cell mapping (`sim (x,y) → renderer (x, 0, y)`).

- [ ] **Step 2: Add `material_for_vs` + `seed_vs_floor` + `VsBridge` to `vs.rs`**

```rust
// VS palette material ids — reuse free slots in the 256-entry palette.
// Pick 3 unused indices (read lib.rs's MAT_* list; e.g. high indices 200/201/202)
pub const MAT_VS_FLOOR: u8 = 200;
pub const MAT_VS_PLAYER: u8 = 201;
pub const MAT_VS_ENEMY: u8 = 202;
pub const MAT_VS_SPAWNER: u8 = 203;

pub fn material_for_vs(role: VsRole) -> u8 {
    match role {
        VsRole::Player => MAT_VS_PLAYER,
        VsRole::Enemy => MAT_VS_ENEMY,
        VsRole::Spawner => MAT_VS_SPAWNER,
    }
}
```

Then `VsBridge`, mirroring `VoxelBridge` but: (a) `new` paints a **flat floor** (`cpu_grid.set(x, 0, y, MAT_VS_FLOOR)` for all `x,y in 0..GRID_X×0..GRID_Y`) — NO walls, NO `voxel_terrain` calls; (b) `refresh(&mut self, ctx, app: &VsViewerApp)` clears last cells, then for each `app.agents()` paints a voxel at `((a.pos[0] + GRID_X as f32/2.0) as u32, splat_height, (a.pos[1] + GRID_X as f32/2.0) as u32)` — the **+48 offset** — with `material_for_vs(a.role)`, bounds-checking `0..GRID_X`/`0..GRID_Z`; (c) re-uploads the GPU texture exactly as `VoxelBridge::refresh` does (destroy + recreate `GpuVoxelTexture`, or `mark_dirty` + upload — match the existing impl). Define the palette RGBA for the 4 new MAT_VS_* ids in `VsBridge::new` (cyan player, orange-red enemy, purple spawner, grey floor).

Build constants (`GRID_X`, `BRIDGE_DIM_*`) come from `dungeon`/`lib.rs` — reference them (`crate::dungeon::GRID_X` etc.) rather than redefining.

- [ ] **Step 3: Compile + unit-test the pure mapping**

Run: `cargo build -p viewer_runtime 2>&1 | tail -25`
Expected: compiles. (Rendering correctness is verified manually in D4; only `material_for_vs`/`role_for_mana`/offset math are unit-testable — the `role_for_mana_bands` test from D1 covers the role mapping.)

Add a tiny offset unit test to `vs.rs` `#[cfg(test)]` if not obvious: assert an agent at world `(0,0)` maps to voxel `(48,48)` and an agent at `(-40,0)` maps to `(8,_)` (inside `[0,96)`).

- [ ] **Step 4: Commit**

```bash
git add crates/viewer_runtime/src/vs.rs crates/viewer_runtime/src/lib.rs
git commit -m "feat(viewer): VsBridge flat-arena voxel render + mana-band palette (Phase D)"
```

---

## Task D3: `vs_viewer` binary — window + render loop

**Files:** Create `crates/viewer_runtime/src/bin/vs_viewer.rs`.

- [ ] **Step 1: Read the binary reference**

Read `crates/viewer_runtime/src/bin/viewer_app.rs` in full — the winit `ApplicationHandler` impl, window/Vulkan-context setup, the `RedrawRequested` tick loop (`SIM_TICK_PERIOD`, `ticks_this_frame < 8`), `bridge.refresh(...)`, camera setup, and the auto-restart-on-termination logic.

- [ ] **Step 2: Write `vs_viewer.rs` as a VS-adapted copy**

Mirror `viewer_app.rs` with these concrete substitutions:
- `ViewerApp` → `viewer_runtime::vs::VsViewerApp`; `VoxelBridge` → `viewer_runtime::vs::VsBridge`.
- The tick loop body: `self.app.step();` already does the drain internally (D1) — so NO separate drain call needed here (confirm: `VsViewerApp::step` drains). Then `bridge.refresh(&ctx, &self.app)`.
- **Camera:** center on the arena: look at `(GRID_X/2, 0, GRID_X/2)` = `(48,0,48)` from an appropriate height (copy the dungeon camera, just change the target/center to the arena middle).
- Remove all dungeon-specific calls (`advance_hero_exploration`, room-fog, hero/boss overlays, `dungeon.*`). The VS loop is just: step → refresh → render.
- Auto-restart: on `app.terminated_at_tick.is_some()` (player dead), increment seed and rebuild `VsViewerApp` (mirror the dungeon restart).
- Default seed const (e.g. `0x5_F00D_CAFE_0001`).

- [ ] **Step 3: Compile**

Run: `cargo build -p viewer_runtime --bin vs_viewer 2>&1 | tail -25`
Expected: compiles. (Cannot run headlessly — verified in D4.)

- [ ] **Step 4: Commit**

```bash
git add crates/viewer_runtime/src/bin/vs_viewer.rs
git commit -m "feat(viewer): vs_viewer binary — windowed VS voxel viewer (Phase D)"
```

---

## Task D4: Manual visual verification

**Files:** none (verification only).

- [ ] **Step 1: Run the viewer**

This requires a display. Tell the user to run it themselves (the harness has no display): suggest they type
`! cargo run -p viewer_runtime --bin vs_viewer --release`
in the session, or run it in their terminal. (If a display is available to the agent, run it; otherwise hand off.)

- [ ] **Step 2: Confirm the visual**

Watch for: a flat arena; one player voxel (cyan) near center kiting away from the swarm; spawner voxels (purple) on the edge ring; enemy voxels (orange-red) appearing in waves (~every 30 ticks) and streaming toward the player; enemies vanishing as weapons kill them. If the frame is blank, the `+48` offset (gotcha 1) is wrong; if no enemies appear, `drain_summons` isn't firing in `VsViewerApp::step` (gotcha 2).

- [ ] **Step 3 (optional): capture a screenshot/GIF for the record** if the viewer or the harness supports it.

---

## Final verification (Phase D)

- [ ] `cargo test -p viewer_runtime --test smoke_vs` → PASS or clean GPU-skip; on a GPU host, player=1 + enemies spawn through the viewer's app type.
- [ ] `cargo build -p viewer_runtime --bin vs_viewer` → compiles.
- [ ] `cargo build -p viewer_runtime` → no new warnings; the existing `ViewerApp`/dungeon path is untouched (`git diff --stat` shows only `vs.rs`, the `pub mod vs;` + visibility lines in lib.rs, the new bin, the new test).
- [ ] Manual: the windowed `vs_viewer` shows the kiting player + spawning swarm (D4).
- [ ] `cargo test -p viewer_runtime --test smoke` (the dungeon smoke) still PASSES — confirms zero regression to the existing viewer.
