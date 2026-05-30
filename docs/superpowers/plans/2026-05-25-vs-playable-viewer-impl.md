# Vampire Survivors Playable Viewer Implementation Plan (Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `vs_viewer` into a playable game: WASD drives the player, a follow-camera tracks them, an egui HUD shows HP/timer/level/XP/kills/enemies, level-ups open a 3-card upgrade menu, and death shows a run summary + restart — all via the `engine_ui` framework (Plan 2) and the `config.ctl` setters (Plan 1).

**Architecture:** `vs_viewer` already calls `present_blit_with_overlay` with a no-op closure (`bin/vs_viewer.rs:338`). Wire `voxel_engine::EguiState` into that closure. A new `vs_ui.rs` owns the host-side game state: per-tick WASD→`set_config_ctl_move_*`, the `PlayerProgress` upgrade levels, the level-up/menu/death state machine, and builds a `UiModel` + per-frame `UiData` from sim readback. `engine_ui::draw` renders it and returns `UiAction`s the host applies (increment a level → push via `set_config_ctl_*`; restart → re-seed).

**Tech Stack:** Rust, `voxel_engine` (winit + ash + egui), `egui = "0.33"`, `engine_ui`. **Depends on Plan 1 (setters) + Plan 2 (engine_ui).**

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - no-op overlay call at `crates/viewer_runtime/src/bin/vs_viewer.rs:338`
  - `voxel_engine::EguiState` (`new`/`handle_window_event`/`run`/`cmd_paint`/`destroy`) at `/home/ricky/Projects/voxel_engine/src/ui/mod.rs:45`
  - `VsViewerApp`/`VsBridge`/`read_*` readback at `crates/viewer_runtime/src/vs.rs`
  - `view_storage_xp_primary_buf` + `fold_view_xp_handles()` on the generated runtime
  - Search method: `Read` + Explore.
- **Decision:** extend `vs_viewer` + `vs.rs`; new `vs_ui.rs` for game-UI state; consume `engine_ui`. No engine-rule code.
- **Rule-compiler touchpoints:** none — viewer/presentation only.
- **Hand-written downstream code:** `vs_ui.rs` + viewer edits — justified: presentation + host input policy, not sim rules (P1 governs rules). Plan 5 moves the *UI declaration* into the DSL; the viewer plumbing remains legitimately hand-written.
- **Constitution check:**
  - P1: PASS — no `impl Rule`/handler code; gameplay stays in the `.sim`.
  - P2/P3/P4/P5/P6/P7/P11: N/A — no sim-state/event/kernel change; input flows through Plan 1's sanctioned setters.
  - P10 (No Runtime Panic): PASS — headless viewer smoke + the per-frame loop guard.
  - P8: PASS — this section.
- **Runtime gate:**
  - `vs_viewer_headless_smoke` at `crates/viewer_runtime/tests/smoke_vs.rs` — construct the playable viewer state, push input, step a few ticks + build `UiData`, no panic (skips without GPU).
  - `level_up_menu_logic` at `crates/viewer_runtime/src/vs_ui.rs` (test mod) — pure: crossing a level threshold opens the menu; applying an `Increment` raises the matching `PlayerProgress` field.
- **Re-evaluation:** [x] AIS reviewed at design phase.  [ ] AIS reviewed post-design.

---

### Task 1: Dependencies + egui wiring (static HUD on screen)

**Files:**
- Modify: `crates/viewer_runtime/Cargo.toml`, `crates/viewer_runtime/src/bin/vs_viewer.rs`

- [ ] **Step 1: Add deps.** In `crates/viewer_runtime/Cargo.toml`:
```toml
engine_ui = { path = "../engine_ui" }
egui = "0.33"
```

- [ ] **Step 2: Construct `EguiState` in `resumed()`.** After the `SwapchainContext` is built, add an `egui: EguiState` to the `Gfx` struct, constructed with the swapchain's format, image views, and extent (find the accessors on `SwapchainContext` — e.g. `swapchain.format()`, `swapchain.image_views()`, `swapchain.extent()`; confirm exact names) and `&window`:
```rust
let egui = voxel_engine::ui::EguiState::new(&ctx, swapchain.format(), swapchain.image_views(), swapchain.extent(), &window)?;
```

- [ ] **Step 3: Forward events to egui.** At the top of the `window_event()` handler, before key/close handling:
```rust
if let Some(gfx) = self.gfx.as_mut() {
    let resp = gfx.egui.handle_window_event(&gfx.window, &event);
    if resp.consumed { return; } // egui ate it (e.g. a menu click)
}
```

- [ ] **Step 4: Run egui + paint in the overlay closure.** Replace the no-op `present_blit_with_overlay` closure (`:338`). Before present, call `run`; inside the closure call `cmd_paint`:
```rust
let cmd_pool = gfx.ctx.command_pool();        // confirm accessor
let queue = gfx.ctx.graphics_queue();         // confirm accessor
gfx.egui.run(&gfx.window, |ectx| {
    egui::Area::new(egui::Id::new("hud_probe")).fixed_pos(egui::pos2(12.0,12.0))
        .show(ectx, |ui| { ui.label("HUD online"); });
});
let egui_ref = &mut gfx.egui;
let ctx_ref = &gfx.ctx;
gfx.swapchain.present_blit_with_overlay(
    &gfx.ctx, gfx.renderer.light_output_image(), WINDOW_W, WINDOW_H,
    ash::vk::Semaphore::null(),
    |cmd, image_index| egui_ref.cmd_paint(ctx_ref, cmd, image_index, queue, cmd_pool),
)
```
(Resolve borrow-checker conflicts by capturing the needed fields before the closure; the closure is `FnOnce`.)

- [ ] **Step 5: Verify visually.** Run: `cargo run -p viewer_runtime --bin vs_viewer --release`. Expected: the voxel arena renders with "HUD online" text overlaid top-left. Commit.
```bash
git add crates/viewer_runtime/Cargo.toml crates/viewer_runtime/src/bin/vs_viewer.rs
git commit -m "feat(viewer): wire egui overlay into vs_viewer (static HUD probe)"
```

### Task 2: Live HUD data + follow-cam + WASD movement

**Files:**
- Modify: `crates/viewer_runtime/src/vs.rs`, `crates/viewer_runtime/src/bin/vs_viewer.rs`
- Create: `crates/viewer_runtime/src/vs_ui.rs`

- [ ] **Step 1: Expose player stats + xp readback on `VsViewerApp`.** In `vs.rs`, add a method that returns the player snapshot and the xp view value. Characterize the `xp` buffer encoding first:
```rust
impl VsViewerApp {
    pub fn player_hp(&self) -> f32 { self.agents.iter().find(|a| a.role == VsRole::Player).map(|a| a.hp).unwrap_or(0.0) }
    pub fn enemy_count(&self) -> usize { self.agents.iter().filter(|a| a.role == VsRole::Enemy).count() }
    pub fn alive(&self) -> bool { self.agents.iter().any(|a| a.role == VsRole::Player) }
    /// Player XP from the materialized view. Reads view_storage_xp_primary_buf[PLAYER_SLOT].
    pub fn player_xp(&mut self) -> f32 {
        let buf = self.state.view_storage_xp_primary_buf.clone();
        let raw = read_u32(&mut self.state, &buf, sims::vampire_survivors_seed::PLAYER_SLOT + 1);
        let w = raw[sims::vampire_survivors_seed::PLAYER_SLOT as usize];
        // CHARACTERIZE: fold `self += 1.0` may store f32 bits or an integer count.
        // Try f32::from_bits first; if values look like small ints (1,2,3..) it's a u32 count — use `w as f32`.
        f32::from_bits(w)
    }
}
```
Add a one-off test to pin the encoding: kill-count vs `player_xp()` after a fixed run; assert it's monotonic and matches kills. Adjust the decode line accordingly.

- [ ] **Step 2: `vs_ui.rs` — host game state + UiData builder + UiModel.**
```rust
// crates/viewer_runtime/src/vs_ui.rs
use engine_ui::{UiModel, UiData, Widget, Screen, NamedScreen, Card, UiAction};

pub const XP_PER_LEVEL: f32 = 5.0; // mirror config.vs.xp_per_level

#[derive(Default, Clone)]
pub struct PlayerProgress {
    pub bolt_level: f32, pub bolt_rate_level: u32, pub nova_level: f32,
    pub move_level: f32, pub garlic_level: f32, pub whip_level: f32,
    pub last_level: u32, pub kills: u32,
}
impl PlayerProgress {
    pub fn level(xp: f32) -> u32 { (xp / XP_PER_LEVEL).floor() as u32 }
    /// Returns true if a new level was reached since last check (opens the menu).
    pub fn check_level_up(&mut self, xp: f32) -> bool {
        let lv = Self::level(xp);
        let up = lv > self.last_level;
        self.last_level = lv;
        up
    }
    pub fn apply(&mut self, action: &UiAction) {
        if let UiAction::Increment(k) = action {
            match k.as_str() {
                "bolt_level" => self.bolt_level += 1.0,
                "bolt_rate_level" => self.bolt_rate_level += 1,
                "nova_level" => self.nova_level += 1.0,
                "move_level" => self.move_level += 1.0,
                "garlic_level" => self.garlic_level += 1.0,
                "whip_level" => self.whip_level += 1.0,
                _ => {}
            }
        }
    }
}

pub fn hud_model() -> UiModel {
    UiModel {
        hud: vec![
            Widget::Bar { label: "HP".into(), value: "hp".into(), max: "hp_max".into(), color: [220,40,40] },
            Widget::Bar { label: "XP".into(), value: "xp_into".into(), max: "xp_per_level".into(), color: [40,160,240] },
            Widget::Text { template: "Lv {level}   Kills {kills}   {time}s   Enemies {enemies}".into() },
        ],
        screens: vec![], // menu/death added in Task 3/4
    }
}

pub fn build_data(d: &mut UiData, hp: f32, hp_max: f32, xp: f32, kills: u32, tick: u64, enemies: usize) {
    d.set("hp", hp).set("hp_max", hp_max)
     .set("level", PlayerProgress::level(xp) as f32)
     .set("xp_into", xp % XP_PER_LEVEL).set("xp_per_level", XP_PER_LEVEL)
     .set("kills", kills as f32).set("time", (tick as f32)*0.1).set("enemies", enemies as f32);
}
```

- [ ] **Step 3: Push WASD into the sim + follow-cam, render the live HUD.** In `vs_viewer.rs`: track held keys (it already has `held_keys`), and each tick before `app.step()` compute a normalized move vector and push it:
```rust
let (mut mx, mut my) = (0.0f32, 0.0f32);
if self.held_keys.contains("w") { my += 1.0; } if self.held_keys.contains("s") { my -= 1.0; }
if self.held_keys.contains("d") { mx += 1.0; } if self.held_keys.contains("a") { mx -= 1.0; }
let len = (mx*mx+my*my).sqrt(); if len > 1e-3 { mx/=len; my/=len; }
self.app.state.set_config_ctl_move_x(mx);
self.app.state.set_config_ctl_move_y(my);
// also push current progress levels so weapons reflect picks:
self.app.state.set_config_ctl_bolt_level(self.progress.bolt_level);
self.app.state.set_config_ctl_bolt_rate_level(self.progress.bolt_rate_level);
self.app.state.set_config_ctl_nova_level(self.progress.nova_level);
self.app.state.set_config_ctl_move_level(self.progress.move_level);
self.app.state.set_config_ctl_garlic_level(self.progress.garlic_level);
self.app.state.set_config_ctl_whip_level(self.progress.whip_level);
```
Follow-cam: re-target `self.camera` on the player's voxel position each frame (use `vs::vs_world_to_voxel(player.pos)` mapped to renderer XZ). In the `run` closure, replace the probe with `engine_ui::draw(ectx, &self.ui_model, &self.ui_data, self.active_screen.as_deref())` (build `self.ui_data` via `build_data` just before).

- [ ] **Step 4: Headless smoke + unit test.** In `crates/viewer_runtime/tests/smoke_vs.rs` add a no-GPU-safe smoke that builds `PlayerProgress`, calls `build_data`, and `engine_ui::draw` in a headless `egui::Context` (no panic). Add the `level_up_menu_logic` pure unit test in `vs_ui.rs`. Run: `cargo test -p viewer_runtime`. Expected PASS.

- [ ] **Step 5: Commit.**
```bash
git add crates/viewer_runtime/src/vs.rs crates/viewer_runtime/src/vs_ui.rs crates/viewer_runtime/src/bin/vs_viewer.rs crates/viewer_runtime/tests/smoke_vs.rs
git commit -m "feat(viewer): live HUD + follow-cam + WASD->config.ctl movement"
```

### Task 3: Level-up menu (pause + pick)

**Files:**
- Modify: `crates/viewer_runtime/src/vs_ui.rs`, `crates/viewer_runtime/src/bin/vs_viewer.rs`

- [ ] **Step 1: Add the menu screen + seeded 3-card draw.** In `vs_ui.rs`:
```rust
pub const UPGRADE_POOL: &[(&str,&str)] = &[
    ("Bolt Damage +","bolt_level"), ("Bolt Rate +","bolt_rate_level"),
    ("Nova +","nova_level"), ("Move Speed +","move_level"),
    ("Garlic +","garlic_level"), ("Whip +","whip_level"),
];
/// Pick 3 distinct cards using a seeded index (per_agent_u32-style determinism).
pub fn menu_screen(seed: u64, level: u32) -> NamedScreen {
    let mut idxs = vec![]; let mut s = seed ^ (level as u64).wrapping_mul(0x9E3779B97F4A7C15);
    while idxs.len() < 3 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let i = (s >> 33) as usize % UPGRADE_POOL.len();
        if !idxs.contains(&i) { idxs.push(i); }
    }
    let cards = idxs.into_iter().map(|i| {
        let (label, key) = UPGRADE_POOL[i];
        Card { label: label.into(), action: UiAction::Increment(key.into()) }
    }).collect();
    NamedScreen { name: "level_up".into(), screen: Screen::Menu { title: "Level Up!".into(), cards } }
}
```

- [ ] **Step 2: Drive the state machine in `vs_viewer.rs`.** Each frame, after computing `xp`:
```rust
if self.active_screen.is_none() && self.progress.check_level_up(xp) {
    // rebuild the model's screens with a fresh menu, pause.
    self.ui_model.screens = vec![vs_ui::menu_screen(self.app.seed, self.progress.last_level)];
    self.active_screen = Some("level_up".into());
    self.paused = true;
}
```
When `engine_ui::draw` returns `Some(action)` while the menu is active: `self.progress.apply(&action); self.active_screen = None; self.paused = false;` (do not `step()` while `self.active_screen.is_some()`).

- [ ] **Step 3: Verify.** `cargo test -p viewer_runtime` (the pure `level_up_menu_logic` test covers threshold + apply). Manual: play until a level-up, see 3 cards, pick one, confirm that weapon visibly changes (e.g. Garlic shows continuous nearby kills). Commit.
```bash
git add crates/viewer_runtime/src/vs_ui.rs crates/viewer_runtime/src/bin/vs_viewer.rs
git commit -m "feat(viewer): level-up menu pauses run and applies upgrade picks"
```

### Task 4: Death screen + restart

**Files:**
- Modify: `crates/viewer_runtime/src/vs_ui.rs`, `crates/viewer_runtime/src/bin/vs_viewer.rs`

- [ ] **Step 1: Add the End screen.** In `hud_model` (or a helper), include:
```rust
NamedScreen { name: "dead".into(), screen: Screen::End {
    title: "You Died".into(),
    summary: vec![("Time".into(),"time".into()), ("Level".into(),"level".into()), ("Kills".into(),"kills".into())],
    restart_label: "Restart (R)".into(),
}}
```

- [ ] **Step 2: Trigger + restart in `vs_viewer.rs`.** Each frame: `if !self.app.alive() && self.active_screen.as_deref() != Some("dead") { self.active_screen = Some("dead".into()); self.paused = true; }`. On `UiAction::Restart` (or the `R` key): re-seed — `self.app = VsViewerApp::try_new(self.seed.wrapping_add(1)).unwrap(); self.progress = Default::default(); self.active_screen = None; self.paused = false;`.

- [ ] **Step 3: Verify.** Manual: die (stand in the swarm), see the summary, press Restart → fresh run. Headless smoke still green. Commit.
```bash
git add crates/viewer_runtime/src/vs_ui.rs crates/viewer_runtime/src/bin/vs_viewer.rs
git commit -m "feat(viewer): death summary screen + restart"
```

## Self-review note
Setter names (`set_config_ctl_*`) and the `view_storage_xp_primary_buf` field/`fold_view_xp_handles()` accessor must match Plans 1/3 and the generated runtime; reconcile against actual generated output. `XP_PER_LEVEL` mirrors `config.vs.xp_per_level` (5.0) — keep in sync. `active_screen`, `paused`, `progress`, `ui_model`, `ui_data` are new fields on the `WindowedVsViewer` struct.
