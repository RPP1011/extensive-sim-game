# Migrate Vampire Survivors to the Generic Path (Plan E — Wave 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Add `render {}` / `controls {}` / `ui {}` blocks to `vampire_survivors.sim` so `play vampire_survivors` reaches parity with the old `vs_viewer`, then **delete** the bespoke `vs_viewer.rs` / `VsBridge` / `vs_ui.rs`.

**Architecture:** Translate the hand-written VS viewer behavior into declarations: the mana-band role→color (`VsBridge`), WASD wiring (`vs_viewer`), HUD/menu/death (`vs_ui` + Plan 4), and the bolt/nova/garlic/whip VFX into `render{}.vfx`. Then remove the now-redundant Rust.

**Tech Stack:** custom DSL; `engine_play`. **Depends on Plan D** (working generic path). Runs parallel with Plan F (different `.sim` file).

---

## Architectural Impact Statement
- **Existing primitives searched:** `vampire_survivors.sim` (current rules + `config ctl`); the deleted-target files `viewer_runtime/src/{bin/vs_viewer.rs, vs.rs, vs_ui.rs}`; the sample VS descriptors (`engine_play_api/fixtures/`) as the translation reference. Method: `Read`.
- **Decision:** move VS's viewer/controls/UI from Rust into `.sim` declarations; delete the Rust (the spec's core deletion).
- **Rule-compiler touchpoints:** `assets/sim/vampire_survivors.sim` (+ three blocks); generated VS runtime re-emits descriptors.
- **Hand-written downstream code:** NONE added; net **removal** of `vs_viewer.rs`/`VsBridge`/`vs_ui.rs`.
- **Constitution check:** P1 PASS (UI/render/controls now compiler-lowered); P2 N/A; P3/P5/P6 PASS; P10 PASS (Plan D's path gated); P8 PASS.
- **Runtime gate:** the existing `vampire_survivors_exec` GPU tests still pass (gameplay unchanged); `make_playable("vampire_survivors")` end-to-end via the Plan-D `update()` smoke; manual parity run.
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: Add the three blocks to vampire_survivors.sim
**Files:** Modify `assets/sim/vampire_survivors.sim`.

- [ ] **Step 1:** Add `render {}` mirroring the sample fixture (`engine_play_api/fixtures/vs_render.json`): arena `config.vs.arena_radius`; follow-cam on the player mana-band; player cyan / enemy orange / swift-yellow / brute-red by mana+move_speed ranges; `vfx on NovaFire period 40 { ring radius config.vs.nova_radius color (255,255,120) }`, `vfx on BoltFire period config.vs.bolt_period { beam_to_nearest when mana in [1.5,2.5] color (200,255,255) }`, plus garlic/whip rings. (Translate the hand-coded VFX I added in the VFX commit.)
- [ ] **Step 2:** Add `controls {}` — WASD→`ctl.move_x/move_y` (the `vs_viewer` mapping).
- [ ] **Step 3:** Add `ui {}` — the HP/XP bars + `Lv/Kills/time/Enemies` text + the level-up menu (bolt/nova/rate/move/garlic/whip cards) + death screen (the Plan-4 `vs_ui` model, now declared).
- [ ] **Step 4:** `cargo build -p sims`; extend `vampire_survivors_compile.rs` with a gate asserting all three descriptors emit + parse. Commit.

### Task 2: Parity check + delete the bespoke Rust
**Files:** Delete `crates/viewer_runtime/src/bin/vs_viewer.rs`, `crates/viewer_runtime/src/vs_ui.rs`; remove `VsBridge` + VS-only helpers from `crates/viewer_runtime/src/vs.rs` (or delete `vs.rs` if nothing else uses it); update `viewer_runtime`'s `lib.rs`/`Cargo.toml` (drop egui/engine_ui deps if now unused); remove the `vs_viewer` bin target.

- [ ] **Step 1 (manual parity gate):** `cargo run -p engine_play --bin play vampire_survivors` on a desktop — confirm parity with the old viewer: WASD movement, HUD, level-up menu changing weapons, death+restart, all four weapon VFX. (Headless env: skip; user runs it.) DO NOT delete until parity confirmed.
- [ ] **Step 2:** Delete `bin/vs_viewer.rs` + `vs_ui.rs`; excise `VsBridge`/`material_for_vs`/`vs_world_to_voxel`/the VS palette + VFX from `vs.rs` (keep any genuinely-shared helper, else remove the file). Keep `VsViewerApp`/seeding only if still referenced by tests; otherwise the `sims` exec tests already cover gameplay.
- [ ] **Step 3:** `cargo build` (workspace) + `cargo test -p sims -p engine_play -p viewer_runtime`. Fix fallout (remove dangling `mod vs_ui;`, smoke tests referencing deleted types — port the still-valuable ones to `engine_play` or delete). Run the existing `vampire_survivors_exec` GPU tests — must still pass (gameplay untouched). Commit:
```bash
git add -A
git commit -m "refactor(vs): migrate vampire_survivors to generic engine_play; delete vs_viewer/VsBridge/vs_ui"
```

## Self-review note
This is the spec's headline deletion — only after the Plan-D path works and the manual parity gate passes. If a VS behavior can't be expressed declaratively yet (e.g. an exotic VFX), document it and keep the minimum Rust, but the default is full deletion. Plan F (predator_prey) is independent of this file and runs in parallel.
