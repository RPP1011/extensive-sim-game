# Engine Integration Implementation Plan (Plan D — Wave 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Wire the real `sims::make_playable` registry into the `play` binary (replacing Plan B's `MockRuntime`) and prove the generic path end-to-end against a **minimal probe `.sim`** that declares all three blocks — so VS and predator_prey migrations (Plans E/F) can both build on a proven path.

**Architecture:** Swap the binary's runtime source from `MockRuntime` to `sims::make_playable(argv)`. Add a tiny `assets/sim/play_probe.sim` (one player-controlled agent + a couple of static agents + render/controls/ui blocks) as the end-to-end fixture that doesn't depend on either real game's migration.

**Tech Stack:** Rust; `engine_play` + `sims` + the compiled probe runtime. **Depends on Plans A + B + C.**

---

## Architectural Impact Statement
- **Existing primitives searched:** `sims::make_playable` (Plan A); `engine_play::bin/play.rs` + `player.rs` (Plan B); the descriptor `from_json`s (Plans 0/C). Method: `Read`.
- **Decision:** connect the three Wave-1 tracks; presentation/runtime wiring, no rule code.
- **Rule-compiler touchpoints:** `assets/sim/play_probe.sim` (new fixture) + `sims/build.rs` (register it).
- **Hand-written downstream code:** binary registry wire — sanctioned runtime glue.
- **Constitution check:** P1 PASS; P2 N/A; P3 PASS; P5/P6 PASS; P10 PASS (the end-to-end smoke); P8 PASS.
- **Runtime gate:** `play_probe_end_to_end` (`crates/engine_play/tests/`) — construct via `make_playable("play_probe")`, run the separable per-frame `update()` N times with a held key, assert the input reached the runtime and the snapshot/UiData are coherent; no panic.
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: Minimal probe fixture with all three blocks
**Files:** Create `assets/sim/play_probe.sim`; Modify `crates/sims/build.rs`.

- [ ] **Step 1:** `play_probe.sim` — a `@runtime config ctl { move_x, move_y: f32 = 0.0 }`, one `PlayerMove` rule (player-band agent moves by `config.ctl.move_*`), a couple of static agents, and minimal `render {}` (arena + color-by-mana for player/other), `controls {}` (WASD→ctl.move_*), `ui {}` (one HP text). Register `"play_probe"` in `build.rs`. Build `cargo build -p sims`.
- [ ] **Step 2 (test):** extend `dsl_compiler/tests/engine_descriptors_emit.rs` (or a probe-specific test) asserting the probe emits all three non-empty descriptors that parse via the Plan-0/C `from_json`s. Commit.

### Task 2: Wire the registry into `play`
**Files:** Modify `crates/engine_play/Cargo.toml` (add `sims` dep), `crates/engine_play/src/bin/play.rs`.

- [ ] **Step 1:** Add `sims = { path = "../sims" }` to `engine_play/Cargo.toml`. (Dependency direction stays acyclic: `engine_play_api` ← `sims` ← `engine_play`.)
- [ ] **Step 2:** `bin/play.rs`: parse `argv[1]` as the fixture name; `let rt = sims::make_playable(&name, seed, agents).unwrap_or_else(|| { eprintln!("unknown/failed fixture {name}"); std::process::exit(2); });` then run the Plan-B player loop with `rt`. Keep a `--mock` flag falling back to `MockRuntime` for windowless smoke. Build `cargo build -p engine_play`.
- [ ] **Step 3 (test):** `crates/engine_play/tests/play_probe_end_to_end.rs` — `make_playable("play_probe", SEED, 64)`, build the player state, run `update()` (the separable per-frame fn from Plan B) ~20 times with `held={"d"}`; assert the runtime received a `ctl.move_x` input (read back a snapshot showing the player moved +X) and no panic. `RUST_MIN_STACK=33554432 cargo test -p engine_play --test play_probe_end_to_end`. Commit.

### Task 3: Manual end-to-end (user-side)
- [ ] **Step 1:** `cargo run -p engine_play --bin play play_probe` on a desktop → a window where WASD moves the dot, HUD shows HP. (Headless env: skip; note for the user.) No commit (manual gate).

## Self-review note
Plan D proves the wiring on a probe, independent of VS/predator migrations — so Plans E and F (different `.sim` files) can both build on it in parallel. The `update()` separation (Plan B) is what makes the end-to-end test runnable headlessly.
