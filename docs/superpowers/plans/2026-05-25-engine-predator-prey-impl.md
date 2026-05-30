# Convert predator_prey to a Playable Game (Plan F — Wave 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Make `predator_prey` playable as the Hare (evade the wolves) through the generic path with **zero new Rust** — the test that the declarative layer isn't vampire-survivors-in-disguise.

**Architecture:** Add a `@runtime config ctl` + a `HareControl` rule (drive the player's Hare by input, replacing autonomous `MoveHare` for it), plus `render {}` / `controls {}` / `ui {}` blocks. `play predator_prey` runs on the same generic binary. Structurally different from VS: evade (no weapons/waves/upgrades), a survive-timer/score HUD, `entity Hare/Wolf` instead of mana-band.

**Tech Stack:** custom DSL only. **Depends on Plan D** (working generic path). Parallel with Plan E (different `.sim` file). **No Rust** beyond what the engine already provides.

---

## Architectural Impact Statement
- **Existing primitives searched:** `assets/sim/predator_prey.sim` (`entity Hare/Wolf : Agent`, `config hunt`, `MoveHare`/the prey movement rule, `closest_prey` query); the generic path (Plans A–D); the playable-VS `@runtime`/blocks idioms. Method: `Read`.
- **Decision:** author a second playable game purely in the `.sim` — the generality proof; surfaces any declarative-layer gap (feeds back to Plan A/B as a small extension if needed).
- **Rule-compiler touchpoints:** `assets/sim/predator_prey.sim` only.
- **Hand-written downstream code:** NONE — the whole point.
- **Constitution check:** P1 PASS (all in `.sim`); P2 N/A (ctl is cfg, not SoA); P3/P5/P6 PASS; P10 PASS (the headless exec smoke); P8 PASS.
- **Runtime gate:** `predator_prey_playable` (`crates/sims/tests/`) — `make_playable("predator_prey")`, set `ctl.move_x`, step, assert the player Hare moves by input (not autonomous flee); the descriptors emit + parse.
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: Input-driven Hare
**Files:** Modify `assets/sim/predator_prey.sim`.

- [ ] **Step 1:** Add a `@runtime config ctl { move_x: f32 = 0.0, move_y: f32 = 0.0 }` block.
- [ ] **Step 2:** Designate one Hare as the player (slot convention mirroring VS's `PLAYER_SLOT`, e.g. the lowest-slot Hare) and add a `HareControl @phase(per_agent)` rule that, for the player Hare, sets pos from `config.ctl.move_*` at `config.hunt.prey_speed`, with the existing arena/bounds clamp; gate the autonomous `MoveHare` off for the player Hare (a disjoint `where` guard, mirroring how VS's `PlayerControl` replaced `KitePlayer`). Other Hares keep autonomous movement.
- [ ] **Step 3:** `cargo build -p sims`; `crates/sims/tests/predator_prey_playable.rs` (or extend an existing pp test): `make_playable("predator_prey", SEED, N)`, seed, `set_input("ctl.move_x", 1.0)`, step ~10, assert the player Hare's x increased (input-driven, not fleeing). `RUST_MIN_STACK=33554432 cargo test -p sims --test predator_prey_playable`. Commit.

### Task 2: render / controls / ui blocks
**Files:** Modify `assets/sim/predator_prey.sim`.

- [ ] **Step 1:** `render {}` — arena from the pp bounds; follow-cam on the player Hare; Hare green / Wolf red by `creature_type` (or the Hare/Wolf discriminant the snapshot exposes); **no weapon VFX** (a deliberate difference from VS — proves vfx is optional).
- [ ] **Step 2:** `controls {}` — WASD→`ctl.move_x/move_y`.
- [ ] **Step 3:** `ui {}` — a survive-timer text (`{time}s`) + a "wolves nearby"/score readout via a `view_value` (reuse an existing pp materialized view, e.g. a kill/danger count) + a death/end screen on the player Hare's death. (If a needed value isn't exposed, that's a real gap → add the column to the snapshot in Plan B/the trait, the single sanctioned extension.)
- [ ] **Step 4:** `cargo build -p sims`; compile-gate that all three descriptors emit + parse. Commit.

### Task 3: Manual end-to-end (user-side) — the generality proof
- [ ] **Step 1:** `cargo run -p engine_play --bin play predator_prey` on a desktop — drive the Hare with WASD, evade wolves, survive timer, death screen. (Headless env: skip; user runs it.) **Success here = the engine authored a second, structurally-different game with zero new Rust.** No commit (manual gate).

## Self-review note
predator_prey is intentionally different (evade, no weapons/waves, entity-subkind identity, survive-timer win) to stress the abstractions. Expect it to surface 0–2 small declarative-layer gaps; the sanctioned fix is extending the snapshot columns or a render/ui field — NOT adding per-game Rust. If a gap needs more than that, log it as a direction-#1 follow-up rather than special-casing predator_prey.
