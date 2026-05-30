# Subkind Seeding — predator_prey Migration (Plan C — Wave 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Seed predator_prey declaratively (a player Hare + autonomous Hares + Wolves at scattered positions) and gate it by subkind, so `play predator_prey` is a real game — completing the zero-Rust generality proof.

**Architecture:** Add a `spawn` `init {}` block; make the player a distinct `PlayerHare` subkind (count 1); switch the mana-band/`mana < 0.5` player hack and all guards to `creature_type`. **Depends on Plan A**. Parallel with Plan B (different `.sim`).

**Tech Stack:** custom DSL; `sims` GPU runtime. No Rust.

---

## Architectural Impact Statement
- **Existing primitives searched:** `predator_prey.sim` (`entity Hare/Wolf : Agent`, `config hunt`, `HareControl`/`MoveHare`/the wolf rules/`StrikePrey`, the `config ctl` + render/controls/ui from Plan F, the `mana < 0.5` player hack); Plan A's `spawn`/`creature_type is` grammar. Method: `Read`.
- **Decision:** add a `PlayerHare` subkind + subkind seeding/gating; retire the `mana < 0.5` player band. No Rust.
- **Rule-compiler touchpoints:** `assets/sim/predator_prey.sim`.
- **Hand-written downstream code:** NONE.
- **Constitution check:** P1 PASS; P2 N/A; P3/P5/P6 PASS; P10 PASS; P8 PASS.
- **Runtime gate:** `predator_prey_playable` (adapted) — `make_playable("predator_prey")` seeds 1 PlayerHare + N Hares + M Wolves (by creature_type) at positions within the scatter radius; `set_input("ctl.move_x",1.0)` moves the PlayerHare +X.
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: PlayerHare subkind + `spawn` seeding
**Files:** `assets/sim/predator_prey.sim`.

- [ ] **Step 1:** Declare `entity PlayerHare : Agent { pos: vec3, vel: vec3 }` (the human-controlled hare, distinct from autonomous `Hare`).
- [ ] **Step 2:** Add the seeding block (counts via `config.hunt` or literals; sum + 1 ≤ agent_count):
```
init {
  spawn PlayerHare count 1   { pos: origin }
  spawn Hare       count 199 { pos: scatter(config.hunt.arena_radius) }
  spawn Wolf       count 8   { pos: scatter(config.hunt.arena_radius) }
}
```
- [ ] **Step 3:** `cargo build -p sims`. Commit.

### Task 2: Subkind gating (retire the `mana < 0.5` hack)
**Files:** `assets/sim/predator_prey.sim`.

- [ ] **Step 1:** `HareControl` guard → `self.creature_type == PlayerHare` (drop `self.mana < 0.5`). The autonomous `MoveHare` guard → `self.creature_type == Hare` (drop the `self.mana >= 0.5` disjoint guard; PlayerHare and Hare are now distinct subkinds, so the write→write SCC split is by subkind). Wolf rules (`MoveWolf`/`StrikePrey`) guard `self.creature_type == Wolf`. The `closest_prey` spatial filter → `candidate.creature_type == Hare || candidate.creature_type == PlayerHare` (wolves hunt both) — or keep the existing `is_prey_of` view if it already expresses predator/prey by creature_type.
- [ ] **Step 2:** Drop the `mana: slot` / `mana < 0.5` seeding remnants from Plan F (the `init { alive:1, mana: slot }` flat block is replaced by the `spawn` blocks). `cargo build -p sims`. Commit.

### Task 3: Render by subkind + exec test
**Files:** `assets/sim/predator_prey.sim`, `crates/sims/tests/predator_prey_playable.rs`.

- [ ] **Step 1:** `render {}`: `agent when creature_type is PlayerHare { color (40,220,90) }` (bright green player), `creature_type is Hare { color (80,200,90) }` (dim green), `creature_type is Wolf { color (210,60,50) }` (red). `camera follow when creature_type is PlayerHare`. No vfx (deliberate).
- [ ] **Step 2:** Adapt `predator_prey_playable.rs`: construct via `make_playable("predator_prey", SEED, N)` (now self-seeds — drop any manual seed); assert by `creature_type` (read `agent_creature_type_buf`): exactly 1 PlayerHare, N Hares, M Wolves, all alive, Hare/Wolf positions within the scatter radius. Keep `player_hare_tracks_input`: `set_input("ctl.move_x", 1.0)`, step, the PlayerHare moves +X.
- [ ] **Step 3:** `RUST_MIN_STACK=33554432 cargo test -p sims --test predator_prey_playable` — green. Commit.

### Task 4: Manual run (user-side) — generality proof complete
- [ ] **Step 1:** `cargo run -p engine_play --bin play predator_prey` on a desktop: drive the green PlayerHare with WASD, evade the red Wolves, survive-timer HUD, death screen. (Headless env: skip; user runs it.) **Success = the engine authored a second, structurally-different game with zero Rust, fully seeded + gated by subkind.** No commit (manual gate).

## Self-review note
PlayerHare as a distinct subkind cleanly replaces the `mana < 0.5` hack and the autonomous/player MoveHare SCC split (now subkind-disjoint). If `is_prey_of`/`closest_prey` already key on creature_type via the `predator_prey(a,b)` stdlib, the wolf-hunts-both-hare-kinds filter may need both subkinds listed — confirm against the existing predator/prey relation.
