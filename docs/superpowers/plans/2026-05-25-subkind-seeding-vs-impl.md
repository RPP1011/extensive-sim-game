# Subkind Seeding — Vampire Survivors Migration (Plan B — Wave 2)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Seed vampire_survivors declaratively (player + Enemy pool) and gate it by subkind, so `make_playable("vampire_survivors")` / `play vampire_survivors` is a real game — the parity gate that unblocks deleting `vs_*`.

**Architecture:** Add a `spawn` `init {}` block and switch every mana-band guard/filter + the `render` block to `creature_type`. The Enemy pool is seeded `alive: 0`; the wave drain flips it `alive: 1` (drain never touches `creature_type`, so the seeded `Enemy` subkind persists — no drain change). **Depends on Plan A** (the grammar). Parallel with Plan C (different `.sim`).

**Tech Stack:** custom DSL; `sims` GPU runtime.

---

## Architectural Impact Statement
- **Existing primitives searched:** `vampire_survivors.sim` (mana-band guards, `config ctl`, the render/controls/ui blocks from engine Plan A, the `Player`/`Enemy` subkind decls); `summon_alloc.rs` drain (writes alive/hp/pos/move_speed, not creature_type); Plan A's `spawn`/`creature_type is` grammar. Method: `Read`.
- **Decision:** migrate VS to subkind seeding + gating; retire its mana-band idiom. No Rust.
- **Rule-compiler touchpoints:** `assets/sim/vampire_survivors.sim`; generated VS runtime re-emits seeding + descriptors.
- **Hand-written downstream code:** NONE.
- **Constitution check:** P1 PASS (all `.sim`); P2 N/A; P3/P5/P6 PASS; P10 PASS (the exec test). P8 PASS.
- **Runtime gate:** `vampire_survivors_exec` (adapted) — `make_playable`/`try_new` seeds a live player (1 agent, creature_type Player) + Enemy pool (alive=0); over T ticks with the drain, enemy count (by creature_type Enemy) grows then is culled; player tracks input.
- **Re-evaluation:** [x] design. [ ] post-design.

---

### Task 1: Seed via `spawn` + gate by subkind
**Files:** `assets/sim/vampire_survivors.sim`.

- [ ] **Step 1:** Add the seeding block:
```
init {
  spawn Player count 1   { hp: config.vs.player_hp, pos: origin }
  spawn Enemy  count 511 { alive: 0 }
}
```
(Enemy count = `agent_count(512) − 1 player − slot0 sentinel`; adjust to the runtime's agent_count. The drain claims these `alive: 0` Enemy-subkind slots.)
- [ ] **Step 2:** Replace every player-band guard `self.mana >= config.vs.player_mana_min && self.mana <= config.vs.player_mana_max` with `self.creature_type == Player` (rules: `PlayerControl`, `BoltFire`, `NovaFire`, `GarlicAura`, `WhipSweep`, the `ChooseUpgrade`-removed set, and the `Spawn*` verbs' `when`). Replace enemy-band guards `... enemy_mana_min/max` with `self.creature_type == Enemy` (`ChasePlayer`) and spatial-query candidate filters `candidate.mana ∈ enemy_band` with `candidate.creature_type == Enemy` (`closest_enemy`, `enemies_in_radius`, `garlic_targets`, `whip_targets`).
- [ ] **Step 3:** Drop the now-unused `config vs` mana-band fields (`player_mana_min/max`, `enemy_mana_min/max`) and any `mana`/`engaged_with` seeding that the mana-band relied on. (Enemies home via `engaged_with`; if the chase needs `engaged_with = player slot`, seed it: `spawn Enemy count 511 { alive: 0, engaged_with: 1 }` — confirm the player lands at slot 1 given the slot-0 skip.)
- [ ] **Step 4:** `cargo build -p sims`; extend `vampire_survivors_compile.rs` gates (the `player_control_reads_input` etc. still pass; add one asserting the render descriptor keys on `creature_type`). Commit.

### Task 2: Render by subkind + exec test
**Files:** `assets/sim/vampire_survivors.sim`, `crates/sims/tests/vampire_survivors_exec.rs`.

- [ ] **Step 1:** In the `render {}` block, replace `agent when mana in [0.5,1.5]` → `agent when creature_type is Player`, `mana in [1.5,2.5]` → `creature_type is Enemy` (keep the swift/brute variants keyed on `move_speed` as-is, or move them to enemy sub-subkinds later — out of scope).
- [ ] **Step 2:** Adapt `vampire_survivors_exec.rs`: drop the manual `seed_initial_state` call (seeding is now in the `.sim` `init`); construct via `make_playable("vampire_survivors", SEED, 512)` (or `GeneratedRuntime::try_new`, which now seeds). Update `enemy_count` to filter by `creature_type == Enemy` (read `agent_creature_type_buf`) instead of the mana band. Keep `player_tracks_input` + `playable_loop_survivable` asserting: a live player post-seed, input-driven movement, enemy count grows under the drain then bounded.
- [ ] **Step 3:** `RUST_MIN_STACK=33554432 cargo test -p sims --test vampire_survivors_exec` — green. Commit.

### Task 3: Manual parity gate (user-side) → unblock Plan E
- [ ] **Step 1:** `cargo run -p engine_play --bin play vampire_survivors` on a desktop: a seeded player kiting a wave-spawned swarm, HUD/menu/death working. (Headless env: skip; user runs it.) **This is the parity gate for the engine spec's Plan E** (delete `vs_viewer`/`VsBridge`/`vs_ui`). No commit (manual gate).

## Self-review note
The drain is unchanged — it relies on the Enemy pool being pre-seeded `creature_type = Enemy, alive: 0`. If the drain ever zeroed `creature_type`, this breaks; the `vampire_survivors_exec` "enemy count grows" assertion (filtering by creature_type) catches it. Player slot: with the slot-0 skip, `spawn Player count 1` lands the player at slot 1 — keep `engaged_with: 1` consistent.
