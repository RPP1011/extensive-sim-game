# Edgeworld Phase 0 — "It Lives, And You Can See It" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A zero-player survival sim where a small band of `Survivor` agents forage `FoodNode`s to fight rising hunger, starve when they fail, and food regrows — rendered as inspectable PNG frames showing a legible boom/bust.

**Architecture:** A new `.sim` fixture (`assets/sim/edgeworld.sim`) compiled by the existing `crates/sims` mega-crate (add it to the build.rs allow-list → `sims::edgeworld::GeneratedRuntime`). Behavior is pure DSL: per-tick `physics` folds for hunger/regrowth/starvation, a spatial neighbor query for perception, and three competing verbs (`Eat`/`SeekFood`/`Wander`) resolved by argmax. A render test seeds the world, steps it, and dumps top-down PNG frames + a population trace for visual verification.

**Tech Stack:** `.sim` DSL (compiled via `dsl_compiler` in `crates/sims/build.rs`), Rust GPU runtime (`wgpu`, `bytemuck`), `image` crate for PNG output (dev-dependency of `crates/sims`).

---

## Architectural Impact Statement (P8)

- **P1 (Compiler-first):** All behavior originates in `edgeworld.sim`. No hand-written engine rules; no hand-written WGSL in the harness. ✅
- **P2 (Schema-hash):** No new engine SoA columns. Per-agent semantic state rides on existing repurposed columns (hunger→`hunger`, food quantity→`mana`). No `.schema_hash` bump. ✅
- **P3 / P5 / P11 (Determinism & parity):** Same seed → same saga. All randomness via the engine's keyed RNG surface (`rng.action()`, as in `detective_investigation`). Reductions (hunger/regrowth folds) are commutative-add on per-agent cells — no cross-agent fan-in in Phase 0, so no sort-then-fold needed yet. ✅
- **P7 (Replayability):** All gameplay events (`Ate`, `Died`) flagged `@replayable`. ✅
- **Risk register:** (a) the death channel — whether a rule can clear `alive` directly vs. routing through hp-damage — is resolved empirically in Task 3. (b) `FoodNode : Agent` (not `: Item`) is a deliberate choice to dodge the documented Item-SoA cross-entity-read gap (`foraging_colony.sim` lines 59-67).

---

## Reference fixtures (read these for exact DSL syntax)

- `assets/sim/detective_investigation.sim` — spatial neighbor loop reading a neighbor Agent's SoA column (`agents.cooldown_next_ready_tick(candidate)`), `@spatial spatial_query`, RNG gating, multi-verb argmax with `when`/`score`, `apply_ability`, `@phase(post)` chronicle consumers. **Primary template.**
- `assets/sim/village_economy.sim` — confirmed working setters: `agents.set_hunger`, `agents.set_mana`, `agents.set_hp`, `agents.set_shield_hp`; verb→`apply_ability`→event→`@phase(post)` fold pattern.
- `assets/sim/foraging_colony.sim` — `physics ... @phase(per_agent)` integrator doing `agents.set_pos(self, self.pos + self.vel * k)`; config blocks; entity decls.
- `crates/sims/tests/detective_investigation_pin.rs` — the exact runtime API: `GeneratedRuntime::try_new(seed, n) -> Option`, `state.step()`, `state.gpu.queue.write_buffer(&state.agent_<field>_buf, 0, bytemuck::cast_slice(&v))`, staging-buffer readback helpers (`readback_u32`, `read_positions`).
- `crates/sims/tests/assassination_visualize.rs` — precedent for a non-pinned render/visualization test driven off `GeneratedRuntime`.

---

## File structure

| File | Responsibility | Create/Modify |
|---|---|---|
| `assets/sim/edgeworld.sim` | The entire sim: entities, events, config, verbs, physics folds | **Create** |
| `crates/sims/build.rs` | Add `"edgeworld"` to the fixture allow-list (one line) | **Modify** (~line in the `matches!` block) |
| `crates/sims/Cargo.toml` | Add `image` to `[dev-dependencies]` | **Modify** |
| `crates/sims/tests/edgeworld_pin.rs` | Smoke/behavioral pin: seed → step → assert dynamics (alive counts, hunger, food) | **Create** |
| `crates/sims/tests/edgeworld_render.rs` | Render test: seed → step → dump PNG frames + population trace to `target/edgeworld_frames/` | **Create** |
| `crates/sims/tests/edgeworld_common/mod.rs` | Shared seed + readback helpers used by both tests (DRY) | **Create** |

**Entity declaration order matters:** discriminants are assigned by alphabetical decl order (`EntityRef.0` in `dsl_ast/src/resolve.rs`). `FoodNode` < `Survivor` alphabetically → **`FoodNode = 0`, `Survivor = 1`**. Both tests must mirror these constants.

**Column repurpose table (Phase 0):**
| Semantic | SoA column | Type | Notes |
|---|---|---|---|
| Survivor hunger | `hunger` | f32 | rises per tick; high = starving |
| FoodNode quantity | `mana` | f32 | depletes on eat, regrows per tick |
| Liveness | `alive` | u32 | engine bitmap; cleared on starvation |

---

## Task 1: Scaffold a compiling minimal `edgeworld.sim`

Get the smallest possible fixture compiling through the mega-crate so `sims::edgeworld::GeneratedRuntime` exists. No behavior yet beyond entities + a no-op tick.

**Files:**
- Create: `assets/sim/edgeworld.sim`
- Modify: `crates/sims/build.rs`
- Test: `crates/sims/tests/edgeworld_pin.rs`

- [ ] **Step 1: Write the minimal sim**

Create `assets/sim/edgeworld.sim`:

```
// edgeworld — zero-player survival world sim. Phase 0: a band of
// Survivors forage FoodNodes against rising hunger; food regrows;
// starvation kills. FoodNode is declared `: Agent` (not `: Item`)
// so survivors can read/deplete its quantity via the spatial-
// neighbour SoA-read pattern proven in detective_investigation.sim
// (the Item-SoA cross-entity field read is still a gap — see
// foraging_colony.sim:59-67).
//
// Entity decl order is alphabetical → FoodNode = 0, Survivor = 1.

event Tick { }

entity FoodNode : Agent {
  pos: vec3,
  vel: vec3,
}

entity Survivor : Agent {
  pos: vec3,
  vel: vec3,
}

config edgeworld {
  // placeholder so the config block exists; real tunables land in
  // later tasks.
  unused: u32 = 0,
}
```

- [ ] **Step 2: Add `edgeworld` to the mega-crate allow-list**

In `crates/sims/build.rs`, inside the `matches!(stem.as_str(), ...)` block, add a new arm next to the other fixtures (alphabetical neighbours like `"ecosystem_cascade"` / `"detective_investigation"`):

```rust
                | "edgeworld"
```

- [ ] **Step 3: Write a smoke test that the runtime constructs**

Create `crates/sims/tests/edgeworld_pin.rs`:

```rust
//! edgeworld Phase 0 behavioral pin. Phase 0 = hunger + food +
//! forage/eat/starve/regrow. This file grows task-by-task; Task 1
//! only asserts the fixture compiles and the runtime constructs.

use sims::edgeworld::GeneratedRuntime;

const SEED: u64 = 0xED6E_W0RLD_u64 as u64; // replaced with a real literal below
const N_TOTAL: u32 = 4;

#[test]
fn edgeworld_runtime_constructs() {
    let state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skipping: no wgpu adapter on host.");
            return;
        }
    };
    // Constructing + dropping the runtime is the Task 1 assertion:
    // the fixture compiled and the GPU pipeline built.
    drop(state);
}
```

Fix the `SEED` constant to a plain literal: `const SEED: u64 = 0xED6E_0001;`.

- [ ] **Step 4: Compile the fixture and run the smoke test**

Run: `cargo test -p sims --test edgeworld_pin --release -- --nocapture`
Expected: the build.rs compiles `edgeworld.sim` (watch for DSL parse/lower errors in the build output — fix them in `edgeworld.sim` until it compiles), then `edgeworld_runtime_constructs` PASSES (or prints the no-adapter skip line and returns).

If the DSL fails to compile: read the compiler error, compare against the reference fixtures' exact syntax, fix `edgeworld.sim`, re-run. This is the expected authoring loop for DSL.

- [ ] **Step 5: Commit**

```bash
git add assets/sim/edgeworld.sim crates/sims/build.rs crates/sims/tests/edgeworld_pin.rs
git commit -m "feat(edgeworld): scaffold compiling Phase 0 fixture (entities + tick)"
```

---

## Task 2: Hunger rises every tick

Add the first need. A per-Survivor physics fold increments `hunger` each tick.

**Files:**
- Modify: `assets/sim/edgeworld.sim`
- Test: `crates/sims/tests/edgeworld_pin.rs`

- [ ] **Step 1: Add hunger config + the hunger-rise fold**

In `assets/sim/edgeworld.sim`, replace the `config edgeworld` block and add a physics rule:

```
config edgeworld {
  // Hunger units added per tick per survivor. At 0.05/tick a
  // survivor that never eats crosses the starvation threshold
  // (1.0) in 20 ticks.
  hunger_rate:     f32 = 0.05,
  // Starvation threshold — hunger at or above this kills.
  hunger_max:      f32 = 1.0,
}

// Per-Survivor hunger accrual. creature_type == 1 is Survivor
// (alphabetical decl order: FoodNode=0, Survivor=1).
physics Hunger @phase(per_agent) {
  on Tick {} where (self.alive && self.creature_type == 1) {
    let prev = agents.hunger(self);
    agents.set_hunger(self, prev + config.edgeworld.hunger_rate);
  }
}
```

- [ ] **Step 2: Add a hunger-rise assertion to the pin**

Append to `crates/sims/tests/edgeworld_pin.rs`. First add a shared readback helper inline (it moves to `edgeworld_common` in Task 5):

```rust
fn read_hunger(state: &mut GeneratedRuntime, n: usize) -> Vec<f32> {
    let bytes = (n as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edgeworld::hunger_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("edgeworld::hunger_readback") });
    enc.copy_buffer_to_buffer(&state.agent_hunger_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = { let v = slice.get_mapped_range(); bytemuck::cast_slice::<u8, f32>(&v)[..n].to_vec() };
    staging.unmap();
    out
}

#[test]
fn edgeworld_hunger_rises() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => { eprintln!("[edgeworld] skipping: no wgpu adapter."); return; }
    };
    // Seed: all 4 slots are Survivors, alive, hunger 0.
    let n = N_TOTAL as usize;
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&vec![0.0f32; n]));

    for _ in 0..10 { state.step(); }
    let hunger = read_hunger(&mut state, n);
    println!("[edgeworld] hunger after 10 ticks: {hunger:?}");
    assert!(hunger[0] > 0.4 && hunger[0] < 0.6,
        "hunger should be ~0.5 after 10 ticks at 0.05/tick, got {}", hunger[0]);
}
```

Add `use` lines at the top if missing: `use wgpu;` is transitive via the crate; `bytemuck` likewise. Mirror the imports the detective pin uses.

- [ ] **Step 3: Run the test to verify it fails (rule not yet wired) then passes**

Run: `cargo test -p sims --test edgeworld_pin edgeworld_hunger_rises --release -- --nocapture`
Expected: PASS with hunger ≈ 0.5. If hunger stays 0.0, the `Hunger` fold didn't fire — check the `creature_type == 1` guard and that the seed wrote alive=1.

- [ ] **Step 4: Commit**

```bash
git add assets/sim/edgeworld.sim crates/sims/tests/edgeworld_pin.rs
git commit -m "feat(edgeworld): hunger rises per tick"
```

---

## Task 3: Starvation kills

When a survivor's hunger reaches `hunger_max`, it dies. **This task resolves the death-channel risk**: try clearing `alive` directly from a rule first; if the compiler/runtime doesn't support `agents.set_alive`, fall back to the proven hp-damage death path.

**Files:**
- Modify: `assets/sim/edgeworld.sim`
- Test: `crates/sims/tests/edgeworld_pin.rs`

- [ ] **Step 1: Add the starvation fold (primary: direct alive-clear)**

Add to `assets/sim/edgeworld.sim`:

```
// Starvation. When hunger crosses the max, the survivor dies.
// PRIMARY shape: clear the alive bitmap directly.
physics Starvation @phase(per_agent) {
  on Tick {} where (self.alive
                    && self.creature_type == 1
                    && agents.hunger(self) >= config.edgeworld.hunger_max) {
    agents.set_alive(self, 0);
  }
}
```

- [ ] **Step 2: Compile. If `agents.set_alive` is unsupported, use the fallback**

Run: `cargo build -p sims 2>&1 | tail -30`
Expected: either compiles, OR the lowerer rejects `set_alive`. If rejected, replace the `Starvation` rule body with the hp-damage path (route death through the engine's existing hp→0 death, the channel `detective_investigation` relies on). Seed survivors with `hp = 1.0`, and on starvation deal lethal self-damage via an ability dispatch. Concretely, replace the rule with a verb that self-targets:

```
// FALLBACK death channel: a self-damage verb that fires when
// starving. apply_ability 1 = a Starve ability whose program is
// `damage 9999 by self target self`. Engine hp->0 clears alive.
verb Starve(self, target: Agent) =
  action StarveAction
  when (self.alive
        && self.creature_type == 1
        && target == self
        && agents.hunger(self) >= config.edgeworld.hunger_max)
  apply_ability 1 by self target self
  score 100000.0
```

If using the fallback, an `.ability` file (`assets/ability_test/edgeworld/Starve.ability`, id 1) is needed; model it on `assets/ability_test/detective_investigation/Accuse.ability` with a large `damage` effect. Document which path was taken in a comment at the top of the rule.

- [ ] **Step 3: Write the starvation test**

Add to `crates/sims/tests/edgeworld_pin.rs` a helper + test:

```rust
fn read_alive(state: &mut GeneratedRuntime, n: usize) -> Vec<u32> {
    let bytes = (n as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edgeworld::alive_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("edgeworld::alive_readback") });
    enc.copy_buffer_to_buffer(&state.agent_alive_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = { let v = slice.get_mapped_range(); bytemuck::cast_slice::<u8, u32>(&v)[..n].to_vec() };
    staging.unmap();
    out
}

#[test]
fn edgeworld_starvation_kills() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => { eprintln!("[edgeworld] skipping: no wgpu adapter."); return; }
    };
    let n = N_TOTAL as usize;
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&vec![0.0f32; n]));
    state.gpu.queue.write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&vec![1.0f32; n])); // for fallback path

    // No food in this scenario → everyone starves. hunger_max=1.0 at
    // 0.05/tick = 20 ticks to threshold; run 30 to be safe.
    for _ in 0..30 { state.step(); }
    let alive = read_alive(&mut state, n);
    let n_alive: u32 = alive.iter().sum();
    println!("[edgeworld] survivors alive after 30 starving ticks: {n_alive}");
    assert_eq!(n_alive, 0, "all survivors should have starved with no food");
}
```

- [ ] **Step 4: Run and verify**

Run: `cargo test -p sims --test edgeworld_pin edgeworld_starvation_kills --release -- --nocapture`
Expected: PASS — 0 alive after 30 foodless ticks.

- [ ] **Step 5: Commit**

```bash
git add assets/sim/edgeworld.sim crates/sims/tests/edgeworld_pin.rs
# include the .ability file + assets if the fallback path was used
git commit -m "feat(edgeworld): starvation kills survivors at hunger_max"
```

---

## Task 4: Food nodes — eat (deplete) and regrow

Survivors adjacent to a non-empty FoodNode eat: their hunger drops, the node's quantity (`mana`) drops. Nodes regrow slowly. Eating uses the spatial-neighbour SoA-read/write pattern from `detective_investigation`.

**Files:**
- Modify: `assets/sim/edgeworld.sim`
- Test: `crates/sims/tests/edgeworld_pin.rs`

- [ ] **Step 1: Add the spatial query, eat config, eat rule, regrow rule**

Add to `assets/sim/edgeworld.sim`:

```
config edgeworld {
  hunger_rate:     f32 = 0.05,
  hunger_max:      f32 = 1.0,
  // Hunger removed per eat tick.
  eat_amount:      f32 = 0.20,
  // Food quantity consumed per eat tick (= hunger removed, 1:1).
  eat_cost:        f32 = 0.20,
  // FoodNode regrowth per tick.
  regrow_rate:     f32 = 0.02,
  // FoodNode quantity ceiling.
  food_max:        f32 = 5.0,
  // Eat / perception radius.
  eat_radius:      f32 = 1.5 @runtime,
}

@spatial(radius = 1.5, kind = [Agent])
spatial_query nearby_food(self: AgentId, candidate: AgentId) =
  candidate != self

// Eat: a Survivor with any hunger, standing within eat_radius of a
// FoodNode that still has quantity, reduces its own hunger and the
// node's quantity. Reads/writes the neighbour FoodNode's `mana`
// column directly (detective_investigation pattern).
physics Eat @phase(per_agent) {
  on Tick {} where (self.alive && self.creature_type == 1 && agents.hunger(self) > 0.0) {
    for candidate in spatial.nearby_food(self) {
      if (candidate.alive
          && candidate.creature_type == 0
          && agents.mana(candidate) >= config.edgeworld.eat_cost) {
        // reduce own hunger (saturating at 0)
        let h = agents.hunger(self);
        let new_h = if (h > config.edgeworld.eat_amount) { h - config.edgeworld.eat_amount } else { 0.0 };
        agents.set_hunger(self, new_h);
        // deplete the food node
        let q = agents.mana(candidate);
        agents.set_mana(candidate, q - config.edgeworld.eat_cost);
      }
    }
  }
}

// Regrowth: every FoodNode regenerates quantity each tick up to the
// ceiling.
physics Regrow @phase(per_agent) {
  on Tick {} where (self.alive && self.creature_type == 0) {
    let q = agents.mana(self);
    let grown = q + config.edgeworld.regrow_rate;
    let capped = if (grown > config.edgeworld.food_max) { config.edgeworld.food_max } else { grown };
    agents.set_mana(self, capped);
  }
}
```

NOTE: confirm the in-loop multi-write (writing both `self`'s hunger and `candidate`'s mana inside a `for ... in spatial` body) lowers cleanly. `detective_investigation` reads a neighbour column in a spatial loop but writes only via `emit`. If the direct neighbour-write is rejected, route the depletion through an event + `@phase(post)` fold instead (emit `Ate { eater: self, node: candidate }`, then a fold that does the two writes). Keep the event-routed version as the fallback and note which was used.

- [ ] **Step 2: Write the eat/regrow test**

Add a `read_mana` helper (copy `read_hunger`, swap buffer to `agent_mana_buf`) and:

```rust
#[test]
fn edgeworld_eating_feeds_and_depletes() {
    let mut state = match GeneratedRuntime::try_new(SEED, 2) {
        Some(s) => s, None => { eprintln!("[edgeworld] skip: no adapter."); return; }
    };
    // slot 0 = FoodNode (type 0), slot 1 = Survivor (type 1), co-located.
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&[0u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0,
        bytemuck::cast_slice(&[[0.0f32,0.0,0.0,0.0],[0.5,0.0,0.0,0.0]]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.6f32]));
    state.gpu.queue.write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&[5.0f32, 0.0f32])); // node full, survivor n/a

    for _ in 0..3 { state.step(); }
    let hunger = read_hunger(&mut state, 2);
    let mana = read_mana(&mut state, 2);
    println!("[edgeworld] survivor hunger={} node quantity={}", hunger[1], mana[0]);
    // Survivor ate: hunger fell from 0.6 (minus eat, plus a little rise).
    assert!(hunger[1] < 0.6, "survivor should have eaten and lowered hunger");
    // Node depleted from 5.0 (minus eats, plus regrow) — net below 5.
    assert!(mana[0] < 5.0, "food node should have been depleted by eating");
}
```

- [ ] **Step 3: Run and verify**

Run: `cargo test -p sims --test edgeworld_pin edgeworld_eating_feeds_and_depletes --release -- --nocapture`
Expected: PASS — survivor hunger drops, node quantity drops.

- [ ] **Step 4: Commit**

```bash
git add assets/sim/edgeworld.sim crates/sims/tests/edgeworld_pin.rs
git commit -m "feat(edgeworld): eat (deplete food) + regrow"
```

---

## Task 5: Movement — SeekFood and Wander verbs

Hungry survivors move toward the nearest FoodNode; otherwise they drift. This makes the colony actually traverse the world instead of only eating when spawned on top of food.

**Files:**
- Modify: `assets/sim/edgeworld.sim`
- Test: `crates/sims/tests/edgeworld_pin.rs`
- Create: `crates/sims/tests/edgeworld_common/mod.rs`

- [ ] **Step 1: Add SeekFood movement + Wander**

Movement-toward-nearest in pure DSL needs a per-agent integrator that nudges `pos` toward the closest in-range FoodNode. Add to `assets/sim/edgeworld.sim`:

```
config edgeworld {
  // ... existing fields ...
  move_speed:      f32 = 0.3,
  wander_scale:    f32 = 0.05,
  perceive_radius: f32 = 12.0 @runtime,
}

@spatial(radius = 12.0, kind = [Agent])
spatial_query food_in_sight(self: AgentId, candidate: AgentId) =
  candidate != self

// SeekFood: hungry survivor steps toward the nearest visible
// FoodNode with quantity. Implemented as a per-agent integrator
// (not a verb) because it mutates pos continuously; the verb/argmax
// layer is reserved for discrete actions (Eat is handled in Task 4's
// fold). Picks the first in-range node; refine to nearest if the
// argmin pattern is available.
physics SeekFood @phase(per_agent) {
  on Tick {} where (self.alive && self.creature_type == 1 && agents.hunger(self) > 0.2) {
    for candidate in spatial.food_in_sight(self) {
      if (candidate.alive && candidate.creature_type == 0
          && agents.mana(candidate) >= config.edgeworld.eat_cost) {
        let dir = candidate.pos - self.pos;
        let step = dir * config.edgeworld.move_speed;
        agents.set_pos(self, self.pos + step);
      }
    }
  }
}
```

NOTE: `dir * scalar` and `candidate.pos` (reading a neighbour's vec3 in a spatial loop) must lower. If `candidate.pos` is unavailable, read via `agents.pos(candidate)` (the accessor form). If normalisation is needed, defer it — an unnormalised nudge still converges and keeps Phase 0 simple. Iterate against the compiler.

- [ ] **Step 2: Extract shared seed/readback helpers to `edgeworld_common`**

Create `crates/sims/tests/edgeworld_common/mod.rs` holding the constants (`CT_FOOD=0`, `CT_SURVIVOR=1`) and the `read_hunger` / `read_mana` / `read_alive` / `read_positions` helpers (move them out of `edgeworld_pin.rs`, import via `mod edgeworld_common;`). This is consumed by both the pin and the render test (DRY).

- [ ] **Step 3: Write a convergence test**

A hungry survivor placed away from a single food node should end the run closer to it:

```rust
#[test]
fn edgeworld_seekfood_moves_toward_food() {
    let mut state = match GeneratedRuntime::try_new(SEED, 2) {
        Some(s) => s, None => { eprintln!("[edgeworld] skip: no adapter."); return; }
    };
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&[0u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0,
        bytemuck::cast_slice(&[[0.0f32,0.0,0.0,0.0],[8.0,0.0,0.0,0.0]])); // food at origin, survivor 8 away
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.5f32]));
    state.gpu.queue.write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&[5.0f32, 0.0f32]));

    let start = read_positions(&mut state, 2)[1][0];
    for _ in 0..10 { state.step(); }
    let end = read_positions(&mut state, 2)[1][0];
    println!("[edgeworld] survivor x: {start} -> {end}");
    assert!(end < start - 1.0, "hungry survivor should move toward food (x decreasing)");
}
```

- [ ] **Step 4: Run and verify**

Run: `cargo test -p sims --test edgeworld_pin edgeworld_seekfood_moves_toward_food --release -- --nocapture`
Expected: PASS — survivor's x moves from 8 toward 0.

- [ ] **Step 5: Commit**

```bash
git add assets/sim/edgeworld.sim crates/sims/tests/edgeworld_pin.rs crates/sims/tests/edgeworld_common/mod.rs
git commit -m "feat(edgeworld): SeekFood movement toward nearest food"
```

---

## Task 6: Render — PNG frame dump + population trace

Make it visible. A render test seeds a real scenario (12 survivors + scattered food), steps it, and every N ticks writes a top-down PNG (survivors as dots, food green by quantity, dead greyed) plus a population-over-time line. These PNGs are inspected directly via the Read tool.

**Files:**
- Modify: `crates/sims/Cargo.toml` (add `image` dev-dependency)
- Create: `crates/sims/tests/edgeworld_render.rs`

- [ ] **Step 1: Add the `image` dev-dependency**

In `crates/sims/Cargo.toml` under `[dev-dependencies]`:

```toml
image = { version = "0.25", default-features = false, features = ["png"] }
```

- [ ] **Step 2: Write the render test**

Create `crates/sims/tests/edgeworld_render.rs`. It reuses `edgeworld_common`. Core shape:

```rust
//! edgeworld render — not a pin. Seeds a 12-survivor + scattered-food
//! world, steps it, and dumps top-down PNG frames + a population
//! trace to target/edgeworld_frames/. Inspect the PNGs to verify the
//! boom/bust dynamics visually.
//!   cargo test -p sims --test edgeworld_render --release -- --nocapture

mod edgeworld_common;
use edgeworld_common::*;
use sims::edgeworld::GeneratedRuntime;
use std::path::Path;

const SEED: u64 = 0xED6E_0001;
const N_SURVIVORS: u32 = 12;
const N_FOOD: u32 = 8;
const N_TOTAL: u32 = N_SURVIVORS + N_FOOD;
const TICKS: u32 = 600;
const FRAME_EVERY: u32 = 30;
const IMG: u32 = 256;        // px
const WORLD_HALF: f32 = 20.0; // world spans [-20, 20]

fn world_to_px(world: f32) -> u32 {
    let t = ((world + WORLD_HALF) / (2.0 * WORLD_HALF)).clamp(0.0, 1.0);
    (t * (IMG as f32 - 1.0)) as u32
}

#[test]
fn edgeworld_render_saga() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s, None => { eprintln!("[edgeworld] skip: no adapter."); return; }
    };
    seed_world(&mut state, N_SURVIVORS, N_FOOD); // helper in edgeworld_common

    let dir = Path::new("target/edgeworld_frames");
    std::fs::create_dir_all(dir).unwrap();
    let mut pop_trace: Vec<u32> = Vec::new();

    for tick in 0..TICKS {
        if tick % FRAME_EVERY == 0 {
            let pos = read_positions(&mut state, N_TOTAL as usize);
            let alive = read_alive(&mut state, N_TOTAL as usize);
            let types = read_creature_types(&mut state, N_TOTAL as usize);
            let mana = read_mana(&mut state, N_TOTAL as usize);
            let mut img = image::RgbImage::from_pixel(IMG, IMG, image::Rgb([16, 16, 20]));
            for i in 0..N_TOTAL as usize {
                if alive[i] == 0 { continue; }
                let (x, y) = (world_to_px(pos[i][0]), world_to_px(pos[i][1]));
                let color = if types[i] == CT_FOOD {
                    let g = (40.0 + 200.0 * (mana[i] / 5.0)).min(255.0) as u8; // green by quantity
                    image::Rgb([10, g, 10])
                } else {
                    image::Rgb([220, 200, 80]) // survivor = amber dot
                };
                draw_blob(&mut img, x, y, color); // 3x3 blob helper
            }
            img.save(dir.join(format!("frame_{tick:04}.png"))).unwrap();
            let alive_survivors: u32 = (0..N_TOTAL as usize)
                .filter(|&i| alive[i] == 1 && types[i] == CT_SURVIVOR).count() as u32;
            pop_trace.push(alive_survivors);
            println!("[edgeworld] tick {tick:>4}: survivors alive = {alive_survivors}");
        }
        state.step();
    }
    // ASCII population sparkline so the trace is legible in test output too.
    println!("[edgeworld] population trace: {}", sparkline(&pop_trace));
    println!("[edgeworld] frames written to {}", dir.display());
}
```

Add the small helpers (`draw_blob`, `sparkline`, `read_creature_types`, `seed_world`) to `edgeworld_common`. `seed_world` places food nodes on a grid/circle with full `mana`, survivors clustered near center with hunger 0, alive 1, hp 1.

- [ ] **Step 3: Run the render and inspect frames**

Run: `cargo test -p sims --test edgeworld_render --release -- --nocapture`
Expected: prints a per-frame survivor count + a sparkline, and writes `target/edgeworld_frames/frame_*.png`.

Then **inspect visually**: Read several frames (`target/edgeworld_frames/frame_0000.png`, `frame_0300.png`, `frame_0570.png`) and confirm survivors cluster on food, food dims as it's eaten and brightens as it regrows, and the population curve shows movement (not flat-line, not instant wipe). Iterate on config tunables (`hunger_rate`, `regrow_rate`, `food_max`, counts) until the dynamics read clearly. **Do not wait for human review on each tuning pass — self-verify from the frames.**

- [ ] **Step 4: Commit**

```bash
git add crates/sims/Cargo.toml crates/sims/tests/edgeworld_render.rs crates/sims/tests/edgeworld_common/mod.rs
git commit -m "feat(edgeworld): PNG frame render + population trace"
```

---

## Task 7: Tune the seed for a legible boom/bust (the Phase 0 success criterion)

Find seed + config values that **reliably render a legible boom/bust**: population holds or grows, strips local food, suffers a visible crash (several deaths in a window), and leaves a surviving remnant — not a flat line, not instant total extinction.

**Files:**
- Modify: `assets/sim/edgeworld.sim` (config tunables)
- Modify: `crates/sims/tests/edgeworld_render.rs` (seed/scenario)
- Test: `crates/sims/tests/edgeworld_pin.rs` (add a boom/bust shape assertion)

- [ ] **Step 1: Add a boom/bust shape pin**

A long-run pin asserting the dynamics are real: starting population survives the early game, suffers at least one crash, and leaves a remnant.

```rust
#[test]
fn edgeworld_boom_then_bust_then_remnant() {
    let mut state = match GeneratedRuntime::try_new(0xED6E_0001, 20) {
        Some(s) => s, None => { eprintln!("[edgeworld] skip: no adapter."); return; }
    };
    seed_world(&mut state, 12, 8);
    let mut min_alive = u32::MAX;
    let mut max_alive = 0u32;
    let mut samples = Vec::new();
    for tick in 0..600 {
        if tick % 20 == 0 {
            let alive = read_alive(&mut state, 20);
            let types = read_creature_types(&mut state, 20);
            let a: u32 = (0..20).filter(|&i| alive[i]==1 && types[i]==CT_SURVIVOR).count() as u32;
            min_alive = min_alive.min(a); max_alive = max_alive.max(a); samples.push(a);
        }
        state.step();
    }
    let final_alive = *samples.last().unwrap();
    println!("[edgeworld] max={max_alive} min={min_alive} final={final_alive} trace={samples:?}");
    assert!(max_alive >= 6, "expected a sustained early population (boom), got max {max_alive}");
    assert!(min_alive < max_alive, "expected a crash (min < max), got flat {min_alive}");
    assert!(final_alive >= 1, "expected a surviving remnant, got extinction");
}
```

- [ ] **Step 2: Tune until the pin passes and the frames read clearly**

Run both:
```
cargo test -p sims --test edgeworld_pin edgeworld_boom_then_bust_then_remnant --release -- --nocapture
cargo test -p sims --test edgeworld_render --release -- --nocapture
```
Adjust `hunger_rate` (pressure), `regrow_rate` / `food_max` (carrying capacity), `N_FOOD`, and survivor/food spawn spread until the pin is green AND the rendered frames + sparkline show a clear boom→crash→remnant. Self-verify from PNGs each pass.

- [ ] **Step 3: Commit**

```bash
git add assets/sim/edgeworld.sim crates/sims/tests/edgeworld_pin.rs crates/sims/tests/edgeworld_render.rs
git commit -m "feat(edgeworld): tuned seed yields legible boom/bust; Phase 0 complete"
```

- [ ] **Step 4: Human sign-off checkpoint**

Surface a contact sheet of frames (early / mid / late) + the population sparkline to the user for the Phase 0 "is this a saga worth watching" sign-off. This is the one human gate in Phase 0 (per the spec's verification posture).

---

## Self-review notes

- **Spec coverage:** Phase 0 of the design spec (hunger + food + forage/eat/starve/regrow + Tier-1 PNG render + population trace + boom/bust success criterion) is covered by Tasks 1–7. Phases 1+ (predators, beliefs, reproduction, society, world, viewer) are explicitly out of scope for this plan.
- **Empirical gates:** DSL authoring is iterative against the compiler by nature. Three steps carry explicit fallbacks where a primitive may not lower: death channel (Task 3: `set_alive` vs hp-damage), in-spatial-loop neighbour write (Task 4: direct write vs event+fold), neighbour vec3 read in movement (Task 5: `candidate.pos` vs `agents.pos(candidate)`). Each names the proven fallback so a blocked primitive doesn't stall the plan.
- **Type consistency:** `CT_FOOD=0` / `CT_SURVIVOR=1` (alphabetical decl order) used identically across `edgeworld.sim`, `edgeworld_common`, and both tests. Hunger→`hunger` column, food quantity→`mana` column used consistently.
- **GPU dependency:** every test guards on `try_new(...) -> None` (no adapter → skip), matching the repo convention. If the host has no GPU adapter, frames can't render; flag that to the user and arrange a host with an adapter for visual verification.
```
