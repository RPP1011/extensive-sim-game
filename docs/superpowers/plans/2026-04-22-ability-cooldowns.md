# Ability Cooldowns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-(agent, ability-slot) local cooldowns alongside the existing per-agent global cooldown (GCD). An ability fires iff both gates pass. Closes a latent correctness bug where all abilities on one agent share the same cooldown cursor.

**Architecture:** Keep `hot_cooldown_next_ready_tick: u32` (existing hot SoA field, semantically narrowed to "global cooldown"). Add `ability_cooldowns: Vec<[u32; MAX_ABILITIES]>` as a new side buffer on `SimState` (per-agent local cooldowns). Helper `can_cast_ability(agent, slot, now)` gates both. Cast-handler writes both post-cast. Configurable via new `combat.global_cooldown_ticks: u32 = 5` DSL config field.

**Tech Stack:** Rust, `engine` crate state, `dsl_compiler` for config field + new DSL primitive, existing CascadeRegistry callers.

**Spec reference:** `docs/superpowers/specs/2026-04-22-ability-cooldowns-design.md`

---

## File structure

### Modified

- `crates/engine/src/state/mod.rs` — add `ability_cooldowns: Vec<[u32; MAX_ABILITIES]>` SoA field alongside the hot-cooldown field.
- `crates/engine/src/state/mod.rs` or equivalent — `can_cast_ability(agent, slot, now)` helper method on `SimState`.
- `crates/engine/src/schema_hash.rs` — add a string line for the new field so the baseline captures the layout bump.
- `crates/engine/.schema_hash` — updated baseline hash.
- `assets/sim/config.sim` — add `global_cooldown_ticks: u32 = 5` to the `combat` block.
- `crates/engine/src/cascade/cast_handler.rs` (or wherever `CastHandler` lives) — post-cast, write both global and local cooldowns.
- Consumers of `hot_cooldown_next_ready_tick`: `crates/engine/src/policy/utility.rs`, scoring DSL (`assets/sim/scoring.sim`), `crates/dsl_compiler/src/` (DSL primitive lowering).
- `crates/dsl_compiler/src/` — new `ability::on_cooldown(ability_ref)` primitive.

### New tests

- `crates/engine/tests/cooldowns_global_gate.rs` — global cooldown gates sibling abilities.
- `crates/engine/tests/cooldowns_local_gate.rs` — local cooldown gates repeat casts of the same ability after GCD clears.
- `crates/engine/tests/cooldowns_independent_across_slots.rs` — A and B on different locals don't interfere beyond GCD.
- `crates/engine/tests/cooldowns_schema_hash.rs` — regression assertion after the field addition.

### Untouched (intentional)

- GPU kernels (`crates/engine_gpu/src/*`) — cooldown state stays CPU-only for this subsystem. Subsystem (3) GPU ability evaluation will add GPU-side exposure.
- `assets/sim/events.sim` — no new events.

---

# Task 1: Add `ability_cooldowns` field to `SimState`

**Files:**
- Modify: `crates/engine/src/state/mod.rs`

- [ ] **Step 1: Locate the SoA cold/hot split in `SimState`**

```
grep -n "hot_cooldown_next_ready_tick\|pub struct SimState\|pub.*Vec<.*>.*cooldown" crates/engine/src/state/mod.rs | head
```

- [ ] **Step 2: Add the new SoA field**

Add `ability_cooldowns: Vec<[u32; MAX_ABILITIES]>` as a COLD field (grouped with `status_effects`, `memberships`, etc. — not hot). `MAX_ABILITIES` comes from `crates/engine/src/ability/program.rs` or the ability IR; grep to confirm name (`MAX_ABILITIES` is a reasonable guess; might be `MAX_ABILITY_SLOTS` or similar).

```rust
/// Per-(agent, ability-slot) local cooldown cursor. `0` = ready.
/// Non-zero = `state.tick` when this specific ability next becomes
/// ready. An ability fires iff
///   `state.tick >= hot_cooldown_next_ready_tick[agent]` (global GCD)
/// AND
///   `state.tick >= ability_cooldowns[agent][slot]` (local).
///
/// Added 2026-04-22 to fix a shared-cursor bug where all abilities
/// on one agent were gated by the single global cursor.
pub ability_cooldowns: Vec<[u32; MAX_ABILITIES]>,
```

- [ ] **Step 3: Initialise in `SimState::new`**

Find `SimState::new(agent_cap, seed)` (around line 100-200 of mod.rs). Wherever the cold Vec fields are initialised, add:

```rust
ability_cooldowns: vec![[0u32; MAX_ABILITIES]; agent_cap as usize],
```

- [ ] **Step 4: Resize in `grow_agent_cap` / `spawn_agent`**

Grep for `agents.push\|resize.*agent_cap\|hot_.*\.push` in `state/mod.rs` to find where per-agent Vec fields are grown. Add `self.ability_cooldowns.push([0u32; MAX_ABILITIES]);` alongside the others.

- [ ] **Step 5: Build**

Run: `cargo build -p engine`
Expected: clean. If `MAX_ABILITIES` isn't in scope, add the correct import at top of `state/mod.rs`.

- [ ] **Step 6: Commit**

```bash
git add crates/engine/src/state/mod.rs
git commit -m "feat(engine): add per-(agent, ability-slot) ability_cooldowns SoA field"
```

---

# Task 2: Add `global_cooldown_ticks` to the combat DSL config

**Files:**
- Modify: `assets/sim/config.sim`

- [ ] **Step 1: Add field to config**

Edit `assets/sim/config.sim` in the `config combat { ... }` block:

```
// Shared "global cooldown" (GCD) gate preventing ability spam across
// all of an agent's abilities. Applies independently of each
// ability's per-slot local cooldown. 5 ticks at 10 Hz = 0.5s, a
// standard MMO/MOBA default. Set to 0 to disable the global gate.
global_cooldown_ticks: u32 = 5,
```

Place it near the existing `engagement_slow_factor` / `kin_flee_bias` fields for grouping.

- [ ] **Step 2: Regenerate DSL**

Run: `cargo run --bin xtask -- compile-dsl 2>&1 | tail -3`
Expected: clean compile; `config_hash` changes; 4 config block(s) updated.

- [ ] **Step 3: Verify engine_rules pickup**

Run: `grep -n "global_cooldown_ticks" crates/engine_generated/src/config/combat.rs`
Expected: field present.

- [ ] **Step 4: Build to verify schema_hash baseline**

Run: `cargo test -p engine schema_hash 2>&1 | tail -5`
Expected: FAIL with new hash. Update the baseline:

```bash
# copy the new hash from the test output into crates/engine/.schema_hash
# (the test prints: `Current: <hash>`)
```

- [ ] **Step 5: Commit**

```bash
git add assets/sim/config.sim \
        crates/engine_generated/ \
        assets/config/default.toml \
        crates/engine/.schema_hash
git commit -m "feat(config): add combat.global_cooldown_ticks (GCD, default 5 ticks)"
```

---

# Task 3: `can_cast_ability` helper method on `SimState`

**Files:**
- Modify: `crates/engine/src/state/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine/tests/cooldowns_helper.rs`:

```rust
//! Unit test for `SimState::can_cast_ability` — gates cast on both
//! global (GCD) and local (per-ability-slot) cooldowns.

use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn spawn(state: &mut SimState) -> engine::ids::AgentId {
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO,
        hp: 100.0,
        ..Default::default()
    }).unwrap()
}

#[test]
fn can_cast_ability_ready_when_both_cooldowns_cleared() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state);
    // Fresh agent: both cooldowns zero; tick is 0.
    assert!(state.can_cast_ability(a, 0, 0), "fresh agent should be ready");
}

#[test]
fn can_cast_ability_blocked_by_global() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state);
    // Set global cooldown cursor to tick 5; local stays at 0.
    let slot = state.agent_slot(a).unwrap();
    state.hot_cooldown_next_ready_tick[slot] = 5;
    assert!(!state.can_cast_ability(a, 0, 3), "global gate should block at tick 3");
    assert!(state.can_cast_ability(a, 0, 5), "global gate should clear at tick 5");
}

#[test]
fn can_cast_ability_blocked_by_local() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state);
    // Local cooldown for slot 0 at tick 10; global at 0.
    let slot = state.agent_slot(a).unwrap();
    state.ability_cooldowns[slot][0] = 10;
    assert!(!state.can_cast_ability(a, 0, 5), "local gate should block at tick 5");
    assert!(state.can_cast_ability(a, 1, 5), "different slot should be ready");
    assert!(state.can_cast_ability(a, 0, 10), "local gate should clear at tick 10");
}
```

Note: `state.agent_slot(id)` may not be the exact name. Grep `crates/engine/src/state/` for how callers convert `AgentId` to hot-SoA slot index (`slot_for`, `slot`, `agent_index`). Adjust.

- [ ] **Step 2: Run test; confirm it fails**

Run: `cargo test -p engine --test cooldowns_helper`
Expected: FAIL — `can_cast_ability` method doesn't exist.

- [ ] **Step 3: Add the helper method**

In `crates/engine/src/state/mod.rs`, add to `impl SimState`:

```rust
/// True iff the given ability slot is off cooldown (both global
/// GCD and per-slot local cooldown have cleared).
///
/// `slot` bounds: `[0, MAX_ABILITIES)`. Out-of-range slots are
/// treated as "always off cooldown" (returns `true`) — defensive
/// default; callers should guard separately since the ability
/// registry is fixed-size.
pub fn can_cast_ability(&self, agent: crate::ids::AgentId, slot: u8, now: u32) -> bool {
    let Some(agent_slot) = self.agent_slot(agent) else {
        return false; // unknown agent can't cast
    };
    if (slot as usize) >= MAX_ABILITIES {
        return true; // out-of-range slot: defensive default
    }
    let global_ready = self.hot_cooldown_next_ready_tick[agent_slot] <= now;
    let local_ready = self.ability_cooldowns[agent_slot][slot as usize] <= now;
    global_ready && local_ready
}
```

Adjust `agent_slot` to the actual name. Add the needed imports.

- [ ] **Step 4: Run test; confirm passes**

Run: `cargo test -p engine --test cooldowns_helper`
Expected: 3/3 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/state/mod.rs crates/engine/tests/cooldowns_helper.rs
git commit -m "feat(engine): SimState::can_cast_ability helper (global + local gate)"
```

---

# Task 4: Cast-handler writes both cooldowns post-cast

**Files:**
- Modify: `crates/engine/src/cascade/cast_handler.rs` (or wherever `CastHandler` lives; grep `CastHandler\|impl.*Cast` in crates/engine/src/)

- [ ] **Step 1: Locate the post-cast cooldown write**

Currently the handler only writes `hot_cooldown_next_ready_tick` with `now + ability.gate.cooldown_ticks`. Find that write.

Run: `grep -rn "hot_cooldown_next_ready_tick" crates/engine/src/cascade/ | head`

- [ ] **Step 2: Rewrite as both-cooldowns write**

Replace the single-cursor write with:

```rust
let slot = ability_slot as usize;
// Global GCD — short shared gate.
let gcd = state.config.combat.global_cooldown_ticks;
state.hot_cooldown_next_ready_tick[agent_slot] = state.tick + gcd;
// Local cooldown — per-ability refresh.
state.ability_cooldowns[agent_slot][slot] =
    state.tick + ability.gate.cooldown_ticks;
```

`ability_slot` comes from whatever parameter carries the ability-slot-index into `CastHandler`. If that's unavailable at the write site (handler receives `AbilityId` but not the slot index), add a lookup:

```rust
let slot = state.ability_slot_for_id(agent, ability_id).unwrap_or(0);
```

or propagate the slot index from the call site.

- [ ] **Step 3: Run cast handler tests**

Run: `cargo test -p engine --test cast_handler_gold --test cast_handler_damage`
(Or whatever cast-handler tests exist; grep `cast_handler` under `crates/engine/tests/`.)
Expected: still pass. The existing tests shouldn't observe cooldown changes; new cooldown-specific tests arrive in Task 5.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/cascade/cast_handler.rs
git commit -m "feat(engine): CastHandler writes both global + local cooldowns post-cast"
```

---

# Task 5: Integration tests — gating behaviour

**Files:**
- Create: `crates/engine/tests/cooldowns_global_gate.rs`
- Create: `crates/engine/tests/cooldowns_local_gate.rs`
- Create: `crates/engine/tests/cooldowns_independent_across_slots.rs`

- [ ] **Step 1: Write the global-gate test**

`crates/engine/tests/cooldowns_global_gate.rs`:

```rust
//! Cast ability A at tick 0. Assert: at tick 1 (< GCD), ability B on
//! the same agent is gated by global cooldown. At tick GCD, B clears.

// ... setup: spawn agent with two abilities A, B on different slots ...
// ... cast A at tick 0 ...
// ... tick forward 1 tick ...
// assert !state.can_cast_ability(agent, 1, state.tick);  // B blocked
// ... tick forward to GCD ticks ...
// assert state.can_cast_ability(agent, 1, state.tick);   // B ready
```

The test setup needs actual cast wiring. Reuse helpers from `cast_handler_gold.rs` if they exist; otherwise, skip the full cascade and manually write cooldown cursors to simulate "just cast A":

```rust
let agent_slot = state.agent_slot(a).unwrap();
state.hot_cooldown_next_ready_tick[agent_slot] = state.tick + 5; // GCD=5
state.ability_cooldowns[agent_slot][0] = state.tick + 30;        // A's local
// at tick 3: global blocks B (different slot, local for B is 0)
assert!(!state.can_cast_ability(a, 1, 3));
// at tick 5: global clears; B's local (slot 1) is 0 so ready
assert!(state.can_cast_ability(a, 1, 5));
```

- [ ] **Step 2: Write the local-gate test**

`crates/engine/tests/cooldowns_local_gate.rs`:

```rust
// Cast A (cooldown 10). At tick GCD+1, global cleared but local still
// blocks A. B on slot 1 clears because its local is 0.
```

Same pattern — set cursors by hand, assert gate outcomes.

- [ ] **Step 3: Write the independent-slots test**

`crates/engine/tests/cooldowns_independent_across_slots.rs`:

```rust
// Cast A (local cd 10) at tick 0. Wait GCD. Cast B (local cd 3) at
// tick GCD. At tick GCD+1, assert A locally gated, B locally gated.
// At tick GCD+4, assert B cleared, A still locally gated.
```

- [ ] **Step 4: Run all three tests**

Run: `cargo test -p engine --test cooldowns_global_gate --test cooldowns_local_gate --test cooldowns_independent_across_slots`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/tests/cooldowns_*.rs
git commit -m "test(engine): cooldown gating behaviour (global, local, independent slots)"
```

---

# Task 6: Migrate CPU consumers to use `can_cast_ability`

**Files:**
- Modify: `crates/engine/src/policy/utility.rs` (scoring helpers)
- Modify: possibly `crates/engine/src/ability/` files
- Modify: any test that reads `hot_cooldown_next_ready_tick` directly

- [ ] **Step 1: Find direct readers of `hot_cooldown_next_ready_tick`**

Run: `grep -rn "hot_cooldown_next_ready_tick" crates/engine/ | grep -v "= \|\.push\|hot_cooldown_next_ready_tick\[.*\] =" | head`

Filter out the writer (cast handler, spawn init) — what's left are readers.

- [ ] **Step 2: Rewrite each reader to use `can_cast_ability`**

For each callsite:

Before:
```rust
if state.hot_cooldown_next_ready_tick[slot] <= state.tick {
    // ability is ready
}
```

After:
```rust
if state.can_cast_ability(agent, ability_slot, state.tick) {
    // ability is ready (both global + local cleared)
}
```

The `ability_slot: u8` is available at every callsite — readers already know which ability slot they're evaluating (it's part of ability-eval scoring iteration).

- [ ] **Step 3: Build + test**

Run: `cargo test -p engine`
Expected: all existing tests pass. The semantic change (abilities now gated by BOTH cooldowns) may cause some previously-passing scoring tests to show different behaviour — that's correct behaviour; if any tests break in a way that reveals stale assumptions, update their expected values with a comment explaining.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/policy/utility.rs  # and others
git commit -m "refactor(engine): migrate cooldown readers to can_cast_ability helper"
```

---

# Task 7: New DSL primitive `ability::on_cooldown(ability_ref)`

**Files:**
- Modify: `crates/dsl_compiler/src/` — scoring grammar parser + emit paths
- Modify: `assets/sim/scoring.sim` — optional, if existing rules can be simplified with the new primitive

- [ ] **Step 1: Locate scoring-expression grammar**

Grep: `grep -rn "pub enum.*Expr\|is_on_cooldown\|ability::\|AbilityRef" crates/dsl_compiler/src/ | head`

The expression IR / parser likely has `NamespaceCall` or `MethodCall` nodes that handle `view::kin_fear(a, b)` etc. Add an `ability::on_cooldown` lowering there.

- [ ] **Step 2: Add the primitive**

In the expression lowering emit:

CPU-side:
```rust
NamespaceCall { ns: "ability", name: "on_cooldown", args: [AbilityRef(slot)] }
  => emit `!state.can_cast_ability(agent, slot, state.tick)`
```

(The primitive name is inverted from `can_cast_ability` — `on_cooldown` returns `true` when the gate blocks.)

- [ ] **Step 3: Add a DSL-compiler test**

In `crates/dsl_compiler/tests/`, add a test that compiles a minimal scoring rule using `ability::on_cooldown` and asserts the emitted Rust calls `can_cast_ability`.

- [ ] **Step 4: Run dsl_compiler tests**

Run: `cargo test -p dsl_compiler`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/
git commit -m "feat(dsl_compiler): ability::on_cooldown DSL primitive"
```

---

# Task 8: Regression sweep + schema hash fence

- [ ] **Step 1: Run the full engine + dsl_compiler suites**

Run:
```
cargo test -p engine
cargo test -p dsl_compiler
cargo test --release --features gpu -p engine_gpu
```

Expected: all green. engine_gpu shouldn't have changed at all — cooldowns are CPU-only this subsystem.

- [ ] **Step 2: Add the schema-hash regression test**

Create `crates/engine/tests/cooldowns_schema_hash.rs`:

```rust
//! Regression fence — the ability_cooldowns field addition bumped
//! the schema hash. Baseline captured 2026-04-22.

#[test]
fn schema_hash_captures_ability_cooldowns_field() {
    let hash_hex = hex::encode(engine::schema_hash::schema_hash());
    // Read baseline from disk; asserting equality is already done
    // by `schema_hash_matches_baseline` — this test's job is to
    // ensure the string for AbilityCooldowns appears in the hash
    // input. Can instead assert a grep-able substring of the hash
    // builder output.
    let _ = hash_hex;
    // Cheap placeholder: the real regression is the existing
    // schema_hash_matches_baseline test. This file exists to name
    // the concept.
}
```

Actually, the `schema_hash_matches_baseline` test in `crates/engine/tests/schema_hash.rs` already fences the whole hash against the baseline. As long as the baseline was updated in Task 2, this task has no new test to write — the regression fence already exists.

- [ ] **Step 3: Run perf sweep to confirm no unrelated regression**

Run: `cargo run --release --features gpu --bin xtask -- chronicle --perf-sweep --use-batch --batch-ticks 100 2>&1 | tail -10`
Expected: N=2048 batch µs/tick matches the Phase-2 baseline (~5500-5700 µs). Cooldowns are CPU-only changes — no GPU perf impact.

- [ ] **Step 4: No commit**

Verification task only.

---

## Notes for the implementing engineer

- **MAX_ABILITIES**: confirm the constant name + value before the first task. It's 8 per prior context but double-check with `grep -n "MAX_ABILITIES\|MAX_ABILITY_SLOTS" crates/engine/src/`.
- **`agent_slot` helper**: every `AgentId → usize` conversion goes through a specific helper method. Match whatever the engine calls it.
- **Schema hash bump discipline**: running `cargo test -p engine schema_hash` should fail after Task 2's config addition (new field → new hash). Update `crates/engine/.schema_hash` baseline immediately to keep the tree green.
- **GCD default (5)**: picked for 0.5s at 10 Hz tick rate. Per spec, configurable via `combat.global_cooldown_ticks`. Default of 5 is designer-tunable.
- **Side buffer on CPU only**: this subsystem lands `ability_cooldowns` as a CPU-side Vec. Subsystem (3) will expose it to GPU via a side buffer. Don't add GPU plumbing here.
- **Tests that were writing `hot_cooldown_next_ready_tick` directly**: they'll keep working (the field is unchanged). Tests that compare against the new semantics need updating — flag any such case with `DONE_WITH_CONCERNS` if the fix extends beyond the listed files.

## Open questions

- **Cooldown reduction mechanics** (abilities that reset other abilities' cooldowns): out of scope; YAGNI.
- **Per-school GCD** (magic/physical/movement gate subsets): out of scope; YAGNI.
- **Cooldown charges / stacks** (abilities with N charges, each replenishing independently): out of scope.
- **`MAX_ABILITIES` raise**: if future subsystems need > 8 slots, the side-buffer approach accommodates — each agent's `[u32; MAX_ABILITIES]` scales linearly. Subsystem (3) work reviews whether 8 is enough.
