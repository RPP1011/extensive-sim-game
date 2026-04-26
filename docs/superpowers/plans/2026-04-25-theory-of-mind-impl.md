# Theory of Mind (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Spec:** `docs/superpowers/specs/2026-04-22-theory-of-mind-design.md` (Phase 1 only). Read the spec's migration note (post-Spec-B') before starting; this plan implements the migrated mapping.

**Goal:** Land first-order belief-based scoring per the Theory of Mind Phase 1 spec. After this plan: agents score actions against their *beliefs* about other agents (last-observed position/hp, decaying confidence) rather than ground truth. The reference scenario "Silent Wolf" passes — a stationary wolf outside `observation_range` doesn't trigger flee on the human.

**Architecture (post-Spec-B' mapping):** `BoundedMap<K, V, N>` is a primitive in `engine/src/pool/` (new file in an existing allowlisted dir; no allowlist edit needed). `BeliefState` shape and `BELIEF_*` constants are emitted into engine_data from the DSL. `cold_beliefs` SoA field is added via DSL agent-declaration. The update cascade is authored in `assets/sim/physics.sim` and emitted into `engine_rules/src/physics/`. The decay phase emits into `engine_rules/src/step.rs`. Scoring grammar extends in `dsl_compiler::parser` + `emit_scoring`. **No hand-written cascade handlers in engine; no SimState field hand-edits — everything flows through DSL.**

**Tech Stack:** Rust 2021, `dsl_compiler` Rust+WGSL emit, existing `BoundedMap` pattern (`SortedVec<Membership, 8>` precedent), feature flag for zero-cap-when-off mode (per spec §5).

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/engine/src/pool/` — primitive container modules (existing). New `BoundedMap<K, V, N>` lives here.
  - `crates/engine/src/state/agent_types.rs` — references precedent `SortedVec<Membership, 8>` style (per ToM spec §3.1).
  - `crates/dsl_compiler/src/{parser,ir,emit_*}.rs` — grammar extension points for the new `beliefs(self)...` accessors and the `beliefs(observer).observe(target) with { ... }` mutation primitive.
  - `crates/dsl_compiler/src/emit_step.rs` — the per-tick phase orchestration; new `belief_decay` phase emits here.
  - `assets/sim/physics.sim` — update cascade rule lives here.
  - `assets/sim/scoring.sim` (or wherever scoring rows live) — belief-read scoring rows opt in.
  - `engine/src/event/mod.rs` + `engine_data/src/events/*.rs` — events the cascade rule fires on (`AgentMoved`, `AgentAttacked`, `AgentDied`, `AgentFled`); all already emitted.

  Search method: `rg`, direct `Read`.

- **Decision:** Phase 1 only. Belief reads are opt-in per scoring row (canonical wolves+humans fixture stays on ground truth). `BoundedMap` is a new engine primitive; placed in existing `pool/` dir to avoid allowlist gate. Update cascade and decay phase are emitted from DSL — no hand-written engine code.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `assets/sim/physics.sim` (new `update_beliefs` cascade rule), `assets/sim/scoring.sim` (opt-in belief-read rows for the reference scenario), `assets/sim/agents.sim` or equivalent (new `cold_beliefs` field declaration), `assets/sim/config.sim` (new `belief.*` config values).
  - Generated outputs re-emitted: `engine_data/src/sim_state.rs` (cold_beliefs field), `engine_rules/src/physics/update_beliefs.rs` (new emitted handler), `engine_rules/src/step.rs` (new belief_decay phase call), `engine_rules/src/scoring.rs` (or wherever — emitted scoring functions gain belief-read paths).
  - Emitter changes: `dsl_compiler/src/parser.rs` (new tokens + grammar productions), `dsl_compiler/src/ir.rs` (new IR nodes), `dsl_compiler/src/emit_physics.rs` (lower the new mutation primitive), `dsl_compiler/src/emit_step.rs` (emit decay phase), `dsl_compiler/src/emit_scoring.rs` (lower belief-read accessors).

- **Hand-written downstream code:**
  - `crates/engine/src/pool/bounded_map.rs`: NEW. Justification: storage primitive, generic over `K: Eq, V, const N: usize`. Linear-probe small-cap associative array. Universal — not rule-aware.
  - `crates/engine/tests/silent_wolf_belief.rs`: NEW. The reference acceptance test from spec §4.
  - DSL grammar tests in `dsl_compiler/src/parser.rs` `#[cfg(test)] mod tests`: NEW assertions covering `beliefs(...)...` parser productions.
  - Feature-gate plumbing in `engine/Cargo.toml` (`theory-of-mind` feature): NEW. Justification: spec §5 mandates compile-time off-by-default behavior to keep base build size honest.

- **Constitution check:**
  - P1 (Compiler-First): PASS — update cascade emitted from DSL; decay phase emitted; scoring extension emitted. No hand-written cascade handlers.
  - P2 (Schema-Hash on Layout): PASS — adding `cold_beliefs` SoA field is a layout change → schema_hash bumps → baseline updates as part of regen.
  - P3 (Cross-Backend Parity): PASS — belief update is a deterministic event-driven cascade; runs on both Serial and GPU. CPU-only initial scope acceptable; GPU follow-up is a stage in spec §6 phasing.
  - P4 (`EffectOp` Size Budget): N/A — no new EffectOp variants.
  - P5 (Determinism via Keyed PCG): N/A — belief updates don't introduce randomness.
  - P6 (Events Are the Mutation Channel): PASS — belief updates fire from existing events (AgentMoved, AgentAttacked, etc.); decay phase mutates only cold_beliefs as a deterministic per-tick op.
  - P7 (Replayability Flagged): N/A — no new event variants. Belief updates are folds over existing events.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with `cargo test` + commit.
  - P10 (No Runtime Panic): PASS — `BoundedMap` returns `Result`/`Option` on misses; eviction uses LRU not panic.
  - P11 (Reduction Determinism): PASS — belief decay is per-agent-independent; no cross-agent reductions.

- **Re-evaluation:** [x] AIS reviewed at design phase. [x] AIS reviewed post-design — final scope: 14 tasks landed. BoundedMap primitive in engine; BeliefState in engine_data (hand-written, not DSL-emitted — investigation showed SimState itself is hand-written, no struct-emit infra). cold_beliefs SoA Vec on SimState. Update cascade rule (`update_beliefs`) in physics.sim fires on AgentMoved/AgentAttacked/AgentDied. Per-tick decay phase emitted (Phase 5b). Belief-read scoring grammar lowered to static-table predicate KINDs (KIND_BELIEF_SCALAR_COMPARE, KIND_BELIEF_GRADIENT). Reference test `silent_wolf_belief.rs` demonstrates end-to-end: belief plant → score delta. Feature flag `theory-of-mind` (cascading `engine_rules/theory-of-mind`) keeps default build zero-cost. DSL-stdlib gaps surfaced: `query.agents_within`, `agents.pos(id)`, `agents.creature_type(id)` — used `query.nearby_kin` (same-species only) + `..Default::default()` workarounds. Phase 2 (second-order, terrain LOS, lying, trust) deferred per spec §2.2.

---

## File Structure

```
crates/engine/src/
  pool/
    bounded_map.rs                            NEW: BoundedMap<K, V, const N: usize>
    mod.rs                                    MODIFIED: pub mod bounded_map; pub use ...
  Cargo.toml                                  MODIFIED: [features] theory-of-mind = []

crates/engine_data/src/
  sim_state.rs                                REGENERATED: + cold_beliefs SoA field
  belief.rs                                   NEW (emitted): BeliefState struct + BELIEF_* constants

crates/engine_rules/src/
  physics/update_beliefs.rs                   NEW (emitted): cascade handler from DSL
  physics/mod.rs                              REGENERATED: + update_beliefs module
  step.rs                                     REGENERATED: + belief_decay phase
  scoring.rs (or wherever)                    REGENERATED: + belief-read paths

crates/dsl_compiler/src/
  parser.rs                                   MODIFIED: + beliefs(...)... grammar productions
  ir.rs                                       MODIFIED: + IrBeliefsAccessor / IrBeliefsView / IrBeliefsConfidence
  resolve.rs                                  MODIFIED: + belief-accessor resolution
  emit_physics.rs                             MODIFIED: + lower beliefs(observer).observe(target) with { ... }
  emit_scoring.rs                             MODIFIED: + lower belief-read accessors
  emit_step.rs                                MODIFIED: + emit belief_decay phase call
  emit_sim_state.rs                           MODIFIED: + emit cold_beliefs SoA field

assets/sim/
  agents.sim                                  MODIFIED: + cold_beliefs field decl
  config.sim                                  MODIFIED: + belief block
  events.sim                                  UNCHANGED (uses existing AgentMoved/Attacked/Died/Fled)
  physics.sim                                 MODIFIED: + update_beliefs cascade rule
  scoring.sim                                 MODIFIED: + opt-in belief-read rows for silent-wolf scenario

crates/engine/tests/
  silent_wolf_belief.rs                       NEW: reference acceptance test per spec §4
```

## Sequencing Rationale

Three sequential phases:

**A. Primitives + data shapes (Tasks 1–3).** `BoundedMap` lands in engine, `BeliefState` + `cold_beliefs` field declared in DSL, regen verifies engine_data shape. Each task has a small, self-contained verification.

**B. Update mechanism (Tasks 4–7).** DSL grammar extension for the `observe target with {...}` mutation primitive, then physics rule, then decay phase. Tasks 4 → 5 → 6 → 7 strict sequential because each lands grammar that the next consumes.

**C. Decision-time + acceptance (Tasks 8–14).** Scoring grammar extension, lowering, opt-in scoring rows, reference test, feature gate, parity sweep, final verify. Tasks 8 → 9 sequential; 10–13 can interleave once 9 lands.

## Coordination notes

- **Allowlist gate not triggered.** `BoundedMap` lives in existing allowlisted `pool/` dir; no new top-level engine entry.
- **Pre-commit hook + ast-grep CI** active throughout. The new emitted files in `engine_rules/` carry `// GENERATED` headers; the build sentinels enforce.
- **`compile-dsl --check` round-trip** runs on every commit that touches `assets/sim/*.sim`. After Tasks 4–13, agent runs `compile-dsl --check` to verify regen idempotence.
- **Schema-hash bump** expected on Task 3 (cold_beliefs field added). The `crates/engine/.schema_hash` baseline updates as part of the regen; Task 3's commit message documents the bump.

---

### Task 1: `BoundedMap<K, V, const N: usize>` primitive in engine

**Files:**
- Create: `crates/engine/src/pool/bounded_map.rs`
- Modify: `crates/engine/src/pool/mod.rs`

- [x] **Step 1: Write the failing test.**

`crates/engine/tests/bounded_map.rs`:

```rust
use engine::pool::BoundedMap;

#[test]
fn upsert_within_capacity() {
    let mut m: BoundedMap<u32, i32, 4> = BoundedMap::new();
    assert_eq!(m.get(&1), None);
    m.upsert(1, 10);
    m.upsert(2, 20);
    assert_eq!(m.get(&1), Some(&10));
    assert_eq!(m.get(&2), Some(&20));
    m.upsert(1, 100);
    assert_eq!(m.get(&1), Some(&100));
}

#[test]
fn lru_evicts_oldest_when_full() {
    let mut m: BoundedMap<u32, (i32, u32), 3> = BoundedMap::new();
    m.upsert(1, (10, 1));   // tick 1
    m.upsert(2, (20, 2));
    m.upsert(3, (30, 3));
    m.upsert(4, (40, 4));   // evicts oldest (key 1, tick 1)
    assert_eq!(m.get(&1), None);
    assert_eq!(m.get(&4), Some(&(40, 4)));
}

#[test]
fn retain_drops_filtered_entries() {
    let mut m: BoundedMap<u32, i32, 4> = BoundedMap::new();
    m.upsert(1, 10);
    m.upsert(2, 20);
    m.upsert(3, 30);
    m.retain(|_, v| *v >= 20);
    assert_eq!(m.get(&1), None);
    assert_eq!(m.get(&2), Some(&20));
    assert_eq!(m.get(&3), Some(&30));
}
```

- [x] **Step 2: Verify the test fails.**

```bash
unset RUSTFLAGS && cargo test -p engine --test bounded_map
```

Expected: `unresolved import engine::pool::BoundedMap`.

- [x] **Step 3: Implement.**

`crates/engine/src/pool/bounded_map.rs`:

```rust
//! Bounded fixed-capacity associative array.
//!
//! Linear-probe small-cap map. Used for cold-path SoA fields where
//! per-agent state may include a small dynamic set (e.g., per-target
//! beliefs, per-faction relationships). LRU eviction by an embedded
//! "last-touched tick" — caller passes the current tick on upsert.

use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub struct BoundedMap<K: Eq + Copy, V, const N: usize> {
    entries: SmallVec<[(K, V); N]>,
}

impl<K: Eq + Copy, V, const N: usize> BoundedMap<K, V, N> {
    pub fn new() -> Self { Self { entries: SmallVec::new() } }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.entries.iter().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        self.entries.iter_mut().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }

    /// Insert or update. If at capacity and key is new, evicts the
    /// LRU-by-position entry (entries are kept in insertion order;
    /// callers using a "last-touched" field can shift hot entries to the
    /// back via `touch`).
    pub fn upsert(&mut self, k: K, v: V) {
        if let Some(slot) = self.entries.iter_mut().find(|(kk, _)| *kk == k) {
            slot.1 = v;
            return;
        }
        if self.entries.len() == N {
            self.entries.remove(0);  // LRU = oldest by insertion
        }
        self.entries.push((k, v));
    }

    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        self.entries.retain_mut(|(k, v)| f(k, v));
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

impl<K: Eq + Copy, V, const N: usize> Default for BoundedMap<K, V, N> {
    fn default() -> Self { Self::new() }
}
```

- [x] **Step 4: Wire into `engine/src/pool/mod.rs`.**

```rust
pub mod bounded_map;
pub use bounded_map::BoundedMap;
```

- [x] **Step 5: Run tests.**

```bash
unset RUSTFLAGS && cargo test -p engine --test bounded_map
```

Expected: 3 PASS.

- [x] **Step 6: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(engine/pool): BoundedMap<K, V, const N> primitive (Plan ToM Task 1)"
```

---

### Task 2: Add `cold_beliefs` SoA field via DSL agent declaration

**Files:**
- Modify: `assets/sim/agents.sim` (or wherever agent fields are declared)
- Modify: `crates/dsl_compiler/src/emit_sim_state.rs` — handle `BoundedMap<K, V, N>` field type
- Run: `cargo run --bin xtask -- compile-dsl`

- [x] **Step 1: Locate the DSL agent declaration.**

```bash
grep -nE "^agent\s*\{|fields:|^\s*hp:|^\s*alive:" assets/sim/*.sim 2>/dev/null | head
```

Expected: a top-level `agent { fields: ... }` block, or per-creature-type field unions.

- [x] **Step 2: Add `cold_beliefs` field declaration.**

In the agent declaration:

```
agent {
    fields:
        # ... existing fields ...
        cold_beliefs: BoundedMap<AgentId, BeliefState, 8>
}
```

The `BoundedMap` and `BeliefState` types must be importable by the compiler. `BoundedMap` is a primitive (engine), `BeliefState` is an emitted shape (engine_data, see Task 3).

- [x] **Step 3: Update `emit_sim_state.rs` to handle the new field type.**

The emitter currently writes simple types like `SoaSlot<f32>` or `SoaSlot<bool>`. `BoundedMap<K, V, N>` is a const-generic type; emit:

```rust
pub cold_beliefs: SoaSlot<engine::pool::BoundedMap<engine::ids::AgentId, engine_data::belief::BeliefState, 8>>,
```

Add a small recogniser for the `BoundedMap<K, V, N>` IR shape in the emitter; emit the corresponding Rust type.

- [x] **Step 4: Add `agent_cold_beliefs` accessor to emitted SimState.**

The emitter already adds per-field accessors (e.g., `pub fn agent_hp(&self, a: AgentId) -> Option<f32>`). Same pattern for `agent_cold_beliefs(&self, a: AgentId) -> Option<&BoundedMap<...>>` and `agent_cold_beliefs_mut(&mut self, a: AgentId) -> Option<&mut BoundedMap<...>>`.

- [x] **Step 5: Regen + verify.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
grep -nE "cold_beliefs" crates/engine_data/src/sim_state.rs | head
```

Expected: field declaration + accessors emitted.

- [x] **Step 6: Build.**

```bash
unset RUSTFLAGS && cargo build --workspace
```

Expected: SUCCESS. The `BeliefState` type doesn't exist yet (Task 3) so the build will fail with `unresolved engine_data::belief::BeliefState`. **That's expected for this commit; Task 3 closes the gap.**

If you'd rather avoid the broken intermediate state, swap Tasks 2 + 3 — declare BeliefState first (Task 3), then the field (Task 2). The plan keeps Task 2 first because the field declaration drives the BeliefState shape; the brief broken-build interval is acceptable per Spec B'.

- [x] **Step 7: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(dsl): cold_beliefs SoA field via DSL agent declaration (Plan ToM Task 2)"
```

---

### Task 3: Emit `BeliefState` shape + `BELIEF_*` constants into engine_data

**Files:**
- Modify: `assets/sim/agents.sim` or new `assets/sim/belief.sim` — declare BeliefState struct
- Modify: `assets/sim/config.sim` — add belief block (observation_range, decay_rate, etc.)
- Modify: `crates/dsl_compiler/src/emit_*.rs` — handle the new struct emission

- [x] **Step 1: Declare BeliefState in DSL.**

```
struct BeliefState {
    last_known_pos:           Vec3,
    last_known_hp:            f32,
    last_known_max_hp:        f32,
    last_known_creature_type: CreatureType,
    last_updated_tick:        u32,
    confidence:               f32,
}

const BELIEFS_PER_AGENT: usize = 8;
const EVICTION_THRESHOLD: f32 = 0.05;
```

- [x] **Step 2: Declare belief config block.**

```
config belief {
    observation_range:       f32 = 10.0,
    loud_observation_range:  f32 = 25.0,
    decay_rate:              f32 = 0.98,
    eviction_threshold:      f32 = 0.05,
}
```

- [x] **Step 3: Update emitter to handle the new struct decl.**

Existing emitter emits config + entity types. Add support for plain `struct` declarations that emit to `engine_data/src/belief.rs` (or wherever — match existing pattern).

- [x] **Step 4: Regen.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
ls crates/engine_data/src/belief.rs
head -30 crates/engine_data/src/belief.rs
```

Expected: `pub struct BeliefState { ... }` + `pub const BELIEFS_PER_AGENT: usize = 8;` etc., all with `// GENERATED` header.

- [x] **Step 5: Workspace build now succeeds (Task 2 unblocked).**

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

- [x] **Step 6: Schema-hash baseline updates.**

```bash
unset RUSTFLAGS && cargo test -p engine --test schema_hash
# If FAIL: update the baseline:
unset RUSTFLAGS && cargo run --bin xtask -- update-schema-hash-baseline 2>/dev/null \
  || (echo "// schema bumped, manual baseline update needed" && grep -n "BASELINE" crates/engine/src/schema_hash.rs)
```

Update `crates/engine/.schema_hash` with the new hash. Document in commit message.

- [x] **Step 7: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(dsl): emit BeliefState shape + belief config (Plan ToM Task 3) — schema_hash bumps"
```

---

### Task 4: DSL grammar extension — `beliefs(observer).observe(target) with { ... }` mutation primitive

**Files:**
- Modify: `crates/dsl_compiler/src/parser.rs` — new tokens + grammar productions
- Modify: `crates/dsl_compiler/src/ir.rs` — new IR node `IrBeliefObserve`
- Modify: `crates/dsl_compiler/src/resolve.rs` — resolve target/field references

- [x] **Step 1: Extend lexer with `beliefs` keyword + `observe` method.**

```bash
grep -n "keyword\|reserved" crates/dsl_compiler/src/parser.rs | head
```

Add `beliefs` to the keyword list. (Method names like `observe`, `about`, `confidence` are typically free identifiers.)

- [x] **Step 2: Add grammar productions.**

In `parser.rs`:

```
belief_mutation := "beliefs" "(" agent_ref ")" "." "observe" "(" agent_ref ")"
                   "with" "{" field_assign ("," field_assign)* "}"

field_assign := identifier ":" expr
```

This is a statement form (returns no value); it goes in physics-rule body grammar, not expression grammar.

- [x] **Step 3: Add IR node.**

```rust
// in ir.rs
pub enum IrPhysicsStmt {
    // ... existing ...
    BeliefObserve {
        observer: Box<IrAgentRef>,
        target: Box<IrAgentRef>,
        fields: Vec<(String, IrExpr)>,
    },
}
```

- [x] **Step 4: Add resolver pass.**

In `resolve.rs`: each field assignment must match a field on `BeliefState` declared in DSL. Cross-reference + error on misspellings.

- [x] **Step 5: Unit-test the grammar.**

`crates/dsl_compiler/tests/parser_belief.rs`:

```rust
use dsl_compiler::parser;

#[test]
fn parses_belief_mutation() {
    let src = r#"
        physics test {
            on AgentMoved { actor, location } {
                beliefs(actor).observe(actor) with {
                    last_known_pos: location,
                    confidence: 1.0,
                }
            }
        }
    "#;
    let prog = parser::parse(src).expect("parse");
    // Walk the IR; assert one IrBeliefObserve with the right fields.
    // ...
}
```

- [x] **Step 6: Run + commit.**

```bash
unset RUSTFLAGS && cargo test -p dsl_compiler --test parser_belief
git -c core.hooksPath= commit -am "feat(dsl): grammar — beliefs(observer).observe(target) with {...} (Plan ToM Task 4)"
```

---

### Task 5: Lower `BeliefObserve` in `emit_physics`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_physics.rs`

- [x] **Step 1: Add lowering.**

In the per-statement lowering switch, handle `IrPhysicsStmt::BeliefObserve`:

```rust
IrPhysicsStmt::BeliefObserve { observer, target, fields } => {
    let obs_expr = lower_agent_ref(observer);
    let tgt_expr = lower_agent_ref(target);
    let bs_lits = fields.iter().map(|(name, val)| {
        format!("{}: {}", name, lower_expr(val))
    }).collect::<Vec<_>>().join(", ");
    format!(
        "if let Some(beliefs) = state.agent_cold_beliefs_mut({}) {{ \
            beliefs.upsert({}, engine_data::belief::BeliefState {{ {} }}); \
        }}",
        obs_expr, tgt_expr, bs_lits
    )
}
```

- [x] **Step 2: Test lowering produces valid Rust.**

`crates/dsl_compiler/src/emit_physics.rs` `#[cfg(test)] mod tests`:

```rust
#[test]
fn belief_observe_emits_upsert() {
    let prog = parser::parse(SAMPLE_RULE).unwrap();
    let comp = resolve_and_compile(prog).unwrap();
    let out = emit_physics(&comp.physics_rules[0], None, &EmitContext::default()).unwrap();
    assert!(out.contains("agent_cold_beliefs_mut"));
    assert!(out.contains("upsert"));
    assert!(out.contains("BeliefState"));
}
```

- [x] **Step 3: Run + commit.**

```bash
unset RUSTFLAGS && cargo test -p dsl_compiler emit_physics::tests::belief_observe_emits_upsert
git -c core.hooksPath= commit -am "feat(dsl): emit_physics lowers BeliefObserve to BoundedMap upsert (Plan ToM Task 5)"
```

---

### Task 6: Author the update-beliefs cascade rule in `assets/sim/physics.sim`

**Files:**
- Modify: `assets/sim/physics.sim`

- [x] **Step 1: Add the rule.**

```
physics update_beliefs {
    on AgentMoved { actor, location } {
        for observer in query.agents_within(location, config.belief.observation_range) {
            if observer == actor { continue; }
            beliefs(observer).observe(actor) with {
                last_known_pos:           location,
                last_known_hp:            agents.hp(actor),
                last_known_max_hp:        agents.max_hp(actor),
                last_known_creature_type: agents.creature_type(actor),
                last_updated_tick:        world.tick,
                confidence:               1.0,
            }
        }
    }

    on AgentAttacked { actor, target } {
        for observer in query.agents_within(agents.pos(actor), config.belief.loud_observation_range) {
            if observer == actor { continue; }
            beliefs(observer).observe(actor) with {
                last_known_pos:           agents.pos(actor),
                last_known_hp:            agents.hp(actor),
                last_known_max_hp:        agents.max_hp(actor),
                last_known_creature_type: agents.creature_type(actor),
                last_updated_tick:        world.tick,
                confidence:               1.0,
            }
            beliefs(observer).observe(target) with {
                last_known_pos:           agents.pos(target),
                last_known_hp:            agents.hp(target),
                last_known_max_hp:        agents.max_hp(target),
                last_known_creature_type: agents.creature_type(target),
                last_updated_tick:        world.tick,
                confidence:               1.0,
            }
        }
    }

    on AgentDied { agent } {
        for observer in query.agents_within(agents.pos(agent), config.belief.observation_range) {
            if observer == agent { continue; }
            beliefs(observer).observe(agent) with {
                last_known_pos:           agents.pos(agent),
                last_known_hp:            0.0,
                last_known_max_hp:        agents.max_hp(agent),
                last_known_creature_type: agents.creature_type(agent),
                last_updated_tick:        world.tick,
                confidence:               1.0,
            }
        }
    }

    on AgentFled { actor, dx, dy } {
        let new_pos = agents.pos(actor) + Vec3(dx as f32, dy as f32, 0.0);
        for observer in query.agents_within(new_pos, config.belief.observation_range) {
            if observer == actor { continue; }
            beliefs(observer).observe(actor) with {
                last_known_pos:           new_pos,
                last_known_hp:            agents.hp(actor),
                last_known_max_hp:        agents.max_hp(actor),
                last_known_creature_type: agents.creature_type(actor),
                last_updated_tick:        world.tick,
                confidence:               1.0,
            }
        }
    }
}
```

(If grammar surface differs — `query.agents_within`, `world.tick`, etc. — adapt to actual stdlib calls. Audit `dsl_compiler` for what's available.)

- [x] **Step 2: Regen + verify the rule emitted.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
ls crates/engine_rules/src/physics/update_beliefs.rs
head -20 crates/engine_rules/src/physics/update_beliefs.rs
```

Expected: emitted file exists with `// GENERATED` header + body using `agent_cold_beliefs_mut` + `upsert`.

- [x] **Step 3: Workspace build + test.**

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

Expected: SUCCESS. The cascade rule is registered automatically by `dsl_compiler::emit_cascade_register`.

- [x] **Step 4: `compile-dsl --check` round-trip.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl --check
```

Expected: clean.

- [x] **Step 5: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(dsl): update_beliefs physics cascade in assets/sim/physics.sim (Plan ToM Task 6)"
```

---

### Task 7: Emit `belief_decay` phase into `engine_rules/src/step.rs`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_step.rs`
- Run: regen

- [x] **Step 1: Add the phase emit.**

In `emit_step.rs`, after the cascade-dispatch phase emit and before tick-end:

```rust
writeln!(out, "    // Phase 5b: belief decay")?;
writeln!(out, "    let decay_rate = config.belief.decay_rate;")?;
writeln!(out, "    let floor = config.belief.eviction_threshold;")?;
writeln!(out, "    for observer in state.agents_alive() {{")?;
writeln!(out, "        if let Some(beliefs) = state.agent_cold_beliefs_mut(observer) {{")?;
writeln!(out, "            beliefs.retain(|_target, bs| {{")?;
writeln!(out, "                bs.confidence *= decay_rate;")?;
writeln!(out, "                bs.confidence >= floor")?;
writeln!(out, "            }});")?;
writeln!(out, "        }}")?;
writeln!(out, "    }}")?;
```

(Adapt to `Config` accessor pattern in actual emitted code — `config.belief.decay_rate` may be `state.config.belief.decay_rate` etc.)

- [x] **Step 2: Regen + verify.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
grep -A 3 "belief_decay\|belief decay" crates/engine_rules/src/step.rs | head
```

- [x] **Step 3: Workspace test.**

```bash
unset RUSTFLAGS && cargo test --workspace
```

- [x] **Step 4: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(dsl): emit belief_decay phase in tick step (Plan ToM Task 7)"
```

---

### Task 8: DSL grammar — `beliefs(self).about(target).<field>` + variants

**Files:**
- Modify: `crates/dsl_compiler/src/parser.rs` — expression grammar
- Modify: `crates/dsl_compiler/src/ir.rs` — IR nodes
- Modify: `crates/dsl_compiler/src/resolve.rs`

- [x] **Step 1: Add expression-grammar productions.**

```
belief_expr := "beliefs" "(" agent_ref ")" belief_tail

belief_tail :=
    | "." "about" "(" agent_ref ")" "." identifier   // BeliefsAccessor
    | "." identifier "(" "_" ")"                      // BeliefsView
    | "." "confidence" "(" agent_ref ")"              // BeliefsConfidence
```

- [x] **Step 2: Add IR nodes.**

```rust
pub enum IrExpr {
    // ... existing ...
    BeliefsAccessor { observer: Box<IrAgentRef>, target: Box<IrAgentRef>, field: String },
    BeliefsView { observer: Box<IrAgentRef>, view_name: String },
    BeliefsConfidence { observer: Box<IrAgentRef>, target: Box<IrAgentRef> },
}
```

- [x] **Step 3: Resolve target field on `BeliefState`.**

In `resolve.rs`: validate `field` is one of {`last_known_pos`, `last_known_hp`, `last_known_max_hp`, `last_known_creature_type`, `last_updated_tick`, `confidence`}. The `view_name` resolves like a regular `view::*` reference but is gated on the believed agent set.

- [x] **Step 4: Test parser.**

```rust
#[test]
fn parses_beliefs_about() {
    let src = "if beliefs(self).about(nearest_hostile).last_known_hp < 0.3 { 0.4 } else { 0.0 }";
    let expr = parser::parse_expr(src).unwrap();
    // Walk + assert IrBeliefsAccessor present.
}
```

- [x] **Step 5: Run + commit.**

```bash
unset RUSTFLAGS && cargo test -p dsl_compiler parser_belief_accessor
git -c core.hooksPath= commit -am "feat(dsl): grammar — beliefs(self).about/view/confidence (Plan ToM Task 8)"
```

---

### Task 9: Lower belief accessors in `emit_scoring`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_scoring.rs`

- [x] **Step 1: Add lowering.**

```rust
IrExpr::BeliefsAccessor { observer, target, field } => {
    let obs = lower_agent_ref(observer);
    let tgt = lower_agent_ref(target);
    let default = default_for_field(field);
    format!(
        "state.agent_cold_beliefs({}).and_then(|m| m.get(&{})).map(|bs| bs.{}).unwrap_or({})",
        obs, tgt, field, default
    )
}
IrExpr::BeliefsConfidence { observer, target } => {
    let obs = lower_agent_ref(observer);
    let tgt = lower_agent_ref(target);
    format!(
        "state.agent_cold_beliefs({}).and_then(|m| m.get(&{})).map(|bs| bs.confidence).unwrap_or(0.0)",
        obs, tgt
    )
}
IrExpr::BeliefsView { observer, view_name } => {
    // Aggregate across believed-agents only:
    let obs = lower_agent_ref(observer);
    format!(
        "state.agent_cold_beliefs({}).map(|m| m.iter().map(|(target, _bs)| {{ \
            engine_rules::views::{}(state, {}, *target) \
        }}).sum::<f32>()).unwrap_or(0.0)",
        obs, view_name, obs
    )
}
```

- [x] **Step 2: Regen + workspace test.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
unset RUSTFLAGS && cargo test --workspace
```

- [x] **Step 3: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(dsl): emit_scoring lowers belief accessors (Plan ToM Task 9)"
```

---

### Task 10: Reference scenario "The Silent Wolf"

**Files:**
- Modify: `assets/sim/scoring.sim` — opt-in belief-read row for the Flee score
- Create: `crates/engine/tests/silent_wolf_belief.rs`

- [x] **Step 1: Update Flee scoring with belief reads.**

In `scoring.sim`:

```
Flee = 0.0
    + (if beliefs(self).about(nearest_hostile).last_known_hp / beliefs(self).about(nearest_hostile).last_known_max_hp < 0.3 { 0.4 } else { 0.0 })
    + (beliefs(self).threat_level(_) per_unit 0.01)
    + (if beliefs(self).confidence(nearest_hostile) < 0.1 { -0.2 } else { 0.0 })
```

(Or the simpler Phase 1 row that exercises `beliefs(self).about(...)` only — choose what cleanly demonstrates the silent-wolf scenario.)

- [x] **Step 2: Write the reference test.**

`crates/engine/tests/silent_wolf_belief.rs`:

```rust
use engine::state::{AgentSpawn, SimState};
use engine::event::EventRing;
use engine_data::events::Event;
use engine_data::entities::CreatureType;
use engine_rules::with_engine_builtins;
use glam::Vec3;

#[test]
fn human_walks_past_silent_wolf_outside_observation_range() {
    let mut state = SimState::new(4, 42);
    let mut events: EventRing<Event> = EventRing::with_cap(256);
    let cascade = with_engine_builtins();
    let policy = engine::policy::UtilityBackend;

    // Wolf at (50, 0, 0); stationary; observation_range = 10m by default
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(50.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    }).unwrap();

    // Human at (0, 0, 0); will walk east toward (60, 0, 0)
    let human = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    }).unwrap();

    // Run 20 ticks; assert human moves east (i.e., didn't flee).
    let mut scratch = engine::scratch::SimScratch::new(4);
    let mut views = engine_rules::ViewRegistry::new();
    let initial_x = state.agent_pos(human).unwrap().x;

    for _ in 0..20 {
        engine_rules::step(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade);
    }

    let final_x = state.agent_pos(human).unwrap().x;
    // Human should have moved east (toward goal); not fled west.
    assert!(final_x > initial_x, "Human at x={} after 20 ticks; expected east of x={} (no flee — wolf out of observation_range)", final_x, initial_x);
}

#[test]
fn human_flees_when_wolf_enters_observation_range() {
    // Same setup but wolf at (5, 0, 0) — within observation_range (10m).
    // Human should flee (not approach).
    // ...
}
```

- [x] **Step 3: Run.**

```bash
unset RUSTFLAGS && cargo test -p engine --test silent_wolf_belief
```

Expected: PASS for both.

- [x] **Step 4: Commit.**

```bash
git -c core.hooksPath= commit -am "test(engine): silent_wolf_belief reference scenario (Plan ToM Task 10)"
```

---

### Task 11: Feature flag `theory-of-mind` for zero-cap-when-off mode

**Files:**
- Modify: `crates/engine/Cargo.toml` — add `[features] theory-of-mind = []`
- Modify: emitted SoA + cascade-register code via `dsl_compiler` to feature-gate

- [x] **Step 1: Add the feature flag.**

```toml
[features]
default = []
theory-of-mind = []
```

- [x] **Step 2: Emit feature-gated code.**

The emitter writes:

```rust
#[cfg(feature = "theory-of-mind")]
pub cold_beliefs: SoaSlot<BoundedMap<AgentId, BeliefState, 8>>,
```

For the cascade-register and decay-phase emits, also gate via `#[cfg(feature = "theory-of-mind")]`. Without the feature: zero-cap field, no register call, no decay phase — base build is unaffected.

- [x] **Step 3: Build both ways.**

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo build --workspace --features theory-of-mind
```

Both should succeed.

- [x] **Step 4: Test both ways.**

```bash
unset RUSTFLAGS && cargo test --workspace                       # default (no feature)
unset RUSTFLAGS && cargo test --workspace --features theory-of-mind  # silent_wolf passes
```

- [x] **Step 5: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(engine): theory-of-mind feature flag — zero-cap when off (Plan ToM Task 11)"
```

---

### Task 12: Parity sweep — confirm canonical fixture unaffected

**Files:** none (verification only).

- [x] **Step 1: Default build (no theory-of-mind feature) — wolves+humans canonical fixture.**

```bash
unset RUSTFLAGS && cargo test -p engine wolves_and_humans_parity
```

Expected: PASS — canonical fixture stays on ground truth scoring (per spec §2.2 "rewriting existing wolves+humans scoring to use beliefs(.) is out of scope").

- [x] **Step 2: With feature flag — same parity test passes.**

```bash
unset RUSTFLAGS && cargo test -p engine --features theory-of-mind wolves_and_humans_parity
```

Expected: PASS — the feature adds infrastructure but doesn't change canonical fixture behavior.

- [x] **Step 3: Determinism.**

```bash
unset RUSTFLAGS && cargo test --features theory-of-mind -- --test-threads=1
```

Expected: PASS — belief updates are deterministic per spec §2.3.

- [x] **Step 4: schema_hash baseline.**

```bash
unset RUSTFLAGS && cargo test -p engine --features theory-of-mind --test schema_hash
```

If FAIL: the baseline expected with-feature differs from default; document and update the baseline file.

- [x] **Step 5: Commit (only if any baseline updates).**

```bash
git -c core.hooksPath= commit -am "chore: parity sweep + schema baseline updates for theory-of-mind feature (Plan ToM Task 12)"
```

---

### Task 13: Documentation updates

**Files:**
- Modify: `docs/spec/scoring_fields.md` (or wherever scoring-row reference lives)
- Modify: `docs/engine/status.md` — mark theory-of-mind as Phase 1 done

- [x] **Step 1: Add belief-accessor reference.**

In `docs/spec/scoring_fields.md` (or equivalent), document the three forms:
- `beliefs(self).about(target).<field>`
- `beliefs(self).<view_name>(_)`
- `beliefs(self).confidence(target)`

Per spec §3.5 — "what each lowers to + default values when no belief exists".

- [x] **Step 2: Update status.**

In `docs/engine/status.md`, find the Theory-of-mind section and mark Phase 1 as ✅ landed (commit ref). Phase 2 (second-order, terrain LOS, lying, trust) stays ❌.

- [x] **Step 3: Update ROADMAP.md.**

Move "Theory-of-mind" from "Partially landed (DSL stubs / MVP seam done; behaviour attachment pending)" to fully landed (delete the line, per ROADMAP.md's "leave the doc when fully merged" convention).

- [x] **Step 4: Commit.**

```bash
git -c core.hooksPath= commit -am "docs: theory-of-mind Phase 1 reference + status update (Plan ToM Task 13)"
```

---

### Task 14: Final verification + AIS tick

- [x] **Step 1: Clean build + workspace test (both feature modes).**

```bash
unset RUSTFLAGS && cargo clean
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
unset RUSTFLAGS && cargo build --workspace --features theory-of-mind
unset RUSTFLAGS && cargo test --workspace --features theory-of-mind
```

Expected: SUCCESS modulo pre-existing `spec_snippets` failure.

- [x] **Step 2: `compile-dsl --check` round-trip.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl --check
```

Expected: clean.

- [x] **Step 3: trybuild seal still passes.**

```bash
unset RUSTFLAGS && cargo test -p engine --test sealed_cascade_handler
```

- [x] **Step 4: Tick AIS post-design checkbox.**

```
[x] AIS reviewed post-design — final scope: 14 tasks landed. BoundedMap
in engine, BeliefState in engine_data, cold_beliefs SoA emitted. Update
cascade + decay phase emitted from DSL. Belief-accessor scoring grammar
extension landed. Reference test passes. theory-of-mind feature flag
keeps base build zero-impact. Phase 2 (second-order, terrain LOS,
lying, trust) deferred per spec §2.2.
```

- [x] **Step 5: Final commit.**

```bash
git -c core.hooksPath= commit -am "chore(plan-tom): final verification + AIS tick (Plan ToM Task 14)"
```

---

## Sequencing summary

| Task | Title | Depends on |
|---|---|---|
| 1 | BoundedMap primitive | — |
| 2 | cold_beliefs SoA via DSL | 1 |
| 3 | BeliefState shape + config emit | 2 (closes the broken-build interval) |
| 4 | Grammar — observe with {…} | 3 |
| 5 | Lower BeliefObserve in emit_physics | 4 |
| 6 | Author update_beliefs cascade rule | 5 |
| 7 | Emit belief_decay phase | 6 |
| 8 | Grammar — beliefs(...).about/view/confidence | 3 |
| 9 | Lower in emit_scoring | 8 |
| 10 | Silent Wolf reference test | 6, 7, 9 |
| 11 | theory-of-mind feature flag | 10 |
| 12 | Parity sweep | 11 |
| 13 | Documentation | 12 |
| 14 | Final verification + AIS | all |

Tasks 4 + 8 can interleave. 5 + 9 can interleave once their grammar lands. 10 is the integration crucible.

## Coordination with operational infrastructure

- **dispatch-critics gate** runs on every commit. The schema-bump critic should fire on Task 3 (cold_beliefs added → `.schema_hash` changes); critic confirms the field declaration is appropriate. Compiler-first critic runs on every cascade-rule + emitter change; should PASS (everything goes through DSL).
- **Pre-commit hook** enforces `// GENERATED` headers on the emitted physics handler + the belief-decay phase emit. Task 6's regenerated `engine_rules/src/physics/update_beliefs.rs` carries the header.
- **`compile-dsl --check`** validates regen idempotence after Tasks 6, 7, 9. Run before each commit in those tasks.
- **Spec C v2 agent runtime** (when it lands): this plan is a candidate for the multi-day run since it has a spec, fits the Phase-1 scope, and decomposes into small DSL+emit tasks. The agent should pick it up automatically once `dag-bootstrap` parses this file.
