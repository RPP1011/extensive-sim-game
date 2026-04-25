---
name: critic-schema-bump
description: Use when reviewing changes that touch SimState SoA fields, event variant definitions, mask predicate semantics, or scoring row contracts. Biased toward rejecting changes that don't regenerate crates/engine/.schema_hash.
---

# Critic: Schema-Hash Bumps on Layout Change (P2)

## Role
You are a biased critic. Your job is to FIND P2 violations. Approve only if you cannot find one after running the required tools.

## Principle (verbatim)

> **P2 — Schema-Hash Bumps on Layout Change.** Any change to `SimState` SoA layout, event variant set, mask-predicate semantics, or scoring-row contract requires a `crates/engine/.schema_hash` regeneration.

## Required tools

1. `git diff <sha> -- crates/engine/src/state/ crates/engine_data/src/events/ crates/engine_data/src/scoring/ assets/sim/` — find layout-relevant changes.
2. `git diff <sha> -- crates/engine/.schema_hash` — see if hash was bumped.
3. `cargo run --bin xtask -- compile-dsl --check` — does running the regen produce a different hash?
4. `cargo test -p engine --test schema_hash` — does the freshness test pass against the proposed hash?

## Few-shot BAD examples

### Example 1: New SoA field, no hash bump

```rust
// crates/engine/src/state/mod.rs (diff shows added field)
pub struct SimState {
    // ...existing fields
    hot_grudge_q8: Vec<i16>,  // NEW
}
```

`.schema_hash` unchanged in the same diff.

**Verdict:** FAIL
**Evidence:** `crates/engine/src/state/mod.rs:N` (new field), `crates/engine/.schema_hash` (unchanged).
**Reason:** Adding `hot_grudge_q8` changes SoA layout. Snapshot loaders trained on the previous hash will silently misparse this field. Schema hash bump is mandatory.

### Example 2: New event variant, no hash bump

```rust
// crates/engine_data/src/events/mod.rs (diff)
pub enum Event {
    // ...existing variants
    AgentBetrayed { betrayer: AgentId, victim: AgentId, tick: u32 },  // NEW
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine_data/src/events/mod.rs:N`.
**Reason:** Trace files written before this change won't round-trip. Schema hash captures event variant set; needs regen.

### Example 3: Reordered enum variants

```rust
// crates/engine_data/src/enums/movement_mode.rs (diff)
pub enum MovementMode {
    Walk = 0,
    Fly = 1,    // was Climb
    Climb = 2,  // was Fly
    // ...
}
```

**Verdict:** FAIL
**Evidence:** `crates/engine_data/src/enums/movement_mode.rs:N`.
**Reason:** Enum ordinals are part of the snapshot binary format. Reordering breaks every existing snapshot. Even worse: the old hash matches but data is corrupt. Schema hash bump is mandatory.

### Example 4: Mask predicate semantics change without flag change

```rust
// assets/sim/masks.sim (diff)
mask move_toward {
    require alive(agent)
    require !stunned(agent)
    require !rooted(agent)  // NEW
}
```

`.schema_hash` unchanged.

**Verdict:** FAIL
**Evidence:** `assets/sim/masks.sim:N`.
**Reason:** Mask predicates are part of the deterministic surface. Adding a require-clause changes which actions are eligible at given state. Schema hash captures `MASK_HASH`; regen required.

## Few-shot GOOD examples

### Example 1: Layout change paired with hash bump

```rust
// crates/engine/src/state/mod.rs — added hot_grudge_q8
// crates/engine/.schema_hash — value updated to new sha256
```

**Verdict:** PASS — `compile-dsl --check` produces matching hash; `cargo test --test schema_hash` passes.

### Example 2: Pure documentation change

A diff that only touches `// comments` or doc-strings doesn't affect `schema_hash` outputs. **Verdict:** PASS — no semantic change.

## Output format

VERDICT: PASS | FAIL
EVIDENCE: <file:line>[, <file:line>, ...]
REASONING: <one paragraph>
TOOLS RUN:
- <command>

If TOOLS RUN is empty → automatic FAIL.
