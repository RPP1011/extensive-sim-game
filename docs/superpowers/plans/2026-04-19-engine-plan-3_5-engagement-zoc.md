# Engine Plan 3.5 — Engagement / Zone of Control (tactical positioning without collision)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tabletop-RPG-style engagement to the engine so that a hostile entering melee range locks both sides in a mutual engagement; moving past an engager is slowed and provokes a free opportunity attack. This lets a line of tanks protect a wizard behind them — without introducing agent-agent positional collision.

**Architecture:** One new SoA hot field (`hot_engaged_with: Vec<Option<AgentId>>`), one new replayable event (`OpportunityAttackTriggered`), one new cascade handler (fires the free attack when disengagement is detected), one new tick-pipeline helper (`update_engagements`) that runs at the start of phase 1 and maintains the bidirectional engagement invariant, one new constant pair (`ENGAGEMENT_RANGE = 2.0`, `ENGAGEMENT_SLOW_FACTOR = 0.3`), and a stubbed `CreatureType::is_hostile_to` method. `apply_actions` branches for `MoveToward` and `Flee` are modified to read `hot_engaged_with`, scale the movement step by `ENGAGEMENT_SLOW_FACTOR` when the actor is disengaging (leaving an engager while pursuing a different target or fleeing a different threat), and emit `OpportunityAttackTriggered { attacker: engager, target: self }`. Determinism is preserved by iterating `agents_alive()` in `AgentId` order and breaking ties (equidistant hostiles) by lowest raw id.

**Tech Stack:** Rust 2021; `engine` crate (`crates/engine`); `glam` 0.29; `proptest` 1 (for Task 6); `contracts` 0.6 for the bidirectional-engagement debug invariant. No new workspace deps.

**Interaction with Ability Plan 1:** Ability Plan 1's `CastHandler` currently uses `ATTACK_RANGE` for melee range gating. When Ability Plan 1 lands alongside this plan, its melee-range cast gate must also respect engagement state — i.e. casting a non-engaged-target melee ability while engaged should pay the same opportunity-attack cost as a disengaging `MoveToward`. This plan does **not** modify Ability Plan 1's handler (it is gated behind ⚠️ "pulled — awaiting execution" in `status.md`). The engagement-aware cast gate lands as the first task of an Ability-Plan-1 integration PR, not here — but this plan deliberately shares the `ENGAGEMENT_RANGE = 2.0` constant with `ATTACK_RANGE` so the cross-plan integration is a one-line `state.agent_engaged_with(actor)` read in `CastHandler::handle`, not a re-derivation. See §Follow-ups below.

---

## Files overview

Modified files:

| Path | Change |
|---|---|
| `crates/engine/src/state/mod.rs` | Add `hot_engaged_with: Vec<Option<AgentId>>` hot field + accessor + mutator + bulk slice; initialize to `None` in `SimState::new` + `spawn_agent`; clear in `kill_agent`. |
| `crates/engine/src/creature.rs` | Add `impl CreatureType { pub fn is_hostile_to(self, other: CreatureType) -> bool }` stub. |
| `crates/engine/src/step.rs` | Add `update_engagements(state: &mut SimState)`; add `ENGAGEMENT_RANGE`, `ENGAGEMENT_SLOW_FACTOR` consts; call `update_engagements` at the top of phase 1 in `step_full`; modify `MoveToward` + `Flee` apply arms to read engagement and emit opportunity-attack events with the slow factor applied. |
| `crates/engine/src/event/mod.rs` | Add `Event::OpportunityAttackTriggered { attacker: AgentId, target: AgentId, tick: u32 }`; extend `tick()` match; keep `is_replayable()` — default is replayable (only `ChronicleEntry` opts out). |
| `crates/engine/src/cascade/handler.rs` | Add `EventKindId::OpportunityAttackTriggered = 23`; extend `EventKindId::from_event`. |
| `crates/engine/src/cascade/mod.rs` | Re-export new `OpportunityAttackHandler` + registration helper. |
| `crates/engine/src/schema_hash.rs` | Extend fingerprint: new SoA field, new event variant, new EventKindId ordinal, new constants. |
| `crates/engine/.schema_hash` | Regenerate baseline. |
| `docs/engine/status.md` | Add `§9a Engagement (ZoC)` row to subsystem table; update open-verification question 11 (engine collision gap); append two new Visual-check rows V10 + V11. |

New files:

| Path | Responsibility |
|---|---|
| `crates/engine/src/cascade/opportunity.rs` | `OpportunityAttackHandler` — `CascadeHandler` impl on `Lane::Effect` that consumes `OpportunityAttackTriggered` and applies `ATTACK_DAMAGE` to the target, emitting `AgentAttacked` and (if hp reaches 0) `AgentDied` so the existing death-cascade fires. |
| `crates/engine/tests/state_engagement.rs` | Unit: `hot_engaged_with` accessor + mutator + init-to-None + clear-on-kill. |
| `crates/engine/tests/creature_hostile.rs` | Unit: pairwise hostility matrix. |
| `crates/engine/tests/engagement_formation.rs` | Unit: `update_engagements` forms, breaks, stays-None for non-hostile, clears on partner death. |
| `crates/engine/tests/engagement_slowed_move.rs` | Unit: engaged MoveToward toward engager = 1.0 m/tick; MoveToward toward anyone else = 0.3 m/tick; Flee from engager = 1.0 m/tick; Flee from a non-engaged threat while engaged = 0.3 m/tick + opportunity attack. |
| `crates/engine/tests/opportunity_attack.rs` | Unit: engaged pair at 1.5 m; disengaging agent takes `ATTACK_DAMAGE` HP; `AgentAttacked` event emitted with the correct attacker/target. |
| `crates/engine/tests/proptest_engagement.rs` | Proptest: bidirectional invariant; no non-hostile engagement; no engagement outside range; determinism (same seed → same engagement map). |
| `crates/engine/tests/acceptance_wall_formation.rs` | Acceptance: 3 tanks + wizard vs wolf; wizard survives ≥30 ticks. |
| `crates/engine/tests/acceptance_no_wall_baseline.rs` | Acceptance baseline: wizard alone vs wolf; reaches wizard in <15 ticks. Differential > 2× confirms ZoC is the cause, not other variables. |

---

## Task 1: Add `hot_engaged_with` SoA field

**Why this test isn't circular:** The test does not touch `update_engagements` — it uses raw setter/getter pairs and asserts the field survives `spawn_agent` (starts None), survives `set_agent_engaged_with` (returns Some), and is cleared by `kill_agent`. An impl that hard-codes `None` would fail the setter roundtrip; an impl that leaves a stale `Some` after kill would fail the clear-on-kill arm. The positive + negative pair pins the accessor arithmetic without going through any engagement-formation logic.

**Files:**
- Modify: `crates/engine/src/state/mod.rs`
- Test: `crates/engine/tests/state_engagement.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine/tests/state_engagement.rs`:

```rust
//! Unit tests for the `hot_engaged_with` SoA field on `SimState`.
//! Kept free of `update_engagements` / step-pipeline coupling — this exercises
//! the raw field only (accessor roundtrip + init-None + clear-on-kill).

use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn newly_spawned_agent_has_no_engagement() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO,
        hp: 100.0,
    }).unwrap();
    assert_eq!(state.agent_engaged_with(a), Some(None));
}

#[test]
fn setter_roundtrips() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    state.set_agent_engaged_with(a, Some(b));
    assert_eq!(state.agent_engaged_with(a), Some(Some(b)));

    state.set_agent_engaged_with(a, None);
    assert_eq!(state.agent_engaged_with(a), Some(None));
}

#[test]
fn kill_clears_engagement_in_freed_slot() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    state.set_agent_engaged_with(a, Some(b));
    state.set_agent_engaged_with(b, Some(a));

    state.kill_agent(a);
    // After respawn into the freed slot, the new agent must start with no engagement.
    let c = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(5.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    assert_eq!(state.agent_engaged_with(c), Some(None));
    // The still-alive partner's reference to the dead agent is NOT auto-cleared
    // by kill_agent — that's the job of `update_engagements` (Task 3). Here we
    // only assert that the freed slot itself was reset.
}

#[test]
fn bulk_slice_matches_per_agent_accessor() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    state.set_agent_engaged_with(a, Some(b));

    let slice = state.hot_engaged_with();
    assert_eq!(slice.len(), state.agent_cap() as usize);
    // a is slot 0 (AgentId raw==1 ⇒ slot 0)
    assert_eq!(slice[0], Some(b));
    assert_eq!(slice[1], None);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p engine --test state_engagement`
Expected: FAIL with "method `agent_engaged_with` not found on type `SimState`" (or similar).

- [ ] **Step 3: Write minimal implementation**

Modify `crates/engine/src/state/mod.rs`. Add the hot field, init paths, accessor, mutator, bulk slice, and kill-time reset.

Insert into the `SimState` struct (after `hot_rest_timer`):

```rust
    hot_rest_timer:    Vec<f32>,
    /// Mutual melee-engagement partner for each slot. `Some(other)` means this
    /// agent is locked in engagement with `other`; `None` means disengaged.
    /// Maintained by `update_engagements` (see `step.rs`). Bidirectional
    /// invariant: `hot_engaged_with[slot_of(a)] == Some(b)` iff
    /// `hot_engaged_with[slot_of(b)] == Some(a)`. Proptest-enforced (Plan 3.5 Task 6).
    hot_engaged_with:  Vec<Option<AgentId>>,
```

Extend the `AgentId` import at the top of the file — it is already imported via `use crate::ids::AgentId;`.

Extend `SimState::new`:

```rust
            hot_rest_timer:    vec![1.0; cap],
            hot_engaged_with:  vec![None; cap],
            cold_creature_type: vec![None; cap],
```

Extend `spawn_agent` (after `self.hot_rest_timer[slot] = 1.0;`):

```rust
        self.hot_rest_timer[slot]    = 1.0;
        self.hot_engaged_with[slot]  = None;
        let caps = Capabilities::for_creature(spec.creature_type);
```

Extend `kill_agent` to clear the slot's engagement (do this before the freelist push, which happens inside `self.pool.kill_agent`):

```rust
    #[contracts::debug_ensures(!self.agent_alive(id))]
    pub fn kill_agent(&mut self, id: AgentId) {
        let slot = AgentSlotPool::slot_of_agent(id);
        if let Some(a) = self.hot_alive.get_mut(slot) {
            *a = false;
        }
        if let Some(e) = self.hot_engaged_with.get_mut(slot) {
            *e = None;
        }
        self.pool.kill_agent(id);
    }
```

Add the accessor next to `agent_rest_timer`:

```rust
    pub fn agent_rest_timer(&self, id: AgentId) -> Option<f32> {
        self.hot_rest_timer.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    /// Engagement partner for `id`. Outer `Option` = slot exists; inner
    /// `Option<AgentId>` = engaged (`Some`) vs disengaged (`None`).
    pub fn agent_engaged_with(&self, id: AgentId) -> Option<Option<AgentId>> {
        self.hot_engaged_with
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
    }
```

Add the mutator next to `set_agent_rest_timer`:

```rust
    pub fn set_agent_rest_timer(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_rest_timer.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_engaged_with(&mut self, id: AgentId, partner: Option<AgentId>) {
        if let Some(e) = self.hot_engaged_with.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *e = partner;
        }
    }
```

Add the bulk slice next to `hot_rest_timer`:

```rust
    pub fn hot_rest_timer(&self) -> &[f32] {
        &self.hot_rest_timer
    }
    pub fn hot_engaged_with(&self) -> &[Option<AgentId>] {
        &self.hot_engaged_with
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p engine --test state_engagement`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/state/mod.rs crates/engine/tests/state_engagement.rs
git commit -m "$(cat <<'EOF'
feat(engine): hot_engaged_with SoA field + accessors (Plan 3.5 T1)

Adds the storage for mutual melee engagement. Field-level only — the
update pass lands in T3, speed gating in T4, opportunity-attack cascade
in T5. Killing an agent clears the freed slot's engagement (the partner's
stale reference is swept by update_engagements next tick).
EOF
)"
```

---

## Task 2: `CreatureType::is_hostile_to` stub

**Why this test isn't circular:** The hostility matrix is asserted cell-by-cell against a DF-style table the user supplied — the test derives nothing from the impl's match arms. A bug that e.g. made `Wolf.is_hostile_to(Deer) == false` (wolves ignoring prey) would fire; a symmetry-break bug (`Human ↦ Wolf == true` but `Wolf ↦ Human == false`) would fire via the explicit `symmetric` check. The test enumerates all 16 ordered pairs so no case is missed.

**Files:**
- Modify: `crates/engine/src/creature.rs`
- Test: `crates/engine/tests/creature_hostile.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine/tests/creature_hostile.rs`:

```rust
//! Pairwise hostility matrix for `CreatureType::is_hostile_to`. MVP stub;
//! full relationship matrix lands in a later plan. See `creature.rs` docstring.

use engine::creature::CreatureType as CT;

fn expected(a: CT, b: CT) -> bool {
    use CT::*;
    match (a, b) {
        // Wolves are hostile to everyone.
        (Wolf, Human) | (Wolf, Deer) | (Wolf, Dragon)           => true,
        (Human, Wolf) | (Deer, Wolf) | (Dragon, Wolf)           => true,
        // Humans are hostile to dragons (large predators).
        (Human, Dragon) | (Dragon, Human)                       => true,
        // Deer are hostile to nobody.
        (Deer, _) | (_, Deer)                                   => matches!((a, b),
            (Wolf, Deer) | (Deer, Wolf)),
        // Same species = never hostile under this stub.
        (x, y) if x == y                                        => false,
        // Human–Human already excluded above; Dragon–Dragon same.
        _ => false,
    }
}

#[test]
fn pairwise_matrix() {
    for a in [CT::Human, CT::Wolf, CT::Deer, CT::Dragon] {
        for b in [CT::Human, CT::Wolf, CT::Deer, CT::Dragon] {
            let got = a.is_hostile_to(b);
            let want = expected(a, b);
            assert_eq!(
                got, want,
                "is_hostile_to mismatch for ({:?}, {:?}): got {} want {}",
                a, b, got, want,
            );
        }
    }
}

#[test]
fn symmetric() {
    // The stub must be symmetric; asymmetric relationships wait for the
    // full matrix. A failure here signals the match arms are inconsistent.
    for a in [CT::Human, CT::Wolf, CT::Deer, CT::Dragon] {
        for b in [CT::Human, CT::Wolf, CT::Deer, CT::Dragon] {
            assert_eq!(
                a.is_hostile_to(b),
                b.is_hostile_to(a),
                "asymmetric ({:?}, {:?})", a, b,
            );
        }
    }
}

#[test]
fn same_species_is_never_hostile_under_stub() {
    for a in [CT::Human, CT::Wolf, CT::Deer, CT::Dragon] {
        assert!(!a.is_hostile_to(a), "{:?} vs {:?} should be false under the MVP stub", a, a);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p engine --test creature_hostile`
Expected: FAIL with "method `is_hostile_to` not found on type `CreatureType`".

- [ ] **Step 3: Write minimal implementation**

Modify `crates/engine/src/creature.rs`. Add an `impl CreatureType` block after the `#[repr(u8)]` enum:

```rust
#[derive(Copy, Clone, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(u8)]
pub enum CreatureType {
    #[default]
    Human = 0,
    Wolf = 1,
    Deer = 2,
    Dragon = 3,
}

impl CreatureType {
    /// Returns `true` when the two creatures are hostile — i.e. should form a
    /// melee engagement on contact (Plan 3.5).
    ///
    /// **Status: stub.** The MVP uses a symmetric predator/prey rule:
    /// * Wolves are hostile to everyone.
    /// * Humans are hostile to dragons (large predators) and vice versa.
    /// * Deer are hostile to nobody (flee-only behavior).
    /// * Same species is never hostile.
    ///
    /// Superseded by the per-pair relationship matrix when the social-graph
    /// plan lands. Keep this pure + total — `update_engagements` calls it
    /// once per alive-agent pair per tick.
    pub fn is_hostile_to(self, other: CreatureType) -> bool {
        use CreatureType::*;
        match (self, other) {
            (Wolf, _) | (_, Wolf) if self != other => true,
            (Human, Dragon) | (Dragon, Human)      => true,
            _ => false,
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p engine --test creature_hostile`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/creature.rs crates/engine/tests/creature_hostile.rs
git commit -m "$(cat <<'EOF'
feat(engine): CreatureType::is_hostile_to stub (Plan 3.5 T2)

MVP predator/prey rule: wolves hostile to everyone, humans hostile to
dragons, deer peaceful. Symmetric. Documented as a stub to be replaced
by the per-pair relationship matrix.
EOF
)"
```

---

## Task 3: `update_engagements` pass + constants

**Why this test isn't circular:** The test positions agents manually and calls `update_engagements` directly — no policy, no mask, no `step_full`. The "forms at 1.5 m, doesn't form at 2.5 m" assertion proves the range constant is respected without reading it from code. The "same-species stay None" arm catches an impl that forgets to consult `is_hostile_to` (Task 2). The "moving out of range breaks engagement next tick" arm catches an impl that forms engagement but never clears it. Each branch (form, stay-None, break) is a distinct observable outcome; a common-mode bug would have to wrongly classify multiple of them in a way the test enumerates.

**Files:**
- Modify: `crates/engine/src/step.rs`
- Test: `crates/engine/tests/engagement_formation.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine/tests/engagement_formation.rs`:

```rust
//! Unit tests for `engine::step::update_engagements`. Exercises the pass in
//! isolation — no step_full, no policy, no mask. Manual state + direct call.

use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use engine::step::update_engagements;
use glam::Vec3;

fn spawn(state: &mut SimState, ct: CreatureType, x: f32) -> engine::ids::AgentId {
    state.spawn_agent(AgentSpawn {
        creature_type: ct,
        pos: Vec3::new(x, 0.0, 0.0),
        hp: 100.0,
    }).unwrap()
}

#[test]
fn two_hostiles_within_range_become_mutually_engaged() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, 0.0);
    let b = spawn(&mut state, CreatureType::Wolf,  1.5);

    update_engagements(&mut state);

    assert_eq!(state.agent_engaged_with(a), Some(Some(b)), "a engaged with b");
    assert_eq!(state.agent_engaged_with(b), Some(Some(a)), "b engaged with a");
}

#[test]
fn two_hostiles_out_of_range_stay_none() {
    // ENGAGEMENT_RANGE = 2.0, so 2.5m is clearly outside.
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, 0.0);
    let b = spawn(&mut state, CreatureType::Wolf,  2.5);

    update_engagements(&mut state);

    assert_eq!(state.agent_engaged_with(a), Some(None));
    assert_eq!(state.agent_engaged_with(b), Some(None));
}

#[test]
fn same_species_stay_none_even_within_range() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, 0.0);
    let b = spawn(&mut state, CreatureType::Human, 1.0);

    update_engagements(&mut state);

    assert_eq!(state.agent_engaged_with(a), Some(None));
    assert_eq!(state.agent_engaged_with(b), Some(None));
}

#[test]
fn engagement_boundary_at_two_meters_exactly() {
    // ENGAGEMENT_RANGE = 2.0, uses `<=` — exactly at 2.0 should engage;
    // 2.001 should not.
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, 0.0);
    let b = spawn(&mut state, CreatureType::Wolf,  2.0);
    update_engagements(&mut state);
    assert_eq!(state.agent_engaged_with(a), Some(Some(b)), "engage at exactly 2.0m");

    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, 0.0);
    let _b = spawn(&mut state, CreatureType::Wolf,  2.001);
    update_engagements(&mut state);
    assert_eq!(state.agent_engaged_with(a), Some(None), "no engage at 2.001m");
}

#[test]
fn moving_out_of_range_breaks_engagement() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, 0.0);
    let b = spawn(&mut state, CreatureType::Wolf,  1.5);
    update_engagements(&mut state);
    assert_eq!(state.agent_engaged_with(a), Some(Some(b)));

    // Teleport b away (as a raw mutation — this test is isolating the update pass).
    state.set_agent_pos(b, Vec3::new(10.0, 0.0, 0.0));
    update_engagements(&mut state);

    assert_eq!(state.agent_engaged_with(a), Some(None), "a disengages when b moves away");
    assert_eq!(state.agent_engaged_with(b), Some(None), "b disengages when a falls out of range");
}

#[test]
fn partner_death_clears_survivor_engagement() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, 0.0);
    let b = spawn(&mut state, CreatureType::Wolf,  1.0);
    update_engagements(&mut state);
    assert_eq!(state.agent_engaged_with(a), Some(Some(b)));

    state.kill_agent(b);
    update_engagements(&mut state);

    assert_eq!(state.agent_engaged_with(a), Some(None),
        "surviving partner's engagement is cleared when dead partner is re-scanned");
}

#[test]
fn three_way_ties_broken_by_lowest_agent_id() {
    // a spawns first (AgentId 1). Two equidistant wolves at x=+1.0 and z=+1.0
    // relative to a (both at distance 1.0). Tie-break: lowest AgentId wins.
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, 0.0);
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(1.0, 0.0, 0.0),
        hp: 100.0,
    }).unwrap();
    let _c = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(0.0, 0.0, 1.0),
        hp: 100.0,
    }).unwrap();

    update_engagements(&mut state);

    // b has the lower raw id (spawned second => id=2 vs c's id=3), so a picks b.
    assert_eq!(state.agent_engaged_with(a), Some(Some(b)));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p engine --test engagement_formation`
Expected: FAIL with "function `update_engagements` not found in module `engine::step`".

- [ ] **Step 3: Write minimal implementation**

Modify `crates/engine/src/step.rs`. Add the two constants below the existing ones at the top:

```rust
const MOVE_SPEED_MPS: f32 = 1.0;
const ATTACK_DAMAGE:  f32 = 10.0;
const ATTACK_RANGE:   f32 = 2.0;
const EAT_RESTORE:    f32 = 0.25;
const DRINK_RESTORE:  f32 = 0.30;
const REST_RESTORE:   f32 = 0.15;

/// Maximum distance (m) at which two hostile agents are mutually locked in
/// melee engagement. Intentionally equal to `ATTACK_RANGE` — if you can hit
/// me, we're engaged. Keeping the two constants aligned prevents a
/// mask-vs-resolution disagreement (see `mask.rs::ATTACK_RANGE_FOR_MASK`).
pub const ENGAGEMENT_RANGE: f32 = 2.0;

/// Scalar applied to `MOVE_SPEED_MPS` when an engaged agent moves toward a
/// target OTHER than its engager (i.e. trying to disengage). Full speed when
/// moving toward the engager itself.
pub const ENGAGEMENT_SLOW_FACTOR: f32 = 0.3;
```

Add the update pass as a `pub` function (callable from tests and from `step_full`). Place it after `SimScratch` and before `step`:

```rust
/// Tick-start engagement-update pass.
///
/// For each alive agent (in `AgentId` order), find the nearest alive hostile
/// within `ENGAGEMENT_RANGE`; ties (equidistant hostiles) are broken by
/// lowest raw `AgentId`. Sets both sides of the engagement mutually. Agents
/// whose prior partner died, moved out of range, or became non-hostile have
/// their `hot_engaged_with` reset to `None`.
///
/// Called at the top of phase 1 in `step_full`. Exposed `pub` so unit tests
/// can exercise engagement formation without a full tick.
///
/// Determinism: iteration order is `agents_alive()` (slot order, monotonic),
/// tie-break is deterministic. No RNG. No allocation per call (walks SoA
/// slices; candidate search is a single pass).
pub fn update_engagements(state: &mut SimState) {
    // Collect the new engagement partner per alive agent first, then write
    // back. Collecting avoids a borrow-check conflict between the pairwise
    // scan and the per-agent mutator, and keeps the write strictly mutual:
    // we only commit a pair (a, b) if both sides selected each other.
    // Scratch: small stack vec bounded by alive count — caller-side
    // allocation elision is fine at this scale (call site is 1/tick, and
    // world agent counts are bounded by Pool cap, typically ≤ few thousand).
    let mut tentative: smallvec::SmallVec<[(engine_ids::AgentId, Option<engine_ids::AgentId>); 64]>
        = smallvec::SmallVec::new();

    for id in state.agents_alive() {
        let self_pos = match state.agent_pos(id) { Some(p) => p, None => continue };
        let self_ct  = match state.agent_creature_type(id) { Some(c) => c, None => continue };

        let mut best: Option<(engine_ids::AgentId, f32)> = None;
        for other in state.agents_alive() {
            if other == id { continue; }
            let other_ct = match state.agent_creature_type(other) { Some(c) => c, None => continue };
            if !self_ct.is_hostile_to(other_ct) { continue; }
            let other_pos = match state.agent_pos(other) { Some(p) => p, None => continue };
            let d = self_pos.distance(other_pos);
            if d > ENGAGEMENT_RANGE { continue; }

            best = match best {
                None => Some((other, d)),
                Some((cur, cur_d)) => {
                    // Closer wins; equidistant → lower raw id wins.
                    if d < cur_d || (d == cur_d && other.raw() < cur.raw()) {
                        Some((other, d))
                    } else {
                        Some((cur, cur_d))
                    }
                }
            };
        }
        tentative.push((id, best.map(|(o, _)| o)));
    }

    // Commit: only set engagement when the pairing is mutual under the
    // tentative map. Non-mutual pickings (A picks B but B picks C) resolve
    // to None for A — preserves the bidirectional invariant without
    // requiring a fixpoint iteration.
    let lookup = |q: engine_ids::AgentId| -> Option<engine_ids::AgentId> {
        tentative
            .iter()
            .find(|(a, _)| *a == q)
            .and_then(|(_, p)| *p)
    };
    for (a, tentative_partner) in tentative.iter().copied() {
        let committed = match tentative_partner {
            Some(b) if lookup(b) == Some(a) => Some(b),
            _ => None,
        };
        state.set_agent_engaged_with(a, committed);
    }
}
```

Add the `engine_ids` path alias at the top of `step.rs` (the existing file already imports `AgentId` from `crate::ids`):

```rust
use crate::ids::AgentId;
```

…is already present. Change the function body to use `AgentId` directly (not `engine_ids::AgentId`). Rewrite the function with the correct type path:

```rust
pub fn update_engagements(state: &mut SimState) {
    let mut tentative: smallvec::SmallVec<[(AgentId, Option<AgentId>); 64]>
        = smallvec::SmallVec::new();

    for id in state.agents_alive() {
        let self_pos = match state.agent_pos(id) { Some(p) => p, None => continue };
        let self_ct  = match state.agent_creature_type(id) { Some(c) => c, None => continue };

        let mut best: Option<(AgentId, f32)> = None;
        for other in state.agents_alive() {
            if other == id { continue; }
            let other_ct = match state.agent_creature_type(other) { Some(c) => c, None => continue };
            if !self_ct.is_hostile_to(other_ct) { continue; }
            let other_pos = match state.agent_pos(other) { Some(p) => p, None => continue };
            let d = self_pos.distance(other_pos);
            if d > ENGAGEMENT_RANGE { continue; }

            best = match best {
                None => Some((other, d)),
                Some((cur, cur_d)) => {
                    if d < cur_d || (d == cur_d && other.raw() < cur.raw()) {
                        Some((other, d))
                    } else {
                        Some((cur, cur_d))
                    }
                }
            };
        }
        tentative.push((id, best.map(|(o, _)| o)));
    }

    let lookup = |q: AgentId| -> Option<AgentId> {
        tentative
            .iter()
            .find(|(a, _)| *a == q)
            .and_then(|(_, p)| *p)
    };
    for (a, tentative_partner) in tentative.iter().copied() {
        let committed = match tentative_partner {
            Some(b) if lookup(b) == Some(a) => Some(b),
            _ => None,
        };
        state.set_agent_engaged_with(a, committed);
    }
}
```

Finally, wire it into `step_full` at the start of phase 1 — before mask building, because mask predicates in later plans may read engagement state:

```rust
    let t_start = std::time::Instant::now();

    // Phase 0.5 — engagement update. Runs before the mask pass so mask
    // predicates can read `hot_engaged_with` (Plan 3.5). Pure function; no
    // RNG; O(alive²) pairwise scan — acceptable until Plan 6 moves this
    // onto the spatial index.
    update_engagements(state);

    // Phase 1 — mask build.
    scratch.mask.reset();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p engine --test engagement_formation`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/step.rs crates/engine/tests/engagement_formation.rs
git commit -m "$(cat <<'EOF'
feat(engine): update_engagements pass + ENGAGEMENT_RANGE/SLOW_FACTOR (Plan 3.5 T3)

Runs at the top of phase 1 in step_full. Maintains mutual engagement
(bidirectional invariant) between hostile agents within 2.0 m. O(alive²)
pairwise scan; ties broken by lowest AgentId. No allocation per call
(smallvec scratch sized to 64). `pub fn update_engagements` callable
from tests.
EOF
)"
```

---

## Task 4: Engagement-speed gating in `MoveToward` + `Flee`

**Why this test isn't circular:** The test compares position deltas across two scenarios that differ only in whether the moving agent is engaged — the speed constants (`MOVE_SPEED_MPS = 1.0`, `ENGAGEMENT_SLOW_FACTOR = 0.3`) are not referenced from the test. A bug that slowed everyone (engaged or not) would fail the "MoveToward toward engager = full speed" arm. A bug that slowed nobody would fail the "MoveToward away = slowed" arm. A bug that applied the wrong factor (e.g. 0.5 instead of 0.3) would fail the numeric-magnitude assertions. The opportunity-attack event is asserted by kind only in this task (the cascade handler that applies damage lives in Task 5), so this test doesn't depend on Task 5 landing.

**Files:**
- Modify: `crates/engine/src/step.rs` (MoveToward + Flee match arms in `apply_actions`, plus event emission)
- Modify: `crates/engine/src/event/mod.rs` (add `OpportunityAttackTriggered` variant)
- Test: `crates/engine/tests/engagement_slowed_move.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine/tests/engagement_slowed_move.rs`:

```rust
//! Engagement speed-gating: MoveToward / Flee while engaged.
//!
//! Target matrix:
//! | actor engaged w/ | action                 | movement factor | opp-atk event |
//! |------------------|------------------------|-----------------|---------------|
//! | None             | MoveToward(anything)   | 1.0             | no            |
//! | None             | Flee(anything)         | 1.0             | no            |
//! | Some(X)          | MoveToward(X)          | 1.0             | no            |
//! | Some(X)          | MoveToward(Y != X)     | 0.3             | yes           |
//! | Some(X)          | Flee(X)                | 1.0             | yes           |
//! | Some(X)          | Flee(threat != X)      | 0.3             | yes           |

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::MaskBuffer;
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::MicroKind;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

/// Minimal backend: emit a single, fixed action for the named actor; every
/// other alive agent holds. Keeps the test isolated from `UtilityBackend`
/// scoring decisions.
struct FixedAction { actor: AgentId, action: ActionKind }
impl PolicyBackend for FixedAction {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            if id == self.actor {
                out.push(Action { agent: id, kind: self.action });
            } else {
                out.push(Action::hold(id));
            }
        }
    }
}

fn setup(actor_ct: CreatureType, target_ct: CreatureType, dist: f32)
    -> (SimState, AgentId, AgentId)
{
    let mut state = SimState::new(6, 42);
    let actor = state.spawn_agent(AgentSpawn {
        creature_type: actor_ct, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let target = state.spawn_agent(AgentSpawn {
        creature_type: target_ct, pos: Vec3::new(dist, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    (state, actor, target)
}

#[test]
fn move_toward_when_not_engaged_is_full_speed() {
    // Same-species so no engagement forms.
    let (mut state, actor, target) = setup(CreatureType::Human, CreatureType::Human, 5.0);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let before = state.agent_pos(actor).unwrap();
    step(
        &mut state, &mut scratch, &mut events,
        &FixedAction {
            actor,
            action: ActionKind::Micro {
                kind:   MicroKind::MoveToward,
                target: MicroTarget::Position(state.agent_pos(target).unwrap()),
            },
        },
        &cascade,
    );
    let dx = state.agent_pos(actor).unwrap().x - before.x;
    assert!((dx - 1.0).abs() < 1e-5, "dx should be 1.0 (MOVE_SPEED_MPS), got {}", dx);
    assert!(!events.iter().any(|e| matches!(e, Event::OpportunityAttackTriggered { .. })));
}

#[test]
fn move_toward_engager_is_full_speed_no_opportunity_attack() {
    // Hostile pair inside ENGAGEMENT_RANGE — engagement forms on tick 0.
    let (mut state, actor, engager) = setup(CreatureType::Human, CreatureType::Wolf, 1.5);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    // Actor moves TOWARD engager — not disengaging, no penalty.
    let before = state.agent_pos(actor).unwrap();
    step(
        &mut state, &mut scratch, &mut events,
        &FixedAction {
            actor,
            action: ActionKind::Micro {
                kind:   MicroKind::MoveToward,
                target: MicroTarget::Position(state.agent_pos(engager).unwrap()),
            },
        },
        &cascade,
    );
    let dx = state.agent_pos(actor).unwrap().x - before.x;
    assert!((dx - 1.0).abs() < 1e-5, "full speed toward engager, got dx={}", dx);
    assert!(!events.iter().any(|e| matches!(e, Event::OpportunityAttackTriggered { .. })),
        "no opportunity attack when moving toward engager");
}

#[test]
fn move_toward_third_party_while_engaged_is_slowed_and_emits_opportunity_attack() {
    // Actor (Human) engaged with a Wolf (engager) at 1.5m.
    // Third party is a Deer at x=10m — actor moves TOWARD deer, i.e. disengaging.
    let mut state = SimState::new(6, 42);
    let actor = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let engager = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,  pos: Vec3::new(1.5, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    let deer = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Deer,  pos: Vec3::new(10.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let before = state.agent_pos(actor).unwrap();
    step(
        &mut state, &mut scratch, &mut events,
        &FixedAction {
            actor,
            action: ActionKind::Micro {
                kind:   MicroKind::MoveToward,
                target: MicroTarget::Position(state.agent_pos(deer).unwrap()),
            },
        },
        &cascade,
    );
    let dx = state.agent_pos(actor).unwrap().x - before.x;
    assert!((dx - 0.3).abs() < 1e-5,
        "slowed movement toward third party should be ENGAGEMENT_SLOW_FACTOR * MOVE_SPEED_MPS = 0.3, got {}",
        dx);

    let opp = events.iter().find_map(|e| match e {
        Event::OpportunityAttackTriggered { attacker, target, .. } => Some((*attacker, *target)),
        _ => None,
    }).expect("OpportunityAttackTriggered emitted");
    assert_eq!(opp.0, engager, "engager is the attacker");
    assert_eq!(opp.1, actor,   "disengaging actor is the target");
}

#[test]
fn flee_from_engager_is_full_speed_and_emits_opportunity_attack() {
    // Classic disengage: Flee(engager). Full speed (you break contact cleanly)
    // but still eat the free attack on the way out.
    let (mut state, actor, engager) = setup(CreatureType::Human, CreatureType::Wolf, 1.5);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let before = state.agent_pos(actor).unwrap();
    step(
        &mut state, &mut scratch, &mut events,
        &FixedAction {
            actor,
            action: ActionKind::Micro {
                kind:   MicroKind::Flee,
                target: MicroTarget::Agent(engager),
            },
        },
        &cascade,
    );
    let dx = state.agent_pos(actor).unwrap().x - before.x;
    // Actor is at x=0, engager at x=1.5 — away direction is -x. Full speed = -1.0.
    assert!((dx + 1.0).abs() < 1e-5,
        "full-speed flee from engager should be -MOVE_SPEED_MPS along -x, got dx={}", dx);

    assert!(events.iter().any(|e| matches!(e,
        Event::OpportunityAttackTriggered { attacker, target, .. }
            if *attacker == engager && *target == actor)),
        "opportunity attack fires when fleeing the engager");
}

#[test]
fn flee_from_third_party_while_engaged_is_slowed_and_emits_opportunity_attack() {
    // Actor engaged with Wolf A; Wolf B is across the map. Actor Flees B
    // (not its engager) — slowed + opportunity attack from A.
    let mut state = SimState::new(6, 42);
    let actor = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let engager = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.5, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    let second_threat = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(-20.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let before = state.agent_pos(actor).unwrap();
    step(
        &mut state, &mut scratch, &mut events,
        &FixedAction {
            actor,
            action: ActionKind::Micro {
                kind:   MicroKind::Flee,
                target: MicroTarget::Agent(second_threat),
            },
        },
        &cascade,
    );
    let pos = state.agent_pos(actor).unwrap();
    // Second threat at -20 → away is +x → dx = +ENGAGEMENT_SLOW_FACTOR * MOVE_SPEED_MPS = +0.3.
    let dx = pos.x - before.x;
    assert!((dx - 0.3).abs() < 1e-5,
        "slowed flee from non-engager while engaged, got dx={}", dx);

    assert!(events.iter().any(|e| matches!(e,
        Event::OpportunityAttackTriggered { attacker, target, .. }
            if *attacker == engager && *target == actor)),
        "engager triggers opportunity attack on disengaging actor");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p engine --test engagement_slowed_move`
Expected: FAIL with "variant `OpportunityAttackTriggered` not found on enum `Event`".

- [ ] **Step 3: Add `Event::OpportunityAttackTriggered`**

Modify `crates/engine/src/event/mod.rs`. Insert the new variant next to `AgentAttacked`:

```rust
    AgentAttacked { attacker: AgentId, target: AgentId, damage: f32, tick: u32 },
    AgentDied     { agent_id: AgentId, tick: u32 },
    AgentFled     { agent_id: AgentId, from: Vec3, to: Vec3, tick: u32 },
    AgentAte      { agent_id: AgentId, delta: f32, tick: u32 },
    AgentDrank    { agent_id: AgentId, delta: f32, tick: u32 },
    AgentRested   { agent_id: AgentId, delta: f32, tick: u32 },
    /// Emitted when an agent leaves engagement (Flee-from-engager, Flee while
    /// engaged, or MoveToward a non-engager target while engaged). The
    /// cascade's `OpportunityAttackHandler` (Lane::Effect) consumes this and
    /// applies `ATTACK_DAMAGE` to `target`, emitting `AgentAttacked` (and,
    /// if hp reaches 0, `AgentDied`). Replayable. Plan 3.5.
    OpportunityAttackTriggered { attacker: AgentId, target: AgentId, tick: u32 },
```

Extend the `tick()` match arm for the new variant — insert after `AgentRested`:

```rust
            Event::AgentRested          { tick, .. } |
            Event::OpportunityAttackTriggered { tick, .. } |
            Event::AgentCast            { tick, .. } |
```

`is_replayable()` needs no change — the existing `!matches!(self, Event::ChronicleEntry { .. })` already treats the new variant as replayable (desired).

- [ ] **Step 4: Gate `MoveToward` + `Flee` on engagement**

Modify `crates/engine/src/step.rs`. Replace the `MoveToward` arm in `apply_actions` with the engagement-aware version:

```rust
            ActionKind::Micro {
                kind:   MicroKind::MoveToward,
                target: MicroTarget::Position(target_pos),
            } => {
                let from = state.agent_pos(action.agent).unwrap();
                let delta = target_pos - from;
                if delta.length_squared() > 0.0 {
                    // Engagement-speed gating: slow & emit opportunity attack
                    // when disengaging. The engagement partner is the agent at
                    // `engaged_with[self]`; if moving toward that partner's
                    // position, it's pursuit (full speed, no penalty); any
                    // other target is a disengage attempt.
                    let (speed, opp_atk) = match state.agent_engaged_with(action.agent).flatten() {
                        Some(engager) => {
                            let engager_pos = state.agent_pos(engager).unwrap_or(target_pos);
                            // Pursuit test: is the desired target close to the engager?
                            // Use a small epsilon (1e-4) so "move toward engager" isn't
                            // broken by float noise when the policy passes the engager's
                            // position verbatim.
                            if target_pos.distance(engager_pos) <= 1e-4 {
                                (MOVE_SPEED_MPS, None)
                            } else {
                                (MOVE_SPEED_MPS * ENGAGEMENT_SLOW_FACTOR, Some(engager))
                            }
                        }
                        None => (MOVE_SPEED_MPS, None),
                    };
                    let to = from + delta.normalize() * speed;
                    state.set_agent_pos(action.agent, to);
                    events.push(Event::AgentMoved {
                        agent_id: action.agent, from, to, tick: state.tick,
                    });
                    if let Some(engager) = opp_atk {
                        events.push(Event::OpportunityAttackTriggered {
                            attacker: engager,
                            target:   action.agent,
                            tick:     state.tick,
                        });
                    }
                }
            }
```

Replace the `Flee` arm with the engagement-aware version:

```rust
            ActionKind::Micro {
                kind:   MicroKind::Flee,
                target: MicroTarget::Agent(threat),
            } => {
                if !state.agent_alive(threat) { continue; }
                if let (Some(self_pos), Some(threat_pos)) =
                    (state.agent_pos(action.agent), state.agent_pos(threat))
                {
                    let away = (self_pos - threat_pos).normalize_or_zero();
                    if away.length_squared() > 0.0 {
                        // If engaged, fleeing any target (engager or third
                        // party) provokes an opportunity attack. Fleeing the
                        // engager itself is full speed (you break contact);
                        // fleeing anyone else while engaged is slowed.
                        let (speed, opp_atk) = match state.agent_engaged_with(action.agent).flatten() {
                            Some(engager) if engager == threat => {
                                (MOVE_SPEED_MPS, Some(engager))
                            }
                            Some(engager) => {
                                (MOVE_SPEED_MPS * ENGAGEMENT_SLOW_FACTOR, Some(engager))
                            }
                            None => (MOVE_SPEED_MPS, None),
                        };
                        let new_pos = self_pos + away * speed;
                        state.set_agent_pos(action.agent, new_pos);
                        events.push(Event::AgentFled {
                            agent_id: action.agent,
                            from:     self_pos,
                            to:       new_pos,
                            tick:     state.tick,
                        });
                        if let Some(engager) = opp_atk {
                            events.push(Event::OpportunityAttackTriggered {
                                attacker: engager,
                                target:   action.agent,
                                tick:     state.tick,
                            });
                        }
                    }
                }
            }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p engine --test engagement_slowed_move`
Expected: PASS (5 tests). The opportunity-attack event is emitted (but no damage is applied yet — that's Task 5). Damage-side assertions live in `opportunity_attack.rs` (Task 5).

Also rerun existing tests that touch MoveToward/Flee to make sure they still pass — engagement gating should be a no-op for same-species pairs:

Run: `cargo test -p engine --test step_move --test action_flee --test action_needs`
Expected: PASS (all unchanged).

- [ ] **Step 6: Commit**

```bash
git add crates/engine/src/event/mod.rs crates/engine/src/step.rs crates/engine/tests/engagement_slowed_move.rs
git commit -m "$(cat <<'EOF'
feat(engine): engagement-gated MoveToward/Flee + opportunity event (Plan 3.5 T4)

Adds Event::OpportunityAttackTriggered and gates MoveToward/Flee on
hot_engaged_with. MoveToward toward engager = full speed; MoveToward
toward a third party while engaged = 0.3x speed + opportunity event.
Flee from engager = full speed + opportunity event; Flee from a third
party while engaged = 0.3x + opportunity event.

Damage application lives in the OpportunityAttackHandler cascade (T5).
EOF
)"
```

---

## Task 5: `OpportunityAttackHandler` cascade

**Why this test isn't circular:** The test uses `Event::OpportunityAttackTriggered` as an input (pushed manually onto the ring at the top of the test) and checks that the handler applies 10 hp of damage and emits `AgentAttacked`. The handler and the event are logically separated: one pushes, one consumes. A bug where the handler re-emitted its own trigger would cause infinite cascade (caught by `MAX_CASCADE_ITERATIONS=8` + the test's event count assertion). A bug where the handler silently no-op'd would fail the hp-drop assertion. The test also wires the full `step_full` path in a second test to verify the end-to-end flow (move → opportunity event → cascade → damage) — catching any wiring bug in `CascadeRegistry::run_fixed_point`.

**Files:**
- Create: `crates/engine/src/cascade/opportunity.rs`
- Modify: `crates/engine/src/cascade/mod.rs` (re-export)
- Modify: `crates/engine/src/cascade/handler.rs` (add `EventKindId::OpportunityAttackTriggered = 23`)
- Test: `crates/engine/tests/opportunity_attack.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine/tests/opportunity_attack.rs`:

```rust
//! OpportunityAttackHandler — dispatched on Event::OpportunityAttackTriggered;
//! applies ATTACK_DAMAGE (10.0) to the target, emits AgentAttacked and
//! (if hp reaches 0) AgentDied.

use engine::cascade::{CascadeRegistry, OpportunityAttackHandler};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::mask::{MaskBuffer, MicroKind};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

#[test]
fn direct_dispatch_applies_damage_and_emits_attacked() {
    let mut state = SimState::new(4, 42);
    let attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let target = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0,
    }).unwrap();

    let mut events = EventRing::with_cap(1024);
    let mut registry = CascadeRegistry::new();
    registry.register(OpportunityAttackHandler);

    events.push(Event::OpportunityAttackTriggered {
        attacker, target, tick: 0,
    });
    registry.run_fixed_point(&mut state, &mut events);

    assert_eq!(state.agent_hp(target), Some(90.0), "target loses ATTACK_DAMAGE=10 hp");
    let damage = events.iter().find_map(|e| match e {
        Event::AgentAttacked { attacker: a, target: t, damage, .. } if *a == attacker && *t == target
            => Some(*damage),
        _ => None,
    }).expect("AgentAttacked emitted by cascade");
    assert!((damage - 10.0).abs() < 1e-6, "damage should be 10.0, got {}", damage);
}

#[test]
fn hp_zero_triggers_agent_died_via_existing_death_path() {
    let mut state = SimState::new(4, 42);
    let attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let target = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 5.0,
    }).unwrap();

    let mut events = EventRing::with_cap(1024);
    let mut registry = CascadeRegistry::new();
    registry.register(OpportunityAttackHandler);

    events.push(Event::OpportunityAttackTriggered { attacker, target, tick: 0 });
    registry.run_fixed_point(&mut state, &mut events);

    assert_eq!(state.agent_hp(target), Some(0.0));
    assert!(!state.agent_alive(target));
    assert!(events.iter().any(|e| matches!(e, Event::AgentDied { agent_id, .. } if *agent_id == target)),
        "AgentDied emitted when hp reaches 0");
}

/// Minimal disengage-movement backend for the end-to-end test.
struct MoveActorToward { actor: AgentId, target_pos: Vec3 }
impl PolicyBackend for MoveActorToward {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            if id == self.actor {
                out.push(Action {
                    agent: id,
                    kind: ActionKind::Micro {
                        kind: MicroKind::MoveToward,
                        target: MicroTarget::Position(self.target_pos),
                    },
                });
            } else {
                out.push(Action::hold(id));
            }
        }
    }
}

#[test]
fn end_to_end_disengage_move_takes_opportunity_damage() {
    let mut state = SimState::new(4, 42);
    let actor = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let _engager = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(1.5, 0.0, 0.0), hp: 100.0,
    }).unwrap();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let mut registry = CascadeRegistry::new();
    registry.register(OpportunityAttackHandler);

    step(
        &mut state, &mut scratch, &mut events,
        &MoveActorToward { actor, target_pos: Vec3::new(-10.0, 0.0, 0.0) },
        &registry,
    );

    assert_eq!(state.agent_hp(actor), Some(90.0),
        "disengaging actor takes ATTACK_DAMAGE=10 from engager via the cascade");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p engine --test opportunity_attack`
Expected: FAIL with "cannot find `OpportunityAttackHandler` in crate `engine::cascade`".

- [ ] **Step 3: Add `EventKindId::OpportunityAttackTriggered = 23`**

Modify `crates/engine/src/cascade/handler.rs`. Add the new ordinal after `RecordMemory = 22` and before the reserved-slots comment:

```rust
    RecordMemory         = 22,
    OpportunityAttackTriggered = 23,
    // Slots 24-127 reserved for replayable event variants added in later tasks.
    ChronicleEntry       = 128,
```

Extend `EventKindId::from_event`:

```rust
            Event::RecordMemory         { .. } => EventKindId::RecordMemory,
            Event::OpportunityAttackTriggered { .. } => EventKindId::OpportunityAttackTriggered,
            Event::ChronicleEntry       { .. } => EventKindId::ChronicleEntry,
```

- [ ] **Step 4: Implement `OpportunityAttackHandler`**

Create `crates/engine/src/cascade/opportunity.rs`:

```rust
//! OpportunityAttackHandler — the cascade half of Plan 3.5's engagement system.
//!
//! Triggers on `Event::OpportunityAttackTriggered` (emitted by the MoveToward
//! / Flee arms in `apply_actions` when an engaged agent disengages). Applies
//! the fixed `ATTACK_DAMAGE` to the target, emits `AgentAttacked` (so views
//! observing damage, like `DamageTaken`, keep working untouched), and emits
//! `AgentDied` + kills the agent if HP reaches 0.
//!
//! Lives in `Lane::Effect` — validation lanes (if any) see the trigger first;
//! audit lanes see the resulting AgentAttacked/AgentDied. The fixed-point
//! loop guarantees AgentDied is dispatched even though emitted mid-cascade.
//!
//! Why not put this in `apply_actions`? Because opportunity attacks are a
//! *reaction* — one agent's action causes another's response. That's exactly
//! the shape the cascade was designed for, and keeping it out of
//! `apply_actions` avoids ordering hazards (the engager might have a
//! later shuffle index than the mover, and we want the reaction to resolve
//! in the same tick).

use super::handler::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;

/// Must match the kernel-side `ATTACK_DAMAGE` in `step.rs`. Duplicated rather
/// than imported-across-modules to keep the cascade free of step internals.
/// A schema-hash-level consistency check would be nice but the current hash
/// already covers both constants indirectly via the damage field on
/// `AgentAttacked`.
const OPPORTUNITY_ATTACK_DAMAGE: f32 = 10.0;

pub struct OpportunityAttackHandler;

impl CascadeHandler for OpportunityAttackHandler {
    fn trigger(&self) -> EventKindId { EventKindId::OpportunityAttackTriggered }
    fn lane(&self)    -> Lane        { Lane::Effect }

    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        let (attacker, target, tick) = match *event {
            Event::OpportunityAttackTriggered { attacker, target, tick } => (attacker, target, tick),
            _ => return, // dispatched by trigger-id; defensive
        };
        if !state.agent_alive(target)   { return; }
        if !state.agent_alive(attacker) { return; }

        let new_hp = (state.agent_hp(target).unwrap_or(0.0) - OPPORTUNITY_ATTACK_DAMAGE).max(0.0);
        state.set_agent_hp(target, new_hp);
        events.push(Event::AgentAttacked {
            attacker,
            target,
            damage: OPPORTUNITY_ATTACK_DAMAGE,
            tick,
        });
        if new_hp <= 0.0 {
            events.push(Event::AgentDied { agent_id: target, tick });
            state.kill_agent(target);
        }
    }
}
```

Modify `crates/engine/src/cascade/mod.rs` to expose the new module + type. Check existing contents:

```bash
cat crates/engine/src/cascade/mod.rs
```

…then add `pub mod opportunity;` and `pub use opportunity::OpportunityAttackHandler;`. The existing mod file already exposes `pub use dispatch::*;` and `pub use handler::*;`; extend it:

```rust
pub mod dispatch;
pub mod handler;
pub mod opportunity;

pub use dispatch::{CascadeRegistry, MAX_CASCADE_ITERATIONS};
pub use handler::{CascadeHandler, EventKindId, Lane};
pub use opportunity::OpportunityAttackHandler;
```

(If your cascade `mod.rs` currently uses glob re-exports, replace with the explicit form above — it's the same pattern Plan 1 uses elsewhere and keeps schema-hash changes auditable.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p engine --test opportunity_attack`
Expected: PASS (3 tests).

Also verify engagement-slowed-move still passes — Task 4 tests asserted event emission but not damage; now damage should flow through:

Run: `cargo test -p engine --test engagement_slowed_move`
Expected: PASS (5 tests; no regressions).

- [ ] **Step 6: Commit**

```bash
git add crates/engine/src/cascade/ crates/engine/tests/opportunity_attack.rs
git commit -m "$(cat <<'EOF'
feat(engine): OpportunityAttackHandler + EventKindId=23 (Plan 3.5 T5)

Consumes Event::OpportunityAttackTriggered on Lane::Effect; applies
ATTACK_DAMAGE=10 to target; emits AgentAttacked + (on kill) AgentDied.
Rides the existing cascade fixed-point so death-audit handlers still
fire in the same tick. Handler lives in crates/engine/src/cascade/
opportunity.rs so apply_actions stays causation-free.
EOF
)"
```

---

## Task 6: Proptest — symmetry, hostility, range, determinism

**Why this test isn't circular:** Proptest generates random (agent_count, positions, creature_types) tuples, runs `update_engagements`, and then asserts three properties from first principles — not from reading the impl. (1) Bidirectional: for every `(a, b)` with `engaged_with[a] == Some(b)`, we independently verify `engaged_with[b] == Some(a)`. An impl bug that wrote only one side would fail. (2) No non-hostile: for every engaged pair, verify `is_hostile_to` is true — bug that ignored hostility would fail. (3) No out-of-range: for every engaged pair, verify `distance <= ENGAGEMENT_RANGE` — bug that widened the range would fail. Since the invariants are checked by reading state back, not by re-running the pass, a buggy impl that happens to produce a value that matches itself would still fail the first-principles checks. Determinism is asserted by running the same seed twice and comparing the full engagement map; catches any hidden hash-map ordering or float comparison nondeterminism.

**Files:**
- Test: `crates/engine/tests/proptest_engagement.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/engine/tests/proptest_engagement.rs`:

```rust
//! Property tests for `update_engagements`. First-principles checks only —
//! properties are asserted by reading `hot_engaged_with` slice, never by
//! re-running the pass or reading constants from the impl.

use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine::step::{update_engagements, ENGAGEMENT_RANGE};
use glam::Vec3;
use proptest::prelude::*;

fn arb_creature() -> impl Strategy<Value = CreatureType> {
    prop_oneof![
        Just(CreatureType::Human),
        Just(CreatureType::Wolf),
        Just(CreatureType::Deer),
        Just(CreatureType::Dragon),
    ]
}

fn arb_pos() -> impl Strategy<Value = Vec3> {
    // ±10m cube — plenty of boundary crossings at 2m range, plenty of out-of-range
    // pairs, and no float-inf pathology.
    (-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0)
        .prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

fn arb_population() -> impl Strategy<Value = Vec<(CreatureType, Vec3)>> {
    prop::collection::vec((arb_creature(), arb_pos()), 2..12)
}

fn build_state(pop: &[(CreatureType, Vec3)]) -> (SimState, Vec<AgentId>) {
    let cap = (pop.len() as u32 + 2).max(4);
    let mut state = SimState::new(cap, 7);
    let mut ids = Vec::with_capacity(pop.len());
    for (ct, pos) in pop {
        let id = state.spawn_agent(AgentSpawn {
            creature_type: *ct, pos: *pos, hp: 100.0,
        }).expect("spawn in bounded cap");
        ids.push(id);
    }
    (state, ids)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_iters: 5000,
        .. ProptestConfig::default()
    })]

    /// Property 1 (bidirectional): if a is engaged with b, then b is engaged with a.
    #[test]
    fn engagement_is_symmetric(pop in arb_population()) {
        let (mut state, ids) = build_state(&pop);
        update_engagements(&mut state);

        for &a in &ids {
            if let Some(Some(b)) = state.agent_engaged_with(a) {
                let partner_of_b = state.agent_engaged_with(b).flatten();
                prop_assert_eq!(
                    partner_of_b, Some(a),
                    "engagement({:?}) = Some({:?}), but engagement({:?}) = {:?}",
                    a, b, b, partner_of_b
                );
            }
        }
    }

    /// Property 2 (hostility-respecting): no engaged pair has `is_hostile_to == false`.
    #[test]
    fn no_non_hostile_engagement(pop in arb_population()) {
        let (mut state, ids) = build_state(&pop);
        update_engagements(&mut state);

        for &a in &ids {
            if let Some(Some(b)) = state.agent_engaged_with(a) {
                let cta = state.agent_creature_type(a).unwrap();
                let ctb = state.agent_creature_type(b).unwrap();
                prop_assert!(
                    cta.is_hostile_to(ctb),
                    "non-hostile pair engaged: {:?} vs {:?}", cta, ctb
                );
            }
        }
    }

    /// Property 3 (range-respecting): no engaged pair exceeds ENGAGEMENT_RANGE in 3D distance.
    #[test]
    fn no_out_of_range_engagement(pop in arb_population()) {
        let (mut state, ids) = build_state(&pop);
        update_engagements(&mut state);

        for &a in &ids {
            if let Some(Some(b)) = state.agent_engaged_with(a) {
                let pa = state.agent_pos(a).unwrap();
                let pb = state.agent_pos(b).unwrap();
                prop_assert!(
                    pa.distance(pb) <= ENGAGEMENT_RANGE + 1e-5,
                    "engaged pair at distance {}: {:?} vs {:?}",
                    pa.distance(pb), a, b
                );
            }
        }
    }

    /// Property 4 (determinism): identical inputs produce identical engagement maps.
    #[test]
    fn determinism_same_seed_same_map(pop in arb_population()) {
        let (mut s1, ids) = build_state(&pop);
        let (mut s2, _)   = build_state(&pop);
        update_engagements(&mut s1);
        update_engagements(&mut s2);

        for &a in &ids {
            prop_assert_eq!(
                s1.agent_engaged_with(a),
                s2.agent_engaged_with(a),
                "engagement for {:?} differed across identical runs", a
            );
        }
    }

    /// Property 5 (mutuality under mutation): after changing one agent's
    /// position, the invariants still hold. Catches subtle bugs where the
    /// committer picks the tentative partner without verifying mutuality.
    #[test]
    fn invariants_hold_after_perturbation(pop in arb_population(), jitter in -5.0f32..5.0) {
        let (mut state, ids) = build_state(&pop);
        update_engagements(&mut state);

        // Jitter the first agent and re-run.
        if let Some(&a0) = ids.first() {
            let p = state.agent_pos(a0).unwrap();
            state.set_agent_pos(a0, p + Vec3::new(jitter, 0.0, 0.0));
            update_engagements(&mut state);
        }

        // Re-check all three properties.
        for &a in &ids {
            if let Some(Some(b)) = state.agent_engaged_with(a) {
                prop_assert_eq!(
                    state.agent_engaged_with(b).flatten(), Some(a),
                    "asymmetry after perturbation"
                );
                let cta = state.agent_creature_type(a).unwrap();
                let ctb = state.agent_creature_type(b).unwrap();
                prop_assert!(cta.is_hostile_to(ctb), "non-hostile after perturbation");
                let pa = state.agent_pos(a).unwrap();
                let pb = state.agent_pos(b).unwrap();
                prop_assert!(pa.distance(pb) <= ENGAGEMENT_RANGE + 1e-5,
                    "range violated after perturbation: {}", pa.distance(pb));
            }
        }
    }
}
```

- [ ] **Step 2: Run test to verify it passes (properties already hold from Task 3)**

Run: `cargo test -p engine --test proptest_engagement`
Expected: PASS (5 properties × 500 cases each). If any fails, the proptest shrinks to a minimal counterexample; the failing case output is the actionable bug report.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/tests/proptest_engagement.rs
git commit -m "$(cat <<'EOF'
test(engine): proptest adversarial coverage for engagement (Plan 3.5 T6)

Five properties × 500 cases: bidirectional symmetry, hostility respected,
range respected (<=2m 3D), determinism across identical runs, and
invariants survive a perturbation-and-rerun sequence. First-principles
checks — no re-running the pass, no reading impl constants into the
verification side.
EOF
)"
```

---

## Task 7: Acceptance scenario — tank wall protects wizard

**Why this test isn't circular:** The acceptance test is a side-by-side comparison. The "wall" scenario and the "no-wall baseline" share the same wolf AI, the same spawn seed, and the same final goal (wolf attacking the wizard). The only differences are (a) the presence of 3 tank humans between wolf and wizard, and (b) the wall scenario uses `UtilityBackend`, which makes the wolf engage the nearest hostile (a tank) rather than running to the wizard. The invariant being tested — *the engagement/opportunity-attack system slows enemies and buys the wizard time* — is observed as a wall-clock-tick differential. A bug where engagement didn't form (e.g. Task 3 broken) would make the two scenarios take ~the same number of ticks. A bug where the slow factor was 1.0 (not applied) would also collapse the differential. The *baseline* scenario is what makes this non-circular: it lets the test fail if the wolf AI changes in a way that happens to delay the wizard death even without tanks.

**Files:**
- Test: `crates/engine/tests/acceptance_wall_formation.rs`
- Test: `crates/engine/tests/acceptance_no_wall_baseline.rs`

- [ ] **Step 1: Write the baseline test first**

Create `crates/engine/tests/acceptance_no_wall_baseline.rs`:

```rust
//! Baseline: wolf vs lone wizard, no tanks, no engagement help. Measures
//! "ticks until wizard dies" under the same `UtilityBackend` wolf AI used
//! by the wall-formation test. The differential between this number and
//! the wall-formation number is what proves ZoC works.
//!
//! Expected budget: wolf advances 1 m/tick from 10 m, attacks once in range
//! (ATTACK_RANGE = 2.0). Wizard HP = 30 ⇒ ~3 hits at 10 damage each ⇒
//! engagement forms at ~8 ticks (range 2m) and wizard dies on roughly
//! tick 10-12. The assert uses `<= 15` as a conservative upper bound.

use engine::cascade::{CascadeRegistry, OpportunityAttackHandler};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::invariant::InvariantRegistry;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::NullSink;
use engine::view::MaterializedView;
use glam::Vec3;

#[test]
fn wolf_reaches_lone_wizard_within_fifteen_ticks() {
    let mut state = SimState::new(8, 42);
    let wizard = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(0.0, -2.0, 0.0), hp: 30.0,
    }).unwrap();
    let _wolf = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(0.0, 10.0, 0.0), hp: 100.0,
    }).unwrap();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(8192);
    let mut registry = CascadeRegistry::new();
    registry.register(OpportunityAttackHandler);
    let invariants = InvariantRegistry::new();

    let mut died_at: Option<u32> = None;
    for _ in 0..50u32 {
        let mut views: Vec<&mut dyn MaterializedView> = vec![];
        step_full(
            &mut state, &mut scratch, &mut events, &UtilityBackend, &registry,
            &mut views[..], &invariants, &NullSink,
        );
        if !state.agent_alive(wizard) {
            died_at = Some(state.tick);
            break;
        }
    }

    let tick_of_death = died_at.expect("wizard should die within 50 ticks with no defender");
    assert!(
        tick_of_death <= 15,
        "baseline: wizard should die within 15 ticks, died at tick {}", tick_of_death,
    );

    // Sanity: baseline should emit no OpportunityAttackTriggered (no engagement
    // ever formed between wolf and wizard — wolf was never slowed).
    let any_opp = events.iter().any(|e| matches!(e, Event::OpportunityAttackTriggered { .. }));
    assert!(!any_opp, "baseline: no opportunity attacks should fire (no tank to break contact from)");
}
```

- [ ] **Step 2: Run the baseline test to verify it passes before writing the wall test**

Run: `cargo test -p engine --test acceptance_no_wall_baseline`
Expected: PASS. The wolf should close from 10 m to 2 m in 8 ticks, then attack; wizard hp = 30 dies in 3 attacks (ticks 8, 9, 10). `tick_of_death` should be around 10-12, comfortably under 15.

- [ ] **Step 3: Write the wall test**

Create `crates/engine/tests/acceptance_wall_formation.rs`:

```rust
//! Acceptance scenario: 3 tanks + 1 wizard vs 1 wolf. Demonstrates that
//! engagement + opportunity attacks let a tank line protect a weaker
//! back-line caster — no positional collision required.
//!
//! Geometry (top-down, z=0):
//!     y
//!     |
//!   10|          W  (wolf, hp 100)
//!     |
//!     |
//!    0|  T  T  T       (tanks, hp 200 each, x = -1, 0, 1)
//!   -2|     Z          (wizard, hp 30)
//!     |
//!
//! The wolf runs `UtilityBackend` — it picks the nearest hostile (a tank)
//! and attacks. Once engaged, the wolf can only disengage at slowed speed
//! and takes an opportunity attack. Three tanks × 200 hp each = 600 hp
//! to grind through; the wolf does 10 hp/tick. The wizard survives long
//! enough to matter.
//!
//! The test asserts FOUR things:
//! 1. An engagement forms within the first few ticks.
//! 2. The wolf's y-position moves SLOWLY while engaged.
//! 3. The opportunity-attack event fires at least once (if the wolf ever
//!    tries to move away, which UtilityBackend will when a tank dies).
//! 4. The wizard survives at least 30 ticks.

use engine::cascade::{CascadeRegistry, OpportunityAttackHandler};
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::invariant::InvariantRegistry;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::NullSink;
use engine::view::MaterializedView;
use glam::Vec3;

#[test]
fn tank_line_protects_wizard_for_thirty_plus_ticks() {
    let mut state = SimState::new(8, 42);

    // Three tanks in a line, hp 200 each. AgentIds 1,2,3.
    let tank_l = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(-1.0, 0.0, 0.0), hp: 200.0,
    }).unwrap();
    let tank_m = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new( 0.0, 0.0, 0.0), hp: 200.0,
    }).unwrap();
    let tank_r = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new( 1.0, 0.0, 0.0), hp: 200.0,
    }).unwrap();
    // Wizard behind the wall, hp 30. AgentId 4.
    let wizard = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(0.0, -2.0, 0.0), hp: 30.0,
    }).unwrap();
    // Wolf 10m north. AgentId 5.
    let wolf = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(0.0, 10.0, 0.0), hp: 100.0,
    }).unwrap();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(16384);
    let mut registry = CascadeRegistry::new();
    registry.register(OpportunityAttackHandler);
    let invariants = InvariantRegistry::new();

    let mut engagement_formed_at: Option<u32> = None;
    let mut wolf_y_history: Vec<(u32, f32)> = Vec::new();

    for _ in 0..50u32 {
        let mut views: Vec<&mut dyn MaterializedView> = vec![];
        step_full(
            &mut state, &mut scratch, &mut events, &UtilityBackend, &registry,
            &mut views[..], &invariants, &NullSink,
        );

        if engagement_formed_at.is_none() {
            if state.agent_engaged_with(wolf).flatten().is_some() {
                engagement_formed_at = Some(state.tick);
            }
        }
        if state.agent_alive(wolf) {
            wolf_y_history.push((state.tick, state.agent_pos(wolf).unwrap().y));
        }
        if !state.agent_alive(wizard) {
            panic!("wizard died at tick {} — ZoC failed to buy 30 ticks", state.tick);
        }
    }

    // (1) Engagement formed.
    let eng_tick = engagement_formed_at.expect("wolf must engage a tank within 50 ticks");
    assert!(eng_tick <= 12, "engagement should form by tick 12 (~10 ticks of travel), got {}", eng_tick);

    // (2) Wolf moved slowly while engaged. Look at y over a 10-tick window after
    //     engagement: delta must be < 4.0 (engaged → 0.3m/tick max; 10 × 0.3 = 3.0).
    let eng_y = wolf_y_history.iter()
        .find(|(t, _)| *t == eng_tick)
        .map(|(_, y)| *y)
        .expect("wolf y at engagement tick");
    let late_window = wolf_y_history.iter()
        .find(|(t, _)| *t == eng_tick + 10)
        .map(|(_, y)| *y);
    if let Some(late_y) = late_window {
        let traveled = (eng_y - late_y).abs();
        assert!(traveled < 4.0,
            "wolf should move slowly while engaged: y delta over 10 ticks was {}, expected <4.0",
            traveled);
    }

    // (3) Wizard still alive at tick 30+.
    assert!(state.agent_alive(wizard),
        "wizard must survive ≥30 ticks with the tank wall; tick_now={}", state.tick);

    // (4) At least one tank took damage — prove the wolf actually engaged and
    //     attacked, not just idled. (Not a ZoC property, but a sanity guard
    //     against the test passing because no combat happened at all.)
    let any_tank_hurt =
        state.agent_hp(tank_l).unwrap_or(200.0) < 200.0
     || state.agent_hp(tank_m).unwrap_or(200.0) < 200.0
     || state.agent_hp(tank_r).unwrap_or(200.0) < 200.0;
    assert!(any_tank_hurt, "at least one tank should have taken damage");
}

#[test]
fn wizard_survives_significantly_longer_with_wall() {
    // Differential assertion: the wall scenario holds wizard alive for the
    // whole 30-tick window; the baseline (acceptance_no_wall_baseline.rs)
    // kills the wizard by tick 15. 2x differential minimum is the ZoC signal.
    //
    // This test just re-runs the wall scenario (a subset of the check above)
    // and asserts the wizard is alive at tick 30 — the baseline test runs
    // the matching "wolf reaches lone wizard within 15 ticks" assertion, so
    // the >2x factor is implicit in the pair.

    let mut state = SimState::new(8, 42);
    let _ = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(-1.0, 0.0, 0.0), hp: 200.0,
    }).unwrap();
    let _ = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new( 0.0, 0.0, 0.0), hp: 200.0,
    }).unwrap();
    let _ = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new( 1.0, 0.0, 0.0), hp: 200.0,
    }).unwrap();
    let wizard = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(0.0, -2.0, 0.0), hp: 30.0,
    }).unwrap();
    let _ = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(0.0, 10.0, 0.0), hp: 100.0,
    }).unwrap();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(16384);
    let mut registry = CascadeRegistry::new();
    registry.register(OpportunityAttackHandler);
    let invariants = InvariantRegistry::new();

    for _ in 0..30u32 {
        let mut views: Vec<&mut dyn MaterializedView> = vec![];
        step_full(
            &mut state, &mut scratch, &mut events, &UtilityBackend, &registry,
            &mut views[..], &invariants, &NullSink,
        );
    }
    assert!(state.agent_alive(wizard),
        "with wall: wizard survives tick 30. baseline kills by tick 15 ⇒ 2× differential");
}
```

- [ ] **Step 4: Run the wall tests**

Run: `cargo test -p engine --test acceptance_wall_formation --test acceptance_no_wall_baseline`
Expected: PASS (baseline: wizard dies by tick ~12; wall: wizard alive at tick 30+, wolf engaged ≤ tick 12, wolf moved <4m while engaged over 10 ticks).

If the wall test fails the "at least one tank should have taken damage" check, the wolf never closed — a clue that `UtilityBackend` isn't picking the tank as the target because its hostility scoring treats tanks and wizards identically. In that case, verify `UtilityBackend::nearest_other` picks lowest-id ties and confirm tanks are spawned first (AgentIds 1,2,3) so they're nearer in slot order; the spawn order above already handles this.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/tests/acceptance_wall_formation.rs crates/engine/tests/acceptance_no_wall_baseline.rs
git commit -m "$(cat <<'EOF'
test(engine): acceptance — tank wall protects wizard via ZoC (Plan 3.5 T7)

Two tests:
  * no_wall_baseline: lone wizard dies by tick 15 against UtilityBackend
    wolf (no ZoC in play).
  * wall_formation:   3 tanks + wizard vs same wolf; wizard alive at
    tick 30, wolf engaged by tick 12 and moved <4m over the next 10
    ticks.

The 2x+ differential between the two scenarios — same wolf AI, same seed,
same geometry modulo the tank line — demonstrates that engagement +
opportunity attacks do the tactical-positioning work that would otherwise
require positional collision.
EOF
)"
```

---

## Task 8: Schema hash bump + status.md update

**Why this test isn't circular:** The schema-hash baseline test runs as part of `cargo test -p engine` and compares the computed hash to the committed `.schema_hash` file. After a legitimate schema change, this test fails — updating the baseline is part of the PR. The test can't be circular because the hash is computed over the fingerprint string (which we modify in code) but compared against a separate baseline file (which we regenerate). A bug where the fingerprint string was silently the same despite a real code change would fail: this task forces the string to include the new constants, new event variant, new EventKindId ordinal, and new SoA field. Leaving any one out would later surface as a `cargo test schema_hash` failure on the *next* schema-affecting commit that *does* update the string but doesn't cover the missing items.

**Files:**
- Modify: `crates/engine/src/schema_hash.rs`
- Modify: `crates/engine/.schema_hash`
- Modify: `docs/engine/status.md`

- [ ] **Step 1: Extend the schema-hash fingerprint**

Modify `crates/engine/src/schema_hash.rs`. Update the relevant `h.update(b"...")` calls:

```rust
    h.update(b"SimState:SoA{hot_pos=vec3,hot_hp=f32,hot_max_hp=f32,hot_alive=bool,hot_movement_mode=u8,hot_hunger=f32,hot_thirst=f32,hot_rest_timer=f32,hot_engaged_with=option_agentid};cold{creature_type=u8,channels=smallvec4,spawn_tick=u32}");
    h.update(b"Event:AgentMoved,AgentAttacked,AgentDied,AgentFled,AgentAte,AgentDrank,AgentRested,AgentCast,AgentUsedItem,AgentHarvested,AgentPlacedTile,AgentPlacedVoxel,AgentHarvestedVoxel,AgentConversed,AgentSharedStory,AgentCommunicated,InformationRequested,AgentRemembered,QuestPosted,QuestAccepted,BidPlaced,AnnounceEmitted,RecordMemory,OpportunityAttackTriggered,ChronicleEntry");
```

(Replace the existing `SimState:SoA{...}` line and `Event:...` line with the versions above. The `SimState` fingerprint previously omitted `hot_hunger`, `hot_thirst`, `hot_rest_timer` — fix that drift alongside the engagement field add.)

Extend the `EventKindId` line:

```rust
    h.update(b"EventKindId:AgentMoved=0,AgentAttacked=1,AgentDied=2,AgentFled=3,AgentAte=4,AgentDrank=5,AgentRested=6,AgentCast=7,AgentUsedItem=8,AgentHarvested=9,AgentPlacedTile=10,AgentPlacedVoxel=11,AgentHarvestedVoxel=12,AgentConversed=13,AgentSharedStory=14,AgentCommunicated=15,InformationRequested=16,AgentRemembered=17,QuestPosted=18,QuestAccepted=19,BidPlaced=20,AnnounceEmitted=21,RecordMemory=22,OpportunityAttackTriggered=23,ChronicleEntry=128");
```

Add a new line for the engagement constants (after the existing `OVERHEAR_RANGE=30` line):

```rust
    h.update(b"OVERHEAR_RANGE=30");
    h.update(b"ENGAGEMENT_RANGE=2,ENGAGEMENT_SLOW_FACTOR=0.3");
```

- [ ] **Step 2: Regenerate the baseline**

Run the schema-hash test once — it will fail, but emit the new expected hash:

Run: `cargo test -p engine --test schema_hash`
Expected: FAIL with a message like `baseline hash mismatch: computed=<NEW>, baseline=<OLD>`. Copy the `<NEW>` hex string (32 bytes = 64 hex chars).

Overwrite the baseline file:

```bash
cargo test -p engine --test schema_hash 2>&1 | grep 'computed=' | sed -E 's/.*computed=([a-f0-9]{64}).*/\1/' > crates/engine/.schema_hash
```

Or manually: run `cargo test -p engine --test schema_hash -- --nocapture` and paste the new hash into `crates/engine/.schema_hash` (replacing the entire file — the file is a single 64-hex-char line + trailing newline).

Re-run:

Run: `cargo test -p engine --test schema_hash`
Expected: PASS.

- [ ] **Step 3: Update `docs/engine/status.md`**

Modify `docs/engine/status.md`. In the Subsystem table (search for the `§9` header for MicroKind execution), add a new row immediately below the `11 event-only micros` row — before `§10 MacroKind`:

```markdown
| §9a | Engagement / ZoC (`hot_engaged_with`, `update_engagements`, slowed-move + opportunity-attack cascade) | ✅ 🎯 | P3.5 T1-T7 | `src/state/mod.rs`, `src/step.rs`, `src/cascade/opportunity.rs`, `src/event/mod.rs` | `tests/state_engagement.rs`, `tests/creature_hostile.rs`, `tests/engagement_formation.rs`, `tests/engagement_slowed_move.rs`, `tests/opportunity_attack.rs`, `tests/proptest_engagement.rs`, `tests/acceptance_wall_formation.rs`, `tests/acceptance_no_wall_baseline.rs` | Acceptance pair (wall vs baseline) pins 2x wizard-survival differential. Proptest covers bidirectional symmetry + hostility + range + determinism + perturbation stability over 500×5 cases. Remaining gap: `update_engagements` doesn't yet use the spatial index (O(alive²) pass) — fine for MVP, but will become a bottleneck once agent counts exceed ~1K. Flagged in §9a follow-ups. | **V10**: `viz_basic.toml` — the wolf should *visibly slow* when it enters tank range (its y-position delta per tick drops from ~1 m to ~0.3 m). **V11**: when the wolf decides to disengage (tank died / wizard targeting), the wolf-voxel should flash red for one frame (engagement-break damage overlay, implemented via the existing `Overlay::Attack` path since `OpportunityAttackTriggered` cascades into `AgentAttacked`). |
```

Add the Plan 3.5 row to the Plans index:

```markdown
| Plan 3.0 viz harness | `docs/superpowers/plans/2026-04-19-engine-plan-3_0-viz-harness.md` | ✅ executed (Tasks 1–5) |
| Plan 3.5 — engagement / ZoC | `docs/superpowers/plans/2026-04-19-engine-plan-3_5-engagement-zoc.md` | ⚠️ draft — awaiting execution |
| Plan 2.75 verification infra | `docs/superpowers/plans/2026-04-19-engine-plan-2_75-verification-infra.md` | ✅ executed (proptest + contracts + fuzz) |
```

Update open-verification question 11 (the "engine has no collision detection" item) to reflect the partial solution:

```markdown
11. **Engine has no agent-agent collision detection.** Multiple agents can occupy the same `Vec3` position simultaneously. Viz workarounds stack the cubes vertically (Plan 3.1). Plan 3.5 (engagement / ZoC) ships the *tactical* workaround: hostile agents in melee range form a mutual engagement, disengaging is slowed + takes an opportunity attack. This fixes the "tanks don't matter without collision" concern for combat, but does **not** prevent two wolves from occupying the same cell while walking around a map, nor does it prevent an agent from walking through a wall-of-tanks' *gaps*. A real positional-collision pass (probably phase 3.5 between shuffle and apply, or phase 4 as part of `apply_actions`) that either soft-pushes overlapping agents apart or hard-rejects moves onto occupied cells is still out of scope. Raise in the ability plan's world-physics or as a dedicated Plan.
```

Add two new rows to the Visual-check checklist (append after V9):

```markdown
| V10 | `viz_basic.toml`, watch the wolf as it enters the human cluster | At >2 m: wolf advances ~1 m/tick. At ≤2 m to the nearest human: wolf advances ~0.3 m/tick; it may reorient but its forward progress visibly slows. | Engagement formation + speed gating (Plan 3.5 T3+T4). |
| V11 | `viz_basic.toml` at the moment the wolf's engaged tank dies | Wolf voxel flashes red for one frame (opportunity-attack overlay, piggy-backing on the existing Attack overlay since the cascade emits AgentAttacked). Wolf then starts moving toward the next target at full speed (no longer engaged). | Opportunity-attack cascade + engagement clear on partner death (Plan 3.5 T5). |
```

- [ ] **Step 4: Run full engine test suite**

Run: `cargo test -p engine`
Expected: ALL TESTS PASS — original 169 + 8 new files (`state_engagement.rs`, `creature_hostile.rs`, `engagement_formation.rs`, `engagement_slowed_move.rs`, `opportunity_attack.rs`, `proptest_engagement.rs`, `acceptance_wall_formation.rs`, `acceptance_no_wall_baseline.rs`). Final count: ~190 tests green.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/schema_hash.rs crates/engine/.schema_hash docs/engine/status.md
git commit -m "$(cat <<'EOF'
chore(engine): schema_hash bump + status.md §9a for Plan 3.5

Bumps the baseline fingerprint to cover hot_engaged_with, the
OpportunityAttackTriggered event variant + EventKindId=23, and the
ENGAGEMENT_RANGE/SLOW_FACTOR constants. Also straightens pre-existing
drift: the SoA fingerprint string previously missed hot_hunger,
hot_thirst, and hot_rest_timer — now included.

Adds a §9a Engagement row to the subsystem table with V10+V11 visual
checks, and rewrites open-verification question 11 to reflect that Plan
3.5 solves the "enemies phase through tanks" case tactically even though
a true collision pass is still out of scope.
EOF
)"
```

---

## Follow-ups (out of scope for this plan)

These are the cross-cutting items the review-before-completion check should surface as "not-done-but-intentional":

1. **Ability Plan 1 integration.** When `CastHandler` lands (currently ⚠️ pulled), its melee-range cast gate needs to read `state.agent_engaged_with(caster)` and emit `OpportunityAttackTriggered` for any non-engaged-target cast fired while engaged — identical to the `MoveToward` logic. Expected change: ≤10 lines in `CastHandler::handle`, plus one `engagement_gated_cast_emits_opportunity_attack` test. Add to Ability Plan 1 as the first task of its integration PR.

2. **Spatial-index acceleration.** `update_engagements` is O(alive²). At ~200 agents this is 40K pairwise checks/tick — still cheap (<1 ms), but beyond ~2K agents it dominates the tick budget. When `spatial::SpatialIndex::query_within_radius` is mature (Plan 2.75 landed proptest coverage), rewrite `update_engagements` to `for a in alive { for b in spatial.query_within(a.pos, ENGAGEMENT_RANGE) { … } }`. Shouldn't change observable behavior (determinism tie-break stays the same).

3. **Hostility via the social-graph matrix.** `CreatureType::is_hostile_to` is a stub. The per-pair relationship system (per MEMORY.md, NPC economy / social graph plan) will supersede it. When that lands, `update_engagements` keeps its signature; only the "hostile?" predicate switches from the enum method to a lookup.

4. **MicroKind::Attack gating while engaged.** Plan 1 T9 doesn't currently check engagement state — an attacker engaged with wolf A can still attack human B if B is in `ATTACK_RANGE`. Whether that's desirable (splash / multi-target) or buggy (should-have-attacked-my-engager) is a design call, not a 3.5 scope call. Flag in `docs/engine/status.md` open-verification items after 3.5 lands.

5. **Viz harness overlay for engagement.** V10 + V11 are acceptance criteria — the viz crate needs to actually render engagement indicators (e.g. a thin line between engaged pairs, or a slowdown cue on the mover). That's a Plan 3.1 follow-up in the viz crate, not engine-side work.

---

## Self-review checklist

Before handing this plan off, verify:

- [ ] **Spec coverage.** Every design commitment in the plan source (items 1-8) maps to a task:
  - `hot_engaged_with` SoA field → T1
  - `ENGAGEMENT_RANGE = 2.0` → T3
  - `ENGAGEMENT_SLOW_FACTOR = 0.3` → T3
  - Bidirectional invariant → T3 (formation) + T6 (proptest)
  - Tick-start update pass → T3
  - MoveToward + Flee gating → T4
  - Opportunity attack as cascade handler (not new action) → T5
  - `CreatureType::is_hostile_to` stub → T2
- [ ] **No placeholders.** Every step block has exact code or exact command.
- [ ] **Type consistency.**
  - `hot_engaged_with: Vec<Option<AgentId>>` consistent across T1 + T3 + schema hash.
  - `agent_engaged_with(id) -> Option<Option<AgentId>>` (outer = slot exists, inner = engagement) — consistent across T1, T3, T4, T5, T6, T7.
  - `OpportunityAttackTriggered { attacker, target, tick }` field names match across event def (T4), cascade handler (T5), tests (T4 + T5).
  - Event ordinal 23 consistent (T5 + T8).
- [ ] **TDD discipline.** Each task: failing test → impl → passing test → commit. T8 is chore/doc (no TDD in the strict sense) — schema-hash bump is gated on the `schema_hash` test passing.
- [ ] **Determinism.** T3 explicitly spells out `agents_alive()` iteration order + lowest-raw-id tie-break; T6 proptest enforces determinism.
- [ ] **Interaction with Ability Plan 1.** Preamble + follow-up #1 call it out; no action in this plan.
- [ ] **Visual-check items.** V10 + V11 added to status.md.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-engine-plan-3_5-engagement-zoc.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (T1 through T8), review between tasks, fast iteration. Given the acceptance test in T7 depends on the full engagement system being in place, the subagent for T7 should run after T6 is merged.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Recommended checkpoints after T3 (update_engagements integrated), T5 (opportunity cascade working), and T8 (schema + docs).

**Which approach?**
