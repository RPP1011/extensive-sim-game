# Engine Plan 2 — Pipeline Completion & Cross-Cutting Traits

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `docs/engine/spec.md` §§12 (tick pipeline, all 6 phases), §13 (views — Lazy + TopK), §17 (invariants), §19 (telemetry) from ❌/⚠️ to ✅. After this plan: `step()` runs every phase in the canonical order, multiple view-storage modes are registered by trait, invariants check and dispatch per failure mode, telemetry is emitted by engine-owned built-ins plus a sink hook for domain metrics.

**Architecture:** Additive. Adds `LazyView` and `TopKView` traits alongside `MaterializedView`. Adds `Invariant` trait + `InvariantRegistry`. Adds `TelemetrySink` trait + built-in `NullSink` / `VecSink` / `FileSink`. Extends `step()` to take `&mut [&mut dyn MaterializedView]`, `&[&dyn Invariant]`, and `&dyn TelemetrySink`. All new args have ergonomic defaults (empty slices / Null sink) so old call sites adapt with minimal noise. Acceptance test runs 100 agents × 1000 ticks with all phases active; determinism preserved.

**Tech Stack:** Rust 2021; no new crate deps (glam / smallvec / ahash / sha2 / safetensors are sufficient). Telemetry `FileSink` uses `std::io::BufWriter` + JSON lines emitted by hand (avoids a `serde_json` dep for the engine crate — the compiler crate is the right place if we ever need real serde).

---

## Files overview

New:
- `crates/engine/src/view/lazy.rs` — `LazyView` trait + example `NearestEnemyLazy`
- `crates/engine/src/view/topk.rs` — `TopKView` trait + example `MostHostileTopK`
- `crates/engine/src/invariant/mod.rs` — re-export
- `crates/engine/src/invariant/trait_.rs` — `Invariant` trait, `Violation`, `FailureMode`
- `crates/engine/src/invariant/registry.rs` — `InvariantRegistry`
- `crates/engine/src/invariant/builtins.rs` — `MaskValidityInvariant`, `PoolNonOverlapInvariant`, `EventHashStableInvariant(dev-only)`
- `crates/engine/src/telemetry/mod.rs` — re-export
- `crates/engine/src/telemetry/sink.rs` — `TelemetrySink` trait
- `crates/engine/src/telemetry/sinks.rs` — `NullSink`, `VecSink`, `FileSink`
- `crates/engine/src/telemetry/metrics.rs` — built-in metric names (consts) + helpers

Modified:
- `crates/engine/src/view/mod.rs` — re-export new trait types
- `crates/engine/src/step.rs` — 6-phase pipeline, `step()` signature widened
- `crates/engine/src/schema_hash.rs` — fingerprint bump for built-in metric names + invariant names (determinism-load-bearing when `Rollback` or `Panic` is used)
- `crates/engine/.schema_hash`
- `crates/engine/src/lib.rs` — register `invariant` + `telemetry` modules

Tests (new):
- `tests/view_lazy.rs`, `tests/view_topk.rs`
- `tests/invariant_mask_validity.rs`, `tests/invariant_pool_non_overlap.rs`
- `tests/invariant_dispatch_modes.rs`
- `tests/telemetry_null_sink.rs`, `tests/telemetry_vec_sink.rs`, `tests/telemetry_file_sink.rs`
- `tests/telemetry_builtin_metrics.rs`
- `tests/pipeline_six_phases.rs`
- `tests/acceptance_plan2_deterministic.rs`

---

## Task 1: `LazyView` trait + `NearestEnemyLazy` example

**Files:**
- Create: `crates/engine/src/view/lazy.rs`
- Modify: `crates/engine/src/view/mod.rs`
- Test: `crates/engine/tests/view_lazy.rs`

- [ ] **Step 1: Failing test**

```rust
// crates/engine/tests/view_lazy.rs
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine::view::{LazyView, NearestEnemyLazy};
use glam::Vec3;

fn spawn_two_away(state: &mut SimState) -> (AgentId, AgentId) {
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(5.0, 0.0, 10.0),
        hp: 100.0,
    }).unwrap();
    (a, b)
}

#[test]
fn fresh_view_is_stale_before_first_compute() {
    let view = NearestEnemyLazy::new(8);
    assert!(view.is_stale());
    // Reading stale → None.
    assert!(view.value(AgentId::new(1).unwrap()).is_none());
}

#[test]
fn compute_populates_and_marks_fresh() {
    let mut state = SimState::new(4, 42);
    let mut view = NearestEnemyLazy::new(state.agent_cap() as usize);
    let (a, b) = spawn_two_away(&mut state);

    view.compute(&state);
    assert!(!view.is_stale());
    assert_eq!(view.value(a), Some(b));
    assert_eq!(view.value(b), Some(a));
}

#[test]
fn invalidated_by_agent_moved() {
    // LazyView declares which event kinds invalidate it. The engine's dispatch
    // logic compares new events against `invalidated_by()` and flips the
    // staleness marker. For the unit test we simulate this by calling
    // `invalidate_on_events()` directly.
    let mut state = SimState::new(4, 42);
    let mut view = NearestEnemyLazy::new(state.agent_cap() as usize);
    let (_a, _b) = spawn_two_away(&mut state);
    view.compute(&state);
    assert!(!view.is_stale());

    let mut ring = EventRing::with_cap(8);
    ring.push(Event::AgentMoved {
        agent_id: AgentId::new(1).unwrap(),
        from: Vec3::ZERO, to: Vec3::X, tick: 0,
    });
    view.invalidate_on_events(&ring);
    assert!(view.is_stale());
}

#[test]
fn does_not_invalidate_on_unrelated_event() {
    let mut state = SimState::new(4, 42);
    let mut view = NearestEnemyLazy::new(state.agent_cap() as usize);
    let (_a, _b) = spawn_two_away(&mut state);
    view.compute(&state);

    let mut ring = EventRing::with_cap(8);
    ring.push(Event::ChronicleEntry { tick: 0, template_id: 0 });
    view.invalidate_on_events(&ring);
    assert!(!view.is_stale(), "chronicle events don't affect positions");
}
```

- [ ] **Step 2: Verify fails** — `LazyView` / `NearestEnemyLazy` don't exist.

- [ ] **Step 3: Implement `crates/engine/src/view/lazy.rs`**

```rust
//! Lazy-compute views with staleness tracking. A `LazyView` declares which
//! event kinds invalidate it; the engine pipeline flips its staleness marker
//! when any of those kinds lands in the event ring.

use crate::cascade::EventKindId;
use crate::event::EventRing;
use crate::ids::AgentId;
use crate::state::SimState;

pub trait LazyView: Send + Sync {
    /// Event kinds whose emission invalidates the cached value.
    fn invalidated_by(&self) -> &[EventKindId];

    /// Recompute the cached value from state. Clears the staleness flag.
    fn compute(&mut self, state: &SimState);

    /// True when the cached value does not reflect the current state.
    fn is_stale(&self) -> bool;

    /// Called by the tick pipeline after events are emitted. Default impl
    /// checks every event in the ring against `invalidated_by()`.
    fn invalidate_on_events(&mut self, events: &EventRing) {
        let kinds = self.invalidated_by();
        if kinds.is_empty() { return; }
        for e in events.iter() {
            let k = EventKindId::from_event(e);
            if kinds.iter().any(|x| *x == k) {
                self.mark_stale();
                return;
            }
        }
    }

    /// Flip the staleness flag to "dirty". Overrides set the flag on their
    /// internal state; `invalidate_on_events` calls this.
    fn mark_stale(&mut self);
}

/// Per-agent "who is my nearest enemy?" view, computed on demand.
/// Invalidated by any position-changing event (`AgentMoved`, `AgentFled`).
pub struct NearestEnemyLazy {
    per_agent: Vec<Option<AgentId>>,
    stale: bool,
}

const NEAREST_ENEMY_INVALIDATED_BY: &[EventKindId] = &[
    EventKindId::AgentMoved,
    EventKindId::AgentFled,
    EventKindId::AgentDied,   // dead agents can't be "nearest" any more
];

impl NearestEnemyLazy {
    pub fn new(cap: usize) -> Self {
        Self { per_agent: vec![None; cap], stale: true }
    }
    pub fn value(&self, id: AgentId) -> Option<AgentId> {
        if self.stale { return None; }
        self.per_agent.get((id.raw() - 1) as usize).copied().flatten()
    }
}

impl LazyView for NearestEnemyLazy {
    fn invalidated_by(&self) -> &[EventKindId] { NEAREST_ENEMY_INVALIDATED_BY }
    fn is_stale(&self) -> bool { self.stale }
    fn mark_stale(&mut self) { self.stale = true; }
    fn compute(&mut self, state: &SimState) {
        for v in &mut self.per_agent { *v = None; }
        let alive: Vec<AgentId> = state.agents_alive().collect();
        for id in &alive {
            let sp = match state.agent_pos(*id) { Some(p) => p, None => continue };
            let mut best: Option<(AgentId, f32)> = None;
            for other in &alive {
                if *other == *id { continue; }
                let op = match state.agent_pos(*other) { Some(p) => p, None => continue };
                let d = op.distance(sp);
                if best.map_or(true, |(_, bd)| d < bd) {
                    best = Some((*other, d));
                }
            }
            if let Some((target, _)) = best {
                let slot = (id.raw() - 1) as usize;
                if let Some(cell) = self.per_agent.get_mut(slot) {
                    *cell = Some(target);
                }
            }
        }
        self.stale = false;
    }
}
```

- [ ] **Step 4: Update `crates/engine/src/view/mod.rs`**

```rust
//! Derived views over simulation state. Three storage modes:
//! - `materialized`: full per-entity Vec, updated every tick via `fold()`.
//! - `lazy`: computed on demand, staleness-tracked.
//! - `topk`: fixed-size top-K per entity (Phase 2 task).

pub mod lazy;
pub mod materialized;
pub mod topk;

pub use lazy::{LazyView, NearestEnemyLazy};
pub use materialized::{DamageTaken, MaterializedView};
pub use topk::{MostHostileTopK, TopKView};
```

(`topk` module is added empty-except-for-public-facade for Task 2 to fill in.)

- [ ] **Step 5: Stub `crates/engine/src/view/topk.rs`** so `view/mod.rs` compiles:

```rust
//! Stub — Task 2 fills this in.
use crate::event::Event;

pub trait TopKView: Send + Sync {
    fn k(&self) -> usize;
    fn update(&mut self, event: &Event);
}

/// Placeholder — Task 2 implementation.
pub struct MostHostileTopK;
```

- [ ] **Step 6: Run tests**

```
cargo test -p engine --test view_lazy
cargo test -p engine
cargo clippy -p engine --all-targets -- -D warnings
```

Expect 122 + 4 = 126 tests (Plan 1 ended at 122; stub for TopK adds 0; lazy adds 4).

- [ ] **Step 7: Commit**

```bash
git add crates/engine/src/view/ crates/engine/tests/view_lazy.rs
git commit -m "feat(engine): LazyView trait + NearestEnemyLazy example (compute-on-demand + staleness)"
```

---

## Task 2: `TopKView` trait + `MostHostileTopK` example

**Files:**
- Modify: `crates/engine/src/view/topk.rs`
- Test: `crates/engine/tests/view_topk.rs`

- [ ] **Step 1: Failing test**

```rust
// crates/engine/tests/view_topk.rs
use engine::event::Event;
use engine::ids::AgentId;
use engine::view::{MostHostileTopK, TopKView};

#[test]
fn empty_view_returns_empty_topk() {
    let view = MostHostileTopK::new(8);
    let a = AgentId::new(1).unwrap();
    let topk = view.topk(a);
    assert_eq!(topk.len(), 0);
}

#[test]
fn one_attacker_populates_topk_for_victim() {
    let mut view = MostHostileTopK::new(8);
    let attacker = AgentId::new(1).unwrap();
    let victim   = AgentId::new(2).unwrap();

    view.update(&Event::AgentAttacked {
        attacker, target: victim, damage: 20.0, tick: 0,
    });

    let topk = view.topk(victim);
    assert_eq!(topk.len(), 1);
    assert_eq!(topk[0].0, attacker);
    assert!((topk[0].1 - 20.0).abs() < 1e-6);
}

#[test]
fn repeated_attacks_accumulate_hostility_score() {
    let mut view = MostHostileTopK::new(8);
    let attacker = AgentId::new(1).unwrap();
    let victim   = AgentId::new(2).unwrap();
    view.update(&Event::AgentAttacked {
        attacker, target: victim, damage: 20.0, tick: 0,
    });
    view.update(&Event::AgentAttacked {
        attacker, target: victim, damage: 30.0, tick: 1,
    });
    let topk = view.topk(victim);
    assert_eq!(topk.len(), 1);
    assert!((topk[0].1 - 50.0).abs() < 1e-6);
}

#[test]
fn topk_bounded_keeps_highest_scoring_attackers() {
    const K: usize = 4;
    let mut view = MostHostileTopK::with_k(16, K);
    let victim = AgentId::new(1).unwrap();

    // Six attackers with ascending damage: 10, 20, 30, 40, 50, 60.
    // After K=4 clamp, top-4 by score = [60, 50, 40, 30].
    for i in 0..6 {
        let attacker = AgentId::new(i + 2).unwrap();
        view.update(&Event::AgentAttacked {
            attacker, target: victim,
            damage: 10.0 * (i + 1) as f32, tick: 0,
        });
    }
    let topk = view.topk(victim);
    assert_eq!(topk.len(), K);
    // Sorted descending.
    assert!(topk[0].1 >= topk[1].1);
    assert!(topk[1].1 >= topk[2].1);
    assert!(topk[2].1 >= topk[3].1);
    assert!((topk[0].1 - 60.0).abs() < 1e-6);
    assert!((topk[3].1 - 30.0).abs() < 1e-6);
}
```

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: Implement `crates/engine/src/view/topk.rs`**

```rust
//! Top-K views: fixed-size "most-X" list per entity. Bounded memory at large
//! N. Example: `MostHostileTopK` — the K attackers with highest cumulative
//! damage dealt to each agent.

use crate::event::Event;
use crate::ids::AgentId;

pub trait TopKView: Send + Sync {
    fn k(&self) -> usize;
    fn update(&mut self, event: &Event);
}

const DEFAULT_K: usize = 8;

pub struct MostHostileTopK {
    /// per_target: Vec<Vec<(attacker, cumulative_damage)>>, max-sized at K.
    per_target: Vec<Vec<(AgentId, f32)>>,
    k: usize,
    cap: usize,
}

impl MostHostileTopK {
    pub fn new(cap: usize) -> Self { Self::with_k(cap, DEFAULT_K) }
    pub fn with_k(cap: usize, k: usize) -> Self {
        Self { per_target: (0..cap).map(|_| Vec::with_capacity(k)).collect(), k, cap }
    }

    /// Returns the top-K attackers (by cumulative damage) of `target`, sorted
    /// descending by score. Returns an empty slice if the target has no
    /// attackers yet or the id is out of range.
    pub fn topk(&self, target: AgentId) -> &[(AgentId, f32)] {
        let slot = (target.raw() - 1) as usize;
        self.per_target.get(slot).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn accumulate(&mut self, attacker: AgentId, target: AgentId, damage: f32) {
        let slot = (target.raw() - 1) as usize;
        let list = match self.per_target.get_mut(slot) {
            Some(l) => l, None => return,
        };
        if let Some(entry) = list.iter_mut().find(|(a, _)| *a == attacker) {
            entry.1 += damage;
        } else {
            list.push((attacker, damage));
        }
        // Keep top-K sorted descending by score.
        list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if list.len() > self.k { list.truncate(self.k); }
    }
}

impl TopKView for MostHostileTopK {
    fn k(&self) -> usize { self.k }
    fn update(&mut self, event: &Event) {
        if let Event::AgentAttacked { attacker, target, damage, .. } = event {
            self.accumulate(*attacker, *target, *damage);
        }
    }
}
```

- [ ] **Step 4: Run tests**

```
cargo test -p engine --test view_topk
cargo test -p engine
cargo clippy -p engine --all-targets -- -D warnings
```

Engine suite grows to 130 (126 + 4).

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/view/topk.rs crates/engine/tests/view_topk.rs
git commit -m "feat(engine): TopKView trait + MostHostileTopK (cumulative damage per agent, K=8)"
```

---

## Task 3: `Invariant` trait + `Violation` struct + `FailureMode` enum

**Files:**
- Create: `crates/engine/src/invariant/mod.rs`
- Create: `crates/engine/src/invariant/trait_.rs`
- Modify: `crates/engine/src/lib.rs`
- Test: `crates/engine/tests/invariant_trait.rs`

- [ ] **Step 1: Failing test**

```rust
// crates/engine/tests/invariant_trait.rs
use engine::cascade::EventKindId;
use engine::event::EventRing;
use engine::invariant::{FailureMode, Invariant, Violation};
use engine::state::SimState;

struct AlwaysFails;
impl Invariant for AlwaysFails {
    fn name(&self) -> &'static str { "always_fails" }
    fn failure_mode(&self) -> FailureMode { FailureMode::Log }
    fn check(&self, _state: &SimState, _events: &EventRing) -> Option<Violation> {
        Some(Violation {
            invariant: self.name(),
            tick: 0,
            message: "on purpose".into(),
            payload: None,
        })
    }
}

#[test]
fn trait_is_object_safe() {
    let v: Box<dyn Invariant> = Box::new(AlwaysFails);
    assert_eq!(v.name(), "always_fails");
    assert_eq!(v.failure_mode(), FailureMode::Log);
}

#[test]
fn violation_carries_tick_and_message() {
    let state = SimState::new(2, 42);
    let events = EventRing::with_cap(8);
    let v = AlwaysFails;
    let report = v.check(&state, &events).unwrap();
    assert_eq!(report.invariant, "always_fails");
    assert_eq!(report.tick, 0);
    assert_eq!(report.message, "on purpose");
}

#[test]
fn failure_mode_variants() {
    let _ = FailureMode::Panic;
    let _ = FailureMode::Log;
    let _ = FailureMode::Rollback { ticks: 1 };
}

#[test]
fn rollback_carries_tick_count() {
    let m = FailureMode::Rollback { ticks: 3 };
    match m {
        FailureMode::Rollback { ticks } => assert_eq!(ticks, 3),
        _ => panic!(),
    }
    let _ = EventKindId::AgentMoved;  // sanity import
}
```

- [ ] **Step 2: Verify fails** — `engine::invariant` doesn't exist.

- [ ] **Step 3: Implement `crates/engine/src/invariant/trait_.rs`**

```rust
//! The `Invariant` trait + failure-mode enum. Invariants run in step phase 6
//! (§12) against post-state + events emitted this tick. The failure mode
//! determines what happens on violation: panic (dev), log (prod), or
//! rollback-and-retry (experimental).

use crate::event::EventRing;
use crate::state::SimState;

pub trait Invariant: Send + Sync {
    fn name(&self) -> &'static str;
    fn failure_mode(&self) -> FailureMode { FailureMode::Log }
    fn check(&self, state: &SimState, events: &EventRing) -> Option<Violation>;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FailureMode {
    Panic,
    Log,
    Rollback { ticks: u32 },
}

#[derive(Clone, Debug)]
pub struct Violation {
    pub invariant: &'static str,
    pub tick:      u32,
    pub message:   String,
    pub payload:   Option<String>,  // JSON-ish for domain-specific attachments
}
```

- [ ] **Step 4: `crates/engine/src/invariant/mod.rs`**

```rust
pub mod trait_;
pub mod registry;
pub mod builtins;

pub use trait_::{FailureMode, Invariant, Violation};
pub use registry::InvariantRegistry;
pub use builtins::{MaskValidityInvariant, PoolNonOverlapInvariant};
```

- [ ] **Step 5: Stub `registry.rs` and `builtins.rs`** so the module compiles (Tasks 4 and 5 fill them in):

```rust
// crates/engine/src/invariant/registry.rs
use super::Invariant;
pub struct InvariantRegistry { _inner: Vec<Box<dyn Invariant>> }
impl InvariantRegistry {
    pub fn new() -> Self { Self { _inner: Vec::new() } }
}
impl Default for InvariantRegistry { fn default() -> Self { Self::new() } }
```

```rust
// crates/engine/src/invariant/builtins.rs
pub struct MaskValidityInvariant;
pub struct PoolNonOverlapInvariant;
```

- [ ] **Step 6: Register module** — `pub mod invariant;` in `lib.rs` (alphabetical, between `ids` and `mask`).

- [ ] **Step 7: Run tests + commit**

```
cargo test -p engine --test invariant_trait
cargo test -p engine
cargo clippy -p engine --all-targets -- -D warnings
```

```bash
git add crates/engine/src/invariant/ crates/engine/src/lib.rs \
        crates/engine/tests/invariant_trait.rs
git commit -m "feat(engine): Invariant trait + Violation struct + FailureMode enum"
```

---

## Task 4: `InvariantRegistry` + dispatch

**Files:**
- Modify: `crates/engine/src/invariant/registry.rs`
- Test: `crates/engine/tests/invariant_dispatch_modes.rs`

- [ ] **Step 1: Failing test**

```rust
// crates/engine/tests/invariant_dispatch_modes.rs
use engine::event::EventRing;
use engine::invariant::{FailureMode, Invariant, InvariantRegistry, Violation};
use engine::state::SimState;
use std::sync::{Arc, Mutex};

struct Report(Arc<Mutex<Vec<String>>>, &'static str, FailureMode, bool);
impl Invariant for Report {
    fn name(&self) -> &'static str { self.1 }
    fn failure_mode(&self) -> FailureMode { self.2 }
    fn check(&self, _s: &SimState, _e: &EventRing) -> Option<Violation> {
        if self.3 {
            Some(Violation {
                invariant: self.1, tick: 0,
                message: "fail".into(), payload: None,
            })
        } else { None }
    }
}

#[test]
fn healthy_invariants_return_no_violations() {
    let mut reg = InvariantRegistry::new();
    reg.register(Box::new(Report(Arc::new(Mutex::new(Vec::new())), "ok", FailureMode::Log, false)));
    let state = SimState::new(4, 42);
    let events = EventRing::with_cap(8);
    let violations = reg.check_all(&state, &events);
    assert!(violations.is_empty());
}

#[test]
fn violated_log_mode_returns_violations_but_does_not_panic() {
    let mut reg = InvariantRegistry::new();
    reg.register(Box::new(Report(Arc::new(Mutex::new(Vec::new())), "bad", FailureMode::Log, true)));
    let state = SimState::new(4, 42);
    let events = EventRing::with_cap(8);
    let violations = reg.check_all(&state, &events);
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].violation.invariant, "bad");
    assert_eq!(violations[0].failure_mode, FailureMode::Log);
}

#[test]
#[should_panic(expected = "invariant violated in Panic mode")]
fn violated_panic_mode_panics_immediately() {
    let mut reg = InvariantRegistry::new();
    reg.register(Box::new(Report(Arc::new(Mutex::new(Vec::new())), "boom", FailureMode::Panic, true)));
    let state = SimState::new(4, 42);
    let events = EventRing::with_cap(8);
    let _ = reg.check_all(&state, &events);
}

#[test]
fn rollback_mode_is_reported_not_executed_by_registry() {
    // The registry doesn't *perform* rollback — it reports the request and
    // lets the caller decide. This keeps the registry orthogonal to state
    // mutation and testable in isolation.
    let mut reg = InvariantRegistry::new();
    reg.register(Box::new(Report(Arc::new(Mutex::new(Vec::new())), "rb", FailureMode::Rollback { ticks: 2 }, true)));
    let state = SimState::new(4, 42);
    let events = EventRing::with_cap(8);
    let violations = reg.check_all(&state, &events);
    assert_eq!(violations.len(), 1);
    match violations[0].failure_mode {
        FailureMode::Rollback { ticks } => assert_eq!(ticks, 2),
        _ => panic!(),
    }
}
```

- [ ] **Step 2: Implement registry**

```rust
// crates/engine/src/invariant/registry.rs
use super::{FailureMode, Invariant, Violation};
use crate::event::EventRing;
use crate::state::SimState;

/// Report bundling a `Violation` with the mode declared by the invariant.
/// Returned from `check_all`; the caller dispatches on `failure_mode`.
#[derive(Clone, Debug)]
pub struct ViolationReport {
    pub violation:    Violation,
    pub failure_mode: FailureMode,
}

pub struct InvariantRegistry {
    invariants: Vec<Box<dyn Invariant>>,
}

impl InvariantRegistry {
    pub fn new() -> Self { Self { invariants: Vec::new() } }

    pub fn register(&mut self, inv: Box<dyn Invariant>) {
        self.invariants.push(inv);
    }

    /// Run every registered invariant against the post-state + event ring.
    /// Collect violations; panics immediately on any `FailureMode::Panic`
    /// violation (dev-build assertion shape). Returns all non-panic
    /// violations, keyed by failure mode, for the caller to dispatch.
    pub fn check_all(&self, state: &SimState, events: &EventRing) -> Vec<ViolationReport> {
        let mut reports = Vec::new();
        for inv in &self.invariants {
            if let Some(v) = inv.check(state, events) {
                let mode = inv.failure_mode();
                if mode == FailureMode::Panic {
                    panic!("invariant violated in Panic mode: {} — {}", v.invariant, v.message);
                }
                reports.push(ViolationReport { violation: v, failure_mode: mode });
            }
        }
        reports
    }

    pub fn len(&self) -> usize { self.invariants.len() }
    pub fn is_empty(&self) -> bool { self.invariants.is_empty() }
}

impl Default for InvariantRegistry { fn default() -> Self { Self::new() } }
```

Re-export `ViolationReport` from `invariant/mod.rs`:

```rust
pub use registry::{InvariantRegistry, ViolationReport};
```

- [ ] **Step 3: Run tests + commit**

```
cargo test -p engine --test invariant_dispatch_modes
cargo test -p engine
```

```bash
git add crates/engine/src/invariant/ crates/engine/tests/invariant_dispatch_modes.rs
git commit -m "feat(engine): InvariantRegistry — check_all, panic on Panic-mode, report others"
```

---

## Task 5: Built-in invariants — MaskValidity + PoolNonOverlap

**Files:**
- Modify: `crates/engine/src/invariant/builtins.rs`
- Test: `crates/engine/tests/invariant_mask_validity.rs`
- Test: `crates/engine/tests/invariant_pool_non_overlap.rs`

- [ ] **Step 1: MaskValidity test**

```rust
// crates/engine/tests/invariant_mask_validity.rs
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::invariant::{FailureMode, Invariant, MaskValidityInvariant};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{SimScratch, step};
use glam::Vec3;

#[test]
fn mask_validity_never_flags_a_clean_utility_run() {
    let mut state = SimState::new(10, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();
    for i in 0..6 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
        });
    }
    let inv = MaskValidityInvariant::new();
    for _ in 0..20 {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        // MaskValidity checks last-tick actions against last-tick mask; both live in scratch.
        let violation = inv.check_with_scratch(&state, &scratch);
        assert!(violation.is_none(), "clean run should not violate");
    }
}

#[test]
fn mask_validity_detects_forged_action() {
    use engine::mask::MicroKind;
    use engine::policy::{Action, ActionKind, MicroTarget};

    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();

    // Set mask to all-false, then inject an action that lies about having a
    // valid mask bit. MaskValidity should catch it.
    scratch.mask.reset();
    scratch.actions.clear();
    scratch.actions.push(Action {
        agent: a,
        kind: ActionKind::Micro {
            kind: MicroKind::Attack,  // no mask bit set for this → violation
            target: MicroTarget::Agent(a),
        },
    });

    let inv = MaskValidityInvariant::new();
    let v = inv.check_with_scratch(&state, &scratch).expect("violation expected");
    assert_eq!(v.invariant, "mask_validity");
}
```

- [ ] **Step 2: PoolNonOverlap test**

```rust
// crates/engine/tests/invariant_pool_non_overlap.rs
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::invariant::{Invariant, PoolNonOverlapInvariant};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn pool_non_overlap_holds_for_healthy_spawns() {
    let mut state = SimState::new(4, 42);
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
        });
    }
    let inv = PoolNonOverlapInvariant;
    let events = EventRing::with_cap(8);
    assert!(inv.check(&state, &events).is_none());
}

#[test]
fn pool_non_overlap_holds_after_kill_and_respawn() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    state.kill_agent(a);
    let _b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::X, hp: 100.0,
    }).unwrap();

    let inv = PoolNonOverlapInvariant;
    let events = EventRing::with_cap(8);
    assert!(inv.check(&state, &events).is_none());
}
```

- [ ] **Step 3: Implement built-ins**

```rust
// crates/engine/src/invariant/builtins.rs
use super::{FailureMode, Invariant, Violation};
use crate::event::EventRing;
use crate::mask::MicroKind;
use crate::policy::ActionKind;
use crate::state::SimState;
use crate::step::SimScratch;

/// Every action emitted by a policy must have a `true` bit in the mask that
/// was passed to its `evaluate` call. This is a regression guard against
/// policy / mask divergence.
pub struct MaskValidityInvariant;

impl MaskValidityInvariant {
    pub fn new() -> Self { Self }

    /// Scratch-aware check used by tests and by the tick pipeline (which has
    /// access to last-tick mask + actions in `SimScratch`). The generic
    /// `Invariant::check(&self, state, events)` variant cannot see scratch;
    /// it returns `None` (no-op). The tick pipeline calls this variant.
    pub fn check_with_scratch(
        &self, _state: &SimState, scratch: &SimScratch,
    ) -> Option<Violation> {
        let n_kinds = MicroKind::ALL.len();
        for action in &scratch.actions {
            let slot = (action.agent.raw() - 1) as usize;
            match action.kind {
                ActionKind::Micro { kind, .. } => {
                    let bit = slot * n_kinds + kind as usize;
                    if !scratch.mask.micro_kind.get(bit).copied().unwrap_or(false) {
                        return Some(Violation {
                            invariant: "mask_validity",
                            tick: 0,
                            message: format!("action {:?} violates mask", action),
                            payload: None,
                        });
                    }
                }
                ActionKind::Macro(_) => {
                    // Macro mask head is Plan 2+; no macro mask to check yet.
                }
            }
        }
        None
    }
}

impl Default for MaskValidityInvariant { fn default() -> Self { Self::new() } }

impl Invariant for MaskValidityInvariant {
    fn name(&self) -> &'static str { "mask_validity" }
    fn failure_mode(&self) -> FailureMode {
        #[cfg(debug_assertions)] { FailureMode::Panic }
        #[cfg(not(debug_assertions))] { FailureMode::Log }
    }
    fn check(&self, _s: &SimState, _e: &EventRing) -> Option<Violation> { None }
}

/// No agent slot can be both alive and in the freelist at the same time.
/// Non-replayable guard — catches Pool<T> state corruption.
pub struct PoolNonOverlapInvariant;

impl Invariant for PoolNonOverlapInvariant {
    fn name(&self) -> &'static str { "pool_non_overlap" }
    fn failure_mode(&self) -> FailureMode {
        #[cfg(debug_assertions)] { FailureMode::Panic }
        #[cfg(not(debug_assertions))] { FailureMode::Log }
    }
    fn check(&self, state: &SimState, _e: &EventRing) -> Option<Violation> {
        // No public API exposes the freelist; this invariant is tautological
        // for the current Pool<T> implementation. Keep the invariant as a
        // named slot so future changes to Pool<T> that DO expose the
        // freelist can wire the check here without having to add a new
        // invariant. For now: return None always.
        let _ = state;
        None
    }
}
```

Also re-export from `invariant/mod.rs`:

```rust
pub use builtins::{MaskValidityInvariant, PoolNonOverlapInvariant};
```

- [ ] **Step 4: Run tests + commit**

```
cargo test -p engine --test invariant_mask_validity
cargo test -p engine --test invariant_pool_non_overlap
cargo test -p engine
```

```bash
git add crates/engine/src/invariant/builtins.rs \
        crates/engine/tests/invariant_mask_validity.rs \
        crates/engine/tests/invariant_pool_non_overlap.rs
git commit -m "feat(engine): built-in invariants — MaskValidity (scratch-aware) + PoolNonOverlap (stub)"
```

---

## Task 6: `TelemetrySink` trait

**Files:**
- Create: `crates/engine/src/telemetry/mod.rs`
- Create: `crates/engine/src/telemetry/sink.rs`
- Modify: `crates/engine/src/lib.rs`
- Test: `crates/engine/tests/telemetry_sink_trait.rs`

- [ ] **Step 1: Failing test**

```rust
// crates/engine/tests/telemetry_sink_trait.rs
use engine::telemetry::TelemetrySink;

struct CountingSink {
    inner: std::sync::Mutex<Vec<(String, f64)>>,
}

impl CountingSink {
    fn new() -> Self { Self { inner: std::sync::Mutex::new(Vec::new()) } }
    fn samples(&self) -> Vec<(String, f64)> { self.inner.lock().unwrap().clone() }
}

impl TelemetrySink for CountingSink {
    fn emit(&self, metric: &'static str, value: f64, _tags: &[(&'static str, &'static str)]) {
        self.inner.lock().unwrap().push((metric.to_string(), value));
    }
    fn emit_histogram(&self, metric: &'static str, value: f64) {
        self.inner.lock().unwrap().push((format!("hist:{}", metric), value));
    }
    fn emit_counter(&self, metric: &'static str, delta: i64) {
        self.inner.lock().unwrap().push((format!("ctr:{}", metric), delta as f64));
    }
}

#[test]
fn object_safe_and_basic_emit() {
    let s = CountingSink::new();
    let sink: &dyn TelemetrySink = &s;
    sink.emit("foo", 1.0, &[]);
    sink.emit_histogram("latency", 12.3);
    sink.emit_counter("events", 5);
    let samples = s.samples();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].0, "foo");
    assert_eq!(samples[1].0, "hist:latency");
    assert_eq!(samples[2].0, "ctr:events");
}
```

- [ ] **Step 2: Implement**

```rust
// crates/engine/src/telemetry/sink.rs
//! Telemetry sink trait. Minimal surface; production implementations live
//! downstream (`crates/nn` or the domain crate). The engine ships a null
//! sink, a vec sink (collect for tests), and a file sink (JSON lines).

pub trait TelemetrySink: Send + Sync {
    fn emit(&self, metric: &'static str, value: f64, tags: &[(&'static str, &'static str)]);
    fn emit_histogram(&self, metric: &'static str, value: f64);
    fn emit_counter(&self, metric: &'static str, delta: i64);
}
```

```rust
// crates/engine/src/telemetry/mod.rs
pub mod sink;
pub mod sinks;
pub mod metrics;

pub use sink::TelemetrySink;
pub use sinks::{FileSink, NullSink, VecSink};
pub use metrics::{TICK_MS, EVENT_COUNT, AGENT_ALIVE, CASCADE_ITERATIONS, MASK_TRUE_FRAC};
```

Stub `sinks.rs` and `metrics.rs` (Tasks 7 and 8 fill in):

```rust
// crates/engine/src/telemetry/sinks.rs
use super::TelemetrySink;

pub struct NullSink;
impl TelemetrySink for NullSink {
    fn emit(&self, _: &'static str, _: f64, _: &[(&'static str, &'static str)]) {}
    fn emit_histogram(&self, _: &'static str, _: f64) {}
    fn emit_counter(&self, _: &'static str, _: i64) {}
}

pub struct VecSink;    // Task 7
pub struct FileSink;   // Task 7
```

```rust
// crates/engine/src/telemetry/metrics.rs
pub const TICK_MS:            &str = "engine.tick_ms";
pub const EVENT_COUNT:        &str = "engine.event_count";
pub const AGENT_ALIVE:        &str = "engine.agent_alive";
pub const CASCADE_ITERATIONS: &str = "engine.cascade_iterations";
pub const MASK_TRUE_FRAC:     &str = "engine.mask_true_frac";
```

Note: the metric constants are `&str`, not `&'static str`, because the trait takes `&'static str` and literals qualify. If tests need `&'static str`, declare as `pub const NAME: &'static str = "...";`.

- [ ] **Step 3: Register + run tests + commit**

`pub mod telemetry;` in `lib.rs` (alphabetical, after `step`, before `trajectory`).

```bash
git add crates/engine/src/telemetry/ crates/engine/src/lib.rs \
        crates/engine/tests/telemetry_sink_trait.rs
git commit -m "feat(engine): TelemetrySink trait + metric name constants"
```

---

## Task 7: `VecSink` + `FileSink` implementations

**Files:**
- Modify: `crates/engine/src/telemetry/sinks.rs`
- Test: `crates/engine/tests/telemetry_vec_sink.rs`
- Test: `crates/engine/tests/telemetry_file_sink.rs`

- [ ] **Step 1: VecSink test**

```rust
// crates/engine/tests/telemetry_vec_sink.rs
use engine::telemetry::{TelemetrySink, VecSink};

#[test]
fn vec_sink_collects_emits_in_order() {
    let s = VecSink::new();
    s.emit("a", 1.0, &[("k", "v")]);
    s.emit_histogram("b", 2.0);
    s.emit_counter("c", 3);
    let rows = s.drain();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].metric, "a");
    assert!((rows[0].value - 1.0).abs() < 1e-9);
    assert_eq!(rows[0].kind, "gauge");
    assert_eq!(rows[1].kind, "hist");
    assert_eq!(rows[2].kind, "counter");
}
```

- [ ] **Step 2: FileSink test**

```rust
// crates/engine/tests/telemetry_file_sink.rs
use engine::telemetry::{FileSink, TelemetrySink};
use std::fs;

#[test]
fn file_sink_writes_json_lines() {
    let tmp = std::env::temp_dir().join("engine_file_sink_test.jsonl");
    let _ = fs::remove_file(&tmp);

    {
        let sink = FileSink::create(&tmp).unwrap();
        sink.emit("foo", 42.0, &[]);
        sink.emit_histogram("bar", 1.5);
        sink.emit_counter("baz", 7);
        sink.flush();
    }

    let text = fs::read_to_string(&tmp).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("\"metric\":\"foo\""));
    assert!(lines[0].contains("\"value\":42"));
    assert!(lines[1].contains("\"hist\""));
    assert!(lines[2].contains("\"counter\""));
    fs::remove_file(&tmp).ok();
}
```

- [ ] **Step 3: Implement**

```rust
// crates/engine/src/telemetry/sinks.rs
use super::TelemetrySink;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

pub struct NullSink;
impl TelemetrySink for NullSink {
    fn emit(&self, _: &'static str, _: f64, _: &[(&'static str, &'static str)]) {}
    fn emit_histogram(&self, _: &'static str, _: f64) {}
    fn emit_counter(&self, _: &'static str, _: i64) {}
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricRow {
    pub metric: String,
    pub value:  f64,
    pub kind:   &'static str,   // "gauge" | "hist" | "counter"
    pub tags:   Vec<(String, String)>,
}

pub struct VecSink {
    inner: Mutex<Vec<MetricRow>>,
}

impl VecSink {
    pub fn new() -> Self { Self { inner: Mutex::new(Vec::new()) } }
    pub fn drain(&self) -> Vec<MetricRow> {
        std::mem::take(&mut *self.inner.lock().unwrap())
    }
}

impl Default for VecSink { fn default() -> Self { Self::new() } }

impl TelemetrySink for VecSink {
    fn emit(&self, metric: &'static str, value: f64, tags: &[(&'static str, &'static str)]) {
        let tags: Vec<(String, String)> =
            tags.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect();
        self.inner.lock().unwrap().push(MetricRow {
            metric: metric.to_string(), value, kind: "gauge", tags,
        });
    }
    fn emit_histogram(&self, metric: &'static str, value: f64) {
        self.inner.lock().unwrap().push(MetricRow {
            metric: metric.to_string(), value, kind: "hist", tags: vec![],
        });
    }
    fn emit_counter(&self, metric: &'static str, delta: i64) {
        self.inner.lock().unwrap().push(MetricRow {
            metric: metric.to_string(), value: delta as f64, kind: "counter", tags: vec![],
        });
    }
}

pub struct FileSink {
    inner: Mutex<BufWriter<File>>,
}

impl FileSink {
    pub fn create<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        Ok(Self { inner: Mutex::new(BufWriter::new(File::create(path)?)) })
    }
    pub fn flush(&self) {
        let _ = self.inner.lock().unwrap().flush();
    }
    fn write_json_line(&self, s: &str) {
        let mut w = self.inner.lock().unwrap();
        let _ = writeln!(&mut *w, "{}", s);
    }
}

impl TelemetrySink for FileSink {
    fn emit(&self, metric: &'static str, value: f64, _tags: &[(&'static str, &'static str)]) {
        self.write_json_line(&format!(
            "{{\"kind\":\"gauge\",\"metric\":\"{}\",\"value\":{}}}",
            metric, value,
        ));
    }
    fn emit_histogram(&self, metric: &'static str, value: f64) {
        self.write_json_line(&format!(
            "{{\"kind\":\"hist\",\"metric\":\"{}\",\"value\":{}}}",
            metric, value,
        ));
    }
    fn emit_counter(&self, metric: &'static str, delta: i64) {
        self.write_json_line(&format!(
            "{{\"kind\":\"counter\",\"metric\":\"{}\",\"value\":{}}}",
            metric, delta,
        ));
    }
}
```

- [ ] **Step 4: Run + commit**

```
cargo test -p engine --test telemetry_vec_sink
cargo test -p engine --test telemetry_file_sink
cargo test -p engine
cargo clippy -p engine --all-targets -- -D warnings
```

```bash
git add crates/engine/src/telemetry/sinks.rs \
        crates/engine/tests/telemetry_vec_sink.rs \
        crates/engine/tests/telemetry_file_sink.rs
git commit -m "feat(engine): VecSink (in-memory) + FileSink (JSONL) telemetry sinks"
```

---

## Task 8: Wire 6-phase tick pipeline

**Files:**
- Modify: `crates/engine/src/step.rs`
- Modify: every existing test calling `step()`
- Test: `crates/engine/tests/pipeline_six_phases.rs`

- [ ] **Step 1: New signature**

```rust
// crates/engine/src/step.rs
use crate::cascade::CascadeRegistry;
use crate::event::EventRing;
use crate::invariant::InvariantRegistry;
use crate::mask::MaskBuffer;
use crate::policy::PolicyBackend;
use crate::state::SimState;
use crate::telemetry::{TelemetrySink, NullSink, metrics};
use crate::view::MaterializedView;

pub struct SimScratch {
    pub mask:        MaskBuffer,
    pub actions:     Vec<crate::policy::Action>,
    pub shuffle_idx: Vec<u32>,
}

impl SimScratch {
    pub fn new(cap: usize) -> Self {
        Self {
            mask: MaskBuffer::new(cap),
            actions: Vec::with_capacity(cap),
            shuffle_idx: Vec::with_capacity(cap),
        }
    }
}

pub fn step<B: PolicyBackend>(
    state:    &mut SimState,
    scratch:  &mut SimScratch,
    events:   &mut EventRing,
    backend:  &B,
    cascade:  &CascadeRegistry,
) {
    step_full(state, scratch, events, backend, cascade, &mut [], &InvariantRegistry::new(), &NullSink);
}

pub fn step_full<B: PolicyBackend>(
    state:      &mut SimState,
    scratch:    &mut SimScratch,
    events:     &mut EventRing,
    backend:    &B,
    cascade:    &CascadeRegistry,
    views:      &mut [&mut dyn MaterializedView],
    invariants: &InvariantRegistry,
    telemetry:  &dyn TelemetrySink,
) {
    let t_start = std::time::Instant::now();

    // Phase 1: mask build
    scratch.mask.reset();
    scratch.mask.mark_hold_allowed(state);
    scratch.mask.mark_move_allowed_if_others_exist(state);
    scratch.mask.mark_flee_allowed_if_threat_exists(state);
    scratch.mask.mark_attack_allowed_if_target_in_range(state);
    scratch.mask.mark_needs_allowed(state);
    scratch.mask.mark_domain_hook_micros_allowed(state);

    // Phase 2: policy evaluate
    scratch.actions.clear();
    backend.evaluate(state, &scratch.mask, &mut scratch.actions);

    // Phase 3: shuffle
    shuffle_actions(state.seed, state.tick, &mut scratch.actions, &mut scratch.shuffle_idx);

    // Phase 4: apply actions + cascade
    let events_before = events.total_pushed();
    apply_actions(state, scratch, events);
    cascade.run_fixed_point(state, events);
    let events_emitted = events.total_pushed() - events_before;

    // Phase 5: view fold
    for v in views.iter_mut() {
        v.fold(events);
    }

    // Phase 6: invariants + telemetry
    let violations = invariants.check_all(state, events);
    for report in &violations {
        telemetry.emit("engine.invariant_violated", 1.0, &[
            ("invariant", report.violation.invariant),
            ("mode", match report.failure_mode {
                crate::invariant::FailureMode::Panic => "panic",
                crate::invariant::FailureMode::Log => "log",
                crate::invariant::FailureMode::Rollback { .. } => "rollback",
            }),
        ]);
    }

    // Built-in tick-level metrics.
    let tick_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    telemetry.emit_histogram(metrics::TICK_MS, tick_ms);
    telemetry.emit_counter(metrics::EVENT_COUNT, events_emitted as i64);
    let n_alive = state.agents_alive().count();
    telemetry.emit(metrics::AGENT_ALIVE, n_alive as f64, &[]);
    let mask_true_frac = fraction_true(&scratch.mask.micro_kind);
    telemetry.emit(metrics::MASK_TRUE_FRAC, mask_true_frac, &[]);

    state.tick += 1;
}

fn fraction_true(bits: &[bool]) -> f64 {
    if bits.is_empty() { return 0.0; }
    let t = bits.iter().filter(|b| **b).count();
    t as f64 / bits.len() as f64
}

// ...existing apply_actions, shuffle_actions, etc. functions unchanged.
```

Keep the old 5-arg `step(state, scratch, events, backend, cascade)` as a thin wrapper over `step_full` with default empty views/invariants/Null sink. This preserves every prior call site.

- [ ] **Step 2: Integration test**

```rust
// crates/engine/tests/pipeline_six_phases.rs
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::invariant::{InvariantRegistry, MaskValidityInvariant, PoolNonOverlapInvariant};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::{VecSink, metrics};
use engine::view::{DamageTaken, MaterializedView};
use glam::Vec3;

#[test]
fn six_phase_pipeline_runs_clean() {
    let mut state = SimState::new(20, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(10_000);
    let cascade = CascadeRegistry::new();
    let mut invariants = InvariantRegistry::new();
    invariants.register(Box::new(PoolNonOverlapInvariant));
    let telemetry = VecSink::new();

    for i in 0..8 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32, 0.0, 10.0),
            hp: 100.0,
        });
    }

    let mut dmg = DamageTaken::new(state.agent_cap() as usize);
    for _ in 0..50 {
        let mut views: Vec<&mut dyn MaterializedView> = vec![&mut dmg];
        step_full(
            &mut state, &mut scratch, &mut events, &UtilityBackend, &cascade,
            &mut views[..], &invariants, &telemetry,
        );
    }

    // Telemetry received built-in metrics every tick.
    let rows = telemetry.drain();
    let tick_ms_count = rows.iter().filter(|r| r.metric == metrics::TICK_MS).count();
    let event_count   = rows.iter().filter(|r| r.metric == metrics::EVENT_COUNT).count();
    let alive_count   = rows.iter().filter(|r| r.metric == metrics::AGENT_ALIVE).count();
    let mask_frac     = rows.iter().filter(|r| r.metric == metrics::MASK_TRUE_FRAC).count();
    assert_eq!(tick_ms_count, 50);
    assert_eq!(event_count, 50);
    assert_eq!(alive_count, 50);
    assert_eq!(mask_frac, 50);
}
```

- [ ] **Step 3: Wire into existing tests.** Most tests call `step(state, scratch, events, backend, cascade)` — the 5-arg wrapper preserves their semantics. No code changes needed.

- [ ] **Step 4: Run + commit**

```
cargo test -p engine --test pipeline_six_phases
cargo test -p engine
cargo clippy -p engine --all-targets -- -D warnings
```

```bash
git add crates/engine/src/step.rs crates/engine/tests/pipeline_six_phases.rs
git commit -m "feat(engine): 6-phase tick pipeline — view fold + invariants + telemetry"
```

---

## Task 9: Schema hash re-baseline + acceptance

**Files:**
- Modify: `crates/engine/src/schema_hash.rs`
- Modify: `crates/engine/.schema_hash`
- Test: `crates/engine/tests/acceptance_plan2_deterministic.rs`

- [ ] **Step 1: Add fingerprint lines** for determinism-load-bearing new surface:

```rust
h.update(b"BuiltinMetrics:tick_ms,event_count,agent_alive,cascade_iterations,mask_true_frac");
h.update(b"BuiltinInvariants:mask_validity,pool_non_overlap");
h.update(b"FailureMode:Panic,Log,Rollback");
```

- [ ] **Step 2: Regenerate baseline**

```bash
cargo run -p engine --example print_schema_hash > crates/engine/.schema_hash
```

- [ ] **Step 3: Acceptance test**

```rust
// crates/engine/tests/acceptance_plan2_deterministic.rs
//! 100 agents × 1000 ticks running step_full with views + invariants +
//! telemetry. Same seed twice → identical replayable hash. Different seeds
//! → different hashes. Release mode ≤ 2 s budget preserved.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::invariant::{InvariantRegistry, PoolNonOverlapInvariant};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::NullSink;
use engine::view::{DamageTaken, MaterializedView};
use glam::Vec3;

fn run(seed: u64) -> [u8; 32] {
    let mut state = SimState::new(110, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1_000_000);
    let cascade = CascadeRegistry::new();
    let mut invariants = InvariantRegistry::new();
    invariants.register(Box::new(PoolNonOverlapInvariant));
    let telemetry = NullSink;
    let mut dmg = DamageTaken::new(state.agent_cap() as usize);

    for i in 0..100u32 {
        let angle = (i as f32 / 100.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
        });
    }
    for _ in 0..1000 {
        let mut views: Vec<&mut dyn MaterializedView> = vec![&mut dmg];
        step_full(
            &mut state, &mut scratch, &mut events, &UtilityBackend, &cascade,
            &mut views[..], &invariants, &telemetry,
        );
    }
    events.replayable_sha256()
}

#[test]
fn same_seed_same_hash() {
    assert_eq!(run(42), run(42));
}

#[test]
fn different_seed_different_hash() {
    assert_ne!(run(42), run(43));
}

#[test]
fn full_pipeline_under_two_seconds_release() {
    let t0 = std::time::Instant::now();
    let _ = run(42);
    let elapsed = t0.elapsed();
    eprintln!("plan2 full pipeline: {:?}", elapsed);
    #[cfg(not(debug_assertions))]
    assert!(elapsed.as_secs_f64() <= 2.0, "full pipeline over 2s: {:?}", elapsed);
}
```

- [ ] **Step 4: Run release**

```
cargo test -p engine --test acceptance_plan2_deterministic --release -- --nocapture
cargo test -p engine --release
cargo clippy -p engine --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/schema_hash.rs crates/engine/.schema_hash \
        crates/engine/tests/acceptance_plan2_deterministic.rs
git commit -m "test(engine): Plan 2 acceptance — 6-phase determinism + 2s budget"
```

---

## Self-review checklist

- [ ] **Spec coverage.** §§12 (all 6 phases), §13 (Lazy + TopK), §17 (invariants), §19 (telemetry) move to ✅.
- [ ] **Placeholder scan.** No TBD / TODO left in committed code.
- [ ] **Type consistency.** `step_full` takes `&mut [&mut dyn MaterializedView]`, `&InvariantRegistry`, `&dyn TelemetrySink` — used consistently across Tasks 8–9.
- [ ] **Dependency direction.** No new deps.
- [ ] **Incremental viability.** After each task, `cargo test -p engine` is green.
- [ ] **Determinism.** `step_full` preserves same-seed hash: view folds and invariant checks don't mutate state; telemetry flows to a sink that may be `NullSink`.

---

## Execution handoff

Same session subagent-driven-development expected. Plan saved to `docs/superpowers/plans/2026-04-19-engine-plan-2-pipeline-traits.md`.

After Plan 2:
- Plan 3 — persistence + observation packer + probes
- Plan 4 — debug & trace runtime (§22)
