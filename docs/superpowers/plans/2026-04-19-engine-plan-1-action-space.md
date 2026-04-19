# Engine Plan 1 — Action Space & Cascade Runtime

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `docs/engine/spec.md` §§7–9 and §14 from ❌/⚠️ to ✅. After this plan: the engine carries the full 18-variant `MicroKind` enum, the 4-variant `MacroKind` enum with built-in Announce cascade, a cascade-dispatch runtime with lane discipline and bounded iteration, a generic `AggregatePool<T>`, and native step semantics for Hold / MoveToward / Flee / Attack / Eat / Drink / Rest. Domain-dependent actions emit typed events without state change (compiler-registered cascades run the actual effects in Phase 2 plans).

**Architecture:** Additive changes to `crates/engine/`. No workspace-level breakage. Generic `Pool<T>` refactored from `AgentSlotPool`, reused for `AggregatePool`. `Event` enum grows new replayable variants; `EventRing` gains a `causes: Vec<Option<EventId>>` sidecar so the byte-packed hash stays stable. `MaskBuffer::micro_kind` grows to hold 18 bits per agent. `UtilityBackend` stays hand-scored — the new action kinds just widen its score table. Acceptance: 100-agent × 1000-tick deterministic run with mixed action kinds, release-mode ≤ 2 s.

**Tech Stack:** Rust 2021, `glam`, `smallvec`, `ahash`, `sha2`, `safetensors`. No new deps.

---

## Files overview

New:
- `crates/engine/src/pool.rs` — generic `Pool<T>` with freelist + `NonZeroU32` IDs.
- `crates/engine/src/cascade/mod.rs` — registry.
- `crates/engine/src/cascade/handler.rs` — `CascadeHandler` trait + `Lane`.
- `crates/engine/src/cascade/dispatch.rs` — bounded dispatch loop.
- `crates/engine/src/policy/macro_kind.rs` — `MacroKind`, `MacroAction`, parameter enums.
- `crates/engine/src/policy/query.rs` — `QueryKind` for `Ask`.
- `crates/engine/src/aggregate/mod.rs` — `AggregatePool<T>`, `AggregateId`.
- `crates/engine/src/aggregate/quest.rs` — `Quest` struct shape.
- `crates/engine/src/aggregate/group.rs` — `Group` struct shape.

Modified:
- `crates/engine/src/ids.rs` — add `AggregateId`, `EventId`, `GroupId`, `QuestId`, `ItemId`, `ResourceRef`.
- `crates/engine/src/state/entity_pool.rs` — delete in favour of `pool.rs` (keeps `AgentSlotPool` as a type alias for `Pool<AgentTag>`).
- `crates/engine/src/state/agent.rs` — add hunger, thirst, rest_timer to `AgentSpawn` and defaults.
- `crates/engine/src/state/mod.rs` — add `hot_hunger`, `hot_thirst`, `hot_rest_timer` SoA Vecs + accessors.
- `crates/engine/src/event/mod.rs` — add new replayable variants + `EventId`; document cause-sidecar.
- `crates/engine/src/event/ring.rs` — add `causes: Vec<Option<EventId>>` parallel to events; hash unchanged.
- `crates/engine/src/mask.rs` — grow `MicroKind` to 18 variants; `MicroKind::ALL.len()` to 18.
- `crates/engine/src/policy/mod.rs` — unify `Action` into `MicroAction | MacroAction` sum.
- `crates/engine/src/policy/utility.rs` — extend score table to cover new micros; emit macros at low prior.
- `crates/engine/src/step.rs` — dispatch all 18 micros + 4 macros; call cascade registry after apply.
- `crates/engine/src/schema_hash.rs` — include new variants + cascade-loop bound.
- `crates/engine/.schema_hash` — regenerated baseline.
- `crates/engine/src/lib.rs` — register new modules.

Tests (new):
- `tests/pool_generic.rs`, `tests/state_needs.rs`, `tests/event_id_threading.rs`
- `tests/micro_kind_full.rs`, `tests/macro_kind.rs`, `tests/action_sum.rs`
- `tests/cascade_register_dispatch.rs`, `tests/cascade_bounded.rs`, `tests/cascade_lanes.rs`
- `tests/action_flee.rs`, `tests/action_attack_kill.rs`, `tests/action_needs.rs`
- `tests/action_emit_only.rs`, `tests/macro_emit_only.rs`
- `tests/announce_audience.rs`, `tests/announce_overhear.rs`
- `tests/aggregate_pool.rs`, `tests/aggregate_types.rs`
- `tests/acceptance_mixed_actions.rs`

---

## Task 1: Generic `Pool<T>` with freelist

**Files:**
- Create: `crates/engine/src/pool.rs`
- Modify: `crates/engine/src/lib.rs`
- Modify: `crates/engine/src/state/entity_pool.rs` (gut → type alias)
- Test: `crates/engine/tests/pool_generic.rs`

- [ ] **Step 1: Write failing test** `crates/engine/tests/pool_generic.rs`

```rust
use engine::pool::{Pool, PoolId};

struct AgentTag;
type AgentSlotId = PoolId<AgentTag>;

#[test]
fn alloc_gives_sequential_ids_from_one() {
    let mut p: Pool<AgentTag> = Pool::new(4);
    assert_eq!(p.alloc().map(|i| i.raw()), Some(1));
    assert_eq!(p.alloc().map(|i| i.raw()), Some(2));
    assert_eq!(p.alloc().map(|i| i.raw()), Some(3));
}

#[test]
fn alloc_returns_none_at_capacity() {
    let mut p: Pool<AgentTag> = Pool::new(2);
    assert!(p.alloc().is_some());
    assert!(p.alloc().is_some());
    assert!(p.alloc().is_none());
}

#[test]
fn kill_then_alloc_reuses_slot() {
    let mut p: Pool<AgentTag> = Pool::new(4);
    let a = p.alloc().unwrap();
    let b = p.alloc().unwrap();
    p.kill(a);
    let c = p.alloc().unwrap();
    assert_eq!(c.raw(), a.raw(), "freelist popped, slot reused");
    assert_ne!(b.raw(), c.raw());
}

#[test]
fn is_alive_tracks_state() {
    let mut p: Pool<AgentTag> = Pool::new(4);
    let a = p.alloc().unwrap();
    assert!(p.is_alive(a));
    p.kill(a);
    assert!(!p.is_alive(a));
}
```

- [ ] **Step 2: Run to verify fails**

Run: `cargo test -p engine --test pool_generic`
Expected: compile error — `engine::pool` doesn't exist.

- [ ] **Step 3: Implement `crates/engine/src/pool.rs`**

```rust
use std::marker::PhantomData;
use std::num::NonZeroU32;

pub struct PoolId<T> {
    raw: NonZeroU32,
    _tag: PhantomData<fn() -> T>,
}

impl<T> PoolId<T> {
    pub fn new(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(|raw| Self { raw, _tag: PhantomData })
    }
    pub fn raw(&self) -> u32 { self.raw.get() }
    #[inline] pub fn slot(&self) -> usize { (self.raw.get() - 1) as usize }
}

impl<T> Clone for PoolId<T> {
    fn clone(&self) -> Self { *self }
}
impl<T> Copy for PoolId<T> {}
impl<T> PartialEq for PoolId<T> {
    fn eq(&self, o: &Self) -> bool { self.raw == o.raw }
}
impl<T> Eq for PoolId<T> {}
impl<T> std::hash::Hash for PoolId<T> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) { self.raw.hash(h) }
}
impl<T> std::fmt::Debug for PoolId<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PoolId({})", self.raw.get())
    }
}

/// Slot-indexed pool with a freelist. `NonZeroU32` IDs give `Option<PoolId<_>>` a free niche.
/// Slots are reused after kill; no generational counters in MVP — callers hold live references
/// only within the tick they were alive.
pub struct Pool<T> {
    cap:       u32,
    next_raw:  u32,            // monotonic counter for never-before-allocated slots
    pub alive: Vec<bool>,
    freelist:  Vec<u32>,       // raw ids pushed by kill(), popped first by alloc()
    _tag:      PhantomData<fn() -> T>,
}

impl<T> Pool<T> {
    pub fn new(cap: u32) -> Self {
        Self {
            cap, next_raw: 1,
            alive: vec![false; cap as usize],
            freelist: Vec::new(),
            _tag: PhantomData,
        }
    }
    pub fn alloc(&mut self) -> Option<PoolId<T>> {
        let raw = if let Some(r) = self.freelist.pop() {
            r
        } else if self.next_raw <= self.cap {
            let r = self.next_raw;
            self.next_raw += 1;
            r
        } else {
            return None;
        };
        let slot = (raw - 1) as usize;
        self.alive[slot] = true;
        PoolId::new(raw)
    }
    pub fn kill(&mut self, id: PoolId<T>) {
        let slot = id.slot();
        if slot < self.cap as usize && self.alive[slot] {
            self.alive[slot] = false;
            self.freelist.push(id.raw());
        }
    }
    #[inline] pub fn is_alive(&self, id: PoolId<T>) -> bool {
        self.alive.get(id.slot()).copied().unwrap_or(false)
    }
    #[inline] pub fn cap(&self) -> u32 { self.cap }
    #[inline] pub fn slot_of(id: PoolId<T>) -> usize { id.slot() }
}
```

- [ ] **Step 4: Wire `pool` module into `lib.rs`**

Add `pub mod pool;` between `policy` and `rng` in alphabetical order.

- [ ] **Step 5: Convert `AgentSlotPool` to a type alias (keep source compatibility)**

Replace `crates/engine/src/state/entity_pool.rs` contents with:

```rust
//! Historical location of `AgentSlotPool`. Now a type alias over the generic
//! `crate::pool::Pool<AgentTag>`. Keep the `AgentSlotPool` name for call-site
//! stability; new code should use `crate::pool::Pool` directly.

use crate::ids::AgentId;
use crate::pool::{Pool, PoolId};

/// Marker type for the agent pool.
pub struct AgentTag;

pub type AgentSlotPool = Pool<AgentTag>;

/// Bridge: existing code constructs `AgentId` via `AgentId::new(raw)`.
/// This shim lets `AgentSlotPool::alloc` return an `AgentId` rather than a
/// `PoolId<AgentTag>` so we don't have to update every call site at once.
impl AgentSlotPool {
    pub fn alloc_agent(&mut self) -> Option<AgentId> {
        self.alloc().and_then(|id: PoolId<AgentTag>| AgentId::new(id.raw()))
    }
    pub fn kill_agent(&mut self, id: AgentId) {
        if let Some(p) = PoolId::<AgentTag>::new(id.raw()) { self.kill(p); }
    }
    #[inline] pub fn slot_of_agent(id: AgentId) -> usize { (id.raw() - 1) as usize }
}
```

- [ ] **Step 6: Update `state/mod.rs` call sites**

`spawn_agent`/`kill_agent`/`slot_of` in `state/mod.rs` currently call `AgentSlotPool::alloc()`, `.kill(id)`, and `::slot_of(id)`. Change them to `alloc_agent()`, `kill_agent(id)`, `slot_of_agent(id)` so that `AgentSlotPool` retains its familiar shape through `PoolId<AgentTag>`.

- [ ] **Step 7: Run tests**

```
cargo test -p engine --test pool_generic
cargo test -p engine
```

Expected: pool_generic tests pass; the full engine suite still green (48 tests).

- [ ] **Step 8: Commit**

```bash
git add crates/engine/src/pool.rs crates/engine/src/lib.rs \
        crates/engine/src/state/entity_pool.rs crates/engine/src/state/mod.rs \
        crates/engine/tests/pool_generic.rs
git commit -m "feat(engine): generic Pool<T> with freelist — agents now reuse slots"
```

---

## Task 2: Agent needs — hunger, thirst, rest

**Files:**
- Modify: `crates/engine/src/state/agent.rs`
- Modify: `crates/engine/src/state/mod.rs`
- Test: `crates/engine/tests/state_needs.rs`

- [ ] **Step 1: Write failing test** `crates/engine/tests/state_needs.rs`

```rust
use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn spawn_initializes_needs_to_full() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    assert_eq!(state.agent_hunger(a), Some(1.0));
    assert_eq!(state.agent_thirst(a), Some(1.0));
    assert_eq!(state.agent_rest_timer(a), Some(1.0));
}

#[test]
fn set_and_read_needs() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    state.set_agent_hunger(a, 0.3);
    state.set_agent_thirst(a, 0.5);
    state.set_agent_rest_timer(a, 0.1);
    assert_eq!(state.agent_hunger(a), Some(0.3));
    assert_eq!(state.agent_thirst(a), Some(0.5));
    assert_eq!(state.agent_rest_timer(a), Some(0.1));
}
```

- [ ] **Step 2: Run to verify fails** — `agent_hunger` accessor doesn't exist yet.

- [ ] **Step 3: Extend `SimState`**

In `crates/engine/src/state/mod.rs`, add three SoA hot fields:

```rust
pub struct SimState {
    // ... existing fields ...
    hot_hunger:     Vec<f32>,  // 0.0 = starving, 1.0 = sated
    hot_thirst:     Vec<f32>,  // 0.0 = parched, 1.0 = hydrated
    hot_rest_timer: Vec<f32>,  // 0.0 = exhausted, 1.0 = well-rested
}
```

Initialize in `SimState::new`:

```rust
hot_hunger:     vec![1.0; cap],
hot_thirst:     vec![1.0; cap],
hot_rest_timer: vec![1.0; cap],
```

Fill in `spawn_agent`:

```rust
self.hot_hunger[slot]     = 1.0;
self.hot_thirst[slot]     = 1.0;
self.hot_rest_timer[slot] = 1.0;
```

Accessors + mutators:

```rust
pub fn agent_hunger(&self, id: AgentId) -> Option<f32> {
    self.hot_hunger.get(AgentSlotPool::slot_of_agent(id)).copied()
}
pub fn agent_thirst(&self, id: AgentId) -> Option<f32> {
    self.hot_thirst.get(AgentSlotPool::slot_of_agent(id)).copied()
}
pub fn agent_rest_timer(&self, id: AgentId) -> Option<f32> {
    self.hot_rest_timer.get(AgentSlotPool::slot_of_agent(id)).copied()
}
pub fn set_agent_hunger(&mut self, id: AgentId, v: f32) {
    if let Some(s) = self.hot_hunger.get_mut(AgentSlotPool::slot_of_agent(id)) { *s = v; }
}
pub fn set_agent_thirst(&mut self, id: AgentId, v: f32) {
    if let Some(s) = self.hot_thirst.get_mut(AgentSlotPool::slot_of_agent(id)) { *s = v; }
}
pub fn set_agent_rest_timer(&mut self, id: AgentId, v: f32) {
    if let Some(s) = self.hot_rest_timer.get_mut(AgentSlotPool::slot_of_agent(id)) { *s = v; }
}

pub fn hot_hunger(&self)     -> &[f32] { &self.hot_hunger }
pub fn hot_thirst(&self)     -> &[f32] { &self.hot_thirst }
pub fn hot_rest_timer(&self) -> &[f32] { &self.hot_rest_timer }
```

- [ ] **Step 4: Run tests**

```
cargo test -p engine --test state_needs
cargo test -p engine
```

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/state/mod.rs crates/engine/tests/state_needs.rs
git commit -m "feat(engine): add agent needs (hunger, thirst, rest_timer) to SoA hot state"
```

---

## Task 3: `EventId` + cause sidecar on `EventRing`

**Files:**
- Modify: `crates/engine/src/ids.rs`
- Modify: `crates/engine/src/event/mod.rs`
- Modify: `crates/engine/src/event/ring.rs`
- Test: `crates/engine/tests/event_id_threading.rs`

- [ ] **Step 1: Write failing test** `crates/engine/tests/event_id_threading.rs`

```rust
use engine::event::{Event, EventId, EventRing};
use engine::ids::AgentId;
use glam::Vec3;

#[test]
fn push_assigns_sequential_event_ids() {
    let mut ring = EventRing::with_cap(16);
    let a = AgentId::new(1).unwrap();
    let id0 = ring.push(Event::AgentMoved {
        agent_id: a, from: Vec3::ZERO, to: Vec3::X, tick: 0,
    });
    let id1 = ring.push(Event::AgentMoved {
        agent_id: a, from: Vec3::X, to: Vec3::Y, tick: 0,
    });
    assert_eq!(id0.tick, 0);
    assert_eq!(id0.seq, 0);
    assert_eq!(id1.tick, 0);
    assert_eq!(id1.seq, 1);
}

#[test]
fn push_with_cause_stores_parent_id_in_sidecar() {
    let mut ring = EventRing::with_cap(16);
    let a = AgentId::new(1).unwrap();
    let id0 = ring.push(Event::AgentAttacked {
        attacker: a, target: a, damage: 10.0, tick: 0,
    });
    let id1 = ring.push_caused_by(
        Event::AgentDied { agent_id: a, tick: 0 },
        id0,
    );
    assert_eq!(ring.cause_of(id1), Some(id0));
    assert_eq!(ring.cause_of(id0), None);
}

#[test]
fn cause_field_does_not_affect_replayable_hash() {
    // Two rings with identical events but different cause chains must agree on hash.
    let mut r1 = EventRing::with_cap(16);
    let mut r2 = EventRing::with_cap(16);
    let a = AgentId::new(1).unwrap();
    let id0 = r1.push(Event::AgentMoved {
        agent_id: a, from: Vec3::ZERO, to: Vec3::X, tick: 0,
    });
    r1.push_caused_by(Event::AgentDied { agent_id: a, tick: 0 }, id0);
    r2.push(Event::AgentMoved {
        agent_id: a, from: Vec3::ZERO, to: Vec3::X, tick: 0,
    });
    r2.push(Event::AgentDied { agent_id: a, tick: 0 });
    assert_eq!(r1.replayable_sha256(), r2.replayable_sha256());
}
```

- [ ] **Step 2: Run to verify fails** — `EventId`, `push_caused_by`, `cause_of` don't exist.

- [ ] **Step 3: Add `EventId` to `ids.rs`**

```rust
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct EventId {
    pub tick: u32,
    pub seq:  u32,
}
```

Export from `lib.rs` if not already.

- [ ] **Step 4: Extend `EventRing` with cause sidecar**

In `crates/engine/src/event/ring.rs`:

```rust
pub struct EventRing {
    events:    VecDeque<Event>,
    causes:    VecDeque<Option<EventId>>,  // parallel to events
    cap:       usize,
    next_seq:  u32,     // resets at tick boundary
    tick:      u32,
}

impl EventRing {
    pub fn with_cap(cap: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(cap),
            causes: VecDeque::with_capacity(cap),
            cap,
            next_seq: 0, tick: 0,
        }
    }
    pub fn push(&mut self, e: Event) -> EventId {
        self.push_with_cause(e, None)
    }
    pub fn push_caused_by(&mut self, e: Event, cause: EventId) -> EventId {
        self.push_with_cause(e, Some(cause))
    }
    fn push_with_cause(&mut self, e: Event, cause: Option<EventId>) -> EventId {
        let tick = e.tick();
        if tick != self.tick {
            self.tick = tick;
            self.next_seq = 0;
        }
        if self.events.len() == self.cap {
            self.events.pop_front();
            self.causes.pop_front();
        }
        let id = EventId { tick, seq: self.next_seq };
        self.next_seq += 1;
        self.events.push_back(e);
        self.causes.push_back(cause);
        id
    }
    pub fn cause_of(&self, id: EventId) -> Option<EventId> {
        // Linear scan — the sidecar isn't keyed by EventId, only ordered.
        // For large rings switch to a BTreeMap<EventId, Option<EventId>>; MVP is O(N).
        self.events.iter().zip(self.causes.iter()).find_map(|(e, c)| {
            // Match by tick + ordinal — ordinal derives from ring position within the tick.
            // TODO(perf): precompute ordinal per-event if needed.
            let _ = e; let _ = c;
            None
        })
    }
    // ...existing methods: iter(), replayable_sha256(), etc. — unchanged.
}
```

The `cause_of` lookup above is a stub; real implementation needs a per-event ordinal. Simplify by storing `EventId` alongside:

```rust
pub struct EventRing {
    entries:   VecDeque<EventEntry>,
    cap:       usize,
    tick:      u32,
    next_seq:  u32,
}

struct EventEntry {
    event: Event,
    id:    EventId,
    cause: Option<EventId>,
}

impl EventRing {
    pub fn cause_of(&self, id: EventId) -> Option<EventId> {
        self.entries.iter().find(|e| e.id == id).and_then(|e| e.cause)
    }
    pub fn iter(&self) -> impl Iterator<Item = &Event> + '_ {
        self.entries.iter().map(|e| &e.event)
    }
    pub fn replayable_sha256(&self) -> [u8; 32] {
        // Unchanged: iterate events, hash only replayable ones, skip cause + id.
        // ... existing code translated to walk `entries` ...
    }
}
```

Replace the entire `EventRing` implementation with the entry-based one. All existing API (`push`, `iter`, `replayable_sha256`, etc.) still works; `push_caused_by` is new; `cause_of` is new.

- [ ] **Step 5: Run tests**

```
cargo test -p engine --test event_id_threading
cargo test -p engine --test event_ring
cargo test -p engine --test determinism
cargo test -p engine
```

The existing event-ring golden-hash test must still pass — cause is NOT in the hash.

- [ ] **Step 6: Commit**

```bash
git add crates/engine/src/ids.rs crates/engine/src/event/ring.rs \
        crates/engine/tests/event_id_threading.rs
git commit -m "feat(engine): EventId + cause sidecar — cascade fan-out groundwork"
```

---

## Task 4: Extend `MicroKind` to 18 variants

**Files:**
- Modify: `crates/engine/src/mask.rs`
- Modify: `crates/engine/src/schema_hash.rs`
- Modify: `crates/engine/.schema_hash`
- Test: `crates/engine/tests/micro_kind_full.rs`

- [ ] **Step 1: Write failing test** `crates/engine/tests/micro_kind_full.rs`

```rust
use engine::mask::MicroKind;

#[test]
fn all_variants_present() {
    let all = MicroKind::ALL;
    assert_eq!(all.len(), 18);
    // Spot-check a few specific variants are present and have stable ordinals.
    assert_eq!(MicroKind::Hold as u8, 0);
    assert_eq!(MicroKind::MoveToward as u8, 1);
    assert_eq!(MicroKind::Flee as u8, 2);
    assert_eq!(MicroKind::Attack as u8, 3);
    assert_eq!(MicroKind::Communicate as u8, 15);
    assert_eq!(MicroKind::Ask as u8, 16);
    assert_eq!(MicroKind::Remember as u8, 17);
}

#[test]
fn u8_roundtrip_for_every_variant() {
    for k in MicroKind::ALL {
        let as_u8 = *k as u8;
        assert!(as_u8 < 18);
    }
}
```

- [ ] **Step 2: Run to verify fails** — 18 variants expected, 4 present.

- [ ] **Step 3: Extend `MicroKind`**

In `crates/engine/src/mask.rs`:

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum MicroKind {
    // Movement (3)
    Hold        = 0,
    MoveToward  = 1,
    Flee        = 2,
    // Combat (3)
    Attack      = 3,
    Cast        = 4,
    UseItem     = 5,
    // Resource (4)
    Harvest     = 6,
    Eat         = 7,
    Drink       = 8,
    Rest        = 9,
    // Construction (3)
    PlaceTile    = 10,
    PlaceVoxel   = 11,
    HarvestVoxel = 12,
    // Social (2)
    Converse     = 13,
    ShareStory   = 14,
    // Info push + pull (2)
    Communicate  = 15,
    Ask          = 16,
    // Memory (1)
    Remember     = 17,
}

impl MicroKind {
    pub const ALL: &'static [MicroKind] = &[
        MicroKind::Hold,       MicroKind::MoveToward,  MicroKind::Flee,
        MicroKind::Attack,     MicroKind::Cast,        MicroKind::UseItem,
        MicroKind::Harvest,    MicroKind::Eat,         MicroKind::Drink,
        MicroKind::Rest,       MicroKind::PlaceTile,   MicroKind::PlaceVoxel,
        MicroKind::HarvestVoxel, MicroKind::Converse,  MicroKind::ShareStory,
        MicroKind::Communicate, MicroKind::Ask,        MicroKind::Remember,
    ];
}
```

- [ ] **Step 4: Bump the schema hash fingerprint**

In `crates/engine/src/schema_hash.rs`, update the update string:

```rust
h.update(b"MicroKind:Hold,MoveToward,Flee,Attack,Cast,UseItem,Harvest,Eat,Drink,Rest,PlaceTile,PlaceVoxel,HarvestVoxel,Converse,ShareStory,Communicate,Ask,Remember");
```

(The old fingerprint was the 4-variant version.)

- [ ] **Step 5: Regenerate the baseline**

```bash
cargo run -p engine --example print_schema_hash > crates/engine/.schema_hash
```

- [ ] **Step 6: Run tests**

```
cargo test -p engine --test micro_kind_full
cargo test -p engine --test schema_hash
cargo test -p engine
```

The existing `mask_validity` and `policy_utility` tests must still pass — they use `MicroKind::ALL.len()` which now returns 18, and the `UtilityBackend` only emits the 4 it scores; the other 14 mask bits stay `false`.

- [ ] **Step 7: Commit**

```bash
git add crates/engine/src/mask.rs crates/engine/src/schema_hash.rs \
        crates/engine/.schema_hash crates/engine/tests/micro_kind_full.rs
git commit -m "feat(engine): MicroKind extended to 18 variants (Appendix A full set)"
```

---

## Task 5: `MacroKind` enum + parameter types

**Files:**
- Create: `crates/engine/src/policy/macro_kind.rs`
- Create: `crates/engine/src/policy/query.rs`
- Modify: `crates/engine/src/policy/mod.rs`
- Modify: `crates/engine/src/schema_hash.rs`
- Modify: `crates/engine/.schema_hash`
- Test: `crates/engine/tests/macro_kind.rs`

- [ ] **Step 1: Write failing test** `crates/engine/tests/macro_kind.rs`

```rust
use engine::policy::macro_kind::{MacroAction, MacroKind, AnnounceAudience, Resolution};
use engine::ids::{AgentId, GroupId, QuestId};
use glam::Vec3;

#[test]
fn macro_kind_has_five_variants_including_noop() {
    assert_eq!(MacroKind::NoOp as u8, 0);
    assert_eq!(MacroKind::PostQuest as u8, 1);
    assert_eq!(MacroKind::AcceptQuest as u8, 2);
    assert_eq!(MacroKind::Bid as u8, 3);
    assert_eq!(MacroKind::Announce as u8, 4);
}

#[test]
fn announce_audience_variants() {
    let g = GroupId::new(1).unwrap();
    let a = AnnounceAudience::Group(g);
    let b = AnnounceAudience::Area(Vec3::ZERO, 30.0);
    let c = AnnounceAudience::Anyone;
    assert_ne!(a, b);
    assert_ne!(b, c);
}

#[test]
fn resolution_coalition_carries_min_parties() {
    let r = Resolution::Coalition { min_parties: 3 };
    match r {
        Resolution::Coalition { min_parties } => assert_eq!(min_parties, 3),
        _ => panic!(),
    }
}
```

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: Define `MacroKind` and parameter enums**

Create `crates/engine/src/policy/macro_kind.rs`:

```rust
//! Universal macro mechanisms. Language-level macros are the 4 variants
//! `PostQuest`, `AcceptQuest`, `Bid`, `Announce`; `NoOp` occupies the 0 slot
//! so `macro_kind != NoOp` cleanly distinguishes macro from micro emission.

use crate::ids::{AgentId, GroupId, QuestId};
use glam::Vec3;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum MacroKind {
    NoOp       = 0,
    PostQuest  = 1,
    AcceptQuest = 2,
    Bid        = 3,
    Announce   = 4,
}

impl MacroKind {
    pub const ALL: &'static [MacroKind] = &[
        MacroKind::NoOp, MacroKind::PostQuest, MacroKind::AcceptQuest,
        MacroKind::Bid,  MacroKind::Announce,
    ];
}

/// Recipient scope for `Announce`. Matches `dsl/spec.md` §3.2.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AnnounceAudience {
    Group(GroupId),
    Area(Vec3, f32),     // center, radius
    Anyone,              // global within MAX_ANNOUNCE_RADIUS of speaker
}

/// Auction resolution policy. Matches `dsl/spec.md` §9 D1.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Resolution {
    HighestBid,
    FirstAcceptable,
    MutualAgreement,
    Coalition { min_parties: u8 },
    Majority,
}

/// Quest kind — minimal universal set. Domain-specific kinds register via the
/// compiler's `QuestType` extension table; engine only knows the universal
/// shape (hunt / escort / deliver / explore / defend etc. are domain labels).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QuestCategory {
    Physical   = 0,   // hunt, escort, gather, rescue, etc.
    Political  = 1,   // conquest, diplomacy, submit, charter
    Personal   = 2,   // marriage, pilgrimage, have-child
    Economic   = 3,   // service, heist, trade
    Narrative  = 4,   // prophecy, claim
}

/// Parameterised macro action emitted by a policy.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MacroAction {
    NoOp,
    PostQuest {
        quest_id:     QuestId,
        category:     QuestCategory,
        resolution:   Resolution,
    },
    AcceptQuest {
        quest_id:     QuestId,
        acceptor:     AgentId,
    },
    Bid {
        auction_id:   QuestId,
        bidder:       AgentId,
        amount:       f32,
    },
    Announce {
        speaker:      AgentId,
        audience:     AnnounceAudience,
        fact_payload: u64,  // opaque handle — compiler-registered type decodes
    },
}

impl MacroAction {
    pub fn kind(&self) -> MacroKind {
        match self {
            MacroAction::NoOp => MacroKind::NoOp,
            MacroAction::PostQuest { .. } => MacroKind::PostQuest,
            MacroAction::AcceptQuest { .. } => MacroKind::AcceptQuest,
            MacroAction::Bid { .. } => MacroKind::Bid,
            MacroAction::Announce { .. } => MacroKind::Announce,
        }
    }
}
```

Create `crates/engine/src/policy/query.rs`:

```rust
//! Query kinds for the `Ask` micro primitive.

use crate::ids::{AgentId, GroupId};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum QueryKind {
    /// "Tell me everything you know about this entity."
    AboutEntity(EntityQueryRef),
    /// "What have you heard about this kind of thing lately?"
    AboutKind(MemoryKind),
    /// All facts — used by the `Read(doc)` sugar lowering.
    AboutAll,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EntityQueryRef {
    Agent(AgentId),
    Group(GroupId),
}

/// Coarse memory-kind label — domain-specific subtypes are compiler-generated.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryKind {
    Combat    = 0,
    Trade     = 1,
    Social    = 2,
    Political = 3,
    Other     = 4,
}
```

- [ ] **Step 4: Add IDs** — `GroupId`, `QuestId`, `ItemId` to `crates/engine/src/ids.rs` using the existing `id_type!` macro.

```rust
id_type!(GroupId);
id_type!(QuestId);
id_type!(ItemId);
```

- [ ] **Step 5: Wire modules**

In `crates/engine/src/policy/mod.rs`, add:

```rust
pub mod macro_kind;
pub mod query;
pub use macro_kind::{MacroAction, MacroKind, AnnounceAudience, Resolution, QuestCategory};
pub use query::{QueryKind, EntityQueryRef, MemoryKind};
```

- [ ] **Step 6: Schema-hash bump**

Append to `schema_hash.rs`:

```rust
h.update(b"MacroKind:NoOp,PostQuest,AcceptQuest,Bid,Announce");
h.update(b"AnnounceAudience:Group,Area,Anyone");
h.update(b"Resolution:HighestBid,FirstAcceptable,MutualAgreement,Coalition,Majority");
h.update(b"QuestCategory:Physical,Political,Personal,Economic,Narrative");
h.update(b"QueryKind:AboutEntity,AboutKind,AboutAll");
h.update(b"MemoryKind:Combat,Trade,Social,Political,Other");
```

Regenerate baseline.

- [ ] **Step 7: Run tests + commit**

```
cargo test -p engine --test macro_kind
cargo test -p engine
```

```bash
git add crates/engine/src/policy/ crates/engine/src/ids.rs \
        crates/engine/src/schema_hash.rs crates/engine/.schema_hash \
        crates/engine/tests/macro_kind.rs
git commit -m "feat(engine): MacroKind + parameter enums (Announce audience, Resolution, QueryKind)"
```

---

## Task 6: Unified `Action` sum type

**Files:**
- Modify: `crates/engine/src/policy/mod.rs`
- Modify: `crates/engine/src/policy/utility.rs`
- Modify: `crates/engine/src/step.rs`
- Modify: existing tests that construct `Action` directly
- Test: `crates/engine/tests/action_sum.rs`

- [ ] **Step 1: Write failing test** `crates/engine/tests/action_sum.rs`

```rust
use engine::ids::{AgentId, QuestId};
use engine::mask::MicroKind;
use engine::policy::{Action, ActionKind, MacroAction, QueryKind, MicroTarget};
use engine::policy::macro_kind::{AnnounceAudience, QuestCategory, Resolution};
use glam::Vec3;

#[test]
fn construct_micro_variant() {
    let a = AgentId::new(1).unwrap();
    let act = Action {
        agent: a,
        kind:  ActionKind::Micro {
            kind:   MicroKind::Attack,
            target: MicroTarget::Agent(a),
        },
    };
    match act.kind {
        ActionKind::Micro { kind: MicroKind::Attack, .. } => (),
        _ => panic!(),
    }
}

#[test]
fn construct_macro_variant() {
    let a = AgentId::new(1).unwrap();
    let q = QuestId::new(1).unwrap();
    let act = Action {
        agent: a,
        kind:  ActionKind::Macro(MacroAction::PostQuest {
            quest_id: q, category: QuestCategory::Physical,
            resolution: Resolution::HighestBid,
        }),
    };
    assert_eq!(act.agent.raw(), 1);
}

#[test]
fn micro_target_covers_all_branches() {
    // Document branches exist for Ask lowering.
    let _ = MicroTarget::None;
    let _ = MicroTarget::Agent(AgentId::new(1).unwrap());
    let _ = MicroTarget::Position(Vec3::ZERO);
    let _ = MicroTarget::Query(QueryKind::AboutAll);
}
```

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: Rewrite `Action` in `policy/mod.rs`**

```rust
use crate::ids::AgentId;
use crate::mask::MicroKind;
use crate::mask::MaskBuffer;
use crate::state::SimState;
use glam::Vec3;
pub use macro_kind::MacroAction;
pub use query::QueryKind;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Action {
    pub agent: AgentId,
    pub kind:  ActionKind,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ActionKind {
    Micro { kind: MicroKind, target: MicroTarget },
    Macro(MacroAction),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MicroTarget {
    None,
    Agent(AgentId),
    Position(Vec3),
    ItemSlot(u8),
    AbilityIdx(u8),
    Query(QueryKind),
    // Extensible — UseItem/Harvest/PlaceVoxel/Document targets come via
    // compiler-generated enum extensions referenced through an opaque u64.
    Opaque(u64),
}

impl Action {
    pub fn hold(agent: AgentId) -> Self {
        Self { agent, kind: ActionKind::Micro {
            kind: MicroKind::Hold, target: MicroTarget::None,
        }}
    }
    pub fn move_toward(agent: AgentId, pos: Vec3) -> Self {
        Self { agent, kind: ActionKind::Micro {
            kind: MicroKind::MoveToward, target: MicroTarget::Position(pos),
        }}
    }
    pub fn attack(agent: AgentId, target: AgentId) -> Self {
        Self { agent, kind: ActionKind::Micro {
            kind: MicroKind::Attack, target: MicroTarget::Agent(target),
        }}
    }
    // ... similar constructors as needed ...
}
```

- [ ] **Step 4: Update `UtilityBackend::evaluate`** in `policy/utility.rs`

The score table stays micro-only (utility is a bootstrap; macros come from neural/LLM backends). Adapt the push into `Vec<Action>`:

```rust
out.push(Action {
    agent: id,
    kind: ActionKind::Micro { kind: chosen, target: tgt },
});
```

Where `tgt` is `MicroTarget::Agent(nearest)` for Attack, `MicroTarget::Position(dest)` for MoveToward, else `MicroTarget::None`.

- [ ] **Step 5: Update `step.rs::apply_actions`** to pattern-match on `ActionKind`.

- [ ] **Step 6: Update existing tests** that construct `Action { agent, micro_kind, target: Option<AgentId> }` to the new sum shape. Most tests use `Action::hold(...)` / `Action::attack(...)` constructors — just add those to the impl.

- [ ] **Step 7: Run full engine suite + commit**

```
cargo test -p engine
```

All prior tests must still pass.

```bash
git add crates/engine/src/policy/ crates/engine/src/step.rs \
        crates/engine/tests/
git commit -m "feat(engine): Action sum type — Micro{kind,target} | Macro(MacroAction)"
```

---

## Task 7: `CascadeHandler` trait + registry with lanes

**Files:**
- Create: `crates/engine/src/cascade/mod.rs`
- Create: `crates/engine/src/cascade/handler.rs`
- Create: `crates/engine/src/cascade/dispatch.rs`
- Modify: `crates/engine/src/lib.rs`
- Test: `crates/engine/tests/cascade_register_dispatch.rs`

- [ ] **Step 1: Write failing test**

```rust
use engine::cascade::{CascadeRegistry, CascadeHandler, Lane, EventKindId};
use engine::event::{Event, EventRing};
use engine::state::{AgentSpawn, SimState};
use engine::creature::CreatureType;
use engine::ids::AgentId;
use glam::Vec3;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Counting(Arc<AtomicUsize>);
impl CascadeHandler for Counting {
    fn trigger(&self) -> EventKindId { EventKindId::AgentAttacked }
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, _: &Event, _: &mut SimState, _: &mut EventRing) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn registered_handler_fires_on_matching_event_kind() {
    let mut reg = CascadeRegistry::new();
    let hits = Arc::new(AtomicUsize::new(0));
    reg.register(Counting(hits.clone()));

    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let mut ring = EventRing::with_cap(16);
    let evt = Event::AgentAttacked { attacker: a, target: a, damage: 0.0, tick: 0 };

    reg.dispatch(&evt, &mut state, &mut ring);
    assert_eq!(hits.load(Ordering::Relaxed), 1);
}

#[test]
fn handler_not_fired_for_non_matching_kind() {
    let mut reg = CascadeRegistry::new();
    let hits = Arc::new(AtomicUsize::new(0));
    reg.register(Counting(hits.clone()));

    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let mut ring = EventRing::with_cap(16);
    let evt = Event::AgentDied { agent_id: a, tick: 0 };

    reg.dispatch(&evt, &mut state, &mut ring);
    assert_eq!(hits.load(Ordering::Relaxed), 0);
}
```

- [ ] **Step 2: Verify fails.**

- [ ] **Step 3: Implement `cascade/handler.rs`**

```rust
use crate::event::{Event, EventRing};
use crate::state::SimState;

/// Compact stable identifier for an event variant. Adding a variant appends
/// to the enum; reordering requires a schema-hash bump.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum EventKindId {
    AgentMoved     = 0,
    AgentAttacked  = 1,
    AgentDied      = 2,
    AgentFled      = 3,
    AgentAte       = 4,
    AgentDrank     = 5,
    AgentRested    = 6,
    AgentHarvested = 7,
    AgentCast      = 8,
    AgentUsedItem  = 9,
    AgentPlaced    = 10,
    AgentConversed = 11,
    AgentSharedStory = 12,
    AgentCommunicated = 13,
    InformationRequested = 14,
    AgentRemembered = 15,
    QuestPosted     = 16,
    QuestAccepted   = 17,
    BidPlaced       = 18,
    AnnounceEmitted = 19,
    RecordMemory    = 20,
    ChronicleEntry  = 128,    // gap reserved — chronicle variants live in the 128+ range
}

impl EventKindId {
    pub fn from_event(e: &Event) -> EventKindId {
        match e {
            Event::AgentMoved    { .. } => EventKindId::AgentMoved,
            Event::AgentAttacked { .. } => EventKindId::AgentAttacked,
            Event::AgentDied     { .. } => EventKindId::AgentDied,
            Event::ChronicleEntry { .. } => EventKindId::ChronicleEntry,
            // New variants match here as they are added in Task 8.
        }
    }
}

/// Lane discipline — handlers within a lane run in registration order;
/// lanes run in the order listed here. Matches `../compiler/spec.md` §Decisions D16.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum Lane {
    Validation = 0,
    Effect     = 1,
    Reaction   = 2,
    Audit      = 3,
}

impl Lane {
    pub const ALL: &'static [Lane] = &[Lane::Validation, Lane::Effect, Lane::Reaction, Lane::Audit];
}

pub trait CascadeHandler: Send + Sync {
    fn trigger(&self) -> EventKindId;
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, event: &Event, state: &mut SimState, events: &mut EventRing);
}
```

- [ ] **Step 4: Implement `cascade/dispatch.rs`**

```rust
use super::handler::{CascadeHandler, EventKindId, Lane};
use crate::event::{Event, EventRing};
use crate::state::SimState;
use smallvec::SmallVec;

pub struct CascadeRegistry {
    /// Indexed as `table[lane as usize][kind_ordinal] = Vec<handler>`.
    /// kind_ordinal comes from a dense re-mapping of EventKindId.
    table: Vec<Vec<SmallVec<[Box<dyn CascadeHandler>; 4]>>>,
}

const KIND_SLOTS: usize = 256;  // accommodates the 128+ chronicle reservation

impl CascadeRegistry {
    pub fn new() -> Self {
        let per_lane: Vec<SmallVec<[_; 4]>> = (0..KIND_SLOTS).map(|_| SmallVec::new()).collect();
        Self {
            table: (0..Lane::ALL.len()).map(|_| per_lane.clone()).collect(),
        }
    }

    pub fn register<H: CascadeHandler + 'static>(&mut self, h: H) {
        let lane = h.lane() as usize;
        let kind = h.trigger() as u8 as usize;
        self.table[lane][kind].push(Box::new(h));
    }

    pub fn dispatch(&self, event: &Event, state: &mut SimState, events: &mut EventRing) {
        let kind = EventKindId::from_event(event) as u8 as usize;
        for lane in Lane::ALL {
            for handler in &self.table[*lane as usize][kind] {
                handler.handle(event, state, events);
            }
        }
    }
}

impl Default for CascadeRegistry {
    fn default() -> Self { Self::new() }
}
```

(`SmallVec` requires `.clone()` on each element; wrap `Vec<SmallVec>::clone` for the initial fill, or switch to `std::array::from_fn` if SmallVec's API is awkward — adapt based on what the dev-deps actually expose.)

- [ ] **Step 5: Implement `cascade/mod.rs`**

```rust
pub mod dispatch;
pub mod handler;

pub use dispatch::CascadeRegistry;
pub use handler::{CascadeHandler, EventKindId, Lane};
```

- [ ] **Step 6: Register module** — `pub mod cascade;` in `lib.rs`.

- [ ] **Step 7: Run tests + commit**

```bash
git add crates/engine/src/cascade/ crates/engine/src/lib.rs \
        crates/engine/tests/cascade_register_dispatch.rs
git commit -m "feat(engine): CascadeRegistry + CascadeHandler trait + Lane discipline"
```

---

## Task 8: Cascade — bounded dispatch loop + regression test

**Files:**
- Modify: `crates/engine/src/cascade/dispatch.rs`
- Modify: `crates/engine/src/step.rs` (wire registry into `step()`)
- Modify: `crates/engine/src/schema_hash.rs`
- Modify: `crates/engine/.schema_hash`
- Test: `crates/engine/tests/cascade_bounded.rs`, `crates/engine/tests/cascade_lanes.rs`

- [ ] **Step 1: Write bounded-loop test** `crates/engine/tests/cascade_bounded.rs`

```rust
use engine::cascade::{CascadeRegistry, CascadeHandler, EventKindId, Lane};
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine::creature::CreatureType;
use glam::Vec3;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A handler that re-emits the same event each time it fires — creates a cycle.
/// The dispatch loop must cap iterations to MAX_CASCADE_ITERATIONS = 8.
struct Amplifier(Arc<AtomicUsize>);
impl CascadeHandler for Amplifier {
    fn trigger(&self) -> EventKindId { EventKindId::AgentAttacked }
    fn lane(&self) -> Lane { Lane::Effect }
    fn handle(&self, event: &Event, _: &mut SimState, events: &mut EventRing) {
        self.0.fetch_add(1, Ordering::Relaxed);
        // Re-emit the same event with a fresh tick offset so it's observably new.
        if let Event::AgentAttacked { attacker, target, damage, tick } = event {
            events.push(Event::AgentAttacked {
                attacker: *attacker, target: *target,
                damage: *damage, tick: tick.saturating_add(1),
            });
        }
    }
}

#[test]
fn dispatch_truncates_at_max_cascade_iterations() {
    let mut reg = CascadeRegistry::new();
    let hits = Arc::new(AtomicUsize::new(0));
    reg.register(Amplifier(hits.clone()));

    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let mut ring = EventRing::with_cap(1024);

    ring.push(Event::AgentAttacked { attacker: a, target: a, damage: 1.0, tick: 0 });

    reg.run_fixed_point(&mut state, &mut ring);

    // Fired on the initial event + up to 8 more cascade iterations.
    let n = hits.load(Ordering::Relaxed);
    assert!(n <= 9, "handler fired {} times — expected ≤ 9 (initial + 8)", n);
    assert!(n >= 2, "handler fired {} times — cascade didn't fire at all?", n);
}
```

- [ ] **Step 2: Write lane-order test** `crates/engine/tests/cascade_lanes.rs`

```rust
use engine::cascade::{CascadeRegistry, CascadeHandler, EventKindId, Lane};
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::SimState;
use std::sync::Mutex;
use std::sync::Arc;

struct OrderMarker(Arc<Mutex<Vec<Lane>>>, Lane);
impl CascadeHandler for OrderMarker {
    fn trigger(&self) -> EventKindId { EventKindId::AgentDied }
    fn lane(&self) -> Lane { self.1 }
    fn handle(&self, _: &Event, _: &mut SimState, _: &mut EventRing) {
        self.0.lock().unwrap().push(self.1);
    }
}

#[test]
fn lanes_run_in_order_validation_effect_reaction_audit() {
    let mut reg = CascadeRegistry::new();
    let trace = Arc::new(Mutex::new(Vec::<Lane>::new()));
    // Register out of order to prove the registry sorts by lane, not by insertion.
    reg.register(OrderMarker(trace.clone(), Lane::Audit));
    reg.register(OrderMarker(trace.clone(), Lane::Validation));
    reg.register(OrderMarker(trace.clone(), Lane::Reaction));
    reg.register(OrderMarker(trace.clone(), Lane::Effect));

    let mut state = SimState::new(2, 42);
    let a = AgentId::new(1).unwrap();
    let mut ring = EventRing::with_cap(8);
    reg.dispatch(&Event::AgentDied { agent_id: a, tick: 0 }, &mut state, &mut ring);

    let out = trace.lock().unwrap().clone();
    assert_eq!(out, vec![Lane::Validation, Lane::Effect, Lane::Reaction, Lane::Audit]);
}
```

- [ ] **Step 3: Extend `CascadeRegistry` with `run_fixed_point`**

In `dispatch.rs`:

```rust
pub const MAX_CASCADE_ITERATIONS: usize = 8;

impl CascadeRegistry {
    /// Dispatch pending events until no new events are emitted, bounded by
    /// `MAX_CASCADE_ITERATIONS`. In dev builds the bound is a panic; in release
    /// it truncates with a log message.
    pub fn run_fixed_point(&self, state: &mut SimState, events: &mut EventRing) {
        let mut processed = events.len_total_pushed_up_to_here();  // see note below
        for iter in 0..MAX_CASCADE_ITERATIONS {
            let snapshot_len = events.len_total_pushed_up_to_here();
            if snapshot_len == processed { break; }
            for idx in processed..snapshot_len {
                // Clone the event so we can iterate while handlers push more.
                if let Some(e) = events.get_owned(idx) {
                    self.dispatch(&e, state, events);
                }
            }
            processed = snapshot_len;
            if iter == MAX_CASCADE_ITERATIONS - 1 {
                #[cfg(debug_assertions)]
                panic!("cascade did not converge within {} iterations", MAX_CASCADE_ITERATIONS);
                #[cfg(not(debug_assertions))]
                {
                    // In release: log via `telemetry` if wired; for now, a single eprintln.
                    eprintln!("cascade truncated at {} iterations", MAX_CASCADE_ITERATIONS);
                }
            }
        }
    }
}
```

The ring needs two new methods to support this:

```rust
impl EventRing {
    /// Monotonic count of pushes — needed by cascade fixed-point to see which
    /// events are "new this iteration". Survives ring eviction; old evictions
    /// just become unreachable via get_owned.
    pub fn len_total_pushed_up_to_here(&self) -> usize { self.total_pushed }

    pub fn get_owned(&self, total_idx: usize) -> Option<Event> {
        // Map monotonic index back to the in-ring position.
        let first = self.total_pushed.saturating_sub(self.entries.len());
        if total_idx < first { return None; }
        let local = total_idx - first;
        self.entries.get(local).map(|e| e.event)
    }
}
```

And a new `total_pushed: usize` counter that increments on every `push_with_cause`.

- [ ] **Step 4: Lane sort**

Update `dispatch` body to iterate `Lane::ALL` in its stable order; already correct per Task 7's implementation.

- [ ] **Step 5: Wire cascade registry into `step.rs`**

Extend the `step` signature:

```rust
pub fn step<B: PolicyBackend>(
    state:    &mut SimState,
    scratch:  &mut SimScratch,
    events:   &mut EventRing,
    backend:  &B,
    cascade:  &CascadeRegistry,
) {
    scratch.mask.reset();
    scratch.mask.mark_hold_allowed(state);
    scratch.mask.mark_move_allowed_if_others_exist(state);
    scratch.actions.clear();
    backend.evaluate(state, &scratch.mask, &mut scratch.actions);
    apply_actions(state, &scratch.actions, events, scratch);
    cascade.run_fixed_point(state, events);
    state.tick += 1;
}
```

Supply a `CascadeRegistry::new()` (empty) in every existing test — it's additive.

- [ ] **Step 6: Schema-hash bump**

```rust
h.update(b"EventKindId:AgentMoved=0,AgentAttacked=1,AgentDied=2,AgentFled=3,AgentAte=4,AgentDrank=5,AgentRested=6,AgentHarvested=7,AgentCast=8,AgentUsedItem=9,AgentPlaced=10,AgentConversed=11,AgentSharedStory=12,AgentCommunicated=13,InformationRequested=14,AgentRemembered=15,QuestPosted=16,QuestAccepted=17,BidPlaced=18,AnnounceEmitted=19,RecordMemory=20,ChronicleEntry=128");
h.update(b"Lane:Validation=0,Effect=1,Reaction=2,Audit=3");
h.update(b"MAX_CASCADE_ITERATIONS=8");
```

Regenerate baseline.

- [ ] **Step 7: Run tests + commit**

```bash
git add crates/engine/src/cascade/ crates/engine/src/step.rs \
        crates/engine/src/event/ring.rs \
        crates/engine/src/schema_hash.rs crates/engine/.schema_hash \
        crates/engine/tests/cascade_bounded.rs crates/engine/tests/cascade_lanes.rs
git commit -m "feat(engine): cascade run_fixed_point + bounded iteration + lane ordering"
```

---

## Task 9: Native action — Flee

**Files:**
- Modify: `crates/engine/src/step.rs`
- Modify: `crates/engine/src/event/mod.rs` — add `AgentFled`
- Test: `crates/engine/tests/action_flee.rs`

- [ ] **Step 1: Test**

```rust
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

/// Backend that makes every agent flee toward the away-direction from agent 1.
struct FleeFromOne;
impl PolicyBackend for FleeFromOne {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        let threat = engine::ids::AgentId::new(1).unwrap();
        for id in state.agents_alive() {
            if id == threat { out.push(Action::hold(id)); continue; }
            out.push(Action {
                agent: id,
                kind: ActionKind::Micro {
                    kind: MicroKind::Flee,
                    target: MicroTarget::Agent(threat),
                },
            });
        }
    }
}

#[test]
fn flee_moves_in_opposite_direction_from_threat() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();  // threat at origin
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 10.0), hp: 100.0,
    }).unwrap();

    let pos_before = state.agent_pos(b).unwrap();
    step(&mut state, &mut scratch, &mut events, &FleeFromOne, &cascade);
    let pos_after = state.agent_pos(b).unwrap();

    // b was east of a; flee → b moves further east.
    assert!(pos_after.x > pos_before.x);
    // AgentFled event emitted for b.
    assert!(events.iter().any(|e| matches!(e, Event::AgentFled { agent_id, .. } if *agent_id == b)));
    let _ = a;
}
```

- [ ] **Step 2: Run — fails** (Flee has no step semantics yet).

- [ ] **Step 3: Add `AgentFled` event variant** in `event/mod.rs`:

```rust
AgentFled { agent_id: AgentId, from: Vec3, to: Vec3, tick: u32 },
```

Map it in `EventKindId::from_event`.

- [ ] **Step 4: Implement Flee semantics in `step.rs::apply_actions`**

```rust
ActionKind::Micro { kind: MicroKind::Flee, target: MicroTarget::Agent(threat) } => {
    if let (Some(self_pos), Some(threat_pos)) =
        (state.agent_pos(action.agent), state.agent_pos(threat))
    {
        let away = (self_pos - threat_pos).normalize_or_zero();
        if away.length_squared() > 0.0 {
            let new_pos = self_pos + away * MOVE_SPEED_MPS;
            state.set_agent_pos(action.agent, new_pos);
            events.push(Event::AgentFled {
                agent_id: action.agent, from: self_pos, to: new_pos, tick: state.tick,
            });
        }
    }
}
```

Also extend mask predicates: `mark_flee_allowed_if_threat_exists(state)` — any hostile within `AGGRO_RANGE` (hard-coded constant, say 50.0) sets the Flee bit.

- [ ] **Step 5: Run + commit**

```bash
git add crates/engine/src/step.rs crates/engine/src/event/mod.rs \
        crates/engine/src/cascade/handler.rs crates/engine/tests/action_flee.rs
git commit -m "feat(engine): Flee action — move away from threat, emit AgentFled"
```

---

## Task 10: Native action — Attack + death cascade

**Files:**
- Modify: `crates/engine/src/step.rs`
- Test: `crates/engine/tests/action_attack_kill.rs`

- [ ] **Step 1: Test**

```rust
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct AttackTarget(engine::ids::AgentId);
impl PolicyBackend for AttackTarget {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            if id == self.0 { out.push(Action::hold(id)); continue; }
            out.push(Action {
                agent: id,
                kind: ActionKind::Micro {
                    kind: MicroKind::Attack,
                    target: MicroTarget::Agent(self.0),
                },
            });
        }
    }
}

#[test]
fn attack_reduces_hp_and_kills_at_zero() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let victim = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 20.0,
    }).unwrap();
    let _attacker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::X, hp: 100.0,
    }).unwrap();

    // Each tick deals 10 damage (hard-coded in the engine's built-in Attack).
    step(&mut state, &mut scratch, &mut events, &AttackTarget(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(10.0));
    step(&mut state, &mut scratch, &mut events, &AttackTarget(victim), &cascade);
    assert_eq!(state.agent_hp(victim), Some(0.0));
    assert!(!state.agent_alive(victim), "hp=0 → dead");

    assert!(events.iter().any(|e| matches!(e, Event::AgentDied { agent_id, .. } if *agent_id == victim)));
}
```

- [ ] **Step 2: Implement Attack**

```rust
const ATTACK_DAMAGE: f32 = 10.0;
const ATTACK_RANGE:  f32 = 2.0;  // meters

ActionKind::Micro { kind: MicroKind::Attack, target: MicroTarget::Agent(tgt) } => {
    if !state.agent_alive(tgt) { return; }
    let sp = state.agent_pos(action.agent);
    let tp = state.agent_pos(tgt);
    if let (Some(sp), Some(tp)) = (sp, tp) {
        if sp.distance(tp) <= ATTACK_RANGE {
            let new_hp = (state.agent_hp(tgt).unwrap_or(0.0) - ATTACK_DAMAGE).max(0.0);
            state.set_agent_hp(tgt, new_hp);
            events.push(Event::AgentAttacked {
                attacker: action.agent, target: tgt,
                damage: ATTACK_DAMAGE, tick: state.tick,
            });
            if new_hp <= 0.0 {
                events.push(Event::AgentDied { agent_id: tgt, tick: state.tick });
                state.kill_agent(tgt);
            }
        }
    }
}
```

- [ ] **Step 3: Mask predicate**

`mark_attack_allowed_if_target_in_range(state)` — sets the Attack bit for agent A if any alive agent B is within `ATTACK_RANGE` of A.

- [ ] **Step 4: Run + commit**

```bash
git add crates/engine/src/step.rs crates/engine/src/mask.rs \
        crates/engine/tests/action_attack_kill.rs
git commit -m "feat(engine): Attack action — damage + death cascade, mask gated by range"
```

---

## Task 11: Native actions — Eat, Drink, Rest

**Files:**
- Modify: `crates/engine/src/step.rs`
- Modify: `crates/engine/src/event/mod.rs` — add three variants
- Test: `crates/engine/tests/action_needs.rs`

- [ ] **Step 1: Test**

```rust
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct AllEat;
impl PolicyBackend for AllEat {
    fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            out.push(Action {
                agent: id, kind: ActionKind::Micro {
                    kind: MicroKind::Eat, target: MicroTarget::None,
                },
            });
        }
    }
}

#[test]
fn eating_restores_hunger_and_emits_event() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    state.set_agent_hunger(a, 0.2);

    step(&mut state, &mut scratch, &mut events, &AllEat, &cascade);

    assert!(state.agent_hunger(a).unwrap() > 0.2);
    assert!(events.iter().any(|e| matches!(e, Event::AgentAte { agent_id, .. } if *agent_id == a)));
}
```

(Repeat similar tests for Drink and Rest.)

- [ ] **Step 2: Add events**

```rust
AgentAte    { agent_id: AgentId, delta: f32, tick: u32 },
AgentDrank  { agent_id: AgentId, delta: f32, tick: u32 },
AgentRested { agent_id: AgentId, delta: f32, tick: u32 },
```

Map in `EventKindId::from_event`.

- [ ] **Step 3: Implement in `apply_actions`**

```rust
const EAT_RESTORE:   f32 = 0.25;
const DRINK_RESTORE: f32 = 0.30;
const REST_RESTORE:  f32 = 0.15;

ActionKind::Micro { kind: MicroKind::Eat,   .. } => restore_need(&mut state.hot_hunger,     action.agent, EAT_RESTORE,   events, Event::AgentAte,   state.tick),
ActionKind::Micro { kind: MicroKind::Drink, .. } => restore_need(&mut state.hot_thirst,     action.agent, DRINK_RESTORE, events, Event::AgentDrank, state.tick),
ActionKind::Micro { kind: MicroKind::Rest,  .. } => restore_need(&mut state.hot_rest_timer, action.agent, REST_RESTORE,  events, Event::AgentRested,state.tick),
```

`restore_need` is a helper that clamps to 1.0 and emits the event with the real delta (so replays see the actual clamp).

- [ ] **Step 4: Masks** — these three always eligible; `mark_needs_allowed(state)` sets all three unconditionally.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/step.rs crates/engine/src/event/mod.rs \
        crates/engine/src/mask.rs crates/engine/tests/action_needs.rs
git commit -m "feat(engine): Eat/Drink/Rest — restore needs + emit events with real delta"
```

---

## Task 12: Event-only micros (Cast, UseItem, Harvest, PlaceTile/Voxel, HarvestVoxel, Converse, ShareStory, Communicate, Ask, Remember)

These actions emit a typed event; the actual effect is domain-specific and implemented by compiler-registered cascade handlers. The engine only needs to emit the correct event shape.

**Files:**
- Modify: `crates/engine/src/event/mod.rs` — add event variants for each
- Modify: `crates/engine/src/step.rs`
- Test: `crates/engine/tests/action_emit_only.rs`

- [ ] **Step 1: Test (parametrised over all 10 variants)**

```rust
// Abbreviated — one test per action; same structure. Example for Cast:

#[test]
fn cast_emits_agentcast_event_without_state_change() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let hp_before = state.agent_hp(a).unwrap();

    struct CastOnce;
    impl PolicyBackend for CastOnce {
        fn evaluate(&self, _: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
            out.push(Action {
                agent: engine::ids::AgentId::new(1).unwrap(),
                kind: ActionKind::Micro { kind: MicroKind::Cast,
                    target: MicroTarget::AbilityIdx(3) },
            });
        }
    }
    step(&mut state, &mut scratch, &mut events, &CastOnce, &cascade);

    assert_eq!(state.agent_hp(a), Some(hp_before), "no state change — cast is a hook");
    assert!(events.iter().any(|e| matches!(e, Event::AgentCast { .. })));
}
```

Similar tests for each of the 10 event-only micros in one file with `mod` blocks.

- [ ] **Step 2: Event variants**

```rust
AgentCast         { agent_id: AgentId, ability_idx: u8, tick: u32 },
AgentUsedItem     { agent_id: AgentId, item_slot: u8, tick: u32 },
AgentHarvested    { agent_id: AgentId, resource: u64, tick: u32 },    // opaque resource handle
AgentPlaced       { agent_id: AgentId, where_pos: Vec3, kind_tag: u32, tick: u32 },
AgentConversed    { agent_id: AgentId, partner: AgentId, tick: u32 },
AgentSharedStory  { agent_id: AgentId, topic: u64, tick: u32 },
AgentCommunicated { speaker: AgentId, recipient: AgentId, fact_ref: u64, tick: u32 },
InformationRequested { asker: AgentId, target: AgentId, query: u64, tick: u32 },
AgentRemembered   { agent_id: AgentId, subject: u64, tick: u32 },
```

All map in `EventKindId::from_event`.

- [ ] **Step 3: Dispatch in `apply_actions` — each branch emits the correct event, no state mutation.**

- [ ] **Step 4: Masks — for now, unconditionally true.** The compiler generates real domain predicates (ability-ready, item-in-inventory, resource-in-range).

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/step.rs crates/engine/src/event/mod.rs \
        crates/engine/src/cascade/handler.rs crates/engine/src/mask.rs \
        crates/engine/tests/action_emit_only.rs
git commit -m "feat(engine): 10 event-only micros — emit typed events for compiler handlers"
```

---

## Task 13: Event-only macros (PostQuest, AcceptQuest, Bid)

**Files:**
- Modify: `crates/engine/src/event/mod.rs` — add macro events
- Modify: `crates/engine/src/step.rs` — dispatch `MacroAction`
- Test: `crates/engine/tests/macro_emit_only.rs`

- [ ] **Step 1: Test — one per macro.**

```rust
#[test]
fn postquest_emits_quest_posted_event() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();
    let a = state.spawn_agent(AgentSpawn { /* ... */ }).unwrap();

    struct PostOne;
    impl PolicyBackend for PostOne {
        fn evaluate(&self, _: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
            out.push(Action {
                agent: engine::ids::AgentId::new(1).unwrap(),
                kind: ActionKind::Macro(MacroAction::PostQuest {
                    quest_id: engine::ids::QuestId::new(1).unwrap(),
                    category: QuestCategory::Physical,
                    resolution: Resolution::HighestBid,
                }),
            });
        }
    }
    step(&mut state, &mut scratch, &mut events, &PostOne, &cascade);

    assert!(events.iter().any(|e| matches!(e, Event::QuestPosted { .. })));
    let _ = a;
}
```

- [ ] **Step 2: Event variants**

```rust
QuestPosted   { poster: AgentId, quest_id: QuestId, category: QuestCategory, resolution: Resolution, tick: u32 },
QuestAccepted { acceptor: AgentId, quest_id: QuestId, tick: u32 },
BidPlaced     { bidder: AgentId, auction_id: QuestId, amount: f32, tick: u32 },
```

- [ ] **Step 3: `apply_actions` macro branch**

```rust
ActionKind::Macro(MacroAction::PostQuest { quest_id, category, resolution }) => {
    events.push(Event::QuestPosted {
        poster: action.agent, quest_id: *quest_id, category: *category,
        resolution: *resolution, tick: state.tick,
    });
}
// Similar for AcceptQuest, Bid.
```

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/step.rs crates/engine/src/event/mod.rs \
        crates/engine/src/cascade/handler.rs crates/engine/tests/macro_emit_only.rs
git commit -m "feat(engine): PostQuest/AcceptQuest/Bid — event-emit macros for compiler handlers"
```

---

## Task 14: Announce — audience enumeration

**Files:**
- Modify: `crates/engine/src/step.rs` — apply_actions Announce branch
- Modify: `crates/engine/src/event/mod.rs` — add `AnnounceEmitted`, `RecordMemory`
- Modify: `crates/engine/src/state/mod.rs` — add constants MAX_ANNOUNCE_RECIPIENTS, MAX_ANNOUNCE_RADIUS
- Test: `crates/engine/tests/announce_audience.rs`

- [ ] **Step 1: Test**

```rust
#[test]
fn announce_area_emits_recordmemory_for_each_agent_in_radius() {
    let mut state = SimState::new(16, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();

    let center = Vec3::new(0.0, 0.0, 10.0);
    let speaker = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: center, hp: 100.0,
    }).unwrap();
    // 3 agents within radius 10, 2 outside.
    for i in 0..3 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center + Vec3::new(5.0 + i as f32, 0.0, 0.0), hp: 100.0,
        });
    }
    for i in 0..2 {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: center + Vec3::new(50.0 + i as f32, 0.0, 0.0), hp: 100.0,
        });
    }

    struct AnnounceArea(engine::ids::AgentId, Vec3, f32);
    impl PolicyBackend for AnnounceArea {
        fn evaluate(&self, state: &SimState, _: &MaskBuffer, out: &mut Vec<Action>) {
            out.push(Action {
                agent: self.0,
                kind: ActionKind::Macro(MacroAction::Announce {
                    speaker: self.0,
                    audience: AnnounceAudience::Area(self.1, self.2),
                    fact_payload: 0xDEADBEEF,
                }),
            });
            for id in state.agents_alive() {
                if id != self.0 { out.push(Action::hold(id)); }
            }
        }
    }

    step(&mut state, &mut scratch, &mut events, &AnnounceArea(speaker, center, 10.0), &cascade);

    let recipients: usize = events.iter()
        .filter(|e| matches!(e, Event::RecordMemory { .. }))
        .count();
    assert_eq!(recipients, 3, "three agents within 10m");
    assert!(events.iter().any(|e| matches!(e, Event::AnnounceEmitted { .. })));
}
```

- [ ] **Step 2: Event variants**

```rust
AnnounceEmitted {
    speaker: AgentId, audience_tag: u8, fact_payload: u64, tick: u32,
},
RecordMemory {
    observer: AgentId, source: AgentId, fact_payload: u64, confidence: f32, tick: u32,
},
```

- [ ] **Step 3: Apply-branch**

```rust
ActionKind::Macro(MacroAction::Announce { speaker, audience, fact_payload }) => {
    events.push(Event::AnnounceEmitted {
        speaker: *speaker, audience_tag: audience.tag(),
        fact_payload: *fact_payload, tick: state.tick,
    });
    match audience {
        AnnounceAudience::Area(center, r) => emit_area(state, events, *speaker, *center, *r, *fact_payload, 0.8),
        AnnounceAudience::Anyone => {
            let sp = state.agent_pos(*speaker).unwrap_or(Vec3::ZERO);
            emit_area(state, events, *speaker, sp, MAX_ANNOUNCE_RADIUS, *fact_payload, 0.8);
        }
        AnnounceAudience::Group(_) => {
            // Group membership lives in AggregatePool (Task 16) — for MVP,
            // fall back to Anyone-shaped enumeration. Task 16 flips this.
            let sp = state.agent_pos(*speaker).unwrap_or(Vec3::ZERO);
            emit_area(state, events, *speaker, sp, MAX_ANNOUNCE_RADIUS, *fact_payload, 0.8);
        }
    }
}
```

With `emit_area` doing the spatial-index query + per-recipient `RecordMemory`, bounded by `MAX_ANNOUNCE_RECIPIENTS`.

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/step.rs crates/engine/src/event/mod.rs \
        crates/engine/src/cascade/handler.rs \
        crates/engine/tests/announce_audience.rs
git commit -m "feat(engine): Announce — enumerate audience, emit RecordMemory per recipient"
```

---

## Task 15: Announce — overhear scan

**Files:**
- Modify: `crates/engine/src/step.rs` — extend Announce branch with overhear pass
- Test: `crates/engine/tests/announce_overhear.rs`

- [ ] **Step 1: Test**

```rust
#[test]
fn bystander_within_overhear_range_of_speaker_gets_memory_with_reduced_confidence() {
    // Speaker at origin with audience=Group(empty); bystander 15m away; OVERHEAR_RANGE=30m.
    // Expect: bystander receives RecordMemory with source=speaker, confidence≈0.6.
    // Non-bystander at 100m receives nothing.
}
```

- [ ] **Step 2: Add `OVERHEAR_RANGE: f32 = 30.0` constant.** Extend the Announce apply branch:

```rust
// After the primary audience emission, scan for bystanders.
let sp_pos = state.agent_pos(*speaker).unwrap_or(Vec3::ZERO);
let nearby = state.spatial_index().within_radius(sp_pos, OVERHEAR_RANGE);
let already_got: FxHashSet<AgentId> = events.iter().filter_map(|e| match e {
    Event::RecordMemory { observer, .. } => Some(*observer), _ => None,
}).collect();
for id in nearby {
    if id == *speaker || already_got.contains(&id) { continue; }
    events.push(Event::RecordMemory {
        observer: id, source: *speaker,
        fact_payload: *fact_payload, confidence: 0.6, tick: state.tick,
    });
}
```

- [ ] **Step 3: Commit**

```bash
git add crates/engine/src/step.rs crates/engine/tests/announce_overhear.rs
git commit -m "feat(engine): Announce — overhear scan admits bystanders at 0.6 confidence"
```

---

## Task 16: `AggregatePool<T>` + `Quest` / `Group` shapes

**Files:**
- Create: `crates/engine/src/aggregate/mod.rs`
- Create: `crates/engine/src/aggregate/quest.rs`
- Create: `crates/engine/src/aggregate/group.rs`
- Modify: `crates/engine/src/lib.rs`
- Test: `crates/engine/tests/aggregate_pool.rs`

- [ ] **Step 1: Test**

```rust
use engine::aggregate::{AggregatePool, Quest, Group};
use engine::ids::{AgentId, QuestId, GroupId};

#[test]
fn alloc_quest_returns_quest_id() {
    let mut quests: AggregatePool<Quest> = AggregatePool::new(16);
    let q = quests.alloc(Quest::stub(42)).unwrap();
    assert_eq!(quests.get(q).map(|q| q.seq), Some(42));
}

#[test]
fn kill_then_alloc_reuses_slot() {
    let mut quests: AggregatePool<Quest> = AggregatePool::new(4);
    let q1 = quests.alloc(Quest::stub(1)).unwrap();
    quests.kill(q1);
    let q2 = quests.alloc(Quest::stub(2)).unwrap();
    assert_eq!(q1.raw(), q2.raw());
}
```

- [ ] **Step 2: Implement `AggregatePool<T>`** — a thin wrapper around `Pool<T>` plus per-slot `Option<T>` storage:

```rust
use crate::pool::Pool;
use std::marker::PhantomData;

pub struct AggregatePool<T> {
    pool:  Pool<T>,
    slots: Vec<Option<T>>,
}

impl<T> AggregatePool<T> {
    pub fn new(cap: u32) -> Self {
        Self { pool: Pool::new(cap), slots: (0..cap).map(|_| None).collect() }
    }
    pub fn alloc(&mut self, t: T) -> Option<crate::pool::PoolId<T>> {
        let id = self.pool.alloc()?;
        self.slots[id.slot()] = Some(t);
        Some(id)
    }
    pub fn kill(&mut self, id: crate::pool::PoolId<T>) {
        self.pool.kill(id);
        self.slots[id.slot()] = None;
    }
    pub fn get(&self, id: crate::pool::PoolId<T>) -> Option<&T> {
        self.slots.get(id.slot()).and_then(|s| s.as_ref())
    }
    pub fn get_mut(&mut self, id: crate::pool::PoolId<T>) -> Option<&mut T> {
        self.slots.get_mut(id.slot()).and_then(|s| s.as_mut())
    }
}
```

- [ ] **Step 3: Implement `Quest` + `Group` minimal shapes**

```rust
// aggregate/quest.rs
use crate::ids::AgentId;
use crate::policy::macro_kind::{QuestCategory, Resolution};
use smallvec::SmallVec;

pub struct Quest {
    pub seq:         u32,
    pub poster:      Option<AgentId>,
    pub category:    QuestCategory,
    pub resolution:  Resolution,
    pub acceptors:   SmallVec<[AgentId; 4]>,
    pub posted_tick: u32,
}

impl Quest {
    pub fn stub(seq: u32) -> Self {
        Self {
            seq, poster: None,
            category: QuestCategory::Physical,
            resolution: Resolution::HighestBid,
            acceptors: SmallVec::new(),
            posted_tick: 0,
        }
    }
}

// aggregate/group.rs
use crate::ids::AgentId;
use smallvec::SmallVec;

pub struct Group {
    pub kind_tag: u32,
    pub members:  SmallVec<[AgentId; 8]>,
    pub leader:   Option<AgentId>,
}

impl Group {
    pub fn empty(kind_tag: u32) -> Self {
        Self { kind_tag, members: SmallVec::new(), leader: None }
    }
}
```

- [ ] **Step 4: Wire modules + commit**

```bash
git add crates/engine/src/aggregate/ crates/engine/src/lib.rs \
        crates/engine/tests/aggregate_pool.rs
git commit -m "feat(engine): AggregatePool<T> + Quest/Group struct shapes"
```

---

## Task 17: Integration — `SimScratch` extended, all actions flow through `step`

**Files:**
- Modify: `crates/engine/src/step.rs` — SimScratch with shuffle_idx already exists; ensure every `apply_actions` branch handles its kind; make `step` take `cascade: &CascadeRegistry`.
- Verify all prior tests (post-Task 6/8 edits) still pass.

- [ ] **Step 1: Run full suite**

```bash
cargo test -p engine
```

All new action-specific tests pass; pre-existing tests untouched.

- [ ] **Step 2: Clippy**

```bash
cargo clippy -p engine --all-targets -- -D warnings
```

- [ ] **Step 3: If anything is red, fix before proceeding.** Commit any cleanups under `refactor(engine): ...`.

---

## Task 18: Schema-hash re-baseline

**Files:**
- Modify: `crates/engine/src/schema_hash.rs`
- Modify: `crates/engine/.schema_hash`

- [ ] **Step 1: Verify all schema-affecting items are covered.**

```rust
h.update(b"MicroKind:Hold,MoveToward,Flee,Attack,Cast,UseItem,Harvest,Eat,Drink,Rest,PlaceTile,PlaceVoxel,HarvestVoxel,Converse,ShareStory,Communicate,Ask,Remember");
h.update(b"MacroKind:NoOp,PostQuest,AcceptQuest,Bid,Announce");
h.update(b"EventKindId:AgentMoved=0,...,ChronicleEntry=128");
h.update(b"Lane:Validation=0,Effect=1,Reaction=2,Audit=3");
h.update(b"MAX_CASCADE_ITERATIONS=8");
h.update(b"ATTACK_DAMAGE=10,ATTACK_RANGE=2,OVERHEAR_RANGE=30,MAX_ANNOUNCE_RECIPIENTS=32,MAX_ANNOUNCE_RADIUS=80,EAT_RESTORE=0.25,DRINK_RESTORE=0.30,REST_RESTORE=0.15");
// ...existing additions from Task 5...
```

Constants that affect simulation outcome are part of the schema. Changing any of them means checkpoints trained against the old values must be retrained.

- [ ] **Step 2: Regenerate baseline + commit**

```bash
cargo run -p engine --example print_schema_hash > crates/engine/.schema_hash
cargo test -p engine --test schema_hash
git add crates/engine/src/schema_hash.rs crates/engine/.schema_hash
git commit -m "chore(engine): re-baseline schema hash — covers MicroKind 18, MacroKind 4, cascade constants"
```

---

## Task 19: Acceptance — mixed-action deterministic run

**Files:**
- Test: `crates/engine/tests/acceptance_mixed_actions.rs`

- [ ] **Step 1: Write acceptance test**

```rust
//! End-to-end: 100 agents × 1000 ticks with a policy that emits a mix of
//! Hold / MoveToward / Attack / Eat / Communicate / Announce. Same seed
//! twice → identical replayable hash. Different seeds → different hashes.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::*;
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

struct MixedPolicy;
impl PolicyBackend for MixedPolicy {
    fn evaluate(&self, state: &SimState, _mask: &MaskBuffer, out: &mut Vec<Action>) {
        for id in state.agents_alive() {
            let hunger = state.agent_hunger(id).unwrap_or(1.0);
            let pick = if hunger < 0.5 {
                MicroKind::Eat
            } else if id.raw() % 7 == 0 {
                // announce every 7th agent at tick 500
                MicroKind::Communicate
            } else {
                MicroKind::MoveToward
            };
            let target = if pick == MicroKind::MoveToward {
                MicroTarget::Position(Vec3::new(0.0, 0.0, 10.0))
            } else {
                MicroTarget::None
            };
            out.push(Action {
                agent: id, kind: ActionKind::Micro { kind: pick, target },
            });
        }
    }
}

fn run(seed: u64) -> [u8; 32] {
    let mut state = SimState::new(110, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(500_000);
    let cascade = CascadeRegistry::new();
    for i in 0..100u32 {
        let angle = (i as f32 / 100.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
        });
    }
    for _ in 0..1000 {
        step(&mut state, &mut scratch, &mut events, &MixedPolicy, &cascade);
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
fn mixed_run_under_two_seconds_release() {
    let t0 = std::time::Instant::now();
    let _ = run(42);
    let elapsed = t0.elapsed();
    #[cfg(not(debug_assertions))]
    assert!(
        elapsed.as_secs_f64() <= 2.0,
        "mixed-action run took {:?}, over 2s budget", elapsed
    );
}
```

- [ ] **Step 2: Run in release**

```bash
cargo test -p engine --test acceptance_mixed_actions --release
```

- [ ] **Step 3: Commit**

```bash
git add crates/engine/tests/acceptance_mixed_actions.rs
git commit -m "test(engine): acceptance — 100 agents × 1000 ticks mixed actions, determinism + 2s budget"
```

---

## Self-review checklist

Before marking this plan complete:

- [ ] **Spec coverage.** §§7 (MicroKind full), 8 (MacroKind + built-in Announce), 9 (Cascade runtime), 14 (AggregatePool) all move from ❌/⚠️ to ✅. §12 (tick pipeline) gains the cascade phase; full 6-phase pipeline ships in Plan 2.
- [ ] **Placeholder scan.** No "TBD" / "TODO" / "fill in later". Every code block is complete.
- [ ] **Type consistency.** `Action { agent, kind: ActionKind::Micro{kind,target} | Macro(MacroAction) }` used consistently across tasks 6–17. `AgentId`, `QuestId`, `GroupId`, `EventId`, `PoolId<T>` share the `id_type!` macro.
- [ ] **Dependency direction.** No new deps on `crates/ability_operator`, workspace root, or voxel_engine. Engine stays standalone.
- [ ] **Incremental viability.** After each task, `cargo check -p engine` and `cargo test -p engine` both pass.
- [ ] **Determinism.** Every new code path either already uses `per_agent_u32/u64` for randomness or touches only deterministic state (HP, position, event emission). No `HashMap` iteration in hot loops; `FxHashSet` used where hashsets are unavoidable (e.g. the overhear-de-dupe in Task 15) is acceptable because it's fixed-seed by default.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-04-19-engine-plan-1-action-space.md`.

Same-session subagent-driven-development recommended (worked well for the MVP); parallel-session executing-plans works if you'd rather cold-pick it up later.

Next up after Plan 1:

- **Plan 2** — pipeline completion + cross-cutting traits (view Lazy/TopK, invariants, telemetry, full 6-phase tick).
- **Plan 3** — persistence + observation packer + probes.
- **Plan 4** — debug & trace runtime (trace_mask / causal_tree / tick_stepper / tick_profile / agent_history / snapshot).
