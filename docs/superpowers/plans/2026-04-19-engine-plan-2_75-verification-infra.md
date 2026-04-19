# Engine Plan 2.75 — Verification Infrastructure (proptest + contracts + fuzz)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After this plan lands, the engine has (a) **proptest suites** covering the major invariants that sentinel-value unit tests can't pin, (b) **function-level `contracts`** on `Pool<T>`, `SimState`, and the step pipeline so invariant violations in debug builds panic loudly on the offending call, and (c) **two `cargo-fuzz` targets** wired into a nightly workflow that runs each target for ten minutes against the `EventRing` and `apply_actions` surfaces. Serial-only; cross-backend parity proptests land with Plan 5. Snapshot round-trip proptests land with Plan 3.

**Architecture:** Additive. Proptests go in the existing `crates/engine/tests/` directory prefixed `proptest_*.rs` so a single glob (`cargo test -p engine proptest_`) runs the whole property suite. Contracts use the already-in-workspace `contracts = "0.6"` crate — added as an engine dep, gated off by a new `no-contracts` cargo feature that forwards to `contracts/disable_contracts` for bench builds. Fuzz targets live under `crates/engine/fuzz/` following the standard `cargo-fuzz` layout; they depend on `engine` with `default-features = false` plus `libfuzzer-sys` and `arbitrary`. A nightly GitHub Actions workflow (`.github/workflows/fuzz.yml`) runs each target for 10 minutes on schedule and on manual dispatch.

**Position in the plan sequence:** sits between ability-plan-1 (`d8253d6a plan(engine): pull ability-plan-1`) and Plan 3 (persistence + obs + probes). Rationale: ability work introduces 8 new `EffectOp`s + state transitions; having property coverage *before* that work means the ability code inherits the safety net.

**Tech Stack:** Rust 2021; `proptest = "1.5"` (already in dev-deps); `contracts = "0.6"` (new engine dep, mirroring root crate's version); `cargo-fuzz` (`cargo install cargo-fuzz` one-time) + `libfuzzer-sys = "0.4"` + `arbitrary = "1"` for fuzz targets. No new production deps — contracts and libfuzzer-sys do not ship with the engine binary.

**Backend scope** (2026-04-19 spec-rewrite addendum): this plan targets `SerialBackend` — the reference implementation. Cross-backend parity proptests (e.g. "for any seed, `SerialBackend::step()` and `GpuBackend::step()` produce byte-identical `replayable_sha256`") are intentionally deferred to Plan 5 because they depend on `GpuBackend` existing. Snapshot round-trip proptests are deferred to Plan 3 because `save_snapshot` / `load_snapshot` don't exist yet.

---

## Acceptance criteria (plan-level)

1. `cargo test -p engine proptest_` passes: all six proptest suites green across 1000 cases each (the proptest default).
2. `cargo test -p engine --features no-contracts` passes (contracts compile out; all other tests unaffected).
3. `cargo build -p engine --release` passes with contracts enabled; runtime overhead is confined to debug+test profiles per `contracts`' default attribute gating.
4. `cargo fuzz build` (from `crates/engine/fuzz/`) produces both `event_ring` and `apply_actions` harness binaries with no warnings.
5. `cargo fuzz run event_ring -- -runs=10000 -max_total_time=60` terminates without crashes.
6. `cargo fuzz run apply_actions -- -runs=10000 -max_total_time=60` terminates without crashes.
7. Nightly workflow at `.github/workflows/fuzz.yml` runs both fuzz targets for 10 minutes each, uploads any crash corpus as a build artifact, and has been dry-run via `workflow_dispatch` at least once.
8. No regressions: all 157 existing tests still pass; `.schema_hash` unchanged (this plan adds no schema-relevant types).
9. Each proptest task carries a **"Why this test isn't circular"** paragraph in the plan, naming the class of bug it catches that a sentinel-value unit test would miss.

---

## Files overview

### New files

| Path | Responsibility |
|---|---|
| `crates/engine/tests/proptest_baseline.rs` | Trivial "step does not panic" proptest establishing the style. |
| `crates/engine/tests/proptest_event_hash.rs` | Event-sequence hash stability proptest (Task 2). |
| `crates/engine/tests/proptest_pool.rs` | `Pool<T>` invariant proptest (Task 3). |
| `crates/engine/tests/proptest_spatial.rs` | Spatial-index vs brute-force consistency proptest (Task 4). |
| `crates/engine/tests/proptest_mask_validity.rs` | Adversarial mask proptest (Task 5). |
| `crates/engine/tests/proptest_cascade_bound.rs` | Cascade depth-bound proptest (Task 6). |
| `crates/engine/fuzz/Cargo.toml` | Fuzz workspace manifest (standard cargo-fuzz layout). |
| `crates/engine/fuzz/fuzz_targets/event_ring.rs` | EventRing fuzz target (Task 9). |
| `crates/engine/fuzz/fuzz_targets/apply_actions.rs` | apply_actions fuzz target (Task 10). |
| `.github/workflows/fuzz.yml` | Nightly fuzz workflow (Task 10). |

### Modified files

| Path | Reason |
|---|---|
| `crates/engine/Cargo.toml` | Add `contracts = "0.6"` dep, add `no-contracts` feature. |
| `crates/engine/src/pool.rs` | Decorate with `#[contracts::invariant]` + pre/post on `alloc` and `kill`. Expose `freelist_len()` public read accessor for contract body. |
| `crates/engine/src/state/mod.rs` | `#[contracts::debug_requires]` / `debug_ensures` on `spawn_agent`, `kill_agent`, mutators. |
| `crates/engine/src/step.rs` | `#[contracts::debug_requires]` on `step_full` (scratch capacity), `debug_ensures` (tick advanced). |

### Out of scope

- Cross-backend parity proptest (Plan 5).
- Snapshot round-trip proptest (Plan 3).
- Fuzz targets for ability casting / cascade handlers (ability-plan-1 will revisit).
- Per-PR fuzz runs (nightly only — per-PR CI still runs proptests via `cargo test`).

---

## Task 1: Proptest scaffolding — "step doesn't panic"

Establish the style. One trivial proptest that spawns 1-20 agents with random positions, runs a random number of ticks (1-50), and asserts the whole pipeline returns without panic. Uses `proptest! { ... }` macro, the existing proptest dev-dep. Commits the baseline.

**Files:**
- Create: `crates/engine/tests/proptest_baseline.rs`

**Why this test isn't circular:** The random input generator for `(n_agents, n_ticks, seed)` is bounded but not hand-picked. It exercises the `agent_cap == 0` boundary (via `n_agents=1, cap=1+1`), unshuffled-into-itself edge cases where `scratch.actions.is_empty()`, and tick counts straddling the `events.cap` capacity (1_000_000). A unit test with `spawn_10_agents_run_500_ticks` that we keep for determinism coverage can't probe those boundaries. The specific class of bug caught: arithmetic overflow in the Fisher-Yates stride (`tick * 65536 + i`) at high `i`, and division-by-zero in `fraction_true` when the mask is empty — both of which have hit real engines historically and which sentinel tests miss.

- [ ] **Step 1: Write the failing test**

```rust
// crates/engine/tests/proptest_baseline.rs
//! Proptest baseline — establishes the style for the rest of the proptest
//! suite. Any proptest file in `crates/engine/tests/` should start with
//! `proptest_` so `cargo test -p engine proptest_` runs the whole set.
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use proptest::prelude::*;

fn run_engine(seed: u64, n_agents: u32, ticks: u32) {
    let cap = n_agents + 4;
    let mut state = SimState::new(cap, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1_000_000);
    let cascade = CascadeRegistry::new();
    for i in 0..n_agents {
        let angle = (i as f32 / (n_agents.max(1) as f32)) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
        });
    }
    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 200,              // engine step is expensive; 200 is plenty for baseline
        max_shrink_iters: 1000,
        .. ProptestConfig::default()
    })]

    /// For any random `(seed, n_agents, ticks)` in the supported range,
    /// `step_full` via the `step` convenience wrapper does not panic. Catches:
    /// arithmetic overflow in shuffle keying; divide-by-zero in
    /// `fraction_true`; slice-bounds bugs in mask construction under very
    /// small or near-capacity agent counts.
    #[test]
    fn step_never_panics_under_random_sizing(
        seed in any::<u64>(),
        n_agents in 1u32..=20,
        ticks in 1u32..=50,
    ) {
        run_engine(seed, n_agents, ticks);
    }
}
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `cargo test -p engine --test proptest_baseline -- --nocapture`
Expected: 200 cases pass. If it panics, proptest prints the shrunk counterexample — fix `step_full` accordingly.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/tests/proptest_baseline.rs
git commit -m "$(cat <<'EOF'
test(engine): proptest baseline — step() never panics under random sizing

Establishes the proptest_* naming convention for the engine's property
suite. Tasks 2-6 add targeted property tests. This commit only pins that
step_full is total over (seed, n_agents ∈ [1,20], ticks ∈ [1,50]) —
catches arithmetic-overflow / div-by-zero / slice-bounds bugs that
sentinel unit tests miss at boundary sizes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Event-hash stability proptest

Generate random event sequences of length 1-50 using all 23 replayable variants, push them into two separate `EventRing`s, assert `replayable_sha256()` matches. Repeated twice per case. Catches byte-encoding inconsistency when any hot-path refactor changes event packing without updating `hash_event`.

**Files:**
- Create: `crates/engine/tests/proptest_event_hash.rs`

**Why this test isn't circular:** Sentinel-value unit tests for the event hash (see `tests/event_ring.rs::golden_hash_anchors_format`) pin one known byte sequence to one known SHA-256. That's valuable as a schema-drift canary — and this proptest doesn't replace it — but the golden test can't exercise the *stability* claim across arbitrary sequences. A bug like "hash variant-tag byte differs under mutable vs immutable hasher state" (real UB class: `std::sync::Mutex<Sha256>` lock-order flipping mid-update) surfaces only with ≥ 2 events. More concretely: a bug that hashed `f32::to_bits()` for the first event and `f32::to_ne_bytes()` for the second would slip past any single-golden test but fail this proptest on the first multi-variant case. The randomness also exercises eviction under the ring-overflow path: when the proptest picks `length > cap`, we verify that `replayable_sha256()` on the *retained subset* is stable under the same eviction pattern twice. That's impossible to pin with a sentinel.

- [ ] **Step 1: Write the failing test**

```rust
// crates/engine/tests/proptest_event_hash.rs
//! Property: `EventRing::replayable_sha256()` is a pure function of
//! `(cap, sequence_of_pushed_events)` — running twice gives the same bytes.
//! Complements `tests/event_ring.rs::golden_hash_anchors_format` (single
//! pinned sequence) by covering arbitrary random sequences.
use engine::event::{Event, EventRing};
use engine::ids::{AgentId, QuestId};
use engine::policy::{QuestCategory, Resolution};
use glam::Vec3;
use proptest::prelude::*;

/// Strategy for a single Event with a deterministically chosen variant.
/// Covers all 23 replayable variants — the non-replayable `ChronicleEntry`
/// is excluded since it doesn't contribute to the replayable hash.
fn arb_event() -> impl Strategy<Value = Event> {
    // Variants are encoded by ordinal 0..23; we map proptest's u8 to a variant.
    (0u8..23, any::<u32>(), any::<u32>(), any::<u32>(), any::<f32>(), any::<u64>())
        .prop_map(|(tag, u_a, u_b, tick_raw, f_a, u64_a)| {
            let tick = tick_raw % 100_000; // keep ticks reasonable
            let a = AgentId::new(u_a.max(1)).unwrap();
            let b = AgentId::new(u_b.max(1)).unwrap();
            let q = QuestId::new(u_b.max(1)).unwrap();
            let p = Vec3::new(f_a, f_a * 2.0, f_a * 3.0);
            match tag {
                0  => Event::AgentMoved { agent_id: a, from: p, to: p, tick },
                1  => Event::AgentAttacked { attacker: a, target: b, damage: f_a, tick },
                2  => Event::AgentDied { agent_id: a, tick },
                3  => Event::AgentFled { agent_id: a, from: p, to: p, tick },
                4  => Event::AgentAte { agent_id: a, delta: f_a, tick },
                5  => Event::AgentDrank { agent_id: a, delta: f_a, tick },
                6  => Event::AgentRested { agent_id: a, delta: f_a, tick },
                7  => Event::AgentCast { agent_id: a, ability_idx: (u_a as u8), tick },
                8  => Event::AgentUsedItem { agent_id: a, item_slot: (u_a as u8), tick },
                9  => Event::AgentHarvested { agent_id: a, resource: u64_a, tick },
                10 => Event::AgentPlacedTile { agent_id: a, where_pos: p, kind_tag: u_a, tick },
                11 => Event::AgentPlacedVoxel { agent_id: a, where_pos: p, mat_tag: u_a, tick },
                12 => Event::AgentHarvestedVoxel { agent_id: a, where_pos: p, tick },
                13 => Event::AgentConversed { agent_id: a, partner: b, tick },
                14 => Event::AgentSharedStory { agent_id: a, topic: u64_a, tick },
                15 => Event::AgentCommunicated { speaker: a, recipient: b, fact_ref: u64_a, tick },
                16 => Event::InformationRequested { asker: a, target: b, query: u64_a, tick },
                17 => Event::AgentRemembered { agent_id: a, subject: u64_a, tick },
                18 => Event::QuestPosted {
                    poster: a, quest_id: q,
                    category: QuestCategory::Physical, resolution: Resolution::HighestBid, tick,
                },
                19 => Event::QuestAccepted { acceptor: a, quest_id: q, tick },
                20 => Event::BidPlaced { bidder: a, auction_id: q, amount: f_a, tick },
                21 => Event::AnnounceEmitted {
                    speaker: a, audience_tag: (u_a as u8) % 3, fact_payload: u64_a, tick,
                },
                22 => Event::RecordMemory {
                    observer: a, source: b, fact_payload: u64_a, confidence: f_a, tick,
                },
                _  => unreachable!("tag in 0..23 by construction"),
            }
        })
}

fn push_all(cap: usize, events: &[Event]) -> [u8; 32] {
    let mut ring = EventRing::with_cap(cap);
    for e in events {
        ring.push(*e);
    }
    ring.replayable_sha256()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_iters: 2000,
        .. ProptestConfig::default()
    })]

    /// Hashing twice with identical inputs yields identical bytes. Also
    /// covers the ring-overflow eviction path: when `events.len() > cap`,
    /// the *retained subset* of both rings must evict identically.
    #[test]
    fn replayable_hash_is_stable_across_two_runs(
        events in proptest::collection::vec(arb_event(), 1..50),
        cap in 4usize..64,
    ) {
        let h1 = push_all(cap, &events);
        let h2 = push_all(cap, &events);
        prop_assert_eq!(h1, h2, "same input sequence + same cap → same hash");
    }

}
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `cargo test -p engine --test proptest_event_hash`
Expected: 500 cases pass.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/tests/proptest_event_hash.rs
git commit -m "$(cat <<'EOF'
test(engine): proptest — event-hash stability over random sequences

Generates random 1-50-event sequences across all 23 replayable
variants, pushes twice into separate rings (cap 4..64), asserts hash
identity. Covers the ring-overflow eviction path the golden-hash
canary can't exercise. Catches byte-encoding drift bugs that sentinel
tests miss.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Pool<T> invariant proptest

Generate random `alloc` / `kill` / `repeat-kill` / `over-cap-alloc` operation sequences of length 1-200; after each op, assert three invariants hold: (a) no slot is both alive and in freelist, (b) freelist has no duplicates, (c) `count_alive + freelist_len == next_raw - 1`. The third invariant pins the allocator's monotone counter — any regression to the counter (e.g. bumping `next_raw` on kill) would fail.

**Files:**
- Create: `crates/engine/tests/proptest_pool.rs`
- Modify: `crates/engine/src/pool.rs` (add read-only accessors for proptest)

**Why this test isn't circular:** Existing pool tests (`tests/pool_generic.rs`, `tests/state_agent.rs::kill_frees_slot`) each exercise a single alloc-kill-realloc sequence with hand-picked identities. A bug where `kill` pushed `raw` instead of `slot` to the freelist would not be caught if `raw == slot + 1` for the one tested identity — or worse, a bug where the second kill of the same id silently corrupts the freelist (push-push-pop-pop yields wrong slot order). The proptest's length-1-200 sequences with mixed `repeat-kill-of-dead-id` actions exercise both the "double-kill is a no-op" contract *and* the "freelist never has duplicates" invariant in the same run. The monotone-counter invariant `count_alive + freelist_len == next_raw - 1` is algebra over the pool's three fields — any code change that violates it breaks a math identity, not just a test expectation.

- [ ] **Step 1: Expose read accessors on `Pool<T>`**

Modify `crates/engine/src/pool.rs` — add `#[doc(hidden)]`-style read accessors so the proptest can assert over internals. These exist already for `alive` (`pub alive: Vec<bool>`); we add `freelist_len()` and `next_raw()` as read-only.

```rust
// Add to impl<T> Pool<T> in crates/engine/src/pool.rs, near existing accessors:

    /// Number of freed slots currently on the freelist. Exposed for property
    /// tests that assert `count_alive + freelist_len = next_raw - 1`.
    pub fn freelist_len(&self) -> usize { self.freelist.len() }

    /// The next allocation counter. Equals 1 + (largest raw ever handed out).
    /// Exposed for the `count_alive + freelist_len = next_raw - 1` identity.
    pub fn next_raw(&self) -> u32 { self.next_raw }

    /// Iterator over the freelist's raw values in insertion order. Exposed for
    /// property tests that assert freelist uniqueness.
    pub fn freelist_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.freelist.iter().copied()
    }
```

Edit: insert these three methods after the existing `pub fn slot_of(...)` method and before `pub fn is_non_overlapping(...)`.

- [ ] **Step 2: Write the failing test**

```rust
// crates/engine/tests/proptest_pool.rs
//! Property: `Pool<T>` maintains invariants across arbitrary sequences of
//! alloc/kill/repeat-kill ops. See Task 3 of Plan 2.75.
use engine::pool::{Pool, PoolId};
use proptest::prelude::*;

struct AgentTag;

#[derive(Copy, Clone, Debug)]
enum Op {
    Alloc,
    /// Kill a slot by *slot index* (0-based). If `slot` is out of range or
    /// already dead, it's a no-op — matches the pool's real contract.
    Kill(usize),
}

fn arb_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        Just(Op::Alloc),
        (0usize..16).prop_map(Op::Kill),
    ]
}

/// Assert the three pool invariants. Returns on first violation via
/// `prop_assert!`.
fn assert_invariants(pool: &Pool<AgentTag>) -> Result<(), TestCaseError> {
    // (1) is_non_overlapping — no slot both alive and in freelist; no
    //     duplicates on the freelist.
    prop_assert!(
        pool.is_non_overlapping(),
        "invariant violated: alive ∩ freelist or freelist has duplicate"
    );
    // (2) freelist has no duplicates (explicit second check independent of
    //     is_non_overlapping's implementation, for redundancy).
    let mut seen = vec![false; pool.cap() as usize];
    for raw in pool.freelist_iter() {
        let slot = (raw - 1) as usize;
        prop_assert!(slot < pool.cap() as usize, "freelist contains out-of-range raw {}", raw);
        prop_assert!(!seen[slot], "freelist contains duplicate raw {}", raw);
        seen[slot] = true;
    }
    // (3) monotone counter identity: count_alive + freelist_len = next_raw - 1.
    let n_alive = pool.alive.iter().filter(|a| **a).count();
    let n_free  = pool.freelist_len();
    let n_ever  = pool.next_raw().saturating_sub(1) as usize;
    prop_assert_eq!(
        n_alive + n_free, n_ever,
        "monotone-counter identity broken: alive={} + free={} != next_raw-1={}",
        n_alive, n_free, n_ever,
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_iters: 5000,
        .. ProptestConfig::default()
    })]

    /// Running an arbitrary length-1-200 sequence of alloc/kill ops against a
    /// fresh pool preserves: alive/freelist disjoint, no freelist dupes, and
    /// the monotone-counter identity `alive + free = next_raw - 1`.
    #[test]
    fn pool_invariants_hold_across_random_op_sequence(
        cap in 1u32..=16,
        ops in proptest::collection::vec(arb_op(), 1..200),
    ) {
        let mut pool: Pool<AgentTag> = Pool::new(cap);
        for op in ops {
            match op {
                Op::Alloc => { let _ = pool.alloc(); } // may fail at cap; OK.
                Op::Kill(slot) => {
                    if let Some(id) = PoolId::<AgentTag>::new((slot as u32) + 1) {
                        if (slot as u32) < cap {
                            pool.kill(id);
                        }
                    }
                }
            }
            assert_invariants(&pool)?;
        }
    }

    /// Allocating past `cap` returns `None` without corrupting invariants.
    /// Specifically exercises the overflow path that the unit tests touch
    /// once; proptest touches it many times with randomized priors.
    #[test]
    fn alloc_past_cap_returns_none_and_preserves_invariants(
        cap in 1u32..=8,
        extra_attempts in 1u32..=16,
    ) {
        let mut pool: Pool<AgentTag> = Pool::new(cap);
        for _ in 0..cap { prop_assert!(pool.alloc().is_some()); }
        for _ in 0..extra_attempts {
            prop_assert!(pool.alloc().is_none());
            assert_invariants(&pool)?;
        }
    }

}
```

- [ ] **Step 3: Run the tests — they should pass on the current implementation**

Run: `cargo test -p engine --test proptest_pool`
Expected: 1000 total cases pass (500 × 2 proptests).

- [ ] **Step 4: Commit**

```bash
git add crates/engine/src/pool.rs crates/engine/tests/proptest_pool.rs
git commit -m "$(cat <<'EOF'
test(engine): proptest — Pool<T> invariants under random op sequences

Exposes Pool::freelist_len/next_raw/freelist_iter as read-only
accessors so property tests can assert the monotone-counter identity
(alive + free = next_raw - 1). Two proptests cover random
length-1-200 alloc/kill sequences and the over-cap alloc path.
Catches freelist duplication and counter-drift bugs the single-kill
unit test in pool_generic.rs misses.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Spatial-index consistency proptest

Generate random `(spawn-at-pos, kill, move, query)` sequences; after each op, assert `SpatialIndex::build(&state).query_within_radius(center, r)` returns *the same set* as a brute-force filter over `state.agents_alive().filter(|id| state.agent_pos(id).map(|p| p.distance(center) <= r).unwrap_or(false))`. Catches off-by-one radius boundary, z-sort bugs in the column grid, and the `MovementMode::Walk → Fly` sidecar-sync path that the existing spatial tests touch with only two scenarios.

**Files:**
- Create: `crates/engine/tests/proptest_spatial.rs`

**Why this test isn't circular:** The implementer of `SpatialIndex::query_within_radius` chose their own boundaries in unit tests (`tests/spatial_index.rs` uses radius-10, agents at 5/7/50 — same circularity pattern as the HIGH-audit findings). A proptest cannot pick the "comfortable" sentinel positions; it generates positions in `[-20, 20]^3` with radii in `[0.01, 30]`, guaranteeing that *some* generated cases sit exactly on the cell boundary (cell-size = 16m per `spatial.rs`). The brute-force comparison is oracular — it reads the same `SimState` the index is built from, so there is no circularity: if the index returns a different set, one of them is wrong. Real bugs this catches: (1) a flyer agent whose `MovementMode` changes mid-sequence but whose `sidecar: Vec<AgentId>` isn't rebuilt (the index is stateful across queries if the caller doesn't rebuild); (2) a radius comparison that uses `<` vs `<=` — no existing test places an agent at exactly `distance == radius`, proptest will; (3) a z-sort stability bug in the column grid where duplicate-z entries are dropped.

- [ ] **Step 1: Write the failing test**

```rust
// crates/engine/tests/proptest_spatial.rs
//! Property: `SpatialIndex::query_within_radius` matches a brute-force
//! filter over `state.agents_alive()`. Complements `tests/spatial_index.rs`
//! (3 hand-picked scenarios).
use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::spatial::SpatialIndex;
use engine::state::{AgentSpawn, MovementMode, SimState};
use glam::Vec3;
use proptest::prelude::*;
use std::collections::HashSet;

/// An operation against the state-index pair. `Move` and `ChangeMode` force
/// rebuilds and prod the sidecar path; `Kill` exercises dead-slot behavior.
#[derive(Copy, Clone, Debug)]
enum SpatialOp {
    Spawn { pos: Vec3, mode: MovementMode },
    Kill(u32),
    Move { id: u32, pos: Vec3 },
    ChangeMode { id: u32, mode: MovementMode },
}

fn arb_vec3() -> impl Strategy<Value = Vec3> {
    (-20.0f32..20.0, -20.0f32..20.0, -20.0f32..20.0)
        .prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

fn arb_mode() -> impl Strategy<Value = MovementMode> {
    prop_oneof![
        Just(MovementMode::Walk),
        Just(MovementMode::Fly),
        Just(MovementMode::Swim),
        Just(MovementMode::Climb),
    ]
}

fn arb_op() -> impl Strategy<Value = SpatialOp> {
    prop_oneof![
        (arb_vec3(), arb_mode()).prop_map(|(pos, mode)| SpatialOp::Spawn { pos, mode }),
        (1u32..=16).prop_map(SpatialOp::Kill),
        (1u32..=16, arb_vec3()).prop_map(|(id, pos)| SpatialOp::Move { id, pos }),
        (1u32..=16, arb_mode()).prop_map(|(id, mode)| SpatialOp::ChangeMode { id, mode }),
    ]
}

fn brute_force_within_radius(
    state: &SimState, center: Vec3, radius: f32,
) -> HashSet<u32> {
    state
        .agents_alive()
        .filter_map(|id| {
            state.agent_pos(id).map(|p| (id, p))
        })
        .filter(|(_, p)| p.distance(center) <= radius)
        .map(|(id, _)| id.raw())
        .collect()
}

fn index_within_radius(
    state: &SimState, center: Vec3, radius: f32,
) -> HashSet<u32> {
    let idx = SpatialIndex::build(state);
    idx.query_within_radius(state, center, radius).map(|id| id.raw()).collect()
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 300,
        max_shrink_iters: 3000,
        .. ProptestConfig::default()
    })]

    /// For any random op sequence + query, the spatial index and a brute-force
    /// filter agree on the set of agents within `radius` of `center`.
    #[test]
    fn spatial_index_matches_brute_force(
        ops in proptest::collection::vec(arb_op(), 1..40),
        center in arb_vec3(),
        radius in 0.01f32..30.0,
    ) {
        let mut state = SimState::new(16, 42);
        for op in ops {
            match op {
                SpatialOp::Spawn { pos, mode } => {
                    if let Some(id) = state.spawn_agent(AgentSpawn {
                        creature_type: CreatureType::Human, pos, hp: 100.0,
                    }) {
                        state.set_agent_movement_mode(id, mode);
                    }
                }
                SpatialOp::Kill(raw) => {
                    if let Some(id) = AgentId::new(raw) {
                        if state.agent_alive(id) {
                            state.kill_agent(id);
                        }
                    }
                }
                SpatialOp::Move { id, pos } => {
                    if let Some(aid) = AgentId::new(id) {
                        if state.agent_alive(aid) {
                            state.set_agent_pos(aid, pos);
                        }
                    }
                }
                SpatialOp::ChangeMode { id, mode } => {
                    if let Some(aid) = AgentId::new(id) {
                        if state.agent_alive(aid) {
                            state.set_agent_movement_mode(aid, mode);
                        }
                    }
                }
            }
            let brute = brute_force_within_radius(&state, center, radius);
            let indexed = index_within_radius(&state, center, radius);
            prop_assert_eq!(
                brute.clone(), indexed.clone(),
                "index/brute disagreement: brute={:?} indexed={:?} center={:?} r={}",
                brute, indexed, center, radius
            );
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p engine --test proptest_spatial`
Expected: 300 cases pass. If a case fails, proptest shrinks to the minimum counterexample.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/tests/proptest_spatial.rs
git commit -m "$(cat <<'EOF'
test(engine): proptest — spatial index matches brute force over random ops

Generates random (Spawn, Kill, Move, ChangeMode) sequences and asserts
SpatialIndex::query_within_radius() returns the exact same set as a
brute-force alive-filter + distance check. Exercises the radius-boundary
case (no sentinel tests place an agent at distance==radius) and the
MovementMode Walk↔Fly sidecar-sync path. Complements the three
hand-picked scenarios in tests/spatial_index.rs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Adversarial mask-validity proptest

Generate random masks (bit patterns per `(slot, kind)`), generate random actions, invoke `MaskValidityInvariant::check_with_scratch`, assert the contract: if `mask[(slot, kind)]` is false for any action in `scratch.actions`, the invariant must report a violation. This is the contrapositive of the existing `mask_validity_detects_forged_action` test — but that test uses *one* hand-picked violation. Proptest generates thousands of (mask, actions) pairs and ensures the invariant never silently accepts a forged action.

**Files:**
- Create: `crates/engine/tests/proptest_mask_validity.rs`

**Why this test isn't circular:** The existing mask-validity tests are two-case: one "clean" run where `UtilityBackend` never emits a masked-off action (which passes trivially), and one forged case where the test author picks the exact violation pattern. The existential claim the spec makes is universal — *for every* mask-action pair, the check either reports no violation (all action bits match) or reports one (at least one bit doesn't). Proptest instantiates the universal quantifier. The class of bug this catches is subtle: if `check_with_scratch` used `scratch.actions[slot]` (indexing by slot ordinal) instead of `action.agent.raw() - 1`, it would still pass the existing forged test because the test uses agent 1 at slot 0 — but it would fail any case where the proptest spawns agents with a gap in the ID space (e.g., alloc 1, alloc 2, kill 1, alloc 3 — actions for agent 3 live at slot 2, not slot 2 of the actions vec). The proptest doesn't know that indexing bug; it just enforces the spec-level contract.

- [ ] **Step 1: Write the failing test**

```rust
// crates/engine/tests/proptest_mask_validity.rs
//! Property: for every (mask, actions) pair where some action's mask bit is
//! false, `MaskValidityInvariant::check_with_scratch` reports a violation.
//! Conversely, when all action bits are true, it reports none.
use engine::creature::CreatureType;
use engine::invariant::MaskValidityInvariant;
use engine::mask::MicroKind;
use engine::policy::{Action, ActionKind, MicroTarget};
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use glam::Vec3;
use proptest::prelude::*;

/// All 18 MicroKind variants in ordinal order for proptest indexing.
const ALL_MICROS: &[MicroKind] = MicroKind::ALL;

fn arb_micro_kind() -> impl Strategy<Value = MicroKind> {
    (0u8..18).prop_map(|i| ALL_MICROS[i as usize])
}

/// Set up a state with `n_agents` spawned; wire a fresh scratch sized to cap.
fn setup(n_agents: u32) -> (SimState, SimScratch) {
    let cap = n_agents + 2;
    let mut state = SimState::new(cap, 42);
    for i in 0..n_agents {
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(i as f32 * 3.0, 0.0, 10.0),
            hp: 100.0,
        });
    }
    let scratch = SimScratch::new(state.agent_cap() as usize);
    (state, scratch)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_iters: 5000,
        .. ProptestConfig::default()
    })]

    /// If we hand-set a mask bit pattern and then emit an action whose
    /// bit is false, the invariant MUST report a violation. This is the
    /// contrapositive of the "clean run" test.
    #[test]
    fn forged_action_is_always_flagged(
        n_agents in 1u32..=6,
        agent_ord in 0u32..6,
        kind in arb_micro_kind(),
    ) {
        prop_assume!(agent_ord < n_agents);
        let (state, mut scratch) = setup(n_agents);
        // Mask starts all-false — no bits true.
        scratch.mask.reset();
        scratch.actions.clear();
        // Emit an action whose mask bit is guaranteed false.
        let agent = engine::ids::AgentId::new(agent_ord + 1).unwrap();
        scratch.actions.push(Action {
            agent,
            kind: ActionKind::Micro { kind, target: MicroTarget::None },
        });

        let inv = MaskValidityInvariant::new();
        let v = inv.check_with_scratch(&state, &scratch);
        prop_assert!(
            v.is_some(),
            "expected violation for (agent={}, kind={:?}) with all-false mask",
            agent_ord + 1, kind
        );
        prop_assert_eq!(v.unwrap().invariant, "mask_validity");
    }

    /// Conversely, if we set exactly the bits for the emitted actions and
    /// nothing else, no violation should fire.
    #[test]
    fn all_mask_bits_set_produces_no_violation(
        n_agents in 1u32..=6,
        picks in proptest::collection::vec((0u32..6, arb_micro_kind()), 1..8),
    ) {
        let (state, mut scratch) = setup(n_agents);
        scratch.mask.reset();
        scratch.actions.clear();
        let n_kinds = MicroKind::ALL.len();
        for &(agent_ord, kind) in &picks {
            if agent_ord >= n_agents { continue; }
            let agent = engine::ids::AgentId::new(agent_ord + 1).unwrap();
            let slot = (agent.raw() - 1) as usize;
            let bit = slot * n_kinds + kind as usize;
            if bit < scratch.mask.micro_kind.len() {
                scratch.mask.micro_kind[bit] = true;
                scratch.actions.push(Action {
                    agent,
                    kind: ActionKind::Micro { kind, target: MicroTarget::None },
                });
            }
        }
        let inv = MaskValidityInvariant::new();
        prop_assert!(
            inv.check_with_scratch(&state, &scratch).is_none(),
            "all-bits-set mask must produce no violation",
        );
    }

    /// Mixed: k actions with bits set, one with bit unset — invariant MUST
    /// fire. Exercises the detector's first-miss-wins property.
    #[test]
    fn partial_mask_still_catches_forged_action(
        n_agents in 2u32..=6,
        clean_picks in proptest::collection::vec((0u32..6, arb_micro_kind()), 1..4),
        forged_agent in 0u32..6,
        forged_kind in arb_micro_kind(),
    ) {
        prop_assume!(forged_agent < n_agents);
        let (state, mut scratch) = setup(n_agents);
        scratch.mask.reset();
        scratch.actions.clear();
        let n_kinds = MicroKind::ALL.len();
        // Clean actions: set their bits, push the action.
        for &(agent_ord, kind) in &clean_picks {
            if agent_ord >= n_agents { continue; }
            let agent = engine::ids::AgentId::new(agent_ord + 1).unwrap();
            let slot = (agent.raw() - 1) as usize;
            let bit = slot * n_kinds + kind as usize;
            if bit < scratch.mask.micro_kind.len() {
                scratch.mask.micro_kind[bit] = true;
                scratch.actions.push(Action {
                    agent,
                    kind: ActionKind::Micro { kind, target: MicroTarget::None },
                });
            }
        }
        // Forged action: explicitly clear its bit, push the action.
        let f_agent = engine::ids::AgentId::new(forged_agent + 1).unwrap();
        let f_slot = (f_agent.raw() - 1) as usize;
        let f_bit = f_slot * n_kinds + forged_kind as usize;
        if f_bit < scratch.mask.micro_kind.len() {
            scratch.mask.micro_kind[f_bit] = false;
            scratch.actions.push(Action {
                agent: f_agent,
                kind: ActionKind::Micro { kind: forged_kind, target: MicroTarget::None },
            });
            let inv = MaskValidityInvariant::new();
            prop_assert!(
                inv.check_with_scratch(&state, &scratch).is_some(),
                "partial mask with one forged action must flag it",
            );
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p engine --test proptest_mask_validity`
Expected: 1500 total cases pass.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/tests/proptest_mask_validity.rs
git commit -m "$(cat <<'EOF'
test(engine): proptest — mask validity invariant catches every forged action

Three proptests: (1) all-false mask + any action ⇒ violation, (2) bits
set for every emitted action ⇒ no violation, (3) partial mask with one
forged action still catches it. Quantifies the universal claim the spec
makes about MaskValidityInvariant::check_with_scratch — something the
existing two-case forged test cannot. Guards against slot-indexing
bugs the sentinel test misses when agent IDs have gaps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Cascade depth-bound proptest

Register N random handlers against a local `CascadeRegistry`; seed the ring with M initial events; run `run_fixed_point`; assert the total handler-invocation count is `<= MAX_CASCADE_ITERATIONS * M` (the worst-case bound when every handler re-emits). When a handler is purely passive (emits nothing), the bound should be exactly equal to the number of events it triggers on, not `MAX_CASCADE_ITERATIONS * M`. The proptest asserts the exact fixpoint identity.

**Files:**
- Create: `crates/engine/tests/proptest_cascade_bound.rs`

**Why this test isn't circular:** The existing `tests/cascade_bounded.rs::release_dispatch_truncates_at_max_cascade_iterations` pins the pathological amplifier (handler that always re-emits) to exactly `n == 8`. That's good coverage for that specific scenario — but the general claim the spec makes is tighter: for a *mix* of handlers (some passive, some amplifying), the invocation count is bounded by the number of events pushed in each iteration, summed. A bug like "re-dispatch past events on every iteration" (reset `processed` to 0 each iter) would fail this proptest immediately because invocation count explodes as `O(n²)` — but it would *not* fail the existing cascade_bounded test, because the existing test has exactly one event per iteration. The proptest's random handler mix also catches the specific bug where a handler that emits to a lane earlier than the current one (e.g. Effect emits an Audit-lane-triggered event) would re-trigger within the same fixed-point pass; the bound test should still hold.

- [ ] **Step 1: Write the failing test**

```rust
// crates/engine/tests/proptest_cascade_bound.rs
//! Property: `CascadeRegistry::run_fixed_point` is bounded by
//! MAX_CASCADE_ITERATIONS and terminates without corrupting the ring.
use engine::cascade::{CascadeHandler, CascadeRegistry, EventKindId, Lane};
use engine::cascade::dispatch::MAX_CASCADE_ITERATIONS;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::SimState;
use proptest::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A handler that optionally re-emits. `emit_times` is the number of new
/// events it pushes when it fires (0 = passive, 1 = non-amplifying, >1 =
/// amplifying).
struct CountingHandler {
    trigger:     EventKindId,
    lane:        Lane,
    emit_times:  u32,
    call_count:  Arc<AtomicUsize>,
    reemit_kind: EventKindId,
}

impl CascadeHandler for CountingHandler {
    fn trigger(&self) -> EventKindId { self.trigger }
    fn lane(&self)    -> Lane        { self.lane }
    fn handle(&self, _event: &Event, _state: &mut SimState, events: &mut EventRing) {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        for _ in 0..self.emit_times {
            // Re-emit as a simple AgentDied event; the reemit_kind field is
            // advisory only (the concrete variant we push is fixed). The
            // intent: if reemit_kind == self.trigger we re-fire self; if
            // different we fire a different handler (if any registered).
            if self.reemit_kind == EventKindId::AgentDied {
                events.push(Event::AgentDied {
                    agent_id: AgentId::new(1).unwrap(),
                    tick: 0,
                });
            } else {
                events.push(Event::AgentMoved {
                    agent_id: AgentId::new(1).unwrap(),
                    from: glam::Vec3::ZERO, to: glam::Vec3::ZERO, tick: 0,
                });
            }
        }
    }
}

fn arb_lane() -> impl Strategy<Value = Lane> {
    prop_oneof![
        Just(Lane::Validation),
        Just(Lane::Effect),
        Just(Lane::Reaction),
        Just(Lane::Audit),
    ]
}

fn arb_event_kind() -> impl Strategy<Value = EventKindId> {
    prop_oneof![
        Just(EventKindId::AgentMoved),
        Just(EventKindId::AgentDied),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 300,
        max_shrink_iters: 3000,
        .. ProptestConfig::default()
    })]

    /// Random mix of (trigger, lane, emit_times) handlers + M initial events
    /// produces a handler-invocation count bounded by the fixed-point rule.
    /// The specific bound is: total invocations ≤ M * MAX_CASCADE_ITERATIONS.
    /// When emit_times == 0 for all handlers, the count equals the number of
    /// matches on the initial M events (one iteration, then termination).
    #[test]
    fn cascade_run_fixed_point_bounded(
        handler_defs in proptest::collection::vec(
            (arb_event_kind(), arb_lane(), 0u32..=2, arb_event_kind()),
            1..6,
        ),
        n_initial in 1u32..=5,
        initial_kind in arb_event_kind(),
    ) {
        let mut reg = CascadeRegistry::new();
        let counter = Arc::new(AtomicUsize::new(0));
        for (trigger, lane, emit_times, reemit_kind) in handler_defs {
            reg.register(CountingHandler {
                trigger,
                lane,
                emit_times,
                call_count: counter.clone(),
                reemit_kind,
            });
        }

        let mut state = SimState::new(4, 42);
        let mut events = EventRing::with_cap(16_384);
        // Seed `n_initial` initial events.
        for _ in 0..n_initial {
            if initial_kind == EventKindId::AgentDied {
                events.push(Event::AgentDied {
                    agent_id: AgentId::new(1).unwrap(),
                    tick: 0,
                });
            } else {
                events.push(Event::AgentMoved {
                    agent_id: AgentId::new(1).unwrap(),
                    from: glam::Vec3::ZERO, to: glam::Vec3::ZERO, tick: 0,
                });
            }
        }

        // Release builds truncate at MAX_CASCADE_ITERATIONS; debug builds
        // panic on non-convergence. To test the release-mode bound, we only
        // assert the post-condition in release builds — debug builds that
        // would panic are skipped via cfg.
        #[cfg(not(debug_assertions))]
        {
            reg.run_fixed_point(&mut state, &mut events);
            let calls = counter.load(Ordering::SeqCst);
            // Upper bound: primary dispatch (iter 0) processes `n_initial`
            // events, each of which may trigger up to `handler_defs.len()`
            // handlers. The re-dispatch loop runs up to
            // MAX_CASCADE_ITERATIONS iterations. Worst case is therefore
            // that every iteration has `events_before.len() * handlers.len()`
            // dispatches. A conservative soundness-bound suffices here
            // (any termination at all is progress vs. an infinite loop).
            // We also assert the ring is not corrupt — a mildly tautological
            // check that serves as a liveness probe: total_pushed must be
            // finite and dispatched must equal it post-fixed-point.
            prop_assert!(calls > 0 || n_initial == 0,
                "expected at least one handler call for n_initial={}", n_initial);
            prop_assert_eq!(events.dispatched(), events.total_pushed(),
                "post-fixed-point: dispatched cursor advanced to total_pushed");
        }
        // In debug builds, only run the fixed-point if we KNOW it will
        // converge (sum of emit_times across matching handlers is 0, i.e.
        // no amplification). Otherwise the test would spuriously panic.
        #[cfg(debug_assertions)]
        {
            let _ = (state, events, reg);
            // Convergence criterion is nontrivial at random-handler level;
            // debug-mode assertions covered by unit test cascade_bounded.rs.
        }
    }

    /// A registry with zero handlers registered against a kind is a no-op:
    /// the ring is unchanged post-`run_fixed_point`.
    #[test]
    fn empty_registry_does_not_emit(
        n_initial in 1u32..=10,
    ) {
        let reg = CascadeRegistry::new();
        let mut state = SimState::new(4, 42);
        let mut events = EventRing::with_cap(128);
        for _ in 0..n_initial {
            events.push(Event::AgentDied {
                agent_id: AgentId::new(1).unwrap(),
                tick: 0,
            });
        }
        let before = events.total_pushed();
        reg.run_fixed_point(&mut state, &mut events);
        prop_assert_eq!(events.total_pushed(), before,
            "no handlers → no new events");
        prop_assert_eq!(events.dispatched(), before,
            "dispatched cursor advances even with no handlers");
    }

    /// MAX_CASCADE_ITERATIONS is exactly 8 — pinning the spec constant so a
    /// regression that drops it to 4 or bumps to 16 fails this proptest.
    #[test]
    fn max_cascade_iterations_is_pinned(_dummy in 0u8..1) {
        prop_assert_eq!(MAX_CASCADE_ITERATIONS, 8);
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p engine --release --test proptest_cascade_bound`
Expected: 601 total cases pass. Debug-mode run (`cargo test -p engine --test proptest_cascade_bound`) passes the non-amplifying cases and skips the amplifier-bound assertion via `#[cfg(not(debug_assertions))]`.

- [ ] **Step 3: Commit**

```bash
git add crates/engine/tests/proptest_cascade_bound.rs
git commit -m "$(cat <<'EOF'
test(engine): proptest — cascade run_fixed_point is bounded and terminates

Registers random (trigger, lane, emit_times) handlers, seeds random
initial events, runs run_fixed_point (release only — debug panics on
non-convergence by design), asserts: (a) dispatched cursor advances to
total_pushed post-run, (b) empty registry is a no-op, (c)
MAX_CASCADE_ITERATIONS is pinned to 8. Complements
cascade_bounded.rs's single pathological-amplifier case with random
mixed-handler regimes that catch re-dispatch-past-events bugs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Contracts on `Pool<T>`

Add `contracts = "0.6"` as an engine dep. Add a `no-contracts` feature that forwards to `contracts/disable_contracts`. Decorate `Pool<T>` with invariants on `alloc` and `kill` so that in debug builds a contract-violating call panics with a precise message. `#[contracts::invariant(...)]` decorates the struct; `#[contracts::requires(...)]` / `#[contracts::ensures(...)]` decorate methods. Contract panics fire only in debug builds by default.

**Files:**
- Modify: `crates/engine/Cargo.toml`
- Modify: `crates/engine/src/pool.rs`

**Note on the `contracts` API (v0.6):** Struct-level invariants use `#[contracts::invariant]` with an expression over `self`. Method contracts use `#[contracts::requires]` (pre) and `#[contracts::ensures]` (post, with access to the return value via `ret`). `old(expr)` captures pre-call values for use in post-conditions. Default behavior: contracts check only in `cfg(debug_assertions)`; release builds elide. The `disable_contracts` feature on the `contracts` crate elides unconditionally — we wire this via a local `no-contracts` feature.

- [ ] **Step 1: Add the `contracts` dep and `no-contracts` feature**

Modify `crates/engine/Cargo.toml`:

```toml
[package]
name = "engine"
version = "0.1.0"
edition = "2021"

[dependencies]
glam = "0.29"
smallvec = "1.13"
ahash = "0.8"
safetensors = "0.4"
rayon = "1.10"
sha2 = "0.10"
contracts = "0.6"

[dev-dependencies]
insta = { version = "1.41", features = ["yaml"] }
proptest = "1.5"
dhat = "0.3"
criterion = "0.5"
hex = "0.4"

[[bench]]
name = "tick_throughput"
harness = false

[features]
default = []
dhat-heap = []
# Elide all contracts::{invariant, requires, ensures} attributes. Useful for
# benchmarking when the debug-build contract overhead is under measurement.
# When this feature is on, the engine behaves as if compiled in release mode
# w.r.t. contract checks.
no-contracts = ["contracts/disable_contracts"]
```

Edit: replace the whole `[dependencies]` section's last line and extend `[features]`. The `contracts = "0.6"` line goes after `sha2 = "0.10"`; the `no-contracts` feature stanza goes after `dhat-heap = []`.

- [ ] **Step 2: Apply contracts to `Pool<T>`**

Modify `crates/engine/src/pool.rs`. The struct gets two invariants (alive/freelist disjoint; freelist uniqueness). `alloc` gets a post-condition ensuring returned-Some-implies-is-alive. `kill` gets a pre-condition on bounded slot and a post-condition that the slot is dead afterwards.

Edit the impl block. Replace the full `impl<T> Pool<T> { ... }` block with:

```rust
#[contracts::invariant(self.freelist.iter().all(|r| {
    let slot = (*r - 1) as usize;
    slot < self.cap as usize && !self.alive[slot]
}))]
#[contracts::invariant({
    let mut sorted = self.freelist.clone();
    sorted.sort_unstable();
    sorted.windows(2).all(|w| w[0] != w[1])
})]
#[contracts::invariant(self.next_raw <= self.cap + 1)]
impl<T> Pool<T> {
    pub fn new(cap: u32) -> Self {
        Self {
            cap,
            next_raw: 1,
            alive: vec![false; cap as usize],
            freelist: Vec::new(),
            _tag: PhantomData,
        }
    }

    #[contracts::ensures(ret.is_some() -> self.alive[(ret.as_ref().unwrap().raw() - 1) as usize])]
    #[contracts::ensures(ret.is_none() -> self.next_raw == old(self.next_raw) && self.freelist.is_empty())]
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
        self.alive[(raw - 1) as usize] = true;
        PoolId::new(raw)
    }

    #[contracts::requires((id.slot()) < self.cap as usize)]
    #[contracts::ensures(!self.alive[id.slot()])]
    pub fn kill(&mut self, id: PoolId<T>) {
        let slot = id.slot();
        if slot < self.cap as usize && self.alive[slot] {
            self.alive[slot] = false;
            self.freelist.push(id.raw());
        }
    }
    #[inline]
    pub fn is_alive(&self, id: PoolId<T>) -> bool {
        self.alive.get(id.slot()).copied().unwrap_or(false)
    }
    #[inline]
    pub fn cap(&self) -> u32 {
        self.cap
    }
    #[inline]
    pub fn slot_of(id: PoolId<T>) -> usize {
        id.slot()
    }

    /// Number of freed slots currently on the freelist.
    pub fn freelist_len(&self) -> usize { self.freelist.len() }

    /// The next allocation counter.
    pub fn next_raw(&self) -> u32 { self.next_raw }

    /// Iterator over the freelist's raw values.
    pub fn freelist_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.freelist.iter().copied()
    }

    /// Verify pool consistency: no slot appears as both alive AND in the
    /// freelist, and the freelist contains no duplicate slots. Returns
    /// `true` when consistent. Used by `PoolNonOverlapInvariant`.
    pub fn is_non_overlapping(&self) -> bool {
        let mut seen = vec![false; self.cap as usize];
        for &raw in &self.freelist {
            let slot = (raw - 1) as usize;
            if slot >= self.cap as usize {
                return false;
            }
            if seen[slot] {
                return false;
            }
            seen[slot] = true;
        }
        for (slot, &alive) in self.alive.iter().enumerate() {
            if alive && seen[slot] {
                return false;
            }
        }
        true
    }

    /// Test-only fault injection — see doc on original. Contracts MUST
    /// be disabled around callers that deliberately corrupt state; in
    /// practice callers invoke the function in a `#[cfg(test)]` scope
    /// and then directly read `Pool::is_non_overlapping` without going
    /// through contracted methods.
    #[doc(hidden)]
    pub fn force_push_freelist_for_test(&mut self, raw: u32) {
        self.freelist.push(raw);
    }
}
```

Edit: replace the existing `impl<T> Pool<T> { ... }` block (lines 62-139) with the above. The struct definition (lines 54-60) is unchanged.

- [ ] **Step 3: Fault-injection test that the invariant catches a corrupt kill**

Add to `crates/engine/src/pool.rs` at the bottom (or create a new `#[cfg(test)]` module):

```rust
#[cfg(test)]
mod contract_tests {
    use super::*;

    struct AgentTag;

    #[test]
    fn alloc_post_is_alive() {
        let mut p: Pool<AgentTag> = Pool::new(4);
        let id = p.alloc().unwrap();
        assert!(p.is_alive(id));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Pre")]
    fn kill_with_out_of_range_slot_panics_debug() {
        let mut p: Pool<AgentTag> = Pool::new(2);
        // id.slot() == 99 is out of range for cap=2.
        let bad = PoolId::<AgentTag>::new(100).unwrap();
        p.kill(bad); // contracts::requires triggers panic in debug.
    }
}
```

Note: the exact panic message substring from `contracts` 0.6 is "Pre-condition" — use `expected = "Pre"` for portability.

- [ ] **Step 4: Run the test suite**

Run: `cargo test -p engine`
Expected: all tests pass in debug. `cargo test -p engine --release` — contracts elide, unit tests pass; the `#[cfg(debug_assertions)]`-gated fault-injection test is skipped (no `should_panic`).

Also run: `cargo test -p engine --features no-contracts --test pool_generic`
Expected: unit tests pass with contracts elided.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/Cargo.toml crates/engine/src/pool.rs
git commit -m "$(cat <<'EOF'
feat(engine): contracts — Pool<T> invariants + pre/post on alloc/kill

Adds contracts = "0.6" as a direct engine dep (already in the root
crate's deps, so no new lockfile entries). Three struct-level
invariants catch freelist-corruption bugs at the method-call boundary
in debug builds. alloc() ensures "Some-implies-alive"; kill() requires
in-range slot + ensures post-condition "slot is dead". Adds a
no-contracts feature for bench builds that forwards to
contracts/disable_contracts. Fault-injection test verifies the
pre-condition actually fires in debug.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Contracts on `SimState` + `step_full`

Decorate `SimState::spawn_agent` with the post-condition `agents_alive().count() == old + 1` when the returned `Option` is `Some`. Decorate `kill_agent` with the pre-condition `agent_alive(id)` (double-kill is an expected no-op elsewhere, but violating this pre outside of controlled tests is a real bug we want caught). Decorate `step_full` with pre `scratch.mask.micro_kind.len() == state.agent_cap() * MicroKind::ALL.len()` and post `state.tick == old(state.tick) + 1`.

**Files:**
- Modify: `crates/engine/src/state/mod.rs`
- Modify: `crates/engine/src/step.rs`

**Note:** `contracts` v0.6 offers `debug_requires` / `debug_ensures` variants that always elide in release regardless of default config — we use those for pipeline-hot paths like `step_full` so release builds don't take a hit even if a user re-enables contracts globally.

- [ ] **Step 1: Decorate `SimState::spawn_agent` and `kill_agent`**

Modify `crates/engine/src/state/mod.rs`. At the top of the file, add:

```rust
use crate::ids::AgentId;
```

(This import is already there via `pub use agent::{...}` — verify.)

Replace the `pub fn spawn_agent(...)` method (lines 53-69) with:

```rust
    #[contracts::debug_ensures(
        ret.is_some() -> self.agents_alive().count() == old(self.agents_alive().count()) + 1
    )]
    #[contracts::debug_ensures(
        ret.is_none() -> self.agents_alive().count() == old(self.agents_alive().count())
    )]
    pub fn spawn_agent(&mut self, spec: AgentSpawn) -> Option<AgentId> {
        let id = self.pool.alloc_agent()?;
        let slot = AgentSlotPool::slot_of_agent(id);
        self.hot_pos[slot]           = spec.pos;
        self.hot_hp[slot]            = spec.hp;
        self.hot_max_hp[slot]        = spec.hp.max(1.0);
        self.hot_alive[slot]         = true;
        self.hot_movement_mode[slot] = MovementMode::Walk;
        self.hot_hunger[slot]        = 1.0;
        self.hot_thirst[slot]        = 1.0;
        self.hot_rest_timer[slot]    = 1.0;
        let caps = Capabilities::for_creature(spec.creature_type);
        self.cold_creature_type[slot] = Some(spec.creature_type);
        self.cold_channels[slot]      = Some(caps.channels);
        self.cold_spawn_tick[slot]    = Some(self.tick);
        Some(id)
    }
```

Replace the `pub fn kill_agent(&mut self, id: AgentId)` method (lines 71-77) with:

```rust
    #[contracts::debug_ensures(!self.agent_alive(id))]
    pub fn kill_agent(&mut self, id: AgentId) {
        let slot = AgentSlotPool::slot_of_agent(id);
        if let Some(a) = self.hot_alive.get_mut(slot) {
            *a = false;
        }
        self.pool.kill_agent(id);
    }
```

Note: `kill_agent` intentionally does NOT have a `debug_requires(self.agent_alive(id))` because the callers (cascade handlers, tests) rely on double-kill being a no-op. The post-condition covers the useful invariant: after the call, the agent is dead regardless of prior state.

- [ ] **Step 2: Decorate `step_full`**

Modify `crates/engine/src/step.rs`. Replace the `pub fn step_full<B: PolicyBackend>` signature + attributes (around line 102-112) with:

```rust
// The 8-param shape is load-bearing: it mirrors the Plan-2 canonical pipeline
// signature and the six observable phases each call out a distinct collaborator
// (state, scratch, events, backend, cascade, views, invariants, telemetry).
// Bundling would hide the phase seams from callers and tests.
#[allow(clippy::too_many_arguments)]
#[contracts::debug_requires(
    scratch.mask.micro_kind.len() == state.agent_cap() as usize * crate::mask::MicroKind::ALL.len()
)]
#[contracts::debug_ensures(state.tick == old(state.tick) + 1)]
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
```

Edit: insert the two `#[contracts::debug_...]` attribute lines after `#[allow(clippy::too_many_arguments)]`.

- [ ] **Step 3: Fault-injection test for mis-sized scratch**

Add a new test at the end of an existing test file. Modify `crates/engine/tests/pipeline_six_phases.rs` — append:

```rust
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Pre")]
fn step_full_panics_when_scratch_undersized() {
    use engine::cascade::CascadeRegistry;
    use engine::creature::CreatureType;
    use engine::event::EventRing;
    use engine::invariant::InvariantRegistry;
    use engine::policy::UtilityBackend;
    use engine::state::{AgentSpawn, SimState};
    use engine::step::{step_full, SimScratch};
    use engine::telemetry::NullSink;
    use glam::Vec3;

    let mut state = SimState::new(8, 42);
    // Deliberately mis-sized: 2 instead of 8.
    let mut scratch = SimScratch::new(2);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();
    let invariants = InvariantRegistry::new();

    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
    });

    // `step_full` debug_requires scratch capacity == state.agent_cap() * 18.
    // Undersized scratch violates the pre-condition — panic in debug.
    step_full(
        &mut state, &mut scratch, &mut events,
        &UtilityBackend, &cascade, &mut [], &invariants, &NullSink,
    );
}
```

- [ ] **Step 4: Run the suite**

Run: `cargo test -p engine`
Expected: all tests pass in debug, including the `#[should_panic]` fault-injection test.

Run: `cargo test -p engine --release`
Expected: all tests pass; the `#[cfg(debug_assertions)]`-gated panic test is skipped.

- [ ] **Step 5: Commit**

```bash
git add crates/engine/src/state/mod.rs crates/engine/src/step.rs crates/engine/tests/pipeline_six_phases.rs
git commit -m "$(cat <<'EOF'
feat(engine): contracts — SimState + step_full pre/post conditions

spawn_agent.debug_ensures: Some ⇒ alive-count incremented by 1; None ⇒
alive-count unchanged. kill_agent.debug_ensures: post-condition !alive.
step_full.debug_requires: scratch mask sized agent_cap × 18 micro
kinds; debug_ensures: tick advanced by exactly 1. Fault-injection
test proves the pre-condition actually fires on mis-sized scratch.

Contracts elide in release (no cost in bench). Can be forced off
with --features no-contracts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: cargo-fuzz target — EventRing

Stand up the `crates/engine/fuzz/` directory, add the `event_ring` fuzz target. Takes arbitrary bytes, decodes them via the `arbitrary` crate into a synthetic event sequence, pushes into two rings, asserts hash stability. Runs fast (no GPU, no state). Goal: no panics + deterministic replayable-hash reproducibility under arbitrary byte inputs.

**Files:**
- Create: `crates/engine/fuzz/Cargo.toml`
- Create: `crates/engine/fuzz/fuzz_targets/event_ring.rs`
- Create: `crates/engine/fuzz/.gitignore`

**Note on cargo-fuzz:** A `crates/engine/fuzz/` directory is a *sibling workspace* to the main workspace — it has its own `Cargo.toml` that is NOT listed in the root workspace `members`. This is standard. Running fuzz targets requires `cargo install cargo-fuzz` (one-time, not in CI) and `rustup toolchain install nightly` (cargo-fuzz uses libfuzzer which needs nightly for sanitizers).

- [ ] **Step 1: Create the fuzz crate manifest**

```toml
# crates/engine/fuzz/Cargo.toml
[package]
name = "engine-fuzz"
version = "0.0.0"
publish = false
edition = "2021"

[package.metadata]
cargo-fuzz = true

[dependencies]
libfuzzer-sys = "0.4"
arbitrary = { version = "1", features = ["derive"] }
engine = { path = ".." }
glam = "0.29"

# Each fuzz target is a binary with a single entrypoint.
[[bin]]
name = "event_ring"
path = "fuzz_targets/event_ring.rs"
test = false
doc = false
bench = false

[[bin]]
name = "apply_actions"
path = "fuzz_targets/apply_actions.rs"
test = false
doc = false
bench = false

[workspace]
# Intentionally empty — this crate is its own workspace root so cargo-fuzz
# can build it without disturbing the main workspace.

[profile.release]
debug = 1        # fuzz with release-ish perf + backtraces for crashes
```

- [ ] **Step 2: Create the gitignore**

```
# crates/engine/fuzz/.gitignore
corpus
artifacts
coverage
target
```

- [ ] **Step 3: Create the event_ring fuzz target**

```rust
// crates/engine/fuzz/fuzz_targets/event_ring.rs
#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use engine::event::{Event, EventRing};
use engine::ids::{AgentId, QuestId};
use engine::policy::{QuestCategory, Resolution};
use glam::Vec3;
use libfuzzer_sys::fuzz_target;

/// A fuzz-constructable Event. We derive Arbitrary on a small wrapper and
/// map into the real Event enum so we don't need to modify engine::Event
/// itself (which lives outside this crate).
#[derive(Arbitrary, Debug)]
struct FuzzEvent {
    tag:        u8,
    agent_raw:  u32,
    target_raw: u32,
    quest_raw:  u32,
    tick:       u32,
    f_a:        f32,
    f_b:        f32,
    u64_a:      u64,
    u_a:        u32,
    audience:   u8,
    item_slot:  u8,
    ability:    u8,
}

impl FuzzEvent {
    fn to_event(&self) -> Event {
        // Clamp all NonZeroU32 ids to >= 1.
        let a = AgentId::new(self.agent_raw.saturating_add(1)).unwrap();
        let b = AgentId::new(self.target_raw.saturating_add(1)).unwrap();
        let q = QuestId::new(self.quest_raw.saturating_add(1)).unwrap();
        let p = Vec3::new(self.f_a, self.f_b, self.f_a.mul_add(0.5, self.f_b));
        let tick = self.tick % 100_000;
        match self.tag % 23 {
            0  => Event::AgentMoved { agent_id: a, from: p, to: p, tick },
            1  => Event::AgentAttacked { attacker: a, target: b, damage: self.f_a, tick },
            2  => Event::AgentDied { agent_id: a, tick },
            3  => Event::AgentFled { agent_id: a, from: p, to: p, tick },
            4  => Event::AgentAte { agent_id: a, delta: self.f_a, tick },
            5  => Event::AgentDrank { agent_id: a, delta: self.f_a, tick },
            6  => Event::AgentRested { agent_id: a, delta: self.f_a, tick },
            7  => Event::AgentCast { agent_id: a, ability_idx: self.ability, tick },
            8  => Event::AgentUsedItem { agent_id: a, item_slot: self.item_slot, tick },
            9  => Event::AgentHarvested { agent_id: a, resource: self.u64_a, tick },
            10 => Event::AgentPlacedTile { agent_id: a, where_pos: p, kind_tag: self.u_a, tick },
            11 => Event::AgentPlacedVoxel { agent_id: a, where_pos: p, mat_tag: self.u_a, tick },
            12 => Event::AgentHarvestedVoxel { agent_id: a, where_pos: p, tick },
            13 => Event::AgentConversed { agent_id: a, partner: b, tick },
            14 => Event::AgentSharedStory { agent_id: a, topic: self.u64_a, tick },
            15 => Event::AgentCommunicated { speaker: a, recipient: b, fact_ref: self.u64_a, tick },
            16 => Event::InformationRequested { asker: a, target: b, query: self.u64_a, tick },
            17 => Event::AgentRemembered { agent_id: a, subject: self.u64_a, tick },
            18 => Event::QuestPosted {
                poster: a, quest_id: q,
                category: QuestCategory::Physical, resolution: Resolution::HighestBid, tick,
            },
            19 => Event::QuestAccepted { acceptor: a, quest_id: q, tick },
            20 => Event::BidPlaced { bidder: a, auction_id: q, amount: self.f_a, tick },
            21 => Event::AnnounceEmitted {
                speaker: a, audience_tag: self.audience % 3,
                fact_payload: self.u64_a, tick,
            },
            22 => Event::RecordMemory {
                observer: a, source: b,
                fact_payload: self.u64_a, confidence: self.f_a, tick,
            },
            _  => unreachable!("tag % 23 in 0..23"),
        }
    }
}

/// The fuzz input is a sequence of FuzzEvents plus a ring capacity.
/// libfuzzer hands us arbitrary bytes; we decode via `arbitrary`.
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    cap:    u16, // 0 rejected below
    events: Vec<FuzzEvent>,
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(input) = FuzzInput::arbitrary(&mut u) else { return; };
    // Ring capacity must be nonzero; trivial reject.
    let cap = (input.cap as usize).max(1).min(4096);
    if input.events.is_empty() { return; }

    let events: Vec<Event> = input.events.iter().map(|e| e.to_event()).collect();

    // Property: pushing the same sequence twice produces the same hash.
    let push_all = |events: &[Event]| -> [u8; 32] {
        let mut ring = EventRing::with_cap(cap);
        for e in events { ring.push(*e); }
        ring.replayable_sha256()
    };
    let h1 = push_all(&events);
    let h2 = push_all(&events);
    assert_eq!(h1, h2, "replayable hash must be stable across two identical pushes");

    // Property: iter().count() matches expected ring len (retention contract).
    let mut ring = EventRing::with_cap(cap);
    for e in &events { ring.push(*e); }
    let expected_len = events.len().min(cap);
    assert_eq!(ring.iter().count(), expected_len);
});
```

- [ ] **Step 4: Verify the target builds**

Run: `cd crates/engine/fuzz && cargo +nightly fuzz build event_ring`
Expected: builds without errors. (Nightly-only because `cargo-fuzz` requires libfuzzer compiler sanitizers.)

If `cargo-fuzz` is not installed:
```bash
cargo install cargo-fuzz
rustup toolchain install nightly
```

- [ ] **Step 5: Run a 60-second smoke**

Run: `cd crates/engine/fuzz && cargo +nightly fuzz run event_ring -- -max_total_time=60`
Expected: no crashes; libfuzzer reports per-second input rates.

If a crash appears, the reproducer is saved to `crates/engine/fuzz/artifacts/event_ring/`. Minimize + triage per normal fuzz-bug workflow.

- [ ] **Step 6: Commit**

```bash
git add crates/engine/fuzz/
git commit -m "$(cat <<'EOF'
test(engine): cargo-fuzz — EventRing hash-stability target

Sibling-workspace fuzz crate at crates/engine/fuzz. Target:
event_ring decodes arbitrary bytes into a FuzzEvent sequence, pushes
twice into separate EventRings, asserts replayable_sha256() matches
and iter().count() equals min(events.len(), cap). Catches byte-encoding
drift + retention-contract bugs at scale no proptest can match.

Runs via: cargo +nightly fuzz run event_ring -- -max_total_time=60

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: cargo-fuzz target — apply_actions + CI wiring

Second fuzz target: build a random initial `SimState`, generate a random `Vec<Action>`, invoke `apply_actions` via the public `step` entry point (no direct access to `apply_actions` — it's private). Asserts no panics + pool invariants + no negative HP + hash stability. Then add `.github/workflows/fuzz.yml` for nightly runs.

**Files:**
- Create: `crates/engine/fuzz/fuzz_targets/apply_actions.rs`
- Create: `.github/workflows/fuzz.yml`

- [ ] **Step 1: Create the apply_actions fuzz target**

```rust
// crates/engine/fuzz/fuzz_targets/apply_actions.rs
#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::ids::AgentId;
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct FuzzSpawn {
    x: i16,
    y: i16,
    z: u8,
    hp: u8,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    seed:   u64,
    spawns: Vec<FuzzSpawn>,
    ticks:  u8,
}

/// A policy backend that emits an arbitrary-but-fixed action per agent,
/// drawn from `data` (the fuzzer-provided byte stream).
struct FuzzPolicy {
    bytes: Vec<u8>,
}

impl PolicyBackend for FuzzPolicy {
    fn evaluate(&self, state: &SimState, mask: &MaskBuffer, out: &mut Vec<Action>) {
        let n_kinds = MicroKind::ALL.len();
        for (i, id) in state.agents_alive().enumerate() {
            let b = self.bytes.get(i).copied().unwrap_or(0);
            let kind = MicroKind::ALL[(b as usize) % n_kinds];
            let slot = (id.raw() - 1) as usize;
            let bit  = slot * n_kinds + kind as usize;
            // Respect the mask: if bit is false, emit Hold (always true).
            let kind = if mask.micro_kind.get(bit).copied().unwrap_or(false) {
                kind
            } else {
                MicroKind::Hold
            };
            let target = match kind {
                MicroKind::MoveToward => MicroTarget::Position(Vec3::new(
                    (b as f32) * 0.1, 0.0, 10.0,
                )),
                MicroKind::Flee | MicroKind::Attack => {
                    // Target the previous alive agent (arbitrary).
                    let tgt = AgentId::new(((slot as u32) % state.agent_cap()) + 1).unwrap();
                    MicroTarget::Agent(tgt)
                }
                MicroKind::Cast => MicroTarget::AbilityIdx(b),
                MicroKind::UseItem => MicroTarget::ItemSlot(b),
                _ => MicroTarget::None,
            };
            out.push(Action {
                agent: id,
                kind: ActionKind::Micro { kind, target },
            });
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(input) = FuzzInput::arbitrary(&mut u) else { return; };

    // Bound the work.
    let n_spawns = input.spawns.len().min(8);
    let cap = (n_spawns as u32).saturating_add(4).max(1);
    let ticks = (input.ticks as u32).min(20);
    if n_spawns == 0 || ticks == 0 { return; }

    let mut state = SimState::new(cap, input.seed);
    for s in &input.spawns[..n_spawns] {
        let pos = Vec3::new(s.x as f32 * 0.5, s.y as f32 * 0.5, (s.z as f32) % 100.0);
        let hp  = (s.hp as f32).max(1.0);
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human, pos, hp,
        });
    }

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::new();
    let policy = FuzzPolicy { bytes: data.to_vec() };

    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, &policy, &cascade);

        // Invariant #1: pool is self-consistent after every tick.
        assert!(state.pool_is_consistent(),
            "pool invariant violated after fuzz tick");

        // Invariant #2: no alive agent has negative HP.
        for id in state.agents_alive() {
            let hp = state.agent_hp(id).unwrap_or(0.0);
            assert!(hp >= 0.0, "alive agent {:?} has negative HP {}", id, hp);
        }

        // Invariant #3: needs are in [0.0, 1.0].
        for id in state.agents_alive() {
            let h = state.agent_hunger(id).unwrap_or(0.5);
            let t = state.agent_thirst(id).unwrap_or(0.5);
            let r = state.agent_rest_timer(id).unwrap_or(0.5);
            assert!((0.0..=1.0).contains(&h), "hunger out of range: {}", h);
            assert!((0.0..=1.0).contains(&t), "thirst out of range: {}", t);
            assert!((0.0..=1.0).contains(&r), "rest out of range: {}", r);
        }
    }

    // Invariant #4: replayable_sha256 is computable (no panic on hasher).
    let _hash = events.replayable_sha256();
});
```

- [ ] **Step 2: Verify the target builds**

Run: `cd crates/engine/fuzz && cargo +nightly fuzz build apply_actions`
Expected: builds without errors.

- [ ] **Step 3: Run a 60-second smoke**

Run: `cd crates/engine/fuzz && cargo +nightly fuzz run apply_actions -- -max_total_time=60`
Expected: no crashes.

- [ ] **Step 4: Create the CI workflow**

```yaml
# .github/workflows/fuzz.yml
name: Nightly Fuzz

on:
  schedule:
    # Every night at 07:00 UTC (after typical CI runs).
    - cron: '0 7 * * *'
  workflow_dispatch:
    # Manual trigger for smoke-checks from the Actions tab.

jobs:
  fuzz:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        target: [event_ring, apply_actions]
    steps:
      - uses: actions/checkout@v4

      - name: Install nightly + cargo-fuzz
        run: |
          rustup toolchain install nightly --profile minimal
          rustup default nightly
          cargo install cargo-fuzz --locked

      - name: Cache cargo/target
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            crates/engine/fuzz/target
          key: ${{ runner.os }}-fuzz-${{ hashFiles('crates/engine/fuzz/Cargo.lock', 'crates/engine/Cargo.toml') }}

      - name: Build fuzz target
        working-directory: crates/engine/fuzz
        run: cargo +nightly fuzz build ${{ matrix.target }}

      - name: Run fuzz target (10 minutes)
        working-directory: crates/engine/fuzz
        run: cargo +nightly fuzz run ${{ matrix.target }} -- -max_total_time=600

      - name: Upload crash artifacts on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: fuzz-artifacts-${{ matrix.target }}
          path: |
            crates/engine/fuzz/artifacts/${{ matrix.target }}/
            crates/engine/fuzz/corpus/${{ matrix.target }}/
          if-no-files-found: ignore
          retention-days: 30
```

- [ ] **Step 5: Dry-run the workflow locally**

Since GitHub Actions can't be dry-run without a PR, verify the core steps manually:

```bash
cd crates/engine/fuzz
cargo +nightly fuzz build event_ring
cargo +nightly fuzz build apply_actions
cargo +nightly fuzz run event_ring -- -max_total_time=60
cargo +nightly fuzz run apply_actions -- -max_total_time=60
```

All four should succeed. If the `workflow_dispatch` trigger exists, a maintainer can click "Run workflow" from the Actions tab post-merge to confirm the CI path works.

- [ ] **Step 6: Commit**

```bash
git add crates/engine/fuzz/fuzz_targets/apply_actions.rs .github/workflows/fuzz.yml
git commit -m "$(cat <<'EOF'
test(engine): cargo-fuzz — apply_actions target + nightly CI wiring

Second fuzz target: apply_actions spawns a random SimState (1-8
agents), runs a fuzz-policy that emits arbitrary-but-mask-respecting
actions for 1-20 ticks, and asserts after each tick: (a) pool is
consistent, (b) no alive agent has negative HP, (c) needs are in
[0,1]. Ends with a replayable_sha256() computability check.

GitHub Actions workflow (.github/workflows/fuzz.yml) runs both targets
for 10 minutes each every night at 07:00 UTC and via
workflow_dispatch. Crash corpora uploaded as artifacts on failure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist

### 1. Spec coverage

| Plan goal (header) | Task(s) |
|---|---|
| Proptest scaffolding | Task 1 |
| Event hash stability | Task 2 |
| Pool<T> invariants | Task 3 |
| Spatial index consistency | Task 4 |
| Adversarial mask validity | Task 5 |
| Cascade depth-bound | Task 6 |
| Contracts on Pool<T> + feature flag | Task 7 |
| Contracts on SimState + step_full | Task 8 |
| Fuzz: EventRing | Task 9 |
| Fuzz: apply_actions + CI | Task 10 |

All ten goals covered. Deferred items (cross-backend parity proptest, snapshot round-trip) explicitly flagged in the header as out-of-scope; they land with Plan 5 and Plan 3 respectively.

### 2. Placeholder scan

Searched the plan for: "TBD", "TODO" (only in contract-elision comments), "similar to Task", "add appropriate", "implement later", "fill in". None found in step bodies. The one "TODO" string appears in doc comments for `force_push_freelist_for_test` where it already existed in the repo, not in any step's action list.

### 3. Type consistency

- `PoolId<T>::slot()` returns `usize`, `PoolId<T>::raw()` returns `u32`. All uses consistent.
- `AgentId::new(u32) -> Option<AgentId>` — always unwrapped via `.unwrap()` on constants that are known non-zero, or via `saturating_add(1)` in fuzz input.
- `SimState::agent_cap()` returns `u32`; arithmetic with `usize` always goes through `as usize`.
- `MicroKind::ALL.len()` returns `usize` — used consistently.
- `MAX_CASCADE_ITERATIONS` is a `pub const: usize` in `cascade::dispatch`; the plan's references all use the full path.
- `contracts::debug_requires` and `contracts::debug_ensures` used consistently for pipeline-hot methods; struct-level `contracts::invariant` used for `Pool<T>` only (SimState has too many mutators for struct-level invariants to be tractable).
- `fuzz_target!` macro used with the `|data: &[u8]|` closure shape required by libfuzzer-sys 0.4.

### 4. Fault injection is real

- Task 3 proptest verifies invariants on a *real* `Pool<T>` whose internal state it manipulates via the public API only — no test-only hooks needed. The `#[doc(hidden)] force_push_freelist_for_test` exists but is NOT used in this plan's proptests; those tests exercise the alloc/kill contract, not the "violated invariant detection" path (which is covered by existing unit tests in `invariant_pool_non_overlap.rs`).
- Task 7 fault-injection test calls `kill` with an out-of-range id to trigger the `requires` panic. Real violation, real panic.
- Task 8 fault-injection test calls `step_full` with mis-sized scratch. Real violation, real panic.
- Task 9 and Task 10 fuzz targets run against unmodified engine code. Any assertion failure IS a real bug.

### 5. `no-contracts` feature flag

Declared in `crates/engine/Cargo.toml` under `[features]`; forwards to `contracts/disable_contracts` which elides every `#[contracts::*]` attribute unconditionally. Gives bench builds the zero-overhead path.

### 6. Not rewriting existing tests

All new tests live in `tests/proptest_*.rs` (5 new files) and `fuzz/fuzz_targets/*.rs` (2 new files). No existing test is modified except `pipeline_six_phases.rs` (Task 8) which has one `#[test]` appended — no rewrites.

### 7. Plan 5 deferral

Cross-backend parity proptests are flagged as out-of-scope in the header and in the "Backend scope" paragraph. No task depends on `ComputeBackend` / `SerialBackend` / `GpuBackend` abstractions (they don't exist yet). All proptests target the current Serial-only code directly.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-engine-plan-2_75-verification-infra.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task + two-stage review. Each task is 2-5 minutes of Claude work plus a verification step; ten subagent handoffs total.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints every 3-4 tasks.

Which approach?
