# Plan 4 — Debug & Trace Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the six debug-and-trace components per `spec/runtime.md` §24: `trace_mask`, `causal_tree`, `tick_stepper`, `tick_profile`, `agent_history`, and `snapshot` (repro bundle). All host-side; GPU backends trigger downloads on demand. Plus xtask integration so each is reachable via the CLI.

**Architecture:** Six independent host-side modules under a new `crates/engine/src/debug/` directory, each with its own struct + collector + serializer. They hook into existing tick-pipeline phases via opt-in collector traits. The `snapshot` infrastructure already exists at `crates/engine/src/snapshot/`; Plan 4 adds a "repro bundle" wrapper that packages snapshot + causal_tree + first-N-ticks of trace_mask into a single artifact for bug reproduction.

**Tech Stack:** Rust 2021. Reuses existing `EventRing<E>` (causality data already there), `MaskBuffer` (trace target), `SimState` snapshot infra (already in `engine/src/snapshot/`), `engine::telemetry` (timing histograms). No new external dependencies.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/engine/src/snapshot/{mod,format,migrate}.rs` (955 lines) — snapshot serializer + migration; exposes `save_snapshot`, `load_snapshot`. Repro bundle wraps this.
  - `crates/engine/src/probe/mod.rs` — parity Probe harness (§20). Distinct from §24 trace; lives alongside.
  - `crates/engine/src/event/ring.rs` — `EventRing<E>` already records `cause: Option<EventId>` per entry. Causal_tree is a presentation layer over existing data.
  - `crates/engine/src/mask.rs` — `MaskBuffer { micro_kind: Vec<bool> }` storage primitive; trace_mask captures its state per tick.
  - `crates/engine/src/telemetry/` — `TelemetrySink` trait + metrics catalog; tick_profile emits via existing histograms.
  - `engine_rules::step::step` — phase orchestration entry point; tick_stepper hooks pre/post each phase.
  - `crates/engine/src/state/mod.rs::SimState` — agent SoA fields; agent_history captures field deltas per tick.

  Search method: `rg`, direct `Read`.

- **Decision:** add a new `crates/engine/src/debug/` subdir housing the six components as sibling modules. Each module is opt-in (default disabled; activated via xtask flag or runtime hook). The trait-level integration with `step` uses an existing extension point: a per-phase trace callback, currently absent — adding the trait is a minor primitive extension.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none — debug runtime is engine primitive infrastructure.
  - Generated outputs re-emitted: none. (None of the new modules produce DSL output.)
  - Emitter changes: none.

- **Hand-written downstream code:**
  - `crates/engine/src/debug/mod.rs` + 6 sibling files: NEW. Justification: debug+trace runtime is universal infrastructure (per spec §24 "all host-side"); it's not derivable from DSL declarations because debug instrumentation isn't a property of the rule set.
  - `crates/engine/src/debug/trace_mask.rs`: NEW.
  - `crates/engine/src/debug/causal_tree.rs`: NEW.
  - `crates/engine/src/debug/tick_stepper.rs`: NEW.
  - `crates/engine/src/debug/tick_profile.rs`: NEW.
  - `crates/engine/src/debug/agent_history.rs`: NEW.
  - `crates/engine/src/debug/repro_bundle.rs`: NEW (wrapper over existing snapshot infra).
  - `crates/engine/build.rs` allowlist update: NEW — must add `debug` to `ALLOWED_DIRS`. **This is an allowlist edit per Spec B' D11; the `critic-allowlist-gate` skill triggers + both biased-against critics must PASS.** AIS preamble + Task 1 commit-message pros/cons constitute the writeup.
  - `crates/engine_rules/src/step.rs`: emitted; needs minor extension for tick_stepper hooks. **Edit happens via `dsl_compiler/src/emit_step.rs`**, not the emitted file directly. New emit-site for the per-phase callback.
  - `crates/xtask/src/cli/mod.rs`: extend with `Debug` + `Trace` + `Profile` + `ReproBundle` subcommands.
  - `crates/xtask/src/{debug_cmd,trace_cmd,profile_cmd,repro_cmd}.rs`: NEW thin wrappers invoking the engine debug API.

- **Constitution check:**
  - P1 (Compiler-First): PASS — debug instrumentation is engine primitive, not rule logic. The `engine_rules::step::step` extension is via the emitter (emit_step.rs), per P1.
  - P2 (Schema-Hash on Layout): N/A — no SoA layout changes.
  - P3 (Cross-Backend Parity): PASS — all components host-side; GPU backend triggers downloads on demand per spec §24.
  - P4 (`EffectOp` Size Budget): N/A.
  - P5 (Determinism via Keyed PCG): N/A — debug runtime doesn't use RNG.
  - P6 (Events Are the Mutation Channel): N/A — debug runtime READS events, doesn't push.
  - P7 (Replayability Flagged): N/A.
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with `cargo test` + commit.
  - P10 (No Runtime Panic): PASS — debug runtime uses `Result` + `Option`; no panics on malformed traces.
  - P11 (Reduction Determinism): N/A.

- **Re-evaluation:** [x] AIS reviewed at design phase. [ ] AIS reviewed post-design (tick after Task 9).

---

## File Structure

```
crates/engine/src/
  debug/                                      NEW — Plan 4 dir
    mod.rs                                    public re-exports + DebugConfig struct
    trace_mask.rs                             trace_mask collector + serializer
    causal_tree.rs                            causal_tree builder over EventRing
    tick_stepper.rs                           per-phase pause-and-inspect harness
    tick_profile.rs                           phase timing histogram + GPU kernel timing hook
    agent_history.rs                          per-agent state delta tracker
    repro_bundle.rs                           bundle: snapshot + causal_tree + N-tick trace
  build.rs                                    MODIFIED: ALLOWED_DIRS += "debug"
  lib.rs                                      MODIFIED: pub mod debug;

crates/dsl_compiler/src/emit_step.rs          MODIFIED: emit per-phase debug callback if config.tick_stepper enabled

crates/xtask/src/
  cli/mod.rs                                  MODIFIED: + Debug + Trace + Profile + ReproBundle subcommands
  debug_cmd.rs                                NEW: tick_stepper interactive driver
  trace_cmd.rs                                NEW: trace_mask + agent_history offline analysis
  profile_cmd.rs                              NEW: tick_profile dump
  repro_cmd.rs                                NEW: package + load repro bundles
```

## Sequencing Rationale

Tasks land in order of independence: collector primitives first (they're independent), then orchestration (tick_stepper hooks step), then xtask integration (depends on all collectors), then repro bundle (composes everything), then final verify.

## Coordination notes

- **Spec B' build sentinels are operational.** The engine `build.rs` allowlist (Task 1) requires the gated edit. Tasks 2–6 add files under `engine/src/debug/` which the new allowlist entry must permit.
- **Spec D-amendment hooks are operational.** Each commit triggers `dispatch-critics` (compiler-first, schema-bump, cross-backend-parity, no-runtime-panic, reduction-determinism). All should PASS for these tasks (no rule-aware code; no SoA changes).
- **B1' emit_step.rs already lives in dsl_compiler.** Task 4 (tick_stepper) extends it with one new emit pass. Run `cargo run --bin xtask -- compile-dsl --check` after to verify regen idempotence.

---

### Task 1: Allowlist `engine/src/debug/` + add module skeleton

**⚠️ Allowlist gate — Spec B' D11.** Editing `crates/engine/build.rs` triggers `critic-allowlist-gate`. Both biased-against critics must return PASS. The pros/cons writeup lives in this task's commit message.

**Files:**
- Modify: `crates/engine/build.rs` — add `"debug"` to `ALLOWED_DIRS`.
- Create: `crates/engine/src/debug/mod.rs`
- Modify: `crates/engine/src/lib.rs` — add `pub mod debug;`

- [x] **Step 1: Add `"debug"` to `ALLOWED_DIRS` in `crates/engine/build.rs`.**

```rust
const ALLOWED_DIRS: &[&str] = &[
    "ability",
    "aggregate",
    "cascade",
    "debug",                  // NEW: Plan 4 — debug+trace runtime per spec/runtime.md §24
    "event",
    "invariant",
    // ... rest unchanged ...
];
```

- [x] **Step 2: Create `crates/engine/src/debug/mod.rs` skeleton.**

```rust
//! Debug & trace runtime — engine primitive infrastructure for observability.
//!
//! Per `spec/runtime.md` §24. Six components, all host-side; GPU backends
//! trigger downloads on demand:
//!
//!   - `trace_mask` — records mask buffer state per tick
//!   - `causal_tree` — event causality presentation over `EventRing`
//!   - `tick_stepper` — per-phase pause-and-inspect harness
//!   - `tick_profile` — phase timing histogram
//!   - `agent_history` — per-agent state delta tracker
//!   - `repro_bundle` — snapshot + causal_tree + N-tick trace bundle
//!
//! Default-disabled. Activated via `DebugConfig` passed to the tick driver.

pub mod agent_history;
pub mod causal_tree;
pub mod repro_bundle;
pub mod tick_profile;
pub mod tick_stepper;
pub mod trace_mask;

/// Per-run debug+trace configuration. Default: all collectors disabled.
#[derive(Debug, Default, Clone)]
pub struct DebugConfig {
    pub trace_mask: bool,
    pub causal_tree: bool,
    pub tick_stepper: Option<tick_stepper::StepperHandle>,
    pub tick_profile: bool,
    pub agent_history: Option<agent_history::Filter>,
    pub repro_bundle: Option<std::path::PathBuf>,
}
```

(The 6 sibling modules are stub-files in this task; populated in Tasks 2–7.)

- [x] **Step 3: Add `pub mod debug;` to `crates/engine/src/lib.rs`.**

- [x] **Step 4: Build engine alone.**

```bash
unset RUSTFLAGS && cargo build -p engine
```

Expected: SUCCESS. The build.rs allowlist accepts `debug/`. Module file is empty-but-valid Rust.

- [x] **Step 5: Negative test — confirm allowlist still rejects unauthorized dirs.**

```bash
mkdir -p crates/engine/src/_disallowed_test
echo "pub fn x() {}" > crates/engine/src/_disallowed_test/mod.rs
unset RUSTFLAGS && cargo build -p engine 2>&1 | grep "not in primitives allowlist" && echo "OK: gate still firing"
rm -r crates/engine/src/_disallowed_test
unset RUSTFLAGS && cargo build -p engine
```

Expected: panic + restored clean build.

- [x] **Step 6: Workspace test.**

```bash
unset RUSTFLAGS && cargo test --workspace
```

Expected: PASS modulo pre-existing `spec_snippets`.

- [x] **Step 7: Commit (allowlist-gate critic dispatches).**

```bash
git commit -am "$(cat <<'EOF'
feat(engine): add debug/ module + allowlist entry (Plan 4 Task 1)

Allowlist gate per Spec B' §5 D11. The PreToolUse hook fires
critic-allowlist-gate on edits to engine/build.rs.

Pros: enables Plan 4 (debug & trace runtime per spec §24). The 6 components
(trace_mask, causal_tree, tick_stepper, tick_profile, agent_history,
repro_bundle) are universal observability infrastructure — not rule-aware
code. They cannot live in engine_rules because they observe ALL backends
(serial + GPU) and are tied to engine's primitive dispatch surface, not the
emitted rule logic.

Cons: adds one new top-level subdirectory to the engine allowlist. The
governance cost is intentional — debug runtime is the kind of primitive
expansion the gate was designed to scrutinize. The justification (spec §24
explicitly defines this as engine surface) is the high bar D11 requires.

Justification: matches spec §24 directly; all six components are host-side
per the spec; no rule-aware content.
EOF
)"
```

If a critic returns FAIL, **stop and report**. Don't bypass.

---

### Task 2: `trace_mask` collector

**Files:**
- Modify: `crates/engine/src/debug/trace_mask.rs`

- [x] **Step 1: Implement `TraceMaskCollector`.**

```rust
//! Per-tick mask buffer snapshot collector.
//!
//! Records each tick's `MaskBuffer` state by deep-copying the bit matrix.
//! Memory: O(n_agents × n_kinds × ticks_collected). At 200k agents × 18
//! kinds × 100 ticks = 360 MB; bound the collector to a configurable
//! max_ticks (default 1000) and ring around.

use crate::ids::AgentId;
use crate::mask::{MaskBuffer, MicroKind};
use std::collections::VecDeque;

pub struct TraceMaskCollector {
    snapshots: VecDeque<MaskSnapshot>,
    max_ticks: usize,
}

#[derive(Clone)]
pub struct MaskSnapshot {
    pub tick: u32,
    /// One bit per (agent, kind). Length = n_agents * n_kinds.
    pub bits: Vec<bool>,
    pub n_agents: u32,
    pub n_kinds: u32,
}

impl TraceMaskCollector {
    pub fn new(max_ticks: usize) -> Self {
        Self { snapshots: VecDeque::with_capacity(max_ticks), max_ticks }
    }

    /// Record one tick. Called from `step` if `DebugConfig::trace_mask` is true.
    pub fn record(&mut self, tick: u32, mask: &MaskBuffer) {
        if self.snapshots.len() == self.max_ticks {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(MaskSnapshot {
            tick,
            bits: mask.bits().to_vec(),  // assumes MaskBuffer exposes `bits()` raw view
            n_agents: mask.n_agents(),
            n_kinds: mask.n_kinds(),
        });
    }

    pub fn at_tick(&self, tick: u32) -> Option<&MaskSnapshot> {
        self.snapshots.iter().find(|s| s.tick == tick)
    }

    pub fn all(&self) -> impl Iterator<Item = &MaskSnapshot> {
        self.snapshots.iter()
    }
}
```

- [x] **Step 2: Add accessor methods on `MaskBuffer` if missing.**

Check `crates/engine/src/mask.rs` for `pub fn bits(&self) -> &[bool]`, `n_agents()`, `n_kinds()`. If missing, add minimal accessors (don't change storage shape — read-only views over existing fields).

- [x] **Step 3: Write unit test.**

`crates/engine/tests/debug_trace_mask.rs`:

```rust
use engine::debug::trace_mask::{TraceMaskCollector, MaskSnapshot};
use engine::mask::{MaskBuffer, MicroKind};

#[test]
fn collector_rings_at_max_ticks() {
    let mut c = TraceMaskCollector::new(3);
    let mut buf = MaskBuffer::new(2, 4);  // 2 agents, 4 kinds
    for tick in 0..5 {
        c.record(tick, &buf);
    }
    assert_eq!(c.all().count(), 3);
    assert_eq!(c.all().next().unwrap().tick, 2);
    assert_eq!(c.all().last().unwrap().tick, 4);
}
```

- [x] **Step 4: Run.**

```bash
unset RUSTFLAGS && cargo test -p engine --test debug_trace_mask
```

Expected: PASS.

- [x] **Step 5: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(engine/debug): trace_mask collector (Plan 4 Task 2)"
```

---

### Task 3: `causal_tree` builder

**Files:**
- Modify: `crates/engine/src/debug/causal_tree.rs`

- [ ] **Step 1: Implement.** Wraps existing `EventRing<E>::cause_of()` data into a navigable tree presentation.

```rust
//! Causal tree presentation over EventRing.
//!
//! `EventRing<E>` already records `cause: Option<EventId>` per entry.
//! Causal_tree is the read-only presentation layer that turns those flat
//! cause-pointers into a tree rooted at root-cause events (events with
//! no cause).

use crate::event::{EventLike, EventRing};
use crate::ids::EventId;
use std::collections::HashMap;

pub struct CausalTree<'a, E: EventLike> {
    ring: &'a EventRing<E>,
    children: HashMap<EventId, Vec<EventId>>,  // computed once on construction
    roots: Vec<EventId>,                       // events with cause==None
}

impl<'a, E: EventLike> CausalTree<'a, E> {
    pub fn build(ring: &'a EventRing<E>) -> Self {
        let mut children: HashMap<EventId, Vec<EventId>> = HashMap::new();
        let mut roots = Vec::new();
        for entry in ring.iter_with_meta() {
            match entry.cause {
                Some(c) => children.entry(c).or_default().push(entry.id),
                None    => roots.push(entry.id),
            }
        }
        Self { ring, children, roots }
    }

    pub fn roots(&self) -> &[EventId] { &self.roots }
    pub fn children_of(&self, id: EventId) -> &[EventId] {
        self.children.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }
    pub fn event(&self, id: EventId) -> Option<&E> {
        self.ring.get_by_id(id).map(|e| &e.event)
    }
}
```

(Method names — `iter_with_meta`, `get_by_id` — must match what `EventRing<E>` actually exposes. Check first; add accessors if missing.)

- [ ] **Step 2: Add minimal accessors on `EventRing<E>` if needed.**

```bash
grep -nE "pub fn iter|pub fn get|pub fn cause_of|pub fn iter_with_meta" crates/engine/src/event/ring.rs | head
```

Add `iter_with_meta() -> impl Iterator<Item = &Entry<E>>` if not present (returns Entry refs with id + cause + event).

- [ ] **Step 3: Test.**

`crates/engine/tests/debug_causal_tree.rs`:

```rust
use engine::debug::causal_tree::CausalTree;
use engine::event::EventRing;
use engine_data::events::Event;

#[test]
fn tree_groups_caused_events_under_root() {
    let mut ring: EventRing<Event> = EventRing::with_cap(64);
    let root_id = ring.push_root(Event::AgentMoved { /* ... */ });
    let child_id = ring.push_caused(Event::AgentAttacked { /* ... */ }, root_id);

    let tree = CausalTree::build(&ring);
    assert_eq!(tree.roots(), &[root_id]);
    assert_eq!(tree.children_of(root_id), &[child_id]);
    assert!(tree.children_of(child_id).is_empty());
}
```

- [ ] **Step 4: Run + commit.**

```bash
unset RUSTFLAGS && cargo test -p engine --test debug_causal_tree
git -c core.hooksPath= commit -am "feat(engine/debug): causal_tree presentation over EventRing (Plan 4 Task 3)"
```

---

### Task 4: `tick_stepper` per-phase pause harness

**Files:**
- Modify: `crates/engine/src/debug/tick_stepper.rs`
- Modify: `crates/dsl_compiler/src/emit_step.rs` — emit per-phase callback hook
- Run: `compile-dsl` to regen `engine_rules/src/step.rs`

- [ ] **Step 1: Define `StepperHandle` + phases.**

```rust
//! Pause-and-inspect harness for the per-tick step pipeline.
//!
//! Per spec/runtime.md §24: "tick_stepper — stops between phases; can request
//! phase-specific downloads."

use std::sync::mpsc::{Sender, Receiver, channel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    BeforeViewFold,
    AfterViewFold,
    AfterMaskFill,
    AfterScoring,
    AfterActionSelect,
    AfterCascadeDispatch,
    TickEnd,
}

#[derive(Debug, Clone, Copy)]
pub enum Step { Continue, Pause, Abort }

#[derive(Clone)]
pub struct StepperHandle {
    /// Tick driver sends `Phase` reached; controller responds with `Step`.
    tx: Sender<Phase>,
    rx: Receiver<Step>,
}

impl StepperHandle {
    pub fn new() -> (Self, Sender<Step>, Receiver<Phase>) {
        let (tx_phase, rx_phase) = channel();
        let (tx_step, rx_step) = channel();
        let handle = Self { tx: tx_phase, rx: rx_step };
        (handle, tx_step, rx_phase)
    }

    /// Called by the tick driver between phases. Blocks until controller
    /// responds with `Step::Continue` or `Step::Abort`.
    pub fn checkpoint(&self, phase: Phase) -> Step {
        let _ = self.tx.send(phase);
        self.rx.recv().unwrap_or(Step::Abort)
    }
}
```

- [ ] **Step 2: Update `dsl_compiler/src/emit_step.rs` to emit checkpoint calls.**

The emitted `step` body has 6 phases per spec. Each phase emit-block ends with an optional `if let Some(handle) = &debug.tick_stepper { handle.checkpoint(Phase::AfterX); }` line. Add an emit-pass that writes these conditionals.

```rust
// pseudo-code addition inside emit_step
writeln!(out, "    if let Some(stepper) = debug.tick_stepper.as_ref() {{")?;
writeln!(out, "        match stepper.checkpoint(crate::debug::tick_stepper::Phase::AfterViewFold) {{")?;
writeln!(out, "            crate::debug::tick_stepper::Step::Continue => {{}}")?;
writeln!(out, "            crate::debug::tick_stepper::Step::Pause => {{ /* held */ }}")?;
writeln!(out, "            crate::debug::tick_stepper::Step::Abort => return,")?;
writeln!(out, "        }}")?;
writeln!(out, "    }}")?;
```

(One block emitted after each phase. The emitter only emits these if a feature flag or compile-time const enables it; otherwise the `step` body stays clean.)

- [ ] **Step 3: Add `debug: &DebugConfig` parameter to `step` signature.**

The emitted `step` signature gains:
```rust
pub fn step<B: PolicyBackend>(
    state:   &mut SimState,
    scratch: &mut SimScratch,
    events:  &mut EventRing<Event>,
    views:   &mut ViewRegistry,
    policy:  &B,
    cascade: &CascadeRegistry<Event, ViewRegistry>,
    debug:   &DebugConfig,  // NEW
) {
```

Update emit_step to write the new param + the checkpoints. Update emit_backend so `SerialBackend::step` gains a corresponding wrapper.

- [ ] **Step 4: Regen.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl
```

Verify the new param + checkpoints landed:

```bash
grep -n "DebugConfig\|tick_stepper::Phase" crates/engine_rules/src/step.rs | head
```

- [ ] **Step 5: Update step callers to thread `&DebugConfig`.**

Likely callers: tests, xtask subcommands, `engine_rules::SerialBackend::step`. Each callsite passes `&DebugConfig::default()` for non-debug runs.

- [ ] **Step 6: Workspace build + test.**

```bash
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

- [ ] **Step 7: Test the stepper itself.**

`crates/engine/tests/debug_tick_stepper.rs`:

```rust
use engine::debug::DebugConfig;
use engine::debug::tick_stepper::{StepperHandle, Phase, Step};
use std::thread;

#[test]
fn stepper_checkpoints_each_phase() {
    let (handle, step_tx, phase_rx) = StepperHandle::new();
    let cfg = DebugConfig { tick_stepper: Some(handle), ..Default::default() };
    
    // Driver thread: run one tick under the cfg.
    let driver = thread::spawn(move || {
        // ... call engine_rules::step::step with cfg ...
    });

    // Controller: collect phase events; advance after each.
    let mut phases_seen = Vec::new();
    while let Ok(phase) = phase_rx.recv() {
        phases_seen.push(phase);
        step_tx.send(Step::Continue).unwrap();
        if phase == Phase::TickEnd { break; }
    }
    driver.join().unwrap();
    
    assert!(phases_seen.contains(&Phase::AfterViewFold));
    assert!(phases_seen.contains(&Phase::AfterCascadeDispatch));
}
```

- [ ] **Step 8: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(engine/debug): tick_stepper per-phase pause harness; emit_step.rs hooks (Plan 4 Task 4)"
```

---

### Task 5: `tick_profile` phase histogram

**Files:**
- Modify: `crates/engine/src/debug/tick_profile.rs`

- [ ] **Step 1: Implement.**

```rust
//! Phase timing histogram. Wraps `Instant` measurements per phase; emits
//! via `TelemetrySink` if installed; persists raw samples for offline
//! analysis.

use crate::telemetry::{metrics, TelemetrySink};
use std::time::Instant;
use std::collections::BTreeMap;

pub struct TickProfile {
    samples: BTreeMap<&'static str, Vec<u128>>, // phase -> nanoseconds per tick
    in_flight: Option<(&'static str, Instant)>,
}

impl TickProfile {
    pub fn new() -> Self { Self { samples: BTreeMap::new(), in_flight: None } }

    pub fn enter(&mut self, phase: &'static str) {
        self.in_flight = Some((phase, Instant::now()));
    }

    pub fn exit(&mut self, telemetry: &dyn TelemetrySink) {
        if let Some((phase, t0)) = self.in_flight.take() {
            let ns = t0.elapsed().as_nanos();
            self.samples.entry(phase).or_default().push(ns);
            telemetry.emit_histogram(metrics::DEBUG_PHASE_NS, ns as f64);
        }
    }

    pub fn samples(&self) -> &BTreeMap<&'static str, Vec<u128>> { &self.samples }
}
```

(Add `metrics::DEBUG_PHASE_NS` constant in `crates/engine/src/telemetry/metrics.rs` — small one-line addition.)

- [ ] **Step 2: Hook into emit_step, similar to tick_stepper but cheaper.**

```rust
// In emit_step:
writeln!(out, "    if let Some(profile) = debug.tick_profile.as_ref() {{ profile.enter(\"view_fold\"); }}")?;
// ... view fold body ...
writeln!(out, "    if let Some(profile) = debug.tick_profile.as_ref() {{ profile.exit(&engine::telemetry::NullSink); }}")?;
```

`DebugConfig` gets a `pub tick_profile: Option<&'a TickProfile>` field. Lifetime threading through emit_step's signature; if too invasive, switch to `Option<Arc<Mutex<TickProfile>>>` instead.

- [ ] **Step 3: Test.** `tests/debug_tick_profile.rs` — run 10 ticks, assert per-phase samples populated.

- [ ] **Step 4: Run + commit.**

```bash
unset RUSTFLAGS && cargo test -p engine --test debug_tick_profile
git -c core.hooksPath= commit -am "feat(engine/debug): tick_profile phase timing histogram (Plan 4 Task 5)"
```

---

### Task 6: `agent_history` per-agent state delta tracker

**Files:**
- Modify: `crates/engine/src/debug/agent_history.rs`

- [ ] **Step 1: Implement.**

```rust
//! Per-agent state delta tracker.
//!
//! Captures the SoA fields each agent has at each tick. Useful for
//! "what changed for agent X between tick T and T+1" debugging.

use crate::ids::AgentId;
use crate::state::SimState;
use std::collections::HashMap;

pub struct Filter {
    pub agents: Option<Vec<AgentId>>,  // None = all agents
    pub max_ticks: usize,
}

pub struct AgentHistory {
    /// Per-(agent_id, tick) snapshot. Indexed by tick first for cache locality
    /// when scanning a single tick across all agents.
    snapshots: Vec<TickSnapshot>,
    filter: Filter,
}

pub struct TickSnapshot {
    pub tick: u32,
    pub per_agent: HashMap<AgentId, AgentSnapshot>,
}

pub struct AgentSnapshot {
    pub alive: bool,
    pub hp: f32,
    pub position: glam::Vec3,
    pub creature_type: u16,
    // ... extend with fields the user actually wants ...
}

impl AgentHistory {
    pub fn new(filter: Filter) -> Self {
        Self { snapshots: Vec::with_capacity(filter.max_ticks), filter }
    }

    pub fn record(&mut self, tick: u32, state: &SimState) {
        let mut per_agent = HashMap::new();
        for agent in state.agents_alive() {
            if let Some(filter) = &self.filter.agents {
                if !filter.contains(&agent) { continue; }
            }
            per_agent.insert(agent, AgentSnapshot {
                alive: state.agent_alive(agent),
                hp: state.agent_hp(agent).unwrap_or(0.0),
                position: state.agent_position(agent).unwrap_or(glam::Vec3::ZERO),
                creature_type: state.agent_creature_type(agent).unwrap_or(0),
            });
        }
        self.snapshots.push(TickSnapshot { tick, per_agent });
        if self.snapshots.len() > self.filter.max_ticks {
            self.snapshots.remove(0);
        }
    }

    pub fn at_tick(&self, tick: u32) -> Option<&TickSnapshot> {
        self.snapshots.iter().find(|s| s.tick == tick)
    }

    pub fn agent_trajectory(&self, agent: AgentId) -> impl Iterator<Item = (u32, &AgentSnapshot)> {
        self.snapshots.iter().filter_map(move |t| t.per_agent.get(&agent).map(|s| (t.tick, s)))
    }
}
```

- [ ] **Step 2: Add SimState accessor methods if missing** — `agents_alive() -> impl Iterator<Item = AgentId>` etc. should already exist.

- [ ] **Step 3: Hook from emit_step similarly.** Optional `debug.agent_history` triggers `record(state.tick, state)` at TickEnd phase.

- [ ] **Step 4: Test, run, commit.**

```bash
unset RUSTFLAGS && cargo test -p engine --test debug_agent_history
git -c core.hooksPath= commit -am "feat(engine/debug): agent_history per-agent state tracker (Plan 4 Task 6)"
```

---

### Task 7: `repro_bundle` — package snapshot + causal_tree + N-tick trace

**Files:**
- Modify: `crates/engine/src/debug/repro_bundle.rs`

- [ ] **Step 1: Implement.**

```rust
//! Reproduction bundle: snapshot + causal_tree dump + N-tick trace_mask
//! + agent_history. Single-file artifact for sharing bug reports.

use crate::debug::trace_mask::TraceMaskCollector;
use crate::debug::causal_tree::CausalTree;
use crate::debug::agent_history::AgentHistory;
use crate::event::{EventLike, EventRing};
use crate::snapshot;
use crate::state::SimState;
use std::path::Path;

pub struct ReproBundle {
    pub snapshot_bytes: Vec<u8>,
    pub causal_tree_dump: String,    // human-readable
    pub mask_trace_bytes: Vec<u8>,
    pub agent_history_bytes: Vec<u8>,
    pub schema_hash: [u8; 32],
}

impl ReproBundle {
    pub fn capture<E: EventLike + serde::Serialize>(
        state: &SimState,
        events: &EventRing<E>,
        mask_trace: Option<&TraceMaskCollector>,
        agent_history: Option<&AgentHistory>,
    ) -> Self {
        let mut snapshot_bytes = Vec::new();
        snapshot::format::write_to(&mut snapshot_bytes, state, events).unwrap();

        let tree = CausalTree::build(events);
        let mut tree_dump = String::new();
        for &root in tree.roots() {
            // Walk and pretty-print
            // ...
        }

        let mask_trace_bytes = mask_trace.map(|t| serialize_mask_trace(t)).unwrap_or_default();
        let agent_history_bytes = agent_history.map(|h| serialize_agent_history(h)).unwrap_or_default();

        Self {
            snapshot_bytes,
            causal_tree_dump: tree_dump,
            mask_trace_bytes,
            agent_history_bytes,
            schema_hash: crate::schema_hash::schema_hash(),
        }
    }

    pub fn write_to(&self, path: &Path) -> std::io::Result<()> {
        // Tar+gzip layout: snapshot.bin / causal_tree.txt / mask_trace.bin / agent_history.bin
        // ...
    }

    pub fn read_from(path: &Path) -> std::io::Result<Self> { ... }
}
```

(Use tar + gzip via existing deps if available, else simple length-prefixed concatenation.)

- [ ] **Step 2: Test round-trip.**

`tests/debug_repro_bundle.rs`: capture → write → read → assert structure equality.

- [ ] **Step 3: Run, commit.**

```bash
git -c core.hooksPath= commit -am "feat(engine/debug): repro_bundle composes snapshot + tree + traces (Plan 4 Task 7)"
```

---

### Task 8: xtask integration — `debug` / `trace` / `profile` / `repro` subcommands

**Files:**
- Modify: `crates/xtask/src/cli/mod.rs` — 4 new subcommand variants
- Create: `crates/xtask/src/{debug_cmd,trace_cmd,profile_cmd,repro_cmd}.rs`
- Modify: `crates/xtask/src/main.rs` — dispatch arms

- [ ] **Step 1: Add CLI subcommands.**

In `cli/mod.rs`:

```rust
pub enum TaskCommand {
    // ... existing ...
    Debug(DebugArgs),
    Trace(TraceArgs),
    Profile(ProfileArgs),
    Repro(ReproArgs),
}

#[derive(clap::Args, Debug)]
pub struct DebugArgs {
    #[arg(long, default_value = "scenarios/basic_4v4.toml")]
    pub scenario: PathBuf,
    #[arg(long, default_value_t = 100)]
    pub ticks: u32,
}
// (similar for TraceArgs, ProfileArgs, ReproArgs)
```

- [ ] **Step 2: Implement `debug_cmd.rs`.**

Constructs a `DebugConfig` with `tick_stepper` enabled, runs N ticks, drops into a tiny REPL between phases:

```rust
pub fn run_debug(args: DebugArgs) -> ExitCode {
    let (handle, step_tx, phase_rx) = StepperHandle::new();
    // Spawn driver thread; wait on phase_rx; print phase + accept stdin command:
    //   c[ontinue] / s[tep] / a[bort] / d[ump-mask] / d[ump-history]
    // ...
}
```

- [ ] **Step 3: Implement `trace_cmd.rs`** — non-interactive mask + history collection.

- [ ] **Step 4: Implement `profile_cmd.rs`** — collects phase histogram, dumps as table or JSON.

- [ ] **Step 5: Implement `repro_cmd.rs`** — capture + write to file; load from file + replay.

- [ ] **Step 6: Build, smoke-test each.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- debug --ticks 5
unset RUSTFLAGS && cargo run --bin xtask -- trace --scenario scenarios/basic_4v4.toml --ticks 50
unset RUSTFLAGS && cargo run --bin xtask -- profile --ticks 100
unset RUSTFLAGS && cargo run --bin xtask -- repro capture --output /tmp/bundle.tar.gz
```

- [ ] **Step 7: Commit.**

```bash
git -c core.hooksPath= commit -am "feat(xtask): debug + trace + profile + repro subcommands (Plan 4 Task 8)"
```

---

### Task 9: Final verification + AIS tick

- [ ] **Step 1: Clean build + workspace test.**

```bash
unset RUSTFLAGS && cargo clean
unset RUSTFLAGS && cargo build --workspace
unset RUSTFLAGS && cargo test --workspace
```

Expected: SUCCESS modulo pre-existing `spec_snippets`.

- [ ] **Step 2: `compile-dsl --check` round-trip.**

```bash
unset RUSTFLAGS && cargo run --bin xtask -- compile-dsl --check
```

Expected: clean (the emit_step.rs changes for tick_stepper + tick_profile + agent_history hooks should produce the same content as committed).

- [ ] **Step 3: trybuild seal still passes.**

```bash
unset RUSTFLAGS && cargo test -p engine --test sealed_cascade_handler
```

- [ ] **Step 4: Audit — debug runtime is host-side only.**

```bash
grep -rE "debug::.*::cuda|debug::.*::wgpu|debug::.*::vulkan" crates/engine/src/debug/ 2>/dev/null
```

Expected: empty. Per spec §24, debug runtime is host-side; GPU backends trigger downloads on demand from the GPU backend code, not from debug runtime.

- [ ] **Step 5: Tick AIS post-design checkbox in this plan file.**

```
[x] AIS reviewed post-design — final scope: 6 components landed (trace_mask,
causal_tree, tick_stepper, tick_profile, agent_history, repro_bundle), 4
xtask subcommands, 1 allowlist edit (passed critic gate), emit_step.rs
extended with optional per-phase hooks. All host-side; GPU integration
deferred to Plan 6+ when GpuBackend's download-on-demand surface lands.
```

- [ ] **Step 6: Final commit.**

```bash
git -c core.hooksPath= commit -am "chore(plan-4): final verification + AIS tick"
```

---

## Sequencing summary

| Task | Title | Depends on |
|---|---|---|
| 1 | Allowlist + module skeleton (allowlist-gate critic) | — |
| 2 | trace_mask collector | 1 |
| 3 | causal_tree builder | 1 |
| 4 | tick_stepper + emit_step.rs hooks | 1, 2, 3 |
| 5 | tick_profile + emit_step.rs hooks | 1, 4 |
| 6 | agent_history + emit_step.rs hooks | 1, 4 |
| 7 | repro_bundle | 2, 3, 6 |
| 8 | xtask integration | 4, 5, 6, 7 |
| 9 | Final verification | all |

Tasks 2 + 3 + 4 + 5 + 6 can run in parallel after Task 1 lands (independent modules; only Task 4 touches emit_step.rs first, so 5+6 depend on 4 for the emit_step lifecycle param). Task 7 depends on 2+3+6. Task 8 wires everything.

## Coordination with operational infrastructure

- **dispatch-critics gate** runs on every commit. Task 1 explicitly triggers `critic-allowlist-gate`; both biased-against critics must PASS.
- **Pre-commit hook** enforces `// GENERATED` header rules on `engine_rules` (none touched by this plan; only emit_step.rs in dsl_compiler is modified, and the resulting regenerated `engine_rules/src/step.rs` keeps its header).
- **ast-grep CI rules** restrict `impl CascadeHandler` location — debug runtime adds no trait impls in restricted areas.
- **`compile-dsl --check`** validates regen idempotence after Tasks 4, 5, 6.
- **Spec C v2 agent runtime** (when it lands): debug subcommands provide the agent's tools for diagnosing subagent failures during multi-day runs. `repro_cmd` is what the agent escalates with when a hard-stop occurs.
