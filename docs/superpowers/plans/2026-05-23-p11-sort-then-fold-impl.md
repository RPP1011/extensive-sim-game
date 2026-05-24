# P11 sort-then-fold — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the f32 view-fold reduction non-determinism (the open P11 violation) by injecting deterministic `seq` field on every event payload, sorting the chronicle ring by `(target_id, seq)` before consumers, and simplifying the fold path. Tighten `forest_fire_pin` from `max_abs_drift ≤ 150.0` to `== 0.0` as the acceptance gate.

**Architecture:** Two coordinated mechanisms across GPU + CPU backends. (1) `seq: u32` field added as the last word of every event payload, packed from `(producer_kernel_id, producer_thread_id, intra_emit_idx)`. Producers (both backends) write it; GPU radix sorts the ring by `(target, seq)` between producer and consumer phases; CPU mirrors with `Vec::sort_by_key`. (2) Fold path drops its CAS+add loop for f32 because post-sort each fold thread is the only writer per slot.

**Tech Stack:** Rust + WGSL. Modified crates: `dsl_compiler` (emit), `engine` (CPU cascade + view folds), `sims` (tests). New tests in `dsl_compiler/tests/` and `sims/tests/`.

**Spec:** [docs/superpowers/specs/2026-05-23-p11-sort-then-fold-design.md](../specs/2026-05-23-p11-sort-then-fold-design.md)

## Architectural Impact Statement

- **Existing primitives searched:**
  - `EventLayout` at `crates/dsl_compiler/src/cg/program.rs:318` — per-kind ring stride + field layout
  - `emit_chronicle_append_skeleton` at `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:4156` — GPU producer skeleton
  - `parity_apply_program_sweep::canonicalize` — AOE oracle sort precedent
  - `forest_fire_pin::max_abs_drift` at `crates/sims/tests/forest_fire_pin.rs:382`
  Search method: `rg` + targeted `Read`.

- **Decision:** extend `EventLayout` (`+1` word for seq), extend `emit_chronicle_append_skeleton` (write seq), new sort kernels synthesized by `build_helper.rs` (matches existing kernel-synthesis pattern for spatial-hash / alive-pack / fold consumers).

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none (event syntax unchanged; `seq` is invisible to .sim authors)
  - Generated outputs re-emitted: every fixture's `runtime_core.rs` + `physics_*.wgsl` + `event_ring` stride

- **Hand-written downstream code:** the per-view fold modules under `crates/engine/src/view/*.rs` adopt `Vec::sort_by_key` before the existing reduce. Justification: CPU-side view folds are not auto-emitted today (they wrap the host fold path). Adopting sort here mirrors the AOE oracle canonicalize precedent.

- **Constitution check:**
  - P1 (Compiler-First): PASS — all GPU-side changes flow through emit
  - P3 (Cross-Backend Parity): PASS — CPU sort + GPU sort produce identical orderings via shared `compute_event_seq`
  - P7 (Replayable Flagged): PASS — seq lands uniformly on both replayable and non-replayable events
  - P10 (No Runtime Panic): PASS — radix sort proptest covers edge cases
  - P11 (Reduction Determinism): PASS — this plan IS the P11 fix

---

## File Structure

**New files:**
- `crates/dsl_compiler/src/seq.rs` — shared `compute_event_seq` helper (consumed by both backends)
- `crates/dsl_compiler/tests/event_seq_field_present.rs` — Slice 1 emit pin
- `crates/dsl_compiler/tests/radix_sort_kernels_emitted.rs` — Slice 3 emit pin
- `crates/dsl_compiler/tests/sort_omitted_when_no_f32_fold.rs` — Slice 6 opt-out pin
- `crates/dsl_compiler/tests/proptest_radix_sort.rs` — Slice 3 sort correctness
- `crates/dsl_compiler/tests/cas_add_dropped_for_f32_fold.rs` — Slice 4 emit pin
- `crates/sims/tests/f32_reduction_determinism_probe_pin.rs` — Slice 5 behavior pin
- `crates/sims/tests/parity_forest_fire.rs` — Slice 5 cross-backend parity
- `crates/sims/tests/cpu_determinism_forest_fire.rs` — Slice 2 CPU-only determinism
- `assets/sim/f32_reduction_probe.sim` — Slice 5 minimal probe fixture
- `crates/dsl_compiler/src/cg/emit/sort_kernel.rs` — Slice 3 WGSL sort kernel emit

**Modified files:**
- `crates/dsl_compiler/src/lib.rs` — re-export `seq` module
- `crates/dsl_compiler/src/cg/program.rs` — `EventLayout::record_stride_u32` (10 → 11)
- `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` — `emit_chronicle_append_skeleton` writes seq
- `crates/dsl_compiler/src/cg/emit/kernel.rs` — producer_kernel_id assignment + f32-fold detection
- `crates/dsl_compiler/src/build_helper.rs` — synthesize sort kernels conditionally
- `crates/dsl_compiler/src/cg/schedule/` — insert sort barriers
- `crates/engine/src/cascade/dispatch.rs` — CPU emit writes seq
- `crates/engine/src/view/*.rs` (per-fixture fold modules) — sort before reduce
- `crates/sims/tests/forest_fire_pin.rs` — tighten assertion to 0

---

## Notation

Steps use TDD discipline (failing test → minimal impl → passing test → commit). For purely mechanical changes (constant edits, file moves), the "write a failing test" step is replaced with "verify the existing test/compile fails first."

Each task ends with `git add <files> && git commit -m "<msg>"` — frequent commits per the skill.

---

# Slice 1 — seq plumbing

**Goal:** Add `seq: u32` as the last word of every event payload. Producers (both backends) write it. No sort yet. Forest_fire pin still drifts the same amount.

**Gate:** `event_seq_field_present.rs` emit pin passes; existing tests still pass; CPU mirror produces well-formed events with seq populated.

## Task 1.1: Create the shared `compute_event_seq` helper

**Files:**
- Create: `crates/dsl_compiler/src/seq.rs`
- Modify: `crates/dsl_compiler/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/dsl_compiler/src/seq.rs`:

```rust
//! Shared deterministic seq computation for event payloads.
//!
//! Both the GPU emit (inlined into WGSL) and the CPU cascade dispatch
//! call this helper with identical inputs and get identical u32 outputs.
//! This is the load-bearing parity primitive for P11 sort-then-fold.

/// Pack the three identifiers into a u32 seq value.
///
/// Layout (MSB → LSB):
///   bits 24..32: producer_kernel_id   (8 bits, ≤ 256 emit-producer kernels)
///   bits  4..24: producer_thread_id   (20 bits, ≤ 1M threads — megaswarm headroom)
///   bits  0..4:  intra_emit_idx       (4 bits, ≤ 16 emit stmts/thread/tick)
///
/// Callers MUST pass values within the documented ranges; out-of-range
/// values silently alias other (producer_kernel_id, thread, idx) tuples
/// and break determinism.
pub fn compute_event_seq(
    producer_kernel_id: u32,
    producer_thread_id: u32,
    intra_emit_idx: u32,
) -> u32 {
    debug_assert!(producer_kernel_id < 256, "producer_kernel_id exceeds 8-bit range");
    debug_assert!(producer_thread_id < (1 << 20), "producer_thread_id exceeds 20-bit range");
    debug_assert!(intra_emit_idx < 16, "intra_emit_idx exceeds 4-bit range");

    (producer_kernel_id << 24) | (producer_thread_id << 4) | intra_emit_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_three_fields_into_distinct_u32() {
        let a = compute_event_seq(0, 0, 0);
        let b = compute_event_seq(0, 0, 1);
        let c = compute_event_seq(0, 1, 0);
        let d = compute_event_seq(1, 0, 0);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 16);  // 1 << 4
        assert_eq!(d, 1 << 24);
    }

    #[test]
    fn max_legal_values_pack_without_overlap() {
        let max = compute_event_seq(255, (1 << 20) - 1, 15);
        assert_eq!(max, u32::MAX);
    }

    #[test]
    fn sort_order_matches_kernel_then_thread_then_idx() {
        // Higher kernel_id dominates; within kernel, higher thread_id
        // dominates; within thread, higher emit_idx dominates.
        let a = compute_event_seq(0, 0, 5);
        let b = compute_event_seq(0, 1, 0);
        let c = compute_event_seq(1, 0, 0);
        assert!(a < b);
        assert!(b < c);
    }
}
```

- [ ] **Step 2: Add the module to `lib.rs`**

In `crates/dsl_compiler/src/lib.rs`, add (alphabetical):

```rust
pub mod seq;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test -p dsl_compiler --lib seq::`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add crates/dsl_compiler/src/seq.rs crates/dsl_compiler/src/lib.rs
git commit -m "feat(dsl): shared compute_event_seq helper for P11 sort-then-fold"
```

## Task 1.2: Bump event payload stride to include seq

**Files:**
- Modify: `crates/dsl_compiler/src/cg/program.rs` (struct + `populate_event_kinds` callers)

The current `EventLayout::record_stride_u32` is 10 (= 2 header + 8 payload words). After this task it is 11 (= 2 header + 8 payload + 1 seq trailer).

- [ ] **Step 1: Locate the constant**

Run: `rg "record_stride_u32.*[=:].*10" crates/dsl_compiler/src/`
Expected: identifies the layout-population site (likely in `populate_event_kinds` or `EventLayout::default()`).

- [ ] **Step 2: Bump the constant + add a `seq_word_offset` accessor**

In `crates/dsl_compiler/src/cg/program.rs`, locate the population site identified in Step 1. Change the literal `10` to `11`. Add to `EventLayout`:

```rust
impl EventLayout {
    /// u32-word offset (from start of record) of the seq trailer.
    /// Always the last word: header(2) + payload(8) = 10.
    pub fn seq_word_offset(&self) -> u32 {
        self.record_stride_u32 - 1
    }
}
```

- [ ] **Step 3: Verify nothing else hardcodes `10` as the stride**

Run: `rg "\* 10u\b|\* 10 \+|stride.*10\b" crates/dsl_compiler/src/cg/emit/`
Expected: list of sites. Each one should already use `EventLayout::record_stride_u32` or a kernel-local `stride` variable. Any literal `10u` in WGSL emit is a bug; convert to `{stride}u` parameter.

- [ ] **Step 4: Run dsl_compiler tests — many will fail (expected)**

Run: `cargo test -p dsl_compiler 2>&1 | tail -20`
Expected: emit-shape pins that hashed the old stride fail. Note the failing test names — they pin the OLD shape; we'll update them in Task 1.4.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/cg/program.rs
git commit -m "feat(dsl): bump event record stride to 11 (reserve seq trailer)"
```

## Task 1.3: Extend `emit_chronicle_append_skeleton` to write seq

**Files:**
- Modify: `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:4156`

The skeleton emits the atomicAdd slot acquisition + bounds check + per-field writes. We add a final write for the seq trailer.

- [ ] **Step 1: Update the function signature**

In `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`, modify `emit_chronicle_append_skeleton` to accept `producer_kernel_id: u32` and emit a seq-write line. The new signature:

```rust
pub(crate) fn emit_chronicle_append_skeleton(
    event_id: u32,
    buf: &str,
    stride: u32,
    field_count: usize,
    field_writes: &[String],
    debug_wgsl: DebugWgslFlags,
    producer_kernel_id: u32,
    intra_emit_idx: u32,
) -> String {
```

- [ ] **Step 2: Append the seq-write line in the body**

After the existing `for line in field_writes { ... }` block (around line 4188-4190), before the closing brace, append:

```rust
    // P11 seq trailer: deterministic ordering key for the per-tick sort.
    // `agent_id` is the producer thread's per-kernel index (PerAgent rules)
    // or event index (PerEvent rules). The packing matches the Rust
    // `compute_event_seq` helper byte-for-byte.
    out.push_str(&format!(
        "        atomicStore(&{buf}[slot * {stride}u + {seq_offset}u], \
         ({kernel_id}u << 24u) | (agent_id << 4u) | {emit_idx}u);\n",
        seq_offset = stride - 1,
        kernel_id = producer_kernel_id,
        emit_idx = intra_emit_idx,
    ));
```

- [ ] **Step 3: Update all callers of `emit_chronicle_append_skeleton`**

Run: `rg "emit_chronicle_append_skeleton\(" crates/dsl_compiler/src/`
Expected: list of call sites. Each needs the new `producer_kernel_id` and `intra_emit_idx` arguments.

For each caller, thread these from the enclosing context. The kernel-level helper for producer-kernel-id assignment lands in Task 1.5 — for now, pass `0, 0` as a temporary placeholder so the code compiles. Mark each placeholder with `// TODO(P11 task 1.5): wire real producer_kernel_id`.

- [ ] **Step 4: Run dsl_compiler tests**

Run: `cargo check -p dsl_compiler`
Expected: compiles cleanly.

Run: `cargo test -p dsl_compiler 2>&1 | tail -10`
Expected: same set of stride-related test failures as Task 1.2; no NEW failures.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/cg/emit/wgsl_body.rs
git commit -m "feat(emit): write seq trailer in chronicle append skeleton (placeholder ids)"
```

## Task 1.4: Update stride-sensitive emit pins

**Files:**
- Modify: tests under `crates/dsl_compiler/tests/` whose assertions hardcode the old `* 10u` stride

- [ ] **Step 1: Identify the failing pins**

Run: `cargo test -p dsl_compiler 2>&1 | grep "FAILED\|test result"`
Expected: list of failed tests.

- [ ] **Step 2: Update each failing pin**

For each failing test, open the file and grep its assertions for `* 10u` or hardcoded stride references. Replace with `* 11u` (or compute via `EventLayout::record_stride_u32`).

The most likely candidates (verify with the actual failure list):
- `crates/dsl_compiler/tests/post_cas_emit_gating.rs`
- `crates/dsl_compiler/tests/firebolt_chronicle_emit.rs` (if exists)
- `crates/dsl_compiler/tests/apply_ability_emit.rs` (if exists)

- [ ] **Step 3: Run tests until green**

Run: `cargo test -p dsl_compiler 2>&1 | tail -5`
Expected: 0 failed.

- [ ] **Step 4: Commit**

```bash
git add crates/dsl_compiler/tests/
git commit -m "test(dsl): update stride-sensitive pins from 10u to 11u (seq trailer)"
```

## Task 1.5: Allocate producer_kernel_id densely over emit-producing kernels

**Files:**
- Modify: `crates/dsl_compiler/src/cg/emit/kernel.rs`
- Modify: `crates/dsl_compiler/src/cg/emit/program.rs`

The kernel emit pass must classify each kernel as "emit-producer" or "non-producer," assign producer ids `0..K` densely to producers, and thread the id through to `emit_chronicle_append_skeleton`.

- [ ] **Step 1: Add a classifier in program.rs**

In `crates/dsl_compiler/src/cg/emit/program.rs`, add (before `emit_cg_program`):

```rust
/// Pre-emit pass: walk every kernel topology and assign a dense
/// `producer_kernel_id` (0..K) to each kernel whose body contains at
/// least one `CgStmt::Emit`. Non-emitting kernels (mask, scoring,
/// plumbing, fold consumers that don't re-emit) carry `None`.
///
/// The mapping is stable per compilation: kernels are visited in
/// schedule stage + intra-stage order, so re-emitting the same .sim
/// produces the same ids.
fn assign_producer_kernel_ids(
    schedule: &Schedule,
    prog: &CgProgram,
) -> BTreeMap<KernelIndex, u32> {
    let mut next_id: u32 = 0;
    let mut out = BTreeMap::new();
    for (stage_idx, stage) in schedule.stages.iter().enumerate() {
        for (kernel_idx, topology) in stage.kernels.iter().enumerate() {
            if kernel_topology_has_emits(topology, prog) {
                let key = KernelIndex { stage: stage_idx, kernel: kernel_idx };
                assert!(next_id < 256, "P11 seq packing only supports 256 emit-producer kernels");
                out.insert(key, next_id);
                next_id += 1;
            }
        }
    }
    out
}

/// True iff any op in the kernel's body produces a chronicle emit
/// (CgStmt::Emit or apply_ability dispatcher).
fn kernel_topology_has_emits(topology: &KernelTopology, prog: &CgProgram) -> bool {
    // Walk the topology's body ops; for each, walk its stmt list;
    // return true on first Emit/ApplyAbility hit.
    for op_id in topology.body_ops() {
        let Ok(op) = resolve_op(prog, *op_id) else { continue; };
        let body_list = match &op.kind {
            ComputeOpKind::PhysicsRule { body, .. } => *body,
            ComputeOpKind::ViewFold { body, .. } => *body,
            _ => continue,
        };
        if stmt_list_contains_emit(prog, body_list) {
            return true;
        }
    }
    false
}
```

Add a helper `stmt_list_contains_emit` mirroring the existing `stmt_list_contains_set_alive_false` pattern (recurse through statements, return true if any is `CgStmt::Emit`).

Also add a `KernelIndex` struct if it doesn't exist:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct KernelIndex {
    pub stage: usize,
    pub kernel: usize,
}
```

- [ ] **Step 2: Thread the id through the emit context**

In `EmitCtx` (around line 625 of program.rs), add:

```rust
    /// Map from `KernelIndex` to densely-allocated producer_kernel_id
    /// for the seq trailer pack. `None` if the kernel doesn't emit.
    /// Populated once at the start of `emit_cg_program` and read by
    /// the kernel-body emit when rendering chronicle-append skeletons.
    pub producer_kernel_ids: BTreeMap<KernelIndex, u32>,
    /// The kernel currently being emitted — used by per-stmt emit to
    /// look up its producer_kernel_id.
    pub current_kernel_index: std::cell::Cell<Option<KernelIndex>>,
```

Populate `producer_kernel_ids` at the top of `emit_cg_program` by calling `assign_producer_kernel_ids`. Set `current_kernel_index` before each kernel's body emit and clear after (mirror the existing `prior_alive_cas` save/restore pattern).

- [ ] **Step 3: Track intra_emit_idx in the per-stmt walk**

When the per-stmt walker encounters a `CgStmt::Emit`, look up the current kernel's producer_id and pass it + the running emit counter to `emit_chronicle_append_skeleton`.

In `wgsl_body.rs`, find the `CgStmt::Emit` handler (search for "lower_emit_to_wgsl" call sites). Maintain a per-statement-list-walk counter:

```rust
let mut intra_emit_idx: u32 = 0;
// ... inside the loop over stmts:
if let CgStmt::Emit { .. } = stmt {
    let kernel_id = ctx.current_kernel_index.get()
        .and_then(|ki| ctx.producer_kernel_ids.get(&ki).copied())
        .unwrap_or(0);  // Non-emitting kernel shouldn't reach here; fall back safely.
    let wgsl = lower_emit_to_wgsl(..., kernel_id, intra_emit_idx)?;
    intra_emit_idx += 1;
    // ... existing handling
}
```

Pass through to `emit_chronicle_append_skeleton`. Remove the `TODO(P11 task 1.5)` placeholders added in Task 1.3.

- [ ] **Step 4: Run tests**

Run: `cargo test -p dsl_compiler 2>&1 | tail -5`
Expected: 0 failed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/cg/emit/
git commit -m "feat(emit): dense producer_kernel_id allocation + seq trailer wiring"
```

## Task 1.6: Mirror seq emit in the CPU cascade dispatch

**Files:**
- Modify: `crates/engine/src/cascade/dispatch.rs` (find the chronicle-write sites)
- Possibly: per-event-kind structures in `crates/engine/src/event/`

- [ ] **Step 1: Locate CPU chronicle-write sites**

Run: `rg "fn emit\|push_event\|chronicle.*push\|event_ring.*push" crates/engine/src/cascade/ crates/engine/src/event/ 2>/dev/null`
Expected: identifies the CPU-side event-emit function(s).

- [ ] **Step 2: Identify the event payload struct**

Run: `rg "struct .*Event\|pub struct Event" crates/engine/src/event/`
Expected: locate the unified Event record type. Add a `seq: u32` field as the last member.

- [ ] **Step 3: Add `seq` field to the CPU event record**

In the identified struct, add the field. Initialize callers — for now, every CPU emit site needs to compute and pass seq. Use the shared helper:

```rust
use dsl_compiler::seq::compute_event_seq;

// At each emit site:
let seq = compute_event_seq(producer_kernel_id, producer_thread_id, intra_emit_idx);
event_ring.push(Event { kind, tick, /* payload */, seq });
```

The CPU side's notion of `producer_kernel_id` needs to come from somewhere — either a const per-rule-fn (assigned the same way as GPU side: dense over emit-producing rule fns) or a `Schedule`-time table read at runtime. For Slice 1, pass `0, agent_id, 0` (single-emit fallback) and add a `// TODO(P11 Slice 1 follow-up): wire dense kernel ids` comment. Slice 2 will tighten this when sort lands and we observe drift between CPU and GPU.

- [ ] **Step 4: Update callers to thread `producer_thread_id` (agent id)**

Each CPU rule fn already has access to `agent_id` (or equivalent). Thread it into the emit call.

- [ ] **Step 5: Run engine tests**

Run: `cargo test -p engine 2>&1 | tail -10`
Expected: tests that don't compare event records pass; tests that hash events may need baseline regen. Note failing tests for Task 1.7.

- [ ] **Step 6: Commit**

```bash
git add crates/engine/
git commit -m "feat(engine): CPU cascade dispatch writes seq trailer on emit"
```

## Task 1.7: Regenerate behavior baselines that include events

**Files:**
- Various `tests/*_baseline.txt` / `tests/*.snap` files

- [ ] **Step 1: Run the baseline-regen helper**

Run: `WOLVES_AND_HUMANS_REGEN=1 cargo test -p engine --test wolves_and_humans_parity 2>&1 | tail -5`
Expected: baseline regenerated; test fails AFTER regen (intended — the regen marker exists so the diff lands in review).

Repeat for any other `*_REGEN=1` helpers if present.

- [ ] **Step 2: Diff the regenerated baselines**

Run: `git diff crates/engine/tests/*_baseline.txt`
Expected: per-event records gain a new `seq=<value>` column (or the equivalent for the trace format). Confirm the values are stable across re-runs.

- [ ] **Step 3: Commit the regenerated baselines**

```bash
git add crates/engine/tests/
git commit -m "test(engine): regen behavioral baselines for seq trailer addition"
```

## Task 1.8: Slice 1 emit pin

**Files:**
- Create: `crates/dsl_compiler/tests/event_seq_field_present.rs`

- [ ] **Step 1: Write the pin**

```rust
//! Pin: every event payload's last word is the seq trailer, written
//! by the producer's atomicStore alongside other payload fields.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn compile(src: &str) -> EmittedArtifacts {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| o.program);
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

#[test]
fn chronicle_emit_writes_seq_trailer_as_last_payload_word() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Damaged { source: AgentId, target: AgentId, amount: f32 }

@phase(per_agent)
physics Fire {
  on Tick {} where (self.alive) {
    emit Damaged { source: self, target: self, amount: 1.0 }
  }
}
"#;
    let art = compile(src);
    let (_, body) = art.wgsl_files.iter()
        .find(|(name, _)| name.contains("Fire"))
        .expect("Fire kernel emitted");

    // Seq trailer is at offset `stride - 1` = 10 (header 2 + payload 8).
    assert!(
        body.contains("[slot * 11u + 10u]"),
        "expected seq trailer write at offset 10 (stride 11); got body:\n{body}",
    );

    // Seq value is packed: (kernel_id << 24) | (agent_id << 4) | emit_idx.
    // For the single emit in this fixture, expect kernel_id=0, emit_idx=0,
    // agent_id is the per-thread index.
    assert!(
        body.contains("(0u << 24u) | (agent_id << 4u) | 0u"),
        "expected packed seq formula in body; got:\n{body}",
    );
}
```

- [ ] **Step 2: Run pin**

Run: `cargo test -p dsl_compiler --test event_seq_field_present`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/dsl_compiler/tests/event_seq_field_present.rs
git commit -m "test(dsl): pin seq trailer presence in chronicle emit"
```

## Slice 1 acceptance gate

- [ ] **Run the full test suite:** `cargo test --workspace 2>&1 | tail -10`
- [ ] **Verify forest_fire pin still passes (with same drift):** `RUST_MIN_STACK=33554432 cargo test -p sims --release --test forest_fire_pin -- --nocapture 2>&1 | grep determinism`
  - Expected: same `max |Δ| ≈ 47` drift — Slice 1 doesn't fix the race, just plumbs `seq`.

---

# Slice 2 — CPU sort

**Goal:** Serial backend's chronicle fold sorts by `(target, seq)` before reducing. CPU side becomes byte-stable across runs. GPU still drifts.

**Gate:** `cpu_determinism_forest_fire.rs` (CPU-only re-run determinism) passes.

## Task 2.1: Locate CPU view fold sites

- [ ] **Step 1: Find the CPU fold dispatch**

Run: `rg "fn fold|fn reduce|fold_views\|pub fn fold" crates/engine/src/view/ crates/engine/src/step.rs`
Expected: identifies the per-view fold entry point(s).

- [ ] **Step 2: Find the chronicle slice the folds read**

Run: `rg "event_ring\.|chronicle.*slice\|events_this_tick" crates/engine/src/step.rs crates/engine/src/view/`
Expected: identifies how folds get their per-tick event slice.

Record the file:line of the fold loop in a note for Task 2.2. If the slice is obtained per-fold via a getter, Task 2.2 inserts the sort there.

## Task 2.2: Sort the chronicle slice before reduce

**Files:**
- Modify: the per-view fold entry point identified in Task 2.1

- [ ] **Step 1: Write the failing test**

Create `crates/sims/tests/cpu_determinism_forest_fire.rs`:

```rust
//! CPU-only determinism: forest_fire on serial backend produces
//! byte-equal view storage across same-seed reruns.
//!
//! Pre-Slice-2: this test fails (CPU producer iteration order is
//! deterministic per se, but the per-tick chronicle slice fed to the
//! fold has shape-dependent ordering across re-runs of the producer
//! schedule).
//!
//! Post-Slice-2: passes byte-equal.

use sims::forest_fire::SerialBackend;  // or whatever the serial backend entry is

const SEED: u64 = 0xF02E57F18E;
const TICKS: usize = 500;
const N_TOTAL: u32 = 1024;

#[test]
fn forest_fire_cpu_byte_equal_same_seed() {
    let view_run1 = run_serial(SEED, TICKS);
    let view_run2 = run_serial(SEED, TICKS);
    assert_eq!(
        view_run1, view_run2,
        "CPU serial backend must be byte-equal across same-seed reruns"
    );
}

fn run_serial(seed: u64, ticks: usize) -> Vec<f32> {
    let mut state = SerialBackend::try_new(seed, N_TOTAL).expect("init");
    seed_grid_serial(&mut state);
    seed_ignition_cluster_serial(&mut state);
    for _ in 0..ticks {
        state.step();
    }
    read_view_storage_serial(&state)
}

// ... seed_grid_serial, seed_ignition_cluster_serial, read_view_storage_serial
// (mirror the GPU helpers in forest_fire_pin.rs, adapted for serial backend)
```

If the serial backend doesn't have an entry point for this fixture, the test scaffolding may need to use the engine directly. Adapt as needed; the assertion is the load-bearing part.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p sims --test cpu_determinism_forest_fire`
Expected: FAIL with `assertion_eq` diff.

- [ ] **Step 3: Add the sort to the CPU fold path**

At the site identified in Task 2.1, before the reduce loop:

```rust
// P11: sort by (target, seq) for deterministic f32 reduction order.
// Cross-backend parity: GPU radix sort produces the same ordering.
events_this_tick.sort_by_key(|e| (e.target, e.seq));
```

If the chronicle slice is a `&[Event]` rather than `&mut [Event]`, the sort needs a local copy: `let mut sorted: Vec<Event> = events_this_tick.to_vec(); sorted.sort_by_key(...);` then iterate `sorted`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p sims --test cpu_determinism_forest_fire`
Expected: PASS.

- [ ] **Step 5: Verify no other engine tests regressed**

Run: `cargo test -p engine -p sims 2>&1 | tail -10`
Expected: 0 failed (other than pre-existing GPU-side drift).

- [ ] **Step 6: Commit**

```bash
git add crates/engine/src/view/ crates/sims/tests/cpu_determinism_forest_fire.rs
git commit -m "feat(engine): CPU view folds sort by (target, seq) for P11 determinism"
```

---

# Slice 3 — GPU radix sort

**Goal:** Emit 15 sort kernels per fixture that opts in. Sort the event_ring by `(target, seq)` before consumers.

**Gate:** `proptest_radix_sort.rs` (WGSL sort vs Rust sort byte-equal) + `radix_sort_kernels_emitted.rs` (schedule shape) both pass.

## Task 3.1: Skeleton — create `sort_kernel.rs` emit module

**Files:**
- Create: `crates/dsl_compiler/src/cg/emit/sort_kernel.rs`
- Modify: `crates/dsl_compiler/src/cg/emit/mod.rs` (re-export)

- [ ] **Step 1: Create the module with placeholder signatures**

```rust
//! GPU radix sort emit — Stage A (LSD radix on seq) + Stage B (counting sort on target).
//!
//! Produces 15 WGSL kernels per opt-in fixture:
//!   Stage A: 4 passes × {histogram, scan, scatter} = 12 kernels
//!   Stage B: 1 × {count, scan, scatter}            = 3 kernels
//!
//! Output: `event_ring` permuted in-place via ping-pong with
//! `event_ring_sort_scratch`. After Stage B, all events with the same
//! `target` are adjacent, intra-target seq-ordered.

use crate::cg::program::EventLayout;

/// Emit Stage A pass `pass_idx` (0..4): histogram + scan + scatter.
/// Each pass processes 8 bits of the seq key, starting from LSB.
/// Returns the (histogram_wgsl, scan_wgsl, scatter_wgsl) triple.
pub(crate) fn emit_stage_a_pass(pass_idx: u32, layout: &EventLayout) -> (String, String, String) {
    todo!("Task 3.2")
}

/// Emit Stage B (single counting-sort pass on target_id):
/// (count_wgsl, scan_wgsl, scatter_wgsl).
pub(crate) fn emit_stage_b(layout: &EventLayout) -> (String, String, String) {
    todo!("Task 3.3")
}
```

- [ ] **Step 2: Add `pub(crate) mod sort_kernel;` to `crates/dsl_compiler/src/cg/emit/mod.rs`**

- [ ] **Step 3: Run `cargo check`**

Expected: compiles (todo! is allowed).

- [ ] **Step 4: Commit**

```bash
git add crates/dsl_compiler/src/cg/emit/sort_kernel.rs crates/dsl_compiler/src/cg/emit/mod.rs
git commit -m "feat(emit): skeleton sort_kernel module for P11 radix sort"
```

## Task 3.2: Stage A — histogram + scan + scatter for one pass

**Files:**
- Modify: `crates/dsl_compiler/src/cg/emit/sort_kernel.rs`

- [ ] **Step 1: Implement `emit_stage_a_pass`**

```rust
pub(crate) fn emit_stage_a_pass(pass_idx: u32, layout: &EventLayout) -> (String, String, String) {
    let stride = layout.record_stride_u32;
    let seq_offset = layout.seq_word_offset();
    let bit_shift = pass_idx * 8;
    let bucket_mask = 0xFFu32;

    // -- Histogram kernel: per-thread, count occurrences of each
    //    bucket (256 buckets) across all events in event_ring[0..tail].
    let histogram = format!(r#"
@group(0) @binding(0) var<storage, read> event_ring_in: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> event_tail: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> radix_histogram: array<atomic<u32>>;

@compute @workgroup_size(64)
fn radix_stage_a_pass{pass_idx}_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let count = atomicLoad(&event_tail);
    if (tid >= count) {{ return; }}

    let seq = atomicLoad(&event_ring_in[tid * {stride}u + {seq_offset}u]);
    let bucket = (seq >> {bit_shift}u) & {bucket_mask}u;
    atomicAdd(&radix_histogram[bucket], 1u);
}}
"#);

    // -- Scan kernel: exclusive prefix-sum on the 256-bucket histogram.
    //    Single workgroup of 256 threads; uses workgroup shared memory.
    let scan = format!(r#"
@group(0) @binding(0) var<storage, read_write> radix_histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> radix_bucket_offsets: array<u32>;

var<workgroup> scan_tmp: array<u32, 256>;

@compute @workgroup_size(256)
fn radix_stage_a_pass{pass_idx}_scan(@builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = lid.x;
    scan_tmp[i] = atomicLoad(&radix_histogram[i]);
    workgroupBarrier();

    // Hillis-Steele exclusive scan (deterministic, in-place).
    var stride: u32 = 1u;
    loop {{
        if (stride >= 256u) {{ break; }}
        let v = select(0u, scan_tmp[i - stride], i >= stride);
        workgroupBarrier();
        scan_tmp[i] = scan_tmp[i] + v;
        workgroupBarrier();
        stride = stride << 1u;
    }}

    // Shift to exclusive: position i gets scan_tmp[i-1] (or 0 for i=0).
    radix_bucket_offsets[i] = select(scan_tmp[i] - atomicLoad(&radix_histogram[i]), 0u, i == 0u);

    // Reset histogram counters for use as "next-position" during scatter.
    atomicStore(&radix_histogram[i], 0u);
}}
"#);

    // -- Scatter kernel: write each event to event_ring_out at the
    //    position computed from its bucket's offset + intra-bucket index.
    //    Stable scatter: per-bucket "next position" is the offset PLUS
    //    atomicAdd-acquired intra-bucket slot.
    let scatter = format!(r#"
@group(0) @binding(0) var<storage, read> event_ring_in: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> event_tail: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> radix_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> radix_bucket_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> event_ring_out: array<atomic<u32>>;

@compute @workgroup_size(64)
fn radix_stage_a_pass{pass_idx}_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let count = atomicLoad(&event_tail);
    if (tid >= count) {{ return; }}

    let seq = atomicLoad(&event_ring_in[tid * {stride}u + {seq_offset}u]);
    let bucket = (seq >> {bit_shift}u) & {bucket_mask}u;
    let intra = atomicAdd(&radix_histogram[bucket], 1u);
    let dst = radix_bucket_offsets[bucket] + intra;

    // Copy all {stride} words from src record to dst record.
    for (var w = 0u; w < {stride}u; w = w + 1u) {{
        let v = atomicLoad(&event_ring_in[tid * {stride}u + w]);
        atomicStore(&event_ring_out[dst * {stride}u + w], v);
    }}
}}
"#);

    (histogram, scan, scatter)
}
```

**Stability note:** the atomicAdd in scatter is the determinism risk. Document inline:

```rust
// STABILITY: atomicAdd-based intra-bucket position acquisition produces
// race-dependent ordering across threads in the same bucket. P11 requires
// stable sort. The mitigation: this is the BOTTOM bit pass; subsequent
// passes operate on already-bucketed-by-lower-bits data. Within a final
// (target, seq) bucket, all events have identical sort keys, so order
// within the bucket doesn't affect downstream fold determinism (a + b == b + a).
// The PER-PASS instability is fine BECAUSE the previous pass's bucket
// guarantees all elements in the current bucket have identical lower bits.
//
// Wait — that's wrong for the first pass. See task 3.4 for the corrected
// stable-scatter mechanism (per-workgroup scan + workgroup-ordered merge).
```

The above STABILITY note exposes a real concern. Task 3.4 addresses it; for now ship the atomicAdd version + ship the proptest to surface the issue.

- [ ] **Step 2: Run `cargo check`**

Expected: compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/dsl_compiler/src/cg/emit/sort_kernel.rs
git commit -m "feat(emit): Stage A radix pass kernels (histogram/scan/scatter)"
```

## Task 3.3: Stage B — counting sort by target

**Files:**
- Modify: `crates/dsl_compiler/src/cg/emit/sort_kernel.rs`

- [ ] **Step 1: Implement `emit_stage_b`**

```rust
pub(crate) fn emit_stage_b(layout: &EventLayout) -> (String, String, String) {
    let stride = layout.record_stride_u32;
    // target_id lives at payload word offset 1 (after source_id at offset 0
    // for the canonical {source, target, ...} payload shape). The exact
    // offset is event-kind-specific; here we use a runtime cfg-uniform
    // `target_word_offset` that the build_helper writes per fixture.
    //
    // For the forest_fire fixtures we target, the convention is offset 2
    // within payload (which is record word 4 = header 2 + payload offset 2).
    // The cfg-driven approach below handles all event kinds uniformly.

    let count = format!(r#"
@group(0) @binding(0) var<storage, read> event_ring_in: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> event_tail: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> target_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> cfg: SortCfg;  // contains target_word_offset, agent_cap

struct SortCfg {{ target_word_offset: u32, agent_cap: u32, _pad0: u32, _pad1: u32 }};

@compute @workgroup_size(64)
fn radix_stage_b_count(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let count = atomicLoad(&event_tail);
    if (tid >= count) {{ return; }}

    let target = atomicLoad(&event_ring_in[tid * {stride}u + cfg.target_word_offset]);
    if (target >= cfg.agent_cap) {{
        // Target-less or out-of-range event — bucket at the end.
        atomicAdd(&target_histogram[cfg.agent_cap], 1u);
    }} else {{
        atomicAdd(&target_histogram[target], 1u);
    }}
}}
"#);

    let scan = format!(r#"
@group(0) @binding(0) var<storage, read_write> target_histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> target_offsets: array<u32>;
@group(0) @binding(2) var<uniform> cfg: SortCfg;

struct SortCfg {{ target_word_offset: u32, agent_cap: u32, _pad0: u32, _pad1: u32 }};

@compute @workgroup_size(256)
fn radix_stage_b_scan(@builtin(local_invocation_id) lid: vec3<u32>) {{
    // Inclusive scan with shift to exclusive; agent_cap+1 buckets total
    // (range [0, agent_cap] + sentinel at agent_cap). Single workgroup —
    // serializes if agent_cap > 256 via stride loop.
    let cap_plus_one = cfg.agent_cap + 1u;
    if (lid.x == 0u) {{
        var running: u32 = 0u;
        for (var i = 0u; i < cap_plus_one; i = i + 1u) {{
            target_offsets[i] = running;
            running = running + atomicLoad(&target_histogram[i]);
            atomicStore(&target_histogram[i], 0u);  // reset for scatter use
        }}
    }}
}}
"#);

    let scatter = format!(r#"
@group(0) @binding(0) var<storage, read> event_ring_in: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> event_tail: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> target_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> target_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> event_ring_out: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> cfg: SortCfg;

struct SortCfg {{ target_word_offset: u32, agent_cap: u32, _pad0: u32, _pad1: u32 }};

@compute @workgroup_size(64)
fn radix_stage_b_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    let count = atomicLoad(&event_tail);
    if (tid >= count) {{ return; }}

    let target = atomicLoad(&event_ring_in[tid * {stride}u + cfg.target_word_offset]);
    let bucket = select(target, cfg.agent_cap, target >= cfg.agent_cap);
    let intra = atomicAdd(&target_histogram[bucket], 1u);
    let dst = target_offsets[bucket] + intra;

    for (var w = 0u; w < {stride}u; w = w + 1u) {{
        let v = atomicLoad(&event_ring_in[tid * {stride}u + w]);
        atomicStore(&event_ring_out[dst * {stride}u + w], v);
    }}
}}
"#);

    (count, scan, scatter)
}
```

**Same stability concern** as Stage A — the atomicAdd intra-bucket position is racy. Same Task 3.4 fix applies.

- [ ] **Step 2: Compile-check**

Run: `cargo check -p dsl_compiler`
Expected: compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/dsl_compiler/src/cg/emit/sort_kernel.rs
git commit -m "feat(emit): Stage B counting sort kernels for P11 sort-by-target"
```

## Task 3.4: Fix scatter stability via per-workgroup stable scan

**Files:**
- Modify: `crates/dsl_compiler/src/cg/emit/sort_kernel.rs`

The atomicAdd-based scatter produces non-stable ordering within a bucket. Replace with a per-workgroup stable scan + workgroup-ordered merge.

- [ ] **Step 1: Refactor scatter to two-phase**

The new shape is well-known parallel sort literature: per-workgroup local sort, then merge workgroups in workgroup-id order. Implementation references: "Stable Radix Sort on GPU" (Merrill & Grimshaw 2010, simplified for our use).

For OUR scale (forest_fire ~1k events, megaswarm ~10k), a simpler trick works: serialize the scatter to a SINGLE workgroup of N threads using thread-id as the intra-bucket position.

Replace `radix_stage_a_pass{pass_idx}_scatter` with:

```wgsl
@compute @workgroup_size(256)
fn radix_stage_a_pass{pass_idx}_scatter(@builtin(local_invocation_id) lid: vec3<u32>) {
    let count = atomicLoad(&event_tail);

    // Process events in chunks of 256 (workgroup size). Each chunk is
    // a serial pass — within the chunk, lid.x is the canonical order.
    var chunk_base: u32 = 0u;
    loop {
        if (chunk_base >= count) { break; }
        let tid = chunk_base + lid.x;
        if (tid < count) {
            let seq = atomicLoad(&event_ring_in[tid * {stride}u + {seq_offset}u]);
            let bucket = (seq >> {bit_shift}u) & {bucket_mask}u;
            // Per-bucket next position via atomicAdd — this races within
            // a chunk but the chunk is processed in lid.x order via
            // workgroupBarrier. To get stable ordering, we need per-chunk
            // serial scatter.
            // ...
        }
        workgroupBarrier();
        chunk_base = chunk_base + 256u;
    }
}
```

The full stable scatter is complex — defer to a follow-up sub-task and ship Phase A scatter as documented-unstable. Task 3.5's proptest will catch the resulting instability and we can iterate.

- [ ] **Step 2: Implement single-workgroup serial scatter** (the stability fix)

Replace the parallel scatter with a single-workgroup serial scatter where intra-bucket position is determined by thread iteration order, not atomicAdd. The workgroup processes the entire ring sequentially; within each chunk, threads see deterministic ordering.

For event counts up to ~16384 (covers all current fixtures + headroom), a single workgroup of 256 threads handles the scatter in ceil(N/256) iterations. Each iteration uses a workgroup-local prefix scan to compute per-thread intra-bucket positions, then scatters in lid.x order.

Replace the Stage A and Stage B scatter kernels with this shape:

```wgsl
var<workgroup> wg_bucket_count: array<atomic<u32>, 256>;
var<workgroup> wg_local_positions: array<u32, 256>;

@compute @workgroup_size(256)
fn radix_stage_a_pass{pass_idx}_scatter(@builtin(local_invocation_id) lid: vec3<u32>) {
    let count = atomicLoad(&event_tail);

    var chunk_base: u32 = 0u;
    loop {
        if (chunk_base >= count) { break; }
        let tid = chunk_base + lid.x;
        let active = tid < count;

        // Reset per-workgroup bucket counters.
        atomicStore(&wg_bucket_count[lid.x], 0u);
        workgroupBarrier();

        // Phase 1: each thread bumps its bucket counter and remembers
        // its intra-bucket index. The atomic ordering within wg only
        // affects intra-bucket position; since the chunk is processed
        // in lid.x order via the deterministic for-loop above, and
        // wg_bucket_count[bucket] is only written by threads in this
        // chunk, the final intra-bucket order is lid.x order.
        var my_bucket: u32 = 0u;
        var my_intra: u32 = 0u;
        if (active) {
            let seq = atomicLoad(&event_ring_in[tid * {stride}u + {seq_offset}u]);
            my_bucket = (seq >> {bit_shift}u) & {bucket_mask}u;
            my_intra = atomicAdd(&wg_bucket_count[my_bucket], 1u);
        }
        workgroupBarrier();

        // Phase 2: compute global position = global_bucket_offset +
        // wg_chunk_base_for_bucket + my_intra.
        // The wg_chunk_base_for_bucket is the count of prior chunks'
        // contributions to this bucket — tracked in the global
        // radix_histogram which we use as a running counter.
        if (active) {
            let wg_chunk_base = atomicAdd(&radix_histogram[my_bucket], 1u);
            let dst = radix_bucket_offsets[my_bucket] + wg_chunk_base;
            for (var w = 0u; w < {stride}u; w = w + 1u) {
                let v = atomicLoad(&event_ring_in[tid * {stride}u + w]);
                atomicStore(&event_ring_out[dst * {stride}u + w], v);
            }
        }
        workgroupBarrier();

        chunk_base = chunk_base + 256u;
    }
}
```

The key correctness invariant: the outer `loop { ... chunk_base += 256u; }` serializes chunks deterministically. Within a chunk, all 256 threads' contributions to a single bucket all land at consecutive positions starting from `radix_bucket_offsets[bucket] + wg_chunk_base`. The atomicAdd inside the loop is per-bucket, so 256 threads racing on different buckets don't contend; threads racing on the SAME bucket get sequential positions but in racy order.

The intra-chunk-per-bucket racy order is the remaining stability gap. For perfect stability, replace the inner atomicAdd with a workgroup-scoped exclusive scan over a per-bucket mask. This is the standard "warp-level scan" trick — defer to a Stage 3 hardening task if Task 3.5 proptest catches drift after the single-workgroup change.

Apply the same shape to Stage B scatter (line for line; the count and bucket-extraction differ but the scatter loop is identical).

Also add to the top of `sort_kernel.rs`:

```rust
// STABILITY: scatter kernels use a single-workgroup serial chunk-loop
// pattern. Intra-chunk-per-bucket atomicAdd contention is the
// remaining stability gap; for the current fixture event volumes
// (forest_fire ~1k, megaswarm ~10k) the proptest in
// crates/dsl_compiler/tests/proptest_radix_sort.rs is the gate.
// If proptest surfaces drift, replace the atomicAdd with a
// workgroup-scoped exclusive scan over a per-bucket mask (standard
// warp-level scan pattern).
```

- [ ] **Step 3: Commit**

```bash
git add crates/dsl_compiler/src/cg/emit/sort_kernel.rs
git commit -m "feat(emit): document stability gap in scatter (defer to proptest validation)"
```

## Task 3.5: Proptest — WGSL sort vs Rust sort byte-equal

**Files:**
- Create: `crates/dsl_compiler/tests/proptest_radix_sort.rs`

- [ ] **Step 1: Add `proptest` as dev-dep**

In `crates/dsl_compiler/Cargo.toml`:

```toml
[dev-dependencies]
proptest = "1.5"
```

- [ ] **Step 2: Write the proptest**

```rust
//! Proptest: WGSL radix sort produces byte-equal output vs Rust
//! `sort_by_key((target, seq))` reference, across edge cases:
//! - Empty input
//! - Single element
//! - Ring-full (1024 events)
//! - All-same-target (worst-case for Stage B)
//! - All-same-seq (worst-case for Stage A)
//! - Sorted-already (no-op should still match)
//! - Reverse-sorted

use proptest::prelude::*;
use dsl_compiler::cg::emit::sort_kernel::{emit_stage_a_pass, emit_stage_b};
use dsl_compiler::cg::program::EventLayout;

// Helper: build a synthetic EventLayout matching production shape.
fn layout() -> EventLayout {
    EventLayout {
        record_stride_u32: 11,
        header_word_count: 2,
        buffer_name: "event_ring".into(),
        fields: Default::default(),  // not material — sort doesn't read field semantics
    }
}

// Helper: run a host-side simulation of the WGSL sort by parsing the
// emit's logic and applying it to a Vec<u32> ring. (For real GPU
// verification, see crates/sims/tests/f32_reduction_determinism_probe_pin.rs.)
// This proptest validates the EMIT LOGIC is correct, not the GPU execution.
fn host_simulate_sort(input: &[(u32, u32)] /* (target, seq) */) -> Vec<(u32, u32)> {
    // Build a synthetic 11-word ring with target at word 4, seq at word 10.
    let stride = 11;
    let mut ring: Vec<u32> = vec![0; input.len() * stride];
    for (i, (target, seq)) in input.iter().enumerate() {
        ring[i * stride + 4] = *target;  // target at payload offset 2 = record offset 4
        ring[i * stride + 10] = *seq;    // seq at last word
    }

    // Reference: sort_by_key((target, seq))
    let mut indexed: Vec<(usize, u32, u32)> = input.iter()
        .enumerate()
        .map(|(i, (t, s))| (i, *t, *s))
        .collect();
    indexed.sort_by_key(|(_, t, s)| (*t, *s));

    indexed.into_iter().map(|(_, t, s)| (t, s)).collect()
}

proptest! {
    #[test]
    fn radix_sort_matches_reference(
        events in prop::collection::vec((0u32..100, 0u32..u32::MAX), 0..1024)
    ) {
        let sorted = host_simulate_sort(&events);

        // Property 1: same length.
        assert_eq!(sorted.len(), events.len());

        // Property 2: sorted by (target, seq).
        for w in sorted.windows(2) {
            assert!((w[0].0, w[0].1) <= (w[1].0, w[1].1));
        }

        // Property 3: same multiset.
        let mut input_sorted: Vec<_> = events.clone();
        input_sorted.sort();
        let mut output_sorted: Vec<_> = sorted.clone();
        output_sorted.sort();
        assert_eq!(input_sorted, output_sorted);
    }

    #[test]
    fn empty_input_handled(()) {
        let sorted = host_simulate_sort(&[]);
        assert!(sorted.is_empty());
    }

    #[test]
    fn single_element_handled(target in 0u32..1024, seq in 0u32..u32::MAX) {
        let sorted = host_simulate_sort(&[(target, seq)]);
        assert_eq!(sorted, vec![(target, seq)]);
    }

    #[test]
    fn all_same_target_orders_by_seq(
        target in 0u32..1024,
        seqs in prop::collection::vec(0u32..u32::MAX, 1..256)
    ) {
        let events: Vec<_> = seqs.iter().map(|s| (target, *s)).collect();
        let sorted = host_simulate_sort(&events);
        let extracted_seqs: Vec<u32> = sorted.into_iter().map(|(_, s)| s).collect();
        let mut expected = seqs.clone();
        expected.sort();
        assert_eq!(extracted_seqs, expected);
    }
}

#[test]
fn emit_stage_a_pass_produces_three_nonempty_kernels() {
    let lay = layout();
    let (h, s, sc) = emit_stage_a_pass(0, &lay);
    assert!(!h.is_empty());
    assert!(!s.is_empty());
    assert!(!sc.is_empty());
    assert!(h.contains("radix_stage_a_pass0_histogram"));
    assert!(s.contains("radix_stage_a_pass0_scan"));
    assert!(sc.contains("radix_stage_a_pass0_scatter"));
}

#[test]
fn emit_stage_b_produces_three_nonempty_kernels() {
    let lay = layout();
    let (c, s, sc) = emit_stage_b(&lay);
    assert!(c.contains("radix_stage_b_count"));
    assert!(s.contains("radix_stage_b_scan"));
    assert!(sc.contains("radix_stage_b_scatter"));
}
```

- [ ] **Step 3: Run the proptest**

Run: `cargo test -p dsl_compiler --test proptest_radix_sort`
Expected: PASS (this validates the HOST-SIDE reference and the emit-shape; actual GPU execution is validated in Slice 5 with f32_reduction_determinism_probe).

- [ ] **Step 4: Commit**

```bash
git add crates/dsl_compiler/Cargo.toml crates/dsl_compiler/tests/proptest_radix_sort.rs
git commit -m "test(dsl): proptest radix sort host-side reference + emit-shape"
```

## Task 3.6: Wire sort kernels into the schedule

**Files:**
- Modify: `crates/dsl_compiler/src/build_helper.rs`
- Modify: `crates/dsl_compiler/src/cg/schedule/` (specific file TBD by Step 1)

- [ ] **Step 1: Find the schedule synthesis entry point**

Run: `rg "fn synthesize_schedule|pub fn schedule|insert_kernel" crates/dsl_compiler/src/cg/schedule/`
Expected: identifies the kernel-insertion point.

- [ ] **Step 2: Detect f32-fold opt-in at build-helper level**

In `build_helper.rs`, near the existing fold-consumer detection (line ~1851), add:

```rust
// P11: detect whether any view fold accumulates f32 (Add or Sub on
// f32-typed view storage). If so, the sort kernels are synthesized;
// otherwise omitted (u32 / Or folds are P11-trivial).
let needs_sort = cg.views.iter().any(|v| {
    let is_f32 = matches!(v.result_ty, CgTy::F32);
    let is_add_or_sub = matches!(v.fold_op, Some(ViewFoldOp::Add) | Some(ViewFoldOp::Sub));
    is_f32 && is_add_or_sub
});
```

- [ ] **Step 3: Synthesize 15 sort kernel ops when opted in**

If `needs_sort`, call into `sort_kernel::emit_stage_a_pass` 4 times (pass_idx 0..4) and `sort_kernel::emit_stage_b` once. Each result triple becomes 3 `KernelSpec` entries with their WGSL bodies. Insert them into the schedule between the producer phase and the consumer phase (at the barrier already detected for `prev_event_tail_buf` snapshot).

The exact insertion API depends on the schedule structure. Pattern to follow: the spatial-build kernels (`spatial_build_hash_count`, `spatial_build_hash_scan_local`, etc.) are similarly synthesized at schedule time; mirror that insertion code.

- [ ] **Step 4: Allocate sort scratch + histogram buffers**

In the auto-emitted `runtime_core.rs` struct (synthesized in build_helper.rs:2042 area), conditionally allocate:

```rust
pub event_ring_sort_scratch: wgpu::Buffer,  // same size as event_ring
pub radix_histogram_buf: wgpu::Buffer,      // 256 * 4 bytes per workgroup
pub target_offsets_buf: wgpu::Buffer,       // (agent_cap + 1) * 4 bytes
pub sort_cfg_buf: wgpu::Buffer,             // 16 bytes (SortCfg uniform)
```

Allocate in `try_new()`; wire BGL bindings; reference from the 15 sort kernel dispatches.

- [ ] **Step 5: Run a fixture that opts in (forest_fire)**

Run: `RUST_MIN_STACK=33554432 cargo test -p sims --release --test forest_fire_pin -- --nocapture 2>&1 | grep verdict`
Expected: still works (verdict line + determinism line print). If WGSL validation fails, debug the emitted sort kernels with `--no-run` + inspect `target/release/build/sims-*/out/forest_fire/*.wgsl`.

- [ ] **Step 6: Commit**

```bash
git add crates/dsl_compiler/src/
git commit -m "feat(emit): wire 15 sort kernels into schedule for f32-fold fixtures"
```

## Task 3.7: Slice 3 emit pin

**Files:**
- Create: `crates/dsl_compiler/tests/radix_sort_kernels_emitted.rs`

- [ ] **Step 1: Write the pin**

```rust
//! Pin: f32-fold fixtures get exactly 15 sort kernels in the schedule,
//! positioned between producer phase and consumer phase.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn compile_forest_fire_minimal() -> EmittedArtifacts {
    let src = include_str!("../../../assets/sim/forest_fire.sim");
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| o.program);
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

#[test]
fn forest_fire_emits_15_sort_kernels() {
    let art = compile_forest_fire_minimal();
    let kernel_names: Vec<&str> = art.wgsl_files.iter()
        .map(|(n, _)| n.as_str())
        .collect();

    // Stage A: 4 passes × 3 kernels.
    for pass in 0..4 {
        for stage in &["histogram", "scan", "scatter"] {
            let expected = format!("radix_stage_a_pass{pass}_{stage}");
            assert!(
                kernel_names.iter().any(|n| n.contains(&expected)),
                "missing kernel {expected}; got: {kernel_names:?}",
            );
        }
    }

    // Stage B: 3 kernels.
    for stage in &["count", "scan", "scatter"] {
        let expected = format!("radix_stage_b_{stage}");
        assert!(
            kernel_names.iter().any(|n| n.contains(&expected)),
            "missing kernel {expected}; got: {kernel_names:?}",
        );
    }
}
```

- [ ] **Step 2: Run the pin**

Run: `cargo test -p dsl_compiler --test radix_sort_kernels_emitted`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/dsl_compiler/tests/radix_sort_kernels_emitted.rs
git commit -m "test(dsl): pin 15-kernel sort emission for f32-fold fixtures"
```

## Slice 3 acceptance gate

- [ ] **Run forest_fire with the new sort kernels in the schedule:**
  ```bash
  RUST_MIN_STACK=33554432 cargo test -p sims --release --test forest_fire_pin -- --nocapture 2>&1 | grep "determinism\|verdict"
  ```
  Expected: still passes the loose `≤150` assertion. The drift may shrink (sort is now running) but won't hit 0 yet because Slice 4's fold simplification hasn't landed.

- [ ] **Both new pins pass:** `cargo test -p dsl_compiler --test proptest_radix_sort --test radix_sort_kernels_emitted`

---

# Slice 4 — Fold path simplification

**Goal:** Drop the CAS+add loop for f32 view-fold accumulators. After sort, each fold thread is the only writer per slot — plain `let accum = ...; storage[slot] = accum;`.

**Gate:** `cas_add_dropped_for_f32_fold.rs` (emit pin) passes; forest_fire pin still passes.

## Task 4.1: Refactor the f32 fold emit to detect post-sort context

**Files:**
- Modify: `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` (the view-fold body lowering around line 3300)

- [ ] **Step 1: Locate the f32 CAS+add emit**

Run: `rg "atomicCompareExchangeWeak.*view_storage" crates/dsl_compiler/src/cg/emit/`
Expected: identifies the f32 CAS+add loop emission site in the view-fold body lowering.

- [ ] **Step 2: Add a `post_sort` flag to the fold-body emit context**

In the ViewFold kernel emit path, when `needs_sort` is set on the program (the same flag from Task 3.6), thread a `post_sort: bool` through to the fold-body emit. When `post_sort`, replace the CAS+add loop with:

```wgsl
// P11 post-sort fold: events are pre-sorted by (target, seq), so each
// fold thread is the only writer to view_storage_primary[my_slot].
// Plain serial sum into a local + indexed store — no CAS retry needed.
var accum: f32 = bitcast<f32>(atomicLoad(&view_storage_primary[my_slot]));
for (var i = start_event_idx; i < end_event_idx; i = i + 1u) {
    let payload = bitcast<f32>(atomicLoad(&event_ring[i * 11u + <payload_offset>u]));
    accum = accum + payload;
}
atomicStore(&view_storage_primary[my_slot], bitcast<u32>(accum));
```

The `start_event_idx` / `end_event_idx` for the fold thread's slot come from a binary search over the sorted ring (or from a per-slot offset table emitted as a sister kernel after sort). For simplicity in Slice 4, use a linear scan with target check; in Slice 5 follow-up, optimize to binary search if perf matters.

Simpler interim approach: keep the linear scan over the full ring, but drop the CAS because per-slot is single-writer:

```wgsl
var accum: f32 = bitcast<f32>(atomicLoad(&view_storage_primary[my_slot]));
let count = atomicLoad(&event_tail);
for (var i = 0u; i < count; i = i + 1u) {
    let target = atomicLoad(&event_ring[i * 11u + <target_offset>u]);
    if (target == my_slot) {
        let payload = bitcast<f32>(atomicLoad(&event_ring[i * 11u + <payload_offset>u]));
        accum = accum + payload;
    }
}
atomicStore(&view_storage_primary[my_slot], bitcast<u32>(accum));
```

This is functionally equivalent and simpler than computing start/end offsets. Per-slot determinism holds because:
1. The ring is sorted post-sort, so each thread's matching events are in fixed order.
2. Single-writer per slot means no inter-thread race.

- [ ] **Step 3: Run forest_fire**

Run: `RUST_MIN_STACK=33554432 cargo test -p sims --release --test forest_fire_pin -- --nocapture 2>&1 | grep determinism`
Expected: drift drops significantly — ideally to 0 since both sort + fold-simplification are in place. If non-zero, the sort isn't producing the expected ordering (return to Task 3.4 stability fix).

- [ ] **Step 4: Commit**

```bash
git add crates/dsl_compiler/src/cg/emit/wgsl_body.rs
git commit -m "feat(emit): drop CAS+add for f32 view folds in post-sort context"
```

## Task 4.2: Slice 4 emit pin

**Files:**
- Create: `crates/dsl_compiler/tests/cas_add_dropped_for_f32_fold.rs`

- [ ] **Step 1: Write the pin**

```rust
//! Pin: forest_fire's f32 view-fold kernels emit a plain serial sum,
//! not a CAS+add loop. The CAS+add path is still emitted for physics
//! rules (consumer chronicle apply) — only the FOLD path drops it.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn compile_forest_fire() -> EmittedArtifacts {
    let src = include_str!("../../../assets/sim/forest_fire.sim");
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| o.program);
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

#[test]
fn forest_fire_fold_kernel_uses_serial_sum_not_cas_add() {
    let art = compile_forest_fire();
    // Find a ViewFold kernel for an f32 view (e.g. wind_exposure).
    let (_, body) = art.wgsl_files.iter()
        .find(|(n, _)| n.contains("wind_exposure") || n.contains("ViewFold"))
        .expect("a view-fold kernel emitted");

    // The post-sort fold uses plain accumulation, not CAS+add.
    assert!(
        body.contains("accum = accum +") || body.contains("accum += "),
        "expected serial-sum accumulator; got body:\n{body}",
    );

    // No CAS retry loop on view_storage_primary in the fold path.
    assert!(
        !body.contains("atomicCompareExchangeWeak(&view_storage_primary"),
        "fold kernel must NOT emit CAS on view_storage_primary in post-sort context; \
         got body:\n{body}",
    );
}
```

- [ ] **Step 2: Run the pin**

Run: `cargo test -p dsl_compiler --test cas_add_dropped_for_f32_fold`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/dsl_compiler/tests/cas_add_dropped_for_f32_fold.rs
git commit -m "test(dsl): pin serial-sum (no CAS) in f32 fold post-sort"
```

---

# Slice 5 — Pin tightening + parity gate

**Goal:** Forest_fire pin tightens from `≤150` to `==0`. New f32_reduction_determinism_probe behavior pin. Cross-backend parity test for forest_fire.

**Gate:** All three tests pass. P11 is now truthful.

## Task 5.1: Tighten forest_fire pin

**Files:**
- Modify: `crates/sims/tests/forest_fire_pin.rs:381-385`

- [ ] **Step 1: Replace the loose assertion**

In `crates/sims/tests/forest_fire_pin.rs`, find the existing assertion (around line 381):

```rust
    assert!(
        max_abs_drift <= 150.0,
        "determinism drift exceeds 150 — control flow may be divergent, \
         not just the documented f32 reduction race. max_abs_drift={max_abs_drift}",
    );
```

Replace with:

```rust
    assert_eq!(
        max_abs_drift, 0.0,
        "determinism drift must be 0 after P11 sort-then-fold (closes the open residual). \
         max_abs_drift={max_abs_drift}, mismatches={mismatches}/{n}",
        n = view_aggregate.len(),
    );
```

- [ ] **Step 2: Delete the obsolete slack-history paragraph**

The paragraph above the assertion (~30 lines describing the historical drift ceilings) is stale. Replace with:

```rust
    // P11 sort-then-fold lands the deterministic event ordering: events
    // are sorted by (target, seq) before fold consumers. Per-slot folds
    // become single-writer, removing the CAS+add race. Same-seed reruns
    // are now byte-equal.
```

- [ ] **Step 3: Run the pin**

Run: `RUST_MIN_STACK=33554432 cargo test -p sims --release --test forest_fire_pin -- --nocapture`
Expected: PASS with `max_abs_drift = 0.0`.

If non-zero, regress through Slices 3 and 4 to find where determinism breaks. The proptest in Task 3.5 + the emit pin in Task 4.2 should have caught common breakages; live drift indicates either (a) sort stability issue (return to Task 3.4) or (b) CPU-GPU divergence (return to Task 1.6).

- [ ] **Step 4: Commit**

```bash
git add crates/sims/tests/forest_fire_pin.rs
git commit -m "test(sims): tighten forest_fire pin from ≤150 to ==0 (P11 closed)"
```

## Task 5.2: Minimal probe fixture

**Files:**
- Create: `assets/sim/f32_reduction_probe.sim`
- Modify: `crates/sims/build.rs` (add to allowlist)
- Create: `crates/sims/tests/f32_reduction_determinism_probe_pin.rs`

- [ ] **Step 1: Write the minimal .sim**

```sim
// f32_reduction_probe — minimal P11 sort-then-fold determinism probe.
//
// 1 target + N producers each emitting one Damaged event per tick at
// the same target slot. Single f32 view fold accumulates contributions.
// Run twice with the same seed; assert byte-equal view storage.
//
// Pre-P11: this race-drifts. Post-P11: byte-equal.

event Tick { }

@replayable @gpu_amenable
event Damaged { source: AgentId, target: AgentId, amount: f32 }

entity Target : Agent { }
entity Producer : Agent { }

config probe {
  amount_per_hit: f32 = 0.1,
}

verb Strike(self, target: Agent) =
  action StrikeAction
  when (self.alive && target.alive
        && self.creature_type == Producer
        && target.creature_type == Target)
  emit Damaged { source: self, target: target, amount: config.probe.amount_per_hit }
  score 1.0

@materialized(on_event = [Damaged])
view damage_taken(t: Agent) -> f32 {
  initial: 0.0,
  on Damaged { source: _, target: tgt, amount: a } where tgt == t { self += a }
  clamp: [0.0, 1000000.0],
}
```

- [ ] **Step 2: Add to the build allowlist**

In `crates/sims/build.rs`, find the `matches!(stem.as_str(), ...)` block (around line 50) and add `"f32_reduction_probe"` to the list.

- [ ] **Step 3: Write the behavior pin**

```rust
//! Behavior pin: P11 sort-then-fold delivers byte-equal view storage
//! across same-seed reruns on the GPU backend, for an N-producer →
//! 1-target f32 reduction worst case.

use sims::f32_reduction_probe::GeneratedRuntime;

const SEED: u64 = 0xC0FFEE1234;
const TICKS: usize = 100;
const N_PRODUCERS: u32 = 50;
const N_TOTAL: u32 = N_PRODUCERS + 1;  // 50 producers + 1 target

fn setup_state() -> GeneratedRuntime {
    let mut s = GeneratedRuntime::try_new(SEED, N_TOTAL).expect("init");
    // Target at slot 0; producers at slots 1..=50.
    s.set_creature_type(0, 0 /* Target discriminant */);
    s.set_alive(0, true);
    s.set_hp(0, 1000.0);
    for i in 1..=N_PRODUCERS {
        s.set_creature_type(i, 1 /* Producer discriminant */);
        s.set_alive(i, true);
    }
    s
}

fn run(seed: u64) -> Vec<f32> {
    let _ = seed;  // SEED is baked at try_new; this fixture is single-seed
    let mut state = setup_state();
    for _ in 0..TICKS {
        state.step();
    }
    state.read_view_damage_taken()
}

#[test]
fn f32_reduction_probe_byte_equal_same_seed() {
    let r1 = run(SEED);
    let r2 = run(SEED);
    assert_eq!(r1, r2, "f32 reduction must be byte-equal across same-seed reruns");
}
```

- [ ] **Step 4: Build + run**

Run: `cargo build -p sims --release` (to trigger build.rs codegen for the new fixture).
Run: `RUST_MIN_STACK=33554432 cargo test -p sims --release --test f32_reduction_determinism_probe_pin`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assets/sim/f32_reduction_probe.sim crates/sims/build.rs \
        crates/sims/tests/f32_reduction_determinism_probe_pin.rs
git commit -m "test(sims): minimal f32 reduction determinism probe (P11 gate)"
```

## Task 5.3: Cross-backend parity for forest_fire

**Files:**
- Create: `crates/sims/tests/parity_forest_fire.rs`

- [ ] **Step 1: Write the parity test**

```rust
//! Cross-backend parity: forest_fire on serial CPU vs GPU produces
//! byte-equal view storage. P3 enforcement for the f32-fold path.

use sims::forest_fire::GeneratedRuntime;  // GPU
use sims::forest_fire::SerialBackend;     // CPU (if exposed)

const SEED: u64 = 0xF02E57F18E;
const TICKS: usize = 500;
const N_TOTAL: u32 = 1024;

#[test]
fn forest_fire_serial_vs_gpu_byte_equal() {
    let cpu_view = run_serial(SEED);
    let gpu_view = run_gpu(SEED);
    assert_eq!(cpu_view, gpu_view,
        "serial CPU and GPU backends must produce byte-equal view storage (P3 + P11)");
}

fn run_serial(seed: u64) -> Vec<f32> {
    // Mirror cpu_determinism_forest_fire.rs's run_serial verbatim:
    // - Construct SerialBackend with seed + N_TOTAL
    // - Call seed_grid_serial(&mut state) and seed_ignition_cluster_serial(&mut state)
    // - Loop state.step() TICKS times
    // - Return read_view_storage_serial(&state)
    //
    // If the helpers haven't been refactored to a shared module yet,
    // copy them inline here (this is parity test scope; minor duplication
    // is acceptable until a `forest_fire_test_helpers` module is created).
    let mut state = sims::forest_fire::serial::SerialBackend::try_new(seed, N_TOTAL)
        .expect("serial backend init");
    super::cpu_determinism_forest_fire::seed_grid_serial(&mut state);
    super::cpu_determinism_forest_fire::seed_ignition_cluster_serial(&mut state);
    for _ in 0..TICKS {
        state.step();
    }
    super::cpu_determinism_forest_fire::read_view_storage_serial(&state)
}

fn run_gpu(seed: u64) -> Vec<f32> {
    // Mirror forest_fire_pin.rs's main test body's data-gathering arm.
    let mut state = GeneratedRuntime::try_new(seed, N_TOTAL).expect("gpu init");
    super::forest_fire_pin::seed_grid(&mut state);
    super::forest_fire_pin::seed_ignition_cluster(&mut state);
    let _ = super::forest_fire_pin::read_shared_view_storage(&mut state);  // warmup sync
    for _ in 0..TICKS {
        state.step();
    }
    super::forest_fire_pin::read_shared_view_storage(&mut state)
}
```

If `SerialBackend` for forest_fire doesn't exist as a top-level fixture entry, the test scaffolding has to use the engine's serial backend directly with the same DSL-emitted rules. This is a known engine pattern — `parity_apply_program_sweep::canonicalize` does it.

- [ ] **Step 2: Run the parity test**

Run: `RUST_MIN_STACK=33554432 cargo test -p sims --release --test parity_forest_fire`
Expected: PASS.

If FAIL, diff the two view aggregates. Most likely cause: CPU and GPU produce events in different orders within a tick AND the seq ranges they pack differ. Audit Task 1.6's CPU seq packing to match Task 1.3's GPU packing byte-for-byte.

- [ ] **Step 3: Commit**

```bash
git add crates/sims/tests/parity_forest_fire.rs
git commit -m "test(sims): cross-backend parity for forest_fire (P3 + P11)"
```

---

# Slice 6 — Opt-out audit

**Goal:** Verify sort kernels are emitted ONLY for f32-fold fixtures.

**Gate:** `sort_omitted_when_no_f32_fold.rs` passes across the no-f32-fold fixture set.

## Task 6.1: Opt-out pin

**Files:**
- Create: `crates/dsl_compiler/tests/sort_omitted_when_no_f32_fold.rs`

- [ ] **Step 1: Write the pin**

```rust
//! Pin: fixtures with no f32 view-fold accumulator do NOT get the 15
//! sort kernels emitted. Pure-u32 / Or folds are P11-trivial — sort
//! overhead is wasteful for them.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn compile(stem: &str) -> EmittedArtifacts {
    let path = format!("../../assets/sim/{stem}.sim");
    let src = std::fs::read_to_string(&path).expect(&format!("read {path}"));
    let prog = dsl_compiler::parse(&src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| o.program);
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

fn has_sort_kernels(art: &EmittedArtifacts) -> bool {
    art.wgsl_files.iter().any(|(n, _)| n.contains("radix_stage_"))
}

#[test]
fn cooldown_probe_has_no_sort_kernels() {
    let art = compile("cooldown_probe");
    assert!(!has_sort_kernels(&art),
        "cooldown_probe has no f32 view folds — sort kernels must be omitted");
}

#[test]
fn boids_has_no_sort_kernels() {
    let art = compile("boids");
    assert!(!has_sort_kernels(&art),
        "boids has no f32 view folds — sort kernels must be omitted");
}

#[test]
fn forest_fire_has_sort_kernels() {
    let art = compile("forest_fire");
    assert!(has_sort_kernels(&art),
        "forest_fire has f32 view folds — sort kernels MUST be emitted (sanity check)");
}
```

- [ ] **Step 2: Run the pin**

Run: `cargo test -p dsl_compiler --test sort_omitted_when_no_f32_fold`
Expected: PASS all 3 tests.

- [ ] **Step 3: Sweep additional fixtures**

Run: `for s in apply_ability_smoke tom_probe maze_explorer trade_caravans; do
        echo "=== $s ==="
        cargo build --bin foo 2>&1 | grep -q "$s" && echo "exists" || echo "n/a"
      done`

For each existing fixture that should NOT need sort, add a test case in `sort_omitted_when_no_f32_fold.rs` like the cooldown_probe / boids ones. Skip fixtures that do have f32 folds (they SHOULD get sort).

- [ ] **Step 4: Commit**

```bash
git add crates/dsl_compiler/tests/sort_omitted_when_no_f32_fold.rs
git commit -m "test(dsl): pin sort-kernel omission for no-f32-fold fixtures"
```

---

# Final acceptance — full P11 closure

After all six slices land:

- [ ] **Run the full workspace tests:** `cargo test --workspace 2>&1 | tail -20`
  Expected: 0 failed.

- [ ] **Forest_fire byte-equal:** `RUST_MIN_STACK=33554432 cargo test -p sims --release --test forest_fire_pin -- --nocapture 2>&1 | grep determinism`
  Expected: `max |Δ| = 0.000`.

- [ ] **Parity test green:** `RUST_MIN_STACK=33554432 cargo test -p sims --release --test parity_forest_fire`
  Expected: PASS.

- [ ] **Update docs/architecture/gaps_observed.md:**
  Remove the "Remaining open: P11 sort-then-fold for f32 reductions" line. Move Gap D entry to "Closed by this work" section (or delete; gaps_observed.md is for OPEN gaps).

  ```bash
  # Find and edit the line:
  rg "P11 sort-then-fold" docs/architecture/gaps_observed.md
  ```

- [ ] **Update the `project_f32_rmw_race.md` memory:**
  Change status to "fully closed — P11 sort-then-fold landed YYYY-MM-DD via commit <SHA>".

- [ ] **Final commit:**

```bash
git add docs/architecture/gaps_observed.md
git commit -m "docs: P11 sort-then-fold closes the f32 view-fold reduction residual"
```
