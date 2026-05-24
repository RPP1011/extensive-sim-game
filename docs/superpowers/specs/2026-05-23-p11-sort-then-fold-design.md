# P11 sort-then-fold — design

> **Status:** Design spec (2026-05-23). Constitutional fix for f32 view-fold reduction non-determinism, the canonical P11 case. Two coordinated changes: inject a deterministic `seq` field on every event payload, and emit a per-tick GPU radix sort of the chronicle ring by `(target_id, seq)` before fold consumers. The CPU oracle path adopts a matching `Vec::sort_by_key`; cross-backend parity holds.
>
> Replaces the `max_abs_drift ≤ 150.0` loose pin in `crates/sims/tests/forest_fire_pin.rs` with a `== 0` byte-equal assertion, and closes the "Remaining open: P11 sort-then-fold for f32 reductions" line in `docs/architecture/gaps_observed.md`.

---

## §1 Problem statement

**Symptom.** Forest_fire pin re-run with the same seed produces view aggregate buffers that drift by max |Δ| = 47.0 across 479/1024 slots. P5/P11 violation; the engine's determinism contract is loose at f32 reductions.

**Root cause.** The event ring uses atomic-append (`atomicAdd(&event_tail[0], 1u)`) for producer slot acquisition. Within a tick, the order in which producer threads acquire ring slots is GPU-scheduling-dependent. Fold kernels then iterate the ring in insertion order and atomically accumulate into `view_storage_primary[slot]` via a CAS+add loop. f32 addition is not associative — the same set of contributions added in different orders produces different sums.

The CAS+add loop guarantees no contribution is dropped (atomicity), but does not pin the addition order. The current code comments claim it satisfies P11; they're wrong.

**Constitutional prescription.** P11 enforcement clause: *"emitter generates `sort_by(target_id)` before `atomic_*` reductions on materialized views; per-tick `seq` ordering injected into GPU atomic-append events."* This spec implements both halves.

## §2 Architectural Impact

- **P1 (Compiler-first):** entirely emit-side — new sort kernels are emitted, no hand-written engine code. Reinforces P1.
- **P3 (Cross-backend parity):** the serial CPU backend's chronicle path adopts the same `seq` + sort discipline. AOE oracle canonicalize precedent extends.
- **P7 (Replayable flagged):** `seq` lands on both replayable and non-replayable events uniformly.
- **P10 (No runtime panic):** radix sort must handle empty/single/full rings — covered by proptest.
- **P11 (Reduction determinism):** the fix itself. The enforcement clause becomes truthful for the first time.

## §3 Mechanism — `seq` + sort key + sort placement

### 3.1 `seq` field

Every event payload gets a new `seq: u32` slot, fixed as the last payload word. Producer threads write it at emit time. Packing:

```
seq = (producer_kernel_id << 24) | (producer_thread_id << 4) | intra_emit_idx
```

| Field | Bits | Range | Notes |
|---|---|---|---|
| `producer_kernel_id` | 8 | 0..256 | Densely allocated over **emit-producing kernels only** (5-15 per typical fixture; non-emit kernels like mask/scoring/plumbing carry no id). |
| `producer_thread_id` | 20 | 0..1M | Per-thread `agent_id` for PerAgent rules; event index for PerEvent rules. Covers the 10k-agent megaswarm fixture with headroom. |
| `intra_emit_idx` | 4 | 0..16 | Per-emit-statement counter within the producer body, assigned at lowering time. |

The lowering pass tracks `(producer_kernel_id, intra_emit_counter)` statically. A shared helper `compute_event_seq(kernel_id, thread_id, emit_idx) -> u32` lives in `dsl_compiler` (a leaf crate both backends consume); GPU shader inlines the same bit-packing math. Bit-equal by construction.

### 3.2 Sort key

Combined 64-bit: `(target_agent_id as u64) << 32 | (seq as u64)`. After sort, events with the same target are adjacent (target high bits dominate); intra-target order follows `seq`.

Target-less events (no `target: AgentId` payload field) use a sentinel high-bit pattern that places them at the end of the ring; folds skip past them via a single bounds check.

### 3.3 Sort placement in the schedule

A new `SortEventRing` kernel slots in at every **producer → consumer barrier**: each point in the schedule where a phase of emits is followed by a phase of reads. For simple fixtures with one barrier per tick (e.g., per_agent emits → folds), one sort per tick covers all downstream consumers. For chronicle cascades (e.g., forest_fire's Spread → Catch → Ignited chain), each re-emit phase introduces a fresh barrier; the schedule pass detects each one and inserts a sort kernel before the subsequent consumer phase.

The existing schedule already places `prev_event_tail_buf` snapshot copies at these barriers; the sort piggybacks on the same barrier-detection logic in `build_helper.rs`. Typical fixture: 1-2 sort kernels per tick. Worst case observed (multi-stage cascade with re-emit chain): ~4 barriers per tick.

### 3.4 Fold-path simplification

After sort, each fold thread sees its slot's events in a contiguous, deterministically-ordered run. The existing CAS+add loop for f32 fold accumulators becomes unnecessary — a plain `let accum = ...; storage[slot] = accum;` works because there is only one writer per slot. Perf win on top of correctness.

### 3.5 Opt-out

The build helper synthesizes sort kernels **only when at least one f32 view-fold accumulator exists**. Pure-u32-or-Or fixtures (commutative+associative, P11-trivial) get zero sort overhead. forest_fire opts in; cooldown_probe doesn't.

## §4 GPU radix sort

### 4.1 Algorithm

Standard parallel LSD radix sort, 8-bit chunks, **stable**. Input: `event_ring` + computed 64-bit sort keys. Output: permuted ring with events in `(target, seq)` order. Ping-pong buffer required — `event_ring_sort_scratch` allocated alongside the ring, same size.

### 4.2 Two-stage structure

To avoid 64-bit key gymnastics in WGSL:

1. **Stage A — sort by `seq` (low 32 bits):** 4 passes of 8-bit stable radix. After Stage A, the ring is globally seq-ordered.
2. **Stage B — stable sort by `target` (high 32 bits):** counting sort, O(N + agent_count). Three kernels: count-per-target, exclusive-scan, scatter. Cheap because target range is tightly bounded (≤ agent_count, typically 1024-10000).

Stage B's stability preserves Stage A's seq ordering as the tiebreak.

Total: 4 × 3 + 3 = **15 emitted sort kernels** per fixture that opts in. They share scratch and histogram buffers.

### 4.3 Stability mechanism

Within a bucket, order must be preserved (that's how `seq` survives the higher-bit `target` sort). Achieved with per-workgroup stable scan + workgroup-ordered merge. Reference: standard "stable parallel LSD radix" — well-documented, ~400-500 lines WGSL across 3 kernel templates (histogram, exclusive-scan, scatter) instantiated per pass.

### 4.4 Cost (estimated)

- **Compute (forest_fire scale, ~1k events/tick):** rough estimate 4-5× the cost of the current naked atomicAdd fold per barrier; multiply by number of barriers per tick. Single-digit milliseconds at megaswarm scale (~10k events/tick). The fold-path CAS+add removal partially offsets. Real numbers come during Slice 3 implementation — if costs exceed the estimate by more than 2×, revisit (a single-pass 32-bit sort using packed `(target_low, seq_low)` keys may be necessary for hot fixtures).
- **Memory:** `event_ring_sort_scratch` (= event_ring size) + `radix_histogram_buf` (256 × 4 bytes × num_workgroups). ~64 KB additional for forest_fire-sized fixtures. Negligible.

## §5 Cross-backend parity

### 5.1 CPU mirror

The serial CPU backend mirrors the GPU exactly, with a simpler implementation:

1. **Same `seq` packing.** Both backends call `compute_event_seq(kernel_id, thread_id, emit_idx)`.
2. **CPU sort matches GPU sort.** CPU fold path sorts the per-tick chronicle slice with `Vec::sort_by_key(|e| (e.target, e.seq))`. `sort_by_key` is stable in std; matches the stable radix LSD output on GPU. No need to mirror the radix algorithm itself — only the input/output ordering matters for parity.
3. **CPU producer iteration order doesn't matter.** Whatever order the CPU iteration happens to produce events, the sort fixes ordering. Same pattern as the AOE oracle (`parity_apply_program_sweep::canonicalize`).

### 5.2 Touchpoints

- `crates/engine/src/cascade/dispatch.rs` — emit `seq` on every event the dispatch path writes.
- `crates/engine/src/view/<fold>.rs` (per-view fold modules) — sort the per-tick events before reducing.
- `parity_apply_program_sweep::canonicalize` extends to all f32-reduction fixtures, not just AOE-keyed ones.

### 5.3 Parity test gate

Existing `tests/parity_*.rs` infrastructure already runs serial vs GPU and asserts byte-equal. Once GPU becomes deterministic and CPU adopts the matching sort, existing parity tests pass for fixtures that previously couldn't (the f32-fold ones). Forest_fire gets a new `parity_forest_fire.rs` as the acceptance gate.

## §6 Testing

### 6.1 Acceptance gates

1. **forest_fire pin tightens.** `max_abs_drift ≤ 150.0` → `== 0.0`. The slack-history comment block is deleted (slack gone).
2. **New probe `f32_reduction_determinism_probe.sim`.** 1 target + N producers each emitting one Damaged/tick at the same target, single f32 view fold. Run twice on GPU, assert byte-equal. ~30 lines `.sim` + ~80 lines pin.
3. **Cross-backend parity** for f32-reduction fixtures via the existing `parity_*.rs` infrastructure (forest_fire as canonical case).
4. **Emit-shape pins** in `crates/dsl_compiler/tests/`:
   - `event_seq_field_present.rs` — every event payload's last word is `seq`; producers write the packed value.
   - `radix_sort_kernels_emitted.rs` — fixtures with an f32 view fold get the 15 sort kernels in the expected schedule slot.
   - `sort_omitted_when_no_f32_fold.rs` — fixtures without f32 reductions don't get sort kernels.
5. **Radix-sort proptest** `crates/dsl_compiler/tests/proptest_radix_sort.rs` — random (key, value) sequences, WGSL sort vs Rust `Vec::sort_by_key`, byte-equal. Edge cases (P10): empty, single, ring-full, all-same-key, sorted-already, reverse-sorted.

### 6.2 Non-changes

- Behavioral pins that hash event payloads pick up the `seq` field automatically via schema regen. No pin rewrites needed.
- The post-CAS-emit-gating tests still pass — the f32 first-writer-wins shape is orthogonal; the gate logic still applies and doesn't interact with the per-stmt CAS detection. Sort doesn't touch the per-stmt CAS pattern.
- Existing emit-shape pins (`memory_ordering_cas_emit.rs`, etc.) keep passing — the CAS-loop emit shape doesn't go away, just becomes unused for f32-fold accumulators specifically.

### 6.3 Out of scope

- **Sort-then-fold for pair-keyed views** (Agent × Item, Agent × Group). Same bug class but the second key isn't an `AgentId`. Separate design.
- **`per_agent_u64` ahash drift** (`project_engine_pcg_ahash_drift`). Different P5 violation, different fix.

## §7 Phasing

Six dependency-ordered slices:

1. **Seq plumbing.** Add `seq: u32` last-word slot in every event payload. Lowering pass tags each emit with `(producer_kernel_id, intra_emit_idx)`. Emit writes packed seq. CPU mirror writes the same packing. No behavior change yet. Gate: `event_seq_field_present.rs`.

2. **CPU sort.** Serial backend's chronicle fold sorts by `(target, seq)` before reducing. CPU byte-stable across runs; GPU still drifts. Gate: CPU-only determinism test passes for forest_fire.

3. **GPU radix sort kernels.** 15 sort kernels (4 LSD-radix passes × 3 + 3 counting-sort kernels). Build_helper synthesizes when an f32 fold exists. Schedule inserts between producer and consumer phases. `event_ring_sort_scratch` allocation. Gate: `proptest_radix_sort.rs` + `radix_sort_kernels_emitted.rs`.

4. **Fold path simplification.** Drop the CAS+add loop for f32 fold kernels (single-writer-per-slot post-sort). Plain serial sum into a local + indexed store. Gate: emit pin asserts CAS+add gone; forest_fire pin still passes.

5. **Pin tightening + parity gate.** Forest_fire pin from `≤150` to `==0`. Add `parity_forest_fire.rs`. P11 becomes truthful. Gate: byte-equal across reruns AND across backends.

6. **Opt-out audit.** Sweep all fixtures; assert no-f32-fold ones (cooldown_probe, boids, etc.) get zero sort overhead. Gate: `sort_omitted_when_no_f32_fold.rs`.

**Parallelization.** Slices 2 and 3 are independent (CPU vs GPU). Slice 4 depends on 3. Slice 5 depends on 2+3+4. Slice 6 can land any time after 3.

**Risk + fallback.** Slice 3 (radix sort) is the highest-risk piece — non-trivial WGSL with stability guarantees. If it slips, Slices 1+2+5(partial) still ship a usable intermediate state: CPU backend deterministic + serial-only parity for f32-fold fixtures. Incomplete-but-shippable while GPU sort matures.
