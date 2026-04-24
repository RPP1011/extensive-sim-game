# Investigation Brief: `step_batch` Chronicle Emit Bug

> **For agentic workers:** This is an **investigation brief**, not a step-by-step plan. The bug's root cause is not known. Task flow: diagnose → hypothesise → test → fix → regression. You may need to iterate several cycles.

**Context:** During Phase 2 of the GPU cold-state replay plan (commits `be5a31e1` + downstream), the snapshot path was wired to read the resident chronicle ring. Two integration tests at `crates/engine_gpu/tests/chronicle_batch_path.rs` were added to assert chronicle records flow through — both are currently `#[ignore]`d.

**Known empirical facts (already gathered; do not re-do):**

- Running `step_batch(5)` with a Human+Wolf combat fixture produces 8 `AgentAttacked` events visible in `batch_events_ring`. Apply_actions IS emitting events correctly.
- `seed_kernel` writes `num_events[0]=1` (for the last tick, which had 1 event) and `indirect_args[0]=(1,1,1)` (1 workgroup of 64 threads). So iter 0 IS dispatched.
- The resident chronicle ring tail stays at `0` across all 5 ticks. First 16 u32s of the records buffer are all zeros — no chronicle writes landed.
- The sync chronicle ring (a DIFFERENT buffer — `PhysicsKernel::chronicle_ring` vs `CascadeResidentCtx::chronicle_ring`) has 2 records from the warmup `step()` call and stays at 2 across `step_batch(5)`. So the resident physics is NOT accidentally writing to the sync buffer either.
- GPU `sim_cfg.tick` advances 1 → 6 correctly across `step_batch(5)` — the seed kernel IS running each tick.
- The isolated `physics_run_batch_resident_smoke::run_batch_resident_nonzero_input_publishes_next_slot` test, which directly calls `PhysicsKernel::run_batch_resident` with a seeded AgentAttacked event in an upload_storage'd `events_in` buffer, DOES produce chronicle records. So the kernel + bind group + chronicle ring work in isolation.
- The resident WGSL (dumped via naga parse test) correctly contains `chronicle_attack` as a function matching `ev_rec.kind == EVENT_KIND_AGENT_ATTACKED`, calling `gpu_emit_chronicle_event(2u, a, t, wgsl_world_tick)`. Dispatcher at `physics_dispatch(event_idx)` correctly routes to chronicle_attack on AgentAttacked.

**Conclusion so far:** the bug is specifically in how `step_batch` wires `apply_event_ring` into the resident physics's `events_in` binding (slot 7). Isolated `run_batch_resident` works; full `step_batch` doesn't. Something on the full-flow path is either:
- Binding a different buffer at slot 7 than what the seed kernel reads from.
- Reading the correct buffer but with stale contents / layout mismatch.
- Running iter 0 on an event that doesn't have kind=AgentAttacked (despite `batch_events_ring` containing 8 such events aggregated across all ticks).
- Some synchronisation issue where apply_actions' atomic writes haven't landed before iter 0 reads events_in.

**Reference:** `docs/GPU_WORKGRAPH.md` — read before starting. `crates/engine_gpu/tests/chronicle_batch_path.rs` has the comprehensive diagnostic notes in the file-level doc.

---

## Investigation approach

### Step 1: Instrument more of the data flow

The existing diagnostics showed `num_events[0]=1` for the LAST tick. But what about earlier ticks? Each tick's seed kernel overwrites `num_events[0]`; only the final value is readable at snapshot time.

Options:
- Snapshot after EACH tick in `step_batch` instead of after all 5 — requires adding a test that calls `step_batch(1)` five times with intervening snapshots. Each call does its own resets + dispatches, so this approximates per-tick visibility.
- Add a GPU-side debug counter that increments once per chronicle emit and sums across all ticks. A single u32 atomic buffer written from a new debug stub.

### Step 2: Probe events_in directly

Write a one-off test that:
1. Calls `step_batch(1)` on the Human+Wolf fixture.
2. Before snapshot runs, read back `apply_event_ring.records_buffer()` directly (`readback_typed::<EventRecord>`) at byte offset 0 for the first 2-4 records.
3. Compare against what `apply_actions.run_resident` should have written.

This pins down whether apply_event_ring.records actually contains the AgentAttacked bytes iter 0 is supposed to read.

### Step 3: Test hypothesis — seed kernel reads stale tail

If apply_event_ring.records HAS the events but `num_events[0]` is 0 when the seed kernel reads the tail, then iter 0 dispatches with 0 workgroups → no chronicle emits. Test: read `apply_event_ring.tail_buffer()` directly after apply_actions but before seed runs. The seed kernel reads it via `atomicLoad(&apply_tail[0])` — if the atomic tail is 0 when read, that's the bug.

Unlikely given the 8 events showing up in `batch_events_ring` (which also reads apply_event_ring.tail), but worth ruling out.

### Step 4: Test hypothesis — iter 0 events_in binding wrong

The `iter_rings(0, apply_event_ring.records_buffer())` call at `crates/engine_gpu/src/cascade_resident.rs:919-937` returns `apply_event_ring.records_buffer()` for iter 0. The bind group at `crates/engine_gpu/src/physics.rs:2067` should bind this buffer at slot 7.

Double-check the bind group cache: if a previous `run_batch_resident` call cached a bind group with a DIFFERENT events_in buffer, the cache might hit on iter 0's key and return the stale BG. Review the ResidentBgKey to see if events_in is in the key. If yes, confirmed OK.

### Step 5: Test hypothesis — synchronisation ordering

All dispatches in `step_batch` are in ONE command encoder submitted once at the end. wgpu docs say compute passes within a single submit execute serially. Verify: is the apply_actions → cascade seed → physics iter 0 chain actually in ONE encoder, or does apply_actions do its own submit internally?

Check:
- `apply_actions.run_resident` (crates/engine_gpu/src/apply_actions.rs:567) — does it call `queue.submit` internally or only `encoder.encode`?
- `cascade_ctx.physics.run_batch_resident` — same check.

If any intermediate stage submits internally, that creates an implicit boundary but should still maintain ordering. Edge case: does `ensure_pool_cap` or `ensure_resident_pool` call `queue.write_buffer` inside the encoder's time window? `write_buffer` is a queue op that may bypass encoder ordering.

### Step 6: Compare with the isolated test that works

`physics_run_batch_resident_smoke::run_batch_resident_nonzero_input_publishes_next_slot`:
- Uses `upload_storage` for events_in (a test-local buffer, not apply_event_ring).
- Seeds `num_events_buf[0]=1` directly (not via seed kernel reading apply_tail).
- Dispatches once, submits once.

Key difference: the isolated test bypasses the seed kernel entirely. If you swap the isolated test to seed via `apply_event_ring` + seed kernel while keeping everything else the same, does it still work? That test would bisect whether the bug is in `apply_event_ring` binding or in `seed_kernel → num_events → iter 0` chain.

### Step 7: The "obvious" thing to try

Read `cascade_resident.rs:run_cascade_resident_with_iter_cap` very carefully, focusing on buffer lifetimes + aliasing. The `apply_ring_ptr: *const GpuEventRing` + `unsafe { &*apply_ring_ptr }` pattern at `crates/engine_gpu/src/lib.rs:1025-1028` is a code smell — there might be a subtle aliasing issue where the reborrow creates a use-after-free or aliases `cascade_ctx.apply_event_ring` through both `&` and `&mut`. Unlikely but worth tracing with `cargo miri` if in doubt.

---

## Scope discipline

- **This is a diagnostic + fix task.** The fix may be small (1-20 lines). The investigation is the bulk of the work.
- Do NOT add new features while investigating. Focused, minimal fixes only.
- Do NOT touch gold_buf, standing_storage, or subsystem 3 code — those are parallel tracks.
- Preserve the ignored tests in `chronicle_batch_path.rs` until your fix is in place; then un-ignore them as part of the fix commit.
- If investigation takes more than ~4-6 hours without progress, pause and report hypotheses + what was ruled out. Don't spin.

## Commit discipline

- Each diagnostic test you add can be its own commit if it's reusable (e.g. "test(engine_gpu): probe apply_event_ring contents after step_batch").
- The fix commit gets `fix(engine_gpu): step_batch chronicle emit — <root cause>` message. Include in the body WHICH hypothesis turned out to be right and why the others were ruled out — future debugging relies on this.
- Un-ignore the two chronicle_batch_path tests in the same commit as the fix.

## Regression

After the fix:
```
cargo test --release --features gpu -p engine_gpu --test chronicle_batch_path 2>&1 | tail -10
cargo test --release --features gpu -p engine_gpu 2>&1 | tail -15
```

Both tests should un-ignore and pass. No new failures.

**Also**: add a large-scale stress test at `N=20000`. Spawn dense combatants, run `step_batch(50)`, snapshot, assert `snap.chronicle_since_last.len() > 0` (or some reasonable minimum given the expected attack cadence). This catches regressions that would only manifest under realistic workloads. Drop any CPU-parity-style assertions — assert only on GPU-path state.

## Test-scale guideline for this investigation

- **Diagnostic probe tests**: may start at small N (2 agents) to isolate the bug. That's fine during investigation.
- **Post-fix regression tests**: land a large-scale (`N=20000`) stress test to prevent regression under the realistic workload. Kernel-level microfixtures stay as they are.
- **GPU-only assertions**: don't compare against sync-path results; the batch path's own invariants are the contract.

## Report format

Status (DONE / DONE_WITH_CONCERNS / BLOCKED / NEEDS_CONTEXT), commit SHAs (list if multiple), what the root cause turned out to be, which hypothesis was right, how many cycles of test → hypothesis → ruled-out before the fix. Under 400 words.

If BLOCKED after significant effort: list every hypothesis you investigated, what the result was for each, and what you'd try next given unlimited time. That sets up the next iteration.
