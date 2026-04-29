# Physics WGSL Runtime Implementation Plan (Phase-F-first)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GPU backend authoritative for physics simulation, with `parity_with_cpu` as a real (not tautological) byte-equality gate. Achieved by retiring the CPU forward FIRST, accepting that parity goes RED, then closing the diff phase-by-phase as physics WGSL bodies land. Each phase's success criterion is a *measurable parity-diff reduction* — never a tautological pass.

**Why Phase-F-first.** The original plan structure (build rules first, retire CPU forward last) had two flaws: (1) 10-19 hours of work would happen under tautological tests that prove nothing; (2) the highest-risk architectural decision (readback strategy) would surface at the END, after all the rule work was already invested. Phase-F-first inverts both: parity is the visible compass from Day 1, and the readback architecture is paid up front.

**Architecture:** `engine_gpu::step_batch`'s tick body currently does `engine_rules::step::step(&mut SerialBackend, ...)` after the SCHEDULE-loop dispatch — the GPU encoder records dispatches but the CPU step that follows overwrites whatever the GPU wrote. This plan removes that CPU forward (Phase 0). After removal, `step_batch` must advance `state.tick` from GPU readback (a 4-byte `sim_cfg.tick` read), and `state.agents_*` SoA mutations come from the GPU agents buffer (per-tick readback for correctness — perf optimization is downstream). `parity_with_cpu` then becomes a real CPU-vs-GPU byte-equality assertion that fails on every rule still on a stub WGSL body. Each subsequent phase lights up a rule family; parity diff shrinks; final acceptance is parity GREEN.

**Tech Stack:** Rust, wgpu 26.0.1, naga (parse + pipeline-creation), `dsl_compiler::emit_physics_wgsl`, `dsl_compiler::emit_runtime_prelude_wgsl` (pattern reuse).

**Architectural Impact Statement (P8):**

- **P1 (compiler-first):** preserved — every WGSL helper this plan emits comes from a `dsl_compiler::emit_*_wgsl` module. No hand-written WGSL ships in `engine_gpu_rules/src/`.
- **P2 (schema-hash):** affects `engine_gpu_rules/.schema_hash` (content hash regenerates as new emitted modules land). `engine/.schema_hash` unaffected unless this plan changes SoA layout (Phase 1 doesn't; Phase 2+ may need a `PackedAbilityRegistry` schema bump — flagged per phase).
- **P3 (cross-backend parity):** **THIS PLAN IS WHAT MAKES P3 REAL.** Today `parity_with_cpu` passes tautologically. After Phase 0 it goes RED — that's the planned, controlled break. Each subsequent phase shrinks the diff. Phase 5's acceptance gate is parity GREEN with no CPU forward.
- **P5 (deterministic RNG):** Phase 1 introduces `per_agent_u32_glsl` to the WGSL runtime. The constitution names `tests/rng_cross_backend.rs` as the gate; this plan creates it.
- **P6 (events as mutation channel):** preserved — every state mutation flows through events into the cascade ring.
- **P10 (no runtime panic):** preserved by construction — WGSL has no panics; host-side wiring uses `Result` / `expect("...")` only on init paths.
- **P11 (reduction determinism):** preserved — atomic-append ordering on `next_event_tail` is per-tick stable; standing/memory mutations use the same sort-then-fold pattern as the existing materialized views.

- **Runtime gate:** This is the canonical case the plan-template-ais runtime-gate field exists for.
  - **During execution (Phase 0 onward):** `parity_with_cpu` is the compass. It will be RED until Phase 5. Reduction in field-divergence count is the per-phase progress signal.
  - **Acceptance (Phase 5):** `parity_with_cpu` PASSES with no CPU forward. `step_batch_runtime_smoke` continues to pass throughout (it asserts only tick advance, which Phase 0 must preserve via GPU readback).

- **What this plan accepts being broken:** `parity_with_cpu` (intentional, tracked as the per-phase compass), `physics_parity` and `cascade_parity` if re-enabled mid-flight (they should stay file-cfg-gated until their physics rules land). Workspace builds must remain clean throughout. `step_batch_runtime_smoke` must remain GREEN throughout (Phase 0 is responsible).

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).

---

## Phase 0 — Retire the CPU forward, accept parity break

The pivotal architectural phase. After this, `step_batch` is GPU-authoritative for state mutation; `parity_with_cpu` goes RED; the diff becomes the compass for every subsequent phase.

### Task 0.1: Pick the readback strategy

**Decision needed before code lands.** Two options:

| Strategy | Latency | `state` accuracy mid-batch | Implementation |
|---|---|---|---|
| Per-tick readback | Slow (sync wait per tick) | Always up-to-date | Simple — read `sim_cfg.tick` + `agents_buf` after every queue.submit |
| Accumulate-then-readback at batch boundary | Fast (one sync per batch) | Stale within batch | Complex — needs CPU/GPU mirror sync at snapshot points |

**Picked: per-tick readback.** Correctness-first. Perf is downstream (`gpu_megakernel_plan`).

- [ ] **Step 1: Document the choice**

Add a comment block at the top of `step_batch` in `crates/engine_gpu/src/lib.rs` explaining the per-tick readback decision and naming the perf follow-up plan.

### Task 0.2: Wire per-tick readback for `state.tick`

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs` — `step_batch` per-tick body

After each `queue.submit(...)` in the per-tick loop, before the next iteration:

```rust
// Per-tick GPU → CPU sync of authoritative scalars. Tick lives at
// sim_cfg[0]; readback is 4 bytes synced via Maintain::Wait.
self.device.poll(wgpu::Maintain::Wait);
let tick_u32 = readback_u32_at_offset(&self.device, &self.queue, sim_cfg_buf, /*offset_bytes=*/ 0);
state.tick = tick_u32;
```

(Helper `readback_u32_at_offset` may already exist in `crates/engine_gpu/src/gpu_util/`. If not, add it.)

- [ ] **Step 1: Locate or add `readback_u32_at_offset`**

`grep -n "readback_typed\|readback_u32" crates/engine_gpu/src/gpu_util/`

If `readback_typed::<u32>` exists (it does — used in snapshot.rs), use it directly with offset arithmetic.

- [ ] **Step 2: Wire the readback after queue.submit**

Inside step_batch's `for _tick_idx in 0..n_ticks` loop, after the submit but before the next iteration's encoder creation, do the tick readback. (Note: needs to happen INSIDE the loop, since each tick must see the prior tick's GPU-advanced tick value before computing the next dispatch's cfg.)

But wait — in the current shape, the SCHEDULE-loop dispatcher records into a single encoder for all n_ticks, then submits once at the end. After Phase 0, we need to submit per-tick (so we can readback per-tick). That's a structural change.

Update the loop:

```rust
for _tick_idx in 0..n_ticks {
    let mut encoder = self.device.create_command_encoder(&Default::default());
    for op in engine_gpu_rules::schedule::SCHEDULE {
        self.dispatch(op, &mut encoder, state)?;
    }
    self.queue.submit(Some(encoder.finish()));
    self.device.poll(wgpu::Maintain::Wait);
    state.tick = readback_tick_from_sim_cfg(&self.device, &self.queue, sim_cfg_buf);
}
```

But `state.tick` won't actually advance unless the seed_indirect kernel (or some kernel) writes to `sim_cfg[0]`. Today it doesn't (the prelude module opted out of the atomic tick bump). The honest replacement: advance `state.tick` host-side **and** sync to GPU. After Phase 0:

```rust
for _tick_idx in 0..n_ticks {
    state.tick += 1;
    upload_sim_cfg_tick(&self.queue, sim_cfg_buf, state.tick);
    let mut encoder = ...;
    for op in SCHEDULE { self.dispatch(...); }
    self.queue.submit(Some(encoder.finish()));
    self.device.poll(wgpu::Maintain::Wait);
}
```

State.tick is now advanced on CPU (cheap) and uploaded to GPU before each tick's dispatch. After all phases, agents_buf readback (Task 0.3) brings agent state back. This is the simplest correctness-first shape.

- [ ] **Step 3: Run `step_batch_runtime_smoke`**

```bash
cargo test -p engine_gpu --test step_batch_runtime_smoke
```

Expected: PASS. tick advances by N after step_batch(N).

### Task 0.3: Wire per-tick readback for `state.agents_*` SoA

After each tick's submit + poll, read the GPU agents buffer back and decode into `SimState`'s SoA. The decoder already exists in `crates/engine_gpu/src/sync_helpers.rs::unpack_agent_slots(state, &slots)`.

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs` — step_batch loop body

```rust
// After submit + poll:
let agent_slots = readback_agents_buf(&self.device, &self.queue, agents_buf, agent_cap);
crate::sync_helpers::unpack_agent_slots(state, &agent_slots);
```

`readback_agents_buf` reads `agent_cap × 64 bytes` of `GpuAgentSlot` Pods.

- [ ] **Step 1: Locate or add the readback helper**

The pre-T16 cascade.rs had this helper; it was deleted. Re-add a minimal version:

```rust
// In crates/engine_gpu/src/sync_helpers.rs:
pub fn readback_agents_buf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    agents_buf: &wgpu::Buffer,
    agent_cap: u32,
) -> Vec<GpuAgentSlot> {
    let bytes = agent_cap as u64 * std::mem::size_of::<GpuAgentSlot>() as u64;
    let raw = crate::gpu_util::readback::readback_typed::<u8>(device, queue, agents_buf, bytes)
        .expect("readback_agents_buf");
    bytemuck::cast_slice::<u8, GpuAgentSlot>(&raw).to_vec()
}
```

- [ ] **Step 2: Wire into step_batch**

After each tick's submit + poll, call `readback_agents_buf` then `unpack_agent_slots`.

- [ ] **Step 3: Run runtime smoke**

```bash
cargo test -p engine_gpu --test step_batch_runtime_smoke
```

Expected: PASS. Tick advance + agent state mirror both work.

### Task 0.4: Remove the CPU forward

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs:588` — delete the `engine_rules::step::step(&mut SerialBackend, ...)` call inside the per-tick loop.
- Modify: `crates/engine_gpu/src/lib.rs:555` — same call in the resident-init-failure fallback. **Keep this one** — it's the fallback when GPU init fails; the simulation needs to advance somehow. The fallback path is rare and not the parity-tested path.

After this task, the per-tick body looks like:

```rust
for _tick_idx in 0..n_ticks {
    state.tick += 1;
    upload_sim_cfg_tick(&self.queue, sim_cfg_buf, state.tick);
    let mut encoder = self.device.create_command_encoder(&Default::default());
    for op in SCHEDULE {
        self.dispatch(op, &mut encoder, state)?;
    }
    self.queue.submit(Some(encoder.finish()));
    self.device.poll(wgpu::Maintain::Wait);
    let agent_slots = readback_agents_buf(...);
    crate::sync_helpers::unpack_agent_slots(state, &agent_slots);
    // NO CPU forward. GPU is authoritative for state.
}
```

- [ ] **Step 1: Delete the call**

- [ ] **Step 2: Run `parity_with_cpu` and observe RED**

```bash
cargo test -p engine_gpu --test parity_with_cpu
```

**Expected: FAILS.** This is the planned, controlled break. Capture the failure output — the printed "first diverging agent slot" lines are the compass for Phase 1.

Sanity-check the failure shape matches expectations: every agent's `hp_bits` and `pos_*_bits` should differ, since GPU runs no real rules and CPU runs all of them.

- [ ] **Step 3: Run `step_batch_runtime_smoke` — must still PASS**

```bash
cargo test -p engine_gpu --test step_batch_runtime_smoke
```

Expected: PASS. Tick advance + tick state mirror still work.

### Task 0.5: Commit

```bash
git add crates/engine_gpu/src/
git commit -m "feat(engine_gpu): retire CPU forward — GPU is authoritative; parity is now RED (Phase 0)

step_batch's tick body no longer calls engine_rules::step::step. Per-
tick readback of state.tick + agents SoA via the sync_helpers
infrastructure. parity_with_cpu now FAILS — the failure is the
compass for Phase 1+. step_batch_runtime_smoke continues to PASS.

Architectural decision: per-tick readback (correctness-first), not
accumulate-then-readback (perf-first). Perf optimization is
downstream (gpu_megakernel_plan)."
```

`parity_with_cpu` is now in a documented RED state. Subsequent phases bring it back to GREEN.

---

## Phase 1 — 7 simple rules + state runtime

Lights up: `damage`, `heal`, `shield`, `stun`, `slow`, `transfer_gold`, `opportunity_attack`. After this phase, `parity_with_cpu` should be GREEN for fixtures that only fire these rules; the simple_combat_fixture_n8 from physics_parity becomes the per-phase compass.

### Task 1.1: WGSL runtime helper module

(Identical to Phase A1 from the original plan — see commit message + structure unchanged.)

**Files:**
- Create: `crates/dsl_compiler/src/emit_physics_runtime_wgsl.rs`

Emits:
- `EVENT_KIND_*` consts (~38 — verified by surveying `crates/engine_data/src/events/mod.rs:88`)
- `EFFECT_OP_KIND_*` consts (~10 — verified by surveying `EffectOp` discriminants)
- `slot_of(id)`, `alive_bit(slot)`, `wgsl_world_tick`
- `state_agent_<field>(slot)` getters: hp / max_hp / shield_hp / attack_damage / alive / creature_type / engaged_with / stun_expires_at / slow_expires_at / slow_factor_q8 / cooldown_next_ready / pos_x / pos_y / pos_z
- `state_set_agent_<field>(slot, value)` writers: hp / shield_hp / alive / stun_expires_at / slow_expires_at / slow_factor_q8
- `state_kill_agent(slot)` shortcut
- `state_add_agent_gold(slot, delta)` against `gold_buf: array<i32>`
- Cascade-side `gpu_emit_event` writing to `next_event_ring`/`next_event_tail` + per-kind helpers

Naga test: synthesize a shader with all bindings + `cs_test` that exercises every helper. Assert parse.

- [ ] Step 1: Survey field offsets (already known — see `sync_helpers::GpuAgentSlot`)
- [ ] Step 2: Survey EVENT_KIND values + EFFECT_OP discriminants
- [ ] Step 3: Implement the emitter
- [ ] Step 4: Naga test
- [ ] Step 5: Register module + commit

### Task 1.2: emit_physics_wgsl_module wrapper + xtask wiring

(Identical to Phase A2 from the original plan.)

**Files:**
- Create: `crates/dsl_compiler/src/emit_physics_wgsl_module.rs`
- Modify: `crates/xtask/src/compile_dsl_cmd.rs`

`emit_physics_wgsl_module(compilation, supported_rules)` produces full physics.wgsl: bindings + runtime + per-rule bodies + dispatcher + entry. `supported_rules` is the per-phase whitelist.

`PHASE_1_RULES = ["damage", "heal", "shield", "stun", "slow", "transfer_gold", "opportunity_attack"]`.

Unsupported rules emit a no-op body:
```wgsl
fn physics_<name>(event_idx: u32) {
    // STUB: rule lit up in Phase <N>.
    let _ev = events_in[event_idx];
}
```

- [ ] Step 1-5: implement, wire, run xtask, naga gate, schema-hash, commit

### Task 1.3: Run parity_with_cpu — observe diff shrink

```bash
cargo test -p engine_gpu --test parity_with_cpu
```

The full simple_combat fixture exercises damage events. After Phase 1, the parity diff for that fixture should be SMALLER than after Phase 0 — fewer divergent fields per agent. **If the diff doesn't shrink, the WGSL bodies have a real bug.** Diagnose via the helper's "first diverging agent" output.

The fixture for `parity_with_cpu` is `smoke_fixture_n4` (4 agents in a square, full HP). At small N and short ticks, attacks are the dominant rule, and after Phase 1 the diff should be 0 or near-0 for that fixture — so `parity_with_cpu_n4_t1` may go GREEN here.

`parity_with_cpu_n4_t100` will likely still be RED — over 100 ticks, cast events fire (Phase 2) and standing/memory updates accumulate (Phase 3).

- [ ] Step 1: Run parity_with_cpu
- [ ] Step 2: Document which sub-tests are green / which still-red and why (commit message)
- [ ] Step 3: If `_t1` is green, that's a Phase 1 milestone — commit

### Task 1.4: Re-enable physics_parity

(Per Phase A3 from original plan. Now also a real test, not tautological.)

```bash
git commit -m "feat(dsl_compiler): physics simple-rule runtime + 7 rule bodies (Phase 1)"
```

---

## Phase 2 — Cast / ability dispatch

Lights up: `cast`. After this phase, the parity diff for cast-heavy fixtures shrinks.

### Task 2.1: PackedAbilityRegistry GPU binding

(Per Phase B1 from original plan.)

### Task 2.2: Ability registry runtime helpers

(Per Phase B2 from original plan.)

### Task 2.3: Lift `cast` into PHASE_12_RULES whitelist

### Task 2.4: Run parity_with_cpu — diff should shrink for cast-heavy fixtures

A new test fixture `cast_combat_fixture` may be needed to exercise abilities — agents spawned with abilities, cast events fire each tick.

```bash
git commit -m "feat(dsl_compiler): physics cast rule + ability-registry runtime (Phase 2)"
```

---

## Phase 3 — Standing + memory

Lights up: `modify_standing`, `record_memory`. After this phase, parity for standing/memory-touching fixtures shrinks.

(Per Phase C from original plan.)

```bash
git commit -m "feat(dsl_compiler): physics standing + memory mutations (Phase 3)"
```

---

## Phase 4 — View reads inside physics

Depends on `view-fold-helpers` plan. If that plan hasn't run, Phase 4 is parked.

(Per Phase D from original plan.)

```bash
git commit -m "feat(dsl_compiler): physics view-read fns (Phase 4)"
```

---

## Phase 5 — Acceptance gate

The binary acceptance test: `parity_with_cpu` PASSES across all 3 sub-tests (n=1, n=10, n=100) on the standard `smoke_fixture_n4`, with no CPU forward in step_batch.

Additional acceptance:
- `physics_parity`, `cascade_parity`, `indirect_cascade_converges`, `batch_iter_cap_convergence` — re-enable, all PASS
- `cold_state_*` (gold, memory, standing) — re-enable, all PASS
- `step_batch_runtime_smoke` continues to PASS
- workspace + gpu-feature builds clean
- critics 3/3 PASS

If parity fails at this gate, root-cause the WGSL — do NOT relax the test.

```bash
git commit -m "test(engine_gpu): physics parity GREEN across all fixtures — acceptance (Phase 5)"
```

---

## Final verification

After all phases, the following invariants hold:

1. `parity_with_cpu_n4_t1`, `_t10`, `_t100` all PASS — the assertion is **real** (CPU runs `engine_rules::step::step`; GPU runs `step_batch` GPU-authoritative; bytes equal).
2. `physics_parity`, `cascade_parity`, `indirect_cascade_converges`, `batch_iter_cap_convergence`, `cold_state_*` all PASS.
3. `cargo build --workspace` clean. `cargo build -p engine_gpu --features gpu` clean.
4. The 24 physics rules are all live (or explicitly tagged `@cpu_only` if appropriate).
5. **No `engine_rules::step::step` call inside `step_batch`'s tick body.** (The init-failure-fallback at line 555 stays.)
6. **GPU is finally doing useful work.** P3 is genuinely satisfied. P1's compiler-first vision is realized for physics.

---

## What this plan deliberately does NOT do

- **Does NOT touch the 6 PairMap/SlotMap fold modules.** Those are owned by `view-fold-helpers` (which Phase 4 depends on but doesn't produce). Run that plan in parallel or after.
- **Does NOT rewrite the 3 spatial kernels.** Owned by `spatial-rewrite`. Independent.
- **Does NOT touch the chronicle pipeline.** Chronicles are non-replayable telemetry; orthogonal.
- **Does NOT optimize.** The emitted WGSL is correctness-first. Performance work (workgroup tuning, memory layout, megakernel fusion) is downstream.
- **Does NOT pretend `parity_with_cpu` is healthy during execution.** It's intentionally RED from Phase 0 through Phase 4. The tracker is the diff, not the binary pass/fail.

---

## Honest scope estimate

| Phase | Scope | Estimated effort |
|---|---|---|
| 0 | Readback + CPU-forward retirement; parity goes RED | 2-3 hours |
| 1 | 7 simple rules + state runtime + EVENT_KIND consts | 3-5 hours |
| 2 | Cast rule + PackedAbilityRegistry binding + dispatch helpers | 3-5 hours |
| 3 | Standing + memory storage writers | 2-4 hours |
| 4 | View reads (depends on view-fold-helpers parallel plan) | 1-2 hours |
| 5 | Per-kernel parity tests re-enable + acceptance | 1-2 hours |
| **Total** | **— end-to-end physics on GPU, parity REAL —** | **12-21 hours** |

Each phase is independently committable. After Phase 1 the small-fixture parity gate goes GREEN. After Phase 2 cast-heavy fixtures GREEN. After Phase 5 everything is GREEN — and the test means it.
