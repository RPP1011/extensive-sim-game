# Physics WGSL Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder body in `engine_gpu_rules/src/physics.wgsl` with a real, naga-validated, runtime-executable physics kernel — and **retire the CPU forward inside `step_batch`'s tick body** so GPU finally becomes authoritative for the simulation. This is the work that turns "parity gate compiles" into "parity gate is real."

**Architecture:** The `dsl_compiler::emit_physics_wgsl` emitter produces per-rule physics function bodies that reference an "integration-phase" runtime layer (state accessors, typed event-slot view of the cascade ring, ability registry walk, view reads, standing/memory mutators). That runtime was the bulk of the pre-T16 hand-written physics WGSL (~3200 lines in `crates/engine_gpu/src/physics.rs`); T16 deleted it; T15+T16+Stream A landed clean compiles but the runtime was never rebuilt. This plan rebuilds it phase-by-phase, with the per-kernel `physics_parity` test as the per-phase gate, and the CPU-forward retirement as the final acceptance gate.

**Tech Stack:** Rust, wgpu 26.0.1, naga (parse + pipeline-creation), `dsl_compiler::emit_physics_wgsl`, `dsl_compiler::emit_runtime_prelude_wgsl` (pattern reuse).

**Architectural Impact Statement (P8):**

- **P1 (compiler-first):** preserved — every WGSL helper this plan emits comes from a `dsl_compiler::emit_*_wgsl` module. No hand-written WGSL ships in `engine_gpu_rules/src/`.
- **P2 (schema-hash):** affects `engine_gpu_rules/.schema_hash` (content hash regenerates as new emitted modules land). `engine/.schema_hash` unaffected unless this plan changes SoA layout (Phase A doesn't; Phase B+ may need a `PackedAbilityRegistry` schema bump — flagged per phase).
- **P3 (cross-backend parity):** **THIS PLAN IS WHAT MAKES P3 REAL.** Today `parity_with_cpu` passes tautologically — both sides run `engine_rules::step::step`. After the CPU forward retires (Phase F), parity becomes a genuine byte-equality assertion of GPU output against CPU reference.
- **P5 (deterministic RNG):** Phase A introduces `per_agent_u32_glsl` to the WGSL runtime. The constitution names `tests/rng_cross_backend.rs` as the gate; this plan creates it.
- **P6 (events as mutation channel):** preserved — every state mutation in physics still flows through events into the cascade ring.
- **P10 (no runtime panic):** preserved by construction — WGSL has no panics; host-side wiring uses `Result` / `expect("...")` only on init paths.
- **P11 (reduction determinism):** preserved — atomic-append ordering on `next_event_tail` is per-tick stable; standing/memory mutations use the same sort-then-fold pattern as the existing materialized views.

- **Runtime gate:** `physics_parity` test (currently file-cfg-gated) re-enabled and PASSING. Plus `parity_with_cpu` continues to pass after the Phase F CPU-forward retirement. Both are byte-equality assertions against `SerialBackend` reference output. Without these, no phase is "done."

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [x] AIS reviewed post-design (after task list stabilises).

**Sequencing:** Each phase is independently committable. Phase A lands the simple-rule runtime (damage/heal/shield/stun/slow/transfer_gold/opportunity_attack — 7 rules). Phase B adds cast/ability dispatch. Phase C adds standing/memory. Phase D wires view reads. Phase E wires the per-kernel `physics_parity` test. Phase F retires the CPU forward — the moment of truth.

---

## Phase A — Core simple-rule runtime

Unblocks 7 of 24 physics rules: `damage`, `heal`, `shield`, `stun`, `slow`, `transfer_gold`, `opportunity_attack`. These don't need ability registry, view reads, or standing/memory.

### Task A1: WGSL runtime helper module — state accessors

**Files:**
- Create: `crates/dsl_compiler/src/emit_physics_runtime_wgsl.rs`

The module emits a parameterizable WGSL prelude for physics kernels. Distinct from `emit_runtime_prelude_wgsl` (which targets movement/apply_actions emitting to `event_ring_records`) — physics emits to `next_event_ring`/`next_event_tail`.

What this task lays down:
- Const block: `EVENT_KIND_*` (~20 consts mirroring `engine_data::events::EventKind`), `EFFECT_OP_KIND_*` (~10 consts mirroring `EffectOp` discriminants — needed for Phase B but cheap to land here).
- `slot_of(id: u32) -> u32` helper (id 1-based → 0-based slot, with `0xFFFFFFFFu` sentinel for invalid).
- `alive_bit(slot: u32) -> bool` against an `alive_bitmap: array<u32>` binding the consumer kernel declares.
- `wgsl_world_tick: u32` const reads `sim_cfg[0]` (tick is at offset 0 of SimCfg — verified via `crates/engine_gpu/src/sync_helpers.rs`).
- `state_agent_<field>(slot: u32) -> <type>` getters for hp / max_hp / shield_hp / attack_damage / alive / creature_type / engaged_with / stun_expires_at / slow_expires_at / slow_factor_q8 / cooldown_next_ready / pos_x / pos_y / pos_z. Each is a single-line offset arithmetic read against `agents: array<u32>` SoA (16 u32 stride per `GpuAgentSlot`).
- `state_set_agent_<field>(slot: u32, value: <type>)` writers for the mutable-by-physics fields: hp, shield_hp, alive (kill_agent shortcut), stun_expires_at, slow_expires_at, slow_factor_q8.
- `state_kill_agent(slot: u32)` shortcut for `state_set_agent_alive(slot, 0u)`.
- `state_add_agent_gold(slot: u32, delta: i32)` against a `gold_buf: array<i32>` binding. Atomic? No — gold mutations land via cascade events, not contention; non-atomic add is fine. (Verify; if multiple events can apply the same delta in one cascade iter, atomicAdd<i32> is needed. Per pre-T16 source there's no atomic — non-atomic is correct.)
- Cascade-side `gpu_emit_event(kind, tick, p0..p7)` writing to `next_event_ring: array<u32>` + `next_event_tail: atomic<u32>`. Mirrors `emit_runtime_prelude_wgsl::gpu_emit_event` but with renamed bindings.
- Per-kind helpers `gpu_emit_agent_attacked`, `gpu_emit_agent_died`, `gpu_emit_effect_*` etc. — same shape as the apply-path prelude, different ring.

- [ ] **Step 1: Survey field offsets**

Run: `grep -n "pub.*: f32\|pub.*: u32" crates/engine_gpu/src/sync_helpers.rs` — captures every `GpuAgentSlot` field in declaration order. Write the offset table into the module's doc comment so future readers don't have to count.

- [ ] **Step 2: Survey EVENT_KIND values**

Run: `grep -nE "^pub enum EventKind|^\s+[A-Z][a-zA-Z]* = " crates/engine_data/src/events.rs` — captures every event kind discriminant. The WGSL `EVENT_KIND_*` consts must match the Rust discriminants byte-for-byte.

- [ ] **Step 3: Implement the emitter**

`pub fn emit_physics_runtime_wgsl() -> String` — returns the helper block. NO bindings declared (the consumer physics.wgsl declares them at fixed slots); the helpers reference identifiers the consumer establishes.

Layout:
```rust
pub fn emit_physics_runtime_wgsl() -> String {
    let mut out = String::new();
    out.push_str(EVENT_KIND_CONSTS);   // const block
    out.push_str(EFFECT_OP_KIND_CONSTS);
    out.push_str(SLOT_HELPERS);         // slot_of, alive_bit, wgsl_world_tick
    out.push_str(&STATE_GETTERS);       // hp, max_hp, ... read fns
    out.push_str(&STATE_SETTERS);       // set_hp, kill_agent, add_gold, ...
    out.push_str(CASCADE_EMIT_FNS);     // gpu_emit_event + per-kind helpers
    out
}
```

- [ ] **Step 4: Naga test with synthetic bindings**

Add an inline `#[cfg(test)]` test that constructs a synthetic shader: declare every binding the runtime expects (agents, alive_bitmap, sim_cfg, gold_buf, next_event_ring, next_event_tail), `include_str!` the runtime, add a `cs_test` entry that calls every fn. Assert `naga::front::wgsl::parse_str` accepts the result.

- [ ] **Step 5: Register module + commit**

Add `pub mod emit_physics_runtime_wgsl;` to `crates/dsl_compiler/src/lib.rs`.

Run: `cargo test -p dsl_compiler --lib emit_physics_runtime_wgsl` — must PASS.

```bash
git add crates/dsl_compiler/src/emit_physics_runtime_wgsl.rs crates/dsl_compiler/src/lib.rs
git commit -m "feat(dsl_compiler): physics WGSL runtime helpers (Phase A1)"
```

### Task A2: Per-rule WGSL emission for the 7 simple rules

**Files:**
- Create: `crates/dsl_compiler/src/emit_physics_wgsl_module.rs` — top-level wrapper that produces the full `physics.wgsl` body (bindings + runtime + per-rule bodies + dispatcher)
- Modify: `crates/xtask/src/compile_dsl_cmd.rs` — call the new wrapper instead of writing the stub

The existing `emit_physics_wgsl` (in `dsl_compiler::emit_physics_wgsl`) emits one rule body at a time. The new wrapper:

1. Declares the BGL slots (10 bindings — see `engine_gpu_rules/src/physics.wgsl` current shape).
2. Includes `emit_physics_runtime_wgsl()` (Phase A1 output).
3. For each rule in `compilation.physics`: skip `@cpu_only`, skip rules that need Phase B/C/D infrastructure (cast, modify_standing, record_memory — emit a stub body that no-ops). Emit the rest via `emit_physics_wgsl(physics, &ctx)`.
4. Emit the `physics_dispatch(event_idx)` switch — calls only the lit-up rules.
5. Emit the `cs_physics(@builtin(global_invocation_id))` entry that walks `current_event_tail` and dispatches.

- [ ] **Step 1: Create the wrapper**

`pub fn emit_physics_wgsl_module(compilation: &Compilation, supported_rules: &[&str]) -> Result<String, EmitError>` — `supported_rules` is the whitelist (Phase A: 7 simple-rule names; later phases extend).

- [ ] **Step 2: Wire xtask**

Replace `compile_dsl_cmd.rs`'s `physics.wgsl` stub-write with a call to `emit_physics_wgsl_module(&compilation, &PHASE_A_RULES)`.

`PHASE_A_RULES` const: `["damage", "heal", "shield", "stun", "slow", "transfer_gold", "opportunity_attack"]`.

- [ ] **Step 3: Run xtask + naga gate**

Run: `cargo run --bin xtask -- compile-dsl`
Run: `cargo test -p engine_gpu_rules --test naga_parse_all`

Expected: PASS. The 7 simple rules' bodies + the runtime helpers + the bindings combine into a naga-valid shader. The 17 unsupported rules (cast, modify_standing, record_memory, plus future) are stubbed to no-op.

- [ ] **Step 4: Schema-hash + workspace + gpu builds**

```bash
cargo test -p engine_gpu_rules --test schema_hash    # PASS (content hash bumped)
cargo build --workspace                               # clean
cargo build -p engine_gpu --features gpu              # clean
```

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/ crates/xtask/ crates/engine_gpu_rules/
git commit -m "feat(dsl_compiler): physics.wgsl real body for 7 simple rules (Phase A2)"
```

### Task A3: Re-enable physics_parity test (gated to A's rule set)

**Files:**
- Modify: `crates/engine_gpu/tests/physics_parity.rs` (remove `#![cfg(any())]`, rewrite)

The pre-T16 test ran the physics kernel against a single tick and compared output to a CPU reference. Post-T16, the equivalent shape is:

1. Build a fixture where ONLY the 7 supported rules fire (mostly: combat-event fixtures — attacks → damage/heal events).
2. Step CPU once via `engine_rules::step::step`.
3. Step GPU once via `step_batch` (CPU forward still in place — yes, this means the gate is still tautological for Phase A; that's OK — Phase F is when the gate becomes real).
4. Assert `assert_cpu_gpu_parity` passes on the fixture.

- [ ] **Step 1: Build a Phase-A-targeted fixture**

Add `simple_combat_fixture_n8()` to `crates/engine_gpu/tests/common/mod.rs` — 8 agents in close proximity, mixed factions, full HP. Stepping triggers attacks → damage → death (covers `damage`, exercises the kernel via cascade events).

- [ ] **Step 2: Rewrite physics_parity.rs**

Same shape as `parity_with_cpu.rs`, with the simple-combat fixture and `n_ticks` chosen so cascade events actually fire (likely 3-5 ticks).

- [ ] **Step 3: Run + commit**

```bash
cargo test -p engine_gpu --test physics_parity
# Expected: PASS (today via CPU forward; remains real after Phase F)
git add crates/engine_gpu/tests/physics_parity.rs crates/engine_gpu/tests/common/mod.rs
git commit -m "test(engine_gpu): re-enable physics_parity for Phase A rule set"
```

---

## Phase B — Cast / ability dispatch

Adds the runtime + binding wiring for the `cast` physics rule. Brings 1 of the 17 deferred rules online, but it's the most-touched rule (every ability-emitting agent fires it).

### Task B1: PackedAbilityRegistry GPU binding

The `PackedAbilityRegistry` Pod struct exists in `engine_gpu/src/sync_helpers.rs` (recovered from pre-T16). Phase B uploads it as a GPU buffer + adds it to the BGL.

**Files:**
- Modify: `crates/engine_gpu_rules/src/external_buffers.rs` (generated — change comes via `dsl_compiler::emit_external_buffers`)
- Modify: `crates/engine_gpu_rules/src/physics.rs` (regenerated — adds binding slot)
- Modify: `crates/engine_gpu/src/lib.rs` — upload the registry to a GPU buffer at backend init

- [ ] **Step 1-N: standard binding-add flow**

Bind a new slot `ability_registry: array<u32>` (raw u32 view of `PackedAbilityRegistry`'s flat layout) to the physics BGL. Wire the upload at `ensure_resident_init`.

### Task B2: Ability registry helper fns

**Files:**
- Modify: `crates/dsl_compiler/src/emit_physics_runtime_wgsl.rs` — add `abilities_is_known`, `abilities_effects_count`, `abilities_effect_op_at` helpers + the `EffectOp` match-on-discriminant pattern.

Helpers do offset arithmetic against `ability_registry: array<u32>` per the `PackedAbilityRegistry` Rust layout.

### Task B3: Lift the `cast` rule into `PHASE_B_RULES`

**Files:**
- Modify: `crates/dsl_compiler/src/emit_physics_wgsl_module.rs` — extend `PHASE_A_RULES` to `PHASE_AB_RULES = [..., "cast"]`.

### Task B4: Re-enable physics_parity covering cast

Extend the fixture to spawn agents with abilities; assert parity after a tick where casts fire.

```bash
git commit -m "feat(dsl_compiler): physics cast rule + ability-registry runtime (Phase B)"
```

---

## Phase C — Standing + memory mutations

`modify_standing` and `record_memory` rules need WGSL helpers that write into the `standing_storage: array<u32>` and `memory_storage: array<u32>` bindings (already in physics.wgsl's BGL — slots 6, 7).

### Task C1: Symmetric_pair_topk WGSL writers

**Files:**
- Modify: `crates/dsl_compiler/src/emit_physics_runtime_wgsl.rs` — add `state_adjust_standing(observer, target, delta: i32)` that walks/inserts/evicts in the symmetric_pair_topk K=8 layout.

Reuses the layout `emit_symmetric_pair_topk_fold_wgsl` reads (from Stream B). The writer is the inverse — finds an existing edge or evicts the weakest. Layout is documented in `dsl_compiler::emit_view_wgsl` for the symmetric_pair_topk shape.

### Task C2: Per_entity_ring WGSL writers

**Files:**
- Modify: `crates/dsl_compiler/src/emit_physics_runtime_wgsl.rs` — add `state_push_agent_memory(observer, subject, flags, confidence, tick)` that does atomic ring-cursor bump + record write.

Layout matches `emit_per_entity_ring_fold_wgsl` from Stream B.

### Task C3: Lift `modify_standing` + `record_memory` into PHASE_ABC_RULES

### Task C4: Re-enable cold_state_standing and cold_state_memory tests

Re-enable: `cold_state_standing.rs`, `cold_state_memory.rs`, `cold_state_gold_transfer.rs`. All cold-state cascade tests should now pass.

```bash
git commit -m "feat(dsl_compiler): physics standing + memory mutations (Phase C)"
```

---

## Phase D — View reads inside physics

Some physics rules (per the IR's view-dependency graph) consult views during their bodies. Today the emitter outputs `view_<name>_get(...)` calls that don't resolve. This phase lights up those view-read fns by reusing the view storage primitives Stream B's view-fold-helpers will produce.

**Note:** Phase D depends on the `view-fold-helpers` follow-up plan landing first (or in parallel). If view-fold-helpers is parked, Phase D is parked with it — but the simple/cast/standing/memory rules (Phases A-C) can ship without view reads.

### Task D1-N: Per-view view_read helper

For each materialized view physics consults, add a `view_<name>_get(...)` WGSL fn to `emit_physics_runtime_wgsl`. Inverse of the fold emitter — reads from the view's primary storage.

```bash
git commit -m "feat(dsl_compiler): physics view-read fns (Phase D)"
```

---

## Phase E — Per-kernel parity + cascade tests

Re-enable the per-kernel parity tests now that physics has real bodies:

- `physics_parity.rs` (already done in Phase A3, extended in B4 + C4)
- `cascade_parity.rs` — exercise the FixedPoint Physics arm
- `indirect_cascade_converges.rs`
- `batch_iter_cap_convergence.rs`

Each follows the smoke-test pattern from Stream C Task 2, but using fixtures that exercise multi-iter cascades.

```bash
git commit -m "test(engine_gpu): re-enable cascade-touching parity gates (Phase E)"
```

---

## Phase F — Retire the CPU forward

**The moment of truth.** All physics rules have real WGSL bodies. The CPU forward inside `step_batch`'s tick body is now redundant — and removing it makes parity REAL.

### Task F1: Remove `engine_rules::step::step` from step_batch's tick body

**Files:**
- Modify: `crates/engine_gpu/src/lib.rs` — remove the per-tick `engine_rules::step::step(...)` call (currently lines ~588-593 inside `step_batch`'s `for _tick_idx in 0..n_ticks`)
- Modify: `crates/engine_gpu/src/lib.rs` — `step_batch` must now do its own state advancement: read GPU buffers back into `SimState`, advance `state.tick`, drain the GPU event ring into `events: &mut EventRing<Event>`. Or: keep state on GPU and only sync at snapshot points.

Architectural decision needed at this step: does `step_batch` immediately readback every tick (slow, defeats GPU), or accumulate on GPU and readback only at batch boundaries (fast, but `state` mirror is stale within a batch)? Pre-T16 chose accumulate; replicate that.

### Task F2: parity_with_cpu becomes real

`parity_with_cpu_n4_t10` and `_t100` must still PASS — but now genuinely. The CPU side runs `engine_rules::step::step`; the GPU side runs `step_batch` which is GPU-authoritative. If they diverge, that's a real correctness bug — fix the WGSL body, not the assertion.

### Task F3: Final verification

```bash
cargo test --workspace
cargo test -p engine_gpu --features gpu
```

All Stream C tests + new physics_parity + cascade_parity must pass. If any fails, root-cause the WGSL body.

```bash
git commit -m "fix(engine_gpu): retire CPU forward — GPU is now authoritative (Phase F)"
```

---

## Final verification

After all phases, the following invariants hold:

1. `parity_with_cpu` PASSES — and the assertion is **real** (CPU runs `engine_rules::step::step`; GPU runs `step_batch` GPU-authoritative; bytes equal).
2. `physics_parity`, `cascade_parity`, `indirect_cascade_converges`, `batch_iter_cap_convergence`, `cold_state_*` all PASS.
3. `cargo build --workspace` clean. `cargo build -p engine_gpu --features gpu` clean.
4. The 24 physics rules are all live (or explicitly tagged `@cpu_only` if appropriate).
5. No `engine_rules::step::step` call inside `step_batch`'s tick body.
6. The runtime gate (`physics_parity`, `parity_with_cpu`) catches any kernel regression.
7. **GPU is finally doing useful work.** P3 is genuinely satisfied. P1's compiler-first vision is realized for physics.

---

## What this plan deliberately does NOT do

- **Does NOT touch the 6 PairMap/SlotMap fold modules.** Those are owned by `view-fold-helpers` (which Phase D depends on but doesn't produce). Run that plan in parallel or after.
- **Does NOT rewrite the 3 spatial kernels.** Owned by `spatial-rewrite`. Independent of physics.
- **Does NOT touch the chronicle pipeline.** Chronicles are non-replayable telemetry; orthogonal.
- **Does NOT optimize.** The emitted WGSL is correctness-first. Performance work (workgroup tuning, memory layout, megakernel fusion) is downstream.

---

## Honest scope estimate

| Phase | Scope | Estimated effort |
|---|---|---|
| A | 7 simple rules + state runtime + EVENT_KIND consts | 2-4 hours |
| B | Cast rule + PackedAbilityRegistry binding + dispatch helpers | 3-5 hours |
| C | Standing + memory storage writers (sym_pair_topk + per_entity_ring) | 2-4 hours |
| D | View reads (depends on view-fold-helpers) | 1-2 hours |
| E | Per-kernel parity tests re-enable | 1-2 hours |
| F | CPU forward retirement + final acceptance gate | 1-2 hours, high-stakes |
| **Total** | **— end-to-end physics on GPU —** | **10-19 hours** |

This is a multi-day plan. Each phase is independently committable; partial completion is meaningful (Phase A alone gets simple combat correctly running on GPU under parity). Phase F is the binary acceptance gate — without it, GPU is still a no-op alongside CPU.
