# GPU Cold-State Replay

**Status:** design approved, plan pending
**Date:** 2026-04-22
**Subsystem:** (2) of the GPU-everything follow-up
**Depends on:** subsystem (1) — needs `@cpu_only` annotation and `SimCfg`
**Prerequisite for:** subsystem (3) — else gold-transfer and standing-shift abilities silently no-op on the batch path

## Problem

After subsystem (1), tick + world seed + sim-wide scalars live on GPU. But the engine's **cold-state replay** — ~11 CPU rule handlers that consume events and mutate host-side state (chronicle narrative, transfer_gold, modify_standing, record_memory, and others) — stays CPU-resident.

The resident cascade (`cascade_resident.rs`) **silently skips** these handlers today: events emitted on GPU that would trigger cold-state rules don't get processed in batch mode. This is masked because:

1. `async_smoke` asserts on agent counts + event counts, not on gold/standing/memory state mutations.
2. The test fixtures are combat-focused; no gold or standing activity.
3. The parity tests run the sync path, which keeps cold-state on CPU.

For subsystem (3) to land (GPU ability evaluation), abilities that use `EffectOp::TransferGold` or `EffectOp::ModifyStanding` need their handlers to actually run on the batch path. Today, those abilities' effects would silently drop when cast during a batch.

## Goal

Port cold-state handlers to run on GPU during the batch path. Three sub-items:

- **4a — chronicle stubs** (8 rules): structured chronicle entries go to GPU via the existing physics-kernel chronicle ring. Narrative-text rules stay `@cpu_only` and are consumed async from the batch snapshot's event log.
- **4b — gold + standing**: `transfer_gold` physics rule (already DSL-defined) gets its existing DSL body auto-lowered to GPU; `SparseStandings` is reborn as a DSL `@materialized @symmetric_pair_topk(K=8)` view and folded GPU-side like `kin_fear`.
- **4c — record_memory**: per-agent memory ring is reborn as a DSL `@materialized @per_entity_ring(K=64)` view.

All rule bodies are DSL-lowered — no hand-written game-logic WGSL. Engine-core infrastructure (side buffers, dispatch plumbing, new view storage shapes) is hand-written Rust + WGSL per the engine's standing rule: "engine core = hand-written; game logic = DSL."

## Non-goals

- Moving the sync `step()` cold-state path. The sync path stays on CPU; this subsystem targets the batch path only.
- Ability-caused gold/standing cascades that require fetching CPU-side strings or unbounded structures. Any such rules stay `@cpu_only`.
- Re-architecting the CPU `CascadeRegistry`. Its existing API is fine; this subsystem adds GPU equivalents for a subset of registered rules.
- Chronicle rendering to strings. Strings stay CPU. Only structured chronicle records go to GPU.

## Architecture

### Dispatch framework (cross-cutting)

One WGSL handler kernel per GPU-resident event kind. At end of each tick's cascade iteration in the batch path, the cascade driver scans observed events by kind and dispatches the matching kernel(s).

Emission is automatic: every DSL `physics rule @phase(event) { on EventKind { ... } }` that isn't `@cpu_only` gets a WGSL kernel emitted by the compiler AND an entry in a generated "dispatch by event-kind" table. No hand-written dispatcher kernel — the driver is Rust glue reading the compiler-emitted dispatch table.

Dispatch is *per event kind*, not *per event* — we don't dispatch a kernel 100× for 100 AgentAttacked events; we dispatch once and the kernel processes all matching events in the slice.

Event kinds with zero observations in a tick skip dispatch entirely (CPU reads the event count per kind, as it already does for the cascade seed kernel).

Placement in `step_batch`: end-of-tick, after cascade converges, before next tick starts. All inside one command encoder — no new submits.

### 4a — Chronicle stubs

**Status on GPU**: the physics kernel already writes `ChronicleEntry` to a dedicated GPU chronicle ring (committed earlier in the session, task 203, commit `b5294ed7`). So chronicle emission from physics rules is already GPU-native.

**Per-rule audit** (8 chronicle rules). Two tiers:

- **Structured chronicle** (port to GPU by removing the CPU-only gate): rules that emit a fixed-layout `ChronicleEntry` record — template id + fixed payload fields. Example: `chronicle_attack` emits `(kind=ChronicleKind::Attack, actor, target, damage)`. These port by changing the rule body in `physics.sim` (or wherever chronicle rules live) to emit through the existing DSL chronicle-emission primitive rather than through a CPU-only side effect.
- **Narrative-text chronicle** (stays `@cpu_only`): rules that format multi-sentence prose with string interpolation. Example: `chronicle_death` that formats `"{name} fell to {killer}'s blow after {ticks} ticks of combat"`. These use `@cpu_only` annotation and run async off the batch snapshot's event stream.

The audit decides per-rule; the spec captures the decision as a table (to be filled during implementation).

No new DSL grammar needed for 4a — the existing chronicle-emission primitive suffices for structured entries, and `@cpu_only` (from subsystem 1) handles narrative ones.

### 4b — Gold

**Rule**: `transfer_gold` already exists in `physics.sim:133` as a DSL rule: `agents.sub_gold(from, a); agents.add_gold(to, a)`. It lowers to `crates/engine/src/generated/physics/transfer_gold.rs` on the CPU side and `emit_physics_wgsl.rs:813` assumes i32 amount already. After the gold-narrowing commit (`d78c565a`), CPU and GPU types align.

**What's missing**: gold needs to be accessible on GPU. Currently `Inventory.gold: i32` lives on CPU in `SimState.cold_inventory`, not in `resident_agents_buf` (`GpuAgentSlot`).

**Storage choice**: dedicated GPU buffer.

```
struct GoldBuf {
    per_agent: array<atomic<i32>>, // length = agent_cap
}
```

Not part of `GpuAgentSlot` — gold is touched by one handler, no reason to bloat every kernel's view. ~8 KB at N=2048, scales linearly. Written by the `transfer_gold` GPU handler (atomic add/sub), read by scoring rules that gate on gold (`gold > threshold`).

At `ensure_resident_init`, gold is uploaded from `SimState.cold_inventory` into `gold_buf`. On `snapshot()`, gold is read back via staging copy into `SimState.cold_inventory` so CPU observers see current values.

### 4b — Standing

**Current state**: `SparseStandings` in `crates/engine/src/state/agent_types.rs:136` — a hand-written `BTreeMap<(AgentId, AgentId), i16>` with `set`, `get`, `adjust` methods. Event-driven via `EffectStandingDelta { a, b, delta }` but the handler is CPU-only.

**Migration**: reborn as a DSL `@materialized` view with a new `@symmetric_pair_topk(K=8)` annotation. Declaration lives alongside other views (likely `views.sim` or equivalent):

```
view standing
    @materialized
    @symmetric_pair_topk(K = 8)
    // pair key is (min(a,b), max(a,b)); value is i16 clamped to [-1000, 1000]
{
    on EffectStandingDelta { a, b, delta } {
        // add delta to pair(a, b); clamp to [-1000, 1000]
        state[min(a,b), max(a,b)] += delta
    }
}
```

DSL compiler work:
1. Parse the new `@symmetric_pair_topk` annotation.
2. Emit CPU fold code (replaces hand-written `SparseStandings::adjust`).
3. Emit GPU fold kernel (analogous to `kin_fear`'s fold kernel, but:
   - Key uses pair-symmetric storage (each agent stores its K edges; reads dedupe via min/max invariant).
   - No decay (`kin_fear` decays; standing is persistent).
   - Clamp value to signed 16-bit range.
4. Delete hand-written `SparseStandings` from `agent_types.rs`. Consumers migrate to `state.views.standing.*` accessors.

Schema hash bump for the view addition and `SparseStandings` removal.

**Per-agent storage**: `N × (8 × 6 bytes + u32 count)` ≈ 56 KB at N=2048, 2.5 MB at N=100k. Bounded. When the K=8 budget overflows on an agent, the weakest (lowest |standing|) entry gets dropped — this is acceptable because standings near zero have no gameplay effect.

### 4c — Record memory

**Current state**: `MemoryEvent` smallvec per agent (`smallvec64<MemoryEvent>`). Event-driven via `RecordMemory` event. CPU-only handler.

**Migration**: reborn as a DSL `@materialized` view with a new `@per_entity_ring(K=64)` annotation:

```
view memory
    @materialized
    @per_entity_ring(K = 64)
{
    on RecordMemory { observer, source, fact_payload, confidence } {
        push into state[observer] as MemoryEvent { source, fact_payload, confidence, tick }
    }
}
```

DSL compiler work:
1. Parse `@per_entity_ring` annotation.
2. Emit CPU fold code (FIFO ring with eviction at K=64; replaces hand-written smallvec push+pop).
3. Emit GPU fold kernel:
   - Per-agent ring with a write cursor (u32 per agent).
   - On push, atomically increment cursor mod K, write `MemoryEvent` record at `ring[agent][cursor % K]`.
   - Reads return all K slots in cursor-relative order.
4. Delete hand-written memory handling in `agent_types.rs`. Consumers migrate to `state.views.memory.*`.

**Per-agent storage**: `N × (64 × 24 bytes + u32 cursor)` ≈ 3 MB at N=2048, ~150 MB at N=100k. At 100k this starts being non-trivial; if it becomes a memory concern, K shrinks or we revisit the ring shape. For now, accept the cost — memory is a load-bearing gameplay feature.

### New DSL annotations (compiler scope)

Two new annotation variants consumed by this subsystem:

- `@symmetric_pair_topk(K)` — pair-keyed per-entity storage, top K most-significant edges per entity, symmetric (each pair stored once).
- `@per_entity_ring(K)` — per-entity FIFO ring of fixed size K.

Both extend the existing `@materialized` view infrastructure that already supports `@per_entity_topk` (used by `kin_fear`).

Compiler implementation:
- New IR node types for the annotations.
- New CPU emit paths (Rust code for the CPU fold handler).
- New GPU emit paths (WGSL kernel for the GPU fold handler + storage binding layout).

### Engine-core vs DSL-lowered split (for this subsystem)

- **Engine-core (hand-written Rust + WGSL)**:
  - Side buffer allocation + binding plumbing for `gold_buf`, standing view storage, memory view storage.
  - Dispatch table driver in Rust (reads compiler-emitted dispatch manifest, dispatches per-event-kind kernels in `step_batch`).
  - Snapshot handshake: `gold_buf` read-back on `snapshot()` that mutates `SimState.cold_inventory`; similar for standing and memory views.
  - `ensure_resident_init` additions for uploading initial state.
- **DSL-lowered**:
  - All rule bodies (`transfer_gold`, `chronicle_*`, `modify_standing`, `record_memory`).
  - All view fold bodies (standing fold, memory fold, etc.).
  - All CPU-side handlers that were previously hand-written — deleted, replaced by lowered equivalents.
  - New annotation processing (`@symmetric_pair_topk`, `@per_entity_ring`, `@cpu_only`).

## Data flow — one tick's cold-state processing on batch path

```
(after cascade converges, end of tick, still in same encoder):

for each GPU-resident event kind K observed this tick:
    if event_count[K] > 0:
        dispatch K's handler kernel (reads events_in, mutates K's target storage)
        // e.g. transfer_gold kernel reads EffectGoldTransfer slice, atomic-ops on gold_buf

view fold kernels run after handlers (unchanged from today, gain new views):
    my_enemies, threat_level, kin_fear, pack_focus, rally_boost, engaged_with, standing, memory
                                                                   ^^^^^^^^   ^^^^^^
                                                                   new        new

(snapshot(), when called):
    gold_buf          → copy into SimState.cold_inventory[*].gold
    standing view     → copy into SimState.cold_standing (or expose via state.views)
    memory view       → copy into SimState.cold_memory (or expose via state.views)
```

## Error handling

- **Dispatch table drift**: if a rule is added to the DSL but the dispatch table isn't regenerated, the rule silently doesn't run on GPU. Mitigation: compile-time check that every non-`@cpu_only` rule appears in the generated dispatch manifest.
- **Standing topk overflow**: when the K=8 budget per agent fills, the lowest-magnitude standing gets evicted. Logged in debug builds.
- **Memory ring overflow**: ring by design evicts oldest on overflow. Not an error.
- **Gold atomic overflow**: with i32 amounts, accumulating +2.1B of deltas overflows. Use `atomicAdd` which wraps; gameplay gets weird but not catastrophic. Cap gold-producing rules at the ability-design level.
- **View storage allocation failure at agent_cap grow**: existing mechanism in `view_storage.rs` — rebuild + reinitialise. Applies to new views too.

## Testing

### New tests

- `cold_state_4a_structured_chronicle.rs` — batch-path cast that triggers a structured chronicle rule. Snapshot the chronicle ring slice. Assert expected records present.
- `cold_state_4a_narrative_is_cpu_only.rs` — confirm a `@cpu_only` narrative chronicle rule still runs on CPU but not on GPU (snapshot.chronicle_since_last empty for that rule's output, but event appears in events_since_last for CPU consumer to process).
- `cold_state_4b_gold.rs` — batch-path cast with `TransferGold` effect. Snapshot gold_buf. Assert balances updated atomically.
- `cold_state_4b_standing.rs` — batch-path cast with `ModifyStanding` effect. Snapshot standing view. Assert pair-symmetric delta applied, clamped.
- `cold_state_4c_memory.rs` — batch-path `RecordMemory` emits. Snapshot memory view. Assert K=64 ring with FIFO eviction behaviour.
- `dsl_symmetric_pair_topk.rs` — DSL-compiler-level test: minimal view with the new annotation lowers to correct CPU + GPU emit.
- `dsl_per_entity_ring.rs` — DSL-compiler-level test: minimal view with the new annotation lowers correctly.
- `dsl_cpu_only_skips_dispatch_table.rs` — end-to-end: `@cpu_only` rule absent from the generated GPU dispatch manifest.

### Regression

All existing tests pass. The sync path continues to use CPU handlers (no behaviour change). Batch-path tests (`async_smoke`, `snapshot_double_buffer`, etc.) still pass; cold-state sub-items that now run on GPU just *add* correctness, they don't break existing assertions.

### Non-goals for testing

- No CPU/GPU parity test for cold-state mutations. Batch path is non-deterministic by design; exact CPU-vs-GPU state equivalence isn't the contract. Statistical parity (gold conservation laws, standing symmetry invariants, memory ring FIFO order) suffices.

## Phase decomposition

Four phases, landed in order:

**Phase 1 — Compiler extensions**. Add `@symmetric_pair_topk(K)` and `@per_entity_ring(K)` DSL annotations. Add CPU + GPU emit paths. Lands first because Phase 2/3/4 consume these. Includes unit tests at the DSL-compiler level.

**Phase 2 — 4a chronicle stubs**. Per-rule audit, port structured chronicle rules by removing CPU-only gates, mark narrative rules `@cpu_only`. Dispatch framework wiring in `step_batch` for cold-state handlers. Integration test for structured chronicle emission.

**Phase 3 — 4b gold + standing**. Add `gold_buf` side buffer and its readback handshake. Port `SparseStandings` to the new `@symmetric_pair_topk` view. Delete hand-written `SparseStandings`. Schema hash bump.

**Phase 4 — 4c record_memory**. Port memory smallvec to `@per_entity_ring` view. Delete hand-written memory handling. Schema hash bump.

Phases 3 and 4 are largely parallel (different storage shapes, different DSL annotations, minimal shared code beyond Phase 1's framework). Phase 2 is independent of 3 and 4.

## Open questions

- **MemoryEvent struct layout on GPU**: needs to be Pod-compatible. Current CPU version has `source: AgentId, kind: u8, payload: u64, confidence_q8: u8, tick: u32`. 24 bytes with padding. Confirm no surprises in alignment when lowered to WGSL.
- **Standing evict policy**: is "lowest |standing|" the right tie-breaker at K-budget overflow? Or "oldest edge"? Probably |standing| since game rules treat near-zero standing as "not notable" — drop those first. Revisit if gameplay suggests otherwise.
- **Gold snapshot granularity**: should `snapshot()` always include gold, or gate behind a flag? Current `GpuSnapshot` doesn't include it. Probably add `snap.gold: Vec<i32>` keyed by agent slot; caller applies to `SimState.cold_inventory` if they care.
- **Per-rule audit for 4a**: the 8 chronicle rules need enumerating with their structured-vs-narrative classification. Placeholder in the spec; fill during Phase 2 implementation kickoff.
