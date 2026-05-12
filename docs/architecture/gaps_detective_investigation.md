# Gaps surfaced by `detective_investigation`

Adversarial fixture authored to stress four mega-crate auto-emit
surfaces simultaneously: pair-keyed view storage growth without decay
over 1000 ticks, `rng.action()` chance gating in physics, conditional
verb dispatch, and multi-row scoring with view-derived weights. The
pin (`crates/sims/tests/detective_investigation_pin.rs`) drives 3
Detectives + 15 Suspects (3 marked guilty by host seed) for 1000
ticks and reports per-detective belief accuracy.

## What works

- **Parse + resolve.** `cargo run -p dsl_compiler --example resolve_one detective_investigation` prints `RESOLVE OK`.
- **Kernel emit.** 20 kernels emit cleanly: `fold_evidence`, `fold_accusation_count`, `fold_total_witnessed`, `physics_ObserveAndAccrue`, `physics_ApplyDamageFromChronicle`, `physics_verb_chronicle_{Investigate,Observe,Accuse}`, `scoring`, the spatial-build chain, the standard plumbing.
- **`rng.action()` lowering inside an inner spatial loop.** The emitted WGSL for `physics_ObserveAndAccrue` includes the `per_agent_u32(seed, agent_id, tick, 1u)` call inside the per-pair candidate loop; the `% 100u < threshold` gate composes via the standard u32 BinaryOp arms (the same shape `stochastic_probe` exercises in a per-agent body).
- **Mixed if-expression with two u32 arms** (the per-suspect `guilty_sentinel` branch picking between `innocent_threshold_q100` and `guilty_threshold_q100`) lowers to `select(config_1, config_2, (local_0 == config_3))`.
- **Pair-keyed view storage sized N²**. `view_storage_primary_buf = N × N × 4 bytes` (matches the existing pair-keyed path).
- **`step()` runs without panic for 1000 ticks** with the 18-agent population. View storage stays valid; readback succeeds; alive bits intact for every slot.
- **Events ARE emitted by `ObserveAndAccrue`** (event_tail = 2 per tick observed via mid-run readback — small but nonzero, confirming the producer path is wired).

## Gaps surfaced

### Gap 1 — Spatial-grid backing buffers under-sized (N×4 instead of GRID_DIM³ × 4)

**The big one.** The auto-emitter sizes every spatial buffer the same way it sizes per-agent buffers — `agent_count × 4 bytes`. But the spatial grid kernels (auto-emitted from any `@spatial(...)` declaration) write into a fixed-grid topology with `SPATIAL_GRID_DIM = 22u`, i.e. **10 648 cells**. The kernel-side reads index by cell:

```wgsl
let _start = spatial_grid_starts[_cell];
let _end = spatial_grid_starts[_cell + 1u];
```

With the buffer sized at 18 u32s but the WGSL reading 10 648 cells worth, every read past index 17 returns zero (silent OOB → returns 0 in WGSL storage). `_start == _end == 0` at every cell except the trampled-upon first 18 → the inner candidate loop never enters → `spatial.nearby_targets(self)` returns the empty set → `Witnessed` events emit at miniscule rate (only the bottom-of-grid cell's contents ever appear).

**Repro:** `RUST_MIN_STACK=33554432 cargo test -p sims --release --test detective_investigation_pin -- --nocapture` shows `event_tail=2` per tick (vs the expected ~7 for the modeled ~17% firing rate × 3 detectives × ~15 candidates). The hill_raid fixture's "siege didn't animate" failure mode at commit `1c565df9` is the same gap surfaced from a different angle.

**Surface to fix:** `crates/dsl_compiler/src/build_helper.rs::slot_count_expr` (or its caller). Spatial buffers (`spatial_grid_cells`, `spatial_grid_offsets`, `spatial_grid_starts`) need a third sizing arm — `(GRID_DIM as u64).pow(3) + 1` for `_starts` (the `+ 1` covers the `_cell + 1u` lookahead read), `agent_cap × MAX_PER_CELL` for `_cells`, etc. The WGSL constants `SPATIAL_GRID_DIM` / `SPATIAL_MAX_PER_CELL` need to flow into the host-side sizing decision.

**Workaround for the pin:** none today — the pin reports the symptom (`event_tail` mismatch) and proceeds to the soft-pin "zero events" branch. The structural assertions still pass (no panic, no overflow, alive bits intact). When the spatial-sizing gap closes the pin's accuracy + accusation count numbers should immediately become meaningful.

### Gap 2 — All views in the fixture share `view_storage_primary_buf`

The auto-emitter emits ONE `view_storage_primary_buf` field on `GeneratedRuntime` and aliases every fold kernel's `view_storage_primary` binding to it. For this fixture three views (evidence, accusation_count, total_witnessed) all write into the same physical buffer with no per-view offsetting. The pair-keyed `evidence` view's writes at `[detective * agent_cap + suspect]` collide with the per-agent `accusation_count` and `total_witnessed` writes at `[detective]`.

**Symptom:** the test pin's per-detective accusation_count readback is meaningless — the values shown are not the accusation count but the lower-left N entries of the evidence pair-cell matrix at slots `[0..N]` (which happen to be the detective-0-indexed row's first N cells). Per-detective `total_witnessed` is similarly aliased.

**Generated code reference:**
```rust
// detective_investigation/runtime_core.rs:427-429
pub view_storage_anchor_buf: wgpu::Buffer,
pub view_storage_ids_buf: wgpu::Buffer,
pub view_storage_primary_buf: wgpu::Buffer,  // shared by 3 views
```
And in step():
```rust
schedule::DispatchOp::Kernel(KernelId::FoldEvidence) => {
    let bindings = fold_evidence::FoldEvidenceBindings {
        view_storage_primary: &self.view_storage_primary_buf,  // alias
        ...
    };
}
schedule::DispatchOp::Kernel(KernelId::FoldAccusationCount) => {
    let bindings = fold_accusation_count::FoldAccusationCountBindings {
        view_storage_primary: &self.view_storage_primary_buf,  // alias
        ...
    };
}
```

**Surface to fix:** `crates/dsl_compiler/src/build_helper.rs::synthesize_generated_runtime_struct` needs to emit one `view_storage_<view_name>_primary_buf` field per `@materialized` view, and the `step()` per-kernel binding code needs to wire each fold kernel to its own dedicated buffer. The `compose_view_storage_prelude` helper at `crates/dsl_compiler/src/cg/emit/program.rs:1099` already names per-view storage as `view_storage_<view_name>_primary` — the naming convention exists, the host-side allocator just doesn't honor it yet.

**Hint for the fix:** the `view_storage_primary` literal binding name should be rewritten to `view_storage_<view_name>_primary` at the kernel-spec emission stage (just like the `view_<id>_get` helper rewrite). Build helper then iterates per-view + emits one buf field each.

### Gap 3 — Indirect dispatch arm is documented-as-blocked

```rust
// detective_investigation/runtime_core.rs:1802-1807
// DispatchOp::Indirect / DispatchOp::FixedPoint
// are intentionally unhandled. See
// `synthesize_generated_runtime_struct` source comment
// for the four-gap blocker (kernel-emit indirect entry,
// schedule order, per-consumer cfg, inject coordination).
_ => {}
```

The schedule for this fixture has 4 Indirect dispatches:
- `physics_ApplyDamageFromChronicle` (chronicle consumer that re-emits Damaged → Accused)
- `physics_verb_chronicle_Investigate`
- `physics_verb_chronicle_Observe`
- `physics_verb_chronicle_Accuse`

None of them fire. The verb scoring kernel SELECTS Accuse (its score is 100 + `cooldown_next_ready_tick(target)` which dominates Observe=60 and Investigate=50 for the guilty suspects), but the chronicle write kernel that translates the scoring decision into an `EffectDamageApplied` ring entry never dispatches. Consequently no `Accused` events are ever emitted, the `accusation_count` view stays at 0, and the per-detective accuracy in the pin is undefined.

This is a documented gap (commit `353527e6` titled "docs(build_helper): document four-gap blocker for Indirect dispatch arm"). Closing it unlocks every fixture with a verb-cascade chronicle re-emit pattern, including duel_abilities, hill_raid, and now this fixture.

### Gap 4 — Schedule order: producer kernels run AFTER consumer folds within a tick

The per-tick kernel order is:
1. fused_mask_verb_Investigate (mask write)
2. fold_evidence
3. fold_accusation_count
4. fold_total_witnessed
5. **physics_ObserveAndAccrue** ← producer of Witnessed
6. (Indirect: chronicle consumers — currently blocked, see Gap 3)
7. scoring
8. spatial_build_hash_* ← spatial grid built LAST

Tick 0:
- tail cleared at start
- folds read uninitialized ring slots (`kind == 0u`, no Witnessed match)
- ObserveAndAccrue emits N events; tail = N
- spatial grid built (but already useless this tick)

Tick 1:
- tail cleared again
- folds read ring slots that STILL contain tick 0's writes (the ring isn't cleared, only the tail)
- the `event_count` cfg uniform is `agent_count` (= 18), so the fold dispatches 18 invocations
- IF tick 0's emit count was ≥ 1, the fold should pick those events up

Combined with Gap 1 (no spatial neighbors found → no events emitted), the fold never sees a Witnessed write, so the pair-keyed evidence view stays empty.

**Recommended fix:** the schedule generation pass at `crates/dsl_compiler/src/cg/schedule/` should toposort by event-flow dependency (producer → consumer) rather than alphabetical-by-name. The hand-written runtime crates that hill_raid + tom_probe were ported from explicitly ran spatial builds first, then physics, then folds; the auto-generated schedule reverses that order.

### Gap 5 — View-call from scoring (deferred — not exercised in this fixture)

The original Accuse design intended to score on `evidence(self, target)` (a pair-keyed view-call from inside the scoring expression). I substituted with `agents.cooldown_next_ready_tick(target)` (the host-stamped guilty-bit channel) as a workaround — the surrogate is enough to exercise the pair-field score + chronicle write path but doesn't probe the view-call-from-scoring surface.

**Predicted blocker:** `cg/emit/program.rs::compose_view_storage_prelude` only emits `view_<id>_get(idx: u32) -> f32` helpers (single-arg, scalar f32 result). Pair-keyed views need `view_<id>_get(observer: u32, subject: u32) -> u32`. The `pair_keyed_view_present` flag exists at the build_helper level but isn't threaded into the prelude composer. Documented for follow-up.

### Gap 6 — Action-id not recoverable from `EffectDamageApplied` chronicle records

`apply_ability` writes an `EffectDamageApplied` (kind 26) record per dispatched effect, but the record's payload is `(actor, target, amount)` — no AbilityId field. So the `ApplyDamageFromChronicle` rule can't tell whether the source was Accuse / Investigate / Observe and re-emits Accused unconditionally. This over-counts accusations by 3× (one per verb).

The pin compensates by reporting the raw count and noting the over-count factor. A real fix lands an extra slot in the chronicle record (slot 5 = `ability_id: u32`) and threads it through `cg/emit/wgsl_body.rs`'s dispatcher arm. Documented for follow-up.

## Reproducer

```bash
cd /home/ricky/Projects/game/.claude/worktrees/agent-a8722ac2a415192a7
RUST_MIN_STACK=33554432 cargo test -p sims --release --test detective_investigation_pin -- --nocapture
```

## Files

- `assets/sim/detective_investigation.sim` — the fixture
- `assets/ability_test/detective_investigation/{Accuse,Investigate,Observe}.ability` — the ability corpus
- `crates/sims/tests/detective_investigation_pin.rs` — the pin
- `crates/sims/build.rs` — added `"detective_investigation"` to the migrated-fixtures allow-list

## Status

Adversarial fixture v1: structural pins green (no panic, alive intact, view storage sized correctly), behavioural pins soft (zero events fire because of Gap 1; verb chronicles silent because of Gap 3). Closing Gap 1 alone should make the pin's accuracy numbers meaningful even without Gap 3 (the evidence view fold path doesn't depend on the chronicle dispatch — it folds the producer's Witnessed events directly).
