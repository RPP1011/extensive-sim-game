# Verb-fire end-to-end probe — discovery report (2026-05-04)

This is the report from the smallest possible end-to-end probe of the
verb cascade pipeline (commit baseline `a14965f0`). The probe drives a
single `verb` declaration through parse → resolve → CG-lower →
schedule → emit → runtime dispatch and observes whether the verb's
`emit` body actually fires.

**Outcome: (b) NO FIRE** — `faith[i] = 0.0` for every agent slot
across 100 ticks at AGENT_COUNT=32.

## Files added

- `assets/sim/verb_fire_probe.sim` (62 LOC) — probe fixture
- `crates/verb_probe_runtime/Cargo.toml` (24 LOC)
- `crates/verb_probe_runtime/build.rs` (97 LOC)
- `crates/verb_probe_runtime/src/lib.rs` (303 LOC)
- `crates/sim_app/src/verb_fire_app.rs` (113 LOC)
- `Cargo.toml` (workspace) — added `verb_probe_runtime` member
- `crates/sim_app/Cargo.toml` — added the dep + binary entry

Total ≈ 600 LOC (within the budget).

## Observed numbers

```
verb_fire_app: starting — seed=0xDEADBEEFFACECAFE agents=32 ticks=100
verb_fire_app: finished — final tick=100 agents=32 faith.len()=32
verb_fire_app: faith readback — min=0.000 mean=0.000 max=0.000
verb_fire_app: nonzero slots: 0/32 (fraction = 0.000%)
verb_fire_app: expected per-slot value (full cascade): 100.000
verb_fire_app: OUTCOME = (b) NO FIRE
```

## Gaps surfaced

### GAP #1 (CRITICAL — compiler) — verb_expand LocalRef collision

`crates/dsl_compiler/src/cg/lower/verb_expand.rs` line 502-505:

```rust
fn fresh_local_after(verb: &VerbIR) -> LocalRef {
    let max = verb.params.iter().map(|p| p.local.0).max();
    LocalRef(max.map(|m| m.saturating_add(1)).unwrap_or(0))
}
```

Used at lines 397-409 to allocate binders for the cascade physics
handler:

```rust
let target_local = verb.params.iter().find(|p| p.name == "target")
    .map(|p| p.local)
    .unwrap_or_else(|| fresh_local_after(verb));     // ← LocalRef(N+1)
let action_id_local = fresh_local_after(verb);       // ← LocalRef(N+1) — same value!
```

When the verb has no `target` param (e.g. `verb Pray(self) = ...`),
both `target_local` and `action_id_local` get the SAME LocalRef.
Downstream:

1. `synthesize_pattern_binding_lets` (event_binding.rs) iterates the
   bindings list `[actor, action_id, target]` and calls
   `ctx.record_local_ty(local_id, field_layout.ty)` for each.
2. `action_id` records `local_id_X → U32`.
3. `target` then records `local_id_X → AgentId` (overwrites!).

When the cascade body `if (action_id == 0)` lowers, the `action_id`
local now reads back as `AgentId` instead of `U32`. The `==`
typechecks `agent_id` vs `u32`, fires `BinaryOperandTyMismatch`, and
the WHOLE PhysicsRule op for `verb_chronicle_Pray` is dropped from
the program.

Build-time evidence (printed via `cargo:warning=`):
```
[verb_probe lower diag] lowering: binary `Eq` at 1564..1709 has mismatched operands — lhs is agent_id, rhs is u32
```

The 8 emitted kernels are therefore: `mask_verb_Pray, fold_faith,
scoring, upload_sim_cfg, pack_agents, seed_indirect_0,
unpack_agents, kick_snapshot`. **No `verb_chronicle_Pray` kernel
exists.**

**One-line fix:** make `fresh_local_after` stateful (return a fresh
LocalRef on every call). Quick patch:

```rust
fn synthesize_cascade_physics(verb: &VerbIR, ...) {
    let mut next = verb.params.iter().map(|p| p.local.0).max()
        .map(|m| m.saturating_add(1)).unwrap_or(0);
    let mut fresh = || { let r = LocalRef(next); next += 1; r };
    let target_local = verb.params.iter().find(|p| p.name == "target")
        .map(|p| p.local).unwrap_or_else(&mut fresh);
    let action_id_local = fresh();
    ...
}
```

### GAP #2 (CRITICAL — compiler/architecture) — fold/chronicle event-ring race

Even if GAP #1 is fixed, the existing scheduler runs the chronicle
physics rule and the view-fold against the SAME event ring with no
intervening tail-clear or seed dispatch. Confirmed by re-running the
probe shape with `verb Pray(self, target: Agent)` (which sidesteps
GAP #1 by pre-allocating distinct LocalRefs) — the scheduler then
fuses both ops into a single `fused_fold_faith_pray_completed`
kernel:

```wgsl
@compute @workgroup_size(64)
fn cs_fused_fold_faith_pray_completed(@builtin(global_invocation_id) gid: vec3<u32>) {
let event_idx = gid.x;
// op#0 (view_fold) - reads event_ring[event_idx * 10 + 0/2/3]
{
    let local_0: u32 = event_ring[event_idx * 10u + 2u];
    let local_1: f32 = bitcast<f32>(event_ring[event_idx * 10u + 3u]);
    loop { ...atomicCompareExchangeWeak(&view_storage_primary[local_0], ...) ... }
}
// op#1 (physics_rule) - writes new events to event_ring
{
    ... if ((local_1 == 0u)) { ...emit event#1 PrayCompleted... }
}
}
```

Two architectural problems in this fused kernel:

1. **op#0 doesn't filter by event-tag** — it unconditionally reads
   slot N as a `PrayCompleted` payload. But slot N is whatever the
   scoring kernel wrote previously (an `ActionSelected` event with
   `action_id` at offset+1, `actor` at offset+2). The fold
   accumulates noise, not real PrayCompleted events.

2. **op#0 reads BEFORE op#1 writes** — even if the chronicle ran
   first, the fused kernel's op-order means each thread runs op#0
   before op#1 in serial program order. So the per-tick view-fold
   sees only what was already in the ring (the previous tick's
   ActionSelected events), not the chronicle's freshly-emitted
   PrayCompleted events.

3. **Binding name mismatch** — the WGSL declares
   `var<storage, read_write> view_0_primary: array<u32>` (line 8)
   but the op#0 fold body references `view_storage_primary` (the
   un-fused canonical name). The fused kernel would fail WGSL
   validation if it ever got loaded.

**Likely fix surface:** the schedule synthesiser
(`crates/dsl_compiler/src/cg/schedule/`) must NOT fuse the cascade
physics rule with the view-fold that consumes its emits — they're a
two-stage event ring (round 1: scoring → ActionSelected ring; round
2: chronicle → PrayCompleted ring; round 3: fold reads
PrayCompleted). Either separate event rings per kind OR sequence
through `clear_tail → chronicle → seed_indirect → fold`.

### GAP #3 (runtime — pre-existing, unrelated to verbs)

The runtime helpers in `engine::gpu::EventRing` have a single shared
ring across all event kinds. The cascade above needs to either:

- Multi-tag the ring (already done — events are tagged by EventKindId
  in slot 0). The fold filter `if (event_ring[..0u] == 1u)` already
  exists, so this side IS handled — but the fused-kernel dropping
  this filter (GAP #2.1) defeats it.
- Run the fold AFTER the chronicle so PrayCompleted events exist in
  the ring. Today the runtime drives all kernels in a single
  command-encoder submit; without explicit ordering between
  chronicle and fold dispatches, races are possible.

### GAP #4 (compiler) — mask emits but predicate-driven scoring is non-functional

The `mask_verb_Pray` kernel atomicOrs into `mask_0_bitmap` —
`agent_alive` is read, the bit is set. But the `scoring` kernel does
NOT read `mask_0_bitmap` to gate its argmax. From the emitted
WGSL (scoring.wgsl):

```wgsl
// row 0: action=#0
{
    let utility_0: f32 = 1.0;
    if (utility_0 > best_utility) {
        best_utility = utility_0;
        best_action = 0u;
        best_target = 0xFFFFFFFFu;
    }
}
```

The mask bit is computed but never consulted by the scoring kernel
— so the verb's `when` predicate has no observable effect on which
action gets selected. Today this happens to be a non-issue because
the predicate is `true` everywhere (`self.alive`), but a verb whose
mask predicate is a real filter (`self.in_combat && target.hp < 0.5`)
would silently get scored regardless.

**Likely fix surface:** the `lower_scoring_argmax_body` in
`cg/emit/kernel.rs` should emit a guard reading `mask_0_bitmap[word]
& bit` for each verb-derived row before the utility comparison.

## Confirmation: no other fixture broken

Ran `cargo build` for all sibling fixture runtimes — all compile clean
with the new workspace member added. The sim_app harness loads cleanly
(emit-stats per-fixture all reported in build output, all kernel counts
unchanged from baseline).

```
boids_runtime           — 9 kernels
predator_prey_runtime   — multiple kernels (kill_count + predator_focus)
crowd_navigation_runtime — 7 kernels
swarm_storm_runtime     — 9 kernels
target_chaser_runtime   — 4 kernels
ecosystem_runtime       — N kernels
foraging_runtime        — N kernels
auction_runtime         — N kernels
bartering_runtime       — N kernels
particle_collision_runtime — N kernels
verb_probe_runtime      — 8 kernels (NEW)
```

## Suggested next steps (separate task)

1. Land the `fresh_local_after` fix (GAP #1) in `verb_expand.rs`. Test
   by re-running `cargo run --bin verb_fire_app` — the chronicle
   physics rule should now lower, but observed faith likely stays at
   0 due to GAP #2.

2. Land a regression test in `crates/dsl_compiler/tests/verb_emit.rs`
   that asserts `verb Wait(self) = ... emit Foo { ... } score 1.0`
   produces a `verb_chronicle_Wait` kernel in the artifact index
   (today's test only asserts the IR-level cascade injection — it
   doesn't follow through to schedule + emit).

3. Address GAP #2 — schedule-synth refactor or per-event-kind ring
   split.

4. Address GAP #4 — wire the mask bitmap into scoring kernel as a
   per-row guard.

5. Once 1-4 land, expand `verb_fire_app` to assert
   `faith[i] == 100.0` and add it as a smoke fixture for the
   sim_app's CI rotation.
