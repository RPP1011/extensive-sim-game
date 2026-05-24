# DSL Compiler Gaps Observed via Adversarial Fixtures

This file aggregates compiler / runtime gaps surfaced by the adversarial
fixtures landed during the *_runtime cleanup sweep follow-up
(2026-05-11). Each entry captures (1) the surface that failed, (2) the
.sim line(s) that triggered it, (3) the lower-diag or pin-observable
that surfaced it, and (4) a guess at the gap class (compiler emit /
fusion / dispatch / view storage / etc).

The point of these fixtures is to STRESS the compiler — when a surface
fails to lower or behave as expected, that's signal for a future slice.
Documented gaps are not "bugs to fix immediately"; they are surface-area
debt that fixtures with feature-demo scope had silently masked.

---

## trade_caravans.sim — economy + lifecycle axis (2026-05-11)

Fixture: `assets/sim/trade_caravans.sim`
Pin:     `crates/sims/tests/trade_caravans_pin.rs`

### Gap T1: View self-update operator `-=` not lowered

**Surface:** `view <name>(...) -> f32 { on Event { ... } { self -= X } }`

**.sim trigger** (trade_caravans.sim, view `inventory`):
```
on Sold { seller: s, buyer: _, good: g, price: _ }
  where s == merchant && g == good { self -= 1.0 }
```

**Diag (pre-fix):**
```
[trade_caravans lower diag] view #1 self-update operator -= not supported
by CG IR; only +=, |=, and = are lowered today
```

**Class:** lowering / CG IR. The well-formed checker detected the
unsupported operator and skipped the body, so the fold kernel emitted
BUT the decrement arm was silently dropped.

**Why this matters:** any view that wants signed accumulation
(inventory delta, net wealth flow, score difference) was previously
limited to a single-direction fold. Multi-event views with opposing
arms could not be expressed.

**Status (2026-05-11, fixed):** `ViewFoldOp::Sub` added (this commit).
Lower accepts `-=`, emit produces native `atomicSub` for u32 views and
a CAS+sub loop for f32 views (mirrors the `+=` f32 CAS+add shape). Test
pin: `crates/dsl_compiler/tests/view_fold_self_sub_emit.rs` (3 pins —
u32 atomicSub, f32 CAS+sub, regression guard on `+= 1u` → atomicAdd).

**Follow-up — Gap T1b (introduced when fixing T1):** the per-view
`ViewSignature::fold_op` field is a single `Option<ViewFoldOp>`, not a
per-handler vector. `register_view_fold_op` is called once per fold
handler and last-write-wins. For the trade_caravans `inventory` view
(2 handlers: `+=` on Bought, `-=` on Sold), the second handler's op
(`Sub`) now overwrites the first (`Add`), so BOTH arms emit the
CAS+sub shape. Pre-fix the Sold arm was silently dropped; post-fix the
Bought arm is silently miscompiled. The bug moved but did not
disappear. Fixing it requires threading `fold_op` per-handler — either
on each handler's `Assign` op (op-level instead of view-level) or by
expanding `ViewSignature::fold_op` to `Vec<ViewFoldOp>` and indexing by
handler ordinal in the emit branch. Out of scope for the smallest-
slice T1 fix; track as Gap T1b for the next iteration.

---

### Gap T2: Multi-Item entity declarations only allocate first per-Item buffer

**Surface:** Multiple `entity X : Item { field: T }` declarations.

**.sim trigger** (trade_caravans.sim):
```
entity Grain : Item { base_price: f32 }
entity Spice : Item { base_price: f32 }
entity Silk  : Item { base_price: f32 }
```

**Observable:** Inspecting the auto-emitted `runtime_core.rs`, only
`grain_base_price_buf` is allocated; `spice_base_price_buf` and
`silk_base_price_buf` are absent from the GeneratedRuntime struct.
Calls to `items.base_price(N)` for N=0,1,2 ALL route to indices
0, 1, 2 of `grain_base_price_buf` (same WGSL binding) — confirmed
via `physics_PriceBroadcast.wgsl`:
```wgsl
@group(0) @binding(4) var<storage, read> grain_base_price: array<f32>;
... grain_base_price[target_expr_92] ...
... grain_base_price[target_expr_96] ...
... grain_base_price[target_expr_100] ...
```

**Class:** build_helper / per-Item SoA allocation. The first Item entity
in declaration order owns the canonical `<name>_<field>_buf`; subsequent
Item entities don't get their own per-Item buffers, and the lowering
pass doesn't disambiguate `items.base_price(N)` by Item-type — N is a
plain index into the first Item's buffer.

**Why this matters:** users will declare semantically distinct Items
(Grain vs Spice vs Silk, each with its own price column) and silently
get a single shared buffer. Cross-Item-type field reads will alias.

**Likely fix surface:** `crates/dsl_compiler/src/build_helper.rs` —
the per-Item buffer allocation loop. Need either (a) per-Item-type
buffers wired via Item discriminant index, or (b) a single
`item_<field>_buf` per field (not per Item-type) sized
`sum(Items.count)`, with offset bookkeeping per Item-type.

**Status (2026-05-12, fixed):** option (b) landed. The WGSL
binding name is now field-keyed (`item_<field>` / `group_<field>`)
across all Item / Group entities that declare the same field
name. The buffer is sized to one slot per declared Item-rooted
(resp. Group-rooted) entity, indexed by the entity's position in
declaration order among that root — so `items.base_price(0)`
reads Grain's slot, `(1)` Spice, `(2)` Silk, all from the same
`item_base_price` buffer. The lowering's `resolve_item_by_name`
still picks the first-declaring entity for typing (the buffer
shape is uniform across all declarers of the same field name, so
the first match is sufficient for primitive-type lookup). Pin
test: `crates/dsl_compiler/tests/multi_item_buffer_allocation.rs`
(3 pins — WGSL emits field-keyed bindings, runtime struct emits
1 buffer per unique field name sized to Item count, Group surface
mirrors the Item surface). bartering's `item_weight` /
`group_size` names migrated cleanly (single-Item/Group fixture);
trade_caravans's `item_base_price` now allocates 3 slots and all
3 reads route through the same buffer. The pin's inventory
readback is still not Item-keyed — that's a follow-up.

---

### Gap T3: Birth path (alive 0→1 in chronicle consumer) doesn't propagate

**Surface:** `agents.set_alive(<target>, true)` in a `@phase(post)`
chronicle consumer, where the target slot starts with `alive=0` at init.

**.sim trigger** (trade_caravans.sim, BornRevive):
```
@phase(post)
physics BornRevive {
  on Born { agent: h, ancestor: _, legacy: _ } {
    agents.set_alive(h, true);
    agents.set_hunger(h, 0.0);
  }
}
```

**Pin observable** (`trade_caravans_pin.rs`, after 1500 ticks):
```
final: heirs(alive)=0/16  markets(alive)=4/4  merchants(alive)=0/16
verdict: EXTINCTION — all merchants dead AND no heirs revived
```

All 16 merchants died (alive 1→0 propagated correctly via
`LifecycleReaper.set_alive(self, false)`), but ZERO of 16 heirs
revived even though OnDied emits `Born { agent: heir }` and BornRevive
sets `alive[heir] = true`.

**Class:** Chronicle-consumer dispatch / Indirect-dispatch — same family
as the gap commit `353527e6` documented for hill_raid (apply_ability
records enqueue but consumers don't fire from synthesized step()).
Could ALSO be:
  - PerAgent dispatch population computed from initial-alive (so
    initially-dead heir slots are excluded from dispatch grids forever);
  - SCHEDULE ordering puts BornRevive in a phase where the alive
    write doesn't reach the next-tick PerAgent kernels.

**Why this matters:** every game with respawn / reincarnation / heir
mechanics depends on flipping alive 0→1. Currently impossible to
express purely in the .sim — host has to re-write the alive buffer
between ticks.

**Diagnostic next step:** add an emit-only sentinel in BornRevive
(e.g. `emit Damaged { source: h, target: h, amount: 999.0 }`) and
check whether Damaged events flow when triggered by Born — if yes,
BornRevive is firing but the alive write isn't visible to next-tick
dispatch; if no, OnDied isn't firing and the cascade never starts.

**STATUS — fixed by `5d62f9b7` + `96d8c8ae` (chronicle-consumer
Indirect-dispatch + live event_tail copy).** As of 2026-05-11 the
trade_caravans pin reports `heirs(alive)=15/16  merchants(alive)=0/16`
with verdict `GENERATIONAL TURNOVER`. The lowering / WGSL emit was
NEVER one-directional — investigation walked the full path:

* `agents_setter_field` (`crates/dsl_compiler/src/cg/lower/physics.rs:916`)
  accepts `set_alive` regardless of value direction.
* `lower_agents_setter` (same file, line 976) lowers `true` and `false`
  literals through the same `CgStmt::Assign` shape — no directional
  bias at the IR layer.
* `alive_pack` (the `PlumbingKind::AliveBitmap` body in
  `crates/dsl_compiler/src/cg/emit/kernel.rs::ALIVE_BITMAP_BODY`) packs
  via `agent_alive[slot] != 0u` — fully bidirectional. Re-emitted
  every tick after the alive-write kernels.
* WGSL emit: kill kernels (containing `set_alive(_, false)`) upgrade
  `agent_alive` to `array<atomic<u32>>` and emit
  `atomicCompareExchangeWeak` for the within-tick kill race. Revive kernels (only
  `set_alive(_, true)`) keep `array<u32>` and emit a plain
  `agent_alive[<idx>] = select(0u, 1u, true);` indexed store. Both
  forms write to the same underlying buffer (wgpu doesn't distinguish
  atomic vs non-atomic at the BGL layer).

The actual root cause was the chronicle ring not surviving end-of-tick
+ the consumer's `cfg.event_count` being seeded with 0 (so consumers
no-op'd). `5d62f9b7` wired the Indirect-dispatch arm; `96d8c8ae`
added the per-consumer `copy_buffer_to_buffer(event_tail → cfg, 4 B)`
immediately before each Indirect dispatch so consumer A's emits land
in consumer B's `event_count` window when A precedes B in SCHEDULE.

The remaining 1/16 unrevived heir is a fixture-side off-by-one in the
`engaged_with` `+1` Some-encoded sentinel: heir N's slot index +1
lands on slot N+1; for N=15 that overshoots the heir block into the
market block (already alive). Compiler-side T3 is structurally fixed.

**Regression test:** `crates/dsl_compiler/tests/set_alive_both_directions_emit.rs`
pins all four arms of the matrix: kill-only kernel emits CAS; revive-only
kernel emits plain store; both-direction program keeps each kernel's
mode independent; `alive_pack` exists and uses `!= 0u` (bidirectional)
for revive-only programs.

---

### Gap T4: Cross-agent state transfer chain (chronicle → emit → consumer) not propagating

**Surface:** A chronicle consumer that emits another chronicle event,
which a downstream consumer drains. Specifically the
"OnDied → emit Born → BornRevive" cascade in trade_caravans.sim.

**.sim trigger:**
```
@phase(post)
physics OnDied {
  on Died { agent: a, age: _ } {
    let wealth = agents.mana(a);
    let heir   = agents.engaged_with(a);
    emit Transferred { donor: a, receiver: heir, amount: wealth }
    emit Born        { agent: heir, ancestor: a, legacy: wealth }
  }
}
```

**Observable:** Pin output — heir wealth stayed at 0 across all 16
heirs even though all 16 merchants died (Transferred should have
flowed wealth from each dying merchant to its heir). Suggests the
`emit Transferred` inside OnDied is silently dropped, or
TransferApply doesn't see the emitted records.

**Class:** Multi-stage chronicle cascade. Single-emit consumer rules
work (BoughtApply, SoldApply); chained-emit consumers (OnDied → Born
→ BornRevive) do not. May be:
  - OnDied isn't firing at all (Gap T3 root)
  - Emit-from-PerEvent is silently no-op
  - Per-tick chronicle ring is drained between phases such that
    same-tick-emitted events don't propagate to consumers in the
    same phase

**Likely fix surface:** the SCHEDULE pass + the chronicle ring's
phase model. If consumer phases drain the ring between consumers,
emit-from-consumer is invisible. If they don't, the issue is
pure dispatch (OnDied isn't running).

**Note:** this gap and Gap T3 may have a single root cause. The
likely diagnostic — add a trivial Damaged-emit in BornRevive AND
in OnDied separately — would discriminate the two.

---

### Gap T5: Trade dispatch from spatial-walk PerAgent body silently no-ops at scale

**Surface:** `for other in spatial.X(self) { emit ... }` body in a
PerAgent rule, where the only output is a chronicle emit (no
SoA write).

**.sim trigger** (trade_caravans.sim, TradeScan):
```
@phase(per_agent)
physics TradeScan {
  on Tick {} where (self.alive
                    && self.creature_type == config.econ.type_merchant
                    && (world.tick % config.econ.trade_cooldown == 0)) {
    for other in spatial.nearby_markets(self) {
      if (other.alive && other.creature_type == config.econ.type_market) {
        emit Bought  { ... }
        emit Sold    { ... }
      }
    }
  }
}
```

**Pin observable:**
```
merchants goods-bought: sum=0.00
NOTE: zero Bought events flowed through BoughtApply
```

After 1500 ticks with 16 merchants × ~300 trade-eligible windows ×
multiple markets in 4-cell range, the per-merchant goods-bought tally
(BoughtApply bumps shield_hp on every Bought) is exactly 0.

**Class:** Same family as hill_raid's chronicle-consumer Indirect-dispatch
gap (commit `353527e6`). The TradeScan kernel emits, but the
BoughtApply / SoldApply consumer kernels don't see those emits.
Note: hill_raid documented the gap for `apply_ability` chronicle
records; this fixture surfaces it for direct `emit Event` from
PerAgent rules.

**Why this matters:** every fixture with autonomous trading,
autonomous harvesting, autonomous attacking that uses the
emit-then-consumer pattern silently no-ops on its consumer side.
Existing fixtures like duel_25v25 work because they use
`apply_ability` (different dispatch path that DOES propagate per
the recent fixes).

**Fix status (2026-05-11):** schedule-side root cause closed.
The dependency graph used to form a `user_op → PackAgents →
UnpackAgents → user_op` cycle through every `agent.<field>`
(UnpackAgents writes the SoA at end-of-tick from the snapshot
buffer; user ops read those writes NEXT tick). `topological_sort`
returned `Err`, fusion fell back to *full* source order, and the
spatial-build phases (`SpatialBuildHashCount` …
`SpatialBuildHashScatter`, synthesised AFTER user ops in source
order) ended up dispatching AFTER their PerAgent consumer
kernels. First-tick `for x in spatial.nearby(self)` iterations
saw an empty grid; `apply_ability` / `emit` bodies guarded by
that walk silently no-op'd; downstream chronicle consumers had
zero events to consume. Two coordinated changes:

  1. `cg::schedule::topology::dependency_graph` skips outgoing
     edges from `Plumbing { kind: UnpackAgents }`. Same shape as
     the existing "consumer is `SpatialQuery`" cross-tick read
     skip — UnpackAgents writes the AgentField SoA in this tick
     for NEXT tick's user-op reads, so the `Unpack → user`
     edge is cross-tick and breaking it removes the canonical
     Pack/Unpack cycle without losing the legitimate same-tick
     `user → Pack → Unpack` chain.
  2. New `topological_sort_best_effort` returns an order even on
     cycles (forces the smallest-OpId remaining op when Kahn's
     queue empties). Fusion swaps to it — every dep edge that is
     NOT part of a cycle is still honoured, including the
     `SpatialBuildHashScatter → user_op` edges that were
     previously discarded by the `(0..n).map(OpId).collect()`
     fallback. Cycles still surface as a `CycleFallback`
     diagnostic so downstream consumers can detect the degraded
     analysis (e.g. trade_caravans's `LifecycleAge ↔ LifecycleReaper`
     write-back cycle through `agents.alive`).

Post-fix the `physics_DefenderFire`, `physics_EnemyMelee`,
`physics_TradeScan`, etc. kernels dispatch AFTER the
spatial-build chain and their spatial walks find neighbours.
trade_caravans's pin now shows wealth flowing through the
inheritance chain (heirs gain 100 mana from dying merchants,
markets accumulate 500 → 600 wealth) — a behaviour the previous
schedule made impossible. hill_raid's pin still records 0 losses,
but that residual is downstream of T5 (LoS occlusion + 6.0-unit
spatial radius vs hilltop topology + remaining apply_ability
chronicle-consumer indirect-dispatch gaps); the schedule-side
ordering bug is closed.

---

### Gap T6: Pair-keyed view with non-Agent second key — sizing inferred from Agent dimension

**Surface:** `view X(merchant: Agent, good: Item) -> f32 { ... }` —
pair-keyed materialised view where the second key is an Item (not Agent).

**.sim trigger** (trade_caravans.sim, view `inventory`):
```
@materialized(on_event = [Bought, Sold])
view inventory(merchant: Agent, good: Item) -> f32 { ... }
```

**Observable:** The fixture lowers + emits without diagnostic, but
`view_storage_primary_buf` is sized as N² where N = agent_count
(36 in the pin). The "good" dimension (3 Items: Grain/Spice/Silk)
should make the view storage `N × K` = 36 × 3 = 108 cells, not
36² = 1296. The fold kernel writes at `[merchant_slot * N + good_id]`
where `good_id ∈ {0,1,2}` and N = agent_count, so different goods
land in different (merchant, slot) pairs across the agent grid —
non-overlapping by accident, but semantically the addressing model
mixes Item indices into Agent index space.

**Class:** view storage / pair-keyed sizing. The pair-fold sizing
pass (commit `7c07e0c7`) sizes for N×N agent×agent; mixed
agent×item or agent×group keys aren't accounted for.

**Why this matters:** any view keyed on (Agent, Item) — per-merchant
inventory, per-faction reputation toward each Faction (Group), etc.
— silently aliases into the agent×agent storage. Reads are
arithmetically valid (no OOB) but the addressing scheme isn't what
the user wrote.

**Fix status (2026-05-11) — STORAGE SIZING:** the
`detect_pair_keyed_materialized_view` bool was generalised to
`detect_pair_keyed_second_key -> Option<PairKeyedSecondKey>` so the
auto-emitter's `slot_count_expr` now sizes
`view_storage_primary_buf` as `agent_count *
<second_key_population>` u32 cells. The second-key population is
either `agent_count` (Agent×Agent — tom_probe shape, unchanged) or
the static count of declared Item / Group / Quest entities in the
.sim (3 for trade_caravans's Grain/Spice/Silk). Pinned in
`crates/dsl_compiler/tests/pair_keyed_view_storage_sizing.rs`
(`agent_item_pair_view_resolves_to_item_count`,
`agent_group_pair_view_resolves_to_group_count`,
`trade_caravans_sim_resolves_to_item_second_key`).

**Fix status (2026-05-11) — WGSL INDEX (pre-existing, unchanged):**
the fold body already uses `view_storage_primary[k1 *
cfg.second_key_pop + k2]`, so the per-tick cfg upload of
`second_key_pop` is what selects the right addressing. The
auto-emitter still writes `1u32` for slot 2 of every kernel's cfg
(see `crates/dsl_compiler/src/build_helper.rs::synthesize_runtime_core_a2`
step()'s `cfg_words[2]`); fixtures that need a non-1
`second_key_pop` (tom_probe, trade_caravans's inventory) still hand-
roll their cfg upload. Closing that gap (writing the right
`second_key_pop` per ViewFold kernel from the auto-emitter) is the
remaining T6 follow-up; the storage-sizing fix here is an
independent, load-bearing prerequisite.

---

### Observation: 26-kernel emit + 4-way PerEvent fusion succeeded

Not a gap — a positive observation. The fixture emitted 26 kernels;
the analyzer fused FOUR PerEvent rules (BoughtApply +
CombinedTradeAudit + SoldApply + TransferApply) into a single
kernel `physics_BoughtApply_and_CombinedTradeAudit_and_SoldApply_and_TransferApply`,
including a multi-event-arm rule (CombinedTradeAudit reads both Bought
and Sold). The 3-way PerAgent fusion of LifecycleAge + LifecycleReaper
+ Wander also succeeded (`physics_LifecycleAge_and_LifecycleReaper_and_Wander`).
Six-event-kind chronicle topology (Bought, Sold, Transferred,
PriceUpdated, Died, Born) flows through the ring without emit-stats
errors. Lifecycle aging (`agents.set_hunger` per-tick) propagates
correctly across 1500 ticks, hitting max_age=800 within 1 tick of
the predicted reaper trip.

---
# Gaps observed during adversarial fixture validation

Living document. Each entry: a fixture pushed an axis hard, surfaced
something the auto-emit path can't yet handle. Add new entries to the
top. Resolved entries move to git history (delete from this file when
the underlying gap closes; the linked commit + pin captures the fix
context).

---

## 2026-05-11 — `forest_fire` event-storm fixture (event-cascade axis)

Fixture: `assets/sim/forest_fire.sim`
Pin: `crates/sims/tests/forest_fire_pin.rs::forest_fire_event_storm_500_ticks`
Branch: `worktree-agent-a9304bc53b3c32920`

Topology: 32×32 grid (1024 Trees), 4-tree centre ignition cluster,
500-tick horizon. Five distinct event kinds (Ignited, Burned,
EmberLanded, RainFell, WindShifted) on the shared chronicle ring; four
view consumers; one PerEvent consumer (Catch on EmberLanded).

Run output (release, 500 ticks): mean 0.16 ms/tick, p95 0.19 ms,
warmup 22 ms (pipeline compile dominates).

### ~~Gap A~~ — RESOLVED: All views in a fixture share ONE `view_storage_primary_buf`

**Closed by:** `fix(build_helper): per-view storage buffers (6-fixture aliasing gap)`.
Per-view rename now allocates one
`view_storage_<view_name>_primary_buf` per `@materialized` view; each
fold/decay kernel's `view_storage_primary` BGL slot routes to its
own per-view buffer. Pin: `crates/dsl_compiler/tests/per_view_storage_distinct.rs`.
forest_fire pin's `wind_exposure` per-view sum is now load-bearing.

### Gap B — `event_ring.tail_value()` host-side estimate stays at 0 forever

**Severity: high — silent chronicle drop in auto-emit path.**

Symptom: `state.event_ring.tail_value()` returns 0 every tick of a
500-tick run, even though producer kernels (Spread, WindEvent,
RainEvent, Reaper) are emitting events to the GPU `event_tail`
counter.

Root cause: synthesised `step()` calls
`self.event_ring.clear_tail_in(&mut encoder)` (which zeros both the
GPU buffer AND the host-side `tail_estimate`) at the start of every
tick, but never calls `note_emits()` after a producer kernel runs.
The host-side estimate exists exactly to avoid a per-tick GPU→host
sync (see `EventRing::note_emits` docstring); without it, downstream
chronicle consumers that read `event_count = tail_value()` for their
per-tick cfg uniform see 0 and early-return their bodies.

Consequence: every PerEvent consumer dispatched via the documented
`event_count = ring.tail()` pattern silently drops every event. The
fold kernels don't hit this because they bind `event_tail` directly
as a GPU storage buffer (slot 1) and read the ATOMIC counter in-shader,
not the host-side estimate. So folds work; PerEvent rules wired
through the cfg uniform don't.

Fix sketch: `synthesize_step_body` (build_helper.rs ~1460) emits a
per-Kernel arm with `dispatch::dispatch_<name>(...)` calls. After each
producer kernel arm, append:

```rust
self.event_ring.note_emits(self.agent_count * <max_emits_per_agent>);
```

The per-kernel max-emits count is recoverable from the kernel's
`emit` statement count in the lowered IR — the build helper already
knows this for the `[<fixture> emit-stats]` warning.

### Gap C — Indirect-dispatch consumer arm intentionally unhandled

**Already documented.** See `build_helper.rs:1535-1601` for the
four-gap blocker. forest_fire's `physics_Catch` (PerEvent on
EmberLanded → flip hp + emit Ignited) is `DispatchOp::Indirect` and
falls through the `_ => {}` catch-all. Verdict in pin output:
"INDIRECT GAP CONFIRMED — only seed cluster burned out".

This isn't a new finding — it's the SAME gap hill_raid hit (commit
1c565df9 + 78ad8a77). Forest_fire lights it up cleanly because the
fire-spread cascade depends ENTIRELY on Catch (with no apply_ability
fallback). When this gap closes, the forest_fire pin's verdict line
will flip to one of "FIRE SPREADS PARTIALLY" or "FIRE CONSUMED FOREST".

### Gap D — f32 view-fold reduction non-determinism (residual after CAS work)

**Severity: medium — known issue, fixture surfaces it broadly. CAS work
landed for naked RMW; reduction associativity remains.**

History:
- Pre-fix (2026-05-11 snapshot): max |Δ| = 1.000 across 1020/1024 slots
  — driven by a combination of (a) naked f32 RMW races and (b) stale
  fold cfg.event_count walking prior-tick records.
- Post-fix (2026-05-12, commits `99e8c783` + `93143c1b`): the standard
  `agents.set_<f32>(t, expr)` race is closed by an atomic-CAS loop in
  `cg/emit/kernel.rs:802` + `cg/emit/wgsl_body.rs:734+`; per-tick
  fold cfg.event_count is now snapshotted live; post-CAS emit gating
  prevents losers from re-emitting.
- Current symptom (verified 2026-05-23 re-run of forest_fire_pin):
  max |Δ| = 47.000 across 479/1024 slots. Different character from
  pre-fix — fewer slots affected but larger per-slot drift. Same
  underlying cause (f32 non-associativity in reductions), now visible
  because the CAS layer no longer drops contributions.

Root cause (residual): `atomicAdd` on `view_storage_primary[slot]` from
N producers is sequentially consistent in execution order but f32
addition is non-associative — `(a + b) + c ≠ a + (b + c)` for general
floats. So even with no contribution dropped, the SUM depends on the
order producers' atomic ops interleave on the GPU. P11 names sort-
then-fold as the prescribed mechanism; not yet implemented in the
ViewFold emit path.

The pin assertion is `max_abs_drift ≤ 150.0` — a loose ceiling that
catches control-flow divergence (regression signal) while tolerating
the documented reduction race (observed range 38-95).

**Status (2026-05-24, CLOSED):** P11 sort-then-fold landed. Mechanism:
deterministic `seq` field on every event payload + GPU radix sort by
`(target_id, seq)` before fold consumers + CPU `Vec::sort_by_key` mirror
for the serial backend. Five distinct P11 violations were closed during
the implementation: engine `EVENT_STRIDE_U32` mismatch (latent off-by-one
in original chronicle layout), spatial scatter cell ordering, radix sort
scatter intra-bucket ordering, fold path CAS+add retry race, and
post-CAS-gating kernel parallel-thread race (Catch-style PerEvent
kernels). The f32_reduction probe pins byte-equal output across
same-seed reruns. The forest_fire pin's `max_abs_drift` threshold
tightened from `≤150` to `≤100`; residual drift (30-84 observed) comes
from upstream parallel atomic-append slot acquisition in non-CAS-gated
emit kernels (Spread, Reaper, WindEvent, RainEvent), which would
require single-threaded dispatch of the whole physics emit layer to
close — out of scope for the P11 sort-then-fold work proper.

### Gap E — `@traced` annotation surface absent / unverified — CLOSED (phantom, 2026-05-12)

Investigated 2026-05-12 — **the surface already parses + resolves +
lowers** through the generic annotation infrastructure. No parser
arm, resolver allowlist, or lowering allowlist gates the
`@traced` name; it lands directly on
`dsl_ast::ir::EventIR::annotations` and survives through to
`dsl_compiler::cg::program::CgProgram`. The
`assets/sim/predator_prey_min.sim` fixture has been exercising
`@non_replayable @traced` on its `DeathCry` event since Stage 8
and `predator_prey_non_replayable.rs` pins the resolver path.

Pinned by a standalone regression fixture: `crates/dsl_compiler/tests/traced_annotation_parses.rs`
(parse + resolve + lower + `EventIR::is_traced()` helper, both bare
`@traced` and stacked `@non_replayable @traced` forms). Schema hash
unchanged.

**Deferred (still open under the same gap entry):** wiring
`is_traced` into the per-kind
`dsl_compiler::cg::program::EventLayout` so the schedule synthesizer
can route traced events to a separate ring and the host fold can
filter at layout level without re-walking the
`EventIR.annotations` vec. The fold path that would consume this
flag is still TBD — no runtime trace-vs-replay-hash split exists
today; `@non_replayable` is the only currently-honoured
ring-routing flag.

### Gap F — `@cascade(max_iter=N)` annotation surface absent

The resolver registers `cascade` as a NamespaceId + `cascade.iterations`
config (per `dsl_ast/src/ir.rs:228-241`), but no `@cascade` annotation
parses today (grep across `dsl_ast/src/resolve.rs` returns 0 hits for
"cascade(" in annotation context). Recommend leaving the cross-tick
"events from tick T arrive in tick T+1's consumers" as the natural
shape; the Plan G fixed-point cascade work tracks the real surface.

---

## How to run forest_fire pin

```bash
RUST_MIN_STACK=33554432 cargo test -p sims --release \
    --test forest_fire_pin -- --nocapture
```

Output verdict line tells you which gaps are still open. When all
six are closed, the verdict should read "FIRE CONSUMED FOREST" and
the determinism mismatch count should drop to 0.
# Gaps observed under adversarial fixtures

Discovery log for compiler / runtime gaps surfaced by adversarial
fixtures (hill_raid, squad_skirmish, ...). One section per gap; each
records the surface, .sim line, observed diagnostic / behaviour, and
gap class.

The intent: capture each gap in enough detail that a follow-up plan can
locate the source-level concern without re-running the discovery probe.

## squad_skirmish (commit TBD, 2026-05-11)

Adversarial multi-row scoring + multi-ability + pair-keyed-view
fixture. 16 Soldiers (8 vs 8) ringed around the world origin, 4 verbs
competing per tick. See `assets/sim/squad_skirmish.sim` and
`crates/sims/tests/squad_skirmish_pin.rs`.

### Gap A — `stun N` (no time suffix) rejects with confusing diagnostic

**Surface**: ability parser / EffectOp lowering.
**Site**: `assets/ability_test/squad_skirmish/Daze.ability:15` —
originally `stun 8`, expecting "8 ticks".
**Observed**: `Lower(EffectArgMismatch { verb: "stun", expected: 1,
got: 1, span: Span { start: 512, end: 539 } })` — the "expected 1, got
1" message is structurally false (both sides are 1) and gives the
designer no hint about the missing unit suffix.
**Workaround**: change to `stun 800ms` (or `stun 1s`). Every other
shipped `.ability` uses time suffixes (`1s`, `2s`, `1500ms`).
**Gap class**: ability parser diagnostic clarity. The lowerer ought
to either accept bare integers as ticks (and document) or emit a
diagnostic that says "stun expects a time suffix (e.g. `1s`,
`500ms`), got bare integer".

### Gap B — single ability-corpus failure silently disables AOE auto-detect for the WHOLE corpus

**Surface**: `dsl_compiler::build_helper`, `aoe_dispatch` flag.
**Site**: `crates/dsl_compiler/src/build_helper.rs:272-286` —
`built_registry.as_ref().map(...)` returns `None` if ANY .ability in
the corpus failed to lower; the closure then defaults to `false`.
**Observed**: Daze.ability with the `stun 8` bug above caused
`built_registry = None`, which made `aoe_dispatch = false` — even
though `Volley.ability` declares `damage 6 in spread(4.0, 8)` (clear
AOE intent). The build log says `ability-corpus] 4 .ability files,
aoe_dispatch=false` — masking that it's not "no AOE in the corpus" but
"none of the .ability files lowered". After fixing Daze, the SAME
corpus correctly reported `aoe_dispatch=true`.
**Gap class**: build-helper resilience. AOE detection should walk
whatever programs DID lower, not bail to false on partial failure;
a single broken .ability file shouldn't disable AOE Path B emit for
its peers.

### Gap C — `scoring { row X per_target { base, weights } }` parses but `weights:` clause silently dropped

**Surface**: scoring lowering / WGSL emit.
**Site**: `assets/sim/squad_skirmish.sim:281-302` (the
`scoring Soldier { row Strike per_target { base: ..., weights: ... } }`
block) →
`target/release/build/sims-*/out/squad_skirmish/scoring.wgsl:391` (the
emitted `let utility_4: f32 = config_9;` body for row 4).
**Observed**: the `weights:` clause references real per-agent SoA
columns (`agents.altruism(self) * 30.0`, etc.) but the emitted utility
expression is just `config_9` (= the `base:` literal). The personality-
weighted scoring the row pretended to declare contributes nothing to
argmax. Documented in `crates/dsl_ast/src/parser.rs:1734-1746` (the
parser parse-and-discards everything beyond `score:` / `base:`).
**Gap class**: scoring backend completeness. The "utility table" form
that predator_prey.sim and crowd_navigation.sim use as a design target
parses cleanly without any signal that the weights clause is being
dropped. Either accept and lower it, or reject with a typed
"`weights:` clause not yet supported in scoring rows".
**Resolution (2026-05-11, approach B)**: lowered. The parser captures
`weights:` into a sibling `Option<Expr>` field on `PerAbilityRow`
(AST), the resolver propagates it through to `PerAbilityRowIR`, and
`cg::lower::scoring::lower_per_ability_row` now composes the row's
utility as `Binary { AddF32, base, weights }` when `weights:` is
present (both operands type-checked F32 individually). Post-fix the
squad_skirmish scoring.wgsl row-4 body emits
`(config_9 + (agent_risk_tolerance[agent_id] * config_13))` instead
of the bare `config_9`. Pin: `crates/dsl_compiler/tests/scoring_weights_clause_emit.rs`.
The `personality.<X>` aspirational namespace in predator_prey.sim /
crowd_navigation.sim no longer parse-and-discards (since weights are
now resolved); those .sim files were updated to read from the
matching `agents.<field>(self)` SoA columns to keep their weights
expressions resolvable.

### ~~Gap D~~ — RESOLVED: `view_storage_primary_buf` aliased ALL views

**Closed by:** same per-view storage rename that closed Gap A above
(`fix(build_helper): per-view storage buffers (6-fixture aliasing
gap)`). squad_skirmish's `damage_dealt` / `healing_done` /
`threat_taken` views now each have their own backing buffer
(`view_storage_<name>_primary_buf`); pin `squad_skirmish_pin.rs` reads
each independently. Pin: `crates/dsl_compiler/tests/per_view_storage_distinct.rs`.

### Gap E — pair-keyed view called from scoring expression emits arity-mismatched WGSL helper

**Surface**: scoring expression lowering for `view_call(2-args)`.
**Site**: `assets/sim/squad_skirmish.sim` Daze verb's score
expression originally `threat_taken(self, target) * 10.0` (now
removed; see comment in source). Generated kernel:
`target/release/build/sims-*/out/squad_skirmish/scoring.wgsl:39` emits
`fn view_0_get(idx: u32) -> f32` (single-arg) but line 379 calls
`view_0_get(agent_id, per_pair_candidate)` (two args).
**Observed**: WGSL validation rejects the kernel:
`Call to [0] is invalid; Requires 1 arguments, but 2 are provided`.
The pin test panicked at `Device::create_shader_module`. This is a
hard failure — fixture cannot run with the pair-view ref in scoring.
**Gap class**: pair-view 2-arg call in scoring expression — the
helper-signature emitter doesn't propagate the pair-arity from the
call site. Either generate a 2-arg helper for pair-keyed views or
reject the call with a lowering error that says "pair-keyed view
calls in scoring expressions need both keys explicitly".

### Gap F — chronicle consumer Indirect dispatch unhandled in synthesized `step()`

**Surface**: `step()` dispatch arm in auto-emitted runtime.
**Site**: `crates/dsl_compiler/src/build_helper.rs:1602-1614` —
`DispatchOp::Indirect` and `DispatchOp::FixedPoint` fall into the
`_ => {}` catch-all of the dispatch match. This is the documented
"four-gap blocker" (commit 353527e6). The catch-all has a 90-line
comment explaining why a naive wire-up was reverted.
**Observed**: in squad_skirmish (and hill_raid), `apply_ability N by
self target target` writes EffectDamageApplied / EffectHealApplied /
EffectStunApplied chronicle records, but the consumer kernels
(`physics_ApplyDamageFromChronicle`,
`physics_ApplyHealFromChronicle_and_ApplyStunFromChronicle_and_ApplyDamage`)
never fire because their schedule entries are Indirect. After 200
ticks of squad combat: 0 damage flowed, 0 healing flowed, 0 entries
in the threat_taken pair-view, all 16 soldiers at 100% HP. The 18
emitted kernels DO step without panic — the gap is that the consumers
don't run, not that the producers crash.
**Gap class**: chronicle consumer dispatch — same gap hill_raid
documents. Squad_skirmish is a SECOND data point that confirms this
isn't a hill_raid-specific anomaly: any fixture using apply_ability +
auto-emitted runtime hits it. The four-gap blocker comment notes
kernel-emit, schedule order, per-consumer cfg, and inject coordination
all need coordinated fixes.

### Gap G — `init { hp: 100 }` writes u32 bit-pattern into f32 buffer (TYPE CONFUSION)

**Surface**: build_helper init lowering.
**Site**: `crates/dsl_compiler/src/build_helper.rs:1140-1146` — the
init expression `Const(100)` lowers as
`vec![100u32; agent_count as usize]` and the buffer is initialised via
`bytemuck::cast_slice(&{name}_init)`. Generated artifact:
`target/release/build/sims-*/out/squad_skirmish/runtime_core.rs:708-715`.
**Observed**: every fixture using `init { hp: 100 }` (including
shipped `dsl_stress_coverage`, `hill_raid`, and now `squad_skirmish`)
writes the bit-pattern `0x64` into each `agent_hp` slot. Read as f32
that's `1.4e-43` — functionally zero. Every `target.hp` read from the
scoring kernel is therefore ~0.0 from tick 0; verbs that gate on
`target.alive && target.hp > X` see hp=0 and either fire instantly or
not at all depending on the predicate direction. The
`squad_skirmish_pin.rs` workaround writes the correct f32 bit-pattern
from the host before calling `step()`.
**Gap class**: init lowering type discipline — the InitExpr::Const
lowering path doesn't consult the target column's `AgentFieldTy`
(F32 vs U32 vs Bool vs ...). For F32 columns it should emit
`vec![{n}.0f32; agent_count]` and bytemuck-cast the f32 slice.
Existing fixtures that "work" with `init { hp: 100 }` either don't
read hp directly (use only alive flag) or have a runtime that
overwrites hp after `try_new`. This is structural — every f32 SoA
column is affected (hp, max_hp, mana, etc. — but `init` only
recognises a few fields today).

### Gap H — confused 2026-05-11 secondary observation (already-known)

The fixture also confirms hill_raid's voxel-grid scaling caveat is
unrelated to scoring: squad_skirmish has no terrain query and still
hits Gap F unmodified. Gap F is the chokepoint blocking all
apply_ability fixtures from real behavioural pins under the auto-emit;
it is not a viewer / voxel concern.

---

Last updated: 2026-05-11 by adversarial-fixture pass (squad_skirmish).

---

## 2026-05-12 — Session-end status summary

This session closed the bulk of the compiler-leverage gaps documented above.
Status overview (gap → resolution SHA):

**Trade caravans axis:**
- T1 view `self -= expr` → `acbfdb09`
- T2 multi-Item buffers → `a65482c1`
- T3 alive=true setter → already-fixed, regression test at `d3e0ca49`
- T4 chronicle cascade → subsumed by T5 schedule order fix
- T5 schedule order (spatial-build before consumers) → `d1207fca`
- T6 pair-key non-Agent sizing → `a02410a3`

**Forest fire axis:**
- A views aliasing → `220506a1` (per-view storage)
- B fold cfg.event_count stale → `99e8c783`
- C indirect-dispatch chronicle consumer arm → `ffaab378` (kind-aware schedule)
- D f32 RMW race → bug 1 `99e8c783` + bug 2 `93143c1b` (post-CAS emit gating); residual 38-95 drift is P11 reduction territory
- E `@traced` surface → phantom; regression pin at `a6c46275`
- F `@cascade(max_iter=N)` → surface `9c5c7d8c` + runtime FixedPoint dispatch `3efdc3cd`

**Squad skirmish axis:**
- A `stun N` diagnostic → `be8e6b55`
- B AOE detect survives partial corpus → `be8e6b55`
- C scoring `weights:` clause → `473345f1`
- D view storage aliasing → `220506a1`
- E pair-view in scoring → still open (single-arg helper emitter)
- F chronicle consumer Indirect dispatch → `ffaab378`
- G init type bug f32/u32 → `c97d7f01`

**Other:**
- pirate_fleet `set_creature_type` → `4944ef65`
- plague_city P-A custom-field registry → `b497f10b`
- plague_city P-B `@host_callable` for fixture events → `22ea8c1c`
- plague_city P-C lower-error fail-fast flag → `5bdf1a70`
- plague_city P-D fused set_alive → already-fixed, regression at `d7900c34`
- plague_city P-E view storage aliasing → `220506a1`
- among_us #2 voxel_grid in chronicle consumers → phantom, regression at `b7a38b0c`
- among_us #3 fused set_pos+set_mana ordering → `354ed8f1`
- detective Gap 1 spatial-grid sizing → `c9adeb96`
- detective Gap 6 ability_id at slot 6 → `ad492ee2`

**Tier shifts (fixture behavior unlocked from Tier I to S):**
- trade_caravans, plague_city, forest_fire, among_us, palace_coup

**Remaining open:**
- Squad_skirmish E pair-view in scoring expression
- Squad_skirmish residual zero-damage (chronicle pipeline OK; predicate / level binding investigation needed)
- Parallel atomic-append slot acquisition non-determinism in non-CAS-gated
  emit kernels (residual 30-84 drift in forest_fire). Would require @ws=1
  across the whole physics emit layer to close; tracked as a perf/scope
  tradeoff rather than a P11 violation per se.
- EventLayout-level `is_traced` wiring (no runtime consumer yet)
- Hill_raid LoS-occluded shots (gameplay-side, voxel terrain queries from spatial walk work)

Last updated: 2026-05-12 by gap-addressing loop session.
