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

**Diag:**
```
[trade_caravans lower diag] view #1 self-update operator -= not supported
by CG IR; only +=, |=, and = are lowered today
```

**Class:** lowering / CG IR. The well-formed checker detects the unsupported
operator and skips the body, so the fold kernel emits BUT the decrement
arm is silently dropped. Per-merchant inventory therefore monotonically
increases on Bought (and on Sold via the additive arm — see Gap T6).

**Why this matters:** any view that wants signed accumulation
(inventory delta, net wealth flow, score difference) is currently
limited to a single-direction fold. Multi-event views with opposing
arms cannot be expressed.

**Likely fix surface:** `crates/dsl_compiler/src/cg/lower/view.rs`
(grep for `+=, |=, and =` to find the gating site). Add `SubAssign`
arm + WGSL emit support for `atomicSub` (or equivalent f32 emulation
via `atomicCompareExchangeWeak`).

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
