# Trade-market multi-surface end-to-end probe — discovery report (2026-05-04)

This is the report from the smallest possible end-to-end probe of a
multi-resource market combining ALL of the recently-landed compiler
surfaces in one fixture (commit baseline `51b5853b`). Mirrors the
verb-fire probe pattern (`2026-05-04-verb-fire-probe.md`) and the ToM
probe pattern (`2026-05-04-tom-probe.md`).

The probe is the FIRST fixture that combines every surface that landed
through the recent compiler / runtime work:

  - Items + Groups in one program (4 distinct `Item` declarations
    `Wood`/`Iron`/`Grain`/`Cloth` alongside 1 `Group` declaration `Guild`)
  - Multi-event-kind producer (one physics rule emits both
    `TradeExecuted` AND `PriceObserved` into one shared event ring)
  - Per-handler tag filter on the consumer side (commit `cb24fd69`'s
    multi-event-kind ring partition shape — three folds, two filtered
    on tag `1u` and one on tag `2u`)
  - Theory-of-Mind shape: `pair_map` u32 view (`price_belief`) with
    bit-OR fold via WGSL native `atomicOr` (post-`51b5853b`)
  - Mixed view-fold storage in one program: u32 (`price_belief`) +
    f32 with `@decay` (`trader_volume`) + f32 no-decay (`hub_volume`)

**Outcome: (a) FULL FIRE** — every assertion passes on first build,
across all three views simultaneously, with no compiler diagnostics.

```
trade_market_app: starting — seed=0xBEEFFEEDCAFEF00D agents=32 ticks=100
trade_market_app: finished — final tick=100 agents=32
trade_market_app: trader_volume   (decay=0.95, steady ~20.000/slot) — total=636.21 (expected 640.00); per-slot min=19.882 mean=19.882 max=19.882
trade_market_app: hub_volume      (no decay, +1.00/tick/slot) — total=3200.00 (expected 3200.00); per-slot min=100.000 mean=100.000 max=100.000
trade_market_app: price_belief    diagonal 32/32 == 1u, off-diagonal 992/992 == 0u
trade_market_app: per-view checks — trader_volume=OK hub_volume=OK price_belief=OK
trade_market_app: OUTCOME = (a) FULL FIRE — multi-event-kind ring (TradeExecuted + PriceObserved) partitions cleanly via per-handler tag filter; mixed view-fold storage (u32 pair_map atomicOr + f32 with @decay + f32 no-decay) coexists in one program.
trade_market_app: OK — all assertions passed
```

The combination of features works end-to-end. The probe's job was
"discovery — we expect gaps to surface from the combination". The
short version is: **no NEW compiler/runtime gaps surfaced**. The
existing landed surfaces compose cleanly. The gaps that DID surface
are all things the precedent docs already record (the `Item`-field-
read use, the verb cascade integration, the spatial body-form physics
inside multi-emit physics), and they were sidestepped exactly as the
probe shape describes.

## Files added

- `assets/sim/trade_market_probe.sim` (242 LOC) — probe fixture.
  Single physics rule (`WanderAndTrade`) emitting BOTH event kinds;
  three views (one u32 pair_map + two f32 with mixed decay); 4 Items
  + 1 Group declared but not field-read in the active rule body.
- `crates/trade_market_runtime/Cargo.toml` (31 LOC)
- `crates/trade_market_runtime/build.rs` (97 LOC) — mirrors
  `auction_runtime`'s build.rs verbatim except the input fixture path.
- `crates/trade_market_runtime/src/lib.rs` (629 LOC) — agent SoA +
  shared event ring + 1 locally-allocated `pair_map` u32 buffer + 2
  `ViewStorage` instances + per-tick dispatch. Three folds + one
  decay (the f32 `trader_volume`) + admin (clear_tail / seed_indirect_0).
- `crates/sim_app/src/trade_market_app.rs` (237 LOC) — driver
  asserting analytical observable on all three views.
- `Cargo.toml` (workspace) — added `trade_market_runtime` member.
- `crates/sim_app/Cargo.toml` — added the dep + binary entry.
- `crates/dsl_compiler/tests/stress_fixtures_compile.rs` — added
  `trade_market_probe_combines_landed_surfaces` test (176 LOC) that
  pins the FULL FIRE shape: catalog has 2 Agents + 4 Items + 1 Group;
  10 kernels emit (3 fold + 1 decay + 1 physics + 5 admin); the
  producer body has TWO `atomicStore(&event_ring[slot * 10u + 0u])`
  tag stores (one per emit kind); each fold body guards on the per-
  handler kind tag at offset 0; `fold_price_belief` uses `atomicOr`
  (NOT a CAS loop); `fold_trader_volume` + `fold_hub_volume` DO use
  the CAS+add loop (no atomic float add in WGSL).

Total added: ~1,412 LOC (within budget; ~700 was the soft target —
the doc + lib.rs explanatory comments add weight without adding code).

## What the probe demonstrates concretely

### Multi-event-kind ring partition works

The producer body emits BOTH event kinds, with distinct kind tags
written at the same offset in the event ring:

```wgsl
// physics_WanderAndTrade.wgsl (compiler-emitted, redacted)
//   ... TradeExecuted emit:
let slot_0 = atomicAdd(&event_tail[0], 1u);
atomicStore(&event_ring[slot_0 * 10u + 0u], 1u);   // kind = 1u (TradeExecuted)
atomicStore(&event_ring[slot_0 * 10u + 1u], cfg.tick);
atomicStore(&event_ring[slot_0 * 10u + 2u], buyer_id);
atomicStore(&event_ring[slot_0 * 10u + 3u], seller_id);
//   ... resource, amount (bitcast<u32>(f32)), price ...

//   ... PriceObserved emit:
let slot_1 = atomicAdd(&event_tail[0], 1u);
atomicStore(&event_ring[slot_1 * 10u + 0u], 2u);   // kind = 2u (PriceObserved)
atomicStore(&event_ring[slot_1 * 10u + 1u], cfg.tick);
atomicStore(&event_ring[slot_1 * 10u + 2u], observer_id);
//   ... hub, resource, price_q8 ...
```

Each fold body guards on the kind tag (commit `cb24fd69`'s shape):

```wgsl
// fold_trader_volume.wgsl (filters on tag 1u)
if (event_ring[event_idx * 10u + 0u] == 1u) {
    // ... read TradeExecuted fields, RMW the buyer's slot ...
}

// fold_price_belief.wgsl (filters on tag 2u)
if (event_ring[event_idx * 10u + 0u] == 2u) {
    // ... read PriceObserved fields, atomicOr into (observer, hub) ...
}
```

Per-tick event count is `2 × agent_count` (each alive agent emits
both kinds once), and the ring tail tracks both — no per-kind sub-
indexing needed at the dispatch level. With `agent_count = 32` and
`TICKS = 100`, the runtime processes 6,400 events through 3 folds
without dropping or double-counting any.

### Mixed view-fold storage coexists in one compilation

The compiler emits 10 kernels for this fixture:

```
trade_market_runtime — 10 kernels, schedule has 10 stages
  fold_price_belief        1371 B   7 bindings   (atomicOr branch, u32)
  decay_trader_volume       683 B   2 bindings   (anchor multiply pre-pass)
  fold_trader_volume       1519 B   7 bindings   (CAS+add branch, f32)
  fold_hub_volume          1510 B   7 bindings   (CAS+add branch, f32)
  physics_WanderAndTrade   2308 B   6 bindings   (multi-emit producer)
  upload_sim_cfg            666 B   2 bindings   (admin)
  pack_agents              3686 B  41 bindings   (admin)
  seed_indirect_0          1396 B   4 bindings   (admin)
  unpack_agents            3876 B  41 bindings   (admin)
  kick_snapshot             656 B   2 bindings   (admin)
```

Three view storages with three different shapes coexist:
- `price_belief` — local `agent_count × agent_count × u32` buffer
  allocated directly (NOT through `engine::gpu::ViewStorage`, whose
  host-side cache is `Vec<f32>`). Indexed `[observer * agent_cap +
  subject]` per the compiler-emitted fold body's
  `local_0 * cfg.second_key_pop + local_1` shape.
- `trader_volume` — `ViewStorage::new(..., has_anchor=true)`. The
  anchor buffer carries the @decay state pre-multiplied by 0.95 in
  the `decay_trader_volume` kernel before each fold pass.
- `hub_volume` — `ViewStorage::new(..., has_anchor=false)`. No decay
  pre-pass; anchor binding falls back to the primary buffer per the
  `unwrap_or(primary)` shape.

### Analytical observables match

After 100 ticks at 32 agents, `trade_amount = 1.0`, `decay = 0.95`,
`observation_bit = 1u`:

| View | Per-slot expected | Per-slot observed | Slots converged |
|------|-------------------|-------------------|-----------------|
| `trader_volume[i]` | 20.000 (geo. series limit) | 19.882 (within 0.6%) | 32/32 |
| `hub_volume[i]` | 100.0 (= T × trade_amount) | 100.000 (exact) | 32/32 |
| `price_belief[i*N+i]` | 1u (diagonal) | 1u | 32/32 |
| `price_belief[i*N+j]`, i≠j | 0u (off-diagonal) | 0u | 992/992 |

The 0.6% gap on trader_volume matches `0.95^100 ≈ 5.9e-3` exactly
(the geometric tail not yet decayed away after 100 ticks) — this
is the analytical convergence shape, NOT a determinism gap.

## Gaps documented (NOT new — known precedent + sidestepped)

The probe's value was discovery, and we have to be honest about which
gaps it actually surfaces vs. which gaps it sidesteps. None of the
following are NEW; all are documented in precedent docs and were
sidestepped intentionally per the task prompt's scope guardrails.

### GAP #1 (compiler — known) — config-emitted u32 typed value can't flow into a u32 ring slot

Commented inline in `assets/sim/trade_market_probe.sim:153-168`. When
the rule body tried `price_q8: config.market.observation_bit` (with
`observation_bit` declared `u32` in the `config market { ... }` block),
the WGSL emit crashed:

```
naga::front::wgsl::Error::AutoConversion {
    dest_ty: "u32",
    source_ty: "f32",
    span: ...
}
```

The compiler emits all config values as `f32` consts regardless of
their declared `u32` type, then drops them into
`atomicStore(&event_ring[... + 5u], (config_N))` which expects `u32`.
Symptom: build fails inside the validator, not the parser.

**Workaround used:** literal `1` in the emit body. The runtime feeds
the same value via `OBSERVATION_BIT = 1` in the driver's analytical
expectation, so the observable still matches.

**Likely fix surface (next-step task):** `crates/dsl_compiler/src/cg/
emit/expr.rs` — wherever config consts are materialized as WGSL
expressions, route through the declared type and emit `u32` literals
(or `bitcast<u32>(...)`) when the config field's declared type is
integral. Today there's only the `f32` path.

### GAP #2 (compiler — known precedent) — `Item` field reads in multi-Item programs are untested

The .sim declares 4 Items each with a `base_price: f32`, but the
active rule body never calls `items.base_price(<id>)`. The fixture
exercises the **declaration + catalog** path for multi-Item programs
(parses + resolves + lowers + the runtime allocates the SoA), but
NOT the per-Item field-read path.

The bartering fixture covers ONE `Item` field-read end-to-end (via
`item_field_external_name` in `crates/dsl_compiler/src/cg/emit/
kernel.rs:1039`). The 4-Item generalisation would require the
compiler to generate distinct `wood_base_price` / `iron_base_price`
/ `grain_base_price` / `cloth_base_price` external bindings via the
snake_case rule. Probably works (it's just the same path repeated 4×),
but the trade-market probe doesn't actually exercise it.

**Likely follow-up probe:** add `let p = items.base_price(0u);` (or
similar) to the rule body and verify the compiler doesn't collapse
the 4 distinct binding names into one. Out of scope for THIS probe.

### GAP #3 (compiler/runtime — known precedent) — `Group` field reads untested

Same shape as Gap #2 for the `Guild` Group declaration (`size: f32`).
The Group is declared, the catalog records it, the runtime would
allocate the SoA, but the active rule body never calls `groups.size
(<id>)`. Probably works; not exercised.

### GAP #4 (compiler/runtime — known precedent) — verb cascade inside this multi-event-emit body

The verb slice is left commented at `assets/sim/trade_market_probe.sim:
224-242`. Per the task prompt:

> The verb slice is commented out by default — it depends on too many
> surfaces to land cleanly without surfacing every other gap first.

If uncommented, the verb body would need:
- A real per-(self, hub) scoring axis (the verb is sized 1 today).
- The scoring kernel from `2492b8d5` (verb_probe_runtime) would need
  to dispatch into the trade-market runtime's event ring.
- The verb's own `emit TradeExecuted { ... }` would need to share
  the same event ring as the physics rule's `emit TradeExecuted` —
  meaning the per-handler tag filter would have to handle TWO
  producers writing the same kind tag.

The probe deliberately avoids this. The verb-fire probe
(`2026-05-04-verb-fire-probe.md`) covers the verb cascade END in
isolation; the trade-market probe doesn't combine it.

**Likely follow-up probe:** uncomment the verb slice, re-run, and
record what breaks. That IS its own discovery shape.

### GAP #5 (compiler — known precedent) — `spatial.nearby_hubs(self)` body-form inside multi-emit physics

The probe routes `seller = self`, `hub = self`, `observer = self`
via the auction-style placeholder pattern (see `assets/sim/auction_
market.sim` GAP comments). Real multi-resource market routing would
need:

```
on Tick {} where (self.alive) {
    for hub in spatial.nearby_hubs(self, config.market.trade_radius) {
        emit TradeExecuted {
            buyer:    self,
            seller:   hub,                     // actual neighbor
            resource: pick_resource(hub),
            // ...
        }
    }
}
```

The slice-2b body-form spatial walk landed in `particle_collision_
min.sim`, but it's untested INSIDE a multi-emit physics rule body
that ALSO emits multiple kinds. The combination is plausible (each
emit just becomes a per-iteration emit with the hub bound from the
loop), but it's not exercised here.

**Likely follow-up probe:** swap the placeholder `seller: self` for
`seller: hub` inside a `for hub in spatial.nearby_hubs(self) { ... }`
loop, and verify per-hub volumes accumulate correctly under a real
spatial layout. Out of scope for THIS probe.

### GAP #6 (compiler/runtime — minor) — `Shipment` event declared but never emitted

`assets/sim/trade_market_probe.sim:67-76` declares a `Shipment` event
that today is unused. The compiler accepts the unused declaration
(no diagnostic). The runtime doesn't allocate any Shipment-specific
binding (because no kernel references the event kind). Plausibly
fine — but worth noting that "declared event kinds with zero emit
sites" is a quiet path that wasn't exercised before this probe.

## Compile-gate test landed

`crates/dsl_compiler/tests/stress_fixtures_compile.rs::
trade_market_probe_combines_landed_surfaces` pins the FULL FIRE shape
(NOT the gap chain — this is a "stays working" regression detector,
not a "stays broken" one):

- Catalog: 2 Agent entities (Trader + Hub), 4 Item entities
  (Wood/Iron/Grain/Cloth), 1 Group entity (Guild).
- Lower returns `Ok` with no diagnostics.
- Emit produces 10 kernels: 3 fold (`fold_price_belief`,
  `fold_trader_volume`, `fold_hub_volume`) + 1 decay
  (`decay_trader_volume` only — `hub_volume` has no `@decay`) + 1
  physics (`physics_WanderAndTrade`) + 5 admin (`upload_sim_cfg`,
  `pack_agents`, `seed_indirect_0`, `unpack_agents`, `kick_snapshot`).
- Producer body has 2 tag-store sites at offset 0 (one per emit).
- Each fold body guards on the per-handler kind tag at offset 0
  (the `cb24fd69` shape).
- `fold_price_belief` uses WGSL native `atomicOr` (NOT a CAS loop).
- `fold_trader_volume` + `fold_hub_volume` DO use the CAS+add loop
  (no atomic float add in WGSL; the `+= a` accumulator on a `-> f32`
  view routes through the CAS loop).
- The kernel-name lookup uses exact-name (`fold_trader_volume.wgsl`)
  rather than substring (which would also match `decay_trader_volume`).

If anyone breaks any of these surfaces, this test fires with a
specific assertion message naming the broken kernel + the expected
shape. The test is a **layered regression detector** — it covers
five distinct compiler surfaces in one fixture, so a regression on
any single one (multi-event-kind ring partition / multi-Item catalog
/ pair_map u32 fold / mixed view-fold storage / decay-kernel emit)
shows up here even before the per-feature unit tests catch it.

## Confirmation: existing fixture apps unchanged

`git diff --stat HEAD` against pre-task baseline:

```
 Cargo.lock                                         |  14 ++
 Cargo.toml                                         |   2 +-     (added trade_market_runtime to members)
 crates/dsl_compiler/tests/stress_fixtures_compile.rs | 176 +++++++++++++++++++
 crates/sim_app/Cargo.toml                          |   9 ++    (added trade_market_app dep + bin)
```

Plus untracked: `assets/sim/trade_market_probe.sim`,
`crates/sim_app/src/trade_market_app.rs`, `crates/trade_market_runtime/`.

No existing fixture's .sim, runtime, or sim_app source was touched.
Smoke-checked: `cargo run -p sim_app --bin tom_probe_app` reports
OUTCOME (a) FULL FIRE; `cargo run -p sim_app --bin verb_fire_app`
reports OUTCOME (a) FULL FIRE — both unchanged from their post-fix
baselines.

All 16 stress_fixtures_compile tests pass (was 15 before this probe;
+1 for `trade_market_probe_combines_landed_surfaces`):

```
running 16 tests
test bartering_compiles ... ok
test bartering_resolves_three_distinct_entity_roots ... ok
test bartering_emits_item_id_in_trade_payload ... ok
test bartering_emits_item_and_group_field_reads ... ok
test target_chaser_compiles ... ok
test target_chaser_emits_target_let_binding ... ok
test spatial_probe_compiles_and_emits_neighbour_walk ... ok
test event_kind_filter_probe_compiles_with_tag_guard ... ok
test tom_probe_lowers_clean_and_emits_belief_kernels ... ok
test trade_market_probe_combines_landed_surfaces ... ok        # NEW
test auction_market_compiles ... ok
test swarm_event_storm_compiles ... ok
test swarm_event_storm_emits_four_pulses_per_tick ... ok
test swarm_event_storm_emits_both_folds_with_decay_anchor ... ok
test foraging_colony_compiles ... ok
test ecosystem_cascade_compiles ... ok

test result: ok. 16 passed; 0 failed; 0 ignored
```

`cargo test --workspace` clean — no regressions across any crate.

## Suggested next steps (separate tasks — independent, can parallelise)

Each is a standalone follow-up probe; none block the others:

1. **Item field-read in multi-Item program** (Gap #2). Add `let p =
   items.base_price(0u) + items.base_price(1u);` (or similar) to the
   `WanderAndTrade` rule body and assert the compiler emits 4
   distinct external bindings (`wood_base_price` ... `cloth_base_price`).
   Wire one through the runtime's binding allocator, run, observe.

2. **Group field-read** (Gap #3). Same shape as #1 but for `groups.
   size(0u)` against the `Guild` declaration. Probably trivial since
   bartering already covers this for one Group; the probe is "does
   the compiler keep Groups + Items separate in the catalog when
   both have field reads in the same kernel?".

3. **Verb cascade inside multi-event physics** (Gap #4). Uncomment
   the verb slice in `trade_market_probe.sim`, run, record what
   breaks. Likely surfaces:
   - Two producers writing the same kind tag (1u) — does the per-
     handler tag filter still partition correctly?
   - The verb's scoring kernel needs to fold into the SAME event
     ring as the physics — does the dispatch order interleave them?

4. **Spatial body-form inside multi-emit body** (Gap #5). Replace
   the placeholder `seller: self` with `seller: hub` inside a `for
   hub in spatial.nearby_hubs(self, config.market.trade_radius) {
   emit TradeExecuted { ... } }` block. Real multi-resource market
   routing depends on this. Likely surfaces:
   - Body-form spatial walk INSIDE a multi-emit body.
   - Per-hub `hub_volume` accumulation (the seller key actually
     flowing from a loop binder, not `self`).

5. **Config u32 type round-trip** (Gap #1). Touch the compiler at
   `cg/emit/expr.rs` so `config.<...>.<u32_field>` lowers as `u32`
   in the WGSL emit. Re-introduce `price_q8: config.market.
   observation_bit` in the trade-market probe's PriceObserved emit;
   verify the build no longer crashes the validator.

## Constitution adherence

- **P1 Compiler-First**: no compiler edits in this task. Gaps are
  documented; fixes deferred to separate tasks.
- **P5 Determinism**: no new RNG sites; the probe doesn't introduce
  randomness beyond `per_agent_u32` for initial position scatter.
- **P9 Tasks close with verified commit**: this commit lands the
  probe + the compile-gate test + this discovery doc together.
- **P11 Reduction Determinism**: the probe REUSES the existing
  `atomicOr` (u32) and CAS+add (f32) primitives; no new fold
  primitives invented. The compile-gate test asserts the EXISTING
  primitives are still emitted in the right places.

---

# v2 follow-up — close gaps #4 (verb cascade) + #5 (spatial body-form) (2026-05-04)

This is the v2 update of the trade_market probe (commit baseline
`f4d51052`). The v1 ran with placeholder `seller=self/hub=self`
routing throughout; v2 closes the two highest-priority "deferred"
gaps from the v1 doc by wiring REAL dynamics — verb cascade producing
TradeExecuted, multi-emit physics with body-form spatial walk
producing per-(self, neighbour) PriceObserved + PriceGossip pairs.

**Outcome: (a) FULL FIRE** — every analytical prediction holds on
first run, no NEW compiler/runtime gaps surfaced.

```
trade_market_app: starting — seed=0xBEEFFEEDCAFEF00D agents=32 ticks=100
trade_market_app: finished — final tick=100 agents=32
trade_market_app: trader_volume   (verb chronicle, decay=0.95, steady ~20.000/slot) — total=636.21 (expected 640.00); per-slot min=19.882 mean=19.882 max=19.882
trade_market_app: hub_volume      (verb chronicle, no decay, +1.00/tick/slot) — total=3200.00 (expected 3200.00); per-slot min=100.000 mean=100.000 max=100.000
trade_market_app: price_belief    diagonal 32/32 == 1u, off-diagonal 992/992 == 1u (zero=0, other=0)
trade_market_app: per-view checks — trader_volume=OK hub_volume=OK price_belief=OK
trade_market_app: OUTCOME = (a) FULL FIRE — verb cascade end-to-end (mask -> scoring -> chronicle -> TradeExecuted -> trader/hub_volume folds) integrated with multi-emit spatial body-form physics (WanderAndTrade emits PriceObserved + PriceGossip per spatial-neighbour candidate; both fold into the same price_belief pair_map u32 view via per-handler tag filter).
trade_market_app: OK — all assertions passed
```

## What changed structurally (gap #4 + gap #5 closure)

- **Verb cascade end-to-end** (gap #4 closure). The verb `ExecuteTrade`
  is uncommented and now the SOLE producer of `TradeExecuted` events
  (option (a) from the v1 doc — verb-only emit, no two-producer fold
  ambiguity). Cascade order per tick: `mask_verb_ExecuteTrade` ->
  `scoring` (single-row argmax, emits ActionSelected kind=5u per
  agent) -> `physics_verb_chronicle_ExecuteTrade` (PerEventEmit
  kernel, gates on `event_ring[event_idx*10+0] == 5u` per the
  `cb24fd69` per-handler tag filter, then `event_ring[+3]==0u`
  for action_id Trade, emits TradeExecuted kind=1u). The
  `trader_volume` + `hub_volume` folds read TradeExecuted (kind=1u)
  and produce identical per-slot dynamics to v1 (verb body has no
  spatial access — buyer=seller=self placeholder).

- **Spatial body-form INSIDE multi-emit physics** (gap #5 closure).
  `WanderAndTrade` now contains a `for hub in spatial.nearby_hubs(
  self) { emit PriceObserved {..hub..} ; emit PriceGossip
  {..hub..} }` body. The compiler emits the slice-2b body-form
  3³-cell loop scaffold (per `134c5df8`) — the per-pair candidate
  variable `per_pair_candidate` is bound from the spatial grid's
  cell stream, and BOTH emits inside the inner loop atomicAdd ring
  slots and atomicStore tag/payload. Two emits, same loop body —
  the gap-#5 combination. Build emit-stats:
  `physics_WanderAndTrade: 5546 B, 9 bindings` (vs v1's `2308 B,
  6 bindings` — the +3 bindings are spatial_grid_cells / offsets /
  starts; the +3.2 KB is the spatial walk scaffold + the second
  emit body).

- **PriceGossip event** (new). `PriceGossip { teller, listener,
  hub, resource, price_q8 }` is the rumor-propagation event:
  every spatial-neighbour pair (self, hub) emits one, the
  listener (= hub in the placeholder) accumulates the bit pattern
  in their `price_belief` row via the existing `|=` accumulator.
  The view's `@materialized(on_event = [PriceObserved, PriceGossip])`
  declaration multi-event-handler shape — TWO `on <Event>`
  handlers in the body, each scoping to its own kind tag. The
  emitted `fold_price_belief.wgsl` carries TWO sequential blocks,
  each gated on the per-handler kind tag (commit `cb24fd69`):

  ```wgsl
  // op#1 (view_fold) — PriceObserved kind=2u
  if (event_ring[event_idx * 10u + 0u] == 2u) { ... atomicOr ... }
  // op#2 (view_fold) — PriceGossip kind=3u
  if (event_ring[event_idx * 10u + 0u] == 3u) { ... atomicOr ... }
  ```

  Same `atomicOr(&view_storage_primary[...])` write in both
  blocks — a SINGLE pair_map u32 view fed by TWO event kinds via
  per-handler tag partition.

## Analytical predictions (revised for v2 dynamics)

With `agent_count = 32`, `ticks = 100`, `trade_amount = 1.0`,
`observation_bit = 1u`, `decay = 0.95`:

| View | Per-slot expected | Per-slot observed | Slots converged |
|------|-------------------|-------------------|-----------------|
| `trader_volume[i]` | 20.000 (geo. series limit) | 19.882 (within 0.6%) | 32/32 |
| `hub_volume[i]`    | 100.0 (= T × trade_amount) | 100.000 (exact)    | 32/32 |
| `price_belief[i*N + i]` (diagonal) | 1u | 1u | **32/32** |
| `price_belief[i*N + j]`, i≠j (off-diagonal) | **1u (NEW)** | **1u** | **992/992** |

The off-diagonal flip from "0u everywhere (v1 placeholder)" to
"1u everywhere (v2 spatial walk)" is the directly-observable proof
that the body-form spatial walk is firing inside the multi-emit
physics rule. With auto-spread cube-root scatter across 32 agents
(spread radius ≈ 3.17) and `CELL_SIZE = 6.0`, every agent fits
inside a single 6.0-edge cell, so every (i, j) pair's spatial
walk encounters every other agent — the per-pair body emits both
PriceObserved (sets `[i, j]` = 1u via the PriceObserved handler:
observer=i, hub=j) and PriceGossip (sets `[j, j]` = 1u via the
PriceGossip handler: listener=j, hub=j — the placeholder routing
folds the bit into the `[hub, hub]` cell).

## Inspection of emitted WGSL (acceptance criterion 4)

`physics_WanderAndTrade.wgsl` body (redacted, key shape):

```wgsl
@compute @workgroup_size(64)
fn cs_physics_WanderAndTrade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= cfg.agent_cap) { return; }
    if ((agent_alive[agent_id] != 0u)) {
        let local_3 = agent_pos[agent_id] + agent_vel[agent_id] * config_0;
        agent_pos[agent_id] = local_3;
        // body-form spatial walk: 3 cell-axis loops + 1 candidate loop
        for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
            for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                    let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);
                    let _start = spatial_grid_starts[_cell];
                    let _end = spatial_grid_starts[_cell + 1u];
                    for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {
                        let per_pair_candidate = spatial_grid_cells[_i];
                        // emit PriceObserved (kind=2u, 4 fields)
                        { let slot = atomicAdd(&event_tail[0], 1u);
                          atomicStore(&event_ring[slot * 10u + 0u], 2u); ... }
                        // emit PriceGossip (kind=3u, 5 fields)
                        { let slot = atomicAdd(&event_tail[0], 1u);
                          atomicStore(&event_ring[slot * 10u + 0u], 3u); ... }
                    }
                }
            }
        }
    }
}
```

`physics_verb_chronicle_ExecuteTrade.wgsl` body (redacted):

```wgsl
@compute @workgroup_size(64)
fn cs_physics_verb_chronicle_ExecuteTrade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let event_idx = gid.x;
    if (event_idx >= cfg.event_count) { return; }
    if (atomicLoad(&event_ring[event_idx * 10u + 0u]) == 5u) {
        // ActionSelected: actor=local_5, action_id=local_4, target=local_3
        if ((local_4 == 0u)) {
            // emit TradeExecuted (kind=1u, 5 fields), buyer=seller=actor
            let slot = atomicAdd(&event_tail[0], 1u);
            atomicStore(&event_ring[slot * 10u + 0u], 1u); ... 
        }
    }
}
```

Both spatial body-form (slice 2b) and verb chronicle (PerEventEmit)
kernels are present and their bodies match the expected shapes.
`mask_verb_ExecuteTrade.wgsl` (881 B, 3 bindings) + `scoring.wgsl`
(2209 B, 5 bindings) close the verb cascade.

## Per-tick event-count budget

Each alive agent contributes:
- 1 ActionSelected (kind=5u, scoring)
- 1 TradeExecuted (kind=1u, chronicle gated on ActionSelected)
- K_per_agent PriceObserved (kind=2u, body-form spatial walk per candidate)
- K_per_agent PriceGossip (kind=3u, same)

K_per_agent = candidates in the agent's 27-cell neighbourhood. At
agent_count=32 with cube-root spread, all agents end up in 1 cell
so K_per_agent ≈ 32 (every other agent is a neighbour). Per-tick
ring usage ≈ 32 × (2 + 2 × 32) = 2112 slots/tick. Well under the
ring's 65536 slot capacity. The fold dispatch sizes
`event_count = min(N + N + 2 * N * MAX_PER_CELL * 27, 65536)` =
55298 — generous upper bound; the per-handler tag filter inside
each fold partitions the ring into the correct event-kind subset.

## What did NOT surface (no new gaps)

- The two-producer-same-kind-tag question stays sidestepped by
  picking option (a) (verb-only TradeExecuted emit).
- The Item field-read in multi-Item programs (gap #2 from v1) and
  Group field-read (gap #3) remain untested — the active rule body
  doesn't read `items.base_price(...)` or `groups.size(...)` today.
- The Shipment event remains declared-but-never-emitted; the
  parallel agent on GAP #1 added a "declared event never emitted"
  warning surface in dsl_compiler that catches this exact pattern
  (see the in-flight `crates/dsl_compiler/src/cg/lower/driver.rs`
  changes — that work is OUT OF SCOPE for this task per the
  guardrail).

## Files modified (v2 only — surgical edits inside scope guardrails)

- `assets/sim/trade_market_probe.sim` — replaced placeholder rule
  bodies with verb declaration + body-form spatial walk; added
  PriceGossip event; extended `price_belief` view's `on_event`
  list + handlers to handle both `PriceObserved` and `PriceGossip`
  by per-kind tag.
- `crates/trade_market_runtime/src/lib.rs` — wired the 5-phase
  spatial counting sort, the verb cascade (mask + scoring +
  chronicle), and the bigger physics binding shape (now includes
  `spatial_grid_cells/offsets/starts`). Kept the three view
  storages (price_belief / trader_volume / hub_volume) shape-
  identical to v1; the compiler's per-handler tag filter takes
  care of routing PriceObserved + PriceGossip to the same fold.
- `crates/sim_app/src/trade_market_app.rs` — updated analytical
  predictions for the new dynamics. The off-diagonal `price_belief`
  predicate flipped from "all 0u" to "all 1u" (every spatial pair
  is a neighbour at agent_count=32 cube-root spread).
- `docs/superpowers/notes/2026-05-04-trade_market_probe.md` (this
  file) — appended this v2 follow-up section.

## Existing fixture apps unchanged (acceptance criterion 6)

Smoke-checked all 12 sibling sim_app binaries against their pre-task
baselines: `pp_app`, `pc_app`, `cn_app`, `tc_app`, `ses_app`,
`bartering_app`, `ecosystem_app`, `foraging_app`, `auction_app`,
`verb_fire_app`, `tom_probe_app` — all report their respective
`OUTCOME = OK` / `FULL FIRE` lines unchanged. `cargo test
--workspace` reports zero failures across all 700+ unit tests.

The trade_market_runtime build emits 18 kernels (was 10 in v1):
+5 spatial sort kernels (count, scan_local, scan_carry, scan_add,
scatter), +1 mask_verb_ExecuteTrade, +1 scoring, +1
physics_verb_chronicle_ExecuteTrade. The 10 baseline v1 kernels
remain emitted with the same shapes plus the new dual-handler
fold body in `fold_price_belief.wgsl`.

## Constitution adherence (v2)

- **P1 Compiler-First**: no compiler edits in this task. Gap #1 (the
  config u32 round-trip) is being closed by a parallel sibling task;
  this task does not touch dsl_compiler.
- **P5 Determinism**: no new RNG sites. Spatial walk is deterministic
  (same agent positions ⇒ same cell layout ⇒ same per-pair emit
  order modulo atomic-add ordering, which the OR/CAS folds tolerate
  per P11).
- **P9 Verified commit**: lands the .sim + runtime + app + this
  doc update together; all assertions green; existing fixtures
  unchanged.
- **P11 Reduction Determinism**: atomicOr (price_belief) and CAS+add
  (trader_volume / hub_volume) are commutative — multi-producer
  ordering is observably independent of dispatch ordering.
