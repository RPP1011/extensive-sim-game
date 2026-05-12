# Gaps surfaced by the `palace_coup` adversarial fixture

**Fixture:** `assets/sim/palace_coup.sim` (24 emitted kernels)
**Pin:** `crates/sims/tests/palace_coup_pin.rs`
**Date:** 2026-05-11
**Adversarial axes stacked:** Plan G `cast { telegraph: ... duration: ... }`
block + discrete trust state machine + pair-keyed beliefs (ToM) + voxel
LoS + multi-row scoring (King/Guard/Conspirator each have a different
verb roster).

This is one of seven parallel fixtures spawned to surface architectural
gaps the auto-emit pipeline missed during the toy-fixture cleanup
sweep. The `hill_raid` discovery (commit `78ad8a77` → `1c565df9`)
already surfaced 3 gaps; `palace_coup` is the cast-block-heavy
companion.

## Run-time behaviour

Topology: 1 King + 8 Guards (palace ring) + 4 Conspirators (perimeter)
+ 20 Civilians + voxel wall ring (palace perimeter at radius 5).

300-tick report from `RUST_MIN_STACK=33554432 cargo test -p sims
--release --test palace_coup_pin -- --nocapture`:

```
==== palace_coup 300-tick report ====
  topology: 1 King, 8 Guards, 4 Conspirators, 20 Civilians
  final:    kings=1/1  guards=8/8  conspirators=4/4  civilians=20/20
  cast:     king_busy_at_some_point = false
  views:    view_storage_primary nonzero cells = 0
            (note: 3 views share this buffer per documented gap)
  beliefs:  any guard_belief cell set = false
  verdict:  STALEMATE — no decree fired (cast-block dispatch
            likely never enabled by gates)
  movement: mean conspirator radius = 0.04 (init = 8.0)
  contract: 24 kernels emit, 300 ticks step without panic, all
            seeded slots survive, voxel terrain wires up, view
            storage allocated, cast block lowers + emits.
```

The conspirator-advance rule fires (mean radius drops from 8.0 to
0.04 — they've all crowded onto the king's hilltop). But every
chronicle-driven event downstream is silently swallowed.

## Discovered / re-confirmed gaps

### G1. `Indirect` chronicle dispatch is a no-op (re-confirmed)

**Status:** documented in commit `353527e6` ("docs(build_helper):
document four-gap blocker for Indirect dispatch arm"); palace_coup
re-pins it at the .sim level.

The auto-emitted `step()` walks `SCHEDULE` and only handles
`DispatchOp::Kernel(_)`; `DispatchOp::Indirect{ kernel, args_buf }` and
`DispatchOp::FixedPoint{ ... }` fall through `_ => {}`. For palace_coup,
the schedule contains 3 Indirect entries (out of 24 stages):

```rust
DispatchOp::Indirect { kernel: KernelId::PhysicsApplyDamageFromChronicleAndRecordCastBegin, ... },
DispatchOp::Indirect { kernel: KernelId::PhysicsApplyDamage, ... },
DispatchOp::Indirect { kernel: KernelId::PhysicsEvidenceConsumer, ... },
```

This silences the entire chronicle path:

* `RecordCastBegin` (Plan G consumer that stamps `busy_until_tick`)
  never fires → `king_was_busy_at_some_point = false` even though
  `KingDispatchDecree` (a per-agent kernel) does run and dispatches
  `apply_ability 4 by self target self` every 60 ticks.
* `ApplyDamage` never fires → no Damaged events drain hp, the
  `Defeated` cascade never triggers.
* `EvidenceConsumer` never fires → trust state machine never advances
  past Loyal.
* The `EvidenceFiled` events `ResolveDecree` and `ApplyDamage` emit go
  into the chronicle ring with no consumer.

**Closing this gap unblocks**:
* End-to-end `cast` block validation (the largest Plan G slice).
* Every chronicle producer→consumer chain in the mega-crate.
* Per-fixture cfg uniform writes for the consumer kernels (one of
  the four sub-gaps `build_helper.rs` documents).

### G2. All materialized views share a single `view_storage_primary_buf`

**Status:** documented in `build_helper.rs::slot_count_expr` ("today's
pair-keyed detection is a single fixture-level bool... TODO: replace
name-list heuristic with proper binding-shape annotation in the AST");
palace_coup is the first fixture that actually has 3 distinct views
and surfaces the collision.

palace_coup declares 3 materialized views:

```dsl
@materialized(on_event = [TrustTransition])
view trust_transitions(target: Agent) -> f32 { ... }

@materialized(on_event = [Damaged])
view damage_dealt(source: Agent) -> f32 { ... }

@materialized(on_event = [BeliefAcquired], storage = pair_map)
view guard_belief(observer: Agent, subject: Agent) -> u32 { ... }
```

The auto-emit allocates ONE `view_storage_primary_buf` (sized N²
because `guard_belief` is pair-keyed) and binds it to all three fold
kernels. The fold kernels write `view_storage_primary[k]` for some
per-view-shape `k`; with three folds writing the same physical buffer
each tick, only the LAST fold's writes survive per cell, and per-agent
views (which only intend to write the first N cells) collide with the
pair view's N² range.

Today this is masked by gap G1 — none of the views actually receive
producers (Damaged isn't emitted because ApplyDamage is silenced;
TrustTransition isn't emitted because EvidenceConsumer is silenced;
BeliefAcquired DOES fire in `GuardObserve` which is a regular kernel
path, but the fold itself appears to never produce an observable write
in the read-back). Once G1 closes, the view-collision behaviour
becomes visible and load-bearing — the runtime needs one
`view_<name>_storage_primary_buf` per view declaration, with the
fold-kernel binding plumbed per-view rather than name-shared.

**Closing this gap unblocks**: any fixture with ≥2 materialized views
producing observable readbacks.

### G3. Plan G `cast { telegraph: ... }` block — parse/lower/emit IS complete

**Surprise discovery (positive):** the cast block IS more complete
than the parent agent's hint suggested. `RoyalDecree.ability` uses

```
cast { duration: 6t telegraph: circle(self.pos, radius: 12) }
effect { heal 0 }
```

and lowers cleanly through:

* `dsl_compiler::ability_lower::lower_ability_decl` populates
  `prog.effects = [EffectOp::CastBegin { ... }]` (per
  `cast_block_lowers_to_cast_begin` test).
* `prog.pending_program = [EffectOp::Heal { amount: 0.0 }]` (per
  `cast_block_effects_lower_into_pending_program` test).
* `prog.telegraph_kind = TelegraphKind::Circle (= 1)`,
  `prog.telegraph_params[0] = 12.0` (per
  `cast_block_with_circle_telegraph_lowers_to_program_fields`).
* `cg::emit::emit_cg_program` produces the
  `physics_ApplyDamageFromChronicle_and_RecordCastBegin` kernel that
  reads `EffectCastBeginApplied` (kind=77) chronicle records and
  stamps `agents.busy_until_tick = world.tick + 6`.

What's still missing end-to-end:
* The `RecordCastBegin` consumer is on the Indirect dispatch arm (gap
  G1) — even though it emits, it's never bound at step time, so the
  king's `busy_until_tick` never updates.
* `apply_pending_program` (the registry-driven dispatch of
  `pending_program` on busy-resolve) is still deferred — for this
  fixture the .sim's `ResolveDecree` rule emits an `EvidenceFiled`
  marker as a per-fixture stand-in (also silenced by G1).

So the cast-block surface is ~70% complete: parse + lower + WGSL emit
+ kernel allocation all work. The 30% gap is the dispatch arm (G1)
plus the registry-side `pending_program` execution.

### G4. Multi-stage cast chains — single-stage MVP only

**Status:** documented in commit-trail and `multi_stage_cast_chain_takes_first_cast_only_today` test;
palace_coup uses a single-stage cast so this isn't a blocker here, but
worth flagging in the gaps log.

`cast { duration: 3t } effect { ... } cast { duration: 5t } effect
{ ... }` lowers but takes only the FIRST cast's CastBegin. Subsequent
`Cast` steps are silently dropped (all `Effects` blocks merge into
one `pending_program`). A future fixture that wants
"channel → release" or "wind-up → strike → cooldown" multi-stage
sequencing will need this.

## Things the fixture validated

These are surfaces palace_coup exercised that did NOT surface a gap —
they work end-to-end in the mega-crate:

1. **24 kernels emit cleanly** from a fixture with 4 entity types,
   12 physics rules, 3 views, 3 spatial queries, 1 cast block — no
   compile errors, no schedule synthesis failures, no missing
   bindings.
2. **Voxel terrain auto-detection** — palace_coup's `terrain.line_of_sight`
   call in `GuardObserve` triggers the `binds_voxel_grid` arm in
   `build_helper.rs`, which allocates `voxel_terrain` + `voxel_mirror`
   on `GeneratedRuntime` and wires them into the `KernelBindingsContext`.
   Same shape `hill_raid` validates.
3. **Pair-map view storage sizing** — the presence of `guard_belief`
   (pair-keyed) up-sizes `view_storage_primary_buf` to N²×4 bytes
   correctly via `slot_count_expr`'s `pair_keyed_view_present`
   detection.
4. **`@spatial` query attribute + `terrain.line_of_sight`** combine
   without conflict — three separate `@spatial(radius=...)` queries
   resolve cleanly.
5. **Discrete-state-machine encoding via re-purposed SoA columns**
   parses + lowers — `agents.mana(t)` as a 4-state discriminant works
   in `where` clauses and in branching expressions; `agents.shield_hp(t)`
   as an evidence accumulator parses fine.
6. **`init { alive: 1, hp: 100 }`** survives 300 ticks of compilation
   (was missing initially → all agents died at tick 0; adding the
   block fixed it). The mega-crate auto-emit honors the init values
   correctly.
7. **`set_busy_until_tick` / `set_busy_with_ability_id` /
   `set_busy_started_at_tick`** AgentField setters resolve and lower
   into the chronicle consumer kernel cleanly. The Plan G SoA columns
   exist in the engine and the DSL knows how to write them.
8. **Conspirator advance via `sum(... in agents where ...)`** with
   `length` + normalize survives 300 ticks without divergence to
   NaN/Inf — same recurrence shape `hill_raid` pins.
9. **Multi-row scoring infrastructure** holds: each entity type
   has its own verb count, the scoring kernel survives without
   schedule errors. (Whether it actually picks the right verb per
   entity is a behaviour question gated on G1 closure.)

## Next steps

* G1 (Indirect dispatch) is the highest-leverage fix — closing it
  validates EVERY chronicle producer/consumer chain in the mega-crate.
  See `build_helper.rs:1953` for the no-op site and the four-gap
  blocker comment.
* G2 (per-view storage buffers) is naturally gated on G1 — without
  G1, no view producers fire so the collision is invisible.
* Once G1 + G2 land, palace_coup's behavioural pins can flip from
  "300 ticks step without panic" to:
  - king_alive at end (assassination outcome is observable)
  - king_busy_at_some_point = true (cast block fired)
  - count of trust transitions > 0 (state machine advanced)
  - count of Damaged events > 0 (violence flowed through)
* Fixture as written can stay — once chronicle dispatch lands, the
  pin asserts strengthen automatically without source changes.
