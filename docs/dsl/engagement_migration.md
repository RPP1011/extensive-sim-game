# Engagement Migration: hand-written Rust → DSL-owned physics

Task 163. Moves `crates/engine/src/engagement.rs` (~254 LOC) to two
`physics` rules in `assets/sim/physics.sim` so the north-star invariant
(`engine = primitives only; DSL = all game logic`) holds for the
event-driven engagement update landed in task 139.

Target shape: the engine keeps only the **spatial index** and the
**engagement-slot SoA field**; the decision of "who engages whom on this
move" is a DSL physics rule the compiler lowers to a SPIR-V-safe Rust fn
under `generated/physics/`.

## Inventory: what engagement.rs currently does

Every semantic chunk maps to one of four buckets:

| Chunk | Lines | Bucket | Notes |
|---|---|---|---|
| `break_reason::{SWITCH, OUT_OF_RANGE, PARTNER_DIED}` u8 consts | 47-56 | **moves to DSL** | three `let` bindings inside the rule body, or three enum variants the engine registers — opt for literal `u8`s (trivial, no new surface). |
| `recompute_engagement_for`: position + creature reads, spatial scan, argmin with id tiebreak, hostility check | 66-172 | **moves to DSL, needs new primitive** | the argmin-with-tiebreak inside a for-loop isn't currently expressible — no mutable `let`s in physics bodies. Wrap as `query.nearest_hostile_to(agent, radius) → Option<Agent>` stdlib primitive, which reuses the engine's existing `SpatialHash::within_radius` + `CreatureType::is_hostile_to` with the same tie-break discipline. |
| Read old partner; compare to new partner | 113-117 | **moves to DSL** | `agents.engaged_with(x)` already exists for masks (returns `Option<AgentId>`). Emitter needs `== None` / `!= None` lowering copied from mask emit. |
| Emit `EngagementBroken` for stale partner, clear both slots eagerly | 121-137 | **moves to DSL** | emit statement already works. Need `agents.set_engaged_with(x, Option<Agent>)` primitive for the eager write. |
| Three-agent displacement (partner was already paired with stranded) | 138-171 | **moves to DSL, needs unwrap** | requires unwrapping `Option<AgentId>` inside the body. Add `agents.engaged_with_or(agent, default) → Agent` primitive so the rule can sentinel on the partner itself (`if stranded != partner && stranded != mover`). |
| `break_engagement_on_death` | 177-195 | **moves to DSL** | same primitives as above. Separate physics rule on `AgentDied`. |
| `dispatch_agent_moved` / `dispatch_agent_died` cascade wiring | 231-254 | **deleted** | compiler emits the dispatcher + `register()` fn. The `AgentDied` dispatcher in `generated/physics/mod.rs` today runs only `chronicle_death` — after this task it also runs the new `engagement_on_death` handler, ordered by rule name (chronicle_death < engagement_on_death alphabetically), which matches the current hand-written flow (chronicle fires first, engagement teardown after). |
| `recompute_all_engagements` back-compat shim | 207-216 | **stays hand-written** | called only by two legacy tests (`proptest_engagement.rs`, `engagement_tick_start.rs`). Keep as a thin `pub fn` in an **engagement-test-helpers module** (`crates/engine/src/engagement.rs` shrinks to this one shim + the three u8 constants) — ~30 LOC. Alternative: convert the tests to drive the cascade (emit `AgentMoved` for each agent, run the registry). That's more surgery than the task asks for. Keep the shim; leave a comment pointing at a future test-migration cleanup. |

## Primitive extensions required

All three generalize beyond engagement:

### 1. `query.nearest_hostile_to(agent: Agent, radius: f32) → Option<Agent>`

Returns the nearest hostile within `radius` of `agent`, ties broken on
raw `AgentId`. Returns `None` when nothing matches. Implemented in the
stdlib by walking the spatial index with the same hostility predicate
the existing code uses (`state.agent_creature_type(a).is_hostile_to(b)`).

**Broader use**: any physics rule that needs "closest enemy" — e.g., a
future "flee-from-nearest-predator" physics, retaliation emission on
aoe-damage-applied, a "witness" view that records who saw what. The
primitive is shaped to read a single element (not an iterable), which
keeps the GPU-emittable subset intact — no unbounded loop appears in
the generated physics body, the bounded loop lives inside the stdlib
helper fn (same pattern as `spatial.within_radius` inside mask
candidate enumerators).

**Why not a more generic `query.nearest_matching(agent, radius,
predicate)`?** Physics bodies can't take closures (`UDF in physics body`
is a validator reject). Until the DSL grows user-defined helpers
restricted to the GPU subset, bespoke filter primitives are how the
stdlib grows — precedent: the existing `query.nearby_agents` /
`query.within_planar` / `query.nearby_items` trio.

### 2. `agents.set_engaged_with(agent: Agent, partner: Option<Agent>)`

Eager write to `hot_engaged_with[slot]`. No-op on out-of-range slots
(matches the bounds-tolerant convention of every other setter in the
stdlib). Used for the same-tick read-your-own-writes invariant that the
hand-written code documents.

**Broader use**: none beyond engagement today, but mirrors every other
`set_*` on `agents.*`. The alternative — have the DSL rule emit
`EngagementCommitted` / `EngagementBroken` and rely on the view-fold
phase for visibility — breaks the in-tick cascade invariant (the view
fold runs once per tick; later cascade handlers need to read the fresh
slot before the fold rebuilds it).

### 3. `agents.engaged_with_or(agent: Agent, default: Agent) → Agent`

Unwrap-or-default for the engagement slot. Returns the partner if set,
else `default`. Lets the DSL rule sentinel on the partner itself:
`let stranded = agents.engaged_with_or(partner, partner)` — if
`stranded != partner`, we know the slot was set.

**Broader use**: unwrap-or is a general Option operation. The spec
leaves generic `Option` unwrapping out of the physics-body surface
because there's no safe "null AgentId" (`AgentId: NonZeroU32`), but a
domain primitive that names its own sentinel is both GPU-emittable and
usable. A future `agents.owner_or(item, self)` on item ownership slots
would follow the same pattern.

### 4. `== None` / `!= None` lowering in `emit_physics.rs`

Copy-paste the six-line stanza already living in `emit_mask.rs` (line
592-614). Lowers to `.is_none()` / `.is_some()` on the emitted Rust.
Precedent for reuse: the mask emitter's `None` handling is already
exercised by `mask Cast`'s engagement-lock clause, so this is
strictly adding parity to a second emitter, not new surface.

## Physics body GPU-safety — no validator changes

The new primitives all pass the task-158 validator unchanged:

- `query.nearest_hostile_to` is a `NamespaceCall` returning a single
  value (not a list), so it never appears as a `for` iteration source;
  `validate_physics_iter_source` isn't invoked. It's a leaf call, same
  shape as `agents.hp` — the validator treats it as pure stdlib.
- `agents.set_engaged_with` / `agents.engaged_with_or` are scalar
  setters / getters (same shape as `agents.set_hp` / `agents.max_hp`).
- `== None` becomes an `IrExpr::Binary(BinOp::Eq, ..., EnumVariant{"None"})`
  which the validator already accepts (it recurses into both sides and
  each side is a pure value).
- No new `for` loops in the rule body. The only iteration is inside
  the stdlib-fn implementation of `nearest_hostile_to`, which is
  CPU-side Rust today and will be hoisted to a SPIR-V spatial-query
  intrinsic the same way `spatial.within_radius` will — the rule body
  stays ceiling-bounded (constant number of expressions).

## DSL rule shape (sketch)

```dsl
physics engagement_on_move @phase(event) {
  on AgentMoved { actor: mover } {
    if agents.alive(mover) {
      let new_partner = query.nearest_hostile_to(mover, config.combat.engagement_range)
      let old_partner = agents.engaged_with(mover)
      if old_partner != new_partner {
        // Break stale pairing, if any.
        let former = agents.engaged_with_or(mover, mover)
        if former != mover {
          agents.set_engaged_with(mover, None)
          agents.set_engaged_with(former, None)
          let reason = if new_partner == None { 1 } else { 0 }   // OUT_OF_RANGE : SWITCH
          emit EngagementBroken { actor: mover, former_target: former, reason: reason }
        }
        // Commit new pairing, if any, breaking partner's prior pair first.
        if new_partner != None {
          let partner = query.nearest_hostile_to(mover, config.combat.engagement_range)  // reread
          // ... three-agent displacement
        }
      }
    }
  }
}
```

The re-read of `nearest_hostile_to` above is the one wart: the DSL
doesn't support narrowing an `Option<Agent>` to `Agent` inside an `if
opt != None { ... }`. Two ways to fix it:

1. **Use `engaged_with_or` style sentinel** for new_partner too — add
   `query.nearest_hostile_to_or(agent, radius, default) → Agent` as a
   fourth primitive. Clean, one-extra-primitive cost, trivial
   implementation.
2. **Add `if let Some(x) = opt { ... }` to physics bodies** — broader,
   more useful, but bigger compiler surface (parser + IR + lowering +
   validator) than the task wants.

**Decision:** go with (1) — `query.nearest_hostile_to_or` returns the
sentinel when nothing matches. The `== default` check distinguishes
"nothing in range" from "partner found" inside the rule body.

Final primitive list:

1. `query.nearest_hostile_to(agent, radius) → Option<Agent>`
2. `query.nearest_hostile_to_or(agent, radius, default) → Agent`
3. `agents.set_engaged_with(agent, Option<Agent>)`
4. `agents.engaged_with_or(agent, default) → Agent`
5. `emit_physics::lower_expr_kind` gains the `== None` / `!= None`
   stanza from `emit_mask`.

Cumulative compiler diff estimate: ~40 LOC across `resolve.rs` (method
sigs + `Optional` arg typing) and `emit_physics.rs` (namespace call
arms). Engine primitive additions: ~50 LOC in `state/mod.rs` (existing
accessors already exist for most; only the two new stdlib helpers are
fresh code) + ~40 LOC new spatial-query helpers in `spatial.rs`. Rough
total: ~130 LOC compiler + engine surface + rule text.

## What stays hand-written

- **Spatial index** (`crates/engine/src/spatial.rs`) — primitive, not
  game logic.
- **SoA `hot_engaged_with` field** and the `agent_engaged_with` /
  `set_agent_engaged_with` getters on `SimState` — engine data, DSL
  only reaches it through stdlib.
- **`recompute_all_engagements` test shim** — thin 10-line helper kept
  in a shrunken `engagement.rs` (~15 LOC total: 3 `break_reason` u8
  consts + the shim). Documented as "for legacy tests only, new code
  relies on the cascade dispatch". The shim itself walks
  `agents_alive()` and **emits `AgentMoved`** events for each agent,
  then runs the cascade to fixed point — it no longer duplicates the
  recompute logic. That's both correct and thin.

## Test plan

1. **Byte-exact baseline parity** (`parity_log_is_byte_identical_to_baseline`):
   the current baseline captures a specific tie-break outcome inside
   `SpatialHash::within_radius` → slot-order-of-candidates. The new
   `nearest_hostile_to` primitive **must iterate the same spatial
   candidate list in the same order** and apply the same tie-break
   (raw AgentId on equal distance). Implementation note: reuse
   `SpatialHash::within_radius` verbatim, walk results in their existing
   (sorted-by-raw-id) order, track `(best_id, best_distance)` with the
   same `<` / equal-then-lower-raw comparison the existing code uses.
   If bit-exact parity holds, no baseline regeneration is needed.
2. **Behavioral tests** (`parity_log_has_expected_structure`,
   `wolves_prefer_wounded_humans`, etc.): these assert on structure,
   not byte identity. They'll stay green as long as the semantics
   (bidirectional commits, three-agent displacement, break-on-death)
   are preserved.
3. **Proptest invariants** (`proptest_engagement.rs`): run with the
   new `recompute_all_engagements` shim-via-cascade. Bidirectional +
   range + same-species-rejection properties all flow from the rule
   body's `nearest_hostile_to` + commit logic.
4. **Legacy `engagement_tick_start.rs`**: same shim. The shim's new
   implementation (emit `AgentMoved` + cascade) may change ordering
   vs. the old in-place recompute for the multi-agent cases; if any
   specific test pins a three-agent outcome that depended on the old
   iteration shape, it'll fail and will need to be examined. The
   module comment already notes that "unilateral" commits replaced
   "tentative-commit", so some divergence here is acceptable.
5. **Chronicle render test** (`chronicle_renders_readable_text`): must
   still see an engagement line in the prose stream — both the
   `chronicle_engagement` physics rule (on `EngagementCommitted`) and
   this task's new `engagement_on_move` rule emit
   `EngagementCommitted`, so the chronicle wiring is unchanged.
6. **Compiler/integration**: the three new stdlib arms in `resolve.rs`
   get arity/type assertions. Add a targeted unit test in
   `emit_physics.rs` for the `== None` / `!= None` lowering (mirroring
   the mask emit's precedent test).
7. **Schema hash**: the event taxonomy is unchanged — no new events,
   no field edits. `engine_rules::EVENT_HASH` should not move.

## Go / no-go criterion

Go criteria (from the task prompt): new compiler/parser surface ≤ 150
LOC, no non-trivial change to the task-158 validator.

Against the plan:

- Compiler surface additions: ~40 LOC (stdlib method sigs + physics
  emitter lowering for `== None` / new stdlib arms). **Under budget.**
- Validator changes: zero. All new primitives pass `validate_physics_expr`
  / `validate_physics_iter_source` as existing-shape `NamespaceCall`s.
  **No validator touch.**
- Parser changes: zero. Existing `for`/`match`/`if`/`let`/`emit`/stdlib-call
  grammar covers the rule bodies.

**Decision: proceed to implementation.**

## Out of scope for this task

- Folding `hot_engaged_with` into the materialized view (eliminating
  the dual-storage awkwardness). That's a separate refactor; the
  task-155 view-storage feasibility work is the right place to pick
  it up.
- Converting the legacy `engagement_tick_start.rs` tests to drive the
  cascade directly instead of the shim.
- Generic `if let Some(x) = opt` support in physics bodies. Not
  needed; `nearest_hostile_to_or` covers the narrow case.
