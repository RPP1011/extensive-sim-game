# World Sim DSL Standard Library

## Introduction

The stdlib is the set of Rust-backed primitives the compiler recognises
without requiring a DSL declaration. It contains **no game-level views** —
no `is_hostile`, no `at_war`, no `married`, no `threat_level`, no
`hunger_pressure`, no `reputation_log`. Those are game-authored and live
in the game's own `.sim` sources as `view` declarations.

The stdlib contains only pure data access, arithmetic, spatial queries,
and namespace accessors. Every identifier listed here is reserved by the
compiler and resolves without DSL source.

Any call whose name does not match a local, a parameter, a user-declared
view / verb, or an entry in this document stays `UnresolvedCall` in the
milestone-1a IR. A later validation pass (1b) turns unresolved calls into
hard errors. **If a name is missing from this document, it is not a
stdlib name.** Do not add it without following the process in the last
section.

No `.sim` files ship with the compiler. Games declare all derived
predicates in their own DSL source.

---

## Primitive functions

Primitives are `Builtin` variants in
`crates/dsl_compiler/src/ir.rs`. They are resolved in
`crates/dsl_compiler/src/resolve.rs::stdlib::seed()` and dispatched at
runtime by the engine. The engine-side implementations will live at
`crates/engine/src/stdlib/<name>.rs` in a later milestone; the compiler
does not need them to resolve calls.

### Spatial

| Name | Signature | Notes |
|---|---|---|
| `distance` | `(Vec3, Vec3) -> f32` | 3D Euclidean distance. |
| `planar_distance` | `(Vec3, Vec3) -> f32` | Distance in the XY plane; ignores Z. |
| `z_separation` | `(Vec3, Vec3) -> f32` | `abs(a.z - b.z)`. |

### Numeric

| Name | Signature | Notes |
|---|---|---|
| `abs` | `(f32) -> f32` <br> `(i32) -> i32` <br> `(i64) -> i64` | Absolute value; overloaded on numeric scalar types. |
| `min` | `(T, T) -> T` where `T ∈ {i32, i64, u32, u64, f32}` | Pairwise minimum. Also usable as an aggregation over an iterable: `min(x in xs, x.field)`. |
| `max` | `(T, T) -> T` where `T ∈ {i32, i64, u32, u64, f32}` | Pairwise maximum. Same aggregation overload as `min`. |
| `clamp` | `(T, T, T) -> T` where `T ∈ {i32, i64, u32, u64, f32}` | `(value, lo, hi)`. |
| `floor` | `(f32) -> f32` | Round toward `-∞`. |
| `ceil` | `(f32) -> f32` | Round toward `+∞`. |
| `round` | `(f32) -> f32` | Banker's rounding (ties to even). |
| `ln` | `(f32) -> f32` | Natural log. |
| `log2` | `(f32) -> f32` | |
| `log10` | `(f32) -> f32` | |
| `sqrt` | `(f32) -> f32` | |

### ID dereference

| Name | Signature | Notes |
|---|---|---|
| `entity` | `(AgentId) -> EntityRow` <br> `(ItemId) -> EntityRow` <br> `(GroupId) -> EntityRow` | Resolves an ID handle to its table row. Three overloads, one per ID kind. |

`EntityRow` is a placeholder named type at 1a (`IrType::Named("EntityRow")`);
1b will promote it to a variant-per-root row type.

### Aggregations and quantifiers

These share `Builtin` variants with the primitives above but are parsed
as dedicated AST nodes (`Fold` / `Quantifier`), not as calls. They are
listed here for completeness — the names are reserved and will not
collide with user views.

| Name | Syntax | Notes |
|---|---|---|
| `count` | `count(b in iter where pred)` / `count[pred]` | Fold. |
| `sum` | `sum(b in iter, expr)` | Fold. |
| `forall` | `forall x in set: body` | Quantifier; returns `bool`. |
| `exists` | `exists x in set: body` | Quantifier; returns `bool`. |

### Purity notes

All primitive functions are pure over their arguments. None take
`&mut SimState`. The only stdlib surface that draws randomness is the
`rng` namespace (below); `rng` methods read and advance the per-sim
`rng_state`, which the deterministic replay harness treats as part of
the engine input contract (see `docs/dsl/spec.md` §7.2).

---

## Namespaces

Each namespace is a pre-seeded identifier. Bare usage resolves to
`IrExpr::Namespace(NamespaceId)` (seen as an iteration source or left as
an opaque accessor). Field access (`world.tick`) resolves to
`NamespaceField { ns, field, ty }`. Method calls (`rng.uniform(...)`)
resolve to `NamespaceCall { ns, method, args }`.

The typed namespaces are listed below. In addition to these, the
compiler preserves six **legacy collection namespaces** — `agents`,
`items`, `groups`, `quests`, `auctions`, `tick` — without declared
field / method schemas. They are used as iteration sources in
aggregations (`count(a in agents where a.alive)`) and will be replaced
with first-class query surfaces in a later milestone.

### `world`

Sim-level read-only accessors.

| Field | Type |
|---|---|
| `world.tick` | `u64` |
| `world.seed` | `u64` |
| `world.n_agents_alive` | `u32` |

No methods.

### `cascade`

Current cascade-loop observables. Valid only inside metric / probe bodies
and post-phase handlers that can observe cascade state.

| Field | Type |
|---|---|
| `cascade.iterations` | `u32` |
| `cascade.phase` | `enum CascadePhase { Pre, Event, Post }` |

No methods.

`CascadePhase` is a placeholder enum at 1a; 1b will align its variants
with the runtime phase enum.

### `event`

Current-event accessor. Valid inside a physics or fold handler.

| Field | Type |
|---|---|
| `event.kind` | `EventKindId` (placeholder named type) |
| `event.tick` | `u64` |

No methods.

### `mask`

Validation / rejection counters. Primarily used in metric bodies.

| Field | Type |
|---|---|
| `mask.rejections` | `u64` |

No methods.

### `action`

Currently-being-scored action accessor. Valid inside a scoring body.

| Field | Type |
|---|---|
| `action.head` | `ActionHeadKind` (placeholder named type) |
| `action.target` | `Option<AnyId>` (placeholder; `AnyId` is a sum over `AgentId \| ItemId \| GroupId \| ...`) |

No methods.

### `rng`

Deterministic random sampling backed by `SimState.rng_state`. Every
call advances the state; replay reproduces it exactly.

| Method | Signature |
|---|---|
| `rng.uniform(lo, hi)` | `(f32, f32) -> f32` |
| `rng.gauss(mu, sigma)` | `(f32, f32) -> f32` |
| `rng.coin()` | `() -> bool` |
| `rng.uniform_int(lo, hi)` | `(i32, i32) -> i32` |

No fields.

### `query`

Spatial and relational queries. Output is always a bounded list; the
engine enforces per-query caps.

| Method | Signature |
|---|---|
| `query.nearby_agents(pos, radius)` | `(Vec3, f32) -> [AgentId]` |
| `query.within_planar(pos, radius)` | `(Vec3, f32) -> [AgentId]` — planar (XY) distance |
| `query.nearby_items(pos, radius)` | `(Vec3, f32) -> [ItemId]` |

No fields.

### `voxel`

Voxel-grid neighbourhood queries.

| Method | Signature |
|---|---|
| `voxel.neighbors_above(pos)` | `(Vec3) -> [Vec3]` |
| `voxel.neighbors_below(pos)` | `(Vec3) -> [Vec3]` |
| `voxel.surface_height(x, y)` | `(f32, f32) -> i32` |

No fields.

### `agents`

Per-agent slot accessors used by `physics` cascades. The legacy collection
namespace is also iterable as a sequence of agent rows; the methods below
are the milestone-3 accessor surface for compiler-emitted physics handlers.

| Method | Signature |
|---|---|
| `agents.alive(a)` | `(AgentId) -> bool` — true if the slot is still live. |
| `agents.pos(a)` | `(AgentId) -> Vec3` — world position; `Vec3::ZERO` for slots without a pos field. |
| `agents.hp(a)` | `(AgentId) -> f32` — current hp; `0.0` for slots without an hp field. |
| `agents.max_hp(a)` | `(AgentId) -> f32` — authored max hp; `0.0` if absent. |
| `agents.shield_hp(a)` | `(AgentId) -> f32` — current shield hp; `0.0` if absent. |
| `agents.attack_damage(a)` | `(AgentId) -> f32` — per-agent melee damage; falls back to the engine's `step::ATTACK_DAMAGE` constant when the slot is absent. |
| `agents.set_hp(a, v)` | `(AgentId, f32) -> ()` — overwrites the hp slot. |
| `agents.set_shield_hp(a, v)` | `(AgentId, f32) -> ()` — overwrites the shield hp slot. |
| `agents.kill(a)` | `(AgentId) -> ()` — flips the alive bit and tears the agent out of the spatial index. Idempotent. |
| `agents.stun_remaining_ticks(a)` | `(AgentId) -> u32` — ticks of stun remaining; `0` if absent. |
| `agents.set_stun_remaining_ticks(a, v)` | `(AgentId, u32) -> ()` — overwrites the stun-remaining slot. |
| `agents.slow_remaining_ticks(a)` | `(AgentId) -> u32` — ticks of slow remaining; `0` if absent. |
| `agents.set_slow_remaining_ticks(a, v)` | `(AgentId, u32) -> ()` — overwrites the slow-remaining slot. |
| `agents.slow_factor_q8(a)` | `(AgentId) -> i16` — q8 fixed-point speed multiplier under slow; `0` if no slow active. |
| `agents.set_slow_factor_q8(a, v)` | `(AgentId, i16) -> ()` — overwrites the slow-factor slot. |
| `agents.gold(a)` | `(AgentId) -> i64` — current gold balance; `0` if the inventory slot is absent. Signed: debt is representable. |
| `agents.set_gold(a, v)` | `(AgentId, i64) -> ()` — overwrites the gold slot. No-op if the inventory slot is absent. |
| `agents.add_gold(a, delta)` | `(AgentId, i64) -> ()` — `gold += delta` via `i64::wrapping_add`. No-op if the slot is absent. |
| `agents.sub_gold(a, delta)` | `(AgentId, i64) -> ()` — `gold -= delta` via `i64::wrapping_sub`. No-op if the slot is absent. |
| `agents.adjust_standing(a, b, delta)` | `(AgentId, AgentId, i16) -> ()` — symmetric pair-standing adjust, clamped to `[-1000, 1000]`. Inserts the pair entry as a side effect. |

These map one-to-one to `SimState::agent_alive`, `agent_pos`, `agent_hp`,
`agent_max_hp`, `agent_shield_hp`, `agent_attack_damage`, `set_agent_hp`,
`set_agent_shield_hp`, `kill_agent`, `agent_stun_remaining`,
`set_agent_stun_remaining`, `agent_slow_remaining`,
`set_agent_slow_remaining`, `agent_slow_factor_q8`,
`set_agent_slow_factor_q8`, `agent_inventory` / `set_agent_inventory` (for
the gold accessors), and `adjust_standing`. The compiler emits the namespace
calls as direct method invocations on the `&mut SimState` parameter passed
to a cascade handler.

---

## Metric wrappers are NOT stdlib functions

`histogram(expr)`, `gauge(expr)`, `counter(expr)` appear in the DSL but
are part of the **metric-declaration grammar**, not callable stdlib
functions.

As of milestone 1a they are still parsed as generic `Call` expressions
and surface in the IR as `UnresolvedCall("histogram", …)`. This is
technical debt: they belong alongside `window` / `emit_every` /
`alert when` on `MetricDecl`, as a `MetricKind` enum. **TODO** (follow-up
to this milestone): lift them into `MetricKind` in the AST / parser /
IR; drop the `UnresolvedCall` fallback.

Do not add them to the stdlib primitive list — this is the wrong layer.

---

## What is NOT in stdlib

The following names are **game-specific derivations**, not engine
primitives. They are NOT recognised by the compiler. A game that uses
them must declare them in its own DSL (`view <name>(...)` for scalar
derivations, `query <name>(...)` for spatial queries):

- `is_hostile`
- `at_war`, `groups_at_war`
- `married`
- `threat_level`
- `hunger_pressure`, `hunger_urgency`
- `reputation`, `reputation_log`
- `relationship`
- `eligibility` / `eligible_*` predicates
- `can_marry`, `can_trade`, any `can_*` capability derivation
- `predator_prey`, `prey_of`

Games should define these as:

```
view is_hostile(a: Agent, b: Agent) -> bool {
  relationship(a, b).valence < HOSTILE_THRESH
  || groups_at_war(a, b)
  || predator_prey(a.creature_type, b.creature_type)
}
```

Until such a declaration exists, references to these names stay
`UnresolvedCall` at 1a, and 1b reports them as unknown calls.

---

## Restriction surfaces

Two closed operator sets govern what the compiler accepts in specific
surfaces. Both are intentionally small so the corresponding emitters can
compile to GPU-friendly kernels without escape hatches.

### Fold body operator set (`@materialized` views)

See `spec.md` §2.3 "Fold body operator set". User-defined helper calls,
cross-view composition, unbounded loops, and recursion are forbidden
inside fold bodies. The compiler raises
`ResolveError::UdfInViewFoldBody` pointing at the offending construct.

### Mask predicate operator set

See `spec.md` §2.5. Supports set membership, quantifiers, bounded
folds, arithmetic comparison, and view calls. No user-defined
functions inside masks.

### Scoring modifier operator set

See `spec.md` §3.4. Scoring entries are expression-sums where each
term is one of: a base literal, a boolean `if <pred> { <lit> } else { 0.0 }`
modifier, or a gradient `<expr> per_unit <delta>` modifier. Anything
else is `EmitError::UnsupportedExprShape`.

---

## Adding to stdlib

The stdlib is not a frequent-change surface. Adding a primitive or a
namespace field requires all three of:

1. An entry in `docs/dsl/spec.md` (§2, §3, or §5 as appropriate) and in
   this document, with a pinned signature.
2. A `Builtin` variant or a `NamespaceId` / `stdlib::field_type` /
   `stdlib::method_sig` extension in `crates/dsl_compiler/src/`, plus a
   fixture in `crates/dsl_compiler/tests/fixtures/` exercising it and
   a blessed IR golden.
3. An engine-side implementation at `crates/engine/src/stdlib/<name>.rs`
   (planned path; may be relocated once the engine crate lands).

A stdlib addition bumps the compiler vocabulary, the schema hash, and
requires buy-in from downstream training / replay consumers. If you're
tempted to add `is_hostile` here to save a line of DSL, that's the
wrong trade — declare it in the game's DSL instead.
