# Language + Stdlib Spec Audit (2026-04-26)

> Audit of `docs/spec/language.md` + `docs/spec/stdlib.md` against `crates/dsl_compiler/`.
> Auditor: agent session; all findings are READ-ONLY observations.

---

## Summary

- **Grammar productions: 43 total** — ✅ 30 | ⚠️ 8 | ❌ 5 | 🤔 0 | ❓ 0
- **Stdlib functions: 57 total** — ✅ 30 | ⚠️ 16 | ❌ 7 | 🤔 4 | ❓ 0

---

## Top gaps (ranked by impact)

### 1. `agents` methods: spec names vs. impl names mismatch (🤔 × 4)

`docs/spec/stdlib.md` §agents specifies these four accessor names:

| Spec name | Impl name (resolver + all emitters) |
|---|---|
| `agents.stun_remaining_ticks(a)` | `agents.stun_expires_at_tick(a)` |
| `agents.set_stun_remaining_ticks(a, v)` | `agents.set_stun_expires_at_tick(a, v)` |
| `agents.slow_remaining_ticks(a)` | `agents.slow_expires_at_tick(a)` |
| `agents.set_slow_remaining_ticks(a, v)` | `agents.set_slow_expires_at_tick(a, v)` |

The semantic change is real: the spec exposes a *remaining* duration (ticks left), but the implementation stores and exposes an *absolute expiry tick*. Task 143 changed the engine storage model but did not update `docs/spec/stdlib.md`. Any DSL author or ToM session reading the spec will write `agents.stun_remaining_ticks(a) > 0` and get an `UnresolvedCall` at compile time. **The spec must be updated to match the implementation.**

### 2. `query.nearby_agents`, `query.within_planar`, `query.nearby_items` — resolves but emits only in masks (⚠️ in physics/scoring/views)

These three methods are registered in `stdlib::method_sig` (resolve.rs:241-249) and parse/resolve correctly. However:

- `emit_mask.rs` lowers `query.nearby_agents` in the `from` clause of a mask (and only that clause).
- `emit_physics.rs` has **no lowering** for `query.nearby_agents`, `query.within_planar`, or `query.nearby_items`; they fall through to the `_ => Err(EmitError::Unsupported(...))` catch-all.
- `emit_scoring.rs` has no lowering for any of the three.
- `emit_view.rs` has no lowering for them.

`physics.sim` comment at line 621 explicitly documents this: *"`query.agents_within(pos, r)` doesn't exist in the physics emitter"* (note: the spec spells it `query.nearby_agents`; `query.agents_within` is a non-spec alias that was attempted and failed). This is the **highest-impact** gap blocking real physics rules that iterate nearby neighbours.

### 3. `agents.creature_type(id)` — resolves in views only; missing from physics/scoring emitters (⚠️)

`emit_view.rs` lowers `agents.creature_type(id)` to `state.agent_creature_type(id)` (line 2348). The method is **not registered** in `stdlib::method_sig` at all — the resolver will accept it only when invoked in a view body via the view emitter's `lower_namespace_call`. In `emit_physics.rs`, it falls to the `_ => Err(Unsupported)` arm. `physics.sim` line 629 confirms: *"`agents.creature_type(id)` is not a registered resolver method"*. Scoring bodies also cannot call it. Practical impact: any physics rule that needs creature-type dispatch (e.g. the ground-snap rule, fear-spread) must hand-code the Rust.

### 4. `voxel.*` methods — resolve but emit only in physics (⚠️ in views/scoring)

`voxel.neighbors_above`, `voxel.neighbors_below`, `voxel.surface_height` are registered in `method_sig`. `emit_physics.rs` lowers all three correctly. However `emit_view.rs` and `emit_scoring.rs` do not lower them — they fall to the catch-all `Unsupported`. The ground-snap worked example in `language.md §7.1` uses `surface_height(a.pos.xy)` from inside a physics body, which is fine. A `@lazy` view that calls `voxel.surface_height` would fail emission.

### 5. `verb`, `invariant`, `probe`, `metric` declarations — parsed and resolved but **not emitted** (❌/⚠️)

`EmittedArtifacts` in `lib.rs` contains no fields for verb, invariant, probe, or metric outputs. They parse into `VerbIR`, `InvariantIR`, `ProbeIR`, `MetricIR` and are carried in `Compilation`, but the `emit_with_per_kind_sources` function never calls an emitter for them. `histogram()`/`gauge()`/`counter()` inside metric blocks surface as `UnresolvedCall` per `stdlib.md`'s own "TODO" note. These are milestone-deferred gaps but block the CI probe surface and runtime observability entirely.

---

## Grammar findings (per production)

### §2.1 `entity` declaration
- **Parse / resolve**: ✅ Full — `EntityDecl`, `EntityField`, `EntityFieldValue`, `@spatial`, `@materialized`, `@capped`, `@inline` annotations all parse. `EntityIR` is built in resolve pass.
- **Emit**: ✅ `emit_entity.rs` lowers entity decls; `emit_entity_mod` aggregates them.
- **Note**: The nested `Capabilities` struct literal, `predator_prey` object, and `default_memberships` list literals all parse as `EntityFieldValue::{StructLiteral,List,Expr}`. No evidence of type-checking the struct field names at 1a.

### §2.2 `event` declaration
- **Parse / resolve**: ✅ `EventDecl`, `@replayable`, `@non_replayable`, `@high_volume`, `@traced`, `@gpu_amenable` annotations parse. `event_tag` decl (`EventTagDecl`) parses. Tags on events via `@tag_name` annotations resolved.
- **Emit**: ✅ `emit_rust.rs` emits event structs, mod, `EventLike` impl. Python mirrors via `emit_python.rs`.
- **Note**: `@non_replayable` String-field enforcement (spec §2.2: "String payloads permitted only on `@non_replayable` events") is NOT checked in the resolver. No `ResolveError::StringOnReplayableEvent`.

### §2.3 `view` declaration

#### `@lazy` view
- **Parse / resolve / emit**: ✅ Lowers to a Rust `fn`. Cross-view calls (`ViewCall`) inside lazy bodies return `Err(Unsupported("nested view::...calls not supported yet"))` — this is a known gap noted in `emit_view.rs:2288`.

#### `@materialized` view — fold body
- **Parse / resolve**: ✅ `FoldHandler`, `on_event`, `initial`, `clamp` all parse.
- **Emit**: ✅ `emit_view.rs` emits pair-map, per-entity-topk, symmetric-pair-topk, per-entity-ring storage shapes.
- **Fold body operator set** (spec §2.3): ⚠️ The `ResolveError::UdfInViewFoldBody` error is **not implemented**. The spec says user-defined helper calls, cross-view composition, and unbounded loops inside fold bodies must be rejected at compile time. The resolver walks fold bodies but has no check that rejects these. The error variant may not even exist in `resolve_error.rs`.

#### `@decay` annotation on `@materialized`
- **Parse / resolve**: ✅ `DecayHint` populated in `ViewIR`.
- **Emit**: ✅ `emit_view.rs` generates anchor-pattern storage with `RATE.powi(dt)` decay. `PerEntityTopK(K=1)` + `@decay` is explicitly rejected (`Unsupported`).

#### `@spatial` query declaration (`query` keyword)
- **Parse / resolve**: ✅ `QueryDecl` with `sort_by`, `limit`, `body`.
- **Emit**: ❌ No `emit_query.rs`. `QueryDecl` items in `comp.queries`... wait — `Compilation` has no `queries` field. Looking at `Decl::Query` in the parser: `Decl::Query(QueryDecl)` exists but `Compilation` in `ir.rs` has no `queries: Vec<QueryIR>` field. The `@spatial` query surface is fully **dropped after parsing** — resolve pass does not process `Decl::Query` items.

#### View annotations
- `@indexed_on(field)`: ❌ Annotation is accepted by the generic annotation parser but the resolver has no interpretation for it.
- `@top_k(K)`: ❌ Same — parsed generically, not resolved/emitted.
- `@backend(cpu | gpu)`: ❌ Same.
- `@fidelity(>= Medium)`: ❌ Same — parsed as `AnnotationValue::Comparator` but never consumed by resolver or emitters.

### §2.4 `physics` cascade rules

- **Parse / resolve**: ✅ `PhysicsDecl`, `@phase(pre|event|post)`, `@cpu_only`, `on EventPattern where expr`, `emit`, `for`, `if/match/let` stmts all parse and resolve.
- **Emit**: ✅ `emit_physics.rs` produces Rust cascade handlers; WGSL path via `emit_physics_wgsl.rs`.
- `@terminating_in(N)`: ✅ Recognised in the cycle-detection validator (resolve.rs:3374).
- `@before(Other)` / `@after(Other)`: ❌ Not recognised or enforced. No ordering constraint is emitted. Accepted silently by the generic annotation parser.
- **Race detection** (spec §2.4 item 3): ❌ Not implemented. The resolver has no check that two unordered same-phase rules writing the same field via non-commutative updates raises an error.
- **Schema-hash drift on event/field refs** (spec §2.4 item 5): ✅ Covered by `schema_hash::rules_hash`.

### §2.5 `mask` predicates

- **Parse / resolve**: ✅ `MaskDecl`, `ActionHead`, `Positional`/`Named`/`None` shapes, typed params (`Cast(ability: AbilityId)` — task 157), `from <expr>` candidate source.
- **Emit**: ✅ `emit_mask.rs` emits Rust predicate fn + WGSL via `emit_mask_wgsl.rs`.
- **GPU-kernel restriction** (spec §2.5 — mask predicates with only scalar intrinsics compile to SPIR-V): ⚠️ `emit_mask_wgsl.rs` exists but coverage is partial. Cross-entity predicates (`at_war(self.faction, f)`) are specified to be "CPU-patched" — no such patching is implemented; they simply fail WGSL emission.
- `explanation kernel` (`trace_mask`): ❌ Not emitted.

### §2.6 `verb` declaration

- **Parse / resolve**: ✅ `VerbDecl` → `VerbIR` fully resolved.
- **Emit**: ❌ No emit path. `EmittedArtifacts` has no `rust_verb_modules`. `VerbIR` is carried in `Compilation` but never lowered to mask expansion + cascade handler + scoring entry.

### §2.8 `invariant` declaration

- **Parse / resolve**: ✅ `InvariantDecl` (Static/Runtime/DebugOnly) → `InvariantIR` resolved.
- **Emit**: ❌ No emit path. `EmittedArtifacts` has no invariant output. Neither compile-time static checks nor runtime assertion fn are generated.
- **`@must_preserve` cascade annotation** (spec §2.8): ❌ Not enforced.

### §2.9 `probe` declaration

- **Parse / resolve**: ✅ `ProbeDecl` → `ProbeIR` with `count[...]`, `pr[...|...]`, `mean[...|...]` assert forms fully resolved.
- **Emit**: ❌ No emit path. Probes compile to the IR but produce no test fixtures, trajectory-query stubs, or CI harness integration.

### §2.11 `metric` declaration

- **Parse / resolve**: ✅ `MetricBlock` → `MetricIR` with `window`, `emit_every`, `conditioned_on`, `alert_when`.
- **Emit**: ❌ No emit path. `histogram()`/`gauge()`/`counter()` wrappers remain as `UnresolvedCall` in the IR (spec `stdlib.md §"Metric wrappers"` explicitly acknowledges this as technical debt).

### §3.2/3.3 Action vocabulary

- **Scoring entries** (`ScoringDecl`): ✅ Parse, resolve, emit via `emit_scoring.rs`.
- **`per_unit` gradient modifiers** (spec §3.4): ✅ Parse (`ExprKind::PerUnit`), resolve (`IrExpr::PerUnit`), emit as `KIND_GRADIENT` modifier row in scoring.
- **`per_ability` rows** (not in spec but in AST/IR): ✅ `PerAbilityRow` / `PerAbilityRowIR` parse, resolve, emit.
- **Macro action heads** — scoring entries referencing `PostQuest{type=Conquest}` etc.: ✅ `action_head_discriminant` in `emit_scoring.rs` handles named-head shapes.
- **`InviteToGroup`, `AcceptInvite`, `WithdrawQuest`, `SetStanding`** as scoring heads: ⚠️ These macro heads are defined in spec §3.3 but `action_head_discriminant` in `emit_scoring.rs` only maps a subset of heads to discriminants. If an unrecognised head is used in a scoring entry it raises `EmitError::UnknownActionHead`. Not all macro variants have discriminant assignments verified.

### §4 Schema versioning

- **Emit**: ✅ `schema_hash.rs` emits `state_hash`, `event_hash`, `rules_hash`, `scoring_hash`, `config_hash`, `enums_hash`, `views_hash`, `combined_hash` per spec §4.
- **`@since` annotations / migration tables**: ✅ Explicitly not implemented per spec; correct.

### §5 Type system

- **Scalar / vector / ID types**: ✅ All seeded in `stdlib::seed`. `InviteId`, `StructureRef`, `PredicateId`, `TagId`, `ArchetypeId`, `RoleTag` from spec §5.1 are **not** in `IrType` or `stdlib_types` — they remain as `IrType::Named("...")` fallbacks if used.
- **Bounded collections** (`Bitset<N>`, `SortedVec<T,K>`, `RingBuffer<T,K>`, `Map<K,V,Cap>`, `OneOf<K,V>`): ✅ `TypeKind::Generic` parses all of these. `IrType::SortedVec`, `RingBuffer`, `SmallVec`, `Array` exist; `Bitset` and `Map` and `OneOf` fall through to `IrType::Named` but parse correctly.
- **`String` on replayable events** (forbidden per §5.2): ❌ Not enforced in the resolver.
- **Unbounded `Vec<T>` on entity/event** (forbidden per §5.2): ❌ Not enforced in the resolver.

---

## Stdlib findings (per namespace)

### Primitive functions

| Function | Spec §ref | Resolver | Emit (physics) | Emit (view) | Emit (scoring) | Notes |
|---|---|---|---|---|---|---|
| `distance` | §Spatial | ✅ `Builtin::Distance` | ✅ | ✅ | ⚠️ see note | scoring uses `lower_view_expr` path; no direct dispatch found |
| `planar_distance` | §Spatial | ✅ `Builtin::PlanarDistance` | ✅ | ❌ no arm | ❌ | emitters missing in view+scoring |
| `z_separation` | §Spatial | ✅ `Builtin::ZSeparation` | ✅ | ❌ | ❌ | same |
| `abs` | §Numeric | ✅ | ✅ | ✅ | ⚠️ | |
| `min` | §Numeric | ✅ | ✅ | ✅ | ⚠️ | |
| `max` | §Numeric | ✅ | ✅ | ✅ | ⚠️ | |
| `clamp` | §Numeric | ✅ | ✅ | ❌ | ⚠️ | view emitter falls to catch-all |
| `floor` | §Numeric | ✅ | ✅ | ❌ | ⚠️ | |
| `ceil` | §Numeric | ✅ | ✅ | ❌ | ⚠️ | |
| `round` | §Numeric | ✅ | ✅ | ❌ | ⚠️ | |
| `ln` | §Numeric | ✅ | ❌ | ❌ | ⚠️ | physics emitter catch-all |
| `log2` | §Numeric | ✅ | ❌ | ❌ | ⚠️ | |
| `log10` | §Numeric | ✅ | ❌ | ❌ | ⚠️ | |
| `sqrt` | §Numeric | ✅ | ✅ | ❌ | ⚠️ | |
| `entity(id)` | §ID deref | ✅ | ❌ | ❌ | ⚠️ | no lowering in any emitter |
| `count` | §Aggregations | ✅ `Builtin::Count` | ✅ | ✅ | ✅ | |
| `sum` | §Aggregations | ✅ | ✅ | ✅ | ✅ | |
| `forall` | §Aggregations | ✅ | ✅ | ✅ | ✅ | |
| `exists` | §Aggregations | ✅ | ✅ | ✅ | ✅ | |
| `saturating_add` | (impl-extra) | ✅ | ✅ | ❌ | ❌ | Not in spec; physics-only |

> **Scoring emitter note**: `emit_scoring.rs` does not expose a `lower_builtin_call` function; scoring expressions that contain builtin calls are passed through `lower_modifier_body` / `lower_gradient_body` which each have their own `UnsupportedExprShape` catch-alls. Basic arithmetic ops (`+`, `-`, `*`, `/`) work; builtins in general are ⚠️ partial.

### Namespace `world`

| Field | Spec type | Resolver | All emitters |
|---|---|---|---|
| `world.tick` | `u64` | ✅ | ✅ |
| `world.seed` | `u64` | ✅ | ✅ |
| `world.n_agents_alive` | `u32` | ✅ | ✅ |

Status: ✅ fully implemented.

### Namespace `cascade`

| Field | Spec type | Resolver | Notes |
|---|---|---|---|
| `cascade.iterations` | `u32` | ✅ | |
| `cascade.phase` | `enum CascadePhase` | ✅ | |
| `cascade.max_iterations` | — (impl-extra) | ✅ `u8` | Not in spec; added for cast-depth guard |

Status: ✅ fully implemented (with one impl-extra field).

### Namespace `event`

| Field | Spec type | Resolver | Notes |
|---|---|---|---|
| `event.kind` | `EventKindId` | ✅ | |
| `event.tick` | `u64` | ✅ | |

Status: ✅ fully implemented.

### Namespace `mask`

| Field | Spec type | Resolver | Notes |
|---|---|---|---|
| `mask.rejections` | `u64` | ✅ | |

Status: ✅ fully implemented.

### Namespace `action`

| Field | Spec type | Resolver | Notes |
|---|---|---|---|
| `action.head` | `ActionHeadKind` | ✅ | |
| `action.target` | `Option<AnyId>` | ✅ | |

Status: ✅ fully implemented.

### Namespace `rng`

| Method | Spec sig | Resolver | Physics | View | Scoring |
|---|---|---|---|---|---|
| `rng.uniform(lo,hi)` | `(f32,f32)->f32` | ✅ | ✅ | ✅ | ⚠️ |
| `rng.gauss(mu,sigma)` | `(f32,f32)->f32` | ✅ | ✅ | ✅ | ⚠️ |
| `rng.coin()` | `()->bool` | ✅ | ✅ | ✅ | ⚠️ |
| `rng.uniform_int(lo,hi)` | `(i32,i32)->i32` | ✅ | ✅ | ✅ | ⚠️ |

Status: ✅ resolve + physics + view emit; ⚠️ scoring emit untested/partial catch-all.

### Namespace `query`

| Method | Spec sig | Resolver | Physics emit | Mask emit | Scoring emit | View emit |
|---|---|---|---|---|---|---|
| `query.nearby_agents(pos,r)` | `(Vec3,f32)->[AgentId]` | ✅ | ❌ | ✅ (from-clause only) | ❌ | ❌ |
| `query.within_planar(pos,r)` | `(Vec3,f32)->[AgentId]` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `query.nearby_items(pos,r)` | `(Vec3,f32)->[ItemId]` | ✅ | ❌ | ❌ | ❌ | ❌ |
| `query.nearest_hostile_to(pos,r)` | (impl-extra) | ✅ | ✅ | — | — | — |
| `query.nearest_hostile_to_or(pos,r,def)` | (impl-extra) | ✅ | ✅ | — | — | — |
| `query.nearby_kin(pos,r)` | (impl-extra) | ✅ | ✅ | — | — | — |

`query.agents_within` is referenced in `physics.sim` comments as a non-spec alias that was attempted — it does **not** exist in either the spec or the resolver. It is not a gap; it is an incorrect name that needs to be replaced with `query.nearby_agents`.

### Namespace `voxel`

| Method | Spec sig | Resolver | Physics emit | View emit | Scoring emit |
|---|---|---|---|---|---|
| `voxel.neighbors_above(pos)` | `(Vec3)->[Vec3]` | ✅ | ✅ | ❌ | ❌ |
| `voxel.neighbors_below(pos)` | `(Vec3)->[Vec3]` | ✅ | ✅ | ❌ | ❌ |
| `voxel.surface_height(x,y)` | `(f32,f32)->i32` | ✅ | ✅ | ❌ | ❌ |

### Namespace `agents`

| Method | Spec sig | Resolver name | Emit name | Status |
|---|---|---|---|---|
| `agents.alive(a)` | `(AgentId)->bool` | ✅ | ✅ all emitters | ✅ |
| `agents.pos(a)` | `(AgentId)->Vec3` | ✅ | ✅ view; ❌ physics (confirmed physics.sim:625) | ⚠️ |
| `agents.hp(a)` | `(AgentId)->f32` | ✅ | ✅ all | ✅ |
| `agents.max_hp(a)` | `(AgentId)->f32` | ✅ | ✅ all | ✅ |
| `agents.shield_hp(a)` | `(AgentId)->f32` | ✅ | ✅ physics; ❌ view/scoring | ⚠️ |
| `agents.attack_damage(a)` | `(AgentId)->f32` | ✅ | ✅ physics; ❌ view | ⚠️ |
| `agents.set_hp(a,v)` | `(AgentId,f32)->()` | ✅ | ✅ physics | ✅ |
| `agents.set_shield_hp(a,v)` | `(AgentId,f32)->()` | ✅ | ✅ physics | ✅ |
| `agents.kill(a)` | `(AgentId)->()` | ✅ | ✅ physics | ✅ |
| `agents.stun_remaining_ticks(a)` | spec name | ❌ not registered | — | 🤔 **name mismatch** — impl uses `stun_expires_at_tick` |
| `agents.set_stun_remaining_ticks(a,v)` | spec name | ❌ | — | 🤔 **name mismatch** — impl: `set_stun_expires_at_tick` |
| `agents.slow_remaining_ticks(a)` | spec name | ❌ | — | 🤔 **name mismatch** — impl: `slow_expires_at_tick` |
| `agents.set_slow_remaining_ticks(a,v)` | spec name | ❌ | — | 🤔 **name mismatch** — impl: `set_slow_expires_at_tick` |
| `agents.slow_factor_q8(a)` | `(AgentId)->i16` | ✅ | ✅ physics + view | ✅ |
| `agents.set_slow_factor_q8(a,v)` | `(AgentId,i16)->()` | ✅ | ✅ physics | ✅ |
| `agents.gold(a)` | `(AgentId)->i64` | ✅ | ✅ physics | ✅ |
| `agents.set_gold(a,v)` | `(AgentId,i64)->()` | ✅ | ✅ physics | ✅ |
| `agents.add_gold(a,d)` | `(AgentId,i64)->()` | ✅ | ✅ physics | ✅ |
| `agents.sub_gold(a,d)` | `(AgentId,i64)->()` | ✅ | ✅ physics | ✅ |
| `agents.adjust_standing(a,b,d)` | `(AgentId,AgentId,i16)->()` | ✅ | ✅ physics | ✅ |
| `agents.hunger(a)` | `(AgentId)->f32` | ✅ | ✅ physics | ✅ |
| `agents.thirst(a)` | `(AgentId)->f32` | ✅ | ✅ physics | ✅ |
| `agents.rest_timer(a)` | `(AgentId)->f32` | ✅ | ✅ physics | ✅ |
| `agents.creature_type(a)` | — (impl-extra) | ✅ view only | ✅ view; ❌ physics (physics.sim:629) | ⚠️ **not in spec** |
| `agents.is_hostile_to(a,b)` | — (impl-extra) | ✅ | ✅ view; ❌ physics | ⚠️ **not in spec** |
| `agents.engaged_with(a)` | — (impl-extra) | ✅ | ✅ physics | ⚠️ **not in spec** |
| `agents.engaged_with_or(a,d)` | — (impl-extra) | ✅ | ✅ physics | ⚠️ **not in spec** |
| `agents.set_engaged_with(a,b)` | — (impl-extra) | ✅ | ✅ physics | ⚠️ **not in spec** |
| `agents.clear_engaged_with(a)` | — (impl-extra) | ✅ | ✅ physics | ⚠️ **not in spec** |
| `agents.record_memory(...)` | — (impl-extra) | ✅ | ✅ physics | ⚠️ **not in spec** |
| `agents.cooldown_next_ready(a)` | — (impl-extra) | ✅ | ✅ physics | ⚠️ **not in spec** |
| `agents.set_cooldown_next_ready(a,v)` | — (impl-extra) | ✅ | ✅ physics | ⚠️ **not in spec** |
| `agents.record_cast_cooldowns(a,ab,now)` | — (impl-extra) | ✅ | ✅ physics | ⚠️ **not in spec** |

### Namespace `abilities` (impl-extra, not in spec)

| Method | Resolver | Physics emit | Notes |
|---|---|---|---|
| `abilities.is_known(id)` | ✅ | ✅ | |
| `abilities.cooldown_ticks(id)` | ✅ | ✅ | |
| `abilities.effects(id)` | ✅ | ✅ | |
| `abilities.known(agent,ability)` | ✅ | ✅ | mask-gate |
| `abilities.cooldown_ready(agent,ability)` | ✅ | ✅ | mask-gate |
| `abilities.on_cooldown(slot)` | ✅ | ✅ | inverted mask-gate |
| `abilities.hostile_only(ability)` | ✅ | ✅ | |
| `abilities.range(ability)` | ✅ | ✅ | |

All not in spec. Entire `abilities` namespace is impl-extra.

### Namespace `terrain` (impl-extra, not in spec)

| Method | Resolver | Physics emit |
|---|---|---|
| `terrain.line_of_sight(from,to)` | ✅ | ✅ |

Not in spec.

### Namespaces with grammar stubs only (roadmap-deferred)

All four of the following namespaces resolve correctly but every method returns `Err(EmitError::Unsupported)` from `emit_physics.rs`. They are correctly documented as "pending runtime impl" in comments.

| Namespace | Methods (all ⚠️ stub) | Spec ref |
|---|---|---|
| `membership` | `is_group_member`, `is_group_leader`, `can_join_group`, `is_outcast` | Not in stdlib.md (roadmap §1) |
| `relationship` | `is_hostile`, `is_friendly`, `knows_well` | Not in stdlib.md (roadmap §3) |
| `theory_of_mind` | `believes_knows`, `can_deceive`, `is_surprised_by` | Not in stdlib.md (roadmap §6) |
| `group` | `exists`, `is_active`, `has_leader`, `can_afford_from_treasury` | Not in stdlib.md (roadmap §7) |
| `quest` | `can_accept`, `is_target`, `party_near_destination` | Not in stdlib.md (roadmap §12) |

These **five namespaces are not in stdlib.md** but are registered in the resolver — the spec document is behind the implementation's roadmap stubs.

### Beliefs accessors (plan ToM — impl-extra, not in spec)

| Surface | Parse | Resolve | Scoring emit | View emit | Physics emit |
|---|---|---|---|---|---|
| `beliefs(obs).about(tgt).<field>` | ✅ `BeliefsAccessor` | ✅ | ✅ modifier path | ✅ partial | ⚠️ not in lower_expr_kind |
| `beliefs(obs).confidence(tgt)` | ✅ | ✅ | ✅ | ✅ partial | ⚠️ |
| `beliefs(obs).<view_name>(_)` | ✅ `BeliefsView` | ✅ | ❌ `UnsupportedExprShape` | ❌ | ❌ |
| `beliefs(obs).observe(tgt) with {...}` | ✅ `BeliefObserve` | ✅ | — (stmt) | — | ✅ T5 lowers to `agent_cold_beliefs_mut` upsert |

Not in stdlib.md spec. BeliefsView (aggregate view) is entirely unimitted.

---

## Cross-cutting observations

1. **Impl has grown beyond the spec.** The implementation registers ~15 methods not in `docs/spec/stdlib.md`: `agents.creature_type`, `agents.is_hostile_to`, `agents.engaged_with*`, `agents.record_memory`, `agents.record_cast_cooldowns`, `agents.cooldown_next_ready`, `query.nearest_hostile_to*`, `query.nearby_kin`, `abilities.*` namespace, `terrain.*` namespace, `cascade.max_iterations`. The spec document is the contract for external consumers; these undocumented methods will cause confusion for DSL authors.

2. **`@spatial query` decl silently dropped.** `Decl::Query(QueryDecl)` is parsed but `Compilation` has no `queries` field. Resolve pass has no branch for `Decl::Query`. A `query nearby_agents(...)` declaration compiles silently to nothing. This is likely intentional (user-authored queries are not yet implemented) but there is no warning or error.

3. **Fold body operator set restriction not enforced.** Spec §2.3 says user-defined helper calls, cross-view composition, and unbounded loops inside `@materialized` fold bodies must be `ResolveError::UdfInViewFoldBody`. This error is referenced in the spec but does not appear to exist in `resolve_error.rs`. Authors can write forbidden constructs in fold bodies without a compile-time error.

4. **`String` and `Vec<T>` forbidden-type rules not enforced.** Spec §5.2 prohibits both on replayable events and entity fields. The resolver accepts them silently.

5. **Verb, invariant, probe, metric emission entirely absent.** These four declaration kinds complete the round-trip parse→resolve but produce no emitted artefacts. They are milestone-deferred but represent ~35% of the top-level declaration surface area.

6. **`@before`/`@after` physics annotations silently accepted but not enforced.** Spec §2.4 defines ordering constraints. They are parsed generically but no ordering-constraint graph is built or checked. Physics rules with ordering requirements will silently run in arbitrary order.

7. **Spec stun/slow naming is stale.** `stdlib.md` documents `stun_remaining_ticks` / `slow_remaining_ticks` but task 143 changed the storage model to absolute expiry ticks and renamed the accessors in the implementation. The spec must be updated to `stun_expires_at_tick` / `slow_expires_at_tick` and the semantic changed accordingly (active = `world.tick < agents.stun_expires_at_tick(a)`).
