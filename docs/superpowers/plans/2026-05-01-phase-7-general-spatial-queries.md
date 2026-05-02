# Phase 7: General Spatial Queries — Plan Stub

> **Status: STUB.** Triggered by user observation (2026-05-01): "We want to rip kin out and support spatial hashes for other things. The target game is nothing like the wolf sim."
>
> The current IR bakes in two domain-specific spatial queries (`KinQuery`, `EngagementQuery`) that assume a faction-based combat model. This plan replaces them with a general spatial-query primitive driven by a filter expression.

**Goal:** Replace `SpatialQueryKind::{KinQuery, EngagementQuery}` with a generic `FilteredWalk { filter: CgExprId }` so the IR doesn't pre-name domain concepts. Support arbitrary boolean filters at query-time without IR changes.

**Architecture (LOCKED to option (a) — predicate-driven):**

```rust
pub enum SpatialQueryKind {
    BuildHash,                                    // generic — build per-cell index
    FilteredWalk { filter: CgExprId },            // walk neighbors, eval filter per-candidate
}
```

The filter expression is a `CgExpr` evaluated per-candidate during the walk. The expression has access to:
- `self` — the querying agent's id (kernel-preamble bound).
- `candidate` — the current neighbor under inspection (per-iteration bound, similar to `per_pair_candidate`).
- Any agent-field reads via `AgentRef::Self_` / `AgentRef::PerPairCandidate`.
- Any namespace-call expression (Task 4 of the cg-lowering-gap plan landed this — `agents.is_hostile_to(self, candidate)` works).

Per-candidate filter cost: a few WGSL instructions (bitcast + compare + branch). Negligible vs the spatial-walk loop body.

**Why option (a) over (b) tag-bitmap or (c) multi-hash:**
- (a) is the most general — any boolean combination of agent-field reads, namespace calls, and constants works without IR additions.
- (b) requires a tag system (a separate Phase) and only handles tag-based filters cheaply.
- (c) costs N× build per filter category and stops scaling at ~5 categories.
- (a) and (b) compose: when the tag system arrives, filters can read tag bitmaps via `agents.has_tag(candidate, X)`.

## Tasks (skeleton)

| # | Task | Files | LOC est. |
|---|---|---|---|
| 1 | Add `SpatialQueryKind::FilteredWalk { filter }` variant; update `dependencies()`, `name()`, snake-name. | `cg/op.rs` | ~50 |
| 2 | Generic spatial-walk body template that evaluates filter per-candidate. | `cg/emit/kernel.rs` (replaces `SPATIAL_KIN_QUERY_BODY` + `SPATIAL_ENGAGEMENT_QUERY_BODY`) | ~80 |
| 3 | Lowering: dispatch shape `PerPair { source: SpatialQuery(kind) }` recognizes the new variant; the filter expression's `CgExprId` is threaded through. | `cg/lower/spatial.rs`, `cg/lower/driver.rs::collect_required_spatial_kinds` | ~100 |
| 4 | DSL grammar: `spatial_query <name> = <filter_expr>` declaration + `@per_pair(<query_name>)` reference. Resolver maps names to filter expressions. | `dsl_ast/src/ast.rs`, `dsl_ast/src/parse.rs`, `dsl_resolver/src/...` | ~150 |
| 5 | Drop `KinQuery`, `EngagementQuery` variants; delete `SPATIAL_KIN_QUERY_BODY` / `SPATIAL_ENGAGEMENT_QUERY_BODY` const bodies. | `cg/op.rs`, `cg/emit/kernel.rs` | -100 |
| 6 | Delete legacy emitters: `emit_kin_query_*.rs`, `emit_engagement_query_*.rs`, `emit_spatial_kin_query.rs`, `emit_spatial_engagement_query.rs`. | `dsl_compiler/src/emit_*` | -300 |
| 7 | Refactor `assets/sim/masks.sim`, `assets/sim/scoring.sim` to use new spatial-query syntax. **OR** delete the wolf-sim fixtures entirely if Phase 7 ships alongside the target-game DSL. | `assets/sim/*` | varies |
| 8 | Update parity test fixture if `smoke_fixture_n4` references the old query names. | `engine_gpu/tests/common/mod.rs` | small |

**Net: ~+380 / -400 LOC**, 4-7 days.

## Architectural Impact Statement (skeleton)

- **Decision:** new IR variant `FilteredWalk { filter }`. Drops two domain-specific variants. Net IR surface decreases.
- **Rule-compiler touchpoints:** DSL grammar adds `spatial_query <name> = <expr>` declaration shape. Existing rules update.
- **Hand-written downstream code:** zero new — the filter is a CG expression, walked by the existing emitter.
- **Constitution check:**
  - P1 (Compiler-First): PASS — every change in compiler infrastructure.
  - P3 (Cross-Backend Parity): PASS — both backends evaluate the same filter expression.
  - P11 (Reduction Determinism): PASS — spatial walk order is deterministic per the existing per-cell traversal; filter evaluation is pure.
- **Runtime gate:** `parity_with_cpu --features gpu` on a target-game fixture (TBD — built in Phase 7 or carried over from Phase 6 if test scenarios survive the kin-removal).

## Dependencies

- **Phase 6 lands first.** Phase 7 needs the parity gate to validate against; Phase 6 produces a known-good CG pipeline. Phase 7 then proves that the new spatial primitive doesn't regress the gate (or, more interestingly, replaces the gate with a target-game-aligned fixture).
- **Namespace-call lowering (cg-lowering-gap-closure Task 4)**: filter expressions like `!agents.is_hostile_to(self, candidate)` already lower correctly. Done.
- **Per-pair candidate binding (Task 1)**: filter expressions read `candidate` via `AgentRef::PerPairCandidate`. Already lowered.

## Out-of-scope

- **Tag-bitmap filter system.** Mentioned in the architecture rationale as composable with option (a) but a separate effort. Phase 7 ships predicate-driven; tags can layer on later when the broader tag/effect system arrives.
- **Multi-hash partitioning.** Discarded as a non-goal — option (a) covers the use cases without N× build cost.

## References

- Triggered by user discussion 2026-05-01 in the Phase 6 implementation thread.
- Current `SpatialQueryKind` definition: `crates/dsl_compiler/src/cg/op.rs:140-150`.
- Existing per-kind body templates: `crates/dsl_compiler/src/cg/emit/kernel.rs:1561-1601`.
- Phase 6 plan stub (must land first): `docs/superpowers/plans/2026-05-01-phase-6-cg-decision-lowering.md`.
