# Phase 7: General Spatial Queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the wolf-sim-specific `SpatialQueryKind::{KinQuery, EngagementQuery}` IR variants with a generic `FilteredWalk { filter: CgExprId }`, so the spatial-query IR carries no domain naming and arbitrary boolean filters compose at query-time without IR additions. Wolf-sim DSL fixtures are refactored onto the new surface; legacy kin/engagement emitters and runtime files are deleted.

**Architecture:** Three-layer change. (1) IR: `SpatialQueryKind::FilteredWalk { filter: CgExprId }` lands additively alongside the existing `BuildHash | KinQuery | EngagementQuery` set; `compute_dependencies` walks the filter expression for additional reads. (2) Lowering: the driver routes from-bearing masks through `FilteredWalk` with a filter built from the existing predicate's "candidate-side" sub-expression (alive + hostility checks); the per-pair `target` binder reuses the established `LoweringCtx::target_local` flag. (3) DSL: a new `spatial_query <name>(self, candidate, <args>) = <filter>` declaration plus a `from spatial.<name>(...)` reference replaces `from query.nearby_agents(...)` at the two existing wolf-sim sites. Once green, the `KinQuery`/`EngagementQuery` variants drop and their legacy emitters + generated runtime files delete.

**Tech Stack:** Rust workspace; `dsl_ast` (parser + resolver), `dsl_compiler` (lowering + emit), `engine_gpu_rules` (CG-emitted output), `engine_gpu` (runtime consumer). `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical` regenerates; `cargo test --workspace` is the breadth gate; `cargo test -p engine_gpu --features gpu --test parity_with_cpu` is the runtime regression gate.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `SpatialQueryKind` enum at `crates/dsl_compiler/src/cg/op.rs:140-150` (3 variants today).
  - `SpatialQueryKind::dependencies` at `op.rs:163-203` (hard-coded `(reads, writes)` per variant).
  - `SpatialQueryKind::label` at `op.rs:206-212` (snake-case label).
  - `ComputeOpKind::compute_dependencies` arm for `SpatialQuery` at `op.rs:597-601` (delegates to `kind.dependencies()`).
  - `PerPairSource::SpatialQuery(SpatialQueryKind)` at `crates/dsl_compiler/src/cg/dispatch.rs:126-133` (dispatch shape).
  - `lower_spatial_queries` at `crates/dsl_compiler/src/cg/lower/spatial.rs:163-180` (driver-supplied list → ops).
  - `mask_spatial_kind` heuristic at `crates/dsl_compiler/src/cg/lower/driver.rs:894-902` (predicate-shape-based kin/engagement classifier).
  - `predicate_uses_engagement_relationship` at `driver.rs:917+` (deep walker for the heuristic).
  - `lower_mask`'s per-pair binding setup at `crates/dsl_compiler/src/cg/lower/mask.rs:170-177` (sets `ctx.target_local = true` for pair-bound predicate lowering — re-usable for filter lowering).
  - `LoweringCtx::target_local` flag at `crates/dsl_compiler/src/cg/lower/expr.rs:139` and `target` LocalRef → `CgExpr::PerPairCandidateId` resolution at `expr.rs:891`.
  - `validate_from_clause_shape` at `mask.rs:288-303` (recognizes `query.nearby_agents(<pos>, <radius>)` only).
  - WGSL body templates: `SPATIAL_KIN_QUERY_BODY` at `crates/dsl_compiler/src/cg/emit/kernel.rs:1660-1669`, `SPATIAL_ENGAGEMENT_QUERY_BODY` at `kernel.rs:1691-1700` — both stubs that touch every binding.
  - `lower_op_body`'s `SpatialQuery` arm at `kernel.rs:1821-1827` (per-kind body dispatch).
  - Snake-name table at `kernel.rs:3389-3390` (`(KinQuery, "spatial_kin_query")`, `(EngagementQuery, "spatial_engagement_query")`).
  - `assets/sim/masks.sim:48` (MoveToward) and `:76` (Attack) — the only two from-clause sites.
  - Legacy emitter: `crates/dsl_compiler/src/emit_spatial_kernel.rs::emit_kin_query_rs` / `emit_engagement_query_rs` (~30 LOC of stubs).
  - Generated runtime files: `crates/engine_gpu_rules/src/spatial_kin_query.{rs,wgsl}`, `spatial_engagement_query.{rs,wgsl}`.
  - Module registration: `crates/engine_gpu_rules/src/lib.rs:28,57` (`pub mod spatial_engagement_query`, `pub use SpatialEngagementQueryKernel`). Note: `spatial_kin_query` is NOT in lib.rs today — only engagement is wired through. Confirm this in Task 7.

  Search method: `rg`, `grep -n`, direct `Read` on cited files.

- **Decision:** Additive IR variant first (`FilteredWalk { filter }`), then migrate, then drop the legacy variants. Refactor (not delete) the wolf-sim fixtures — target-game DSL doesn't exist yet; deleting fixtures with no replacement breaks every downstream test. The stub's "OR delete the wolf-sim fixtures entirely if Phase 7 ships alongside the target-game DSL" path is explicitly out of scope until target-game DSL lands.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `assets/sim/masks.sim` (refactor 2 from-clause sites). New `assets/sim/spatial.sim` for the two `spatial_query` declarations (or fold into `masks.sim` — Task 5 picks).
  - DSL grammar extended: `dsl_ast/src/parse.rs` for the `spatial_query` declaration; `dsl_ast/src/ast.rs` + `ir.rs` for the AST/IR node; `dsl_ast/src/resolve.rs` for `from spatial.<name>(...)` resolution.
  - Generated outputs re-emitted: `crates/engine_gpu_rules/src/*.{rs,wgsl}` regen at every step that touches lowering.
  - Generated outputs deleted: `crates/engine_gpu_rules/src/spatial_kin_query.{rs,wgsl}`, `spatial_engagement_query.{rs,wgsl}` at Task 7.

- **Hand-written downstream code:**
  - Deleted: `crates/dsl_compiler/src/emit_spatial_kernel.rs::emit_kin_query_rs / emit_engagement_query_rs` (legacy emitter stubs, ~30 LOC).
  - Deleted: `crates/engine_gpu_rules/src/spatial_kin_query.{rs,wgsl}` and `spatial_engagement_query.{rs,wgsl}` (~80 LOC).

- **Constitution check:**
  - P1 (Compiler-First Engine Extension): PASS — every behavioral change is in the DSL compiler. No new `impl Rule` outside generated dirs.
  - P2 (Schema-Hash on Layout): N/A — no `SimState` SoA, event variant, mask predicate semantic, or scoring-row contract change. The query-results scratch buffer shape is unchanged.
  - P3 (Cross-Backend Parity): PASS — both backends route through the same lowered filter expression. New per-candidate semantics are pure CG; no backend-specific divergence introduced. Verified by Task 8 running the full `cargo test --workspace` and the parity test.
  - P4 (`EffectOp` Size Budget): N/A — no `EffectOp` change.
  - P5 (Determinism via Keyed PCG): PASS — RNG paths preserved. Filter expressions are pure functions of `(self, candidate, agent fields)`; no RNG inside filters in v1 (and if a future filter needs RNG, it routes through the existing keyed-PCG `CgExpr::Rng { purpose }` shape).
  - P6 (Events Are the Mutation Channel): PASS — spatial queries write only to scratch (`spatial_query_results`), not to events. No mutation-channel change.
  - P7 (Replayability Flagged): N/A — no event variant added.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS — every task ends with `git commit` + reference to the produced SHA in the closing entry.
  - P10 (No Runtime Panic): PASS — filter expressions lower to pure `CgExpr` nodes; the existing `lower_expr` infrastructure surfaces failures as `LoweringError`, not panics. The new well_formed walk for `FilteredWalk` validates the filter expression's `CgExprId` references the arena before emit (no dangling-id panic at WGSL emit time).
  - P11 (Reduction Determinism): PASS — spatial walk visits candidates in deterministic per-cell traversal order (already true for `KinQuery`/`EngagementQuery`); filter evaluation is pure per-candidate; output `query_results` write order preserved.

- **Runtime gate:**
  - `cargo test --workspace` — full workspace gate. All passing tests at `7d566a92` checkpoint must stay green. Task 8 is the closure gate.
  - `cargo test -p dsl_compiler --lib` — 800+ unit tests; expect new tests added per task, no regressions.
  - `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical` — must regen successfully (no lowering panics, no naga-validation failures on emitted WGSL) at every task that touches lowering.
  - `cargo test -p engine_gpu --features gpu --test parity_with_cpu` — pre-existing RED state (smoke fixture doesn't exercise pos updates correctly per Phase 8 plan). This plan does NOT change parity status; the gate here is "no further regressions" — what was RED stays RED with the same divergence shape, what was GREEN stays GREEN.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill, 2026-05-02 expansion of 2026-05-01 stub). [ ] AIS reviewed post-design (after task list stabilises).

## Out-of-scope

- **Target-game DSL.** The stub mentions "rebuild fixtures around the target game" as an option. No target-game DSL exists today; this plan refactors the wolf-sim fixtures onto the new surface but does not design or migrate to a target-game DSL. That is a separate plan, picked up after Phase 8.
- **Tag-bitmap filter system.** Mentioned in the original stub as compositionally orthogonal to predicate-driven filters; tracked as a future layer (filter expressions can read `agents.has_tag(candidate, X)` once a tag system exists, no IR change needed).
- **Multi-hash partitioning** (one hash per filter category). Discarded as a non-goal — predicate-driven covers the use cases without N× build cost.
- **Filter-expression deduplication** (two masks with identical filters sharing one `FilteredWalk` op). The CG schedule's existing dedup machinery handles this if the filter expressions hash equal; if they don't (different `CgExprId`s for structurally identical expressions), each mask gets its own walk. Optimisation; not required for correctness.
- **`spatial_query` arity beyond `(self, candidate)` plus value args.** No multi-candidate filters (e.g., "this candidate's neighbour's neighbour"). The walk model is one candidate at a time.

## File Structure

This plan touches existing files plus one new DSL file (`assets/sim/spatial.sim`).

- Modify: `crates/dsl_compiler/src/cg/op.rs` — add `SpatialQueryKind::FilteredWalk { filter }` variant, extend `dependencies()`/`label()`, extend `compute_dependencies` walker.
- Modify: `crates/dsl_compiler/src/cg/emit/kernel.rs` — add `SPATIAL_FILTERED_WALK_BODY` template, extend `lower_op_body`'s SpatialQuery arm, extend snake-name table.
- Modify: `crates/dsl_compiler/src/cg/lower/spatial.rs` — accept `FilteredWalk` kinds (lowering pass already takes `&[SpatialQueryKind]` opaquely; only test additions needed).
- Modify: `crates/dsl_compiler/src/cg/lower/driver.rs` — extend `mask_spatial_kind` to produce `FilteredWalk { filter }` when the new DSL surface is used; keep `KinQuery`/`EngagementQuery` arms while the v1 surface co-exists.
- Modify: `crates/dsl_compiler/src/cg/lower/mask.rs` — extend `validate_from_clause_shape` to recognize `spatial.<name>(...)` references in addition to `query.nearby_agents`.
- Modify: `crates/dsl_ast/src/ast.rs` — add `Decl::SpatialQuery { name, params, filter }` AST node.
- Modify: `crates/dsl_ast/src/parse.rs` — parse `spatial_query <name>(...) = <expr>` declarations.
- Modify: `crates/dsl_ast/src/ir.rs` — add `SpatialQueryDecl` IR record.
- Modify: `crates/dsl_ast/src/resolve.rs` — register `spatial_query` decls in a registry; resolve `from spatial.<name>(...)` references against the registry; produce a resolved filter expr for the lowering layer.
- Create: `assets/sim/spatial.sim` — declares `nearby_alive_other(self, candidate, radius: f32)` + `nearby_hostile_in_melee(self, candidate, radius: f32)` (the two filters wolf-sim's existing `MoveToward`/`Attack` masks need).
- Modify: `assets/sim/masks.sim` — switch `MoveToward` and `Attack` from `query.nearby_agents` to `spatial.<name>` references; remove the redundant per-pair predicate fragments now subsumed by the spatial filter.
- Delete: `crates/dsl_compiler/src/emit_spatial_kernel.rs::emit_kin_query_rs / emit_engagement_query_rs` functions (the file may stay for `emit_build_hash_rs` — Task 7 picks).
- Delete: `crates/engine_gpu_rules/src/spatial_kin_query.{rs,wgsl}`, `spatial_engagement_query.{rs,wgsl}` (after regen produces nothing referencing them).
- Modify: `crates/engine_gpu_rules/src/lib.rs` — drop the `spatial_engagement_query` module declaration + `pub use` (and `spatial_kin_query` if present after regen).

---

## Task 1: Add `SpatialQueryKind::FilteredWalk { filter }` variant (additive)

The variant lands additively alongside `BuildHash | KinQuery | EngagementQuery` so the rest of the codebase compiles continuously while migration proceeds. The variant carries a `CgExprId` filter; `dependencies()` returns the same `(grid_cells + grid_offsets) → query_results` signature as the legacy walks (the filter's expression-level reads are added by the auto-walker `compute_dependencies`, NOT duplicated here).

**Files:**
- Modify: `crates/dsl_compiler/src/cg/op.rs` — variant + `dependencies()` arm + `label()` arm.
- Modify: `crates/dsl_compiler/src/cg/op.rs` — `compute_dependencies` arm for `SpatialQuery` walks the filter for `FilteredWalk` variant.
- Test: `crates/dsl_compiler/src/cg/op.rs::tests` — variant snapshot, dependencies match, filter reads propagate.

- [ ] **Step 1: Read the current variant + table to lock in shape**

Run: `Read crates/dsl_compiler/src/cg/op.rs offset=140 limit=80`. Confirm the enum starts at line 140 with `pub enum SpatialQueryKind { ... }` and the dependencies table is at lines 163-203. If the line numbers have drifted (any commit since `7d566a92`), use `rg "pub enum SpatialQueryKind"` to relocate.

- [ ] **Step 2: Write the failing test for `FilteredWalk` dependencies signature**

Append to the existing `mod tests` block in `crates/dsl_compiler/src/cg/op.rs` (find `// ---- SpatialQueryKind ----` around line 839 and add this test below the existing `KinQuery`/`EngagementQuery` dependency tests):

```rust
#[test]
fn filtered_walk_dependencies_match_legacy_walk_signature() {
    use crate::cg::data_handle::{CgExprId, SpatialStorageKind};
    // FilteredWalk's static (reads, writes) match the legacy walk
    // shape: reads grid_cells + grid_offsets, writes query_results.
    // Per-filter additional reads come from compute_dependencies
    // walking the filter expression — NOT from this static table.
    let kind = SpatialQueryKind::FilteredWalk {
        filter: CgExprId(0),
    };
    let (r, w) = kind.dependencies();
    assert_eq!(
        r,
        vec![
            DataHandle::SpatialStorage {
                kind: SpatialStorageKind::GridCells,
            },
            DataHandle::SpatialStorage {
                kind: SpatialStorageKind::GridOffsets,
            },
        ]
    );
    assert_eq!(
        w,
        vec![DataHandle::SpatialStorage {
            kind: SpatialStorageKind::QueryResults,
        }]
    );
}

#[test]
fn filtered_walk_label_includes_filter_id() {
    use crate::cg::data_handle::CgExprId;
    let kind = SpatialQueryKind::FilteredWalk {
        filter: CgExprId(7),
    };
    assert_eq!(kind.label(), "filtered_walk(filter=#7)");
}
```

- [ ] **Step 3: Run test, expect FAIL with "no variant FilteredWalk"**

Run: `cargo test -p dsl_compiler --lib filtered_walk_dependencies_match_legacy_walk_signature`
Expected: compile error — `no variant or associated item named FilteredWalk found for enum SpatialQueryKind`.

- [ ] **Step 4: Add the variant**

In `crates/dsl_compiler/src/cg/op.rs`, modify the enum declaration (around line 140-150) to add the variant:

```rust
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpatialQueryKind {
    /// Build the per-cell agent index — reads agents (positions),
    /// writes the cell + offset arrays.
    BuildHash,
    /// Per-agent kin-of-team neighborhood walk — reads the grid,
    /// writes the per-agent query-results scratch.
    ///
    /// **DEPRECATED**: superseded by `FilteredWalk { filter }` once
    /// the wolf-sim DSL fixtures migrate (Task 5). Variant remains
    /// in the IR through Phase 7 to keep the additive migration
    /// gate green; dropped in Task 6.
    KinQuery,
    /// Per-agent engagement-target neighborhood walk — reads the grid,
    /// writes the per-agent query-results scratch.
    ///
    /// **DEPRECATED**: see `KinQuery`.
    EngagementQuery,
    /// Per-agent neighborhood walk filtered by a per-candidate
    /// boolean expression. The filter is a `CgExprId` evaluated
    /// per-candidate at WGSL emit time; the expression has access
    /// to `self` (the querying agent) and the per-pair
    /// `candidate` (bound via the same `LoweringCtx::target_local`
    /// flag the per-pair mask predicate uses). Replaces the
    /// domain-specific `KinQuery`/`EngagementQuery` variants;
    /// see `docs/superpowers/plans/2026-05-01-phase-7-general-spatial-queries.md`.
    FilteredWalk { filter: CgExprId },
}
```

Add the `use` if not already present (check the top of `op.rs`):

```rust
use crate::cg::data_handle::CgExprId;
```

(`CgExprId` is already used elsewhere in this file via `crate::cg::data_handle::CgExprId` — search with `grep -n "CgExprId" crates/dsl_compiler/src/cg/op.rs` to confirm; if not present at the top, add to the existing `use crate::cg::data_handle::{...}` line.)

- [ ] **Step 5: Add the `dependencies()` arm**

In `crates/dsl_compiler/src/cg/op.rs::SpatialQueryKind::dependencies` (around line 163-203), add an arm for `FilteredWalk` matching the existing `KinQuery`/`EngagementQuery` shape (filter's expr-level reads are added by `compute_dependencies`, not duplicated here):

```rust
            SpatialQueryKind::FilteredWalk { filter: _ } => (
                vec![
                    DataHandle::SpatialStorage {
                        kind: SpatialStorageKind::GridCells,
                    },
                    DataHandle::SpatialStorage {
                        kind: SpatialStorageKind::GridOffsets,
                    },
                ],
                vec![DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::QueryResults,
                }],
            ),
```

- [ ] **Step 6: Add the `label()` arm**

In `crates/dsl_compiler/src/cg/op.rs::SpatialQueryKind::label` (around line 206-212), the function returns `&'static str` today — that won't fit a per-filter label without leaking. Change the signature to return `String` (audit callers first):

```bash
grep -rn "\.label()" crates/dsl_compiler/src/cg/ | grep -i spatial | head -10
```

If callers tolerate `String` (most do — `format!("{}", kind)` triggers Display, which today writes via `f.write_str(self.label())`), update both `label()` and the `Display` impl:

```rust
    pub fn label(&self) -> String {
        match self {
            SpatialQueryKind::BuildHash => String::from("build_hash"),
            SpatialQueryKind::KinQuery => String::from("kin_query"),
            SpatialQueryKind::EngagementQuery => String::from("engagement_query"),
            SpatialQueryKind::FilteredWalk { filter } => {
                format!("filtered_walk(filter=#{})", filter.0)
            }
        }
    }
```

```rust
impl fmt::Display for SpatialQueryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}
```

If callers depend on `&'static str` (e.g., a const table), keep `label()` returning `&'static str` for the legacy variants and add a separate `display_label(&self) -> String` for `FilteredWalk`. The simpler path is to migrate callers to `String`; verify with the grep above and pick.

- [ ] **Step 7: Run tests for the additions**

Run: `cargo test -p dsl_compiler --lib spatial_query`
Expected: the two new tests PASS. The pre-existing snapshot tests (`snapshot_kin_query_op_display`, `snapshot_build_hash_op_display`) still PASS. If a snapshot test fails because `Display`'s output shape changed, the assertion line is the only thing affected — the format strings should be identical (both go through `label()`).

- [ ] **Step 8: Extend `compute_dependencies` to walk the filter expression for `FilteredWalk`**

In `crates/dsl_compiler/src/cg/op.rs::ComputeOpKind::compute_dependencies` (the `SpatialQuery` arm at line 597-601), replace with:

```rust
            ComputeOpKind::SpatialQuery { kind } => {
                let (r, w) = kind.dependencies();
                reads.extend(r);
                writes.extend(w);
                // FilteredWalk additionally reads whatever the filter
                // expression touches (agent fields, view storage,
                // namespace-resolved values). Walk the filter expr
                // for those reads so the schedule's BGL synthesis
                // sees them.
                if let SpatialQueryKind::FilteredWalk { filter } = kind {
                    collect_expr_reads(*filter, exprs, &mut reads);
                }
            }
```

`collect_expr_reads` is the same helper already used by `MaskPredicate`/`ScoringArgmax` arms above; verify it's in scope (`grep -n "fn collect_expr_reads" crates/dsl_compiler/src/cg/op.rs`).

- [ ] **Step 9: Add a test that filter reads propagate**

Append to the same `mod tests` block:

```rust
#[test]
fn filtered_walk_op_includes_filter_expr_reads() {
    use crate::cg::data_handle::{AgentFieldId, AgentRef, DataHandle};
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::CgExpr;
    use crate::cg::program::CgProgramBuilder;
    use dsl_ast::ast::Span;

    // Build a filter expression that reads `agent_pos` for the
    // PerPairCandidate. The op's reads should include both the
    // static spatial-storage reads AND the agent_pos read from
    // the filter walk.
    let mut builder = CgProgramBuilder::new();
    let filter_id = builder
        .push_expr(
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::PerPairCandidate,
            }),
            Span::dummy(),
        )
        .expect("filter pushes");

    let op_id = builder
        .add_op(
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::FilteredWalk { filter: filter_id },
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .expect("op pushes");

    let prog = builder.finish();
    let op = &prog.ops[op_id.0 as usize];
    assert!(op.reads.contains(&DataHandle::AgentField {
        field: AgentFieldId::Pos,
        target: AgentRef::PerPairCandidate,
    }), "filter's agent_pos read should propagate to op.reads, got {:?}", op.reads);
    assert!(op.reads.iter().any(|h| matches!(
        h,
        DataHandle::SpatialStorage { kind: SpatialStorageKind::GridCells }
    )), "static grid_cells read still present");
}
```

The `push_expr` helper name may differ — check `crates/dsl_compiler/src/cg/program.rs` for the actual builder method (search `grep -n "fn push_expr\|fn add_expr" crates/dsl_compiler/src/cg/program.rs`). Adapt the test to use the exact name; the rest of the assertion logic is unchanged.

- [ ] **Step 10: Run all SpatialQueryKind + ComputeOp tests**

Run: `cargo test -p dsl_compiler --lib`
Expected: ALL pass. The new variant compiles and exhaustive-match arms surface as compile errors at the next sites that need them — that's Task 2's signal.

- [ ] **Step 11: Commit**

```bash
git add crates/dsl_compiler/src/cg/op.rs
git commit -m "feat(dsl_compiler): add SpatialQueryKind::FilteredWalk { filter } variant (Phase 7 Task 1)"
```

Record the commit SHA: `git rev-parse HEAD`.

---

## Task 2: Generic `SPATIAL_FILTERED_WALK_BODY` WGSL template + emit dispatch

The new variant needs an exhaustive arm in `lower_op_body`'s `SpatialQuery` match, plus a new template constant. The template walks the per-cell neighborhood, evaluates the lowered filter WGSL per candidate, and writes accepted candidates into `query_results`. Today's `KinQuery`/`EngagementQuery` templates are stubs that touch every binding — the new template is also a stub at this task (real per-cell walk lands in a separate plan when the runtime BGL wiring matures), but it threads the lowered filter expression through so the structural shape is right.

**Files:**
- Modify: `crates/dsl_compiler/src/cg/emit/kernel.rs` — add `SPATIAL_FILTERED_WALK_BODY` const + extend `lower_op_body` SpatialQuery arm + extend snake-name table.
- Test: `crates/dsl_compiler/src/cg/emit/kernel.rs::tests` — emitted body contains the lowered filter WGSL; snake-name resolves.

- [ ] **Step 1: Write the failing test for the emit shape**

Append to `crates/dsl_compiler/src/cg/emit/kernel.rs::tests` (find the existing spatial body tests via `grep -n "fn snapshot_spatial\|spatial_kin_query.*Display\|SpatialEngagementQuery" crates/dsl_compiler/src/cg/emit/kernel.rs`):

```rust
#[test]
fn filtered_walk_emit_threads_filter_wgsl_into_body() {
    use crate::cg::data_handle::{AgentFieldId, AgentRef, DataHandle};
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::CgExpr;
    use crate::cg::op::{ComputeOpKind, SpatialQueryKind};
    use crate::cg::program::CgProgramBuilder;
    use dsl_ast::ast::Span;

    let mut builder = CgProgramBuilder::new();
    // Filter: PerPairCandidate.alive (reads agent_alive[per_pair_candidate]).
    let filter_id = builder
        .push_expr(
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::PerPairCandidate,
            }),
            Span::dummy(),
        )
        .expect("push filter");
    let op_id = builder
        .add_op(
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::FilteredWalk { filter: filter_id },
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .expect("push op");
    let prog = builder.finish();

    let ctx = EmitCtx::structural(&prog);
    let op = &prog.ops[op_id.0 as usize];
    let body = lower_op_body(op, &op.shape, &ctx).expect("emit body");

    // Body should reference the per-cell walk preamble + a filter
    // evaluation site that names the candidate.
    assert!(
        body.contains("for (var cell"),
        "filtered-walk body must include per-cell walk loop, got: {body}"
    );
    assert!(
        body.contains("agent_alive["),
        "filter (alive read) must lower into the body, got: {body}"
    );
    assert!(
        body.contains("query_results["),
        "body must write into query_results, got: {body}"
    );
}
```

The exact `EmitCtx::structural` constructor name + signature is in `crates/dsl_compiler/src/cg/emit/kernel.rs`; locate via `grep -n "impl.*EmitCtx\|fn structural\|EmitCtx::" crates/dsl_compiler/src/cg/emit/kernel.rs` and adapt if the constructor has moved.

- [ ] **Step 2: Run test, expect FAIL**

Run: `cargo test -p dsl_compiler --lib filtered_walk_emit_threads_filter_wgsl_into_body`
Expected: compile error — `lower_op_body` match is non-exhaustive over `SpatialQueryKind` after Task 1's variant addition.

- [ ] **Step 3: Add the `SPATIAL_FILTERED_WALK_BODY` template + `lower_op_body` arm**

In `crates/dsl_compiler/src/cg/emit/kernel.rs`, immediately after the `SPATIAL_ENGAGEMENT_QUERY_BODY` const (around line 1700), add:

```rust
/// WGSL body template for [`SpatialQueryKind::FilteredWalk`].
///
/// Walks the per-cell neighborhood for each agent, evaluates the
/// lowered filter expression per candidate, writes accepted
/// candidates into `query_results`. The filter WGSL is interpolated
/// into the `{filter_wgsl}` slot via [`format!`] in
/// [`spatial_filtered_walk_body`].
///
/// Bindings consumed (per `SpatialQueryKind::FilteredWalk` static
/// dependencies + filter walk's collected reads):
/// - `spatial_grid_cells`        (Pool, read)
/// - `spatial_grid_offsets`      (Pool, read)
/// - `spatial_query_results`     (Pool, read_write)
/// - `agent_*`                   (per-field reads collected from the filter expression)
/// - `cfg`                       (uniform, agent_cap)
///
/// # Limitations
///
/// - **Stub per-cell walk.** Mirrors the structural shape of the
///   legacy `engine_gpu_rules/src/spatial_kin_query.wgsl` until
///   runtime BGL wiring matures. The per-cell + per-candidate
///   iteration is structural; the filter WGSL is real and the
///   accept-write is real, but agent_pos lookups + radius checks
///   are not yet emitted at this task.
/// - **No per-cell radius bounds.** The walk visits every cell
///   in the grid — quadratic in agent count for the smoke fixture.
///   Acceptable for v1; bounded radius lookup is a follow-up.
fn spatial_filtered_walk_body(filter_wgsl: &str) -> String {
    format!(
        "// SpatialQuery::FilteredWalk — per-cell walk + per-candidate filter.\n\
         // Touches every binding so naga keeps them live; structural\n\
         // walk shape mirrors the legacy spatial_kin_query.wgsl stub.\n\
         var write_cursor: u32 = 0u;\n\
         for (var cell: u32 = 0u; cell < cfg.agent_cap; cell = cell + 1u) {{\n\
         \x20   let cell_start = spatial_grid_offsets[cell];\n\
         \x20   let cell_end = spatial_grid_offsets[cell + 1u];\n\
         \x20   for (var slot: u32 = cell_start; slot < cell_end; slot = slot + 1u) {{\n\
         \x20       let candidate = spatial_grid_cells[slot];\n\
         \x20       let filter_value: bool = {filter_wgsl};\n\
         \x20       if (filter_value) {{\n\
         \x20           spatial_query_results[write_cursor] = candidate;\n\
         \x20           write_cursor = write_cursor + 1u;\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n\
         _ = cfg.agent_cap;",
        filter_wgsl = filter_wgsl
    )
}
```

In the same file, modify `lower_op_body`'s `SpatialQuery` arm (around line 1821-1827) to add a `FilteredWalk` arm:

```rust
        ComputeOpKind::SpatialQuery { kind } => Ok(match kind {
            SpatialQueryKind::BuildHash => SPATIAL_BUILD_HASH_BODY.to_string(),
            SpatialQueryKind::KinQuery => SPATIAL_KIN_QUERY_BODY.to_string(),
            SpatialQueryKind::EngagementQuery => {
                SPATIAL_ENGAGEMENT_QUERY_BODY.to_string()
            }
            SpatialQueryKind::FilteredWalk { filter } => {
                let filter_wgsl = lower_cg_expr_to_wgsl(*filter, ctx)?;
                spatial_filtered_walk_body(&filter_wgsl)
            }
        }),
```

- [ ] **Step 4: Add a snake-name entry**

In `crates/dsl_compiler/src/cg/emit/kernel.rs`, find the snake-name table (line 3389 area, `(SpatialQueryKind::KinQuery, "spatial_kin_query")`). The table is hard-coded against the legacy variants; for `FilteredWalk { filter }`, the snake name needs the filter id baked in to disambiguate distinct walks. Replace the static table with a function:

```bash
grep -n "spatial_kin_query\|spatial_engagement_query\|fn .*snake_name\|fn .*kernel_name" crates/dsl_compiler/src/cg/emit/kernel.rs | head -20
```

Find the function (often called `spatial_query_snake_name` or inlined in `kernel_name_for_op`); adapt to:

```rust
fn spatial_query_snake_name(kind: &SpatialQueryKind) -> String {
    match kind {
        SpatialQueryKind::BuildHash => String::from("spatial_build_hash"),
        SpatialQueryKind::KinQuery => String::from("spatial_kin_query"),
        SpatialQueryKind::EngagementQuery => String::from("spatial_engagement_query"),
        SpatialQueryKind::FilteredWalk { filter } => {
            format!("spatial_filtered_walk_{}", filter.0)
        }
    }
}
```

If no such function exists today (the table is inlined in a callsite), introduce it as part of this task — single source of truth for snake names — and update the callsites.

- [ ] **Step 5: Run the test, expect PASS**

Run: `cargo test -p dsl_compiler --lib filtered_walk_emit_threads_filter_wgsl_into_body`
Expected: PASS. If the body shape assertions fail, adjust the template's literal strings to match (the test asserts substrings, not byte-equal output).

- [ ] **Step 6: Run the entire emit module's tests**

Run: `cargo test -p dsl_compiler --lib emit::kernel`
Expected: ALL pass. The pre-existing `spatial_kin_query`/`spatial_engagement_query` snapshot tests still pass — the new variant doesn't perturb their templates.

- [ ] **Step 7: Commit**

```bash
git add crates/dsl_compiler/src/cg/emit/kernel.rs
git commit -m "feat(dsl_compiler): SPATIAL_FILTERED_WALK_BODY template + emit dispatch (Phase 7 Task 2)"
```

---

## Task 3: Lowering wires `FilteredWalk` through dispatch + filter binding

The lowering pass `lower_spatial_queries` already takes `&[SpatialQueryKind]` opaquely — passing `FilteredWalk { filter }` works without modification. This task adds:

1. A test confirming the existing pass round-trips `FilteredWalk` correctly.
2. A `lower_filter_expr` helper invoked from `lower_mask` (or a new sibling) that builds the filter `CgExprId` from a candidate-side filter expression with `ctx.target_local = true` (so `target` resolves to `CgExpr::PerPairCandidateId`).
3. A well_formed walk that validates `FilteredWalk { filter }`'s `filter` references a Bool-typed expression in the arena.

**Files:**
- Modify: `crates/dsl_compiler/src/cg/lower/spatial.rs` — add a test for `FilteredWalk` round-trip; no production code change (pass takes the kind opaquely).
- Modify: `crates/dsl_compiler/src/cg/lower/driver.rs` — add a `lower_filter_for_mask` helper that produces a `CgExprId` for a filter expression with `target_local = true`.
- Modify: `crates/dsl_compiler/src/cg/well_formed.rs` — add a `FilteredWalkFilterNotBool` variant + check in the well_formed walk.
- Test: each modified module gets one focused test.

- [ ] **Step 1: Write the round-trip test in `lower/spatial.rs`**

Append to `crates/dsl_compiler/src/cg/lower/spatial.rs::tests`:

```rust
#[test]
fn filtered_walk_kind_round_trips_through_lowering() {
    use crate::cg::data_handle::{CgExprId, SpatialStorageKind};

    let mut builder = CgProgramBuilder::new();
    let mut ctx = LoweringCtx::new(&mut builder);

    let kind = SpatialQueryKind::FilteredWalk {
        filter: CgExprId(0),
    };
    let ids = lower_spatial_queries(&[kind], &mut ctx).expect("lowers");
    assert_eq!(ids.len(), 1);

    let prog = builder.finish();
    match prog.ops[0].kind {
        ComputeOpKind::SpatialQuery {
            kind: SpatialQueryKind::FilteredWalk { filter },
        } => assert_eq!(filter.0, 0),
        ref other => panic!("unexpected: {other:?}"),
    }
    // Static dependencies match the legacy walk shape.
    assert_eq!(
        prog.ops[0].writes,
        vec![DataHandle::SpatialStorage {
            kind: SpatialStorageKind::QueryResults
        }]
    );
}
```

- [ ] **Step 2: Run, expect PASS — the lowering pass takes the kind opaquely**

Run: `cargo test -p dsl_compiler --lib filtered_walk_kind_round_trips_through_lowering`
Expected: PASS immediately (no production change needed — the pass already handles any `SpatialQueryKind` value).

- [ ] **Step 3: Add `lower_filter_for_mask` helper in `driver.rs`**

The helper builds a filter `CgExprId` from a mask's per-pair filter expression, with `ctx.target_local = true` so the filter sees the candidate. Append to `crates/dsl_compiler/src/cg/lower/driver.rs` (above `mask_spatial_kind`, around line 887):

```rust
/// Lower a per-pair filter expression to a `CgExprId` with the
/// per-pair candidate binder (`target` LocalRef → `PerPairCandidateId`)
/// active for the duration of the lowering. Mirrors the
/// `target_local` flag toggle in [`super::mask::lower_mask`] — the
/// flag is restored before returning so a recursive lowering can't
/// leak the binding upward.
///
/// Returns the lowered `CgExprId` (the well-formed pass validates
/// the type is `Bool` separately; this helper does not).
fn lower_filter_for_mask(
    expr: &IrExprNode,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let prev = ctx.target_local;
    ctx.target_local = true;
    let result = lower_expr(expr, ctx);
    ctx.target_local = prev;
    result
}
```

Add the necessary `use` lines if not present:

```rust
use crate::cg::data_handle::CgExprId;
use super::expr::lower_expr;
use dsl_ast::ir::IrExprNode;
```

(Verify with `grep -n "use crate::cg::data_handle\|use super::expr\|use dsl_ast::ir" crates/dsl_compiler/src/cg/lower/driver.rs`; consolidate with existing `use` lines.)

- [ ] **Step 4: Write a focused test for the helper**

Append to `crates/dsl_compiler/src/cg/lower/driver.rs::tests` (find the existing test mod via `grep -n "mod tests" crates/dsl_compiler/src/cg/lower/driver.rs`):

```rust
#[test]
fn lower_filter_for_mask_binds_target_to_per_pair_candidate() {
    use crate::cg::data_handle::{AgentFieldId, AgentRef};
    use crate::cg::expr::CgExpr;
    use crate::cg::program::CgProgramBuilder;
    use dsl_ast::ir::{IrExpr, IrExprNode, LocalRef};

    // Filter expression: target.alive (LocalRef points at "target")
    let alive_field = IrExprNode {
        kind: IrExpr::Field {
            base: Box::new(IrExprNode {
                kind: IrExpr::Local(LocalRef(0), "target".to_string()),
                span: dsl_ast::ast::Span::dummy(),
            }),
            field_name: "alive".to_string(),
            field: Some(IrField::Alive), // adjust to the actual IrField variant
        },
        span: dsl_ast::ast::Span::dummy(),
    };

    let mut builder = CgProgramBuilder::new();
    let mut ctx = LoweringCtx::new(&mut builder);
    let filter_id = lower_filter_for_mask(&alive_field, &mut ctx).expect("lowers");

    let prog = builder.finish();
    let node = &prog.exprs[filter_id.0 as usize];
    // The lowered expression should resolve `target` to the per-pair
    // candidate, so reading `target.alive` lowers to a Read on
    // AgentRef::PerPairCandidate.
    match node.kind {
        CgExpr::Read(crate::cg::data_handle::DataHandle::AgentField { target, .. }) => {
            assert_eq!(target, AgentRef::PerPairCandidate);
        }
        ref other => panic!("expected Read(AgentField{{PerPairCandidate, ..}}), got {other:?}"),
    }

    // Helper restored target_local on exit.
    assert!(!ctx.target_local, "target_local should be restored");
}
```

The exact `IrField` variant + path for the `field` field on `IrExpr::Field` differs by IR version — check `crates/dsl_ast/src/ir.rs::IrExpr` and `IrField` to match. If `field: Option<IrField>` doesn't exist or has a different shape, simplify the test to use a primitive Local read instead — the assertion that matters is that `target_local` flips PerPairCandidate resolution on inside the helper.

- [ ] **Step 5: Run, fix until PASS**

Run: `cargo test -p dsl_compiler --lib lower_filter_for_mask_binds_target_to_per_pair_candidate`
Expected: PASS. If the IrField-shape mismatch trips the test, adjust the fixture to use the simplest expression that resolves through `lower_expr` (e.g., a bare `Local("target")` whose lowering produces `CgExpr::PerPairCandidateId`).

- [ ] **Step 6: Add `FilteredWalkFilterNotBool` to well_formed**

In `crates/dsl_compiler/src/cg/well_formed.rs`, find the existing `CgError` enum + walk:

```bash
grep -n "pub enum CgError\|fn walk_op\|fn check_op\|MaskPredicateNotBool" crates/dsl_compiler/src/cg/well_formed.rs | head -10
```

Add a variant alongside `MaskPredicateNotBool` (the closest analog):

```rust
    FilteredWalkFilterNotBool {
        op_index: usize,
        filter: CgExprId,
        got: CgTy,
    },
```

In the well_formed walk's `SpatialQuery` arm (search `SpatialQuery` in well_formed.rs), add:

```rust
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::FilteredWalk { filter },
            } => {
                // Filter must be a Bool-typed expression; the per-cell
                // walk uses it as the accept predicate.
                let node = prog.exprs.get(filter.0 as usize).ok_or_else(|| {
                    CgError::DanglingExprId {
                        op_index,
                        node: *filter,
                    }
                })?;
                let resolver = view_signature_resolver(prog);
                let tc_ctx = TypeCheckCtx::with_view_signature(&prog.exprs, &resolver);
                let ty = type_check(node, *filter, &tc_ctx).map_err(|e| {
                    CgError::FilteredWalkFilterTypeCheck {
                        op_index,
                        filter: *filter,
                        error: e,
                    }
                })?;
                if ty != CgTy::Bool {
                    return Err(CgError::FilteredWalkFilterNotBool {
                        op_index,
                        filter: *filter,
                        got: ty,
                    });
                }
            }
```

(`view_signature_resolver` may not exist as a separate helper — adapt to the closest pattern in `mask.rs::predicate_node_ty` which builds the resolver inline. The exact line shapes depend on the well_formed walk's signature; if it takes a `prog: &CgProgram`, all the helpers are accessible.)

If a sibling `FilteredWalkFilterTypeCheck` variant doesn't exist yet, add it too (mirror `MaskPredicateTypeCheckFailure`'s shape).

- [ ] **Step 7: Add a test for the well_formed gate**

Append to `crates/dsl_compiler/src/cg/well_formed.rs::tests`:

```rust
#[test]
fn filtered_walk_with_non_bool_filter_rejected() {
    use crate::cg::data_handle::CgExprId;
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::CgExpr;
    use crate::cg::op::{ComputeOpKind, SpatialQueryKind};
    use crate::cg::program::CgProgramBuilder;
    use dsl_ast::ast::Span;

    let mut builder = CgProgramBuilder::new();
    // Push a filter that is u32, not bool — should fail the gate.
    let filter_id = builder
        .push_expr(CgExpr::Lit(crate::cg::expr::CgLit::U32(7)), Span::dummy())
        .expect("push lit");
    builder
        .add_op(
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::FilteredWalk { filter: filter_id },
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .expect("push op");
    let prog = builder.finish();

    let result = check_well_formed(&prog);
    assert!(
        matches!(result, Err(CgError::FilteredWalkFilterNotBool { .. })),
        "expected FilteredWalkFilterNotBool, got {result:?}"
    );
}
```

- [ ] **Step 8: Run + verify all dsl_compiler tests pass**

Run: `cargo test -p dsl_compiler --lib`
Expected: ALL pass. The new well_formed test PASSes; no pre-existing test regresses.

- [ ] **Step 9: Commit**

```bash
git add crates/dsl_compiler/src/cg/lower/spatial.rs \
        crates/dsl_compiler/src/cg/lower/driver.rs \
        crates/dsl_compiler/src/cg/well_formed.rs
git commit -m "feat(dsl_compiler): wire FilteredWalk through lowering + well_formed (Phase 7 Task 3)"
```

---

## Task 4: DSL grammar — `spatial_query <name>(...) = <filter>` declaration + resolver

The DSL surface change has three pieces:

1. **Parser**: recognize `spatial_query <name>(<params>) = <expr>` as a top-level declaration.
2. **AST/IR**: add `Decl::SpatialQuery` (AST) and `SpatialQueryDecl` (resolved IR record) carrying `(name, params, filter_expr)`.
3. **Resolver**: register declarations in a `spatial_query_registry: HashMap<String, SpatialQueryDecl>`; resolve `from spatial.<name>(args)` references against the registry, producing a resolved filter expression with the args substituted in. The resolved filter is what the lowering pass (driver.rs) feeds into `FilteredWalk { filter }`.

The `params` shape is fixed: `(self, candidate, <named-value-args>)`. The first two are positional (mapped at lowering time to `Self_` / `PerPairCandidate`); the rest are typed value args (e.g., `radius: f32`) substituted from the call-site.

**Files:**
- Modify: `crates/dsl_ast/src/ast.rs` — add `Decl::SpatialQuery { name, params, filter }`.
- Modify: `crates/dsl_ast/src/parse.rs` — parse the declaration.
- Modify: `crates/dsl_ast/src/ir.rs` — add `SpatialQueryDecl { name, params, filter: IrExprNode }` + index entries.
- Modify: `crates/dsl_ast/src/resolve.rs` — registry + `from spatial.<name>` resolution.
- Test: per-module focused test for parse + resolve.

- [ ] **Step 1: Define the AST node**

In `crates/dsl_ast/src/ast.rs`, find the `Decl` enum (`grep -n "pub enum Decl" crates/dsl_ast/src/ast.rs`). Add a variant:

```rust
    /// Declares a named spatial-query filter:
    ///
    /// ```text
    /// spatial_query nearby_alive_other(self, candidate, radius: f32) =
    ///   agents.alive(candidate)
    ///   && distance(agents.pos(self), agents.pos(candidate)) < radius
    /// ```
    ///
    /// The first two params are fixed positional binders (`self` is the
    /// querying agent, `candidate` the per-pair neighbor). Remaining
    /// params are typed value args substituted at the call site (e.g.,
    /// `from spatial.nearby_alive_other(self, target, config.movement.max_move_radius)`).
    ///
    /// Lowering produces a `CgExprId` filter for `SpatialQueryKind::FilteredWalk`.
    SpatialQuery {
        name: String,
        params: Vec<SpatialQueryParam>,
        filter: ExprNode,
        span: Span,
    },
```

Add the `SpatialQueryParam` struct nearby:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialQueryParam {
    pub name: String,
    /// `None` for the fixed `self` / `candidate` binders; `Some(ty)`
    /// for typed value args (`radius: f32`, etc.).
    pub ty: Option<TypeRef>,
    pub span: Span,
}
```

(Use the existing `TypeRef` shape — `grep -n "pub.*TypeRef\|pub enum TypeRef" crates/dsl_ast/src/ast.rs` to confirm the name.)

- [ ] **Step 2: Add a parser test (failing)**

In `crates/dsl_ast/src/parse.rs::tests` (or the test module that lives near the parser), add:

```rust
#[test]
fn parse_spatial_query_decl_minimal() {
    let src = r#"
        spatial_query nearby_alive_other(self, candidate) =
          agents.alive(candidate)
    "#;
    let ast = parse(src).expect("parses");
    let decl = ast.decls.iter().find(|d| matches!(d, Decl::SpatialQuery { name, .. } if name == "nearby_alive_other"))
        .expect("declaration present");
    match decl {
        Decl::SpatialQuery { params, .. } => {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "self");
            assert_eq!(params[1].name, "candidate");
            assert!(params[0].ty.is_none());
        }
        _ => unreachable!(),
    }
}

#[test]
fn parse_spatial_query_decl_with_typed_arg() {
    let src = r#"
        spatial_query nearby_in_radius(self, candidate, radius: f32) =
          distance(agents.pos(self), agents.pos(candidate)) < radius
    "#;
    let ast = parse(src).expect("parses");
    let decl = ast.decls.iter().find(|d| matches!(d, Decl::SpatialQuery { name, .. } if name == "nearby_in_radius"))
        .expect("declaration present");
    match decl {
        Decl::SpatialQuery { params, .. } => {
            assert_eq!(params.len(), 3);
            assert_eq!(params[2].name, "radius");
            assert!(params[2].ty.is_some(), "typed arg has TypeRef");
        }
        _ => unreachable!(),
    }
}
```

The `parse` entry-point function name may differ — check the test fixtures already present in `parse.rs` for the canonical form (`grep -n "fn parse(" crates/dsl_ast/src/parse.rs`).

- [ ] **Step 3: Run, expect FAIL — parser doesn't recognize the keyword**

Run: `cargo test -p dsl_ast --lib parse_spatial_query_decl`
Expected: FAIL — the parser doesn't recognize `spatial_query` as a top-level keyword.

- [ ] **Step 4: Implement the parser**

In `crates/dsl_ast/src/parse.rs`, find the top-level decl dispatcher (e.g., `fn parse_decl` or a `match` on the leading token). Add a `spatial_query` keyword arm. The shape mirrors `view`/`mask` decl parsing — find the closest sibling and copy its skeleton:

```bash
grep -n "fn parse_view\|fn parse_mask\|\"view\"\|\"mask\"" crates/dsl_ast/src/parse.rs | head -10
```

The arm typically:
1. Consumes the `spatial_query` keyword.
2. Reads the identifier name.
3. Consumes `(`.
4. Parses comma-separated `SpatialQueryParam`s (`name` or `name: TypeRef`).
5. Consumes `)` then `=`.
6. Parses the filter as an `ExprNode` (reuse `parse_expr`).
7. Returns `Decl::SpatialQuery { name, params, filter, span }`.

Implement following the closest sibling's pattern. Pseudocode for the parser body:

```rust
fn parse_spatial_query_decl(p: &mut Parser) -> ParseResult<Decl> {
    let start = p.expect_keyword("spatial_query")?;
    let name = p.parse_ident()?;
    p.expect_symbol("(")?;
    let mut params = Vec::new();
    while !p.peek_symbol(")") {
        let param_start = p.peek_span();
        let param_name = p.parse_ident()?;
        let ty = if p.consume_symbol(":") {
            Some(p.parse_type_ref()?)
        } else {
            None
        };
        params.push(SpatialQueryParam {
            name: param_name,
            ty,
            span: param_start.merge(p.last_span()),
        });
        if !p.consume_symbol(",") {
            break;
        }
    }
    p.expect_symbol(")")?;
    p.expect_symbol("=")?;
    let filter = parse_expr(p)?;
    let end = p.last_span();
    Ok(Decl::SpatialQuery {
        name,
        params,
        filter,
        span: start.merge(end),
    })
}
```

The `Parser`/`ParseResult`/`expect_keyword`/etc. API names are codebase-specific — adapt to the existing parser style. Find a sibling decl-parse function and follow its conventions exactly.

- [ ] **Step 5: Run parser tests, fix until PASS**

Run: `cargo test -p dsl_ast --lib parse_spatial_query_decl`
Expected: both PASS.

- [ ] **Step 6: Add the resolved IR record + registry**

In `crates/dsl_ast/src/ir.rs`, add:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialQueryDecl {
    pub name: String,
    pub params: Vec<SpatialQueryParam>,
    /// Resolved filter expression. References to `self` resolve to
    /// `LocalRef(0)`, `candidate` to `LocalRef(1)`, value args to
    /// `LocalRef(2..)`. Substitution at call sites replaces the
    /// LocalRefs with the caller's argument expressions.
    pub filter: IrExprNode,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpatialQueryParam {
    pub name: String,
    pub ty: Option<IrType>,
    pub span: Span,
}
```

In `crates/dsl_ast/src/resolve.rs`, add a `spatial_queries: BTreeMap<String, SpatialQueryDecl>` field to the resolved IR's compilation root (find the existing root via `grep -n "pub struct Compilation\|pub struct ResolvedProgram" crates/dsl_ast/src/ir.rs`), and a resolution pass:

```rust
fn resolve_spatial_query_decl(
    ast: &SpatialQueryDeclAst,
    ctx: &mut ResolveCtx,
) -> Result<SpatialQueryDecl, ResolveError> {
    // Validate fixed binders.
    if ast.params.len() < 2
        || ast.params[0].name != "self"
        || ast.params[1].name != "candidate"
    {
        return Err(ResolveError::SpatialQueryRequiresSelfCandidateBinders {
            decl_name: ast.name.clone(),
            span: ast.span,
        });
    }
    // Bind LocalRef(0)=self, LocalRef(1)=candidate, LocalRef(2..)=value args.
    let mut local_scope = ctx.push_local_scope();
    local_scope.bind("self", LocalRef(0), IrType::AgentId);
    local_scope.bind("candidate", LocalRef(1), IrType::AgentId);
    for (i, param) in ast.params.iter().enumerate().skip(2) {
        let ty = resolve_type_ref(param.ty.as_ref().ok_or_else(|| {
            ResolveError::SpatialQueryValueArgRequiresType {
                decl_name: ast.name.clone(),
                arg_name: param.name.clone(),
                span: param.span,
            }
        })?, ctx)?;
        local_scope.bind(&param.name, LocalRef(i as u32), ty);
    }
    let filter = resolve_expr(&ast.filter, ctx)?;
    ctx.pop_local_scope();
    Ok(SpatialQueryDecl {
        name: ast.name.clone(),
        params: resolve_params(&ast.params),
        filter,
        span: ast.span,
    })
}
```

(Adapt names to the actual resolver API. The bind-LocalRef pattern mirrors how `view` declarations register their parameters.)

Add a registry pass to walk all `Decl::SpatialQuery` and populate `compilation.spatial_queries`.

- [ ] **Step 7: Add `from spatial.<name>(args)` recognition in `validate_from_clause_shape`**

In `crates/dsl_compiler/src/cg/lower/mask.rs::validate_from_clause_shape` (line 288-303), extend the match:

```rust
fn validate_from_clause_shape(
    mask_id: MaskId,
    source: &IrExprNode,
) -> Result<(), LoweringError> {
    match &source.kind {
        IrExpr::NamespaceCall { ns, method, args }
            if *ns == NamespaceId::Query && method == "nearby_agents" && args.len() == 2 =>
        {
            Ok(())
        }
        // New: spatial.<name>(args) reference to a registered
        // spatial_query declaration. The args are validated at
        // resolve-time against the decl's param signature.
        IrExpr::NamespaceCall { ns, .. } if *ns == NamespaceId::Spatial => Ok(()),
        _ => Err(LoweringError::UnsupportedMaskFromClause {
            mask: mask_id,
            span: source.span,
        }),
    }
}
```

Add `NamespaceId::Spatial` to the namespace enum (`grep -n "pub enum NamespaceId" crates/dsl_ast/src/ir.rs`):

```rust
pub enum NamespaceId {
    Agents,
    Abilities,
    Query,
    Spatial,  // new
    World,
    // ... existing variants ...
}
```

In `crates/dsl_ast/src/resolve.rs`, register `"spatial"` as a known namespace prefix (find the `(NamespaceId::Query, "nearby_agents") => ...` match around line 250 — the same dispatcher):

```rust
            (NamespaceId::Spatial, name) => {
                let decl = ctx.spatial_queries.get(name).ok_or_else(|| {
                    ResolveError::UnknownSpatialQuery {
                        name: name.to_string(),
                        span: call_span,
                    }
                })?;
                // Validate arg count + types.
                if args.len() + 2 != decl.params.len() {
                    return Err(ResolveError::SpatialQueryArgCountMismatch {
                        name: name.to_string(),
                        expected: decl.params.len() - 2,
                        got: args.len(),
                        span: call_span,
                    });
                }
                // Per-arg type-check is a TODO for v1 — the lowering
                // pass surfaces type errors via `lower_expr` when it
                // walks the substituted filter.
                Ok(IrExpr::NamespaceCall { ns: NamespaceId::Spatial, method: name.to_string(), args })
            }
```

- [ ] **Step 8: Add a resolver test**

In the resolver test module (find via `grep -n "mod tests" crates/dsl_ast/src/resolve.rs`):

```rust
#[test]
fn resolve_spatial_query_decl_registers_in_compilation() {
    let src = r#"
        spatial_query nearby_alive_other(self, candidate) =
          agents.alive(candidate)
    "#;
    let ast = parse(src).expect("parses");
    let comp = resolve(ast).expect("resolves");
    let decl = comp.spatial_queries.get("nearby_alive_other").expect("registered");
    assert_eq!(decl.params.len(), 2);
    assert_eq!(decl.params[0].name, "self");
    assert_eq!(decl.params[1].name, "candidate");
}

#[test]
fn resolve_spatial_query_call_resolves_against_registry() {
    let src = r#"
        spatial_query my_query(self, candidate) = agents.alive(candidate)

        mask MyMask(target)
          from spatial.my_query(self, target)
          when target != self
    "#;
    let ast = parse(src).expect("parses");
    let comp = resolve(ast).expect("resolves");
    let mask = comp.masks.iter().find(|m| m.head.name == "MyMask").expect("mask present");
    match &mask.candidate_source.as_ref().unwrap().kind {
        IrExpr::NamespaceCall { ns, method, .. } => {
            assert_eq!(*ns, NamespaceId::Spatial);
            assert_eq!(method, "my_query");
        }
        other => panic!("expected NamespaceCall(Spatial, ...), got {other:?}"),
    }
}
```

- [ ] **Step 9: Run, fix until PASS**

Run: `cargo test -p dsl_ast --lib spatial_query`
Expected: all new tests PASS. Pre-existing tests still pass.

- [ ] **Step 10: Commit**

```bash
git add crates/dsl_ast/src/ast.rs \
        crates/dsl_ast/src/parse.rs \
        crates/dsl_ast/src/ir.rs \
        crates/dsl_ast/src/resolve.rs \
        crates/dsl_compiler/src/cg/lower/mask.rs
git commit -m "feat(dsl_ast): spatial_query <name>(...) = <filter> grammar + resolver registry (Phase 7 Task 4)"
```

---

## Task 5: Refactor wolf-sim DSL — declare + use `spatial_query` for the two from-clause sites

The wolf-sim DSL has exactly two from-clause sites (`assets/sim/masks.sim:48` MoveToward, `:76` Attack). Refactor each to:

1. Pull the candidate-side filter out of the mask `when` clause into a named `spatial_query` declaration.
2. Replace the `from query.nearby_agents(...)` reference with `from spatial.<name>(...)`.

The mask's `when` clause keeps any portion of the predicate that involves the mask binder (e.g., `Cast(ability)`'s ability-side checks); only the candidate-filtering portion moves into the spatial query.

For the wolf-sim's two sites:
- `MoveToward(target)`: filter is `agents.alive(target) && target != self` — both candidate-side, both move into the `spatial_query`.
- `Attack(target)`: filter is `agents.alive(target) && is_hostile(self, target) && distance(agents.pos(self), agents.pos(target)) < 2.0` — also entirely candidate-side, all moves into the `spatial_query`.

After Task 5, both masks have `when true` (or the `when` clause drops entirely if the parser supports omitting it).

**Files:**
- Create: `assets/sim/spatial.sim` — two `spatial_query` declarations.
- Modify: `assets/sim/masks.sim` — switch `MoveToward` and `Attack` to `from spatial.<name>(...)`.
- Modify: `crates/dsl_compiler/src/cg/lower/driver.rs` — update `mask_spatial_kind` to recognize `from spatial.<name>(...)` and produce `FilteredWalk { filter: <resolved-substituted-expr> }`.
- Test: `xtask compile-dsl --cg-canonical` regen succeeds; `cargo test --workspace` doesn't regress.

- [ ] **Step 1: Create `assets/sim/spatial.sim`**

```text
// Spatial-query filter declarations.
//
// Each declaration names a per-candidate boolean filter; masks
// reference the filter by `from spatial.<name>(args)`. The first two
// params are the fixed `self` / `candidate` binders; remaining
// params are typed value args substituted from the call site.

// Used by mask MoveToward — accepts every alive agent except self.
spatial_query nearby_alive_other(self, candidate) =
  agents.alive(candidate) && candidate != self

// Used by mask Attack — accepts hostile, alive candidates within
// melee range. Distance check uses the candidate-side position read
// from the spatial walk.
spatial_query nearby_hostile_in_melee(self, candidate) =
  agents.alive(candidate)
  && is_hostile(self, candidate)
  && distance(agents.pos(self), agents.pos(candidate)) < 2.0
```

- [ ] **Step 2: Refactor `assets/sim/masks.sim`**

Replace `MoveToward` (lines 47-49) with:

```text
mask MoveToward(target)
  from spatial.nearby_alive_other(self, target)
```

Replace `Attack` (lines 75-79) with:

```text
mask Attack(target)
  from spatial.nearby_hostile_in_melee(self, target)
```

Note: the mask's pre-Task-5 `from query.nearby_agents(agents.pos(self), config.movement.max_move_radius)` carries the radius implicitly. Today's spatial walk does NOT honor a per-walk radius (the body template visits every cell — see Task 2's "no per-cell radius bounds" limitation). This is preserved as-is — radius bounding lands as a follow-up. The `nearby_alive_other` decl above does not take a radius; the `nearby_hostile_in_melee` decl bakes the 2.0 melee distance into the filter (the existing wolf-sim Attack mask already used a hard-coded 2.0).

If the existing mask's radius (`config.movement.max_move_radius`, `config.combat.attack_range`) needs to be preserved for downstream tests, add a radius param to the spatial query and pass the config value at the call site. Since today's body template visits every cell anyway, the radius is purely declarative; the test gate (Task 8) reveals whether anything actually depends on it.

- [ ] **Step 3: Update `mask_spatial_kind` in driver.rs**

In `crates/dsl_compiler/src/cg/lower/driver.rs::mask_spatial_kind` (line 894), extend to recognize the new namespace and produce `FilteredWalk { filter }`:

```rust
fn mask_spatial_kind(
    mask: &MaskIR,
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
) -> Option<SpatialQueryKind> {
    let source = mask.candidate_source.as_ref()?;
    match &source.kind {
        // New: spatial.<name>(args) — look up the decl, substitute
        // args into the filter, lower to a CgExprId, wrap in
        // FilteredWalk.
        IrExpr::NamespaceCall { ns: NamespaceId::Spatial, method, args } => {
            let decl = comp.spatial_queries.get(method)?;
            let filter_with_args = substitute_spatial_query_args(decl, args);
            let filter_id = lower_filter_for_mask(&filter_with_args, ctx).ok()?;
            Some(SpatialQueryKind::FilteredWalk { filter: filter_id })
        }
        // Legacy: query.nearby_agents — kept for any wolf-sim DSL
        // not yet migrated. Falls through to the heuristic-based
        // KinQuery/EngagementQuery selection below.
        IrExpr::NamespaceCall { ns: NamespaceId::Query, method, .. } if method == "nearby_agents" => {
            if predicate_uses_engagement_relationship(&mask.predicate) {
                Some(SpatialQueryKind::EngagementQuery)
            } else {
                Some(SpatialQueryKind::KinQuery)
            }
        }
        _ => None,
    }
}
```

The `comp` and `ctx` parameters need to be threaded — `mask_spatial_kind`'s current signature is `(mask: &MaskIR) -> Option<SpatialQueryKind>`. Update the signature and every call site (search `grep -n "mask_spatial_kind" crates/dsl_compiler/src/cg/lower/driver.rs`).

Add `substitute_spatial_query_args`:

```rust
/// Substitute call-site arguments into a spatial_query's filter
/// expression. Walks the IR tree, replacing `LocalRef(i)` for each
/// param i (where i >= 2, since 0=self, 1=candidate are bound at
/// lowering time, not substituted) with the corresponding arg
/// expression.
///
/// `self` (LocalRef(0)) and `candidate` (LocalRef(1)) are
/// preserved — the lowering pass `lower_filter_for_mask` resolves
/// `self` to the per-agent slot and `candidate` to the per-pair
/// candidate via `ctx.target_local`.
fn substitute_spatial_query_args(
    decl: &SpatialQueryDecl,
    args: &[IrCallArg],
) -> IrExprNode {
    let value_args: Vec<(LocalRef, IrExprNode)> = args.iter().enumerate()
        .map(|(i, arg)| (LocalRef((i + 2) as u32), arg.value.clone()))
        .collect();
    walk_substitute(&decl.filter, &value_args)
}

fn walk_substitute(node: &IrExprNode, subs: &[(LocalRef, IrExprNode)]) -> IrExprNode {
    let new_kind = match &node.kind {
        IrExpr::Local(local_ref, _) => {
            if let Some((_, replacement)) = subs.iter().find(|(lr, _)| lr == local_ref) {
                return replacement.clone();
            }
            node.kind.clone()
        }
        IrExpr::Field { base, field_name, field } => IrExpr::Field {
            base: Box::new(walk_substitute(base, subs)),
            field_name: field_name.clone(),
            field: field.clone(),
        },
        IrExpr::Binary { op, lhs, rhs } => IrExpr::Binary {
            op: *op,
            lhs: Box::new(walk_substitute(lhs, subs)),
            rhs: Box::new(walk_substitute(rhs, subs)),
        },
        IrExpr::Unary(op, inner) => IrExpr::Unary(*op, Box::new(walk_substitute(inner, subs))),
        IrExpr::NamespaceCall { ns, method, args } => IrExpr::NamespaceCall {
            ns: *ns,
            method: method.clone(),
            args: args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: walk_substitute(&a.value, subs),
                    span: a.span,
                })
                .collect(),
        },
        // ... cover every IrExpr variant exhaustively ...
        other => other.clone(),
    };
    IrExprNode {
        kind: new_kind,
        span: node.span,
    }
}
```

The `IrExpr` variant set is large — `cargo build` will surface non-exhaustive matches at compile time; cover each. For variants with no sub-expressions (`Lit`, etc.), `other.clone()` is correct.

- [ ] **Step 4: Verify compile-dsl regen succeeds**

Run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical`
Expected: succeeds with no panics. The output `crates/engine_gpu_rules/src/` files regen; the SCHEDULE now contains `FilteredWalk { filter: #N }` ops instead of `KinQuery`/`EngagementQuery`.

If lowering panics or surfaces a diagnostic for the new spatial.sim:
- Read the diagnostic carefully — it likely points to a missing arm in `walk_substitute` or a missed `IrExpr` variant.
- Fix incrementally; rerun.

- [ ] **Step 5: Verify SCHEDULE shape**

Run: `grep "spatial" crates/engine_gpu_rules/src/schedule.rs`
Expected: `FilteredWalk` references in the SCHEDULE (in place of `KinQuery`/`EngagementQuery`). `BuildHash` still present.

- [ ] **Step 6: Run dsl_compiler tests**

Run: `cargo test -p dsl_compiler --lib`
Expected: all pass. The `mask_spatial_kind` heuristic tests (`mask_spatial_kind_*` in driver.rs, lines 1871-1950 area) may need their fixtures updated to reflect the new lookup path; if a test asserts a specific KinQuery/EngagementQuery routing, leave it for backwards compat (legacy `from query.nearby_agents` path still selects those variants).

- [ ] **Step 7: Run workspace breadth gate**

Run: `cargo test --workspace`
Expected: at least no regressions vs `7d566a92`. Pre-existing failures (parity_with_cpu RED) stay RED with the same divergence shape.

- [ ] **Step 8: Commit**

```bash
git add assets/sim/spatial.sim \
        assets/sim/masks.sim \
        crates/dsl_compiler/src/cg/lower/driver.rs \
        crates/engine_gpu_rules/
git commit -m "refactor(dsl): wolf-sim masks use spatial_query for from-clauses (Phase 7 Task 5)"
```

---

## Task 6: Drop `KinQuery` / `EngagementQuery` variants + heuristic-driven dispatch

Once Task 5 confirms wolf-sim's two from-clause sites no longer route through the heuristic-based KinQuery/EngagementQuery selection, the variants drop. The fallback arm in `mask_spatial_kind` (lines that branch on `predicate_uses_engagement_relationship`) becomes dead code; the helper itself becomes dead code.

Removal is done in this order to keep each step a green build:

1. Search for every `SpatialQueryKind::KinQuery` / `EngagementQuery` reference (production + tests).
2. Replace tests that asserted the legacy variants with equivalent `FilteredWalk` assertions, OR delete tests that exclusively gate the legacy heuristic.
3. Delete the variants from the enum.
4. Delete `predicate_uses_engagement_relationship` + its `mask_spatial_kind` fallback branch.
5. Delete the `KinQuery`/`EngagementQuery` arms in `dependencies()`, `label()`, `compute_dependencies` walk, snake-name table, `lower_op_body`.
6. Delete the `SPATIAL_KIN_QUERY_BODY` and `SPATIAL_ENGAGEMENT_QUERY_BODY` const templates.

**Files:**
- Modify: `crates/dsl_compiler/src/cg/op.rs` — drop variants + arms.
- Modify: `crates/dsl_compiler/src/cg/emit/kernel.rs` — drop body templates + arms.
- Modify: `crates/dsl_compiler/src/cg/lower/driver.rs` — drop helper + fallback branch.
- Modify: `crates/dsl_compiler/src/cg/lower/spatial.rs::tests` — remove tests pinned to dropped variants.
- Modify: `crates/dsl_compiler/src/cg/dispatch.rs::tests` — same.

- [ ] **Step 1: Inventory every KinQuery / EngagementQuery reference**

Run: `grep -rn "KinQuery\|EngagementQuery\|SPATIAL_KIN_QUERY_BODY\|SPATIAL_ENGAGEMENT_QUERY_BODY\|predicate_uses_engagement_relationship\|spatial_kin_query\|spatial_engagement_query" crates/dsl_compiler/src/ crates/engine_gpu_rules/src/`

Save the output. Annotate each line with one of: (a) production code arm to delete, (b) test fixture to delete or migrate, (c) doc comment to update.

- [ ] **Step 2: Delete tests pinned to dropped variants**

For each test in the inventory that exclusively asserts a `KinQuery`/`EngagementQuery` shape (e.g., `snapshot_kin_query_op_display`, `auto_walker_matches_dependencies_signature_for_each_kind` if it iterates over the legacy variants), either:
- **Delete** the test if its concern is only the legacy variant.
- **Migrate** the test to assert the equivalent `FilteredWalk` shape.

The `auto_walker_matches_dependencies_signature_for_each_kind` test in `spatial.rs::tests` iterates over all variants — drop the legacy variants from the iteration after Step 4.

- [ ] **Step 3: Delete the variants from the enum + dependencies/label arms**

In `crates/dsl_compiler/src/cg/op.rs`:

```rust
pub enum SpatialQueryKind {
    BuildHash,
    FilteredWalk { filter: CgExprId },
}
```

Remove the corresponding arms in `dependencies()`, `label()`, and `compute_dependencies`. The compiler will reject every remaining match as non-exhaustive and surface every site that still references the dropped variants.

- [ ] **Step 4: Delete the heuristic + fallback in driver.rs**

In `crates/dsl_compiler/src/cg/lower/driver.rs`:
- Delete `predicate_uses_engagement_relationship` (line 917+).
- Simplify `mask_spatial_kind` to only handle `Spatial`-namespace calls. The `Query::nearby_agents` arm becomes a hard error (or returns `None` and lets the mask resolve to PerAgent — the latter is a silent miscompile, prefer the hard error):

```rust
fn mask_spatial_kind(
    mask: &MaskIR,
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
) -> Option<SpatialQueryKind> {
    let source = mask.candidate_source.as_ref()?;
    match &source.kind {
        IrExpr::NamespaceCall { ns: NamespaceId::Spatial, method, args } => {
            let decl = comp.spatial_queries.get(method)?;
            let filter_with_args = substitute_spatial_query_args(decl, args);
            let filter_id = lower_filter_for_mask(&filter_with_args, ctx).ok()?;
            Some(SpatialQueryKind::FilteredWalk { filter: filter_id })
        }
        // query.nearby_agents is no longer supported as a from-clause.
        // Returning None here would silently fall back to PerAgent;
        // emit a typed error instead via the lowering pipeline's
        // diagnostic accumulator.
        IrExpr::NamespaceCall { ns: NamespaceId::Query, method, .. } if method == "nearby_agents" => {
            // The mask resolution layer will surface
            // UnsupportedMaskFromClause when this returns None;
            // returning None is correct here — the validation already
            // happens upstream once `validate_from_clause_shape` drops
            // the legacy arm in Step 6.
            None
        }
        _ => None,
    }
}
```

- [ ] **Step 5: Delete the body templates in kernel.rs**

Delete `SPATIAL_KIN_QUERY_BODY` and `SPATIAL_ENGAGEMENT_QUERY_BODY` consts. Simplify the `lower_op_body` SpatialQuery arm:

```rust
        ComputeOpKind::SpatialQuery { kind } => Ok(match kind {
            SpatialQueryKind::BuildHash => SPATIAL_BUILD_HASH_BODY.to_string(),
            SpatialQueryKind::FilteredWalk { filter } => {
                let filter_wgsl = lower_cg_expr_to_wgsl(*filter, ctx)?;
                spatial_filtered_walk_body(&filter_wgsl)
            }
        }),
```

Delete the dropped variants from the snake-name table:

```rust
fn spatial_query_snake_name(kind: &SpatialQueryKind) -> String {
    match kind {
        SpatialQueryKind::BuildHash => String::from("spatial_build_hash"),
        SpatialQueryKind::FilteredWalk { filter } => {
            format!("spatial_filtered_walk_{}", filter.0)
        }
    }
}
```

- [ ] **Step 6: Drop `query.nearby_agents` from `validate_from_clause_shape`**

In `crates/dsl_compiler/src/cg/lower/mask.rs::validate_from_clause_shape`:

```rust
fn validate_from_clause_shape(
    mask_id: MaskId,
    source: &IrExprNode,
) -> Result<(), LoweringError> {
    match &source.kind {
        IrExpr::NamespaceCall { ns: NamespaceId::Spatial, .. } => Ok(()),
        _ => Err(LoweringError::UnsupportedMaskFromClause {
            mask: mask_id,
            span: source.span,
        }),
    }
}
```

- [ ] **Step 7: Build + fix every cascade error**

Run: `cargo build --workspace 2>&1 | head -80`
Expected: a list of non-exhaustive matches and dead-code warnings. Walk each:
- Match arm needed → delete the dropped-variant arm.
- Dead code → delete.
- Test pinned to dropped variant → delete or migrate per Step 2.

Iterate `cargo build` until clean.

- [ ] **Step 8: Run all tests**

Run: `cargo test --workspace`
Expected: all pass. No new regressions vs `7d566a92`. Parity test stays RED with the same divergence shape (this plan does not change parity).

- [ ] **Step 9: Regen + verify SCHEDULE only references FilteredWalk**

Run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical`
Run: `grep -E "KinQuery|EngagementQuery|spatial_kin|spatial_engagement" crates/engine_gpu_rules/src/`
Expected: zero matches.

- [ ] **Step 10: Commit**

```bash
git add crates/dsl_compiler/src/cg/op.rs \
        crates/dsl_compiler/src/cg/emit/kernel.rs \
        crates/dsl_compiler/src/cg/lower/driver.rs \
        crates/dsl_compiler/src/cg/lower/mask.rs \
        crates/dsl_compiler/src/cg/lower/spatial.rs \
        crates/dsl_compiler/src/cg/dispatch.rs \
        crates/engine_gpu_rules/
git commit -m "refactor(dsl_compiler): drop SpatialQueryKind::{KinQuery,EngagementQuery} variants (Phase 7 Task 6)"
```

---

## Task 7: Delete legacy emitters + generated runtime files

`crates/dsl_compiler/src/emit_spatial_kernel.rs::emit_kin_query_rs` and `emit_engagement_query_rs` are now dead — no callers, no kinds. Delete the functions; if the file is left empty, delete the file and remove the `mod emit_spatial_kernel` declaration.

Generated files in `crates/engine_gpu_rules/src/spatial_kin_query.{rs,wgsl}` and `spatial_engagement_query.{rs,wgsl}` should NOT regenerate after Task 6 (the FilteredWalk variant produces `spatial_filtered_walk_<N>.{rs,wgsl}` named files instead). Delete the stale files and verify the next regen does not recreate them.

Update `crates/engine_gpu_rules/src/lib.rs` to drop the `pub mod spatial_engagement_query` + `pub use SpatialEngagementQueryKernel` lines (and `spatial_kin_query` if it appears post-regen).

**Files:**
- Modify: `crates/dsl_compiler/src/emit_spatial_kernel.rs` — delete `emit_kin_query_rs` + `emit_engagement_query_rs`. If no other public API remains, delete the file.
- Modify: `crates/dsl_compiler/src/lib.rs` — drop `mod emit_spatial_kernel` if the file deleted.
- Delete: `crates/engine_gpu_rules/src/spatial_kin_query.{rs,wgsl}`.
- Delete: `crates/engine_gpu_rules/src/spatial_engagement_query.{rs,wgsl}`.
- Modify: `crates/engine_gpu_rules/src/lib.rs` — drop the matching `pub mod` + `pub use` lines.

- [ ] **Step 1: Confirm callers are gone**

Run: `grep -rn "emit_kin_query_rs\|emit_engagement_query_rs" crates/`
Expected: zero matches (Task 6 should have removed every reference; if any remain, fix them before continuing).

- [ ] **Step 2: Delete the functions**

In `crates/dsl_compiler/src/emit_spatial_kernel.rs`, delete the two functions. Read the file with `Read crates/dsl_compiler/src/emit_spatial_kernel.rs` first and check what else lives there. If `emit_build_hash_rs` (or anything else) remains, keep the file and just delete the two functions. If the file is now empty (or contains only a docstring), delete the file and drop the `mod emit_spatial_kernel` declaration in `crates/dsl_compiler/src/lib.rs`.

- [ ] **Step 3: Regen + capture the new generated file set**

Run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical`
Run: `ls crates/engine_gpu_rules/src/spatial_*.{rs,wgsl}`
Expected: only `spatial_build_hash.{rs,wgsl}` and `spatial_filtered_walk_<N>.{rs,wgsl}` (one pair per distinct filter expression in the SCHEDULE). No `spatial_kin_query.*` or `spatial_engagement_query.*`.

- [ ] **Step 4: Delete the stale runtime files**

Run: `rm crates/engine_gpu_rules/src/spatial_kin_query.rs crates/engine_gpu_rules/src/spatial_kin_query.wgsl crates/engine_gpu_rules/src/spatial_engagement_query.rs crates/engine_gpu_rules/src/spatial_engagement_query.wgsl`

(Check first that none survived the regen by listing the dir; only delete files that are NOT regenerated by the lowering output. The regen output pattern at Step 3 confirmed this.)

- [ ] **Step 5: Update `crates/engine_gpu_rules/src/lib.rs`**

Read the file:

```bash
cat crates/engine_gpu_rules/src/lib.rs
```

Remove the lines:
- `pub mod spatial_engagement_query;`
- `pub use spatial_engagement_query::SpatialEngagementQueryKernel;`
- (If `spatial_kin_query` is also present, drop it too.)

Add the new `pub mod spatial_filtered_walk_<N>;` + `pub use spatial_filtered_walk_<N>::SpatialFilteredWalk<N>Kernel;` lines per the regenerated file set. If `lib.rs` is itself generated (header says `// GENERATED by dsl_compiler`), the regen does this automatically — verify the diff and skip the manual edit.

- [ ] **Step 6: Build + test**

Run: `cargo build --workspace`
Expected: clean build.

Run: `cargo test --workspace`
Expected: all tests pass; no regressions vs `7d566a92`.

- [ ] **Step 7: Verify zero residual references**

Run: `grep -rn "spatial_kin_query\|spatial_engagement_query\|KinQuery\|EngagementQuery\|emit_kin_query_rs\|emit_engagement_query_rs" crates/`
Expected: zero matches.

- [ ] **Step 8: Commit**

```bash
git add crates/dsl_compiler/src/emit_spatial_kernel.rs \
        crates/dsl_compiler/src/lib.rs \
        crates/engine_gpu_rules/
git rm crates/engine_gpu_rules/src/spatial_kin_query.rs \
       crates/engine_gpu_rules/src/spatial_kin_query.wgsl \
       crates/engine_gpu_rules/src/spatial_engagement_query.rs \
       crates/engine_gpu_rules/src/spatial_engagement_query.wgsl
git commit -m "chore: delete legacy kin_query/engagement_query emitters + runtime files (Phase 7 Task 7)"
```

---

## Task 8: Final regression + parity-status preserved verification

This is the closure gate. The plan does not change parity status (parity_with_cpu RED stays RED with the same divergence shape — Phase 8 is what closes parity). The gate here is:

1. Full workspace test suite passes.
2. `compile-dsl --cg-canonical` regen succeeds with no panics, no naga validation failures.
3. The SCHEDULE produced by the regenerated CG IR contains `FilteredWalk` ops in place of every former `KinQuery`/`EngagementQuery` op, and no other op count changes.
4. `parity_with_cpu --features gpu` on the smoke fixture has the SAME failing fields it had at `7d566a92` (no new divergence; same RED state).

**Files:** None modified. Verification only.

- [ ] **Step 1: Regen + compare op count**

Run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical 2>&1 | tee /tmp/phase7_regen.log`
Run: `grep -c "DispatchOp::Kernel" crates/engine_gpu_rules/src/schedule.rs`
Expected: same kernel count as at `7d566a92` (26), modulo the kin/engagement → filtered_walk substitution.

- [ ] **Step 2: Run the full workspace tests**

Run: `cargo test --workspace 2>&1 | tee /tmp/phase7_test.log`
Expected: same pass/fail set as `7d566a92`. Parity tests stay in their pre-existing state (RED for parity_with_cpu_*; pre-existing per_entity_ring_emits_wgsl_fold_kernel stays RED if it was). No NEW failures.

If a new failure appears:
1. `git diff 7d566a92 -- <failing-test-file>` to confirm the test wasn't touched by Phase 7.
2. Bisect by reverting Phase 7 commits one at a time to identify which task broke the test.
3. Open a focused fix; commit; re-verify.

- [ ] **Step 3: Compare parity divergence shape**

Run: `cargo test -p engine_gpu --features gpu --test parity_with_cpu -- --nocapture 2>&1 | tee /tmp/phase7_parity.log`
Expected: same divergence pattern as `7d566a92` (per Phase 8 plan: divergence on `pos_x_bits` for moving agents). No new divergence fields.

If a new divergence appears (e.g., a field that was matching before now diverges):
1. Confirm via `git stash && cargo test -p engine_gpu --features gpu --test parity_with_cpu` on `7d566a92`.
2. If reproduced on the old SHA, the divergence is pre-existing — annotate and continue.
3. If new on Phase 7, bisect by reverting tasks and re-running parity until the offending task surfaces. Fix the regression in the task's commit (or open a focused follow-up commit).

- [ ] **Step 4: Confirm zero kin/engagement footprint workspace-wide**

Run: `grep -rn "KinQuery\|EngagementQuery\|spatial_kin_query\|spatial_engagement_query\|emit_kin_query_rs\|emit_engagement_query_rs\|predicate_uses_engagement_relationship" crates/ assets/ docs/`
Expected: matches only in `docs/superpowers/plans/` (this plan + sibling plans). Zero matches in `crates/` or `assets/`.

- [ ] **Step 5: Commit a closure marker**

```bash
git commit --allow-empty -m "chore: Phase 7 closure — FilteredWalk replaces kin/engagement, parity preserved (Phase 7 Task 8)"
```

Mark this commit as the Phase 7 closure SHA in the plan's status header (next step).

- [ ] **Step 6: Update plan header + cross-link siblings**

Edit this file's header to reflect closure:

```markdown
> **STATUS: CLOSED at HEAD `<sha>` (2026-MM-DD).** All 8 tasks complete. `SpatialQueryKind::{KinQuery, EngagementQuery}` removed; `FilteredWalk { filter: CgExprId }` is the sole walk variant. Wolf-sim DSL fixtures refactored. Legacy emitters + generated runtime files deleted. Workspace tests + parity divergence shape preserved (no new regressions).
```

Update `docs/superpowers/plans/2026-05-01-phase-8-cg-body-emit-parity.md`'s header note that Phase 7 has landed (search the file for "Phase 7" cross-references and adjust tone from "recommended to land before this one" to "landed at SHA `<sha>`").

```bash
git add docs/superpowers/plans/2026-05-01-phase-7-general-spatial-queries.md \
        docs/superpowers/plans/2026-05-01-phase-8-cg-body-emit-parity.md
git commit -m "docs(plans): mark Phase 7 closed; update Phase 8 cross-link"
```

---

## References

- **Predecessor plan**: [`2026-05-01-cg-lowering-gap-closure.md`](./2026-05-01-cg-lowering-gap-closure.md) — closed at checkpoint (33/39 lowering diagnostics + Movement-as-rule IR landed; Tasks 6, 11, 12 deferred to Phase 6/7/8).
- **Sibling plan**: [`2026-05-01-phase-6-cg-decision-lowering.md`](./2026-05-01-phase-6-cg-decision-lowering.md) — Tasks 1-5 done. IR + schema + Movement-as-rule + namespace registry in place; Phase 7 reuses the namespace-call lowering Task 4 landed.
- **Successor plan**: [`2026-05-01-phase-8-cg-body-emit-parity.md`](./2026-05-01-phase-8-cg-body-emit-parity.md) — closes the body-emit gap to drive `parity_with_cpu --features gpu` green. Phase 8's body-content tasks (4-7) target the new `FilteredWalk` IR shape this plan produces.
- Constitution: `docs/constitution.md` — P1, P3, P5, P10, P11.
- Triggered by user discussion 2026-05-01: "We want to rip kin out and support spatial hashes for other things. The target game is nothing like the wolf sim."
- Phase 7 stub (this file's predecessor revision, in git history at `a50218ac`).
