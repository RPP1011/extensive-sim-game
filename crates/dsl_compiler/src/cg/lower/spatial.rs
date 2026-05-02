//! Spatial-query lowering — driver-supplied [`SpatialQueryKind`] list →
//! one [`ComputeOpKind::SpatialQuery`] op per entry.
//!
//! Phase 2, Task 2.6 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`). Unlike
//! the mask / view / physics / scoring passes, this pass does NOT
//! consume an AST IR sub-tree: the DSL surface today has no
//! `@spatial query` annotation type, and the only spatial-query-bearing
//! AST shape is the `from query.nearby_agents(...)` clause on masks
//! (consumed structurally by [`super::mask::lower_mask`] when it
//! resolves a [`crate::cg::dispatch::PerPairSource::SpatialQuery`]).
//!
//! Instead, the pass takes a driver-supplied list of
//! [`SpatialQueryKind`] values that the schedule synthesizer (Phase 3)
//! will need ops for: each variant produces ONE
//! [`crate::cg::op::ComputeOp`] whose [`ComputeOpKind`] is
//! [`ComputeOpKind::SpatialQuery`]. Reads and writes for each op are
//! auto-derived by [`crate::cg::op::ComputeOpKind::compute_dependencies`]
//! from the variant's hard-coded
//! [`SpatialQueryKind::dependencies`] signature — the lowering does
//! not duplicate that table.
//!
//! # Pass shape
//!
//! 1. For each [`SpatialQueryKind`] in the input list (in source
//!    order):
//!    1.1. Build a [`ComputeOpKind::SpatialQuery { kind }`].
//!    1.2. Push it onto the builder via
//!         [`crate::cg::program::CgProgramBuilder::add_op`] with
//!         [`crate::cg::dispatch::DispatchShape::PerAgent`] (every
//!         spatial kernel — `BuildHash`, `KinQuery`, `EngagementQuery`
//!         — dispatches one thread per agent slot).
//! 2. Return the freshly-allocated [`OpId`]s in input order.
//!
//! # Why no AST input
//!
//! The plan body says "`@spatial query` annotations on masks and
//! physics rules become `ComputeOp { kind: SpatialQuery { … } }`". Today
//! that annotation surface does not exist in the AST IR (verified
//! against `crates/dsl_ast/src/ir.rs` — no `@spatial` keyword, no
//! corresponding annotation type). Real `from` clauses on masks
//! (`from query.nearby_agents(...)`) are handled by
//! [`super::mask::lower_mask`]: the mask itself becomes a per-pair
//! [`ComputeOpKind::MaskPredicate`] op routed over a
//! [`crate::cg::dispatch::PerPairSource::SpatialQuery`] source. Task 2.6
//! produces the *underlying* spatial-query ops (`BuildHash`,
//! `KinQuery`, `EngagementQuery`) that those per-pair masks need to read
//! from at runtime.
//!
//! The driver (Task 2.8) is responsible for deciding which kinds the
//! tick needs (typically: walk every mask + physics rule, collect the
//! distinct [`SpatialQueryKind`] values referenced by their
//! `from`-clause resolutions, and pass the resulting list — plus a
//! preceding `BuildHash` — into this pass).
//!
//! # Duplicate kinds
//!
//! The lowering does NOT deduplicate. Two `KinQuery` entries produce
//! two ops. The plan body says "the schedule synthesis later decides
//! whether they share a hash-build with other queries" — implying
//! deduplication is the schedule's job, not the lowering's. Source-
//! order is preserved: passing `[KinQuery, BuildHash]` produces ops in
//! that order even though `BuildHash` writes the grid that `KinQuery`
//! reads. The schedule pass (Phase 3) topologically sorts on the
//! reads/writes graph; lowering's contract is one-op-per-input,
//! source-ordered.
//!
//! # Spans
//!
//! Spatial-query ops have no AST source today (no `@spatial query`
//! parsing). Each op is constructed with [`Span::dummy`]. When the
//! annotation surface lands, this pass should grow a per-kind span
//! parameter (or take a `(SpatialQueryKind, Span)` tuple) so the op
//! carries the correct source location for diagnostics. Until then the
//! `Span::dummy` placeholder is honest about the absent provenance.
//!
//! # Typed-error surface
//!
//! All defects surface as variants on the unified
//! [`super::error::LoweringError`]. The pass touches only one shared
//! variant: [`LoweringError::BuilderRejected`] (catches builder
//! invariant drift if `add_op` ever rejects a spatial-query kind —
//! shouldn't normally fire because [`ComputeOpKind::SpatialQuery`]
//! carries no expr / list ids and so passes
//! [`crate::cg::program::CgProgramBuilder::validate_op_kind_refs`]
//! trivially). No `Spatial*` variant is added: the small surface area
//! of the pass means every conceivable defect either reuses an existing
//! variant or is structurally impossible.
//!
//! # Limitations
//!
//! - **No AST input.** Today's pass takes a driver-supplied
//!   `&[SpatialQueryKind]`. When the `@spatial query` annotation
//!   surface lands in the AST, this pass should grow an alternative
//!   entry point that consumes the annotation directly and threads the
//!   AST span.
//! - **No span threading.** Every op uses [`Span::dummy`]. Future work
//!   threads spans from the masks / physics rules that triggered each
//!   query (see "Spans" section above).
//! - **No deduplication.** Two identical input kinds produce two ops.
//!   Schedule synthesis (Phase 3) is responsible for hash-build sharing
//!   and query coalescing.

use dsl_ast::ast::Span;

use crate::cg::dispatch::DispatchShape;
use crate::cg::op::{ComputeOpKind, OpId, SpatialQueryKind};

use super::error::LoweringError;
use super::expr::LoweringCtx;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower a driver-supplied list of [`SpatialQueryKind`] values into one
/// [`ComputeOpKind::SpatialQuery`] op per entry, pushed onto
/// `ctx.builder` in source order.
///
/// # Parameters
///
/// - `kinds`: the spatial-query kernels the schedule needs ops for.
///   The driver (Task 2.8) collects these by walking every mask and
///   physics rule and recording each distinct `from`-clause
///   resolution; it then prepends a `BuildHash` so the per-cell grid
///   is populated before any `*Query` reads it. This pass treats the
///   list as opaque: source order is preserved, duplicates are
///   allowed, no validation is performed against the rest of the
///   program.
/// - `ctx`: the lowering context (carries the in-flight builder). The
///   pass touches only `ctx.builder.add_op` — no expression-level
///   helpers, no view / action / variant registries.
///
/// # Returns
///
/// One [`OpId`] per input entry, in input order. Empty input returns
/// an empty `Vec` and pushes nothing.
///
/// # Errors
///
/// See [`LoweringError`] for the closed defect set. Only
/// [`LoweringError::BuilderRejected`] is reachable today — a defense-
/// in-depth wrap around [`crate::cg::program::CgProgramBuilder::add_op`]
/// for catching builder invariant drift. The op kind carries no
/// expr / list id references, so
/// [`crate::cg::program::CgProgramBuilder::validate_op_kind_refs`]'s
/// `SpatialQuery { .. }` arm is `Ok(())` unconditionally.
///
/// # Side effects
///
/// On success: `kinds.len()` ops pushed onto the builder; no
/// expression sub-trees are produced (spatial-query kinds carry no
/// embedded `CgExpr`). On failure at index `i`: the first `i` ops
/// have been pushed; ops past the failure point are not added.
///
/// # Limitations
///
/// See the module docstring's "Limitations" section. The pass takes a
/// driver-supplied list rather than reading the AST because no
/// `@spatial query` annotation type exists in `dsl_ast::ir` today. All
/// ops use [`Span::dummy`]; duplicate kinds are allowed (schedule
/// synthesis dedupes); source order is preserved.
pub fn lower_spatial_queries(
    kinds: &[SpatialQueryKind],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<OpId>, LoweringError> {
    let mut op_ids = Vec::with_capacity(kinds.len());
    for &kind in kinds {
        let computekind = ComputeOpKind::SpatialQuery { kind };
        let op_id = ctx
            .builder
            .add_op(computekind, DispatchShape::PerAgent, Span::dummy())
            .map_err(|e| LoweringError::BuilderRejected {
                error: e,
                span: Span::dummy(),
            })?;
        op_ids.push(op_id);
    }
    Ok(op_ids)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cg::data_handle::{DataHandle, SpatialStorageKind};
    use crate::cg::program::CgProgramBuilder;

    // ---- 1. Smallest happy: a single BuildHash --------------------------

    #[test]
    fn lowers_single_build_hash() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);

        let ids = lower_spatial_queries(&[SpatialQueryKind::BuildHash], &mut ctx)
            .expect("lowers");
        assert_eq!(ids, vec![OpId(0)]);

        let prog = builder.finish();
        assert_eq!(prog.ops.len(), 1);
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::PerAgent);
        match &op.kind {
            ComputeOpKind::SpatialQuery { kind } => {
                assert_eq!(*kind, SpatialQueryKind::BuildHash);
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    // ---- 2. Multiple distinct kinds preserve source order ---------------

    #[test]
    fn lowers_multiple_kinds_in_input_order() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);

        let input = [
            SpatialQueryKind::BuildHash,
            SpatialQueryKind::KinQuery,
            SpatialQueryKind::EngagementQuery,
        ];
        let ids = lower_spatial_queries(&input, &mut ctx).expect("lowers");
        assert_eq!(ids, vec![OpId(0), OpId(1), OpId(2)]);

        let prog = builder.finish();
        assert_eq!(prog.ops.len(), 3);

        // Each op carries its kind in input order.
        for (i, expected) in input.iter().enumerate() {
            match &prog.ops[i].kind {
                ComputeOpKind::SpatialQuery { kind } => assert_eq!(kind, expected),
                other => panic!("op#{i}: unexpected kind {other:?}"),
            }
            assert_eq!(prog.ops[i].shape, DispatchShape::PerAgent);
        }
    }

    // ---- 3. Duplicate kinds are allowed (schedule synthesis dedupes) ----

    #[test]
    fn duplicate_kinds_produce_distinct_ops() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);

        let ids = lower_spatial_queries(
            &[SpatialQueryKind::KinQuery, SpatialQueryKind::KinQuery],
            &mut ctx,
        )
        .expect("lowers");
        assert_eq!(ids, vec![OpId(0), OpId(1)]);

        let prog = builder.finish();
        assert_eq!(prog.ops.len(), 2);
        for op in &prog.ops {
            match &op.kind {
                ComputeOpKind::SpatialQuery { kind } => {
                    assert_eq!(*kind, SpatialQueryKind::KinQuery);
                }
                other => panic!("unexpected kind: {other:?}"),
            }
        }
    }

    // ---- 4. Empty input is Ok(vec![]) and pushes nothing ----------------

    #[test]
    fn empty_input_pushes_no_ops() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);

        let ids = lower_spatial_queries(&[], &mut ctx).expect("lowers");
        assert!(ids.is_empty());

        let prog = builder.finish();
        assert!(prog.ops.is_empty(), "empty input pushes nothing");
        assert!(prog.exprs.is_empty(), "no expression sub-trees produced");
    }

    // ---- 5. Auto-walker reads/writes match SpatialQueryKind::dependencies()

    #[test]
    fn auto_walker_matches_dependencies_signature_for_each_kind() {
        // Every kind's lowered op's reads/writes vectors must equal the
        // hard-coded `(reads, writes)` table on `SpatialQueryKind` —
        // the auto-walker (Task 1.3) is the single source of truth and
        // this lowering does not invent its own.
        for kind in [
            SpatialQueryKind::BuildHash,
            SpatialQueryKind::KinQuery,
            SpatialQueryKind::EngagementQuery,
        ] {
            let mut builder = CgProgramBuilder::new();
            let mut ctx = LoweringCtx::new(&mut builder);
            lower_spatial_queries(&[kind], &mut ctx).expect("lowers");
            let prog = builder.finish();
            let op = &prog.ops[0];
            let (expected_reads, expected_writes) = kind.dependencies();
            assert_eq!(
                op.reads, expected_reads,
                "kind={kind}: reads diverged from SpatialQueryKind::dependencies()"
            );
            assert_eq!(
                op.writes, expected_writes,
                "kind={kind}: writes diverged from SpatialQueryKind::dependencies()"
            );
        }
    }

    // ---- 6. Snapshot: pinned `Display` form for a lowered op ------------

    #[test]
    fn snapshot_kin_query_op_display() {
        // Pins the wire format produced by `ComputeOp`'s Display impl
        // for a `KinQuery` op. Downstream consumers (snapshot tests,
        // structured logs) depend on this exact string shape — the
        // auto-walker's read/write entries render via
        // `DataHandle::SpatialStorage`'s `spatial.{kind}` form.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_spatial_queries(&[SpatialQueryKind::KinQuery], &mut ctx).expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=spatial_query(kin_query) shape=per_agent reads=[spatial.grid_cells, spatial.grid_offsets] writes=[spatial.query_results]"
        );
    }

    #[test]
    fn snapshot_build_hash_op_display() {
        // Companion snapshot pinning `BuildHash`'s shape — empty reads,
        // grid_cells + grid_offsets writes. Catches a regression in the
        // dependencies table or in the `DataHandle` Display impl.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_spatial_queries(&[SpatialQueryKind::BuildHash], &mut ctx).expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=spatial_query(build_hash) shape=per_agent reads=[] writes=[spatial.grid_cells, spatial.grid_offsets]"
        );
    }

    // ---- 7. Source-order preservation under "wrong" semantic order ------

    #[test]
    fn source_order_preserved_even_when_semantically_inverted() {
        // Driver passes `[KinQuery, BuildHash]` — semantically wrong
        // (KinQuery reads the grid that BuildHash writes) but the
        // lowering preserves source order. Schedule synthesis (Phase 3)
        // is the layer that topologically sorts on reads/writes; this
        // pass's contract is one-op-per-input, source-ordered.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ids = lower_spatial_queries(
            &[SpatialQueryKind::KinQuery, SpatialQueryKind::BuildHash],
            &mut ctx,
        )
        .expect("lowers");
        assert_eq!(ids, vec![OpId(0), OpId(1)]);

        let prog = builder.finish();
        match &prog.ops[0].kind {
            ComputeOpKind::SpatialQuery { kind } => {
                assert_eq!(*kind, SpatialQueryKind::KinQuery);
            }
            other => panic!("op#0: unexpected kind {other:?}"),
        }
        match &prog.ops[1].kind {
            ComputeOpKind::SpatialQuery { kind } => {
                assert_eq!(*kind, SpatialQueryKind::BuildHash);
            }
            other => panic!("op#1: unexpected kind {other:?}"),
        }
    }

    // ---- 8. Touching the builder twice in a row accumulates ops ---------

    #[test]
    fn successive_calls_append_ops() {
        // The driver may call this pass multiple times if it surfaces
        // spatial-query needs incrementally (e.g., once per scope).
        // Each call appends to the builder's existing op list — ids
        // come from the builder's monotonic counter, not from a
        // per-call reset.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);

        let first = lower_spatial_queries(&[SpatialQueryKind::BuildHash], &mut ctx)
            .expect("first call");
        assert_eq!(first, vec![OpId(0)]);

        let second = lower_spatial_queries(
            &[SpatialQueryKind::KinQuery, SpatialQueryKind::EngagementQuery],
            &mut ctx,
        )
        .expect("second call");
        assert_eq!(second, vec![OpId(1), OpId(2)]);

        let prog = builder.finish();
        assert_eq!(prog.ops.len(), 3);
    }

    // ---- 9. Output handle shape pin (catches dependencies-table drift) --

    #[test]
    fn engagement_query_writes_query_results_only() {
        // A more targeted version of test 5 — pins the exact
        // `DataHandle::SpatialStorage { kind: QueryResults }` write for
        // `EngagementQuery`. Catches a regression where the
        // dependencies table mistakenly adds a stray write
        // (e.g., GridCells, breaking schedule synthesis's
        // hash-build-coalescing analysis).
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_spatial_queries(&[SpatialQueryKind::EngagementQuery], &mut ctx)
            .expect("lowers");
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(
            op.writes,
            vec![DataHandle::SpatialStorage {
                kind: SpatialStorageKind::QueryResults,
            }]
        );
        assert_eq!(
            op.reads,
            vec![
                DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::GridCells,
                },
                DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::GridOffsets,
                },
            ]
        );
    }

    // ---- 10. FilteredWalk round-trips through the lowering pass ----------

    #[test]
    fn filtered_walk_kind_round_trips_through_lowering() {
        use crate::cg::expr::{CgExpr, LitValue};

        let mut builder = CgProgramBuilder::new();
        // Push a real Bool literal so validate_op_kind_refs accepts the
        // filter id (it calls check_expr_id which requires the id to be
        // in-range). CgExprId(0) is the first allocated id.
        let filter_id = builder
            .add_expr(CgExpr::Lit(LitValue::Bool(true)))
            .expect("push lit expr");

        let mut ctx = LoweringCtx::new(&mut builder);

        let kind = SpatialQueryKind::FilteredWalk { filter: filter_id };
        let ids = lower_spatial_queries(&[kind], &mut ctx).expect("lowers");
        assert_eq!(ids.len(), 1);

        let prog = builder.finish();
        match prog.ops[0].kind {
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::FilteredWalk { filter },
            } => assert_eq!(filter, filter_id),
            ref other => panic!("unexpected: {other:?}"),
        }
        assert_eq!(
            prog.ops[0].writes,
            vec![DataHandle::SpatialStorage {
                kind: SpatialStorageKind::QueryResults
            }]
        );
    }
}
