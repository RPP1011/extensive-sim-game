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


    // ---- 3. Duplicate kinds are allowed (schedule synthesis dedupes) ----


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


    // ---- 6. Snapshot: pinned `Display` form for a lowered op ------------


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


    // ---- 8. Touching the builder twice in a row accumulates ops ---------


    // ---- 9. Output handle shape pin (catches dependencies-table drift) --


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
