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
use dsl_ast::ir::{IrCallArg, IrExpr, IrExprNode, NamespaceId};

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
        // Three-phase parallel prefix scan:
        //   - ScanLocal/ScanAdd dispatch one lane per cell, batched
        //     into 256-cell scan chunks (`PerScanChunk`).
        //   - ScanCarry is a tiny serial fix-up (~42 entries for
        //     boids' 10 648-cell grid) and runs single-threaded.
        // Every other spatial-query kind is per-agent today.
        let shape = match kind {
            SpatialQueryKind::BuildHashScanLocal
            | SpatialQueryKind::BuildHashScanAdd => DispatchShape::PerScanChunk,
            SpatialQueryKind::BuildHashScanCarry => DispatchShape::OneShot,
            _ => DispatchShape::PerAgent,
        };
        let op_id = ctx
            .builder
            .add_op(computekind, shape, Span::dummy())
            .map_err(|e| LoweringError::BuilderRejected {
                error: e,
                span: Span::dummy(),
            })?;
        op_ids.push(op_id);
    }
    Ok(op_ids)
}

// ---------------------------------------------------------------------------
// Spatial namespace-call shape — shared across mask / fold / future
// per-pair body-iter contexts (slice 2 of stdlib-into-CG-IR plan).
// ---------------------------------------------------------------------------

/// Description of a `spatial.<method>(self, ...)` /
/// `query.nearby_agents(<pos>, <radius>)` call shape recognised at
/// lowering time. Returned by [`lower_spatial_namespace_call`] and
/// consumed by every emit context that needs to project a per-pair /
/// per-neighborhood iter from a spatial-namespace expression.
///
/// The shape is intentionally small: structural recognition + cell
/// radius. Filter-expression resolution (the `FilteredWalk` body) and
/// the choice of `SpatialQueryKind` requires `Compilation` access (to
/// look up `comp.spatial_queries[name]`) and pushes its own
/// expression sub-tree, so the helper keeps that step at the call
/// site (the driver's `mask_spatial_kind` + future per-pair body-iter
/// driver). What the helper does centralise:
///
/// 1. The valid `(ns, method, args)` shapes (today: any
///    `spatial.<method>(self, ...)` call OR the legacy
///    `query.nearby_agents(<pos>, <radius>)` with arity 2).
/// 2. The cell radius for the bounded walk (currently hard-coded to 1
///    cell; a future surface threads the radius from a call arg).
/// 3. The method name, captured for downstream consumers that need to
///    look up the registered `spatial_query <name>` decl.
///
/// What stays at the call site (per the slice-2 guardrail of "no new
/// top-level CG IR variants"):
///
/// - `SpatialQueryKind::FilteredWalk { filter }` resolution. The
///   filter expression lives in `comp.spatial_queries[name].filter`
///   and lowers via `lower_filter_for_mask` (driver-private). The
///   helper returns the method name; the consumer does the lookup.
/// - The op/stmt push. Mask lowering routes the kind into the
///   `MaskPredicate` op's `DispatchShape::PerPair`; fold lowering
///   pushes a `CgStmt::ForEachNeighbor`. Both shapes are existing CG
///   IR — the helper doesn't allocate either; the call site does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialIterShape {
    /// The recognised spatial-namespace flavour of the call —
    /// distinguishes `spatial.<name>` (Phase 7 surface, optionally
    /// resolves to a registered `spatial_query` decl) from the
    /// legacy `query.nearby_agents(<pos>, <radius>)` shape.
    pub flavour: SpatialIterFlavour,
    /// Inclusive half-width of the cell neighborhood walked
    /// per-thread. `1` walks the 3³ = 27 cells centred on the
    /// querying agent's cell; `2` walks 5³ = 125. For every shipping
    /// fixture today the runtime sizes `CELL_SIZE` to the
    /// per-fixture perception radius, so a single-cell radius
    /// covers every plausible neighbour. A future surface will
    /// thread the radius from a call arg; the field is exposed now
    /// so consumers don't hard-code `1` independently of one another.
    pub radius_cells: u32,
}

/// Recognised spatial-namespace call flavours.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialIterFlavour {
    /// `spatial.<name>(self, ...)` — Phase 7 named-query surface.
    /// `name` resolves (at the call site, against
    /// `Compilation::spatial_queries`) to a registered
    /// `spatial_query <name>(self, candidate, ...) = <filter>` decl
    /// whose filter becomes the `SpatialQueryKind::FilteredWalk`
    /// body. An unresolved name is a consumer-side concern — the
    /// helper recognises the shape but doesn't reject unknown names
    /// (mask consumers fail soft to `KinQuery`-flavoured walks; fold
    /// consumers treat the name as informational).
    Spatial { method: String },
    /// `query.nearby_agents(<pos>, <radius>)` — the legacy
    /// pre-Phase-7 wolf-sim convention. Arity is exactly 2 (a
    /// position vec3 and a radius f32). Surfaces today only as a
    /// mask `from` clause; the legacy `query.<other>` methods are
    /// rejected by [`lower_spatial_namespace_call`] with
    /// [`LoweringError::UnsupportedNamespaceCall`].
    QueryNearbyAgents,
}

/// Recognise a spatial-namespace call (`spatial.<...>` or the
/// legacy `query.nearby_agents(<pos>, <radius>)`) and return a
/// [`SpatialIterShape`] describing the per-pair / per-neighborhood
/// iter it implies.
///
/// # Parameters
///
/// - `ns`: the namespace tag from the source `IrExpr::NamespaceCall`.
///   Recognises [`NamespaceId::Spatial`] and [`NamespaceId::Query`];
///   any other namespace surfaces as
///   [`LoweringError::UnsupportedNamespaceCall`].
/// - `method`: the source-level method name (e.g.
///   `"nearby_agents"`, `"nearby_particles"`,
///   `"closest_prey"`).
/// - `args`: the call's argument list. Validated for arity against the
///   recognised shape:
///     - `query.nearby_agents` requires exactly 2 args.
///     - `spatial.<method>` accepts ≥1 args today (the `self`
///       receiver plus any value-args the registered
///       `spatial_query <name>` decl declares; the helper does not
///       enforce per-decl arity because that lookup needs comp
///       access — the call site does it).
/// - `span`: source span used in the typed error payload if the
///   shape is rejected.
/// - `_ctx`: future hook for arg-lowering (unused today; the
///   helper is purely shape-recognising).
///
/// # Returns
///
/// On success: a [`SpatialIterShape`] describing the iter. The
/// caller decides whether to (a) push a `CgStmt::ForEachNeighbor`
/// for fold consumers, (b) build a
/// `DispatchShape::PerPair { source: PerPairSource::SpatialQuery(kind) }`
/// for mask consumers, or (c) reject in contexts where per-pair
/// iter isn't supported (e.g., today's physics body iter — pending
/// per-pair body-iter wiring; see slice-2 plan).
///
/// # Errors
///
/// - [`LoweringError::UnsupportedNamespaceCall`] when:
///     - `ns` is neither `Spatial` nor `Query`.
///     - `ns == Query` but `method != "nearby_agents"`.
/// - [`LoweringError::NamespaceCallArityMismatch`] when:
///     - `query.nearby_agents` is called with `args.len() != 2`.
///     - `spatial.<method>` is called with zero args (no `self`).
///
/// # No side effects
///
/// The helper does NOT push CG nodes onto the builder. It is a
/// pure structural recogniser: every consumer is responsible for
/// allocating the surrounding stmt / op / dispatch shape. This
/// keeps the helper safely callable from contexts that may also
/// abort lowering (e.g., a fold whose body fails type-checking
/// after the iter is recognised).
pub fn lower_spatial_namespace_call(
    ns: NamespaceId,
    method: &str,
    args: &[IrCallArg],
    span: Span,
    _ctx: &mut LoweringCtx<'_>,
) -> Result<SpatialIterShape, LoweringError> {
    match ns {
        NamespaceId::Spatial => {
            if args.is_empty() {
                return Err(LoweringError::NamespaceCallArityMismatch {
                    ns,
                    method: method.to_string(),
                    expected: 1,
                    got: 0,
                    span,
                });
            }
            Ok(SpatialIterShape {
                flavour: SpatialIterFlavour::Spatial {
                    method: method.to_string(),
                },
                // Hard-coded for v1; see SpatialIterShape::radius_cells.
                radius_cells: 1,
            })
        }
        NamespaceId::Query if method == "nearby_agents" => {
            if args.len() != 2 {
                return Err(LoweringError::NamespaceCallArityMismatch {
                    ns,
                    method: method.to_string(),
                    expected: 2,
                    got: args.len(),
                    span,
                });
            }
            Ok(SpatialIterShape {
                flavour: SpatialIterFlavour::QueryNearbyAgents,
                radius_cells: 1,
            })
        }
        _ => Err(LoweringError::UnsupportedNamespaceCall {
            ns,
            method: method.to_string(),
            span,
        }),
    }
}

/// Convenience: recognise a spatial call from its `IrExpr` form.
/// Returns `Ok(Some(shape))` when `node` is a recognised spatial
/// namespace call, `Ok(None)` for any other expression shape, and
/// propagates the typed error from
/// [`lower_spatial_namespace_call`] for a malformed spatial call.
///
/// Used by fold-iter classification (`expr::lower_fold_over_agents`)
/// and any future per-pair body-iter dispatch. Lets callers do the
/// "is this a spatial iter?" check without re-implementing the
/// match-and-arity scaffolding.
pub fn try_recognise_spatial_iter(
    node: &IrExprNode,
    ctx: &mut LoweringCtx<'_>,
) -> Result<Option<SpatialIterShape>, LoweringError> {
    match &node.kind {
        IrExpr::NamespaceCall { ns, method, args }
            if matches!(ns, NamespaceId::Spatial | NamespaceId::Query) =>
        {
            let shape = lower_spatial_namespace_call(*ns, method.as_str(), args, node.span, ctx)?;
            Ok(Some(shape))
        }
        _ => Ok(None),
    }
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
        // Companion snapshot pinning `BuildHash`'s shape: reads
        // `agent.self.pos` (the position the per-cell hash bucket is
        // computed from), writes `spatial.grid_cells` +
        // `spatial.grid_offsets`. Catches a regression in the
        // dependencies table or in the `DataHandle` Display impl.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_spatial_queries(&[SpatialQueryKind::BuildHash], &mut ctx).expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=spatial_query(build_hash) shape=per_agent reads=[agent.self.pos] writes=[spatial.grid_cells, spatial.grid_offsets]"
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

    // ---- 11. lower_spatial_namespace_call helper coverage ----------------

    fn helper_span() -> Span {
        Span::new(0, 0)
    }

    fn helper_arg(value_kind: IrExpr) -> IrCallArg {
        let s = helper_span();
        IrCallArg {
            name: None,
            value: IrExprNode {
                kind: value_kind,
                span: s,
            },
            span: s,
        }
    }

    #[test]
    fn helper_recognises_basic_spatial_call() {
        // `spatial.nearby_agents(self, radius)` — the canonical
        // Phase 7 fold/mask iter shape. Helper validates arity ≥ 1
        // and returns a `Spatial { method }` flavour with the
        // hard-coded radius_cells = 1.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let args = vec![
            helper_arg(IrExpr::LitFloat(0.0)),
            helper_arg(IrExpr::LitFloat(5.0)),
        ];
        let shape = lower_spatial_namespace_call(
            NamespaceId::Spatial,
            "nearby_agents",
            &args,
            helper_span(),
            &mut ctx,
        )
        .expect("recognises spatial.nearby_agents");
        assert_eq!(shape.radius_cells, 1);
        match shape.flavour {
            SpatialIterFlavour::Spatial { method } => {
                assert_eq!(method, "nearby_agents");
            }
            other => panic!("expected Spatial, got {other:?}"),
        }
    }

    #[test]
    fn helper_recognises_legacy_query_nearby_agents() {
        // Legacy mask `from query.nearby_agents(<pos>, <radius>)`
        // shape — exactly two args, distinct flavour from
        // `spatial.<...>`.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let args = vec![
            helper_arg(IrExpr::LitFloat(0.0)),
            helper_arg(IrExpr::LitFloat(5.0)),
        ];
        let shape = lower_spatial_namespace_call(
            NamespaceId::Query,
            "nearby_agents",
            &args,
            helper_span(),
            &mut ctx,
        )
        .expect("recognises legacy query.nearby_agents");
        assert_eq!(shape.radius_cells, 1);
        assert_eq!(shape.flavour, SpatialIterFlavour::QueryNearbyAgents);
    }

    #[test]
    fn helper_rejects_unknown_namespace() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_spatial_namespace_call(
            NamespaceId::Agents,
            "pos",
            &[],
            helper_span(),
            &mut ctx,
        )
        .expect_err("agents namespace is not a spatial iter");
        match err {
            LoweringError::UnsupportedNamespaceCall { ns, method, .. } => {
                assert_eq!(ns, NamespaceId::Agents);
                assert_eq!(method, "pos");
            }
            other => panic!("expected UnsupportedNamespaceCall, got {other:?}"),
        }
    }

    #[test]
    fn helper_rejects_legacy_query_with_wrong_arity() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_spatial_namespace_call(
            NamespaceId::Query,
            "nearby_agents",
            &[helper_arg(IrExpr::LitFloat(1.0))],
            helper_span(),
            &mut ctx,
        )
        .expect_err("legacy query.nearby_agents needs 2 args");
        match err {
            LoweringError::NamespaceCallArityMismatch {
                ns, method, expected, got, ..
            } => {
                assert_eq!(ns, NamespaceId::Query);
                assert_eq!(method, "nearby_agents");
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected NamespaceCallArityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn helper_rejects_zero_arg_spatial_call() {
        // `spatial.<method>()` is malformed — every spatial query
        // must at minimum receive `self`.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_spatial_namespace_call(
            NamespaceId::Spatial,
            "nearby_agents",
            &[],
            helper_span(),
            &mut ctx,
        )
        .expect_err("spatial.<method>() requires self");
        match err {
            LoweringError::NamespaceCallArityMismatch {
                ns, method, expected, got, ..
            } => {
                assert_eq!(ns, NamespaceId::Spatial);
                assert_eq!(method, "nearby_agents");
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
            }
            other => panic!("expected NamespaceCallArityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn try_recognise_returns_none_for_non_spatial() {
        // `try_recognise_spatial_iter` returns `Ok(None)` on any
        // non-spatial expression — the convenience for fold-iter
        // classification (`expr::lower_fold_over_agents`) which
        // needs to distinguish `agents` (None) from `spatial.<...>`
        // (Some) without re-implementing the match scaffolding.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let node = IrExprNode {
            kind: IrExpr::Namespace(NamespaceId::Agents),
            span: helper_span(),
        };
        let recognised = try_recognise_spatial_iter(&node, &mut ctx).expect("no error");
        assert!(recognised.is_none(), "agents namespace is not a spatial iter");
    }

    #[test]
    fn try_recognise_returns_some_for_spatial_call() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let node = IrExprNode {
            kind: IrExpr::NamespaceCall {
                ns: NamespaceId::Spatial,
                method: "nearby_particles".to_string(),
                args: vec![helper_arg(IrExpr::LitFloat(0.0))],
            },
            span: helper_span(),
        };
        let shape = try_recognise_spatial_iter(&node, &mut ctx)
            .expect("no error")
            .expect("recognised");
        match shape.flavour {
            SpatialIterFlavour::Spatial { method } => {
                assert_eq!(method, "nearby_particles");
            }
            other => panic!("expected Spatial, got {other:?}"),
        }
    }
}
