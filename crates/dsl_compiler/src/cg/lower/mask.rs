//! Mask lowering — `MaskIR → ComputeOpKind::MaskPredicate`.
//!
//! Phase 2, Task 2.2 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`). Each
//! `mask <Name>` decl in the resolved DSL IR becomes one
//! [`ComputeOp`] whose [`ComputeOpKind`] is
//! [`ComputeOpKind::MaskPredicate`].
//!
//! The pass:
//!
//! 1. Lowers the mask predicate body via [`super::expr::lower_expr`],
//!    reusing the expression-level lowering wholesale.
//! 2. Type-checks the produced node — a mask predicate must be `Bool`.
//! 3. Resolves the `from <expr>` clause:
//!    - Absent → [`DispatchShape::PerAgent`].
//!    - `query.nearby_agents(...)` shape with a caller-supplied
//!      [`SpatialQueryKind`] → [`DispatchShape::PerPair`] with
//!      [`PerPairSource::SpatialQuery`].
//!    - Any other shape → typed [`MaskLoweringError::UnsupportedFromClause`].
//! 4. Builds the op via [`CgProgramBuilder::add_op`] (which
//!    auto-derives reads/writes from `kind`).
//! 5. Interns the mask name on the builder for pretty-printing.
//!
//! # Why a separate `MaskLoweringError`
//!
//! Task 2.1's [`super::expr::LoweringError`] is the typed defect set
//! for *expression*-level lowering — it has no vocabulary for "mask
//! predicate produced a non-Bool" or "from clause shape unrecognized".
//! Rather than mutate the expression-level enum to absorb mask-shaped
//! variants, this pass returns its own [`MaskLoweringError`] which
//! wraps an `expr::LoweringError` for the predicate-body case and adds
//! mask-level variants alongside.
//!
//! # Spatial query resolution
//!
//! The DSL surface for v1 has exactly one `from`-clause shape:
//! `query.nearby_agents(<pos>, <radius>)`. The post-resolution AST does
//! NOT yet split this into kin / engagement / future-flavour queries —
//! that distinction belongs to the spatial-query lowering pass
//! (Task 2.6) which decides which kernel slot a given mask routes to.
//! For Task 2.2 the caller supplies the resolved
//! [`SpatialQueryKind`] alongside the [`MaskIR`]; the lowering only
//! validates that the AST shape is recognized and threads the kind
//! into the dispatch shape.

use std::fmt;

use dsl_ast::ast::Span;
use dsl_ast::ir::{IrExpr, IrExprNode, MaskIR, NamespaceId};

use crate::cg::data_handle::{CgExprId, MaskId};
use crate::cg::dispatch::{DispatchShape, PerPairSource};
use crate::cg::expr::{type_check, CgTy, TypeCheckCtx, TypeError};
use crate::cg::op::{ComputeOpKind, OpId, SpatialQueryKind};
use crate::cg::program::BuilderError;

use super::expr::{lower_expr, LoweringCtx, LoweringError};

// ---------------------------------------------------------------------------
// MaskLoweringError — typed defects the mask lowering can report.
// ---------------------------------------------------------------------------

/// Typed defect surfaced by [`lower_mask`].
///
/// Keeps the mask-level concerns (predicate-not-Bool, from-clause
/// shape, missing/unexpected spatial-query kind) separate from the
/// expression-level [`LoweringError`] vocabulary. The
/// [`MaskLoweringError::Predicate`] variant wraps an inner
/// `LoweringError` so failures inside the predicate body propagate
/// without losing typed information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaskLoweringError {
    /// The mask's predicate body failed to lower at the expression
    /// level. The inner [`LoweringError`] names the offending sub-node.
    Predicate(LoweringError),

    /// The predicate body type-checked to a non-`Bool` type. Mask
    /// predicates write into a per-agent bitmap; the predicate value
    /// must be `Bool` so each tick's value is a single bit. Distinct
    /// from [`LoweringError::IllTypedExpression`] because the constraint
    /// here is mask-level (the bitmap shape), not operator-level.
    PredicateNotBool {
        mask: MaskId,
        got: CgTy,
        span: Span,
    },

    /// A type-check of the predicate node itself (after lowering) failed.
    /// Surfaces the underlying [`TypeError`] without panicking; should
    /// not normally fire because [`lower_expr`] already type-checks every
    /// node it constructs.
    PredicateTypeCheckFailure {
        mask: MaskId,
        error: TypeError,
        span: Span,
    },

    /// The mask's `from <expr>` clause is a shape mask lowering does
    /// not recognize. v1 supports only
    /// `query.nearby_agents(<pos>, <radius>)`; any other shape (a bare
    /// view call, a different namespace method, a literal) surfaces
    /// here. Span points at the `from`-clause expression.
    UnsupportedFromClause {
        mask: MaskId,
        span: Span,
    },

    /// The mask has a `from` clause but the caller supplied no
    /// [`SpatialQueryKind`]. The driver (Task 2.6 / 2.8) is responsible
    /// for resolving each from-bearing mask to a kin / engagement /
    /// future kind; a missing resolution is a driver defect, surfaced
    /// here as a typed error rather than a panic.
    MissingSpatialQueryKind {
        mask: MaskId,
        span: Span,
    },

    /// The caller supplied a [`SpatialQueryKind`] but the mask has no
    /// `from` clause. Catches the inverse bug (driver pre-allocated a
    /// kind for a self-only mask).
    UnexpectedSpatialQueryKind {
        mask: MaskId,
        kind: SpatialQueryKind,
        span: Span,
    },

    /// The builder rejected `add_op` for the constructed mask op. Wraps
    /// any [`BuilderError`] (dangling expr id, duplicate intern entry,
    /// …) so the typed reason survives.
    BuilderRejected {
        error: BuilderError,
        span: Span,
    },
}

impl fmt::Display for MaskLoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskLoweringError::Predicate(inner) => write!(f, "mask predicate: {}", inner),
            MaskLoweringError::PredicateNotBool { mask, got, span } => write!(
                f,
                "mask#{} predicate at {}..{} produced {} — must be Bool",
                mask.0, span.start, span.end, got
            ),
            MaskLoweringError::PredicateTypeCheckFailure { mask, error, span } => write!(
                f,
                "mask#{} predicate at {}..{} failed type-check — {}",
                mask.0, span.start, span.end, error
            ),
            MaskLoweringError::UnsupportedFromClause { mask, span } => write!(
                f,
                "mask#{} `from` clause at {}..{} has an unsupported shape — only `query.nearby_agents(<pos>, <radius>)` is recognised",
                mask.0, span.start, span.end
            ),
            MaskLoweringError::MissingSpatialQueryKind { mask, span } => write!(
                f,
                "mask#{} at {}..{} has a `from` clause but no SpatialQueryKind was supplied by the driver",
                mask.0, span.start, span.end
            ),
            MaskLoweringError::UnexpectedSpatialQueryKind { mask, kind, span } => write!(
                f,
                "mask#{} at {}..{} has no `from` clause but the driver supplied SpatialQueryKind::{}",
                mask.0, span.start, span.end, kind
            ),
            MaskLoweringError::BuilderRejected { error, span } => write!(
                f,
                "mask op at {}..{} rejected by builder — {}",
                span.start, span.end, error
            ),
        }
    }
}

impl std::error::Error for MaskLoweringError {}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower a single [`MaskIR`] into a [`ComputeOpKind::MaskPredicate`] op
/// pushed onto `ctx.builder`.
///
/// # Parameters
///
/// - `mask_id`: the [`MaskId`] this mask binds to. Allocated by the
///   driver (Task 2.7 / 2.8) so multiple masks composed in one program
///   land on distinct ids; tests construct ids directly.
/// - `spatial_query_kind`: `Some(kind)` iff the mask carries a `from`
///   clause AND the driver has resolved the candidate source to a
///   spatial query. The lowering does not invent a default — a missing
///   resolution surfaces as
///   [`MaskLoweringError::MissingSpatialQueryKind`].
/// - `ir`: the resolved AST mask. Its `head.name` is interned on the
///   builder (idempotent per id+name); its `predicate` is lowered via
///   [`lower_expr`]; its optional `candidate_source` is recognised in
///   the limited shapes Task 2.2 supports.
/// - `ctx`: the lowering context (carries the in-flight builder, view
///   resolution maps, and diagnostic accumulator).
///
/// # Returns
///
/// The freshly-allocated [`OpId`] of the pushed mask op. The op is
/// retrievable via `ctx.builder.program().ops[id.0 as usize]`.
///
/// # Errors
///
/// See [`MaskLoweringError`] for the closed defect set. Predicate-body
/// failures wrap the underlying [`LoweringError`]; mask-shape concerns
/// (non-Bool predicate, from-clause shape, spatial-query resolution
/// mismatches, builder rejections) surface as their own variants.
///
/// # Side effects
///
/// On success: one expression sub-tree (the predicate) and one op are
/// pushed onto the builder; the mask's name is interned. On failure:
/// any partial sub-tree pushes inside `lower_expr` are left as orphans
/// in the arena (see `lower_expr`'s "Orphan behavior" note); no op is
/// added.
pub fn lower_mask(
    mask_id: MaskId,
    spatial_query_kind: Option<SpatialQueryKind>,
    ir: &MaskIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<OpId, MaskLoweringError> {
    // Step 1: resolve the dispatch shape from the (from-clause,
    // spatial-query-kind) pairing. Done first because a mismatched pair
    // is a hard error that should not produce a half-built program.
    let shape = resolve_dispatch_shape(mask_id, spatial_query_kind, ir)?;

    // Step 2: lower the predicate body. Reuses the expression-level
    // lowering — the predicate is just a `IrExprNode` whose value
    // must be `Bool`.
    let predicate_id = lower_expr(&ir.predicate, ctx).map_err(MaskLoweringError::Predicate)?;

    // Step 3: confirm the predicate type-checks to `Bool`. `lower_expr`
    // already type-checks the node it pushes, but its check is
    // operator-level (claimed_ty vs operand types); this check is
    // mask-level (the bitmap shape requires Bool).
    let predicate_ty = predicate_node_ty(ctx, predicate_id, mask_id, ir.predicate.span)?;
    if predicate_ty != CgTy::Bool {
        return Err(MaskLoweringError::PredicateNotBool {
            mask: mask_id,
            got: predicate_ty,
            span: ir.predicate.span,
        });
    }

    // Step 4: build the op. `add_op` runs its own dangling-id
    // validation against the arena and constructs the `ComputeOp` via
    // `ComputeOp::new` (auto-deriving reads + writes from kind).
    let kind = ComputeOpKind::MaskPredicate {
        mask: mask_id,
        predicate: predicate_id,
    };
    let op_id = ctx
        .builder
        .add_op(kind, shape, ir.span)
        .map_err(|e| MaskLoweringError::BuilderRejected {
            error: e,
            span: ir.span,
        })?;

    // Step 5: intern the mask's source-level name. Idempotent for the
    // same (id, name); a duplicate-name conflict surfaces as a typed
    // builder error.
    ctx.builder
        .intern_mask_name(mask_id, ir.head.name.clone())
        .map_err(|e| MaskLoweringError::BuilderRejected {
            error: e,
            span: ir.head.span,
        })?;

    Ok(op_id)
}

// ---------------------------------------------------------------------------
// Per-step helpers
// ---------------------------------------------------------------------------

/// Resolve `(candidate_source, spatial_query_kind)` to a
/// [`DispatchShape`]. Refuses to invent defaults: a `from` clause
/// without a kind, or a kind without a `from` clause, both surface as
/// typed errors.
fn resolve_dispatch_shape(
    mask_id: MaskId,
    spatial_query_kind: Option<SpatialQueryKind>,
    ir: &MaskIR,
) -> Result<DispatchShape, MaskLoweringError> {
    match (&ir.candidate_source, spatial_query_kind) {
        // Self-only mask — no `from` clause, no spatial query.
        (None, None) => Ok(DispatchShape::PerAgent),

        // Pair-driven mask — `from query.nearby_agents(...)` resolves
        // to a per-pair dispatch over the named spatial query.
        (Some(source), Some(kind)) => {
            validate_from_clause_shape(mask_id, source)?;
            Ok(DispatchShape::PerPair {
                source: PerPairSource::SpatialQuery(kind),
            })
        }

        // Mask has a `from` clause but no kind was resolved — driver
        // must populate the spatial-query map for every from-bearing
        // mask. Surface as typed error rather than fall back to
        // PerAgent (which would silently skip the candidate
        // enumerator).
        (Some(source), None) => Err(MaskLoweringError::MissingSpatialQueryKind {
            mask: mask_id,
            span: source.span,
        }),

        // Inverse bug: caller pre-allocated a spatial query for a
        // self-only mask. Catches driver invariant drift.
        (None, Some(kind)) => Err(MaskLoweringError::UnexpectedSpatialQueryKind {
            mask: mask_id,
            kind,
            span: ir.span,
        }),
    }
}

/// Validate that the `from` clause expression has the recognised
/// `query.nearby_agents(<pos>, <radius>)` shape. Emit and emit_mask
/// already enforce the same constraint at the AST level for the
/// existing direct-emit path; this lowering pass enforces it again
/// inside the CG layer so a future caller (the driver, snapshots) gets
/// the typed defect rather than a silent miscompile.
fn validate_from_clause_shape(
    mask_id: MaskId,
    source: &IrExprNode,
) -> Result<(), MaskLoweringError> {
    match &source.kind {
        IrExpr::NamespaceCall { ns, method, args }
            if *ns == NamespaceId::Query && method == "nearby_agents" && args.len() == 2 =>
        {
            Ok(())
        }
        _ => Err(MaskLoweringError::UnsupportedFromClause {
            mask: mask_id,
            span: source.span,
        }),
    }
}

/// Look up the type of the just-pushed predicate node in the
/// builder's expression arena, surfacing any dangling-id /
/// type-check defect as a typed mask-level error.
fn predicate_node_ty(
    ctx: &LoweringCtx<'_>,
    predicate_id: CgExprId,
    mask_id: MaskId,
    span: Span,
) -> Result<CgTy, MaskLoweringError> {
    let prog = ctx.builder.program();
    let node = prog
        .exprs
        .get(predicate_id.0 as usize)
        .ok_or(MaskLoweringError::PredicateTypeCheckFailure {
            mask: mask_id,
            error: TypeError::DanglingExprId {
                node: predicate_id,
                referenced: predicate_id,
            },
            span,
        })?;
    let resolver: &dyn Fn(crate::cg::data_handle::ViewId) -> Option<(Vec<CgTy>, CgTy)> =
        &|view_id| {
            ctx.view_signatures
                .get(&view_id)
                .map(|(args, result)| (args.clone(), *result))
        };
    let tc_ctx = TypeCheckCtx::with_view_signature(&prog.exprs, resolver);
    type_check(node, predicate_id, &tc_ctx).map_err(|e| {
        MaskLoweringError::PredicateTypeCheckFailure {
            mask: mask_id,
            error: e,
            span,
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use dsl_ast::ast::{BinOp, Span as AstSpan};
    use dsl_ast::ir::{
        IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, LocalRef, MaskIR,
    };

    use crate::cg::data_handle::{AgentFieldId, AgentRef, DataHandle};
    use crate::cg::program::CgProgramBuilder;

    // ---- helpers --------------------------------------------------------

    fn span(start: usize, end: usize) -> AstSpan {
        AstSpan::new(start, end)
    }

    fn node(kind: IrExpr) -> IrExprNode {
        IrExprNode {
            kind,
            span: span(0, 0),
        }
    }

    fn local_self() -> IrExprNode {
        node(IrExpr::Local(LocalRef(0), "self".to_string()))
    }

    fn field_self(name: &str) -> IrExprNode {
        node(IrExpr::Field {
            base: Box::new(local_self()),
            field_name: name.to_string(),
            field: None,
        })
    }

    fn mask_head(name: &str) -> IrActionHead {
        IrActionHead {
            name: name.to_string(),
            shape: IrActionHeadShape::None,
            span: span(0, 0),
        }
    }

    fn arg(value: IrExprNode) -> IrCallArg {
        let s = value.span;
        IrCallArg {
            name: None,
            value,
            span: s,
        }
    }

    fn nearby_agents_call() -> IrExprNode {
        node(IrExpr::NamespaceCall {
            ns: NamespaceId::Query,
            method: "nearby_agents".to_string(),
            args: vec![arg(field_self("pos")), arg(node(IrExpr::LitFloat(20.0)))],
        })
    }

    // ---- Smallest happy path: self-only mask with `agent.alive` ---------

    #[test]
    fn lowers_self_only_alive_predicate_to_per_agent() {
        // mask Hold when agents.alive(self)
        // (Modeled with the simpler `self.alive` shape — same Bool
        // predicate, same lowering.)
        let mask = MaskIR {
            head: mask_head("Hold"),
            candidate_source: None,
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let op_id = lower_mask(MaskId(0), None, &mask, &mut ctx).expect("lowers");

        assert_eq!(op_id, OpId(0));
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::PerAgent);
        match &op.kind {
            ComputeOpKind::MaskPredicate { mask, predicate: _ } => {
                assert_eq!(*mask, MaskId(0));
            }
            other => panic!("unexpected kind: {other:?}"),
        }
        // Auto-derived reads/writes round-trip through ComputeOp::new.
        assert_eq!(
            op.reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(op.writes, vec![DataHandle::MaskBitmap { mask: MaskId(0) }]);
        // Mask name interned for pretty-printing.
        assert_eq!(prog.interner.get_mask_name(MaskId(0)), Some("Hold"));
    }

    // ---- Composite predicate: `hp < 0.5 * max_hp` -----------------------

    #[test]
    fn lowers_composite_low_hp_predicate() {
        // mask LowHp when self.hp < 0.5 * self.max_hp
        let predicate = node(IrExpr::Binary(
            BinOp::Lt,
            Box::new(field_self("hp")),
            Box::new(node(IrExpr::Binary(
                BinOp::Mul,
                Box::new(node(IrExpr::LitFloat(0.5))),
                Box::new(field_self("max_hp")),
            ))),
        ));
        let mask = MaskIR {
            head: mask_head("LowHp"),
            candidate_source: None,
            predicate,
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let op_id = lower_mask(MaskId(2), None, &mask, &mut ctx).expect("lowers");
        assert_eq!(op_id, OpId(0));
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::PerAgent);
        // Reads include both `hp` and `max_hp` (auto-derived from the
        // expression tree).
        assert!(op.reads.contains(&DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        }));
        assert!(op.reads.contains(&DataHandle::AgentField {
            field: AgentFieldId::MaxHp,
            target: AgentRef::Self_,
        }));
        assert_eq!(op.writes, vec![DataHandle::MaskBitmap { mask: MaskId(2) }]);
        assert_eq!(prog.interner.get_mask_name(MaskId(2)), Some("LowHp"));
    }

    // ---- `from` clause + spatial query → PerPair ------------------------

    #[test]
    fn lowers_from_clause_to_per_pair_with_kin_query() {
        // mask MoveToward(target)
        //   from query.nearby_agents(self.pos, 20.0)
        //   when self.alive
        // (Predicate simplified to `self.alive` — Task 2.2 doesn't yet
        // surface the `target` binding through the expression layer;
        // the dispatch shape is what's under test here.)
        let mask = MaskIR {
            head: mask_head("MoveToward"),
            candidate_source: Some(nearby_agents_call()),
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let op_id = lower_mask(
            MaskId(1),
            Some(SpatialQueryKind::KinQuery),
            &mask,
            &mut ctx,
        )
        .expect("lowers");

        assert_eq!(op_id, OpId(0));
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(
            op.shape,
            DispatchShape::PerPair {
                source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
            }
        );
        match &op.kind {
            ComputeOpKind::MaskPredicate { mask, .. } => assert_eq!(*mask, MaskId(1)),
            other => panic!("unexpected kind: {other:?}"),
        }
        assert_eq!(prog.interner.get_mask_name(MaskId(1)), Some("MoveToward"));
    }

    #[test]
    fn lowers_from_clause_with_engagement_query_kind() {
        // Same shape as the KinQuery case, but the driver supplied
        // EngagementQuery — confirms the kind is threaded straight
        // through without translation.
        let mask = MaskIR {
            head: mask_head("Attack"),
            candidate_source: Some(nearby_agents_call()),
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_mask(
            MaskId(3),
            Some(SpatialQueryKind::EngagementQuery),
            &mask,
            &mut ctx,
        )
        .expect("lowers");

        let prog = builder.finish();
        assert_eq!(
            prog.ops[0].shape,
            DispatchShape::PerPair {
                source: PerPairSource::SpatialQuery(SpatialQueryKind::EngagementQuery),
            }
        );
    }

    // ---- Negative: predicate not Bool -----------------------------------

    #[test]
    fn rejects_non_bool_predicate() {
        // `mask X when self.hp` — `hp` is F32, predicate must be Bool.
        let mask = MaskIR {
            head: mask_head("Bad"),
            candidate_source: None,
            predicate: field_self("hp"),
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_mask(MaskId(0), None, &mask, &mut ctx).expect_err("non-bool predicate");
        match err {
            MaskLoweringError::PredicateNotBool { mask, got, .. } => {
                assert_eq!(mask, MaskId(0));
                assert_eq!(got, CgTy::F32);
            }
            other => panic!("expected PredicateNotBool, got {other:?}"),
        }
        // No op was pushed on the builder — only the orphaned predicate
        // sub-tree from `lower_expr` remains.
        let prog = builder.finish();
        assert!(prog.ops.is_empty());
    }

    // ---- Negative: unsupported from-clause shape ------------------------

    #[test]
    fn rejects_unsupported_from_clause_shape() {
        // `from self.pos` — not a `query.nearby_agents` call.
        let mask = MaskIR {
            head: mask_head("Weird"),
            candidate_source: Some(field_self("pos")),
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(11, 22),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_mask(
            MaskId(7),
            Some(SpatialQueryKind::KinQuery),
            &mask,
            &mut ctx,
        )
        .expect_err("unsupported from clause");
        match err {
            MaskLoweringError::UnsupportedFromClause { mask, .. } => {
                assert_eq!(mask, MaskId(7));
            }
            other => panic!("expected UnsupportedFromClause, got {other:?}"),
        }
        let prog = builder.finish();
        assert!(prog.ops.is_empty());
    }

    // ---- Negative: from clause but no spatial-query kind ----------------

    #[test]
    fn rejects_from_clause_without_spatial_query_kind() {
        let mask = MaskIR {
            head: mask_head("MoveToward"),
            candidate_source: Some(nearby_agents_call()),
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_mask(MaskId(0), None, &mask, &mut ctx)
            .expect_err("from clause without resolved kind");
        match err {
            MaskLoweringError::MissingSpatialQueryKind { mask, .. } => {
                assert_eq!(mask, MaskId(0));
            }
            other => panic!("expected MissingSpatialQueryKind, got {other:?}"),
        }
    }

    // ---- Negative: spatial-query kind but no from clause ----------------

    #[test]
    fn rejects_spatial_query_kind_without_from_clause() {
        let mask = MaskIR {
            head: mask_head("Hold"),
            candidate_source: None,
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_mask(
            MaskId(0),
            Some(SpatialQueryKind::KinQuery),
            &mask,
            &mut ctx,
        )
        .expect_err("kind without from");
        match err {
            MaskLoweringError::UnexpectedSpatialQueryKind { mask, kind, .. } => {
                assert_eq!(mask, MaskId(0));
                assert_eq!(kind, SpatialQueryKind::KinQuery);
            }
            other => panic!("expected UnexpectedSpatialQueryKind, got {other:?}"),
        }
    }

    // ---- Negative: predicate body itself fails to lower -----------------

    #[test]
    fn predicate_lowering_failure_propagates() {
        // `self.hp_pct` — not a registered AgentFieldId, so lower_expr
        // returns UnknownAgentField. The mask lowering wraps it in
        // MaskLoweringError::Predicate.
        let mut bad = field_self("hp_pct");
        bad.span = span(3, 9);
        let mask = MaskIR {
            head: mask_head("Bad"),
            candidate_source: None,
            predicate: bad,
            annotations: vec![],
            span: span(0, 30),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_mask(MaskId(0), None, &mask, &mut ctx).expect_err("unknown field");
        match err {
            MaskLoweringError::Predicate(LoweringError::UnknownAgentField {
                field_name,
                ..
            }) => {
                assert_eq!(field_name, "hp_pct");
            }
            other => panic!("expected Predicate(UnknownAgentField), got {other:?}"),
        }
    }

    // ---- Snapshot: pinned `Display` form for a lowered op ---------------

    #[test]
    fn snapshot_self_only_alive_op_display() {
        // Pins the wire format produced by `ComputeOp`'s Display impl
        // for a self-only Bool predicate. Downstream consumers
        // (snapshot tests, structured logs) depend on this exact
        // string shape.
        let mask = MaskIR {
            head: mask_head("Hold"),
            candidate_source: None,
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_mask(MaskId(0), None, &mask, &mut ctx).expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=mask_predicate(mask=#0) shape=per_agent reads=[agent.self.alive] writes=[mask[#0].bitmap]"
        );
    }

    #[test]
    fn snapshot_per_pair_kin_query_op_display() {
        // Pins the wire format for a from-clause mask routed to
        // KinQuery. Confirms `per_pair(spatial_query(kin_query))`
        // renders as expected.
        let mask = MaskIR {
            head: mask_head("MoveToward"),
            candidate_source: Some(nearby_agents_call()),
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_mask(
            MaskId(5),
            Some(SpatialQueryKind::KinQuery),
            &mask,
            &mut ctx,
        )
        .expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=mask_predicate(mask=#5) shape=per_pair(spatial_query(kin_query)) reads=[agent.self.alive] writes=[mask[#5].bitmap]"
        );
    }

    // ---- Display impl coverage for MaskLoweringError --------------------

    #[test]
    fn mask_lowering_error_display_predicate_not_bool() {
        let e = MaskLoweringError::PredicateNotBool {
            mask: MaskId(3),
            got: CgTy::F32,
            span: span(7, 12),
        };
        let s = format!("{}", e);
        assert!(s.contains("mask#3"));
        assert!(s.contains("7..12"));
        assert!(s.contains("Bool"));
    }

    #[test]
    fn mask_lowering_error_display_unsupported_from_clause() {
        let e = MaskLoweringError::UnsupportedFromClause {
            mask: MaskId(0),
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("`from`"));
        assert!(s.contains("nearby_agents"));
    }

    #[test]
    fn mask_lowering_error_display_missing_spatial_query_kind() {
        let e = MaskLoweringError::MissingSpatialQueryKind {
            mask: MaskId(2),
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("mask#2"));
        assert!(s.contains("SpatialQueryKind"));
    }

    #[test]
    fn mask_lowering_error_display_predicate_wraps_inner() {
        let inner = LoweringError::UnknownAgentField {
            field_name: "hp_pct".to_string(),
            span: span(3, 9),
        };
        let e = MaskLoweringError::Predicate(inner);
        let s = format!("{}", e);
        assert!(s.contains("mask predicate"));
        assert!(s.contains("hp_pct"));
    }
}
