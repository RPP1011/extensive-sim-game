//! Mask lowering — `MaskIR → ComputeOpKind::MaskPredicate`.
//!
//! Phase 2, Task 2.2 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`). Each
//! `mask <Name>` decl in the resolved DSL IR becomes one
//! [`crate::cg::op::ComputeOp`] whose [`ComputeOpKind`] is
//! [`ComputeOpKind::MaskPredicate`].
//!
//! The pass:
//!
//! 1. Resolves the dispatch shape from the `(candidate_source,
//!    spatial_query_kind, head.shape)` triple. Mismatches surface as
//!    typed [`LoweringError`] variants (see step list below).
//! 2. Lowers the mask predicate body via [`super::expr::lower_expr`],
//!    reusing the expression-level lowering wholesale.
//! 3. Type-checks the produced node — a mask predicate must be `Bool`.
//! 4. Builds the op via [`crate::cg::program::CgProgramBuilder::add_op`]
//!    (which auto-derives reads/writes from `kind`).
//! 5. Interns the mask name on the builder for pretty-printing.
//!
//! # Typed-error surface
//!
//! All defects surface as variants on the unified
//! [`super::error::LoweringError`]. Mask-specific variants carry the
//! `Mask*` prefix (`MaskPredicateNotBool`, `UnsupportedMaskFromClause`,
//! `UnsupportedMaskHeadShape`, …) per the convention documented on
//! `error.rs`. Predicate-body failures returned by
//! [`super::expr::lower_expr`] propagate unchanged via `?` — there is no
//! wrapper variant.
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

use dsl_ast::ast::Span;
use dsl_ast::ir::{IrActionHeadShape, IrExprNode, MaskIR};

use crate::cg::data_handle::{CgExprId, MaskId};
use crate::cg::dispatch::{DispatchShape, PerPairSource};
use crate::cg::expr::{type_check, CgTy, TypeCheckCtx, TypeError};
use crate::cg::op::{ComputeOpKind, OpId, SpatialQueryKind};

use super::error::LoweringError;
use super::expr::{lower_expr, LoweringCtx};
use super::spatial::try_recognise_spatial_iter;

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
///   [`LoweringError::MissingSpatialQueryKind`].
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
/// See [`LoweringError`] for the closed defect set. Predicate-body
/// failures returned by [`lower_expr`] propagate unchanged; mask-shape
/// concerns (non-Bool predicate, from-clause shape, head-shape gating,
/// spatial-query resolution mismatches, builder rejections) surface as
/// the corresponding `Mask*` / construct-shared variants.
///
/// # Side effects
///
/// On success: one expression sub-tree (the predicate) and one op are
/// pushed onto the builder; the mask's name is interned. On failure:
/// any partial sub-tree pushes inside `lower_expr` are left as orphans
/// in the arena (see `lower_expr`'s "Orphan behavior" note); no op is
/// added.
///
/// # Limitations
///
/// Pair-bound mask predicates that reference `target.<field>` (e.g.,
/// `mask MoveToward(target) from query.nearby_agents(...)` whose
/// predicate reads `target.alive` or `target.pos`) lower cleanly as of
/// Task 5.5a: the lowering sets [`LoweringCtx::target_local`] before
/// lowering the predicate so [`crate::cg::lower::expr::lower_field`]
/// resolves `target.<field>` to a
/// `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`. The
/// emit layer (Task 4.x) is responsible for binding
/// [`crate::cg::data_handle::AgentRef::PerPairCandidate`] to the
/// candidate-side agent slot implied by the dispatch shape's
/// `PerPair { source }` at codegen time.
///
/// Parametric heads without an explicit `from` clause (e.g.,
/// `mask Cast(ability: AbilityId)` from `assets/sim/masks.sim`) are
/// still rejected with [`LoweringError::UnsupportedMaskHeadShape`]
/// until Task 2.6 adds the matching [`PerPairSource`] variant
/// (e.g., `AbilityCatalog`).
pub fn lower_mask(
    mask_id: MaskId,
    spatial_query_kind: Option<SpatialQueryKind>,
    ir: &MaskIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<OpId, LoweringError> {
    // Step 1a: gate parametric heads (`Positional` / `Named`) without a
    // `from` clause. See `LoweringError::UnsupportedMaskHeadShape`'s
    // doc comment for the rationale; Task 2.6 (spatial query lowering)
    // is the harmonization point that will add the matching
    // `PerPairSource::AbilityCatalog` (or similar) variant.
    if let Some(head_label) = parametric_head_label(&ir.head.shape) {
        if ir.candidate_source.is_none() {
            return Err(LoweringError::UnsupportedMaskHeadShape {
                mask: mask_id,
                head_label,
                span: ir.head.span,
            });
        }
    }

    // Step 1b: resolve the dispatch shape from the (from-clause,
    // spatial-query-kind) pairing. Done before lowering the predicate
    // so a mismatched pair is a hard error that does not produce a
    // half-built program.
    let shape = resolve_dispatch_shape(mask_id, spatial_query_kind, ir, ctx)?;

    // Step 2: lower the predicate body. Reuses the expression-level
    // lowering — the predicate is just a `IrExprNode` whose value
    // must be `Bool`.
    //
    // For pair-bound dispatch (`PerPair { source }`), bind `target` as
    // the per-pair candidate context for the duration of the predicate
    // lowering — `target.<field>` then resolves to a
    // `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`
    // via [`super::expr::lower_field`]. The previous flag value is
    // restored after lowering so a recursive mask-inside-physics call
    // (or any future driver pattern that nests lowerings) can't leak
    // the binding upward.
    //
    // # Limitations
    //
    // The flag is a single boolean — it captures "is `target` bound
    // here?" but not which dispatch shape's candidate context. Today
    // only one pair-bound shape exists
    // ([`crate::cg::dispatch::PerPairSource::SpatialQuery`]); when
    // future shapes accrete (e.g., ability-catalog pairs in Task 2.6),
    // the flag may need to grow into a typed enum tracking which
    // candidate context is active. The current shape suffices for
    // Task 5.5a (closing the per-pair mask predicate gap) and
    // Task 5.5b/c (per-pair scoring / fold-body event binders) will
    // tighten it as those land.
    let prev_target_local = ctx.target_local;
    let pair_bound = matches!(shape, DispatchShape::PerPair { .. });
    if pair_bound {
        ctx.target_local = true;
    }
    let predicate_result = lower_expr(&ir.predicate, ctx);
    ctx.target_local = prev_target_local;
    let predicate_id = predicate_result?;

    // Step 3: confirm the predicate type-checks to `Bool`. `lower_expr`
    // already type-checks the node it pushes, but its check is
    // operator-level (claimed_ty vs operand types); this check is
    // mask-level (the bitmap shape requires Bool).
    let predicate_ty = predicate_node_ty(ctx, predicate_id, mask_id, ir.predicate.span)?;
    if predicate_ty != CgTy::Bool {
        return Err(LoweringError::MaskPredicateNotBool {
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
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: ir.span,
        })?;

    // Step 5: intern the mask's source-level name. Idempotent for the
    // same (id, name); a duplicate-name conflict surfaces as a typed
    // builder error.
    ctx.builder
        .intern_mask_name(mask_id, ir.head.name.clone())
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: ir.head.span,
        })?;

    Ok(op_id)
}

// ---------------------------------------------------------------------------
// Per-step helpers
// ---------------------------------------------------------------------------

/// Return a `&'static str` tag for parametric head shapes that today's
/// dispatch surface cannot represent without a `from` clause. Returns
/// `None` for [`IrActionHeadShape::None`] (the self-only mask shape that
/// routes cleanly to [`DispatchShape::PerAgent`]).
///
/// Closed-set tags (`"positional"` | `"named"`) keep the typed-error
/// payload free of `String` allocations; the gate is a `&'static str`
/// match in [`LoweringError::UnsupportedMaskHeadShape`].
fn parametric_head_label(shape: &IrActionHeadShape) -> Option<&'static str> {
    match shape {
        IrActionHeadShape::None => None,
        IrActionHeadShape::Positional(_) => Some("positional"),
        IrActionHeadShape::Named(_) => Some("named"),
    }
}

/// Resolve `(candidate_source, spatial_query_kind)` to a
/// [`DispatchShape`]. Refuses to invent defaults: a `from` clause
/// without a kind, or a kind without a `from` clause, both surface as
/// typed errors.
fn resolve_dispatch_shape(
    mask_id: MaskId,
    spatial_query_kind: Option<SpatialQueryKind>,
    ir: &MaskIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<DispatchShape, LoweringError> {
    match (&ir.candidate_source, spatial_query_kind) {
        // Self-only mask — no `from` clause, no spatial query.
        (None, None) => Ok(DispatchShape::PerAgent),

        // Pair-driven mask — `from query.nearby_agents(...)` resolves
        // to a per-pair dispatch over the named spatial query.
        (Some(source), Some(kind)) => {
            validate_from_clause_shape(mask_id, source, ctx)?;
            Ok(DispatchShape::PerPair {
                source: PerPairSource::SpatialQuery(kind),
            })
        }

        // Mask has a `from` clause but no kind was resolved — driver
        // must populate the spatial-query map for every from-bearing
        // mask. Surface as typed error rather than fall back to
        // PerAgent (which would silently skip the candidate
        // enumerator).
        (Some(source), None) => Err(LoweringError::MissingSpatialQueryKind {
            mask: mask_id,
            span: source.span,
        }),

        // Inverse bug: caller pre-allocated a spatial query for a
        // self-only mask. Catches driver invariant drift.
        (None, Some(kind)) => Err(LoweringError::UnexpectedSpatialQueryKind {
            mask: mask_id,
            kind,
            span: ir.span,
        }),
    }
}

/// Validate that the `from` clause expression has a recognised shape.
///
/// Two shapes are accepted:
///   - `query.nearby_agents(<pos>, <radius>)` — the legacy direct-emit
///     path. `emit` and `emit_mask` already enforce the same
///     constraint at the AST level; this lowering pass mirrors it
///     inside the CG layer so a future caller (the driver,
///     snapshots) gets the typed defect rather than a silent
///     miscompile.
///   - `spatial.<name>(<args...>)` — Phase 7 Task 4. The
///     `spatial_query <name>(self, candidate, …)` declaration
///     supplies the per-pair filter; argument shape and arity are
///     validated by the resolver against the registered
///     declaration. Task 5 wires this through `mask_spatial_kind`
///     to produce a `SpatialQueryKind::FilteredWalk`. Until then
///     this arm is dead-code on the existing fixtures (no
///     `from spatial.*` clause in `assets/sim/`).
///
/// **Slice 2a (stdlib-into-CG-IR):** the recognised-shape table is
/// owned by [`super::spatial::lower_spatial_namespace_call`]; this
/// helper delegates the structural check, then maps unrecognised
/// expression shapes to the mask-specific
/// `UnsupportedMaskFromClause` variant (so mask consumers still
/// see a mask-typed error rather than a generic
/// `UnsupportedNamespaceCall`).
fn validate_from_clause_shape(
    mask_id: MaskId,
    source: &IrExprNode,
    ctx: &mut LoweringCtx<'_>,
) -> Result<(), LoweringError> {
    // Try the shared recogniser first. If the expression is a
    // recognised spatial call, we're done. If it's a malformed spatial
    // call (e.g. `query.nearby_agents` with the wrong arity), the
    // helper's typed error propagates unchanged. If the expression is
    // not a spatial call at all (`Ok(None)`), we map to the
    // mask-specific `UnsupportedMaskFromClause` so the caller sees a
    // mask-typed error.
    match try_recognise_spatial_iter(source, ctx)? {
        Some(_) => Ok(()),
        None => Err(LoweringError::UnsupportedMaskFromClause {
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
) -> Result<CgTy, LoweringError> {
    let prog = ctx.builder.program();
    let node = prog
        .exprs
        .get(predicate_id.0 as usize)
        .ok_or(LoweringError::MaskPredicateTypeCheckFailure {
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
        LoweringError::MaskPredicateTypeCheckFailure {
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
        IrActionHead, IrActionHeadShape, IrCallArg, IrExpr, IrExprNode, IrType, LocalRef, MaskIR,
        NamespaceId,
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


    /// Pair-bound mask whose predicate references `target.alive`.
    /// Confirms the Task 5.5a wiring: the dispatch shape is
    /// `PerPair`, so `lower_mask` flips `ctx.target_local` for the
    /// duration of predicate lowering, and `target.alive` resolves to
    /// `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`.

    /// Restoration test: the flag must be reset on the failure path
    /// (predicate lowering errors out) just as it is on the happy path.


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
            LoweringError::MaskPredicateNotBool { mask, got, .. } => {
                assert_eq!(mask, MaskId(0));
                assert_eq!(got, CgTy::F32);
            }
            other => panic!("expected MaskPredicateNotBool, got {other:?}"),
        }
        // No op was pushed on the builder — only the orphaned predicate
        // sub-tree from `lower_expr` remains.
        let prog = builder.finish();
        assert!(prog.ops.is_empty());
    }

    // ---- Negative: unsupported from-clause shape ------------------------


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
            LoweringError::MissingSpatialQueryKind { mask, .. } => {
                assert_eq!(mask, MaskId(0));
            }
            other => panic!("expected MissingSpatialQueryKind, got {other:?}"),
        }
    }

    // ---- Negative: spatial-query kind but no from clause ----------------


    // ---- Negative: predicate body itself fails to lower -----------------

    #[test]
    fn predicate_lowering_failure_propagates() {
        // `self.<bogus>` — not a registered AgentFieldId nor a virtual
        // field, so lower_expr returns UnknownAgentField. After the
        // unification, the mask pass propagates it unchanged via `?`
        // (no wrapper variant). Historically used `hp_pct`; that name
        // is now a virtual field synthesized to `hp / max_hp`.
        let mut bad = field_self("nonexistent_field");
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
            LoweringError::UnknownAgentField { field_name, .. } => {
                assert_eq!(field_name, "nonexistent_field");
            }
            other => panic!("expected UnknownAgentField, got {other:?}"),
        }
    }

    // ---- Negative: parametric head without `from` clause ----------------

    #[test]
    fn rejects_positional_head_without_from_clause() {
        // `mask Cast(ability: AbilityId)` from `assets/sim/masks.sim` —
        // a `Positional` head with no `from` clause. Dispatch surface
        // can't represent the `(agent × ability)` pair semantics yet,
        // so the lowering refuses with `UnsupportedMaskHeadShape`.
        let head = IrActionHead {
            name: "Cast".to_string(),
            shape: IrActionHeadShape::Positional(vec![(
                "ability".to_string(),
                LocalRef(1),
                IrType::AbilityId,
            )]),
            span: span(5, 25),
        };
        let mask = MaskIR {
            head,
            candidate_source: None,
            predicate: field_self("alive"),
            annotations: vec![],
            span: span(0, 30),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_mask(MaskId(9), None, &mask, &mut ctx)
            .expect_err("parametric head without from");
        match err {
            LoweringError::UnsupportedMaskHeadShape {
                mask,
                head_label,
                span,
            } => {
                assert_eq!(mask, MaskId(9));
                assert_eq!(head_label, "positional");
                // Span points at the head, not the whole mask.
                assert_eq!(span.start, 5);
                assert_eq!(span.end, 25);
            }
            other => panic!("expected UnsupportedMaskHeadShape, got {other:?}"),
        }
        // No op was pushed; the head-shape gate fires before the
        // predicate is even lowered.
        let prog = builder.finish();
        assert!(prog.ops.is_empty());
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


    // ---- Display impl coverage for unified LoweringError mask variants --

    #[test]
    fn lowering_error_display_mask_predicate_not_bool() {
        let e = LoweringError::MaskPredicateNotBool {
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
    fn lowering_error_display_unsupported_mask_from_clause() {
        let e = LoweringError::UnsupportedMaskFromClause {
            mask: MaskId(0),
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("`from`"));
        assert!(s.contains("nearby_agents"));
    }

    #[test]
    fn lowering_error_display_missing_spatial_query_kind() {
        let e = LoweringError::MissingSpatialQueryKind {
            mask: MaskId(2),
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("mask#2"));
        assert!(s.contains("SpatialQueryKind"));
    }

    #[test]
    fn lowering_error_display_unsupported_mask_head_shape() {
        let e = LoweringError::UnsupportedMaskHeadShape {
            mask: MaskId(4),
            head_label: "positional",
            span: span(2, 8),
        };
        let s = format!("{}", e);
        assert!(s.contains("mask#4"));
        assert!(s.contains("positional"));
        assert!(s.contains("from"));
        assert!(s.contains("Task 2.6"));
    }
}
