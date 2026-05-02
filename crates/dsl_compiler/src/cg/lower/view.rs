//! View lowering — `ViewIR → ComputeOpKind::ViewFold | … | ()`.
//!
//! Phase 2, Task 2.3 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`). Each
//! `view <Name>` decl in the resolved DSL IR resolves to one of two
//! shapes:
//!
//! - **Materialized** (`@materialized(...)`, `ViewBodyIR::Fold`): one
//!   [`ComputeOpKind::ViewFold`] op per fold handler, dispatched
//!   [`DispatchShape::PerEvent`] over the handler's source ring. Each
//!   op's body is a [`crate::cg::stmt::CgStmtList`] lowered from the
//!   handler's [`IrStmt`] sequence.
//! - **Lazy** (`@lazy`, `ViewBodyIR::Expr`): no compute ops. The view's
//!   name is interned on the program and call sites continue to use the
//!   parametric [`crate::cg::expr::BuiltinId::ViewCall`] form (registered
//!   by the surrounding driver). Inlining of the lazy body, if/when it
//!   happens, is a later pass.
//!
//! The pass:
//!
//! 1. Resolves the view kind (lazy vs materialized) and gates the body
//!    shape against it; a `@lazy` view with a fold body — or
//!    vice-versa — surfaces as a typed [`LoweringError::ViewKindBodyMismatch`].
//! 2. Interns the view name on the builder.
//! 3. For each fold handler: lowers the handler body to a
//!    [`crate::cg::stmt::CgStmtList`], builds the
//!    [`ComputeOpKind::ViewFold`] kind, picks the
//!    [`DispatchShape::PerEvent { source_ring }`] dispatch, and pushes
//!    the op via [`crate::cg::program::CgProgramBuilder::add_op`]. The
//!    source-ring identity is captured by the dispatch shape itself
//!    (the auto-walker cannot synthesize an explicit
//!    [`crate::cg::data_handle::DataHandle::EventRing`] read; see
//!    `op.rs`'s "Scope of the auto-walker" docs and the
//!    "Source-ring read registration" note below).
//! 4. Validates every [`crate::cg::data_handle::DataHandle::ViewStorage`]
//!    write the body produces against the view's storage hint;
//!    out-of-shape slots surface as
//!    [`LoweringError::InvalidViewStorageSlot`].
//!
//! # Typed-error surface
//!
//! All defects surface as variants on the unified
//! [`super::error::LoweringError`]. View-specific variants carry the
//! `View*` prefix (`UnsupportedViewFoldStmt`, `InvalidViewStorageSlot`,
//! `ViewHandlerResolutionLengthMismatch`, `ViewKindBodyMismatch`) per
//! the convention documented on `error.rs`. Expression-body failures
//! returned by [`super::expr::lower_expr`] propagate unchanged via `?`
//! — there is no wrapper variant.
//!
//! # Source-ring resolution
//!
//! The DSL surface names the events a view folds over via
//! `@materialized(on_event = [...])` plus `on <Event> { ... }` handler
//! patterns; the resolved AST exposes the per-handler event name as
//! `IrEventPattern::name`. The mapping `event-name → (EventKindId,
//! EventRingId)` is a registry concern owned by the surrounding driver
//! (Tasks 2.7 / 2.8). For Task 2.3 the caller supplies a
//! `Vec<HandlerResolution>` aligned with the view's fold-handler list
//! — one `(EventKindId, EventRingId)` per handler, in the same order
//! the AST presents them.
//!
//! # Statement-body coverage
//!
//! Today's `lower_view` recognises a single fold-body statement form:
//! `IrStmt::SelfUpdate { op, value }`. It lowers to
//! `CgStmt::Assign { target: ViewStorage{view, slot: Primary},
//! value: <lowered value expr> }`. The "+= / -= / *= / /=" semantics
//! are *implicit* in the [`ComputeOpKind::ViewFold`] wrapper — the
//! engine knows the fold accumulates per (key, event) under the
//! storage hint's update rule; the body expresses *what to add* (the
//! value), not *how to merge* (the operator). This matches the
//! existing `emit_view_fold_kernel.rs` contract: every real fold body
//! in `assets/sim/views.sim` is a single `self += <expr>` (or `=`),
//! and the kernel does the merge.
//!
//! Other body forms (`If`, `Let`, `Expr`) surface as typed
//! [`LoweringError::UnsupportedViewFoldStmt`] deferrals. `If` would
//! lower cleanly via [`crate::cg::stmt::CgStmt::If`] but no real fold
//! body uses it today; lighting it up is one match-arm away once a
//! consumer needs it. `Let` and bare `Expr` deferral matches the
//! expression-pass precedent.
//!
//! # Source-ring read registration
//!
//! The [`DispatchShape::PerEvent { source_ring }`] field captures the
//! ring identity directly; downstream consumers that need an explicit
//! `op.reads.contains(EventRing { ring, kind: Read })` entry can
//! synthesise it from the dispatch shape. Mirroring the auto-walker
//! convention (Task 1.3) — "structural reads/writes only; registry-
//! resolved ones come from the driver" — `lower_view` does NOT push a
//! redundant `EventRing` read into `op.reads` itself. The
//! [`crate::cg::op::ComputeOp::record_read`] post-construction seam
//! exists for cases where the explicit entry IS desired (e.g., fusion
//! analysis predicated on `op.reads`); the surrounding driver
//! (Tasks 2.7 / 2.8) is the natural place to invoke it once the full
//! op set is built. Task 2.3 itself stops at the dispatch-shape
//! representation.
//!
//! # Limitations
//!
//! - Lazy views are not inlined here. The lazy body is left in the AST
//!   for later passes to consult; the only Task 2.3 side effect for
//!   `@lazy` views is interning the view's name. Call sites continue
//!   to use [`crate::cg::expr::BuiltinId::ViewCall`] (registered by
//!   the surrounding driver). Inline expansion of the lazy body — and
//!   the registration of `BuiltinSignature::ViewCall { view }` for
//!   each lazy view in the program — lands in a follow-up pass
//!   (the driver, Task 2.7+).
//! - The explicit [`crate::cg::data_handle::DataHandle::EventRing`]
//!   read entry for the source ring is NOT recorded by Task 2.3 — see
//!   the "Source-ring read registration" section above. Callers that
//!   need it route through `record_read` post-construction; the
//!   driver (Task 2.7+) is the canonical caller.
//! - The `value` expression of a `SelfUpdate` is lowered through
//!   [`super::expr::lower_expr`], inheriting Task 2.1's coverage. Real
//!   fold bodies reference event-pattern bindings (`on AgentAttacked
//!   { actor: b, target: a } { self += 1.0 }` reads no agent fields —
//!   the value is a literal — but `on EffectStandingDelta { ..., delta
//!   } { self += delta }` references the bound `delta` local). Locals
//!   other than `self` surface as
//!   [`LoweringError::UnsupportedLocalBinding`] until the driver task
//!   wires the per-handler binding scope through `LoweringCtx`.
//! - The AST `IrStmt::Let` / `IrStmt::Expr` / `IrStmt::If` forms
//!   surface as [`LoweringError::UnsupportedViewFoldStmt`]; the resolver
//!   already rejects `Emit`, `For`, `Match`, and `BeliefObserve` inside
//!   fold bodies, so those AST shapes will not normally reach
//!   `lower_view`.
//! - `IrStmt::SelfUpdate` accepts only the `+=` operator. The merge
//!   semantics for `=`, `-=`, `*=`, `/=` are not threaded into
//!   [`ComputeOpKind::ViewFold`] today, so accepting them silently
//!   would lower to the same CG IR as `+=` and silently miscompile.
//!   Non-`+=` operators surface as
//!   [`LoweringError::UnsupportedFoldOperator`]; this gate retires
//!   when `ComputeOpKind::ViewFold` gains an explicit operator field
//!   (Task 2.8 / driver-IR shape change).

use dsl_ast::ir::{
    DecayHint, FoldHandlerIR, IrPattern, IrPatternBinding, IrStmt, StorageHint, ViewBodyIR,
    ViewIR, ViewKind,
};

use crate::cg::data_handle::{CgExprId, DataHandle, EventRingId, ViewId, ViewStorageSlot};
use crate::cg::dispatch::DispatchShape;
use crate::cg::expr::CgExpr;
use crate::cg::op::{ComputeOpKind, EventKindId, OpId};
use crate::cg::stmt::{CgStmt, CgStmtId, CgStmtList};

use super::error::LoweringError;
use super::expr::{lower_expr, LoweringCtx};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-handler resolution supplied by the driver. One entry per
/// [`FoldHandlerIR`] in the view's body, in the same order the AST
/// presents them.
///
/// The driver looks up the handler's `pattern.event` (an AST
/// [`dsl_ast::ir::EventRef`]) in its event registry to produce the
/// typed pair; Task 2.3 accepts the resolved values rather than
/// duplicating the registry walk.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct HandlerResolution {
    /// Stable id for the event variant this handler matches. Becomes
    /// the `on_event` field of the produced
    /// [`ComputeOpKind::ViewFold`] op.
    pub event_kind: EventKindId,
    /// Source event ring driving the dispatch. Becomes the `source_ring`
    /// field of the produced [`DispatchShape::PerEvent`] dispatch. The
    /// ring identity is captured by the dispatch shape *only*; see the
    /// module-level "Source-ring read registration" docstring for why
    /// Task 2.3 does not also push a redundant
    /// [`crate::cg::data_handle::DataHandle::EventRing`] read into
    /// `op.reads`.
    pub source_ring: EventRingId,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower a single [`ViewIR`] to its compute-op set.
///
/// # Parameters
///
/// - `view_id`: the [`ViewId`] this view binds to. Allocated by the
///   driver (Task 2.7 / 2.8); tests construct ids directly.
/// - `ir`: the resolved AST view. Its `name` is interned on the
///   builder; its `kind` selects the lowering shape; its `body` (when
///   `Fold`) is walked handler-by-handler.
/// - `handler_resolutions`: per-handler `(EventKindId, EventRingId)`
///   pairs supplied by the driver. Length must match the AST's fold
///   handler list; an empty slice is required for `@lazy` views and
///   the empty-handlers `@materialized` shape (the latter is rejected
///   by the resolver, but Task 2.3 is defensive).
/// - `ctx`: the lowering context (carries the in-flight builder, view
///   resolution maps, and diagnostic accumulator).
///
/// # Returns
///
/// A `Vec<OpId>` with one entry per fold handler. Lazy views return
/// `Ok(vec![])` — no compute ops are produced; the side effect is
/// interning the view's name.
///
/// # Errors
///
/// See [`LoweringError`] for the closed defect set. Body-statement
/// failures from [`super::expr::lower_expr`] propagate unchanged via
/// `?`. View-shape concerns (kind/body mismatch, unsupported statement
/// form, invalid storage slot, handler-list length mismatch, builder
/// rejections) surface as the corresponding `View*` /
/// construct-shared variants.
///
/// # Side effects
///
/// On success: zero or more expression sub-trees (one per handler-body
/// expression), zero or more statement nodes + lists, zero or more ops
/// pushed onto the builder; the view's name is interned. On failure:
/// any partial sub-trees pushed inside `lower_expr` are left as
/// orphans (see `lower_expr`'s "Orphan behavior" note); no op past
/// the failure point is added.
///
/// # Limitations
///
/// See the module docstring's "Limitations" section. The deferrals
/// today: lazy-view inline expansion (driver task), event-pattern
/// local bindings (driver task), and the `Let` / `Expr` / `If`
/// fold-body statement forms (`UnsupportedViewFoldStmt` until a real
/// consumer needs them).
pub fn lower_view(
    view_id: ViewId,
    ir: &ViewIR,
    handler_resolutions: &[HandlerResolution],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<OpId>, LoweringError> {
    // Single-pass dispatch on `(kind, body)`. The resolver normally
    // enforces the kind/body invariant (`@lazy` requires an Expr body,
    // `@materialized` requires a Fold body — see
    // `validate_view_kind_body_invariants` in resolve.rs) but we
    // re-enforce it here so a snapshot fixture or a hand-built
    // `ViewIR` cannot smuggle a malformed shape past the lowering
    // boundary. The match is exhaustive over the `(ViewKind,
    // ViewBodyIR)` cross-product — four arms, no wildcard.
    match (&ir.kind, &ir.body) {
        // ---- Lazy view path: no compute ops, name interned, return ----
        (ViewKind::Lazy, ViewBodyIR::Expr(_)) => {
            // The lazy body is *not* lowered here. Call sites resolve
            // through at-call-site inlining (Task 5.5c) consulting
            // `ctx.lazy_view_bodies`, which the driver populated
            // during Phase 1. See the module-level "Limitations"
            // docstring.
            //
            // `handler_resolutions` MUST be empty for lazy views — they
            // have no fold handlers. The driver supplying entries here
            // is a defect; we surface it as a typed length mismatch
            // (expected = 0, got = N) so the misuse is observable.
            // Check before name interning so a length-mismatch defect
            // doesn't leave the program with a half-applied side effect.
            if !handler_resolutions.is_empty() {
                return Err(LoweringError::ViewHandlerResolutionLengthMismatch {
                    view: view_id,
                    expected: 0,
                    got: handler_resolutions.len(),
                    span: ir.span,
                });
            }
            intern_view_name(view_id, ir, ctx)?;
            Ok(Vec::new())
        }

        // ---- Materialized view path: one ViewFold op per handler ----
        (ViewKind::Materialized(hint), ViewBodyIR::Fold { handlers, .. }) => {
            // Length-match invariant — driver supplies one resolution
            // per AST handler, in the same order. Checked before name
            // interning so a defect doesn't leave a half-applied side
            // effect on the program.
            if handler_resolutions.len() != handlers.len() {
                return Err(LoweringError::ViewHandlerResolutionLengthMismatch {
                    view: view_id,
                    expected: handlers.len(),
                    got: handler_resolutions.len(),
                    span: ir.span,
                });
            }
            intern_view_name(view_id, ir, ctx)?;
            // Note: view signatures for materialized views are
            // populated in Phase 1 (`populate_view_bodies_and_signatures`)
            // — see driver.rs.
            lower_fold_handlers(
                view_id,
                *hint,
                ir.decay.as_ref(),
                handlers,
                handler_resolutions,
                ctx,
            )
        }

        // ---- Kind/body mismatch arms ----
        (ViewKind::Lazy, ViewBodyIR::Fold { .. }) => Err(LoweringError::ViewKindBodyMismatch {
            view: view_id,
            kind_label: "lazy",
            body_label: "fold",
            span: ir.span,
        }),
        (ViewKind::Materialized(_), ViewBodyIR::Expr(_)) => {
            Err(LoweringError::ViewKindBodyMismatch {
                view: view_id,
                kind_label: "materialized",
                body_label: "expr",
                span: ir.span,
            })
        }
    }
}

/// Intern the view's source-level name on the builder, surfacing a
/// duplicate-name conflict (different name for the same id) as a typed
/// [`LoweringError::BuilderRejected`]. Idempotent for `(id, name)`
/// pairs already present.
fn intern_view_name(
    view_id: ViewId,
    ir: &ViewIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<(), LoweringError> {
    ctx.builder
        .intern_view_name(view_id, ir.name.clone())
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: ir.span,
        })
}

// ---------------------------------------------------------------------------
// Fold-handler lowering
// ---------------------------------------------------------------------------

/// Lower every fold handler on a materialized view. One
/// [`ComputeOpKind::ViewFold`] op per handler. The caller is
/// responsible for ensuring `handlers.len() == handler_resolutions.len()`
/// — `lower_view` checks the invariant before delegating here.
///
/// `decay` is the view's `@decay(...)` annotation if present; it is
/// threaded into the storage-slot validator so `(PairMap, Anchor)`
/// becomes legal precisely when decay is set (see
/// `storage_hint_exposes_slot`'s docstring).
fn lower_fold_handlers(
    view_id: ViewId,
    hint: StorageHint,
    decay: Option<&DecayHint>,
    handlers: &[FoldHandlerIR],
    handler_resolutions: &[HandlerResolution],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<OpId>, LoweringError> {
    debug_assert_eq!(
        handlers.len(),
        handler_resolutions.len(),
        "lower_view checked length match before calling lower_fold_handlers"
    );
    let mut op_ids = Vec::with_capacity(handlers.len());
    for (handler, resolution) in handlers.iter().zip(handler_resolutions.iter()) {
        let op_id = lower_one_handler(view_id, hint, decay, handler, *resolution, ctx)?;
        op_ids.push(op_id);
    }
    Ok(op_ids)
}

/// Lower a single fold handler to one [`ComputeOpKind::ViewFold`] op.
fn lower_one_handler(
    view_id: ViewId,
    hint: StorageHint,
    decay: Option<&DecayHint>,
    handler: &FoldHandlerIR,
    resolution: HandlerResolution,
    ctx: &mut LoweringCtx<'_>,
) -> Result<OpId, LoweringError> {
    // Synthesize a `CgStmt::Let` per fold-handler event-pattern
    // binder. Mirrors the physics path's
    // `synthesize_pattern_binding_lets`: the surface form
    //   on EffectStandingDelta { a: a, b: b, delta: delta } { self += delta }
    // introduces three binders (`a`, `b`, `delta`) read inside the
    // fold body. Without prelude Lets, the body's
    // `IrExpr::Local(local_ref, "delta")` resolves to nothing in
    // `lower_bare_local` and surfaces as `UnsupportedLocalBinding`.
    // The synthesis pushes typed `CgStmt::Let { local, value:
    // CgExpr::EventField{..}, ty }` per binder, registers the
    // `LocalRef → LocalId` and `LocalId → CgTy` mappings on `ctx`,
    // and lets the body lowering walk unchanged.
    let mut prelude_stmt_ids =
        synthesize_pattern_binding_lets(view_id, resolution.event_kind, &handler.pattern.bindings, ctx)?;

    // Lower each statement in the handler body, then wrap the resulting
    // ids into a `CgStmtList`. The `lower_stmt` helper validates the
    // storage-slot invariant per `Assign` it produces — see
    // `lower_stmt`'s docstring.
    let body_stmt_ids = lower_stmt_list(view_id, hint, decay, &handler.body, ctx)?;
    prelude_stmt_ids.extend(body_stmt_ids);
    let stmt_ids = prelude_stmt_ids;
    let list = CgStmtList::new(stmt_ids);
    let body_list_id = ctx
        .builder
        .add_stmt_list(list)
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: handler.span,
        })?;

    // Build the op. `add_op` validates every reference (the body list
    // id) and runs the auto-walker to populate reads/writes from the
    // body's `Assign` targets and expression reads.
    let kind = ComputeOpKind::ViewFold {
        view: view_id,
        on_event: resolution.event_kind,
        body: body_list_id,
    };
    let shape = DispatchShape::PerEvent {
        source_ring: resolution.source_ring,
    };
    let op_id = ctx
        .builder
        .add_op(kind, shape, handler.span)
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: handler.span,
        })?;

    // The source-ring read identity lives on the `DispatchShape::PerEvent
    // { source_ring }` field already — see the module-level
    // "Source-ring read registration" docstring for why Task 2.3 does
    // not push a redundant `DataHandle::EventRing { kind: Read }` entry
    // into `op.reads`. Drivers that need the explicit handle invoke
    // [`crate::cg::op::ComputeOp::record_read`] post-construction.

    Ok(op_id)
}

/// Walk every event-pattern binder on the fold handler's `on <Event>`
/// head, allocate a fresh [`crate::cg::stmt::LocalId`] per binder,
/// resolve `(field_index, ty)` against the driver-supplied event-
/// payload layout, and synthesize one `CgStmt::Let` per binder.
/// Mirrors physics's `synthesize_pattern_binding_lets` — the only
/// reason it isn't shared is that the two callers pass distinct
/// subject ids (`PhysicsRuleId` vs `ViewId`) into diagnostics.
fn synthesize_pattern_binding_lets(
    view_id: ViewId,
    event_kind: EventKindId,
    bindings: &[IrPatternBinding],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<CgStmtId>, LoweringError> {
    use super::error::PatternBindingSubject;

    if bindings.is_empty() {
        return Ok(Vec::new());
    }

    let layout_fields = match ctx.event_layouts.get(&event_kind) {
        Some(l) => l.fields.clone(),
        None => {
            return Err(LoweringError::UnregisteredEventKindLayout {
                subject: PatternBindingSubject::View(view_id),
                event: event_kind,
                span: bindings
                    .first()
                    .map(|b| b.span)
                    .unwrap_or_else(dsl_ast::ast::Span::dummy),
            });
        }
    };

    let mut stmt_ids = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let binder_local = match &binding.value {
            IrPattern::Bind { name: _, local } => *local,
            IrPattern::Struct { .. } => {
                return Err(LoweringError::UnsupportedEventPatternBinding {
                    subject: PatternBindingSubject::View(view_id),
                    field_name: binding.field.clone(),
                    pattern_label: "Struct",
                    span: binding.span,
                });
            }
            IrPattern::Ctor { .. } => {
                return Err(LoweringError::UnsupportedEventPatternBinding {
                    subject: PatternBindingSubject::View(view_id),
                    field_name: binding.field.clone(),
                    pattern_label: "Ctor",
                    span: binding.span,
                });
            }
            IrPattern::Expr(_) => {
                return Err(LoweringError::UnsupportedEventPatternBinding {
                    subject: PatternBindingSubject::View(view_id),
                    field_name: binding.field.clone(),
                    pattern_label: "Expr",
                    span: binding.span,
                });
            }
            IrPattern::Wildcard => continue,
        };

        let field_layout = layout_fields.get(&binding.field).copied().ok_or_else(|| {
            LoweringError::UnregisteredEventFieldLayout {
                subject: PatternBindingSubject::View(view_id),
                event: event_kind,
                field_name: binding.field.clone(),
                span: binding.span,
            }
        })?;

        let local_id = match ctx.local_ids.get(&binder_local).copied() {
            Some(id) => id,
            None => ctx.allocate_local(binder_local),
        };

        let value_expr = CgExpr::EventField {
            event_kind,
            field_index: field_layout.word_offset_in_payload,
            ty: field_layout.ty,
        };
        let value_id = ctx
            .builder
            .add_expr(value_expr)
            .map_err(|e| LoweringError::BuilderRejected {
                error: e,
                span: binding.span,
            })?;
        super::expr::typecheck_node(ctx, value_id, binding.span)?;

        ctx.record_local_ty(local_id, field_layout.ty);

        let stmt = CgStmt::Let {
            local: local_id,
            value: value_id,
            ty: field_layout.ty,
        };
        let stmt_id = ctx
            .builder
            .add_stmt(stmt)
            .map_err(|e| LoweringError::BuilderRejected {
                error: e,
                span: binding.span,
            })?;
        stmt_ids.push(stmt_id);
    }

    Ok(stmt_ids)
}

// ---------------------------------------------------------------------------
// Statement lowering — fold-body subset
// ---------------------------------------------------------------------------

/// Lower a sequence of `IrStmt`s inside a fold-handler body to a list
/// of `CgStmtId`s in source-code order. Each produced `CgStmt` is
/// validated for storage-slot legality before being pushed.
///
/// Kept private to `view.rs` per the plan's guidance: Task 2.4
/// (physics) will need similar helpers but with a different statement
/// vocabulary (cascade `Emit`, agent-field `Assign`, etc.) — promoting
/// to a shared `cg/lower/stmt.rs` is a deliberate choice the
/// physics-task author makes when they have the second user.
fn lower_stmt_list(
    view_id: ViewId,
    hint: StorageHint,
    decay: Option<&DecayHint>,
    body: &[IrStmt],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<CgStmtId>, LoweringError> {
    let mut ids = Vec::with_capacity(body.len());
    for stmt in body {
        let id = lower_stmt(view_id, hint, decay, stmt, ctx)?;
        ids.push(id);
    }
    Ok(ids)
}

/// Lower a single fold-body statement to a `CgStmtId`.
///
/// Recognised shapes:
///
/// - `IrStmt::SelfUpdate { op, value, span }` with `op == "+="` →
///   `CgStmt::Assign { target: ViewStorage{view, slot: Primary},
///   value: <lower_expr(value)> }`. Other operators (`=`, `-=`,
///   `*=`, `/=`, or any unrecognised string) surface as
///   [`LoweringError::UnsupportedFoldOperator`] — the merge
///   semantics for non-`+=` operators are not yet represented on
///   the [`ComputeOpKind::ViewFold`] wrapper, so silent acceptance
///   would miscompile any future fold body that uses them. See the
///   module-level "Statement-body coverage" docs and the
///   [`LoweringError::UnsupportedFoldOperator`] docstring.
///
/// Every other AST statement shape (`Let`, `Expr`, `If`,
/// resolver-rejected `Emit` / `For` / `Match` / `BeliefObserve`)
/// surfaces as [`LoweringError::UnsupportedViewFoldStmt`] with a
/// closed-set `&'static str` tag.
fn lower_stmt(
    view_id: ViewId,
    hint: StorageHint,
    decay: Option<&DecayHint>,
    stmt: &IrStmt,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    match stmt {
        IrStmt::SelfUpdate { op, value, span } => {
            // Operator gate: only `+=` is threaded through the
            // existing `ComputeOpKind::ViewFold` wrapper. The other
            // four canonical operators (`=`, `-=`, `*=`, `/=`) are
            // grammatically valid but semantically distinct merges;
            // accepting them silently would produce identical CG IR
            // to `+=` and silently miscompile (e.g., a multiplicative
            // decay rule). Reject up front with a typed deferral.
            //
            // Defense-in-depth: the resolver enforces the 5-element
            // vocabulary, but the AST holds `op` as a free-form
            // `String`, so unrecognised strings also route here under
            // the closed-set tag `"unknown"`.
            let op_label = canonical_self_update_op_label(op.as_str());
            if op_label != "+=" {
                return Err(LoweringError::UnsupportedFoldOperator {
                    view: view_id,
                    op_label,
                    span: *span,
                });
            }

            // Lower the value expression. `lower_expr` does its own
            // type-checking and pushes the sub-tree into the builder.
            let value_id = lower_expr(value, ctx)?;

            // Build the assign target — Primary slot of the view's
            // storage. The Primary slot is the only slot a fold body
            // ever writes through `self += <expr>` semantics; the
            // hint-specific auxiliary slots (Anchor, Counts, Cursors,
            // Ids) are managed by the kernel wrapper, not by the body.
            // We still validate the slot against the hint so a future
            // shape that needed an alternate primary surface would
            // surface as a typed defect rather than silent miscompile.
            let slot = ViewStorageSlot::Primary;
            validate_storage_slot(view_id, hint, decay, slot, *span)?;
            let target = DataHandle::ViewStorage {
                view: view_id,
                slot,
            };
            push_assign(target, value_id, *span, ctx)
        }
        IrStmt::Let { span, .. } => Err(LoweringError::UnsupportedViewFoldStmt {
            view: view_id,
            ast_label: "Let",
            span: *span,
        }),
        IrStmt::If { span, .. } => Err(LoweringError::UnsupportedViewFoldStmt {
            view: view_id,
            ast_label: "If",
            span: *span,
        }),
        IrStmt::Expr(e) => Err(LoweringError::UnsupportedViewFoldStmt {
            view: view_id,
            ast_label: "Expr",
            span: e.span,
        }),
        IrStmt::Emit(emit) => Err(LoweringError::UnsupportedViewFoldStmt {
            view: view_id,
            ast_label: "Emit",
            span: emit.span,
        }),
        IrStmt::For { span, .. } => Err(LoweringError::UnsupportedViewFoldStmt {
            view: view_id,
            ast_label: "For",
            span: *span,
        }),
        IrStmt::Match { span, .. } => Err(LoweringError::UnsupportedViewFoldStmt {
            view: view_id,
            ast_label: "Match",
            span: *span,
        }),
        IrStmt::BeliefObserve { span, .. } => Err(LoweringError::UnsupportedViewFoldStmt {
            view: view_id,
            ast_label: "BeliefObserve",
            span: *span,
        }),
    }
}

/// Push an `Assign` statement onto the builder, surfacing builder
/// rejection as a typed lowering error.
fn push_assign(
    target: DataHandle,
    value: CgExprId,
    span: dsl_ast::ast::Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    ctx.builder
        .add_stmt(CgStmt::Assign { target, value })
        .map_err(|e| LoweringError::BuilderRejected { error: e, span })
}

/// Validate that `slot` is exposed by the view's `hint`.
///
/// The mapping `(StorageHint, ViewStorageSlot) → valid?` mirrors the
/// table documented on [`ViewStorageSlot`] in `data_handle.rs`. The
/// `decay` parameter encodes the one cell that is conditional on a
/// view-level annotation rather than the hint variant itself —
/// `(PairMap, Anchor)` is exposed precisely when `@decay(...)` is set
/// on the view (the existing `emit_view_fold_kernel` exposes Anchor
/// for `PairMap + @decay` views; without decay there is no anchor
/// pattern to update). Every other cell is unconditional.
///
/// - `PairMap` (no decay) → Primary
/// - `PairMap + @decay` → Primary, Anchor
/// - `PerEntityTopK { k = 1 }` → Primary
/// - `PerEntityTopK { k >= 2 }` → Primary, Ids
/// - `SymmetricPairTopK { k }` → Primary, Counts
/// - `PerEntityRing { k }` → Primary, Cursors
/// - `LazyCached` → Primary
///
/// A slot outside the hint's set surfaces as
/// [`LoweringError::InvalidViewStorageSlot`].
///
/// Note: today's `lower_stmt` only writes to `Primary`, so this
/// validator is checking a tautology in production. It exists so a
/// future statement-form lowering that emits writes to Anchor /
/// Counts / Cursors / Ids automatically inherits the storage-hint
/// gate without needing to re-derive the table.
fn validate_storage_slot(
    view_id: ViewId,
    hint: StorageHint,
    decay: Option<&DecayHint>,
    slot: ViewStorageSlot,
    span: dsl_ast::ast::Span,
) -> Result<(), LoweringError> {
    if storage_hint_exposes_slot(hint, slot, decay) {
        Ok(())
    } else {
        Err(LoweringError::InvalidViewStorageSlot {
            view: view_id,
            hint_label: storage_hint_label(hint),
            requested_slot: slot,
            span,
        })
    }
}

/// Closed-set classification of which slots a hint exposes. Source of
/// truth for [`validate_storage_slot`].
///
/// Encoded as a fully-expanded `(hint, slot)` match — every
/// (`StorageHint`, `ViewStorageSlot`) pair has an explicit arm. Adding
/// a new variant to either enum is a compile-time error here, so the
/// hint-slot table cannot drift silently.
///
/// The `decay` parameter encodes the single conditional cell:
/// `(PairMap, Anchor)` becomes `true` precisely when `decay.is_some()`,
/// matching the `emit_view_fold_kernel`'s anchor-pattern lowering for
/// `PairMap + @decay` views. Every other cell is unconditional —
/// adding `decay` must NOT introduce wildcards in the table; the
/// 5×5 cross-product stays exhaustive so no future hint/slot
/// combination drifts past this gate silently.
fn storage_hint_exposes_slot(
    hint: StorageHint,
    slot: ViewStorageSlot,
    decay: Option<&DecayHint>,
) -> bool {
    use StorageHint as H;
    use ViewStorageSlot as S;
    match (hint, slot) {
        // PairMap → Primary always; Anchor exposed iff @decay is set
        // on the view (the anchor-pattern lowering is the consumer).
        (H::PairMap, S::Primary) => true,
        (H::PairMap, S::Anchor) => decay.is_some(),
        (H::PairMap, S::Ids) => false,
        (H::PairMap, S::Counts) => false,
        (H::PairMap, S::Cursors) => false,

        // PerEntityTopK { k = 1 } → Primary; k >= 2 → Primary + Ids.
        (H::PerEntityTopK { .. }, S::Primary) => true,
        (H::PerEntityTopK { k, .. }, S::Ids) => k >= 2,
        (H::PerEntityTopK { .. }, S::Anchor) => false,
        (H::PerEntityTopK { .. }, S::Counts) => false,
        (H::PerEntityTopK { .. }, S::Cursors) => false,

        // SymmetricPairTopK → Primary + Counts.
        (H::SymmetricPairTopK { .. }, S::Primary) => true,
        (H::SymmetricPairTopK { .. }, S::Counts) => true,
        (H::SymmetricPairTopK { .. }, S::Anchor) => false,
        (H::SymmetricPairTopK { .. }, S::Ids) => false,
        (H::SymmetricPairTopK { .. }, S::Cursors) => false,

        // PerEntityRing → Primary + Cursors.
        (H::PerEntityRing { .. }, S::Primary) => true,
        (H::PerEntityRing { .. }, S::Cursors) => true,
        (H::PerEntityRing { .. }, S::Anchor) => false,
        (H::PerEntityRing { .. }, S::Ids) => false,
        (H::PerEntityRing { .. }, S::Counts) => false,

        // LazyCached → Primary only.
        (H::LazyCached, S::Primary) => true,
        (H::LazyCached, S::Anchor) => false,
        (H::LazyCached, S::Ids) => false,
        (H::LazyCached, S::Counts) => false,
        (H::LazyCached, S::Cursors) => false,
    }
}

/// Map an AST `IrStmt::SelfUpdate.op` string to its canonical
/// closed-set tag. The DSL grammar admits five operators; any other
/// string surfaces as `"unknown"` so a synthetic AST cannot smuggle
/// an unsupported operator past the gate in `lower_stmt`.
fn canonical_self_update_op_label(op: &str) -> &'static str {
    match op {
        "=" => "=",
        "+=" => "+=",
        "-=" => "-=",
        "*=" => "*=",
        "/=" => "/=",
        _ => "unknown",
    }
}

/// Closed-set `&'static str` tag for a [`StorageHint`]. Used in the
/// typed-error payload so the `Display` form names the offending
/// hint without a free-form `format!` allocation.
fn storage_hint_label(hint: StorageHint) -> &'static str {
    match hint {
        StorageHint::PairMap => "pair_map",
        StorageHint::PerEntityTopK { .. } => "per_entity_topk",
        StorageHint::SymmetricPairTopK { .. } => "symmetric_pair_topk",
        StorageHint::PerEntityRing { .. } => "per_entity_ring",
        StorageHint::LazyCached => "lazy_cached",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use dsl_ast::ast::Span as AstSpan;
    use dsl_ast::ir::{
        FoldHandlerIR, IrEmit, IrEventPattern, IrExpr, IrExprNode, IrParam, IrStmt, IrType,
        ViewBodyIR, ViewIR, ViewKind,
    };

    use crate::cg::data_handle::DataHandle;
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

    fn lit_f32(v: f32) -> IrExprNode {
        node(IrExpr::LitFloat(v as f64))
    }

    fn handler_with_self_assign(event_name: &str, value: IrExprNode) -> FoldHandlerIR {
        FoldHandlerIR {
            pattern: IrEventPattern {
                name: event_name.to_string(),
                event: None,
                bindings: vec![],
                span: span(0, 0),
            },
            body: vec![IrStmt::SelfUpdate {
                op: "+=".to_string(),
                value,
                span: span(0, 0),
            }],
            span: span(0, 0),
        }
    }

    fn fold_view(
        name: &str,
        hint: StorageHint,
        handlers: Vec<FoldHandlerIR>,
    ) -> ViewIR {
        ViewIR {
            name: name.to_string(),
            params: vec![IrParam {
                name: "a".to_string(),
                local: dsl_ast::ir::LocalRef(0),
                ty: IrType::AgentId,
                span: span(0, 0),
            }],
            return_ty: IrType::F32,
            body: ViewBodyIR::Fold {
                initial: lit_f32(0.0),
                handlers,
                clamp: None,
            },
            annotations: vec![],
            kind: ViewKind::Materialized(hint),
            decay: None,
            span: span(0, 0),
        }
    }

    fn lazy_view(name: &str, body: IrExprNode) -> ViewIR {
        ViewIR {
            name: name.to_string(),
            params: vec![],
            return_ty: IrType::Bool,
            body: ViewBodyIR::Expr(body),
            annotations: vec![],
            kind: ViewKind::Lazy,
            decay: None,
            span: span(0, 0),
        }
    }

    // ---- Smallest happy path: one handler, Primary slot -----------------

    #[test]
    fn lowers_single_fold_handler_to_view_fold_op() {
        // view threat_level(a, b) -> f32 {
        //   on AgentAttacked { ... } { self += 1.0 }
        // }
        let view = fold_view(
            "threat_level",
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            vec![handler_with_self_assign("AgentAttacked", lit_f32(1.0))],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ops = lower_view(
            ViewId(0),
            &view,
            &[HandlerResolution {
                event_kind: EventKindId(7),
                source_ring: EventRingId(2),
            }],
            &mut ctx,
        )
        .expect("lowers");

        assert_eq!(ops.len(), 1);
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(
            op.shape,
            DispatchShape::PerEvent {
                source_ring: EventRingId(2),
            }
        );
        match &op.kind {
            ComputeOpKind::ViewFold {
                view,
                on_event,
                body: _,
            } => {
                assert_eq!(*view, ViewId(0));
                assert_eq!(*on_event, EventKindId(7));
            }
            other => panic!("unexpected kind: {other:?}"),
        }
        // Auto-walker recorded the body's Primary write.
        assert!(op.writes.contains(&DataHandle::ViewStorage {
            view: ViewId(0),
            slot: ViewStorageSlot::Primary,
        }));
        // The source-ring identity lives on the dispatch shape (see
        // the module-level "Source-ring read registration" doc); Task 2.3
        // does not push a redundant `EventRing` read into `op.reads`.
        assert!(
            op.reads.is_empty(),
            "expected no auto-recorded reads (literal value, source-ring read deferred to driver), got {:?}",
            op.reads
        );
        // View name interned for pretty-printing.
        assert_eq!(prog.interner.get_view_name(ViewId(0)), Some("threat_level"));
    }

    // ---- Lazy view path: zero ops, name interned ------------------------

    #[test]
    fn lowers_lazy_view_to_zero_ops_with_name_interned() {
        // @lazy view is_hostile(...) -> bool { ... }
        // Body uses `LitBool(true)` as a stand-in — the lazy body is
        // not lowered by Task 2.3 anyway, so any expression suffices.
        let view = lazy_view("is_hostile", node(IrExpr::LitBool(true)));

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ops = lower_view(ViewId(3), &view, &[], &mut ctx).expect("lowers");

        assert!(ops.is_empty(), "lazy view should produce no compute ops");
        let prog = builder.finish();
        assert!(prog.ops.is_empty(), "no ops pushed");
        assert_eq!(prog.interner.get_view_name(ViewId(3)), Some("is_hostile"));
    }

    // ---- Multi-handler view: two fold handlers → two ops ----------------

    #[test]
    fn lowers_multi_handler_view_to_one_op_per_handler() {
        // @materialized view threat_level(...) -> f32 {
        //   on AgentAttacked { ... } { self += 1.0 }
        //   on EffectDamageApplied { ... } { self += 1.0 }
        // }
        let view = fold_view(
            "threat_level",
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            vec![
                handler_with_self_assign("AgentAttacked", lit_f32(1.0)),
                handler_with_self_assign("EffectDamageApplied", lit_f32(1.0)),
            ],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ops = lower_view(
            ViewId(1),
            &view,
            &[
                HandlerResolution {
                    event_kind: EventKindId(11),
                    source_ring: EventRingId(2),
                },
                HandlerResolution {
                    event_kind: EventKindId(12),
                    source_ring: EventRingId(2),
                },
            ],
            &mut ctx,
        )
        .expect("lowers");

        assert_eq!(ops.len(), 2);
        let prog = builder.finish();
        assert_eq!(prog.ops.len(), 2);

        // Distinct on_event ids, same source_ring.
        let mut events = Vec::new();
        for op_id in &ops {
            let op = &prog.ops[op_id.0 as usize];
            assert_eq!(
                op.shape,
                DispatchShape::PerEvent {
                    source_ring: EventRingId(2),
                }
            );
            match &op.kind {
                ComputeOpKind::ViewFold { on_event, view, .. } => {
                    assert_eq!(*view, ViewId(1));
                    events.push(on_event.0);
                }
                other => panic!("unexpected kind: {other:?}"),
            }
        }
        assert_eq!(events, vec![11, 12]);
    }

    // ---- Negative: handler-resolution length mismatch -------------------

    #[test]
    fn rejects_handler_resolution_length_mismatch() {
        let view = fold_view(
            "threat_level",
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            vec![
                handler_with_self_assign("AgentAttacked", lit_f32(1.0)),
                handler_with_self_assign("EffectDamageApplied", lit_f32(1.0)),
            ],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_view(
            ViewId(0),
            &view,
            // One resolution entry, two handlers — driver defect.
            &[HandlerResolution {
                event_kind: EventKindId(0),
                source_ring: EventRingId(0),
            }],
            &mut ctx,
        )
        .expect_err("length mismatch");
        match err {
            LoweringError::ViewHandlerResolutionLengthMismatch {
                view, expected, got, ..
            } => {
                assert_eq!(view, ViewId(0));
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected ViewHandlerResolutionLengthMismatch, got {other:?}"),
        }
        let prog = builder.finish();
        // Length check fires before name interning — no side effects on
        // the program when the driver supplies a mismatched count.
        assert!(prog.ops.is_empty());
        assert_eq!(prog.interner.get_view_name(ViewId(0)), None);
    }

    // ---- Negative: lazy view with handler resolutions -------------------

    #[test]
    fn rejects_lazy_view_with_handler_resolutions() {
        let view = lazy_view("is_hostile", node(IrExpr::LitBool(true)));

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_view(
            ViewId(0),
            &view,
            &[HandlerResolution {
                event_kind: EventKindId(0),
                source_ring: EventRingId(0),
            }],
            &mut ctx,
        )
        .expect_err("lazy view with resolutions");
        match err {
            LoweringError::ViewHandlerResolutionLengthMismatch {
                expected, got, ..
            } => {
                assert_eq!(expected, 0);
                assert_eq!(got, 1);
            }
            other => panic!("expected ViewHandlerResolutionLengthMismatch, got {other:?}"),
        }
    }

    // ---- Negative: kind/body mismatch -----------------------------------

    #[test]
    fn rejects_lazy_kind_with_fold_body() {
        // Hand-built `@lazy` view with a fold body — the resolver
        // normally rejects this; lowering's Step-1 gate catches it
        // structurally.
        let mut view = fold_view(
            "is_hostile",
            StorageHint::PerEntityTopK { k: 1, keyed_on: 0 },
            vec![],
        );
        view.kind = ViewKind::Lazy;

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_view(ViewId(0), &view, &[], &mut ctx).expect_err("kind/body mismatch");
        match err {
            LoweringError::ViewKindBodyMismatch {
                view,
                kind_label,
                body_label,
                ..
            } => {
                assert_eq!(view, ViewId(0));
                assert_eq!(kind_label, "lazy");
                assert_eq!(body_label, "fold");
            }
            other => panic!("expected ViewKindBodyMismatch, got {other:?}"),
        }
        // Step-1 fires before name interning — no side effects on the
        // builder.
        let prog = builder.finish();
        assert!(prog.ops.is_empty());
        assert_eq!(prog.interner.get_view_name(ViewId(0)), None);
    }

    #[test]
    fn rejects_materialized_kind_with_expr_body() {
        let mut view = lazy_view("threat_level", node(IrExpr::LitFloat(0.0)));
        view.kind = ViewKind::Materialized(StorageHint::PairMap);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_view(ViewId(0), &view, &[], &mut ctx).expect_err("kind/body mismatch");
        match err {
            LoweringError::ViewKindBodyMismatch {
                kind_label,
                body_label,
                ..
            } => {
                assert_eq!(kind_label, "materialized");
                assert_eq!(body_label, "expr");
            }
            other => panic!("expected ViewKindBodyMismatch, got {other:?}"),
        }
    }

    // ---- Negative: storage-hint slot mismatch ---------------------------

    #[test]
    fn validate_storage_slot_pair_map_admits_primary_only() {
        // Sanity that the table is encoded correctly. Direct test of
        // the validator avoids constructing a full view body that
        // writes Anchor (no real fold body does — see the
        // module-level statement-coverage docs). `decay = None` here
        // is the no-decay PairMap path.
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::PairMap,
            None,
            ViewStorageSlot::Primary,
            span(0, 0)
        )
        .is_ok());

        let err = validate_storage_slot(
            ViewId(0),
            StorageHint::PairMap,
            None,
            ViewStorageSlot::Anchor,
            span(5, 9),
        )
        .expect_err("PairMap without @decay doesn't expose Anchor");
        match err {
            LoweringError::InvalidViewStorageSlot {
                view,
                hint_label,
                requested_slot,
                ..
            } => {
                assert_eq!(view, ViewId(0));
                assert_eq!(hint_label, "pair_map");
                assert_eq!(requested_slot, ViewStorageSlot::Anchor);
            }
            other => panic!("expected InvalidViewStorageSlot, got {other:?}"),
        }
    }

    #[test]
    fn validate_storage_slot_per_entity_topk_k1_no_ids() {
        // K=1 → slot_map shape, only Primary. K>=2 → sparse, Primary+Ids.
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::PerEntityTopK { k: 1, keyed_on: 0 },
            None,
            ViewStorageSlot::Ids,
            span(0, 0)
        )
        .is_err());
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            None,
            ViewStorageSlot::Ids,
            span(0, 0)
        )
        .is_ok());
    }

    #[test]
    fn validate_storage_slot_symmetric_pair_topk_admits_counts() {
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::SymmetricPairTopK { k: 8 },
            None,
            ViewStorageSlot::Counts,
            span(0, 0)
        )
        .is_ok());
        // …but not Cursors.
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::SymmetricPairTopK { k: 8 },
            None,
            ViewStorageSlot::Cursors,
            span(0, 0)
        )
        .is_err());
    }

    #[test]
    fn validate_storage_slot_per_entity_ring_admits_cursors() {
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::PerEntityRing { k: 64 },
            None,
            ViewStorageSlot::Cursors,
            span(0, 0)
        )
        .is_ok());
        // …but not Counts.
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::PerEntityRing { k: 64 },
            None,
            ViewStorageSlot::Counts,
            span(0, 0)
        )
        .is_err());
    }

    #[test]
    fn validate_storage_slot_lazy_cached_admits_primary_only() {
        for slot in [
            ViewStorageSlot::Anchor,
            ViewStorageSlot::Ids,
            ViewStorageSlot::Counts,
            ViewStorageSlot::Cursors,
        ] {
            assert!(validate_storage_slot(
                ViewId(0),
                StorageHint::LazyCached,
                None,
                slot,
                span(0, 0)
            )
            .is_err());
        }
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::LazyCached,
            None,
            ViewStorageSlot::Primary,
            span(0, 0)
        )
        .is_ok());
    }

    // ---- Conditional: PairMap + @decay exposes Anchor -------------------

    #[test]
    fn pair_map_with_decay_exposes_anchor() {
        // Construct a `DecayHint` and confirm the table flips
        // `(PairMap, Anchor)` from rejection to acceptance. This is
        // the only conditional cell — every other cell is unconditional.
        let decay = DecayHint {
            rate: 0.95,
            per: dsl_ast::ir::DecayUnit::Tick,
            span: span(0, 0),
        };
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::PairMap,
            Some(&decay),
            ViewStorageSlot::Anchor,
            span(0, 0)
        )
        .is_ok());
        // Primary is still legal regardless of decay.
        assert!(validate_storage_slot(
            ViewId(0),
            StorageHint::PairMap,
            Some(&decay),
            ViewStorageSlot::Primary,
            span(0, 0)
        )
        .is_ok());
        // The other three slots remain unconditionally rejected even
        // with @decay set.
        for slot in [
            ViewStorageSlot::Ids,
            ViewStorageSlot::Counts,
            ViewStorageSlot::Cursors,
        ] {
            assert!(validate_storage_slot(
                ViewId(0),
                StorageHint::PairMap,
                Some(&decay),
                slot,
                span(0, 0)
            )
            .is_err());
        }
    }

    #[test]
    fn pair_map_without_decay_does_not_expose_anchor() {
        // Negative-direction sanity — `decay = None` keeps the
        // `(PairMap, Anchor)` cell at `false`, matching the no-decay
        // PairMap kernel that has no anchor pattern.
        let err = validate_storage_slot(
            ViewId(0),
            StorageHint::PairMap,
            None,
            ViewStorageSlot::Anchor,
            span(0, 0),
        )
        .expect_err("PairMap without @decay must reject Anchor");
        match err {
            LoweringError::InvalidViewStorageSlot {
                hint_label,
                requested_slot,
                ..
            } => {
                assert_eq!(hint_label, "pair_map");
                assert_eq!(requested_slot, ViewStorageSlot::Anchor);
            }
            other => panic!("expected InvalidViewStorageSlot, got {other:?}"),
        }
    }

    // ---- Negative: unsupported fold-body statement form -----------------

    #[test]
    fn rejects_let_in_fold_body() {
        // A `let` statement in a fold body — the resolver permits
        // `Let`, but Task 2.3's `lower_stmt` defers it.
        let handler = FoldHandlerIR {
            pattern: IrEventPattern {
                name: "AgentAttacked".to_string(),
                event: None,
                bindings: vec![],
                span: span(0, 0),
            },
            body: vec![IrStmt::Let {
                name: "tmp".to_string(),
                local: dsl_ast::ir::LocalRef(0),
                value: lit_f32(1.0),
                span: span(7, 14),
            }],
            span: span(0, 0),
        };
        let view = fold_view(
            "v",
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            vec![handler],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_view(
            ViewId(0),
            &view,
            &[HandlerResolution {
                event_kind: EventKindId(0),
                source_ring: EventRingId(0),
            }],
            &mut ctx,
        )
        .expect_err("Let unsupported in fold body");
        match err {
            LoweringError::UnsupportedViewFoldStmt {
                view,
                ast_label,
                span,
            } => {
                assert_eq!(view, ViewId(0));
                assert_eq!(ast_label, "Let");
                assert_eq!(span.start, 7);
                assert_eq!(span.end, 14);
            }
            other => panic!("expected UnsupportedViewFoldStmt(Let), got {other:?}"),
        }
        let prog = builder.finish();
        assert!(prog.ops.is_empty());
    }

    #[test]
    fn rejects_emit_in_fold_body() {
        // `emit` is rejected by the resolver, but the statement form
        // can still appear in synthetic AST. Confirm the typed tag.
        let handler = FoldHandlerIR {
            pattern: IrEventPattern {
                name: "X".to_string(),
                event: None,
                bindings: vec![],
                span: span(0, 0),
            },
            body: vec![IrStmt::Emit(IrEmit {
                event_name: "X".to_string(),
                event: None,
                fields: vec![],
                span: span(2, 5),
            })],
            span: span(0, 0),
        };
        let view = fold_view(
            "v",
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            vec![handler],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_view(
            ViewId(0),
            &view,
            &[HandlerResolution {
                event_kind: EventKindId(0),
                source_ring: EventRingId(0),
            }],
            &mut ctx,
        )
        .expect_err("Emit unsupported in fold body");
        match err {
            LoweringError::UnsupportedViewFoldStmt { ast_label, .. } => {
                assert_eq!(ast_label, "Emit");
            }
            other => panic!("expected UnsupportedViewFoldStmt(Emit), got {other:?}"),
        }
    }

    // ---- SelfUpdate operator gate ---------------------------------------

    /// Build a fold handler with a single `IrStmt::SelfUpdate` carrying
    /// the supplied operator string. Used by the operator-gate tests
    /// below to exercise non-`+=` and unrecognised-string paths.
    fn handler_with_self_update_op(
        event_name: &str,
        op: &str,
        value: IrExprNode,
        stmt_span: AstSpan,
    ) -> FoldHandlerIR {
        FoldHandlerIR {
            pattern: IrEventPattern {
                name: event_name.to_string(),
                event: None,
                bindings: vec![],
                span: span(0, 0),
            },
            body: vec![IrStmt::SelfUpdate {
                op: op.to_string(),
                value,
                span: stmt_span,
            }],
            span: span(0, 0),
        }
    }

    #[test]
    fn rejects_self_update_with_minus_equals() {
        // `-=` is grammatically valid (resolver permits it) but
        // semantically distinct from `+=`. The CG IR's
        // `ComputeOpKind::ViewFold` does not yet thread the operator,
        // so silent acceptance would lower `self -= x` identically to
        // `self += x` — a guaranteed miscompile. Reject with a typed
        // deferral.
        let view = fold_view(
            "v",
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            vec![handler_with_self_update_op(
                "AgentAttacked",
                "-=",
                lit_f32(1.0),
                span(11, 17),
            )],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_view(
            ViewId(0),
            &view,
            &[HandlerResolution {
                event_kind: EventKindId(0),
                source_ring: EventRingId(0),
            }],
            &mut ctx,
        )
        .expect_err("non-`+=` operator must be rejected");
        match err {
            LoweringError::UnsupportedFoldOperator {
                view,
                op_label,
                span,
            } => {
                assert_eq!(view, ViewId(0));
                assert_eq!(op_label, "-=");
                assert_eq!(span.start, 11);
                assert_eq!(span.end, 17);
            }
            other => panic!("expected UnsupportedFoldOperator, got {other:?}"),
        }
    }

    #[test]
    fn rejects_self_update_with_unknown_operator_string() {
        // Defense-in-depth — the resolver enforces the 5-element
        // operator vocabulary at parse time, but the AST holds the
        // operator as `String`. A synthetic AST that smuggles in an
        // arbitrary string surfaces under the closed-set tag
        // `"unknown"` rather than falling through silently.
        let view = fold_view(
            "v",
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            vec![handler_with_self_update_op(
                "AgentAttacked",
                "&=",
                lit_f32(1.0),
                span(0, 0),
            )],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_view(
            ViewId(0),
            &view,
            &[HandlerResolution {
                event_kind: EventKindId(0),
                source_ring: EventRingId(0),
            }],
            &mut ctx,
        )
        .expect_err("unknown operator string must be rejected");
        match err {
            LoweringError::UnsupportedFoldOperator { op_label, .. } => {
                assert_eq!(op_label, "unknown");
            }
            other => panic!("expected UnsupportedFoldOperator, got {other:?}"),
        }
    }

    #[test]
    fn accepts_self_update_with_plus_equals() {
        // Regression guard — the existing `+=` path must continue to
        // lower successfully through the operator gate. Mirrors the
        // smallest-happy-path test above but pinned next to the
        // negative cases for readability.
        let view = fold_view(
            "v",
            StorageHint::PerEntityTopK { k: 8, keyed_on: 0 },
            vec![handler_with_self_update_op(
                "AgentAttacked",
                "+=",
                lit_f32(1.0),
                span(0, 0),
            )],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ops = lower_view(
            ViewId(0),
            &view,
            &[HandlerResolution {
                event_kind: EventKindId(0),
                source_ring: EventRingId(0),
            }],
            &mut ctx,
        )
        .expect("`+=` is the supported operator and must lower cleanly");
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn lowering_error_display_unsupported_fold_operator() {
        let e = LoweringError::UnsupportedFoldOperator {
            view: ViewId(7),
            op_label: "*=",
            span: span(3, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("view #7"));
        assert!(s.contains("*="));
        assert!(s.contains("only += is lowered today"));
    }

    // ---- Snapshot: pinned `Display` form for a lowered op ---------------

    #[test]
    fn snapshot_single_handler_view_fold_op_display() {
        // Pins the wire format produced by `ComputeOp`'s Display impl
        // for a single-handler ViewFold over a per_entity_ring view.
        // Downstream consumers (snapshot tests, structured logs) depend
        // on this exact string shape.
        //
        // Note: `op.reads` is empty — the source-ring identity is
        // captured in `shape=per_event(ring=#3)`, not duplicated as an
        // `EventRing` data handle. See the module-level "Source-ring
        // read registration" doc for the rationale.
        let view = fold_view(
            "memory",
            StorageHint::PerEntityRing { k: 64 },
            vec![handler_with_self_assign("RecordMemory", lit_f32(1.0))],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_view(
            ViewId(2),
            &view,
            &[HandlerResolution {
                event_kind: EventKindId(5),
                source_ring: EventRingId(3),
            }],
            &mut ctx,
        )
        .expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=view_fold(view=#2, on_event=#5) shape=per_event(ring=#3) reads=[] writes=[view[#2].primary]"
        );
    }

    // ---- Display impl coverage for unified LoweringError view variants --

    #[test]
    fn lowering_error_display_unsupported_view_fold_stmt() {
        let e = LoweringError::UnsupportedViewFoldStmt {
            view: ViewId(3),
            ast_label: "Let",
            span: span(7, 12),
        };
        let s = format!("{}", e);
        assert!(s.contains("view#3"));
        assert!(s.contains("7..12"));
        assert!(s.contains("Let"));
    }

    #[test]
    fn lowering_error_display_handler_resolution_length_mismatch() {
        let e = LoweringError::ViewHandlerResolutionLengthMismatch {
            view: ViewId(2),
            expected: 3,
            got: 1,
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("view#2"));
        assert!(s.contains("3 fold handler"));
        assert!(s.contains("1 (EventKindId, EventRingId) entries"));
    }

    #[test]
    fn lowering_error_display_invalid_view_storage_slot() {
        let e = LoweringError::InvalidViewStorageSlot {
            view: ViewId(4),
            hint_label: "pair_map",
            requested_slot: ViewStorageSlot::Anchor,
            span: span(2, 8),
        };
        let s = format!("{}", e);
        assert!(s.contains("view#4"));
        assert!(s.contains("anchor"));
        assert!(s.contains("pair_map"));
    }

    #[test]
    fn lowering_error_display_view_kind_body_mismatch() {
        let e = LoweringError::ViewKindBodyMismatch {
            view: ViewId(1),
            kind_label: "lazy",
            body_label: "fold",
            span: span(0, 30),
        };
        let s = format!("{}", e);
        assert!(s.contains("view#1"));
        assert!(s.contains("lazy"));
        assert!(s.contains("fold"));
    }

}
