//! Physics-rule lowering — `PhysicsIR → ComputeOpKind::PhysicsRule`.
//!
//! Phase 2, Task 2.4 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`). Each
//! `physics <name> @phase(...) { on <Event> { ... } ... }` decl in the
//! resolved DSL IR produces one [`ComputeOpKind::PhysicsRule`] op per
//! `on <Event>` handler, dispatched [`DispatchShape::PerEvent`] over
//! the handler's source event ring. Each op's body is a
//! [`crate::cg::stmt::CgStmtList`] lowered from the handler's
//! [`IrStmt`] sequence.
//!
//! The pass:
//!
//! 1. Interns the physics rule's name on the builder.
//! 2. For each handler: lowers the handler body to a
//!    [`crate::cg::stmt::CgStmtList`], builds the
//!    [`ComputeOpKind::PhysicsRule`] kind (carrying the per-rule
//!    `replayable` flag from the constitution P7 surface), picks the
//!    [`DispatchShape::PerEvent { source_ring }`] dispatch, and pushes
//!    the op via [`crate::cg::program::CgProgramBuilder::add_op`].
//!
//! # Replayability flag
//!
//! Every physics rule lowers with an explicit
//! [`ReplayabilityFlag`] supplied by the driver. The constitution
//! (P7) gates replayability on the *event* surface, but at the
//! per-rule lowering level the flag is the binding the emit layer
//! consults to decide which ring an `Emit` writes into. Replayable
//! rules (`@phase(event)`) emit into the deterministic ring that
//! folds into the trace hash; non-replayable rules (`@phase(post)`)
//! emit into chronicle / telemetry rings the runtime fold ignores.
//! The driver computes the flag from the rule's `@phase(...)`
//! annotation and threads it through; tests pass it directly.
//!
//! # Statement-body coverage
//!
//! Today's `lower_physics` recognises four body-statement forms:
//!
//! - `IrStmt::Emit { event, fields }` →
//!   [`crate::cg::stmt::CgStmt::Emit`] (after resolving the event
//!   name through [`LoweringCtx::event_kind_ids`] and each field
//!   name through [`LoweringCtx::event_field_indices`] — both are
//!   driver-populated registries; tests populate them directly).
//!   Fielded emits (`emit EffectDamageApplied { actor: caster, ... }`)
//!   light up as of Task 5.5b.
//! - `IrStmt::If { cond, then_body, else_body }` →
//!   [`crate::cg::stmt::CgStmt::If`] (with the cond expression
//!   type-checked to `Bool`).
//! - `IrStmt::Match { scrutinee, arms }` →
//!   [`crate::cg::stmt::CgStmt::Match`] (with each arm's variant +
//!   binders resolved via the driver-supplied
//!   [`LoweringCtx::variant_ids`] / [`LoweringCtx::local_ids`]
//!   maps).
//! - `IrStmt::Let { name, local, value, .. }` →
//!   [`crate::cg::stmt::CgStmt::Let`] (allocates a fresh
//!   [`crate::cg::stmt::LocalId`] when the driver hasn't already
//!   pre-registered the mapping; registers `local_ids[ast_ref] = id`
//!   so the read-side resolution can reach it once Task 5.5d lands).
//!
//! Other AST statement shapes (`IrStmt::For`, `IrStmt::Expr`,
//! `IrStmt::SelfUpdate`, `IrStmt::BeliefObserve`) surface as typed
//! [`LoweringError::UnsupportedPhysicsStmt`] deferrals. The plan
//! body assigns `For` to a future task; bare `Expr` (namespace
//! setter calls like `agents.set_hp(t, x)`) needs namespace
//! lowering (Task 5.5c); `SelfUpdate` is forbidden in physics (the
//! resolver permits it only inside fold bodies); `BeliefObserve`
//! decomposes into a sequence of typed `Assign`s against the
//! BeliefState SoA surface — a separate task.
//!
//! # Field-index resolution
//!
//! `IrStmt::Emit` carries field names as `String`s. Lowering each to
//! a [`crate::cg::stmt::EventField`] requires the per-event field
//! schema (name → index in declaration order). The driver supplies
//! the schema through the per-handler resolution; today the
//! [`HandlerResolution`] struct carries an optional schema vector
//! whose ordering is the variant's declared field order. A field
//! name not present in the schema surfaces as
//! [`LoweringError::UnsupportedPhysicsStmt`] with an
//! `"Emit-unknown-field"` tag — but in practice tests build emit
//! statements with no fields, mirroring the chronicle-class rules
//! (`emit ChronicleEntry { template_id, agent, target }` once the
//! resolver populates locals).
//!
//! # P6 mutation channel
//!
//! Physics rule bodies must NOT contain a
//! [`crate::cg::stmt::CgStmt::Assign`] whose target is a
//! [`crate::cg::data_handle::DataHandle::AgentField`]: the constitution
//! P6 forbids agent-field writes outside `ViewFold`. Today's body
//! coverage doesn't produce `Assign` from any AST shape (no
//! `IrStmt::Assign` exists in the AST surface — agent mutation flows
//! through namespace setter calls which surface as `Expr`, deferred);
//! the well-formed pass (Task 1.6) walks the body for `Assign {
//! target: AgentField, .. }` defensively and reports
//! [`crate::cg::well_formed::CgError::P6Violation`] if a synthetic
//! body smuggles one through.
//!
//! # Source-ring read registration
//!
//! Mirroring [`super::view`]: the [`DispatchShape::PerEvent { source_ring }`]
//! field captures the ring identity directly. Task 2.4 does NOT
//! push a redundant
//! [`crate::cg::data_handle::DataHandle::EventRing`] read entry into
//! `op.reads` — the driver invokes
//! [`crate::cg::op::ComputeOp::record_read`] post-construction if
//! the explicit handle is needed (e.g., for fusion analysis
//! predicated on `op.reads`).
//!
//! # Limitations
//!
//! - Lazy, top-level `Match` is supported but **not within `For`**
//!   loops. Real DSL physics (e.g., the `cast` rule) wraps the
//!   `match op { Damage { … } => … }` form in `for op in
//!   abilities.effects(ab)`; deferring `For` means the `cast` rule
//!   does not lower today. Other physics rules (damage, heal,
//!   shield, stun, slow, transfer_gold, modify_standing,
//!   chronicle_*) don't use `For`.
//! - Real handler bodies routinely use `Let` + namespace setter
//!   calls (`agents.set_hp(t, x)`) which surface as deferred
//!   `UnsupportedPhysicsStmt`. Lighting them up is a follow-up
//!   that requires (a) local-binding resolution at the expression
//!   layer and (b) a typed namespace-call lowering for the agent /
//!   abilities / world setters. Today's tests exercise synthetic
//!   bodies (literal-only emits, lit-conditioned `If`s, and a
//!   variant-resolved `Match`) — same scope discipline as Task 2.3.
//! - Emit field-name → field-index resolution is the driver's
//!   responsibility. The [`HandlerResolution`] type accepts an
//!   optional schema vector; today tests pass an empty `fields` list
//!   on the emit (no payload) so resolution is a no-op. A real
//!   driver populates the schema once it walks the event registry.

use dsl_ast::ast::Span;
use dsl_ast::ir::{
    IrEmit, IrPattern, IrPatternBinding, IrPhysicsPattern, IrStmt, PhysicsHandlerIR, PhysicsIR,
};

use crate::cg::dispatch::DispatchShape;
use crate::cg::op::{ComputeOpKind, OpId, PhysicsRuleId};
use crate::cg::stmt::{CgMatchArm, CgStmt, CgStmtId, CgStmtList, MatchArmBinding};

use super::error::LoweringError;
use super::expr::{lower_expr, LoweringCtx};
use super::view::HandlerResolution;

// Re-export the canonical typed P7 flag at the lowering surface so
// `pub use physics::ReplayabilityFlag` in `mod.rs` continues to work.
// The IR-canonical definition lives in `crate::cg::op` (sibling to
// the other Phase 1 closed-set kinds).
pub use crate::cg::op::ReplayabilityFlag;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower a single [`PhysicsIR`] to its compute-op set.
///
/// # Parameters
///
/// - `rule_id`: the [`PhysicsRuleId`] this rule binds to. Allocated
///   by the driver (Task 2.7 / 2.8); tests construct ids directly.
/// - `replayable`: per-rule [`ReplayabilityFlag`] — propagates to
///   every op produced for this rule's handlers (one per handler).
///   The driver computes the flag from the rule's `@phase(...)`
///   annotation; tests pass it directly.
/// - `ir`: the resolved AST physics rule. Its `name` is interned on
///   the builder; its `handlers` are walked in source order.
/// - `handler_resolutions`: per-handler `(EventKindId, EventRingId)`
///   pairs supplied by the driver. Length must match
///   `ir.handlers.len()`; ordering matches the AST's handler list.
/// - `ctx`: the lowering context (carries the in-flight builder,
///   variant / local resolution maps, and diagnostic accumulator).
///
/// # Returns
///
/// A `Vec<OpId>` with one entry per handler.
///
/// # Errors
///
/// See [`LoweringError`] for the closed defect set. Body-statement
/// failures from [`super::expr::lower_expr`] propagate unchanged via
/// `?`. Physics-shape concerns surface as `Physics*` /
/// construct-shared variants.
///
/// # Side effects
///
/// On success: zero or more expression sub-trees, zero or more
/// statement nodes + lists, one op per handler pushed onto the
/// builder; the rule's name is interned. On failure: any partial
/// sub-trees pushed inside `lower_expr` are left as orphans (see
/// `lower_expr`'s "Orphan behavior" note); no op past the failure
/// point is added.
///
/// # Limitations
///
/// See the module docstring's "Limitations" section. Today's
/// supported body-statement forms are `Emit`, `If`, and `Match`;
/// other shapes (`Let`, `For`, `Expr`, `SelfUpdate`,
/// `BeliefObserve`) surface as
/// [`LoweringError::UnsupportedPhysicsStmt`] deferrals. Real DSL
/// physics rules routinely use `Let` + namespace setter calls; the
/// driver task lights those up once the expression layer grows
/// local-binding + namespace-call resolution.
pub fn lower_physics(
    rule_id: PhysicsRuleId,
    replayable: ReplayabilityFlag,
    ir: &PhysicsIR,
    handler_resolutions: &[HandlerResolution],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<OpId>, LoweringError> {
    // Length-match invariant — driver supplies one resolution per AST
    // handler, in the same order. Checked before name interning so a
    // defect doesn't leave a half-applied side effect on the program.
    if handler_resolutions.len() != ir.handlers.len() {
        return Err(LoweringError::PhysicsHandlerResolutionLengthMismatch {
            rule: rule_id,
            expected: ir.handlers.len(),
            got: handler_resolutions.len(),
            span: ir.span,
        });
    }

    intern_rule_name(rule_id, ir, ctx)?;

    let mut op_ids = Vec::with_capacity(ir.handlers.len());
    for (handler, resolution) in ir.handlers.iter().zip(handler_resolutions.iter()) {
        let op_id = lower_one_handler(rule_id, replayable, handler, *resolution, ctx)?;
        op_ids.push(op_id);
    }
    Ok(op_ids)
}

/// Intern the rule's source-level name on the builder, surfacing a
/// duplicate-name conflict (different name for the same id) as a typed
/// [`LoweringError::BuilderRejected`]. Idempotent for `(id, name)`
/// pairs already present.
fn intern_rule_name(
    rule_id: PhysicsRuleId,
    ir: &PhysicsIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<(), LoweringError> {
    ctx.builder
        .intern_physics_rule_name(rule_id, ir.name.clone())
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: ir.span,
        })
}

// ---------------------------------------------------------------------------
// Per-handler lowering
// ---------------------------------------------------------------------------

/// Lower a single physics handler to one [`ComputeOpKind::PhysicsRule`] op.
fn lower_one_handler(
    rule_id: PhysicsRuleId,
    replayable: ReplayabilityFlag,
    handler: &PhysicsHandlerIR,
    resolution: HandlerResolution,
    ctx: &mut LoweringCtx<'_>,
) -> Result<OpId, LoweringError> {
    // The handler's `where_clause` is part of the dispatch predicate,
    // not the body — at this layer we treat it as a deferral until
    // the driver wires per-handler-where filtering. None of today's
    // physics rules use `where`, so this branch is a defensive
    // no-op.
    if let Some(_where_expr) = &handler.where_clause {
        // The rule-pattern carries its own span; reuse it for the
        // diagnostic anchor. Surfacing as a deferred AST node lets
        // the driver task light it up cleanly when needed.
        return Err(LoweringError::UnsupportedPhysicsStmt {
            rule: rule_id,
            ast_label: "where",
            span: handler.pattern.span(),
        });
    }

    // Tag-pattern handlers (`on tag SomeTag { ... }`) are resolved
    // at registry-walk time but not yet lowered through `lower_physics`
    // — the driver is the natural caller because tag dispatch
    // expands to N kind-pattern handlers. Today no real physics
    // rule uses tag patterns, so this branch is a defensive
    // deferral.
    if let IrPhysicsPattern::Tag { span, .. } = &handler.pattern {
        return Err(LoweringError::UnsupportedPhysicsStmt {
            rule: rule_id,
            ast_label: "TagPattern",
            span: *span,
        });
    }

    // Synthesize a `CgStmt::Let` per event-pattern binder BEFORE
    // walking the body. The handler's surface form is
    //   on EffectDamageApplied { actor: c, target: t, amount: a } { ... }
    // where `c`, `t`, `a` are `LocalRef`s introduced by the resolver
    // and read inside the body. The IR's `lower_bare_local` chain
    // resolves bare-local reads through `ctx.local_ids` + `ctx.local_tys`
    // — so we make each binder a real `LocalId` whose value comes
    // from a typed `CgExpr::EventField` keyed on the event's
    // `word_offset_in_payload` + `field_ty` schema. After this prelude
    // the existing body lowering walks the user-authored statements
    // unchanged and every read of `c`/`t`/`a` resolves cleanly.
    //
    // Schema lookup goes through `ctx.event_layouts[on_event]`. A
    // missing entry surfaces as `UnregisteredEventKindLayout`; a
    // missing field within the layout (e.g. the binder named a field
    // the event doesn't declare) surfaces as
    // `UnregisteredEventField`.
    let mut prelude_stmt_ids = super::event_binding::synthesize_pattern_binding_lets(
        super::error::PatternBindingSubject::Physics(rule_id),
        resolution.event_kind,
        handler.pattern.bindings(),
        ctx,
    )?;

    // Lower the handler body.
    let body_stmt_ids = lower_stmt_list(rule_id, &handler.body, ctx)?;
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
    // body's `Assign` targets, expression reads, and Match arm bodies.
    let kind = ComputeOpKind::PhysicsRule {
        rule: rule_id,
        on_event: resolution.event_kind,
        body: body_list_id,
        replayable,
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
    // "Source-ring read registration" docstring for why Task 2.4 does
    // not push a redundant `DataHandle::EventRing { kind: Read }` entry
    // into `op.reads`. Drivers that need the explicit handle invoke
    // [`crate::cg::op::ComputeOp::record_read`] post-construction.

    Ok(op_id)
}

// ---------------------------------------------------------------------------
// Statement lowering — physics-body subset
// ---------------------------------------------------------------------------

/// Lower a sequence of `IrStmt`s inside a physics handler body to a
/// list of `CgStmtId`s in source-code order.
///
/// Kept private to `physics.rs` per the post-2.3 plan guidance:
/// promotion to a shared `cg/lower/stmt.rs` is a deliberate choice
/// that lands when a third concrete user emerges. View and physics
/// have overlapping but not identical statement vocabularies (view
/// only handles `SelfUpdate`; physics handles `Emit` + `If` +
/// `Match`); generalising prematurely would force one helper to
/// thread both the view storage-hint validator and the physics
/// variant registry.
fn lower_stmt_list(
    rule_id: PhysicsRuleId,
    body: &[IrStmt],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<CgStmtId>, LoweringError> {
    let mut ids = Vec::with_capacity(body.len());
    for stmt in body {
        let id = lower_stmt(rule_id, stmt, ctx)?;
        ids.push(id);
    }
    Ok(ids)
}

/// Lower a single physics-body statement to a `CgStmtId`.
///
/// Recognised shapes:
///
/// - `IrStmt::Emit(IrEmit { event_name, event, fields, span })` →
///   `CgStmt::Emit { event: <resolved EventKindId>, fields: <empty
///   for now until driver wires field-schema> }`. The
///   `event_name → EventKindId` mapping comes from the
///   per-handler resolution today (Task 2.4 wires only the
///   already-known EventKindId from the dispatch shape; field
///   resolution is the driver's responsibility — see the module-
///   level "Field-index resolution" docs).
/// - `IrStmt::If { cond, then_body, else_body }` →
///   `CgStmt::If { cond, then, else_ }` (cond is type-checked to
///   `Bool` by the well-formed pass; lowering does not pre-validate
///   the type — `lower_expr` already type-checks each constructed
///   node).
/// - `IrStmt::Match { scrutinee, arms }` →
///   `CgStmt::Match { scrutinee, arms }` (each arm's variant +
///   binders resolved through `ctx.variant_ids` / `ctx.local_ids`).
///
/// Other AST shapes surface as typed
/// [`LoweringError::UnsupportedPhysicsStmt`].
fn lower_stmt(
    rule_id: PhysicsRuleId,
    stmt: &IrStmt,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    match stmt {
        IrStmt::Emit(emit) => lower_emit(rule_id, emit, ctx),
        IrStmt::If {
            cond,
            then_body,
            else_body,
            span,
        } => lower_if(rule_id, cond, then_body, else_body.as_deref(), *span, ctx),
        IrStmt::Match {
            scrutinee,
            arms,
            span,
        } => lower_match(rule_id, scrutinee, arms, *span, ctx),
        IrStmt::Let {
            local,
            value,
            span,
            ..
        } => lower_let(rule_id, *local, value, *span, ctx),
        IrStmt::For { span, .. } => Err(LoweringError::UnsupportedPhysicsStmt {
            rule: rule_id,
            ast_label: "For",
            span: *span,
        }),
        IrStmt::Expr(e) => Err(LoweringError::UnsupportedPhysicsStmt {
            rule: rule_id,
            ast_label: "Expr",
            span: e.span,
        }),
        IrStmt::SelfUpdate { span, .. } => Err(LoweringError::UnsupportedPhysicsStmt {
            rule: rule_id,
            ast_label: "SelfUpdate",
            span: *span,
        }),
        IrStmt::BeliefObserve { span, .. } => Err(LoweringError::UnsupportedPhysicsStmt {
            rule: rule_id,
            ast_label: "BeliefObserve",
            span: *span,
        }),
    }
}

/// Lower an `IrStmt::Emit` to `CgStmt::Emit`.
///
/// The emit's destination event-name is resolved through
/// [`LoweringCtx::event_kind_ids`] — the dedicated event-kind
/// registry (distinct from [`LoweringCtx::variant_ids`], which is
/// the sum-type variant registry consulted by `Match` arm
/// lowering). The two id spaces are independent: a driver
/// allocating `Damage → VariantId(0)` and `AgentDied →
/// EventKindId(0)` produces no ambiguity at the emit / match
/// dispatch sites because each side reads its own map. The Task
/// 2.4 follow-up split this registry to make the dispatch typed
/// end-to-end.
///
/// Inside an arm body, the enclosing op's `on_event` is not the
/// emit's *destination* event — the emit may target a different
/// event variant (e.g., `physics damage` reads `EffectDamageApplied`
/// and emits `AgentAttacked` + `AgentDied`). The driver populates
/// the registry from the event surface; tests populate it
/// directly via [`LoweringCtx::register_event_kind`].
///
/// As of Task 5.5b, fielded emits lower to `CgStmt::Emit { event,
/// fields: Vec<(EventField, CgExprId)> }` — each
/// `IrFieldInit { name, value }` resolves through
/// [`LoweringCtx::event_field_indices`] (the per-event field-name →
/// index registry) to a typed [`crate::cg::stmt::EventField`] with
/// `(event, index)`, and `value` lowers via [`super::expr::lower_expr`].
/// A missing field-index entry surfaces as
/// [`LoweringError::UnknownEventField`]. An unknown event-name
/// surfaces as [`LoweringError::UnknownMatchVariant`] (the closed-set
/// name is reused to avoid widening the error vocabulary for a
/// deferral that the same driver task lights up).
///
/// # Limitations
///
/// - The driver populates `event_field_indices` from the event
///   variant's declared field list (Task 5.7); tests pre-populate
///   the map directly via [`LoweringCtx::register_event_field`].
fn lower_emit(
    rule_id: PhysicsRuleId,
    emit: &IrEmit,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    let event_kind = ctx
        .event_kind_ids
        .get(&emit.event_name)
        .copied()
        .ok_or_else(|| LoweringError::UnknownMatchVariant {
            rule: rule_id,
            variant_name: emit.event_name.clone(),
            span: emit.span,
        })?;

    // Resolve each field name → typed (EventField, value-expr) pair.
    // Field order matches the source AST (the `IrFieldInit` list);
    // the schema entry pins the field's variant-relative index.
    let mut cg_fields = Vec::with_capacity(emit.fields.len());
    for field_init in &emit.fields {
        let index = ctx
            .event_field_indices
            .get(&(event_kind, field_init.name.clone()))
            .copied()
            .ok_or_else(|| LoweringError::UnknownEventField {
                event: event_kind,
                field_name: field_init.name.clone(),
                span: field_init.span,
            })?;
        let value_id = lower_expr(&field_init.value, ctx)?;
        cg_fields.push((
            crate::cg::stmt::EventField {
                event: event_kind,
                index,
            },
            value_id,
        ));
    }

    let stmt = CgStmt::Emit {
        event: event_kind,
        fields: cg_fields,
    };
    push_stmt(stmt, emit.span, ctx)
}

/// Lower an `IrStmt::If` to `CgStmt::If`.
fn lower_if(
    rule_id: PhysicsRuleId,
    cond: &dsl_ast::ir::IrExprNode,
    then_body: &[IrStmt],
    else_body: Option<&[IrStmt]>,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    let cond_id = lower_expr(cond, ctx)?;

    // Lower the then-arm body to a CgStmtList.
    let then_ids = lower_stmt_list(rule_id, then_body, ctx)?;
    let then_list_id = ctx
        .builder
        .add_stmt_list(CgStmtList::new(then_ids))
        .map_err(|e| LoweringError::BuilderRejected { error: e, span })?;

    // Lower the optional else-arm body.
    let else_list_id = match else_body {
        Some(body) => {
            let ids = lower_stmt_list(rule_id, body, ctx)?;
            let id = ctx
                .builder
                .add_stmt_list(CgStmtList::new(ids))
                .map_err(|e| LoweringError::BuilderRejected { error: e, span })?;
            Some(id)
        }
        None => None,
    };

    let stmt = CgStmt::If {
        cond: cond_id,
        then: then_list_id,
        else_: else_list_id,
    };
    push_stmt(stmt, span, ctx)
}

/// Lower an `IrStmt::Let { local, value, .. }` to `CgStmt::Let`.
///
/// Allocates a fresh [`crate::cg::stmt::LocalId`] for the binding
/// (or reuses a driver-pre-registered one if present) and registers
/// the AST `LocalRef → LocalId` mapping in
/// [`LoweringCtx::local_ids`] so subsequent expression-tree
/// references to the same binder can resolve through the same map
/// once the read-side wiring lands.
///
/// # Limitations
///
/// References to the bound local from later statements still
/// surface as
/// [`LoweringError::UnsupportedLocalBinding`] at the expression
/// layer — wiring the read-side resolution of `IrExpr::Local` is a
/// separate task (5.5d). Today's `Let` arm represents the binding
/// structurally; consumers (later statements that read the local)
/// are a follow-up.
fn lower_let(
    _rule_id: PhysicsRuleId,
    ast_local: dsl_ast::ir::LocalRef,
    value: &dsl_ast::ir::IrExprNode,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    // Lower the bound value expression first so the type-check has
    // already run and the type is known before we declare the local.
    let value_id = lower_expr(value, ctx)?;
    let value_ty = super::expr::typecheck_node(ctx, value_id, value.span)?;

    // Resolve the local id. If the driver pre-registered the
    // mapping (the future shape), reuse it; otherwise allocate a
    // fresh id and store it. Both paths leave `ctx.local_ids`
    // populated for the read-side resolution to consult.
    let local_id = match ctx.local_ids.get(&ast_local).copied() {
        Some(id) => id,
        None => ctx.allocate_local(ast_local),
    };

    // Task 5.5d: record the binding's CG type so bare-local reads
    // (`IrExpr::Local(local_ref, _)` resolved to this binding's
    // `LocalId`) can reconstruct `CgExpr::ReadLocal { local, ty }`.
    ctx.record_local_ty(local_id, value_ty);

    let stmt = CgStmt::Let {
        local: local_id,
        value: value_id,
        ty: value_ty,
    };
    push_stmt(stmt, span, ctx)
}

/// Lower an `IrStmt::Match` to `CgStmt::Match`.
///
/// Each arm's pattern must be `IrPattern::Struct { name, bindings }`
/// — the shape stdlib `EffectOp` matches use. The `name` resolves
/// through `ctx.variant_ids` to a [`crate::cg::stmt::VariantId`];
/// each binding resolves through `ctx.local_ids` to a
/// [`crate::cg::stmt::LocalId`].
fn lower_match(
    rule_id: PhysicsRuleId,
    scrutinee: &dsl_ast::ir::IrExprNode,
    arms: &[dsl_ast::ir::IrStmtMatchArm],
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    if arms.is_empty() {
        return Err(LoweringError::EmptyMatchArms {
            rule: rule_id,
            span,
        });
    }

    let scrutinee_id = lower_expr(scrutinee, ctx)?;

    let mut cg_arms = Vec::with_capacity(arms.len());
    for arm in arms {
        cg_arms.push(lower_match_arm(rule_id, arm, ctx)?);
    }

    let stmt = CgStmt::Match {
        scrutinee: scrutinee_id,
        arms: cg_arms,
    };
    push_stmt(stmt, span, ctx)
}

/// Lower a single match arm. Returns the typed [`CgMatchArm`].
fn lower_match_arm(
    rule_id: PhysicsRuleId,
    arm: &dsl_ast::ir::IrStmtMatchArm,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgMatchArm, LoweringError> {
    // Resolve the arm pattern. Today only `IrPattern::Struct {
    // name, bindings }` is wired (the stdlib enum-variant match
    // shape).
    let (variant_id, bindings) = match &arm.pattern {
        IrPattern::Struct { name, bindings, .. } => {
            let variant = ctx.variant_ids.get(name).copied().ok_or_else(|| {
                LoweringError::UnknownMatchVariant {
                    rule: rule_id,
                    variant_name: name.clone(),
                    span: arm.span,
                }
            })?;
            let cg_bindings = lower_struct_bindings(rule_id, bindings, ctx)?;
            (variant, cg_bindings)
        }
        IrPattern::Bind { .. } => {
            return Err(LoweringError::UnsupportedMatchPattern {
                rule: rule_id,
                pattern_label: "Bind",
                span: arm.span,
            })
        }
        IrPattern::Ctor { .. } => {
            return Err(LoweringError::UnsupportedMatchPattern {
                rule: rule_id,
                pattern_label: "Ctor",
                span: arm.span,
            })
        }
        IrPattern::Expr(_) => {
            return Err(LoweringError::UnsupportedMatchPattern {
                rule: rule_id,
                pattern_label: "Expr",
                span: arm.span,
            })
        }
        IrPattern::Wildcard => {
            return Err(LoweringError::UnsupportedMatchPattern {
                rule: rule_id,
                pattern_label: "Wildcard",
                span: arm.span,
            })
        }
    };

    // Lower the arm body.
    let stmt_ids = lower_stmt_list(rule_id, &arm.body, ctx)?;
    let list = CgStmtList::new(stmt_ids);
    let body_id = ctx
        .builder
        .add_stmt_list(list)
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: arm.span,
        })?;

    Ok(CgMatchArm {
        variant: variant_id,
        bindings,
        body: body_id,
    })
}

/// Lower the binding list inside an `IrPattern::Struct` to typed
/// [`MatchArmBinding`]s.
fn lower_struct_bindings(
    rule_id: PhysicsRuleId,
    bindings: &[IrPatternBinding],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<MatchArmBinding>, LoweringError> {
    let mut out = Vec::with_capacity(bindings.len());
    for b in bindings {
        out.push(lower_struct_binding(rule_id, b, ctx)?);
    }
    Ok(out)
}

/// Lower a single struct-pattern binding. Today only the shorthand
/// `IrPattern::Bind { name, local }` form (matches `Damage { amount
/// }` where `amount` introduces a binder of the same name) is
/// recognised.
fn lower_struct_binding(
    rule_id: PhysicsRuleId,
    binding: &IrPatternBinding,
    ctx: &mut LoweringCtx<'_>,
) -> Result<MatchArmBinding, LoweringError> {
    match &binding.value {
        IrPattern::Bind { name, local } => {
            let local_id = ctx.local_ids.get(local).copied().ok_or_else(|| {
                LoweringError::UnknownLocalRef {
                    rule: rule_id,
                    binder_name: name.clone(),
                    span: binding.span,
                }
            })?;
            Ok(MatchArmBinding {
                field_name: binding.field.clone(),
                local: local_id,
            })
        }
        IrPattern::Struct { .. } => Err(LoweringError::UnsupportedMatchBindingShape {
            rule: rule_id,
            field_name: binding.field.clone(),
            pattern_label: "Struct",
            span: binding.span,
        }),
        IrPattern::Ctor { .. } => Err(LoweringError::UnsupportedMatchBindingShape {
            rule: rule_id,
            field_name: binding.field.clone(),
            pattern_label: "Ctor",
            span: binding.span,
        }),
        IrPattern::Expr(_) => Err(LoweringError::UnsupportedMatchBindingShape {
            rule: rule_id,
            field_name: binding.field.clone(),
            pattern_label: "Expr",
            span: binding.span,
        }),
        IrPattern::Wildcard => Err(LoweringError::UnsupportedMatchBindingShape {
            rule: rule_id,
            field_name: binding.field.clone(),
            pattern_label: "Wildcard",
            span: binding.span,
        }),
    }
}

/// Push a constructed `CgStmt` onto the builder, surfacing builder
/// rejection as a typed lowering error.
fn push_stmt(
    stmt: CgStmt,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    ctx.builder
        .add_stmt(stmt)
        .map_err(|e| LoweringError::BuilderRejected { error: e, span })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use dsl_ast::ast::Span as AstSpan;
    use dsl_ast::ir::{
        IrEmit, IrEventPattern, IrExpr, IrExprNode, IrFieldInit, IrPatternBinding,
        IrPhysicsPattern, IrStmt, IrStmtMatchArm, LocalRef, PhysicsHandlerIR, PhysicsIR,
    };

    use crate::cg::data_handle::EventRingId;
    use crate::cg::op::EventKindId;
    use crate::cg::program::CgProgramBuilder;
    use crate::cg::stmt::{CgStmt, EventField, LocalId, VariantId};

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

    fn lit_bool(v: bool) -> IrExprNode {
        node(IrExpr::LitBool(v))
    }

    fn lit_f32(v: f32) -> IrExprNode {
        node(IrExpr::LitFloat(v as f64))
    }

    /// Build a single-handler `PhysicsIR` whose body is the supplied
    /// `IrStmt` sequence. Pattern is a kind-pattern that names the
    /// supplied event name; bindings are empty.
    fn rule_with_body(name: &str, event_name: &str, body: Vec<IrStmt>) -> PhysicsIR {
        PhysicsIR {
            name: name.to_string(),
            handlers: vec![PhysicsHandlerIR {
                pattern: IrPhysicsPattern::Kind(IrEventPattern {
                    name: event_name.to_string(),
                    event: None,
                    bindings: vec![],
                    span: span(0, 0),
                }),
                where_clause: None,
                body,
                span: span(0, 0),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(0, 0),
        }
    }

    /// Standard handler resolution for tests.
    fn standard_resolutions() -> Vec<HandlerResolution> {
        vec![HandlerResolution {
            event_kind: EventKindId(7),
            source_ring: EventRingId(2),
        }]
    }

    // ---- 1. Smallest happy path: payload-free Emit ----------------------

    #[test]
    fn lowers_single_handler_with_payload_free_emit() {
        // physics chronicle_death @phase(post) {
        //   on AgentDied { ... } { emit ChronicleEntry {} }
        // }
        let rule = rule_with_body(
            "chronicle_death",
            "AgentDied",
            vec![IrStmt::Emit(IrEmit {
                event_name: "ChronicleEntry".to_string(),
                event: None,
                fields: vec![],
                span: span(10, 30),
            })],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // Driver supplies the event-kind registry. `ChronicleEntry`
        // is the emit destination — registered via `event_kind_ids`,
        // not `variant_ids` (which is the sum-type-variant registry
        // for match arms). The two id spaces are distinct after the
        // Task 2.4 follow-up split.
        ctx.register_event_kind("ChronicleEntry", EventKindId(9));

        let ops = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::NonReplayable,
            &rule,
            &standard_resolutions(),
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
            ComputeOpKind::PhysicsRule {
                rule,
                on_event,
                replayable,
                ..
            } => {
                assert_eq!(*rule, PhysicsRuleId(0));
                assert_eq!(*on_event, EventKindId(7));
                assert_eq!(
                    *replayable,
                    ReplayabilityFlag::NonReplayable,
                    "rule lowered with NonReplayable flag"
                );
            }
            other => panic!("unexpected kind: {other:?}"),
        }
        // Rule name interned for pretty-printing.
        assert_eq!(
            prog.interner.get_physics_rule_name(PhysicsRuleId(0)),
            Some("chronicle_death")
        );
        // The Emit is lowered into the body.
        let stmt_count = prog.stmts.len();
        assert_eq!(stmt_count, 1);
        match &prog.stmts[0] {
            CgStmt::Emit { event, fields } => {
                assert_eq!(*event, EventKindId(9));
                assert!(fields.is_empty());
            }
            other => panic!("expected Emit, got {other:?}"),
        }
    }

    // ---- 2. Multi-statement body: If with two-Emit then-arm -------------

    #[test]
    fn lowers_multi_statement_body_with_if() {
        // physics x @phase(event) {
        //   on E { ... } { if true { emit A {}; emit B {} } }
        // }
        let rule = rule_with_body(
            "x",
            "E",
            vec![IrStmt::If {
                cond: lit_bool(true),
                then_body: vec![
                    IrStmt::Emit(IrEmit {
                        event_name: "A".to_string(),
                        event: None,
                        fields: vec![],
                        span: span(0, 0),
                    }),
                    IrStmt::Emit(IrEmit {
                        event_name: "B".to_string(),
                        event: None,
                        fields: vec![],
                        span: span(0, 0),
                    }),
                ],
                else_body: None,
                span: span(0, 30),
            }],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_event_kind("A", EventKindId(11));
        ctx.register_event_kind("B", EventKindId(12));

        let ops = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("lowers");
        assert_eq!(ops.len(), 1);

        let prog = builder.finish();
        // Three stmts: 2 emits + 1 if (the then-list bundles the emits).
        assert_eq!(prog.stmts.len(), 3);
        // Body list is the outermost (just the If).
        let op = &prog.ops[0];
        match &op.kind {
            ComputeOpKind::PhysicsRule { body, replayable, .. } => {
                assert_eq!(
                    *replayable,
                    ReplayabilityFlag::Replayable,
                    "rule lowered with Replayable flag"
                );
                let body_list = &prog.stmt_lists[body.0 as usize];
                assert_eq!(body_list.stmts.len(), 1);
                match &prog.stmts[body_list.stmts[0].0 as usize] {
                    CgStmt::If {
                        cond: _,
                        then,
                        else_,
                    } => {
                        assert!(else_.is_none(), "no else-arm in source");
                        let then_list = &prog.stmt_lists[then.0 as usize];
                        assert_eq!(then_list.stmts.len(), 2);
                    }
                    other => panic!("expected If, got {other:?}"),
                }
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    // ---- 3. Match lowering: two arms with one binder each ---------------

    #[test]
    fn lowers_match_with_struct_arms() {
        // physics x @phase(event) {
        //   on E { ... } {
        //     match scrutinee {
        //       Damage { amount } => { emit A {} }
        //       Heal { amount }   => { emit B {} }
        //     }
        //   }
        // }
        //
        // Bind locals:
        //   Damage.amount → LocalRef(1) → LocalId(101)
        //   Heal.amount   → LocalRef(2) → LocalId(102)
        let damage_local = LocalRef(1);
        let heal_local = LocalRef(2);
        let damage_arm = IrStmtMatchArm {
            pattern: IrPattern::Struct {
                name: "Damage".to_string(),
                ctor: None,
                bindings: vec![IrPatternBinding {
                    field: "amount".to_string(),
                    value: IrPattern::Bind {
                        name: "amount".to_string(),
                        local: damage_local,
                    },
                    span: span(0, 0),
                }],
            },
            body: vec![IrStmt::Emit(IrEmit {
                event_name: "A".to_string(),
                event: None,
                fields: vec![],
                span: span(0, 0),
            })],
            span: span(0, 0),
        };
        let heal_arm = IrStmtMatchArm {
            pattern: IrPattern::Struct {
                name: "Heal".to_string(),
                ctor: None,
                bindings: vec![IrPatternBinding {
                    field: "amount".to_string(),
                    value: IrPattern::Bind {
                        name: "amount".to_string(),
                        local: heal_local,
                    },
                    span: span(0, 0),
                }],
            },
            body: vec![IrStmt::Emit(IrEmit {
                event_name: "B".to_string(),
                event: None,
                fields: vec![],
                span: span(0, 0),
            })],
            span: span(0, 0),
        };

        let rule = rule_with_body(
            "x",
            "E",
            vec![IrStmt::Match {
                scrutinee: lit_f32(0.0),
                arms: vec![damage_arm, heal_arm],
                span: span(0, 100),
            }],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // Sum-type variants matched in arms — registered via the
        // variant-id space.
        ctx.register_variant("Damage", VariantId(0));
        ctx.register_variant("Heal", VariantId(1));
        // Emit destinations — registered via the event-kind space.
        // The split (Task 2.4 follow-up) keeps the two id spaces from
        // silently aliasing.
        ctx.register_event_kind("A", EventKindId(2));
        ctx.register_event_kind("B", EventKindId(3));
        ctx.register_local(damage_local, LocalId(101));
        ctx.register_local(heal_local, LocalId(102));

        let ops = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("lowers");
        assert_eq!(ops.len(), 1);

        let prog = builder.finish();
        // Find the Match stmt — it's the outermost in the body.
        let op = &prog.ops[0];
        let body_id = match &op.kind {
            ComputeOpKind::PhysicsRule { body, .. } => *body,
            other => panic!("unexpected kind: {other:?}"),
        };
        let body_list = &prog.stmt_lists[body_id.0 as usize];
        assert_eq!(body_list.stmts.len(), 1);
        let match_stmt = &prog.stmts[body_list.stmts[0].0 as usize];
        match match_stmt {
            CgStmt::Match { scrutinee: _, arms } => {
                assert_eq!(arms.len(), 2);
                // Arm 0 = Damage(amount = LocalId(101)) → body emits A
                assert_eq!(arms[0].variant, VariantId(0));
                assert_eq!(arms[0].bindings.len(), 1);
                assert_eq!(arms[0].bindings[0].field_name, "amount");
                assert_eq!(arms[0].bindings[0].local, LocalId(101));
                // Arm 1 = Heal(amount = LocalId(102)) → body emits B
                assert_eq!(arms[1].variant, VariantId(1));
                assert_eq!(arms[1].bindings[0].local, LocalId(102));
            }
            other => panic!("expected Match, got {other:?}"),
        }
    }

    // ---- 4. Replayability propagation: both flags ----------------------

    #[test]
    fn replayability_flag_propagates_to_op_kind() {
        let rule = rule_with_body(
            "r",
            "E",
            vec![IrStmt::Emit(IrEmit {
                event_name: "A".to_string(),
                event: None,
                fields: vec![],
                span: span(0, 0),
            })],
        );

        // Replayable variant.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_event_kind("A", EventKindId(0));
        let _ = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("lowers replayable");
        let prog = builder.finish();
        match &prog.ops[0].kind {
            ComputeOpKind::PhysicsRule { replayable, .. } => {
                assert_eq!(*replayable, ReplayabilityFlag::Replayable);
            }
            other => panic!("unexpected kind: {other:?}"),
        }

        // NonReplayable variant.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_event_kind("A", EventKindId(0));
        let _ = lower_physics(
            PhysicsRuleId(1),
            ReplayabilityFlag::NonReplayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("lowers non-replayable");
        let prog = builder.finish();
        match &prog.ops[0].kind {
            ComputeOpKind::PhysicsRule { replayable, .. } => {
                assert_eq!(*replayable, ReplayabilityFlag::NonReplayable);
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    // ---- 5. Negative: For loop in body → typed deferral -----------------

    #[test]
    fn rejects_for_loop_in_body() {
        // physics cast @phase(event) {
        //   on E { ... } { for op in ... { ... } }
        // }
        let rule = rule_with_body(
            "cast",
            "E",
            vec![IrStmt::For {
                binder: LocalRef(0),
                binder_name: "op".to_string(),
                iter: lit_f32(0.0),
                filter: None,
                body: vec![],
                span: span(5, 12),
            }],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect_err("For unsupported in physics body");
        match err {
            LoweringError::UnsupportedPhysicsStmt {
                rule,
                ast_label,
                span,
            } => {
                assert_eq!(rule, PhysicsRuleId(0));
                assert_eq!(ast_label, "For");
                assert_eq!(span.start, 5);
                assert_eq!(span.end, 12);
            }
            other => panic!("expected UnsupportedPhysicsStmt(For), got {other:?}"),
        }
    }

    // ---- 6. Negative: handler-resolution length mismatch ---------------

    #[test]
    fn rejects_handler_resolution_length_mismatch() {
        // Two handlers; only one resolution supplied.
        let rule = PhysicsIR {
            name: "r".to_string(),
            handlers: vec![
                PhysicsHandlerIR {
                    pattern: IrPhysicsPattern::Kind(IrEventPattern {
                        name: "A".to_string(),
                        event: None,
                        bindings: vec![],
                        span: span(0, 0),
                    }),
                    where_clause: None,
                    body: vec![],
                    span: span(0, 0),
                },
                PhysicsHandlerIR {
                    pattern: IrPhysicsPattern::Kind(IrEventPattern {
                        name: "B".to_string(),
                        event: None,
                        bindings: vec![],
                        span: span(0, 0),
                    }),
                    where_clause: None,
                    body: vec![],
                    span: span(0, 0),
                },
            ],
            annotations: vec![],
            cpu_only: false,
            span: span(0, 0),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            // Only one resolution for two handlers.
            &[HandlerResolution {
                event_kind: EventKindId(0),
                source_ring: EventRingId(0),
            }],
            &mut ctx,
        )
        .expect_err("length mismatch");
        match err {
            LoweringError::PhysicsHandlerResolutionLengthMismatch {
                rule,
                expected,
                got,
                ..
            } => {
                assert_eq!(rule, PhysicsRuleId(0));
                assert_eq!(expected, 2);
                assert_eq!(got, 1);
            }
            other => panic!("expected PhysicsHandlerResolutionLengthMismatch, got {other:?}"),
        }
        let prog = builder.finish();
        // Length check fires before name interning — no side effects.
        assert!(prog.ops.is_empty());
        assert_eq!(prog.interner.get_physics_rule_name(PhysicsRuleId(0)), None);
    }

    // ---- 7. Negative: unknown match variant ----------------------------

    #[test]
    fn rejects_unknown_match_variant() {
        // Match arm references a variant the registry doesn't know.
        let arm = IrStmtMatchArm {
            pattern: IrPattern::Struct {
                name: "UnknownVariant".to_string(),
                ctor: None,
                bindings: vec![],
            },
            body: vec![],
            span: span(20, 35),
        };
        let rule = rule_with_body(
            "r",
            "E",
            vec![IrStmt::Match {
                scrutinee: lit_f32(0.0),
                arms: vec![arm],
                span: span(0, 50),
            }],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // No registration of "UnknownVariant".
        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect_err("unknown variant");
        match err {
            LoweringError::UnknownMatchVariant {
                rule,
                variant_name,
                span,
            } => {
                assert_eq!(rule, PhysicsRuleId(0));
                assert_eq!(variant_name, "UnknownVariant");
                assert_eq!(span.start, 20);
                assert_eq!(span.end, 35);
            }
            other => panic!("expected UnknownMatchVariant, got {other:?}"),
        }
    }

    // ---- 8. Negative: unknown LocalRef in match binder -----------------

    #[test]
    fn rejects_unknown_local_ref_in_match_binder() {
        let arm = IrStmtMatchArm {
            pattern: IrPattern::Struct {
                name: "Damage".to_string(),
                ctor: None,
                bindings: vec![IrPatternBinding {
                    field: "amount".to_string(),
                    value: IrPattern::Bind {
                        name: "amount".to_string(),
                        local: LocalRef(99),
                    },
                    span: span(7, 13),
                }],
            },
            body: vec![],
            span: span(0, 0),
        };
        let rule = rule_with_body(
            "r",
            "E",
            vec![IrStmt::Match {
                scrutinee: lit_f32(0.0),
                arms: vec![arm],
                span: span(0, 50),
            }],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_variant("Damage", VariantId(0));
        // No local registered for LocalRef(99).
        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect_err("unknown local");
        match err {
            LoweringError::UnknownLocalRef {
                rule,
                binder_name,
                span,
            } => {
                assert_eq!(rule, PhysicsRuleId(0));
                assert_eq!(binder_name, "amount");
                assert_eq!(span.start, 7);
                assert_eq!(span.end, 13);
            }
            other => panic!("expected UnknownLocalRef, got {other:?}"),
        }
    }

    // ---- 9. Negative: empty match arms ---------------------------------

    #[test]
    fn rejects_empty_match_arms() {
        let rule = rule_with_body(
            "r",
            "E",
            vec![IrStmt::Match {
                scrutinee: lit_f32(0.0),
                arms: vec![],
                span: span(3, 11),
            }],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect_err("empty arms");
        match err {
            LoweringError::EmptyMatchArms { rule, span } => {
                assert_eq!(rule, PhysicsRuleId(0));
                assert_eq!(span.start, 3);
                assert_eq!(span.end, 11);
            }
            other => panic!("expected EmptyMatchArms, got {other:?}"),
        }
    }

    // ---- 10. Negative: SelfUpdate in physics body ----------------------

    #[test]
    fn rejects_self_update_in_physics_body() {
        // SelfUpdate is the view-fold-only shape; physics rejects it
        // up front.
        let rule = rule_with_body(
            "r",
            "E",
            vec![IrStmt::SelfUpdate {
                op: "+=".to_string(),
                value: lit_f32(1.0),
                span: span(2, 6),
            }],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect_err("SelfUpdate forbidden in physics");
        match err {
            LoweringError::UnsupportedPhysicsStmt {
                ast_label, span, ..
            } => {
                assert_eq!(ast_label, "SelfUpdate");
                assert_eq!(span.start, 2);
                assert_eq!(span.end, 6);
            }
            other => panic!("expected UnsupportedPhysicsStmt(SelfUpdate), got {other:?}"),
        }
    }

    // ---- 11. Snapshot: pinned `Display` form for a lowered op ----------

    #[test]
    fn snapshot_single_handler_physics_rule_op_display() {
        // Pins the wire format produced by `ComputeOp`'s Display impl
        // for a single-handler PhysicsRule.
        let rule = rule_with_body(
            "chronicle_death",
            "AgentDied",
            vec![IrStmt::Emit(IrEmit {
                event_name: "ChronicleEntry".to_string(),
                event: None,
                fields: vec![],
                span: span(0, 0),
            })],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_event_kind("ChronicleEntry", EventKindId(9));
        lower_physics(
            PhysicsRuleId(2),
            ReplayabilityFlag::NonReplayable,
            &rule,
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
            "op#0 kind=physics_rule(rule=#2, on_event=#5, replayable=non_replayable) shape=per_event(ring=#3) reads=[] writes=[]"
        );
    }

    // ---- 12. Display impl coverage for new error variants --------------

    #[test]
    fn lowering_error_display_unsupported_physics_stmt() {
        let e = LoweringError::UnsupportedPhysicsStmt {
            rule: PhysicsRuleId(3),
            ast_label: "For",
            span: span(7, 12),
        };
        let s = format!("{}", e);
        assert!(s.contains("physics#3"));
        assert!(s.contains("7..12"));
        assert!(s.contains("For"));
    }

    #[test]
    fn lowering_error_display_unknown_match_variant() {
        let e = LoweringError::UnknownMatchVariant {
            rule: PhysicsRuleId(4),
            variant_name: "Whatever".to_string(),
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("physics#4"));
        assert!(s.contains("Whatever"));
    }

    #[test]
    fn lowering_error_display_unknown_local_ref() {
        let e = LoweringError::UnknownLocalRef {
            rule: PhysicsRuleId(2),
            binder_name: "amount".to_string(),
            span: span(0, 6),
        };
        let s = format!("{}", e);
        assert!(s.contains("physics#2"));
        assert!(s.contains("amount"));
    }

    #[test]
    fn lowering_error_display_empty_match_arms() {
        let e = LoweringError::EmptyMatchArms {
            rule: PhysicsRuleId(1),
            span: span(0, 10),
        };
        let s = format!("{}", e);
        assert!(s.contains("physics#1"));
        assert!(s.contains("no arms"));
    }

    #[test]
    fn lowering_error_display_unsupported_match_pattern() {
        let e = LoweringError::UnsupportedMatchPattern {
            rule: PhysicsRuleId(0),
            pattern_label: "Wildcard",
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("physics#0"));
        assert!(s.contains("Wildcard"));
    }

    #[test]
    fn lowering_error_display_unsupported_match_binding_shape() {
        let e = LoweringError::UnsupportedMatchBindingShape {
            rule: PhysicsRuleId(0),
            field_name: "amount".to_string(),
            pattern_label: "Wildcard",
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("physics#0"));
        assert!(s.contains("amount"));
        assert!(s.contains("Wildcard"));
    }

    // ---- 13. Variant-id / event-kind-id registry split ------------------

    /// Regression test for the Task 2.4 follow-up split: emit-name
    /// resolution and match-variant resolution traverse independent
    /// registries on [`LoweringCtx`]. The natural per-sequence
    /// allocation pattern (the same id `0` used for both a sum-type
    /// variant and an event kind) used to be ambiguous because both
    /// resolutions read a single `variant_ids` map. The split makes
    /// each side route through its own typed map; the test asserts
    /// emits resolve to `EventKindId(0)` while match arms resolve to
    /// the variant's own `VariantId(0)` (driver-distinct, type-distinct).
    #[test]
    fn variant_id_and_event_kind_id_registries_are_independent() {
        // Body: match scrutinee { Foo {} => { emit AgentDied {} } }
        let arm = IrStmtMatchArm {
            pattern: IrPattern::Struct {
                name: "Foo".to_string(),
                ctor: None,
                bindings: vec![],
            },
            body: vec![IrStmt::Emit(IrEmit {
                event_name: "AgentDied".to_string(),
                event: None,
                fields: vec![],
                span: span(0, 0),
            })],
            span: span(0, 0),
        };
        let rule = rule_with_body(
            "r",
            "E",
            vec![IrStmt::Match {
                scrutinee: lit_f32(0.0),
                arms: vec![arm],
                span: span(0, 100),
            }],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // Driver populates BOTH spaces with the natural id-0
        // allocation. The split keeps them from aliasing.
        ctx.register_variant("Foo", VariantId(0));
        ctx.register_event_kind("AgentDied", EventKindId(0));

        lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::NonReplayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("lowers");

        let prog = builder.finish();
        // Find the emit and the match — both at id 0 in their
        // respective spaces. The emit's event must be `EventKindId(0)`
        // (resolved through `event_kind_ids`); the match arm's
        // variant must be `VariantId(0)` (resolved through
        // `variant_ids`). The two ids are typed-distinct even though
        // their numeric payloads collide — this is exactly the
        // ambiguity the split prevents.
        let mut found_emit = false;
        let mut found_match_arm = false;
        for stmt in &prog.stmts {
            match stmt {
                CgStmt::Emit { event, .. } => {
                    assert_eq!(*event, EventKindId(0));
                    found_emit = true;
                }
                CgStmt::Match { arms, .. } => {
                    assert_eq!(arms.len(), 1);
                    assert_eq!(arms[0].variant, VariantId(0));
                    found_match_arm = true;
                }
                CgStmt::Assign { .. } | CgStmt::If { .. } | CgStmt::Let { .. } => {
                    // Other body shapes — not produced by this fixture.
                }
            }
        }
        assert!(found_emit, "expected an Emit stmt in the lowered body");
        assert!(found_match_arm, "expected a Match stmt in the lowered body");
    }

    /// Without the split, an emit-name absent from the
    /// (now-distinct) event-kind registry surfaces as
    /// [`LoweringError::UnknownMatchVariant`] — even when the same
    /// name IS registered in the variant-id space. Confirms the two
    /// registries are read independently at lowering.
    #[test]
    fn emit_resolution_does_not_fall_through_to_variant_ids() {
        let rule = rule_with_body(
            "r",
            "E",
            vec![IrStmt::Emit(IrEmit {
                event_name: "Damage".to_string(),
                event: None,
                fields: vec![],
                span: span(8, 14),
            })],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // Damage IS registered as a sum-type variant, but NOT as an
        // event kind. Pre-split, the emit would have silently
        // resolved through the unified map; post-split, it must
        // surface as a typed error.
        ctx.register_variant("Damage", VariantId(7));
        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::NonReplayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect_err("emit lookup must not fall through to variant_ids");
        match err {
            LoweringError::UnknownMatchVariant {
                variant_name, span, ..
            } => {
                assert_eq!(variant_name, "Damage");
                assert_eq!(span.start, 8);
                assert_eq!(span.end, 14);
            }
            other => panic!("expected UnknownMatchVariant for emit, got {other:?}"),
        }
    }

    // ---- 14. Task 5.5b: Let binding lowering --------------------------

    /// Smallest happy path for Task 5.5b's `IrStmt::Let` arm. The
    /// body `let new_hp = 5.0` lowers to `CgStmt::Let { local,
    /// value, ty: F32 }` with a freshly-allocated [`LocalId`] and
    /// the value-expression's type.
    #[test]
    fn lowers_let_binding_to_cg_stmt_let() {
        let new_hp_local = LocalRef(7);
        let rule = rule_with_body(
            "damage",
            "EffectDamageApplied",
            vec![IrStmt::Let {
                name: "new_hp".to_string(),
                local: new_hp_local,
                value: lit_f32(5.0),
                span: span(2, 12),
            }],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ops = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("Let lowers cleanly");
        assert_eq!(ops.len(), 1);

        // The driver/test pre-registers no LocalId for the binder, so
        // physics-pass `Let` lowering allocated a fresh id and stored
        // the mapping in the lowering ctx.
        let allocated = *ctx
            .local_ids
            .get(&new_hp_local)
            .expect("Let registered the LocalRef → LocalId mapping");

        let prog = builder.finish();
        // Body shape: a single `CgStmt::Let { ty: F32 }` whose value
        // points at the F32 literal.
        match &prog.stmts[0] {
            CgStmt::Let { local, value: _, ty } => {
                assert_eq!(*local, allocated);
                assert_eq!(*ty, crate::cg::expr::CgTy::F32);
            }
            other => panic!("expected CgStmt::Let, got {other:?}"),
        }
    }

    /// Driver-pre-registered `LocalRef → LocalId` mappings are
    /// preserved by the Let arm — the arm reuses the existing id
    /// instead of allocating a new one.
    #[test]
    fn lowers_let_binding_reuses_pre_registered_local_id() {
        let new_hp_local = LocalRef(3);
        let rule = rule_with_body(
            "damage",
            "EffectDamageApplied",
            vec![IrStmt::Let {
                name: "new_hp".to_string(),
                local: new_hp_local,
                value: lit_f32(0.0),
                span: span(0, 5),
            }],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // Driver pre-allocates the LocalId.
        ctx.register_local(new_hp_local, LocalId(42));
        lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("Let lowers cleanly");
        let prog = builder.finish();
        match &prog.stmts[0] {
            CgStmt::Let { local, .. } => assert_eq!(*local, LocalId(42)),
            other => panic!("expected CgStmt::Let, got {other:?}"),
        }
    }

    /// A `Let` nested inside an `If`-then body lowers cleanly — the
    /// statement-list arm walks every shape, so deeper nesting is
    /// structurally fine.
    #[test]
    fn lowers_let_inside_if_body() {
        let inner_local = LocalRef(2);
        let rule = rule_with_body(
            "x",
            "E",
            vec![IrStmt::If {
                cond: lit_bool(true),
                then_body: vec![IrStmt::Let {
                    name: "x".to_string(),
                    local: inner_local,
                    value: lit_f32(1.0),
                    span: span(0, 5),
                }],
                else_body: None,
                span: span(0, 30),
            }],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("Let-in-If lowers cleanly");
        // Snapshot the local-id mapping before consuming the builder
        // (the borrow checker would otherwise see ctx as held).
        let local_was_registered = ctx.local_ids.contains_key(&inner_local);
        drop(ctx);
        let prog = builder.finish();
        // The body contains a Let stmt nested under the If's then-arm.
        let mut found_let = false;
        for stmt in &prog.stmts {
            if matches!(stmt, CgStmt::Let { .. }) {
                found_let = true;
            }
        }
        assert!(found_let, "expected a CgStmt::Let in the lowered program");
        // And the LocalRef → LocalId mapping was registered.
        assert!(local_was_registered, "Let arm registered the inner local");
    }

    // ---- 15. Task 5.5b: Fielded Emit lowering -------------------------

    /// Fielded emits resolve each `IrFieldInit { name, value }` to a
    /// typed `(EventField { event, index }, lowered_value_id)`
    /// using the per-event field schema on `LoweringCtx::event_field_indices`.
    #[test]
    fn lowers_fielded_emit_with_event_field_schema() {
        // Body: emit AgentDied { actor: 1.0, tick: 2.0 }
        let rule = rule_with_body(
            "damage",
            "EffectDamageApplied",
            vec![IrStmt::Emit(IrEmit {
                event_name: "AgentDied".to_string(),
                event: None,
                fields: vec![
                    IrFieldInit {
                        name: "actor".to_string(),
                        value: lit_f32(1.0),
                        span: span(0, 0),
                    },
                    IrFieldInit {
                        name: "tick".to_string(),
                        value: lit_f32(2.0),
                        span: span(0, 0),
                    },
                ],
                span: span(0, 30),
            })],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_event_kind("AgentDied", EventKindId(5));
        // Driver populates the field schema in declaration order.
        ctx.register_event_field(EventKindId(5), "actor", 0);
        ctx.register_event_field(EventKindId(5), "tick", 1);
        lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("fielded Emit lowers cleanly");
        let prog = builder.finish();
        // The body contains an Emit with two typed fields, in source
        // order, each pointing at the lowered value expression.
        let emit = prog
            .stmts
            .iter()
            .find_map(|s| match s {
                CgStmt::Emit { event, fields } => Some((*event, fields.clone())),
                _ => None,
            })
            .expect("expected a CgStmt::Emit");
        assert_eq!(emit.0, EventKindId(5));
        assert_eq!(emit.1.len(), 2);
        assert_eq!(
            emit.1[0].0,
            EventField {
                event: EventKindId(5),
                index: 0,
            }
        );
        assert_eq!(
            emit.1[1].0,
            EventField {
                event: EventKindId(5),
                index: 1,
            }
        );
    }

    /// Task 1 of the CG Lowering Gap Closure plan: a top-of-body
    /// `IrStmt::Let` followed by a fielded emit reading the bound
    /// local must lower without an `UnsupportedLocalBinding`
    /// diagnostic for the reader. Today the dispatcher routes
    /// `IrStmt::Let` through `lower_let`, which both pushes
    /// `CgStmt::Let` AND populates `ctx.local_ids` / `ctx.local_tys`
    /// for the binding; a downstream `IrExpr::Local(local_ref, name)`
    /// inside the emit's field-init then resolves through
    /// `lower_bare_local` to `CgExpr::ReadLocal { local, ty }`.
    ///
    /// Surface analogue:
    ///
    /// ```text
    /// physics record_damage @phase(event) {
    ///   on AgentAttacked { actor: a, target: tgt, damage: amt } {
    ///     let t = world.tick;          // top-of-body Let
    ///     emit DamageRecorded {        // reads `t` in field init
    ///       target: tgt,
    ///       stamp: t,
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// The fixture below builds the IR directly (matching the
    /// sibling `rule_with_body` style) so the test is hermetic
    /// against the resolver / parser surface.
    #[test]
    fn lower_physics_handles_top_of_body_let_binding() {
        let t_local = LocalRef(11);
        let rule = rule_with_body(
            "record_damage",
            "AgentAttacked",
            vec![
                IrStmt::Let {
                    name: "t".to_string(),
                    local: t_local,
                    value: lit_f32(7.0),
                    span: span(0, 5),
                },
                IrStmt::Emit(IrEmit {
                    event_name: "DamageRecorded".to_string(),
                    event: None,
                    fields: vec![IrFieldInit {
                        name: "stamp".to_string(),
                        // `t` reader: must lower via local_ids → ReadLocal.
                        value: node(IrExpr::Local(t_local, "t".to_string())),
                        span: span(20, 21),
                    }],
                    span: span(15, 35),
                }),
            ],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_event_kind("DamageRecorded", EventKindId(13));
        ctx.register_event_field(EventKindId(13), "stamp", 0);

        let result = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        );

        // Surface the diagnostic shape directly: the AIS hard-asserts
        // that no `UnsupportedLocalBinding { name: "t" }` fires.
        match &result {
            Ok(_) => {}
            Err(LoweringError::UnsupportedLocalBinding { name, .. }) if name == "t" => {
                panic!(
                    "let-bound `t` must lower without UnsupportedLocalBinding diagnostic; got {result:?}"
                );
            }
            Err(other) => {
                panic!("unexpected lowering error (not UnsupportedLocalBinding for `t`): {other:?}");
            }
        }
    }

    /// A fielded emit whose field name has no entry in
    /// `event_field_indices` surfaces as
    /// [`LoweringError::UnknownEventField`] with the resolved
    /// [`EventKindId`] and the source-level field name.
    #[test]
    fn rejects_fielded_emit_with_unknown_field() {
        let rule = rule_with_body(
            "damage",
            "EffectDamageApplied",
            vec![IrStmt::Emit(IrEmit {
                event_name: "AgentDied".to_string(),
                event: None,
                fields: vec![IrFieldInit {
                    name: "unregistered_field".to_string(),
                    value: lit_f32(0.0),
                    span: span(7, 25),
                }],
                span: span(0, 30),
            })],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_event_kind("AgentDied", EventKindId(5));
        // No `register_event_field` for `unregistered_field`.
        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect_err("missing field index must fail");
        match err {
            LoweringError::UnknownEventField {
                event,
                field_name,
                span,
            } => {
                assert_eq!(event, EventKindId(5));
                assert_eq!(field_name, "unregistered_field");
                assert_eq!(span.start, 7);
                assert_eq!(span.end, 25);
            }
            other => panic!("expected UnknownEventField, got {other:?}"),
        }
    }

    // ---- Task 1 (CG Lowering Gap Closure): event-pattern binding -------

    /// Helper — build a single-handler `PhysicsIR` whose pattern carries
    /// the supplied event-name + bindings, and whose body is the
    /// supplied stmt list.
    fn rule_with_pattern(
        name: &str,
        event_name: &str,
        bindings: Vec<IrPatternBinding>,
        body: Vec<IrStmt>,
    ) -> PhysicsIR {
        PhysicsIR {
            name: name.to_string(),
            handlers: vec![PhysicsHandlerIR {
                pattern: IrPhysicsPattern::Kind(IrEventPattern {
                    name: event_name.to_string(),
                    event: None,
                    bindings,
                    span: span(0, 0),
                }),
                where_clause: None,
                body,
                span: span(0, 0),
            }],
            annotations: vec![],
            cpu_only: false,
            span: span(0, 0),
        }
    }

    /// A bare local read inside the handler body resolves through the
    /// synthesized event-pattern Lets — the binder's name is registered
    /// via `register_local` in the synthesis pass and the read-side
    /// `lower_bare_local` walks `local_ids` cleanly.
    #[test]
    fn lower_physics_handles_event_pattern_binding() {
        use crate::cg::expr::{CgExpr, CgTy};
        use crate::cg::program::{EventLayout, FieldLayout};

        let target_local = LocalRef(20);

        // Pattern: `on EffectDamageApplied { target: t }` — one binder.
        let bindings = vec![IrPatternBinding {
            field: "target".to_string(),
            value: IrPattern::Bind {
                name: "t".to_string(),
                local: target_local,
            },
            span: span(0, 0),
        }];

        // Body: `emit ChronicleEntry { agent: t }` — reads the bound `t`.
        let body = vec![IrStmt::Emit(IrEmit {
            event_name: "ChronicleEntry".to_string(),
            event: None,
            fields: vec![IrFieldInit {
                name: "agent".to_string(),
                value: node(IrExpr::Local(target_local, "t".to_string())),
                span: span(20, 21),
            }],
            span: span(15, 35),
        })];

        let rule = rule_with_pattern("test", "EffectDamageApplied", bindings, body);
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);

        // Driver registers the dispatched event and the layout.
        ctx.register_event_kind("EffectDamageApplied", EventKindId(7));
        ctx.register_event_kind("ChronicleEntry", EventKindId(99));
        ctx.register_event_field(EventKindId(99), "agent", 0);

        let mut layout_fields = std::collections::BTreeMap::new();
        layout_fields.insert(
            "actor".to_string(),
            FieldLayout {
                word_offset_in_payload: 0,
                word_count: 1,
                ty: CgTy::AgentId,
            },
        );
        layout_fields.insert(
            "target".to_string(),
            FieldLayout {
                word_offset_in_payload: 1,
                word_count: 1,
                ty: CgTy::AgentId,
            },
        );
        ctx.register_event_layout(
            EventKindId(7),
            EventLayout {
                record_stride_u32: 10,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields: layout_fields,
            },
        );

        // The standard_resolutions helper maps to EventKindId(7) — the
        // Effect ring. Lower.
        lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect("event-pattern binding lowers cleanly");

        let prog = builder.finish();

        // Find the synthesized Let in the body.
        let let_stmt = prog
            .stmts
            .iter()
            .find_map(|s| match s {
                CgStmt::Let { local, value, ty } => Some((*local, *value, *ty)),
                _ => None,
            })
            .expect("expected a CgStmt::Let synthesized for the binder");

        // The Let's value-side is a CgExpr::EventField with the right
        // (event_kind, word_offset_in_payload, ty).
        let value_expr = &prog.exprs[let_stmt.1 .0 as usize];
        match value_expr {
            CgExpr::EventField {
                event_kind,
                word_offset_in_payload,
                ty,
            } => {
                assert_eq!(*event_kind, EventKindId(7));
                assert_eq!(*word_offset_in_payload, 1, "target is at word offset 1");
                assert_eq!(*ty, CgTy::AgentId);
            }
            other => panic!("expected CgExpr::EventField, got {other:?}"),
        }

        // The Let's claimed type matches the field's CgTy.
        assert_eq!(let_stmt.2, CgTy::AgentId);
    }

    /// A pattern binding whose layout entry is missing surfaces as
    /// `UnregisteredEventFieldLayout` rather than silently producing
    /// an `EventField` that fails at WGSL emit time.
    #[test]
    fn rejects_event_pattern_binding_with_missing_field_layout() {
        use crate::cg::program::EventLayout;
        use crate::cg::lower::error::PatternBindingSubject;

        let local = LocalRef(30);
        let bindings = vec![IrPatternBinding {
            field: "phantom_field".to_string(),
            value: IrPattern::Bind {
                name: "x".to_string(),
                local,
            },
            span: span(50, 60),
        }];
        let rule = rule_with_pattern("test", "EffectDamageApplied", bindings, vec![]);
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_event_kind("EffectDamageApplied", EventKindId(7));
        // Layout exists, but `phantom_field` is not in it.
        ctx.register_event_layout(
            EventKindId(7),
            EventLayout {
                record_stride_u32: 10,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields: std::collections::BTreeMap::new(),
            },
        );

        let err = lower_physics(
            PhysicsRuleId(0),
            ReplayabilityFlag::Replayable,
            &rule,
            &standard_resolutions(),
            &mut ctx,
        )
        .expect_err("missing field layout must fail");
        match err {
            LoweringError::UnregisteredEventFieldLayout {
                subject,
                event,
                field_name,
                span,
            } => {
                assert!(matches!(subject, PatternBindingSubject::Physics(PhysicsRuleId(0))));
                assert_eq!(event, EventKindId(7));
                assert_eq!(field_name, "phantom_field");
                assert_eq!(span.start, 50);
            }
            other => panic!("expected UnregisteredEventFieldLayout, got {other:?}"),
        }
    }
}
