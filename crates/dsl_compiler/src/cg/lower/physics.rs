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
//! Today's `lower_physics` recognises three body-statement forms:
//!
//! - `IrStmt::Emit { event, fields }` →
//!   [`crate::cg::stmt::CgStmt::Emit`] (after resolving each field
//!   name to the (event, index) pair via the driver-supplied event
//!   schema map carried on [`LoweringCtx`]).
//! - `IrStmt::If { cond, then_body, else_body }` →
//!   [`crate::cg::stmt::CgStmt::If`] (with the cond expression
//!   type-checked to `Bool`).
//! - `IrStmt::Match { scrutinee, arms }` →
//!   [`crate::cg::stmt::CgStmt::Match`] (with each arm's variant +
//!   binders resolved via the driver-supplied
//!   [`LoweringCtx::variant_ids`] / [`LoweringCtx::local_ids`]
//!   maps).
//!
//! Other AST statement shapes (`IrStmt::Let`, `IrStmt::For`,
//! `IrStmt::Expr`, `IrStmt::SelfUpdate`, `IrStmt::BeliefObserve`)
//! surface as typed [`LoweringError::UnsupportedPhysicsStmt`]
//! deferrals. The plan body assigns `For` to a future task; `Let`
//! requires local-binding resolution on the expression layer; bare
//! `Expr` (namespace setter calls like `agents.set_hp(t, x)`) needs
//! namespace lowering; `SelfUpdate` is forbidden in physics (the
//! resolver permits it only inside fold bodies); `BeliefObserve`
//! decomposes into a sequence of typed `Assign`s against the
//! BeliefState SoA surface — both a separate task.
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
use crate::cg::op::{ComputeOpKind, EventKindId, OpId, PhysicsRuleId};
use crate::cg::stmt::{CgMatchArm, CgStmt, CgStmtId, CgStmtList, MatchArmBinding};

use super::error::LoweringError;
use super::expr::{lower_expr, LoweringCtx};
use super::view::HandlerResolution;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-rule replayability flag. Propagated from the constitution P7
/// surface (the rule's `@phase(...)` annotation) into the lowered
/// op so emit can sort the rule's emissions into the right ring.
///
/// Encoded as a typed enum rather than a bare `bool` so call sites
/// read self-explanatory (`ReplayabilityFlag::Replayable` rather than
/// a positional `true`).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ReplayabilityFlag {
    /// `@phase(event)` — emissions land in the deterministic ring
    /// the runtime folds into the trace hash.
    Replayable,
    /// `@phase(post)` — emissions land in chronicle / telemetry
    /// rings the runtime fold ignores.
    NonReplayable,
}

impl ReplayabilityFlag {
    /// Convert to the `bool` payload carried on
    /// [`ComputeOpKind::PhysicsRule`].
    pub fn as_bool(self) -> bool {
        matches!(self, ReplayabilityFlag::Replayable)
    }
}

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

    // Lower the handler body.
    let stmt_ids = lower_stmt_list(rule_id, &handler.body, ctx)?;
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
        replayable: replayable.as_bool(),
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
        IrStmt::Let { span, .. } => Err(LoweringError::UnsupportedPhysicsStmt {
            rule: rule_id,
            ast_label: "Let",
            span: *span,
        }),
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
/// Today's lowering threads the `event` reference through
/// [`LoweringCtx::event_kind_ids`] (driver-supplied) — but Task 2.4
/// runs without a global event registry; tests that exercise emits
/// inject the [`EventKindId`] directly via the
/// [`HandlerResolution`] flowing into the enclosing op. Inside an
/// arm body, however, the enclosing-op's `on_event` is not the
/// emit's *destination* event — the emit may target a different
/// event variant (e.g., `physics damage` reads `EffectDamageApplied`
/// and emits `AgentAttacked` + `AgentDied`). For that reason the
/// per-emit `EventKindId` resolution is left to the driver's
/// future event registry; until then physics tests build emits
/// with `event: Some(EventRef(...))` and the lowering pulls the
/// id from a `ctx.event_kind_ids` map (added in a follow-up task).
///
/// Today we surface non-test, no-event emits as a typed deferral
/// — Task 2.4 ships the `Emit` lowering with `fields: vec![]` for
/// payload-free emits (the chronicle-class shape) and refuses
/// fielded emits until the driver wires the field-schema.
fn lower_emit(
    rule_id: PhysicsRuleId,
    emit: &IrEmit,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgStmtId, LoweringError> {
    // Today's lowering recognises the typed pre-resolved event ref
    // through [`LoweringCtx::variant_ids`] keyed on the event name —
    // the same map physics-Match uses for variant resolution. A
    // driver populates the registry once; both consumers share it.
    let event_kind = ctx
        .variant_ids
        .get(&emit.event_name)
        .copied()
        .map(|v| EventKindId(v.0))
        .ok_or_else(|| LoweringError::UnknownMatchVariant {
            rule: rule_id,
            variant_name: emit.event_name.clone(),
            span: emit.span,
        })?;

    // Emit fields require name → index resolution against the
    // event variant's declared field list; the driver's event
    // registry owns that mapping. Until it's wired, fielded emits
    // surface as deferrals — chronicle-class rules with no payload
    // (`emit Foo {}`) lower cleanly.
    if !emit.fields.is_empty() {
        return Err(LoweringError::UnsupportedPhysicsStmt {
            rule: rule_id,
            ast_label: "Emit-fielded",
            span: emit.span,
        });
    }

    // Construct the typed `CgStmt::Emit`. Today's tests use
    // payload-free emits; a future task will resolve each
    // `IrFieldInit { name, value }` to `(EventField { event,
    // index }, lowered_value_id)` once the driver supplies the
    // event-schema map.
    let stmt = CgStmt::Emit {
        event: event_kind,
        fields: Vec::new(),
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
        IrEmit, IrEventPattern, IrExpr, IrExprNode, IrPatternBinding, IrPhysicsPattern, IrStmt,
        IrStmtMatchArm, LocalRef, PhysicsHandlerIR, PhysicsIR,
    };

    use crate::cg::data_handle::EventRingId;
    use crate::cg::program::CgProgramBuilder;
    use crate::cg::stmt::{CgStmt, LocalId, VariantId};

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
        // Driver supplies the variant-id registry. `ChronicleEntry`
        // resolves to the EventKindId we use as scrutinee.
        ctx.register_variant("ChronicleEntry", VariantId(9));

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
                assert!(!*replayable, "rule lowered with NonReplayable flag");
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
        ctx.register_variant("A", VariantId(11));
        ctx.register_variant("B", VariantId(12));

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
                assert!(*replayable, "rule lowered with Replayable flag");
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
        ctx.register_variant("Damage", VariantId(0));
        ctx.register_variant("Heal", VariantId(1));
        ctx.register_variant("A", VariantId(2));
        ctx.register_variant("B", VariantId(3));
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
        ctx.register_variant("A", VariantId(0));
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
            ComputeOpKind::PhysicsRule { replayable, .. } => assert!(*replayable),
            other => panic!("unexpected kind: {other:?}"),
        }

        // NonReplayable variant.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_variant("A", VariantId(0));
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
            ComputeOpKind::PhysicsRule { replayable, .. } => assert!(!*replayable),
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
        ctx.register_variant("ChronicleEntry", VariantId(9));
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
            "op#0 kind=physics_rule(rule=#2, on_event=#5, replayable=false) shape=per_event(ring=#3) reads=[] writes=[]"
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
}
