//! Event-pattern binding synthesis — shared by physics and view-fold
//! handler lowering.
//!
//! Both physics rules (`physics <name> { on <Event> { ... } }`) and
//! view fold handlers (`view <name> @materialized(...) { on <Event>
//! { self += ... } }`) introduce per-binder locals on the handler's
//! `on <Event> { ... }` head:
//!
//! ```dsl
//! on EffectDamageApplied { actor: c, target: t, amount: a } { ... }
//! ```
//!
//! `c`, `t`, `a` are AST `LocalRef`s the resolver introduced; the body
//! reads them through `IrExpr::Local(local_ref, _)` which the
//! expression lowering routes through
//! [`super::expr::LoweringCtx::local_ids`] +
//! [`super::expr::LoweringCtx::local_tys`]. Without prelude `Let`s
//! those reads resolve to nothing and the body's
//! `lower_bare_local` surfaces `UnsupportedLocalBinding`.
//!
//! [`synthesize_pattern_binding_lets`] walks the binder list, allocates
//! a [`crate::cg::stmt::LocalId`] per binder, looks up
//! `(word_offset_in_payload, ty)` in the driver-supplied event-payload
//! layout
//! ([`super::expr::LoweringCtx::event_layouts`]), and synthesizes one
//! `CgStmt::Let { local, value: CgExpr::EventField{...}, ty }` per
//! binder. Returns the synthesised stmt ids in declaration order.
//!
//! The two callers (`super::physics::lower_one_handler` and
//! `super::view::lower_one_handler`) each construct the right
//! [`PatternBindingSubject`] (`Physics(rule_id)` vs `View(view_id)`)
//! and call this helper. The subject discriminant flows through into
//! every diagnostic so error messages render the right
//! `physics#N` / `view#N` prefix.
//!
//! # Why one helper
//!
//! Pre-consolidation, physics and view each carried a ~120-LOC clone
//! of this routine. The bodies were 95% identical; the only structural
//! difference was the subject id construction. A subsequent edit (e.g.,
//! lighting up nested-pattern binders, or threading a per-handler
//! local-binding override) would otherwise need to land twice with
//! mechanical care to keep the two paths in sync. Sharing the helper
//! removes that drift surface entirely.

use dsl_ast::ast::Span;
use dsl_ast::ir::{IrPattern, IrPatternBinding};

use crate::cg::expr::CgExpr;
use crate::cg::op::EventKindId;
use crate::cg::stmt::{CgStmt, CgStmtId};

use super::error::{LoweringError, PatternBindingSubject};
use super::expr::LoweringCtx;

/// Walk every event-pattern binder on the handler's `on <Event> {
/// ... }` head, allocate a fresh [`crate::cg::stmt::LocalId`] per
/// binder, resolve the binder's typed `(word_offset_in_payload, ty)`
/// against the driver-supplied event-payload layout
/// ([`LoweringCtx::event_layouts`]), and synthesize one
/// `CgStmt::Let { local, value: CgExpr::EventField{...}, ty }`
/// statement per binder. Returns the synthesised stmt ids in
/// declaration order.
///
/// The synthesised lets establish:
/// - `ctx.local_ids[binder.value.local_ref] = LocalId(N)` — so every
///   subsequent `IrExpr::Local(local_ref, _)` read inside the body
///   resolves through `lower_bare_local` to `CgExpr::ReadLocal`.
/// - `ctx.local_tys[LocalId(N)] = field_layout.ty` — so the read-side
///   resolution can reconstruct the typed `CgExpr::ReadLocal { local,
///   ty }` without re-walking the schema.
///
/// The Let value-side is `CgExpr::EventField { event_kind,
/// word_offset_in_payload, ty }`, which at WGSL emit time becomes a
/// schema-driven `event_ring[event_idx * RECORD_STRIDE + HEADER +
/// FIELD_OFFSET]` access — see `lower_cg_expr_to_wgsl`'s `EventField`
/// arm.
///
/// # Parameters
///
/// - `subject`: discriminates the calling lowering pass
///   ([`PatternBindingSubject::Physics`] vs
///   [`PatternBindingSubject::View`]). Flows into every produced
///   diagnostic so error messages name the right caller.
/// - `event_kind`: the dispatched event kind. Keys into
///   [`LoweringCtx::event_layouts`] for layout resolution.
/// - `bindings`: the AST binder list from the handler's pattern head.
///   Each entry's `field` resolves to a layout entry; each entry's
///   `value` must be the shorthand `IrPattern::Bind { name, local }`
///   shape (the only shape today's DSL produces — nested patterns
///   surface as [`LoweringError::UnsupportedEventPatternBinding`]).
///
/// # Errors
///
/// - [`LoweringError::UnregisteredEventKindLayout`] — the
///   `event_kind` has no entry in `ctx.event_layouts`.
/// - [`LoweringError::UnregisteredEventFieldLayout`] — the binder's
///   field name is not declared in the event's payload layout.
/// - [`LoweringError::UnsupportedEventPatternBinding`] — a binder's
///   nested pattern shape is not the shorthand `IrPattern::Bind`
///   form (today's only supported shape).
/// - [`LoweringError::BuilderRejected`] — defensive surface for
///   builder invariant failures on `add_expr` / `add_stmt`. Should
///   not fire under normal operation.
pub(crate) fn synthesize_pattern_binding_lets(
    subject: PatternBindingSubject,
    event_kind: EventKindId,
    bindings: &[IrPatternBinding],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<CgStmtId>, LoweringError> {
    if bindings.is_empty() {
        return Ok(Vec::new());
    }

    // Snapshot the layout reference up front. Holding a borrow of
    // `ctx.event_layouts` while mutating other ctx fields would
    // double-borrow; clone the field map (small, ≤8 fields per
    // event today) into a local owned table first.
    let layout_fields = match ctx.event_layouts.get(&event_kind) {
        Some(l) => l.fields.clone(),
        None => {
            // No layout registered for this kind. Surface a typed
            // diagnostic instead of constructing an `EventField` whose
            // schema lookup would fail at WGSL emit time.
            return Err(LoweringError::UnregisteredEventKindLayout {
                subject,
                event: event_kind,
                span: bindings.first().map(|b| b.span).unwrap_or(Span::dummy()),
            });
        }
    };

    let mut stmt_ids = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let binder_local = match &binding.value {
            IrPattern::Bind { name: _, local } => *local,
            IrPattern::Struct { .. } => {
                return Err(LoweringError::UnsupportedEventPatternBinding {
                    subject,
                    field_name: binding.field.clone(),
                    pattern_label: "Struct",
                    span: binding.span,
                });
            }
            IrPattern::Ctor { .. } => {
                return Err(LoweringError::UnsupportedEventPatternBinding {
                    subject,
                    field_name: binding.field.clone(),
                    pattern_label: "Ctor",
                    span: binding.span,
                });
            }
            IrPattern::Expr(_) => {
                return Err(LoweringError::UnsupportedEventPatternBinding {
                    subject,
                    field_name: binding.field.clone(),
                    pattern_label: "Expr",
                    span: binding.span,
                });
            }
            IrPattern::Wildcard => {
                // Wildcards bind nothing — skip without synthesising
                // a Let. The body cannot reference the field through a
                // wildcard.
                continue;
            }
        };

        // Resolve the field's layout entry.
        let field_layout = layout_fields.get(&binding.field).copied().ok_or_else(|| {
            LoweringError::UnregisteredEventFieldLayout {
                subject,
                event: event_kind,
                field_name: binding.field.clone(),
                span: binding.span,
            }
        })?;

        // Allocate a fresh LocalId for the binder. Mirrors `lower_let`'s
        // allocation strategy: pick an id past every existing one so
        // subsequent allocations never collide.
        let local_id = match ctx.local_ids.get(&binder_local).copied() {
            Some(id) => id,
            None => ctx.allocate_local(binder_local),
        };

        // Build the typed value-side expression. The well-formed pass
        // re-checks the schema (`(event_kind, word_offset_in_payload)`
        // exists + claimed type matches the layout's type) defensively
        // at program level, so a synthetic IR with a mismatched ty
        // still gets caught.
        let value_expr = CgExpr::EventField {
            event_kind,
            word_offset_in_payload: field_layout.word_offset_in_payload,
            ty: field_layout.ty,
        };
        let value_id = ctx
            .builder
            .add_expr(value_expr)
            .map_err(|e| LoweringError::BuilderRejected {
                error: e,
                span: binding.span,
            })?;
        // Type-check is a no-op for `EventField` (the type is
        // self-pinned), but we run it through the standard helper for
        // symmetry with other expression pushes.
        super::expr::typecheck_node(ctx, value_id, binding.span)?;

        // Record the binder's CG type so bare-local reads
        // (`IrExpr::Local(local_ref, _)`) inside the body resolve via
        // `ctx.local_tys` to `CgExpr::ReadLocal { local, ty }`.
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
