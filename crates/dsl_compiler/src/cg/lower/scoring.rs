//! Scoring lowering — `ScoringIR → ComputeOpKind::ScoringArgmax`.
//!
//! Each `scoring { … }` decl in the resolved DSL IR produces ONE
//! [`ComputeOpKind::ScoringArgmax`] op (one per scoring, typically one
//! per agent kind since scoring is per-action argmax), dispatched
//! [`crate::cg::dispatch::DispatchShape::PerAgent`] over the agent
//! population. The op carries every row from both
//! [`dsl_ast::ir::ScoringIR::entries`] (standard rows) and
//! [`dsl_ast::ir::ScoringIR::per_ability_rows`].
//!
//! The pass:
//!
//! 1. Resolves the [`crate::cg::op::ScoringId`] (allocator-driven —
//!    the driver picks; tests pass directly).
//! 2. For each standard `ScoringEntryIR { head, expr, … }`: resolves
//!    the head name through [`super::expr::LoweringCtx::action_ids`]
//!    to a typed [`ActionId`], lowers `expr` via
//!    [`super::expr::lower_expr`], type-checks the result is `F32`,
//!    and builds a [`ScoringRowOp`] with `target = None, guard =
//!    None`.
//! 3. For each `PerAbilityRowIR { name, guard, score, target, … }`:
//!    resolves the row name through `action_ids`, lowers `score`
//!    (type-checked F32), lowers `target` if present (type-checked
//!    `AgentId`), lowers `guard` if present (type-checked `Bool`),
//!    and builds a [`ScoringRowOp`] with the populated optional
//!    fields.
//! 4. Builds the [`ComputeOpKind::ScoringArgmax`] kind, picks
//!    [`DispatchShape::PerAgent`], and pushes the op via
//!    [`crate::cg::program::CgProgramBuilder::add_op`].
//! 5. Interns the scoring's source-level name on the builder.
//!
//! # Field-shape rationale
//!
//! [`crate::cg::op::ScoringRowOp::target`] and
//! [`crate::cg::op::ScoringRowOp::guard`] are `Option<CgExprId>`.
//! Standard rows lower with both `None`: the runtime resolves the
//! target implicitly from the action kind (e.g., `Hold = 0.1` has no
//! per-row target). Per-ability rows populate one or both fields
//! directly from the AST. The IR amendment that introduced these
//! options is the surgical Phase 1 change carried by Task 2.5 — same
//! precedent as Task 2.4's [`crate::cg::stmt::CgStmt::Match`] +
//! `replayable: bool` carry.
//!
//! # Action-id resolution
//!
//! Both standard heads (the `Head =` form's `head.name`) and
//! per-ability row names lower through
//! [`super::expr::LoweringCtx::action_ids`]. The driver populates the
//! map per-scoring-decl with one allocation per distinct head /
//! per-ability row name. An unregistered name surfaces as
//! [`LoweringError::UnknownScoringAction`] — a dedicated typed
//! variant that mirrors the physics pass's
//! [`LoweringError::UnknownMatchVariant`] precedent. Tests populate
//! the map directly via
//! [`super::expr::LoweringCtx::register_action`].
//!
//! # Action head-shape gate
//!
//! Standard scoring rows (`Head = <expr>`) carry an
//! [`dsl_ast::ir::IrActionHead`] whose `shape` field is one of
//! [`dsl_ast::ir::IrActionHeadShape::None`] (bare name —
//! `Hold = 0.1`),
//! [`dsl_ast::ir::IrActionHeadShape::Positional`] (e.g.,
//! `Attack(target) = …`), or
//! [`dsl_ast::ir::IrActionHeadShape::Named`] (e.g.,
//! `Cast(ability: AbilityId) = …`).
//!
//! Phase 6 Task 2 lights up the canonical positional shape:
//! single-binder `(target: AgentId)` heads bind `target` to the
//! per-pair candidate context for the row body's lowering. Bare
//! `target` resolves to [`crate::cg::expr::CgExpr::PerPairCandidateId`]
//! via `lower_bare_local`'s `target_local` arm; `target.<field>`
//! resolves to `Read(AgentField { target: AgentRef::PerPairCandidate,
//! .. })` via `lower_field`'s same-named arm. The flag is
//! save/restored around the body lowering so adjacent rows don't
//! observe a leaked binding (mirrors `lower_mask`'s protocol).
//!
//! Other parametric shapes still reject with
//! [`LoweringError::UnsupportedScoringHeadShape`]: multi-binder
//! positional heads, non-`target` positional names, non-`AgentId`
//! positional types, and any `Named` head. The closed-set
//! `head_label` payload (`"positional"` | `"named"`) keeps the
//! diagnostic allocation-free.
//!
//! # Typed-error surface
//!
//! All defects surface as variants on the unified
//! [`super::error::LoweringError`]. Scoring-specific variants carry
//! the `Scoring*` prefix (`ScoringUtilityNotF32`,
//! `ScoringTargetNotAgentId`, `ScoringGuardNotBool`,
//! `ScoringRowTypeCheckFailure`) or the `*Scoring*` infix
//! (`UnknownScoringAction`, `UnsupportedScoringHeadShape`) per the
//! convention documented on `error.rs`. Expression-body failures
//! returned by [`super::expr::lower_expr`] propagate unchanged via
//! `?` — there is no wrapper variant.
//!
//! # Limitations
//!
//! - **`IrActionHead` shapes.** The canonical single-binder
//!   `(target: AgentId)` positional shape lowers as of Phase 6
//!   Task 2 (see the "Action head-shape gate" section). Other
//!   shapes — `Named`, multi-binder positional, non-`target`
//!   positional names, non-`AgentId` types — still surface as
//!   [`LoweringError::UnsupportedScoringHeadShape`]. Per-ability
//!   rows that reference `target.<field>` inside their `score` /
//!   `guard` expressions will surface as
//!   [`LoweringError::UnsupportedLocalBinding`] from the underlying
//!   `lower_expr` until the driver task wires per-row local binding.
//! - **Real-fixture coverage.** Hand-written scoring decls in
//!   `assets/sim/scoring.sim` exercise expression constructs (binary
//!   `+`, `if-else` modifiers, `view::*` calls, namespace-field
//!   reads on `target`) some of which Task 2.1's expression lowering
//!   recognises and others (e.g., aliased `target` field access,
//!   per-unit modifiers) it doesn't. Lighting up real-fixture
//!   coverage is the driver task's (2.8) responsibility — same
//!   scope discipline as Tasks 2.2 / 2.3 / 2.4. Today's tests
//!   exercise synthetic IRs (literal-only utilities, simple per-
//!   ability rows with literal target / guard) — sufficient for the
//!   Phase 1 amendment to the [`ScoringRowOp`] shape.

use dsl_ast::ast::Span;
use dsl_ast::ir::{
    IrActionHeadShape, IrExpr, IrExprNode, IrType, LocalRef, PerAbilityRowIR, ScoringEntryIR,
    ScoringIR,
};

use crate::cg::data_handle::CgExprId;
use crate::cg::dispatch::DispatchShape;
use crate::cg::expr::{type_check, BinaryOp, CgExpr, CgTy, TypeCheckCtx, TypeError};
use crate::cg::op::{ActionId, ComputeOpKind, OpId, ScoringId, ScoringRowOp};
use crate::cg::stmt::LocalId;

use super::error::{LoweringError, ScoringRowSubject};
use super::expr::{lower_expr, LoweringCtx};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower a single [`ScoringIR`] to one [`ComputeOpKind::ScoringArgmax`]
/// op pushed onto `ctx.builder`.
///
/// # Parameters
///
/// - `scoring_id`: the [`ScoringId`] this scoring binds to. Allocated
///   by the driver (Task 2.7 / 2.8); tests construct ids directly.
/// - `ir`: the resolved AST scoring decl. Its `entries` (standard
///   rows) and `per_ability_rows` are lowered in source order; the
///   resulting `ScoringRowOp`s are concatenated into the op's
///   `rows` vec (entries first, per-ability rows after — preserves
///   source-order determinism).
/// - `ctx`: the lowering context (carries the in-flight builder,
///   action / view / variant resolution maps, and diagnostic
///   accumulator).
///
/// # Returns
///
/// The freshly-allocated [`OpId`] of the pushed scoring op. Per the
/// plan: one op per scoring decl.
///
/// # Errors
///
/// See [`LoweringError`] for the closed defect set. Expression-body
/// failures from [`super::expr::lower_expr`] propagate unchanged via
/// `?`. Scoring-specific concerns surface as `Scoring*` variants:
///
/// - [`LoweringError::ScoringUtilityNotF32`] — utility expression
///   resolved to a non-F32 type.
/// - [`LoweringError::ScoringTargetNotAgentId`] — per-ability row
///   target resolved to a non-AgentId type.
/// - [`LoweringError::ScoringGuardNotBool`] — per-ability row guard
///   resolved to a non-Bool type.
/// - [`LoweringError::ScoringRowTypeCheckFailure`] — a constructed
///   sub-expression node failed re-typecheck (defensive — should
///   not normally fire because [`super::expr::lower_expr`]
///   type-checks every node it constructs).
///
/// # Side effects
///
/// On success: zero or more expression sub-trees and one op pushed
/// onto the builder; the scoring's name is interned. On failure: any
/// partial expression sub-trees pushed inside `lower_expr` are left
/// as orphans (see `lower_expr`'s "Orphan behavior" note); no op
/// past the failure point is added.
///
/// # Limitations
///
/// See the module docstring's "Limitations" section. Real DSL
/// scoring decls (`assets/sim/scoring.sim`) routinely use
/// expression constructs that the Task 2.1 expression lowering does
/// not yet support (`target.<field>` reads, `view::*` calls with
/// non-`self` arguments, per-unit modifiers); those decls won't
/// lower cleanly until the driver task lights up the missing
/// expression-layer plumbing. Today's tests exercise synthetic IRs
/// — same scope discipline as Tasks 2.2 / 2.3 / 2.4.
pub fn lower_scoring(
    scoring_id: ScoringId,
    ir: &ScoringIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<OpId, LoweringError> {
    intern_scoring_name(scoring_id, ir, ctx)?;

    let mut rows = Vec::with_capacity(ir.entries.len() + ir.per_ability_rows.len());

    for entry in &ir.entries {
        rows.push(lower_standard_row(scoring_id, entry, ctx)?);
    }
    for row in &ir.per_ability_rows {
        rows.push(lower_per_ability_row(scoring_id, row, ctx)?);
    }

    let kind = ComputeOpKind::ScoringArgmax {
        scoring: scoring_id,
        rows,
    };
    let op_id = ctx
        .builder
        .add_op(kind, DispatchShape::PerAgent, ir.span)
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: ir.span,
        })?;

    Ok(op_id)
}

/// Intern the scoring decl's source-level name on the builder.
/// `ScoringIR` does not carry a `name` field of its own — there is at
/// most one scoring decl per source file (the engine's argmax is over
/// every action), so the canonical name is derived from the
/// allocator's id assignment. We intern a stable label so
/// pretty-printing and diagnostics can name the op.
///
/// The name is `"scoring"` for the single-decl case; multi-decl
/// programs (if any) get `"scoring{n}"` appended by the driver.
fn intern_scoring_name(
    scoring_id: ScoringId,
    ir: &ScoringIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<(), LoweringError> {
    // ScoringIR has no name in the AST — synthesize a stable label
    // from the id. The driver task may override this with a
    // source-derived name once a richer scoring registry exists.
    let name = format!("scoring{}", scoring_id.0);
    ctx.builder
        .intern_scoring_name(scoring_id, name)
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: ir.span,
        })
}

// ---------------------------------------------------------------------------
// Per-row lowering
// ---------------------------------------------------------------------------

/// Lower a standard scoring row (`Head = expr`) to a
/// [`ScoringRowOp`] with `target = None, guard = None`.
///
/// # Positional head binders (Phase 6 Task 2)
///
/// Rows like `Attack(target) = ...` carry an
/// [`IrActionHeadShape::Positional`] head whose binder names a
/// per-pair candidate (today: `target: AgentId`). The body's reads of
/// `target` / `target.<field>` resolve against the per-pair candidate
/// context — same shape as pair-bound mask predicates (see
/// [`super::mask::lower_mask`]'s `target_local` flag). The lowering
/// flips [`super::expr::LoweringCtx::target_local`] for the duration
/// of `lower_expr` so:
///
/// - Bare `target` resolves through `lower_bare_local`'s step 3
///   (`"target" if ctx.target_local`) to [`crate::cg::expr::CgExpr::PerPairCandidateId`].
/// - `target.<field>` resolves through `lower_field`'s
///   `Local(_, "target") if ctx.target_local` arm to
///   `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`.
///
/// The flag is restored after the row body lowers so downstream rows
/// (and any nested per-ability rows) don't observe a leaked binding.
///
/// # Rejected positional shapes
///
/// - **Non-`target` binder names** (e.g., a future `Attack(victim)`)
///   surface as [`LoweringError::UnsupportedScoringHeadShape`] with
///   `head_label = "positional"` — today's CG-IR surface only
///   recognises the canonical `target` name (matching the legacy
///   emit-side convention in `crate::emit_scoring::lower_entry`).
/// - **Multi-binder positional heads** (`Attack(t, u)`) likewise
///   reject — only the single-binder `target: AgentId` shape is
///   wired today.
/// - **Non-`AgentId` binder types** (e.g., `Cast(ability: AbilityId)`
///   if a future row used it as a scoring head) reject too — the
///   `PerPairCandidate` semantic only covers AgentId-typed candidate
///   sides.
/// - **`Named` heads** (`Cast(ability: AbilityId)`-style) remain
///   rejected as before — there is no per-pair candidate semantic for
///   non-positional binders.
fn lower_standard_row(
    scoring_id: ScoringId,
    entry: &ScoringEntryIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<ScoringRowOp, LoweringError> {
    // Positional head binders bind a per-pair candidate (today:
    // `target: AgentId`). The body lowering flips `target_local` so
    // `target.<field>` reads resolve against the candidate side.
    // Other parametric shapes (`Named`, multi-binder positional,
    // non-AgentId positional, non-`target` positional name) reject
    // before action-id resolution — `action: None` signals the gate
    // fired before resolution succeeded.
    let positional_target_binder = match &entry.head.shape {
        IrActionHeadShape::None => None,
        IrActionHeadShape::Positional(binders) => {
            // Today's CG-IR routing only recognises a single
            // `target: AgentId` positional binder. Anything else
            // surfaces the same typed gate as before.
            let recognised = binders.len() == 1
                && binders[0].0 == "target"
                && binders[0].2 == IrType::AgentId;
            if !recognised {
                return Err(LoweringError::UnsupportedScoringHeadShape {
                    scoring: scoring_id,
                    action: None,
                    head_label: "positional",
                    span: entry.head.span,
                });
            }
            // Carry the binder's `LocalRef` through so we can
            // shadow it in `local_ids` for the body lowering. See
            // the save/restore protocol below.
            Some(binders[0].1)
        }
        IrActionHeadShape::Named(_) => {
            return Err(LoweringError::UnsupportedScoringHeadShape {
                scoring: scoring_id,
                action: None,
                head_label: "named",
                span: entry.head.span,
            });
        }
    };

    let action = resolve_action_id(scoring_id, &entry.head.name, entry.head.span, ctx)?;

    // Bind `target` to the per-pair candidate context for the
    // duration of the body lowering. Restore the flag after so
    // downstream rows don't observe a leaked binding (mirrors
    // `super::mask::lower_mask`'s save/restore protocol).
    //
    // ALSO shadow the binder's `LocalRef` in `ctx.local_ids` for the
    // body lowering. Without this, an upstream pass (today the verb
    // expander's chronicle-physics lowering — `cg::lower::verb_expand::
    // synthesize_cascade_physics`) may have registered the verb's
    // `target` LocalRef → some LocalId for the chronicle's
    // event-pattern binder. `lower_bare_local`'s Step 1 (let-bound
    // local) wins over Step 2-3 (structural `target`/`candidate`
    // names), so a bare `target` reference in the score body would
    // resolve to `ReadLocal { local: <chronicle-local> }` instead of
    // `PerPairCandidateId`. The downstream per-pair detector
    // (`expr_references_per_pair_candidate` in `cg::emit::kernel`)
    // then sees no candidate-side reference, the scoring kernel
    // skips the inner candidate loop, and the WGSL body emits an
    // unresolved `target_expr_<N>` identifier — the gap pinned by
    // the pair_scoring_probe (2026-05-04). Shadowing here keeps the
    // verb-injected scoring entry's body lowering hermetic.
    let prev_target_local = ctx.target_local;
    let prev_local_id = positional_target_binder
        .and_then(|binder_local| ctx.local_ids.remove(&binder_local).map(|id| (binder_local, id)));
    if positional_target_binder.is_some() {
        ctx.target_local = true;
    }
    // Shadow any `self` binders the score expression references. The
    // verb expander's cascade-physics lowering registers the verb's
    // `self` LocalRef → some LocalId in `ctx.local_ids` so the verb
    // body's `self.<field>` reads resolve to that event-payload actor.
    // In scoring context, the same `self` LocalRef appearing in the
    // score expression must instead resolve to `AgentSelfId` (the
    // per-agent dispatch row). Without shadowing, `lower_bare_local`'s
    // Step 1 (let-bound local) wins over Step 2 (`name == "self"`
    // structural arm) and the score body emits a `local_<N>` reference
    // pointing at a let-binding that exists in the cascade-physics
    // kernel, NOT the scoring kernel — producing an undefined
    // identifier in the scoring WGSL output.
    //
    // The walker below collects every `IrExpr::Local(_, "self")` in
    // the score expression and removes its `local_ids` entry for the
    // duration of the body lowering. Restored after so adjacent rows
    // (and downstream lowering passes) see the prior bindings intact.
    // Mirrors the `target` shadowing protocol above.
    let prev_self_ids = shadow_self_binders(&entry.expr, ctx);
    let utility_result = lower_expr(&entry.expr, ctx);
    restore_self_binders(prev_self_ids, ctx);
    ctx.target_local = prev_target_local;
    if let Some((binder_local, prior_id)) = prev_local_id {
        ctx.local_ids.insert(binder_local, prior_id);
    }
    let utility = utility_result?;

    check_utility_f32(scoring_id, action, utility, entry.expr.span, ctx)?;
    Ok(ScoringRowOp {
        action,
        utility,
        target: None,
        guard: None,
    })
}

/// Lower a `row <name> per_ability { guard, score, target }` row.
///
/// Per-ability rows do NOT today carry a `Positional(target)` binder
/// in their head — the per-ability candidate side is implicit (the
/// scoring kernel iterates the agent's own ability slots). But the
/// row's `score` / `target` / `guard` expressions can still reference
/// `self` (the caster), and the verb expander's cascade-physics
/// lowering may have registered the verb's `self` LocalRef in
/// `ctx.local_ids` ahead of the scoring pass. Each sub-expression's
/// lowering is therefore wrapped in the [`shadow_self_binders`] /
/// [`restore_self_binders`] pair so a bare `self` reference in the
/// per-ability row body resolves to `AgentSelfId` rather than the
/// stale event-payload local. Same rationale as
/// [`lower_standard_row`]'s self-shadow protocol.
fn lower_per_ability_row(
    scoring_id: ScoringId,
    row: &PerAbilityRowIR,
    ctx: &mut LoweringCtx<'_>,
) -> Result<ScoringRowOp, LoweringError> {
    let action = resolve_action_id(scoring_id, &row.name, row.span, ctx)?;

    // Score (utility base) — required, must be F32.
    let prev_self = shadow_self_binders(&row.score, ctx);
    let score_result = lower_expr(&row.score, ctx);
    restore_self_binders(prev_self, ctx);
    let base_utility = score_result?;
    check_utility_f32(scoring_id, action, base_utility, row.score.span, ctx)?;

    // Weights (utility addend) — optional, must be F32. When present
    // the row's effective utility is `base + weights`. Closes Gap C
    // from `docs/architecture/gaps_observed.md` (2026-05-11): the
    // utility-table form of scoring rows used by predator_prey,
    // crowd_navigation, squad_skirmish (`row X per_target { base:
    // <const>, weights: <expr> }`) was parse-and-discarded pre-fix,
    // so personality-weighted scoring contributed nothing to argmax.
    // The composed Add stays a CgExpr::Binary AddF32 node so the
    // existing well_formed type-check (`type_check_row`) walks it
    // unchanged.
    let utility = match &row.weights {
        Some(weights_node) => {
            let prev_self = shadow_self_binders(weights_node, ctx);
            let weights_result = lower_expr(weights_node, ctx);
            restore_self_binders(prev_self, ctx);
            let weights_id = weights_result?;
            // Re-use the utility-F32 gate so a non-F32 weights surfaces
            // as the same `ScoringUtilityNotF32` variant authors already
            // know from a non-F32 `score:`. The span pins the
            // `weights:` sub-expression so the diagnostic still
            // localises to the offending clause.
            check_utility_f32(scoring_id, action, weights_id, weights_node.span, ctx)?;
            compose_base_plus_weights(
                scoring_id,
                action,
                base_utility,
                weights_id,
                row.span,
                ctx,
            )?
        }
        None => base_utility,
    };

    // Target — optional, must be AgentId when Some.
    let target = match &row.target {
        Some(target_node) => {
            let prev_self = shadow_self_binders(target_node, ctx);
            let target_result = lower_target(scoring_id, action, target_node, ctx);
            restore_self_binders(prev_self, ctx);
            Some(target_result?)
        }
        None => None,
    };

    // Guard — optional, must be Bool when Some.
    let guard = match &row.guard {
        Some(guard_node) => {
            let prev_self = shadow_self_binders(guard_node, ctx);
            let guard_result = lower_guard(scoring_id, action, guard_node, ctx);
            restore_self_binders(prev_self, ctx);
            Some(guard_result?)
        }
        None => None,
    };

    Ok(ScoringRowOp {
        action,
        utility,
        target,
        guard,
    })
}

/// Compose `base + weights` as a single `CgExpr::Binary { AddF32 }`
/// node and re-confirm the result type-checks to F32. Both operands
/// have already been F32-checked individually; the wrapper guards
/// against a constructed node that fails re-typecheck (defensive —
/// should not normally fire, mirrors the contract on
/// `check_utility_f32`).
fn compose_base_plus_weights(
    scoring_id: ScoringId,
    action: ActionId,
    base: CgExprId,
    weights: CgExprId,
    span: Span,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let composed = ctx
        .builder
        .add_expr(CgExpr::Binary {
            op: BinaryOp::AddF32,
            lhs: base,
            rhs: weights,
            ty: CgTy::F32,
        })
        .map_err(|e| LoweringError::BuilderRejected { error: e, span })?;
    check_utility_f32(scoring_id, action, composed, span, ctx)?;
    Ok(composed)
}

// ---------------------------------------------------------------------------
// Per-step helpers
// ---------------------------------------------------------------------------

/// Resolve a scoring row's source-level name to a typed
/// [`ActionId`]. The driver populates
/// [`super::expr::LoweringCtx::action_ids`] with one entry per
/// distinct head / per-ability-row name across the scoring decl;
/// tests register names directly via
/// [`super::expr::LoweringCtx::register_action`]. An unregistered
/// name surfaces as [`LoweringError::UnknownScoringAction`] — a
/// dedicated typed variant that mirrors the physics pass's
/// [`LoweringError::UnknownMatchVariant`] precedent.
fn resolve_action_id(
    scoring_id: ScoringId,
    name: &str,
    span: Span,
    ctx: &LoweringCtx<'_>,
) -> Result<ActionId, LoweringError> {
    ctx.action_ids
        .get(name)
        .copied()
        .ok_or_else(|| LoweringError::UnknownScoringAction {
            scoring: scoring_id,
            name: name.to_string(),
            span,
        })
}

/// Type-check `utility` resolves to `CgTy::F32`. Surfaces a non-F32
/// result as [`LoweringError::ScoringUtilityNotF32`]; a re-typecheck
/// failure on the constructed node surfaces as
/// [`LoweringError::ScoringRowTypeCheckFailure`] with
/// `subject = ScoringRowSubject::Utility`.
fn check_utility_f32(
    scoring_id: ScoringId,
    action: ActionId,
    utility: CgExprId,
    span: Span,
    ctx: &LoweringCtx<'_>,
) -> Result<(), LoweringError> {
    let ty = node_ty(scoring_id, action, ScoringRowSubject::Utility, utility, span, ctx)?;
    if ty != CgTy::F32 {
        return Err(LoweringError::ScoringUtilityNotF32 {
            scoring: scoring_id,
            action,
            got: ty,
            span,
        });
    }
    Ok(())
}

/// Lower a per-ability-row target expression and confirm it
/// type-checks to `CgTy::AgentId`. Returns the lowered id on
/// success; surfaces a non-AgentId result as
/// [`LoweringError::ScoringTargetNotAgentId`].
fn lower_target(
    scoring_id: ScoringId,
    action: ActionId,
    target_node: &IrExprNode,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let target_id = lower_expr(target_node, ctx)?;
    let ty = node_ty(
        scoring_id,
        action,
        ScoringRowSubject::Target,
        target_id,
        target_node.span,
        ctx,
    )?;
    if ty != CgTy::AgentId {
        return Err(LoweringError::ScoringTargetNotAgentId {
            scoring: scoring_id,
            action,
            got: ty,
            span: target_node.span,
        });
    }
    Ok(target_id)
}

/// Lower a per-ability-row guard expression and confirm it
/// type-checks to `CgTy::Bool`. Returns the lowered id on success;
/// surfaces a non-Bool result as
/// [`LoweringError::ScoringGuardNotBool`].
fn lower_guard(
    scoring_id: ScoringId,
    action: ActionId,
    guard_node: &IrExprNode,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let guard_id = lower_expr(guard_node, ctx)?;
    let ty = node_ty(
        scoring_id,
        action,
        ScoringRowSubject::Guard,
        guard_id,
        guard_node.span,
        ctx,
    )?;
    if ty != CgTy::Bool {
        return Err(LoweringError::ScoringGuardNotBool {
            scoring: scoring_id,
            action,
            got: ty,
            span: guard_node.span,
        });
    }
    Ok(guard_id)
}

/// Look up the type of `expr_id` in the builder's expression arena.
/// Mirrors `mask::predicate_node_ty`: dangling-id / type-check
/// defects surface as
/// [`LoweringError::ScoringRowTypeCheckFailure`] with the supplied
/// [`ScoringRowSubject`] so the diagnostic names which
/// sub-expression failed.
fn node_ty(
    scoring_id: ScoringId,
    action: ActionId,
    subject: ScoringRowSubject,
    expr_id: CgExprId,
    span: Span,
    ctx: &LoweringCtx<'_>,
) -> Result<CgTy, LoweringError> {
    let prog = ctx.builder.program();
    let node = prog.exprs.get(expr_id.0 as usize).ok_or(
        LoweringError::ScoringRowTypeCheckFailure {
            scoring: scoring_id,
            action,
            subject,
            error: TypeError::DanglingExprId {
                node: expr_id,
                referenced: expr_id,
            },
            span,
        },
    )?;
    let resolver: &dyn Fn(crate::cg::data_handle::ViewId) -> Option<(Vec<CgTy>, CgTy)> =
        &|view_id| {
            ctx.view_signatures
                .get(&view_id)
                .map(|(args, result)| (args.clone(), *result))
        };
    let tc_ctx = TypeCheckCtx::with_view_signature(prog, resolver);
    type_check(node, expr_id, &tc_ctx).map_err(|e| {
        LoweringError::ScoringRowTypeCheckFailure {
            scoring: scoring_id,
            action,
            subject,
            error: e,
            span,
        }
    })
}

// ---------------------------------------------------------------------------
// `self` shadowing for scoring expression bodies
// ---------------------------------------------------------------------------

/// Walk `expr` and remove every `local_ids` entry whose `LocalRef`
/// appears as the binder of an `IrExpr::Local(local_ref, "self")`
/// reference. Returns the saved `(LocalRef, LocalId)` pairs so
/// [`restore_self_binders`] can put them back after the body lowers.
///
/// # Why this exists
///
/// In a scoring score / target / guard expression, `self` MUST resolve
/// to the per-agent dispatch row (`AgentSelfId`) — the agent doing the
/// scoring. The `lower_bare_local`'s Step 2 handles that for any
/// `Local(_, "self")` that has no `local_ids` entry.
///
/// However, an upstream pass — specifically the verb expander's
/// cascade-physics synthesis (`cg::lower::verb_expand::
/// synthesize_cascade_physics`) — registers the verb's `self` LocalRef
/// → some `LocalId` in `ctx.local_ids` so the verb body's
/// `self.<field>` reads inside the `on ActionSelected { actor: self,
/// … }` handler resolve to the event-payload actor (a let-binding the
/// physics kernel emits). When scoring then lowers the verb's score
/// expression — which references the SAME `self` LocalRef — Step 1
/// (let-bound local) wins over Step 2's structural-name match, and the
/// score body emits a `local_<N>` reference pointing at a binding that
/// exists ONLY in the cascade-physics kernel, not the scoring kernel.
/// The result is undefined-identifier WGSL in the scoring output (the
/// "local_3" gap reported on `assets/sim/boss_fight.sim`'s HeroHeal
/// verb, 2026-05-07).
///
/// Shadowing the `local_ids` entry for the duration of the body
/// lowering keeps Step 1 from firing, so the structural-name arm
/// produces `AgentSelfId` as intended. Restored afterward so adjacent
/// rows / downstream lowering observe the prior bindings intact.
///
/// # Walk shape
///
/// Recursive over `IrExprNode`. Cheap: scoring expressions are tens of
/// nodes at most. Exhaustive over `IrExpr` so a future variant forces
/// an explicit decision (matches the precedent set by
/// `expr::expr_contains_wildcard`).
fn shadow_self_binders(
    expr: &IrExprNode,
    ctx: &mut LoweringCtx<'_>,
) -> Vec<(LocalRef, LocalId)> {
    let mut refs = Vec::new();
    collect_self_local_refs(expr, &mut refs);
    let mut saved = Vec::with_capacity(refs.len());
    for local_ref in refs {
        if let Some(prior) = ctx.local_ids.remove(&local_ref) {
            saved.push((local_ref, prior));
        }
    }
    saved
}

/// Restore the `local_ids` entries removed by [`shadow_self_binders`].
fn restore_self_binders(saved: Vec<(LocalRef, LocalId)>, ctx: &mut LoweringCtx<'_>) {
    for (local_ref, prior) in saved {
        ctx.local_ids.insert(local_ref, prior);
    }
}

/// Recursive walker collecting every `LocalRef` whose name is `"self"`
/// (i.e., the source-level `self` keyword) inside `expr`. The collected
/// refs are de-duplicated by [`shadow_self_binders`] via the
/// `local_ids.remove` call (a missing entry is a no-op).
fn collect_self_local_refs(expr: &IrExprNode, out: &mut Vec<LocalRef>) {
    match &expr.kind {
        IrExpr::Local(local_ref, name) if name == "self" => {
            out.push(*local_ref);
        }
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Local(_, _)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::View(_)
        | IrExpr::Verb(_)
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. }
        | IrExpr::EnumVariant { .. }
        | IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit { .. }
        | IrExpr::AbilityRange => {}
        IrExpr::Unary(_, e) => collect_self_local_refs(e, out),
        IrExpr::Binary(_, lhs, rhs)
        | IrExpr::In(lhs, rhs)
        | IrExpr::Contains(lhs, rhs)
        | IrExpr::Index(lhs, rhs) => {
            collect_self_local_refs(lhs, out);
            collect_self_local_refs(rhs, out);
        }
        IrExpr::PerUnit { expr, delta } => {
            collect_self_local_refs(expr, out);
            collect_self_local_refs(delta, out);
        }
        IrExpr::Field { base, .. } => collect_self_local_refs(base, out),
        IrExpr::ViewCall(_, args)
        | IrExpr::VerbCall(_, args)
        | IrExpr::BuiltinCall(_, args)
        | IrExpr::UnresolvedCall(_, args)
        | IrExpr::NamespaceCall { args, .. } => {
            for a in args {
                collect_self_local_refs(&a.value, out);
            }
        }
        IrExpr::If {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_self_local_refs(cond, out);
            collect_self_local_refs(then_expr, out);
            if let Some(e) = else_expr {
                collect_self_local_refs(e, out);
            }
        }
        IrExpr::Quantifier { iter, body, .. } => {
            collect_self_local_refs(iter, out);
            collect_self_local_refs(body, out);
        }
        IrExpr::Fold { iter, body, .. } => {
            if let Some(i) = iter {
                collect_self_local_refs(i, out);
            }
            collect_self_local_refs(body, out);
        }
        IrExpr::Match { scrutinee, arms } => {
            collect_self_local_refs(scrutinee, out);
            for arm in arms {
                collect_self_local_refs(&arm.body, out);
            }
        }
        IrExpr::List(items) | IrExpr::Tuple(items) => {
            for it in items {
                collect_self_local_refs(it, out);
            }
        }
        IrExpr::StructLit { fields, .. } => {
            for f in fields {
                collect_self_local_refs(&f.value, out);
            }
        }
        IrExpr::Ctor { args, .. } => {
            for a in args {
                collect_self_local_refs(a, out);
            }
        }
        // Catch-all for leaf variants without IrExprNode children
        // (AbilityOnCooldown payload-bearing variants, Raw, etc.). If a
        // future variant adds children, the match arm needs an explicit
        // case — Rust's exhaustiveness check forces the update because
        // we already enumerated every leaf above by name.
        _ => {}
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
        IrActionHead, IrActionHeadShape, IrExpr, IrExprNode, PerAbilityRowIR, ScoringEntryIR,
        ScoringIR,
    };

    use crate::cg::data_handle::{AgentFieldId, AgentRef, DataHandle};
    use crate::cg::expr::{CgExpr, LitValue};
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

    fn lit_bool(v: bool) -> IrExprNode {
        node(IrExpr::LitBool(v))
    }

    fn action_head(name: &str) -> IrActionHead {
        IrActionHead {
            name: name.to_string(),
            shape: IrActionHeadShape::None,
            span: span(0, 0),
        }
    }

    fn standard_entry(name: &str, expr: IrExprNode) -> ScoringEntryIR {
        ScoringEntryIR {
            head: action_head(name),
            expr,
            span: span(0, 0),
        }
    }

    fn scoring_with(entries: Vec<ScoringEntryIR>, rows: Vec<PerAbilityRowIR>) -> ScoringIR {
        ScoringIR {
            entries,
            per_ability_rows: rows,
            annotations: vec![],
            span: span(0, 0),
        }
    }

    // ---- 1. Smallest happy path: one standard row -----------------------

    #[test]
    fn lowers_single_standard_row_with_literal_utility() {
        // scoring { Hold = 0.1 }
        let ir = scoring_with(vec![standard_entry("Hold", lit_f32(0.1))], vec![]);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Hold", ActionId(0));

        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers");
        assert_eq!(op_id, OpId(0));

        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::PerAgent);
        match &op.kind {
            ComputeOpKind::ScoringArgmax { scoring, rows } => {
                assert_eq!(*scoring, ScoringId(0));
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].action, ActionId(0));
                assert!(rows[0].target.is_none(), "standard row → target = None");
                assert!(rows[0].guard.is_none(), "standard row → guard = None");
            }
            other => panic!("unexpected kind: {other:?}"),
        }
        // Output binding produced by the auto-walker.
        assert_eq!(op.writes, vec![DataHandle::ScoringOutput]);
        // Scoring name interned for pretty-printing.
        assert_eq!(prog.interner.get_scoring_name(ScoringId(0)), Some("scoring0"));
    }

    // ---- 2. Multi-row scoring with three standard rows ------------------

    #[test]
    fn lowers_multiple_standard_rows() {
        // scoring { Hold = 0.1, MoveToward = 0.3, Flee = 0.4 }
        let ir = scoring_with(
            vec![
                standard_entry("Hold", lit_f32(0.1)),
                standard_entry("MoveToward", lit_f32(0.3)),
                standard_entry("Flee", lit_f32(0.4)),
            ],
            vec![],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Hold", ActionId(0));
        ctx.register_action("MoveToward", ActionId(1));
        ctx.register_action("Flee", ActionId(2));

        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers");
        assert_eq!(op_id, OpId(0));

        let prog = builder.finish();
        match &prog.ops[0].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0].action, ActionId(0));
                assert_eq!(rows[1].action, ActionId(1));
                assert_eq!(rows[2].action, ActionId(2));
                for row in rows {
                    assert!(row.target.is_none());
                    assert!(row.guard.is_none());
                }
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    // ---- 3. Per-ability row with target + guard --------------------------

    #[test]
    fn lowers_per_ability_row_with_target_and_guard() {
        // row Cast0 per_ability {
        //   guard: true
        //   score: 0.5
        //   target: self.engaged_with  (AgentId-typed field)
        // }
        //
        // The canonical AgentId-typed expression today is a field read
        // against an AgentId-typed agent slot (e.g., `self.engaged_with`).
        // The expression layer doesn't yet surface `AgentId` literals from
        // the AST surface, so a synthetic `LitInt(0)` target would lower
        // to U32 and fail the AgentId typecheck — pick an honest shape
        // that exercises the field-read path the expression layer wires.
        let target_node = node(IrExpr::Field {
            base: Box::new(node(IrExpr::Local(
                dsl_ast::ir::LocalRef(0),
                "self".to_string(),
            ))),
            field_name: "engaged_with".to_string(),
            field: None,
        });
        let row = PerAbilityRowIR {
            name: "Cast0".to_string(),
            guard: Some(lit_bool(true)),
            score: lit_f32(0.5),
            target: Some(target_node),
            weights: None,
            span: span(0, 0),
        };
        let ir = scoring_with(vec![], vec![row]);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Cast0", ActionId(7));

        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers per-ability row");
        let prog = builder.finish();

        match &prog.ops[op_id.0 as usize].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                assert_eq!(rows.len(), 1);
                let row = &rows[0];
                assert_eq!(row.action, ActionId(7));
                assert!(row.target.is_some(), "per-ability row → target = Some");
                assert!(row.guard.is_some(), "per-ability row → guard = Some");
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    // ---- 4. Per-ability row with neither target nor guard ---------------

    #[test]
    fn lowers_per_ability_row_with_neither_target_nor_guard() {
        // row R per_ability { score: 0.5 }  (no guard, no target)
        let row = PerAbilityRowIR {
            name: "R".to_string(),
            guard: None,
            score: lit_f32(0.5),
            target: None,
            weights: None,
            span: span(0, 0),
        };
        let ir = scoring_with(vec![], vec![row]);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("R", ActionId(3));

        let op_id =
            lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers naked per-ability row");
        let prog = builder.finish();

        match &prog.ops[op_id.0 as usize].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert!(rows[0].target.is_none());
                assert!(rows[0].guard.is_none());
                assert_eq!(rows[0].action, ActionId(3));
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    // ---- 5. Negative: utility expression returns Bool -------------------

    #[test]
    fn rejects_utility_returning_bool() {
        // scoring { Hold = true } — Bool utility, not F32.
        let ir = scoring_with(vec![standard_entry("Hold", lit_bool(true))], vec![]);
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Hold", ActionId(0));

        let err =
            lower_scoring(ScoringId(0), &ir, &mut ctx).expect_err("Bool utility must be rejected");
        match err {
            LoweringError::ScoringUtilityNotF32 {
                scoring,
                action,
                got,
                ..
            } => {
                assert_eq!(scoring, ScoringId(0));
                assert_eq!(action, ActionId(0));
                assert_eq!(got, CgTy::Bool);
            }
            other => panic!("expected ScoringUtilityNotF32, got {other:?}"),
        }
    }

    // ---- 6. Negative: target expression returns F32 ---------------------

    #[test]
    fn rejects_target_returning_f32() {
        // row R per_ability { score: 0.5, target: 1.0 (F32) }
        let row = PerAbilityRowIR {
            name: "R".to_string(),
            guard: None,
            score: lit_f32(0.5),
            target: Some(lit_f32(1.0)),
            weights: None,
            span: span(0, 0),
        };
        let ir = scoring_with(vec![], vec![row]);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("R", ActionId(0));

        let err =
            lower_scoring(ScoringId(0), &ir, &mut ctx).expect_err("F32 target must be rejected");
        match err {
            LoweringError::ScoringTargetNotAgentId {
                scoring,
                action,
                got,
                ..
            } => {
                assert_eq!(scoring, ScoringId(0));
                assert_eq!(action, ActionId(0));
                assert_eq!(got, CgTy::F32);
            }
            other => panic!("expected ScoringTargetNotAgentId, got {other:?}"),
        }
    }

    // ---- 7. Negative: guard expression returns U32 ----------------------

    #[test]
    fn rejects_guard_returning_u32() {
        // row R per_ability { guard: 5 (U32-typed lit), score: 0.5 }
        let row = PerAbilityRowIR {
            name: "R".to_string(),
            guard: Some(node(IrExpr::LitInt(5))),
            score: lit_f32(0.5),
            target: None,
            weights: None,
            span: span(0, 0),
        };
        let ir = scoring_with(vec![], vec![row]);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("R", ActionId(0));

        let err =
            lower_scoring(ScoringId(0), &ir, &mut ctx).expect_err("U32 guard must be rejected");
        match err {
            LoweringError::ScoringGuardNotBool {
                scoring,
                action,
                got,
                ..
            } => {
                assert_eq!(scoring, ScoringId(0));
                assert_eq!(action, ActionId(0));
                assert_eq!(got, CgTy::U32);
            }
            other => panic!("expected ScoringGuardNotBool, got {other:?}"),
        }
    }

    // ---- 8. Mixed: standard rows + per-ability rows in one scoring ------

    #[test]
    fn lowers_mixed_standard_and_per_ability_rows() {
        // scoring {
        //   Hold = 0.1
        //   MoveToward = 0.3
        //   row Cast0 per_ability { score: 0.5 }
        // }
        let ir = scoring_with(
            vec![
                standard_entry("Hold", lit_f32(0.1)),
                standard_entry("MoveToward", lit_f32(0.3)),
            ],
            vec![PerAbilityRowIR {
                name: "Cast0".to_string(),
                guard: None,
                score: lit_f32(0.5),
                target: None,
                weights: None,
                span: span(0, 0),
            }],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Hold", ActionId(0));
        ctx.register_action("MoveToward", ActionId(1));
        ctx.register_action("Cast0", ActionId(7));

        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers mixed rows");
        let prog = builder.finish();

        match &prog.ops[op_id.0 as usize].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                // Source order: entries first, per-ability rows after.
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0].action, ActionId(0));
                assert_eq!(rows[1].action, ActionId(1));
                assert_eq!(rows[2].action, ActionId(7));
                // First two standard, third per-ability — but with no
                // target/guard the field shape is identical.
                assert!(rows[0].target.is_none() && rows[0].guard.is_none());
                assert!(rows[1].target.is_none() && rows[1].guard.is_none());
                assert!(rows[2].target.is_none() && rows[2].guard.is_none());
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    // ---- 9. Snapshot: pinned `Display` form for a lowered op ------------

    #[test]
    fn snapshot_single_row_scoring_op_display() {
        // Pin the wire format produced by `ComputeOp`'s Display impl
        // for a single-row scoring decl. `Hold = 0.1`.
        let ir = scoring_with(vec![standard_entry("Hold", lit_f32(0.1))], vec![]);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Hold", ActionId(0));
        lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=scoring_argmax(scoring=#0, rows=1) shape=per_agent reads=[] writes=[scoring.output]"
        );
    }

    // ---- 10. Display impl coverage for new error variants ---------------

    #[test]
    fn lowering_error_display_scoring_utility_not_f32() {
        let e = LoweringError::ScoringUtilityNotF32 {
            scoring: ScoringId(2),
            action: ActionId(5),
            got: CgTy::Bool,
            span: span(7, 12),
        };
        let s = format!("{}", e);
        assert!(s.contains("scoring#2"), "missing scoring tag in {s:?}");
        assert!(s.contains("action=#5"), "missing action tag in {s:?}");
        // CgTy renders lowercase via Display.
        assert!(s.contains("bool"), "missing bool tag in {s:?}");
        assert!(s.contains("F32"), "missing F32 tag in {s:?}");
    }

    #[test]
    fn lowering_error_display_scoring_target_not_agent_id() {
        let e = LoweringError::ScoringTargetNotAgentId {
            scoring: ScoringId(0),
            action: ActionId(1),
            got: CgTy::F32,
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("scoring#0"), "missing scoring tag in {s:?}");
        assert!(s.contains("action=#1"), "missing action tag in {s:?}");
        // CgTy renders lowercase via Display ("f32"); the error message
        // pins the expected type as the literal "AgentId" word.
        assert!(s.contains("f32"), "missing f32 tag in {s:?}");
        assert!(s.contains("AgentId"), "missing AgentId tag in {s:?}");
    }

    #[test]
    fn lowering_error_display_scoring_guard_not_bool() {
        let e = LoweringError::ScoringGuardNotBool {
            scoring: ScoringId(3),
            action: ActionId(2),
            got: CgTy::U32,
            span: span(2, 9),
        };
        let s = format!("{}", e);
        assert!(s.contains("scoring#3"), "missing scoring tag in {s:?}");
        assert!(s.contains("action=#2"), "missing action tag in {s:?}");
        assert!(s.contains("u32"), "missing u32 tag in {s:?}");
        assert!(s.contains("Bool"), "missing Bool tag in {s:?}");
    }

    #[test]
    fn lowering_error_display_scoring_row_type_check_failure() {
        let e = LoweringError::ScoringRowTypeCheckFailure {
            scoring: ScoringId(0),
            action: ActionId(0),
            subject: ScoringRowSubject::Utility,
            error: TypeError::DanglingExprId {
                node: CgExprId(0),
                referenced: CgExprId(0),
            },
            span: span(0, 5),
        };
        let s = format!("{}", e);
        assert!(s.contains("scoring#0"));
        // Display renders the typed enum as its lowercase tag.
        assert!(s.contains("utility"), "missing utility tag in {s:?}");
    }

    // ---- 11. Empty scoring decl produces empty-rows op ------------------

    #[test]
    fn lowers_empty_scoring_to_op_with_no_rows() {
        // scoring { } — vacuous, but the IR shape permits it.
        let ir = scoring_with(vec![], vec![]);
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers empty scoring");
        let prog = builder.finish();
        match &prog.ops[op_id.0 as usize].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                assert!(rows.is_empty());
            }
            other => panic!("unexpected kind: {other:?}"),
        }
        // Auto-walker still records the ScoringOutput write.
        assert_eq!(
            prog.ops[op_id.0 as usize].writes,
            vec![DataHandle::ScoringOutput]
        );
    }

    // ---- 12. Auto-walker descends into target + guard exprs -------------

    #[test]
    fn auto_walker_collects_reads_from_target_and_guard() {
        // row R per_ability {
        //   guard: self.alive   (Bool, reads alive)
        //   score: self.hp      (F32, reads hp)  -- DSL allows hp on its
        //                       own as a score; lowers cleanly.
        //   target: self.engaged_with  (AgentId, reads engaged_with)
        // }
        let alive = node(IrExpr::Field {
            base: Box::new(node(IrExpr::Local(
                dsl_ast::ir::LocalRef(0),
                "self".to_string(),
            ))),
            field_name: "alive".to_string(),
            field: None,
        });
        let hp = node(IrExpr::Field {
            base: Box::new(node(IrExpr::Local(
                dsl_ast::ir::LocalRef(0),
                "self".to_string(),
            ))),
            field_name: "hp".to_string(),
            field: None,
        });
        let engaged = node(IrExpr::Field {
            base: Box::new(node(IrExpr::Local(
                dsl_ast::ir::LocalRef(0),
                "self".to_string(),
            ))),
            field_name: "engaged_with".to_string(),
            field: None,
        });
        let row = PerAbilityRowIR {
            name: "R".to_string(),
            guard: Some(alive),
            score: hp,
            target: Some(engaged),
            weights: None,
            span: span(0, 0),
        };
        let ir = scoring_with(vec![], vec![row]);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("R", ActionId(0));
        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers");
        let prog = builder.finish();

        // The auto-walker must have collected reads from utility, target,
        // AND guard. All three field reads should appear in `op.reads`.
        let reads = &prog.ops[op_id.0 as usize].reads;
        assert!(
            reads.contains(&DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
            "reads should include hp (utility), got {:?}",
            reads
        );
        assert!(
            reads.contains(&DataHandle::AgentField {
                field: AgentFieldId::EngagedWith,
                target: AgentRef::Self_,
            }),
            "reads should include engaged_with (target), got {:?}",
            reads
        );
        assert!(
            reads.contains(&DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::Self_,
            }),
            "reads should include alive (guard), got {:?}",
            reads
        );
    }

    // ---- 13. Unregistered action name surfaces as typed UnknownScoringAction

    #[test]
    fn unregistered_action_name_surfaces_as_unknown_scoring_action() {
        // scoring { Hold = 0.1 } — but no register_action call, so the
        // resolution fails. Asserts the dedicated typed variant fires
        // rather than overloading `ScoringRowTypeCheckFailure`.
        let ir = scoring_with(
            vec![ScoringEntryIR {
                head: IrActionHead {
                    name: "Hold".to_string(),
                    shape: IrActionHeadShape::None,
                    span: span(11, 15),
                },
                expr: lit_f32(0.1),
                span: span(0, 0),
            }],
            vec![],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // No registration — driver-side defect.

        let err = lower_scoring(ScoringId(0), &ir, &mut ctx)
            .expect_err("unregistered head must surface");
        match err {
            LoweringError::UnknownScoringAction {
                scoring,
                name,
                span,
            } => {
                assert_eq!(scoring, ScoringId(0));
                assert_eq!(name, "Hold");
                assert_eq!(span.start, 11);
                assert_eq!(span.end, 15);
            }
            other => panic!("expected UnknownScoringAction, got {other:?}"),
        }
    }

    // ---- 13b. UnknownScoringAction Display arm coverage -----------------

    #[test]
    fn lowering_error_display_unknown_scoring_action() {
        let e = LoweringError::UnknownScoringAction {
            scoring: ScoringId(2),
            name: "Hold".to_string(),
            span: span(7, 12),
        };
        let s = format!("{}", e);
        assert!(s.contains("scoring #2"), "missing scoring tag in {s:?}");
        assert!(s.contains("`Hold`"), "missing action name in {s:?}");
        assert!(s.contains("unknown action"), "missing diagnostic in {s:?}");
    }

    // ---- 13c. Parametric scoring head shape gate ------------------------

    /// Phase 6 Task 2: a non-`target` positional binder (e.g.,
    /// `Attack(victim)`) is not yet routable — only the canonical
    /// single-binder `(target: AgentId)` shape lowers. The typed
    /// gate fires before action-id resolution per the variant doc.
    #[test]
    fn rejects_positional_scoring_head_with_non_target_name() {
        let ir = scoring_with(
            vec![ScoringEntryIR {
                head: IrActionHead {
                    name: "Attack".to_string(),
                    shape: IrActionHeadShape::Positional(vec![(
                        "victim".to_string(),
                        dsl_ast::ir::LocalRef(0),
                        dsl_ast::ir::IrType::AgentId,
                    )]),
                    span: span(0, 14),
                },
                expr: lit_f32(0.5),
                span: span(0, 0),
            }],
            vec![],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Attack", ActionId(9));

        let err = lower_scoring(ScoringId(0), &ir, &mut ctx)
            .expect_err("non-`target` positional binder must be rejected");
        match err {
            LoweringError::UnsupportedScoringHeadShape {
                scoring,
                action,
                head_label,
                span,
            } => {
                assert_eq!(scoring, ScoringId(0));
                assert_eq!(action, None);
                assert_eq!(head_label, "positional");
                assert_eq!(span.start, 0);
                assert_eq!(span.end, 14);
            }
            other => panic!("expected UnsupportedScoringHeadShape, got {other:?}"),
        }
    }

    /// Phase 6 Task 2: a non-`AgentId` positional binder (e.g.,
    /// `Attack(target: AbilityId)`) rejects too — the
    /// `PerPairCandidate` semantic only covers AgentId-typed
    /// candidate sides.
    #[test]
    fn rejects_positional_scoring_head_with_non_agent_id_type() {
        let ir = scoring_with(
            vec![ScoringEntryIR {
                head: IrActionHead {
                    name: "Cast".to_string(),
                    shape: IrActionHeadShape::Positional(vec![(
                        "target".to_string(),
                        dsl_ast::ir::LocalRef(0),
                        dsl_ast::ir::IrType::AbilityId,
                    )]),
                    span: span(0, 14),
                },
                expr: lit_f32(0.5),
                span: span(0, 0),
            }],
            vec![],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Cast", ActionId(7));

        let err = lower_scoring(ScoringId(0), &ir, &mut ctx)
            .expect_err("non-AgentId positional binder must be rejected");
        match err {
            LoweringError::UnsupportedScoringHeadShape {
                scoring,
                action,
                head_label,
                ..
            } => {
                assert_eq!(scoring, ScoringId(0));
                assert_eq!(action, None);
                assert_eq!(head_label, "positional");
            }
            other => panic!("expected UnsupportedScoringHeadShape, got {other:?}"),
        }
    }

    /// Phase 6 Task 2: multi-binder positional heads (`Attack(t,
    /// u)`) reject — only the single-binder shape is wired today.
    #[test]
    fn rejects_positional_scoring_head_with_multiple_binders() {
        let ir = scoring_with(
            vec![ScoringEntryIR {
                head: IrActionHead {
                    name: "Attack".to_string(),
                    shape: IrActionHeadShape::Positional(vec![
                        (
                            "target".to_string(),
                            dsl_ast::ir::LocalRef(0),
                            dsl_ast::ir::IrType::AgentId,
                        ),
                        (
                            "secondary".to_string(),
                            dsl_ast::ir::LocalRef(1),
                            dsl_ast::ir::IrType::AgentId,
                        ),
                    ]),
                    span: span(0, 24),
                },
                expr: lit_f32(0.5),
                span: span(0, 0),
            }],
            vec![],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Attack", ActionId(9));

        let err = lower_scoring(ScoringId(0), &ir, &mut ctx)
            .expect_err("multi-binder positional head must be rejected");
        match err {
            LoweringError::UnsupportedScoringHeadShape {
                scoring,
                action,
                head_label,
                ..
            } => {
                assert_eq!(scoring, ScoringId(0));
                assert_eq!(action, None);
                assert_eq!(head_label, "positional");
            }
            other => panic!("expected UnsupportedScoringHeadShape, got {other:?}"),
        }
    }

    /// Phase 6 Task 2: `Named` heads (`Cast(ability: AbilityId)`-
    /// style) still reject — there is no per-pair candidate
    /// semantic for non-positional binders.
    #[test]
    fn rejects_named_scoring_head_shape() {
        let ir = scoring_with(
            vec![ScoringEntryIR {
                head: IrActionHead {
                    name: "Cast".to_string(),
                    shape: IrActionHeadShape::Named(vec![]),
                    span: span(0, 14),
                },
                expr: lit_f32(0.5),
                span: span(0, 0),
            }],
            vec![],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Cast", ActionId(7));

        let err = lower_scoring(ScoringId(0), &ir, &mut ctx)
            .expect_err("named head must be rejected");
        match err {
            LoweringError::UnsupportedScoringHeadShape {
                scoring,
                action,
                head_label,
                ..
            } => {
                assert_eq!(scoring, ScoringId(0));
                assert_eq!(action, None);
                assert_eq!(head_label, "named");
            }
            other => panic!("expected UnsupportedScoringHeadShape, got {other:?}"),
        }
    }

    // ---- 13c.2. Phase 6 Task 2: positional `target` lowers --------------

    /// Canonical `Attack(target) = ...` row lowers cleanly: the row
    /// body's `target.<field>` reads resolve against
    /// [`AgentRef::PerPairCandidate`] (mirrors pair-bound mask
    /// lowering's `target_local` semantic).
    #[test]
    fn lower_scoring_positional_target_binder_resolves_to_per_pair_candidate() {
        // scoring { Attack(target) = if target.alive { 0.5 } else { 0.0 } }
        let target_alive = node(IrExpr::Field {
            base: Box::new(node(IrExpr::Local(
                dsl_ast::ir::LocalRef(0),
                "target".to_string(),
            ))),
            field_name: "alive".to_string(),
            field: None,
        });
        let utility = node(IrExpr::If {
            cond: Box::new(target_alive),
            then_expr: Box::new(lit_f32(0.5)),
            else_expr: Some(Box::new(lit_f32(0.0))),
        });

        let ir = scoring_with(
            vec![ScoringEntryIR {
                head: IrActionHead {
                    name: "Attack".to_string(),
                    shape: IrActionHeadShape::Positional(vec![(
                        "target".to_string(),
                        dsl_ast::ir::LocalRef(0),
                        dsl_ast::ir::IrType::AgentId,
                    )]),
                    span: span(0, 14),
                },
                expr: utility,
                span: span(0, 0),
            }],
            vec![],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Attack", ActionId(9));

        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx)
            .expect("canonical Attack(target) row must lower");
        let prog = builder.finish();

        // The row body's `target.alive` read must surface as a
        // `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`
        // in the op's collected reads. That's the contract Phase 6
        // Task 2 unblocks.
        let reads = &prog.ops[op_id.0 as usize].reads;
        assert!(
            reads.contains(&DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::PerPairCandidate,
            }),
            "expected per-pair-candidate `target.alive` read, got {:?}",
            reads
        );
    }

    /// `target_local` is restored after the row body lowers — adjacent
    /// non-positional rows must NOT see the per-pair binding leak.
    /// Mirrors `super::mask::lower_mask`'s save/restore protocol.
    #[test]
    fn target_local_restored_after_positional_row_body() {
        // scoring {
        //   Attack(target) = 0.5    (positional — flips target_local)
        //   Hold = 0.1              (None — must NOT see the leak)
        // }
        let ir = scoring_with(
            vec![
                ScoringEntryIR {
                    head: IrActionHead {
                        name: "Attack".to_string(),
                        shape: IrActionHeadShape::Positional(vec![(
                            "target".to_string(),
                            dsl_ast::ir::LocalRef(0),
                            dsl_ast::ir::IrType::AgentId,
                        )]),
                        span: span(0, 14),
                    },
                    expr: lit_f32(0.5),
                    span: span(0, 0),
                },
                standard_entry("Hold", lit_f32(0.1)),
            ],
            vec![],
        );
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Attack", ActionId(9));
        ctx.register_action("Hold", ActionId(0));

        // Pre-condition: the entrant ctx has `target_local = false`.
        assert!(!ctx.target_local);
        lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers");
        // Post-condition: the flag is restored after every row.
        assert!(
            !ctx.target_local,
            "target_local must restore to its prior value after a positional row body lowers"
        );
    }

    #[test]
    fn lowering_error_display_unsupported_scoring_head_shape() {
        let e = LoweringError::UnsupportedScoringHeadShape {
            scoring: ScoringId(1),
            action: None,
            head_label: "named",
            span: span(3, 8),
        };
        let s = format!("{}", e);
        assert!(s.contains("scoring#1"), "missing scoring tag in {s:?}");
        assert!(s.contains("named"), "missing head_label in {s:?}");
        assert!(
            s.contains("<unresolved>"),
            "missing unresolved-action label in {s:?}"
        );
    }

    // ---- 14. Constructed CgExpr arena content sanity --------------------

    /// The lowered op references valid arena entries — every
    /// `CgExprId` carried by `target` / `guard` must resolve to a
    /// real `CgExpr`. Defense-in-depth — the builder's
    /// `validate_op_kind_refs` already enforces this, but a regression
    /// in the lowering would surface here as a panic before the
    /// well-formed pass runs.
    #[test]
    fn lowered_per_ability_row_expr_ids_resolve_to_real_arena_entries() {
        let row = PerAbilityRowIR {
            name: "R".to_string(),
            guard: Some(lit_bool(true)),
            score: lit_f32(0.5),
            target: None,
            weights: None,
            span: span(0, 0),
        };
        let ir = scoring_with(vec![], vec![row]);
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("R", ActionId(0));
        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers");
        let prog = builder.finish();

        match &prog.ops[op_id.0 as usize].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                let r = &rows[0];
                let utility = prog
                    .exprs
                    .get(r.utility.0 as usize)
                    .expect("utility id must resolve");
                match utility {
                    CgExpr::Lit(LitValue::F32(v)) => assert!((v - 0.5).abs() < f32::EPSILON),
                    other => panic!("expected F32 lit utility, got {other:?}"),
                }
                let guard_id = r.guard.expect("guard populated");
                let guard = prog.exprs.get(guard_id.0 as usize).expect("guard resolves");
                match guard {
                    CgExpr::Lit(LitValue::Bool(true)) => {}
                    other => panic!("expected Bool(true) guard, got {other:?}"),
                }
            }
            other => panic!("unexpected kind: {other:?}"),
        }
    }

    // ---- 15. `self` shadowing in scoring score expression ---------------

    /// Regression: a `self` LocalRef pre-registered in `ctx.local_ids`
    /// (as the verb expander's cascade-physics lowering does for the
    /// verb's `self` param) MUST be shadowed for the duration of the
    /// score body lowering so a bare `self` reference resolves to the
    /// structural `AgentSelfId` (the per-agent dispatch row), NOT the
    /// stale `ReadLocal { local: <chronicle-local> }` that points at a
    /// let-binding the scoring kernel never emits.
    ///
    /// Without the fix, `target != self` in a scoring score expression
    /// emits `(... != local_<N>)` where `local_<N>` is undefined in the
    /// scoring WGSL output — the "local_3" gap reported on
    /// `assets/sim/boss_fight.sim`'s HeroHeal verb (2026-05-07).
    ///
    /// The fix: `lower_standard_row`'s body lowering walks the score
    /// expression for `IrExpr::Local(_, "self")` references, removes
    /// matching `local_ids` entries before lowering, restores them
    /// after.
    #[test]
    fn self_in_score_expr_shadows_pre_registered_local_id() {
        use crate::cg::expr::CgExpr;
        use dsl_ast::ast::BinOp;
        use dsl_ast::ir::LocalRef;

        // scoring entry: `Attack(target) = (target != self)` cast to
        // f32 via `if (target != self) { 1.0 } else { 0.0 }`.
        // (`target != self` is Bool, so a direct utility-as-Bool
        // constructed entry would fail F32 typecheck; wrap in `if` so
        // the utility lowers as F32.)
        let target_neq_self = node(IrExpr::Binary(
            BinOp::NotEq,
            Box::new(node(IrExpr::Local(LocalRef(2), "target".to_string()))),
            Box::new(node(IrExpr::Local(LocalRef(1), "self".to_string()))),
        ));
        let utility = node(IrExpr::If {
            cond: Box::new(target_neq_self),
            then_expr: Box::new(lit_f32(1.0)),
            else_expr: Some(Box::new(lit_f32(0.0))),
        });

        let ir = scoring_with(
            vec![ScoringEntryIR {
                head: IrActionHead {
                    name: "Attack".to_string(),
                    shape: IrActionHeadShape::Positional(vec![(
                        "target".to_string(),
                        LocalRef(2),
                        dsl_ast::ir::IrType::AgentId,
                    )]),
                    span: span(0, 14),
                },
                expr: utility,
                span: span(0, 0),
            }],
            vec![],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Attack", ActionId(9));

        // Pre-register `self` LocalRef → some LocalId, mirroring what
        // `synthesize_pattern_binding_lets` would do during cascade
        // physics lowering for the verb's `on ActionSelected { actor:
        // self, ... }` handler. Without the fix, scoring's
        // `lower_bare_local` Step 1 would route `self` through this
        // entry as `ReadLocal { local: LocalId(99) }`.
        let pre_registered_self = LocalId(99);
        ctx.register_local(LocalRef(1), pre_registered_self);
        ctx.record_local_ty(pre_registered_self, CgTy::AgentId);

        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("lowers cleanly");
        // Snapshot post-condition state before consuming the builder.
        let restored = ctx.local_ids.get(&LocalRef(1)).copied();
        let prog = builder.finish();

        // Walk the score row's CgExpr graph and assert NO `ReadLocal`
        // references the pre-registered `self` LocalId. Every `self`
        // reference must resolve to `AgentSelfId` instead.
        match &prog.ops[op_id.0 as usize].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                let utility_id = rows[0].utility;
                let mut saw_agent_self_id = false;
                let mut visited = vec![false; prog.exprs.len()];
                let mut stack = vec![utility_id];
                while let Some(id) = stack.pop() {
                    if visited[id.0 as usize] {
                        continue;
                    }
                    visited[id.0 as usize] = true;
                    let n = &prog.exprs[id.0 as usize];
                    match n {
                        CgExpr::ReadLocal { local, .. } => {
                            assert_ne!(
                                *local, pre_registered_self,
                                "`self` in score expr leaked through to ReadLocal {{ local#{} }}; \
                                 expected AgentSelfId. The verb expander's cascade-physics \
                                 lowering pre-registered this local; scoring must shadow it.",
                                local.0
                            );
                        }
                        CgExpr::AgentSelfId => saw_agent_self_id = true,
                        CgExpr::Binary { lhs, rhs, .. } => {
                            stack.push(*lhs);
                            stack.push(*rhs);
                        }
                        CgExpr::Select { cond, then, else_, .. } => {
                            stack.push(*cond);
                            stack.push(*then);
                            stack.push(*else_);
                        }
                        _ => {}
                    }
                }
                assert!(
                    saw_agent_self_id,
                    "expected `self` reference to lower to AgentSelfId; \
                     none observed in the score expr's CgExpr graph"
                );
            }
            other => panic!("unexpected kind: {other:?}"),
        }

        // Post-condition: the pre-registered `self` LocalRef → LocalId
        // mapping is restored after the body lowers. Adjacent
        // downstream passes (or the next row) must observe the prior
        // binding intact.
        assert_eq!(
            restored,
            Some(pre_registered_self),
            "shadowed `self` LocalRef must restore to its prior LocalId after the row lowers"
        );
    }

    /// Per-ability-row variant of the same regression: `self` shadowing
    /// applies to the row's `score` / `target` / `guard` sub-expressions
    /// independently. Without the fix, a per-ability row whose `score`
    /// references `self` (e.g., `if (target != self) { ... } else { ... }`)
    /// emits the same undefined `local_<N>` gap.
    #[test]
    fn self_in_per_ability_row_score_shadows_pre_registered_local_id() {
        use crate::cg::expr::CgExpr;
        use dsl_ast::ast::BinOp;
        use dsl_ast::ir::LocalRef;

        let target_neq_self = node(IrExpr::Binary(
            BinOp::NotEq,
            Box::new(node(IrExpr::Local(LocalRef(2), "target".to_string()))),
            Box::new(node(IrExpr::Local(LocalRef(1), "self".to_string()))),
        ));
        // Per-ability rows don't have a Positional(target) head today,
        // so `target` does not lower against per-pair candidate. Wrap
        // the `target != self` expression so only the `self` half
        // exercises the shadowing path; use a simpler shape.
        let score_expr = node(IrExpr::If {
            cond: Box::new(node(IrExpr::Binary(
                BinOp::Eq,
                Box::new(node(IrExpr::Local(LocalRef(1), "self".to_string()))),
                Box::new(node(IrExpr::Local(LocalRef(1), "self".to_string()))),
            ))),
            then_expr: Box::new(lit_f32(1.0)),
            else_expr: Some(Box::new(lit_f32(0.0))),
        });
        let _ = target_neq_self; // future use if/when per-ability row gets target-binding semantics

        let row = PerAbilityRowIR {
            name: "Cast0".to_string(),
            guard: None,
            score: score_expr,
            target: None,
            weights: None,
            span: span(0, 0),
        };
        let ir = scoring_with(vec![], vec![row]);

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Cast0", ActionId(7));

        // Pre-register `self` as a let-bound local, same shape as
        // cascade physics lowering does.
        let pre_registered_self = LocalId(42);
        ctx.register_local(LocalRef(1), pre_registered_self);
        ctx.record_local_ty(pre_registered_self, CgTy::AgentId);

        let op_id =
            lower_scoring(ScoringId(0), &ir, &mut ctx).expect("per-ability row lowers cleanly");
        let restored = ctx.local_ids.get(&LocalRef(1)).copied();
        let prog = builder.finish();

        match &prog.ops[op_id.0 as usize].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                let utility_id = rows[0].utility;
                let mut visited = vec![false; prog.exprs.len()];
                let mut stack = vec![utility_id];
                while let Some(id) = stack.pop() {
                    if visited[id.0 as usize] {
                        continue;
                    }
                    visited[id.0 as usize] = true;
                    let n = &prog.exprs[id.0 as usize];
                    if let CgExpr::ReadLocal { local, .. } = n {
                        assert_ne!(
                            *local, pre_registered_self,
                            "per-ability row score: `self` leaked through to \
                             ReadLocal {{ local#{} }}",
                            local.0
                        );
                    }
                    match n {
                        CgExpr::Binary { lhs, rhs, .. } => {
                            stack.push(*lhs);
                            stack.push(*rhs);
                        }
                        CgExpr::Select { cond, then, else_, .. } => {
                            stack.push(*cond);
                            stack.push(*then);
                            stack.push(*else_);
                        }
                        _ => {}
                    }
                }
            }
            other => panic!("unexpected kind: {other:?}"),
        }

        // Post-condition: restored.
        assert_eq!(restored, Some(pre_registered_self));
    }

    /// Gap Z regression: a verb with both `self` and `target`
    /// parameters lowering a score expression that field-accesses
    /// `self.<field>` (e.g. `Heal(self, target: Agent)` with
    /// `score (100.0 - self.hp)`).
    ///
    /// The verb expander emits a scoring entry whose head shape is
    /// `Positional [(target, target_ref, AgentId)]` (mirrored from the
    /// mask head via `build_mask_head`); the cascade-physics lowering
    /// (which runs BEFORE scoring) registers BOTH the verb's `self`
    /// and `target` LocalRefs in `ctx.local_ids` via
    /// `synthesize_pattern_binding_lets`.
    ///
    /// Without `self`-shadowing, the score body's `self.hp` would hit
    /// `lower_field`'s "let-bound AgentId local" arm (line ~1050) and
    /// route through `AgentRef::Target(<base_id>)` — producing a WGSL
    /// `agent_hp[target_expr_<N>]` read with no matching `let` binding
    /// (the cascade-kernel-local binding doesn't exist in scoring's
    /// kernel scope). The pre-existing `target` shadow handles the
    /// `target` binder; this test pins that the SYMMETRIC `self`
    /// shadow is in place too.
    ///
    /// Asserts the lowered CgExpr graph contains
    /// `Read(AgentField { field: Hp, target: AgentRef::Self_ })` and
    /// NOT `AgentRef::Target(_)` for `self.hp`.
    #[test]
    fn self_field_in_pair_field_score_routes_through_caster() {
        use crate::cg::expr::CgExpr;
        use dsl_ast::ast::BinOp;
        use dsl_ast::ir::LocalRef;
        use std::collections::HashSet;

        // Verb shape: `Heal(self, target: Agent)` → scoring entry
        // with positional head `(target: AgentId)`. Score expression
        // `(100.0 - self.hp)` — only references `self.<field>`, not
        // `target.<field>`.
        let self_ref = LocalRef(1);
        let target_ref = LocalRef(2);
        let self_hp = node(IrExpr::Field {
            base: Box::new(node(IrExpr::Local(self_ref, "self".to_string()))),
            field_name: "hp".to_string(),
            field: None,
        });
        let utility = node(IrExpr::Binary(
            BinOp::Sub,
            Box::new(lit_f32(100.0)),
            Box::new(self_hp),
        ));

        let ir = scoring_with(
            vec![ScoringEntryIR {
                head: IrActionHead {
                    name: "Heal".to_string(),
                    shape: IrActionHeadShape::Positional(vec![(
                        "target".to_string(),
                        target_ref,
                        dsl_ast::ir::IrType::AgentId,
                    )]),
                    span: span(0, 20),
                },
                expr: utility,
                span: span(0, 0),
            }],
            vec![],
        );

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        ctx.register_action("Heal", ActionId(3));

        // Pre-register BOTH `self` and `target` LocalRefs as
        // let-bound AgentId locals — mirroring what
        // `synthesize_pattern_binding_lets` does during the verb's
        // cascade-physics lowering (`on ActionSelected { actor: self,
        // target: target, action_id: ... }`). Without the
        // `self`-shadowing in `lower_standard_row`, `self.hp` in the
        // score body would route through the let-bound arm in
        // `lower_field` and produce `AgentRef::Target(<base_id>)`,
        // emitting `agent_hp[target_expr_<N>]` in the WGSL — Naga
        // rejects with "no definition in scope for identifier:
        // target_expr_<N>".
        let pre_registered_self = LocalId(50);
        let pre_registered_target = LocalId(51);
        ctx.register_local(self_ref, pre_registered_self);
        ctx.record_local_ty(pre_registered_self, CgTy::AgentId);
        ctx.register_local(target_ref, pre_registered_target);
        ctx.record_local_ty(pre_registered_target, CgTy::AgentId);

        let op_id = lower_scoring(ScoringId(0), &ir, &mut ctx).expect("Heal score lowers cleanly");
        let restored_self = ctx.local_ids.get(&self_ref).copied();
        let restored_target = ctx.local_ids.get(&target_ref).copied();
        let prog = builder.finish();

        // Walk the row's CgExpr graph and collect every AgentRef
        // referenced by an `AgentField` Read. The `self.hp` access
        // must produce `AgentRef::Self_`; no `AgentRef::Target(_)` may
        // appear (would mean self.hp leaked through the let-bound arm).
        match &prog.ops[op_id.0 as usize].kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                let utility_id = rows[0].utility;
                let mut visited = HashSet::new();
                let mut stack = vec![utility_id];
                let mut saw_self_hp = false;
                while let Some(id) = stack.pop() {
                    if !visited.insert(id) {
                        continue;
                    }
                    let n = &prog.exprs[id.0 as usize];
                    match n {
                        CgExpr::Read(DataHandle::AgentField { field, target }) => {
                            assert!(
                                !matches!(target, AgentRef::Target(_)),
                                "self.hp leaked through let-bound arm to AgentRef::Target — \
                                 the verb expander pre-registered `self` LocalRef and \
                                 `lower_standard_row` failed to shadow it; field={:?} target={:?}",
                                field,
                                target
                            );
                            if *field == AgentFieldId::Hp && matches!(target, AgentRef::Self_) {
                                saw_self_hp = true;
                            }
                        }
                        CgExpr::Binary { lhs, rhs, .. } => {
                            stack.push(*lhs);
                            stack.push(*rhs);
                        }
                        CgExpr::Select { cond, then, else_, .. } => {
                            stack.push(*cond);
                            stack.push(*then);
                            stack.push(*else_);
                        }
                        _ => {}
                    }
                }
                assert!(
                    saw_self_hp,
                    "expected `self.hp` to lower to Read(AgentField {{ field: Hp, target: \
                     AgentRef::Self_ }}); none observed"
                );
            }
            other => panic!("unexpected kind: {other:?}"),
        }

        // Post-condition: BOTH pre-registered bindings are restored.
        // The `target` binder is shadowed by the positional-head arm,
        // the `self` binder by `shadow_self_binders`. Both must
        // restore cleanly so adjacent rows / downstream passes
        // observe the prior bindings intact.
        assert_eq!(
            restored_self,
            Some(pre_registered_self),
            "shadowed `self` LocalRef must restore to its prior LocalId"
        );
        assert_eq!(
            restored_target,
            Some(pre_registered_target),
            "shadowed `target` LocalRef must restore to its prior LocalId"
        );
    }
}
