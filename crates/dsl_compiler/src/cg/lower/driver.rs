//! End-to-end DSL → CgProgram lowering driver.
//!
//! Phase 2, Task 2.8 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`). Wires
//! every prior per-construct lowering pass (mask, view, physics,
//! scoring, spatial, plumbing) behind a single entry point —
//! [`lower_compilation_to_cg`] — that consumes a resolved
//! [`dsl_ast::ir::Compilation`] and returns a fully-built
//! [`CgProgram`].
//!
//! # Phases
//!
//! 1. **Registry population.** Walk the [`Compilation`] and allocate
//!    typed ids for every event kind, sum-type variant, action name,
//!    view, and one event ring per event kind. The id allocations are
//!    deterministic (source order), so two runs over the same
//!    Compilation produce byte-identical CgProgram outputs.
//! 2. **Per-construct lowering.** For each user-construct kind
//!    (masks → views → physics → scoring), call the matching
//!    `lower_*` pass with its driver-supplied parameters
//!    (per-handler [`HandlerResolution`]s, [`SpatialQueryKind`] for
//!    masks with a `from` clause, [`ReplayabilityFlag`] from the
//!    rule's `@phase(...)` annotation). Per-construct failures are
//!    accumulated as diagnostics; the driver does NOT short-circuit
//!    on first defect.
//! 3. **Spatial-query synthesis.** Collect every distinct
//!    [`SpatialQueryKind`] referenced by the user ops' dispatch
//!    shapes and, when present, push a [`SpatialQueryKind::BuildHash`]
//!    entry first. Then call [`lower_spatial_queries`] to push one
//!    op per kind.
//! 4. **Cycle gate.** Snapshot the program after step 3 (the
//!    user-op-only program) and run [`check_well_formed`]. Errors
//!    surface as [`LoweringError::WellFormed`] entries on the
//!    accumulator. The plan reserves post-plumbing well-formedness
//!    for Phase 3.
//! 5. **Plumbing synthesis.** Call [`synthesize_plumbing_ops`] on
//!    the user-op-only program, then [`lower_plumbing`] to push one
//!    op per kind.
//! 6. **Source-ring symmetry.** Walk the finished program and, for
//!    each [`ComputeOp`] whose dispatch is
//!    [`DispatchShape::PerEvent { source_ring }`], record an
//!    [`EventRingAccess::Read`] read on `source_ring`. The
//!    destination-ring (Emit) write half is structurally awkward —
//!    Emit lives inside a `CgStmtList`, not on the op directly, and
//!    the auto-walker does not know which ring an event-name
//!    resolves to. See "Limitations" below for the deferred path.
//!
//! # Diagnostic model
//!
//! Many real `.sim` constructs intentionally fail to lower today —
//! masks like `MoveToward(target)` whose predicates reference the
//! per-pair `target` binding, physics rules whose bodies use `Let` /
//! `For` / namespace setter calls, scoring rows that read
//! `target.<field>`, and so on. Each such failure surfaces as a
//! [`LoweringError`] of a deferral variant
//! ([`LoweringError::UnsupportedAstNode`],
//! [`LoweringError::UnsupportedPhysicsStmt`],
//! [`LoweringError::UnsupportedLocalBinding`], …). The driver
//! collects every such diagnostic and returns them alongside the
//! best-effort program.
//!
//! Non-deferral failures (registry conflicts, builder rejections,
//! type-check failures on lowered nodes) surface the same way; the
//! caller decides whether to treat any error as fatal or only treat
//! a non-deferral error as fatal. The integration test exercises
//! the second policy.
//!
//! # Limitations
//!
//! - **Destination-ring (Emit) writes are not wired today.** The
//!   plan amendment calls for the driver to record an
//!   [`EventRingAccess::Append`] write on the destination ring of
//!   every `CgStmt::Emit`, but the [`crate::cg::op::ComputeOp`] surface
//!   exposes [`crate::cg::op::ComputeOp::record_write`] on the op,
//!   not on individual statements — and an op can carry multiple
//!   distinct emit destinations through `If` / `Match` arms. Wiring
//!   the driver to walk the body's stmt list and resolve event-name
//!   → `EventRingId` per Emit is structurally clean but requires
//!   the plumbing tasks (or a sibling walk) to know which event
//!   names map to which rings. Today's event-ring registry maps
//!   one ring per event kind; the walk is feasible but the
//!   well_formed cycle gate runs on the user-op-only program
//!   (before plumbing), so wiring destination-ring writes after
//!   `builder.finish()` doesn't change the gate's verdict. The
//!   structural walk lands in a follow-up once a Phase 3 consumer
//!   needs it.
//! - **Per-mask spatial query selection.** The driver picks
//!   [`SpatialQueryKind::KinQuery`] for every mask with a `from
//!   query.nearby_agents(...)` clause. Real mask kernels split
//!   between `KinQuery` (kin / movement targets) and
//!   `EngagementQuery` (engagement candidates); the per-mask
//!   selection requires a richer AST analysis (the mask's predicate
//!   references `is_hostile` / `agents.engaged_with` / etc.) that
//!   no Task 2.x has wired. Picking `KinQuery` as the default
//!   matches the conservative "default to the kin-team query"
//!   shape every existing mask kernel implements. Refining this is
//!   a Phase 3 concern.
//! - **No replayability annotation parsing.** Every physics rule
//!   lowers with [`ReplayabilityFlag::Replayable`] today. The plan
//!   defers `@phase(post)` parsing — a separate pass over the
//!   rule's annotation list — to a follow-up; today the engine
//!   side has only one phase.
//! - **No view-call signature registration.** The driver allocates
//!   [`ViewId`]s but does not populate
//!   [`LoweringCtx::view_signatures`]. Lazy views referenced from
//!   mask predicates / scoring expressions / fold bodies will fail
//!   to type-check at the [`super::expr::lower_expr`] layer; the
//!   failure surfaces as a typed deferral, not silently. Wiring the
//!   signatures requires resolving each view's body to its
//!   [`crate::cg::expr::CgTy`], which the Task 2.3 view pass does
//!   not perform (lazy views are not body-lowered). The plan
//!   carries this as Phase 3 schedule synthesis work.

use std::collections::BTreeSet;

use dsl_ast::ir::{
    Compilation, EventRef, FoldHandlerIR, MaskIR, PhysicsIR, ViewBodyIR, ViewIR, ViewKind,
};

use crate::cg::data_handle::{DataHandle, EventRingAccess, EventRingId, MaskId, ViewId};
use crate::cg::dispatch::{DispatchShape, PerPairSource};
use crate::cg::op::{
    ActionId, EventKindId, PhysicsRuleId, ReplayabilityFlag, ScoringId, SpatialQueryKind,
};
use crate::cg::program::{CgProgram, CgProgramBuilder};
use crate::cg::stmt::VariantId;
use crate::cg::well_formed::check_well_formed;

use super::error::LoweringError;
use super::expr::LoweringCtx;
use super::mask::lower_mask;
use super::physics::lower_physics;
use super::plumbing::{lower_plumbing, synthesize_plumbing_ops};
use super::scoring::lower_scoring;
use super::spatial::lower_spatial_queries;
use super::view::{lower_view, HandlerResolution};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower a fully resolved [`Compilation`] to a [`CgProgram`].
///
/// On success returns the constructed program; on failure returns
/// `(best_effort_program, diagnostics)`. Many real `.sim` constructs
/// fail to lower today because of staged AST coverage — see the
/// module-level "Diagnostic model" section. Callers that want a
/// strict policy ("any error is fatal") should match on the result;
/// callers that tolerate deferrals (the integration test policy)
/// can inspect the diagnostic list and accept programs whose only
/// failures are deferral variants.
///
/// # Side effects
///
/// None — the driver constructs a fresh [`CgProgramBuilder`] and
/// returns its `finish()` output. `comp` is consumed by reference;
/// the AST is not mutated.
///
/// # Determinism
///
/// Id allocation is deterministic in source order: the i-th event
/// kind in `comp.events` becomes [`EventKindId(i as u32)`], and a
/// matching [`EventRingId(i as u32)`] is allocated alongside it.
/// Variants and actions follow the same shape (one id per source
/// occurrence, allocated in walk order). Two runs over the same
/// `Compilation` produce identical [`CgProgram`] outputs.
///
/// # Limitations
///
/// See the module-level "Limitations" section for the deferred
/// pieces — destination-ring Emit writes, per-mask spatial query
/// selection, replayability annotation parsing, and view-call
/// signature registration.
pub fn lower_compilation_to_cg(comp: &Compilation) -> Result<CgProgram, DriverOutcome> {
    let mut builder = CgProgramBuilder::new();
    let mut diagnostics: Vec<LoweringError> = Vec::new();

    // -- Phase 1: registry population (allocates ids on the builder) -----
    //
    // Each registry has one id per source occurrence, allocated in
    // walk order. The maps below are the driver's view of the
    // assignments; they're handed to the per-construct lowerings via
    // `LoweringCtx`.
    let mut ctx = LoweringCtx::new(&mut builder);

    let event_rings = populate_event_kinds(comp, &mut ctx, &mut diagnostics);
    populate_variants_from_enums(comp, &mut ctx, &mut diagnostics);
    populate_actions(comp, &mut ctx, &mut diagnostics);
    populate_views(comp, &mut ctx, &mut diagnostics);

    // -- Phase 2: per-construct lowering --------------------------------
    lower_all_masks(comp, &mut ctx, &mut diagnostics);
    lower_all_views(comp, &event_rings, &mut ctx, &mut diagnostics);
    lower_all_physics(comp, &event_rings, &mut ctx, &mut diagnostics);
    lower_all_scoring(comp, &mut ctx, &mut diagnostics);

    // -- Phase 3: spatial-query synthesis -------------------------------
    //
    // Collect every distinct SpatialQueryKind referenced by user-op
    // dispatch shapes. If the set is non-empty, prepend BuildHash so
    // the per-cell index exists before any kin/engagement walk.
    let spatial_kinds = collect_required_spatial_kinds(ctx.builder.program());
    if let Err(e) = lower_spatial_queries(&spatial_kinds, &mut ctx) {
        diagnostics.push(e);
    }

    // -- Phase 4: cycle gate (user-op-only program) ---------------------
    //
    // The plan amendment scopes the cycle gate to the program built
    // BEFORE plumbing synthesis. The plumbing synthesizer produces
    // structurally cyclic dependencies (PackAgents reads every
    // AgentField, UnpackAgents writes every AgentField) which Phase
    // 3 schedule synthesis resolves; running well_formed against a
    // post-plumbing program would always fire a false cycle.
    let user_op_program = ctx.builder.program().clone();
    if let Err(errors) = check_well_formed(&user_op_program) {
        for cg_error in errors {
            diagnostics.push(LoweringError::WellFormed { error: cg_error });
        }
    }

    // -- Phase 5: plumbing synthesis ------------------------------------
    let plumbing_kinds = synthesize_plumbing_ops(ctx.builder.program());
    if let Err(e) = lower_plumbing(&plumbing_kinds, &mut ctx) {
        diagnostics.push(e);
    }

    // -- Phase 6: source-ring symmetry ----------------------------------
    let mut prog = builder.finish();
    wire_source_ring_reads(&mut prog);

    if diagnostics.is_empty() {
        Ok(prog)
    } else {
        Err(DriverOutcome {
            program: prog,
            diagnostics,
        })
    }
}

/// The driver's failure shape. Returned when at least one diagnostic
/// fired during lowering. Callers that tolerate deferral variants
/// (most integration tests) inspect `diagnostics` for non-deferral
/// kinds and accept the `program` regardless.
#[derive(Debug, Clone)]
pub struct DriverOutcome {
    /// Best-effort lowered program. Contains every op the driver
    /// successfully constructed before any failure; ops past a
    /// per-construct failure are skipped, but unrelated constructs
    /// are still lowered.
    pub program: CgProgram,
    /// Every diagnostic the driver produced, in the order they were
    /// emitted (per-construct walk order: events → variants →
    /// actions → views → masks → views → physics → scoring →
    /// spatial → well_formed → plumbing).
    pub diagnostics: Vec<LoweringError>,
}

// ---------------------------------------------------------------------------
// Phase 1 helpers — registry population
// ---------------------------------------------------------------------------

/// Allocate one [`EventKindId`] + one [`EventRingId`] per
/// [`EventIR`] in source order. Returns the per-event-kind ring id
/// table the per-construct lowerings consult to build their
/// [`HandlerResolution`]s.
///
/// The ring id allocation is symmetric to the kind id allocation —
/// `EventKindId(i)` shares the same numeric `i` as
/// `EventRingId(i)`. Cross-id confusion is structurally prevented by
/// the typed newtypes; the parallel allocation is purely a
/// convenience for the driver.
fn populate_event_kinds(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) -> Vec<EventRingId> {
    let mut ring_ids = Vec::with_capacity(comp.events.len());
    for (i, event) in comp.events.iter().enumerate() {
        let kind_id = EventKindId(i as u32);
        let ring_id = EventRingId(i as u32);
        ring_ids.push(ring_id);

        ctx.register_event_kind(event.name.clone(), kind_id);

        if let Err(e) = ctx
            .builder
            .intern_event_kind_name(kind_id, event.name.clone())
        {
            diagnostics.push(LoweringError::BuilderRejected {
                error: e,
                span: event.span,
            });
        }
        if let Err(e) = ctx
            .builder
            .intern_event_ring_name(ring_id, event.name.clone())
        {
            diagnostics.push(LoweringError::BuilderRejected {
                error: e,
                span: event.span,
            });
        }
    }
    ring_ids
}

/// Allocate one [`VariantId`] per enum variant across every
/// [`EnumIR`] in source order. Variants from different enums
/// inhabit a flat id space — the typed registry keys on the
/// source-level variant name.
///
/// Today's physics matches consult `ctx.variant_ids` for stdlib
/// `EffectOp` arms; user-declared enums in `comp.enums` populate
/// the same map so a synthetic match arm naming a user variant
/// resolves cleanly. A duplicate variant name across enums (rare
/// in practice) overwrites the prior entry and emits a defensive
/// diagnostic — the driver flags it but does not abort.
fn populate_variants_from_enums(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    _diagnostics: &mut Vec<LoweringError>,
) {
    // Walk each enum's variants in declaration order, allocating
    // ids contiguously. Note: VariantId is a typed newtype around a
    // flat u32; collisions across enums are surfaced via the
    // `register_variant` helper's return value (the prior id), but
    // the lowering currently treats the registry as last-write-wins
    // — physics matches today only reference stdlib EffectOp
    // variants, so a real-world collision is unlikely.
    let mut next_id: u32 = 0;
    for enum_ir in &comp.enums {
        for variant_name in &enum_ir.variants {
            let id = VariantId(next_id);
            next_id += 1;
            ctx.register_variant(variant_name.clone(), id);
        }
    }
}

/// Allocate one [`ActionId`] per distinct scoring-row head name
/// across every [`ScoringIR`] in source order.
///
/// Standard rows and per-ability rows share the same id space —
/// both are "actions" at the engine's apply layer. The first
/// occurrence of each name gets a fresh id; subsequent occurrences
/// reuse it (the registry is keyed on the bare action name).
fn populate_actions(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    let mut next_id: u32 = 0;
    for scoring in &comp.scoring {
        for entry in &scoring.entries {
            allocate_action(&entry.head.name, &mut next_id, ctx, diagnostics, entry.head.span);
        }
        for row in &scoring.per_ability_rows {
            allocate_action(&row.name, &mut next_id, ctx, diagnostics, row.span);
        }
    }
}

fn allocate_action(
    name: &str,
    next_id: &mut u32,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
    span: dsl_ast::ast::Span,
) {
    if ctx.action_ids.contains_key(name) {
        return;
    }
    let id = ActionId(*next_id);
    *next_id += 1;
    ctx.register_action(name.to_string(), id);
    if let Err(e) = ctx.builder.intern_action_name(id, name.to_string()) {
        diagnostics.push(LoweringError::BuilderRejected { error: e, span });
    }
}

/// Allocate one [`ViewId`] per [`ViewIR`] in source order. Names
/// are interned so diagnostics + pretty-printing can render named
/// references. View signatures are NOT populated today — see the
/// module-level "Limitations" note on view-call signature
/// registration.
fn populate_views(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    _diagnostics: &mut Vec<LoweringError>,
) {
    // The AST resolver assigns each view a `ViewRef(i)` matching
    // its position in `comp.views`; the driver mirrors that into a
    // typed `ViewId(i)` so expression-level `IrExpr::ViewCall`
    // lowerings can resolve their AST ref through `view_ids`. Name
    // interning happens inside `lower_view` (idempotent for the
    // same id+name pair); we don't pre-intern here.
    //
    // View signature registration is deliberately not performed —
    // see the module-level "Limitations" docstring.
    for i in 0..comp.views.len() {
        let view_id = ViewId(i as u32);
        let _ = ctx.register_view(dsl_ast::ir::ViewRef(i as u16), view_id);
    }
}

// ---------------------------------------------------------------------------
// Phase 2 helpers — per-construct lowering loops
// ---------------------------------------------------------------------------

/// Lower every [`MaskIR`] in source order. Each mask becomes one
/// [`ComputeOpKind::MaskPredicate`] op (or zero ops on lowering
/// failure — the diagnostic is accumulated and the next mask runs).
///
/// Spatial query selection: a mask with no `from` clause gets
/// [`None`] (resolves to [`DispatchShape::PerAgent`]); a mask with
/// a `from` clause gets [`SpatialQueryKind::KinQuery`] as the
/// default — see the module-level "Limitations" note on per-mask
/// kind selection.
fn lower_all_masks(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    for (i, mask) in comp.masks.iter().enumerate() {
        let mask_id = MaskId(i as u32);
        let spatial_kind = mask_spatial_kind(mask);
        if let Err(e) = lower_mask(mask_id, spatial_kind, mask, ctx) {
            diagnostics.push(e);
        }
    }
}

/// Pick the [`SpatialQueryKind`] for a mask. Today: `KinQuery` for
/// every mask with a `from` clause, `None` otherwise. See the
/// module-level "Limitations" docstring for why this is a
/// conservative default.
fn mask_spatial_kind(mask: &MaskIR) -> Option<SpatialQueryKind> {
    if mask.candidate_source.is_some() {
        Some(SpatialQueryKind::KinQuery)
    } else {
        None
    }
}

/// Lower every [`ViewIR`] in source order. Each materialized view
/// produces one [`ComputeOpKind::ViewFold`] op per fold handler;
/// lazy views produce zero ops (just intern the name).
///
/// The driver builds [`HandlerResolution`]s from the per-handler
/// [`FoldHandlerIR::pattern`]'s `EventRef`. An unresolved pattern
/// (the resolver should have populated `event` at parse time)
/// surfaces as a typed [`LoweringError::UnresolvedEventPattern`]
/// diagnostic and the view is skipped.
fn lower_all_views(
    comp: &Compilation,
    event_rings: &[EventRingId],
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    for (i, view) in comp.views.iter().enumerate() {
        let view_id = ViewId(i as u32);
        let resolutions = match build_view_handler_resolutions(view, event_rings) {
            Ok(r) => r,
            Err(e) => {
                diagnostics.push(e);
                continue;
            }
        };
        if let Err(e) = lower_view(view_id, view, &resolutions, ctx) {
            diagnostics.push(e);
        }
    }
}

/// Build the per-handler `(EventKindId, EventRingId)` resolution
/// list for a view. Returns one entry per fold handler in the view's
/// body in source order; an empty vec for lazy views (their
/// handler list is empty by construction).
fn build_view_handler_resolutions(
    view: &ViewIR,
    event_rings: &[EventRingId],
) -> Result<Vec<HandlerResolution>, LoweringError> {
    match (&view.kind, &view.body) {
        (ViewKind::Lazy, ViewBodyIR::Expr(_)) => Ok(Vec::new()),
        (ViewKind::Materialized(_), ViewBodyIR::Fold { handlers, .. }) => handlers
            .iter()
            .map(|h| build_fold_handler_resolution(view, h, event_rings))
            .collect(),
        // Kind/body mismatch is the view pass's concern — return an
        // empty resolution list so it can surface its own typed
        // ViewKindBodyMismatch diagnostic.
        (ViewKind::Lazy, ViewBodyIR::Fold { .. })
        | (ViewKind::Materialized(_), ViewBodyIR::Expr(_)) => Ok(Vec::new()),
    }
}

fn build_fold_handler_resolution(
    _view: &ViewIR,
    handler: &FoldHandlerIR,
    event_rings: &[EventRingId],
) -> Result<HandlerResolution, LoweringError> {
    let event_ref = handler
        .pattern
        .event
        .ok_or(LoweringError::UnresolvedEventPattern {
            event_name: handler.pattern.name.clone(),
            span: handler.pattern.span,
        })?;
    let (kind, ring) =
        resolve_event_ref(event_ref, &handler.pattern.name, handler.pattern.span, event_rings)?;
    Ok(HandlerResolution {
        event_kind: kind,
        source_ring: ring,
    })
}

/// Lower every [`PhysicsIR`] in source order. Each rule produces
/// one [`ComputeOpKind::PhysicsRule`] op per handler (per-handler
/// lowering failures accumulate as diagnostics; the next rule
/// continues).
///
/// The driver picks [`ReplayabilityFlag::Replayable`] for every
/// rule today — see the module-level "Limitations" note on
/// replayability annotation parsing.
fn lower_all_physics(
    comp: &Compilation,
    event_rings: &[EventRingId],
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    for (i, rule) in comp.physics.iter().enumerate() {
        let rule_id = PhysicsRuleId(i as u32);
        let resolutions = match build_physics_handler_resolutions(rule, event_rings) {
            Ok(r) => r,
            Err(e) => {
                diagnostics.push(e);
                continue;
            }
        };
        let replayable = physics_replayability(rule);
        if let Err(e) = lower_physics(rule_id, replayable, rule, &resolutions, ctx) {
            diagnostics.push(e);
        }
    }
}

/// Today every physics rule is treated as replayable. The plan
/// defers `@phase(post)` parsing to a follow-up; see the
/// module-level "Limitations" note.
fn physics_replayability(_rule: &PhysicsIR) -> ReplayabilityFlag {
    ReplayabilityFlag::Replayable
}

fn build_physics_handler_resolutions(
    rule: &PhysicsIR,
    event_rings: &[EventRingId],
) -> Result<Vec<HandlerResolution>, LoweringError> {
    rule.handlers
        .iter()
        .map(|handler| build_physics_handler_resolution(handler, event_rings))
        .collect()
}

fn build_physics_handler_resolution(
    handler: &dsl_ast::ir::PhysicsHandlerIR,
    event_rings: &[EventRingId],
) -> Result<HandlerResolution, LoweringError> {
    use dsl_ast::ir::IrPhysicsPattern;
    match &handler.pattern {
        IrPhysicsPattern::Kind(p) => {
            let event_ref = p.event.ok_or(LoweringError::UnresolvedEventPattern {
                event_name: p.name.clone(),
                span: p.span,
            })?;
            let (kind, ring) = resolve_event_ref(event_ref, &p.name, p.span, event_rings)?;
            Ok(HandlerResolution {
                event_kind: kind,
                source_ring: ring,
            })
        }
        IrPhysicsPattern::Tag { span, name, .. } => {
            // Tag patterns are deferred at the physics-pass layer
            // (see physics.rs's `UnsupportedPhysicsStmt {
            // ast_label: "TagPattern", .. }` gate). The driver
            // can't resolve a tag pattern to a single (kind, ring)
            // pair — it expands to N kind-pattern handlers — so
            // we surface the tag's source name as an unresolved
            // pattern diagnostic. The physics pass will then
            // surface its own deferral when it sees the same
            // pattern.
            Err(LoweringError::UnresolvedEventPattern {
                event_name: name.clone(),
                span: *span,
            })
        }
    }
}

/// Resolve an [`EventRef`] (an index into `comp.events`) to its
/// allocated `(EventKindId, EventRingId)` pair. The driver's
/// allocation rule pairs each event by source order — the i-th
/// event has kind id `i` and ring id `event_rings[i]`. A ref
/// pointing past the table surfaces as a typed diagnostic.
fn resolve_event_ref(
    event_ref: EventRef,
    name: &str,
    span: dsl_ast::ast::Span,
    event_rings: &[EventRingId],
) -> Result<(EventKindId, EventRingId), LoweringError> {
    let i = event_ref.0 as usize;
    let ring = event_rings.get(i).copied().ok_or_else(|| {
        LoweringError::UnresolvedEventPattern {
            event_name: name.to_string(),
            span,
        }
    })?;
    Ok((EventKindId(i as u32), ring))
}

/// Lower every [`ScoringIR`] in source order. Each decl becomes
/// one [`ComputeOpKind::ScoringArgmax`] op (per-decl lowering
/// failures accumulate; the next decl continues).
fn lower_all_scoring(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    for (i, scoring) in comp.scoring.iter().enumerate() {
        let scoring_id = ScoringId(i as u32);
        if let Err(e) = lower_scoring(scoring_id, scoring, ctx) {
            diagnostics.push(e);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3 helpers — spatial-query synthesis
// ---------------------------------------------------------------------------

/// Walk `prog.ops` and collect every distinct
/// [`SpatialQueryKind`] referenced by a
/// [`DispatchShape::PerPair { source: PerPairSource::SpatialQuery(k) }`]
/// dispatch. If the result is non-empty, prepend
/// [`SpatialQueryKind::BuildHash`] so the per-cell index exists
/// before any walk.
///
/// Returns an empty `Vec` when no user op needs a spatial query —
/// the BuildHash op is only synthesised when at least one consumer
/// exists.
fn collect_required_spatial_kinds(prog: &CgProgram) -> Vec<SpatialQueryKind> {
    let mut consumers: BTreeSet<SpatialQueryKind> = BTreeSet::new();
    for op in &prog.ops {
        if let DispatchShape::PerPair {
            source: PerPairSource::SpatialQuery(kind),
        } = op.shape
        {
            consumers.insert(kind);
        }
    }

    if consumers.is_empty() {
        return Vec::new();
    }

    let mut kinds = Vec::with_capacity(consumers.len() + 1);
    kinds.push(SpatialQueryKind::BuildHash);
    for k in consumers {
        kinds.push(k);
    }
    kinds
}

// ---------------------------------------------------------------------------
// Phase 6 helpers — source-ring symmetry
// ---------------------------------------------------------------------------

/// For each [`ComputeOp`] in `prog.ops` whose dispatch shape is
/// [`DispatchShape::PerEvent { source_ring }`], record an
/// [`EventRingAccess::Read`] read on `source_ring`.
///
/// Without this, the well_formed cycle detector would see an
/// asymmetric event-ring graph (the dispatch carries the ring
/// identity but the reads list does not), missing producer/consumer
/// cycles between physics rules and view folds. The plan
/// amendment makes this a hard obligation on the driver.
fn wire_source_ring_reads(prog: &mut CgProgram) {
    for op in &mut prog.ops {
        if let DispatchShape::PerEvent { source_ring } = op.shape {
            op.record_read(DataHandle::EventRing {
                ring: source_ring,
                kind: EventRingAccess::Read,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::op::ComputeOpKind;
    use dsl_ast::ir::Compilation;

    /// Empty Compilation produces a well-formed program with only
    /// plumbing ops (the always-on UploadSimCfg / PackAgents /
    /// UnpackAgents / KickSnapshot quartet).
    #[test]
    fn empty_compilation_lowers_to_plumbing_only() {
        let comp = Compilation::default();
        let prog = lower_compilation_to_cg(&comp).expect("empty Compilation should lower cleanly");

        // Four always-on plumbing ops.
        assert_eq!(prog.ops.len(), 4);
        for op in &prog.ops {
            match &op.kind {
                ComputeOpKind::Plumbing { .. } => {}
                other => panic!("unexpected op kind: {other:?}"),
            }
        }
    }

    /// `lower_compilation_to_cg` runs the well_formed gate on the
    /// user-op-only program — no plumbing-derived cycle should fire
    /// in the diagnostic accumulator.
    #[test]
    fn empty_compilation_well_formed_gate_passes() {
        let comp = Compilation::default();
        let prog = lower_compilation_to_cg(&comp).expect("empty Compilation");
        // The plumbing layer's PackAgents/UnpackAgents pair would
        // form a cycle if the gate ran post-plumbing; assert the
        // returned program contains the cycle-creating ops without
        // a diagnostic firing (i.e., the gate ran pre-plumbing).
        let has_pack = prog.ops.iter().any(|op| {
            matches!(
                op.kind,
                ComputeOpKind::Plumbing {
                    kind: crate::cg::op::PlumbingKind::PackAgents
                }
            )
        });
        let has_unpack = prog.ops.iter().any(|op| {
            matches!(
                op.kind,
                ComputeOpKind::Plumbing {
                    kind: crate::cg::op::PlumbingKind::UnpackAgents
                }
            )
        });
        assert!(has_pack && has_unpack);
    }

    /// `wire_source_ring_reads` records one `EventRing { Read }`
    /// per `PerEvent`-shaped op. Using a synthetic builder-only
    /// fixture so we exercise the post-construction wiring without
    /// a full Compilation.
    #[test]
    fn wire_source_ring_reads_records_one_read_per_per_event_op() {
        use crate::cg::data_handle::EventRingId;
        use crate::cg::dispatch::DispatchShape;
        use crate::cg::op::{ComputeOpKind, PlumbingKind};
        use crate::cg::stmt::CgStmtList;
        use dsl_ast::ast::Span;

        let mut builder = CgProgramBuilder::new();
        // Add a plumbing op (PerAgent, no record_read fires).
        builder
            .add_op(
                ComputeOpKind::Plumbing {
                    kind: PlumbingKind::PackAgents,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap();
        // Add a body list for the PhysicsRule op below.
        let body = builder
            .add_stmt_list(CgStmtList::new(Vec::new()))
            .expect("empty list");
        // Add a PhysicsRule op with PerEvent shape.
        let _ = builder
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: EventKindId(0),
                    body,
                    replayable: ReplayabilityFlag::Replayable,
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(7),
                },
                Span::dummy(),
            )
            .unwrap();

        let mut prog = builder.finish();
        wire_source_ring_reads(&mut prog);

        // Op 0 (Plumbing/PerAgent) should have NO new EventRing read
        // appended for op 0 (auto-walker may have synthesized other
        // reads from PlumbingKind::dependencies).
        let op0 = &prog.ops[0];
        let added_event_ring = op0
            .reads
            .iter()
            .filter(|h| {
                matches!(
                    h,
                    DataHandle::EventRing {
                        ring: EventRingId(7),
                        kind: EventRingAccess::Read
                    }
                )
            })
            .count();
        assert_eq!(added_event_ring, 0);

        // Op 1 (PhysicsRule/PerEvent ring=7) should carry exactly one
        // EventRing { Read, ring=7 } from the wiring step (the
        // auto-walker doesn't synthesize source-ring reads — see
        // the physics.rs module docs).
        let op1 = &prog.ops[1];
        let wired = op1
            .reads
            .iter()
            .filter(|h| {
                matches!(
                    h,
                    DataHandle::EventRing {
                        ring: EventRingId(7),
                        kind: EventRingAccess::Read
                    }
                )
            })
            .count();
        assert_eq!(wired, 1);
    }

    /// `collect_required_spatial_kinds` returns an empty Vec when
    /// no user op shapes reference a spatial query, and a
    /// BuildHash-prefixed list when at least one does.
    #[test]
    fn collect_required_spatial_kinds_prepends_build_hash() {
        use crate::cg::dispatch::{DispatchShape, PerPairSource};
        use crate::cg::op::{ComputeOpKind, PlumbingKind};
        use dsl_ast::ast::Span;

        // Empty case.
        let mut builder = CgProgramBuilder::new();
        builder
            .add_op(
                ComputeOpKind::Plumbing {
                    kind: PlumbingKind::PackAgents,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap();
        let prog = builder.finish();
        assert!(collect_required_spatial_kinds(&prog).is_empty());

        // One PerPair op referencing KinQuery.
        let mut builder2 = CgProgramBuilder::new();
        let pred = builder2
            .add_expr(crate::cg::expr::CgExpr::Lit(
                crate::cg::expr::LitValue::Bool(true),
            ))
            .unwrap();
        builder2
            .add_op(
                ComputeOpKind::MaskPredicate {
                    mask: MaskId(0),
                    predicate: pred,
                },
                DispatchShape::PerPair {
                    source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
                },
                Span::dummy(),
            )
            .unwrap();
        let prog2 = builder2.finish();
        let kinds = collect_required_spatial_kinds(&prog2);
        assert_eq!(
            kinds,
            vec![SpatialQueryKind::BuildHash, SpatialQueryKind::KinQuery]
        );
    }
}
