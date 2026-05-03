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
//! 4. **Ring-edge wiring (pre-gate).** For every user op whose
//!    dispatch shape is [`DispatchShape::PerEvent { source_ring }`],
//!    record an [`EventRingAccess::Read`] read on `source_ring`. For
//!    every [`crate::cg::stmt::CgStmt::Emit`] reachable from any
//!    op's body (descending through `If` / `Match` arms), record an
//!    [`EventRingAccess::Append`] write on the destination ring.
//!    Both wirings mutate the in-progress builder via
//!    [`CgProgramBuilder::ops_mut`] so the cycle gate (step 5) sees
//!    the symmetric ring graph.
//! 5. **Cycle gate.** Snapshot the program after step 4 (the
//!    user-op-only program with ring edges wired) and run
//!    [`check_well_formed`]. Errors surface as
//!    [`LoweringError::WellFormed`] entries on the accumulator. The
//!    plan reserves post-plumbing well-formedness for Phase 3.
//! 6. **Plumbing synthesis.** Call [`synthesize_plumbing_ops`] on
//!    the user-op-only program, then [`lower_plumbing`] to push one
//!    op per kind.
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
//! - **Per-mask spatial query selection.** The driver routes
//!   from-bearing masks to [`SpatialQueryKind::EngagementQuery`]
//!   when their predicate references engagement-flavoured access
//!   patterns (`agents.is_hostile_to`, `agents.engaged_with`, or
//!   any `IrExpr::ViewCall` — conservative widening; see
//!   `predicate_uses_engagement_relationship`). All other
//!   from-bearing masks route to [`SpatialQueryKind::KinQuery`].
//!   Refining the ViewCall test to gate on the called view's name
//!   is a follow-up — punted because no current counterexample
//!   exists in `assets/sim/masks.sim`.
//! - **No replayability annotation parsing.** Every physics rule
//!   lowers with [`ReplayabilityFlag::Replayable`] today. The plan
//!   defers `@phase(post)` parsing — a separate pass over the
//!   rule's annotation list — to a follow-up; today the engine
//!   side has only one phase.
//! - **Lazy view inlining.** Lazy view bodies are captured into
//!   [`super::expr::LoweringCtx::lazy_view_bodies`] during Phase 1
//!   (see `populate_view_bodies_and_signatures`); call sites
//!   inline the body via
//!   [`super::expr::lower_expr`]'s `IrExpr::ViewCall` arm.
//!   Materialized view signatures are populated in the same Phase
//!   1 walk; downstream `BuiltinId::ViewCall { view }` lowerings
//!   resolve through `ctx.view_signatures`.

use std::collections::{BTreeMap, BTreeSet};

use dsl_ast::ir::{
    Compilation, EventRef, FoldHandlerIR, IrCallArg, IrExpr, IrExprNode, IrType, MaskIR,
    NamespaceId, PhysicsIR, ViewBodyIR, ViewIR, ViewKind,
};

use crate::cg::data_handle::{
    AgentFieldId, AgentRef, ConfigConstId, CgExprId, DataHandle, EventRingAccess, EventRingId,
    MaskId, ViewId,
};
use crate::cg::dispatch::{DispatchShape, PerPairSource};
use crate::cg::expr::CgTy;
use crate::cg::op::{
    ActionId, ComputeOp, ComputeOpKind, EventKindId, PhysicsRuleId, ReplayabilityFlag, ScoringId,
    SpatialQueryKind,
};
use crate::cg::program::{
    CgProgram, CgProgramBuilder, EventLayout, FieldDef, FieldLayout, MethodDef, NamespaceDef,
    NamespaceRegistry, WgslAccessForm,
};
use crate::cg::stmt::{CgStmt, CgStmtList, CgStmtListId, VariantId};
use crate::cg::well_formed::check_well_formed;

use super::error::LoweringError;
use super::expr::{lower_expr, LoweringCtx};
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
/// pieces — per-mask spatial query selection, replayability
/// annotation parsing, and view-call signature registration.
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
    populate_config_consts(comp, &mut ctx, &mut diagnostics);
    populate_views(comp, &mut ctx, &mut diagnostics);
    populate_view_bodies_and_signatures(comp, &mut ctx, &mut diagnostics);
    populate_namespace_registry(&mut ctx);

    // -- Phase 2: per-construct lowering --------------------------------
    lower_all_masks(comp, &mut ctx, &mut diagnostics);
    lower_all_views(comp, &event_rings, &mut ctx, &mut diagnostics);
    lower_all_physics(comp, &event_rings, &mut ctx, &mut diagnostics);
    lower_all_scoring(comp, &mut ctx, &mut diagnostics);

    // -- Phase 2b: Movement synthesis (Phase 6 Task 3) ------------------
    //
    // Movement is a per-agent rule that consumes scoring's chosen
    // action+target output and writes the agent's position. It is
    // structurally a `PhysicsRule` op with `on_event = None`
    // (PerAgent dispatch over the alive bitmap), distinct from the
    // PerEvent physics rules `lower_all_physics` lowered above. The
    // body is a hand-written WGSL fragment at emit time
    // (`MOVEMENT_BODY` in cg/emit/kernel.rs); the op's reads/writes
    // are recorded explicitly so the BGL synthesis sees the right
    // bindings.
    //
    // The same shape generalizes to any future per-agent sweep
    // (cooldown ticking, stun expiry, need decay, regen, …) — each
    // becomes another `PhysicsRule { on_event: None }` op with its
    // own body template + reads/writes signature.
    //
    // **Phase 6 Task 4 (2026-04-30): Movement-as-rule synth deferred
    // pending Scoring lowering**. The CG-emitted Movement op produces
    // a placeholder kernel body (`MOVEMENT_BODY` const in
    // `cg/emit/kernel.rs`) that touches its bindings but does not
    // mutate position. Real position updates would require:
    //   1. The IR to express vec3 deltas + action-conditional
    //      branching (today there's no abstraction for either).
    //   2. The BGL to bind the agents SoA as `array<u32>` (single
    //      buffer, manual offset arithmetic) instead of the
    //      per-AgentField shape `BindingMetadata` produces today
    //      (`array<vec3<f32>>` aliases the same buffer at the wrong
    //      stride).
    //   3. Scoring to write real (action, target) tuples — today
    //      `scoring.wgsl`'s body is a stub writing ACTION_HOLD.
    //
    // While (1) and (2) are tractable structural extensions, (3) is
    // the upstream blocker: even with a perfect Movement WGSL body,
    // every agent's action is `ACTION_HOLD` and Movement's
    // conditional branch `if (action == ACTION_MOVE_TOWARD)` never
    // fires. The Route C splice
    // (`xtask::compile_dsl_cmd::route_c_movement`) carries a
    // hand-written `movement.wgsl` with the real body shape; it sits
    // at the end of the SCHEDULE and runs on the same
    // `transient.action_buf` Scoring writes to.
    //
    // For Phase 6 Task 4 we drop the CG-emitted FusedMovement
    // (placeholder, no-op) and rely on the Route C splice's Movement
    // (real body, blocked on Scoring stub). When Scoring lowers for
    // real (a follow-up Phase 6 task or its own plan), Movement-as-rule
    // can be re-synthesised here with a real WGSL body.
    let _ = synthesize_movement_op; // silence dead-code; keep helper for future re-enable
    if comp.scoring.is_empty() {
        // No scoring → no scoring_output to consume → no Movement.
        // (A pure-events test fixture would land here.)
    }

    // -- Phase 3: spatial-query synthesis -------------------------------
    //
    // Collect every distinct SpatialQueryKind referenced by user-op
    // dispatch shapes. If the set is non-empty, prepend BuildHash so
    // the per-cell index exists before any kin/engagement walk.
    let spatial_kinds = collect_required_spatial_kinds(ctx.builder.program());
    if let Err(e) = lower_spatial_queries(&spatial_kinds, &mut ctx) {
        diagnostics.push(e);
    }

    // -- Phase 4: ring-edge wiring (pre-gate) ---------------------------
    //
    // The plan amendment (lines 575–595 of
    // `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`)
    // makes ring-edge symmetry a hard obligation on the driver. For
    // every PerEvent-shaped op the driver records an
    // `EventRingAccess::Read` on its source ring; for every
    // `CgStmt::Emit` reachable from any op's body the driver records
    // an `EventRingAccess::Append` on the destination ring (mapped
    // 1:1 by `EventRingId(i) ↔ EventKindId(i)` per Phase 1's
    // allocation rule). Without this, `check_well_formed`'s
    // `detect_cycles` (which consults only `op.reads` /
    // `op.writes`) silently misses event-ring producer/consumer
    // cycles between physics rules and view folds.
    //
    // The destination-ring walk needs the program's statement
    // arenas; we collect a snapshot first, compute (op_index, dest
    // rings) pairs against it, then apply both wirings to the
    // builder's ops via `ops_mut`.
    let arena_snapshot = ctx.builder.program().clone();
    let emit_writes = collect_emit_destination_rings(&arena_snapshot);
    wire_source_ring_reads(ctx.builder.ops_mut());
    apply_emit_destination_rings(ctx.builder.ops_mut(), &emit_writes);

    // -- Phase 5: cycle gate (user-op-only program) ---------------------
    //
    // The plan amendment scopes the cycle gate to the program built
    // BEFORE plumbing synthesis. The plumbing synthesizer produces
    // structurally cyclic dependencies (PackAgents reads every
    // AgentField, UnpackAgents writes every AgentField) which Phase
    // 3 schedule synthesis resolves; running well_formed against a
    // post-plumbing program would always fire a false cycle.
    //
    // Ring edges (Phase 4) must be wired BEFORE this snapshot —
    // see the rationale on Phase 4.
    //
    // View signatures must be populated on the builder's program
    // BEFORE the snapshot too — `check_well_formed`'s view-key
    // relaxation rule (Task 5) consults `prog.view_signatures` when
    // accepting `Assign(ViewStorage{Primary}, scalar)` shapes whose
    // value is the underlying scalar (e.g., `f32 += 1.0` against
    // `view_key<f32>`). Without this, the cycle gate would see the
    // unpopulated registry and reject every materialized-view fold
    // body's `+= scalar`.
    let view_signatures_snapshot: BTreeMap<u32, crate::cg::program::ViewSignature> = ctx
        .view_signatures
        .iter()
        .map(|(view_id, (args, result))| {
            (
                view_id.0,
                crate::cg::program::ViewSignature {
                    args: args.clone(),
                    result: *result,
                },
            )
        })
        .collect();
    ctx.builder
        .set_view_signatures(view_signatures_snapshot.clone());

    let user_op_program = ctx.builder.program().clone();
    if let Err(errors) = check_well_formed(&user_op_program) {
        for cg_error in errors {
            diagnostics.push(LoweringError::WellFormed { error: cg_error });
        }
    }

    // -- Phase 6: plumbing synthesis ------------------------------------
    let plumbing_kinds = synthesize_plumbing_ops(ctx.builder.program());
    if let Err(e) = lower_plumbing(&plumbing_kinds, &mut ctx) {
        diagnostics.push(e);
    }

    // Snapshot the per-kind layouts populated by `populate_event_kinds`
    // BEFORE `finish` consumes the builder. The WGSL emit consults the
    // program's `event_layouts` — copying here is the single hand-off
    // from the lowering-time `LoweringCtx` to the post-finish program.
    let event_layouts_snapshot: BTreeMap<u32, EventLayout> = ctx
        .event_layouts
        .iter()
        .map(|(k, v)| (k.0, v.clone()))
        .collect();
    let namespace_registry_snapshot = ctx.namespace_registry.clone();

    let mut prog = builder.finish();
    prog.event_layouts = event_layouts_snapshot;
    prog.namespace_registry = namespace_registry_snapshot;
    // `view_signatures` was set on the builder's program BEFORE the
    // cycle gate (above); `finish()` preserves it. No re-snapshot
    // needed here.

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

/// Allocate one [`EventKindId`] per [`EventIR`] in source order, and
/// allocate ONE shared [`EventRingId`] for every event kind. Returns
/// the per-event-kind ring id table the per-construct lowerings
/// consult to build their [`HandlerResolution`]s.
///
/// All event kinds share `EventRingId(0)`, named `batch_events` on
/// the interner. This mirrors the runtime contract: the
/// resident-context owns ONE `batch_events_ring` buffer that carries
/// every event tag interleaved (see
/// `crates/engine_gpu_rules/src/resident_context.rs`). The earlier
/// 1:1 [`EventKindId`]↔[`EventRingId`] allocation rule was a
/// Phase-1 placeholder; per-kind ring identity is preserved at the
/// WGSL level via the in-kernel `event.tag` decode, and the
/// dispatch layer drives a single ring's tail count.
fn populate_event_kinds(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) -> Vec<EventRingId> {
    let shared_ring = EventRingId(0);
    if let Err(e) = ctx
        .builder
        .intern_event_ring_name(shared_ring, "batch_events".to_string())
    {
        diagnostics.push(LoweringError::BuilderRejected {
            error: e,
            span: dsl_ast::ast::Span::dummy(),
        });
    }

    let mut ring_ids = Vec::with_capacity(comp.events.len());
    for (i, event) in comp.events.iter().enumerate() {
        let kind_id = EventKindId(i as u32);
        ring_ids.push(shared_ring);

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

        // Populate per-event payload layout + field-index registry.
        // The layout mirrors the runtime's `pack_event` source of truth
        // at `crates/engine_gpu/src/event_ring.rs`: every `event_tag`-
        // implied field (`tick`) plus the user-declared fields are
        // packed into the payload in declaration order. Variable-width
        // primitives (`Vec3` = 3 words, `u64`-bearing fields = 2 words)
        // are mirrored from `pack_event` here so the WGSL emit reads
        // the same layout the CPU writes.
        //
        // Today every kind shares the global stride 10 (2 header + 8
        // payload — sized for `AgentMoved`/`AgentFled`); when the
        // runtime moves to per-kind ring fanout, this is where the
        // per-kind values populate.
        let mut fields = BTreeMap::new();
        let mut next_offset: u32 = 0;
        for fld in &event.fields {
            let (ty, word_count) = match cg_ty_for_event_field(&fld.ty) {
                Some(p) => p,
                None => {
                    // Defer: an event field whose IrType has no GPU
                    // representation (e.g. `String`, `List<...>`) is a
                    // runtime-only event channel; the GPU pack table
                    // wouldn't carry it either. Skip the field's entry
                    // in the layout — emits referencing it surface as
                    // a separate diagnostic via the existing
                    // `event_field_indices` path.
                    continue;
                }
            };
            // Mirror `event_field_indices`: declaration-order index.
            // Existing tests pre-register these; the driver here is
            // additive — they can coexist.
            let index_u8 = u8::try_from(fields.len()).unwrap_or(u8::MAX);
            ctx.register_event_field(kind_id, fld.name.clone(), index_u8);
            fields.insert(
                fld.name.clone(),
                FieldLayout {
                    word_offset_in_payload: next_offset,
                    word_count,
                    ty,
                },
            );
            next_offset += word_count;
        }
        // Implicit `tick: u32` is the second header word (index 1) —
        // see `event_ring.rs::EventRecord` (kind, tick, payload). It is
        // NOT a payload field, so it does not get a `FieldLayout` here;
        // a body that reads `tick` would resolve through a different
        // mechanism (today nothing reads it directly via the pattern
        // binder surface — `tick` is implicit, never named in `on
        // <Event> { ... }` bindings).
        let layout = EventLayout {
            // Today's runtime: shared ring with stride 10 (= 2 header
            // + 8 payload words). See `crates/engine_gpu/src/event_ring.rs`
            // PAYLOAD_WORDS = 8.
            record_stride_u32: 10,
            header_word_count: 2,
            // Post-iter-2 every event kind reads from the shared
            // `EventRingId(0)` ring; the structural namer drops the
            // `_<ring.0>` suffix so the binding name `event_ring`
            // matches the ViewFold preamble convention. When the
            // runtime moves to per-kind ring fanout, this becomes
            // `event_ring_<ring_id>` per kind (and the structural
            // namer restores its suffix in tandem).
            buffer_name: "event_ring".to_string(),
            fields,
        };
        ctx.register_event_layout(kind_id, layout);
    }
    ring_ids
}

/// Map an [`IrType`] to a `(CgTy, word_count)` pair for event-field
/// layout. `word_count` is the number of u32 words the CPU's
/// `pack_event` writes for this field; the WGSL emit reads the same
/// number of words back. Returns `None` for non-GPU-representable
/// types (variable-length lists, strings) — those fields are skipped
/// in the layout.
///
/// The mapping mirrors `pack_event` at
/// `crates/engine_gpu/src/event_ring.rs`:
/// - `AgentId`, `AbilityId`, `QuestId`, `ItemId`, `EventId`, `GroupId`
///   → 1 word, `CgTy::AgentId` (single u32 slot for any opaque id).
/// - `u8`, `u16`, `u32`, `i8`, `i16` → 1 word (CPU-side widening), `U32` / `I32`.
/// - `f32` → 1 word, `F32`.
/// - `vec3` → 3 words, `Vec3F32`.
/// - `u64`, `i64` → 2 words via `split_u64`, `U32` (typed as low/high
///   pair; the binder's lowering treats it as opaque u32 today —
///   future: explicit `U64` CgTy variant).
/// - User-declared `Enum { ... }` → 1 word, `U32`.
/// - `String`, `List<...>`, `Optional<...>` → not GPU-representable
///   today; layout omits them.
fn cg_ty_for_event_field(ty: &IrType) -> Option<(CgTy, u32)> {
    use IrType::*;
    Some(match ty {
        Bool => (CgTy::Bool, 1),
        I8 | I16 | I32 => (CgTy::I32, 1),
        U8 | U16 | U32 => (CgTy::U32, 1),
        F32 => (CgTy::F32, 1),
        Vec3 => (CgTy::Vec3F32, 3),
        // Opaque ids — the CPU packs every one into a single u32 slot.
        // The binder's CG type is `AgentId` for the agent ids the
        // pattern names; `AbilityId` / `QuestId` etc. surface as
        // `AgentId` at the IR level too because the IR's CgTy doesn't
        // distinguish opaque-id flavours yet. This is intentional —
        // type-safety on opaque ids is a separate concern from layout.
        AgentId | AbilityId | ItemId | GroupId | QuestId | AuctionId | EventId => {
            (CgTy::AgentId, 1)
        }
        // 64-bit fields are split into (lo, hi) by `split_u64` /
        // `join_u64`. The IR's CgTy doesn't have a U64 variant; the
        // binder reads only the low word as U32 today. A future
        // pattern-binder request for the full u64 would surface as a
        // separate concern (the runtime side already round-trips
        // correctly via the two-word slot).
        I64 | U64 => (CgTy::U32, 2),
        // User enums are repr(u8) but widened to u32 at the slot
        // boundary (see `EffectGoldTransfer`'s reason / kind_tag
        // handling). One slot, U32-typed.
        Enum { .. } => (CgTy::U32, 1),
        // Non-representable: deliberately left for a future task to
        // light up (no current event uses these in a payload binder).
        F64 | String | SortedVec(..) | RingBuffer(..) | SmallVec(..) | Array(..)
        | Optional(..) | Tuple(..) | List(..) | EntityRef(..) | EventRef(..) => return None,
        // Resolver placeholders — `Unknown` is the un-typed default;
        // `Named` is a forward-resolution stub. Neither has a stable
        // GPU width.
        Unknown | Named(_) => return None,
    })
}

/// Populate the stdlib namespace registry — schema for every
/// [`super::super::expr::CgExpr::NamespaceCall`] /
/// [`super::super::expr::CgExpr::NamespaceField`] the lowering may
/// produce. The registry is the source of truth for return types +
/// arg signatures + WGSL emit forms; adding a new namespace symbol is
/// a one-edit-here change, not an IR shape change.
///
/// **B1 stubs (Task 4 of the CG lowering gap closure plan):** the WGSL
/// stub bodies are semantic no-ops chosen so the shader compiles and
/// the kernel runs without panicking. Real implementations are runtime-
/// format work (Task 9-11 territory). The registered methods today are:
///
/// * `agents.is_hostile_to(target)` → `bool`. B1: returns `false`.
///   Real semantics: `CreatureType::is_hostile_to` from `entities.sim`.
/// * `agents.engaged_with_or(target, fallback)` → `AgentId`. B1:
///   returns `fallback`. Real semantics: read the target's
///   engagement slot, sentinel-coerce to `fallback` on `INVALID`.
/// * `query.nearest_hostile_to_or(actor, range, fallback)` →
///   `AgentId`. B1: returns `fallback`. Real semantics: spatial-
///   query walk for nearest hostile, sentinel on miss.
///
/// And the registered fields:
///
/// * `world.tick` → `u32`. B1: kernel-preamble local `tick` (bound by
///   the fold-kernel's `let tick = cfg.tick;` line). The view-fold
///   preamble was renamed from `_tick` to `tick` so the access form
///   resolves cleanly; non-fold kernels that read `world.tick` would
///   need the same preamble entry, but today no non-fold kernel uses
///   it.
fn populate_namespace_registry(ctx: &mut LoweringCtx<'_>) {
    let mut registry = NamespaceRegistry::default();

    // -- agents namespace --
    let mut agents = NamespaceDef {
        name: "agents".to_string(),
        ..NamespaceDef::default()
    };
    // `agents.is_hostile_to(a, b)` — verified against
    // `assets/sim/views.sim:25` (`@lazy view is_hostile`):
    //   `agents.is_hostile_to(a, b)`
    // → 2 args, both AgentId. Returns `bool`.
    agents.methods.insert(
        "is_hostile_to".to_string(),
        MethodDef {
            return_ty: CgTy::Bool,
            arg_tys: vec![CgTy::AgentId, CgTy::AgentId],
            wgsl_fn_name: "agents_is_hostile_to".to_string(),
            wgsl_stub: "fn agents_is_hostile_to(a: u32, b: u32) -> bool { return false; }"
                .to_string(),
        },
    );
    agents.methods.insert(
        "engaged_with_or".to_string(),
        MethodDef {
            return_ty: CgTy::AgentId,
            arg_tys: vec![CgTy::AgentId, CgTy::AgentId],
            wgsl_fn_name: "agents_engaged_with_or".to_string(),
            wgsl_stub:
                "fn agents_engaged_with_or(target: u32, fallback: u32) -> u32 { return fallback; }"
                    .to_string(),
        },
    );
    registry.namespaces.insert(NamespaceId::Agents, agents);

    // -- query namespace --
    let mut query = NamespaceDef {
        name: "query".to_string(),
        ..NamespaceDef::default()
    };
    // `query.nearest_hostile_to_or(actor, range, fallback)` — verified
    // against `assets/sim/physics.sim:441`:
    //   `query.nearest_hostile_to_or(mover, config.combat.engagement_range, mover)`
    // → 3 args: AgentId, F32, AgentId.
    query.methods.insert(
        "nearest_hostile_to_or".to_string(),
        MethodDef {
            return_ty: CgTy::AgentId,
            arg_tys: vec![CgTy::AgentId, CgTy::F32, CgTy::AgentId],
            wgsl_fn_name: "query_nearest_hostile_to_or".to_string(),
            wgsl_stub:
                "fn query_nearest_hostile_to_or(actor: u32, range: f32, fallback: u32) -> u32 { return fallback; }"
                    .to_string(),
        },
    );
    registry.namespaces.insert(NamespaceId::Query, query);

    // -- world namespace --
    let mut world = NamespaceDef {
        name: "world".to_string(),
        ..NamespaceDef::default()
    };
    world.fields.insert(
        "tick".to_string(),
        FieldDef {
            ty: CgTy::U32,
            wgsl_access: WgslAccessForm::PreambleLocal {
                local_name: "tick".to_string(),
            },
        },
    );
    registry.namespaces.insert(NamespaceId::World, world);

    ctx.namespace_registry = registry;
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
/// in practice) overwrites the prior entry and pushes a typed
/// [`LoweringError::DuplicateVariantInRegistry`] — the driver
/// flags it but does not abort.
fn populate_variants_from_enums(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    // Walk each enum's variants in declaration order, allocating
    // ids contiguously. Collisions across enums are surfaced via
    // `register_variant`'s return value (the prior id) — the
    // lowering treats the registry as last-write-wins and pushes
    // a typed diagnostic so callers can refuse the program.
    let mut next_id: u32 = 0;
    for enum_ir in &comp.enums {
        for variant_name in &enum_ir.variants {
            let id = VariantId(next_id);
            next_id += 1;
            if let Some(prior_id) = ctx.register_variant(variant_name.clone(), id) {
                diagnostics.push(LoweringError::DuplicateVariantInRegistry {
                    name: variant_name.clone(),
                    prior_id,
                    new_id: id,
                });
            }
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
///
/// A duplicate registration (the same `AstViewRef` resolving twice)
/// is a driver-side defect — `ViewId`s are allocated in source
/// order and the AST resolver assigns each view a unique ref. The
/// driver pushes a typed
/// [`LoweringError::DuplicateViewInRegistry`] if it ever observes
/// one and continues with last-write-wins.
fn populate_views(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
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
        let ast_ref = dsl_ast::ir::ViewRef(i as u16);
        if let Some(prior_id) = ctx.register_view(ast_ref, view_id) {
            diagnostics.push(LoweringError::DuplicateViewInRegistry {
                ast_ref,
                prior_id,
                new_id: view_id,
            });
        }
    }
}

/// Allocate one [`ConfigConstId`] per (block, field) pair across
/// every [`dsl_ast::ir::ConfigIR`] in source order, register each
/// into `ctx.config_const_ids` keyed on
/// `(NamespaceId::Config, "<block>.<field>")`, and intern the
/// human-readable name on the builder for diagnostics +
/// pretty-printing. The id allocation is deterministic — the
/// flat numeric `i` reflects walk order.
///
/// A duplicate registration (same (block, field) pair across two
/// `ConfigIR`s) is a driver-side defect; surfaced as a typed
/// [`LoweringError::DuplicateConfigConstInRegistry`] diagnostic
/// with last-write-wins semantics.
fn populate_config_consts(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    let mut next_id: u32 = 0;
    for cfg in &comp.configs {
        for fld in &cfg.fields {
            let id = ConfigConstId(next_id);
            next_id += 1;
            let key = format!("{}.{}", cfg.name, fld.name);
            if let Some(prior) =
                ctx.register_config_const(NamespaceId::Config, key.clone(), id)
            {
                diagnostics.push(LoweringError::DuplicateConfigConstInRegistry {
                    key: key.clone(),
                    prior_id: prior,
                    new_id: id,
                });
            }
            if let Err(e) = ctx.builder.intern_config_const_name(id, key) {
                diagnostics.push(LoweringError::BuilderRejected {
                    error: e,
                    span: fld.span,
                });
            }
            // Capture the literal default into the program's
            // `config_const_values` map so the WGSL emit can produce
            // an inline `const config_<id>: f32 = <value>;` for every
            // referenced const. Today F32-only — boids' fixture is
            // all-float and the parser surface accepts numeric forms
            // the cast can compress to f32; non-numeric defaults
            // (Bool / String) skip silently because no compute kernel
            // references them.
            use dsl_ast::ast::ConfigDefault;
            let value: Option<f32> = match &fld.default {
                ConfigDefault::Float(v) => Some(*v as f32),
                ConfigDefault::Int(v) => Some(*v as f32),
                ConfigDefault::Uint(v) => Some(*v as f32),
                ConfigDefault::Bool(_) | ConfigDefault::String(_) => None,
            };
            if let Some(v) = value {
                ctx.builder.set_config_const_value(id, v);
            }
        }
    }
}

/// For every view in source order: capture lazy bodies into
/// `ctx.lazy_view_bodies` (so `lower_view_call` can inline them at
/// call sites), and register materialized view signatures into
/// `ctx.view_signatures` (so the type checker can resolve
/// `BuiltinId::ViewCall { view }` shapes). Task 5.5c.
fn populate_view_bodies_and_signatures(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) {
    for (i, view) in comp.views.iter().enumerate() {
        let view_id = ViewId(i as u32);
        match (&view.kind, &view.body) {
            (ViewKind::Lazy, ViewBodyIR::Expr(body)) => {
                let snapshot = super::expr::LazyViewSnapshot {
                    param_locals: view.params.iter().map(|p| p.local).collect(),
                    body: body.clone(),
                };
                if ctx.register_lazy_view_body(view_id, snapshot).is_some() {
                    diagnostics.push(LoweringError::DuplicateLazyViewBodyRegistration {
                        view: view_id,
                        span: view.span,
                    });
                }
            }
            (ViewKind::Materialized(_), ViewBodyIR::Fold { .. }) => {
                let arg_tys: Vec<crate::cg::expr::CgTy> = view
                    .params
                    .iter()
                    .map(|p| super::expr::ir_type_to_cg_ty(&p.ty))
                    .collect();
                let result_ty = super::expr::ir_type_to_cg_ty(&view.return_ty);
                ctx.register_view_signature(view_id, arg_tys, result_ty);
            }
            // Kind/body mismatches are reported at lower_view time
            // with a structural diagnostic; the registry walk skips
            // them so the diagnostic isn't doubled.
            (ViewKind::Lazy, ViewBodyIR::Fold { .. })
            | (ViewKind::Materialized(_), ViewBodyIR::Expr(_)) => {}
        }
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
        let spatial_kind = mask_spatial_kind(mask, comp, ctx);
        if let Err(e) = lower_mask(mask_id, spatial_kind, mask, ctx) {
            diagnostics.push(e);
        }
    }
}

/// Lower a per-pair filter expression to a `CgExprId` with the
/// per-pair candidate binder (`target` / `candidate` LocalRef →
/// `PerPairCandidateId`) active for the duration of the lowering.
/// Mirrors the `target_local` flag toggle in
/// [`super::mask::lower_mask`] — the flag is restored before returning
/// so a recursive lowering can't leak the binding upward.
///
/// Returns the lowered `CgExprId`. Type-validation that the filter
/// is `Bool` happens later in `cg::well_formed` (the TypeCheckCtx
/// wiring lives there); this helper is purely the lowering shim.
///
/// Phase 7 Task 5 wired this into [`mask_spatial_kind`]: the
/// `from spatial.<name>(args)` mask source resolves the named
/// `spatial_query` decl, substitutes the call-site value-args via
/// [`walk_substitute`], then lowers the filter expression here.
fn lower_filter_for_mask(
    expr: &IrExprNode,
    ctx: &mut LoweringCtx<'_>,
) -> Result<CgExprId, LoweringError> {
    let prev = ctx.target_local;
    ctx.target_local = true;
    let result = lower_expr(expr, ctx);
    ctx.target_local = prev;
    result
}

/// Substitute call-site value-args into a `spatial_query` filter
/// expression. Walks the IR tree, replacing each
/// `IrExpr::Local(LocalRef(i), _)` for `i >= 2` (value-args, since
/// `LocalRef(0) = self` and `LocalRef(1) = candidate`) with the
/// corresponding call-site argument expression.
///
/// `self` (`LocalRef(0)`) and `candidate` (`LocalRef(1)`) are NOT
/// substituted. They are bound by the lowering layer at filter-lower
/// time: [`lower_filter_for_mask`] sets `ctx.target_local = true`,
/// so `target` / `candidate` LocalRef reads resolve to
/// `CgExpr::PerPairCandidateId` and `self` resolves to
/// `CgExpr::AgentSelfId` via the standard lowering path.
///
/// Per Phase 7's call-site arity convention (Task 4 Adjustment A),
/// the call site at `from spatial.<name>(args)` passes
/// `(self, value_args...)` — `candidate` is not a call-site arg.
/// So `value_args` here corresponds to LocalRef(2..), indexed
/// sequentially. (Today every wolf-sim spatial_query has zero
/// value-args, so this loop is a no-op; the substitution machinery
/// is wired so that future `spatial.nearby_in_radius(self, radius)`
/// uses just work.)
///
/// Walk is fully exhaustive over [`IrExpr`]: every variant carrying
/// nested `IrExprNode` recurses; leaf variants clone. No new binders
/// are introduced inside spatial_query filters today (the resolver
/// rejects `for` / `let` / `match` inside the filter expression),
/// but the recursive structure tolerates them by passing
/// `value_args` straight through — a stray binder would just be
/// re-checked against the same value-arg slice.
fn walk_substitute(node: &IrExprNode, value_args: &[IrCallArg]) -> IrExprNode {
    let new_kind = match &node.kind {
        IrExpr::Local(local_ref, name) => {
            // Substitute LocalRef(2..) with the corresponding value
            // arg. LocalRef(0)=self / LocalRef(1)=candidate are not
            // substituted — they fall through and get resolved by
            // `lower_expr` (self → AgentSelfId; candidate → PerPair
            // when `ctx.target_local` is set).
            if local_ref.0 >= 2 {
                let idx = (local_ref.0 - 2) as usize;
                if idx < value_args.len() {
                    return value_args[idx].value.clone();
                }
            }
            IrExpr::Local(*local_ref, name.clone())
        }
        IrExpr::Field { base, field_name, field } => IrExpr::Field {
            base: Box::new(walk_substitute(base, value_args)),
            field_name: field_name.clone(),
            field: field.clone(),
        },
        IrExpr::Index(lhs, rhs) => IrExpr::Index(
            Box::new(walk_substitute(lhs, value_args)),
            Box::new(walk_substitute(rhs, value_args)),
        ),
        IrExpr::Binary(op, lhs, rhs) => IrExpr::Binary(
            *op,
            Box::new(walk_substitute(lhs, value_args)),
            Box::new(walk_substitute(rhs, value_args)),
        ),
        IrExpr::Unary(op, inner) => {
            IrExpr::Unary(*op, Box::new(walk_substitute(inner, value_args)))
        }
        IrExpr::In(lhs, rhs) => IrExpr::In(
            Box::new(walk_substitute(lhs, value_args)),
            Box::new(walk_substitute(rhs, value_args)),
        ),
        IrExpr::Contains(lhs, rhs) => IrExpr::Contains(
            Box::new(walk_substitute(lhs, value_args)),
            Box::new(walk_substitute(rhs, value_args)),
        ),
        IrExpr::Quantifier { kind, binder, binder_name, iter, body } => IrExpr::Quantifier {
            kind: *kind,
            binder: *binder,
            binder_name: binder_name.clone(),
            iter: Box::new(walk_substitute(iter, value_args)),
            body: Box::new(walk_substitute(body, value_args)),
        },
        IrExpr::Fold { kind, binder, binder_name, iter, body } => IrExpr::Fold {
            kind: kind.clone(),
            binder: *binder,
            binder_name: binder_name.clone(),
            iter: iter.as_ref().map(|i| Box::new(walk_substitute(i, value_args))),
            body: Box::new(walk_substitute(body, value_args)),
        },
        IrExpr::List(items) => IrExpr::List(
            items.iter().map(|i| walk_substitute(i, value_args)).collect(),
        ),
        IrExpr::Tuple(items) => IrExpr::Tuple(
            items.iter().map(|i| walk_substitute(i, value_args)).collect(),
        ),
        IrExpr::ViewCall(vr, args) => IrExpr::ViewCall(
            *vr,
            args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: walk_substitute(&a.value, value_args),
                    span: a.span,
                })
                .collect(),
        ),
        IrExpr::VerbCall(vr, args) => IrExpr::VerbCall(
            *vr,
            args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: walk_substitute(&a.value, value_args),
                    span: a.span,
                })
                .collect(),
        ),
        IrExpr::BuiltinCall(b, args) => IrExpr::BuiltinCall(
            *b,
            args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: walk_substitute(&a.value, value_args),
                    span: a.span,
                })
                .collect(),
        ),
        IrExpr::UnresolvedCall(name, args) => IrExpr::UnresolvedCall(
            name.clone(),
            args.iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: walk_substitute(&a.value, value_args),
                    span: a.span,
                })
                .collect(),
        ),
        IrExpr::NamespaceCall { ns, method, args } => IrExpr::NamespaceCall {
            ns: *ns,
            method: method.clone(),
            args: args
                .iter()
                .map(|a| IrCallArg {
                    name: a.name.clone(),
                    value: walk_substitute(&a.value, value_args),
                    span: a.span,
                })
                .collect(),
        },
        IrExpr::StructLit { name, ctor, fields } => IrExpr::StructLit {
            name: name.clone(),
            ctor: ctor.clone(),
            fields: fields
                .iter()
                .map(|f| dsl_ast::ir::IrFieldInit {
                    name: f.name.clone(),
                    value: walk_substitute(&f.value, value_args),
                    span: f.span,
                })
                .collect(),
        },
        IrExpr::Ctor { name, ctor, args } => IrExpr::Ctor {
            name: name.clone(),
            ctor: ctor.clone(),
            args: args.iter().map(|a| walk_substitute(a, value_args)).collect(),
        },
        IrExpr::Match { scrutinee, arms } => IrExpr::Match {
            scrutinee: Box::new(walk_substitute(scrutinee, value_args)),
            arms: arms
                .iter()
                .map(|arm| dsl_ast::ir::IrMatchArm {
                    pattern: arm.pattern.clone(),
                    body: walk_substitute(&arm.body, value_args),
                    span: arm.span,
                })
                .collect(),
        },
        IrExpr::If { cond, then_expr, else_expr } => IrExpr::If {
            cond: Box::new(walk_substitute(cond, value_args)),
            then_expr: Box::new(walk_substitute(then_expr, value_args)),
            else_expr: else_expr
                .as_ref()
                .map(|e| Box::new(walk_substitute(e, value_args))),
        },
        IrExpr::PerUnit { expr, delta } => IrExpr::PerUnit {
            expr: Box::new(walk_substitute(expr, value_args)),
            delta: Box::new(walk_substitute(delta, value_args)),
        },
        IrExpr::AbilityOnCooldown(inner) => {
            IrExpr::AbilityOnCooldown(Box::new(walk_substitute(inner, value_args)))
        }
        IrExpr::BeliefsAccessor { observer, target, field } => IrExpr::BeliefsAccessor {
            observer: Box::new(walk_substitute(observer, value_args)),
            target: Box::new(walk_substitute(target, value_args)),
            field: field.clone(),
        },
        IrExpr::BeliefsConfidence { observer, target } => IrExpr::BeliefsConfidence {
            observer: Box::new(walk_substitute(observer, value_args)),
            target: Box::new(walk_substitute(target, value_args)),
        },
        IrExpr::BeliefsView { observer, view_name } => IrExpr::BeliefsView {
            observer: Box::new(walk_substitute(observer, value_args)),
            view_name: view_name.clone(),
        },
        // Leaves carrying no nested `IrExprNode` — clone directly.
        IrExpr::LitBool(_)
        | IrExpr::LitInt(_)
        | IrExpr::LitFloat(_)
        | IrExpr::LitString(_)
        | IrExpr::Event(_)
        | IrExpr::Entity(_)
        | IrExpr::View(_)
        | IrExpr::Verb(_)
        | IrExpr::Namespace(_)
        | IrExpr::NamespaceField { .. }
        | IrExpr::EnumVariant { .. }
        | IrExpr::AbilityTag { .. }
        | IrExpr::AbilityHint
        | IrExpr::AbilityHintLit(_)
        | IrExpr::AbilityRange
        | IrExpr::Raw(_) => node.kind.clone(),
    };
    IrExprNode {
        kind: new_kind,
        span: node.span,
    }
}

/// Pick the [`SpatialQueryKind`] for a mask. Three routing branches:
///
/// 1. **Phase 7 — `from spatial.<name>(args)`** (the new
///    general-spatial-queries surface). Look up the registered
///    `spatial_query <name>(self, candidate, …)` decl in
///    `comp.spatial_queries`, substitute the call-site value-args
///    into the filter via [`walk_substitute`], lower with
///    `target_local = true` via [`lower_filter_for_mask`], and wrap
///    the resulting `CgExprId` in
///    [`SpatialQueryKind::FilteredWalk`].
/// 2. **Legacy — `from query.nearby_agents(...)`** (pre-Phase-7
///    wolf-sim convention). Routes to
///    [`SpatialQueryKind::EngagementQuery`] when the predicate
///    references engagement-flavoured access patterns
///    (`agents.is_hostile_to`, `agents.engaged_with`, any
///    `IrExpr::ViewCall` — conservative widening), otherwise
///    [`SpatialQueryKind::KinQuery`]. Phase 7 Task 6 will retire
///    this branch once all wolf-sim masks have migrated; until
///    then, the heuristic stays for backwards compat.
/// 3. **No `from` clause** — returns `None` (resolves to
///    [`DispatchShape::PerAgent`]).
fn mask_spatial_kind(
    mask: &MaskIR,
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
) -> Option<SpatialQueryKind> {
    let source = mask.candidate_source.as_ref()?;
    match &source.kind {
        IrExpr::NamespaceCall {
            ns: NamespaceId::Spatial,
            method,
            args,
        } => {
            let decl = comp
                .spatial_queries
                .iter()
                .find(|s| &s.name == method)?;
            let filter_with_args = walk_substitute(&decl.filter, args);
            let filter_id = lower_filter_for_mask(&filter_with_args, ctx).ok()?;
            Some(SpatialQueryKind::FilteredWalk { filter: filter_id })
        }
        // No other from-clause shapes recognised. Phase 7 dropped the
        // legacy `query.nearby_agents` heuristic; only `spatial.<name>`
        // (registered `spatial_query` decls) is supported.
        _ => None,
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

/// Synthesize the Movement op: a per-agent
/// [`ComputeOpKind::PhysicsRule`] (with `on_event = None`) that
/// consumes scoring's chosen action+target output and updates each
/// agent's position. The op carries an empty body; its WGSL emit is
/// driven by the `MOVEMENT_BODY` template in `cg/emit/kernel.rs`,
/// which reads from the structural buffer bindings derived from the
/// op's `reads` / `writes` signature.
///
/// Allocates the next free [`PhysicsRuleId`] (one past the highest
/// rule id allocated by `lower_all_physics`) and interns the rule
/// name `"movement"` on the program's interner. Surfaces interner
/// duplicate-name conflicts as
/// [`LoweringError::BuilderRejected`].
///
/// Movement is `replayable` because the `AgentMoved` events it
/// emits feed into the deterministic ring (the engagement_on_move
/// physics rule and its descendants fold them into trace state).
/// `AgentFled` shares the same ring; both are listed in
/// `assets/sim/events.sim` as the canonical replayable surface.
///
/// # Phase 6 Task 3 contract
///
/// The op's `reads` set names every binding the WGSL body touches:
/// - `ScoringOutput` (action + target lookup).
/// - `AgentField { Pos, Self_ }` and `AgentField { Pos, Target(_) }`
///   (self pos + target pos, the latter resolves through the
///   structural placeholder until per-thread target resolution
///   lands).
/// - `AgentField { Alive, Self_ }` (dead-agent skip predicate).
/// - `SimCfgBuffer` (read for tick + move_speed).
///
/// The `writes` set names:
/// - `AgentField { Pos, Self_ }` (the per-agent position update).
/// - `EventRing { ring: 0, kind: Append }` (AgentMoved /
///   AgentFled emits land in the shared event ring).
fn synthesize_movement_op(ctx: &mut LoweringCtx<'_>) -> Result<(), LoweringError> {
    // Allocate the next PhysicsRuleId past whatever lower_all_physics
    // already used. The interner's `physics_rules` map size is the
    // high water mark — every PhysicsRuleId ever interned is keyed
    // there.
    let next_rule_id = PhysicsRuleId(
        ctx.builder.program().interner.physics_rules.len() as u32,
    );
    ctx.builder
        .intern_physics_rule_name(next_rule_id, "movement".to_string())
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: dsl_ast::ast::Span::dummy(),
        })?;

    // Empty body: the WGSL emit short-circuits PerAgent PhysicsRule
    // to a hand-written `MOVEMENT_BODY` template, so no IR statements
    // are required for emit fidelity. The op's `reads` / `writes`
    // (recorded below via `record_read` / `record_write`) drive BGL
    // synthesis.
    let body = ctx
        .builder
        .add_stmt_list(CgStmtList::new(vec![]))
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: dsl_ast::ast::Span::dummy(),
        })?;

    let kind = ComputeOpKind::PhysicsRule {
        rule: next_rule_id,
        on_event: None,
        body,
        replayable: ReplayabilityFlag::Replayable,
    };
    let op_id = ctx
        .builder
        .add_op(kind, DispatchShape::PerAgent, dsl_ast::ast::Span::dummy())
        .map_err(|e| LoweringError::BuilderRejected {
            error: e,
            span: dsl_ast::ast::Span::dummy(),
        })?;

    // Inject the structural reads + writes the WGSL body touches.
    // The auto-walker can't see them (the body is empty) so this is
    // the canonical seam — same shape as `wire_source_ring_reads`
    // for PerEvent rules.
    //
    // **Runtime aliasing constraint** (Phase 6 Task 5 / 6 territory):
    // every `AgentField { … }` handle structurally aliases onto the
    // single resident `agents` buffer. wgpu rejects two bindings to
    // the same buffer where one is read_only and the other is
    // read_write in the same compute pass. Movement records ONLY
    // `Pos` (a single read_write alias) here — the `Alive`-skip
    // predicate the body needs is bounded by `cfg.agent_cap` at the
    // PerAgent preamble layer (`if (agent_id >= cfg.agent_cap)
    // { return; }`); the dead-agent case is one extra branch that
    // doesn't change the buffer alias surface. This minimizes the
    // alias footprint; future Apply-actions (Phase 6 Task 4) deals
    // with the multi-AgentField alias issue at its layer.
    //
    // **EventRing append**: the `AgentMoved` / `AgentFled` events
    // Movement should emit are NOT recorded here yet — adding a
    // third binding (`event_ring`, atomic-rw) plus the ring-cycle
    // edge it implies isn't gated cleanly by today's runtime
    // cascade pipeline. The emit lands when Phase 6 Task 4
    // (Apply-actions chain) ports the event-emit layer into CG.
    let ops = ctx.builder.ops_mut();
    let op = &mut ops[op_id.0 as usize];
    op.record_read(DataHandle::ScoringOutput);
    op.record_read(DataHandle::SimCfgBuffer);
    op.record_write(DataHandle::AgentField {
        field: AgentFieldId::Pos,
        target: AgentRef::Self_,
    });

    Ok(())
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
    let mut needs_build_hash = false;
    for op in &prog.ops {
        if let DispatchShape::PerPair {
            source: PerPairSource::SpatialQuery(kind),
        } = op.shape
        {
            consumers.insert(kind);
            needs_build_hash = true;
        }
        // ForEachNeighbor consumers: any op (typically a per-agent
        // PhysicsRule) that surfaces a `SpatialStorage` read in its
        // dependency walk needs the spatial grid populated. The
        // ForEachNeighbor stmt's body-walk
        // (`collect_list_dependencies`) pushes
        // `DataHandle::SpatialStorage { GridCells/GridOffsets }` as
        // structural reads, so we detect them here regardless of the
        // op's dispatch shape.
        if op.reads.iter().any(|h| matches!(
            h,
            crate::cg::data_handle::DataHandle::SpatialStorage { .. }
        )) {
            needs_build_hash = true;
        }
    }

    if !needs_build_hash {
        return Vec::new();
    }

    // Real counting sort: schedule the three phases in dependency
    // order before any consumer. The bounded `BuildHash` legacy
    // variant is no longer scheduled — the new three-phase build
    // produces an uncapped per-cell layout that the tiled-MoveBoid
    // emit consumes via `spatial_grid_starts[c..c+1]` slicing.
    // Pre-existing FilteredWalk consumers (the wolf-sim mask flow,
    // currently dormant in this fixture) were also wired against
    // the bounded layout; if a fixture re-enables them, they need
    // to be ported to the new starts/cells layout too.
    let mut kinds = Vec::with_capacity(consumers.len() + 3);
    kinds.push(SpatialQueryKind::BuildHashCount);
    kinds.push(SpatialQueryKind::BuildHashScan);
    kinds.push(SpatialQueryKind::BuildHashScatter);
    for k in consumers {
        kinds.push(k);
    }
    kinds
}

// ---------------------------------------------------------------------------
// Phase 4 helpers — ring-edge wiring (pre-gate)
// ---------------------------------------------------------------------------

/// For each [`ComputeOp`] in `ops` whose dispatch shape is
/// [`DispatchShape::PerEvent { source_ring }`], record an
/// [`EventRingAccess::Read`] read on `source_ring`.
///
/// Without this, the well_formed cycle detector would see an
/// asymmetric event-ring graph (the dispatch carries the ring
/// identity but the reads list does not), missing producer/consumer
/// cycles between physics rules and view folds. The plan
/// amendment makes this a hard obligation on the driver.
///
/// Operates on a `&mut [ComputeOp]` rather than `&mut CgProgram` so
/// the driver can call it on the in-progress builder via
/// [`CgProgramBuilder::ops_mut`] before the cycle gate snapshot.
fn wire_source_ring_reads(ops: &mut [ComputeOp]) {
    for op in ops.iter_mut() {
        if let DispatchShape::PerEvent { source_ring } = op.shape {
            op.record_read(DataHandle::EventRing {
                ring: source_ring,
                kind: EventRingAccess::Read,
            });
        }
    }
}

/// Walk every user op's body's statement list and collect, per op
/// index, the set of destination [`EventRingId`]s every reachable
/// [`CgStmt::Emit`] resolves to. The driver's allocation rule pairs
/// `EventKindId(i)` with `EventRingId(i)`, so the walker can
/// translate each `Emit { event: EventKindId(i), .. }` directly.
///
/// Returns `(op_index, dest_ring)` pairs — duplicates are preserved
/// (an op that emits twice into the same ring records two entries;
/// downstream `record_write` consumers tolerate duplicates the same
/// way the auto-walker does for repeated `Assign`s).
///
/// Two-phase shape (collect-then-apply) avoids holding a mutable
/// borrow on the op list while traversing the (immutable) statement
/// arenas. See [`apply_emit_destination_rings`] for the application
/// half.
fn collect_emit_destination_rings(prog: &CgProgram) -> Vec<(usize, EventRingId)> {
    let mut out: Vec<(usize, EventRingId)> = Vec::new();
    for (op_index, op) in prog.ops.iter().enumerate() {
        let body_list = body_list_for_op_kind(&op.kind);
        let Some(list_id) = body_list else { continue };
        let mut emits: Vec<EventKindId> = Vec::new();
        collect_emits_in_list(list_id, prog, &mut emits);
        for _kind in emits {
            // Iter-2 unification: every event kind shares the single
            // `EventRingId(0)` ring (named `batch_events`). Pre-iter-2
            // this allocated `EventRingId(kind.0)` per-kind, but the
            // runtime has only one ring buffer; the per-kind ring ids
            // produced bindings like `event_ring_37` that didn't match
            // the unified `event_ring_0` source binding.
            //
            // See `populate_event_kinds` — `shared_ring = EventRingId(0)`.
            out.push((op_index, EventRingId(0)));
        }
    }
    out
}

/// Apply the (op_index, dest_ring) pairs collected by
/// [`collect_emit_destination_rings`] to `ops` via
/// [`ComputeOp::record_write`]. Pairs naming an op index past the
/// slice's length are silently dropped — the caller built the pairs
/// from a snapshot of the same builder, so this should never trip
/// in practice.
fn apply_emit_destination_rings(ops: &mut [ComputeOp], pairs: &[(usize, EventRingId)]) {
    for &(op_index, ring) in pairs {
        if let Some(op) = ops.get_mut(op_index) {
            op.record_write(DataHandle::EventRing {
                ring,
                kind: EventRingAccess::Append,
            });
        }
    }
}

/// Pick the body [`CgStmtListId`] for ops whose kind carries one;
/// `None` for kinds that don't have a stmt-list body.
///
/// Listed exhaustively rather than with a `_ =>` fallthrough so a
/// future op kind that introduces a new body shape forces an
/// explicit decision here instead of silently bypassing the Emit
/// walker.
fn body_list_for_op_kind(kind: &ComputeOpKind) -> Option<CgStmtListId> {
    match kind {
        ComputeOpKind::PhysicsRule { body, .. } => Some(*body),
        ComputeOpKind::ViewFold { body, .. } => Some(*body),
        ComputeOpKind::MaskPredicate { .. } => None,
        ComputeOpKind::ScoringArgmax { .. } => None,
        ComputeOpKind::SpatialQuery { .. } => None,
        ComputeOpKind::Plumbing { .. } => None,
    }
}

/// Recursively collect every [`CgStmt::Emit`]'s [`EventKindId`] from
/// the statement list named by `list_id`, descending through `If`
/// arms (both `then` and `else_`) and `Match` arm bodies.
///
/// Listed exhaustively over [`CgStmt`] variants — no `_ =>` arm —
/// so a future statement variant that introduces an emit-bearing
/// body forces an explicit case here.
fn collect_emits_in_list(list_id: CgStmtListId, prog: &CgProgram, out: &mut Vec<EventKindId>) {
    let Some(list) = prog.stmt_lists.get(list_id.0 as usize) else {
        return;
    };
    for &stmt_id in &list.stmts {
        let Some(stmt) = prog.stmts.get(stmt_id.0 as usize) else {
            continue;
        };
        match stmt {
            CgStmt::Emit { event, .. } => out.push(*event),
            CgStmt::If { then, else_, .. } => {
                collect_emits_in_list(*then, prog, out);
                if let Some(else_list) = else_ {
                    collect_emits_in_list(*else_list, prog, out);
                }
            }
            CgStmt::Match { arms, .. } => {
                for arm in arms {
                    collect_emits_in_list(arm.body, prog, out);
                }
            }
            CgStmt::Assign { .. }
            | CgStmt::Let { .. }
            | CgStmt::ForEachAgent { .. }
            | CgStmt::ForEachNeighbor { .. } => {}
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
                    on_event: Some(EventKindId(0)),
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
        wire_source_ring_reads(&mut prog.ops);

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

    /// `populate_variants_from_enums` surfaces a typed
    /// `DuplicateVariantInRegistry` diagnostic when two enums declare
    /// the same source-level variant name. Last-write-wins semantics
    /// remain in place; the diagnostic exists so callers (or future
    /// tests) can refuse the program without scanning the registry by
    /// hand.
    #[test]
    fn populate_variants_from_enums_flags_duplicate_variant() {
        use dsl_ast::ast::Span;
        use dsl_ast::ir::EnumIR;

        let mut comp = Compilation::default();
        // Two enums, both declaring a `Damage` variant. The second
        // occurrence collides with the first.
        comp.enums.push(EnumIR {
            name: "EffectOpA".to_string(),
            variants: vec!["Damage".to_string(), "Heal".to_string()],
            annotations: Vec::new(),
            span: Span::dummy(),
        });
        comp.enums.push(EnumIR {
            name: "EffectOpB".to_string(),
            variants: vec!["Damage".to_string()], // collides with EnumA's Damage
            annotations: Vec::new(),
            span: Span::dummy(),
        });

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let mut diagnostics: Vec<LoweringError> = Vec::new();
        populate_variants_from_enums(&comp, &mut ctx, &mut diagnostics);

        // Exactly one duplicate diagnostic for `Damage` — the second
        // registration. `Heal` was unique; the first `Damage`
        // registered without conflict.
        let dup_count = diagnostics
            .iter()
            .filter(|d| matches!(d, LoweringError::DuplicateVariantInRegistry { name, .. } if name == "Damage"))
            .count();
        assert_eq!(
            dup_count, 1,
            "expected one DuplicateVariantInRegistry for `Damage`; got diagnostics: {diagnostics:?}"
        );
    }

    // ---- Task 5.5c, Patch 1: populate_config_consts -----------------

    /// Two ConfigIR blocks → 4 entries with ids 0..3 in source order.
    #[test]
    fn populate_config_consts_allocates_per_block_field_in_source_order() {
        use dsl_ast::ast::{ConfigDefault, Span};
        use dsl_ast::ir::{ConfigFieldIR, ConfigIR, IrType};

        let mut comp = Compilation::default();
        comp.configs.push(ConfigIR {
            name: "combat".to_string(),
            fields: vec![
                ConfigFieldIR {
                    name: "attack_range".to_string(),
                    ty: IrType::F32,
                    default: ConfigDefault::Float(1.0),
                    span: Span::dummy(),
                },
                ConfigFieldIR {
                    name: "aggro_range".to_string(),
                    ty: IrType::F32,
                    default: ConfigDefault::Float(2.0),
                    span: Span::dummy(),
                },
            ],
            annotations: Vec::new(),
            span: Span::dummy(),
        });
        comp.configs.push(ConfigIR {
            name: "movement".to_string(),
            fields: vec![ConfigFieldIR {
                name: "move_speed_mps".to_string(),
                ty: IrType::F32,
                default: ConfigDefault::Float(3.0),
                span: Span::dummy(),
            }],
            annotations: Vec::new(),
            span: Span::dummy(),
        });

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let mut diagnostics: Vec<LoweringError> = Vec::new();
        populate_config_consts(&comp, &mut ctx, &mut diagnostics);

        assert!(diagnostics.is_empty(), "no diagnostics: {diagnostics:?}");
        assert_eq!(ctx.config_const_ids.len(), 3);
        assert_eq!(
            ctx.config_const_ids
                .get(&(NamespaceId::Config, "combat.attack_range".to_string())),
            Some(&ConfigConstId(0))
        );
        assert_eq!(
            ctx.config_const_ids
                .get(&(NamespaceId::Config, "combat.aggro_range".to_string())),
            Some(&ConfigConstId(1))
        );
        assert_eq!(
            ctx.config_const_ids
                .get(&(NamespaceId::Config, "movement.move_speed_mps".to_string())),
            Some(&ConfigConstId(2))
        );
    }

    /// Pre-seeding the registry surfaces a typed
    /// DuplicateConfigConstInRegistry diagnostic.
    #[test]
    fn populate_config_consts_flags_duplicate_key() {
        use dsl_ast::ast::{ConfigDefault, Span};
        use dsl_ast::ir::{ConfigFieldIR, ConfigIR, IrType};

        let mut comp = Compilation::default();
        comp.configs.push(ConfigIR {
            name: "combat".to_string(),
            fields: vec![ConfigFieldIR {
                name: "attack_range".to_string(),
                ty: IrType::F32,
                default: ConfigDefault::Float(1.0),
                span: Span::dummy(),
            }],
            annotations: Vec::new(),
            span: Span::dummy(),
        });

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // Pre-seed a colliding entry.
        ctx.register_config_const(
            NamespaceId::Config,
            "combat.attack_range".to_string(),
            ConfigConstId(99),
        );

        let mut diagnostics: Vec<LoweringError> = Vec::new();
        populate_config_consts(&comp, &mut ctx, &mut diagnostics);

        let dup = diagnostics.iter().find_map(|d| match d {
            LoweringError::DuplicateConfigConstInRegistry {
                key,
                prior_id,
                new_id,
            } => Some((key.clone(), prior_id.0, new_id.0)),
            _ => None,
        });
        assert_eq!(
            dup,
            Some(("combat.attack_range".to_string(), 99, 0)),
            "expected DuplicateConfigConstInRegistry; diagnostics: {diagnostics:?}"
        );
    }

    // ---- Task 5.5c, Patch 3: mask_spatial_kind routing --------------

    fn mk_mask(predicate_kind: IrExpr, has_from: bool) -> MaskIR {
        use dsl_ast::ast::Span;
        use dsl_ast::ir::{IrActionHead, IrActionHeadShape, IrExprNode};
        MaskIR {
            head: IrActionHead {
                name: "M".to_string(),
                shape: IrActionHeadShape::None,
                span: Span::dummy(),
            },
            predicate: IrExprNode {
                kind: predicate_kind,
                span: Span::dummy(),
            },
            candidate_source: if has_from {
                Some(IrExprNode {
                    kind: IrExpr::NamespaceCall {
                        ns: NamespaceId::Query,
                        method: "nearby_agents".to_string(),
                        args: Vec::new(),
                    },
                    span: Span::dummy(),
                })
            } else {
                None
            },
            annotations: Vec::new(),
            span: Span::dummy(),
        }
    }

    /// Helper: build an empty Compilation + LoweringCtx for the
    /// `mask_spatial_kind` tests. The legacy heuristic branches don't
    /// touch `comp.spatial_queries` or `ctx`, so a default builder is
    /// safe; the new `from spatial.<name>` branch needs a registered
    /// decl which the legacy-routing tests intentionally don't exercise.
    fn mk_test_ctx() -> (Compilation, CgProgramBuilder) {
        (Compilation::default(), CgProgramBuilder::new())
    }





    #[test]
    fn mask_spatial_kind_returns_none_when_no_candidate_source() {
        let mask = mk_mask(IrExpr::LitBool(true), false);
        let (comp, mut builder) = mk_test_ctx();
        let mut ctx = LoweringCtx::new(&mut builder);
        assert_eq!(mask_spatial_kind(&mask, &comp, &mut ctx), None);
    }


    /// `populate_views` surfaces a typed `DuplicateViewInRegistry`
    /// diagnostic if `register_view` ever observes the same AST view
    /// ref twice. Driver allocates `ViewId(i)` in source order, so a
    /// real-world collision would be a driver-side defect — the
    /// typed surface lets a test assert the contract.
    #[test]
    fn populate_views_flags_duplicate_view() {
        use dsl_ast::ir::ViewRef;

        // We can't easily make `populate_views` itself emit a
        // duplicate (it iterates `0..comp.views.len()` and assigns
        // unique ids), but the typed registry is the same one used
        // by `register_view`. Pre-register a colliding entry to
        // exercise the diagnostic path; then run `populate_views`
        // on a one-view Compilation and assert the collision is
        // surfaced.
        let mut comp = Compilation::default();
        comp.views.push(dsl_ast::ir::ViewIR {
            name: "v0".to_string(),
            kind: dsl_ast::ir::ViewKind::Lazy,
            params: Vec::new(),
            return_ty: dsl_ast::ir::IrType::F32,
            body: dsl_ast::ir::ViewBodyIR::Expr(dsl_ast::ir::IrExprNode {
                kind: dsl_ast::ir::IrExpr::LitFloat(0.0),
                span: dsl_ast::ast::Span::dummy(),
            }),
            annotations: Vec::new(),
            decay: None,
            span: dsl_ast::ast::Span::dummy(),
        });

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        // Pre-seed the registry with a stale entry for ViewRef(0)
        // so the driver's call collides.
        let prior_id = ViewId(99);
        let prior = ctx.register_view(ViewRef(0), prior_id);
        assert!(prior.is_none(), "registry should be empty before pre-seed");

        let mut diagnostics: Vec<LoweringError> = Vec::new();
        populate_views(&comp, &mut ctx, &mut diagnostics);

        let dup = diagnostics.iter().find_map(|d| match d {
            LoweringError::DuplicateViewInRegistry {
                ast_ref,
                prior_id,
                new_id,
            } => Some((ast_ref.0, prior_id.0, new_id.0)),
            _ => None,
        });
        assert_eq!(
            dup,
            Some((0, 99, 0)),
            "expected DuplicateViewInRegistry(ast_ref=0, prior=99, new=0); got diagnostics: {diagnostics:?}"
        );
    }

    // ---- lower_filter_for_mask helper -------------------------------------

    #[test]
    fn lower_filter_for_mask_binds_target_to_per_pair_candidate() {
        use crate::cg::expr::CgExpr;
        use crate::cg::program::CgProgramBuilder;
        use dsl_ast::ir::{IrExpr, IrExprNode, LocalRef};

        // Filter expression: bare `target` local. With target_local=true,
        // this should lower to CgExpr::PerPairCandidateId.
        let target_local = IrExprNode {
            kind: IrExpr::Local(LocalRef(0), "target".to_string()),
            span: dsl_ast::ast::Span::dummy(),
        };

        let mut builder = CgProgramBuilder::new();
        let filter_id = {
            let mut ctx = LoweringCtx::new(&mut builder);
            let id = lower_filter_for_mask(&target_local, &mut ctx)
                .expect("lowers target to PerPairCandidateId");
            // Helper must restore target_local to false on exit — verify
            // while ctx is still in scope (before it drops / builder is freed).
            assert!(
                !ctx.target_local,
                "target_local should be restored to false after lower_filter_for_mask"
            );
            id
        };

        let prog = builder.finish();
        let node = &prog.exprs[filter_id.0 as usize];
        match node {
            CgExpr::PerPairCandidateId => {} // expected
            other => panic!("expected PerPairCandidateId, got {other:?}"),
        }
    }

    #[test]
    fn lower_filter_for_mask_restores_target_local_on_lower_expr_failure() {
        use crate::cg::program::CgProgramBuilder;
        use dsl_ast::ir::{IrExpr, IrExprNode, LocalRef};

        // `IrExpr::Local` with an unrecognized name (not "self" or "target")
        // and no let-binding → `lower_bare_local` returns
        // `LoweringError::UnsupportedLocalBinding`. This exercises the
        // error path of `lower_filter_for_mask` without relying on any
        // arena or upstream-resolver machinery.
        let bad_expr = IrExprNode {
            kind: IrExpr::Local(LocalRef(99), "undefined_local".to_string()),
            span: dsl_ast::ast::Span::dummy(),
        };

        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        assert!(!ctx.target_local, "precondition: target_local starts false");

        let result = lower_filter_for_mask(&bad_expr, &mut ctx);
        assert!(result.is_err(), "lowering of undefined local should error");
        // Flag must be restored to its prior value (false) even on the
        // error path — the same save/restore contract as lower_mask.
        assert!(!ctx.target_local, "target_local must be restored even on error");
    }
}
