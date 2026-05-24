//! Inner-expression and inner-statement WGSL emission.
//!
//! Walks a [`CgExpr`] / [`CgStmt`] tree and produces a WGSL source
//! fragment — never a complete kernel, never a binding declaration.
//! Composing fragments into kernel bodies is Task 4.2's job; assembling
//! the kernel module is Task 4.3.
//!
//! # Task 5.3 (ViewFold body parity) note
//!
//! Task 5.3's ViewFold-specific WGSL body composition is plumbed
//! through [`super::kernel::build_view_fold_wgsl_body`], which calls
//! [`lower_cg_stmt_list_to_wgsl`] (this module) on each handler's
//! [`crate::cg::stmt::CgStmtList`] body. The inner-expression and
//! inner-statement walks here are storage-hint-agnostic — the fold
//! body's `CgStmt::Assign { target: ViewStorage{view,slot}, value }`
//! lowers to a plain WGSL assignment, and any storage-hint-specific
//! update primitives (atomicAdd vs sort-and-write vs ring-append-modulo)
//! are wired by Task 5.5. The Task 5.3 cut surfaces the entry-point +
//! event-count gate around whatever Task 4.1 produces; per-storage-hint
//! body templates are deferred.
//!
//! # Limitations
//!
//! - **Naming strategy.** Today only [`HandleNamingStrategy::Structural`]
//!   is implemented. Each [`DataHandle`] prints as a deterministic
//!   identifier-shaped name (`agent_hp[agent_id]`, `view_3_primary`,
//!   `mask_2_bitmap`, …) — useful for snapshot tests and as a
//!   placeholder until BGL slot assignment lands. Task 4.2 will plug in
//!   a slot-aware strategy that emits the actual buffer access form
//!   (e.g. `agents.hp[gid.x]` or `view_3_primary[a]`).
//! - **`AgentRef::Target(expr_id)`.** A target reference is a per-thread
//!   runtime value: a `CgExprId` whose lowered WGSL produces the slot
//!   index into the agent SoA. The first `Read` / `Assign` of an
//!   `AgentField { target: Target(expr_id), … }` within a block emits
//!   `agent_<field>[target_expr_<N>]` AND queues a stmt-prefix
//!   `let target_expr_<N>: u32 = <lowered_target>;` via
//!   [`EmitCtx::pending_target_lets`]; subsequent reads in the same
//!   block reuse the binding without re-emitting (`bound_target_exprs`).
//!   The bound set is cloned + restored at every stmt-list boundary so
//!   inner-block bindings can't leak outward. Mirrors the existing
//!   `AgentRef::PerPairCandidate` pre-binding pattern.
//! - **Custom builtins.** [`BuiltinId::PlanarDistance`],
//!   [`BuiltinId::ZSeparation`], [`BuiltinId::SaturatingAdd`],
//!   `is_hostile`, `kin_count_within`, etc. are emitted as direct
//!   function calls (`planar_distance(a, b)`, `saturating_add(x, y)`).
//!   Task 4.3 wires the WGSL prelude that provides these helpers.
//! - **`Match` lowering.** Lowered as an `if`-chain over each arm's
//!   variant tag (`if (scrutinee_tag == VARIANT_<N>) { ... }`). WGSL
//!   does support `switch`, but the IR's variant ids are not yet
//!   resolved to compact case constants — `if`-chain is the honest
//!   placeholder until the prelude lands. Arm-binding locals
//!   (`MatchArmBinding::local`) are not yet referenced from arm bodies
//!   (the IR errors on local reads in expression lowering today).
//! - **Event emit shape.** The emit form here is a placeholder
//!   `emit_event_<N>(field0: ..., field1: ...);` — Task 4.2 wires the
//!   actual ring-append form once event-ring slot assignment is known.
//! - **Vec3 swizzles.** Writes to a `Vec3` field as a whole are
//!   supported; per-component writes are an emit-time concern not yet
//!   surfaced in the IR.
//!
//! # Reuse from prior layers
//!
//! [`crate::cg::CgExpr`], [`crate::cg::CgStmt`], [`DataHandle`],
//! [`crate::cg::BinaryOp`], [`crate::cg::UnaryOp`], [`BuiltinId`] are
//! consumed read-only — no IR shapes are added by Task 4.1. New
//! lowerings of those types extend the match arms here exhaustively
//! (no `_ =>` fallthroughs in production code).

use std::fmt;

use crate::cg::data_handle::{
    AgentFieldId, AgentFieldTy, AgentRef, AgentScratchKind, CgExprId, DataHandle, EventRingAccess,
    RngPurpose, SpatialStorageKind, ViewStorageSlot,
};
use crate::cg::expr::{BinaryOp, BuiltinId, CgExpr, CgTy, ExprArena, LitValue, NumericTy, UnaryOp};
use crate::cg::lower::driver::DebugWgslFlags;
use crate::cg::op::EventKindId;
use crate::cg::program::CgProgram;
use crate::cg::stmt::{
    CgMatchArm, CgStmt, CgStmtId, CgStmtListId, EventField, LocalId, MatchArmBinding, StmtArena,
    StmtListArena,
};

// ---------------------------------------------------------------------------
// EmitCtx
// ---------------------------------------------------------------------------

/// Strategy for naming a [`DataHandle`] when it appears as the bare
/// operand of a `Read` / `Assign`. Task 4.1 ships only the
/// [`Structural`] strategy; future tasks add a slot-aware variant.
///
/// [`Structural`]: HandleNamingStrategy::Structural
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum HandleNamingStrategy {
    /// Each handle prints as a deterministic identifier-shaped name
    /// (`agent_hp[agent_id]`, `view_3_primary`, `mask_2_bitmap`,
    /// `event_ring_5_read`, `rng_action`, …). The shape mirrors
    /// [`DataHandle::Display`]'s output but stripped down to
    /// WGSL-valid identifier characters (`[A-Za-z0-9_]` only). Used by
    /// snapshot tests and as the Task-4.1 placeholder before BGL slot
    /// assignment lands.
    Structural,
}

/// Context carried through the inner WGSL walks. Holds just the
/// program (for arena lookups) and the active handle naming strategy.
///
/// Constructed by Task 4.2's kernel-body composer; Task 4.1's tests
/// build it directly.
pub struct EmitCtx<'a> {
    /// The program — every [`CgExprId`] / [`CgStmtId`] / [`CgStmtListId`]
    /// is resolved against this program's arenas via the
    /// [`ExprArena`] / [`StmtArena`] / [`StmtListArena`] trait impls.
    pub prog: &'a CgProgram,
    /// Strategy for printing a [`DataHandle`] as a WGSL identifier.
    pub naming: HandleNamingStrategy,
    /// When set, every emit of `Read(AgentField { target: PerPairCandidate, .. })`
    /// for `Pos` / `Vel` redirects to the workgroup-local tile arrays
    /// (`tile_pos[<index>]` / `tile_vel[<index>]`) using the
    /// expression in this `Cell` as the index. Used by the tiled
    /// MoveBoid emit (DispatchShape::PerCell) to swap the inner-loop
    /// global-memory reads for shared-memory lookups. Cleared
    /// (`String::new()`) outside the inner walk so other emit
    /// contexts (cell-walk, agent-walk, etc.) keep their default
    /// `agent_<field>[per_pair_candidate]` indexing.
    ///
    /// Interior mutability (`std::cell::RefCell`) keeps the EmitCtx
    /// shareable behind `&` — the existing emit fns thread `&EmitCtx`
    /// throughout, and routing every signature through `&mut` would
    /// touch dozens of call sites for a pure emit-time scratch flag.
    pub tile_walk_index: std::cell::RefCell<Option<String>>,
    /// Name of the WGSL iteration variable for the currently-emitting
    /// ForEach* loop body. `Some("per_pair_candidate")` when emit is
    /// inside a `ForEachAgentBody` / `ForEachNeighborBody`; `None` at
    /// kernel-body top level. `CgExpr::Rng` reads this; when set, it
    /// routes through `per_agent_u32_with_extra(..., loop_var)` so
    /// each iteration draws from a distinct PCG stream (closes the
    /// forest_fire-style bug where `rng.action()` inside a spatial
    /// walk gave the same draw to every neighbour).
    ///
    /// Interior mutability so emit stays `&EmitCtx`. Save/restored
    /// at every loop-body emit so nested loops see only the
    /// innermost iter var.
    pub rng_loop_iter_var: std::cell::RefCell<Option<String>>,
    /// Set of static table names referenced by `CgExpr::TableLookup`
    /// nodes within the current kernel body emit. Populated as a
    /// side effect of `lower_cg_expr_to_wgsl`; the kernel composer
    /// (`compose_wgsl_file`) reads this back to prepend a
    /// `const <name>: array<u32, N> = array<u32, N>(…);` module-level
    /// declaration for each referenced table. Interior mutability so
    /// the emit path can stay `&EmitCtx` throughout. Cleared between
    /// kernels by the composer.
    pub tables_referenced:
        std::cell::RefCell<std::collections::BTreeSet<String>>,
    /// Dispatch shape of the kernel currently being emitted, set by
    /// `lower_op_body` before each per-op body emit. Exists so the
    /// downstream `ForEachNeighbor` / fused-fold emitters can pick a
    /// tile-walk WGSL form when the enclosing kernel is
    /// [`crate::cg::dispatch::DispatchShape::PerCell`] vs the
    /// default cell-walk form for `PerAgent`. `None` means the
    /// emitter is being driven by a test or harness that doesn't
    /// route through `lower_op_body` — those paths stay on the
    /// default per-agent shape.
    pub dispatch: std::cell::Cell<Option<crate::cg::dispatch::DispatchShape>>,
    /// View-fold body emit scratch: the LocalIds of every
    /// `Let { value: EventField, ty: AgentId, … }` emitted in the
    /// current stmt list, in source order. ViewStorage assigns
    /// ("self += value") pick up these locals to index into
    /// `view_storage_primary`. The shape depends on the view's
    /// storage hint (looked up via
    /// [`crate::cg::program::ViewSignature::storage_hint`]):
    ///
    /// - `PairMap` (2-D pair-keyed): index =
    ///   `local_<first> * cfg.second_key_pop + local_<second>`. Both
    ///   binders flow into the address compose so the per-(k1, k2)
    ///   slot accumulates independently — without this, single-keying
    ///   on the last binder folded all `(*, k2)` events into the same
    ///   slot.
    /// - Single-key (default): index = `local_<last>` — the legacy
    ///   shape. Kept by routing the LAST AgentId binder.
    ///
    /// The CAS-loop wrapper (`atomicLoad` +
    /// `atomicCompareExchangeWeak`) is the same in both shapes; only
    /// the index expression differs.
    ///
    /// Cleared on every stmt-list emit start so cross-list state
    /// can't leak. Tracking via interior mutability mirrors
    /// `tile_walk_index` — keeps the existing `&EmitCtx` signature
    /// intact.
    pub view_target_locals: std::cell::RefCell<Vec<u32>>,

    /// Cross-agent target-read scratch.
    ///
    /// When a `Read(AgentField { target: AgentRef::Target(expr_id), … })`
    /// is lowered for the first time within a block, the expression
    /// emit pushes `(expr_id, lowered_target_wgsl)` here and adds
    /// `expr_id` to [`Self::bound_target_exprs`]. The next call to
    /// [`lower_cg_stmt_to_wgsl`] drains entries pushed during *this*
    /// stmt's expression sub-tree and emits them as
    /// `let target_expr_<N>: u32 = <wgsl>;` lines BEFORE the stmt body,
    /// so the body's `agent_<field>[target_expr_<N>]` access has a
    /// declared identifier in scope.
    ///
    /// Per-stmt: each `lower_cg_stmt_to_wgsl` call snapshots the
    /// length, lowers the body (which may push), then drains entries
    /// `[snapshot..end]` as the stmt's pre-bindings.
    pub pending_target_lets: std::cell::RefCell<Vec<(CgExprId, String)>>,

    /// Set of `CgExprId`s already pre-bound as `let target_expr_<N>`
    /// in the surrounding block. A `Target(_)` read whose `expr_id` is
    /// in this set reuses the existing binding (just emits
    /// `agent_<field>[target_expr_<N>]`); an `expr_id` not in the set
    /// triggers a new pending entry.
    ///
    /// Save+restore at every stmt-list boundary
    /// ([`lower_cg_stmt_list_to_wgsl`]) so a binding emitted in an
    /// inner scope (e.g. inside an `if` body) can't leak into the
    /// surrounding scope where its declaration isn't visible. Outer-
    /// scope bindings *are* visible to nested scopes (WGSL
    /// function-scope let), so save+restore is the right asymmetry:
    /// inherit on entry, restore on exit.
    pub bound_target_exprs: std::cell::RefCell<std::collections::HashSet<CgExprId>>,

    /// When set, `CgExpr::EventField` reads of `event_ring[...]` emit
    /// via `atomicLoad(&event_ring[...])` instead of plain index
    /// reads. Set by the kernel emit when the kernel's `event_ring`
    /// binding has been declared `array<atomic<u32>>` (PerEvent-
    /// dispatched physics rules whose body also `Emit`s — the same
    /// buffer hosts both atomic stores from the producer-side `Emit`
    /// and per-thread payload reads via `EventField`; WGSL forbids
    /// non-atomic indexing on an atomic-typed binding). The view-fold
    /// path keeps this `false` because its `build_view_fold_bindings`
    /// declares `event_ring` as plain `array<u32>` (read-only
    /// consumer side, no in-kernel `Emit`).
    pub event_ring_atomic_loads: std::cell::Cell<bool>,

    /// When set, `agent_alive` is declared as `array<atomic<u32>>` in
    /// the active kernel; `agents.set_alive(t, false)` writes lower to
    /// `atomicCompareExchangeWeak(&agent_alive[t], 1u, 0u)` and any
    /// subsequent statements in the SAME stmt-list are wrapped in
    /// `if (cas.exchanged) { ... }` so the "first kill wins" semantics
    /// hold under within-tick contention: multiple Damaged threads
    /// landing on one target in one tick would otherwise all observe
    /// the same `old_hp > 0.0` and all emit Defeated; the atomic-CAS
    /// guard collapses the redundant emits to one.
    ///
    /// Set by the kernel emit when the body contains the
    /// `Assign(AgentField::Alive, _, Lit(Bool(false)))` pattern
    /// (detected by [`crate::cg::emit::kernel`]), restored on exit.
    pub alive_atomic_writes: std::cell::Cell<bool>,

    /// Bitset of [`AgentFieldId`] f32 fields that have been upgraded to
    /// `array<atomic<u32>>` in the active kernel because the kernel body
    /// contains an `Assign(AgentField{f32, …}, …)` (chronicle-consumer
    /// RMW write — `agents.set_<f>(t, agents.<f>(t) ± x)` and friends).
    ///
    /// When a bit is set:
    ///   - Reads of `Read(AgentField{f, target})` for `f` in the bitset
    ///     lower as `bitcast<f32>(atomicLoad(&agent_<f>[<idx>]))` instead
    ///     of plain `agent_<f>[<idx>]` (the binding declaration only
    ///     accepts atomic accesses).
    ///   - Writes of `Assign(AgentField{f, target}, value)` lower as a
    ///     CAS loop on `bitcast<u32>(value)` — see
    ///     `lower_cg_stmt_list_to_wgsl` for the full transform (which
    ///     also `var`-promotes any preceding Lets that read the upgraded
    ///     field so the loop body sees a fresh snapshot each iteration).
    ///
    /// Bit indices are produced by [`f32_field_atomic_bit`]; only f32-
    /// typed fields receive a bit. Default 0 (no upgrades) preserves
    /// the existing plain-RMW shape verbatim for kernels whose bodies
    /// don't write any f32 SoA column.
    ///
    /// **P5 (deterministic / replay-equivalent) fix.** Without the CAS,
    /// N chronicle events targeting the same agent slot in one dispatch
    /// raced on the f32 RMW: only one write landed, last-writer-wins on
    /// f32 RMW, and the same seed produced different results across
    /// reruns (observed in `wave_defense`'s `physics_ApplyDamage` where
    /// AOE cleave dispatched many Damaged events at one settler in one
    /// tick).
    pub f32_atomic_field_writes: std::cell::Cell<u64>,

    /// Set of [`LocalId`]s that have been var-promoted for the active
    /// CAS-loop emit. A var-promoted local emits as `local_N = <value>;`
    /// (assignment to a previously-declared `var local_N: T;`) instead
    /// of the default `let local_N: T = <value>;`. Used by the f32 RMW
    /// upgrade in [`lower_cg_stmt_list_to_wgsl`] to:
    ///   1. Declare `var local_N: T;` once BEFORE the CAS loop, so the
    ///      post-Assign suffix stmts can reference `local_N` after the
    ///      loop exits.
    ///   2. Re-execute the chain Lets INSIDE the loop body as
    ///      assignments — each iteration re-reads the upgraded field
    ///      via `atomicLoad` (per the
    ///      [`f32_atomic_field_writes`](Self::f32_atomic_field_writes)
    ///      gate) and re-derives every dependent local. After the CAS
    ///      succeeds, `local_N` holds the committed snapshot the
    ///      successful write was computed against — so post-Assign
    ///      conditional checks (e.g. `if (old_hp > 0.0 && new_hp <=
    ///      0.0)`) reflect the actual transition that won the CAS,
    ///      not a stale read from before contention.
    ///
    /// Populated only during the CAS-loop emit pre-pass; restored to
    /// its prior contents on emit return. Default empty preserves the
    /// existing per-stmt emit shape verbatim.
    pub var_promoted_locals: std::cell::RefCell<std::collections::HashSet<LocalId>>,

    /// 2026-05-09 (Compiler debug mode Phase 2): WGSL-side
    /// atomic-counter instrumentation bitset, mirrored from
    /// [`crate::cg::program::CgProgram::debug_wgsl`] at EmitCtx
    /// construction time. When [`Self::debug_wgsl.event_kind_histogram`]
    /// is set, the chronicle-append skeleton (and dispatcher arm
    /// chain) bumps `event_kind_counts[<kind>]` alongside the existing
    /// `atomicAdd(&event_tail[0], 1u)`. When `mask_hit_rate` is set,
    /// every MaskPredicate body bumps `mask_total[<mask_id>]` per
    /// candidate visit and `mask_passed[<mask_id>]` per pass. When
    /// `score_kernel_visits` is set, every scoring argmax row
    /// bumps `score_kernel_visits[agent_id]` per candidate considered.
    /// Default [`crate::cg::lower::driver::DebugWgslFlags::NONE`]
    /// preserves the existing emit shape verbatim.
    ///
    /// Read-only on `&EmitCtx` — the gate is checked in the same
    /// emit functions that emit the chronicle/mask/scoring bodies.
    pub debug_wgsl: DebugWgslFlags,

    /// "First-writer-wins" CAS gate. When `Some(stmt_id)`, the per-stmt
    /// emit for that exact `Assign` stmt id (an
    /// `Assign(AgentField{f32}, target, Lit(F32(_)))` inside an `If`
    /// whose cond reads the same field) emits a CAS-loop variant that
    /// declares `var _f32_cas_did_transition_<sid>: bool = false;`
    /// outside the loop and sets it to `(_old_bits != _new_bits)` on
    /// the iteration that wins the CAS. The surrounding stmt-list emit
    /// reads the same gate and wraps subsequent stmts (especially
    /// `Emit`) in `if (_f32_cas_did_transition_<sid>) { ... }` so only
    /// the thread that caused the actual state transition fires the
    /// post-write side effects.
    ///
    /// The bare CAS loop's natural retry covers per-thread RHS shapes
    /// (`set_hp(t, hp - 1)`) — every thread sees
    /// a real transition because each contribution is real. But for
    /// a `set_hp(t, 99)` literal-RHS shape guarded by `if (hp == 100)`,
    /// CAS losers retry, see hp==99, and CAS(99→99) succeeds with NO
    /// real transition; without this gate they all run the post-write
    /// `emit Foo`, producing N emits per single 100→99 transition.
    /// Forest_fire's Catch handler exhibits exactly this shape (see
    /// `assets/sim/forest_fire.sim` Catch on EmberLanded).
    ///
    /// Set by [`lower_cg_stmt_body_to_wgsl`]'s `If` arm when the
    /// `(field, target)` of the inner Assign is also read by the
    /// outer cond, restored to `None` after the inner stmt list
    /// returns. Default `None` preserves the existing per-stmt emit
    /// shape verbatim — non-gated CAS sites stay bit-identical to
    /// the pre-fix loop body (so `memory_ordering_cas_emit`'s exact
    /// `.exchanged) { break; }` substring assertion still holds for
    /// the chronicle-damage shape).
    pub f32_first_writer_gate: std::cell::Cell<Option<u32>>,

    /// Dense producer-kernel-id map built by `assign_producer_kernel_ids`
    /// before the kernel loop. Keyed by `(stage, kernel)` index.
    /// Looked up in `lower_emit_to_wgsl` to fill the seq trailer's
    /// `kernel_id` nibble.
    pub producer_kernel_ids:
        std::collections::BTreeMap<crate::cg::emit::program::KernelIndex, u32>,

    /// Index of the kernel currently being emitted — set before each
    /// kernel body emit, cleared (None) outside. Used with `producer_kernel_ids`
    /// to resolve the current kernel's producer id.
    pub current_kernel_index: std::cell::Cell<Option<crate::cg::emit::program::KernelIndex>>,

    /// Per-kernel intra-emit index. Reset to 0 before each kernel body
    /// emit; incremented by `lower_emit_to_wgsl` for each `CgStmt::Emit`
    /// encountered during the kernel walk. Packs into the low 4 bits of
    /// the seq trailer `(kernel_id << 24) | (thread_idx << 4) | emit_idx`.
    pub intra_emit_idx: std::cell::Cell<u32>,

    /// When set, f32+Add single-key ViewFold kernels emit a PerAgent serial
    /// scan (one thread per observer slot, inner loop over all events) instead
    /// of the default PerEvent CAS+add loop.  Each thread is the sole writer
    /// for its slot — no contention, no retry, deterministic accumulation.
    ///
    /// Computed by the program emitter by scanning `prog.ops` for any
    /// `ViewFold` with f32 result type and `Add` fold op; set to `true` when
    /// such an op exists.  Set by `emit_cg_program_with_debug`; default
    /// `false` leaves the existing PerEvent CAS path unchanged for programs
    /// that have no f32+Add folds.
    pub serial_f32_fold: std::cell::Cell<bool>,

    /// When set, the current stmt-list emit is inside a serial fold's inner
    /// event-scan loop.  Two emission behaviours change:
    ///
    /// 1. `CgExpr::EventField` reads use `_ei` (the scan loop variable)
    ///    instead of `event_idx`.
    /// 2. `CgStmt::Assign { target: ViewStorage(f32+Add) }` emits
    ///    `if (local_N == observer_slot) { accum = accum + rhs; }`
    ///    instead of the CAS retry loop — the serial scan guarantees
    ///    single-writer per slot so no atomics are needed.
    ///
    /// Set by `build_view_fold_wgsl_body`'s serial-scan path around the
    /// `lower_cg_stmt_list_to_wgsl` call; restored to `false` on return.
    /// Default `false` preserves the existing per-event CAS path verbatim.
    pub in_serial_fold_body: std::cell::Cell<bool>,
}

impl<'a> EmitCtx<'a> {
    /// Construct an emit context with the [`HandleNamingStrategy::Structural`]
    /// strategy — the only one Task 4.1 ships.
    pub fn structural(prog: &'a CgProgram) -> Self {
        Self {
            prog,
            naming: HandleNamingStrategy::Structural,
            tile_walk_index: std::cell::RefCell::new(None),
            rng_loop_iter_var: std::cell::RefCell::new(None),
            tables_referenced: std::cell::RefCell::new(
                std::collections::BTreeSet::new(),
            ),
            dispatch: std::cell::Cell::new(None),
            view_target_locals: std::cell::RefCell::new(Vec::new()),
            pending_target_lets: std::cell::RefCell::new(Vec::new()),
            bound_target_exprs: std::cell::RefCell::new(std::collections::HashSet::new()),
            event_ring_atomic_loads: std::cell::Cell::new(false),
            alive_atomic_writes: std::cell::Cell::new(false),
            f32_atomic_field_writes: std::cell::Cell::new(0),
            var_promoted_locals: std::cell::RefCell::new(std::collections::HashSet::new()),
            // 2026-05-09 (Compiler debug mode Phase 2): mirror the
            // program-level WGSL instrumentation bitset so emit
            // functions can gate their atomic-counter additions on
            // a single pure read of `ctx.debug_wgsl`. Default
            // `DebugWgslFlags::NONE` (when no opts threaded) leaves
            // the existing emit shape unchanged.
            debug_wgsl: prog.debug_wgsl,
            f32_first_writer_gate: std::cell::Cell::new(None),
            producer_kernel_ids: std::collections::BTreeMap::new(),
            current_kernel_index: std::cell::Cell::new(None),
            intra_emit_idx: std::cell::Cell::new(0),
            serial_f32_fold: std::cell::Cell::new(false),
            in_serial_fold_body: std::cell::Cell::new(false),
        }
    }

    /// Render `handle` as a WGSL identifier per the active naming
    /// strategy.
    ///
    /// # Limitations
    ///
    /// - With [`HandleNamingStrategy::Structural`], every variant
    ///   produces a deterministic identifier; [`AgentRef::Target(id)`]
    ///   renders as `agent_target_expr_<N>_<field>` *for the bare
    ///   handle name only* (snapshot tests). The active per-stmt emit
    ///   uses [`agent_field_access`]'s indexed form
    ///   `agent_<field>[target_expr_<N>]` paired with a hoisted
    ///   `let target_expr_<N>` — see the module-level note for the
    ///   threading mechanism.
    /// - Plumbing-only handles ([`DataHandle::AliveBitmap`],
    ///   [`DataHandle::IndirectArgs`], [`DataHandle::AgentScratch`],
    ///   [`DataHandle::SimCfgBuffer`], [`DataHandle::SnapshotKick`])
    ///   never appear inside an expression body in a well-formed
    ///   program (they live on `PlumbingKind` ops). The Structural
    ///   strategy still gives them a deterministic name so error
    ///   diagnostics on a malformed IR remain readable.
    pub fn handle_name(&self, h: &DataHandle) -> String {
        match self.naming {
            HandleNamingStrategy::Structural => structural_handle_name(h),
        }
    }
}

/// True when `stmt` is `Assign(AgentField::Alive, _, Lit(Bool(false)))` —
/// the "kill transition" pattern targeted by the atomicCAS guard.
/// The check is structural: the assigned value resolves
/// through the expression arena to a literal `false`. Used both by
/// the kernel emit (to upgrade the `agent_alive` binding to
/// AtomicStorage and set [`EmitCtx::alive_atomic_writes`]) and by
/// the per-stmt-list emit (to rewrite the assign + wrap subsequent
/// stmts in `if (cas.exchanged) { ... }`).
pub(crate) fn stmt_is_set_alive_false(prog: &CgProgram, stmt: &CgStmt) -> bool {
    let CgStmt::Assign { target, value } = stmt else {
        return false;
    };
    let DataHandle::AgentField {
        field: AgentFieldId::Alive,
        ..
    } = target
    else {
        return false;
    };
    matches!(
        <CgProgram as ExprArena>::get(prog, *value),
        Some(CgExpr::Lit(LitValue::Bool(false)))
    )
}

/// Recursively scan a stmt-list (and any nested If/Match/ForEachNeighborBody
/// bodies) for the alive-CAS pattern. Returns `true` if at least one
/// `Assign(AgentField::Alive, _, Lit(Bool(false)))` is reachable.
/// Called once per kernel by the kernel emitter to decide whether the
/// `agent_alive` binding needs the AtomicStorage upgrade.
pub(crate) fn stmt_list_contains_set_alive_false(
    prog: &CgProgram,
    list_id: CgStmtListId,
) -> bool {
    let Some(list) = <CgProgram as StmtListArena>::get(prog, list_id) else {
        return false;
    };
    for stmt_id in &list.stmts {
        let Some(stmt) = <CgProgram as StmtArena>::get(prog, *stmt_id) else {
            continue;
        };
        if stmt_is_set_alive_false(prog, stmt) {
            return true;
        }
        match stmt {
            CgStmt::If { then, else_, .. } => {
                if stmt_list_contains_set_alive_false(prog, *then) {
                    return true;
                }
                if let Some(e) = else_ {
                    if stmt_list_contains_set_alive_false(prog, *e) {
                        return true;
                    }
                }
            }
            CgStmt::Match { arms, .. } => {
                for arm in arms {
                    if stmt_list_contains_set_alive_false(prog, arm.body) {
                        return true;
                    }
                }
            }
            CgStmt::ForEachNeighborBody { body, .. } => {
                if stmt_list_contains_set_alive_false(prog, *body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Map an [`AgentFieldId`] of [`AgentFieldTy::F32`] to a stable bit
/// index in `[0..64)` for use in [`EmitCtx::f32_atomic_field_writes`]
/// bitsets. Returns `None` for non-f32 fields.
///
/// The mapping is **stable across compiler runs** — the variant order
/// in the match arm is the source of truth. Adding a new f32 variant
/// to `AgentFieldId` requires appending a new arm here (the match is
/// exhaustive over the f32 subset). Total f32 fields today: 28 (well
/// under the u64 capacity).
///
/// Used by:
///   - [`stmt_is_f32_agent_field_assign`] — to test if a stmt's target
///     field is f32 (and thus eligible for the RMW upgrade).
///   - [`stmt_list_collect_f32_atomic_writes`] — to OR each found
///     write's bit into the per-kernel bitset.
///   - The kernel-emit binding upgrade (in `cg/emit/kernel.rs`) — to
///     decide which `agent_<f>` binding lines to declare as
///     `array<atomic<u32>>`.
///   - The Read / Assign arms of the per-stmt lowering — to gate the
///     `bitcast<f32>(atomicLoad(…))` and CAS-loop emit shapes.
pub(crate) fn f32_field_atomic_bit(field: AgentFieldId) -> Option<u8> {
    use AgentFieldId::*;
    let bit = match field {
        Hp => 0,
        MaxHp => 1,
        ShieldHp => 2,
        Armor => 3,
        MagicResist => 4,
        AttackDamage => 5,
        AbilityPower => 6,
        AttackRange => 7,
        Mana => 8,
        MaxMana => 9,
        MoveSpeed => 10,
        MoveSpeedMult => 11,
        Hunger => 12,
        Thirst => 13,
        RestTimer => 14,
        Safety => 15,
        Shelter => 16,
        Social => 17,
        Purpose => 18,
        Esteem => 19,
        RiskTolerance => 20,
        SocialDrive => 21,
        Ambition => 22,
        Altruism => 23,
        Curiosity => 24,
        TravelDestX => 25,
        TravelDestY => 26,
        TravelDestZ => 27,
        // Non-f32 fields — no bit. Pos / Vel are vec3 (handled
        // separately, vec3 atomics aren't supported by WGSL anyway);
        // u32 / Bool / I16 / EnumU8 / OptAgentId / OptEnumU32 don't
        // need the f32 RMW upgrade (their writes already lower
        // through the appropriate scalar atomics or skip the upgrade
        // because they aren't part of the f32 RMW race class).
        _ => return None,
    };
    Some(bit)
}

/// True when `stmt` is `Assign(AgentField{f32, …}, …)` — the f32 RMW
/// pattern targeted by the [`EmitCtx::f32_atomic_field_writes`]
/// upgrade. Returns the field id when matched, so callers can OR its
/// [`f32_field_atomic_bit`] into the per-kernel bitset.
///
/// Note that this matches **any** assign to an f32 SoA column, not
/// just `agents.set_<f>(t, agents.<f>(t) ± x)` shapes — a literal
/// `agents.set_hp(t, 5.0)` write is still racy under N events targeting
/// one slot, so the safe choice is to treat every f32 SoA write as
/// needing the CAS upgrade. The CAS loop body still terminates in
/// O(retries-until-no-contention) which is bounded by the number of
/// concurrent threads in the dispatch — typically tiny.
pub(crate) fn stmt_is_f32_agent_field_assign(
    _prog: &CgProgram,
    stmt: &CgStmt,
) -> Option<AgentFieldId> {
    let CgStmt::Assign { target, .. } = stmt else {
        return None;
    };
    let DataHandle::AgentField { field, .. } = target else {
        return None;
    };
    if !matches!(field.ty(), AgentFieldTy::F32) {
        return None;
    }
    Some(*field)
}

/// True when `expr_id` (or any descendant CgExpr) reads the upgraded
/// f32 field `field` OR a `ReadLocal(L)` for `L in chain_locals`.
/// Used by [`lower_cg_stmt_list_to_wgsl`]'s f32 RMW pre-pass to
/// decide which Lets need to re-execute INSIDE the CAS loop (those
/// whose value-expression is "tainted" by the upgraded field, so a
/// CAS retry needs to recompute against the latest snapshot). Lets
/// not in the chain stay as ordinary outside-loop bindings.
///
/// Walks the expression arena recursively, descending through every
/// arm of every variant. Detection is structural — a transitive
/// dependency through a `ReadLocal(L)` is captured because earlier
/// chain-membership decisions populated `chain_locals` (forward
/// walk over the residual stmt list).
fn expr_depends_on_upgraded_field(
    expr_id: CgExprId,
    field: AgentFieldId,
    chain_locals: &std::collections::HashSet<LocalId>,
    prog: &CgProgram,
) -> bool {
    let Some(node) = <CgProgram as ExprArena>::get(prog, expr_id) else {
        return false;
    };
    match node {
        CgExpr::Read(DataHandle::AgentField { field: f, .. }) => *f == field,
        CgExpr::Read(_) => false,
        CgExpr::ReadLocal { local, .. } => chain_locals.contains(local),
        CgExpr::Lit(_)
        | CgExpr::AgentSelfId
        | CgExpr::PerPairCandidateId
        | CgExpr::EventField { .. }
        | CgExpr::Rng { .. }
        | CgExpr::NamespaceField { .. } => false,
        CgExpr::Unary { arg, .. } => {
            expr_depends_on_upgraded_field(*arg, field, chain_locals, prog)
        }
        CgExpr::Binary { lhs, rhs, .. } => {
            expr_depends_on_upgraded_field(*lhs, field, chain_locals, prog)
                || expr_depends_on_upgraded_field(*rhs, field, chain_locals, prog)
        }
        CgExpr::Builtin { args, .. } => args
            .iter()
            .any(|a| expr_depends_on_upgraded_field(*a, field, chain_locals, prog)),
        CgExpr::Select { cond, then, else_, .. } => {
            expr_depends_on_upgraded_field(*cond, field, chain_locals, prog)
                || expr_depends_on_upgraded_field(*then, field, chain_locals, prog)
                || expr_depends_on_upgraded_field(*else_, field, chain_locals, prog)
        }
        CgExpr::NamespaceCall { args, .. } => args
            .iter()
            .any(|a| expr_depends_on_upgraded_field(*a, field, chain_locals, prog)),
        CgExpr::TableLookup { index, .. } => {
            expr_depends_on_upgraded_field(*index, field, chain_locals, prog)
        }
    }
}

/// True when `expr_id` (or any descendant CgExpr) reads any local in
/// `chain_locals` via [`CgExpr::ReadLocal`]. Used by the f32 RMW pass
/// to decide if a non-chain prefix stmt (e.g. an `agents.set_pos`
/// Assign whose RHS references a chain-let) MUST run inside the CAS
/// loop body — otherwise the var-promoted local has no committed
/// value at the stmt's source-order position and the emitted WGSL
/// references it before its declaration.
///
/// Closes Gap #3 of `docs/architecture/gaps_among_us.md`.
fn expr_reads_any_chain_local(
    expr_id: CgExprId,
    chain_locals: &std::collections::HashSet<LocalId>,
    prog: &CgProgram,
) -> bool {
    let Some(node) = <CgProgram as ExprArena>::get(prog, expr_id) else {
        return false;
    };
    match node {
        CgExpr::ReadLocal { local, .. } => chain_locals.contains(local),
        CgExpr::Read(_)
        | CgExpr::Lit(_)
        | CgExpr::AgentSelfId
        | CgExpr::PerPairCandidateId
        | CgExpr::EventField { .. }
        | CgExpr::Rng { .. }
        | CgExpr::NamespaceField { .. } => false,
        CgExpr::Unary { arg, .. } => {
            expr_reads_any_chain_local(*arg, chain_locals, prog)
        }
        CgExpr::Binary { lhs, rhs, .. } => {
            expr_reads_any_chain_local(*lhs, chain_locals, prog)
                || expr_reads_any_chain_local(*rhs, chain_locals, prog)
        }
        CgExpr::Builtin { args, .. } => args
            .iter()
            .any(|a| expr_reads_any_chain_local(*a, chain_locals, prog)),
        CgExpr::Select { cond, then, else_, .. } => {
            expr_reads_any_chain_local(*cond, chain_locals, prog)
                || expr_reads_any_chain_local(*then, chain_locals, prog)
                || expr_reads_any_chain_local(*else_, chain_locals, prog)
        }
        CgExpr::NamespaceCall { args, .. } => args
            .iter()
            .any(|a| expr_reads_any_chain_local(*a, chain_locals, prog)),
        CgExpr::TableLookup { index, .. } => {
            expr_reads_any_chain_local(*index, chain_locals, prog)
        }
    }
}

/// True when `stmt` reads any local in `chain_locals` via any of its
/// expression children. Wraps [`expr_reads_any_chain_local`] over the
/// stmt's expression operands (Assign value, Emit field exprs, If
/// cond, Match scrutinee, Let value, ApplyAbility ability/caster/
/// target, ViewStorageAppend field exprs, ForEachAgent/ForEachNeighbor
/// init+projection). Nested stmt lists (If branches, Match arms,
/// ForEach*Body) are NOT walked — the chain-local promotion only
/// covers the current stmt list scope; nested lists open their own
/// scopes and are emitted via recursive `lower_cg_stmt_list_to_wgsl`
/// calls that re-derive their own f32 RMW state.
fn stmt_reads_any_chain_local(
    stmt: &CgStmt,
    chain_locals: &std::collections::HashSet<LocalId>,
    prog: &CgProgram,
) -> bool {
    match stmt {
        CgStmt::Assign { value, .. } => {
            expr_reads_any_chain_local(*value, chain_locals, prog)
        }
        CgStmt::Emit { fields, .. } => fields
            .iter()
            .any(|(_, e)| expr_reads_any_chain_local(*e, chain_locals, prog)),
        CgStmt::If { cond, .. } => {
            expr_reads_any_chain_local(*cond, chain_locals, prog)
        }
        CgStmt::Match { scrutinee, .. } => {
            expr_reads_any_chain_local(*scrutinee, chain_locals, prog)
        }
        CgStmt::Let { value, .. } => {
            expr_reads_any_chain_local(*value, chain_locals, prog)
        }
        CgStmt::ForEachAgent { init, projection, .. }
        | CgStmt::ForEachNeighbor { init, projection, .. } => {
            expr_reads_any_chain_local(*init, chain_locals, prog)
                || expr_reads_any_chain_local(*projection, chain_locals, prog)
        }
        CgStmt::ForEachNeighborBody { .. } | CgStmt::ForEachAgentBody { .. } => false,
        CgStmt::ApplyAbility { ability, caster, target, .. } => {
            expr_reads_any_chain_local(*ability, chain_locals, prog)
                || expr_reads_any_chain_local(*caster, chain_locals, prog)
                || expr_reads_any_chain_local(*target, chain_locals, prog)
        }
        CgStmt::ViewStorageAppend { fields, .. } => fields
            .iter()
            .any(|(_, e)| expr_reads_any_chain_local(*e, chain_locals, prog)),
    }
}

/// True when `stmt` is `Assign(AgentField{f32, …}, target, Lit(F32(_)))`
/// — the "first-writer-wins" candidate shape. Returns the field id
/// and target ref so the caller can correlate
/// against an enclosing `If` cond reading the same field.
///
/// Restricting to literal-F32 RHS is the load-bearing safety filter:
/// a per-thread RHS shape (`set_hp(t, hp - 1)`) does NOT need
/// transition gating — the natural CAS retry covers correctness
/// because every retry computes a fresh contribution. Only the
/// constant-RHS shape produces no-op CAS retries (loser sees the
/// already-written constant and CAS-stores-the-same-value succeeds
/// trivially), and only that shape causes downstream side-effects
/// to over-fire.
pub(crate) fn stmt_is_f32_const_assign(
    prog: &CgProgram,
    stmt: &CgStmt,
) -> Option<(AgentFieldId, AgentRef)> {
    let CgStmt::Assign { target, value } = stmt else {
        return None;
    };
    let DataHandle::AgentField { field, target: agent_ref } = target else {
        return None;
    };
    if !matches!(field.ty(), AgentFieldTy::F32) {
        return None;
    }
    match <CgProgram as ExprArena>::get(prog, *value) {
        Some(CgExpr::Lit(LitValue::F32(_))) => Some((*field, agent_ref.clone())),
        _ => None,
    }
}

/// True when `expr_id` (or any descendant CgExpr) reads
/// `Read(AgentField { field: <field>, .. })` — without requiring the
/// `target` to match. Used by the post-CAS emit-gating detection:
/// when the inner `Assign` writes an f32 field with a literal RHS
/// AND the enclosing `If`'s cond reads the same
/// field id (any target), we infer "first-writer-wins" intent and
/// gate subsequent stmts on actual transition.
///
/// Loose target match is intentional. Two `AgentRef::Target(expr_id)`
/// values referencing the same source local end up with different
/// expr ids in the IR (each `lower_field` call allocates a fresh
/// `CgExprId` for the base) — strict equality would miss the
/// canonical Catch-handler shape `if (agents.hp(t) >= 100.0) { …
/// agents.set_hp(t, 99.0); … }` even though both reads target the
/// same row. Field-id equality is sufficient because the gating
/// predicate is "did THIS CAS transition?" — even if the cond reads
/// hp(other) and the assign writes hp(self), the transition-gate
/// remains correct (only fires the post-write side-effect when the
/// CAS produced a real value change on the assigned slot).
fn expr_reads_agent_field_id(
    expr_id: CgExprId,
    field: AgentFieldId,
    prog: &CgProgram,
) -> bool {
    let Some(node) = <CgProgram as ExprArena>::get(prog, expr_id) else {
        return false;
    };
    match node {
        CgExpr::Read(DataHandle::AgentField { field: f, .. }) => *f == field,
        CgExpr::Read(_)
        | CgExpr::Lit(_)
        | CgExpr::AgentSelfId
        | CgExpr::PerPairCandidateId
        | CgExpr::EventField { .. }
        | CgExpr::Rng { .. }
        | CgExpr::NamespaceField { .. }
        | CgExpr::ReadLocal { .. } => false,
        CgExpr::Unary { arg, .. } => {
            expr_reads_agent_field_id(*arg, field, prog)
        }
        CgExpr::Binary { lhs, rhs, .. } => {
            expr_reads_agent_field_id(*lhs, field, prog)
                || expr_reads_agent_field_id(*rhs, field, prog)
        }
        CgExpr::Builtin { args, .. } => args
            .iter()
            .any(|a| expr_reads_agent_field_id(*a, field, prog)),
        CgExpr::Select { cond, then, else_, .. } => {
            expr_reads_agent_field_id(*cond, field, prog)
                || expr_reads_agent_field_id(*then, field, prog)
                || expr_reads_agent_field_id(*else_, field, prog)
        }
        CgExpr::NamespaceCall { args, .. } => args
            .iter()
            .any(|a| expr_reads_agent_field_id(*a, field, prog)),
        CgExpr::TableLookup { index, .. } => {
            expr_reads_agent_field_id(*index, field, prog)
        }
    }
}

/// Scan a stmt list for the FIRST `Assign(AgentField{f32}, target,
/// Lit(F32(_)))` — the inner-Assign side of the "first-writer-wins"
/// pattern. Returns the stmt id (so the per-stmt emit can match
/// against the gate) and the field id. Returns `None` if no such
/// stmt exists at the top level of `list_id`.
///
/// Top-level only — nested `If` / `Match` arms have their own
/// inner stmt lists and are detected by their own enclosing `If` arm
/// (recursion happens through `lower_cg_stmt_body_to_wgsl`).
pub(crate) fn stmt_list_first_f32_const_assign(
    prog: &CgProgram,
    list_id: CgStmtListId,
) -> Option<(CgStmtId, AgentFieldId)> {
    let list = <CgProgram as StmtListArena>::get(prog, list_id)?;
    for stmt_id in &list.stmts {
        let stmt = <CgProgram as StmtArena>::get(prog, *stmt_id)?;
        if let Some((field, _)) = stmt_is_f32_const_assign(prog, stmt) {
            return Some((*stmt_id, field));
        }
    }
    None
}

/// Recursive walk: collect the set of f32 [`AgentFieldId`]s assigned
/// inside the stmt list named by `list_id`. Returns a u64 bitset keyed
/// by [`f32_field_atomic_bit`]. Descends through `If` / `Match` /
/// `ForEachNeighborBody` / `ForEachAgentBody` so an f32 RMW write
/// hidden inside a nested arm still triggers the binding upgrade.
///
/// Called once per kernel by the kernel emit (in `cg/emit/kernel.rs`)
/// to decide which `agent_<f>` bindings need the AtomicStorage upgrade
/// for THIS kernel.
pub(crate) fn stmt_list_collect_f32_atomic_writes(
    prog: &CgProgram,
    list_id: CgStmtListId,
) -> u64 {
    let mut bits = 0u64;
    let Some(list) = <CgProgram as StmtListArena>::get(prog, list_id) else {
        return 0;
    };
    for stmt_id in &list.stmts {
        let Some(stmt) = <CgProgram as StmtArena>::get(prog, *stmt_id) else {
            continue;
        };
        if let Some(field) = stmt_is_f32_agent_field_assign(prog, stmt) {
            if let Some(bit) = f32_field_atomic_bit(field) {
                bits |= 1u64 << bit;
            }
        }
        match stmt {
            CgStmt::If { then, else_, .. } => {
                bits |= stmt_list_collect_f32_atomic_writes(prog, *then);
                if let Some(e) = else_ {
                    bits |= stmt_list_collect_f32_atomic_writes(prog, *e);
                }
            }
            CgStmt::Match { arms, .. } => {
                for arm in arms {
                    bits |= stmt_list_collect_f32_atomic_writes(prog, arm.body);
                }
            }
            CgStmt::ForEachNeighborBody { body, .. } => {
                bits |= stmt_list_collect_f32_atomic_writes(prog, *body);
            }
            _ => {}
        }
    }
    bits
}

// ---------------------------------------------------------------------------
// Structural handle naming
// ---------------------------------------------------------------------------

/// Render `handle` as a deterministic WGSL identifier — the
/// [`HandleNamingStrategy::Structural`] form. Stable across runs.
fn structural_handle_name(h: &DataHandle) -> String {
    match h {
        DataHandle::AgentField { field, target } => {
            format!("agent_{}_{}", agent_ref_token(target), field.snake())
        }
        // Item / Group field handles emit the same structural shape
        // the kernel binding names use; WGSL bodies that read them
        // produce `<entity>_<field>[<expr>]` via the dedicated
        // `Read` arm in `lower_cg_expr_to_wgsl` rather than this
        // generic name. Keeping a stable structural name for the
        // catch-all fallback path.
        DataHandle::ItemField { field, target } => {
            format!("item_{}_{}_target_{}", field.entity, field.slot, target.0)
        }
        DataHandle::GroupField { field, target } => {
            format!("group_{}_{}_target_{}", field.entity, field.slot, target.0)
        }
        DataHandle::ViewStorage { view, slot } => {
            format!("view_{}_{}", view.0, view_slot_token(*slot))
        }
        DataHandle::EventRing { ring, kind } => {
            format!("event_ring_{}_{}", ring.0, event_ring_access_token(*kind))
        }
        DataHandle::ConfigConst { id } => format!("config_{}", id.0),
        DataHandle::MaskBitmap { mask } => format!("mask_{}_bitmap", mask.0),
        DataHandle::ScoringOutput => "scoring_output".to_string(),
        DataHandle::SpatialStorage { kind } => {
            format!("spatial_{}", spatial_storage_token(*kind))
        }
        DataHandle::Rng { purpose } => format!("rng_{}", rng_purpose_token(*purpose)),
        DataHandle::AliveBitmap => "alive_bitmap".to_string(),
        DataHandle::IndirectArgs { ring } => format!("indirect_args_{}", ring.0),
        DataHandle::AgentScratch { kind } => {
            format!("agent_scratch_{}", agent_scratch_token(*kind))
        }
        DataHandle::SimCfgBuffer => "sim_cfg_buffer".to_string(),
        DataHandle::SnapshotKick => "snapshot_kick".to_string(),
        DataHandle::AbilityRegistryColumn { column } => {
            // The dispatcher kernel reads via `ability_registry_<column>[i]`.
            // Stable per-column WGSL identifier so the BGL composer's
            // structural binding-name pass (in `cg/emit/kernel.rs`) lines
            // up against the same string the body references.
            format!("ability_registry_{}", ability_registry_column_token(*column))
        }
        DataHandle::BeliefStateColumn { column } => column.binding_name().to_string(),
    }
}

/// Stable snake_case token for a [`DataHandle::AbilityRegistryColumn`] —
/// used by both the WGSL body emit (indexed access) and the BGL
/// composer (binding-name composition). Naming MUST stay in sync with
/// the `PackedAbilityRegistryGpu` field names in
/// `crates/engine/src/ability/registry_gpu.rs`.
fn ability_registry_column_token(column: super::super::data_handle::AbilityRegistryColumn) -> &'static str {
    use super::super::data_handle::AbilityRegistryColumn::*;
    match column {
        Hints           => "hints",
        CooldownTicks   => "cooldown_ticks",
        Range           => "range",
        GateFlags       => "gate_flags",
        DeliveryKind    => "delivery_kind",
        EffectKinds     => "effect_kinds",
        EffectPayloadA  => "effect_payload_a",
        EffectPayloadB  => "effect_payload_b",
        TagValues       => "tag_values",
        Stackings       => "stackings",
        Chances         => "chances",
        LifetimeKinds   => "lifetime_kinds",
        LifetimePayloads => "lifetime_payloads",
        AreaKinds       => "area_kinds",
        AreaArgs        => "area_args",
        ScalingStatRefs => "scaling_stat_refs",
        ScalingPercents => "scaling_percents",
        NestedEffectKinds    => "nested_effect_kinds",
        NestedEffectPayloadA => "nested_effect_payload_a",
        NestedEffectPayloadB => "nested_effect_payload_b",
        WhenPredBinder       => "when_pred_binder",
        WhenPredField        => "when_pred_field",
        WhenPredOp           => "when_pred_op",
        WhenPredLiteral      => "when_pred_literal",
        // Plan H slice 2 — telegraph metadata. Names mirror the
        // PackedAbilityRegistryGpu field names so the BGL composer's
        // `ability_registry_<token>` binding lookup resolves to the
        // matching wgpu::Buffer at upload time.
        TelegraphKind        => "telegraph_kind",
        TelegraphParams      => "telegraph_params",
    }
}

/// Render `agent_<field>[<index_expr>]` — the indexed access on the
/// shared SoA binding for `DataHandle::AgentField { field, target }`.
///
/// The index expression depends on the agent-ref:
///   - `Self_` → `agent_id` (kernel-bound for PerAgent dispatch)
///   - `EventTarget` → `event_target_id` (PerEvent preamble-bound)
///   - `PerPairCandidate` → `per_pair_candidate` (PerPair preamble-bound)
///   - `Actor` → `actor_id` (PerEvent preamble-bound)
///
/// `Target(expr_id)` resolves to `target_expr_<N>` (where `<N>` is
/// `expr_id.0`) — the caller is responsible for ensuring a stmt-prefix
/// `let target_expr_<N>: u32 = <wgsl>;` is in scope. The `Read` /
/// `Assign` arms of [`lower_cg_expr_to_wgsl`] / [`lower_cg_stmt_to_wgsl`]
/// queue that binding via [`EmitCtx::pending_target_lets`] on first
/// reference; the public stmt-emit drains pending entries as
/// pre-stmt let lines.
///
/// The binding side (`structural_binding_name` in `cg/emit/kernel.rs`)
/// already drops the agent-ref discriminator and uses just
/// `agent_<field>` — so the body's indexed access lines up against
/// the declared `array<...>` binding without naming drift.
fn agent_field_access(field: AgentFieldId, target: &AgentRef) -> String {
    let index = match target {
        AgentRef::Self_ => "agent_id".to_string(),
        AgentRef::EventTarget => "event_target_id".to_string(),
        AgentRef::Actor => "actor_id".to_string(),
        AgentRef::PerPairCandidate => "per_pair_candidate".to_string(),
        AgentRef::Target(id) => format!("target_expr_{}", id.0),
    };
    let raw = format!("agent_{}[{}]", field.snake(), index);
    // Bool fields are stored as `array<u32>` on the GPU (boolean
    // storage isn't host-shareable in WGSL, see `kernel.rs`'s
    // `AgentFieldTy::Bool => "array<u32>"`); coerce back to bool at
    // every read site so the WGSL type-checker accepts the value in
    // bool position (`if`, `&&`, `!`, etc.).
    match field.ty() {
        AgentFieldTy::Bool => format!("({raw} != 0u)"),
        _ => raw,
    }
}

/// Render an [`AgentField`] read with the [`EmitCtx`] atomic-upgrade
/// gates applied. Wraps [`agent_field_access`]'s plain indexed form
/// in `bitcast<f32>(atomicLoad(&agent_<f>[<idx>]))` when the f32 RMW
/// upgrade applies to `field` in the active kernel
/// (`ctx.f32_atomic_field_writes` bit set). Otherwise returns the
/// plain form unchanged.
///
/// The wrapper exists so the per-stmt and per-expr emit paths share
/// one decision point — extending the gate (e.g. adding a u32 RMW
/// upgrade later) requires editing only this helper, not every read
/// site. See [`f32_field_atomic_bit`] for the bit mapping.
fn agent_field_read_wgsl(field: AgentFieldId, target: &AgentRef, ctx: &EmitCtx) -> String {
    if matches!(field.ty(), AgentFieldTy::F32) {
        if let Some(bit) = f32_field_atomic_bit(field) {
            if (ctx.f32_atomic_field_writes.get() >> bit) & 1 == 1 {
                let lhs = agent_field_access_lvalue(field, target);
                return format!("bitcast<f32>(atomicLoad(&{lhs}))");
            }
        }
    }
    agent_field_access(field, target)
}

/// Render `agent_<field>[<idx>]` (or its `bitcast<f32>(atomicLoad(…))`
/// equivalent) as the value-side of an apply-ability dispatcher case,
/// e.g. `agent_hp[caster_slot]` becomes
/// `bitcast<f32>(atomicLoad(&agent_hp[caster_slot]))` when the active
/// kernel upgraded `agent_hp` to `array<atomic<u32>>`. Used by the
/// dispatcher template emit (`build_apply_ability_per_target_body`'s
/// caller) to keep stat-dispatch reads valid against atomic-typed
/// bindings. The `idx_var` is the WGSL identifier (`caster_slot`,
/// `pred_agent`, etc.) — passed verbatim into the indexed access.
fn dispatcher_f32_field_read(field: AgentFieldId, idx_var: &str, ctx: &EmitCtx) -> String {
    let snake = field.snake();
    if let Some(bit) = f32_field_atomic_bit(field) {
        if (ctx.f32_atomic_field_writes.get() >> bit) & 1 == 1 {
            return format!("bitcast<f32>(atomicLoad(&agent_{snake}[{idx_var}]))");
        }
    }
    format!("agent_{snake}[{idx_var}]")
}

/// Raw indexed access — no bool coercion. Used as the LHS of an
/// assignment (`agent_alive[t] = u32(value);`) since the read-side
/// `(x != 0u)` wrapper is not a valid lvalue.
fn agent_field_access_lvalue(field: AgentFieldId, target: &AgentRef) -> String {
    let index = match target {
        AgentRef::Self_ => "agent_id".to_string(),
        AgentRef::EventTarget => "event_target_id".to_string(),
        AgentRef::Actor => "actor_id".to_string(),
        AgentRef::PerPairCandidate => "per_pair_candidate".to_string(),
        AgentRef::Target(id) => format!("target_expr_{}", id.0),
    };
    format!("agent_{}[{}]", field.snake(), index)
}

/// Identifier token for an [`AgentRef`]. `Target(expr_id)` maps to the
/// placeholder `target_expr_<N>` per the module-level limitations note;
/// [`AgentRef::PerPairCandidate`] maps to the placeholder
/// `per_pair_candidate` until Task 4.x resolves it to the per-pair
/// candidate buffer + per-thread offset implied by the surrounding
/// [`crate::cg::dispatch::DispatchShape::PerPair`] shape.
fn agent_ref_token(target: &AgentRef) -> String {
    match target {
        AgentRef::Self_ => "self".to_string(),
        AgentRef::Actor => "actor".to_string(),
        AgentRef::EventTarget => "event_target".to_string(),
        AgentRef::Target(id) => format!("target_expr_{}", id.0),
        AgentRef::PerPairCandidate => "per_pair_candidate".to_string(),
    }
}

/// Resolve an Item / Group field's binding name via the program's
/// catalog. Returns the field-keyed form `item_<field_snake>` (e.g.
/// `item_base_price`) for Item fields or `group_<field_snake>` for
/// Group fields when the (entity, slot) pair has a catalog entry;
/// falls back to the opaque structural form `item_<entity>_<slot>`
/// / `group_<entity>_<slot>` so the WGSL still parses if the catalog
/// is missing the entry (a lowering defect).
///
/// Gap T2 fix (2026-05-12): the binding name was previously
/// `<entity_snake>_<field_snake>` (e.g. `coin_weight` /
/// `grain_base_price`), which produced one binding per Item entity.
/// The build_helper alloc loop only emitted the FIRST per field name,
/// silently dropping the others. Post-fix the binding is field-keyed
/// across all Items declaring that field name; the discriminant lives
/// in the user index expression `items.<field>(N)` (N = position in
/// Item-only declaration order).
pub(crate) fn item_field_binding_name(
    prog: &CgProgram,
    entity_ref: u16,
    slot: u16,
    is_item: bool,
) -> String {
    let resolved = if is_item {
        prog.entity_field_catalog
            .resolve_item(crate::cg::data_handle::ItemFieldId {
                entity: entity_ref,
                slot,
                ty: crate::cg::data_handle::AgentFieldTy::U32,
            })
    } else {
        prog.entity_field_catalog
            .resolve_group(crate::cg::data_handle::GroupFieldId {
                entity: entity_ref,
                slot,
                ty: crate::cg::data_handle::AgentFieldTy::U32,
            })
    };
    let prefix = if is_item { "item" } else { "group" };
    match resolved {
        Some((_entity_name, field_name, _)) => {
            format!("{}_{}", prefix, field_name)
        }
        None => {
            format!("{}_{}_{}", prefix, entity_ref, slot)
        }
    }
}

/// True iff `expr` is a `CgExpr::EventField` read — the binder
/// extraction shape that fold-handler bodies produce when they
/// destructure event payload fields like `on Killed { by: predator }`.
/// Used by the per-stmt emit to recognise the per-row index local
/// for downstream `Assign { target: ViewStorage, … }` writes.
fn is_event_field_read(expr: &CgExpr) -> bool {
    matches!(expr, CgExpr::EventField { .. })
}

fn view_slot_token(slot: ViewStorageSlot) -> &'static str {
    match slot {
        ViewStorageSlot::Primary => "primary",
        ViewStorageSlot::Anchor => "anchor",
        ViewStorageSlot::Ids => "ids",
        ViewStorageSlot::Counts => "counts",
        ViewStorageSlot::Cursors => "cursors",
    }
}

fn event_ring_access_token(kind: EventRingAccess) -> &'static str {
    match kind {
        EventRingAccess::Read => "read",
        EventRingAccess::Append => "append",
        EventRingAccess::Drain => "drain",
    }
}

fn spatial_storage_token(kind: SpatialStorageKind) -> &'static str {
    match kind {
        SpatialStorageKind::GridCells => "grid_cells",
        SpatialStorageKind::GridOffsets => "grid_offsets",
        SpatialStorageKind::QueryResults => "query_results",
        SpatialStorageKind::NonemptyCells => "nonempty_cells",
        SpatialStorageKind::NonemptyCellsIndirectArgs => "nonempty_indirect_args",
        SpatialStorageKind::GridStarts => "grid_starts",
        SpatialStorageKind::ChunkSums => "chunk_sums",
    }
}

fn rng_purpose_token(purpose: RngPurpose) -> &'static str {
    // Routes through the canonical snake-case label so adding a new
    // RngPurpose variant requires only one update site (the enum impl).
    purpose.snake()
}

fn agent_scratch_token(kind: AgentScratchKind) -> &'static str {
    match kind {
        AgentScratchKind::Packed => "packed",
    }
}

// ---------------------------------------------------------------------------
// EmitError
// ---------------------------------------------------------------------------

/// Errors a Task-4.1 lowering can raise. Every variant names a typed
/// id — no free-form `String` reasons — so callers can match on the
/// shape of the failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// A [`CgExprId`] reference was past the end of the program's
    /// expression arena.
    ExprIdOutOfRange { id: CgExprId, arena_len: u32 },
    /// A [`CgStmtId`] reference was past the end of the program's
    /// statement arena.
    StmtIdOutOfRange { id: CgStmtId, arena_len: u32 },
    /// A [`CgStmtListId`] reference was past the end of the program's
    /// statement-list arena.
    StmtListIdOutOfRange {
        id: CgStmtListId,
        arena_len: u32,
    },
    /// The active [`HandleNamingStrategy`] does not produce a WGSL name
    /// for `handle`. Today nothing raises this — Task 4.2's slot-aware
    /// strategy will use it for handles that have no slot assignment.
    UnsupportedHandle {
        handle: DataHandle,
        reason: &'static str,
    },
    /// A [`CgExpr::EventField`] referenced an [`EventKindId`] that has
    /// no entry in [`CgProgram::event_layouts`]. The driver populates
    /// the schema in `populate_event_kinds`; a missing entry is a
    /// driver-side defect (or the program was constructed without the
    /// driver). Surfaces as a typed emit error so callers can render
    /// the offending kind id.
    UnregisteredEventKind { kind: EventKindId },
    /// A [`CgExpr::EventField`]'s claimed [`CgTy`] has no WGSL-emit
    /// shape today. The runtime's `pack_event` source-of-truth at
    /// `crates/engine_gpu/src/event_ring.rs` packs every event field
    /// into a closed set of types (`AgentId`, `U32`, `I32`, `F32`,
    /// `Vec3F32`, `Bool`, `Tick`); a `ViewKey<...>` field is structurally
    /// nonsensical and surfaces here. Adding a new event-field type
    /// means adding a matching arm in `lower_cg_expr_to_wgsl`'s
    /// `EventField` branch.
    EventFieldUnsupportedType {
        kind: EventKindId,
        word_offset_in_payload: u32,
        got: CgTy,
    },
    /// A [`CgExpr::NamespaceCall`] referenced an `(ns, method)` pair
    /// that has no entry in [`CgProgram::namespace_registry`]. The
    /// driver populates the registry in `populate_namespace_registry`;
    /// a missing entry is a driver-side defect or a hand-built program
    /// that bypassed the driver. Surfaces as a typed emit error so
    /// callers can render the offending pair.
    UnregisteredNamespaceMethod {
        ns: dsl_ast::ir::NamespaceId,
        method: String,
    },
    /// A [`CgExpr::NamespaceField`] referenced an `(ns, field)` pair
    /// that has no entry in [`CgProgram::namespace_registry`]. Same
    /// failure mode as [`Self::UnregisteredNamespaceMethod`].
    UnregisteredNamespaceField {
        ns: dsl_ast::ir::NamespaceId,
        field: String,
    },
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmitError::ExprIdOutOfRange { id, arena_len } => write!(
                f,
                "CgExprId(#{}) out of range (expr arena holds {} entries)",
                id.0, arena_len
            ),
            EmitError::StmtIdOutOfRange { id, arena_len } => write!(
                f,
                "CgStmtId(#{}) out of range (stmt arena holds {} entries)",
                id.0, arena_len
            ),
            EmitError::StmtListIdOutOfRange { id, arena_len } => write!(
                f,
                "CgStmtListId(#{}) out of range (stmt-list arena holds {} entries)",
                id.0, arena_len
            ),
            EmitError::UnsupportedHandle { handle, reason } => {
                write!(f, "unsupported handle {handle}: {reason}")
            }
            EmitError::UnregisteredEventKind { kind } => write!(
                f,
                "EventField references EventKindId(#{}) with no entry in event_layouts",
                kind.0
            ),
            EmitError::EventFieldUnsupportedType {
                kind,
                word_offset_in_payload,
                got,
            } => write!(
                f,
                "EventField(event#{}, word_off#{}) has no WGSL emit shape for type {}",
                kind.0, word_offset_in_payload, got
            ),
            EmitError::UnregisteredNamespaceMethod { ns, method } => write!(
                f,
                "NamespaceCall references {:?}.{} with no entry in namespace_registry",
                ns, method
            ),
            EmitError::UnregisteredNamespaceField { ns, field } => write!(
                f,
                "NamespaceField references {:?}.{} with no entry in namespace_registry",
                ns, field
            ),
        }
    }
}

impl std::error::Error for EmitError {}

// ---------------------------------------------------------------------------
// Op-symbol mappings
// ---------------------------------------------------------------------------

/// WGSL infix symbol for a [`BinaryOp`]. Per-variant exhaustive — no
/// fallthrough — so adding a new `BinaryOp` variant forces a decision
/// here.
fn binary_op_to_wgsl(op: BinaryOp) -> &'static str {
    use BinaryOp::*;
    match op {
        AddF32 | AddU32 | AddI32 | AddVec3 => "+",
        SubF32 | SubU32 | SubI32 | SubVec3 => "-",
        MulF32 | MulU32 | MulI32 | MulVec3ByF32 => "*",
        DivF32 | DivU32 | DivI32 | DivVec3ByF32 => "/",
        ModF32 | ModU32 | ModI32 => "%",
        LtF32 | LtU32 | LtI32 => "<",
        LeF32 | LeU32 | LeI32 => "<=",
        GtF32 | GtU32 | GtI32 => ">",
        GeF32 | GeU32 | GeI32 => ">=",
        EqBool | EqU32 | EqI32 | EqF32 | EqAgentId => "==",
        NeBool | NeU32 | NeI32 | NeF32 | NeAgentId => "!=",
        And => "&&",
        Or => "||",
        BitOrU32 => "|",
        BitXorU32 => "^",
        BitAndU32 => "&",
    }
}

/// Render `op(arg)` for unary ops. Some unaries are prefix operators
/// (`-x`, `!x`); others are call-form (`abs(x)`, `sqrt(x)`,
/// `normalize(x)`). Returned tag selects the shape so the caller can
/// build the right string.
enum UnaryShape {
    /// `<symbol><arg>` — prefix operator.
    Prefix(&'static str),
    /// `<name>(<arg>)` — function call.
    Call(&'static str),
}

fn unary_op_shape(op: UnaryOp) -> UnaryShape {
    use UnaryOp::*;
    match op {
        NotBool => UnaryShape::Prefix("!"),
        NegF32 | NegI32 => UnaryShape::Prefix("-"),
        AbsF32 | AbsI32 => UnaryShape::Call("abs"),
        SqrtF32 => UnaryShape::Call("sqrt"),
        NormalizeVec3F32 => UnaryShape::Call("normalize"),
    }
}

/// WGSL function name for a [`BuiltinId`]. View calls embed the view
/// id structurally so each view's getter has a stable, distinct name.
///
/// Plan G G3f note: the four `Builtin::Threats*` AST variants do NOT
/// produce a `BuiltinId::Threats*` here. They lower at the CG layer
/// (`cg/lower/expr.rs::lower_builtin_call`) to direct
/// `CgExpr::Lit(...)` sentinels (`false` / `0.0` / `AgentId(0)` /
/// `vec3(0,0,0)`) — the threats materialised view (G3g, future) wires
/// the per-cell walk via the view-call infrastructure; it does not
/// add new [`BuiltinId`] variants. TODO(g3g): once the threats view
/// declares its kernels, the four scoring primitives can rewrite to
/// `BuiltinId::ViewCall { view: <threats view id> }` (with extra
/// per-method aggregation glue) without touching this module.
fn builtin_name(id: BuiltinId) -> String {
    use BuiltinId::*;
    match id {
        Distance => "distance".to_string(),
        PlanarDistance => "planar_distance".to_string(),
        ZSeparation => "z_separation".to_string(),
        // WGSL natives — `length(vec3<f32>) -> f32` and
        // `dot(vec3<f32>, vec3<f32>) -> f32`. No prelude needed.
        LengthVec3F32 => "length".to_string(),
        DotVec3F32 => "dot".to_string(),
        // Vec3 component access — surfaces only as a fallback for
        // pretty-print / unreachable in the WGSL emit (the
        // `CgExpr::Builtin` arm short-circuits these to postfix
        // `(<arg>).x` / `.y` / `.z` before reaching `builtin_name`).
        // The names mirror `BuiltinId::label`.
        Vec3X => "vec3_x".to_string(),
        Vec3Y => "vec3_y".to_string(),
        Vec3Z => "vec3_z".to_string(),
        Min(t) => format!("min_{}", numeric_ty_token(t)),
        Max(t) => format!("max_{}", numeric_ty_token(t)),
        Clamp(t) => format!("clamp_{}", numeric_ty_token(t)),
        SaturatingAdd(t) => format!("saturating_add_{}", numeric_ty_token(t)),
        Floor => "floor".to_string(),
        Ceil => "ceil".to_string(),
        Round => "round".to_string(),
        Ln => "log".to_string(),
        Log2 => "log2".to_string(),
        Log10 => "log10".to_string(),
        Entity => "entity".to_string(),
        ViewCall { view } => format!("view_{}_get", view.0),
        // Plan G G3f follow-up — `threats.nearest(observer)` per-view
        // ring-walk reduction. The helper is emitted by
        // `compose_view_storage_prelude` and returns `u32` (AgentId).
        ThreatsNearest { view } => format!("view_{}_nearest", view.0),
        // Sibling — same argmin walk, returns the unit vec3 pointing
        // away from the closest cell's center.
        ThreatsDirAwayFromNearest { view } => {
            format!("view_{}_dir_away_from_nearest", view.0)
        }
        // Plan H slice 3 — ability registry telegraph reads. Names
        // suffixed `_at` so the body-scan in `cg/emit/kernel.rs` can
        // distinguish the helper substring from the bare binding name
        // `ability_registry_telegraph_kind` (would otherwise match
        // both and double-trigger the BGL inclusion).
        AbilityTelegraphKind => "ability_registry_telegraph_kind_at".to_string(),
        AbilityTelegraphParam0 => "ability_registry_telegraph_param_0_at".to_string(),
        // WGSL has a built-in `vec3<f32>` constructor; emit the call
        // as-is so `vec3(x, y, z)` lowers to `vec3<f32>(x, y, z)`.
        Vec3Ctor => "vec3<f32>".to_string(),
        // WGSL native scalar conversion `f32(<u32-or-i32>)`. Emitted
        // as-is so `AsF32(U32)` / `AsF32(I32)` produce `f32(<arg>)`.
        // The `NumericTy` payload is informational at IR level (it
        // pins the source type for typing); WGSL infers the source
        // from the argument's type.
        AsF32(_) => "f32".to_string(),
        // Sibling explicit casts to `u32` / `i32`. WGSL has native
        // `u32(...)` / `i32(...)` constructors; same emit-as-is
        // treatment as `AsF32`.
        AsU32(_) => "u32".to_string(),
        AsI32(_) => "i32".to_string(),
    }
}

fn numeric_ty_token(t: NumericTy) -> &'static str {
    match t {
        NumericTy::F32 => "f32",
        NumericTy::U32 => "u32",
        NumericTy::I32 => "i32",
    }
}

// ---------------------------------------------------------------------------
// Literal emission
// ---------------------------------------------------------------------------

/// Render an `f32` as a WGSL float literal, matching the legacy
/// `emit_view::format_f32_lit` convention so Phase-5 byte-for-byte
/// parity with the legacy emit path holds.
///
/// Convention (ported locally — does **not** depend on `emit_view.rs`,
/// which is slated for retirement in Task 5.2):
/// 1. Format via `Display` (`{v}`) — gives `"1"` for `1.0`, `"1.5"` for
///    `1.5`, `"0.00001"` for `1e-5`, `"1000000000000000000000000000000"`
///    for `1e30`, and the fully-expanded decimal for sub-normals.
/// 2. If the result already contains `.`, `e`, or `E`, return as-is.
/// 3. Otherwise append `".0"` so WGSL parses the literal as `f32`,
///    not an abstract integer.
///
/// # WGSL syntax notes
///
/// - Integer-valued: `1.0` → `"1.0"`. Round-trip safe.
/// - Sub-unit: `0.5` → `"0.5"`, `-0.5` → `"-0.5"`. Both retain the dot.
/// - Very large: `1e30` → `"1000…0.0"` — a 31-digit literal. Legal WGSL,
///   but ugly; well-formed sim programs do not use literals this large.
/// - Very small: `1e-30` → `"0.000…01"` — a 32-digit literal. Same caveat.
/// - `f32::MIN_POSITIVE` (`~1.175e-38`) — the fully-expanded decimal is
///   45+ characters; well-formed sim programs do not embed it as a literal.
fn format_f32_lit(v: f32) -> String {
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Render a [`LitValue`] as a WGSL constant fragment. `f32` and the
/// three components of `Vec3F32` route through [`format_f32_lit`] so
/// output is byte-identical to the legacy emit path.
fn lower_literal(lit: &LitValue) -> String {
    match lit {
        LitValue::Bool(true) => "true".to_string(),
        LitValue::Bool(false) => "false".to_string(),
        LitValue::U32(v) => format!("{}u", v),
        LitValue::I32(v) => format!("{}i", v),
        LitValue::F32(v) => format_f32_lit(*v),
        // Tick is u32 at the WGSL level — see `CgTy::Tick` doc.
        LitValue::Tick(v) => format!("{}u", v),
        // AgentId is a u32 slot index at the WGSL level.
        LitValue::AgentId(v) => format!("{}u", v),
        LitValue::Vec3F32 { x, y, z } => {
            format!(
                "vec3<f32>({}, {}, {})",
                format_f32_lit(*x),
                format_f32_lit(*y),
                format_f32_lit(*z)
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Expression emission
// ---------------------------------------------------------------------------

/// Lower a single [`CgExpr`] (resolved by id from `ctx.prog`) into a
/// WGSL source fragment.
///
/// # Limitations
///
/// - Walks are pure: no decisions, no kernel boilerplate, no new
///   bindings. Each variant maps to a fixed WGSL form.
/// - `Read` produces the bare handle name (Task 4.2 wraps with the
///   actual buffer indexing form).
/// - `Rng` produces a structural call `per_agent_u32(seed, agent_id, tick, "<purpose>")`;
///   the actual seed/agent/tick arguments are wired by Task 4.2.
/// - `Builtin` emits the WGSL function name from [`builtin_name`];
///   custom helpers (`planar_distance`, `saturating_add_<ty>`,
///   `view_<id>_get`) are assumed to live in the prelude (Task 4.3).
/// - `Select` emits WGSL's `select(false_val, true_val, cond)` shape —
///   note the false-value-first ordering.
///
/// # Errors
///
/// Returns [`EmitError::ExprIdOutOfRange`] if any descendant id is past
/// the end of `ctx.prog.exprs`.
pub fn lower_cg_expr_to_wgsl(expr_id: CgExprId, ctx: &EmitCtx) -> Result<String, EmitError> {
    let arena_len = ctx.prog.exprs.len() as u32;
    let node = <CgProgram as ExprArena>::get(ctx.prog, expr_id).ok_or(
        EmitError::ExprIdOutOfRange {
            id: expr_id,
            arena_len,
        },
    )?;
    match node {
        CgExpr::Read(handle) => {
            // AgentField reads emit an indexed access on the shared
            // SoA binding (`agent_<field>[<index>]`). The index
            // expression depends on the agent-ref:
            //   Self_ → kernel-bound `agent_id`
            //   EventTarget → preamble-bound `event_target_id`
            //   PerPairCandidate → preamble-bound `per_pair_candidate`
            //   Actor → preamble-bound `actor_id`
            //   Target(expr_id) → stmt-scope hoisted `target_expr_<N>`
            //     (see `pending_target_lets` on EmitCtx). The first
            //     reference within a block lowers the target expression
            //     to WGSL, queues a pre-stmt
            //     `let target_expr_<N>: u32 = <wgsl>;` for the enclosing
            //     stmt, and returns `agent_<field>[target_expr_<N>]`.
            //     Subsequent references in the same block reuse the
            //     binding without re-emitting.
            if let DataHandle::AgentField { field, target } = handle {
                if let AgentRef::Target(target_expr_id) = target {
                    // Skip re-binding if the same target expression
                    // has already been hoisted in the surrounding
                    // block. The bound set is cloned + restored at
                    // every stmt-list boundary so inner-scope
                    // bindings can't leak outward.
                    let already_bound = ctx
                        .bound_target_exprs
                        .borrow()
                        .contains(target_expr_id);
                    if !already_bound {
                        // Recursive lowering: the target expression
                        // itself may contain further `Target(_)` reads;
                        // each pushes its own pending entry, all
                        // emitted before the enclosing stmt.
                        let target_wgsl =
                            lower_cg_expr_to_wgsl(*target_expr_id, ctx)?;
                        ctx.pending_target_lets
                            .borrow_mut()
                            .push((*target_expr_id, target_wgsl));
                        ctx.bound_target_exprs
                            .borrow_mut()
                            .insert(*target_expr_id);
                    }
                    return Ok(agent_field_read_wgsl(*field, target, ctx));
                }
                // Tile-walk substitution: when the tiled-MoveBoid emit
                // path is active and we're inside its inner cell-walk
                // loop, every `Pos` / `Vel` read keyed on
                // `PerPairCandidate` redirects to the workgroup-local
                // tile array (`tile_pos[<index>]` / `tile_vel[<index>]`)
                // instead of the global `agent_pos[per_pair_candidate]`.
                // The tile-walk index is set in the inner-loop preamble
                // emitted by `build_tiled_per_cell_wgsl_body` and
                // cleared on exit. Other AgentField targets (Self_,
                // EventTarget, Actor) keep the default global-memory
                // access — only the per-candidate reads benefit from
                // the tile.
                if matches!(target, AgentRef::PerPairCandidate) {
                    if let Some(idx_expr) = ctx.tile_walk_index.borrow().as_ref() {
                        match field {
                            AgentFieldId::Pos => {
                                return Ok(format!("tile_pos[{idx_expr}]"));
                            }
                            AgentFieldId::Vel => {
                                return Ok(format!("tile_vel[{idx_expr}]"));
                            }
                            // Other fields fall through — the tile
                            // only mirrors pos+vel today (the boids
                            // fixture's projections only read those
                            // two via per_pair_candidate). A future
                            // fixture that reads `agent_<other>[
                            // per_pair_candidate]` inside a tiled
                            // ForEachNeighbor would need to extend the
                            // tile arrays; until then the default
                            // global access stays correct (just slow).
                            _ => {}
                        }
                    }
                }
                return Ok(agent_field_read_wgsl(*field, target, ctx));
            }
            // Item / Group fields: emit `item_<field>[<idx>]` (or
            // `group_<field>[<idx>]`). The binding name is field-keyed
            // across all entities of the same root that declare the
            // field (Gap T2 fix, 2026-05-12); the user index `<idx>`
            // is the entity discriminant (position in declaration
            // order among Item-rooted entities). Sourced from the
            // program's `entity_field_catalog` so kernel binding names
            // + body accesses agree on the same identifier (e.g.
            // `item_base_price`). The `<idx>` expression lowers
            // identically to the AgentField `Target(_)` path
            // (recursive lowering with stmt-prefix `let item_target_<N>`
            // hoisting via `pending_target_lets`).
            if let DataHandle::ItemField { field, target } = handle {
                let already_bound = ctx
                    .bound_target_exprs
                    .borrow()
                    .contains(target);
                if !already_bound {
                    let target_wgsl = lower_cg_expr_to_wgsl(*target, ctx)?;
                    ctx.pending_target_lets
                        .borrow_mut()
                        .push((*target, target_wgsl));
                    ctx.bound_target_exprs
                        .borrow_mut()
                        .insert(*target);
                }
                let bind_name = item_field_binding_name(
                    ctx.prog,
                    field.entity,
                    field.slot,
                    /* is_item */ true,
                );
                return Ok(format!("{}[target_expr_{}]", bind_name, target.0));
            }
            if let DataHandle::GroupField { field, target } = handle {
                let already_bound = ctx
                    .bound_target_exprs
                    .borrow()
                    .contains(target);
                if !already_bound {
                    let target_wgsl = lower_cg_expr_to_wgsl(*target, ctx)?;
                    ctx.pending_target_lets
                        .borrow_mut()
                        .push((*target, target_wgsl));
                    ctx.bound_target_exprs
                        .borrow_mut()
                        .insert(*target);
                }
                let bind_name = item_field_binding_name(
                    ctx.prog,
                    field.entity,
                    field.slot,
                    /* is_item */ false,
                );
                return Ok(format!("{}[target_expr_{}]", bind_name, target.0));
            }
            Ok(ctx.handle_name(handle))
        }
        CgExpr::Lit(v) => Ok(lower_literal(v)),
        CgExpr::Binary { op, lhs, rhs, ty: _ } => {
            // Peephole: `distance(a, b) <op> r` where <op> is an
            // ordered comparison rewrites to `dot(d, d) <op> r*r`
            // (where `d = a - b`). Avoids the `sqrt` inside
            // `distance(...)`. Same semantics whenever `r >= 0`,
            // which is the only case sim radii hit (perception /
            // separation radii are always positive). When the peephole
            // doesn't apply we fall through to the generic
            // `(<lhs> <op> <rhs>)` form.
            //
            // The rewrite duplicates `a` and `b` in the emitted
            // expression so the WGSL compiler can CSE them; this is
            // safe as long as both are pure (no side-effects, no
            // mutation between reads). For boids the operands are
            // always `agent_pos[agent_id]` / `agent_pos[per_pair_candidate]`
            // — pure storage reads, trivially CSE-able. We assert
            // pureness via `expr_is_pure_for_hoisting` rather than
            // emitting a let-binding (WGSL has no expression-position
            // let-binding short of a synthetic block, which would
            // break the surrounding statement composition).
            if let Some(rewritten) = try_rewrite_distance_compare(*op, *lhs, *rhs, ctx)? {
                return Ok(rewritten);
            }
            let l = lower_cg_expr_to_wgsl(*lhs, ctx)?;
            let r = lower_cg_expr_to_wgsl(*rhs, ctx)?;
            Ok(format!("({} {} {})", l, binary_op_to_wgsl(*op), r))
        }
        CgExpr::Unary { op, arg, ty: _ } => {
            let a = lower_cg_expr_to_wgsl(*arg, ctx)?;
            match unary_op_shape(*op) {
                UnaryShape::Prefix(sym) => Ok(format!("({}{})", sym, a)),
                UnaryShape::Call(name) => Ok(format!("{}({})", name, a)),
            }
        }
        CgExpr::Builtin { fn_id, args, ty: _ } => {
            let mut parts = Vec::with_capacity(args.len());
            for a in args {
                parts.push(lower_cg_expr_to_wgsl(*a, ctx)?);
            }
            // Gap #A close (stdlib_math_probe, 2026-05-04): WGSL has
            // `log` (natural log) and `log2` natively, but no `log10`.
            // Emit the math identity `log2(x) / log2(10.0)` inline so
            // no kernel prelude is required. `log2(10.0)` is a
            // constant — WGSL's optimiser folds the divisor.
            if matches!(fn_id, BuiltinId::Log10) {
                debug_assert_eq!(parts.len(), 1, "log10 takes one arg");
                return Ok(format!("(log2({}) / log2(10.0))", parts[0]));
            }
            // Vec3 component access is postfix in WGSL — emit
            // `(<arg>).x` / `.y` / `.z` rather than the synthetic
            // `vec3_x(<arg>)` function name (Gap dungeon_stealth#1,
            // 2026-05-12). Mirrors `Log10`'s shape special-case above.
            if let Some(comp) = match fn_id {
                BuiltinId::Vec3X => Some("x"),
                BuiltinId::Vec3Y => Some("y"),
                BuiltinId::Vec3Z => Some("z"),
                _ => None,
            } {
                debug_assert_eq!(parts.len(), 1, "vec3 component takes one arg");
                return Ok(format!("({}).{}", parts[0], comp));
            }
            Ok(format!("{}({})", builtin_name(*fn_id), parts.join(", ")))
        }
        CgExpr::Rng { purpose, extra, ty: _ } => {
            // `per_agent_u32(seed, agent_id, tick, <purpose_id>u)` —
            // calls the WGSL prelude function emitted by
            // [`super::program::compose_rng_prelude`] when any kernel
            // body references `per_agent_u32(`. `seed` / `agent_id` /
            // `tick` are bound by `thread_indexing_preamble`; the
            // purpose is a stable numeric id from
            // `RngPurpose::wgsl_id()` (WGSL has no string literals —
            // stochastic_probe Gap #3, 2026-05-04).
            //
            // Gap #D close (stdlib_math_probe, 2026-05-04): the
            // typed-RNG `Coin` purpose carries `CgTy::Bool` per the
            // typed-RNG invariant in `data_handle.rs` (purpose →
            // CgTy). The bare `per_agent_u32(...)` call returns u32 —
            // assigning it into a `let local_N: bool = ...` fails the
            // naga text parser ("expected `bool`, but got `u32`").
            // Wrap the u32 draw in a bit-extract `(... & 1u) == 0u`
            // so the expression types as bool.
            //
            // Gap #E close (stdlib_math_probe, 2026-05-04): the
            // typed-RNG `Uniform` / `Gauss` purposes carry `CgTy::F32`.
            // The surrounding lowering builds an f32-arithmetic
            // wrapper (`lo + draw * (hi - lo)` for Uniform; `mu +
            // draw * sigma` for Gauss). If `draw` emits as a bare
            // `per_agent_u32(...)` u32 expression, the surrounding
            // `0.0` / `1.0` literals stay abstract-floats — the wgpu
            // FULL validator rejects "Abstract types may only appear
            // in constant expressions". Per-purpose conversion at
            // THIS site (the expression emit) makes the whole
            // subexpression concretely-typed f32:
            //   - Uniform → `(f32(per_agent_u32(...)) / 4294967295.0)`
            //     yields a unit-interval `f32` (in `[0, 1]`); the
            //     surrounding `lo + draw * (hi - lo)` is then pure
            //     f32 arithmetic.
            //   - Gauss → standard Box-Muller pair-draw using two
            //     independent streams (`Gauss` purpose + `GaussB`
            //     purpose at id 9). Computes `sqrt(-2*log(u1)) *
            //     cos(2π*u2)` for a unit-normal `f32`. `max(u1, 1e-9)`
            //     guards against `log(0) = -inf`.
            //   - UniformInt → bare u32 (post Gap #C the surface IS
            //     u32; no bitcast needed).
            // Three-arm RNG emit:
            //   1. Outside loops, extra==0 → bare `per_agent_u32(...)`
            //      (preserves every existing fixture's RNG stream).
            //   2. Outside loops, extra>0 → `per_agent_u32_with_extra(
            //      ..., extra_const)` for multi-same-purpose calls
            //      in one rule body (fixes maze_explorer_smart's
            //      `rng.action()` collision).
            //   3. Inside a ForEach* loop body → `per_agent_u32_with_extra(
            //      ..., loop_iter_var + extra_const)` so each
            //      iteration draws from a distinct stream. Closes
            //      the forest_fire-style bug where every neighbour
            //      candidate saw the same `rng.action()` draw.
            //
            // The loop-iter var (e.g. `per_pair_candidate`) is set by
            // `emit_for_each_{neighbor,agent}_body` before lowering
            // the body, and cleared on exit.
            let loop_var = ctx.rng_loop_iter_var.borrow().clone();
            let raw = match (loop_var, *extra) {
                (None, 0) => format!(
                    "per_agent_u32(seed, agent_id, tick, {}u)",
                    purpose.wgsl_id()
                ),
                (None, e) => format!(
                    "per_agent_u32_with_extra(seed, agent_id, tick, {}u, {}u)",
                    purpose.wgsl_id(),
                    e
                ),
                (Some(var), 0) => format!(
                    "per_agent_u32_with_extra(seed, agent_id, tick, {}u, {})",
                    purpose.wgsl_id(),
                    var
                ),
                (Some(var), e) => format!(
                    "per_agent_u32_with_extra(seed, agent_id, tick, {}u, {} + {}u)",
                    purpose.wgsl_id(),
                    var,
                    e
                ),
            };
            match purpose {
                RngPurpose::Coin => Ok(format!("(({} & 1u) == 0u)", raw)),
                RngPurpose::Uniform => {
                    // Cast u32 to f32 then normalise to `[0, 1]` by
                    // dividing by `u32::MAX as f32`. The divisor
                    // literal carries an explicit `f32(...)` cast so
                    // it's a concrete f32, not an abstract-float.
                    Ok(format!("(f32({}) / f32(4294967295u))", raw))
                }
                RngPurpose::Gauss => {
                    // Box-Muller pair-draw — see the prelude doc on
                    // `RngPurpose::GaussB` and the gap report.
                    // `Gauss` (purpose 6) is u1; `GaussB` (purpose 9)
                    // is u2. The `max(..., 1e-9)` guards against
                    // `log(0) = -inf` if `u1 == 0`. The constant
                    // `6.283185307179586` is `2π` to ~17 digits so
                    // f32 truncation lands on the nearest
                    // representable value.
                    let raw_b = format!(
                        "per_agent_u32(seed, agent_id, tick, {}u)",
                        RngPurpose::GaussB.wgsl_id()
                    );
                    Ok(format!(
                        "(sqrt(-2.0 * log(max(f32({}) / f32(4294967295u), 1e-9))) \
                         * cos(6.283185307179586 * (f32({}) / f32(4294967295u))))",
                        raw, raw_b
                    ))
                }
                _ => Ok(raw),
            }
        }
        CgExpr::Select {
            cond,
            then,
            else_,
            ty: _,
        } => {
            let c = lower_cg_expr_to_wgsl(*cond, ctx)?;
            let t = lower_cg_expr_to_wgsl(*then, ctx)?;
            let e = lower_cg_expr_to_wgsl(*else_, ctx)?;
            // WGSL's `select(false_val, true_val, cond)` — note the
            // false-value-first order.
            Ok(format!("select({}, {}, {})", e, t, c))
        }
        // Bare actor / candidate id reads — emit the kernel-local
        // identifier the surrounding template binds. The MaskPredicate
        // PerAgent template binds `agent_id`; the PerPair template
        // binds `per_pair_candidate`. Naming is kept in sync with the
        // existing AgentRef tokens (wgsl_body.rs `agent_ref_token`).
        CgExpr::AgentSelfId => Ok("agent_id".to_string()),
        CgExpr::PerPairCandidateId => Ok("per_pair_candidate".to_string()),
        // Let-bound local — emit the `let local_<N>: <ty> = ...;` name
        // produced by `CgStmt::Let` emission.
        CgExpr::ReadLocal { local, ty: _ } => Ok(format!("local_{}", local.0)),
        // Schema-driven access into the current event's payload. The
        // surrounding PerEvent kernel template binds `event_idx` and
        // selects `event_ring` (today the shared ring; future per-kind
        // ring fanout swaps `buffer_name` per-kind without touching
        // this emit shape). See `CgExpr::EventField` docs for the
        // forward-compat contract.
        CgExpr::EventField {
            event_kind,
            word_offset_in_payload,
            ty,
        } => {
            let layout = ctx.prog.event_layouts.get(&event_kind.0).ok_or(
                EmitError::UnregisteredEventKind {
                    kind: *event_kind,
                },
            )?;
            let total_offset = layout.header_word_count + word_offset_in_payload;
            let buf = layout.buffer_name.as_str();
            let stride = layout.record_stride_u32;
            // PerEventEmit kernels declare `event_ring` as
            // `array<atomic<u32>>` so the body's `Emit`-side
            // `atomicStore` type-checks; in that mode every read of a
            // payload word also has to go through `atomicLoad` (WGSL
            // forbids non-atomic indexing on an atomic-typed binding).
            // ViewFold's path keeps this `false` (its `event_ring`
            // binding stays plain `array<u32>`), so the existing
            // plain-index reads continue to compile.
            // Serial fold body uses `_ei` (the per-op inner scan loop
            // variable) instead of the kernel-preamble `event_idx`.
            let idx_var = if ctx.in_serial_fold_body.get() { "_ei" } else { "event_idx" };
            let read_word = |off: u32| -> String {
                if ctx.event_ring_atomic_loads.get() {
                    format!("atomicLoad(&{}[{} * {}u + {}u])", buf, idx_var, stride, off)
                } else {
                    format!("{}[{} * {}u + {}u]", buf, idx_var, stride, off)
                }
            };
            Ok(match ty {
                CgTy::AgentId | CgTy::U32 | CgTy::Tick => read_word(total_offset),
                CgTy::I32 => format!("bitcast<i32>({})", read_word(total_offset)),
                CgTy::F32 => format!("bitcast<f32>({})", read_word(total_offset)),
                CgTy::Vec3F32 => format!(
                    "vec3<f32>(bitcast<f32>({}), bitcast<f32>({}), bitcast<f32>({}))",
                    read_word(total_offset),
                    read_word(total_offset + 1),
                    read_word(total_offset + 2),
                ),
                CgTy::Bool => format!("({} != 0u)", read_word(total_offset)),
                CgTy::ViewKey { .. } => {
                    return Err(EmitError::EventFieldUnsupportedType {
                        kind: *event_kind,
                        word_offset_in_payload: *word_offset_in_payload,
                        got: *ty,
                    });
                }
            })
        }
        // Schema-driven stdlib namespace-method call (e.g.
        // `agents.is_hostile_to(target)`). The kernel composer prepends
        // a B1-stub prelude function for each `(ns, method)` referenced
        // by the kernel body; here we just emit the function call.
        CgExpr::NamespaceCall {
            ns,
            method,
            args,
            ty: _,
        } => {
            let def = ctx
                .prog
                .namespace_registry
                .namespaces
                .get(ns)
                .and_then(|nd| nd.methods.get(method))
                .ok_or(EmitError::UnregisteredNamespaceMethod {
                    ns: *ns,
                    method: method.clone(),
                })?;
            let mut parts = Vec::with_capacity(args.len());
            for a in args {
                parts.push(lower_cg_expr_to_wgsl(*a, ctx)?);
            }
            Ok(format!("{}({})", def.wgsl_fn_name, parts.join(", ")))
        }
        // Schema-driven stdlib namespace-field read (e.g. `world.tick`).
        // Resolves to either a kernel-preamble local or a uniform-bound
        // field per the registered `WgslAccessForm`.
        CgExpr::NamespaceField { ns, field, ty: _ } => {
            let def = ctx
                .prog
                .namespace_registry
                .namespaces
                .get(ns)
                .and_then(|nd| nd.fields.get(field))
                .ok_or(EmitError::UnregisteredNamespaceField {
                    ns: *ns,
                    field: field.clone(),
                })?;
            Ok(match &def.wgsl_access {
                crate::cg::program::WgslAccessForm::PreambleLocal { local_name } => {
                    local_name.clone()
                }
                crate::cg::program::WgslAccessForm::UniformField { binding, field } => {
                    format!("{}.{}", binding, field)
                }
            })
        }
        // Static lookup table read — `tables.<name>(<idx>)`. Emit a
        // module-level `const <name>: array<u32, N> = …;` declaration
        // (via the side-channel `tables_referenced` set picked up by
        // `compose_wgsl_file`); the body just indexes the const.
        CgExpr::TableLookup { name, index, .. } => {
            ctx.tables_referenced
                .borrow_mut()
                .insert(name.clone());
            let idx_wgsl = lower_cg_expr_to_wgsl(*index, ctx)?;
            Ok(format!("{name}[{idx_wgsl}]"))
        }
    }
}

// ---------------------------------------------------------------------------
// Statement emission
// ---------------------------------------------------------------------------

/// Indent every line of `s` by `indent` four-space levels — matches
/// the convention used throughout the legacy emit path
/// (`emit_view_wgsl.rs`, etc.) so Phase-5 parity holds without
/// whitespace drift.
fn indent_block(s: &str, indent: usize) -> String {
    let prefix: String = "    ".repeat(indent);
    s.lines()
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{}{}", prefix, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Lower a single [`CgStmt`] into a WGSL source fragment. The output
/// contains no leading indentation — the caller composes it with its
/// surrounding context.
///
/// # Limitations
///
/// - `Assign` produces `<target> = <value>;` using the active naming
///   strategy for the target.
/// - `Emit` produces a placeholder call form
///   `emit_event_<N>(field_<I>: <expr>, ...);`. Task 4.2 wires the
///   actual ring-append shape.
/// - `If` emits `if (...) { ... }` (or `if (...) { ... } else { ... }`)
///   using brace-and-newline structure.
/// - `Match` emits an `if`-chain over each arm's variant tag — see
///   the module-level limitations note.
///
/// # Errors
///
/// Returns one of [`EmitError::ExprIdOutOfRange`],
/// [`EmitError::StmtIdOutOfRange`], or
/// [`EmitError::StmtListIdOutOfRange`] for any dangling id.
pub fn lower_cg_stmt_to_wgsl(stmt_id: CgStmtId, ctx: &EmitCtx) -> Result<String, EmitError> {
    // Snapshot the pending-target-let buffer length so we can detect
    // entries pushed *during this stmt's expression sub-tree* and
    // drain them as the stmt's pre-bindings. Entries already in the
    // buffer at entry belong to a caller's stmt and must not be
    // consumed here. See `EmitCtx::pending_target_lets` doc.
    let snapshot_len = ctx.pending_target_lets.borrow().len();
    let body = lower_cg_stmt_body_to_wgsl(stmt_id, ctx)?;
    let mut pending = ctx.pending_target_lets.borrow_mut();
    if pending.len() == snapshot_len {
        return Ok(body);
    }
    let new_lets: Vec<(CgExprId, String)> = pending.drain(snapshot_len..).collect();
    drop(pending);
    let lets_wgsl: String = new_lets
        .iter()
        .map(|(id, w)| format!("let target_expr_{}: u32 = {};", id.0, w))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(format!("{}\n{}", lets_wgsl, body))
}

/// Inner per-stmt lowering. Produces the raw WGSL fragment for the
/// stmt body without the cross-agent target pre-bindings — those are
/// drained + prepended by the public [`lower_cg_stmt_to_wgsl`]
/// wrapper.
fn lower_cg_stmt_body_to_wgsl(
    stmt_id: CgStmtId,
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    let arena_len = ctx.prog.stmts.len() as u32;
    let node = <CgProgram as StmtArena>::get(ctx.prog, stmt_id).ok_or(
        EmitError::StmtIdOutOfRange {
            id: stmt_id,
            arena_len,
        },
    )?;
    match node {
        CgStmt::Assign { target, value } => {
            // B1 no-op fallback for ViewStorage assigns: the structural
            // name `view_<id>_<slot>` isn't a declared binding (the
            // BGL-bound name is `view_storage_<slot>`, indexed by
            // target_id which the structural strategy can't synthesize).
            // Path B's slot-aware lowering produces the real
            // `view_storage_primary[target_id] += value` form. For B1
            // we evaluate the RHS as a phony WGSL discard so the body
            // parses; for trivial fixtures the fold loop is empty so
            // this never runs.
            if let DataHandle::ViewStorage { view, slot } = target {
                let rhs = lower_cg_expr_to_wgsl(*value, ctx)?;
                // When the surrounding stmt list captured per-row
                // index locals (e.g. `Let local_<N> = EventField(by,
                // AgentId)`), emit the accumulator add directly:
                // `view_storage_<slot>[<idx>] = view_storage_<slot>[
                // <idx>] + rhs`. Without index locals fall back to a
                // phony discard for now — non-fold callers (e.g.
                // driver tests) drive Assign-to-ViewStorage in shapes
                // that don't surface a binder yet.
                //
                // The index expression depends on the view's storage
                // hint (looked up via
                // `prog.view_signatures[view].storage_hint`):
                //
                // - PairMap with 2+ AgentId binders: `local_<first> *
                //   cfg.second_key_pop + local_<second>`. Composes a
                //   2-D pair index so each (k1, k2) slot accumulates
                //   independently. The runtime supplies the
                //   second-key population through cfg.second_key_pop
                //   (= agent_cap for Agent×Agent, item count for
                //   Agent×Item, …).
                // - Otherwise: `local_<last>` — single-key shape,
                //   matches the legacy emit that all single-key
                //   views (kill_count, threat_level, …) ship with.
                let locals = ctx.view_target_locals.borrow();
                if !locals.is_empty() {
                    let storage = format!(
                        "view_storage_{}",
                        view_slot_token(*slot),
                    );
                    let storage_hint = ctx
                        .prog
                        .view_signatures
                        .get(&view.0)
                        .and_then(|sig| sig.storage_hint);
                    let is_pair_map = matches!(
                        storage_hint,
                        Some(crate::cg::program::CgStorageHint::PairMap)
                    );
                    let idx_expr = if is_pair_map && locals.len() >= 2 {
                        format!(
                            "(local_{} * cfg.second_key_pop + local_{})",
                            locals[0], locals[1]
                        )
                    } else {
                        // Single-key: index by the LAST AgentId binder
                        // (mirrors the pre-fix shape — every shipped
                        // single-key view's fold body binds a single
                        // event-row key like `by` or `actor`).
                        format!("local_{}", locals[locals.len() - 1])
                    };
                    // The view storage binding is
                    // `array<atomic<u32>>` (see
                    // build_view_fold_bindings); the per-element
                    // semantics depend on the view's declared
                    // `result` type from `view_signatures`:
                    //   - `f32` (most shipped views): the
                    //     accumulator-add is racy under contention
                    //     (multiple GPU threads writing the same
                    //     slot per tick) so we emit a CAS loop —
                    //     atomicLoad → bitcast<f32> → add rhs →
                    //     bitcast<u32> → atomicCompareExchangeWeak,
                    //     retrying on the weak-CAS failure path.
                    //     Satisfies P11 (Reduction Determinism) at
                    //     the cost of a per-thread spin under heavy
                    //     contention.
                    //   - `u32` (Theory-of-Mind `beliefs` view):
                    //     bit-OR accumulator. WGSL's native
                    //     `atomicOr` is commutative + associative so
                    //     no CAS retry is needed — one atomic op per
                    //     event, no per-thread spin. P11 (Reduction
                    //     Determinism) is satisfied trivially.
                    //   - other element types: not supported yet;
                    //     fall through to the f32 CAS shape and the
                    //     well-formed pass would have rejected the
                    //     fold body before reaching emit if the
                    //     types didn't line up.
                    let sig = ctx.prog.view_signatures.get(&view.0);
                    let view_result_ty = sig.map(|s| s.result);
                    let fold_op = sig.and_then(|s| s.fold_op);
                    // Branch on (fold_op, result_ty). Pre-fix
                    // (Gap C — `docs/superpowers/notes/2026-05-04-
                    // quest_probe.md`) the emitter branched on
                    // result_ty alone, so `+= 1u` on a u32 view
                    // silently routed through `atomicOr` (idempotent
                    // — every emit left the slot at `1u`). The
                    // operator is now snapshotted onto
                    // `ViewSignature::fold_op` at lower time so this
                    // branch can pick the right primitive:
                    //
                    //   - `Or`  + u32 → atomicOr (commutative + assoc).
                    //   - `Add` + u32 → atomicAdd (commutative + assoc).
                    //   - `Add` + f32 → CAS+add loop (P11 via retry).
                    //   - `None` (structural-strategy programs that
                    //     bypass the view-body lowerer) falls back to
                    //     the legacy result-type branch — u32 routes
                    //     through atomicOr (pre-fix shape). Today
                    //     this only matters for the test builder
                    //     paths that synthesize Assigns directly.
                    let use_atomic_or = matches!(
                        view_result_ty,
                        Some(crate::cg::expr::CgTy::U32)
                    ) && match fold_op {
                        Some(crate::cg::program::ViewFoldOp::Or) => true,
                        Some(crate::cg::program::ViewFoldOp::Add) => false,
                        Some(crate::cg::program::ViewFoldOp::Sub) => false,
                        Some(crate::cg::program::ViewFoldOp::Set) => false,
                        None => true,
                    };
                    let use_atomic_add = matches!(
                        view_result_ty,
                        Some(crate::cg::expr::CgTy::U32)
                    ) && matches!(
                        fold_op,
                        Some(crate::cg::program::ViewFoldOp::Add)
                    );
                    // `self -= rhs` mirrors `self += rhs`'s emit shape.
                    // u32 routes through native `atomicSub` (commutative
                    // + associative under modular arithmetic — P11
                    // trivially satisfied, no CAS retry). f32 falls
                    // through to the explicit CAS+sub branch below
                    // (the trailing CAS+add fallthrough only applies to
                    // Add+f32; Sub+f32 needs the matching subtract op
                    // inside the loop body).
                    let use_atomic_sub = matches!(
                        view_result_ty,
                        Some(crate::cg::expr::CgTy::U32)
                    ) && matches!(
                        fold_op,
                        Some(crate::cg::program::ViewFoldOp::Sub)
                    );
                    let use_cas_sub_f32 = matches!(
                        view_result_ty,
                        Some(crate::cg::expr::CgTy::F32)
                    ) && matches!(
                        fold_op,
                        Some(crate::cg::program::ViewFoldOp::Sub)
                    );
                    // `self = rhs` lowers to `atomicStore` regardless
                    // of element type — for u32 the store is a single
                    // atomic op, for f32 we bitcast the rhs to u32 bits
                    // first (the storage binding is `array<atomic<u32>>`).
                    // Last-writer-wins semantics: the DSL author owns
                    // determinism (idempotent constant set or
                    // single-writer-per-slot per tick).
                    let use_atomic_store = matches!(
                        fold_op,
                        Some(crate::cg::program::ViewFoldOp::Set)
                    );
                    if use_atomic_or {
                        return Ok(format!(
                            "{{\n\
                             \x20   let _idx = {idx_expr};\n\
                             \x20   atomicOr(&{storage}[_idx], ({rhs}));\n\
                             }}"
                        ));
                    }
                    if use_atomic_add {
                        return Ok(format!(
                            "{{\n\
                             \x20   let _idx = {idx_expr};\n\
                             \x20   atomicAdd(&{storage}[_idx], ({rhs}));\n\
                             }}"
                        ));
                    }
                    if use_atomic_sub {
                        return Ok(format!(
                            "{{\n\
                             \x20   let _idx = {idx_expr};\n\
                             \x20   atomicSub(&{storage}[_idx], ({rhs}));\n\
                             }}"
                        ));
                    }
                    if use_cas_sub_f32 {
                        // f32 has no native atomicSub in WGSL — the
                        // storage binding is `array<atomic<u32>>` and
                        // we round-trip through bitcast. Mirrors the
                        // CAS+add f32 fallthrough below but with `-`
                        // inside the new_val computation.
                        return Ok(format!(
                            "loop {{\n\
                             \x20   let _idx = {idx_expr};\n\
                             \x20   let old = atomicLoad(&{storage}[_idx]);\n\
                             \x20   let new_val = bitcast<u32>(bitcast<f32>(old) - ({rhs}));\n\
                             \x20   let result = atomicCompareExchangeWeak(&{storage}[_idx], old, new_val);\n\
                             \x20   if (result.exchanged) {{ break; }}\n\
                             }}"
                        ));
                    }
                    if use_atomic_store {
                        // Bitcast the rhs to u32 bits when the view
                        // result type is f32 (the storage binding is
                        // `array<atomic<u32>>`); for u32 results pass
                        // the rhs through directly.
                        let stored = if matches!(
                            view_result_ty,
                            Some(crate::cg::expr::CgTy::F32)
                        ) {
                            format!("bitcast<u32>(({rhs}))")
                        } else {
                            format!("({rhs})")
                        };
                        return Ok(format!(
                            "{{\n\
                             \x20   let _idx = {idx_expr};\n\
                             \x20   atomicStore(&{storage}[_idx], {stored});\n\
                             }}"
                        ));
                    }
                    // Serial fold path: each thread owns its observer_slot
                    // and accumulates matching events into a local `accum`
                    // var declared in the surrounding scan loop.  No atomics
                    // needed — single writer per slot by construction.
                    if ctx.in_serial_fold_body.get() {
                        return Ok(format!(
                            "if ({idx_expr} == observer_slot) {{\n\
                             \x20   accum = accum + ({rhs});\n\
                             }}"
                        ));
                    }
                    return Ok(format!(
                        "loop {{\n\
                         \x20   let _idx = {idx_expr};\n\
                         \x20   let old = atomicLoad(&{storage}[_idx]);\n\
                         \x20   let new_val = bitcast<u32>(bitcast<f32>(old) + ({rhs}));\n\
                         \x20   let result = atomicCompareExchangeWeak(&{storage}[_idx], old, new_val);\n\
                         \x20   if (result.exchanged) {{ break; }}\n\
                         }}"
                    ));
                }
                return Ok(format!("_ = ({});", rhs));
            }
            // AgentField writes emit indexed access on the shared SoA
            // binding (`agent_<field>[<index>] = <value>`). See the
            // matching Read arm above for the agent-ref → index map.
            // Target(expr_id) writes go through the same stmt-scope
            // pre-binding as reads (`pending_target_lets`), so
            // `agents.set_<field>(other, value)` becomes
            // `agent_<field>[target_expr_<N>] = <value>;` with the
            // target index hoisted to a stmt-prefix `let`.
            if let DataHandle::AgentField { field, target: agent_ref } = target {
                // AtomicCAS guard: when the kernel has been
                // marked as needing atomic alive-writes (the kernel
                // emit's body scan found the
                // `Assign(Alive, _, Lit(Bool(false)))` pattern and
                // upgraded `agent_alive` to AtomicStorage), every
                // such pattern in this body lowers to
                // `atomicCompareExchangeWeak(&agent_alive[t], 1u, 0u)`
                // captured in a stable per-stmt local
                // `_alive_cas_<stmt_id>`. The surrounding
                // `lower_cg_stmt_list_to_wgsl` reads the same flag
                // and wraps subsequent stmts in
                // `if (_alive_cas_<stmt_id>.exchanged) { ... }` so
                // only the thread that won the transition runs the
                // post-kill side effects (e.g. `emit Defeated`).
                let is_alive_false_cas = ctx.alive_atomic_writes.get()
                    && matches!(field, AgentFieldId::Alive)
                    && matches!(
                        <CgProgram as ExprArena>::get(ctx.prog, *value),
                        Some(CgExpr::Lit(LitValue::Bool(false)))
                    );
                let rhs = lower_cg_expr_to_wgsl(*value, ctx)?;
                if let AgentRef::Target(target_expr_id) = agent_ref {
                    let already_bound = ctx
                        .bound_target_exprs
                        .borrow()
                        .contains(target_expr_id);
                    if !already_bound {
                        let target_wgsl =
                            lower_cg_expr_to_wgsl(*target_expr_id, ctx)?;
                        ctx.pending_target_lets
                            .borrow_mut()
                            .push((*target_expr_id, target_wgsl));
                        ctx.bound_target_exprs
                            .borrow_mut()
                            .insert(*target_expr_id);
                    }
                }
                if is_alive_false_cas {
                    // The lvalue index expression — same as the
                    // generic write below (e.g. `target_expr_<N>` for
                    // AgentRef::Target, `agent_id` for Self_, etc.).
                    let idx = match agent_ref {
                        AgentRef::Self_ => "agent_id".to_string(),
                        AgentRef::EventTarget => "event_target_id".to_string(),
                        AgentRef::Actor => "actor_id".to_string(),
                        AgentRef::PerPairCandidate => "per_pair_candidate".to_string(),
                        AgentRef::Target(id) => format!("target_expr_{}", id.0),
                    };
                    // WGSL forbids user identifiers starting with `__`,
                    // so we let the type be inferred — the call returns
                    // `__atomic_compare_exchange_result<T, AS>` and naga
                    // infers it from the call expression. The
                    // `.exchanged` field access on the result remains
                    // valid.
                    return Ok(format!(
                        "let _alive_cas_{} = atomicCompareExchangeWeak(\
                         &agent_alive[{}], 1u, 0u);",
                        stmt_id.0, idx,
                    ));
                }
                // f32 RMW atomic CAS loop. When the active kernel
                // upgraded `agent_<f>` to `array<atomic<u32>>` because
                // its body contains an `Assign(AgentField{f32, …}, …)`
                // (chronicle-consumer RMW write), every such Assign
                // lowers to a CAS loop on `bitcast<u32>(value)`.
                // ReadLocal substitution via `EmitCtx::inline_locals`
                // (populated by `lower_cg_stmt_list_to_wgsl` before the
                // loop emit) inlines any preceding Lets that read the
                // same field, so each loop iteration recomputes the
                // value-chain against the latest `atomicLoad`. The CAS
                // succeeds only when no other thread interfered between
                // the snapshot and the write — guaranteeing the final
                // SoA value reflects every concurrent decrement /
                // increment for the same target slot. P5 fix.
                //
                // P11 caveat: float associativity / commutativity is
                // broken only by floating-point rounding, so the order
                // of contributions can change the low-order bits of the
                // final value across re-runs. For typical HP-drain
                // magnitudes the rounding error is well below 1 unit;
                // strict bit-equal cross-run determinism would require a
                // chronicle pre-sort step (deferred to a separate
                // slice). The wave_defense `same_seed_same_death_tick`
                // pin holds because the death tick is robust to
                // sub-unit float drift — the alive_cas (above)
                // serializes the kill transition, so the chronicle
                // record's `tick` is deterministic.
                let is_f32_atomic_rmw = matches!(field.ty(), AgentFieldTy::F32)
                    && f32_field_atomic_bit(*field).map_or(false, |bit| {
                        (ctx.f32_atomic_field_writes.get() >> bit) & 1 == 1
                    });
                if is_f32_atomic_rmw {
                    let idx = match agent_ref {
                        AgentRef::Self_ => "agent_id".to_string(),
                        AgentRef::EventTarget => "event_target_id".to_string(),
                        AgentRef::Actor => "actor_id".to_string(),
                        AgentRef::PerPairCandidate => "per_pair_candidate".to_string(),
                        AgentRef::Target(id) => format!("target_expr_{}", id.0),
                    };
                    let snake = field.snake();
                    // When the enclosing `If` arm marked this Assign
                    // as a "first-writer-wins"
                    // candidate (literal-RHS f32 write inside an If
                    // whose cond reads the same field), emit the
                    // CAS-loop variant that captures the transition
                    // outcome in `_f32_cas_did_transition_<sid>` —
                    // declared OUTSIDE the loop so the surrounding
                    // stmt-list emit can wrap subsequent stmts in
                    // `if (_f32_cas_did_transition_<sid>) { ... }`.
                    // The transition predicate is `_old_bits !=
                    // _new_bits`, which is `false` for CAS-losers
                    // that retried, observed the already-written
                    // constant, and stored the same constant
                    // (a no-op transition) — exactly the threads
                    // whose post-write side-effects (e.g. `emit
                    // Ignited`) must NOT fire under the first-
                    // writer-wins contract.
                    //
                    // Non-gated path (gate is `None` or refers to a
                    // different stmt id): emit the bit-identical
                    // legacy loop, preserving `memory_ordering_cas_emit`'s
                    // exact-substring assertion on `.exchanged) { break; }`
                    // for the chronicle-damage shape.
                    let gate_matches = ctx.f32_first_writer_gate.get() == Some(stmt_id.0);
                    if gate_matches {
                        return Ok(format!(
                            "var _f32_cas_did_transition_{sid}: bool = false;\n\
                             loop {{\n\
                             \x20   let _old_bits_{sid} = atomicLoad(&agent_{snake}[{idx}]);\n\
                             \x20   let _new_bits_{sid} = bitcast<u32>({rhs});\n\
                             \x20   let _r_{sid} = atomicCompareExchangeWeak(&agent_{snake}[{idx}], _old_bits_{sid}, _new_bits_{sid});\n\
                             \x20   if (_r_{sid}.exchanged) {{\n\
                             \x20       _f32_cas_did_transition_{sid} = (_old_bits_{sid} != _new_bits_{sid});\n\
                             \x20       break;\n\
                             \x20   }}\n\
                             }}",
                            sid = stmt_id.0,
                            snake = snake,
                            idx = idx,
                            rhs = rhs,
                        ));
                    }
                    return Ok(format!(
                        "loop {{\n\
                         \x20   let _old_bits_{sid} = atomicLoad(&agent_{snake}[{idx}]);\n\
                         \x20   let _new_bits_{sid} = bitcast<u32>({rhs});\n\
                         \x20   let _r_{sid} = atomicCompareExchangeWeak(&agent_{snake}[{idx}], _old_bits_{sid}, _new_bits_{sid});\n\
                         \x20   if (_r_{sid}.exchanged) {{ break; }}\n\
                         }}",
                        sid = stmt_id.0,
                        snake = snake,
                        idx = idx,
                        rhs = rhs,
                    ));
                }
                // LHS uses the raw indexed access (no `(x != 0u)`
                // coercion — that wrapper is not a valid lvalue). For
                // bool fields the RHS must be coerced to u32 since
                // the storage is `array<u32>`.
                let lhs = agent_field_access_lvalue(*field, agent_ref);
                let coerced_rhs = match field.ty() {
                    AgentFieldTy::Bool => format!("select(0u, 1u, {rhs})"),
                    _ => rhs,
                };
                return Ok(format!("{} = {};", lhs, coerced_rhs));
            }
            let lhs = ctx.handle_name(target);
            let rhs = lower_cg_expr_to_wgsl(*value, ctx)?;
            Ok(format!("{} = {};", lhs, rhs))
        }
        CgStmt::Emit { event, fields } => lower_emit_to_wgsl(event.0, fields, ctx),
        CgStmt::If { cond, then, else_ } => {
            let c = lower_cg_expr_to_wgsl(*cond, ctx)?;
            // Detect the "first-writer-wins" shape
            // — an inner Assign of a literal F32 to an f32 SoA column
            // whose field is also read by THIS If's cond. If matched,
            // set the per-stmt gate to the inner Assign's stmt_id so
            // its CAS-loop emit captures the transition outcome AND
            // the inner stmt-list emit knows to wrap subsequent
            // stmts. Restored after the inner-list lowering returns
            // so the gate doesn't leak to sibling Ifs.
            //
            // The eligibility check requires the f32 RMW upgrade
            // (`f32_atomic_field_writes`) to be active for the
            // assigned field — without it the inner Assign would
            // emit as a plain `agent_<f>[idx] = …` write (no CAS,
            // no transition predicate possible). The forest_fire
            // Catch handler trips the upgrade because its body
            // contains `agents.set_hp(t, 99.0)`, which
            // `stmt_list_collect_f32_atomic_writes` records.
            let gate_save = ctx.f32_first_writer_gate.get();
            if let Some((assign_sid, field)) =
                stmt_list_first_f32_const_assign(ctx.prog, *then)
            {
                let upgraded = f32_field_atomic_bit(field).map_or(false, |bit| {
                    (ctx.f32_atomic_field_writes.get() >> bit) & 1 == 1
                });
                if upgraded && expr_reads_agent_field_id(*cond, field, ctx.prog) {
                    // Only set the gate if there are post-Assign
                    // stmts in the then-list to actually gate.
                    // Without subsequent stmts, the gate would
                    // only declare a dead `var
                    // _f32_cas_did_transition_<sid>` and emit no
                    // wrap (`if (bool) { }` is valid WGSL but
                    // noisy + a no-op). The plague_city
                    // `ApplyLastRites` body
                    // `if (hunger > 0) { set_hunger(t, 0.0); }`
                    // is the canonical no-tail case.
                    let has_post_assign_stmts = <CgProgram as StmtListArena>::get(
                        ctx.prog, *then,
                    )
                    .map(|tl| {
                        tl.stmts
                            .iter()
                            .position(|sid| *sid == assign_sid)
                            .map(|pos| pos + 1 < tl.stmts.len())
                            .unwrap_or(false)
                    })
                    .unwrap_or(false);
                    if has_post_assign_stmts {
                        ctx.f32_first_writer_gate.set(Some(assign_sid.0));
                    }
                }
            }
            let then_body = lower_cg_stmt_list_to_wgsl(*then, ctx)?;
            ctx.f32_first_writer_gate.set(gate_save);
            match else_ {
                Some(else_id) => {
                    let else_body = lower_cg_stmt_list_to_wgsl(*else_id, ctx)?;
                    Ok(format!(
                        "if ({}) {{\n{}\n}} else {{\n{}\n}}",
                        c,
                        indent_block(&then_body, 1),
                        indent_block(&else_body, 1)
                    ))
                }
                None => Ok(format!(
                    "if ({}) {{\n{}\n}}",
                    c,
                    indent_block(&then_body, 1)
                )),
            }
        }
        CgStmt::Match { scrutinee, arms } => lower_match_to_wgsl(*scrutinee, arms, ctx),
        CgStmt::Let { local, value, ty } => {
            // `let local_<N>: <wgsl-ty> = <value>;`. The local is
            // visible to subsequent statements in the same body —
            // their value-expressions resolve to `local_<N>` once
            // `IrExpr::Local` resolution lands at the expression
            // layer (Task 5.5d).
            let v = lower_cg_expr_to_wgsl(*value, ctx)?;
            // View-fold target-row capture: when the let extracts an
            // event field of type AgentId (the `on Killed { by:
            // predator, prey: victim }` binder shape), append the
            // local id so any subsequent ViewStorage assign in the
            // same stmt list can index into a 1-D or 2-D address
            // based on the view's storage hint. See the Assign-to-
            // ViewStorage arm above for the consumer.
            //
            // Source order matters: pair_map composes
            // `local_<first> * cfg.second_key_pop + local_<second>`
            // — the first AgentId binder is the outer (k1) key and
            // the second is the inner (k2) key. The fold-handler
            // lowering walks the event-pattern bindings in
            // declaration order (`pattern.bindings.iter()` in
            // `synthesize_pattern_binding_lets`), so the WGSL Let
            // statements emit in the same order — guaranteeing
            // `(by, prey)` lands as `(local_first, local_second)`.
            // I.3b extension — also track `U32` binders so pair-keyed
            // views with a key-typed second param (Agent, u8|u32|i32)
            // compose the correct `(local_first * K + local_second)`
            // index expression. The Agent×Agent shape only emitted
            // AgentId binders, so a U32 binder appearing here can only
            // come from an I.3b-style event payload field
            // (`EnteredRoom { observer: o, room: r }`-style). Existing
            // Agent×Agent fixtures bind two AgentIds and never bind a
            // U32 event field that the fold body consumes, so the
            // extension is additive on the binding-set front. Other
            // U32 event-field reads in non-belief view bodies happen
            // outside of view-storage Assigns and so don't get
            // consumed by the index composition path.
            if matches!(ty, CgTy::AgentId | CgTy::U32) {
                if let Some(value_node) =
                    <CgProgram as ExprArena>::get(ctx.prog, *value)
                {
                    if matches!(value_node, CgExpr::Read(DataHandle::EventRing { .. }))
                        || is_event_field_read(value_node)
                    {
                        ctx.view_target_locals.borrow_mut().push(local.0);
                    }
                }
            }
            // Var-promotion: when `local` has been var-declared above
            // an enclosing CAS-loop body (the f32 RMW upgrade —
            // [`EmitCtx::var_promoted_locals`]), emit the binding as
            // an assignment (`local_N = v;`) instead of the default
            // `let local_N: T = v;`. The var declaration was emitted
            // by the surrounding `lower_cg_stmt_list_to_wgsl` pre-pass.
            if ctx.var_promoted_locals.borrow().contains(local) {
                return Ok(format!("local_{} = {};", local.0, v));
            }
            Ok(format!(
                "let local_{}: {} = {};",
                local.0,
                cg_ty_to_wgsl(*ty),
                v
            ))
        }
        CgStmt::ForEachNeighbor { .. } => {
            // Singleton path — defer to the multi-accumulator helper
            // with a one-element vec. This keeps a single emitter
            // covering both the standalone case (a fold whose
            // siblings aren't fusable) and the fused case (a run of
            // adjacent ForEachNeighbor stmts collapsed in
            // `lower_cg_stmt_list_to_wgsl`). The helper does not
            // dedup or reorder; it walks the supplied list and emits
            // an accumulator-update line per slot inside the inner
            // loop in the order given.
            emit_fused_for_each_neighbor(&[node], ctx)
        }
        CgStmt::ForEachNeighborBody {
            binder: _,
            body,
            radius_cells,
            origin,
        } => emit_for_each_neighbor_body(*body, *radius_cells, *origin, ctx),
        CgStmt::ForEachAgentBody { binder: _, body } => {
            emit_for_each_agent_body(*body, ctx)
        }
        CgStmt::ForEachAgent {
            acc_local,
            acc_ty,
            init,
            projection,
        } => {
            // var local_<N>: <ty> = <init>;
            // for (var per_pair_candidate: u32 = 0u; per_pair_candidate < cfg.agent_cap; ...) {
            //     local_<N> = local_<N> + <projection>;
            // }
            //
            // The loop variable name `per_pair_candidate` matches the
            // existing pair-bound emit convention so reads of
            // `binder.<field>` inside the projection lower to
            // `agent_<field>[per_pair_candidate]` via
            // `AgentRef::PerPairCandidate`. Subsequent reads of the
            // accumulator surface as `CgExpr::ReadLocal { local: acc_local }`
            // and emit as `local_<N>` — a `var` reads the same as a
            // `let` at the WGSL access site.
            let init_wgsl = lower_cg_expr_to_wgsl(*init, ctx)?;
            let proj_wgsl = lower_cg_expr_to_wgsl(*projection, ctx)?;
            let ty_wgsl = cg_ty_to_wgsl(*acc_ty);
            let n = acc_local.0;
            let body = format!(
                "var local_{n}: {ty_wgsl} = {init_wgsl};\n\
                 for (var per_pair_candidate: u32 = 0u; per_pair_candidate < cfg.agent_cap; per_pair_candidate = per_pair_candidate + 1u) {{\n\
                 \x20\x20\x20\x20local_{n} = (local_{n} + ({proj_wgsl}));\n\
                 }}"
            );
            Ok(body)
        }
        CgStmt::ApplyAbility { ability, caster, target, with_aoe_dispatch } => {
            // #136 slice β step 2: per-effect-slot dispatch loop.
            // Reads `ability_id` from the operand expression, walks
            // every effect slot in the PackedAbilityRegistry SoA,
            // and branches on `effect_kinds[i]` to the matching
            // apply path. Slot iteration count is the engine
            // constant `MAX_EFFECTS_PER_PROGRAM = 6` (pinned in the
            // schema hash); `EFFECT_KIND_EMPTY = 0xFFu` skips unused
            // slots.
            //
            // The apply paths themselves emit chronicle-ring records
            // via inline `atomicAdd(&event_tail[0], 1u)` slot
            // acquisition + `atomicStore` writes (the same shape
            // `lower_emit_to_wgsl` produces for declared events). The
            // event-kind tag for each variant is sourced from
            // `EFFECT_KIND_TO_EVENT_KIND_ID` above — that table is
            // pinned against the engine's `EventKindId` enum so a
            // discriminant rename surfaces at build time.
            //
            // Slice γ wires the chronicle write for the seven variants
            // the runtime currently has chronicle kinds for (Damage /
            // Heal / Shield / Stun / Slow / TransferGold /
            // ModifyStanding — EventKindIds 26–32). Other variants
            // keep their `// TODO slice γ` markers until the runtime
            // grows matching `EventKindId` slots (next would be Root /
            // Silence / Fear / Taunt at slot 39+, sharing Stun's
            // `expires_at_tick` payload shape).
            //
            // **Caster/target convention.** Slice δ + ε
            // (`92572af8` / `d0bc37fd`) plumbed explicit `caster` and
            // `target` operands onto `CgStmt::ApplyAbility`. Source
            // surface: `apply_ability <a> [by <c>] [target <t>]`.
            // Defaults: caster = `AgentSelfId` for PerAgent rules
            // (typed error for PerEvent without explicit `by`);
            // target = caster (slice-γ self-cast preserved when
            // source omits `target <expr>`). The dispatcher reads
            // both operands and writes them into actor (slot 2) and
            // target (slot 3) chronicle payload words respectively.
            //
            // This whole arm is dead at HEAD for any sim that
            // doesn't use `apply_ability` (the corpus uses it only
            // in `assets/sim/apply_ability_smoke.sim` today). The
            // wider runtime wire-up (#138 — replace inline emit in
            // duel_abilities with apply_ability) lights it up at
            // sim-level.
            //
            // **Path B (GPU AOE multi-target) — emit landed
            // 2026-05-07 (#121 follow-on).** The `with_aoe_dispatch`
            // flag on `CgStmt::ApplyAbility` (per-fixture build-time
            // opt-in via `LowerOpts::aoe_dispatch`) gates the AOE walk:
            //
            //   - `false` (every production runtime today — duel,
            //     tactical_squad_5v5, boss_fight, mass_battle, …):
            //     emit ONLY the single-target chain. No spatial
            //     reads, no `agent_pos[]` reads, no AreaKinds/AreaArgs
            //     SoA reads. BGL composer surfaces zero spatial
            //     bindings on the dispatcher — production runtimes
            //     keep their zero-spatial-overhead BGL.
            //
            //   - `true` (`apply_ability_smoke_runtime` today): wrap
            //     the `if (when_passes && chance_passes) {…}` body in
            //     a runtime branch on `area_kinds[effect_base + i]`:
            //       * 0u (Circle): walk the 27-cell spatial
            //         neighborhood around `aoe_center =
            //         agent_pos[target_slot]`; for each candidate
            //         within `dot(d, d) <= radius²`, run the
            //         chronicle arm chain + nested loop inside a
            //         block that shadows `target_slot = candidate`.
            //       * else (sentinel 0xFFu or unrecognised shape):
            //         fall through to the single-target chain
            //         (existing chain executes with the cast's
            //         explicit `target_slot`).
            //     P11 sort: GPU's atomicAdd ring claim does NOT
            //     preserve AgentId order. The CPU oracle
            //     (`apply_program_aoe`) sorts by AgentId ascending; the
            //     parity comparison sorts both sides post-readback
            //     (`parity_apply_program_sweep::canonicalize`).
            //
            // The when-predicate (`when_passes`) evaluates against
            // the EXPLICIT cast target — NOT per-AOE-target. Same
            // semantic as the CPU oracle's slot-keyed gate (`apply.rs:
            // apply_program_aoe`): when the slot's gate fires, every
            // in-circle target receives the chronicle record; when
            // the gate fails, none do.
            //
            // BGL ripple: when `with_aoe_dispatch=true`, the
            // dispatcher's body references `agent_pos`,
            // `spatial_grid_cells`, `spatial_grid_starts`, plus the
            // `ability_registry_area_kinds` / `area_args` SoA columns.
            // The driver's `wire_apply_ability_aoe_reads` (sibling to
            // `wire_ability_registry_column_reads`) records those
            // reads on flag-on dispatcher ops only — opt-out fixtures
            // never see them.
            let ability_wgsl = lower_cg_expr_to_wgsl(*ability, ctx)?;
            // Slice δ (#161): caster operand is now an explicit
            // CgExpr lowered through the same path as any other
            // expression. PerAgent rules lower this to
            // `CgExpr::AgentSelfId` → `agent_id` in WGSL; future
            // PerEvent rules would lower it to a different
            // identifier (event payload's actor field). The
            // dispatcher's chronicle writes use `caster_slot`
            // instead of the prior hardcoded `agent_id`.
            let caster_wgsl = lower_cg_expr_to_wgsl(*caster, ctx)?;
            // Slice ε part 1: target operand. Lowered separately so
            // the dispatcher can write it into chronicle payload
            // word 3 (target slot) — distinct from caster which goes
            // into payload word 2 (actor slot). When the source
            // omitted `target <expr>`, lowering populated this with
            // the caster expression, preserving slice-γ self-cast
            // semantics for callers that don't need explicit targets.
            let target_wgsl = lower_cg_expr_to_wgsl(*target, ctx)?;
            // Wave 1.5#4 GPU wire-up (2026-05-07): the `let *_event_id`
            // resolutions previously rendered into the inline primary
            // chain were dropped — `emit_chronicle_arm_chain` calls
            // `event_kind_id_for_effect_kind` itself (same pinned
            // table). The `expect` panic-on-missing semantic remains
            // sound there.
            // Wave 1.5#9 (2026-05-06): the nested-effect walk emits the
            // SAME if-else chain shape as the primary, just reading
            // from `ability_registry_nested_effect_*` columns at a
            // deeper indent. `emit_chronicle_arm_chain` builds the
            // shared chain once at 12-space indent (12 = inside the
            // outer for-loop's 8-space indent + 4 for the inner
            // for-loop body).
            //
            // Wave 1.5#4 GPU wire-up (this slice, 2026-05-07): the
            // primary walk now ALSO routes through
            // `emit_chronicle_arm_chain` (at 8-space indent), replacing
            // the prior inline copy of the if-chain. Single-source
            // arm-chain means scale_bonus folding into f32-amount arms
            // lives in one place. Nested ops carry no scaling slot
            // (mirrors `apply.rs`'s nested-op `scale_bonus = 0.0`
            // contract) so the nested walk passes a literal `0.0`
            // identifier; the primary walk passes `scale_bonus`
            // (computed from `scaling_stat_refs`/`scaling_percents` SoA
            // + per-stat agent SoA reads at `caster_slot` above the
            // chain).
            // Path B emit: for AOE dispatch the arm chains run inside
            // a per-target spatial walk that shadows `target_slot` to
            // the in-circle candidate, so the inner-most indent picks
            // up an extra 4 spaces (12 → 16 for primary, 16 → 20 for
            // nested) when the flag is on. The arm-chain helper is
            // pure-string and indent-parametric; flipping the indent
            // is the only emit-time difference between the two paths.
            let (primary_indent, nested_indent) = if *with_aoe_dispatch {
                ("            ",     "                ")
            } else {
                ("        ",         "            ")
            };
            // P11 seq trailer: resolve kernel id and reserve 46 emit-idx slots per
            // arm chain (primary) + 46 (nested) = 92 total.
            let _chain_kernel_id = ctx.current_kernel_index.get()
                .and_then(|ki| ctx.producer_kernel_ids.get(&ki).copied())
                .unwrap_or(0);
            let _primary_first_emit_idx = ctx.intra_emit_idx.get();
            ctx.intra_emit_idx.set(_primary_first_emit_idx + 46);
            let _nested_first_emit_idx = ctx.intra_emit_idx.get();
            ctx.intra_emit_idx.set(_nested_first_emit_idx + 46);
            let primary_arm_chain =
                emit_chronicle_arm_chain(primary_indent, "scale_bonus", ctx.debug_wgsl, _chain_kernel_id, _primary_first_emit_idx);
            let nested_arm_chain =
                emit_chronicle_arm_chain(nested_indent, "nested_scale_bonus", ctx.debug_wgsl, _chain_kernel_id, _nested_first_emit_idx);
            // Engine pins MAX_EFFECTS_PER_PROGRAM = 6 + EFFECT_KIND_EMPTY = 0xFFu
            // (see crates/engine/src/ability/program.rs:28 +
            // crates/engine/src/ability/packed.rs). Inlining the
            // constants keeps the kernel self-contained without
            // pulling in a shared `consts.wgsl` preamble.
            //
            // Wave 1.5#9 nested-effect dispatch (2026-05-06). After the
            // primary's chronicle write, the dispatcher walks
            // `ability_registry_nested_effect_kinds` (stride =
            // MAX_EFFECTS_PER_PROGRAM × MAX_NESTED_PER_EFFECT, both =
            // 6 × 2 = 12 entries per ability) and writes a chronicle
            // record per chronicle-bearing nested op. Closes the
            // documented gap surfaced by the Reap verb swap (commit
            // `72a35307`): Reap's `{ stun 1s }` produces an
            // EffectStunApplied chronicle record alongside
            // EffectExecuteApplied. The arm-chain is structurally
            // identical to the primary's — same kind/payload encoding,
            // same EventKindId mapping (`pack_effect` in
            // `crates/engine/src/ability/packed.rs` is the single
            // source of truth) — so the inner walk wraps in its own
            // `{}` block scope to re-declare the fresh `kind` /
            // `payload_a` / `payload_b` locals from the nested SoA
            // columns.
            // Atomic-aware case-line builders — when the active kernel
            // upgraded one or more f32 SoA columns to atomic, the
            // dispatcher's stat-dispatch reads must wrap in
            // `bitcast<f32>(atomicLoad(…))`. `dispatcher_f32_field_read`
            // does that conditionally (P5).
            let stat_case_0 = dispatcher_f32_field_read(AgentFieldId::AttackDamage, "caster_slot", ctx);
            let stat_case_1 = dispatcher_f32_field_read(AgentFieldId::AbilityPower, "caster_slot", ctx);
            let stat_case_2 = dispatcher_f32_field_read(AgentFieldId::MaxHp, "caster_slot", ctx);
            let stat_case_3 = dispatcher_f32_field_read(AgentFieldId::Hp, "caster_slot", ctx);
            let stat_case_4 = dispatcher_f32_field_read(AgentFieldId::Armor, "caster_slot", ctx);
            let stat_case_5 = dispatcher_f32_field_read(AgentFieldId::MagicResist, "caster_slot", ctx);
            let stat_case_6 = dispatcher_f32_field_read(AgentFieldId::MoveSpeed, "caster_slot", ctx);
            let stat_case_7 = dispatcher_f32_field_read(AgentFieldId::Mana, "caster_slot", ctx);
            let pred_case_0 = dispatcher_f32_field_read(AgentFieldId::AttackDamage, "pred_agent", ctx);
            let pred_case_1 = dispatcher_f32_field_read(AgentFieldId::AbilityPower, "pred_agent", ctx);
            let pred_case_2 = dispatcher_f32_field_read(AgentFieldId::MaxHp, "pred_agent", ctx);
            let pred_case_3 = dispatcher_f32_field_read(AgentFieldId::Hp, "pred_agent", ctx);
            let pred_case_4 = dispatcher_f32_field_read(AgentFieldId::Armor, "pred_agent", ctx);
            let pred_case_5 = dispatcher_f32_field_read(AgentFieldId::MagicResist, "pred_agent", ctx);
            let pred_case_6 = dispatcher_f32_field_read(AgentFieldId::MoveSpeed, "pred_agent", ctx);
            let pred_case_7 = dispatcher_f32_field_read(AgentFieldId::Mana, "pred_agent", ctx);

            let body = format!(
                "// #136 apply_ability dispatcher (slice β step 2)\n\
                 // Wave 1.5#4 GPU wire-up: per-effect slot reads\n\
                 // `scaling_stat_refs` + `scaling_percents` SoA + per-stat\n\
                 // agent SoA at `caster_slot` to compute the additive\n\
                 // `scale_bonus = Σ percent * caster_stat`. Mirrors the\n\
                 // CPU oracle in `engine::ability::apply::apply_program`\n\
                 // (sums `inner.iter().map(|s| s.percent * stats.get(s.stat_ref))`\n\
                 // — same iteration order j=0 then j=1 per P11 reduction\n\
                 // ordering).\n\
                 {{\n\
                 \x20\x20\x20\x20let ability_id__u32: u32 = u32({ability_wgsl});\n\
                 \x20\x20\x20\x20let caster_slot: u32 = u32({caster_wgsl});\n\
                 \x20\x20\x20\x20let target_slot: u32 = u32({target_wgsl});\n\
                 \x20\x20\x20\x20// AbilityId is 1-based (NonZeroU32); slot index is id - 1.\n\
                 \x20\x20\x20\x20let ability_slot: u32 = ability_id__u32 - 1u;\n\
                 \x20\x20\x20\x20let effect_base: u32 = ability_slot * 6u; // MAX_EFFECTS_PER_PROGRAM\n\
                 \x20\x20\x20\x20// Wave 1.5#4 GPU scaling: per-(effect, scaling-slot) stride\n\
                 \x20\x20\x20\x20// = MAX_EFFECTS_PER_PROGRAM × MAX_SCALINGS_PER_EFFECT = 6 × 2 = 12.\n\
                 \x20\x20\x20\x20let scaling_base: u32 = ability_slot * 12u;\n\
                 \x20\x20\x20\x20// Wave 1.5#9 nested base: ability_slot × MAX_EFFECTS_PER_PROGRAM\n\
                 \x20\x20\x20\x20// × MAX_NESTED_PER_EFFECT = 6 × 2 entries per ability.\n\
                 \x20\x20\x20\x20let nested_base: u32 = ability_slot * 12u;\n\
                 \x20\x20\x20\x20for (var i: u32 = 0u; i < 6u; i = i + 1u) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20let kind: u32 = ability_registry_effect_kinds[effect_base + i];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20if (kind == 0xFFu) {{ continue; }} // EFFECT_KIND_EMPTY\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// P11 chance gate (Wave 1.5#5 GPU wire-up). Mirrors CPU\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// `apply_program`'s `(per_agent_u32_pcg_with_extra(seed,\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// caster_slot, tick, RngPurpose::Chance=10, slot_idx) &\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// 0xFFFF) < q16` test. Sentinel `chances[i] == 0xFFFFu`\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// (CHANCE_NONE_SENTINEL) → no gate authored, fire\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// unconditionally.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20var chance_passes: bool = true;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20let chance_q16: u32 = ability_registry_chances[effect_base + i];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20if (chance_q16 != 0xFFFFu) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let chance_draw: u32 = per_agent_u32_with_extra(seed, caster_slot, tick, 10u, i) & 0xFFFFu;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20chance_passes = chance_draw < chance_q16;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20let payload_a: u32 = ability_registry_effect_payload_a[effect_base + i];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20let payload_b: u32 = ability_registry_effect_payload_b[effect_base + i];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Wave 1.5#4: compute scale_bonus from the slot's two\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// `scaling_stat_refs`/`scaling_percents` entries (sentinel\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// 0xFFu = unused slot → 0.0 contribution).\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20var scale_bonus: f32 = 0.0;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20for (var k: u32 = 0u; k < 2u; k = k + 1u) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let s_off: u32 = scaling_base + i * 2u + k;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let s_tag: u32 = ability_registry_scaling_stat_refs[s_off];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20if (s_tag == 0xFFu) {{ continue; }} // SCALING_STAT_NONE_SENTINEL\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let s_pct: f32 = ability_registry_scaling_percents[s_off];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20// agent_stat: dispatch s_tag → SoA read at caster_slot.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20// Mirrors `CasterStats::get` in engine/src/ability/program.rs.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20var stat_v: f32 = 0.0;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20switch (s_tag) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 0u: {{ stat_v = {stat_case_0}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 1u: {{ stat_v = {stat_case_1}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 2u: {{ stat_v = {stat_case_2}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 3u: {{ stat_v = {stat_case_3}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 4u: {{ stat_v = {stat_case_4}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 5u: {{ stat_v = {stat_case_5}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 6u: {{ stat_v = {stat_case_6}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 7u: {{ stat_v = {stat_case_7}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20default: {{ stat_v = 0.0; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20scale_bonus = scale_bonus + s_pct * stat_v;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Wave 1.5#7 GPU eval: per-effect when-predicate.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Task #227 — compound predicates serialize to up to\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// MAX_PRED_NODES_PER_EFFECT (12) RPN nodes per effect slot;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// per-ability stride is MAX_EFFECTS_PER_PROGRAM*12 = 6*12 = 72.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Mirrors `apply::evaluate_when_tree` (CPU oracle) — same\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// stat dispatch table as the scale_bonus switch above.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Sentinel binder == 0xFF → end of nodes; operator\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// markers: 0xFE=AND, 0xFD=OR, 0xFC=NOT.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20var when_passes: bool = true;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20{{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let pred_node_base: u32 = ability_slot * 72u + i * 12u;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20var pred_stack: array<bool, 12>;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20var pred_sp: u32 = 0u;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20for (var pi: u32 = 0u; pi < 12u; pi = pi + 1u) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let pn_binder: u32 = ability_registry_when_pred_binder[pred_node_base + pi];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20if (pn_binder == 0xFFu) {{ break; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20if (pn_binder == 0xFEu) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let r_v = pred_stack[pred_sp - 1u];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let l_v = pred_stack[pred_sp - 2u];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20pred_sp = pred_sp - 1u;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20pred_stack[pred_sp - 1u] = l_v && r_v;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}} else if (pn_binder == 0xFDu) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let r_v = pred_stack[pred_sp - 1u];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let l_v = pred_stack[pred_sp - 2u];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20pred_sp = pred_sp - 1u;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20pred_stack[pred_sp - 1u] = l_v || r_v;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}} else if (pn_binder == 0xFCu) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let v_v = pred_stack[pred_sp - 1u];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20pred_stack[pred_sp - 1u] = !v_v;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}} else {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20// Atom: <binder>.<field> <op> <literal>\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let pred_field: u32   = ability_registry_when_pred_field[pred_node_base + pi];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let pred_op: u32      = ability_registry_when_pred_op[pred_node_base + pi];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let pred_literal: f32 = ability_registry_when_pred_literal[pred_node_base + pi];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20var pred_agent: u32 = caster_slot;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20if (pn_binder == 1u) {{ pred_agent = target_slot; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20var pred_lhs: f32 = 0.0;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20switch (pred_field) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 0u: {{ pred_lhs = {pred_case_0}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 1u: {{ pred_lhs = {pred_case_1}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 2u: {{ pred_lhs = {pred_case_2}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 3u: {{ pred_lhs = {pred_case_3}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 4u: {{ pred_lhs = {pred_case_4}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 5u: {{ pred_lhs = {pred_case_5}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 6u: {{ pred_lhs = {pred_case_6}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 7u: {{ pred_lhs = {pred_case_7}; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20default: {{ pred_lhs = 0.0; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20var atom_v: bool = false;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20switch (pred_op) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 0u: {{ atom_v = pred_lhs <  pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 1u: {{ atom_v = pred_lhs <= pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 2u: {{ atom_v = pred_lhs >  pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 3u: {{ atom_v = pred_lhs >= pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 4u: {{ atom_v = pred_lhs == pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 5u: {{ atom_v = pred_lhs != pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20default: {{ atom_v = false; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20pred_stack[pred_sp] = atom_v;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20pred_sp = pred_sp + 1u;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20if (pred_sp > 0u) {{ when_passes = pred_stack[0u]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20if (when_passes && chance_passes) {{\n\
                 {per_target_body}\
                 \x20\x20\x20\x20\x20\x20\x20\x20}} // end if (when_passes && chance_passes)\n\
                 \x20\x20\x20\x20}}\n\
                 }}",
                per_target_body = build_apply_ability_per_target_body(
                    *with_aoe_dispatch,
                    &primary_arm_chain,
                    &nested_arm_chain,
                ),
            );
            Ok(body)
        }
        CgStmt::ViewStorageAppend { .. } => {
            // Plan G G3b/G3c — struct-payload ring append. The actual
            // WGSL is hand-synthesised by `build_view_fold_ring_append_body`
            // in `cg/emit/kernel.rs` from the registered `ViewLayout` +
            // the storage hint's K. The generic stmt → wgsl walker is
            // bypassed entirely for PerEntityRing fold bodies (the
            // kernel composer special-cases the body), so this arm is
            // structurally unreachable in production. Emit an empty
            // placeholder so a synthetic IR walking through the generic
            // path still produces parseable WGSL.
            Ok(String::from(
                "    // ViewStorageAppend — handled by ring-append emit at kernel.rs\n",
            ))
        }
    }
}

/// Render the per-target body of the apply_ability dispatcher's
/// `if (when_passes && chance_passes) { … }` block. Used by the
/// `CgStmt::ApplyAbility` arm above; factored out so the AOE Path B
/// branching shape stays readable (the underlying single-target +
/// nested-loop emit is identical between flag values, just wrapped in
/// a 27-cell spatial walk when AOE is on).
///
/// The two arm chains (`primary_arm_chain` for the per-effect-slot
/// chronicle write, `nested_arm_chain` for the inner Wave 1.5#9
/// nested-effect walk) are pre-rendered at the appropriate indent
/// (single-target indents = 8/12 spaces; AOE indents = 12/16 spaces,
/// shifted by the extra 4 spaces of the outer per-target loop) so this
/// function purely composes them into the right outer scaffolding.
///
/// **Single-target shape (`with_aoe_dispatch == false`)**: emits the
/// existing primary-arm-chain + nested-loop sequence verbatim. No
/// spatial reads, no `agent_pos`/`area_kinds`/`area_args` references.
/// BGL composer surfaces zero spatial bindings on the dispatcher —
/// production runtimes preserve their zero-spatial-overhead BGL.
///
/// **AOE shape (`with_aoe_dispatch == true`)**: branches on
/// `area_kinds[effect_base + i]`:
///   - `0u` (Circle): walk the 27-cell spatial neighborhood around
///     `aoe_center = agent_pos[target_slot]`; for each candidate within
///     `dot(d, d) <= radius²`, run the chronicle arm chain + nested
///     loop in a `{ let target_slot = candidate; … }` block that
///     shadows the outer `target_slot` for chronicle writes. P11 sort
///     handles atomicAdd-induced order non-determinism on readback.
///   - `1u` (Cone, #178): walk the 27-cell neighborhood around the
///     APEX (`agent_pos[caster_slot]`); for each candidate within
///     `dist² ≤ range²` AND `dot(normalize(cand - apex), direction)
///     ≥ cos(half_angle_rad)`, shadow `target_slot = candidate` and
///     run the chronicle arm chain. `direction = normalize(target -
///     apex)` (degenerate when caster targets self — guarded by the
///     `dir_len_sq >= 1e-6` branch which emits no records). Apex-
///     coincident candidates (caster + co-located agents) are
///     excluded by the `_dist_sq < 1e-6 -> continue` guard so the
///     CPU oracle (`apply_program_aoe_cone_filter`) and GPU emit
///     match bit-for-bit.
///   - `5u` (Box, #179): walk the 27-cell neighborhood around
///     `aoe_center = agent_pos[target_slot]` (same convention as
///     Circle); for each candidate within
///     `abs(d.x) <= wx && abs(d.y) <= wy && abs(d.z) <= wz` (closed
///     AABB — candidates exactly at a wall are in-box), shadow
///     `target_slot = candidate` and run the chronicle arm chain.
///     CPU oracle is `apply_program_aoe_box_filter`. **Spatial walk
///     limitation:** any half-extent exceeding `SPATIAL_CELL_SIZE`
///     would miss candidates beyond the 27-cell ring; test fixtures
///     keep extents ≤ cell size.
///   - else (sentinel `0xFFu` = no area, or non-AOE-Path-B shape):
///     fall through to the single-target chain with the explicit
///     cast `target_slot`. Other shapes (Line/Sphere/Capsule/etc.)
///     are deferred — their additional geometry kernels would extend
///     this branch.
///
/// Why the spatial-walk lives at this layer (rather than in the
/// per-cast caller): the chronicle arm chain consumes `target_slot`
/// at every chronicle-bearing variant (`atomicStore(...&event_ring[_slot
/// * 11u + 3u], (target_slot))`), so the per-target shadow has to
/// surround the entire arm chain (primary + nested). Hoisting the
/// spatial walk out would require lifting target_slot into a parameter
/// of `emit_chronicle_arm_chain` — strictly more code, and the chain
/// helper stays usefully indent-parametric without the additional
/// hook.
fn build_apply_ability_per_target_body(
    with_aoe_dispatch: bool,
    primary_arm_chain: &str,
    nested_arm_chain:  &str,
) -> String {
    if !with_aoe_dispatch {
        // Single-target path: emit existing primary chain + nested loop
        // at indent depth 8/12 spaces. No spatial bindings touched.
        let mut s = String::new();
        s.push_str(primary_arm_chain);
        s.push_str(
            "        // Variant 7 (CastAbility) — recursive dispatch. The\n\
             \x20       // nested ability_id lives in payload_a; recursing\n\
             \x20       // requires either a depth-bounded re-entry into this\n\
             \x20       // loop or a separate work queue. Deferred to slice δ.\n\
             \x20       // Wave 1.5#9 nested-effect walk. After the primary's\n\
             \x20       // chronicle write resolves, walk up to\n\
             \x20       // MAX_NESTED_PER_EFFECT (=2) nested ops on this slot\n\
             \x20       // and write a chronicle record per chronicle-bearing\n\
             \x20       // kind. Block-scoped so the inner `kind` / `payload_a`\n\
             \x20       // / `payload_b` locals don't shadow the primary walk's.\n\
             \x20       // Nested ops carry no scaling slot in the registry today\n\
             \x20       // (mirrors `apply.rs`'s `push_effect_event(..., 0.0)` for\n\
             \x20       // nested), so `nested_scale_bonus` is forced to 0.0.\n\
             \x20       // Wave 1.5#7: nested loop INSIDE the `if (when_passes)` block.\n\
             \x20       let nested_slot_base: u32 = nested_base + i * 2u;\n\
             \x20       for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
             \x20           let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
             \x20           if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
             \x20           let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
             \x20           let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
             \x20           let nested_scale_bonus: f32 = 0.0;\n",
        );
        s.push_str(nested_arm_chain);
        s.push_str("            }\n");
        return s;
    }

    // AOE Path B: branch on `area_kinds[effect_base + i]`. The arm
    // chains were pre-rendered with extra indent (12/16 spaces) so
    // the inner `{ let target_slot = candidate; … }` block reads
    // cleanly nested inside the cell-walk loop. Path B today only
    // expands `Circle` (kind == 0u); other kinds fall through to the
    // single-target chain (the `else` arm of the area_kind branch).
    //
    // Spatial walk shape mirrors the cell-walk path of
    // `emit_for_each_neighbor_body` (cell_index helper bound by the
    // spatial prelude). Center = `agent_pos[target_slot]` (the
    // explicit cast target's world position — same convention as
    // `apply_program_aoe`'s `state.spatial().within_radius(state,
    // primary_target_pos, radius)` site).
    //
    // P11 ordering: GPU's atomicAdd ring claim does NOT preserve
    // AgentId order. The CPU oracle (`apply_program_aoe`) emits
    // chronicle records in `aoe_targets` order (sorted ascending by
    // AgentId). Cross-backend parity therefore requires the test
    // harness to sort both sides post-readback (already done by
    // `parity_apply_program_sweep::canonicalize`).
    let mut s = String::new();
    s.push_str(
        "        // #121 AOE Path B: per-effect-slot AOE expansion. Flag is on\n\
         \x20       // → branch on `area_kinds[effect_base + i]`. Circle (0u)\n\
         \x20       // walks the 27-cell spatial neighborhood around\n\
         \x20       // `agent_pos[target_slot]` and runs the chronicle arm chain\n\
         \x20       // once per in-radius candidate (shadowing target_slot).\n\
         \x20       // Cone (1u) walks the 27 cells around `agent_pos[caster_slot]`\n\
         \x20       // (the apex), gates each candidate on range² ∧ angular dot,\n\
         \x20       // and shadows target_slot the same way (#178).\n\
         \x20       // Box (5u) walks the 27 cells around `agent_pos[target_slot]`,\n\
         \x20       // gates each candidate on per-axis `|d.<axis>| <= w<axis>`\n\
         \x20       // (closed AABB), and shadows target_slot the same way (#179).\n\
         \x20       // Sphere (6u) is mathematically equivalent to Circle today\n\
         \x20       // (3D dist² ≤ radius² over the 27-cell ring); separate branch\n\
         \x20       // for code clarity (#180).\n\
         \x20       // Ring (3u) walks the 27 cells around `agent_pos[target_slot]`,\n\
         \x20       // gates each candidate on `inner² ≤ dist² ≤ outer²` (closed\n\
         \x20       // annulus, #180).\n\
         \x20       // Line (2u) walks the 27 cells around `agent_pos[caster_slot]`\n\
         \x20       // (the apex), gates each candidate on `0 ≤ along ≤ length` ∧\n\
         \x20       // `perp_sq ≤ (width/2)²` (forward-facing rectangle, #180).\n\
         \x20       // Sentinel 0xFFu / unrecognised shape → single-target\n\
         \x20       // fallback at the same indent the non-AOE path uses.\n\
         \x20       let area_kind: u32 = ability_registry_area_kinds[effect_base + i];\n\
         \x20       if (area_kind == 0u) {\n\
         \x20           // Circle. area_args layout: [radius, _, _, _] (4 f32 per slot).\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let radius: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let radius_sq: f32 = radius * radius;\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           // 27-cell neighborhood walk (mirrors the cell-walk path\n\
         \x20           // of `emit_for_each_neighbor_body` — same `cell_index`\n\
         \x20           // helper from the spatial prelude, same\n\
         \x20           // `spatial_grid_starts[cell..+1]` slot iteration).\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           if (dot(_dvec, _dvec) <= radius_sq) {\n\
         \x20                               // Shadow target_slot for the arm chain's chronicle writes.\n\
         \x20                               let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                            // Wave 1.5#9 nested loop runs per-target inside the\n\
         \x20                           // AOE walk — each in-circle target receives the\n\
         \x20                           // primary chronicle record AND each chronicle-\n\
         \x20                           // bearing nested op (mirrors `apply_program_aoe`'s\n\
         \x20                           // `for nested_op in nested { push_effect_event(...) }`\n\
         \x20                           // inside the per-target loop).\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                           } // end if (in radius)\n\
         \x20                       } // end for _i\n\
         \x20                   } // end for dx\n\
         \x20               } // end for dy\n\
         \x20           } // end for dz\n\
         \x20       } else if (area_kind == 1u) {\n\
         \x20           // Cone (#178). area_args layout: [half_angle_deg, range, _, _].\n\
         \x20           // Apex = caster position; direction = normalize(target_pos -\n\
         \x20           // apex). Per-candidate gate: dist² ≤ range² ∧ dot(\n\
         \x20           // normalize(cand-apex), direction) ≥ cos(half_angle). Apex-\n\
         \x20           // coincident candidates (incl. caster) are excluded — the\n\
         \x20           // cone never hits its own origin (degenerate direction).\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let half_angle_deg: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let range: f32 = ability_registry_area_args[area_args_base + 1u];\n\
         \x20           let half_angle_rad: f32 = half_angle_deg * 0.01745329252; // π/180\n\
         \x20           let cos_half_angle: f32 = cos(half_angle_rad);\n\
         \x20           let range_sq: f32 = range * range;\n\
         \x20           let apex: vec3<f32> = agent_pos[caster_slot];\n\
         \x20           let target_pos_local: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let direction_raw: vec3<f32> = target_pos_local - apex;\n\
         \x20           let dir_len_sq: f32 = dot(direction_raw, direction_raw);\n\
         \x20           if (dir_len_sq >= 1e-6) {\n\
         \x20               // Non-degenerate cone. Direction = normalize(target -\n\
         \x20               // apex); inverseSqrt matches CPU's `recip().sqrt()`.\n\
         \x20               let direction: vec3<f32> = direction_raw * inverseSqrt(dir_len_sq);\n\
         \x20               let _self_cell_f = (apex + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20               let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20               let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20               let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20               let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20               for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20                   for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                       for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                           let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                           let _start = spatial_grid_starts[_cell];\n\
         \x20                           let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                           for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                               let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                               let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                               let _to_cand: vec3<f32> = _cand_pos - apex;\n\
         \x20                               let _dist_sq: f32 = dot(_to_cand, _to_cand);\n\
         \x20                               // Apex exclusion (caster + co-located candidates).\n\
         \x20                               if (_dist_sq < 1e-6) { continue; }\n\
         \x20                               if (_dist_sq > range_sq) { continue; }\n\
         \x20                               let _cand_dir: vec3<f32> = _to_cand * inverseSqrt(_dist_sq);\n\
         \x20                               if (dot(_cand_dir, direction) < cos_half_angle) { continue; }\n\
         \x20                               // Shadow target_slot for the arm chain's chronicle writes.\n\
         \x20                               let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-cone target —\n\
         \x20                               // same shape as the Circle branch.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                           } // end for _i (cone walk)\n\
         \x20                       } // end for dx (cone walk)\n\
         \x20                   } // end for dy (cone walk)\n\
         \x20               } // end for dz (cone walk)\n\
         \x20           } // end if (dir_len_sq >= 1e-6) — degenerate cone is no-op\n\
         \x20       } else if (area_kind == 5u) {\n\
         \x20           // Box (#179). area_args layout: [wx, wy, wz, _] (half-\n\
         \x20           // extents along each world axis). Center =\n\
         \x20           // `agent_pos[target_slot]` — same convention as Circle.\n\
         \x20           // Per-candidate gate: per-axis `|d.<axis>| <= w<axis>`\n\
         \x20           // (closed AABB; candidates exactly at a wall are in-box).\n\
         \x20           //\n\
         \x20           // **Spatial walk limitation.** The 27-cell walk only\n\
         \x20           // visits the immediate cell ring around the center.\n\
         \x20           // Any half-extent exceeding `SPATIAL_CELL_SIZE` would\n\
         \x20           // miss candidates beyond the 27-cell footprint.\n\
         \x20           // Test fixtures must keep extents ≤ cell size.\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let wx: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let wy: f32 = ability_registry_area_args[area_args_base + 1u];\n\
         \x20           let wz: f32 = ability_registry_area_args[area_args_base + 2u];\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           if (abs(_dvec.x) <= wx && abs(_dvec.y) <= wy && abs(_dvec.z) <= wz) {\n\
         \x20                               // Shadow target_slot for the arm chain's chronicle writes.\n\
         \x20                               let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-box target —\n\
         \x20                               // same shape as the Circle/Cone branches.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                           } // end if (in box)\n\
         \x20                       } // end for _i (box walk)\n\
         \x20                   } // end for dx (box walk)\n\
         \x20               } // end for dy (box walk)\n\
         \x20           } // end for dz (box walk)\n\
         \x20       } else if (area_kind == 6u) {\n\
         \x20           // Sphere (#180). area_args layout: [radius, _, _, _].\n\
         \x20           // Mathematically equivalent to Circle today (3D dist² ≤\n\
         \x20           // radius² over the 27-cell ring around\n\
         \x20           // `agent_pos[target_slot]`); separate branch for code\n\
         \x20           // clarity. A future divergence (e.g. flat-disk Circle vs\n\
         \x20           // true 3D Sphere) would update both branches.\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let radius: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let radius_sq: f32 = radius * radius;\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           if (dot(_dvec, _dvec) <= radius_sq) {\n\
         \x20                               // Shadow target_slot for the arm chain's chronicle writes.\n\
         \x20                               let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-sphere target —\n\
         \x20                               // same shape as the Circle/Cone/Box branches.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                           } // end if (in sphere)\n\
         \x20                       } // end for _i (sphere walk)\n\
         \x20                   } // end for dx (sphere walk)\n\
         \x20               } // end for dy (sphere walk)\n\
         \x20           } // end for dz (sphere walk)\n\
         \x20       } else if (area_kind == 3u) {\n\
         \x20           // Ring (#180). area_args layout: [inner_radius,\n\
         \x20           // outer_radius, _, _]. Annulus gate: `inner² ≤ dist² ≤\n\
         \x20           // outer²` (closed on both edges — candidates exactly at\n\
         \x20           // either wall are in-ring).\n\
         \x20           //\n\
         \x20           // **Edge case: inner > outer.** Predicate is\n\
         \x20           // unsatisfiable (lhs > rhs) ⇒ empty in-ring set. Both\n\
         \x20           // backends agree.\n\
         \x20           //\n\
         \x20           // **Spatial walk limitation.** 27-cell walk; if outer\n\
         \x20           // radius exceeds `SPATIAL_CELL_SIZE`, candidates beyond\n\
         \x20           // the 27-cell footprint are missed. Fixtures must keep\n\
         \x20           // outer ≤ cell size to stay byte-equal across backends.\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let inner: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let outer: f32 = ability_registry_area_args[area_args_base + 1u];\n\
         \x20           let inner_sq: f32 = inner * inner;\n\
         \x20           let outer_sq: f32 = outer * outer;\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           let _dist_sq: f32 = dot(_dvec, _dvec);\n\
         \x20                           if (_dist_sq >= inner_sq && _dist_sq <= outer_sq) {\n\
         \x20                               // Shadow target_slot for the arm chain's chronicle writes.\n\
         \x20                               let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-ring target —\n\
         \x20                               // same shape as the Circle/Cone/Box branches.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                           } // end if (in ring)\n\
         \x20                       } // end for _i (ring walk)\n\
         \x20                   } // end for dx (ring walk)\n\
         \x20               } // end for dy (ring walk)\n\
         \x20           } // end for dz (ring walk)\n\
         \x20       } else if (area_kind == 2u) {\n\
         \x20           // Line (#180). area_args layout: [length, width, _, _].\n\
         \x20           // Forward-facing rectangle: apex = caster position;\n\
         \x20           // direction = normalize(target_pos - apex). Per-candidate\n\
         \x20           // gate (Pythagoras avoids 3D cross-product, matching the\n\
         \x20           // CPU oracle):\n\
         \x20           //   1. along = dot(to_cand, direction)\n\
         \x20           //   2. 0 ≤ along ≤ length (in front, within length)\n\
         \x20           //   3. perp_sq = dot(to_cand, to_cand) - along*along\n\
         \x20           //   4. perp_sq ≤ (width/2)²\n\
         \x20           // Degenerate (caster == target) ⇒ direction undefined ⇒\n\
         \x20           // no targets. The `dir_len_sq < 1e-6` branch matches the\n\
         \x20           // CPU oracle's `apply_program_aoe_line_filter`.\n\
         \x20           //\n\
         \x20           // **Spatial walk limitation.** 27-cell walk around the\n\
         \x20           // apex; if length exceeds `SPATIAL_CELL_SIZE` candidates\n\
         \x20           // past the 27-cell footprint are missed. Fixtures must\n\
         \x20           // keep length ≤ cell size to stay byte-equal.\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let length: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let width: f32 = ability_registry_area_args[area_args_base + 1u];\n\
         \x20           let half_width: f32 = width * 0.5;\n\
         \x20           let half_width_sq: f32 = half_width * half_width;\n\
         \x20           let apex: vec3<f32> = agent_pos[caster_slot];\n\
         \x20           let target_pos_local: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let direction_raw: vec3<f32> = target_pos_local - apex;\n\
         \x20           let dir_len_sq: f32 = dot(direction_raw, direction_raw);\n\
         \x20           if (dir_len_sq >= 1e-6) {\n\
         \x20               // Non-degenerate line. Direction = normalize(target -\n\
         \x20               // apex); inverseSqrt matches CPU's `recip().sqrt()`.\n\
         \x20               let direction: vec3<f32> = direction_raw * inverseSqrt(dir_len_sq);\n\
         \x20               let _self_cell_f = (apex + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20               let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20               let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20               let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20               let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20               for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20                   for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                       for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                           let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                           let _start = spatial_grid_starts[_cell];\n\
         \x20                           let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                           for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                               let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                               let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                               let _to_cand: vec3<f32> = _cand_pos - apex;\n\
         \x20                               let _along: f32 = dot(_to_cand, direction);\n\
         \x20                               if (_along < 0.0) { continue; }\n\
         \x20                               if (_along > length) { continue; }\n\
         \x20                               let _dist_sq: f32 = dot(_to_cand, _to_cand);\n\
         \x20                               let _perp_sq: f32 = _dist_sq - _along * _along;\n\
         \x20                               if (_perp_sq > half_width_sq) { continue; }\n\
         \x20                               // Shadow target_slot for the arm chain's chronicle writes.\n\
         \x20                               let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-line target —\n\
         \x20                               // same shape as the Cone branch.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                           } // end for _i (line walk)\n\
         \x20                       } // end for dx (line walk)\n\
         \x20                   } // end for dy (line walk)\n\
         \x20               } // end for dz (line walk)\n\
         \x20           } // end if (dir_len_sq >= 1e-6) — degenerate line is no-op\n\
         \x20       } else if (area_kind == 7u) {\n\
         \x20           // Column (#181). area_args layout: [radius, height, _, _].\n\
         \x20           // Vertical cylinder centered on target_slot, extending UP\n\
         \x20           // ONLY (`0 ≤ dy ≤ height`) — distinct from Cylinder which is\n\
         \x20           // symmetric. Per-candidate gate:\n\
         \x20           //   1. dist_xz_sq = dx*dx + dz*dz ≤ radius²\n\
         \x20           //   2. 0 ≤ dy ≤ height\n\
         \x20           //\n\
         \x20           // **Spatial walk limitation.** 27-cell walk; if either radius\n\
         \x20           // or height exceeds `SPATIAL_CELL_SIZE` candidates beyond the\n\
         \x20           // 27-cell footprint are missed.\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let radius: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let height: f32 = ability_registry_area_args[area_args_base + 1u];\n\
         \x20           let radius_sq: f32 = radius * radius;\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           let _dist_xz_sq: f32 = _dvec.x * _dvec.x + _dvec.z * _dvec.z;\n\
         \x20                           if (_dist_xz_sq > radius_sq) { continue; }\n\
         \x20                           if (_dvec.y < 0.0) { continue; }\n\
         \x20                           if (_dvec.y > height) { continue; }\n\
         \x20                           let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-column target.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                       } // end for _i (column walk)\n\
         \x20                   } // end for dx (column walk)\n\
         \x20               } // end for dy (column walk)\n\
         \x20           } // end for dz (column walk)\n\
         \x20       } else if (area_kind == 8u) {\n\
         \x20           // Wall (#181). area_args layout: [length, height, thickness,\n\
         \x20           // facing_deg]. Wall is the only 4-arg shape (the others use\n\
         \x20           // 1-3 args + zero-padding). Facing-bearing rectangular slab.\n\
         \x20           //\n\
         \x20           // Convention: facing direction = (cos θ, 0, sin θ) where θ\n\
         \x20           // = facing_deg · π/180 (0deg = +X, 90deg = +Z, CCW). Lateral\n\
         \x20           // axis = (-sin θ, 0, cos θ). Vertical extends UP from center\n\
         \x20           // (matching Column). Slab covers:\n\
         \x20           //   0 ≤ forward ≤ thickness\n\
         \x20           //   |lateral| ≤ length/2\n\
         \x20           //   0 ≤ dy ≤ height\n\
         \x20           //\n\
         \x20           // **Spatial walk limitation.** 27-cell walk around target_slot;\n\
         \x20           // any of length, height, thickness exceeding SPATIAL_CELL_SIZE\n\
         \x20           // means candidates beyond the 27-cell footprint are missed.\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let length: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let height: f32 = ability_registry_area_args[area_args_base + 1u];\n\
         \x20           let thickness: f32 = ability_registry_area_args[area_args_base + 2u];\n\
         \x20           let facing_deg: f32 = ability_registry_area_args[area_args_base + 3u];\n\
         \x20           let half_length: f32 = length * 0.5;\n\
         \x20           let theta_rad: f32 = facing_deg * 0.01745329252; // π/180\n\
         \x20           let dir_x: f32 = cos(theta_rad);\n\
         \x20           let dir_z: f32 = sin(theta_rad);\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _to_cand: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           let _forward: f32 = _to_cand.x * dir_x + _to_cand.z * dir_z;\n\
         \x20                           if (_forward < 0.0) { continue; }\n\
         \x20                           if (_forward > thickness) { continue; }\n\
         \x20                           let _lateral: f32 = -_to_cand.x * dir_z + _to_cand.z * dir_x;\n\
         \x20                           if (abs(_lateral) > half_length) { continue; }\n\
         \x20                           if (_to_cand.y < 0.0) { continue; }\n\
         \x20                           if (_to_cand.y > height) { continue; }\n\
         \x20                           let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-wall target.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                       } // end for _i (wall walk)\n\
         \x20                   } // end for dx (wall walk)\n\
         \x20               } // end for dy (wall walk)\n\
         \x20           } // end for dz (wall walk)\n\
         \x20       } else if (area_kind == 9u) {\n\
         \x20           // Cylinder (#181). area_args layout: [radius, height, _, _].\n\
         \x20           // 3D cylinder centered on target_slot, symmetric vertically\n\
         \x20           // (`|dy| ≤ height/2`) — distinct from Column which extends UP\n\
         \x20           // only. Per-candidate gate:\n\
         \x20           //   1. dist_xz_sq = dx*dx + dz*dz ≤ radius²\n\
         \x20           //   2. |dy| ≤ height/2\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let radius: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let height: f32 = ability_registry_area_args[area_args_base + 1u];\n\
         \x20           let radius_sq: f32 = radius * radius;\n\
         \x20           let half_height: f32 = height * 0.5;\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           let _dist_xz_sq: f32 = _dvec.x * _dvec.x + _dvec.z * _dvec.z;\n\
         \x20                           if (_dist_xz_sq > radius_sq) { continue; }\n\
         \x20                           if (abs(_dvec.y) > half_height) { continue; }\n\
         \x20                           let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-cylinder target.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                       } // end for _i (cylinder walk)\n\
         \x20                   } // end for dx (cylinder walk)\n\
         \x20               } // end for dy (cylinder walk)\n\
         \x20           } // end for dz (cylinder walk)\n\
         \x20       } else if (area_kind == 10u) {\n\
         \x20           // Dome (#181). area_args layout: [radius, _, _, _]. Half-\n\
         \x20           // sphere covering the upper hemisphere above target_slot's\n\
         \x20           // horizontal plane:\n\
         \x20           //   1. dist² ≤ radius² (3D distance — Sphere gate)\n\
         \x20           //   2. dy ≥ 0 (above plane, inclusive)\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let radius: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let radius_sq: f32 = radius * radius;\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           if (dot(_dvec, _dvec) > radius_sq) { continue; }\n\
         \x20                           if (_dvec.y < 0.0) { continue; }\n\
         \x20                           let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-dome target.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                       } // end for _i (dome walk)\n\
         \x20                   } // end for dx (dome walk)\n\
         \x20               } // end for dy (dome walk)\n\
         \x20           } // end for dz (dome walk)\n\
         \x20       } else if (area_kind == 11u) {\n\
         \x20           // Hull (#181). area_args layout: [radius, _, _, _].\n\
         \x20           //\n\
         \x20           // **Semantics pinned in Task #231 (2026-05-08).** Hull is\n\
         \x20           // the **castle-footprint** per ability.md §9.2: a cube of\n\
         \x20           // half-extent `r` with the 8 corners chamfered off by a\n\
         \x20           // bevel sphere of radius `r·√2`. In-hull predicate:\n\
         \x20           //   1. Cube gate:  |dx| ≤ r ∧ |dy| ≤ r ∧ |dz| ≤ r\n\
         \x20           //   2. Bevel gate: dx² + dy² + dz² ≤ 2·r² (= (r·√2)²)\n\
         \x20           //\n\
         \x20           // Distinct from both Sphere(r) (smaller — only within r)\n\
         \x20           // and Box(r,r,r) (larger — corners up to r·√3 not clipped).\n\
         \x20           // Mirrors `apply_program_aoe_hull_filter` bit-for-bit.\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let radius: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let bevel_sq: f32 = 2.0 * radius * radius;\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           // Cube gate (half-extent r per axis).\n\
         \x20                           if (abs(_dvec.x) > radius) { continue; }\n\
         \x20                           if (abs(_dvec.y) > radius) { continue; }\n\
         \x20                           if (abs(_dvec.z) > radius) { continue; }\n\
         \x20                           // Bevel sphere gate (radius r·√2).\n\
         \x20                           if (dot(_dvec, _dvec) > bevel_sq) { continue; }\n\
         \x20                           let target_slot: u32 = _candidate;\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per in-hull target.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20                       } // end for _i (hull walk)\n\
         \x20                   } // end for dx (hull walk)\n\
         \x20               } // end for dy (hull walk)\n\
         \x20           } // end for dz (hull walk)\n\
         \x20       } else if (area_kind == 4u) {\n\
         \x20           // Spread (#183). area_args layout: [radius, max_targets, _, _]\n\
         \x20           // (max_targets is stored as f32 in the registry; cast to u32).\n\
         \x20           //\n\
         \x20           // Path B: Circle gate + sort by AgentId ascending + truncate to\n\
         \x20           // max_targets. P11 requires the kept set to be the lowest-K\n\
         \x20           // AgentIds; sorting then truncating gives that ordering on\n\
         \x20           // both backends.\n\
         \x20           //\n\
         \x20           // Per-thread implementation: each WGSL thread processes one\n\
         \x20           // ActionSelected/cast independently — there is nothing for\n\
         \x20           // peer threads to cooperate on (each cast has its own center,\n\
         \x20           // its own candidates, its own kept set). Cross-thread workgroup\n\
         \x20           // memory cooperation is therefore not applicable; the sort runs\n\
         \x20           // entirely within one thread's private storage.\n\
         \x20           //\n\
         \x20           // **Bitonic sort (task #230, 2026-05-08).** The previous shape\n\
         \x20           // used a 16-slot per-thread insertion sort (O(K²)). The new\n\
         \x20           // shape is a bitonic sort over a private 256-slot array (O(K\n\
         \x20           // log² K)) — the asymptotic win the task description called\n\
         \x20           // for, achieved without a `var<workgroup>` decl (which would\n\
         \x20           // not fit the 16 KB WebGPU minimum at workgroup_size=64 ×\n\
         \x20           // K=256 = 64 KB — outside spec). The 256-cap leaves headroom\n\
         \x20           // for fixtures with dense in-radius candidate counts.\n\
         \x20           //\n\
         \x20           // **256-slot cap (documented limitation).** The local array is\n\
         \x20           // sized for at most 256 in-radius candidates per cast. Overflow\n\
         \x20           // beyond 256 silently drops the spatial-walk-late candidates,\n\
         \x20           // which can produce non-AgentId-ordered selection. Fixtures\n\
         \x20           // targeting > 256 simultaneous candidates per cast must keep\n\
         \x20           // n_in_radius ≤ 256 to stay byte-equal with the CPU oracle.\n\
         \x20           //\n\
         \x20           // **P11 determinism.** AgentIds are unique by construction, so\n\
         \x20           // the bitonic sort never encounters ties — no secondary tie-\n\
         \x20           // break key needed. Pad slots hold `0xFFFFFFFFu` (max u32) so\n\
         \x20           // they sort to the high end and never enter the truncated\n\
         \x20           // kept set (n_emit = min(n_collected, max_targets)).\n\
         \x20           //\n\
         \x20           // **Spatial walk limitation.** 27-cell walk; if radius exceeds\n\
         \x20           // SPATIAL_CELL_SIZE candidates beyond the 27-cell footprint\n\
         \x20           // are missed (same caveat as Circle/Sphere/Box).\n\
         \x20           let area_args_base: u32 = (effect_base + i) * 4u;\n\
         \x20           let radius: f32 = ability_registry_area_args[area_args_base + 0u];\n\
         \x20           let max_targets: u32 = u32(ability_registry_area_args[area_args_base + 1u]);\n\
         \x20           let radius_sq: f32 = radius * radius;\n\
         \x20           let aoe_center: vec3<f32> = agent_pos[target_slot];\n\
         \x20           // Bitonic-sort scratch: 256-slot private array, padded with\n\
         \x20           // 0xFFFFFFFFu so unused slots sort to the high end. The cap\n\
         \x20           // is 256 (power of 2 — required for bitonic's structural\n\
         \x20           // halving); log₂(256) = 8 stages.\n\
         \x20           var collected: array<u32, 256>;\n\
         \x20           for (var _pad: u32 = 0u; _pad < 256u; _pad = _pad + 1u) {\n\
         \x20               collected[_pad] = 0xFFFFFFFFu;\n\
         \x20           }\n\
         \x20           var n_collected: u32 = 0u;\n\
         \x20           let _self_cell_f = (aoe_center + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20           let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20           let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20           let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20           for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {\n\
         \x20               for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {\n\
         \x20                   for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {\n\
         \x20                       let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20                       let _start = spatial_grid_starts[_cell];\n\
         \x20                       let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20                       for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {\n\
         \x20                           let _candidate: u32 = spatial_grid_cells[_i];\n\
         \x20                           let _cand_pos: vec3<f32> = agent_pos[_candidate];\n\
         \x20                           let _dvec: vec3<f32> = _cand_pos - aoe_center;\n\
         \x20                           if (dot(_dvec, _dvec) <= radius_sq) {\n\
         \x20                               if (n_collected < 256u) {\n\
         \x20                                   collected[n_collected] = _candidate;\n\
         \x20                                   n_collected = n_collected + 1u;\n\
         \x20                               }\n\
         \x20                               // else: silently drop pre-sort overflow.\n\
         \x20                           }\n\
         \x20                       } // end for _i (spread collect)\n\
         \x20                   } // end for dx (spread collect)\n\
         \x20               } // end for dy (spread collect)\n\
         \x20           } // end for dz (spread collect)\n\
         \x20           // Bitonic sort by AgentId ascending — O(K log² K) over the\n\
         \x20           // padded 256-slot array. log₂(256) = 8 outer stages × 8 inner\n\
         \x20           // sub-stages × 128 compare-swaps = 8192 ops; vs the prior\n\
         \x20           // insertion sort's O(K²) which would be 65 536 ops at K=256.\n\
         \x20           // Standard Batcher bitonic-sort direction rule: subsequence\n\
         \x20           // [_ii .. _ii^_step] sorts ascending when (_ii & _stage) == 0\n\
         \x20           // and descending when (_ii & _stage) != 0; the pairwise\n\
         \x20           // direction alternation produces a single ascending run after\n\
         \x20           // the final stage. Padding (0xFFFFFFFFu) drifts to the high\n\
         \x20           // end as required.\n\
         \x20           for (var _stage: u32 = 2u; _stage <= 256u; _stage = _stage << 1u) {\n\
         \x20               for (var _step: u32 = _stage >> 1u; _step > 0u; _step = _step >> 1u) {\n\
         \x20                   for (var _ii: u32 = 0u; _ii < 256u; _ii = _ii + 1u) {\n\
         \x20                       let _ixor: u32 = _ii ^ _step;\n\
         \x20                       if (_ixor > _ii) {\n\
         \x20                           let _a: u32 = collected[_ii];\n\
         \x20                           let _b: u32 = collected[_ixor];\n\
         \x20                           let _ascending: bool = (_ii & _stage) == 0u;\n\
         \x20                           // Ascending arm wants (lo, hi); descending\n\
         \x20                           // wants (hi, lo). min/max do the compare-swap\n\
         \x20                           // branchlessly.\n\
         \x20                           let _lo: u32 = min(_a, _b);\n\
         \x20                           let _hi: u32 = max(_a, _b);\n\
         \x20                           if (_ascending) {\n\
         \x20                               collected[_ii] = _lo;\n\
         \x20                               collected[_ixor] = _hi;\n\
         \x20                           } else {\n\
         \x20                               collected[_ii] = _hi;\n\
         \x20                               collected[_ixor] = _lo;\n\
         \x20                           }\n\
         \x20                       }\n\
         \x20                   }\n\
         \x20               }\n\
         \x20           }\n\
         \x20           // Truncate to max_targets and emit one chronicle record per\n\
         \x20           // kept slot via the standard arm chain (target_slot shadowed).\n\
         \x20           // Padding (0xFFFFFFFFu) sorted to the high end of the array,\n\
         \x20           // so slots [0..n_collected) hold real AgentIds in ascending\n\
         \x20           // order — clamping by min(n_collected, max_targets) keeps the\n\
         \x20           // pad out of the kept set automatically.\n\
         \x20           let n_emit: u32 = min(n_collected, max_targets);\n\
         \x20           for (var _ii: u32 = 0u; _ii < n_emit; _ii = _ii + 1u) {\n\
         \x20               // Shadow target_slot for the arm chain's chronicle writes.\n\
         \x20               let target_slot: u32 = collected[_ii];\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                                // Nested loop runs per kept spread target —\n\
         \x20                               // same shape as Circle/Cone/Box/Sphere.\n\
         \x20                               let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20                               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                                }\n\
         \x20           } // end for _ii (spread emit)\n\
         \x20       } else {\n\
         \x20           // Sentinel 0xFFu (= no area) or unrecognised shape →\n\
         \x20           // single-target fallback at the same indent the non-AOE\n\
         \x20           // path uses, just deeper (12/16 spaces) since the AOE\n\
         \x20           // arm-chain helper rendered for the in-radius branch.\n\
         \x20           // All 12 enumerated AOE shapes (Circle, Cone, Box,\n\
         \x20           // Sphere, Ring, Line, Spread, Column, Wall, Cylinder,\n\
         \x20           // Dome, Hull) have explicit branches above.\n",
    );
    s.push_str(primary_arm_chain);
    s.push_str(
        "                let nested_slot_base: u32 = nested_base + i * 2u;\n\
         \x20               for (var j: u32 = 0u; j < 2u; j = j + 1u) {\n\
         \x20                   let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
         \x20                   if (kind == 0xFFu) { continue; } // EFFECT_KIND_EMPTY\n\
         \x20                   let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
         \x20                   let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
         \x20                   let nested_scale_bonus: f32 = 0.0;\n",
    );
    s.push_str(nested_arm_chain);
    s.push_str(
        "                }\n\
         \x20       } // end if (area_kind == 0u) … else if (1u) … else if (5u) … else if (6u) … else if (3u) … else if (2u) … else if (7u) … else if (8u) … else if (9u) … else if (10u) … else if (11u) … else if (4u Spread) … else (sentinel)\n",
    );
    s
}

/// Lower a [`CgStmt::Emit`] body. **B1 no-op fallback**: the prior shape
/// `emit_event_<N>(field_<I>: <expr>, ...)` used Rust-style named-arg
/// syntax that's not valid WGSL — naga rejected every kernel that emits
/// events. Until the runtime ring-append form lands (a future task that
/// requires per-event-kind prelude functions + atomic ring append), emit
/// a phony WGSL discard per field so the body parses and the trivial-
/// fixture parity gate runs. For trivial fixtures the cascade event ring
/// is empty so this code is dead at runtime; for non-trivial fixtures
/// emitted events vanish, but that's the same B1 trade-off ViewStorage
/// Assign uses (and the same task list — Tasks 9-11).
/// Lower a `CgStmt::Emit` to a real WGSL ring-append: atomicAdd a
/// slot off `event_tail`, then write the tag + tick + payload words
/// to `event_ring[slot * stride + offset]`. Bounds-checked against
/// `event_ring_cap` so a producer that overflows the ring drops the
/// event silently (the runtime's per-tick clear ensures the ring
/// holds at most one tick's worth of events; if the cap is hit the
/// fixture is producing more events than configured for).
///
/// Bindings touched:
///   - `event_ring`: `var<storage, read_write> array<u32>`
///   - `event_tail`: `var<storage, read_write> atomic<u32>`
///   - kernel preamble-bound `tick: u32` (header word 1)
///
/// The PhysicsRule op's reads/writes table must record EventRing
/// (Append) + EventTail so the binding-generator includes both
/// bindings; without that the WGSL emitted here references undeclared
/// identifiers. See `cg/lower/physics.rs::lower_emit` for the
/// op-side metadata wire-up (Phase-8 task piece 2).
fn lower_emit_to_wgsl(
    event_id: u32,
    fields: &[(EventField, CgExprId)],
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    let kind = crate::cg::op::EventKindId(event_id);
    let layout = ctx
        .prog
        .event_layouts
        .get(&event_id)
        .ok_or(EmitError::UnregisteredEventKind { kind })?;
    let stride = layout.record_stride_u32;
    let header = layout.header_word_count;
    let buf = layout.buffer_name.as_str();
    let ordered = layout.fields_in_declaration_order();

    // Pre-evaluate every payload value-expr BEFORE touching the
    // tail counter. Lowering a value may emit auxiliary `let`s into
    // the surrounding stmt list (fold pre-pass) — doing it before
    // the atomicAdd keeps the slot-acquired window short and avoids
    // double-evaluating the expression in the bounds-check vs the
    // commit branch.
    // The producer-side `event_ring` binding is `array<atomic<u32>>`
    // (per `handle_to_binding_metadata` for EventRing-Append), so
    // ring writes go through `atomicStore(&ring[idx], value)`. Slot
    // ownership comes from the atomicAdd on `event_tail`, so the
    // atomicStore here only needs to write into a slot we already
    // own — no race vs. other producers.
    let mut field_writes: Vec<String> = Vec::with_capacity(fields.len());
    for (field_ref, expr_id) in fields {
        let layout_entry = ordered
            .get(field_ref.index as usize)
            .ok_or(EmitError::UnregisteredEventKind { kind })?;
        let (_name, fl) = layout_entry;
        let value_wgsl = lower_cg_expr_to_wgsl(*expr_id, ctx)?;
        let off = header + fl.word_offset_in_payload;
        let store = |out: &mut Vec<String>, off: u32, val: String| {
            out.push(format!(
                "    atomicStore(&{buf}[slot * {stride}u + {off}u], {val});",
            ));
        };
        match fl.ty {
            CgTy::AgentId | CgTy::U32 | CgTy::Tick => {
                store(&mut field_writes, off, format!("({value_wgsl})"));
            }
            CgTy::I32 | CgTy::F32 => {
                store(
                    &mut field_writes,
                    off,
                    format!("bitcast<u32>({value_wgsl})"),
                );
            }
            CgTy::Vec3F32 => {
                // Materialize once so we don't re-evaluate the
                // source vec3 expression three times across the
                // .x/.y/.z stores.
                let tmp = format!("_emit_v_{}_{}", event_id, field_ref.index);
                field_writes
                    .push(format!("    let {tmp}: vec3<f32> = ({value_wgsl});"));
                store(&mut field_writes, off, format!("bitcast<u32>({tmp}.x)"));
                store(&mut field_writes, off + 1, format!("bitcast<u32>({tmp}.y)"));
                store(&mut field_writes, off + 2, format!("bitcast<u32>({tmp}.z)"));
            }
            CgTy::Bool => {
                store(
                    &mut field_writes,
                    off,
                    format!("select(0u, 1u, ({value_wgsl}))"),
                );
            }
            CgTy::ViewKey { .. } => {
                return Err(EmitError::EventFieldUnsupportedType {
                    kind,
                    word_offset_in_payload: fl.word_offset_in_payload,
                    got: fl.ty,
                });
            }
        }
    }

    // For the seq trailer, use the correct per-thread index expression:
    // PerEvent dispatch loops over `event_idx`; all others use `agent_id`.
    let thread_idx_expr = match ctx.dispatch.get() {
        Some(crate::cg::dispatch::DispatchShape::PerEvent { .. }) => "event_idx",
        _ => "agent_id",
    };
    let producer_kernel_id = ctx.current_kernel_index.get()
        .and_then(|ki| ctx.producer_kernel_ids.get(&ki).copied())
        .unwrap_or(0);
    let emit_idx = ctx.intra_emit_idx.get();
    ctx.intra_emit_idx.set(emit_idx + 1);
    Ok(emit_chronicle_append_skeleton(
        event_id,
        buf,
        stride,
        fields.len(),
        &field_writes,
        ctx.debug_wgsl,
        producer_kernel_id,
        emit_idx,
        thread_idx_expr,
    ))
}

/// Render the chronicle-ring atomic-append skeleton for an event of a
/// given kind. Pure WGSL string-builder — takes the event id, the SoA
/// buffer name (`buf`), the per-record stride in u32-words, and a
/// pre-built list of field-write lines (each already starting with
/// 4-space indent and including its trailing semicolon).
///
/// Shape:
/// ```wgsl
/// // emit event#<event_id> (N fields)
/// {
///     let slot = atomicAdd(&event_tail[0], 1u);
///     if (slot < <CAP>u) {
///         atomicStore(&<buf>[slot * <stride>u + 0u], <event_id>u);
///         atomicStore(&<buf>[slot * <stride>u + 1u], tick);
///         <field_writes…>
///     }
/// }
/// ```
///
/// Used by:
///   - `lower_emit_to_wgsl` — the canonical compile-time-known-event
///     emit path. Field values are CG-lowered then handed to this
///     helper as pre-rendered strings.
///   - The #136 ApplyAbility dispatcher (slice γ + δ follow-ups) —
///     each branch arm constructs the field-write lines from
///     `payload_a/b` decodes and calls this helper with the matching
///     kind/buf/stride. Without the shared helper, every dispatcher
///     arm would duplicate the atomicAdd / bounds-check / header-
///     write boilerplate; centralizing keeps slot-acquisition
///     semantics consistent across both paths.
///
/// `field_count` is purely cosmetic — used in the header comment for
/// frame-capture readability.
///
/// `debug_wgsl` (Compiler debug mode Phase 2): when
/// [`crate::cg::lower::driver::DebugWgslFlags::event_kind_histogram`]
/// is set, emits a parallel `atomicAdd(&event_kind_counts[<event_id>], 1u)`
/// alongside the existing tail bump. The counter buffer must be
/// declared by the kernel's BGL synthesis when any chronicle producer
/// in the kernel body has the flag set; the BGL fanout is deferred to
/// the per-runtime opt-in slice (no production runtime opts in
/// today). Default [`crate::cg::lower::driver::DebugWgslFlags::NONE`]
/// emits the existing skeleton verbatim — bit-for-bit identical to
/// the prior shape.
pub(crate) fn emit_chronicle_append_skeleton(
    event_id: u32,
    buf: &str,
    stride: u32,
    field_count: usize,
    field_writes: &[String],
    debug_wgsl: DebugWgslFlags,
    producer_kernel_id: u32,
    intra_emit_idx: u32,
    // WGSL expression for the per-thread producer index used in the seq
    // trailer packing (`(kernel_id << 24) | (thread_idx << 4) | emit_idx`).
    // For `PerAgent` dispatch kernels this is `"agent_id"`. For `PerEvent`
    // dispatch kernels (`@phase(post)`) this is `"event_idx"`.
    thread_idx_expr: &str,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("// emit event#{event_id} ({field_count} fields)\n"));
    out.push_str("{\n");
    out.push_str("    let slot = atomicAdd(&event_tail[0], 1u);\n");
    if debug_wgsl.event_kind_histogram {
        // Phase 2 debug instrumentation: bump the per-kind histogram
        // alongside the ring's tail counter. The increment is
        // observation-only (P5) and the atomic is commutative so
        // counts remain deterministic across thread orderings (P11).
        out.push_str(&format!(
            "    // debug_wgsl.event_kind_histogram: per-kind chronicle counter\n\
             \x20   atomicAdd(&event_kind_counts[{event_id}u], 1u);\n"
        ));
    }
    out.push_str(&format!(
        "    if (slot < {}u) {{\n",
        DEFAULT_EVENT_RING_CAP_SLOTS
    ));
    out.push_str(&format!(
        "        atomicStore(&{buf}[slot * {stride}u + 0u], {event_id}u);\n"
    ));
    out.push_str(&format!(
        "        atomicStore(&{buf}[slot * {stride}u + 1u], tick);\n"
    ));
    for line in field_writes {
        out.push_str(&format!("    {line}\n"));
    }
    // Seq trailer: deterministic ordering key for the per-tick sort.
    // `thread_idx_expr` is the producer thread's per-kernel index
    // (`agent_id` for PerAgent dispatch, `event_idx` for PerEvent dispatch).
    // The packing matches the Rust `compute_event_seq` helper byte-for-byte:
    //   (kernel_id << 24) | (thread_idx << 4) | emit_idx
    out.push_str(&format!(
        "        atomicStore(&{buf}[slot * {stride}u + {seq_offset}u], \
         ({kernel_id}u << 24u) | ({thread_idx} << 4u) | {emit_idx}u);\n",
        seq_offset = stride - 1,
        kernel_id = producer_kernel_id,
        thread_idx = thread_idx_expr,
        emit_idx = intra_emit_idx,
    ));
    out.push_str("    }\n");
    out.push_str("}");
    out
}

/// Default event-ring slot capacity — 1 048 576 events per tick.
/// The runtime sizes the `event_ring` buffer to `cap * stride * 4`
/// bytes; the WGSL emitter bounds-checks `slot < cap` to silently
/// drop overflow producers.
///
/// Bumped 65 536 → 1 048 576 (16×) on 2026-05-09 (Task #239) — the
/// previous cap saturated at agent_cap ≈ 480-500 in max-density AOE
/// fixtures (per stress sweep at `crates/stress_cast_density_runtime/`).
/// At 40 MB GPU mem per fixture, this is < 1% of typical VRAM.
///
/// MUST stay in sync with `engine::gpu::event_ring::EVENT_RING_CAP_SLOTS`
/// AND with the 47 hardcoded `if (_slot < 1048576u)` literals in this
/// file (search `1048576u`). Future raises: bump all three together.
const DEFAULT_EVENT_RING_CAP_SLOTS: u32 = 1_048_576;

/// Sibling-emitter accessor for [`DEFAULT_EVENT_RING_CAP_SLOTS`].
///
/// The scoring-argmax body emit (in `kernel.rs`) inlines its own
/// ring-append for the verb-expander-injected `ActionSelected` event
/// (it doesn't route through `lower_emit_to_wgsl` because the emit
/// happens after the per-row argmax loop, outside any `CgStmt::Emit`
/// in the IR). Both producers must agree on the same cap so the
/// runtime's single-buffer sizing covers the worst case from either
/// path.
pub(crate) fn default_event_ring_cap_slots() -> u32 {
    DEFAULT_EVENT_RING_CAP_SLOTS
}

/// `(EffectOp discriminant, runtime EventKindId)` pairs for the
/// chronicle-bearing variants the `apply_ability` dispatcher emits
/// records for. Sourced from:
///   - left  — `pack_effect` in `crates/engine/src/ability/packed.rs`
///     (the schema-pinned `#[repr(u8)]` ordinal each `EffectOp` packs to)
///   - right — `EventKindId` in `crates/engine/src/cascade/handler.rs`
///     (the runtime's chronicle-ring kind tag)
///
/// Only the variants whose runtime apply path produces a 1:1
/// `Event::EffectXxxApplied` chronicle record appear here. Variants
/// whose apply path produces a different shape (e.g. `Dash` writes to
/// position SoA + an `AgentMoved` event; `Buff` writes to per-agent
/// buff SoA without a dedicated chronicle kind today) are absent —
/// slice γ's first wire-up will only thread the entries that have a
/// 1:1 mapping here. Adding new entries means: (a) the runtime grows
/// a new `Event::Effect*Applied` kind, (b) `pack_effect`'s discriminant
/// for that variant is unchanged, (c) the dispatcher arm for that
/// variant calls `chronicle_append` against the new kind.
///
/// Pinned by `effect_kind_to_event_kind_map_matches_engine` (see
/// the test module below) so a divergence between this table and
/// either source-of-truth surfaces as a CI failure rather than a
/// silent run-time mismatch.
///
/// `#[allow(dead_code)]`: dead at HEAD because the dispatcher arms
/// still emit `// TODO slice γ: chronicle_append_*` placeholders
/// rather than indexing this map. The pin keeps the table on file
/// (and the cross-crate test enforcing it active) so the slice γ
/// wire-up has a vetted starting point — the moment the first arm
/// replaces its TODO with a real `emit_chronicle_append_skeleton`
/// call sourcing `event_id` from this table, the lint clears.
#[allow(dead_code)]
pub(crate) const EFFECT_KIND_TO_EVENT_KIND_ID: &[(u32, u32)] = &[
    // EffectOp::Damage          → EventKindId::EffectDamageApplied
    (0,  26),
    // EffectOp::Heal            → EventKindId::EffectHealApplied
    (1,  27),
    // EffectOp::Shield          → EventKindId::EffectShieldApplied
    (2,  28),
    // EffectOp::Stun            → EventKindId::EffectStunApplied
    (3,  29),
    // EffectOp::Slow            → EventKindId::EffectSlowApplied
    (4,  30),
    // EffectOp::TransferGold    → EventKindId::EffectGoldTransfer
    (5,  31),
    // EffectOp::ModifyStanding  → EventKindId::EffectStandingDelta
    (6,  32),
    // EffectOp::SelfDamage      → EventKindId::EffectSelfDamageApplied
    // (Bleed verb swap, Task #138 follow-on, 2026-05-06).
    (17, 39),
    // EffectOp::LifeSteal       → EventKindId::EffectLifeStealApplied
    // (Vampirize verb swap, Task #138 follow-on, mirror of Bleed).
    (18, 40),
    // EffectOp::DamageModify    → EventKindId::EffectDamageModifyApplied
    // (Fortify verb swap, Task #138 follow-on, mirror of Vampirize).
    (19, 41),
    // EffectOp::Execute         → EventKindId::EffectExecuteApplied
    // (Reap verb swap, Task #138 follow-on, mirror of Fortify). Closes
    // the slice across all 8 duel_abilities verbs.
    (16, 42),
    // Wave 2 piece 1 — control statuses. Each shares Stun's shape
    // (kind=3 → 29) but lands on a unique EventKindId so consumer
    // physics rules can disambiguate. The packed effect-kind ordinals
    // (Root=8, Silence=9, Fear=10, Taunt=11) come from
    // `pack_effect` in `crates/engine/src/ability/packed.rs`; the
    // dispatcher arm bodies for these in `emit_chronicle_arm_chain`
    // (below) match these ordinals via `kind == 8u..=11u`.
    (8,  43), // EffectOp::Root    → EventKindId::EffectRootApplied
    (9,  44), // EffectOp::Silence → EventKindId::EffectSilenceApplied
    (10, 45), // EffectOp::Fear    → EventKindId::EffectFearApplied
    (11, 46), // EffectOp::Taunt   → EventKindId::EffectTauntApplied
    // Wave 2 piece 2 — movement EffectOps. Dash/Blink are caster-self
    // motion (payload = actor + f32 distance). Knockback/Pull are
    // forced motion on a target (payload = actor + target + f32
    // distance). The packed effect-kind ordinals (Dash=12, Blink=13,
    // Knockback=14, Pull=15) come from `pack_effect` in
    // `crates/engine/src/ability/packed.rs`; the dispatcher arm
    // bodies for these in `emit_chronicle_arm_chain` (below) match
    // these ordinals via `kind == 12u..=15u`.
    (12, 47), // EffectOp::Dash      → EventKindId::EffectDashApplied
    (13, 48), // EffectOp::Blink     → EventKindId::EffectBlinkApplied
    (14, 49), // EffectOp::Knockback → EventKindId::EffectKnockbackApplied
    (15, 50), // EffectOp::Pull      → EventKindId::EffectPullApplied
    // Wave 1.5+ — multi-tick effects. DamageOverTime / HealOverTime
    // share a 4-payload-word shape (actor + target + amount-per-tick
    // f32 + duration_ticks u32). TimedShield has the same payload
    // shape with `amount` as the one-shot shield magnitude (with
    // scale_bonus already folded in by the existing arm). The packed
    // effect-kind ordinals (DamageOverTime=20, HealOverTime=21,
    // TimedShield=22) come from `pack_effect` in
    // `crates/engine/src/ability/packed.rs`; the dispatcher arm
    // bodies for these in `emit_chronicle_arm_chain` (below) match
    // these ordinals via `kind == 20u..=22u`.
    (20, 51), // EffectOp::DamageOverTime → EventKindId::EffectDamageOverTimeApplied
    (21, 52), // EffectOp::HealOverTime   → EventKindId::EffectHealOverTimeApplied
    (22, 53), // EffectOp::TimedShield    → EventKindId::EffectTimedShieldApplied
    // Extended-corpus statuses — Stealth (caster-self) plus Charm/
    // Grounded/Suppress (target-cast). Stealth shares Dash's payload
    // shape: actor + payload_a-as-u32 (duration_ticks here, not bitcast
    // f32 like Dash's distance) at slot 3, no target word. Charm/
    // Grounded/Suppress share Stun's 3-payload-word shape (actor +
    // target + duration_ticks at slot 4) but store raw `duration_ticks`
    // rather than `expires_at_tick` — consistent with the multi-tick
    // effect family (DoT/HoT/TimedShield, kinds 51..53). The packed
    // effect-kind ordinals (Stealth=27, Charm=28, Grounded=29,
    // Suppress=30) come from `pack_effect` in
    // `crates/engine/src/ability/packed.rs`; the dispatcher arm bodies
    // for these in `emit_chronicle_arm_chain` (below) match these
    // ordinals via `kind == 27u..=30u`.
    (27, 54), // EffectOp::Stealth   → EventKindId::EffectStealthApplied
    (28, 55), // EffectOp::Charm     → EventKindId::EffectCharmApplied
    (29, 56), // EffectOp::Grounded  → EventKindId::EffectGroundedApplied
    (30, 57), // EffectOp::Suppress  → EventKindId::EffectSuppressApplied
    // Slice γ tail — Buff/Harvest/PlaceVoxel/Reflect. Four distinct
    // shapes:
    //   - Buff (kind 23 → ID 58): target-cast with packed payload.
    //     The dispatcher writes raw `payload_a` (which packs
    //     `stat_ordinal` in low byte | `magnitude_q8` in bits 8..) and
    //     raw `payload_b` (= duration_ticks) — consumer rules decode
    //     the packed bits.
    //   - Harvest (kind 25 → ID 59): caster-self resource gather.
    //     `payload_a` = kind_hash (u32 FxHash of the resource ident),
    //     `payload_b` = amount (u32, widened from u16 EffectOp side).
    //     No target field on the engine event.
    //   - PlaceVoxel (kind 26 → ID 60): caster-self voxel placement.
    //     `payload_a` = kind_hash; placement at cast's target world
    //     position (implicit, not in the chronicle record). No target
    //     field on the engine event.
    //   - Reflect (kind 31 → ID 61): target-cast fraction-of-damage
    //     bounce. `payload_a` = duration_ticks (u32), `payload_b`'s
    //     low 16 bits = fraction_q8 (i16, sign-extended on read).
    //     Same shape family as Slow/LifeSteal/DamageModify (duration
    //     + signed q8 fraction/multiplier) — chronicle stores raw u32
    //     payloads, consumer sign-extends.
    //
    // Buff / Reflect carry packed payloads with signed sub-fields —
    // the chronicle ring stores raw u32 (= `payload_a` / `payload_b`
    // verbatim from the dispatcher's effect_payload_a/b SoA columns)
    // and consumers downcast/sign-extend on read. No decomposition at
    // dispatch time — the dispatcher arm bodies write the raw words.
    //
    // The packed effect-kind ordinals (Buff=23, Harvest=25, PlaceVoxel
    // =26, Reflect=31) come from `pack_effect` in
    // `crates/engine/src/ability/packed.rs`; the dispatcher arm bodies
    // for these in `emit_chronicle_arm_chain` (below) match these
    // ordinals via `kind == 23u | 25u | 26u | 31u`.
    (23, 58), // EffectOp::Buff       → EventKindId::EffectBuffApplied
    (25, 59), // EffectOp::Harvest    → EventKindId::EffectHarvestApplied
    (26, 60), // EffectOp::PlaceVoxel → EventKindId::EffectPlaceVoxelApplied
    (31, 61), // EffectOp::Reflect    → EventKindId::EffectReflectApplied
    // Slice γ closer — Summon (kind 24 → ID 62), the last `// TODO
    // slice γ` placeholder in the dispatcher arm chain. Caster-self
    // with packed payload (5 payload words: actor + template_hash +
    // count + lifetime_ticks). The earlier "multi-spawn semantics need
    // a new dispatch shape" deferral was misleading — per
    // `crates/engine/src/ability/apply.rs`, the CPU side writes ONE
    // `ApplyEvent::Summon` per cast carrying the packed (count,
    // lifetime), exactly the same shape family as Buff (kind 23 →
    // 58). Downstream N-entity spawning is a separate consumer
    // concern, distinct from the dispatcher's chronicle-record write.
    // The packed effect-kind ordinal Summon=24 comes from
    // `pack_effect` in `crates/engine/src/ability/packed.rs`; the
    // dispatcher arm body in `emit_chronicle_arm_chain` (below)
    // matches via `kind == 24u`.
    (24, 62), // EffectOp::Summon     → EventKindId::EffectSummonApplied
    // Wave 3 ToM Phase 1 — `plant_belief` bit-flag primitive (kind 32 →
    // ID 63). Caster CAUSES target's belief map for `subject_idx` to
    // gain `1u << fact_bit` via atomic-OR. The packed effect-kind
    // ordinal (PlantBelief=32) comes from `pack_effect` in
    // `crates/engine/src/ability/packed.rs`; the dispatcher arm body
    // for this in `emit_chronicle_arm_chain` (below) matches via
    // `kind == 32u`. 5-payload-word chronicle shape (caster + target +
    // subject_idx (= payload_a) + fact_bit_mask (= payload_b, pre-
    // shifted at pack time so the dispatcher and downstream view
    // consumer's `self |= b` body don't re-shift). The pair_map
    // atomicOr lives in the downstream view consumer (existing
    // `tom_probe.sim::beliefs` fold pattern), NOT in this arm — the
    // dispatcher only writes the chronicle record. This keeps the
    // apply_ability dispatcher's BGL stable (no new pair_map binding)
    // and routes the bit-fold through the standard view-fold pipeline.
    (32, 63), // EffectOp::PlantBelief → EventKindId::EffectPlantBeliefApplied
    // Wave 3 ToM Phase 3 — `observe` self-observe-target verb (kind 33 →
    // ID 64). Caster refreshes its own belief row about target. The
    // packed effect-kind ordinal (Observe=33) comes from `pack_effect`
    // in `crates/engine/src/ability/packed.rs`; the dispatcher arm
    // body for this in `emit_chronicle_arm_chain` (below) matches via
    // `kind == 33u`. 4-payload-word chronicle shape (caster + target +
    // tick + target_observer u8 in payload_a). The downstream
    // BeliefState SoA writeback (reading target's pos / creature_type
    // from agent SoA, writing into the 6 columns at
    // `[caster * agent_cap + target]`) lives in a runtime consumer
    // (`tom_probe_runtime` Phase 3) — not on the WGSL fold path. The
    // dispatcher only writes the chronicle record; this keeps the
    // apply_ability dispatcher's BGL stable (no new SoA bindings).
    (33, 64), // EffectOp::Observe → EventKindId::EffectObserveApplied
    // Wave 3 ToM Phase 3.5 — `scry` cross-observer access (kind 34 → ID
    // 65). Caster reads `target_observer`'s beliefs about
    // `subject_idx`; the dispatcher writes the chronicle record with
    // payload_a = target_observer u8 widened, payload_b = subject_idx
    // u32. The downstream 6-column copy from `[target_observer * N +
    // subject_idx]` to `[caster * N + subject_idx]` lives in a runtime
    // consumer (`tom_probe_runtime` Phase 3.5) — not on the WGSL fold
    // path. The dispatcher only writes the chronicle record; this keeps
    // the apply_ability dispatcher's BGL stable (no new SoA bindings).
    (34, 65), // EffectOp::Scry → EventKindId::EffectScryApplied
    // Wave 3 ToM Phase 3.5 — `reveal` one-to-many propagation (kind 35
    // → ID 66). Caster broadcasts its beliefs about `subject_idx`; the
    // dispatcher writes the chronicle record with payload_a =
    // subject_idx u32, payload_b = 0. The downstream fan-out (caster's
    // beliefs about subject → every observer's beliefs about subject)
    // lives in a runtime consumer (`tom_probe_runtime` Phase 3.5).
    (35, 66), // EffectOp::Reveal → EventKindId::EffectRevealApplied
    // Wave 3 ToM Phase 4 — deception verbs (Disguise/Decoy/EraseBelief).
    // Each is the chronicle counterpart of the matching `EffectOp` slot
    // (kinds 36/37/38). The dispatcher writes a single record per cast;
    // downstream BeliefState SoA mutation lives in compiler-emitted
    // `physics @phase(post)` consumer rules in `tom_probe.sim` (mirror
    // of Phase 3.8 observe/scry/reveal authoring).
    (36, 67), // EffectOp::Disguise    → EventKindId::EffectDisguiseApplied
    (37, 68), // EffectOp::Decoy       → EventKindId::EffectDecoyApplied
    (38, 69), // EffectOp::EraseBelief → EventKindId::EffectEraseBeliefApplied
    // Lift A — multi-tick travel. Dispatcher writes one chronicle
    // record per cast (kind=70). The downstream consumer rule sets
    // `busy_until_tick` and populates `travel_dest_{x,y,z}` SoA cells;
    // a per-tick travel kernel interpolates `pos` toward the destination
    // over `eta_ticks` ticks.
    (39, 70), // EffectOp::TravelTo    → EventKindId::EffectTravelToApplied
    // Lift B — items / inventory + production / recipes. Dispatcher
    // writes one chronicle record per cast for each verb. Per-fixture
    // consumer rules read `RecipeRegistry[recipe_id]` (Recipe) or look
    // up the caster's tool of `tool_kind` (WearTool) and emit the
    // inventory / wear deltas. See `docs/spec/economy.md §4.1` (recipes)
    // + §4.3 (capital goods).
    (40, 71), // EffectOp::Recipe      → EventKindId::EffectRecipeApplied
    (41, 72), // EffectOp::WearTool    → EventKindId::EffectWearToolApplied
    // Lift C — bilateral consent + observer fan-out. Dispatcher writes
    // one chronicle record per cast for each verb. Per-fixture consumer
    // rules register the proposal in a ContractRegistry (Propose) or
    // walk the spatial-hash and emit per-observer perception events
    // (Announce). See `docs/spec/economy.md §6` (observer fan-out) +
    // §7 (contracts).
    (42, 73), // EffectOp::Propose     → EventKindId::EffectProposeApplied
    (43, 74), // EffectOp::Announce    → EventKindId::EffectAnnounceApplied
    // Lift D — knowledge / skills + obligation registry. Self-cast
    // skill growth + persistent agent-to-agent obligation registration.
    // The packed effect-kind ordinals (GainSkill=44, CreateObligation=45)
    // come from `pack_effect` in `crates/engine/src/ability/packed.rs`;
    // the dispatcher arm bodies for these in `emit_chronicle_arm_chain`
    // (below) match these ordinals via `kind == 44u..=45u`.
    (44, 75), // EffectOp::GainSkill        → EventKindId::EffectGainSkillApplied
    (45, 76), // EffectOp::CreateObligation → EventKindId::EffectCreateObligationApplied
    // Plan G (2026-05-09) — generic deferred-cast intent. Dispatcher
    // writes one chronicle record per cast initiation. Consumer
    // (compiler-emitted physics_BusyTick kernel, G2.3+) reads the
    // record + sets per-agent busy SoA + emits CastBegan as the
    // public lifecycle event.
    (46, 77), // EffectOp::CastBegin        → EventKindId::EffectCastBeginApplied
];

/// Look up the runtime `EventKindId` for an `EffectOp` discriminant.
/// Returns `None` for variants that have no 1:1 chronicle counterpart
/// today (the dispatcher arms for those keep their `// TODO slice γ`
/// markers until a future runtime change adds the kind).
///
/// Used by the dispatcher arms to render the `event_id` constant in
/// the chronicle_append skeleton without re-stating the mapping each
/// time.
pub(crate) fn event_kind_id_for_effect_kind(effect_kind: u32) -> Option<u32> {
    EFFECT_KIND_TO_EVENT_KIND_ID
        .iter()
        .find(|(ek, _)| *ek == effect_kind)
        .map(|(_, vk)| *vk)
}

/// Wave 1.5#9: render the apply_ability dispatcher's per-effect
/// `if (kind == X)` arm-chain at the given indent prefix. Reads
/// `kind`, `payload_a`, `payload_b`, `tick`, `caster_slot`,
/// `target_slot`, **`ability_id__u32`** as outer-scope WGSL
/// identifiers and writes chronicle records via
/// `atomicAdd(&event_tail[0], 1u)` + `atomicStore(&event_ring[...])`
/// per chronicle-bearing variant.
///
/// **Slot 6 = ability_id (Gap detective#6, 2026-05-12).** Every arm's
/// chronicle write ends with
/// `atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);`.
/// Downstream chronicle consumers (e.g. detective_investigation's
/// `ApplyDamageFromChronicle` rule that re-emits Accused on damage)
/// read this slot to discriminate which verb produced the record.
/// Pre-fix, all damage records looked alike — Accuse/Investigate/
/// Observe over-counted accusations 3x. The 10-word per-record
/// stride is unchanged (slots 7..9 stay reserved); the field lands
/// at the first previously-unused slot.
///
/// Reused by the primary effect walk and the nested-effect walk
/// (`nested_per_effect[i]` SoA) — both produce identical chronicle
/// records given identical (kind, payload_a, payload_b, caster,
/// target, tick) tuples, so the arm-chain emit is single-source.
///
/// `indent` is the leading whitespace per line — `"        "` (8
/// spaces) for the primary walk inside `for (var i ...) {`, and
/// `"            "` (12 spaces) for the nested walk inside the inner
/// `for (var j ...) {`. The arm-chain has its own internal extra
/// indent stride (4 + 4 + 4 spaces) for the if-body, atomicAdd
/// block, and atomicStore lines respectively.
///
/// `scale_bonus_var` is the WGSL identifier (in scope at `indent`) that
/// holds the per-effect-slot `Σ percent * caster_stat` bonus, added to
/// the f32 `amount` field of every amount-bearing chronicle arm
/// (Damage / Heal / Shield / SelfDamage / DamageOverTime / HealOverTime /
/// TimedShield). The primary walk passes `"scale_bonus"` (computed
/// from `scaling_stat_refs`/`scaling_percents` SoA + per-stat agent
/// SoA reads at `caster_slot` above the chain); the nested walk passes
/// `"nested_scale_bonus"` which is forced to `0.0` because nested ops
/// have no scaling slot in the registry today (mirrors the CPU's
/// `apply.rs` line ~233-237 — `push_effect_event(... 0.0)` for nested).
///
/// Pinned by `apply_ability_dispatcher_emits_chronicle_arms_test`
/// (and the various other dispatcher tests) — any per-arm payload
/// drift surfaces there. The chain is structurally identical to
/// `pack_effect`'s variant ordering in
/// `crates/engine/src/ability/packed.rs`.
fn emit_chronicle_arm_chain(
    indent: &str,
    scale_bonus_var: &str,
    debug_wgsl: DebugWgslFlags,
    producer_kernel_id: u32,
    first_emit_idx: u32,
) -> String {
    let damage_event_id = event_kind_id_for_effect_kind(0)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Damage=0");
    let heal_event_id = event_kind_id_for_effect_kind(1)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Heal=1");
    let shield_event_id = event_kind_id_for_effect_kind(2)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Shield=2");
    let stun_event_id = event_kind_id_for_effect_kind(3)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Stun=3");
    let slow_event_id = event_kind_id_for_effect_kind(4)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Slow=4");
    let transfer_gold_event_id = event_kind_id_for_effect_kind(5)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain TransferGold=5");
    let modify_standing_event_id = event_kind_id_for_effect_kind(6)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain ModifyStanding=6");
    let self_damage_event_id = event_kind_id_for_effect_kind(17)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain SelfDamage=17");
    let life_steal_event_id = event_kind_id_for_effect_kind(18)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain LifeSteal=18");
    let damage_modify_event_id = event_kind_id_for_effect_kind(19)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain DamageModify=19");
    let execute_event_id = event_kind_id_for_effect_kind(16)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Execute=16");
    let root_event_id = event_kind_id_for_effect_kind(8)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Root=8");
    let silence_event_id = event_kind_id_for_effect_kind(9)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Silence=9");
    let fear_event_id = event_kind_id_for_effect_kind(10)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Fear=10");
    let taunt_event_id = event_kind_id_for_effect_kind(11)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Taunt=11");
    let dash_event_id = event_kind_id_for_effect_kind(12)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Dash=12");
    let blink_event_id = event_kind_id_for_effect_kind(13)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Blink=13");
    let knockback_event_id = event_kind_id_for_effect_kind(14)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Knockback=14");
    let pull_event_id = event_kind_id_for_effect_kind(15)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Pull=15");
    let damage_over_time_event_id = event_kind_id_for_effect_kind(20)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain DamageOverTime=20");
    let heal_over_time_event_id = event_kind_id_for_effect_kind(21)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain HealOverTime=21");
    let timed_shield_event_id = event_kind_id_for_effect_kind(22)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain TimedShield=22");
    let stealth_event_id = event_kind_id_for_effect_kind(27)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Stealth=27");
    let charm_event_id = event_kind_id_for_effect_kind(28)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Charm=28");
    let grounded_event_id = event_kind_id_for_effect_kind(29)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Grounded=29");
    let suppress_event_id = event_kind_id_for_effect_kind(30)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Suppress=30");
    let buff_event_id = event_kind_id_for_effect_kind(23)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Buff=23");
    let harvest_event_id = event_kind_id_for_effect_kind(25)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Harvest=25");
    let place_voxel_event_id = event_kind_id_for_effect_kind(26)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain PlaceVoxel=26");
    let reflect_event_id = event_kind_id_for_effect_kind(31)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Reflect=31");
    let summon_event_id = event_kind_id_for_effect_kind(24)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Summon=24");
    let plant_belief_event_id = event_kind_id_for_effect_kind(32)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain PlantBelief=32");
    let observe_event_id = event_kind_id_for_effect_kind(33)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Observe=33");
    let scry_event_id = event_kind_id_for_effect_kind(34)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Scry=34");
    let reveal_event_id = event_kind_id_for_effect_kind(35)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Reveal=35");
    let disguise_event_id = event_kind_id_for_effect_kind(36)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Disguise=36");
    let decoy_event_id = event_kind_id_for_effect_kind(37)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Decoy=37");
    let erase_belief_event_id = event_kind_id_for_effect_kind(38)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain EraseBelief=38");
    let travel_to_event_id = event_kind_id_for_effect_kind(39)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain TravelTo=39");
    let recipe_event_id = event_kind_id_for_effect_kind(40)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Recipe=40");
    let wear_tool_event_id = event_kind_id_for_effect_kind(41)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain WearTool=41");
    let propose_event_id = event_kind_id_for_effect_kind(42)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Propose=42");
    let announce_event_id = event_kind_id_for_effect_kind(43)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain Announce=43");
    let gain_skill_event_id = event_kind_id_for_effect_kind(44)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain GainSkill=44");
    let create_obligation_event_id = event_kind_id_for_effect_kind(45)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain CreateObligation=45");
    let cast_begin_event_id = event_kind_id_for_effect_kind(46)
        .expect("EFFECT_KIND_TO_EVENT_KIND_ID must contain CastBegin=46");

    let i4  = indent;                   // arm `if`/`else if` lines
    let i8  = format!("{i4}    ");      // body of arm
    let i12 = format!("{i4}        ");  // inside chronicle-write `{`
    let i16 = format!("{i4}            "); // inside `if (_slot < 1048576u)`

    let mut s = String::new();

    // Compiler debug mode Phase 2: per-arm helper that emits the
    // event-kind histogram bump just below the slot acquisition. Empty
    // when `event_kind_histogram=false` so the existing arm-chain
    // shape is bit-for-bit unchanged for non-opt-in fixtures.
    //
    // Each arm's `<kind>_event_id` is statically known at emit time
    // (resolved via `event_kind_id_for_effect_kind` above), so the
    // bump targets a known index in `event_kind_counts`. The atomic
    // is commutative + thread-order-independent (P11) so counts are
    // deterministic across launches.
    let hist_bump = |event_id: u32| -> String {
        if debug_wgsl.event_kind_histogram {
            format!(
                "{i12}// debug_wgsl.event_kind_histogram: per-kind chronicle counter\n\
                 {i12}atomicAdd(&event_kind_counts[{event_id}u], 1u);\n"
            )
        } else {
            String::new()
        }
    };

    // Damage = 0 → 26
    s.push_str(&format!("{i4}// Damage = 0 → EventKindId::EffectDamageApplied = 26\n"));
    s.push_str(&format!("{i4}if (kind == 0u) {{\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectDamageApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(damage_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {damage_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 0));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Heal = 1 → 27
    s.push_str(&format!("{i4}}} else if (kind == 1u) {{\n"));
    s.push_str(&format!("{i8}// Heal = 1 → EventKindId::EffectHealApplied = 27\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectHealApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(heal_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {heal_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 1));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Shield = 2 → 28
    s.push_str(&format!("{i4}}} else if (kind == 2u) {{\n"));
    s.push_str(&format!("{i8}// Shield = 2 → EventKindId::EffectShieldApplied = 28\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectShieldApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(shield_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {shield_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 2));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Stun = 3 → 29
    s.push_str(&format!("{i4}}} else if (kind == 3u) {{\n"));
    s.push_str(&format!("{i8}// Stun = 3 → EventKindId::EffectStunApplied = 29\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); expires_at_tick = tick + duration\n"));
    s.push_str(&format!("{i8}let expires_at_tick: u32 = tick + payload_a;\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectStunApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(stun_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {stun_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 3));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Slow = 4 → 30
    s.push_str(&format!("{i4}}} else if (kind == 4u) {{\n"));
    s.push_str(&format!("{i8}// Slow = 4 → EventKindId::EffectSlowApplied = 30\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); expires = tick + duration\n"));
    s.push_str(&format!("{i8}// payload_b sign-widened i16 → factor_q8 (i32 via bitcast)\n"));
    s.push_str(&format!("{i8}let expires_at_tick: u32 = tick + payload_a;\n"));
    s.push_str(&format!("{i8}let factor_q8: i32 = bitcast<i32>(payload_b);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectSlowApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(slow_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {slow_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], bitcast<u32>(factor_q8));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 4));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Wave 2 piece 1 — control statuses (Root/Silence/Fear/Taunt).
    // Each mirrors Stun (kind == 3u): payload_a = duration_ticks (u32),
    // expires_at_tick = tick + duration. 3-payload-word chronicle write
    // (actor=caster, target, expires_at_tick) — same arm shape as Stun.

    // Root = 8 → 43
    s.push_str(&format!("{i4}}} else if (kind == 8u) {{\n"));
    s.push_str(&format!("{i8}// Root = 8 → EventKindId::EffectRootApplied = 43\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); expires_at_tick = tick + duration\n"));
    s.push_str(&format!("{i8}let expires_at_tick: u32 = tick + payload_a;\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectRootApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(root_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {root_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 5));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Silence = 9 → 44
    s.push_str(&format!("{i4}}} else if (kind == 9u) {{\n"));
    s.push_str(&format!("{i8}// Silence = 9 → EventKindId::EffectSilenceApplied = 44\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); expires_at_tick = tick + duration\n"));
    s.push_str(&format!("{i8}let expires_at_tick: u32 = tick + payload_a;\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectSilenceApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(silence_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {silence_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 6));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Fear = 10 → 45
    s.push_str(&format!("{i4}}} else if (kind == 10u) {{\n"));
    s.push_str(&format!("{i8}// Fear = 10 → EventKindId::EffectFearApplied = 45\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); expires_at_tick = tick + duration\n"));
    s.push_str(&format!("{i8}let expires_at_tick: u32 = tick + payload_a;\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectFearApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(fear_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {fear_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 7));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Taunt = 11 → 46
    s.push_str(&format!("{i4}}} else if (kind == 11u) {{\n"));
    s.push_str(&format!("{i8}// Taunt = 11 → EventKindId::EffectTauntApplied = 46\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); expires_at_tick = tick + duration\n"));
    s.push_str(&format!("{i8}let expires_at_tick: u32 = tick + payload_a;\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectTauntApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(taunt_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {taunt_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 8));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Wave 2 piece 2 — movement EffectOps (Dash/Blink/Knockback/Pull).
    // Dash and Blink are caster-self motion: 2-payload-word chronicle
    // record (actor + distance, no target slot in the engine event).
    // The dispatcher still writes `target_slot` into ring offset 3 to
    // keep the 10-word stride consistent across all chronicle records
    // — the engine event struct ignores that slot and the cascade
    // decode in `event_to_fields` reads only `actor` + `distance`.
    // Knockback and Pull are forced motion on a target: 3-payload-word
    // chronicle record (actor + target + distance) — same shape family
    // as Damage/Heal/Shield (also bitcast<f32> at ring offset 4).

    // Dash = 12 → 47
    s.push_str(&format!("{i4}}} else if (kind == 12u) {{\n"));
    s.push_str(&format!("{i8}// Dash = 12 → EventKindId::EffectDashApplied = 47\n"));
    s.push_str(&format!("{i8}// payload_a = distance (f32 via bitcast); caster-self motion (no target field on engine event)\n"));
    s.push_str(&format!("{i8}let distance: f32 = bitcast<f32>(payload_a);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectDashApplied (caster_slot + distance)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(dash_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {dash_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], bitcast<u32>(distance));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 9));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Blink = 13 → 48
    s.push_str(&format!("{i4}}} else if (kind == 13u) {{\n"));
    s.push_str(&format!("{i8}// Blink = 13 → EventKindId::EffectBlinkApplied = 48\n"));
    s.push_str(&format!("{i8}// payload_a = distance (f32 via bitcast); caster-self motion (no target field on engine event)\n"));
    s.push_str(&format!("{i8}let distance: f32 = bitcast<f32>(payload_a);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectBlinkApplied (caster_slot + distance)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(blink_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {blink_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], bitcast<u32>(distance));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 10));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Knockback = 14 → 49
    s.push_str(&format!("{i4}}} else if (kind == 14u) {{\n"));
    s.push_str(&format!("{i8}// Knockback = 14 → EventKindId::EffectKnockbackApplied = 49\n"));
    s.push_str(&format!("{i8}// payload_a = distance (f32 via bitcast); forced motion on target\n"));
    s.push_str(&format!("{i8}let distance: f32 = bitcast<f32>(payload_a);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectKnockbackApplied (caster_slot + target_slot + distance)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(knockback_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {knockback_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(distance));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 11));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Pull = 15 → 50
    s.push_str(&format!("{i4}}} else if (kind == 15u) {{\n"));
    s.push_str(&format!("{i8}// Pull = 15 → EventKindId::EffectPullApplied = 50\n"));
    s.push_str(&format!("{i8}// payload_a = distance (f32 via bitcast); forced motion on target\n"));
    s.push_str(&format!("{i8}let distance: f32 = bitcast<f32>(payload_a);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectPullApplied (caster_slot + target_slot + distance)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(pull_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {pull_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(distance));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 12));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Extended-corpus statuses (Stealth/Charm/Grounded/Suppress).
    // Stealth is caster-self stealth: 2-payload-word chronicle record
    // (actor + duration_ticks at ring slot 3) — same family as
    // Dash/Blink (caster-self motion). The dispatcher writes
    // `target_slot` is NOT consulted here — the engine event has no
    // target field, so we mirror Dash's slot layout: payload_a (raw u32
    // duration) lands at slot 3.
    // Charm/Grounded/Suppress are target-cast: 3-payload-word chronicle
    // record (actor + target + duration_ticks at ring slot 4) — same
    // family as Knockback/Pull (forced-motion-on-target shape). Distinct
    // from Stun/Root/Silence/Fear/Taunt: those fold the deadline
    // (`expires_at_tick = tick + duration_ticks`); we store the raw
    // duration here, consistent with the multi-tick effect family
    // (DoT/HoT/TimedShield, kinds 51..53), so a future consumer rule
    // can compute its own per-tick re-emission window.

    // Stealth = 27 → 54
    s.push_str(&format!("{i4}}} else if (kind == 27u) {{\n"));
    s.push_str(&format!("{i8}// Stealth = 27 → EventKindId::EffectStealthApplied = 54\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); caster-self stealth\n"));
    s.push_str(&format!("{i8}// (no target field on engine event — same shape as Dash/Blink)\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectStealthApplied (caster_slot + duration_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(stealth_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {stealth_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 13));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Charm = 28 → 55
    s.push_str(&format!("{i4}}} else if (kind == 28u) {{\n"));
    s.push_str(&format!("{i8}// Charm = 28 → EventKindId::EffectCharmApplied = 55\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); target-cast charm\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectCharmApplied (caster_slot + target_slot + duration_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(charm_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {charm_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 14));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Grounded = 29 → 56
    s.push_str(&format!("{i4}}} else if (kind == 29u) {{\n"));
    s.push_str(&format!("{i8}// Grounded = 29 → EventKindId::EffectGroundedApplied = 56\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); target-cast grounded\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectGroundedApplied (caster_slot + target_slot + duration_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(grounded_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {grounded_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 15));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Suppress = 30 → 57
    s.push_str(&format!("{i4}}} else if (kind == 30u) {{\n"));
    s.push_str(&format!("{i8}// Suppress = 30 → EventKindId::EffectSuppressApplied = 57\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); target-cast suppress\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectSuppressApplied (caster_slot + target_slot + duration_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(suppress_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {suppress_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 16));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // TransferGold = 5 → 31
    s.push_str(&format!("{i4}}} else if (kind == 5u) {{\n"));
    s.push_str(&format!("{i8}// TransferGold = 5 → EventKindId::EffectGoldTransfer = 31\n"));
    s.push_str(&format!("{i8}// payload_a = amount (i32 sign-widened to u32)\n"));
    s.push_str(&format!("{i8}// Engine event carries amount as i64 — GPU writes the low 32\n"));
    s.push_str(&format!("{i8}// bits + zero-extends. Cascade chronicle decode reads the u32\n"));
    s.push_str(&format!("{i8}// then sign-extends back to i64 (matches the EffectOp's i32\n"));
    s.push_str(&format!("{i8}// source-of-truth — i64 is host-side widening only).\n"));
    s.push_str(&format!("{i8}let amount_i32: i32 = bitcast<i32>(payload_a);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectGoldTransfer (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(transfer_gold_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {transfer_gold_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount_i32));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 17));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // ModifyStanding = 6 → 32
    s.push_str(&format!("{i4}}} else if (kind == 6u) {{\n"));
    s.push_str(&format!("{i8}// ModifyStanding = 6 → EventKindId::EffectStandingDelta = 32\n"));
    s.push_str(&format!("{i8}// payload_a = delta (i16 sign-widened to u32)\n"));
    s.push_str(&format!("{i8}let delta_i32: i32 = bitcast<i32>(payload_a);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectStandingDelta (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(modify_standing_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {modify_standing_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(delta_i32));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 18));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Execute = 16 → 42
    s.push_str(&format!("{i4}}} else if (kind == 16u) {{\n"));
    s.push_str(&format!("{i8}// Execute = 16 → EventKindId::EffectExecuteApplied = 42\n"));
    s.push_str(&format!("{i8}// payload_a = hp_threshold (f32 via bitcast). The when-\n"));
    s.push_str(&format!("{i8}// condition `target.hp < hp_threshold` is NOT evaluated\n"));
    s.push_str(&format!("{i8}// here — that's the .ability's `when_per_effect[i]` and\n"));
    s.push_str(&format!("{i8}// stays unconsulted by apply_program today. Duel_abilities\n"));
    s.push_str(&format!("{i8}// Reap's outer verb gate already enforces the threshold.\n"));
    s.push_str(&format!("{i8}// Same shape family as EffectDamageApplied — 3 payload\n"));
    s.push_str(&format!("{i8}// words (actor, target, hp_threshold).\n"));
    s.push_str(&format!("{i8}let hp_threshold: f32 = bitcast<f32>(payload_a);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectExecuteApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(execute_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {execute_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(hp_threshold));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 19));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // SelfDamage = 17 → 39
    s.push_str(&format!("{i4}}} else if (kind == 17u) {{\n"));
    s.push_str(&format!("{i8}// SelfDamage = 17 → EventKindId::EffectSelfDamageApplied = 39\n"));
    s.push_str(&format!("{i8}// payload_a = amount (f32 via bitcast). Self-damage targets\n"));
    s.push_str(&format!("{i8}// the caster — the chronicle writes caster_slot into BOTH actor\n"));
    s.push_str(&format!("{i8}// (slot 2) and target (slot 3) so the re-emit physics rule's\n"));
    s.push_str(&format!("{i8}// pattern can ferry both ids verbatim into Damaged.\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectSelfDamageApplied (caster_slot for both actor + target)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(self_damage_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {self_damage_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 20));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // LifeSteal = 18 → 40
    s.push_str(&format!("{i4}}} else if (kind == 18u) {{\n"));
    s.push_str(&format!("{i8}// LifeSteal = 18 → EventKindId::EffectLifeStealApplied = 40\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); expires = tick + duration\n"));
    s.push_str(&format!("{i8}// payload_b sign-widened i16 → fraction_q8 (i32 via bitcast)\n"));
    s.push_str(&format!("{i8}// Same shape as Slow (kind == 4u): 4 payload words —\n"));
    s.push_str(&format!("{i8}// actor, target, expires_at_tick, fraction_q8.\n"));
    s.push_str(&format!("{i8}let expires_at_tick: u32 = tick + payload_a;\n"));
    s.push_str(&format!("{i8}let fraction_q8: i32 = bitcast<i32>(payload_b);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectLifeStealApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(life_steal_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {life_steal_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], bitcast<u32>(fraction_q8));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 21));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // DamageModify = 19 → 41
    s.push_str(&format!("{i4}}} else if (kind == 19u) {{\n"));
    s.push_str(&format!("{i8}// DamageModify = 19 → EventKindId::EffectDamageModifyApplied = 41\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); expires = tick + duration\n"));
    s.push_str(&format!("{i8}// payload_b sign-widened i16 → multiplier_q8 (i32 via bitcast)\n"));
    s.push_str(&format!("{i8}// Same shape as Slow (kind == 4u) / LifeSteal (kind == 18u):\n"));
    s.push_str(&format!("{i8}// 4 payload words — actor, target, expires_at_tick, multiplier_q8.\n"));
    s.push_str(&format!("{i8}let expires_at_tick: u32 = tick + payload_a;\n"));
    s.push_str(&format!("{i8}let multiplier_q8: i32 = bitcast<i32>(payload_b);\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectDamageModifyApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(damage_modify_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {damage_modify_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], bitcast<u32>(multiplier_q8));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 22));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Wave 1.5+ — multi-tick effects (DamageOverTime/HealOverTime/
    // TimedShield). All three share a 5-word chronicle record:
    //   slot 0 = kind tag (51 / 52 / 53)
    //   slot 1 = tick
    //   slot 2 = caster_slot
    //   slot 3 = target_slot
    //   slot 4 = bitcast<u32>(amount)            // amount already includes scale_bonus
    //   slot 5 = duration_ticks (raw u32)
    // The cast records the magnitude + window once; a future consumer
    // rule will re-emit per-tick damage/heal events. Wave 1.5#4 GPU
    // wire-up already folded scale_bonus into the amount above this
    // chain (the existing `bitcast<f32>(payload_a) + scale_bonus_var`
    // is correct); we just bitcast the result back to u32 for the
    // ring storage.
    // Buff(23) tickAmount uses scale_bonus, but period_ticks does not —
    // not relevant here because Buff packs `magnitude_q8` not `amount`.

    // DamageOverTime = 20 → 51
    s.push_str(&format!("{i4}}} else if (kind == 20u) {{\n"));
    s.push_str(&format!("{i8}// DamageOverTime = 20 → EventKindId::EffectDamageOverTimeApplied = 51\n"));
    s.push_str(&format!("{i8}// payload_a = amount-per-tick (f32, scale_bonus folded in),\n"));
    s.push_str(&format!("{i8}// payload_b = duration_ticks (u32)\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectDamageOverTimeApplied (caster_slot + target_slot + amount + duration)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(damage_over_time_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {damage_over_time_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 23));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // HealOverTime = 21 → 52
    s.push_str(&format!("{i4}}} else if (kind == 21u) {{\n"));
    s.push_str(&format!("{i8}// HealOverTime = 21 → EventKindId::EffectHealOverTimeApplied = 52\n"));
    s.push_str(&format!("{i8}// payload_a = amount-per-tick (f32, scale_bonus folded in),\n"));
    s.push_str(&format!("{i8}// payload_b = duration_ticks (u32)\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectHealOverTimeApplied (caster_slot + target_slot + amount + duration)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(heal_over_time_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {heal_over_time_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 24));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // TimedShield = 22 → 53
    s.push_str(&format!("{i4}}} else if (kind == 22u) {{\n"));
    s.push_str(&format!("{i8}// TimedShield = 22 → EventKindId::EffectTimedShieldApplied = 53\n"));
    s.push_str(&format!("{i8}// payload_a = amount (f32, scale_bonus folded in),\n"));
    s.push_str(&format!("{i8}// payload_b = duration_ticks (u32)\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectTimedShieldApplied (caster_slot + target_slot + amount + duration)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(timed_shield_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {timed_shield_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 25));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Slice γ tail — Buff/Harvest/PlaceVoxel/Reflect. Four distinct
    // shapes, all storing raw u32 payload words for consumer-side
    // decode (no decomposition at dispatch time):
    //   - Buff (kind 23 → 58, target-cast): 5-payload-word record
    //     (caster + target + raw payload_a + raw payload_b). payload_a
    //     packs `stat_ordinal` (u8 low byte) | `magnitude_q8` (i16 bits
    //     8..); payload_b is duration_ticks. Consumers sign-extend the
    //     magnitude on read.
    //   - Harvest (kind 25 → 59, caster-self): 4-payload-word record
    //     (caster + kind_hash + amount). No target field on engine event.
    //   - PlaceVoxel (kind 26 → 60, caster-self): 3-payload-word record
    //     (caster + kind_hash). Position is implicit from the cast's
    //     target world position (not stored in the chronicle record).
    //   - Reflect (kind 31 → 61, target-cast): 5-payload-word record
    //     (caster + target + raw payload_a + raw payload_b). payload_a
    //     is duration_ticks; payload_b's low 16 bits are fraction_q8
    //     (i16). Consumers sign-extend the fraction on read.
    //
    // Same convention as the multi-tick effect family (DoT/HoT/
    // TimedShield, kinds 51..53): chronicle stores raw `payload_a` /
    // `payload_b` u32 words; downstream consumer rules / cascade
    // decoders compute typed values from the bits.

    // Buff = 23 → 58
    s.push_str(&format!("{i4}}} else if (kind == 23u) {{\n"));
    s.push_str(&format!("{i8}// Buff = 23 → EventKindId::EffectBuffApplied = 58\n"));
    s.push_str(&format!("{i8}// payload_a packs (stat_ordinal in low byte | magnitude_q8 in bits 8..);\n"));
    s.push_str(&format!("{i8}// payload_b = duration_ticks. magnitude_q8 is i16 sign-extended.\n"));
    s.push_str(&format!("{i8}// Chronicle stores raw payload_a / payload_b — consumers decode.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectBuffApplied (caster_slot + target_slot + payload_a + payload_b)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(buff_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {buff_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 26));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Summon = 24 → 62 (slice γ closer)
    s.push_str(&format!("{i4}}} else if (kind == 24u) {{\n"));
    s.push_str(&format!("{i8}// Summon = 24 → EventKindId::EffectSummonApplied = 62\n"));
    s.push_str(&format!("{i8}// payload_a = template_hash (u32),\n"));
    s.push_str(&format!("{i8}// payload_b = (count in high byte | lifetime_ticks in low 24 bits)\n"));
    s.push_str(&format!("{i8}// — same packing as `pack_effect`'s Summon arm in\n"));
    s.push_str(&format!("{i8}// `crates/engine/src/ability/packed.rs`. Caster-self — no\n"));
    s.push_str(&format!("{i8}// target field on engine event. The dispatcher writes count\n"));
    s.push_str(&format!("{i8}// and lifetime into distinct ring slots so consumers don't\n"));
    s.push_str(&format!("{i8}// have to redo the bit-unpack on read; downstream N-entity\n"));
    s.push_str(&format!("{i8}// spawning is a separate consumer concern.\n"));
    s.push_str(&format!("{i8}let summon_count: u32 = (payload_b >> 24u) & 0xFFu;\n"));
    s.push_str(&format!("{i8}let summon_lifetime: u32 = payload_b & 0x00FFFFFFu;\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectSummonApplied (caster_slot + template_hash + count + lifetime_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(summon_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {summon_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], summon_count);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], summon_lifetime);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 27));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Harvest = 25 → 59
    s.push_str(&format!("{i4}}} else if (kind == 25u) {{\n"));
    s.push_str(&format!("{i8}// Harvest = 25 → EventKindId::EffectHarvestApplied = 59\n"));
    s.push_str(&format!("{i8}// payload_a = kind_hash (u32 FxHash of resource ident),\n"));
    s.push_str(&format!("{i8}// payload_b = amount (u32, widened from u16 EffectOp side).\n"));
    s.push_str(&format!("{i8}// Caster-self — no target field on engine event.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectHarvestApplied (caster_slot + kind_hash + amount)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(harvest_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {harvest_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 28));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // PlaceVoxel = 26 → 60
    s.push_str(&format!("{i4}}} else if (kind == 26u) {{\n"));
    s.push_str(&format!("{i8}// PlaceVoxel = 26 → EventKindId::EffectPlaceVoxelApplied = 60\n"));
    s.push_str(&format!("{i8}// payload_a = kind_hash (u32 FxHash of voxel kind ident).\n"));
    s.push_str(&format!("{i8}// Position is implicit from cast's target world position (not in record).\n"));
    s.push_str(&format!("{i8}// Caster-self — no target field on engine event.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectPlaceVoxelApplied (caster_slot + kind_hash)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(place_voxel_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {place_voxel_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 29));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Reflect = 31 → 61
    s.push_str(&format!("{i4}}} else if (kind == 31u) {{\n"));
    s.push_str(&format!("{i8}// Reflect = 31 → EventKindId::EffectReflectApplied = 61\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32),\n"));
    s.push_str(&format!("{i8}// payload_b's low 16 bits = fraction_q8 (i16, sign-extended on read).\n"));
    s.push_str(&format!("{i8}// Chronicle stores raw payload_b — consumers sign-extend.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectReflectApplied (caster_slot + target_slot + duration + fraction_q8)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(reflect_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {reflect_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 30));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // PlantBelief = 32 → 63 (Wave 3 ToM Phase 1 bit-flag primitive)
    s.push_str(&format!("{i4}}} else if (kind == 32u) {{\n"));
    s.push_str(&format!("{i8}// PlantBelief = 32 → EventKindId::EffectPlantBeliefApplied = 63\n"));
    s.push_str(&format!("{i8}// payload_a = subject_idx (u32 — agent slot the belief is ABOUT),\n"));
    s.push_str(&format!("{i8}// payload_b = fact_bit_mask (= 1u << fact_bit, pre-shifted at\n"));
    s.push_str(&format!("{i8}// pack time so the downstream view consumer's `self |= b` body\n"));
    s.push_str(&format!("{i8}// doesn't re-shift). The dispatcher writes only the chronicle\n"));
    s.push_str(&format!("{i8}// record here; the actual atomicOr write into the pair_map\n"));
    s.push_str(&format!("{i8}// cell happens in a downstream view fold consumer (existing\n"));
    s.push_str(&format!("{i8}// `tom_probe.sim::beliefs` shape: view ... -> u32 with\n"));
    s.push_str(&format!("{i8}// `on EffectPlantBeliefApplied {{ ... }} {{ self |= b }}`. This keeps\n"));
    s.push_str(&format!("{i8}// the apply_ability dispatcher's BGL stable (no new pair_map\n"));
    s.push_str(&format!("{i8}// binding) and routes the bit-fold through the standard\n"));
    s.push_str(&format!("{i8}// view-fold pipeline.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectPlantBeliefApplied (caster_slot + target_slot + subject_idx + fact_bit_mask)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(plant_belief_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {plant_belief_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 31));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Observe = 33 → 64 (Wave 3 ToM Phase 3 self-observe-target verb)
    s.push_str(&format!("{i4}}} else if (kind == 33u) {{\n"));
    s.push_str(&format!("{i8}// Observe = 33 → EventKindId::EffectObserveApplied = 64\n"));
    s.push_str(&format!("{i8}// payload_a = target_observer (u8 widened to u32 — future-extension\n"));
    s.push_str(&format!("{i8}// hook for non-self observe shapes; today only `0` (self) is wired).\n"));
    s.push_str(&format!("{i8}// payload_b = 0 (unused — the consumer reads target's CURRENT\n"));
    s.push_str(&format!("{i8}// pos / creature_type from the agent SoA at consume tick rather\n"));
    s.push_str(&format!("{i8}// than carrying them on the chronicle record). The actual\n"));
    s.push_str(&format!("{i8}// writeback into the BeliefState SoA's 6 columns at\n"));
    s.push_str(&format!("{i8}// `[caster * agent_cap + target]` indexing happens in a\n"));
    s.push_str(&format!("{i8}// downstream runtime consumer (`tom_probe_runtime` Phase 3).\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectObserveApplied (caster_slot + target_slot + target_observer)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(observe_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {observe_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 32));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Scry = 34 → 65 (Wave 3 ToM Phase 3.5 cross-observer access)
    s.push_str(&format!("{i4}}} else if (kind == 34u) {{\n"));
    s.push_str(&format!("{i8}// Scry = 34 → EventKindId::EffectScryApplied = 65\n"));
    s.push_str(&format!("{i8}// payload_a = target_observer (u8 widened to u32 — agent slot whose\n"));
    s.push_str(&format!("{i8}// beliefs caster reads). payload_b = subject_idx (u32 — agent slot\n"));
    s.push_str(&format!("{i8}// the belief is ABOUT). The downstream 6-column copy from\n"));
    s.push_str(&format!("{i8}// `[target_observer * N + subject_idx]` to\n"));
    s.push_str(&format!("{i8}// `[caster * N + subject_idx]` lives in a runtime consumer\n"));
    s.push_str(&format!("{i8}// (`tom_probe_runtime` Phase 3.5). The dispatcher only writes the\n"));
    s.push_str(&format!("{i8}// chronicle record; this keeps the apply_ability dispatcher's BGL\n"));
    s.push_str(&format!("{i8}// stable (no new SoA bindings).\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectScryApplied (caster_slot + target_slot + target_observer + subject_idx)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(scry_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {scry_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 33));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Reveal = 35 → 66 (Wave 3 ToM Phase 3.5 one-to-many propagation)
    s.push_str(&format!("{i4}}} else if (kind == 35u) {{\n"));
    s.push_str(&format!("{i8}// Reveal = 35 → EventKindId::EffectRevealApplied = 66\n"));
    s.push_str(&format!("{i8}// payload_a = subject_idx (u32 — agent slot the broadcast is\n"));
    s.push_str(&format!("{i8}// ABOUT). payload_b = 0 (unused — the fan-out target set is `all\n"));
    s.push_str(&format!("{i8}// observers` at consume time). The downstream fan-out (caster's\n"));
    s.push_str(&format!("{i8}// beliefs about subject → every observer's beliefs about subject)\n"));
    s.push_str(&format!("{i8}// lives in a runtime consumer (`tom_probe_runtime` Phase 3.5).\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectRevealApplied (caster_slot + target_slot + subject_idx)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(reveal_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {reveal_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 34));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Disguise = 36 → 67 (Wave 3 ToM Phase 4 deception verb)
    s.push_str(&format!("{i4}}} else if (kind == 36u) {{\n"));
    s.push_str(&format!("{i8}// Disguise = 36 → EventKindId::EffectDisguiseApplied = 67\n"));
    s.push_str(&format!("{i8}// payload_a = (duration_ticks << 8) | fake_type (low byte = u8\n"));
    s.push_str(&format!("{i8}// fake_type, high 24 bits = duration_ticks). The consumer reads\n"));
    s.push_str(&format!("{i8}// `payload_a & 0xFFu` for fake_type and `payload_a >> 8u` for the\n"));
    s.push_str(&format!("{i8}// duration. payload_b = 0 (unused). The downstream consumer writes\n"));
    s.push_str(&format!("{i8}// per-agent `disguise_expires_at_tick` and `disguise_fake_type`\n"));
    s.push_str(&format!("{i8}// SoA columns (one cell per agent). Subsequent observe calls read\n"));
    s.push_str(&format!("{i8}// these columns to substitute fake_type for the true creature_type.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectDisguiseApplied (caster_slot + target_slot + packed_payload_a)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(disguise_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {disguise_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 35));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Decoy = 37 → 68 (Wave 3 ToM Phase 4 deception verb)
    s.push_str(&format!("{i4}}} else if (kind == 37u) {{\n"));
    s.push_str(&format!("{i8}// Decoy = 37 → EventKindId::EffectDecoyApplied = 68\n"));
    s.push_str(&format!("{i8}// payload_a = subject_idx (u32 — the agent slot the belief is\n"));
    s.push_str(&format!("{i8}// ABOUT — distinct from `target` which is the OBSERVER whose row\n"));
    s.push_str(&format!("{i8}// caster writes). payload_b = packed (x_q8 lo, y_q8, z_q8, fake_type\n"));
    s.push_str(&format!("{i8}// hi) quartet — pre-packed at pack time so the consumer's bit\n"));
    s.push_str(&format!("{i8}// extracts (`payload_b & 0xFFu`, `(payload_b >> 8u) & 0xFFu`, …)\n"));
    s.push_str(&format!("{i8}// recover the per-byte values without re-packing.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectDecoyApplied (caster_slot + target_slot + subject_idx + fake_pos)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(decoy_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {decoy_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 36));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // EraseBelief = 38 → 69 (Wave 3 ToM Phase 4 deception verb)
    s.push_str(&format!("{i4}}} else if (kind == 38u) {{\n"));
    s.push_str(&format!("{i8}// EraseBelief = 38 → EventKindId::EffectEraseBeliefApplied = 69\n"));
    s.push_str(&format!("{i8}// payload_a = subject_idx (u32 — the agent slot the belief is\n"));
    s.push_str(&format!("{i8}// ABOUT). payload_b's low byte = fields bitset (bit 0 = pos, 1 =\n"));
    s.push_str(&format!("{i8}// type, 2 = tick, 3 = confidence, 4 = suspicion, 5 = flags). The\n"));
    s.push_str(&format!("{i8}// consumer reads the bitset and clears matching cells in target's\n"));
    s.push_str(&format!("{i8}// row about subject_idx (one if-block per bit).\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectEraseBeliefApplied (caster_slot + target_slot + subject_idx + fields)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(erase_belief_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {erase_belief_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 37));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // TravelTo = 39 → 70 (Lift A multi-tick travel)
    s.push_str(&format!("{i4}}} else if (kind == 39u) {{\n"));
    s.push_str(&format!("{i8}// TravelTo = 39 → EventKindId::EffectTravelToApplied = 70\n"));
    s.push_str(&format!("{i8}// payload_a packs (dest_y_q8 << 16) | (dest_x_q8 & 0xFFFF) —\n"));
    s.push_str(&format!("{i8}// the consumer sign-extends each half via:\n"));
    s.push_str(&format!("{i8}//   dest_x = f32(bitcast<i32>(payload_a << 16u) >> 16u) / 256.0;\n"));
    s.push_str(&format!("{i8}//   dest_y = f32(bitcast<i32>(payload_a) >> 16u)        / 256.0;\n"));
    s.push_str(&format!("{i8}// payload_b = eta_ticks (u32). The downstream consumer rule sets\n"));
    s.push_str(&format!("{i8}// `busy_until_tick = world.tick + eta_ticks` and populates the\n"));
    s.push_str(&format!("{i8}// per-agent `travel_dest_{{x,y,z}}` SoA cells; a per-tick travel\n"));
    s.push_str(&format!("{i8}// interpolation kernel walks `pos` toward the destination over\n"));
    s.push_str(&format!("{i8}// `eta_ticks` ticks. Travel is self-cast: `target_slot ==\n"));
    s.push_str(&format!("{i8}// caster_slot` by convention (the dispatcher emits the same slot\n"));
    s.push_str(&format!("{i8}// in both fields for shape-uniformity).\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectTravelToApplied (caster_slot + caster_slot + packed_dest + eta_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(travel_to_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {travel_to_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 38));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Recipe = 40 → 71 (Lift B production verb)
    s.push_str(&format!("{i4}}} else if (kind == 40u) {{\n"));
    s.push_str(&format!("{i8}// Recipe = 40 → EventKindId::EffectRecipeApplied = 71\n"));
    s.push_str(&format!("{i8}// payload_a low 16 bits = recipe_id (registry index); next 8 bits =\n"));
    s.push_str(&format!("{i8}// target_tool slot (0xFF sentinel = no tool target). payload_b = 0.\n"));
    s.push_str(&format!("{i8}// The consumer unpacks via:\n"));
    s.push_str(&format!("{i8}//   recipe_id   = payload_a & 0xFFFFu;\n"));
    s.push_str(&format!("{i8}//   target_tool = (payload_a >> 16u) & 0xFFu;\n"));
    s.push_str(&format!("{i8}// then reads `RecipeRegistry[recipe_id]`, validates the caster's\n"));
    s.push_str(&format!("{i8}// inventory + (optionally) the `target_tool` slot, and emits the\n"));
    s.push_str(&format!("{i8}// ingredient/output inventory deltas. Self-cast: recipes act on\n"));
    s.push_str(&format!("{i8}// the caster's inventory (`target_slot == caster_slot`).\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectRecipeApplied (caster_slot + caster_slot + packed_recipe + 0)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(recipe_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {recipe_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 39));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // WearTool = 41 → 72 (Lift B capital-goods wear)
    s.push_str(&format!("{i4}}} else if (kind == 41u) {{\n"));
    s.push_str(&format!("{i8}// WearTool = 41 → EventKindId::EffectWearToolApplied = 72\n"));
    s.push_str(&format!("{i8}// payload_a low 8 bits = tool_kind ordinal; next 16 bits = amount\n"));
    s.push_str(&format!("{i8}// (q8 fraction-of-durability). payload_b = 0. The consumer unpacks\n"));
    s.push_str(&format!("{i8}// via:\n"));
    s.push_str(&format!("{i8}//   tool_kind = payload_a & 0xFFu;\n"));
    s.push_str(&format!("{i8}//   amount    = (payload_a >> 8u) & 0xFFFFu;\n"));
    s.push_str(&format!("{i8}// then looks up the caster's owned tool of `tool_kind` and bumps\n"));
    s.push_str(&format!("{i8}// its wear cell by `amount`; at `wear >= durability` flips the\n"));
    s.push_str(&format!("{i8}// broken bit. Self-cast: wear acts on the caster's owned tool\n"));
    s.push_str(&format!("{i8}// (`target_slot == caster_slot`).\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectWearToolApplied (caster_slot + caster_slot + packed_wear + 0)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(wear_tool_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {wear_tool_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 40));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Propose = 42 → 73 (Lift C bilateral consent)
    s.push_str(&format!("{i4}}} else if (kind == 42u) {{\n"));
    s.push_str(&format!("{i8}// Propose = 42 → EventKindId::EffectProposeApplied = 73\n"));
    s.push_str(&format!("{i8}// payload_a low 8 bits = contract_kind ordinal. payload_b =\n"));
    s.push_str(&format!("{i8}// expires_at_tick (0 sentinel = no expiry). The consumer reads\n"));
    s.push_str(&format!("{i8}// the proposal pair (caster_slot → target_slot) and registers it\n"));
    s.push_str(&format!("{i8}// in the per-fixture ContractRegistry, to be resolved when the\n"));
    s.push_str(&format!("{i8}// target later fires the companion accept / decline verb.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectProposeApplied (caster_slot + target_slot + payload_a + payload_b)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(propose_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {propose_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 41));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Announce = 43 → 74 (Lift C observer fan-out)
    s.push_str(&format!("{i4}}} else if (kind == 43u) {{\n"));
    s.push_str(&format!("{i8}// Announce = 43 → EventKindId::EffectAnnounceApplied = 74\n"));
    s.push_str(&format!("{i8}// payload_a low 8 bits = announcement_kind; next 16 bits = radius_q8\n"));
    s.push_str(&format!("{i8}// (q8 fraction-of-cell — 256 = 1.0 cell). payload_b = 0. The\n"));
    s.push_str(&format!("{i8}// consumer walks the spatial-hash within `radius_q8 / 256` cells\n"));
    s.push_str(&format!("{i8}// of the caster and emits per-observer perception events. Self-\n"));
    s.push_str(&format!("{i8}// origin: announcements radiate from the caster's cell\n"));
    s.push_str(&format!("{i8}// (`target_slot == caster_slot`).\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectAnnounceApplied (caster_slot + caster_slot + payload_a + 0)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(announce_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {announce_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 42));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // GainSkill = 44 → 75 (Lift D self-cast skill growth)
    s.push_str(&format!("{i4}}} else if (kind == 44u) {{\n"));
    s.push_str(&format!("{i8}// GainSkill = 44 → EventKindId::EffectGainSkillApplied = 75\n"));
    s.push_str(&format!("{i8}// payload_a low 8 bits = skill_id; next 16 bits = amount_q8\n"));
    s.push_str(&format!("{i8}// (q8 fraction-of-mastery — 256 = full mastery). payload_b = 0.\n"));
    s.push_str(&format!("{i8}// Self-cast: target = caster (skill grows on the caster's per-\n"));
    s.push_str(&format!("{i8}// agent SoA cell). The consumer adds amount_q8 / 256.0 to the\n"));
    s.push_str(&format!("{i8}// per-agent per-skill cell, clamped to [0.0, 1.0].\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectGainSkillApplied (caster_slot + caster_slot + payload_a + 0)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(gain_skill_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {gain_skill_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 43));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // CreateObligation = 45 → 76 (Lift D obligation registry write)
    s.push_str(&format!("{i4}}} else if (kind == 45u) {{\n"));
    s.push_str(&format!("{i8}// CreateObligation = 45 → EventKindId::EffectCreateObligationApplied = 76\n"));
    s.push_str(&format!("{i8}// payload_a low 16 bits = obligation_id (registry slot); next 8\n"));
    s.push_str(&format!("{i8}// bits = kind (Debt=0, Future=1, Insurance=2, Retainer=3,\n"));
    s.push_str(&format!("{i8}// Service=4). payload_b = 0. caster = creditor / claimant;\n"));
    s.push_str(&format!("{i8}// target = debtor / promisor. The consumer registers the\n"));
    s.push_str(&format!("{i8}// obligation in the AggregatePool and updates per-agent debtor /\n"));
    s.push_str(&format!("{i8}// creditor indices.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectCreateObligationApplied (caster_slot + target_slot + payload_a + 0)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(create_obligation_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {create_obligation_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 44));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // CastBegin = 46 → 77 (Plan G generic deferred cast)
    s.push_str(&format!("{i4}}} else if (kind == 46u) {{\n"));
    s.push_str(&format!("{i8}// CastBegin = 46 → EventKindId::EffectCastBeginApplied = 77\n"));
    s.push_str(&format!("{i8}// payload_a low 16 bits = ability_id; high 16 bits = duration_ticks.\n"));
    s.push_str(&format!("{i8}// payload_b = the q8 target position packed as (x_q8 | y_q8 << 16),\n"));
    s.push_str(&format!("{i8}// or 0 when the EffectOp had no per-op position override.\n"));
    s.push_str(&format!("{i8}// chronicle target_slot is the runtime resolved target — the\n"));
    s.push_str(&format!("{i8}// busy-resolution kernel (G2.4) keys the deferred resolve off it.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectCastBeginApplied (caster_slot + target_slot + payload_a + payload_b)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&hist_bump(cast_begin_event_id));
    s.push_str(&format!("{i12}if (_slot < 1048576u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 0u], {cast_begin_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 6u], ability_id__u32);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 11u + 10u], ({kernel_id}u << 24u) | (caster_slot << 4u) | {emit_idx}u);\n",
        kernel_id = producer_kernel_id,
        emit_idx = first_emit_idx + 45));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    s.push_str(&format!("{i4}}}\n"));

    s
}

/// Lower a [`CgStmt::Match`] as a scrutinee-bound `if`-chain. WGSL's
/// `switch` would be a future-tense option; today the chain is the
/// honest placeholder.
///
/// The scrutinee is bound to a local variable `_scrut_<N>` *before* the
/// chain so non-identifier scrutinees (e.g. a `Binary { ... }` node
/// lowered to `(x + 1)`) produce valid WGSL — `((x + 1)_tag)` is
/// nonsense, `_scrut_<N>.tag` is fine. `<N>` is the scrutinee's
/// [`CgExprId`] (the only id this function has access to — `CgStmtId` /
/// `CgStmtListId` are not threaded through). Since each `Match`
/// statement has a distinct scrutinee expression node in the arena, the
/// id is unique-per-match-site within a program.
///
/// Arm-binding locals are still emitted as a comment for now, but the
/// comment references `_scrut_<N>.<field>` so a future Task 4.x can
/// flip the comment into a real `let local_<N>: <ty> = _scrut_<N>.<field>;`
/// without changing the surrounding shape.
fn lower_match_to_wgsl(
    scrutinee: CgExprId,
    arms: &[CgMatchArm],
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    let s = lower_cg_expr_to_wgsl(scrutinee, ctx)?;
    if arms.is_empty() {
        // Empty match body — emit a comment so the generated WGSL is
        // still syntactically inert. (Should not occur in well-formed
        // programs.)
        return Ok(format!("// match {} {{ /* no arms */ }}", s));
    }
    let scrut_name = format!("_scrut_{}", scrutinee.0);
    let mut out = format!("let {} = {};\n", scrut_name, s);
    for (i, arm) in arms.iter().enumerate() {
        let body = lower_cg_stmt_list_to_wgsl(arm.body, ctx)?;
        let bindings_comment = if arm.bindings.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = arm
                .bindings
                .iter()
                .map(|b: &MatchArmBinding| {
                    format!(
                        "{name}=local_{lid} from {scrut}.{name}",
                        name = b.field_name,
                        lid = b.local.0,
                        scrut = scrut_name,
                    )
                })
                .collect();
            format!(" /* bindings: {} */", pairs.join(", "))
        };
        if i == 0 {
            out.push_str(&format!(
                "if ({}.tag == VARIANT_{}u) {{{}\n{}\n}}",
                scrut_name,
                arm.variant.0,
                bindings_comment,
                indent_block(&body, 1)
            ));
        } else {
            out.push_str(&format!(
                " else if ({}.tag == VARIANT_{}u) {{{}\n{}\n}}",
                scrut_name,
                arm.variant.0,
                bindings_comment,
                indent_block(&body, 1)
            ));
        }
    }
    Ok(out)
}

/// Lower a [`crate::cg::CgStmtList`] as a sequence of statements,
/// joined with `\n`. Empty lists produce the empty string.
///
/// # Limitations
///
/// Same as [`lower_cg_stmt_to_wgsl`].
pub fn lower_cg_stmt_list_to_wgsl(
    list_id: CgStmtListId,
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    let arena_len = ctx.prog.stmt_lists.len() as u32;
    let list = <CgProgram as StmtListArena>::get(ctx.prog, list_id).ok_or(
        EmitError::StmtListIdOutOfRange {
            id: list_id,
            arena_len,
        },
    )?;
    // Reset per-list scratch so the view-fold target-local capture
    // from a previous stmt list (e.g. an earlier handler in the same
    // op) can't leak into this one. The per-stmt Let/Assign sequence
    // re-establishes the target locals for the current list.
    let saved_view_targets = ctx.view_target_locals.replace(Vec::new());

    // Snapshot the cross-agent target-let bound set so any new
    // bindings emitted inside this list (which live in WGSL block
    // scope) can't leak into the surrounding scope when the list
    // returns. Outer-scope bindings *do* remain visible to nested
    // emit (cloned-then-restored, not reset-then-restored) — this
    // matches WGSL's function-scope let visibility, where an
    // outer-block binding is in scope inside any nested block.
    let saved_bound_targets = ctx.bound_target_exprs.borrow().clone();

    // Fold-fusion pre-pass: collect every `ForEachNeighbor` in the
    // list whose `init` + `projection` are pure (no `ReadLocal`
    // dependencies on prior stmts). Pure folds can be hoisted to the
    // front of the list and emitted as one fused walk; the remaining
    // stmts (Let / Assign / etc.) follow in source order. The
    // accumulator locals every fold writes are still available for
    // the deferred stmts because hoisting only moves them
    // _earlier_ in execution. See `emit_fused_for_each_neighbor`'s
    // docstring for why this matters: a single walk replaces N
    // redundant 27-cell traversals + agent_pos lookups, the dominant
    // memory-bandwidth cost in boids-style bodies.
    //
    // A fold whose projection reads a `ReadLocal` cannot be safely
    // hoisted — the bound local lives on a `Let` stmt that comes
    // before the fold in source order; moving the fold up would
    // reference an undeclared `local_<N>`. Such folds stay in their
    // original position and emit as singletons.
    //
    // Folds with mixed `radius_cells` cannot share a single walk
    // (the loop bounds differ), so we partition by radius too.
    // Today every spatial fold uses `radius_cells = 1`, so the
    // partition is single-element in practice.
    // Partition key: `(radius_cells, lowered origin WGSL)`. Gap
    // dungeon_horde#1 widened the fused-emit precondition from "same
    // radius" to "same radius AND same origin expression" — the gate
    // prefix references one origin, so fusing folds with different
    // origins would silently use whichever origin landed first. The
    // lowered WGSL is the structural canonical form here: two distinct
    // `CgExprId`s that both render `agent_id` (e.g. two independent
    // `CgExpr::AgentSelfId` allocations from sibling fold calls) hash
    // to the same key and fuse correctly; an `AgentSelfId` vs. a
    // `ReadLocal` would hash differently and stay split.
    let mut hoistable: std::collections::BTreeMap<(u32, String), Vec<&CgStmt>> =
        std::collections::BTreeMap::new();
    let mut residual: Vec<usize> = Vec::with_capacity(list.stmts.len());
    for (idx, stmt_id) in list.stmts.iter().enumerate() {
        let stmt_node = <CgProgram as StmtArena>::get(ctx.prog, *stmt_id).ok_or(
            EmitError::StmtIdOutOfRange {
                id: *stmt_id,
                arena_len: ctx.prog.stmts.len() as u32,
            },
        )?;
        if let CgStmt::ForEachNeighbor {
            radius_cells,
            init,
            projection,
            origin,
            ..
        } = stmt_node
        {
            if expr_is_pure_for_hoisting(*init, ctx)
                && expr_is_pure_for_hoisting(*projection, ctx)
            {
                let origin_wgsl = lower_cg_expr_to_wgsl(*origin, ctx)?;
                hoistable.entry((*radius_cells, origin_wgsl)).or_default().push(stmt_node);
                continue;
            }
        }
        residual.push(idx);
    }

    let mut parts: Vec<String> = Vec::new();
    // Emit the fused walks first, partitioned by (radius, origin) for
    // deterministic output (BTreeMap iteration is sorted lexicographic
    // on the tuple — radius first, then origin WGSL string).
    for (_key, folds) in &hoistable {
        parts.push(emit_fused_for_each_neighbor(folds, ctx)?);
    }
    // f32 RMW CAS-loop pre-pass: when the active kernel upgraded an
    // f32 SoA column to `array<atomic<u32>>` (because the body
    // contains an `Assign(AgentField{f32, …}, …)` chronicle-consumer
    // RMW write), identify the FIRST such Assign in the residual list
    // and var-promote every Let in the prefix so the CAS-loop body
    // can re-execute the chain each iteration. Lets in the prefix
    // emit as `var local_N: T;` declarations BEFORE the loop and as
    // `local_N = V;` assignments INSIDE the loop. After the loop
    // exits, `local_N` holds the snapshot the successful CAS was
    // computed against — so post-Assign conditional checks (e.g.
    // `if (old_hp > 0.0 && new_hp <= 0.0)`) reflect the actual
    // transition that won the CAS, not a stale read from before
    // contention.
    //
    // Today this handles a SINGLE upgraded RMW Assign per stmt list
    // (the canonical chronicle-consumer shape). Multiple upgraded
    // RMW Assigns in one list are not yet collapsed into a single
    // CAS loop — they fall back to the per-stmt CAS-loop emit for
    // each, which is correct but doesn't amortise the snapshot
    // across writes. No production fixture exercises that shape today.
    let f32_rmw_idx_and_id: Option<(usize, AgentFieldId)> = if ctx.f32_atomic_field_writes.get() != 0 {
        residual.iter().enumerate().find_map(|(pos, &idx)| {
            let stmt_id = list.stmts[idx];
            let stmt_node = <CgProgram as StmtArena>::get(ctx.prog, stmt_id)?;
            stmt_is_f32_agent_field_assign(ctx.prog, stmt_node).and_then(|field| {
                let bit = f32_field_atomic_bit(field)?;
                if (ctx.f32_atomic_field_writes.get() >> bit) & 1 == 1 {
                    Some((pos, field))
                } else {
                    None
                }
            })
        })
    } else {
        None
    };

    // Snapshot the var-promoted set so any locals we promote here
    // don't leak into a sibling stmt list (e.g. an else branch).
    let saved_var_promoted = ctx.var_promoted_locals.borrow().clone();

    // If we found an upgraded RMW Assign, identify the chain of
    // preceding Let stmts whose value-expression depends (directly or
    // transitively) on a Read of the upgraded f32 field. Those
    // Lets must re-execute INSIDE the CAS loop so retries see fresh
    // `atomicLoad` snapshots; we mark their locals for var-promotion
    // and accumulate `var local_N: T;` declarations to emit before
    // the loop block.
    //
    // Lets whose value-expression does NOT depend on the upgraded
    // field (e.g. event-payload reads `local_5 = atomicLoad(
    // &event_ring[…])` for the target slot) are NOT promoted —
    // var-promoting them would defer their first assignment to inside
    // the loop body, but stmts EARLIER in the residual list (e.g. a
    // hoisted `target_expr_<N> = local_5` index binding pushed via
    // `pending_target_lets`) reference them BEFORE the loop. Such
    // Lets stay as ordinary `let` bindings outside the loop and are
    // read normally inside the loop.
    //
    // Chain analysis is forward over the residual stmts: walk in
    // source order; for each Let with `value` containing
    // `Read(AgentField{upgraded_f, …})` OR a `ReadLocal(L)` for
    // `L in chain_set`, add the let's local to `chain_set` AND
    // record its (local, ty) for var declaration. Non-Let stmts
    // (e.g. ForEachNeighborBody) before the Assign are emitted
    // normally — they run once outside the loop.
    let mut var_decls: Vec<(LocalId, CgTy)> = Vec::new();
    let mut chain_locals: std::collections::HashSet<LocalId> = std::collections::HashSet::new();
    if let Some((rmw_pos, field)) = f32_rmw_idx_and_id {
        for pos in 0..rmw_pos {
            let stmt_id = list.stmts[residual[pos]];
            if let Some(CgStmt::Let { local, value, ty }) =
                <CgProgram as StmtArena>::get(ctx.prog, stmt_id)
            {
                if expr_depends_on_upgraded_field(*value, field, &chain_locals, ctx.prog) {
                    chain_locals.insert(*local);
                    var_decls.push((*local, *ty));
                }
            }
        }
        let mut promoted = ctx.var_promoted_locals.borrow_mut();
        for (lid, _) in &var_decls {
            promoted.insert(*lid);
        }
    }

    // Then the residual stmts (everything not hoisted) in their
    // original order. Each is emitted via the per-stmt path which
    // handles its own (non-fused) ForEachNeighbor singleton case.
    //
    // AtomicCAS guard: when the kernel was flagged for atomic
    // alive-writes (the kernel emit's body scan found
    // `Assign(Alive, _, Lit(Bool(false)))`), an Assign matching that
    // pattern at residual index `i` lowers to a CAS let-binding
    // (handled in `lower_cg_stmt_body_to_wgsl`'s Assign arm). Every
    // subsequent residual stmt must execute ONLY when the CAS won
    // the transition (the thread that flipped alive 1→0); we wrap
    // them in `if (_alive_cas_<stmt_id>.exchanged) { ... }` here.
    //
    // The wrap is a structural list-level transformation (the per-
    // stmt path can't see "siblings after me" in isolation); both
    // the rewrite-the-Assign and wrap-the-tail steps must agree on
    // the same flag, hence the shared `EmitCtx::alive_atomic_writes`
    // gate.
    //
    // f32 RMW wrap: when `f32_rmw_idx_and_id` matched, the prefix
    // Lets + the upgraded Assign are gathered into `cas_loop_body`
    // (the prefix Lets emit as var-assignments per the var-promotion
    // above; the Assign emits as the CAS-attempt-and-break per the
    // f32 RMW arm in `lower_cg_stmt_body_to_wgsl`). The loop block
    // is added to `parts` once after walking the prefix. Suffix
    // stmts then emit normally (var-promoted locals are visible via
    // the outer-scope var declarations).
    let mut wrap_open: Option<u32> = None;
    let mut wrapped: Vec<String> = Vec::new();
    let mut cas_loop_body: Vec<String> = Vec::new();
    let mut cas_loop_parts_idx: Option<usize> = None;
    // Post-CAS emit gating accumulator. When the
    // per-stmt path emits the gated CAS variant for the f32
    // first-writer-wins shape, subsequent residual stmts collect into
    // `f32_wrapped` and are emitted at the end of the list as
    // `if (_f32_cas_did_transition_<sid>) { ... }`. The gate is the
    // stmt_id of the gated Assign — read off `EmitCtx::f32_first_writer_gate`
    // (which the enclosing `If` arm set BEFORE recursing into this list).
    // Mirrors the alive-CAS `wrap_open` / `wrapped` mechanism but
    // gates on `_f32_cas_did_transition_<sid>` instead of
    // `_alive_cas_<sid>.exchanged`.
    let mut f32_wrap_open: Option<u32> = None;
    let mut f32_wrapped: Vec<String> = Vec::new();
    for (pos, &idx) in residual.iter().enumerate() {
        let stmt_id = list.stmts[idx];
        let stmt_node = <CgProgram as StmtArena>::get(ctx.prog, stmt_id).ok_or(
            EmitError::StmtIdOutOfRange {
                id: stmt_id,
                arena_len: ctx.prog.stmts.len() as u32,
            },
        )?;
        let is_alive_cas_site = ctx.alive_atomic_writes.get()
            && stmt_is_set_alive_false(ctx.prog, stmt_node);
        let stmt_wgsl = lower_cg_stmt_to_wgsl(stmt_id, ctx)?;

        // f32 RMW handling — collect chain-dependent Let stmts before
        // the upgraded RMW Assign into the CAS loop body, emit the
        // loop block once at the position of the Assign. Lets that
        // don't depend on the upgraded field (per `chain_locals` /
        // `var_promoted_locals`) flow through the normal path so they
        // run once outside the loop — this matters for index-binding
        // lets like the event-payload `target_slot` extraction whose
        // value is referenced by hoisted `target_expr_<N>: u32 = …`
        // bindings emitted BEFORE the loop block.
        if let Some((rmw_pos, _)) = f32_rmw_idx_and_id {
            if pos < rmw_pos {
                // Is this a Let whose local was marked as a chain
                // member?
                let stmt_node_check = <CgProgram as StmtArena>::get(ctx.prog, stmt_id);
                let is_chain_let = matches!(
                    stmt_node_check,
                    Some(CgStmt::Let { local, .. }) if chain_locals.contains(local),
                );
                if is_chain_let {
                    cas_loop_body.push(stmt_wgsl);
                    continue;
                }
                // Gap among_us#3: non-chain prefix stmts that READ a
                // chain-local (e.g. `agents.set_pos(self, new_pos)`
                // where `new_pos` was promoted to `var local_N` for
                // recompute-on-retry) must ALSO run inside the CAS
                // loop body. Otherwise the emitter places the read
                // (`agent_pos[agent_id] = local_N;`) at the source-
                // order position BEFORE the var declarations are
                // inserted, producing WGSL with a use-before-decl.
                //
                // Soundness: per-agent SoA writes (e.g. `agent_pos`)
                // are single-writer per slot in PerAgent kernels —
                // running the same write on every CAS retry stores
                // the same recomputed value (each retry sees the
                // same loop-iteration snapshot of the chain). The
                // final iteration's value is what persists. No
                // cross-thread race because each `agent_id` only
                // owns its own slot.
                let reads_chain = stmt_node_check
                    .map(|s| stmt_reads_any_chain_local(s, &chain_locals, ctx.prog))
                    .unwrap_or(false);
                if reads_chain {
                    cas_loop_body.push(stmt_wgsl);
                    continue;
                }
                // Non-chain prefix stmt with no chain reads — emit
                // normally below (e.g. event-payload index lets,
                // hoisted target_expr bindings).
            }
            if pos == rmw_pos {
                // The upgraded Assign — its lowered WGSL is already
                // the full `loop { … }` body (per the f32 RMW arm in
                // `lower_cg_stmt_body_to_wgsl`). We rewrite it here:
                // the body of THAT loop becomes the CAS attempt; the
                // prefix Lets we collected above prepend it. Note
                // the Assign-arm's lowered fragment is itself a
                // `loop { let _old_bits …; let _new_bits …; CAS;
                // break; }` shape; we splice the prefix Lets into
                // that loop's body just before the `_new_bits`
                // computation so each iteration re-derives the
                // chain.
                //
                // Simplest mechanical splice: the Assign-arm emits
                // `loop {\n    let _old_bits_<sid> = …;\n    let
                // _new_bits_<sid> = …;\n    let _r_<sid> = …;\n
                // if (_r_<sid>.exchanged) { break; }\n}` — we
                // string-insert the prefix Lets between the
                // `_old_bits` line and the `_new_bits` line. The
                // marker is the literal `let _new_bits_<sid>`.
                let marker = format!("let _new_bits_{}", stmt_id.0);
                let prefix_block = if cas_loop_body.is_empty() {
                    String::new()
                } else {
                    let inner = cas_loop_body.join("\n");
                    format!("{}\n", indent_block(&inner, 1))
                };
                let composed = if let Some(at) = stmt_wgsl.find(&marker) {
                    let (head, tail) = stmt_wgsl.split_at(at);
                    format!("{}{}{}", head, prefix_block, tail)
                } else {
                    // Defensive: if the marker isn't found (the
                    // Assign-arm emit shape changed), fall back to
                    // emitting prefix + assign without splicing.
                    // The CAS still works but the prefix Lets won't
                    // recompute per iteration — same race the
                    // upgrade aims to fix. The assert tells the
                    // builder to update this splice if the shape
                    // ever drifts.
                    debug_assert!(false, "f32 RMW splice marker not found in Assign emit");
                    format!("{}\n{}", prefix_block.trim_end(), stmt_wgsl)
                };
                cas_loop_parts_idx = Some(parts.len());
                parts.push(composed);
                // If this Assign was the gated first-writer-wins
                // write (per `EmitCtx::f32_first_writer_gate`),
                // open the post-CAS emit-gating wrap. Subsequent
                // residual stmts collect into `f32_wrapped` and emit
                // as `if (_f32_cas_did_transition_<sid>) { ... }` at
                // list close. Without this, all CAS-losers (which
                // retried, observed the already-written constant,
                // and CAS'd the same value with no real transition)
                // would still execute the post-write side-effects
                // (e.g. `emit Ignited`), producing N emits per single
                // semantic transition.
                if ctx.f32_first_writer_gate.get() == Some(stmt_id.0) {
                    f32_wrap_open = Some(stmt_id.0);
                }
                continue;
            }
            // pos > rmw_pos — suffix stmt, falls through to the
            // normal alive_cas + plain-push path below.
        }

        if is_alive_cas_site {
            // The Assign-arm above already emitted the CAS
            // let-binding under the name `_alive_cas_<stmt_id>`. Push
            // it as the LAST top-level part, then start collecting
            // the wrap body.
            parts.push(stmt_wgsl);
            wrap_open = Some(stmt_id.0);
            continue;
        }
        // If a gated f32 first-writer-wins CAS is open, route this
        // suffix stmt into the gating wrap instead of
        // the normal `parts` (or alive-CAS `wrapped`). Subsequent
        // alive-CAS sites can still appear inside the gated tail (the
        // alive flip + Defeated emit chain), and the per-stmt emit will
        // route them through their own CAS — but the structural wrap
        // above takes precedence: the entire tail (including any
        // alive-CAS sub-tree) only runs when the f32 transition won.
        if f32_wrap_open.is_some() {
            f32_wrapped.push(stmt_wgsl);
            continue;
        }
        if wrap_open.is_some() {
            wrapped.push(stmt_wgsl);
        } else {
            parts.push(stmt_wgsl);
        }
    }
    if let Some(cas_id) = wrap_open {
        let inner = wrapped.join("\n");
        parts.push(format!(
            "if (_alive_cas_{}.exchanged) {{\n{}\n}}",
            cas_id,
            indent_block(&inner, 1),
        ));
    }
    if let Some(gate_sid) = f32_wrap_open {
        // Skip the empty wrap when there were no post-CAS stmts to
        // gate (e.g. plague_city's `ApplyLastRites` whose body is
        // just `if (hunger > 0) { set_hunger(t, 0.0); }` with no
        // emit). The wrap would still type-check (`if (bool) { }`
        // is valid WGSL) but it's noisy and a no-op. The transition
        // tracking variable's `var` decl is still emitted by the
        // per-stmt CAS arm — harmless dead store, and removing the
        // emit conditionally on the wrap being non-empty would
        // require a second source-of-truth for the gate decision.
        // Keep it simple: emit the wrap only when there's something
        // inside.
        if !f32_wrapped.is_empty() {
            let inner = f32_wrapped.join("\n");
            parts.push(format!(
                "if (_f32_cas_did_transition_{}) {{\n{}\n}}",
                gate_sid,
                indent_block(&inner, 1),
            ));
        }
    }
    // Prepend var declarations for the f32 RMW chain (must precede
    // the CAS loop in the WGSL block scope so the loop body's
    // `local_N = V;` assignments resolve, AND the suffix `if`'s
    // `local_N` reads see the post-loop committed values). Inserted
    // at the position the CAS loop occupies in `parts` so anything
    // emitted before the chain (e.g. fold hoists, non-chain prefix
    // stmts) keeps its source order.
    if let Some(idx) = cas_loop_parts_idx {
        if !var_decls.is_empty() {
            let decls: String = var_decls
                .iter()
                .map(|(lid, ty)| format!("var local_{}: {};", lid.0, cg_ty_to_wgsl(*ty)))
                .collect::<Vec<_>>()
                .join("\n");
            parts.insert(idx, decls);
        }
    }
    // Restore the var-promotion set so promotions made here don't
    // leak into sibling stmt lists.
    ctx.var_promoted_locals.replace(saved_var_promoted);
    // Restore the outer scope's view-fold target-locals capture so a
    // nested stmt list (e.g. an If branch inside a fold body) can't
    // permanently reset it for the surrounding handler.
    ctx.view_target_locals.replace(saved_view_targets);
    // Restore the outer scope's cross-agent target-let bound set so
    // bindings emitted inside this list don't shadow outer-scope
    // identifiers when control returns to the surrounding emit.
    ctx.bound_target_exprs.replace(saved_bound_targets);
    Ok(parts.join("\n"))
}

/// Try the `distance(a, b) <cmp> r` → `dot(d, d) <cmp> r*r` peephole
/// rewrite. Returns `Ok(Some(wgsl))` when the pattern matches and the
/// rewrite is safe; `Ok(None)` when the binary should fall through to
/// the generic emit path.
///
/// **Pattern**: lhs is `Builtin { fn_id: Distance, args: [a, b] }`
/// and op is one of `LtF32` / `LeF32` / `GtF32` / `GeF32`. Both `a`
/// and `b` must be pure (re-evaluating them is correct and cheap)
/// AND the comparison's `rhs` must also be pure (it gets squared, so
/// `r * r` would re-evaluate `r` once).
///
/// **Why pureness matters**: WGSL has no expression-position
/// `let`-binding, so we inline the operands twice (`a-b` and `a-b`
/// inside `dot`). Re-evaluation is fine for pure reads but would
/// double-fire any side effect or atomic.
///
/// **Soundness**: `||a-b||² < r²` is equivalent to `||a-b|| < r`
/// when `r >= 0`. Sim radii are always positive (perception /
/// separation / view radii are config-const f32s with positive
/// defaults); we don't gate on a runtime sign check. If a future
/// fixture introduces a negative-radius compare (semantically
/// `false` for any agent pair, since distance is non-negative),
/// the peephole would silently flip results — flag this in the
/// caller's contract if the radius can ever be < 0.
fn try_rewrite_distance_compare(
    op: BinaryOp,
    lhs: CgExprId,
    rhs: CgExprId,
    ctx: &EmitCtx,
) -> Result<Option<String>, EmitError> {
    use BinaryOp::*;
    if !matches!(op, LtF32 | LeF32 | GtF32 | GeF32) {
        return Ok(None);
    }
    let lhs_node = match <CgProgram as ExprArena>::get(ctx.prog, lhs) {
        Some(n) => n,
        None => return Ok(None),
    };
    let (a, b) = match lhs_node {
        CgExpr::Builtin {
            fn_id: BuiltinId::Distance,
            args,
            ..
        } if args.len() == 2 => (args[0], args[1]),
        _ => return Ok(None),
    };
    if !expr_is_pure_for_hoisting(a, ctx)
        || !expr_is_pure_for_hoisting(b, ctx)
        || !expr_is_pure_for_hoisting(rhs, ctx)
    {
        return Ok(None);
    }
    let a_wgsl = lower_cg_expr_to_wgsl(a, ctx)?;
    let b_wgsl = lower_cg_expr_to_wgsl(b, ctx)?;
    let r_wgsl = lower_cg_expr_to_wgsl(rhs, ctx)?;
    let cmp = binary_op_to_wgsl(op);
    // dot((a)-(b), (a)-(b)) <cmp> ((r)*(r))
    Ok(Some(format!(
        "(dot(({a}) - ({b}), ({a}) - ({b})) {cmp} (({r}) * ({r})))",
        a = a_wgsl,
        b = b_wgsl,
        r = r_wgsl,
        cmp = cmp,
    )))
}

/// True iff the expression rooted at `expr_id` reads only structural
/// values (`AgentField`, `ConfigConst`, `Lit`, `AgentSelfId`,
/// `PerPairCandidateId`) and not any `ReadLocal`. Used by the
/// fold-fusion pre-pass to decide whether a `ForEachNeighbor` can be
/// hoisted past intervening `Let` stmts. A fold whose projection
/// references a `ReadLocal` is bound to a sibling `Let`'s
/// `local_<N>`; moving the fold ahead of that `Let` would emit
/// WGSL that references an undeclared local.
fn expr_is_pure_for_hoisting(expr_id: CgExprId, ctx: &EmitCtx) -> bool {
    expr_is_pure_for_hoisting_in_prog(expr_id, ctx.prog)
}

/// Same predicate as [`expr_is_pure_for_hoisting`] but driven directly
/// off a [`CgProgram`] — usable from non-emit contexts (e.g. lowering
/// passes that need to decide tile-eligibility before any emit context
/// exists). The two share the same recursive structure; this is the
/// CG-program-arena form.
pub fn expr_is_pure_for_hoisting_in_prog(expr_id: CgExprId, prog: &CgProgram) -> bool {
    let Some(node) = <CgProgram as ExprArena>::get(prog, expr_id) else {
        return false;
    };
    match node {
        CgExpr::ReadLocal { .. } => false,
        CgExpr::Read(_)
        | CgExpr::Lit(_)
        | CgExpr::Rng { .. }
        | CgExpr::AgentSelfId
        | CgExpr::PerPairCandidateId
        | CgExpr::EventField { .. }
        | CgExpr::NamespaceField { .. } => true,
        CgExpr::Binary { lhs, rhs, .. } => {
            expr_is_pure_for_hoisting_in_prog(*lhs, prog)
                && expr_is_pure_for_hoisting_in_prog(*rhs, prog)
        }
        CgExpr::Unary { arg, .. } => expr_is_pure_for_hoisting_in_prog(*arg, prog),
        CgExpr::Builtin { args, .. } => args
            .iter()
            .all(|a| expr_is_pure_for_hoisting_in_prog(*a, prog)),
        CgExpr::Select {
            cond, then, else_, ..
        } => {
            expr_is_pure_for_hoisting_in_prog(*cond, prog)
                && expr_is_pure_for_hoisting_in_prog(*then, prog)
                && expr_is_pure_for_hoisting_in_prog(*else_, prog)
        }
        CgExpr::NamespaceCall { args, .. } => args
            .iter()
            .all(|a| expr_is_pure_for_hoisting_in_prog(*a, prog)),
        // Static-table reads are pure functions of the index
        // expression — values are baked into the IR node, no
        // runtime mutation possible.
        CgExpr::TableLookup { index, .. } => {
            expr_is_pure_for_hoisting_in_prog(*index, prog)
        }
    }
}

/// Emit one cell-walk that updates every accumulator in `folds` (each
/// a `CgStmt::ForEachNeighbor`). All entries must share the same
/// `radius_cells` — the caller (`lower_cg_stmt_list_to_wgsl`) checks
/// this invariant when greedy-grouping adjacent fold stmts. Used for
/// both the singleton case (one fold, equivalent to the prior emit)
/// and the fused case (multiple folds collapsed into one walk).
///
/// # Why fuse
///
/// The dominant cost in a boids-style body is the inner-loop
/// dereferences (`spatial_grid_cells[..]`, `agent_pos[per_pair_candidate]`)
/// and the `distance` compare inside each projection. With N
/// independent folds, every neighbor pays for those N times even
/// though the cell walk and `per_pair_candidate` stream are
/// identical. Fusing collapses to one walk + one stream, with N
/// projection updates per neighbor — a near-N× reduction in memory
/// traffic on the dominant axis.
///
/// The acc init (`var local_<N>: <ty> = <init>`) lands BEFORE the
/// nested loops; the per-neighbor accumulator updates land inside
/// the innermost loop in source order. Each accumulator's projection
/// expression resolves independently against the shared
/// `per_pair_candidate` binding.
fn emit_fused_for_each_neighbor(
    folds: &[&CgStmt],
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    debug_assert!(!folds.is_empty(), "caller groups at least one fold");
    let radius = match folds[0] {
        CgStmt::ForEachNeighbor { radius_cells, .. } => *radius_cells as i32,
        _ => unreachable!("caller restricts to ForEachNeighbor"),
    };
    // Gap dungeon_horde#1: the fused-emit shape requires every fold in
    // the group to share the same lowered origin expression — the gate
    // prefix can reference only one origin. The hoister at the call
    // site keys its partition by `(radius, lowered origin WGSL)` so
    // this invariant holds for every group it produces. We pick the
    // first fold's origin and trust the hoister's grouping.
    let origin = match folds[0] {
        CgStmt::ForEachNeighbor { origin, .. } => *origin,
        _ => unreachable!("caller restricts to ForEachNeighbor"),
    };

    // Are we emitting inside a tiled-MoveBoid kernel
    // (DispatchShape::PerCell)? If so, the surrounding kernel
    // preamble (in `kernel.rs::tiled_per_cell_preamble`) has already
    // populated `tile_pos` / `tile_vel` / `tile_count` workgroup
    // arrays. We emit a single per-lane walk over those arrays, and
    // engage the agent_field_access tile substitution (so each
    // projection's `agent_pos[per_pair_candidate]` reads land on
    // `tile_pos[<tile-index>]` instead of global memory). The
    // cell-walk path (the else branch below) keeps the original
    // 27-cell global-memory walk for non-tiled kernels.
    let is_tiled = matches!(
        ctx.dispatch.get(),
        Some(crate::cg::dispatch::DispatchShape::PerCell)
    );

    // Pre-render every fold's init expression. We hold off on the
    // projection until we know whether to enter tile-walk mode, so
    // the substitution into tile_pos / tile_vel happens correctly.
    let mut prepared: Vec<(u32, String, String, CgExprId)> = Vec::with_capacity(folds.len());
    for f in folds {
        match f {
            CgStmt::ForEachNeighbor {
                acc_local,
                acc_ty,
                init,
                projection,
                ..
            } => {
                let init_wgsl = lower_cg_expr_to_wgsl(*init, ctx)?;
                let ty_wgsl = cg_ty_to_wgsl(*acc_ty);
                prepared.push((acc_local.0, ty_wgsl, init_wgsl, *projection));
            }
            _ => unreachable!("caller restricts to ForEachNeighbor"),
        }
    }

    // var local_<N>: <ty> = <init>;  (one line per fold, top-level)
    let mut head = String::new();
    for (n, ty_wgsl, init_wgsl, _) in &prepared {
        head.push_str(&format!("var local_{n}: {ty_wgsl} = {init_wgsl};\n"));
    }

    if is_tiled {
        // Tile-walk: lanes process one home agent each (already
        // bound to `agent_id` by the per-cell preamble). The fold
        // walks the 27 neighbor slots loaded into `tile_*` by the
        // workgroup. Engaging `ctx.tile_walk_index` inside the inner
        // loop redirects every `agent_pos[per_pair_candidate]` /
        // `agent_vel[per_pair_candidate]` projection read to the
        // workgroup-local tile.
        //
        // `_tile_idx = nbr_lane * SPATIAL_MAX_PER_CELL + _i` is the
        // shared expression both projections agree on; the
        // substitution emit reads it from the tile_walk_index
        // RefCell.
        let prior_idx = ctx
            .tile_walk_index
            .replace(Some("_tile_idx".to_string()));
        let mut updates = String::new();
        for (n, _, _, projection_id) in &prepared {
            let proj_wgsl = lower_cg_expr_to_wgsl(*projection_id, ctx)?;
            updates.push_str(&format!(
                "            local_{n} = (local_{n} + ({proj_wgsl}));\n"
            ));
        }
        ctx.tile_walk_index.replace(prior_idx);

        // Iterate over the 27 cells in the tile. We still need
        // `per_pair_candidate` for the projection's `!= self` check
        // and any other AgentId reads — the tile doesn't store ids
        // (they'd take another 3 KB of workgroup memory we'd rather
        // not spend), so we re-read from spatial_grid_cells. That's
        // one global read per inner iteration, which the per-tile
        // pos/vel cache offsets several-fold.
        let body = format!(
            "{head}{{\n\
             \x20   for (var nbr_lane: u32 = 0u; nbr_lane < 27u; nbr_lane = nbr_lane + 1u) {{\n\
             \x20       let _nbr_count = tile_count[nbr_lane];\n\
             \x20       let _dz = i32(nbr_lane / 9u) - 1;\n\
             \x20       let _dy = i32((nbr_lane / 3u) % 3u) - 1;\n\
             \x20       let _dx = i32(nbr_lane % 3u) - 1;\n\
             \x20       let _nbr_cell = cell_index(\n\
             \x20           i32(home_cx) + _dx,\n\
             \x20           i32(home_cy) + _dy,\n\
             \x20           i32(home_cz) + _dz,\n\
             \x20       );\n\
             \x20       let _nbr_start = spatial_grid_starts[_nbr_cell];\n\
             \x20       for (var _i: u32 = 0u; _i < _nbr_count; _i = _i + 1u) {{\n\
             \x20           let per_pair_candidate = spatial_grid_cells[_nbr_start + _i];\n\
             \x20           let _tile_idx = nbr_lane * SPATIAL_MAX_PER_CELL + _i;\n\
             {updates}\
             \x20       }}\n\
             \x20   }}\n\
             }}",
            head = head,
            updates = updates,
        );
        Ok(body)
    } else {
        // Cell-walk (per-agent dispatch fallback): emit the original
        // 27-cell global-memory walk. Projections render against
        // global agent_pos / agent_vel reads (no tile substitution).
        let mut updates = String::new();
        for (n, _, _, projection_id) in &prepared {
            let proj_wgsl = lower_cg_expr_to_wgsl(*projection_id, ctx)?;
            updates.push_str(&format!(
                "                    local_{n} = (local_{n} + ({proj_wgsl}));\n"
            ));
        }
        // Auto-injected distance gate (Gap dungeon_layout#1) — same
        // motivation as `emit_for_each_neighbor_body`. The boundary-
        // cell clamp pools OOB agents into the boundary cell; without
        // a per-pair distance guard, the per-candidate projection
        // would see those OOB agents as if they were genuine
        // neighbours. Gating squared distance against the
        // cell-neighbourhood diagonal `((r+1) * cell_size * sqrt(3))²
        // = 3 * (r+1)² * cell_size²` kills the pool without rejecting
        // any in-window candidate.
        let r_plus_one = (radius as u32 + 1) as u32;
        let r_plus_one_sq = r_plus_one * r_plus_one;
        // Gap dungeon_horde#1: lower the shared origin once. See the
        // body-form's `emit_for_each_neighbor_body` for the contract.
        let origin_wgsl = lower_cg_expr_to_wgsl(origin, ctx)?;
        let body = format!(
            "{head}{{\n\
             \x20   let _gate_origin_id: u32 = {origin_wgsl};\n\
             \x20   let _gate_origin_pos: vec3<f32> = agent_pos[_gate_origin_id];\n\
             \x20   let _self_cell_f = (_gate_origin_pos + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
             \x20   let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
             \x20   let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
             \x20   let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
             \x20   let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
             \x20   let _gate_radius_sq = 3.0 * f32({r_plus_one_sq}u) * SPATIAL_CELL_SIZE * SPATIAL_CELL_SIZE;\n\
             \x20   for (var dz: i32 = -{r}; dz <= {r}; dz = dz + 1) {{\n\
             \x20       for (var dy: i32 = -{r}; dy <= {r}; dy = dy + 1) {{\n\
             \x20           for (var dx: i32 = -{r}; dx <= {r}; dx = dx + 1) {{\n\
             \x20               let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
             \x20               let _start = spatial_grid_starts[_cell];\n\
             \x20               let _end = spatial_grid_starts[_cell + 1u];\n\
             \x20               for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {{\n\
             \x20                   let per_pair_candidate = spatial_grid_cells[_i];\n\
             \x20                   let _gate_dxyz = agent_pos[per_pair_candidate] - _gate_origin_pos;\n\
             \x20                   let _gate_dist_sq = dot(_gate_dxyz, _gate_dxyz);\n\
             \x20                   if (_gate_dist_sq > _gate_radius_sq) {{ continue; }}\n\
             {updates}\
             \x20               }}\n\
             \x20           }}\n\
             \x20       }}\n\
             \x20   }}\n\
             }}",
            r = radius,
            r_plus_one_sq = r_plus_one_sq,
            head = head,
            origin_wgsl = origin_wgsl,
            updates = updates,
        );
        Ok(body)
    }
}

/// Emit a per-slot body block for [`CgStmt::ForEachAgentBody`] —
/// the source-level `for_each_agent <binder> { … }` body-shape
/// primitive.
///
/// Walks every alive agent slot in deterministic linear order
/// (`0..agent_cap`). The body executes once per `agent_alive[i] != 0u`
/// candidate; each candidate slot id is bound to `per_pair_candidate`
/// (matching the existing pair-bound emit convention) so the body's
/// `agent_<field>[per_pair_candidate]` accesses (lowered via
/// [`AgentRef::PerPairCandidate`]) resolve against the global SoA
/// buffers without inventing a parallel naming scheme.
///
/// # Iteration-order contract (P3, P11)
///
/// Linear scan from slot 0 to slot `agent_cap - 1`, identical on CPU
/// and GPU backends. Bodies that perform sibling-slot writes commit in
/// slot-id order; the surrounding rule retags its dispatch to
/// `OneShot` (see `lower_one_handler` in `cg::lower::physics`) so a
/// single thread serialises the entire scan.
///
/// # P5 RNG note
///
/// Per-rule RNG draws inside the body must continue to flow through
/// `per_agent_u32(seed, agent_id, tick, purpose)` — the iteration
/// index `per_pair_candidate` is the candidate slot id and is the
/// correct value to thread as `agent_id` for per-candidate draws.
fn emit_for_each_agent_body(
    body_list: crate::cg::stmt::CgStmtListId,
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    // Set the per-iteration RNG iter var so any `rng.<method>()`
    // inside the body routes through `per_agent_u32_with_extra(...,
    // per_pair_candidate)`. Save/restore so nested loops + non-loop
    // emit don't see this var.
    let prev = ctx.rng_loop_iter_var.borrow_mut().replace("per_pair_candidate".to_string());
    let body_wgsl = lower_cg_stmt_list_to_wgsl(body_list, ctx)?;
    *ctx.rng_loop_iter_var.borrow_mut() = prev;
    // Indent the body so it nests under the per-iteration alive guard
    // (one outer loop level + one if-guard level → two indents = 8
    // spaces).
    let indented_body = indent_block(&body_wgsl, 2);
    let out = format!(
        "{{\n\
         \x20   for (var per_pair_candidate: u32 = 0u; per_pair_candidate < cfg.agent_cap; per_pair_candidate = per_pair_candidate + 1u) {{\n\
         \x20       if (agent_alive[per_pair_candidate] != 0u) {{\n\
         {indented_body}\n\
         \x20       }}\n\
         \x20   }}\n\
         }}",
        indented_body = indented_body,
    );
    Ok(out)
}

/// Emit a per-candidate body block for [`CgStmt::ForEachNeighborBody`].
///
/// Mirrors the cell-walk path of [`emit_fused_for_each_neighbor`] but
/// substitutes the body's lowered WGSL for the per-candidate
/// accumulator update line. Each candidate slot id is bound to
/// `per_pair_candidate` (matching the existing pair-bound emit
/// convention) so the body's `agent_<field>[per_pair_candidate]`
/// accesses (lowered via [`AgentRef::PerPairCandidate`]) resolve
/// against the global SoA buffers.
///
/// The emit is wrapped in `{}` so the helper-level locals
/// (`_self_cell_f`, `_self_cx`, …) don't leak into the surrounding
/// kernel scope — the same scoping the fold-form path uses.
fn emit_for_each_neighbor_body(
    body_list: crate::cg::stmt::CgStmtListId,
    radius_cells: u32,
    origin: CgExprId,
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    // Set the per-iteration RNG iter var — see
    // `emit_for_each_agent_body` for rationale.
    let prev = ctx.rng_loop_iter_var.borrow_mut().replace("per_pair_candidate".to_string());
    let body_wgsl = lower_cg_stmt_list_to_wgsl(body_list, ctx)?;
    *ctx.rng_loop_iter_var.borrow_mut() = prev;
    let r = radius_cells as i32;
    // Gap dungeon_horde#1: lower the caller-supplied origin expression
    // to WGSL. For `spatial.<...>(self)` this is `agent_id`, matching
    // the pre-fix hard-coded reference exactly; for non-self origins
    // (e.g. `spatial.<...>(s)` where `s` is event-pattern bound) it's
    // the let-bound local's name. The `_gate_origin_id` variable below
    // binds the lowered value once so the gate prefix can reuse it
    // without re-lowering.
    let origin_wgsl = lower_cg_expr_to_wgsl(origin, ctx)?;
    // Indent each line of the body so it nests cleanly inside the
    // 4-deep loop chain (3 cell-axis loops + 1 candidate loop). Six
    // levels of 4-space indent → 24 spaces.
    let indented_body = indent_block(&body_wgsl, 6);
    // Auto-injected distance gate (Gap dungeon_layout#1). The spatial
    // grid's `cell_index` helper clamps OOB axis coordinates to
    // `GRID_DIM - 1`, which silently pools every distant agent into
    // the boundary cell. Walks on a boundary-side cell then see those
    // OOB agents as "neighbours" by cell-coord arithmetic even though
    // they are FAR in world coordinates. Gating per-candidate by
    // squared distance against the cell-neighbourhood diagonal kills
    // the OOB pool without rejecting any in-window candidate.
    //
    // Bound: the cell-walk visits `(2r+1)³` cells centered on self's
    // cell. The maximum world distance between self at one corner of
    // its cell and a candidate at the opposite corner of the most-
    // diagonal visited cell is `(r+1) * cell_size * sqrt(3)` (3D
    // diagonal). Squaring: `3 * (r+1)² * cell_size²`. Using this as
    // the upper bound guarantees no false negative — any candidate
    // genuinely inside the cell window passes the gate.
    //
    // The gate body re-uses `agent_pos[agent_id]` (already bound — the
    // template's `_self_cell_f` read surfaces it) plus the candidate
    // read `agent_pos[per_pair_candidate]`. The latter introduces no
    // new buffer binding requirement: `agent_pos` is already in the
    // kernel's BGL via the implicit-self.pos read surfaced by
    // `collect_stmt_dependencies` for `CgStmt::ForEachNeighborBody`.
    let r_plus_one = (radius_cells + 1) as u32;
    let r_plus_one_sq = r_plus_one * r_plus_one;
    let out = format!(
        "{{\n\
         \x20   let _gate_origin_id: u32 = {origin_wgsl};\n\
         \x20   let _gate_origin_pos: vec3<f32> = agent_pos[_gate_origin_id];\n\
         \x20   let _self_cell_f = (_gate_origin_pos + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20   let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20   let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20   let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20   let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20   let _gate_radius_sq = 3.0 * f32({r_plus_one_sq}u) * SPATIAL_CELL_SIZE * SPATIAL_CELL_SIZE;\n\
         \x20   for (var dz: i32 = -{r}; dz <= {r}; dz = dz + 1) {{\n\
         \x20       for (var dy: i32 = -{r}; dy <= {r}; dy = dy + 1) {{\n\
         \x20           for (var dx: i32 = -{r}; dx <= {r}; dx = dx + 1) {{\n\
         \x20               let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20               let _start = spatial_grid_starts[_cell];\n\
         \x20               let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20               for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {{\n\
         \x20                   let per_pair_candidate = spatial_grid_cells[_i];\n\
         \x20                   let _gate_dxyz = agent_pos[per_pair_candidate] - _gate_origin_pos;\n\
         \x20                   let _gate_dist_sq = dot(_gate_dxyz, _gate_dxyz);\n\
         \x20                   if (_gate_dist_sq > _gate_radius_sq) {{ continue; }}\n\
         {indented_body}\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20   }}\n\
         }}",
        r = r,
        r_plus_one_sq = r_plus_one_sq,
        origin_wgsl = origin_wgsl,
        indented_body = indented_body,
    );
    Ok(out)
}

// ---------------------------------------------------------------------------
// CgTy → WGSL type name (used by snapshot-style harnesses; not the
// public surface but kept here so the mapping has one home).
// ---------------------------------------------------------------------------

/// WGSL type name for a [`CgTy`]. Useful in tests + future kernel
/// emission. Exhaustive — adding a CgTy variant forces a decision.
pub fn cg_ty_to_wgsl(ty: CgTy) -> String {
    match ty {
        CgTy::Bool => "bool".to_string(),
        CgTy::U32 => "u32".to_string(),
        CgTy::I32 => "i32".to_string(),
        CgTy::F32 => "f32".to_string(),
        CgTy::Vec3F32 => "vec3<f32>".to_string(),
        // AgentId, Tick both lower to u32 at the WGSL boundary — the
        // engine narrows ticks (u64 → u32) and represents agent slot
        // ids as u32 indices.
        CgTy::AgentId | CgTy::Tick => "u32".to_string(),
        // ViewKey is a phantom u32 at the WGSL level — its semantic
        // payload is whatever the view's primary storage carries.
        CgTy::ViewKey { .. } => "u32".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{
        AgentFieldId, ConfigConstId, EventRingId, MaskId, ViewId,
    };
    use crate::cg::op::EventKindId;
    use crate::cg::stmt::{
        CgMatchArm, CgStmt, CgStmtId, CgStmtList, CgStmtListId, EventField, LocalId,
        MatchArmBinding, VariantId,
    };

    /// Build a fresh `CgProgram` and populate it directly via the
    /// `pub` arena fields. Task 4.1 tests don't need a full builder
    /// pass — they only need to wire ids that resolve.
    fn empty_prog() -> CgProgram {
        CgProgram::default()
    }

    fn push_expr(prog: &mut CgProgram, e: CgExpr) -> CgExprId {
        let id = CgExprId(prog.exprs.len() as u32);
        prog.exprs.push(e);
        id
    }

    fn push_stmt(prog: &mut CgProgram, s: CgStmt) -> CgStmtId {
        let id = CgStmtId(prog.stmts.len() as u32);
        prog.stmts.push(s);
        id
    }

    fn push_list(prog: &mut CgProgram, l: CgStmtList) -> CgStmtListId {
        let id = CgStmtListId(prog.stmt_lists.len() as u32);
        prog.stmt_lists.push(l);
        id
    }

    // ---- 1. LitValue per-variant ----

    #[test]
    fn lower_lit_each_variant() {
        let mut prog = empty_prog();
        let cases: Vec<(LitValue, &'static str)> = vec![
            (LitValue::Bool(true), "true"),
            (LitValue::Bool(false), "false"),
            (LitValue::U32(7), "7u"),
            (LitValue::I32(-3), "-3i"),
            (LitValue::F32(1.5), "1.5"),
            (LitValue::Tick(42), "42u"),
            (LitValue::AgentId(11), "11u"),
        ];
        for (lit, expected) in cases {
            let id = push_expr(&mut prog, CgExpr::Lit(lit));
            let ctx = EmitCtx::structural(&prog);
            assert_eq!(lower_cg_expr_to_wgsl(id, &ctx).unwrap(), expected);
        }

        // Vec3F32 separately — `{:?}` on f32 → "1.0", "2.0", "3.0".
        let id = push_expr(
            &mut prog,
            CgExpr::Lit(LitValue::Vec3F32 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            }),
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(id, &ctx).unwrap(),
            "vec3<f32>(1.0, 2.0, 3.0)"
        );
    }

    // ---- 2. BinaryOp class coverage (arith, comparison, logical) ----

    #[test]
    fn lower_binary_arith_comparison_logical() {
        // (hp + 1.0)
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let add = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: hp,
                rhs: one,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(add, &ctx).unwrap(),
            "(agent_hp[agent_id] + 1.0)"
        );

        // (hp < 5.0)
        let five = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(5.0)));
        let lt = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: five,
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(lt, &ctx).unwrap(),
            "(agent_hp[agent_id] < 5.0)"
        );

        // (true && false)
        let t = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
        let f = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(false)));
        let and = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::And,
                lhs: t,
                rhs: f,
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(lower_cg_expr_to_wgsl(and, &ctx).unwrap(), "(true && false)");
    }

    /// Spot-check every `BinaryOp` symbol mapping (smoke test for the
    /// exhaustive match).
    #[test]
    fn binary_op_to_wgsl_covers_each_class() {
        // Arithmetic
        assert_eq!(binary_op_to_wgsl(BinaryOp::AddF32), "+");
        assert_eq!(binary_op_to_wgsl(BinaryOp::SubU32), "-");
        assert_eq!(binary_op_to_wgsl(BinaryOp::MulI32), "*");
        assert_eq!(binary_op_to_wgsl(BinaryOp::DivF32), "/");
        // Comparisons
        assert_eq!(binary_op_to_wgsl(BinaryOp::LtF32), "<");
        assert_eq!(binary_op_to_wgsl(BinaryOp::LeU32), "<=");
        assert_eq!(binary_op_to_wgsl(BinaryOp::GtI32), ">");
        assert_eq!(binary_op_to_wgsl(BinaryOp::GeF32), ">=");
        // Equality
        assert_eq!(binary_op_to_wgsl(BinaryOp::EqU32), "==");
        assert_eq!(binary_op_to_wgsl(BinaryOp::EqAgentId), "==");
        assert_eq!(binary_op_to_wgsl(BinaryOp::NeF32), "!=");
        // Logical
        assert_eq!(binary_op_to_wgsl(BinaryOp::And), "&&");
        assert_eq!(binary_op_to_wgsl(BinaryOp::Or), "||");
    }

    // ---- 3. UnaryOp class coverage ----

    #[test]
    fn lower_unary_neg_not_abs_sqrt_normalize() {
        let mut prog = empty_prog();
        // -hp
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let neg = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::NegF32,
                arg: hp,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(lower_cg_expr_to_wgsl(neg, &ctx).unwrap(), "(-agent_hp[agent_id])");

        // !alive
        let alive = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::Self_,
            }),
        );
        let not_alive = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::NotBool,
                arg: alive,
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(not_alive, &ctx).unwrap(),
            "(!(agent_alive[agent_id] != 0u))"
        );

        // abs(slow_factor_q8)
        let sf = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::SlowFactorQ8,
                target: AgentRef::Self_,
            }),
        );
        let abs = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::AbsI32,
                arg: sf,
                ty: CgTy::I32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(abs, &ctx).unwrap(),
            "abs(agent_slow_factor_q8[agent_id])"
        );

        // sqrt(2.0)
        let two = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.0)));
        let sq = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::SqrtF32,
                arg: two,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(lower_cg_expr_to_wgsl(sq, &ctx).unwrap(), "sqrt(2.0)");

        // normalize(pos)
        let pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let norm = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::NormalizeVec3F32,
                arg: pos,
                ty: CgTy::Vec3F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(norm, &ctx).unwrap(),
            "normalize(agent_pos[agent_id])"
        );
    }

    // ---- 4. Builtin coverage ----

    #[test]
    fn lower_builtin_distance_min_clamp_view_call() {
        let mut prog = empty_prog();
        let pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let actor_pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Actor,
            }),
        );
        // distance(self.pos, actor.pos)
        let dist = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Distance,
                args: vec![pos, actor_pos],
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(dist, &ctx).unwrap(),
            "distance(agent_pos[agent_id], agent_pos[actor_id])"
        );

        // min_f32(1.0, 2.0)
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let two = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.0)));
        let min = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Min(NumericTy::F32),
                args: vec![one, two],
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(min, &ctx).unwrap(),
            "min_f32(1.0, 2.0)"
        );

        // clamp_u32(level, 1, 99)
        let level = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Level,
                target: AgentRef::Self_,
            }),
        );
        let lo = push_expr(&mut prog, CgExpr::Lit(LitValue::U32(1)));
        let hi = push_expr(&mut prog, CgExpr::Lit(LitValue::U32(99)));
        let cl = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Clamp(NumericTy::U32),
                args: vec![level, lo, hi],
                ty: CgTy::U32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(cl, &ctx).unwrap(),
            "clamp_u32(agent_level[agent_id], 1u, 99u)"
        );

        // view_2_get(self_pos)
        let vc = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::ViewCall { view: ViewId(2) },
                args: vec![pos],
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(vc, &ctx).unwrap(),
            "view_2_get(agent_pos[agent_id])"
        );

        // saturating_add_u32 spot-check
        assert_eq!(
            builtin_name(BuiltinId::SaturatingAdd(NumericTy::U32)),
            "saturating_add_u32"
        );
        // log/log2/log10/floor/ceil/round + planar_distance + z_separation + entity
        assert_eq!(builtin_name(BuiltinId::Floor), "floor");
        assert_eq!(builtin_name(BuiltinId::Ceil), "ceil");
        assert_eq!(builtin_name(BuiltinId::Round), "round");
        assert_eq!(builtin_name(BuiltinId::Ln), "log");
        assert_eq!(builtin_name(BuiltinId::Log2), "log2");
        assert_eq!(builtin_name(BuiltinId::Log10), "log10");
        assert_eq!(builtin_name(BuiltinId::PlanarDistance), "planar_distance");
        assert_eq!(builtin_name(BuiltinId::ZSeparation), "z_separation");
        assert_eq!(builtin_name(BuiltinId::Entity), "entity");
    }

    // ---- 5. DataHandle Read coverage (each variant) ----

    #[test]
    fn lower_read_each_data_handle_variant() {
        let mut prog = empty_prog();
        // AgentField — Self_ / Actor / EventTarget / Target(expr_id)
        let target_expr_id = push_expr(&mut prog, CgExpr::Lit(LitValue::AgentId(0)));
        let cases: Vec<(DataHandle, &str)> = vec![
            (
                DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                "agent_hp[agent_id]",
            ),
            (
                DataHandle::AgentField {
                    field: AgentFieldId::Pos,
                    target: AgentRef::Actor,
                },
                "agent_pos[actor_id]",
            ),
            (
                DataHandle::AgentField {
                    field: AgentFieldId::Alive,
                    target: AgentRef::EventTarget,
                },
                "(agent_alive[event_target_id] != 0u)",
            ),
            (
                // Slice 1 (2026-05-03 stdlib-into-CG-IR): `Target(_)`
                // reads now emit indexed access against the SoA.
                // The pre-stmt `let target_expr_<N>: u32 = …;` binding
                // is queued via `pending_target_lets` and drained by
                // `lower_cg_stmt_to_wgsl`; this `lower_cg_expr_to_wgsl`-
                // only test only sees the indexed access form. The
                // dedicated `target_read_emits_stmt_scope_let_binding`
                // test below covers the let-emission via the stmt-
                // level wrapper.
                DataHandle::AgentField {
                    field: AgentFieldId::Pos,
                    target: AgentRef::Target(target_expr_id),
                },
                "agent_pos[target_expr_0]",
            ),
            (
                DataHandle::ViewStorage {
                    view: ViewId(2),
                    slot: ViewStorageSlot::Primary,
                },
                "view_2_primary",
            ),
            (
                DataHandle::EventRing {
                    ring: EventRingId(5),
                    kind: EventRingAccess::Read,
                },
                "event_ring_5_read",
            ),
            (
                DataHandle::ConfigConst {
                    id: ConfigConstId(11),
                },
                "config_11",
            ),
            (
                DataHandle::MaskBitmap { mask: MaskId(3) },
                "mask_3_bitmap",
            ),
            (DataHandle::ScoringOutput, "scoring_output"),
            (
                DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::GridCells,
                },
                "spatial_grid_cells",
            ),
            (
                DataHandle::Rng {
                    purpose: RngPurpose::Action,
                },
                "rng_action",
            ),
        ];
        for (h, expected) in cases {
            let id = push_expr(&mut prog, CgExpr::Read(h));
            let ctx = EmitCtx::structural(&prog);
            assert_eq!(
                lower_cg_expr_to_wgsl(id, &ctx).unwrap(),
                expected,
                "naming for variant {expected}"
            );
        }

        // Plumbing handles still get a structural name (defense-in-
        // depth — they should not appear in expressions but the strategy
        // must round-trip every variant).
        assert_eq!(structural_handle_name(&DataHandle::AliveBitmap), "alive_bitmap");
        assert_eq!(
            structural_handle_name(&DataHandle::IndirectArgs {
                ring: EventRingId(7)
            }),
            "indirect_args_7"
        );
        assert_eq!(
            structural_handle_name(&DataHandle::AgentScratch {
                kind: AgentScratchKind::Packed
            }),
            "agent_scratch_packed"
        );
        assert_eq!(structural_handle_name(&DataHandle::SimCfgBuffer), "sim_cfg_buffer");
        assert_eq!(structural_handle_name(&DataHandle::SnapshotKick), "snapshot_kick");
    }

    // ---- 6. Rng — every purpose ----

    #[test]
    fn lower_rng_every_purpose() {
        // Purpose tags are emitted as numeric `<id>u` literals (WGSL has
        // no string type; stochastic_probe Gap #3 close, 2026-05-04). The
        // ids come from `RngPurpose::wgsl_id()` and are fixed forever
        // (host parity helper `engine::rng::per_agent_u32_pcg` accepts
        // the same ids — P11 cross-backend bit-equality).
        let mut prog = empty_prog();
        let cases = [
            (
                RngPurpose::Action,
                "per_agent_u32(seed, agent_id, tick, 1u)",
            ),
            (
                RngPurpose::Sample,
                "per_agent_u32(seed, agent_id, tick, 2u)",
            ),
            (
                RngPurpose::Shuffle,
                "per_agent_u32(seed, agent_id, tick, 3u)",
            ),
            (
                RngPurpose::Conception,
                "per_agent_u32(seed, agent_id, tick, 4u)",
            ),
        ];
        for (purpose, expected) in cases {
            let id = push_expr(
                &mut prog,
                CgExpr::Rng {
            extra: 0,
                    purpose,
                    ty: CgTy::U32,
                },
            );
            let ctx = EmitCtx::structural(&prog);
            assert_eq!(lower_cg_expr_to_wgsl(id, &ctx).unwrap(), expected);
        }
    }

    // ---- 7. Select ----

    #[test]
    fn lower_select_emits_wgsl_select_with_false_first_order() {
        // select(true, hp, 0.0)
        // → WGSL: select(0.0, agent_hp[agent_id], true)  -- false_val FIRST.
        let mut prog = empty_prog();
        let cond = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let sel = push_expr(
            &mut prog,
            CgExpr::Select {
                cond,
                then: hp,
                else_: zero,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(sel, &ctx).unwrap(),
            "select(0.0, agent_hp[agent_id], true)"
        );
    }

    // ---- 8. Statement coverage ----

    #[test]
    fn lower_assign_stmt() {
        // assign(hp <- (hp + 1.0))
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let add = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: hp,
                rhs: one,
                ty: CgTy::F32,
            },
        );
        let s = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: add,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_stmt_to_wgsl(s, &ctx).unwrap(),
            "agent_hp[agent_id] = (agent_hp[agent_id] + 1.0);"
        );
    }

    #[test]
    fn lower_emit_stmt() {
        use crate::cg::program::{EventLayout, FieldLayout};
        let mut prog = empty_prog();
        // Real ring-append needs an event layout to resolve field
        // indices to (offset, ty). Two F32 fields at consecutive
        // payload offsets (0, 1).
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "hp".to_string(),
            FieldLayout {
                word_offset_in_payload: 0,
                word_count: 1,
                ty: CgTy::F32,
            },
        );
        fields.insert(
            "zero".to_string(),
            FieldLayout {
                word_offset_in_payload: 1,
                word_count: 1,
                ty: CgTy::F32,
            },
        );
        prog.event_layouts.insert(
            7,
            EventLayout {
                record_stride_u32: 11,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields,
            },
        );
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let s = push_stmt(
            &mut prog,
            CgStmt::Emit {
                event: EventKindId(7),
                fields: vec![
                    (
                        EventField {
                            event: EventKindId(7),
                            index: 0,
                        },
                        hp,
                    ),
                    (
                        EventField {
                            event: EventKindId(7),
                            index: 1,
                        },
                        zero,
                    ),
                ],
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_stmt_to_wgsl(s, &ctx).unwrap();
        // Real ring-append form: atomicAdd to event_tail[0], bounds-
        // check, then atomicStore tag/tick/payload writes. F32
        // fields wrap in bitcast<u32>.
        assert!(
            wgsl.contains("let slot = atomicAdd(&event_tail[0], 1u);"),
            "expected atomicAdd-to-tail; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 0u], 7u);"),
            "expected tag (event id 7) write; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 2u], bitcast<u32>(agent_hp[agent_id]));"),
            "expected hp f32 bitcast write at offset 2; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 3u], bitcast<u32>(0.0));"),
            "expected zero f32 bitcast write at offset 3; got:\n{wgsl}"
        );
    }

    #[test]
    fn lower_if_with_and_without_else() {
        let mut prog = empty_prog();
        // assign hp <- 1.0
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let assign_one = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: one,
            },
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let assign_zero = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: zero,
            },
        );
        let then_list = push_list(&mut prog, CgStmtList::new(vec![assign_one]));
        let else_list = push_list(&mut prog, CgStmtList::new(vec![assign_zero]));
        let cond_lit = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));

        let if_with_else = push_stmt(
            &mut prog,
            CgStmt::If {
                cond: cond_lit,
                then: then_list,
                else_: Some(else_list),
            },
        );
        let if_no_else = push_stmt(
            &mut prog,
            CgStmt::If {
                cond: cond_lit,
                then: then_list,
                else_: None,
            },
        );

        let ctx = EmitCtx::structural(&prog);
        let with_else = lower_cg_stmt_to_wgsl(if_with_else, &ctx).unwrap();
        assert_eq!(
            with_else,
            "if (true) {\n    agent_hp[agent_id] = 1.0;\n} else {\n    agent_hp[agent_id] = 0.0;\n}"
        );

        let no_else = lower_cg_stmt_to_wgsl(if_no_else, &ctx).unwrap();
        assert_eq!(no_else, "if (true) {\n    agent_hp[agent_id] = 1.0;\n}");
    }

    #[test]
    fn lower_match_stmt_emits_if_chain() {
        // match hp { variant#0 { amount=local#0 } => assign(hp <- 1.0),
        //            variant#1 => assign(hp <- 0.0) }
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let arm0_assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: one,
            },
        );
        let arm1_assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: zero,
            },
        );
        let arm0_body = push_list(&mut prog, CgStmtList::new(vec![arm0_assign]));
        let arm1_body = push_list(&mut prog, CgStmtList::new(vec![arm1_assign]));
        let match_stmt = push_stmt(
            &mut prog,
            CgStmt::Match {
                scrutinee: hp,
                arms: vec![
                    CgMatchArm {
                        variant: VariantId(0),
                        bindings: vec![MatchArmBinding {
                            field_name: "amount".to_string(),
                            local: LocalId(0),
                        }],
                        body: arm0_body,
                    },
                    CgMatchArm {
                        variant: VariantId(1),
                        bindings: vec![],
                        body: arm1_body,
                    },
                ],
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let out = lower_cg_stmt_to_wgsl(match_stmt, &ctx).unwrap();
        // Scrutinee `hp` has CgExprId(0) → binding name `_scrut_0`.
        let expected = "let _scrut_0 = agent_hp[agent_id];\n\
                        if (_scrut_0.tag == VARIANT_0u) { /* bindings: amount=local_0 from _scrut_0.amount */\n\
                        \x20\x20\x20\x20agent_hp[agent_id] = 1.0;\n\
                        } else if (_scrut_0.tag == VARIANT_1u) {\n\
                        \x20\x20\x20\x20agent_hp[agent_id] = 0.0;\n\
                        }";
        assert_eq!(out, expected);
    }

    /// Non-identifier scrutinee — verify the `let _scrut_<N> = (...);`
    /// binding makes the emission valid even when the scrutinee lowers
    /// to a parenthesised expression like `(agent_hp[agent_id] + 1.0)`.
    /// Without the binding, the old shape produced
    /// `((agent_hp[agent_id] + 1.0)_tag) == ...` which is invalid WGSL.
    #[test]
    fn lower_match_with_non_identifier_scrutinee_binds_local() {
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        // Scrutinee is `hp + 1.0` — lowers to `(agent_hp[agent_id] + 1.0)`.
        let scrutinee_expr = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: hp,
                rhs: one,
                ty: CgTy::F32,
            },
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let arm_assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: zero,
            },
        );
        let arm_body = push_list(&mut prog, CgStmtList::new(vec![arm_assign]));
        let match_stmt = push_stmt(
            &mut prog,
            CgStmt::Match {
                scrutinee: scrutinee_expr,
                arms: vec![CgMatchArm {
                    variant: VariantId(0),
                    bindings: vec![],
                    body: arm_body,
                }],
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let out = lower_cg_stmt_to_wgsl(match_stmt, &ctx).unwrap();
        // scrutinee_expr is the third pushed expression → CgExprId(2).
        let expected = "let _scrut_2 = (agent_hp[agent_id] + 1.0);\n\
                        if (_scrut_2.tag == VARIANT_0u) {\n\
                        \x20\x20\x20\x20agent_hp[agent_id] = 0.0;\n\
                        }";
        assert_eq!(out, expected);
    }

    // ---- 9. Snapshot test on a non-trivial expression ----

    /// Pin the lowered string of a non-trivial expression to detect
    /// drift in any of: literal formatting, infix bracketing, builtin
    /// naming, handle naming, select arg ordering.
    #[test]
    fn snapshot_select_clamp_distance_expression() {
        // select(
        //     hp < 5.0,
        //     clamp_f32(distance(self.pos, actor.pos), 0.0, 100.0),
        //     0.0,
        // )
        let mut prog = empty_prog();
        let hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let five = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(5.0)));
        let cond = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: five,
                ty: CgTy::Bool,
            },
        );
        let pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let actor_pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Actor,
            }),
        );
        let dist = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Distance,
                args: vec![pos, actor_pos],
                ty: CgTy::F32,
            },
        );
        let zero = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let hundred = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(100.0)));
        let cl = push_expr(
            &mut prog,
            CgExpr::Builtin {
                fn_id: BuiltinId::Clamp(NumericTy::F32),
                args: vec![dist, zero, hundred],
                ty: CgTy::F32,
            },
        );
        let zero2 = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let sel = push_expr(
            &mut prog,
            CgExpr::Select {
                cond,
                then: cl,
                else_: zero2,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_expr_to_wgsl(sel, &ctx).unwrap(),
            "select(0.0, \
             clamp_f32(distance(agent_pos[agent_id], agent_pos[actor_id]), 0.0, 100.0), \
             (agent_hp[agent_id] < 5.0))"
        );
    }

    // ---- 10. Determinism ----

    /// The same program must produce the same lowered string on every
    /// invocation — no `HashMap` ordering, no float locale, no random
    /// padding.
    #[test]
    fn wgsl_emit_is_deterministic() {
        let mut prog = empty_prog();
        let pos = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let normalize = push_expr(
            &mut prog,
            CgExpr::Unary {
                op: UnaryOp::NormalizeVec3F32,
                arg: pos,
                ty: CgTy::Vec3F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let first = lower_cg_expr_to_wgsl(normalize, &ctx).unwrap();
        for _ in 0..32 {
            assert_eq!(lower_cg_expr_to_wgsl(normalize, &ctx).unwrap(), first);
        }
    }

    /// Edge-case coverage for `format_f32_lit` — pin the legacy
    /// (`emit_view::format_f32_lit`) convention's output for the values
    /// most likely to surface differences with `{:?}` / `{}` alone.
    /// A regression here breaks Phase-5 byte-for-byte parity.
    #[test]
    fn format_f32_lit_edge_cases() {
        // Integer-valued: Display gives "1", we append ".0".
        assert_eq!(format_f32_lit(1.0), "1.0");
        assert_eq!(format_f32_lit(0.0), "0.0");
        assert_eq!(format_f32_lit(-1.0), "-1.0");
        assert_eq!(format_f32_lit(100.0), "100.0");
        // Sub-unit: Display already contains '.', return as-is.
        assert_eq!(format_f32_lit(0.5), "0.5");
        assert_eq!(format_f32_lit(-0.5), "-0.5");
        assert_eq!(format_f32_lit(1.5), "1.5");
        // Very large: Display fully expands, no '.' / 'e', append ".0".
        // Well-formed sim programs do not embed literals this large, but
        // the lowering must not panic on them.
        assert_eq!(
            format_f32_lit(1e30),
            "1000000000000000000000000000000.0"
        );
        // Very small (denormal-adjacent): Display contains '.', return
        // as-is — the literal's enormous length is a known caveat for
        // pathological inputs, not for well-formed programs.
        assert!(format_f32_lit(1e-30).contains('.'));
        assert!(format_f32_lit(1e-5).starts_with("0."));
        // f32::MIN_POSITIVE — sub-normal-adjacent. Same caveat.
        assert!(format_f32_lit(f32::MIN_POSITIVE).contains('.'));
    }

    // ---- 11. Error cases ----

    #[test]
    fn dangling_expr_id_returns_out_of_range() {
        let prog = empty_prog();
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_expr_to_wgsl(CgExprId(0), &ctx).unwrap_err();
        assert_eq!(
            err,
            EmitError::ExprIdOutOfRange {
                id: CgExprId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn dangling_stmt_id_returns_out_of_range() {
        let prog = empty_prog();
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_stmt_to_wgsl(CgStmtId(0), &ctx).unwrap_err();
        assert_eq!(
            err,
            EmitError::StmtIdOutOfRange {
                id: CgStmtId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn dangling_stmt_list_id_returns_out_of_range() {
        let prog = empty_prog();
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_stmt_list_to_wgsl(CgStmtListId(3), &ctx).unwrap_err();
        assert_eq!(
            err,
            EmitError::StmtListIdOutOfRange {
                id: CgStmtListId(3),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn nested_dangling_expr_inside_stmt_propagates() {
        // assign(hp <- expr#9) where expr#9 doesn't exist.
        let mut prog = empty_prog();
        let s = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(9),
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_stmt_to_wgsl(s, &ctx).unwrap_err();
        match err {
            EmitError::ExprIdOutOfRange { id, .. } => assert_eq!(id, CgExprId(9)),
            other => panic!("expected ExprIdOutOfRange, got {other:?}"),
        }
    }

    // ---- 12. Display impl on EmitError ----

    #[test]
    fn emit_error_display_each_variant() {
        let e1 = EmitError::ExprIdOutOfRange {
            id: CgExprId(7),
            arena_len: 3,
        };
        assert_eq!(
            format!("{}", e1),
            "CgExprId(#7) out of range (expr arena holds 3 entries)"
        );
        let e2 = EmitError::StmtIdOutOfRange {
            id: CgStmtId(1),
            arena_len: 0,
        };
        assert_eq!(
            format!("{}", e2),
            "CgStmtId(#1) out of range (stmt arena holds 0 entries)"
        );
        let e3 = EmitError::StmtListIdOutOfRange {
            id: CgStmtListId(4),
            arena_len: 2,
        };
        assert_eq!(
            format!("{}", e3),
            "CgStmtListId(#4) out of range (stmt-list arena holds 2 entries)"
        );
        let e4 = EmitError::UnsupportedHandle {
            handle: DataHandle::ScoringOutput,
            reason: "no slot",
        };
        assert_eq!(
            format!("{}", e4),
            "unsupported handle scoring.output: no slot"
        );
    }

    // ---- 13. Statement-list joining ----

    #[test]
    fn stmt_list_emits_newline_joined() {
        let mut prog = empty_prog();
        let one = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let two = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.0)));
        let s0 = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: one,
            },
        );
        let s1 = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::ShieldHp,
                    target: AgentRef::Self_,
                },
                value: two,
            },
        );
        let list = push_list(&mut prog, CgStmtList::new(vec![s0, s1]));
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(
            lower_cg_stmt_list_to_wgsl(list, &ctx).unwrap(),
            "agent_hp[agent_id] = 1.0;\nagent_shield_hp[agent_id] = 2.0;"
        );
    }

    #[test]
    fn stmt_list_empty_emits_empty_string() {
        let mut prog = empty_prog();
        let list = push_list(&mut prog, CgStmtList::new(vec![]));
        let ctx = EmitCtx::structural(&prog);
        assert_eq!(lower_cg_stmt_list_to_wgsl(list, &ctx).unwrap(), "");
    }

    // ---- 14. cg_ty_to_wgsl spot-check ----

    #[test]
    fn cg_ty_to_wgsl_each_variant() {
        assert_eq!(cg_ty_to_wgsl(CgTy::Bool), "bool");
        assert_eq!(cg_ty_to_wgsl(CgTy::U32), "u32");
        assert_eq!(cg_ty_to_wgsl(CgTy::I32), "i32");
        assert_eq!(cg_ty_to_wgsl(CgTy::F32), "f32");
        assert_eq!(cg_ty_to_wgsl(CgTy::Vec3F32), "vec3<f32>");
        assert_eq!(cg_ty_to_wgsl(CgTy::AgentId), "u32");
        assert_eq!(cg_ty_to_wgsl(CgTy::Tick), "u32");
        assert_eq!(cg_ty_to_wgsl(CgTy::ViewKey { view: ViewId(2) }), "u32");
    }

    // ---- Task 1 (CG Lowering Gap Closure): EventField emit ----

    /// `CgExpr::EventField` produces a schema-driven access expression.
    /// With the today-default layout (stride=11, header=2,
    /// buffer="event_ring") and a `target` field at payload offset 1
    /// typed as `AgentId`, the WGSL renders to
    /// `event_ring[event_idx * 11u + 3u]`.
    #[test]
    fn event_field_emits_schema_driven_wgsl_access_for_agent_id() {
        use crate::cg::program::{EventLayout, FieldLayout};
        let mut prog = empty_prog();
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "target".to_string(),
            FieldLayout {
                word_offset_in_payload: 1,
                word_count: 1,
                ty: CgTy::AgentId,
            },
        );
        prog.event_layouts.insert(
            7,
            EventLayout {
                record_stride_u32: 11,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields,
            },
        );

        let id = push_expr(
            &mut prog,
            CgExpr::EventField {
                event_kind: EventKindId(7),
                word_offset_in_payload: 1,
                ty: CgTy::AgentId,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_expr_to_wgsl(id, &ctx).expect("EventField lowers");
        assert_eq!(wgsl, "event_ring[event_idx * 11u + 3u]");
    }

    /// F32-typed `EventField` emits a `bitcast<f32>` access. The
    /// payload word is u32 on the GPU side; `bitcast<f32>` reinterprets
    /// the bit pattern as the typed float — same shape `pack_event`
    /// writes via `f32::to_bits` on the CPU.
    #[test]
    fn event_field_emits_bitcast_for_f32() {
        use crate::cg::program::{EventLayout, FieldLayout};
        let mut prog = empty_prog();
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "amount".to_string(),
            FieldLayout {
                word_offset_in_payload: 2,
                word_count: 1,
                ty: CgTy::F32,
            },
        );
        prog.event_layouts.insert(
            3,
            EventLayout {
                record_stride_u32: 11,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields,
            },
        );

        let id = push_expr(
            &mut prog,
            CgExpr::EventField {
                event_kind: EventKindId(3),
                word_offset_in_payload: 2,
                ty: CgTy::F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_expr_to_wgsl(id, &ctx).expect("EventField F32 lowers");
        assert_eq!(wgsl, "bitcast<f32>(event_ring[event_idx * 11u + 4u])");
    }

    /// An `EventField` whose `event_kind` has no entry in
    /// `prog.event_layouts` surfaces as
    /// `EmitError::UnregisteredEventKind`.
    #[test]
    fn event_field_unregistered_kind_surfaces_typed_error() {
        let mut prog = empty_prog();
        let id = push_expr(
            &mut prog,
            CgExpr::EventField {
                event_kind: EventKindId(99),
                word_offset_in_payload: 0,
                ty: CgTy::AgentId,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_expr_to_wgsl(id, &ctx).expect_err("missing layout fails");
        match err {
            EmitError::UnregisteredEventKind { kind } => assert_eq!(kind, EventKindId(99)),
            other => panic!("expected UnregisteredEventKind, got {other:?}"),
        }
    }

    /// `Vec3F32`-typed `EventField` emits a 3-element `vec3<f32>(...)`
    /// constructor with three independent `bitcast<f32>` reads at
    /// `total_offset`, `total_offset+1`, `total_offset+2`. With
    /// `header_word_count=2` and a Vec3F32 field at
    /// `word_offset_in_payload=4` (stride=11), the first base is
    /// `2 + 4 = 6`; the three accesses land at offsets `6`, `7`, `8`.
    /// This is the most error-prone CgTy arm because the format
    /// string carries `o2 = total_offset + 1` / `o3 = total_offset + 2`
    /// arithmetic — locking the exact emitted form catches any future
    /// drift in the offset arithmetic.
    #[test]
    fn event_field_emits_vec3f32_triple_bitcast() {
        use crate::cg::program::{EventLayout, FieldLayout};
        let mut prog = empty_prog();
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "pos".to_string(),
            FieldLayout {
                word_offset_in_payload: 4,
                word_count: 3,
                ty: CgTy::Vec3F32,
            },
        );
        prog.event_layouts.insert(
            5,
            EventLayout {
                record_stride_u32: 11,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields,
            },
        );

        let id = push_expr(
            &mut prog,
            CgExpr::EventField {
                event_kind: EventKindId(5),
                word_offset_in_payload: 4,
                ty: CgTy::Vec3F32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_expr_to_wgsl(id, &ctx).expect("EventField Vec3F32 lowers");
        assert_eq!(
            wgsl,
            "vec3<f32>(bitcast<f32>(event_ring[event_idx * 11u + 6u]), bitcast<f32>(event_ring[event_idx * 11u + 7u]), bitcast<f32>(event_ring[event_idx * 11u + 8u]))"
        );
    }

    /// `Bool`-typed `EventField` emits a `(... != 0u)` predicate form.
    /// The payload word is u32 on the GPU side; non-zero u32 reads as
    /// `true`. With `header_word_count=2` and a Bool field at
    /// `word_offset_in_payload=0` (stride=11), the read lands at offset
    /// `2`, producing `(event_ring[event_idx * 11u + 2u] != 0u)`.
    #[test]
    fn event_field_emits_bool_predicate_form() {
        use crate::cg::program::{EventLayout, FieldLayout};
        let mut prog = empty_prog();
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "flag".to_string(),
            FieldLayout {
                word_offset_in_payload: 0,
                word_count: 1,
                ty: CgTy::Bool,
            },
        );
        prog.event_layouts.insert(
            6,
            EventLayout {
                record_stride_u32: 11,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields,
            },
        );

        let id = push_expr(
            &mut prog,
            CgExpr::EventField {
                event_kind: EventKindId(6),
                word_offset_in_payload: 0,
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_expr_to_wgsl(id, &ctx).expect("EventField Bool lowers");
        assert_eq!(wgsl, "(event_ring[event_idx * 11u + 2u] != 0u)");
    }

    /// `I32`-typed `EventField` emits a `bitcast<i32>` access. The
    /// payload word is u32 on the GPU side; `bitcast<i32>` reinterprets
    /// the bit pattern as the typed signed int — same shape
    /// `pack_event` writes via `i32::to_ne_bytes`-style reinterpretation
    /// on the CPU. With `header_word_count=2` and an I32 field at
    /// `word_offset_in_payload=3` (stride=11), the read lands at offset
    /// `5`.
    #[test]
    fn event_field_emits_i32_signed_cast() {
        use crate::cg::program::{EventLayout, FieldLayout};
        let mut prog = empty_prog();
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "delta".to_string(),
            FieldLayout {
                word_offset_in_payload: 3,
                word_count: 1,
                ty: CgTy::I32,
            },
        );
        prog.event_layouts.insert(
            8,
            EventLayout {
                record_stride_u32: 11,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields,
            },
        );

        let id = push_expr(
            &mut prog,
            CgExpr::EventField {
                event_kind: EventKindId(8),
                word_offset_in_payload: 3,
                ty: CgTy::I32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_expr_to_wgsl(id, &ctx).expect("EventField I32 lowers");
        assert_eq!(wgsl, "bitcast<i32>(event_ring[event_idx * 11u + 5u])");
    }

    // ---- Task 4 (CG Lowering Gap Closure): NamespaceCall / NamespaceField emit ----

    /// `CgExpr::NamespaceCall` emits a function call to the registry-
    /// resolved `wgsl_fn_name` with each argument lowered in source
    /// order. The kernel composer prepends a B1-stub prelude function
    /// for the `(ns, method)` reference; the body itself is just the
    /// call-form.
    #[test]
    fn namespace_call_emits_wgsl_fn_call_via_registry() {
        use crate::cg::program::{MethodDef, NamespaceDef};
        let mut prog = empty_prog();
        let mut agents = NamespaceDef {
            name: "agents".to_string(),
            ..NamespaceDef::default()
        };
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
        prog.namespace_registry
            .namespaces
            .insert(dsl_ast::ir::NamespaceId::Agents, agents);

        let arg_a = push_expr(&mut prog, CgExpr::AgentSelfId);
        let arg_b = push_expr(&mut prog, CgExpr::PerPairCandidateId);
        let id = push_expr(
            &mut prog,
            CgExpr::NamespaceCall {
                ns: dsl_ast::ir::NamespaceId::Agents,
                method: "is_hostile_to".to_string(),
                args: vec![arg_a, arg_b],
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_expr_to_wgsl(id, &ctx).expect("NamespaceCall lowers");
        assert_eq!(wgsl, "agents_is_hostile_to(agent_id, per_pair_candidate)");
    }

    /// `CgExpr::NamespaceField` with a `PreambleLocal` access form
    /// emits the bare local identifier (`tick` for `world.tick`). The
    /// kernel composer is responsible for binding the local in the
    /// preamble (`let tick = cfg.tick;`); this emit just names it.
    #[test]
    fn namespace_field_preamble_local_emits_bare_identifier() {
        use crate::cg::program::{FieldDef, NamespaceDef, WgslAccessForm};
        let mut prog = empty_prog();
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
        prog.namespace_registry
            .namespaces
            .insert(dsl_ast::ir::NamespaceId::World, world);

        let id = push_expr(
            &mut prog,
            CgExpr::NamespaceField {
                ns: dsl_ast::ir::NamespaceId::World,
                field: "tick".to_string(),
                ty: CgTy::U32,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_expr_to_wgsl(id, &ctx).expect("NamespaceField lowers");
        assert_eq!(wgsl, "tick");
    }

    /// A `NamespaceCall` with no registry entry surfaces as
    /// `EmitError::UnregisteredNamespaceMethod`.
    #[test]
    fn namespace_call_unregistered_method_surfaces_typed_error() {
        let mut prog = empty_prog();
        let id = push_expr(
            &mut prog,
            CgExpr::NamespaceCall {
                ns: dsl_ast::ir::NamespaceId::Agents,
                method: "missing_method".to_string(),
                args: vec![],
                ty: CgTy::Bool,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let err = lower_cg_expr_to_wgsl(id, &ctx).expect_err("missing method fails");
        match err {
            EmitError::UnregisteredNamespaceMethod { ns, method } => {
                assert_eq!(ns, dsl_ast::ir::NamespaceId::Agents);
                assert_eq!(method, "missing_method");
            }
            other => panic!("expected UnregisteredNamespaceMethod, got {other:?}"),
        }
    }

    /// `CgStmt::Emit` lowers to a real ring-append: atomicAdd a slot
    /// off `event_tail`, then write the tag + tick + payload words to
    /// `event_ring[slot * stride + offset]`. Replaces the prior B1
    /// phony-discard placeholder. The (event_kind, field index) lookup
    /// resolves through `EventLayout::fields_in_declaration_order`.
    ///
    /// This test pins the WGSL shape directly via the per-stmt emit
    /// path, independent of the kernel-binding generator (which still
    /// needs to declare both `event_ring: array<u32>` and
    /// `event_tail: atomic<u32>` for non-test compilation; tracked
    /// separately).
    #[test]
    fn emit_lowers_to_ring_append_with_atomic_tail() {
        use crate::cg::op::EventKindId;
        use crate::cg::program::{EventLayout, FieldLayout};
        use crate::cg::stmt::{CgStmt, EventField};

        // Killed { by: AgentId, prey: AgentId, pos: Vec3F32 } — same
        // shape predator_prey_min.sim's Killed declares.
        let mut prog = empty_prog();
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "by".to_string(),
            FieldLayout {
                word_offset_in_payload: 0,
                word_count: 1,
                ty: CgTy::AgentId,
            },
        );
        fields.insert(
            "prey".to_string(),
            FieldLayout {
                word_offset_in_payload: 1,
                word_count: 1,
                ty: CgTy::AgentId,
            },
        );
        fields.insert(
            "pos".to_string(),
            FieldLayout {
                word_offset_in_payload: 2,
                word_count: 3,
                ty: CgTy::Vec3F32,
            },
        );
        prog.event_layouts.insert(
            1,
            EventLayout {
                record_stride_u32: 11,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields,
            },
        );

        let by_value = push_expr(&mut prog, CgExpr::AgentSelfId);
        let prey_value = push_expr(&mut prog, CgExpr::AgentSelfId);
        let pos_value = push_expr(
            &mut prog,
            CgExpr::Lit(LitValue::Vec3F32 { x: 1.0, y: 2.0, z: 3.0 }),
        );
        let stmt = CgStmt::Emit {
            event: EventKindId(1),
            fields: vec![
                (EventField { event: EventKindId(1), index: 0 }, by_value),
                (EventField { event: EventKindId(1), index: 1 }, prey_value),
                (EventField { event: EventKindId(1), index: 2 }, pos_value),
            ],
        };
        let stmt_id = push_stmt(&mut prog, stmt);
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_stmt_to_wgsl(stmt_id, &ctx).expect("Emit lowers");

        // Atomic-add the slot off event_tail[0].
        assert!(
            wgsl.contains("let slot = atomicAdd(&event_tail[0], 1u);"),
            "expected atomicAdd-to-tail; got:\n{wgsl}"
        );
        // Bounds check before commit.
        assert!(
            wgsl.contains("if (slot < 1048576u)"),
            "expected slot bounds check; got:\n{wgsl}"
        );
        // Tag write at offset 0 (event_id is 1) via atomicStore.
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 0u], 1u);"),
            "expected tag write at offset 0; got:\n{wgsl}"
        );
        // Tick write at offset 1.
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 1u], tick);"),
            "expected tick write at offset 1; got:\n{wgsl}"
        );
        // by AgentId at payload offset 0 (header+0 = 2).
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 2u], (agent_id));"),
            "expected `by` at offset 2; got:\n{wgsl}"
        );
        // prey AgentId at payload offset 1 (header+1 = 3).
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 3u], (agent_id));"),
            "expected `prey` at offset 3; got:\n{wgsl}"
        );
        // Vec3 pos with bitcast<u32>(.x/.y/.z) at offsets 4/5/6.
        assert!(
            wgsl.contains("bitcast<u32>(_emit_v_1_2.x)"),
            "expected vec3 .x bitcast; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 4u], bitcast<u32>(_emit_v_1_2.x));"),
            "expected vec3 .x at offset 4; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 11u + 6u], bitcast<u32>(_emit_v_1_2.z));"),
            "expected vec3 .z at offset 6; got:\n{wgsl}"
        );
        // No phony discard left over from the old B1 placeholder.
        assert!(
            !wgsl.contains("_ = ("),
            "phony discard should be gone; got:\n{wgsl}"
        );
    }

    // ---- Cross-agent target reads via stmt-scope let hoisting ----
    //
    // Slice 1 (2026-05-03 "stdlib into CG IR" plan) replaces the prior
    // B1 typed-default fallback for `Read(AgentField{Target(_)})` with
    // a real `let target_expr_<N>: u32 = …;` pre-binding emitted at
    // stmt scope, so `agents.pos(other)` becomes `agent_pos[
    // target_expr_<N>]` paired with a hoisted let declaring the index.
    // These tests lock the behavior so a later refactor can't silently
    // re-introduce a placeholder.

    /// `Read(AgentField{Pos, Target(some_lit_id)})` lowered as the
    /// value of an `Assign { target: AgentField{Pos, Self_}, … }`
    /// stmt produces:
    /// ```text
    /// let target_expr_0: u32 = 11u;
    /// agent_pos[agent_id] = agent_pos[target_expr_0];
    /// ```
    /// The pre-binding is the slice 1 fix; without it the body
    /// returns `vec3<f32>(0.0)` (the B1 placeholder).
    #[test]
    fn target_read_emits_stmt_scope_let_binding() {
        let mut prog = empty_prog();
        // Target expression: a literal AgentId(11) stand-in for a
        // computed cross-agent reference (in real DSL this would be
        // `agents.engaged_with_or(self, fallback)` etc.).
        let target_id_expr = push_expr(&mut prog, CgExpr::Lit(LitValue::AgentId(11)));
        // RHS: `agents.pos(target)` — Read of AgentField{Pos,
        // Target(target_id_expr)}.
        let rhs = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Target(target_id_expr),
            }),
        );
        // LHS: `self.pos = …` (Assign target Pos on Self_).
        let assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Pos,
                    target: AgentRef::Self_,
                },
                value: rhs,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_stmt_to_wgsl(assign, &ctx).expect("stmt lowers");
        // Pre-binding for the target expression — emitted at stmt
        // scope so the indexed access has a declared identifier.
        assert!(
            wgsl.contains("let target_expr_0: u32 = 11u;"),
            "expected pre-stmt let binding; got:\n{wgsl}"
        );
        // Indexed access on the SoA, NOT the old B1 default.
        assert!(
            wgsl.contains("agent_pos[target_expr_0]"),
            "expected indexed access; got:\n{wgsl}"
        );
        assert!(
            !wgsl.contains("vec3<f32>(0.0)"),
            "B1 typed-default placeholder must not appear; got:\n{wgsl}"
        );
    }

    /// Two reads of the same target expression within one stmt
    /// (`Pos` and `Vel` both on `Target(N)`) emit a single
    /// `let target_expr_<N>` pre-binding, not two. Validates the
    /// `bound_target_exprs` dedup on first reference.
    #[test]
    fn duplicate_target_reads_share_one_let_binding() {
        let mut prog = empty_prog();
        let target_id_expr = push_expr(&mut prog, CgExpr::Lit(LitValue::AgentId(7)));
        // Read pos and vel on the same Target(target_id_expr).
        let pos_read = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Target(target_id_expr),
            }),
        );
        let vel_read = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Vel,
                target: AgentRef::Target(target_id_expr),
            }),
        );
        // Compose: `self.pos = pos_read + vel_read` so both reads
        // appear in one stmt's expression sub-tree.
        let sum = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: BinaryOp::AddVec3,
                lhs: pos_read,
                rhs: vel_read,
                ty: CgTy::Vec3F32,
            },
        );
        let assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Pos,
                    target: AgentRef::Self_,
                },
                value: sum,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_stmt_to_wgsl(assign, &ctx).expect("stmt lowers");
        // Exactly one let-binding for target_expr_0.
        let count = wgsl.matches("let target_expr_0: u32 =").count();
        assert_eq!(
            count, 1,
            "expected one let binding for the shared target expr; got {count}:\n{wgsl}"
        );
        // Both indexed accesses present.
        assert!(
            wgsl.contains("agent_pos[target_expr_0]"),
            "expected agent_pos indexed access; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("agent_vel[target_expr_0]"),
            "expected agent_vel indexed access; got:\n{wgsl}"
        );
    }

    /// `Assign { target: AgentField{Pos, Target(N)}, value }`
    /// (`agents.set_pos(other, …)`) emits the same pre-binding +
    /// indexed write, replacing the prior phony `_ = (…);` discard.
    #[test]
    fn for_each_neighbor_body_emits_per_candidate_walk_with_inner_emit() {
        // Body-form spatial walk: empty stmt body smoke-test pinning
        // the per-candidate cell-walk scaffold. Slice 2b of the
        // stdlib-into-CG-IR plan. The emitted WGSL must contain:
        //
        // - The 4-deep loop chain: 3 cell-axis iterators (`dz/dy/dx`)
        //   plus the inner per-candidate loop bound by
        //   `spatial_grid_starts[_cell..+1]`.
        // - The `let per_pair_candidate = spatial_grid_cells[_i];`
        //   binding — the pair-bound emit convention's slot id.
        let mut prog = empty_prog();
        // Empty inner body; the test focuses on the scaffold.
        let inner_list = push_list(&mut prog, CgStmtList::new(vec![]));
        // Origin = AgentSelfId (lowers to WGSL `agent_id`), matching the
        // legacy `spatial.<...>(self)` shape — exercises backward-compat
        // emit after Gap dungeon_horde#1.
        let origin = push_expr(&mut prog, CgExpr::AgentSelfId);
        let body_stmt = push_stmt(
            &mut prog,
            CgStmt::ForEachNeighborBody {
                binder: LocalId(7),
                body: inner_list,
                radius_cells: 1,
                origin,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_stmt_to_wgsl(body_stmt, &ctx).expect("body-form spatial walk lowers");
        assert!(
            wgsl.contains("let per_pair_candidate = spatial_grid_cells[_i];"),
            "expected per-candidate slot binding; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("for (var dz: i32 = -1; dz <= 1; dz = dz + 1)"),
            "expected the cell-walk z-axis loop; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("let _start = spatial_grid_starts[_cell];"),
            "expected the cell-slice start binding; got:\n{wgsl}"
        );
    }

    #[test]
    fn target_assign_emits_indexed_write_not_phony_discard() {
        let mut prog = empty_prog();
        let target_id_expr = push_expr(&mut prog, CgExpr::Lit(LitValue::AgentId(3)));
        let rhs = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }),
        );
        let assign = push_stmt(
            &mut prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Pos,
                    target: AgentRef::Target(target_id_expr),
                },
                value: rhs,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_stmt_to_wgsl(assign, &ctx).expect("stmt lowers");
        assert!(
            wgsl.contains("let target_expr_0: u32 = 3u;"),
            "expected pre-stmt let; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("agent_pos[target_expr_0] = agent_pos[agent_id];"),
            "expected indexed write; got:\n{wgsl}"
        );
        assert!(
            !wgsl.contains("_ = ("),
            "phony discard from the old placeholder must not appear; got:\n{wgsl}"
        );
    }

    // ---- #136 slice β step 2: apply_ability dispatcher emit ----

    #[test]
    fn apply_ability_emits_dispatcher_loop_with_branch_arms() {
        // Build a minimal program with one ApplyAbility stmt that
        // reads a literal AbilityId(1) — the simplest possible
        // operand. Emit should produce the slot/base/loop scaffold,
        // the EFFECT_KIND_EMPTY skip, the four implemented arms
        // (Damage=0, Heal=1, Stun=3, Slow=4), and the chronicle-
        // append TODO markers.
        let mut prog = empty_prog();
        let ability_lit = push_expr(
            &mut prog,
            CgExpr::Lit(LitValue::U32(1)),
        );
        // Slice δ (#161): caster operand is now part of the stmt.
        // Use AgentSelfId — the per-thread agent in PerAgent kernel
        // shape. Assertion below pins the resulting `caster_slot`
        // identifier emit.
        let caster_self = push_expr(&mut prog, CgExpr::AgentSelfId);
        // Slice ε part 1: target operand. Use the same caster_self
        // expression so the test pins the slice-γ self-cast default
        // (target = caster when source omits explicit `target`).
        let target_self = caster_self;
        let stmt_id = push_stmt(
            &mut prog,
            CgStmt::ApplyAbility {
                ability: ability_lit,
                caster: caster_self,
                target: target_self,
                with_aoe_dispatch: false,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_stmt_to_wgsl(stmt_id, &ctx).expect("lower");

        // Operand expression — read the lit and coerce to u32.
        assert!(wgsl.contains("ability_id__u32: u32 = u32(1u)"),
            "operand should be u32-coerced from the lit;\n{wgsl}");

        // Slot/base computation.
        assert!(wgsl.contains("ability_slot: u32 = ability_id__u32 - 1u"),
            "AbilityId is 1-based; slot index is id - 1;\n{wgsl}");
        assert!(wgsl.contains("ability_slot * 6u"),
            "stride MAX_EFFECTS_PER_PROGRAM = 6 must be inlined;\n{wgsl}");

        // Loop bound + sentinel skip.
        assert!(wgsl.contains("for (var i: u32 = 0u; i < 6u"),
            "loop walks every slot;\n{wgsl}");
        assert!(wgsl.contains("if (kind == 0xFFu) { continue; }"),
            "EFFECT_KIND_EMPTY must early-out via continue;\n{wgsl}");

        // SoA reads via the new BindingMetadata.
        assert!(wgsl.contains("ability_registry_effect_kinds[effect_base + i]"),
            "kind read must hit the new column binding;\n{wgsl}");
        assert!(wgsl.contains("ability_registry_effect_payload_a[effect_base + i]"),
            "payload_a read must hit the new column binding;\n{wgsl}");
        assert!(wgsl.contains("ability_registry_effect_payload_b[effect_base + i]"),
            "payload_b read must hit the new column binding;\n{wgsl}");

        // Thirty-one implemented variant arms — every EffectOp
        // variant except `CastAbility` (= 7), which needs a
        // recursive dispatch shape (deferred to slice δ).
        for (kind, label) in &[
            (0,  "Damage"),
            (1,  "Heal"),
            (2,  "Shield"),
            (3,  "Stun"),
            (4,  "Slow"),
            (5,  "TransferGold"),
            (6,  "ModifyStanding"),
            (8,  "Root"),
            (9,  "Silence"),
            (10, "Fear"),
            (11, "Taunt"),
            (12, "Dash"),
            (13, "Blink"),
            (14, "Knockback"),
            (15, "Pull"),
            (16, "Execute"),
            (17, "SelfDamage"),
            (18, "LifeSteal"),
            (19, "DamageModify"),
            (20, "DamageOverTime"),
            (21, "HealOverTime"),
            (22, "TimedShield"),
            (23, "Buff"),
            (24, "Summon"),
            (25, "Harvest"),
            (26, "PlaceVoxel"),
            (27, "Stealth"),
            (28, "Charm"),
            (29, "Grounded"),
            (30, "Suppress"),
            (31, "Reflect"),
            // Wave 3 ToM Phase 1 — `plant_belief` bit-flag primitive.
            (32, "PlantBelief"),
            // Wave 3 ToM Phase 3 — `observe` self-observe-target verb.
            (33, "Observe"),
            // Wave 3 ToM Phase 3.5 — `scry` cross-observer access verb.
            (34, "Scry"),
            // Wave 3 ToM Phase 3.5 — `reveal` one-to-many propagation verb.
            (35, "Reveal"),
            // Wave 3 ToM Phase 4 — deception verbs.
            (36, "Disguise"),
            (37, "Decoy"),
            (38, "EraseBelief"),
        ] {
            let kind_token = if *kind == 0 {
                format!("if (kind == {kind}u)")
            } else {
                format!("else if (kind == {kind}u)")
            };
            assert!(
                wgsl.contains(&kind_token),
                "{label} arm (discriminant {kind}u);\n{wgsl}"
            );
        }

        // f32-bitcast payload count: Damage / Heal / Shield (3) +
        // Execute / SelfDamage (2) + DoT / HoT / TimedShield (3) +
        // 4 movement verbs (4) = 12 arms total bitcast payload_a.
        assert!(wgsl.matches("bitcast<f32>(payload_a)").count() >= 12,
            "12 amount/distance variants must bitcast payload_a to f32;\n{wgsl}");

        // Summon decoders. Slice γ tail wired the Buff arm with raw
        // payload_a / payload_b stores — no WGSL-side decode now (the
        // chronicle records the packed payload verbatim and consumers
        // decode on read). Summon (kind 24 → 62, slice γ closer)
        // KEEPS the WGSL-side decode because the dispatcher needs to
        // split the packed `payload_b` into distinct ring slots
        // (slot 4 = count, slot 5 = lifetime_ticks) so consumers
        // don't have to redo the bit-unpack on read — the engine
        // event struct carries `count: u8` and `lifetime_ticks: u32`
        // as separate fields.
        assert!(wgsl.contains("(payload_b >> 24u) & 0xFFu"),
            "Summon count extracted from payload_b high byte;\n{wgsl}");
        assert!(wgsl.contains("payload_b & 0x00FFFFFFu"),
            "Summon lifetime extracted from payload_b low 24 bits;\n{wgsl}");

        // chronicle_append TODO markers — one per implemented arm
        // that hasn't yet been wired to a real chronicle write.
        // **Removed when wired** (slice γ — self-cast assumption):
        //   - chronicle_append_damage          → EffectDamageApplied
        //   - chronicle_append_heal            → EffectHealApplied
        //   - chronicle_append_shield          → EffectShieldApplied
        //   - chronicle_append_stun            → EffectStunApplied
        //   - chronicle_append_slow            → EffectSlowApplied
        //   - chronicle_append_transfer_gold   → EffectGoldTransfer
        //   - chronicle_append_modify_standing → EffectStandingDelta
        //   - chronicle_append_self_damage     → EffectSelfDamageApplied
        //                                        (Bleed verb swap, Task #138 follow-on)
        //   - chronicle_append_life_steal      → EffectLifeStealApplied
        //                                        (Vampirize verb swap, Task #138 follow-on)
        //   - chronicle_append_damage_modify   → EffectDamageModifyApplied
        //                                        (Fortify verb swap, Task #138 follow-on)
        //   - chronicle_append_execute         → EffectExecuteApplied
        //                                        (Reap verb swap, Task #138 follow-on)
        // Below-list arms keep their TODO markers because the runtime
        // has no 1:1 chronicle counterpart (Root / Silence / Fear /
        // Taunt / movement verbs / etc.) — slice δ scope or a future
        // engine event-kind extension.
        // Wave 2 piece 1 — Root/Silence/Fear/Taunt are now wired (kinds
        // 43/44/45/46), no longer carry TODO markers; see the explicit
        // assertions below.
        // Wave 2 piece 2 — Dash/Blink/Knockback/Pull are now wired (kinds
        // 47/48/49/50), no longer carry TODO markers; see the explicit
        // assertions below.
        // Wave 1.5+ — DamageOverTime/HealOverTime/TimedShield are now
        // wired (kinds 51/52/53), no longer carry TODO markers; see the
        // explicit assertions below.
        // Extended-status slice — Stealth/Charm/Grounded/Suppress are
        // now wired (kinds 54/55/56/57), no longer carry TODO markers;
        // see the explicit assertions below.
        // Slice γ tail — Buff/Harvest/PlaceVoxel/Reflect are now wired
        // (kinds 58/59/60/61), no longer carry TODO markers; see the
        // explicit assertions below.
        // Slice γ closer — Summon (kind 24 → 62) is now wired too;
        // the dispatcher arm body decodes packed (count, lifetime) and
        // writes them into distinct ring slots (slot 4 = count, slot
        // 5 = lifetime_ticks). NO `// TODO slice γ` arms remain in
        // the apply_ability dispatcher — the slice is closed.
        assert!(
            !wgsl.contains("TODO slice γ:"),
            "all `// TODO slice γ` placeholders should be wired; slice γ \
             closer (Summon) was the last one;\n{wgsl}"
        );

        // Wave 2 piece 1 — control-status arms now write real chronicle
        // records (kinds 43/44/45/46). Pin the kind tags so a regression
        // that drops the wire-up surfaces here.
        for (kind_token, expected_event_id, name) in &[
            ("kind == 8u",  43u32, "Root"),
            ("kind == 9u",  44u32, "Silence"),
            ("kind == 10u", 45u32, "Fear"),
            ("kind == 11u", 46u32, "Taunt"),
        ] {
            assert!(
                !wgsl.contains(&format!(
                    "TODO slice γ: chronicle_append_{}",
                    name.to_lowercase()
                )),
                "{name} arm should no longer carry the TODO marker;\n{wgsl}"
            );
            assert!(
                wgsl.contains(kind_token),
                "{name} arm dispatch ({kind_token}) must be present;\n{wgsl}"
            );
            assert!(
                wgsl.contains(&format!(
                    "atomicStore(&event_ring[_slot * 11u + 0u], {expected_event_id}u);"
                )),
                "{name} arm must store kind={expected_event_id};\n{wgsl}"
            );
        }

        // Wave 2 piece 2 — movement EffectOps now write real chronicle
        // records (kinds 47/48/49/50). Dash/Blink are caster-self motion
        // (no target slot in the engine event); Knockback/Pull are
        // forced motion on a target. Pin the kind tags so a regression
        // that drops the wire-up surfaces here.
        for (kind_token, expected_event_id, name) in &[
            ("kind == 12u", 47u32, "Dash"),
            ("kind == 13u", 48u32, "Blink"),
            ("kind == 14u", 49u32, "Knockback"),
            ("kind == 15u", 50u32, "Pull"),
        ] {
            assert!(
                !wgsl.contains(&format!(
                    "TODO slice γ: chronicle_append_{}",
                    name.to_lowercase()
                )),
                "{name} arm should no longer carry the TODO marker;\n{wgsl}"
            );
            assert!(
                wgsl.contains(kind_token),
                "{name} arm dispatch ({kind_token}) must be present;\n{wgsl}"
            );
            assert!(
                wgsl.contains(&format!(
                    "atomicStore(&event_ring[_slot * 11u + 0u], {expected_event_id}u);"
                )),
                "{name} arm must store kind={expected_event_id};\n{wgsl}"
            );
        }

        // Wave 2 piece 2 — Knockback/Pull store distance at payload
        // word 2 (= ring slot offset 4), same shape as Damage/Heal/
        // Shield. Dash/Blink store distance at payload word 1 (= ring
        // slot offset 3) since the engine event has no target field.
        // Pin both shapes so a regression that swaps them surfaces here.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 3u], bitcast<u32>(distance));"),
            "Dash/Blink arms must store distance at payload word 1 (ring offset 3);\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(distance));"),
            "Knockback/Pull arms must store distance at payload word 2 (ring offset 4);\n{wgsl}"
        );

        // Wave 1.5+ — multi-tick effects (DoT/HoT/TimedShield) now write
        // real chronicle records (kinds 51/52/53). All three share the
        // same 5-payload-word shape: actor + target + amount (bitcast
        // f32 → u32) at payload word 2 (ring slot offset 4) +
        // duration_ticks (raw u32) at payload word 3 (ring slot offset
        // 5). Pin per-variant kind tags + the duration write so a
        // regression that drops the duration surfaces here.
        for (kind_token, expected_event_id, name) in &[
            ("kind == 20u", 51u32, "DamageOverTime"),
            ("kind == 21u", 52u32, "HealOverTime"),
            ("kind == 22u", 53u32, "TimedShield"),
        ] {
            // The TODO markers used Rust snake_case (e.g.
            // chronicle_append_damage_over_time). Since the shorthand
            // form would be ambiguous (DamageOverTime → damage_over_time),
            // we hard-code the snake_case form per name.
            let snake = match *name {
                "DamageOverTime" => "damage_over_time",
                "HealOverTime"   => "heal_over_time",
                "TimedShield"    => "timed_shield",
                _ => unreachable!(),
            };
            assert!(
                !wgsl.contains(&format!("TODO slice γ: chronicle_append_{snake}")),
                "{name} arm should no longer carry the TODO marker;\n{wgsl}"
            );
            assert!(
                wgsl.contains(kind_token),
                "{name} arm dispatch ({kind_token}) must be present;\n{wgsl}"
            );
            assert!(
                wgsl.contains(&format!(
                    "atomicStore(&event_ring[_slot * 11u + 0u], {expected_event_id}u);"
                )),
                "{name} arm must store kind={expected_event_id};\n{wgsl}"
            );
        }
        // DoT/HoT/TimedShield: amount at slot 4 (bitcast<u32>(amount)),
        // duration_ticks at slot 5 (raw u32 from payload_b). Pin the
        // duration write — distinct from the q8 / expires_at_tick
        // shapes so it surfaces here on swap regressions.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));"),
            "DoT/HoT/TimedShield arms must store duration_ticks at payload \
             word 3 (ring offset 5) as raw u32 (= payload_b);\n{wgsl}"
        );

        // Extended-status slice — Stealth/Charm/Grounded/Suppress now
        // write real chronicle records (kinds 54/55/56/57). Stealth is
        // caster-self status (no target slot in the engine event),
        // duration at payload word 1 (= ring slot offset 3). Charm/
        // Grounded/Suppress are target-cast statuses, duration at
        // payload word 2 (= ring slot offset 4). Pin the kind tags so
        // a regression that drops the wire-up surfaces here.
        for (kind_token, expected_event_id, name) in &[
            ("kind == 27u", 54u32, "Stealth"),
            ("kind == 28u", 55u32, "Charm"),
            ("kind == 29u", 56u32, "Grounded"),
            ("kind == 30u", 57u32, "Suppress"),
        ] {
            assert!(
                !wgsl.contains(&format!(
                    "TODO slice γ: chronicle_append_{}",
                    name.to_lowercase()
                )),
                "{name} arm should no longer carry the TODO marker;\n{wgsl}"
            );
            assert!(
                wgsl.contains(kind_token),
                "{name} arm dispatch ({kind_token}) must be present;\n{wgsl}"
            );
            assert!(
                wgsl.contains(&format!(
                    "atomicStore(&event_ring[_slot * 11u + 0u], {expected_event_id}u);"
                )),
                "{name} arm must store kind={expected_event_id};\n{wgsl}"
            );
        }
        // Stealth: duration_ticks at slot 3 (raw u32 from payload_a, no
        // target field). Charm/Grounded/Suppress: duration_ticks at slot
        // 4 (raw u32 from payload_a, target at slot 3). Pin both shapes
        // so a regression that swaps them surfaces here.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 3u], (payload_a));"),
            "Stealth arm must store duration_ticks at payload word 1 \
             (ring offset 3) as raw u32 (= payload_a, no target field);\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 4u], (payload_a));"),
            "Charm/Grounded/Suppress arms must store duration_ticks at \
             payload word 2 (ring offset 4) as raw u32 (= payload_a);\n{wgsl}"
        );

        // Slice γ tail — Buff/Harvest/PlaceVoxel/Reflect now write real
        // chronicle records (kinds 58/59/60/61). Four distinct shapes:
        //   - Buff (kind 23 → 58): target-cast with packed payload.
        //     5-payload-word record (caster + target + raw payload_a +
        //     raw payload_b). Consumer decodes packed bits.
        //   - Harvest (kind 25 → 59): caster-self. 4-payload-word record
        //     (caster + kind_hash + amount). No target field.
        //   - PlaceVoxel (kind 26 → 60): caster-self. 3-payload-word
        //     record (caster + kind_hash). Position implicit.
        //   - Reflect (kind 31 → 61): target-cast with packed payload.
        //     5-payload-word record (caster + target + raw payload_a +
        //     raw payload_b). Consumer sign-extends fraction_q8 from
        //     payload_b's low 16 bits.
        // Pin the kind tags so a regression that drops the wire-up
        // surfaces here.
        for (kind_token, expected_event_id, name) in &[
            ("kind == 23u", 58u32, "Buff"),
            ("kind == 25u", 59u32, "Harvest"),
            ("kind == 26u", 60u32, "PlaceVoxel"),
            ("kind == 31u", 61u32, "Reflect"),
            // Slice γ closer — Summon (the last `// TODO slice γ` arm).
            ("kind == 24u", 62u32, "Summon"),
        ] {
            // Use snake_case for the marker text; match `chronicle_append_<name>`
            // form. PlaceVoxel needs explicit snake_case.
            let snake = match *name {
                "Buff"       => "buff",
                "Harvest"    => "harvest",
                "PlaceVoxel" => "place_voxel",
                "Reflect"    => "reflect",
                "Summon"     => "summon",
                _ => unreachable!(),
            };
            assert!(
                !wgsl.contains(&format!("TODO slice γ: chronicle_append_{snake}")),
                "{name} arm should no longer carry the TODO marker;\n{wgsl}"
            );
            assert!(
                wgsl.contains(kind_token),
                "{name} arm dispatch ({kind_token}) must be present;\n{wgsl}"
            );
            assert!(
                wgsl.contains(&format!(
                    "atomicStore(&event_ring[_slot * 11u + 0u], {expected_event_id}u);"
                )),
                "{name} arm must store kind={expected_event_id};\n{wgsl}"
            );
        }
        // Buff/Reflect: target-cast with packed payload — store BOTH
        // raw payload_a (slot 4) AND raw payload_b (slot 5). Harvest:
        // caster-self with payload_a at slot 3 + payload_b at slot 4.
        // PlaceVoxel: caster-self with payload_a at slot 3 only.
        // Pin the raw `(payload_b)` write so a regression that bitcasts
        // (or omits) the second payload word surfaces here. Note:
        // `(payload_b)` already appears in DoT/HoT/TimedShield arms;
        // having additional sites for Buff/Reflect just reuses the
        // same pattern.
        assert!(
            wgsl.matches("atomicStore(&event_ring[_slot * 11u + 5u], (payload_b));").count() >= 5,
            "expected ≥5 raw payload_b writes at slot 5 (DoT + HoT + TimedShield + \
             Buff + Reflect);\n{wgsl}"
        );

        // Slice γ — Damage arm wiring assertions.
        // The Damage arm replaced its TODO marker with a real chronicle
        // write that mirrors `lower_emit_to_wgsl`'s shape: atomicAdd
        // for slot acquisition, bounds-check against ring cap, then
        // header + payload atomicStores against the SAME `event_ring`
        // buffer the runtime cascade reads from.
        // Tight pattern — `chronicle_append_damage(` excludes the
        // (still-TODO) `chronicle_append_damage_over_time(` arm which
        // shares a prefix. (`chronicle_append_damage_modify(` is no
        // longer TODO — wired by the Fortify verb swap, Task #138
        // follow-on, mirror of Vampirize.)
        assert!(
            !wgsl.contains("TODO slice γ: chronicle_append_damage("),
            "Damage arm should no longer carry the TODO marker;\n{wgsl}"
        );
        // Header tag — EventKindId::EffectDamageApplied = 26.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 0u], 26u);"),
            "Damage arm must store kind=26 (EffectDamageApplied);\n{wgsl}"
        );
        // Header tick.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 1u], tick);"),
            "Damage arm must store tick at header word 1;\n{wgsl}"
        );
        // Self-cast caster + target (slice γ uses agent_id for both;
        // explicit caster/target arrives when CgStmt::ApplyAbility
        // grows those fields).
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 2u], (caster_slot));"),
            "Damage arm must store caster=agent_id at payload word 0;\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 3u], (target_slot));"),
            "Damage arm must store target=agent_id at payload word 1 (slice γ self-cast);\n{wgsl}"
        );
        // Amount payload — bitcast f32 → u32.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 4u], bitcast<u32>(amount));"),
            "Damage arm must store amount as bitcast<u32>(f32);\n{wgsl}"
        );
        // Bounds check against DEFAULT_EVENT_RING_CAP_SLOTS.
        assert!(
            wgsl.contains("if (_slot < 1048576u) {"),
            "Damage arm must bounds-check _slot;\n{wgsl}"
        );
        // Slot acquisition via atomicAdd on event_tail.
        assert!(
            wgsl.contains("let _slot: u32 = atomicAdd(&event_tail[0], 1u);"),
            "Damage arm must acquire slot via atomicAdd on event_tail;\n{wgsl}"
        );

        // Slice γ — remaining 6 chronicle-bearing arms wired:
        //   Heal=27, Shield=28, Stun=29, Slow=30,
        //   TransferGold=31, ModifyStanding=32.
        // Each pinned by a kind-tag header store + the matching
        // expected-payload assertion. Per-variant body shapes vary
        // (Stun/Slow compute expires_at_tick, Slow has 4 payload
        // fields, TransferGold/ModifyStanding bitcast i32 deltas);
        // pinning the kind-tag write is the minimal sufficient guard
        // against the dispatcher wiring drifting from the discriminant
        // table.
        for (variant_label, expected_kind_tag) in &[
            ("Heal",            27u32),
            ("Shield",          28u32),
            ("Stun",            29u32),
            ("Slow",            30u32),
            ("TransferGold",    31u32),
            ("ModifyStanding",  32u32),
            // Bleed verb swap (Task #138 follow-on, 2026-05-06):
            // SelfDamage = 17 → EventKindId::EffectSelfDamageApplied = 39.
            ("SelfDamage",      39u32),
            // Vampirize verb swap (Task #138 follow-on, mirror of Bleed):
            // LifeSteal = 18 → EventKindId::EffectLifeStealApplied = 40.
            ("LifeSteal",       40u32),
            // Fortify verb swap (Task #138 follow-on, mirror of Vampirize):
            // DamageModify = 19 → EventKindId::EffectDamageModifyApplied = 41.
            ("DamageModify",    41u32),
            // Reap verb swap (Task #138 follow-on, mirror of Fortify):
            // Execute = 16 → EventKindId::EffectExecuteApplied = 42.
            ("Execute",         42u32),
        ] {
            let needle = format!(
                "atomicStore(&event_ring[_slot * 11u + 0u], {expected_kind_tag}u);"
            );
            assert!(
                wgsl.contains(&needle),
                "{variant_label} arm must store kind={expected_kind_tag} \
                 (header word 0 of chronicle ring);\n{wgsl}"
            );
        }

        // Slow's 4-field payload — factor_q8 lives at payload word 3
        // (= ring slot offset 5). Pin it explicitly.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 5u], bitcast<u32>(factor_q8));"),
            "Slow arm must store factor_q8 at payload word 3 (ring offset 5);\n{wgsl}"
        );
        // LifeSteal's 4-field payload — fraction_q8 lives at payload
        // word 3 (= ring slot offset 5), same shape as Slow.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 5u], bitcast<u32>(fraction_q8));"),
            "LifeSteal arm must store fraction_q8 at payload word 3 (ring offset 5);\n{wgsl}"
        );
        // DamageModify's 4-field payload — multiplier_q8 lives at
        // payload word 3 (= ring slot offset 5), same shape as Slow /
        // LifeSteal. (Fortify verb swap, Task #138 follow-on, mirror
        // of Vampirize.)
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 11u + 5u], bitcast<u32>(multiplier_q8));"),
            "DamageModify arm must store multiplier_q8 at payload word 3 (ring offset 5);\n{wgsl}"
        );

        // Stun, Slow, LifeSteal, DamageModify, Root, Silence, Fear, and
        // Taunt each compute expires_at_tick = tick + duration. Eight
        // arms × one statement = 8 occurrences in the primary walk, and
        // Wave 1.5#9 added a structurally-identical nested walk that
        // re-emits the same chain at a deeper indent — total 16
        // occurrences across the dispatcher.
        // Wave 2 piece 1 added Root/Silence/Fear/Taunt (kinds 8/9/10/11),
        // doubling the count from 8 to 16.
        assert_eq!(
            wgsl.matches("let expires_at_tick: u32 = tick + payload_a;").count(),
            16,
            "Stun + Slow + LifeSteal + DamageModify + Root + Silence + Fear + Taunt \
             arms each compute expires_at_tick twice (primary + nested walks); \
             expected 16 occurrences across the dispatcher;\n{wgsl}"
        );

        // Wave 1.5#9 nested-effect walk pin: the dispatcher reads the
        // nested SoA columns and walks MAX_NESTED_PER_EFFECT (=2)
        // entries per slot, after the primary's chronicle write.
        assert!(
            wgsl.contains("ability_registry_nested_effect_kinds[nested_slot_base + j]"),
            "nested walk must read nested_effect_kinds SoA column;\n{wgsl}"
        );
        assert!(
            wgsl.contains("ability_registry_nested_effect_payload_a[nested_slot_base + j]"),
            "nested walk must read nested_effect_payload_a SoA column;\n{wgsl}"
        );
        assert!(
            wgsl.contains("ability_registry_nested_effect_payload_b[nested_slot_base + j]"),
            "nested walk must read nested_effect_payload_b SoA column;\n{wgsl}"
        );
        assert!(
            wgsl.contains("for (var j: u32 = 0u; j < 2u"),
            "nested walk must iterate MAX_NESTED_PER_EFFECT (=2) entries per slot;\n{wgsl}"
        );
        assert!(
            wgsl.contains("nested_base: u32 = ability_slot * 12u"),
            "nested base = ability_slot * MAX_EFFECTS_PER_PROGRAM * MAX_NESTED_PER_EFFECT \
             = ability_slot * 12;\n{wgsl}"
        );
    }

    // ---- Task #230 (2026-05-08): per-thread workgroup-shared bitonic
    //      sort for the AOE Spread shape. Pins the new emit shape so a
    //      regression to insertion sort or to the 16-slot cap surfaces
    //      at compile time, not from a fixture pin two crates downstream.

    #[test]
    fn apply_ability_aoe_spread_emits_bitonic_sort_with_256_slot_cap() {
        // Mirrors `apply_ability_emits_dispatcher_loop_with_branch_arms`
        // but with `with_aoe_dispatch: true` so the rendered body
        // includes the `area_kind == 4u` (Spread) branch. Asserts:
        //   1. The Spread branch is present (area_kind == 4u arm).
        //   2. The collection buffer is sized for 256 candidates (not
        //      the prior 16-slot cap).
        //   3. The sort uses a bitonic compare-swap shape — three
        //      nested loops (`_stage`, `_step`, `_ii`) over an XOR-mate
        //      index, with min/max compare-swap. Direction alternates
        //      via `(_ii & _stage) == 0u` per Batcher's bitonic sort.
        //   4. Padding is initialised to 0xFFFFFFFFu so unused slots
        //      sort to the high end and never enter the kept set.
        //   5. The prior O(K²) insertion-sort markers are gone — no
        //      more `loop { if (_j < 0)` or `collected[u32(_j)] <= _key`
        //      patterns.
        let mut prog = empty_prog();
        let ability_lit = push_expr(&mut prog, CgExpr::Lit(LitValue::U32(1)));
        let caster_self = push_expr(&mut prog, CgExpr::AgentSelfId);
        let stmt_id = push_stmt(
            &mut prog,
            CgStmt::ApplyAbility {
                ability: ability_lit,
                caster: caster_self,
                target: caster_self,
                with_aoe_dispatch: true,
            },
        );
        let ctx = EmitCtx::structural(&prog);
        let wgsl = lower_cg_stmt_to_wgsl(stmt_id, &ctx).expect("AOE-enabled lower");

        // 1. Spread branch present.
        assert!(
            wgsl.contains("} else if (area_kind == 4u) {"),
            "Spread branch (area_kind == 4u) must be in the AOE dispatch chain;\n{wgsl}",
        );

        // 2. 256-slot collection buffer.
        assert!(
            wgsl.contains("var collected: array<u32, 256>;"),
            "Spread collection buffer must be sized for 256 candidates (task #230 cap bump);\n{wgsl}",
        );
        assert!(
            wgsl.contains("if (n_collected < 256u)"),
            "Spread collection guard must reflect the 256-slot cap;\n{wgsl}",
        );
        // The prior 16-slot shape must be gone.
        assert!(
            !wgsl.contains("var collected: array<u32, 16>;"),
            "Prior 16-slot Spread cap must not regress;\n{wgsl}",
        );
        assert!(
            !wgsl.contains("if (n_collected < 16u)"),
            "Prior 16-slot Spread guard must not regress;\n{wgsl}",
        );

        // 3. Bitonic sort markers — three nested loops, XOR mate index,
        //    direction bit derived from (_ii & _stage), min/max compare-
        //    swap.
        assert!(
            wgsl.contains("for (var _stage: u32 = 2u; _stage <= 256u; _stage = _stage << 1u)"),
            "Bitonic outer stage loop must double from 2 to 256;\n{wgsl}",
        );
        assert!(
            wgsl.contains("for (var _step: u32 = _stage >> 1u; _step > 0u; _step = _step >> 1u)"),
            "Bitonic inner step loop must halve from stage/2 to 1;\n{wgsl}",
        );
        assert!(
            wgsl.contains("let _ixor: u32 = _ii ^ _step;"),
            "Bitonic compare-swap mate index = _ii XOR _step;\n{wgsl}",
        );
        assert!(
            wgsl.contains("let _ascending: bool = (_ii & _stage) == 0u;"),
            "Bitonic direction bit must come from (_ii & _stage);\n{wgsl}",
        );
        assert!(
            wgsl.contains("let _lo: u32 = min(_a, _b);"),
            "Bitonic compare-swap must use min for lo;\n{wgsl}",
        );
        assert!(
            wgsl.contains("let _hi: u32 = max(_a, _b);"),
            "Bitonic compare-swap must use max for hi;\n{wgsl}",
        );

        // 4. Padding initialised to 0xFFFFFFFFu (max u32) so unused
        //    slots drift to the high end and never enter the kept set.
        assert!(
            wgsl.contains("collected[_pad] = 0xFFFFFFFFu;"),
            "Padding init must use 0xFFFFFFFFu sentinel;\n{wgsl}",
        );

        // 5. Insertion-sort markers must be gone.
        assert!(
            !wgsl.contains("if (_j < 0) { break; }"),
            "Insertion-sort `if (_j < 0) break` regression detected;\n{wgsl}",
        );
        assert!(
            !wgsl.contains("if (collected[u32(_j)] <= _key)"),
            "Insertion-sort comparison regression detected;\n{wgsl}",
        );
        assert!(
            !wgsl.contains("collected[u32(_j) + 1u] = collected[u32(_j)];"),
            "Insertion-sort shift regression detected;\n{wgsl}",
        );

        // The truncate-to-max_targets logic that follows the sort is
        // unchanged — pin it so the kept-set selection contract stays
        // explicit.
        assert!(
            wgsl.contains("let n_emit: u32 = min(n_collected, max_targets);"),
            "Truncate-to-max_targets clamp must follow the sort;\n{wgsl}",
        );
    }

    // ---- emit_chronicle_append_skeleton — shared by lower_emit_to_wgsl
    //      and the #136 ApplyAbility dispatcher arms (slice γ+).

    #[test]
    fn chronicle_skeleton_renders_atomicadd_bounds_check_and_header_writes() {
        let field_writes = vec![
            "        atomicStore(&my_ring[slot * 4u + 2u], (caster_id));"
                .to_string(),
            "        atomicStore(&my_ring[slot * 4u + 3u], bitcast<u32>(amount));"
                .to_string(),
        ];
        let wgsl = emit_chronicle_append_skeleton(
            /*event_id*/ 26,
            /*buf*/ "my_ring",
            /*stride*/ 4,
            /*field_count*/ 2,
            &field_writes,
            DebugWgslFlags::NONE,
            0, 0,
            "agent_id",
        );

        // Header comment carries event id + field count for capture
        // diagnostics.
        assert!(wgsl.contains("// emit event#26 (2 fields)"),
            "header comment must include id + field count;\n{wgsl}");

        // Slot acquisition via atomicAdd on the canonical event_tail.
        assert!(wgsl.contains("let slot = atomicAdd(&event_tail[0], 1u);"),
            "slot acquisition must use atomicAdd on event_tail[0];\n{wgsl}");

        // Bounds check against DEFAULT_EVENT_RING_CAP_SLOTS (65536).
        assert!(wgsl.contains("if (slot < 1048576u) {"),
            "must bounds-check slot against DEFAULT_EVENT_RING_CAP_SLOTS;\n{wgsl}");

        // Header words: event-kind tag at offset 0, tick at offset 1.
        assert!(wgsl.contains("atomicStore(&my_ring[slot * 4u + 0u], 26u);"),
            "tag header at slot*stride+0;\n{wgsl}");
        assert!(wgsl.contains("atomicStore(&my_ring[slot * 4u + 1u], tick);"),
            "tick header at slot*stride+1;\n{wgsl}");

        // Caller's field-write lines round-trip verbatim.
        assert!(wgsl.contains("atomicStore(&my_ring[slot * 4u + 2u], (caster_id));"),
            "field-write lines must round-trip;\n{wgsl}");
        assert!(wgsl.contains("atomicStore(&my_ring[slot * 4u + 3u], bitcast<u32>(amount));"),
            "field-write lines must round-trip;\n{wgsl}");
    }

    #[test]
    fn chronicle_skeleton_zero_field_emit_still_writes_header() {
        // Some events (e.g. AgentDied = no payload beyond agent id in
        // the standard layout's header) have zero declared fields.
        // The skeleton must still emit the slot acquisition + tag/tick
        // header writes, just with no field-write lines.
        let wgsl = emit_chronicle_append_skeleton(
            2,
            "ring",
            2,
            0,
            &[],
            DebugWgslFlags::NONE,
            0, 0,
            "agent_id",
        );
        assert!(wgsl.contains("atomicAdd(&event_tail[0], 1u);"));
        assert!(wgsl.contains("atomicStore(&ring[slot * 2u + 0u], 2u);"));
        assert!(wgsl.contains("atomicStore(&ring[slot * 2u + 1u], tick);"));
    }

    // ---- Compiler debug mode Phase 2 (DebugWgslFlags) ----

    /// `event_kind_histogram=true` emits a parallel
    /// `atomicAdd(&event_kind_counts[<kind>], 1u)` alongside the
    /// existing `event_tail` bump. Default `NONE` does NOT emit it.
    #[test]
    fn chronicle_skeleton_emits_event_kind_histogram_when_flag_set() {
        // Default `NONE` — the histogram increment must NOT appear.
        let baseline = emit_chronicle_append_skeleton(
            27,
            "my_ring",
            4,
            0,
            &[],
            DebugWgslFlags::NONE,
            0, 0,
            "agent_id",
        );
        assert!(
            !baseline.contains("event_kind_counts"),
            "DebugWgslFlags::NONE must not emit histogram counter;\n{baseline}"
        );

        // With the flag set — the per-kind atomicAdd must appear,
        // referencing the same event_id (27) the rest of the skeleton
        // bakes into the tag store.
        let flagged = emit_chronicle_append_skeleton(
            27,
            "my_ring",
            4,
            0,
            &[],
            crate::cg::lower::driver::DebugWgslFlags {
                event_kind_histogram: true,
                ..crate::cg::lower::driver::DebugWgslFlags::NONE
            },
            0, 0,
            "agent_id",
        );
        assert!(
            flagged.contains("atomicAdd(&event_kind_counts[27u], 1u);"),
            "event_kind_histogram=true must emit per-kind atomicAdd;\n{flagged}"
        );
        // The existing `event_tail` bump is preserved verbatim.
        assert!(
            flagged.contains("let slot = atomicAdd(&event_tail[0], 1u);"),
            "event_tail bump must remain;\n{flagged}"
        );
    }

    // ---- EFFECT_KIND_TO_EVENT_KIND_ID — slice γ pre-fact pin.
    //      Asserts each entry agrees with both source-of-truths:
    //        - LEFT  : `pack_effect`'s discriminant (engine pack table)
    //        - RIGHT : engine `EventKindId` enum

    #[test]
    fn effect_kind_to_event_kind_map_matches_engine() {
        use engine::ability::{
            AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
            PackedAbilityRegistry,
        };
        use engine::ability::program::BuffStat;
        use engine::cascade::handler::EventKindId as EngineEventKindId;

        let pack_one = |op: EffectOp| -> u32 {
            let prog = AbilityProgram::new_single_target(
                5.0,
                Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
                [op],
            );
            let mut b = AbilityRegistryBuilder::new();
            b.register(prog);
            let reg = b.build();
            PackedAbilityRegistry::pack(&reg).effect_kinds[0] as u32
        };

        // LEFT side: each entry's effect-kind discriminant matches the
        // pack table's output for a representative `EffectOp` value.
        let representative_for = |kind: u32| -> EffectOp {
            match kind {
                0  => EffectOp::Damage    { amount: 10.0 },
                1  => EffectOp::Heal      { amount: 5.0 },
                2  => EffectOp::Shield    { amount: 25.0 },
                3  => EffectOp::Stun      { duration_ticks: 10 },
                4  => EffectOp::Slow      { duration_ticks: 10, factor_q8: 128 },
                5  => EffectOp::TransferGold   { amount: 7 },
                6  => EffectOp::ModifyStanding { delta: 3 },
                8  => EffectOp::Root      { duration_ticks: 30 },
                9  => EffectOp::Silence   { duration_ticks: 30 },
                10 => EffectOp::Fear      { duration_ticks: 30 },
                11 => EffectOp::Taunt     { duration_ticks: 30 },
                12 => EffectOp::Dash      { distance: 10.0 },
                13 => EffectOp::Blink     { distance: 10.0 },
                14 => EffectOp::Knockback { distance: 5.0 },
                15 => EffectOp::Pull      { distance: 5.0 },
                16 => EffectOp::Execute   { hp_threshold: 20.0 },
                17 => EffectOp::SelfDamage { amount: 5.0 },
                18 => EffectOp::LifeSteal { duration_ticks: 50, fraction_q8: 128 },
                19 => EffectOp::DamageModify { duration_ticks: 50, multiplier_q8: 128 },
                20 => EffectOp::DamageOverTime { amount: 5.0, duration_ticks: 30 },
                21 => EffectOp::HealOverTime   { amount: 3.0, duration_ticks: 30 },
                22 => EffectOp::TimedShield    { amount: 25.0, duration_ticks: 30 },
                27 => EffectOp::Stealth   { duration_ticks: 50 },
                28 => EffectOp::Charm     { duration_ticks: 50 },
                29 => EffectOp::Grounded  { duration_ticks: 50 },
                30 => EffectOp::Suppress  { duration_ticks: 50 },
                23 => EffectOp::Buff       { stat: BuffStat::MoveSpeed, magnitude_q8: 64, duration_ticks: 50 },
                24 => EffectOp::Summon     { template_hash: 0xDEADBEEF, count: 3, lifetime_ticks: 120 },
                25 => EffectOp::Harvest    { kind_hash: 0xCAFEBABE, amount: 5 },
                26 => EffectOp::PlaceVoxel { kind_hash: 0xFACEFEED },
                31 => EffectOp::Reflect    { duration_ticks: 50, fraction_q8: 64 },
                32 => EffectOp::PlantBelief { subject_idx: 7, fact_bit: 5 },
                33 => EffectOp::Observe     { target_observer: 0 },
                34 => EffectOp::Scry        { target_observer: 3, subject_idx: 4 },
                35 => EffectOp::Reveal      { subject_idx: 4 },
                36 => EffectOp::Disguise    { fake_type: 7, duration_ticks: 200 },
                37 => EffectOp::Decoy       { subject_idx: 4, fake_pos: 0xDEADBEEF },
                38 => EffectOp::EraseBelief { subject_idx: 4, fields: 0b00111111 },
                39 => EffectOp::TravelTo    { dest_x_q8: 1280, dest_y_q8: 1280, eta_ticks: 50 },
                40 => EffectOp::Recipe      { recipe_id: 42, target_tool: 0xFF },
                41 => EffectOp::WearTool    { tool_kind: 3, amount: 64 },
                42 => EffectOp::Propose     { contract_kind: 1, expires_at_tick: 0 },
                43 => EffectOp::Announce    { announcement_kind: 7, radius_q8: 896 },
                44 => EffectOp::GainSkill   { skill_id: 2, amount_q8: 64 },
                45 => EffectOp::CreateObligation { obligation_id: 17, kind: 0 },
                46 => EffectOp::CastBegin { ability_id: 1, duration_ticks: 3, target_slot: 0, target_x_q8: 0, target_y_q8: 0 },
                _ => panic!("test only covers chronicle-bearing variants 0..=6 + 8..=15 + 16 + 17 + 18 + 19 + 20..=22 + 27..=30 + 23/24/25/26/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46"),
            }
        };

        // RIGHT side: each entry's event-kind id matches the engine
        // enum's `as u32`.
        let event_kind_id_for = |effect_kind: u32| -> u32 {
            match effect_kind {
                0  => EngineEventKindId::EffectDamageApplied as u32,
                1  => EngineEventKindId::EffectHealApplied   as u32,
                2  => EngineEventKindId::EffectShieldApplied as u32,
                3  => EngineEventKindId::EffectStunApplied   as u32,
                4  => EngineEventKindId::EffectSlowApplied   as u32,
                5  => EngineEventKindId::EffectGoldTransfer  as u32,
                6  => EngineEventKindId::EffectStandingDelta as u32,
                8  => EngineEventKindId::EffectRootApplied   as u32,
                9  => EngineEventKindId::EffectSilenceApplied as u32,
                10 => EngineEventKindId::EffectFearApplied   as u32,
                11 => EngineEventKindId::EffectTauntApplied  as u32,
                12 => EngineEventKindId::EffectDashApplied   as u32,
                13 => EngineEventKindId::EffectBlinkApplied  as u32,
                14 => EngineEventKindId::EffectKnockbackApplied as u32,
                15 => EngineEventKindId::EffectPullApplied   as u32,
                16 => EngineEventKindId::EffectExecuteApplied as u32,
                17 => EngineEventKindId::EffectSelfDamageApplied as u32,
                18 => EngineEventKindId::EffectLifeStealApplied as u32,
                19 => EngineEventKindId::EffectDamageModifyApplied as u32,
                20 => EngineEventKindId::EffectDamageOverTimeApplied as u32,
                21 => EngineEventKindId::EffectHealOverTimeApplied   as u32,
                22 => EngineEventKindId::EffectTimedShieldApplied    as u32,
                27 => EngineEventKindId::EffectStealthApplied        as u32,
                28 => EngineEventKindId::EffectCharmApplied          as u32,
                29 => EngineEventKindId::EffectGroundedApplied       as u32,
                30 => EngineEventKindId::EffectSuppressApplied       as u32,
                23 => EngineEventKindId::EffectBuffApplied           as u32,
                24 => EngineEventKindId::EffectSummonApplied         as u32,
                25 => EngineEventKindId::EffectHarvestApplied        as u32,
                26 => EngineEventKindId::EffectPlaceVoxelApplied     as u32,
                31 => EngineEventKindId::EffectReflectApplied        as u32,
                32 => EngineEventKindId::EffectPlantBeliefApplied     as u32,
                33 => EngineEventKindId::EffectObserveApplied         as u32,
                34 => EngineEventKindId::EffectScryApplied            as u32,
                35 => EngineEventKindId::EffectRevealApplied          as u32,
                36 => EngineEventKindId::EffectDisguiseApplied        as u32,
                37 => EngineEventKindId::EffectDecoyApplied           as u32,
                38 => EngineEventKindId::EffectEraseBeliefApplied     as u32,
                39 => EngineEventKindId::EffectTravelToApplied        as u32,
                40 => EngineEventKindId::EffectRecipeApplied          as u32,
                41 => EngineEventKindId::EffectWearToolApplied        as u32,
                42 => EngineEventKindId::EffectProposeApplied         as u32,
                43 => EngineEventKindId::EffectAnnounceApplied        as u32,
                44 => EngineEventKindId::EffectGainSkillApplied        as u32,
                45 => EngineEventKindId::EffectCreateObligationApplied as u32,
                46 => EngineEventKindId::EffectCastBeginApplied         as u32,
                _ => panic!("test only covers chronicle-bearing variants 0..=6 + 8..=15 + 16 + 17 + 18 + 19 + 20..=22 + 27..=30 + 23/24/25/26/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46"),
            }
        };

        for &(effect_kind, event_kind_id) in EFFECT_KIND_TO_EVENT_KIND_ID {
            let packed = pack_one(representative_for(effect_kind));
            assert_eq!(
                packed, effect_kind,
                "EFFECT_KIND_TO_EVENT_KIND_ID left ({effect_kind}) drifted from \
                 pack_effect (got {packed}); a renumbering of EffectOp \
                 silently rewrites this table"
            );
            let expected_event = event_kind_id_for(effect_kind);
            assert_eq!(
                event_kind_id, expected_event,
                "EFFECT_KIND_TO_EVENT_KIND_ID right ({event_kind_id}) for effect \
                 discriminant {effect_kind} drifted from EngineEventKindId \
                 (got {expected_event}); chronicle records will route to the \
                 wrong cascade handler"
            );
        }
    }

    #[test]
    fn event_kind_id_for_effect_kind_lookup_matches_table() {
        // Spot-check the helper against the table itself plus the
        // negative case (an effect-kind absent from the map returns
        // None — these arms keep their TODO marker until a runtime
        // change adds a chronicle counterpart).
        assert_eq!(event_kind_id_for_effect_kind(0), Some(26),
            "Damage → EffectDamageApplied");
        assert_eq!(event_kind_id_for_effect_kind(1), Some(27),
            "Heal → EffectHealApplied");
        assert_eq!(event_kind_id_for_effect_kind(6), Some(32),
            "ModifyStanding → EffectStandingDelta");
        // Wave 2 piece 1 — control statuses now wired:
        assert_eq!(event_kind_id_for_effect_kind(8), Some(43),
            "Root → EffectRootApplied (Wave 2 piece 1)");
        assert_eq!(event_kind_id_for_effect_kind(9), Some(44),
            "Silence → EffectSilenceApplied (Wave 2 piece 1)");
        assert_eq!(event_kind_id_for_effect_kind(10), Some(45),
            "Fear → EffectFearApplied (Wave 2 piece 1)");
        assert_eq!(event_kind_id_for_effect_kind(11), Some(46),
            "Taunt → EffectTauntApplied (Wave 2 piece 1)");
        // Wave 2 piece 2 — movement EffectOps now wired:
        assert_eq!(event_kind_id_for_effect_kind(12), Some(47),
            "Dash → EffectDashApplied (Wave 2 piece 2)");
        assert_eq!(event_kind_id_for_effect_kind(13), Some(48),
            "Blink → EffectBlinkApplied (Wave 2 piece 2)");
        assert_eq!(event_kind_id_for_effect_kind(14), Some(49),
            "Knockback → EffectKnockbackApplied (Wave 2 piece 2)");
        assert_eq!(event_kind_id_for_effect_kind(15), Some(50),
            "Pull → EffectPullApplied (Wave 2 piece 2)");
        // Wave 1.5+ — multi-tick effects now wired:
        assert_eq!(event_kind_id_for_effect_kind(20), Some(51),
            "DamageOverTime → EffectDamageOverTimeApplied (Wave 1.5+)");
        assert_eq!(event_kind_id_for_effect_kind(21), Some(52),
            "HealOverTime → EffectHealOverTimeApplied (Wave 1.5+)");
        assert_eq!(event_kind_id_for_effect_kind(22), Some(53),
            "TimedShield → EffectTimedShieldApplied (Wave 1.5+)");
        // Extended-corpus statuses now wired:
        assert_eq!(event_kind_id_for_effect_kind(27), Some(54),
            "Stealth → EffectStealthApplied (extended status)");
        assert_eq!(event_kind_id_for_effect_kind(28), Some(55),
            "Charm → EffectCharmApplied (extended status)");
        assert_eq!(event_kind_id_for_effect_kind(29), Some(56),
            "Grounded → EffectGroundedApplied (extended status)");
        assert_eq!(event_kind_id_for_effect_kind(30), Some(57),
            "Suppress → EffectSuppressApplied (extended status)");
        // Slice γ tail now wired:
        assert_eq!(event_kind_id_for_effect_kind(23), Some(58),
            "Buff → EffectBuffApplied (slice γ tail)");
        assert_eq!(event_kind_id_for_effect_kind(25), Some(59),
            "Harvest → EffectHarvestApplied (slice γ tail)");
        assert_eq!(event_kind_id_for_effect_kind(26), Some(60),
            "PlaceVoxel → EffectPlaceVoxelApplied (slice γ tail)");
        assert_eq!(event_kind_id_for_effect_kind(31), Some(61),
            "Reflect → EffectReflectApplied (slice γ tail)");
        // Slice γ closer — Summon (the last `// TODO slice γ` arm) now
        // wired:
        assert_eq!(event_kind_id_for_effect_kind(24), Some(62),
            "Summon → EffectSummonApplied (slice γ closer)");
        assert_eq!(event_kind_id_for_effect_kind(7), None,
            "CastAbility (recursive dispatch) has no chronicle kind");
    }

    #[test]
    fn effect_kind_to_event_kind_map_covers_chronicle_bearing_variants_only() {
        // 31 chronicle-bearing variants today — Damage/Heal/Shield/Stun/
        // Slow/TransferGold/ModifyStanding + SelfDamage (Bleed verb
        // swap, Task #138 follow-on, 2026-05-06) + LifeSteal (Vampirize
        // verb swap, Task #138 follow-on, mirror of Bleed) + DamageModify
        // (Fortify verb swap, Task #138 follow-on, mirror of Vampirize)
        // + Execute (Reap verb swap, Task #138 follow-on, mirror of
        // Fortify — closes the slice across all 8 duel_abilities verbs)
        // + Root/Silence/Fear/Taunt (Wave 2 piece 1, control statuses)
        // + Dash/Blink/Knockback/Pull (Wave 2 piece 2, movement EffectOps)
        // + DamageOverTime/HealOverTime/TimedShield (Wave 1.5+ multi-tick)
        // + Stealth/Charm/Grounded/Suppress (extended-corpus statuses)
        // + Buff/Harvest/PlaceVoxel/Reflect (slice γ tail — closed 4 of
        // the 5 remaining `// TODO slice γ` arms)
        // + Summon (slice γ closer — closes the last remaining
        // `// TODO slice γ` arm; CastAbility=7 is the only EffectOp
        // without a chronicle counterpart, by design — recursive
        // dispatch).
        // + PlantBelief (Wave 3 ToM Phase 1 — bit-flag belief verb;
        // dispatcher writes EffectPlantBeliefApplied=63 records that
        // downstream `view ... -> u32` consumers fold into pair_map
        // cells via `self |= b`, same shape as `tom_probe.sim::beliefs`).
        // + Observe (Wave 3 ToM Phase 3 — self-observe-target verb;
        // dispatcher writes EffectObserveApplied=64 records that a
        // downstream runtime consumer reads to refresh the BeliefState
        // SoA's 6 columns from the agent SoA at consume tick).
        // + Scry (Wave 3 ToM Phase 3.5 — cross-observer access verb;
        // dispatcher writes EffectScryApplied=65 records that a
        // downstream runtime consumer copies the 6 BeliefState columns
        // from `[target_observer * N + subject_idx]` to
        // `[caster * N + subject_idx]`).
        // + Reveal (Wave 3 ToM Phase 3.5 — one-to-many propagation verb;
        // dispatcher writes EffectRevealApplied=66 records that a
        // downstream runtime consumer iterates every observer slot and
        // copies the 6 BeliefState columns from `[caster * N +
        // subject_idx]` to `[observer * N + subject_idx]`).
        // If this number changes, either the engine grew a new
        // `EffectXxxApplied` event (in which case the map gets a new
        // entry) or a variant lost its chronicle counterpart (in which
        // case the map drops an entry). Pin the count so the gap between
        // source-of-truths is loud.
        // + TravelTo (Lift A — multi-tick travel verb; dispatcher writes
        // EffectTravelToApplied=70 records that a downstream consumer
        // rule turns into `busy_until_tick` + `travel_dest_{x,y,z}` SoA
        // updates plus a per-tick interpolation kernel).
        // + Recipe + WearTool (Lift B — items / inventory + production /
        // recipes; dispatcher writes EffectRecipeApplied=71 +
        // EffectWearToolApplied=72 records that per-fixture consumer
        // rules turn into inventory ingredient/output deltas + tool wear
        // increments. See `docs/spec/economy.md §4.1` + §4.3).
        // + Propose + Announce (Lift C — bilateral consent + observer
        // fan-out; dispatcher writes EffectProposeApplied=73 +
        // EffectAnnounceApplied=74 records that per-fixture consumer
        // rules turn into ContractRegistry registrations + spatial-hash
        // observer broadcasts. See `docs/spec/economy.md §6` + §7).
        // + GainSkill + CreateObligation (Lift D — knowledge / skills +
        // obligation registry; dispatcher writes EffectGainSkillApplied=75
        // + EffectCreateObligationApplied=76 records that per-fixture
        // consumer rules turn into per-agent skill SoA bumps + obligation
        // pool registrations. See `docs/spec/economy.md §7` + §8).
        assert_eq!(
            EFFECT_KIND_TO_EVENT_KIND_ID.len(), 46,
            "EFFECT_KIND_TO_EVENT_KIND_ID should cover exactly the 46 \
             chronicle-bearing variants today (Plan G added CastBegin=46 \
             → EffectCastBeginApplied=77); if you added or removed an \
             entry, update this assertion (and the slice γ wire-up that \
             consumes the new entry)"
        );
    }
}
