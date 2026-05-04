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
use crate::cg::op::EventKindId;
use crate::cg::program::CgProgram;
use crate::cg::stmt::{
    CgMatchArm, CgStmt, CgStmtId, CgStmtListId, EventField, MatchArmBinding, StmtArena,
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
}

impl<'a> EmitCtx<'a> {
    /// Construct an emit context with the [`HandleNamingStrategy::Structural`]
    /// strategy — the only one Task 4.1 ships.
    pub fn structural(prog: &'a CgProgram) -> Self {
        Self {
            prog,
            naming: HandleNamingStrategy::Structural,
            tile_walk_index: std::cell::RefCell::new(None),
            dispatch: std::cell::Cell::new(None),
            view_target_locals: std::cell::RefCell::new(Vec::new()),
            pending_target_lets: std::cell::RefCell::new(Vec::new()),
            bound_target_exprs: std::cell::RefCell::new(std::collections::HashSet::new()),
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
/// catalog. Returns `<entity_snake>_<field_snake>` (e.g. `coin_weight`)
/// when the (entity, slot) pair has a catalog entry; falls back to the
/// opaque structural form `item_<entity>_<slot>` /
/// `group_<entity>_<slot>` so the WGSL still parses if the catalog is
/// missing the entry (a lowering defect).
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
    match resolved {
        Some((entity_name, field_name, _)) => {
            format!("{}_{}", to_snake_case(entity_name), field_name)
        }
        None => {
            let prefix = if is_item { "item" } else { "group" };
            format!("{}_{}_{}", prefix, entity_ref, slot)
        }
    }
}

/// Convert a PascalCase / camelCase identifier to snake_case. Mirrors
/// the helper of the same name in `cg/emit/kernel.rs` — kept here so
/// the body emit doesn't need to depend on the kernel emit's private
/// helpers. Adding the kernel-side helper to `pub(crate)` would couple
/// the two files; the duplicated four-line helper is the lower-friction
/// choice.
fn to_snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i != 0 {
            out.push('_');
        }
        for low in ch.to_lowercase() {
            out.push(low);
        }
    }
    out
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
        LtF32 | LtU32 | LtI32 => "<",
        LeF32 | LeU32 | LeI32 => "<=",
        GtF32 | GtU32 | GtI32 => ">",
        GeF32 | GeU32 | GeI32 => ">=",
        EqBool | EqU32 | EqI32 | EqF32 | EqAgentId => "==",
        NeBool | NeU32 | NeI32 | NeF32 | NeAgentId => "!=",
        And => "&&",
        Or => "||",
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
fn builtin_name(id: BuiltinId) -> String {
    use BuiltinId::*;
    match id {
        Distance => "distance".to_string(),
        PlanarDistance => "planar_distance".to_string(),
        ZSeparation => "z_separation".to_string(),
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
        // WGSL has a built-in `vec3<f32>` constructor; emit the call
        // as-is so `vec3(x, y, z)` lowers to `vec3<f32>(x, y, z)`.
        Vec3Ctor => "vec3<f32>".to_string(),
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
                    return Ok(agent_field_access(*field, target));
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
                return Ok(agent_field_access(*field, target));
            }
            // Item / Group fields: emit `<entity_snake>_<field>[<idx>]`.
            // The binding name is sourced from the program's
            // `entity_field_catalog` so kernel binding names + body
            // accesses agree on the same identifier (e.g. `coin_weight`).
            // The `<idx>` expression is the catalog-resolved target id;
            // it lowers identically to the AgentField `Target(_)` path
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
            Ok(format!("{}({})", builtin_name(*fn_id), parts.join(", ")))
        }
        CgExpr::Rng { purpose, ty: _ } => {
            // `per_agent_u32(seed, agent_id, tick, "<purpose>")` —
            // matches the engine RNG primitive named in
            // `engine::rng::per_agent_u32`. The seed/agent/tick names
            // are placeholders for Task 4.2, which knows the kernel's
            // local variable bindings.
            Ok(format!(
                "per_agent_u32(seed, agent_id, tick, \"{}\")",
                rng_purpose_token(*purpose)
            ))
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
            Ok(match ty {
                CgTy::AgentId | CgTy::U32 | CgTy::Tick => {
                    format!("{}[event_idx * {}u + {}u]", buf, stride, total_offset)
                }
                CgTy::I32 => format!(
                    "bitcast<i32>({}[event_idx * {}u + {}u])",
                    buf, stride, total_offset
                ),
                CgTy::F32 => format!(
                    "bitcast<f32>({}[event_idx * {}u + {}u])",
                    buf, stride, total_offset
                ),
                CgTy::Vec3F32 => format!(
                    "vec3<f32>(bitcast<f32>({buf}[event_idx * {s}u + {o}u]), bitcast<f32>({buf}[event_idx * {s}u + {o2}u]), bitcast<f32>({buf}[event_idx * {s}u + {o3}u]))",
                    buf = buf,
                    s = stride,
                    o = total_offset,
                    o2 = total_offset + 1,
                    o3 = total_offset + 2,
                ),
                CgTy::Bool => format!(
                    "({}[event_idx * {}u + {}u] != 0u)",
                    buf, stride, total_offset
                ),
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
                    // build_view_fold_bindings) but every shipped
                    // view today is f32-typed. The accumulator add
                    // is racy under contention (multiple GPU threads
                    // writing the same slot per tick) so we emit a
                    // CAS loop: atomicLoad → bitcast<f32> → add rhs
                    // → bitcast<u32> → atomicCompareExchangeWeak,
                    // retrying on the weak-CAS failure path. This
                    // satisfies P11 (Reduction Determinism) at the
                    // cost of a per-thread spin under heavy
                    // contention; sort-then-fold is a future
                    // enhancement that side-steps the spin entirely.
                    // When non-f32 view storage lands, this
                    // branches on the view's element type from
                    // `view_signatures`.
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
                let lhs = agent_field_access(*field, agent_ref);
                return Ok(format!("{} = {};", lhs, rhs));
            }
            let lhs = ctx.handle_name(target);
            let rhs = lower_cg_expr_to_wgsl(*value, ctx)?;
            Ok(format!("{} = {};", lhs, rhs))
        }
        CgStmt::Emit { event, fields } => lower_emit_to_wgsl(event.0, fields, ctx),
        CgStmt::If { cond, then, else_ } => {
            let c = lower_cg_expr_to_wgsl(*cond, ctx)?;
            let then_body = lower_cg_stmt_list_to_wgsl(*then, ctx)?;
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
            if matches!(ty, CgTy::AgentId) {
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
        } => emit_for_each_neighbor_body(*body, *radius_cells, ctx),
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
    }
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

    // Wrap the ring-append in `{ … }` so the `slot` let doesn't
    // collide with sibling emits. `event_ring_cap` is supplied as a
    // wgsl const by the kernel preamble (todo: thread through
    // PhysicsRule cfg uniform). For now hardcode 65536 slots — the
    // runtime allocates that many u32-words / stride.
    let mut out = String::new();
    out.push_str(&format!("// emit event#{event_id} ({} fields)\n", fields.len()));
    out.push_str("{\n");
    // Tail is `array<atomic<u32>>` with a single element — slot 0 is
    // the count. atomicAdd returns the prior value (this producer's
    // unique slot index).
    out.push_str("    let slot = atomicAdd(&event_tail[0], 1u);\n");
    // Bounds check — silently drop if ring full. Runtime sizes
    // event_ring to `DEFAULT_EVENT_RING_CAP_SLOTS * stride * 4` bytes.
    out.push_str(&format!(
        "    if (slot < {}u) {{\n",
        DEFAULT_EVENT_RING_CAP_SLOTS
    ));
    // Tag + tick header words also go through atomicStore since the
    // binding is `array<atomic<u32>>`.
    out.push_str(&format!(
        "        atomicStore(&{buf}[slot * {stride}u + 0u], {event_id}u);\n"
    ));
    out.push_str(&format!(
        "        atomicStore(&{buf}[slot * {stride}u + 1u], tick);\n"
    ));
    for line in &field_writes {
        // Each `field_writes` entry already starts with 4-space
        // indent; bump to 8 for the nested-if scope.
        out.push_str(&format!("    {line}\n"));
    }
    out.push_str("    }\n");
    out.push_str("}");
    Ok(out)
}

/// Default event-ring slot capacity — 65 536 events per tick. The
/// runtime sizes the `event_ring` buffer to `cap * stride * 4` bytes;
/// the WGSL emitter bounds-checks `slot < cap` to silently drop
/// overflow producers. A future tunable would thread this through the
/// per-rule cfg uniform.
const DEFAULT_EVENT_RING_CAP_SLOTS: u32 = 65_536;

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
    let mut hoistable: std::collections::BTreeMap<u32, Vec<&CgStmt>> =
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
            ..
        } = stmt_node
        {
            if expr_is_pure_for_hoisting(*init, ctx)
                && expr_is_pure_for_hoisting(*projection, ctx)
            {
                hoistable.entry(*radius_cells).or_default().push(stmt_node);
                continue;
            }
        }
        residual.push(idx);
    }

    let mut parts: Vec<String> = Vec::new();
    // Emit the fused walks first, partitioned by radius for
    // deterministic output (BTreeMap iteration is sorted).
    for (_radius, folds) in &hoistable {
        parts.push(emit_fused_for_each_neighbor(folds, ctx)?);
    }
    // Then the residual stmts (everything not hoisted) in their
    // original order. Each is emitted via the per-stmt path which
    // handles its own (non-fused) ForEachNeighbor singleton case.
    for idx in residual {
        let stmt_id = list.stmts[idx];
        parts.push(lower_cg_stmt_to_wgsl(stmt_id, ctx)?);
    }
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
        let body = format!(
            "{head}{{\n\
             \x20   let _self_cell_f = (agent_pos[agent_id] + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
             \x20   let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
             \x20   let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
             \x20   let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
             \x20   let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
             \x20   for (var dz: i32 = -{r}; dz <= {r}; dz = dz + 1) {{\n\
             \x20       for (var dy: i32 = -{r}; dy <= {r}; dy = dy + 1) {{\n\
             \x20           for (var dx: i32 = -{r}; dx <= {r}; dx = dx + 1) {{\n\
             \x20               let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
             \x20               let _start = spatial_grid_starts[_cell];\n\
             \x20               let _end = spatial_grid_starts[_cell + 1u];\n\
             \x20               for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {{\n\
             \x20                   let per_pair_candidate = spatial_grid_cells[_i];\n\
             {updates}\
             \x20               }}\n\
             \x20           }}\n\
             \x20       }}\n\
             \x20   }}\n\
             }}",
            r = radius,
            head = head,
            updates = updates,
        );
        Ok(body)
    }
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
    ctx: &EmitCtx,
) -> Result<String, EmitError> {
    let body_wgsl = lower_cg_stmt_list_to_wgsl(body_list, ctx)?;
    let r = radius_cells as i32;
    // Indent each line of the body so it nests cleanly inside the
    // 4-deep loop chain (3 cell-axis loops + 1 candidate loop). Six
    // levels of 4-space indent → 24 spaces.
    let indented_body = indent_block(&body_wgsl, 6);
    let out = format!(
        "{{\n\
         \x20   let _self_cell_f = (agent_pos[agent_id] + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;\n\
         \x20   let _max_idx = i32(SPATIAL_GRID_DIM) - 1;\n\
         \x20   let _self_cx = clamp(i32(max(_self_cell_f.x, 0.0)), 0, _max_idx);\n\
         \x20   let _self_cy = clamp(i32(max(_self_cell_f.y, 0.0)), 0, _max_idx);\n\
         \x20   let _self_cz = clamp(i32(max(_self_cell_f.z, 0.0)), 0, _max_idx);\n\
         \x20   for (var dz: i32 = -{r}; dz <= {r}; dz = dz + 1) {{\n\
         \x20       for (var dy: i32 = -{r}; dy <= {r}; dy = dy + 1) {{\n\
         \x20           for (var dx: i32 = -{r}; dx <= {r}; dx = dx + 1) {{\n\
         \x20               let _cell = cell_index(_self_cx + dx, _self_cy + dy, _self_cz + dz);\n\
         \x20               let _start = spatial_grid_starts[_cell];\n\
         \x20               let _end = spatial_grid_starts[_cell + 1u];\n\
         \x20               for (var _i: u32 = _start; _i < _end; _i = _i + 1u) {{\n\
         \x20                   let per_pair_candidate = spatial_grid_cells[_i];\n\
         {indented_body}\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20   }}\n\
         }}",
        r = r,
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
        let mut prog = empty_prog();
        let cases = [
            (
                RngPurpose::Action,
                "per_agent_u32(seed, agent_id, tick, \"action\")",
            ),
            (
                RngPurpose::Sample,
                "per_agent_u32(seed, agent_id, tick, \"sample\")",
            ),
            (
                RngPurpose::Shuffle,
                "per_agent_u32(seed, agent_id, tick, \"shuffle\")",
            ),
            (
                RngPurpose::Conception,
                "per_agent_u32(seed, agent_id, tick, \"conception\")",
            ),
        ];
        for (purpose, expected) in cases {
            let id = push_expr(
                &mut prog,
                CgExpr::Rng {
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
                record_stride_u32: 10,
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
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 0u], 7u);"),
            "expected tag (event id 7) write; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 2u], bitcast<u32>(agent_hp[agent_id]));"),
            "expected hp f32 bitcast write at offset 2; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 3u], bitcast<u32>(0.0));"),
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
    /// With the today-default layout (stride=10, header=2,
    /// buffer="event_ring") and a `target` field at payload offset 1
    /// typed as `AgentId`, the WGSL renders to
    /// `event_ring[event_idx * 10u + 3u]`.
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
                record_stride_u32: 10,
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
        assert_eq!(wgsl, "event_ring[event_idx * 10u + 3u]");
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
                record_stride_u32: 10,
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
        assert_eq!(wgsl, "bitcast<f32>(event_ring[event_idx * 10u + 4u])");
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
    /// `word_offset_in_payload=4` (stride=10), the first base is
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
                record_stride_u32: 10,
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
            "vec3<f32>(bitcast<f32>(event_ring[event_idx * 10u + 6u]), bitcast<f32>(event_ring[event_idx * 10u + 7u]), bitcast<f32>(event_ring[event_idx * 10u + 8u]))"
        );
    }

    /// `Bool`-typed `EventField` emits a `(... != 0u)` predicate form.
    /// The payload word is u32 on the GPU side; non-zero u32 reads as
    /// `true`. With `header_word_count=2` and a Bool field at
    /// `word_offset_in_payload=0` (stride=10), the read lands at offset
    /// `2`, producing `(event_ring[event_idx * 10u + 2u] != 0u)`.
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
                record_stride_u32: 10,
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
        assert_eq!(wgsl, "(event_ring[event_idx * 10u + 2u] != 0u)");
    }

    /// `I32`-typed `EventField` emits a `bitcast<i32>` access. The
    /// payload word is u32 on the GPU side; `bitcast<i32>` reinterprets
    /// the bit pattern as the typed signed int — same shape
    /// `pack_event` writes via `i32::to_ne_bytes`-style reinterpretation
    /// on the CPU. With `header_word_count=2` and an I32 field at
    /// `word_offset_in_payload=3` (stride=10), the read lands at offset
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
                record_stride_u32: 10,
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
        assert_eq!(wgsl, "bitcast<i32>(event_ring[event_idx * 10u + 5u])");
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
                record_stride_u32: 10,
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
            wgsl.contains("if (slot < 65536u)"),
            "expected slot bounds check; got:\n{wgsl}"
        );
        // Tag write at offset 0 (event_id is 1) via atomicStore.
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 0u], 1u);"),
            "expected tag write at offset 0; got:\n{wgsl}"
        );
        // Tick write at offset 1.
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 1u], tick);"),
            "expected tick write at offset 1; got:\n{wgsl}"
        );
        // by AgentId at payload offset 0 (header+0 = 2).
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 2u], (agent_id));"),
            "expected `by` at offset 2; got:\n{wgsl}"
        );
        // prey AgentId at payload offset 1 (header+1 = 3).
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 3u], (agent_id));"),
            "expected `prey` at offset 3; got:\n{wgsl}"
        );
        // Vec3 pos with bitcast<u32>(.x/.y/.z) at offsets 4/5/6.
        assert!(
            wgsl.contains("bitcast<u32>(_emit_v_1_2.x)"),
            "expected vec3 .x bitcast; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 4u], bitcast<u32>(_emit_v_1_2.x));"),
            "expected vec3 .x at offset 4; got:\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[slot * 10u + 6u], bitcast<u32>(_emit_v_1_2.z));"),
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
        let body_stmt = push_stmt(
            &mut prog,
            CgStmt::ForEachNeighborBody {
                binder: LocalId(7),
                body: inner_list,
                radius_cells: 1,
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
}
