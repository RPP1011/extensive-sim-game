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
            event_ring_atomic_loads: std::cell::Cell::new(false),
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
        DataHandle::AbilityRegistryColumn { column } => {
            // The dispatcher kernel reads via `ability_registry_<column>[i]`.
            // Stable per-column WGSL identifier so the BGL composer's
            // structural binding-name pass (in `cg/emit/kernel.rs`) lines
            // up against the same string the body references.
            format!("ability_registry_{}", ability_registry_column_token(*column))
        }
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
        // WGSL native scalar conversion `f32(<u32-or-i32>)`. Emitted
        // as-is so `AsF32(U32)` / `AsF32(I32)` produce `f32(<arg>)`.
        // The `NumericTy` payload is informational at IR level (it
        // pins the source type for typing); WGSL infers the source
        // from the argument's type.
        AsF32(_) => "f32".to_string(),
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
            // Gap #A close (stdlib_math_probe, 2026-05-04): WGSL has
            // `log` (natural log) and `log2` natively, but no `log10`.
            // Emit the math identity `log2(x) / log2(10.0)` inline so
            // no kernel prelude is required. `log2(10.0)` is a
            // constant — WGSL's optimiser folds the divisor.
            if matches!(fn_id, BuiltinId::Log10) {
                debug_assert_eq!(parts.len(), 1, "log10 takes one arg");
                return Ok(format!("(log2({}) / log2(10.0))", parts[0]));
            }
            Ok(format!("{}({})", builtin_name(*fn_id), parts.join(", ")))
        }
        CgExpr::Rng { purpose, ty: _ } => {
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
            let raw = format!(
                "per_agent_u32(seed, agent_id, tick, {}u)",
                purpose.wgsl_id()
            );
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
            let read_word = |off: u32| -> String {
                if ctx.event_ring_atomic_loads.get() {
                    format!("atomicLoad(&{}[event_idx * {}u + {}u])", buf, stride, off)
                } else {
                    format!("{}[event_idx * {}u + {}u]", buf, stride, off)
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
                        None => true,
                    };
                    let use_atomic_add = matches!(
                        view_result_ty,
                        Some(crate::cg::expr::CgTy::U32)
                    ) && matches!(
                        fold_op,
                        Some(crate::cg::program::ViewFoldOp::Add)
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
        CgStmt::ApplyAbility { ability, caster, target, with_aoe_dispatch: _with_aoe_dispatch } => {
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
            // **Path B (GPU AOE multi-target) — BGL opt-in landed
            // 2026-05-07; emit-side TODO remains.** The
            // `with_aoe_dispatch` flag on `CgStmt::ApplyAbility` is
            // now plumbed through lowering — every fixture's flag
            // value is read here as `_with_aoe_dispatch`. When the
            // flag is `true`, the WGSL emit will gate the spatial
            // walk + multi-target chronicle write on
            // `area_kinds[i] == 0u` (Circle); when `false`, it
            // emits the existing single-target chain. **The walk
            // itself is not yet wired** — every fixture today
            // (smoke + production runtimes alike) has the flag at
            // its default `false`, so the dispatcher unconditionally
            // emits the single-target chain. Wiring the AOE walk +
            // surfacing the spatial reads via
            // `wire_apply_ability_aoe_reads` is the next slice
            // (Path B emit). The CPU oracle (`apply_program_aoe`)
            // expands Circle slots via `state.spatial().within_radius`
            // and emits one ApplyEvent per in-circle target. The
            // analogous GPU shape is to wrap the `if (when_passes)
            // {…}` block below in a per-target loop:
            //   1. Read `area_kinds[effect_base + i]`. If sentinel
            //      (0xFFu) or non-zero → fall through to single-target
            //      (existing chain executes with the cast's
            //      `target_slot`).
            //   2. If 0u (Circle): read `area_args[(effect_base+i)*4]`
            //      as radius. Compute `aoe_center = agent_pos[target_slot]`.
            //      Walk the 27-cell neighborhood: for each cell,
            //      iterate `_start..end = spatial_grid_starts[cell..+1]`,
            //      bind `let candidate = spatial_grid_cells[_i];`,
            //      compute `let _d = agent_pos[candidate] - aoe_center;`,
            //      gate on `dot(_d, _d) <= radius*radius`, then run the
            //      arm chain inside a `{ let target_slot = candidate; … }`
            //      block to shadow the outer `target_slot` for the
            //      chronicle writes.
            //   3. P11 sort: GPU's atomicAdd ring claim does NOT
            //      preserve AgentId order. The CPU oracle sorts by
            //      AgentId ascending; the parity comparison sorts both
            //      sides post-readback (already done in
            //      `parity_apply_program_sweep::canonicalize`).
            // The shape is straightforward; the blocker is the
            // **BGL composer + scheduler ripple**. Adding `agent_pos`
            // + `spatial_grid_starts/_cells/_offsets` + `area_kinds`
            // + `area_args` reads to this op auto-fires the five
            // build-hash phases (see
            // `collect_required_spatial_kinds` in driver.rs) in EVERY
            // `apply_ability`-using fixture. Three production runtimes
            // (`boss_fight_runtime`, `duel_abilities_runtime`,
            // `tactical_squad_5v5_runtime`) currently bind NO spatial
            // buffers; force-firing the build phases would require
            // them to (a) allocate ~1.4 MB of spatial buffers per
            // fixture, (b) uphold an `agent_pos` SoA contract many
            // don't keep populated for `caster_slot` indexing today.
            // The slice's brief calls a "STOP and document" outcome
            // for fundamental dispatcher refactoring; that's where
            // we landed for this iteration. Next slice should gate
            // the AOE emit on a per-fixture build-time `AoeOpts`
            // flag so opt-in fixtures (smoke runtime first) ship the
            // walk + bindings while production runtimes preserve
            // their zero-spatial-overhead BGL until they're ready
            // to opt in.
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
            let primary_arm_chain = emit_chronicle_arm_chain("        ", "scale_bonus");
            let nested_arm_chain = emit_chronicle_arm_chain("            ", "nested_scale_bonus");
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
            let body = format!(
                "// #136 apply_ability dispatcher (slice β step 2)\n\
                 // Wave 1.5#4 GPU wire-up: per-effect slot reads\n\
                 // `scaling_stat_refs` + `scaling_percents` SoA + per-stat\n\
                 // agent SoA at `caster_slot` to compute the additive\n\
                 // `scale_bonus = Σ percent * caster_stat`. Mirrors the\n\
                 // CPU oracle in `engine::ability::apply::apply_program`\n\
                 // (sums `inner.iter().map(|s| s.percent * stats.get(s.stat_ref))`\n\
                 // — same iteration order j=0 then j=1 per P11 reduction\n\
                 // ordering). AbilityPower(tag=1) returns 0.0 — no agent\n\
                 // SoA slot for it today (LoL-only stat).\n\
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
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 0u: {{ stat_v = agent_attack_damage[caster_slot]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 1u: {{ stat_v = 0.0; }} // AbilityPower — no agent SoA slot\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 2u: {{ stat_v = agent_max_hp[caster_slot]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 3u: {{ stat_v = agent_hp[caster_slot]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 4u: {{ stat_v = agent_armor[caster_slot]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 5u: {{ stat_v = agent_magic_resist[caster_slot]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 6u: {{ stat_v = agent_move_speed[caster_slot]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 7u: {{ stat_v = agent_mana[caster_slot]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20default: {{ stat_v = 0.0; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20scale_bonus = scale_bonus + s_pct * stat_v;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Wave 1.5#7 GPU eval: per-effect when-predicate.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Mirrors `apply::evaluate_predicate` (CPU oracle) — same\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// stat dispatch table as the scale_bonus switch above.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Sentinel binder == 0xFF → no predicate (fire).\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20var when_passes: bool = true;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20let pred_binder: u32 = ability_registry_when_pred_binder[effect_base + i];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20if (pred_binder != 0xFFu) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let pred_field: u32   = ability_registry_when_pred_field[effect_base + i];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let pred_op: u32      = ability_registry_when_pred_op[effect_base + i];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let pred_literal: f32 = ability_registry_when_pred_literal[effect_base + i];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20var pred_agent: u32 = caster_slot;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20if (pred_binder == 1u) {{ pred_agent = target_slot; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20var pred_lhs: f32 = 0.0;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20switch (pred_field) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 0u: {{ pred_lhs = agent_attack_damage[pred_agent]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 1u: {{ pred_lhs = 0.0; }} // AbilityPower — no agent SoA slot\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 2u: {{ pred_lhs = agent_max_hp[pred_agent]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 3u: {{ pred_lhs = agent_hp[pred_agent]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 4u: {{ pred_lhs = agent_armor[pred_agent]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 5u: {{ pred_lhs = agent_magic_resist[pred_agent]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 6u: {{ pred_lhs = agent_move_speed[pred_agent]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 7u: {{ pred_lhs = agent_mana[pred_agent]; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20default: {{ pred_lhs = 0.0; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20switch (pred_op) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 0u: {{ when_passes = pred_lhs <  pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 1u: {{ when_passes = pred_lhs <= pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 2u: {{ when_passes = pred_lhs >  pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 3u: {{ when_passes = pred_lhs >= pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 4u: {{ when_passes = pred_lhs == pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20case 5u: {{ when_passes = pred_lhs != pred_literal; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20default: {{ when_passes = false; }}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20if (when_passes && chance_passes) {{\n\
                 {primary_arm_chain}\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Variant 7 (CastAbility) — recursive dispatch. The\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// nested ability_id lives in payload_a; recursing\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// requires either a depth-bounded re-entry into this\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// loop or a separate work queue. Deferred to slice δ.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Wave 1.5#9 nested-effect walk. After the primary's\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// chronicle write resolves, walk up to\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// MAX_NESTED_PER_EFFECT (=2) nested ops on this slot\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// and write a chronicle record per chronicle-bearing\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// kind. Block-scoped so the inner `kind` / `payload_a`\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// / `payload_b` locals don't shadow the primary walk's.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Nested ops carry no scaling slot in the registry today\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// (mirrors `apply.rs`'s `push_effect_event(..., 0.0)` for\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// nested), so `nested_scale_bonus` is forced to 0.0.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20// Wave 1.5#7: nested loop INSIDE the `if (when_passes)` block.\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20let nested_slot_base: u32 = nested_base + i * 2u;\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20for (var j: u32 = 0u; j < 2u; j = j + 1u) {{\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let kind: u32 = ability_registry_nested_effect_kinds[nested_slot_base + j];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20if (kind == 0xFFu) {{ continue; }} // EFFECT_KIND_EMPTY\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let payload_a: u32 = ability_registry_nested_effect_payload_a[nested_slot_base + j];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let payload_b: u32 = ability_registry_nested_effect_payload_b[nested_slot_base + j];\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20let nested_scale_bonus: f32 = 0.0;\n\
                 {nested_arm_chain}\
                 \x20\x20\x20\x20\x20\x20\x20\x20}}\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20}} // end if (when_passes && chance_passes)\n\
                 \x20\x20\x20\x20}}\n\
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

    Ok(emit_chronicle_append_skeleton(
        event_id,
        buf,
        stride,
        fields.len(),
        &field_writes,
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
pub(crate) fn emit_chronicle_append_skeleton(
    event_id: u32,
    buf: &str,
    stride: u32,
    field_count: usize,
    field_writes: &[String],
) -> String {
    let mut out = String::new();
    out.push_str(&format!("// emit event#{event_id} ({field_count} fields)\n"));
    out.push_str("{\n");
    out.push_str("    let slot = atomicAdd(&event_tail[0], 1u);\n");
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
    out.push_str("    }\n");
    out.push_str("}");
    out
}

/// Default event-ring slot capacity — 65 536 events per tick. The
/// runtime sizes the `event_ring` buffer to `cap * stride * 4` bytes;
/// the WGSL emitter bounds-checks `slot < cap` to silently drop
/// overflow producers. A future tunable would thread this through the
/// per-rule cfg uniform.
const DEFAULT_EVENT_RING_CAP_SLOTS: u32 = 65_536;

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
    // ordinals via `kind == 23u | 25u | 26u | 31u`. Summon (kind 24)
    // is the only remaining `// TODO slice γ` arm — its multi-spawn
    // semantics need a new dispatch shape (one cast → N entity spawns)
    // and is deferred.
    (23, 58), // EffectOp::Buff       → EventKindId::EffectBuffApplied
    (25, 59), // EffectOp::Harvest    → EventKindId::EffectHarvestApplied
    (26, 60), // EffectOp::PlaceVoxel → EventKindId::EffectPlaceVoxelApplied
    (31, 61), // EffectOp::Reflect    → EventKindId::EffectReflectApplied
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
/// `target_slot` as outer-scope WGSL identifiers and writes
/// chronicle records via `atomicAdd(&event_tail[0], 1u)` +
/// `atomicStore(&event_ring[...])` per chronicle-bearing variant.
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
fn emit_chronicle_arm_chain(indent: &str, scale_bonus_var: &str) -> String {
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

    let i4  = indent;                   // arm `if`/`else if` lines
    let i8  = format!("{i4}    ");      // body of arm
    let i12 = format!("{i4}        ");  // inside chronicle-write `{`
    let i16 = format!("{i4}            "); // inside `if (_slot < 65536u)`

    let mut s = String::new();

    // Damage = 0 → 26
    s.push_str(&format!("{i4}// Damage = 0 → EventKindId::EffectDamageApplied = 26\n"));
    s.push_str(&format!("{i4}if (kind == 0u) {{\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectDamageApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {damage_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Heal = 1 → 27
    s.push_str(&format!("{i4}}} else if (kind == 1u) {{\n"));
    s.push_str(&format!("{i8}// Heal = 1 → EventKindId::EffectHealApplied = 27\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectHealApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {heal_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Shield = 2 → 28
    s.push_str(&format!("{i4}}} else if (kind == 2u) {{\n"));
    s.push_str(&format!("{i8}// Shield = 2 → EventKindId::EffectShieldApplied = 28\n"));
    s.push_str(&format!("{i8}let amount: f32 = bitcast<f32>(payload_a) + {scale_bonus_var};\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectShieldApplied (caster_slot + target_slot)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {shield_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {stun_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (expires_at_tick));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {slow_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 5u], bitcast<u32>(factor_q8));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {root_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (expires_at_tick));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {silence_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (expires_at_tick));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {fear_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (expires_at_tick));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {taunt_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (expires_at_tick));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {dash_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], bitcast<u32>(distance));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {blink_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], bitcast<u32>(distance));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {knockback_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(distance));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {pull_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(distance));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {stealth_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (payload_a));\n"));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Charm = 28 → 55
    s.push_str(&format!("{i4}}} else if (kind == 28u) {{\n"));
    s.push_str(&format!("{i8}// Charm = 28 → EventKindId::EffectCharmApplied = 55\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); target-cast charm\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectCharmApplied (caster_slot + target_slot + duration_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {charm_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Grounded = 29 → 56
    s.push_str(&format!("{i4}}} else if (kind == 29u) {{\n"));
    s.push_str(&format!("{i8}// Grounded = 29 → EventKindId::EffectGroundedApplied = 56\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); target-cast grounded\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectGroundedApplied (caster_slot + target_slot + duration_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {grounded_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    // Suppress = 30 → 57
    s.push_str(&format!("{i4}}} else if (kind == 30u) {{\n"));
    s.push_str(&format!("{i8}// Suppress = 30 → EventKindId::EffectSuppressApplied = 57\n"));
    s.push_str(&format!("{i8}// payload_a = duration_ticks (u32); target-cast suppress\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectSuppressApplied (caster_slot + target_slot + duration_ticks)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {suppress_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (payload_a));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {transfer_gold_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount_i32));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {modify_standing_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(delta_i32));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {execute_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(hp_threshold));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {self_damage_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {life_steal_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 5u], bitcast<u32>(fraction_q8));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {damage_modify_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (expires_at_tick));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 5u], bitcast<u32>(multiplier_q8));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {damage_over_time_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 5u], (payload_b));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {heal_over_time_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 5u], (payload_b));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {timed_shield_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 5u], (payload_b));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {buff_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 5u], (payload_b));\n"));
    s.push_str(&format!("{i12}}}\n"));
    s.push_str(&format!("{i8}}}\n"));

    s.push_str(&format!("{i4}}} else if (kind == 24u) {{\n"));
    s.push_str(&format!("{i8}// Summon: payload_a = template_hash (u32),\n"));
    s.push_str(&format!("{i8}// payload_b = (count in high byte | lifetime in low 24 bits)\n"));
    s.push_str(&format!("{i8}let summon_count: u32 = (payload_b >> 24u) & 0xFFu;\n"));
    s.push_str(&format!("{i8}let summon_lifetime: u32 = payload_b & 0x00FFFFFFu;\n"));
    s.push_str(&format!("{i8}// TODO slice γ: chronicle_append_summon(caster, payload_a, summon_count, summon_lifetime);\n"));
    s.push_str(&format!("{i8}// Deferred — multi-spawn semantics need a new dispatch shape\n"));
    s.push_str(&format!("{i8}// (one cast → N entity spawns); not closed by the slice γ tail.\n"));

    // Harvest = 25 → 59
    s.push_str(&format!("{i4}}} else if (kind == 25u) {{\n"));
    s.push_str(&format!("{i8}// Harvest = 25 → EventKindId::EffectHarvestApplied = 59\n"));
    s.push_str(&format!("{i8}// payload_a = kind_hash (u32 FxHash of resource ident),\n"));
    s.push_str(&format!("{i8}// payload_b = amount (u32, widened from u16 EffectOp side).\n"));
    s.push_str(&format!("{i8}// Caster-self — no target field on engine event.\n"));
    s.push_str(&format!("{i8}// chronicle: emit EffectHarvestApplied (caster_slot + kind_hash + amount)\n"));
    s.push_str(&format!("{i8}{{\n"));
    s.push_str(&format!("{i12}let _slot: u32 = atomicAdd(&event_tail[0], 1u);\n"));
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {harvest_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (payload_b));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {place_voxel_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (payload_a));\n"));
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
    s.push_str(&format!("{i12}if (_slot < 65536u) {{\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 0u], {reflect_event_id}u);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 1u], tick);\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 4u], (payload_a));\n"));
    s.push_str(&format!("{i16}atomicStore(&event_ring[_slot * 10u + 5u], (payload_b));\n"));
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
        // decode on read). Summon (kind 24) keeps the WGSL-side decode
        // because its arm body still emits the local lets for the
        // future chronicle_append_summon call (deferred — see TODO
        // marker on the Summon arm).
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
        // explicit assertions below. Summon (kind 24) is the only
        // remaining `// TODO slice γ` arm — its multi-spawn semantics
        // need a new dispatch shape and is deferred.
        for marker in &[
            "chronicle_append_summon",
        ] {
            assert!(
                wgsl.contains(&format!("TODO slice γ: {marker}")),
                "{marker} arm must keep the TODO marker (deferred);\n{wgsl}"
            );
        }

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
                    "atomicStore(&event_ring[_slot * 10u + 0u], {expected_event_id}u);"
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
                    "atomicStore(&event_ring[_slot * 10u + 0u], {expected_event_id}u);"
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
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 3u], bitcast<u32>(distance));"),
            "Dash/Blink arms must store distance at payload word 1 (ring offset 3);\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(distance));"),
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
                    "atomicStore(&event_ring[_slot * 10u + 0u], {expected_event_id}u);"
                )),
                "{name} arm must store kind={expected_event_id};\n{wgsl}"
            );
        }
        // DoT/HoT/TimedShield: amount at slot 4 (bitcast<u32>(amount)),
        // duration_ticks at slot 5 (raw u32 from payload_b). Pin the
        // duration write — distinct from the q8 / expires_at_tick
        // shapes so it surfaces here on swap regressions.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 5u], (payload_b));"),
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
                    "atomicStore(&event_ring[_slot * 10u + 0u], {expected_event_id}u);"
                )),
                "{name} arm must store kind={expected_event_id};\n{wgsl}"
            );
        }
        // Stealth: duration_ticks at slot 3 (raw u32 from payload_a, no
        // target field). Charm/Grounded/Suppress: duration_ticks at slot
        // 4 (raw u32 from payload_a, target at slot 3). Pin both shapes
        // so a regression that swaps them surfaces here.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 3u], (payload_a));"),
            "Stealth arm must store duration_ticks at payload word 1 \
             (ring offset 3) as raw u32 (= payload_a, no target field);\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 4u], (payload_a));"),
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
        ] {
            // Use snake_case for the marker text; match `chronicle_append_<name>`
            // form. PlaceVoxel needs explicit snake_case.
            let snake = match *name {
                "Buff"       => "buff",
                "Harvest"    => "harvest",
                "PlaceVoxel" => "place_voxel",
                "Reflect"    => "reflect",
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
                    "atomicStore(&event_ring[_slot * 10u + 0u], {expected_event_id}u);"
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
            wgsl.matches("atomicStore(&event_ring[_slot * 10u + 5u], (payload_b));").count() >= 5,
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
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 0u], 26u);"),
            "Damage arm must store kind=26 (EffectDamageApplied);\n{wgsl}"
        );
        // Header tick.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 1u], tick);"),
            "Damage arm must store tick at header word 1;\n{wgsl}"
        );
        // Self-cast caster + target (slice γ uses agent_id for both;
        // explicit caster/target arrives when CgStmt::ApplyAbility
        // grows those fields).
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 2u], (caster_slot));"),
            "Damage arm must store caster=agent_id at payload word 0;\n{wgsl}"
        );
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 3u], (target_slot));"),
            "Damage arm must store target=agent_id at payload word 1 (slice γ self-cast);\n{wgsl}"
        );
        // Amount payload — bitcast f32 → u32.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 4u], bitcast<u32>(amount));"),
            "Damage arm must store amount as bitcast<u32>(f32);\n{wgsl}"
        );
        // Bounds check against DEFAULT_EVENT_RING_CAP_SLOTS.
        assert!(
            wgsl.contains("if (_slot < 65536u) {"),
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
                "atomicStore(&event_ring[_slot * 10u + 0u], {expected_kind_tag}u);"
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
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 5u], bitcast<u32>(factor_q8));"),
            "Slow arm must store factor_q8 at payload word 3 (ring offset 5);\n{wgsl}"
        );
        // LifeSteal's 4-field payload — fraction_q8 lives at payload
        // word 3 (= ring slot offset 5), same shape as Slow.
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 5u], bitcast<u32>(fraction_q8));"),
            "LifeSteal arm must store fraction_q8 at payload word 3 (ring offset 5);\n{wgsl}"
        );
        // DamageModify's 4-field payload — multiplier_q8 lives at
        // payload word 3 (= ring slot offset 5), same shape as Slow /
        // LifeSteal. (Fortify verb swap, Task #138 follow-on, mirror
        // of Vampirize.)
        assert!(
            wgsl.contains("atomicStore(&event_ring[_slot * 10u + 5u], bitcast<u32>(multiplier_q8));"),
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
        );

        // Header comment carries event id + field count for capture
        // diagnostics.
        assert!(wgsl.contains("// emit event#26 (2 fields)"),
            "header comment must include id + field count;\n{wgsl}");

        // Slot acquisition via atomicAdd on the canonical event_tail.
        assert!(wgsl.contains("let slot = atomicAdd(&event_tail[0], 1u);"),
            "slot acquisition must use atomicAdd on event_tail[0];\n{wgsl}");

        // Bounds check against DEFAULT_EVENT_RING_CAP_SLOTS (65536).
        assert!(wgsl.contains("if (slot < 65536u) {"),
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
        let wgsl = emit_chronicle_append_skeleton(2, "ring", 2, 0, &[]);
        assert!(wgsl.contains("atomicAdd(&event_tail[0], 1u);"));
        assert!(wgsl.contains("atomicStore(&ring[slot * 2u + 0u], 2u);"));
        assert!(wgsl.contains("atomicStore(&ring[slot * 2u + 1u], tick);"));
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
                25 => EffectOp::Harvest    { kind_hash: 0xCAFEBABE, amount: 5 },
                26 => EffectOp::PlaceVoxel { kind_hash: 0xFACEFEED },
                31 => EffectOp::Reflect    { duration_ticks: 50, fraction_q8: 64 },
                _ => panic!("test only covers chronicle-bearing variants 0..=6 + 8..=15 + 16 + 17 + 18 + 19 + 20..=22 + 27..=30 + 23/25/26/31"),
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
                25 => EngineEventKindId::EffectHarvestApplied        as u32,
                26 => EngineEventKindId::EffectPlaceVoxelApplied     as u32,
                31 => EngineEventKindId::EffectReflectApplied        as u32,
                _ => panic!("test only covers chronicle-bearing variants 0..=6 + 8..=15 + 16 + 17 + 18 + 19 + 20..=22 + 27..=30 + 23/25/26/31"),
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
        assert_eq!(event_kind_id_for_effect_kind(7), None,
            "CastAbility (recursive dispatch) has no chronicle kind");
        assert_eq!(event_kind_id_for_effect_kind(24), None,
            "Summon (multi-spawn) has no chronicle counterpart today (deferred)");
    }

    #[test]
    fn effect_kind_to_event_kind_map_covers_chronicle_bearing_variants_only() {
        // 30 chronicle-bearing variants today — Damage/Heal/Shield/Stun/
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
        // + Buff/Harvest/PlaceVoxel/Reflect (slice γ tail — closes 4 of
        // the 5 remaining `// TODO slice γ` arms; Summon kind 24 deferred).
        // If this number changes, either the engine grew a new
        // `EffectXxxApplied` event (in which case the map gets a new
        // entry) or a variant lost its chronicle counterpart (in which
        // case the map drops an entry). Pin the count so the gap between
        // source-of-truths is loud.
        assert_eq!(
            EFFECT_KIND_TO_EVENT_KIND_ID.len(), 30,
            "EFFECT_KIND_TO_EVENT_KIND_ID should cover exactly the 30 \
             chronicle-bearing variants today; if you added or removed an \
             entry, update this assertion (and the slice γ wire-up that \
             consumes the new entry)"
        );
    }
}
