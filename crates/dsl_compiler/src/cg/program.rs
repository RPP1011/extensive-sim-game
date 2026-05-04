//! `CgProgram` — top-level container for the Compute-Graph IR.
//!
//! `CgProgram` owns every arena the IR layer indexes into: the op
//! arena, the expression arena, the statement arena, and the statement
//! list arena. It also owns the [`Interner`] (stable id → human-readable
//! name lookup, for diagnostics) and the typed [`CgDiagnostic`] log
//! produced by lowering and well-formedness passes.
//!
//! Every consumer of the IR — well-formed checks (Task 1.6), HIR/MIR
//! lowering, schedule synthesis, emit — walks a `CgProgram`. The struct
//! is the single point of truth for "what compute did the DSL produce
//! this build?".
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 1.5, for the design rationale.
//!
//! # Construction
//!
//! Build a program with [`CgProgramBuilder`]. The builder allocates ids
//! contiguously from zero and validates every reference at insertion
//! time — adding a node that references an out-of-range id returns a
//! typed [`BuilderError`], not a runtime panic at consumption time.
//! The arenas on [`CgProgram`] are `pub` so passes can iterate them
//! without an accessor wall, but **direct mutation of the arenas
//! bypasses the builder's invariants** — only mutate via the builder
//! during construction, or via [`ComputeOp::record_read`] /
//! [`ComputeOp::record_write`] (the documented post-construction seams)
//! during lowering.
//!
//! # `Vec<CgStmt>` arena
//!
//! The plan body lists four arenas (`ops`, `exprs`, `stmt_lists`,
//! diagnostics) but `CgStmtList` references statements by [`CgStmtId`]
//! which indexes into a `Vec<CgStmt>`. The plan's omission of the
//! statement arena is editorial — it is required for the program to
//! resolve `CgStmtListId → CgStmtList → CgStmtId → CgStmt`. We add the
//! `stmts: Vec<CgStmt>` arena here and document its presence at the
//! field level.
//!
//! # `record_read` / `record_write`
//!
//! Lowering passes that need to inject a registry-resolved binding
//! (source event ring read for a [`DispatchShape::PerEvent`] op, ring
//! write for an `Emit`, etc.) reach into `program.ops[op_id.0 as
//! usize]` and call [`ComputeOp::record_read`] /
//! [`ComputeOp::record_write`] **after** the op has been added via
//! [`CgProgramBuilder::add_op`]. This is the documented post-
//! construction seam for the parts of the dependency graph the
//! auto-walker can't synthesize structurally.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::data_handle::{
    CgExprId, ConfigConstId, DataHandle, DataHandleNameResolver, EventRingId, IdKind, MaskId,
    ViewId,
};
use super::dispatch::DispatchShape;
use super::expr::{CgExpr, ExprArena};
use super::op::{
    ActionId, ComputeOp, ComputeOpKind, EventKindId, OpId, PhysicsRuleId, ScoringId,
    SpatialQueryKind, Span,
};
use super::stmt::{CgStmt, CgStmtId, CgStmtList, CgStmtListId, StmtArena, StmtListArena};

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Diagnostic severity. The IR layer only emits non-fatal kinds —
/// errors that block compilation are returned as `Result<_, Vec<…>>`
/// from the well-formed pass (Task 1.6) rather than logged here.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Information — lowering observed a structural curiosity worth
    /// surfacing, but compilation proceeds normally.
    Note,
    /// Warning — lowering detected a likely defect (orphaned emit,
    /// unread view) that does not block compilation.
    Warning,
    /// Lint — style or efficiency suggestion. Currently unused; kept
    /// distinct from `Warning` so future fusion-suggestion passes have
    /// a non-warning channel.
    Lint,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Note => f.write_str("note"),
            Severity::Warning => f.write_str("warning"),
            Severity::Lint => f.write_str("lint"),
        }
    }
}

// ---------------------------------------------------------------------------
// CgDiagnosticKind — typed diagnostic payloads
// ---------------------------------------------------------------------------

/// Typed payload for a [`CgDiagnostic`].
///
/// Each variant captures the structural information the diagnostic
/// reports — no `String` reasons. New diagnostic kinds add a variant
/// here.
///
/// **Production of these diagnostics is deferred to the well-formed
/// pass (Task 1.6) and lowering passes (Phase 2);** Task 1.5 only
/// defines the type and the storage. The [`CgProgramBuilder::add_diagnostic`]
/// entry point exists so those later passes have a place to push.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CgDiagnosticKind {
    /// An `Emit` statement references an event kind that no event-ring
    /// binding ever reads. Flagged for lowering review — likely a
    /// dead-end emit.
    OrphanedEmit { event: EventKindId },
    /// A view storage slot is read but never written. Likely a fold
    /// rule was forgotten.
    UnreadView { view: ViewId },
    /// A mask is computed but never read. Dead op — fusion / DCE
    /// candidate.
    UnusedMask { mask: MaskId },
    /// Multiple ops claim to write the same `DataHandle` without an
    /// explicit ordering. Schedule synthesis will need to serialize
    /// these.
    WriteConflict {
        handle: DataHandle,
        ops: Vec<OpId>,
    },
    /// An `event <Name> { ... }` declaration that no rule body ever
    /// emits via `emit <Name> { ... }`. Soft warning — non-emitted
    /// events are sometimes intentional (placeholder for staged work,
    /// declared-then-unblocked-later), so the gate is non-fatal.
    /// Surfaced so the trade_market_probe-style "declared `Shipment`
    /// event accepted silently" path no longer compiles silently.
    /// See `docs/superpowers/notes/2026-05-04-trade_market_probe.md`
    /// GAP #6.
    DeclaredEventNeverEmitted { event: EventKindId },
}

impl fmt::Display for CgDiagnosticKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CgDiagnosticKind::OrphanedEmit { event } => {
                write!(f, "orphaned_emit(event=#{})", event.0)
            }
            CgDiagnosticKind::UnreadView { view } => {
                write!(f, "unread_view(view=#{})", view.0)
            }
            CgDiagnosticKind::UnusedMask { mask } => {
                write!(f, "unused_mask(mask=#{})", mask.0)
            }
            CgDiagnosticKind::WriteConflict { handle, ops } => {
                write!(f, "write_conflict(handle={}, ops=[", handle)?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "op#{}", op.0)?;
                }
                f.write_str("])")
            }
            CgDiagnosticKind::DeclaredEventNeverEmitted { event } => {
                write!(f, "declared_event_never_emitted(event=#{})", event.0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CgDiagnostic
// ---------------------------------------------------------------------------

/// A single IR-level diagnostic. Severity + optional source span +
/// typed kind. The `kind` field is the single source of truth for the
/// diagnostic's structural meaning; severity affects only how callers
/// surface the entry.
///
/// `span` is `serde(skip)`'d for the same reason
/// [`super::op::ComputeOp::span`] is — the AST `Span` does not implement
/// `Deserialize`. Round-tripping a diagnostic resets `span` to
/// `None`; structural fields (`severity`, `kind`) round-trip intact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgDiagnostic {
    pub severity: Severity,
    /// Source location, if the diagnostic has a single anchor. Some
    /// diagnostics are program-wide (e.g. a `WriteConflict` spanning
    /// multiple ops) and carry `None`.
    #[serde(skip)]
    pub span: Option<Span>,
    pub kind: CgDiagnosticKind,
}

impl CgDiagnostic {
    /// Construct a diagnostic with no source span.
    pub fn new(severity: Severity, kind: CgDiagnosticKind) -> Self {
        Self {
            severity,
            span: None,
            kind,
        }
    }

    /// Construct a diagnostic anchored at a specific source span.
    pub fn with_span(severity: Severity, span: Span, kind: CgDiagnosticKind) -> Self {
        Self {
            severity,
            span: Some(span),
            kind,
        }
    }
}

impl fmt::Display for CgDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.severity, self.kind)
    }
}

// ---------------------------------------------------------------------------
// Interner — stable id → human-readable name tables
// ---------------------------------------------------------------------------

/// Stable interned ids → human-readable names, used by diagnostics and
/// by [`CgProgram::display_with_names`] to render `DataHandle`s with
/// source-level names instead of opaque `#N` ids.
///
/// Per-id-type tables. Adding a new id type adds a `BTreeMap` field
/// here plus a `get_*_name` / `intern_*_name` pair. `BTreeMap` keys keep
/// the [`fmt::Display`] output deterministic (sorted by id) without an
/// explicit sort step.
///
/// Entries are populated by lowering passes via the builder's
/// `intern_*_name` methods; an id with no entry is rendered in `#N`
/// form by [`CgProgram::display_with_names`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Interner {
    pub views: BTreeMap<u32, String>,
    pub masks: BTreeMap<u32, String>,
    pub scorings: BTreeMap<u32, String>,
    pub physics_rules: BTreeMap<u32, String>,
    pub event_kinds: BTreeMap<u32, String>,
    pub actions: BTreeMap<u32, String>,
    pub config_consts: BTreeMap<u32, String>,
    pub event_rings: BTreeMap<u32, String>,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_view_name(&self, id: ViewId) -> Option<&str> {
        self.views.get(&id.0).map(String::as_str)
    }
    pub fn get_mask_name(&self, id: MaskId) -> Option<&str> {
        self.masks.get(&id.0).map(String::as_str)
    }
    pub fn get_scoring_name(&self, id: ScoringId) -> Option<&str> {
        self.scorings.get(&id.0).map(String::as_str)
    }
    pub fn get_physics_rule_name(&self, id: PhysicsRuleId) -> Option<&str> {
        self.physics_rules.get(&id.0).map(String::as_str)
    }
    pub fn get_event_kind_name(&self, id: EventKindId) -> Option<&str> {
        self.event_kinds.get(&id.0).map(String::as_str)
    }
    pub fn get_action_name(&self, id: ActionId) -> Option<&str> {
        self.actions.get(&id.0).map(String::as_str)
    }
    pub fn get_config_const_name(&self, id: ConfigConstId) -> Option<&str> {
        self.config_consts.get(&id.0).map(String::as_str)
    }
    pub fn get_event_ring_name(&self, id: EventRingId) -> Option<&str> {
        self.event_rings.get(&id.0).map(String::as_str)
    }
}

// ---------------------------------------------------------------------------
// Event payload layout — schema for `CgExpr::EventField`
// ---------------------------------------------------------------------------

/// Per-event-kind payload layout. Drives the WGSL emit for
/// [`CgExpr::EventField`]: the emitter reads
/// `(record_stride_u32, header_word_count, buffer_name, fields)`
/// from this struct to produce `event_ring[event_idx * stride + header
/// + offset]` accesses.
///
/// **Forward-compat with per-kind ring fanout.** Today every kind
/// shares one ring (`event_ring`, stride 10 = 2 header + 8 payload,
/// sized for `AgentMoved` / `AgentFled`). When an event variant blows
/// past 8 payload words (e.g., `AgentSpawned` with template +
/// equipment list, `EffectAreaApplied` with hit_targets[N],
/// `MemoryRecorded` with embedding[16]), the runtime is expected to
/// fanout to per-kind rings. The same emit then produces
/// `event_ring_<kind>[event_idx * <kind_stride> + header + offset]`
/// purely by the schema returning per-kind `record_stride_u32` and
/// `buffer_name` — no IR shape change required.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLayout {
    /// u32 words per record. Today: 10 for every kind (= 2 header + 8
    /// payload). Future per-kind ring fanout: per-kind value.
    pub record_stride_u32: u32,
    /// Header word count (kind tag + tick). Constant 2 across runtime
    /// strategies — kept on the layout struct for symmetry and so a
    /// future header-shape change is a single-edit migration.
    pub header_word_count: u32,
    /// WGSL identifier for the storage buffer this kind reads from.
    /// Today: "event_ring" for every kind. Future: per-kind names like
    /// "event_ring_AgentMoved".
    pub buffer_name: String,
    /// Field name → offset within payload (u32 words from start of
    /// payload, NOT including header). `BTreeMap` keys keep the
    /// `Display` / serde output deterministic across runs.
    pub fields: BTreeMap<String, FieldLayout>,
}

impl EventLayout {
    /// Return the GPU-representable fields in declaration order.
    /// Sort key is `word_offset_in_payload`: `populate_event_kinds`
    /// assigns offsets cumulatively as fields are walked, so sorting
    /// by offset reconstructs the original AST field order.
    /// Non-GPU-representable fields (Strings, lists) are skipped at
    /// layout-population time and never land here.
    ///
    /// Used by [`crate::cg::emit::wgsl_body::lower_emit_to_wgsl`] to
    /// resolve `EventField { event, index }` (the
    /// declaration-ordered index used in `CgStmt::Emit`) back to a
    /// concrete `(name, FieldLayout)` pair without needing a separate
    /// index → name lookup table.
    pub fn fields_in_declaration_order(&self) -> Vec<(&String, &FieldLayout)> {
        let mut v: Vec<(&String, &FieldLayout)> = self.fields.iter().collect();
        v.sort_by_key(|(_, layout)| layout.word_offset_in_payload);
        v
    }
}

/// Per-field layout entry within an [`EventLayout`].
///
/// `word_offset_in_payload` is the 0-based u32-word offset within the
/// event's payload (NOT including the 2-word header). `word_count` is
/// the number of u32 words this field occupies (1 for scalars, 3 for
/// `Vec3F32`, 2 for `u64`-bearing fields like resource ids). `ty` is
/// the [`super::expr::CgTy`] the WGSL emitter uses to pick the right
/// access shape (raw u32, `bitcast<f32>`, `vec3<f32>(...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldLayout {
    pub word_offset_in_payload: u32,
    pub word_count: u32,
    pub ty: super::expr::CgTy,
}

// ---------------------------------------------------------------------------
// View signature — typed `(args, result)` schema per materialized view
// ---------------------------------------------------------------------------

/// Typed signature for a materialized view: parameter types and result
/// type. Used by [`super::well_formed::check_well_formed`] to resolve
/// the underlying scalar type of a view's `Primary` storage slot — the
/// slot's structural type is `view_key<#N>` (a phantom over the view id),
/// but for the purpose of accepting fold-body assignments the runtime
/// treats `view_key<T>` as `T`. Without this metadata the well-formed
/// check sees `view_key<#N>` and rejects every fold body's `+= scalar`
/// as a type mismatch even though the lowering correctly produced the
/// scalar.
///
/// Populated by [`super::lower::driver::lower_compilation_to_cg`] from
/// the materialized-view registration walk in
/// `populate_view_bodies_and_signatures`. Empty for programs without
/// materialized views.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewSignature {
    /// Argument types in declaration order.
    pub args: Vec<super::expr::CgTy>,
    /// Result type — also the underlying scalar type of the view's
    /// `Primary` storage slot.
    pub result: super::expr::CgTy,
    /// Storage hint copied from the source-level
    /// `@materialized(storage = <hint>)` annotation. Drives the
    /// per-storage-shape WGSL emit + dispatch sizing — most notably
    /// `PairMap` enables the 2-D `(k1 * second_pop + k2)` index in the
    /// fold body's RMW. `None` means the view is not materialized (or
    /// the test fixture didn't populate the hint); structurally
    /// equivalent to "single-key dense_per_agent" for emit purposes.
    #[serde(default)]
    pub storage_hint: Option<CgStorageHint>,
}

/// CG-side storage-hint enum mirroring the discriminator subset of
/// [`dsl_ast::ir::StorageHint`] that the WGSL emit + dispatch sizing
/// path actually consults. Decoupled from the AST type so the cg crate
/// doesn't pull `dsl_ast` into [`CgProgram`] purely for variant labels;
/// the lowering driver translates AST → CG hint at registration time.
///
/// Today the only variant the emit branches on is [`Self::PairMap`]
/// (composes a 2-D index in the fold body's RMW). The other variants
/// reduce to the same single-key emit shape, so we collapse them into
/// [`Self::SingleKey`] until the per-shape WGSL templates land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CgStorageHint {
    /// 2-D pair-keyed map. Fold body's RMW indexes
    /// `view_storage_primary[k1 * cfg.second_key_pop + k2]`; decay
    /// kernel iterates `cfg.slot_count` slots (= `agent_cap *
    /// second_key_pop` for Agent×Agent).
    PairMap,
    /// Every other shape (`PerEntityTopK`, `SymmetricPairTopK`,
    /// `PerEntityRing`, `LazyCached`). Fold body's RMW indexes
    /// `view_storage_primary[k_last]`; decay kernel iterates
    /// `cfg.agent_cap` slots.
    SingleKey,
}

// ---------------------------------------------------------------------------
// Namespace registry — schema for `CgExpr::NamespaceCall` / `NamespaceField`
// ---------------------------------------------------------------------------

/// Registry of stdlib namespace methods + fields with their schema
/// (return type, arg signature, WGSL emit form). The single source of
/// truth for `CgExpr::NamespaceCall` and `CgExpr::NamespaceField`
/// lowering and emission.
///
/// Adding a new namespace method or field is a registry edit (add an
/// entry to `populate_namespace_registry` in `lower::driver`); it is
/// **not** an IR change. The lowering and emit walks consult the
/// registry; only the registry's contents change. Mirrors the
/// schema-driven approach used for [`EventLayout`].
///
/// # Storage forms
///
/// Methods carry a [`MethodDef`] with the WGSL function name; the
/// kernel composer prepends a B1-stub prelude function for each
/// distinct `(ns, method)` pair referenced by the kernel body. Fields
/// carry a [`FieldDef`] whose [`WgslAccessForm`] selects between
/// preamble-locals (`world.tick` → bare `tick` identifier) and uniform
/// fields (`config.combat.engagement_range` → `cfg.engagement_range`)
/// without a new IR variant.
///
/// **Distinct from [`EventLayout`].** That struct is per-event-kind
/// payload schema (offsets within a packed event record); this one is
/// per-namespace symbol schema (return types + emit forms for stdlib
/// callouts). They share no keys and no consumers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespaceRegistry {
    /// Per-namespace definitions. Keyed on the typed [`dsl_ast::ir::NamespaceId`]
    /// (`Agents`, `Query`, `World`, …). Empty for programs that don't
    /// reference any registered namespace symbol.
    pub namespaces: BTreeMap<dsl_ast::ir::NamespaceId, NamespaceDef>,
}

/// Per-namespace symbol table.
///
/// `name` is the source-level namespace identifier (`"agents"`,
/// `"query"`, `"world"`) — kept for diagnostics; the typed
/// [`dsl_ast::ir::NamespaceId`] is the canonical key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespaceDef {
    pub name: String,
    /// Field name → field definition. Keyed on the source-level field
    /// identifier (`"tick"` for `world.tick`).
    pub fields: BTreeMap<String, FieldDef>,
    /// Method name → method definition. Keyed on the source-level
    /// method identifier (`"is_hostile_to"` for
    /// `agents.is_hostile_to(target)`).
    pub methods: BTreeMap<String, MethodDef>,
}

/// Field definition — schema for a `CgExpr::NamespaceField` lowering /
/// emission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldDef {
    /// The result type of the field read.
    pub ty: super::expr::CgTy,
    /// How the WGSL emitter renders the field access.
    pub wgsl_access: WgslAccessForm,
}

/// Method definition — schema for a `CgExpr::NamespaceCall` lowering /
/// emission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MethodDef {
    /// The result type of the call.
    pub return_ty: super::expr::CgTy,
    /// The expected types of the call's positional arguments. Length
    /// is the arity. Used by the lowering for arity validation; the
    /// individual arg types are not enforced today (the type checker
    /// already validated each arg expression's claimed type).
    pub arg_tys: Vec<super::expr::CgTy>,
    /// WGSL function name that the prelude provides (e.g.
    /// `"agents_is_hostile_to"` for `agents.is_hostile_to`). The
    /// kernel composer emits a B1-stub `fn <wgsl_fn_name>(...)` per
    /// distinct method referenced by the kernel body.
    pub wgsl_fn_name: String,
    /// Source-level WGSL stub body returned for this method's B1
    /// implementation (semantic no-op). The stub returns a typed
    /// default (`false` for bool, `fallback` for AgentId, etc.) so the
    /// shader compiles and the kernel runs without panicking; real
    /// semantics arrive in Task 9-11 territory.
    ///
    /// The stub is the **complete function body** including the
    /// surrounding `fn name(arg0: ty0, ...) -> ret { ... }` shape, so
    /// the registry is the single source of truth for both signature
    /// and stub. This avoids duplicating the type table in the kernel
    /// composer.
    pub wgsl_stub: String,
}

/// How the WGSL emitter renders a `CgExpr::NamespaceField` access.
///
/// Adding a new emit form is a one-line variant addition + a matching
/// arm in `lower_cg_expr_to_wgsl`'s `NamespaceField` branch. The
/// `wgsl_access` field on [`FieldDef`] selects the form per-field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WgslAccessForm {
    /// Reference a kernel-preamble-bound local (e.g.
    /// `let tick = cfg.tick;` followed by bare `tick` reads). The
    /// kernel composer is responsible for emitting the preamble
    /// `let` line; this form only names the resulting identifier.
    PreambleLocal { local_name: String },
    /// Reference a field on a uniform-bound struct, rendered as
    /// `<binding>.<field>` (e.g. `cfg.engagement_range`).
    UniformField { binding: String, field: String },
}

// ---------------------------------------------------------------------------
// Entity field catalog (Item + Group)
// ---------------------------------------------------------------------------

/// Per-fixture catalog of `entity X : Item { ... }` and `entity X :
/// Group { ... }` field declarations.
///
/// Items + Groups are catalog-indexed entities (no SoA dispatch,
/// read-only or write-via-events), so their per-field SoA storage
/// is per-fixture rather than per-engine-baked. The catalog resolves
/// opaque [`crate::cg::data_handle::ItemFieldId`] /
/// [`crate::cg::data_handle::GroupFieldId`] pairs to the names + types
/// the WGSL emit and per-fixture runtime use to declare and bind the
/// underlying buffers. Populated by the driver's
/// `populate_entity_field_catalog` from `comp.entities`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityFieldCatalog {
    /// Item entities, keyed by [`dsl_ast::ir::EntityRef`] ordinal.
    pub items: BTreeMap<u16, EntityFieldRecord>,
    /// Group entities, keyed by [`dsl_ast::ir::EntityRef`] ordinal.
    pub groups: BTreeMap<u16, EntityFieldRecord>,
}

/// Per-entity field declaration record. `entity_name` is the
/// source-level entity identifier (`"Coin"`); `fields` lists the
/// declared fields in source order. Both are consumed by emit-time
/// naming (`<entity_snake>_<field_snake>`) and by the per-fixture
/// runtime.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityFieldRecord {
    pub entity_name: String,
    pub fields: Vec<EntityFieldEntry>,
}

/// Per-field entry inside an [`EntityFieldRecord`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityFieldEntry {
    pub name: String,
    pub ty: super::data_handle::AgentFieldTy,
}

impl EntityFieldCatalog {
    /// Resolve an [`crate::cg::data_handle::ItemFieldId`] to its
    /// (entity name, field name, type) triple. Returns `None` if the
    /// catalog has no entry for the given (entity, slot) pair —
    /// indicates a stale handle (the lowering should have surfaced
    /// `UnknownItemField` before producing the handle).
    pub fn resolve_item(
        &self,
        id: super::data_handle::ItemFieldId,
    ) -> Option<(&str, &str, super::data_handle::AgentFieldTy)> {
        let rec = self.items.get(&id.entity)?;
        let fld = rec.fields.get(id.slot as usize)?;
        Some((rec.entity_name.as_str(), fld.name.as_str(), fld.ty))
    }
    /// Resolve a [`crate::cg::data_handle::GroupFieldId`] — same
    /// shape as [`Self::resolve_item`].
    pub fn resolve_group(
        &self,
        id: super::data_handle::GroupFieldId,
    ) -> Option<(&str, &str, super::data_handle::AgentFieldTy)> {
        let rec = self.groups.get(&id.entity)?;
        let fld = rec.fields.get(id.slot as usize)?;
        Some((rec.entity_name.as_str(), fld.name.as_str(), fld.ty))
    }

    /// Resolve an Item field by its source-level name (e.g. `"weight"`).
    /// Returns the (entity ref, slot, primitive type) triple of the
    /// first declaration carrying a field with that name. Used by the
    /// `items.<field>(idx)` lowering arm to fill a typed
    /// [`crate::cg::data_handle::ItemFieldId`] without forcing the
    /// caller to know which entity owns the field.
    ///
    /// The current contract is "field names are unique across all
    /// Item entities" — a future ambiguity (two Item entities both
    /// declaring `weight`) would surface here as the first match
    /// winning; the resolve pass is the right place to forbid that.
    pub fn resolve_item_by_name(
        &self,
        field_name: &str,
    ) -> Option<(u16, u16, super::data_handle::AgentFieldTy)> {
        for (entity_ref, rec) in &self.items {
            for (slot_idx, fld) in rec.fields.iter().enumerate() {
                if fld.name == field_name {
                    return Some((*entity_ref, slot_idx as u16, fld.ty));
                }
            }
        }
        None
    }

    /// Resolve a Group field by its source-level name — same shape as
    /// [`Self::resolve_item_by_name`].
    pub fn resolve_group_by_name(
        &self,
        field_name: &str,
    ) -> Option<(u16, u16, super::data_handle::AgentFieldTy)> {
        for (entity_ref, rec) in &self.groups {
            for (slot_idx, fld) in rec.fields.iter().enumerate() {
                if fld.name == field_name {
                    return Some((*entity_ref, slot_idx as u16, fld.ty));
                }
            }
        }
        None
    }
}

/// Helper — record a name in an interner table. Idempotent: passing
/// the same id twice with the same name is a no-op. Conflicting names
/// surface as [`BuilderError::DuplicateInternEntry`]. Used by every
/// `CgProgramBuilder::intern_*_name` method.
fn intern_into(
    table: &mut BTreeMap<u32, String>,
    id_kind: &'static str,
    id: u32,
    name: String,
) -> Result<(), BuilderError> {
    match table.get(&id) {
        Some(prior) if prior != &name => Err(BuilderError::DuplicateInternEntry {
            id_kind,
            id,
            prior: prior.clone(),
            new: name,
        }),
        Some(_) => Ok(()),
        None => {
            table.insert(id, name);
            Ok(())
        }
    }
}

impl fmt::Display for Interner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("interner {\n")?;
        write_table(f, "views", &self.views)?;
        write_table(f, "masks", &self.masks)?;
        write_table(f, "scorings", &self.scorings)?;
        write_table(f, "physics_rules", &self.physics_rules)?;
        write_table(f, "event_kinds", &self.event_kinds)?;
        write_table(f, "actions", &self.actions)?;
        write_table(f, "config_consts", &self.config_consts)?;
        write_table(f, "event_rings", &self.event_rings)?;
        f.write_str("}")
    }
}

fn write_table(
    f: &mut fmt::Formatter<'_>,
    label: &str,
    table: &BTreeMap<u32, String>,
) -> fmt::Result {
    write!(f, "    {}: {{", label)?;
    if table.is_empty() {
        f.write_str("}\n")?;
        return Ok(());
    }
    let mut first = true;
    for (id, name) in table {
        if !first {
            f.write_str(",")?;
        }
        first = false;
        write!(f, " #{} -> \"{}\"", id, name)?;
    }
    f.write_str(" }\n")
}

// ---------------------------------------------------------------------------
// BuilderError
// ---------------------------------------------------------------------------

/// Typed errors returned by [`CgProgramBuilder`]'s `add_*` methods when
/// a node references an id not yet allocated in the relevant arena, or
/// when an argument list itself contains a dangling id.
///
/// Every variant names the offending id; no `String` reasons. The
/// builder's fail-at-insertion-time design means consumers can rely on
/// `program.exprs[id.0 as usize]` succeeding for every id referenced
/// inside `program`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuilderError {
    /// A `CgExprId` referenced in the new node is out of range.
    DanglingExprId { referenced: CgExprId, arena_len: u32 },
    /// A `CgStmtId` referenced in the new node is out of range.
    DanglingStmtId { referenced: CgStmtId, arena_len: u32 },
    /// A `CgStmtListId` referenced in the new node is out of range.
    DanglingStmtListId {
        referenced: CgStmtListId,
        arena_len: u32,
    },
    /// An `intern_*_name` call attempted to overwrite an existing entry
    /// for the same id with a *different* name. Idempotent re-interns
    /// (same id, same name) are accepted silently and never produce
    /// this error. `id_kind` is one of a closed set of `&'static str`
    /// tags (`"view"`, `"mask"`, `"scoring"`, `"physics_rule"`,
    /// `"event_kind"`, `"action"`, `"config_const"`, `"event_ring"`)
    /// — not a free-form string. `prior` and `new` are display values
    /// (the existing and conflicting names); they exist so error
    /// messages can name both names without forcing the caller to
    /// re-look-up the prior entry.
    DuplicateInternEntry {
        id_kind: &'static str,
        id: u32,
        prior: String,
        new: String,
    },
}

impl fmt::Display for BuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuilderError::DanglingExprId {
                referenced,
                arena_len,
            } => write!(
                f,
                "dangling CgExprId(#{}) (expr arena holds {} entries)",
                referenced.0, arena_len
            ),
            BuilderError::DanglingStmtId {
                referenced,
                arena_len,
            } => write!(
                f,
                "dangling CgStmtId(#{}) (stmt arena holds {} entries)",
                referenced.0, arena_len
            ),
            BuilderError::DanglingStmtListId {
                referenced,
                arena_len,
            } => write!(
                f,
                "dangling CgStmtListId(#{}) (stmt-list arena holds {} entries)",
                referenced.0, arena_len
            ),
            BuilderError::DuplicateInternEntry {
                id_kind,
                id,
                prior,
                new,
            } => write!(
                f,
                "duplicate intern entry for {}#{}: prior name {:?}, new name {:?}",
                id_kind, id, prior, new
            ),
        }
    }
}

impl std::error::Error for BuilderError {}

// ---------------------------------------------------------------------------
// ConfigConstValue — typed default for a `config <block>.<field>` literal
// ---------------------------------------------------------------------------

/// Typed payload for a `config <block> { <field>: <ty> = <default>, ... }`
/// declaration's default literal. The variant pins the WGSL scalar type
/// the [`crate::cg::emit::program`] surface emits as
/// `const config_<id>: <ty> = <value>;`. Without this distinction every
/// config const is materialised as `f32`, so a `u32`-declared field that
/// flows into a `u32` ring slot (atomicStore / atomicAdd / atomicOr)
/// would crash the WGSL validator with an `f32 → u32` auto-conversion
/// error. See `docs/superpowers/notes/2026-05-04-trade_market_probe.md`
/// GAP #1.
///
/// `Bool` and `String` defaults parse but currently have no kernel-side
/// emission (no compute kernel reads them); they're tracked here only so
/// the future surface can switch on the same enum.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConfigConstValue {
    /// Unsigned 32-bit integer — emitted as `<n>u`.
    U32(u32),
    /// Signed 32-bit integer — emitted as `<n>i`.
    I32(i32),
    /// Single-precision float — emitted via Rust's `{:?}` debug format
    /// (preserves `1.0` shape rather than `1`).
    F32(f32),
}

impl ConfigConstValue {
    /// WGSL literal form for the constant's RHS, with the scalar-type
    /// suffix (`u`, `i`, or none-with-decimal) baked in. Used by the
    /// kernel composer's `const config_<id>: <ty> = <here>;` emission.
    pub fn wgsl_literal(&self) -> String {
        match self {
            ConfigConstValue::U32(v) => format!("{v}u"),
            ConfigConstValue::I32(v) => format!("{v}i"),
            // `{:?}` keeps the `1.0` form for whole numbers; `{}` would
            // render `1` which WGSL would parse as `i32`.
            ConfigConstValue::F32(v) => format!("{v:?}"),
        }
    }

    /// WGSL scalar type token (`u32` / `i32` / `f32`) for the
    /// `const config_<id>: <here> = <lit>;` declaration.
    pub fn wgsl_scalar_ty(&self) -> &'static str {
        match self {
            ConfigConstValue::U32(_) => "u32",
            ConfigConstValue::I32(_) => "i32",
            ConfigConstValue::F32(_) => "f32",
        }
    }
}

// ---------------------------------------------------------------------------
// CgProgram
// ---------------------------------------------------------------------------

/// Top-level container for the Compute-Graph IR.
///
/// Owns every arena IR-layer passes index into. Construct with
/// [`CgProgramBuilder`]; the builder enforces the indexing invariants
/// every later pass relies on.
///
/// `serde::Deserialize` round-trips structural fields; spans on
/// embedded [`ComputeOp`]s are reset to [`Span::dummy`] (`#[serde(skip,
/// default = "Span::dummy")]`), and spans on [`CgDiagnostic`]s are
/// reset to `None`. Compare on the structural fields, not the spans.
///
/// # Mutation rules
///
/// - During construction, mutate only via [`CgProgramBuilder`].
/// - After construction, the only sanctioned mutations are
///   [`ComputeOp::record_read`] / [`ComputeOp::record_write`] on
///   `program.ops[op_id.0 as usize]`, used by lowering passes to
///   inject registry-resolved bindings the auto-walker can't
///   synthesize.
/// - Direct `program.exprs.push(...)` etc. is allowed only when the
///   caller has manually validated that no existing entry references
///   an id beyond the new arena length — no consumer of the IR
///   defends against arena truncation, so don't shrink the arenas.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CgProgram {
    /// Every compute op the DSL produced. Indexed by [`OpId`].
    pub ops: Vec<ComputeOp>,
    /// Every [`CgExpr`] — flattened arena indexed by [`CgExprId`].
    pub exprs: Vec<CgExpr>,
    /// Every [`CgStmt`] — flattened arena indexed by [`CgStmtId`].
    /// Not listed in the plan body but required for `CgStmtListId →
    /// CgStmtList → CgStmtId → CgStmt` resolution; see the module-
    /// level note on the editorial gap.
    pub stmts: Vec<CgStmt>,
    /// Every [`CgStmtList`] — flattened arena indexed by
    /// [`CgStmtListId`].
    pub stmt_lists: Vec<CgStmtList>,
    /// Stable interned ids → human-readable names (for diagnostics
    /// and pretty-printing).
    pub interner: Interner,
    /// IR-level diagnostics (warnings, lints, fusion suggestions).
    pub diagnostics: Vec<CgDiagnostic>,
    /// Per-event-kind payload layouts. Drives the WGSL emit for
    /// [`super::expr::CgExpr::EventField`]: each entry resolves
    /// `(record_stride_u32, header_word_count, buffer_name,
    /// fields)` for the kind. Populated by the driver's
    /// `populate_event_kinds` (via `LoweringCtx::event_layouts`);
    /// consumed by `lower_cg_expr_to_wgsl`'s `EventField` arm.
    /// Empty for programs that don't have event-pattern bindings.
    pub event_layouts: BTreeMap<u32, EventLayout>,
    /// Stdlib namespace registry — schema for
    /// [`super::expr::CgExpr::NamespaceCall`] and
    /// [`super::expr::CgExpr::NamespaceField`] lowering + emission.
    /// Populated by the driver's `populate_namespace_registry`;
    /// consumed by `lower_cg_expr_to_wgsl`'s namespace-call /
    /// namespace-field arms and by the kernel composer's
    /// prelude-stub emission. Empty for programs that don't
    /// reference any registered namespace symbol.
    pub namespace_registry: NamespaceRegistry,
    /// Per-view typed signature — `view_id → (arg_tys, result_ty)`
    /// for every materialized view registered during lowering. The
    /// result type is also the view's `Primary` storage slot's
    /// underlying scalar type; consulted by
    /// [`super::well_formed::check_well_formed`] when validating
    /// fold-body `Assign(ViewStorage{Primary}, scalar)` shapes (the
    /// view-key strict-equality relaxation rule). Populated by the
    /// driver's `populate_view_bodies_and_signatures` (via
    /// [`super::lower::expr::LoweringCtx::view_signatures`]). Empty
    /// for programs that don't have materialized views.
    pub view_signatures: BTreeMap<u32, ViewSignature>,
    /// Per-`ConfigConstId` literal default value, harvested from the
    /// resolved DSL `config <block> { <field>: <ty> = <default>, ... }`
    /// declarations by the driver's `populate_config_consts`. Populated
    /// in id-allocation order; serialised + emitted as inline WGSL
    /// `const config_<id>: <ty> = <value>;` declarations at the top of
    /// each kernel that references the id. The variant of
    /// [`ConfigConstValue`] picks the WGSL scalar type (`u32` / `i32` /
    /// `f32`) so a `u32`-declared config field flows into a `u32` slot
    /// (atomic store / RMW) without requiring a `bitcast`. The runtime
    /// side stays inert for these consts: they're baked into the WGSL
    /// at compile time rather than passed via a runtime UBO. Per-tick /
    /// runtime-tunable config goes through a future cfg-uniform
    /// extension; today every config field is a compile-time constant.
    pub config_const_values: BTreeMap<u32, ConfigConstValue>,
    /// Per-fixture catalog of `entity X : Item { ... }` and `entity Y :
    /// Group { ... }` field declarations. Resolves opaque
    /// [`crate::cg::data_handle::ItemFieldId`] /
    /// [`crate::cg::data_handle::GroupFieldId`] pairs (entity ref +
    /// slot) to the (entity name, field name, primitive type) triple
    /// the WGSL emit and per-fixture runtime use to name and type the
    /// SoA buffer. Populated by the driver's
    /// `populate_entity_field_catalog`. Empty for fixtures that
    /// declare no Item / Group entities.
    pub entity_field_catalog: EntityFieldCatalog,
}

impl CgProgram {
    /// Construct an empty program. Equivalent to `CgProgram::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Render `handle` consulting `self.interner` — `view[#3]` becomes
    /// `view[standing]` if the interner has a name for `ViewId(3)`,
    /// else stays `view[#3]`. Delegates to [`DataHandle::fmt_with`] so
    /// the rendering shape is single-source: adding a new
    /// `DataHandle` variant updates both
    /// [`DataHandle::Display`] (opaque form) and `display_with_names`
    /// (named form) simultaneously.
    pub fn display_with_names(&self, handle: &DataHandle) -> String {
        format!("{}", DataHandleWithNames(handle, &self.interner))
    }
}

/// The interner serves as a [`DataHandleNameResolver`]: each
/// `IdKind` variant routes to the matching `get_*_name` table.
/// Adding a new `IdKind` variant adds a match arm here so the
/// dispatch is exhaustive.
impl DataHandleNameResolver for Interner {
    fn name_for(&self, kind: IdKind, id: u32) -> Option<&str> {
        match kind {
            IdKind::View => self.get_view_name(ViewId(id)),
            IdKind::Mask => self.get_mask_name(MaskId(id)),
            IdKind::EventRing => self.get_event_ring_name(EventRingId(id)),
            IdKind::ConfigConst => self.get_config_const_name(ConfigConstId(id)),
        }
    }
}

/// Adapter that pairs a [`DataHandle`] with an [`Interner`] so the two
/// can be passed to `format!` / `write!` together. The `Display` impl
/// routes through [`DataHandle::fmt_with`], so the named-rendering path
/// is the single source of truth for handle output.
struct DataHandleWithNames<'a>(&'a DataHandle, &'a Interner);

impl<'a> fmt::Display for DataHandleWithNames<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_with(f, self.1)
    }
}

// --- Arena trait impls --------------------------------------------------

impl ExprArena for CgProgram {
    fn get(&self, id: CgExprId) -> Option<&CgExpr> {
        ExprArena::get(&self.exprs, id)
    }

    /// Refine a `ConfigConst` read's type from the registered
    /// [`ConfigConstValue`] variant. Returns `Some(CgTy::U32)` for a
    /// `u32`-declared field (e.g. `observation_tick_mod: u32 = 3`),
    /// `Some(CgTy::I32)` for an `i32`-declared field, and
    /// `Some(CgTy::F32)` for an `f32`-declared field. Returns `None`
    /// when the id has no entry (driver skipped non-numeric defaults
    /// or hand-built test programs that bypass `set_config_const_value`);
    /// the type checker then falls back to the `data_handle_ty` default.
    /// Closes Gap #3 from
    /// `docs/superpowers/notes/2026-05-04-diplomacy_probe.md` —
    /// `world.tick % config.<ns>.<u32_field>` lowers without a
    /// `BinaryOperandTyMismatch`.
    fn config_const_ty(
        &self,
        id: crate::cg::data_handle::ConfigConstId,
    ) -> Option<crate::cg::expr::CgTy> {
        use crate::cg::expr::CgTy;
        self.config_const_values.get(&id.0).map(|v| match v {
            ConfigConstValue::U32(_) => CgTy::U32,
            ConfigConstValue::I32(_) => CgTy::I32,
            ConfigConstValue::F32(_) => CgTy::F32,
        })
    }
}

impl StmtArena for CgProgram {
    fn get(&self, id: CgStmtId) -> Option<&CgStmt> {
        StmtArena::get(&self.stmts, id)
    }
}

impl StmtListArena for CgProgram {
    fn get(&self, id: CgStmtListId) -> Option<&CgStmtList> {
        StmtListArena::get(&self.stmt_lists, id)
    }
}

// --- Display ------------------------------------------------------------

impl fmt::Display for CgProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("program {\n")?;

        // Diagnostics block
        f.write_str("    diagnostics: [")?;
        if self.diagnostics.is_empty() {
            f.write_str("],\n")?;
        } else {
            f.write_str("\n")?;
            for d in &self.diagnostics {
                writeln!(f, "        {},", d)?;
            }
            f.write_str("    ],\n")?;
        }

        // Interner block — already prints multi-line.
        // Indent each line by 4 spaces.
        let interner_text = format!("{}", self.interner);
        for line in interner_text.lines() {
            writeln!(f, "    {}", line)?;
        }

        // Ops block
        f.write_str("    ops: [")?;
        if self.ops.is_empty() {
            f.write_str("],\n")?;
        } else {
            f.write_str("\n")?;
            for op in &self.ops {
                writeln!(f, "        {},", op)?;
            }
            f.write_str("    ],\n")?;
        }

        f.write_str("}")
    }
}

// ---------------------------------------------------------------------------
// CgProgramBuilder
// ---------------------------------------------------------------------------

/// Constructs a [`CgProgram`] with reference invariants enforced at
/// insertion time.
///
/// Each `add_*` method returns the freshly-allocated id; ids are
/// contiguous starting from zero, in insertion order. Methods that take
/// a node containing references (`add_expr` for a `Binary`, `add_op`
/// for a `MaskPredicate`, …) validate every referenced id against the
/// current arena length before pushing — a dangling id surfaces as a
/// typed [`BuilderError`].
///
/// Lowering passes are the canonical caller. Tests can also build
/// programs directly when exercising consumer code.
pub struct CgProgramBuilder {
    inner: CgProgram,
}

impl Default for CgProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CgProgramBuilder {
    /// Start a new builder with empty arenas, an empty interner, and
    /// no diagnostics.
    pub fn new() -> Self {
        Self {
            inner: CgProgram::new(),
        }
    }

    // --- Interner intern_* methods --------------------------------------

    /// Record a human-readable name for `id` — used by
    /// [`CgProgram::display_with_names`] and by diagnostics.
    ///
    /// Idempotent: passing the same name for the same id twice returns
    /// `Ok(())` without overwriting. Passing a *different* name for an
    /// already-interned id returns
    /// [`BuilderError::DuplicateInternEntry`] — the lowering pass that
    /// triggered the conflict has a name-allocation defect.
    pub fn intern_view_name(
        &mut self,
        id: ViewId,
        name: impl Into<String>,
    ) -> Result<(), BuilderError> {
        intern_into(&mut self.inner.interner.views, "view", id.0, name.into())
    }
    pub fn intern_mask_name(
        &mut self,
        id: MaskId,
        name: impl Into<String>,
    ) -> Result<(), BuilderError> {
        intern_into(&mut self.inner.interner.masks, "mask", id.0, name.into())
    }
    pub fn intern_scoring_name(
        &mut self,
        id: ScoringId,
        name: impl Into<String>,
    ) -> Result<(), BuilderError> {
        intern_into(
            &mut self.inner.interner.scorings,
            "scoring",
            id.0,
            name.into(),
        )
    }
    pub fn intern_physics_rule_name(
        &mut self,
        id: PhysicsRuleId,
        name: impl Into<String>,
    ) -> Result<(), BuilderError> {
        intern_into(
            &mut self.inner.interner.physics_rules,
            "physics_rule",
            id.0,
            name.into(),
        )
    }
    pub fn intern_event_kind_name(
        &mut self,
        id: EventKindId,
        name: impl Into<String>,
    ) -> Result<(), BuilderError> {
        intern_into(
            &mut self.inner.interner.event_kinds,
            "event_kind",
            id.0,
            name.into(),
        )
    }
    pub fn intern_action_name(
        &mut self,
        id: ActionId,
        name: impl Into<String>,
    ) -> Result<(), BuilderError> {
        intern_into(
            &mut self.inner.interner.actions,
            "action",
            id.0,
            name.into(),
        )
    }
    pub fn intern_config_const_name(
        &mut self,
        id: ConfigConstId,
        name: impl Into<String>,
    ) -> Result<(), BuilderError> {
        intern_into(
            &mut self.inner.interner.config_consts,
            "config_const",
            id.0,
            name.into(),
        )
    }
    /// Record the literal default value for a config const id. Used
    /// by the WGSL emit's inline `const config_<id>: <ty> = <v>;`
    /// declarations. The [`ConfigConstValue`] variant pins the WGSL
    /// scalar type (`u32` / `i32` / `f32`) so a u32-declared field
    /// emits as `5u` not `5.0`. Inserts (or overwrites) the entry —
    /// the driver only calls this once per id during
    /// `populate_config_consts`, so overwrites would only happen if a
    /// downstream pass mutates the same id with a new value, which is
    /// intentional last-write semantics.
    pub fn set_config_const_value(&mut self, id: ConfigConstId, value: ConfigConstValue) {
        self.inner.config_const_values.insert(id.0, value);
    }
    pub fn intern_event_ring_name(
        &mut self,
        id: EventRingId,
        name: impl Into<String>,
    ) -> Result<(), BuilderError> {
        intern_into(
            &mut self.inner.interner.event_rings,
            "event_ring",
            id.0,
            name.into(),
        )
    }

    /// Replace the program's [`EntityFieldCatalog`] in one shot. Called
    /// by the driver's `populate_entity_field_catalog` after walking
    /// `comp.entities` for every `EntityRoot::Item` / `EntityRoot::Group`
    /// declaration. Idempotent only with respect to repeated identical
    /// catalogs — last-write-wins; the driver only calls this once.
    pub fn set_entity_field_catalog(&mut self, catalog: EntityFieldCatalog) {
        self.inner.entity_field_catalog = catalog;
    }

    // --- Diagnostic push ------------------------------------------------

    /// Append a diagnostic. Accepted unconditionally — the IR layer
    /// does not validate diagnostic structure (severity + kind are
    /// already typed at the call site).
    pub fn add_diagnostic(&mut self, diag: CgDiagnostic) {
        self.inner.diagnostics.push(diag);
    }

    // --- Arena length helpers (used by validators) ---------------------

    fn expr_len(&self) -> u32 {
        self.inner.exprs.len() as u32
    }
    fn stmt_len(&self) -> u32 {
        self.inner.stmts.len() as u32
    }
    fn list_len(&self) -> u32 {
        self.inner.stmt_lists.len() as u32
    }

    fn check_expr_id(&self, id: CgExprId) -> Result<(), BuilderError> {
        if id.0 < self.expr_len() {
            Ok(())
        } else {
            Err(BuilderError::DanglingExprId {
                referenced: id,
                arena_len: self.expr_len(),
            })
        }
    }

    fn check_stmt_id(&self, id: CgStmtId) -> Result<(), BuilderError> {
        if id.0 < self.stmt_len() {
            Ok(())
        } else {
            Err(BuilderError::DanglingStmtId {
                referenced: id,
                arena_len: self.stmt_len(),
            })
        }
    }

    fn check_list_id(&self, id: CgStmtListId) -> Result<(), BuilderError> {
        if id.0 < self.list_len() {
            Ok(())
        } else {
            Err(BuilderError::DanglingStmtListId {
                referenced: id,
                arena_len: self.list_len(),
            })
        }
    }

    // --- Expr validation -----------------------------------------------

    /// Validate every `CgExprId` reference inside `expr`. Returns the
    /// first dangling id encountered, in left-to-right operand order.
    fn validate_expr_refs(&self, expr: &CgExpr) -> Result<(), BuilderError> {
        match expr {
            CgExpr::Read(_)
            | CgExpr::Lit(_)
            | CgExpr::Rng { .. }
            | CgExpr::AgentSelfId
            | CgExpr::PerPairCandidateId
            | CgExpr::ReadLocal { .. }
            | CgExpr::EventField { .. }
            | CgExpr::NamespaceField { .. } => Ok(()),
            CgExpr::Binary { lhs, rhs, .. } => {
                self.check_expr_id(*lhs)?;
                self.check_expr_id(*rhs)?;
                Ok(())
            }
            CgExpr::Unary { arg, .. } => self.check_expr_id(*arg),
            CgExpr::Builtin { args, .. } => {
                for a in args {
                    self.check_expr_id(*a)?;
                }
                Ok(())
            }
            CgExpr::Select {
                cond, then, else_, ..
            } => {
                self.check_expr_id(*cond)?;
                self.check_expr_id(*then)?;
                self.check_expr_id(*else_)?;
                Ok(())
            }
            CgExpr::NamespaceCall { args, .. } => {
                for a in args {
                    self.check_expr_id(*a)?;
                }
                Ok(())
            }
        }
    }

    // --- Stmt validation -----------------------------------------------

    /// Validate every id reference inside `stmt`. `Assign` walks its
    /// value expression; `Emit` walks each field expression; `If`
    /// walks the condition expression and the then/else list ids;
    /// `Match` walks the scrutinee expression and each arm's body
    /// list id (the typed [`crate::cg::stmt::VariantId`] /
    /// [`crate::cg::stmt::LocalId`] payloads are not arena-relative,
    /// so the builder does not range-check them — the well-formed
    /// pass owns that diagnostic if a registry-resolution defect ever
    /// surfaces).
    fn validate_stmt_refs(&self, stmt: &CgStmt) -> Result<(), BuilderError> {
        match stmt {
            CgStmt::Assign { value, .. } => self.check_expr_id(*value),
            CgStmt::Emit { fields, .. } => {
                for (_, expr_id) in fields {
                    self.check_expr_id(*expr_id)?;
                }
                Ok(())
            }
            CgStmt::If { cond, then, else_ } => {
                self.check_expr_id(*cond)?;
                self.check_list_id(*then)?;
                if let Some(else_id) = else_ {
                    self.check_list_id(*else_id)?;
                }
                Ok(())
            }
            CgStmt::Match { scrutinee, arms } => {
                self.check_expr_id(*scrutinee)?;
                for arm in arms {
                    self.check_list_id(arm.body)?;
                }
                Ok(())
            }
            CgStmt::Let { value, .. } => {
                // The bound value's expression id must already exist
                // in the arena. The `local` and `ty` payloads are
                // arena-independent (typed ids + closed-set CgTy)
                // and need no range check at insertion time.
                self.check_expr_id(*value)
            }
            CgStmt::ForEachAgent {
                init, projection, ..
            }
            | CgStmt::ForEachNeighbor {
                init, projection, ..
            } => {
                // Both child expression ids (the accumulator's initial
                // value and the per-candidate projection added inside
                // the loop) must already exist in the arena. The
                // `acc_local` LocalId is arena-independent; `acc_ty`
                // is closed-set CgTy.
                self.check_expr_id(*init)?;
                self.check_expr_id(*projection)
            }
            CgStmt::ForEachNeighborBody { body, .. } => {
                // The nested body stmt list must already exist; its
                // children (statements + their expression ids) are
                // validated when each was added. The `binder` LocalId
                // is arena-independent and `radius_cells` is a
                // primitive payload.
                self.check_list_id(*body)
            }
        }
    }

    // --- Add_* entry points --------------------------------------------

    /// Push a `CgExpr` and return its freshly-allocated [`CgExprId`].
    /// Validates every child id reference against the current expr
    /// arena length before pushing.
    pub fn add_expr(&mut self, expr: CgExpr) -> Result<CgExprId, BuilderError> {
        self.validate_expr_refs(&expr)?;
        let id = CgExprId(self.expr_len());
        self.inner.exprs.push(expr);
        Ok(id)
    }

    /// Push a `CgStmt` and return its freshly-allocated [`CgStmtId`].
    /// Validates every referenced expr / list id.
    pub fn add_stmt(&mut self, stmt: CgStmt) -> Result<CgStmtId, BuilderError> {
        self.validate_stmt_refs(&stmt)?;
        let id = CgStmtId(self.stmt_len());
        self.inner.stmts.push(stmt);
        Ok(id)
    }

    /// Push a `CgStmtList` and return its freshly-allocated
    /// [`CgStmtListId`]. Validates that every contained `CgStmtId`
    /// resolves.
    pub fn add_stmt_list(&mut self, list: CgStmtList) -> Result<CgStmtListId, BuilderError> {
        for stmt_id in &list.stmts {
            self.check_stmt_id(*stmt_id)?;
        }
        let id = CgStmtListId(self.list_len());
        self.inner.stmt_lists.push(list);
        Ok(id)
    }

    /// Add an op. The builder assigns the [`OpId`], validates every
    /// expr / list id referenced by `kind`, then constructs the
    /// [`ComputeOp`] via [`ComputeOp::new`] (which auto-derives reads
    /// + writes from the current arenas).
    pub fn add_op(
        &mut self,
        kind: ComputeOpKind,
        shape: DispatchShape,
        span: Span,
    ) -> Result<OpId, BuilderError> {
        self.validate_op_kind_refs(&kind)?;
        let id = OpId(self.inner.ops.len() as u32);
        let op = ComputeOp::new(
            id,
            kind,
            shape,
            span,
            &self.inner.exprs,
            &self.inner.stmts,
            &self.inner.stmt_lists,
        );
        self.inner.ops.push(op);
        Ok(id)
    }

    /// Validate every id reference inside `kind`. Each variant has
    /// distinct reference shapes — mask predicates carry one expr id,
    /// scoring rows carry one required (`utility`) plus up to two
    /// optional (`target`, `guard`) expr ids each, physics/view-fold
    /// carry one stmt list id, spatial query / plumbing carry none.
    fn validate_op_kind_refs(&self, kind: &ComputeOpKind) -> Result<(), BuilderError> {
        match kind {
            ComputeOpKind::MaskPredicate { predicate, .. } => self.check_expr_id(*predicate),
            ComputeOpKind::ScoringArgmax { rows, .. } => {
                for row in rows {
                    self.check_expr_id(row.utility)?;
                    if let Some(target_id) = row.target {
                        self.check_expr_id(target_id)?;
                    }
                    if let Some(guard_id) = row.guard {
                        self.check_expr_id(guard_id)?;
                    }
                }
                Ok(())
            }
            ComputeOpKind::PhysicsRule { body, .. } => self.check_list_id(*body),
            ComputeOpKind::ViewFold { body, .. } => self.check_list_id(*body),
            // ViewDecay carries no `CgExprId` / `CgStmtListId` references —
            // the kernel body is hand-synthesised at emit time from the
            // `(view, rate_bits)` pair.
            ComputeOpKind::ViewDecay { .. } => Ok(()),
            ComputeOpKind::SpatialQuery { kind } => {
                if let SpatialQueryKind::FilteredWalk { filter } = kind {
                    self.check_expr_id(*filter)?;
                }
                Ok(())
            }
            // Plumbing kinds carry no `CgExprId` / `CgStmtListId`
            // references (every variant's reads/writes are sourced
            // from `PlumbingKind::dependencies()` as typed
            // `DataHandle`s); nothing to validate.
            ComputeOpKind::Plumbing { .. } => Ok(()),
        }
    }

    /// Finalize the builder and return the constructed program. After
    /// `finish`, post-construction mutations (record_read /
    /// record_write on individual ops) are still permitted; arena
    /// growth via the builder is not.
    pub fn finish(self) -> CgProgram {
        self.inner
    }

    // --- Read-only accessors (used by lowering callers) ----------------

    /// Borrow the program-in-progress for read-only inspection — used
    /// by lowering passes that want to consult the current expr arena
    /// while building (e.g. to share a sub-expression they already
    /// added).
    pub fn program(&self) -> &CgProgram {
        &self.inner
    }

    /// Replace the in-progress program's `view_signatures` map. The
    /// driver calls this once after registering materialized-view
    /// signatures into [`super::lower::expr::LoweringCtx::view_signatures`]
    /// and BEFORE the cycle-gate snapshot, so
    /// [`super::well_formed::check_well_formed`]'s view-key relaxation
    /// rule can consult the underlying scalar type while validating
    /// fold-body `Assign(ViewStorage{Primary}, scalar)` shapes.
    pub fn set_view_signatures(&mut self, sigs: BTreeMap<u32, ViewSignature>) {
        self.inner.view_signatures = sigs;
    }

    /// Append-only seam for post-construction registry-resolved bindings
    /// (e.g., [`ComputeOp::record_read`] for source rings,
    /// [`ComputeOp::record_write`] for `Emit` destination rings). Use
    /// sparingly — most reads/writes should be auto-derived via
    /// [`ComputeOp::new`].
    ///
    /// The driver needs this to wire ring edges (source-ring reads,
    /// destination-ring writes) BEFORE the cycle gate snapshot. Wiring
    /// post-`finish()` doesn't change the gate's verdict because the
    /// gate consults `op.reads` / `op.writes`; the symmetry between
    /// `DispatchShape::PerEvent` and `CgStmt::Emit` and the ring graph
    /// must be present at gate time.
    pub fn ops_mut(&mut self) -> &mut Vec<ComputeOp> {
        &mut self.inner.ops
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, EventRingAccess, RngPurpose, SpatialStorageKind, ViewStorageSlot,
    };
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::{BinaryOp, CgTy, LitValue};
    use crate::cg::op::{
        ActionId, EventKindId, PhysicsRuleId, ReplayabilityFlag, ScoringRowOp, SpatialQueryKind,
    };
    use crate::cg::stmt::EventField;

    // --- helpers -------------------------------------------------------

    fn read_self_hp() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        })
    }

    fn lit_f32(v: f32) -> CgExpr {
        CgExpr::Lit(LitValue::F32(v))
    }

    // --- Severity / CgDiagnostic --------------------------------------

    #[test]
    fn severity_display_distinct_per_variant() {
        assert_eq!(format!("{}", Severity::Note), "note");
        assert_eq!(format!("{}", Severity::Warning), "warning");
        assert_eq!(format!("{}", Severity::Lint), "lint");
    }

    #[test]
    fn cg_diagnostic_kind_display_orphaned_emit() {
        let d = CgDiagnosticKind::OrphanedEmit {
            event: EventKindId(7),
        };
        assert_eq!(format!("{}", d), "orphaned_emit(event=#7)");
    }

    #[test]
    fn cg_diagnostic_kind_display_unread_view() {
        let d = CgDiagnosticKind::UnreadView { view: ViewId(3) };
        assert_eq!(format!("{}", d), "unread_view(view=#3)");
    }

    #[test]
    fn cg_diagnostic_kind_display_unused_mask() {
        let d = CgDiagnosticKind::UnusedMask { mask: MaskId(5) };
        assert_eq!(format!("{}", d), "unused_mask(mask=#5)");
    }

    #[test]
    fn cg_diagnostic_kind_display_write_conflict() {
        let d = CgDiagnosticKind::WriteConflict {
            handle: DataHandle::MaskBitmap { mask: MaskId(2) },
            ops: vec![OpId(0), OpId(3)],
        };
        assert_eq!(
            format!("{}", d),
            "write_conflict(handle=mask[#2].bitmap, ops=[op#0, op#3])"
        );
    }

    #[test]
    fn cg_diagnostic_constructors_set_severity_and_span() {
        let d1 = CgDiagnostic::new(
            Severity::Warning,
            CgDiagnosticKind::UnusedMask { mask: MaskId(1) },
        );
        assert_eq!(d1.severity, Severity::Warning);
        assert_eq!(d1.span, None);

        let d2 = CgDiagnostic::with_span(
            Severity::Note,
            Span::new(10, 20),
            CgDiagnosticKind::UnreadView { view: ViewId(2) },
        );
        assert_eq!(d2.severity, Severity::Note);
        assert_eq!(d2.span, Some(Span::new(10, 20)));
    }

    #[test]
    fn cg_diagnostic_display_combines_severity_and_kind() {
        let d = CgDiagnostic::new(
            Severity::Warning,
            CgDiagnosticKind::OrphanedEmit {
                event: EventKindId(4),
            },
        );
        assert_eq!(format!("{}", d), "warning: orphaned_emit(event=#4)");
    }

    #[test]
    fn cg_diagnostic_serde_round_trip_modulo_span() {
        let d = CgDiagnostic::with_span(
            Severity::Warning,
            Span::new(10, 20),
            CgDiagnosticKind::UnusedMask { mask: MaskId(7) },
        );
        let json = serde_json::to_string(&d).expect("serialize");
        let back: CgDiagnostic = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.severity, d.severity);
        assert_eq!(back.kind, d.kind);
        // Span is not serialized — round-trip resets to None.
        assert_eq!(back.span, None);
    }

    // --- Interner ------------------------------------------------------

    #[test]
    fn interner_lookup_returns_inserted_name() {
        let mut interner = Interner::new();
        interner.views.insert(0, "standing".to_string());
        interner.masks.insert(3, "low_hp".to_string());
        assert_eq!(interner.get_view_name(ViewId(0)), Some("standing"));
        assert_eq!(interner.get_mask_name(MaskId(3)), Some("low_hp"));
        assert_eq!(interner.get_view_name(ViewId(99)), None);
    }

    #[test]
    fn interner_display_sorted_by_id() {
        let mut interner = Interner::new();
        interner.views.insert(2, "engaged".to_string());
        interner.views.insert(0, "standing".to_string());
        interner.views.insert(1, "fleeing".to_string());
        let s = format!("{}", interner);
        // Each table prints sorted (BTreeMap iteration order).
        assert!(s.contains("#0 -> \"standing\""));
        assert!(s.contains("#1 -> \"fleeing\""));
        assert!(s.contains("#2 -> \"engaged\""));
        // Ordering: standing before fleeing before engaged.
        let i_standing = s.find("standing").unwrap();
        let i_fleeing = s.find("fleeing").unwrap();
        let i_engaged = s.find("engaged").unwrap();
        assert!(i_standing < i_fleeing && i_fleeing < i_engaged);
    }

    #[test]
    fn interner_default_is_empty() {
        let interner = Interner::default();
        assert!(interner.views.is_empty());
        assert!(interner.masks.is_empty());
        assert!(interner.scorings.is_empty());
        assert!(interner.physics_rules.is_empty());
        assert!(interner.event_kinds.is_empty());
        assert!(interner.actions.is_empty());
        assert!(interner.config_consts.is_empty());
        assert!(interner.event_rings.is_empty());
    }

    // --- Builder add_expr ---------------------------------------------

    #[test]
    fn builder_add_expr_assigns_contiguous_ids() {
        let mut b = CgProgramBuilder::new();
        let a = b.add_expr(read_self_hp()).unwrap();
        let b1 = b.add_expr(lit_f32(1.0)).unwrap();
        assert_eq!(a, CgExprId(0));
        assert_eq!(b1, CgExprId(1));
    }

    #[test]
    fn builder_add_expr_validates_binary_ids() {
        let mut b = CgProgramBuilder::new();
        let lhs = b.add_expr(read_self_hp()).unwrap();
        // rhs id is dangling — only one expr in the arena so far.
        let err = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs,
                rhs: CgExprId(99),
                ty: CgTy::F32,
            })
            .expect_err("dangling rhs");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(99),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_expr_validates_select_ids() {
        let mut b = CgProgramBuilder::new();
        let cond = b.add_expr(CgExpr::Lit(LitValue::Bool(true))).unwrap();
        let err = b
            .add_expr(CgExpr::Select {
                cond,
                then: CgExprId(7),
                else_: CgExprId(0),
                ty: CgTy::F32,
            })
            .expect_err("dangling then");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(7),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_expr_accepts_lit_and_rng_with_no_refs() {
        let mut b = CgProgramBuilder::new();
        b.add_expr(lit_f32(2.5)).unwrap();
        b.add_expr(CgExpr::Rng {
            purpose: RngPurpose::Action,
            ty: CgTy::U32,
        })
        .unwrap();
        assert_eq!(b.program().exprs.len(), 2);
    }

    // --- Builder add_stmt + add_stmt_list -----------------------------

    #[test]
    fn builder_add_stmt_validates_assign_value() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(0),
            })
            .expect_err("dangling value");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_stmt_emit_walks_field_exprs() {
        let mut b = CgProgramBuilder::new();
        let _ = b.add_expr(read_self_hp()).unwrap();
        // Field expr id #99 dangles.
        let err = b
            .add_stmt(CgStmt::Emit {
                event: EventKindId(0),
                fields: vec![(
                    EventField {
                        event: EventKindId(0),
                        index: 0,
                    },
                    CgExprId(99),
                )],
            })
            .expect_err("dangling field expr");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(99),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_stmt_if_validates_then_else_lists() {
        let mut b = CgProgramBuilder::new();
        let cond = b.add_expr(CgExpr::Lit(LitValue::Bool(true))).unwrap();
        // No lists yet — `then` dangles.
        let err = b
            .add_stmt(CgStmt::If {
                cond,
                then: CgStmtListId(0),
                else_: None,
            })
            .expect_err("dangling then list");
        assert_eq!(
            err,
            BuilderError::DanglingStmtListId {
                referenced: CgStmtListId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_stmt_list_validates_contained_stmt_ids() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_stmt_list(CgStmtList::new(vec![CgStmtId(7)]))
            .expect_err("dangling stmt id");
        assert_eq!(
            err,
            BuilderError::DanglingStmtId {
                referenced: CgStmtId(7),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_stmt_list_accepts_empty() {
        let mut b = CgProgramBuilder::new();
        let id = b.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        assert_eq!(id, CgStmtListId(0));
    }

    // --- Builder add_op ------------------------------------------------

    #[test]
    fn builder_add_op_mask_predicate_validates_predicate_id() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_op(
                ComputeOpKind::MaskPredicate {
                    mask: MaskId(0),
                    predicate: CgExprId(0),
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .expect_err("dangling predicate");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_op_mask_predicate_succeeds_when_expr_exists() {
        let mut b = CgProgramBuilder::new();
        let hp = b.add_expr(read_self_hp()).unwrap();
        let half = b.add_expr(lit_f32(0.5)).unwrap();
        let pred = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: half,
                ty: CgTy::Bool,
            })
            .unwrap();
        let op_id = b
            .add_op(
                ComputeOpKind::MaskPredicate {
                    mask: MaskId(0),
                    predicate: pred,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap();
        assert_eq!(op_id, OpId(0));

        let prog = b.finish();
        // reads/writes auto-populated by ComputeOp::new walking the arenas.
        assert_eq!(
            prog.ops[0].reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(
            prog.ops[0].writes,
            vec![DataHandle::MaskBitmap { mask: MaskId(0) }]
        );
    }

    #[test]
    fn builder_add_op_scoring_validates_each_row() {
        let mut b = CgProgramBuilder::new();
        let utility = b.add_expr(lit_f32(1.0)).unwrap();
        // Target id is dangling — only one expr in arena.
        let err = b
            .add_op(
                ComputeOpKind::ScoringArgmax {
                    scoring: ScoringId(0),
                    rows: vec![ScoringRowOp {
                        action: ActionId(0),
                        utility,
                        target: Some(CgExprId(99)),
                        guard: None,
                    }],
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .expect_err("dangling row target");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(99),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_op_scoring_validates_optional_guard() {
        // Guard id is dangling — utility + target are valid.
        let mut b = CgProgramBuilder::new();
        let utility = b.add_expr(lit_f32(1.0)).unwrap();
        let err = b
            .add_op(
                ComputeOpKind::ScoringArgmax {
                    scoring: ScoringId(0),
                    rows: vec![ScoringRowOp {
                        action: ActionId(0),
                        utility,
                        target: None,
                        guard: Some(CgExprId(42)),
                    }],
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .expect_err("dangling row guard");
        assert_eq!(
            err,
            BuilderError::DanglingExprId {
                referenced: CgExprId(42),
                arena_len: 1,
            }
        );
    }

    #[test]
    fn builder_add_op_scoring_accepts_standard_row_with_no_target_no_guard() {
        // Standard row — target = None, guard = None. Must lower
        // cleanly with only the utility expression.
        let mut b = CgProgramBuilder::new();
        let utility = b.add_expr(lit_f32(0.1)).unwrap();
        let op_id = b
            .add_op(
                ComputeOpKind::ScoringArgmax {
                    scoring: ScoringId(0),
                    rows: vec![ScoringRowOp {
                        action: ActionId(0),
                        utility,
                        target: None,
                        guard: None,
                    }],
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .expect("standard row lowers cleanly");
        let prog = b.finish();
        assert_eq!(prog.ops[op_id.0 as usize].id, op_id);
    }

    #[test]
    fn builder_add_op_physics_rule_validates_body_id() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: Some(EventKindId(0)),
                    body: CgStmtListId(0),
                    replayable: ReplayabilityFlag::Replayable,
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
                Span::dummy(),
            )
            .expect_err("dangling body list");
        assert_eq!(
            err,
            BuilderError::DanglingStmtListId {
                referenced: CgStmtListId(0),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_op_view_fold_validates_body_id() {
        let mut b = CgProgramBuilder::new();
        let err = b
            .add_op(
                ComputeOpKind::ViewFold {
                    view: ViewId(0),
                    on_event: EventKindId(0),
                    body: CgStmtListId(2),
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
                Span::dummy(),
            )
            .expect_err("dangling body list");
        assert_eq!(
            err,
            BuilderError::DanglingStmtListId {
                referenced: CgStmtListId(2),
                arena_len: 0,
            }
        );
    }

    #[test]
    fn builder_add_op_spatial_query_no_id_refs() {
        let mut b = CgProgramBuilder::new();
        let id = b
            .add_op(
                ComputeOpKind::SpatialQuery {
                    kind: SpatialQueryKind::BuildHash,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap();
        assert_eq!(id, OpId(0));
    }


    // --- Diagnostics + interner intern_* ------------------------------

    #[test]
    fn builder_add_diagnostic_pushes_into_arena() {
        let mut b = CgProgramBuilder::new();
        b.add_diagnostic(CgDiagnostic::new(
            Severity::Warning,
            CgDiagnosticKind::UnusedMask { mask: MaskId(0) },
        ));
        let prog = b.finish();
        assert_eq!(prog.diagnostics.len(), 1);
        assert_eq!(prog.diagnostics[0].severity, Severity::Warning);
    }

    #[test]
    fn builder_intern_methods_populate_interner_tables() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing").unwrap();
        b.intern_mask_name(MaskId(1), "low_hp").unwrap();
        b.intern_scoring_name(ScoringId(2), "combat").unwrap();
        b.intern_physics_rule_name(PhysicsRuleId(3), "on_attack")
            .unwrap();
        b.intern_event_kind_name(EventKindId(4), "AttackHit")
            .unwrap();
        b.intern_action_name(ActionId(5), "MoveToward").unwrap();
        b.intern_config_const_name(ConfigConstId(6), "attack_range")
            .unwrap();
        b.intern_event_ring_name(EventRingId(7), "apply_ring")
            .unwrap();
        let prog = b.finish();
        assert_eq!(
            prog.interner.get_view_name(ViewId(0)),
            Some("standing")
        );
        assert_eq!(
            prog.interner.get_mask_name(MaskId(1)),
            Some("low_hp")
        );
        assert_eq!(
            prog.interner.get_scoring_name(ScoringId(2)),
            Some("combat")
        );
        assert_eq!(
            prog.interner.get_physics_rule_name(PhysicsRuleId(3)),
            Some("on_attack")
        );
        assert_eq!(
            prog.interner.get_event_kind_name(EventKindId(4)),
            Some("AttackHit")
        );
        assert_eq!(
            prog.interner.get_action_name(ActionId(5)),
            Some("MoveToward")
        );
        assert_eq!(
            prog.interner.get_config_const_name(ConfigConstId(6)),
            Some("attack_range")
        );
        assert_eq!(
            prog.interner.get_event_ring_name(EventRingId(7)),
            Some("apply_ring")
        );
    }

    // --- CgProgram arena traits ---------------------------------------

    #[test]
    fn program_implements_expr_arena() {
        let mut b = CgProgramBuilder::new();
        let id = b.add_expr(read_self_hp()).unwrap();
        let prog = b.finish();
        let node: &CgExpr = ExprArena::get(&prog, id).expect("expr id resolves");
        match node {
            CgExpr::Read(DataHandle::AgentField { field, .. }) => {
                assert_eq!(*field, AgentFieldId::Hp);
            }
            other => panic!("expected hp read, got {other:?}"),
        }
    }

    #[test]
    fn program_implements_stmt_arena() {
        let mut b = CgProgramBuilder::new();
        let val = b.add_expr(lit_f32(0.0)).unwrap();
        let stmt_id = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: val,
            })
            .unwrap();
        let prog = b.finish();
        let node: &CgStmt = StmtArena::get(&prog, stmt_id).expect("stmt id resolves");
        match node {
            CgStmt::Assign { value, .. } => assert_eq!(*value, val),
            other => panic!("expected Assign, got {other:?}"),
        }
    }

    #[test]
    fn program_implements_stmt_list_arena() {
        let mut b = CgProgramBuilder::new();
        let val = b.add_expr(lit_f32(0.0)).unwrap();
        let stmt_id = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: val,
            })
            .unwrap();
        let list_id = b
            .add_stmt_list(CgStmtList::new(vec![stmt_id]))
            .unwrap();
        let prog = b.finish();
        let node: &CgStmtList = StmtListArena::get(&prog, list_id).expect("list id resolves");
        assert_eq!(node.stmts, vec![stmt_id]);
    }

    // --- display_with_names -------------------------------------------

    #[test]
    fn display_with_names_substitutes_view_name() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(3), "engaged").unwrap();
        let prog = b.finish();
        let h = DataHandle::ViewStorage {
            view: ViewId(3),
            slot: ViewStorageSlot::Primary,
        };
        assert_eq!(prog.display_with_names(&h), "view[engaged].primary");
    }

    #[test]
    fn display_with_names_falls_back_to_id_form_without_interner_entry() {
        let prog = CgProgram::new();
        let h = DataHandle::ViewStorage {
            view: ViewId(3),
            slot: ViewStorageSlot::Primary,
        };
        assert_eq!(prog.display_with_names(&h), "view[#3].primary");
    }

    #[test]
    fn display_with_names_does_not_modify_data_handle_display() {
        // The opaque-id Display on DataHandle is preserved (Task 1.1
        // contract). display_with_names is purely additive.
        let h = DataHandle::ViewStorage {
            view: ViewId(3),
            slot: ViewStorageSlot::Primary,
        };
        assert_eq!(format!("{}", h), "view[#3].primary");
    }

    #[test]
    fn display_with_names_for_each_handle_variant() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing").unwrap();
        b.intern_mask_name(MaskId(1), "low_hp").unwrap();
        b.intern_event_ring_name(EventRingId(2), "apply").unwrap();
        b.intern_config_const_name(ConfigConstId(3), "attack_range")
            .unwrap();
        let prog = b.finish();

        // AgentField — no interner involvement (field name is fixed).
        assert_eq!(
            prog.display_with_names(&DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
            "agent.self.hp"
        );
        // ViewStorage — named.
        assert_eq!(
            prog.display_with_names(&DataHandle::ViewStorage {
                view: ViewId(0),
                slot: ViewStorageSlot::Primary,
            }),
            "view[standing].primary"
        );
        // EventRing — named.
        assert_eq!(
            prog.display_with_names(&DataHandle::EventRing {
                ring: EventRingId(2),
                kind: EventRingAccess::Append,
            }),
            "event_ring[apply].append"
        );
        // ConfigConst — named.
        assert_eq!(
            prog.display_with_names(&DataHandle::ConfigConst {
                id: ConfigConstId(3),
            }),
            "config[attack_range]"
        );
        // MaskBitmap — named.
        assert_eq!(
            prog.display_with_names(&DataHandle::MaskBitmap { mask: MaskId(1) }),
            "mask[low_hp].bitmap"
        );
        // ScoringOutput — atomic.
        assert_eq!(
            prog.display_with_names(&DataHandle::ScoringOutput),
            "scoring.output"
        );
        // SpatialStorage — atomic.
        assert_eq!(
            prog.display_with_names(&DataHandle::SpatialStorage {
                kind: SpatialStorageKind::GridCells,
            }),
            "spatial.grid_cells"
        );
        // Rng — atomic.
        assert_eq!(
            prog.display_with_names(&DataHandle::Rng {
                purpose: RngPurpose::Action,
            }),
            "rng(action)"
        );
    }

    // --- CgProgram serde round-trip -----------------------------------

    #[test]
    fn program_serde_round_trip_preserves_structure_modulo_spans() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing").unwrap();
        b.intern_mask_name(MaskId(1), "low_hp").unwrap();
        let hp = b.add_expr(read_self_hp()).unwrap();
        let half = b.add_expr(lit_f32(0.5)).unwrap();
        let pred = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: half,
                ty: CgTy::Bool,
            })
            .unwrap();
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(1),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::new(10, 20),
        )
        .unwrap();
        b.add_diagnostic(CgDiagnostic::with_span(
            Severity::Warning,
            Span::new(30, 40),
            CgDiagnosticKind::UnusedMask { mask: MaskId(7) },
        ));
        let prog = b.finish();

        let json = serde_json::to_string(&prog).expect("serialize");
        let back: CgProgram = serde_json::from_str(&json).expect("deserialize");

        // Structural fields round-trip.
        assert_eq!(back.exprs, prog.exprs);
        assert_eq!(back.stmts, prog.stmts);
        assert_eq!(back.stmt_lists, prog.stmt_lists);
        assert_eq!(back.interner, prog.interner);
        // Op compares modulo span (Span has no Deserialize, so it
        // resets to dummy on round-trip). The other op fields match.
        assert_eq!(back.ops.len(), 1);
        assert_eq!(back.ops[0].id, prog.ops[0].id);
        assert_eq!(back.ops[0].kind, prog.ops[0].kind);
        assert_eq!(back.ops[0].reads, prog.ops[0].reads);
        assert_eq!(back.ops[0].writes, prog.ops[0].writes);
        assert_eq!(back.ops[0].shape, prog.ops[0].shape);
        assert_eq!(back.ops[0].span, Span::dummy());
        // Diagnostic span is None on round-trip; severity + kind preserved.
        assert_eq!(back.diagnostics.len(), 1);
        assert_eq!(back.diagnostics[0].severity, Severity::Warning);
        assert_eq!(back.diagnostics[0].kind, prog.diagnostics[0].kind);
        assert_eq!(back.diagnostics[0].span, None);
    }

    // --- Pretty-printer fixture ---------------------------------------

    /// Build a small but representative program — one mask, one
    /// scoring, one physics rule, one view fold, one spatial query —
    /// and pin the full pretty-printed output.
    fn build_fixture() -> CgProgram {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing").unwrap();
        b.intern_mask_name(MaskId(0), "low_hp").unwrap();

        // Expression arena layout (deterministic, in insertion order).
        let hp = b.add_expr(read_self_hp()).unwrap(); // #0
        let half = b.add_expr(lit_f32(0.5)).unwrap(); // #1
        let pred = b
            .add_expr(CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: hp,
                rhs: half,
                ty: CgTy::Bool,
            })
            .unwrap(); // #2
        let utility = b.add_expr(lit_f32(1.0)).unwrap(); // #3
        let target = b.add_expr(CgExpr::Lit(LitValue::AgentId(0))).unwrap(); // #4
        let zero = b.add_expr(lit_f32(0.0)).unwrap(); // #5

        // Statement arena layout.
        let assign_hp = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: zero,
            })
            .unwrap(); // stmt#0
        let assign_view = b
            .add_stmt(CgStmt::Assign {
                target: DataHandle::ViewStorage {
                    view: ViewId(0),
                    slot: ViewStorageSlot::Primary,
                },
                value: hp,
            })
            .unwrap(); // stmt#1

        // Statement list arena layout.
        let physics_body = b
            .add_stmt_list(CgStmtList::new(vec![assign_hp]))
            .unwrap(); // list#0
        let view_body = b
            .add_stmt_list(CgStmtList::new(vec![assign_view]))
            .unwrap(); // list#1

        // Ops.
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap(); // op#0
        b.add_op(
            ComputeOpKind::ScoringArgmax {
                scoring: ScoringId(0),
                rows: vec![ScoringRowOp {
                    action: ActionId(0),
                    utility,
                    target: Some(target),
                    guard: None,
                }],
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap(); // op#1
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: Some(EventKindId(0)),
                body: physics_body,
                replayable: ReplayabilityFlag::Replayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap(); // op#2
        b.add_op(
            ComputeOpKind::ViewFold {
                view: ViewId(0),
                on_event: EventKindId(0),
                body: view_body,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
        )
        .unwrap(); // op#3
        b.add_op(
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::BuildHash,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap(); // op#4

        // Diagnostic.
        b.add_diagnostic(CgDiagnostic::new(
            Severity::Warning,
            CgDiagnosticKind::UnusedMask { mask: MaskId(0) },
        ));

        b.finish()
    }

    #[test]
    fn program_pretty_print_matches_fixture() {
        let prog = build_fixture();
        let printed = format!("{}", prog);
        let expected = "\
program {
    diagnostics: [
        warning: unused_mask(mask=#0),
    ],
    interner {
        views: { #0 -> \"standing\" }
        masks: { #0 -> \"low_hp\" }
        scorings: {}
        physics_rules: {}
        event_kinds: {}
        actions: {}
        config_consts: {}
        event_rings: {}
    }
    ops: [
        op#0 kind=mask_predicate(mask=#0) shape=per_agent reads=[agent.self.hp] writes=[mask[#0].bitmap],
        op#1 kind=scoring_argmax(scoring=#0, rows=1) shape=per_agent reads=[] writes=[scoring.output],
        op#2 kind=physics_rule(rule=#0, on_event=#0, replayable=replayable) shape=per_event(ring=#0) reads=[] writes=[agent.self.hp],
        op#3 kind=view_fold(view=#0, on_event=#0) shape=per_event(ring=#0) reads=[agent.self.hp] writes=[view[#0].primary],
        op#4 kind=spatial_query(build_hash) shape=per_agent reads=[agent.self.pos] writes=[spatial.grid_cells, spatial.grid_offsets],
    ],
}";
        assert_eq!(printed, expected, "actual output:\n{}", printed);
    }

    #[test]
    fn program_pretty_print_empty_program() {
        let prog = CgProgram::new();
        let printed = format!("{}", prog);
        let expected = "\
program {
    diagnostics: [],
    interner {
        views: {}
        masks: {}
        scorings: {}
        physics_rules: {}
        event_kinds: {}
        actions: {}
        config_consts: {}
        event_rings: {}
    }
    ops: [],
}";
        assert_eq!(printed, expected, "actual output:\n{}", printed);
    }

    // --- record_read / record_write seam ------------------------------

    #[test]
    fn record_write_through_program_ops_appends_to_writes() {
        // Lowering pattern: build the op, then reach into
        // `prog.ops[op_id]` and call record_write to inject the
        // registry-resolved binding.
        let mut b = CgProgramBuilder::new();
        let body = b.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        let op_id = b
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: Some(EventKindId(7)),
                    body,
                    replayable: ReplayabilityFlag::Replayable,
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(7),
                },
                Span::dummy(),
            )
            .unwrap();
        let mut prog = b.finish();

        let dest = DataHandle::EventRing {
            ring: EventRingId(9),
            kind: EventRingAccess::Append,
        };
        prog.ops[op_id.0 as usize].record_write(dest.clone());
        assert!(prog.ops[op_id.0 as usize].writes.contains(&dest));
    }

    // --- BuilderError display -----------------------------------------

    #[test]
    fn builder_error_display_dangling_expr() {
        let e = BuilderError::DanglingExprId {
            referenced: CgExprId(7),
            arena_len: 3,
        };
        assert_eq!(
            format!("{}", e),
            "dangling CgExprId(#7) (expr arena holds 3 entries)"
        );
    }

    #[test]
    fn builder_error_display_dangling_stmt() {
        let e = BuilderError::DanglingStmtId {
            referenced: CgStmtId(2),
            arena_len: 0,
        };
        assert_eq!(
            format!("{}", e),
            "dangling CgStmtId(#2) (stmt arena holds 0 entries)"
        );
    }

    #[test]
    fn builder_error_display_dangling_stmt_list() {
        let e = BuilderError::DanglingStmtListId {
            referenced: CgStmtListId(1),
            arena_len: 0,
        };
        assert_eq!(
            format!("{}", e),
            "dangling CgStmtListId(#1) (stmt-list arena holds 0 entries)"
        );
    }

    #[test]
    fn builder_error_display_duplicate_intern_entry() {
        let e = BuilderError::DuplicateInternEntry {
            id_kind: "view",
            id: 3,
            prior: "engaged".to_string(),
            new: "fleeing".to_string(),
        };
        let s = format!("{}", e);
        assert!(s.contains("view#3"), "missing id token: {s}");
        assert!(s.contains("engaged"), "missing prior name: {s}");
        assert!(s.contains("fleeing"), "missing new name: {s}");
    }

    // --- intern_*_name idempotency + duplicate detection -------------

    #[test]
    fn intern_view_name_is_idempotent_for_same_name() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing").unwrap();
        // Same id, same name — accepted silently.
        b.intern_view_name(ViewId(0), "standing").unwrap();
        let prog = b.finish();
        assert_eq!(prog.interner.get_view_name(ViewId(0)), Some("standing"));
    }

    #[test]
    fn intern_view_name_rejects_conflicting_name_for_same_id() {
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing").unwrap();
        let err = b
            .intern_view_name(ViewId(0), "fleeing")
            .expect_err("duplicate intern entry");
        assert_eq!(
            err,
            BuilderError::DuplicateInternEntry {
                id_kind: "view",
                id: 0,
                prior: "standing".to_string(),
                new: "fleeing".to_string(),
            }
        );
        // Original entry preserved — failed re-intern does not overwrite.
        let prog = b.finish();
        assert_eq!(prog.interner.get_view_name(ViewId(0)), Some("standing"));
    }

    #[test]
    fn intern_methods_use_distinct_id_kind_tags() {
        // Each typed intern method routes to a distinct `id_kind` tag.
        // Verifies the closed set agreed with `IdKind` usage in
        // `DataHandleNameResolver` for `Interner`.
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "v").unwrap();
        b.intern_mask_name(MaskId(0), "m").unwrap();
        b.intern_scoring_name(ScoringId(0), "s").unwrap();
        b.intern_physics_rule_name(PhysicsRuleId(0), "p").unwrap();
        b.intern_event_kind_name(EventKindId(0), "e").unwrap();
        b.intern_action_name(ActionId(0), "a").unwrap();
        b.intern_config_const_name(ConfigConstId(0), "c").unwrap();
        b.intern_event_ring_name(EventRingId(0), "r").unwrap();

        // Each table records its own (id, name) — the `id_kind` tag in
        // a duplicate-error helps disambiguate.
        let err = b
            .intern_view_name(ViewId(0), "v2")
            .expect_err("view dup");
        assert!(
            matches!(
                err,
                BuilderError::DuplicateInternEntry {
                    id_kind: "view",
                    ..
                }
            ),
            "expected view tag on duplicate, got {err:?}"
        );
        let err = b
            .intern_mask_name(MaskId(0), "m2")
            .expect_err("mask dup");
        assert!(
            matches!(
                err,
                BuilderError::DuplicateInternEntry {
                    id_kind: "mask",
                    ..
                }
            ),
            "expected mask tag on duplicate, got {err:?}"
        );
    }

    // --- DataHandle::fmt_with via Interner ---------------------------

    #[test]
    fn data_handle_fmt_with_named_path_renders_named_ids_and_falls_back_to_opaque() {
        // Program with named ViewId(0) but unnamed EventRingId(3).
        // `display_with_names` prints `view[standing]` (named) and
        // `event_ring[#3]` (opaque), confirming the resolver routes
        // each id-kind through the right interner table.
        let mut b = CgProgramBuilder::new();
        b.intern_view_name(ViewId(0), "standing").unwrap();
        // No intern call for EventRingId(3) — its name renders opaque.
        let prog = b.finish();

        let view_handle = DataHandle::ViewStorage {
            view: ViewId(0),
            slot: super::super::data_handle::ViewStorageSlot::Primary,
        };
        let ring_handle = DataHandle::EventRing {
            ring: EventRingId(3),
            kind: super::super::data_handle::EventRingAccess::Read,
        };
        assert_eq!(
            prog.display_with_names(&view_handle),
            "view[standing].primary"
        );
        assert_eq!(
            prog.display_with_names(&ring_handle),
            "event_ring[#3].read"
        );
        // Bare Display on the same handles always renders opaque —
        // names are not consulted.
        assert_eq!(format!("{}", view_handle), "view[#0].primary");
        assert_eq!(format!("{}", ring_handle), "event_ring[#3].read");
    }

    #[test]
    fn data_handle_fmt_with_unit_resolver_matches_display() {
        // `DataHandle::fmt_with` with the unit resolver `&()` produces
        // identical output to bare `Display`. Verifies the Display impl
        // is a delegate, not a parallel implementation.
        let cases = vec![
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            },
            DataHandle::ViewStorage {
                view: ViewId(2),
                slot: super::super::data_handle::ViewStorageSlot::Primary,
            },
            DataHandle::EventRing {
                ring: EventRingId(4),
                kind: super::super::data_handle::EventRingAccess::Append,
            },
            DataHandle::ConfigConst {
                id: ConfigConstId(7),
            },
            DataHandle::MaskBitmap { mask: MaskId(1) },
            DataHandle::ScoringOutput,
            DataHandle::SpatialStorage {
                kind: SpatialStorageKind::GridCells,
            },
            DataHandle::Rng {
                purpose: RngPurpose::Action,
            },
        ];
        for h in cases {
            let display = format!("{}", h);
            // The fmt_with(&()) path is the same path Display itself
            // invokes — equality is by construction, but we assert it
            // anyway as a regression guard.
            struct WithUnit<'a>(&'a DataHandle);
            impl<'a> fmt::Display for WithUnit<'a> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.0.fmt_with(f, &())
                }
            }
            let via_fmt_with = format!("{}", WithUnit(&h));
            assert_eq!(display, via_fmt_with, "Display vs fmt_with(&()) drift");
        }
    }
}
