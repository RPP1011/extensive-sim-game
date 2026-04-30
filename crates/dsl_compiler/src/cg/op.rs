//! `ComputeOp` — the unit of compute in the Compute-Graph IR.
//!
//! Every act of computation the DSL produces — a mask predicate
//! evaluation, a scoring-row argmax, a physics handler body, a view
//! fold body, a spatial query, or a piece of plumbing (alive bitmap
//! pack, sim_cfg upload, …) — lowers to a single [`ComputeOp`]. The
//! op's [`ComputeOpKind`] carries the variant-specific payload; its
//! [`DispatchShape`] determines how threads are laid out; its `reads`
//! and `writes` are derived from the embedded expression/statement
//! trees and used downstream for fusion analysis + bind-group-layout
//! synthesis.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 1.3, for the design rationale.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::data_handle::{
    DataHandle, EventRingAccess, EventRingId, MaskId, SpatialStorageKind, ViewId,
};
use super::dispatch::DispatchShape;
use super::expr::ExprArena;
use super::stmt::{
    collect_expr_reads, collect_list_dependencies, CgStmtListId, StmtArena, StmtListArena,
};

// Re-export the AST `Span` for diagnostic provenance. Reusing the
// existing AST type keeps spans consistent across passes (every CG op
// can carry the same `Span` the AST node it lowered from carried). The
// AST `Span` is `Copy + Clone + PartialEq + Eq + Serialize`. It does
// NOT implement `Deserialize` — see the `serde_skip_span` note on
// [`ComputeOp`] below for how `ComputeOp` round-trips without spans.
pub use dsl_ast::ast::Span;

// ---------------------------------------------------------------------------
// Newtype IDs
// ---------------------------------------------------------------------------

/// Stable id for a [`ComputeOp`] in the program's op arena. Sibling to
/// [`crate::cg::CgExprId`]; the program (Task 1.5) holds the
/// `Vec<ComputeOp>` indexed by this id.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct OpId(pub u32);

/// Stable id for a `scoring` declaration. One per top-level scoring
/// block in the DSL source; assignment order matches resolved IR
/// order so snapshots are reproducible.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ScoringId(pub u32);

/// Stable id for a `physics` rule. One per physics declaration; the
/// rule's `on Event => …` handlers each become a [`ComputeOpKind::PhysicsRule`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PhysicsRuleId(pub u32);

/// Stable id for an event variant. One per `event` declaration in the
/// DSL. Used by [`ComputeOpKind::PhysicsRule`] /
/// [`ComputeOpKind::ViewFold`] to name which event triggers the body
/// and by [`crate::cg::stmt::CgStmt::Emit`] to name the variant being
/// emitted.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct EventKindId(pub u32);

/// Stable id for an action — the head of a scoring row. Each row's
/// argmax candidate is identified by an `ActionId`; the engine maps
/// the winning id to a concrete behaviour at apply time.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ActionId(pub u32);

// ---------------------------------------------------------------------------
// SpatialQueryKind — typed enumeration of spatial-grid ops
// ---------------------------------------------------------------------------

/// Spatial-grid compute kinds.
///
/// The variants here mirror the kernel-wrapper functions exposed by
/// `crates/dsl_compiler/src/emit_spatial_kernel.rs`:
///
/// - `BuildHash` → `emit_spatial_hash_rs` (writes the grid).
/// - `KinQuery` → `emit_kin_query_rs` (writes per-agent kin-of-team
///   query results).
/// - `EngagementQuery` → `emit_engagement_query_rs` (writes per-agent
///   engagement-target query results).
///
/// Adding a new spatial kernel adds a variant here. The IR-level
/// `(reads, writes)` signature for each variant is encoded in
/// [`SpatialQueryKind::dependencies`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum SpatialQueryKind {
    /// Build the per-cell agent index — reads agents (positions),
    /// writes the cell + offset arrays.
    BuildHash,
    /// Per-agent kin-of-team neighborhood walk — reads the grid,
    /// writes the per-agent query-results scratch.
    KinQuery,
    /// Per-agent engagement-target neighborhood walk — reads the grid,
    /// writes the per-agent query-results scratch.
    EngagementQuery,
}

impl SpatialQueryKind {
    /// `(reads, writes)` signature. Hard-coded per kernel — the IR
    /// encodes the structural dependencies that the underlying
    /// hand-written GPU kernels touch.
    ///
    /// Reads/writes use the [`DataHandle::SpatialStorage`] variant to
    /// name grid cells/offsets/query-results; the per-agent positions
    /// (read by `BuildHash`) and the spatial-query-driven downstream
    /// reads of agent fields are emit-time concerns and aren't
    /// surfaced here — the shape only records the spatial-storage
    /// touches.
    pub fn dependencies(self) -> (Vec<DataHandle>, Vec<DataHandle>) {
        match self {
            SpatialQueryKind::BuildHash => (
                vec![],
                vec![
                    DataHandle::SpatialStorage {
                        kind: SpatialStorageKind::GridCells,
                    },
                    DataHandle::SpatialStorage {
                        kind: SpatialStorageKind::GridOffsets,
                    },
                ],
            ),
            SpatialQueryKind::KinQuery => (
                vec![
                    DataHandle::SpatialStorage {
                        kind: SpatialStorageKind::GridCells,
                    },
                    DataHandle::SpatialStorage {
                        kind: SpatialStorageKind::GridOffsets,
                    },
                ],
                vec![DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::QueryResults,
                }],
            ),
            SpatialQueryKind::EngagementQuery => (
                vec![
                    DataHandle::SpatialStorage {
                        kind: SpatialStorageKind::GridCells,
                    },
                    DataHandle::SpatialStorage {
                        kind: SpatialStorageKind::GridOffsets,
                    },
                ],
                vec![DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::QueryResults,
                }],
            ),
        }
    }

    /// Stable snake_case label for pretty-printing.
    pub fn label(self) -> &'static str {
        match self {
            SpatialQueryKind::BuildHash => "build_hash",
            SpatialQueryKind::KinQuery => "kin_query",
            SpatialQueryKind::EngagementQuery => "engagement_query",
        }
    }
}

impl fmt::Display for SpatialQueryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// PlumbingKind — typed enumeration of one-shot scratch ops
// ---------------------------------------------------------------------------

/// One-shot plumbing kinds — emit-strategy operations that the schedule
/// synthesizer can choose to inline or split out, but that are not
/// directly user-authored DSL declarations.
///
/// The variants here mirror the standalone WGSL emitters in
/// `crates/dsl_compiler/src/emit_*_wgsl.rs`:
///
/// - `AlivePack` → `emit_alive_pack_wgsl` — packs per-agent alive
///   bits into a `u32` bitmap.
/// - `FusedAgentUnpack` → unpacks the agents-input scratch into the
///   per-frame agent SoA + per-mask scratch.
/// - `SeedIndirect` → seeds the cascade's indirect-args buffer based
///   on the apply-path event ring's tail count.
/// - `SimCfgUpload` → uploads `sim_cfg` (tick, agent_cap, …) into
///   the GPU uniform buffer.
/// - `EventRingDrain { ring }` → resets a ring's tail counter back
///   to zero at the end of a tick. The ring id distinguishes apply-
///   path vs cascade rings.
///
/// Each variant has a fixed (reads, writes) signature, encoded in
/// [`PlumbingKind::dependencies`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum PlumbingKind {
    /// Pack per-agent `alive` flags into a `u32` bitmap, one word
    /// per 32 agents. One thread per output word.
    AlivePack,

    /// Unpack a fused agents-input buffer into the agent SoA + mask
    /// scratch. One thread per agent slot.
    FusedAgentUnpack,

    /// Seed the cascade's indirect-args buffer (workgroup count) from
    /// the apply-path event ring's tail. Single-threaded.
    SeedIndirect,

    /// Upload sim_cfg (tick, agent_cap, registry slots, …) into the
    /// GPU uniform buffer. Single-threaded.
    SimCfgUpload,

    /// Reset an event ring's tail counter to zero at end-of-tick.
    /// `ring` names which ring (apply-path vs cascade) is being
    /// drained.
    EventRingDrain { ring: EventRingId },
}

impl PlumbingKind {
    /// `(reads, writes)` signature.
    ///
    /// `AlivePack` reads `agent.self.alive` (the SoA alive flag) and
    /// writes a packed bitmap. The bitmap has no first-class
    /// [`DataHandle`] variant today; the emit-time `alive_bitmap`
    /// pool slot is a [`DataHandle::SpatialStorage`]-adjacent scratch.
    /// We surface the read explicitly and surface the write as the
    /// `MaskBitmap { mask: MaskId(0) }` sentinel — the alive-pack
    /// path does not have a user-visible mask id, so the IR treats the
    /// "alive bitmap" as mask id 0 by convention. Lowering passes
    /// (Phase 2) re-target this if the convention changes.
    ///
    /// `FusedAgentUnpack` reads the agents-input scratch and writes
    /// the per-mask scratch + per-agent fields. The per-frame agents
    /// SoA is a virtual buffer that doesn't have a single
    /// [`DataHandle`] variant; the IR records the touched mask
    /// bitmap (mask id 0) and a generic alive-flag write to surface
    /// the per-agent side.
    ///
    /// `SeedIndirect` reads the apply-path event ring's tail counter
    /// and writes the cascade indirect-args buffer; we surface the
    /// read as an [`DataHandle::EventRing`] read on ring 0 (the
    /// apply-path A-ring by convention) and the write as a
    /// [`DataHandle::SpatialStorage::QueryResults`] sentinel — the
    /// indirect-args buffer is pool-resident scratch, not first-class
    /// IR storage. This is consistent with how Task 1.1 surfaces
    /// pool buffers.
    ///
    /// `SimCfgUpload` writes the `ConfigConst { id: 0 }` slot —
    /// `sim_cfg` is a single config struct, treated as id 0 here.
    /// Reads are empty (the values come from CPU state).
    ///
    /// `EventRingDrain { ring }` writes the ring's tail back to zero;
    /// we surface this as an [`DataHandle::EventRing`] append on the
    /// ring (semantically a write of the tail counter).
    ///
    /// These conventions are deliberately coarse — the schedule
    /// synthesizer (Phase 3) only needs to know *which buffers* a
    /// plumbing op touches, not the field-level decomposition.
    pub fn dependencies(self) -> (Vec<DataHandle>, Vec<DataHandle>) {
        use super::data_handle::{
            AgentFieldId, AgentRef, ConfigConstId, DataHandle as DH, MaskId, SpatialStorageKind,
        };
        match self {
            PlumbingKind::AlivePack => (
                vec![DH::AgentField {
                    field: AgentFieldId::Alive,
                    target: AgentRef::Self_,
                }],
                vec![DH::MaskBitmap { mask: MaskId(0) }],
            ),
            PlumbingKind::FusedAgentUnpack => (
                vec![DH::AgentField {
                    field: AgentFieldId::Alive,
                    target: AgentRef::Self_,
                }],
                vec![DH::MaskBitmap { mask: MaskId(0) }],
            ),
            PlumbingKind::SeedIndirect => (
                vec![DH::EventRing {
                    ring: EventRingId(0),
                    kind: EventRingAccess::Read,
                }],
                vec![DH::SpatialStorage {
                    kind: SpatialStorageKind::QueryResults,
                }],
            ),
            PlumbingKind::SimCfgUpload => (
                vec![],
                vec![DH::ConfigConst {
                    id: ConfigConstId(0),
                }],
            ),
            PlumbingKind::EventRingDrain { ring } => (
                vec![],
                vec![DH::EventRing {
                    ring,
                    kind: EventRingAccess::Append,
                }],
            ),
        }
    }

    /// Stable snake_case label for pretty-printing.
    pub fn label(self) -> String {
        match self {
            PlumbingKind::AlivePack => "alive_pack".to_string(),
            PlumbingKind::FusedAgentUnpack => "fused_agent_unpack".to_string(),
            PlumbingKind::SeedIndirect => "seed_indirect".to_string(),
            PlumbingKind::SimCfgUpload => "sim_cfg_upload".to_string(),
            PlumbingKind::EventRingDrain { ring } => {
                format!("event_ring_drain(ring=#{})", ring.0)
            }
        }
    }
}

impl fmt::Display for PlumbingKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

// ---------------------------------------------------------------------------
// ScoringRowOp
// ---------------------------------------------------------------------------

/// One row of a scoring argmax. The DSL surface gives each row a head
/// (`Attack(target)`, `MoveToward(target)`, …) which lowering resolves
/// to a stable [`ActionId`]; the row's body is a utility expression
/// (the score) and a target expression (the agent id to act upon).
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ScoringRowOp {
    pub action: ActionId,
    pub utility: super::data_handle::CgExprId,
    pub target: super::data_handle::CgExprId,
}

impl fmt::Display for ScoringRowOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "row(action=#{}, utility=expr#{}, target=expr#{})",
            self.action.0, self.utility.0, self.target.0
        )
    }
}

// ---------------------------------------------------------------------------
// ComputeOpKind
// ---------------------------------------------------------------------------

/// Variant-specific payload for a [`ComputeOp`]. Six kinds today, each
/// covering one structural role the DSL surface produces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputeOpKind {
    /// Per-agent predicate evaluation. Lowered from `mask` decls. The
    /// embedded expression is a `Bool`-typed `CgExpr`; emit walks it
    /// to produce the mask-bit set logic.
    MaskPredicate {
        mask: MaskId,
        predicate: super::data_handle::CgExprId,
    },

    /// Per-agent utility computation + argmax. Lowered from `scoring`
    /// decls. `rows` is the action-keyed list of (utility, target)
    /// pairs.
    ScoringArgmax {
        scoring: ScoringId,
        rows: Vec<ScoringRowOp>,
    },

    /// Per-event handler. Lowered from `physics` rules. `body` is a
    /// `CgStmtList` (sibling tree to `CgExpr` — assignments + emits).
    PhysicsRule {
        rule: PhysicsRuleId,
        on_event: EventKindId,
        body: CgStmtListId,
    },

    /// Per-event view-storage update. Lowered from `view fold { on
    /// Event => … }` handlers.
    ViewFold {
        view: ViewId,
        on_event: EventKindId,
        body: CgStmtListId,
    },

    /// Spatial query — dispatch shape determines hash-build vs
    /// query-walk.
    SpatialQuery { kind: SpatialQueryKind },

    /// One-shot scratch op. See [`PlumbingKind`].
    Plumbing { kind: PlumbingKind },
}

impl ComputeOpKind {
    /// Compute the (reads, writes) signature for this kind by walking
    /// the embedded expression and statement trees.
    ///
    /// The walk order is deterministic: depth-first in source-order,
    /// duplicates allowed. Callers that need a unique set should
    /// dedupe; the IR records *what* the op syntactically references,
    /// not a normalised set.
    ///
    /// This is the function called by [`ComputeOp::new`] to populate
    /// the `reads` and `writes` fields. Those fields must NEVER be
    /// set independently of this — they are derived state.
    pub fn compute_dependencies(
        &self,
        exprs: &dyn ExprArena,
        stmts: &dyn StmtArena,
        lists: &dyn StmtListArena,
    ) -> (Vec<DataHandle>, Vec<DataHandle>) {
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        match self {
            ComputeOpKind::MaskPredicate { mask, predicate } => {
                collect_expr_reads(*predicate, exprs, &mut reads);
                writes.push(DataHandle::MaskBitmap { mask: *mask });
            }
            ComputeOpKind::ScoringArgmax { scoring: _, rows } => {
                for row in rows {
                    collect_expr_reads(row.utility, exprs, &mut reads);
                    collect_expr_reads(row.target, exprs, &mut reads);
                }
                writes.push(DataHandle::ScoringOutput);
            }
            ComputeOpKind::PhysicsRule {
                rule: _,
                on_event,
                body,
            } => {
                // Reading the source event ring is an implicit input
                // for every per-event handler. The ring id is the
                // event variant's id by convention (Task 1.1's
                // EventRingId is opaque; lowering resolves the
                // mapping).
                reads.push(DataHandle::EventRing {
                    ring: EventRingId(on_event.0),
                    kind: EventRingAccess::Read,
                });
                collect_list_dependencies(*body, exprs, stmts, lists, &mut reads, &mut writes);
            }
            ComputeOpKind::ViewFold {
                view,
                on_event,
                body,
            } => {
                // Same implicit event-ring read as PhysicsRule.
                reads.push(DataHandle::EventRing {
                    ring: EventRingId(on_event.0),
                    kind: EventRingAccess::Read,
                });
                collect_list_dependencies(*body, exprs, stmts, lists, &mut reads, &mut writes);
                // Whatever slots the body wrote are already in
                // `writes`; additionally, the view's primary storage
                // is the canonical "fold output" slot, so surface it
                // even when the body didn't touch a per-slot handle
                // explicitly. (The body of a fold lowering today
                // always writes ViewStorage{view, slot=Primary}, but
                // recording it here is defensive — duplicate-tolerant
                // by design.)
                writes.push(DataHandle::ViewStorage {
                    view: *view,
                    slot: super::data_handle::ViewStorageSlot::Primary,
                });
            }
            ComputeOpKind::SpatialQuery { kind } => {
                let (r, w) = kind.dependencies();
                reads.extend(r);
                writes.extend(w);
            }
            ComputeOpKind::Plumbing { kind } => {
                let (r, w) = kind.dependencies();
                reads.extend(r);
                writes.extend(w);
            }
        }
        (reads, writes)
    }

    /// Stable snake_case label for the kind, used by pretty-printing
    /// and structured logs.
    pub fn label(&self) -> String {
        match self {
            ComputeOpKind::MaskPredicate { mask, .. } => {
                format!("mask_predicate(mask=#{})", mask.0)
            }
            ComputeOpKind::ScoringArgmax { scoring, rows } => {
                format!(
                    "scoring_argmax(scoring=#{}, rows={})",
                    scoring.0,
                    rows.len()
                )
            }
            ComputeOpKind::PhysicsRule { rule, on_event, .. } => {
                format!(
                    "physics_rule(rule=#{}, on_event=#{})",
                    rule.0, on_event.0
                )
            }
            ComputeOpKind::ViewFold { view, on_event, .. } => {
                format!("view_fold(view=#{}, on_event=#{})", view.0, on_event.0)
            }
            ComputeOpKind::SpatialQuery { kind } => {
                format!("spatial_query({})", kind)
            }
            ComputeOpKind::Plumbing { kind } => format!("plumbing({})", kind),
        }
    }
}

impl fmt::Display for ComputeOpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

// ---------------------------------------------------------------------------
// ComputeOp
// ---------------------------------------------------------------------------

/// A single unit of compute in the Compute-Graph IR.
///
/// `reads` and `writes` are derived from `kind` at construction by
/// [`ComputeOp::new`]. They are exposed as `pub` for inspection (every
/// later pass — fusion analysis, BGL synthesis, schedule planning —
/// reads them) but **must not be set independently**: a caller that
/// mutates `kind` is responsible for calling
/// [`ComputeOp::recompute_dependencies`] before any consumer observes
/// the op. If a stale `reads`/`writes` reaches a consumer, the
/// well-formed pass (Task 1.6) will catch the divergence.
///
/// `span` carries the source location of the AST node this op was
/// lowered from — used by diagnostics. The span is `serde(skip)`'d
/// because [`Span`] does not implement `Deserialize`; round-tripping
/// a `ComputeOp` discards the span and replaces it with [`Span::dummy`]
/// at deserialization time. Snapshot tests should compare on the kind
/// + reads + writes + shape only, not the span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeOp {
    pub id: OpId,
    pub kind: ComputeOpKind,
    pub reads: Vec<DataHandle>,
    pub writes: Vec<DataHandle>,
    pub shape: DispatchShape,
    /// Source location for diagnostics. Skipped by serde — see the
    /// type-level docstring.
    #[serde(skip, default = "Span::dummy")]
    pub span: Span,
}

impl ComputeOp {
    /// Construct a new [`ComputeOp`] with `reads` and `writes`
    /// auto-derived from `kind`.
    pub fn new(
        id: OpId,
        kind: ComputeOpKind,
        shape: DispatchShape,
        span: Span,
        exprs: &dyn ExprArena,
        stmts: &dyn StmtArena,
        lists: &dyn StmtListArena,
    ) -> Self {
        let (reads, writes) = kind.compute_dependencies(exprs, stmts, lists);
        Self {
            id,
            kind,
            reads,
            writes,
            shape,
            span,
        }
    }

    /// Re-derive `reads` and `writes` from the current `kind`. Call
    /// this after mutating `kind` to keep the dependency cache in
    /// sync.
    pub fn recompute_dependencies(
        &mut self,
        exprs: &dyn ExprArena,
        stmts: &dyn StmtArena,
        lists: &dyn StmtListArena,
    ) {
        let (reads, writes) = self.kind.compute_dependencies(exprs, stmts, lists);
        self.reads = reads;
        self.writes = writes;
    }
}

impl fmt::Display for ComputeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "op#{} kind={} shape={} reads=[",
            self.id.0, self.kind, self.shape
        )?;
        for (i, h) in self.reads.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", h)?;
        }
        f.write_str("] writes=[")?;
        for (i, h) in self.writes.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", h)?;
        }
        f.write_str("]")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, CgExprId, ConfigConstId, EventRingAccess, EventRingId, MaskId,
        SpatialStorageKind, ViewId, ViewStorageSlot,
    };
    use crate::cg::expr::{BinaryOp, CgExpr, CgTy, LitValue};
    use crate::cg::stmt::{CgStmt, CgStmtId, CgStmtList, EventField};

    fn assert_roundtrip<T>(v: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        let json = serde_json::to_string(v).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, v, "round-trip changed value (json was {json})");
    }

    fn read_self_hp() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        })
    }

    fn read_self_pos() -> CgExpr {
        CgExpr::Read(DataHandle::AgentField {
            field: AgentFieldId::Pos,
            target: AgentRef::Self_,
        })
    }

    fn lit_f32(v: f32) -> CgExpr {
        CgExpr::Lit(LitValue::F32(v))
    }

    fn lit_bool(b: bool) -> CgExpr {
        CgExpr::Lit(LitValue::Bool(b))
    }

    // ---- Newtype roundtrips ----

    #[test]
    fn newtype_ids_roundtrip() {
        assert_roundtrip(&OpId(0));
        assert_roundtrip(&OpId(7));
        assert_roundtrip(&ScoringId(3));
        assert_roundtrip(&PhysicsRuleId(11));
        assert_roundtrip(&EventKindId(42));
        assert_roundtrip(&ActionId(5));
    }

    // ---- SpatialQueryKind ----

    #[test]
    fn spatial_query_kind_display_and_roundtrip() {
        let cases = [
            (SpatialQueryKind::BuildHash, "build_hash"),
            (SpatialQueryKind::KinQuery, "kin_query"),
            (SpatialQueryKind::EngagementQuery, "engagement_query"),
        ];
        for (kind, expected) in cases {
            assert_eq!(format!("{}", kind), expected);
            assert_roundtrip(&kind);
        }
    }

    #[test]
    fn spatial_query_kind_dependencies_build_hash() {
        let (r, w) = SpatialQueryKind::BuildHash.dependencies();
        assert!(r.is_empty());
        assert_eq!(
            w,
            vec![
                DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::GridCells,
                },
                DataHandle::SpatialStorage {
                    kind: SpatialStorageKind::GridOffsets,
                },
            ]
        );
    }

    #[test]
    fn spatial_query_kind_dependencies_kin_query() {
        let (r, w) = SpatialQueryKind::KinQuery.dependencies();
        assert_eq!(r.len(), 2);
        assert_eq!(
            w,
            vec![DataHandle::SpatialStorage {
                kind: SpatialStorageKind::QueryResults,
            }]
        );
    }

    #[test]
    fn spatial_query_kind_dependencies_engagement_query() {
        let (r, w) = SpatialQueryKind::EngagementQuery.dependencies();
        assert_eq!(r.len(), 2);
        assert_eq!(
            w,
            vec![DataHandle::SpatialStorage {
                kind: SpatialStorageKind::QueryResults,
            }]
        );
    }

    // ---- PlumbingKind ----

    #[test]
    fn plumbing_kind_display_and_roundtrip() {
        let cases = [
            (PlumbingKind::AlivePack, "alive_pack"),
            (PlumbingKind::FusedAgentUnpack, "fused_agent_unpack"),
            (PlumbingKind::SeedIndirect, "seed_indirect"),
            (PlumbingKind::SimCfgUpload, "sim_cfg_upload"),
            (
                PlumbingKind::EventRingDrain {
                    ring: EventRingId(2),
                },
                "event_ring_drain(ring=#2)",
            ),
        ];
        for (kind, expected) in cases {
            assert_eq!(format!("{}", kind), expected);
            assert_roundtrip(&kind);
        }
    }

    #[test]
    fn plumbing_alive_pack_dependencies() {
        let (r, w) = PlumbingKind::AlivePack.dependencies();
        assert_eq!(
            r,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(w, vec![DataHandle::MaskBitmap { mask: MaskId(0) }]);
    }

    #[test]
    fn plumbing_fused_agent_unpack_dependencies() {
        let (r, w) = PlumbingKind::FusedAgentUnpack.dependencies();
        assert_eq!(r.len(), 1);
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn plumbing_seed_indirect_dependencies() {
        let (r, w) = PlumbingKind::SeedIndirect.dependencies();
        assert_eq!(
            r,
            vec![DataHandle::EventRing {
                ring: EventRingId(0),
                kind: EventRingAccess::Read,
            }]
        );
        assert_eq!(
            w,
            vec![DataHandle::SpatialStorage {
                kind: SpatialStorageKind::QueryResults,
            }]
        );
    }

    #[test]
    fn plumbing_sim_cfg_upload_dependencies() {
        let (r, w) = PlumbingKind::SimCfgUpload.dependencies();
        assert!(r.is_empty());
        assert_eq!(
            w,
            vec![DataHandle::ConfigConst {
                id: ConfigConstId(0),
            }]
        );
    }

    #[test]
    fn plumbing_event_ring_drain_dependencies() {
        let (r, w) = PlumbingKind::EventRingDrain {
            ring: EventRingId(5),
        }
        .dependencies();
        assert!(r.is_empty());
        assert_eq!(
            w,
            vec![DataHandle::EventRing {
                ring: EventRingId(5),
                kind: EventRingAccess::Append,
            }]
        );
    }

    // ---- ScoringRowOp ----

    #[test]
    fn scoring_row_op_display_and_roundtrip() {
        let row = ScoringRowOp {
            action: ActionId(3),
            utility: CgExprId(7),
            target: CgExprId(9),
        };
        assert_eq!(
            format!("{}", row),
            "row(action=#3, utility=expr#7, target=expr#9)"
        );
        assert_roundtrip(&row);
    }

    // ---- ComputeOpKind ----

    #[test]
    fn compute_op_kind_label_distinct_per_variant() {
        let labels: Vec<String> = vec![
            ComputeOpKind::MaskPredicate {
                mask: MaskId(0),
                predicate: CgExprId(0),
            }
            .label(),
            ComputeOpKind::ScoringArgmax {
                scoring: ScoringId(0),
                rows: vec![],
            }
            .label(),
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(0),
                body: CgStmtListId(0),
            }
            .label(),
            ComputeOpKind::ViewFold {
                view: ViewId(0),
                on_event: EventKindId(0),
                body: CgStmtListId(0),
            }
            .label(),
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::BuildHash,
            }
            .label(),
            ComputeOpKind::Plumbing {
                kind: PlumbingKind::AlivePack,
            }
            .label(),
        ];
        let mut seen = std::collections::HashSet::new();
        for l in &labels {
            assert!(seen.insert(l.clone()), "duplicate label: {l}");
        }
    }

    // ---- compute_dependencies — per-kind ----

    #[test]
    fn mask_predicate_deps_walks_predicate_and_writes_bitmap() {
        // predicate: hp < 0.5
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(0.5),   // 1
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::Bool,
            }, // 2
        ];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        let kind = ComputeOpKind::MaskPredicate {
            mask: MaskId(4),
            predicate: CgExprId(2),
        };
        let (r, w) = kind.compute_dependencies(&exprs, &stmts, &lists);
        assert_eq!(
            r,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(w, vec![DataHandle::MaskBitmap { mask: MaskId(4) }]);
    }

    #[test]
    fn scoring_argmax_deps_walks_each_row() {
        // row 0: utility = hp + 1.0,  target = (lit u32 0)
        // row 1: utility = (read pos.x) — encoded as read_self_pos for
        //   the test (the collector doesn't decompose vec3),
        //   target = (read agent.self.engaged_with)
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(),                     // 0
            lit_f32(1.0),                       // 1
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::F32,
            }, // 2  -- row0 utility
            CgExpr::Lit(LitValue::AgentId(0)),  // 3  -- row0 target
            read_self_pos(),                    // 4  -- row1 utility (semantic-shape stand-in)
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::EngagedWith,
                target: AgentRef::Self_,
            }),                                  // 5  -- row1 target
        ];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(1),
            rows: vec![
                ScoringRowOp {
                    action: ActionId(0),
                    utility: CgExprId(2),
                    target: CgExprId(3),
                },
                ScoringRowOp {
                    action: ActionId(1),
                    utility: CgExprId(4),
                    target: CgExprId(5),
                },
            ],
        };
        let (r, w) = kind.compute_dependencies(&exprs, &stmts, &lists);
        // row0 contributes hp (utility) + nothing (lit target).
        // row1 contributes pos (utility) + engaged_with (target).
        assert_eq!(r.len(), 3);
        assert_eq!(
            r[0],
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }
        );
        assert_eq!(
            r[1],
            DataHandle::AgentField {
                field: AgentFieldId::Pos,
                target: AgentRef::Self_,
            }
        );
        assert_eq!(
            r[2],
            DataHandle::AgentField {
                field: AgentFieldId::EngagedWith,
                target: AgentRef::Self_,
            }
        );
        assert_eq!(w, vec![DataHandle::ScoringOutput]);
    }

    #[test]
    fn physics_rule_deps_walks_body_and_reads_event_ring() {
        // body: { if cond { assign(hp <- 0.0) } emit(event#9, ...) }
        let exprs: Vec<CgExpr> = vec![
            lit_bool(true), // 0  -- cond
            lit_f32(0.0),   // 1  -- new hp
            read_self_hp(), // 2  -- emit field
        ];
        let stmts: Vec<CgStmt> = vec![
            // 0: assign (in then-arm)
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: CgExprId(1),
            },
            // 1: if(cond, then=list#0, no else)
            CgStmt::If {
                cond: CgExprId(0),
                then: CgStmtListId(0),
                else_: None,
            },
            // 2: emit
            CgStmt::Emit {
                event: EventKindId(9),
                fields: vec![(
                    EventField {
                        event: EventKindId(9),
                        index: 0,
                    },
                    CgExprId(2),
                )],
            },
        ];
        let lists: Vec<CgStmtList> = vec![
            // 0: just the assign
            CgStmtList::new(vec![CgStmtId(0)]),
            // 1: top-level body — [if, emit]
            CgStmtList::new(vec![CgStmtId(1), CgStmtId(2)]),
        ];
        let kind = ComputeOpKind::PhysicsRule {
            rule: PhysicsRuleId(2),
            on_event: EventKindId(7),
            body: CgStmtListId(1),
        };
        let (r, w) = kind.compute_dependencies(&exprs, &stmts, &lists);
        // First read = the implicit event-ring read for on_event=7.
        assert_eq!(
            r[0],
            DataHandle::EventRing {
                ring: EventRingId(7),
                kind: EventRingAccess::Read,
            }
        );
        // Then: the if's cond is a lit (no read), then-arm assigns hp
        // (its value is a lit, no read), then emit's field is hp read.
        assert!(r.contains(&DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        }));
        // Writes: hp (from assign) + event-ring append (from emit).
        assert_eq!(w.len(), 2);
        assert_eq!(
            w[0],
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }
        );
        assert_eq!(
            w[1],
            DataHandle::EventRing {
                ring: EventRingId(9),
                kind: EventRingAccess::Append,
            }
        );
    }

    #[test]
    fn view_fold_deps_walks_body_and_writes_view_primary() {
        // body: { assign(view[#3].primary <- hp) }
        let exprs: Vec<CgExpr> = vec![read_self_hp()];
        let stmts: Vec<CgStmt> = vec![CgStmt::Assign {
            target: DataHandle::ViewStorage {
                view: ViewId(3),
                slot: ViewStorageSlot::Primary,
            },
            value: CgExprId(0),
        }];
        let lists: Vec<CgStmtList> = vec![CgStmtList::new(vec![CgStmtId(0)])];
        let kind = ComputeOpKind::ViewFold {
            view: ViewId(3),
            on_event: EventKindId(2),
            body: CgStmtListId(0),
        };
        let (r, w) = kind.compute_dependencies(&exprs, &stmts, &lists);
        // Reads: implicit event-ring read on event 2 + the body's hp read.
        assert_eq!(
            r,
            vec![
                DataHandle::EventRing {
                    ring: EventRingId(2),
                    kind: EventRingAccess::Read,
                },
                DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
            ]
        );
        // Writes: body's view-primary write + the appended fold-output
        // view-primary handle (duplicate-tolerant by design).
        assert_eq!(w.len(), 2);
        let view_primary = DataHandle::ViewStorage {
            view: ViewId(3),
            slot: ViewStorageSlot::Primary,
        };
        assert_eq!(w[0], view_primary);
        assert_eq!(w[1], view_primary);
    }

    #[test]
    fn spatial_query_kind_op_deps_match_kind_signature() {
        let exprs: Vec<CgExpr> = vec![];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        for kind in [
            SpatialQueryKind::BuildHash,
            SpatialQueryKind::KinQuery,
            SpatialQueryKind::EngagementQuery,
        ] {
            let op_kind = ComputeOpKind::SpatialQuery { kind };
            let (op_r, op_w) = op_kind.compute_dependencies(&exprs, &stmts, &lists);
            let (k_r, k_w) = kind.dependencies();
            assert_eq!(op_r, k_r);
            assert_eq!(op_w, k_w);
        }
    }

    #[test]
    fn plumbing_kind_op_deps_match_kind_signature() {
        let exprs: Vec<CgExpr> = vec![];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        for kind in [
            PlumbingKind::AlivePack,
            PlumbingKind::FusedAgentUnpack,
            PlumbingKind::SeedIndirect,
            PlumbingKind::SimCfgUpload,
            PlumbingKind::EventRingDrain {
                ring: EventRingId(11),
            },
        ] {
            let op_kind = ComputeOpKind::Plumbing { kind };
            let (op_r, op_w) = op_kind.compute_dependencies(&exprs, &stmts, &lists);
            let (k_r, k_w) = kind.dependencies();
            assert_eq!(op_r, k_r);
            assert_eq!(op_w, k_w);
        }
    }

    // ---- ComputeOp constructor + recompute ----

    #[test]
    fn compute_op_new_populates_reads_and_writes() {
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(0.5),   // 1
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::Bool,
            }, // 2
        ];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        let op = ComputeOp::new(
            OpId(0),
            ComputeOpKind::MaskPredicate {
                mask: MaskId(1),
                predicate: CgExprId(2),
            },
            DispatchShape::PerAgent,
            Span::dummy(),
            &exprs,
            &stmts,
            &lists,
        );
        assert_eq!(op.id, OpId(0));
        assert_eq!(op.shape, DispatchShape::PerAgent);
        assert_eq!(
            op.reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(op.writes, vec![DataHandle::MaskBitmap { mask: MaskId(1) }]);
    }

    #[test]
    fn compute_op_recompute_after_kind_mutation() {
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(0.5),   // 1
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::Bool,
            }, // 2
            read_self_pos(), // 3 — never read
            // a different predicate: hp == 0
            lit_f32(0.0), // 4
            CgExpr::Binary {
                op: BinaryOp::EqF32,
                lhs: CgExprId(0),
                rhs: CgExprId(4),
                ty: CgTy::Bool,
            }, // 5
        ];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        let mut op = ComputeOp::new(
            OpId(0),
            ComputeOpKind::MaskPredicate {
                mask: MaskId(1),
                predicate: CgExprId(2),
            },
            DispatchShape::PerAgent,
            Span::dummy(),
            &exprs,
            &stmts,
            &lists,
        );
        let original_reads = op.reads.clone();
        let original_writes = op.writes.clone();

        // Mutate kind to use a different mask + predicate that walks
        // the same expression set (still hp).
        op.kind = ComputeOpKind::MaskPredicate {
            mask: MaskId(2),
            predicate: CgExprId(5),
        };
        op.recompute_dependencies(&exprs, &stmts, &lists);

        // Reads still hp (the new predicate also walks expr#0).
        assert_eq!(op.reads, original_reads);
        // Writes have changed — new mask id.
        assert_ne!(op.writes, original_writes);
        assert_eq!(op.writes, vec![DataHandle::MaskBitmap { mask: MaskId(2) }]);
    }

    // ---- Determinism ----

    #[test]
    fn deterministic_walk_order_for_repeated_constructions() {
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(),  // 0
            read_self_pos(), // 1
            CgExpr::Binary {
                op: BinaryOp::AddF32,
                lhs: CgExprId(0),
                rhs: CgExprId(0),
                ty: CgTy::F32,
            }, // 2  — duplicate hp read
        ];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        let kind_a = ComputeOpKind::MaskPredicate {
            mask: MaskId(0),
            predicate: CgExprId(2),
        };
        let kind_b = ComputeOpKind::MaskPredicate {
            mask: MaskId(0),
            predicate: CgExprId(2),
        };
        let (ra, _) = kind_a.compute_dependencies(&exprs, &stmts, &lists);
        let (rb, _) = kind_b.compute_dependencies(&exprs, &stmts, &lists);
        assert_eq!(ra, rb, "two identical kinds must produce identical reads");
        // Duplicates retained — both lhs and rhs read hp.
        assert_eq!(ra.len(), 2);
        assert_eq!(
            ra[0],
            DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }
        );
        assert_eq!(ra[1], ra[0]);
    }

    // ---- Display for ComputeOp ----

    #[test]
    fn compute_op_display_format_pinned() {
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(0.5),   // 1
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::Bool,
            }, // 2
        ];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        let op = ComputeOp::new(
            OpId(7),
            ComputeOpKind::MaskPredicate {
                mask: MaskId(3),
                predicate: CgExprId(2),
            },
            DispatchShape::PerAgent,
            Span::dummy(),
            &exprs,
            &stmts,
            &lists,
        );
        assert_eq!(
            format!("{}", op),
            "op#7 kind=mask_predicate(mask=#3) shape=per_agent reads=[agent.self.hp] writes=[mask[#3].bitmap]"
        );
    }

    // ---- ComputeOp serde round-trip (sans span) ----

    #[test]
    fn compute_op_roundtrips_kind_reads_writes_shape() {
        let exprs: Vec<CgExpr> = vec![
            read_self_hp(), // 0
            lit_f32(0.5),   // 1
            CgExpr::Binary {
                op: BinaryOp::LtF32,
                lhs: CgExprId(0),
                rhs: CgExprId(1),
                ty: CgTy::Bool,
            }, // 2
        ];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        let op = ComputeOp::new(
            OpId(0),
            ComputeOpKind::MaskPredicate {
                mask: MaskId(1),
                predicate: CgExprId(2),
            },
            DispatchShape::PerAgent,
            Span::new(10, 20),
            &exprs,
            &stmts,
            &lists,
        );
        let json = serde_json::to_string(&op).expect("serialize");
        let back: ComputeOp = serde_json::from_str(&json).expect("deserialize");
        // Span is reset to dummy on round-trip (Span has no
        // Deserialize), but everything else round-trips intact.
        assert_eq!(back.id, op.id);
        assert_eq!(back.kind, op.kind);
        assert_eq!(back.reads, op.reads);
        assert_eq!(back.writes, op.writes);
        assert_eq!(back.shape, op.shape);
        assert_eq!(back.span, Span::dummy());
    }

    #[test]
    fn compute_op_kind_roundtrips_each_variant() {
        let kinds = vec![
            ComputeOpKind::MaskPredicate {
                mask: MaskId(1),
                predicate: CgExprId(2),
            },
            ComputeOpKind::ScoringArgmax {
                scoring: ScoringId(0),
                rows: vec![ScoringRowOp {
                    action: ActionId(0),
                    utility: CgExprId(1),
                    target: CgExprId(2),
                }],
            },
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(1),
                body: CgStmtListId(0),
            },
            ComputeOpKind::ViewFold {
                view: ViewId(0),
                on_event: EventKindId(1),
                body: CgStmtListId(0),
            },
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::BuildHash,
            },
            ComputeOpKind::Plumbing {
                kind: PlumbingKind::AlivePack,
            },
        ];
        for k in &kinds {
            let json = serde_json::to_string(k).expect("serialize");
            let back: ComputeOpKind = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(&back, k, "kind round-trip mismatch (json was {json})");
        }
    }
}
