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
    AgentFieldId, AgentRef, AgentScratchKind, DataHandle, EventRingAccess, EventRingId, MaskId,
    SpatialStorageKind, ViewId,
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
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OpId(pub u32);

/// Stable id for a `scoring` declaration. One per top-level scoring
/// block in the DSL source; assignment order matches resolved IR
/// order so snapshots are reproducible.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScoringId(pub u32);

/// Stable id for a `physics` rule. One per physics declaration; the
/// rule's `on Event => …` handlers each become a [`ComputeOpKind::PhysicsRule`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PhysicsRuleId(pub u32);

/// Stable id for an event variant. One per `event` declaration in the
/// DSL. Used by [`ComputeOpKind::PhysicsRule`] /
/// [`ComputeOpKind::ViewFold`] to name which event triggers the body
/// and by [`crate::cg::stmt::CgStmt::Emit`] to name the variant being
/// emitted.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EventKindId(pub u32);

/// Stable id for an action — the head of a scoring row. Each row's
/// argmax candidate is identified by an `ActionId`; the engine maps
/// the winning id to a concrete behaviour at apply time.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActionId(pub u32);

// ---------------------------------------------------------------------------
// ReplayabilityFlag — typed P7 replayability binding for PhysicsRule
// ---------------------------------------------------------------------------

/// Per-rule replayability flag carried on
/// [`ComputeOpKind::PhysicsRule`]. Propagated from the constitution P7
/// surface (the rule's `@phase(...)` annotation) into the lowered op
/// so emit can sort the rule's emissions into the right ring.
///
/// Encoded as a typed enum rather than a bare `bool` so call sites
/// read self-explanatory (`ReplayabilityFlag::Replayable` rather than
/// a positional `true`). Style matches the other Phase 1 closed-set
/// kinds in this module ([`SpatialQueryKind`],
/// [`crate::cg::data_handle::EventRingAccess`]).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReplayabilityFlag {
    /// `@phase(event)` — emissions land in the deterministic ring
    /// the runtime folds into the trace hash.
    Replayable,
    /// `@phase(post)` — emissions land in chronicle / telemetry
    /// rings the runtime fold ignores.
    NonReplayable,
}

impl ReplayabilityFlag {
    /// Stable snake_case label for pretty-printing.
    pub fn label(self) -> &'static str {
        match self {
            ReplayabilityFlag::Replayable => "replayable",
            ReplayabilityFlag::NonReplayable => "non_replayable",
        }
    }

    /// Convert to a `bool` for downstream emit consumers that key off
    /// the binary distinction. The IR-canonical carrier is the enum;
    /// this helper exists so emit code that historically read a
    /// `bool` field can keep its call sites unchanged.
    pub fn as_bool(self) -> bool {
        matches!(self, ReplayabilityFlag::Replayable)
    }
}

impl fmt::Display for ReplayabilityFlag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

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
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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
// PlumbingKind — Task 2.7
// ---------------------------------------------------------------------------

/// One-shot plumbing kinds — emit-strategy operations that are not
/// directly user-authored DSL declarations but still appear in the
/// compute graph because the schedule needs them.
///
/// Variants enumerate each piece of "between-kernel" work the runtime
/// performs every tick: packing/unpacking agents, refreshing the alive
/// bitmap, draining cascade-event rings, uploading sim_cfg, kicking the
/// snapshot, and seeding indirect-dispatch args. Each variant carries
/// its full structural read/write set via
/// [`PlumbingKind::dependencies`] — no sentinel ids, no `String` keys,
/// every read and write is a typed [`DataHandle`].
///
/// The synthesizer (`synthesize_plumbing_ops`) walks a program's user-
/// facing ops and decides which plumbing kinds are needed; the lowering
/// (`lower_plumbing`) pushes them onto the builder.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PlumbingKind {
    /// Pack every per-agent SoA field into the
    /// [`AgentScratchKind::Packed`] scratch buffer for the GPU upload.
    /// Reads every [`AgentFieldId`] variant; writes the packed scratch.
    /// Per-agent dispatch.
    PackAgents,

    /// Unpack the [`AgentScratchKind::Packed`] scratch buffer back into
    /// per-field SoA storage after the GPU dispatch finishes. Reads the
    /// packed scratch; writes every [`AgentFieldId`] variant. Per-agent
    /// dispatch.
    UnpackAgents,

    /// Refresh the [`DataHandle::AliveBitmap`] (one bit per agent) from
    /// the per-agent [`AgentFieldId::Alive`] field. Per-word dispatch
    /// (each thread owns one 32-agent word).
    AliveBitmap,

    /// Drain a cascade-event ring after the apply pass consumed its
    /// entries. Reads the ring with [`EventRingAccess::Drain`]; writes
    /// nothing structural at the IR level (the drain mutates the ring's
    /// internal tail counter, but that is encoded by the access mode
    /// rather than as a separate write target). Per-event dispatch on
    /// the ring it drains.
    DrainEvents { ring: EventRingId },

    /// Upload the sim_cfg uniform buffer to the GPU. Reads nothing the
    /// IR can see structurally (the data lives on the host); writes
    /// [`DataHandle::SimCfgBuffer`] as a whole-buffer write. One-shot.
    UploadSimCfg,

    /// Trigger the snapshot dump for this tick. Reads nothing
    /// structural; writes [`DataHandle::SnapshotKick`]. One-shot.
    KickSnapshot,

    /// Seed the indirect-dispatch args buffer for `ring`'s next per-
    /// event dispatch from the ring's current tail count. Reads the
    /// ring with [`EventRingAccess::Read`] (count only); writes
    /// [`DataHandle::IndirectArgs { ring }`]. One-shot.
    SeedIndirectArgs { ring: EventRingId },
}

impl PlumbingKind {
    /// Stable snake_case label for diagnostics + pretty-printing.
    pub fn label(&self) -> String {
        match self {
            PlumbingKind::PackAgents => String::from("pack_agents"),
            PlumbingKind::UnpackAgents => String::from("unpack_agents"),
            PlumbingKind::AliveBitmap => String::from("alive_bitmap"),
            PlumbingKind::DrainEvents { ring } => {
                format!("drain_events(ring=#{})", ring.0)
            }
            PlumbingKind::UploadSimCfg => String::from("upload_sim_cfg"),
            PlumbingKind::KickSnapshot => String::from("kick_snapshot"),
            PlumbingKind::SeedIndirectArgs { ring } => {
                format!("seed_indirect_args(ring=#{})", ring.0)
            }
        }
    }

    /// Structural `(reads, writes)` signature for the plumbing kind.
    /// Every entry is a typed [`DataHandle`]; no sentinel ids. Mirrors
    /// the `dependencies()` method on [`SpatialQueryKind`] so the
    /// auto-walker (`ComputeOpKind::compute_dependencies`) has a
    /// single source of truth for plumbing reads/writes.
    pub fn dependencies(&self) -> (Vec<DataHandle>, Vec<DataHandle>) {
        match self {
            PlumbingKind::PackAgents => (
                AgentFieldId::all_agent_field_handles(),
                vec![DataHandle::AgentScratch {
                    kind: AgentScratchKind::Packed,
                }],
            ),
            PlumbingKind::UnpackAgents => (
                vec![DataHandle::AgentScratch {
                    kind: AgentScratchKind::Packed,
                }],
                AgentFieldId::all_agent_field_handles(),
            ),
            PlumbingKind::AliveBitmap => (
                vec![DataHandle::AgentField {
                    field: AgentFieldId::Alive,
                    target: AgentRef::Self_,
                }],
                vec![DataHandle::AliveBitmap],
            ),
            PlumbingKind::DrainEvents { ring } => (
                vec![DataHandle::EventRing {
                    ring: *ring,
                    kind: EventRingAccess::Drain,
                }],
                vec![],
            ),
            PlumbingKind::UploadSimCfg => (vec![], vec![DataHandle::SimCfgBuffer]),
            PlumbingKind::KickSnapshot => (vec![], vec![DataHandle::SnapshotKick]),
            PlumbingKind::SeedIndirectArgs { ring } => (
                vec![DataHandle::EventRing {
                    ring: *ring,
                    kind: EventRingAccess::Read,
                }],
                vec![DataHandle::IndirectArgs { ring: *ring }],
            ),
        }
    }

    /// Canonical [`DispatchShape`] for the plumbing kind.
    ///
    /// Each plumbing kind has a single natural shape: per-agent for
    /// pack/unpack, per-word for the alive bitmap, per-event for the
    /// drain (the ring drives the count), and one-shot for the
    /// scalar-output variants (sim_cfg upload, snapshot kick, indirect-
    /// args seed). Returning the shape from the kind keeps the lowering
    /// pass from re-deriving it per variant.
    pub fn dispatch_shape(&self) -> DispatchShape {
        match self {
            PlumbingKind::PackAgents => DispatchShape::PerAgent,
            PlumbingKind::UnpackAgents => DispatchShape::PerAgent,
            PlumbingKind::AliveBitmap => DispatchShape::PerWord,
            PlumbingKind::DrainEvents { ring } => DispatchShape::PerEvent {
                source_ring: *ring,
            },
            PlumbingKind::UploadSimCfg => DispatchShape::OneShot,
            PlumbingKind::KickSnapshot => DispatchShape::OneShot,
            PlumbingKind::SeedIndirectArgs { .. } => DispatchShape::OneShot,
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
/// (the score) plus optional target / guard expressions.
///
/// **Field optionality.** The DSL surface today carries TWO scoring-row
/// shapes (see [`dsl_ast::ir::ScoringRowKind`]):
///
/// - **Standard rows** (`Head = expr`) — the row's target is implicit
///   in the action at runtime (the engine resolves which agent to
///   apply against based on the action kind and the per-action
///   selector). Standard rows lower with `target = None, guard =
///   None`.
/// - **Per-ability rows** (`row <name> per_ability { guard, score,
///   target }`) — the row carries an explicit target expression
///   (typed `AgentId`) and an optional guard predicate (typed
///   `Bool`). Per-ability rows populate `target` / `guard` directly
///   from the AST.
///
/// Carrying both fields as `Option<CgExprId>` keeps the shape honest
/// at the IR level: a synthetic placeholder for standard rows would
/// have no defined runtime semantics, and routing a guard through a
/// separately-named field (rather than baking it into `utility` via
/// `if guard then score else -inf`) lets the well-formed pass type-
/// check guard separately and lets the kernel emitter short-circuit
/// the row when guard is false.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ScoringRowOp {
    pub action: ActionId,
    /// Score / utility expression — required, must type-check to
    /// [`crate::cg::expr::CgTy::F32`] (per the well-formed pass).
    pub utility: super::data_handle::CgExprId,
    /// Per-ability target agent-id expression. `None` for standard
    /// rows; populated only by per-ability row lowering. The well-
    /// formed pass type-checks `Some` values to
    /// [`crate::cg::expr::CgTy::AgentId`] via
    /// [`crate::cg::well_formed::CgError::ScoringTargetNotAgentId`].
    pub target: Option<super::data_handle::CgExprId>,
    /// Per-ability guard predicate. `None` for standard rows AND for
    /// per-ability rows that carry no guard (`guard = None` parses
    /// as `true`); populated only when the AST surfaces an explicit
    /// boolean predicate. The well-formed pass type-checks `Some`
    /// values to [`crate::cg::expr::CgTy::Bool`].
    pub guard: Option<super::data_handle::CgExprId>,
}

impl fmt::Display for ScoringRowOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "row(action=#{}, utility=expr#{}", self.action.0, self.utility.0)?;
        match self.target {
            Some(t) => write!(f, ", target=Some(expr#{})", t.0)?,
            None => write!(f, ", target=None")?,
        }
        match self.guard {
            Some(g) => write!(f, ", guard=Some(expr#{}))", g.0)?,
            None => write!(f, ", guard=None)")?,
        }
        Ok(())
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
    ///
    /// `replayable` propagates the constitution P7 flag from the IR.
    /// Replayable rules (`@phase(event)`) emit into the deterministic
    /// ring that folds into the trace hash; non-replayable rules
    /// (`@phase(post)`) emit into chronicle / telemetry rings that
    /// the runtime fold ignores. The driver computes the flag from
    /// the rule's `@phase(...)` annotation and threads it through
    /// lowering so emit can sort the rule into the right ring.
    PhysicsRule {
        rule: PhysicsRuleId,
        on_event: EventKindId,
        body: CgStmtListId,
        replayable: ReplayabilityFlag,
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
    ///
    /// **Scope of the auto-walker.** This walker reports only the
    /// dependencies that are structurally derivable from the IR's
    /// expression and statement trees. It does NOT synthesize:
    ///
    /// - The source event ring read for [`PhysicsRule`] /
    ///   [`ViewFold`]. The ring identity is recorded in
    ///   [`DispatchShape::PerEvent { source_ring }`], populated by
    ///   the lowering pass that consults the event registry.
    /// - The target event ring write for [`crate::cg::stmt::CgStmt::Emit`].
    ///   The walker descends into each field-value expression for
    ///   reads, but the ring binding is added by the AST → HIR
    ///   lowering pass via [`ComputeOp::record_write`].
    /// - View-storage writes inside a [`ViewFold`] body. The walker
    ///   records whatever real `Assign` targets the body contains; if
    ///   the body writes nothing, that's a real signal, not a defect
    ///   to paper over.
    ///
    /// [`PhysicsRule`]: ComputeOpKind::PhysicsRule
    /// [`ViewFold`]: ComputeOpKind::ViewFold
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
                    if let Some(target_id) = row.target {
                        collect_expr_reads(target_id, exprs, &mut reads);
                    }
                    if let Some(guard_id) = row.guard {
                        collect_expr_reads(guard_id, exprs, &mut reads);
                    }
                }
                writes.push(DataHandle::ScoringOutput);
            }
            ComputeOpKind::PhysicsRule {
                rule: _,
                on_event: _,
                body,
                replayable: _,
            } => {
                // Source event ring read is recorded in
                // `DispatchShape::PerEvent { source_ring }`, populated
                // by lowering. The auto-walker only reports what the
                // body's statements/expressions touch.
                //
                // The `replayable` flag is metadata (it informs which
                // ring `Emit` writes into, captured via
                // `record_write` post-construction by the driver); it
                // does not contribute structural reads/writes the
                // walker can synthesize.
                collect_list_dependencies(*body, exprs, stmts, lists, &mut reads, &mut writes);
            }
            ComputeOpKind::ViewFold {
                view: _,
                on_event: _,
                body,
            } => {
                // Source event ring read is recorded in
                // `DispatchShape::PerEvent { source_ring }`, populated
                // by lowering. View-storage writes are recorded by
                // `Assign` statements in the body, also populated by
                // lowering — if the body writes nothing, the op
                // writes nothing.
                collect_list_dependencies(*body, exprs, stmts, lists, &mut reads, &mut writes);
            }
            ComputeOpKind::SpatialQuery { kind } => {
                let (r, w) = kind.dependencies();
                reads.extend(r);
                writes.extend(w);
            }
            ComputeOpKind::Plumbing { kind } => {
                // Plumbing kinds carry their (reads, writes) signature
                // on `PlumbingKind::dependencies()` — single source of
                // truth shared with the schedule synthesizer.
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
            ComputeOpKind::PhysicsRule {
                rule,
                on_event,
                replayable,
                ..
            } => {
                format!(
                    "physics_rule(rule=#{}, on_event=#{}, replayable={})",
                    rule.0,
                    on_event.0,
                    replayable.label()
                )
            }
            ComputeOpKind::ViewFold { view, on_event, .. } => {
                format!("view_fold(view=#{}, on_event=#{})", view.0, on_event.0)
            }
            ComputeOpKind::SpatialQuery { kind } => {
                format!("spatial_query({})", kind)
            }
            ComputeOpKind::Plumbing { kind } => format!("plumbing({})", kind.label()),
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

    /// Append a read recorded by the lowering pass.
    ///
    /// The auto-walker covers what the IR can know structurally — the
    /// reads/writes its expression and statement trees express
    /// directly. Lowering uses this method (and [`Self::record_write`])
    /// to add bindings the walker can't synthesize: the source ring
    /// identity behind a [`DispatchShape::PerEvent`] dispatch, the
    /// destination ring an [`crate::cg::stmt::CgStmt::Emit`] resolves
    /// to, and similar registry-resolved bindings.
    ///
    /// Lowering is responsible for calling this exactly once per such
    /// dependency. Duplicate entries are tolerated by downstream
    /// consumers (the auto-walker itself records duplicates), so this
    /// method does no de-duplication.
    pub fn record_read(&mut self, handle: DataHandle) {
        self.reads.push(handle);
    }

    /// Append a write recorded by the lowering pass. See
    /// [`Self::record_read`] for the rationale.
    pub fn record_write(&mut self, handle: DataHandle) {
        self.writes.push(handle);
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
        AgentFieldId, AgentRef, CgExprId, EventRingAccess, EventRingId, MaskId, SpatialStorageKind,
        ViewId, ViewStorageSlot,
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

    // ---- PlumbingKind (Task 2.7) ----

    #[test]
    fn plumbing_kind_display_and_roundtrip() {
        let cases = [
            (PlumbingKind::PackAgents, "pack_agents"),
            (PlumbingKind::UnpackAgents, "unpack_agents"),
            (PlumbingKind::AliveBitmap, "alive_bitmap"),
            (
                PlumbingKind::DrainEvents {
                    ring: EventRingId(2),
                },
                "drain_events(ring=#2)",
            ),
            (PlumbingKind::UploadSimCfg, "upload_sim_cfg"),
            (PlumbingKind::KickSnapshot, "kick_snapshot"),
            (
                PlumbingKind::SeedIndirectArgs {
                    ring: EventRingId(5),
                },
                "seed_indirect_args(ring=#5)",
            ),
        ];
        for (kind, expected) in cases {
            assert_eq!(format!("{}", kind), expected);
            assert_roundtrip(&kind);
        }
    }

    #[test]
    fn plumbing_kind_dispatch_shape_per_variant() {
        assert_eq!(PlumbingKind::PackAgents.dispatch_shape(), DispatchShape::PerAgent);
        assert_eq!(
            PlumbingKind::UnpackAgents.dispatch_shape(),
            DispatchShape::PerAgent
        );
        assert_eq!(PlumbingKind::AliveBitmap.dispatch_shape(), DispatchShape::PerWord);
        assert_eq!(
            PlumbingKind::DrainEvents {
                ring: EventRingId(2)
            }
            .dispatch_shape(),
            DispatchShape::PerEvent {
                source_ring: EventRingId(2),
            }
        );
        assert_eq!(PlumbingKind::UploadSimCfg.dispatch_shape(), DispatchShape::OneShot);
        assert_eq!(PlumbingKind::KickSnapshot.dispatch_shape(), DispatchShape::OneShot);
        assert_eq!(
            PlumbingKind::SeedIndirectArgs {
                ring: EventRingId(5)
            }
            .dispatch_shape(),
            DispatchShape::OneShot
        );
    }

    #[test]
    fn plumbing_kind_dependencies_pack_unpack_symmetric() {
        let (pack_r, pack_w) = PlumbingKind::PackAgents.dependencies();
        let (unpack_r, unpack_w) = PlumbingKind::UnpackAgents.dependencies();
        // PackAgents reads every AgentField, UnpackAgents writes them.
        assert_eq!(pack_r, unpack_w);
        // PackAgents writes the packed scratch, UnpackAgents reads it.
        assert_eq!(pack_w, unpack_r);
        assert_eq!(pack_r.len(), AgentFieldId::all_variants().len());
    }

    #[test]
    fn plumbing_kind_dependencies_alive_bitmap_reads_self_alive() {
        let (r, w) = PlumbingKind::AliveBitmap.dependencies();
        assert_eq!(
            r,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(w, vec![DataHandle::AliveBitmap]);
    }

    #[test]
    fn plumbing_kind_dependencies_drain_events_uses_drain_access() {
        let (r, w) = PlumbingKind::DrainEvents {
            ring: EventRingId(7),
        }
        .dependencies();
        assert_eq!(
            r,
            vec![DataHandle::EventRing {
                ring: EventRingId(7),
                kind: EventRingAccess::Drain,
            }]
        );
        assert!(w.is_empty());
    }

    #[test]
    fn plumbing_kind_dependencies_seed_indirect_args_pairs_ring_with_indirect() {
        let (r, w) = PlumbingKind::SeedIndirectArgs {
            ring: EventRingId(3),
        }
        .dependencies();
        assert_eq!(
            r,
            vec![DataHandle::EventRing {
                ring: EventRingId(3),
                kind: EventRingAccess::Read,
            }]
        );
        assert_eq!(
            w,
            vec![DataHandle::IndirectArgs {
                ring: EventRingId(3),
            }]
        );
    }

    // ---- ScoringRowOp ----

    #[test]
    fn scoring_row_op_display_and_roundtrip_standard_row() {
        // Standard row (no target, no guard) — both fields None.
        let row = ScoringRowOp {
            action: ActionId(3),
            utility: CgExprId(7),
            target: None,
            guard: None,
        };
        assert_eq!(
            format!("{}", row),
            "row(action=#3, utility=expr#7, target=None, guard=None)"
        );
        assert_roundtrip(&row);
    }

    #[test]
    fn scoring_row_op_display_and_roundtrip_per_ability_row() {
        // Per-ability row with both target + guard populated.
        let row = ScoringRowOp {
            action: ActionId(3),
            utility: CgExprId(7),
            target: Some(CgExprId(9)),
            guard: Some(CgExprId(11)),
        };
        assert_eq!(
            format!("{}", row),
            "row(action=#3, utility=expr#7, target=Some(expr#9), guard=Some(expr#11))"
        );
        assert_roundtrip(&row);
    }

    #[test]
    fn scoring_row_op_display_per_ability_row_target_only() {
        // Per-ability row with target but no guard.
        let row = ScoringRowOp {
            action: ActionId(0),
            utility: CgExprId(1),
            target: Some(CgExprId(2)),
            guard: None,
        };
        assert_eq!(
            format!("{}", row),
            "row(action=#0, utility=expr#1, target=Some(expr#2), guard=None)"
        );
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
                replayable: ReplayabilityFlag::Replayable,
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
                kind: PlumbingKind::PackAgents,
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
                    target: Some(CgExprId(3)),
                    guard: None,
                },
                ScoringRowOp {
                    action: ActionId(1),
                    utility: CgExprId(4),
                    target: Some(CgExprId(5)),
                    guard: None,
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
    fn physics_rule_deps_walks_body_only() {
        // body: { if cond { assign(hp <- 0.0) } emit(event#9, ...) }
        //
        // Auto-walker reports only what the body's statements/exprs
        // touch. The source event ring read for on_event=7 is recorded
        // in `DispatchShape::PerEvent { source_ring }` by lowering, not
        // synthesized here. The destination ring write for the `Emit`
        // is added by lowering via `record_write`.
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
            replayable: ReplayabilityFlag::Replayable,
        };
        let (r, w) = kind.compute_dependencies(&exprs, &stmts, &lists);
        // Reads: the if's cond is a lit (no read), then-arm assigns hp
        // (its value is a lit, no read), then emit's field is hp read.
        // No synthesized event-ring read.
        assert_eq!(
            r,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        // Writes: hp (from the assign) only. The emit's destination
        // ring is bound by lowering, not by the auto-walker.
        assert_eq!(
            w,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
    }

    #[test]
    fn view_fold_deps_walks_body_only() {
        // body: { assign(view[#3].primary <- hp) }
        //
        // Auto-walker reports only what the body's statements/exprs
        // touch. The source event ring read is recorded in
        // `DispatchShape::PerEvent` by lowering; no defensive
        // view-storage write is synthesized — if the body writes the
        // view-primary slot, that real `Assign` is what surfaces.
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
        // Reads: just the body's hp read — no synthesized event-ring read.
        assert_eq!(
            r,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }]
        );
        // Writes: just the body's view-primary write — no defensive
        // duplicate.
        assert_eq!(
            w,
            vec![DataHandle::ViewStorage {
                view: ViewId(3),
                slot: ViewStorageSlot::Primary,
            }]
        );
    }

    #[test]
    fn view_fold_with_empty_body_has_no_writes() {
        // A view fold with an empty body produces no writes — the
        // auto-walker reports the real signal (this body writes
        // nothing) instead of papering over with a synthesized
        // ViewStorage handle.
        let exprs: Vec<CgExpr> = vec![];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![CgStmtList::new(vec![])];
        let kind = ComputeOpKind::ViewFold {
            view: ViewId(3),
            on_event: EventKindId(2),
            body: CgStmtListId(0),
        };
        let (r, w) = kind.compute_dependencies(&exprs, &stmts, &lists);
        assert!(r.is_empty(), "expected no reads, got {r:?}");
        assert!(w.is_empty(), "expected no writes, got {w:?}");
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
        // Every plumbing kind's lowered op's reads/writes vectors must
        // equal the `(reads, writes)` table on `PlumbingKind` — the
        // auto-walker (Task 1.3) is the single source of truth and
        // this test pins that the `Plumbing` arm of
        // `compute_dependencies` routes through `dependencies()`.
        let exprs: Vec<CgExpr> = vec![];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![];
        for kind in [
            PlumbingKind::PackAgents,
            PlumbingKind::UnpackAgents,
            PlumbingKind::AliveBitmap,
            PlumbingKind::DrainEvents {
                ring: EventRingId(2),
            },
            PlumbingKind::UploadSimCfg,
            PlumbingKind::KickSnapshot,
            PlumbingKind::SeedIndirectArgs {
                ring: EventRingId(3),
            },
        ] {
            let op_kind = ComputeOpKind::Plumbing { kind };
            let (op_r, op_w) = op_kind.compute_dependencies(&exprs, &stmts, &lists);
            let (k_r, k_w) = kind.dependencies();
            assert_eq!(op_r, k_r, "plumbing kind {kind}: reads diverged");
            assert_eq!(op_w, k_w, "plumbing kind {kind}: writes diverged");
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
                    target: Some(CgExprId(2)),
                    guard: None,
                }],
            },
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(1),
                body: CgStmtListId(0),
                replayable: ReplayabilityFlag::Replayable,
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
                kind: PlumbingKind::PackAgents,
            },
            ComputeOpKind::Plumbing {
                kind: PlumbingKind::DrainEvents {
                    ring: EventRingId(2),
                },
            },
            ComputeOpKind::Plumbing {
                kind: PlumbingKind::SeedIndirectArgs {
                    ring: EventRingId(5),
                },
            },
        ];
        for k in &kinds {
            let json = serde_json::to_string(k).expect("serialize");
            let back: ComputeOpKind = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(&back, k, "kind round-trip mismatch (json was {json})");
        }
    }

    // ---- record_read / record_write ----

    #[test]
    fn record_write_appends_lowering_supplied_handle() {
        // Construct a physics-rule op (auto-walker captures only the
        // body's writes; lowering then records the destination event
        // ring for the embedded Emit).
        let exprs: Vec<CgExpr> = vec![read_self_hp()];
        let stmts: Vec<CgStmt> = vec![CgStmt::Emit {
            event: EventKindId(9),
            fields: vec![(
                EventField {
                    event: EventKindId(9),
                    index: 0,
                },
                CgExprId(0),
            )],
        }];
        let lists: Vec<CgStmtList> = vec![CgStmtList::new(vec![CgStmtId(0)])];
        let mut op = ComputeOp::new(
            OpId(0),
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(7),
                body: CgStmtListId(0),
                replayable: ReplayabilityFlag::Replayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(7),
            },
            Span::dummy(),
            &exprs,
            &stmts,
            &lists,
        );
        // Lowering supplies the destination ring write.
        let dest = DataHandle::EventRing {
            ring: EventRingId(9),
            kind: EventRingAccess::Append,
        };
        assert!(!op.writes.contains(&dest));
        op.record_write(dest.clone());
        assert!(op.writes.contains(&dest));
    }

    #[test]
    fn record_read_appends_lowering_supplied_handle() {
        let exprs: Vec<CgExpr> = vec![];
        let stmts: Vec<CgStmt> = vec![];
        let lists: Vec<CgStmtList> = vec![CgStmtList::new(vec![])];
        let mut op = ComputeOp::new(
            OpId(0),
            ComputeOpKind::PhysicsRule {
                rule: PhysicsRuleId(0),
                on_event: EventKindId(7),
                body: CgStmtListId(0),
                replayable: ReplayabilityFlag::Replayable,
            },
            DispatchShape::PerEvent {
                source_ring: EventRingId(7),
            },
            Span::dummy(),
            &exprs,
            &stmts,
            &lists,
        );
        // Lowering supplies the source ring read.
        let src = DataHandle::EventRing {
            ring: EventRingId(7),
            kind: EventRingAccess::Read,
        };
        assert!(op.reads.is_empty());
        op.record_read(src.clone());
        assert_eq!(op.reads, vec![src]);
    }
}
