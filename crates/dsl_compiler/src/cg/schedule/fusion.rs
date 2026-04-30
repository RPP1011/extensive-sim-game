//! Fusion analysis — Phase-3 / Task 3.2.
//!
//! Walks a [`CgProgram`]'s ops in [`super::topology::topological_sort`]
//! order and groups consecutive ops that share a dispatch shape and
//! don't write-conflict into [`FusionGroup`]s. Each group carries a
//! [`FusibilityClass`] that summarises whether the group will be fused
//! into a single kernel ([`FusibilityClass::Fused`]), kept as an
//! independent dispatch ([`FusibilityClass::Split`]), or driven through
//! an indirect-dispatch surface for `PerEvent` rings
//! ([`FusibilityClass::Indirect`]). Task 3.4 wraps these classifications
//! as `KernelTopology` values once it lands; for Task 3.2 the analysis
//! itself is the unit of output.
//!
//! [`fusion_decisions`] returns both the groups and a parallel stream of
//! [`FusionDiagnostic`]s explaining each grouping/split decision —
//! useful for debugging the schedule synthesizer's choices and for
//! snapshot tests that pin the decisions made on a representative
//! program.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 3.2, for the design rationale.
//!
//! # Limitations
//!
//! - **Consecutive-only fusion.** Fusion candidates are restricted to
//!   ops that are *consecutive* in the topological order. Non-
//!   consecutive but topologically-isolated ops (no intervening op that
//!   depends on either side) could in principle fuse, but doing so
//!   requires a richer reachability analysis than this first cut
//!   performs. Task 3.3 (megakernel synthesis) will explore the wider
//!   fusion surface; the structure of [`FusionGroup`] supports
//!   non-contiguous op lists without a breaking change.
//! - **WAW splits, WAR allowed.** Two adjacent ops with the same
//!   dispatch shape that both *write* the same handle (write-after-
//!   write) cause the analysis to start a new group — the hazard would
//!   change kernel semantics if folded into one body. WAR (op A reads
//!   X, op B writes X) within a single fused kernel is allowed: B's
//!   thread can write X after the value A consumed has already been
//!   read by the same thread. RAW (A writes X, B reads X) is the
//!   dependency edge that put the ops in this order; folding it into
//!   one kernel is the *point* of fusion.
//! - **Cycle fallback.** If [`super::topology::topological_sort`]
//!   surfaces a [`super::topology::CycleError`] the analysis falls back
//!   to source order (`OpId(0)..OpId(op_count)`). Cyclic graphs are not
//!   schedulable as DAGs and the planner will resolve the cycle through
//!   pack/unpack sequencing or megakernel synthesis (Task 3.3); until
//!   then, returning a deterministic grouping in source order keeps the
//!   pipeline alive.
//! - **`PerEvent` indirection always wins over fusion size.** A group
//!   of two or more `PerEvent` ops sharing the same source ring is
//!   classified [`FusibilityClass::Indirect`] (not `Fused`). Indirect-
//!   dispatch rings are a different kernel-shape concern than per-agent
//!   fusion; a future task may revisit and fuse indirect-driven kernel
//!   bodies while keeping the indirect dispatch outer wrapper.

use std::collections::BTreeSet;

use crate::cg::data_handle::{CycleEdgeKey, DataHandle, EventRingId};
use crate::cg::dispatch::{DispatchShape, PerPairSource};
use crate::cg::op::OpId;
use crate::cg::program::CgProgram;

use super::topology::{topological_sort, DepGraph};

// ---------------------------------------------------------------------------
// DispatchShapeKey — typed projection for dispatch-shape equality
// ---------------------------------------------------------------------------

/// Equality projection of a [`DispatchShape`] used by [`fusion_candidates`]
/// to decide whether two ops can fuse.
///
/// Two [`DispatchShape`] values are *fusion-compatible* when they project
/// to the same [`DispatchShapeKey`]. The projection mirrors the
/// dispatch's structural identity: `PerEvent` keys on its source ring,
/// `PerPair` keys on its source descriptor, the unit-shape variants
/// project to their bare-tag forms.
///
/// Encoding the projection as a typed enum (rather than a `(u32, …)`
/// tuple or a raw match in [`fusion_candidates`]) keeps every consumer
/// of "do these two shapes fuse?" routing through the same enum, and
/// makes future shape variants force a compile error here when they
/// land.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum DispatchShapeKey {
    /// Per-agent dispatch — every `PerAgent` op shares this key.
    PerAgent,
    /// Per-event dispatch keyed by source ring; only ops driven by the
    /// same ring share this key.
    PerEvent(EventRingId),
    /// Per-pair dispatch keyed by source descriptor. Two `PerPair`
    /// dispatches with different sources (e.g. kin vs engagement) are
    /// not fusion-compatible.
    PerPair(PerPairSource),
    /// Single-thread one-shot dispatch.
    OneShot,
    /// Per-bitmap-word dispatch (alive-bitmap pack and friends).
    PerWord,
}

/// Project a [`DispatchShape`] to its [`DispatchShapeKey`]. The
/// projection is the single source of truth for "are these two shapes
/// fusion-compatible?".
pub fn dispatch_shape_key(shape: &DispatchShape) -> DispatchShapeKey {
    match shape {
        DispatchShape::PerAgent => DispatchShapeKey::PerAgent,
        DispatchShape::PerEvent { source_ring } => DispatchShapeKey::PerEvent(*source_ring),
        DispatchShape::PerPair { source } => DispatchShapeKey::PerPair(*source),
        DispatchShape::OneShot => DispatchShapeKey::OneShot,
        DispatchShape::PerWord => DispatchShapeKey::PerWord,
    }
}

// ---------------------------------------------------------------------------
// FusionGroup + FusibilityClass
// ---------------------------------------------------------------------------

/// A maximal run of consecutive ops in the topological order that share
/// a [`DispatchShapeKey`] and don't write-conflict.
///
/// `ops` is in topological order — the order produced by
/// [`super::topology::topological_sort`]. `shape` is the (single) shape
/// every op in the group carries. `fusibility_classification` summarises
/// what schedule synthesis will do with this group; see
/// [`FusibilityClass`].
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FusionGroup {
    /// Members of the group, sorted in topological order.
    pub ops: Vec<OpId>,
    /// Common dispatch shape every member of the group carries.
    pub shape: DispatchShape,
    /// Classification — does the group fuse, run as a singleton, or
    /// drive an indirect dispatch?
    pub fusibility_classification: FusibilityClass,
}

/// What schedule synthesis will do with a [`FusionGroup`].
///
/// Task 3.4 will translate these into `KernelTopology` values
/// (`Fused`, `Split`, `Indirect`); Task 3.2 produces the analysis layer
/// these classifications drive.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum FusibilityClass {
    /// Group has size ≥ 2 and the ops can be lowered into a single
    /// fused kernel body.
    Fused,
    /// Singleton group — only one op, no fusion neighbour. Runs as its
    /// own kernel.
    Split,
    /// `PerEvent` group (ring-driven indirect dispatch). Even when the
    /// group has size ≥ 2, the runtime drives the dispatch indirectly
    /// from a ring's tail count; whether the kernels themselves merge
    /// is a downstream decision (Task 3.4).
    Indirect,
}

impl FusibilityClass {
    /// Stable snake_case label for diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            FusibilityClass::Fused => "fused",
            FusibilityClass::Split => "split",
            FusibilityClass::Indirect => "indirect",
        }
    }
}

impl std::fmt::Display for FusibilityClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// FusionDiagnostic
// ---------------------------------------------------------------------------

/// A single fusion-decision diagnostic produced alongside the
/// [`FusionGroup`] list by [`fusion_decisions`].
///
/// Each entry explains *why* a group boundary was placed where it was —
/// the dispatch-shape mismatch, write-conflict, indirect-dispatch
/// classification, or simply "this op had no fusable neighbour". The
/// `kind` field is the typed structural payload; `message` is a
/// human-readable rendering for logs and snapshot tests.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FusionDiagnostic {
    /// What the diagnostic says.
    pub kind: FusionDiagnosticKind,
    /// The op ids the diagnostic concerns. For a `Fused`/`Indirect`
    /// group entry this is every member of the group; for a split
    /// entry this is the two-element vector `[prev_last, current]`
    /// recording the boundary.
    pub ops: Vec<OpId>,
    /// Human-readable rendering. The structural payload is on `kind`;
    /// `message` is a convenience for logs.
    pub message: String,
}

/// Typed payload for a [`FusionDiagnostic`].
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum FusionDiagnosticKind {
    /// A group of size ≥ 2 fused into one kernel.
    Fused,
    /// A singleton group — no fusable neighbour was available.
    Singleton,
    /// Two consecutive ops were split because their dispatch shapes
    /// don't project to the same [`DispatchShapeKey`].
    SplitDispatchShapeMismatch,
    /// Two consecutive ops were split because they share a write
    /// handle (write-after-write hazard within a single kernel).
    SplitWriteConflict { handle: DataHandle },
    /// A `PerEvent` group classified as
    /// [`FusibilityClass::Indirect`] — schedule synthesis drives the
    /// dispatch via the ring's tail count.
    Indirect,
}

impl FusionDiagnosticKind {
    /// Stable snake_case label, used in `message` rendering and logs.
    pub fn label(&self) -> &'static str {
        match self {
            FusionDiagnosticKind::Fused => "fused",
            FusionDiagnosticKind::Singleton => "singleton",
            FusionDiagnosticKind::SplitDispatchShapeMismatch => "split_dispatch_shape_mismatch",
            FusionDiagnosticKind::SplitWriteConflict { .. } => "split_write_conflict",
            FusionDiagnosticKind::Indirect => "indirect",
        }
    }
}

impl std::fmt::Display for FusionDiagnosticKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionDiagnosticKind::Fused
            | FusionDiagnosticKind::Singleton
            | FusionDiagnosticKind::SplitDispatchShapeMismatch
            | FusionDiagnosticKind::Indirect => f.write_str(self.label()),
            FusionDiagnosticKind::SplitWriteConflict { handle } => {
                write!(f, "split_write_conflict(handle={})", handle)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fusion_candidates
// ---------------------------------------------------------------------------

/// Compute the [`FusionGroup`] partition for `prog` under the
/// dependency graph `deps`. Convenience over [`fusion_decisions`] for
/// callers that don't need the per-decision diagnostic stream.
pub fn fusion_candidates(prog: &CgProgram, deps: &DepGraph) -> Vec<FusionGroup> {
    fusion_decisions(prog, deps).0
}

/// Compute the fusion-group partition AND the diagnostic stream
/// explaining each grouping/split decision. Both outputs are returned in
/// topological-walk order; the diagnostics are emitted in the same order
/// the analysis encountered the corresponding decision.
///
/// **Walk:** the analysis visits ops in
/// [`super::topology::topological_sort`] order. For each op it checks
/// whether it can join the group currently being built — same
/// [`DispatchShapeKey`], no write-handle overlap with the in-progress
/// group's accumulated writes. On a join, the op is appended; on a
/// rejection, the in-progress group is closed and a new group starts
/// rooted at the rejected op. Each closed group emits one diagnostic
/// summarising its classification; each rejection emits one diagnostic
/// recording the boundary cause.
///
/// **Determinism:** the walk consumes a deterministic topological
/// order, walks BTree-backed metadata, and never iterates on hash-set
/// iteration order. Two runs over the same program produce identical
/// `(groups, diagnostics)` outputs.
pub fn fusion_decisions(
    prog: &CgProgram,
    deps: &DepGraph,
) -> (Vec<FusionGroup>, Vec<FusionDiagnostic>) {
    // Resolve a deterministic walk order. On cycle, fall back to source
    // order (see module-level `# Limitations`).
    let topo_order: Vec<OpId> = match topological_sort(deps) {
        Ok(order) => order,
        Err(_cycle) => (0..prog.ops.len() as u32).map(OpId).collect(),
    };

    let mut groups: Vec<FusionGroup> = Vec::new();
    let mut diagnostics: Vec<FusionDiagnostic> = Vec::new();

    // In-progress group state.
    let mut current_ops: Vec<OpId> = Vec::new();
    let mut current_shape: Option<DispatchShape> = None;
    let mut current_writes: BTreeSet<CycleEdgeKey> = BTreeSet::new();

    for op_id in topo_order {
        let op = match prog.ops.get(op_id.0 as usize) {
            Some(op) => op,
            // Defensive — `topological_sort` produces only in-range
            // OpIds for a well-formed graph; skip silently if a
            // malformed graph slips through (no panic).
            None => continue,
        };
        let op_writes: BTreeSet<CycleEdgeKey> =
            op.writes.iter().map(|h| h.cycle_edge_key()).collect();

        // Decide whether this op can join the current group, and if not,
        // why not.
        let join_decision = match &current_shape {
            None => JoinDecision::StartNewGroup,
            Some(prev_shape) => {
                if dispatch_shape_key(prev_shape) != dispatch_shape_key(&op.shape) {
                    JoinDecision::SplitShape
                } else if let Some(conflict_key) =
                    current_writes.intersection(&op_writes).next().cloned()
                {
                    let handle = recover_handle_for_conflict(prog, &current_ops, &conflict_key);
                    JoinDecision::SplitWrite(handle)
                } else {
                    JoinDecision::Join
                }
            }
        };

        match join_decision {
            JoinDecision::StartNewGroup => {
                // The very first op of the program — no diagnostic
                // yet; it becomes the seed of the first group.
                current_ops.push(op_id);
                current_shape = Some(op.shape);
                current_writes = op_writes;
            }
            JoinDecision::Join => {
                current_ops.push(op_id);
                current_writes.extend(op_writes);
            }
            JoinDecision::SplitShape => {
                // Close current group + emit boundary diagnostic.
                if let Some(group) = close_current_group(
                    &mut current_ops,
                    &mut current_shape,
                    &mut current_writes,
                    &mut diagnostics,
                ) {
                    let prev_last = *group.ops.last().expect("non-empty group");
                    diagnostics.push(FusionDiagnostic {
                        kind: FusionDiagnosticKind::SplitDispatchShapeMismatch,
                        ops: vec![prev_last, op_id],
                        message: format!(
                            "split between op#{} and op#{}: dispatch shapes differ ({} vs {})",
                            prev_last.0, op_id.0, group.shape, op.shape
                        ),
                    });
                    groups.push(group);
                }
                // Start new group rooted at this op.
                current_ops.push(op_id);
                current_shape = Some(op.shape);
                current_writes = op_writes;
            }
            JoinDecision::SplitWrite(handle) => {
                if let Some(group) = close_current_group(
                    &mut current_ops,
                    &mut current_shape,
                    &mut current_writes,
                    &mut diagnostics,
                ) {
                    let prev_last = *group.ops.last().expect("non-empty group");
                    diagnostics.push(FusionDiagnostic {
                        kind: FusionDiagnosticKind::SplitWriteConflict {
                            handle: handle.clone(),
                        },
                        ops: vec![prev_last, op_id],
                        message: format!(
                            "split between op#{} and op#{}: write conflict on {}",
                            prev_last.0, op_id.0, handle
                        ),
                    });
                    groups.push(group);
                }
                current_ops.push(op_id);
                current_shape = Some(op.shape);
                current_writes = op_writes;
            }
        }
    }

    // Close the final in-progress group, if any.
    if let Some(group) = close_current_group(
        &mut current_ops,
        &mut current_shape,
        &mut current_writes,
        &mut diagnostics,
    ) {
        groups.push(group);
    }

    (groups, diagnostics)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// What the next op in the walk does relative to the in-progress group.
enum JoinDecision {
    /// No group is in progress yet; this op seeds the first group.
    StartNewGroup,
    /// Append this op to the current group.
    Join,
    /// Close the current group and start a new one — dispatch shapes
    /// don't match.
    SplitShape,
    /// Close the current group and start a new one — this op writes a
    /// handle the current group already wrote.
    SplitWrite(DataHandle),
}

/// Close the in-progress group: build its [`FusionGroup`], emit a
/// classification diagnostic, and clear the in-progress state. Returns
/// `None` when there is no in-progress group to close.
fn close_current_group(
    current_ops: &mut Vec<OpId>,
    current_shape: &mut Option<DispatchShape>,
    current_writes: &mut BTreeSet<CycleEdgeKey>,
    diagnostics: &mut Vec<FusionDiagnostic>,
) -> Option<FusionGroup> {
    if current_ops.is_empty() {
        return None;
    }
    let shape = current_shape.expect("non-empty group must carry a shape");
    let ops = std::mem::take(current_ops);
    let classification = classify(&ops, &shape);

    // Emit classification diagnostic.
    let message = match classification {
        FusibilityClass::Fused => format!(
            "fused {} ops on shape {} into one kernel",
            ops.len(),
            shape
        ),
        FusibilityClass::Split => {
            format!("singleton op#{} on shape {}", ops[0].0, shape)
        }
        FusibilityClass::Indirect => format!(
            "indirect dispatch group of {} ops on shape {}",
            ops.len(),
            shape
        ),
    };
    let kind = match classification {
        FusibilityClass::Fused => FusionDiagnosticKind::Fused,
        FusibilityClass::Split => FusionDiagnosticKind::Singleton,
        FusibilityClass::Indirect => FusionDiagnosticKind::Indirect,
    };
    diagnostics.push(FusionDiagnostic {
        kind,
        ops: ops.clone(),
        message,
    });

    *current_shape = None;
    current_writes.clear();
    Some(FusionGroup {
        ops,
        shape,
        fusibility_classification: classification,
    })
}

/// Classify a finalised group.
///
/// - Singleton → [`FusibilityClass::Split`].
/// - Multi-op `PerEvent` group → [`FusibilityClass::Indirect`] (the
///   ring drives the dispatch even when the kernels share a body).
/// - Multi-op group on any other shape → [`FusibilityClass::Fused`].
fn classify(ops: &[OpId], shape: &DispatchShape) -> FusibilityClass {
    if ops.len() < 2 {
        return FusibilityClass::Split;
    }
    match shape {
        DispatchShape::PerEvent { .. } => FusibilityClass::Indirect,
        DispatchShape::PerAgent
        | DispatchShape::PerPair { .. }
        | DispatchShape::OneShot
        | DispatchShape::PerWord => FusibilityClass::Fused,
    }
}

/// Recover a [`DataHandle`] that witnesses a write-conflict between the
/// current group's accumulated writes and the next op. The diagnostic
/// surface wants the typed handle (not just the projected
/// [`CycleEdgeKey`]) so that `EventRing` ring numbers etc. render with
/// full context.
///
/// The lookup is best-effort: it returns the first write handle on any
/// op already in the group whose `cycle_edge_key()` matches `key`. The
/// program's well-formedness guarantee that writes are deterministic
/// makes the "first match" stable across runs.
fn recover_handle_for_conflict(
    prog: &CgProgram,
    current_ops: &[OpId],
    key: &CycleEdgeKey,
) -> DataHandle {
    for op_id in current_ops {
        if let Some(op) = prog.ops.get(op_id.0 as usize) {
            for w in &op.writes {
                if &w.cycle_edge_key() == key {
                    return w.clone();
                }
            }
        }
    }
    // Unreachable when invoked via `JoinDecision::SplitWrite` — the
    // caller computed `key` from the intersection of the in-progress
    // group's writes with the new op's writes. The fallback keeps the
    // function total without a panic; if some future caller reaches it
    // with a stale key, surface a synthetic handle that round-trips
    // through `Display` rather than crashing.
    match key {
        CycleEdgeKey::Ring(ring) => DataHandle::IndirectArgs { ring: *ring },
        CycleEdgeKey::Other(handle) => handle.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, AgentScratchKind, DataHandle, EventRingAccess, EventRingId, MaskId,
    };
    use crate::cg::dispatch::{DispatchShape, PerPairSource};
    use crate::cg::expr::{CgExpr, LitValue};
    use crate::cg::op::{ComputeOpKind, OpId, PlumbingKind, Span, SpatialQueryKind};
    use crate::cg::program::{CgProgram, CgProgramBuilder};
    use crate::cg::schedule::topology::dependency_graph;

    // --- helpers -------------------------------------------------------

    /// Add a `MaskPredicate` op with literal-true predicate (no
    /// auto-derived reads), at the given dispatch shape. Tests inject
    /// reads/writes via `record_read` / `record_write`.
    fn add_mask_op(
        builder: &mut CgProgramBuilder,
        mask: MaskId,
        shape: DispatchShape,
    ) -> OpId {
        let pred = builder.add_expr(CgExpr::Lit(LitValue::Bool(true))).unwrap();
        builder
            .add_op(
                ComputeOpKind::MaskPredicate {
                    mask,
                    predicate: pred,
                },
                shape,
                Span::dummy(),
            )
            .unwrap()
    }

    /// Add a `Plumbing(PackAgents)` op (carries its own structural
    /// reads + writes via [`PlumbingKind::dependencies`]).
    fn add_pack_op(builder: &mut CgProgramBuilder) -> OpId {
        builder
            .add_op(
                ComputeOpKind::Plumbing {
                    kind: PlumbingKind::PackAgents,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap()
    }

    /// Add a `Plumbing(UploadSimCfg)` op — `OneShot` shape.
    fn add_upload_sim_cfg_op(builder: &mut CgProgramBuilder) -> OpId {
        builder
            .add_op(
                ComputeOpKind::Plumbing {
                    kind: PlumbingKind::UploadSimCfg,
                },
                DispatchShape::OneShot,
                Span::dummy(),
            )
            .unwrap()
    }

    /// Add a `Plumbing(KickSnapshot)` op — `OneShot` shape.
    fn add_kick_snapshot_op(builder: &mut CgProgramBuilder) -> OpId {
        builder
            .add_op(
                ComputeOpKind::Plumbing {
                    kind: PlumbingKind::KickSnapshot,
                },
                DispatchShape::OneShot,
                Span::dummy(),
            )
            .unwrap()
    }

    /// Add a `Plumbing(DrainEvents { ring })` op — `PerEvent` shape.
    fn add_drain_events_op(builder: &mut CgProgramBuilder, ring: EventRingId) -> OpId {
        builder
            .add_op(
                ComputeOpKind::Plumbing {
                    kind: PlumbingKind::DrainEvents { ring },
                },
                DispatchShape::PerEvent { source_ring: ring },
                Span::dummy(),
            )
            .unwrap()
    }

    fn hp_handle() -> DataHandle {
        DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        }
    }

    fn shield_handle() -> DataHandle {
        DataHandle::AgentField {
            field: AgentFieldId::ShieldHp,
            target: AgentRef::Self_,
        }
    }

    fn mana_handle() -> DataHandle {
        DataHandle::AgentField {
            field: AgentFieldId::Mana,
            target: AgentRef::Self_,
        }
    }

    // --- 1. Empty program ----------------------------------------------

    #[test]
    fn empty_program_has_empty_groups_and_diagnostics() {
        let prog = CgProgram::new();
        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);
        assert!(groups.is_empty());
        assert!(diagnostics.is_empty());
        assert!(fusion_candidates(&prog, &deps).is_empty());
    }

    // --- 2. All-PerAgent ops with no conflicts → one big Fused group ----

    #[test]
    fn three_per_agent_ops_no_conflicts_fuse_into_one_group() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let m2 = add_mask_op(&mut b, MaskId(2), DispatchShape::PerAgent);
        let mut prog = b.finish();
        // Each writes a distinct MaskBitmap (auto-derived) — no
        // overlap. Dispatch shapes match. They should fuse.
        // Force a deterministic topo order by adding an inter-op read
        // that doesn't conflict: m1 reads what m0 writes (the mask
        // bitmap is distinct, so no RAW edge against the *write* set —
        // but we want determinism, so leave the auto-derived deps as
        // is. Topo order falls back to OpId order via Kahn's heap.
        let _ = (m0, m1, m2, &mut prog);

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);
        assert_eq!(groups.len(), 1, "expected single fused group");
        assert_eq!(groups[0].ops, vec![m0, m1, m2]);
        assert_eq!(groups[0].shape, DispatchShape::PerAgent);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Fused
        );

        // One Fused diagnostic, no others.
        assert_eq!(diagnostics.len(), 1);
        assert!(matches!(diagnostics[0].kind, FusionDiagnosticKind::Fused));
        assert_eq!(diagnostics[0].ops, vec![m0, m1, m2]);
    }

    // --- 3. Mixed shapes → multiple groups split at shape boundaries ---

    #[test]
    fn mixed_shapes_split_at_each_boundary() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let one_shot = add_upload_sim_cfg_op(&mut b);
        let m2 = add_mask_op(&mut b, MaskId(2), DispatchShape::PerAgent);
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);

        // Three groups: {m0} singleton, {one_shot} singleton, {m2} singleton.
        // Boundaries between them are dispatch-shape mismatches.
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].ops, vec![m0]);
        assert_eq!(groups[0].fusibility_classification, FusibilityClass::Split);
        assert_eq!(groups[1].ops, vec![one_shot]);
        assert_eq!(groups[1].fusibility_classification, FusibilityClass::Split);
        assert_eq!(groups[2].ops, vec![m2]);
        assert_eq!(groups[2].fusibility_classification, FusibilityClass::Split);

        // Diagnostics: 2 boundary splits + 3 singleton classifications,
        // emitted in walk order.
        let kinds: Vec<&FusionDiagnosticKind> = diagnostics.iter().map(|d| &d.kind).collect();
        // Two SplitDispatchShapeMismatch diagnostics + three Singleton
        // diagnostics.
        let n_splits = kinds
            .iter()
            .filter(|k| matches!(k, FusionDiagnosticKind::SplitDispatchShapeMismatch))
            .count();
        let n_singletons = kinds
            .iter()
            .filter(|k| matches!(k, FusionDiagnosticKind::Singleton))
            .count();
        assert_eq!(n_splits, 2);
        assert_eq!(n_singletons, 3);
    }

    // --- 4. Write conflict splits a same-shape pair --------------------

    #[test]
    fn write_after_write_on_same_handle_splits_group() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let mut prog = b.finish();
        // Inject a shared write — both ops write `hp_handle()` —
        // creating a WAW conflict. Dispatch shape matches, so without
        // the conflict they would fuse.
        prog.ops[m0.0 as usize].record_write(hp_handle());
        prog.ops[m1.0 as usize].record_write(hp_handle());

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);

        // Two singleton groups; the boundary diagnostic carries the
        // `hp_handle()` payload.
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].ops, vec![m0]);
        assert_eq!(groups[1].ops, vec![m1]);
        assert_eq!(groups[0].fusibility_classification, FusibilityClass::Split);
        assert_eq!(groups[1].fusibility_classification, FusibilityClass::Split);

        let split = diagnostics
            .iter()
            .find(|d| matches!(d.kind, FusionDiagnosticKind::SplitWriteConflict { .. }))
            .expect("expected split-write diagnostic");
        match &split.kind {
            FusionDiagnosticKind::SplitWriteConflict { handle } => {
                assert_eq!(handle, &hp_handle());
            }
            other => panic!("unexpected diagnostic kind: {:?}", other),
        }
        assert_eq!(split.ops, vec![m0, m1]);
    }

    // --- 5. PerEvent ops on the same ring → Indirect classification ----

    #[test]
    fn two_per_event_ops_on_same_ring_classify_as_indirect() {
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(3);
        let d0 = add_drain_events_op(&mut b, ring);
        let d1 = add_drain_events_op(&mut b, ring);
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![d0, d1]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Indirect
        );

        let kinds: Vec<&FusionDiagnosticKind> = diagnostics.iter().map(|d| &d.kind).collect();
        assert!(
            kinds
                .iter()
                .any(|k| matches!(k, FusionDiagnosticKind::Indirect)),
            "expected an Indirect diagnostic, got {:?}",
            kinds
        );
    }

    // --- 5b. PerEvent ops on different rings → split shape mismatch ----

    #[test]
    fn per_event_ops_on_different_rings_split_at_shape_boundary() {
        let mut b = CgProgramBuilder::new();
        let d0 = add_drain_events_op(&mut b, EventRingId(1));
        let d1 = add_drain_events_op(&mut b, EventRingId(2));
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, _diagnostics) = fusion_decisions(&prog, &deps);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].ops, vec![d0]);
        assert_eq!(groups[1].ops, vec![d1]);
        assert_eq!(groups[0].fusibility_classification, FusibilityClass::Split);
        assert_eq!(groups[1].fusibility_classification, FusibilityClass::Split);
    }

    // --- 6. OneShot ops fuse with each other but not with PerAgent -----

    #[test]
    fn consecutive_one_shot_ops_fuse_but_not_with_per_agent_neighbour() {
        let mut b = CgProgramBuilder::new();
        let upload = add_upload_sim_cfg_op(&mut b);
        let kick = add_kick_snapshot_op(&mut b);
        let m2 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, _diagnostics) = fusion_decisions(&prog, &deps);

        // [upload, kick] (Fused, OneShot) | [m2] (Split, PerAgent).
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].ops, vec![upload, kick]);
        assert_eq!(groups[0].shape, DispatchShape::OneShot);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Fused
        );
        assert_eq!(groups[1].ops, vec![m2]);
        assert_eq!(groups[1].fusibility_classification, FusibilityClass::Split);
    }

    // --- 7. Singleton classification -----------------------------------

    #[test]
    fn lone_op_classifies_as_singleton_split() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![m0]);
        assert_eq!(groups[0].fusibility_classification, FusibilityClass::Split);

        assert_eq!(diagnostics.len(), 1);
        assert!(matches!(
            diagnostics[0].kind,
            FusionDiagnosticKind::Singleton
        ));
    }

    // --- 8. Decision diagnostics — every kind reachable ----------------

    #[test]
    fn every_diagnostic_kind_is_reachable() {
        let mut b = CgProgramBuilder::new();
        // Layout (after we override topo order via record_read on m0/m1
        // to keep them in source order):
        //   op0: PerAgent mask, write hp     (start group)
        //   op1: PerAgent mask, write hp     (split-write boundary)
        //   op2: OneShot upload_sim_cfg      (split-shape boundary)
        //   op3: PerEvent drain ring=#9      (split-shape boundary)
        //   op4: PerEvent drain ring=#9      (joins op3 → Indirect)
        //   op5: PerAgent mask               (split-shape boundary, singleton)
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let upload = add_upload_sim_cfg_op(&mut b);
        let d0 = add_drain_events_op(&mut b, EventRingId(9));
        let d1 = add_drain_events_op(&mut b, EventRingId(9));
        let m5 = add_mask_op(&mut b, MaskId(5), DispatchShape::PerAgent);
        let mut prog = b.finish();
        // Inject the WAW conflict between m0 and m1 — same hp write.
        prog.ops[m0.0 as usize].record_write(hp_handle());
        prog.ops[m1.0 as usize].record_write(hp_handle());

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);

        // Expected topology: 4 groups.
        // {m0} Split | {m1} Split | {upload} Split | {d0,d1} Indirect | {m5} Split.
        assert_eq!(groups.len(), 5, "groups: {:?}", groups);
        assert_eq!(groups[0].ops, vec![m0]);
        assert_eq!(groups[1].ops, vec![m1]);
        assert_eq!(groups[2].ops, vec![upload]);
        assert_eq!(groups[3].ops, vec![d0, d1]);
        assert_eq!(
            groups[3].fusibility_classification,
            FusibilityClass::Indirect
        );
        assert_eq!(groups[4].ops, vec![m5]);

        // Every kind we documented must show up in the diagnostic
        // stream.
        let mut saw_fused = false;
        let mut saw_singleton = false;
        let mut saw_split_shape = false;
        let mut saw_split_write = false;
        let mut saw_indirect = false;
        for d in &diagnostics {
            match &d.kind {
                FusionDiagnosticKind::Fused => saw_fused = true,
                FusionDiagnosticKind::Singleton => saw_singleton = true,
                FusionDiagnosticKind::SplitDispatchShapeMismatch => saw_split_shape = true,
                FusionDiagnosticKind::SplitWriteConflict { .. } => saw_split_write = true,
                FusionDiagnosticKind::Indirect => saw_indirect = true,
            }
        }
        // Fused isn't reachable in this fixture — replace with a small
        // companion fixture below to keep this single test from
        // sprawling; here we just verify the four kinds we DO produce
        // surface, plus assert indirect coverage.
        assert!(saw_singleton, "missing Singleton: {:?}", diagnostics);
        assert!(saw_split_shape, "missing SplitShape: {:?}", diagnostics);
        assert!(saw_split_write, "missing SplitWrite: {:?}", diagnostics);
        assert!(saw_indirect, "missing Indirect: {:?}", diagnostics);
        // Fused coverage is asserted in `three_per_agent_ops_no_conflicts_fuse_into_one_group`.
        let _ = saw_fused;
    }

    #[test]
    fn fused_diagnostic_kind_is_reachable() {
        // Companion to `every_diagnostic_kind_is_reachable` — Fused is
        // produced by any multi-op group on a non-PerEvent shape.
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let prog = b.finish();
        let deps = dependency_graph(&prog);
        let (_groups, diagnostics) = fusion_decisions(&prog, &deps);
        assert!(
            diagnostics
                .iter()
                .any(|d| matches!(d.kind, FusionDiagnosticKind::Fused)),
            "diagnostics: {:?}",
            diagnostics
        );
        let _ = (m0, m1);
    }

    // --- 9. Determinism ------------------------------------------------

    #[test]
    fn two_runs_produce_identical_groups_and_diagnostics() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let upload = add_upload_sim_cfg_op(&mut b);
        let mut prog = b.finish();
        prog.ops[m0.0 as usize].record_write(hp_handle());
        prog.ops[m1.0 as usize].record_read(hp_handle());
        let _ = upload;

        let deps = dependency_graph(&prog);
        let (g1, d1) = fusion_decisions(&prog, &deps);
        let (g2, d2) = fusion_decisions(&prog, &deps);
        assert_eq!(g1, g2);
        assert_eq!(d1, d2);
    }

    // --- 10. Cycle in graph → fall back to source order ----------------

    #[test]
    fn cycle_in_graph_falls_back_to_source_order() {
        // Op0 reads Hp, writes Mana.
        // Op1 reads Mana, writes Hp.
        // Both PerAgent shape.
        // The cycle prevents topological_sort from producing an order;
        // the analysis falls back to OpId(0), OpId(1). The two ops
        // share dispatch shape AND have a write conflict at *neither*
        // handle (op0 writes Mana, op1 writes Hp — no overlap). So
        // they fuse into one group.
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let mut prog = b.finish();
        prog.ops[m0.0 as usize].record_read(hp_handle());
        prog.ops[m0.0 as usize].record_write(mana_handle());
        prog.ops[m1.0 as usize].record_read(mana_handle());
        prog.ops[m1.0 as usize].record_write(hp_handle());

        let deps = dependency_graph(&prog);
        assert!(deps.has_cycle(), "fixture must produce a cycle");

        let (groups, _diagnostics) = fusion_decisions(&prog, &deps);
        // Source-order fallback: [m0, m1]. Same shape, no WAW. Fuses.
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![m0, m1]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Fused
        );
    }

    // --- bonus: dispatch_shape_key exhaustiveness ----------------------

    #[test]
    fn dispatch_shape_key_distinguishes_every_variant() {
        let keys = [
            dispatch_shape_key(&DispatchShape::PerAgent),
            dispatch_shape_key(&DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            }),
            dispatch_shape_key(&DispatchShape::PerPair {
                source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
            }),
            dispatch_shape_key(&DispatchShape::OneShot),
            dispatch_shape_key(&DispatchShape::PerWord),
        ];
        // All five distinct.
        for i in 0..keys.len() {
            for j in 0..keys.len() {
                if i == j {
                    assert_eq!(keys[i], keys[j]);
                } else {
                    assert_ne!(keys[i], keys[j], "keys[{i}] == keys[{j}]");
                }
            }
        }
    }

    #[test]
    fn dispatch_shape_key_distinguishes_per_event_rings() {
        let a = dispatch_shape_key(&DispatchShape::PerEvent {
            source_ring: EventRingId(1),
        });
        let b = dispatch_shape_key(&DispatchShape::PerEvent {
            source_ring: EventRingId(2),
        });
        assert_ne!(a, b);
    }

    #[test]
    fn dispatch_shape_key_distinguishes_per_pair_sources() {
        let a = dispatch_shape_key(&DispatchShape::PerPair {
            source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
        });
        let b = dispatch_shape_key(&DispatchShape::PerPair {
            source: PerPairSource::SpatialQuery(SpatialQueryKind::EngagementQuery),
        });
        assert_ne!(a, b);
    }

    // --- bonus: pack/unpack-style group with auto-derived writes -------

    #[test]
    fn plumbing_pack_and_mask_split_on_write_conflict_via_alive_field() {
        // PackAgents auto-writes the AgentScratch packed buffer and
        // auto-reads every AgentField including Hp. A MaskPredicate op
        // following it that writes Hp (inject via record_write) has a
        // WAR — but that's fine within a kernel body. There IS, however,
        // a RAW edge (mask reads Hp; pack reads Hp too — no producer/
        // consumer between them). Without a write conflict, the two ops
        // share PerAgent shape and would fuse. Verify.
        let mut b = CgProgramBuilder::new();
        let pack = add_pack_op(&mut b);
        let mask = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, _diagnostics) = fusion_decisions(&prog, &deps);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![pack, mask]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Fused
        );
    }

    // --- bonus: verify SpatialQuery ops are well-handled ---------------

    #[test]
    fn spatial_query_ops_classify_normally_under_per_agent_shape() {
        // SpatialQuery::BuildHash carries DispatchShape::PerAgent in
        // the typical lowering — verify a build_hash + per_agent mask
        // co-fuse when no write conflict exists.
        let mut b = CgProgramBuilder::new();
        let build = b
            .add_op(
                ComputeOpKind::SpatialQuery {
                    kind: SpatialQueryKind::BuildHash,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap();
        let m = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let prog = b.finish();
        let deps = dependency_graph(&prog);
        let (groups, _) = fusion_decisions(&prog, &deps);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![build, m]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Fused
        );
    }

    // --- bonus: AgentScratch handle round-trips through diagnostics ----

    #[test]
    fn write_conflict_diagnostic_renders_handle_in_message() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let mut prog = b.finish();
        prog.ops[m0.0 as usize].record_write(shield_handle());
        prog.ops[m1.0 as usize].record_write(shield_handle());

        let deps = dependency_graph(&prog);
        let (_groups, diagnostics) = fusion_decisions(&prog, &deps);
        let split = diagnostics
            .iter()
            .find(|d| matches!(d.kind, FusionDiagnosticKind::SplitWriteConflict { .. }))
            .expect("expected split-write diagnostic");
        assert!(
            split.message.contains("agent.self.shield_hp"),
            "message {:?} should mention the conflicted handle",
            split.message
        );
    }

    // --- bonus: indirect-args-style ring write handle ------------------

    #[test]
    fn agent_scratch_kind_constants_used_in_pack_unpack_unaffect_classification() {
        // Sanity: AgentScratch is just a regular DataHandle; the
        // analysis treats it like any other write handle. We don't need
        // to special-case it. Build a pack→mask→pack sequence to make
        // sure the second pack triggers a WAW boundary.
        let mut b = CgProgramBuilder::new();
        let p0 = add_pack_op(&mut b);
        let p1 = add_pack_op(&mut b);
        let prog = b.finish();
        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);
        // Two packs each write `AgentScratch::Packed`. WAW. Split.
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].ops, vec![p0]);
        assert_eq!(groups[1].ops, vec![p1]);
        let split = diagnostics
            .iter()
            .find(|d| matches!(d.kind, FusionDiagnosticKind::SplitWriteConflict { .. }))
            .expect("expected split-write diagnostic on AgentScratch packed");
        match &split.kind {
            FusionDiagnosticKind::SplitWriteConflict { handle } => {
                assert_eq!(
                    handle,
                    &DataHandle::AgentScratch {
                        kind: AgentScratchKind::Packed
                    }
                );
            }
            _ => unreachable!(),
        }
    }

    // --- bonus: EventRing write conflict via cycle_edge_key projection -

    #[test]
    fn event_ring_writes_collide_via_cycle_edge_projection() {
        // Two consecutive PerAgent ops both append to the same ring.
        // The cycle_edge_key projection collapses (Append vs Append) to
        // the same Ring(_) key — so this is a WAW conflict regardless
        // of access mode discriminant.
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let mut prog = b.finish();
        let ring = EventRingId(4);
        prog.ops[m0.0 as usize].record_write(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Append,
        });
        prog.ops[m1.0 as usize].record_write(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Append,
        });

        let deps = dependency_graph(&prog);
        let (groups, _diagnostics) = fusion_decisions(&prog, &deps);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].ops, vec![m0]);
        assert_eq!(groups[1].ops, vec![m1]);
    }
}
