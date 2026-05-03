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
//! - **`PerEvent` indirection always wins over fusion size.** Any
//!   `PerEvent` group — singleton or multi-op, single ring or otherwise
//!   — is classified [`FusibilityClass::Indirect`] (not `Fused`, not
//!   `Split`). Even a lone `DrainEvents` / `SeedIndirectArgs` op
//!   dispatches indirectly via the ring's tail count, so the indirect
//!   classification is unconditional on group size. Indirect-dispatch
//!   rings are a different kernel-shape concern than per-agent fusion;
//!   a future task may revisit and fuse indirect-driven kernel bodies
//!   while keeping the indirect dispatch outer wrapper.

use std::collections::BTreeSet;

use crate::cg::data_handle::{CycleEdgeKey, DataHandle, EventRingId, ViewId};
use crate::cg::dispatch::{DispatchShape, PerPairSource};
use crate::cg::op::{ComputeOp, ComputeOpKind, OpId};
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
    /// [`super::topology::topological_sort`] returned a cycle error;
    /// fusion analysis fell back to source order. Downstream consumers
    /// should treat the resulting groups with caution — the cycle may
    /// indicate a missing schedule barrier or a programming error in
    /// the IR. Inspect `op.reads` / `op.writes` for the cycling
    /// handles.
    ///
    /// Emitted as the FIRST diagnostic in the stream when fallback
    /// fires, before any per-group classification diagnostic.
    CycleFallback,
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
            FusionDiagnosticKind::CycleFallback => "cycle_fallback",
        }
    }
}

impl std::fmt::Display for FusionDiagnosticKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionDiagnosticKind::Fused
            | FusionDiagnosticKind::Singleton
            | FusionDiagnosticKind::SplitDispatchShapeMismatch
            | FusionDiagnosticKind::Indirect
            | FusionDiagnosticKind::CycleFallback => f.write_str(self.label()),
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
    let mut groups: Vec<FusionGroup> = Vec::new();
    let mut diagnostics: Vec<FusionDiagnostic> = Vec::new();

    // Resolve a deterministic walk order. On cycle, fall back to source
    // order (see module-level `# Limitations`) AND surface a typed
    // `CycleFallback` diagnostic as the first entry in the stream so
    // downstream consumers can detect the degraded analysis.
    let topo_order: Vec<OpId> = match topological_sort(deps) {
        Ok(order) => order,
        Err(_cycle) => {
            diagnostics.push(FusionDiagnostic {
                kind: FusionDiagnosticKind::CycleFallback,
                ops: Vec::new(),
                message: "fusion analysis fell back to source order due to a cycle in \
                          the dependency graph"
                    .to_string(),
            });
            (0..prog.ops.len() as u32).map(OpId).collect()
        }
    };

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
        let join_decision = decide_join(
            prog,
            &current_shape,
            &current_ops,
            &current_writes,
            op,
            &op_writes,
        );

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
    /// handle the current group already wrote, or the two ops fall
    /// into different fusion-ownership domains (cross-view ViewFold,
    /// cross-replayability PhysicsRule). The carried handle is a
    /// representative witness for the conflict — for a true WAW it's
    /// the witnessed shared write; for a cross-domain split it's a
    /// representative write from one of the two domains.
    SplitWrite(DataHandle),
}

/// Decide whether `op` can join the current in-progress group.
///
/// The eligibility rules layered on top of dispatch-shape equality and
/// the WAW-write-conflict check (Task 3.2's original logic):
///
/// 1. **Cross-view ViewFold split** — two `ViewFold` ops on different
///    views must split, even if their write sets are disjoint. They
///    belong to different kernel-ownership domains.
/// 2. **Cross-replayability PhysicsRule split** — two `PhysicsRule`
///    ops with different `replayable` flags must split (replayable
///    rules emit into the deterministic ring; non-replayable into the
///    chronicle ring; fusing them across the boundary would mix
///    determinism domains).
/// 3. **Same-view ViewFold accumulator** — two `ViewFold` ops on the
///    same view that WAW-conflict on that view's storage are a
///    co-accumulator pattern, NOT a fusion blocker. Both handlers
///    contribute to the same view storage; fusion is the *desired*
///    outcome.
fn decide_join(
    prog: &CgProgram,
    current_shape: &Option<DispatchShape>,
    current_ops: &[OpId],
    current_writes: &BTreeSet<CycleEdgeKey>,
    op: &ComputeOp,
    op_writes: &BTreeSet<CycleEdgeKey>,
) -> JoinDecision {
    let prev_shape = match current_shape {
        Some(s) => s,
        None => return JoinDecision::StartNewGroup,
    };
    if dispatch_shape_key(prev_shape) != dispatch_shape_key(&op.shape) {
        return JoinDecision::SplitShape;
    }
    let prev_last_id = match current_ops.last().copied() {
        Some(id) => id,
        // current_shape is Some but current_ops is empty —
        // unreachable in practice (the two are kept in lockstep);
        // returning StartNewGroup keeps the branch total without
        // a panic if a future refactor decouples them.
        None => return JoinDecision::StartNewGroup,
    };
    let prev_op = match prog.ops.get(prev_last_id.0 as usize) {
        Some(p) => p,
        None => {
            // Defensive: malformed graph slipped past
            // `topological_sort`. Fall back to the conventional WAW
            // check; cross-domain rules can't be evaluated without
            // the prev op's kind.
            return waw_check(prog, current_ops, current_writes, op_writes);
        }
    };
    // Cross-domain split (rules 1 + 2) — fires regardless of write
    // disjointness.
    if let Some(decision) = cross_domain_split_decision(prev_op, op) {
        return decision;
    }
    // Same-view ViewFold accumulator override (rule 3) and the
    // conventional WAW-conflict check.
    if let Some(conflict_key) = current_writes.intersection(op_writes).next().cloned() {
        if same_view_view_fold_accumulator(prev_op, op, &conflict_key) {
            JoinDecision::Join
        } else {
            let handle = recover_handle_for_conflict(prog, current_ops, &conflict_key);
            JoinDecision::SplitWrite(handle)
        }
    } else {
        JoinDecision::Join
    }
}

/// Conventional WAW check — used as the defensive fallback inside
/// [`decide_join`] when the prev op pointer can't be resolved.
fn waw_check(
    prog: &CgProgram,
    current_ops: &[OpId],
    current_writes: &BTreeSet<CycleEdgeKey>,
    op_writes: &BTreeSet<CycleEdgeKey>,
) -> JoinDecision {
    if let Some(conflict_key) = current_writes.intersection(op_writes).next().cloned() {
        let handle = recover_handle_for_conflict(prog, current_ops, &conflict_key);
        JoinDecision::SplitWrite(handle)
    } else {
        JoinDecision::Join
    }
}

/// If `prev` and `next` belong to incompatible fusion-ownership
/// domains, return the corresponding split decision; otherwise return
/// `None` (the conventional WAW check applies). The match is
/// exhaustive over [`ComputeOpKind`] — every variant pair is
/// considered explicitly.
fn cross_domain_split_decision(
    prev: &ComputeOp,
    next: &ComputeOp,
) -> Option<JoinDecision> {
    match (&prev.kind, &next.kind) {
        // Cross-view ViewFold split — different views are different
        // kernel-ownership domains, even if writes don't intersect.
        (
            ComputeOpKind::ViewFold {
                view: v1,
                on_event: _,
                body: _,
            },
            ComputeOpKind::ViewFold {
                view: v2,
                on_event: _,
                body: _,
            },
        ) if v1 != v2 => Some(JoinDecision::SplitWrite(
            cross_view_split_witness(prev, next, *v2),
        )),
        // Same-view ViewFolds: the WAW check (with the accumulator
        // override) handles them.
        (
            ComputeOpKind::ViewFold { .. },
            ComputeOpKind::ViewFold { .. },
        ) => None,
        // Cross-replayability PhysicsRule split — replayable and
        // non-replayable rules emit into different rings and live
        // in different determinism domains. Even if the bodies don't
        // share writes, fusing across the boundary would mix
        // determinism contracts.
        (
            ComputeOpKind::PhysicsRule {
                rule: _,
                on_event: _,
                body: _,
                replayable: r1,
            },
            ComputeOpKind::PhysicsRule {
                rule: _,
                on_event: _,
                body: _,
                replayable: r2,
            },
        ) if r1 != r2 => Some(JoinDecision::SplitWrite(
            cross_replayability_split_witness(prev, next),
        )),
        // Same-replayability PhysicsRules: WAW check applies as usual.
        (
            ComputeOpKind::PhysicsRule { .. },
            ComputeOpKind::PhysicsRule { .. },
        ) => None,
        // All other prev/next combinations defer to the conventional
        // WAW check. List the remaining pairs explicitly so adding a
        // new `ComputeOpKind` variant forces an explicit decision
        // here (rather than silently inheriting the WAW-only fallback).
        (ComputeOpKind::MaskPredicate { .. }, _)
        | (ComputeOpKind::ScoringArgmax { .. }, _)
        | (ComputeOpKind::SpatialQuery { .. }, _)
        | (ComputeOpKind::Plumbing { .. }, _)
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::MaskPredicate { .. })
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::ScoringArgmax { .. })
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::PhysicsRule { .. })
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::SpatialQuery { .. })
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::Plumbing { .. })
        | (ComputeOpKind::PhysicsRule { .. }, ComputeOpKind::MaskPredicate { .. })
        | (ComputeOpKind::PhysicsRule { .. }, ComputeOpKind::ScoringArgmax { .. })
        | (ComputeOpKind::PhysicsRule { .. }, ComputeOpKind::ViewFold { .. })
        | (ComputeOpKind::PhysicsRule { .. }, ComputeOpKind::SpatialQuery { .. })
        | (ComputeOpKind::PhysicsRule { .. }, ComputeOpKind::Plumbing { .. }) => None,
    }
}

/// True iff `prev` and `next` are both `ViewFold` ops on the same
/// view AND `conflict_key` is a `ViewStorage` write on that view (any
/// slot). Used by [`decide_join`] to convert a same-view-view-fold
/// WAW into a fusion-permitting accumulator pattern.
fn same_view_view_fold_accumulator(
    prev: &ComputeOp,
    next: &ComputeOp,
    conflict_key: &CycleEdgeKey,
) -> bool {
    let view = match (&prev.kind, &next.kind) {
        (
            ComputeOpKind::ViewFold {
                view: v1,
                on_event: _,
                body: _,
            },
            ComputeOpKind::ViewFold {
                view: v2,
                on_event: _,
                body: _,
            },
        ) => {
            if v1 == v2 {
                *v1
            } else {
                return false;
            }
        }
        // Any non-(ViewFold, ViewFold) pairing is by definition not
        // a same-view ViewFold accumulator. Enumerate the prev-side
        // and next-side variants so a future `ComputeOpKind` addition
        // forces an explicit decision here.
        (ComputeOpKind::MaskPredicate { .. }, _)
        | (ComputeOpKind::ScoringArgmax { .. }, _)
        | (ComputeOpKind::PhysicsRule { .. }, _)
        | (ComputeOpKind::SpatialQuery { .. }, _)
        | (ComputeOpKind::Plumbing { .. }, _)
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::MaskPredicate { .. })
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::ScoringArgmax { .. })
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::PhysicsRule { .. })
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::SpatialQuery { .. })
        | (ComputeOpKind::ViewFold { .. }, ComputeOpKind::Plumbing { .. }) => return false,
    };
    match conflict_key {
        CycleEdgeKey::Other(DataHandle::ViewStorage {
            view: vh,
            slot: _,
        }) => *vh == view,
        CycleEdgeKey::Other(DataHandle::AgentField { .. })
        | CycleEdgeKey::Other(DataHandle::ConfigConst { .. })
        | CycleEdgeKey::Other(DataHandle::MaskBitmap { .. })
        | CycleEdgeKey::Other(DataHandle::ScoringOutput)
        | CycleEdgeKey::Other(DataHandle::SpatialStorage { .. })
        | CycleEdgeKey::Other(DataHandle::Rng { .. })
        | CycleEdgeKey::Other(DataHandle::AliveBitmap)
        | CycleEdgeKey::Other(DataHandle::IndirectArgs { .. })
        | CycleEdgeKey::Other(DataHandle::AgentScratch { .. })
        | CycleEdgeKey::Other(DataHandle::SimCfgBuffer)
        | CycleEdgeKey::Other(DataHandle::SnapshotKick)
        | CycleEdgeKey::Other(DataHandle::EventRing { .. })
        | CycleEdgeKey::Ring(_) => false,
    }
}

/// Synthesize a representative `DataHandle` for a cross-view
/// ViewFold split. We prefer the *candidate's* first `ViewStorage`
/// write (it exposes the new view), falling back to the prev op's
/// first `ViewStorage` write, then to a synthetic `ViewStorage`
/// keyed on `next_view` if neither op recorded any (defensive — the
/// emitter always records at least one).
fn cross_view_split_witness(
    prev: &ComputeOp,
    next: &ComputeOp,
    next_view: ViewId,
) -> DataHandle {
    if let Some(h) = first_view_storage_write(next) {
        return h;
    }
    if let Some(h) = first_view_storage_write(prev) {
        return h;
    }
    DataHandle::ViewStorage {
        view: next_view,
        slot: crate::cg::data_handle::ViewStorageSlot::Primary,
    }
}

/// Synthesize a representative `DataHandle` for a cross-replayability
/// PhysicsRule split. Prefer the candidate's first `EventRing`
/// `Append`, falling back to the prev op's first `EventRing`
/// `Append`, then to the candidate's first write of any kind.
fn cross_replayability_split_witness(
    prev: &ComputeOp,
    next: &ComputeOp,
) -> DataHandle {
    if let Some(h) = first_event_ring_append(next) {
        return h;
    }
    if let Some(h) = first_event_ring_append(prev) {
        return h;
    }
    if let Some(h) = next.writes.first() {
        return h.clone();
    }
    if let Some(h) = prev.writes.first() {
        return h.clone();
    }
    // Last-resort fallback: a synthetic indirect-args handle on
    // ring 0. The PhysicsRule emitter always records at least one
    // ring write via `record_write`, so this branch is unreachable
    // in practice.
    DataHandle::IndirectArgs {
        ring: EventRingId(0),
    }
}

fn first_view_storage_write(op: &ComputeOp) -> Option<DataHandle> {
    op.writes.iter().find_map(|h| match h {
        DataHandle::ViewStorage { .. } => Some(h.clone()),
        DataHandle::AgentField { .. }
        | DataHandle::ConfigConst { .. }
        | DataHandle::MaskBitmap { .. }
        | DataHandle::ScoringOutput
        | DataHandle::SpatialStorage { .. }
        | DataHandle::Rng { .. }
        | DataHandle::AliveBitmap
        | DataHandle::IndirectArgs { .. }
        | DataHandle::AgentScratch { .. }
        | DataHandle::SimCfgBuffer
        | DataHandle::SnapshotKick
        | DataHandle::EventRing { .. } => None,
    })
}

fn first_event_ring_append(op: &ComputeOp) -> Option<DataHandle> {
    op.writes.iter().find_map(|h| match h {
        DataHandle::EventRing {
            ring: _,
            kind: crate::cg::data_handle::EventRingAccess::Append,
        } => Some(h.clone()),
        DataHandle::EventRing { .. }
        | DataHandle::AgentField { .. }
        | DataHandle::ViewStorage { .. }
        | DataHandle::ConfigConst { .. }
        | DataHandle::MaskBitmap { .. }
        | DataHandle::ScoringOutput
        | DataHandle::SpatialStorage { .. }
        | DataHandle::Rng { .. }
        | DataHandle::AliveBitmap
        | DataHandle::IndirectArgs { .. }
        | DataHandle::AgentScratch { .. }
        | DataHandle::SimCfgBuffer
        | DataHandle::SnapshotKick => None,
    })
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
/// - Any `PerEvent` group (including singleton) →
///   [`FusibilityClass::Indirect`]. Even a lone `DrainEvents` /
///   `SeedIndirectArgs` op dispatches indirectly via the ring's tail
///   count, so the indirect topology classification is unconditional on
///   group size.
/// - Singleton non-`PerEvent` group → [`FusibilityClass::Split`].
/// - Multi-op group on any non-`PerEvent` shape →
///   [`FusibilityClass::Fused`].
///
/// Visible to the [`super`] schedule module (used by Task 3.3's
/// `strategy` layer to classify Conservative-strategy singletons without
/// re-deriving the rule).
pub(super) fn classify(ops: &[OpId], shape: &DispatchShape) -> FusibilityClass {
    match shape {
        DispatchShape::PerEvent { .. } => FusibilityClass::Indirect,
        DispatchShape::PerAgent
        | DispatchShape::PerPair { .. }
        | DispatchShape::OneShot
        | DispatchShape::PerWord => {
            if ops.len() < 2 {
                FusibilityClass::Split
            } else {
                FusibilityClass::Fused
            }
        }
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
        ViewId, ViewStorageSlot,
    };
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::{CgExpr, LitValue};
    use crate::cg::op::{
        ComputeOpKind, EventKindId, OpId, PlumbingKind, ReplayabilityFlag, Span, SpatialQueryKind,
    };
    use crate::cg::program::{CgProgram, CgProgramBuilder};
    use crate::cg::schedule::topology::dependency_graph;
    use crate::cg::stmt::CgStmtList;

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

        // The two `PerEvent` ops dispatch on distinct rings, so they
        // can't fuse — the analysis splits at the shape boundary. Each
        // resulting singleton is still a `PerEvent` op, however, so its
        // classification is `Indirect` (the ring drives an indirect
        // dispatch even at group size 1).
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].ops, vec![d0]);
        assert_eq!(groups[1].ops, vec![d1]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Indirect
        );
        assert_eq!(
            groups[1].fusibility_classification,
            FusibilityClass::Indirect
        );
    }

    // --- 5c. Singleton PerEvent op classifies as Indirect --------------

    #[test]
    fn singleton_per_event_op_classified_as_indirect() {
        // A lone `DrainEvents` op — no fusable neighbour, but it still
        // dispatches indirectly via the ring's tail count. The
        // classification must be `Indirect`, not `Split`, regardless of
        // group size.
        let mut b = CgProgramBuilder::new();
        let d = add_drain_events_op(&mut b, EventRingId(7));
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![d]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Indirect
        );

        // The diagnostic stream must record the indirect classification
        // (one diagnostic, kind = Indirect).
        assert_eq!(diagnostics.len(), 1);
        assert!(matches!(
            diagnostics[0].kind,
            FusionDiagnosticKind::Indirect
        ));
        assert_eq!(diagnostics[0].ops, vec![d]);
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
        let mut saw_cycle_fallback = false;
        for d in &diagnostics {
            match &d.kind {
                FusionDiagnosticKind::Fused => saw_fused = true,
                FusionDiagnosticKind::Singleton => saw_singleton = true,
                FusionDiagnosticKind::SplitDispatchShapeMismatch => saw_split_shape = true,
                FusionDiagnosticKind::SplitWriteConflict { .. } => saw_split_write = true,
                FusionDiagnosticKind::Indirect => saw_indirect = true,
                FusionDiagnosticKind::CycleFallback => saw_cycle_fallback = true,
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
        // Fused coverage is asserted in
        // `three_per_agent_ops_no_conflicts_fuse_into_one_group`.
        // CycleFallback coverage is asserted in
        // `cycle_in_graph_falls_back_to_source_order` (must NOT fire on
        // an acyclic fixture like this one).
        let _ = saw_fused;
        assert!(
            !saw_cycle_fallback,
            "CycleFallback fired on an acyclic fixture: {:?}",
            diagnostics
        );
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

        let (groups, diagnostics) = fusion_decisions(&prog, &deps);
        // Source-order fallback: [m0, m1]. Same shape, no WAW. Fuses.
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![m0, m1]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Fused
        );

        // The cycle fallback must surface as the FIRST diagnostic in
        // the stream — downstream consumers detect degraded analysis
        // by inspecting `diagnostics[0].kind`.
        assert!(
            !diagnostics.is_empty(),
            "expected at least one diagnostic"
        );
        assert!(
            matches!(diagnostics[0].kind, FusionDiagnosticKind::CycleFallback),
            "expected CycleFallback as first diagnostic, got {:?}",
            diagnostics[0].kind
        );
        assert!(diagnostics[0].ops.is_empty());
        assert!(
            diagnostics[0].message.contains("cycle"),
            "diagnostic message should mention the cycle: {:?}",
            diagnostics[0].message
        );
    }

    // --- bonus: dispatch_shape_key exhaustiveness ----------------------


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

    // --- helpers for ViewFold / PhysicsRule fusion tests ---------------

    /// Add a `ViewFold` op (empty body, no auto-derived writes) on the
    /// given view + event kind. Tests inject view-storage writes via
    /// `record_write` to simulate the body's effects without spinning
    /// up a full statement tree.
    fn add_view_fold_op(
        builder: &mut CgProgramBuilder,
        view: ViewId,
        on_event: EventKindId,
        ring: EventRingId,
    ) -> OpId {
        let body = builder.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        builder
            .add_op(
                ComputeOpKind::ViewFold {
                    view,
                    on_event,
                    body,
                },
                DispatchShape::PerEvent { source_ring: ring },
                Span::dummy(),
            )
            .unwrap()
    }

    /// Add a `PhysicsRule` op (empty body) on the given event kind +
    /// replayability flag. The dispatch shape is `PerEvent` on the
    /// given ring — `replayable` and non-`replayable` rules typically
    /// share a ring under the iter-2 shared-event-ring policy.
    fn add_physics_rule_op(
        builder: &mut CgProgramBuilder,
        rule: crate::cg::op::PhysicsRuleId,
        on_event: EventKindId,
        replayable: ReplayabilityFlag,
        ring: EventRingId,
    ) -> OpId {
        let body = builder.add_stmt_list(CgStmtList::new(vec![])).unwrap();
        builder
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule,
                    on_event: Some(on_event),
                    body,
                    replayable,
                },
                DispatchShape::PerEvent { source_ring: ring },
                Span::dummy(),
            )
            .unwrap()
    }

    fn view_primary_handle(view: ViewId) -> DataHandle {
        DataHandle::ViewStorage {
            view,
            slot: ViewStorageSlot::Primary,
        }
    }

    // --- 11. Same-view ViewFolds fuse despite WAW on view storage -----

    #[test]
    fn same_view_view_folds_fuse_despite_waw() {
        // Two ViewFold handlers on the same view, both writing the
        // view's `Primary` slot. Pre-iter3 this produced a WAW split;
        // post-iter3 the same-view-accumulator rule overrides the
        // WAW and the two handlers fuse into one Indirect group.
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(0);
        let view = ViewId(3);
        let v0 = add_view_fold_op(&mut b, view, EventKindId(0), ring);
        let v1 = add_view_fold_op(&mut b, view, EventKindId(1), ring);
        let mut prog = b.finish();
        // Inject the WAW: both ops write view[#3].primary.
        prog.ops[v0.0 as usize].record_write(view_primary_handle(view));
        prog.ops[v1.0 as usize].record_write(view_primary_handle(view));

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);

        // ONE Indirect group with both ops — the same-view-accumulator
        // override converted the WAW into a fusion-allowing pattern.
        assert_eq!(groups.len(), 1, "groups: {:?}", groups);
        assert_eq!(groups[0].ops, vec![v0, v1]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Indirect
        );

        // No SplitWriteConflict diagnostic should fire — the WAW was
        // legitimised by the accumulator rule.
        assert!(
            !diagnostics
                .iter()
                .any(|d| matches!(d.kind, FusionDiagnosticKind::SplitWriteConflict { .. })),
            "unexpected SplitWriteConflict in: {:?}",
            diagnostics
        );
    }

    // --- 12. Cross-view ViewFolds split into separate kernels ---------

    #[test]
    fn cross_view_view_folds_split_into_separate_kernels() {
        // Two ViewFold ops on DIFFERENT views, sharing the same ring
        // (so dispatch shapes match). Their write sets are disjoint
        // (one writes view[#3], the other view[#4]). Pre-iter3 they
        // would have fused (no WAW, same shape); post-iter3 the
        // cross-view-domain rule splits them into separate Indirect
        // singletons.
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(0);
        let view_a = ViewId(3);
        let view_b = ViewId(4);
        let v0 = add_view_fold_op(&mut b, view_a, EventKindId(0), ring);
        let v1 = add_view_fold_op(&mut b, view_b, EventKindId(0), ring);
        let mut prog = b.finish();
        prog.ops[v0.0 as usize].record_write(view_primary_handle(view_a));
        prog.ops[v1.0 as usize].record_write(view_primary_handle(view_b));

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);

        // Two singleton Indirect groups — cross-view rejection.
        assert_eq!(groups.len(), 2, "groups: {:?}", groups);
        assert_eq!(groups[0].ops, vec![v0]);
        assert_eq!(groups[1].ops, vec![v1]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Indirect
        );
        assert_eq!(
            groups[1].fusibility_classification,
            FusibilityClass::Indirect
        );

        // The cross-view split surfaces as a SplitWriteConflict
        // diagnostic carrying a ViewStorage witness — no new
        // diagnostic kind was added in iter-3, so cross-domain splits
        // re-use the existing write-conflict variant with a
        // representative handle.
        let split = diagnostics
            .iter()
            .find(|d| matches!(d.kind, FusionDiagnosticKind::SplitWriteConflict { .. }))
            .expect("expected SplitWriteConflict diagnostic");
        match &split.kind {
            FusionDiagnosticKind::SplitWriteConflict { handle } => {
                // Witness should be a ViewStorage handle on one of the
                // two views (we prefer the candidate's, i.e. view_b).
                match handle {
                    DataHandle::ViewStorage { view, slot: _ } => {
                        assert!(
                            *view == view_a || *view == view_b,
                            "witness on unexpected view: {:?}",
                            handle
                        );
                    }
                    other => panic!("unexpected witness handle: {:?}", other),
                }
            }
            _ => unreachable!(),
        }
        assert_eq!(split.ops, vec![v0, v1]);
    }

    // --- 13. Same-replayability PhysicsRule fuses (no cross split) ----

    #[test]
    fn same_replayability_physics_fuses() {
        // Two PhysicsRule ops with the SAME replayable flag on the
        // same ring, with disjoint writes. They share the dispatch
        // shape and don't WAW-conflict, and the cross-replayability
        // rule does NOT fire — so they fuse into one Indirect group.
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(0);
        let p0 = add_physics_rule_op(
            &mut b,
            crate::cg::op::PhysicsRuleId(0),
            EventKindId(0),
            ReplayabilityFlag::Replayable,
            ring,
        );
        let p1 = add_physics_rule_op(
            &mut b,
            crate::cg::op::PhysicsRuleId(1),
            EventKindId(1),
            ReplayabilityFlag::Replayable,
            ring,
        );
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, _diagnostics) = fusion_decisions(&prog, &deps);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![p0, p1]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Indirect
        );
    }

    // --- 14. Cross-replayability PhysicsRule splits -------------------

    #[test]
    fn cross_replayability_physics_dont_fuse() {
        // Two PhysicsRule ops on the same ring + dispatch, with
        // DIFFERENT replayable flags. Even if their writes are
        // disjoint, replayable and non-replayable rules emit into
        // different determinism domains — fusion is not allowed.
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(0);
        let p0 = add_physics_rule_op(
            &mut b,
            crate::cg::op::PhysicsRuleId(0),
            EventKindId(0),
            ReplayabilityFlag::Replayable,
            ring,
        );
        let p1 = add_physics_rule_op(
            &mut b,
            crate::cg::op::PhysicsRuleId(1),
            EventKindId(1),
            ReplayabilityFlag::NonReplayable,
            ring,
        );
        let prog = b.finish();

        let deps = dependency_graph(&prog);
        let (groups, diagnostics) = fusion_decisions(&prog, &deps);
        assert_eq!(groups.len(), 2, "groups: {:?}", groups);
        assert_eq!(groups[0].ops, vec![p0]);
        assert_eq!(groups[1].ops, vec![p1]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Indirect
        );
        assert_eq!(
            groups[1].fusibility_classification,
            FusibilityClass::Indirect
        );

        // Cross-replayability split surfaces as SplitWriteConflict
        // (the iter-3 rules re-use the existing diagnostic kind).
        let split = diagnostics
            .iter()
            .find(|d| matches!(d.kind, FusionDiagnosticKind::SplitWriteConflict { .. }))
            .expect("expected SplitWriteConflict diagnostic");
        assert_eq!(split.ops, vec![p0, p1]);
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
