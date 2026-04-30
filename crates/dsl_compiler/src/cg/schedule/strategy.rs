//! Schedule-strategy selector — Phase-3 / Task 3.3.
//!
//! The fusion analysis from Task 3.2
//! ([`super::fusion::fusion_candidates`]) implements one specific
//! grouping policy: walk the topological order, fuse consecutive ops
//! that share a [`super::fusion::DispatchShapeKey`] and don't
//! write-conflict, otherwise split. That policy is one of three the
//! Phase-3 plan calls out.
//!
//! [`ScheduleStrategy`] is the typed selector across the three
//! policies. The strategy layer wraps Task 3.2's analysis: `Default`
//! delegates to it unchanged; `Conservative` collapses every op to its
//! own singleton group; `Megakernel` collapses every op into a single
//! aggregated group regardless of dispatch-shape mismatches or
//! write-after-write hazards.
//!
//! [`fusion_candidates_with_strategy`] and
//! [`fusion_decisions_with_strategy`] are the strategy-aware analogues
//! of Task 3.2's [`super::fusion::fusion_candidates`] /
//! [`super::fusion::fusion_decisions`]. They take a
//! [`ScheduleStrategy`] and return groups (and, for the `_decisions`
//! variant, a decision-diagnostic stream) shaped per the chosen policy.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 3.3, for the design rationale.
//!
//! # Limitations
//!
//! - **Strategy is a fusion policy, not a complete schedule.** The
//!   strategy chooses how the topological walk groups ops; it does
//!   not produce the `ComputeSchedule` value Phase-3 / Task 3.4 will
//!   wrap around the groups (kernel topology, dispatch arguments,
//!   barrier placement). Calling
//!   [`fusion_candidates_with_strategy`] yields the partition; turning
//!   that partition into a schedulable artefact is Task 3.4's job. The
//!   plan's forward-looking
//!   `CgProgramBuilder::build_with_strategy(strategy)` API is therefore
//!   *not* added in Task 3.3 — the strategy is exposed at the
//!   fusion-analysis layer until `ComputeSchedule` exists.
//! - **`Megakernel` ignores dispatch-shape mismatches at the IR
//!   level.** The IR-level decision is "merge every op into one
//!   group". Per-thread branching across mixed dispatch shapes
//!   (PerAgent + OneShot + PerWord etc.) is a Phase-4 emit concern;
//!   this analysis does not attempt to emit the megakernel body or
//!   reason about the runtime dispatch the merged group implies.
//! - **`Megakernel` ignores write-after-write hazards.** The merged
//!   group's [`super::fusion::FusionGroup`] carries every op in
//!   topological order, regardless of whether two members write the
//!   same handle. Phase-4 emit will resolve in-kernel ordering through
//!   per-thread sequencing or barrier insertion; the strategy layer
//!   does not. Callers asking for `Default` semantics MUST pass
//!   [`ScheduleStrategy::Default`] explicitly.
//! - **`Megakernel` representative shape.** The merged group is given
//!   the dispatch shape of the *first* op in topological order. This
//!   is convention only — Phase-4 emit may override it (e.g. emit at
//!   a synthetic mega-shape that drives per-thread branching). The IR
//!   decision is "merge them all"; the representative shape is just a
//!   placeholder so [`super::fusion::FusionGroup`] remains a single
//!   shared structure.
//! - **Cycle fallback is per-strategy.** All three strategies fall
//!   back to source order when [`super::topology::topological_sort`]
//!   reports a cycle, and surface the
//!   [`super::fusion::FusionDiagnosticKind::CycleFallback`] diagnostic
//!   as the first entry in the stream. `Default` inherits this from
//!   Task 3.2's analysis; `Conservative` and `Megakernel` re-emit it
//!   here so downstream consumers see consistent behaviour across
//!   strategies.

use crate::cg::dispatch::DispatchShape;
use crate::cg::op::OpId;
use crate::cg::program::CgProgram;

use super::fusion::{
    classify, fusion_decisions, FusibilityClass, FusionDiagnostic, FusionDiagnosticKind,
    FusionGroup,
};
use super::topology::{topological_sort, DepGraph};

// ---------------------------------------------------------------------------
// ScheduleStrategy
// ---------------------------------------------------------------------------

/// Selector for the schedule-synthesis policy applied by
/// [`fusion_candidates_with_strategy`] /
/// [`fusion_decisions_with_strategy`].
///
/// The plan's three strategies:
///
/// - [`ScheduleStrategy::Conservative`]: no fusion. Every op is its own
///   group. Used as a debugging baseline / regression aide and for
///   backends that don't (yet) support fused kernels.
/// - [`ScheduleStrategy::Default`]: Task 3.2's policy unchanged. Walks
///   the topological order, fuses consecutive ops with matching
///   dispatch shape and no WAW conflict. The production default.
/// - [`ScheduleStrategy::Megakernel`]: aggressive. Every op is merged
///   into a single fused group regardless of dispatch shape or WAW
///   hazards. The IR decision the megakernel emit (Phase-4) consumes.
///
/// `Display` renders `"conservative"` / `"default"` / `"megakernel"`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum ScheduleStrategy {
    /// No fusion — every op becomes its own group, classified per
    /// [`classify`] (singleton `PerEvent` ops are still `Indirect`,
    /// every other singleton is `Split`).
    Conservative,
    /// Task 3.2's fusion policy: consecutive ops with matching dispatch
    /// shape and no WAW conflict fuse into one group; everything else
    /// splits at the boundary. The production default.
    Default,
    /// Aggressive: merge every op (in topological order) into one
    /// fused group, ignoring dispatch-shape mismatches and WAW
    /// hazards. Produces a single [`FusibilityClass::Fused`] group
    /// (or, for an empty program, an empty `Vec`).
    Megakernel,
}

impl ScheduleStrategy {
    /// Stable snake_case label for diagnostics, logs, and CLI flags.
    pub fn label(self) -> &'static str {
        match self {
            ScheduleStrategy::Conservative => "conservative",
            ScheduleStrategy::Default => "default",
            ScheduleStrategy::Megakernel => "megakernel",
        }
    }
}

impl std::fmt::Display for ScheduleStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Strategy-aware fusion entry points
// ---------------------------------------------------------------------------

/// Compute the [`FusionGroup`] partition for `prog` under the
/// dependency graph `deps`, applying the schedule-synthesis policy
/// selected by `strategy`.
///
/// Convenience over [`fusion_decisions_with_strategy`] for callers
/// that don't need the per-decision diagnostic stream.
pub fn fusion_candidates_with_strategy(
    prog: &CgProgram,
    deps: &DepGraph,
    strategy: ScheduleStrategy,
) -> Vec<FusionGroup> {
    fusion_decisions_with_strategy(prog, deps, strategy).0
}

/// Compute the strategy-shaped fusion-group partition AND the
/// diagnostic stream explaining each grouping decision.
///
/// **Determinism:** every strategy walks a deterministic topological
/// order (or, on a cycle, source order with a
/// [`FusionDiagnosticKind::CycleFallback`] diagnostic). Two runs over
/// the same `(prog, deps, strategy)` produce identical
/// `(groups, diagnostics)` outputs.
pub fn fusion_decisions_with_strategy(
    prog: &CgProgram,
    deps: &DepGraph,
    strategy: ScheduleStrategy,
) -> (Vec<FusionGroup>, Vec<FusionDiagnostic>) {
    match strategy {
        ScheduleStrategy::Default => fusion_decisions(prog, deps),
        ScheduleStrategy::Conservative => conservative_decisions(prog, deps),
        ScheduleStrategy::Megakernel => megakernel_decisions(prog, deps),
    }
}

// ---------------------------------------------------------------------------
// Conservative strategy
// ---------------------------------------------------------------------------

/// `Conservative` analogue of [`fusion_decisions`]: emit one singleton
/// [`FusionGroup`] per op, in topological order.
///
/// Each op is classified by [`super::fusion::classify`] (so a lone
/// `PerEvent` op surfaces as [`FusibilityClass::Indirect`], and every
/// other lone op as [`FusibilityClass::Split`]). The diagnostic stream
/// records one [`FusionDiagnosticKind::Singleton`] (or
/// [`FusionDiagnosticKind::Indirect`]) entry per group, plus a leading
/// [`FusionDiagnosticKind::CycleFallback`] entry if the topological
/// sort reports a cycle.
fn conservative_decisions(
    prog: &CgProgram,
    deps: &DepGraph,
) -> (Vec<FusionGroup>, Vec<FusionDiagnostic>) {
    let mut groups: Vec<FusionGroup> = Vec::new();
    let mut diagnostics: Vec<FusionDiagnostic> = Vec::new();

    let topo_order = resolve_walk_order(prog, deps, &mut diagnostics);

    for op_id in topo_order {
        let op = match prog.ops.get(op_id.0 as usize) {
            Some(op) => op,
            // Defensive: same posture as `fusion_decisions` — a
            // well-formed graph yields only in-range OpIds, but a
            // malformed graph slipping through must not panic.
            None => continue,
        };
        let shape = op.shape;
        let ops = vec![op_id];
        let classification = classify(&ops, &shape);
        let (kind, message) = match classification {
            FusibilityClass::Split => (
                FusionDiagnosticKind::Singleton,
                format!(
                    "conservative: singleton op#{} on shape {}",
                    op_id.0, shape
                ),
            ),
            FusibilityClass::Indirect => (
                FusionDiagnosticKind::Indirect,
                format!(
                    "conservative: indirect singleton op#{} on shape {}",
                    op_id.0, shape
                ),
            ),
            FusibilityClass::Fused => {
                // Unreachable for size-1 ops under `classify`'s rules
                // (multi-op-only path). Emit a Singleton diagnostic
                // and downgrade the classification to keep the
                // strategy invariant ("Conservative emits no Fused
                // groups") true. This branch is structurally
                // unreachable; keeping it explicit avoids a `_ =>`
                // fallthrough.
                (
                    FusionDiagnosticKind::Singleton,
                    format!(
                        "conservative: singleton op#{} on shape {} (downgraded)",
                        op_id.0, shape
                    ),
                )
            }
        };
        let final_class = match classification {
            FusibilityClass::Fused => FusibilityClass::Split,
            other => other,
        };
        diagnostics.push(FusionDiagnostic {
            kind,
            ops: ops.clone(),
            message,
        });
        groups.push(FusionGroup {
            ops,
            shape,
            fusibility_classification: final_class,
        });
    }

    (groups, diagnostics)
}

// ---------------------------------------------------------------------------
// Megakernel strategy
// ---------------------------------------------------------------------------

/// `Megakernel` analogue of [`fusion_decisions`]: collect every op (in
/// topological order) into a single [`FusibilityClass::Fused`] group,
/// regardless of dispatch-shape mismatches or write-after-write
/// hazards. Empty programs yield an empty `Vec`.
///
/// The merged group's `shape` is the first op's dispatch shape — see
/// the module-level `# Limitations` for the convention. The
/// diagnostic stream emits one [`FusionDiagnosticKind::Fused`] entry
/// covering every member op (plus a leading
/// [`FusionDiagnosticKind::CycleFallback`] entry on cycle).
fn megakernel_decisions(
    prog: &CgProgram,
    deps: &DepGraph,
) -> (Vec<FusionGroup>, Vec<FusionDiagnostic>) {
    let mut groups: Vec<FusionGroup> = Vec::new();
    let mut diagnostics: Vec<FusionDiagnostic> = Vec::new();

    let topo_order = resolve_walk_order(prog, deps, &mut diagnostics);

    // Filter to in-range OpIds and capture the representative shape
    // (first op in walk order). An empty program produces an empty
    // `Vec` and no group/classification diagnostic — matches
    // [`fusion_decisions`]'s empty-program behaviour.
    let mut ops: Vec<OpId> = Vec::with_capacity(topo_order.len());
    let mut representative_shape: Option<DispatchShape> = None;
    for op_id in topo_order {
        let op = match prog.ops.get(op_id.0 as usize) {
            Some(op) => op,
            None => continue,
        };
        if representative_shape.is_none() {
            representative_shape = Some(op.shape);
        }
        ops.push(op_id);
    }

    let shape = match representative_shape {
        Some(shape) => shape,
        None => return (groups, diagnostics),
    };

    let message = format!(
        "megakernel: merged {} ops into one fused group on representative shape {}",
        ops.len(),
        shape
    );
    diagnostics.push(FusionDiagnostic {
        kind: FusionDiagnosticKind::Fused,
        ops: ops.clone(),
        message,
    });
    groups.push(FusionGroup {
        ops,
        shape,
        fusibility_classification: FusibilityClass::Fused,
    });

    (groups, diagnostics)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Resolve a deterministic walk order. On cycle, fall back to source
/// order AND surface a [`FusionDiagnosticKind::CycleFallback`]
/// diagnostic as the first entry in the stream. Mirrors
/// [`fusion_decisions`]'s cycle posture so all three strategies agree
/// on the diagnostic shape.
fn resolve_walk_order(
    prog: &CgProgram,
    deps: &DepGraph,
    diagnostics: &mut Vec<FusionDiagnostic>,
) -> Vec<OpId> {
    match topological_sort(deps) {
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
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cg::data_handle::{AgentFieldId, AgentRef, DataHandle, EventRingId, MaskId};
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::{CgExpr, LitValue};
    use crate::cg::op::{ComputeOpKind, OpId, PlumbingKind, Span};
    use crate::cg::program::{CgProgram, CgProgramBuilder};
    use crate::cg::schedule::fusion::fusion_candidates;
    use crate::cg::schedule::topology::dependency_graph;

    // --- helpers -------------------------------------------------------

    fn add_mask_op(
        builder: &mut CgProgramBuilder,
        mask: MaskId,
        shape: DispatchShape,
    ) -> OpId {
        let pred = builder
            .add_expr(CgExpr::Lit(LitValue::Bool(true)))
            .unwrap();
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

    fn mana_handle() -> DataHandle {
        DataHandle::AgentField {
            field: AgentFieldId::Mana,
            target: AgentRef::Self_,
        }
    }

    // --- 1. Empty program × 3 strategies -------------------------------

    #[test]
    fn empty_program_yields_empty_groups_under_every_strategy() {
        let prog = CgProgram::new();
        let deps = dependency_graph(&prog);
        for strategy in [
            ScheduleStrategy::Conservative,
            ScheduleStrategy::Default,
            ScheduleStrategy::Megakernel,
        ] {
            let (groups, diagnostics) =
                fusion_decisions_with_strategy(&prog, &deps, strategy);
            assert!(
                groups.is_empty(),
                "strategy {:?}: expected empty groups, got {:?}",
                strategy,
                groups
            );
            assert!(
                diagnostics.is_empty(),
                "strategy {:?}: expected empty diagnostics, got {:?}",
                strategy,
                diagnostics
            );
            assert!(fusion_candidates_with_strategy(&prog, &deps, strategy).is_empty());
        }
    }

    // --- 2. Conservative — multi-op program: one group per op ----------

    #[test]
    fn conservative_emits_one_group_per_op() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let m2 = add_mask_op(&mut b, MaskId(2), DispatchShape::PerAgent);
        let prog = b.finish();
        let deps = dependency_graph(&prog);

        let (groups, diagnostics) =
            fusion_decisions_with_strategy(&prog, &deps, ScheduleStrategy::Conservative);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].ops, vec![m0]);
        assert_eq!(groups[1].ops, vec![m1]);
        assert_eq!(groups[2].ops, vec![m2]);
        for group in &groups {
            assert_eq!(group.shape, DispatchShape::PerAgent);
            assert_eq!(group.fusibility_classification, FusibilityClass::Split);
        }
        // 3 Singleton diagnostics, no cycle-fallback.
        assert_eq!(diagnostics.len(), 3);
        for d in &diagnostics {
            assert!(matches!(d.kind, FusionDiagnosticKind::Singleton));
        }
    }

    // --- 2b. Conservative classifies a lone PerEvent op as Indirect ----

    #[test]
    fn conservative_classifies_singleton_per_event_as_indirect() {
        let mut b = CgProgramBuilder::new();
        let d = add_drain_events_op(&mut b, EventRingId(7));
        let prog = b.finish();
        let deps = dependency_graph(&prog);

        let (groups, diagnostics) =
            fusion_decisions_with_strategy(&prog, &deps, ScheduleStrategy::Conservative);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ops, vec![d]);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Indirect
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(matches!(diagnostics[0].kind, FusionDiagnosticKind::Indirect));
    }

    // --- 2c. Conservative splits even on fusable adjacent ops ----------

    #[test]
    fn conservative_does_not_fuse_adjacent_compatible_ops() {
        // Two PerAgent ops with no WAW conflict — Default would fuse
        // them. Conservative MUST NOT.
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let prog = b.finish();
        let deps = dependency_graph(&prog);

        let conservative_groups =
            fusion_candidates_with_strategy(&prog, &deps, ScheduleStrategy::Conservative);
        let default_groups =
            fusion_candidates_with_strategy(&prog, &deps, ScheduleStrategy::Default);

        assert_eq!(conservative_groups.len(), 2);
        assert_eq!(conservative_groups[0].ops, vec![m0]);
        assert_eq!(conservative_groups[1].ops, vec![m1]);
        for group in &conservative_groups {
            assert_eq!(group.fusibility_classification, FusibilityClass::Split);
        }

        // Default fuses the same input.
        assert_eq!(default_groups.len(), 1);
        assert_eq!(default_groups[0].ops, vec![m0, m1]);
        assert_eq!(
            default_groups[0].fusibility_classification,
            FusibilityClass::Fused
        );
    }

    // --- 3. Default delegates to Task 3.2 unchanged --------------------

    #[test]
    fn default_strategy_matches_task_3_2_fusion_candidates() {
        // Build a non-trivial fixture exercising shape-mismatch +
        // singleton + indirect grouping. The strategy-aware entry
        // point at `Default` MUST produce byte-identical output to
        // [`fusion_candidates`] / [`fusion_decisions`].
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let upload = add_upload_sim_cfg_op(&mut b);
        let d0 = add_drain_events_op(&mut b, EventRingId(2));
        let d1 = add_drain_events_op(&mut b, EventRingId(2));
        let m5 = add_mask_op(&mut b, MaskId(5), DispatchShape::PerAgent);
        let prog = b.finish();
        let deps = dependency_graph(&prog);

        let (groups_a, diagnostics_a) =
            fusion_decisions_with_strategy(&prog, &deps, ScheduleStrategy::Default);
        let direct_groups = fusion_candidates(&prog, &deps);
        let (direct_groups_check, direct_diagnostics) =
            crate::cg::schedule::fusion::fusion_decisions(&prog, &deps);

        assert_eq!(groups_a, direct_groups);
        assert_eq!(groups_a, direct_groups_check);
        assert_eq!(diagnostics_a, direct_diagnostics);
        // Sanity touch-up to keep ids alive in the assert chain.
        let _ = (m0, m1, upload, d0, d1, m5);
    }

    // --- 4. Megakernel — multi-op program: one giant group -------------

    #[test]
    fn megakernel_collapses_every_op_into_one_fused_group() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let m2 = add_mask_op(&mut b, MaskId(2), DispatchShape::PerAgent);
        let prog = b.finish();
        let deps = dependency_graph(&prog);

        let (groups, diagnostics) =
            fusion_decisions_with_strategy(&prog, &deps, ScheduleStrategy::Megakernel);

        assert_eq!(groups.len(), 1, "megakernel must produce a single group");
        assert_eq!(groups[0].ops, vec![m0, m1, m2]);
        assert_eq!(groups[0].shape, DispatchShape::PerAgent);
        assert_eq!(
            groups[0].fusibility_classification,
            FusibilityClass::Fused
        );

        // One Fused diagnostic covering every member op.
        assert_eq!(diagnostics.len(), 1);
        assert!(matches!(diagnostics[0].kind, FusionDiagnosticKind::Fused));
        assert_eq!(diagnostics[0].ops, vec![m0, m1, m2]);
    }

    // --- 5. Megakernel ignores WAW conflicts ---------------------------

    #[test]
    fn megakernel_ignores_write_after_write_conflicts() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let mut prog = b.finish();
        // Inject WAW: both ops write hp.
        prog.ops[m0.0 as usize].record_write(hp_handle());
        prog.ops[m1.0 as usize].record_write(hp_handle());

        let deps = dependency_graph(&prog);
        let mega_groups =
            fusion_candidates_with_strategy(&prog, &deps, ScheduleStrategy::Megakernel);
        let default_groups =
            fusion_candidates_with_strategy(&prog, &deps, ScheduleStrategy::Default);

        // Default splits at the WAW boundary.
        assert_eq!(default_groups.len(), 2);
        assert_eq!(default_groups[0].ops, vec![m0]);
        assert_eq!(default_groups[1].ops, vec![m1]);

        // Megakernel ignores WAW and produces a single fused group.
        assert_eq!(mega_groups.len(), 1);
        assert_eq!(mega_groups[0].ops, vec![m0, m1]);
        assert_eq!(
            mega_groups[0].fusibility_classification,
            FusibilityClass::Fused
        );
    }

    // --- 6. Megakernel ignores dispatch-shape mismatches ---------------

    #[test]
    fn megakernel_merges_per_agent_one_shot_and_per_event_into_one_group() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let upload = add_upload_sim_cfg_op(&mut b);
        let d0 = add_drain_events_op(&mut b, EventRingId(3));
        let prog = b.finish();
        let deps = dependency_graph(&prog);

        let mega_groups =
            fusion_candidates_with_strategy(&prog, &deps, ScheduleStrategy::Megakernel);
        let default_groups =
            fusion_candidates_with_strategy(&prog, &deps, ScheduleStrategy::Default);

        // Default splits at every shape boundary → 3 singleton groups.
        assert_eq!(default_groups.len(), 3);

        // Megakernel folds them all together. Representative shape =
        // first op's (PerAgent).
        assert_eq!(mega_groups.len(), 1);
        assert_eq!(mega_groups[0].ops, vec![m0, upload, d0]);
        assert_eq!(mega_groups[0].shape, DispatchShape::PerAgent);
        assert_eq!(
            mega_groups[0].fusibility_classification,
            FusibilityClass::Fused
        );
    }

    // --- 7. Topo order is consistent across strategies -----------------

    #[test]
    fn strategy_choice_does_not_change_topological_order() {
        // Build a fixture where dependency edges force a specific
        // topo order (m0 before m1 because m1 reads what m0 writes),
        // and verify all three strategies preserve that order in
        // their group `ops` lists when flattened.
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let m2 = add_mask_op(&mut b, MaskId(2), DispatchShape::PerAgent);
        let mut prog = b.finish();
        prog.ops[m0.0 as usize].record_write(hp_handle());
        prog.ops[m1.0 as usize].record_read(hp_handle());
        prog.ops[m1.0 as usize].record_write(mana_handle());
        prog.ops[m2.0 as usize].record_read(mana_handle());

        let deps = dependency_graph(&prog);

        for strategy in [
            ScheduleStrategy::Conservative,
            ScheduleStrategy::Default,
            ScheduleStrategy::Megakernel,
        ] {
            let groups = fusion_candidates_with_strategy(&prog, &deps, strategy);
            let flattened: Vec<OpId> =
                groups.iter().flat_map(|g| g.ops.iter().copied()).collect();
            assert_eq!(
                flattened,
                vec![m0, m1, m2],
                "strategy {:?}: flattened order changed",
                strategy
            );
        }
    }

    // --- 8. Cycle fallback under each strategy -------------------------

    #[test]
    fn cycle_input_falls_back_to_source_order_under_every_strategy() {
        // Cyclic fixture: op0 reads Hp, writes Mana; op1 reads Mana,
        // writes Hp. Both PerAgent. `topological_sort` returns
        // CycleError; every strategy must fall back to source order
        // and surface a leading CycleFallback diagnostic.
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

        for strategy in [
            ScheduleStrategy::Conservative,
            ScheduleStrategy::Default,
            ScheduleStrategy::Megakernel,
        ] {
            let (groups, diagnostics) =
                fusion_decisions_with_strategy(&prog, &deps, strategy);

            assert!(
                !diagnostics.is_empty(),
                "strategy {:?}: expected at least one diagnostic on cycle",
                strategy
            );
            assert!(
                matches!(
                    diagnostics[0].kind,
                    FusionDiagnosticKind::CycleFallback
                ),
                "strategy {:?}: expected CycleFallback as first diagnostic, got {:?}",
                strategy,
                diagnostics[0].kind
            );

            // Source-order fallback = [m0, m1] for every strategy. The
            // partition shape differs by strategy but the flattened
            // sequence is invariant.
            let flattened: Vec<OpId> =
                groups.iter().flat_map(|g| g.ops.iter().copied()).collect();
            assert_eq!(
                flattened,
                vec![m0, m1],
                "strategy {:?}: expected source-order [m0, m1] on cycle",
                strategy
            );
        }
    }

    // --- 9. Determinism across two runs --------------------------------

    #[test]
    fn two_runs_under_each_strategy_produce_identical_output() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let upload = add_upload_sim_cfg_op(&mut b);
        let d0 = add_drain_events_op(&mut b, EventRingId(4));
        let mut prog = b.finish();
        prog.ops[m0.0 as usize].record_write(hp_handle());
        prog.ops[m1.0 as usize].record_read(hp_handle());
        let _ = (upload, d0);

        let deps = dependency_graph(&prog);

        for strategy in [
            ScheduleStrategy::Conservative,
            ScheduleStrategy::Default,
            ScheduleStrategy::Megakernel,
        ] {
            let first = fusion_decisions_with_strategy(&prog, &deps, strategy);
            let second = fusion_decisions_with_strategy(&prog, &deps, strategy);
            assert_eq!(
                first, second,
                "strategy {:?}: two runs differ",
                strategy
            );
        }
    }

    // --- bonus: ScheduleStrategy Display labels ------------------------

    #[test]
    fn schedule_strategy_display_renders_snake_case_labels() {
        assert_eq!(format!("{}", ScheduleStrategy::Conservative), "conservative");
        assert_eq!(format!("{}", ScheduleStrategy::Default), "default");
        assert_eq!(format!("{}", ScheduleStrategy::Megakernel), "megakernel");
        assert_eq!(ScheduleStrategy::Conservative.label(), "conservative");
        assert_eq!(ScheduleStrategy::Default.label(), "default");
        assert_eq!(ScheduleStrategy::Megakernel.label(), "megakernel");
    }

    // --- bonus: single-op behaviour across strategies ------------------

    #[test]
    fn single_op_program_behaviour_across_strategies() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let prog = b.finish();
        let deps = dependency_graph(&prog);

        // Conservative + Default: one Split group.
        for strategy in [ScheduleStrategy::Conservative, ScheduleStrategy::Default] {
            let groups = fusion_candidates_with_strategy(&prog, &deps, strategy);
            assert_eq!(groups.len(), 1);
            assert_eq!(groups[0].ops, vec![m0]);
            assert_eq!(
                groups[0].fusibility_classification,
                FusibilityClass::Split,
                "strategy {:?}",
                strategy
            );
        }

        // Megakernel: one Fused group (even with a single op — the
        // strategy unconditionally classifies the merged group as
        // Fused, in contrast to `classify`'s singleton-as-Split rule).
        let mega_groups =
            fusion_candidates_with_strategy(&prog, &deps, ScheduleStrategy::Megakernel);
        assert_eq!(mega_groups.len(), 1);
        assert_eq!(mega_groups[0].ops, vec![m0]);
        assert_eq!(
            mega_groups[0].fusibility_classification,
            FusibilityClass::Fused
        );
    }
}
