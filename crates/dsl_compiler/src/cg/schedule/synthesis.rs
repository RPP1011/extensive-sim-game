//! Schedule synthesis — Phase-3 / Task 3.4.
//!
//! Wraps the strategy-aware fusion analysis from Task 3.3
//! ([`super::strategy::fusion_decisions_with_strategy`]) in a
//! schedulable artefact: a [`ComputeSchedule`] of [`ComputeStage`]s,
//! each carrying a [`KernelTopology`] that names the fused / split /
//! indirect-dispatch lowering Phase-4 emit will produce.
//!
//! Where Task 3.2 / 3.3 stop at "groups of ops with a fusibility
//! classification", Task 3.4 makes two additional decisions:
//!
//! 1. **Translate [`super::fusion::FusionGroup`] →
//!    [`KernelTopology`].** A `Fused` group becomes
//!    [`KernelTopology::Fused`]; a `Split` singleton becomes
//!    [`KernelTopology::Split`]; a `PerEvent` group classified
//!    `Indirect` becomes [`KernelTopology::Indirect`] when a
//!    `SeedIndirectArgs` producer is identifiable in the program (and
//!    falls back to [`KernelTopology::Fused`] with a
//!    [`ScheduleDiagnostic`] otherwise).
//! 2. **Stage the kernels.** Each [`KernelTopology`] is wrapped in its
//!    own [`ComputeStage`], yielding a `Vec<ComputeStage>` whose
//!    sequential execution preserves Task 3.2's topological order.
//!
//! [`synthesize_schedule`] is the entry point: it takes a
//! [`CgProgram`] + [`ScheduleStrategy`] and returns a
//! [`ScheduleSynthesisResult`] (the schedule plus the fusion- and
//! schedule-level diagnostic streams).
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 3.4, for the design rationale.
//!
//! # Limitations
//!
//! - **One kernel per stage.** Today every [`KernelTopology`] becomes
//!   its own [`ComputeStage`]. Future passes (Task 3.5+) may merge
//!   non-conflicting kernels — kernels whose reads / writes do not
//!   overlap — into a single stage so the runtime can dispatch them
//!   in parallel. The [`ComputeStage`] type carries `Vec<KernelTopology>`
//!   to leave room for that without an API break.
//! - **Indirect producer lookup is structural, not data-flow.** The
//!   [`KernelTopology::Indirect`] producer is discovered by scanning
//!   `prog.ops` for a [`PlumbingKind::SeedIndirectArgs { ring }`]
//!   matching the consumer ops' `DispatchShape::PerEvent { source_ring }`.
//!   When no such op exists in the program (e.g. the plumbing pass
//!   has not run, or the producer lives in a different program
//!   fragment), the topology falls back to [`KernelTopology::Fused`]
//!   and a [`ScheduleDiagnosticKind::IndirectProducerMissing`]
//!   diagnostic surfaces the degraded lowering. Phase-4 emit treats
//!   the fallback as a direct dispatch.
//! - **Single-op `Fused` topologies are valid.** When the strategy is
//!   [`ScheduleStrategy::Megakernel`] and the program contains
//!   exactly one op, the synthesizer surfaces a
//!   [`KernelTopology::Fused`] with `ops.len() == 1`. Phase-4 emit
//!   MUST handle that case symmetrically with multi-op fused groups
//!   (no minimum-size special case). See [`super::strategy`]'s
//!   `# Limitations` for the underlying invariant.
//! - **Cycle fallback inherits from fusion analysis.** When
//!   [`super::topology::topological_sort`] reports a cycle, the
//!   underlying fusion analysis falls back to source order and
//!   surfaces a [`super::fusion::FusionDiagnosticKind::CycleFallback`]
//!   diagnostic. The schedule synthesizer does not add a redundant
//!   layer: the resulting [`ComputeSchedule`] reflects the source-
//!   order grouping and downstream consumers detect the degraded
//!   analysis by inspecting the fusion diagnostic stream.

use std::fmt;

use crate::cg::data_handle::EventRingId;
use crate::cg::dispatch::DispatchShape;
use crate::cg::op::{ComputeOpKind, OpId, PlumbingKind};
use crate::cg::program::CgProgram;

use super::fusion::{FusibilityClass, FusionDiagnostic, FusionGroup};
use super::strategy::{fusion_decisions_with_strategy, ScheduleStrategy};
use super::topology::dependency_graph;

// ---------------------------------------------------------------------------
// KernelTopology
// ---------------------------------------------------------------------------

/// How a single kernel in a [`ComputeStage`] is laid out for emit.
///
/// Each variant corresponds to one [`super::fusion::FusibilityClass`]
/// outcome:
///
/// - [`KernelTopology::Fused`] — multi-op (or, under
///   [`ScheduleStrategy::Megakernel`], single-op) fused kernel sharing
///   one dispatch shape. Produced from
///   [`FusibilityClass::Fused`] groups, and from
///   [`FusibilityClass::Indirect`] groups that lack an identifiable
///   `SeedIndirectArgs` producer (fallback).
/// - [`KernelTopology::Split`] — singleton kernel running as its own
///   dispatch. Produced from [`FusibilityClass::Split`] groups.
/// - [`KernelTopology::Indirect`] — split into a producer (the
///   `SeedIndirectArgs` op writing the indirect-args buffer) and one
///   or more consumers (the `PerEvent` ops driven by the ring's tail
///   count). Phase-4 emit lowers the consumers as
///   `dispatch_workgroups_indirect(producer_buffer)` calls. Produced
///   from [`FusibilityClass::Indirect`] groups when the producer is
///   identifiable.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum KernelTopology {
    /// Multi-op (or megakernel-singleton) kernel sharing one dispatch
    /// shape. `ops` is in topological order.
    Fused {
        /// Members of the fused kernel, in topological order.
        ops: Vec<OpId>,
        /// Dispatch shape every member shares.
        dispatch: DispatchShape,
    },
    /// Singleton kernel running as its own dispatch.
    Split {
        /// The lone op driving the kernel.
        op: OpId,
        /// Dispatch shape the op carries.
        dispatch: DispatchShape,
    },
    /// Indirect-dispatch topology — `producer` writes the indirect-
    /// args buffer; `consumers` are the `PerEvent` ops driven by the
    /// ring's tail count. `consumers` is in topological order; the
    /// dispatch shape every consumer carries is implicit in the
    /// program (each consumer's op has a
    /// [`DispatchShape::PerEvent`] referencing the same `source_ring`
    /// as the producer's [`PlumbingKind::SeedIndirectArgs`]).
    Indirect {
        /// The `SeedIndirectArgs` producer op writing the indirect-
        /// args buffer.
        producer: OpId,
        /// The `PerEvent` consumer ops driven by the producer's
        /// indirect-args buffer.
        consumers: Vec<OpId>,
    },
}

impl KernelTopology {
    /// Stable snake_case label for diagnostics, logs, and snapshot
    /// tests. Mirrors the [`FusibilityClass::label`] vocabulary so
    /// schedule and fusion diagnostics share a stable lexicon.
    pub fn label(&self) -> &'static str {
        match self {
            KernelTopology::Fused { .. } => "fused",
            KernelTopology::Split { .. } => "split",
            KernelTopology::Indirect { .. } => "indirect",
        }
    }

    /// Every op the topology covers, in topological order. Convenience
    /// for callers that want a flat op list without matching on the
    /// variant.
    pub fn ops(&self) -> Vec<OpId> {
        match self {
            KernelTopology::Fused { ops, .. } => ops.clone(),
            KernelTopology::Split { op, .. } => vec![*op],
            KernelTopology::Indirect {
                producer,
                consumers,
            } => {
                let mut all = Vec::with_capacity(consumers.len() + 1);
                all.push(*producer);
                all.extend(consumers.iter().copied());
                all
            }
        }
    }
}

impl fmt::Display for KernelTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelTopology::Fused { ops, dispatch } => {
                write!(f, "fused(ops=[")?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "op#{}", op.0)?;
                }
                write!(f, "], dispatch={})", dispatch)
            }
            KernelTopology::Split { op, dispatch } => {
                write!(f, "split(op=op#{}, dispatch={})", op.0, dispatch)
            }
            KernelTopology::Indirect {
                producer,
                consumers,
            } => {
                write!(f, "indirect(producer=op#{}, consumers=[", producer.0)?;
                for (i, op) in consumers.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "op#{}", op.0)?;
                }
                f.write_str("])")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeStage
// ---------------------------------------------------------------------------

/// One stage of a [`ComputeSchedule`]. Today each stage carries
/// exactly one [`KernelTopology`]; the `Vec` shape leaves room for
/// future merging passes (Task 3.5+) to fold non-conflicting kernels
/// into a single parallel-dispatch stage. See the module-level
/// `# Limitations`.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ComputeStage {
    /// Kernels in this stage. Ordered identically to the
    /// [`KernelTopology`] sequence in the surrounding
    /// [`ComputeSchedule`].
    pub kernels: Vec<KernelTopology>,
}

impl fmt::Display for ComputeStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("stage([")?;
        for (i, k) in self.kernels.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", k)?;
        }
        f.write_str("])")
    }
}

// ---------------------------------------------------------------------------
// ComputeSchedule
// ---------------------------------------------------------------------------

/// A schedulable lowering of a [`CgProgram`]. The schedule's stages
/// run sequentially; each stage's kernels run as a unit (today, one
/// kernel per stage — see the module-level `# Limitations`).
///
/// Constructed via [`synthesize_schedule`].
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ComputeSchedule {
    /// Stages, in execution order.
    pub stages: Vec<ComputeStage>,
}

impl ComputeSchedule {
    /// Number of stages in the schedule.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Total kernel count across every stage. Today equal to the stage
    /// count (see `# Limitations`); diverges once a future pass merges
    /// non-conflicting kernels into a single stage.
    pub fn kernel_count(&self) -> usize {
        self.stages.iter().map(|s| s.kernels.len()).sum()
    }
}

impl fmt::Display for ComputeSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("schedule([")?;
        for (i, stage) in self.stages.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", stage)?;
        }
        f.write_str("])")
    }
}

// ---------------------------------------------------------------------------
// ScheduleDiagnostic
// ---------------------------------------------------------------------------

/// Schedule-synthesis-level diagnostic — distinct from the upstream
/// [`FusionDiagnostic`] stream so callers can attribute findings to
/// the correct pass.
///
/// Today the only kind is
/// [`ScheduleDiagnosticKind::IndirectProducerMissing`], surfaced when
/// an [`FusibilityClass::Indirect`] group has no identifiable
/// `SeedIndirectArgs` producer in the program and the synthesizer
/// falls back to [`KernelTopology::Fused`].
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct ScheduleDiagnostic {
    /// Typed payload — the structural reason for the diagnostic.
    pub kind: ScheduleDiagnosticKind,
    /// Index of the [`ComputeStage`] in
    /// [`ComputeSchedule::stages`] the diagnostic concerns.
    pub stage_index: usize,
    /// Human-readable rendering. The structural payload is on `kind`;
    /// `message` is a convenience for logs.
    pub message: String,
}

/// Typed payload for a [`ScheduleDiagnostic`].
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum ScheduleDiagnosticKind {
    /// An [`FusibilityClass::Indirect`] group surfaced a `PerEvent`
    /// dispatch shape, but no [`PlumbingKind::SeedIndirectArgs { ring }`]
    /// op writing the matching `source_ring` was found in the
    /// program. The synthesizer fell back to
    /// [`KernelTopology::Fused`]; downstream emit will dispatch
    /// directly. Inspect the program's plumbing op set if the indirect
    /// lowering was expected.
    IndirectProducerMissing {
        /// The `source_ring` the consumers' `PerEvent` shape
        /// references.
        source_ring: EventRingId,
    },
}

impl ScheduleDiagnosticKind {
    /// Stable snake_case label for diagnostics + log routing.
    pub fn label(&self) -> &'static str {
        match self {
            ScheduleDiagnosticKind::IndirectProducerMissing { .. } => {
                "indirect_producer_missing"
            }
        }
    }
}

impl fmt::Display for ScheduleDiagnosticKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScheduleDiagnosticKind::IndirectProducerMissing { source_ring } => {
                write!(f, "indirect_producer_missing(ring=#{})", source_ring.0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScheduleSynthesisResult
// ---------------------------------------------------------------------------

/// The full output of [`synthesize_schedule`]: the lowered
/// [`ComputeSchedule`] plus both diagnostic streams (fusion-level from
/// Task 3.2 / 3.3, and schedule-level produced here).
///
/// Keeping the two diagnostic streams separate lets callers route
/// fusion-pass concerns and schedule-pass concerns through their own
/// log facilities without re-tagging.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ScheduleSynthesisResult {
    /// The synthesised schedule.
    pub schedule: ComputeSchedule,
    /// Diagnostics surfaced by the upstream fusion analysis (Task 3.2 /
    /// 3.3). Includes cycle-fallback notices and per-group
    /// classification entries.
    pub fusion_diagnostics: Vec<FusionDiagnostic>,
    /// Diagnostics surfaced by schedule synthesis itself. Today the
    /// only kind is
    /// [`ScheduleDiagnosticKind::IndirectProducerMissing`].
    pub schedule_diagnostics: Vec<ScheduleDiagnostic>,
}

// ---------------------------------------------------------------------------
// synthesize_schedule
// ---------------------------------------------------------------------------

/// Synthesize a [`ComputeSchedule`] from `prog` under `strategy`.
///
/// Algorithm:
///
/// 1. Build the dependency graph via [`dependency_graph`].
/// 2. Run [`fusion_decisions_with_strategy`] to obtain the
///    strategy-shaped [`FusionGroup`] partition + fusion-diagnostic
///    stream.
/// 3. For each [`FusionGroup`], translate into a [`KernelTopology`]:
///    - [`FusibilityClass::Fused`] → [`KernelTopology::Fused`].
///    - [`FusibilityClass::Split`] (always a singleton) →
///      [`KernelTopology::Split`].
///    - [`FusibilityClass::Indirect`] → look up the
///      `SeedIndirectArgs` producer for the consumers' `source_ring`;
///      on hit, [`KernelTopology::Indirect`]; on miss,
///      [`KernelTopology::Fused`] + a
///      [`ScheduleDiagnosticKind::IndirectProducerMissing`]
///      diagnostic.
/// 4. Wrap each topology in a single-kernel [`ComputeStage`].
///
/// **Determinism:** the walk consumes a deterministic topological
/// order and never iterates on hash-set iteration order. Two runs over
/// the same `(prog, strategy)` produce byte-identical outputs.
///
/// # Limitations
///
/// See the module-level `# Limitations` block for the full list. The
/// short version: today each [`ComputeStage`] holds exactly one
/// [`KernelTopology`] (no kernel-merging pass yet); indirect producer
/// lookup is structural only (a missing producer triggers a `Fused`
/// fallback with a diagnostic); single-op `Fused` topologies are
/// valid under [`ScheduleStrategy::Megakernel`]; cycle fallback is
/// inherited from the fusion analysis.
pub fn synthesize_schedule(
    prog: &CgProgram,
    strategy: ScheduleStrategy,
) -> ScheduleSynthesisResult {
    let deps = dependency_graph(prog);
    let (groups, fusion_diagnostics) =
        fusion_decisions_with_strategy(prog, &deps, strategy);

    let mut stages: Vec<ComputeStage> = Vec::with_capacity(groups.len());
    let mut schedule_diagnostics: Vec<ScheduleDiagnostic> = Vec::new();

    for group in groups {
        let stage_index = stages.len();
        let topology = group_to_topology(prog, group, stage_index, &mut schedule_diagnostics);
        stages.push(ComputeStage {
            kernels: vec![topology],
        });
    }

    ScheduleSynthesisResult {
        schedule: ComputeSchedule { stages },
        fusion_diagnostics,
        schedule_diagnostics,
    }
}

// ---------------------------------------------------------------------------
// Translation helpers
// ---------------------------------------------------------------------------

/// Translate a [`FusionGroup`] into a [`KernelTopology`]. The
/// `Indirect` arm performs a structural producer lookup; misses fall
/// back to `Fused` and append a [`ScheduleDiagnostic`] to
/// `schedule_diagnostics`.
fn group_to_topology(
    prog: &CgProgram,
    group: FusionGroup,
    stage_index: usize,
    schedule_diagnostics: &mut Vec<ScheduleDiagnostic>,
) -> KernelTopology {
    let FusionGroup {
        ops,
        shape,
        fusibility_classification,
    } = group;

    match fusibility_classification {
        FusibilityClass::Fused => KernelTopology::Fused {
            ops,
            dispatch: shape,
        },
        FusibilityClass::Split => {
            // `Split` is always a singleton — see
            // `super::fusion::classify`'s rules. Encode that
            // invariant via expect rather than introducing a `_ =>`
            // fallthrough.
            let op = *ops.first().expect(
                "FusibilityClass::Split groups always carry exactly one op \
                 (see super::fusion::classify)",
            );
            debug_assert_eq!(
                ops.len(),
                1,
                "FusibilityClass::Split groups must be singletons"
            );
            KernelTopology::Split {
                op,
                dispatch: shape,
            }
        }
        FusibilityClass::Indirect => {
            // `Indirect` is reserved for `PerEvent` groups by
            // `super::fusion::classify`; extract the `source_ring` and
            // hunt for the matching `SeedIndirectArgs` producer.
            let source_ring = match shape {
                DispatchShape::PerEvent { source_ring } => source_ring,
                DispatchShape::PerAgent
                | DispatchShape::PerPair { .. }
                | DispatchShape::OneShot
                | DispatchShape::PerWord
                | DispatchShape::PerCell
                | DispatchShape::PerScanChunk => unreachable!(
                    "FusibilityClass::Indirect groups always carry \
                     DispatchShape::PerEvent (see super::fusion::classify)"
                ),
            };

            match find_indirect_producer(prog, source_ring) {
                Some(producer) => KernelTopology::Indirect {
                    producer,
                    consumers: ops,
                },
                None => {
                    schedule_diagnostics.push(ScheduleDiagnostic {
                        kind: ScheduleDiagnosticKind::IndirectProducerMissing {
                            source_ring,
                        },
                        stage_index,
                        message: format!(
                            "indirect group at stage {} (consumers={}) has no \
                             SeedIndirectArgs producer for ring=#{}; falling \
                             back to KernelTopology::Fused with direct dispatch",
                            stage_index,
                            ops.len(),
                            source_ring.0
                        ),
                    });
                    KernelTopology::Fused {
                        ops,
                        dispatch: shape,
                    }
                }
            }
        }
    }
}

/// Scan `prog.ops` for a [`PlumbingKind::SeedIndirectArgs { ring }`]
/// op whose `ring` matches `source_ring`. Returns the first match in
/// program order — `prog.ops` is the deterministic IR storage, so the
/// "first match" is stable across runs.
fn find_indirect_producer(prog: &CgProgram, source_ring: EventRingId) -> Option<OpId> {
    for (idx, op) in prog.ops.iter().enumerate() {
        if let ComputeOpKind::Plumbing { kind } = &op.kind {
            if let PlumbingKind::SeedIndirectArgs { ring } = kind {
                if *ring == source_ring {
                    return Some(OpId(idx as u32));
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, DataHandle, EventRingId, MaskId,
    };
    use crate::cg::expr::{CgExpr, LitValue};
    use crate::cg::op::{ComputeOpKind, OpId, PlumbingKind, Span};
    use crate::cg::program::{CgProgram, CgProgramBuilder};

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

    fn add_seed_indirect_args_op(
        builder: &mut CgProgramBuilder,
        ring: EventRingId,
    ) -> OpId {
        builder
            .add_op(
                ComputeOpKind::Plumbing {
                    kind: PlumbingKind::SeedIndirectArgs { ring },
                },
                DispatchShape::OneShot,
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

    // --- 1. Empty program → empty schedule ----------------------------

    #[test]
    fn empty_program_yields_empty_schedule_under_every_strategy() {
        let prog = CgProgram::new();
        for strategy in [
            ScheduleStrategy::Conservative,
            ScheduleStrategy::Default,
            ScheduleStrategy::Megakernel,
        ] {
            let result = synthesize_schedule(&prog, strategy);
            assert!(
                result.schedule.stages.is_empty(),
                "strategy {:?}: expected empty stages, got {:?}",
                strategy,
                result.schedule.stages
            );
            assert!(result.fusion_diagnostics.is_empty());
            assert!(result.schedule_diagnostics.is_empty());
            assert_eq!(result.schedule.stage_count(), 0);
            assert_eq!(result.schedule.kernel_count(), 0);
        }
    }

    // --- 2. Single op → one stage with Split ---------------------------

    #[test]
    fn single_op_default_yields_one_stage_with_split_kernel() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);

        assert_eq!(result.schedule.stages.len(), 1);
        let stage = &result.schedule.stages[0];
        assert_eq!(stage.kernels.len(), 1);
        match &stage.kernels[0] {
            KernelTopology::Split { op, dispatch } => {
                assert_eq!(*op, m0);
                assert_eq!(*dispatch, DispatchShape::PerAgent);
            }
            other => panic!("expected Split, got {:?}", other),
        }
        assert!(result.schedule_diagnostics.is_empty());
    }

    // --- 3. Multi-op fused → one stage with Fused kernel --------------

    #[test]
    fn multi_op_default_fuses_into_single_stage() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let m2 = add_mask_op(&mut b, MaskId(2), DispatchShape::PerAgent);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);

        assert_eq!(result.schedule.stages.len(), 1);
        match &result.schedule.stages[0].kernels[0] {
            KernelTopology::Fused { ops, dispatch } => {
                assert_eq!(*ops, vec![m0, m1, m2]);
                assert_eq!(*dispatch, DispatchShape::PerAgent);
            }
            other => panic!("expected Fused, got {:?}", other),
        }
    }

    // --- 4. Mix of fused and split → multiple stages, mixed kernels ----

    #[test]
    fn mixed_fused_and_split_yield_multiple_stages() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let upload = add_upload_sim_cfg_op(&mut b);
        let m3 = add_mask_op(&mut b, MaskId(3), DispatchShape::PerAgent);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);

        // Default fusion: [m0, m1] (Fused PerAgent) | [upload] (Split
        // OneShot) | [m3] (Split PerAgent).
        assert_eq!(result.schedule.stages.len(), 3);
        match &result.schedule.stages[0].kernels[0] {
            KernelTopology::Fused { ops, dispatch } => {
                assert_eq!(*ops, vec![m0, m1]);
                assert_eq!(*dispatch, DispatchShape::PerAgent);
            }
            other => panic!("expected Fused, got {:?}", other),
        }
        match &result.schedule.stages[1].kernels[0] {
            KernelTopology::Split { op, dispatch } => {
                assert_eq!(*op, upload);
                assert_eq!(*dispatch, DispatchShape::OneShot);
            }
            other => panic!("expected Split, got {:?}", other),
        }
        match &result.schedule.stages[2].kernels[0] {
            KernelTopology::Split { op, dispatch } => {
                assert_eq!(*op, m3);
                assert_eq!(*dispatch, DispatchShape::PerAgent);
            }
            other => panic!("expected Split, got {:?}", other),
        }
        assert!(result.schedule_diagnostics.is_empty());
    }

    // --- 5. Indirect with producer present → KernelTopology::Indirect --

    #[test]
    fn indirect_with_producer_present_yields_indirect_topology() {
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(7);
        let seed = add_seed_indirect_args_op(&mut b, ring);
        let d0 = add_drain_events_op(&mut b, ring);
        let d1 = add_drain_events_op(&mut b, ring);
        let prog = b.finish();

        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);

        // Stage 0: Split { seed (OneShot) }. Stage 1: Indirect {
        // producer=seed, consumers=[d0, d1] }.
        assert_eq!(result.schedule.stages.len(), 2);
        match &result.schedule.stages[0].kernels[0] {
            KernelTopology::Split { op, dispatch } => {
                assert_eq!(*op, seed);
                assert_eq!(*dispatch, DispatchShape::OneShot);
            }
            other => panic!("expected Split for seed op, got {:?}", other),
        }
        match &result.schedule.stages[1].kernels[0] {
            KernelTopology::Indirect {
                producer,
                consumers,
            } => {
                assert_eq!(*producer, seed);
                assert_eq!(*consumers, vec![d0, d1]);
            }
            other => panic!("expected Indirect for drain ops, got {:?}", other),
        }
        assert!(
            result.schedule_diagnostics.is_empty(),
            "expected no schedule diagnostics, got {:?}",
            result.schedule_diagnostics
        );
    }

    // --- 6. Indirect without producer → Fused fallback + diagnostic ----

    #[test]
    fn indirect_without_producer_falls_back_to_fused_with_diagnostic() {
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(11);
        // NOTE: no SeedIndirectArgs op for `ring` — only drain ops.
        let d0 = add_drain_events_op(&mut b, ring);
        let d1 = add_drain_events_op(&mut b, ring);
        let prog = b.finish();

        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);

        assert_eq!(result.schedule.stages.len(), 1);
        match &result.schedule.stages[0].kernels[0] {
            KernelTopology::Fused { ops, dispatch } => {
                assert_eq!(*ops, vec![d0, d1]);
                assert_eq!(*dispatch, DispatchShape::PerEvent { source_ring: ring });
            }
            other => panic!(
                "expected Fused fallback for missing producer, got {:?}",
                other
            ),
        }

        // One IndirectProducerMissing diagnostic, stage_index 0.
        assert_eq!(result.schedule_diagnostics.len(), 1);
        let diag = &result.schedule_diagnostics[0];
        assert_eq!(diag.stage_index, 0);
        match &diag.kind {
            ScheduleDiagnosticKind::IndirectProducerMissing { source_ring } => {
                assert_eq!(*source_ring, ring);
            }
        }
        assert!(
            diag.message.contains("ring=#11"),
            "diagnostic message must mention the ring: {:?}",
            diag.message
        );
    }

    // --- 6b. Singleton PerEvent without producer also falls back -------

    #[test]
    fn singleton_per_event_without_producer_falls_back_to_fused() {
        // A lone `DrainEvents` op classifies `Indirect` (singleton or
        // not). With no producer in the program, the synthesizer must
        // still fall back to `Fused` and emit the diagnostic — the
        // missing-producer path is not gated on group size.
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(2);
        let d = add_drain_events_op(&mut b, ring);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);

        assert_eq!(result.schedule.stages.len(), 1);
        match &result.schedule.stages[0].kernels[0] {
            KernelTopology::Fused { ops, dispatch } => {
                assert_eq!(*ops, vec![d]);
                assert_eq!(*dispatch, DispatchShape::PerEvent { source_ring: ring });
            }
            other => panic!("expected Fused fallback, got {:?}", other),
        }
        assert_eq!(result.schedule_diagnostics.len(), 1);
    }

    // --- 7. Megakernel strategy → one giant Fused kernel ---------------

    #[test]
    fn megakernel_strategy_yields_one_stage_with_fused_kernel() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let upload = add_upload_sim_cfg_op(&mut b);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Megakernel);

        assert_eq!(result.schedule.stages.len(), 1);
        match &result.schedule.stages[0].kernels[0] {
            KernelTopology::Fused { ops, dispatch } => {
                assert_eq!(*ops, vec![m0, m1, upload]);
                // Representative shape = first op's (PerAgent), per the
                // strategy convention.
                assert_eq!(*dispatch, DispatchShape::PerAgent);
            }
            other => panic!("expected Fused, got {:?}", other),
        }
        assert!(result.schedule_diagnostics.is_empty());
    }

    #[test]
    fn megakernel_with_single_op_yields_size_one_fused() {
        // The megakernel-singleton case the module-level Limitations
        // doc calls out: even one op produces a Fused topology.
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Megakernel);

        assert_eq!(result.schedule.stages.len(), 1);
        match &result.schedule.stages[0].kernels[0] {
            KernelTopology::Fused { ops, dispatch } => {
                assert_eq!(*ops, vec![m0]);
                assert_eq!(ops.len(), 1, "len-1 Fused must be supported");
                assert_eq!(*dispatch, DispatchShape::PerAgent);
            }
            other => panic!("expected Fused, got {:?}", other),
        }
    }

    // --- 8. Conservative strategy → N stages, each with Split ----------

    #[test]
    fn conservative_strategy_yields_one_stage_per_op() {
        let mut b = CgProgramBuilder::new();
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let m2 = add_mask_op(&mut b, MaskId(2), DispatchShape::PerAgent);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Conservative);

        assert_eq!(result.schedule.stages.len(), 3);
        for (i, expected) in [m0, m1, m2].iter().enumerate() {
            match &result.schedule.stages[i].kernels[0] {
                KernelTopology::Split { op, dispatch } => {
                    assert_eq!(op, expected);
                    assert_eq!(*dispatch, DispatchShape::PerAgent);
                }
                other => panic!("stage {i}: expected Split, got {other:?}"),
            }
        }
    }

    #[test]
    fn conservative_with_producer_and_consumers_yields_split_then_indirect() {
        // Conservative emits one singleton group per op, but the
        // singleton PerEvent ops still classify Indirect — so the
        // synthesizer still surfaces KernelTopology::Indirect for them
        // when a producer is present.
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(5);
        let seed = add_seed_indirect_args_op(&mut b, ring);
        let d0 = add_drain_events_op(&mut b, ring);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Conservative);

        assert_eq!(result.schedule.stages.len(), 2);
        match &result.schedule.stages[0].kernels[0] {
            KernelTopology::Split { op, .. } => assert_eq!(*op, seed),
            other => panic!("expected Split, got {:?}", other),
        }
        match &result.schedule.stages[1].kernels[0] {
            KernelTopology::Indirect {
                producer,
                consumers,
            } => {
                assert_eq!(*producer, seed);
                assert_eq!(*consumers, vec![d0]);
            }
            other => panic!("expected Indirect, got {:?}", other),
        }
        assert!(result.schedule_diagnostics.is_empty());
    }

    // --- 9. Determinism: two runs produce identical output -------------

    #[test]
    fn two_runs_produce_byte_identical_output() {
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(3);
        let m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let _m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let _seed = add_seed_indirect_args_op(&mut b, ring);
        let _d0 = add_drain_events_op(&mut b, ring);
        let _d1 = add_drain_events_op(&mut b, ring);
        let mut prog = b.finish();
        // Inject a single write to keep dependency edges live.
        prog.ops[m0.0 as usize].record_write(hp_handle());

        for strategy in [
            ScheduleStrategy::Conservative,
            ScheduleStrategy::Default,
            ScheduleStrategy::Megakernel,
        ] {
            let first = synthesize_schedule(&prog, strategy);
            let second = synthesize_schedule(&prog, strategy);
            assert_eq!(
                first, second,
                "strategy {:?}: two runs differ",
                strategy
            );
        }
    }

    // --- 10. Snapshot test pinning Display on a non-trivial schedule ---

    #[test]
    fn schedule_display_snapshot() {
        // Fixture: two PerAgent ops fuse, an upload (OneShot) splits,
        // a SeedIndirectArgs producer + two drain consumers form an
        // Indirect topology. Default strategy. The Display must be
        // stable across runs.
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(4);
        let _m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let _m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let _upload = add_upload_sim_cfg_op(&mut b);
        let _seed = add_seed_indirect_args_op(&mut b, ring);
        let _d0 = add_drain_events_op(&mut b, ring);
        let _d1 = add_drain_events_op(&mut b, ring);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);
        let rendered = format!("{}", result.schedule);

        // Pinned snapshot — schedule structure under Default fusion:
        //   stage 0: fused(m0, m1)        PerAgent
        //   stage 1: fused(upload, seed)  OneShot   ← upload + seed
        //                                            share OneShot,
        //                                            no WAW conflict,
        //                                            so they fuse
        //   stage 2: indirect(producer=seed, consumers=[d0, d1])
        //
        // Note: even when `seed` co-fuses into stage 1 with `upload`,
        // the producer-lookup in stage 2 still locates op#3 (the
        // SeedIndirectArgs op) by scanning `prog.ops` — fusion does
        // not erase the underlying program structure.
        let expected = "schedule([\
            stage([fused(ops=[op#0, op#1], dispatch=per_agent)]), \
            stage([fused(ops=[op#2, op#3], dispatch=one_shot)]), \
            stage([indirect(producer=op#3, consumers=[op#4, op#5])])\
        ])";
        assert_eq!(rendered, expected, "schedule Display drift");
    }

    // --- bonus: KernelTopology Display + label + ops ------------------

    #[test]
    fn kernel_topology_display_and_label_render() {
        let fused = KernelTopology::Fused {
            ops: vec![OpId(0), OpId(1)],
            dispatch: DispatchShape::PerAgent,
        };
        let split = KernelTopology::Split {
            op: OpId(7),
            dispatch: DispatchShape::OneShot,
        };
        let indirect = KernelTopology::Indirect {
            producer: OpId(3),
            consumers: vec![OpId(4), OpId(5)],
        };

        assert_eq!(format!("{}", fused), "fused(ops=[op#0, op#1], dispatch=per_agent)");
        assert_eq!(format!("{}", split), "split(op=op#7, dispatch=one_shot)");
        assert_eq!(
            format!("{}", indirect),
            "indirect(producer=op#3, consumers=[op#4, op#5])"
        );
        assert_eq!(fused.label(), "fused");
        assert_eq!(split.label(), "split");
        assert_eq!(indirect.label(), "indirect");

        assert_eq!(fused.ops(), vec![OpId(0), OpId(1)]);
        assert_eq!(split.ops(), vec![OpId(7)]);
        assert_eq!(indirect.ops(), vec![OpId(3), OpId(4), OpId(5)]);
    }

    // --- bonus: ScheduleDiagnosticKind Display ------------------------

    #[test]
    fn schedule_diagnostic_kind_display_renders() {
        let kind = ScheduleDiagnosticKind::IndirectProducerMissing {
            source_ring: EventRingId(9),
        };
        assert_eq!(kind.label(), "indirect_producer_missing");
        assert_eq!(format!("{}", kind), "indirect_producer_missing(ring=#9)");
    }

    // --- bonus: stage_count and kernel_count helpers ------------------

    #[test]
    fn schedule_count_helpers_match_stage_population() {
        let mut b = CgProgramBuilder::new();
        let _m0 = add_mask_op(&mut b, MaskId(0), DispatchShape::PerAgent);
        let _m1 = add_mask_op(&mut b, MaskId(1), DispatchShape::PerAgent);
        let _upload = add_upload_sim_cfg_op(&mut b);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);
        assert_eq!(result.schedule.stage_count(), 2);
        // Today: kernel_count == stage_count.
        assert_eq!(result.schedule.kernel_count(), 2);
    }

    // --- bonus: Indirect producer lookup picks first match ------------

    #[test]
    fn indirect_producer_lookup_picks_first_match_in_program_order() {
        // Two SeedIndirectArgs ops, both for the same ring. The lookup
        // returns the FIRST one — program order is the deterministic
        // tiebreaker.
        let mut b = CgProgramBuilder::new();
        let ring = EventRingId(8);
        let seed_a = add_seed_indirect_args_op(&mut b, ring);
        let _seed_b = add_seed_indirect_args_op(&mut b, ring);
        let _d0 = add_drain_events_op(&mut b, ring);
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);

        // Find the Indirect stage and assert producer = first seed.
        let indirect_stage = result
            .schedule
            .stages
            .iter()
            .find_map(|s| match &s.kernels[0] {
                KernelTopology::Indirect {
                    producer,
                    consumers,
                } => Some((*producer, consumers.clone())),
                _ => None,
            })
            .expect("expected an Indirect stage");
        assert_eq!(indirect_stage.0, seed_a);
    }

    // --- bonus: Indirect producer for a different ring → fallback -----

    #[test]
    fn indirect_producer_for_unrelated_ring_does_not_match_consumers() {
        // Producer for ring #1 + consumers for ring #2 → no match,
        // fallback to Fused.
        let mut b = CgProgramBuilder::new();
        let _seed = add_seed_indirect_args_op(&mut b, EventRingId(1));
        let _d = add_drain_events_op(&mut b, EventRingId(2));
        let prog = b.finish();
        let result = synthesize_schedule(&prog, ScheduleStrategy::Default);

        // Stage 0: Split (seed). Stage 1: Fused fallback for ring #2
        // consumer.
        assert_eq!(result.schedule.stages.len(), 2);
        match &result.schedule.stages[1].kernels[0] {
            KernelTopology::Fused { ops, dispatch } => {
                assert_eq!(ops.len(), 1);
                assert!(matches!(
                    dispatch,
                    DispatchShape::PerEvent {
                        source_ring: EventRingId(2)
                    }
                ));
            }
            other => panic!("expected Fused fallback, got {:?}", other),
        }
        assert_eq!(result.schedule_diagnostics.len(), 1);
        match &result.schedule_diagnostics[0].kind {
            ScheduleDiagnosticKind::IndirectProducerMissing { source_ring } => {
                assert_eq!(*source_ring, EventRingId(2));
            }
        }
    }
}
