//! Phase-3 schedule synthesis — turns a `CgProgram` into an executable
//! sequence of dispatches.
//!
//! Phase 1 defined the IR shape (`ComputeOp`, `DataHandle`, …). Phase 2
//! lowered every DSL construct to ops with auto-derived reads + writes
//! and driver-injected source-ring reads / Emit destination-ring
//! writes. Phase 3 walks those ops, decides what runs in which order
//! (and eventually how to fuse adjacent ops into megakernels), and
//! hands the result to per-backend emitters.
//!
//! Task 3.1 ships [`topology`]: a read/write [`topology::DepGraph`]
//! plus a Kahn's [`topology::topological_sort`]. Task 3.2 ships
//! [`fusion`]: walking the topological order to group consecutive ops
//! that share a dispatch shape and don't write-conflict, plus a
//! decision-diagnostic stream. Task 3.3 ships [`strategy`]: a typed
//! selector ([`strategy::ScheduleStrategy`]) over Conservative /
//! Default / Megakernel fusion policies, plus the strategy-aware
//! [`strategy::fusion_candidates_with_strategy`] /
//! [`strategy::fusion_decisions_with_strategy`] entry points that wrap
//! Task 3.2's analysis. Subsequent Phase-3 tasks (3.4 backend
//! emission) accrete here.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Phase 3, for the design rationale.

pub mod fusion;
pub mod strategy;
pub mod topology;

pub use fusion::{
    dispatch_shape_key, fusion_candidates, fusion_decisions, DispatchShapeKey, FusibilityClass,
    FusionDiagnostic, FusionDiagnosticKind, FusionGroup,
};
pub use strategy::{
    fusion_candidates_with_strategy, fusion_decisions_with_strategy, ScheduleStrategy,
};
pub use topology::{dependency_graph, topological_sort, CycleError, DepGraph};
