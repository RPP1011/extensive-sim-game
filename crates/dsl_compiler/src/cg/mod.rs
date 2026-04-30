//! Compute-Graph IR (CG) — the multi-layer transpiler pipeline that
//! sits between resolved DSL IR and the per-backend emitters. See
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md` for the
//! full architecture.
//!
//! Phase 1 (this module's home) defines the shared data model that
//! every later layer (HIR/MIR/LIR) and pass walks. Subsequent tasks
//! land here:
//!
//! - Task 1.1 (this file's submodule): [`data_handle`] — typed
//!   references to simulation state, the basic vocabulary every
//!   compute op uses for reads and writes.
//! - Task 1.2: `expr` — the compute-graph expression tree.
//! - Task 1.3: `op` — `ComputeOp` + `ComputeOpKind`.
//! - Task 1.4: `dispatch` — `DispatchShape` + helpers.
//! - Task 1.5: `program` — top-level `CgProgram`.
//! - Task 1.6: `well_formed` — invariant-checking pass.
//!
//! Each submodule is added in its own task; only Task 1.1 ships in
//! this commit.

pub mod data_handle;
pub mod dispatch;
pub mod emit;
pub mod expr;
pub mod lower;
pub mod op;
pub mod program;
pub mod schedule;
pub mod stmt;
pub mod well_formed;

pub use data_handle::*;
pub use dispatch::*;
pub use emit::{
    lower_cg_expr_to_wgsl, lower_cg_stmt_list_to_wgsl, lower_cg_stmt_to_wgsl, EmitCtx, EmitError,
    HandleNamingStrategy,
};
pub use expr::*;
pub use lower::{lower_expr, LoweringCtx, LoweringError};
pub use op::*;
pub use program::*;
pub use schedule::{dependency_graph, topological_sort, CycleError, DepGraph};
pub use stmt::*;
pub use well_formed::*;
