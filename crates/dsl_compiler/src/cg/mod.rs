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
pub mod expr;
pub mod op;
pub mod stmt;

pub use data_handle::*;
pub use dispatch::*;
pub use expr::*;
pub use op::*;
pub use stmt::*;
