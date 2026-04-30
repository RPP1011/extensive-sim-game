//! AST → CG lowering passes.
//!
//! Phase 2 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`). Each
//! submodule consumes resolved DSL IR (`dsl_ast::ir`) and pushes nodes
//! into a [`crate::cg::CgProgramBuilder`].
//!
//! Submodules land one-per-task:
//!
//! - Task 2.1: [`expr`] — expression-level lowering used by every
//!   later pass that needs to lower an `IrExprNode`.
//! - Task 2.2 (this commit): [`mask`] — `MaskIR → ComputeOp::MaskPredicate`.
//! - Task 2.3: `view` — `ViewIR → ComputeOp::ViewFold | …`.
//! - Task 2.4: `physics` — physics rules → `ComputeOp::PhysicsRule`.
//! - Task 2.5: `scoring` — scoring rows → `ComputeOp::ScoringArgmax`.
//! - Task 2.6: `spatial` — spatial queries.
//! - Task 2.7: `plumbing` — driver glue (`lower_compilation`).
//!
//! Submodules wire in incrementally as each task lands.

pub mod expr;
pub mod mask;

pub use expr::{lower_expr, LoweringCtx, LoweringError};
pub use mask::{lower_mask, MaskLoweringError};
