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
//! - Task 2.2: [`mask`] — `MaskIR → ComputeOp::MaskPredicate`.
//! - Task 2.3: `view` — `ViewIR → ComputeOp::ViewFold | …`.
//! - Task 2.4: `physics` — physics rules → `ComputeOp::PhysicsRule`.
//! - Task 2.5: `scoring` — scoring rows → `ComputeOp::ScoringArgmax`.
//! - Task 2.6: `spatial` — spatial queries.
//! - Task 2.7: `plumbing` — driver glue (`lower_compilation`).
//!
//! Submodules wire in incrementally as each task lands.
//!
//! # Typed-error surface
//!
//! Every pass returns the unified [`LoweringError`] declared in
//! [`error`]. Per-pass variants are prefixed with the construct name
//! (`Mask*`, `View*`, …) so the enum stays readable as it grows. See
//! `error.rs`'s module docs for the convention.

pub mod error;
pub mod expr;
pub mod mask;

pub use error::LoweringError;
pub use expr::{lower_expr, LoweringCtx};
pub use mask::lower_mask;
