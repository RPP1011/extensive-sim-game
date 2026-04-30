//! Compute-Graph IR → WGSL emission.
//!
//! Phase 4 of the DSL compute-graph plan
//! (`docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Tasks 4.1–4.3) lowers a [`crate::cg::CgProgram`] to per-backend
//! source. This module is the WGSL side.
//!
//! # Layering
//!
//! Task 4.1 (this commit) — *expression and statement walks*.
//!
//! - [`wgsl_body::EmitCtx`] — minimal context (program reference +
//!   handle naming strategy) carried through the inner walks.
//! - [`wgsl_body::lower_cg_expr_to_wgsl`] — single [`crate::cg::CgExpr`]
//!   → WGSL fragment (e.g. `(self_hp + 1.0)`). No new bindings, no
//!   kernel boilerplate; pure tree walk.
//! - [`wgsl_body::lower_cg_stmt_to_wgsl`] — single [`crate::cg::CgStmt`]
//!   → WGSL fragment (e.g. `self_hp = (self_hp + 1.0);`).
//! - [`wgsl_body::lower_cg_stmt_list_to_wgsl`] — list-of-statements
//!   joiner.
//!
//! Task 4.2 will compose these per-expression strings into kernel
//! bodies. Task 4.3 will assemble full kernel modules.
//!
//! # Why string emission here
//!
//! The plan emphasises "no string concatenation builds new bindings"
//! (binding-graph IR + lowerings is the canonical surface). Task 4.1
//! is intentionally outside that rule: it produces inner expression
//! and statement source — fragments like `(hp + 1.0) * 2.0` —
//! which are inherently strings. The bindings, BGL slot resolution,
//! and kernel boilerplate stay structural; only the WGSL leaves
//! produced here are textual.
//!
//! # Determinism
//!
//! Outputs are byte-identical across runs: every input is structural
//! (typed enums, arena ids), `f32` formatting routes through
//! `{value:?}` (round-trip-safe debug format), and no `HashMap`
//! iteration appears in any code path. The
//! `wgsl_emit_is_deterministic` test pins this contract.

pub mod wgsl_body;

pub use wgsl_body::{
    lower_cg_expr_to_wgsl, lower_cg_stmt_list_to_wgsl, lower_cg_stmt_to_wgsl, EmitCtx, EmitError,
    HandleNamingStrategy,
};
