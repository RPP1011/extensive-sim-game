//! Compiler-emitted modules — DSL source → Rust output.
//!
//! Submodules here are written by `dsl_compiler` from `assets/sim/*.sim`.
//! Edit the `.sim` source and rerun `cargo run --bin xtask -- compile-dsl`.
//! Reviewers should diff both the DSL source and the regenerated Rust; if
//! the emitted Rust looks wrong the fix is in `dsl_compiler/src/emit_*.rs`,
//! not here.
//!
//! Why this lives in `engine` rather than `engine_rules`: physics handlers
//! reference `engine::cascade::CascadeHandler`, `engine::state::SimState`,
//! and `engine::event::EventRing`, which would require `engine_rules` to
//! depend on `engine` — inverting the existing dep direction. See
//! `docs/game/feature_flow.md` for the full rationale.
//!
//! Lint suppression: the emitter wraps every binary subexpression in
//! parens for parser-safety. Rust's `unused_parens` lint complains about
//! the resulting redundant grouping. Suppressing here keeps the emitter
//! mechanical (no precedence tracking) at the cost of slightly noisier
//! diff review — a deliberate tradeoff for milestone 3.

#![allow(unused_parens)]

pub mod mask;
pub mod physics;
