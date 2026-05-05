//! World Sim DSL compiler — Compute-Graph IR + WGSL emission.
//!
//! Frontend (parser, AST, IR, name resolution) lives in the `dsl_ast`
//! crate and is re-exported here for backward compatibility. This crate
//! owns the Compute-Graph (CG) IR + lowering + WGSL emission. Emitted
//! files land on disk via the xtask `compile-dsl` subcommand.
//!
//! **Phase 7 wolf-sim wipe (2026-05-02):** the legacy emitter modules
//! (`emit_mask`, `emit_view*`, `emit_scoring*`, `emit_physics*`,
//! `emit_movement_kernel`, `emit_step`, `emit_pick_ability_kernel`,
//! `emit_megakernel`, `emit_spatial_kernel`, etc.) and the legacy
//! `EmittedArtifacts` API have been removed. The compute-graph path
//! (`cg::*`) is the sole supported emission pipeline. Build a new DSL
//! fixture under `assets/sim/` to exercise it.

// Frontend re-exports
pub use dsl_ast::ast;
pub use dsl_ast::error;
pub use dsl_ast::ir;
pub use dsl_ast::parser;
pub use dsl_ast::resolve;
pub use dsl_ast::resolve_error;
pub use dsl_ast::tokens;

// Compute-Graph IR — the canonical emission path.
pub mod cg;

// Wave 1.6 — `.ability` AST -> engine-runtime `AbilityProgram` lowering.
// See `ability_lower.rs` for scope (only the 8 currently-implemented
// `EffectOp` variants and the 5 Wave-1 header keys are wired today).
pub mod ability_lower;

// Shared kernel-emit infrastructure (reused by `cg::emit::*`).
pub mod kernel_binding_ir;
pub mod kernel_lowerings;

// Schema-hash helper — used by `crates/engine/.schema_hash` regeneration
// (P2: schema-hash bumps on layout change).
pub mod schema_hash;

// Top-level symbol re-exports
pub use ast::{Decl, Program, Span, Spanned};
pub use error::ParseError;
pub use ir::Compilation;
pub use resolve_error::ResolveError;
pub use dsl_ast::{compile, compile_ast, parse, CompileError};

// Helper used by emit modules + tests for case conversion. Kept here
// because it has no natural home in `cg/*` and is consumed by the
// schema_hash module.
pub fn snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}
