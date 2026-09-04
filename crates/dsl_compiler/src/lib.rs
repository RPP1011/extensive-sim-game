//! World Sim DSL compiler — Compute-Graph IR + WGSL emission.
//!
//! Frontend (parser, AST, IR, name resolution) lives in the `dsl_ast`
//! crate and is re-exported here for backward compatibility. This crate
//! owns the Compute-Graph (CG) IR + lowering + WGSL emission. Emitted
//! files land on disk via the xtask `compile-dsl` subcommand.
//!
//! The compute-graph path (`cg::*`) is the sole supported emission
//! pipeline. Legacy per-declaration-kind emitters (`emit_mask`,
//! `emit_view*`, `emit_scoring*`, `emit_physics*`, `emit_movement_kernel`,
//! `emit_step`, `emit_pick_ability_kernel`, `emit_megakernel`,
//! `emit_spatial_kernel`, etc.) and the legacy `EmittedArtifacts` API do
//! not exist in this codebase. Build a new DSL fixture under
//! `assets/sim/` to exercise it.

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

// Re-export the CG lower module at the crate root so that tests and
// downstream crates can write `dsl_compiler::lower::lower_terrain`.
pub use cg::lower as lower;

// Re-export the CG emit module at the crate root so that tests and
// downstream crates can write `dsl_compiler::emit::emit_terrain`.
pub use cg::emit as emit;

// Re-export terrain lowering error at the crate root for test ergonomics.
pub use cg::lower::LowerError;

// Multi-file import resolver — `ImportError` enum + `resolve_import_path`.
// See `docs/superpowers/specs/2026-05-17-terrain-dsl-multifile-design.md`.
pub mod imports;
pub use imports::{parse_with_imports, ImportError};

// Wave 1.6 — `.ability` AST -> engine-runtime `AbilityProgram` lowering.
// See `ability_lower.rs` for scope (only the 8 currently-implemented
// `EffectOp` variants and the 5 Wave-1 header keys are wired today).
pub mod ability_lower;

// Wave 1.7 — assemble lowered `AbilityProgram`s into a frozen
// `AbilityRegistry` and resolve `cast <Name>` references across files.
// Owns duplicate-name + unresolved-cast + cast-cycle diagnostics. See
// `ability_registry.rs` for the scope statement.
pub mod ability_registry;

// Shared kernel-emit infrastructure (reused by `cg::emit::*`).
pub mod kernel_binding_ir;
pub mod kernel_lowerings;

// Schema-hash helper — used by `crates/engine/.schema_hash` regeneration
// (P2: schema-hash bumps on layout change).
pub mod schema_hash;

// Shared seq packing for deterministic event ordering (P11).
pub mod seq;

// Plan E-A1 — shared build-script helper. Each `crates/*_runtime/build.rs`
// becomes a 1-line stub: `dsl_compiler::build_helper::emit("<fixture>")`.
// Replaces ~110 LOC of identical pipeline boilerplate × 60 fixtures.
pub mod build_helper;

// Gap plague_city#P-A — custom per-agent SoA field registry. Lets
// .sim files declare new `field <name>: <ty>` columns without
// touching the closed `AgentFieldId` enum. See module docs.
pub mod custom_agent_fields;

// CPU reference for the WGSL apply_ability dispatcher's chronicle
// output — establishes the contract that the GPU dispatcher's
// chronicle records mirror, ahead of the runtime crate that #133
// will use for full CPU↔GPU parity.
pub mod cpu_chronicle_reference;

// Wave 3 ToM Phase 3.8 — generated WGSL for the per-tick BeliefState
// decay sweep. Replaces the hand-written `DECAY_WGSL` constant the
// `tom_probe_runtime` crate used to carry. The runtime now obtains
// the decay kernel WGSL by calling [`belief_decay_wgsl::decay_kernel_wgsl`]
// at construction time — the WGSL string lives with the compiler, not
// the runtime, per the user's "no hand-written WGSL in runtime crates"
// constraint. There is no DSL surface for the decay rule today
// (`for_each_agent` body shape isn't a DSL primitive); the kernel is
// authored as a function in the compiler instead of as a literal
// const string in the runtime.
pub mod belief_decay_wgsl;

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
