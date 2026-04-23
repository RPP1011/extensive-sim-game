//! World Sim DSL frontend: parser, AST, typed IR, and name resolution.
//!
//! Extracted from `dsl_compiler` so both the emitter crate and the engine
//! interpreter can share one parse + resolve pipeline. See
//! `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1.

pub mod ast;
pub mod tokens;
pub mod error;
pub mod resolve_error;