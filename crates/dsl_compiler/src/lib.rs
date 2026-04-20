//! World Sim DSL compiler — parser + AST.
//!
//! Milestone 0 scope: turn DSL source text into a typed AST with spans.
//! No lowering, no emission, no IR passes.

pub mod ast;
pub mod error;
pub mod parser;
pub mod tokens;

pub use ast::{Decl, Program, Span, Spanned};
pub use error::ParseError;

/// Parse a DSL source string into a `Program` AST.
pub fn parse(source: &str) -> Result<Program, ParseError> {
    parser::parse_program(source)
}
