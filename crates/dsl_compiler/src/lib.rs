//! World Sim DSL compiler — parser + AST + IR + name resolution.
//!
//! Milestone 1a scope: AST → typed IR with cross-references resolved.
//! No validation, no emission, no desugaring.

pub mod ast;
pub mod error;
pub mod ir;
pub mod parser;
pub mod resolve;
pub mod resolve_error;
pub mod tokens;

pub use ast::{Decl, Program, Span, Spanned};
pub use error::ParseError;
pub use ir::Compilation;
pub use resolve_error::ResolveError;

/// Parse a DSL source string into a `Program` AST.
pub fn parse(source: &str) -> Result<Program, ParseError> {
    parser::parse_program(source)
}

/// Resolve a parsed `Program` into a typed IR `Compilation`.
pub fn compile_ast(program: Program) -> Result<Compilation, ResolveError> {
    resolve::resolve(program)
}

/// Parse + resolve in one step.
pub fn compile(source: &str) -> Result<Compilation, CompileError> {
    let program = parse(source).map_err(CompileError::Parse)?;
    compile_ast(program).map_err(CompileError::Resolve)
}

#[derive(Debug)]
pub enum CompileError {
    Parse(ParseError),
    Resolve(ResolveError),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Parse(e) => write!(f, "{e}"),
            CompileError::Resolve(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CompileError {}
