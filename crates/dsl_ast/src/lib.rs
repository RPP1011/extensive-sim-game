//! World Sim DSL frontend: parser, AST, typed IR, and name resolution.
//!
//! Extracted from `dsl_compiler` so both the emitter crate and the engine
//! interpreter can share one parse + resolve pipeline. See
//! `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1.

pub mod ability_parser;
pub mod ast;
pub mod tokens;
pub mod error;
pub mod resolve_error;
pub mod ir;
pub mod parser;
pub mod resolve;
pub mod eval;

pub use ability_parser::parse_ability_file;
pub use ast::{
    AbilityDecl, AbilityFile, AbilityHeader, CostAmount, CostResource, CostSpec, Decl,
    DeliverBlock, Duration, EffectArea, EffectArg, EffectChance, EffectCondition, EffectDuration,
    EffectLifetime, EffectScaling, EffectStmt, EffectTag, HintName, MorphBlock, PassiveDecl,
    PassiveHeader, Program, RecastValue, Span, Spanned, StackingMode, StructureDecl, TargetMode,
    TemplateArg, TemplateDecl, TemplateInstantiation, TemplateParam, TemplateParamTy,
};
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

impl From<ParseError> for CompileError {
    fn from(e: ParseError) -> Self {
        CompileError::Parse(e)
    }
}

impl From<ResolveError> for CompileError {
    fn from(e: ResolveError) -> Self {
        CompileError::Resolve(e)
    }
}