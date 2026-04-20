//! World Sim DSL compiler — parser + AST + IR + name resolution + emission.
//!
//! - Milestone 1a: AST → typed IR with cross-references resolved.
//! - Milestone 2: IR → Rust + Python + schema hash for `event` declarations.
//!
//! No engine integration; emitted files land on disk via the xtask
//! `compile-dsl` subcommand.

pub mod ast;
pub mod emit_python;
pub mod emit_rust;
pub mod error;
pub mod ir;
pub mod parser;
pub mod resolve;
pub mod resolve_error;
pub mod schema_hash;
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

// ---------------------------------------------------------------------------
// Emission bundle
// ---------------------------------------------------------------------------

/// Rust + Python source strings, ready to write to disk. Filenames are
/// snake-cased event names (minus extension); the caller decides what
/// extension and parent directory to use.
#[derive(Debug, Clone)]
pub struct EmittedArtifacts {
    /// Content of `crates/engine_rules/src/events/mod.rs`.
    pub rust_events_mod: String,
    /// `(filename_without_dir, content)` pairs, one per event.
    pub rust_event_structs: Vec<(String, String)>,
    /// Content of `generated/python/events/__init__.py`.
    pub python_events_init: String,
    /// `(filename_without_dir, content)` pairs, one per event.
    pub python_event_modules: Vec<(String, String)>,
    /// Raw 32-byte schema hash covering the event taxonomy.
    pub event_hash: [u8; 32],
    /// Content of `crates/engine_rules/src/schema.rs`.
    pub schema_rs: String,
}

/// Emit the full artefact bundle for a resolved `Compilation`. At milestone
/// 2 only `events` are covered; other declaration kinds produce no output.
pub fn emit(comp: &Compilation) -> EmittedArtifacts {
    emit_with_source(comp, None)
}

/// Like [`emit`], but stamp `source_file` into per-event headers. Used by
/// the xtask when every event came from a single `.sim` file.
pub fn emit_with_source(comp: &Compilation, source_file: Option<&str>) -> EmittedArtifacts {
    let mut rust_event_structs = Vec::with_capacity(comp.events.len());
    let mut python_event_modules = Vec::with_capacity(comp.events.len());
    for event in &comp.events {
        let stem = snake_case(&event.name);
        let rs = emit_rust::emit_event(event, source_file);
        let py = emit_python::emit_event_dataclass(event, source_file);
        rust_event_structs.push((format!("{stem}.rs"), rs));
        python_event_modules.push((format!("{stem}.py"), py));
    }
    rust_event_structs.sort_by(|a, b| a.0.cmp(&b.0));
    python_event_modules.sort_by(|a, b| a.0.cmp(&b.0));

    let event_hash = schema_hash::event_hash(&comp.events);
    EmittedArtifacts {
        rust_events_mod: emit_rust::emit_events_mod(&comp.events),
        rust_event_structs,
        python_events_init: emit_python::emit_events_init(&comp.events),
        python_event_modules,
        event_hash,
        schema_rs: schema_hash::emit_schema_rs(&event_hash),
    }
}

fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    let mut prev_upper = false;
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 && !prev_upper {
                out.push('_');
            }
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            prev_upper = true;
        } else {
            out.push(ch);
            prev_upper = false;
        }
    }
    out
}
