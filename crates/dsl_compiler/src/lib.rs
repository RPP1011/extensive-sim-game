//! World Sim DSL compiler — parser + AST + IR + name resolution + emission.
//!
//! - Milestone 1a: AST → typed IR with cross-references resolved.
//! - Milestone 2: IR → Rust + Python + schema hash for `event` declarations.
//! - Milestone 3: IR → Rust `impl CascadeHandler` blocks for `physics`
//!   declarations + `rules_hash` sub-hash + combined-hash bookkeeping.
//!
//! No engine integration; emitted files land on disk via the xtask
//! `compile-dsl` subcommand.

pub mod ast;
pub mod emit_physics;
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
/// snake-cased declaration names (minus extension); the caller decides what
/// extension and parent directory to use.
#[derive(Debug, Clone)]
pub struct EmittedArtifacts {
    /// Content of the events-mod file (`crates/engine_rules/src/events/mod.rs`).
    pub rust_events_mod: String,
    /// `(filename_without_dir, content)` pairs, one per event.
    pub rust_event_structs: Vec<(String, String)>,
    /// Physics-rule modules. `(filename_without_dir, content)` pairs, one
    /// per `physics` declaration. Path target is `crates/engine/src/generated/physics/`
    /// — see `docs/game/feature_flow.md` for the rationale (engine_rules
    /// can't import `engine::cascade::*` without breaking the dep graph).
    pub rust_physics_modules: Vec<(String, String)>,
    /// Content of `crates/engine/src/generated/physics/mod.rs` — the
    /// aggregator that exposes a single `pub fn register(registry)` for the
    /// engine's builtin registration to call.
    pub rust_physics_mod: String,
    /// Content of `generated/python/events/__init__.py`.
    pub python_events_init: String,
    /// `(filename_without_dir, content)` pairs, one per event.
    pub python_event_modules: Vec<(String, String)>,
    /// Raw 32-byte schema hash covering the event taxonomy.
    pub event_hash: [u8; 32],
    /// Raw 32-byte schema hash covering the physics-rule subset of the rules
    /// taxonomy. Will grow to also cover masks (milestone 4) and verbs
    /// (milestone 7) without changing the API surface here.
    pub rules_hash: [u8; 32],
    /// Combined hash per `docs/compiler/spec.md` §2 — `sha256(state_hash ||
    /// event_hash || rules_hash || scoring_hash)`. State and scoring are
    /// all-zero placeholders until their milestones land.
    pub combined_hash: [u8; 32],
    /// Content of `crates/engine_rules/src/schema.rs`.
    pub schema_rs: String,
}

/// Emit the full artefact bundle for a resolved `Compilation`. Covers
/// events (milestone 2) and physics rules (milestone 3); other declaration
/// kinds produce no output until their milestones land.
pub fn emit(comp: &Compilation) -> EmittedArtifacts {
    emit_with_source(comp, None)
}

/// Like [`emit`], but stamp a single `source_file` into every per-decl
/// header — used when the compilation is sourced from one `.sim` file.
pub fn emit_with_source(comp: &Compilation, source_file: Option<&str>) -> EmittedArtifacts {
    emit_with_per_kind_sources(
        comp,
        EmissionSources { events: source_file, physics: source_file },
    )
}

/// Per-decl-kind source paths. Each is the path to stamp into that kind's
/// emitted-file headers. `None` for a kind means "emit a generic header"
/// (no path stamp). The xtask uses this when events live in `events.sim`
/// and physics in `physics.sim` — each kind's emission gets the right
/// source-file annotation.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmissionSources<'a> {
    pub events: Option<&'a str>,
    pub physics: Option<&'a str>,
}

/// Like [`emit`], but stamp the appropriate source file into each kind's
/// per-decl headers. Used by the xtask which tracks per-kind source files.
pub fn emit_with_per_kind_sources(
    comp: &Compilation,
    sources: EmissionSources<'_>,
) -> EmittedArtifacts {
    let mut rust_event_structs = Vec::with_capacity(comp.events.len());
    let mut python_event_modules = Vec::with_capacity(comp.events.len());
    for event in &comp.events {
        let stem = snake_case(&event.name);
        let rs = emit_rust::emit_event(event, sources.events);
        let py = emit_python::emit_event_dataclass(event, sources.events);
        rust_event_structs.push((format!("{stem}.rs"), rs));
        python_event_modules.push((format!("{stem}.py"), py));
    }
    rust_event_structs.sort_by(|a, b| a.0.cmp(&b.0));
    python_event_modules.sort_by(|a, b| a.0.cmp(&b.0));

    // Physics emission is fallible (unsupported IR shape, unresolved event
    // ref). On error we panic with a diagnostic — the xtask catches user
    // errors via `compile()` returning IR errors, so a resolved IR shape we
    // can't emit is a compiler bug.
    let mut rust_physics_modules: Vec<(String, String)> = Vec::with_capacity(comp.physics.len());
    for physics in &comp.physics {
        let stem = snake_case(&physics.name);
        match emit_physics::emit_physics(physics, sources.physics) {
            Ok(rs) => rust_physics_modules.push((format!("{stem}.rs"), rs)),
            Err(e) => {
                panic!(
                    "physics emission failed for `{}`: {e}",
                    physics.name
                );
            }
        }
    }
    rust_physics_modules.sort_by(|a, b| a.0.cmp(&b.0));
    let rust_physics_mod = emit_physics::emit_physics_mod(&comp.physics);

    let event_hash = schema_hash::event_hash(&comp.events);
    let rules_hash = schema_hash::rules_hash(&comp.physics);
    let state_hash = [0u8; 32];
    let scoring_hash = [0u8; 32];
    let combined_hash = schema_hash::combined_hash(&state_hash, &event_hash, &rules_hash, &scoring_hash);
    EmittedArtifacts {
        rust_events_mod: emit_rust::emit_events_mod(&comp.events),
        rust_event_structs,
        rust_physics_modules,
        rust_physics_mod,
        python_events_init: emit_python::emit_events_init(&comp.events),
        python_event_modules,
        event_hash,
        rules_hash,
        combined_hash,
        schema_rs: schema_hash::emit_schema_rs(&state_hash, &event_hash, &rules_hash, &scoring_hash),
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
