//! Filesystem-aware import resolver + merger for `.sim` files.
//! See `docs/superpowers/specs/2026-05-17-terrain-dsl-multifile-design.md`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum ImportError {
    FileNotFound { path: String, attempted_roots: Vec<PathBuf> },
    Cycle { path_chain: Vec<PathBuf> },
    DuplicateDefinition {
        kind: String,
        name: String,
        first_seen_at: PathBuf,
        second_seen_at: PathBuf,
    },
    IoError { path: PathBuf, source: std::io::Error },
    Parse { path: PathBuf, inner: String },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::FileNotFound { path, attempted_roots } => {
                write!(f, "import not found: `{path}`; attempted: {attempted_roots:?}")
            }
            ImportError::Cycle { path_chain } => {
                write!(f, "import cycle: {path_chain:?}")
            }
            ImportError::DuplicateDefinition { kind, name, first_seen_at, second_seen_at } => {
                write!(f, "duplicate {kind} `{name}`: first at {first_seen_at:?}, second at {second_seen_at:?}")
            }
            ImportError::IoError { path, source } => {
                write!(f, "I/O error reading `{path:?}`: {source}")
            }
            ImportError::Parse { path, inner } => {
                write!(f, "parse error in `{path:?}`: {inner}")
            }
        }
    }
}

impl std::error::Error for ImportError {}

/// Parse a top-level `.sim` file, recursively follow `import` statements,
/// and return a merged `Program`. Decls are merged depth-first post-order:
/// imports are appended before the importing file's own decls.
pub fn parse_with_imports(
    top_path: &Path,
    stdlib_root: &Path,
    sandbox_root: &Path,
) -> Result<dsl_ast::ast::Program, ImportError> {
    let top_canonical = top_path.canonicalize().map_err(|e| ImportError::IoError {
        path: top_path.to_path_buf(),
        source: e,
    })?;
    let mut visited: HashSet<PathBuf> = HashSet::new();
    let mut stack: Vec<PathBuf> = Vec::new();
    let mut merged_decls: Vec<dsl_ast::ast::Decl> = Vec::new();
    let mut merged_decl_sources: Vec<PathBuf> = Vec::new();
    let mut merged_terrain: Option<dsl_ast::terrain::TerrainBlock> = None;
    // Plan A — player-facing descriptor singletons. Merged like `terrain`.
    let mut merged_controls: Option<dsl_ast::ast::ControlsDecl> = None;
    let mut merged_render: Option<dsl_ast::ast::RenderDecl> = None;
    let mut merged_ui: Option<dsl_ast::ast::UiDecl> = None;
    let mut imports_resolved: Vec<PathBuf> = Vec::new();

    visit(
        &top_canonical,
        stdlib_root,
        sandbox_root,
        &mut visited,
        &mut stack,
        &mut merged_decls,
        &mut merged_decl_sources,
        &mut merged_terrain,
        &mut merged_controls,
        &mut merged_render,
        &mut merged_ui,
        &mut imports_resolved,
    )?;

    // Post-merge collision pass: detect duplicate (kind, name) pairs.
    // Terrain singleton collisions are caught eagerly in `visit`; this pass
    // covers all named Decl variants.
    // `merged_decl_sources` is a parallel vec (one entry per decl) recording
    // which file contributed each decl, enabling accurate attribution.
    {
        let mut seen: HashMap<(&'static str, String), PathBuf> = HashMap::new();
        for (i, decl) in merged_decls.iter().enumerate() {
            if let Some((kind, name)) = decl_kind_and_name(decl) {
                let key = (kind, name.clone());
                let this_source = merged_decl_sources[i].clone();
                if let Some(first_path) = seen.get(&key) {
                    return Err(ImportError::DuplicateDefinition {
                        kind: kind.to_string(),
                        name,
                        first_seen_at: first_path.clone(),
                        second_seen_at: this_source,
                    });
                }
                seen.insert(key, this_source);
            }
        }
    }

    Ok(dsl_ast::ast::Program {
        imports: Vec::new(), // merged file has no further imports
        imports_resolved,
        decls: merged_decls,
        terrain: merged_terrain,
        controls: merged_controls,
        render: merged_render,
        ui: merged_ui,
    })
}

fn visit(
    path: &Path,
    stdlib_root: &Path,
    sandbox_root: &Path,
    visited: &mut HashSet<PathBuf>,
    stack: &mut Vec<PathBuf>,
    merged_decls: &mut Vec<dsl_ast::ast::Decl>,
    merged_decl_sources: &mut Vec<PathBuf>,
    merged_terrain: &mut Option<dsl_ast::terrain::TerrainBlock>,
    merged_controls: &mut Option<dsl_ast::ast::ControlsDecl>,
    merged_render: &mut Option<dsl_ast::ast::RenderDecl>,
    merged_ui: &mut Option<dsl_ast::ast::UiDecl>,
    imports_resolved: &mut Vec<PathBuf>,
) -> Result<(), ImportError> {
    if stack.iter().any(|p| p == path) {
        let mut chain = stack.clone();
        chain.push(path.to_path_buf());
        return Err(ImportError::Cycle { path_chain: chain });
    }
    if !visited.insert(path.to_path_buf()) {
        // Already merged in a sibling branch — diamond import.
        return Ok(());
    }
    stack.push(path.to_path_buf());
    imports_resolved.push(path.to_path_buf());

    let src = std::fs::read_to_string(path).map_err(|e| ImportError::IoError {
        path: path.to_path_buf(),
        source: e,
    })?;
    let program = dsl_ast::parse(&src).map_err(|e| ImportError::Parse {
        path: path.to_path_buf(),
        inner: format!("{e}"),
    })?;

    // Recurse into imports first (depth-first post-order).
    for imp in &program.imports {
        let resolved = resolve_import_path(&imp.path, path, stdlib_root, sandbox_root)?;
        visit(
            &resolved,
            stdlib_root,
            sandbox_root,
            visited,
            stack,
            merged_decls,
            merged_decl_sources,
            merged_terrain,
            merged_controls,
            merged_render,
            merged_ui,
            imports_resolved,
        )?;
    }

    // Append this file's own decls, tracking source per decl.
    for decl in program.decls {
        merged_decls.push(decl);
        merged_decl_sources.push(path.to_path_buf());
    }
    if let Some(t) = program.terrain {
        if merged_terrain.is_some() {
            // Singleton collision — handled by Task 7's collision pass,
            // but emit the same error here so it surfaces at parse time.
            return Err(ImportError::DuplicateDefinition {
                kind: "terrain".to_string(),
                name: "<singleton>".to_string(),
                first_seen_at: imports_resolved[0].clone(),
                second_seen_at: path.to_path_buf(),
            });
        }
        *merged_terrain = Some(t);
    }
    // Plan A singletons — same collision policy as `terrain`.
    if let Some(d) = program.controls {
        if merged_controls.is_some() {
            return Err(ImportError::DuplicateDefinition {
                kind: "controls".to_string(),
                name: "<singleton>".to_string(),
                first_seen_at: imports_resolved[0].clone(),
                second_seen_at: path.to_path_buf(),
            });
        }
        *merged_controls = Some(d);
    }
    if let Some(d) = program.render {
        if merged_render.is_some() {
            return Err(ImportError::DuplicateDefinition {
                kind: "render".to_string(),
                name: "<singleton>".to_string(),
                first_seen_at: imports_resolved[0].clone(),
                second_seen_at: path.to_path_buf(),
            });
        }
        *merged_render = Some(d);
    }
    if let Some(d) = program.ui {
        if merged_ui.is_some() {
            return Err(ImportError::DuplicateDefinition {
                kind: "ui".to_string(),
                name: "<singleton>".to_string(),
                first_seen_at: imports_resolved[0].clone(),
                second_seen_at: path.to_path_buf(),
            });
        }
        *merged_ui = Some(d);
    }

    stack.pop();
    Ok(())
}

/// Returns a `(kind_tag, name)` pair for decls that carry a top-level name.
/// The kind tag is a static string so that `entity Wolf` and `event Wolf`
/// live in separate namespaces and do NOT collide with each other.
/// Decl variants without a top-level name (Init, Debug, Mask, Scoring,
/// Metric) return `None` and bypass the collision check.
fn decl_kind_and_name(decl: &dsl_ast::ast::Decl) -> Option<(&'static str, String)> {
    use dsl_ast::ast::Decl::*;
    match decl {
        Entity(d)       => Some(("entity",       d.name.clone())),
        Event(d)        => Some(("event",        d.name.clone())),
        EventTag(d)     => Some(("event_tag",    d.name.clone())),
        Enum(d)         => Some(("enum",         d.name.clone())),
        View(d)         => Some(("view",         d.name.clone())),
        Belief(d)       => Some(("belief",       d.name.clone())),
        Query(d)        => Some(("query",        d.name.clone())),
        Physics(d)      => Some(("physics",      d.name.clone())),
        PhysicsApply(d) => Some(("physics",      d.name.clone())),
        Verb(d)         => Some(("verb",         d.name.clone())),
        Invariant(d)    => Some(("invariant",    d.name.clone())),
        Probe(d)        => Some(("probe",        d.name.clone())),
        Config(d)       => Some(("config",       d.name.clone())),
        SpatialQuery(d) => Some(("spatial_query", d.name.clone())),
        AgentField(d)   => Some(("agent_field",  d.name.clone())),
        Table(d)        => Some(("table",        d.name.clone())),
        RegionKind(d)   => Some(("region_kind",  d.name.clone())),
        RegionIndices(d) => Some(("region_indices", d.name.clone())),
        Index(d)        => Some(("index",        d.name.clone())),
        // Variants without a top-level name bypass collision detection.
        Init(_) | Debug(_) | Mask(_) | Scoring(_) | Metric(_) => None,
    }
}

/// Resolves an import path string to a canonicalised absolute path on disk.
///
/// Modes:
/// - `std/<rest>` — resolves against `stdlib_root`.
/// - `./<rest>` or `../<rest>` — resolves against `importing_file`'s directory.
///
/// The canonicalised result is verified to be inside `sandbox_root` (after
/// canonicalising sandbox_root too). Sandbox-escape produces `FileNotFound`
/// with the attempted paths listed.
pub fn resolve_import_path(
    import_path: &str,
    importing_file: &Path,
    stdlib_root: &Path,
    sandbox_root: &Path,
) -> Result<PathBuf, ImportError> {
    let importing_dir = importing_file.parent()
        .ok_or_else(|| ImportError::FileNotFound {
            path: import_path.to_string(),
            attempted_roots: vec![importing_file.to_path_buf()],
        })?;
    let stripped_std = import_path.strip_prefix("std/");
    let candidate = if let Some(rest) = stripped_std {
        stdlib_root.join(rest)
    } else if import_path.starts_with("./") || import_path.starts_with("../") {
        importing_dir.join(import_path)
    } else {
        // Bare paths are not supported in v1.
        return Err(ImportError::FileNotFound {
            path: import_path.to_string(),
            attempted_roots: vec![],
        });
    };
    // canonicalize requires the file to exist.
    let resolved = candidate.canonicalize().map_err(|_| ImportError::FileNotFound {
        path: import_path.to_string(),
        attempted_roots: vec![candidate.clone()],
    })?;
    // Sandbox check.
    let sandbox = sandbox_root.canonicalize().map_err(|e| ImportError::IoError {
        path: sandbox_root.to_path_buf(),
        source: e,
    })?;
    if !resolved.starts_with(&sandbox) {
        return Err(ImportError::FileNotFound {
            path: import_path.to_string(),
            attempted_roots: vec![resolved],
        });
    }
    Ok(resolved)
}
