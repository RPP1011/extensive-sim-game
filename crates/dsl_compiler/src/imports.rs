//! Filesystem-aware import resolver + merger for `.sim` files.
//! See `docs/superpowers/specs/2026-05-17-terrain-dsl-multifile-design.md`.

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
