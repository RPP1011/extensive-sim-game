# DSL Multi-File Imports — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `import <path>;` top-of-file statement to the `.sim` DSL. The compiler grows a multi-file-aware entry point that resolves imports (relative + `std/` rooted), merges files depth-first into a single `Program`, detects collisions, and produces `imports_resolved` for build-system rerun wiring. A new top-level `stdlib/` directory is created with one seed example.

**Architecture:** `dsl_ast` stays pure: it only learns one new AST type (`Import`) and parser arm. The filesystem-aware resolver, cycle detector, file cache, and merger all live in `crates/dsl_compiler/src/imports.rs`. The new `parse_with_imports(top_path, stdlib_root, sandbox_root)` is exposed from `dsl_compiler` and called by `build_helper::emit_namespaced`. Existing single-file `parse(src)` is unchanged.

**Tech Stack:** Rust workspace. Pure parser in `dsl_ast`; filesystem-aware import resolver + `parse_with_imports` in `dsl_compiler`; build-time call site is `crates/sims/build.rs` via `dsl_compiler::build_helper::emit_namespaced`.

**Source spec:** `docs/superpowers/specs/2026-05-17-terrain-dsl-multifile-design.md`

---

## Architectural Impact Statement

- **Existing primitives searched:**
  - `dsl_ast::parse(source: &str) -> Result<Program, ParseError>` at `crates/dsl_ast/src/lib.rs:33`
  - `dsl_ast::ast::Program { decls, terrain }` at `crates/dsl_ast/src/ast.rs:40`
  - `dsl_ast::parser::parse_program(source: &str)` at `crates/dsl_ast/src/parser.rs:40`
  - `dsl_compiler::build_helper::emit_namespaced` / `emit_into` at `crates/dsl_compiler/src/build_helper.rs:136`
  - `crates/sims/build.rs` for the workspace_root / OUT_DIR pattern
  Search method: `rg -n`, direct `Read`.

- **Decision:** extend `dsl_ast` with one new AST type (`Import`) and a parser arm. Place the filesystem-aware resolver and merger in `dsl_compiler::imports` rather than in `dsl_ast`. This is a small **deviation from the spec**, which placed `parse_with_imports` in `dsl_ast`. Rationale: `dsl_ast` is currently a pure (no-IO) crate; moving filesystem access into it would invert the layering. The user-facing API surface (`dsl_compiler::parse_with_imports`) is functionally identical to what the spec described; only the crate that hosts it differs. All call sites still see `dsl_compiler::parse_with_imports`.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `crates/dsl_ast/src/ast.rs` (new `Import` type, new `Program.imports` field), `crates/dsl_ast/src/parser.rs` (`import` keyword arm), `crates/dsl_compiler/src/imports.rs` (new file — resolver, cycle detector, merger), `crates/dsl_compiler/src/lib.rs` (re-export `parse_with_imports`), `crates/dsl_compiler/src/build_helper.rs` (switch `emit_into` to the new entry point).
  - Generated outputs re-emitted: existing `OUT_DIR/<fixture>/{generated.rs, runtime_core.rs, terrain_gen.rs}`. No new outputs.

- **Hand-written downstream code:** NONE beyond resolver + merger library code (the same kind of build-time-only library code that already exists in `dsl_compiler::build_helper`). The resolver is not rule logic; P1 scope is unchanged.

- **Constitution check:**
  - P1 (Compiler-First): PASS — import resolution lives in the compiler, feeds the existing emitter, no hand-written rule handlers added.
  - P2 (Schema-Hash on Layout): N/A — no `SimState` SoA fields change. `Program.imports` is a build-time AST type.
  - P3 (Cross-Backend Parity): N/A — parse-time concern, `@cpu_only` umbrella as terrain.
  - P4 (`EffectOp` Size Budget): N/A — no new event variants.
  - P5 (Determinism via Keyed PCG): PASS — no RNG. Cycle detection uses a `Vec<PathBuf>` traversal stack (not HashMap). Path canonicalisation stable across symlink layouts.
  - P6 (Events Are the Mutation Channel): N/A — no state mutation.
  - P7 (Replayability Flagged): N/A — no new events.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS — each task ends with a `git commit`.
  - P10 (No Runtime Panic): PASS — errors are `Result`s. `build.rs` panics on resolver failure, which is the standard build-time failure mode.
  - P11 (Reduction Determinism): N/A — no reductions.

- **Runtime gate:** Task 12 adds a smoke fixture `assets/sim/terrain_probe_imported.sim` that imports `stdlib/materials/basic.sim`, builds through `crates/sims/build.rs`, and exercises `sims::terrain_probe_imported::generate_terrain` to confirm the merged `Program`'s materials come from stdlib. This is the observable post-condition on the changed code path.
  - `terrain_probe_imported_smoke` at `crates/sims/tests/terrain_probe_imported_smoke.rs` — "build a fixture using an import → emitter sees merged materials → MATERIALS table reflects stdlib content".

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

---

## Files touched

- Modify: `crates/dsl_ast/src/ast.rs` — add `pub struct Import { path: String }`, add `pub imports: Vec<Import>` to `Program`.
- Modify: `crates/dsl_ast/src/parser.rs` — recognise `import <path>;` at the top of the file; reject after non-import decls.
- Modify: `crates/dsl_ast/src/error.rs` (or wherever `ParseError` lives) — add `ImportAfterDecl` variant.
- Create: `crates/dsl_compiler/src/imports.rs` — `ImportError` enum, resolver (`resolve_import_path`), cycle detector, merger (`parse_with_imports`), collision check.
- Modify: `crates/dsl_compiler/src/lib.rs` — `pub mod imports;` + `pub use imports::{parse_with_imports, ImportError};`.
- Modify: `crates/dsl_compiler/src/build_helper.rs` — switch `emit_into` from `dsl_ast::parse(src)` to `crates/dsl_compiler::parse_with_imports(file_path, &stdlib_root, &sandbox_root)`.
- Modify: `crates/sims/build.rs` — read `WORLDSIM_STDLIB_ROOT` / `WORLDSIM_SANDBOX_ROOT` env vars (with workspace-root defaults); thread paths through to `emit_namespaced`; emit `cargo:rerun-if-changed` for each `imports_resolved` path.
- Modify: `crates/dsl_compiler/src/build_helper.rs` — extend `emit_namespaced_with_strategy` to take and use these roots (signature change is internal — public `emit_namespaced(fixture)` still resolves env-var defaults if not passed explicitly).
- Create: `stdlib/README.md` — explains what the directory is for.
- Create: `stdlib/materials/basic.sim` — minimal seed material palette (grass, stone, sand).
- Create: `assets/sim/terrain_probe_imported.sim` — smoke fixture that imports the stdlib materials.
- Modify: `crates/sims/build.rs` — add `"terrain_probe_imported"` to the allow-list.
- Create: `crates/sims/tests/terrain_probe_imported_smoke.rs` — runtime-gate test.

---

## Task 1: AST node — `Import` + `Program.imports`

**Files:**
- Modify: `crates/dsl_ast/src/ast.rs`
- Test: `crates/dsl_ast/tests/import_node.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_ast/tests/import_node.rs
use dsl_ast::ast::{Import, Program};

#[test]
fn import_construct_and_field_access() {
    let imp = Import { path: "std/materials/basic.sim".into() };
    assert_eq!(imp.path, "std/materials/basic.sim");
}

#[test]
fn program_grows_imports_field_defaulting_empty() {
    let p = Program::default();
    assert!(p.imports.is_empty());
}
```

(`Program::default()` may not exist — the test prefers `Default` to direct construction so the field is verified to default to `vec![]`. If `Default` is not derived, the test should use `Program { imports: vec![], decls: vec![], terrain: None }` and assert on `imports.is_empty()`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_ast --test import_node`
Expected: FAIL — `Import` does not exist, or `Program.imports` is missing.

- [ ] **Step 3: Add the AST type and Program field**

In `crates/dsl_ast/src/ast.rs`, add (alongside existing types — place near `Program`):

```rust
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Import {
    pub path: String,
}
```

Modify `Program` to add `pub imports: Vec<Import>`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    pub imports: Vec<Import>,
    pub decls: Vec<Decl>,
    pub terrain: Option<crate::terrain::TerrainBlock>,
}
```

Also update the `lib.rs` re-export (`pub use ast::{..., Import, ...};`) if AST types are listed there.

Find every direct `Program { ... }` construction site (use `rg -n "Program *{" crates/`) and add `imports: vec![]` to each. The terrain DSL plan already established this pattern.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_ast --test import_node`
Expected: 2 passed.

Also run: `cargo check --workspace`
Expected: clean (all `Program { ... }` direct construction sites updated).

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_ast/src/ast.rs crates/dsl_ast/src/lib.rs crates/dsl_ast/tests/import_node.rs
git commit -m "feat(dsl_ast): Import AST type + Program.imports field"
```

---

## Task 2: Parser arm — recognise `import <path>;`

**Files:**
- Modify: `crates/dsl_ast/src/parser.rs` — add `import` keyword arm at the top of `parse_program`.
- Test: `crates/dsl_compiler/tests/import_parse_basic.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/import_parse_basic.rs
use dsl_compiler::parse;

#[test]
fn parses_three_imports() {
    let src = r#"
import std/materials/basic.sim;
import ./local.sim;
import ../shared/foo.sim;

agent Wolf { n: 1 }
"#;
    let program = parse(src).expect("parse");
    assert_eq!(program.imports.len(), 3);
    assert_eq!(program.imports[0].path, "std/materials/basic.sim");
    assert_eq!(program.imports[1].path, "./local.sim");
    assert_eq!(program.imports[2].path, "../shared/foo.sim");
}

#[test]
fn parses_zero_imports_when_absent() {
    let src = r#"agent Wolf { n: 1 }"#;
    let program = parse(src).expect("parse");
    assert!(program.imports.is_empty());
}

#[test]
fn import_path_requires_dot_sim_suffix() {
    // Required extension keeps imports grep-able.
    let src = "import std/materials/basic;\n";
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("import path must end in `.sim`"), "got: {msg}");
}

#[test]
fn import_must_have_semicolon() {
    let src = "import std/materials/basic.sim\n";
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains(";"), "got: {msg}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test import_parse_basic`
Expected: FAIL — parser doesn't recognise `import`.

- [ ] **Step 3: Add the parser arm**

In `crates/dsl_ast/src/parser.rs`, inside `parse_program` (before the main decl-loop), add a peek-and-consume loop for `import`:

```rust
// Inside parse_program, near the top of the function:
let mut imports: Vec<crate::ast::Import> = Vec::new();
loop {
    // Skip whitespace + comments using whatever helper the parser uses.
    // Bail if we're at EOF.
    if at_eof(c) { break; }
    // Peek next token. If it's the keyword `import`, consume.
    if peek_keyword(c, "import") {
        consume_keyword(c, "import")?;
        // Read the path as a sequence of non-whitespace, non-semicolon chars.
        // (Bare path, not a string literal — matches the spec's grep-ability goal.)
        let path = consume_import_path(c)?;
        if !path.ends_with(".sim") {
            return Err(parser_error(c, "import path must end in `.sim`"));
        }
        expect_punct(c, ";")?;
        imports.push(crate::ast::Import { path });
        continue;
    }
    break;
}
```

Adapt the helper names (`peek_keyword`, `consume_keyword`, `consume_import_path`, `expect_punct`, `parser_error`) to match what `parser.rs` already provides. If a helper like `consume_import_path` doesn't exist, implement it inline by collecting characters until whitespace or `;`.

Set `program.imports = imports;` at the point where the `Program` is constructed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test import_parse_basic`
Expected: 4 passed.

Also: existing parser tests still pass — `cargo test -p dsl_compiler --test terrain_parse_basic terrain_parse_materials terrain_parse_layers 2>&1 | tail -10` → 8 passed total.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_ast/src crates/dsl_compiler/tests/import_parse_basic.rs
git commit -m "feat(dsl): parse `import <path>;` at top of .sim file"
```

---

## Task 3: Reject `import` after a non-import decl

**Files:**
- Modify: `crates/dsl_ast/src/parser.rs`
- Test: `crates/dsl_compiler/tests/import_parse_order.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/import_parse_order.rs
use dsl_compiler::parse;

#[test]
fn rejects_import_after_agent_decl() {
    let src = r#"
agent Wolf { n: 1 }
import std/materials/basic.sim;
"#;
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("import") && msg.contains("after"),
            "expected ImportAfterDecl error, got: {msg}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test import_parse_order`
Expected: FAIL — parser currently treats a later `import` either as an unknown ident or silently mis-parses.

- [ ] **Step 3: Implement the rejection**

In `parse_program`, after the decl-loop is running, if the loop encounters the `import` keyword, return a parser error containing the substrings "import" and "after":

```rust
// Inside the main decl loop:
if peek_keyword(c, "import") {
    return Err(parser_error(c, "`import` statements must appear before any other top-level decl; found `import` after a decl"));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test import_parse_order`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_ast/src/parser.rs crates/dsl_compiler/tests/import_parse_order.rs
git commit -m "feat(dsl): reject `import` after a non-import top-level decl"
```

---

## Task 4: `ImportError` enum + resolver helper

**Files:**
- Create: `crates/dsl_compiler/src/imports.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (register `pub mod imports;`)
- Test: `crates/dsl_compiler/tests/imports_resolve.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/imports_resolve.rs
use dsl_compiler::imports::{resolve_import_path, ImportError};
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn stdlib_resolves_against_stdlib_root() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(stdlib.join("materials")).unwrap();
    let target = stdlib.join("materials/basic.sim");
    std::fs::write(&target, "// stdlib content\n").unwrap();

    let importing = tmp.path().join("a.sim");
    std::fs::write(&importing, "import std/materials/basic.sim;\n").unwrap();

    let resolved = resolve_import_path(
        "std/materials/basic.sim",
        &importing,
        &stdlib,
        tmp.path(), // sandbox root
    ).unwrap();
    assert_eq!(resolved, target.canonicalize().unwrap());
}

#[test]
fn relative_resolves_against_importing_file() {
    let tmp = tempdir().unwrap();
    let local = tmp.path().join("local.sim");
    std::fs::write(&local, "// local\n").unwrap();
    let importing = tmp.path().join("a.sim");
    std::fs::write(&importing, "").unwrap();

    let resolved = resolve_import_path(
        "./local.sim",
        &importing,
        &tmp.path().join("stdlib"), // doesn't need to exist for ./ path
        tmp.path(),
    ).unwrap();
    assert_eq!(resolved, local.canonicalize().unwrap());
}

#[test]
fn parent_traversal_resolves_against_importing_file() {
    let tmp = tempdir().unwrap();
    let sub = tmp.path().join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    let shared = tmp.path().join("shared.sim");
    std::fs::write(&shared, "").unwrap();
    let importing = sub.join("a.sim");
    std::fs::write(&importing, "").unwrap();

    let resolved = resolve_import_path(
        "../shared.sim",
        &importing,
        &tmp.path().join("stdlib"),
        tmp.path(),
    ).unwrap();
    assert_eq!(resolved, shared.canonicalize().unwrap());
}

#[test]
fn sandbox_escape_via_parent_is_rejected() {
    // Importing file is in tmp/inner/a.sim, sandbox is tmp/inner,
    // so `../outside.sim` would escape.
    let tmp = tempdir().unwrap();
    let inner = tmp.path().join("inner");
    std::fs::create_dir_all(&inner).unwrap();
    let outside = tmp.path().join("outside.sim");
    std::fs::write(&outside, "").unwrap();
    let importing = inner.join("a.sim");
    std::fs::write(&importing, "").unwrap();

    let err = resolve_import_path(
        "../outside.sim",
        &importing,
        &inner.join("stdlib"),
        &inner, // sandbox is restricted to inner/
    ).err().unwrap();
    assert!(matches!(err, ImportError::FileNotFound { .. }),
            "expected FileNotFound for sandbox escape, got: {err:?}");
}

#[test]
fn missing_file_is_file_not_found() {
    let tmp = tempdir().unwrap();
    let importing = tmp.path().join("a.sim");
    std::fs::write(&importing, "").unwrap();

    let err = resolve_import_path(
        "std/does_not_exist.sim",
        &importing,
        &tmp.path().join("stdlib"),
        tmp.path(),
    ).err().unwrap();
    assert!(matches!(err, ImportError::FileNotFound { .. }), "got: {err:?}");
}
```

Confirm `tempfile` is already in `crates/dsl_compiler/Cargo.toml` `[dev-dependencies]` (it was added in the terrain DSL plan T9). If absent, add `tempfile = "3"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test imports_resolve`
Expected: FAIL — `dsl_compiler::imports` does not exist.

- [ ] **Step 3: Implement the resolver**

Create `crates/dsl_compiler/src/imports.rs`:

```rust
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
```

In `crates/dsl_compiler/src/lib.rs`, add `pub mod imports;`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test imports_resolve`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/imports.rs crates/dsl_compiler/src/lib.rs crates/dsl_compiler/tests/imports_resolve.rs
git commit -m "feat(dsl): import path resolver with std/, ./, ../ + sandbox check"
```

---

## Task 5: `parse_with_imports` — basic two-file merge + transitive

**Files:**
- Modify: `crates/dsl_compiler/src/imports.rs`
- Modify: `crates/dsl_compiler/src/lib.rs` (re-export `parse_with_imports`)
- Test: `crates/dsl_compiler/tests/imports_merge.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/imports_merge.rs
use dsl_compiler::parse_with_imports;
use tempfile::tempdir;

#[test]
fn import_free_file_equivalent_to_parse() {
    // Regression: a .sim file with zero imports must produce the same
    // Program shape via parse_with_imports as via parse(src).
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let a = tmp.path().join("a.sim");
    let src = "entity Wolf : Agent { hp: 10.0 }\nentity Goblin : Agent { hp: 5.0 }\n";
    std::fs::write(&a, src).unwrap();

    let via_pwi = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    let via_pure = dsl_compiler::parse(src).unwrap();

    assert_eq!(via_pwi.decls.len(), via_pure.decls.len());
    assert_eq!(via_pwi.terrain, via_pure.terrain);
    // imports_resolved on parse_with_imports has exactly one entry (the
    // top file itself); pure parse() has none.
    assert_eq!(via_pwi.imports_resolved.len(), 1);
    assert!(via_pure.imports_resolved.is_empty());
}

#[test]
fn two_file_import_merges_decls() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");

    std::fs::create_dir_all(&stdlib).unwrap();
    std::fs::write(&b, "entity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\n\nentity Goblin : Agent { hp: 5.0 }\n").unwrap();

    let program = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    // Imports flatten: b's decls before a's decls.
    assert_eq!(program.decls.len(), 2);
    // First decl is from b (Wolf), second from a (Goblin).
    // Inspect via Debug — exact match shape depends on Decl enum.
    let debug_first = format!("{:?}", program.decls[0]);
    let debug_second = format!("{:?}", program.decls[1]);
    assert!(debug_first.contains("Wolf"), "first should be Wolf, got: {debug_first}");
    assert!(debug_second.contains("Goblin"), "second should be Goblin, got: {debug_second}");
}

#[test]
fn transitive_import_flattens_depth_first() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let c = tmp.path().join("c.sim");
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&c, "entity Rat : Agent { hp: 1.0 }\n").unwrap();
    std::fs::write(&b, "import ./c.sim;\nentity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nentity Goblin : Agent { hp: 5.0 }\n").unwrap();

    let program = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    assert_eq!(program.decls.len(), 3);
    // Order: C (Rat), B (Wolf), A (Goblin).
    let debugs: Vec<String> = program.decls.iter().map(|d| format!("{:?}", d)).collect();
    assert!(debugs[0].contains("Rat"),    "expected Rat first, got: {}", debugs[0]);
    assert!(debugs[1].contains("Wolf"),   "expected Wolf second, got: {}", debugs[1]);
    assert!(debugs[2].contains("Goblin"), "expected Goblin third, got: {}", debugs[2]);
}

#[test]
fn diamond_import_loads_each_file_once() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let d = tmp.path().join("d.sim");
    let b = tmp.path().join("b.sim");
    let c = tmp.path().join("c.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&d, "entity Rat : Agent { hp: 1.0 }\n").unwrap();
    std::fs::write(&b, "import ./d.sim;\nentity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&c, "import ./d.sim;\nentity Bear : Agent { hp: 20.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nimport ./c.sim;\nentity Goblin : Agent { hp: 5.0 }\n").unwrap();

    let program = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    // D contributes Rat exactly once; B contributes Wolf; C contributes Bear; A contributes Goblin.
    assert_eq!(program.decls.len(), 4);
    let rat_count = program.decls.iter().filter(|d| format!("{d:?}").contains("Rat")).count();
    assert_eq!(rat_count, 1, "Rat must appear exactly once in diamond import");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test imports_merge`
Expected: FAIL — `parse_with_imports` does not exist.

- [ ] **Step 3: Implement `parse_with_imports`**

Append to `crates/dsl_compiler/src/imports.rs`:

```rust
use std::collections::HashSet;

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
    let mut merged_terrain: Option<dsl_ast::terrain::TerrainBlock> = None;
    let mut imports_resolved: Vec<PathBuf> = Vec::new();

    visit(
        &top_canonical,
        stdlib_root,
        sandbox_root,
        &mut visited,
        &mut stack,
        &mut merged_decls,
        &mut merged_terrain,
        &mut imports_resolved,
    )?;

    Ok(dsl_ast::ast::Program {
        imports: Vec::new(), // merged file has no further imports
        decls: merged_decls,
        terrain: merged_terrain,
        // If Program grows additional fields in future, fill defaults.
    })
    // Note: imports_resolved is exposed in Task 8 (separate field on Program).
}

fn visit(
    path: &Path,
    stdlib_root: &Path,
    sandbox_root: &Path,
    visited: &mut HashSet<PathBuf>,
    stack: &mut Vec<PathBuf>,
    merged_decls: &mut Vec<dsl_ast::ast::Decl>,
    merged_terrain: &mut Option<dsl_ast::terrain::TerrainBlock>,
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
            merged_terrain,
            imports_resolved,
        )?;
    }

    // Append this file's own decls.
    merged_decls.extend(program.decls);
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

    stack.pop();
    Ok(())
}
```

In `crates/dsl_compiler/src/lib.rs`, add: `pub use imports::{parse_with_imports, ImportError};`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test imports_merge`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/imports.rs crates/dsl_compiler/src/lib.rs crates/dsl_compiler/tests/imports_merge.rs
git commit -m "feat(dsl): parse_with_imports — depth-first merge, diamond-safe"
```

---

## Task 6: Cycle detection

**Files:**
- Modify: `crates/dsl_compiler/src/imports.rs` (already covers it; this task adds the test pin).
- Test: `crates/dsl_compiler/tests/imports_cycle.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/imports_cycle.rs
use dsl_compiler::{parse_with_imports, ImportError};
use tempfile::tempdir;

#[test]
fn direct_self_import_is_cycle() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let a = tmp.path().join("a.sim");
    std::fs::write(&a, "import ./a.sim;\n").unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    assert!(matches!(err, ImportError::Cycle { .. }), "got: {err:?}");
}

#[test]
fn indirect_cycle_a_b_a() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let a = tmp.path().join("a.sim");
    let b = tmp.path().join("b.sim");
    std::fs::write(&a, "import ./b.sim;\n").unwrap();
    std::fs::write(&b, "import ./a.sim;\n").unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    match err {
        ImportError::Cycle { path_chain } => {
            assert!(path_chain.len() >= 3, "chain should be a -> b -> a; got: {path_chain:?}");
        }
        other => panic!("expected Cycle, got: {other:?}"),
    }
}
```

- [ ] **Step 2: Run test to verify it passes (the implementation already handles cycles from Task 5)**

Run: `cargo test -p dsl_compiler --test imports_cycle`
Expected: 2 passed. If it fails, the visit-function's stack check has a bug — fix it.

- [ ] **Step 3: Commit**

```bash
git add crates/dsl_compiler/tests/imports_cycle.rs
git commit -m "test(dsl): cycle detection in parse_with_imports"
```

---

## Task 7: Collision detection (DuplicateDefinition)

**Files:**
- Modify: `crates/dsl_compiler/src/imports.rs` — add a post-merge collision pass.
- Test: `crates/dsl_compiler/tests/imports_collision.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/imports_collision.rs
use dsl_compiler::{parse_with_imports, ImportError};
use tempfile::tempdir;

#[test]
fn duplicate_entity_across_files_is_error() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&b, "entity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nentity Wolf : Agent { hp: 99.0 }\n").unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    match err {
        ImportError::DuplicateDefinition { kind, name, .. } => {
            assert_eq!(kind, "entity");
            assert_eq!(name, "Wolf");
        }
        other => panic!("expected DuplicateDefinition, got: {other:?}"),
    }
}

#[test]
fn different_kinds_same_name_does_not_collide() {
    // entity Wolf and event Wolf live in different kind-namespaces;
    // both can coexist in the merged Program without error.
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&b, "event Wolf {}\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nentity Wolf : Agent { hp: 10.0 }\n").unwrap();
    // Should succeed — kind-namespaced collision check lets these coexist.
    let program = parse_with_imports(&a, &stdlib, tmp.path()).expect("kind-scoped collision allows entity+event with same name");
    assert_eq!(program.decls.len(), 2);
}

#[test]
fn two_terrain_blocks_via_imports_is_error() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let materials_block = r#"
terrain {
  extent: 4
  cell_size: 1.0
  seed_purpose: 0x1
  materials { grass { id: 1, walkable: true, hardness: 1, color: 0x4A8B3A } }
  layer fill { material: grass }
}
"#;
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&b, materials_block).unwrap();
    std::fs::write(&a, format!("import ./b.sim;\n{materials_block}")).unwrap();
    let err = parse_with_imports(&a, &stdlib, tmp.path()).err().unwrap();
    assert!(matches!(err, ImportError::DuplicateDefinition { ref kind, .. } if kind == "terrain"), "got: {err:?}");
}
```

- [ ] **Step 2: Run test to verify it fails (or passes — depends on Task 5's terrain handling)**

Run: `cargo test -p dsl_compiler --test imports_collision`
Expected: the first test FAILs (no decl-level collision check yet). The second test may already PASS due to Task 5's `terrain.is_some()` guard.

- [ ] **Step 3: Add the collision pass**

Append to `crates/dsl_compiler/src/imports.rs`:

```rust
use std::collections::HashMap;

/// Returns a (Kind, Name) tag for a decl. Kind tag is namespaced
/// so e.g. `entity Wolf` and `event Wolf` do NOT collide.
fn decl_kind_and_name(decl: &dsl_ast::ast::Decl) -> Option<(&'static str, String)> {
    use dsl_ast::ast::Decl::*;
    match decl {
        Entity(d)     => Some(("entity",     d.name.clone())),
        Event(d)      => Some(("event",      d.name.clone())),
        EventTag(d)   => Some(("event_tag",  d.name.clone())),
        Enum(d)       => Some(("enum",       d.name.clone())),
        View(d)       => Some(("view",       d.name.clone())),
        Belief(d)     => Some(("belief",     d.name.clone())),
        Query(d)      => Some(("query",      d.name.clone())),
        // Add other Decl variants as needed; if a variant has no name,
        // return None and let it through (collision is name-scoped).
        _ => None,
    }
}
```

(Inspect `crates/dsl_ast/src/ast.rs` for the full `Decl` enum and add arms for every variant that has a `name` field. If a variant has no name, return None so it doesn't enter the collision check.)

In `parse_with_imports`, after the `visit` call (before constructing the final `Program`), add a collision pass:

```rust
let mut seen: HashMap<(&'static str, String), PathBuf> = HashMap::new();
for (decl, source) in merged_decls.iter().zip(imports_resolved.iter()) {
    // Note: this attribution is approximate — `imports_resolved` is per-file,
    // not per-decl. For accurate per-decl attribution we'd need to thread the
    // source path into each decl, which is a bigger change. v1 emits "this
    // collision happened, here's the file the second occurrence came from".
    // The check is correct; only the error's `first` field is approximate.
    let _ = source;
    if let Some((kind, name)) = decl_kind_and_name(decl) {
        let key = (kind, name.clone());
        if let Some(first_path) = seen.get(&key) {
            return Err(ImportError::DuplicateDefinition {
                kind: kind.to_string(),
                name,
                first_seen_at: first_path.clone(),
                second_seen_at: imports_resolved.last().cloned().unwrap_or_default(),
            });
        }
        seen.insert(key, imports_resolved.last().cloned().unwrap_or_default());
    }
}
```

This emits a collision error if any two decls in the merged Program share a (kind, name). Decl-to-file attribution is approximate; a follow-up plan can thread per-decl source spans through the parser.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test imports_collision`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/imports.rs crates/dsl_compiler/tests/imports_collision.rs
git commit -m "feat(dsl): post-merge collision check (kind-scoped duplicate decl)"
```

---

## Task 8: `Program.imports_resolved` field

**Files:**
- Modify: `crates/dsl_ast/src/ast.rs` — add `imports_resolved: Vec<PathBuf>` to `Program`.
- Modify: `crates/dsl_compiler/src/imports.rs` — populate the field.
- Test: `crates/dsl_compiler/tests/imports_resolved_paths.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/imports_resolved_paths.rs
use dsl_compiler::parse_with_imports;
use tempfile::tempdir;

#[test]
fn resolved_paths_are_absolute_canonicalised() {
    let tmp = tempdir().unwrap();
    let stdlib = tmp.path().join("stdlib");
    std::fs::create_dir_all(&stdlib).unwrap();
    let b = tmp.path().join("b.sim");
    let a = tmp.path().join("a.sim");
    std::fs::write(&b, "entity Wolf : Agent { hp: 10.0 }\n").unwrap();
    std::fs::write(&a, "import ./b.sim;\nentity Goblin : Agent { hp: 5.0 }\n").unwrap();

    let program = parse_with_imports(&a, &stdlib, tmp.path()).unwrap();
    assert!(program.imports_resolved.len() >= 2);
    for p in &program.imports_resolved {
        assert!(p.is_absolute(), "expected absolute path, got {p:?}");
        // Canonicalised: contains no `..` segments.
        assert!(!p.components().any(|c| matches!(c, std::path::Component::ParentDir)),
                "expected canonicalised path, got {p:?}");
    }
    // The top file and the imported file both appear.
    let names: Vec<_> = program.imports_resolved.iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect();
    assert!(names.contains(&"a.sim".to_string()), "a.sim missing: {names:?}");
    assert!(names.contains(&"b.sim".to_string()), "b.sim missing: {names:?}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test imports_resolved_paths`
Expected: FAIL — `Program.imports_resolved` does not exist.

- [ ] **Step 3: Add the field**

In `crates/dsl_ast/src/ast.rs`:

```rust
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    pub imports: Vec<Import>,
    pub imports_resolved: Vec<PathBuf>,   // populated by parse_with_imports; empty for parse(src)
    pub decls: Vec<Decl>,
    pub terrain: Option<crate::terrain::TerrainBlock>,
}
```

If `Serialize` complains about `PathBuf`, add `#[serde(skip)]` to that field — `imports_resolved` is a runtime metadata field, not part of the serialised AST.

Update every direct `Program { ... }` construction site (parser, tests, etc.) with `imports_resolved: vec![]`. Most callers don't populate it; only `parse_with_imports` does.

In `crates/dsl_compiler/src/imports.rs`'s `parse_with_imports`, set the field:

```rust
Ok(dsl_ast::ast::Program {
    imports: Vec::new(),
    imports_resolved,
    decls: merged_decls,
    terrain: merged_terrain,
})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test imports_resolved_paths`
Expected: 1 passed.

Also run: `cargo check --workspace`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_ast crates/dsl_compiler
git commit -m "feat(dsl): Program.imports_resolved — canonicalised contributing paths"
```

---

## Task 9: Wire `parse_with_imports` into `build_helper`

**Files:**
- Modify: `crates/dsl_compiler/src/build_helper.rs`
- Test: `crates/dsl_compiler/tests/build_helper_with_imports.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/dsl_compiler/tests/build_helper_with_imports.rs
//! Asserts emit_namespaced uses parse_with_imports under the hood: if a
//! fixture's .sim has `import ./shared.sim;` and shared.sim exists in
//! the fake workspace, the merged Program's decls are visible in the
//! emitted artifacts.

use std::path::PathBuf;
use std::sync::Mutex;
use dsl_compiler::build_helper;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn fake_env(tmp: &tempfile::TempDir, sim_name: &str, sim_src: &str) -> PathBuf {
    let sims_dir = tmp.path().join("crates/sims");
    let assets_dir = tmp.path().join("assets/sim");
    let stdlib_dir = tmp.path().join("stdlib");
    std::fs::create_dir_all(&sims_dir).unwrap();
    std::fs::create_dir_all(&assets_dir).unwrap();
    std::fs::create_dir_all(&stdlib_dir).unwrap();
    let sim_path = assets_dir.join(format!("{sim_name}.sim"));
    std::fs::write(&sim_path, sim_src).unwrap();
    let out_dir = tmp.path().join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    std::env::set_var("CARGO_MANIFEST_DIR", &sims_dir);
    std::env::set_var("OUT_DIR", &out_dir);
    std::env::set_var("WORLDSIM_STDLIB_ROOT", &stdlib_dir);
    std::env::set_var("WORLDSIM_SANDBOX_ROOT", tmp.path());
    out_dir
}

#[test]
fn fixture_with_import_merges_shared_entity() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let assets = tmp.path().join("assets/sim");
    std::fs::create_dir_all(&assets).unwrap();
    // shared.sim defines Wolf; main.sim imports it and defines Goblin.
    std::fs::write(assets.join("shared.sim"), "entity Wolf : Agent { hp: 10.0 }\n").unwrap();
    let out_dir = fake_env(&tmp, "fixture_with_import",
        "import ./shared.sim;\nentity Goblin : Agent { hp: 5.0 }\n");
    build_helper::emit_namespaced("fixture_with_import");
    // The generated.rs should reflect both Wolf and Goblin (or fail to
    // emit if the merge didn't happen — either failure mode signals a
    // missing wire-up).
    let generated = out_dir.join("fixture_with_import/generated.rs");
    assert!(generated.exists(), "generated.rs not written: {generated:?}");
    let body = std::fs::read_to_string(&generated).unwrap();
    // Look for both names somewhere in the emitted code.
    assert!(body.contains("Wolf"),   "Wolf missing from emitted code");
    assert!(body.contains("Goblin"), "Goblin missing from emitted code");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p dsl_compiler --test build_helper_with_imports`
Expected: FAIL — `build_helper::emit_into` still uses `parse(src)` and ignores imports.

- [ ] **Step 3: Modify `emit_into` to use `parse_with_imports`**

Locate `emit_into` in `crates/dsl_compiler/src/build_helper.rs` (around line 156 onwards). Find the line that calls `dsl_ast::parse(&src)` or `crate::parse(&src)`. Replace it with:

```rust
// Resolve roots from env vars with workspace_root defaults. workspace_root
// is already computed earlier in this function.
use std::path::PathBuf;
let stdlib_root: PathBuf = match std::env::var_os("WORLDSIM_STDLIB_ROOT") {
    Some(s) => PathBuf::from(s),
    None    => workspace_root.join("stdlib"),
};
let sandbox_root: PathBuf = match std::env::var_os("WORLDSIM_SANDBOX_ROOT") {
    Some(s) => PathBuf::from(s),
    None    => workspace_root.clone(),
};
let program = crate::imports::parse_with_imports(
    &sim_path, &stdlib_root, &sandbox_root,
).unwrap_or_else(|e| panic!("parse {sim_path:?} with imports: {e}"));
```

Adapt the variable names (`sim_path`, `workspace_root`) to match what `emit_into` already has. The downstream code that consumes `program` doesn't need to change — `Program` shape is preserved.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p dsl_compiler --test build_helper_with_imports`
Expected: 1 passed.

Also run all prior tests to verify no regressions: `cargo test -p dsl_compiler --tests 2>&1 | grep "test result"` → all green.

- [ ] **Step 5: Commit**

```bash
git add crates/dsl_compiler/src/build_helper.rs crates/dsl_compiler/tests/build_helper_with_imports.rs
git commit -m "feat(dsl): build_helper uses parse_with_imports — stdlib + sandbox env vars"
```

---

## Task 10: Create `stdlib/` directory + seed example

**Files:**
- Create: `stdlib/README.md`
- Create: `stdlib/materials/basic.sim`

- [ ] **Step 1: Create the directory + seed content**

`stdlib/README.md`:

```markdown
# `stdlib/` — Shared DSL fragments

This directory holds reusable `.sim` fragments that any fixture can
`import` via the `std/<path>.sim` prefix. The stdlib root is resolved
at build time from the `WORLDSIM_STDLIB_ROOT` env var, defaulting to
`<workspace-root>/stdlib/`.

See `docs/superpowers/specs/2026-05-17-terrain-dsl-multifile-design.md`
for the import system design.

## Conventions

- Each file is a normal `.sim` snippet. Whole-file import semantics
  apply: every top-level decl in the imported file enters the
  importing file's scope.
- Names must be unique across all imported files in any given
  fixture's merged Program. Collisions are compile errors.
```

`stdlib/materials/basic.sim`:

```text
// Basic outdoor materials palette: grass, stone, sand.
// Imported via `import std/materials/basic.sim;`.

terrain {
  extent: 8
  cell_size: 1.0
  seed_purpose: 0xBAS1_C001
  materials {
    grass { id: 1, walkable: true,  hardness: 1, color: 0x4A8B3A }
    stone { id: 2, walkable: false, hardness: 8, color: 0x808080 }
    sand  { id: 3, walkable: true,  hardness: 2, color: 0xD9C28A, movement_cost: 1.5 }
  }
  layer fill { material: stone }
}
```

(Note: the `terrain { ... }` block ships a complete singleton because that's the only kind of fixture v1 actually supports — fixtures that import this file effectively *adopt* this terrain. A future spec could split materials out of `terrain` so they can be shared without committing to a layout. v1 keeps it simple.)

- [ ] **Step 2: Verify no breakage**

Run: `cargo check --workspace`
Expected: clean. Adding files to `stdlib/` does not affect the build (no fixture imports it yet — that's Task 11).

- [ ] **Step 3: Commit**

```bash
git add stdlib/
git commit -m "feat(stdlib): seed stdlib/ with README + basic materials palette"
```

---

## Task 11: Smoke fixture `terrain_probe_imported.sim`

**Files:**
- Create: `assets/sim/terrain_probe_imported.sim`
- Modify: `crates/sims/build.rs` — add `"terrain_probe_imported"` to the allow-list.

- [ ] **Step 1: Create the fixture**

```text
// assets/sim/terrain_probe_imported.sim
// Smoke fixture: validates that a real fixture can pull terrain
// (including its materials) from stdlib via `import`. The merged
// Program's MATERIALS table should come from `stdlib/materials/basic.sim`.

import std/materials/basic.sim;

entity Probe : Agent { hp: 1.0 }

init {
  spawn(Probe, n: 1)
}

physics TerrainProbeImportedNoop {
  // No physics — the smoke test just verifies terrain/materials reached
  // the emitter via the import path.
}
```

(Adjust the fixture's agent/init/physics scaffolding to match what `terrain_probe.sim` did in the previous plan — the goal is the minimum that `runtime_core.rs` accepts. Copy the structure from `assets/sim/terrain_probe.sim` if needed.)

- [ ] **Step 2: Allow-list the fixture**

In `crates/sims/build.rs`, add `"terrain_probe_imported"` to the `matches!` allow-list, alphabetically near `"terrain_probe"`.

- [ ] **Step 3: Verify the megacrate compiles**

Run: `WORLDSIM_STDLIB_ROOT=$(pwd)/stdlib cargo check -p sims 2>&1 | tail -20`
Expected: clean compile. The `sims::terrain_probe_imported` module should exist and expose `generate_terrain`, `MATERIALS`, `EXTENT`.

If the build complains about `WORLDSIM_STDLIB_ROOT` not being set, the build.rs hasn't been updated to read it yet — check `crates/sims/build.rs` for whether it already reads the env var (it should, after Task 9's wire-up).

- [ ] **Step 4: Commit**

```bash
git add assets/sim/terrain_probe_imported.sim crates/sims/build.rs
git commit -m "feat(sims): terrain_probe_imported smoke fixture importing stdlib materials"
```

---

## Task 12: Runtime-gate smoke test

**Files:**
- Create: `crates/sims/tests/terrain_probe_imported_smoke.rs`

- [ ] **Step 1: Write the test**

```rust
// crates/sims/tests/terrain_probe_imported_smoke.rs
//! Runtime gate: the merged Program's materials come from stdlib via
//! the import system, and the emitted generate_terrain produces a
//! VoxelTerrain reflecting those materials.

use engine_voxel::TerrainGenHandle;

#[test]
fn imported_materials_reach_generate_terrain() {
    let handle = TerrainGenHandle::spawn(42, sims::terrain_probe_imported::generate_terrain);
    let terrain = handle.block_until_ready().expect("gen succeeds");

    // The stdlib basic.sim declares grass (id=1), stone (id=2), sand (id=3)
    // and layer fill { material: stone }. So every cell should be stone (id=2).
    assert_eq!(terrain.extent(), 8);
    assert_eq!(terrain.cell_at(0, 0, 0), 2, "fill from stdlib should be stone");
    assert_eq!(terrain.cell_at(7, 7, 7), 2, "fill from stdlib should be stone");

    // Materials table contains the stdlib materials by id.
    let mtable = terrain.materials();
    let grass = mtable.get(1).expect("grass id=1");
    assert!(grass.walkable);
    let stone = mtable.get(2).expect("stone id=2");
    assert!(!stone.walkable);
    let sand = mtable.get(3).expect("sand id=3");
    assert!((sand.movement_cost - 1.5).abs() < 1e-6);
}
```

- [ ] **Step 2: Run the test**

Run: `WORLDSIM_STDLIB_ROOT=$(pwd)/stdlib cargo test -p sims --test terrain_probe_imported_smoke`
Expected: 1 passed.

If `WORLDSIM_STDLIB_ROOT` is required for the build but not set in a fresh shell, document the requirement in the test header. A long-term solution is to have `crates/sims/build.rs` default to `<workspace>/stdlib` so the env var is optional.

- [ ] **Step 3: Commit**

```bash
git add crates/sims/tests/terrain_probe_imported_smoke.rs
git commit -m "test(sims): runtime-gate smoke — stdlib import reaches emitted terrain"
```

---

## Task 13: `cargo:rerun-if-changed` for resolved imports

**Files:**
- Modify: `crates/sims/build.rs`

- [ ] **Step 1: Note the existing rerun-if-changed pattern**

Run: `rg "rerun-if-changed" crates/sims/build.rs`
Expected: at least two existing entries (`build.rs`, `sims_dir`). The new entries are emitted per fixture.

- [ ] **Step 2: After each `emit_namespaced(f)` call, emit rerun lines**

`crates/sims/build.rs` currently calls `dsl_compiler::build_helper::emit_namespaced(f)` in the per-fixture loop. The `Program.imports_resolved` data is not currently surfaced back to the build.rs — only the emit side-effects are. Two options:

**Option A (recommended):** Extend `emit_namespaced` to return the `Vec<PathBuf>` of contributing paths, OR write them to a sibling file (`OUT_DIR/<fixture>/imports.txt`) the build.rs can read.

**Option B:** Re-parse the fixture from `build.rs` directly using `dsl_compiler::parse_with_imports` to get `imports_resolved`, then emit the rerun lines. Slightly wasteful (double parse) but no API surface change.

Going with B for minimum invasiveness:

```rust
// In the per-fixture loop in crates/sims/build.rs, after emit_namespaced(f):
let sim_path = sims_dir.join(format!("{f}.sim"));
let stdlib_root: PathBuf = match env::var_os("WORLDSIM_STDLIB_ROOT") {
    Some(s) => PathBuf::from(s),
    None    => workspace_root.join("stdlib"),
};
let sandbox_root: PathBuf = match env::var_os("WORLDSIM_SANDBOX_ROOT") {
    Some(s) => PathBuf::from(s),
    None    => workspace_root.clone(),
};
match dsl_compiler::parse_with_imports(&sim_path, &stdlib_root, &sandbox_root) {
    Ok(program) => {
        for p in &program.imports_resolved {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
    Err(_e) => {
        // emit_namespaced will have already panicked or surfaced this;
        // we just skip rerun-if-changed for this fixture here.
    }
}
```

This adds one re-parse per fixture, but `.sim` files are small (typically <1KB) so the cost is negligible at build time.

- [ ] **Step 3: Verify the build still works**

Run: `WORLDSIM_STDLIB_ROOT=$(pwd)/stdlib cargo build -p sims 2>&1 | grep "rerun-if-changed" | head -5`
Expected: per-fixture rerun lines visible in the output (cargo prints `cargo:rerun-if-changed=` lines from build scripts in verbose mode; the `-vv` flag shows them, but just confirming the build succeeds is enough).

Also run: `WORLDSIM_STDLIB_ROOT=$(pwd)/stdlib cargo test -p sims --test terrain_probe_imported_smoke` → still 1 passed.

- [ ] **Step 4: Commit**

```bash
git add crates/sims/build.rs
git commit -m "feat(sims): cargo:rerun-if-changed for resolved import paths"
```

---

## Task 14: Workspace `cargo test` stays green

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `WORLDSIM_STDLIB_ROOT=$(pwd)/stdlib RUST_MIN_STACK=33554432 cargo test --workspace 2>&1 | tail -40`
Expected: all green. New tests in `dsl_ast` (1 file), `dsl_compiler` (6 files), `sims` (1 file) should appear and pass. Pre-existing `dungeon_horde_pin` is a long-running test (~30 min); if running on a time budget, skip with `--exclude` for the sweep.

- [ ] **Step 2: Address any failures**

Common causes:
- `Program { ... }` construction sites missed in Task 1 or Task 8 — find and add `imports: vec![]` / `imports_resolved: vec![]`.
- A fixture in `assets/sim/*.sim` containing the literal token `import` as an identifier — Task 2 made `import` a keyword, so rename if needed.
- Stdlib path not found in tests — the test must set `WORLDSIM_STDLIB_ROOT` or accept the workspace-root default.

- [ ] **Step 3: Commit fixes if any were needed**

```bash
git status
# If clean: nothing to commit.
# If files changed:
git add <files>
git commit -m "fix(multifile-imports): workspace test fallout from import keyword + Program fields"
```

---

## Plan complete — exit criteria

- [ ] `import std/<path>.sim;` and `import ./<path>.sim;` are recognised by the parser.
- [ ] `parse_with_imports` merges files depth-first; transitive imports flatten; diamond imports load once; cycles error.
- [ ] Collisions (duplicate kind+name across files; duplicate `terrain` singleton) produce `DuplicateDefinition` errors.
- [ ] `Program.imports_resolved` lists canonicalised contributing paths.
- [ ] `build_helper::emit_into` uses `parse_with_imports`, reading `WORLDSIM_STDLIB_ROOT` + `WORLDSIM_SANDBOX_ROOT` env vars.
- [ ] `stdlib/materials/basic.sim` exists and is importable.
- [ ] `sims::terrain_probe_imported` builds and its smoke test passes — the runtime-gate post-condition.
- [ ] `cargo:rerun-if-changed` is emitted for each resolved import path.
- [ ] Full workspace `cargo test` passes.

## Follow-up plans (out of scope here)

1. **Separate compilation + runtime linking** — per-module IR artifacts loaded by a runtime linker; mode flag to choose compile-time merge vs runtime link; hot-reload of individual modules. Designed in a separate spec after this plan ships.
2. **Selective imports** — `import std/materials::{grass, stone};` and `import foo as bar;`. Additive to the v1 syntax.
3. **Per-decl source attribution** — currently the collision error's `first`/`second` fields point at file paths, not exact decl spans. Threading source spans through every Decl variant is a larger refactor and a separate plan.
4. **Materials block split from terrain singleton** — so stdlib can share materials without forcing a terrain layout on importers. Requires a small DSL design discussion.
5. **Stdlib content** — populating `stdlib/` with more materials palettes, layer templates, common agent kinds, etc. is per-content effort, not a single plan.
