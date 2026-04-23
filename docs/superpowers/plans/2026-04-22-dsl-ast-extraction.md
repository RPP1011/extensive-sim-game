# dsl_ast Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the DSL frontend (AST, tokens, parser, IR, name resolution, plus the top-level `parse` / `compile` entrypoints) from `crates/dsl_compiler` into a new `crates/dsl_ast` crate. `dsl_compiler` retains only the emission layer (`emit_*.rs`, `schema_hash.rs`) and re-exports the frontend surface for backward compatibility.

**Architecture:** Pure refactor. Move the parse-and-resolve pipeline (7 source files ~10 KLoC) into a new leaf crate; `dsl_compiler` gains a `dsl_ast = { path = "../dsl_ast" }` dep and re-exports each moved module at the same path it lived before (`dsl_compiler::ast`, `dsl_compiler::ir`, etc.). Internal `use crate::<mod>::*` inside `dsl_compiler` continues to resolve via the re-exports, so `emit_*.rs` does not need file-by-file import rewrites. Downstream consumers (root `game` crate / xtask, `engine_gpu` tests) continue using `dsl_compiler::*` unchanged.

**Tech Stack:** Rust 2021 workspace. No new dependencies — `dsl_ast` uses `serde` (already a dsl_compiler dep). `winnow` and `sha2` are not needed in `dsl_ast`; the parser is hand-rolled recursive descent and `schema_hash.rs` stays in `dsl_compiler`.

**Prerequisites:** None. This is the first plan for the DSL Authoring Engine (spec: `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1, §10 step 1). Must land before any interpreter work (P1b+) begins.

---

## File Structure

### New files

```
crates/dsl_ast/
├── Cargo.toml                  (new, workspace member)
└── src/
    ├── lib.rs                  (new; re-exports + top-level compile())
    ├── ast.rs                  (moved from dsl_compiler/src/ast.rs, unchanged)
    ├── tokens.rs               (moved from dsl_compiler/src/tokens.rs, unchanged)
    ├── error.rs                (moved from dsl_compiler/src/error.rs, unchanged)
    ├── resolve_error.rs        (moved from dsl_compiler/src/resolve_error.rs, unchanged)
    ├── ir.rs                   (moved from dsl_compiler/src/ir.rs, unchanged)
    ├── parser.rs               (moved from dsl_compiler/src/parser.rs, unchanged)
    └── resolve.rs              (moved from dsl_compiler/src/resolve.rs, unchanged)
```

### Modified files

- `Cargo.toml` (root): add `"crates/dsl_ast"` to `[workspace].members`.
- `crates/dsl_compiler/Cargo.toml`: add `dsl_ast = { path = "../dsl_ast" }` dep.
- `crates/dsl_compiler/src/lib.rs`: replace `pub mod <name>;` declarations
  for the 7 moved modules with `pub use dsl_ast::<name>;` re-exports; move
  the body of `parse` / `compile_ast` / `compile` / `CompileError` into
  `dsl_ast/src/lib.rs` and re-export them from `dsl_compiler/src/lib.rs`.

### Deleted files

- `crates/dsl_compiler/src/ast.rs`
- `crates/dsl_compiler/src/tokens.rs`
- `crates/dsl_compiler/src/error.rs`
- `crates/dsl_compiler/src/resolve_error.rs`
- `crates/dsl_compiler/src/ir.rs`
- `crates/dsl_compiler/src/parser.rs`
- `crates/dsl_compiler/src/resolve.rs`

### Out of scope

- `schema_hash.rs` stays in `dsl_compiler` — it depends on
  `emit_entity::schema_hash_input`, which is emitter-adjacent. Extracting
  it cleanly is a follow-up refactor, not part of this plan.
- All `emit_*.rs` files stay in `dsl_compiler`.
- No behavior changes. No new public API beyond the re-export surface.

---

## Acceptance criteria (plan-level)

1. **`crates/dsl_ast` builds clean.** `cargo check -p dsl_ast` passes.
2. **`dsl_compiler` still builds clean.** `cargo check -p dsl_compiler` passes.
3. **Workspace builds clean.** `cargo check --workspace` passes.
4. **All existing `dsl_compiler` tests pass, unchanged.** `cargo test -p dsl_compiler` has the same green-test count as before the refactor.
5. **`engine_gpu` parity tests still pass.** `cargo test -p engine_gpu` green.
6. **`xtask` builds.** `cargo build --bin xtask` green.
7. **No file-by-file import rewrites inside `dsl_compiler/src/emit_*.rs`.** Internal `use crate::<mod>::*` statements resolve via re-exports at the crate root. If this constraint forces an exception, document it in the task where it arises.
8. **`dsl_compiler/src/` contains no frontend files.** Only `lib.rs`, `schema_hash.rs`, and `emit_*.rs`.
9. **Downstream API surface preserved.** Every symbol that was previously accessible as `dsl_compiler::<path>::<item>` remains accessible at the same path. Spot-check with the test fixtures referenced in `tests/ir_golden.rs` (e.g., `dsl_compiler::ir::DecayUnit`, `dsl_compiler::ir::ViewKind`, `dsl_compiler::CompileError`, `dsl_compiler::ResolveError`).

If any of (1)–(9) fails, the plan is not done.

---

## Task 1: Scaffold the `dsl_ast` crate

**Files:**
- Create: `crates/dsl_ast/Cargo.toml`
- Create: `crates/dsl_ast/src/lib.rs`
- Modify: `Cargo.toml` (root, workspace members list)

- [ ] **Step 1.1: Create the crate manifest**

```toml
# crates/dsl_ast/Cargo.toml
[package]
name = "dsl_ast"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
```

- [ ] **Step 1.2: Create the crate lib.rs as an empty placeholder**

```rust
// crates/dsl_ast/src/lib.rs
//! World Sim DSL frontend: parser, AST, typed IR, and name resolution.
//!
//! Extracted from `dsl_compiler` so both the emitter crate and the engine
//! interpreter can share one parse + resolve pipeline. See
//! `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1.
```

- [ ] **Step 1.3: Register the crate in the workspace**

Open `Cargo.toml` at the repo root. Locate the line:

```toml
[workspace]
members = [".", "crates/tactical_sim", "crates/engine", "crates/engine_generated", "crates/engine_rules", "crates/engine_gpu", "crates/viz", "crates/dsl_compiler"]
```

Add `"crates/dsl_ast"` to the list (alphabetical order is fine):

```toml
[workspace]
members = [".", "crates/tactical_sim", "crates/engine", "crates/engine_generated", "crates/engine_rules", "crates/engine_gpu", "crates/viz", "crates/dsl_ast", "crates/dsl_compiler"]
```

- [ ] **Step 1.4: Verify the new crate compiles**

Run: `cargo check -p dsl_ast`
Expected: `Finished ...` with no errors. A warning about empty lib is fine.

- [ ] **Step 1.5: Commit**

```bash
git add Cargo.toml crates/dsl_ast/
git commit -m "refactor(dsl_ast): scaffold empty crate in workspace

Prerequisite for extracting the DSL frontend from dsl_compiler.
See docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Move `tokens.rs` and `ast.rs` to `dsl_ast`

These two files have no `crate::` imports (both are leaf modules within
`dsl_compiler`), so they move cleanest. Everything else depends on them.

**Files:**
- Move: `crates/dsl_compiler/src/tokens.rs` → `crates/dsl_ast/src/tokens.rs`
- Move: `crates/dsl_compiler/src/ast.rs` → `crates/dsl_ast/src/ast.rs`
- Modify: `crates/dsl_ast/src/lib.rs` (add module decls + re-exports)
- Modify: `crates/dsl_compiler/Cargo.toml` (add `dsl_ast` dep)
- Modify: `crates/dsl_compiler/src/lib.rs` (replace `pub mod ast;` + `pub mod tokens;` with re-exports)

- [ ] **Step 2.1: Move the files via git**

Run:
```bash
git mv crates/dsl_compiler/src/tokens.rs crates/dsl_ast/src/tokens.rs
git mv crates/dsl_compiler/src/ast.rs crates/dsl_ast/src/ast.rs
```

Expected: git records the moves; files are now under `dsl_ast/src/`.

- [ ] **Step 2.2: Register the moved modules in `dsl_ast/src/lib.rs`**

Replace the contents of `crates/dsl_ast/src/lib.rs` with:

```rust
//! World Sim DSL frontend: parser, AST, typed IR, and name resolution.
//!
//! Extracted from `dsl_compiler` so both the emitter crate and the engine
//! interpreter can share one parse + resolve pipeline. See
//! `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1.

pub mod ast;
pub mod tokens;
```

- [ ] **Step 2.3: Add `dsl_ast` as a dependency of `dsl_compiler`**

Open `crates/dsl_compiler/Cargo.toml` and add under `[dependencies]`:

```toml
dsl_ast = { path = "../dsl_ast" }
```

So the `[dependencies]` block reads (keep other entries as-is):

```toml
[dependencies]
dsl_ast = { path = "../dsl_ast" }
winnow = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
sha2 = "0.10"
```

- [ ] **Step 2.4: Re-export the moved modules from `dsl_compiler/src/lib.rs`**

Open `crates/dsl_compiler/src/lib.rs`. Locate the `pub mod` declarations (around lines 11–31). Replace:

```rust
pub mod ast;
```

with:

```rust
pub use dsl_ast::ast;
```

And replace:

```rust
pub mod tokens;
```

with:

```rust
pub use dsl_ast::tokens;
```

Leave all other `pub mod` declarations untouched. The top-level re-exports that already exist (`pub use ast::{Decl, Program, Span, Spanned};`) keep working because `ast` is now a re-exported module at the crate root.

- [ ] **Step 2.5: Verify dsl_ast builds**

Run: `cargo check -p dsl_ast`
Expected: `Finished ...` no errors.

- [ ] **Step 2.6: Verify dsl_compiler builds and tests pass**

Run: `cargo check -p dsl_compiler`
Expected: clean build; `use crate::ast::*` inside `error.rs`, `ir.rs`, `parser.rs`, `resolve.rs`, `resolve_error.rs`, and `emit_*.rs` all resolve through the re-export.

Run: `cargo test -p dsl_compiler`
Expected: all existing tests green. If any test fails, fix the re-export in `dsl_compiler/src/lib.rs` before moving on; do not edit other files.

- [ ] **Step 2.7: Commit**

```bash
git add crates/dsl_ast/ crates/dsl_compiler/Cargo.toml crates/dsl_compiler/src/lib.rs
git commit -m "refactor(dsl_ast): move ast + tokens out of dsl_compiler

Pure move. dsl_compiler re-exports ast and tokens at the same
paths so internal uses and downstream consumers are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Move `error.rs` and `resolve_error.rs`

These are the error types for parsing and name resolution. Both depend
only on `ast::Span`, which is now in `dsl_ast`.

**Files:**
- Move: `crates/dsl_compiler/src/error.rs` → `crates/dsl_ast/src/error.rs`
- Move: `crates/dsl_compiler/src/resolve_error.rs` → `crates/dsl_ast/src/resolve_error.rs`
- Modify: `crates/dsl_ast/src/lib.rs` (add module decls + keep any existing re-exports)
- Modify: `crates/dsl_compiler/src/lib.rs` (replace `pub mod error;` / `pub mod resolve_error;` with re-exports)

- [ ] **Step 3.1: Move the files**

Run:
```bash
git mv crates/dsl_compiler/src/error.rs crates/dsl_ast/src/error.rs
git mv crates/dsl_compiler/src/resolve_error.rs crates/dsl_ast/src/resolve_error.rs
```

- [ ] **Step 3.2: Register the moved modules in `dsl_ast/src/lib.rs`**

Append to `crates/dsl_ast/src/lib.rs`:

```rust
pub mod error;
pub mod resolve_error;
```

The file after this step should read:

```rust
//! World Sim DSL frontend: parser, AST, typed IR, and name resolution.
//!
//! Extracted from `dsl_compiler` so both the emitter crate and the engine
//! interpreter can share one parse + resolve pipeline. See
//! `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1.

pub mod ast;
pub mod tokens;
pub mod error;
pub mod resolve_error;
```

Verify `crate::ast::Span` resolves inside the moved files. Inside `dsl_ast/src/error.rs` and `dsl_ast/src/resolve_error.rs` the existing `use crate::ast::Span;` now refers to `dsl_ast::ast::Span` — no edit needed because both files are in the same crate.

- [ ] **Step 3.3: Re-export the modules from `dsl_compiler/src/lib.rs`**

In `crates/dsl_compiler/src/lib.rs`, replace:

```rust
pub mod error;
```

with:

```rust
pub use dsl_ast::error;
```

And replace:

```rust
pub mod resolve_error;
```

with:

```rust
pub use dsl_ast::resolve_error;
```

Leave the existing `pub use error::ParseError;` and `pub use resolve_error::ResolveError;` lines at the crate root untouched — they resolve through the re-exported modules.

- [ ] **Step 3.4: Verify**

Run: `cargo check -p dsl_ast` — passes.
Run: `cargo check -p dsl_compiler` — passes.
Run: `cargo test -p dsl_compiler` — green.

- [ ] **Step 3.5: Commit**

```bash
git add crates/dsl_ast/src/lib.rs crates/dsl_ast/src/error.rs crates/dsl_ast/src/resolve_error.rs crates/dsl_compiler/src/lib.rs
git commit -m "refactor(dsl_ast): move error + resolve_error out of dsl_compiler

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Move `ir.rs`

The typed IR depends only on `ast::{Annotation, BinOp, QuantKind, Span, UnOp}`.

**Files:**
- Move: `crates/dsl_compiler/src/ir.rs` → `crates/dsl_ast/src/ir.rs`
- Modify: `crates/dsl_ast/src/lib.rs`
- Modify: `crates/dsl_compiler/src/lib.rs`

- [ ] **Step 4.1: Move the file**

```bash
git mv crates/dsl_compiler/src/ir.rs crates/dsl_ast/src/ir.rs
```

- [ ] **Step 4.2: Register in `dsl_ast/src/lib.rs`**

Append `pub mod ir;` to `crates/dsl_ast/src/lib.rs`. File after:

```rust
//! World Sim DSL frontend: parser, AST, typed IR, and name resolution.
//!
//! Extracted from `dsl_compiler` so both the emitter crate and the engine
//! interpreter can share one parse + resolve pipeline. See
//! `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1.

pub mod ast;
pub mod tokens;
pub mod error;
pub mod resolve_error;
pub mod ir;
```

- [ ] **Step 4.3: Re-export from `dsl_compiler/src/lib.rs`**

Replace:

```rust
pub mod ir;
```

with:

```rust
pub use dsl_ast::ir;
```

The existing `pub use ir::Compilation;` at the crate root resolves via the re-exported module.

- [ ] **Step 4.4: Verify**

Run: `cargo check -p dsl_ast` — passes.
Run: `cargo check -p dsl_compiler` — passes.

Note: `dsl_compiler/src/schema_hash.rs` imports `crate::ir::{...}`; this resolves through the re-export. `dsl_compiler/src/emit_rust.rs` imports `crate::ir::{EventField, EventIR, IrType}`; same story. If either fails to compile, inspect the compiler error — do not edit `emit_*.rs` or `schema_hash.rs` to work around it; fix the re-export instead.

Run: `cargo test -p dsl_compiler` — green.

- [ ] **Step 4.5: Commit**

```bash
git add crates/dsl_ast/src/lib.rs crates/dsl_ast/src/ir.rs crates/dsl_compiler/src/lib.rs
git commit -m "refactor(dsl_ast): move ir out of dsl_compiler

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Move `parser.rs`

The parser depends on `crate::ast::*`, `crate::error::ParseError`,
`crate::tokens::{is_ident_cont, is_ident_start, unicode_op_ascii, Cursor}`
— all now in `dsl_ast`, so `crate::` inside `dsl_ast` resolves them
naturally.

**Files:**
- Move: `crates/dsl_compiler/src/parser.rs` → `crates/dsl_ast/src/parser.rs`
- Modify: `crates/dsl_ast/src/lib.rs`
- Modify: `crates/dsl_compiler/src/lib.rs`

- [ ] **Step 5.1: Move the file**

```bash
git mv crates/dsl_compiler/src/parser.rs crates/dsl_ast/src/parser.rs
```

- [ ] **Step 5.2: Register in `dsl_ast/src/lib.rs`**

Append `pub mod parser;` and add a top-level `parse` convenience so callers can write `dsl_ast::parse(src)`. File after:

```rust
//! World Sim DSL frontend: parser, AST, typed IR, and name resolution.
//!
//! Extracted from `dsl_compiler` so both the emitter crate and the engine
//! interpreter can share one parse + resolve pipeline. See
//! `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1.

pub mod ast;
pub mod tokens;
pub mod error;
pub mod resolve_error;
pub mod ir;
pub mod parser;

pub use ast::{Decl, Program, Span, Spanned};
pub use error::ParseError;
pub use resolve_error::ResolveError;

/// Parse a DSL source string into a `Program` AST.
pub fn parse(source: &str) -> Result<Program, ParseError> {
    parser::parse_program(source)
}
```

- [ ] **Step 5.3: Re-export from `dsl_compiler/src/lib.rs`**

Replace:

```rust
pub mod parser;
```

with:

```rust
pub use dsl_ast::parser;
```

The top-level `pub fn parse(source: &str) -> Result<Program, ParseError>` definition inside `dsl_compiler/src/lib.rs` can stay — it still delegates to `parser::parse_program(source)` via the re-export. (It becomes a thin wrapper around `dsl_ast::parse`; Task 7 collapses it.)

- [ ] **Step 5.4: Verify**

Run: `cargo check -p dsl_ast` — passes.
Run: `cargo check -p dsl_compiler` — passes.
Run: `cargo test -p dsl_compiler` — green.

- [ ] **Step 5.5: Commit**

```bash
git add crates/dsl_ast/src/lib.rs crates/dsl_ast/src/parser.rs crates/dsl_compiler/src/lib.rs
git commit -m "refactor(dsl_ast): move parser out of dsl_compiler

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Move `resolve.rs`

The resolver depends on `crate::ast::{...}`, `crate::ir::*`, and
`crate::resolve_error::ResolveError` — all now in `dsl_ast`.

**Files:**
- Move: `crates/dsl_compiler/src/resolve.rs` → `crates/dsl_ast/src/resolve.rs`
- Modify: `crates/dsl_ast/src/lib.rs`
- Modify: `crates/dsl_compiler/src/lib.rs`

- [ ] **Step 6.1: Move the file**

```bash
git mv crates/dsl_compiler/src/resolve.rs crates/dsl_ast/src/resolve.rs
```

- [ ] **Step 6.2: Register in `dsl_ast/src/lib.rs`** and expose `compile_ast`

Append `pub mod resolve;` plus a `compile_ast` convenience. File after:

```rust
//! World Sim DSL frontend: parser, AST, typed IR, and name resolution.
//!
//! Extracted from `dsl_compiler` so both the emitter crate and the engine
//! interpreter can share one parse + resolve pipeline. See
//! `docs/superpowers/specs/2026-04-22-dsl-authoring-engine-design.md` §4.1.

pub mod ast;
pub mod tokens;
pub mod error;
pub mod resolve_error;
pub mod ir;
pub mod parser;
pub mod resolve;

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
```

- [ ] **Step 6.3: Re-export from `dsl_compiler/src/lib.rs`**

Replace:

```rust
pub mod resolve;
```

with:

```rust
pub use dsl_ast::resolve;
```

- [ ] **Step 6.4: Verify**

Run: `cargo check -p dsl_ast` — passes.
Run: `cargo check -p dsl_compiler` — passes.
Run: `cargo test -p dsl_compiler` — green.

- [ ] **Step 6.5: Commit**

```bash
git add crates/dsl_ast/src/lib.rs crates/dsl_ast/src/resolve.rs crates/dsl_compiler/src/lib.rs
git commit -m "refactor(dsl_ast): move resolve out of dsl_compiler

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Move top-level `compile()` and `CompileError` into `dsl_ast`

With all seven frontend files now in `dsl_ast`, the top-level convenience
functions (`compile`, `CompileError`) that glue parse + resolve belong in
`dsl_ast` too. `dsl_compiler` keeps its own top-level surface by
re-exporting from `dsl_ast`.

**Files:**
- Modify: `crates/dsl_ast/src/lib.rs` (add `compile` and `CompileError`)
- Modify: `crates/dsl_compiler/src/lib.rs` (remove local `compile` / `CompileError` defs; re-export from `dsl_ast`)

- [ ] **Step 7.1: Add `compile` and `CompileError` to `dsl_ast/src/lib.rs`**

Append the following to `crates/dsl_ast/src/lib.rs`:

```rust
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
```

The final `dsl_ast/src/lib.rs` now owns the full frontend API:
`parse`, `compile_ast`, `compile`, `CompileError`, `ParseError`,
`ResolveError`, `Compilation`, `Decl`, `Program`, `Span`, `Spanned`.

- [ ] **Step 7.2: Collapse `dsl_compiler/src/lib.rs` to re-export from `dsl_ast`**

Open `crates/dsl_compiler/src/lib.rs`. Locate the local definitions of
`parse`, `compile_ast`, `compile`, and `CompileError` (lines roughly
39–69 of the original file). Delete them.

Locate the top-level re-exports:

```rust
pub use ast::{Decl, Program, Span, Spanned};
pub use error::ParseError;
pub use ir::Compilation;
pub use resolve_error::ResolveError;
```

These still resolve correctly because `ast`, `error`, `ir`,
`resolve_error` are all re-exported modules from `dsl_ast`. Leave them
as-is.

Add at the same section:

```rust
pub use dsl_ast::{compile, compile_ast, parse, CompileError};
```

The file should now contain, roughly: the doc-comment header, the
module-re-exports for `ast`, `tokens`, `error`, `resolve_error`, `ir`,
`parser`, `resolve`, the `pub mod` declarations for the emit_* /
schema_hash files, the top-level symbol re-exports including
`dsl_ast::{compile, compile_ast, parse, CompileError}`, and the
`EmittedArtifacts` struct + `emit` / `emit_with_source` /
`emit_with_per_kind_sources` / `snake_case` / `config_snake_case`
definitions. Nothing else.

- [ ] **Step 7.3: Verify downstream consumers still see the same surface**

Run: `cargo check -p dsl_ast` — passes.
Run: `cargo check -p dsl_compiler` — passes.
Run: `cargo test -p dsl_compiler` — green. Specifically,
`tests/ir_golden.rs` exercises `dsl_compiler::compile`,
`dsl_compiler::parse`, `dsl_compiler::CompileError`, and
`dsl_compiler::ResolveError` — all must remain addressable at those paths.

- [ ] **Step 7.4: Commit**

```bash
git add crates/dsl_ast/src/lib.rs crates/dsl_compiler/src/lib.rs
git commit -m "refactor(dsl_ast): own the top-level compile() / CompileError

dsl_compiler now re-exports compile / compile_ast / parse / CompileError
from dsl_ast. Downstream callers using dsl_compiler::{compile, parse,
CompileError} are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Full-workspace validation

Extraction is complete. This task runs the full workspace build and test
matrix to confirm nothing downstream broke, and documents the final
state of `dsl_compiler/src/`.

**Files:**
- No edits. Verification only.

- [ ] **Step 8.1: Confirm `dsl_compiler/src/` is frontend-clean**

Run: `ls crates/dsl_compiler/src/`

Expected output (order may vary):

```
emit_config.rs
emit_entity.rs
emit_enum.rs
emit_mask.rs
emit_mask_wgsl.rs
emit_physics.rs
emit_physics_wgsl.rs
emit_python.rs
emit_rust.rs
emit_scoring.rs
emit_scoring_wgsl.rs
emit_view.rs
emit_view_wgsl.rs
lib.rs
schema_hash.rs
```

If any of `ast.rs`, `tokens.rs`, `error.rs`, `resolve_error.rs`, `ir.rs`, `parser.rs`, or `resolve.rs` still exists under `crates/dsl_compiler/src/`, something went wrong — go back to the task that was supposed to move it.

- [ ] **Step 8.2: Confirm `dsl_ast/src/` has the extracted files**

Run: `ls crates/dsl_ast/src/`

Expected output:

```
ast.rs
error.rs
ir.rs
lib.rs
parser.rs
resolve.rs
resolve_error.rs
tokens.rs
```

- [ ] **Step 8.3: Full workspace build**

Run: `cargo check --workspace`
Expected: clean compile, no errors. Warnings about unused deps in `dsl_compiler` (e.g., `winnow` is now unused in `dsl_compiler`) are acceptable — the cleanup is a follow-up, and does not affect correctness.

- [ ] **Step 8.4: Full workspace test**

Run: `cargo test --workspace`
Expected: all tests pass. Specifically:
  - `dsl_compiler`: every test previously green is still green.
  - `engine_gpu`: `physics_parity` and `cascade_parity` still green (they call `dsl_compiler::compile`).
  - `engine`: all tests green — they do not import `dsl_compiler` directly but link via generated code.
  - Root `game` crate: green.

If any test fails, compare the failure to the baseline by checking out
the commit immediately before this plan started (`4b2020b3` or
equivalent). A failure that reproduces on the baseline is pre-existing
and not a regression from this plan. A failure that only occurs after
extraction is a real bug and must be fixed before this task closes.

- [ ] **Step 8.5: xtask smoke test**

Run: `cargo build --bin xtask`
Expected: clean build. The xtask's `compile-dsl` subcommand uses
`dsl_compiler::compile` + `dsl_compiler::emit`; the re-exports from
Task 7 keep both paths live.

- [ ] **Step 8.6: Remove the `winnow` dep from `dsl_compiler` if unused**

Run: `grep -rn "winnow" crates/dsl_compiler/src/`
Expected: no results (the parser moved to `dsl_ast` and does not depend on winnow; `dsl_compiler` itself never used it).

If the grep returns empty: edit `crates/dsl_compiler/Cargo.toml` and remove the `winnow = "0.7"` line. Run `cargo check -p dsl_compiler`; it should still pass.

If the grep returns any result: leave the dep in place and note which file uses it — a follow-up can remove it.

- [ ] **Step 8.7: Commit the dep cleanup (only if Step 8.6 removed winnow)**

```bash
git add crates/dsl_compiler/Cargo.toml
git commit -m "chore(dsl_compiler): drop unused winnow dep after frontend extraction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8.8: Verify final state**

Run: `git log --oneline -20`
Expected: 7 commits from this plan (Tasks 1–7) plus the optional Task 8.7 cleanup, on top of the pre-plan HEAD.

Run: `git status`
Expected: clean working tree.

Plan complete.
