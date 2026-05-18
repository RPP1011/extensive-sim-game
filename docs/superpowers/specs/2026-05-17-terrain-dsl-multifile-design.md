# DSL Multi-File Imports — Design

**Status:** Design — awaiting user review before plan write-up.
**Date:** 2026-05-17
**Companion plan (TBD):** `docs/superpowers/plans/2026-05-1x-dsl-multifile-imports-*.md`.
**Related shipped specs:**
`docs/superpowers/specs/2026-05-17-terrain-dsl-spec-design.md` —
this design realises that spec's "Multi-file terrain inputs"
follow-up, generalised across the DSL (not terrain-specific).

## Summary

Add a new top-of-file `import <path>;` statement to the DSL. The
parser grows a multi-file-aware entry point that follows imports
recursively and returns a single merged `Program` with all top-level
decls flattened into one scope. Imports resolve against either the
importing file (`./`, `../`) or a build-time-configured stdlib root
(`std/`). Whole-file import semantics — every top-level decl in the
imported file enters the merged scope. Collisions are hard errors.
A new top-level `stdlib/` directory is introduced for shared content
that lives outside `crates/`.

## Motivation

`.sim` files are currently fully self-contained: every fixture
copy-pastes its materials, agent definitions, and rule fragments.
Two pain points:

1. **No reuse across fixtures.** When two scenarios want the same
   creature kinds (e.g. `Wolf`, `Goblin`) or the same materials
   palette, they duplicate the source. Edits diverge.
2. **No shared library.** There's no place to gather curated
   known-good DSL fragments (`std::materials::forest`,
   `std::layers::caves`) that future fixtures can build on.

The original terrain DSL design (2026-05-17) explicitly deferred
multi-file inputs to a follow-up. This is that follow-up — scoped
beyond terrain to the whole DSL, since the import mechanism is
inherently general-purpose.

## Out of scope (this spec)

The brainstorming surfaced a related but distinct goal:
**separate compilation + runtime linking.** Each `.sim` would
compile to a self-contained IR artifact; a runtime loader would
link modules at startup, enabling runtime authoring (edit a `.sim`
while the sim is running, reload the affected module). This is
deferred to **a separate follow-up spec** so it can be designed
with real Phase-1 usage as evidence.

The `import` syntax in this spec is intended to remain unchanged
when the runtime-linking mode lands. Only the back-end mechanism
(parser-time merge vs. per-module artifact + runtime loader) will
differ between the two modes, behind a build-time flag.

## Design

### Architecture & integration

- New top-of-file `import <path>;` statement in the DSL grammar.
- New public parser entry point in `dsl_ast`:

  ```rust
  pub fn parse_with_imports(
      top_path: &std::path::Path,
      stdlib_root: &std::path::Path,
      sandbox_root: &std::path::Path,
  ) -> Result<Program, ParseError>;
  ```

  Reads the top-level file, follows imports recursively, returns a
  single merged `Program` with all top-level decls flattened into
  one scope. `sandbox_root` is the upper bound for `../` traversal
  (defaults to the workspace root for in-tree usage).
- Existing `parse(src: &str) -> Result<Program, ParseError>` is
  unchanged. A file with zero `import` statements behaves
  identically under either entry point.
- Lowering, emission, schema-hash, and runtime stages stay
  oblivious to multi-file — they see a single merged Program.
  Contains the architectural surface to one stage of the pipeline.
- `crates/sims/build.rs` and any other per-runtime `build.rs`
  switches from `dsl_compiler::parse(src)` to
  `parse_with_imports(file_path, stdlib_root)`. Stdlib root is read
  from the env var `WORLDSIM_STDLIB_ROOT`, defaulting to
  `<workspace-root>/stdlib/`.

### Import syntax & resolution

```text
// At the top of any .sim file, before other top-level decls:
import std/materials/forest.sim;
import ./local_overrides.sim;
import ../shared/dungeons.sim;

// Then normal decls (parse error if imports appear after this):
agent Wolf { n: 10 }
terrain { ... }
```

**Resolution rules:**

- `std/<path>` — resolved against the stdlib root (configured
  at parser-entry time, default `<workspace-root>/stdlib/`).
- `./<path>` — resolved relative to the importing file's
  directory.
- `../<path>` — resolved relative to the importing file's
  directory. Parent traversal is allowed but bounded: the
  resolved path must remain within the **sandbox root** — a
  second parameter to `parse_with_imports`, defaulting to the
  workspace root. A canonicalised path outside the sandbox
  produces `ImportError::FileNotFound` with the attempted path
  and the sandbox root in the error message. This prevents
  walk-up escape if untrusted `.sim` is ever loaded.
- Path extension `.sim` is **required** in the import path. No
  extension inference. Keeps imports grep-able.
- Import statements **must precede all other top-level decls**.
  Mixing imports with decls is a parse error
  (`ImportError::ImportAfterDecl`). Keeps the import list
  scannable at the top of the file.

### Merge semantics

- **Depth-first post-order over the import DAG.** Imports of `b`
  are recursively flattened first, then `b`'s own decls; same for
  every imported file; finally the top-level file's own decls.
  Result: imports appear "before" their importers in the merged
  decl order.
- **Transitive flattening.** If `a` imports `b` and `b` imports
  `c`, then parsing `a` produces a merged Program containing
  `c`'s decls, then `b`'s decls, then `a`'s decls (in source
  order within each file).
- **Diamond imports — load once.** If `a` imports `b` and `c`,
  and both `b` and `c` import `d`, then `d` is loaded and parsed
  exactly once. Caching is by canonicalised path identity.
- **Source order within a file preserved.** Stable, deterministic
  output regardless of filesystem-iteration quirks.

### Error handling

| Variant | When |
|---|---|
| `ImportError::FileNotFound { path, attempted_roots }` | Can't resolve `std/…` or `./…` to a real file. |
| `ImportError::Cycle { path_chain }` | A imports B, B imports A (any length cycle). |
| `ImportError::DuplicateDefinition { kind, name, first, second }` | Same kind + name from two files in the merged Program. |
| `ImportError::ImportAfterDecl { file, import_line, prior_decl_line }` | `import` statement appears after a non-import top-level decl. |
| `ImportError::IoError { path, source }` | Filesystem failure during import resolution. |
| `ImportError::Parse { path, inner }` | Wraps a parse error from an imported file, adding the file path. |

All variants integrate into `dsl_ast::error::ParseError` as a new
variant family — same call-site error type, no breaking change for
existing callers.

### Collision detection

After the merge step, a single pass over the merged Program:

1. Build a `HashMap<(Kind, Name), SourcePath>` over all top-level
   decls.
2. Any duplicate insert produces
   `ImportError::DuplicateDefinition { kind, name, first, second }`
   with both source locations.

**`Kind` is namespaced.** Two different kinds may share a name —
`entity Wolf` and `event Wolf` could coexist, since they live in
different kind-namespaces. Collisions are kind-scoped.

**Singleton blocks are kind-scoped duplicates of themselves.** The
shipped terrain spec allows at most one `terrain { ... }` per
program; with imports, this becomes at most one across the merged
program. Two `terrain` blocks reaching the merged Program (even
from different files) is a duplicate-singleton error.

### Stdlib structure

- New top-level directory `stdlib/` at the workspace root.
- Lives outside `crates/`. Checked into git so all developers
  share one stdlib.
- v1 ships an empty stdlib + a README + one trivial example
  fixture under `stdlib/materials/basic.sim` declaring grass /
  stone / sand materials. Later contributions populate it.
- The stdlib path is build-time configurable via the
  `WORLDSIM_STDLIB_ROOT` environment variable; default is
  `<workspace-root>/stdlib/`.

### Build wiring

- `crates/sims/build.rs` (and any other per-runtime `build.rs`)
  resolves the stdlib + sandbox roots once at the top of `main()`:
  ```rust
  let stdlib_root: PathBuf = match env::var_os("WORLDSIM_STDLIB_ROOT") {
      Some(s) => PathBuf::from(s),
      None    => workspace_root.join("stdlib"),
  };
  // sandbox_root is the workspace root in-tree; an env-var override
  // is available for tests that want a tighter or wider sandbox.
  let sandbox_root: PathBuf = match env::var_os("WORLDSIM_SANDBOX_ROOT") {
      Some(s) => PathBuf::from(s),
      None    => workspace_root.clone(),
  };
  ```
- Calls into `dsl_compiler::build_helper::emit_namespaced` (or
  its inner `emit_into`) flow through `parse_with_imports(file_path,
  &stdlib_root, &sandbox_root)` instead of `parse(src)`.
- The merged `Program` returned from `parse_with_imports` carries
  a new `pub imports_resolved: Vec<PathBuf>` field listing every
  file that contributed to the merge (canonicalised paths). Each
  per-runtime `build.rs` iterates and emits one
  `cargo:rerun-if-changed=<path>` per entry. Stdlib edits and
  local-relative-import edits both trigger fixture rebuilds.

### Determinism

- **Path canonicalisation:** resolved paths are canonicalised
  (symlinks followed) before going into `imports_resolved`. Stable
  across symlink-heavy worktrees.
- **Merge order is source-order:** the post-order traversal
  follows AST order; filesystem-iteration order does not enter.
- **No RNG / time / env** in the merge pipeline.

## Testing strategy

| Test | Location | What it pins |
|---|---|---|
| Single-file (no imports) regression | existing `dsl_compiler` tests | `parse(src)` and `parse_with_imports(import-free file)` produce identical Programs. |
| Two-file import | `crates/dsl_compiler/tests/multifile_import_basic.rs` | `a.sim` imports `b.sim`; merged Program contains decls from both, b before a. |
| Stdlib resolution | `crates/dsl_compiler/tests/multifile_import_stdlib.rs` | `import std/materials/basic.sim;` resolves against the stdlib_root parameter. |
| Relative resolution (`./`, `../`) | same file | `./local.sim` and `../shared/foo.sim` resolve against importing file's directory. |
| Transitive flatten | `crates/dsl_compiler/tests/multifile_transitive.rs` | A imports B, B imports C → merged Program has C, B, A decls in that order. |
| Diamond import (single load) | same file | A imports B and C; both import D. D's decls appear exactly once. |
| Cycle detection | `crates/dsl_compiler/tests/multifile_cycle.rs` | A → B → A → `ImportError::Cycle` with the full path chain. |
| Collision: duplicate kind+name | `crates/dsl_compiler/tests/multifile_collision.rs` | Two files defining `entity Wolf` → `DuplicateDefinition` with both source paths. |
| Collision: two terrain blocks | same file | Two imported files each with a `terrain` block → duplicate-singleton error. |
| Different kinds, same name OK | same file | `entity Wolf` + `event Wolf` from different imported files merge cleanly. |
| `import` after a decl is rejected | `crates/dsl_compiler/tests/multifile_import_order.rs` | `agent Foo {}` then `import ...;` → `ImportAfterDecl` error. |
| Missing file | `crates/dsl_compiler/tests/multifile_missing.rs` | `import std/nonexistent.sim;` → `FileNotFound` with both attempted roots listed. |
| `imports_resolved` is canonicalised | `crates/dsl_compiler/tests/multifile_rerun_paths.rs` | Returned `Program.imports_resolved` paths are absolute + canonicalised. |
| Stdlib smoke fixture | `assets/sim/terrain_probe_imported.sim` + smoke test | Real fixture importing `stdlib/materials/basic.sim`, building via `crates/sims/build.rs`, producing a working `sims::terrain_probe_imported::generate_terrain` whose materials come from stdlib. |

**No new parity test (P3).** Imports are a pure parse-time concern,
under the same `@cpu_only` umbrella as the shipped terrain spec.

**No schema-hash bump (P2).** No new `SimState` SoA fields. The
new `Program.imports_resolved` field is a build-time AST property,
not part of runtime state.

## Out of scope (v1)

- **Runtime linking / hot-reload.** Separate compilation of `.sim`
  modules into linkable IR artifacts + a runtime loader. Designed
  in a separate follow-up spec after v1 ships, using real Phase-1
  usage as input.
- **Selective imports.** No `import std::materials::{grass};`
  syntax; whole-file imports only. Can be added later without
  changing v1 syntax.
- **Rename-on-import.** No `import foo.sim as bar;`. Collisions
  must be resolved by editing the upstream definition.
- **Glob imports.** No `import std/**/*.sim;`. Explicit paths
  only.
- **External (over-the-network) imports.** No URL imports. Local
  filesystem only.
- **Mid-file imports.** All imports must precede the first
  non-import decl.
- **Versioning of stdlib content.** v1 stdlib is in-tree; the same
  git commit always pins the same stdlib. A future spec can
  add versioning if stdlib becomes a separate publishable
  artifact.

## Constitution touchpoints (for plan AIS)

- **P1 (Compiler-First):** PASS — the import mechanism lives in
  the parser layer and feeds the existing emitter; no hand-written
  rule logic added.
- **P2 (Schema-Hash):** N/A — no new `SimState` SoA fields.
- **P3 (Cross-Backend Parity):** N/A — pure parse-time concern.
- **P4 (`EffectOp` Size):** N/A — no new event variants.
- **P5 (Determinism via Keyed PCG):** PASS — no RNG entered.
  Merge order is AST-source-order; path canonicalisation makes
  resolution stable across symlink layouts.
- **P6 (Events Are the Mutation Channel):** N/A — no state mutation.
- **P7 (Replayability Flagged):** N/A — no new event variants.
- **P8 (AIS Required):** the implementation plan will carry the
  full AIS template; this design summarises the touchpoints.
- **P10 (No Runtime Panic):** PASS — parse-time errors are
  `Result`s, not panics. `build.rs` panics on import errors,
  which is the expected build-time failure mode (cargo surfaces
  as compile error).
- **P11 (Reduction Determinism):** N/A — no reductions.

## Known risks

- **Stdlib churn.** Once fixtures depend on `stdlib/`, edits to
  stdlib content cascade. The `imports_resolved` rerun mechanism
  catches this for builds, but the impact on review / blame /
  versioning needs ongoing care.
- **Cycle detection cost.** O(depth) stack check per import is
  fine; the implementation must NOT use a HashMap for cycle
  detection (P5-style determinism concern) — use a
  `Vec<PathBuf>` traversal stack instead. Even though this is
  build-time, ahash drift would cascade into error message
  ordering.
- **Path canonicalisation on Windows.** This is a Linux-first
  workspace, but if Windows is ever in scope, path normalisation
  (case-insensitive, separator-normalised) needs extra care.
  Out of scope here.
