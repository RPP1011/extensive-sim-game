# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this crate is

The DSL *frontend*: source text → AST → typed IR. It owns tokenizing, parsing, and name
resolution for both DSL surfaces in this repo — the `.sim` simulation DSL (`assets/sim/*.sim`)
and the `.ability` sub-language (`dataset/abilities/lol_heroes/*.ability`). It does **not**
generate Rust, WGSL, or Python — that's `dsl_compiler`'s job. Canonical grammar/semantics spec:
`docs/spec/dsl.md` (§1.2 lists the ten-ish top-level declaration kinds; §9 is the compiler
architecture this crate implements the front half of).

**Boundary with `dsl_compiler`** (confirmed via `grep dsl_ast:: crates/dsl_compiler/src`):
`dsl_compiler` depends on `dsl_ast` and consumes `dsl_ast::parse` / `parse_with_imports`,
`dsl_ast::ast::*` types, and — mostly — `dsl_ast::ir::*` (the typed `Compilation`, `IrExprNode`,
`IrStmt`, `IrType`, `LocalRef`, `NamespaceId`, `ViewKind`, `StorageHint`, `Packing`, …). Its own
`cg/lower/*` and `cg/emit/*` modules take that IR and lower/emit it further into Rust, WGSL, and
Python — a second, codegen-focused IR layer that lives entirely in `dsl_compiler`, not here.
Everything upstream of `ir::Compilation` (tokens, AST, name resolution) lives in this crate.

There is a second, independent consumer: `engine` depends on `dsl_ast` directly (see
`crates/engine/Cargo.toml`) for the `eval` module's `ReadContext` / `CascadeContext` /
`ViewContext` traits, used by `engine/src/evaluator`, `engine/src/cascade`, and
`engine/src/policy`. This is a tree-walking-interpreter contract, unrelated to `dsl_compiler`'s
codegen path — don't assume `dsl_compiler` is the only downstream reader of this crate.

## Commands

```bash
cargo build -p dsl_ast
cargo test -p dsl_ast                                   # all unit + integration tests
cargo test -p dsl_ast --test ability_parser_wave_1_5     # one integration test file (tests/*.rs)
cargo test -p dsl_ast --test ability_parser_wave_1_5 some_test_fn
cargo test -p dsl_ast some_test_fn                       # by name, across files
```
No crate-specific test harness quirks; the workspace-root `cargo test -- --test-threads=1` advice
(for determinism tests) doesn't apply here — this crate is pure parsing/resolution, no simulation
state.

## Architecture

Pipeline: `tokens.rs` (`Cursor`, byte-offset lexer helpers) → `parser.rs` / `ability_parser.rs`
(AST) → `resolve.rs` (typed IR). Both parsers are **hand-rolled recursive descent** — no
combinator library (`dsl_ast`'s own `Cargo.toml` has no parser-combinator dep; `winnow` is a
`dsl_compiler`-only dependency, unrelated to this crate's parsing).

- **Two separate grammars, two entry points, no shared `Program` type.** `parser::parse_program`
  (source → `ast::Program`) parses `.sim` files; `ability_parser::parse_ability_file` (source →
  `ast::AbilityFile`) parses `.ability` files. They share the `ast.rs` module (spans, exprs, some
  leaf types) but are otherwise independent — don't assume a helper in one applies to the other.
- **Declaration-kind mapping is 1:1 by construction.** Every top-level `.sim` decl kind in
  `docs/spec/dsl.md` §1.2 (`entity`, `event`, `view`, `physics`, `mask`, `verb`, `scoring`,
  `invariant`, `probe`, `metric`, `config`, `spatial_query`, plus `belief`, `terrain`, `table`,
  region/index decls…) has: an AST variant in `ast::Decl` (parsed by a `parse_<kind>_decl`
  function in `parser.rs`, e.g. `parse_entity_fields`, `parse_view_body`, `parse_physics_handler`,
  `parse_scoring_entry`, `parse_metric_decl`), and — after `resolve::resolve` — a same-shaped
  `*IR` struct collected into a flat `Vec<*IR>` field on `ir::Compilation` (e.g.
  `Compilation::views: Vec<ViewIR>`), addressed elsewhere via a typed `*Ref(u16)` newtype
  (`ViewRef`, `PhysicsRef`, `MaskRef`, …; `ref_newtype!` macro in `ir.rs`). To see how one
  declaration kind actually lowers, trace `parse_<kind>_decl` in `parser.rs` → the matching arm in
  `resolve.rs` → the `*IR` struct definition in `ir.rs`; there is no other indirection.
- **`resolve.rs` is two-pass** (module doc, top of file): pass 1 collects every top-level decl
  name into a `SymbolTable` and assigns IR indices (duplicate same-kind names error); pass 2 walks
  each decl body resolving identifiers against a local scope stack + stdlib symbol table + the
  pass-1 top-level table. Unresolvable call callees become `UnresolvedCall` (deferred to a later
  compiler milestone); bare unresolved identifiers are hard errors.
- **Two distinct error types, deliberately different shapes.** `error::ParseError` carries a
  rendered, ready-to-print caret-pointer string (built eagerly in `ParseError::new` via
  `render()`) plus a `context: Vec<String>` breadcrumb trail (outer rule → inner rule, printed
  innermost-first). `resolve_error::ResolveError` is a plain enum of typed variants (span-pinned,
  no pre-rendering) — rendering is left to the caller. Don't conflate the two or assume
  `ResolveError` has a `Display` impl doing the same thing `ParseError` does.
- **`eval/` is a separate concern from parsing/resolution**: engine-agnostic interpreter context
  traits (`ReadContext`/`CascadeContext`/`ViewContext`, split along the read/mutate axis per rule
  class — see the module doc in `eval/mod.rs`) plus local mirrors of engine ID types (`AgentId`,
  `Vec3`, `EffectOp`) that must stay byte/discriminant-compatible with the real `engine` types by
  hand (no compile-time assertion exists yet; `eval/mod.rs` has a TODO for one). This exists
  because `dsl_ast` must not depend on `engine`.
- **`engine_events.rs`** is a manually-synced three-way table: DSL event names that alias
  hardcoded engine `EventKindId` discriminants (`crates/engine/src/cascade/handler.rs`). It must
  track both that enum *and* `EFFECT_KIND_TO_EVENT_KIND_ID` in
  `crates/dsl_compiler/src/cg/emit/wgsl_body.rs`; `assign_event_kind_ids` is the single allocator
  every consumer (resolver + `dsl_compiler`'s lowering driver) is required to call — see the file
  doc comment for the reserved-ID-skipping policy.
- **`ability_emit.rs`** is the inverse of `ability_parser.rs` (`AbilityDecl` → source text), used
  for round-trip grammar-coverage testing (`tests/ability_grammar_walker.rs`) and as groundwork
  for future hot-reload. It only emits the surface the parser fully models — opaque blocks
  (`deliver`, `morph`, `template`, `structure`, `program`) round-trip as raw strings, not emitted
  structurally.

## Non-obvious things

- `Program.imports_resolved` (canonicalized paths of every `.sim` file that contributed to a
  program) is populated only by `dsl_compiler::imports`' multi-file path, not by the bare
  `parse(source)` / `parser::parse_program` entry point — it's always empty when you parse a
  single string directly.
- Per `docs/spec/dsl.md` §1.2, `query` declarations parse successfully (`Decl::Query`) but are
  **silently dropped during resolve** — no `QueryIR` exists on `Compilation`. Don't be surprised
  when a `query` decl in a fixture has no downstream effect; this is documented drift (audit U6),
  not a bug to fix opportunistically.
- `event_tag` and `enum` are parsed but have minimal real usage (`event_tag` has zero shipped
  uses in `assets/sim/` per the same audit) — grammar exists ahead of feature adoption.
- The `tests/` directory is organized one file per feature/milestone (e.g.
  `ability_parser_wave_1_1.rs` … `_1_5.rs`, `cast_lifecycle_event_aliases.rs`,
  `field_decl_annotation_boundary.rs`) rather than one broad `parser_tests.rs` — follow that
  convention when adding coverage for a new grammar slice rather than appending to an unrelated
  file.
- `ast.rs`'s own header comment states the AST is "deliberately verbose and one-variant-per-shape"
  — don't try to collapse similar-looking `Decl`/`Stmt`/`Expr` variants for DRY's sake; the
  verbosity is intentional so each shape stays independently evolvable.
