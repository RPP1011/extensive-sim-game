# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## What this is

Shared, importable fragments of the `.sim` DSL (see `crates/dsl_compiler`, `crates/dsl_ast`). Each file is a normal `.sim` snippet, not a special format — `materials/basic.sim` declares a `terrain { materials { ... } }` block, `rules/chase.sim` declares a generic `physics chase(target, aggro, speed)` rule. Any fixture under `assets/sim/*.sim` can pull one in with `import std/<path>.sim;` (the `std/` prefix specifically means "resolve against the stdlib root," as opposed to a local-relative import).

Read `stdlib/README.md` first — it's short and already covers the import-scope conventions (whole-file import, names must be unique across a fixture's merged Program, collisions are compile errors). Don't duplicate it here.

## What consumes it

- **Resolution**: `WORLDSIM_STDLIB_ROOT` env var, defaulting to `<workspace-root>/stdlib/`. Set in two places with identical fallback logic: `crates/dsl_compiler/src/build_helper.rs` (`emit_into`) and `crates/sims/build.rs` (mirrors it so stdlib edits trigger incremental rebuilds via a second parse pass).
- **Import parsing**: `crates/dsl_compiler/src/imports.rs` — the `std/<rest>` prefix strip-and-resolve happens here.
- **Actual importers today**: only two fixtures use it —
  - `assets/sim/param_rule_smoke.sim` → `import std/rules/chase.sim;`
  - `assets/sim/terrain_probe_imported.sim` → `import std/materials/basic.sim;`
- Compiled in as part of the normal `crates/sims` build.rs fixture pipeline — there's no separate stdlib build step or crate.
- Design doc: `docs/superpowers/specs/2026-05-17-terrain-dsl-multifile-design.md`.

## Status

Live and actively wired into the build (build.rs on both `dsl_compiler` and `sims` reference it, and real fixtures import from it), but the corpus itself is thin — 2 files, 2 consumers. Treat additions here the same as any other `.sim` source: they get compiled and must parse/lower/emit cleanly (errors surface during `cargo check` via the consuming fixture's build.rs, per the workspace root CLAUDE.md).

## Non-obvious

- `chase.sim`'s rule body is intentionally inert (`_aggro_squared`, `_target_kind`, `_move_speed` are just param refs bound to unused `let`s) — it exists to exercise monomorphisation of parameterised rules, not as real chase-AI logic. Don't mistake it for a working aggro implementation.
- Name collisions across imported files are a hard compile error, not a shadow/override — new stdlib decls must have globally-unique names relative to anything else a fixture might import.
