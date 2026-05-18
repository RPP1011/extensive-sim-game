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
