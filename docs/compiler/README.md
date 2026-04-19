# World Sim DSL Compiler — Design Docs

Working folder for the DSL-compiler design effort. Extracted from `docs/dsl/` on 2026-04-19 so compiler-layer concerns (codegen, lowering, schema emission) do not crowd the language-reference material. Status: settled on surface; implementation pending.

## Docs (reading order)

1. **`spec.md`** — Compiler contract. How DSL source text lowers to engine calls. Covers compilation targets (native Rust + GPU + hot/cold storage split), schema emission (how the compiler stamps schema hashes and enforces them via CI), lowering passes (verb desugaring, view storage-hint selection, cascade dispatch codegen), compiler-lowering decisions (D11, D16, D24), and compiler non-goals.
2. **`README.md`** — this file.

## Scope of this tree

- **In scope:** codegen targets, lowering passes, diagnostic emission, compile-time schema hashing, CI guards, compiler-specific decisions.
- **Out of scope (belongs in `docs/dsl/`):** language grammar, type system, policy/observation/action grammar, runtime semantics, worked examples, language-level decisions, stories, state catalog, systems inventory.
- **Out of scope (belongs in `docs/engine/`, in progress):** pools, determinism contract, event ring, mask representation, policy trait, tick pipeline at runtime.

## When to add new docs here

- New top-level compiler concerns (e.g. `ir.md` once the IR stabilises, `diagnostics.md` once error-message design firms up).
- Per-pass deep dives if a single lowering pass grows past a page in `spec.md`.

Until then, keep new material in `spec.md` and split out when it fights the one-doc flow.

## Cross-reference convention

- `spec.md §N` — section N of this compiler spec.
- `dsl/spec.md §N` — section N of the language reference.
- `engine/spec.md §N` — section N of the runtime contract (forthcoming).
