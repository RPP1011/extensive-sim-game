# Specification

Canonical specification of the world-sim system. Each file locks one
contract; together they specify the full surface from DSL source text
to GPU dispatch.

## Files (reading order)

| File | Owns |
|---|---|
| `language.md` | World-sim DSL — grammar, types, semantics, settled decisions |
| `state.md` | Field catalog — every SoA field, who reads, who writes |
| `stdlib.md` | Pinned built-in functions and namespaces the compiler resolves against |
| `scoring_fields.md` | `field_id` ABI table — load-bearing for `SCORING_HASH` |
| `runtime.md` | Engine runtime contract — state, events, mask, policy, cascade, tick pipeline (§§1–26) |
| `compiler.md` | DSL → engine lowering: codegen, lowering passes, schema-hash emission |
| `gpu.md` | GPU backend contract — resident cascade, sim-state mirroring, cold-state replay, ability eval, pipeline reference |
| `ability.md` | `.ability` DSL — language reference for ability definitions, IR, lowering to `EffectOp` |
| `economy.md` | Economic system — recipes, contracts, labor, market structure, macro dynamics. Three-phase implementation; extends ability.md `EffectOp` catalog. |

## Cross-reference convention

- `<file>.md §N` — section N (e.g., `runtime.md §14` = the six-phase tick pipeline).
- `<file>.md §N.M` — sub-section.
- Live implementation status (per-subsystem ✅/⚠️/❌) lives in `../engine/status.md`,
  not here. The spec describes contract; status describes execution.

## What this spec doesn't cover

- **Plans.** Implementation intent for in-flight work lives in `../superpowers/plans/`.
- **Brainstorms / research.** Design exploration lives in `../superpowers/{specs,research,notes}/`.
- **History.** "Why we got here" prose is in git history, not in this spec.

## Stability rule

Bump `crates/engine/.schema_hash` for any change here that alters layout or
semantics. CI catches drift.
