# Specification

Canonical specification of the world-sim system. Each file locks one
contract; together they specify the full surface from DSL source text
to GPU dispatch.

## Files (reading order)

| File | Owns | Audit |
|---|---|---|
| `language.md` | World-sim DSL — grammar, types, semantics, settled decisions | [`audit-language-stdlib`](../superpowers/notes/2026-04-26-audit-language-stdlib.md) |
| `state.md` | Field catalog — every SoA field, who reads, who writes | [`audit-state`](../superpowers/notes/2026-04-26-audit-state.md) |
| `stdlib.md` | Pinned built-in functions and namespaces the compiler resolves against | [`audit-language-stdlib`](../superpowers/notes/2026-04-26-audit-language-stdlib.md) |
| `scoring_fields.md` | `field_id` ABI table — load-bearing for `SCORING_HASH` | — |
| `engine.md` | Engine runtime + GPU backend contract — state, events, mask, policy, cascade, tick pipeline, resident cascade, sim-state mirroring, cold-state replay, ability eval, pipeline reference (§§1–26 + GPU-1..7) | [`audit-runtime`](../superpowers/notes/2026-04-26-audit-runtime.md), [`audit-gpu`](../superpowers/notes/2026-04-26-audit-gpu.md) |
| `runtime.md` | (Legacy — superseded by `engine.md` §§1–26) | [`audit-runtime`](../superpowers/notes/2026-04-26-audit-runtime.md) |
| `gpu.md` | (Legacy — superseded by `engine.md` §§GPU-1..7) | [`audit-gpu`](../superpowers/notes/2026-04-26-audit-gpu.md) |
| `compiler.md` | DSL → engine lowering: codegen, lowering passes, schema-hash emission | — |
| `ability.md` | `.ability` DSL — language reference for ability definitions, IR, lowering to `EffectOp` | [`audit-ability`](../superpowers/notes/2026-04-26-audit-ability.md) |
| `economy.md` | Economic system — recipes, contracts, labor, market structure, macro dynamics. Three-phase implementation; extends ability.md `EffectOp` catalog. | [`audit-economy`](../superpowers/notes/2026-04-26-audit-economy.md) |

## Audit notes (2026-04-26)

Six spec audits compared each file's claims against `crates/` implementation. Findings flag stale sections, naming drift (e.g., `agents.stun_remaining_ticks` → `stun_expires_at_tick`), silent-drop bugs (decls that parse but emit nothing — `verb`, `invariant`, `probe`, `metric`, `per_ability` rows, `@spatial query`), and structural gaps (e.g., entire `AgentData` sub-struct, all Aggregate / WorldState entities, `pick_ability` GPU kernel). Inline `> ⚠️ Audit 2026-04-26:` callouts mark divergences; the linked notes hold per-section detail. **The audits do not propose fixes** — flagged divergences are surfaced for plan-drafting, not resolved here.

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
