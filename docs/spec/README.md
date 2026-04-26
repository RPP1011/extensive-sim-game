# Specification

Canonical specification of the world-sim system. Each file locks one
contract; together they specify the full surface from DSL source text
to GPU dispatch.

## Files (reading order)

| File | Owns | Audit |
|---|---|---|
| `engine.md` | Engine runtime + GPU backend contract — state, events, mask, policy, cascade, tick pipeline, resident cascade, sim-state mirroring, cold-state replay, ability eval, kernel/buffer reference | [`audit-runtime`](../superpowers/notes/2026-04-26-audit-runtime.md), [`audit-gpu`](../superpowers/notes/2026-04-26-audit-gpu.md) |
| `state.md` | SoA field catalog — every field, who reads, who writes | [`audit-state`](../superpowers/notes/2026-04-26-audit-state.md) |
| `dsl.md` | DSL language reference — grammar, types, expressions, statements, stdlib namespaces, scoring grammar, compiler architecture | [`audit-language-stdlib`](../superpowers/notes/2026-04-26-audit-language-stdlib.md) |
| `ability.md` | `.ability` DSL — ability definitions, IR, lowering to `EffectOp` | [`audit-ability`](../superpowers/notes/2026-04-26-audit-ability.md) |
| `economy.md` | Economic system — recipes, contracts, labor, market structure, macro dynamics. Three-phase implementation; extends `ability.md` `EffectOp` catalog | [`audit-economy`](../superpowers/notes/2026-04-26-audit-economy.md) |

## What used to live elsewhere

The spec layout was consolidated 2026-04-26 (commits `7fd128b9`, `3e8dca5d`, `118d5953`):

| Old file (deleted) | Now lives in |
|---|---|
| `runtime.md` | `engine.md` |
| `gpu.md` | `engine.md` (GPU annexes — §§9–12) |
| `language.md` | `dsl.md` |
| `stdlib.md` | `dsl.md` §7 |
| `compiler.md` | `dsl.md` §9 |
| `scoring_fields.md` | `dsl.md` §8 |

## Audit notes (2026-04-26)

Six spec audits compared each file's claims against `crates/` implementation. Findings flag stale sections, naming drift (e.g., `agents.stun_remaining_ticks` → `stun_expires_at_tick`), silent-drop bugs (decls that parse but emit nothing — `verb`, `invariant`, `probe`, `metric`, `per_ability` rows, `@spatial query`), and structural gaps (e.g., entire `AgentData` sub-struct, all Aggregate / WorldState entities, `pick_ability` GPU kernel). Inline `> ⚠️ Audit 2026-04-26:` callouts mark divergences; the linked notes hold per-section detail. **The audits do not propose fixes** — flagged divergences are surfaced for plan-drafting, not resolved here.

## Cross-reference convention

- `<file>.md §N` — section N (e.g., `engine.md §4` = the per-phase tick contracts).
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
