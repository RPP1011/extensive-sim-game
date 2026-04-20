# Compiler progress

Tracks which DSL declaration kinds the compiler can emit and which legacy engine handlers have been retired. The compiler grows one milestone at a time; each milestone replaces a slice of hand-written Rust in `crates/engine/`.

## Ground rule

**No new hand-written game logic enters the codebase.** Every game rule — a cascade handler, a mask predicate, a scoring entry, a creature definition — enters the codebase as compiler output from DSL source. If you find yourself adding a `pub const BALANCE_NUMBER` or an `impl CascadeHandler` by hand, stop. Extend the compiler instead.

The existing game logic inside `crates/engine/src/ability/*.rs`, `crates/engine/src/creature.rs`, `crates/engine/src/policy/utility.rs`, and the balance constants in `crates/engine/src/step.rs` is **legacy**. It runs the current wolves+humans visualization. It does not grow. As each compiler milestone lands, the equivalent legacy code is deleted in the same commit.

## Milestone tracker

| # | Milestone | Compiler emits | Legacy retired | Status |
|---|---|---|---|---|
| 0 | Compiler scaffold | empty module + xtask wiring | — | done |
| 1 | `event` | Event enum variants; matching Python `@dataclass` | legacy event enum decls | done |
| 2 | (renumbered) `event` integration | events emitted into `engine_rules`, engine consumes via re-export | hand-written `engine::event::Event` | done |
| 3 | `physics` rule | `impl CascadeHandler` + registration call into `engine::cascade::CascadeRegistry` | `ability/*.rs` damage/heal/shield/stun/slow/gold/standing/opportunity_attack/record_memory handlers all DSL-owned; `CastHandler` carries `Arc<AbilityRegistry>` state and stays legacy (deferred — see wolves+humans doc) | done (minus Cast) |
| 4 | `mask` | predicate fn + target-enumerator fn | all 7 action-head masks (Hold, MoveToward, Flee, Eat, Drink, Rest, Attack) DSL-owned; the Cast mask gate stays in `ability/gate.rs` alongside the deferred CastHandler | done |
| 4 | `scoring` | per-action utility table | `policy/utility.rs` scoring body; 18-row table covers every `MicroKind`; `UtilityBackend` argmaxes over target-bound candidates from the emitted enumerators (task 138 retired `nearest_other`) | done |
| 5 | `entity` | spawn template + CreatureType variant + capability struct | `creature.rs` enum + `is_hostile_to` + `for_creature` — all four creatures (Human/Wolf/Deer/Dragon) DSL-owned | done |
| 6 | `view` (`@lazy` + `@materialized`) | inline fn + event-fold registry | view-like helpers scattered in engine; `crate::rules::is_hostile` shim retired (task 140); `engaged_with` materialized view replaced `tick_start`'s tentative-commit engagement pass (task 141) | done |
| 7 | `verb` | lowering to mask + cascade + scoring entries | — (verbs are new; no legacy) | not started |
| 8 | `invariant` | runtime-check fn | engine's `PoolNonOverlapInvariant`-style checks stay (engine invariants), game invariants move | not started |
| 9 | `probe` | fixture-test scaffolding | existing `tests/` that encode game behavior | not started |
| 10 | `metric` | metric emission points + alert plumbing | engine's hardcoded trace counters | not started |
| 11 | Python emission | dataclass module + pytorch Dataset | — (new capability) | not started |
| 12 | SPIR-V emission | mask + cascade + view kernels | — (new capability, gated on GpuBackend) | not started |
| 13 | `config` | per-block Rust structs + `Default` impls + `Config::from_toml`; `assets/config/default.toml`; new `CONFIG_HASH` sub-hash | 16 balance consts from `step.rs` / `mask.rs` / `ability/expire.rs` / `channel.rs` folded into `config.combat.*` / `config.movement.*` / `config.needs.*` / `config.communication.*`. Pre-config `pub const` shims retired in task 142 — every test reads `Config::default()`. | done |

Status: explicit "done" / "not started" keyword; see `wolves_and_humans.md` for the parity anchor and the deferred-items list.

## How a milestone lands

Each milestone follows the same shape:

1. **Extend the compiler** — grammar (parser), lowering (AST → IR), emission (IR → Rust or Python).
2. **Author DSL source** for the legacy feature being replaced. Canonical path: `assets/sim/<feature>.sim`.
3. **Compile** — `cargo run --bin xtask -- compile-dsl`. Emission goes to `crates/engine_rules/src/<feature>.rs` (generated, checked in for readability; regenerated on every compile).
4. **Wire** — the engine's `CascadeRegistry::register_all(state)` picks up the newly emitted modules automatically via the compiler-emitted `lib.rs`.
5. **Delete the legacy** — in the same commit that lands the compiler support, delete the hand-written Rust it replaces. No parallel paths.
6. **Regression check** — the existing fixture tests (currently exercising the legacy code) now exercise the compiler-emitted code. They must still pass bit-for-bit: same seed, same event log.

If the fixture tests don't pass after a milestone, the milestone isn't done. There is no "mostly emitted" state.

## Why no parallel bootstrap crate

Earlier plans called for extracting `crates/bootstrap_rules/` as a hand-written emission-target spec. That approach was dropped (2026-04-19): it risks committing to an emission shape that the compiler can't match, investing heavily in hand-written Rust that gets thrown away, and repeating the pattern that caused the previous two months of tech-debt churn.

The compiler-first approach defers all game-rule work until the compiler can emit it, but every line of game logic that lands is load-bearing and permanent.

## What's still allowed as "hand-written"

- **Engine internals** — `crates/engine/src/state/mod.rs`, `spatial.rs`, `cascade/*`, `event.rs`, `step.rs` (the orchestration kernel, not the balance constants). These are primitives, not game logic.
- **Compiler internals** — `crates/dsl_compiler/*`. The compiler itself is hand-written.
- **Tests** — fixture scenarios, regression probes, property tests. Tests don't get compiled from DSL; they *test* the compiled output.
- **External tooling** — Python training scripts, viz binaries, xtask commands. None of these are sim behavior.

Anywhere else, the rule is: if the DSL could have declared it, the DSL must declare it. If the DSL's grammar doesn't cover the case, extend the grammar.

## Wolves+humans DSL port complete

As of task 142 (commit series 135 through 142, `5ce0a689` … task-142-commit), the wolves+humans scenario runs **entirely** on DSL-emitted code modulo CastHandler, which is dormant (no cast-capable creatures in the fixture). The regression anchor is `crates/engine/tests/wolves_and_humans_parity.rs` — 3 humans + 2 wolves, fixed seed, 100 ticks, byte-identical event-log comparison against a committed baseline.

CastHandler migration is a separate follow-up milestone; the stateless-handler emitter needs a "handler state" extension before `Arc<AbilityRegistry>` can flow into a compiler-emitted handler. See `docs/game/wolves_and_humans.md` for the authoritative walkthrough, the deferred-items list, and the migration checklist for the next scenario.

The `world-sim` visualization (`cargo run --bin xtask -- world-sim`) runs on the same engine and therefore also on the DSL-owned rules. Further milestones (verbs, invariants, probes, metrics, Python/SPIR-V emission) extend the compiler's reach to new scenarios rather than to new parts of the wolves+humans stack, which is now frozen-and-parity-pinned.
