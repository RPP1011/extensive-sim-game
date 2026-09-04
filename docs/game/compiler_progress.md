# Compiler progress

Tracks which DSL declaration kinds the compiler can emit and which legacy engine handlers have been retired. The compiler grows one milestone at a time; each milestone replaces a slice of hand-written Rust in `crates/engine/`.

**Reading this file today — important context.** Most of the milestone history below documents the pre-2026-05-02 "wolves+humans" scenario and the `xtask` / `crates/engine_rules` tooling that built it. Both the scenario and that tooling were deleted wholesale in the **Phase 7 wolf-sim wipe** (commit `c624896c`, 2026-05-02) — there is no `xtask` binary, no `crates/engine_rules`, and no `crates/engine/tests/wolves_and_humans_parity.rs` (or its baseline) anywhere in the current tree; the pre-wipe `crates/engine` and `crates/engine_data` were deleted in the same commit and later rebuilt from scratch under the same names but with different (and more general) content. `crates/dsl_ast` and `dsl_compiler`'s `cg::*` pipeline survived the wipe intact and remain the sole compiler today — see `crates/dsl_compiler/CLAUDE.md`. Where a row or passage below cites `crates/engine_rules/...` or an `xtask` command, read it as a historical record of what was true before 2026-05-02, not as a description of anything present in the codebase now. For the corrected current-state summary, see `docs/game/overview.md`'s "Where the game currently lives" section; for how a `.sim` fixture is actually compiled today, see `crates/sims/CLAUDE.md`.

## Ground rule

**No new hand-written game logic enters the codebase.** Every game rule — a cascade handler, a mask predicate, a scoring entry, a creature definition — enters the codebase as compiler output from DSL source. If you find yourself adding a `pub const BALANCE_NUMBER` or an `impl CascadeHandler` by hand, stop. Extend the compiler instead.

`crates/engine/src/ability/*.rs` and `crates/engine/src/policy/utility.rs` are pre-wipe legacy scaffolding, but they do not run a "wolves+humans visualization" today — no such scenario exists in the current codebase (see note above). `crates/engine/src/creature.rs` is now just a doc-comment pointer; the vocabulary itself moved to `engine_data::entities` at Milestone 6 and is hand-maintained there (see `crates/engine_data/CLAUDE.md`, "Provenance"). `crates/engine/src/step.rs` no longer holds balance constants — it is `unimplemented!()` compile-only stubs kept so old `#[ignore]`d tests still import cleanly; the real tick driver for any given fixture today is compiler-emitted per-fixture code inside `crates/sims`' generated `GeneratedRuntime` modules (or a legacy `crates/*_runtime` crate), not an engine-wide orchestrator. The underlying discipline still holds going forward: as each compiler milestone lands, the equivalent hand-written code it replaces should be deleted in the same commit. The Phase 7 wipe was the one historical exception — an entire pre-wipe scenario, compiler output included, was deleted wholesale rather than retired incrementally per-milestone.

## Milestone tracker

Historical (pre-2026-05-02) unless noted otherwise — see the note at the top of this file.

| # | Milestone | Compiler emits | Legacy retired | Status |
|---|---|---|---|---|
| 0 | Compiler scaffold | empty module + build wiring via the (now-retired) `xtask` binary | — | done |
| 1 | `event` | Event enum variants; matching Python `@dataclass` | legacy event enum decls | done |
| 2 | (renumbered) `event` integration | events emitted into the pre-wipe `engine_rules` crate (deleted in the Phase 7 wolf-sim wipe, 2026-05-02; no successor crate — event vocabulary today lives hand-maintained in `engine_data::events`), engine consumed via re-export | hand-written `engine::event::Event` | done |
| 3 | `physics` rule | `impl CascadeHandler` + registration call into `engine::cascade::CascadeRegistry` | `ability/*.rs` damage/heal/shield/stun/slow/gold/standing/opportunity_attack/record_memory handlers all DSL-owned; as of 2026-04-19 the `cast` rule landed too — compiler grew `for ... in <collection>` loops and `match` over stdlib-known sum-type variants (`EffectOp::*`, `TargetSelector::*`), and the last hand-written cascade handler with game logic (`crates/engine/src/ability/cast.rs`) is retired | done |
| 4 | `mask` | predicate fn + target-enumerator fn | all 7 action-head masks (Hold, MoveToward, Flee, Eat, Drink, Rest, Attack) DSL-owned; the Cast mask gate stays in `ability/gate.rs` — the gate reaches into `state.ability_registry` for per-program range/hostility/cooldown data that the mask DSL doesn't yet expose, so it's the last hand-written mask predicate | done |
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

Status: explicit "done" / "not started" keyword; see `wolves_and_humans.md` for the (now-historical) parity anchor and deferred-items list — per the note at the top of this file, that scenario and its parity test no longer exist in the current tree.

## How a milestone lands

This described the wolves+humans-era workflow (pre-2026-05-02); it no longer matches how a `.sim` fixture is built today, because "milestone" in that era meant "a shared rule file feeding the one running wolves+humans scenario," whereas today's `assets/sim/*.sim` fixtures are each independent, self-contained runtimes (`sims::<fixture>::GeneratedRuntime`) with no single scenario to port pieces of. Kept below for historical shape; see `crates/sims/CLAUDE.md` ("To add a new fixture") for the actual current recipe.

1. **Extend the compiler** — grammar (`dsl_ast`'s parser/resolver), lowering, emission (`dsl_compiler`'s `cg::*` pipeline: CG IR → schedule → WGSL/Rust). This part is still accurate today.
2. **Author DSL source.** Canonical path: `assets/sim/<fixture>.sim`. Today this is usually net-new fixture content, not a port of an existing hand-written feature.
3. **Compile.** There is no `xtask` command. For a fixture inside the `sims` mega-crate: add its stem to the allowlist `matches!` arm in `crates/sims/build.rs`, then `cargo build -p sims` (its `build.rs` calls `dsl_compiler::build_helper::emit_namespaced`). For the two remaining legacy per-fixture crates (`tom_probe_runtime`, `viewer_runtime`), `cargo build -p <crate>` triggers that crate's own `build.rs` the same way. Either path emits Rust + WGSL into `OUT_DIR` — **build-time only, never checked into the repo** (unlike the pre-wipe `engine_rules` pattern, which committed emitted Rust for readability).
4. **Wire.** No manual registry-wiring step exists in the current pipeline. `build.rs` auto-generates an `OUT_DIR/sim_modules.rs` (or equivalent) that `include!`s the generated modules into the crate, and — for `sims` specifically — also emits a `make_playable`/`PLAYABLE_FIXTURES` name-keyed registry. (This replaces the old `CascadeRegistry::register_all(state)` step described here previously, which was specific to the deleted wolves+humans wiring.)
5. **Delete the legacy** — still the rule whenever a fixture genuinely replaces hand-written Rust: delete the hand-written code in the same commit. No parallel paths.
6. **Regression check.** Today's fixture tests are pin tests (`crates/sims/tests/<fixture>_pin.rs`) that seed a `GeneratedRuntime`, drive known ticks, and assert hardcoded expected values and/or cross-run determinism directly — not a diff against a checked-in baseline log (that pattern was specific to `wolves_and_humans_parity.rs`, which no longer exists).

If the fixture tests don't pass after a milestone, the milestone isn't done. There is no "mostly emitted" state.

## Why no parallel bootstrap crate

Earlier plans called for extracting `crates/bootstrap_rules/` as a hand-written emission-target spec. That approach was dropped (2026-04-19): it risks committing to an emission shape that the compiler can't match, investing heavily in hand-written Rust that gets thrown away, and repeating the pattern that caused the previous two months of tech-debt churn.

The compiler-first approach defers all game-rule work until the compiler can emit it, but every line of game logic that lands is load-bearing and permanent.

## What's still allowed as "hand-written"

- **Engine internals** — `crates/engine/src/state/mod.rs`, `spatial.rs`, `cascade/*`, `event.rs`. These are primitives, not game logic. (`step.rs` is *not* a live orchestration kernel today — it's `unimplemented!()` compile-only stubs; see the Ground rule section above.)
- **Compiler internals** — `crates/dsl_compiler/*`. The compiler itself is hand-written.
- **Tests** — fixture scenarios, regression probes, property tests. Tests don't get compiled from DSL; they *test* the compiled output.
- **External tooling** — Python training scripts, and the `sim_app` crate's feature-gated runnable binaries (`viz_app`, `tom_probe_app`, each behind a `bin-<name>` cargo feature; there is no `xtask` umbrella binary today, see the note at the top of this file). None of these are sim behavior.

Anywhere else, the rule is: if the DSL could have declared it, the DSL must declare it. If the DSL's grammar doesn't cover the case, extend the grammar.

## Wolves+humans DSL port complete (historical — the scenario itself is gone)

As of task 142 (commit series 135 through 142, `5ce0a689` … task-142-commit, 2026-04-19 through 2026-04-2x), the wolves+humans scenario ran **entirely** on DSL-emitted code. The regression anchor at the time was `crates/engine/tests/wolves_and_humans_parity.rs` — 3 humans + 2 wolves, fixed seed, 100 ticks, byte-identical event-log comparison against a committed baseline.

CastHandler migration landed 2026-04-19 once the compiler grew `for ... in <collection>` loops and `match` over stdlib-known sum-type variants; the DSL `physics cast` rule read the ability registry through a small `abilities.*` stdlib namespace and dispatched effects with a `for` + `match` over `EffectOp`, bounding recursion depth with `cascade.max_iterations`.

**None of this exists in the current codebase.** The Phase 7 wolf-sim wipe (commit `c624896c`, 2026-05-02 — see the note at the top of this file) deleted the wolves+humans `.sim` sources, the parity test and its baseline, `crates/engine_rules`, `crates/xtask`, and the pre-wipe `crates/engine`/`crates/engine_data` in one commit, in favor of rebuilding around a more general per-fixture model. `crates/dsl_ast` and `dsl_compiler`'s `cg::*` pipeline survived unchanged; new `crates/engine`/`crates/engine_data` crates were rebuilt afterward under the same names but with different content and no wolves+humans scenario. `docs/game/wolves_and_humans.md` still documents this pre-wipe milestone in detail and is itself stale on the same points — treat both docs' wolves+humans-specific claims as closed history, not present-day fact.

There is no current single "flagship" scenario analogous to wolves+humans; DSL fixtures today are the ~100 independent `.sim` files under `assets/sim/`, most compiled into `crates/sims`' generated `GeneratedRuntime`s (see `crates/sims/CLAUDE.md`). Milestones 7–12 (`verb`, `invariant`, `probe`, `metric`, Python/SPIR-V emission) remain not-started against that current fixture set, not against the deleted wolves+humans stack.
