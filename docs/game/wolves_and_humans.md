# Wolves and humans — the first DSL-owned scenario (retired)

## Status: historical

**This scenario, and the crate that ran it, no longer exist in the workspace.** Everything described below was deleted wholesale by `c624896c` — `chore: nuke wolf-sim — DSL fixtures, legacy emitters, runtime crates` (2026-05-02, the "Phase 7 wolf-sim wipe" other docs refer to) — and it was never rebuilt under this name. Concretely, as of this writing:

- The `.sim` sources this doc walks through (`assets/sim/events.sim`, `physics.sim`, `masks.sim`, `scoring.sim`, `entities.sim`, `views.sim`, `enums.sim`, `config.sim`) are gone. `assets/sim/` holds 107 `.sim` files today and none of them are these.
- `crates/engine_rules` — the emission target throughout the table below — is gone and is not a workspace member.
- `xtask` (`cargo run --bin xtask -- compile-dsl`) is gone; there is no umbrella compiler CLI today.
- `crates/engine/tests/wolves_and_humans_parity.rs`, the parity anchor this whole doc is organized around, is gone (deleted in the same commit).
- `crates/engine` and `crates/engine_data` were restored the same day (per the Phase-7 comment in root `Cargo.toml`), but as generic infrastructure crates — SoA state, event ring, spatial hash, snapshot, RNG — with **no** wolves+humans-specific code. See `crates/engine/CLAUDE.md`.

**There is no direct successor fixture.** The closest living relative is the `predator_prey` family (`assets/sim/predator_prey.sim`, `predator_prey_min.sim`, `predator_prey_real.sim`, compiled into `crates/sims` — see `crates/sims/CLAUDE.md`), which covers similar ground — predator/prey hostility, spatial hunting, `Killed` events, materialized kill-count views, utility scoring — but it is a different scenario built fresh after the wipe (Wolf vs. `Hare`/`Sheep`, not Wolf vs. `Human`) with no code or history continuity from what's described below. If you want a live, runnable example of these DSL patterns, start there instead of here.

This document is kept as an archival record of the task-142 DSL port for anyone who lands here from old commit messages, ADRs, or `docs/game/compiler_progress.md` history. Nothing past this point is runnable today; commands are shown as they existed at the time, not as current instructions.

---

## Milestone complete — task 142 (historical)

Wolves+humans DSL port complete. The sequence of commits that got us here:

- **135** `feat(dsl): @decay sugar + gradient scoring modifiers + view fold-body UDF restriction` (`70252c3e`)
- **136** `refactor(dsl): rename event actor fields to canonical "actor"; fill @harmful + @visible tag contracts; delete legacy handler shims` (`a8d89cf0`)
- **137** `feat(dsl): mask + scoring coverage for all 18 micros; retire legacy mask.rs` (`5ce0a689`)
- **138** `feat(dsl): @lazy + @materialized view emission — inline fns + event-fold registry` (`23719994`)
- **139** `feat(compiler-first): target selection via scoring-argmax over masked candidates; retire nearest_other` (`794472a4`)
- **140** `refactor(compiler-first): is_hostile + record_memory as DSL; retire engine::rules shim` (`c755d7d8`)
- **141** `feat(compiler-first): engagement as event-driven physics; retire tick_start` (`011c5432`)
- **142** `chore(compiler-first): wolves+humans DSL port complete — balance audit, legacy shim cleanup, final docs` (`d2c85060`)

At the time, the parity anchor (`crates/engine/tests/wolves_and_humans_parity.rs`) stayed byte-identical across this whole series — every handoff from hand-written to DSL-emitted preserved the event log exactly. That test, the crate hosting it, and the `.sim` sources it pinned were all removed less than two weeks later by the Phase 7 wipe (`c624896c`, 2026-05-02); none of it survived into the current codebase.

## Scenario setup (historical)

The canonical fixture (as it existed in `crates/engine/tests/wolves_and_humans_parity.rs` before that file was deleted):

- **3 humans** at `(0, 0, 0)`, `(2, 0, 0)`, `(-2, 0, 0)` with HP 100 each.
- **2 wolves** at `(3, 0, 0)` and `(-3, 0, 0)` with HP 80 each.
- Fixed seed `0xD00D_FACE_0042_0042`.
- 100 ticks under the full 6-phase `step_full` pipeline with `UtilityBackend` + `CascadeRegistry::with_engine_builtins()`.

Expected behaviour inside 100 ticks:

- Wolves and their nearest human are within 1 m so the attack mask's Attack bit is set on tick 0 for both wolves and the humans in melee range.
- Cross-species damage accumulates; at least one human dies inside the run. Baseline at the time: humans 1, 2, 3 all die (ticks 6, 11, 16); the two wolves survive.
- The middle human (`id=1` at the origin) also walks toward the wolf at `(3,0,0)` on the first few ticks — `MoveToward` scores above `Hold` whenever an `AttackAllowed` neighbour isn't in melee.

## DSL sources (historical — none of these files exist today)

Five `.sim` files under `assets/sim/` drove the scenario end-to-end. Each one listed the declaration kinds it owned and the emission target it landed in — all of these targets (`crates/engine_rules/...`) were deleted along with the crate itself.

| DSL source | Owned | Emitted to |
|---|---|---|
| `assets/sim/events.sim` | 37 `event` declarations (every variant of `Event`, including `EngagementCommitted` / `EngagementBroken`) | `crates/engine_rules/src/events/*.rs` + `mod.rs` (re-exported by `engine::event`) |
| `assets/sim/physics.sim` | 9 `physics` rules (`damage`, `heal`, `shield`, `stun`, `slow`, `transfer_gold`, `modify_standing`, `opportunity_attack`, `record_memory`) | `crates/engine/src/generated/physics/*.rs` + `mod.rs` |
| `assets/sim/masks.sim` | 7 `mask` declarations (`Hold`, `MoveToward`, `Flee`, `Eat`, `Drink`, `Rest`, `Attack`) | `crates/engine/src/generated/mask/*.rs` + `mod.rs` |
| `assets/sim/scoring.sim` | 1 `scoring` block with 18 rows (every `MicroKind`, though most still scored `0.0`) | `crates/engine_rules/src/scoring/scoring_000.rs` + `mod.rs` |
| `assets/sim/entities.sim` | 4 `entity` declarations (`Human`, `Wolf`, `Deer`, `Dragon`) with `Capabilities` and `PredatorPrey` | `crates/engine_rules/src/entities/*.rs` + `mod.rs` (re-exported by `engine::creature`) |
| `assets/sim/views.sim` | `@lazy` + `@materialized` views (`is_hostile`, `engaged_with`, `threat_level`, `nearest_hostile`, …) | `crates/engine/src/generated/views/*.rs` |
| `assets/sim/enums.sim` | shared enums (`CreatureType`, `CommunicationChannel`, …) | `crates/engine_rules/src/types.rs` |
| `assets/sim/config.sim` | 4 `config` blocks (`combat`, `movement`, `needs`, `communication`) with 16 tunable fields | `crates/engine_rules/src/config/*.rs` + `assets/config/default.toml` |

The `physics.sim` rule bodies compiled to `impl CascadeHandler` unit structs. The `masks.sim` predicates compiled to a mix of free predicate fns (`fn mask_<name>(&SimState, AgentId[, AgentId]) -> bool`) and target-enumerator fns (`fn mask_<name>_candidates(&SimState, AgentId, &mut TargetMask)`). The `scoring.sim` rows compiled to a `pub static SCORING_TABLE: &[ScoringEntry]`. The `entities.sim` declarations compiled to a `CreatureType` enum plus a `for_creature(CreatureType) -> Capabilities` fn and an `is_hostile_to` pairwise table. The `views.sim` declarations compiled to `@lazy` inline fns and `@materialized` event-fold registries.

For comparison, `entity`/`physics`/`mask`/`scoring`/`view`/`config` declarations are still very much alive as DSL constructs — see e.g. `assets/sim/predator_prey.sim` — the specific *files* and *emission targets* above are what's gone, not the language features.

## Compilation (historical — command no longer exists)

```
cargo run --bin xtask -- compile-dsl
```

This walked every `.sim` file under `assets/sim/`, parsed and resolved all declarations into one `Compilation`, and wrote emitted Rust (rustfmt-normalised) + Python dataclasses + schema hashes into the targets listed above. CI ran `cargo run --bin xtask -- compile-dsl --check` to ensure the committed emission was byte-identical to a fresh run. Neither `xtask` nor a `compile-dsl --check` step exists today; there is no umbrella compiler binary in the current workspace.

Today, a `.sim` fixture is compiled at `cargo build` time by whichever crate's `build.rs` calls into `dsl_compiler::build_helper::emit`/`emit_namespaced` — for most fixtures that's `crates/sims/build.rs` (an allow-listed `.sim` stem becomes `sims::<stem>::GeneratedRuntime`, generated output written to `OUT_DIR`, never checked in); a couple of legacy fixtures (`crates/tom_probe_runtime`, `crates/viewer_runtime`) still compile their own `.sim` independently via their own `build.rs`. See `crates/sims/CLAUDE.md` and `crates/dsl_compiler/CLAUDE.md` for the current pipeline in full.

## This scenario ran on (historical)

### DSL-owned (compiler emitted, engine consumed)

- **Events** — 37 variants in `events.sim`.
- **Physics cascade handlers** — 9 rules in `physics.sim` (every stateless effect handler, including `RecordMemory` and `OpportunityAttack`).
- **Masks** — all 7 self-only / target-bound mask predicates in `masks.sim` (Hold, MoveToward, Flee, Eat, Drink, Rest, Attack).
- **Scoring** — full 18-row table in `scoring.sim`; the `UtilityBackend` iterated it, argmaxing over target-bound candidates from the compiler-emitted enumerators.
- **Entities** — taxonomy, capabilities, and the symmetric-closure `is_hostile_to` in `entities.sim`.
- **Views** — `@lazy` + `@materialized` in `views.sim`; the hostility view replaced the retired `engine::rules` shim (task 140), and `engaged_with` was event-folded (tasks 138–141).
- **Engagement** — event-driven via `engagement_on_move` / `engagement_on_death` cascade handlers (task 141) plus the `@materialized view engaged_with` fold.
- **Balance constants** — 16 tunable values in `config.sim`, TOML-editable via `assets/config/default.toml`.

### Engine-primitive (hand-written, not game logic)

At the time, the only hand-written files in `crates/engine/src/` were primitives with no DSL counterpart by design — `lib.rs`, `state/`, `spatial.rs`, `step.rs`, `mask.rs`, `policy/`, `cascade/`, `engagement.rs`, `event/`/`view/`/`invariant/`/`aggregate/`/`telemetry/`, `ability/expire.rs`, `channel.rs`, `ability/{id,mod,program,registry}.rs`. That `crates/engine` no longer exists; today's `crates/engine` is a post-wipe rebuild and its primitive/generated split should be read from its own `CLAUDE.md`, not inferred from this list.

## Fixture test (historical — file deleted, never replaced)

`crates/engine/tests/wolves_and_humans_parity.rs` was the regression anchor, with three tests: a byte-identical-baseline diff (`parity_log_is_byte_identical_to_baseline`, regenerable via `WOLVES_AND_HUMANS_REGEN=1`), a structural-invariant check (`parity_log_has_expected_structure`), and a same-process determinism check (`parity_log_is_deterministic_across_runs`). This file was deleted in the Phase 7 wipe (`c624896c`, 2026-05-02) and was never recreated — there is no `wolves_and_humans_parity.rs`, `wolves_and_humans_baseline.txt`, or `WOLVES_AND_HUMANS_REGEN` env var in the workspace today.

The nearest thing to this methodology in the current codebase is `crates/sims`' "pin" tests (`crates/sims/tests/<fixture>_pin.rs`) — but note the methodology differs: per `crates/sims/CLAUDE.md`, pin tests do **not** carry a checked-in baseline file to diff against; they seed a `GeneratedRuntime`, drive a fixed number of ticks, and assert hardcoded expected values plus (for some fixtures) a same-seed replay comparison for determinism. There is no byte-identical whole-event-log baseline mechanism anywhere in the current workspace.

## Adding another scenario

The checklist that used to live here (`cargo run --bin xtask -- compile-dsl`, delete the hand-written counterpart, regen the wolves+humans baseline if it drifted) no longer applies — the tooling and the fixture it referenced are both gone. For the current process of adding a `.sim` fixture, see `crates/sims/CLAUDE.md`'s "To add a new fixture" steps (drop `assets/sim/<name>.sim`, add its stem to the `matches!` allowlist in `crates/sims/build.rs`, `cargo build -p sims`, write `crates/sims/tests/<name>_pin.rs`) — there is no baseline-regen step in that flow because pin tests don't use checked-in baselines.
