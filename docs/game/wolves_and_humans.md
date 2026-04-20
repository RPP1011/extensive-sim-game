# Wolves and humans — the first DSL-owned scenario

This is the walkthrough anchor for the "wolves + humans on a plane" scenario. It walks the stack from the DSL sources through the compiler to the engine and out to the parity test. As compiler milestones landed, the "still hand-written" list shrank; with task 142 it has now emptied for this scenario — wolves+humans is fully DSL-owned modulo a single dormant subsystem (CastHandler, formally deferred below).

## Milestone complete — task 142

Wolves+humans DSL port complete. The sequence of commits that got us here:

- **135** `feat(dsl): @decay sugar + gradient scoring modifiers + view fold-body UDF restriction` (`70252c3e`)
- **136** `refactor(dsl): rename event actor fields to canonical "actor"; fill @harmful + @visible tag contracts; delete legacy handler shims` (`a8d89cf0`)
- **137** `feat(dsl): mask + scoring coverage for all 18 micros; retire legacy mask.rs` (`5ce0a689`)
- **138** `feat(dsl): @lazy + @materialized view emission — inline fns + event-fold registry` (`23719994`)
- **139** `feat(compiler-first): target selection via scoring-argmax over masked candidates; retire nearest_other` (`794472a4`)
- **140** `refactor(compiler-first): is_hostile + record_memory as DSL; retire engine::rules shim` (`c755d7d8`)
- **141** `feat(compiler-first): engagement as event-driven physics; retire tick_start` (`011c5432`)
- **142** (this commit) `chore(compiler-first): wolves+humans DSL port complete — balance audit, legacy shim cleanup, final docs`

The parity anchor (`crates/engine/tests/wolves_and_humans_parity.rs`) stays byte-identical across the whole series — every handoff from hand-written to DSL-emitted preserves the event log exactly.

## Scenario setup

The canonical fixture (see `crates/engine/tests/wolves_and_humans_parity.rs`):

- **3 humans** at `(0, 0, 0)`, `(2, 0, 0)`, `(-2, 0, 0)` with HP 100 each.
- **2 wolves** at `(3, 0, 0)` and `(-3, 0, 0)` with HP 80 each.
- Fixed seed `0xD00D_FACE_0042_0042`.
- 100 ticks under the full 6-phase `step_full` pipeline with `UtilityBackend` + `CascadeRegistry::with_engine_builtins()`.

Expected behaviour inside 100 ticks:

- Wolves and their nearest human are within 1 m so the attack mask's Attack bit is set on tick 0 for both wolves and the humans in melee range.
- Cross-species damage accumulates; at least one human dies inside the run. Current baseline: humans 1, 2, 3 all die (ticks 6, 11, 16); the two wolves survive.
- The middle human (`id=1` at the origin) also walks toward the wolf at `(3,0,0)` on the first few ticks — `MoveToward` scores above `Hold` whenever an `AttackAllowed` neighbour isn't in melee.

## DSL sources

Five `.sim` files under `assets/sim/` drive the scenario end-to-end. Each one lists the declaration kinds it owns and the emission target it lands in.

| DSL source | Owns | Emits to |
|---|---|---|
| `assets/sim/events.sim` | 37 `event` declarations (every variant of `Event`, including `EngagementCommitted` / `EngagementBroken`) | `crates/engine_rules/src/events/*.rs` + `mod.rs` (re-exported by `engine::event`) |
| `assets/sim/physics.sim` | 9 `physics` rules (`damage`, `heal`, `shield`, `stun`, `slow`, `transfer_gold`, `modify_standing`, `opportunity_attack`, `record_memory`) | `crates/engine/src/generated/physics/*.rs` + `mod.rs` |
| `assets/sim/masks.sim` | 7 `mask` declarations (`Hold`, `MoveToward`, `Flee`, `Eat`, `Drink`, `Rest`, `Attack`) | `crates/engine/src/generated/mask/*.rs` + `mod.rs` |
| `assets/sim/scoring.sim` | 1 `scoring` block with 18 rows (every `MicroKind`, though most still score `0.0`) | `crates/engine_rules/src/scoring/scoring_000.rs` + `mod.rs` |
| `assets/sim/entities.sim` | 4 `entity` declarations (`Human`, `Wolf`, `Deer`, `Dragon`) with `Capabilities` and `PredatorPrey` | `crates/engine_rules/src/entities/*.rs` + `mod.rs` (re-exported by `engine::creature`) |
| `assets/sim/views.sim` | `@lazy` + `@materialized` views (`is_hostile`, `engaged_with`, `threat_level`, `nearest_hostile`, …) | `crates/engine/src/generated/views/*.rs` |
| `assets/sim/enums.sim` | shared enums (`CreatureType`, `CommunicationChannel`, …) | `crates/engine_rules/src/types.rs` |
| `assets/sim/config.sim` | 4 `config` blocks (`combat`, `movement`, `needs`, `communication`) with 16 tunable fields | `crates/engine_rules/src/config/*.rs` + `assets/config/default.toml` |

The `physics.sim` rule bodies compile to `impl CascadeHandler` unit structs. The `masks.sim` predicates compile to a mix of free predicate fns (`fn mask_<name>(&SimState, AgentId[, AgentId]) -> bool`) and target-enumerator fns (`fn mask_<name>_candidates(&SimState, AgentId, &mut TargetMask)`). The `scoring.sim` rows compile to a `pub static SCORING_TABLE: &[ScoringEntry]`. The `entities.sim` declarations compile to a `CreatureType` enum plus a `for_creature(CreatureType) -> Capabilities` fn and an `is_hostile_to` pairwise table. The `views.sim` declarations compile to `@lazy` inline fns and `@materialized` event-fold registries.

## Compilation

```
cargo run --bin xtask -- compile-dsl
```

This walks every `.sim` file under `assets/sim/`, parses and resolves all declarations into one `Compilation`, and writes emitted Rust (rustfmt-normalised) + Python dataclasses + schema hashes into the targets listed above. CI runs `cargo run --bin xtask -- compile-dsl --check` to ensure the committed emission is byte-identical to a fresh run.

## This scenario runs on

### DSL-owned (compiler emits, engine consumes)

- **Events** — 37 variants in `events.sim`.
- **Physics cascade handlers** — 9 rules in `physics.sim` (every stateless effect handler, including `RecordMemory` and `OpportunityAttack`).
- **Masks** — all 7 self-only / target-bound mask predicates in `masks.sim` (Hold, MoveToward, Flee, Eat, Drink, Rest, Attack).
- **Scoring** — full 18-row table in `scoring.sim`; the `UtilityBackend` iterates it, argmaxing over target-bound candidates from the compiler-emitted enumerators.
- **Entities** — taxonomy, capabilities, and the symmetric-closure `is_hostile_to` in `entities.sim`.
- **Views** — `@lazy` + `@materialized` in `views.sim`; the hostility view replaces the retired `engine::rules` shim (task 140), and `engaged_with` is event-folded (tasks 138–141).
- **Engagement** — event-driven via `engagement_on_move` / `engagement_on_death` cascade handlers (task 141) plus the `@materialized view engaged_with` fold.
- **Balance constants** — 16 tunable values in `config.sim`, TOML-editable via `assets/config/default.toml`.

### Engine-primitive (hand-written, not game logic)

The only hand-written files in `crates/engine/src/` are primitives that have no DSL counterpart by design:

- **`lib.rs`** — module declarations + `VERSION`.
- **`state/`** — `SimState`, `AgentSpawn`, the SoA hot/cold storage (engine primitive; not game rules).
- **`spatial.rs`** — incremental uniform-grid spatial hash (`CELL_SIZE = 16m` is a cache-line tuning parameter, not a balance knob).
- **`step.rs`** — the 6-phase tick pipeline (mask → policy → shuffle → apply → view-fold → invariants). Orchestration kernel, not game rules.
- **`mask.rs`** — the `MicroKind` enum, `TargetMask` storage, and the thin `mark_*_allowed` wrappers that dispatch to the compiler-emitted predicates. Table-dispatching primitive.
- **`policy/`** — `PolicyBackend` trait + `UtilityBackend` (argmax over the compiler-emitted `SCORING_TABLE`). Engine primitive, not game logic.
- **`cascade/`** — cascade dispatcher + `MAX_CASCADE_ITERATIONS = 8` safety bound.
- **`engagement.rs`** — event-driven engagement helpers (stateless handlers that the compiler-emitted view-fold consumes).
- **`event/`**, **`view/`**, **`invariant/`**, **`aggregate/`**, **`telemetry/`** — primitive infrastructure (ring buffer, trait surface, dispatch tables).
- **`ability/expire.rs`** — stateless stun/slow timer decrement. The timer model itself is a scheduling primitive; migration to a timestamp-based cooldown model is a follow-up milestone unrelated to the DSL port.
- **`channel.rs`** — per-channel range formula (dispatch table keyed on an engine enum; the per-channel base distances moved into `config.communication.*` at task 142, only the dispatch logic stays).
- **`ability/` (cast + gate + registry)** — see "Deferred" below.

### Deferred — CastHandler (cast-based abilities)

Cast-based actions (magic abilities, multi-effect spells) remain hand-written in `crates/engine/src/ability/cast.rs`. The wolves+humans scenario does not exercise magic; the four entities in `entities.sim` (Human, Wolf, Deer, Dragon) have no cast-capable capability flag and no `.ability` registration, so CastHandler is dormant throughout the 100-tick fixture.

`CastHandler` is stateful: it carries an `Arc<AbilityRegistry>` so multiple cascade lanes can share one registry cheaply. The physics emitter produces stateless unit-struct handlers (`impl CascadeHandler` with no fields); adding a "handler state" concept to the emitter is ~500 LOC and outside the scope of the wolves+humans DSL-port claim.

Runtime contract: wolves+humans has zero cast-capable creatures → CastHandler never fires → its implementation is irrelevant for parity. When a scenario with cast-capable creatures lands, it MUST formally register a cast handler via `CascadeRegistry::register_cast_handler(Arc<AbilityRegistry>)`; parity for that scenario is a separate milestone.

Tracked as a follow-up tech-debt item, not a blocker. Related files:

- `crates/engine/src/ability/cast.rs` — `CastHandler` (stateful, deferred).
- `crates/engine/src/ability/gate.rs` — `evaluate_cast_gate` (the cast-time cooldown/stun/range/hostility/engagement-lock predicate; migrates when the Cast mask does).
- `crates/engine/src/ability/registry.rs`, `program.rs`, `id.rs`, `mod.rs` — ability registry primitive + the `AbilityId` newtype.

## Fixture test

`crates/engine/tests/wolves_and_humans_parity.rs` is the regression anchor. Three tests:

1. `parity_log_is_byte_identical_to_baseline` — renders the 100-tick event log as one line per event and compares against the committed `tests/wolves_and_humans_baseline.txt`. Any DSL edit that changes the wolves+humans trace fails this test. To refresh the baseline (intentionally): `WOLVES_AND_HUMANS_REGEN=1 cargo test -p engine --test wolves_and_humans_parity`.
2. `parity_log_has_expected_structure` — asserts event counts, tick monotonicity, and that the majority of `AgentAttacked` events cross species (validates that the `is_hostile` gate is in effect).
3. `parity_log_is_deterministic_across_runs` — two fresh runs in the same process must produce byte-identical logs.

Run the full suite with `cargo test -p engine --release`. CI runs it in both debug and release so the determinism path is exercised on both contract-check configurations.

## Adding another scenario

The checklist is always the same:

1. Extend the DSL grammar if needed (e.g. a new declaration kind for a new behaviour class).
2. Add the declaration to the appropriate `.sim` file.
3. `cargo run --bin xtask -- compile-dsl`.
4. Delete the hand-written counterpart in the SAME commit.
5. Run `cargo test -p engine --release` — every existing fixture plus the parity test must still pass bit-for-bit.
6. If the migration changes the wolves+humans event log, regen the baseline with `WOLVES_AND_HUMANS_REGEN=1` and include the diff in review.

The parity anchor fails loud, which is the point.
