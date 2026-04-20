# Wolves and humans — the first DSL-owned scenario

This is the walkthrough anchor for the "wolves + humans on a plane" scenario. It walks the stack from the DSL sources through the compiler to the engine and out to the parity test. As compiler milestones land, the "still hand-written" list at the bottom shrinks; when that list empties, the scenario is fully DSL-owned.

As of 2026-04-19 the scenario runs on compiler milestones 2-6 + physics parity. Event, physics (minus cast), the Attack mask, the four-row scoring table, and the Human/Wolf/Deer/Dragon entity taxonomy are all DSL-owned. Cast dispatch and a handful of balance constants are still hand-written; see the list at the bottom.

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
| `assets/sim/events.sim` | 35 `event` declarations (every variant of `Event`) | `crates/engine_rules/src/events/*.rs` + `mod.rs` (re-exported by `engine::event`) |
| `assets/sim/physics.sim` | 8 `physics` rules (`damage`, `heal`, `shield`, `stun`, `slow`, `transfer_gold`, `modify_standing`, `opportunity_attack`) | `crates/engine/src/generated/physics/*.rs` + `mod.rs` |
| `assets/sim/masks.sim` | 1 `mask` declaration (`Attack`) | `crates/engine/src/generated/mask/attack.rs` + `mod.rs` |
| `assets/sim/scoring.sim` | 1 `scoring` block with 4 rows (`Hold`, `MoveToward`, `Attack`, `Eat`) | `crates/engine_rules/src/scoring/scoring_000.rs` + `mod.rs` |
| `assets/sim/entities.sim` | 4 `entity` declarations (`Human`, `Wolf`, `Deer`, `Dragon`) with `Capabilities` and `PredatorPrey` | `crates/engine_rules/src/entities/*.rs` + `mod.rs` (re-exported by `engine::creature`) |

The `physics.sim` rule bodies compile to `impl CascadeHandler` unit structs. The `masks.sim` attack predicate compiles to a free `fn mask_attack(&SimState, AgentId, AgentId) -> bool`. The `scoring.sim` rows compile to a `pub static SCORING_TABLE: &[ScoringEntry]`. The `entities.sim` declarations compile to a `CreatureType` enum plus a `for_creature(CreatureType) -> Capabilities` fn and an `is_hostile_to` pairwise table.

## Compilation

```
cargo run --bin xtask -- compile-dsl
```

This walks every `.sim` file under `assets/sim/`, parses and resolves all declarations into one `Compilation`, and writes emitted Rust (rustfmt-normalised) + Python dataclasses + schema hashes into the targets listed above. CI runs `cargo run --bin xtask -- compile-dsl --check` to ensure the committed emission is byte-identical to a fresh run.

## Engine consumption

The engine plugs each emitted module in exactly one place.

- **Event enum shim** — `crates/engine/src/event/mod.rs` re-exports `engine_rules::events::Event`. Every `use crate::event::Event` in the engine transparently reaches the compiler-emitted enum.
- **Cascade registration** — `CascadeRegistry::with_engine_builtins()` in `crates/engine/src/cascade/dispatch.rs` calls `crate::generated::physics::register(self)`. That in turn boxes and registers one handler per emitted physics rule. The non-DSL `RecordMemoryHandler` is registered alongside.
- **Mask build path** — `MaskBuffer::mark_attack_allowed_if_target_in_range` in `crates/engine/src/mask.rs` calls `crate::generated::mask::mask_attack(state, id, other)` per (agent, candidate) pair discovered via the spatial index.
- **Utility backend** — `UtilityBackend::evaluate` in `crates/engine/src/policy/utility.rs` iterates `engine_rules::scoring::SCORING_TABLE`, evaluates each row's predicate against the current agent, and keeps the highest-scoring mask-allowed `MicroKind`.
- **Entity taxonomy** — `crates/engine/src/creature.rs` re-exports `engine_rules::entities::{CreatureType, Capabilities}`. Spawn-time `AgentSpawn { creature_type, .. }` resolves to the DSL-defined variant; mask-time `is_hostile(state, a, b)` in `crates/engine/src/rules/mod.rs` dispatches through `CreatureType::is_hostile_to`.

## Fixture test

`crates/engine/tests/wolves_and_humans_parity.rs` is the regression anchor. Three tests:

1. `parity_log_is_byte_identical_to_baseline` — renders the 100-tick event log as one line per event and compares against the committed `tests/wolves_and_humans_baseline.txt`. Any DSL edit that changes the wolves+humans trace fails this test. To refresh the baseline (intentionally): `WOLVES_AND_HUMANS_REGEN=1 cargo test -p engine --test wolves_and_humans_parity`.
2. `parity_log_has_expected_structure` — asserts event counts, tick monotonicity, and that the majority of `AgentAttacked` events cross species (validates that the `is_hostile` gate is in effect). Note: same-species attacks can leak through because `UtilityBackend` target-selection uses `nearest_other` without a hostility filter — this is current DSL-owned behaviour and will close when target-selection moves into the DSL (follow-up to milestone 6).
3. `parity_log_is_deterministic_across_runs` — two fresh runs in the same process must produce byte-identical logs.

Run the full suite with `cargo test -p engine --release`. CI runs it in both debug and release so the determinism path is exercised on both contract-check configurations.

## Still hand-written

Honest list of game-logic surfaces that remain hand-written after milestones 2-6 + physics parity. Each entry names the file and the migration prerequisite.

### Cast dispatch (stateful)

- `crates/engine/src/ability/cast.rs` — `CastHandler`. Registered via `CascadeRegistry::register_cast_handler(Arc<AbilityRegistry>)`. The DSL's physics emitter currently produces stateless unit-struct handlers; `CastHandler` carries an `Arc<AbilityRegistry>` so a DSL cutover requires extending the emitter with a "handler state" concept. Tracked as a follow-up to the milestone-3 physics series; documented as deferred in the physics-parity commit.
- `crates/engine/src/ability/gate.rs` — `evaluate_cast_gate`. The cast-time predicate (cooldown + stun + range + hostility + engagement-lock) that the Cast mask and `CastHandler` both consult. Migrates when the Cast mask moves.
- `crates/engine/src/ability/record_memory.rs` — `RecordMemoryHandler`. Not a cast path but also stateless-but-engine-specific; consumes `Event::RecordMemory` and folds into per-agent cold memory. Tracked separately; low priority.

### Hostility view

- `crates/engine/src/rules/mod.rs` — `pub fn is_hostile(state, a, b) -> bool`. The compiler-emitted mask lowers the DSL call `is_hostile(self, target)` to `crate::rules::is_hostile(state, self, target)`. The fn body delegates to `CreatureType::is_hostile_to` (DSL-emitted), so this is a one-line shim. Retires when milestone 6 (`view` declarations on the compiler ladder) lands — the DSL will declare `view is_hostile(a, b)` directly.

### Masks still hand-written

Attack is the only mask the DSL owns. The five others live in `crates/engine/src/mask.rs`:

- `MaskBuffer::mark_hold_allowed` (always-on for alive agents) — trivial.
- `MaskBuffer::mark_move_allowed_if_others_exist` — any-other-alive gate.
- `MaskBuffer::mark_flee_allowed_if_threat_exists` — any-other-within-`AGGRO_RANGE` gate (`AGGRO_RANGE = 50.0` at `mask.rs:10`).
- `MaskBuffer::mark_needs_allowed` — unconditional for `Eat`/`Drink`/`Rest`.
- `MaskBuffer::mark_domain_hook_micros_allowed` — the Cast branch gates through `evaluate_cast_gate`; the other ten micros (UseItem, Harvest, etc.) are permissive. Migration cost: each mask needs a `mask M(...)` declaration in `assets/sim/masks.sim` plus emitter support if the predicate references anything beyond `agents.*` + stdlib. Tracked as a follow-up milestone (part of the "masks" follow-up to milestone 4).

### Scoring still hand-written

The scoring table covers only four rows: `Hold`, `MoveToward`, `Attack`, `Eat`. The other 14 `MicroKind` variants (Flee, Cast, UseItem, Harvest, Drink, Rest, PlaceTile, PlaceVoxel, HarvestVoxel, Converse, ShareStory, Communicate, Ask, Remember) have no DSL scoring row, so `UtilityBackend` never picks them — they land in the policy only via tests that drive actions directly. Migration cost: add rows to `assets/sim/scoring.sim`. The emitter already supports the full `<lit> + (if <pred> { <lit> } else { 0.0 })` grammar.

Target-selection logic (`nearest_other` + the `Action::move_toward` / `Action::attack` builders in `utility.rs`) stays hand-written and is not a scoring row — it's action construction. Migrating target-selection into DSL is a separate, larger milestone.

### Hand-written scratch helpers in step.rs / mask.rs

These aren't "game rules" in the narrow sense but they do encode game-level knobs:

- `crates/engine/src/step.rs:71` — `pub const MOVE_SPEED_MPS: f32 = 1.0;` — movement kernel (MoveToward / Flee both multiply by it). Game knob.
- `crates/engine/src/step.rs:72` — `pub const ATTACK_DAMAGE: f32 = 10.0;` — per-agent fallback used when `SimState::agent_attack_damage` returns `None`. Also referenced by `crates/engine/src/generated/physics/opportunity_attack.rs:33`.
- `crates/engine/src/step.rs:73` — `pub const ATTACK_RANGE: f32 = 2.0;` — per-agent fallback for `SimState::agent_attack_range`. The DSL-emitted `mask_attack` predicate hardcodes the matching 2.0 m literal (`assets/sim/masks.sim:24`).
- `crates/engine/src/step.rs:74` — `const EAT_RESTORE: f32 = 0.25;` — hunger-restore delta applied on `MicroKind::Eat`.
- `crates/engine/src/step.rs:75` — `const DRINK_RESTORE: f32 = 0.30;` — thirst-restore delta applied on `MicroKind::Drink`.
- `crates/engine/src/step.rs:76` — `const REST_RESTORE: f32 = 0.15;` — rest-timer delta applied on `MicroKind::Rest`.
- `crates/engine/src/step.rs:80` — `pub const MAX_ANNOUNCE_RECIPIENTS: usize = 32;` — per-announce event-ring pressure cap.
- `crates/engine/src/step.rs:84` — `pub const MAX_ANNOUNCE_RADIUS: f32 = 80.0;` — default hearing radius for `AnnounceAudience::Anyone`/`Group`.
- `crates/engine/src/step.rs:89` — `pub const OVERHEAR_RANGE: f32 = 30.0;` — bystander overhear radius around the speaker.
- `crates/engine/src/step.rs:18` — `pub const DEFAULT_VOCAL_STRENGTH: f32 = 1.0;` — fallback vocal strength for `channel_range` (per-agent vocal strength lands when the Capability SoA grows the field).
- `crates/engine/src/mask.rs:10` — `const AGGRO_RANGE: f32 = 50.0;` — flee-mask threat-detection radius.
- `crates/engine/src/mask.rs:18` — `const ATTACK_SPATIAL_RADIUS: f32 = 2.0;` — mask spatial-iterator radius; intentionally tied to the DSL literal 2.0 m above.
- `crates/engine/src/ability/expire.rs:29` — `pub const ENGAGEMENT_RANGE: f32 = 2.0;` — engagement detection radius (tick-start bidirectional commit).
- `crates/engine/src/ability/expire.rs:35` — `pub const ENGAGEMENT_SLOW_FACTOR: f32 = 0.3;` — speed multiplier when an engaged agent moves away from its engager.

All of these are game knobs that deserve DSL declarations (`const ATTACK_DAMAGE = 10.0` or equivalent, ideally inside a scoped `balance` block). They land when the DSL grows a `const` or `balance` declaration kind. Each is also covered by `engine::schema_hash::schema_hash` so any silent drift shows up as a hash mismatch in CI.

## What retires next

When you migrate one of the above, the checklist is always the same:

1. Extend the DSL grammar if needed (e.g. a new declaration kind for balance constants).
2. Add the declaration to the appropriate `.sim` file.
3. `cargo run --bin xtask -- compile-dsl`.
4. Delete the hand-written counterpart in the SAME commit.
5. Run `cargo test -p engine --release` — the parity test in this walkthrough plus the 339 engine tests must all still pass.
6. If the migration changes the wolves+humans event log (e.g. a new scoring row picks a different `MicroKind` at the same score), regen the baseline with `WOLVES_AND_HUMANS_REGEN=1` and include the diff in review.

The parity anchor fails loud, which is the point.
