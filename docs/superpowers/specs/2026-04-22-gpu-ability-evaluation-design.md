# GPU Ability Evaluation

**Status:** design approved, plan pending
**Date:** 2026-04-22
**Subsystem:** (3) of the GPU-everything follow-up
**Depends on:** subsystem (1) — needs `SimCfg`; subsystem (2) — else gold-transfer and standing-shift abilities silently no-op on batch path; ability-cooldowns micro-subsystem — else per-slot cooldowns don't exist
**Out of scope:** V5 neural ability head — separate follow-up subsystem

## Problem

After subsystems (1)+(2), almost the entire sim runs on GPU during a batch. The last CPU-resident piece is **ability selection**: per agent per tick, deciding which ability to cast (if any) and at which target.

Current CPU architecture (three-layer stack):
1. Heuristic `evaluate_hero_ability` — hand-tuned rules in `src/ai/core/ability_eval/`.
2. V5 neural transformer (optional, training-time).
3. Pointer-action translator — emits `AgentCast` events.

In batch mode, none of this runs. The cascade's `cast` physics rule (which handles the *effect* of a cast once an `AgentCast` event is in the stream) IS GPU-native, but no GPU equivalent emits the triggering `AgentCast` events in the first place. Abilities simply don't fire during batch.

## Goal

A GPU scoring kernel, compiler-emitted from DSL source, that per-agent selects an ability-to-cast (or none) and writes `chosen_ability` + target. `apply_actions` consumes this side buffer and emits `AgentCast` events for the cascade to handle.

**The scoring primitive is ability tags.** Each ability's `.ability` file carries tag vectors today (`hint: damage`, per-effect `[PHYSICAL: 50]`, etc.). These are exposed through lowering into a GPU-readable lookup. The scoring grammar in `scoring.sim` gains a new primitive (`ability::tag(TAG)`) and a new row type (`per_ability`) that lets designers hand-tune combination rules.

Per the engine architecture rule, **no hand-written WGSL for the ability scoring kernel**. The grammar extension + lowering produces `pick_ability` WGSL as compiler output.

## Non-goals

- V5 neural ability head. Separate subsystem. Also: the model needs retraining first, so porting current weights is wasted work.
- Learned target-selection pointer attention. Heuristic target selection only.
- Chain-casting / cast-in-response (already handled by cascade's `cast` physics rule on the effect side; ability *selection* cascades aren't a goal).
- Reinforcement-learning-driven scoring — this subsystem ports the existing hand-tuned heuristic; RL is a separate concern.

## Architecture

### Scoring primitive — ability tags

Each `.ability` file has:
- `hint:` — coarse category enum (`damage` | `defense` | `crowd_control` | `utility`). One per ability.
- Per-effect `[TAG: value]` — numeric power ratings. Multiple per effect line.

The ability IR (`AbilityDef`) has these tags. Audit: confirm they flow through to the packed registry consumed by kernels. If not fully exposed, Phase 1 of this subsystem closes that gap — serialise per-ability tag vectors into the ability registry buffer.

### New `scoring.sim` primitive: `ability::tag(TAG_NAME)`

Reads the tag value for the ability being scored in the current row. Returns f32 (or 0 if the ability doesn't carry that tag). Example:

```
score = ability::tag(PHYSICAL) * (1 - target.hp_frac) + ability::tag(CROWD_CONTROL) * engaged_with_kin
```

Also reads the ability-level `hint` as a string-enum compare primitive:

```
score = (if ability::hint == damage { 1.0 } else { 0.0 }) * ...
```

### New `scoring.sim` row type: `per_ability`

Today scoring rows run once per agent (one score per agent for each scoring category like "attack nearest," "flee," etc.). `per_ability` rows iterate over an agent's abilities and produce one score per (agent, ability).

Syntax sketch:

```
row pick_ability per_ability {
    guard:    !ability::on_cooldown(ability)
    score:    ability::tag(PHYSICAL) * (1 - target.hp_frac)
            + ability::tag(CROWD_CONTROL) * (if engaged_with_hostile { 0.5 } else { 0.0 })
            + ability::tag(DEFENSE) * (if self.hp_frac < 0.3 { 1.0 } else { 0.0 })
    target:   nearest_hostile_in_range(ability::range)
}
```

Output: per agent, `(ability_slot, target_agent_id)` of the highest-scoring ability whose guard passes. If no ability's guard passes, the agent emits no cast this tick.

### Compiler emit paths (the core of this subsystem)

New DSL-compiler work inside `crates/dsl_compiler/`:

1. **Parser additions**: `per_ability` row type, `ability::tag(...)` primitive, `ability::hint` read, `ability::on_cooldown(...)` primitive (from the cooldowns micro-subsystem), `ability::range` primitive, `nearest_hostile_in_range(...)` (already exists for other rows or added).
2. **IR additions**: `ScoringRowKind::PerAbility`, `IrExpr::AbilityTag(name)`, `IrExpr::AbilityHint`, etc.
3. **CPU emit path**: Rust code that, for each agent, iterates abilities, evaluates guard + score, argmax. Drop-in replacement for the current hand-tuned `evaluate_hero_ability`.
4. **GPU emit path**: WGSL kernel (`pick_ability.wgsl` — compiler-emitted, not hand-written) that does the same per-agent evaluation over the bound ability registry + tag table + per-slot cooldown buffer.

The kernel dispatches once per batch tick, writes `chosen_ability: array<u64>` (packed `(ability_slot: u8, target_agent_id: u32, _pad: u24)`).

### Target selection

Simplest CPU-heuristic-equivalent: **nearest hostile within the ability's range**. Spec explicitly defers pointer-attention / learned targeting / lowest-HP targeting to a future subsystem.

Implementation: the `per_ability` row's `target:` clause calls `nearest_hostile_in_range(ability::range)`. `ability::range` comes from the ability's `Area::SingleTarget { range }` field. `nearest_hostile_in_range` uses the existing GPU spatial hash output (kin + hostile result buffers from `spatial_gpu.rs`), bounded by range.

### Engine-core infrastructure (hand-written Rust + WGSL)

- `chosen_ability_buf: array<u64>` side buffer — one packed u64 per agent (ability_slot, target_agent_id, sentinel-for-no-cast). Consumed by `apply_actions`.
- `ability_cooldowns_buf: array<array<u32, MAX_ABILITIES>>` — landed by the ability-cooldowns micro-subsystem; this subsystem just binds it into `pick_ability`.
- `apply_actions` gains logic: if `chosen_ability_buf[agent]` is non-sentinel and cooldowns pass, emit `AgentCast { caster, ability, target }` event. The `cast` physics rule (already GPU-native) picks it up.
- Wire `pick_ability` kernel dispatch into `step_batch`'s per-tick loop — before `apply_actions`, after scoring.
- `ensure_resident_init` additions: upload tag table (subset of packed ability registry) if not already exposed via Phase 1 IR audit.

### Where `pick_ability` sits in the tick pipeline

```
mask → scoring → pick_ability → apply_actions → movement → spatial → cascade(N iters) → fold → cold_state_dispatch(subsystem 2)
                 ^^^^^^^^^^^^
                 new
```

Scoring runs first and produces per-agent action scores (attack/move/flee/hold). `pick_ability` runs next with its own output buffer. `apply_actions` reads both — if an ability is chosen, emit `AgentCast`; otherwise, emit the action the scoring kernel picked.

### Engine-core vs DSL-lowered split (for this subsystem)

- **Engine-core (hand-written Rust + WGSL)**:
  - Side buffer allocation + binding plumbing for `chosen_ability_buf`.
  - Dispatch of the compiler-emitted `pick_ability.wgsl` kernel in `step_batch`.
  - `apply_actions` extension to read `chosen_ability_buf` and emit `AgentCast` events.
  - Tag table exposure: if per-ability tags aren't fully serialised into the packed ability registry, hand-written Rust code to add them is engine-core.
- **DSL-lowered**:
  - `pick_ability.wgsl` (compiler output from the new `per_ability` row).
  - CPU-side `pick_ability` handler (compiler output, replaces hand-tuned `evaluate_hero_ability`).
  - All tag reads + scoring arithmetic expressible in `scoring.sim`.
  - `scoring.sim` content that expresses the old `evaluate_hero_ability` logic in new DSL primitives.

Zero hand-written game-logic WGSL.

## Data flow — one tick with ability evaluation

```
mask kernel      writes mask bitmaps (unchanged)
scoring kernel   writes score_output (per-agent action scoring, unchanged)
pick_ability     reads: agent_slots, ability_registry, tag_table, cooldowns_buf, spatial_hash_output, sim_cfg
                 writes: chosen_ability_buf (per-agent u64: slot + target, or sentinel)
apply_actions    reads score_output AND chosen_ability_buf
                 if chosen_ability is non-sentinel: emit AgentCast + apply cooldowns
                 else: emit score_output's chosen action
                 (unchanged rest of behaviour)
```

## Error handling

- **Ability tag missing**: `ability::tag(UNKNOWN_TAG)` returns 0 silently. Not an error — tags are sparse per ability. Spec-compile-time warning for unknown tag names is a later nicety.
- **All abilities on cooldown**: `pick_ability` writes sentinel; `apply_actions` falls through to `score_output`. Ordinary behaviour.
- **Tag value overflow**: tags are `f32` in WGSL, scored with hand-tunable weights. Designer-visible; if a combination overflows it's a design error, not a runtime error.
- **Cooldown buffer not initialised**: depends on ability-cooldowns micro-subsystem landing first. Verify at `ensure_resident_init`.

## Testing

### New tests

- `pick_ability_respects_cooldowns.rs` — spawn agent with two abilities on different cooldowns, cast one, assert next tick picks the other, not the first.
- `pick_ability_targets_nearest_hostile_in_range.rs` — spawn agent with short-range + long-range abilities, place hostiles at varying distances, assert range-gated targeting.
- `pick_ability_batch_produces_cast_events.rs` — step_batch(50) with cast-happy fixture, snapshot events, assert `AgentCast` events appear (today they wouldn't — abilities don't fire on batch).
- `pick_ability_vs_sync_statistical.rs` — compare cast counts between sync and batch paths on the same fixture. Allow ±25% (non-deterministic); tighter would be hard-coding numerical specifics of target selection.
- `dsl_per_ability_row.rs` — DSL-compiler-level test: minimal `per_ability` scoring row compiles, lowers to CPU + GPU emit paths without errors, runs on a tiny fixture.
- `dsl_ability_tag_primitive.rs` — compiler-level: `ability::tag(PHYSICAL)` in a scoring expression lowers correctly, returns expected value.

### Regression

All existing tests pass. Sync `step()` continues to use the old `evaluate_hero_ability` (until Phase 4 swaps it to the DSL-emitted CPU handler). The batch path gains functionality, doesn't lose any.

### Non-goals for testing

- No byte-exact parity with the old CPU heuristic. The port from hand-tuned Rust code to DSL tag-combinations necessarily introduces drift (the old heuristic has hand-coded thresholds that map imperfectly to tag weights). Statistical parity (cast cadence, ability-type distributions) is the contract, per the same non-determinism disclaimer as the GPU-resident cascade spec.

## Phase decomposition

**Phase 1 — Ability tag exposure audit**. Confirm `AbilityDef.tags` fully serialise into the packed ability registry that kernels consume. If not, bridge it. CPU-only work inside the DSL compiler.

**Phase 2 — DSL grammar extensions**. `per_ability` row type, `ability::tag`, `ability::hint`, `ability::range`, `ability::on_cooldown`. Parser + IR + test fixtures.

**Phase 3 — CPU emit path**. Lower `per_ability` to Rust code that replaces `evaluate_hero_ability`. Sync path swaps over. Regression tests for equivalent gameplay behaviour (statistical parity).

**Phase 4 — GPU emit path + engine-core wiring**. Compiler emits `pick_ability.wgsl` from `per_ability` rows. `step_batch` gains the dispatch. `apply_actions` reads `chosen_ability_buf` to emit `AgentCast`. Integration tests.

**Phase 5 — Port CPU heuristic to DSL**. Rewrite the hand-tuned `evaluate_hero_ability` logic as `scoring.sim` rows using tag combinations. This is game-design work, not infrastructure work — the sim ships when the port produces acceptable cast behaviour on representative scenarios.

Phase 5 is where the realistic project timeline absorbs most of the effort (the tag-weight-tuning is iterative and game-design-driven). Phases 1-4 are ~2 weeks of engineering; Phase 5 is open-ended depending on desired gameplay fidelity.

## Open questions

- **Tag registry shape**: are tag names (PHYSICAL, CROWD_CONTROL, etc.) a fixed enum, or user-extensible strings? Fixed enum is simpler to lower (each tag becomes a known buffer index); string-identified tags need a symbol table. Recommend fixed enum for v1; extensibility via DSL grammar later.
- **Ability range sourcing**: `ability::range` comes from `Area::SingleTarget { range }` today. What if future `Area` variants (Cone, Circle) carry multiple range-relevant fields? Defer until those variants exist.
- **Per-agent ability count variance**: different heroes have different ability counts. The `per_ability` row evaluates all slots but slots with no ability score 0. Could be wasteful if most heroes have 3 abilities and MAX_ABILITIES = 8. Probably fine; 5 wasted evaluations per agent per tick at N=2048 is negligible.
- **Cascade between cast events**: casting an ability triggers `cast` physics rule, which can emit further events. Can a single agent's cast in tick T lead to another `AgentCast` event in tick T from the same agent? Probably no — agents pick one cast per tick via `pick_ability`, and the cascade effect handles the downstream events. But worth confirming during Phase 4.
- **Neural head landing later**: the follow-up neural-head subsystem will produce an alternative `chosen_ability_buf` writer. Design point: should the two writers coexist (toggle at backend init) or replace? Defer until that subsystem exists.
