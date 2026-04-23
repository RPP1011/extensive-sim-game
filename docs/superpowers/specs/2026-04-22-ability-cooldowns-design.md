# Ability Cooldowns — Global + Local

**Status:** design approved, plan pending
**Date:** 2026-04-22
**Micro-subsystem:** prerequisite for GPU ability evaluation (subsystem 3)

## Problem

The engine's current cooldown model has a single cursor per agent: `hot_cooldown_next_ready_tick: u32` in the hot SoA (per `schema_hash.rs:21`). Every ability an agent owns shares this cursor. An agent with abilities A (cooldown 5s), B (cooldown 15s), C (cooldown 60s) has them all gated by whichever cast was most recent.

This is wrong by game-design convention and surfaces as a real gameplay bug: casting A forces B and C to wait 5s before they could even possibly be ready, then casting B forces A into a 15s effective cooldown. The MOBA/MMO-standard model is **both** cooldowns together:

- **Global cooldown (GCD)** — a short shared gate (~0.5s) preventing spam across all abilities. Per-agent.
- **Local cooldown** — per-ability, enforces the ability's own refresh time. Per-(agent, ability-slot).

An ability can fire iff `now >= global[agent] && now >= local[agent][ability_slot]`.

Subsystem (3) (GPU ability evaluation) needs per-ability cooldowns to be readable on GPU. Landing this fix first decouples the cooldown-correctness work from the GPU-migration work and keeps (3)'s scope focused.

## Goal

Add per-ability local cooldowns as a new side buffer alongside the existing per-agent global cooldown. CPU and GPU paths both read the combined gate. Consumer call-sites update to query the combined gate instead of the single existing cursor.

## Non-goals

- Migrating the cooldown field into GPU-only storage. This spec lands the correctness fix on the CPU path; GPU-side consumption arrives in subsystem (3).
- Changing the cooldown refresh logic (how cooldowns get reset post-cast). Those events continue to work as today; the delta is *read-side*, not *write-side*.
- Breaking the existing `hot_cooldown_next_ready_tick` field. It keeps its name and position; semantics narrow to "global cooldown only."
- Adding cooldown reduction mechanics, cooldown resets from effects, stackable cooldown charges, etc. Out of scope.

## Architecture

### Data

Keep `hot_cooldown_next_ready_tick: u32` in hot SoA, **relabel semantics to "global cooldown cursor."** No schema change.

Add per-(agent, ability-slot) local cooldowns. Two viable storage shapes:

- **A: Fixed `[u32; MAX_ABILITIES]` on the agent struct**. Simple Rust array; ~32 B per agent at MAX_ABILITIES=8. N=2048 → 64 KB total. On GPU: naturally lives as `array<array<u32, 8>>` or flat `array<u32>` keyed by `agent*MAX + slot`.
- **B: Dedicated side buffer outside agent struct**. Same shape but separate. Decouples from agent SoA layout; easier to extend MAX_ABILITIES later without schema bumps.

Choose **B** — matches the "side buffer" pattern already used by subsystems (2)/(3) and doesn't bloat GpuAgentSlot with game-config-specific state.

### Gate evaluation

Helper function, both CPU and (later, subsystem 3) GPU:

```
fn can_cast(agent: AgentId, ability_slot: u8, now: u32) -> bool {
    state.hot.cooldown_next_ready_tick[agent] <= now
        && state.ability_cooldowns[agent][ability_slot] <= now
}
```

After a successful cast:
- Set `hot_cooldown_next_ready_tick[agent] = now + GCD_TICKS` (existing behaviour, unchanged)
- Set `ability_cooldowns[agent][ability_slot] = now + ability_definition.cooldown_ticks` (new)

### GCD value

Pick a single constant for GCD across all abilities: `GCD_TICKS = 5` (0.5s at 10 Hz tick rate). Matches MMO convention (~1s GCD at 2-5 Hz tick rates, scaled to our 10 Hz). Configurable via `combat.global_cooldown_ticks` in the DSL config block. A per-ability "no-GCD" flag is out of scope (YAGNI).

### CPU consumers to update

Audit all sites that currently read `hot_cooldown_next_ready_tick` directly. Likely ~5 sites:

1. `crates/engine/src/policy/utility.rs` — scoring rules that gate on cooldown.
2. `crates/engine/src/cascade/` handlers for cast-dispatch.
3. `crates/engine/src/generated/scoring/` — generated scoring from `scoring.sim`.
4. Test files that set/check cooldowns explicitly.
5. `src/ai/core/ability_eval/` — CPU ability evaluator (to be migrated in subsystem 3).

Each callsite replaces direct field read with `can_cast(agent, ability_slot, now)`. The `ability_slot` is available at every callsite (it's the slot being evaluated). Compile-time enforcement by making `hot_cooldown_next_ready_tick` a `pub(crate)` field accessed only through the helper.

### DSL surface

The scoring/ability DSL doesn't currently have a first-class primitive for "can this ability cast?" Expression. Add one:

```
ability::on_cooldown(ability_ref)
    // true iff either global or local gate blocks cast
```

Emitted by the compiler to call the `can_cast` helper (inverted). Available in both `scoring.sim` row guards and ability gate predicates.

### Dependency on `@cpu_only`

None. This subsystem lands on the CPU only; GPU exposure happens in subsystem (3). No DSL-emit-skip needed.

## Components

### Modified files

- `crates/engine/src/state/` — new `ability_cooldowns: Vec<[u32; MAX_ABILITIES]>` field on `SimState`. Initialised to zeros on spawn.
- `crates/engine/src/state/agent_types.rs` — add a helper method `can_cast_ability(agent, slot, now)` on `SimState`.
- `crates/engine/src/config.rs` (or equivalent) — `combat.global_cooldown_ticks: u32` config, default 5.
- `crates/engine/src/policy/utility.rs` + `crates/engine/src/generated/scoring/` — callsites switched to `can_cast`.
- `crates/engine/src/cascade/cast_handler.rs` (or wherever cast events are processed) — set both global and local cooldowns post-cast.
- `crates/dsl_compiler/src/` — add `ability::on_cooldown(...)` primitive; emit to `can_cast`.
- Schema hash bump: add `AbilityCooldowns{slots=MAX_ABILITIES,ticks=u32}` string to `schema_hash.rs` so the baseline captures the new state.

### New files

None. Small enough to fit in existing files.

### Engine-core vs DSL-lowered split

- **Engine-core (hand-written Rust)**: `ability_cooldowns` storage on `SimState`, `can_cast` helper, cast-handler writeback of both cooldowns, schema hash entry.
- **DSL-lowered**: the `ability::on_cooldown` primitive in scoring/ability grammar. Compiler emits call to `can_cast`.

## Error handling

- **ability_slot out of range**: `can_cast` clamps to `MAX_ABILITIES - 1` and logs a warning in debug builds. Callsites should already guard since the ability registry is fixed-size.
- **Schema hash drift from tests**: the baseline file `crates/engine/.schema_hash` gets bumped with this change. CI catches regressions the usual way.

## Testing

### New tests

- `cooldowns_global_gate.rs` — spawn agent with abilities A, B. Cast A at tick 0. Assert: at tick 1, B is gated by global cooldown. At tick `GCD_TICKS`, B is ready.
- `cooldowns_local_gate.rs` — spawn agent with ability A (cooldown 10). Cast A at tick 0. At tick `GCD_TICKS + 1`, assert global passed but local still gates. At tick 11, both open.
- `cooldowns_independent_across_slots.rs` — agent with abilities A (cd 10) and B (cd 3). Cast A at tick 0, wait 3 ticks, cast B at tick 3 (should succeed because B's local cd is 3 and global's cleared). At tick 4, assert A still locally gated, B locally gated.
- `cooldowns_schema_hash.rs` — regression assertion on baseline.

### Regression

Existing cast-handler tests + scoring tests must still pass. Any test that manipulates `hot_cooldown_next_ready_tick` directly gets updated to use the helper.

### Non-goals for testing

- No tests of cooldown reduction effects. YAGNI.
- No performance tests. The side buffer is tiny (~64 KB at N=2048); no perf concerns.

## Phase decomposition

Single phase. Small spec, small plan.

## Open questions

- **Should the global cooldown be per-ability-category?** Some games gate casts by ability school (physical/magic/movement) rather than globally. YAGNI for now; revisit if game design calls for it.
- **Testing MAX_ABILITIES > 8**: the current registry is fixed at 8. If subsystem (3) raises it, this buffer grows. Side-buffer choice avoids the schema-hash churn that agent-struct embedding would cause.
