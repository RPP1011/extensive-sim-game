# Gaps surfaced by the `dungeon_horde` fixture

**Fixture:** `assets/sim/dungeon_horde.sim` (36 emitted kernels)
**Pin:** `crates/sims/tests/dungeon_horde_pin.rs`
**Date:** 2026-05-12
**Stage:** 3 of 3 (the dungeon-crawl series; final stage — scales the
stage 2 design to 5v~800 by packing rooms with 20-100 enemies each
instead of 1-3).

This fixture is the first to exercise the stage 2 information-asymmetry
layer (bilateral beliefs, hero stealth, enemy alert, sound broadcast,
patrols) at horde scale. It validates that the same 36-kernel schedule
that works at N=35 still steps cleanly at N≈800 and that the
spatial-grid migration of `MissingAllySuspicion` produces non-zero
alert outputs.

## Pass behaviour

```
==== dungeon_horde 100-tick report ====
  dungeon: 22 rooms (3059 floor cells, spawn=slot12, boss=slot35)
  init:    heroes=5/5  enemies=797 (archers=291 brutes=80 goblins=426) patrol=135
  final:   heroes=5/5  enemies=765 (archers=280 brutes=80 goblins=405)
  combat:  total enemy kills = 32
  stealth: any-hero-stealthed-observed=true
  alert:   16/765 alive enemies have alert>0  (max alert=3)
  verdict: PARTY EXPLORING — combat ongoing against horde
```

Load-bearing pin asserts:

  1. All 5 heroes alive at tick 30 (early-game safety — stealth +
     LoS-gated detection prevent any wipe at horde scale).
  2. No NaN/Inf in agent positions after the run.
  3. Stealth round-trip end-to-end (some hero observed with
     `stealth_until_tick > tick` at some sample point).
  4. Combat fired at scale: ≥5 enemy kills in 100 ticks (verifies
     the verb cascade still engages real targets at N≈800).
  5. `MissingAllySuspicion` produces output: ≥1 enemy with alert>0
     (verifies the spatial-migrated sum walk fires correctly at
     scale).

The test runs in ~4 seconds at N≈800 / 100 ticks (release build),
demonstrating the per-tick cost is dominated by the spatial-grid
build (constant ≈10ms) plus the `for_each_agent` walks in the three
post-phase consumers (linear in N per event, but events are sparse).

## Discovered gaps

### Gap dungeon_horde#1 — `spatial.nearby(<expr>)` references unbound `agent_id` in `@phase(post)` bodies

**Status:** worked around in-fixture by leaving three post-phase rules
on `for_each_agent`. Migration was attempted, codegen accepted the
input, runtime crashed at WGSL parse time.

**Affected rules** (would have benefited from spatial-grid migration):
  - `SoundDetectFromDamage`  — fires on hero-sourced `Damaged`; walks all N
  - `BroadcastAlertOnAllyDeath` — fires on `AllyDied`; walks all N
  - `ScoutBroadcast` — fires on Rogue Scout `EffectDamageApplied`;
    walks all N

**Discovery sequence:**

  1. Per the stage 3 design spec (commit 375cfc77), `spatial.nearby_*`
     walks now auto-inject a squared-distance gate sized to the
     cell-window 3D-diagonal. The intent: the four scale-hotspot rules
     in stage 2 should be migrated to `spatial.nearby(<id>)` to avoid
     O(N) walks at horde scale.
  2. The migration was straightforward syntactically — replace
     `for_each_agent slot { ... }` with `for slot in spatial.nearby(s) { ... }`
     where `s` is the event-bound source agent. Compiler accepted
     the rewrite without diagnostics; all 36 kernels emitted; one of
     them was `physics_SoundDetectFromDamage_and_BroadcastAlertOnAllyDeath`
     (fused) plus `physics_ScoutBroadcast`.
  3. wgpu's naga validator rejected the WGSL at runtime:
     ```
     Shader '...physics_SoundDetectFromDamage_and_BroadcastAlertOnAllyDeath::wgsl' parsing error:
     no definition in scope for identifier: `agent_id`
        ┌─ wgsl:73:51
        │
     73 │ let _self_cell_f = (agent_pos[agent_id] + vec3<f32>(SPATIAL_WORLD_HALF_EXTENT)) / SPATIAL_CELL_SIZE;
        │                               ^^^^^^^^ unknown identifier
     ```
  4. The auto-injected cell-window selection emits
     `agent_pos[agent_id]` as the spatial query origin. In
     `@phase(per_agent)` rules, `agent_id` is the implicit per-agent
     loop variable; in `@phase(post)` event-handler bodies, no such
     binding exists (the implicit loop is over `event_idx`, the agent
     identity is event-field-bound — `s` / `d` / `a` in the three
     affected rules).

**Fix surface (engine-side, deferred):**

  The spatial walk's gate-emit needs to use the *actual* origin
  expression supplied to `spatial.nearby(<expr>)` rather than an
  unconditional `agent_pos[agent_id]`:

  - `crates/dsl_compiler/src/cg/lower/spatial.rs` plumbs the spatial
    iter call's first arg into the iter shape; today it appears to be
    discarded for the gate-prefix emit.
  - `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` emits the gate
    prefix (the `_self_cell_f = (agent_pos[agent_id] + ...)` block).
    Either:
    (a) read the spatial iter's bound origin from the IR and substitute
        it into the gate prefix, OR
    (b) bind a local `let agent_id = <origin_expr>;` immediately above
        the gate prefix when the surrounding rule is `@phase(post)`.
  - Verification: re-attempt the migration in this fixture; expect
    36 kernels emit + green pin run (with the same asserts).

**Workaround impact:** at N=800, each Damaged/AllyDied/EffectDamageApplied
event triggers a full-population O(N) walk. Profiling at this scale
showed the run completes in 4 seconds (release), so the for_each_agent
fallback is acceptable for stage 3. At N≥10k the cost scales linearly
with population × event count — the gap should close before any
horde-scale fixture beyond stage 3 lands.

## Architectural notes

### Spatial-grid extent and grid sizing

The spatial-grid `WORLD_HALF_EXTENT = 64` (set in
`crates/dsl_compiler/src/cg/emit/spatial.rs`) hard-caps positions to
the [-64, +64] world-cube. Stage 3 keeps the voxel grid at 72×72
(positions 0..72) for compatibility — the 64..72 column lands in the
boundary cell (silent clamp). A 96×96 grid was considered but
rejected: positions 64..96 would all pile into the boundary cell,
collapsing spatial culling for the most populated rooms. Future
stages (or an N=10k fixture) should either bump `WORLD_HALF_EXTENT`
or accept the clamp's perf impact.

### MissingAllySuspicion at scale

The migrated rule fires every 30 ticks for every alive non-hero. At
N=800 with 765 surviving enemies and a 12-unit `missing_ally_radius`,
each enemy walks its 27-cell window (~30-50 candidates per cell at
this density) instead of all 800. The per-tick cost drops from
800 × 800 = 640k distance checks to ~800 × 50 = 40k — a 16× speedup.
The pin's `final_enemies_with_alert ≥ 1` assertion verifies the rule
produces correct output (non-zero alert deltas) under this rewrite.

### Information-deficit observability

The pin samples per-hero hp every 5 ticks; if hp drops while no hero
believes-detects any Archer, that's an info-deficit moment (hero hit
by an unseen ranged enemy). At the current 100-tick run, this signal
fires inconsistently — heroes don't take damage in the first 100
ticks of every roll because the spawn-room neighbours are
mostly Goblins (slow melee, easy to outrun). Tuning the seed or
boosting Archer density in BFS-distance-1 rooms would make the
signal more reliable. The assertion is *not* load-bearing today (it
prints status only). Promoting it would require either denser
Archer placement or a longer tick budget.
