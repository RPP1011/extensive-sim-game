# Gaps surfaced by the `dungeon_layout` fixture

**Fixture:** `assets/sim/dungeon_layout.sim` (28 emitted kernels)
**Pin:** `crates/sims/tests/dungeon_layout_pin.rs`
**Date:** 2026-05-12
**Stage:** 1 of 3 (the dungeon-crawl series; stage 2 will add hero
stealth + enemy alert + bilateral beliefs; stage 3 will scale to 1000+
agents).

This fixture is the first to stress voxel-procedural terrain at scale
(72×72×8 grid, 6×6 room slots × CA-noisy interiors), the custom-field
registry for per-agent SoA columns (`field role: u32`, `field
target_room_idx: u32`), and host-side-driven exploration via direct
GPU buffer writes between ticks. It runs 5 heroes + ~30 mixed enemies
(Goblin/Brute/Archer) for 500 ticks.

## Pass behaviour

```
==== dungeon_layout 500-tick report ====
  dungeon: 18 rooms (~2500 floor cells, spawn=slotN, boss=slotM)
  init:    heroes=5/5  enemies=~30
  final:   heroes=K/5  enemies=remaining
  combat:  total enemy kills = >= 10
  hero hp/rooms-visited:
    hero[0] role=Warrior ... rooms_visited >= 3
    ...
  verdict: PARTY ADVANCING — meaningful enemy attrition
```

Load-bearing pin asserts:

  1. All 5 heroes alive at tick 30 (no early-game wipe).
  2. Every hero has visited ≥ 3 distinct rooms by tick 200.
  3. Total enemy kills ≥ 10 by tick 400.
  4. No NaN/Inf in agent positions after 500 ticks.

## Discovered gaps

### Gap #1 — `spatial.nearby_*` returns out-of-range candidates due to cell-index clamp

**Status:** worked around in-fixture via per-pair `length(...) <
range` gates inside every verb body's if-filter. The underlying
spatial-cell-clamp behavior is the compiler's, not this fixture's.

**Discovery sequence (2026-05-12):**

  1. Without per-pair distance gates, heroes clustered at spawn
     centroid `(6, 30)` took 75 damage_dealt each from `HeroStrike`
     in the first 5 ticks even though the closest enemy was 21.27
     units away. The if-filter `candidate.creature_type != Hero`
     should have rejected hero-self-as-target candidates from the
     spatial walk.
  2. Disabling `HeroStrike` revealed Goblin `apply_ability 7`
     dispatches from slot 25 at world-pos `(58.5, 5.5)` (cell `(20,
     11, 10)`) hitting heroes at world-pos `(6, 30)` (cell `(11, 15,
     10)`). The two cells are 9 grid cells apart on the x axis —
     well outside the spatial walk's 27-cell window
     (`dx,dy,dz ∈ -1..=1`).
  3. The smoking gun: WGSL `cell_index(cx, cy, cz)` clamps any
     out-of-bounds component to `grid_dim - 1` (`21u`). For a
     world spanning 0..72 worldcoords, positions with x>60 all
     map to cell column `cx=21` (the boundary). The 27-cell walk
     centered on `(20, 11, 10)` includes `cell_index(21, ...)` —
     and after clamp, **any walk on a boundary-side cell shares
     bucket contents with every other walk that touched the same
     clamped cell across the world.** Effectively, the spatial
     grid silently pools distant agents into the boundary cell.
  4. Adding `length(candidate.pos - self.pos) < <range>` inside
     every verb's `if (...)` makes the filter explicit and
     bypasses the clamp pooling.

**Fix surface (engine-side, deferred to follow-up):**

  - Option A: bump `SPATIAL_GRID_DIM` so common world sizes (72,
    100, 256) fit unclamped at the current 6.0-unit cell size.
    Compile-time const today (`crates/dsl_compiler/src/cg/emit/...`).
  - Option B: emit a per-verb distance gate automatically when the
    .sim's `@spatial(radius=R)` is declared, mirroring the
    semantics of the AOE Path B dispatcher's `_dist_sq > range_sq`
    check.
  - Option C: enforce range from the .ability metadata
    (`ability { range: 4.0 }`) inside `apply_ability`'s damage
    dispatcher. Today `range:` is a slot-metadata field that the
    chronicle dispatcher reads only inside AOE walks; single-target
    effects (Damage / Stun / Heal kind=0/27/29) trust the caller
    to enforce range.

Until one of those lands, every .sim author has to manually re-
state the per-pair distance gate in the verb body. The pattern is
visible across `hill_raid` (`fire_radius` declared but not used),
`palace_coup` (uses melee/los/decree radii but not actually
checked per-pair), and now `dungeon_layout` (explicit per-pair
gates added). Adding the auto-gate is a one-arm change in
`cg/emit/wgsl_body.rs::emit_for_each_neighbor_body`.

### Gap #2 — enemy charge phases through walls

**Status:** masked in stage 1 by `length(...) < charge_range` gates
in `GoblinCharge` and `BruteCharge`. Stage 2 plans to gate movement
on `terrain.walkable(new_pos, MovementMode::Walk)`.

The fixture's enemies use the normalize-direction sum-of-deltas
shape `hill_raid` uses: `toward = sum(heroes) - self.pos`. There's
no terrain-walkable check before applying the step. With voxel
walls between rooms, an enemy who clips a wall mid-step doesn't
get bounced — it teleports through.

Behavioural impact in stage 1: minimal, because the per-pair
charge_range gate keeps enemies idling unless heroes have already
walked into adjacent rooms (carved by doorways). Stage 2's stealth
mechanics need real movement bounds; the fix surface is adding
`if (!terrain.walkable(new_pos, Walk)) { new_pos = self.pos; }`
into both Charge rules, OR — cleaner — adding a `@movement(walk)`
verb annotation that lowers to walkable-gated `set_pos`.

### Gap #3 — `apply_ability` single-target dispatcher ignores `range:` metadata

**Status:** same root cause as Gap #1 (single-target dispatch
trusts the caller). The .ability files declare `range: 4.0` /
`range: 18.0` etc. The dispatcher's damage emit at
`kind == 0u` reads `payload_a` (= damage amount) but NOT the
ability's range. Out-of-range single-target casts produce
chronicle damage records that hit the target regardless of
distance.

The AOE branches (`Circle`, `Spread`, `Cone`) DO honor
`area_args[*]` for range, but the **single-target Damage / Heal /
Stun kinds (id 0 / 1 / 2)** dispatch unconditionally once the
caller has set `target_slot`.

Fix surface: in the per-effect dispatch loop in
`crates/dsl_compiler/src/cg/emit/wgsl_body.rs`, add an early
`if (length(target_pos - caster_pos) > ability_registry_range[ability_slot])
{ continue; }` check before each chronicle emit. The metadata
is already in `ability_registry_*`; just isn't read.

## Stage 2/3 bolt-on plan (informational, not gaps)

Stage 1 holds these slots stable for stage 2/3 backfill:

  - `Backstab.ability` (id 1), `Scout.ability` (id 4),
    `Stealth.ability` (id 6) parse cleanly but no verb dispatches
    them. Stage 2 wires `Stealth` to Rogue (gate `role == 5 && hp
    >= 100 && enemy_visible`), drives bilateral detection via the
    pair-keyed `threat_taken` view + a new
    `belief_enemy_alert(observer, subject)` pair_map view,
    bolts `Backstab` to stealthed-Rogue → enemy-back-line, and
    drives `Scout` as a long-range belief broadcast.
  - The unused `rooms_visited_lo` / `current_room_idx` host-side
    tracker promotes to GPU SoA columns when stage 2 needs them
    for kernel-side belief gating.

## What this fixture proves works

  - **`field <name>: <type>` registry at scale**: `agent_role_buf`
    + `agent_target_room_idx_buf` auto-allocate, route through the
    fused/standalone physics-rule kernels, and read correctly from
    host-side direct writes between `state.step()` calls.
  - **Voxel terrain at 72×72×8** (28x previous test-pin scale):
    the `voxel_terrain.set_cell` + `voxel_mirror.mark_dirty` pair
    handles thousands of dirty cells per pin and flushes correctly
    before the first kernel runs.
  - **Host-side roomgen + procedural-content pipeline**: 18-room
    BoI-style dungeons re-roll deterministically across runs
    keyed by `SEED_U64`. The CA-noisy interiors make per-pair
    LoS values genuinely variable (not all "open hallway").
