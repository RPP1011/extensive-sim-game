# World Sim Tick Flow

A single `WorldSim::tick()` call advances the world by 0.1 seconds of game time (DT_SEC = 0.1). At steady state (post-warmup, mid-simulation) on the default world, a tick currently runs at ~466 ticks/sec, ~2.15 ms per tick.

This document traces the phases of a tick end-to-end and names the functions and data structures involved at each step.

## Top-level phases

```
WorldSim::tick()
├── 0. PREP         — clear per-tick buffers, rebuild spatial index
├── 1. COMPUTE      — iterate entities + systems, emit WorldDelta buffer
├── 2. MERGE        — collapse deltas into FlatMergedDeltas (dirty-bit arrays)
├── 3. APPLY        — write MergedDeltas back into entity/settlement fields
├── 4. POST-APPLY   — ~30 sequential systems mutate state directly
└── 5. SYNC         — sync hot arrays, compact dead entities, record tick profile
```

Phases 1–3 are the “delta system”: all writes are buffered as data, batched, then applied. Phase 4 is the “post-apply” path: systems that mutate state directly (for operations that don't fit cleanly into deltas, or that need ordering guarantees).

Entry point: `src/world_sim/runtime.rs:1488` — `WorldSim::tick(&mut self) -> TickProfile`.

---

## Phase 0 — Prep

```rust
self.state.structural_events.clear();
self.spatial.rebuild(&self.state.entities);
```

1. **Clear per-tick event buffers** — `state.structural_events` (voxel collapse events from the previous tick) are dropped here so only this tick's events remain.
2. **Rebuild spatial index** — `SpatialIndex::rebuild` (`src/world_sim/spatial.rs`) reads hot-array positions and produces a cell grid for O(K) nearest-neighbor queries. Cost is linear in entity count; this is called every tick so downstream systems can use the index without caching.

---

## Phase 1 — Compute

Two loops produce a `Vec<WorldDelta>` buffer (`self.delta_buf`) without mutating state.

### 1a. Per-entity fidelity-dispatched compute

```rust
for i in 0..self.state.hot.len() {
    let h = &self.state.hot[i];
    if !h.alive { continue; }
    match hot_entity_fidelity(h, &self.state) {
        Fidelity::High       => compute_high::compute_entity_deltas_into(...),
        Fidelity::Medium     => compute_medium::compute_entity_deltas_into(...),
        Fidelity::Low        => compute_low::compute_entity_deltas_into(...),
        Fidelity::Background => { /* skipped */ }
    }
}
```

The sim has a four-tier fidelity ladder. Entities in grids with active combat run at High fidelity (full combat resolution); most NPCs in a quiet settlement run at Low or Background. `hot` is a densely-packed array of pared-down entity data — `alive`, `pos`, `team`, `grid_id`, `kind` — so the dispatch loop is cache-friendly.

- **compute_high**: full combat — abilities, damage, status effects, ability cooldowns. Emits most of the combat deltas.
- **compute_medium**: movement + facing, no combat resolution.
- **compute_low**: trickle-level updates (needs drift, passive state).
- **Background**: counted only; no per-entity work.

### 1b. Campaign systems

```rust
if entities.len() > 10_000 {
    self.compute_campaign_systems_par();  // rayon over settlements
} else {
    systems::compute_all_systems(&self.state, &mut self.delta_buf);
}
```

Campaign systems don't iterate entities one-by-one; they iterate settlements, grids, quests, factions, or economic state. Two invocations in `systems/mod.rs`:

- `compute_settlement_systems` — ~45 systems, each scoped to a single settlement (with shortcut: iterate entities once and skip non-matching `home_settlement_id`; entities are sorted by settlement so the branch predictor handles it). Examples: `economy`, `food`, `progression`, `buildings`, `crafting`, `festivals`, `battles`.
- `compute_global_systems` — ~40 systems that span all settlements: `travel`, `threat`, `weather`, `migration`, `faction_ai`, `diplomacy`, `civil_war`, `chronicle`, `marriages`, `grudges`, `auction`, `black_market`, `artifacts`, `charter`, `victory_conditions`, etc.

Each `run_system!` expands to `{ $name: $call(state, out); }` with optional profiling wrapper. Systems emit deltas into the shared `out: &mut Vec<WorldDelta>`.

### 1c. Grid (zone) compute

```rust
for i in 0..fidelity_zones.len() {
    compute_grid_deltas_into(&fidelity_zones[i], &self.spatial, &mut self.delta_buf);
}
```

Fidelity zones are the game's tactical regions. This phase emits grid-level deltas (fidelity escalation, battle state).

**Output of Phase 1**: `self.delta_buf: Vec<WorldDelta>` — typically ~8,000 deltas per tick in the default world.

---

## Phase 2 — Merge

```rust
self.merged.clear();
for delta in self.delta_buf.drain(..) {
    self.merged.merge_one(delta);
}
```

`FlatMergedDeltas` is a struct-of-arrays keyed by entity id: `damage[i]`, `heals[i]`, `shields[i]`, `force_x[i]`, `force_y[i]`, plus a dense `entity_dirty: Vec<u32>` of touched ids. Merging:

- **Damage/heal/shield**: additive. Multiple `DealDamage { id: 42, amount: 10 }` deltas accumulate into `damage[42] += 10`.
- **Movement forces**: additive.
- **SetPos (teleport)**: last-write-wins, recorded as `(entity_id, pos)` pairs.
- **Settlement-level**: `treasury_delta`, `food_delta`, etc., accumulate similarly.
- **World events / chronicle / price reports / trade completions**: pushed to vectors (ordered).

The merge is per-tick dense-array work with no hashing. `ensure_capacity` grows the arrays when `next_id` increases.

---

## Phase 3 — Apply

`apply_flat(&mut state, &self.merged) -> ApplyProfile` (`runtime.rs:509`) writes merged deltas into the real state. The order matters — later applies read what earlier applies wrote.

Sub-phases, each timed:

1. **HP** — walk `entity_dirty`, apply `(heal - damage)` + shield deltas; revive dead entities that receive heal-only.
2. **Movement** — first apply teleports (`SetPos`), then apply force-based movement for the rest (clamped to `move_speed * DT_SEC`).
3. **Status effects** — stun, slow, burn, bleed applications.
4. **Economy** — treasury/food/gold transfers at settlement and entity level.
5. **Deaths** — entities whose `hp <= 0` get `alive = false` and an `EntityDied` world event emitted.
6. **Grid** — fidelity-zone escalations, battle state transitions.
7. **Fidelity** — settlement-level fidelity updates.
8. **Price reports** — append to rolling market history.

After `apply_flat` returns, `state.tick += 1`.

---

## Phase 4 — Post-apply

Post-apply runs ~30 systems that mutate `state` directly. Called in two groups:

### 4a. Early post-apply helpers (unconditional)

```rust
self.process_world_events();        // react to EntityDied etc.
self.process_trade_completions();   // record profitable trades
if tick % GRID_MEMBERSHIP_INTERVAL == 0 { self.update_grid_membership(); }
if tick % CLASS_MATCHING_INTERVAL == 0 { self.run_class_matching(); }
```

- `process_world_events` handles Died/Conquered/etc. for downstream systems.
- `process_trade_completions` records profitable caravan completions so `trade_routes` can establish new routes.
- `update_grid_membership` reassigns entities to the nearest grid — this is what lets a monster entering a settlement trigger fidelity escalation → next tick it runs at High fidelity.
- `run_class_matching` promotes NPCs into new classes based on accumulated behavior tags.

### 4b. Settlement systems (`run_settlement_sys!` block)

Roughly 30 systems that need mutable `state` access:

```
advance_movement, advance_monster_ecology, advance_death_consequences,
update_agent_inner_states, evaluate_and_act, advance_pathfinding,
advance_work_states, scan_all_npc_resources,
structural_tick (every 10 ticks),
advance_debt, advance_contracts, update_building_specializations,
advance_social_gatherings, advance_adventuring, advance_sea_travel,
advance_item_durability, advance_titles, advance_haunted,
advance_world_ages, advance_oaths, advance_monster_naming,
advance_cultural_identity, advance_warfare, advance_succession,
advance_legends, advance_prophecies, advance_outlaws,
advance_trade_routes, advance_trade_guilds, advance_settlement_founding,
advance_betrayal, advance_family, sync_stockpiles_from_buildings,
tick_resource_regrowth, (inventory sync loop), advance_interiors,
sync_npc_actions
```

Key systems whose cost has shown up in the flamegraph:

- **`update_agent_inner_states`** — drifts NPC needs/emotions/morale, accumulates behavior tags from personality pressure and world events (grief, war, famine). Every 5 ticks.
- **`scan_all_npc_resources`** — per-cell census of harvestable voxel materials around each NPC's sight disk; populates `cell_census`. Every 20 ticks. Uses `FlatSurfaceGrid` warmed at world load.
- **`evaluate_and_act`** — action selection for each NPC: scores candidate actions (eat, work, travel, fight, socialize) using behavior tags + needs, picks highest-scoring.
- **`structural_tick`** — voxel collapse (unsupported blocks fall), every 10 ticks.

All post-apply systems are sequential. Some are scoped to a single cadence (`tick % INTERVAL == 0`) and short-circuit otherwise.

### 4c. Late post-apply

```rust
self.grow_cities();                                           // legacy CA, disabled
systems::construction::advance_construction(&mut self.state); // BuildSeed → rooms via flood-fill
if tick % ENTITY_COMPACTION_INTERVAL == 0 {
    state.compact_dead_entities();
}
```

`advance_construction` runs every 10 ticks, processing each `BuildSeed` (placed by an NPC with an intent to build) by flood-filling the current floor tiles, expanding if too small, or closing with walls if at minimum size.

---

## Phase 5 — Sync + profile

```rust
if entities.len() != hot.len() {
    state.rebuild_all_indices();   // spawned/removed entities → resort by (settlement, party)
} else {
    state.sync_hot_from_entities();// scalar copy, no re-sort
}
profile.total_us = tick_start.elapsed().as_micros() as u64;
```

The hot array is the cache-dense projection of entities used by Phase 1's dispatch loop. Most ticks only need a scalar sync (`pos`, `alive`, `team`, `grid_id`). On spawn/removal, the full index gets rebuilt, including:

- `entity_index[id] -> idx` — O(1) lookup table used throughout.
- `group_index` — contiguous ranges by `(settlement_id, party_id)`.
- `settlement_index` — contiguous ranges per settlement.

---

## Cadences

Systems don't all run every tick. Key intervals:

| Interval | Every (ticks) | Every (sec) | Systems |
|----------|--------------:|------------:|---------|
| hot path | 1             | 0.1         | compute_high/medium/low, spatial rebuild, apply_flat, most campaign systems |
| fast     | 3             | 0.3         | exploration (tile-level) |
| medium   | 5             | 0.5         | agent_inner |
| medium   | 10            | 1.0         | structural_tick, construction |
| medium   | 20            | 2.0         | scan_all_npc_resources |
| slow     | 100           | 10.0        | succession, legends, prophecy |
| slowest  | 500           | 50.0        | compact_dead_entities, BuildSeed prune |

Systems check `state.tick % INTERVAL != 0 { return; }` at the top and noop on off-ticks.

---

## Key data structures

- **`WorldState`** (`state.rs`) — single source of truth. ~40 top-level fields: `entities`, `hot`, `settlements`, `tiles`, `voxel_world`, `regions`, `fidelity_zones`, `world_events`, `chronicle`, `economy`, `factions`, `build_seeds`, `group_index`, `sim_scratch`, `surface_grid`, `cell_census`.
- **`Entity`** (`state.rs`) — flat struct with `id`, `pos`, `hp`, `team`, `grid_id`, `kind`, plus optional heavy per-kind data: `npc: Option<Box<NpcData>>`, `building: Option<Box<BuildingData>>`, `monster: Option<Box<MonsterData>>`, etc. Boxed to keep the Entity array slim and cache-friendly.
- **`HotEntity`** — a compact `{id, pos, team, grid_id, kind, alive}` slice used by the Phase 1 dispatch loop. Synced each tick.
- **`WorldDelta`** — enum of all possible writes. Examples: `DealDamage`, `HealTarget`, `AddShield`, `ApplyForce`, `SetPos`, `TransferGold`, `UpdateTreasury`, `EscalateFidelity`, `SpawnEntity`, `AppendChronicle`.
- **`FlatMergedDeltas`** — struct-of-arrays view after merge: `damage[i]`, `heals[i]`, `shields[i]`, etc., indexed by `entity_id`.
- **`SimScratch`** — pool of reusable buffers: `flood_visited`, `flood_queue`, `flood_interior`, `flood_boundary`, etc. Taken with `std::mem::take`, used, then written back. Eliminates per-tick allocations in hot loops.
- **`FlatSurfaceGrid`** — dense per-settlement `Vec<i16>` of surface heights, populated once by `warm_surface_cache` at world load. Replaces the earlier `SurfaceCache: HashMap<u64, i32>` in `compute_cell_census`'s hot path.
- **`CellCensus: HashMap<(i32, i32), [u32; N], ahash>`** — per-resource-cell count of harvestable materials in the surface band. Populated lazily from voxel data as NPCs' sight disks reach new cells.

---

## Warm-up (`xtask world-sim ...`)

Before the first tick:

1. `WorldSim::new(seed)` — generate region plan (biomes/rivers/roads via noise), place settlements, spawn starter NPCs/buildings, generate voxel chunks near settlements.
2. `warm_surface_cache(state)` — for each settlement, populate a dense `FlatSurfaceTile` of surface heights (±margin around the settlement center). Hoists fbm_2d cost out of the first ~500 ticks.
3. Tick loop runs for `--ticks N`. Each call returns a `TickProfile`.

---

## Performance characteristics (current)

Default world, 5000 ticks, release build:

- **466 ticks/sec** (~2.15 ms/tick average).
- Wall time: ~10.7 s.
- ~8,000 deltas per tick.
- `WorldSim::tick` self+children: ~10% of total program time (the rest is warmup: fbm_2d in terrain gen + warm_surface_cache).

Breakdown inside `WorldSim::tick` (flamegraph, post recent optimizations):

| Function                                  | % of total |
|-------------------------------------------|-----------:|
| `flood_fill_with_boundary` (construction) |       4.3% |
| `apply_flat`                              |       3.1% |
| `evaluate_and_act` (action_eval)          |       2.0% |
| `update_agent_inner_states`               |       1.6% |
| `scan_all_npc_resources` + census         |       1.3% |
| `advance_party_quests` (adventuring)      |       0.9% |
| misc post-apply systems                   |       ~2% |

Small world (single settlement, smaller extent): ~100,000 ticks/sec.

---

## Determinism

All randomness flows through `state.rng_state` via `next_rand_u32()`. Unit-processing order is shuffled each tick inside combat to prevent first-mover bias. No `thread_rng` or external RNG in simulation code. Tests in `src/ai/core/tests/determinism.rs` verify reproducibility byte-for-byte across runs at a given seed.
