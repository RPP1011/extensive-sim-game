# Terrain Integration — The Gap I Missed

> Companion to `2026-04-22-engine-expressiveness-gaps.md`. That doc
> split terrain across three sub-bullets (movement modes, line-of-sight,
> structures) without naming the real architectural mega-gap: **the
> combat sim has no terrain awareness at all**. This doc separates it
> out because it's the single biggest expressive constraint and because
> the integration surface is large enough to warrant its own design
> conversation.

## What "no terrain" actually means here

`crates/engine/src/*` contains zero references to `voxel_engine`,
zero height-field queries, zero walkability checks, zero line-of-sight
raycasts. Agent positions are `Vec3`s in an unbounded, featureless R³.
The mask, scoring, physics, and cast-dispatch pipelines all treat
position as just `(x, y, z)` — a metric, not a geometry.

Meanwhile the repo literally ships a voxel terrain engine
(`/home/ricky/Projects/voxel_engine`) with: SDF terrain generation,
chunk pool (1024-slot LRU), GPU terrain compute, fluid physics, voxel
place/harvest, a 3D camera. That whole subsystem is used by the
`viz` crate and the `world_sim` (root `src/world_sim/`, the zero-player
DF-style layer) but never by the combat sim.

The wolves+humans canonical fixture, the parity anchor, the four
example scenes, and the whole DSL — all happen in the void.

## Mechanics this blocks

Naming these separately from the generic expressiveness gaps doc
because each one collapses to "needs terrain" and the fix is one
integration, not fifteen.

1. **High ground bonuses.** Classic RTS: fire from elevation, damage
   bonus + sight bonus. Currently z-coordinate contributes to 3D
   distance but nothing else.
2. **Chokepoints.** A 2-tile-wide gap between cliffs that funnels
   attackers. Thermopylae. Impossible without walkable-tile data.
3. **Cover.** A rock occludes line-of-sight from an archer to a
   unit behind it. The archer's ranged attack mask needs a
   line-of-sight check.
4. **Flanking.** An agent's rear is exposed because the cliff behind
   them blocks retreat. No terrain → no "rear" is meaningful.
5. **Real flight advantage.** My `ecosystem.rs` and
   `starcraft_tribute.rs` spawn dragons at `z = 4..5` and pretend
   that's flight. Engine doesn't enforce "ground units can't cross
   this river" because there is no river. Flying is cosmetic.
6. **Different terrain-speed.** Mud, snow, road. The per-species /
   per-terrain speed modifier needs per-tile data.
7. **Destructible terrain.** Dragon's breath melts a wall, opening
   a path. Siege breaks through a gate. Without terrain there is
   no wall to melt.
8. **Placement phase.** Build a fortification before the enemy
   arrives. Needs persistent static geometry.
9. **Spatial memory.** "I know there's a cave to the east" — agent
   beliefs about geography. Needs a world shape to remember.
10. **Vision / scouting / fog of war.** Line-of-sight + visibility
    range → info asymmetry. Currently every agent sees every other
    agent (aggro_range is 50m Euclidean).
11. **Ambushes.** Hidden-in-thicket → `detect_threshold` based on
    terrain features. Currently hostility detection is pure distance.
12. **Environmental hazards.** Lava, chasms, falling trees. Today
    "terrain" can't kill anything.
13. **Navigation / pathfinding.** Currently MoveToward is a straight-
    line vector update. Real movement needs A*/flow-field over
    walkable tiles. `Delivery::Projectile` would also need this for
    arc computation.

Everything on that list in a canonical tactics game. Everything
missing today.

## The voxel_engine already has the primitives

From `/home/ricky/Projects/voxel_engine/src/`:

- `terrain_compute.rs` — SDF-based terrain field, chunk-keyed
- `voxel/` — voxel grid primitives, raycast, place/harvest
- `physics/` — fluid + rigid-body (for falling objects)
- `scene/` — camera + world representation
- `world/` — chunk management, streaming
- `ai/spatial.rs` — spatial hash (the combat engine has its own
  parallel impl — see `crates/engine/src/spatial.rs`)

So a lot of what combat would need is ALREADY implemented — just
on the other side of a crate boundary the combat sim doesn't cross.

## Integration surface

Two shapes for how the combat engine could query terrain.

### Option A — Direct dep: `engine` imports `voxel_engine`

```rust
// crates/engine/Cargo.toml
voxel_engine = { path = "../../voxel_engine", default-features = false }

// crates/engine/src/terrain.rs  (new)
pub use voxel_engine::world::{ChunkId, HeightField};
pub trait TerrainQuery {
    fn height_at(&self, x: f32, y: f32) -> f32;
    fn walkable(&self, pos: Vec3, mode: MovementMode) -> bool;
    fn line_of_sight(&self, from: Vec3, to: Vec3) -> bool;
    fn ray_cast(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<TerrainHit>;
}
```

Then `SimState` owns a `terrain: Arc<dyn TerrainQuery + Send + Sync>`
and mask/scoring/physics rules consult it.

Pros: straightforward, everything in one place.
Cons: pulls the voxel renderer's full dep graph (wgpu / vulkan /
gpu-allocator) into every build that uses `engine`, including headless
ML training. Voxel_engine's compile cost is non-trivial.

### Option B — Trait-object injection: `engine` defines the query shape, caller provides impl

```rust
// crates/engine/src/terrain.rs
pub trait TerrainQuery: Send + Sync {
    fn height_at(&self, x: f32, y: f32) -> f32;
    fn walkable(&self, pos: Vec3, mode: MovementMode) -> bool;
    fn line_of_sight(&self, from: Vec3, to: Vec3) -> bool;
}

// SimState holds Arc<dyn TerrainQuery>; a flat-plane default impl
// covers the headless case (every build where terrain doesn't matter
// or where training runs on synthetic terrain).
```

Then `crates/engine_voxel/` (new crate, optional) wraps voxel_engine
and produces a `TerrainQuery` impl. Wolves+humans tests keep using
a no-op flat-plane terrain. New examples (`terrain_combat.rs`,
`siege.rs`) can opt into voxel terrain.

Pros: engine stays headless, zero voxel_engine cost for default
builds, parity tests unaffected, terrain becomes optional.
Cons: one more indirection; trait object overhead on every height
query (can be mitigated by concrete-type generic param on the
per-tick closures that need it).

**Recommend B.** The engine's design philosophy throughout (backend
traits, evaluator traits, context traits, policy trait) is
"abstract the environment, let the caller inject." Terrain fits the
same pattern.

## Minimal viable first slice

Not "implement all of it." The smallest thing that unblocks
follow-on mechanics:

1. `TerrainQuery` trait + `FlatPlane` default impl in
   `crates/engine/src/terrain.rs`.
2. `SimState.terrain: Arc<dyn TerrainQuery>` defaulting to
   `Arc::new(FlatPlane::default())`.
3. One new mask primitive: `terrain.line_of_sight(self, target)`
   accessible from scoring / mask DSL.
4. One new scoring predicate using it: a height-bonus row on
   Attack that fires when `self.z > target.z + 2.0` AND LOS is
   clear.
5. Wolves+humans parity test unaffected (still uses flat plane).
6. New example `hill_fight.rs`: defenders on a height field,
   attackers below, narrated. Uses a hand-authored small terrain
   impl (a 4×4 m hill). Shows real elevation advantage.

That's maybe 300–400 LoC of actual engine work — trait, default impl,
one new mask primitive, one new scoring gate, one example. It ships
an extension point that every future terrain-using feature hangs off.

## What follow-on features cascade cleanly after this slice lands

- **Walkability:** `MovementMode` on agent, `TerrainQuery::walkable`
  filters MoveToward candidates.
- **Cover:** LOS already in; damage modifier per blocked segment.
- **Chokepoints:** mask considers only walkable positions.
- **Destructible:** a new EffectOp::TerrainDamage mutates terrain
  via a callback on the TerrainQuery.
- **Placement:** structure-building abilities that call a
  `TerrainBuilder` sibling trait.

Each is <500 LoC once the terrain seam exists.

## Connection to the earlier gaps doc

Items #4 (movement modes), #8 (LOS/cover), #12 (structures),
and parts of #10 (unit spawning into contested terrain) all
collapse to "terrain + specific extension."

I should have named this in the earlier doc as gap #0 — the
architectural prerequisite that unlocks the others.

## Why this wasn't done yet

Guessing at the design history from reading around: the combat
engine was developed heads-down on the wolves+humans DSL harness
(parity anchor, interpreter path, Context traits) so the MVP's
scope stayed focused. Terrain integration is a cross-cutting
architectural change that deserves its own plan, its own spec, and
is genuinely the next big architectural frontier for the combat
layer.
