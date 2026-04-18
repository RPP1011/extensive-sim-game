# World Sim Optimization Log

Methodology: flamegraph at each step, identify top cost, surgical fix, rerun.

## Results

| Step | Small-world tps | Default tps | Commit |
|---|---|---|---|
| baseline | 11.5 | 0.07 | — |
| ahash chunks + analytical surface | 129 (11×) | 9 | 0b5a688e |
| ahash voxel sets in structural_tick | 193 (17×) | — | ab973959 |
| pre-size VoxelSets | 223 (19×) | — | 895e5feb |
| flat bool arrays in BFS | 549 (48×) | — | acecbde1 |
| chunk-batch z-scan in scan_voxel_resources | 700 (61×) | — | aa0a3309 |
| buffer reuse in structural_tick | 698 | — | 333f6d0e |
| clear chunk.dirty | 869 (76×) | — | c47fef96 |
| memoize surface_height across NPCs | 906 (79×) | — | 2d4ef62c |
| pre-size SurfaceCache | 941 (82×) | — | 7d134c90 |
| persist surface_cache on WorldState | 4874 (424×) | 13 (186×) | 3f33951f |
| precompute disk offsets | 5015 | — | 6fa9e53c |
| u64 packed key for surface_cache | 5015 | — | 28a2bc6f |
| chunk-major scan | 6321 (550×) | 22 (314×) | ea384ddd |
| hoist cell_x | 6685 | 23 | ed634be9 |
| div_euclid → shift | **10539 (917×)** | **29 (414×)** | 6f2029c5 |
| 2-slot z-chunk cache | 10155 | 30 | 613b4630 |
| counts ahash | 10286 | 31 | 75ea535a |
| construction ahash | 10323 (898×) | 30 | 9517184a |

**Cumulative: small-world ~898×, default world ~430×.**

## Failed experiments

Documented so future optimizers don't repeat:

1. **Identity hash on packed u64 keys** — catastrophic bucket collisions when
   vx/vy near origin (high bits zero). Small-world 5000→25 tps. Reverted.
2. **Flat Vec prefetch of surface heights** — same total hashmap lookups,
   just front-loaded. Cache locality benefit was outweighed by the
   overhead of materializing the Vec. Small-world regressed -10%. Reverted.
3. **Rayon parallelization of scan_all_npc_resources** — RwLock per-position
   overhead + rayon thread-pool overhead at small N. Both worlds regressed
   ~8×. Reverted.

## Remaining architectural wins (not surgical)

At current state, default world is ~50% `scan_all_npc_resources` inside
`WorldSim::tick` (~54% of total). The 12% `surface_cache.get` is the
dominant remaining HashMap cost. Fixing it requires:

1. **Tile-cached surface grid** — `HashMap<tile_id, Box<[i32; 128*128]>>`.
   Reduces hash lookups from 20K/NPC to ~4/NPC (one per tile). Medium
   refactor, ~500 LoC.
2. **Parallel scan with per-thread caches** — rayon thread-local
   SurfaceCaches, merge at end. Would eliminate lock contention of the
   naive RwLock attempt. Complex but potentially 8-32× on a multicore.
3. **Precomputed global surface grid** — if region_plan bounds known at
   startup, materialize a flat `Vec<i32>` for the whole world at init.
   ~4-16 MB RAM cost. O(1) access with zero hashing.

Each of these is its own ticket, not a quick fix. The surgical-fix
regime has been fully exhausted.

## Also noted from flamegraph

Now that `scan_all_npc_resources` is down to 50% of tick, actual AI systems
finally appear:

- `evaluate_and_act`: 2.29%
- `flood_fill_floor`: 2.12% (ahash'd — now 1.36%)
- `score_npc_actions`: 2.1%
- `fbm_2d` inside simulation paths: 1.6-1.8%

These become the next natural targets after the surface-cache
architectural work lands.
