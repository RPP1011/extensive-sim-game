# World Sim Optimization Targets — 2026-04-18

Flamegraph-driven analysis. `scripts/perf_bench.sh <ticks> <world>` produces a
`cargo flamegraph` SVG at full sim speed (no instrumentation overhead). The
line-level hotspot view is what tells us where to put SIMD — or more often,
where a non-SIMD fix is the bigger win.

## Methodology

```bash
sudo sysctl kernel.perf_event_paranoid=1   # one-time, reverts on reboot
./scripts/perf_bench.sh 500 small          # ~1.5 min wall, produces SVG in generated/flamegraphs/
python3 -m http.server --directory generated/flamegraphs 8765 &  # to view in Chrome
```

Inspect in a browser — hover for exact %s, click to zoom into sub-trees.

## Findings (2026-04-18, small world, 500 ticks)

| % of total | Frame |
|---|---|
| **72%** | `scan_voxel_resources` (NPC resource discovery) |
| 72% | ↳ `surface_height` |
| 70% | ↳ `get_voxel` |
| **67%** | ↳↳ `HashMap<ChunkPos, Chunk>::get` |
| **54%** | ↳↳↳ **SipHash `finish` / `Sip13Rounds` on ChunkPos** |
| 37% | ↳↳↳↳ `d_rounds` (SipHash mixing round) |
| 18% | 2nd-level `scan_voxel_resources` (recursive / re-entry) |

### What this tells us

**The dominant cost is hashing, not simulation math.** Over half of total
program time is `std::collections::HashMap<ChunkPos, Chunk>` using the default
`RandomState` → `SipHash-1-3` to look up chunks by coordinate. `ChunkPos` is a
3-tuple of `i32` — SipHash is cryptographic overkill for that key, and the same
chunk gets looked up many times in a row during a voxel scan.

### Recommended optimizations (ordered by expected win)

1. **Replace `HashMap<ChunkPos, Chunk>` with a non-cryptographic hasher or
   flat store.** Immediate candidates:
   - `ahash::AHashMap<ChunkPos, Chunk>` — drop-in, expected 4-10× on hash
   - `rustc_hash::FxHashMap` — even faster for small integer keys
   - If chunk coords are bounded (e.g., finite world), use a 3D `Vec<Vec<Vec<Chunk>>>` or `Vec<Option<Chunk>>` indexed by `x + y*w + z*w*h`
   - Expected wall-time impact: 40-60% reduction
2. **Cache the last-looked-up chunk in `get_voxel`.** A single-element LRU
   (`{last_chunk_pos, last_chunk_ref}`) would catch the repeated-access pattern
   the flamegraph implies.
3. **Inline `surface_height`.** Its 72% self-attribution suggests it's not
   being inlined across the `HashMap` boundary — mark `#[inline]` or
   restructure to take the chunk as an arg rather than looking it up each
   call.

### What about SIMD?

At small-world scale, no f32-array hot loops are visible — the HashMap
bottleneck dominates everything else. Real sim-side SIMD candidates
(movement, economy, HP updates) should surface in the flamegraph once the
hash overhead is eliminated. Then rerun at 10K+ entity scale to see them.

**Anti-candidates confirmed disqualified:**
- `delta::merge_deltas` (HashMap-heavy, already flagged in spec)
- `scan_voxel_resources` itself — pointer-heavy voxel traversal

## Why flamegraph > in-process per-system timing

Original plan built per-system nanosecond counters via `profile-systems`
feature. During this session we discovered that approach has ~100× overhead
at default-world scale (HashMap insertion in `thread_record` + 580 per-settlement
timer pairs per tick). Flamegraph delivered the same targeting information in
~1.5 minutes of profiling at full speed, plus *line-level* hotspots the
per-system counters can never see.

**Going forward:** prefer flamegraph for point-in-time target selection; use
in-process timing only if CI regression tracking becomes a need (and if so,
swap the HashMap for a `Vec<(interned_idx, ns, calls)>` to get overhead back
to ~1%).
