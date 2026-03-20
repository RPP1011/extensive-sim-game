# Overworld ASCII Rendering Pipeline

**Companion to: Project Chimera ASCII UI Design Document**
**March 2026**

---

## Overview

The overworld is a 150x80+ character ASCII landscape rendered in egui, with procedurally generated terrain, faction territories, settlements, and 30+ roaming parties. The rendering pipeline transforms Voronoi region polygons into a painted, interactive map in five layers.

This document covers three phases: generation (run once), per-frame rendering (hot path), and interactive systems (travel, entities, camera).

---

## Phase 1: Generation Pipeline

Run once at campaign start. Transforms `mapgen_voronoi` output into an `OverworldGrid` stored in `CampaignState`.

### Step 1: Rasterize Voronoi Polygons to Cell Grid

`mapgen_voronoi` outputs region polygons + an adjacency graph. The renderer needs a per-cell `region_id`. Since Voronoi regions are defined by nearest-seed-point, rasterization is a nearest-seed assignment: for each cell `(x, y)`, find the Voronoi seed with minimum distance.

For 150x80 with ~20 regions, this is ~240,000 distance comparisons — trivial. No complex point-in-polygon testing needed.

```rust
struct MapCell {
    region_id: u16,
    faction_id: u8,
    terrain: TerrainType,
    height: f32,
    moisture: f32,
    glyph: char,
    fg_color: Color32,
    is_border: bool,
    border_glyph: char,
}

struct OverworldGrid {
    width: u16,
    height: u16,
    cells: Vec<MapCell>,        // row-major, len = w * h
    settlements: Vec<Settlement>,
    roads: Vec<RoadSegment>,
    faction_version: u64,       // incremented on territory changes
}
```

### Step 2: Per-Cell Terrain Noise

Two noise passes over the grid: height (low-frequency simplex, 2-3 octaves) and moisture (different seed, similar frequency). The combination of region base terrain + height + moisture determines the final `TerrainType`:

```rust
enum TerrainType {
    DeepWater, ShallowWater, Coast,
    Plains, Grassland, Marsh,
    Forest, DenseForest,
    Hills, Foothills,
    Mountain, Peak,
    Road, Settlement,
}
```

A mountain region with low height noise becomes foothills. A plains region with high moisture becomes marshland. The region provides the base tendency; noise provides local variation.

### Step 3: Post-Processing Passes

Raw noise-to-terrain mapping produces visual static. Four post-processing passes create coherent landscape features:

**Pass 1 — Mountain ridge tracing.** Find cells with height > 0.7, then walk east/west along the gradient to trace connected ridgelines. Mark ridge cells as `Peak`, adjacent cells below as `Mountain`, with foothills tapering by distance. This turns scattered `^` into `/\ /\` mountain range silhouettes.

**Pass 2 — Forest clumping via cellular automata.** Run 2-3 iterations: a forest cell survives if 4+ of 8 neighbors are forest; a non-forest cell becomes forest if 5+ neighbors are forest. This grows coherent groves and eliminates isolated single-tree cells. A second pass marks cells surrounded by 7+ forest neighbors as `DenseForest`.

**Pass 3 — River carving via downhill flow.** Pick high-moisture, mid-elevation cells as sources. From each source, walk steepest-descent until reaching water or the map edge, marking cells as `ShallowWater`. Rivers merge naturally when paths converge. Source count is a generation parameter (~3-5 rivers per map).

**Pass 4 — Road pathfinding via A*.** Compute a minimum spanning tree of settlements to avoid redundant roads. For each MST edge, run A* over the terrain grid with terrain-dependent costs (plains cheap, mountains expensive, water impassable). Mark path cells as `Road`. Roads naturally follow valleys and plains.

### Step 4: Border Computation

A cell is a border cell if any cardinal neighbor has a different `faction_id`. A 4-bit adjacency mask (N=1, E=2, S=4, W=8) maps to box-drawing characters:

| Mask Pattern | Glyph | Meaning |
|-------------|-------|---------|
| N or S or N+S | `─` | Horizontal border |
| E or W or E+W | `│` | Vertical border |
| N+E | `╭` | Corner |
| E+S | `╮` | Corner |
| S+W | `╰` | Corner |
| W+N | `╯` | Corner |
| 3+ edges | `┼` | Intersection |

Border glyphs override terrain glyphs at those cells. Rendered in a single neutral color at ~50-60% opacity.

### Step 5: Glyph + Color Assignment

Pure lookup table. Each `TerrainType` maps to a glyph and a muted foreground color:

| Terrain | Glyph | Color | Hex |
|---------|-------|-------|-----|
| Plains | `.` | Dim green-gray | `#8A9A7A` |
| Grassland | `.` | Slightly greener | `#789664` |
| Forest | `♣` | Dark green | `#5A7A4A` |
| DenseForest | `♣` | Darker green | `#3C5F32` |
| Hills | `~` | Tan | `#9A8A6A` |
| Mountain | `^` | Brown | `#8A7A5A` |
| Peak | `▲` | Light stone | `#AAA096` |
| ShallowWater | `≈` | Steel blue | `#7A9ABB` |
| DeepWater | `≈` | Deeper blue | `#5078AA` |
| Coast | `,` | Sandy | `#B4AA8C` |
| Marsh | `~` | Olive | `#648C6E` |
| Road | `═` | Light gray | `#A09B91` |

All terrain colors are deliberately muted and desaturated so that bright entity glyphs and faction tints pop against them.

---

## Phase 2: Per-Frame Render Pipeline

The hot path. Runs every frame, paints the visible portion of the map.

### Layer Order

| Layer | Content | Count | Method |
|-------|---------|-------|--------|
| L0 | Faction background tint | ~12,000 rects | `rect_filled()`, faction color @ 10-15% alpha |
| L1 | Terrain glyphs | ~12,000 quads | Textured quads from glyph atlas |
| L2 | Faction borders | ~500-1,500 quads | Same atlas, neutral color |
| L3 | Settlements + parties | ~20-50 quads | Same atlas, bright faction colors |
| L4 | UI overlays | Variable | egui native (tooltips, path preview, selection) |

### Glyph Atlas (Critical Optimization)

Calling `painter.text()` 12,000 times per frame is the performance cliff. Instead:

1. At startup, render every unique glyph character into a texture atlas (a grid of small glyph bitmaps).
2. Each frame, emit one textured quad per visible cell into a single `egui::Mesh`.
3. Color is applied per-vertex, so a single white glyph bitmap gets tinted to any color.
4. The entire map (L0 + L1 + L2 + L3) is one `painter.add(Shape::Mesh(mesh))` call.

This collapses ~25,000 draw primitives into a single draw call.

### Terrain Mesh Caching

The terrain grid is static — it only changes when faction territories shift (campaign turns, not every frame). Cache the L0+L1+L2 mesh and only rebuild when:

- Camera position or zoom changes (panning/scrolling)
- Faction territory state changes (`faction_version` increments)

Entity quads (L3) are rebuilt every frame but that's only ~50 quads — trivial. Most frames skip the expensive terrain rebuild entirely.

### Viewport Culling

Only cells within the camera's visible rect are rendered. At 150x80 total with a typical viewport showing ~80x40 cells, you're rendering ~3,200 cells instead of 12,000. The visible rect is computed from `camera.pos + viewport_pixel_size / cell_pixel_size`.

---

## Phase 3: Camera System

### State

```rust
struct Camera {
    pos: Vec2,          // world position (top-left, in cell coords)
    target_pos: Vec2,   // for smooth interpolation
    zoom: f32,          // 1.0 = default
    target_zoom: f32,
}
```

### Controls

- **Pan:** WASD / arrow keys, edge-of-screen mouse hover, middle-mouse drag.
- **Zoom:** Scroll wheel adjusts `target_zoom` (clamped to 0.5 – 2.0 range).
- **Follow:** During travel, camera smoothly tracks player position with `camera.follow(player_cell, viewport_cells)`.

### Smooth Interpolation

Each frame: `pos = lerp(pos, target_pos, 6.0 * dt)`. This creates smooth panning with no jarring jumps. The lerp factor of 6.0 means the camera reaches ~95% of target position within ~0.5 seconds.

---

## Phase 4: Entity Systems

### Party Travel Animation

When a destination is selected, A* computes a path over the terrain grid using the same cost function as road carving. The party marker interpolates between waypoints:

- `TravelState` tracks: path waypoints, current segment index, progress within segment (0.0 – 1.0), effective speed.
- Speed varies by terrain: roads are 1.5x, plains 1.0x, forest 0.6x, mountains 0.25x.
- Visual position is a float lerp between current and next waypoint, quantized to nearest cell for glyph placement.
- Camera follows the player during travel.

### Travel Path Preview

Before confirming travel, the planned A* path renders as dotted `·` characters in a distinct bright color. The player sees exactly which route they'll take, including terrain-dependent speed (the dots could be spaced wider in slow terrain to hint at travel time).

### Entity Collision and Clustering

With 30+ roaming parties, multiple entities will share cells. Each frame:

1. Group entities by cell position into clusters.
2. Single-entity clusters render normally (one glyph, faction color).
3. Multi-entity same-faction clusters render a count digit (`2`, `3`, etc.) in the faction color.
4. Multi-entity mixed-faction clusters render `*` in a neutral color.
5. Hovering a cluster shows a popup listing all parties: name, faction, size.

### Entity Glyph Vocabulary

| Glyph | Entity Type | Color |
|-------|------------|-------|
| `@` | Player party | Always bright green, unique |
| `◆` | Roaming party | Faction color |
| `⌂` | Town | Faction color |
| `■` | Castle | Faction color |
| `▲` | Camp | Faction color |
| `†` | Ruin | Neutral / dim |

---

## Performance Budget

### Target: 60 FPS at 150x80 map, ~80x40 visible viewport

| Operation | Per-Frame Cost | Strategy |
|-----------|---------------|----------|
| Terrain mesh (L0+L1+L2) | ~3,200 quads | Cached, rebuild only on pan/zoom/faction change |
| Entity mesh (L3) | ~50 quads | Rebuilt every frame (trivial) |
| Total draw calls | 1-2 | Single batched mesh + egui overlay |
| Camera update | Trivial | Two lerps |
| Entity clustering | O(N) where N ≈ 50 | HashMap grouping |
| Travel interpolation | O(P) where P = party count | One lerp per traveling party |

### Potential Bottlenecks

- **Glyph atlas creation** (startup only): rendering ~30 unique characters to a texture. One-time cost, ~10ms.
- **Mesh rebuild on pan:** rebuilding 3,200 textured quads. Should be <1ms.
- **Font shaping bypass:** the glyph atlas entirely skips egui's text layout pipeline, which is the main performance win.

---

## Open Questions

- **Zoom levels:** At extreme zoom-out (showing the full 150x80 map in a small viewport), glyphs become unreadable. Should there be a LOD threshold where terrain simplifies to colored blocks?
- **Fog of war:** Should unexplored regions render as `?` or blank space, or show terrain with entities hidden?
- **Seasonal/time variation:** Could terrain colors shift slightly based on campaign time (greener in spring, browner in autumn)?
- **Minimap:** Would a small corner minimap showing the full map at block-color resolution be useful for orientation?

---

*Technical companion to the ASCII UI Design Document. Covers the generation pipeline (Voronoi rasterization, terrain noise, post-processing), per-frame rendering (glyph atlas batching, layer ordering, viewport culling), camera system, and entity systems (travel animation, collision clustering).*
