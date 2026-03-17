# Map & Room Generation

The project includes two levels of procedural generation: overworld maps and
combat room layouts.

## Overworld Map Generation

**Module:** `src/mapgen_voronoi/`

Overworld maps are generated using Voronoi diagrams:

1. **Seed points** — randomly placed points on the map area
2. **Voronoi tessellation** — generates region boundaries
3. **Graph extraction** — adjacent Voronoi cells become connected regions
4. **Faction placement** — factions are assigned to clusters of regions
5. **Terrain assignment** — terrain types based on position and noise

## Combat Room Generation

**Module:** `src/mission/room_gen/`

Room layouts for combat encounters:

1. **Floorplan** — rectangular, L-shaped, or Voronoi-based room shapes
2. **Obstacles** — pillars, walls, and blocking terrain
3. **Cover** — half-walls and objects that provide damage reduction
4. **Elevation** — height variation for tactical advantage
5. **Spawn points** — hero and enemy starting positions
6. **Navigation mesh** — pathfinding grid derived from the layout

## ML-Assisted Room Generation

**Module:** `training/roomgen/`

A diffusion transformer generates room layouts:

- **Architecture:** DiT (Diffusion Transformer) with text conditioning
- **Training:** Flow matching on curated room designs
- **Inference:** Produces room layouts from text descriptions

```bash
# Generate rooms with ML
uv run --with numpy --with torch python training/roomgen/sample.py \
    --prompt "large arena with central pillars" \
    --count 10 \
    --output generated/rooms/
```

## Preview Tool

```bash
# Preview a generated room
cargo run --bin room_preview -- --room generated/rooms/room_001.json
```
