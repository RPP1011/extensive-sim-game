# Room Generation

Rooms are procedurally generated spaces where combat takes place. The generation
system creates varied, tactically interesting layouts with cover, elevation,
and chokepoints.

## Module: `src/mission/room_gen/`

```
room_gen/
├── mod.rs           # Public API
├── floorplan.rs     # Layout algorithms
├── ml_gen.rs        # ML-assisted generation
├── nav.rs           # Navigation mesh generation
├── validation.rs    # Layout validation
├── cover.rs         # Cover placement
├── elevation.rs     # Height map generation
├── obstacles.rs     # Obstacle placement
├── spawns.rs        # Unit spawn point placement
└── theme.rs         # Visual theme selection
```

## Generation Pipeline

```
Parameters ──▶ Floorplan ──▶ Obstacles ──▶ Cover ──▶ Spawns ──▶ NavGrid
(size, theme)  (walls)       (pillars)     (half-    (hero/     (pathfinding
                                            walls)    enemy)     mesh)
```

## Floorplan

The floorplan defines the room's shape and walls. Several algorithms are
available:
- **Rectangular** — simple rooms with optional internal walls
- **L-shaped / T-shaped** — more interesting geometry
- **Voronoi** — organic, cave-like rooms
- **ML-generated** — diffusion-based generation for complex layouts

## Cover System

Cover provides damage reduction to nearby units. Cover objects have:
- **Height** — full cover (blocks LOS) or half cover (damage reduction only)
- **Position** — placed at tactically interesting positions
- **Destructibility** — some cover can be destroyed by AoE abilities

## Elevation

Rooms can have varying elevation:
- High ground provides +damage and +range bonuses
- Low ground has reduced vision
- Elevation differences affect LOS calculations

## ML-Assisted Generation

The `ml_gen.rs` module uses a diffusion transformer (trained in
`training/roomgen/`) to generate room layouts:
- Trained on handcrafted and curated room designs
- Uses flow matching for the generative process
- Can be conditioned on text descriptions of desired layouts

## Validation

Generated rooms are validated (`validation.rs`) to ensure:
- Both teams can reach each other (pathfinding exists)
- Spawn points are not inside walls
- Cover density is within acceptable bounds
- The room is playable (not too cramped or too open)
