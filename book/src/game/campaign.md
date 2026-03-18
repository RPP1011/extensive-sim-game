# Campaign & Overworld

The campaign layer is the strategic outer loop of the game. Players manage a
party of heroes, navigate an overworld map, and make decisions that affect
faction relationships and narrative progression.

## Module: `src/game_core/`

```
game_core/
├── types.rs                  # CampaignState, HeroState, Party
├── overworld_types.rs        # OverworldRegion, map data
├── roster_types.rs           # Hero roster, recruitment
├── setup.rs                  # Campaign initialization
├── generation.rs             # Procedural overworld generation
├── roster_gen.rs             # Hero roster generation
├── campaign_systems.rs       # Turn processing
├── overworld_systems.rs      # Region updates
├── overworld_nav.rs          # Map navigation
├── flashpoint_spawn.rs       # Crisis event generation
├── flashpoint_progression.rs # Crisis resolution
├── diplomacy_systems.rs      # Faction relationships
├── consequence_systems.rs    # Action consequences
├── campaign_outcome.rs       # Win/loss conditions
├── mission_systems.rs        # Mission dispatch
├── attention_systems.rs      # Narrative attention
├── companion.rs              # Companion story arcs
├── save.rs                   # Serialization
├── migrate.rs                # Save version migration
└── verify.rs                 # State verification
```

## CampaignState

The top-level campaign state tracks everything about an ongoing campaign:

```rust
pub struct CampaignState {
    pub turn: u32,
    pub party: Party,
    pub regions: Vec<OverworldRegion>,
    pub factions: Vec<Faction>,
    pub diplomacy: DiplomacyState,
    pub flashpoints: Vec<Flashpoint>,
    pub narrative: NarrativeState,
    // ...
}
```

## Turn Processing

Each campaign turn:

1. **Player action** — move to a region, start a mission, rest, trade
2. **Faction AI** — NPC factions take their turns (expand, attack, negotiate)
3. **Flashpoint check** — roll for new crisis events
4. **Diplomacy update** — relationships shift based on actions
5. **Consequence resolution** — pending consequences from previous turns resolve
6. **Narrative progression** — story beats advance

## Overworld Map

The overworld is a graph of `OverworldRegion` nodes:

```rust
pub struct OverworldRegion {
    pub name: String,
    pub faction: Option<FactionId>,
    pub connections: Vec<RegionId>,
    pub terrain: TerrainType,
    pub resources: Resources,
    pub threat_level: f32,
    // ...
}
```

Regions are connected in a graph (not a grid), allowing natural geographic
layouts. Players move between connected regions.

## Procedural Generation

Campaign maps are procedurally generated using Voronoi diagrams
(`src/mapgen_voronoi/`) for region boundaries, combined with graph algorithms
for connectivity and faction placement.
