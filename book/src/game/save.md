# Save System

The save system serializes the full campaign state to JSON for persistence.

## Module: `src/game_core/save.rs`

## Save Format

Campaign state is serialized via serde to JSON:

```rust
pub fn save_campaign(state: &CampaignState, path: &Path) -> Result<()>
pub fn load_campaign(path: &Path) -> Result<CampaignState>
```

The save file contains:
- Full `CampaignState` (turn, party, regions, factions, diplomacy)
- Hero roster with persistent stats
- Narrative progression state
- Active flashpoints

## Version Migration

As the game evolves, save formats change. The migration system
(`src/game_core/migrate.rs`) handles upgrading old saves:

```rust
pub fn migrate(save_data: Value) -> Result<CampaignState>
```

Each migration is a function that transforms a JSON `Value` from version N to
version N+1. Migrations are chained to handle arbitrarily old saves.

## Verification

`src/game_core/verify.rs` validates save integrity after loading:
- All referenced unit IDs exist
- Region connections are bidirectional
- Faction relationships are consistent
- No impossible states (negative HP, cooldowns below 0, etc.)
