//! Data-driven world templates for campaign initialization.
//!
//! Instead of always starting with the same 3 regions / 2 factions / 4 locations,
//! campaigns can load a [`WorldTemplate`] from TOML files in `assets/world_templates/`.

use serde::{Deserialize, Serialize};

use super::state::{
    DiplomacyMatrix, DiplomaticStance, FactionState, Location, LocationType, NpcRelationship,
    NpcType, Region,
};

/// A complete world configuration that can be loaded from a TOML file.
///
/// Defines the starting regions, locations, factions, diplomacy, NPC relationships,
/// and global threat level for a campaign.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldTemplate {
    pub name: String,
    pub regions: Vec<Region>,
    pub locations: Vec<Location>,
    pub factions: Vec<FactionState>,
    pub diplomacy: DiplomacyMatrix,
    pub npc_relationships: Vec<NpcRelationship>,
    pub global_threat_level: f32,
}

impl WorldTemplate {
    /// Load a single world template from a TOML file.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    /// Load all world templates from a directory of TOML files.
    ///
    /// Returns them sorted by filename for deterministic ordering.
    pub fn load_from_dir(dir: &std::path::Path) -> Result<Vec<Self>, String> {
        let mut templates = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read {}: {}", dir.display(), e))?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                let template = Self::load_from_file(&path)?;
                templates.push(template);
            }
        }
        Ok(templates)
    }

    /// Return the built-in default world template (the "frontier" configuration).
    ///
    /// This matches the original inline defaults: 3 regions, 2 factions, 4 locations.
    pub fn default_frontier() -> Self {
        WorldTemplate {
            name: "Frontier".into(),
            regions: vec![
                Region {
                    id: 0,
                    name: "Greenhollow".into(),
                    owner_faction_id: 0,
                    neighbors: vec![1, 2],
                    unrest: 10.0,
                    control: 80.0,
                    threat_level: 15.0,
                },
                Region {
                    id: 1,
                    name: "Ironridge".into(),
                    owner_faction_id: 1,
                    neighbors: vec![0, 2],
                    unrest: 30.0,
                    control: 60.0,
                    threat_level: 40.0,
                },
                Region {
                    id: 2,
                    name: "Mistfen".into(),
                    owner_faction_id: 0,
                    neighbors: vec![0, 1],
                    unrest: 20.0,
                    control: 70.0,
                    threat_level: 25.0,
                },
            ],
            locations: vec![
                Location {
                    id: 0,
                    name: "Thornwall Keep".into(),
                    position: (10.0, 5.0),
                    location_type: LocationType::Settlement,
                    threat_level: 10.0,
                    resource_availability: 70.0,
                    faction_owner: Some(0),
                    scouted: true,
                },
                Location {
                    id: 1,
                    name: "The Sunken Crypt".into(),
                    position: (-15.0, 20.0),
                    location_type: LocationType::Dungeon,
                    threat_level: 55.0,
                    resource_availability: 30.0,
                    faction_owner: None,
                    scouted: false,
                },
                Location {
                    id: 2,
                    name: "Trader's Rest".into(),
                    position: (5.0, -10.0),
                    location_type: LocationType::Settlement,
                    threat_level: 5.0,
                    resource_availability: 85.0,
                    faction_owner: Some(0),
                    scouted: true,
                },
                Location {
                    id: 3,
                    name: "Wolfcrag Ruins".into(),
                    position: (-20.0, -15.0),
                    location_type: LocationType::Ruin,
                    threat_level: 45.0,
                    resource_availability: 40.0,
                    faction_owner: None,
                    scouted: false,
                },
            ],
            factions: vec![
                FactionState {
                    id: 0,
                    name: "The Accord".into(),
                    relationship_to_guild: 40.0,
                    military_strength: 50.0,
                    territory_size: 2,
                    diplomatic_stance: DiplomaticStance::Friendly,
                    recent_actions: Vec::new(),
                },
                FactionState {
                    id: 1,
                    name: "Iron Dominion".into(),
                    relationship_to_guild: -50.0,
                    military_strength: 120.0,
                    territory_size: 1,
                    diplomatic_stance: DiplomaticStance::AtWar,
                    recent_actions: Vec::new(),
                },
            ],
            diplomacy: DiplomacyMatrix {
                relations: vec![vec![0, -20], vec![-20, 0]],
                guild_faction_id: 0,
            },
            npc_relationships: vec![
                NpcRelationship {
                    npc_id: 100,
                    npc_name: "Old Gareth".into(),
                    npc_type: NpcType::Mercenary,
                    relationship_score: 30.0,
                    last_interaction_ms: 0,
                    rescue_available: false,
                    rescue_cost: 60.0,
                },
                NpcRelationship {
                    npc_id: 101,
                    npc_name: "Lady Voss".into(),
                    npc_type: NpcType::FactionLeader,
                    relationship_score: 55.0,
                    last_interaction_ms: 0,
                    rescue_available: true,
                    rescue_cost: 0.0,
                },
            ],
            global_threat_level: 20.0,
        }
    }
}
