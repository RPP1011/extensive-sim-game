//! BuildingEnv configuration: action encoding, curriculum levels.

use crate::world_sim::state::BuildingType;

/// Building types the agent can place.
pub const PLACEABLE_TYPES: &[BuildingType] = &[
    BuildingType::House,
    BuildingType::Longhouse,
    BuildingType::Manor,
    BuildingType::Farm,
    BuildingType::Mine,
    BuildingType::Sawmill,
    BuildingType::Forge,
    BuildingType::Workshop,
    BuildingType::Apothecary,
    BuildingType::Market,
    BuildingType::Warehouse,
    BuildingType::Inn,
    BuildingType::TradePost,
    BuildingType::GuildHall,
    BuildingType::Temple,
    BuildingType::Barracks,
    BuildingType::Watchtower,
    BuildingType::Library,
    BuildingType::CourtHouse,
    BuildingType::Well,
    BuildingType::Tent,
    BuildingType::Camp,
    BuildingType::Shrine,
];

pub const NUM_PLACEABLE_TYPES: usize = 23; // PLACEABLE_TYPES.len()

/// Voxel grid observation region: 128x128 centered on settlement.
pub const GRID_SIZE: usize = 128;
pub const GRID_CELLS: usize = GRID_SIZE * GRID_SIZE;

/// Total action space: pass (0) + grid_cells * building_types.
pub const NUM_ACTIONS: usize = 1 + GRID_CELLS * NUM_PLACEABLE_TYPES;

/// Decode a raw action index into pass or (voxel offset, building_type).
pub fn decode_action(action: usize) -> ActionChoice {
    if action == 0 {
        ActionChoice::Pass
    } else {
        let idx = action - 1;
        let cell = idx / NUM_PLACEABLE_TYPES;
        let btype = idx % NUM_PLACEABLE_TYPES;
        let col = (cell % GRID_SIZE) as i32;
        let row = (cell / GRID_SIZE) as i32;
        ActionChoice::Place {
            grid_offset: (col, row),
            building_type: PLACEABLE_TYPES[btype],
        }
    }
}

/// Encode a (grid offset, building_type) into a raw action index.
pub fn encode_action(col: i32, row: i32, building_type: BuildingType) -> usize {
    let btype_idx = PLACEABLE_TYPES.iter().position(|&t| t == building_type);
    match btype_idx {
        Some(bi) => 1 + (row as usize * GRID_SIZE + col as usize) * NUM_PLACEABLE_TYPES + bi,
        None => 0, // non-placeable type → pass
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActionChoice {
    Pass,
    Place {
        /// Offset from settlement center in voxel coords.
        grid_offset: (i32, i32),
        building_type: BuildingType,
    },
}

/// Curriculum level controls episode parameters.
#[derive(Debug, Clone)]
pub struct CurriculumLevel {
    pub level: u8,
    pub tick_budget: u64,
    pub heartbeat_interval: u64,
    pub max_actions: usize,
    pub min_severity: f32,
    pub max_severity: f32,
    pub max_challenges: usize,
}

impl CurriculumLevel {
    pub fn level_1() -> Self {
        Self { level: 1, tick_budget: 2000, heartbeat_interval: 200, max_actions: 10, min_severity: 0.3, max_severity: 0.5, max_challenges: 1 }
    }
    pub fn level_2() -> Self {
        Self { level: 2, tick_budget: 5000, heartbeat_interval: 200, max_actions: 20, min_severity: 0.5, max_severity: 0.8, max_challenges: 1 }
    }
    pub fn level_3() -> Self {
        Self { level: 3, tick_budget: 8000, heartbeat_interval: 200, max_actions: 25, min_severity: 0.3, max_severity: 0.9, max_challenges: 3 }
    }
    pub fn level_4() -> Self {
        Self { level: 4, tick_budget: 15000, heartbeat_interval: 200, max_actions: usize::MAX, min_severity: 0.1, max_severity: 1.0, max_challenges: 5 }
    }
}

/// Full env configuration.
#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub curriculum: CurriculumLevel,
    pub seed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_action_is_zero() {
        assert_eq!(decode_action(0), ActionChoice::Pass);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let col = 10i32;
        let row = 20i32;
        let btype = BuildingType::Barracks;
        let encoded = encode_action(col, row, btype);
        assert!(encoded > 0);
        let decoded = decode_action(encoded);
        assert_eq!(decoded, ActionChoice::Place { grid_offset: (col, row), building_type: btype });
    }

    #[test]
    fn all_placeable_types_roundtrip() {
        for (i, &btype) in PLACEABLE_TYPES.iter().enumerate() {
            let encoded = encode_action(0, 0, btype);
            assert_eq!(encoded, 1 + i);
            let decoded = decode_action(encoded);
            assert_eq!(decoded, ActionChoice::Place { grid_offset: (0, 0), building_type: btype });
        }
    }

    #[test]
    fn non_placeable_type_encodes_as_pass() {
        let encoded = encode_action(5, 5, BuildingType::Treasury);
        assert_eq!(encoded, 0);
    }

    #[test]
    fn max_action_index_valid() {
        let decoded = decode_action(NUM_ACTIONS - 1);
        match decoded {
            ActionChoice::Place { grid_offset: (col, row), building_type } => {
                assert!(col < GRID_SIZE as i32);
                assert!(row < GRID_SIZE as i32);
                assert_eq!(building_type, *PLACEABLE_TYPES.last().unwrap());
            }
            _ => panic!("expected Place"),
        }
    }
}
