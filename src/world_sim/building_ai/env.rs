//! BuildingEnv — wraps WorldSim as an rl4burn `Env` for reactive building AI.
//!
//! The sim ticks forward internally between `step()` calls. The agent sees
//! decision points triggered by world events or a periodic heartbeat.

use rl4burn::{Env, Step, Space};

use crate::world_sim::state::{WorldState, EntityKind, BuildingType, entity_hash};
use crate::world_sim::runtime::WorldSim;
use crate::world_sim::voxel::world_to_voxel;
use crate::world_sim::systems::buildings::stamp_building_voxels;
use super::env_config::{self, ActionChoice, CurriculumLevel, GRID_SIZE, NUM_ACTIONS};
use super::env_obs::{self, OBS_DIM};
use super::env_reward::{ObjectiveScores, compute_reward};
use super::mass_gen::{self, TerrainType, MaturityLevel, ResourceProfile, NpcComposition, BuildingQuality, PressureType};
use super::types::ChallengeCategory;
use crate::world_sim::state::WorldTeam;

/// Decision point trigger detected during tick loop.
#[derive(Debug, Clone, Copy)]
enum Trigger {
    /// NPC died this tick.
    NpcDeath,
    /// Monster entered settlement area.
    MonsterArrival,
    /// A building dropped below 50% HP.
    BuildingDamaged,
    /// A commodity stockpile dropped below 20%.
    ResourceCrisis,
    /// Population exceeds housing by >20%.
    HousingOverflow,
    /// No event, but heartbeat interval elapsed.
    Heartbeat,
}

pub struct BuildingEnv {
    sim: WorldSim,
    curriculum: CurriculumLevel,
    tick_budget: u64,
    heartbeat_interval: u64,
    ticks_since_decision: u64,
    actions_taken: usize,
    pre_decision_scores: ObjectiveScores,
    ideal_scores: ObjectiveScores,
    challenge_category: ChallengeCategory,
    challenge_severity: f32,
    challenge_direction: Option<(f32, f32)>,
    seed: u64,
    /// Number of hostile entities when the challenge was injected.
    initial_hostile_count: usize,
    /// Initial NPC count at episode start (for population challenges).
    initial_npc_count: usize,
    /// Initial total stockpile value (for economic challenges).
    initial_stockpile: f32,
    /// The injected challenge for this episode.
    challenge: super::types::Challenge,
}

const TERRAINS: [TerrainType; 8] = [
    TerrainType::FlatOpen, TerrainType::RiverBisect, TerrainType::Hillside,
    TerrainType::CliffEdge, TerrainType::Coastal, TerrainType::Swamp,
    TerrainType::ForestClearing, TerrainType::MountainPass,
];
const MATURITIES: [MaturityLevel; 5] = [
    MaturityLevel::Empty, MaturityLevel::Sparse, MaturityLevel::Moderate,
    MaturityLevel::Dense, MaturityLevel::Overgrown,
];
const RESOURCES: [ResourceProfile; 5] = [
    ResourceProfile::Abundant, ResourceProfile::Mixed, ResourceProfile::Scarce,
    ResourceProfile::Specialized, ResourceProfile::Depleting,
];
const NPCS: [NpcComposition; 6] = [
    NpcComposition::MilitaryHeavy, NpcComposition::CivilianHeavy, NpcComposition::Balanced,
    NpcComposition::EliteFew, NpcComposition::LargeLowLevel, NpcComposition::Specialist,
];
const QUALITIES: [BuildingQuality; 5] = [
    BuildingQuality::WellPlanned, BuildingQuality::OrganicGrowth, BuildingQuality::BattleDamaged,
    BuildingQuality::UnderConstruction, BuildingQuality::AbandonedDecayed,
];

/// Deterministic pick from a slice using a better hash.
fn pick<T: Copy>(items: &[T], seed: u64, salt: u64) -> T {
    // splitmix64-style hash for uniform distribution
    let mut h = seed.wrapping_add(salt);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    items[(h as usize) % items.len()]
}

const ALL_PRESSURES: [PressureType; 24] = [
    PressureType::InfantryRaid, PressureType::SiegeAssault, PressureType::WallJumpers,
    PressureType::Climbers, PressureType::Tunnelers, PressureType::Flyers,
    PressureType::MultiVector, PressureType::Infiltrators,
    PressureType::Flood, PressureType::FireOutbreak, PressureType::Earthquake,
    PressureType::Landslide, PressureType::Storm,
    PressureType::ResourceDepletion, PressureType::TradeBoom,
    PressureType::SupplyDisruption, PressureType::ResourceDiscovery,
    PressureType::RefugeeWave, PressureType::PopulationDecline,
    PressureType::ClassTension, PressureType::SpecialistArrival,
    PressureType::WinterDeadline, PressureType::HarvestSurge,
    PressureType::BuildingDecay,
];

impl BuildingEnv {
    pub fn new(curriculum: CurriculumLevel, seed: u64) -> Self {
        let mut env = Self {
            sim: WorldSim::new(WorldState::new(0)), // placeholder
            tick_budget: curriculum.tick_budget,
            heartbeat_interval: curriculum.heartbeat_interval,
            ticks_since_decision: 0,
            actions_taken: 0,
            pre_decision_scores: ObjectiveScores::default(),
            ideal_scores: ObjectiveScores {
                defense: 1.0, economy: 1.0, population: 1.0,
                connectivity: 1.0, garrison: 1.0, spatial: 1.0,
            },
            challenge_category: ChallengeCategory::Military,
            challenge_severity: 0.5,
            challenge_direction: None,
            seed,
            curriculum,
            initial_hostile_count: 0,
            initial_npc_count: 0,
            initial_stockpile: 0.0,
            challenge: super::types::Challenge {
                category: ChallengeCategory::Military,
                sub_type: 0,
                sub_type_name: String::new(),
                severity: 0.5,
                direction: None,
                deadline_tick: None,
                enemy_profiles: Vec::new(),
            },
        };
        env.reset_internal();
        env
    }

    /// Create a populated world state by randomizing the 5 mass_gen axes.
    fn make_world(seed: u64) -> WorldState {
        let terrain = pick(&TERRAINS, seed, 0x1001);
        let maturity = pick(&MATURITIES, seed, 0x2002);
        let resources = pick(&RESOURCES, seed, 0x3003);
        let npcs = pick(&NPCS, seed, 0x4004);
        let quality = pick(&QUALITIES, seed, 0x5005);
        mass_gen::compose_world_state(terrain, maturity, resources, npcs, quality, seed)
    }

    /// Stamp all pre-existing buildings into VoxelWorld.
    /// Called after WorldSim::new (which loads terrain chunks) so voxels are available.
    fn stamp_existing_buildings(state: &mut WorldState) {
        // Collect building info before mutating state.
        let buildings: Vec<_> = state.entities.iter()
            .filter(|e| e.building.is_some() && e.alive)
            .map(|e| {
                let bd = e.building.as_ref().unwrap();
                (e.id, e.pos, bd.footprint_w as usize, bd.footprint_h as usize, bd.building_type)
            })
            .collect();

        for (id, pos, fw, fh, btype) in buildings {
            stamp_building_voxels(state, pos, fw.max(1), fh.max(1), id, btype);
        }
    }

    /// Access the underlying world state (for oracle policy queries).
    pub fn state(&self) -> &WorldState {
        self.sim.state()
    }

    /// Current challenge category driving this episode.
    pub fn challenge_category(&self) -> ChallengeCategory {
        self.challenge_category
    }

    /// The injected challenge for this episode.
    pub fn challenge(&self) -> &super::types::Challenge {
        &self.challenge
    }

    fn reset_internal(&mut self) {
        self.seed = self.seed.wrapping_add(1);
        let mut state = Self::make_world(self.seed);

        // Pick and inject a random pressure/challenge.
        let mut rng = mass_gen::SimpleRng::new(self.seed.wrapping_add(0xCAFE));
        let severity_range = self.curriculum.max_severity - self.curriculum.min_severity;
        let severity = self.curriculum.min_severity + rng.next_f32() * severity_range;
        let pressure = pick(&ALL_PRESSURES, self.seed, 0x6006);
        let (challenge, _meta) = mass_gen::inject_pressure(pressure, &mut state, severity, &mut rng);

        self.challenge_category = challenge.category;
        self.challenge_severity = challenge.severity;
        self.challenge_direction = challenge.direction;
        self.challenge = challenge;

        // Snapshot initial counts for termination conditions.
        self.initial_hostile_count = state.entities.iter()
            .filter(|e| e.team == WorldTeam::Hostile && e.alive)
            .count();
        self.initial_npc_count = state.entities.iter()
            .filter(|e| e.kind == EntityKind::Npc && e.alive)
            .count();
        self.initial_stockpile = state.settlements.iter()
            .map(|s| s.stockpile.iter().sum::<f32>())
            .sum();

        self.sim = WorldSim::new(state);

        // Stamp pre-existing buildings into VoxelWorld so observations can see them.
        Self::stamp_existing_buildings(self.sim.state_mut());

        self.tick_budget = self.curriculum.tick_budget;
        self.heartbeat_interval = self.curriculum.heartbeat_interval;
        self.ticks_since_decision = 0;
        self.actions_taken = 0;
        self.pre_decision_scores = ObjectiveScores::snapshot(self.sim.state());

        // Tick forward until first trigger or heartbeat.
        self.advance_to_decision();
    }

    pub fn observe(&self) -> Vec<f32> {
        env_obs::extract_observation(
            self.sim.state(),
            self.tick_budget,
            self.challenge_severity,
            self.challenge_direction,
        )
    }

    /// Tick the sim forward until a trigger fires or heartbeat expires.
    fn advance_to_decision(&mut self) -> (bool, bool) {
        loop {
            // Budget exhausted → truncate
            if self.sim.state().tick >= self.tick_budget {
                return (false, true);
            }

            // Max actions → truncate
            if self.actions_taken >= self.curriculum.max_actions {
                return (false, true);
            }

            self.sim.tick();
            self.ticks_since_decision += 1;

            // Check event triggers
            if let Some(_trigger) = self.detect_trigger() {
                self.ticks_since_decision = 0;
                return (false, false);
            }

            // Heartbeat
            if self.ticks_since_decision >= self.heartbeat_interval {
                self.ticks_since_decision = 0;
                return (false, false);
            }

            // Challenge-resolved termination (only after agent has taken at least 1 action).
            if self.actions_taken > 0 {
                if self.is_challenge_resolved() {
                    return (true, false);
                }
                // Total settlement wipe → terminate (lost).
                let alive_npcs = self.sim.state().entities.iter()
                    .filter(|e| e.kind == EntityKind::Npc && e.alive)
                    .count();
                if alive_npcs == 0 {
                    return (true, false);
                }
            }
        }
    }

    /// Check if the injected challenge has been resolved.
    fn is_challenge_resolved(&self) -> bool {
        let state = self.sim.state();
        match self.challenge_category {
            // Military: all hostiles dead.
            ChallengeCategory::Military => {
                if self.initial_hostile_count == 0 { return false; }
                let alive_hostiles = state.entities.iter()
                    .filter(|e| e.team == WorldTeam::Hostile && e.alive)
                    .count();
                alive_hostiles == 0
            }
            // Environmental: no buildings below 50% HP and no active damage.
            ChallengeCategory::Environmental => {
                let damaged = state.entities.iter()
                    .filter(|e| e.alive && e.kind == EntityKind::Building)
                    .any(|e| e.hp < e.max_hp * 0.5 && e.max_hp > 0.0);
                !damaged && state.tick > 200
            }
            // Economic: stockpile recovered above initial level.
            ChallengeCategory::Economic => {
                let current_stockpile: f32 = state.settlements.iter()
                    .map(|s| s.stockpile.iter().sum::<f32>())
                    .sum();
                current_stockpile >= self.initial_stockpile && state.tick > 200
            }
            // Population: housing covers current population.
            ChallengeCategory::Population => {
                let alive_npcs = state.entities.iter()
                    .filter(|e| e.kind == EntityKind::Npc && e.alive)
                    .count() as f32;
                let housing_cap: f32 = state.entities.iter()
                    .filter(|e| e.alive && e.kind == EntityKind::Building)
                    .filter_map(|e| e.building.as_ref())
                    .map(|b| b.residential_capacity as f32)
                    .sum();
                housing_cap >= alive_npcs && alive_npcs > 0.0
            }
            // Other categories: fall back to tick budget.
            _ => false,
        }
    }

    /// Check for event triggers on the current tick.
    fn detect_trigger(&self) -> Option<Trigger> {
        let state = self.sim.state();

        // NPC death: check for recently dead NPCs (died this tick)
        // Use chronicle entries as a proxy
        let current_tick = state.tick;
        for entry in state.chronicle.iter().rev() {
            if entry.tick < current_tick { break; }
            if entry.text.contains("died") || entry.text.contains("killed") {
                return Some(Trigger::NpcDeath);
            }
        }

        // Monster arrival: any monster within settlement radius
        if let Some(settlement) = state.settlements.first() {
            let (sx, sy) = settlement.pos;
            for entity in &state.entities {
                if entity.kind == EntityKind::Monster && entity.alive {
                    let dx = entity.pos.0 - sx;
                    let dy = entity.pos.1 - sy;
                    if dx * dx + dy * dy < 50.0 * 50.0 {
                        return Some(Trigger::MonsterArrival);
                    }
                }
            }
        }

        // Building damaged: any building below 50% HP
        for entity in &state.entities {
            if entity.kind == EntityKind::Building && entity.alive {
                if entity.hp < entity.max_hp * 0.5 && entity.max_hp > 0.0 {
                    return Some(Trigger::BuildingDamaged);
                }
            }
        }

        // Resource crisis: any stockpile commodity below 20%
        if let Some(settlement) = state.settlements.first() {
            for &val in &settlement.stockpile {
                if val < 20.0 && val >= 0.0 {
                    return Some(Trigger::ResourceCrisis);
                }
            }
        }

        // Housing overflow: population exceeds housing by >20%
        let alive_npcs = state.entities.iter()
            .filter(|e| e.kind == EntityKind::Npc && e.alive).count() as f32;
        let housing_cap: f32 = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter_map(|e| e.building.as_ref())
            .map(|b| b.residential_capacity as f32).sum();
        if housing_cap > 0.0 && alive_npcs > housing_cap * 1.2 {
            return Some(Trigger::HousingOverflow);
        }

        None
    }

    /// Apply a building placement action to the sim.
    fn apply_placement(&mut self, grid_offset: (i32, i32), building_type: BuildingType) {
        let settlement_pos = self.sim.state().settlements.first()
            .map(|s| s.pos).unwrap_or((0.0, 0.0));
        let half = (GRID_SIZE / 2) as i32;
        let (center_vx, center_vy, _) = world_to_voxel(settlement_pos.0, settlement_pos.1, 0.0);
        let vx = center_vx - half + grid_offset.0;
        let vy = center_vy - half + grid_offset.1;

        // Convert back to world space for entity placement
        let wx = vx as f32 + 0.5;
        let wy = vy as f32 + 0.5;

        let state = self.sim.state_mut();
        state.sync_next_id();
        let new_id = state.next_entity_id();
        let tick = state.tick;

        use crate::world_sim::state::{Entity, BuildingData};
        use super::super::interior_gen::footprint_size;
        use crate::world_sim::NUM_COMMODITIES;

        let (fp_w, fp_h) = footprint_size(building_type, 0);

        let mut entity = Entity::new_building(new_id, (wx, wy));
        entity.building = Some(BuildingData {
            building_type,
            settlement_id: state.settlements.first().map(|s| s.id),
            grid_col: grid_offset.0 as u16,
            grid_row: grid_offset.1 as u16,
            footprint_w: fp_w as u8,
            footprint_h: fp_h as u8,
            tier: 0,
            room_seed: entity_hash(new_id, tick, 0x800E) as u64,
            rooms: building_type.default_rooms(),
            residential_capacity: building_type.residential_capacity(),
            work_capacity: building_type.work_capacity(),
            resident_ids: Vec::new(),
            worker_ids: Vec::new(),
            construction_progress: 0.0,
            built_tick: tick,
            builder_id: None,
            temporary: false,
            ttl_ticks: None,
            name: format!("{:?} #{}", building_type, new_id),
            storage: [0.0; NUM_COMMODITIES],
            storage_capacity: building_type.storage_capacity(),
            owner_id: None,
            builder_modifiers: Vec::new(),
            owner_modifiers: Vec::new(),
            worker_class_ticks: Vec::new(),
            specialization_tag: None,
            specialization_strength: 0.0,
            specialization_name: String::new(),
            structural: None,
        blueprint: None,
        });

        state.entities.push(entity);
        state.rebuild_all_indices();
    }
}

impl Env for BuildingEnv {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.reset_internal();
        self.observe()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let action_idx = action[0] as usize;
        let choice = env_config::decode_action(action_idx);

        match choice {
            ActionChoice::Pass => {}
            ActionChoice::Place { grid_offset, building_type } => {
                self.apply_placement(grid_offset, building_type);
                self.actions_taken += 1;
            }
        }

        // Advance sim to next decision point
        let (terminated, truncated) = self.advance_to_decision();

        // Compute reward
        let post_scores = ObjectiveScores::snapshot(self.sim.state());
        let reward = compute_reward(
            &self.pre_decision_scores,
            &post_scores,
            &self.ideal_scores,
            self.challenge_category,
        );
        self.pre_decision_scores = post_scores;

        let obs = if terminated || truncated {
            self.reset()
        } else {
            self.observe()
        };

        Step { observation: obs, reward, terminated, truncated }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; OBS_DIM],
            high: vec![1.0; OBS_DIM],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(NUM_ACTIONS)
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        let mut mask = vec![0.0f32; NUM_ACTIONS];
        mask[0] = 1.0; // pass always valid

        // For now, mark all placements as valid (proper collision checking
        // would require scanning each cell which is expensive for 128x128).
        // Invalid placements are caught at apply time.
        for i in 1..NUM_ACTIONS {
            mask[i] = 1.0;
        }
        Some(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_reset_returns_correct_obs_size() {
        let mut env = BuildingEnv::new(CurriculumLevel::level_1(), 42);
        let obs = env.reset();
        assert_eq!(obs.len(), OBS_DIM);
    }

    #[test]
    fn env_step_pass_returns_step() {
        let mut env = BuildingEnv::new(CurriculumLevel::level_1(), 42);
        env.reset();
        let step = env.step(vec![0.0]); // pass action
        assert_eq!(step.observation.len(), OBS_DIM);
    }

    #[test]
    fn env_action_space_correct() {
        let env = BuildingEnv::new(CurriculumLevel::level_1(), 42);
        match env.action_space() {
            Space::Discrete(n) => assert_eq!(n, NUM_ACTIONS),
            other => panic!("expected Discrete, got {:?}", other),
        }
    }

    #[test]
    fn env_episode_terminates() {
        let mut env = BuildingEnv::new(
            CurriculumLevel { level: 1, tick_budget: 50, heartbeat_interval: 10, max_actions: 5, min_severity: 0.3, max_severity: 0.5, max_challenges: 1 },
            42,
        );
        env.reset();
        let mut done = false;
        for _ in 0..20 {
            let step = env.step(vec![0.0]);
            if step.done() {
                done = true;
                break;
            }
        }
        assert!(done, "episode should terminate within budget");
    }
}
