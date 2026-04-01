use serde::{Deserialize, Serialize};

use super::generation::build_seeded_overworld;

#[derive(Clone, Serialize, Deserialize)]
pub struct AttentionState {
    pub switch_cooldown_turns: u32,
    pub switch_cooldown_max: u32,
    pub global_energy: f32,
    pub max_energy: f32,
    pub switch_cost: f32,
    pub regen_per_turn: f32,
}

impl Default for AttentionState {
    fn default() -> Self {
        AttentionState {
            switch_cooldown_turns: 0,
            switch_cooldown_max: 3,
            global_energy: 100.0,
            max_energy: 100.0,
            switch_cost: 20.0,
            regen_per_turn: 6.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverworldRegion {
    pub id: usize,
    pub name: String,
    pub neighbors: Vec<usize>,
    pub owner_faction_id: usize,
    pub mission_slot: Option<usize>,
    pub unrest: f32,
    pub control: f32,
    pub intel_level: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VassalSpecialty {
    Siege,
    Patrol,
    Escort,
    Logistics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VassalPost {
    Roaming,
    ZoneManager,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionVassal {
    pub id: u32,
    pub name: String,
    pub martial: f32,
    pub loyalty: f32,
    pub specialty: VassalSpecialty,
    pub post: VassalPost,
    pub home_region_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionState {
    pub id: usize,
    pub name: String,
    pub strength: f32,
    pub cohesion: f32,
    pub war_goal_faction_id: Option<usize>,
    pub war_focus: f32,
    pub vassals: Vec<FactionVassal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverworldMap {
    pub regions: Vec<OverworldRegion>,
    pub factions: Vec<FactionState>,
    pub current_region: usize,
    pub selected_region: usize,
    pub travel_cooldown_turns: u32,
    pub travel_cooldown_max: u32,
    pub travel_cost: f32,
    pub next_vassal_id: u32,
    pub map_seed: u64,
}

pub(crate) const DEFAULT_OVERWORLD_SEED: u64 = 0x0A11_CE55_1BAD_C0DE;

impl OverworldMap {
    pub fn from_seed(seed: u64) -> Self {
        build_seeded_overworld(seed)
    }
}

impl Default for OverworldMap {
    fn default() -> Self {
        OverworldMap::from_seed(DEFAULT_OVERWORLD_SEED)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashpointChain {
    pub id: u32,
    pub mission_slot: usize,
    pub region_id: usize,
    pub attacker_faction_id: usize,
    pub defender_faction_id: usize,
    pub stage: u8,
    pub completed: bool,
    #[serde(default)]
    pub companion_hook_hero_id: Option<u32>,
    #[serde(default)]
    pub intent: FlashpointIntent,
    #[serde(default)]
    pub objective: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashpointState {
    pub chains: Vec<FlashpointChain>,
    pub next_id: u32,
    pub notice: String,
}

impl Default for FlashpointState {
    fn default() -> Self {
        FlashpointState {
            chains: Vec::new(),
            next_id: 1,
            notice: "No active flashpoints.".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FlashpointIntent {
    #[default]
    StealthPush,
    DirectAssault,
    CivilianFirst,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionCommander {
    pub faction_id: usize,
    pub name: String,
    pub aggression: f32,
    pub cooperation_bias: f32,
    pub competence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommanderIntentKind {
    StabilizeBorder,
    JointMission,
    Raid,
    TrainingExchange,
    RecruitBorrow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommanderIntent {
    pub faction_id: usize,
    pub region_id: usize,
    pub mission_slot: Option<usize>,
    pub urgency: f32,
    pub kind: CommanderIntentKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommanderState {
    pub commanders: Vec<FactionCommander>,
    pub intents: Vec<CommanderIntent>,
}

impl Default for CommanderState {
    fn default() -> Self {
        CommanderState {
            commanders: vec![
                FactionCommander {
                    faction_id: 0,
                    name: "Marshal Elowen".to_string(),
                    aggression: 0.35,
                    cooperation_bias: 0.75,
                    competence: 0.82,
                },
                FactionCommander {
                    faction_id: 1,
                    name: "Lord Caradoc".to_string(),
                    aggression: 0.78,
                    cooperation_bias: 0.38,
                    competence: 0.74,
                },
                FactionCommander {
                    faction_id: 2,
                    name: "Steward Nima".to_string(),
                    aggression: 0.44,
                    cooperation_bias: 0.68,
                    competence: 0.71,
                },
            ],
            intents: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiplomacyState {
    pub player_faction_id: usize,
    pub relations: Vec<Vec<i32>>,
}

impl Default for DiplomacyState {
    fn default() -> Self {
        DiplomacyState {
            player_faction_id: 0,
            relations: vec![vec![0, 10, 16], vec![10, 0, -8], vec![16, -8, 0]],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionOfferKind {
    JointMission,
    RivalRaid,
    TrainingLoan,
    RecruitBorrow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionOffer {
    pub id: u32,
    pub from_faction_id: usize,
    pub region_id: usize,
    pub mission_slot: Option<usize>,
    pub kind: InteractionOfferKind,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionBoard {
    pub offers: Vec<InteractionOffer>,
    pub selected: usize,
    pub notice: String,
    pub next_offer_id: u32,
}

impl Default for InteractionBoard {
    fn default() -> Self {
        InteractionBoard {
            offers: Vec::new(),
            selected: 0,
            notice: "No diplomatic proposals yet.".to_string(),
            next_offer_id: 1,
        }
    }
}
