use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Component)]
pub struct Hero {
    pub name: String,
}

#[derive(Component)]
pub struct Enemy {
    pub name: String,
}

#[derive(Component)]
pub struct Stress {
    pub value: f32,
    pub max: f32,
}

#[derive(Component)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

#[derive(Component)]
pub struct HeroAbilities {
    pub focus_fire_cooldown: u32,
    pub stabilize_cooldown: u32,
    pub sabotage_charge_cooldown: u32,
}

#[derive(Component)]
pub struct EnemyAI {
    pub base_attack_power: f32,
    pub turns_until_attack: u32,
    pub attack_interval: u32,
    pub enraged_threshold: f32,
}

#[derive(Component)]
pub struct MissionObjective {
    pub description: String,
    pub completed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoomType {
    Entry,
    Pressure,
    Pivot,
    Setpiece,
    Recovery,
    Climax,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionNode {
    pub verb: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomMetadata {
    pub room_id: String,
    pub room_name: String,
    pub room_type: RoomType,
    pub interaction_nodes: Vec<InteractionNode>,
    pub threat_budget: f32,
    pub sabotage_threshold: f32,
}

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct MissionMap {
    pub map_name: String,
    pub rooms: Vec<RoomMetadata>,
}

impl Default for MissionMap {
    fn default() -> Self {
        MissionMap {
            map_name: "Emberwell Reliquary".to_string(),
            rooms: vec![
                RoomMetadata {
                    room_id: "entry_vault".to_string(),
                    room_name: "Moon Gate Vault".to_string(),
                    room_type: RoomType::Entry,
                    interaction_nodes: vec![InteractionNode {
                        verb: "Drain".to_string(),
                        description: "Turn the rusted sluice wheel to lower cursed water."
                            .to_string(),
                    }],
                    threat_budget: 8.0,
                    sabotage_threshold: 0.0,
                },
                RoomMetadata {
                    room_id: "flood_walk".to_string(),
                    room_name: "Floodgate Walk".to_string(),
                    room_type: RoomType::Pressure,
                    interaction_nodes: vec![InteractionNode {
                        verb: "Flood".to_string(),
                        description: "Crack the gate to split enemy lines at the cost of alert."
                            .to_string(),
                    }],
                    threat_budget: 14.0,
                    sabotage_threshold: 18.0,
                },
                RoomMetadata {
                    room_id: "reliquary_fork".to_string(),
                    room_name: "Reliquary Fork".to_string(),
                    room_type: RoomType::Pivot,
                    interaction_nodes: vec![InteractionNode {
                        verb: "Dispel".to_string(),
                        description: "Shatter wardstones to remove cultist protection auras."
                            .to_string(),
                    }],
                    threat_budget: 20.0,
                    sabotage_threshold: 36.0,
                },
                RoomMetadata {
                    room_id: "bone_engine".to_string(),
                    room_name: "Bone Engine Chamber".to_string(),
                    room_type: RoomType::Setpiece,
                    interaction_nodes: vec![InteractionNode {
                        verb: "Collapse".to_string(),
                        description: "Drop ossuary chains to stagger the grave colossus."
                            .to_string(),
                    }],
                    threat_budget: 28.0,
                    sabotage_threshold: 58.0,
                },
                RoomMetadata {
                    room_id: "quiet_cloister".to_string(),
                    room_name: "Quiet Cloister".to_string(),
                    room_type: RoomType::Recovery,
                    interaction_nodes: vec![InteractionNode {
                        verb: "Sanctify".to_string(),
                        description: "Channel a sanctuary sigil to reduce stress for lost time."
                            .to_string(),
                    }],
                    threat_budget: 12.0,
                    sabotage_threshold: 78.0,
                },
                RoomMetadata {
                    room_id: "ritual_nave".to_string(),
                    room_name: "Ritual Nave".to_string(),
                    room_type: RoomType::Climax,
                    interaction_nodes: vec![InteractionNode {
                        verb: "Rupture".to_string(),
                        description: "Overload the final anchor and expose the lich conduit."
                            .to_string(),
                    }],
                    threat_budget: 32.0,
                    sabotage_threshold: 92.0,
                },
            ],
        }
    }
}

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct RunState {
    pub global_turn: u32,
}

impl Default for RunState {
    fn default() -> Self {
        RunState { global_turn: 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionResult {
    InProgress,
    Victory,
    Defeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TacticalMode {
    Balanced,
    Aggressive,
    Defensive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionSnapshot {
    pub mission_name: String,
    pub bound_region_id: Option<usize>,
    pub mission_active: bool,
    pub result: MissionResult,
    pub turns_remaining: u32,
    pub reactor_integrity: f32,
    pub sabotage_progress: f32,
    pub sabotage_goal: f32,
    pub alert_level: f32,
    pub room_index: usize,
    pub tactical_mode: TacticalMode,
    pub command_cooldown_turns: u32,
    pub unattended_turns: u32,
    pub outcome_recorded: bool,
}

impl MissionSnapshot {
    pub fn into_components(self, id: u32) -> (MissionData, MissionProgress, MissionTactics) {
        let data = MissionData {
            id,
            mission_name: self.mission_name,
            bound_region_id: self.bound_region_id,
        };
        let progress = MissionProgress {
            mission_active: self.mission_active,
            result: self.result,
            turns_remaining: self.turns_remaining,
            reactor_integrity: self.reactor_integrity,
            sabotage_progress: self.sabotage_progress,
            sabotage_goal: self.sabotage_goal,
            alert_level: self.alert_level,
            room_index: self.room_index,
            unattended_turns: self.unattended_turns,
            outcome_recorded: self.outcome_recorded,
        };
        let tactics = MissionTactics {
            tactical_mode: self.tactical_mode,
            command_cooldown_turns: self.command_cooldown_turns,
            force_sabotage_order: false,
            force_stabilize_order: false,
        };
        (data, progress, tactics)
    }

    pub fn from_components(data: &MissionData, progress: &MissionProgress, tactics: &MissionTactics) -> Self {
        MissionSnapshot {
            mission_name: data.mission_name.clone(),
            bound_region_id: data.bound_region_id,
            mission_active: progress.mission_active,
            result: progress.result,
            turns_remaining: progress.turns_remaining,
            reactor_integrity: progress.reactor_integrity,
            sabotage_progress: progress.sabotage_progress,
            sabotage_goal: progress.sabotage_goal,
            alert_level: progress.alert_level,
            room_index: progress.room_index,
            tactical_mode: tactics.tactical_mode,
            command_cooldown_turns: tactics.command_cooldown_turns,
            unattended_turns: progress.unattended_turns,
            outcome_recorded: progress.outcome_recorded,
        }
    }
}

/// Component: static identity of a mission entity.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct MissionData {
    pub id: u32,
    pub mission_name: String,
    pub bound_region_id: Option<usize>,
}

/// Component: runtime mutable state of a mission.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct MissionProgress {
    pub mission_active: bool,
    pub result: MissionResult,
    pub turns_remaining: u32,
    pub reactor_integrity: f32,
    pub sabotage_progress: f32,
    pub sabotage_goal: f32,
    pub alert_level: f32,
    pub room_index: usize,
    pub unattended_turns: u32,
    pub outcome_recorded: bool,
}

/// Component: tactical directives for a mission.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct MissionTactics {
    pub tactical_mode: TacticalMode,
    pub command_cooldown_turns: u32,
    pub force_sabotage_order: bool,
    pub force_stabilize_order: bool,
}

/// Component: which hero companion is assigned to a mission entity.
#[derive(Component, Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssignedHero {
    pub hero_id: Option<u32>,
}

/// Marker component: the mission entity currently receiving focused attention.
#[derive(Component, Default)]
pub struct ActiveMission;

/// Resource: lightweight registry mapping entity handles to mission slots.
#[derive(Resource, Default)]
pub struct MissionBoard {
    pub entities: Vec<Entity>,
    pub next_id: u32,
}

pub fn default_mission_snapshots() -> Vec<MissionSnapshot> {
    let mk = |name: &str,
              turns: u32,
              progress: f32,
              alert: f32,
              integrity: f32,
              room_index: usize,
              mode: TacticalMode| MissionSnapshot {
        mission_name: name.to_string(),
        bound_region_id: None,
        mission_active: true,
        result: MissionResult::InProgress,
        turns_remaining: turns,
        reactor_integrity: integrity,
        sabotage_progress: progress,
        sabotage_goal: 100.0,
        alert_level: alert,
        room_index,
        tactical_mode: mode,
        command_cooldown_turns: 0,
        unattended_turns: 0,
        outcome_recorded: false,
    };
    vec![
        mk(
            "Sabotage: Emberwell Reliquary",
            30,
            0.0,
            4.0,
            100.0,
            0,
            TacticalMode::Balanced,
        ),
        mk(
            "Containment: Hollowspire Breach",
            26,
            18.0,
            16.0,
            93.0,
            1,
            TacticalMode::Defensive,
        ),
        mk(
            "Rescue: Ashfen Convoy",
            24,
            28.0,
            22.0,
            90.0,
            2,
            TacticalMode::Aggressive,
        ),
    ]
}

#[derive(Resource, Clone, Serialize, Deserialize)]
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

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
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

const DEFAULT_OVERWORLD_SEED: u64 = 0x0A11_CE55_1BAD_C0DE;

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

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
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

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
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

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
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

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersonalityArchetype {
    Vanguard,
    Guardian,
    Tactician,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroCompanion {
    pub id: u32,
    pub name: String,
    pub origin_faction_id: usize,
    pub origin_region_id: usize,
    pub backstory: String,
    pub archetype: PersonalityArchetype,
    pub loyalty: f32,
    pub stress: f32,
    pub fatigue: f32,
    pub injury: f32,
    pub resolve: f32,
    pub active: bool,
    pub deserter: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecruitCandidate {
    pub id: u32,
    pub codename: String,
    pub origin_faction_id: usize,
    pub origin_region_id: usize,
    pub backstory: String,
    pub archetype: PersonalityArchetype,
    pub resolve: f32,
    pub loyalty_bias: f32,
    pub risk_tolerance: f32,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct CampaignRoster {
    pub heroes: Vec<HeroCompanion>,
    pub recruit_pool: Vec<RecruitCandidate>,
    pub next_id: u32,
    pub generation_counter: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsequenceRecord {
    pub turn: u32,
    pub mission_name: String,
    pub result: MissionResult,
    pub hero_id: Option<u32>,
    pub summary: String,
}

#[derive(Resource, Debug, Clone, Default, Serialize, Deserialize)]
pub struct CampaignLedger {
    pub records: Vec<ConsequenceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignEvent {
    pub turn: u32,
    pub summary: String,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct CampaignEventLog {
    pub entries: Vec<CampaignEvent>,
    pub max_entries: usize,
}

impl Default for CampaignEventLog {
    fn default() -> Self {
        CampaignEventLog {
            entries: Vec::new(),
            max_entries: 120,
        }
    }
}

fn push_campaign_event(log: &mut CampaignEventLog, turn: u32, summary: String) {
    log.entries.push(CampaignEvent { turn, summary });
    if log.entries.len() > log.max_entries {
        let overflow = log.entries.len() - log.max_entries;
        log.entries.drain(0..overflow);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompanionQuestKind {
    Reckoning,
    Homefront,
    RivalOath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompanionQuestStatus {
    Active,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionQuest {
    pub id: u32,
    pub hero_id: u32,
    pub kind: CompanionQuestKind,
    pub status: CompanionQuestStatus,
    pub title: String,
    pub objective: String,
    pub progress: u32,
    pub target: u32,
    pub issued_turn: u32,
    pub reward_loyalty: f32,
    pub reward_resolve: f32,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct CompanionStoryState {
    pub quests: Vec<CompanionQuest>,
    pub next_id: u32,
    pub processed_ledger_len: usize,
    pub notice: String,
}

impl Default for CompanionStoryState {
    fn default() -> Self {
        CompanionStoryState {
            quests: Vec::new(),
            next_id: 1,
            processed_ledger_len: 0,
            notice: "No companion quest updates.".to_string(),
        }
    }
}

fn quest_for_hero(state: &CompanionStoryState, hero_id: u32) -> Option<&CompanionQuest> {
    state
        .quests
        .iter()
        .find(|q| q.hero_id == hero_id && q.status == CompanionQuestStatus::Active)
}

fn quest_for_hero_mut(
    state: &mut CompanionStoryState,
    hero_id: u32,
) -> Option<&mut CompanionQuest> {
    state
        .quests
        .iter_mut()
        .find(|q| q.hero_id == hero_id && q.status == CompanionQuestStatus::Active)
}

fn build_companion_quest(
    hero: &HeroCompanion,
    run_turn: u32,
    seed: u64,
    next_id: u32,
    overworld: &OverworldMap,
) -> CompanionQuest {
    let roll = splitmix64(seed ^ hero.id as u64 ^ run_turn as u64);
    let kind = match roll % 3 {
        0 => CompanionQuestKind::Reckoning,
        1 => CompanionQuestKind::Homefront,
        _ => CompanionQuestKind::RivalOath,
    };
    let region_name = overworld
        .regions
        .iter()
        .find(|r| r.id == hero.origin_region_id)
        .map(|r| r.name.as_str())
        .unwrap_or("the frontier");
    let faction_name = overworld
        .factions
        .get(hero.origin_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("their old banner");

    let (title, objective, target, reward_loyalty, reward_resolve) = match kind {
        CompanionQuestKind::Reckoning => (
            format!("{}: Reckoning Oath", hero.name),
            "Win two assigned missions without a defeat.".to_string(),
            2,
            6.0,
            4.0,
        ),
        CompanionQuestKind::Homefront => (
            format!("{}: Homefront Debt", hero.name),
            format!("Secure one victory tied to {}.", region_name),
            1,
            4.0,
            6.0,
        ),
        CompanionQuestKind::RivalOath => (
            format!("{}: Rival Banner", hero.name),
            format!("Claim two victories to weaken enemies of {}.", faction_name),
            2,
            5.0,
            5.0,
        ),
    };

    CompanionQuest {
        id: next_id,
        hero_id: hero.id,
        kind,
        status: CompanionQuestStatus::Active,
        title,
        objective,
        progress: 0,
        target,
        issued_turn: run_turn,
        reward_loyalty,
        reward_resolve,
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn rand01(seed: u64, salt: u64) -> f32 {
    let v = splitmix64(seed ^ salt);
    (v as f64 / u64::MAX as f64) as f32
}

fn archetype_for(seed: u64, salt: u64) -> PersonalityArchetype {
    match (splitmix64(seed ^ salt) % 3) as u8 {
        0 => PersonalityArchetype::Vanguard,
        1 => PersonalityArchetype::Guardian,
        _ => PersonalityArchetype::Tactician,
    }
}

fn recruit_codename(id: u32, seed: u64) -> String {
    let adj = ["Ash", "Stone", "Silver", "Rook", "Ember", "Gale"];
    let noun = ["Fox", "Lance", "Warden", "Bell", "Sable", "Moth"];
    let a = (splitmix64(seed ^ id as u64) % adj.len() as u64) as usize;
    let n = (splitmix64(seed ^ (id as u64).wrapping_mul(17)) % noun.len() as u64) as usize;
    format!("{} {}", adj[a], noun[n])
}

fn pick_origin(overworld: &OverworldMap, seed: u64, id: u32) -> (usize, usize) {
    if overworld.regions.is_empty() {
        return (0, 0);
    }
    let region_idx =
        (splitmix64(seed ^ (id as u64).wrapping_mul(73)) % overworld.regions.len() as u64) as usize;
    let region = &overworld.regions[region_idx];
    (region.owner_faction_id, region.id)
}

fn backstory_for_recruit(
    codename: &str,
    archetype: PersonalityArchetype,
    faction_name: &str,
    region_name: &str,
    unrest: f32,
    control: f32,
) -> String {
    let archetype_hook = match archetype {
        PersonalityArchetype::Vanguard => "learned to lead from the front",
        PersonalityArchetype::Guardian => "became a shield for civilians",
        PersonalityArchetype::Tactician => "survived by reading every battlefield angle",
    };
    let pressure_hook = if unrest >= 60.0 {
        "after repeated border collapses"
    } else if control >= 70.0 {
        "under disciplined garrison rule"
    } else {
        "through constant jurisdiction disputes"
    };
    format!(
        "{codename} was raised in {region_name} under {faction_name}, {pressure_hook}, and {archetype_hook}."
    )
}

fn vassal_name(faction: &str, id: u32) -> String {
    let titles = [
        "Warden",
        "Reeve",
        "Marshal",
        "Castellan",
        "Captain",
        "Steward",
    ];
    let tags = ["Ash", "Stone", "Gale", "Sable", "Iron", "Dawn"];
    let t = (splitmix64(id as u64) % titles.len() as u64) as usize;
    let g = (splitmix64((id as u64).wrapping_mul(13)) % tags.len() as u64) as usize;
    format!("{} {} of {}", titles[t], tags[g], faction)
}

fn specialty_for(id: u32) -> VassalSpecialty {
    match splitmix64((id as u64).wrapping_mul(29)) % 4 {
        0 => VassalSpecialty::Siege,
        1 => VassalSpecialty::Patrol,
        2 => VassalSpecialty::Escort,
        _ => VassalSpecialty::Logistics,
    }
}

fn target_vassal_count(strength: f32) -> usize {
    ((strength / 18.0).round() as i32).clamp(2, 14) as usize
}

fn hex_distance(a: (i32, i32), b: (i32, i32)) -> i32 {
    let dq = a.0 - b.0;
    let dr = a.1 - b.1;
    (dq.abs() + dr.abs() + (dq + dr).abs()) / 2
}

fn build_seeded_overworld(seed: u64) -> OverworldMap {
    let faction_names = ["Guild Compact", "Iron Marches", "River Concord"];
    let mut coords = Vec::new();
    let radius = 2;
    for q in -radius..=radius {
        let r_min = (-radius).max(-q - radius);
        let r_max = radius.min(-q + radius);
        for r in r_min..=r_max {
            coords.push((q, r));
        }
    }
    coords.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let name_left = [
        "Ash", "Hollow", "Rift", "Ember", "Dawn", "Stone", "Mist", "Iron",
    ];
    let name_right = [
        "Bastion", "Reach", "March", "Fen", "Spire", "Crossing", "Watch", "Roads",
    ];
    let neighbor_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

    let mut regions = Vec::with_capacity(coords.len());
    for (id, coord) in coords.iter().enumerate() {
        let mut neighbors = Vec::new();
        for (dq, dr) in neighbor_dirs {
            let target = (coord.0 + dq, coord.1 + dr);
            if let Some(idx) = coords.iter().position(|c| *c == target) {
                neighbors.push(idx);
            }
        }
        neighbors.sort_unstable();
        let left =
            (splitmix64(seed ^ (id as u64).wrapping_mul(17)) % name_left.len() as u64) as usize;
        let right =
            (splitmix64(seed ^ (id as u64).wrapping_mul(31)) % name_right.len() as u64) as usize;
        regions.push(OverworldRegion {
            id,
            name: format!("{} {} {}", name_left[left], name_right[right], id + 1),
            neighbors,
            owner_faction_id: 0,
            mission_slot: None,
            unrest: 0.0,
            control: 0.0,
            intel_level: 0.0,
        });
    }

    let capital_coords = [(-2, 0), (2, 0), (0, 2)];
    let capital_ids = capital_coords
        .iter()
        .map(|cc| coords.iter().position(|c| c == cc).unwrap_or(0))
        .collect::<Vec<_>>();

    for (id, coord) in coords.iter().enumerate() {
        let mut best_faction = 0usize;
        let mut best_score = f32::MAX;
        for (faction_id, cap_coord) in capital_coords.iter().enumerate() {
            let dist = hex_distance(*coord, *cap_coord) as f32;
            let jitter = (rand01(seed, 10_000 + (id as u64) * 73 + faction_id as u64) - 0.5) * 1.1;
            let score = dist + jitter;
            if score < best_score {
                best_score = score;
                best_faction = faction_id;
            }
        }
        regions[id].owner_faction_id = best_faction;
    }
    for (faction_id, cap_id) in capital_ids.iter().enumerate() {
        if *cap_id < regions.len() {
            regions[*cap_id].owner_faction_id = faction_id;
        }
    }

    let owners = regions
        .iter()
        .map(|r| r.owner_faction_id)
        .collect::<Vec<_>>();
    for region in &mut regions {
        let border_pressure = region
            .neighbors
            .iter()
            .filter(|n| {
                owners.get(**n).copied().unwrap_or(region.owner_faction_id)
                    != region.owner_faction_id
            })
            .count() as f32;
        let unrest =
            (14.0 + border_pressure * 11.0 + rand01(seed, 20_000 + region.id as u64) * 20.0)
                .clamp(0.0, 100.0);
        region.unrest = unrest;
        region.control = (100.0 - unrest).clamp(0.0, 100.0);
        let intel = (35.0
            + rand01(seed, 25_000 + region.id as u64) * 50.0
            + if region.owner_faction_id == 0 {
                8.0
            } else {
                0.0
            })
        .clamp(0.0, 100.0);
        region.intel_level = intel;
    }

    for (slot, faction_id) in (0..faction_names.len()).enumerate() {
        if let Some((idx, _)) = regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.owner_faction_id == faction_id)
            .max_by(|(_, a), (_, b)| a.unrest.total_cmp(&b.unrest).then(a.id.cmp(&b.id)))
        {
            regions[idx].mission_slot = Some(slot);
        }
    }

    let mut factions = Vec::with_capacity(faction_names.len());
    for (id, name) in faction_names.iter().enumerate() {
        let owned = regions
            .iter()
            .filter(|r| r.owner_faction_id == id)
            .collect::<Vec<_>>();
        let owned_count = owned.len().max(1) as f32;
        let avg_unrest = owned.iter().map(|r| r.unrest).sum::<f32>() / owned_count;
        let avg_control = owned.iter().map(|r| r.control).sum::<f32>() / owned_count;
        let strength = (70.0
            + owned_count * 8.0
            + avg_control * 0.35
            + rand01(seed, 30_000 + id as u64) * 12.0)
            .clamp(40.0, 180.0);
        let cohesion = (48.0 + avg_control * 0.22 - avg_unrest * 0.08
            + rand01(seed, 40_000 + id as u64) * 10.0)
            .clamp(20.0, 95.0);
        factions.push(FactionState {
            id,
            name: (*name).to_string(),
            strength,
            cohesion,
            war_goal_faction_id: None,
            war_focus: 0.0,
            vassals: Vec::new(),
        });
    }

    let current_region = capital_ids[0].min(regions.len().saturating_sub(1));
    let mut map = OverworldMap {
        regions,
        factions,
        current_region,
        selected_region: current_region,
        travel_cooldown_turns: 0,
        travel_cooldown_max: 2,
        travel_cost: 12.0,
        next_vassal_id: 1,
        map_seed: seed,
    };
    for idx in 0..map.factions.len() {
        let owned = map
            .regions
            .iter()
            .filter(|r| r.owner_faction_id == idx)
            .map(|r| r.id)
            .collect::<Vec<_>>();
        rebalance_faction_vassals(&mut map.factions[idx], &owned, &mut map.next_vassal_id);
    }
    map
}

fn build_vassal(faction: &FactionState, id: u32) -> FactionVassal {
    FactionVassal {
        id,
        name: vassal_name(&faction.name, id),
        martial: 38.0 + rand01(id as u64, 7) * 48.0,
        loyalty: 42.0 + rand01(id as u64, 17) * 46.0,
        specialty: specialty_for(id),
        post: VassalPost::Roaming,
        home_region_id: faction.id,
    }
}

fn rebalance_faction_vassals(
    faction: &mut FactionState,
    owned_regions: &[usize],
    next_vassal_id: &mut u32,
) {
    let target = target_vassal_count(faction.strength);
    while faction.vassals.len() < target {
        let id = *next_vassal_id;
        *next_vassal_id = next_vassal_id.saturating_add(1);
        faction.vassals.push(build_vassal(faction, id));
    }
    if faction.vassals.len() > target {
        faction
            .vassals
            .sort_by(|a, b| a.loyalty.total_cmp(&b.loyalty).then(a.id.cmp(&b.id)));
        faction.vassals.truncate(target);
    }

    if faction.vassals.is_empty() {
        return;
    }
    let region_cycle = if owned_regions.is_empty() {
        vec![faction.id]
    } else {
        owned_regions.to_vec()
    };
    let desired_managers = usize::min(
        region_cycle.len(),
        usize::max(1, (faction.vassals.len() as f32 * 0.35).round() as usize),
    );

    faction.vassals.sort_by_key(|v| v.id);
    for (idx, vassal) in faction.vassals.iter_mut().enumerate() {
        let home = region_cycle[idx % region_cycle.len()];
        vassal.home_region_id = home;
        vassal.post = if idx < desired_managers {
            VassalPost::ZoneManager
        } else {
            VassalPost::Roaming
        };
    }
}

pub fn generate_recruit(seed: u64, id: u32) -> RecruitCandidate {
    let map = OverworldMap::default();
    generate_recruit_for_overworld(seed, id, &map)
}

pub fn generate_recruit_for_overworld(
    seed: u64,
    id: u32,
    overworld: &OverworldMap,
) -> RecruitCandidate {
    let codename = recruit_codename(id, seed);
    let archetype = archetype_for(seed, id as u64);
    let (origin_faction_id, origin_region_id) = pick_origin(overworld, seed, id);
    let (faction_name, region_name, unrest, control) = {
        let region = overworld
            .regions
            .iter()
            .find(|r| r.id == origin_region_id)
            .or_else(|| overworld.regions.first());
        let fallback_region = "Unknown March";
        let fallback_faction = "Unaligned House";
        match region {
            Some(r) => {
                let f = overworld
                    .factions
                    .get(r.owner_faction_id)
                    .map(|x| x.name.as_str())
                    .unwrap_or(fallback_faction);
                (f.to_string(), r.name.clone(), r.unrest, r.control)
            }
            None => (
                fallback_faction.to_string(),
                fallback_region.to_string(),
                50.0,
                50.0,
            ),
        }
    };
    RecruitCandidate {
        id,
        codename: codename.clone(),
        origin_faction_id,
        origin_region_id,
        backstory: backstory_for_recruit(
            &codename,
            archetype,
            &faction_name,
            &region_name,
            unrest,
            control,
        ),
        archetype,
        resolve: 52.0 + rand01(seed, id as u64 + 11) * 34.0,
        loyalty_bias: 44.0 + rand01(seed, id as u64 + 31) * 30.0,
        risk_tolerance: 28.0 + rand01(seed, id as u64 + 53) * 55.0,
    }
}

pub fn sync_roster_lore_with_overworld_system(
    overworld: Res<OverworldMap>,
    mut roster: ResMut<CampaignRoster>,
) {
    if overworld.regions.is_empty() || overworld.factions.is_empty() {
        return;
    }
    for recruit in &mut roster.recruit_pool {
        let seed = overworld.map_seed ^ 0xA11C_E555_u64;
        let refreshed = generate_recruit_for_overworld(seed, recruit.id, &overworld);
        recruit.origin_faction_id = refreshed.origin_faction_id;
        recruit.origin_region_id = refreshed.origin_region_id;
        recruit.backstory = refreshed.backstory;
    }
    for hero in &mut roster.heroes {
        if !hero.backstory.is_empty() {
            continue;
        }
        let seed = overworld.map_seed ^ 0xBACC_5700_u64;
        let generated = generate_recruit_for_overworld(seed, hero.id, &overworld);
        hero.origin_faction_id = generated.origin_faction_id;
        hero.origin_region_id = generated.origin_region_id;
        hero.backstory = generated.backstory;
    }
}

impl Default for CampaignRoster {
    fn default() -> Self {
        let mut roster = CampaignRoster {
            heroes: vec![
                HeroCompanion {
                    id: 1,
                    name: "Warden Lyra".to_string(),
                    origin_faction_id: 0,
                    origin_region_id: 0,
                    backstory: String::new(),
                    archetype: PersonalityArchetype::Guardian,
                    loyalty: 72.0,
                    stress: 12.0,
                    fatigue: 10.0,
                    injury: 4.0,
                    resolve: 78.0,
                    active: true,
                    deserter: false,
                },
                HeroCompanion {
                    id: 2,
                    name: "Kade Ember".to_string(),
                    origin_faction_id: 0,
                    origin_region_id: 0,
                    backstory: String::new(),
                    archetype: PersonalityArchetype::Vanguard,
                    loyalty: 64.0,
                    stress: 20.0,
                    fatigue: 18.0,
                    injury: 8.0,
                    resolve: 66.0,
                    active: true,
                    deserter: false,
                },
                HeroCompanion {
                    id: 3,
                    name: "Iris Quill".to_string(),
                    origin_faction_id: 0,
                    origin_region_id: 0,
                    backstory: String::new(),
                    archetype: PersonalityArchetype::Tactician,
                    loyalty: 68.0,
                    stress: 16.0,
                    fatigue: 14.0,
                    injury: 2.0,
                    resolve: 74.0,
                    active: true,
                    deserter: false,
                },
            ],
            recruit_pool: Vec::new(),
            next_id: 4,
            generation_counter: 0,
        };
        refill_recruit_pool(&mut roster);
        let map = OverworldMap::default();
        for hero in &mut roster.heroes {
            let generated =
                generate_recruit_for_overworld(map.map_seed ^ 0xBACC_5700, hero.id, &map);
            hero.origin_faction_id = generated.origin_faction_id;
            hero.origin_region_id = generated.origin_region_id;
            hero.backstory = generated.backstory;
        }
        roster
    }
}

pub fn refill_recruit_pool(roster: &mut CampaignRoster) {
    while roster.recruit_pool.len() < 3 {
        let id = roster.next_id;
        let seed = 0xC0DE_0000_0000_0042_u64 ^ roster.generation_counter;
        roster.recruit_pool.push(generate_recruit(seed, id));
        roster.next_id += 1;
    }
}

pub fn sign_top_recruit(roster: &mut CampaignRoster) -> Option<HeroCompanion> {
    if roster.recruit_pool.is_empty() {
        refill_recruit_pool(roster);
    }
    let recruit = roster.recruit_pool.remove(0);
    let hero = HeroCompanion {
        id: recruit.id,
        name: recruit.codename,
        origin_faction_id: recruit.origin_faction_id,
        origin_region_id: recruit.origin_region_id,
        backstory: recruit.backstory,
        archetype: recruit.archetype,
        loyalty: recruit.loyalty_bias,
        stress: (6.0 + recruit.risk_tolerance * 0.12).clamp(0.0, 100.0),
        fatigue: (5.0 + recruit.risk_tolerance * 0.08).clamp(0.0, 100.0),
        injury: 0.0,
        resolve: recruit.resolve,
        active: true,
        deserter: false,
    };
    roster.heroes.push(hero.clone());
    roster.generation_counter = roster.generation_counter.wrapping_add(1);
    refill_recruit_pool(roster);
    Some(hero)
}

pub fn setup_test_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let hero_mesh = Mesh::from(bevy::math::primitives::Capsule3d {
        radius: 0.6,
        half_length: 0.8,
        ..default()
    });
    let enemy_mesh = Mesh::from(bevy::math::primitives::Cuboid::new(1.2, 1.2, 1.2));
    let ground_mesh = Mesh::from(bevy::math::primitives::Cuboid::new(12.0, 0.2, 8.0));

    let hero_mesh_handle = meshes.add(hero_mesh);
    let enemy_mesh_handle = meshes.add(enemy_mesh);
    let ground_mesh_handle = meshes.add(ground_mesh);

    let hero_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.70, 0.73, 0.82),
        perceptual_roughness: 0.5,
        metallic: 0.05,
        ..default()
    });
    let enemy_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.85, 0.22, 0.22),
        perceptual_roughness: 0.65,
        metallic: 0.1,
        ..default()
    });
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.12, 0.14, 0.18),
        perceptual_roughness: 0.95,
        ..default()
    });

    commands.spawn(PbrBundle {
        mesh: ground_mesh_handle,
        material: ground_material,
        transform: Transform::from_xyz(0.0, -0.1, 0.0),
        ..default()
    });

    commands.spawn((
        Hero {
            name: "Warden Lyra".to_string(),
        },
        Stress {
            value: 0.0,
            max: 100.0,
        },
        Health {
            current: 100.0,
            max: 100.0,
        },
        HeroAbilities {
            focus_fire_cooldown: 0,
            stabilize_cooldown: 0,
            sabotage_charge_cooldown: 0,
        },
        PbrBundle {
            mesh: hero_mesh_handle,
            material: hero_material,
            transform: Transform::from_xyz(-2.5, 0.85, 0.0),
            ..default()
        },
    ));

    commands.spawn((
        Enemy {
            name: "Crypt Sentinel".to_string(),
        },
        Health {
            current: 40.0,
            max: 40.0,
        },
        EnemyAI {
            base_attack_power: 6.0,
            turns_until_attack: 1,
            attack_interval: 2,
            enraged_threshold: 0.5,
        },
        PbrBundle {
            mesh: enemy_mesh_handle,
            material: enemy_material,
            transform: Transform::from_xyz(2.5, 0.6, 0.0),
            ..default()
        },
    ));

    commands.spawn(MissionObjective {
        description: "Rupture the ritual anchor".to_string(),
        completed: false,
    });
}

pub fn setup_test_scene_headless(mut commands: Commands, mut board: ResMut<MissionBoard>) {
    commands.spawn((
        Hero {
            name: "Warden Lyra".to_string(),
        },
        Stress {
            value: 0.0,
            max: 100.0,
        },
        Health {
            current: 100.0,
            max: 100.0,
        },
        HeroAbilities {
            focus_fire_cooldown: 0,
            stabilize_cooldown: 0,
            sabotage_charge_cooldown: 0,
        },
    ));

    commands.spawn((
        Enemy {
            name: "Crypt Sentinel".to_string(),
        },
        Health {
            current: 40.0,
            max: 40.0,
        },
        EnemyAI {
            base_attack_power: 6.0,
            turns_until_attack: 1,
            attack_interval: 2,
            enraged_threshold: 0.5,
        },
    ));

    commands.spawn(MissionObjective {
        description: "Rupture the ritual anchor".to_string(),
        completed: false,
    });

    // Spawn mission entities for the headless scene (mirrors spawn_mission_entities).
    for (i, snap) in default_mission_snapshots().into_iter().enumerate() {
        let id = board.next_id;
        board.next_id += 1;
        let (data, progress, tactics) = snap.into_components(id);
        let mut entity_cmd = commands.spawn((data, progress, tactics, AssignedHero::default()));
        if i == 0 {
            entity_cmd.insert(ActiveMission);
        }
        // We cannot push to board.entities here since commands are deferred;
        // spawn_mission_entities is the canonical path for normal startup.
    }
}

/// Startup system: spawns one ECS entity per default mission and registers
/// entity handles in `MissionBoard`.
pub fn spawn_mission_entities(mut commands: Commands, mut board: ResMut<MissionBoard>) {
    for (i, snap) in default_mission_snapshots().into_iter().enumerate() {
        let id = board.next_id;
        board.next_id += 1;
        let (data, progress, tactics) = snap.into_components(id);
        let entity = commands
            .spawn((data, progress, tactics, AssignedHero::default()))
            .id();
        if i == 0 {
            commands.entity(entity).insert(ActiveMission);
        }
        board.entities.push(entity);
    }
}

pub fn print_game_state(
    run_state: Res<RunState>,
    mission_map: Res<MissionMap>,
    active_query: Query<(&MissionData, &MissionProgress, &MissionTactics), With<ActiveMission>>,
    hero_query: Query<(&Hero, &Stress, &Health)>,
    enemy_query: Query<(&Enemy, &Health)>,
    objective_query: Query<&MissionObjective>,
) {
    println!("--- Global Turn: {} ---", run_state.global_turn);
    if let Ok((data, progress, tactics)) = active_query.get_single() {
        println!(
            "Mission: {} (Active: {}, Result: {:?}, Timer: {})",
            data.mission_name,
            progress.mission_active,
            progress.result,
            progress.turns_remaining
        );
        println!(
            "Sabotage: progress {:.1}/{:.1}, reactor {:.1}, alert {:.1}",
            progress.sabotage_progress,
            progress.sabotage_goal,
            progress.reactor_integrity,
            progress.alert_level
        );
        println!(
            "Tactical mode: {:?}, command cooldown: {}",
            tactics.tactical_mode, tactics.command_cooldown_turns
        );
        if let Some(room) = mission_map.rooms.get(progress.room_index) {
            println!(
                "Map: {} | Room: {} [{} / {:?}] | Threat budget: {:.1}",
                mission_map.map_name, room.room_name, room.room_id, room.room_type, room.threat_budget
            );
            if let Some(interaction) = room.interaction_nodes.first() {
                println!(
                    "Interaction: {} -> {}",
                    interaction.verb, interaction.description
                );
            }
        }
    }
    for (hero, stress, health) in hero_query.iter() {
        println!(
            "Hero: {}, HP: {:.1}/{:.1}, Stress: {:.1}/{:.1}",
            hero.name, health.current, health.max, stress.value, stress.max
        );
    }
    for (enemy, health) in enemy_query.iter() {
        println!(
            "Enemy: {}, HP: {:.1}/{:.1}",
            enemy.name, health.current, health.max
        );
    }
    for objective in objective_query.iter() {
        println!(
            "Objective: {} (Completed: {})",
            objective.description, objective.completed
        );
    }
    println!("--------------------");
}

pub fn attention_management_system(
    run_state: Res<RunState>,
    mut attention: ResMut<AttentionState>,
) {
    if run_state.global_turn == 0 {
        return;
    }
    if attention.switch_cooldown_turns > 0 {
        attention.switch_cooldown_turns -= 1;
    }
    attention.global_energy =
        (attention.global_energy + attention.regen_per_turn).min(attention.max_energy);
}

pub fn overworld_cooldown_system(run_state: Res<RunState>, mut overworld: ResMut<OverworldMap>) {
    if run_state.global_turn == 0 {
        return;
    }
    if overworld.travel_cooldown_turns > 0 {
        overworld.travel_cooldown_turns -= 1;
    }
}

fn ensure_faction_mission_slots(overworld: &mut OverworldMap) {
    for region in &mut overworld.regions {
        region.mission_slot = None;
    }
    for slot in 0..overworld.factions.len() {
        if let Some((idx, _)) = overworld
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.owner_faction_id == slot)
            .max_by(|(_, a), (_, b)| a.unrest.total_cmp(&b.unrest).then(a.id.cmp(&b.id)))
        {
            overworld.regions[idx].mission_slot = Some(slot);
        }
    }
}

const FLASHPOINT_TRIGGER_PRESSURE: f32 = 86.0;
const FLASHPOINT_TOTAL_STAGES: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlashpointCompanionHookKind {
    Homefront,
    Aggressor,
    Defender,
}

fn flashpoint_companion_hook_kind(
    chain: &FlashpointChain,
    roster: &CampaignRoster,
) -> Option<(u32, String, FlashpointCompanionHookKind)> {
    let hero_id = chain.companion_hook_hero_id?;
    let hero = roster.heroes.iter().find(|h| h.id == hero_id)?;
    if hero.origin_region_id == chain.region_id {
        return Some((
            hero_id,
            hero.name.clone(),
            FlashpointCompanionHookKind::Homefront,
        ));
    }
    if hero.origin_faction_id == chain.attacker_faction_id {
        return Some((
            hero_id,
            hero.name.clone(),
            FlashpointCompanionHookKind::Aggressor,
        ));
    }
    if hero.origin_faction_id == chain.defender_faction_id {
        return Some((
            hero_id,
            hero.name.clone(),
            FlashpointCompanionHookKind::Defender,
        ));
    }
    None
}

fn apply_flashpoint_companion_hook(
    mission: &mut MissionSnapshot,
    chain: &FlashpointChain,
    roster: &CampaignRoster,
) -> Option<String> {
    let (_hero_id, hero_name, kind) = flashpoint_companion_hook_kind(chain, roster)?;
    match kind {
        FlashpointCompanionHookKind::Homefront => {
            mission.turns_remaining = mission.turns_remaining.saturating_add(2);
            mission.sabotage_progress = (mission.sabotage_progress + 8.0).clamp(0.0, 100.0);
            mission.reactor_integrity = (mission.reactor_integrity + 5.0).clamp(0.0, 100.0);
        }
        FlashpointCompanionHookKind::Aggressor => {
            mission.turns_remaining = mission.turns_remaining.saturating_sub(2).max(8);
            mission.alert_level = (mission.alert_level + 9.0).clamp(0.0, 100.0);
            mission.sabotage_progress = (mission.sabotage_progress + 6.0).clamp(0.0, 100.0);
        }
        FlashpointCompanionHookKind::Defender => {
            mission.alert_level = (mission.alert_level - 7.0).clamp(0.0, 100.0);
            mission.reactor_integrity = (mission.reactor_integrity + 9.0).clamp(0.0, 100.0);
        }
    }
    Some(hero_name)
}

fn flashpoint_stage_label(stage: u8) -> &'static str {
    match stage {
        1 => "Recon Sweep",
        2 => "Sabotage Push",
        _ => "Decisive Assault",
    }
}

fn flashpoint_intent_label(intent: FlashpointIntent) -> &'static str {
    match intent {
        FlashpointIntent::StealthPush => "Stealth Push",
        FlashpointIntent::DirectAssault => "Direct Assault",
        FlashpointIntent::CivilianFirst => "Civilian First",
    }
}

fn flashpoint_hook_objective_suffix(
    chain: &FlashpointChain,
    roster: &CampaignRoster,
) -> Option<String> {
    let (_hero_id, hero_name, kind) = flashpoint_companion_hook_kind(chain, roster)?;
    let detail = match kind {
        FlashpointCompanionHookKind::Homefront => "evacuate districts and secure local allies",
        FlashpointCompanionHookKind::Aggressor => "break enemy command relays before counterfire",
        FlashpointCompanionHookKind::Defender => "hold relief corridors and preserve defenses",
    };
    Some(format!("{hero_name}: {detail}"))
}

fn flashpoint_projection_suffix(
    chain: &FlashpointChain,
    overworld: &OverworldMap,
) -> (String, String) {
    let attacker = overworld
        .factions
        .get(chain.attacker_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Attackers");
    let defender = overworld
        .factions
        .get(chain.defender_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Defenders");
    (
        format!("win=>{} border+recruit", attacker),
        format!("lose=>{} hold-line", defender),
    )
}

fn rewrite_flashpoint_mission_name(
    mission: &mut MissionSnapshot,
    chain: &FlashpointChain,
    overworld: &OverworldMap,
    roster: Option<&CampaignRoster>,
) {
    let stage = chain.stage.clamp(1, FLASHPOINT_TOTAL_STAGES);
    let region_name = overworld
        .regions
        .iter()
        .find(|r| r.id == chain.region_id)
        .map(|r| r.name.as_str())
        .unwrap_or("Unknown Region");
    let attacker_name = overworld
        .factions
        .get(chain.attacker_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Rival host");
    let defender_name = overworld
        .factions
        .get(chain.defender_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Defenders");
    let (win_proj, lose_proj) = flashpoint_projection_suffix(chain, overworld);
    let hook = roster
        .and_then(|r| flashpoint_hook_objective_suffix(chain, r))
        .unwrap_or_else(|| "No companion hook".to_string());
    mission.mission_name = format!(
        "Flashpoint {}/{} [{}|{}]: {} ({} vs {}) | obj={} | {} {}",
        stage,
        FLASHPOINT_TOTAL_STAGES,
        flashpoint_stage_label(stage),
        flashpoint_intent_label(chain.intent),
        region_name,
        attacker_name,
        defender_name,
        hook,
        win_proj,
        lose_proj
    );
}

fn pick_flashpoint_attacker(region: &OverworldRegion, overworld: &OverworldMap) -> Option<usize> {
    let defender = region.owner_faction_id;
    let mut best = None;
    let mut best_score = f32::MIN;
    for neighbor in &region.neighbors {
        let Some(other) = overworld.regions.get(*neighbor) else {
            continue;
        };
        let attacker = other.owner_faction_id;
        if attacker == defender {
            continue;
        }
        let hostility = overworld
            .factions
            .get(attacker)
            .and_then(|f| f.war_goal_faction_id)
            .map(|goal| if goal == defender { 22.0 } else { 0.0 })
            .unwrap_or(0.0);
        let score = overworld
            .factions
            .get(attacker)
            .map(|f| f.strength + f.war_focus * 0.55 + f.cohesion * 0.25 + hostility)
            .unwrap_or(0.0);
        if score > best_score {
            best_score = score;
            best = Some(attacker);
        }
    }
    best
}

fn configure_flashpoint_stage_mission(
    mission: &mut MissionSnapshot,
    chain: &FlashpointChain,
    _overworld: &OverworldMap,
    seed: u64,
) {
    let stage = chain.stage.clamp(1, FLASHPOINT_TOTAL_STAGES);
    let stage_ix = stage as usize - 1;
    let jitter = (rand01(
        seed ^ chain.id as u64,
        chain.region_id as u64 * 97 + stage as u64 * 17,
    ) - 0.5)
        * 6.0;
    let turn_bases = [24.0, 20.0, 16.0];
    let alert_bases = [42.0, 58.0, 74.0];
    let integrity_bases = [84.0, 72.0, 58.0];
    let progress_bases = [20.0, 34.0, 46.0];
    mission.turns_remaining = (turn_bases[stage_ix] + jitter).clamp(10.0, 30.0) as u32;
    mission.alert_level = (alert_bases[stage_ix] + jitter * 1.4).clamp(12.0, 96.0);
    mission.reactor_integrity = (integrity_bases[stage_ix] - jitter * 1.3).clamp(20.0, 100.0);
    mission.sabotage_progress = (progress_bases[stage_ix] + jitter).clamp(0.0, 72.0);
    mission.tactical_mode = match stage {
        1 => TacticalMode::Defensive,
        2 => TacticalMode::Balanced,
        _ => TacticalMode::Aggressive,
    };
    mission.bound_region_id = Some(chain.region_id);
    mission.mission_active = true;
    mission.result = MissionResult::InProgress;
    mission.command_cooldown_turns = 0;
    mission.unattended_turns = 0;
    mission.outcome_recorded = false;
}

fn apply_flashpoint_intent(mission: &mut MissionSnapshot, chain: &FlashpointChain) {
    let stage_scale = 1.0 + (chain.stage as f32 - 1.0) * 0.2;
    match chain.intent {
        FlashpointIntent::StealthPush => {
            mission.alert_level = (mission.alert_level - 7.0 * stage_scale).clamp(0.0, 100.0);
            mission.sabotage_progress =
                (mission.sabotage_progress + 6.0 * stage_scale).clamp(0.0, 100.0);
        }
        FlashpointIntent::DirectAssault => {
            mission.turns_remaining = mission.turns_remaining.saturating_sub(2).max(8);
            mission.alert_level = (mission.alert_level + 9.0 * stage_scale).clamp(0.0, 100.0);
            mission.sabotage_progress =
                (mission.sabotage_progress + 8.0 * stage_scale).clamp(0.0, 100.0);
        }
        FlashpointIntent::CivilianFirst => {
            mission.turns_remaining = mission.turns_remaining.saturating_add(2);
            mission.alert_level = (mission.alert_level - 4.0 * stage_scale).clamp(0.0, 100.0);
            mission.reactor_integrity =
                (mission.reactor_integrity + 8.0 * stage_scale).clamp(0.0, 100.0);
        }
    }
}

fn inject_recruit_for_faction(
    roster: &mut CampaignRoster,
    overworld: &OverworldMap,
    faction_id: usize,
    seed: u64,
) {
    let id = roster.next_id.max(
        roster
            .recruit_pool
            .iter()
            .map(|r| r.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
    );
    roster.next_id = id.saturating_add(1);

    let mut recruit = generate_recruit_for_overworld(seed, id, overworld);
    if recruit.origin_faction_id != faction_id {
        let fallback_region = overworld.regions.first().map(|r| r.id).unwrap_or(0);
        let region_id = overworld
            .regions
            .iter()
            .find(|r| r.owner_faction_id == faction_id)
            .map(|r| r.id)
            .unwrap_or(fallback_region);
        let region = overworld
            .regions
            .iter()
            .find(|r| r.id == region_id)
            .or_else(|| overworld.regions.first());
        let faction_name = overworld
            .factions
            .get(faction_id)
            .map(|f| f.name.as_str())
            .unwrap_or("Unaligned House");
        let (region_name, unrest, control) = region
            .map(|r| (r.name.as_str(), r.unrest, r.control))
            .unwrap_or(("Unknown March", 50.0, 50.0));
        recruit.origin_faction_id = faction_id;
        recruit.origin_region_id = region_id;
        recruit.backstory = backstory_for_recruit(
            &recruit.codename,
            recruit.archetype,
            faction_name,
            region_name,
            unrest,
            control,
        );
    }
    roster.recruit_pool.push(recruit);
}

fn build_pressure_mission_snapshot(
    seed: u64,
    turn: u32,
    slot: usize,
    region: &OverworldRegion,
    owner: &FactionState,
    pressure: f32,
) -> MissionSnapshot {
    let salt = splitmix64(
        seed ^ (turn as u64).wrapping_mul(97) ^ (region.id as u64).wrapping_mul(131) ^ slot as u64,
    );
    let variant = (salt % 5) as usize;
    let labels = [
        "Frontline Clash",
        "Insurgent Network",
        "Supply Collapse",
        "Arcane Breach",
        "Border Uprising",
    ];
    let mode = match variant {
        0 => TacticalMode::Aggressive,
        1 => TacticalMode::Defensive,
        2 => TacticalMode::Balanced,
        3 => TacticalMode::Defensive,
        _ => TacticalMode::Aggressive,
    };
    let reveal = if region.intel_level < 30.0 {
        "Unconfirmed Crisis"
    } else {
        labels[variant]
    };
    let mission_name = format!("{}: {} [{}]", reveal, region.name, owner.name);
    let variance = (rand01(salt, 5) - 0.5) * 8.0;
    let turns_remaining =
        (28.0 - pressure * 0.18 - owner.war_focus * 0.04 + variance).clamp(10.0, 32.0) as u32;
    let alert_level =
        (16.0 + pressure * 0.55 + owner.war_focus * 0.12 + variance * 0.6).clamp(8.0, 96.0);
    let reactor_integrity =
        (100.0 - pressure * 0.45 - owner.war_focus * 0.06 + variance * 0.5).clamp(12.0, 100.0);
    let sabotage_progress = (rand01(salt, 17) * 16.0 + region.intel_level * 0.06).clamp(0.0, 45.0);

    MissionSnapshot {
        mission_name,
        bound_region_id: Some(region.id),
        mission_active: true,
        result: MissionResult::InProgress,
        turns_remaining,
        reactor_integrity,
        sabotage_progress,
        sabotage_goal: 100.0,
        alert_level,
        room_index: (salt % 6) as usize,
        tactical_mode: mode,
        command_cooldown_turns: 0,
        unattended_turns: 0,
        outcome_recorded: false,
    }
}

pub fn pressure_spawn_missions_system(
    run_state: Res<RunState>,
    overworld: Res<OverworldMap>,
    board: Res<MissionBoard>,
    mut mission_query: Query<(&mut MissionData, &mut MissionProgress, &mut MissionTactics)>,
    roster: Option<Res<CampaignRoster>>,
    mut flashpoints: Option<ResMut<FlashpointState>>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0
        || overworld.regions.is_empty()
        || overworld.factions.is_empty()
        || board.entities.is_empty()
    {
        return;
    }

    let max_slot = usize::min(overworld.factions.len(), board.entities.len());
    for slot in 0..max_slot {
        let Some(&entity) = board.entities.get(slot) else {
            continue;
        };
        let Ok((mut data, mut progress, mut tactics)) = mission_query.get_mut(entity) else {
            continue;
        };

        let Some(region) = overworld
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
        else {
            continue;
        };
        let owner_id = region.owner_faction_id.min(overworld.factions.len() - 1);
        let owner = &overworld.factions[owner_id];
        let contested_edges = region
            .neighbors
            .iter()
            .filter(|n| {
                overworld
                    .regions
                    .get(**n)
                    .map(|r| r.owner_faction_id != owner_id)
                    .unwrap_or(false)
            })
            .count() as f32;
        let pressure =
            (region.unrest * 0.64 + (100.0 - region.control) * 0.36 + contested_edges * 10.0)
                .clamp(0.0, 100.0);

        let active_chain = flashpoints.as_ref().and_then(|state| {
            state
                .chains
                .iter()
                .find(|chain| !chain.completed && chain.mission_slot == slot)
                .cloned()
        });
        if let Some(ref chain) = active_chain {
            data.bound_region_id = Some(chain.region_id);
            if progress.result == MissionResult::InProgress && progress.mission_active {
                progress.alert_level = progress
                    .alert_level
                    .max((pressure * (0.38 + chain.stage as f32 * 0.04)).clamp(18.0, 96.0));
                progress.reactor_integrity = progress
                    .reactor_integrity
                    .min((100.0 - pressure * 0.18).clamp(18.0, 100.0));
                continue;
            }
        }
        let needs_replace = data.bound_region_id != Some(region.id)
            || progress.result != MissionResult::InProgress
            || !progress.mission_active
            || (pressure >= 74.0 && progress.alert_level <= pressure * 0.52)
            || (pressure >= 82.0 && progress.turns_remaining > 18);

        let can_start_flashpoint = active_chain.is_none()
            && pressure >= FLASHPOINT_TRIGGER_PRESSURE
            && contested_edges >= 1.0
            && needs_replace;
        if can_start_flashpoint {
            let Some(attacker_id) = pick_flashpoint_attacker(region, &overworld) else {
                continue;
            };
            if let Some(state) = flashpoints.as_mut() {
                let chain_id = state.next_id;
                state.next_id = state.next_id.saturating_add(1);
                // Determine companion hook from assigned hero on this slot's entity.
                let companion_hook_hero_id: Option<u32> = None; // set by sync_mission_assignments
                let chain = FlashpointChain {
                    id: chain_id,
                    mission_slot: slot,
                    region_id: region.id,
                    attacker_faction_id: attacker_id,
                    defender_faction_id: owner_id,
                    stage: 1,
                    completed: false,
                    companion_hook_hero_id,
                    intent: FlashpointIntent::StealthPush,
                    objective: String::new(),
                };
                let mut snap = build_pressure_mission_snapshot(
                    overworld.map_seed,
                    run_state.global_turn,
                    slot,
                    region,
                    owner,
                    pressure,
                );
                configure_flashpoint_stage_mission(
                    &mut snap,
                    &chain,
                    &overworld,
                    overworld.map_seed ^ run_state.global_turn as u64,
                );
                let mut chain = chain;
                if let Some(roster) = roster.as_ref() {
                    let _ = apply_flashpoint_companion_hook(&mut snap, &chain, roster);
                    chain.objective = flashpoint_hook_objective_suffix(&chain, roster)
                        .unwrap_or_else(|| "No companion hook".to_string());
                }
                apply_flashpoint_intent(&mut snap, &chain);
                rewrite_flashpoint_mission_name(&mut snap, &chain, &overworld, roster.as_deref());
                // Write snapshot back to components.
                data.bound_region_id = snap.bound_region_id;
                data.mission_name = snap.mission_name;
                progress.mission_active = snap.mission_active;
                progress.result = snap.result;
                progress.turns_remaining = snap.turns_remaining;
                progress.reactor_integrity = snap.reactor_integrity;
                progress.sabotage_progress = snap.sabotage_progress;
                progress.sabotage_goal = snap.sabotage_goal;
                progress.alert_level = snap.alert_level;
                progress.room_index = snap.room_index;
                progress.unattended_turns = snap.unattended_turns;
                progress.outcome_recorded = snap.outcome_recorded;
                tactics.tactical_mode = snap.tactical_mode;
                tactics.command_cooldown_turns = snap.command_cooldown_turns;
                let attacker_name = overworld
                    .factions
                    .get(attacker_id)
                    .map(|f| f.name.as_str())
                    .unwrap_or("Rivals");
                state.notice = format!(
                    "Flashpoint opened in {}: {} pushes against {}.",
                    region.name, attacker_name, owner.name
                );
                state.chains.push(chain);
                if let Some(log) = event_log.as_mut() {
                    push_campaign_event(log, run_state.global_turn, state.notice.clone());
                }
                continue;
            }
        }

        if needs_replace {
            let snap = build_pressure_mission_snapshot(
                overworld.map_seed,
                run_state.global_turn,
                slot,
                region,
                owner,
                pressure,
            );
            progress.mission_active = snap.mission_active;
            progress.result = snap.result;
            progress.turns_remaining = snap.turns_remaining;
            progress.reactor_integrity = snap.reactor_integrity;
            progress.sabotage_progress = snap.sabotage_progress;
            progress.sabotage_goal = snap.sabotage_goal;
            progress.alert_level = snap.alert_level;
            progress.room_index = snap.room_index;
            progress.unattended_turns = snap.unattended_turns;
            progress.outcome_recorded = snap.outcome_recorded;
            tactics.tactical_mode = snap.tactical_mode;
            tactics.command_cooldown_turns = snap.command_cooldown_turns;
            data.bound_region_id = snap.bound_region_id;
            data.mission_name = snap.mission_name;
            continue;
        }

        data.bound_region_id = Some(region.id);
        progress.alert_level = progress.alert_level.max((pressure * 0.38).clamp(8.0, 92.0));
        progress.reactor_integrity = progress
            .reactor_integrity
            .min((100.0 - pressure * 0.2).clamp(20.0, 100.0));
    }
}

pub fn update_faction_war_goals_system(
    diplomacy: Res<DiplomacyState>,
    mut overworld: ResMut<OverworldMap>,
) {
    let n = overworld.factions.len();
    if n < 2 {
        return;
    }
    let mut border_counts = vec![vec![0_u32; n]; n];
    for region in &overworld.regions {
        for neighbor in &region.neighbors {
            if let Some(other) = overworld.regions.get(*neighbor) {
                let a = region.owner_faction_id;
                let b = other.owner_faction_id;
                if a < n && b < n && a != b {
                    border_counts[a][b] = border_counts[a][b].saturating_add(1);
                }
            }
        }
    }

    for faction_id in 0..n {
        let mut best_target = None;
        let mut best_score = f32::MIN;
        for target in 0..n {
            if target == faction_id {
                continue;
            }
            let relation = diplomacy
                .relations
                .get(faction_id)
                .and_then(|row| row.get(target))
                .copied()
                .unwrap_or(0);
            let hostility = (-relation) as f32;
            let border = border_counts[faction_id][target] as f32 * 6.0;
            let target_strength = overworld
                .factions
                .get(target)
                .map(|f| f.strength)
                .unwrap_or(50.0);
            let own_strength = overworld.factions[faction_id].strength;
            let strength_tension = ((target_strength - own_strength) * 0.06).abs();
            let score = hostility + border + strength_tension;
            if score > best_score {
                best_score = score;
                best_target = Some(target);
            }
        }

        let cohesion = overworld.factions[faction_id].cohesion;
        let strategic_drive = (best_score + (100.0 - cohesion) * 0.22).clamp(0.0, 100.0);
        overworld.factions[faction_id].war_goal_faction_id = best_target;
        overworld.factions[faction_id].war_focus = strategic_drive;
    }
}

pub fn overworld_ai_border_pressure_system(
    run_state: Res<RunState>,
    mut overworld: ResMut<OverworldMap>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0 || overworld.regions.is_empty() || overworld.factions.len() < 2 {
        return;
    }

    let n_regions = overworld.regions.len();
    let mut control_delta = vec![0.0_f32; n_regions];
    let mut unrest_delta = vec![0.0_f32; n_regions];
    for rid in 0..n_regions {
        let owner_a = overworld.regions[rid].owner_faction_id;
        for neighbor in overworld.regions[rid].neighbors.clone() {
            if rid >= neighbor {
                continue;
            }
            let owner_b = overworld.regions[neighbor].owner_faction_id;
            if owner_a == owner_b {
                continue;
            }
            let Some(fa) = overworld.factions.get(owner_a) else {
                continue;
            };
            let Some(fb) = overworld.factions.get(owner_b) else {
                continue;
            };
            let roam_a = fa
                .vassals
                .iter()
                .filter(|v| v.post == VassalPost::Roaming)
                .map(|v| v.martial * 0.012)
                .sum::<f32>();
            let roam_b = fb
                .vassals
                .iter()
                .filter(|v| v.post == VassalPost::Roaming)
                .map(|v| v.martial * 0.012)
                .sum::<f32>();
            let power_a = fa.strength * 0.66 + fa.cohesion * 0.34 + roam_a + fa.war_focus * 0.18;
            let power_b = fb.strength * 0.66 + fb.cohesion * 0.34 + roam_b + fb.war_focus * 0.18;
            let jitter = (rand01(
                overworld.map_seed ^ run_state.global_turn as u64,
                60_000 + rid as u64 * 131 + neighbor as u64 * 17,
            ) - 0.5)
                * 1.5;
            let pressure = ((power_a - power_b) * 0.02 + jitter).clamp(-6.5, 6.5);
            if pressure > 0.0 {
                control_delta[rid] += pressure * 0.45;
                unrest_delta[rid] -= pressure * 0.30;
                control_delta[neighbor] -= pressure * 0.64;
                unrest_delta[neighbor] += pressure * 0.72;
            } else if pressure < 0.0 {
                let p = -pressure;
                control_delta[rid] -= p * 0.64;
                unrest_delta[rid] += p * 0.72;
                control_delta[neighbor] += p * 0.45;
                unrest_delta[neighbor] -= p * 0.30;
            }
        }
    }

    for rid in 0..n_regions {
        let region = &mut overworld.regions[rid];
        region.control = (region.control + control_delta[rid]).clamp(0.0, 100.0);
        region.unrest = (region.unrest + unrest_delta[rid]).clamp(0.0, 100.0);
    }

    let mut owned_counts = vec![0_u32; overworld.factions.len()];
    for region in &overworld.regions {
        if region.owner_faction_id < owned_counts.len() {
            owned_counts[region.owner_faction_id] =
                owned_counts[region.owner_faction_id].saturating_add(1);
        }
    }
    let mut changed_ownership = false;
    let mut pending_events = Vec::new();
    for rid in 0..n_regions {
        let owner = overworld.regions[rid].owner_faction_id;
        if overworld.regions[rid].control >= 22.0 || overworld.regions[rid].unrest <= 60.0 {
            continue;
        }
        if owner >= owned_counts.len() || owned_counts[owner] <= 1 {
            continue;
        }
        let mut contender = owner;
        let mut contender_score = f32::MIN;
        for neighbor in &overworld.regions[rid].neighbors {
            let maybe_owner = overworld
                .regions
                .get(*neighbor)
                .map(|r| r.owner_faction_id)
                .unwrap_or(owner);
            if maybe_owner == owner {
                continue;
            }
            let score = overworld
                .factions
                .get(maybe_owner)
                .map(|f| f.strength + f.war_focus * 0.4 + f.cohesion * 0.2)
                .unwrap_or(0.0);
            if score > contender_score {
                contender_score = score;
                contender = maybe_owner;
            }
        }
        if contender != owner {
            let owner_power = overworld
                .factions
                .get(owner)
                .map(|f| f.strength + f.cohesion * 0.2)
                .unwrap_or(0.0);
            if contender_score > owner_power + 8.0 {
                let region_name = overworld.regions[rid].name.clone();
                let old_owner = overworld
                    .factions
                    .get(owner)
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| format!("Faction {}", owner));
                let new_owner = overworld
                    .factions
                    .get(contender)
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| format!("Faction {}", contender));
                overworld.regions[rid].owner_faction_id = contender;
                overworld.regions[rid].control = 52.0;
                overworld.regions[rid].unrest = 48.0;
                if owner < owned_counts.len() {
                    owned_counts[owner] = owned_counts[owner].saturating_sub(1);
                }
                if contender < owned_counts.len() {
                    owned_counts[contender] = owned_counts[contender].saturating_add(1);
                }
                changed_ownership = true;
                pending_events.push(format!(
                    "Border shift: {} captured {} from {}.",
                    new_owner, region_name, old_owner
                ));
            }
        }
    }
    if changed_ownership {
        ensure_faction_mission_slots(&mut overworld);
    }
    if let Some(log) = event_log.as_mut() {
        for event in pending_events {
            push_campaign_event(log, run_state.global_turn, event);
        }
    }
}

pub fn overworld_intel_update_system(
    run_state: Res<RunState>,
    board: Res<MissionBoard>,
    mission_query: Query<(&MissionProgress,)>,
    mut overworld: ResMut<OverworldMap>,
) {
    if run_state.global_turn == 0 || overworld.regions.is_empty() {
        return;
    }
    let current = overworld.current_region.min(overworld.regions.len() - 1);
    let selected = overworld.selected_region.min(overworld.regions.len() - 1);

    // Build a slot->pressure lookup from mission entities.
    let slot_pressure: Vec<Option<(f32, f32)>> = board
        .entities
        .iter()
        .map(|e| {
            mission_query
                .get(*e)
                .ok()
                .map(|(p,)| (p.alert_level, p.reactor_integrity))
        })
        .collect();

    for rid in 0..overworld.regions.len() {
        let region = &mut overworld.regions[rid];
        let mission_bonus = region
            .mission_slot
            .and_then(|slot| slot_pressure.get(slot))
            .and_then(|v| *v)
            .map(|(alert, integrity)| alert * 0.015 + (100.0 - integrity) * 0.01)
            .unwrap_or(0.0);
        let decay = 0.8 + region.unrest * 0.012;
        region.intel_level = (region.intel_level - decay + mission_bonus).clamp(0.0, 100.0);
        if region.owner_faction_id == 0 {
            region.intel_level = region.intel_level.max(30.0);
        }
    }

    let neighbors = overworld.regions[current].neighbors.clone();
    if let Some(region) = overworld.regions.get_mut(current) {
        region.intel_level = (region.intel_level + 16.0).clamp(0.0, 100.0);
    }
    for neighbor in neighbors {
        if let Some(region) = overworld.regions.get_mut(neighbor) {
            region.intel_level = (region.intel_level + 8.0).clamp(0.0, 100.0);
        }
    }
    if let Some(region) = overworld.regions.get_mut(selected) {
        region.intel_level = (region.intel_level + 5.0).clamp(0.0, 100.0);
    }
}

pub fn overworld_sync_from_missions_system(
    board: Res<MissionBoard>,
    mission_query: Query<(&MissionProgress,)>,
    mut overworld: ResMut<OverworldMap>,
) {
    let mut faction_bonus = vec![0.0_f32; overworld.factions.len()];
    let mut region_manager_bonus = vec![0.0_f32; overworld.regions.len()];
    for faction in &overworld.factions {
        let patrol_weight = faction
            .vassals
            .iter()
            .filter(|v| {
                v.post == VassalPost::Roaming
                    && (v.specialty == VassalSpecialty::Patrol
                        || v.specialty == VassalSpecialty::Escort)
            })
            .count() as f32;
        let roaming_quality = faction
            .vassals
            .iter()
            .filter(|v| v.post == VassalPost::Roaming)
            .map(|v| v.martial * 0.008 + v.loyalty * 0.004)
            .sum::<f32>();
        for v in faction
            .vassals
            .iter()
            .filter(|v| v.post == VassalPost::ZoneManager)
        {
            if v.home_region_id < region_manager_bonus.len() {
                region_manager_bonus[v.home_region_id] += v.martial * 0.012 + v.loyalty * 0.006;
            }
        }
        faction_bonus[faction.id] =
            patrol_weight * 0.22 + faction.cohesion * 0.01 + roaming_quality;
    }

    // Build slot->MissionProgress snapshot.
    let slot_progress: Vec<Option<MissionProgress>> = board
        .entities
        .iter()
        .map(|e| mission_query.get(*e).ok().map(|(p,)| p.clone()))
        .collect();

    for region in &mut overworld.regions {
        let Some(slot) = region.mission_slot else {
            let bonus = faction_bonus
                .get(region.owner_faction_id)
                .copied()
                .unwrap_or(0.0);
            region.unrest = (region.unrest + 0.2 - bonus * 0.08).clamp(0.0, 100.0);
            region.control = (100.0 - region.unrest).clamp(0.0, 100.0);
            continue;
        };
        let Some(Some(mission)) = slot_progress.get(slot) else {
            continue;
        };
        let bonus = faction_bonus
            .get(region.owner_faction_id)
            .copied()
            .unwrap_or(0.0);
        let manager_bonus = region_manager_bonus.get(region.id).copied().unwrap_or(0.0);
        let pressure = (mission.alert_level + (100.0 - mission.reactor_integrity)) * 0.5;
        let progress_relief = mission.sabotage_progress * 0.08;
        let mut unrest = (pressure * 0.6 - progress_relief - bonus * 0.22 - manager_bonus * 0.2)
            .clamp(0.0, 100.0);
        if mission.result == MissionResult::Victory {
            unrest = (unrest - 12.0).max(0.0);
        } else if mission.result == MissionResult::Defeat {
            unrest = (unrest + 12.0).min(100.0);
        }
        region.unrest = unrest;
        region.control = (100.0 - unrest).clamp(0.0, 100.0);
    }
}

pub fn overworld_faction_autonomy_system(
    run_state: Res<RunState>,
    mut overworld: ResMut<OverworldMap>,
) {
    if run_state.global_turn == 0 || overworld.factions.is_empty() {
        return;
    }

    let n = overworld.factions.len();
    let mut unrest_sum = vec![0.0_f32; n];
    let mut control_sum = vec![0.0_f32; n];
    let mut counts = vec![0_u32; n];
    let mut owned_by_faction = vec![Vec::<usize>::new(); n];
    for region in &overworld.regions {
        if region.owner_faction_id < n {
            unrest_sum[region.owner_faction_id] += region.unrest;
            control_sum[region.owner_faction_id] += region.control;
            counts[region.owner_faction_id] += 1;
            owned_by_faction[region.owner_faction_id].push(region.id);
        }
    }

    let mut next_vassal_id = overworld.next_vassal_id;
    for faction in &mut overworld.factions {
        let c = counts[faction.id].max(1) as f32;
        let avg_unrest = unrest_sum[faction.id] / c;
        let avg_control = control_sum[faction.id] / c;
        faction.cohesion =
            (faction.cohesion + (avg_control - avg_unrest) * 0.015).clamp(10.0, 95.0);
        faction.strength =
            (faction.strength + (avg_control * 0.03) - (avg_unrest * 0.025)).clamp(30.0, 180.0);
        let owned = owned_by_faction[faction.id].clone();
        rebalance_faction_vassals(faction, &owned, &mut next_vassal_id);
    }
    overworld.next_vassal_id = next_vassal_id;
}

pub fn overworld_cycle_selection(overworld: &mut OverworldMap, forward: bool) -> bool {
    let current = overworld
        .current_region
        .min(overworld.regions.len().saturating_sub(1));
    let Some(region) = overworld.regions.get(current) else {
        return false;
    };
    if region.neighbors.is_empty() {
        return false;
    }

    if !region.neighbors.contains(&overworld.selected_region) {
        overworld.selected_region = region.neighbors[0];
        return true;
    }

    let pos = region
        .neighbors
        .iter()
        .position(|id| *id == overworld.selected_region)
        .unwrap_or(0);
    let next = if forward {
        (pos + 1) % region.neighbors.len()
    } else if pos == 0 {
        region.neighbors.len() - 1
    } else {
        pos - 1
    };
    overworld.selected_region = region.neighbors[next];
    true
}

/// Returns `Some(slot_index)` if travel succeeds and the destination region has a mission slot.
pub fn try_overworld_travel(
    overworld: &mut OverworldMap,
    attention: &mut AttentionState,
) -> Option<usize> {
    if overworld.travel_cooldown_turns > 0 || attention.global_energy < overworld.travel_cost {
        return None;
    }
    let current = overworld
        .current_region
        .min(overworld.regions.len().saturating_sub(1));
    let target = overworld
        .selected_region
        .min(overworld.regions.len().saturating_sub(1));
    if current == target {
        return None;
    }
    let Some(region) = overworld.regions.get(current) else {
        return None;
    };
    if !region.neighbors.contains(&target) {
        return None;
    }

    overworld.current_region = target;
    attention.global_energy = (attention.global_energy - overworld.travel_cost).max(0.0);
    overworld.travel_cooldown_turns = overworld.travel_cooldown_max;
    let neighbors = overworld.regions[target].neighbors.clone();
    if let Some(region) = overworld.regions.get_mut(target) {
        region.intel_level = (region.intel_level + 24.0).clamp(0.0, 100.0);
    }
    for neighbor in neighbors {
        if let Some(region) = overworld.regions.get_mut(neighbor) {
            region.intel_level = (region.intel_level + 10.0).clamp(0.0, 100.0);
        }
    }
    overworld.regions[target].mission_slot
}

pub fn overworld_hub_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut commands: Commands,
    mut overworld: ResMut<OverworldMap>,
    board: Res<MissionBoard>,
    mut attention: ResMut<AttentionState>,
    active_query: Query<Entity, With<ActiveMission>>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };
    if keyboard.just_pressed(KeyCode::KeyJ) {
        let _ = overworld_cycle_selection(&mut overworld, false);
    }
    if keyboard.just_pressed(KeyCode::KeyL) {
        let _ = overworld_cycle_selection(&mut overworld, true);
    }
    if keyboard.just_pressed(KeyCode::KeyT) {
        if let Some(slot) = try_overworld_travel(&mut overworld, &mut attention) {
            if let Some(&new_entity) = board.entities.get(slot.min(board.entities.len().saturating_sub(1))) {
                for old in active_query.iter() {
                    commands.entity(old).remove::<ActiveMission>();
                }
                commands.entity(new_entity).insert(ActiveMission);
            }
        }
    }
}

pub fn flashpoint_intent_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    run_state: Res<RunState>,
    overworld: Res<OverworldMap>,
    board: Res<MissionBoard>,
    mut mission_query: Query<(&mut MissionData, &mut MissionProgress, &mut MissionTactics)>,
    active_query: Query<Entity, With<ActiveMission>>,
    mut flashpoints: ResMut<FlashpointState>,
    roster: Option<Res<CampaignRoster>>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };
    let intent = if keyboard.just_pressed(KeyCode::Digit1) {
        Some(FlashpointIntent::StealthPush)
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        Some(FlashpointIntent::DirectAssault)
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        Some(FlashpointIntent::CivilianFirst)
    } else {
        None
    };
    let Some(intent) = intent else {
        return;
    };

    // Determine the active mission's slot index.
    let Ok(active_entity) = active_query.get_single() else {
        return;
    };
    let slot = board.entities.iter().position(|&e| e == active_entity).unwrap_or(0);

    let Some(chain_idx) = flashpoints
        .chains
        .iter()
        .position(|c| !c.completed && c.mission_slot == slot)
    else {
        return;
    };
    let mut chain = flashpoints.chains[chain_idx].clone();
    if chain.intent == intent {
        flashpoints.notice = format!("Flashpoint intent unchanged: {}.", flashpoint_intent_label(intent));
        return;
    }
    chain.intent = intent;

    // Apply changes via snapshot round-trip.
    if let Some(&entity) = board.entities.get(slot) {
        if let Ok((mut data, mut progress, mut tactics)) = mission_query.get_mut(entity) {
            let mut snap = MissionSnapshot::from_components(&data, &progress, &tactics);
            configure_flashpoint_stage_mission(
                &mut snap,
                &chain,
                &overworld,
                overworld.map_seed ^ (run_state.global_turn as u64).wrapping_mul(733),
            );
            if let Some(roster) = roster.as_ref() {
                let _ = apply_flashpoint_companion_hook(&mut snap, &chain, roster);
                chain.objective = flashpoint_hook_objective_suffix(&chain, roster)
                    .unwrap_or_else(|| "No companion hook".to_string());
            } else {
                chain.objective = "No companion hook".to_string();
            }
            apply_flashpoint_intent(&mut snap, &chain);
            rewrite_flashpoint_mission_name(&mut snap, &chain, &overworld, roster.as_deref());
            // Write back.
            progress.mission_active = snap.mission_active;
            progress.result = snap.result;
            progress.turns_remaining = snap.turns_remaining;
            progress.reactor_integrity = snap.reactor_integrity;
            progress.sabotage_progress = snap.sabotage_progress;
            progress.sabotage_goal = snap.sabotage_goal;
            progress.alert_level = snap.alert_level;
            progress.room_index = snap.room_index;
            progress.unattended_turns = snap.unattended_turns;
            progress.outcome_recorded = snap.outcome_recorded;
            tactics.tactical_mode = snap.tactical_mode;
            tactics.command_cooldown_turns = snap.command_cooldown_turns;
            data.mission_name = snap.mission_name.clone();
            data.bound_region_id = snap.bound_region_id;
        }
    }

    flashpoints.notice = format!(
        "Flashpoint intent set to {} on slot {} (keys: 1 stealth, 2 assault, 3 civilian).",
        flashpoint_intent_label(intent),
        slot + 1
    );
    if let Some(log) = event_log.as_mut() {
        push_campaign_event(log, run_state.global_turn, flashpoints.notice.clone());
    }
    flashpoints.chains[chain_idx] = chain;
}

fn relation_to_player(diplomacy: &DiplomacyState, faction_id: usize) -> i32 {
    if faction_id >= diplomacy.relations.len()
        || diplomacy.player_faction_id >= diplomacy.relations.len()
    {
        return 0;
    }
    diplomacy.relations[faction_id][diplomacy.player_faction_id]
}

fn commander_primary_region(
    faction_id: usize,
    overworld: &OverworldMap,
) -> Option<&OverworldRegion> {
    overworld
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == faction_id)
        .max_by(|a, b| a.unrest.total_cmp(&b.unrest).then(a.id.cmp(&b.id)))
}

pub fn generate_commander_intents_system(
    overworld: Res<OverworldMap>,
    board: Res<MissionBoard>,
    mission_query: Query<(&MissionProgress,)>,
    diplomacy: Res<DiplomacyState>,
    mut commanders: ResMut<CommanderState>,
) {
    if commanders.commanders.is_empty() || overworld.regions.is_empty() {
        return;
    }

    // Build slot->pressure lookup.
    let slot_pressure: Vec<Option<f32>> = board
        .entities
        .iter()
        .map(|e| {
            mission_query
                .get(*e)
                .ok()
                .map(|(p,)| p.alert_level + (100.0 - p.reactor_integrity))
        })
        .collect();

    let mut intents = Vec::with_capacity(commanders.commanders.len());
    for commander in &commanders.commanders {
        let Some(region) = commander_primary_region(commander.faction_id, &overworld) else {
            continue;
        };
        let faction_state = overworld.factions.get(commander.faction_id);
        let relation = relation_to_player(&diplomacy, commander.faction_id);
        let mission_slot = region.mission_slot;
        let mission_pressure = mission_slot
            .and_then(|s| slot_pressure.get(s).and_then(|v| *v))
            .unwrap_or(region.unrest);
        let war_focus = faction_state.map(|f| f.war_focus).unwrap_or(0.0);
        let war_target_player = faction_state
            .and_then(|f| f.war_goal_faction_id)
            .map(|id| id == diplomacy.player_faction_id)
            .unwrap_or(false);
        let urgency =
            (mission_pressure * 0.5 + region.unrest * 0.5 + war_focus * 0.15).clamp(0.0, 100.0);

        let kind = if relation >= 20 && commander.cooperation_bias >= 0.55 && urgency >= 30.0 {
            CommanderIntentKind::JointMission
        } else if (commander.aggression >= 0.65 || war_target_player)
            && relation <= 8
            && urgency >= 26.0
        {
            CommanderIntentKind::Raid
        } else if commander.cooperation_bias >= 0.6
            && relation >= 12
            && commander.competence >= 0.68
        {
            CommanderIntentKind::TrainingExchange
        } else if relation >= 18 && commander.competence >= 0.7 {
            CommanderIntentKind::RecruitBorrow
        } else {
            CommanderIntentKind::StabilizeBorder
        };

        intents.push(CommanderIntent {
            faction_id: commander.faction_id,
            region_id: region.id,
            mission_slot,
            urgency,
            kind,
        });
    }
    commanders.intents = intents;
}

fn offer_kind_for_intent(kind: CommanderIntentKind) -> Option<InteractionOfferKind> {
    match kind {
        CommanderIntentKind::JointMission => Some(InteractionOfferKind::JointMission),
        CommanderIntentKind::Raid => Some(InteractionOfferKind::RivalRaid),
        CommanderIntentKind::TrainingExchange => Some(InteractionOfferKind::TrainingLoan),
        CommanderIntentKind::RecruitBorrow => Some(InteractionOfferKind::RecruitBorrow),
        CommanderIntentKind::StabilizeBorder => None,
    }
}

pub fn refresh_interaction_offers_system(
    commanders: Res<CommanderState>,
    overworld: Res<OverworldMap>,
    diplomacy: Res<DiplomacyState>,
    mut board: ResMut<InteractionBoard>,
) {
    let mut offers = Vec::new();
    let mut next_id = board.next_offer_id;
    for intent in &commanders.intents {
        if intent.faction_id == diplomacy.player_faction_id {
            continue;
        }
        let Some(kind) = offer_kind_for_intent(intent.kind) else {
            continue;
        };
        let relation = relation_to_player(&diplomacy, intent.faction_id);
        if matches!(
            kind,
            InteractionOfferKind::JointMission
                | InteractionOfferKind::TrainingLoan
                | InteractionOfferKind::RecruitBorrow
        ) && relation < 8
        {
            continue;
        }
        let faction_name = overworld
            .factions
            .get(intent.faction_id)
            .map(|f| f.name.as_str())
            .unwrap_or("Unknown Faction");
        let region_name = overworld
            .regions
            .iter()
            .find(|r| r.id == intent.region_id)
            .map(|r| r.name.as_str())
            .unwrap_or("Unknown Region");
        let summary = match kind {
            InteractionOfferKind::JointMission => {
                format!("{faction_name} proposes a joint strike in {region_name}.")
            }
            InteractionOfferKind::RivalRaid => {
                format!("{faction_name} is preparing a rival raid near {region_name}.")
            }
            InteractionOfferKind::TrainingLoan => {
                format!("{faction_name} offers cross-faction training in {region_name}.")
            }
            InteractionOfferKind::RecruitBorrow => {
                format!("{faction_name} proposes a recruit exchange linked to {region_name}.")
            }
        };
        offers.push(InteractionOffer {
            id: next_id,
            from_faction_id: intent.faction_id,
            region_id: intent.region_id,
            mission_slot: intent.mission_slot,
            kind,
            summary,
        });
        next_id = next_id.saturating_add(1);
    }
    board.offers = offers;
    if board.selected >= board.offers.len() {
        board.selected = board.offers.len().saturating_sub(1);
    }
    board.next_offer_id = next_id;
}

fn adjust_relation(diplomacy: &mut DiplomacyState, a: usize, b: usize, delta: i32) {
    if a >= diplomacy.relations.len() || b >= diplomacy.relations.len() {
        return;
    }
    diplomacy.relations[a][b] = (diplomacy.relations[a][b] + delta).clamp(-100, 100);
    diplomacy.relations[b][a] = (diplomacy.relations[b][a] + delta).clamp(-100, 100);
}

pub fn resolve_interaction_offer(
    offer: &InteractionOffer,
    accepted: bool,
    mission_snapshots: &mut [MissionSnapshot],
    attention: &mut AttentionState,
    roster: &mut CampaignRoster,
    diplomacy: &mut DiplomacyState,
) -> String {
    if !accepted {
        adjust_relation(
            diplomacy,
            offer.from_faction_id,
            diplomacy.player_faction_id,
            -1,
        );
        return format!(
            "Declined offer #{} from faction {}.",
            offer.id, offer.from_faction_id
        );
    }

    match offer.kind {
        InteractionOfferKind::JointMission => {
            if let Some(slot) = offer
                .mission_slot
                .and_then(|s| mission_snapshots.get_mut(s))
            {
                slot.alert_level = (slot.alert_level - 6.0).max(0.0);
                slot.turns_remaining = (slot.turns_remaining + 2).min(45);
            }
            adjust_relation(
                diplomacy,
                offer.from_faction_id,
                diplomacy.player_faction_id,
                4,
            );
            "Joint mission accepted: pressure reduced and timeline extended.".to_string()
        }
        InteractionOfferKind::RivalRaid => {
            if let Some(slot) = offer
                .mission_slot
                .and_then(|s| mission_snapshots.get_mut(s))
            {
                slot.sabotage_progress = (slot.sabotage_progress + 5.0).min(slot.sabotage_goal);
                slot.alert_level = (slot.alert_level + 4.0).min(100.0);
            }
            attention.global_energy = (attention.global_energy - 4.0).max(0.0);
            adjust_relation(
                diplomacy,
                offer.from_faction_id,
                diplomacy.player_faction_id,
                -6,
            );
            "Rival raid embraced: objective gain at diplomatic and alert cost.".to_string()
        }
        InteractionOfferKind::TrainingLoan => {
            if let Some(hero) = roster.heroes.iter_mut().find(|h| h.active && !h.deserter) {
                hero.stress = (hero.stress - 8.0).max(0.0);
                hero.fatigue = (hero.fatigue - 7.0).max(0.0);
                hero.resolve = (hero.resolve + 2.0).min(100.0);
            }
            attention.global_energy = (attention.global_energy - 6.0).max(0.0);
            adjust_relation(
                diplomacy,
                offer.from_faction_id,
                diplomacy.player_faction_id,
                3,
            );
            "Training loan accepted: companion readiness improved.".to_string()
        }
        InteractionOfferKind::RecruitBorrow => {
            let signed = sign_top_recruit(roster);
            if let Some(hero) = signed {
                if let Some(last) = roster.heroes.iter_mut().find(|h| h.id == hero.id) {
                    last.loyalty = last.loyalty.min(58.0);
                }
            }
            attention.global_energy = (attention.global_energy - 7.0).max(0.0);
            adjust_relation(
                diplomacy,
                offer.from_faction_id,
                diplomacy.player_faction_id,
                2,
            );
            "Recruit borrowing accepted: a provisional companion joins your roster.".to_string()
        }
    }
}

pub fn interaction_offer_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut board: ResMut<InteractionBoard>,
    mission_board: Res<MissionBoard>,
    mut mission_query: Query<(&MissionData, &mut MissionProgress, &mut MissionTactics)>,
    mut attention: ResMut<AttentionState>,
    mut roster: ResMut<CampaignRoster>,
    mut diplomacy: ResMut<DiplomacyState>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };
    if keyboard.just_pressed(KeyCode::KeyO) && !board.offers.is_empty() {
        board.selected = (board.selected + 1) % board.offers.len();
    }
    if keyboard.just_pressed(KeyCode::KeyU) && !board.offers.is_empty() {
        board.selected = if board.selected == 0 {
            board.offers.len() - 1
        } else {
            board.selected - 1
        };
    }
    if board.offers.is_empty() {
        return;
    }

    if keyboard.just_pressed(KeyCode::KeyY) || keyboard.just_pressed(KeyCode::KeyN) {
        let accepted = keyboard.just_pressed(KeyCode::KeyY);
        let selected = board.selected;
        let offer = board.offers[selected].clone();

        // Build snapshots for mutation.
        let mut snapshots: Vec<MissionSnapshot> = mission_board
            .entities
            .iter()
            .filter_map(|e| {
                mission_query.get(*e).ok().map(|(d, p, t)| MissionSnapshot::from_components(d, &p, &t))
            })
            .collect();

        board.notice = resolve_interaction_offer(
            &offer,
            accepted,
            &mut snapshots,
            &mut attention,
            &mut roster,
            &mut diplomacy,
        );

        // Write snapshots back to components.
        for (slot, &entity) in mission_board.entities.iter().enumerate() {
            if let Some(snap) = snapshots.get(slot) {
                if let Ok((_, mut progress, mut tactics)) = mission_query.get_mut(entity) {
                    progress.alert_level = snap.alert_level;
                    progress.turns_remaining = snap.turns_remaining;
                    progress.sabotage_progress = snap.sabotage_progress;
                    progress.reactor_integrity = snap.reactor_integrity;
                    tactics.tactical_mode = snap.tactical_mode;
                    tactics.command_cooldown_turns = snap.command_cooldown_turns;
                }
            }
        }

        board.offers.remove(selected);
        if board.selected >= board.offers.len() {
            board.selected = board.offers.len().saturating_sub(1);
        }
    }
}

/// Returns the new focus index if the shift is permitted, `None` otherwise.
pub fn try_shift_focus(
    entity_count: usize,
    attention: &mut AttentionState,
    current_idx: usize,
    delta: i32,
) -> Option<usize> {
    if delta == 0 || entity_count < 2 {
        return None;
    }
    if attention.switch_cooldown_turns > 0 || attention.global_energy < attention.switch_cost {
        return None;
    }

    let len = entity_count as i32;
    let current = (current_idx as i32).clamp(0, len - 1);
    let next = (current + delta).rem_euclid(len) as usize;
    if next == current_idx {
        return None;
    }

    attention.switch_cooldown_turns = attention.switch_cooldown_max;
    attention.global_energy = (attention.global_energy - attention.switch_cost).max(0.0);
    Some(next)
}

pub fn focus_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut commands: Commands,
    board: Res<MissionBoard>,
    active_query: Query<(Entity, &MissionData), With<ActiveMission>>,
    mut attention: ResMut<AttentionState>,
) {
    let Some(keyboard) = keyboard_input else {
        return;
    };

    let mut delta = 0_i32;
    if keyboard.just_pressed(KeyCode::Tab) || keyboard.just_pressed(KeyCode::BracketRight) {
        delta = 1;
    } else if keyboard.just_pressed(KeyCode::BracketLeft) {
        delta = -1;
    }
    if delta == 0 {
        return;
    }

    let Ok((active_entity, active_data)) = active_query.get_single() else {
        return;
    };
    let current_idx = board
        .entities
        .iter()
        .position(|&e| e == active_entity)
        .unwrap_or(0);

    if let Some(next_idx) = try_shift_focus(board.entities.len(), &mut attention, current_idx, delta) {
        if let Some(&next_entity) = board.entities.get(next_idx) {
            commands.entity(active_entity).remove::<ActiveMission>();
            commands.entity(next_entity).insert(ActiveMission);
            println!("Attention shifted to mission entity slot {}.", next_idx);
        }
    } else {
        println!(
            "Focus shift blocked (cooldown: {}, energy: {:.1}/{:.1}).",
            attention.switch_cooldown_turns, attention.global_energy, attention.max_energy
        );
        let _ = active_data; // suppress unused warning
    }
}

pub fn focused_attention_intervention_system(
    run_state: Res<RunState>,
    mut active_query: Query<(&mut MissionProgress, &MissionTactics), With<ActiveMission>>,
    mut attention: ResMut<AttentionState>,
) {
    if run_state.global_turn == 0 {
        return;
    }
    let Ok((mut progress, tactics)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    let base_gain = match tactics.tactical_mode {
        TacticalMode::Balanced => 4.8,
        TacticalMode::Aggressive => 6.0,
        TacticalMode::Defensive => 3.6,
    };
    let energy_spend = 4.0_f32.min(attention.global_energy.max(0.0));
    let leverage = (energy_spend / 4.0).clamp(0.0, 1.0);
    attention.global_energy = (attention.global_energy - energy_spend).max(0.0);

    if leverage <= 0.0 {
        return;
    }

    progress.sabotage_progress =
        (progress.sabotage_progress + base_gain * leverage).min(progress.sabotage_goal);
    let alert_relief = match tactics.tactical_mode {
        TacticalMode::Balanced => 0.9,
        TacticalMode::Aggressive => 0.4,
        TacticalMode::Defensive => 1.3,
    };
    progress.alert_level = (progress.alert_level - alert_relief * leverage).max(0.0);
}

pub fn simulate_unfocused_missions_system(
    run_state: Res<RunState>,
    mut unfocused_query: Query<(&mut MissionProgress, &MissionTactics), Without<ActiveMission>>,
    mut active_query: Query<&mut MissionProgress, With<ActiveMission>>,
) {
    if run_state.global_turn == 0 {
        return;
    }

    // Reset unattended counter on the active mission.
    for mut progress in active_query.iter_mut() {
        progress.unattended_turns = 0;
    }

    for (mut mission, tactics) in unfocused_query.iter_mut() {
        if !mission.mission_active || mission.result != MissionResult::InProgress {
            continue;
        }

        mission.unattended_turns = mission.unattended_turns.saturating_add(1);
        if mission.turns_remaining > 0 {
            mission.turns_remaining -= 1;
        }

        let base_progress = match tactics.tactical_mode {
            TacticalMode::Balanced => 6.0,
            TacticalMode::Aggressive => 8.0,
            TacticalMode::Defensive => 4.0,
        };
        let unattended_factor = 1.0 + (mission.unattended_turns as f32 * 0.08).min(0.96);
        let pressure_tax = (0.8 + mission.alert_level * 0.02) * unattended_factor;
        mission.sabotage_progress = (mission.sabotage_progress + base_progress - pressure_tax)
            .clamp(0.0, mission.sabotage_goal);
        mission.alert_level = (mission.alert_level + 1.6 * unattended_factor).min(100.0);
        mission.reactor_integrity = (mission.reactor_integrity
            - ((0.6 + mission.alert_level * 0.015) * unattended_factor))
            .max(0.0);

        if mission.sabotage_progress >= mission.sabotage_goal {
            mission.mission_active = false;
            mission.result = MissionResult::Victory;
            continue;
        }
        if mission.turns_remaining == 0 || mission.reactor_integrity <= 0.0 {
            mission.mission_active = false;
            mission.result = MissionResult::Defeat;
        }
    }
}

pub fn sync_mission_assignments_system(
    roster: Res<CampaignRoster>,
    mut mission_query: Query<(&MissionProgress, &mut AssignedHero)>,
) {
    if roster.heroes.is_empty() {
        return;
    }

    for (progress, mut assigned) in mission_query.iter_mut() {
        if progress.result != MissionResult::InProgress || !progress.mission_active {
            assigned.hero_id = None;
            continue;
        }
        let valid = assigned
            .hero_id
            .and_then(|id| {
                roster
                    .heroes
                    .iter()
                    .find(|h| h.id == id && h.active && !h.deserter)
                    .map(|_| id)
            })
            .is_some();
        if valid {
            continue;
        }
        let pick = roster
            .heroes
            .iter()
            .filter(|h| h.active && !h.deserter)
            .max_by(|a, b| {
                let sa = (a.loyalty + a.resolve) - (a.stress + a.fatigue + a.injury);
                let sb = (b.loyalty + b.resolve) - (b.stress + b.fatigue + b.injury);
                sa.total_cmp(&sb).then(a.id.cmp(&b.id))
            })
            .map(|h| h.id);
        assigned.hero_id = pick;
    }
}

pub fn companion_mission_impact_system(
    run_state: Res<RunState>,
    roster: Res<CampaignRoster>,
    mut mission_query: Query<(&mut MissionProgress, &AssignedHero)>,
) {
    if run_state.global_turn == 0 || roster.heroes.is_empty() {
        return;
    }
    for (mut progress, assigned) in mission_query.iter_mut() {
        if !progress.mission_active || progress.result != MissionResult::InProgress {
            continue;
        }
        let Some(hero_id) = assigned.hero_id else {
            continue;
        };
        let Some(hero) = roster.heroes.iter().find(|h| h.id == hero_id) else {
            continue;
        };

        let archetype_bonus = match hero.archetype {
            PersonalityArchetype::Vanguard => 0.8,
            PersonalityArchetype::Guardian => 0.5,
            PersonalityArchetype::Tactician => 1.0,
        };
        let composure = ((hero.resolve + hero.loyalty)
            - (hero.stress + hero.fatigue + hero.injury))
            .clamp(-40.0, 60.0);
        let progress_delta = (composure * 0.03 + archetype_bonus).clamp(-1.5, 2.8);
        progress.sabotage_progress =
            (progress.sabotage_progress + progress_delta).clamp(0.0, progress.sabotage_goal);

        let alert_delta = if composure >= 0.0 {
            -(0.4 + composure * 0.01)
        } else {
            0.6 + (-composure) * 0.015
        };
        progress.alert_level = (progress.alert_level + alert_delta).clamp(0.0, 100.0);
    }
}

pub fn companion_state_drift_system(
    run_state: Res<RunState>,
    mission_query: Query<(&MissionProgress, &AssignedHero)>,
    mut roster: ResMut<CampaignRoster>,
) {
    if run_state.global_turn == 0 || roster.heroes.is_empty() {
        return;
    }

    // Build a map from hero_id -> (alert_level, is_active_in_progress).
    let hero_mission: std::collections::HashMap<u32, (f32, bool)> = mission_query
        .iter()
        .filter_map(|(progress, assigned)| {
            let hero_id = assigned.hero_id?;
            let on_active = progress.mission_active && progress.result == MissionResult::InProgress;
            Some((hero_id, (progress.alert_level, on_active)))
        })
        .collect();

    for hero in &mut roster.heroes {
        if let Some(&(alert_level, on_active)) = hero_mission.get(&hero.id) {
            if on_active {
                hero.stress = (hero.stress + 0.7 + alert_level * 0.01).min(100.0);
                hero.fatigue = (hero.fatigue + 0.6).min(100.0);
                if alert_level > 55.0 {
                    hero.loyalty = (hero.loyalty - 0.35).max(0.0);
                } else {
                    hero.loyalty = (hero.loyalty + 0.08).min(100.0);
                }
                continue;
            }
        }

        hero.stress = (hero.stress - 0.5).max(0.0);
        hero.fatigue = (hero.fatigue - 0.7).max(0.0);
        hero.loyalty = (hero.loyalty + 0.04).min(100.0);
    }
}

pub fn generate_companion_story_quests_system(
    run_state: Res<RunState>,
    mission_query: Query<(&MissionProgress, &AssignedHero)>,
    overworld: Res<OverworldMap>,
    roster: Res<CampaignRoster>,
    mut story: ResMut<CompanionStoryState>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0 || roster.heroes.is_empty() || overworld.regions.is_empty() {
        return;
    }

    // Build hero_id -> pressure lookup from mission entities.
    let hero_pressure: std::collections::HashMap<u32, f32> = mission_query
        .iter()
        .filter_map(|(progress, assigned)| {
            let id = assigned.hero_id?;
            Some((id, progress.alert_level + (100.0 - progress.reactor_integrity) * 0.4))
        })
        .collect();

    let mut issued_events = Vec::new();
    for hero in roster.heroes.iter().filter(|h| h.active && !h.deserter) {
        if quest_for_hero(&story, hero.id).is_some() {
            continue;
        }
        let assigned_pressure = hero_pressure.get(&hero.id).copied().unwrap_or(0.0);
        let home_pressure = overworld
            .regions
            .iter()
            .find(|r| r.id == hero.origin_region_id)
            .map(|r| r.unrest + (100.0 - r.control) * 0.6)
            .unwrap_or(45.0);
        let trigger = hero.stress >= 42.0
            || hero.loyalty <= 55.0
            || assigned_pressure >= 58.0
            || home_pressure >= 70.0;
        if !trigger {
            continue;
        }
        let id = story.next_id;
        story.next_id = story.next_id.saturating_add(1);
        let quest = build_companion_quest(
            hero,
            run_state.global_turn,
            overworld.map_seed ^ 0xCA11_5100,
            id,
            &overworld,
        );
        story.notice = format!("Story quest issued: {}", quest.title);
        issued_events.push(format!(
            "Companion quest issued for {}: {}",
            hero.name, quest.title
        ));
        story.quests.push(quest);
    }
    if let Some(log) = event_log.as_mut() {
        for event in issued_events {
            push_campaign_event(log, run_state.global_turn, event);
        }
    }
}

pub fn progress_companion_story_quests_system(
    mut roster: ResMut<CampaignRoster>,
    ledger: Res<CampaignLedger>,
    mut story: ResMut<CompanionStoryState>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if roster.heroes.is_empty() {
        return;
    }

    for hero in &roster.heroes {
        if hero.deserter {
            if let Some(quest) = quest_for_hero_mut(&mut story, hero.id) {
                quest.status = CompanionQuestStatus::Failed;
                story.notice = format!("Story quest failed: {}", quest.title);
            }
        }
    }

    if story.processed_ledger_len > ledger.records.len() {
        story.processed_ledger_len = 0;
    }
    let mut pending_notice: Option<String> = None;
    let mut pending_events = Vec::new();
    for record in ledger.records.iter().skip(story.processed_ledger_len) {
        let Some(hero_id) = record.hero_id else {
            continue;
        };
        let Some(quest) = quest_for_hero_mut(&mut story, hero_id) else {
            continue;
        };
        match record.result {
            MissionResult::Victory => {
                quest.progress = quest.progress.saturating_add(1);
            }
            MissionResult::Defeat => {
                if quest.progress > 0 {
                    quest.progress -= 1;
                } else {
                    quest.status = CompanionQuestStatus::Failed;
                    let msg = format!("Story quest failed: {}", quest.title);
                    pending_notice = Some(msg.clone());
                    pending_events.push(msg);
                }
            }
            MissionResult::InProgress => {}
        }
        if quest.status == CompanionQuestStatus::Active && quest.progress >= quest.target {
            quest.status = CompanionQuestStatus::Completed;
            if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == quest.hero_id) {
                hero.loyalty = (hero.loyalty + quest.reward_loyalty).clamp(0.0, 100.0);
                hero.resolve = (hero.resolve + quest.reward_resolve).clamp(0.0, 100.0);
                hero.stress = (hero.stress - 6.0).max(0.0);
            }
            let msg = format!("Story quest complete: {}", quest.title);
            pending_notice = Some(msg.clone());
            pending_events.push(msg);
        }
    }
    if let Some(msg) = pending_notice {
        story.notice = msg;
    }
    if let Some(log) = event_log.as_mut() {
        for event in pending_events {
            let turn = ledger.records.last().map(|r| r.turn).unwrap_or(0);
            push_campaign_event(log, turn, event);
        }
    }
    story.processed_ledger_len = ledger.records.len();
}

pub fn resolve_mission_consequences_system(
    run_state: Res<RunState>,
    mut mission_query: Query<(&MissionData, &mut MissionProgress, &mut AssignedHero)>,
    mut roster: ResMut<CampaignRoster>,
    mut ledger: ResMut<CampaignLedger>,
    mut flashpoints: Option<ResMut<FlashpointState>>,
    board: Res<MissionBoard>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0 {
        return;
    }

    for (slot, &entity) in board.entities.iter().enumerate() {
        let Ok((data, mut progress, mut assigned)) = mission_query.get_mut(entity) else {
            continue;
        };
        if progress.result == MissionResult::InProgress || progress.outcome_recorded {
            continue;
        }

        let hero_id = assigned.hero_id;
        if let Some(state) = flashpoints.as_mut() {
            if let Some(chain) = state
                .chains
                .iter_mut()
                .find(|c| !c.completed && c.mission_slot == slot)
            {
                chain.companion_hook_hero_id = hero_id;
            }
        }
        let mut summary = format!("{} resolved as {:?}", data.mission_name, progress.result);

        if let Some(id) = hero_id {
            if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == id) {
                match progress.result {
                    MissionResult::Victory => {
                        hero.stress = (hero.stress - 6.0).max(0.0);
                        hero.fatigue = (hero.fatigue + 1.5).min(100.0);
                        hero.injury = (hero.injury - 1.0).max(0.0);
                        hero.loyalty = (hero.loyalty + 1.8).min(100.0);
                        summary = format!(
                            "{} succeeded on '{}'. morale improved.",
                            hero.name, data.mission_name
                        );
                    }
                    MissionResult::Defeat => {
                        let pressure =
                            (progress.alert_level * 0.05) + (progress.unattended_turns as f32 * 0.25);
                        hero.stress = (hero.stress + 12.0 + pressure).min(100.0);
                        hero.fatigue = (hero.fatigue + 8.0 + pressure * 0.5).min(100.0);
                        hero.injury = (hero.injury + 6.0 + pressure * 0.8).min(100.0);
                        hero.loyalty = (hero.loyalty - (3.0 + pressure * 0.4)).max(0.0);

                        if hero.injury >= 95.0 || (hero.loyalty <= 10.0 && hero.stress >= 70.0) {
                            hero.deserter = true;
                            hero.active = false;
                            summary = format!(
                                "{} deserted after '{}' collapse.",
                                hero.name, data.mission_name
                            );
                        } else if hero.injury >= 65.0 {
                            hero.active = false;
                            summary = format!(
                                "{} is sidelined with severe injuries from '{}'.",
                                hero.name, data.mission_name
                            );
                        } else {
                            summary = format!(
                                "{} survived defeat at '{}'.",
                                hero.name, data.mission_name
                            );
                        }
                    }
                    MissionResult::InProgress => {}
                }
            }
        }

        progress.outcome_recorded = true;
        assigned.hero_id = None;
        ledger.records.push(ConsequenceRecord {
            turn: run_state.global_turn,
            mission_name: data.mission_name.clone(),
            result: progress.result,
            hero_id,
            summary,
        });
        if let Some(log) = event_log.as_mut() {
            if let Some(last) = ledger.records.last() {
                push_campaign_event(log, run_state.global_turn, last.summary.clone());
            }
        }
    }
}

pub fn flashpoint_progression_system(
    run_state: Res<RunState>,
    board: Res<MissionBoard>,
    mut mission_query: Query<(&mut MissionData, &mut MissionProgress, &mut MissionTactics)>,
    mut overworld: ResMut<OverworldMap>,
    mut roster: ResMut<CampaignRoster>,
    mut flashpoints: ResMut<FlashpointState>,
    mut story: Option<ResMut<CompanionStoryState>>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if run_state.global_turn == 0 || flashpoints.chains.is_empty() {
        return;
    }

    let mut pending_events = Vec::new();
    for slot in 0..board.entities.len() {
        let Some(chain_idx) = flashpoints
            .chains
            .iter()
            .position(|c| !c.completed && c.mission_slot == slot)
        else {
            continue;
        };

        let Some(&entity) = board.entities.get(slot) else {
            continue;
        };
        let Ok((mut data, mut progress, mut tactics)) = mission_query.get_mut(entity) else {
            continue;
        };
        if progress.result == MissionResult::InProgress || !progress.outcome_recorded {
            continue;
        }

        // Build snapshot for helper functions that take &mut MissionSnapshot.
        let mut snap = MissionSnapshot::from_components(&data, &progress, &tactics);

        let mut chain = flashpoints.chains[chain_idx].clone();
        let region_name = overworld
            .regions
            .iter()
            .find(|r| r.id == chain.region_id)
            .map(|r| r.name.clone())
            .unwrap_or_else(|| format!("Region {}", chain.region_id));
        let attacker_name = overworld
            .factions
            .get(chain.attacker_faction_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", chain.attacker_faction_id));
        let defender_name = overworld
            .factions
            .get(chain.defender_faction_id)
            .map(|f| f.name.clone())
            .unwrap_or_else(|| format!("Faction {}", chain.defender_faction_id));

        match progress.result {
            MissionResult::Victory if chain.stage < FLASHPOINT_TOTAL_STAGES => {
                chain.stage = chain.stage.saturating_add(1);
                configure_flashpoint_stage_mission(
                    &mut snap,
                    &chain,
                    &overworld,
                    overworld.map_seed ^ (run_state.global_turn as u64).wrapping_mul(911),
                );
                let _ = apply_flashpoint_companion_hook(&mut snap, &chain, &roster);
                chain.objective = flashpoint_hook_objective_suffix(&chain, &roster)
                    .unwrap_or_else(|| "No companion hook".to_string());
                apply_flashpoint_intent(&mut snap, &chain);
                rewrite_flashpoint_mission_name(&mut snap, &chain, &overworld, Some(&roster));
                // Write snapshot back.
                progress.mission_active = snap.mission_active;
                progress.result = snap.result;
                progress.turns_remaining = snap.turns_remaining;
                progress.reactor_integrity = snap.reactor_integrity;
                progress.sabotage_progress = snap.sabotage_progress;
                progress.sabotage_goal = snap.sabotage_goal;
                progress.alert_level = snap.alert_level;
                progress.room_index = snap.room_index;
                progress.unattended_turns = snap.unattended_turns;
                progress.outcome_recorded = snap.outcome_recorded;
                tactics.tactical_mode = snap.tactical_mode;
                tactics.command_cooldown_turns = snap.command_cooldown_turns;
                data.mission_name = snap.mission_name.clone();
                data.bound_region_id = snap.bound_region_id;
                flashpoints.notice = format!(
                    "Flashpoint advanced in {}: stage {}/{}.",
                    region_name, chain.stage, FLASHPOINT_TOTAL_STAGES
                );
                pending_events.push(flashpoints.notice.clone());
                flashpoints.chains[chain_idx] = chain;
                continue;
            }
            MissionResult::Victory => {
                chain.completed = true;
                if let Some(region) = overworld.regions.iter_mut().find(|r| r.id == chain.region_id)
                {
                    region.owner_faction_id = chain.attacker_faction_id;
                    region.control = (region.control + 18.0).clamp(0.0, 100.0);
                    region.unrest = (region.unrest - 14.0).clamp(0.0, 100.0);
                }
                ensure_faction_mission_slots(&mut overworld);
                if let Some(attacker) = overworld.factions.get_mut(chain.attacker_faction_id) {
                    attacker.strength = (attacker.strength + 7.0).clamp(30.0, 180.0);
                    attacker.cohesion = (attacker.cohesion + 4.0).clamp(10.0, 95.0);
                }
                if let Some(defender) = overworld.factions.get_mut(chain.defender_faction_id) {
                    defender.strength = (defender.strength - 5.0).clamp(30.0, 180.0);
                    defender.cohesion = (defender.cohesion - 3.0).clamp(10.0, 95.0);
                }
                inject_recruit_for_faction(
                    &mut roster,
                    &overworld,
                    chain.attacker_faction_id,
                    overworld.map_seed ^ run_state.global_turn as u64 ^ chain.id as u64,
                );
                if let Some((hero_id, hero_name, _kind)) =
                    flashpoint_companion_hook_kind(&chain, &roster)
                {
                    if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == hero_id) {
                        hero.loyalty = (hero.loyalty + 4.0).clamp(0.0, 100.0);
                        hero.resolve = (hero.resolve + 3.0).clamp(0.0, 100.0);
                        hero.stress = (hero.stress - 4.0).clamp(0.0, 100.0);
                    }
                    pending_events.push(format!(
                        "{hero_name} gained renown from flashpoint resolution in {}.",
                        region_name
                    ));
                    if let Some(story_state) = story.as_mut() {
                        if let Some(quest) = story_state
                            .quests
                            .iter_mut()
                            .find(|q| {
                                q.hero_id == hero_id
                                    && q.status == CompanionQuestStatus::Active
                                    && q.progress < q.target
                            })
                        {
                            quest.progress = quest.progress.saturating_add(1).min(quest.target);
                            story_state.notice = format!(
                                "Flashpoint beat advanced companion quest: {}",
                                quest.title
                            );
                            pending_events.push(story_state.notice.clone());
                        }
                    }
                }
                if !chain.objective.is_empty() {
                    pending_events.push(format!(
                        "Hook objective resolved at {}: {}",
                        region_name, chain.objective
                    ));
                }
                flashpoints.notice = format!(
                    "Flashpoint resolved: {} seized {} from {} and opened new recruits.",
                    attacker_name, region_name, defender_name
                );
                pending_events.push(flashpoints.notice.clone());
            }
            MissionResult::Defeat => {
                chain.completed = true;
                if let Some(region) = overworld.regions.iter_mut().find(|r| r.id == chain.region_id)
                {
                    region.owner_faction_id = chain.defender_faction_id;
                    region.control = (region.control + 12.0).clamp(0.0, 100.0);
                    region.unrest = (region.unrest - 9.0).clamp(0.0, 100.0);
                }
                ensure_faction_mission_slots(&mut overworld);
                if let Some(defender) = overworld.factions.get_mut(chain.defender_faction_id) {
                    defender.strength = (defender.strength + 5.0).clamp(30.0, 180.0);
                    defender.cohesion = (defender.cohesion + 3.0).clamp(10.0, 95.0);
                }
                if let Some(attacker) = overworld.factions.get_mut(chain.attacker_faction_id) {
                    attacker.strength = (attacker.strength - 4.0).clamp(30.0, 180.0);
                }
                if let Some(idx) = roster
                    .recruit_pool
                    .iter()
                    .position(|r| r.origin_faction_id == chain.attacker_faction_id)
                {
                    roster.recruit_pool.remove(idx);
                }
                if let Some((hero_id, hero_name, _kind)) =
                    flashpoint_companion_hook_kind(&chain, &roster)
                {
                    if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == hero_id) {
                        hero.loyalty = (hero.loyalty - 3.0).clamp(0.0, 100.0);
                        hero.stress = (hero.stress + 6.0).clamp(0.0, 100.0);
                    }
                    pending_events.push(format!(
                        "{hero_name} took a morale hit from the failed flashpoint at {}.",
                        region_name
                    ));
                    if let Some(story_state) = story.as_mut() {
                        if let Some(quest) = story_state
                            .quests
                            .iter_mut()
                            .find(|q| q.hero_id == hero_id && q.status == CompanionQuestStatus::Active)
                        {
                            if quest.progress > 0 {
                                quest.progress -= 1;
                            }
                            story_state.notice = format!(
                                "Flashpoint setback affected companion quest: {}",
                                quest.title
                            );
                            pending_events.push(story_state.notice.clone());
                        }
                    }
                }
                if !chain.objective.is_empty() {
                    pending_events.push(format!(
                        "Hook objective failed at {}: {}",
                        region_name, chain.objective
                    ));
                }
                flashpoints.notice = format!(
                    "Flashpoint collapsed in {}: {} held against {}.",
                    region_name, defender_name, attacker_name
                );
                pending_events.push(flashpoints.notice.clone());
            }
            MissionResult::InProgress => {}
        }

        flashpoints.chains[chain_idx] = chain;
    }

    flashpoints.chains.retain(|c| !c.completed);
    if flashpoints.chains.is_empty() && flashpoints.notice.is_empty() {
        flashpoints.notice = "No active flashpoints.".to_string();
    }
    if let Some(log) = event_log.as_mut() {
        for event in pending_events {
            push_campaign_event(log, run_state.global_turn, event);
        }
    }
}

pub fn companion_recovery_system(run_state: Res<RunState>, mut roster: ResMut<CampaignRoster>) {
    if run_state.global_turn == 0 || run_state.global_turn % 3 != 0 {
        return;
    }

    for hero in &mut roster.heroes {
        if hero.deserter {
            continue;
        }
        if hero.active {
            continue;
        }
        hero.stress = (hero.stress - 3.0).max(0.0);
        hero.fatigue = (hero.fatigue - 4.0).max(0.0);
        hero.injury = (hero.injury - 3.5).max(0.0);
        hero.loyalty = (hero.loyalty + 0.7).min(100.0);

        if hero.injury <= 40.0 && hero.fatigue <= 40.0 {
            hero.active = true;
        }
    }
}

pub fn turn_management_system(
    run_state: Res<RunState>,
    mut active_query: Query<(&mut MissionProgress, &mut MissionTactics), With<ActiveMission>>,
) {
    let Ok((mut progress, mut tactics)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    if run_state.global_turn > 0 && progress.turns_remaining > 0 {
        progress.turns_remaining -= 1;
    }
    if tactics.command_cooldown_turns > 0 {
        tactics.command_cooldown_turns -= 1;
    }
}

pub fn auto_increase_stress(mut query: Query<&mut Stress, With<Hero>>, run_state: Res<RunState>) {
    if run_state.global_turn % 10 == 0 && run_state.global_turn > 0 {
        for mut stress in query.iter_mut() {
            stress.value = (stress.value + 5.0).min(stress.max);
            println!(
                "Stress automatically increased for hero! Current stress: {:.1}",
                stress.value
            );
        }
    }
}

pub fn activate_mission_system(
    run_state: Res<RunState>,
    mut active_query: Query<(&MissionData, &mut MissionProgress), With<ActiveMission>>,
) {
    let Ok((data, mut progress)) = active_query.get_single_mut() else {
        return;
    };
    if run_state.global_turn == 5 && !progress.mission_active {
        progress.mission_active = true;
        progress.result = MissionResult::InProgress;
        println!("Mission '{}' is now ACTIVE!", data.mission_name);
    }
}

pub fn mission_map_progression_system(
    run_state: Res<RunState>,
    mission_map: Res<MissionMap>,
    mut active_query: Query<&mut MissionProgress, With<ActiveMission>>,
) {
    let Ok(mut progress) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active || run_state.global_turn == 0 {
        return;
    }

    let next_room_index = progress.room_index + 1;
    let Some(next_room) = mission_map.rooms.get(next_room_index) else {
        return;
    };
    let next_room_name = next_room.room_name.clone();
    let next_room_type = next_room.room_type;
    let next_room_threshold = next_room.sabotage_threshold;

    if progress.sabotage_progress >= next_room_threshold {
        progress.room_index = next_room_index;
        println!(
            "Map progression: entered '{}' ({:?}).",
            next_room_name, next_room_type
        );
    }
}

pub fn player_command_input_system(
    keyboard_input: Option<Res<ButtonInput<KeyCode>>>,
    mut active_query: Query<(&MissionProgress, &mut MissionTactics), With<ActiveMission>>,
) {
    let Ok((progress, mut tactics)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    let Some(keyboard) = keyboard_input else {
        return;
    };

    if keyboard.just_pressed(KeyCode::Digit1) {
        tactics.tactical_mode = TacticalMode::Balanced;
        println!("Command: switched tactical mode to BALANCED.");
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        tactics.tactical_mode = TacticalMode::Aggressive;
        println!("Command: switched tactical mode to AGGRESSIVE.");
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        tactics.tactical_mode = TacticalMode::Defensive;
        println!("Command: switched tactical mode to DEFENSIVE.");
    }

    if tactics.command_cooldown_turns == 0 && keyboard.just_pressed(KeyCode::KeyB) {
        tactics.force_sabotage_order = true;
        tactics.command_cooldown_turns = 3;
        println!("Command: BREACH ORDER issued.");
    }
    if tactics.command_cooldown_turns == 0 && keyboard.just_pressed(KeyCode::KeyR) {
        tactics.force_stabilize_order = true;
        tactics.command_cooldown_turns = 3;
        println!("Command: REGROUP ORDER issued.");
    }
}

pub fn hero_ability_system(
    run_state: Res<RunState>,
    mut active_query: Query<(&mut MissionProgress, &mut MissionTactics), With<ActiveMission>>,
    mut hero_query: Query<
        (&mut HeroAbilities, &mut Health, &mut Stress),
        (With<Hero>, Without<Enemy>),
    >,
    mut enemy_query: Query<(&Enemy, &mut Health), (With<Enemy>, Without<Hero>)>,
) {
    let Ok((mut progress, mut tactics)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active || run_state.global_turn == 0 {
        return;
    }

    for (mut abilities, mut hero_health, mut stress) in hero_query.iter_mut() {
        let (focus_damage, stabilize_heal, stabilize_stress_relief, sabotage_gain) =
            match tactics.tactical_mode {
                TacticalMode::Balanced => (12.0, 6.0, 8.0, 18.0),
                TacticalMode::Aggressive => (15.0, 4.0, 6.0, 22.0),
                TacticalMode::Defensive => (9.0, 10.0, 12.0, 14.0),
            };

        if abilities.focus_fire_cooldown > 0 {
            abilities.focus_fire_cooldown -= 1;
        }
        if abilities.stabilize_cooldown > 0 {
            abilities.stabilize_cooldown -= 1;
        }
        if abilities.sabotage_charge_cooldown > 0 {
            abilities.sabotage_charge_cooldown -= 1;
        }

        if abilities.focus_fire_cooldown == 0 {
            let mut best_target: Option<(String, Mut<Health>)> = None;
            for (enemy, enemy_health) in enemy_query.iter_mut() {
                if enemy_health.current <= 0.0 {
                    continue;
                }
                match &best_target {
                    None => best_target = Some((enemy.name.clone(), enemy_health)),
                    Some((_, existing_health))
                        if enemy_health.current > existing_health.current =>
                    {
                        best_target = Some((enemy.name.clone(), enemy_health));
                    }
                    _ => {}
                }
            }

            if let Some((enemy_name, mut target_health)) = best_target {
                target_health.current = (target_health.current - focus_damage).max(0.0);
                abilities.focus_fire_cooldown = 3;
                println!(
                    "Hero ability: Focus Fire hit {} for {:.1} damage.",
                    enemy_name, focus_damage
                );
            }
        }

        let stabilize_threshold = if tactics.tactical_mode == TacticalMode::Defensive {
            14.0
        } else {
            20.0
        };
        if (abilities.stabilize_cooldown == 0 && stress.value >= stabilize_threshold)
            || tactics.force_stabilize_order
        {
            stress.value = (stress.value - stabilize_stress_relief).max(0.0);
            hero_health.current = (hero_health.current + stabilize_heal).min(hero_health.max);
            abilities.stabilize_cooldown = 7;
            println!("Hero ability: Stabilize recovered HP and reduced stress.");
            tactics.force_stabilize_order = false;
        }

        if abilities.sabotage_charge_cooldown == 0 || tactics.force_sabotage_order {
            progress.sabotage_progress =
                (progress.sabotage_progress + sabotage_gain).min(progress.sabotage_goal);
            abilities.sabotage_charge_cooldown = 4;
            println!("Hero ability: Sabotage Charge advanced ritual breach progress.");
            tactics.force_sabotage_order = false;
        }

        if tactics.tactical_mode == TacticalMode::Aggressive {
            stress.value = (stress.value + 1.0).min(stress.max);
        }
    }
}

pub fn enemy_ai_system(
    run_state: Res<RunState>,
    mut active_query: Query<&mut MissionProgress, With<ActiveMission>>,
    mut hero_query: Query<&mut Health, (With<Hero>, Without<Enemy>)>,
    mut enemy_ai_query: Query<(&Enemy, &mut EnemyAI, &Health), (With<Enemy>, Without<Hero>)>,
) {
    let Ok(mut progress) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active || run_state.global_turn == 0 {
        return;
    }

    let enemies_alive = enemy_ai_query
        .iter_mut()
        .any(|(_, _, health)| health.current > 0.0);
    if !enemies_alive {
        return;
    }

    for (enemy, mut ai, enemy_health) in enemy_ai_query.iter_mut() {
        if enemy_health.current <= 0.0 {
            continue;
        }

        if ai.turns_until_attack > 0 {
            ai.turns_until_attack -= 1;
            continue;
        }

        let enraged = (enemy_health.current / enemy_health.max) <= ai.enraged_threshold;
        let mut damage = if enraged {
            ai.base_attack_power * 1.5
        } else {
            ai.base_attack_power
        };
        if progress.alert_level >= 40.0 {
            damage += 1.5;
        }

        for mut hero_health in hero_query.iter_mut() {
            hero_health.current = (hero_health.current - damage).max(0.0);
        }

        progress.alert_level += 6.0;
        progress.sabotage_progress = (progress.sabotage_progress - 3.0).max(0.0);
        progress.reactor_integrity = (progress.reactor_integrity - 2.5).max(0.0);

        ai.turns_until_attack = ai.attack_interval;
        println!(
            "Enemy AI: {} attacked for {:.1} damage (enraged: {}).",
            enemy.name, damage, enraged
        );
    }
}

pub fn combat_system(
    run_state: Res<RunState>,
    active_query: Query<&MissionProgress, With<ActiveMission>>,
    mut hero_stress_query: Query<&mut Stress, With<Hero>>,
    mut enemy_health_query: Query<&mut Health, (With<Enemy>, Without<Hero>)>,
) {
    let Ok(progress) = active_query.get_single() else {
        return;
    };
    if !progress.mission_active || run_state.global_turn == 0 || run_state.global_turn % 5 != 0
    {
        return;
    }

    let mut any_enemy_alive_after_attack = false;
    for mut enemy_health in enemy_health_query.iter_mut() {
        if enemy_health.current <= 0.0 {
            continue;
        }
        enemy_health.current = (enemy_health.current - 6.0).max(0.0);
        any_enemy_alive_after_attack |= enemy_health.current > 0.0;
    }

    if any_enemy_alive_after_attack {
        for mut stress in hero_stress_query.iter_mut() {
            stress.value = (stress.value + 2.0).min(stress.max);
        }
    }
}

pub fn complete_objective_system(
    mut objective_query: Query<&mut MissionObjective>,
    active_query: Query<&MissionProgress, With<ActiveMission>>,
) {
    let Ok(progress) = active_query.get_single() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    if progress.sabotage_progress < progress.sabotage_goal {
        return;
    }

    for mut objective in objective_query.iter_mut() {
        if !objective.completed {
            objective.completed = true;
            println!("Objective '{}' COMPLETED!", objective.description);
        }
    }
}

pub fn end_mission_system(
    objective_query: Query<&MissionObjective>,
    hero_health_query: Query<&Health, With<Hero>>,
    mut active_query: Query<(&MissionData, &mut MissionProgress), With<ActiveMission>>,
) {
    let Ok((data, mut progress)) = active_query.get_single_mut() else {
        return;
    };
    if !progress.mission_active {
        return;
    }

    let hero_alive = hero_health_query.iter().all(|health| health.current > 0.0);
    if !hero_alive {
        progress.mission_active = false;
        progress.result = MissionResult::Defeat;
        println!(
            "Mission '{}' FAILED: hero defeated.",
            data.mission_name
        );
        return;
    }

    if progress.reactor_integrity <= 0.0 {
        progress.mission_active = false;
        progress.result = MissionResult::Defeat;
        println!(
            "Mission '{}' FAILED: ward lattice collapsed.",
            data.mission_name
        );
        return;
    }

    let all_objectives_completed = objective_query.iter().all(|objective| objective.completed);
    if all_objectives_completed {
        progress.mission_active = false;
        progress.result = MissionResult::Victory;
        println!(
            "Mission '{}' COMPLETED! Mission is now INACTIVE.",
            data.mission_name
        );
        return;
    }

    if progress.turns_remaining == 0 {
        progress.mission_active = false;
        progress.result = MissionResult::Defeat;
        println!(
            "Mission '{}' FAILED: time expired.",
            data.mission_name
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn active_progress(world: &mut World) -> MissionProgress {
        let mut qs = world.query_filtered::<&MissionProgress, With<ActiveMission>>();
        qs.single(world).clone()
    }

    fn active_tactics(world: &mut World) -> MissionTactics {
        let mut qs = world.query_filtered::<&MissionTactics, With<ActiveMission>>();
        qs.single(world).clone()
    }

    fn set_active_progress<F: FnOnce(&mut MissionProgress)>(world: &mut World, f: F) {
        let mut qs = world.query_filtered::<&mut MissionProgress, With<ActiveMission>>();
        let mut p = qs.single_mut(world);
        f(&mut p);
    }

    fn set_active_tactics<F: FnOnce(&mut MissionTactics)>(world: &mut World, f: F) {
        let mut qs = world.query_filtered::<&mut MissionTactics, With<ActiveMission>>();
        let mut t = qs.single_mut(world);
        f(&mut t);
    }

    fn nth_progress(world: &mut World, n: usize) -> MissionProgress {
        let entity = world.resource::<MissionBoard>().entities[n];
        world.get::<MissionProgress>(entity).expect("MissionProgress on entity").clone()
    }

    fn set_nth_progress<F: FnOnce(&mut MissionProgress)>(world: &mut World, n: usize, f: F) {
        let entity = world.resource::<MissionBoard>().entities[n];
        let mut p = world.get_mut::<MissionProgress>(entity).expect("MissionProgress");
        f(&mut p);
    }

    fn mission_count(world: &mut World) -> usize {
        world.resource::<MissionBoard>().entities.len()
    }

    fn board_active_idx(world: &mut World) -> usize {
        let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        entities
            .iter()
            .position(|&e| world.get::<ActiveMission>(e).is_some())
            .unwrap_or(0)
    }

    /// Overwrites all three components on a mission entity from a snapshot.
    fn overwrite_mission_entity(world: &mut World, entity: Entity, snap: MissionSnapshot) {
        let id = world.get::<MissionData>(entity).unwrap().id;
        let (data, progress, tactics) = snap.into_components(id);
        *world.get_mut::<MissionData>(entity).unwrap() = data;
        *world.get_mut::<MissionProgress>(entity).unwrap() = progress;
        *world.get_mut::<MissionTactics>(entity).unwrap() = tactics;
    }

    /// Spawns default mission entities into `world`, populating `MissionBoard.entities`.
    fn spawn_test_missions(world: &mut World) {
        let snaps = default_mission_snapshots();
        let mut entities = Vec::new();
        for (i, snap) in snaps.into_iter().enumerate() {
            let id = {
                let mut board = world.resource_mut::<MissionBoard>();
                let id = board.next_id;
                board.next_id += 1;
                id
            };
            let (data, progress, tactics) = snap.into_components(id);
            let entity = if i == 0 {
                world
                    .spawn((data, progress, tactics, AssignedHero::default(), ActiveMission))
                    .id()
            } else {
                world
                    .spawn((data, progress, tactics, AssignedHero::default()))
                    .id()
            };
            entities.push(entity);
        }
        world.resource_mut::<MissionBoard>().entities = entities;
    }

    // ── app builders ─────────────────────────────────────────────────────────

    fn build_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<RunState>()
            .init_resource::<MissionBoard>()
            .init_resource::<MissionMap>()
            .add_systems(
                Update,
                (
                    increment_turn_for_tests,
                    turn_management_system,
                    activate_mission_system,
                    mission_map_progression_system,
                    player_command_input_system,
                    hero_ability_system,
                    enemy_ai_system,
                    combat_system,
                    complete_objective_system,
                    end_mission_system,
                )
                    .chain(),
            );
        // Spawn hero, enemy, objective, and mission entities directly.
        app.world.spawn((
            Hero {
                name: "Warden Lyra".to_string(),
            },
            Stress {
                value: 0.0,
                max: 100.0,
            },
            Health {
                current: 100.0,
                max: 100.0,
            },
            HeroAbilities {
                focus_fire_cooldown: 0,
                stabilize_cooldown: 0,
                sabotage_charge_cooldown: 0,
            },
        ));
        app.world.spawn((
            Enemy {
                name: "Crypt Sentinel".to_string(),
            },
            Health {
                current: 40.0,
                max: 40.0,
            },
            EnemyAI {
                base_attack_power: 6.0,
                turns_until_attack: 1,
                attack_interval: 2,
                enraged_threshold: 0.5,
            },
        ));
        app.world.spawn(MissionObjective {
            description: "Rupture the ritual anchor".to_string(),
            completed: false,
        });
        spawn_test_missions(&mut app.world);
        app
    }

    fn increment_turn_for_tests(mut run_state: ResMut<RunState>) {
        run_state.global_turn += 1;
    }

    fn build_triage_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<RunState>()
            .init_resource::<MissionBoard>()
            .init_resource::<MissionMap>()
            .init_resource::<AttentionState>()
            .init_resource::<OverworldMap>()
            .init_resource::<CampaignRoster>()
            .init_resource::<CampaignLedger>()
            .init_resource::<CompanionStoryState>()
            .add_systems(
                Update,
                (
                    increment_turn_for_tests,
                    attention_management_system,
                    overworld_cooldown_system,
                    overworld_sync_from_missions_system,
                    turn_management_system,
                    focused_attention_intervention_system,
                    mission_map_progression_system,
                    simulate_unfocused_missions_system,
                    sync_mission_assignments_system,
                    companion_mission_impact_system,
                    companion_state_drift_system,
                    resolve_mission_consequences_system,
                    progress_companion_story_quests_system,
                    generate_companion_story_quests_system,
                    companion_recovery_system,
                )
                    .chain(),
            );
        spawn_test_missions(&mut app.world);
        app
    }

    // ── tests ────────────────────────────────────────────────────────────────

    #[test]
    fn mission_activates_on_turn_five() {
        let mut app = build_test_app();

        for _ in 0..5 {
            app.update();
        }

        let progress = active_progress(&mut app.world);
        assert!(progress.mission_active);
        assert_eq!(progress.result, MissionResult::InProgress);
    }

    #[test]
    fn hero_abilities_advance_sabotage_progress() {
        let mut app = build_test_app();

        for _ in 0..10 {
            app.update();
        }

        let progress = active_progress(&mut app.world);
        assert!(progress.sabotage_progress > 0.0);
    }

    #[test]
    fn enemy_ai_can_damage_hero() {
        let mut app = build_test_app();

        for _ in 0..12 {
            app.update();
        }

        let mut hero_query = app.world.query::<(&Hero, &Health)>();
        let (_, hero_health) = hero_query.single(&app.world);
        assert!(hero_health.current < hero_health.max);
    }

    #[test]
    fn mission_can_end_in_victory() {
        let mut app = build_test_app();

        for _ in 0..40 {
            app.update();
            let p = active_progress(&mut app.world);
            if !p.mission_active && p.result == MissionResult::Victory {
                return;
            }
        }

        assert_eq!(active_progress(&mut app.world).result, MissionResult::Victory);
    }

    #[test]
    fn aggressive_mode_increases_sabotage_speed() {
        let mut app = build_test_app();

        set_active_tactics(&mut app.world, |t| t.tactical_mode = TacticalMode::Aggressive);
        for _ in 0..10 {
            app.update();
        }
        let aggressive_progress = active_progress(&mut app.world).sabotage_progress;

        let mut baseline = build_test_app();
        for _ in 0..10 {
            baseline.update();
        }
        let baseline_progress = active_progress(&mut baseline.world).sabotage_progress;

        assert!(aggressive_progress > baseline_progress);
    }

    #[test]
    fn mission_defeats_when_timer_expires() {
        let mut app = build_test_app();
        set_active_progress(&mut app.world, |p| {
            p.mission_active = true;
            p.result = MissionResult::InProgress;
            p.turns_remaining = 1;
        });

        app.update();

        let p = active_progress(&mut app.world);
        assert!(!p.mission_active);
        assert_eq!(p.result, MissionResult::Defeat);
    }

    #[test]
    fn map_progresses_when_threshold_is_crossed() {
        let mut app = build_test_app();
        set_active_progress(&mut app.world, |p| {
            p.mission_active = true;
            p.sabotage_progress = 20.0;
        });

        app.update();

        let p = active_progress(&mut app.world);
        assert_eq!(p.room_index, 1);
    }

    #[test]
    fn command_cooldown_counts_down_each_turn_when_active() {
        let mut app = build_test_app();
        set_active_progress(&mut app.world, |p| p.mission_active = true);
        set_active_tactics(&mut app.world, |t| t.command_cooldown_turns = 3);

        app.update();

        let t = active_tactics(&mut app.world);
        assert_eq!(t.command_cooldown_turns, 2);
    }

    #[test]
    fn enemy_pressure_raises_alert_and_reduces_integrity() {
        let mut app = build_test_app();
        set_active_progress(&mut app.world, |p| p.mission_active = true);
        let initial_alert = active_progress(&mut app.world).alert_level;
        let initial_integrity = active_progress(&mut app.world).reactor_integrity;

        app.update();
        app.update();

        let p = active_progress(&mut app.world);
        assert!(p.alert_level > initial_alert);
        assert!(p.reactor_integrity < initial_integrity);
    }

    #[test]
    fn pressure_curve_is_deterministic_across_identical_runs() {
        let mut a = build_test_app();
        let mut b = build_test_app();
        set_active_progress(&mut a.world, |p| p.mission_active = true);
        set_active_progress(&mut b.world, |p| p.mission_active = true);

        for _ in 0..12 {
            a.update();
            b.update();
        }

        let pa = active_progress(&mut a.world);
        let pb = active_progress(&mut b.world);
        assert_eq!(pa.sabotage_progress, pb.sabotage_progress);
        assert_eq!(pa.alert_level, pb.alert_level);
        assert_eq!(pa.reactor_integrity, pb.reactor_integrity);
        assert_eq!(pa.turns_remaining, pb.turns_remaining);
    }

    #[test]
    fn unfocused_missions_progress_without_focus() {
        let mut app = build_triage_app();
        let initial = nth_progress(&mut app.world, 1);

        for _ in 0..4 {
            app.update();
        }

        let p = nth_progress(&mut app.world, 1);
        assert!(p.turns_remaining < initial.turns_remaining);
        assert!(p.alert_level > initial.alert_level);
        assert!(p.reactor_integrity < initial.reactor_integrity);
        assert!(p.sabotage_progress < initial.sabotage_progress + 20.0);
        assert!(p.unattended_turns >= 4);
    }

    #[test]
    fn try_shift_focus_spends_attention_and_sets_cooldown() {
        let mut attention = AttentionState::default();

        let result = try_shift_focus(2, &mut attention, 0, 1);

        assert_eq!(result, Some(1));
        assert_eq!(attention.switch_cooldown_turns, attention.switch_cooldown_max);
        assert!(attention.global_energy < attention.max_energy);
    }

    #[test]
    fn try_shift_focus_blocks_when_attention_is_exhausted() {
        let mut attention = AttentionState::default();
        attention.global_energy = 0.0;

        let result = try_shift_focus(2, &mut attention, 0, 1);

        assert!(result.is_none());
    }

    #[test]
    fn triage_simulation_is_deterministic_for_same_initial_state() {
        let mut a = build_triage_app();
        let mut b = build_triage_app();

        for _ in 0..8 {
            a.update();
            b.update();
        }

        let count = mission_count(&mut a.world);
        assert_eq!(count, mission_count(&mut b.world));
        assert_eq!(board_active_idx(&mut a.world), board_active_idx(&mut b.world));
        for idx in 0..count {
            let pa = nth_progress(&mut a.world, idx);
            let pb = nth_progress(&mut b.world, idx);
            assert_eq!(pa.mission_active, pb.mission_active);
            assert_eq!(pa.result, pb.result);
            assert_eq!(pa.turns_remaining, pb.turns_remaining);
            assert_eq!(pa.sabotage_progress, pb.sabotage_progress);
            assert_eq!(pa.alert_level, pb.alert_level);
            assert_eq!(pa.reactor_integrity, pb.reactor_integrity);
            assert_eq!(pa.unattended_turns, pb.unattended_turns);
        }
    }

    #[test]
    fn focused_mission_outpaces_matching_unfocused_mission() {
        let mut app = build_triage_app();
        let entities: Vec<Entity> = app.world.resource::<MissionBoard>().entities.clone();
        let focused_snap = MissionSnapshot {
            mission_name: "Focus".to_string(),
            bound_region_id: Some(0),
            mission_active: true,
            result: MissionResult::InProgress,
            turns_remaining: 20,
            reactor_integrity: 95.0,
            sabotage_progress: 30.0,
            sabotage_goal: 100.0,
            alert_level: 20.0,
            room_index: 1,
            tactical_mode: TacticalMode::Balanced,
            command_cooldown_turns: 0,
            unattended_turns: 0,
            outcome_recorded: false,
        };
        let unfocused_snap = MissionSnapshot {
            mission_name: "Unfocused".to_string(),
            bound_region_id: Some(1),
            mission_active: true,
            result: MissionResult::InProgress,
            turns_remaining: 20,
            reactor_integrity: 95.0,
            sabotage_progress: 30.0,
            sabotage_goal: 100.0,
            alert_level: 20.0,
            room_index: 1,
            tactical_mode: TacticalMode::Balanced,
            command_cooldown_turns: 0,
            unattended_turns: 0,
            outcome_recorded: false,
        };
        overwrite_mission_entity(&mut app.world, entities[0], focused_snap);
        overwrite_mission_entity(&mut app.world, entities[1], unfocused_snap);

        for _ in 0..6 {
            app.update();
        }

        let focused = nth_progress(&mut app.world, 0);
        let unfocused = nth_progress(&mut app.world, 1);
        assert!(focused.sabotage_progress > unfocused.sabotage_progress);
        assert!(focused.reactor_integrity >= unfocused.reactor_integrity);
        assert!(focused.alert_level <= unfocused.alert_level);
    }

    #[test]
    fn sustained_focus_consumes_attention_energy() {
        let mut app = build_triage_app();
        let initial_energy = app.world.resource::<AttentionState>().global_energy;

        for _ in 0..5 {
            app.update();
        }

        let attention = app.world.resource::<AttentionState>();
        assert!(attention.global_energy < initial_energy);
    }

    #[test]
    fn unattended_escalation_accelerates_with_time() {
        let mut world = World::new();
        world.insert_resource(RunState { global_turn: 1 });
        world.insert_resource(MissionBoard::default());

        // Spawn active entity (not simulated by simulate_unfocused_missions_system).
        let active_snap = MissionSnapshot {
            mission_name: "Active".to_string(),
            bound_region_id: Some(0),
            mission_active: true,
            result: MissionResult::InProgress,
            turns_remaining: 30,
            reactor_integrity: 100.0,
            sabotage_progress: 0.0,
            sabotage_goal: 100.0,
            alert_level: 4.0,
            room_index: 0,
            tactical_mode: TacticalMode::Balanced,
            command_cooldown_turns: 0,
            unattended_turns: 0,
            outcome_recorded: false,
        };
        let unfocused_snap = MissionSnapshot {
            mission_name: "Unfocused".to_string(),
            bound_region_id: Some(1),
            mission_active: true,
            result: MissionResult::InProgress,
            turns_remaining: 20,
            reactor_integrity: 95.0,
            sabotage_progress: 40.0,
            sabotage_goal: 100.0,
            alert_level: 10.0,
            room_index: 0,
            tactical_mode: TacticalMode::Balanced,
            command_cooldown_turns: 0,
            unattended_turns: 0,
            outcome_recorded: false,
        };
        let (d0, p0, t0) = active_snap.into_components(0);
        let e0 = world
            .spawn((d0, p0, t0, AssignedHero::default(), ActiveMission))
            .id();
        let (d1, p1, t1) = unfocused_snap.into_components(1);
        let e1 = world.spawn((d1, p1, t1, AssignedHero::default())).id();
        world.resource_mut::<MissionBoard>().entities = vec![e0, e1];

        let mut schedule = Schedule::default();
        schedule.add_systems(simulate_unfocused_missions_system);

        let mut previous_progress = world.get::<MissionProgress>(e1).unwrap().sabotage_progress;
        let mut previous_integrity = world.get::<MissionProgress>(e1).unwrap().reactor_integrity;
        let mut first_drop = 0.0_f32;
        let mut last_drop = 0.0_f32;

        for step in 0..6 {
            schedule.run(&mut world);
            let p = world.get::<MissionProgress>(e1).unwrap();
            let progress_drop = previous_progress - p.sabotage_progress;
            let integrity_drop = previous_integrity - p.reactor_integrity;
            if step == 0 {
                first_drop = progress_drop + integrity_drop;
            }
            last_drop = progress_drop + integrity_drop;
            previous_progress = p.sabotage_progress;
            previous_integrity = p.reactor_integrity;
        }

        assert!(last_drop > first_drop);
        let p = world.get::<MissionProgress>(e1).unwrap();
        assert!(p.unattended_turns >= 6);
    }

    #[test]
    fn recruit_generation_is_deterministic_for_same_seed_and_id() {
        let a = generate_recruit(0x1234_5678, 11);
        let b = generate_recruit(0x1234_5678, 11);
        assert_eq!(a.codename, b.codename);
        assert_eq!(a.origin_faction_id, b.origin_faction_id);
        assert_eq!(a.origin_region_id, b.origin_region_id);
        assert_eq!(a.backstory, b.backstory);
        assert_eq!(a.archetype, b.archetype);
        assert_eq!(a.resolve, b.resolve);
        assert_eq!(a.loyalty_bias, b.loyalty_bias);
        assert_eq!(a.risk_tolerance, b.risk_tolerance);
    }

    #[test]
    fn recruit_backstory_references_overworld_faction_and_region() {
        let map = OverworldMap::from_seed(0x0000_BEEF);
        let r = generate_recruit_for_overworld(0x1234_5678, 11, &map);
        let faction_name = map
            .factions
            .get(r.origin_faction_id)
            .map(|f| f.name.as_str())
            .unwrap_or("Unknown");
        let region_name = map
            .regions
            .iter()
            .find(|x| x.id == r.origin_region_id)
            .map(|x| x.name.as_str())
            .unwrap_or("Unknown");
        assert!(r.backstory.contains(faction_name));
        assert!(r.backstory.contains(region_name));
    }

    #[test]
    fn roster_lore_sync_updates_recruit_origins_to_active_overworld() {
        let mut world = World::new();
        world.insert_resource(OverworldMap::from_seed(0x00CA_FE01));
        world.insert_resource(CampaignRoster::default());
        let mut schedule = Schedule::default();
        schedule.add_systems(sync_roster_lore_with_overworld_system);
        schedule.run(&mut world);
        let map = world.resource::<OverworldMap>();
        let roster = world.resource::<CampaignRoster>();
        for recruit in &roster.recruit_pool {
            let faction_name = map
                .factions
                .get(recruit.origin_faction_id)
                .map(|f| f.name.as_str())
                .unwrap_or("Unknown");
            assert!(recruit.backstory.contains(faction_name));
        }
    }

    #[test]
    fn signing_recruit_persists_in_roster_and_refills_pool() {
        let mut roster = CampaignRoster::default();
        let first_id = roster.recruit_pool[0].id;
        let initial_hero_count = roster.heroes.len();
        let initial_pool = roster.recruit_pool.len();

        let signed = sign_top_recruit(&mut roster).expect("expected recruit");

        assert_eq!(signed.id, first_id);
        assert_eq!(roster.heroes.len(), initial_hero_count + 1);
        assert_eq!(roster.recruit_pool.len(), initial_pool);
    }

    #[test]
    fn companion_state_persists_and_modifies_board() {
        let mut app = build_triage_app();
        set_nth_progress(&mut app.world, 0, |p| {
            p.alert_level = 32.0;
            p.sabotage_progress = 25.0;
        });
        let initial = nth_progress(&mut app.world, 0);
        let initial_hero_snapshot = app.world.resource::<CampaignRoster>().heroes[0].clone();

        for _ in 0..5 {
            app.update();
        }

        let mission = nth_progress(&mut app.world, 0);
        assert!(mission.sabotage_progress > initial.sabotage_progress);
        assert!(mission.alert_level <= initial.alert_level + 8.0);

        let roster = app.world.resource::<CampaignRoster>();
        let hero = roster
            .heroes
            .iter()
            .find(|h| h.id == initial_hero_snapshot.id)
            .expect("hero must exist");
        assert!(
            hero.stress != initial_hero_snapshot.stress
                || hero.fatigue != initial_hero_snapshot.fatigue
        );
    }

    #[test]
    fn companion_story_quest_issues_when_pressure_is_high() {
        let mut world = World::new();
        let mut roster = CampaignRoster::default();
        roster.heroes[0].stress = 78.0;
        world.insert_resource(RunState { global_turn: 4 });
        world.insert_resource(MissionBoard::default());
        world.insert_resource(OverworldMap::default());
        world.insert_resource(roster);
        world.insert_resource(CompanionStoryState::default());

        let mut schedule = Schedule::default();
        schedule.add_systems(generate_companion_story_quests_system);
        schedule.run(&mut world);

        let story = world.resource::<CompanionStoryState>();
        assert!(!story.quests.is_empty());
        assert_eq!(story.quests[0].status, CompanionQuestStatus::Active);
    }

    #[test]
    fn companion_story_quest_completion_rewards_hero() {
        let mut world = World::new();
        let mut roster = CampaignRoster::default();
        let hero_id = roster.heroes[0].id;
        roster.heroes[0].loyalty = 40.0;
        roster.heroes[0].resolve = 55.0;
        let base_loyalty = roster.heroes[0].loyalty;
        let base_resolve = roster.heroes[0].resolve;

        let story = CompanionStoryState {
            quests: vec![CompanionQuest {
                id: 1,
                hero_id,
                kind: CompanionQuestKind::Reckoning,
                status: CompanionQuestStatus::Active,
                title: "Test Quest".to_string(),
                objective: "Win once".to_string(),
                progress: 0,
                target: 1,
                issued_turn: 1,
                reward_loyalty: 7.0,
                reward_resolve: 5.0,
            }],
            next_id: 2,
            processed_ledger_len: 0,
            notice: String::new(),
        };
        let ledger = CampaignLedger {
            records: vec![ConsequenceRecord {
                turn: 6,
                mission_name: "Test".to_string(),
                result: MissionResult::Victory,
                hero_id: Some(hero_id),
                summary: "ok".to_string(),
            }],
        };

        world.insert_resource(roster);
        world.insert_resource(ledger);
        world.insert_resource(story);
        let mut schedule = Schedule::default();
        schedule.add_systems(progress_companion_story_quests_system);
        schedule.run(&mut world);

        let story = world.resource::<CompanionStoryState>();
        let quest = &story.quests[0];
        assert_eq!(quest.status, CompanionQuestStatus::Completed);
        let roster = world.resource::<CampaignRoster>();
        let hero = &roster.heroes[0];
        assert!(hero.loyalty > base_loyalty);
        assert!(hero.resolve > base_resolve);
    }

    #[test]
    fn mission_outcome_records_consequence_once() {
        let mut app = build_triage_app();
        set_active_progress(&mut app.world, |p| {
            p.result = MissionResult::Defeat;
            p.mission_active = false;
            p.alert_level = 62.0;
            p.unattended_turns = 8;
        });
        {
            let entity = app.world.resource::<MissionBoard>().entities[0];
            app.world.get_mut::<AssignedHero>(entity).unwrap().hero_id = Some(1);
        }

        app.update();
        app.update();

        let ledger = app.world.resource::<CampaignLedger>();
        assert_eq!(ledger.records.len(), 1);
        let p = active_progress(&mut app.world);
        assert!(p.outcome_recorded);
    }

    #[test]
    fn extreme_defeat_can_cause_desertion() {
        let mut world = World::new();
        world.insert_resource(RunState { global_turn: 9 });
        world.insert_resource(MissionBoard::default());
        world.insert_resource(CampaignLedger::default());

        let mut roster = CampaignRoster::default();
        let hero_id = roster.heroes[0].id;
        roster.heroes[0].loyalty = 8.0;
        roster.heroes[0].stress = 70.0;
        world.insert_resource(roster);

        let snap = MissionSnapshot {
            mission_name: "Overrun".to_string(),
            bound_region_id: Some(0),
            mission_active: false,
            result: MissionResult::Defeat,
            turns_remaining: 0,
            reactor_integrity: 2.0,
            sabotage_progress: 3.0,
            sabotage_goal: 100.0,
            alert_level: 95.0,
            room_index: 2,
            tactical_mode: TacticalMode::Aggressive,
            command_cooldown_turns: 0,
            unattended_turns: 15,
            outcome_recorded: false,
        };
        let (data, progress, tactics) = snap.into_components(0);
        let assigned = AssignedHero { hero_id: Some(hero_id) };
        let entity = world
            .spawn((data, progress, tactics, assigned, ActiveMission))
            .id();
        world.resource_mut::<MissionBoard>().entities.push(entity);

        let mut schedule = Schedule::default();
        schedule.add_systems(resolve_mission_consequences_system);
        schedule.run(&mut world);

        let roster = world.resource::<CampaignRoster>();
        let hero = &roster.heroes[0];
        assert!(hero.deserter);
        assert!(!hero.active);
    }

    #[test]
    fn sidelined_hero_recovers_over_time() {
        let mut world = World::new();
        let mut roster = CampaignRoster::default();
        roster.heroes[0].active = false;
        roster.heroes[0].deserter = false;
        roster.heroes[0].injury = 38.0;
        roster.heroes[0].fatigue = 39.0;
        roster.heroes[0].stress = 35.0;
        world.insert_resource(roster);
        world.insert_resource(RunState { global_turn: 3 });
        let mut schedule = Schedule::default();
        schedule.add_systems(companion_recovery_system);
        schedule.run(&mut world);

        let roster = world.resource::<CampaignRoster>();
        let hero = &roster.heroes[0];
        assert!(hero.active);
    }

    #[test]
    fn overworld_travel_moves_focus_to_linked_region_mission() {
        let mut overworld = OverworldMap::default();
        let mut attention = AttentionState::default();
        let source = overworld.current_region;
        let target = overworld.regions[source].neighbors[0];
        overworld.selected_region = target;
        let slot = try_overworld_travel(&mut overworld, &mut attention);
        assert!(slot.is_some() || overworld.regions[target].mission_slot.is_none());
        assert_eq!(overworld.current_region, target);
        if let Some(expected_slot) = overworld.regions[target].mission_slot {
            assert_eq!(slot, Some(expected_slot));
        }
        assert!(attention.global_energy < attention.max_energy);
    }

    #[test]
    fn overworld_travel_blocks_when_not_neighbor_or_low_energy() {
        let mut overworld = OverworldMap::default();
        let mut attention = AttentionState::default();
        let source = overworld.current_region;
        let non_neighbor = overworld
            .regions
            .iter()
            .find(|r| r.id != source && !overworld.regions[source].neighbors.contains(&r.id))
            .map(|r| r.id)
            .expect("non-neighbor region");
        overworld.selected_region = non_neighbor;
        assert!(try_overworld_travel(&mut overworld, &mut attention).is_none());
        overworld.selected_region = overworld.regions[source].neighbors[0];
        attention.global_energy = 0.0;
        assert!(try_overworld_travel(&mut overworld, &mut attention).is_none());
    }

    #[test]
    fn overworld_sync_tracks_mission_pressure() {
        let mut world = World::new();
        world.insert_resource(MissionBoard::default());
        world.insert_resource(OverworldMap::default());

        // Spawn 3 default mission entities, then modify entity 1.
        spawn_test_missions(&mut world);
        set_nth_progress(&mut world, 1, |p| {
            p.alert_level = 70.0;
            p.reactor_integrity = 40.0;
            p.sabotage_progress = 20.0;
        });

        let mut schedule = Schedule::default();
        schedule.add_systems(overworld_sync_from_missions_system);
        schedule.run(&mut world);
        let overworld = world.resource::<OverworldMap>();
        let linked = overworld
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(1))
            .expect("region");
        assert!(linked.unrest >= 20.0);
        assert!(linked.control <= 80.0);
    }

    #[test]
    fn faction_vassal_count_scales_with_strength() {
        assert!(target_vassal_count(40.0) < target_vassal_count(120.0));
    }

    #[test]
    fn faction_autonomy_rebalances_vassals_after_strength_shift() {
        let mut world = World::new();
        let mut map = OverworldMap::default();
        map.factions[0].strength = 150.0;
        map.factions[0].vassals.clear();
        world.insert_resource(map);
        world.insert_resource(RunState { global_turn: 2 });
        let mut schedule = Schedule::default();
        schedule.add_systems(overworld_faction_autonomy_system);
        schedule.run(&mut world);
        let map = world.resource::<OverworldMap>();
        assert!(map.factions[0].vassals.len() >= 7);
    }

    #[test]
    fn overworld_default_is_deterministic_for_factions_and_vassals() {
        let a = OverworldMap::default();
        let b = OverworldMap::default();
        assert_eq!(a.map_seed, b.map_seed);
        assert_eq!(a.regions.len(), b.regions.len());
        for idx in 0..a.regions.len() {
            let ar = &a.regions[idx];
            let br = &b.regions[idx];
            assert_eq!(ar.name, br.name);
            assert_eq!(ar.neighbors, br.neighbors);
            assert_eq!(ar.owner_faction_id, br.owner_faction_id);
            assert_eq!(ar.mission_slot, br.mission_slot);
        }
        assert_eq!(a.factions.len(), b.factions.len());
        for idx in 0..a.factions.len() {
            let af = &a.factions[idx];
            let bf = &b.factions[idx];
            assert_eq!(af.name, bf.name);
            assert_eq!(af.vassals.len(), bf.vassals.len());
            let an = af
                .vassals
                .iter()
                .map(|v| v.name.clone())
                .collect::<Vec<_>>();
            let bn = bf
                .vassals
                .iter()
                .map(|v| v.name.clone())
                .collect::<Vec<_>>();
            assert_eq!(an, bn);
        }
    }

    #[test]
    fn factions_have_stationary_zone_managers() {
        let map = OverworldMap::default();
        for faction in &map.factions {
            let managers = faction
                .vassals
                .iter()
                .filter(|v| v.post == VassalPost::ZoneManager)
                .collect::<Vec<_>>();
            assert!(!managers.is_empty());
            for manager in managers {
                let owns_home = map
                    .regions
                    .iter()
                    .any(|r| r.id == manager.home_region_id && r.owner_faction_id == faction.id);
                assert!(owns_home);
            }
        }
    }

    #[test]
    fn seeded_overworld_changes_with_seed() {
        let a = OverworldMap::from_seed(11);
        let b = OverworldMap::from_seed(12);
        assert_eq!(a.regions.len(), b.regions.len());
        let owner_vec_a = a
            .regions
            .iter()
            .map(|r| r.owner_faction_id)
            .collect::<Vec<_>>();
        let owner_vec_b = b
            .regions
            .iter()
            .map(|r| r.owner_faction_id)
            .collect::<Vec<_>>();
        assert_ne!(owner_vec_a, owner_vec_b);
    }

    #[test]
    fn seeded_overworld_has_bidirectional_neighbors_and_faction_presence() {
        let map = OverworldMap::from_seed(0x1234_5678);
        for region in &map.regions {
            assert!(!region.neighbors.is_empty());
            for neighbor in &region.neighbors {
                let other = map.regions.get(*neighbor).expect("neighbor exists");
                assert!(other.neighbors.contains(&region.id));
            }
        }
        for faction_id in 0..map.factions.len() {
            assert!(map.regions.iter().any(|r| r.owner_faction_id == faction_id));
        }
    }

    #[test]
    fn war_goals_are_assigned_to_other_factions() {
        let mut world = World::new();
        world.insert_resource(OverworldMap::default());
        world.insert_resource(DiplomacyState::default());
        let mut schedule = Schedule::default();
        schedule.add_systems(update_faction_war_goals_system);
        schedule.run(&mut world);
        let map = world.resource::<OverworldMap>();
        for faction in &map.factions {
            let goal = faction.war_goal_faction_id.expect("war goal");
            assert_ne!(goal, faction.id);
            assert!(goal < map.factions.len());
            assert!((0.0..=100.0).contains(&faction.war_focus));
        }
    }

    #[test]
    fn border_pressure_is_deterministic() {
        let mut world_a = World::new();
        world_a.insert_resource(OverworldMap::default());
        world_a.insert_resource(RunState { global_turn: 6 });
        let mut schedule_a = Schedule::default();
        schedule_a.add_systems(overworld_ai_border_pressure_system);
        schedule_a.run(&mut world_a);
        let map_a = world_a.resource::<OverworldMap>();
        let sig_a = map_a
            .regions
            .iter()
            .map(|r| {
                format!(
                    "{}:{:.1}:{:.1}:{}",
                    r.id, r.unrest, r.control, r.owner_faction_id
                )
            })
            .collect::<Vec<_>>();

        let mut world_b = World::new();
        world_b.insert_resource(OverworldMap::default());
        world_b.insert_resource(RunState { global_turn: 6 });
        let mut schedule_b = Schedule::default();
        schedule_b.add_systems(overworld_ai_border_pressure_system);
        schedule_b.run(&mut world_b);
        let map_b = world_b.resource::<OverworldMap>();
        let sig_b = map_b
            .regions
            .iter()
            .map(|r| {
                format!(
                    "{}:{:.1}:{:.1}:{}",
                    r.id, r.unrest, r.control, r.owner_faction_id
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(sig_a, sig_b);
    }

    #[test]
    fn border_pressure_can_shift_region_ownership_when_defender_has_depth() {
        let mut world = World::new();
        let mut map = OverworldMap::default();
        let mut chosen = None;
        for rid in 0..map.regions.len() {
            let owner = map.regions[rid].owner_faction_id;
            for neighbor in &map.regions[rid].neighbors {
                let attacker = map.regions[*neighbor].owner_faction_id;
                if attacker == owner {
                    continue;
                }
                let owner_count = map
                    .regions
                    .iter()
                    .filter(|r| r.owner_faction_id == owner)
                    .count();
                if owner_count > 1 {
                    chosen = Some((rid, owner, attacker));
                    break;
                }
            }
            if chosen.is_some() {
                break;
            }
        }
        let (target_rid, defender, attacker) = chosen.expect("border candidate");
        map.regions[target_rid].control = 5.0;
        map.regions[target_rid].unrest = 92.0;
        map.factions[attacker].strength = 175.0;
        map.factions[attacker].cohesion = 90.0;
        map.factions[attacker].war_focus = 95.0;
        map.factions[defender].strength = 38.0;
        map.factions[defender].cohesion = 18.0;
        map.factions[defender].war_focus = 5.0;
        world.insert_resource(map);
        world.insert_resource(RunState { global_turn: 9 });

        let mut schedule = Schedule::default();
        schedule.add_systems(overworld_ai_border_pressure_system);
        schedule.run(&mut world);
        let map = world.resource::<OverworldMap>();
        assert_eq!(map.regions[target_rid].owner_faction_id, attacker);
    }

    #[test]
    fn intel_update_prioritizes_current_region_and_neighbors() {
        let mut world = World::new();
        let mut map = OverworldMap::default();
        for region in &mut map.regions {
            region.intel_level = 20.0;
        }
        let current = map.current_region;
        let neighbor = map.regions[current].neighbors[0];
        let far = map
            .regions
            .iter()
            .find(|r| r.id != current && !map.regions[current].neighbors.contains(&r.id))
            .map(|r| r.id)
            .expect("far region");
        map.selected_region = far;
        world.insert_resource(map);
        world.insert_resource(RunState { global_turn: 4 });
        world.insert_resource(MissionBoard::default());

        let mut schedule = Schedule::default();
        schedule.add_systems(overworld_intel_update_system);
        schedule.run(&mut world);
        let map = world.resource::<OverworldMap>();
        assert!(map.regions[current].intel_level > map.regions[far].intel_level);
        assert!(map.regions[neighbor].intel_level > map.regions[far].intel_level);
    }

    #[test]
    fn pressure_spawn_binds_each_slot_to_current_region_anchor() {
        let mut world = World::new();
        world.insert_resource(RunState { global_turn: 3 });
        world.insert_resource(OverworldMap::default());
        world.insert_resource(MissionBoard::default());
        spawn_test_missions(&mut world);

        let mut schedule = Schedule::default();
        schedule.add_systems(pressure_spawn_missions_system);
        schedule.run(&mut world);

        let map = world.resource::<OverworldMap>();
        let board_entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        let max_slot = usize::min(map.factions.len(), board_entities.len());
        for slot in 0..max_slot {
            let region = map
                .regions
                .iter()
                .find(|r| r.mission_slot == Some(slot))
                .expect("region per slot");
            let entity = board_entities[slot];
            let data = world.get::<MissionData>(entity).unwrap();
            let progress = world.get::<MissionProgress>(entity).unwrap();
            assert_eq!(progress.mission_active || !data.mission_name.is_empty(), true);
            assert_eq!(data.bound_region_id, Some(region.id));
            assert!(data.mission_name.contains(&region.name));
        }
    }

    #[test]
    fn pressure_spawn_replaces_resolved_slot_mission() {
        let mut world = World::new();
        let map = OverworldMap::default();
        world.insert_resource(RunState { global_turn: 5 });
        world.insert_resource(map);
        world.insert_resource(MissionBoard::default());
        spawn_test_missions(&mut world);

        // Mark entity 1 as defeated.
        set_nth_progress(&mut world, 1, |p| {
            p.mission_active = false;
            p.result = MissionResult::Defeat;
        });
        {
            let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
            world.get_mut::<MissionData>(entities[1]).unwrap().mission_name =
                "Old Mission".to_string();
        }

        let mut schedule = Schedule::default();
        schedule.add_systems(pressure_spawn_missions_system);
        schedule.run(&mut world);

        let map = world.resource::<OverworldMap>();
        let board_entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        let region = map
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(1))
            .expect("slot region");
        let entity1 = board_entities[1];
        let data = world.get::<MissionData>(entity1).unwrap();
        let progress = world.get::<MissionProgress>(entity1).unwrap();
        assert!(progress.mission_active);
        assert_eq!(progress.result, MissionResult::InProgress);
        assert_ne!(data.mission_name, "Old Mission");
        assert_eq!(data.bound_region_id, Some(region.id));
    }

    #[test]
    fn pressure_spawn_can_open_flashpoint_chain() {
        let mut world = World::new();
        let mut map = OverworldMap::default();
        let slot = 0usize;
        let rid = map
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
            .map(|r| r.id)
            .expect("slot region");
        map.regions[rid].unrest = 96.0;
        map.regions[rid].control = 7.0;

        world.insert_resource(RunState { global_turn: 9 });
        world.insert_resource(map);
        world.insert_resource(MissionBoard::default());
        world.insert_resource(FlashpointState::default());
        spawn_test_missions(&mut world);

        let mut schedule = Schedule::default();
        schedule.add_systems(pressure_spawn_missions_system);
        schedule.run(&mut world);

        let board_entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        let flashpoints = world.resource::<FlashpointState>();
        assert!(!flashpoints.chains.is_empty());
        assert!(flashpoints.chains.iter().any(|c| c.mission_slot == slot));
        let data = world.get::<MissionData>(board_entities[slot]).unwrap();
        assert!(data.mission_name.contains("Flashpoint 1/3"));
    }

    #[test]
    fn flashpoint_stage_victory_promotes_next_stage() {
        let mut world = World::new();
        let map = OverworldMap::default();
        let slot = 0usize;
        let rid = map
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
            .map(|r| r.id)
            .expect("slot region");
        let defender = map.regions[rid].owner_faction_id;
        let attacker = map.regions[rid]
            .neighbors
            .iter()
            .find_map(|n| {
                let f = map.regions[*n].owner_faction_id;
                if f != defender {
                    Some(f)
                } else {
                    None
                }
            })
            .expect("attacker");

        world.insert_resource(RunState { global_turn: 12 });
        world.insert_resource(MissionBoard::default());
        world.insert_resource(map);
        world.insert_resource(CampaignRoster::default());
        world.insert_resource(FlashpointState {
            chains: vec![FlashpointChain {
                id: 1,
                mission_slot: slot,
                region_id: rid,
                attacker_faction_id: attacker,
                defender_faction_id: defender,
                stage: 1,
                completed: false,
                companion_hook_hero_id: None,
                intent: FlashpointIntent::StealthPush,
                objective: String::new(),
            }],
            next_id: 2,
            notice: String::new(),
        });
        spawn_test_missions(&mut world);

        // Configure entity for slot as stage-1 victory.
        let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        {
            let mut data = world.get_mut::<MissionData>(entities[slot]).unwrap();
            data.mission_name = "Flashpoint 1/3 [Recon Sweep]".to_string();
            data.bound_region_id = Some(rid);
        }
        {
            let mut progress = world.get_mut::<MissionProgress>(entities[slot]).unwrap();
            progress.mission_active = false;
            progress.result = MissionResult::Victory;
            progress.outcome_recorded = true;
        }

        let mut schedule = Schedule::default();
        schedule.add_systems(flashpoint_progression_system);
        schedule.run(&mut world);

        let flashpoints = world.resource::<FlashpointState>();
        assert_eq!(flashpoints.chains[0].stage, 2);
        let progress = world.get::<MissionProgress>(entities[slot]).unwrap();
        assert_eq!(progress.result, MissionResult::InProgress);
        let data = world.get::<MissionData>(entities[slot]).unwrap();
        assert!(data.mission_name.contains("Flashpoint 2/3"));
    }

    #[test]
    fn flashpoint_stage_hook_applies_companion_homefront_tuning() {
        let mut world = World::new();
        let map = OverworldMap::default();
        let slot = 0usize;
        let rid = map
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
            .map(|r| r.id)
            .expect("slot region");
        let defender = map.regions[rid].owner_faction_id;
        let attacker = map.regions[rid]
            .neighbors
            .iter()
            .find_map(|n| {
                let f = map.regions[*n].owner_faction_id;
                if f != defender {
                    Some(f)
                } else {
                    None
                }
            })
            .expect("attacker");

        let mut roster = CampaignRoster::default();
        let hero_id = roster.heroes[0].id;
        roster.heroes[0].origin_region_id = rid;

        world.insert_resource(RunState { global_turn: 13 });
        world.insert_resource(MissionBoard::default());
        world.insert_resource(map);
        world.insert_resource(roster);
        world.insert_resource(FlashpointState {
            chains: vec![FlashpointChain {
                id: 3,
                mission_slot: slot,
                region_id: rid,
                attacker_faction_id: attacker,
                defender_faction_id: defender,
                stage: 1,
                completed: false,
                companion_hook_hero_id: Some(hero_id),
                intent: FlashpointIntent::StealthPush,
                objective: String::new(),
            }],
            next_id: 4,
            notice: String::new(),
        });
        spawn_test_missions(&mut world);

        let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        {
            let mut data = world.get_mut::<MissionData>(entities[slot]).unwrap();
            data.bound_region_id = Some(rid);
        }
        {
            let mut progress = world.get_mut::<MissionProgress>(entities[slot]).unwrap();
            progress.mission_active = false;
            progress.result = MissionResult::Victory;
            progress.outcome_recorded = true;
        }

        let mut schedule = Schedule::default();
        schedule.add_systems(flashpoint_progression_system);
        schedule.run(&mut world);

        let data = world.get::<MissionData>(entities[slot]).unwrap();
        assert!(data.mission_name.contains("obj="));
        assert!(!data.mission_name.contains("No companion hook"));
    }

    #[test]
    fn decisive_flashpoint_victory_shifts_border_and_unlocks_recruit() {
        let mut world = World::new();
        let map = OverworldMap::default();
        let slot = 0usize;
        let rid = map
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
            .map(|r| r.id)
            .expect("slot region");
        let defender = map.regions[rid].owner_faction_id;
        let attacker = map.regions[rid]
            .neighbors
            .iter()
            .find_map(|n| {
                let f = map.regions[*n].owner_faction_id;
                if f != defender {
                    Some(f)
                } else {
                    None
                }
            })
            .expect("attacker");

        let mut roster = CampaignRoster::default();
        roster.recruit_pool.clear();

        world.insert_resource(RunState { global_turn: 18 });
        world.insert_resource(MissionBoard::default());
        world.insert_resource(map);
        world.insert_resource(roster);
        world.insert_resource(FlashpointState {
            chains: vec![FlashpointChain {
                id: 8,
                mission_slot: slot,
                region_id: rid,
                attacker_faction_id: attacker,
                defender_faction_id: defender,
                stage: 3,
                completed: false,
                companion_hook_hero_id: None,
                intent: FlashpointIntent::StealthPush,
                objective: String::new(),
            }],
            next_id: 9,
            notice: String::new(),
        });
        spawn_test_missions(&mut world);

        let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        {
            let mut data = world.get_mut::<MissionData>(entities[slot]).unwrap();
            data.bound_region_id = Some(rid);
        }
        {
            let mut progress = world.get_mut::<MissionProgress>(entities[slot]).unwrap();
            progress.mission_active = false;
            progress.result = MissionResult::Victory;
            progress.outcome_recorded = true;
        }

        let mut schedule = Schedule::default();
        schedule.add_systems(flashpoint_progression_system);
        schedule.run(&mut world);

        let map = world.resource::<OverworldMap>();
        let roster = world.resource::<CampaignRoster>();
        let flashpoints = world.resource::<FlashpointState>();
        assert_eq!(map.regions[rid].owner_faction_id, attacker);
        assert!(roster
            .recruit_pool
            .iter()
            .any(|r| r.origin_faction_id == attacker));
        assert!(flashpoints.chains.is_empty());
    }

    #[test]
    fn flashpoint_intent_input_updates_stage_profile_and_telemetry() {
        let mut world = World::new();
        let map = OverworldMap::default();
        let slot = 0usize;
        let rid = map
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
            .map(|r| r.id)
            .expect("slot region");
        let defender = map.regions[rid].owner_faction_id;
        let attacker = map.regions[rid]
            .neighbors
            .iter()
            .find_map(|n| {
                let f = map.regions[*n].owner_faction_id;
                if f != defender {
                    Some(f)
                } else {
                    None
                }
            })
            .expect("attacker");

        let chain = FlashpointChain {
            id: 21,
            mission_slot: slot,
            region_id: rid,
            attacker_faction_id: attacker,
            defender_faction_id: defender,
            stage: 2,
            completed: false,
            companion_hook_hero_id: None,
            intent: FlashpointIntent::StealthPush,
            objective: String::new(),
        };

        world.insert_resource(RunState { global_turn: 33 });
        world.insert_resource(MissionBoard::default());
        world.insert_resource(map);
        world.insert_resource(FlashpointState {
            chains: vec![chain.clone()],
            next_id: 22,
            notice: String::new(),
        });
        let mut keyboard = ButtonInput::<KeyCode>::default();
        keyboard.press(KeyCode::Digit2);
        world.insert_resource(keyboard);
        spawn_test_missions(&mut world);

        // Configure the slot entity as an active flashpoint stage mission.
        let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        {
            let data_ref = world.get::<MissionData>(entities[slot]).unwrap();
            let progress_ref = world.get::<MissionProgress>(entities[slot]).unwrap();
            let tactics_ref = world.get::<MissionTactics>(entities[slot]).unwrap();
            let mut snap = MissionSnapshot::from_components(data_ref, progress_ref, tactics_ref);
            let overworld = world.resource::<OverworldMap>();
            configure_flashpoint_stage_mission(&mut snap, &chain, overworld, 77);
            rewrite_flashpoint_mission_name(&mut snap, &chain, overworld, None);
            let id = data_ref.id;
            drop(data_ref);
            drop(progress_ref);
            drop(tactics_ref);
            drop(overworld);
            let (new_data, new_progress, new_tactics) = snap.into_components(id);
            *world.get_mut::<MissionData>(entities[slot]).unwrap() = new_data;
            *world.get_mut::<MissionProgress>(entities[slot]).unwrap() = new_progress;
            *world.get_mut::<MissionTactics>(entities[slot]).unwrap() = new_tactics;
        }
        let before_alert = world.get::<MissionProgress>(entities[slot]).unwrap().alert_level;

        let mut schedule = Schedule::default();
        schedule.add_systems(flashpoint_intent_input_system);
        schedule.run(&mut world);

        let chain_result = &world.resource::<FlashpointState>().chains[0];
        assert_eq!(chain_result.intent, FlashpointIntent::DirectAssault);
        let data = world.get::<MissionData>(entities[slot]).unwrap();
        assert!(data.mission_name.contains("Direct Assault"));
        assert!(data.mission_name.contains("win=>"));
        let progress = world.get::<MissionProgress>(entities[slot]).unwrap();
        assert!(progress.alert_level > before_alert);
    }

    #[test]
    fn flashpoint_victory_advances_hooked_companion_quest() {
        let mut world = World::new();
        let map = OverworldMap::default();
        let slot = 0usize;
        let rid = map
            .regions
            .iter()
            .find(|r| r.mission_slot == Some(slot))
            .map(|r| r.id)
            .expect("slot region");
        let defender = map.regions[rid].owner_faction_id;
        let attacker = map.regions[rid]
            .neighbors
            .iter()
            .find_map(|n| {
                let f = map.regions[*n].owner_faction_id;
                if f != defender {
                    Some(f)
                } else {
                    None
                }
            })
            .expect("attacker");

        let mut roster = CampaignRoster::default();
        let hero_id = roster.heroes[0].id;
        roster.heroes[0].origin_region_id = rid;

        let mut story = CompanionStoryState::default();
        story.quests.push(CompanionQuest {
            id: 1,
            hero_id,
            kind: CompanionQuestKind::Homefront,
            status: CompanionQuestStatus::Active,
            title: "Test Quest".to_string(),
            objective: "Do a thing".to_string(),
            progress: 0,
            target: 2,
            issued_turn: 1,
            reward_loyalty: 1.0,
            reward_resolve: 1.0,
        });

        world.insert_resource(RunState { global_turn: 40 });
        world.insert_resource(MissionBoard::default());
        world.insert_resource(map);
        world.insert_resource(roster);
        world.insert_resource(story);
        world.insert_resource(FlashpointState {
            chains: vec![FlashpointChain {
                id: 44,
                mission_slot: slot,
                region_id: rid,
                attacker_faction_id: attacker,
                defender_faction_id: defender,
                stage: 3,
                completed: false,
                companion_hook_hero_id: Some(hero_id),
                intent: FlashpointIntent::StealthPush,
                objective: "hook objective".to_string(),
            }],
            next_id: 45,
            notice: String::new(),
        });
        spawn_test_missions(&mut world);

        let entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        {
            let mut data = world.get_mut::<MissionData>(entities[slot]).unwrap();
            data.bound_region_id = Some(rid);
        }
        {
            let mut progress = world.get_mut::<MissionProgress>(entities[slot]).unwrap();
            progress.mission_active = false;
            progress.result = MissionResult::Victory;
            progress.outcome_recorded = true;
        }

        let mut schedule = Schedule::default();
        schedule.add_systems(flashpoint_progression_system);
        schedule.run(&mut world);

        let story = world.resource::<CompanionStoryState>();
        assert_eq!(story.quests[0].progress, 1);
        assert!(story.notice.contains("Flashpoint beat advanced companion quest"));
    }

    #[test]
    fn commander_intents_are_deterministic() {
        let mut world_a = World::new();
        world_a.insert_resource(OverworldMap::default());
        world_a.insert_resource(MissionBoard::default());
        world_a.insert_resource(DiplomacyState::default());
        world_a.insert_resource(CommanderState::default());
        let mut schedule = Schedule::default();
        schedule.add_systems(generate_commander_intents_system);
        schedule.run(&mut world_a);
        let intents_a = world_a.resource::<CommanderState>().intents.clone();

        let mut world_b = World::new();
        world_b.insert_resource(OverworldMap::default());
        world_b.insert_resource(MissionBoard::default());
        world_b.insert_resource(DiplomacyState::default());
        world_b.insert_resource(CommanderState::default());
        let mut schedule_b = Schedule::default();
        schedule_b.add_systems(generate_commander_intents_system);
        schedule_b.run(&mut world_b);
        let intents_b = world_b.resource::<CommanderState>().intents.clone();

        assert_eq!(intents_a.len(), intents_b.len());
        for i in 0..intents_a.len() {
            assert_eq!(intents_a[i].faction_id, intents_b[i].faction_id);
            assert_eq!(intents_a[i].region_id, intents_b[i].region_id);
            assert_eq!(intents_a[i].mission_slot, intents_b[i].mission_slot);
            assert_eq!(intents_a[i].kind, intents_b[i].kind);
            assert_eq!(intents_a[i].urgency, intents_b[i].urgency);
        }
    }

    #[test]
    fn interaction_offer_acceptance_changes_state() {
        let offer = InteractionOffer {
            id: 7,
            from_faction_id: 1,
            region_id: 1,
            mission_slot: Some(1),
            kind: InteractionOfferKind::JointMission,
            summary: "test".to_string(),
        };
        let mut missions = default_mission_snapshots();
        missions[1].alert_level = 44.0;
        let mut attention = AttentionState::default();
        let mut roster = CampaignRoster::default();
        let mut diplomacy = DiplomacyState::default();

        let msg = resolve_interaction_offer(
            &offer,
            true,
            &mut missions,
            &mut attention,
            &mut roster,
            &mut diplomacy,
        );
        assert!(msg.contains("Joint mission accepted"));
        assert!(missions[1].alert_level < 44.0);
        assert!(diplomacy.relations[0][1] > 10);
    }

    fn campaign_signature(world: &mut World) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        let mix = |h: &mut u64, v: u64| {
            *h ^= v;
            *h = h.wrapping_mul(0x1000_0000_01b3);
        };

        let board_entities: Vec<Entity> = world.resource::<MissionBoard>().entities.clone();
        let active_idx = board_entities
            .iter()
            .position(|&e| world.get::<ActiveMission>(e).is_some())
            .unwrap_or(0);
        mix(&mut h, active_idx as u64);

        for &entity in &board_entities {
            if let Some(p) = world.get::<MissionProgress>(entity) {
                mix(&mut h, p.turns_remaining as u64);
                mix(&mut h, (p.sabotage_progress * 10.0) as u64);
                mix(&mut h, (p.alert_level * 10.0) as u64);
                mix(&mut h, (p.reactor_integrity * 10.0) as u64);
                mix(&mut h, p.unattended_turns as u64);
                mix(&mut h, p.outcome_recorded as u64);
                mix(&mut h, p.result as u64);
            }
        }

        let hero_data: Vec<(u32, f32, f32, f32, f32, bool, bool)> = world
            .resource::<CampaignRoster>()
            .heroes
            .iter()
            .map(|h| (h.id, h.loyalty, h.stress, h.fatigue, h.injury, h.active, h.deserter))
            .collect();
        for (id, loyalty, stress, fatigue, injury, active, deserter) in hero_data {
            mix(&mut h, id as u64);
            mix(&mut h, (loyalty * 10.0) as u64);
            mix(&mut h, (stress * 10.0) as u64);
            mix(&mut h, (fatigue * 10.0) as u64);
            mix(&mut h, (injury * 10.0) as u64);
            mix(&mut h, active as u64);
            mix(&mut h, deserter as u64);
        }
        let ledger_len = world.resource::<CampaignLedger>().records.len();
        mix(&mut h, ledger_len as u64);
        h
    }

    #[test]
    fn campaign_cycle_regression_snapshot() {
        // Test that the campaign simulation is deterministic (two identical runs produce the same hash).
        let run_scenario = || {
            let mut app = build_triage_app();
            for _ in 0..14 {
                app.update();
            }
            set_nth_progress(&mut app.world, 1, |p| {
                p.mission_active = false;
                p.result = MissionResult::Defeat;
                p.alert_level = 66.0;
                p.unattended_turns = 9;
            });
            for _ in 0..2 {
                app.update();
            }
            campaign_signature(&mut app.world)
        };
        let sig_a = run_scenario();
        let sig_b = run_scenario();
        assert_eq!(sig_a, sig_b, "campaign cycle must be deterministic");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Input simulation helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn press(key: KeyCode) -> ButtonInput<KeyCode> {
        let mut kb = ButtonInput::<KeyCode>::default();
        kb.press(key);
        kb
    }

    fn run_player_command(world: &mut World, key: KeyCode) {
        world.insert_resource(press(key));
        let mut s = Schedule::default();
        s.add_systems(player_command_input_system);
        s.run(world);
    }

    fn run_overworld_hub(world: &mut World, key: KeyCode) {
        world.insert_resource(press(key));
        let mut s = Schedule::default();
        s.add_systems(overworld_hub_input_system);
        s.run(world);
    }

    fn run_interaction_offer(world: &mut World, key: KeyCode) {
        world.insert_resource(press(key));
        let mut s = Schedule::default();
        s.add_systems(interaction_offer_input_system);
        s.run(world);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // player_command_input_system
    // ─────────────────────────────────────────────────────────────────────────

    fn player_command_world() -> World {
        let mut world = World::new();
        world.init_resource::<MissionBoard>();
        spawn_test_missions(&mut world);
        set_active_progress(&mut world, |p| p.mission_active = true);
        world
    }

    #[test]
    fn player_command_digit1_switches_to_balanced() {
        let mut world = player_command_world();
        set_active_tactics(&mut world, |t| t.tactical_mode = TacticalMode::Aggressive);
        run_player_command(&mut world, KeyCode::Digit1);
        assert_eq!(active_tactics(&mut world).tactical_mode, TacticalMode::Balanced);
    }

    #[test]
    fn player_command_digit2_switches_to_aggressive() {
        let mut world = player_command_world();
        run_player_command(&mut world, KeyCode::Digit2);
        assert_eq!(active_tactics(&mut world).tactical_mode, TacticalMode::Aggressive);
    }

    #[test]
    fn player_command_digit3_switches_to_defensive() {
        let mut world = player_command_world();
        run_player_command(&mut world, KeyCode::Digit3);
        assert_eq!(active_tactics(&mut world).tactical_mode, TacticalMode::Defensive);
    }

    #[test]
    fn player_command_keyb_issues_breach_order() {
        let mut world = player_command_world();
        run_player_command(&mut world, KeyCode::KeyB);
        let t = active_tactics(&mut world);
        assert!(t.force_sabotage_order);
        assert_eq!(t.command_cooldown_turns, 3);
    }

    #[test]
    fn player_command_keyr_issues_regroup_order() {
        let mut world = player_command_world();
        run_player_command(&mut world, KeyCode::KeyR);
        let t = active_tactics(&mut world);
        assert!(t.force_stabilize_order);
        assert_eq!(t.command_cooldown_turns, 3);
    }

    #[test]
    fn player_command_keyb_blocked_by_cooldown() {
        let mut world = player_command_world();
        set_active_tactics(&mut world, |t| t.command_cooldown_turns = 1);
        run_player_command(&mut world, KeyCode::KeyB);
        assert!(!active_tactics(&mut world).force_sabotage_order);
    }

    #[test]
    fn player_command_noop_when_mission_inactive() {
        let mut world = player_command_world();
        set_active_progress(&mut world, |p| p.mission_active = false);
        run_player_command(&mut world, KeyCode::Digit2);
        assert_eq!(active_tactics(&mut world).tactical_mode, TacticalMode::Balanced);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // focus_input_system
    // ─────────────────────────────────────────────────────────────────────────

    fn focus_app(attention: AttentionState) -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.init_resource::<MissionBoard>();
        app.insert_resource(attention);
        app.add_systems(Update, focus_input_system);
        spawn_test_missions(&mut app.world);
        app
    }

    #[test]
    fn focus_input_tab_advances_active_mission() {
        let mut app = focus_app(AttentionState::default());
        app.insert_resource(press(KeyCode::Tab));
        app.update();
        assert_eq!(board_active_idx(&mut app.world), 1);
    }

    #[test]
    fn focus_input_bracket_right_advances_active_mission() {
        let mut app = focus_app(AttentionState::default());
        app.insert_resource(press(KeyCode::BracketRight));
        app.update();
        assert_eq!(board_active_idx(&mut app.world), 1);
    }

    #[test]
    fn focus_input_bracket_left_retreats_active_mission() {
        let mut app = focus_app(AttentionState::default());
        // Move active marker to slot 1 so we can retreat to slot 0.
        let entities = app.world.resource::<MissionBoard>().entities.clone();
        app.world.entity_mut(entities[0]).remove::<ActiveMission>();
        app.world.entity_mut(entities[1]).insert(ActiveMission);
        app.insert_resource(press(KeyCode::BracketLeft));
        app.update();
        assert_eq!(board_active_idx(&mut app.world), 0);
    }

    #[test]
    fn focus_input_blocked_by_switch_cooldown() {
        let mut app = focus_app(AttentionState { switch_cooldown_turns: 1, ..AttentionState::default() });
        app.insert_resource(press(KeyCode::Tab));
        app.update();
        assert_eq!(board_active_idx(&mut app.world), 0, "cooldown should block focus shift");
    }

    #[test]
    fn focus_input_blocked_by_low_energy() {
        let mut app = focus_app(AttentionState { global_energy: 0.0, ..AttentionState::default() });
        app.insert_resource(press(KeyCode::Tab));
        app.update();
        assert_eq!(board_active_idx(&mut app.world), 0, "insufficient energy should block focus shift");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // overworld_hub_input_system
    // ─────────────────────────────────────────────────────────────────────────

    fn overworld_hub_world() -> World {
        let mut world = World::new();
        world.init_resource::<MissionBoard>();
        world.insert_resource(OverworldMap::default());
        world.insert_resource(AttentionState::default());
        spawn_test_missions(&mut world);
        world
    }

    #[test]
    fn overworld_hub_keyl_sets_selected_to_neighbor() {
        let mut world = overworld_hub_world();
        run_overworld_hub(&mut world, KeyCode::KeyL);
        let overworld = world.resource::<OverworldMap>();
        let current = overworld.current_region;
        assert!(
            overworld.regions[current].neighbors.contains(&overworld.selected_region),
            "KeyL should set selected_region to a neighbor of current_region"
        );
    }

    #[test]
    fn overworld_hub_keyj_sets_selected_to_neighbor() {
        let mut world = overworld_hub_world();
        run_overworld_hub(&mut world, KeyCode::KeyJ);
        let overworld = world.resource::<OverworldMap>();
        let current = overworld.current_region;
        assert!(
            overworld.regions[current].neighbors.contains(&overworld.selected_region),
            "KeyJ should set selected_region to a neighbor of current_region"
        );
    }

    #[test]
    fn overworld_hub_keyt_commits_travel_to_selected_region() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.init_resource::<MissionBoard>();
        app.add_systems(Update, overworld_hub_input_system);

        let mut overworld = OverworldMap::default();
        let current = overworld.current_region;
        let Some(&target) = overworld.regions[current].neighbors.first() else {
            return; // degenerate map; skip
        };
        overworld.selected_region = target;
        overworld.travel_cooldown_turns = 0;
        app.insert_resource(overworld);
        app.insert_resource(AttentionState { global_energy: 9999.0, ..AttentionState::default() });
        spawn_test_missions(&mut app.world);

        app.insert_resource(press(KeyCode::KeyT));
        app.update();

        assert_eq!(
            app.world.resource::<OverworldMap>().current_region,
            target,
            "KeyT should commit travel and update current_region"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // interaction_offer_input_system
    // ─────────────────────────────────────────────────────────────────────────

    fn make_interaction_world() -> World {
        let mut world = World::new();
        world.init_resource::<MissionBoard>();
        world.insert_resource(AttentionState::default());
        world.insert_resource(CampaignRoster::default());
        world.insert_resource(DiplomacyState::default());
        world.insert_resource(InteractionBoard {
            offers: vec![
                InteractionOffer {
                    id: 1,
                    from_faction_id: 1,
                    region_id: 0,
                    mission_slot: None,
                    kind: InteractionOfferKind::JointMission,
                    summary: "Joint strike".into(),
                },
                InteractionOffer {
                    id: 2,
                    from_faction_id: 2,
                    region_id: 1,
                    mission_slot: None,
                    kind: InteractionOfferKind::TrainingLoan,
                    summary: "Training exchange".into(),
                },
                InteractionOffer {
                    id: 3,
                    from_faction_id: 1,
                    region_id: 2,
                    mission_slot: None,
                    kind: InteractionOfferKind::RivalRaid,
                    summary: "Raid support".into(),
                },
            ],
            selected: 0,
            notice: String::new(),
            next_offer_id: 4,
        });
        spawn_test_missions(&mut world);
        world
    }

    #[test]
    fn interaction_offer_keyo_increments_selection() {
        let mut world = make_interaction_world();
        run_interaction_offer(&mut world, KeyCode::KeyO);
        assert_eq!(world.resource::<InteractionBoard>().selected, 1);
    }

    #[test]
    fn interaction_offer_keyo_wraps_to_zero_at_end() {
        let mut world = make_interaction_world();
        world.resource_mut::<InteractionBoard>().selected = 2; // last of 3
        run_interaction_offer(&mut world, KeyCode::KeyO);
        assert_eq!(world.resource::<InteractionBoard>().selected, 0);
    }

    #[test]
    fn interaction_offer_keyu_wraps_to_last_from_zero() {
        let mut world = make_interaction_world();
        run_interaction_offer(&mut world, KeyCode::KeyU);
        assert_eq!(world.resource::<InteractionBoard>().selected, 2);
    }

    #[test]
    fn interaction_offer_keyn_removes_selected_offer() {
        let mut world = make_interaction_world();
        world.resource_mut::<InteractionBoard>().selected = 1;
        run_interaction_offer(&mut world, KeyCode::KeyN);
        let board = world.resource::<InteractionBoard>();
        assert_eq!(board.offers.len(), 2, "rejected offer should be removed");
        assert!(board.selected < board.offers.len(), "selected should remain in bounds");
    }

    #[test]
    fn interaction_offer_empty_board_is_noop() {
        let mut world = make_interaction_world();
        world.resource_mut::<InteractionBoard>().offers.clear();
        run_interaction_offer(&mut world, KeyCode::KeyO);
        run_interaction_offer(&mut world, KeyCode::KeyN);
        let board = world.resource::<InteractionBoard>();
        assert_eq!(board.offers.len(), 0);
        assert_eq!(board.selected, 0);
    }
}
