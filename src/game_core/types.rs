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

impl RoomType {
    pub fn from_str(s: &str) -> Option<RoomType> {
        match s {
            "Entry" => Some(RoomType::Entry),
            "Pressure" => Some(RoomType::Pressure),
            "Pivot" => Some(RoomType::Pivot),
            "Setpiece" => Some(RoomType::Setpiece),
            "Recovery" => Some(RoomType::Recovery),
            "Climax" => Some(RoomType::Climax),
            _ => None,
        }
    }
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

    pub fn from_components(
        data: &MissionData,
        progress: &MissionProgress,
        tactics: &MissionTactics,
    ) -> Self {
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
