use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ai::core::{SimState, SimVec2, Team, UnitIntent, UnitState};
use crate::ai::core::ability_eval::AbilityEvalWeights;
use crate::ai::core::ability_encoding::AbilityEncoder;
use crate::ai::phase::AiPhase;

use super::personality::{infer_personality, Personality};

// ---------------------------------------------------------------------------
// Movement profile -- continuous interpolation from personality
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormationMode {
    Hold,
    Advance,
    Retreat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SquadBlackboard {
    pub focus_target: Option<u32>,
    pub mode: FormationMode,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RoleProfile {
    pub preferred_range_min: f32,
    pub preferred_range_max: f32,
    pub leash_distance: f32,
}

pub(super) fn personality_movement_profile(personality: &Personality, unit: &UnitState) -> RoleProfile {
    let aggro = personality.aggression;
    let caution = personality.caution;

    let base_min = unit.attack_range * 0.8;
    let base_max = unit.attack_range * 1.5;
    let range_offset = (caution - aggro) * 2.0;
    let preferred_range_min = (base_min + range_offset).max(0.5);
    let preferred_range_max = (base_max + range_offset * 1.2).max(preferred_range_min + 0.3);

    let leash_distance = 14.0 + (caution - aggro + 0.5).clamp(0.0, 1.0) * 5.0;

    RoleProfile {
        preferred_range_min,
        preferred_range_max,
        leash_distance,
    }
}

// ---------------------------------------------------------------------------
// Drift triggers
// ---------------------------------------------------------------------------

const DRIFT_CAP: f32 = 0.15;

fn clamp_drift(base: &Personality, drift: &mut Personality) {
    fn cap(base_val: f32, drift_val: &mut f32) {
        *drift_val = drift_val.clamp(-DRIFT_CAP, DRIFT_CAP);
        let effective = base_val + *drift_val;
        if effective > 1.0 {
            *drift_val = 1.0 - base_val;
        } else if effective < 0.0 {
            *drift_val = -base_val;
        }
    }
    cap(base.aggression, &mut drift.aggression);
    cap(base.compassion, &mut drift.compassion);
    cap(base.caution, &mut drift.caution);
    cap(base.discipline, &mut drift.discipline);
    cap(base.cunning, &mut drift.cunning);
    cap(base.tenacity, &mut drift.tenacity);
    cap(base.patience, &mut drift.patience);
}

#[derive(Debug, Clone, Copy)]
pub enum DriftTrigger {
    SurvivedLowHp,
    AllyDiedNearby,
    ScoredKill,
    TookHeavyBurst,
    CcdPriorityTarget,
    HeldAbilityOptimal,
}

// ---------------------------------------------------------------------------
// Unit memory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct UnitMemory {
    pub anchor_position: SimVec2,
    pub sticky_target: Option<u32>,
    pub lock_ticks: u32,
}

// ---------------------------------------------------------------------------
// SquadAiState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SquadAiState {
    pub(super) personality_by_unit: HashMap<u32, Personality>,
    pub(super) drift_by_unit: HashMap<u32, Personality>,
    pub(super) memory: HashMap<u32, UnitMemory>,
    pub(super) blackboard_by_team: HashMap<Team, SquadBlackboard>,
    pub(super) eval_every_ticks: u64,
    /// Optional ability evaluator weights for interrupt-driven ability usage.
    pub ability_eval_weights: Option<AbilityEvalWeights>,
    /// Optional frozen ability encoder for embedding-enriched evaluation.
    pub ability_encoder: Option<AbilityEncoder>,
}

impl SquadAiState {
    pub fn new(initial: &SimState, personality_by_unit: HashMap<u32, Personality>) -> Self {
        let memory = initial
            .units
            .iter()
            .map(|u| {
                (
                    u.id,
                    UnitMemory {
                        anchor_position: u.position,
                        sticky_target: None,
                        lock_ticks: 0,
                    },
                )
            })
            .collect();

        let drift_by_unit = personality_by_unit
            .keys()
            .map(|&id| (id, Personality::zero()))
            .collect();

        let blackboard_by_team = HashMap::from([
            (
                Team::Hero,
                SquadBlackboard {
                    focus_target: None,
                    mode: FormationMode::Hold,
                },
            ),
            (
                Team::Enemy,
                SquadBlackboard {
                    focus_target: None,
                    mode: FormationMode::Hold,
                },
            ),
        ]);

        Self {
            personality_by_unit,
            drift_by_unit,
            memory,
            blackboard_by_team,
            eval_every_ticks: 5,
            ability_eval_weights: None,
            ability_encoder: None,
        }
    }

    /// Load ability evaluator weights from a JSON file.
    pub fn load_ability_eval_weights(&mut self, path: &std::path::Path) -> Result<(), String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse JSON: {e}"))?;
        self.ability_eval_weights = Some(AbilityEvalWeights::from_json(&json));
        Ok(())
    }

    /// Load a frozen ability encoder for embedding-enriched ability evaluation.
    pub fn load_ability_encoder(&mut self, path: &std::path::Path) -> Result<(), String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        let encoder = AbilityEncoder::from_json(&data)?;
        self.ability_encoder = Some(encoder);
        Ok(())
    }

    /// Construct from role map (backward compatibility for control.rs / core.rs tests).
    pub fn new_from_roles(
        initial: &SimState,
        roles: HashMap<u32, crate::ai::roles::Role>,
    ) -> Self {
        let personality_by_unit = roles
            .into_iter()
            .map(|(id, role)| (id, role_to_personality(role)))
            .collect();
        Self::new(initial, personality_by_unit)
    }

    /// Auto-infer personalities from unit stats.
    pub fn new_inferred(initial: &SimState) -> Self {
        let personality_by_unit = initial
            .units
            .iter()
            .filter(|u| u.hp > 0)
            .map(|u| (u.id, infer_personality(u)))
            .collect();
        Self::new(initial, personality_by_unit)
    }

    pub(super) fn personality_for(&self, unit_id: u32) -> Personality {
        let base = self
            .personality_by_unit
            .get(&unit_id)
            .copied()
            .unwrap_or_else(Personality::default_balanced);
        let drift = self
            .drift_by_unit
            .get(&unit_id)
            .copied()
            .unwrap_or_else(Personality::zero);
        base.effective(&drift)
    }

    pub fn apply_drift(&mut self, unit_id: u32, trigger: DriftTrigger) {
        let base = self
            .personality_by_unit
            .get(&unit_id)
            .copied()
            .unwrap_or_else(Personality::default_balanced);
        let drift = self
            .drift_by_unit
            .entry(unit_id)
            .or_insert_with(Personality::zero);

        match trigger {
            DriftTrigger::SurvivedLowHp => {
                drift.aggression += 0.02;
                drift.caution -= 0.01;
            }
            DriftTrigger::AllyDiedNearby => {
                drift.compassion += 0.03;
                drift.caution += 0.01;
            }
            DriftTrigger::ScoredKill => {
                drift.aggression += 0.01;
                drift.tenacity += 0.01;
            }
            DriftTrigger::TookHeavyBurst => {
                drift.caution += 0.02;
            }
            DriftTrigger::CcdPriorityTarget => {
                drift.cunning += 0.01;
            }
            DriftTrigger::HeldAbilityOptimal => {
                drift.patience += 0.01;
            }
        }

        clamp_drift(&base, drift);
    }

    pub fn evaluate_blackboards_if_needed(&mut self, state: &SimState) {
        if state.tick % self.eval_every_ticks != 0 {
            return;
        }
        self.blackboard_by_team
            .insert(Team::Hero, evaluate_blackboard(state, Team::Hero));
        self.blackboard_by_team
            .insert(Team::Enemy, evaluate_blackboard(state, Team::Enemy));
    }

    pub(super) fn blackboard(&self, team: Team) -> SquadBlackboard {
        self.blackboard_by_team
            .get(&team)
            .copied()
            .unwrap_or(SquadBlackboard {
                focus_target: None,
                mode: FormationMode::Hold,
            })
    }

    pub fn set_blackboard(&mut self, team: Team, board: SquadBlackboard) {
        self.blackboard_by_team.insert(team, board);
    }

    /// Public read-only access to a team's blackboard.
    pub fn blackboard_for_team(&self, team: Team) -> Option<&SquadBlackboard> {
        self.blackboard_by_team.get(&team)
    }

    /// Override the focus target for a team's blackboard.
    pub fn set_focus_target(&mut self, team: Team, target: Option<u32>) {
        if let Some(board) = self.blackboard_by_team.get_mut(&team) {
            board.focus_target = target;
        }
    }
}

/// Convert a discrete Role to an approximate Personality.
pub(super) fn role_to_personality(role: crate::ai::roles::Role) -> Personality {
    use crate::ai::roles::Role;
    match role {
        Role::Tank => Personality {
            aggression: 0.8,
            compassion: 0.3,
            caution: 0.2,
            discipline: 0.75,
            cunning: 0.4,
            tenacity: 0.7,
            patience: 0.4,
        },
        Role::Dps => Personality {
            aggression: 0.6,
            compassion: 0.3,
            caution: 0.45,
            discipline: 0.5,
            cunning: 0.6,
            tenacity: 0.5,
            patience: 0.4,
        },
        Role::Healer => Personality {
            aggression: 0.2,
            compassion: 0.9,
            caution: 0.8,
            discipline: 0.6,
            cunning: 0.4,
            tenacity: 0.4,
            patience: 0.8,
        },
    }
}

impl AiPhase for SquadAiState {
    fn generate_intents(&mut self, state: &SimState, dt_ms: u32) -> Vec<UnitIntent> {
        super::intents::generate_intents(state, self, dt_ms)
    }
}

// ---------------------------------------------------------------------------
// Blackboard evaluation
// ---------------------------------------------------------------------------

fn evaluate_blackboard(state: &SimState, team: Team) -> SquadBlackboard {
    let allies = alive_by_team(state, team);
    let enemies = alive_by_team(state, opposite(team));

    let focus_target = enemies.iter().copied().min_by(|a, b| {
        let hpa = state
            .units
            .iter()
            .find(|u| u.id == *a)
            .map_or(i32::MAX, |u| u.hp);
        let hpb = state
            .units
            .iter()
            .find(|u| u.id == *b)
            .map_or(i32::MAX, |u| u.hp);
        hpa.cmp(&hpb).then_with(|| a.cmp(b))
    });

    let ally_avg = average_hp_pct(state, &allies);
    let enemy_avg = average_hp_pct(state, &enemies);

    let mode =
        if allies.len() * 2 < enemies.len() || (ally_avg < 0.35 && enemy_avg > ally_avg + 0.10) {
            FormationMode::Retreat
        } else if ally_avg > enemy_avg + 0.12 || allies.len() > enemies.len() * 2 {
            FormationMode::Advance
        } else {
            FormationMode::Hold
        };

    SquadBlackboard { focus_target, mode }
}

fn average_hp_pct(state: &SimState, ids: &[u32]) -> f32 {
    if ids.is_empty() {
        return 0.0;
    }
    let sum = ids
        .iter()
        .filter_map(|id| state.units.iter().find(|u| u.id == *id))
        .map(|u| u.hp.max(0) as f32 / u.max_hp.max(1) as f32)
        .sum::<f32>();
    sum / ids.len() as f32
}

pub(super) fn opposite(team: Team) -> Team {
    match team {
        Team::Hero => Team::Enemy,
        Team::Enemy => Team::Hero,
    }
}

pub(super) fn alive_by_team(state: &SimState, team: Team) -> Vec<u32> {
    let mut ids = state
        .units
        .iter()
        .filter(|u| u.hp > 0 && u.team == team)
        .map(|u| u.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

// ---------------------------------------------------------------------------
// Per-tick context -- precomputed lookups to avoid O(n^2) scans
// ---------------------------------------------------------------------------

pub(super) struct TickContext {
    /// unit_id -> index in state.units
    index: HashMap<u32, usize>,
    /// alive hero unit IDs (sorted)
    heroes: Vec<u32>,
    /// alive enemy unit IDs (sorted)
    enemies: Vec<u32>,
}

impl TickContext {
    pub fn new(state: &SimState) -> Self {
        let mut index = HashMap::with_capacity(state.units.len());
        let mut heroes = Vec::new();
        let mut enemies = Vec::new();
        for (i, u) in state.units.iter().enumerate() {
            index.insert(u.id, i);
            if u.hp > 0 {
                match u.team {
                    Team::Hero => heroes.push(u.id),
                    Team::Enemy => enemies.push(u.id),
                }
            }
        }
        heroes.sort_unstable();
        enemies.sort_unstable();
        Self { index, heroes, enemies }
    }

    #[inline]
    pub fn unit<'a>(&self, state: &'a SimState, id: u32) -> Option<&'a UnitState> {
        self.index.get(&id).map(|&i| &state.units[i])
    }

    #[inline]
    pub fn allies(&self, team: Team) -> &[u32] {
        match team {
            Team::Hero => &self.heroes,
            Team::Enemy => &self.enemies,
        }
    }

    #[inline]
    pub fn enemies_of(&self, team: Team) -> &[u32] {
        self.allies(opposite(team))
    }
}
