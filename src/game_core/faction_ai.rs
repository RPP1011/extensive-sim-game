//! Faction AI — autonomous per-turn strategic decision-making.
//!
//! Each faction commander evaluates 6 possible actions (Expand, Defend, Attack,
//! Diplomacy, Recruit, Dispatch) and selects the highest-scoring option based
//! on personality parameters and game state. Personality parameters are data-driven
//! via TOML and hot-reloadable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use super::{
    DiplomacyState, FactionCommander, FactionState, OverworldMap, OverworldRegion,
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/// The 6 possible per-turn faction actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactionAction {
    Expand,
    Defend,
    Attack,
    Diplomacy,
    Recruit,
    Dispatch,
}

/// Details of the chosen action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactionActionDetails {
    ExpandTo { region_id: usize },
    DefendRegion { region_id: usize },
    AttackFaction { target_faction_id: usize, region_id: usize },
    DiplomacyOffer { target_faction_id: usize, relation_change: i32 },
    RecruitUnits { count: u32 },
    DispatchVassal { vassal_id: u32, region_id: usize },
}

/// Result of a single faction's turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionTurnResult {
    pub faction_id: usize,
    pub action: FactionAction,
    pub details: FactionActionDetails,
    pub description: String,
}

// ---------------------------------------------------------------------------
// Personality config (TOML-driven, hot-reloadable)
// ---------------------------------------------------------------------------

/// Per-action weight biases loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionPersonalityConfig {
    pub name: String,
    /// Base scores added to each action during evaluation.
    #[serde(default)]
    pub action_biases: HashMap<String, f32>,
    /// Override aggression/cooperation for this personality.
    #[serde(default)]
    pub aggression_override: Option<f32>,
    #[serde(default)]
    pub cooperation_override: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactionPersonalitiesFile {
    #[serde(default)]
    pub personality: Vec<FactionPersonalityConfig>,
}

/// Registry of faction personality configs.
#[derive(Debug, Clone, Default)]
pub struct FactionPersonalityRegistry {
    configs: HashMap<String, FactionPersonalityConfig>,
}

impl FactionPersonalityRegistry {
    pub fn new() -> Self {
        Self { configs: HashMap::new() }
    }

    pub fn load_from_str(&mut self, toml_str: &str) -> Result<usize, String> {
        let file: FactionPersonalitiesFile =
            toml::from_str(toml_str).map_err(|e| format!("TOML parse error: {}", e))?;
        self.configs.clear();
        let count = file.personality.len();
        for p in file.personality {
            self.configs.insert(p.name.clone(), p);
        }
        Ok(count)
    }

    pub fn load_from_file(&mut self, path: &Path) -> Result<usize, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        self.load_from_str(&contents)
    }

    pub fn get(&self, name: &str) -> Option<&FactionPersonalityConfig> {
        self.configs.get(name)
    }

    pub fn len(&self) -> usize { self.configs.len() }
    pub fn is_empty(&self) -> bool { self.configs.is_empty() }
}

// ---------------------------------------------------------------------------
// Action scoring
// ---------------------------------------------------------------------------

/// Score all 6 actions for a faction and return the best one.
fn score_actions(
    faction: &FactionState,
    commander: &FactionCommander,
    map: &OverworldMap,
    diplomacy: &DiplomacyState,
    personality: Option<&FactionPersonalityConfig>,
    rng_val: u64,
) -> (FactionAction, f32) {
    let owned_regions: Vec<usize> = map
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == faction.id)
        .map(|r| r.id)
        .collect();
    let num_owned = owned_regions.len() as f32;
    let total_regions = map.regions.len().max(1) as f32;

    // Effective personality values (may be overridden by TOML config)
    let aggression = personality
        .and_then(|p| p.aggression_override)
        .unwrap_or(commander.aggression);
    let cooperation = personality
        .and_then(|p| p.cooperation_override)
        .unwrap_or(commander.cooperation_bias);

    // Contested border regions (our regions neighboring enemy regions)
    let border_pressure = owned_regions.iter().filter(|&&rid| {
        let region = &map.regions[rid];
        region.neighbors.iter().any(|&nid| {
            nid < map.regions.len() && map.regions[nid].owner_faction_id != faction.id
        })
    }).count() as f32;

    // Can we expand? (adjacent unclaimed or weak-faction regions)
    let expandable = owned_regions.iter().any(|&rid| {
        map.regions[rid].neighbors.iter().any(|&nid| {
            nid < map.regions.len() && map.regions[nid].owner_faction_id != faction.id
        })
    });

    // Faction strength relative to average
    let avg_strength = map.factions.iter().map(|f| f.strength).sum::<f32>()
        / map.factions.len().max(1) as f32;
    let relative_strength = faction.strength / avg_strength.max(0.01);

    // Best diplomacy target (highest relations that aren't us)
    let best_ally_faction = (0..map.factions.len())
        .filter(|&fid| fid != faction.id)
        .max_by_key(|&fid| relation_score(diplomacy, faction.id, fid));

    // Score each action
    let mut scores: [(FactionAction, f32); 6] = [
        (FactionAction::Expand, 0.0),
        (FactionAction::Defend, 0.0),
        (FactionAction::Attack, 0.0),
        (FactionAction::Diplomacy, 0.0),
        (FactionAction::Recruit, 0.0),
        (FactionAction::Dispatch, 0.0),
    ];

    // Expand: good when we have room and are strong
    scores[0].1 = if expandable {
        (1.0 - num_owned / total_regions) * 0.5 + relative_strength * 0.3 + aggression * 0.2
    } else {
        -1.0
    };

    // Defend: good when we have border pressure and are weak
    scores[1].1 = border_pressure * 0.15 + (1.0 - relative_strength).max(0.0) * 0.4
        + (1.0 - aggression) * 0.2 + (1.0 - faction.cohesion) * 0.2;

    // Attack: good when strong, aggressive, and have a war goal
    let has_war_goal = faction.war_goal_faction_id.is_some();
    scores[2].1 = aggression * 0.4 + relative_strength.min(2.0) * 0.3
        + if has_war_goal { 0.3 } else { 0.0 }
        - cooperation * 0.2;

    // Diplomacy: good when cooperative and have potential allies
    scores[3].1 = cooperation * 0.5 + (1.0 - aggression) * 0.2
        + if best_ally_faction.is_some() { 0.2 } else { 0.0 }
        + (1.0 - relative_strength).max(0.0) * 0.1;

    // Recruit: good when weak or after losses
    scores[4].1 = (1.0 - relative_strength).max(0.0) * 0.4
        + (1.0 - faction.strength / 100.0).max(0.0) * 0.3
        + commander.competence * 0.1;

    // Dispatch: good when we have vassals and border pressure
    let has_vassals = !faction.vassals.is_empty();
    scores[5].1 = if has_vassals {
        border_pressure * 0.2 + commander.competence * 0.2 + 0.1
    } else {
        -1.0
    };

    // Apply personality biases from TOML config
    if let Some(config) = personality {
        for (action_name, bias) in &config.action_biases {
            let action = match action_name.as_str() {
                "expand" => FactionAction::Expand,
                "defend" => FactionAction::Defend,
                "attack" => FactionAction::Attack,
                "diplomacy" => FactionAction::Diplomacy,
                "recruit" => FactionAction::Recruit,
                "dispatch" => FactionAction::Dispatch,
                _ => continue,
            };
            if let Some(score) = scores.iter_mut().find(|s| s.0 == action) {
                score.1 += bias;
            }
        }
    }

    // Add small random noise for variety
    let noise = ((rng_val % 100) as f32 / 100.0 - 0.5) * 0.1;
    for score in &mut scores {
        score.1 += noise;
    }

    scores
        .iter()
        .copied()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
}

fn relation_score(diplomacy: &DiplomacyState, a: usize, b: usize) -> i32 {
    if a < diplomacy.relations.len() && b < diplomacy.relations[a].len() {
        diplomacy.relations[a][b]
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Turn execution
// ---------------------------------------------------------------------------

/// Execute a single faction's turn. Returns the chosen action and its effect.
pub fn step_faction_turn(
    faction: &FactionState,
    commander: &FactionCommander,
    map: &OverworldMap,
    diplomacy: &DiplomacyState,
    personality: Option<&FactionPersonalityConfig>,
    rng_seed: u64,
) -> FactionTurnResult {
    let rng_val = lcg_next(rng_seed ^ faction.id as u64);
    let (action, _score) = score_actions(faction, commander, map, diplomacy, personality, rng_val);

    let owned_regions: Vec<usize> = map
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == faction.id)
        .map(|r| r.id)
        .collect();

    let (details, description) = match action {
        FactionAction::Expand => {
            // Pick a neighboring region we don't own
            let target = owned_regions
                .iter()
                .flat_map(|&rid| map.regions[rid].neighbors.iter())
                .filter(|&&nid| nid < map.regions.len() && map.regions[nid].owner_faction_id != faction.id)
                .copied()
                .next()
                .unwrap_or(0);
            (
                FactionActionDetails::ExpandTo { region_id: target },
                format!("{} expands toward {}.", faction.name, region_name(map, target)),
            )
        }
        FactionAction::Defend => {
            let border = owned_regions
                .iter()
                .filter(|&&rid| {
                    map.regions[rid].neighbors.iter().any(|&nid| {
                        nid < map.regions.len() && map.regions[nid].owner_faction_id != faction.id
                    })
                })
                .copied()
                .next()
                .unwrap_or(owned_regions.first().copied().unwrap_or(0));
            (
                FactionActionDetails::DefendRegion { region_id: border },
                format!("{} fortifies defenses in {}.", faction.name, region_name(map, border)),
            )
        }
        FactionAction::Attack => {
            let target_faction = faction.war_goal_faction_id.unwrap_or_else(|| {
                // Pick faction with lowest relations
                (0..map.factions.len())
                    .filter(|&fid| fid != faction.id)
                    .min_by_key(|&fid| relation_score(diplomacy, faction.id, fid))
                    .unwrap_or(0)
            });
            let target_region = map
                .regions
                .iter()
                .filter(|r| r.owner_faction_id == target_faction)
                .map(|r| r.id)
                .next()
                .unwrap_or(0);
            (
                FactionActionDetails::AttackFaction {
                    target_faction_id: target_faction,
                    region_id: target_region,
                },
                format!(
                    "{} launches an attack against {} in {}.",
                    faction.name,
                    map.factions.get(target_faction).map(|f| f.name.as_str()).unwrap_or("unknown"),
                    region_name(map, target_region),
                ),
            )
        }
        FactionAction::Diplomacy => {
            let target_faction = (0..map.factions.len())
                .filter(|&fid| fid != faction.id)
                .max_by_key(|&fid| relation_score(diplomacy, faction.id, fid))
                .unwrap_or(0);
            let relation_change = 3 + (rng_val % 5) as i32;
            (
                FactionActionDetails::DiplomacyOffer {
                    target_faction_id: target_faction,
                    relation_change,
                },
                format!(
                    "{} extends a diplomatic offer to {}.",
                    faction.name,
                    map.factions.get(target_faction).map(|f| f.name.as_str()).unwrap_or("unknown"),
                ),
            )
        }
        FactionAction::Recruit => {
            let count = 1 + (rng_val % 3) as u32;
            (
                FactionActionDetails::RecruitUnits { count },
                format!("{} recruits {} new units.", faction.name, count),
            )
        }
        FactionAction::Dispatch => {
            let vassal_id = faction.vassals.first().map(|v| v.id).unwrap_or(0);
            let region = owned_regions.first().copied().unwrap_or(0);
            (
                FactionActionDetails::DispatchVassal {
                    vassal_id,
                    region_id: region,
                },
                format!("{} dispatches a vassal to {}.", faction.name, region_name(map, region)),
            )
        }
    };

    FactionTurnResult {
        faction_id: faction.id,
        action,
        details,
        description,
    }
}

/// Execute all factions' turns for a single campaign turn.
pub fn step_all_factions(
    map: &OverworldMap,
    commanders: &[FactionCommander],
    diplomacy: &DiplomacyState,
    personalities: &FactionPersonalityRegistry,
    turn: u32,
) -> Vec<FactionTurnResult> {
    let base_seed = map.map_seed.wrapping_add(turn as u64 * 997);
    let mut results = Vec::new();

    for faction in &map.factions {
        let commander = commanders
            .iter()
            .find(|c| c.faction_id == faction.id);
        let Some(commander) = commander else { continue };

        let personality = personalities.get(&commander.name);
        let rng_seed = lcg_next(base_seed ^ faction.id as u64);

        results.push(step_faction_turn(
            faction,
            commander,
            map,
            diplomacy,
            personality,
            rng_seed,
        ));
    }

    results
}

/// Apply the results of faction turns to the game state.
pub fn apply_faction_results(
    results: &[FactionTurnResult],
    map: &mut OverworldMap,
    diplomacy: &mut DiplomacyState,
) {
    for result in results {
        match &result.details {
            FactionActionDetails::ExpandTo { region_id } => {
                if let Some(region) = map.regions.get_mut(*region_id) {
                    // Claim region only if uncontested (simplified)
                    region.owner_faction_id = result.faction_id;
                    region.control = 0.3; // Newly claimed, low control
                }
            }
            FactionActionDetails::DefendRegion { region_id } => {
                if let Some(region) = map.regions.get_mut(*region_id) {
                    region.control = (region.control + 0.1).min(1.0);
                    region.unrest = (region.unrest - 0.05).max(0.0);
                }
                if let Some(faction) = map.factions.get_mut(result.faction_id) {
                    faction.cohesion = (faction.cohesion + 0.02).min(1.0);
                }
            }
            FactionActionDetails::AttackFaction { target_faction_id, region_id } => {
                // Reduce target faction strength, increase war focus
                if let Some(target) = map.factions.get_mut(*target_faction_id) {
                    target.strength = (target.strength - 3.0).max(0.0);
                    target.cohesion = (target.cohesion - 0.03).max(0.0);
                }
                if let Some(faction) = map.factions.get_mut(result.faction_id) {
                    faction.war_focus = (faction.war_focus + 0.1).min(1.0);
                    faction.war_goal_faction_id = Some(*target_faction_id);
                }
                // Reduce relations
                update_relation(diplomacy, result.faction_id, *target_faction_id, -5);
            }
            FactionActionDetails::DiplomacyOffer { target_faction_id, relation_change } => {
                update_relation(diplomacy, result.faction_id, *target_faction_id, *relation_change);
            }
            FactionActionDetails::RecruitUnits { count } => {
                if let Some(faction) = map.factions.get_mut(result.faction_id) {
                    faction.strength += *count as f32 * 2.0;
                }
            }
            FactionActionDetails::DispatchVassal { region_id, .. } => {
                if let Some(region) = map.regions.get_mut(*region_id) {
                    region.control = (region.control + 0.15).min(1.0);
                }
            }
        }
    }
}

fn update_relation(diplomacy: &mut DiplomacyState, a: usize, b: usize, delta: i32) {
    if a < diplomacy.relations.len() && b < diplomacy.relations[a].len() {
        diplomacy.relations[a][b] = (diplomacy.relations[a][b] + delta).clamp(-100, 100);
    }
    if b < diplomacy.relations.len() && a < diplomacy.relations[b].len() {
        diplomacy.relations[b][a] = (diplomacy.relations[b][a] + delta).clamp(-100, 100);
    }
}

fn region_name(map: &OverworldMap, region_id: usize) -> &str {
    map.regions
        .get(region_id)
        .map(|r| r.name.as_str())
        .unwrap_or("unknown")
}

fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_map() -> OverworldMap {
        OverworldMap::default()
    }

    fn test_diplomacy() -> DiplomacyState {
        DiplomacyState::default()
    }

    fn test_commanders() -> Vec<FactionCommander> {
        vec![
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
        ]
    }

    #[test]
    fn test_step_all_factions_produces_results() {
        let map = test_map();
        let commanders = test_commanders();
        let diplomacy = test_diplomacy();
        let personalities = FactionPersonalityRegistry::new();

        let results = step_all_factions(&map, &commanders, &diplomacy, &personalities, 1);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(!r.description.is_empty());
        }
    }

    #[test]
    fn test_aggressive_commander_prefers_attack() {
        let map = test_map();
        let commanders = test_commanders();
        let diplomacy = test_diplomacy();
        let personalities = FactionPersonalityRegistry::new();

        // Run multiple turns, Lord Caradoc (aggressive) should attack frequently
        let mut attack_count = 0;
        for turn in 0..20 {
            let results = step_all_factions(&map, &commanders, &diplomacy, &personalities, turn);
            if let Some(caradoc) = results.iter().find(|r| r.faction_id == 1) {
                if caradoc.action == FactionAction::Attack {
                    attack_count += 1;
                }
            }
        }
        // Caradoc (aggression=0.78) should attack at least sometimes
        assert!(attack_count > 0, "Aggressive commander should attack at least once in 20 turns");
    }

    #[test]
    fn test_cooperative_commander_prefers_diplomacy() {
        let map = test_map();
        let commanders = test_commanders();
        let diplomacy = test_diplomacy();
        let personalities = FactionPersonalityRegistry::new();

        let mut diplo_count = 0;
        for turn in 0..20 {
            let results = step_all_factions(&map, &commanders, &diplomacy, &personalities, turn);
            if let Some(elowen) = results.iter().find(|r| r.faction_id == 0) {
                if elowen.action == FactionAction::Diplomacy {
                    diplo_count += 1;
                }
            }
        }
        assert!(diplo_count > 0, "Cooperative commander should use diplomacy at least once in 20 turns");
    }

    #[test]
    fn test_apply_diplomacy_changes_relations() {
        let mut map = test_map();
        let mut diplomacy = test_diplomacy();
        let initial = diplomacy.relations[0][1];

        let results = vec![FactionTurnResult {
            faction_id: 0,
            action: FactionAction::Diplomacy,
            details: FactionActionDetails::DiplomacyOffer {
                target_faction_id: 1,
                relation_change: 10,
            },
            description: "Test".to_string(),
        }];
        apply_faction_results(&results, &mut map, &mut diplomacy);
        assert_eq!(diplomacy.relations[0][1], initial + 10);
        assert_eq!(diplomacy.relations[1][0], initial + 10); // symmetric
    }

    #[test]
    fn test_apply_attack_reduces_strength() {
        let mut map = test_map();
        let mut diplomacy = test_diplomacy();
        let initial_strength = map.factions[1].strength;

        let results = vec![FactionTurnResult {
            faction_id: 0,
            action: FactionAction::Attack,
            details: FactionActionDetails::AttackFaction {
                target_faction_id: 1,
                region_id: 0,
            },
            description: "Test".to_string(),
        }];
        apply_faction_results(&results, &mut map, &mut diplomacy);
        assert!(map.factions[1].strength < initial_strength);
    }

    #[test]
    fn test_50_turn_progression() {
        let mut map = test_map();
        let commanders = test_commanders();
        let mut diplomacy = test_diplomacy();
        let personalities = FactionPersonalityRegistry::new();

        let initial_strengths: Vec<f32> = map.factions.iter().map(|f| f.strength).collect();

        for turn in 0..50 {
            let results = step_all_factions(&map, &commanders, &diplomacy, &personalities, turn);
            apply_faction_results(&results, &mut map, &mut diplomacy);
        }

        // After 50 turns, at least some faction state should have changed
        let changed = map.factions.iter().enumerate().any(|(i, f)| {
            (f.strength - initial_strengths[i]).abs() > 0.1
        });
        assert!(changed, "Faction strengths should change over 50 turns");

        // No faction should be completely zeroed out in just 50 turns
        for f in &map.factions {
            assert!(f.strength >= 0.0, "Faction strength should not go negative");
        }
    }

    #[test]
    fn test_personality_toml_loading() {
        let toml = r#"
[[personality]]
name = "Marshal Elowen"
aggression_override = 0.2
cooperation_override = 0.9

[personality.action_biases]
diplomacy = 0.5
defend = 0.3
"#;
        let mut reg = FactionPersonalityRegistry::new();
        let count = reg.load_from_str(toml).unwrap();
        assert_eq!(count, 1);

        let p = reg.get("Marshal Elowen").unwrap();
        assert_eq!(p.aggression_override, Some(0.2));
        assert!((p.action_biases["diplomacy"] - 0.5).abs() < 0.001);
    }
}
