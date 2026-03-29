//! Unit tier system — data-driven progression from Adventurer to Legend.
//!
//! Tiers provide flat bonuses + scaling coefficients. Levels are the
//! accumulated scaling value within a tier. Tiers are loaded from
//! `dataset/campaign/unit_tiers/*.toml`.

use serde::{Deserialize, Serialize};

use super::state::*;

// ---------------------------------------------------------------------------
// Tier template (loaded from TOML)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TierTemplate {
    pub name: String,
    pub tier_index: u32,
    pub description: String,
    #[serde(default)]
    pub requirements: TierRequirements,
    #[serde(default)]
    pub flat_bonuses: TierBonuses,
    #[serde(default)]
    pub scaling: TierScaling,
    #[serde(default)]
    pub behavior: TierBehavior,
    #[serde(default)]
    pub demotion: TierDemotion,
    #[serde(default)]
    pub candidacy: Option<TierCandidacy>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TierRequirements {
    #[serde(default)]
    pub min_level: u32,
    #[serde(default)]
    pub min_quests_completed: usize,
    #[serde(default)]
    pub min_fame: f32,
    #[serde(default)]
    pub min_party_victories: usize,
    #[serde(default)]
    pub min_group_size: usize,
    #[serde(default)]
    pub min_allied_factions: usize,
    #[serde(default)]
    pub min_prior_tier: u32,
    #[serde(default)]
    pub requires_active_crisis: bool,
    #[serde(default)]
    pub requires_quest_completion: Option<String>,
    #[serde(default)]
    pub min_gold_invested: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TierBonuses {
    #[serde(default)]
    pub attack: f32,
    #[serde(default)]
    pub defense: f32,
    #[serde(default)]
    pub max_hp: f32,
    #[serde(default)]
    pub resolve: f32,
    #[serde(default)]
    pub speed: f32,
    #[serde(default)]
    pub ability_power: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TierScaling {
    #[serde(default)]
    pub coefficient: f32,
    #[serde(default)]
    pub source: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TierBehavior {
    #[serde(default)]
    pub required_actions: Vec<String>,
    #[serde(default)]
    pub forbidden_actions: Vec<String>,
    #[serde(default)]
    pub crisis_actions_required: bool,
    #[serde(default)]
    pub idle_decay_ticks: u64,
    #[serde(default)]
    pub decay_rate: f32,
    #[serde(default)]
    pub betrayal_actions: Vec<String>,
    #[serde(default)]
    pub betrayal_fame_penalty: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TierDemotion {
    #[serde(default)]
    pub fame_threshold: f32,
    #[serde(default)]
    pub trigger: String,
    #[serde(default)]
    pub fall_to_tier: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TierCandidacy {
    pub bonus_fraction_as_candidate: f32,
}

// ---------------------------------------------------------------------------
// Unit tier status (stored on adventurer)
// ---------------------------------------------------------------------------

/// An adventurer's current tier status.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UnitTierStatus {
    /// Current tier index (0=Adventurer, 1=Named, 2=Champion, etc.)
    pub tier: u32,
    /// Accumulated fame within this tier.
    pub fame: f32,
    /// Peak fame ever reached (for Legend scaling).
    pub peak_fame: f32,
    /// Tick of last action that matched the behavioral contract.
    pub last_contract_action_tick: u64,
    /// What group this unit champions (faction_id or guild).
    pub champion_of: Option<ChampionOf>,
    /// Whether this is a hero candidate (quest not yet complete).
    pub is_candidate: bool,
    /// Which crisis this hero opposes (if hero tier).
    pub opposing_crisis_index: Option<usize>,
    /// Gold invested toward hero promotion.
    pub gold_invested: f32,
    /// Total quests completed (tracked for tier requirements).
    pub quests_completed: usize,
    /// Party victories led.
    pub party_victories: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChampionOf {
    Guild,
    Faction(usize),
    Coalition(Vec<usize>),
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

static TIERS: std::sync::OnceLock<Vec<TierTemplate>> = std::sync::OnceLock::new();

pub fn get_or_load_tiers() -> &'static Vec<TierTemplate> {
    TIERS.get_or_init(|| {
        let dir = std::path::Path::new("dataset/campaign/unit_tiers");
        let mut tiers = load_tiers_from_dir(dir);
        tiers.sort_by_key(|t| t.tier_index);
        if !tiers.is_empty() {
            eprintln!("Unit tiers: loaded {} tiers", tiers.len());
        }
        tiers
    })
}

fn load_tiers_from_dir(dir: &std::path::Path) -> Vec<TierTemplate> {
    let mut tiers = Vec::new();
    if !dir.exists() {
        return tiers;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return tiers,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => match toml::from_str::<TierTemplate>(&content) {
                Ok(tier) => tiers.push(tier),
                Err(e) => eprintln!("Warning: failed to parse tier {}: {}", path.display(), e),
            },
            Err(e) => eprintln!("Warning: failed to read {}: {}", path.display(), e),
        }
    }
    tiers
}

// ---------------------------------------------------------------------------
// Scaling computation
// ---------------------------------------------------------------------------

/// Compute the effective stat multiplier for a unit at a given tier and fame.
pub fn compute_tier_multiplier(
    status: &UnitTierStatus,
    state: &CampaignState,
) -> f32 {
    let tiers = get_or_load_tiers();
    let tier = match tiers.iter().find(|t| t.tier_index == status.tier) {
        Some(t) => t,
        None => return 1.0,
    };

    let scaling_value = match tier.scaling.source.as_str() {
        "none" => 0.0,
        "party_size" => {
            // Find the party this adventurer is in
            4.0 // default party size estimate
        }
        "faction_strength" => {
            match &status.champion_of {
                Some(ChampionOf::Faction(fid)) => {
                    state.factions.iter()
                        .find(|f| f.id == *fid)
                        .map(|f| f.military_strength)
                        .unwrap_or(0.0)
                }
                Some(ChampionOf::Guild) => {
                    state.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead)
                        .count() as f32 * 10.0
                }
                _ => 0.0,
            }
        }
        "coalition_strength" => {
            match &status.champion_of {
                Some(ChampionOf::Coalition(fids)) => {
                    fids.iter()
                        .filter_map(|fid| state.factions.iter().find(|f| f.id == *fid))
                        .map(|f| f.military_strength)
                        .sum()
                }
                _ => 0.0,
            }
        }
        "crisis_severity" => {
            // Sum of all active crisis severities
            state.overworld.active_crises.iter().map(|c| match c {
                ActiveCrisis::SleepingKing { champions_arrived, .. } => {
                    *champions_arrived as f32 * 50.0
                }
                ActiveCrisis::Breach { wave_strength, wave_number, .. } => {
                    *wave_strength + *wave_number as f32 * 10.0
                }
                ActiveCrisis::Corruption { corrupted_regions, .. } => {
                    corrupted_regions.len() as f32 * 30.0
                }
                ActiveCrisis::Unifier { absorbed_factions, .. } => {
                    absorbed_factions.len() as f32 * 40.0
                }
                ActiveCrisis::Decline { severity, .. } => {
                    *severity * 20.0
                }
            }).sum()
        }
        "fame" => status.fame.max(status.peak_fame),
        _ => 0.0,
    };

    let raw_bonus = tier.scaling.coefficient * scaling_value;

    // Apply candidacy fraction if applicable
    let fraction = if status.is_candidate {
        tier.candidacy.as_ref()
            .map(|c| c.bonus_fraction_as_candidate)
            .unwrap_or(0.5)
    } else {
        1.0
    };

    1.0 + (raw_bonus / 100.0) * fraction
}

/// Get the flat bonuses for a tier, accounting for candidacy.
pub fn compute_tier_flat_bonuses(status: &UnitTierStatus) -> TierBonuses {
    let tiers = get_or_load_tiers();
    let tier = match tiers.iter().find(|t| t.tier_index == status.tier) {
        Some(t) => t,
        None => return TierBonuses::default(),
    };

    let fraction = if status.is_candidate {
        tier.candidacy.as_ref()
            .map(|c| c.bonus_fraction_as_candidate)
            .unwrap_or(0.5)
    } else {
        1.0
    };

    TierBonuses {
        attack: tier.flat_bonuses.attack * fraction,
        defense: tier.flat_bonuses.defense * fraction,
        max_hp: tier.flat_bonuses.max_hp * fraction,
        resolve: tier.flat_bonuses.resolve * fraction,
        speed: tier.flat_bonuses.speed * fraction,
        ability_power: tier.flat_bonuses.ability_power * fraction,
    }
}

/// Check if an adventurer meets the requirements for a tier.
pub fn meets_tier_requirements(
    tier_index: u32,
    adventurer: &Adventurer,
    status: &UnitTierStatus,
    state: &CampaignState,
) -> bool {
    let tiers = get_or_load_tiers();
    let tier = match tiers.iter().find(|t| t.tier_index == tier_index) {
        Some(t) => t,
        None => return false,
    };

    let req = &tier.requirements;

    if adventurer.level < req.min_level { return false; }
    if status.quests_completed < req.min_quests_completed { return false; }
    if status.fame < req.min_fame { return false; }
    if status.party_victories < req.min_party_victories { return false; }
    if status.tier < req.min_prior_tier { return false; }

    if req.min_group_size > 0 {
        let alive = state.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead)
            .count();
        if alive < req.min_group_size { return false; }
    }

    if req.min_allied_factions > 0 {
        let allied = state.factions.iter()
            .filter(|f| matches!(f.diplomatic_stance, DiplomaticStance::Friendly | DiplomaticStance::Coalition))
            .count();
        if allied < req.min_allied_factions { return false; }
    }

    if req.requires_active_crisis && state.overworld.active_crises.is_empty() {
        return false;
    }

    if req.min_gold_invested > 0.0 && status.gold_invested < req.min_gold_invested {
        return false;
    }

    true
}
