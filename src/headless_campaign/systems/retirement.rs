//! Adventurer retirement and legacy system — fires every 500 ticks.
//!
//! High-level adventurers can retire, becoming mentors or NPCs that provide
//! lasting bonuses to the guild. Legacy type is determined by archetype and
//! history tags. Bonuses stack up to 3 of the same type.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to check for retirement eligibility (in ticks).
const RETIREMENT_INTERVAL: u64 = 500;

/// Minimum level required for retirement.
const MIN_RETIREMENT_LEVEL: u32 = 10;

/// Minimum morale required for retirement.
const MIN_RETIREMENT_MORALE: f32 = 70.0;

/// Minimum loyalty required for retirement.
const MIN_RETIREMENT_LOYALTY: f32 = 60.0;

/// Chance per eligible adventurer per check to auto-retire.
const AUTO_RETIRE_CHANCE: f32 = 0.10;

/// Maximum number of retired adventurers with the same legacy type whose
/// bonuses stack.
const MAX_SAME_TYPE_STACK: usize = 3;

/// Check for and process adventurer retirements every `RETIREMENT_INTERVAL` ticks.
pub fn tick_retirement(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % RETIREMENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Collect eligible adventurer IDs first to avoid borrow issues.
    let eligible_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| is_eligible_for_retirement(a))
        .map(|a| a.id)
        .collect();

    for adv_id in eligible_ids {
        let roll = lcg_f32(&mut state.rng);
        if roll < AUTO_RETIRE_CHANCE {
            retire_adventurer(state, adv_id, events);
        }
    }
}

/// Check if an adventurer meets retirement eligibility criteria.
fn is_eligible_for_retirement(adv: &Adventurer) -> bool {
    adv.status != AdventurerStatus::Dead
        && adv.level >= MIN_RETIREMENT_LEVEL
        && adv.morale > MIN_RETIREMENT_MORALE
        && adv.loyalty > MIN_RETIREMENT_LOYALTY
}

/// Retire an adventurer: mark them dead/removed, create a RetiredAdventurer,
/// and emit events. Called from both auto-retirement and manual action.
pub fn retire_adventurer(
    state: &mut CampaignState,
    adventurer_id: u32,
    events: &mut Vec<WorldEvent>,
) -> bool {
    let adv = match state.adventurers.iter().find(|a| a.id == adventurer_id) {
        Some(a) => a,
        None => return false,
    };

    if adv.status == AdventurerStatus::Dead {
        return false;
    }

    let name = adv.name.clone();
    let archetype = adv.archetype.clone();
    let level = adv.level;
    let history_tags = adv.history_tags.clone();

    // Select legacy type based on archetype + history tags
    let legacy_type = select_legacy_type(&archetype, &history_tags, &mut state.rng);

    // Scale bonus by level (base * level / 10)
    let bonus_value = legacy_type.base_bonus() * (level as f32 / 10.0);

    let retired = RetiredAdventurer {
        id: adventurer_id,
        name: name.clone(),
        archetype: archetype.clone(),
        level,
        legacy_type: legacy_type.clone(),
        bonus_value,
        retired_at_tick: state.tick,
    };

    let bonus_desc = format!("{} ({:.1})", legacy_type.bonus_description(), bonus_value);

    // Mark adventurer as dead (retired) and remove from party
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        if let Some(pid) = adv.party_id {
            // Remove from party
            if let Some(party) = state.parties.iter_mut().find(|p| p.id == pid) {
                party.member_ids.retain(|&id| id != adventurer_id);
            }
        }
        adv.status = AdventurerStatus::Dead;
        adv.party_id = None;
    }

    // Emit retirement event
    events.push(WorldEvent::AdventurerRetired {
        adventurer_id,
        name: name.clone(),
        legacy_type: format!("{:?}", retired.legacy_type),
        bonus_description: bonus_desc.clone().to_string(),
    });

    // Check stacking limit
    let same_type_count = state
        .retired_adventurers
        .iter()
        .filter(|r| std::mem::discriminant(&r.legacy_type) == std::mem::discriminant(&retired.legacy_type))
        .count();

    if same_type_count < MAX_SAME_TYPE_STACK {
        events.push(WorldEvent::LegacyBonusApplied {
            legacy_type: format!("{:?}", retired.legacy_type),
            description: bonus_desc.to_string(),
        });
    }

    state.retired_adventurers.push(retired);
    true
}

/// Select a legacy type based on archetype and history tags.
fn select_legacy_type(
    archetype: &str,
    history_tags: &std::collections::HashMap<String, u32>,
    rng: &mut u64,
) -> LegacyType {
    // History tag overrides first
    if history_tags.get("diplomatic").copied().unwrap_or(0) > 3 {
        return LegacyType::Diplomat;
    }
    if history_tags.get("solo").copied().unwrap_or(0) > 5 {
        return LegacyType::TrainingMentor;
    }

    // Archetype-based selection with random tiebreak
    let roll = lcg_f32(rng);
    match archetype {
        "knight" | "paladin" | "guardian" | "tank" | "berserker" | "warrior" => {
            if roll < 0.5 {
                LegacyType::Strategist
            } else {
                LegacyType::TrainingMentor
            }
        }
        "rogue" | "assassin" | "monk" => {
            if roll < 0.5 {
                LegacyType::Quartermaster
            } else {
                LegacyType::Recruiter
            }
        }
        "mage" | "cleric" | "druid" | "shaman" | "warlock" | "necromancer" | "healer" => {
            if roll < 0.5 {
                LegacyType::Lorekeeper
            } else {
                LegacyType::Diplomat
            }
        }
        "bard" => {
            if roll < 0.5 {
                LegacyType::Diplomat
            } else {
                LegacyType::Recruiter
            }
        }
        "ranger" | "artificer" => {
            if roll < 0.5 {
                LegacyType::Quartermaster
            } else {
                LegacyType::Strategist
            }
        }
        _ => {
            // Fallback: pick uniformly
            let idx = (lcg_next(rng) as usize) % 6;
            match idx {
                0 => LegacyType::TrainingMentor,
                1 => LegacyType::Quartermaster,
                2 => LegacyType::Diplomat,
                3 => LegacyType::Strategist,
                4 => LegacyType::Recruiter,
                _ => LegacyType::Lorekeeper,
            }
        }
    }
}

/// Compute aggregate legacy bonuses from all retired adventurers.
/// Respects the stacking limit (max 3 of same type).
pub fn legacy_bonus(retired: &[RetiredAdventurer], legacy_type: &LegacyType) -> f32 {
    let mut matching: Vec<f32> = retired
        .iter()
        .filter(|r| std::mem::discriminant(&r.legacy_type) == std::mem::discriminant(legacy_type))
        .map(|r| r.bonus_value)
        .collect();

    // Sort descending to keep the best bonuses within the stack limit
    matching.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    matching.truncate(MAX_SAME_TYPE_STACK);
    matching.iter().sum()
}

/// Convenience: total XP multiplier from TrainingMentor legacies.
/// Returns a multiplier >= 1.0 (e.g. 1.1 for +10%).
pub fn xp_multiplier(retired: &[RetiredAdventurer]) -> f32 {
    1.0 + legacy_bonus(retired, &LegacyType::TrainingMentor)
}

/// Convenience: supply consumption multiplier from Quartermaster legacies.
/// Returns a multiplier <= 1.0 (e.g. 0.9 for -10%).
pub fn supply_multiplier(retired: &[RetiredAdventurer]) -> f32 {
    1.0 - legacy_bonus(retired, &LegacyType::Quartermaster)
}

/// Convenience: faction relation bonus from Diplomat legacies.
pub fn diplomat_bonus(retired: &[RetiredAdventurer]) -> f32 {
    legacy_bonus(retired, &LegacyType::Diplomat)
}

/// Convenience: combat power multiplier from Strategist legacies.
/// Returns a multiplier >= 1.0 (e.g. 1.05 for +5%).
pub fn combat_power_multiplier(retired: &[RetiredAdventurer]) -> f32 {
    1.0 + legacy_bonus(retired, &LegacyType::Strategist)
}

/// Convenience: base stat bonus for new recruits from Recruiter legacies.
pub fn recruit_stat_bonus(retired: &[RetiredAdventurer]) -> f32 {
    legacy_bonus(retired, &LegacyType::Recruiter)
}

/// Convenience: quest gold reward multiplier from Lorekeeper legacies.
/// Returns a multiplier >= 1.0 (e.g. 1.1 for +10%).
pub fn quest_reward_multiplier(retired: &[RetiredAdventurer]) -> f32 {
    1.0 + legacy_bonus(retired, &LegacyType::Lorekeeper)
}
