//! Wound persistence — every 100 ticks (~10s).
//!
//! Battle survivors can sustain persistent wounds that decay over time.
//! Wounds penalize stats until healed, forcing rest/roster rotation decisions.
//! Wound generation happens post-battle in `quest_lifecycle.rs`.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{AdventurerStatus, CampaignState, WoundType};

/// Heal interval in ticks (every 100 ticks = ~10 seconds of game time).
const WOUND_HEAL_INTERVAL: u64 = 3;

/// Per-tick heal rate when Idle.
const HEAL_RATE_IDLE: f32 = 0.01;
/// Per-tick heal rate when OnMission/Traveling/Assigned.
const HEAL_RATE_ON_MISSION: f32 = 0.005;
/// Per-tick heal rate when Fighting (no healing).
const HEAL_RATE_FIGHTING: f32 = 0.0;

pub fn tick_wound_persistence(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % WOUND_HEAL_INTERVAL != 0 {
        return;
    }

    // Collect (adventurer_id, wound_type) pairs for wounds that healed this tick.
    let mut healed: Vec<(u32, WoundType)> = Vec::new();

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead || adv.wounds.is_empty() {
            continue;
        }

        let heal_rate = match adv.status {
            AdventurerStatus::Idle | AdventurerStatus::Injured => HEAL_RATE_IDLE,
            AdventurerStatus::OnMission
            | AdventurerStatus::Traveling
            | AdventurerStatus::Assigned => HEAL_RATE_ON_MISSION,
            AdventurerStatus::Fighting => HEAL_RATE_FIGHTING,
            _ => HEAL_RATE_ON_MISSION,
        };

        // Apply BattleFatigue stress penalty before healing wounds.
        let has_fatigue = adv.wounds.iter().any(|w| w.wound_type == WoundType::BattleFatigue);
        if has_fatigue {
            adv.stress = (adv.stress + 10.0).min(100.0);
        }

        for wound in &mut adv.wounds {
            wound.heal_progress += heal_rate;
            if wound.heal_progress >= wound.severity {
                healed.push((adv.id, wound.wound_type));
            }
        }

        // Remove healed wounds.
        adv.wounds.retain(|w| w.heal_progress < w.severity);
    }

    for (adv_id, wound_type) in healed {
        events.push(WorldEvent::WoundHealed {
            adventurer_id: adv_id,
            wound_type,
        });
    }
}

/// Generate wounds for an adventurer after a battle, based on HP ratio.
///
/// Called from `quest_lifecycle.rs` when a battle completes.
/// - HP ratio < 0.3: 60% chance of wound
/// - HP ratio < 0.5: 30% chance of wound
/// - HP ratio >= 0.5: no wound
pub fn maybe_generate_wound(
    adv_id: u32,
    hp_ratio: f32,
    rng: &mut u64,
    events: &mut Vec<WorldEvent>,
) -> Option<super::super::state::PersistentWound> {
    use crate::headless_campaign::state::{lcg_f32, PersistentWound, WoundType};

    let chance = if hp_ratio < 0.3 {
        0.6
    } else if hp_ratio < 0.5 {
        0.3
    } else {
        return None;
    };

    let roll = lcg_f32(rng);
    if roll >= chance {
        return None;
    }

    // Pick wound type based on RNG.
    let type_roll = lcg_f32(rng);
    let wound_type = WoundType::ALL[(type_roll * WoundType::ALL.len() as f32) as usize
        % WoundType::ALL.len()];

    // Severity scales with damage taken.
    let severity = (1.0 - hp_ratio) * lcg_f32(rng);

    events.push(WorldEvent::WoundSustained {
        adventurer_id: adv_id,
        wound_type,
        severity,
    });

    Some(PersistentWound {
        wound_type,
        severity,
        heal_progress: 0.0,
    })
}
