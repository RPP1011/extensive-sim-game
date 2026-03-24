//! Adventurer recovery — every 100 ticks (~10s).
//!
//! Injured adventurers recover over time. Reactivated when injury drops
//! below threshold.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{AdventurerStatus, CampaignState};

pub fn tick_adventurer_recovery(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let cfg = &state.config.adventurer_recovery;
    if state.tick % cfg.recovery_interval_ticks != 0 {
        return;
    }

    let stress_recovery = cfg.stress_recovery;
    let fatigue_recovery = cfg.fatigue_recovery;
    let injury_recovery = cfg.injury_recovery;
    let loyalty_recovery = cfg.loyalty_recovery;
    let injury_threshold = cfg.injury_threshold;
    let fatigue_threshold = cfg.fatigue_threshold;

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        // All non-dead adventurers recover passively
        // Idle adventurers recover faster, active ones recover slowly
        let recovery_rate = match adv.status {
            AdventurerStatus::Idle => 1.0,
            AdventurerStatus::Injured => 0.8,
            AdventurerStatus::Assigned => 0.5,
            AdventurerStatus::Traveling | AdventurerStatus::OnMission => 0.2,
            AdventurerStatus::Fighting => 0.0, // no recovery in combat
            _ => 0.3,
        };

        adv.stress = (adv.stress - stress_recovery * recovery_rate).max(0.0);
        adv.fatigue = (adv.fatigue - fatigue_recovery * recovery_rate).max(0.0);
        adv.loyalty = (adv.loyalty + loyalty_recovery * recovery_rate).min(100.0);

        // Injury only recovers for Idle/Injured (not on mission)
        if matches!(adv.status, AdventurerStatus::Idle | AdventurerStatus::Injured | AdventurerStatus::Assigned) {
            adv.injury = (adv.injury - injury_recovery * recovery_rate).max(0.0);
        }

        // Reactivate injured adventurers when recovered enough
        if adv.status == AdventurerStatus::Injured
            && adv.injury <= injury_threshold
            && adv.fatigue <= fatigue_threshold
        {
            adv.status = AdventurerStatus::Idle;
            events.push(WorldEvent::AdventurerRecovered {
                adventurer_id: adv.id,
            });
        }
    }
}
