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
        if adv.status != AdventurerStatus::Injured {
            continue;
        }

        adv.stress = (adv.stress - stress_recovery).max(0.0);
        adv.fatigue = (adv.fatigue - fatigue_recovery).max(0.0);
        adv.injury = (adv.injury - injury_recovery).max(0.0);
        adv.loyalty = (adv.loyalty + loyalty_recovery).min(100.0);

        if adv.injury <= injury_threshold && adv.fatigue <= fatigue_threshold {
            adv.status = AdventurerStatus::Idle;
            events.push(WorldEvent::AdventurerRecovered {
                adventurer_id: adv.id,
            });
        }
    }
}
