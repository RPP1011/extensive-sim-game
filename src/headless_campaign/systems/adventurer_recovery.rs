//! Adventurer recovery — every 100 ticks (~10s).
//!
//! Injured adventurers recover over time. Reactivated when injury drops
//! below threshold.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{AdventurerStatus, CampaignState};

const RECOVERY_INJURY_THRESHOLD: f32 = 40.0;
const RECOVERY_FATIGUE_THRESHOLD: f32 = 40.0;

pub fn tick_adventurer_recovery(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 100 != 0 {
        return;
    }

    for adv in &mut state.adventurers {
        if adv.status != AdventurerStatus::Injured {
            continue;
        }

        adv.stress = (adv.stress - 3.0).max(0.0);
        adv.fatigue = (adv.fatigue - 4.0).max(0.0);
        adv.injury = (adv.injury - 3.5).max(0.0);
        adv.loyalty = (adv.loyalty + 0.7).min(100.0);

        if adv.injury <= RECOVERY_INJURY_THRESHOLD && adv.fatigue <= RECOVERY_FATIGUE_THRESHOLD {
            adv.status = AdventurerStatus::Idle;
            events.push(WorldEvent::AdventurerRecovered {
                adventurer_id: adv.id,
            });
        }
    }
}
