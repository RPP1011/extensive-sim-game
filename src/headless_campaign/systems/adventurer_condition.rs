//! Adventurer condition drift — every 10 ticks (~1s).
//!
//! Stress, fatigue, and morale drift based on adventurer status.

use crate::headless_campaign::actions::{AdventurerStatDelta, StepDeltas, WorldEvent};
use crate::headless_campaign::state::{AdventurerStatus, CampaignState};

pub fn tick_adventurer_condition(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 10 != 0 {
        return;
    }

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        let (stress_d, fatigue_d, morale_d, loyalty_d) = match adv.status {
            AdventurerStatus::Fighting => (0.8, 0.6, -0.5, -0.1),
            AdventurerStatus::OnMission | AdventurerStatus::Traveling => (0.3, 0.4, -0.2, 0.0),
            AdventurerStatus::Idle => (-0.5, -0.3, 0.3, 0.05),
            AdventurerStatus::Injured => (-0.2, -0.5, -0.1, -0.05),
            _ => (0.0, 0.0, 0.0, 0.0),
        };

        adv.stress = (adv.stress + stress_d).clamp(0.0, 100.0);
        adv.fatigue = (adv.fatigue + fatigue_d).clamp(0.0, 100.0);
        adv.morale = (adv.morale + morale_d).clamp(0.0, 100.0);
        adv.loyalty = (adv.loyalty + loyalty_d).clamp(0.0, 100.0);

        // Desertion check: low loyalty + high stress
        if adv.loyalty <= 10.0 && adv.stress >= 70.0 && adv.status == AdventurerStatus::Idle {
            adv.status = AdventurerStatus::Dead; // "deserted" = removed from play
            events.push(WorldEvent::AdventurerDeserted {
                adventurer_id: adv.id,
                reason: "Low loyalty and high stress".into(),
            });
        }

        if stress_d.abs() > 0.01 || fatigue_d.abs() > 0.01 {
            deltas.adventurer_stat_changes.push(AdventurerStatDelta {
                adventurer_id: adv.id,
                stress_delta: stress_d,
                fatigue_delta: fatigue_d,
                injury_delta: 0.0,
                loyalty_delta: loyalty_d,
                morale_delta: morale_d,
                source: "condition_drift".into(),
            });
        }
    }
}
