//! Adventurer condition drift — every 10 ticks (~1s).
//!
//! Stress, fatigue, and morale drift based on adventurer status.

use crate::actions::{AdventurerStatDelta, StepDeltas, WorldEvent};
use crate::state::{AdventurerStatus, CampaignState};

pub fn tick_adventurer_condition(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let cfg = &state.config.adventurer_condition;
    if state.tick % cfg.drift_interval_ticks != 0 {
        return;
    }

    let fighting_drift = cfg.fighting_drift;
    let on_mission_drift = cfg.on_mission_drift;
    let idle_drift = cfg.idle_drift;
    let injured_drift = cfg.injured_drift;
    let desertion_loyalty = cfg.desertion_loyalty_threshold;
    let desertion_stress = cfg.desertion_stress_threshold;

    // Pre-compute party membership for bond lookups (avoids borrow conflict).
    let party_members_map: std::collections::HashMap<u32, Vec<u32>> = state
        .parties
        .iter()
        .flat_map(|p| {
            p.member_ids.iter().map(move |&id| (id, p.member_ids.clone()))
        })
        .collect();

    // Clone bonds ref for the loop (HashMap is behind &mut state).
    let bonds = state.adventurer_bonds.clone();

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        let [stress_d, fatigue_d, mut morale_d, loyalty_d] = match adv.status {
            AdventurerStatus::Fighting => fighting_drift,
            AdventurerStatus::OnMission | AdventurerStatus::Traveling => on_mission_drift,
            AdventurerStatus::Idle => idle_drift,
            AdventurerStatus::Injured => injured_drift,
            _ => [0.0, 0.0, 0.0, 0.0],
        };

        // Bond morale bonus: +5% when in party with bonded ally (bond > 30).
        if let Some(party_ids) = party_members_map.get(&adv.id) {
            let bonus = super::bonds::morale_bonus(&bonds, adv.id, party_ids);
            if bonus > 0.0 {
                morale_d += bonus * 0.1; // Scale to per-drift-tick increment
            }
        }

        adv.stress = (adv.stress + stress_d).clamp(0.0, 100.0);
        adv.fatigue = (adv.fatigue + fatigue_d).clamp(0.0, 100.0);
        adv.morale = (adv.morale + morale_d).clamp(0.0, 100.0);
        adv.loyalty = (adv.loyalty + loyalty_d).clamp(0.0, 100.0);

        // Desertion check: low loyalty + high stress.
        // Battle Brothers (bond > 80) resist desertion.
        if adv.loyalty <= desertion_loyalty && adv.stress >= desertion_stress && adv.status == AdventurerStatus::Idle {
            let party_ids = party_members_map.get(&adv.id).map(|v| v.as_slice()).unwrap_or(&[]);
            if !super::bonds::has_battle_brothers(&bonds, adv.id, party_ids) {
                adv.status = AdventurerStatus::Dead; // "deserted" = removed from play
                events.push(WorldEvent::AdventurerDeserted {
                    adventurer_id: adv.id,
                    reason: "Low loyalty and high stress".into(),
                });
            }
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
