//! Battle tick — every tick.
//!
//! Updates ongoing battles. In Oracle mode, battles resolve over a predicted
//! number of ticks with health ratios interpolated. In FullSim mode, the
//! actual combat sim would run (not yet implemented).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{BattleStatus, CampaignState, CombatMode};

/// How many campaign ticks a battle lasts (oracle mode estimate).
/// Real duration would come from the oracle's predicted ticks.
const DEFAULT_BATTLE_DURATION_TICKS: u64 = 50;

pub fn tick_battles(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    for battle in &mut state.active_battles {
        if battle.status != BattleStatus::Active {
            continue;
        }

        battle.elapsed_ticks += 1;

        match state.combat_mode {
            CombatMode::Oracle => {
                // Interpolate toward predicted outcome over expected duration
                let progress =
                    (battle.elapsed_ticks as f32 / DEFAULT_BATTLE_DURATION_TICKS as f32).min(1.0);

                if battle.predicted_outcome > 0.0 {
                    // Trending toward victory
                    battle.enemy_health_ratio =
                        (1.0 - progress * (1.0 - 0.0)).max(0.0);
                    battle.party_health_ratio =
                        (1.0 - progress * (1.0 - battle.predicted_outcome.abs() * 0.3))
                            .max(0.1);
                } else {
                    // Trending toward defeat
                    battle.party_health_ratio =
                        (1.0 - progress * (1.0 - 0.0)).max(0.0);
                    battle.enemy_health_ratio =
                        (1.0 - progress * (1.0 - battle.predicted_outcome.abs() * 0.3))
                            .max(0.1);
                }

                // Emit updates every 10 ticks
                if battle.elapsed_ticks % 10 == 0 {
                    events.push(WorldEvent::BattleUpdate {
                        battle_id: battle.id,
                        party_health_ratio: battle.party_health_ratio,
                        enemy_health_ratio: battle.enemy_health_ratio,
                    });
                }

                // Resolve when duration exceeded
                if battle.elapsed_ticks >= DEFAULT_BATTLE_DURATION_TICKS {
                    if battle.predicted_outcome > 0.0 {
                        battle.status = BattleStatus::Victory;
                    } else {
                        battle.status = BattleStatus::Defeat;
                    }
                    deltas.battles_ended += 1;
                    events.push(WorldEvent::BattleEnded {
                        battle_id: battle.id,
                        result: battle.status,
                    });
                }
            }
            CombatMode::FullSim => {
                // TODO: Integrate with run_rl_episode() for full combat sim.
                // For now, fall back to oracle-like behavior.
            }
        }
    }
}
