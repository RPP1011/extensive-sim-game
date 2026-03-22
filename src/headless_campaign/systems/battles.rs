//! Battle tick — every tick.
//!
//! Updates ongoing battles. In Oracle mode, battles resolve over a predicted
//! number of ticks with health ratios interpolated. In FullSim mode, the
//! actual combat sim would run (not yet implemented).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{BattleStatus, CampaignState, CombatMode};

pub fn tick_battles(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let duration_ticks = state.config.battle.default_duration_ticks;
    let victory_damage = state.config.battle.victory_party_damage;
    let defeat_damage = state.config.battle.defeat_enemy_damage;
    let update_interval = state.config.battle.update_interval_ticks;

    for battle in &mut state.active_battles {
        if battle.status != BattleStatus::Active {
            continue;
        }

        battle.elapsed_ticks += 1;

        match state.combat_mode {
            CombatMode::Oracle => {
                // Interpolate toward predicted outcome over expected duration
                let progress =
                    (battle.elapsed_ticks as f32 / duration_ticks as f32).min(1.0);

                if battle.predicted_outcome > 0.0 {
                    // Trending toward victory
                    battle.enemy_health_ratio =
                        (1.0 - progress * (1.0 - 0.0)).max(0.0);
                    battle.party_health_ratio =
                        (1.0 - progress * (1.0 - battle.predicted_outcome.abs() * victory_damage))
                            .max(0.1);
                } else {
                    // Trending toward defeat
                    battle.party_health_ratio =
                        (1.0 - progress * (1.0 - 0.0)).max(0.0);
                    battle.enemy_health_ratio =
                        (1.0 - progress * (1.0 - battle.predicted_outcome.abs() * defeat_damage))
                            .max(0.1);
                }

                // Emit updates at configured interval
                if battle.elapsed_ticks % update_interval == 0 {
                    events.push(WorldEvent::BattleUpdate {
                        battle_id: battle.id,
                        party_health_ratio: battle.party_health_ratio,
                        enemy_health_ratio: battle.enemy_health_ratio,
                    });
                }

                // Resolve when duration exceeded
                if battle.elapsed_ticks >= duration_ticks {
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
