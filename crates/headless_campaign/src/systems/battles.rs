//! Battle tick — every tick.
//!
//! Updates ongoing battles. In Oracle mode, battles resolve over a predicted
//! number of ticks with health ratios interpolated. In TacticalSim mode,
//! the full deterministic combat sim runs on the first tick and resolves
//! immediately — generated abilities directly affect outcomes.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{BattleStatus, CampaignState, CombatMode};

pub fn tick_battles(
    state: &mut CampaignState,
    deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let duration_ticks = state.config.battle.default_duration_ticks;
    let victory_damage = state.config.battle.victory_party_damage;
    let defeat_damage = state.config.battle.defeat_enemy_damage;
    let update_interval = state.config.battle.update_interval_ticks;
    let combat_mode = state.combat_mode;

    // --- TacticalSim pre-pass: run combat for newly active battles ---
    // We must do this before the mutable loop because run_tactical_combat
    // needs immutable access to the full campaign state.
    #[cfg(feature = "combat-bridge")]
    if combat_mode == CombatMode::TacticalSim {
        // Collect (index, party_id, enemy_strength, seed) for battles needing resolution
        let to_resolve: Vec<(usize, u32, f32, u64)> = state
            .active_battles
            .iter()
            .enumerate()
            .filter(|(_, b)| b.status == BattleStatus::Active && b.elapsed_ticks == 0)
            .map(|(i, b)| {
                let seed = state.rng ^ (b.id as u64 * 6364136223846793005);
                (i, b.party_id, b.enemy_strength, seed)
            })
            .collect();

        // Run tactical combat for each pending battle
        let results: Vec<(usize, crate::combat_oracle::tactical_bridge::TacticalResult)> = to_resolve
            .into_iter()
            .map(|(idx, party_id, enemy_strength, seed)| {
                let members: Vec<&crate::state::Adventurer> = state
                    .adventurers
                    .iter()
                    .filter(|a| a.party_id == Some(party_id))
                    .collect();
                let result = crate::combat_oracle::tactical_bridge::run_tactical_combat(
                    &members,
                    enemy_strength,
                    seed,
                    state,
                );
                (idx, result)
            })
            .collect();

        // Apply results
        for (idx, result) in results {
            let battle = &mut state.active_battles[idx];
            battle.elapsed_ticks += 1;

            let avg_hp = if result.hp_remaining.is_empty() {
                0.0
            } else {
                result.hp_remaining.iter().sum::<f32>() / result.hp_remaining.len() as f32
            };

            if result.victory {
                battle.party_health_ratio = avg_hp.max(0.01);
                battle.enemy_health_ratio = 0.0;
                battle.status = BattleStatus::Victory;
            } else {
                battle.party_health_ratio = avg_hp;
                battle.enemy_health_ratio = 0.3;
                battle.status = BattleStatus::Defeat;
            }

            deltas.battles_ended += 1;
            events.push(WorldEvent::BattleEnded {
                battle_id: battle.id,
                result: battle.status,
            });
        }
    }

    // --- Oracle / FullSim tick-based resolution ---
    for battle in &mut state.active_battles {
        if battle.status != BattleStatus::Active {
            continue;
        }

        battle.elapsed_ticks += 1;

        match combat_mode {
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
            CombatMode::TacticalSim => {
                // Already resolved in pre-pass above. Any battles still active
                // at this point were already in progress before this tick.
            }
            CombatMode::FullSim => {
                // TODO: Integrate with run_rl_episode() for full combat sim.
                // For now, fall back to oracle-like behavior.
            }
        }
    }
}
