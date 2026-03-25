//! Alchemical potion addiction system — every 100 ticks.
//!
//! Potions have dependency ratings; exceeding thresholds triggers withdrawal
//! debuffs, creating potion-economy tradeoffs.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{AdventurerStatus, CampaignState, lcg_f32};

/// Addiction tick cadence (every 100 ticks = 10 seconds game time).
const ADDICTION_TICK_INTERVAL: u64 = 100;

/// Dependency increase per healing potion consumed.
const HEALING_POTION_DEPENDENCY: f32 = 0.03;
/// Dependency increase per combat buff potion consumed.
const BUFF_POTION_DEPENDENCY: f32 = 0.05;
/// Natural dependency decay per tick when no potions used.
const DEPENDENCY_DECAY: f32 = 0.005;
/// Dependency threshold above which withdrawal can trigger.
const WITHDRAWAL_DEPENDENCY_THRESHOLD: f32 = 0.3;
/// Ticks since last potion before withdrawal kicks in.
const WITHDRAWAL_TICKS_THRESHOLD: u32 = 500;
/// Dependency threshold below which withdrawal_severity recovers.
const RECOVERY_DEPENDENCY_THRESHOLD: f32 = 0.2;
/// Clean bonus: stat multiplier when dependency is exactly 0.
const CLEAN_STAT_BONUS: f32 = 0.05;

/// Process addiction states for all adventurers.
pub fn tick_addiction(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % ADDICTION_TICK_INTERVAL != 0 {
        return;
    }

    for adv in &mut state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        let prev_dependency = adv.potion_dependency;
        let prev_withdrawal = adv.withdrawal_severity;

        // --- Natural decay: dependency decreases when no potions used ---
        adv.ticks_since_last_potion = adv.ticks_since_last_potion.saturating_add(1);
        adv.potion_dependency = (adv.potion_dependency - DEPENDENCY_DECAY).max(0.0);

        // --- Withdrawal onset ---
        if adv.potion_dependency > WITHDRAWAL_DEPENDENCY_THRESHOLD
            && adv.ticks_since_last_potion > WITHDRAWAL_TICKS_THRESHOLD
        {
            // Withdrawal severity scales with dependency level
            let target_severity = (adv.potion_dependency - WITHDRAWAL_DEPENDENCY_THRESHOLD)
                / (1.0 - WITHDRAWAL_DEPENDENCY_THRESHOLD);
            // Ramp up gradually
            adv.withdrawal_severity =
                (adv.withdrawal_severity + 0.05).min(target_severity).clamp(0.0, 1.0);

            // Emit onset event when crossing into withdrawal
            if prev_withdrawal < 0.01 && adv.withdrawal_severity >= 0.01 {
                events.push(WorldEvent::WithdrawalOnset {
                    adventurer_id: adv.id,
                    severity: adv.withdrawal_severity,
                });
            }

            // Apply withdrawal effects based on severity
            if adv.withdrawal_severity >= 0.6 {
                // Severe: -30 morale, +30 stress, risk of desertion
                adv.morale = (adv.morale - 30.0 * 0.01).max(0.0); // Scaled per tick
                adv.stress = (adv.stress + 30.0 * 0.01).min(100.0);

                // Risk of desertion (5% per tick at severe withdrawal, idle only)
                if adv.status == AdventurerStatus::Idle {
                    let roll = lcg_f32(&mut state.rng);
                    if roll < 0.05 {
                        adv.status = AdventurerStatus::Dead;
                        events.push(WorldEvent::AdventurerDeserted {
                            adventurer_id: adv.id,
                            reason: "Severe potion withdrawal".into(),
                        });
                        continue;
                    }
                }
            } else if adv.withdrawal_severity >= 0.3 {
                // Moderate: -15 morale, +15 stress
                adv.morale = (adv.morale - 15.0 * 0.01).max(0.0);
                adv.stress = (adv.stress + 15.0 * 0.01).min(100.0);
            } else if adv.withdrawal_severity >= 0.1 {
                // Mild: -5 morale, +5 stress
                adv.morale = (adv.morale - 5.0 * 0.01).max(0.0);
                adv.stress = (adv.stress + 5.0 * 0.01).min(100.0);
            }
        }

        // --- Recovery: withdrawal eases when dependency drops below threshold ---
        if adv.potion_dependency < RECOVERY_DEPENDENCY_THRESHOLD {
            adv.withdrawal_severity = (adv.withdrawal_severity - 0.02).max(0.0);

            // Emit overcome event when fully clean after having been addicted
            if prev_dependency >= RECOVERY_DEPENDENCY_THRESHOLD
                && adv.potion_dependency < RECOVERY_DEPENDENCY_THRESHOLD
                && adv.withdrawal_severity < 0.01
            {
                events.push(WorldEvent::AddictionOvercome {
                    adventurer_id: adv.id,
                });
            }
        }

        // --- Addiction developed event ---
        // Emit when crossing the withdrawal threshold upward
        if prev_dependency <= WITHDRAWAL_DEPENDENCY_THRESHOLD
            && adv.potion_dependency > WITHDRAWAL_DEPENDENCY_THRESHOLD
        {
            events.push(WorldEvent::AddictionDeveloped {
                adventurer_id: adv.id,
                dependency_level: adv.potion_dependency,
            });
        }
    }
}

/// Record a healing potion consumption for an adventurer.
/// Called from other systems (battles, recovery, etc.) when potions are used.
pub fn record_healing_potion(state: &mut CampaignState, adventurer_id: u32) {
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        adv.potion_dependency = (adv.potion_dependency + HEALING_POTION_DEPENDENCY).min(1.0);
        adv.ticks_since_last_potion = 0;
        adv.total_potions_consumed = adv.total_potions_consumed.saturating_add(1);
    }
}

/// Record a combat buff potion consumption for an adventurer.
/// Called from other systems when buff potions are used.
pub fn record_buff_potion(state: &mut CampaignState, adventurer_id: u32) {
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        adv.potion_dependency = (adv.potion_dependency + BUFF_POTION_DEPENDENCY).min(1.0);
        adv.ticks_since_last_potion = 0;
        adv.total_potions_consumed = adv.total_potions_consumed.saturating_add(1);
    }
}

/// Combat stat multiplier from addiction/withdrawal state.
/// - Clean (0 dependency): +5% stats
/// - Moderate withdrawal (0.3-0.6): -10% stats
/// - Severe withdrawal (0.6+): -25% stats
pub fn addiction_combat_multiplier(dependency: f32, withdrawal: f32) -> f32 {
    if dependency == 0.0 {
        1.0 + CLEAN_STAT_BONUS // Clear-headed bonus
    } else if withdrawal >= 0.6 {
        0.75 // Severe: -25%
    } else if withdrawal >= 0.3 {
        0.90 // Moderate: -10%
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::state::{
        Adventurer, AdventurerStats, CampaignState, DiseaseStatus, Equipment, MoodState,
    };

    fn test_adventurer(id: u32) -> Adventurer {
        Adventurer {
            id,
            name: format!("Test_{}", id),
            archetype: "ranger".into(),
            level: 3,
            xp: 0,
            stats: AdventurerStats {
                max_hp: 80.0,
                attack: 14.0,
                defense: 8.0,
                speed: 12.0,
                ability_power: 6.0,
            },
            equipment: Equipment::default(),
            traits: Vec::new(),
            status: AdventurerStatus::Idle,
            loyalty: 60.0,
            stress: 10.0,
            fatigue: 5.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 70.0,
            party_id: None,
            guild_relationship: 40.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),
            backstory: None,
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: DiseaseStatus::Healthy,
            mood_state: MoodState::default(),
            fears: Vec::new(),
            personal_goal: None,
            journal: Vec::new(),
            equipped_items: Vec::new(),
            nicknames: Vec::new(),
            secret_past: None,
            wounds: Vec::new(),
            potion_dependency: 0.0,
            withdrawal_severity: 0.0,
            ticks_since_last_potion: 0,
            total_potions_consumed: 0,
            behavior_ledger: BehaviorLedger::default(),
            classes: Vec::new(),
            skill_state: Default::default(),
        }
    }

    #[test]
    fn clean_adventurer_gets_stat_bonus() {
        let mult = addiction_combat_multiplier(0.0, 0.0);
        assert!((mult - 1.05).abs() < 0.001);
    }

    #[test]
    fn severe_withdrawal_reduces_stats() {
        let mult = addiction_combat_multiplier(0.8, 0.7);
        assert!((mult - 0.75).abs() < 0.001);
    }

    #[test]
    fn moderate_withdrawal_reduces_stats() {
        let mult = addiction_combat_multiplier(0.5, 0.4);
        assert!((mult - 0.90).abs() < 0.001);
    }

    #[test]
    fn healing_potion_increases_dependency() {
        let mut state = CampaignState::default_test_campaign(42);
        state.adventurers.push(test_adventurer(1));

        record_healing_potion(&mut state, 1);

        let adv = state.adventurers.iter().find(|a| a.id == 1).unwrap();
        assert!((adv.potion_dependency - HEALING_POTION_DEPENDENCY).abs() < 0.001);
        assert_eq!(adv.ticks_since_last_potion, 0);
        assert_eq!(adv.total_potions_consumed, 1);
    }

    #[test]
    fn buff_potion_increases_dependency_more() {
        let mut state = CampaignState::default_test_campaign(42);
        state.adventurers.push(test_adventurer(1));

        record_buff_potion(&mut state, 1);

        let adv = state.adventurers.iter().find(|a| a.id == 1).unwrap();
        assert!((adv.potion_dependency - BUFF_POTION_DEPENDENCY).abs() < 0.001);
        assert_eq!(adv.total_potions_consumed, 1);
    }

    #[test]
    fn dependency_decays_over_time() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut adv = test_adventurer(1);
        adv.potion_dependency = 0.1;
        state.adventurers.push(adv);

        state.tick = 100;
        let mut events = Vec::new();
        tick_addiction(&mut state, &mut StepDeltas::default(), &mut events);

        let adv = state.adventurers.iter().find(|a| a.id == 1).unwrap();
        assert!((adv.potion_dependency - 0.095).abs() < 0.001);
    }

    #[test]
    fn withdrawal_triggers_above_threshold() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut adv = test_adventurer(1);
        adv.potion_dependency = 0.5;
        adv.ticks_since_last_potion = 600;
        state.adventurers.push(adv);

        state.tick = 100;
        let mut events = Vec::new();
        tick_addiction(&mut state, &mut StepDeltas::default(), &mut events);

        let adv = state.adventurers.iter().find(|a| a.id == 1).unwrap();
        assert!(adv.withdrawal_severity > 0.0, "Withdrawal should have started");
        assert!(events.iter().any(|e| matches!(e, WorldEvent::WithdrawalOnset { .. })));
    }

    #[test]
    fn no_withdrawal_below_threshold() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut adv = test_adventurer(1);
        adv.potion_dependency = 0.2;
        adv.ticks_since_last_potion = 600;
        state.adventurers.push(adv);

        state.tick = 100;
        let mut events = Vec::new();
        tick_addiction(&mut state, &mut StepDeltas::default(), &mut events);

        let adv = state.adventurers.iter().find(|a| a.id == 1).unwrap();
        assert!(adv.withdrawal_severity < 0.01);
    }

    #[test]
    fn dependency_capped_at_1() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut adv = test_adventurer(1);
        adv.potion_dependency = 0.99;
        state.adventurers.push(adv);

        record_buff_potion(&mut state, 1);

        let adv = state.adventurers.iter().find(|a| a.id == 1).unwrap();
        assert!(adv.potion_dependency <= 1.0);
    }
}
