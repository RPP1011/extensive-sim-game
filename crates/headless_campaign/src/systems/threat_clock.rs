//! World threat clock system — every 100 ticks.
//!
//! A world-ending entity accumulates power each tick unless periodically
//! disrupted, creating a global priority competing with local objectives.
//! At power thresholds (0.2, 0.4, 0.6, 0.8, 1.0) escalating effects fire.
//! Completing disruption quests or defeating manifestations reduces power.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often the threat clock advances (in ticks).
const CLOCK_INTERVAL: u64 = 3;

/// How often growth_rate accelerates (in ticks).
const GROWTH_ACCELERATION_INTERVAL: u64 = 33;

/// Growth rate acceleration per interval.
const GROWTH_ACCELERATION: f32 = 0.0001;

/// Power reduction range for disruptions [min, max].
const DISRUPTION_MIN: f32 = 0.1;
const DISRUPTION_MAX: f32 = 0.3;

/// Growth rate reduction per disruption.
const GROWTH_RATE_REDUCTION: f32 = 0.0002;

/// Minimum growth rate (can't be reduced below this).
const MIN_GROWTH_RATE: f32 = 0.001;

/// Tick at which the threat clock activates.
const ACTIVATION_TICK: u64 = 17;

/// Tick the world threat clock.
///
/// Every 100 ticks:
/// 1. Activate the clock if not yet active (after tick 500)
/// 2. Advance power by growth_rate
/// 3. Accelerate growth_rate every 1000 ticks
/// 4. Check for threshold crossings and emit events
/// 5. At power 1.0 — trigger endgame crisis
pub fn tick_threat_clock(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CLOCK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Activate on first eligible tick
    if !state.world_threat_clock.active {
        if state.tick >= ACTIVATION_TICK {
            activate_threat_clock(state, events);
        }
        return;
    }

    let clock = &mut state.world_threat_clock;

    // Already hit 1.0 — no further advancement
    if clock.power >= 1.0 {
        return;
    }

    let old_power = clock.power;

    // Advance power
    clock.power = (clock.power + clock.growth_rate).min(1.0);

    // Accelerate growth rate every 1000 ticks
    if state.tick % GROWTH_ACCELERATION_INTERVAL == 0 {
        clock.growth_rate += GROWTH_ACCELERATION;
    }

    // Emit advancement event
    let threshold_crossed = check_threshold_crossing(old_power, clock.power);
    events.push(WorldEvent::ThreatClockAdvanced {
        power: clock.power,
        threshold_crossed,
    });

    // Process threshold effects
    if let Some(threshold) = threshold_crossed {
        apply_threshold_effects(state, threshold, events);
    }
}

/// Select a threat type deterministically from the campaign RNG and activate.
fn activate_threat_clock(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let r = lcg_f32(&mut state.rng);
    let threat_type = match (r * 5.0) as u32 {
        0 => WorldThreat::AncientLich,
        1 => WorldThreat::DemonLord,
        2 => WorldThreat::VoidRift,
        3 => WorldThreat::DragonAwakening,
        _ => WorldThreat::BlightHeart,
    };

    // Randomize initial growth rate in [0.001, 0.005]
    let gr = lcg_f32(&mut state.rng);
    let growth_rate = 0.001 + gr * 0.004;

    state.world_threat_clock = WorldThreatClock {
        threat_type,
        power: 0.0,
        growth_rate,
        disruptions: 0,
        warnings_issued: 0,
        active: true,
    };

    events.push(WorldEvent::WorldThreatActivated { threat_type });
}

/// Check if a power threshold was crossed between old and new values.
fn check_threshold_crossing(old: f32, new: f32) -> Option<f32> {
    const THRESHOLDS: [f32; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];
    for &t in &THRESHOLDS {
        if old < t && new >= t {
            return Some(t);
        }
    }
    None
}

/// Apply world effects when a threat threshold is crossed.
fn apply_threshold_effects(
    state: &mut CampaignState,
    threshold: f32,
    events: &mut Vec<WorldEvent>,
) {
    let clock = &mut state.world_threat_clock;
    clock.warnings_issued += 1;

    let threat_type = clock.threat_type;
    let effect = match threshold as u32 {
        _ if (threshold - 0.2).abs() < 0.01 => {
            // Omens and rumors — mild warning
            state.rumors.push(Rumor {
                id: state.rumors.len() as u32,
                text: format!(
                    "Dark omens spread across the land — the {} stirs.",
                    threat_type_name(threat_type)
                ),
                rumor_type: RumorType::CrisisWarning,
                accuracy: 0.9,
                source_tick: state.tick,
                revealed: false,
                target_region_id: None,
                target_faction_id: None,
            });
            "omens_and_rumors".to_string()
        }
        _ if (threshold - 0.4).abs() < 0.01 => {
            // Minor manifestations — monster spawns increase, unrest rises
            for region in &mut state.overworld.regions {
                region.unrest = (region.unrest + 5.0).min(100.0);
            }
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level + 10.0).min(100.0);
            "minor_manifestations".to_string()
        }
        _ if (threshold - 0.6).abs() < 0.01 => {
            // Major effects — faction panic, trade disruption, refugees
            for faction in &mut state.factions {
                if faction.id != state.diplomacy.guild_faction_id {
                    faction.relationship_to_guild -= 5.0;
                }
            }
            for region in &mut state.overworld.regions {
                region.unrest = (region.unrest + 15.0).min(100.0);
            }
            state.overworld.global_threat_level =
                (state.overworld.global_threat_level + 20.0).min(100.0);
            "major_effects".to_string()
        }
        _ if (threshold - 0.8).abs() < 0.01 => {
            // Critical — random adventurer deaths, regions devastated
            // Kill the weakest adventurer if any are alive
            let weakest_id = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .min_by(|a, b| a.level.partial_cmp(&b.level).unwrap_or(std::cmp::Ordering::Equal))
                .map(|a| a.id);
            if let Some(id) = weakest_id {
                if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == id) {
                    adv.status = AdventurerStatus::Dead;
                    events.push(WorldEvent::AdventurerDied {
                        adventurer_id: id,
                        cause: format!(
                            "Slain by manifestation of the {}",
                            threat_type_name(threat_type)
                        ),
                    });
                }
            }
            for region in &mut state.overworld.regions {
                region.unrest = (region.unrest + 25.0).min(100.0);
                region.control = (region.control - 20.0).max(0.0);
            }
            "critical_devastation".to_string()
        }
        _ => {
            // 1.0 — endgame crisis forces a final battle
            state.overworld.global_threat_level = 100.0;
            // Set endgame calamity if not already set
            if state.overworld.endgame_calamity.is_none() {
                state.overworld.endgame_calamity = Some(CalamityType::MajorMonster {
                    name: threat_type_name(threat_type).to_string(),
                    strength: 100.0,
                });
            }
            "world_ending_crisis".to_string()
        }
    };

    events.push(WorldEvent::ThreatManifested {
        threat_type,
        effect,
    });
}

/// Apply a disruption to the threat clock (called externally when quests/battles
/// reduce the world threat).
pub fn disrupt_threat_clock(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    if !state.world_threat_clock.active || state.world_threat_clock.power <= 0.0 {
        return;
    }

    let clock = &mut state.world_threat_clock;

    // Randomize reduction in [DISRUPTION_MIN, DISRUPTION_MAX]
    let r = lcg_f32(&mut state.rng);
    let reduction = DISRUPTION_MIN + r * (DISRUPTION_MAX - DISRUPTION_MIN);

    let old_power = clock.power;
    clock.power = (clock.power - reduction).max(0.0);
    clock.disruptions += 1;

    // Slightly reduce growth rate
    clock.growth_rate = (clock.growth_rate - GROWTH_RATE_REDUCTION).max(MIN_GROWTH_RATE);

    events.push(WorldEvent::ThreatDisrupted {
        power_reduction: old_power - clock.power,
        new_power: clock.power,
    });
}

/// Human-readable name for a world threat type.
fn threat_type_name(threat: WorldThreat) -> &'static str {
    match threat {
        WorldThreat::AncientLich => "Ancient Lich",
        WorldThreat::DemonLord => "Demon Lord",
        WorldThreat::VoidRift => "Void Rift",
        WorldThreat::DragonAwakening => "Dragon Awakening",
        WorldThreat::BlightHeart => "Blight Heart",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threat_clock_activates_at_tick_500() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        // Before activation tick — nothing happens
        state.tick = 400;
        tick_threat_clock(&mut state, &mut deltas, &mut events);
        assert!(!state.world_threat_clock.active);
        assert!(events.is_empty());

        // At activation tick
        state.tick = 500;
        tick_threat_clock(&mut state, &mut deltas, &mut events);
        assert!(state.world_threat_clock.active);
        assert_eq!(state.world_threat_clock.power, 0.0);
        assert!(events.iter().any(|e| matches!(e, WorldEvent::WorldThreatActivated { .. })));
    }

    #[test]
    fn test_threat_clock_advances_power() {
        let mut state = CampaignState::default_test_campaign(42);
        state.world_threat_clock = WorldThreatClock {
            threat_type: WorldThreat::AncientLich,
            power: 0.0,
            growth_rate: 0.01,
            disruptions: 0,
            warnings_issued: 0,
            active: true,
        };

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        state.tick = 100;
        tick_threat_clock(&mut state, &mut deltas, &mut events);
        assert!((state.world_threat_clock.power - 0.01).abs() < 0.001);
        assert!(events.iter().any(|e| matches!(e, WorldEvent::ThreatClockAdvanced { .. })));
    }

    #[test]
    fn test_threshold_crossing_fires_manifestation() {
        let mut state = CampaignState::default_test_campaign(42);
        state.world_threat_clock = WorldThreatClock {
            threat_type: WorldThreat::VoidRift,
            power: 0.19,
            growth_rate: 0.02,
            disruptions: 0,
            warnings_issued: 0,
            active: true,
        };

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        state.tick = 200;
        tick_threat_clock(&mut state, &mut deltas, &mut events);
        assert!(state.world_threat_clock.power >= 0.2);
        assert!(events.iter().any(|e| matches!(e, WorldEvent::ThreatManifested { .. })));
        assert_eq!(state.world_threat_clock.warnings_issued, 1);
    }

    #[test]
    fn test_disruption_reduces_power() {
        let mut state = CampaignState::default_test_campaign(42);
        state.world_threat_clock = WorldThreatClock {
            threat_type: WorldThreat::DemonLord,
            power: 0.5,
            growth_rate: 0.003,
            disruptions: 0,
            warnings_issued: 0,
            active: true,
        };

        let mut events = Vec::new();
        disrupt_threat_clock(&mut state, &mut events);

        assert!(state.world_threat_clock.power < 0.5);
        assert!(state.world_threat_clock.power >= 0.2); // max reduction is 0.3
        assert_eq!(state.world_threat_clock.disruptions, 1);
        assert!(state.world_threat_clock.growth_rate < 0.003);
        assert!(events.iter().any(|e| matches!(e, WorldEvent::ThreatDisrupted { .. })));
    }

    #[test]
    fn test_power_capped_at_one() {
        let mut state = CampaignState::default_test_campaign(42);
        state.world_threat_clock = WorldThreatClock {
            threat_type: WorldThreat::BlightHeart,
            power: 0.99,
            growth_rate: 0.05,
            disruptions: 0,
            warnings_issued: 0,
            active: true,
        };

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        state.tick = 300;
        tick_threat_clock(&mut state, &mut deltas, &mut events);
        assert!((state.world_threat_clock.power - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_growth_rate_accelerates() {
        let mut state = CampaignState::default_test_campaign(42);
        state.world_threat_clock = WorldThreatClock {
            threat_type: WorldThreat::DragonAwakening,
            power: 0.1,
            growth_rate: 0.002,
            disruptions: 0,
            warnings_issued: 0,
            active: true,
        };

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        state.tick = 1000;
        let old_rate = state.world_threat_clock.growth_rate;
        tick_threat_clock(&mut state, &mut deltas, &mut events);
        assert!(state.world_threat_clock.growth_rate > old_rate);
    }

    #[test]
    fn test_no_advance_after_power_one() {
        let mut state = CampaignState::default_test_campaign(42);
        state.world_threat_clock = WorldThreatClock {
            threat_type: WorldThreat::AncientLich,
            power: 1.0,
            growth_rate: 0.01,
            disruptions: 0,
            warnings_issued: 0,
            active: true,
        };

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        state.tick = 400;
        tick_threat_clock(&mut state, &mut deltas, &mut events);
        // No events emitted when already at max
        assert!(events.is_empty());
    }
}
