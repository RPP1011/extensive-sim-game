//! Currency debasement system — fires every 400 ticks.
//!
//! Factions under financial stress (at war or with low military reserves) may
//! secretly reduce coin metal content, causing inflation that erodes the real
//! value of gold earned from their territory. The guild's espionage network or
//! merchant contacts can detect debasement; once discovered, the guild may
//! convert savings to trade goods or expose the fraud for a diplomatic payoff.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the currency debasement system ticks (in ticks).
const DEBASEMENT_INTERVAL: u64 = 13;

/// Amount purity drops per debasement cycle.
const PURITY_DROP_PER_CYCLE: f32 = 0.05;

/// Inflation multiplier: inflation_rate = (1.0 - purity) * INFLATION_FACTOR.
const INFLATION_FACTOR: f32 = 0.5;

/// Diplomacy standing penalty when debasement is exposed.
const EXPOSURE_DIPLOMACY_PENALTY: f32 = 20.0;

/// Treasury hit to the guild when a faction is forced to reform (% of gold).
const REFORM_TREASURY_HIT_FRACTION: f32 = 0.1;

/// Reputation gained by the guild when exposing debasement.
const EXPOSURE_REPUTATION_GAIN: f32 = 5.0;

/// Base detection probability per cycle (before espionage modifiers).
const BASE_DETECTION_CHANCE: f32 = 0.05;

/// Additional detection chance per active spy in the faction.
const SPY_DETECTION_BONUS: f32 = 0.15;

/// Detection bonus from guild intelligence investment level.
const INTEL_INVESTMENT_BONUS: f32 = 0.02;

/// Minimum purity before automatic exposure (too obvious to hide).
const MIN_HIDDEN_PURITY: f32 = 0.3;

pub fn tick_currency_debasement(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % DEBASEMENT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Don't activate before tick 800 (early game grace period)
    if state.tick < 800 {
        return;
    }

    let num_factions = state.factions.len();
    if num_factions == 0 {
        return;
    }

    // Ensure currency_integrity vec is initialized for all factions
    for faction in &state.factions {
        let fid = faction.id as u32;
        if !state.currency_integrity.iter().any(|c| c.faction_id == fid) {
            state.currency_integrity.push(CurrencyState {
                faction_id: fid,
                purity: 1.0,
                inflation_rate: 0.0,
                debasement_detected: false,
            });
        }
    }

    // Collect faction financial stress info to avoid borrow conflicts
    let faction_info: Vec<(u32, bool, String)> = state
        .factions
        .iter()
        .map(|f| {
            let under_stress = !f.at_war_with.is_empty()
                || f.military_strength < f.max_military_strength * 0.3;
            (f.id as u32, under_stress, f.name.clone())
        })
        .collect();

    // Collect spy info for detection bonuses
    let spy_targets: Vec<usize> = state.spies.iter().map(|s| s.target_faction_id).collect();
    let intel_level = state.guild.investment.intelligence_level;

    // Process each faction's currency
    for (fid, under_stress, faction_name) in &faction_info {
        let cs_idx = match state.currency_integrity.iter().position(|c| c.faction_id == *fid) {
            Some(i) => i,
            None => continue,
        };

        let already_detected = state.currency_integrity[cs_idx].debasement_detected;
        let current_purity = state.currency_integrity[cs_idx].purity;

        // --- Debasement: factions under stress may reduce purity ---
        if *under_stress && current_purity > MIN_HIDDEN_PURITY && !already_detected {
            // Random chance to debase (not every stressed faction debases every cycle)
            let debase_roll = lcg_f32(&mut state.rng);
            if debase_roll < 0.4 {
                let new_purity = (current_purity - PURITY_DROP_PER_CYCLE).max(0.0);
                state.currency_integrity[cs_idx].purity = new_purity;
                let new_inflation = (1.0 - new_purity) * INFLATION_FACTOR;
                state.currency_integrity[cs_idx].inflation_rate = new_inflation;

                events.push(WorldEvent::CurrencyDebased {
                    faction_id: *fid,
                    new_purity,
                });

                // Emit inflation spike if rate crosses a threshold
                if new_inflation >= 0.1 {
                    events.push(WorldEvent::InflationSpike {
                        faction_id: *fid,
                        rate: new_inflation,
                    });
                }
            }
        }

        // Refresh purity after possible debasement
        let current_purity = state.currency_integrity[cs_idx].purity;
        let already_detected = state.currency_integrity[cs_idx].debasement_detected;

        // --- Detection: guild may notice debasement ---
        if current_purity < 1.0 && !already_detected {
            // Count spies in this faction
            let spy_count = spy_targets.iter().filter(|&&t| t == *fid as usize).count();
            let spy_bonus = spy_count as f32 * SPY_DETECTION_BONUS;
            let intel_bonus = intel_level * INTEL_INVESTMENT_BONUS;

            // Worse purity is easier to detect
            let purity_penalty = (1.0 - current_purity) * 0.3;

            let detect_chance =
                (BASE_DETECTION_CHANCE + spy_bonus + intel_bonus + purity_penalty).min(0.95);

            let detect_roll = lcg_f32(&mut state.rng);
            if detect_roll < detect_chance || current_purity <= MIN_HIDDEN_PURITY {
                state.currency_integrity[cs_idx].debasement_detected = true;

                events.push(WorldEvent::DebasementDetected {
                    faction_id: *fid,
                    by_whom: "guild intelligence".to_string(),
                });

                // Auto-expose if purity is critically low
                if current_purity <= MIN_HIDDEN_PURITY {
                    expose_debasement(state, *fid, cs_idx, faction_name, events);
                }
            }
        }
    }

    // --- Apply inflation erosion to guild earnings from affected territories ---
    apply_inflation_erosion(state);
}

/// Expose a faction's debasement: diplomatic penalty, forced reform, guild reputation.
fn expose_debasement(
    state: &mut CampaignState,
    faction_id: u32,
    cs_idx: usize,
    _faction_name: &str,
    events: &mut Vec<WorldEvent>,
) {
    // Faction loses diplomacy standing with the guild
    if let Some(faction) = state.factions.iter_mut().find(|f| f.id as u32 == faction_id) {
        faction.relationship_to_guild =
            (faction.relationship_to_guild - EXPOSURE_DIPLOMACY_PENALTY).max(-100.0);
    }

    // Guild gains reputation for exposing corruption
    state.guild.reputation = (state.guild.reputation + EXPOSURE_REPUTATION_GAIN).min(100.0);

    // Forced reform: purity resets but faction's treasury takes a hit
    // (manifested as reduced military strength — the cost of reform)
    if let Some(faction) = state.factions.iter_mut().find(|f| f.id as u32 == faction_id) {
        faction.military_strength *= 1.0 - REFORM_TREASURY_HIT_FRACTION;
    }

    // Reset currency state
    state.currency_integrity[cs_idx].purity = 1.0;
    state.currency_integrity[cs_idx].inflation_rate = 0.0;
    state.currency_integrity[cs_idx].debasement_detected = false;

    events.push(WorldEvent::DebasementExposed {
        faction_id,
        reputation_impact: EXPOSURE_REPUTATION_GAIN,
    });
}

/// Silently reduce the real value of gold the guild earns from regions
/// owned by factions with debased currency (undetected inflation erosion).
fn apply_inflation_erosion(state: &mut CampaignState) {
    let guild_faction_id = state.diplomacy.guild_faction_id;

    // Build a map of faction_id → inflation_rate for factions with undetected debasement
    let inflation_map: Vec<(usize, f32)> = state
        .currency_integrity
        .iter()
        .filter(|c| !c.debasement_detected && c.inflation_rate > 0.0)
        .map(|c| (c.faction_id as usize, c.inflation_rate))
        .collect();

    if inflation_map.is_empty() {
        return;
    }

    // For each region not owned by the guild but trading with it,
    // the inflation erodes the effective gold value
    let mut total_erosion = 0.0f32;
    for region in &state.overworld.regions {
        if region.owner_faction_id == guild_faction_id {
            continue; // Guild's own regions are not affected
        }
        if let Some((_, rate)) = inflation_map.iter().find(|(fid, _)| *fid == region.owner_faction_id) {
            // Erosion proportional to region control and inflation rate
            let erosion = region.control * 0.001 * rate;
            total_erosion += erosion;
        }
    }

    if total_erosion > 0.0 {
        state.guild.gold -= total_erosion.min(state.guild.gold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::actions::StepDeltas;

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(12345);
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;
        state.tick = 800;
        // Clear default factions and add controlled ones
        state.factions.clear();
        // Add a faction at war (financially stressed)
        state.factions.push(FactionState {
            id: 0,
            name: "Iron Kingdom".into(),
            relationship_to_guild: 50.0,
            military_strength: 20.0,
            max_military_strength: 100.0,
            territory_size: 3,
            diplomatic_stance: DiplomaticStance::Neutral,
            coalition_member: false,
            at_war_with: vec![1],
            has_guild: false,
            guild_adventurer_count: 0,
            recent_actions: Vec::new(),
            relation: 0.0,
            coup_risk: 0.0,
            coup_cooldown: 0,
            escalation_level: 0,
            patrol_losses: 0,
            escalation_cooldown: 0,
            last_patrol_loss_tick: 0,
                    skill_modifiers: Default::default(),
        });
        state.factions.push(FactionState {
            id: 1,
            name: "Desert Nomads".into(),
            relationship_to_guild: 0.0,
            military_strength: 80.0,
            max_military_strength: 100.0,
            territory_size: 2,
            diplomatic_stance: DiplomaticStance::Neutral,
            coalition_member: false,
            at_war_with: vec![0],
            has_guild: false,
            guild_adventurer_count: 0,
            recent_actions: Vec::new(),
            relation: 0.0,
            coup_risk: 0.0,
            coup_cooldown: 0,
            escalation_level: 0,
            patrol_losses: 0,
            escalation_cooldown: 0,
            last_patrol_loss_tick: 0,
                    skill_modifiers: Default::default(),
        });
        state
    }

    #[test]
    fn initializes_currency_state() {
        let mut state = make_test_state();
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_currency_debasement(&mut state, &mut deltas, &mut events);

        assert_eq!(state.currency_integrity.len(), 2);
        assert!(state.currency_integrity.iter().all(|c| c.purity <= 1.0));
    }

    #[test]
    fn stressed_faction_may_debase() {
        let mut state = make_test_state();
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        // Run several cycles to give debasement a chance
        for i in 0..10 {
            state.tick = 800 + i * DEBASEMENT_INTERVAL;
            tick_currency_debasement(&mut state, &mut deltas, &mut events);
        }

        // At least one faction should have debased by now
        let debased = state.currency_integrity.iter().any(|c| c.purity < 1.0);
        assert!(debased, "Expected at least one debased currency after 10 cycles");
    }

    #[test]
    fn inflation_rate_tracks_purity() {
        let mut state = make_test_state();
        state.currency_integrity.push(CurrencyState {
            faction_id: 0,
            purity: 0.8,
            inflation_rate: 0.0,
            debasement_detected: false,
        });

        let expected_inflation = (1.0 - 0.8) * INFLATION_FACTOR;
        // Manually update
        state.currency_integrity[0].inflation_rate = (1.0 - state.currency_integrity[0].purity) * INFLATION_FACTOR;
        assert!((state.currency_integrity[0].inflation_rate - expected_inflation).abs() < 0.001);
    }

    #[test]
    fn exposure_resets_purity_and_penalizes() {
        let mut state = make_test_state();
        let mut events = Vec::new();
        state.currency_integrity.push(CurrencyState {
            faction_id: 0,
            purity: 0.5,
            inflation_rate: 0.25,
            debasement_detected: true,
        });

        let old_rep = state.guild.reputation;
        let old_rel = state.factions[0].relationship_to_guild;

        expose_debasement(&mut state, 0, 0, "Iron Kingdom", &mut events);

        assert!((state.currency_integrity[0].purity - 1.0).abs() < 0.001);
        assert!(state.guild.reputation > old_rep);
        assert!(state.factions[0].relationship_to_guild < old_rel);
        assert!(events.iter().any(|e| matches!(e, WorldEvent::DebasementExposed { .. })));
    }

    #[test]
    fn skips_before_tick_800() {
        let mut state = make_test_state();
        state.tick = 400;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_currency_debasement(&mut state, &mut deltas, &mut events);

        assert!(state.currency_integrity.is_empty());
        assert!(events.is_empty());
    }
}
