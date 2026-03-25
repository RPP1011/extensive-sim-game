//! Divine favor economy system — fires every 200 ticks.
//!
//! Each deity (religious order) maintains accumulated devotion that can be
//! spent on miracles. Favor grows from temples, devotees, sacrifices, and
//! shrine visits. It decays naturally and is consumed by miracles.
//!
//! - When favor > 30: deity may grant a random blessing (heal, crop boost, smite)
//! - When favor < -10: divine displeasure (curse, bad omens reducing morale)
//! - Guild can invest gold via temple donations (tracked as sacrifice events)

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the divine favor system ticks (in ticks).
const DIVINE_FAVOR_INTERVAL: u64 = 7;

/// Favor gained per temple of this order.
const FAVOR_PER_TEMPLE: f32 = 0.5;

/// Favor gained per active devotee (adventurer in party near a temple).
const FAVOR_PER_DEVOTEE: f32 = 0.1;

/// Favor gained per recent sacrifice event.
const FAVOR_PER_SACRIFICE: f32 = 2.0;

/// Favor gained per shrine visit.
const FAVOR_PER_SHRINE_VISIT: f32 = 0.3;

/// Natural favor decay per tick of this system.
const FAVOR_DECAY: f32 = 0.1;

/// Favor threshold to potentially grant a miracle.
const MIRACLE_THRESHOLD: f32 = 30.0;

/// Favor threshold below which displeasure events fire.
const DISPLEASURE_THRESHOLD: f32 = -10.0;

/// Minimum miracle cost.
const MIRACLE_COST_MIN: f32 = 10.0;

/// Maximum miracle cost.
const MIRACLE_COST_MAX: f32 = 50.0;

/// Chance per tick (when above threshold) that a miracle fires.
const MIRACLE_CHANCE: f32 = 0.25;

/// Chance per tick (when below threshold) that displeasure fires.
const DISPLEASURE_CHANCE: f32 = 0.20;

/// Tick the divine favor economy: accumulate favor, decay, miracles, displeasure.
pub fn tick_divine_favor(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % DIVINE_FAVOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Ensure we have a divine favor entry for every order that has temples
    ensure_favor_entries(state);

    let num_entries = state.divine_favor.len();
    for i in 0..num_entries {
        let order = state.divine_favor[i].order;

        // --- Accumulate favor sources ---
        let temple_count = state
            .temples
            .iter()
            .filter(|t| t.order == order)
            .count() as f32;

        let active_devotees = count_active_devotees(state, order);
        let recent_sacrifices = state.divine_favor[i].recent_sacrifices as f32;
        let shrine_visits = state.divine_favor[i].shrine_visits as f32;

        let gain = temple_count * FAVOR_PER_TEMPLE
            + active_devotees * FAVOR_PER_DEVOTEE
            + recent_sacrifices * FAVOR_PER_SACRIFICE
            + shrine_visits * FAVOR_PER_SHRINE_VISIT;

        // --- Apply gain and decay ---
        let old_favor = state.divine_favor[i].divine_favor;
        state.divine_favor[i].divine_favor += gain - FAVOR_DECAY;

        // Reset per-tick accumulators
        state.divine_favor[i].recent_sacrifices = 0;
        state.divine_favor[i].shrine_visits = 0;

        let new_favor = state.divine_favor[i].divine_favor;

        // Emit change event if significant
        let delta = (new_favor - old_favor).abs();
        if delta > 0.01 {
            let reason = if gain > FAVOR_DECAY {
                format!(
                    "temples:{:.0} devotees:{:.0} sacrifices:{:.0} shrines:{:.0}",
                    temple_count, active_devotees, recent_sacrifices, shrine_visits
                )
            } else {
                "natural decay".to_string()
            };
            events.push(WorldEvent::DivineFavorChanged {
                order_idx: i,
                new_favor,
                reason,
            });
        }

        // --- Miracle check (favor > 30) ---
        if new_favor > MIRACLE_THRESHOLD {
            let roll = lcg_f32(&mut state.rng);
            if roll < MIRACLE_CHANCE {
                let miracle_cost =
                    MIRACLE_COST_MIN + lcg_f32(&mut state.rng) * (MIRACLE_COST_MAX - MIRACLE_COST_MIN);
                state.divine_favor[i].divine_favor -= miracle_cost;
                state.divine_favor[i].miracles_granted += 1;

                let (miracle_type, beneficiary) =
                    apply_miracle(state, order, &mut state.rng.clone());
                // Re-read rng advancement (we cloned to avoid borrow issues)
                // Advance rng by the same amount
                let _ = lcg_f32(&mut state.rng);

                events.push(WorldEvent::MiracleGranted {
                    order_idx: i,
                    miracle_type,
                    beneficiary,
                });
            }
        }

        // --- Displeasure check (favor < -10) ---
        if new_favor < DISPLEASURE_THRESHOLD {
            let roll = lcg_f32(&mut state.rng);
            if roll < DISPLEASURE_CHANCE {
                state.divine_favor[i].displeasure_events += 1;

                let effect = apply_displeasure(state, order);

                events.push(WorldEvent::DivineDispleasure {
                    order_idx: i,
                    effect,
                });
            }
        }
    }
}

/// Ensure every order with at least one temple has a divine favor entry.
fn ensure_favor_entries(state: &mut CampaignState) {
    for temple in &state.temples {
        let order = temple.order;
        if !state.divine_favor.iter().any(|e| e.order == order) {
            state.divine_favor.push(DivineFavorEntry::new(order));
        }
    }
}

/// Count adventurers that are active (not dead, not idle) as devotees for an order.
fn count_active_devotees(state: &CampaignState, order: ReligiousOrder) -> f32 {
    // Count adventurers in parties that are near temples of this order
    let temple_regions: Vec<usize> = state
        .temples
        .iter()
        .filter(|t| t.order == order)
        .map(|t| t.region_id)
        .collect();

    if temple_regions.is_empty() {
        return 0.0;
    }

    let mut count = 0u32;
    for adv in &state.adventurers {
        if adv.status != AdventurerStatus::Dead && adv.status != AdventurerStatus::Idle {
            count += 1;
        }
    }
    // Scale by fraction of regions with temples
    let region_coverage = temple_regions.len().min(state.overworld.regions.len()) as f32
        / state.overworld.regions.len().max(1) as f32;
    count as f32 * region_coverage
}

/// Apply a miracle effect and return (miracle_type, beneficiary).
fn apply_miracle(
    state: &mut CampaignState,
    order: ReligiousOrder,
    rng: &mut u64,
) -> (String, String) {
    let roll = lcg_f32(rng);
    match order {
        ReligiousOrder::OrderOfLight => {
            if roll < 0.33 {
                // Heal an injured adventurer
                if let Some(adv) = state
                    .adventurers
                    .iter_mut()
                    .find(|a| a.status != AdventurerStatus::Dead && a.injury > 10.0)
                {
                    let name = adv.name.clone();
                    adv.injury = (adv.injury - 20.0).max(0.0);
                    ("divine_healing".to_string(), name)
                } else {
                    ("blessing_of_light".to_string(), "guild".to_string())
                }
            } else if roll < 0.66 {
                // Boost morale
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead {
                        adv.morale = (adv.morale + 10.0).min(100.0);
                    }
                }
                ("morale_inspiration".to_string(), "all_adventurers".to_string())
            } else {
                ("crop_blessing".to_string(), "guild".to_string())
            }
        }
        ReligiousOrder::BrotherhoodOfSteel => {
            ("forge_blessing".to_string(), "guild_armory".to_string())
        }
        ReligiousOrder::CircleOfNature => {
            // Heal injuries across the board
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead && adv.injury > 0.0 {
                    adv.injury = (adv.injury - 5.0).max(0.0);
                }
            }
            ("nature_restoration".to_string(), "all_adventurers".to_string())
        }
        ReligiousOrder::ShadowCovenant => {
            ("shadow_veil".to_string(), "guild_spies".to_string())
        }
        ReligiousOrder::ScholarsGuild => {
            // Grant XP
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead {
                    adv.xp += 10;
                }
            }
            ("divine_knowledge".to_string(), "all_adventurers".to_string())
        }
    }
}

/// Apply a displeasure effect and return a description.
fn apply_displeasure(state: &mut CampaignState, _order: ReligiousOrder) -> String {
    // Reduce morale across all adventurers (bad omens)
    let mut affected = 0u32;
    for adv in &mut state.adventurers {
        if adv.status != AdventurerStatus::Dead {
            adv.morale = (adv.morale - 5.0).max(0.0);
            affected += 1;
        }
    }
    format!("bad_omens: {} adventurers lost morale", affected)
}

/// Record a temple donation (guild invests gold to boost favor).
pub fn donate_to_order(state: &mut CampaignState, order: ReligiousOrder, gold: f32) -> bool {
    if state.guild.gold < gold {
        return false;
    }
    state.guild.gold -= gold;

    // Find or create favor entry
    if let Some(entry) = state.divine_favor.iter_mut().find(|e| e.order == order) {
        entry.recent_sacrifices += (gold / 10.0) as u32;
    } else {
        let mut entry = DivineFavorEntry::new(order);
        entry.recent_sacrifices = (gold / 10.0) as u32;
        state.divine_favor.push(entry);
    }
    true
}
