//! Insurance contracts system — fires every 200 ticks (~20s game time).
//!
//! Guilds pay premiums to protect against caravan raids, quest failures,
//! adventurer death, natural disasters, and cargo losses. Claims are
//! auto-processed when matching events are detected.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often to process insurance (in ticks).
const INSURANCE_INTERVAL: u64 = 200;

/// Base premium per tick for each insurance type.
const BASE_PREMIUM_CARAVAN: f32 = 0.8;
const BASE_PREMIUM_QUEST_FAILURE: f32 = 0.5;
const BASE_PREMIUM_LIFE: f32 = 1.0;
const BASE_PREMIUM_PROPERTY: f32 = 0.6;
const BASE_PREMIUM_CARGO: f32 = 0.7;

/// Default coverage ratio (fraction of loss covered).
const DEFAULT_COVERAGE: f32 = 0.75;

/// Default max claims per policy before renewal is needed.
const DEFAULT_MAX_CLAIMS: u32 = 3;

/// Policy duration in ticks (5000 ticks = ~8 min game time).
const POLICY_DURATION_TICKS: u64 = 5000;

/// Base cost to purchase a new policy.
pub const INSURANCE_PURCHASE_COST: f32 = 25.0;

/// Process insurance premiums and claims every `INSURANCE_INTERVAL` ticks.
pub fn tick_insurance(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % INSURANCE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Drain premiums ---
    let mut lapsed_ids = Vec::new();
    for policy in &state.insurance_policies {
        if policy.claims_made >= policy.max_claims {
            continue; // Exhausted, skip premium
        }
        if state.tick >= policy.expires_tick {
            continue; // Expired
        }
        let premium = policy.premium_per_tick * INSURANCE_INTERVAL as f32;
        if state.guild.gold < premium {
            lapsed_ids.push(policy.id);
        }
    }

    // Deduct premiums for non-lapsed policies
    let mut policies_to_keep = Vec::new();
    for policy in std::mem::take(&mut state.insurance_policies) {
        if lapsed_ids.contains(&policy.id) {
            events.push(WorldEvent::InsuranceLapsed {
                policy_id: policy.id,
                policy_type: format!("{:?}", policy.policy_type),
            });
            continue;
        }
        if policy.claims_made >= policy.max_claims || state.tick >= policy.expires_tick {
            // Expired or exhausted — silently drop
            continue;
        }
        let premium = policy.premium_per_tick * INSURANCE_INTERVAL as f32;
        state.guild.gold = (state.guild.gold - premium).max(0.0);
        policies_to_keep.push(policy);
    }
    state.insurance_policies = policies_to_keep;

    // --- Process claims against recent events ---
    // Scan completed quests for failures since last insurance tick
    process_claims(state, events);
}

/// Scan game state for claimable events and auto-pay claims.
fn process_claims(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let lookback_start = state.tick.saturating_sub(INSURANCE_INTERVAL);

    // Collect claim payouts to apply (avoid double-borrow)
    let mut payouts: Vec<(u32, f32, String)> = Vec::new(); // (policy_id, payout, reason)

    // --- QuestFailure claims ---
    // Check completed quests that failed within the lookback window
    let failed_quest_reward: f32 = state
        .completed_quests
        .iter()
        .filter(|q| {
            q.result == QuestResult::Defeat
                && q.completed_at_ms / CAMPAIGN_TICK_MS as u64 > lookback_start
        })
        .map(|q| q.reward_applied.gold + q.reward_applied.reputation * 0.5)
        .sum();

    if failed_quest_reward > 0.0 {
        for policy in &state.insurance_policies {
            if policy.policy_type == InsuranceType::QuestFailure
                && policy.claims_made < policy.max_claims
            {
                let payout = policy.coverage * failed_quest_reward;
                payouts.push((policy.id, payout, "Quest failure compensation".into()));
                break; // One claim per interval per type
            }
        }
    }

    // --- AdventurerLife claims ---
    // Check for dead adventurers (status == Dead, scan all)
    let dead_adventurers: Vec<(u32, u32)> = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Dead)
        .map(|a| (a.id, a.level))
        .collect();

    // Only claim for deaths we haven't claimed before (track via claims_made incrementing)
    let total_deaths = dead_adventurers.len() as u32;
    for policy in &state.insurance_policies {
        if policy.policy_type == InsuranceType::AdventurerLife
            && policy.claims_made < policy.max_claims
            && policy.claims_made < total_deaths
        {
            // Payout for the most recently dead adventurer
            if let Some(&(_, level)) = dead_adventurers.get(policy.claims_made as usize) {
                let payout = policy.coverage * (level as f32 * 10.0);
                payouts.push((policy.id, payout, "Adventurer life insurance payout".into()));
            }
            break;
        }
    }

    // --- CaravanProtection claims ---
    // Check for escort quests that failed (proxy for caravan raids)
    let raided_value: f32 = state
        .completed_quests
        .iter()
        .filter(|q| {
            q.quest_type == QuestType::Escort
                && q.result == QuestResult::Defeat
                && q.completed_at_ms / CAMPAIGN_TICK_MS as u64 > lookback_start
        })
        .map(|q| q.reward_applied.gold.max(50.0)) // Minimum cargo value of 50
        .sum();

    if raided_value > 0.0 {
        for policy in &state.insurance_policies {
            if policy.policy_type == InsuranceType::CaravanProtection
                && policy.claims_made < policy.max_claims
            {
                let payout = policy.coverage * raided_value;
                payouts.push((policy.id, payout, "Caravan raid compensation".into()));
                break;
            }
        }
    }

    // --- PropertyDamage claims ---
    // Check for building damage from random events (tracked via event_log)
    let property_damage: f32 = state
        .event_log
        .iter()
        .filter(|e| {
            e.tick > lookback_start
                && matches!(
                    e.description.as_str(),
                    d if d.contains("earthquake") || d.contains("fire") || d.contains("storm") || d.contains("flood")
                )
        })
        .count() as f32
        * 30.0; // Each disaster ~30g repair cost

    if property_damage > 0.0 {
        for policy in &state.insurance_policies {
            if policy.policy_type == InsuranceType::PropertyDamage
                && policy.claims_made < policy.max_claims
            {
                let payout = policy.coverage * property_damage;
                payouts.push((policy.id, payout, "Property damage compensation".into()));
                break;
            }
        }
    }

    // --- CargoInsurance claims ---
    // Check for gather/escort quests that failed (lost trade goods)
    let cargo_loss: f32 = state
        .completed_quests
        .iter()
        .filter(|q| {
            (q.quest_type == QuestType::Gather || q.quest_type == QuestType::Escort)
                && q.result == QuestResult::Defeat
                && q.completed_at_ms / CAMPAIGN_TICK_MS as u64 > lookback_start
        })
        .map(|q| q.reward_applied.gold.max(30.0))
        .sum();

    if cargo_loss > 0.0 {
        for policy in &state.insurance_policies {
            if policy.policy_type == InsuranceType::CargoInsurance
                && policy.claims_made < policy.max_claims
            {
                let payout = policy.coverage * cargo_loss;
                payouts.push((policy.id, payout, "Cargo loss compensation".into()));
                break;
            }
        }
    }

    // --- Apply payouts ---
    for (policy_id, payout, reason) in payouts {
        state.guild.gold += payout;
        if let Some(policy) = state
            .insurance_policies
            .iter_mut()
            .find(|p| p.id == policy_id)
        {
            policy.claims_made += 1;
        }
        events.push(WorldEvent::InsuranceClaimed {
            policy_id,
            payout,
            reason,
        });
    }
}

/// Calculate the premium for a given insurance type based on current risk.
pub fn calculate_premium(state: &CampaignState, policy_type: &InsuranceType) -> f32 {
    let base = match policy_type {
        InsuranceType::CaravanProtection => {
            // More active escort quests = higher premium
            let escort_count = state
                .active_quests
                .iter()
                .filter(|q| q.request.quest_type == QuestType::Escort)
                .count() as f32;
            BASE_PREMIUM_CARAVAN * (1.0 + escort_count * 0.3)
        }
        InsuranceType::QuestFailure => {
            // Higher average threat = higher premium
            let avg_threat = if state.active_quests.is_empty() {
                0.5
            } else {
                state
                    .active_quests
                    .iter()
                    .map(|q| q.request.threat_level)
                    .sum::<f32>()
                    / state.active_quests.len() as f32
                    / 100.0
            };
            BASE_PREMIUM_QUEST_FAILURE * (1.0 + avg_threat)
        }
        InsuranceType::AdventurerLife => {
            // More adventurers = higher premium
            let alive = state
                .adventurers
                .iter()
                .filter(|a| a.status != AdventurerStatus::Dead)
                .count() as f32;
            BASE_PREMIUM_LIFE * (1.0 + alive * 0.15)
        }
        InsuranceType::PropertyDamage => {
            // Higher global threat = higher premium
            let threat_factor = state.overworld.global_threat_level / 100.0;
            BASE_PREMIUM_PROPERTY * (1.0 + threat_factor)
        }
        InsuranceType::CargoInsurance => {
            // More gather/escort quests = higher premium
            let cargo_count = state
                .active_quests
                .iter()
                .filter(|q| {
                    matches!(
                        q.request.quest_type,
                        QuestType::Gather | QuestType::Escort
                    )
                })
                .count() as f32;
            BASE_PREMIUM_CARGO * (1.0 + cargo_count * 0.25)
        }
    };

    // Scale by global threat multiplier
    let threat_mult = 1.0 + state.overworld.global_threat_level / 200.0;
    base * threat_mult
}

/// Purchase a new insurance policy.
pub fn purchase_insurance(
    state: &mut CampaignState,
    policy_type: InsuranceType,
    events: &mut Vec<WorldEvent>,
) -> Result<u32, String> {
    // Check if already have this type
    let already_has = state
        .insurance_policies
        .iter()
        .any(|p| p.policy_type == policy_type && p.claims_made < p.max_claims);
    if already_has {
        return Err(format!(
            "Already have active {:?} insurance",
            policy_type
        ));
    }

    if state.guild.gold < INSURANCE_PURCHASE_COST {
        return Err("Not enough gold to purchase insurance".into());
    }

    let premium = calculate_premium(state, &policy_type);
    let policy_id = state.next_insurance_id;
    state.next_insurance_id += 1;

    state.guild.gold = (state.guild.gold - INSURANCE_PURCHASE_COST).max(0.0);

    let policy = InsurancePolicy {
        id: policy_id,
        policy_type: policy_type.clone(),
        premium_per_tick: premium,
        coverage: DEFAULT_COVERAGE,
        started_tick: state.tick,
        expires_tick: state.tick + POLICY_DURATION_TICKS,
        claims_made: 0,
        max_claims: DEFAULT_MAX_CLAIMS,
    };

    state.insurance_policies.push(policy);

    events.push(WorldEvent::InsurancePurchased {
        policy_id,
        policy_type: format!("{:?}", policy_type),
        premium_per_tick: premium,
    });

    Ok(policy_id)
}

/// Cancel an active insurance policy. No refund.
pub fn cancel_insurance(
    state: &mut CampaignState,
    policy_id: u32,
    events: &mut Vec<WorldEvent>,
) -> Result<(), String> {
    let idx = state
        .insurance_policies
        .iter()
        .position(|p| p.id == policy_id);
    match idx {
        Some(i) => {
            state.insurance_policies.remove(i);
            events.push(WorldEvent::InsuranceCanceled { policy_id });
            Ok(())
        }
        None => Err(format!("Insurance policy {} not found", policy_id)),
    }
}
