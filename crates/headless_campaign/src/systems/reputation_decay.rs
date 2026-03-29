//! Reputation decay and maintenance system.
//!
//! Reputation is not permanent — it requires active maintenance through good
//! deeds, or it fades from public memory. Every 200 ticks the guild's
//! reputation decays by a base amount, accelerated or reduced by various
//! campaign conditions.
//!
//! Decay accelerators:
//! - No quest completed recently (guild seen as inactive)
//! - High corruption (scandal)
//! - Recent battle losses (guild looks weak)
//! - Negative rumors spreading
//!
//! Decay reducers:
//! - Recent quest completions
//! - Active propaganda campaigns
//! - War memorial festivals
//! - High guild tier (established reputation)
//!
//! Floor: reputation won't decay below 10 (baseline recognition).

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Cadence: reputation decay applies every 200 ticks.
const TICK_CADENCE: u64 = 7;

/// Base reputation decay per tick.
const BASE_DECAY: f32 = 0.1;

/// Reputation floor — won't decay below this.
const REPUTATION_FLOOR: f32 = 10.0;

/// Ticks without a quest completion before decay doubles.
const INACTIVITY_THRESHOLD: u64 = 67;

/// Corruption threshold above which scandal penalty applies.
const CORRUPTION_THRESHOLD: f32 = 50.0;

/// How many recent ticks to look back for battle losses.
const RECENT_LOSS_WINDOW: u64 = 33;

/// How many recent ticks to look back for quest completions (for reducer).
const RECENT_QUEST_WINDOW: u64 = 17;

/// Reputation trajectory thresholds.
const TRAJECTORY_RISING_THRESHOLD: f32 = 0.5;
const TRAJECTORY_FALLING_THRESHOLD: f32 = -0.5;

/// Run the reputation decay system for one tick.
///
/// - Calculates decay accelerators and reducers.
/// - Applies net decay (clamped to floor).
/// - Records reputation deeds and trajectory changes.
/// - Emits `WorldEvent` variants for decay, maintenance, and trajectory shifts.
pub fn tick_reputation_decay(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    let current_rep = state.guild.reputation;
    if current_rep <= REPUTATION_FLOOR {
        return;
    }

    // --- Calculate base decay ---
    let mut decay = BASE_DECAY;
    let mut reasons: Vec<String> = Vec::new();

    // --- Decay accelerators ---

    // 1. No quest completed in INACTIVITY_THRESHOLD ticks → decay doubles.
    let last_quest_tick = state
        .completed_quests
        .iter()
        .map(|q| q.completed_at_ms / (CAMPAIGN_TURN_SECS as u64 * 1000))
        .max()
        .unwrap_or(0);
    let ticks_since_quest = state.tick.saturating_sub(last_quest_tick);
    if ticks_since_quest > INACTIVITY_THRESHOLD {
        decay += BASE_DECAY; // doubles
        reasons.push("guild inactive".into());
    }

    // 2. Corruption > 50 (check crisis state for corruption).
    let has_corruption = state.overworld.active_crises.iter().any(|c| {
        matches!(c, ActiveCrisis::Corruption { .. })
    });
    // Check corrupted region count as a proxy for corruption severity.
    let corruption_severity: f32 = state
        .overworld
        .active_crises
        .iter()
        .filter_map(|c| match c {
            ActiveCrisis::Corruption {
                corrupted_regions, ..
            } => Some(corrupted_regions.len() as f32 * 20.0),
            _ => None,
        })
        .sum();
    if has_corruption && corruption_severity > CORRUPTION_THRESHOLD {
        decay += 0.3;
        reasons.push("corruption scandal".into());
    }

    // 3. Recent battle losses.
    let recent_losses = state
        .completed_quests
        .iter()
        .filter(|q| {
            let q_tick = q.completed_at_ms / (CAMPAIGN_TURN_SECS as u64 * 1000);
            state.tick.saturating_sub(q_tick) < RECENT_LOSS_WINDOW
                && q.result == QuestResult::Defeat
        })
        .count() as f32;
    if recent_losses > 0.0 {
        decay += 0.2 * recent_losses;
        reasons.push(format!("{} recent losses", recent_losses as u32));
    }

    // 4. Negative rumors spreading.
    let negative_rumor_count = state
        .rumors
        .iter()
        .filter(|r| matches!(r.rumor_type, RumorType::AmbushThreat | RumorType::CrisisWarning))
        .count() as f32;
    if negative_rumor_count > 0.0 {
        decay += 0.1 * negative_rumor_count;
        reasons.push(format!("{} negative stories", negative_rumor_count as u32));
    }

    // --- Decay reducers ---
    let mut reduction_mult = 1.0_f32;

    // 1. Recent quest completions → -50% decay.
    let recent_completions = state
        .completed_quests
        .iter()
        .filter(|q| {
            let q_tick = q.completed_at_ms / (CAMPAIGN_TURN_SECS as u64 * 1000);
            state.tick.saturating_sub(q_tick) < RECENT_QUEST_WINDOW
                && q.result == QuestResult::Victory
        })
        .count();
    if recent_completions > 0 {
        reduction_mult *= 0.5;
    }

    // 2. Active propaganda campaigns → -30% decay.
    let has_propaganda = state
        .propaganda_campaigns
        .iter()
        .any(|c| c.campaign_type == PropagandaType::BoostReputation);
    if has_propaganda {
        reduction_mult *= 0.7;
    }

    // 3. War memorial festivals → -10% decay each.
    let memorial_count = state
        .active_festivals
        .iter()
        .filter(|f| f.festival_type == FestivalType::WarMemorial)
        .count();
    for _ in 0..memorial_count {
        reduction_mult *= 0.9;
    }

    // 4. High guild tier → -20% decay.
    let tier_bonus = match state.guild.guild_tier {
        GuildTier::Silver => 0.9,
        GuildTier::Gold => 0.8,
        GuildTier::Legendary => 0.7,
        GuildTier::Bronze => 1.0,
    };
    reduction_mult *= tier_bonus;

    // --- Apply net decay ---
    let net_decay = decay * reduction_mult;
    let new_rep = (current_rep - net_decay).max(REPUTATION_FLOOR);
    let actual_decay = current_rep - new_rep;

    if actual_decay > 0.001 {
        state.guild.reputation = new_rep;

        // Record the deed.
        let reason = if reasons.is_empty() {
            "people forget".into()
        } else {
            reasons.join(", ")
        };

        state.reputation_history.recent_deeds.push(ReputationDeed {
            tick: state.tick,
            amount: -actual_decay,
            source: reason.clone(),
        });

        // Trim deed history to last 50 entries.
        if state.reputation_history.recent_deeds.len() > 50 {
            let drain_count = state.reputation_history.recent_deeds.len() - 50;
            state.reputation_history.recent_deeds.drain(..drain_count);
        }

        events.push(WorldEvent::ReputationDecayed {
            amount: actual_decay,
            reason,
        });
    }

    // --- Maintenance: active propaganda costs gold ---
    if has_propaganda {
        let maintenance_cost = state.reputation_history.maintenance_cost;
        if state.guild.gold >= maintenance_cost && maintenance_cost > 0.0 {
            state.guild.gold -= maintenance_cost;
            events.push(WorldEvent::ReputationMaintained {
                cost: maintenance_cost,
            });
        }
    }

    // --- Track trajectory ---
    // Calculate net reputation change over the last 5 decay cycles (1000 ticks).
    let recent_net: f32 = state
        .reputation_history
        .recent_deeds
        .iter()
        .filter(|d| state.tick.saturating_sub(d.tick) < 1000)
        .map(|d| d.amount)
        .sum();

    let new_trend = if recent_net > TRAJECTORY_RISING_THRESHOLD {
        ReputationTrend::Rising
    } else if recent_net < TRAJECTORY_FALLING_THRESHOLD {
        ReputationTrend::Falling
    } else {
        ReputationTrend::Stable
    };

    let old_trend = state.reputation_history.trend;
    if new_trend != old_trend {
        state.reputation_history.trend = new_trend;
        events.push(WorldEvent::ReputationTrajectoryChanged { trend: new_trend });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::StepDeltas;

    #[test]
    fn base_decay_reduces_reputation() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        state.guild.reputation = 50.0;
        state.tick = TICK_CADENCE - 1; // will be at cadence after +1
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        // Set tick to cadence boundary.
        state.tick = TICK_CADENCE;
        tick_reputation_decay(&mut state, &mut deltas, &mut events);

        assert!(state.guild.reputation < 50.0, "reputation should have decayed");
        assert!(!events.is_empty(), "should emit decay event");
    }

    #[test]
    fn reputation_floor_at_10() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        state.guild.reputation = 10.0;
        state.tick = TICK_CADENCE;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_reputation_decay(&mut state, &mut deltas, &mut events);

        assert_eq!(state.guild.reputation, 10.0, "should not decay below floor");
        assert!(events.is_empty(), "should not emit event at floor");
    }

    #[test]
    fn inactivity_doubles_decay() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        state.guild.reputation = 50.0;
        state.tick = TICK_CADENCE;

        // First: decay with activity (add a recent quest completion).
        state.completed_quests.push(CompletedQuest {
            id: 1,
            quest_type: QuestType::Escort,
            result: QuestResult::Victory,
            reward_applied: QuestReward { gold: 10.0, reputation: 0.0, ..Default::default() },
            completed_at_ms: (state.tick - 10) * CAMPAIGN_TURN_SECS as u64 * 1000,
            party_id: 1,
            casualties: 0,
                threat_level: 0.0,
        });
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_reputation_decay(&mut state, &mut deltas, &mut events);
        let rep_with_activity = state.guild.reputation;

        // Reset and test without activity.
        state.guild.reputation = 50.0;
        state.completed_quests.clear();
        state.tick = INACTIVITY_THRESHOLD + TICK_CADENCE + 1;
        // Make sure tick is on cadence.
        state.tick = ((state.tick / TICK_CADENCE) + 1) * TICK_CADENCE;
        let mut events2 = Vec::new();
        tick_reputation_decay(&mut state, &mut deltas, &mut events2);
        let rep_without_activity = state.guild.reputation;

        // Inactivity should cause more decay.
        assert!(
            rep_without_activity < rep_with_activity,
            "inactivity should cause more decay: {} vs {}",
            rep_without_activity,
            rep_with_activity
        );
    }

    #[test]
    fn guild_tier_reduces_decay() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        state.guild.reputation = 80.0;
        state.tick = TICK_CADENCE;

        // Bronze tier (no reduction).
        state.guild.guild_tier = GuildTier::Bronze;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_reputation_decay(&mut state, &mut deltas, &mut events);
        let bronze_rep = state.guild.reputation;

        // Legendary tier (30% reduction).
        state.guild.reputation = 80.0;
        state.guild.guild_tier = GuildTier::Legendary;
        state.reputation_history.recent_deeds.clear();
        let mut events2 = Vec::new();
        tick_reputation_decay(&mut state, &mut deltas, &mut events2);
        let legendary_rep = state.guild.reputation;

        assert!(
            legendary_rep > bronze_rep,
            "legendary guild should decay less: {} vs {}",
            legendary_rep,
            bronze_rep
        );
    }

    #[test]
    fn trajectory_changes_emitted() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        state.guild.reputation = 50.0;
        state.tick = TICK_CADENCE;
        state.reputation_history.trend = ReputationTrend::Stable;

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_reputation_decay(&mut state, &mut deltas, &mut events);

        // After decay, trend should shift to Falling.
        let has_trajectory_event = events.iter().any(|e| {
            matches!(e, WorldEvent::ReputationTrajectoryChanged { .. })
        });
        assert!(has_trajectory_event, "should emit trajectory change event");
    }
}
