//! Heroic last stand system — checked during battle resolution.
//!
//! When a party is about to be wiped (battle trending toward Defeat),
//! eligible adventurers may trigger a dramatic turnaround based on their
//! traits, loyalty, bonds, and history. Outcomes are weighted by adventurer
//! level and traits. All randomness flows through the campaign LCG.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;
use crate::systems::bonds;

/// Check whether any party member qualifies for a last stand.
///
/// Called from `tick_battles` when a battle's predicted outcome is Defeat
/// (party_health_ratio approaching zero). Returns the index of the best
/// candidate adventurer, if any.
fn find_last_stand_candidate(
    state: &CampaignState,
    battle: &BattleState,
) -> Option<u32> {
    let party = state.parties.iter().find(|p| p.id == battle.party_id)?;
    let member_ids = &party.member_ids;

    // Collect (adventurer_id, score) for each eligible member
    let mut best: Option<(u32, f32)> = None;

    for &mid in member_ids {
        let adv = state.adventurers.iter().find(|a| a.id == mid)?;
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        let mut score: f32 = 0.0;

        // Loyalty > 80: fight for their brothers
        if adv.loyalty > 80.0 {
            score += 1.0;
        }

        // History tag "near_death" > 2: been through this before
        if adv.history_tags.get("near_death").copied().unwrap_or(0) > 2 {
            score += 1.0;
        }

        // Trait "The Undying": never give up
        if adv.traits.iter().any(|t| t == "The Undying") {
            score += 1.0;
        }

        // Bond > 80 with a dying party member (rage)
        let has_strong_bond = member_ids.iter().any(|&other_id| {
            if other_id == mid {
                return false;
            }
            bonds::bond_strength(&state.adventurer_bonds, mid, other_id) > 80.0
        });
        if has_strong_bond {
            score += 1.0;
        }

        // Must meet at least one trigger condition
        if score > 0.0 {
            // Weight by level for tiebreaking
            score += adv.level as f32 * 0.01;
            match &best {
                Some((_, best_score)) if *best_score >= score => {}
                _ => best = Some((mid, score)),
            }
        }
    }

    best.map(|(id, _)| id)
}

/// Roll the last stand outcome using deterministic RNG.
///
/// Outcome weights:
///   HeroicVictory: 20%  — party wins against odds
///   GloriousDefeat: 50% — party loses but inflicts heavy damage, one survives
///   MiraculousEscape: 30% — party escapes, no loot but no deaths
fn roll_outcome(rng: &mut u64, adv_level: u32) -> LastStandOutcome {
    let roll = lcg_f32(rng);

    // Higher-level adventurers shift odds slightly toward victory
    let level_bonus = (adv_level as f32 * 0.005).min(0.10);

    let victory_threshold = 0.20 + level_bonus;
    let defeat_threshold = victory_threshold + 0.50;

    if roll < victory_threshold {
        LastStandOutcome::HeroicVictory
    } else if roll < defeat_threshold {
        LastStandOutcome::GloriousDefeat
    } else {
        LastStandOutcome::MiraculousEscape
    }
}

/// Attempt a last stand for a battle that is heading toward defeat.
///
/// Returns `Some(record)` if a last stand was triggered, `None` otherwise.
/// The caller is responsible for mutating battle status based on the outcome.
pub fn check_last_stand(
    state: &mut CampaignState,
    battle_idx: usize,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) -> Option<LastStandRecord> {
    let battle = &state.active_battles[battle_idx];

    // Only trigger when the battle is active and the party is losing badly
    if battle.status != BattleStatus::Active {
        return None;
    }
    if battle.party_health_ratio > 0.15 || battle.predicted_outcome > -0.3 {
        return None;
    }

    let candidate_id = find_last_stand_candidate(state, battle)?;

    let adv = state.adventurers.iter().find(|a| a.id == candidate_id)?;
    let adv_name = adv.name.clone();
    let adv_level = adv.level;

    // Determine which faction we're fighting (from the quest's source)
    let against_faction = state
        .active_quests
        .iter()
        .find(|q| q.id == battle.quest_id)
        .and_then(|q| q.request.source_faction_id);

    let tick = state.tick;

    // Emit trigger event
    let trigger_desc = format!(
        "{} refuses to fall — a last stand begins!",
        adv_name
    );
    events.push(WorldEvent::LastStandTriggered {
        adventurer_name: adv_name.clone(),
        description: trigger_desc.clone(),
    });

    // Roll outcome
    let outcome = roll_outcome(&mut state.rng, adv_level);

    let battle = &mut state.active_battles[battle_idx];
    let description;

    match outcome {
        LastStandOutcome::HeroicVictory => {
            description = format!(
                "{} rallies with impossible fury — the enemy breaks and flees!",
                adv_name
            );
            battle.status = BattleStatus::Victory;
            battle.party_health_ratio = 0.05; // barely alive
            battle.enemy_health_ratio = 0.0;

            // +20 morale to all party members
            let party_id = battle.party_id;
            for adv in &mut state.adventurers {
                if adv.party_id == Some(party_id) && adv.status != AdventurerStatus::Dead {
                    adv.morale = (adv.morale + 20.0).min(100.0);
                }
            }

            // Reputation boost
            state.guild.reputation = (state.guild.reputation + 10.0).min(100.0);
        }
        LastStandOutcome::GloriousDefeat => {
            description = format!(
                "{} stands alone against the horde, buying time for the others. \
                 The enemy pays dearly for this victory.",
                adv_name
            );
            battle.status = BattleStatus::Defeat;
            battle.party_health_ratio = 0.0;
            battle.enemy_health_ratio = 0.2; // heavy enemy losses

            // Reputation from the sacrifice
            state.guild.reputation = (state.guild.reputation + 5.0).min(100.0);
        }
        LastStandOutcome::MiraculousEscape => {
            description = format!(
                "{} creates a desperate opening — the party escapes into the wilderness!",
                adv_name
            );
            battle.status = BattleStatus::Retreat;
            battle.party_health_ratio = 0.1;
            battle.enemy_health_ratio = 0.6;
        }
    }

    // Emit resolution event
    events.push(WorldEvent::LastStandResolved {
        outcome,
        description: description.clone(),
    });

    // Apply history tags to the last-stand adventurer
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == candidate_id) {
        *adv.history_tags.entry("last_stand".into()).or_default() += 1;
        *adv.history_tags.entry("heroic".into()).or_default() += 1;
    }

    // Chronicle entry (high significance)
    let event_id = state.next_event_id;
    state.next_event_id += 1;
    state.event_log.push(CampaignEvent {
        id: event_id,
        tick,
        description: description.clone(),
    });

    let record = LastStandRecord {
        adventurer_id: candidate_id,
        tick,
        against_faction,
        outcome,
        description,
    };

    state.last_stand_records.push(record.clone());

    Some(record)
}
