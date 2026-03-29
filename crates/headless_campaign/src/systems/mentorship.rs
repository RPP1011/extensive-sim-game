//! Mentor-apprentice training system — every 100 ticks.
//!
//! Experienced adventurers can mentor newer ones, accelerating their growth.
//! Mentorship requires both parties to be Idle at the guild. The mentor's
//! archetype influences the apprentice's stat growth, and history tags
//! transfer partially over time.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{
    AdventurerStatus, CampaignState, lcg_f32,
};
use crate::systems::bonds::bond_key;

/// Base XP per mentorship tick (scaled by mentor level).
const BASE_MENTOR_XP: f32 = 5.0;

/// Ticks of mentorship before a history tag transfers.
const TAG_TRANSFER_THRESHOLD: u64 = 17;

/// Maximum mentorship duration in ticks.
const MAX_MENTORSHIP_TICKS: u64 = 67;

/// Maximum active mentorships per mentor.
pub const MAX_APPRENTICES_PER_MENTOR: usize = 2;

/// Tick cadence.
const MENTORSHIP_CADENCE: u64 = 3;

/// Main tick function. Called every 100 ticks.
///
/// For each active mentorship where both adventurers are Idle:
/// 1. Apprentice gains XP scaled by mentor level
/// 2. Mentor archetype biases apprentice stat growth
/// 3. History tags transfer partially after 500 ticks
/// 4. Bond between mentor and apprentice increases
/// 5. Mentorship ends after 2000 ticks or apprentice near mentor level
pub fn tick_mentorship(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % MENTORSHIP_CADENCE != 0 {
        return;
    }

    // Collect info needed to process mentorships without borrowing state mutably.
    struct MentorInfo {
        idx: usize,
        mentor_id: u32,
        apprentice_id: u32,
        mentor_level: u32,
        mentor_archetype: String,
        apprentice_level: u32,
        both_idle: bool,
        started_tick: u64,
        elapsed: u64,
    }

    let infos: Vec<MentorInfo> = state
        .mentor_assignments
        .iter()
        .enumerate()
        .map(|(idx, ma)| {
            let mentor = state.adventurers.iter().find(|a| a.id == ma.mentor_id);
            let apprentice = state.adventurers.iter().find(|a| a.id == ma.apprentice_id);

            let (mentor_level, mentor_archetype, mentor_idle) = mentor
                .map(|m| (m.level, m.archetype.clone(), m.status == AdventurerStatus::Idle))
                .unwrap_or((0, String::new(), false));

            let (apprentice_level, apprentice_idle) = apprentice
                .map(|a| (a.level, a.status == AdventurerStatus::Idle))
                .unwrap_or((0, false));

            let mentor_alive = mentor.map(|m| m.status != AdventurerStatus::Dead).unwrap_or(false);
            let apprentice_alive = apprentice.map(|a| a.status != AdventurerStatus::Dead).unwrap_or(false);

            MentorInfo {
                idx,
                mentor_id: ma.mentor_id,
                apprentice_id: ma.apprentice_id,
                mentor_level,
                mentor_archetype,
                apprentice_level,
                both_idle: mentor_idle && apprentice_idle && mentor_alive && apprentice_alive,
                started_tick: ma.started_tick,
                elapsed: state.tick.saturating_sub(ma.started_tick),
            }
        })
        .collect();

    // Track which mentorships should be completed.
    let mut completed_indices: Vec<usize> = Vec::new();

    for info in &infos {
        // Check termination conditions.
        let level_cap_reached = info.apprentice_level + 2 >= info.mentor_level;
        let duration_expired = info.elapsed >= MAX_MENTORSHIP_TICKS;

        if duration_expired || level_cap_reached {
            completed_indices.push(info.idx);
            let xp_gained = state.mentor_assignments[info.idx].xp_transferred;
            events.push(WorldEvent::MentorshipCompleted {
                mentor_id: info.mentor_id,
                apprentice_id: info.apprentice_id,
                xp_gained,
            });
            continue;
        }

        if !info.both_idle {
            continue;
        }

        // --- XP gain ---
        let xp_gain = BASE_MENTOR_XP * (1.0 + info.mentor_level as f32 / 20.0);
        state.mentor_assignments[info.idx].xp_transferred += xp_gain;

        if let Some(apprentice) = state.adventurers.iter_mut().find(|a| a.id == info.apprentice_id) {
            apprentice.xp += xp_gain as u32;

            // --- Archetype-biased stat growth ---
            // Small stat nudge per mentorship tick based on mentor's archetype.
            let stat_gain = 0.1 * lcg_f32(&mut state.rng) + 0.05;
            match info.mentor_archetype.as_str() {
                "knight" | "warrior" | "paladin" | "berserker" => {
                    apprentice.stats.defense += stat_gain;
                    apprentice.stats.max_hp += stat_gain * 2.0;
                }
                "ranger" | "rogue" | "assassin" => {
                    apprentice.stats.speed += stat_gain;
                    apprentice.stats.attack += stat_gain;
                }
                "mage" | "warlock" | "sorcerer" => {
                    apprentice.stats.ability_power += stat_gain;
                }
                "cleric" | "priest" | "healer" | "shaman" => {
                    apprentice.stats.ability_power += stat_gain * 0.5;
                    apprentice.stats.defense += stat_gain * 0.5;
                }
                _ => {
                    // Generic: spread evenly
                    apprentice.stats.attack += stat_gain * 0.5;
                    apprentice.stats.defense += stat_gain * 0.5;
                }
            }

            // --- Level up check (same formula as quest_lifecycle) ---
            let xp_mult = state.config.quest_lifecycle.level_up_xp_multiplier;
            let threshold = apprentice.level * apprentice.level * xp_mult;
            if apprentice.xp >= threshold {
                apprentice.level += 1;
                apprentice.stats.max_hp += state.config.quest_lifecycle.level_hp_gain;
                apprentice.stats.attack += state.config.quest_lifecycle.level_attack_gain;
                apprentice.stats.defense += state.config.quest_lifecycle.level_defense_gain;
                events.push(WorldEvent::AdventurerLevelUp {
                    adventurer_id: info.apprentice_id,
                    new_level: apprentice.level,
                });
            }
        }

        // --- History tag transfer ---
        // After TAG_TRANSFER_THRESHOLD ticks, transfer partial tag credit.
        if info.elapsed >= TAG_TRANSFER_THRESHOLD && info.elapsed % TAG_TRANSFER_THRESHOLD == 0 {
            // Collect mentor tags.
            let mentor_tags: Vec<(String, u32)> = state
                .adventurers
                .iter()
                .find(|a| a.id == info.mentor_id)
                .map(|m| {
                    m.history_tags
                        .iter()
                        .filter(|(_, &v)| v >= 5)
                        .map(|(k, _)| (k.clone(), 1))
                        .collect()
                })
                .unwrap_or_default();

            if !mentor_tags.is_empty() {
                if let Some(apprentice) = state.adventurers.iter_mut().find(|a| a.id == info.apprentice_id) {
                    for (tag, credit) in &mentor_tags {
                        *apprentice.history_tags.entry(tag.clone()).or_default() += credit;
                        events.push(WorldEvent::SkillTransferred {
                            from_id: info.mentor_id,
                            to_id: info.apprentice_id,
                            tag: tag.clone(),
                        });
                    }
                }
            }
        }

        // --- Bond increase ---
        let key = bond_key(info.mentor_id, info.apprentice_id);
        let entry = state.adventurer_bonds.entry(key).or_insert(0.0);
        *entry = (*entry + 1.0).min(100.0);
    }

    // Remove completed mentorships (reverse order to preserve indices).
    completed_indices.sort_unstable();
    for idx in completed_indices.into_iter().rev() {
        state.mentor_assignments.swap_remove(idx);
    }
}

/// Validate whether a mentor assignment is legal.
pub fn validate_assign_mentor(
    state: &CampaignState,
    mentor_id: u32,
    apprentice_id: u32,
) -> Result<(), String> {
    let mentor = state
        .adventurers
        .iter()
        .find(|a| a.id == mentor_id)
        .ok_or_else(|| format!("Mentor {} not found", mentor_id))?;

    let apprentice = state
        .adventurers
        .iter()
        .find(|a| a.id == apprentice_id)
        .ok_or_else(|| format!("Apprentice {} not found", apprentice_id))?;

    if mentor.status == AdventurerStatus::Dead {
        return Err("Mentor is dead".into());
    }
    if apprentice.status == AdventurerStatus::Dead {
        return Err("Apprentice is dead".into());
    }
    if mentor.status != AdventurerStatus::Idle {
        return Err("Mentor must be idle at guild".into());
    }
    if apprentice.status != AdventurerStatus::Idle {
        return Err("Apprentice must be idle at guild".into());
    }
    if mentor.level < apprentice.level + 3 {
        return Err(format!(
            "Mentor level ({}) must be at least 3 higher than apprentice level ({})",
            mentor.level, apprentice.level
        ));
    }

    // Check mentor capacity.
    let mentor_count = state
        .mentor_assignments
        .iter()
        .filter(|ma| ma.mentor_id == mentor_id)
        .count();
    if mentor_count >= MAX_APPRENTICES_PER_MENTOR {
        return Err(format!(
            "Mentor already has {} apprentices (max {})",
            mentor_count, MAX_APPRENTICES_PER_MENTOR
        ));
    }

    // Check apprentice not already mentored.
    let has_mentor = state
        .mentor_assignments
        .iter()
        .any(|ma| ma.apprentice_id == apprentice_id);
    if has_mentor {
        return Err("Apprentice already has a mentor".into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::MentorAssignment;

    fn make_test_state() -> CampaignState {
        CampaignState::default_test_campaign(42)
    }

    #[test]
    fn validate_level_requirement() {
        let mut state = make_test_state();
        // Set up two adventurers
        if state.adventurers.len() >= 2 {
            state.adventurers[0].level = 5;
            state.adventurers[0].status = AdventurerStatus::Idle;
            state.adventurers[1].level = 1;
            state.adventurers[1].status = AdventurerStatus::Idle;
            let m_id = state.adventurers[0].id;
            let a_id = state.adventurers[1].id;
            // Level diff = 4 >= 3, should be valid
            assert!(validate_assign_mentor(&state, m_id, a_id).is_ok());

            // Make levels too close
            state.adventurers[1].level = 4;
            assert!(validate_assign_mentor(&state, m_id, a_id).is_err());
        }
    }

    #[test]
    fn max_apprentices_enforced() {
        let mut state = make_test_state();
        if state.adventurers.len() >= 4 {
            let m_id = state.adventurers[0].id;
            state.adventurers[0].level = 10;
            state.adventurers[0].status = AdventurerStatus::Idle;
            for i in 1..4 {
                state.adventurers[i].level = 1;
                state.adventurers[i].status = AdventurerStatus::Idle;
            }
            let a1 = state.adventurers[1].id;
            let a2 = state.adventurers[2].id;
            let a3 = state.adventurers[3].id;

            state.mentor_assignments.push(MentorAssignment {
                mentor_id: m_id,
                apprentice_id: a1,
                started_tick: 0,
                xp_transferred: 0.0,
                skill_focus: None,
            });
            state.mentor_assignments.push(MentorAssignment {
                mentor_id: m_id,
                apprentice_id: a2,
                started_tick: 0,
                xp_transferred: 0.0,
                skill_focus: None,
            });

            // Third should fail
            assert!(validate_assign_mentor(&state, m_id, a3).is_err());
        }
    }

    #[test]
    fn apprentice_single_mentor() {
        let mut state = make_test_state();
        if state.adventurers.len() >= 3 {
            state.adventurers[0].level = 10;
            state.adventurers[0].status = AdventurerStatus::Idle;
            state.adventurers[1].level = 10;
            state.adventurers[1].status = AdventurerStatus::Idle;
            state.adventurers[2].level = 1;
            state.adventurers[2].status = AdventurerStatus::Idle;
            let m1 = state.adventurers[0].id;
            let m2 = state.adventurers[1].id;
            let a = state.adventurers[2].id;

            state.mentor_assignments.push(MentorAssignment {
                mentor_id: m1,
                apprentice_id: a,
                started_tick: 0,
                xp_transferred: 0.0,
                skill_focus: None,
            });

            // Second mentor for same apprentice should fail
            assert!(validate_assign_mentor(&state, m2, a).is_err());
        }
    }
}
