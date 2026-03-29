//! Skill challenge system — fires every 200 ticks.
//!
//! Non-combat skill checks (persuasion, stealth, knowledge, athletics, etc.)
//! occur during active quests. Resolution is based on adventurer archetype,
//! level, traits, companion bonuses, and mood, plus a random roll.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often to tick skill challenges (in ticks).
const CHALLENGE_INTERVAL: u64 = 7;

/// Chance per active quest of generating a skill challenge each tick.
const CHALLENGE_CHANCE: f32 = 0.30;

/// Tick skill challenges: generate new ones for active quests and resolve pending ones.
pub fn tick_skill_challenges(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % CHALLENGE_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Phase 1: Generate new challenges for in-progress quests.
    generate_challenges(state, events);

    // Phase 2: Resolve unresolved challenges.
    resolve_challenges(state, events);
}

// ---------------------------------------------------------------------------
// Challenge generation
// ---------------------------------------------------------------------------

fn generate_challenges(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect quest info we need: (quest_id, quest_type, party_member_ids, threat_level).
    let quest_infos: Vec<(u32, QuestType, Vec<u32>, f32)> = state
        .active_quests
        .iter()
        .filter(|q| q.status == ActiveQuestStatus::InProgress)
        .filter_map(|q| {
            let party = q
                .dispatched_party_id
                .and_then(|pid| state.parties.iter().find(|p| p.id == pid));
            party.map(|p| {
                (
                    q.id,
                    q.request.quest_type,
                    p.member_ids.clone(),
                    q.request.threat_level,
                )
            })
        })
        .collect();

    for (quest_id, quest_type, member_ids, threat_level) in quest_infos {
        // Roll chance per quest.
        let roll = lcg_f32(&mut state.rng);
        if roll >= CHALLENGE_CHANCE {
            continue;
        }

        if member_ids.is_empty() {
            continue;
        }

        // Pick skill type based on quest type.
        let skill_type = pick_skill_for_quest(quest_type, &mut state.rng);

        // Pick adventurer from party.
        let adv_idx = (lcg_next(&mut state.rng) as usize) % member_ids.len();
        let adventurer_id = member_ids[adv_idx];

        // Difficulty scales with quest threat level + some randomness.
        let base_difficulty = threat_level * 0.8;
        let noise = lcg_f32(&mut state.rng) * 20.0 - 10.0; // [-10, 10]
        let difficulty = (base_difficulty + noise).clamp(10.0, 95.0);

        let challenge_id = state.next_event_id;
        state.next_event_id += 1;

        let challenge = SkillChallenge {
            id: challenge_id,
            skill_type,
            difficulty,
            quest_id: Some(quest_id),
            adventurer_id,
            succeeded: false,
            resolved: false,
        };

        state.skill_challenges.push(challenge);

        events.push(WorldEvent::SkillChallengePresented {
            challenge_id,
            skill_type,
            difficulty,
            adventurer_id,
            quest_id: Some(quest_id),
        });
    }
}

/// Pick a skill type appropriate for the quest type.
fn pick_skill_for_quest(quest_type: QuestType, rng: &mut u64) -> SkillType {
    let candidates = match quest_type {
        QuestType::Diplomatic => &[SkillType::Persuasion, SkillType::Intimidation][..],
        QuestType::Exploration => &[SkillType::Perception, SkillType::Survival][..],
        QuestType::Combat => &[SkillType::Athletics, SkillType::Intimidation][..],
        QuestType::Gather => &[SkillType::Craftsmanship, SkillType::Survival][..],
        QuestType::Escort => &[SkillType::Perception, SkillType::Athletics][..],
        QuestType::Rescue => &[SkillType::Stealth, SkillType::Athletics][..],
    };
    let idx = (lcg_next(rng) as usize) % candidates.len();
    candidates[idx]
}

// ---------------------------------------------------------------------------
// Challenge resolution
// ---------------------------------------------------------------------------

fn resolve_challenges(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect indices of unresolved challenges.
    let unresolved_indices: Vec<usize> = state
        .skill_challenges
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.resolved)
        .map(|(i, _)| i)
        .collect();

    for idx in unresolved_indices {
        // Extract challenge data we need.
        let challenge_id = state.skill_challenges[idx].id;
        let skill_type = state.skill_challenges[idx].skill_type;
        let difficulty = state.skill_challenges[idx].difficulty;
        let adventurer_id = state.skill_challenges[idx].adventurer_id;
        let quest_id = state.skill_challenges[idx].quest_id;

        // Compute adventurer skill.
        let skill = compute_skill(state, adventurer_id, skill_type);

        // Random roll [0, 20).
        let roll = lcg_f32(&mut state.rng) * 20.0;
        let total = skill + roll;
        let succeeded = total > difficulty;

        // Determine critical results.
        let is_critical_success = skill > difficulty + 30.0;
        let is_critical_failure = skill < difficulty - 30.0 && !succeeded;

        // Mark challenge as resolved.
        state.skill_challenges[idx].resolved = true;
        state.skill_challenges[idx].succeeded = succeeded;

        if is_critical_success {
            apply_critical_success(state, adventurer_id, quest_id, events);
            events.push(WorldEvent::CriticalSuccess {
                challenge_id,
                adventurer_id,
                skill_type,
            });
        } else if is_critical_failure {
            apply_critical_failure(state, adventurer_id, quest_id, events);
            events.push(WorldEvent::CriticalFailure {
                challenge_id,
                adventurer_id,
                skill_type,
            });
        } else if succeeded {
            apply_success(state, adventurer_id, quest_id, events);
            events.push(WorldEvent::SkillChallengeSucceeded {
                challenge_id,
                adventurer_id,
                skill_type,
            });
        } else {
            apply_failure(state, adventurer_id, quest_id, events);
            events.push(WorldEvent::SkillChallengeFailed {
                challenge_id,
                adventurer_id,
                skill_type,
            });
        }
    }
}

/// Compute an adventurer's effective skill for a given skill type.
///
/// Skill = base(archetype) + level/5 + trait_bonus + bond_bonus + mood_bonus
fn compute_skill(state: &CampaignState, adventurer_id: u32, skill_type: SkillType) -> f32 {
    let adv = match state.adventurers.iter().find(|a| a.id == adventurer_id) {
        Some(a) => a,
        None => return 0.0,
    };

    // Base skill from archetype.
    let base = archetype_base_skill(&adv.archetype, skill_type);

    // Level scaling.
    let level_bonus = adv.level as f32 / 5.0;

    // Trait bonus: certain traits give bonuses to certain skills.
    let trait_bonus = compute_trait_bonus(&adv.traits, skill_type);

    // Companion/bond bonus: average bond strength with party members.
    let bond_bonus = if let Some(party) = adv
        .party_id
        .and_then(|pid| state.parties.iter().find(|p| p.id == pid))
    {
        super::bonds::average_party_bond(&state.adventurer_bonds, &party.member_ids) * 0.1
    } else {
        0.0
    };

    // Mood bonus from morale.
    let mood_bonus = (adv.morale - 50.0) * 0.2; // [-10, +10] for morale 0-100

    base + level_bonus + trait_bonus + bond_bonus + mood_bonus
}

/// Base skill score for an archetype/skill combination.
fn archetype_base_skill(archetype: &str, skill_type: SkillType) -> f32 {
    match (archetype, skill_type) {
        // Warriors/knights — good at athletics and intimidation
        ("knight" | "warrior" | "paladin", SkillType::Athletics) => 50.0,
        ("knight" | "warrior" | "paladin", SkillType::Intimidation) => 45.0,
        ("knight" | "warrior" | "paladin", _) => 30.0,

        // Rangers/scouts — good at survival and perception
        ("ranger" | "scout", SkillType::Survival) => 50.0,
        ("ranger" | "scout", SkillType::Perception) => 50.0,
        ("ranger" | "scout", SkillType::Stealth) => 45.0,
        ("ranger" | "scout", _) => 30.0,

        // Mages/wizards — good at knowledge
        ("mage" | "wizard" | "sage", SkillType::Knowledge) => 55.0,
        ("mage" | "wizard" | "sage", SkillType::Perception) => 40.0,
        ("mage" | "wizard" | "sage", _) => 25.0,

        // Rogues/assassins — good at stealth
        ("rogue" | "assassin" | "thief", SkillType::Stealth) => 55.0,
        ("rogue" | "assassin" | "thief", SkillType::Perception) => 45.0,
        ("rogue" | "assassin" | "thief", SkillType::Craftsmanship) => 40.0,
        ("rogue" | "assassin" | "thief", _) => 30.0,

        // Healers/clerics — good at persuasion and knowledge
        ("healer" | "cleric" | "priest", SkillType::Persuasion) => 50.0,
        ("healer" | "cleric" | "priest", SkillType::Knowledge) => 45.0,
        ("healer" | "cleric" | "priest", _) => 30.0,

        // Bards — good at persuasion and intimidation
        ("bard", SkillType::Persuasion) => 55.0,
        ("bard", SkillType::Intimidation) => 40.0,
        ("bard", _) => 35.0,

        // Craftsmen/engineers
        ("engineer" | "blacksmith" | "crafter", SkillType::Craftsmanship) => 55.0,
        ("engineer" | "blacksmith" | "crafter", SkillType::Knowledge) => 40.0,
        ("engineer" | "blacksmith" | "crafter", _) => 30.0,

        // Default for unknown archetypes.
        (_, _) => 35.0,
    }
}

/// Bonus from traits relevant to the skill type.
fn compute_trait_bonus(traits: &[String], skill_type: SkillType) -> f32 {
    let mut bonus = 0.0_f32;
    for t in traits {
        let t_lower = t.to_lowercase();
        bonus += match skill_type {
            SkillType::Persuasion if t_lower.contains("charisma") || t_lower.contains("diplomat") => 10.0,
            SkillType::Stealth if t_lower.contains("sneaky") || t_lower.contains("shadow") => 10.0,
            SkillType::Knowledge if t_lower.contains("scholar") || t_lower.contains("lore") => 10.0,
            SkillType::Athletics if t_lower.contains("strong") || t_lower.contains("endur") => 10.0,
            SkillType::Survival if t_lower.contains("woodsman") || t_lower.contains("wilder") => 10.0,
            SkillType::Intimidation if t_lower.contains("fierce") || t_lower.contains("menac") => 10.0,
            SkillType::Perception if t_lower.contains("vigilant") || t_lower.contains("keen") => 10.0,
            SkillType::Craftsmanship if t_lower.contains("craft") || t_lower.contains("tinker") => 10.0,
            _ => 0.0,
        };
    }
    bonus
}

// ---------------------------------------------------------------------------
// Effect application
// ---------------------------------------------------------------------------

fn apply_success(
    state: &mut CampaignState,
    adventurer_id: u32,
    quest_id: Option<u32>,
    _events: &mut Vec<WorldEvent>,
) {
    // Quest progress +20%: reduce quest elapsed_ms to simulate faster completion.
    // We approximate by adding a quest event marker.
    if let Some(qid) = quest_id {
        if let Some(quest) = state.active_quests.iter_mut().find(|q| q.id == qid) {
            quest.events.push(QuestEvent {
                tick: state.tick,
                description: "Skill challenge succeeded — quest progress boosted".into(),
            });
        }
    }

    // Bonus XP.
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        adv.xp += 15;
        // History tag.
        *adv.history_tags.entry("skill_success".into()).or_insert(0) += 1;
    }

    // Bonus loot: small gold reward.
    state.guild.gold += 5.0;
}

fn apply_failure(
    state: &mut CampaignState,
    adventurer_id: u32,
    _quest_id: Option<u32>,
    _events: &mut Vec<WorldEvent>,
) {
    // Minor injury and morale loss.
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        adv.injury = (adv.injury + 5.0).min(100.0);
        adv.morale = (adv.morale - 5.0).max(0.0);
    }
}

fn apply_critical_success(
    state: &mut CampaignState,
    adventurer_id: u32,
    quest_id: Option<u32>,
    _events: &mut Vec<WorldEvent>,
) {
    // Double rewards compared to normal success.
    if let Some(qid) = quest_id {
        if let Some(quest) = state.active_quests.iter_mut().find(|q| q.id == qid) {
            quest.events.push(QuestEvent {
                tick: state.tick,
                description: "Legendary skill triumph — quest greatly advanced".into(),
            });
        }
    }

    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        adv.xp += 30; // double XP
        *adv.history_tags.entry("legendary_skill".into()).or_insert(0) += 1;
    }

    // Double gold bonus.
    state.guild.gold += 10.0;

    // Reputation boost.
    state.guild.reputation = (state.guild.reputation + 2.0).min(100.0);
}

fn apply_critical_failure(
    state: &mut CampaignState,
    adventurer_id: u32,
    quest_id: Option<u32>,
    _events: &mut Vec<WorldEvent>,
) {
    // Major setback.
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
        adv.injury = (adv.injury + 15.0).min(100.0);
        adv.morale = (adv.morale - 15.0).max(0.0);
        adv.stress = (adv.stress + 10.0).min(100.0);
    }

    // Quest complication: add delay event.
    if let Some(qid) = quest_id {
        if let Some(quest) = state.active_quests.iter_mut().find(|q| q.id == qid) {
            quest.events.push(QuestEvent {
                tick: state.tick,
                description: "Critical skill failure — quest complications".into(),
            });
        }
    }

    // Gold penalty.
    state.guild.gold = (state.guild.gold - 5.0).max(0.0);
}
