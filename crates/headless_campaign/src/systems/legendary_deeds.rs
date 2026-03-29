//! Legendary deeds and reputation titles — every 200 ticks.
//!
//! Adventurers earn titles from gameplay achievements (kill counts,
//! diplomatic actions, exploration, near-death survivals, etc.).
//! Each deed can only be earned once per adventurer.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{AdventurerStatus, CampaignState, DeedType};

/// Check adventurers for legendary deed thresholds every 200 ticks.
pub fn tick_legendary_deeds(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 7 != 0 || state.tick == 0 {
        return;
    }

    // We need to iterate adventurers mutably, so collect deed grants first.
    let mut grants: Vec<(usize, DeedType, String, String, crate::state::DeedBonus)> = Vec::new();

    for (idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        let tags = &adv.history_tags;

        // Combat kills (combat + high_threat tags as proxy for kills)
        let kill_count = tags.get("combat").copied().unwrap_or(0)
            + tags.get("high_threat").copied().unwrap_or(0);
        if kill_count > 10 && !has_deed(&adv.deeds, &DeedType::Slayer) {
            grants.push((
                idx,
                DeedType::Slayer,
                "Slayer".into(),
                format!("{} has slain over 10 fearsome foes", adv.name),
                crate::state::DeedBonus::CombatPowerBoost(0.05),
            ));
        }

        // Diplomatic history
        let diplo_count = tags.get("diplomatic").copied().unwrap_or(0);
        if diplo_count > 5 && !has_deed(&adv.deeds, &DeedType::Peacemaker) {
            grants.push((
                idx,
                DeedType::Peacemaker,
                "Peacemaker".into(),
                format!("{} is renowned for resolving crises through diplomacy", adv.name),
                crate::state::DeedBonus::FactionRelationBoost(10.0),
            ));
        }

        // Exploration
        let explore_count = tags.get("exploration").copied().unwrap_or(0);
        if explore_count > 8 && !has_deed(&adv.deeds, &DeedType::Explorer) {
            grants.push((
                idx,
                DeedType::Explorer,
                "Explorer".into(),
                format!("{} has charted the unknown reaches of the world", adv.name),
                crate::state::DeedBonus::QuestRewardBoost(0.15),
            ));
        }

        // Near-death survivals
        let near_death = tags.get("near_death").copied().unwrap_or(0);
        if near_death > 3 && !has_deed(&adv.deeds, &DeedType::Survivor) {
            grants.push((
                idx,
                DeedType::Survivor,
                "The Undying".into(),
                format!("{} has cheated death more times than anyone can count", adv.name),
                crate::state::DeedBonus::MoraleAura(0.10),
            ));
        }

        // Quest completions (sum of all quest-type history tags)
        let quest_count = tags.get("combat").copied().unwrap_or(0)
            + tags.get("exploration").copied().unwrap_or(0)
            + tags.get("diplomatic").copied().unwrap_or(0)
            + tags.get("escort").copied().unwrap_or(0)
            + tags.get("rescue").copied().unwrap_or(0)
            + tags.get("gather").copied().unwrap_or(0);
        if quest_count > 15 && !has_deed(&adv.deeds, &DeedType::Wealthy) {
            grants.push((
                idx,
                DeedType::Wealthy,
                "Legendary".into(),
                format!("{} has completed over 15 quests and earned legendary status", adv.name),
                crate::state::DeedBonus::QuestRewardBoost(0.20),
            ));
        }

        // Solo kills
        let solo_count = tags.get("solo").copied().unwrap_or(0);
        if solo_count > 5 && !has_deed(&adv.deeds, &DeedType::Undefeated) {
            grants.push((
                idx,
                DeedType::Undefeated,
                "Lone Wolf".into(),
                format!("{} has proven themselves a deadly solo combatant", adv.name),
                crate::state::DeedBonus::CombatPowerBoost(0.10),
            ));
        }

        // Region defense
        let defense_count = tags.get("region_defense").copied().unwrap_or(0);
        if defense_count > 5 && !has_deed(&adv.deeds, &DeedType::Defender) {
            grants.push((
                idx,
                DeedType::Defender,
                "Shield of the Realm".into(),
                format!("{} has defended the realm against countless threats", adv.name),
                crate::state::DeedBonus::RecruitmentBoost(0.10),
            ));
        }

        // Rescue/escort (savior)
        let rescue_count = tags.get("rescue").copied().unwrap_or(0)
            + tags.get("escort").copied().unwrap_or(0);
        if rescue_count > 5 && !has_deed(&adv.deeds, &DeedType::Savior) {
            grants.push((
                idx,
                DeedType::Savior,
                "Savior".into(),
                format!("{} is known far and wide for saving those in need", adv.name),
                crate::state::DeedBonus::FactionRelationBoost(5.0),
            ));
        }
    }

    // Apply grants
    let tick = state.tick;
    for (idx, deed_type, title, description, bonus) in grants {
        let adv = &mut state.adventurers[idx];
        let adv_id = adv.id;

        adv.deeds.push(crate::state::LegendaryDeed {
            title: title.clone(),
            earned_at_tick: tick,
            deed_type,
            bonus,
        });

        events.push(WorldEvent::LegendaryDeedEarned {
            adventurer_id: adv_id,
            title,
            description,
        });
    }
}

/// Check whether an adventurer already has a deed of the given type.
fn has_deed(deeds: &[crate::state::LegendaryDeed], dtype: &DeedType) -> bool {
    deeds.iter().any(|d| &d.deed_type == dtype)
}
