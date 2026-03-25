//! Guild charter/constitution system — every 500 ticks.
//!
//! The guild has a written charter that codifies rules and policies, affecting
//! behavior and creating political tension when violated. Articles constrain
//! guild actions but provide bonuses when upheld. Low legitimacy triggers
//! council challenges and desertions.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Tick the charter system every 500 ticks.
pub fn tick_charter(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 17 != 0 || state.tick == 0 {
        return;
    }

    // Skip if no charter has been ratified
    if state.charter.ratified_tick == 0 {
        // Auto-ratify a basic charter on first check (tick 500)
        ratify_default_charter(state, events);
        return;
    }

    // --- Check for violations ---
    check_violations(state, events);

    // --- Apply legitimacy effects ---
    apply_legitimacy_effects(state, events);
}

/// Ratify a default charter with basic articles.
fn ratify_default_charter(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    state.charter.ratified_tick = state.tick;
    state.charter.legitimacy = 60.0;
    state.charter.articles.push(CharterArticle {
        id: 1,
        article_type: ArticleType::MeritPromotion,
        description: "Leadership positions awarded by demonstrated competence".into(),
        active: true,
    });
    state.charter.articles.push(CharterArticle {
        id: 2,
        article_type: ArticleType::SharedLoot,
        description: "Quest rewards shared equally among all guild members".into(),
        active: true,
    });

    events.push(WorldEvent::CharterAmended {
        description: "Guild charter ratified with founding articles".into(),
    });
    events.push(WorldEvent::LegitimacyChanged {
        old: 0.0,
        new: 60.0,
    });
}

/// Check if current guild state violates any active charter articles.
fn check_violations(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let mut violation_count = 0u32;

    for article in &state.charter.articles {
        if !article.active {
            continue;
        }
        let violated = match article.article_type {
            ArticleType::NoOffensiveWars => {
                // Violated if guild is at war with any faction
                state
                    .factions
                    .iter()
                    .any(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
            }
            ArticleType::NoBlackMarket => {
                // Violated if black market has active deals or accumulated profit
                !state.black_market.available_deals.is_empty()
                    || state.black_market.total_profit > 0.0
            }
            ArticleType::MercenaryBan => {
                // Violated if any mercenaries are currently hired
                !state.hired_mercenaries.is_empty()
            }
            ArticleType::TithingRule => {
                // Violated if no temples exist or devotion is very low
                state.temples.is_empty()
                    || state
                        .temples
                        .iter()
                        .all(|t| t.devotion < 10.0)
            }
            // These articles don't have violation conditions — they're passive bonuses
            ArticleType::SharedLoot
            | ArticleType::DemocraticLeadership
            | ArticleType::MeritPromotion
            | ArticleType::OpenRecruitment => false,
        };

        if violated {
            violation_count += 1;
            events.push(WorldEvent::CharterViolation {
                article: article.description.clone(),
            });
        }
    }

    if violation_count > 0 {
        let old_legit = state.charter.legitimacy;
        let penalty = 5.0 * violation_count as f32;
        state.charter.legitimacy = (state.charter.legitimacy - penalty).max(0.0);

        // Morale hit from charter violations
        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale - 3.0).max(0.0);
            }
        }

        if (old_legit - state.charter.legitimacy).abs() > 0.01 {
            events.push(WorldEvent::LegitimacyChanged {
                old: old_legit,
                new: state.charter.legitimacy,
            });
        }
    } else {
        // No violations — slow legitimacy recovery
        let old_legit = state.charter.legitimacy;
        state.charter.legitimacy = (state.charter.legitimacy + 2.0).min(100.0);
        if (old_legit - state.charter.legitimacy).abs() > 0.01 {
            events.push(WorldEvent::LegitimacyChanged {
                old: old_legit,
                new: state.charter.legitimacy,
            });
        }
    }
}

/// Apply effects based on current legitimacy level.
fn apply_legitimacy_effects(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let legitimacy = state.charter.legitimacy;

    if legitimacy > 70.0 {
        // High legitimacy: morale boost, reputation gains
        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale + 5.0).min(100.0);
            }
        }
        state.guild.reputation = (state.guild.reputation + 1.0).min(100.0);
    } else if legitimacy < 30.0 {
        // Low legitimacy: morale penalty, desertion risk
        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale - 5.0).max(0.0);
            }
        }

        // Desertion risk: adventurers with low loyalty may leave
        let rng = &mut state.rng;
        let mut deserted_ids = Vec::new();
        for adv in &state.adventurers {
            if adv.status != AdventurerStatus::Dead
                && adv.loyalty < 30.0
                && adv.morale < 20.0
            {
                let roll = lcg_f32(rng);
                if roll < 0.1 {
                    deserted_ids.push(adv.id);
                }
            }
        }

        for did in &deserted_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *did) {
                adv.status = AdventurerStatus::Dead;
                events.push(WorldEvent::AdventurerDeserted {
                    adventurer_id: *did,
                    reason: "Lost faith in guild's legitimacy".into(),
                });
            }
        }

        // Council may vote to amend if democratic leadership is active
        let has_democratic = state.charter.articles.iter().any(|a| {
            a.active && matches!(a.article_type, ArticleType::DemocraticLeadership)
        });
        if has_democratic && !state.council_votes.iter().any(|v| {
            !v.resolved && v.topic.description().contains("charter")
        }) {
            let vote_id = state.next_vote_id;
            state.next_vote_id += 1;
            state.council_votes.push(CouncilVote {
                id: vote_id,
                topic: VoteTopic::ChangePolicy {
                    policy: "Amend guild charter to restore legitimacy".into(),
                },
                proposed_tick: state.tick,
                deadline_tick: state.tick + 400,
                votes_for: Vec::new(),
                votes_against: Vec::new(),
                resolved: false,
                passed: false,
            });
            events.push(WorldEvent::CouncilVoteProposed {
                topic: "Amend guild charter to restore legitimacy".into(),
            });
        }
    }

    // --- Article-specific passive effects ---
    for article in &state.charter.articles {
        if !article.active {
            continue;
        }
        match article.article_type {
            ArticleType::SharedLoot => {
                // Morale bonus for shared loot (already applied above for high legitimacy,
                // but shared loot gives a small extra boost regardless)
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead {
                        adv.morale = (adv.morale + 1.0).min(100.0);
                    }
                }
            }
            ArticleType::NoOffensiveWars => {
                // Diplomatic bonus when not at war
                if !state
                    .factions
                    .iter()
                    .any(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
                {
                    for f in &mut state.factions {
                        if f.diplomatic_stance != DiplomaticStance::Coalition {
                            f.relationship_to_guild =
                                (f.relationship_to_guild + 0.5).min(100.0);
                        }
                    }
                }
            }
            ArticleType::TithingRule => {
                // Gold cost: 10% of current gold as tithe
                let tithe = state.guild.gold * 0.001; // small per-tick tithe
                state.guild.gold = (state.guild.gold - tithe).max(0.0);
                // Boost temple devotion
                for t in &mut state.temples {
                    t.devotion = (t.devotion + 0.5).min(100.0);
                }
            }
            ArticleType::NoBlackMarket => {
                // Reputation bonus for clean guild
                state.guild.reputation = (state.guild.reputation + 0.2).min(100.0);
            }
            ArticleType::OpenRecruitment => {
                // Slightly boost recruitment chances (handled by recruitment system,
                // but we give a small loyalty penalty for diversity tension)
                // No-op here; recruitment system checks charter articles
            }
            ArticleType::MeritPromotion => {
                // Loyalty boost for high-level adventurers
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead && adv.level >= 5 {
                        adv.loyalty = (adv.loyalty + 1.0).min(100.0);
                    }
                }
            }
            ArticleType::DemocraticLeadership | ArticleType::MercenaryBan => {
                // DemocraticLeadership: enables council vote requirement (checked in apply_action)
                // MercenaryBan: morale boost from principled stance
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead {
                        adv.morale = (adv.morale + 0.5).min(100.0);
                    }
                }
            }
        }
    }
}

/// Process an AmendCharter action.
pub fn apply_amend_charter(
    state: &mut CampaignState,
    article_type_str: &str,
    add: bool,
    events: &mut Vec<WorldEvent>,
) -> Result<String, String> {
    // Parse article type from string
    let article_type = match article_type_str {
        "NoOffensiveWars" => ArticleType::NoOffensiveWars,
        "SharedLoot" => ArticleType::SharedLoot,
        "DemocraticLeadership" => ArticleType::DemocraticLeadership,
        "MeritPromotion" => ArticleType::MeritPromotion,
        "NoBlackMarket" => ArticleType::NoBlackMarket,
        "OpenRecruitment" => ArticleType::OpenRecruitment,
        "TithingRule" => ArticleType::TithingRule,
        "MercenaryBan" => ArticleType::MercenaryBan,
        _ => return Err(format!("Unknown article type: {}", article_type_str)),
    };

    // Amendment cost
    if state.guild.gold < 50.0 {
        return Err("Not enough gold (need 50) to amend charter".into());
    }

    // Check for council requirement
    let needs_council = state.charter.articles.iter().any(|a| {
        a.active && matches!(a.article_type, ArticleType::DemocraticLeadership)
    });

    if needs_council {
        // Check if a council vote passed for this amendment
        let vote_passed = state.council_votes.iter().any(|v| {
            v.resolved
                && v.passed
                && v.topic.description().contains("charter")
        });
        if !vote_passed {
            return Err("Democratic Leadership requires council vote for charter amendments".into());
        }
    }

    state.guild.gold -= 50.0;
    state.charter.amendments += 1;

    if add {
        // Check if already exists and active
        if state
            .charter
            .articles
            .iter()
            .any(|a| a.active && std::mem::discriminant(&a.article_type) == std::mem::discriminant(&article_type))
        {
            return Err(format!("Article {:?} is already active", article_type));
        }

        let next_id = state
            .charter
            .articles
            .iter()
            .map(|a| a.id)
            .max()
            .unwrap_or(0)
            + 1;

        let description = article_description(&article_type);
        state.charter.articles.push(CharterArticle {
            id: next_id,
            article_type: article_type.clone(),
            description: description.clone(),
            active: true,
        });

        events.push(WorldEvent::CharterAmended {
            description: format!("Added article: {}", description),
        });

        Ok(format!("Charter amended: added {:?}", article_type))
    } else {
        // Remove (deactivate) an existing article
        let found = state
            .charter
            .articles
            .iter_mut()
            .find(|a| a.active && std::mem::discriminant(&a.article_type) == std::mem::discriminant(&article_type));

        match found {
            Some(article) => {
                article.active = false;
                let desc = article.description.clone();
                events.push(WorldEvent::CharterAmended {
                    description: format!("Removed article: {}", desc),
                });
                Ok(format!("Charter amended: removed {:?}", article_type))
            }
            None => Err(format!("No active article of type {:?} to remove", article_type)),
        }
    }
}

/// Human-readable description for an article type.
fn article_description(article_type: &ArticleType) -> String {
    match article_type {
        ArticleType::NoOffensiveWars => {
            "The guild shall not declare offensive wars".into()
        }
        ArticleType::SharedLoot => {
            "Quest rewards shared equally among all guild members".into()
        }
        ArticleType::DemocraticLeadership => {
            "Major decisions require council majority vote".into()
        }
        ArticleType::MeritPromotion => {
            "Leadership positions awarded by demonstrated competence".into()
        }
        ArticleType::NoBlackMarket => {
            "Black market dealings are forbidden".into()
        }
        ArticleType::OpenRecruitment => {
            "The guild accepts recruits regardless of background".into()
        }
        ArticleType::TithingRule => {
            "Ten percent of guild income tithed to temples".into()
        }
        ArticleType::MercenaryBan => {
            "Hiring of mercenary companies is forbidden".into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_charter_ratified_at_tick_500() {
        let mut state = CampaignState::default_test_campaign(42);
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        // Before tick 500, no charter
        assert_eq!(state.charter.ratified_tick, 0);

        // Manually set tick to 500
        state.tick = 500;
        tick_charter(&mut state, &mut deltas, &mut events);

        assert_eq!(state.charter.ratified_tick, 500);
        assert_eq!(state.charter.legitimacy, 60.0);
        assert_eq!(state.charter.articles.len(), 2);
        assert!(events
            .iter()
            .any(|e| matches!(e, WorldEvent::CharterAmended { .. })));
    }

    #[test]
    fn charter_violation_reduces_legitimacy() {
        let mut state = CampaignState::default_test_campaign(42);
        state.charter.ratified_tick = 1;
        state.charter.legitimacy = 80.0;

        // Add NoOffensiveWars article
        state.charter.articles.push(CharterArticle {
            id: 1,
            article_type: ArticleType::NoOffensiveWars,
            description: "No wars".into(),
            active: true,
        });

        // Put a faction at war
        if let Some(f) = state.factions.first_mut() {
            f.diplomatic_stance = DiplomaticStance::AtWar;
        }

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        state.tick = 1000;
        tick_charter(&mut state, &mut deltas, &mut events);

        // Legitimacy should have dropped by 5 for violation, then +effects
        assert!(state.charter.legitimacy < 80.0);
        assert!(events
            .iter()
            .any(|e| matches!(e, WorldEvent::CharterViolation { .. })));
    }

    #[test]
    fn amend_charter_add_article() {
        let mut state = CampaignState::default_test_campaign(42);
        state.charter.ratified_tick = 1;
        state.charter.legitimacy = 60.0;
        state.guild.gold = 200.0;

        let mut events = Vec::new();
        let result =
            apply_amend_charter(&mut state, "MercenaryBan", true, &mut events);
        assert!(result.is_ok());
        assert_eq!(state.guild.gold, 150.0);
        assert!(state
            .charter
            .articles
            .iter()
            .any(|a| matches!(a.article_type, ArticleType::MercenaryBan)));
    }

    #[test]
    fn amend_charter_insufficient_gold() {
        let mut state = CampaignState::default_test_campaign(42);
        state.charter.ratified_tick = 1;
        state.guild.gold = 10.0;

        let mut events = Vec::new();
        let result =
            apply_amend_charter(&mut state, "MercenaryBan", true, &mut events);
        assert!(result.is_err());
    }

    #[test]
    fn amend_charter_remove_article() {
        let mut state = CampaignState::default_test_campaign(42);
        state.charter.ratified_tick = 1;
        state.guild.gold = 200.0;
        state.charter.articles.push(CharterArticle {
            id: 1,
            article_type: ArticleType::SharedLoot,
            description: "Shared loot".into(),
            active: true,
        });

        let mut events = Vec::new();
        let result =
            apply_amend_charter(&mut state, "SharedLoot", false, &mut events);
        assert!(result.is_ok());
        assert!(!state.charter.articles[0].active);
    }
}
