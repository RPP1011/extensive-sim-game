//! Faction technology/upgrade system — fires every 500 ticks.
//!
//! Factions research improvements in military, economic, and diplomatic
//! domains based on their situation. Tech levels provide passive bonuses
//! to combat strength, trade income, and relation gains. Milestones at
//! 25/50/75 grant capability boost events.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often faction tech advances (in ticks).
const TECH_INTERVAL: u64 = 17;

/// Advance faction technology levels based on each faction's situation.
pub fn tick_faction_tech(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TECH_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Ensure faction_techs vec is populated for all factions.
    while state.faction_techs.len() < state.factions.len() {
        let fid = state.faction_techs.len();
        state.faction_techs.push(FactionTech {
            faction_id: fid,
            ..Default::default()
        });
    }

    let n = state.factions.len();
    if n == 0 {
        return;
    }

    // Snapshot faction data we need to determine focus and growth rates.
    // We collect into vecs to avoid borrowing state while mutating faction_techs.
    struct FactionSnapshot {
        at_war: bool,
        is_coalition: bool,
        is_friendly: bool,
        relationship: f32,
        military_strength: f32,
    }

    let snapshots: Vec<FactionSnapshot> = state
        .factions
        .iter()
        .map(|f| FactionSnapshot {
            at_war: f.diplomatic_stance == DiplomaticStance::AtWar
                || !f.at_war_with.is_empty(),
            is_coalition: f.coalition_member,
            is_friendly: matches!(
                f.diplomatic_stance,
                DiplomaticStance::Friendly | DiplomaticStance::Coalition
            ),
            relationship: f.relationship_to_guild,
            military_strength: f.military_strength,
        })
        .collect();

    // Count trade agreements and alliances per faction for focus determination.
    // A faction that is friendly/coalition with the guild counts as having agreements.
    for fi in 0..n {
        let snap = &snapshots[fi];
        let tech = &mut state.faction_techs[fi];

        // Determine research focus based on situation.
        let focus = if snap.at_war {
            TechFocus::Military
        } else if snap.is_coalition {
            TechFocus::Diplomatic
        } else if snap.is_friendly && snap.relationship > 40.0 {
            TechFocus::Economic
        } else {
            TechFocus::Balanced
        };
        tech.research_focus = focus;

        // Track old levels for milestone detection.
        let old_mil = tech.military_tech;
        let old_eco = tech.economic_tech;
        let old_dip = tech.diplomatic_tech;

        // Apply research growth.
        match focus {
            TechFocus::Military => {
                let rate = if snap.at_war { 1.0 } else { 0.5 };
                tech.military_tech = (tech.military_tech + rate).min(100.0);
                tech.economic_tech = (tech.economic_tech + 0.1).min(100.0);
                tech.diplomatic_tech = (tech.diplomatic_tech + 0.1).min(100.0);
            }
            TechFocus::Economic => {
                // Wealthy factions (friendly, high relationship) research faster.
                let rate = if snap.relationship > 60.0 { 0.7 } else { 0.5 };
                tech.economic_tech = (tech.economic_tech + rate).min(100.0);
                tech.military_tech = (tech.military_tech + 0.1).min(100.0);
                tech.diplomatic_tech = (tech.diplomatic_tech + 0.1).min(100.0);
            }
            TechFocus::Diplomatic => {
                let rate = if snap.is_coalition { 0.7 } else { 0.5 };
                tech.diplomatic_tech = (tech.diplomatic_tech + rate).min(100.0);
                tech.military_tech = (tech.military_tech + 0.1).min(100.0);
                tech.economic_tech = (tech.economic_tech + 0.1).min(100.0);
            }
            TechFocus::Balanced => {
                tech.military_tech = (tech.military_tech + 0.3).min(100.0);
                tech.economic_tech = (tech.economic_tech + 0.3).min(100.0);
                tech.diplomatic_tech = (tech.diplomatic_tech + 0.3).min(100.0);
            }
        }

        // Emit advancement events for the primary focus.
        let primary_tech = match focus {
            TechFocus::Military => ("military", tech.military_tech),
            TechFocus::Economic => ("economic", tech.economic_tech),
            TechFocus::Diplomatic => ("diplomatic", tech.diplomatic_tech),
            TechFocus::Balanced => {
                // Pick whichever is highest.
                if tech.military_tech >= tech.economic_tech
                    && tech.military_tech >= tech.diplomatic_tech
                {
                    ("military", tech.military_tech)
                } else if tech.economic_tech >= tech.diplomatic_tech {
                    ("economic", tech.economic_tech)
                } else {
                    ("diplomatic", tech.diplomatic_tech)
                }
            }
        };

        events.push(WorldEvent::FactionTechAdvanced {
            faction: fi,
            tech: primary_tech.0.to_string(),
            level: primary_tech.1,
        });

        // Check milestones (25, 50, 75) for each tech type.
        check_milestone(fi, "military", old_mil, tech.military_tech, events);
        check_milestone(fi, "economic", old_eco, tech.economic_tech, events);
        check_milestone(fi, "diplomatic", old_dip, tech.diplomatic_tech, events);

        // Apply passive effects to faction state.
        // Military tech: +0.5% combat strength per 10 tech.
        let mil_bonus = tech.military_tech / 10.0 * 0.005;
        state.factions[fi].military_strength *= 1.0 + mil_bonus;
        // Cap at max.
        if state.factions[fi].military_strength > state.factions[fi].max_military_strength * 1.5 {
            state.factions[fi].military_strength = state.factions[fi].max_military_strength * 1.5;
        }

        // Diplomatic tech: +5% relation gains per 20 tech (applied as a small
        // per-tick drift toward the guild when relations are positive).
        let dip_bonus = tech.diplomatic_tech / 20.0 * 0.05;
        if state.factions[fi].relationship_to_guild > 0.0 {
            state.factions[fi].relationship_to_guild = (state.factions[fi]
                .relationship_to_guild
                * (1.0 + dip_bonus * 0.01))
                .min(100.0);
        }
    }

    // Economic tech effects: +1% trade income per 10 tech for friendly factions.
    // This applies to guild trade income from all friendly factions.
    let eco_bonus_total: f32 = state
        .faction_techs
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            *i < state.factions.len()
                && matches!(
                    state.factions[*i].diplomatic_stance,
                    DiplomaticStance::Friendly | DiplomaticStance::Coalition
                )
        })
        .map(|(_, ft)| ft.economic_tech / 10.0 * 0.01)
        .sum();

    if eco_bonus_total > 0.0 {
        let bonus_gold = state.guild.total_trade_income * eco_bonus_total;
        state.guild.gold += bonus_gold;
    }
}

/// Check if a tech crossed a milestone threshold and emit an event.
fn check_milestone(
    faction: usize,
    tech_name: &str,
    old: f32,
    new: f32,
    events: &mut Vec<WorldEvent>,
) {
    for &threshold in &[25.0, 50.0, 75.0] {
        if old < threshold && new >= threshold {
            let capability = match (tech_name, threshold as u32) {
                ("military", 25) => "improved_weapons",
                ("military", 50) => "siege_units",
                ("military", 75) => "armor_bonus",
                ("economic", 25) => "market_access",
                ("economic", 50) => "trade_network",
                ("economic", 75) => "banking_system",
                ("diplomatic", 25) => "embassy",
                ("diplomatic", 50) => "agreement_proposals",
                ("diplomatic", 75) => "diplomatic_immunity",
                _ => "advancement",
            };
            events.push(WorldEvent::FactionTechMilestone {
                faction,
                tech: tech_name.to_string(),
                capability: capability.to_string(),
            });
        }
    }
}
