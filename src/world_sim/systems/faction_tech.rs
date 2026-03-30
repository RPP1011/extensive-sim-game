#![allow(unused)]
//! Faction technology system — every 17 ticks.
//!
//! Factions research improvements in military, economic, and diplomatic
//! domains based on their situation. Tech levels provide passive bonuses
//! to combat strength, trade income, and relation gains. Milestones at
//! 25/50/75 grant capability boost events.
//!
//! Ported from `crates/headless_campaign/src/systems/faction_tech.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::entity_hash_f32;

//   FactionState { id, diplomatic_stance, coalition_member, relationship_to_guild,
//                  military_strength, max_military_strength, at_war_with }
//   DiplomaticStance enum (AtWar, Hostile, Neutral, Friendly, Coalition)
//   FactionTechState { faction_id: u32, military_tech: f32, economic_tech: f32,
//                      diplomatic_tech: f32, research_focus: TechFocus }
//   TechFocus enum { Military, Economic, Diplomatic, Balanced }


/// How often faction tech advances (in ticks).
const TECH_INTERVAL: u64 = 17;


pub fn compute_faction_tech(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % TECH_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Once state.factions, state.faction_techs, state.guild exist, enable this.

    /*
    // Ensure faction_techs is populated (in the real system this is handled
    // by initialization; here we skip factions without a tech entry).

    for faction in &state.factions {
        let fi = faction.id;

        let tech = match state.faction_techs.iter().find(|t| t.faction_id == fi) {
            Some(t) => t,
            None => continue,
        };

        // Determine research focus based on situation.
        let at_war = faction.diplomatic_stance == DiplomaticStance::AtWar
            || !faction.at_war_with.is_empty();
        let is_coalition = faction.coalition_member;
        let is_friendly = matches!(
            faction.diplomatic_stance,
            DiplomaticStance::Friendly | DiplomaticStance::Coalition
        );

        let focus = if at_war {
            TechFocus::Military
        } else if is_coalition {
            TechFocus::Diplomatic
        } else if is_friendly && faction.relationship_to_guild > 40.0 {
            TechFocus::Economic
        } else {
            TechFocus::Balanced
        };

        out.push(WorldDelta::SetResearchFocus {
            faction_id: fi,
            focus,
        });

        // Track old levels for milestone detection.
        let old_mil = tech.military_tech;
        let old_eco = tech.economic_tech;
        let old_dip = tech.diplomatic_tech;

        // Apply research growth.
        match focus {
            TechFocus::Military => {
                let rate = if at_war { 1.0 } else { 0.5 };
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "military".to_string(),
                    delta: rate,
                });
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "economic".to_string(),
                    delta: 0.1,
                });
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "diplomatic".to_string(),
                    delta: 0.1,
                });
            }
            TechFocus::Economic => {
                let rate = if faction.relationship_to_guild > 60.0 { 0.7 } else { 0.5 };
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "economic".to_string(),
                    delta: rate,
                });
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "military".to_string(),
                    delta: 0.1,
                });
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "diplomatic".to_string(),
                    delta: 0.1,
                });
            }
            TechFocus::Diplomatic => {
                let rate = if is_coalition { 0.7 } else { 0.5 };
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "diplomatic".to_string(),
                    delta: rate,
                });
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "military".to_string(),
                    delta: 0.1,
                });
                out.push(WorldDelta::AdjustTech {
                    faction_id: fi,
                    tech_type: "economic".to_string(),
                    delta: 0.1,
                });
            }
            TechFocus::Balanced => {
                for tech_type in &["military", "economic", "diplomatic"] {
                    out.push(WorldDelta::AdjustTech {
                        faction_id: fi,
                        tech_type: tech_type.to_string(),
                        delta: 0.3,
                    });
                }
            }
        }

        // Apply passive effects via existing deltas.

        // Military tech: +0.5% combat strength per 10 tech.
        let mil_bonus = tech.military_tech / 10.0 * 0.005;
        out.push(WorldDelta::AdjustMilitaryStrength {
            faction_id: fi,
            factor: 1.0 + mil_bonus,
        });

        // Diplomatic tech: small relation drift toward guild when positive.
        if faction.relationship_to_guild > 0.0 {
            let dip_bonus = tech.diplomatic_tech / 20.0 * 0.05;
            let drift = faction.relationship_to_guild * dip_bonus * 0.01;
            out.push(WorldDelta::AdjustRelationship {
                faction_id: fi,
                delta: drift,
            });
        }
    }

    // Economic tech effects: +1% trade income per 10 tech for friendly factions.
    // Applied as a gold bonus to the guild.
    let eco_bonus_total: f32 = state
        .faction_techs
        .iter()
        .filter(|ft| {
            state
                .factions
                .iter()
                .find(|f| f.id == ft.faction_id)
                .map(|f| {
                    matches!(
                        f.diplomatic_stance,
                        DiplomaticStance::Friendly | DiplomaticStance::Coalition
                    )
                })
                .unwrap_or(false)
        })
        .map(|ft| ft.economic_tech / 10.0 * 0.01)
        .sum();

    if eco_bonus_total > 0.0 {
        let bonus_gold = state.guild.total_trade_income * eco_bonus_total;
        // Map to TransferGold from a virtual "economy" source to guild.
        out.push(WorldDelta::TransferGold {
            from_entity: u32::MAX, // virtual source
            to_entity: state.diplomacy.guild_faction_id,
            amount: bonus_gold,
        });
    }
    */
}
