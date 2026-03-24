//! Diplomacy tick — every 200 ticks (~20s).
//!
//! Manages diplomatic agreements between factions:
//! - Expires agreements past their deadline
//! - AI factions propose trade, NAPs, and alliances based on relations
//! - War ceasefire checks (strength < 20 or heavy losses)
//! - Alliance coordination (shared enemies)
//! - Trade agreements generate gold income

use crate::headless_campaign::actions::{agreement_type_name, StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Diplomacy cadence: every 200 ticks.
const DIPLOMACY_INTERVAL: u64 = 200;

/// Duration of a trade agreement in ticks.
const TRADE_DURATION: u64 = 5000;
/// Duration of a non-aggression pact in ticks.
const NAP_DURATION: u64 = 8000;
/// Duration of a military alliance in ticks.
const ALLIANCE_DURATION: u64 = 10000;

/// Gold per tick for trade agreements.
const TRADE_GOLD_PER_TICK: f32 = 0.05;

pub fn tick_diplomacy(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // Trade agreements generate gold every tick (not gated by interval).
    tick_trade_agreement_income(state, events);

    if state.tick % DIPLOMACY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // 1. Expire agreements past their deadline
    expire_agreements(state, events);

    // 2. War ceasefire checks
    check_war_ceasefires(state, events);

    // 3. AI faction proposals (trade, NAP, alliance)
    ai_propose_agreements(state, events);

    // 4. Alliance coordination — allied factions share enemies
    coordinate_alliances(state, events);
}

/// Generate gold income from active trade agreements involving the guild faction.
fn tick_trade_agreement_income(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    let guild_fid = state.diplomacy.guild_faction_id;
    let mut income = 0.0f32;

    for (fa, fb, ag) in &state.diplomacy.agreements {
        if let DiplomaticAgreement::TradeAgreement { gold_per_tick, .. } = ag {
            if *fa == guild_fid || *fb == guild_fid {
                income += gold_per_tick;
            }
        }
    }

    if income > 0.0 {
        state.guild.gold += income;
    }
}

/// Remove agreements that have expired.
fn expire_agreements(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let mut expired = Vec::new();

    for (i, (fa, fb, ag)) in state.diplomacy.agreements.iter().enumerate() {
        let expires = match ag {
            DiplomaticAgreement::TradeAgreement { expires_tick, .. } => *expires_tick,
            DiplomaticAgreement::NonAggressionPact { expires_tick } => *expires_tick,
            DiplomaticAgreement::MilitaryAlliance { expires_tick } => *expires_tick,
        };
        if tick >= expires {
            expired.push((i, *fa, *fb, agreement_type_name(ag).to_string()));
        }
    }

    // Remove in reverse order to preserve indices.
    expired.reverse();
    for (idx, fa, fb, atype) in expired {
        state.diplomacy.agreements.remove(idx);
        events.push(WorldEvent::AgreementExpired {
            faction_a: fa,
            faction_b: fb,
            agreement_type: atype,
        });
    }
}

/// Factions at war with strength < 20 seek ceasefire.
fn check_war_ceasefires(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n = state.factions.len();

    for fi in 0..n {
        let strength = state.factions[fi].military_strength;
        if strength >= 20.0 {
            continue;
        }

        let war_targets = state.factions[fi].at_war_with.clone();
        for &target in &war_targets {
            if target >= n {
                continue;
            }

            // Both sides must be weak, or the weak side sues for peace
            let target_strength = state.factions[target].military_strength;
            let should_ceasefire = strength < 20.0 || target_strength < 20.0;

            if should_ceasefire {
                // Roll for ceasefire acceptance (stronger side less likely to accept)
                let accept_chance = if target_strength < 30.0 { 0.8 } else { 0.3 };
                let roll = lcg_f32(&mut state.rng);
                if roll < accept_chance {
                    // Remove war between both factions
                    state.factions[fi].at_war_with.retain(|&id| id != target);
                    state.factions[target].at_war_with.retain(|&id| id != fi);

                    // Update stances if they were at war with the guild
                    let guild_fid = state.diplomacy.guild_faction_id;
                    if fi == guild_fid || target == guild_fid {
                        let other = if fi == guild_fid { target } else { fi };
                        if state.factions[other].diplomatic_stance == DiplomaticStance::AtWar
                            && state.factions[other].at_war_with.is_empty()
                        {
                            state.factions[other].diplomatic_stance = DiplomaticStance::Hostile;
                        }
                    }

                    // Create a NAP to prevent immediate re-declaration
                    state.diplomacy.agreements.push((
                        fi,
                        target,
                        DiplomaticAgreement::NonAggressionPact {
                            expires_tick: state.tick + NAP_DURATION / 2,
                        },
                    ));

                    events.push(WorldEvent::WarCeasefire {
                        faction_a: fi,
                        faction_b: target,
                        reason: format!(
                            "Exhaustion ceasefire (strength {:.0} vs {:.0})",
                            strength, target_strength
                        ),
                    });
                }
            }
        }
    }
}

/// AI factions propose agreements based on relationship levels.
fn ai_propose_agreements(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n = state.factions.len();
    let guild_fid = state.diplomacy.guild_faction_id;

    for fi in 0..n {
        for fj in (fi + 1)..n {
            // Determine the relationship between factions fi and fj.
            let relation = if fi == guild_fid {
                state.factions[fj].relationship_to_guild
            } else if fj == guild_fid {
                state.factions[fi].relationship_to_guild
            } else {
                // Use the diplomacy matrix for non-guild factions.
                if fi < state.diplomacy.relations.len()
                    && fj < state.diplomacy.relations[fi].len()
                {
                    state.diplomacy.relations[fi][fj] as f32
                } else {
                    0.0
                }
            };

            // Skip factions at war
            if state.factions[fi].at_war_with.contains(&fj)
                || state.factions[fj].at_war_with.contains(&fi)
            {
                continue;
            }

            // Check existing agreements
            let has_trade = state.diplomacy.agreements.iter().any(|(a, b, ag)| {
                ((*a == fi && *b == fj) || (*a == fj && *b == fi))
                    && matches!(ag, DiplomaticAgreement::TradeAgreement { .. })
            });
            let has_nap = state.diplomacy.agreements.iter().any(|(a, b, ag)| {
                ((*a == fi && *b == fj) || (*a == fj && *b == fi))
                    && matches!(ag, DiplomaticAgreement::NonAggressionPact { .. })
            });
            let has_alliance = state.diplomacy.agreements.iter().any(|(a, b, ag)| {
                ((*a == fi && *b == fj) || (*a == fj && *b == fi))
                    && matches!(ag, DiplomaticAgreement::MilitaryAlliance { .. })
            });

            // Factions with relation > 50: 5% chance propose trade
            if relation > 50.0 && !has_trade {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.05 {
                    state.diplomacy.agreements.push((
                        fi,
                        fj,
                        DiplomaticAgreement::TradeAgreement {
                            gold_per_tick: TRADE_GOLD_PER_TICK,
                            expires_tick: state.tick + TRADE_DURATION,
                        },
                    ));
                    events.push(WorldEvent::AgreementFormed {
                        faction_a: fi,
                        faction_b: fj,
                        agreement_type: "Trade".into(),
                    });
                }
            }

            // Factions with relation > 30: 3% chance propose NAP
            if relation > 30.0 && !has_nap {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.03 {
                    state.diplomacy.agreements.push((
                        fi,
                        fj,
                        DiplomaticAgreement::NonAggressionPact {
                            expires_tick: state.tick + NAP_DURATION,
                        },
                    ));
                    events.push(WorldEvent::AgreementFormed {
                        faction_a: fi,
                        faction_b: fj,
                        agreement_type: "NAP".into(),
                    });
                }
            }

            // Factions with relation > 70: 2% chance propose alliance
            if relation > 70.0 && !has_alliance {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.02 {
                    state.diplomacy.agreements.push((
                        fi,
                        fj,
                        DiplomaticAgreement::MilitaryAlliance {
                            expires_tick: state.tick + ALLIANCE_DURATION,
                        },
                    ));
                    events.push(WorldEvent::AgreementFormed {
                        faction_a: fi,
                        faction_b: fj,
                        agreement_type: "Alliance".into(),
                    });
                }
            }
        }
    }
}

/// Alliance coordination: factions with military alliances share enemies.
/// If one ally is at war, the other joins if they aren't already fighting.
fn coordinate_alliances(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect alliances first to avoid borrow issues.
    let alliances: Vec<(usize, usize)> = state
        .diplomacy
        .agreements
        .iter()
        .filter_map(|(a, b, ag)| {
            if matches!(ag, DiplomaticAgreement::MilitaryAlliance { .. }) {
                Some((*a, *b))
            } else {
                None
            }
        })
        .collect();

    for (ally_a, ally_b) in &alliances {
        // Check if ally_a is at war with anyone ally_b isn't
        let a_wars = state.factions[*ally_a].at_war_with.clone();
        for &enemy in &a_wars {
            if enemy == *ally_b {
                continue; // Don't drag ally into war with themselves
            }
            if !state.factions[*ally_b].at_war_with.contains(&enemy) {
                // 10% chance per tick to join the war
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.10 {
                    state.factions[*ally_b].at_war_with.push(enemy);
                    if !state.factions[enemy].at_war_with.contains(ally_b) {
                        state.factions[enemy].at_war_with.push(*ally_b);
                    }
                    events.push(WorldEvent::FactionActionTaken {
                        faction_id: *ally_b,
                        action: format!(
                            "Joined war against {} (alliance with {})",
                            state.factions[enemy].name,
                            state.factions[*ally_a].name
                        ),
                    });
                }
            }
        }

        // Check reverse direction
        let b_wars = state.factions[*ally_b].at_war_with.clone();
        for &enemy in &b_wars {
            if enemy == *ally_a {
                continue;
            }
            if !state.factions[*ally_a].at_war_with.contains(&enemy) {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.10 {
                    state.factions[*ally_a].at_war_with.push(enemy);
                    if !state.factions[enemy].at_war_with.contains(ally_a) {
                        state.factions[enemy].at_war_with.push(*ally_a);
                    }
                    events.push(WorldEvent::FactionActionTaken {
                        faction_id: *ally_a,
                        action: format!(
                            "Joined war against {} (alliance with {})",
                            state.factions[enemy].name,
                            state.factions[*ally_b].name
                        ),
                    });
                }
            }
        }
    }
}
