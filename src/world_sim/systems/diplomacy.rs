#![allow(unused)]
//! Diplomacy system — every 7 ticks.
//!
//! Manages diplomatic agreements between factions:
//! - Expires agreements past their deadline
//! - AI factions propose trade, NAPs, and alliances based on relations
//! - War ceasefire checks (strength < 20 or heavy losses)
//! - Alliance coordination (shared enemies)
//! - Trade agreements generate gold income every tick
//!
//! Ported from `crates/headless_campaign/src/systems/diplomacy.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{ActionTags, WorldState, tags};

// NEEDS STATE: factions: Vec<FactionState> on WorldState
// NEEDS STATE: diplomacy: DiplomacyState on WorldState
//   DiplomacyState { guild_faction_id, agreements: Vec<(u32, u32, DiplomaticAgreement)>,
//                    relations: Vec<Vec<i32>> }
// NEEDS STATE: DiplomaticAgreement enum { TradeAgreement { gold_per_tick, expires_tick },
//              NonAggressionPact { expires_tick }, MilitaryAlliance { expires_tick } }
// NEEDS STATE: FactionState { id, diplomatic_stance, relationship_to_guild,
//              military_strength, at_war_with }
// NEEDS STATE: DiplomaticStance enum (AtWar, Hostile, Neutral, Friendly, Coalition)

// NEEDS DELTA: AddAgreement { faction_a: u32, faction_b: u32, agreement: DiplomaticAgreement }
// NEEDS DELTA: RemoveAgreement { faction_a: u32, faction_b: u32, agreement_type: String }
// NEEDS DELTA: SetDiplomaticStance { faction_id: u32, stance: DiplomaticStance }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: EndWar { faction_a: u32, faction_b: u32 }
// NEEDS DELTA: DeclareWar { attacker_id: u32, defender_id: u32 }

/// Cadence for full diplomacy processing.
const DIPLOMACY_INTERVAL: u64 = 7;

/// Duration of a trade agreement in ticks.
const TRADE_DURATION: u64 = 167;
/// Duration of a non-aggression pact in ticks.
const NAP_DURATION: u64 = 8000;
/// Duration of a military alliance in ticks.
const ALLIANCE_DURATION: u64 = 10000;

/// Gold per tick for trade agreements.
const TRADE_GOLD_PER_TICK: f32 = 0.05;

/// Deterministic hash for pseudo-random decisions.
#[inline]
fn deterministic_roll(tick: u64, a: u32, b: u32, salt: u32) -> f32 {
    let mut h = tick.wrapping_mul(6364136223846793005)
        .wrapping_add(a as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(b as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(salt as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

pub fn compute_diplomacy(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Trade agreement income runs every tick (not gated by interval).
    compute_trade_income(state, out);

    if state.tick % DIPLOMACY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // 1. Expire agreements past their deadline
    compute_expire_agreements(state, out);

    // 2. War ceasefire checks
    compute_ceasefires(state, out);

    // 3. AI faction proposals (trade, NAP, alliance)
    compute_proposals(state, out);

    // 4. Alliance coordination
    compute_alliance_coordination(state, out);
}

/// Trade agreements involving the guild generate gold income via TransferGold.
fn compute_trade_income(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Once state.diplomacy exists, iterate active trade agreements
    // involving the guild and emit TransferGold deltas.
    // TODO: When enabled, emit AddBehaviorTags with tags::DIPLOMACY(2.0) + tags::NEGOTIATION(1.0)
    //       alongside AddAgreement deltas.

    /*
    let guild_fid = state.diplomacy.guild_faction_id;

    for (fa, fb, ag) in &state.diplomacy.agreements {
        if let DiplomaticAgreement::TradeAgreement { gold_per_tick, .. } = ag {
            if *fa == guild_fid || *fb == guild_fid {
                // Model trade income as gold transferred from the trading partner
                let partner = if *fa == guild_fid { *fb } else { *fa };
                out.push(WorldDelta::TransferGold {
                    from_id: partner,
                    to_id: guild_fid,
                    amount: *gold_per_tick,
                });
            }
        }
    }
    */
}

/// Remove agreements that have expired.
fn compute_expire_agreements(state: &WorldState, out: &mut Vec<WorldDelta>) {
    /*
    let tick = state.tick;

    for (fa, fb, ag) in &state.diplomacy.agreements {
        let expires = match ag {
            DiplomaticAgreement::TradeAgreement { expires_tick, .. } => *expires_tick,
            DiplomaticAgreement::NonAggressionPact { expires_tick } => *expires_tick,
            DiplomaticAgreement::MilitaryAlliance { expires_tick } => *expires_tick,
        };
        if tick >= expires {
            out.push(WorldDelta::RemoveAgreement {
                faction_a: *fa,
                faction_b: *fb,
                agreement_type: agreement_type_name(ag).to_string(),
            });
        }
    }
    */
}

/// Factions at war with strength < 20 seek ceasefire.
fn compute_ceasefires(state: &WorldState, out: &mut Vec<WorldDelta>) {
    /*
    let n = state.factions.len();
    let guild_fid = state.diplomacy.guild_faction_id;

    for faction in &state.factions {
        let fi = faction.id;
        let strength = faction.military_strength;
        if strength >= 20.0 {
            continue;
        }

        for &target in &faction.at_war_with {
            if target as usize >= n { continue; }

            let target_strength = state.factions.iter()
                .find(|f| f.id == target)
                .map(|f| f.military_strength)
                .unwrap_or(0.0);

            let should_ceasefire = strength < 20.0 || target_strength < 20.0;
            if !should_ceasefire { continue; }

            // Deterministic ceasefire acceptance roll
            let accept_chance = if target_strength < 30.0 { 0.8 } else { 0.3 };
            let roll = deterministic_roll(state.tick, fi, target, 1);
            if roll < accept_chance {
                // End war between both factions
                out.push(WorldDelta::EndWar { faction_a: fi, faction_b: target });

                // Update stance if at war with guild
                if fi == guild_fid || target == guild_fid {
                    let other = if fi == guild_fid { target } else { fi };
                    // Check if the other faction has no remaining wars
                    let other_wars = state.factions.iter()
                        .find(|f| f.id == other)
                        .map(|f| f.at_war_with.len())
                        .unwrap_or(0);
                    if other_wars <= 1 {
                        // This was their only war
                        out.push(WorldDelta::SetDiplomaticStance {
                            faction_id: other,
                            stance: DiplomaticStance::Hostile,
                        });
                    }
                }

                // Create a NAP to prevent immediate re-declaration
                out.push(WorldDelta::AddAgreement {
                    faction_a: fi,
                    faction_b: target,
                    agreement: DiplomaticAgreement::NonAggressionPact {
                        expires_tick: state.tick + NAP_DURATION / 2,
                    },
                });
            }
        }
    }
    */
}

/// AI factions propose agreements based on relationship levels.
fn compute_proposals(state: &WorldState, out: &mut Vec<WorldDelta>) {
    /*
    let n = state.factions.len();
    let guild_fid = state.diplomacy.guild_faction_id;

    for (i, fi_faction) in state.factions.iter().enumerate() {
        for fj_faction in state.factions.iter().skip(i + 1) {
            let fi = fi_faction.id;
            let fj = fj_faction.id;

            // Determine relationship between factions
            let relation = if fi == guild_fid {
                fj_faction.relationship_to_guild
            } else if fj == guild_fid {
                fi_faction.relationship_to_guild
            } else {
                // Non-guild faction pair — use diplomacy matrix
                state.diplomacy.relations
                    .get(fi as usize)
                    .and_then(|row| row.get(fj as usize))
                    .copied()
                    .unwrap_or(0) as f32
            };

            // Skip factions at war
            if fi_faction.at_war_with.contains(&fj)
                || fj_faction.at_war_with.contains(&fi)
            {
                continue;
            }

            // Check existing agreements (read-only)
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

            // Trade proposal: relation > 50, 5% chance
            if relation > 50.0 && !has_trade {
                let roll = deterministic_roll(state.tick, fi, fj, 10);
                if roll < 0.05 {
                    out.push(WorldDelta::AddAgreement {
                        faction_a: fi,
                        faction_b: fj,
                        agreement: DiplomaticAgreement::TradeAgreement {
                            gold_per_tick: TRADE_GOLD_PER_TICK,
                            expires_tick: state.tick + TRADE_DURATION,
                        },
                    });
                }
            }

            // NAP proposal: relation > 30, 3% chance
            if relation > 30.0 && !has_nap {
                let roll = deterministic_roll(state.tick, fi, fj, 20);
                if roll < 0.03 {
                    out.push(WorldDelta::AddAgreement {
                        faction_a: fi,
                        faction_b: fj,
                        agreement: DiplomaticAgreement::NonAggressionPact {
                            expires_tick: state.tick + NAP_DURATION,
                        },
                    });
                }
            }

            // Alliance proposal: relation > 70, 2% chance
            if relation > 70.0 && !has_alliance {
                let roll = deterministic_roll(state.tick, fi, fj, 30);
                if roll < 0.02 {
                    out.push(WorldDelta::AddAgreement {
                        faction_a: fi,
                        faction_b: fj,
                        agreement: DiplomaticAgreement::MilitaryAlliance {
                            expires_tick: state.tick + ALLIANCE_DURATION,
                        },
                    });
                }
            }
        }
    }
    */
}

/// Alliance coordination: factions with military alliances share enemies.
fn compute_alliance_coordination(state: &WorldState, out: &mut Vec<WorldDelta>) {
    /*
    // Collect alliances
    let alliances: Vec<(u32, u32)> = state.diplomacy.agreements.iter()
        .filter_map(|(a, b, ag)| {
            if matches!(ag, DiplomaticAgreement::MilitaryAlliance { .. }) {
                Some((*a, *b))
            } else {
                None
            }
        })
        .collect();

    for (ally_a, ally_b) in &alliances {
        let a_faction = match state.factions.iter().find(|f| f.id == *ally_a) {
            Some(f) => f,
            None => continue,
        };

        // Check if ally_a is at war with anyone ally_b isn't
        for &enemy in &a_faction.at_war_with {
            if enemy == *ally_b { continue; }

            let b_at_war = state.factions.iter()
                .find(|f| f.id == *ally_b)
                .map(|f| f.at_war_with.contains(&enemy))
                .unwrap_or(false);

            if !b_at_war {
                // 10% chance per tick to join the war
                let roll = deterministic_roll(state.tick, *ally_b, enemy, 40);
                if roll < 0.10 {
                    out.push(WorldDelta::DeclareWar {
                        attacker_id: *ally_b,
                        defender_id: enemy,
                    });
                }
            }
        }

        // Reverse direction
        let b_faction = match state.factions.iter().find(|f| f.id == *ally_b) {
            Some(f) => f,
            None => continue,
        };

        for &enemy in &b_faction.at_war_with {
            if enemy == *ally_a { continue; }

            let a_at_war = a_faction.at_war_with.contains(&enemy);
            if !a_at_war {
                let roll = deterministic_roll(state.tick, *ally_a, enemy, 41);
                if roll < 0.10 {
                    out.push(WorldDelta::DeclareWar {
                        attacker_id: *ally_a,
                        defender_id: enemy,
                    });
                }
            }
        }
    }
    */
}
