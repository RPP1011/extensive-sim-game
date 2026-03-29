#![allow(unused)]
//! Alliance bloc system — every 17 ticks.
//!
//! Factions formalize power blocs that coordinate military, economic, and
//! diplomatic actions as a unified force. Manages bloc formation, cohesion
//! drift, coordinated attacks, shared defense, trade benefits, and dissolution.
//!
//! Ported from `crates/headless_campaign/src/systems/alliance_blocs.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;

// NEEDS STATE: factions: Vec<FactionState> on WorldState
//   FactionState { id, name, military_strength, relationship_to_guild,
//                  at_war_with: Vec<u32> }
// NEEDS STATE: alliance_blocs: Vec<AllianceBlocState> on WorldState
//   AllianceBlocState { id, name, member_factions: Vec<u32>, leader_faction_id: u32,
//                       formed_tick: u64, strength: f32, cohesion: f32 }
// NEEDS STATE: diplomacy: DiplomacyState on WorldState
//   diplomacy.agreements: Vec<(u32, u32, DiplomaticAgreement)>
//   DiplomaticAgreement::MilitaryAlliance, TradeAgreement { gold_per_tick }
//   diplomacy.relations: Vec<Vec<i32>>, guild_faction_id: u32

// NEEDS DELTA: FormBloc { id: u32, name: String, members: Vec<u32>, leader: u32,
//              strength: f32, cohesion: f32 }
// NEEDS DELTA: AdjustBlocCohesion { bloc_id: u32, delta: f32 }
// NEEDS DELTA: SetBlocStrength { bloc_id: u32, value: f32 }
// NEEDS DELTA: DissolveBloc { bloc_id: u32 }
// NEEDS DELTA: AddBlocMember { bloc_id: u32, faction_id: u32 }
// NEEDS DELTA: DeclareWar { attacker_id: u32, defender_id: u32 }

/// Cadence: every 17 ticks.
const BLOC_INTERVAL: u64 = 17;

/// Maximum number of active blocs allowed at any time.
const MAX_BLOCS: usize = 2;

/// Minimum mutual relation to qualify for bloc formation.
const FORMATION_RELATION_THRESHOLD: f32 = 60.0;

/// Chance per eligible pair to form a bloc each interval.
const FORMATION_CHANCE: f32 = 0.10;

/// Trade income bonus multiplier for intra-bloc trade agreements.
const BLOC_TRADE_BONUS: f32 = 0.10;

/// Cohesion decay per system tick.
const COHESION_DECAY: f32 = 1.0;

/// Cohesion gained when bloc members fight together against a common enemy.
const COHESION_FIGHT_BONUS: f32 = 5.0;

/// Cohesion lost when member interests diverge (internal wars).
const COHESION_DIVERGE_PENALTY: f32 = 10.0;

/// Cohesion threshold below which the bloc dissolves.
const DISSOLUTION_THRESHOLD: f32 = 20.0;

/// Deterministic hash for pseudo-random decisions.
#[inline]
fn deterministic_roll(tick: u64, a: u32, b: u32, salt: u32) -> f32 {
    let mut h = tick
        .wrapping_mul(6364136223846793005)
        .wrapping_add(a as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(b as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(salt as u64);
    h = h
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (h >> 33) as f32 / (1u64 << 31) as f32
}

pub fn compute_alliance_blocs(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % BLOC_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Once state.factions, state.alliance_blocs, state.diplomacy exist, enable this.

    /*
    // --- Phase 1: Try to form new blocs ---
    compute_form_blocs(state, out);

    // --- Phase 2: Update existing blocs ---
    compute_existing_blocs(state, out);
    */
}

/*
// ---------------------------------------------------------------------------
// Bloc formation
// ---------------------------------------------------------------------------

fn compute_form_blocs(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.alliance_blocs.len() >= MAX_BLOCS {
        return;
    }

    let guild_fid = state.diplomacy.guild_faction_id;

    let already_in_bloc = |fid: u32| -> bool {
        state
            .alliance_blocs
            .iter()
            .any(|b| b.member_factions.contains(&fid))
    };

    for fi in &state.factions {
        if already_in_bloc(fi.id) {
            continue;
        }

        for fj in &state.factions {
            if fj.id <= fi.id {
                continue;
            }
            if already_in_bloc(fj.id) {
                continue;
            }
            if state.alliance_blocs.len() >= MAX_BLOCS {
                return;
            }

            let relation = get_relation(state, fi.id, fj.id, guild_fid);
            if relation < FORMATION_RELATION_THRESHOLD {
                continue;
            }

            // Must have a military alliance agreement.
            let has_alliance = state.diplomacy.agreements.iter().any(|(a, b, ag)| {
                ((*a == fi.id && *b == fj.id) || (*a == fj.id && *b == fi.id))
                    && matches!(ag, DiplomaticAgreement::MilitaryAlliance { .. })
            });
            if !has_alliance {
                continue;
            }

            let roll = deterministic_roll(state.tick, fi.id, fj.id, 0);
            if roll >= FORMATION_CHANCE {
                continue;
            }

            let leader = if fi.military_strength >= fj.military_strength {
                fi.id
            } else {
                fj.id
            };
            let strength = fi.military_strength + fj.military_strength;

            // NOTE: bloc_id assignment should be handled by apply phase or
            // use a deterministic ID derived from tick + members.
            let bloc_id = state.tick as u32 ^ fi.id ^ fj.id;

            let name = format!("{}-{} Pact", fi.name, fj.name);

            out.push(WorldDelta::FormBloc {
                id: bloc_id,
                name,
                members: vec![fi.id, fj.id],
                leader,
                strength,
                cohesion: 50.0,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tick existing blocs
// ---------------------------------------------------------------------------

fn compute_existing_blocs(state: &WorldState, out: &mut Vec<WorldDelta>) {
    let guild_fid = state.diplomacy.guild_faction_id;

    for bloc in &state.alliance_blocs {
        let members = &bloc.member_factions;
        let mut cohesion_delta = -COHESION_DECAY;

        // --- Check for internal wars (interests diverge) ---
        let mut internal_war = false;
        for &ma in members {
            let faction_a = match state.factions.iter().find(|f| f.id == ma) {
                Some(f) => f,
                None => continue,
            };
            for &mb in members {
                if ma >= mb {
                    continue;
                }
                if faction_a.at_war_with.contains(&mb) {
                    internal_war = true;
                    break;
                }
            }
            if internal_war {
                break;
            }
        }

        if internal_war {
            cohesion_delta -= COHESION_DIVERGE_PENALTY;
        }

        // --- Check for shared enemies (fighting together) ---
        if members.len() >= 2 {
            let first_faction = state.factions.iter().find(|f| f.id == members[0]);
            if let Some(first) = first_faction {
                for &enemy in &first.at_war_with {
                    let all_fighting = members[1..].iter().any(|&m| {
                        state
                            .factions
                            .iter()
                            .find(|f| f.id == m)
                            .map(|f| f.at_war_with.contains(&enemy))
                            .unwrap_or(false)
                    });
                    if all_fighting {
                        cohesion_delta += COHESION_FIGHT_BONUS;
                        break;
                    }
                }
            }
        }

        out.push(WorldDelta::AdjustBlocCohesion {
            bloc_id: bloc.id,
            delta: cohesion_delta,
        });

        // --- Update aggregate strength ---
        let strength: f32 = members
            .iter()
            .filter_map(|&m| state.factions.iter().find(|f| f.id == m))
            .map(|f| f.military_strength)
            .sum();
        out.push(WorldDelta::SetBlocStrength {
            bloc_id: bloc.id,
            value: strength,
        });

        // --- Coordinated attacks ---
        compute_coordinated_attack(state, bloc, members, out);

        // --- Shared defense ---
        compute_shared_defense(state, bloc, members, out);

        // --- Intra-bloc trade bonus ---
        // Map to TransferGold: trade bonus gold goes to guild treasury.
        compute_trade_bonus(state, members, guild_fid, out);

        // --- Guild interaction: guild may join bloc ---
        compute_guild_interaction(state, bloc, members, guild_fid, out);

        // --- Dissolution check ---
        let projected_cohesion = bloc.cohesion + cohesion_delta;
        if projected_cohesion < DISSOLUTION_THRESHOLD || internal_war {
            out.push(WorldDelta::DissolveBloc {
                bloc_id: bloc.id,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Coordinated attacks
// ---------------------------------------------------------------------------

fn compute_coordinated_attack(
    state: &WorldState,
    bloc: &AllianceBlocState,
    members: &[u32],
    out: &mut Vec<WorldDelta>,
) {
    // Find common enemies.
    let mut common_enemies: Vec<u32> = Vec::new();
    for &m in members {
        if let Some(faction) = state.factions.iter().find(|f| f.id == m) {
            for &enemy in &faction.at_war_with {
                if !members.contains(&enemy) && !common_enemies.contains(&enemy) {
                    common_enemies.push(enemy);
                }
            }
        }
    }

    if common_enemies.is_empty() {
        return;
    }

    let target = common_enemies[0];

    // Members not yet at war with this target join (30% chance).
    for &m in members {
        let already_at_war = state
            .factions
            .iter()
            .find(|f| f.id == m)
            .map(|f| f.at_war_with.contains(&target))
            .unwrap_or(true);
        if already_at_war {
            continue;
        }

        let roll = deterministic_roll(state.tick, m, target, 100);
        if roll < 0.30 {
            out.push(WorldDelta::DeclareWar {
                attacker_id: m,
                defender_id: target,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Shared defense
// ---------------------------------------------------------------------------

fn compute_shared_defense(
    state: &WorldState,
    bloc: &AllianceBlocState,
    members: &[u32],
    out: &mut Vec<WorldDelta>,
) {
    let mut attackers: Vec<u32> = Vec::new();
    for &m in members {
        if let Some(faction) = state.factions.iter().find(|f| f.id == m) {
            for &enemy in &faction.at_war_with {
                if !members.contains(&enemy) && !attackers.contains(&enemy) {
                    attackers.push(enemy);
                }
            }
        }
    }

    for &attacker in &attackers {
        for &defender in members {
            let already_at_war = state
                .factions
                .iter()
                .find(|f| f.id == defender)
                .map(|f| f.at_war_with.contains(&attacker))
                .unwrap_or(true);
            if already_at_war {
                continue;
            }

            let roll = deterministic_roll(state.tick, defender, attacker, 200);
            if roll < 0.40 {
                out.push(WorldDelta::DeclareWar {
                    attacker_id: defender,
                    defender_id: attacker,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intra-bloc trade bonus
// ---------------------------------------------------------------------------

fn compute_trade_bonus(
    state: &WorldState,
    members: &[u32],
    guild_fid: u32,
    out: &mut Vec<WorldDelta>,
) {
    let mut bonus = 0.0f32;

    for (fa, fb, ag) in &state.diplomacy.agreements {
        if let DiplomaticAgreement::TradeAgreement { gold_per_tick, .. } = ag {
            if members.contains(fa) && members.contains(fb) {
                if *fa == guild_fid || *fb == guild_fid {
                    bonus += gold_per_tick * BLOC_TRADE_BONUS;
                }
            }
        }
    }

    if bonus > 0.0 {
        // Map trade bonus to guild treasury via UpdateTreasury.
        // Use guild_fid as the settlement proxy or a dedicated guild entity.
        out.push(WorldDelta::UpdateTreasury {
            location_id: guild_fid,
            delta: bonus,
        });
    }
}

// ---------------------------------------------------------------------------
// Guild interaction
// ---------------------------------------------------------------------------

fn compute_guild_interaction(
    state: &WorldState,
    bloc: &AllianceBlocState,
    members: &[u32],
    guild_fid: u32,
    out: &mut Vec<WorldDelta>,
) {
    // Guild already in this bloc.
    if members.contains(&guild_fid) {
        return;
    }

    // Guild already in another bloc.
    let guild_in_any = state
        .alliance_blocs
        .iter()
        .any(|b| b.member_factions.contains(&guild_fid));
    if guild_in_any {
        return;
    }

    // Check if any member has high enough relation with guild.
    let eligible = members.iter().any(|&m| {
        state
            .factions
            .iter()
            .find(|f| f.id == m)
            .map(|f| f.relationship_to_guild > FORMATION_RELATION_THRESHOLD)
            .unwrap_or(false)
    });
    if !eligible {
        return;
    }

    // 5% chance per interval.
    let roll = deterministic_roll(state.tick, guild_fid, bloc.id, 300);
    if roll >= 0.05 {
        return;
    }

    out.push(WorldDelta::AddBlocMember {
        bloc_id: bloc.id,
        faction_id: guild_fid,
    });
    out.push(WorldDelta::AdjustBlocCohesion {
        bloc_id: bloc.id,
        delta: 10.0,
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_relation(state: &WorldState, fi: u32, fj: u32, guild_fid: u32) -> f32 {
    if fi == guild_fid {
        state
            .factions
            .iter()
            .find(|f| f.id == fj)
            .map(|f| f.relationship_to_guild)
            .unwrap_or(0.0)
    } else if fj == guild_fid {
        state
            .factions
            .iter()
            .find(|f| f.id == fi)
            .map(|f| f.relationship_to_guild)
            .unwrap_or(0.0)
    } else if (fi as usize) < state.diplomacy.relations.len()
        && (fj as usize) < state.diplomacy.relations[fi as usize].len()
    {
        state.diplomacy.relations[fi as usize][fj as usize] as f32
    } else {
        0.0
    }
}
*/
