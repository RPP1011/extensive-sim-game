//! Alliance bloc system — factions formalize power blocs that coordinate
//! military, economic, and diplomatic actions as a unified force.
//!
//! Fires every 500 ticks. Manages bloc formation, cohesion drift, coordinated
//! attacks, shared defense, trade benefits, and dissolution.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: every 500 ticks.
const BLOC_INTERVAL: u64 = 17;

/// Maximum number of active blocs allowed at any time.
const MAX_BLOCS: usize = 2;

/// Minimum mutual relation to qualify for bloc formation.
const FORMATION_RELATION_THRESHOLD: f32 = 60.0;

/// Chance per eligible pair to form a bloc each interval.
const FORMATION_CHANCE: f32 = 0.10;

/// Trade income bonus multiplier for intra-bloc trade agreements.
const BLOC_TRADE_BONUS: f32 = 0.10;

/// Cohesion decay per tick of the system (every 500 game ticks).
const COHESION_DECAY: f32 = 1.0;

/// Cohesion gained when bloc members fight together against a common enemy.
const COHESION_FIGHT_BONUS: f32 = 5.0;

/// Cohesion lost when member interests diverge (e.g. members at war with each other).
const COHESION_DIVERGE_PENALTY: f32 = 10.0;

/// Cohesion threshold below which the bloc dissolves.
const DISSOLUTION_THRESHOLD: f32 = 20.0;

pub fn tick_alliance_blocs(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % BLOC_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // 1. Try to form new blocs
    try_form_blocs(state, events);

    // 2. Update existing blocs: cohesion, coordinated attacks, trade, dissolution
    tick_existing_blocs(state, events);
}

// ---------------------------------------------------------------------------
// Bloc formation
// ---------------------------------------------------------------------------

fn try_form_blocs(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    if state.alliance_blocs.len() >= MAX_BLOCS {
        return;
    }

    let n = state.factions.len();
    let guild_fid = state.diplomacy.guild_faction_id;

    // Find pairs with mutual alliance (relation > 60) not already in a bloc.
    let already_in_bloc = |fid: usize, blocs: &[AllianceBloc]| -> bool {
        blocs.iter().any(|b| b.member_factions.contains(&fid))
    };

    for fi in 0..n {
        if already_in_bloc(fi, &state.alliance_blocs) {
            continue;
        }

        for fj in (fi + 1)..n {
            if already_in_bloc(fj, &state.alliance_blocs) {
                continue;
            }
            if state.alliance_blocs.len() >= MAX_BLOCS {
                return;
            }

            let relation = get_relation(state, fi, fj, guild_fid);
            if relation < FORMATION_RELATION_THRESHOLD {
                continue;
            }

            // Must have a military alliance agreement
            let has_alliance = state.diplomacy.agreements.iter().any(|(a, b, ag)| {
                ((*a == fi && *b == fj) || (*a == fj && *b == fi))
                    && matches!(ag, DiplomaticAgreement::MilitaryAlliance { .. })
            });
            if !has_alliance {
                continue;
            }

            let roll = lcg_f32(&mut state.rng);
            if roll >= FORMATION_CHANCE {
                continue;
            }

            // Form the bloc — leader is the faction with greater military strength
            let leader = if state.factions[fi].military_strength
                >= state.factions[fj].military_strength
            {
                fi
            } else {
                fj
            };

            let strength = state.factions[fi].military_strength
                + state.factions[fj].military_strength;

            let bloc_id = state.next_bloc_id;
            state.next_bloc_id += 1;

            let name = format!(
                "{}-{} Pact",
                state.factions[fi].name, state.factions[fj].name
            );

            let bloc = AllianceBloc {
                id: bloc_id,
                name: name.clone(),
                member_factions: vec![fi, fj],
                leader_faction_id: leader,
                formed_tick: state.tick,
                strength,
                cohesion: 50.0,
            };

            events.push(WorldEvent::BlocFormed {
                bloc_id,
                name: name.clone(),
                member_factions: bloc.member_factions.clone(),
                leader_faction_id: leader,
            });

            state.alliance_blocs.push(bloc);
        }
    }
}

// ---------------------------------------------------------------------------
// Tick existing blocs
// ---------------------------------------------------------------------------

fn tick_existing_blocs(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let guild_fid = state.diplomacy.guild_faction_id;

    // We process blocs by index; collect dissolution indices to remove afterwards.
    let mut to_dissolve: Vec<usize> = Vec::new();

    for bi in 0..state.alliance_blocs.len() {
        let old_cohesion = state.alliance_blocs[bi].cohesion;

        // --- Cohesion decay ---
        state.alliance_blocs[bi].cohesion -= COHESION_DECAY;

        // --- Check for internal wars (interests diverge) ---
        let members = state.alliance_blocs[bi].member_factions.clone();
        let mut internal_war = false;
        for &ma in &members {
            for &mb in &members {
                if ma >= mb {
                    continue;
                }
                if state.factions[ma].at_war_with.contains(&mb) {
                    internal_war = true;
                    break;
                }
            }
            if internal_war {
                break;
            }
        }

        if internal_war {
            state.alliance_blocs[bi].cohesion -= COHESION_DIVERGE_PENALTY;
        }

        // --- Check for shared enemies (fighting together) ---
        let mut shared_fighting = false;
        if members.len() >= 2 {
            // Collect all enemies of the first member
            let first_wars: Vec<usize> = state.factions[members[0]].at_war_with.clone();
            for &enemy in &first_wars {
                // If any other member is also at war with this enemy
                let all_fighting = members[1..]
                    .iter()
                    .any(|&m| state.factions[m].at_war_with.contains(&enemy));
                if all_fighting {
                    shared_fighting = true;
                    break;
                }
            }
        }

        if shared_fighting {
            state.alliance_blocs[bi].cohesion += COHESION_FIGHT_BONUS;
        }

        // Clamp cohesion to [0, 100]
        state.alliance_blocs[bi].cohesion = state.alliance_blocs[bi].cohesion.clamp(0.0, 100.0);

        let new_cohesion = state.alliance_blocs[bi].cohesion;

        // Emit cohesion change event if significant (>= 1.0 delta)
        if (new_cohesion - old_cohesion).abs() >= 1.0 {
            let reason = if internal_war {
                "internal conflict".to_string()
            } else if shared_fighting {
                "fighting together".to_string()
            } else {
                "natural drift".to_string()
            };
            events.push(WorldEvent::BlocCohesionChanged {
                bloc_id: state.alliance_blocs[bi].id,
                old_cohesion,
                new_cohesion,
                reason,
            });
        }

        // --- Update aggregate strength ---
        let strength: f32 = members
            .iter()
            .map(|&m| state.factions[m].military_strength)
            .sum();
        state.alliance_blocs[bi].strength = strength;

        // --- Coordinated attacks: bloc attacks a shared enemy ---
        coordinated_attack(state, bi, &members, events);

        // --- Shared defense: if any member is attacked, others join ---
        shared_defense(state, bi, &members, events);

        // --- Trade benefits: boost gold for intra-bloc trade ---
        intra_bloc_trade_bonus(state, &members, guild_fid);

        // --- Guild interaction: guild faction may join bloc ---
        guild_bloc_interaction(state, bi, &members, guild_fid, events);

        // --- Dissolution check ---
        if state.alliance_blocs[bi].cohesion < DISSOLUTION_THRESHOLD || internal_war {
            to_dissolve.push(bi);
        }
    }

    // Remove dissolved blocs in reverse order
    to_dissolve.sort_unstable();
    to_dissolve.reverse();
    for bi in to_dissolve {
        let bloc = state.alliance_blocs.remove(bi);
        let reason = if bloc.cohesion < DISSOLUTION_THRESHOLD {
            format!("Cohesion dropped to {:.0}", bloc.cohesion)
        } else {
            "Internal war between members".to_string()
        };
        events.push(WorldEvent::BlocDissolved {
            bloc_id: bloc.id,
            name: bloc.name,
            reason,
        });
    }
}

// ---------------------------------------------------------------------------
// Coordinated attacks
// ---------------------------------------------------------------------------

/// Bloc members coordinate to attack the same enemy faction.
fn coordinated_attack(
    state: &mut CampaignState,
    bloc_idx: usize,
    members: &[usize],
    events: &mut Vec<WorldEvent>,
) {
    // Find a common enemy that any member is at war with
    let mut common_enemies: Vec<usize> = Vec::new();
    for &m in members {
        for &enemy in &state.factions[m].at_war_with {
            if !members.contains(&enemy) && !common_enemies.contains(&enemy) {
                common_enemies.push(enemy);
            }
        }
    }

    if common_enemies.is_empty() {
        return;
    }

    // Pick the first common enemy deterministically
    let target = common_enemies[0];

    // Members not yet at war with this target join the attack (30% chance)
    for &m in members {
        if state.factions[m].at_war_with.contains(&target) {
            continue;
        }
        let roll = lcg_f32(&mut state.rng);
        if roll < 0.30 {
            state.factions[m].at_war_with.push(target);
            if !state.factions[target].at_war_with.contains(&m) {
                state.factions[target].at_war_with.push(m);
            }
            let bloc_id = state.alliance_blocs[bloc_idx].id;
            events.push(WorldEvent::BlocAttack {
                bloc_id,
                target_faction: target,
                description: format!(
                    "{} joins bloc war against {}",
                    state.factions[m].name, state.factions[target].name
                ),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Shared defense
// ---------------------------------------------------------------------------

/// Attack on one member triggers defense from all others.
fn shared_defense(
    state: &mut CampaignState,
    bloc_idx: usize,
    members: &[usize],
    events: &mut Vec<WorldEvent>,
) {
    // Collect all attackers of any member
    let mut attackers: Vec<usize> = Vec::new();
    for &m in members {
        for &enemy in &state.factions[m].at_war_with {
            if !members.contains(&enemy) && !attackers.contains(&enemy) {
                attackers.push(enemy);
            }
        }
    }

    for &attacker in &attackers {
        for &defender in members {
            if state.factions[defender].at_war_with.contains(&attacker) {
                continue; // already fighting
            }
            // 40% chance to join defense each interval
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.40 {
                state.factions[defender].at_war_with.push(attacker);
                if !state.factions[attacker].at_war_with.contains(&defender) {
                    state.factions[attacker].at_war_with.push(defender);
                }
                let bloc_id = state.alliance_blocs[bloc_idx].id;
                events.push(WorldEvent::BlocAttack {
                    bloc_id,
                    target_faction: attacker,
                    description: format!(
                        "{} defends bloc ally against {}",
                        state.factions[defender].name, state.factions[attacker].name
                    ),
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Intra-bloc trade bonus
// ---------------------------------------------------------------------------

/// +10% gold on trade agreements between bloc members.
fn intra_bloc_trade_bonus(
    state: &mut CampaignState,
    members: &[usize],
    guild_fid: usize,
) {
    let mut bonus = 0.0f32;

    for (fa, fb, ag) in &state.diplomacy.agreements {
        if let DiplomaticAgreement::TradeAgreement { gold_per_tick, .. } = ag {
            if members.contains(fa) && members.contains(fb) {
                // Only credit to guild if guild is in this bloc
                if *fa == guild_fid || *fb == guild_fid {
                    bonus += gold_per_tick * BLOC_TRADE_BONUS;
                }
            }
        }
    }

    if bonus > 0.0 {
        state.guild.gold += bonus;
    }
}

// ---------------------------------------------------------------------------
// Guild interaction
// ---------------------------------------------------------------------------

/// If the guild faction is not in a bloc, a bloc member may propose membership.
fn guild_bloc_interaction(
    state: &mut CampaignState,
    bloc_idx: usize,
    members: &[usize],
    guild_fid: usize,
    events: &mut Vec<WorldEvent>,
) {
    // Guild already in this bloc
    if members.contains(&guild_fid) {
        return;
    }

    // Guild already in another bloc
    let guild_in_any = state
        .alliance_blocs
        .iter()
        .any(|b| b.member_factions.contains(&guild_fid));
    if guild_in_any {
        return;
    }

    // Check if any member has high enough relation with guild
    let mut eligible = false;
    for &m in members {
        let relation = state.factions[m].relationship_to_guild;
        if relation > FORMATION_RELATION_THRESHOLD {
            eligible = true;
            break;
        }
    }

    if !eligible {
        return;
    }

    // 5% chance per interval
    let roll = lcg_f32(&mut state.rng);
    if roll >= 0.05 {
        return;
    }

    // Guild joins the bloc
    state.alliance_blocs[bloc_idx]
        .member_factions
        .push(guild_fid);

    let bloc_name = state.alliance_blocs[bloc_idx].name.clone();
    let bloc_id = state.alliance_blocs[bloc_idx].id;

    // Cohesion boost from new member
    state.alliance_blocs[bloc_idx].cohesion =
        (state.alliance_blocs[bloc_idx].cohesion + 10.0).min(100.0);

    events.push(WorldEvent::GuildJoinedBloc {
        bloc_id,
        bloc_name,
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the relation between two factions, handling the guild specially.
fn get_relation(state: &CampaignState, fi: usize, fj: usize, guild_fid: usize) -> f32 {
    if fi == guild_fid {
        state.factions[fj].relationship_to_guild
    } else if fj == guild_fid {
        state.factions[fi].relationship_to_guild
    } else if fi < state.diplomacy.relations.len()
        && fj < state.diplomacy.relations[fi].len()
    {
        state.diplomacy.relations[fi][fj] as f32
    } else {
        0.0
    }
}
