//! Adventurer grudges and vendettas system — every 300 ticks.
//!
//! Adventurers develop deep grudges against factions, specific nemesis enemies,
//! or regions where traumatic events occurred. Grudges drive personal narrative
//! arcs: bonus combat power against grudge targets, stress in traumatic regions,
//! volunteering for revenge quests, and reckless behavior at high intensity.
//!
//! Grudge formation triggers:
//! - Ally killed by faction → Faction grudge (intensity 70)
//! - Near-death experience (injury ≥ 80) → Region grudge (intensity 40)
//! - Nemesis encounter (party fought a nemesis) → Nemesis grudge (intensity 80)
//! - Betrayal / broken oath → Faction grudge (intensity 90)
//!
//! Resolution: defeating the grudge target grants +20 morale and a
//! "vengeance_fulfilled" history tag. Grudges decay -1 intensity per 500 ticks
//! if no reinforcing events occur. Unresolved grudges older than 5000 ticks
//! impose -5 morale (bitterness).

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{
    lcg_f32, AdventurerStatus, CampaignState, Grudge, GrudgeTarget,
};

/// Combat power bonus multiplier when fighting a grudge target.
const GRUDGE_COMBAT_BONUS: f32 = 0.20;

/// Stress added each tick cycle when an adventurer is in a grudge region.
const GRUDGE_REGION_STRESS: f32 = 3.0;

/// Intensity threshold above which an adventurer acts recklessly.
const RECKLESS_THRESHOLD: f32 = 70.0;

/// Ticks between grudge decay steps.
const DECAY_INTERVAL: u64 = 17;

/// Intensity lost per decay step.
const DECAY_AMOUNT: f32 = 1.0;

/// Ticks after which an unresolved grudge causes bitterness.
const BITTERNESS_THRESHOLD: u64 = 167;

/// Morale penalty for bitterness (applied once when crossing threshold).
const BITTERNESS_MORALE_PENALTY: f32 = 5.0;

/// Morale bonus when a grudge is resolved via vengeance.
const VENGEANCE_MORALE_BONUS: f32 = 20.0;

/// Maximum number of active grudges per adventurer.
const MAX_GRUDGES_PER_ADVENTURER: usize = 5;

// -------------------------------------------------------------------------
// Public API — called from other systems
// -------------------------------------------------------------------------

/// Called when an ally is killed by a specific faction. Party members of the
/// deceased gain a faction grudge.
pub fn on_ally_killed_by_faction(
    state: &mut CampaignState,
    dead_adventurer_id: u32,
    killer_faction_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    // Find the party the dead adventurer belonged to
    let party_id = state
        .adventurers
        .iter()
        .find(|a| a.id == dead_adventurer_id)
        .and_then(|a| a.party_id);

    let party_member_ids: Vec<u32> = if let Some(pid) = party_id {
        state
            .adventurers
            .iter()
            .filter(|a| {
                a.party_id == Some(pid)
                    && a.id != dead_adventurer_id
                    && a.status != AdventurerStatus::Dead
            })
            .map(|a| a.id)
            .collect()
    } else {
        Vec::new()
    };

    for adv_id in party_member_ids {
        add_grudge(
            state,
            adv_id,
            GrudgeTarget::Faction(killer_faction_id),
            70.0,
            format!("ally killed by faction {}", killer_faction_id),
            events,
        );
    }
}

/// Called when an adventurer has a near-death experience (injury >= 80) in a
/// region. Creates a region grudge.
pub fn on_near_death(
    state: &mut CampaignState,
    adventurer_id: u32,
    region_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    add_grudge(
        state,
        adventurer_id,
        GrudgeTarget::Region(region_id),
        40.0,
        format!("near-death in region {}", region_id),
        events,
    );
}

/// Called when an adventurer encounters a nemesis. Creates a nemesis grudge.
pub fn on_nemesis_encounter(
    state: &mut CampaignState,
    adventurer_id: u32,
    nemesis_id: u32,
    events: &mut Vec<WorldEvent>,
) {
    add_grudge(
        state,
        adventurer_id,
        GrudgeTarget::Nemesis(nemesis_id),
        80.0,
        format!("nemesis encounter (id {})", nemesis_id),
        events,
    );
}

/// Called on betrayal or broken oath by a faction.
pub fn on_betrayal(
    state: &mut CampaignState,
    adventurer_id: u32,
    faction_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    add_grudge(
        state,
        adventurer_id,
        GrudgeTarget::Faction(faction_id),
        90.0,
        format!("betrayal by faction {}", faction_id),
        events,
    );
}

/// Returns the combat power bonus multiplier for an adventurer against enemies
/// from a specific faction. Returns 0.0 if no relevant grudge.
pub fn grudge_combat_bonus(state: &CampaignState, adventurer_id: u32, enemy_faction_id: usize) -> f32 {
    let has_faction_grudge = state.grudges.iter().any(|g| {
        g.adventurer_id == adventurer_id
            && !g.resolved
            && matches!(g.target, GrudgeTarget::Faction(fid) if fid == enemy_faction_id)
    });
    if has_faction_grudge {
        return GRUDGE_COMBAT_BONUS;
    }

    // Also check nemesis grudges — if any active nemesis belongs to this faction
    let has_nemesis_grudge = state.grudges.iter().any(|g| {
        g.adventurer_id == adventurer_id
            && !g.resolved
            && matches!(g.target, GrudgeTarget::Nemesis(nid) if {
                state.nemeses.iter().any(|n| n.id == nid && n.faction_id == enemy_faction_id)
            })
    });
    if has_nemesis_grudge {
        GRUDGE_COMBAT_BONUS
    } else {
        0.0
    }
}

/// Returns true if the adventurer would volunteer for a quest targeting their
/// grudge faction or region.
pub fn would_volunteer(state: &CampaignState, adventurer_id: u32, target_faction: Option<usize>, target_region: Option<usize>) -> bool {
    state.grudges.iter().any(|g| {
        g.adventurer_id == adventurer_id && !g.resolved && g.intensity > 30.0 && {
            match &g.target {
                GrudgeTarget::Faction(fid) => target_faction == Some(*fid),
                GrudgeTarget::Region(rid) => target_region == Some(*rid),
                GrudgeTarget::Nemesis(nid) => {
                    // Volunteer if quest targets the nemesis's faction
                    state.nemeses.iter().any(|n| n.id == *nid && target_faction == Some(n.faction_id))
                }
            }
        }
    })
}

/// Returns true if the adventurer acts recklessly (ignoring retreat orders)
/// due to a high-intensity grudge against the given faction.
pub fn acts_recklessly(state: &CampaignState, adventurer_id: u32, enemy_faction_id: usize) -> bool {
    state.grudges.iter().any(|g| {
        g.adventurer_id == adventurer_id
            && !g.resolved
            && g.intensity > RECKLESS_THRESHOLD
            && matches!(&g.target, GrudgeTarget::Faction(fid) if *fid == enemy_faction_id)
    })
}

// -------------------------------------------------------------------------
// Main tick
// -------------------------------------------------------------------------

/// Main tick function. Called every tick; gates internally on tick % 300.
pub fn tick_grudges(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 10 != 0 {
        return;
    }

    let tick = state.tick;

    // --- 1. Detect new grudge triggers ---
    detect_near_death_grudges(state, events);
    detect_nemesis_grudges(state, events);

    // --- 2. Grudge effects: stress from being in grudge region ---
    apply_region_stress(state);

    // --- 3. Grudge decay ---
    if tick % DECAY_INTERVAL == 0 {
        decay_grudges(state, events);
    }

    // --- 4. Bitterness from unresolved grudges ---
    apply_bitterness(state, events);

    // --- 5. Check for vendetta fulfillment ---
    check_vendetta_fulfillment(state, events);
}

// -------------------------------------------------------------------------
// Internal helpers
// -------------------------------------------------------------------------

fn add_grudge(
    state: &mut CampaignState,
    adventurer_id: u32,
    target: GrudgeTarget,
    intensity: f32,
    cause: String,
    events: &mut Vec<WorldEvent>,
) {
    // Check per-adventurer cap
    let existing_count = state
        .grudges
        .iter()
        .filter(|g| g.adventurer_id == adventurer_id && !g.resolved)
        .count();
    if existing_count >= MAX_GRUDGES_PER_ADVENTURER {
        return;
    }

    // If an identical grudge (same adventurer + target) already exists, intensify it
    if let Some(existing) = state.grudges.iter_mut().find(|g| {
        g.adventurer_id == adventurer_id && !g.resolved && g.target == target
    }) {
        let old = existing.intensity;
        existing.intensity = (existing.intensity + intensity * 0.5).min(100.0);
        events.push(WorldEvent::GrudgeIntensified {
            adventurer_id,
            target: target.clone(),
            old_intensity: old,
            new_intensity: existing.intensity,
        });
        return;
    }

    let id = state.next_grudge_id;
    state.next_grudge_id += 1;

    let grudge = Grudge {
        id,
        adventurer_id,
        target: target.clone(),
        intensity,
        cause: cause.clone(),
        formed_tick: state.tick,
        resolved: false,
    };
    state.grudges.push(grudge);

    events.push(WorldEvent::GrudgeFormed {
        adventurer_id,
        target,
        intensity,
        cause,
    });
}

/// Scan adventurers for near-death conditions (injury >= 80) and form region
/// grudges if they are in the field.
fn detect_near_death_grudges(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect candidates: adventurers with injury >= 80 who are on mission or fighting
    let candidates: Vec<(u32, Option<u32>)> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.injury >= 80.0
                && matches!(
                    a.status,
                    AdventurerStatus::OnMission | AdventurerStatus::Fighting
                )
        })
        .map(|a| (a.id, a.party_id))
        .collect();

    for (adv_id, party_id) in candidates {
        // Don't form duplicate near-death grudges too often — 15% chance
        let roll = lcg_f32(&mut state.rng);
        if roll >= 0.15 {
            continue;
        }

        // Find region from party position
        let region_id = party_id
            .and_then(|pid| state.parties.iter().find(|p| p.id == pid))
            .and_then(|party| {
                state
                    .overworld
                    .regions
                    .iter()
                    .min_by_key(|r| {
                        let dx = party.position.0 - (r.id as f32 * 10.0);
                        let dy = party.position.1 - (r.id as f32 * 10.0);
                        ((dx * dx + dy * dy) * 100.0) as i64
                    })
                    .map(|r| r.id)
            });

        if let Some(rid) = region_id {
            add_grudge(
                state,
                adv_id,
                GrudgeTarget::Region(rid),
                40.0,
                format!("near-death in region {}", rid),
                events,
            );
        }
    }
}

/// Scan for adventurers that recently fought a nemesis and form nemesis grudges.
fn detect_nemesis_grudges(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Find active (undefeated) nemeses and check if any party is near them
    let nemesis_data: Vec<(u32, Option<usize>)> = state
        .nemeses
        .iter()
        .filter(|n| !n.defeated)
        .map(|n| (n.id, n.region_id))
        .collect();

    for (nem_id, nem_region) in nemesis_data {
        let nem_region = match nem_region {
            Some(r) => r,
            None => continue,
        };

        // Find adventurers in parties near the nemesis region
        let nearby_advs: Vec<u32> = state
            .parties
            .iter()
            .filter(|p| {
                matches!(
                    p.status,
                    super::super::state::PartyStatus::Fighting
                        | super::super::state::PartyStatus::OnMission
                )
            })
            .filter(|p| {
                // Check proximity to nemesis region
                state
                    .overworld
                    .regions
                    .iter()
                    .find(|r| r.id == nem_region)
                    .map(|_r| {
                        let dx = p.position.0 - (nem_region as f32 * 10.0);
                        let dy = p.position.1 - (nem_region as f32 * 10.0);
                        (dx * dx + dy * dy).sqrt() < 5.0
                    })
                    .unwrap_or(false)
            })
            .flat_map(|p| {
                state
                    .adventurers
                    .iter()
                    .filter(|a| {
                        a.party_id == Some(p.id) && a.status != AdventurerStatus::Dead
                    })
                    .map(|a| a.id)
                    .collect::<Vec<_>>()
            })
            .collect();

        for adv_id in nearby_advs {
            // Only 10% chance per tick cycle to form a grudge from proximity
            let roll = lcg_f32(&mut state.rng);
            if roll >= 0.10 {
                continue;
            }

            // Don't duplicate
            let already_has = state.grudges.iter().any(|g| {
                g.adventurer_id == adv_id
                    && !g.resolved
                    && matches!(g.target, GrudgeTarget::Nemesis(nid) if nid == nem_id)
            });
            if already_has {
                continue;
            }

            add_grudge(
                state,
                adv_id,
                GrudgeTarget::Nemesis(nem_id),
                80.0,
                format!("nemesis encounter (id {})", nem_id),
                events,
            );
        }
    }
}

/// Apply stress to adventurers who are in a region they hold a grudge against.
fn apply_region_stress(state: &mut CampaignState) {
    // Collect (adv_index, region_id) pairs for region grudges
    let grudge_regions: Vec<(u32, usize)> = state
        .grudges
        .iter()
        .filter(|g| !g.resolved)
        .filter_map(|g| match g.target {
            GrudgeTarget::Region(rid) => Some((g.adventurer_id, rid)),
            _ => None,
        })
        .collect();

    for (adv_id, grudge_region) in grudge_regions {
        // Find if adventurer is near the grudge region
        let adv = match state.adventurers.iter().find(|a| a.id == adv_id) {
            Some(a) => a,
            None => continue,
        };
        if adv.status == AdventurerStatus::Dead || adv.status == AdventurerStatus::Idle {
            continue;
        }
        let party_id = match adv.party_id {
            Some(pid) => pid,
            None => continue,
        };
        let party_pos = match state.parties.iter().find(|p| p.id == party_id) {
            Some(p) => p.position,
            None => continue,
        };

        // Check if party is near the grudge region
        let dx = party_pos.0 - (grudge_region as f32 * 10.0);
        let dy = party_pos.1 - (grudge_region as f32 * 10.0);
        if (dx * dx + dy * dy).sqrt() < 8.0 {
            if let Some(adv_mut) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
                adv_mut.stress = (adv_mut.stress + GRUDGE_REGION_STRESS).min(100.0);
            }
        }
    }
}

/// Decay grudge intensity over time.
fn decay_grudges(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    for grudge in &mut state.grudges {
        if grudge.resolved {
            continue;
        }

        grudge.intensity -= DECAY_AMOUNT;
        if grudge.intensity <= 0.0 {
            grudge.intensity = 0.0;
            grudge.resolved = true;
            events.push(WorldEvent::GrudgeFaded {
                adventurer_id: grudge.adventurer_id,
                target: grudge.target.clone(),
            });
        }
    }
}

/// Apply morale penalty for unresolved grudges older than the bitterness threshold.
fn apply_bitterness(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;

    // Collect adventurer ids that should receive bitterness penalty
    let bitter_advs: Vec<u32> = state
        .grudges
        .iter()
        .filter(|g| {
            !g.resolved
                && tick.saturating_sub(g.formed_tick) > BITTERNESS_THRESHOLD
                && g.intensity > 0.0
        })
        .map(|g| g.adventurer_id)
        .collect();

    // Deduplicate — only apply once per adventurer even with multiple bitter grudges
    let mut seen = std::collections::HashSet::new();
    for adv_id in bitter_advs {
        if !seen.insert(adv_id) {
            continue;
        }
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale - BITTERNESS_MORALE_PENALTY).max(0.0);
            }
        }
    }

    // No separate event for bitterness — it's a slow drain, not a discrete event
    let _ = events; // suppress unused warning
}

/// Check if any grudge targets have been defeated and resolve those grudges.
fn check_vendetta_fulfillment(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect resolutions to apply
    let mut resolutions: Vec<(usize, u32)> = Vec::new(); // (grudge_index, adventurer_id)

    for (idx, grudge) in state.grudges.iter().enumerate() {
        if grudge.resolved {
            continue;
        }

        let fulfilled = match &grudge.target {
            GrudgeTarget::Nemesis(nid) => {
                // Check if nemesis has been defeated
                state.nemeses.iter().any(|n| n.id == *nid && n.defeated)
            }
            GrudgeTarget::Faction(fid) => {
                // Faction grudge resolved if faction is destroyed (strength <= 0
                // or no controlled regions)
                state
                    .factions
                    .iter()
                    .find(|f| f.id == *fid)
                    .map(|f| f.military_strength <= 0.0)
                    .unwrap_or(true) // faction doesn't exist = resolved
            }
            GrudgeTarget::Region(_) => {
                // Region grudges are only resolved by decay or explicit quest
                // completion (handled via on_revenge_quest_completed)
                false
            }
        };

        if fulfilled {
            resolutions.push((idx, grudge.adventurer_id));
        }
    }

    // Apply resolutions
    for (grudge_idx, adv_id) in resolutions {
        let target = state.grudges[grudge_idx].target.clone();
        state.grudges[grudge_idx].resolved = true;

        // +20 morale
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale + VENGEANCE_MORALE_BONUS).min(100.0);
                adv.history_tags
                    .entry("vengeance_fulfilled".to_string())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }

        events.push(WorldEvent::VendettaFulfilled {
            adventurer_id: adv_id,
            target,
        });
    }
}

/// Explicitly resolve a region grudge when a revenge quest is completed.
pub fn on_revenge_quest_completed(
    state: &mut CampaignState,
    adventurer_id: u32,
    region_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    for grudge in &mut state.grudges {
        if grudge.adventurer_id == adventurer_id
            && !grudge.resolved
            && matches!(grudge.target, GrudgeTarget::Region(rid) if rid == region_id)
        {
            grudge.resolved = true;

            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adventurer_id) {
                if adv.status != AdventurerStatus::Dead {
                    adv.morale = (adv.morale + VENGEANCE_MORALE_BONUS).min(100.0);
                    adv.history_tags
                        .entry("vengeance_fulfilled".to_string())
                        .and_modify(|c| *c += 1)
                        .or_insert(1);
                }
            }

            events.push(WorldEvent::VendettaFulfilled {
                adventurer_id,
                target: GrudgeTarget::Region(region_id),
            });
        }
    }
}
