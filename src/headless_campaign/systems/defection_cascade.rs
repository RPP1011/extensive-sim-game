//! Defection cascade system — every 200 ticks.
//!
//! When a high-status adventurer defects from a faction, it triggers a cascade
//! through loyalty graphs affecting morale, bonds, and faction standing.
//! Cascades are depth-limited to 3 hops to prevent total faction collapse.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;
use crate::headless_campaign::systems::bonds::bond_strength;

/// Maximum cascade depth to prevent total faction collapse.
const MAX_CASCADE_DEPTH: u32 = 3;

/// Cadenced system — fires every 200 ticks.
pub fn tick_defection_cascade(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 200 != 0 || state.tick == 0 {
        return;
    }

    // Collect potential defectors: adventurers belonging to a faction (not the guild)
    // whose faction_relation < -30, loyalty < 20, and a "pull" faction exists with
    // relation > 40.
    let n_factions = state.factions.len();
    if n_factions < 2 {
        return;
    }

    // Gather candidates: (adventurer_id, from_faction, to_faction)
    let mut candidates: Vec<(u32, usize, usize)> = Vec::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let from_faction = match adv.faction_id {
            Some(fid) => fid,
            None => continue, // Guild adventurers don't defect via this system
        };
        if from_faction >= n_factions {
            continue;
        }

        // Check defection conditions on the adventurer
        if adv.loyalty >= 20.0 {
            continue;
        }

        // Check faction relation: the adventurer's faction must have relation < -30
        // We use the diplomacy matrix relations between factions.
        // The adventurer's own guild_relationship represents their personal relation.
        // For faction-level check, use the faction's relationship_to_guild as a proxy
        // for general dissatisfaction.
        let faction_relation = state.factions[from_faction].relationship_to_guild;
        if faction_relation >= -30.0 {
            continue;
        }

        // Find a "pull" faction with relation > 40 (from the diplomacy matrix)
        let mut best_pull: Option<usize> = None;
        for fi in 0..n_factions {
            if fi == from_faction {
                continue;
            }
            // Check faction-to-faction relation from the diplomacy matrix
            let pull_relation = if from_faction < state.diplomacy.relations.len()
                && fi < state.diplomacy.relations[from_faction].len()
            {
                state.diplomacy.relations[fi][from_faction] // how fi feels about from_faction's members
            } else {
                // Fallback: use the target faction's relationship_to_guild as proxy
                state.factions[fi].relationship_to_guild as i32
            };
            if pull_relation > 40 {
                best_pull = Some(fi);
                break;
            }
        }

        if let Some(to_faction) = best_pull {
            candidates.push((adv.id, from_faction, to_faction));
        }
    }

    // Process each candidate defection
    for (adv_id, from_faction, to_faction) in candidates {
        process_defection(state, events, adv_id, from_faction, to_faction);
    }
}

/// Check whether an adventurer qualifies as "high-status" (level > 5 or has legendary deeds).
fn is_high_status(adv: &Adventurer) -> bool {
    adv.level > 5 || !adv.deeds.is_empty()
}

/// Process a single defection and its potential cascade.
fn process_defection(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    defector_id: u32,
    from_faction: usize,
    to_faction: usize,
) {
    // Move the defector
    if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == defector_id) {
        adv.faction_id = Some(to_faction);
    }

    events.push(WorldEvent::DefectionOccurred {
        adventurer_id: defector_id,
        from_faction,
        to_faction,
    });

    // Check if this is a high-status defection that triggers a cascade
    let high_status = state
        .adventurers
        .iter()
        .find(|a| a.id == defector_id)
        .map(|a| is_high_status(a))
        .unwrap_or(false);

    let mut cascade_count = 0u32;

    if high_status {
        // Cascade: bonded allies (bond > 50) with loyalty < 40 also defect
        let cascade_ids = run_cascade(state, events, defector_id, from_faction, to_faction, 0);
        cascade_count = cascade_ids.len() as u32;

        if !cascade_ids.is_empty() {
            events.push(WorldEvent::DefectionCascade {
                trigger_id: defector_id,
                cascade_ids: cascade_ids.clone(),
                faction_id: from_faction,
            });
        }

        // Remaining adventurers in the losing faction lose 5-15 morale
        let morale_loss = 5.0 + lcg_f32(&mut state.rng) * 10.0;
        for adv in &mut state.adventurers {
            if adv.faction_id == Some(from_faction)
                && adv.status != AdventurerStatus::Dead
                && adv.id != defector_id
            {
                adv.morale = (adv.morale - morale_loss).max(0.0);
            }
        }

        // Faction standing with the losing faction drops by 10
        if from_faction < state.factions.len() {
            state.factions[from_faction].relationship_to_guild =
                (state.factions[from_faction].relationship_to_guild - 10.0).max(-100.0);
        }
    }

    // Record the defection event
    state.defection_events.push(DefectionEvent {
        adventurer_id: defector_id,
        from_faction: from_faction as u32,
        to_faction: to_faction as u32,
        cascade_count,
        tick: state.tick as u32,
    });
}

/// Recursively cascade defections through bonded allies.
/// Returns the IDs of all adventurers who defected in the cascade.
/// Depth-limited to 3 hops.
fn run_cascade(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    trigger_id: u32,
    from_faction: usize,
    to_faction: usize,
    depth: u32,
) -> Vec<u32> {
    if depth >= MAX_CASCADE_DEPTH {
        return Vec::new();
    }

    // Find bonded allies (bond > 50) in the same faction with loyalty < 40
    let cascade_candidates: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.id != trigger_id
                && a.faction_id == Some(from_faction)
                && a.status != AdventurerStatus::Dead
                && a.loyalty < 40.0
                && bond_strength(&state.adventurer_bonds, trigger_id, a.id) > 50.0
        })
        .map(|a| a.id)
        .collect();

    let mut all_cascaded = Vec::new();

    for cid in cascade_candidates {
        // Move the cascaded adventurer
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == cid) {
            adv.faction_id = Some(to_faction);
        }

        events.push(WorldEvent::DefectionOccurred {
            adventurer_id: cid,
            from_faction,
            to_faction,
        });

        all_cascaded.push(cid);

        // Recurse for further cascades (depth + 1)
        let sub_cascade = run_cascade(state, events, cid, from_faction, to_faction, depth + 1);
        all_cascaded.extend(sub_cascade);
    }

    all_cascaded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_status_level_threshold() {
        // Level 5 or below, no deeds => not high status
        let mut adv = Adventurer {
            id: 1,
            name: "Test".into(),
            archetype: "knight".into(),
            level: 5,
            deeds: Vec::new(),
            ..make_default_adventurer()
        };
        assert!(!is_high_status(&adv));

        // Level 6 => high status
        adv.level = 6;
        assert!(is_high_status(&adv));
    }

    #[test]
    fn high_status_legendary_deeds() {
        let mut adv = Adventurer {
            id: 1,
            name: "Test".into(),
            archetype: "knight".into(),
            level: 1,
            deeds: Vec::new(),
            ..make_default_adventurer()
        };
        assert!(!is_high_status(&adv));

        adv.deeds.push(LegendaryDeed {
            title: "Dragon Slayer".into(),
            earned_at_tick: 100,
            deed_type: DeedType::Slayer,
            bonus: DeedBonus::MoraleAura(5.0),
        });
        assert!(is_high_status(&adv));
    }

    #[test]
    fn cascade_depth_limit() {
        assert!(MAX_CASCADE_DEPTH == 3);
    }

    /// Helper: creates an adventurer with all required fields set to defaults.
    fn make_default_adventurer() -> Adventurer {
        Adventurer {
            id: 0,
            name: String::new(),
            archetype: String::new(),
            level: 1,
            xp: 0,
            stats: AdventurerStats::default(),
            equipment: Equipment::default(),
            traits: Vec::new(),
            status: AdventurerStatus::Idle,
            loyalty: 50.0,
            stress: 0.0,
            fatigue: 0.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 70.0,
            party_id: None,
            guild_relationship: 0.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),
            backstory: None,
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: Default::default(),
            mood_state: Default::default(),
            fears: Vec::new(),
            personal_goal: None,
            journal: Vec::new(),
            equipped_items: Vec::new(),
            nicknames: Vec::new(),
            secret_past: None,
            wounds: Vec::new(),
            potion_dependency: 0.0,
            withdrawal_severity: 0.0,
            ticks_since_last_potion: 0,
            total_potions_consumed: 0,
        }
    }
}
