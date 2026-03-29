//! Mercenary contract system — every 200 ticks.
//!
//! Guilds can hire temporary mercenary companies for specific campaigns.
//! Mercenaries rotate on the market every 1000 ticks, drain gold each tick
//! they're hired, and have loyalty mechanics that can cause desertion or
//! betrayal.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Tick the mercenary contract system every 200 ticks.
pub fn tick_mercenaries(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 7 != 0 || state.tick == 0 {
        return;
    }

    // --- Rotate available mercenary pool every 1000 ticks ---
    if state.tick % 33 == 0 {
        generate_available_mercenaries(state);
    }

    // --- Process hired mercenaries ---
    let mut deserted = Vec::new();
    let mut betrayed = Vec::new();
    let mut expired = Vec::new();

    let guild_rep_tier = (state.guild.reputation / 25.0).floor(); // 0-4

    for merc in &mut state.hired_mercenaries {
        // Drain gold per tick interval (cost_per_tick * 200 ticks since last check)
        let drain = merc.cost_per_tick * 200.0;

        if state.guild.gold >= drain {
            state.guild.gold -= drain;
            // Payment consistency: loyalty increases
            merc.loyalty = (merc.loyalty + 1.0).min(100.0);
        } else {
            // Can't pay — loyalty drops sharply
            state.guild.gold = 0.0;
            merc.loyalty = (merc.loyalty - 10.0).max(0.0);

            // 10% chance mercenaries turn hostile when unpaid
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.10 {
                betrayed.push(merc.id);
                continue;
            }

            // If loyalty bottomed out, they leave
            if merc.loyalty <= 0.0 {
                deserted.push(merc.id);
                continue;
            }
        }

        // Guild reputation loyalty bonus
        merc.loyalty = (merc.loyalty + 0.5 * guild_rep_tier).min(100.0);

        // Check contract expiration
        if state.tick >= merc.hired_tick + merc.contract_duration {
            expired.push(merc.id);
            continue;
        }

        // Low loyalty desertion check
        if merc.loyalty < 20.0 {
            let roll = lcg_f32(&mut state.rng);
            if roll < 0.05 {
                // 5% chance to switch sides
                betrayed.push(merc.id);
                continue;
            }
        }
    }

    // Process betrayals — mercenaries turn hostile
    for id in &betrayed {
        if let Some(merc) = state.hired_mercenaries.iter().find(|m| m.id == *id) {
            // Damage a random guild region's control as "betrayal"
            let strength = merc.strength;
            if let Some(region) = state
                .overworld
                .regions
                .iter_mut()
                .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
                .min_by(|a, b| a.control.partial_cmp(&b.control).unwrap_or(std::cmp::Ordering::Equal))
            {
                region.control = (region.control - strength * 5.0).max(0.0);
            }
            events.push(WorldEvent::MercenaryBetrayedGuild {
                mercenary_id: *id,
                name: merc.name.clone(),
                strength,
            });
        }
        // Unassign from party
        unassign_mercenary(state, *id);
    }
    state.hired_mercenaries.retain(|m| !betrayed.contains(&m.id));

    // Process desertions
    for id in &deserted {
        if let Some(merc) = state.hired_mercenaries.iter().find(|m| m.id == *id) {
            events.push(WorldEvent::MercenaryDeserted {
                mercenary_id: *id,
                name: merc.name.clone(),
            });
        }
        unassign_mercenary(state, *id);
    }
    state.hired_mercenaries.retain(|m| !deserted.contains(&m.id));

    // Process expirations
    for id in &expired {
        if let Some(merc) = state.hired_mercenaries.iter().find(|m| m.id == *id) {
            events.push(WorldEvent::MercenaryContractExpired {
                mercenary_id: *id,
                name: merc.name.clone(),
            });
        }
        unassign_mercenary(state, *id);
    }
    state.hired_mercenaries.retain(|m| !expired.contains(&m.id));
}

/// Remove mercenary assignment from its party.
fn unassign_mercenary(state: &mut CampaignState, mercenary_id: u32) {
    if let Some(merc) = state.hired_mercenaries.iter_mut().find(|m| m.id == mercenary_id) {
        merc.assigned_party_id = None;
    }
}

/// Generate 1-3 available mercenary companies.
fn generate_available_mercenaries(state: &mut CampaignState) {
    state.available_mercenaries.clear();

    let count = 1 + (lcg_next(&mut state.rng) % 3) as usize;

    let company_names = [
        "Iron Wolves",
        "Crimson Lance",
        "Storm Ravens",
        "Black Shields",
        "Silver Talons",
        "Ember Guard",
        "Frost Fangs",
        "Shadow Company",
        "Thunder Hammers",
        "Bronze Eagles",
        "Night Stalkers",
        "Steel Serpents",
    ];

    let specialties = [
        MercenarySpecialty::HeavyInfantry,
        MercenarySpecialty::Scouts,
        MercenarySpecialty::Archers,
        MercenarySpecialty::Siege,
        MercenarySpecialty::Cavalry,
        MercenarySpecialty::Assassins,
    ];

    for i in 0..count {
        let name_idx = (lcg_next(&mut state.rng) as usize) % company_names.len();
        let spec_idx = (lcg_next(&mut state.rng) as usize) % specialties.len();
        let strength = 5.0 + lcg_f32(&mut state.rng) * 15.0; // 5-20
        let cost_per_tick = strength * 0.05; // proportional to strength

        let id = state.tick as u32 * 100 + i as u32;

        state.available_mercenaries.push(MercenaryCompany {
            id,
            name: company_names[name_idx].into(),
            strength,
            cost_per_tick,
            loyalty: 50.0, // start neutral
            specialty: specialties[spec_idx],
            hired_tick: 0,
            contract_duration: 3000, // ~5 minutes of game time
            assigned_party_id: None,
        });
    }
}

/// Apply combat losses to mercenary loyalty.
/// Called from the battle system when a battle involving a mercenary's party ends.
pub fn apply_combat_losses(state: &mut CampaignState, party_id: u32) {
    for merc in &mut state.hired_mercenaries {
        if merc.assigned_party_id == Some(party_id) {
            merc.loyalty = (merc.loyalty - 5.0).max(0.0);
        }
    }
}

/// Calculate total mercenary strength bonus for a party.
pub fn mercenary_strength_for_party(state: &CampaignState, party_id: u32) -> f32 {
    state
        .hired_mercenaries
        .iter()
        .filter(|m| m.assigned_party_id == Some(party_id))
        .map(|m| m.strength)
        .sum()
}
