//! Adventurer romance system — every 300 ticks.
//!
//! Adventurers can develop romantic relationships that create powerful bonds
//! but also vulnerabilities and drama. Romances progress through stages:
//! Attraction → Courting → Together → (optionally) Strained → BrokenUp.
//!
//! Benefits: morale boosts, combat bonuses, accelerated bond growth.
//! Risks: grief on partner death/injury, strained relations, breakup fallout.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    lcg_f32, AdventurerStatus, CampaignState, Romance, RomanceStage,
};
use crate::headless_campaign::systems::bonds::{bond_key, bond_strength};

/// Maximum number of active romances in the guild.
const MAX_ROMANCES: usize = 3;

/// Ticks of separation before a romance becomes strained.
const SEPARATION_STRAIN_TICKS: u64 = 2000;

/// Ticks after breakup during which both refuse to be in the same party.
const BREAKUP_COOLDOWN_TICKS: u64 = 1500;

/// Check if an adventurer is in an active (non-broken-up) romance.
pub fn active_romance_partner(state: &CampaignState, adv_id: u32) -> Option<u32> {
    state.romances.iter().find_map(|r| {
        if matches!(r.stage, RomanceStage::BrokenUp) {
            return None;
        }
        if r.adventurer_a == adv_id {
            Some(r.adventurer_b)
        } else if r.adventurer_b == adv_id {
            Some(r.adventurer_a)
        } else {
            None
        }
    })
}

/// Whether two adventurers refuse to be in the same party due to a recent breakup.
pub fn refuses_party_romance(state: &CampaignState, a: u32, b: u32) -> bool {
    let (lo, hi) = (a.min(b), a.max(b));
    state.romances.iter().any(|r| {
        r.adventurer_a == lo
            && r.adventurer_b == hi
            && matches!(r.stage, RomanceStage::BrokenUp)
            && state.tick.saturating_sub(r.started_tick) < BREAKUP_COOLDOWN_TICKS
    })
}

/// Morale bonus from being in the same party as a Together-stage partner.
/// Returns +15 if partner is in the given party member list, 0 otherwise.
pub fn romance_morale_bonus(state: &CampaignState, adv_id: u32, party_member_ids: &[u32]) -> f32 {
    if let Some(partner) = active_romance_partner(state, adv_id) {
        let rom = state.romances.iter().find(|r| {
            let (lo, hi) = (r.adventurer_a, r.adventurer_b);
            ((lo == adv_id && hi == partner) || (lo == partner && hi == adv_id))
                && matches!(r.stage, RomanceStage::Together)
        });
        if rom.is_some() && party_member_ids.contains(&partner) {
            return 15.0;
        }
    }
    0.0
}

/// Combat power multiplier from fighting alongside a Together-stage partner.
/// Returns 1.10 (10% bonus) if partner present, 1.0 otherwise.
pub fn romance_combat_multiplier(
    state: &CampaignState,
    adv_id: u32,
    party_member_ids: &[u32],
) -> f32 {
    if let Some(partner) = active_romance_partner(state, adv_id) {
        let rom = state.romances.iter().find(|r| {
            let (lo, hi) = (r.adventurer_a, r.adventurer_b);
            ((lo == adv_id && hi == partner) || (lo == partner && hi == adv_id))
                && matches!(r.stage, RomanceStage::Together)
        });
        if rom.is_some() && party_member_ids.contains(&partner) {
            return 1.10;
        }
    }
    1.0
}

/// Main tick function. Called every 300 ticks.
///
/// 1. Form new romances (bond > 50, same party, compatible morale, 5% chance)
/// 2. Progress stages: Attraction → Courting (bond > 60) → Together (bond > 75)
/// 3. Apply Together benefits: bond growth 2x
/// 4. Detect strain: rivalry with partner, long separation, jealousy
/// 5. Handle breakups: Strained → BrokenUp after 500 ticks
/// 6. Clean up stale romances (dead adventurers)
pub fn tick_romance(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 300 != 0 {
        return;
    }

    // Count active (non-broken-up) romances.
    let active_count = state
        .romances
        .iter()
        .filter(|r| !matches!(r.stage, RomanceStage::BrokenUp))
        .count();

    // --- 1. Romance formation ---
    if active_count < MAX_ROMANCES {
        try_form_romance(state, events);
    }

    // --- 2-5. Progress / strain / breakup existing romances ---
    tick_existing_romances(state, events);

    // --- 6. Clean up romances involving dead adventurers ---
    cleanup_dead_romances(state, events);
}

/// Try to form a new romance between compatible adventurers.
fn try_form_romance(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Build list of party groupings (active parties with their member IDs).
    let party_members: Vec<Vec<u32>> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                crate::headless_campaign::state::PartyStatus::OnMission
                    | crate::headless_campaign::state::PartyStatus::Fighting
                    | crate::headless_campaign::state::PartyStatus::Traveling
            )
        })
        .map(|p| p.member_ids.clone())
        .collect();

    // Already-romanced adventurer IDs (active romances only).
    let romanced: Vec<u32> = state
        .romances
        .iter()
        .filter(|r| !matches!(r.stage, RomanceStage::BrokenUp))
        .flat_map(|r| [r.adventurer_a, r.adventurer_b])
        .collect();

    // Search for candidate pairs.
    for members in &party_members {
        for (i, &a) in members.iter().enumerate() {
            for &b in &members[i + 1..] {
                // Skip if either already in an active romance.
                if romanced.contains(&a) || romanced.contains(&b) {
                    continue;
                }

                // Bond must be > 50.
                let bond = bond_strength(&state.adventurer_bonds, a, b);
                if bond <= 50.0 {
                    continue;
                }

                // Both must be alive and not dead.
                let a_alive = state
                    .adventurers
                    .iter()
                    .any(|adv| adv.id == a && adv.status != AdventurerStatus::Dead);
                let b_alive = state
                    .adventurers
                    .iter()
                    .any(|adv| adv.id == b && adv.status != AdventurerStatus::Dead);
                if !a_alive || !b_alive {
                    continue;
                }

                // Compatible morale: both > 40 (not demoralized).
                let a_morale = state
                    .adventurers
                    .iter()
                    .find(|adv| adv.id == a)
                    .map(|adv| adv.morale)
                    .unwrap_or(0.0);
                let b_morale = state
                    .adventurers
                    .iter()
                    .find(|adv| adv.id == b)
                    .map(|adv| adv.morale)
                    .unwrap_or(0.0);
                if a_morale <= 40.0 || b_morale <= 40.0 {
                    continue;
                }

                // No active rivalry between them.
                if super::rivalries::has_rivalry(state, a, b) {
                    continue;
                }

                // 5% chance.
                let roll = lcg_f32(&mut state.rng);
                if roll >= 0.05 {
                    continue;
                }

                // Form romance!
                let (lo, hi) = (a.min(b), a.max(b));
                state.romances.push(Romance {
                    adventurer_a: lo,
                    adventurer_b: hi,
                    stage: RomanceStage::Attraction,
                    strength: 10.0,
                    started_tick: state.tick,
                });

                events.push(WorldEvent::RomanceBegan {
                    adventurer_a: lo,
                    adventurer_b: hi,
                });

                // Only form one per tick.
                return;
            }
        }
    }
}

/// Progress, strain, or break up existing romances.
fn tick_existing_romances(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Build lookup: adventurer_id → party_id (if any).
    let adv_party: Vec<(u32, Option<u32>)> = state
        .adventurers
        .iter()
        .map(|a| (a.id, a.party_id))
        .collect();

    let tick = state.tick;
    let bonds = &state.adventurer_bonds;

    // Collect updates to apply after iteration.
    struct RomanceUpdate {
        idx: usize,
        new_stage: Option<RomanceStage>,
        strength_delta: f32,
        bond_boost: Option<(u32, u32)>,
    }

    let mut updates: Vec<RomanceUpdate> = Vec::new();

    for (idx, rom) in state.romances.iter().enumerate() {
        if matches!(rom.stage, RomanceStage::BrokenUp) {
            continue;
        }

        let a = rom.adventurer_a;
        let b = rom.adventurer_b;
        let bond = bond_strength(bonds, a, b);

        let a_party = adv_party.iter().find(|(id, _)| *id == a).and_then(|(_, p)| *p);
        let b_party = adv_party.iter().find(|(id, _)| *id == b).and_then(|(_, p)| *p);
        let same_party = a_party.is_some() && a_party == b_party;

        match rom.stage {
            RomanceStage::Attraction => {
                if bond > 60.0 {
                    updates.push(RomanceUpdate {
                        idx,
                        new_stage: Some(RomanceStage::Courting),
                        strength_delta: 5.0,
                        bond_boost: None,
                    });
                } else {
                    updates.push(RomanceUpdate {
                        idx,
                        new_stage: None,
                        strength_delta: if same_party { 2.0 } else { -0.5 },
                        bond_boost: None,
                    });
                }
            }
            RomanceStage::Courting => {
                if bond > 75.0 {
                    updates.push(RomanceUpdate {
                        idx,
                        new_stage: Some(RomanceStage::Together),
                        strength_delta: 10.0,
                        bond_boost: None,
                    });
                } else {
                    updates.push(RomanceUpdate {
                        idx,
                        new_stage: None,
                        strength_delta: if same_party { 3.0 } else { -1.0 },
                        bond_boost: None,
                    });
                }
            }
            RomanceStage::Together => {
                // Check for strain conditions.
                let has_rivalry = super::rivalries::has_rivalry(
                    // Can't call with &CampaignState during iteration — check rivalries vec directly.
                    state, a, b,
                );

                // Long separation: different parties for SEPARATION_STRAIN_TICKS.
                let separated_long = !same_party
                    && tick.saturating_sub(rom.started_tick) > SEPARATION_STRAIN_TICKS;

                if has_rivalry || separated_long {
                    updates.push(RomanceUpdate {
                        idx,
                        new_stage: Some(RomanceStage::Strained),
                        strength_delta: -10.0,
                        bond_boost: None,
                    });
                } else {
                    // Together benefits: bond grows 2x.
                    updates.push(RomanceUpdate {
                        idx,
                        new_stage: None,
                        strength_delta: if same_party { 5.0 } else { 1.0 },
                        bond_boost: if same_party { Some((a, b)) } else { None },
                    });
                }
            }
            RomanceStage::Strained => {
                // Strained for 500+ ticks → breaks up.
                // We use the started_tick of the strain transition — approximate with
                // checking if strength has dropped enough.
                if rom.strength < 20.0 {
                    updates.push(RomanceUpdate {
                        idx,
                        new_stage: Some(RomanceStage::BrokenUp),
                        strength_delta: 0.0,
                        bond_boost: None,
                    });
                } else {
                    // Can recover if both back in same party and bond recovering.
                    if same_party && bond > 60.0 {
                        updates.push(RomanceUpdate {
                            idx,
                            new_stage: Some(RomanceStage::Together),
                            strength_delta: 5.0,
                            bond_boost: None,
                        });
                    } else {
                        updates.push(RomanceUpdate {
                            idx,
                            new_stage: None,
                            strength_delta: -3.0,
                            bond_boost: None,
                        });
                    }
                }
            }
            RomanceStage::BrokenUp => unreachable!(),
        }
    }

    // Apply updates.
    for upd in updates {
        let rom = &mut state.romances[upd.idx];
        rom.strength = (rom.strength + upd.strength_delta).clamp(0.0, 100.0);

        if let Some(new_stage) = upd.new_stage {
            let a = rom.adventurer_a;
            let b = rom.adventurer_b;
            let old_stage = rom.stage.clone();
            rom.stage = new_stage.clone();

            match &new_stage {
                RomanceStage::BrokenUp => {
                    rom.started_tick = tick; // Track breakup time for cooldown.
                    // Morale penalty.
                    for &id in &[a, b] {
                        if let Some(adv) = state.adventurers.iter_mut().find(|x| x.id == id) {
                            adv.morale = (adv.morale - 10.0).max(0.0);
                        }
                    }
                    // Chance of rivalry forming.
                    let roll = lcg_f32(&mut state.rng);
                    if roll < 0.30 {
                        let (lo, hi) = (a.min(b), a.max(b));
                        if !super::rivalries::has_rivalry(state, lo, hi) {
                            state.rivalries.push(crate::headless_campaign::state::Rivalry {
                                adventurer_a: lo,
                                adventurer_b: hi,
                                intensity: 20.0,
                                cause: crate::headless_campaign::state::RivalryCause::PersonalInsult,
                                started_tick: tick,
                            });
                        }
                    }
                    events.push(WorldEvent::RomanceBrokenUp {
                        adventurer_a: a,
                        adventurer_b: b,
                    });
                }
                RomanceStage::Strained => {
                    events.push(WorldEvent::RomanceStrained {
                        adventurer_a: a,
                        adventurer_b: b,
                    });
                }
                _ => {
                    // Attraction→Courting or Courting→Together or Strained→Together.
                    let is_recovery = matches!(old_stage, RomanceStage::Strained);
                    if !is_recovery {
                        events.push(WorldEvent::RomanceProgressed {
                            adventurer_a: a,
                            adventurer_b: b,
                            new_stage: new_stage.clone(),
                        });
                    } else {
                        // Recovery from strained.
                        events.push(WorldEvent::RomanceProgressed {
                            adventurer_a: a,
                            adventurer_b: b,
                            new_stage,
                        });
                    }
                }
            }
        }

        // Bond boost for Together couples in the same party (2x growth).
        if let Some((a, b)) = upd.bond_boost {
            let key = bond_key(a, b);
            let entry = state.adventurer_bonds.entry(key).or_insert(0.0);
            // Extra +0.5 on top of the normal bond tick (effectively 2x).
            *entry = (*entry + 0.5).min(100.0);
        }
    }

    // --- Romantic gestures (small flavor events) ---
    for rom in &state.romances {
        if !matches!(rom.stage, RomanceStage::Together | RomanceStage::Courting) {
            continue;
        }
        // 3% chance per tick for a gesture.
        let mut rng = state.rng;
        let roll = lcg_f32(&mut rng);
        state.rng = rng;
        if roll < 0.03 {
            events.push(WorldEvent::RomanticGesture {
                adventurer_a: rom.adventurer_a,
                adventurer_b: rom.adventurer_b,
            });
            break; // Max one gesture per tick.
        }
    }
}

/// Handle partner death: grieving effects on surviving partner.
pub fn on_partner_died(state: &mut CampaignState, dead_id: u32, events: &mut Vec<WorldEvent>) {
    // Find romances involving the dead adventurer.
    let partner_ids: Vec<u32> = state
        .romances
        .iter()
        .filter(|r| !matches!(r.stage, RomanceStage::BrokenUp))
        .filter_map(|r| {
            if r.adventurer_a == dead_id {
                Some(r.adventurer_b)
            } else if r.adventurer_b == dead_id {
                Some(r.adventurer_a)
            } else {
                None
            }
        })
        .collect();

    for partner_id in &partner_ids {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *partner_id) {
            // -30 morale, +30 stress (grieving).
            adv.morale = (adv.morale - 30.0).max(0.0);
            adv.stress = (adv.stress + 30.0).min(100.0);
        }
    }

    // Remove romances involving the dead adventurer.
    state.romances.retain(|r| r.adventurer_a != dead_id && r.adventurer_b != dead_id);
}

/// Handle partner injury: apply fear effect on surviving partner.
pub fn on_partner_injured(state: &mut CampaignState, injured_id: u32, _events: &mut Vec<WorldEvent>) {
    let partner_ids: Vec<u32> = state
        .romances
        .iter()
        .filter(|r| matches!(r.stage, RomanceStage::Together))
        .filter_map(|r| {
            if r.adventurer_a == injured_id {
                Some(r.adventurer_b)
            } else if r.adventurer_b == injured_id {
                Some(r.adventurer_a)
            } else {
                None
            }
        })
        .collect();

    for partner_id in &partner_ids {
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == *partner_id) {
            // Fearful: +15 stress.
            adv.stress = (adv.stress + 15.0).min(100.0);
        }
    }
}

/// Clean up romances involving dead adventurers.
fn cleanup_dead_romances(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    let dead_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Dead)
        .map(|a| a.id)
        .collect();

    state
        .romances
        .retain(|r| !dead_ids.contains(&r.adventurer_a) && !dead_ids.contains(&r.adventurer_b));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_party_after_breakup() {
        let mut state = CampaignState::default_test_campaign(42);
        state.romances.push(Romance {
            adventurer_a: 1,
            adventurer_b: 2,
            stage: RomanceStage::BrokenUp,
            strength: 0.0,
            started_tick: 100,
        });
        state.tick = 200;
        assert!(refuses_party_romance(&state, 1, 2));
        assert!(refuses_party_romance(&state, 2, 1));

        // After cooldown, they can party again.
        state.tick = 100 + BREAKUP_COOLDOWN_TICKS + 1;
        assert!(!refuses_party_romance(&state, 1, 2));
    }

    #[test]
    fn romance_morale_bonus_only_when_together_in_party() {
        let mut state = CampaignState::default_test_campaign(42);
        state.romances.push(Romance {
            adventurer_a: 1,
            adventurer_b: 2,
            stage: RomanceStage::Together,
            strength: 80.0,
            started_tick: 0,
        });
        // Partner in party.
        assert_eq!(romance_morale_bonus(&state, 1, &[1, 2, 3]), 15.0);
        // Partner not in party.
        assert_eq!(romance_morale_bonus(&state, 1, &[1, 3]), 0.0);
        // Courting stage — no bonus.
        state.romances[0].stage = RomanceStage::Courting;
        assert_eq!(romance_morale_bonus(&state, 1, &[1, 2, 3]), 0.0);
    }

    #[test]
    fn romance_combat_multiplier_works() {
        let mut state = CampaignState::default_test_campaign(42);
        state.romances.push(Romance {
            adventurer_a: 1,
            adventurer_b: 2,
            stage: RomanceStage::Together,
            strength: 80.0,
            started_tick: 0,
        });
        assert_eq!(romance_combat_multiplier(&state, 1, &[1, 2]), 1.10);
        assert_eq!(romance_combat_multiplier(&state, 1, &[1, 3]), 1.0);
    }

    #[test]
    fn active_romance_partner_finds_partner() {
        let mut state = CampaignState::default_test_campaign(42);
        state.romances.push(Romance {
            adventurer_a: 1,
            adventurer_b: 2,
            stage: RomanceStage::Courting,
            strength: 50.0,
            started_tick: 0,
        });
        assert_eq!(active_romance_partner(&state, 1), Some(2));
        assert_eq!(active_romance_partner(&state, 2), Some(1));
        assert_eq!(active_romance_partner(&state, 3), None);

        // Broken up — no active partner.
        state.romances[0].stage = RomanceStage::BrokenUp;
        assert_eq!(active_romance_partner(&state, 1), None);
    }
}
