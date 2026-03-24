//! Funeral and memorial system -- every 300 ticks.
//!
//! When adventurers die, the guild can hold funerals that affect morale,
//! bonds, and create lasting memorials. Pending funerals auto-resolve as
//! SimpleFunerals after 500 ticks if the player has not acted.

use crate::headless_campaign::actions::{ActionResult, StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;
use crate::headless_campaign::systems::bonds::bond_key;

/// How often to run the memorial system (in ticks).
const MEMORIAL_INTERVAL: u64 = 300;

/// Auto-resolve pending funerals after this many ticks.
const AUTO_FUNERAL_DELAY: u64 = 500;

/// Maximum number of memorials kept (oldest non-monument replaced).
const MAX_MEMORIALS: usize = 5;

/// Morale duration for temporary funeral effects (ticks).
const TEMP_MORALE_DURATION: u64 = 500;

// ---------------------------------------------------------------------------
// Tick entry point
// ---------------------------------------------------------------------------

/// Detect new deaths, auto-resolve stale funerals, and apply memorial morale.
pub fn tick_memorials(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % MEMORIAL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Detect new deaths and create pending funerals ---
    detect_new_deaths(state, events);

    // --- Auto-resolve stale pending funerals ---
    auto_resolve_funerals(state, events);

    // --- Apply ongoing memorial morale effects ---
    apply_memorial_morale(state, events);
}

// ---------------------------------------------------------------------------
// Death detection
// ---------------------------------------------------------------------------

/// Scan adventurers for deaths not yet in pending_funerals or memorials.
fn detect_new_deaths(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    let dead_ids: Vec<(u32, String, u32)> = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Dead)
        .map(|a| (a.id, a.name.clone(), a.level))
        .collect();

    for (id, name, level) in dead_ids {
        // Skip if already tracked
        let already_pending = state.pending_funerals.iter().any(|f| f.adventurer_id == id);
        let already_memorialized = state.memorials.iter().any(|m| m.adventurer_id == id);
        if already_pending || already_memorialized {
            continue;
        }

        state.pending_funerals.push(FuneralPending {
            adventurer_id: id,
            name,
            level,
            death_tick: state.tick,
        });
    }
}

// ---------------------------------------------------------------------------
// Auto-resolve
// ---------------------------------------------------------------------------

/// Auto-resolve funerals that have been pending for too long as SimpleFunerals.
fn auto_resolve_funerals(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let stale: Vec<FuneralPending> = state
        .pending_funerals
        .iter()
        .filter(|f| tick >= f.death_tick + AUTO_FUNERAL_DELAY)
        .cloned()
        .collect();

    for funeral in stale {
        // Remove from pending
        state
            .pending_funerals
            .retain(|f| f.adventurer_id != funeral.adventurer_id);

        // Create a simple funeral
        create_memorial(
            state,
            funeral.adventurer_id,
            &funeral.name,
            MemorialType::SimpleFuneral,
            events,
        );
    }
}

// ---------------------------------------------------------------------------
// Memorial morale application
// ---------------------------------------------------------------------------

/// Apply morale bonuses from active memorials. Temporary ones expire.
fn apply_memorial_morale(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;

    // Collect memorial info to avoid borrow issues
    let memorial_info: Vec<(u32, MemorialType, u64, f32, u32)> = state
        .memorials
        .iter()
        .map(|m| (m.id, m.memorial_type, m.created_tick, m.morale_bonus, m.adventurer_id))
        .collect();

    for (memorial_id, mtype, created_tick, bonus, deceased_id) in &memorial_info {
        let age = tick.saturating_sub(*created_tick);

        // Check if this temporary memorial has expired
        let is_permanent = matches!(mtype, MemorialType::Monument | MemorialType::NamedBuilding);
        if !is_permanent && age > TEMP_MORALE_DURATION {
            continue; // Skip expired temporary memorials
        }

        // Apply morale bonus to all living adventurers
        for adv in &mut state.adventurers {
            if adv.status == AdventurerStatus::Dead {
                continue;
            }

            let mut effective_bonus = *bonus;

            // Bonded allies of the deceased get extra morale from the funeral
            let bond = *state
                .adventurer_bonds
                .get(&bond_key(adv.id, *deceased_id))
                .unwrap_or(&0.0);
            if bond > 20.0 {
                effective_bonus += bond * 0.05; // Up to +5 extra for max bond
            }

            // Scale down per-tick: total bonus spread over MEMORIAL_INTERVAL ticks
            let per_tick = effective_bonus / MEMORIAL_INTERVAL as f32;
            adv.morale = (adv.morale + per_tick).min(100.0);

            if per_tick > 0.1 {
                events.push(WorldEvent::MemorialMorale {
                    adventurer_id: adv.id,
                    morale_delta: per_tick,
                    source: format!("memorial_{}", memorial_id),
                });
            }
        }
    }

    // Remove expired temporary memorials
    state.memorials.retain(|m| {
        let is_permanent = matches!(
            m.memorial_type,
            MemorialType::Monument | MemorialType::NamedBuilding
        );
        is_permanent || tick.saturating_sub(m.created_tick) <= TEMP_MORALE_DURATION
    });
}

// ---------------------------------------------------------------------------
// Action handler
// ---------------------------------------------------------------------------

/// Handle the HoldFuneral action from the player.
pub fn apply_hold_funeral(
    state: &mut CampaignState,
    adventurer_id: u32,
    memorial_type_str: &str,
    events: &mut Vec<WorldEvent>,
) -> ActionResult {
    // Validate pending funeral exists
    let funeral = state
        .pending_funerals
        .iter()
        .find(|f| f.adventurer_id == adventurer_id);

    let funeral = match funeral {
        Some(f) => f.clone(),
        None => {
            return ActionResult::InvalidAction(format!(
                "No pending funeral for adventurer {}",
                adventurer_id
            ));
        }
    };

    // Parse memorial type
    let mtype = match memorial_type_str {
        "SimpleFuneral" => MemorialType::SimpleFuneral,
        "HerosFuneral" => MemorialType::HerosFuneral,
        "Monument" => MemorialType::Monument,
        "NamedBuilding" => MemorialType::NamedBuilding,
        "LegendaryTale" => MemorialType::LegendaryTale,
        _ => {
            return ActionResult::InvalidAction(format!(
                "Unknown memorial type: {}",
                memorial_type_str
            ));
        }
    };

    // Check gold costs
    let cost = memorial_cost(mtype);
    if state.guild.gold < cost {
        return ActionResult::Failed(format!(
            "Not enough gold for {:?}: need {}, have {}",
            mtype, cost, state.guild.gold
        ));
    }

    // Deduct gold
    state.guild.gold -= cost;
    if cost > 0.0 {
        events.push(WorldEvent::GoldChanged {
            amount: -cost,
            reason: format!("{:?} for {}", mtype, funeral.name),
        });
    }

    // Remove from pending
    state
        .pending_funerals
        .retain(|f| f.adventurer_id != adventurer_id);

    // Create the memorial
    create_memorial(state, adventurer_id, &funeral.name, mtype, events);

    ActionResult::Success(format!(
        "{:?} held for {}",
        mtype, funeral.name
    ))
}

// ---------------------------------------------------------------------------
// Memorial creation
// ---------------------------------------------------------------------------

/// Create a memorial and apply its immediate effects.
fn create_memorial(
    state: &mut CampaignState,
    adventurer_id: u32,
    name: &str,
    mtype: MemorialType,
    events: &mut Vec<WorldEvent>,
) {
    let memorial_id = state.next_memorial_id;
    state.next_memorial_id += 1;

    let (morale_bonus, description) = match mtype {
        MemorialType::SimpleFuneral => (
            5.0,
            format!("A simple funeral was held for {}. The guild mourns.", name),
        ),
        MemorialType::HerosFuneral => {
            // +5 reputation
            state.guild.reputation = (state.guild.reputation + 5.0).min(100.0);
            (
                15.0,
                format!(
                    "A hero's funeral was held for {}. Their sacrifice is honored across the land.",
                    name
                ),
            )
        }
        MemorialType::Monument => (
            2.0,
            format!(
                "A stone monument was erected in memory of {}. It stands as a permanent reminder.",
                name
            ),
        ),
        MemorialType::NamedBuilding => (
            3.0,
            format!(
                "A guild building was renamed in honor of {}. Their name endures in the guild hall.",
                name
            ),
        ),
        MemorialType::LegendaryTale => {
            // Create chronicle entry
            state.chronicle.push(ChronicleEntry {
                tick: state.tick,
                entry_type: ChronicleType::Memorial,
                text: format!(
                    "The tale of {} has been immortalized in the guild chronicles. Their legend inspires future adventurers.",
                    name
                ),
                participants: vec![adventurer_id],
                location_id: None,
                faction_id: None,
                significance: 9.0,
            });
            (
                0.0,
                format!(
                    "The legendary tale of {} has been chronicled, inspiring future quests of vengeance and remembrance.",
                    name
                ),
            )
        }
    };

    let memorial = Memorial {
        id: memorial_id,
        adventurer_name: name.to_string(),
        adventurer_id,
        memorial_type: mtype,
        created_tick: state.tick,
        morale_bonus,
        description: description.clone(),
    };

    // Enforce max memorials: oldest non-monument replaced
    if state.memorials.len() >= MAX_MEMORIALS {
        if let Some(pos) = state.memorials.iter().position(|m| {
            !matches!(
                m.memorial_type,
                MemorialType::Monument | MemorialType::NamedBuilding
            )
        }) {
            state.memorials.remove(pos);
        }
    }

    state.memorials.push(memorial);

    // Emit events
    events.push(WorldEvent::FuneralHeld {
        adventurer_id,
        adventurer_name: name.to_string(),
        memorial_type: format!("{:?}", mtype),
    });

    events.push(WorldEvent::MemorialCreated {
        memorial_id,
        adventurer_name: name.to_string(),
        memorial_type: format!("{:?}", mtype),
        description,
    });

    // Bonded allies get immediate extra morale from the funeral
    let bond_pairs: Vec<(u32, f32)> = state
        .adventurer_bonds
        .iter()
        .filter_map(|(&(a, b), &strength)| {
            if a == adventurer_id {
                Some((b, strength))
            } else if b == adventurer_id {
                Some((a, strength))
            } else {
                None
            }
        })
        .collect();

    for (bonded_id, strength) in bond_pairs {
        if strength > 10.0 {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == bonded_id && a.status != AdventurerStatus::Dead)
            {
                let grief_comfort = strength * 0.1; // Up to +10 morale for max bond
                adv.morale = (adv.morale + grief_comfort).min(100.0);
                events.push(WorldEvent::MemorialMorale {
                    adventurer_id: bonded_id,
                    morale_delta: grief_comfort,
                    source: format!("bond_grief_comfort_for_{}", name),
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gold cost for a given memorial type.
fn memorial_cost(mtype: MemorialType) -> f32 {
    match mtype {
        MemorialType::SimpleFuneral => 0.0,
        MemorialType::HerosFuneral => 30.0,
        MemorialType::Monument => 100.0,
        MemorialType::NamedBuilding => 50.0,
        MemorialType::LegendaryTale => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::actions::StepDeltas;

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        state.tick = 299;
        state
    }

    #[test]
    fn detect_deaths_creates_pending_funeral() {
        let mut state = make_test_state();
        state.tick = 300;

        // Kill an adventurer
        if let Some(adv) = state.adventurers.first_mut() {
            adv.status = AdventurerStatus::Dead;
        }

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_memorials(&mut state, &mut deltas, &mut events);

        assert!(
            !state.pending_funerals.is_empty(),
            "Expected pending funeral for dead adventurer"
        );
    }

    #[test]
    fn auto_resolve_after_delay() {
        let mut state = make_test_state();
        state.tick = 900; // 300 * 3

        // Add a stale pending funeral
        state.pending_funerals.push(FuneralPending {
            adventurer_id: 1,
            name: "Fallen Hero".into(),
            level: 5,
            death_tick: 100, // 800 ticks ago > 500 threshold
        });

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();
        tick_memorials(&mut state, &mut deltas, &mut events);

        assert!(
            state.pending_funerals.is_empty(),
            "Stale funeral should be auto-resolved"
        );
        assert!(
            !state.memorials.is_empty(),
            "Auto-resolved funeral should create a memorial"
        );
        assert_eq!(state.memorials[0].memorial_type, MemorialType::SimpleFuneral);
    }

    #[test]
    fn hold_heros_funeral() {
        let mut state = make_test_state();
        state.guild.gold = 100.0;

        state.pending_funerals.push(FuneralPending {
            adventurer_id: 99,
            name: "Brave Knight".into(),
            level: 10,
            death_tick: state.tick,
        });

        let mut events = Vec::new();
        let result = apply_hold_funeral(&mut state, 99, "HerosFuneral", &mut events);

        assert!(
            matches!(result, ActionResult::Success(_)),
            "HerosFuneral should succeed"
        );
        assert_eq!(state.guild.gold, 70.0, "Should cost 30 gold");
        assert!(
            state.pending_funerals.is_empty(),
            "Pending funeral should be consumed"
        );
        assert_eq!(state.memorials.len(), 1);
        assert_eq!(state.memorials[0].memorial_type, MemorialType::HerosFuneral);
        assert!(
            events.iter().any(|e| matches!(e, WorldEvent::FuneralHeld { .. })),
            "Should emit FuneralHeld event"
        );
    }

    #[test]
    fn max_memorials_evicts_oldest_non_monument() {
        let mut state = make_test_state();

        // Fill up with simple funerals
        for i in 0..MAX_MEMORIALS {
            state.memorials.push(Memorial {
                id: i as u32,
                adventurer_name: format!("Hero {}", i),
                adventurer_id: i as u32,
                memorial_type: MemorialType::SimpleFuneral,
                created_tick: i as u64 * 100,
                morale_bonus: 5.0,
                description: String::new(),
            });
        }

        state.pending_funerals.push(FuneralPending {
            adventurer_id: 100,
            name: "New Hero".into(),
            level: 1,
            death_tick: state.tick,
        });

        let mut events = Vec::new();
        let result = apply_hold_funeral(&mut state, 100, "SimpleFuneral", &mut events);

        assert!(matches!(result, ActionResult::Success(_)));
        assert_eq!(state.memorials.len(), MAX_MEMORIALS);
        assert!(
            state.memorials.iter().any(|m| m.adventurer_id == 100),
            "New memorial should be present"
        );
    }

    #[test]
    fn insufficient_gold_fails() {
        let mut state = make_test_state();
        state.guild.gold = 10.0;

        state.pending_funerals.push(FuneralPending {
            adventurer_id: 1,
            name: "Poor Hero".into(),
            level: 1,
            death_tick: state.tick,
        });

        let mut events = Vec::new();
        let result = apply_hold_funeral(&mut state, 1, "Monument", &mut events);

        assert!(
            matches!(result, ActionResult::Failed(_)),
            "Monument should fail with insufficient gold"
        );
        assert_eq!(state.guild.gold, 10.0, "Gold should not change");
    }
}
