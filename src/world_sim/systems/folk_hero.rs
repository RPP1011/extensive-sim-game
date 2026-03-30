//! Folk hero reputation system — delta architecture port.
//!
//! Tracks how common people view the guild, separate from faction reputation.
//! Fame grows from heroic deeds (defending settlements, defeating monsters)
//! and decays from neglect. Regional fame thresholds unlock benefits
//! (cheaper prices, militia) or penalties (suspicion, hostility).
//!
//! Original: `crates/headless_campaign/src/systems/folk_hero.rs`
//! Cadence: every 17 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, WorldState};

//   FolkReputation { regional_fame: HashMap<u32, f32>, overall_fame: f32, folk_tales: Vec<FolkTale> }
//   FolkTale { id, adventurer_id, tale, region_id, fame_impact, created_tick }

/// Cadence gate.
const FOLK_HERO_TICK_INTERVAL: u64 = 17;

/// Max folk tales.
const MAX_FOLK_TALES: usize = 20;

/// Ticks before a folk tale fades.
const TALE_LIFETIME: u64 = 167;

/// Fame thresholds.
const FAME_POSITIVE_THRESHOLD: f32 = 50.0;
const FAME_HERO_THRESHOLD: f32 = 75.0;
const FAME_SUSPICION_THRESHOLD: f32 = 20.0;

/// Passive fame decay per tick interval.
const FAME_DECAY_RATE: f32 = 0.5;

/// Compute folk hero deltas: fame changes, tale generation, fame effects.
///
/// Fame changes from heroic deeds could be expressed via UpdateTreasury
/// (tribute gold from hero status) and settlement-level effects.
pub fn compute_folk_hero(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % FOLK_HERO_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Detect fame-generating events ---
    // Positive sources:
    //   - Defended settlement (quest victory near settlement) → +8 fame
    //   - Defeated monsters (region threat reduced) → +4 fame
    //   - Charitable guild (high treasury + reputation) → +5 fame
    //
    // Negative sources:
    //   - High threat in region with no active defense → -10 fame
    //   - Settlement treasury very low (neglect) → -5 fame

    // Use region threat_level as proxy for fame-relevant events
    for region in &state.regions {
        let _threat = region.threat_level;
        // High threat + no friendly entities nearby → "ignored crisis"
        // Low threat after reduction → "defeated monsters"
    }

    // --- Phase 2: Apply fame deltas ---

    // --- Phase 3: Generate folk tales from significant events ---

    // --- Phase 4: Spread tales to adjacent regions (diminished) ---

    // --- Phase 5: Fame threshold effects ---
    // fame > 75 (hero): tribute gold, unrest reduction
    //   out.push(WorldDelta::UpdateTreasury { settlement_id: settlement.id, delta: 2.0 })
    // fame > 50 (positive): slight unrest reduction
    // fame < 20 (suspicion): increase unrest

    // Apply tribute from hero-status settlements
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_folk_hero_for_settlement(state, settlement.id, &state.entities[range], out);
    }

    // --- Phase 6: Passive decay toward neutral (50) ---

    // --- Phase 7: Expire old tales ---
}

/// Per-settlement variant for parallel dispatch.
///
/// Applies fame threshold effects (tribute gold) for a single settlement's entities.
pub fn compute_folk_hero_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    _entities: &[Entity],
    _out: &mut Vec<WorldDelta>,
) {
    if state.tick % FOLK_HERO_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // If fame > FAME_HERO_THRESHOLD:
    //   out.push(WorldDelta::UpdateTreasury {
    //       settlement_id: settlement_id,
    //       delta: 2.0,
    //   });
}
