//! Guild trophy hall system — delta architecture port.
//!
//! Trophies are earned from major victories (defeating nemeses, resolving
//! crises, conquering factions, discovering artifacts, quest milestones).
//! Each trophy provides a passive bonus that stacks (up to 10 trophies).
//!
//! Original: `crates/headless_campaign/src/systems/trophies.rs`
//! Cadence: every 17 ticks (skips tick 0).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{Entity, WorldState};

//   Trophy { id, name, trophy_type, source_description, earned_tick, bonus: TrophyBonus }
//   TrophyType: NemesisSkull, AncientArtifact, MonsterTrophy, CrisisRelic, FactionBanner, QuestToken
//   TrophyBonus: RecruitmentBoost, MoraleBoost, ReputationBoost, CombatBoost, GoldBoost, XpBoost

/// Cadence gate.
const TROPHY_TICK_INTERVAL: u64 = 17;

/// Maximum trophies the hall can hold.
const MAX_TROPHIES: usize = 10;

/// Compute trophy deltas: check triggers, apply passive bonuses.
///
/// Trophy gold bonuses can be expressed via UpdateTreasury. Morale
/// bonuses require per-entity AdjustMorale deltas.
pub fn compute_trophies(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % TROPHY_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Check for trophy-earning triggers ---

    // NemesisSkull: high-threat combat quest victory (threat >= 80)
    //   out.push(WorldDelta::EarnTrophy { NemesisSkull, CombatBoost(0.05) })

    // AncientArtifact: exploration quest victory in high-threat area (>= 60)
    //   out.push(WorldDelta::EarnTrophy { AncientArtifact, XpBoost(0.10) })

    // MonsterTrophy: combat victory with threat 70-80
    //   out.push(WorldDelta::EarnTrophy { MonsterTrophy, MoraleBoost(0.03) })

    // CrisisRelic: crisis resolved, high reputation, campaign progress > 0.5
    //   out.push(WorldDelta::EarnTrophy { CrisisRelic, ReputationBoost(0.10) })

    // FactionBanner: faction with 0 territory and low military strength
    //   out.push(WorldDelta::EarnTrophy { FactionBanner, RecruitmentBoost(0.05) })

    // QuestToken: every 10th completed quest
    //   out.push(WorldDelta::EarnTrophy { QuestToken, GoldBoost(0.05) })

    // If at max capacity, replace weakest trophy (lowest bonus magnitude)

    // --- Apply passive bonuses ---

    // Reputation boost: out.push(WorldDelta::AdjustReputation { delta: reputation * total_rep_boost })
    // Gold boost: flat gold per application
    //   out.push(WorldDelta::UpdateTreasury { settlement_id: guild_settlement, delta: 10.0 * total_gold_boost })
    // Morale boost: for each alive NPC:
    //   out.push(WorldDelta::AdjustMorale { entity_id, delta: total_morale * 100.0 })

    // Apply gold boost from trophy hall (structural skeleton)
    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_trophies_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

/// Per-settlement variant for parallel dispatch.
pub fn compute_trophies_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    _entities: &[Entity],
    _out: &mut Vec<WorldDelta>,
) {
    if state.tick % TROPHY_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // out.push(WorldDelta::UpdateTreasury {
    //     settlement_id: settlement_id,
    //     delta: 10.0 * total_gold_boost,
    // });
}
