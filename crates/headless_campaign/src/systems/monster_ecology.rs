//! Monster ecology system — every 200 ticks.
//!
//! Monster populations breed, migrate between regions, and compete for territory.
//! Overhunting depletes regions (no monster quests); neglect lets populations
//! explode (settlement attacks, caravan raids).

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Population cap per species per region.
const POP_CAP: f32 = 100.0;
/// Base growth rate per ecology tick (2%).
const BASE_GROWTH_RATE: f32 = 0.02;
/// Additional growth per 500 ticks since last hunt (1%).
const NEGLECT_GROWTH_BONUS: f32 = 0.01;
/// Migration overflow threshold — population must exceed this to migrate.
const MIGRATION_THRESHOLD: f32 = 70.0;
/// Chance (0-1) that overflow migrates when above threshold.
const MIGRATION_CHANCE: f32 = 0.10;
/// Fraction of overflow that migrates.
const MIGRATION_FRACTION: f32 = 0.5;
/// Population level at which settlement attacks begin.
const SETTLEMENT_ATTACK_THRESHOLD: f32 = 80.0;
/// Population level at which region threat increases.
const THREAT_INCREASE_THRESHOLD: f32 = 50.0;
/// Aggression level at which monsters attack caravans/parties.
const CARAVAN_ATTACK_AGGRESSION: f32 = 60.0;
/// Population reduction from settlement attacks.
const SETTLEMENT_ATTACK_POP_COST: f32 = 5.0;
/// Tick cadence for this system.
const ECOLOGY_TICK_INTERVAL: u64 = 7;

/// Tick the monster ecology system. Runs every 200 ticks.
pub fn tick_monster_ecology(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % ECOLOGY_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let current_tick = state.tick;
    let num_regions = state.overworld.regions.len();
    if num_regions == 0 {
        return;
    }

    // Collect neighbor info for migration (avoid borrow conflicts).
    let region_neighbors: Vec<Vec<usize>> = state
        .overworld
        .regions
        .iter()
        .map(|r| r.neighbors.clone())
        .collect();

    // --- Growth phase ---
    for pop in &mut state.monster_populations {
        let ticks_since_hunt = current_tick.saturating_sub(pop.last_hunted_tick);
        let neglect_bonus = (ticks_since_hunt / 500) as f32 * NEGLECT_GROWTH_BONUS;
        let growth_rate = BASE_GROWTH_RATE + neglect_bonus;
        pop.population = (pop.population * (1.0 + growth_rate)).min(POP_CAP);

        // Aggression scales with population density.
        let density_aggression = (pop.population / POP_CAP) * 50.0;
        pop.aggression = (pop.aggression * 0.95 + density_aggression * 0.05).clamp(0.0, 100.0);
    }

    // --- Migration phase ---
    // Collect migrations first, then apply (to avoid borrow issues).
    let mut migrations: Vec<(usize, usize, MonsterSpecies, f32)> = Vec::new();

    for pop in &state.monster_populations {
        if pop.population > MIGRATION_THRESHOLD {
            let roll = lcg_f32(&mut state.rng.clone());
            // Advance RNG deterministically
            let _ = lcg_next(&mut state.rng);
            if roll < MIGRATION_CHANCE {
                if let Some(neighbors) = region_neighbors.get(pop.region_id) {
                    if !neighbors.is_empty() {
                        let target_idx =
                            (lcg_next(&mut state.rng) as usize) % neighbors.len();
                        let target_region = neighbors[target_idx];
                        let overflow = pop.population - MIGRATION_THRESHOLD;
                        let migrating = overflow * MIGRATION_FRACTION;
                        migrations.push((pop.region_id, target_region, pop.species, migrating));
                    }
                }
            }
        }
    }

    // Apply migrations.
    for (from, to, species, amount) in &migrations {
        // Reduce source population.
        if let Some(src) = state
            .monster_populations
            .iter_mut()
            .find(|p| p.region_id == *from && p.species == *species)
        {
            src.population = (src.population - amount).max(0.0);
        }

        // Add to destination (create entry if needed).
        if let Some(dst) = state
            .monster_populations
            .iter_mut()
            .find(|p| p.region_id == *to && p.species == *species)
        {
            dst.population = (dst.population + amount).min(POP_CAP);
        } else {
            state.monster_populations.push(MonsterPopulation {
                region_id: *to,
                species: *species,
                population: amount.min(POP_CAP),
                aggression: 20.0,
                last_hunted_tick: 0,
            });
        }

        events.push(WorldEvent::MonsterMigration {
            from: format!("{}", from),
            to: format!("{}", to),
            species: format!("{:?}", species),
        });
    }

    // --- Effects phase ---
    // Collect indices and data for events/modifications to avoid borrow issues.
    let mut threat_increases: Vec<usize> = Vec::new();
    let mut settlement_attacks: Vec<(usize, MonsterSpecies, f32)> = Vec::new();
    let mut caravan_attacks: Vec<(usize, MonsterSpecies)> = Vec::new();

    for pop in &state.monster_populations {
        // Threat increase at population > 50.
        if pop.population > THREAT_INCREASE_THRESHOLD {
            threat_increases.push(pop.region_id);
        }

        // Settlement attacks at population > 80.
        if pop.population > SETTLEMENT_ATTACK_THRESHOLD {
            let damage = (pop.population - SETTLEMENT_ATTACK_THRESHOLD) * 0.5;
            settlement_attacks.push((pop.region_id, pop.species, damage));
        }

        // Caravan/party attacks at high aggression.
        if pop.aggression > CARAVAN_ATTACK_AGGRESSION {
            caravan_attacks.push((pop.region_id, pop.species));
        }
    }

    // Apply threat increases to regions.
    for region_id in &threat_increases {
        if let Some(region) = state
            .overworld
            .regions
            .iter_mut()
            .find(|r| r.id == *region_id)
        {
            region.threat_level = (region.threat_level + 2.0).min(100.0);
        }
    }

    // Apply settlement attacks.
    for (region_id, species, damage) in &settlement_attacks {
        // Reduce population slightly (defenders fight back).
        if let Some(pop) = state
            .monster_populations
            .iter_mut()
            .find(|p| p.region_id == *region_id && p.species == *species)
        {
            pop.population = (pop.population - SETTLEMENT_ATTACK_POP_COST).max(0.0);
        }

        // Reduce region control (civilian morale impact).
        if let Some(region) = state
            .overworld
            .regions
            .iter_mut()
            .find(|r| r.id == *region_id)
        {
            region.control = (region.control - 10.0).max(0.0);
            region.unrest = (region.unrest + 5.0).min(100.0);
        }

        events.push(WorldEvent::MonsterAttack {
            region: format!("{}", region_id),
            species: format!("{:?}", species),
            damage: *damage,
        });
    }

    // Caravan attacks emit swarm events (high aggression populations).
    for (region_id, species) in &caravan_attacks {
        events.push(WorldEvent::MonsterSwarm {
            region: format!("{}", region_id),
            species: format!("{:?}", species),
        });
    }
}

/// Apply hunting effect when a combat quest is completed in a region.
///
/// Reduces the population of a matching monster species by 10-30 (deterministic).
pub fn apply_hunting(state: &mut CampaignState, region_id: usize) {
    // Find populations in the region.
    let candidates: Vec<usize> = state
        .monster_populations
        .iter()
        .enumerate()
        .filter(|(_, p)| p.region_id == region_id && p.population > 0.0)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return;
    }

    // Pick a species deterministically.
    let idx = (lcg_next(&mut state.rng) as usize) % candidates.len();
    let pop_idx = candidates[idx];

    // Reduce by 10-30 based on RNG.
    let reduction = 10.0 + (lcg_f32(&mut state.rng) * 20.0);
    let pop = &mut state.monster_populations[pop_idx];
    pop.population = (pop.population - reduction).max(0.0);
    pop.aggression = (pop.aggression - 15.0).max(0.0);
    pop.last_hunted_tick = state.tick;
}

/// Check if a region has any huntable monster populations (population >= 10).
///
/// Regions with all species below 10 have no monster quests available.
pub fn region_has_monsters(state: &CampaignState, region_id: usize) -> bool {
    state
        .monster_populations
        .iter()
        .any(|p| p.region_id == region_id && p.population >= 10.0)
}

/// Initialize monster populations for a set of regions.
///
/// Each region gets 2-3 species based on the seed for determinism.
pub fn init_monster_populations(regions: &[Region], seed: u64) -> Vec<MonsterPopulation> {
    let mut rng = seed.wrapping_add(0xDEAD_BEEF);
    let mut populations = Vec::new();

    for region in regions {
        // Determine how many species (2-3).
        let species_count = 2 + (lcg_next(&mut rng) % 2) as usize;

        // Pick species without replacement.
        let mut available = MonsterSpecies::ALL.to_vec();
        for _ in 0..species_count {
            if available.is_empty() {
                break;
            }
            let idx = (lcg_next(&mut rng) as usize) % available.len();
            let species = available.remove(idx);

            // Starting population: 15-40.
            let starting_pop = 15.0 + lcg_f32(&mut rng) * 25.0;
            // Starting aggression: 10-30.
            let starting_aggr = 10.0 + lcg_f32(&mut rng) * 20.0;

            populations.push(MonsterPopulation {
                region_id: region.id,
                species,
                population: starting_pop,
                aggression: starting_aggr,
                last_hunted_tick: 0,
            });
        }
    }

    populations
}
