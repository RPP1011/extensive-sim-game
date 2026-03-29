//! Migration and refugee system — fires every 200 ticks (~20s game time).
//!
//! Wars, crises, persecution, and overcrowding cause population to flee
//! between regions. Arriving refugees create humanitarian quests and
//! demographic shifts. The guild can accept refugees for supplies cost,
//! gaining potential recruits.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often migration triggers are checked (in ticks).
const MIGRATION_INTERVAL: u64 = 7;

/// Progress increment per tick (reaches 1.0 in 10 ticks = 1 second game time).
const PROGRESS_PER_TICK: f32 = 0.1;

/// Minimum population a region needs to trigger any outward migration.
const MIN_POPULATION_FOR_MIGRATION: u32 = 50;

/// Supplies cost to accept refugees at the guild base.
pub const ACCEPT_REFUGEES_SUPPLY_COST: f32 = 20.0;

/// Tick migration progress and check for new migration triggers.
pub fn tick_migration(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    // --- Progress existing migrations every tick ---
    advance_migrations(state, events);

    // --- Check for new migrations every MIGRATION_INTERVAL ticks ---
    if state.tick % MIGRATION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    check_war_migration(state, events);
    check_crisis_migration(state, events);
    check_persecution_migration(state, events);
    check_opportunity_migration(state, events);
}

/// Advance all active migration waves toward their destination.
/// On arrival: transfer population and apply destination effects.
fn advance_migrations(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Clean up old arrived migrations that were not accepted (after 500 ticks)
    let current_tick = state.tick;
    state.migrations.retain(|m| {
        if m.progress >= 1.0 && current_tick.saturating_sub(m.started_tick) > 500 {
            return false; // expired, refugees dispersed
        }
        true
    });

    let mut arrived_indices = Vec::new();

    for (i, migration) in state.migrations.iter_mut().enumerate() {
        if migration.progress >= 1.0 {
            continue; // already arrived, waiting for cleanup or acceptance
        }
        migration.progress += PROGRESS_PER_TICK;
        if migration.progress >= 1.0 {
            migration.progress = 1.0;
            arrived_indices.push(i);
        }
    }

    // Process arrivals
    for &i in &arrived_indices {
        let migration = &state.migrations[i];
        let dest_id = migration.destination_region_id;
        let origin_id = migration.origin_region_id;
        let count = migration.population_count;

        // Transfer population
        if let Some(origin) = state.overworld.regions.get_mut(origin_id) {
            origin.population = origin.population.saturating_sub(count);
        }
        if let Some(dest) = state.overworld.regions.get_mut(dest_id) {
            dest.population += count;
            // Temporary unrest from overcrowding (+10, capped at 100)
            dest.unrest = (dest.unrest + 10.0).min(100.0);
        }

        let dest_name = state
            .overworld
            .regions
            .get(dest_id)
            .map(|r| r.name.clone())
            .unwrap_or_else(|| format!("Region {}", dest_id));

        events.push(WorldEvent::RefugeesArrived {
            region: dest_name,
            count,
        });

        // Generate refugee aid quests on the request board
        generate_refugee_quests(state, dest_id, count, events);
    }
}

/// Generate humanitarian quests when refugees arrive.
fn generate_refugee_quests(
    state: &mut CampaignState,
    dest_region_id: usize,
    refugee_count: u32,
    events: &mut Vec<WorldEvent>,
) {
    let dest_pos = state
        .overworld
        .regions
        .get(dest_region_id)
        .and_then(|r| {
            // Find a location in this region's vicinity
            state
                .overworld
                .locations
                .iter()
                .find(|l| l.faction_owner == Some(r.owner_faction_id))
                .map(|l| l.position)
        })
        .unwrap_or((50.0, 50.0));

    let base_pos = state.guild.base.position;
    let dx = dest_pos.0 - base_pos.0;
    let dy = dest_pos.1 - base_pos.1;
    let distance = (dx * dx + dy * dy).sqrt();

    // Escort quest — help refugees reach safety
    let quest_id = state.next_quest_id;
    state.next_quest_id += 1;

    let threat = 20.0 + (refugee_count as f32 * 0.05).min(30.0);
    let quest_type_roll = lcg_f32(&mut state.rng);
    let quest_type = if quest_type_roll < 0.4 {
        QuestType::Escort
    } else if quest_type_roll < 0.7 {
        QuestType::Rescue
    } else {
        QuestType::Gather
    };

    let type_label = match quest_type {
        QuestType::Escort => "Escort Refugees",
        QuestType::Rescue => "Rescue Stranded Refugees",
        QuestType::Gather => "Supply Refugee Camp",
        _ => "Aid Refugees",
    };

    let request = QuestRequest {
        id: quest_id,
        source_faction_id: state.overworld.regions.get(dest_region_id).map(|r| r.owner_faction_id),
        source_area_id: Some(dest_region_id),
        quest_type,
        threat_level: threat,
        reward: QuestReward {
            gold: 30.0 + refugee_count as f32 * 0.2,
            reputation: 5.0,
            relation_faction_id: state.overworld.regions.get(dest_region_id).map(|r| r.owner_faction_id),
            relation_change: 8.0,
            supply_reward: 0.0,
            potential_loot: false,
        },
        distance,
        target_position: dest_pos,
        deadline_ms: state.elapsed_ms + 167 * CAMPAIGN_TURN_SECS as u64 * 1000,
        description: format!(
            "{}: {} refugees need help in the region.",
            type_label, refugee_count
        ),
        arrived_at_ms: state.elapsed_ms,
    };

    events.push(WorldEvent::QuestRequestArrived {
        request_id: quest_id,
        quest_type,
        threat_level: threat,
    });

    state.request_board.push(request);
}

/// War migration: regions at war with unrest > 40 lose 20% population.
fn check_war_migration(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n_regions = state.overworld.regions.len();
    if n_regions == 0 {
        return;
    }

    // Collect regions at war (owner faction is at war with someone)
    let war_regions: Vec<usize> = (0..n_regions)
        .filter(|&ri| {
            let region = &state.overworld.regions[ri];
            if region.population < MIN_POPULATION_FOR_MIGRATION || region.unrest <= 40.0 {
                return false;
            }
            let owner = region.owner_faction_id;
            state
                .factions
                .get(owner)
                .map(|f| {
                    f.diplomatic_stance == DiplomaticStance::AtWar || !f.at_war_with.is_empty()
                })
                .unwrap_or(false)
        })
        .collect();

    for ri in war_regions {
        // Check if there's already an active migration from this region
        if state
            .migrations
            .iter()
            .any(|m| m.origin_region_id == ri && m.progress < 1.0)
        {
            continue;
        }

        let population = state.overworld.regions[ri].population;
        let fleeing = (population as f32 * 0.20) as u32;
        if fleeing == 0 {
            continue;
        }

        if let Some(dest) = pick_destination(state, ri) {
            spawn_migration(state, ri, dest, fleeing, MigrationCause::War, events);
        }
    }
}

/// Crisis migration: regions with active crises lose 10% population.
fn check_crisis_migration(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n_regions = state.overworld.regions.len();
    if n_regions == 0 {
        return;
    }

    // Collect region IDs affected by crises
    let mut crisis_region_ids: Vec<usize> = Vec::new();
    for crisis in &state.overworld.active_crises {
        match crisis {
            ActiveCrisis::Corruption {
                origin_region_id,
                corrupted_regions,
                ..
            } => {
                crisis_region_ids.push(*origin_region_id);
                crisis_region_ids.extend(corrupted_regions);
            }
            ActiveCrisis::Breach {
                source_location_id, ..
            } => {
                // Find region containing this location
                if let Some(loc) = state
                    .overworld
                    .locations
                    .iter()
                    .find(|l| l.id == *source_location_id)
                {
                    if let Some(owner) = loc.faction_owner {
                        for (i, r) in state.overworld.regions.iter().enumerate() {
                            if r.owner_faction_id == owner {
                                crisis_region_ids.push(i);
                                break;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    crisis_region_ids.sort_unstable();
    crisis_region_ids.dedup();

    for ri in crisis_region_ids {
        if ri >= n_regions {
            continue;
        }
        let region = &state.overworld.regions[ri];
        if region.population < MIN_POPULATION_FOR_MIGRATION {
            continue;
        }
        if state
            .migrations
            .iter()
            .any(|m| m.origin_region_id == ri && m.progress < 1.0)
        {
            continue;
        }

        let fleeing = (region.population as f32 * 0.10) as u32;
        if fleeing == 0 {
            continue;
        }

        if let Some(dest) = pick_destination(state, ri) {
            spawn_migration(state, ri, dest, fleeing, MigrationCause::Crisis, events);
        }
    }
}

/// Persecution migration: regions with civilian_morale < 20 lose 15% population.
fn check_persecution_migration(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n_regions = state.overworld.regions.len();

    let persecuted: Vec<usize> = (0..n_regions)
        .filter(|&ri| {
            let r = &state.overworld.regions[ri];
            r.civilian_morale < 20.0 && r.population >= MIN_POPULATION_FOR_MIGRATION
        })
        .collect();

    for ri in persecuted {
        if state
            .migrations
            .iter()
            .any(|m| m.origin_region_id == ri && m.progress < 1.0)
        {
            continue;
        }

        let population = state.overworld.regions[ri].population;
        let fleeing = (population as f32 * 0.15) as u32;
        if fleeing == 0 {
            continue;
        }

        if let Some(dest) = pick_destination(state, ri) {
            spawn_migration(
                state,
                ri,
                dest,
                fleeing,
                MigrationCause::Persecution,
                events,
            );
        }
    }
}

/// Opportunity migration: overpopulated regions (>700) send 5% to underpopulated neighbors (<200).
fn check_opportunity_migration(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n_regions = state.overworld.regions.len();

    let overpopulated: Vec<usize> = (0..n_regions)
        .filter(|&ri| state.overworld.regions[ri].population > 700)
        .collect();

    for ri in overpopulated {
        if state
            .migrations
            .iter()
            .any(|m| m.origin_region_id == ri && m.progress < 1.0)
        {
            continue;
        }

        // Find an underpopulated neighbor
        let neighbors = state.overworld.regions[ri].neighbors.clone();
        let dest = neighbors
            .iter()
            .copied()
            .find(|&ni| ni < n_regions && state.overworld.regions[ni].population < 200);

        if let Some(dest_id) = dest {
            let population = state.overworld.regions[ri].population;
            let fleeing = (population as f32 * 0.05) as u32;
            if fleeing == 0 {
                continue;
            }
            spawn_migration(
                state,
                ri,
                dest_id,
                fleeing,
                MigrationCause::Opportunity,
                events,
            );
        }
    }
}

/// Pick a destination region for refugees from `origin_id`.
/// Prefers neighbors with lower unrest and higher civilian morale.
fn pick_destination(state: &mut CampaignState, origin_id: usize) -> Option<usize> {
    let neighbors = state.overworld.regions[origin_id].neighbors.clone();
    if neighbors.is_empty() {
        return None;
    }

    let n_regions = state.overworld.regions.len();

    // Score each neighbor: prefer low unrest, high morale, not at war
    let mut best_id = None;
    let mut best_score = f32::NEG_INFINITY;

    for &ni in &neighbors {
        if ni >= n_regions {
            continue;
        }
        let r = &state.overworld.regions[ni];
        let owner = r.owner_faction_id;
        let at_war = state
            .factions
            .get(owner)
            .map(|f| f.diplomatic_stance == DiplomaticStance::AtWar || !f.at_war_with.is_empty())
            .unwrap_or(false);

        if at_war {
            continue; // don't flee to a war zone
        }

        let score = r.civilian_morale - r.unrest - (r.threat_level * 0.5);
        if score > best_score {
            best_score = score;
            best_id = Some(ni);
        }
    }

    // If all neighbors are bad, pick a random one with deterministic RNG
    if best_id.is_none() && !neighbors.is_empty() {
        let valid: Vec<usize> = neighbors
            .iter()
            .copied()
            .filter(|&ni| ni < n_regions)
            .collect();
        if !valid.is_empty() {
            let idx = (lcg_next(&mut state.rng) as usize) % valid.len();
            best_id = Some(valid[idx]);
        }
    }

    best_id
}

/// Create a new migration wave and emit the start event.
fn spawn_migration(
    state: &mut CampaignState,
    origin: usize,
    destination: usize,
    count: u32,
    cause: MigrationCause,
    events: &mut Vec<WorldEvent>,
) {
    let id = state.next_migration_id;
    state.next_migration_id += 1;

    let origin_name = state
        .overworld
        .regions
        .get(origin)
        .map(|r| r.name.clone())
        .unwrap_or_else(|| format!("Region {}", origin));
    let dest_name = state
        .overworld
        .regions
        .get(destination)
        .map(|r| r.name.clone())
        .unwrap_or_else(|| format!("Region {}", destination));

    let cause_str = match cause {
        MigrationCause::War => "War",
        MigrationCause::Famine => "Famine",
        MigrationCause::Crisis => "Crisis",
        MigrationCause::Persecution => "Persecution",
        MigrationCause::Opportunity => "Opportunity",
    };

    events.push(WorldEvent::MigrationStarted {
        from: origin_name,
        to: dest_name,
        count,
        cause: cause_str.to_string(),
    });

    state.migrations.push(MigrationWave {
        id,
        origin_region_id: origin,
        destination_region_id: destination,
        population_count: count,
        cause,
        started_tick: state.tick,
        progress: 0.0,
    });
}
