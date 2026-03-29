use std::process::ExitCode;

use bevy_game::world_sim::*;
use bevy_game::world_sim::runtime::WorldSim;

use super::cli::WorldSimArgs;

pub fn run_world_sim(args: WorldSimArgs) -> ExitCode {
    let state = build_world(&args);
    let npc_count = state.entities.iter().filter(|e| e.kind == EntityKind::Npc).count();
    let monster_count = state.entities.iter().filter(|e| e.kind == EntityKind::Monster).count();
    println!("=== World Sim ===");
    println!("Entities: {} ({} NPCs, {} monsters)", state.entities.len(), npc_count, monster_count);
    println!("Settlements: {}", state.settlements.len());
    println!("Regions: {}", state.regions.len());
    println!("Factions: {}", state.factions.len());
    println!("Trade routes: {}", state.trade_routes.len());
    println!("Grids: {}", state.grids.len());
    if let Some(dur) = args.duration_secs {
        println!("Duration: {}s", dur);
    } else {
        println!("Ticks: {}", args.ticks);
    }
    if args.rich { println!("Mode: resource-rich"); }
    println!();

    // World generation summary
    print_world_summary(&state);

    let mut sim = WorldSim::new(state);
    let mut last_profile = TickProfile::default();
    let mut total_ticks = 0u64;

    let wall_start = std::time::Instant::now();
    let deadline = args.duration_secs.map(|s| wall_start + std::time::Duration::from_secs(s));
    let max_ticks = if deadline.is_some() { u64::MAX } else { args.ticks };

    let mut next_report = std::time::Instant::now() + std::time::Duration::from_secs(10);

    loop {
        if total_ticks >= max_ticks { break; }
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl { break; }
        }

        let profile = sim.tick();
        total_ticks += 1;

        // Report every 10 seconds of wall time.
        let now = std::time::Instant::now();
        if now >= next_report {
            let elapsed = wall_start.elapsed().as_secs_f64();
            let alive = sim.state().entities.iter().filter(|e| e.alive).count();
            let tps = total_ticks as f64 / elapsed;
            let game_hours = (total_ticks as f64 * 0.1) / 3600.0;
            println!("[{:.0}s] tick {} ({:.1} game-hours) | alive: {} | {:.0} ticks/sec | last: {}µs",
                elapsed, total_ticks, game_hours, alive, tps, profile.total_us);
            next_report = now + std::time::Duration::from_secs(10);
        }

        last_profile = profile;
    }
    let wall_elapsed = wall_start.elapsed();

    println!("\n=== Results ===");
    println!("{}", sim.profile_acc);
    let tps = total_ticks as f64 / wall_elapsed.as_secs_f64();
    let game_hours = (total_ticks as f64 * 0.1) / 3600.0;
    println!("Wall time: {:.2}s | {} ticks ({:.1} game-hours) | {:.0} ticks/sec",
        wall_elapsed.as_secs_f64(), total_ticks, game_hours, tps);

    println!("\n--- Last tick breakdown ---");
    println!("  Compute: {}µs", last_profile.compute_us);
    println!("  Merge: {}µs ({} deltas)", last_profile.merge_us, last_profile.delta_count);
    println!("  Apply: {}µs (hp:{}µs mv:{}µs econ:{}µs)",
        last_profile.apply_us, last_profile.apply_hp_us, last_profile.apply_movement_us,
        last_profile.apply_economy_us);

    // World summary
    let s = sim.state();
    let alive_npcs = s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc).count();
    let alive_monsters = s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Monster).count();
    println!("\n--- World state ---");
    println!("Alive: {} NPCs, {} monsters", alive_npcs, alive_monsters);

    let total_food: f32 = s.settlements.iter().map(|s| s.stockpile[0]).sum();
    let total_treasury: f32 = s.settlements.iter().map(|s| s.treasury).sum();
    let total_gold: f32 = s.entities.iter()
        .filter_map(|e| e.npc.as_ref())
        .map(|n| n.gold)
        .sum();
    println!("Economy: {:.0} food, {:.0} treasury, {:.0} NPC gold", total_food, total_treasury, total_gold);

    // Top/bottom settlements
    let mut settlements: Vec<&SettlementState> = s.settlements.iter().collect();
    settlements.sort_by(|a, b| b.treasury.partial_cmp(&a.treasury).unwrap());
    println!("\nRichest settlements:");
    for s in settlements.iter().take(5) {
        println!("  {} ({}) — food:{:.0} iron:{:.0} wood:{:.0} treasury:{:.0}",
            s.name, s.specialty, s.stockpile[0], s.stockpile[1], s.stockpile[2], s.treasury);
    }
    if settlements.len() > 5 {
        println!("Poorest settlements:");
        for s in settlements.iter().rev().take(3) {
            println!("  {} ({}) — food:{:.0} iron:{:.0} wood:{:.0} treasury:{:.0}",
                s.name, s.specialty, s.stockpile[0], s.stockpile[1], s.stockpile[2], s.treasury);
        }
    }

    ExitCode::SUCCESS
}

/// Print a summary of the generated world before simulation starts.
fn print_world_summary(state: &WorldState) {
    println!("--- Terrain distribution ---");
    for terrain in Terrain::ALL {
        let count = state.regions.iter().filter(|r| r.terrain == terrain).count();
        if count > 0 {
            let region_names: Vec<&str> = state.regions.iter()
                .filter(|r| r.terrain == terrain)
                .map(|r| r.name.as_str())
                .collect();
            println!("  {}: {} region(s) — {}", terrain, count, region_names.join(", "));
        }
    }

    println!("\n--- Faction territories ---");
    for faction in &state.factions {
        let settlements: Vec<&str> = state.settlements.iter()
            .filter(|s| s.faction_id == Some(faction.id))
            .map(|s| s.name.as_str())
            .collect();
        let stance = match faction.diplomatic_stance {
            DiplomaticStance::Friendly => "friendly",
            DiplomaticStance::Neutral => "neutral",
            DiplomaticStance::Hostile => "hostile",
            DiplomaticStance::AtWar => "at war",
            DiplomaticStance::Coalition => "coalition",
        };
        println!("  {} (id:{}, {}) — {} settlement(s): {}",
            faction.name, faction.id, stance,
            settlements.len(), settlements.join(", "));
    }

    println!("\n--- Settlement specialties ---");
    for spec in SettlementSpecialty::ALL {
        let count = state.settlements.iter().filter(|s| s.specialty == spec).count();
        if count > 0 {
            let names: Vec<&str> = state.settlements.iter()
                .filter(|s| s.specialty == spec)
                .map(|s| s.name.as_str())
                .collect();
            println!("  {}: {} — {}", spec, count, names.join(", "));
        }
    }

    if !state.trade_routes.is_empty() {
        println!("\n--- Trade routes ---");
        for &(a, b) in &state.trade_routes {
            let name_a = state.settlements.iter().find(|s| s.id == a).map(|s| s.name.as_str()).unwrap_or("?");
            let name_b = state.settlements.iter().find(|s| s.id == b).map(|s| s.name.as_str()).unwrap_or("?");
            println!("  {} <-> {}", name_a, name_b);
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Simple LCG helpers for deterministic world generation
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn lcg_u32(state: &mut u64) -> u32 {
    lcg_next(state);
    (*state >> 32) as u32
}

fn lcg_f32(state: &mut u64) -> f32 {
    lcg_u32(state) as f32 / u32::MAX as f32
}

fn lcg_range(state: &mut u64, lo: usize, hi: usize) -> usize {
    if hi <= lo { return lo; }
    lo + (lcg_u32(state) as usize % (hi - lo))
}

// ---------------------------------------------------------------------------
// Terrain assignment: simple 2D hash noise
// ---------------------------------------------------------------------------

fn assign_terrain(col: usize, row: usize, rng: &mut u64) -> Terrain {
    // Mix position into RNG for spatial coherence with some randomness.
    let hash = (col as u64).wrapping_mul(2654435761)
        ^ (row as u64).wrapping_mul(40503)
        ^ *rng;
    let idx = (hash % 7) as usize;
    Terrain::ALL[idx]
}

// ---------------------------------------------------------------------------
// Settlement specialty selection based on terrain
// ---------------------------------------------------------------------------

fn choose_specialty(terrain: Terrain, rng: &mut u64) -> SettlementSpecialty {
    // Terrain-weighted specialty selection.
    let candidates: &[SettlementSpecialty] = match terrain {
        Terrain::Plains => &[
            SettlementSpecialty::FarmingVillage,
            SettlementSpecialty::FarmingVillage,
            SettlementSpecialty::General,
            SettlementSpecialty::TradeHub,
            SettlementSpecialty::MilitaryOutpost,
        ],
        Terrain::Forest => &[
            SettlementSpecialty::General,
            SettlementSpecialty::CraftingGuild,
            SettlementSpecialty::MilitaryOutpost,
            SettlementSpecialty::ScholarCity,
        ],
        Terrain::Mountains => &[
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::MilitaryOutpost,
            SettlementSpecialty::CraftingGuild,
        ],
        Terrain::Coast => &[
            SettlementSpecialty::PortTown,
            SettlementSpecialty::PortTown,
            SettlementSpecialty::TradeHub,
            SettlementSpecialty::FarmingVillage,
        ],
        Terrain::Swamp => &[
            SettlementSpecialty::CraftingGuild,
            SettlementSpecialty::General,
            SettlementSpecialty::ScholarCity,
        ],
        Terrain::Desert => &[
            SettlementSpecialty::TradeHub,
            SettlementSpecialty::MiningTown,
            SettlementSpecialty::ScholarCity,
            SettlementSpecialty::MilitaryOutpost,
        ],
        Terrain::Tundra => &[
            SettlementSpecialty::MilitaryOutpost,
            SettlementSpecialty::General,
            SettlementSpecialty::MiningTown,
        ],
    };
    let idx = lcg_range(rng, 0, candidates.len());
    candidates[idx]
}

// ---------------------------------------------------------------------------
// Faction name generation
// ---------------------------------------------------------------------------

const FACTION_PREFIXES: &[&str] = &[
    "Iron", "Silver", "Golden", "Shadow", "Storm", "Crimson", "Emerald", "Azure",
    "Obsidian", "Ivory", "Scarlet", "Jade", "Onyx", "Amber", "Cobalt", "Marble",
];

const FACTION_SUFFIXES: &[&str] = &[
    "Dominion", "Alliance", "Compact", "League", "Order", "Pact", "Realm", "Confederacy",
    "Republic", "Empire", "Syndicate", "Covenant", "Tribunal", "Council", "Principality",
];

fn generate_faction_name(rng: &mut u64) -> String {
    let prefix = FACTION_PREFIXES[lcg_range(rng, 0, FACTION_PREFIXES.len())];
    let suffix = FACTION_SUFFIXES[lcg_range(rng, 0, FACTION_SUFFIXES.len())];
    format!("{} {}", prefix, suffix)
}

// ---------------------------------------------------------------------------
// Settlement name generation
// ---------------------------------------------------------------------------

const SETTLEMENT_PREFIXES: &[&str] = &[
    "Oak", "Iron", "River", "Stone", "Green", "Red", "North", "South", "East", "West",
    "High", "Low", "Dark", "Bright", "Old", "New", "White", "Black", "Grey", "Blue",
    "Pine", "Elm", "Ash", "Birch", "Cedar", "Maple", "Willow", "Holly",
];

const SETTLEMENT_SUFFIXES: &[&str] = &[
    "haven", "ford", "bridge", "gate", "watch", "hold", "stead", "dale", "vale",
    "port", "keep", "wick", "bury", "moor", "fell", "marsh", "town", "field",
    "grove", "wood", "crest", "peak", "hollow", "bend", "crossing",
];

fn generate_settlement_name(rng: &mut u64) -> String {
    let prefix = SETTLEMENT_PREFIXES[lcg_range(rng, 0, SETTLEMENT_PREFIXES.len())];
    let suffix = SETTLEMENT_SUFFIXES[lcg_range(rng, 0, SETTLEMENT_SUFFIXES.len())];
    format!("{}{}", prefix, suffix)
}

// ---------------------------------------------------------------------------
// Region name generation
// ---------------------------------------------------------------------------

const REGION_NAMES: &[&str] = &[
    "Ashenvale", "Thornmarch", "Windrift", "Sunreach", "Mistwood", "Frostpeak",
    "Dusthollow", "Irondeep", "Greenmantle", "Stormveil", "Shadowfen", "Goldfields",
    "Silverbark", "Blackmoor", "Crystalshore", "Dragonspine", "Serpentine Reach",
    "Ember Wastes", "Coral Basin", "Howling Steppe", "Silent Vale", "Thundercleft",
    "Pilgrim's Rest", "Wraith Hollow", "Dawn Plateau", "Twilight Fen",
];

fn generate_region_name(idx: usize, rng: &mut u64) -> String {
    if idx < REGION_NAMES.len() {
        REGION_NAMES[idx].to_string()
    } else {
        // Fallback for large worlds.
        format!("Region_{}", lcg_u32(rng) % 10000)
    }
}

// ---------------------------------------------------------------------------
// build_world — terrain-aware, faction-controlled world generation
// ---------------------------------------------------------------------------

fn build_world(args: &WorldSimArgs) -> WorldState {
    let mut state = WorldState::new(args.seed);
    let npcs = args.entities.saturating_sub(args.monsters);
    let rich = args.rich;
    let num_factions = args.factions.max(1);

    let terrain_seed = args.terrain_seed.unwrap_or(args.seed);
    let mut rng = terrain_seed;

    // -----------------------------------------------------------------------
    // 1. Create regions with terrain types
    // -----------------------------------------------------------------------
    let cols = (args.settlements as f32).sqrt().ceil() as usize;
    let rows = (args.settlements + cols - 1) / cols;
    let spacing = if rich { 200.0 } else { 100.0 };

    // One region per grid cell.
    let num_regions = cols * rows;
    for i in 0..num_regions {
        let row = i / cols;
        let col = i % cols;
        let terrain = assign_terrain(col, row, &mut rng);
        state.regions.push(RegionState {
            id: i as u32,
            name: generate_region_name(i, &mut rng),
            terrain,
            monster_density: match terrain {
                Terrain::Swamp | Terrain::Tundra => 0.6 + lcg_f32(&mut rng) * 0.3,
                Terrain::Mountains | Terrain::Desert => 0.4 + lcg_f32(&mut rng) * 0.3,
                Terrain::Forest => 0.3 + lcg_f32(&mut rng) * 0.3,
                _ => 0.1 + lcg_f32(&mut rng) * 0.2,
            },
            faction_id: None, // assigned after factions are created
            threat_level: 0.0,
            unrest: lcg_f32(&mut rng) * 0.2,
            control: 0.5 + lcg_f32(&mut rng) * 0.5,
        });
    }

    // -----------------------------------------------------------------------
    // 2. Create factions
    // -----------------------------------------------------------------------
    let stances = [
        DiplomaticStance::Friendly,
        DiplomaticStance::Neutral,
        DiplomaticStance::Neutral,
        DiplomaticStance::Hostile,
        DiplomaticStance::Neutral,
        DiplomaticStance::Coalition,
    ];

    for f in 0..num_factions {
        let stance = stances[f % stances.len()];
        let relationship = match stance {
            DiplomaticStance::Friendly => 30.0 + lcg_f32(&mut rng) * 40.0,
            DiplomaticStance::Hostile => -60.0 + lcg_f32(&mut rng) * 30.0,
            DiplomaticStance::AtWar => -80.0 + lcg_f32(&mut rng) * 20.0,
            DiplomaticStance::Coalition => 50.0 + lcg_f32(&mut rng) * 30.0,
            DiplomaticStance::Neutral => -10.0 + lcg_f32(&mut rng) * 20.0,
        };
        state.factions.push(FactionState {
            id: f as u32,
            name: generate_faction_name(&mut rng),
            relationship_to_guild: relationship,
            military_strength: 20.0 + lcg_f32(&mut rng) * 80.0,
            max_military_strength: 100.0,
            territory_size: 0, // computed after assignment
            diplomatic_stance: stance,
            treasury: 500.0 + lcg_f32(&mut rng) * 2000.0,
            at_war_with: Vec::new(),
            coup_risk: lcg_f32(&mut rng) * 0.3,
            escalation_level: 0,
            tech_level: 1 + (lcg_u32(&mut rng) % 3) as u32,
            recent_actions: Vec::new(),
        });
    }

    // Set some faction wars: hostile factions fight each other.
    {
        let hostile_ids: Vec<u32> = state.factions.iter()
            .filter(|f| f.diplomatic_stance == DiplomaticStance::Hostile)
            .map(|f| f.id)
            .collect();
        if hostile_ids.len() >= 2 {
            // First hostile faction is at war with the second.
            let a = hostile_ids[0];
            let b = hostile_ids[1 % hostile_ids.len()];
            if a != b {
                if let Some(fa) = state.faction_mut(a) { fa.at_war_with.push(b); }
                if let Some(fb) = state.faction_mut(b) { fb.at_war_with.push(a); }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. Create settlements with terrain-aware specialties
    // -----------------------------------------------------------------------
    for i in 0..args.settlements {
        let row = i / cols;
        let col = i % cols;
        let pos = (col as f32 * spacing, row as f32 * spacing);
        let region_idx = row * cols + col;
        let region_idx = region_idx.min(state.regions.len().saturating_sub(1));
        let terrain = state.regions[region_idx].terrain;

        let name = generate_settlement_name(&mut rng);
        let mut settlement = SettlementState::new(i as u32, name, pos);

        // Assign specialty based on terrain.
        settlement.specialty = choose_specialty(terrain, &mut rng);

        // Assign faction ownership: distribute settlements round-robin among factions.
        let faction_id = (i % num_factions) as u32;
        settlement.faction_id = Some(faction_id);

        // Set production rates based on terrain + specialty.
        // Base stockpile from terrain.
        let base_amount = if rich { 500.0 } else { 100.0 };
        for &(commodity, mult) in terrain.primary_commodities() {
            settlement.stockpile[commodity] = base_amount * mult;
        }
        // Specialty bonuses on top.
        for &(commodity, mult) in settlement.specialty.production_bonuses() {
            settlement.stockpile[commodity] += base_amount * (mult - 1.0);
        }

        // Set prices: cheap what you produce, expensive what you don't.
        for c in 0..NUM_COMMODITIES {
            let terrain_produces = terrain.primary_commodities().iter().any(|&(ci, _)| ci == c);
            let specialty_produces = settlement.specialty.production_bonuses().iter().any(|&(ci, _)| ci == c);
            settlement.prices[c] = if terrain_produces && specialty_produces {
                0.4 + lcg_f32(&mut rng) * 0.2
            } else if terrain_produces || specialty_produces {
                0.7 + lcg_f32(&mut rng) * 0.3
            } else {
                1.5 + lcg_f32(&mut rng) * 1.0
            };
        }

        // Treasury based on specialty and terrain.
        settlement.treasury = match settlement.specialty {
            SettlementSpecialty::TradeHub | SettlementSpecialty::PortTown => {
                200.0 + lcg_f32(&mut rng) * 500.0
            }
            SettlementSpecialty::MiningTown | SettlementSpecialty::CraftingGuild => {
                150.0 + lcg_f32(&mut rng) * 300.0
            }
            _ => 50.0 + lcg_f32(&mut rng) * 200.0,
        };
        if rich { settlement.treasury *= 3.0; }

        settlement.population = (npcs / args.settlements) as u32;

        // Also tag the region with the faction.
        if region_idx < state.regions.len() && state.regions[region_idx].faction_id.is_none() {
            state.regions[region_idx].faction_id = Some(faction_id);
        }

        state.settlements.push(settlement);

        state.grids.push(LocalGrid {
            id: i as u32,
            fidelity: Fidelity::Medium,
            center: pos,
            radius: if rich { 50.0 } else { 30.0 },
            entity_ids: Vec::new(),
        });
    }

    // Update faction territory sizes.
    for faction in &mut state.factions {
        faction.territory_size = state.settlements.iter()
            .filter(|s| s.faction_id == Some(faction.id))
            .count() as u32;
    }

    // -----------------------------------------------------------------------
    // 4. Trade routes between nearby settlements
    // -----------------------------------------------------------------------
    if args.trade_routes {
        let max_dist = spacing * 1.8; // connect settlements within ~2 grid cells
        for i in 0..state.settlements.len() {
            for j in (i + 1)..state.settlements.len() {
                let a = &state.settlements[i];
                let b = &state.settlements[j];
                let dx = a.pos.0 - b.pos.0;
                let dy = a.pos.1 - b.pos.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= max_dist {
                    state.trade_routes.push((a.id, b.id));
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 5. Distribute NPCs with specialty-aware archetypes
    // -----------------------------------------------------------------------
    let mut id = 0u32;

    for i in 0..npcs {
        let settlement_idx = i % args.settlements;
        let s = &state.settlements[settlement_idx];
        let specialty = s.specialty;
        let settlement_faction = s.faction_id;
        let settlement_treasury = s.treasury;

        let jitter_x = ((id as f32 * 7.13).sin()) * 5.0;
        let jitter_y = ((id as f32 * 3.71).cos()) * 5.0;

        let mut npc = Entity::new_npc(id, (s.pos.0 + jitter_x, s.pos.1 + jitter_y));
        npc.grid_id = Some(settlement_idx as u32);
        npc.level = 1 + (id % 10) as u32;

        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(settlement_idx as u32);
        npc_data.faction_id = settlement_faction;

        // Choose archetype matching settlement specialty.
        let archetypes = specialty.preferred_archetypes();
        npc_data.archetype = archetypes[id as usize % archetypes.len()].to_string();

        // Set production based on terrain region.
        let region_idx = settlement_idx.min(state.regions.len().saturating_sub(1));
        let terrain = if region_idx < state.regions.len() {
            state.regions[region_idx].terrain
        } else {
            Terrain::Plains
        };

        if rich {
            // Varied production: terrain + specialty influenced.
            let terrain_commodities = terrain.primary_commodities();
            if !terrain_commodities.is_empty() {
                let primary = terrain_commodities[id as usize % terrain_commodities.len()];
                npc_data.behavior_production = vec![(primary.0, 0.15)];
                // Add a secondary from specialty if available.
                let spec_bonuses = specialty.production_bonuses();
                if !spec_bonuses.is_empty() {
                    let secondary = spec_bonuses[id as usize % spec_bonuses.len()];
                    if secondary.0 != primary.0 {
                        npc_data.behavior_production.push((secondary.0, 0.05));
                    }
                }
            } else {
                // Sparse terrain: low production.
                npc_data.behavior_production = vec![(0, 0.05)];
            }

            // Varied starting gold based on settlement wealth.
            let wealth_factor = settlement_treasury / 200.0;
            npc_data.gold = 5.0 + wealth_factor * lcg_f32(&mut rng) * 20.0;
            npc_data.morale = 30.0 + lcg_f32(&mut rng) * 50.0;
            npc_data.loyalty = 20.0 + lcg_f32(&mut rng) * 60.0;
        } else {
            let terrain_commodities = terrain.primary_commodities();
            if !terrain_commodities.is_empty() {
                let primary = terrain_commodities[id as usize % terrain_commodities.len()];
                npc_data.behavior_production = vec![(primary.0, 0.1)];
            } else {
                npc_data.behavior_production = vec![(0, 0.05)];
            }
            npc_data.gold = 2.0 + lcg_f32(&mut rng) * 8.0;
            npc_data.morale = 40.0 + lcg_f32(&mut rng) * 30.0;
            npc_data.loyalty = 30.0 + lcg_f32(&mut rng) * 40.0;
        }

        state.grids[settlement_idx].entity_ids.push(id);
        state.entities.push(npc);
        id += 1;
    }

    // -----------------------------------------------------------------------
    // 6. Spawn monsters in the wilderness (density weighted by region)
    // -----------------------------------------------------------------------
    let max_extent = (cols as f32) * spacing;
    for i in 0..args.monsters {
        let angle = (i as f32 / args.monsters.max(1) as f32) * std::f32::consts::TAU;
        let base_radius = max_extent * 0.6 + lcg_f32(&mut rng) * max_extent * 0.4;
        let cx = max_extent / 2.0;
        let cy = max_extent / 2.0;
        let pos = (cx + base_radius * angle.cos(), cy + base_radius * angle.sin());
        let level = 1 + (i % 10) as u32;
        let mut monster = Entity::new_monster(id, pos, level);
        if rich {
            monster.hp *= 2.0;
            monster.max_hp *= 2.0;
            monster.attack_damage *= 1.5;
        }
        state.entities.push(monster);
        id += 1;
    }

    state
}
