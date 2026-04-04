use std::process::ExitCode;

use super::cli::{BuildingAiCommand, BuildingAiSubcommand};
use game::world_sim::building_ai::mass_gen;
use game::world_sim::building_ai::oracle;
use game::world_sim::building_ai::scenario_config;
use game::world_sim::building_ai::scenario_gen;
use game::world_sim::building_ai::scoring;
use game::world_sim::building_ai::types::DecisionTier;
use game::world_sim::building_ai::validation;
use game::world_sim::state::{EntityKind, WorldState, WorldTeam};

fn print_diagnostics(state: &WorldState) {
    println!("\n=== Spatial Diagnostics ===");

    let settlement_pos = state.settlements.first().map(|s| s.pos).unwrap_or((0.0, 0.0));

    // 1. Building positions — are they spread out or clustered at origin?
    println!("\n[1] Building Positions");
    let buildings: Vec<_> = state.entities.iter()
        .filter(|e| e.building.is_some())
        .collect();
    let mut at_origin = 0;
    let mut spread = 0;
    for b in &buildings {
        let bd = b.building.as_ref().unwrap();
        let dist = ((b.pos.0 - settlement_pos.0).powi(2) + (b.pos.1 - settlement_pos.1).powi(2)).sqrt();
        if dist < 0.1 {
            at_origin += 1;
        } else {
            spread += 1;
        }
        if buildings.len() <= 30 {
            println!(
                "  {:20} id={:<3} grid=({:>2},{:>2}) world=({:>7.1},{:>7.1}) dist={:.1}",
                bd.name, b.id, bd.grid_col, bd.grid_row, b.pos.0, b.pos.1, dist,
            );
        }
    }
    println!("  {} at origin, {} spread out (total {})", at_origin, spread, buildings.len());
    if at_origin > 0 {
        println!("  FAIL: {} buildings stuck at settlement origin", at_origin);
    } else {
        println!("  PASS: all buildings have distinct world positions");
    }

    // 2. NPC positions — are they near buildings or at origin?
    println!("\n[2] NPC Positions");
    let npcs: Vec<_> = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Npc && e.alive)
        .collect();
    let mut npc_at_origin = 0;
    for npc in &npcs {
        let dist = ((npc.pos.0 - settlement_pos.0).powi(2) + (npc.pos.1 - settlement_pos.1).powi(2)).sqrt();
        if dist < 0.1 { npc_at_origin += 1; }

        let name = npc.npc.as_ref().map(|n| n.name.as_str()).unwrap_or("?");
        println!(
            "  {:15} id={:<3} pos=({:>7.1},{:>7.1}) hp={:.0}/{:.0}",
            name, npc.id, npc.pos.0, npc.pos.1, npc.hp, npc.max_hp,
        );
    }
    println!("  {} at origin (total {})", npc_at_origin, npcs.len());

    // 3. Monster positions — are they outside walls? Did walls block them?
    println!("\n[3] Monster Positions");
    let monsters: Vec<_> = state.entities.iter()
        .filter(|e| e.kind == EntityKind::Monster || e.team == WorldTeam::Hostile)
        .collect();
    let alive_monsters = monsters.iter().filter(|e| e.alive).count();
    let dead_monsters = monsters.iter().filter(|e| !e.alive).count();
    let mut monster_near_settlement = 0;
    for m in &monsters {
        let dist = ((m.pos.0 - settlement_pos.0).powi(2) + (m.pos.1 - settlement_pos.1).powi(2)).sqrt();
        if dist < 50.0 { monster_near_settlement += 1; }
        if monsters.len() <= 30 {
            let name = game::world_sim::naming::entity_display_name(m);
            println!(
                "  {:25} id={:<3} pos=({:>7.1},{:>7.1}) dist={:>5.1} alive={}",
                name, m.id, m.pos.0, m.pos.1, dist, m.alive,
            );
        }
    }
    println!("  {} alive, {} dead, {} near settlement (<50 units)", alive_monsters, dead_monsters, monster_near_settlement);

    // 4. NPC work/home assignments
    println!("\n[4] NPC Assignments");
    let mut has_home = 0;
    let mut has_work = 0;
    let mut unassigned = 0;
    for e in &state.entities {
        if e.kind != EntityKind::Npc || !e.alive { continue; }
        if let Some(npc) = &e.npc {
            let home = npc.home_building_id.is_some();
            let work = npc.work_building_id.is_some();
            if home { has_home += 1; }
            if work { has_work += 1; }
            if !home && !work { unassigned += 1; }
        }
    }
    let total_npcs = npcs.len();
    println!("  Home: {}/{} | Work: {}/{} | Unassigned: {}", has_home, total_npcs, has_work, total_npcs, unassigned);
}

fn print_validation(label: &str, errors: &[validation::ValidationError]) {
    if errors.is_empty() {
        println!("{} validation: PASS", label);
    } else {
        let fatal = validation::fatal_errors(errors);
        println!(
            "{} validation: {} fatal, {} warnings",
            label,
            fatal.len(),
            errors.len() - fatal.len()
        );
        for e in errors {
            println!("  {}", e);
        }
    }
}

pub fn run_building_ai(cmd: BuildingAiCommand) -> ExitCode {
    match cmd.sub {
        BuildingAiSubcommand::Run(args) => {
            let scenario = match scenario_config::load_scenario(&args.scenario) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to load scenario: {}", e);
                    return ExitCode::FAILURE;
                }
            };

            println!("=== {} ===", scenario.meta.name);
            println!("{}", scenario.meta.description);
            println!(
                "Tags: [{}]",
                scenario.meta.tags.join(", ")
            );
            println!(
                "Challenges: {}  |  Sim ticks: {}  |  Baselines: {}",
                scenario.challenges.len(),
                scenario.sim_ticks,
                scenario.num_random_baselines
            );
            println!();

            // Build world state from seed.
            let resolved = scenario_gen::resolve_seed(&scenario.seed, &args.base_dir);
            let mut state = scenario_gen::generate_from_seed(&resolved, 42);

            // Inject challenges.
            let mut challenges = Vec::new();
            for ccfg in &scenario.challenges {
                let challenge = scenario_gen::resolve_challenge(ccfg, &args.base_dir);
                scenario_gen::inject_challenge(&mut state, &challenge);
                challenges.push(challenge);
            }

            let settlement_id = state.settlements.first().map(|s| s.id).unwrap_or(1);
            let building_count = state.entities.iter()
                .filter(|e| e.building.is_some())
                .count();
            let npc_count = state.entities.iter()
                .filter(|e| e.npc.is_some() && e.alive)
                .count();
            println!(
                "WorldState: {} buildings, {} NPCs, {} entities total",
                building_count, npc_count, state.entities.len()
            );

            // Phase 1: Strategic observation + oracle.
            let memory = if let Some(c) = challenges.first() {
                scenario_gen::populate_memory(&state, c, settlement_id)
            } else {
                game::world_sim::building_ai::types::ConstructionMemory::new()
            };
            let spatial = game::world_sim::building_ai::features::compute_spatial_features(&state, settlement_id);
            let strat_obs = scenario_gen::build_observation(
                &state, settlement_id, &challenges, &memory, &spatial,
                DecisionTier::Strategic,
            );

            println!("\n--- Strategic Tier ---");
            println!(
                "Housing pressure: {:.2}  |  Components: {}  |  Challenges: {}",
                strat_obs.spatial.population.housing_pressure,
                strat_obs.spatial.connectivity.connected_components,
                strat_obs.challenges.len(),
            );

            let strat_actions = oracle::strategic_oracle(&strat_obs);
            println!("Oracle: {} strategic actions", strat_actions.len());
            for a in &strat_actions {
                println!("  {:?}", a.action);
            }

            // Validate strategic actions.
            if args.validate {
                let errors = validation::validate_all(&state, &strat_obs, &strat_actions);
                print_validation("Strategic", &errors);
            }

            // All strategic actions pass through — CityGrid cell-state filtering has been removed.
            // VoxelWorld collision checks happen during apply_actions instead.
            let valid_strat_actions: Vec<_> = strat_actions.iter().enumerate().collect::<Vec<_>>();

            let filtered_actions: Vec<_> = valid_strat_actions.iter().map(|(_, a)| (*a).clone()).collect();
            let filtered_indices: Vec<usize> = valid_strat_actions.iter().map(|(i, _)| *i).collect();
            println!("Applying {} of {} strategic actions", filtered_actions.len(), strat_actions.len());

            // Phase 2: Apply filtered strategic actions, then structural.
            let max_id_before = state.entities.iter().map(|e| e.id).max().unwrap_or(0);
            scoring::apply_actions(&mut state, &filtered_actions);
            state.rebuild_all_indices();

            // Map synthetic IDs using original indices (oracle structural refs use original indices).
            let mut id_map = std::collections::HashMap::new();
            let mut new_id = max_id_before + 1;
            for &orig_idx in &filtered_indices {
                if matches!(strat_actions[orig_idx].action, game::world_sim::building_ai::types::ActionPayload::PlaceBuilding { .. }) {
                    id_map.insert(10000 + orig_idx as u32, new_id);
                    new_id += 1;
                }
            }

            let spatial2 = game::world_sim::building_ai::features::compute_spatial_features(&state, settlement_id);
            let struct_obs = scenario_gen::build_observation(
                &state, settlement_id, &challenges, &memory, &spatial2,
                DecisionTier::Structural,
            );

            println!("\n--- Structural Tier ---");
            let struct_actions = oracle::structural_oracle(&struct_obs, &strat_actions);

            // Remap synthetic entity IDs to real ones, dropping actions for filtered-out entities.
            let struct_actions: Vec<_> = struct_actions.into_iter().filter_map(|mut a| {
                use game::world_sim::building_ai::types::ActionPayload;
                match &mut a.action {
                    ActionPayload::SetFootprint { building_id, .. }
                    | ActionPayload::SetVertical { building_id, .. }
                    | ActionPayload::SetInteriorLayout { building_id, .. }
                    | ActionPayload::SetFoundation { building_id, .. }
                    | ActionPayload::SetRoofSpec { building_id, .. }
                    | ActionPayload::SetOpenings { building_id, .. }
                    | ActionPayload::SetMaterial { building_id, .. }
                    | ActionPayload::SetBuildPriority { building_id, .. }
                    | ActionPayload::Demolish { building_id, .. }
                    | ActionPayload::Renovate { building_id, .. } => {
                        if *building_id >= 10000 {
                            let real_id = id_map.get(building_id)?;
                            *building_id = *real_id;
                        }
                    }
                    ActionPayload::SetWallSpec { segment_id, .. } => {
                        if *segment_id >= 10000 {
                            let real_id = id_map.get(segment_id)?;
                            *segment_id = *real_id;
                        }
                    }
                    _ => {}
                }
                Some(a)
            }).collect();

            println!("Oracle: {} structural actions", struct_actions.len());
            for a in &struct_actions {
                println!("  {:?}", a.action);
            }

            // Validate structural actions against updated state.
            if args.validate {
                let errors = validation::validate_all(&state, &struct_obs, &struct_actions);
                print_validation("Structural", &errors);
            }

            // Apply structural actions to state before sim or output.
            scoring::apply_actions(&mut state, &struct_actions);
            state.rebuild_all_indices();

            // Optionally run the world sim on the post-oracle state.
            if let Some(ticks) = args.sim_ticks {
                println!("\n--- World Sim ({} ticks) ---", ticks);

                let mut sim = game::world_sim::runtime::WorldSim::new(state);
                let start = std::time::Instant::now();
                let mut last_chronicle_len = sim.state().chronicle.len();
                let mut last_events_len = sim.state().world_events.len();
                for t in 0..ticks {
                    sim.tick();
                    let s = sim.state();

                    // Print new chronicle entries as they appear.
                    if s.chronicle.len() > last_chronicle_len {
                        for entry in &s.chronicle[last_chronicle_len..] {
                            println!(
                                "  [t{}] {:?}: {}",
                                entry.tick, entry.category, entry.text,
                            );
                        }
                        last_chronicle_len = s.chronicle.len();
                    }

                    // Print new world events.
                    if s.world_events.len() > last_events_len {
                        for event in &s.world_events[last_events_len..] {
                            match event {
                                game::world_sim::state::WorldEvent::EntityDied { entity_id, cause } => {
                                    let name = s.entities.iter()
                                        .find(|e| e.id == *entity_id)
                                        .map(|e| game::world_sim::naming::entity_display_name(e))
                                        .unwrap_or_else(|| format!("Entity #{}", entity_id));
                                    println!("  [t{}] DEATH: {} (id={}) — {}", s.tick, name, entity_id, cause);
                                }
                                game::world_sim::state::WorldEvent::BattleStarted { grid_id, participants } => {
                                    println!("  [t{}] BATTLE START: grid={}, {} combatants", s.tick, grid_id, participants.len());
                                }
                                game::world_sim::state::WorldEvent::BattleEnded { grid_id, victor_team } => {
                                    println!("  [t{}] BATTLE END: grid={}, victor={:?}", s.tick, grid_id, victor_team);
                                }
                                game::world_sim::state::WorldEvent::SeasonChanged { new_season } => {
                                    let name = match new_season { 0 => "Spring", 1 => "Summer", 2 => "Autumn", 3 => "Winter", _ => "?" };
                                    println!("  [t{}] SEASON: {}", s.tick, name);
                                }
                                _ => {}
                            }
                        }
                        last_events_len = s.world_events.len();
                    }

                    // Periodic summary.
                    if (t + 1) % 1000 == 0 || t + 1 == ticks {
                        let alive = s.entities.iter().filter(|e| e.alive).count();
                        let npcs = s.entities.iter().filter(|e| e.npc.is_some() && e.alive).count();
                        let buildings = s.entities.iter().filter(|e| e.building.is_some()).count();
                        println!(
                            "  --- tick {} summary: {} alive ({} NPCs, {} buildings) ---",
                            s.tick, alive, npcs, buildings,
                        );
                    }
                }
                let elapsed = start.elapsed();
                println!("Sim completed in {:.2}s ({:.0} ticks/s)", elapsed.as_secs_f64(), ticks as f64 / elapsed.as_secs_f64());

                if args.diagnostics {
                    print_diagnostics(sim.state());
                }

                // Save final state if requested.
                if let Some(path) = &args.output {
                    match serde_json::to_string(sim.state()) {
                        Ok(json) => {
                            if let Err(e) = std::fs::write(path, json) {
                                eprintln!("Failed to write output: {}", e);
                                return ExitCode::FAILURE;
                            }
                            println!("Saved state to {}", path.display());
                        }
                        Err(e) => {
                            eprintln!("Failed to serialize state: {}", e);
                            return ExitCode::FAILURE;
                        }
                    }
                }
            } else {
                // No sim requested.
                if args.diagnostics {
                    print_diagnostics(&state);
                }
                if let Some(path) = &args.output {
                    match serde_json::to_string(&state) {
                        Ok(json) => {
                            if let Err(e) = std::fs::write(path, json) {
                                eprintln!("Failed to write output: {}", e);
                                return ExitCode::FAILURE;
                            }
                            println!("Saved state to {}", path.display());
                        }
                        Err(e) => {
                            eprintln!("Failed to serialize state: {}", e);
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }

            ExitCode::SUCCESS
        }
        BuildingAiSubcommand::Generate(args) => {
            let config = mass_gen::GenConfig {
                target_pairs: args.pairs,
                min_cell: args.min_cell,
                seed: args.seed,
                output_path: args.output.to_string_lossy().to_string(),
                coverage_path: Some(args.coverage.to_string_lossy().to_string()),
            };
            let report = mass_gen::generate_dataset(&config);
            println!(
                "Generated {} pairs across {} scenarios (min active cell: {})",
                report.total_pairs, report.total_scenarios, report.min_active_cell
            );
            ExitCode::SUCCESS
        }
        BuildingAiSubcommand::Coverage(args) => {
            let report = mass_gen::compute_coverage(&args.dataset);
            println!("Coverage Report");
            println!("===============");
            println!("Total scenarios: {}", report.total_scenarios);
            println!("Total pairs:     {}", report.total_pairs);
            println!("Min active cell: {}", report.min_active_cell);
            println!("Dead cells:      {}", report.dead_cells.len());

            // Print matrix header.
            let categories = [
                "Military", "Environmental", "Economic", "Population", "Temporal",
                "Terrain", "MultiSettlement", "UnitCapability", "HighValueNpc", "LevelScaled",
            ];
            let decisions = [
                "Place", "Prior", "Route", "Zone", "Demol",
                "Foot", "Vert", "Wall", "Roof", "Found",
                "Open", "IntFl", "Room", "Matl", "DefIn",
                "EnvAd", "Expan", "Renov",
            ];

            // Header row.
            print!("{:>14}", "");
            for d in &decisions {
                print!("{:>6}", d);
            }
            println!();

            // Matrix rows.
            for (i, cat) in categories.iter().enumerate() {
                print!("{:>14}", cat);
                for j in 0..18 {
                    let val = report.matrix[i][j];
                    if report.dead_cells.contains(&(i, j)) {
                        print!("{:>6}", "-");
                    } else if val == 0 {
                        print!("{:>6}", ".");
                    } else {
                        print!("{:>6}", val);
                    }
                }
                println!();
            }

            // Output as JSON too.
            if let Ok(json) = serde_json::to_string_pretty(&report) {
                println!("\n{}", json);
            }

            ExitCode::SUCCESS
        }
        BuildingAiSubcommand::FillGaps(args) => {
            let report = mass_gen::fill_gaps(
                &args.dataset,
                args.min_cell,
                &args.output,
                args.seed,
            );
            println!(
                "Generated {} supplemental scenarios (min active cell: {})",
                report.total_scenarios, report.min_active_cell
            );
            ExitCode::SUCCESS
        }
    }
}
