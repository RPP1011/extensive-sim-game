use std::process::ExitCode;

use bevy_game::world_sim::*;
use bevy_game::world_sim::runtime::WorldSim;

use super::cli::WorldSimArgs;

pub fn run_world_sim(args: WorldSimArgs) -> ExitCode {
    let state = build_world(&args);
    println!("=== World Sim Benchmark (zero-alloc runtime) ===");
    println!("Entities: {} ({} NPCs, {} monsters)", state.entities.len(),
        state.entities.iter().filter(|e| e.kind == EntityKind::Npc).count(),
        state.entities.iter().filter(|e| e.kind == EntityKind::Monster).count(),
    );
    println!("Settlements: {}", state.settlements.len());
    println!("Grids: {}", state.grids.len());
    println!("Ticks: {}\n", args.ticks);

    let mut sim = WorldSim::new(state);
    let mut last_profile = TickProfile::default();

    let wall_start = std::time::Instant::now();
    for t in 0..args.ticks {
        let profile = sim.tick();

        if (t + 1) % 1000 == 0 {
            let alive = sim.state().entities.iter().filter(|e| e.alive).count();
            println!("[tick {}] alive: {} | last: {}", t + 1, alive, profile);
        }

        last_profile = profile;
    }
    let wall_elapsed = wall_start.elapsed();

    println!("\n=== Results ===");
    println!("{}", sim.profile_acc);
    println!("Wall time: {:.2}s ({:.1} ticks/sec)",
        wall_elapsed.as_secs_f64(),
        args.ticks as f64 / wall_elapsed.as_secs_f64(),
    );

    println!("\n--- Last tick detailed breakdown ---");
    println!("  Compute: {}µs", last_profile.compute_us);
    println!("    High:   {}µs ({} entities)", last_profile.compute_high_us, last_profile.high_count);
    println!("    Medium: {}µs ({} entities)", last_profile.compute_medium_us, last_profile.medium_count);
    println!("    Low:    {}µs ({} entities)", last_profile.compute_low_us, last_profile.low_count);
    println!("    Grid:   {}µs", last_profile.compute_grid_us);
    println!("  Merge: {}µs ({} deltas)", last_profile.merge_us, last_profile.delta_count);
    println!("  Apply: {}µs", last_profile.apply_us);
    println!("    Clone:    {}µs", last_profile.apply_clone_us);
    println!("    HP:       {}µs", last_profile.apply_hp_us);
    println!("    Movement: {}µs", last_profile.apply_movement_us);
    println!("    Status:   {}µs", last_profile.apply_status_us);
    println!("    Economy:  {}µs", last_profile.apply_economy_us);
    println!("    Deaths:   {}µs", last_profile.apply_deaths_us);
    println!("    Grid:     {}µs", last_profile.apply_grid_us);
    println!("    Fidelity: {}µs", last_profile.apply_fidelity_us);
    println!("    Reports:  {}µs", last_profile.apply_price_reports_us);

    let s = sim.state();
    let alive_npcs = s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Npc).count();
    let alive_monsters = s.entities.iter().filter(|e| e.alive && e.kind == EntityKind::Monster).count();
    println!("\nFinal state: {} alive NPCs, {} alive monsters", alive_npcs, alive_monsters);

    for settlement in &s.settlements {
        println!("  {} — food: {:.0}, treasury: {:.0}", settlement.name, settlement.stockpile[0], settlement.treasury);
    }

    ExitCode::SUCCESS
}

fn build_world(args: &WorldSimArgs) -> WorldState {
    let mut state = WorldState::new(args.seed);
    let npcs = args.entities.saturating_sub(args.monsters);

    for i in 0..args.settlements {
        let angle = (i as f32 / args.settlements as f32) * std::f32::consts::TAU;
        let radius = 100.0;
        let pos = (radius * angle.cos(), radius * angle.sin());

        let mut settlement = SettlementState::new(i as u32, format!("Settlement_{i}"), pos);
        settlement.stockpile[0] = 500.0;
        settlement.stockpile[1] = 100.0;
        settlement.stockpile[2] = 200.0;
        state.settlements.push(settlement);

        state.grids.push(LocalGrid {
            id: i as u32,
            fidelity: Fidelity::Medium,
            center: pos,
            radius: 30.0,
            entity_ids: Vec::new(),
        });
    }

    let mut id = 0u32;
    for i in 0..npcs {
        let settlement_idx = i % args.settlements;
        let s = &state.settlements[settlement_idx];
        let jitter_x = ((id as f32 * 7.13).sin()) * 5.0;
        let jitter_y = ((id as f32 * 3.71).cos()) * 5.0;

        let mut npc = Entity::new_npc(id, (s.pos.0 + jitter_x, s.pos.1 + jitter_y));
        npc.grid_id = Some(settlement_idx as u32);
        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(settlement_idx as u32);
        let commodity = id as usize % 3;
        npc_data.behavior_production = vec![(commodity, 0.1)];

        state.grids[settlement_idx].entity_ids.push(id);
        state.entities.push(npc);
        id += 1;
    }

    for i in 0..args.monsters {
        let angle = (i as f32 / args.monsters as f32) * std::f32::consts::TAU;
        let radius = 150.0 + (i as f32 * 2.37).sin() * 50.0;
        let pos = (radius * angle.cos(), radius * angle.sin());
        let level = (i % 5) as u32 + 1;
        state.entities.push(Entity::new_monster(id, pos, level));
        id += 1;
    }

    state
}
