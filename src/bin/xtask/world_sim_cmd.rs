use std::process::ExitCode;

use bevy_game::world_sim::*;
use bevy_game::world_sim::state::*;
use bevy_game::world_sim::runtime::WorldSim;

use super::cli::WorldSimArgs;

pub fn run_world_sim(args: WorldSimArgs) -> ExitCode {
    let state = build_world(&args);
    let npc_count = state.entities.iter().filter(|e| e.kind == EntityKind::Npc).count();
    let monster_count = state.entities.iter().filter(|e| e.kind == EntityKind::Monster).count();
    println!("=== World Sim ===");
    println!("Entities: {} ({} NPCs, {} monsters)", state.entities.len(), npc_count, monster_count);
    println!("Settlements: {}", state.settlements.len());
    println!("Grids: {}", state.grids.len());
    if let Some(dur) = args.duration_secs {
        println!("Duration: {}s", dur);
    } else {
        println!("Ticks: {}", args.ticks);
    }
    if args.rich { println!("Mode: resource-rich"); }
    println!();

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
        println!("  {} — food:{:.0} iron:{:.0} wood:{:.0} treasury:{:.0}",
            s.name, s.stockpile[0], s.stockpile[1], s.stockpile[2], s.treasury);
    }
    if settlements.len() > 5 {
        println!("Poorest settlements:");
        for s in settlements.iter().rev().take(3) {
            println!("  {} — food:{:.0} iron:{:.0} wood:{:.0} treasury:{:.0}",
                s.name, s.stockpile[0], s.stockpile[1], s.stockpile[2], s.treasury);
        }
    }

    ExitCode::SUCCESS
}

fn build_world(args: &WorldSimArgs) -> WorldState {
    let mut state = WorldState::new(args.seed);
    let npcs = args.entities.saturating_sub(args.monsters);
    let rich = args.rich;

    // Spread settlements in a grid pattern for large counts.
    let cols = (args.settlements as f32).sqrt().ceil() as usize;
    let spacing = if rich { 200.0 } else { 100.0 };

    let mut rng = args.seed;

    for i in 0..args.settlements {
        let row = i / cols;
        let col = i % cols;
        let pos = (col as f32 * spacing, row as f32 * spacing);

        let mut settlement = SettlementState::new(i as u32, format!("Settlement_{i}"), pos);

        if rich {
            // Resource-rich: varied per settlement, high stockpiles.
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let specialty = (rng >> 32) as usize % 8;
            for c in 0..8 {
                settlement.stockpile[c] = if c == specialty { 5000.0 } else { 500.0 };
            }
            settlement.treasury = 1000.0 + (i as f32 * 100.0);
            settlement.population = (npcs / args.settlements) as u32;
            // Varied prices based on specialty (cheap what you produce, expensive what you don't)
            for c in 0..8 {
                settlement.prices[c] = if c == specialty { 0.5 } else { 2.0 + (c as f32) * 0.3 };
            }
        } else {
            settlement.stockpile[0] = 500.0;
            settlement.stockpile[1] = 100.0;
            settlement.stockpile[2] = 200.0;
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

    // Distribute NPCs across settlements with varied production.
    let mut id = 0u32;
    let archetypes = ["knight", "ranger", "mage", "cleric", "rogue", "merchant", "farmer", "smith"];

    for i in 0..npcs {
        let settlement_idx = i % args.settlements;
        let s = &state.settlements[settlement_idx];
        let jitter_x = ((id as f32 * 7.13).sin()) * 5.0;
        let jitter_y = ((id as f32 * 3.71).cos()) * 5.0;

        let mut npc = Entity::new_npc(id, (s.pos.0 + jitter_x, s.pos.1 + jitter_y));
        npc.grid_id = Some(settlement_idx as u32);
        npc.level = 1 + (id % 10) as u32;

        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(settlement_idx as u32);
        npc_data.archetype = archetypes[id as usize % archetypes.len()].to_string();

        if rich {
            // Varied production: each NPC produces 1-2 commodities based on archetype.
            let primary = id as usize % 8;
            let secondary = (id as usize + 3) % 8;
            npc_data.behavior_production = vec![(primary, 0.15), (secondary, 0.05)];
            npc_data.gold = 10.0 + (id % 50) as f32;
            npc_data.morale = 40.0 + (id % 40) as f32;
            npc_data.loyalty = 30.0 + (id % 50) as f32;
        } else {
            let commodity = id as usize % 3;
            npc_data.behavior_production = vec![(commodity, 0.1)];
        }

        state.grids[settlement_idx].entity_ids.push(id);
        state.entities.push(npc);
        id += 1;
    }

    // Spawn monsters in the wilderness.
    for i in 0..args.monsters {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let angle = (i as f32 / args.monsters as f32) * std::f32::consts::TAU;
        let max_extent = (cols as f32) * spacing;
        let radius = max_extent * 0.6 + ((rng >> 32) as f32 / u32::MAX as f32) * max_extent * 0.4;
        let cx = max_extent / 2.0;
        let cy = max_extent / 2.0;
        let pos = (cx + radius * angle.cos(), cy + radius * angle.sin());
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
