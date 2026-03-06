use crate::ai;

pub fn run_pathing_simulation() {
    let ticks = 420;
    let result = ai::advanced::run_horde_chokepoint_sample(101, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();
    let hero_alive = result
        .final_state
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Hero && u.hp > 0)
        .count();
    let enemy_alive = result
        .final_state
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Enemy && u.hp > 0)
        .count();

    println!("Pathing horde chokepoint simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!(
        "Alive units: {} (hero={}, enemy={})",
        alive_units, hero_alive, enemy_alive
    );
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | First death tick: {:?}",
        result.metrics.winner, result.metrics.tick_to_first_death
    );
    println!("Final HP: {:?}", result.metrics.final_hp_by_unit);
    println!(
        "Casts completed: {} | Heals completed: {} | Repositions: {}",
        result.metrics.casts_completed,
        result.metrics.heals_completed,
        result.metrics.reposition_for_range_events
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_pathing_hero_win_simulation() {
    let ticks = 420;
    let result =
        ai::advanced::run_horde_chokepoint_hero_favored_sample(202, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();
    let hero_alive = result
        .final_state
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Hero && u.hp > 0)
        .count();
    let enemy_alive = result
        .final_state
        .units
        .iter()
        .filter(|u| u.team == ai::core::Team::Enemy && u.hp > 0)
        .count();

    println!("Pathing horde chokepoint simulation (hero favored)");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!(
        "Alive units: {} (hero={}, enemy={})",
        alive_units, hero_alive, enemy_alive
    );
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | First death tick: {:?}",
        result.metrics.winner, result.metrics.tick_to_first_death
    );
    println!("Final HP: {:?}", result.metrics.final_hp_by_unit);
    println!(
        "Casts completed: {} | Heals completed: {} | Repositions: {}",
        result.metrics.casts_completed,
        result.metrics.heals_completed,
        result.metrics.reposition_for_range_events
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_pathing_hero_hp_ablation_simulation() {
    let ticks = 420;
    let scales = [1.0_f32, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
    println!("Pathing horde hero-favored HP ablation");
    println!("Ticks: {}", ticks);
    for scale in scales {
        let result = ai::advanced::run_horde_chokepoint_hero_favored_hp_scaled_sample(
            202,
            ticks,
            ai::core::FIXED_TICK_MS,
            scale,
        );
        let hero_alive = result
            .final_state
            .units
            .iter()
            .filter(|u| u.team == ai::core::Team::Hero && u.hp > 0)
            .count();
        let enemy_alive = result
            .final_state
            .units
            .iter()
            .filter(|u| u.team == ai::core::Team::Enemy && u.hp > 0)
            .count();
        println!(
            "hp_scale={:.2} winner={:?} alive(hero/enemy)={}/{} first_death={:?} event_hash={:016x}",
            scale,
            result.metrics.winner,
            hero_alive,
            enemy_alive,
            result.metrics.tick_to_first_death,
            result.event_log_hash
        );
    }
}
