use crate::ai;

pub fn run_phase0_simulation() {
    let ticks = 120;
    let initial = ai::core::sample_duel_state(7);
    let script = ai::core::sample_duel_script(ticks);
    let result = ai::core::run_replay(initial, &script, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 0 deterministic simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Per-tick state hashes: {}",
        result.per_tick_state_hashes.len()
    );
    println!(
        "Winner: {:?}, first death tick: {:?}",
        result.metrics.winner, result.metrics.tick_to_first_death
    );
    println!(
        "Ticks elapsed (metric): {} | Final HP: {:?}",
        result.metrics.ticks_elapsed, result.metrics.final_hp_by_unit
    );
    println!(
        "Seconds elapsed: {:.2} | DPS by unit: {:?}",
        result.metrics.seconds_elapsed, result.metrics.dps_by_unit
    );
    println!(
        "Damage dealt: {:?} | Damage taken: {:?}",
        result.metrics.total_damage_by_unit, result.metrics.damage_taken_by_unit
    );
    println!(
        "Overkill: {} | Reposition events: {}",
        result.metrics.overkill_damage_total, result.metrics.reposition_for_range_events
    );
    println!(
        "Casts started/completed/failed: {}/{}/{} | avg cast delay: {:.2} ms",
        result.metrics.casts_started,
        result.metrics.casts_completed,
        result.metrics.casts_failed_out_of_range,
        result.metrics.avg_cast_delay_ms
    );
    println!(
        "Heals started/completed: {}/{} | healing by unit: {:?}",
        result.metrics.heals_started,
        result.metrics.heals_completed,
        result.metrics.total_healing_by_unit
    );
    println!(
        "Attack intents: {} | executed: {} | blocked cooldown: {} | blocked invalid: {} | dead source: {}",
        result.metrics.attack_intents,
        result.metrics.executed_attack_intents,
        result.metrics.blocked_cooldown_intents,
        result.metrics.blocked_invalid_target_intents,
        result.metrics.dead_source_attack_intents
    );
    println!(
        "Range ticks in/out: {:?}/{:?}",
        result.metrics.in_range_ticks_by_unit, result.metrics.out_of_range_ticks_by_unit
    );
    println!(
        "Movement x100: {:?} | chase ticks: {:?}",
        result.metrics.movement_distance_x100_by_unit, result.metrics.chase_ticks_by_unit
    );
    println!(
        "Focus-fire ticks: {} | max targeters: {} | target switches: {:?}",
        result.metrics.focus_fire_ticks,
        result.metrics.max_targeters_on_single_target,
        result.metrics.target_switches_by_unit
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase1_simulation() {
    let ticks = 200;
    let result = ai::utility::run_phase1_sample(11, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 1 utility AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Casts started/completed: {}/{} | avg cast delay: {:.2} ms",
        result.metrics.casts_started,
        result.metrics.casts_completed,
        result.metrics.avg_cast_delay_ms
    );
    println!(
        "Attack intents: {} | executed: {} | blocked cooldown: {} | blocked invalid: {}",
        result.metrics.attack_intents,
        result.metrics.executed_attack_intents,
        result.metrics.blocked_cooldown_intents,
        result.metrics.blocked_invalid_target_intents
    );
    println!(
        "Range ticks in/out: {:?}/{:?}",
        result.metrics.in_range_ticks_by_unit, result.metrics.out_of_range_ticks_by_unit
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase2_simulation() {
    let ticks = 260;
    let result = ai::roles::run_phase2_sample(17, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 2 role-contract AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Damage taken: {:?} | Healing by unit: {:?}",
        result.metrics.damage_taken_by_unit, result.metrics.total_healing_by_unit
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase3_simulation() {
    let ticks = 280;
    let run = ai::squad::run_phase3_sample(23, ticks, ai::core::FIXED_TICK_MS);
    let result = run.replay;
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 3 squad-blackboard AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    if let Some(last_boards) = run.board_history.last() {
        println!(
            "Final hero board: focus={:?} mode={:?}",
            last_boards
                .get(&ai::core::Team::Hero)
                .map(|b| b.focus_target)
                .flatten(),
            last_boards.get(&ai::core::Team::Hero).map(|b| b.mode)
        );
        println!(
            "Final enemy board: focus={:?} mode={:?}",
            last_boards
                .get(&ai::core::Team::Enemy)
                .map(|b| b.focus_target)
                .flatten(),
            last_boards.get(&ai::core::Team::Enemy).map(|b| b.mode)
        );
    }
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase4_simulation() {
    let ticks = 320;
    let run = ai::control::run_phase4_sample(29, ticks, ai::core::FIXED_TICK_MS);
    let result = run.replay;
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 4 CC-reservation AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Reservations observed: {} | control windows observed: {}",
        run.reservation_history
            .iter()
            .filter(|v| !v.is_empty())
            .count(),
        run.control_windows.len()
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase5_simulation() {
    let ticks = 320;
    let run = ai::personality::run_phase5_sample(31, ticks, ai::core::FIXED_TICK_MS);
    let result = run.replay;
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 5 personality AI simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    if let Some(last_modes) = run.mode_history.last() {
        println!("Final modes: {:?}", last_modes);
    }
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase7_simulation() {
    let ticks = 320;
    let result = ai::spatial::run_spatial_sample(37, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 7 spatial reasoning simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase8_simulation() {
    let ticks = 320;
    let result = ai::tactics::run_tactical_sample(37, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 8 encounter-aware tactics simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase9_simulation() {
    let ticks = 320;
    let result = ai::coordination::run_coordination_sample(37, ticks, ai::core::FIXED_TICK_MS);
    let alive_units = result
        .final_state
        .units
        .iter()
        .filter(|unit| unit.hp > 0)
        .count();

    println!("Phase 9 advanced coordination simulation");
    println!("Ticks: {}", ticks);
    println!("Events: {}", result.events.len());
    println!("Alive units: {}", alive_units);
    println!("Event log hash: {:016x}", result.event_log_hash);
    println!("Final state hash: {:016x}", result.final_state_hash);
    println!(
        "Winner: {:?} | Final HP: {:?}",
        result.metrics.winner, result.metrics.final_hp_by_unit
    );
    println!(
        "Casts completed: {} | Heals completed: {}",
        result.metrics.casts_completed, result.metrics.heals_completed
    );
    println!(
        "Invariant violations: {}",
        result.metrics.invariant_violations
    );
}

pub fn run_phase6_report() {
    println!("Phase 6 tooling report");

    println!("\n== Scenario Matrix ==");
    let scenarios = ai::tooling::run_scenario_matrix();
    for s in scenarios {
        println!(
            "{} | winner={} | tfd={:?} | team_ttk={:?} elim={:?} | deaths(h/e)={}/{} | casts={} heals={} | deterministic={} | ehash={:016x} shash={:016x}",
            s.name,
            s.winner,
            s.tick_to_first_death,
            s.team_ttk,
            s.eliminated_team,
            s.hero_deaths,
            s.enemy_deaths,
            s.casts,
            s.heals,
            s.deterministic,
            s.event_hash,
            s.state_hash,
        );
    }

    println!("\n== Decision Debug (first 3 ticks, top-3) ==");
    let debug = ai::tooling::build_phase5_debug(31, 3, 3);
    for tick in debug {
        println!("tick {}", tick.tick);
        for decision in tick.decisions {
            println!(
                "  unit {} mode {:?} chose {:?}",
                decision.unit_id, decision.mode, decision.chosen
            );
            for score in decision.top_k {
                println!(
                    "    score {:>6.2} {:?} reason={}",
                    score.score, score.action, score.reason
                );
            }
        }
    }

    println!("\n== CC Timeline ==");
    println!("{}", ai::tooling::reservation_timeline_summary(29, 320));
    let cc = ai::tooling::analyze_phase4_cc_metrics(29, 320);
    println!(
        "CC metrics: target={:?} windows={} links={} coverage={:.2} overlap={:.2} avg_gap={:.2}",
        cc.primary_target,
        cc.windows,
        cc.links,
        cc.coverage_ratio,
        cc.overlap_ratio,
        cc.avg_gap_ticks
    );

    println!("\n== Tuning Grid (top 5) ==");
    let tuning = ai::tooling::run_personality_grid_tuning();
    for row in tuning.into_iter().take(5) {
        println!(
            "score={} agg={:.2} ctrl={:.2} altru={:.2} winner={} hash={:016x}",
            row.score, row.aggression, row.control_bias, row.altruism, row.winner, row.event_hash
        );
    }
}
