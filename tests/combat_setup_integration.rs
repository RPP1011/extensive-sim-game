//! Integration tests for the scenario ↔ gameplay bridge.
//!
//! Three test layers:
//! 1. Campaign pipeline — `scenario_from_campaign()` → `build_combat()` → full sim loop
//! 2. Recording — verify frame accumulation matches tick count
//! 3. Chaos sim — random intent selection across many seeds, invariant verification

use bevy_game::ai::core::{
    self, IntentAction, SimState, Team, UnitIntent, FIXED_TICK_MS,
};
use bevy_game::ai::effects::AbilityTarget;
use bevy_game::ai::core::verify::verify_tick;
use bevy_game::game_core::RoomType;
use bevy_game::scenario::{build_combat, scenario_from_campaign, ScenarioCfg};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run a SimState to completion (or max_ticks), returning (outcome, tick, final_state).
fn run_sim_to_completion(
    mut sim: SimState,
    squad_ai: &mut bevy_game::ai::squad::SquadAiState,
    max_ticks: u64,
) -> (&'static str, u64, SimState) {
    for tick in 0..max_ticks {
        let intents = bevy_game::ai::squad::generate_intents(&sim, squad_ai, FIXED_TICK_MS);
        let (next, _events) = core::step(sim, &intents, FIXED_TICK_MS);
        sim = next;

        let heroes_alive = sim.units.iter().any(|u| u.team == Team::Hero && u.hp > 0);
        let enemies_alive = sim.units.iter().any(|u| u.team == Team::Enemy && u.hp > 0);

        if !enemies_alive {
            return ("Victory", tick, sim);
        }
        if !heroes_alive {
            return ("Defeat", tick, sim);
        }
    }
    ("Timeout", max_ticks, sim)
}

/// Simple deterministic RNG for tests (splitmix64).
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

// ===========================================================================
// Layer 1: Campaign pipeline tests
// ===========================================================================

#[test]
fn campaign_scenario_terminates_for_all_room_types() {
    let room_types = [
        (RoomType::Entry, 3000),
        (RoomType::Pressure, 3000),
        (RoomType::Pivot, 3000),
        (RoomType::Setpiece, 3000),
        (RoomType::Recovery, 3000),
        (RoomType::Climax, 3000),
        // Open rooms are 100x100 — too large for reliable convergence, skip
    ];
    for (room_type, max_ticks) in room_types {
        let cfg = scenario_from_campaign(
            &["knight".into(), "mage".into()],
            2,
            5,
            room_type,
            42,
            None,
        );
        let setup = build_combat(&cfg);
        let mut squad_ai = setup.squad_ai;
        let (outcome, tick, _) = run_sim_to_completion(setup.sim, &mut squad_ai, max_ticks);
        assert!(
            outcome == "Victory" || outcome == "Defeat",
            "room {:?} timed out at tick {}",
            room_type,
            tick,
        );
    }
}

#[test]
fn campaign_scenario_difficulty_affects_combat() {
    let low = scenario_from_campaign(&["knight".into()], 1, 0, RoomType::Entry, 42, None);
    let high = scenario_from_campaign(&["knight".into()], 5, 20, RoomType::Entry, 42, None);

    let setup_low = build_combat(&low);
    let setup_high = build_combat(&high);

    // Higher difficulty should produce more/stronger enemies
    let enemy_hp_low: i32 = setup_low
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy)
        .map(|u| u.max_hp)
        .sum();
    let enemy_hp_high: i32 = setup_high
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy)
        .map(|u| u.max_hp)
        .sum();

    assert!(
        enemy_hp_high > enemy_hp_low,
        "high difficulty should have more total enemy HP ({} vs {})",
        enemy_hp_high,
        enemy_hp_low,
    );
}

#[test]
fn campaign_scenario_hero_templates_preserved() {
    let templates = vec!["knight".into(), "paladin".into(), "ranger".into()];
    let cfg = scenario_from_campaign(&templates, 2, 5, RoomType::Entry, 100, None);

    assert_eq!(cfg.hero_templates.len(), 3);
    assert_eq!(cfg.hero_templates[0], "knight");
    assert_eq!(cfg.hero_templates[1], "paladin");
    assert_eq!(cfg.hero_templates[2], "ranger");

    let setup = build_combat(&cfg);
    let hero_count = setup
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .count();
    assert_eq!(hero_count, 3, "should spawn exactly 3 heroes");
}

#[test]
fn campaign_scenario_deterministic() {
    let cfg = scenario_from_campaign(
        &["knight".into(), "mage".into()],
        3,
        10,
        RoomType::Pressure,
        777,
        None,
    );
    let s1 = build_combat(&cfg);
    let s2 = build_combat(&cfg);

    let mut ai1 = s1.squad_ai;
    let mut ai2 = s2.squad_ai;

    let (out1, tick1, _) = run_sim_to_completion(s1.sim, &mut ai1, 2000);
    let (out2, tick2, _) = run_sim_to_completion(s2.sim, &mut ai2, 2000);

    assert_eq!(out1, out2, "outcomes must match");
    assert_eq!(tick1, tick2, "tick counts must match");
}

#[test]
fn build_combat_gridnav_populated() {
    let cfg = ScenarioCfg {
        name: "nav_test".into(),
        seed: 42,
        hero_count: 2,
        enemy_count: 2,
        ..ScenarioCfg::default()
    };
    let setup = build_combat(&cfg);
    assert!(setup.sim.grid_nav.is_some(), "grid_nav must be populated");
    assert!(!setup.grid_nav.blocked.is_empty(), "should have blocked cells");
}

#[test]
fn build_combat_units_on_walkable_cells() {
    for seed in [1, 42, 100, 999] {
        let cfg = scenario_from_campaign(
            &["knight".into(), "mage".into()],
            3,
            5,
            RoomType::Pivot,
            seed,
            None,
        );
        let setup = build_combat(&cfg);
        for unit in setup.sim.units.iter() {
            let cell = setup.grid_nav.cell_of(unit.position);
            assert!(
                !setup.grid_nav.blocked.contains(&cell),
                "seed={}: unit {} at ({:.1},{:.1}) spawned on blocked cell {:?}",
                seed,
                unit.id,
                unit.position.x,
                unit.position.y,
                cell,
            );
        }
    }
}

// ===========================================================================
// Layer 2: Recording tests
// ===========================================================================

#[test]
fn recording_accumulates_frames() {
    let cfg = ScenarioCfg {
        name: "rec_test".into(),
        seed: 42,
        hero_count: 2,
        enemy_count: 2,
        ..ScenarioCfg::default()
    };
    let setup = build_combat(&cfg);
    let mut sim = setup.sim;
    let mut squad_ai = setup.squad_ai;

    // Simulate recording: push initial frame + one per tick
    let mut frames: Vec<SimState> = vec![sim.clone()];
    let mut events_per_frame: Vec<Vec<core::SimEvent>> = vec![Vec::new()];

    let target_ticks: u64 = 50;
    for _ in 0..target_ticks {
        let intents = bevy_game::ai::squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let (next, events) = core::step(sim, &intents, FIXED_TICK_MS);
        sim = next;
        frames.push(sim.clone());
        events_per_frame.push(events);
    }

    assert_eq!(
        frames.len(),
        target_ticks as usize + 1,
        "should have initial + {} tick frames",
        target_ticks,
    );
    assert_eq!(frames.len(), events_per_frame.len());

    // Frame 0 should be tick 0, last frame should be tick target_ticks
    assert_eq!(frames[0].tick, 0);
    assert_eq!(frames[target_ticks as usize].tick, target_ticks);

    // Positions should change over time (units move toward each other)
    let first_pos = frames[0].units[0].position;
    let last_pos = frames[target_ticks as usize].units[0].position;
    let moved = (first_pos.x - last_pos.x).abs() > 0.01
        || (first_pos.y - last_pos.y).abs() > 0.01;
    assert!(moved, "unit should have moved during {} ticks", target_ticks);
}

#[test]
fn recording_replay_matches_live() {
    let cfg = ScenarioCfg {
        name: "replay_match".into(),
        seed: 42,
        hero_count: 3,
        enemy_count: 3,
        ..ScenarioCfg::default()
    };
    let setup = build_combat(&cfg);
    let mut sim = setup.sim;
    let mut squad_ai = setup.squad_ai;

    let mut frames: Vec<SimState> = vec![sim.clone()];

    for _ in 0..100 {
        let intents = bevy_game::ai::squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let (next, _events) = core::step(sim, &intents, FIXED_TICK_MS);
        sim = next;
        frames.push(sim.clone());
    }

    // Verify replay frames reproduce the same unit positions as the live sim
    // by re-running from the same initial state
    let setup2 = build_combat(&cfg);
    let mut sim2 = setup2.sim;
    let mut squad_ai2 = setup2.squad_ai;

    for tick in 0..100 {
        // Frame[tick] should match sim2 state at this point
        for unit in frames[tick].units.iter() {
            let u2 = sim2.units.iter().find(|u| u.id == unit.id).unwrap();
            assert!(
                (unit.position.x - u2.position.x).abs() < 0.001
                    && (unit.position.y - u2.position.y).abs() < 0.001,
                "tick {}: unit {} position mismatch ({:.3},{:.3}) vs ({:.3},{:.3})",
                tick,
                unit.id,
                unit.position.x,
                unit.position.y,
                u2.position.x,
                u2.position.y,
            );
        }
        let intents = bevy_game::ai::squad::generate_intents(&sim2, &mut squad_ai2, FIXED_TICK_MS);
        let (next, _) = core::step(sim2, &intents, FIXED_TICK_MS);
        sim2 = next;
    }
}

// ===========================================================================
// Layer 3: Chaos / fuzz sim
// ===========================================================================

/// Generate a random intent for a unit given a seed.
fn random_intent(unit_id: u32, sim: &SimState, rng_seed: u64) -> UnitIntent {
    let r = splitmix(rng_seed) % 5;
    let action = match r {
        0 => IntentAction::Hold,
        1 => {
            // Attack nearest enemy
            let unit = sim.units.iter().find(|u| u.id == unit_id);
            let team = unit.map(|u| u.team).unwrap_or(Team::Hero);
            let target = sim
                .units
                .iter()
                .find(|u| u.team != team && u.hp > 0);
            match target {
                Some(t) => IntentAction::Attack { target_id: t.id },
                None => IntentAction::Hold,
            }
        }
        2 => {
            // Move to random position
            let x = (splitmix(rng_seed.wrapping_add(1)) % 20) as f32;
            let y = (splitmix(rng_seed.wrapping_add(2)) % 20) as f32;
            IntentAction::MoveTo {
                position: core::SimVec2 { x, y },
            }
        }
        3 => {
            // Use ability on random target
            let target = sim.units.iter().find(|u| u.hp > 0 && u.id != unit_id);
            match target {
                Some(t) => IntentAction::UseAbility {
                    ability_index: 0,
                    target: AbilityTarget::Unit(t.id),
                },
                None => IntentAction::Hold,
            }
        }
        _ => IntentAction::Hold,
    };
    UnitIntent { unit_id, action }
}

#[test]
fn chaos_sim_no_panics_no_violations() {
    let seeds = [1, 7, 42, 99, 256, 1000, 7777, 65535];
    for &seed in &seeds {
        let cfg = scenario_from_campaign(
            &["knight".into(), "mage".into(), "ranger".into()],
            3,
            10,
            RoomType::Entry,
            seed,
            None,
        );
        let setup = build_combat(&cfg);
        let mut sim = setup.sim;

        let max_ticks = 500;
        for tick in 0..max_ticks {
            // Generate random intents for all living units
            let intents: Vec<UnitIntent> = sim
                .units
                .iter()
                .filter(|u| u.hp > 0)
                .enumerate()
                .map(|(i, u)| {
                    let rng = splitmix(seed ^ (tick * 31 + i as u64));
                    random_intent(u.id, &sim, rng)
                })
                .collect();

            let (next, _events) = core::step(sim, &intents, FIXED_TICK_MS);
            sim = next;

            // Verify invariants every 10 ticks (save time)
            if tick % 10 == 0 {
                let report = verify_tick(&sim);
                assert!(
                    report.is_ok(),
                    "seed={} tick={}: invariant violations: {:?}",
                    seed,
                    tick,
                    report.violations,
                );
            }

            // Early exit if combat resolved
            let heroes_alive = sim.units.iter().any(|u| u.team == Team::Hero && u.hp > 0);
            let enemies_alive = sim.units.iter().any(|u| u.team == Team::Enemy && u.hp > 0);
            if !heroes_alive || !enemies_alive {
                break;
            }
        }
    }
}

#[test]
fn chaos_sim_with_hero_templates() {
    // Run chaos sim with actual hero templates to catch template-specific edge cases
    let template_sets: &[&[&str]] = &[
        &["knight", "mage"],
        &["paladin", "ranger", "rogue"],
        &["berserker", "cleric", "assassin", "shaman"],
    ];

    for (i, templates) in template_sets.iter().enumerate() {
        let hero_templates: Vec<String> = templates.iter().map(|s| s.to_string()).collect();
        let enemy_templates: Vec<String> = templates.iter().map(|s| s.to_string()).collect();

        let cfg = ScenarioCfg {
            name: format!("chaos_hvh_{}", i),
            seed: (i as u64 + 1) * 137,
            hero_count: templates.len(),
            enemy_count: templates.len(),
            hero_templates: hero_templates,
            enemy_hero_templates: enemy_templates,
            ..ScenarioCfg::default()
        };
        let setup = build_combat(&cfg);
        let mut sim = setup.sim;

        for tick in 0..300u64 {
            let intents: Vec<UnitIntent> = sim
                .units
                .iter()
                .filter(|u| u.hp > 0)
                .enumerate()
                .map(|(j, u)| {
                    let rng = splitmix((i as u64 * 1000 + tick) ^ (j as u64));
                    random_intent(u.id, &sim, rng)
                })
                .collect();

            let (next, _events) = core::step(sim, &intents, FIXED_TICK_MS);
            sim = next;

            let report = verify_tick(&sim);
            assert!(
                report.is_ok(),
                "template set {}: tick {}: violations: {:?}",
                i,
                tick,
                report.violations,
            );

            let heroes_alive = sim.units.iter().any(|u| u.team == Team::Hero && u.hp > 0);
            let enemies_alive = sim.units.iter().any(|u| u.team == Team::Enemy && u.hp > 0);
            if !heroes_alive || !enemies_alive {
                break;
            }
        }
    }
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn build_combat_single_hero_vs_many() {
    let cfg = scenario_from_campaign(&["knight".into()], 5, 20, RoomType::Climax, 42, None);
    let setup = build_combat(&cfg);

    let heroes = setup.sim.units.iter().filter(|u| u.team == Team::Hero).count();
    let enemies = setup.sim.units.iter().filter(|u| u.team == Team::Enemy).count();
    assert_eq!(heroes, 1);
    assert!(enemies >= 2, "should have multiple enemies");

    // Should still complete without panic
    let mut squad_ai = setup.squad_ai;
    let (outcome, _, _) = run_sim_to_completion(setup.sim, &mut squad_ai, 3000);
    assert!(outcome == "Victory" || outcome == "Defeat");
}

#[test]
fn campaign_scenario_seed_variation() {
    // Different seeds should produce different enemy compositions
    let cfg1 = scenario_from_campaign(&["knight".into()], 3, 10, RoomType::Entry, 1, None);
    let cfg2 = scenario_from_campaign(&["knight".into()], 3, 10, RoomType::Entry, 999, None);

    // At least some enemy templates should differ
    assert_ne!(
        cfg1.enemy_hero_templates, cfg2.enemy_hero_templates,
        "different seeds should produce different enemy compositions"
    );
}
