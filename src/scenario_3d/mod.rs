mod setup;
mod systems;
mod types;
pub mod champion_models;

pub use setup::*;
pub use systems::*;
pub use types::*;

use crate::ai;

pub fn build_horde_3d_bundle(
    hero_favored: bool,
) -> (ai::tooling::CustomScenario, Vec<ai::core::SimState>) {
    let seed = if hero_favored { 202 } else { 101 };
    let ticks = 420;
    let (initial, script) = if hero_favored {
        ai::advanced::build_horde_chokepoint_hero_favored_script(
            seed,
            ticks,
            ai::core::FIXED_TICK_MS,
        )
    } else {
        ai::advanced::build_horde_chokepoint_script(seed, ticks, ai::core::FIXED_TICK_MS)
    };
    let mut scenario = horde_initial_to_custom_scenario(&initial, seed, ticks, hero_favored);
    scenario.obstacles = horde_chokepoint_obstacles();
    let (frames, _events) = build_frames_from_script(initial, &script, ai::core::FIXED_TICK_MS);
    (scenario, frames)
}

fn build_frames_from_script(
    mut state: ai::core::SimState,
    script: &[Vec<ai::core::UnitIntent>],
    dt_ms: u32,
) -> (Vec<ai::core::SimState>, Vec<Vec<ai::core::SimEvent>>) {
    let mut frames = Vec::with_capacity(script.len() + 1);
    let mut all_events = Vec::with_capacity(script.len() + 1);
    frames.push(state.clone());
    all_events.push(Vec::new());
    for intents in script {
        let (next, events) = ai::core::step(state, intents, dt_ms);
        state = next;
        frames.push(state.clone());
        all_events.push(events);
    }
    (frames, all_events)
}

fn horde_chokepoint_obstacles() -> Vec<ai::tooling::ScenarioObstacle> {
    vec![
        ai::tooling::ScenarioObstacle {
            min_x: -0.8,
            max_x: 0.8,
            min_y: -9.5,
            max_y: -1.4,
        },
        ai::tooling::ScenarioObstacle {
            min_x: -0.8,
            max_x: 0.8,
            min_y: 1.4,
            max_y: 9.5,
        },
    ]
}

fn horde_initial_to_custom_scenario(
    initial: &ai::core::SimState,
    seed: u64,
    ticks: u32,
    hero_favored: bool,
) -> ai::tooling::CustomScenario {
    let units = initial
        .units
        .iter()
        .map(|u| ai::tooling::ScenarioUnit {
            id: u.id,
            team: match u.team {
                ai::core::Team::Hero => "Hero".to_string(),
                ai::core::Team::Enemy => "Enemy".to_string(),
            },
            x: u.position.x,
            y: u.position.y,
            elevation: 0.0,
            hp: u.hp,
            max_hp: u.max_hp,
            move_speed: u.move_speed_per_sec,
            attack_damage: u.attack_damage,
            attack_range: u.attack_range,
            ability_damage: u.ability_damage,
            ability_range: u.ability_range,
            heal_amount: u.heal_amount,
            heal_range: u.heal_range,
        })
        .collect::<Vec<_>>();

    ai::tooling::CustomScenario {
        name: if hero_favored {
            "horde_chokepoint_hero_favored".to_string()
        } else {
            "horde_chokepoint".to_string()
        },
        seed,
        ticks,
        world_min_x: -20.0,
        world_max_x: 20.0,
        world_min_y: -10.0,
        world_max_y: 10.0,
        cell_size: 0.7,
        elevation_zones: Vec::new(),
        slope_zones: Vec::new(),
        obstacles: Vec::new(),
        units,
    }
}
