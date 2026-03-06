use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io;

use crate::ai::core::{run_replay, sim_vec2, step, SimState, Team, UnitIntent, UnitState};
use crate::ai::pathing::GridNav;
use crate::ai::personality::{
    default_personalities, generate_scripted_intents, sample_phase5_party_state,
};

use super::events::{build_event_rows, build_frame_rows, obstacle_rows_from_nav_cells};
use super::types::{CustomScenario, ScenarioObstacle, ScenarioUnit};
use super::viz_template::build_visualization_html;

fn parse_team(label: &str) -> Team {
    if label.eq_ignore_ascii_case("hero") {
        Team::Hero
    } else {
        Team::Enemy
    }
}

fn custom_scenario_to_state(s: &CustomScenario) -> SimState {
    let mut units = s
        .units
        .iter()
        .map(|u| UnitState {
            id: u.id,
            team: parse_team(&u.team),
            hp: u.hp,
            max_hp: u.max_hp,
            position: sim_vec2(u.x, u.y),
            move_speed_per_sec: u.move_speed,
            attack_damage: u.attack_damage,
            attack_range: u.attack_range,
            attack_cooldown_ms: 700,
            attack_cast_time_ms: 250,
            cooldown_remaining_ms: 0,
            ability_damage: u.ability_damage,
            ability_range: u.ability_range,
            ability_cooldown_ms: 2_800,
            ability_cast_time_ms: 420,
            ability_cooldown_remaining_ms: 0,
            heal_amount: u.heal_amount,
            heal_range: u.heal_range,
            heal_cooldown_ms: 2_100,
            heal_cast_time_ms: 380,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: Vec::new(),
            passives: Vec::new(),
            status_effects: Vec::new(),
            shield_hp: 0,
            resistance_tags: HashMap::new(),
            state_history: VecDeque::new(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        })
        .collect::<Vec<_>>();
    units.sort_by_key(|u| u.id);
    SimState {
        tick: 0,
        rng_state: s.seed,
        units,
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

fn build_custom_nav(s: &CustomScenario) -> GridNav {
    let mut nav = GridNav::new(
        s.world_min_x,
        s.world_max_x,
        s.world_min_y,
        s.world_max_y,
        s.cell_size.max(0.2),
    );
    for o in &s.obstacles {
        nav.add_block_rect(o.min_x, o.max_x, o.min_y, o.max_y);
    }
    for zone in &s.elevation_zones {
        nav.set_elevation_rect(
            zone.min_x,
            zone.max_x,
            zone.min_y,
            zone.max_y,
            zone.elevation,
        );
    }
    for zone in &s.slope_zones {
        nav.set_slope_cost_rect(
            zone.min_x,
            zone.max_x,
            zone.min_y,
            zone.max_y,
            zone.slope_cost_multiplier,
        );
    }
    nav
}

fn custom_scenario_intents(state: &SimState, nav: &GridNav, dt_ms: u32) -> Vec<UnitIntent> {
    crate::ai::advanced::build_environment_reactive_intents(state, nav, dt_ms)
}

fn build_custom_scenario_script(
    s: &CustomScenario,
    dt_ms: u32,
) -> (SimState, Vec<Vec<UnitIntent>>) {
    let nav = build_custom_nav(s);
    let initial = custom_scenario_to_state(s);
    let mut state = initial.clone();
    let mut script = Vec::with_capacity(s.ticks as usize);
    for _ in 0..s.ticks {
        let intents = custom_scenario_intents(&state, &nav, dt_ms);
        script.push(intents.clone());
        let (next, _) = step(state, &intents, dt_ms);
        state = next;
    }
    (initial, script)
}

pub(crate) fn build_phase5_event_visualization_html(seed: u64, ticks: u32) -> String {
    let run =
        crate::ai::personality::run_phase5_sample(seed, ticks, crate::ai::core::FIXED_TICK_MS);
    let replay = run.replay;
    let event_rows = build_event_rows(&replay);

    let initial = sample_phase5_party_state(seed);
    let roles = crate::ai::roles::default_roles();
    let personalities = default_personalities();
    let (script, _mode_history) = generate_scripted_intents(
        &initial,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
        roles,
        personalities,
    );
    let frame_rows = build_frame_rows(&initial, &script, crate::ai::core::FIXED_TICK_MS);

    build_visualization_html(
        "AI Event Visualization",
        "Phase 5 personality timeline",
        &replay,
        &event_rows,
        &frame_rows,
        "",
        seed,
        ticks,
    )
}

pub fn export_phase5_event_visualization(path: &str, seed: u64, ticks: u32) -> io::Result<()> {
    let html = build_phase5_event_visualization_html(seed, ticks);
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}

pub fn export_horde_chokepoint_visualization(path: &str, seed: u64, ticks: u32) -> io::Result<()> {
    let (initial, script) = crate::ai::advanced::build_horde_chokepoint_script(
        seed,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let replay = run_replay(
        initial.clone(),
        &script,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let frame_rows = build_frame_rows(&initial, &script, crate::ai::core::FIXED_TICK_MS);
    let event_rows = build_event_rows(&replay);
    let nav = crate::ai::advanced::horde_chokepoint_nav();
    let obstacle_rows = obstacle_rows_from_nav_cells(&nav);

    let html = build_visualization_html(
        "Pathing Horde Visualization",
        "Chokepoint wall/gate with A* waypointing",
        &replay,
        &event_rows,
        &frame_rows,
        &obstacle_rows,
        seed,
        ticks,
    );
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}

pub fn export_horde_chokepoint_hero_favored_visualization(
    path: &str,
    seed: u64,
    ticks: u32,
) -> io::Result<()> {
    let (initial, script) = crate::ai::advanced::build_horde_chokepoint_hero_favored_script(
        seed,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let replay = run_replay(
        initial.clone(),
        &script,
        ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let frame_rows = build_frame_rows(&initial, &script, crate::ai::core::FIXED_TICK_MS);
    let event_rows = build_event_rows(&replay);
    let nav = crate::ai::advanced::horde_chokepoint_nav();
    let obstacle_rows = obstacle_rows_from_nav_cells(&nav);

    let html = build_visualization_html(
        "Pathing Horde Visualization (Hero Favored)",
        "Chokepoint wall/gate with hero-favored roster and pressure tactics",
        &replay,
        &event_rows,
        &frame_rows,
        &obstacle_rows,
        seed,
        ticks,
    );
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}

pub fn write_custom_scenario_template(path: &str) -> io::Result<()> {
    let scenario = CustomScenario {
        name: "chokepoint_demo".to_string(),
        seed: 777,
        ticks: 360,
        world_min_x: -20.0,
        world_max_x: 20.0,
        world_min_y: -10.0,
        world_max_y: 10.0,
        cell_size: 0.7,
        elevation_zones: vec![],
        slope_zones: vec![],
        obstacles: vec![ScenarioObstacle {
            min_x: -0.8,
            max_x: 0.8,
            min_y: -9.0,
            max_y: 9.0,
        }],
        units: vec![
            ScenarioUnit {
                id: 1,
                team: "Hero".to_string(),
                x: -14.0,
                y: -0.6,
                elevation: 0.0,
                hp: 165,
                max_hp: 165,
                move_speed: 4.1,
                attack_damage: 16,
                attack_range: 1.4,
                ability_damage: 28,
                ability_range: 2.0,
                heal_amount: 0,
                heal_range: 0.0,
            },
            ScenarioUnit {
                id: 2,
                team: "Hero".to_string(),
                x: -15.2,
                y: 1.0,
                elevation: 0.0,
                hp: 96,
                max_hp: 96,
                move_speed: 4.3,
                attack_damage: 9,
                attack_range: 1.3,
                ability_damage: 0,
                ability_range: 0.0,
                heal_amount: 28,
                heal_range: 2.8,
            },
            ScenarioUnit {
                id: 10,
                team: "Enemy".to_string(),
                x: 12.0,
                y: -1.8,
                elevation: 0.0,
                hp: 82,
                max_hp: 82,
                move_speed: 4.5,
                attack_damage: 12,
                attack_range: 1.2,
                ability_damage: 16,
                ability_range: 1.9,
                heal_amount: 0,
                heal_range: 0.0,
            },
            ScenarioUnit {
                id: 11,
                team: "Enemy".to_string(),
                x: 12.8,
                y: 0.0,
                elevation: 0.0,
                hp: 82,
                max_hp: 82,
                move_speed: 4.5,
                attack_damage: 12,
                attack_range: 1.2,
                ability_damage: 16,
                ability_range: 1.9,
                heal_amount: 0,
                heal_range: 0.0,
            },
            ScenarioUnit {
                id: 12,
                team: "Enemy".to_string(),
                x: 12.0,
                y: 1.8,
                elevation: 0.0,
                hp: 82,
                max_hp: 82,
                move_speed: 4.5,
                attack_damage: 12,
                attack_range: 1.2,
                ability_damage: 16,
                ability_range: 1.9,
                heal_amount: 0,
                heal_range: 0.0,
            },
        ],
    };
    let body = serde_json::to_string_pretty(&scenario)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, body)?;
    Ok(())
}

pub fn export_custom_scenario_visualization(
    scenario_path: &str,
    output_path: &str,
) -> io::Result<()> {
    let text = fs::read_to_string(scenario_path)?;
    let scenario: CustomScenario = serde_json::from_str(&text)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let (initial, script) = build_custom_scenario_script(&scenario, crate::ai::core::FIXED_TICK_MS);
    let replay = run_replay(
        initial.clone(),
        &script,
        scenario.ticks,
        crate::ai::core::FIXED_TICK_MS,
    );
    let frame_rows = build_frame_rows(&initial, &script, crate::ai::core::FIXED_TICK_MS);
    let event_rows = build_event_rows(&replay);
    let nav = build_custom_nav(&scenario);
    let obstacle_rows = obstacle_rows_from_nav_cells(&nav);
    let html = build_visualization_html(
        "Custom Scenario Visualization",
        &format!("{} | {}", scenario.name, scenario_path),
        &replay,
        &event_rows,
        &frame_rows,
        &obstacle_rows,
        scenario.seed,
        scenario.ticks,
    );
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, html)?;
    Ok(())
}

pub fn build_custom_scenario_state_frames(s: &CustomScenario, dt_ms: u32) -> Vec<SimState> {
    let (initial, script) = build_custom_scenario_script(s, dt_ms);
    let mut frames = Vec::with_capacity(s.ticks as usize + 1);
    let mut state = initial;
    frames.push(state.clone());
    for intents in script.iter().take(s.ticks as usize) {
        let (next, _) = step(state, intents, dt_ms);
        state = next;
        frames.push(state.clone());
    }
    frames
}

pub fn export_visualization_index(path: &str, links: &[(String, String)]) -> io::Result<()> {
    let mut items = String::new();
    for (label, href) in links {
        items.push_str(&format!(
            "<li><a href=\"{}\">{}</a></li>",
            href.replace('"', ""),
            label
        ));
    }
    let html = format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>AI Visualization Index</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 20px; background:#0f1420; color:#e6ebff; }}
h2 {{ margin:0 0 8px 0; }}
ul {{ line-height: 1.9; }}
a {{ color:#8bc7ff; }}
code {{ background:#1a2436; padding:2px 5px; border-radius:4px; }}
</style></head>
<body>
<h2>AI Visualization Index</h2>
<p>Use <code>--phase6-viz</code>, <code>--pathing-viz</code>, or <code>--scenario-viz &lt;json&gt;</code> to generate pages.</p>
<ul>{}</ul>
</body></html>"#,
        items
    );
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)?;
    Ok(())
}
