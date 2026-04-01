//! Headless simulation bridge for LFM agent integration.
//!
//! Communicates via NDJSON over stdin/stdout. Python stderr for logging.
//!
//! Protocol:
//! -> {"type":"init","scenario":"basic_4v4","seed":42,"ticks":320,"decision_interval":10}
//! <- {"type":"state","tick":10,...}
//! -> {"type":"decision","personality_updates":{...},"squad_overrides":{...}}
//! ...
//! <- {"type":"done","winner":"Hero","metrics":{...}}

mod types;
mod helpers;

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use game::ai::control::ControlAiState;
use game::ai::core::{IntentAction, SimEvent, Team, UnitIntent, FIXED_TICK_MS};
use game::ai::pathing::clamp_step_to_walkable;
use game::ai::personality::{
    apply_personality_overrides, PersonalityAiState, PersonalityProfile,
};
use game::ai::squad::{FormationMode, SquadBlackboard};
use game::scenario::{load_scenario_file, run_scenario_to_state_with_room};

use serde_json::Value;

use types::*;
use helpers::*;

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();

    // 1. Read init message
    let mut init_line = String::new();
    reader.read_line(&mut init_line).expect("failed to read init");

    let init_val: Value = serde_json::from_str(init_line.trim()).expect("invalid init JSON");
    let msg_type = init_val
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if msg_type != "init" {
        eprintln!("expected init message, got: {}", msg_type);
        std::process::exit(1);
    }

    let init: InitMessage = serde_json::from_value(init_val).expect("failed to parse init");
    eprintln!(
        "[sim_bridge] scenario={} ticks={} interval={}",
        init.scenario, init.ticks, init.decision_interval
    );

    // 2. Build initial state from scenario file
    let scenario_path = std::path::Path::new(&init.scenario);
    let scenario_file = load_scenario_file(scenario_path).expect("failed to load scenario");
    let mut cfg = scenario_file.scenario;
    if let Some(seed) = init.seed {
        cfg.seed = seed;
    }
    cfg.max_ticks = init.ticks as u64;

    let (mut state, _squad_state, grid_nav) = run_scenario_to_state_with_room(&cfg);
    let room_width = grid_nav.max_x - grid_nav.min_x;
    let room_depth = grid_nav.max_y - grid_nav.min_y;

    // 3. Set up AI systems
    let roles = assign_roles(&state);
    let mut personalities: HashMap<u32, PersonalityProfile> = state
        .units
        .iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| (u.id, PersonalityProfile::tactician()))
        .collect();
    // Enemy units get default personalities too
    for u in state.units.iter().filter(|u| u.team == Team::Enemy) {
        personalities.insert(u.id, PersonalityProfile::vanguard());
    }

    // Find game root by walking up from scenario path until we find assets/behaviors/
    let game_root = {
        let mut p = scenario_path.canonicalize().unwrap_or_else(|_| scenario_path.to_path_buf());
        loop {
            if p.join("assets").join("behaviors").is_dir() {
                break;
            }
            if !p.pop() {
                p = std::path::PathBuf::from(".");
                break;
            }
        }
        p
    };

    let mut control_ai = if init.goap {
        eprintln!("[sim_bridge] GOAP mode enabled");
        match build_goap_for_heroes(&state, &roles, &game_root, &cfg.hero_templates) {
            Some(goap) => ControlAiState::new_with_goap(&state, roles.clone(), goap),
            None => {
                eprintln!("[sim_bridge] No GOAP defs loaded, falling back to squad AI");
                ControlAiState::new(&state, roles.clone())
            }
        }
    } else {
        ControlAiState::new(&state, roles.clone())
    };
    let mut personality_ai = PersonalityAiState::new(&state, roles.clone(), personalities.clone());

    let mut all_events: Vec<SimEvent> = Vec::new();
    let mut interval_events: Vec<SimEvent> = Vec::new();

    // 4. Main simulation loop
    let total_ticks = init.ticks;
    let interval = init.decision_interval;
    let mut tick_count: u32 = 0;

    loop {
        if tick_count >= total_ticks {
            break;
        }

        // Check if battle is over
        let heroes_alive = state
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0)
            .count();
        let enemies_alive = state
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count();
        if heroes_alive == 0 || enemies_alive == 0 {
            break;
        }

        // Run `interval` ticks
        interval_events.clear();
        let ticks_to_run = interval.min(total_ticks - tick_count);
        for _ in 0..ticks_to_run {
            // Generate intents using control AI (includes squad + CC)
            let (base_intents, reservations) =
                control_ai.generate_intents_tick(&state, FIXED_TICK_MS);

            // Apply personality overlays
            let (final_intents, _modes) = apply_personality_overrides(
                &state,
                &mut personality_ai,
                &base_intents,
                &reservations,
                FIXED_TICK_MS,
            );

            // Clamp MoveTo intents to walkable cells (room obstacles / walls)
            let clamped_intents: Vec<UnitIntent> = final_intents
                .iter()
                .map(|intent| match intent.action {
                    IntentAction::MoveTo { position } => {
                        let unit_pos = state
                            .units
                            .iter()
                            .find(|u| u.id == intent.unit_id)
                            .map(|u| u.position)
                            .unwrap_or(position);
                        UnitIntent {
                            unit_id: intent.unit_id,
                            action: IntentAction::MoveTo {
                                position: clamp_step_to_walkable(
                                    &grid_nav, unit_pos, position,
                                ),
                            },
                        }
                    }
                    _ => *intent,
                })
                .collect();

            // Step simulation
            let (new_state, events) =
                game::ai::core::step(state, &clamped_intents, FIXED_TICK_MS);
            control_ai.update_from_events(new_state.tick, &events);

            interval_events.extend(events.iter().cloned());
            all_events.extend(events);
            state = new_state;
            tick_count += 1;

            // Check for battle end mid-interval
            let h = state
                .units
                .iter()
                .filter(|u| u.team == Team::Hero && u.hp > 0)
                .count();
            let e = state
                .units
                .iter()
                .filter(|u| u.team == Team::Enemy && u.hp > 0)
                .count();
            if h == 0 || e == 0 {
                break;
            }
        }

        // Check if battle ended
        let heroes_alive = state
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0)
            .count();
        let enemies_alive = state
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count();

        if heroes_alive == 0 || enemies_alive == 0 {
            // Send final done message
            let winner = if heroes_alive > 0 {
                "Hero"
            } else if enemies_alive > 0 {
                "Enemy"
            } else {
                "Draw"
            };
            let done = DoneMessage {
                r#type: "done".into(),
                winner: winner.into(),
                tick: state.tick,
                hero_hp_total: state
                    .units
                    .iter()
                    .filter(|u| u.team == Team::Hero)
                    .map(|u| u.hp.max(0))
                    .sum(),
                enemy_hp_total: state
                    .units
                    .iter()
                    .filter(|u| u.team == Team::Enemy)
                    .map(|u| u.hp.max(0))
                    .sum(),
                hero_alive: heroes_alive,
                enemy_alive: enemies_alive,
            };
            let json = serde_json::to_string(&done).unwrap();
            writeln!(writer, "{}", json).unwrap();
            writer.flush().unwrap();
            break;
        }

        // 5. Serialize state and send
        let mut msg =
            condense_state(&state, &roles, &personalities, &interval_events, room_width, room_depth);

        // Add squad blackboard info
        for team in [Team::Hero, Team::Enemy] {
            let team_name = team_str(team).to_string();
            // Read the blackboard from the control AI's squad state
            msg.squads.insert(
                team_name,
                CondensedSquad {
                    focus_target: None,
                    mode: "Hold".into(),
                },
            );
        }

        let json = serde_json::to_string(&msg).unwrap();
        writeln!(writer, "{}", json).unwrap();
        writer.flush().unwrap();

        // 6. Read decision
        let mut decision_line = String::new();
        match reader.read_line(&mut decision_line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("[sim_bridge] read error: {}", e);
                break;
            }
        }

        let decision_val: Value = match serde_json::from_str(decision_line.trim()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[sim_bridge] invalid decision JSON: {}", e);
                continue;
            }
        };

        let dtype = decision_val
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if dtype != "decision" {
            eprintln!("[sim_bridge] expected decision, got: {}", dtype);
            continue;
        }

        let decision: DecisionMessage = match serde_json::from_value(decision_val) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[sim_bridge] failed to parse decision: {}", e);
                continue;
            }
        };

        // 7. Apply personality updates
        for (unit_id_str, profile) in &decision.personality_updates {
            if let Ok(unit_id) = unit_id_str.parse::<u32>() {
                personality_ai.set_personality(unit_id, *profile);
                personalities.insert(unit_id, *profile);
            }
        }

        // 8. Apply squad overrides
        for (team_str_key, override_val) in &decision.squad_overrides {
            let team = match team_str_key.to_lowercase().as_str() {
                "hero" => Team::Hero,
                "enemy" => Team::Enemy,
                _ => continue,
            };
            let board = SquadBlackboard {
                focus_target: override_val.focus_target,
                mode: override_val
                    .mode
                    .as_deref()
                    .map(parse_formation)
                    .unwrap_or(FormationMode::Hold),
            };
            control_ai.squad_ai_mut().set_blackboard(team, board);
        }
    }

    // If we exited the loop without sending done (timeout)
    let heroes_alive = state
        .units
        .iter()
        .filter(|u| u.team == Team::Hero && u.hp > 0)
        .count();
    let enemies_alive = state
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0)
        .count();

    if tick_count >= total_ticks || (heroes_alive > 0 && enemies_alive > 0) {
        let winner = if heroes_alive > enemies_alive {
            "Hero"
        } else if enemies_alive > heroes_alive {
            "Enemy"
        } else {
            "Draw"
        };
        let done = DoneMessage {
            r#type: "done".into(),
            winner: winner.into(),
            tick: state.tick,
            hero_hp_total: state
                .units
                .iter()
                .filter(|u| u.team == Team::Hero)
                .map(|u| u.hp.max(0))
                .sum(),
            enemy_hp_total: state
                .units
                .iter()
                .filter(|u| u.team == Team::Enemy)
                .map(|u| u.hp.max(0))
                .sum(),
            hero_alive: heroes_alive,
            enemy_alive: enemies_alive,
        };
        let json = serde_json::to_string(&done).unwrap();
        let _ = writeln!(writer, "{}", json);
        let _ = writer.flush();
    }
}
