use std::io::{BufWriter, Write, BufRead, BufReader};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cli::{RoomgenCommand, RoomgenSubcommand, RoomgenExportArgs, RoomgenRenderArgs, RoomgenSimulateArgs, RoomgenFloorplanArgs};

pub(crate) fn run_roomgen_cmd(cmd: RoomgenCommand) -> ExitCode {
    match cmd.sub {
        RoomgenSubcommand::Export(args) => run_export(args),
        RoomgenSubcommand::Render(args) => run_render(args),
        RoomgenSubcommand::Simulate(args) => run_simulate(args),
        RoomgenSubcommand::Floorplan(args) => run_floorplan(args),
    }
}

// ---------------------------------------------------------------------------
// Export: batch-generate rooms as JSONL
// ---------------------------------------------------------------------------

fn run_export(args: RoomgenExportArgs) -> ExitCode {
    use bevy_game::game_core::RoomType;
    use bevy_game::mission::room_gen::generate_room_varied;

    let room_types = [
        RoomType::Entry,
        RoomType::Pressure,
        RoomType::Pivot,
        RoomType::Setpiece,
        RoomType::Recovery,
        RoomType::Climax,
    ];

    let threads = if args.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        args.threads
    };

    // Build work items: (seed, room_type)
    let mut work: Vec<(u64, RoomType)> = Vec::new();
    for &rt in &room_types {
        for i in 0..args.count_per_type {
            let seed = args.seed.wrapping_add(i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            work.push((seed, rt));
        }
    }

    let total = work.len();
    eprintln!(
        "Generating {} rooms ({} per type × {} types) using {} threads...",
        total,
        args.count_per_type,
        room_types.len(),
        threads,
    );

    // Parallel generation
    let chunk_size = (work.len() + threads - 1) / threads;
    let chunks: Vec<_> = work.chunks(chunk_size).map(|c| c.to_vec()).collect();
    let varied = args.varied;

    let results: Vec<Vec<String>> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                s.spawn(move || {
                    let mut lines = Vec::with_capacity(chunk.len());
                    for (seed, rt) in chunk {
                        let layout = if varied {
                            generate_room_varied(seed, rt, None)
                        } else {
                            bevy_game::mission::room_gen::generate_room(seed, rt)
                        };
                        lines.push(layout_to_json(&layout, seed));
                    }
                    lines
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Write output
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(&args.output).unwrap();
    let mut writer = BufWriter::new(file);
    let mut count = 0usize;
    for batch in &results {
        for line in batch {
            writeln!(writer, "{}", line).unwrap();
            count += 1;
        }
    }
    writer.flush().unwrap();

    eprintln!("Wrote {} rooms to {}", count, args.output.display());
    ExitCode::SUCCESS
}

fn layout_to_json(layout: &bevy_game::mission::room_gen::RoomLayout, seed: u64) -> String {
    let grid = layout.to_grid();
    let metrics = layout.compute_metrics();

    let rt_str = match layout.room_type {
        bevy_game::game_core::RoomType::Entry => "Entry",
        bevy_game::game_core::RoomType::Pressure => "Pressure",
        bevy_game::game_core::RoomType::Pivot => "Pivot",
        bevy_game::game_core::RoomType::Setpiece => "Setpiece",
        bevy_game::game_core::RoomType::Recovery => "Recovery",
        bevy_game::game_core::RoomType::Climax => "Climax",
        bevy_game::game_core::RoomType::Open => "Open",
    };

    // Build 2D arrays as JSON strings manually for efficiency
    let width = grid.width;
    let depth = grid.depth;

    let mut obs_rows = String::from("[");
    let mut height_rows = String::from("[");
    let mut elev_rows = String::from("[");

    for r in 0..depth {
        if r > 0 {
            obs_rows.push(',');
            height_rows.push(',');
            elev_rows.push(',');
        }
        obs_rows.push('[');
        height_rows.push('[');
        elev_rows.push('[');
        for c in 0..width {
            if c > 0 {
                obs_rows.push(',');
                height_rows.push(',');
                elev_rows.push(',');
            }
            let idx = r * width + c;
            obs_rows.push_str(&grid.obstacle_type[idx].to_string());
            // Round floats to 2 decimal places
            height_rows.push_str(&format!("{:.2}", grid.height[idx]));
            elev_rows.push_str(&format!("{:.2}", grid.elevation[idx]));
        }
        obs_rows.push(']');
        height_rows.push(']');
        elev_rows.push(']');
    }
    obs_rows.push(']');
    height_rows.push(']');
    elev_rows.push(']');

    // Spawn positions
    let player_spawn: Vec<String> = layout
        .player_spawn
        .positions
        .iter()
        .map(|p| format!("[{:.1},{:.1}]", p.x, p.y))
        .collect();
    let enemy_spawn: Vec<String> = layout
        .enemy_spawn
        .positions
        .iter()
        .map(|p| format!("[{:.1},{:.1}]", p.x, p.y))
        .collect();

    format!(
        r#"{{"seed":{},"room_type":"{}","width":{},"depth":{},"grid":{{"channels":["obstacle_type","height","elevation"],"obstacle_type":{},"height":{},"elevation":{}}},"metrics":{{"blocked_pct":{:.4},"chokepoint_count":{},"cover_density":{:.4},"elevation_zones":{},"flanking_routes":{},"spawn_quality_diff":{:.4},"mean_wall_proximity":{:.4},"aspect_ratio":{:.4}}},"player_spawn":[{}],"enemy_spawn":[{}]}}"#,
        seed,
        rt_str,
        width,
        depth,
        obs_rows,
        height_rows,
        elev_rows,
        metrics.blocked_pct,
        metrics.chokepoint_count,
        metrics.cover_density,
        metrics.elevation_zones,
        metrics.flanking_routes,
        metrics.spawn_quality_diff,
        metrics.mean_wall_proximity,
        metrics.aspect_ratio,
        player_spawn.join(","),
        enemy_spawn.join(","),
    )
}

// ---------------------------------------------------------------------------
// Render: top-down PNG images
// ---------------------------------------------------------------------------

fn run_render(args: RoomgenRenderArgs) -> ExitCode {
    use bevy_game::mission::room_gen::*;

    let input = match std::fs::File::open(&args.input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open {}: {}", args.input.display(), e);
            return ExitCode::FAILURE;
        }
    };

    std::fs::create_dir_all(&args.output_dir).ok();

    let reader = BufReader::new(input);
    let ppc = args.pixels_per_cell;
    let mut count = 0usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }

        // Parse just the fields we need
        let parsed: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Skipping invalid JSON line: {}", e);
                continue;
            }
        };

        let seed = parsed["seed"].as_u64().unwrap_or(0);
        let room_type = parsed["room_type"].as_str().unwrap_or("Unknown");
        let width = parsed["width"].as_u64().unwrap_or(0) as usize;
        let depth = parsed["depth"].as_u64().unwrap_or(0) as usize;

        if width == 0 || depth == 0 {
            continue;
        }

        let obs_grid = &parsed["grid"]["obstacle_type"];
        let elev_grid = &parsed["grid"]["elevation"];

        let img_w = (width as u32) * ppc;
        let img_h = (depth as u32) * ppc;
        let mut img_buf = vec![0u8; (img_w * img_h * 3) as usize];

        // Parse player/enemy spawn cells
        let player_cells: std::collections::HashSet<(usize, usize)> = parsed["player_spawn"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| {
                        let x = p[0].as_f64()? as usize;
                        let y = p[1].as_f64()? as usize;
                        Some((x, y))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let enemy_cells: std::collections::HashSet<(usize, usize)> = parsed["enemy_spawn"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| {
                        let x = p[0].as_f64()? as usize;
                        let y = p[1].as_f64()? as usize;
                        Some((x, y))
                    })
                    .collect()
            })
            .unwrap_or_default();

        for r in 0..depth {
            for c in 0..width {
                let obs_type = obs_grid[r][c].as_u64().unwrap_or(0) as u8;
                let elevation = elev_grid[r][c].as_f64().unwrap_or(0.0) as f32;

                let (cr, cg, cb) = if obs_type == OBS_WALL {
                    // Perimeter wall = black
                    (0u8, 0u8, 0u8)
                } else if obs_type != OBS_FLOOR && obs_type != OBS_RAMP {
                    // Interior obstacles: darker = taller
                    let base = 0x60u8;
                    let darkness = ((obs_type as f32 * 4.0).min(32.0)) as u8;
                    (base - darkness.min(base), base - darkness.min(base), base - darkness.min(base))
                } else if player_cells.contains(&(c, r)) {
                    // Player spawn = green
                    (0x22, 0xAA, 0x22)
                } else if enemy_cells.contains(&(c, r)) {
                    // Enemy spawn = red
                    (0xCC, 0x22, 0x22)
                } else if elevation > 0.1 {
                    // Elevated walkable = blue tint, brighter = higher
                    let brightness = (elevation / 1.5).clamp(0.0, 1.0);
                    let r = (0x66 as f32 + brightness * 0x33 as f32) as u8;
                    let g = (0x88 as f32 + brightness * 0x33 as f32) as u8;
                    let b = (0xCC as f32 + brightness * 0x33 as f32) as u8;
                    (r, g, b)
                } else {
                    // Floor = white
                    (0xFF, 0xFF, 0xFF)
                };

                // Fill ppc×ppc pixel block
                for py in 0..ppc {
                    for px in 0..ppc {
                        let img_x = c as u32 * ppc + px;
                        let img_y = r as u32 * ppc + py;
                        let pixel_idx = ((img_y * img_w + img_x) * 3) as usize;
                        img_buf[pixel_idx] = cr;
                        img_buf[pixel_idx + 1] = cg;
                        img_buf[pixel_idx + 2] = cb;
                    }
                }
            }
        }

        let filename = format!("{}_{}.png", room_type, seed);
        let path = args.output_dir.join(&filename);

        let img = image::RgbImage::from_raw(img_w, img_h, img_buf).unwrap();
        img.save(&path).unwrap();
        count += 1;

        if count % 1000 == 0 {
            eprintln!("Rendered {} images...", count);
        }
    }

    eprintln!("Rendered {} images to {}", count, args.output_dir.display());
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Floorplan: generate and render mission floorplans
// ---------------------------------------------------------------------------

fn run_floorplan(args: RoomgenFloorplanArgs) -> ExitCode {
    use bevy_game::mission::room_gen::floorplan::{generate_floorplan, FloorplanConfig, RoomRole};
    use bevy_game::mission::room_gen::*;

    std::fs::create_dir_all(&args.output).ok();

    let config = FloorplanConfig {
        room_count: args.rooms,
        grid_width: args.width,
        grid_height: args.height,
        ..FloorplanConfig::default()
    };

    let ppc = args.pixels_per_cell;

    for i in 0..args.count {
        let seed = args.seed.wrapping_add(i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let fp = generate_floorplan(seed, &config);

        let grid = fp.to_grid();
        let w = grid.width;
        let h = grid.depth;

        // Collect spawn cells for coloring
        let mut player_cells = std::collections::HashSet::new();
        let mut enemy_cells = std::collections::HashSet::new();
        for room in &fp.rooms {
            for p in &room.player_spawns {
                player_cells.insert((p.x as usize, p.y as usize));
            }
            for p in &room.enemy_spawns {
                enemy_cells.insert((p.x as usize, p.y as usize));
            }
        }

        let img_w = w as u32 * ppc;
        let img_h = h as u32 * ppc;
        let mut img_buf = vec![0u8; (img_w * img_h * 3) as usize];

        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                let obs_type = grid.obstacle_type[idx];
                let elevation = grid.elevation[idx];

                let (cr, cg, cb) = if player_cells.contains(&(c, r)) {
                    (0x22u8, 0xAAu8, 0x22u8)
                } else if enemy_cells.contains(&(c, r)) {
                    (0xCCu8, 0x22u8, 0x22u8)
                } else if obs_type == OBS_WALL {
                    (0x1Au8, 0x1Au8, 0x1Au8)
                } else if obs_type == OBS_FLOOR {
                    if elevation > 0.1 {
                        let b = (0x99 as f32 + (elevation / 1.5) * 0x44 as f32).min(255.0) as u8;
                        (0x66, 0x88, b)
                    } else {
                        (0xE8u8, 0xE0u8, 0xD0u8) // warm off-white floor
                    }
                } else {
                    // Interior obstacle — shade by type
                    let shade = (0x70u8).saturating_sub((obs_type as u8).saturating_mul(6));
                    (shade, shade, shade)
                };

                for py in 0..ppc {
                    for px in 0..ppc {
                        let ix = c as u32 * ppc + px;
                        let iy = r as u32 * ppc + py;
                        let pi = ((iy * img_w + ix) * 3) as usize;
                        img_buf[pi] = cr;
                        img_buf[pi + 1] = cg;
                        img_buf[pi + 2] = cb;
                    }
                }
            }
        }

        let filename = format!("floorplan_{}.png", seed);
        let path = args.output.join(&filename);
        let img = image::RgbImage::from_raw(img_w, img_h, img_buf).unwrap();
        img.save(&path).unwrap();

        eprintln!(
            "[{}/{}] {} — {}×{}, {} rooms, {} corridors",
            i + 1, args.count, filename, w, h,
            fp.rooms.len(), fp.corridors.len(),
        );
    }

    eprintln!("Generated {} floorplans in {}", args.count, args.output.display());
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Simulate: run HvH combat on generated rooms
// ---------------------------------------------------------------------------

fn run_simulate(args: RoomgenSimulateArgs) -> ExitCode {
    use bevy_game::ai::core::{step, SimVec2, Team, FIXED_TICK_MS};
    use bevy_game::ai::effects::HeroToml;
    use bevy_game::mission::hero_templates::parse_hero_toml_with_dsl;
    use bevy_game::mission::room_gen::NavGrid;
    use bevy_game::mission::sim_bridge::build_sim_with_hero_templates;
    use bevy_game::scenario::build_unified_ai;

    // 1. Discover hero templates from directory
    let hero_tomls = load_hero_dir(&args.heroes);
    if hero_tomls.is_empty() {
        eprintln!("No hero templates found in {}", args.heroes.display());
        return ExitCode::FAILURE;
    }
    let hero_names: Vec<String> = hero_tomls.iter().map(|(n, _)| n.clone()).collect();
    eprintln!("Loaded {} hero templates from {}", hero_names.len(), args.heroes.display());

    // 2. Load rooms from JSONL (just seed, width, depth, room_type, spawn positions)
    let rooms = load_room_records(&args.rooms);
    if rooms.is_empty() {
        eprintln!("No valid rooms found in {}", args.rooms.display());
        return ExitCode::FAILURE;
    }
    eprintln!("Loaded {} rooms from {}", rooms.len(), args.rooms.display());

    let threads = if args.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    } else {
        args.threads
    };

    // 3. Build work items: (room_index, hero_team_indices, enemy_team_indices)
    let team_size = 4usize;
    let n_heroes = hero_tomls.len();
    let mut work: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();
    let mut lcg = LcgSimple::new(42);

    for _ in 0..args.max_matches {
        let room_idx = lcg.next() % rooms.len();
        let heroes: Vec<usize> = (0..team_size).map(|_| lcg.next() % n_heroes).collect();
        let enemies: Vec<usize> = (0..team_size).map(|_| lcg.next() % n_heroes).collect();
        work.push((room_idx, heroes, enemies));
    }

    let total = work.len();
    eprintln!("Running {} matches ({} rooms × random 4v4 comps) on {} threads...", total, rooms.len(), threads);

    // 4. Run in parallel
    let progress = AtomicUsize::new(0);
    let chunk_size = (work.len() + threads - 1) / threads;
    let work_chunks: Vec<_> = work.chunks(chunk_size).map(|c| c.to_vec()).collect();
    let rooms_ref = &rooms;
    let hero_tomls_ref = &hero_tomls;
    let hero_names_ref = &hero_names;
    let progress_ref = &progress;

    let results: Vec<Vec<String>> = std::thread::scope(|s| {
        let handles: Vec<_> = work_chunks
            .into_iter()
            .map(|chunk| {
                s.spawn(move || {
                    let mut lines = Vec::with_capacity(chunk.len());
                    for (room_idx, hero_idxs, enemy_idxs) in &chunk {
                        let room = &rooms_ref[*room_idx];

                        let h_tomls: Vec<HeroToml> = hero_idxs.iter()
                            .map(|&i| hero_tomls_ref[i].1.clone())
                            .collect();
                        let e_tomls: Vec<HeroToml> = enemy_idxs.iter()
                            .map(|&i| hero_tomls_ref[i].1.clone())
                            .collect();
                        let h_names: Vec<&str> = hero_idxs.iter()
                            .map(|&i| hero_names_ref[i].as_str())
                            .collect();
                        let e_names: Vec<&str> = enemy_idxs.iter()
                            .map(|&i| hero_names_ref[i].as_str())
                            .collect();

                        let result = run_single_match(
                            room, &h_tomls, &e_tomls, &h_names, &e_names, room.seed,
                        );
                        lines.push(result);

                        let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 200 == 0 {
                            eprintln!("  {}/{} matches...", done, total);
                        }
                    }
                    lines
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // 5. Write results + summary
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(&args.output).unwrap();
    let mut writer = BufWriter::new(file);

    let mut victories = 0usize;
    let mut defeats = 0usize;
    let mut timeouts = 0usize;
    let mut total_ticks = 0u64;
    let mut count = 0usize;

    for batch in &results {
        for line in batch {
            writeln!(writer, "{}", line).unwrap();
            count += 1;
            // Parse outcome for summary
            if line.contains("\"outcome\":\"Victory\"") {
                victories += 1;
            } else if line.contains("\"outcome\":\"Defeat\"") {
                defeats += 1;
            } else {
                timeouts += 1;
            }
            // Extract tick count
            if let Some(pos) = line.find("\"ticks\":") {
                let rest = &line[pos + 8..];
                if let Some(end) = rest.find(|c: char| !c.is_ascii_digit()) {
                    if let Ok(t) = rest[..end].parse::<u64>() {
                        total_ticks += t;
                    }
                }
            }
        }
    }
    writer.flush().unwrap();

    let avg_ticks = if count > 0 { total_ticks / count as u64 } else { 0 };
    eprintln!("\n=== Results ({} matches) ===", count);
    eprintln!("  Hero wins:  {:5} ({:.1}%)", victories, victories as f64 / count as f64 * 100.0);
    eprintln!("  Hero losses:{:5} ({:.1}%)", defeats, defeats as f64 / count as f64 * 100.0);
    eprintln!("  Timeouts:   {:5} ({:.1}%)", timeouts, timeouts as f64 / count as f64 * 100.0);
    eprintln!("  Avg ticks:  {}", avg_ticks);
    eprintln!("Wrote {} results to {}", count, args.output.display());

    ExitCode::SUCCESS
}

/// Run a single HvH match on a room. Returns a JSON line.
fn run_single_match(
    room: &RoomRecord,
    hero_tomls: &[bevy_game::ai::effects::HeroToml],
    enemy_tomls: &[bevy_game::ai::effects::HeroToml],
    hero_names: &[&str],
    enemy_names: &[&str],
    seed: u64,
) -> String {
    use bevy_game::ai::core::{step, SimVec2, Team, FIXED_TICK_MS};
    use bevy_game::mission::room_gen::NavGrid;
    use bevy_game::mission::sim_bridge::build_sim_with_hero_templates;
    use bevy_game::scenario::build_unified_ai;

    // Reconstruct NavGrid from room record
    let nav = room_record_to_navgrid(room);
    let grid_nav = nav.to_gridnav();

    // Build sim with hero templates, placing on spawn positions
    let mut sim = bevy_game::scenario::build_hvh_with_spawns_and_tomls(
        hero_tomls, enemy_tomls, seed,
        &room.player_spawn, &room.enemy_spawn,
    );
    sim.grid_nav = Some(grid_nav);

    let mut squad_state = build_unified_ai(&sim);

    // Compute enemy spawn centroids for seek behavior
    let hero_target = if !room.enemy_spawn.is_empty() {
        let cx = room.enemy_spawn.iter().map(|p| p.x).sum::<f32>() / room.enemy_spawn.len() as f32;
        let cy = room.enemy_spawn.iter().map(|p| p.y).sum::<f32>() / room.enemy_spawn.len() as f32;
        SimVec2 { x: cx, y: cy }
    } else {
        SimVec2 { x: room.width as f32 / 2.0, y: room.depth as f32 / 2.0 }
    };
    let enemy_target = if !room.player_spawn.is_empty() {
        let cx = room.player_spawn.iter().map(|p| p.x).sum::<f32>() / room.player_spawn.len() as f32;
        let cy = room.player_spawn.iter().map(|p| p.y).sum::<f32>() / room.player_spawn.len() as f32;
        SimVec2 { x: cx, y: cy }
    } else {
        SimVec2 { x: room.width as f32 / 2.0, y: room.depth as f32 / 2.0 }
    };

    let max_ticks = 5000u64;
    let mut outcome = "Timeout";

    for _ in 0..max_ticks {
        let mut intents = bevy_game::ai::squad::generate_intents(&sim, &mut squad_state, FIXED_TICK_MS);

        // Seek override: units far from all enemies move directly toward enemy spawn.
        // This forces engagement instead of units wandering around obstacles.
        for intent in &mut intents {
            let Some(unit) = sim.units.iter().find(|u| u.id == intent.unit_id && u.hp > 0) else {
                continue;
            };
            if unit.casting.is_some() || unit.control_remaining_ms > 0 {
                continue;
            }
            // Find nearest enemy distance
            let nearest_enemy_dist = sim.units.iter()
                .filter(|e| e.team != unit.team && e.hp > 0)
                .map(|e| bevy_game::ai::core::distance(unit.position, e.position))
                .fold(f32::MAX, f32::min);
            // If no enemy within 8 cells, override to seek toward enemy spawn
            if nearest_enemy_dist > 8.0 {
                let target = if unit.team == Team::Hero { hero_target } else { enemy_target };
                intent.action = bevy_game::ai::core::IntentAction::MoveTo { position: target };
            }
        }

        let (new_sim, _events) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();

        if enemies_alive == 0 {
            outcome = "Victory";
            break;
        }
        if heroes_alive == 0 {
            outcome = "Defeat";
            break;
        }
    }

    let hero_hp: i32 = sim.units.iter()
        .filter(|u| u.team == Team::Hero && u.hp > 0)
        .map(|u| u.hp)
        .sum();
    let enemy_hp: i32 = sim.units.iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0)
        .map(|u| u.hp)
        .sum();

    format!(
        r#"{{"room_seed":{},"room_type":"{}","width":{},"depth":{},"heroes":[{}],"enemies":[{}],"outcome":"{}","ticks":{},"hero_hp_remaining":{},"enemy_hp_remaining":{}}}"#,
        room.seed,
        room.room_type,
        room.width,
        room.depth,
        hero_names.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(","),
        enemy_names.iter().map(|n| format!("\"{}\"", n)).collect::<Vec<_>>().join(","),
        outcome,
        sim.tick,
        hero_hp,
        enemy_hp,
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct RoomRecord {
    seed: u64,
    room_type: String,
    width: usize,
    depth: usize,
    obstacle_type: Vec<Vec<u8>>,
    elevation: Vec<Vec<f32>>,
    player_spawn: Vec<bevy_game::ai::core::SimVec2>,
    enemy_spawn: Vec<bevy_game::ai::core::SimVec2>,
}

fn load_room_records(path: &PathBuf) -> Vec<RoomRecord> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() { continue; }
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let width = v["width"].as_u64().unwrap_or(0) as usize;
        let depth = v["depth"].as_u64().unwrap_or(0) as usize;
        if width < 8 || depth < 8 { continue; }

        let obs = parse_grid_u8(&v["grid"]["obstacle_type"], width, depth);
        let elev = parse_grid_f32(&v["grid"]["elevation"], width, depth);
        if obs.is_none() || elev.is_none() { continue; }

        let player_spawn = parse_spawn_positions(&v["player_spawn"]);
        let enemy_spawn = parse_spawn_positions(&v["enemy_spawn"]);

        records.push(RoomRecord {
            seed: v["seed"].as_u64().unwrap_or(0),
            room_type: v["room_type"].as_str().unwrap_or("Unknown").to_string(),
            width,
            depth,
            obstacle_type: obs.unwrap(),
            elevation: elev.unwrap(),
            player_spawn,
            enemy_spawn,
        });
    }
    records
}

fn parse_spawn_positions(v: &serde_json::Value) -> Vec<bevy_game::ai::core::SimVec2> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    Some(bevy_game::ai::core::SimVec2 {
                        x: p[0].as_f64()? as f32,
                        y: p[1].as_f64()? as f32,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_grid_u8(v: &serde_json::Value, w: usize, d: usize) -> Option<Vec<Vec<u8>>> {
    let arr = v.as_array()?;
    if arr.len() != d { return None; }
    let mut out = Vec::with_capacity(d);
    for row in arr {
        let r = row.as_array()?;
        if r.len() != w { return None; }
        out.push(r.iter().map(|c| c.as_u64().unwrap_or(0) as u8).collect());
    }
    Some(out)
}

fn parse_grid_f32(v: &serde_json::Value, w: usize, d: usize) -> Option<Vec<Vec<f32>>> {
    let arr = v.as_array()?;
    if arr.len() != d { return None; }
    let mut out = Vec::with_capacity(d);
    for row in arr {
        let r = row.as_array()?;
        if r.len() != w { return None; }
        out.push(r.iter().map(|c| c.as_f64().unwrap_or(0.0) as f32).collect());
    }
    Some(out)
}

fn room_record_to_navgrid(room: &RoomRecord) -> bevy_game::mission::room_gen::NavGrid {
    let mut nav = bevy_game::mission::room_gen::NavGrid::new(room.width, room.depth, 1.0);

    // Perimeter
    nav.set_walkable_rect(0, 0, room.width - 1, 0, false);
    nav.set_walkable_rect(0, room.depth - 1, room.width - 1, room.depth - 1, false);
    nav.set_walkable_rect(0, 0, 0, room.depth - 1, false);
    nav.set_walkable_rect(room.width - 1, 0, room.width - 1, room.depth - 1, false);

    for r in 1..room.depth - 1 {
        for c in 1..room.width - 1 {
            let t = room.obstacle_type[r][c];
            let elev = room.elevation[r][c];
            if t != 0 && t != 8 {
                // Blocked
                let idx = nav.idx(c, r);
                nav.walkable[idx] = false;
            }
            if elev > 0.1 {
                nav.set_elevation_rect(c, r, c, r, elev);
            }
        }
    }
    nav
}

fn load_hero_dir(dir: &PathBuf) -> Vec<(String, bevy_game::ai::effects::HeroToml)> {
    use bevy_game::mission::hero_templates::parse_hero_toml_with_dsl;

    let mut heroes = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Cannot read hero directory {}: {}", dir.display(), e);
            return heroes;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "toml").unwrap_or(false) {
            let name = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            // Skip non-combat templates
            if name.contains("dummy") || name.contains("punching_bag") || name.contains("fortress") {
                continue;
            }

            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    // Load companion .ability DSL file if it exists
                    let ability_path = path.with_extension("ability");
                    let dsl_content = std::fs::read_to_string(&ability_path).ok();
                    match parse_hero_toml_with_dsl(&content, dsl_content.as_deref()) {
                        Ok(toml) => heroes.push((name, toml)),
                        Err(e) => eprintln!("  Skip {}: {}", path.display(), e),
                    }
                }
                Err(e) => eprintln!("  Skip {}: {}", path.display(), e),
            }
        }
    }
    heroes.sort_by(|a, b| a.0.cmp(&b.0));
    heroes
}

struct LcgSimple(u64);
impl LcgSimple {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e3779b97f4a7c15))
    }
    fn next(&mut self) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }
}
