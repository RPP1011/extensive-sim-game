//! Preview room layouts as ASCII grids.
//!
//! Usage:
//!   cargo run --bin room_preview                     # all scenarios
//!   cargo run --bin room_preview -- Entry 42         # specific type + seed
//!   cargo run --bin room_preview -- all              # one of each type, seed 42

use bevy_game::game_core::RoomType;
use bevy_game::mission::room_gen::{generate_room, NavGrid, SpawnZone};

fn all_room_types() -> Vec<(&'static str, RoomType)> {
    vec![
        ("Entry", RoomType::Entry),
        ("Pressure", RoomType::Pressure),
        ("Pivot", RoomType::Pivot),
        ("Setpiece", RoomType::Setpiece),
        ("Recovery", RoomType::Recovery),
        ("Climax", RoomType::Climax),
    ]
}

fn render_room(nav: &NavGrid, player_spawn: &SpawnZone, enemy_spawn: &SpawnZone) -> String {
    // Build a char grid.
    let mut grid = vec![vec!['.'; nav.cols]; nav.rows];

    // Mark blocked cells.
    for r in 0..nav.rows {
        for c in 0..nav.cols {
            let idx = r * nav.cols + c;
            if !nav.walkable[idx] {
                grid[r][c] = '#';
            } else if nav.elevation[idx] > 0.0 {
                grid[r][c] = '~';
            }
        }
    }

    // Mark spawn positions.
    for pos in &player_spawn.positions {
        let (c, r) = nav.cell_of(*pos);
        if r < nav.rows && c < nav.cols {
            grid[r][c] = 'P';
        }
    }
    for pos in &enemy_spawn.positions {
        let (c, r) = nav.cell_of(*pos);
        if r < nav.rows && c < nav.cols {
            grid[r][c] = 'E';
        }
    }

    let mut out = String::new();
    for row in &grid {
        let line: String = row.iter().collect();
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn print_room(name: &str, room_type: RoomType, seed: u64) {
    let layout = generate_room(seed, room_type);
    let (w, d) = (layout.width as usize, layout.depth as usize);
    let ps = layout.player_spawn.positions.len();
    let es = layout.enemy_spawn.positions.len();

    println!("=== {name} ({w}x{d}) seed={seed} | spawns: P={ps} E={es} ===");
    print!("{}", render_room(&layout.nav, &layout.player_spawn, &layout.enemy_spawn));
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.len() {
        0 => {
            // Load all scenario files.
            let scenario_dir = std::path::Path::new("scenarios");
            if !scenario_dir.is_dir() {
                eprintln!("No scenarios/ directory found, showing one of each type instead.");
                for (name, rt) in all_room_types() {
                    print_room(name, rt, 42);
                }
                return;
            }

            let mut entries: Vec<_> = std::fs::read_dir(scenario_dir)
                .expect("failed to read scenarios/")
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
                .collect();
            entries.sort_by_key(|e| e.file_name());

            for entry in entries {
                let path = entry.path();
                let content = std::fs::read_to_string(&path).expect("failed to read file");
                let file: bevy_game::scenario::ScenarioFile =
                    toml::from_str(&content).expect("failed to parse TOML");
                let cfg = &file.scenario;
                let room_type =
                    RoomType::from_str(&cfg.room_type).unwrap_or(RoomType::Entry);
                let label = format!(
                    "{} [{}v{}]",
                    cfg.name, cfg.hero_count, cfg.enemy_count
                );
                print_room(&label, room_type, cfg.seed);
            }
        }
        1 if args[0] == "all" => {
            for (name, rt) in all_room_types() {
                print_room(name, rt, 42);
            }
        }
        2 => {
            let rt_str = &args[0];
            let seed: u64 = args[1].parse().expect("second arg must be a seed (u64)");
            let room_type = RoomType::from_str(rt_str)
                .unwrap_or_else(|| panic!("unknown room type: {rt_str}"));
            print_room(rt_str, room_type, seed);
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  room_preview              # all scenarios");
            eprintln!("  room_preview all          # one of each type, seed 42");
            eprintln!("  room_preview Entry 42     # specific type + seed");
            std::process::exit(1);
        }
    }
}
