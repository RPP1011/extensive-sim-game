//! Render Entry room layouts as ASCII art.
//!
//! Run with: cargo test --test terrain_visualize -- --nocapture

use bevy_game::mission::room_gen::generate_room;
use bevy_game::game_core::RoomType;

fn render_room(seed: u64) {
    let layout = generate_room(seed, RoomType::Entry);
    let nav = &layout.nav;
    let cols = nav.cols;
    let rows = nav.rows;

    // Collect spawn positions as grid cells
    let player_cells: Vec<(usize, usize)> = layout
        .player_spawn
        .positions
        .iter()
        .map(|p| nav.cell_of(*p))
        .collect();
    let enemy_cells: Vec<(usize, usize)> = layout
        .enemy_spawn
        .positions
        .iter()
        .map(|p| nav.cell_of(*p))
        .collect();

    println!("=== Entry room seed={seed} ({}x{}) ===", cols, rows);

    // Header with column numbers
    print!("   ");
    for c in 0..cols {
        print!("{}", c % 10);
    }
    println!();

    for r in 0..rows {
        print!("{:2} ", r);
        for c in 0..cols {
            let idx = r * cols + c;
            let is_player = player_cells.contains(&(c, r));
            let is_enemy = enemy_cells.contains(&(c, r));
            let walkable = nav.walkable[idx];
            let elev = nav.elevation[idx];

            let ch = if is_player {
                'P'
            } else if is_enemy {
                'E'
            } else if !walkable {
                if r == 0 || r == rows - 1 || c == 0 || c == cols - 1 {
                    '#'
                } else {
                    '█'
                }
            } else if elev > 0.8 {
                '▲'
            } else if elev > 0.3 {
                '△'
            } else {
                '·'
            };
            print!("{ch}");
        }
        println!();
    }

    // Legend
    let blocked_interior = (1..rows-1)
        .flat_map(|r| (1..cols-1).map(move |c| (r, c)))
        .filter(|&(r, c)| !nav.walkable[r * cols + c])
        .count();
    let total_interior = (rows - 2) * (cols - 2);
    let pct = blocked_interior as f32 / total_interior as f32 * 100.0;
    println!(
        "    P=hero spawn  E=enemy spawn  █=obstacle  #=wall  △/▲=elevation  ·=open"
    );
    println!(
        "    Blocked: {blocked_interior}/{total_interior} ({pct:.0}%)  Player spawns: {}  Enemy spawns: {}",
        layout.player_spawn.positions.len(),
        layout.enemy_spawn.positions.len(),
    );
    println!();
}

#[test]
fn show_entry_rooms() {
    for seed in [0, 1, 2, 3, 42, 99, 123, 777] {
        render_room(seed);
    }
}
