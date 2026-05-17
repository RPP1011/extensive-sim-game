//! Visualization companion to `assassination_threat_test_pin.rs`.
//!
//! Runs the same scenario but renders ASCII frames of the 16×16 city
//! at six representative attempts (the warmup quartet plus mid-run
//! and final). For each attempt shows:
//!   * The guard layout produced by the planner's redistribution
//!     read of `threat(king, q)` after the prior attempt.
//!   * The assassin's spawn position (★) in its approach quadrant.
//!   * A trace (·) sampled every 4 ticks across the attempt.
//!   * The end-of-attempt position (A) and outcome marker (✕ on
//!     assassin death, ☠ on king death, … on timeout).
//!
//! Not pinned — purely diagnostic. The pin (and behavioural Δ
//! claims) live in `assassination_threat_test_pin.rs`. Run with
//!   `cargo test -p sims --test assassination_visualize --release -- --nocapture`
//! to see the frames.

#![allow(non_snake_case)]

use sims::assassination_threat_test::GeneratedRuntime;

use engine_voxel::{
    build_navgrid, Aabb, NavgridIndex, VoxelRegion, VoxelRegionBounds, VoxelRegionKind,
    VoxelRegionRegistry,
};

const SEED: u64 = 0xA55A_551_DEAD_BEEF; // matches the pin's seed
const N_ASSASSINS: u32 = 1;
const N_GUARDS: u32 = 8;
const N_KING: u32 = 1;
const N_TOTAL: u32 = N_ASSASSINS + N_GUARDS + N_KING; // 10

const CT_ASSASSIN: u32 = 0;
const CT_GUARD: u32 = 1;
const CT_KING: u32 = 2;

const ASSASSIN_SLOT: usize = 0;
const GUARD_SLOT_BASE: usize = 1;
const KING_SLOT: usize = 9;

const NUM_QUADRANTS: u32 = 4;
const NUM_ATTEMPTS: u32 = 50;
const MAX_TICKS_PER_ATTEMPT: u32 = 80;
const HALF_EXTENT: f32 = 8.0;
const GRID_SIZE: usize = 16;
const TRACE_EVERY: u32 = 4;

const CITY_MIN: [f32; 3] = [-HALF_EXTENT, 0.0, -HALF_EXTENT];
const CITY_MAX: [f32; 3] = [HALF_EXTENT, 4.0, HALF_EXTENT];

// Which attempts to render. Mix of warmup + mid + final.
const FRAMES: &[u32] = &[0, 1, 2, 3, 24, 49];

#[test]
fn render_assassin_scenario_frames() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[assassination_visualize] skipping: no wgpu adapter");
            return;
        }
    };

    let mut registry = VoxelRegionRegistry::new();
    let id = registry.register(
        VoxelRegionKind(0),
        VoxelRegionBounds::Aabb(Aabb {
            min: CITY_MIN,
            max: CITY_MAX,
        }),
        0,
    );
    let region: &VoxelRegion = registry.get(id).expect("region just registered");
    let navgrid: NavgridIndex = build_navgrid(region, |_x, y, _z| y == 0)
        .expect("navgrid build");
    state.upload_navgrid(&navgrid);

    let mut guard_quadrants: [u32; N_GUARDS as usize] = [0, 0, 1, 1, 2, 2, 3, 3];
    let mut last_threat: [f32; 4] = [0.0; 4];

    println!();
    println!("==== assassination scenario frames ====");
    println!();
    println!("  Legend:  K = king   G = guard   A = assassin (end)   ★ = spawn");
    println!("           · = assassin trail (every {TRACE_EVERY} ticks)");
    println!("           ☠ = king died here    ✕ = assassin died here");
    println!("           . = walkable floor");
    println!();
    println!("  City is 16×16; origin (cell 0,0) maps to world (-8,-8).");
    println!();

    for attempt in 0..NUM_ATTEMPTS {
        let approach_q = decide_approach_q(attempt);

        // Drain residual chronicle from prior attempt — same flow as
        // the pin.
        if attempt > 0 {
            kill_all(&mut state);
            state.step();
            state.step();
        }
        seed_attempt(&mut state, approach_q, &guard_quadrants);

        let render_this = FRAMES.contains(&attempt);

        // Capture trace samples + outcome.
        let mut trail: Vec<(u32, [f32; 3])> = Vec::new();
        if render_this {
            trail.push((0, read_pos(&mut state, ASSASSIN_SLOT)));
        }

        let mut outcome = Outcome::Timeout;
        let mut last_tick_run: u32 = MAX_TICKS_PER_ATTEMPT;
        for tick in 0..MAX_TICKS_PER_ATTEMPT {
            state.step();
            if render_this && (tick + 1) % TRACE_EVERY == 0 {
                trail.push((tick + 1, read_pos(&mut state, ASSASSIN_SLOT)));
            }
            let king_alive = read_alive_at(&mut state, KING_SLOT);
            if !king_alive && tick >= 18 {
                outcome = Outcome::KingKilled(tick);
                last_tick_run = tick;
                break;
            }
            let assassin_alive = read_alive_at(&mut state, ASSASSIN_SLOT);
            if !assassin_alive && tick >= 18 {
                outcome = Outcome::AssassinKilled(tick);
                last_tick_run = tick;
                break;
            }
        }

        if render_this {
            let guard_positions: Vec<[f32; 3]> = (0..N_GUARDS as usize)
                .map(|i| read_pos(&mut state, GUARD_SLOT_BASE + i))
                .collect();
            let king_pos = read_pos(&mut state, KING_SLOT);
            let assassin_pos = read_pos(&mut state, ASSASSIN_SLOT);
            let spawn_pos = trail.first().map(|(_, p)| *p).unwrap_or(assassin_pos);

            print_header(attempt, approach_q, &outcome, &last_threat);
            print_frame(
                &guard_positions,
                king_pos,
                assassin_pos,
                spawn_pos,
                &trail,
                &outcome,
            );
            println!(
                "    end pos: A=({:+.1},{:+.1})  K=({:+.1},{:+.1})  ran {last_tick_run} ticks",
                assassin_pos[0], assassin_pos[2], king_pos[0], king_pos[2],
            );
            println!();
        }

        // Planner reads the threat belief AFTER the run; redistribute
        // guards proportionally for the next attempt.
        last_threat = read_threat_belief(&mut state);
        let top_q = argmax(&last_threat);
        guard_quadrants = redistribute_guards(&last_threat);
        let _ = top_q;
    }
}

#[derive(Debug)]
enum Outcome {
    KingKilled(u32),
    AssassinKilled(u32),
    Timeout,
}

fn print_header(attempt: u32, approach_q: u32, outcome: &Outcome, prior_threat: &[f32; 4]) {
    let result = match outcome {
        Outcome::KingKilled(t) => format!("SUCCESS (king ☠ at tick {t})"),
        Outcome::AssassinKilled(t) => format!("THWARTED (assassin ✕ at tick {t})"),
        Outcome::Timeout => "TIMEOUT (king survived 80 ticks)".to_string(),
    };
    let q_name = ["NE (+x +z)", "NW (-x +z)", "SW (-x -z)", "SE (+x -z)"][approach_q as usize];
    println!(
        "  attempt {attempt:2}  approach q={approach_q} {q_name}  →  {result}",
    );
    println!(
        "    pre-attempt threat(king, q) = [{:.1}, {:.1}, {:.1}, {:.1}]",
        prior_threat[0], prior_threat[1], prior_threat[2], prior_threat[3],
    );
}

fn print_frame(
    guard_positions: &[[f32; 3]],
    king_pos: [f32; 3],
    assassin_pos: [f32; 3],
    spawn_pos: [f32; 3],
    trail: &[(u32, [f32; 3])],
    outcome: &Outcome,
) {
    let mut grid: Vec<Vec<char>> = vec![vec!['.'; GRID_SIZE]; GRID_SIZE];

    // Trail dots first (so spawn / end overwrite).
    for (_tick, p) in trail.iter().skip(1) {
        if let Some((cx, cz)) = world_to_cell(*p) {
            grid[cz][cx] = '·';
        }
    }
    // Spawn marker.
    if let Some((cx, cz)) = world_to_cell(spawn_pos) {
        grid[cz][cx] = '★';
    }
    // Guards.
    for gp in guard_positions {
        if let Some((cx, cz)) = world_to_cell(*gp) {
            // Don't stomp the king cell with a guard label.
            if grid[cz][cx] != 'K' {
                grid[cz][cx] = 'G';
            }
        }
    }
    // King — always last so it takes precedence at origin.
    if let Some((cx, cz)) = world_to_cell(king_pos) {
        grid[cz][cx] = 'K';
    }
    // Assassin end-position (or outcome marker).
    if let Some((cx, cz)) = world_to_cell(assassin_pos) {
        let marker = match outcome {
            Outcome::KingKilled(_) => '☠',
            Outcome::AssassinKilled(_) => '✕',
            Outcome::Timeout => 'A',
        };
        // Don't double-stamp on the king cell unless king died.
        match outcome {
            Outcome::KingKilled(_) => grid[cz][cx] = '☠',
            _ => {
                if grid[cz][cx] != 'K' {
                    grid[cz][cx] = marker;
                }
            }
        }
    }

    // Print top-down: world +z is "north" → render cz descending so
    // higher z is the top row.
    println!("       +---+ {} -+", "-".repeat(GRID_SIZE * 2 - 4));
    for cz in (0..GRID_SIZE).rev() {
        let line: String = grid[cz]
            .iter()
            .map(|c| format!(" {c}"))
            .collect::<Vec<_>>()
            .join("");
        let z_label = (cz as i32) - 8;
        println!("       | z={z_label:+3} {line} |");
    }
    println!("       +---+ {} -+", "-".repeat(GRID_SIZE * 2 - 4));
    let x_axis: String = (0..GRID_SIZE)
        .map(|cx| {
            let x = (cx as i32) - 8;
            if cx == 0 || cx == 8 || cx == GRID_SIZE - 1 {
                format!("{x:+2}")
            } else {
                "  ".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("");
    println!("              {x_axis}");
}

fn world_to_cell(p: [f32; 3]) -> Option<(usize, usize)> {
    let cx = ((p[0] + HALF_EXTENT).floor() as i32).clamp(0, GRID_SIZE as i32 - 1) as usize;
    let cz = ((p[2] + HALF_EXTENT).floor() as i32).clamp(0, GRID_SIZE as i32 - 1) as usize;
    Some((cx, cz))
}

// --- Reused from the pin (verbatim modulo namespace).

fn decide_approach_q(attempt: u32) -> u32 {
    if attempt < 4 {
        attempt
    } else if (attempt % 3) == 0 {
        0
    } else {
        (attempt + 1) % NUM_QUADRANTS
    }
}

fn argmax(v: &[f32; 4]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = v[0];
    for i in 1..4 {
        if v[i] > best_v {
            best_v = v[i];
            best_i = i as u32;
        }
    }
    best_i
}

fn redistribute_guards(threat: &[f32; 4]) -> [u32; N_GUARDS as usize] {
    let sum: f32 = threat.iter().sum();
    let mut counts: [u32; 4] = [1, 1, 1, 1];
    let mut remaining: u32 = N_GUARDS - 4;
    if sum > 0.0 {
        let mut ranked: Vec<(usize, f32)> = threat.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut i = 0;
        while remaining > 0 {
            counts[ranked[i % 4].0] += 1;
            remaining -= 1;
            i += 1;
        }
    } else {
        for q in 0..4 {
            counts[q] += 1;
        }
        counts[0] -= 1;
    }
    let mut out = [0u32; N_GUARDS as usize];
    let mut idx = 0usize;
    for q in 0..NUM_QUADRANTS {
        for _ in 0..counts[q as usize] {
            if idx < N_GUARDS as usize {
                out[idx] = q;
                idx += 1;
            }
        }
    }
    out
}

fn quadrant_pos(q: u32, radius: f32) -> (f32, f32) {
    let (sx, sz): (f32, f32) = match q {
        0 => (1.0, 1.0),
        1 => (-1.0, 1.0),
        2 => (-1.0, -1.0),
        3 => (1.0, -1.0),
        _ => unreachable!(),
    };
    (sx * radius, sz * radius)
}

fn seed_attempt(
    state: &mut GeneratedRuntime,
    approach_q: u32,
    guard_quadrants: &[u32; N_GUARDS as usize],
) {
    let n = N_TOTAL as usize;
    let mut positions: Vec<[f32; 4]> = vec![[0.0; 4]; n];
    let (ax, az) = quadrant_pos(approach_q, HALF_EXTENT - 1.0);
    positions[ASSASSIN_SLOT] = [ax, 0.0, az, 0.0];
    for (i, q) in guard_quadrants.iter().enumerate() {
        let (gx, gz) = quadrant_pos(*q, 3.0);
        let jitter = (i as f32) * 0.4 - 1.6;
        positions[GUARD_SLOT_BASE + i] = [gx + jitter * 0.3, 0.0, gz + jitter * 0.3, 0.0];
    }
    positions[KING_SLOT] = [0.0, 0.0, 0.0, 0.0];

    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );

    let mut ct: Vec<u32> = Vec::with_capacity(n);
    ct.push(CT_ASSASSIN);
    for _ in 0..N_GUARDS {
        ct.push(CT_GUARD);
    }
    ct.push(CT_KING);
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&ct),
    );
    let alive: Vec<u32> = vec![1u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
    let hp: Vec<f32> = vec![100.0_f32; n];
    state.gpu.queue.write_buffer(
        &state.agent_hp_buf,
        0,
        bytemuck::cast_slice(&hp),
    );
}

fn kill_all(state: &mut GeneratedRuntime) {
    let n = N_TOTAL as usize;
    let dead: Vec<u32> = vec![0u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&dead),
    );
}

fn read_alive_at(state: &mut GeneratedRuntime, slot: usize) -> bool {
    let buf = state.agent_alive_buf.clone();
    let v = readback_u32(state, &buf, N_TOTAL as usize);
    v[slot] != 0
}

fn read_pos(state: &mut GeneratedRuntime, slot: usize) -> [f32; 3] {
    let n = N_TOTAL as usize;
    let bytes = (n as u64) * 16;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viz::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("viz::pos_readback") },
    );
    enc.copy_buffer_to_buffer(&state.agent_pos_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let v = slice.get_mapped_range();
        let f: &[[f32; 4]] = bytemuck::cast_slice(&v);
        let p = f[slot];
        [p[0], p[1], p[2]]
    };
    staging.unmap();
    out
}

fn read_threat_belief(state: &mut GeneratedRuntime) -> [f32; 4] {
    let cell_count = (N_TOTAL as usize) * 4;
    let buf = state.view_storage_threat_primary_buf.clone();
    let words = readback_u32(state, &buf, cell_count);
    let mut out = [0.0; 4];
    for q in 0..4 {
        out[q] = f32::from_bits(words[KING_SLOT * 4 + q]);
    }
    out
}

fn readback_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viz::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("viz::u32_readback") },
    );
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let v = slice.get_mapped_range();
        let w: &[u32] = bytemuck::cast_slice(&v);
        w[..count].to_vec()
    };
    staging.unmap();
    out
}
