//! edgeworld Phase 0 PNG frame dump + population trace (Task 6).
//!
//! NOT a pin — a diagnostic/visualization companion to
//! `edgeworld_pin.rs`. Seeds a compact 16×16 world (the `@spatial`
//! perception ring caps survivor foraging at ~±1 cell / ~6 world
//! units, so the world must be small enough that survivors can
//! actually see food), then steps the survival sim and every
//! `FRAME_EVERY` ticks renders a 256×256 top-down PNG into
//! `target/edgeworld_frames/`:
//!   * FoodNode  → green, brightness scaled by quantity (mana/food_max)
//!   * Survivor  → amber dot (dead survivors are skipped)
//!
//! It also records the alive-survivor count per frame and prints a
//! population trace + ASCII sparkline. The bar for Task 6 is: frames
//! render readably and the trace prints. Boom/bust tuning is Task 7.
//!
//! Run with:
//!   cargo test -p sims --test edgeworld_render --release -- --nocapture

use sims::edgeworld::GeneratedRuntime;

mod edgeworld_common;
use edgeworld_common::*;

const SEED: u64 = 0xED6E_0006;

const N_FOOD: usize = 8;
const N_SURVIVORS: usize = 12;
const N_TOTAL: usize = N_FOOD + N_SURVIVORS; // 20

// Slot layout: [0..8) = FoodNode, [8..20) = Survivor. Kept consistent
// everywhere below.
const FOOD_BASE: usize = 0;
const SURVIVOR_BASE: usize = N_FOOD;

const WORLD_HALF: f32 = 8.0; // world spans [-8, 8], 16 wide
const FOOD_MAX: f32 = 5.0;

const TICKS: u32 = 600;
const FRAME_EVERY: u32 = 30;
const IMG_SIZE: u32 = 256;

#[test]
fn edgeworld_render_frames() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL as u32) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld_render] skipping: no wgpu adapter on host.");
            return;
        }
    };

    seed_world(&mut state, N_SURVIVORS, N_FOOD, WORLD_HALF);

    let frames_dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .parent()
        .map(|p| p.join("edgeworld_frames"))
        .unwrap_or_else(|| std::path::PathBuf::from("target/edgeworld_frames"));
    std::fs::create_dir_all(&frames_dir).expect("create frames dir");

    let mut pop_trace: Vec<u32> = Vec::new();
    let mut frame_ticks: Vec<u32> = Vec::new();

    // Render the initial state (tick 0) plus every FRAME_EVERY ticks.
    for tick in 0..=TICKS {
        if tick > 0 {
            state.step();
        }
        if tick % FRAME_EVERY != 0 {
            continue;
        }

        let positions = read_positions(&mut state, N_TOTAL);
        let alive = read_alive(&mut state, N_TOTAL);
        let types = read_creature_types(&mut state, N_TOTAL);
        let mana = read_mana(&mut state, N_TOTAL);

        let mut img = vec![[18u8, 18u8, 24u8]; (IMG_SIZE * IMG_SIZE) as usize];

        // Draw food first (background layer), survivors on top.
        for i in 0..N_TOTAL {
            if types[i] != CT_FOOD || alive[i] == 0 {
                continue;
            }
            let q = (mana[i] / FOOD_MAX).clamp(0.0, 1.0);
            // Dim floor of 40 so even near-depleted food is visible.
            let g = (40.0 + q * 215.0) as u8;
            let color = [10u8, g, 20u8];
            let (px, py) = world_to_px(positions[i], WORLD_HALF, IMG_SIZE);
            draw_blob(&mut img, IMG_SIZE, px, py, color);
        }
        for i in 0..N_TOTAL {
            if types[i] != CT_SURVIVOR || alive[i] == 0 {
                continue;
            }
            let color = [240u8, 170u8, 40u8]; // amber
            let (px, py) = world_to_px(positions[i], WORLD_HALF, IMG_SIZE);
            draw_blob(&mut img, IMG_SIZE, px, py, color);
        }

        // Flatten to RGB bytes and write PNG.
        let mut bytes = Vec::with_capacity(img.len() * 3);
        for px in &img {
            bytes.extend_from_slice(px);
        }
        let path = frames_dir.join(format!("frame_{tick:04}.png"));
        image::save_buffer(
            &path,
            &bytes,
            IMG_SIZE,
            IMG_SIZE,
            image::ColorType::Rgb8,
        )
        .expect("write png");

        // Population: alive survivors.
        let n_alive: u32 = (0..N_TOTAL)
            .filter(|&i| types[i] == CT_SURVIVOR && alive[i] != 0)
            .count() as u32;
        pop_trace.push(n_alive);
        frame_ticks.push(tick);
    }

    // Population trace + sparkline.
    println!();
    println!("==== edgeworld population trace ====");
    println!(
        "  world: {}×{} (half {WORLD_HALF}), {N_FOOD} food + {N_SURVIVORS} survivors, {TICKS} ticks",
        (WORLD_HALF * 2.0) as i32,
        (WORLD_HALF * 2.0) as i32,
    );
    println!();
    for (t, n) in frame_ticks.iter().zip(pop_trace.iter()) {
        let bar = "#".repeat(*n as usize);
        println!("  tick {t:4}  survivors {n:2}  {bar}");
    }
    println!();
    println!("  sparkline: {}", sparkline(&pop_trace));
    println!(
        "  range: min={} max={}  (start {} → end {})",
        pop_trace.iter().min().copied().unwrap_or(0),
        pop_trace.iter().max().copied().unwrap_or(0),
        pop_trace.first().copied().unwrap_or(0),
        pop_trace.last().copied().unwrap_or(0),
    );
    println!("  frames written to: {}", frames_dir.display());
    println!();
}

/// Seed a compact survival world. Slots `[0..n_food)` are FoodNodes
/// spread across the world on a grid at full quantity; slots
/// `[n_food..n_food+n_survivors)` are Survivors clustered near the
/// center (inside the ~6-unit perception ring of some food) at zero
/// hunger. Deterministic — explicit math only, no rng.
fn seed_world(
    state: &mut GeneratedRuntime,
    n_survivors: usize,
    n_food: usize,
    world_half: f32,
) {
    let n = n_food + n_survivors;
    let mut positions: Vec<[f32; 4]> = vec![[0.0; 4]; n];
    let mut types: Vec<u32> = vec![CT_FOOD; n];
    let mut alive: Vec<u32> = vec![1u32; n];
    let mut hunger: Vec<f32> = vec![0.0f32; n];
    let mut mana: Vec<f32> = vec![0.0f32; n];
    let hp: Vec<f32> = vec![1.0f32; n]; // for the starvation fallback path

    // Food on a near-uniform grid spanning ~[-half/2, half/2] so every
    // node sits within a survivor's perception ring of the centre
    // cluster. 8 nodes → 4×2 grid.
    let cols = 4usize;
    let rows = (n_food + cols - 1) / cols; // 2
    let span = world_half * 0.75; // food occupies inner [-6, 6]
    for f in 0..n_food {
        let slot = FOOD_BASE + f;
        let cx = f % cols;
        let cy = f / cols;
        let fx = if cols > 1 {
            -span + 2.0 * span * (cx as f32) / ((cols - 1) as f32)
        } else {
            0.0
        };
        let fz = if rows > 1 {
            -span * 0.6 + 1.2 * span * (cy as f32) / ((rows - 1) as f32)
        } else {
            0.0
        };
        positions[slot] = [fx, 0.0, fz, 0.0];
        types[slot] = CT_FOOD;
        mana[slot] = FOOD_MAX;
    }

    // Survivors clustered near origin in a small ring so they fan out
    // toward different food nodes. Radius 2.0 keeps them inside the
    // perception ring of the inner food grid.
    for s in 0..n_survivors {
        let slot = SURVIVOR_BASE + s;
        let theta = (s as f32) / (n_survivors as f32) * std::f32::consts::TAU;
        let r = 2.0;
        let sx = r * theta.cos();
        let sz = r * theta.sin();
        positions[slot] = [sx, 0.0, sz, 0.0];
        types[slot] = CT_SURVIVOR;
        hunger[slot] = 0.0;
    }

    state
        .gpu
        .queue
        .write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&positions));
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&types),
    );
    state
        .gpu
        .queue
        .write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    state
        .gpu
        .queue
        .write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&hunger));
    state
        .gpu
        .queue
        .write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&mana));
    state
        .gpu
        .queue
        .write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&hp));

    // Silence unused-mut warnings on vecs we only write once.
    let _ = (&mut alive, &mut hunger, &mut mana, &mut types, &mut positions);
}

/// Map a world position `[x,_,z,_]` to pixel coords on an `img_size`
/// square. World +z renders downward (row index increases with z), x
/// rightward. Returns `(px, py)` clamped to the image.
fn world_to_px(world: [f32; 4], world_half: f32, img_size: u32) -> (u32, u32) {
    let nx = (world[0] + world_half) / (2.0 * world_half); // 0..1
    let nz = (world[2] + world_half) / (2.0 * world_half); // 0..1
    let px = (nx * (img_size as f32)).clamp(0.0, (img_size - 1) as f32) as u32;
    let py = (nz * (img_size as f32)).clamp(0.0, (img_size - 1) as f32) as u32;
    (px, py)
}

/// Stamp a 3×3 blob of `color` centered at `(cx, cy)` into the RGB
/// pixel buffer.
fn draw_blob(img: &mut [[u8; 3]], img_size: u32, cx: u32, cy: u32, color: [u8; 3]) {
    let s = img_size as i32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let x = cx as i32 + dx;
            let y = cy as i32 + dy;
            if x < 0 || y < 0 || x >= s || y >= s {
                continue;
            }
            img[(y * s + x) as usize] = color;
        }
    }
}

/// 8-level ASCII sparkline scaled to the value range.
fn sparkline(vals: &[u32]) -> String {
    const RAMP: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if vals.is_empty() {
        return String::new();
    }
    let min = *vals.iter().min().unwrap();
    let max = *vals.iter().max().unwrap();
    let span = (max - min).max(1);
    vals.iter()
        .map(|&v| {
            let idx = ((v - min) as usize * (RAMP.len() - 1)) / span as usize;
            RAMP[idx]
        })
        .collect()
}
