//! edgeworld PNG frame dump + population trace (Task 6 / Task 4).
//!
//! NOT a pin — a diagnostic/visualization companion to
//! `edgeworld_pin.rs`. Seeds a compact survival world (food oasis +
//! survivor crowd + a wolf pack on the rim), then steps the survival sim
//! and every `FRAME_EVERY` ticks renders a 256×256 top-down PNG into
//! `target/edgeworld_frames/`:
//!   * FoodNode  → green, brightness scaled by quantity (mana/food_max)
//!   * Survivor  → amber dot (dead survivors are skipped)
//!   * Wolf      → red blob (the predators; Task 4)
//!
//! FRAME BOUNDS ARE DYNAMIC. Phase 1 has no world bounds and Flee
//! dominates SeekFood, so chased survivors drift well past the seeded
//! world edge (observed |coord| ~12.6 against a world half of 8) before
//! they starve in the wilderness. A fixed `WORLD_HALF` viewport would push
//! the most interesting part of the hunt off-frame. Instead each frame
//! computes a square viewport that fits every alive agent (plus a margin),
//! so the scatter stays legible no matter how far the chase ranges. The
//! seed world-half is retained only for the initial-spread seeding.
//!
//! It also records the alive-survivor count per frame and prints a
//! population trace + ASCII sparkline.
//!
//! Run with:
//!   cargo test -p sims --test edgeworld_render --release -- --nocapture

use sims::edgeworld::GeneratedRuntime;

mod edgeworld_common;
use edgeworld_common::*;

const SEED: u64 = 0xED6E_0006;

// Tuned to the Phase 1 predator scenario (mirrors edgeworld_pin.rs
// edgeworld_predators_reduce_remnant: N_SURV=28, N_FOODN=3, N_WOLVES=4):
// an over-seeded survivor crowd at the oasis, a 4-wolf pack on the rim.
// The pack chases survivors off-world and culls them to extinction in the
// opening ~40 ticks; the wolves then persist (low wolf_hunger_rate).
const N_FOOD: usize = 3;
const N_SURVIVORS: usize = 28;
const N_WOLVES: usize = 4;
const N_TOTAL: usize = N_FOOD + N_SURVIVORS + N_WOLVES; // 35

// Slot layout matches the shared seeder: [0..N_FOOD) food, then
// survivors, then wolves. Used only for the render-side fan-out offset.
const SURVIVOR_BASE: usize = N_FOOD;

const WORLD_HALF: f32 = 8.0; // seed spread half-extent (initial layout only;
                             // the render viewport is computed per-frame)

const TICKS: u32 = 600;
// The whole hunt (28 → 0 survivors) plays out in the opening ~40 ticks
// as the pack chases the crowd off-world. Sample densely early so the
// chase is captured frame-to-frame, then coarsely once it is just the
// surviving pack idling near the oasis.
const FRAME_EVERY: u32 = 5;
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

    seed_world(&mut state, N_SURVIVORS, N_FOOD, N_WOLVES, WORLD_HALF);

    let frames_dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .parent()
        .map(|p| p.join("edgeworld_frames"))
        .unwrap_or_else(|| std::path::PathBuf::from("target/edgeworld_frames"));
    std::fs::create_dir_all(&frames_dir).expect("create frames dir");

    let mut pop_trace: Vec<u32> = Vec::new();
    let mut wolf_trace: Vec<u32> = Vec::new();
    let mut frame_ticks: Vec<u32> = Vec::new();

    // Render the initial state (tick 0) plus frames on an adaptive
    // cadence: every FRAME_EVERY ticks through the opening hunt window
    // (tick <= HUNT_WINDOW, where the 28 → 0 cull happens), then coarsely
    // (every 30 ticks) once it is just the idling pack — so the chase is
    // captured densely without 120 near-identical late frames.
    const HUNT_WINDOW: u32 = 60;
    for tick in 0..=TICKS {
        if tick > 0 {
            state.step();
        }
        let cadence = if tick <= HUNT_WINDOW { FRAME_EVERY } else { 30 };
        if tick % cadence != 0 {
            continue;
        }

        let positions = read_positions(&mut state, N_TOTAL);
        let alive = read_alive(&mut state, N_TOTAL);
        let types = read_creature_types(&mut state, N_TOTAL);
        let mana = read_mana(&mut state, N_TOTAL);

        let mut img = vec![[18u8, 18u8, 24u8]; (IMG_SIZE * IMG_SIZE) as usize];

        // DYNAMIC VIEWPORT: fit every alive agent in-frame. Chased
        // survivors drift well past the seed world-half before they
        // starve, so a fixed viewport would clip the most active part of
        // the hunt. Compute a square half-extent that contains all alive
        // agents (centred on the world origin) plus a margin, floored at
        // the seed WORLD_HALF so an empty/quiet frame still shows the
        // oasis at a sane zoom.
        let view_half = {
            let mut m = 0.0f32;
            for i in 0..N_TOTAL {
                if alive[i] == 0 {
                    continue;
                }
                m = m.max(positions[i][0].abs()).max(positions[i][2].abs());
            }
            (m * 1.12).max(WORLD_HALF)
        };

        // Draw food first (background layer), survivors over it, wolves on top.
        for i in 0..N_TOTAL {
            if types[i] != CT_FOOD || alive[i] == 0 {
                continue;
            }
            let q = (mana[i] / FOOD_MAX).clamp(0.0, 1.0);
            // Dim floor of 40 so even near-depleted food is visible.
            let g = (40.0 + q * 215.0) as u8;
            let color = [10u8, g, 20u8];
            let (px, py) = world_to_px(positions[i], view_half, IMG_SIZE);
            draw_blob(&mut img, IMG_SIZE, px, py, color);
        }
        for i in 0..N_TOTAL {
            if types[i] != CT_SURVIVOR || alive[i] == 0 {
                continue;
            }
            let color = [240u8, 170u8, 40u8]; // amber
            let (px, py) = world_to_px(positions[i], view_half, IMG_SIZE);
            // Render-only fan-out: the surviving remnant converges onto a
            // single food node (SeekFood is first-candidate + the eat
            // query spans the compact world), so co-located survivors
            // would stack on one pixel and undercount the remnant
            // visually. Spread each survivor by a small deterministic
            // per-slot offset on a ring so the huddle reads as a legible
            // cluster of distinct dots. This is a visualization choice
            // only — the simulation positions are untouched.
            let s_idx = i.saturating_sub(SURVIVOR_BASE);
            let ang = (s_idx as f32) * 2.399_963; // golden angle
            let ring = 3.0 + 2.0 * ((s_idx % 3) as f32); // 3..7 px
            let jx = (px as i32 + (ring * ang.cos()) as i32).clamp(0, (IMG_SIZE - 1) as i32) as u32;
            let jy = (py as i32 + (ring * ang.sin()) as i32).clamp(0, (IMG_SIZE - 1) as i32) as u32;
            draw_blob(&mut img, IMG_SIZE, jx, jy, color);
        }
        // Wolves on top — red, drawn last so a predator on a kill reads
        // clearly over its prey. A fatter 5×5 blob so the pack stands out
        // against the amber crowd.
        //
        // Render-only fan-out (mirror of the survivor spread): the seeded
        // pack is rotationally symmetric on the rim and every wolf pursues
        // the same first-in-range survivor identically, so all K wolves
        // travel in lockstep and collapse onto one pixel — they would draw
        // as a single red dot and hide the pack size. Nudge each wolf by a
        // small deterministic per-slot ring offset so the lockstep pack
        // reads as a legible cluster of distinct predators. Visualization
        // only; sim positions are untouched.
        let wolf_base = N_FOOD + N_SURVIVORS;
        for i in 0..N_TOTAL {
            if types[i] != CT_WOLF || alive[i] == 0 {
                continue;
            }
            let color = [220u8, 40u8, 40u8]; // red
            let (px, py) = world_to_px(positions[i], view_half, IMG_SIZE);
            let w_idx = i.saturating_sub(wolf_base);
            let ang = (w_idx as f32) * 1.5708 + 0.3; // quarter-turns apart
            let ring = 6.0; // px — small pack cluster
            let jx = (px as i32 + (ring * ang.cos()) as i32).clamp(0, (IMG_SIZE - 1) as i32) as u32;
            let jy = (py as i32 + (ring * ang.sin()) as i32).clamp(0, (IMG_SIZE - 1) as i32) as u32;
            draw_fat_blob(&mut img, IMG_SIZE, jx, jy, color);
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

        // Population: alive survivors + alive wolves.
        let n_alive: u32 = (0..N_TOTAL)
            .filter(|&i| types[i] == CT_SURVIVOR && alive[i] != 0)
            .count() as u32;
        let n_wolves: u32 = (0..N_TOTAL)
            .filter(|&i| types[i] == CT_WOLF && alive[i] != 0)
            .count() as u32;
        pop_trace.push(n_alive);
        wolf_trace.push(n_wolves);
        frame_ticks.push(tick);
    }

    // Population trace + sparkline.
    println!();
    println!("==== edgeworld population trace ====");
    println!(
        "  seed world: {}×{} (half {WORLD_HALF}), {N_FOOD} food + {N_SURVIVORS} survivors + {N_WOLVES} wolves, {TICKS} ticks (dynamic viewport)",
        (WORLD_HALF * 2.0) as i32,
        (WORLD_HALF * 2.0) as i32,
    );
    println!();
    for ((t, n), w) in frame_ticks.iter().zip(pop_trace.iter()).zip(wolf_trace.iter()) {
        let bar = "#".repeat(*n as usize);
        let wbar = "x".repeat(*w as usize);
        println!("  tick {t:4}  survivors {n:2} {bar}  | wolves {w} {wbar}");
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

/// Stamp a 5×5 blob of `color` centered at `(cx, cy)` — used for wolves
/// so the predators read as bigger, bolder marks than survivors/food.
fn draw_fat_blob(img: &mut [[u8; 3]], img_size: u32, cx: u32, cy: u32, color: [u8; 3]) {
    let s = img_size as i32;
    for dy in -2..=2 {
        for dx in -2..=2 {
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
