//! edgeworld PNG frame dump + population trace (Task 6 / Task 4).
//!
//! NOT a pin — a diagnostic/visualization companion to
//! `edgeworld_pin.rs`. Seeds a compact survival world (food oasis +
//! survivor crowd + a wolf pack on the rim), then steps the survival sim
//! and every `FRAME_EVERY` ticks renders a 256×256 top-down PNG into
//! `target/edgeworld_frames/`:
//!   * FoodNode  → green, brightness scaled by quantity (mana/food_max)
//!   * Survivor  → FEAR-TINTED dot (Phase 2 Task 3): calm survivors are
//!     amber `[240,170,40]`; as the decaying FEAR column (shield_hp, read
//!     via `read_fear`) rises the dot interpolates toward a terrified
//!     HOT-WHITE `[255,255,235]` (fear/FEAR_TINT_MAX, clamped). The white
//!     terror color is maximally distinct from BOTH calm amber AND the
//!     pure-red wolves, so a panicking survivor flashes bright-white and is
//!     never confused with a predator. The gradient amber→white is the
//!     legible fear signal: calm foragers glow amber, survivors with a wolf
//!     in sight flare white, and as fear drains (wolf passes) they fade back
//!     through orange to amber — the calm/spike/re-settle cycle.
//!   * Wolf      → pure red `[255,40,40]` fat blob (the predators), drawn
//!     last; pure red is distinct from amber AND from the white terror tint
//!     so fearful survivors are never confused with predators.
//!
//! FRAME BOUNDS ARE DYNAMIC but now effectively static: Phase 2 added world
//! bounds (config.edgeworld.world_min/max = [-16, 16] on the x/z plane), so
//! every agent is clamped on-world and no agent drifts off-frame. The
//! per-frame viewport still fits every alive agent (plus a margin) floored
//! at the seed WORLD_HALF, but since positions are bounded it stays stable
//! (no ballooning), so the bounded chase + forage saga reads cleanly frame
//! to frame.
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

// Tuned to the Phase 2 coexistence scenario (mirrors edgeworld_pin.rs
// edgeworld_predators_reduce_remnant / edgeworld_coexistence_at_600:
// N_SURV=50, N_FOODN=50, N_WOLVES=3): an over-seeded survivor crowd over a
// broad larder, a 3-wolf pack on the rim. The seeded overshoot starves
// (50 -> 42), the pack reaches the cluster ~tick 22 and culls hard
// (42 -> 12), then the faster prey escape to a defensible foraging remnant
// while the pack persists on the feast: the run ends in a STABLE MIXED
// EQUILIBRIUM (both survivors and wolves alive at tick 600) — a real
// bounded ecosystem. (NOTE: this render reads agent buffers every frame,
// which perturbs the GPU trajectory slightly vs the pins' end-only reads —
// a known determinism sensitivity — so the per-frame counts here may differ
// in detail from the pin trajectory, but the qualitative saga is the same.)
const N_FOOD: usize = 50;
const N_SURVIVORS: usize = 50;
const N_WOLVES: usize = 3;
const N_TOTAL: usize = N_FOOD + N_SURVIVORS + N_WOLVES; // 103

// Slot layout matches the shared seeder: [0..N_FOOD) food, then
// survivors, then wolves. Used only for the render-side fan-out offset.
const SURVIVOR_BASE: usize = N_FOOD;

const WORLD_HALF: f32 = 11.0; // seed spread half-extent (initial layout only;
                              // the render viewport is computed per-frame)

const TICKS: u32 = 600;
// The chase/cull (60 → 6 survivors) plays out in the opening ~50 ticks as
// the pack reaches the cluster and culls; the run then settles to the 6+3
// coexistence remnant. Sample densely early so the cull is captured
// frame-to-frame, then coarsely once the mixed equilibrium has settled.
const FRAME_EVERY: u32 = 5;
const IMG_SIZE: u32 = 256;

// Fear-tint normalizer: fear (shield_hp) at FEAR_TINT_MAX or above renders
// the survivor at the fully-terrified hot-white. With Perceive +1.0/sighting
// (per wolf in sight) and DecayFear *0.75/tick (Task 3 tuned 0.90 -> 0.75 so
// fear is more responsive and re-settles faster), a survivor with ONE wolf
// steadily in sight climbs toward steady-state 0.75/(1-0.75)=3; the cornered
// remnant sighting multiple wolves saturates higher (~12-30). 8.0 maps "a
// couple of wolves sustained in sight" to full white, so the gradient lives
// in the behaviourally-meaningful band: a survivor that just lost sight of a
// wolf decays through fear ~5 -> 1.5 — showing ORANGE, then fading to AMBER
// as fear drains under the 1.5 gate (the lingering-fear re-settle made
// visible) — while a survivor in an active clinch flares full WHITE. Values
// above 8 clamp to white (the cornered remnant), which honestly reads as
// "permanently wary", matching the bounded predator-prey clinch.
const FEAR_TINT_MAX: f32 = 8.0;

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
    // Per-frame fear diagnostics: max + mean fear over alive survivors
    // (from read_fear, the behaviour driver) plus max belief (read_threats),
    // so the trace SHOWS the fear staying live (not frozen) over 600 ticks
    // and lets the sign-off pick frames where fear spikes (wolf near) vs
    // settles (calm forage).
    let mut fear_max_trace: Vec<f32> = Vec::new();
    let mut fear_mean_trace: Vec<f32> = Vec::new();
    let mut threat_max_trace: Vec<f32> = Vec::new();
    let mut last_threats: Vec<f32> = vec![0.0; N_TOTAL];

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
        // FEAR column (shield_hp) — the TRUE behaviour driver (read_fear).
        // Read every rendered frame so the survivor tint tracks live fear.
        // The belief observable (read_threats) is read alongside purely as a
        // cross-check that the column has not frozen/degenerated over the
        // 600-tick run (the Task 2 single-pair freeze concern). In this
        // crowded world the column stays live; we print both per frame.
        let fear = read_fear(&mut state, N_TOTAL);
        // The belief observable is a cross-check only; read it sparsely (every
        // 30 ticks) to keep the extra staging readback from perturbing the GPU
        // trajectory more than the tint read already does. Between reads we
        // carry the last sample forward for the trace.
        if tick % 30 == 0 {
            last_threats = read_threats(&mut state, N_TOTAL);
        }
        let threats = &last_threats;

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
            // FEAR TINT: interpolate calm amber → terrified orange-white by
            // the survivor's live fear level (read_fear == shield_hp, the
            // value the Flee/SeekFood gates actually read). Scale fear by
            // FEAR_TINT_MAX and clamp to [0,1]; t=0 is calm amber
            // [240,170,40], t=1 is bright orange-white [255,215,150]. The
            // green+blue channels climb with fear (the dot pales/brightens)
            // while red stays pinned high — so a terrified survivor is a
            // PALE BRIGHT amber, clearly distinct from the pure-red wolves.
            let t = (fear[i] / FEAR_TINT_MAX).clamp(0.0, 1.0);
            let lerp = |a: f32, b: f32| (a + (b - a) * t) as u8;
            let color = [
                lerp(240.0, 255.0), // R: amber → white
                lerp(170.0, 255.0), // G: amber → white (drives amber→white)
                lerp(40.0, 235.0),  // B: low → near-white (the terror flash)
            ];
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
            let color = [255u8, 40u8, 40u8]; // pure red — distinct from any
                                             // fear-tinted (paler) survivor
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

        // Fear diagnostics over alive survivors.
        let mut fmax = 0.0f32;
        let mut fsum = 0.0f32;
        let mut tmax = 0.0f32;
        let mut nsurv = 0u32;
        for i in 0..N_TOTAL {
            if types[i] != CT_SURVIVOR || alive[i] == 0 {
                continue;
            }
            fmax = fmax.max(fear[i]);
            fsum += fear[i];
            tmax = tmax.max(threats[i]);
            nsurv += 1;
        }
        fear_max_trace.push(fmax);
        fear_mean_trace.push(if nsurv > 0 { fsum / nsurv as f32 } else { 0.0 });
        threat_max_trace.push(tmax);
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
    for ((((t, n), w), fmax), fmean) in frame_ticks
        .iter()
        .zip(pop_trace.iter())
        .zip(wolf_trace.iter())
        .zip(fear_max_trace.iter())
        .zip(fear_mean_trace.iter())
    {
        let bar = "#".repeat(*n as usize);
        let wbar = "x".repeat(*w as usize);
        println!(
            "  tick {t:4}  survivors {n:2} {bar}  | wolves {w} {wbar}  | fear max={fmax:5.2} mean={fmean:5.2}"
        );
    }
    println!();
    // FEAR LIVENESS / FREEZE report. The fear column is LIVE through the
    // opening hunt window (it climbs 0 -> ~30 as wolves close and the cull
    // plays out, then relaxes as the remnant settles) — this is the tint
    // signal driving the calm->spike->re-settle render. Once the remnant is
    // locked in a bounded clinch with the pack (~tick 60+), every survivor
    // sights wolves every tick and the f32 fear column saturates and FREEZES
    // at its steady value (the known single-pair freeze artifact, here
    // landing on the clinch steady state rather than degenerating) — so the
    // late-game tint is a stable "permanently wary" white, which is the
    // honest read of the clinch. We report the live (early) span and the
    // frozen (late) value separately. The tint is driven by read_fear (the
    // TRUE behaviour driver); read_threats is a host-readable cross-check
    // only and tracks it (both rise on sightings, both decay).
    let fmax_lo = fear_max_trace.iter().cloned().fold(f32::INFINITY, f32::min);
    let fmax_hi = fear_max_trace.iter().cloned().fold(0.0f32, f32::max);
    let fear_late = fear_max_trace.last().copied().unwrap_or(0.0);
    println!(
        "  fear(read_fear) max: live span lo={fmax_lo:.2} hi={fmax_hi:.2} (early hunt window); \
         frozen late-game steady value={fear_late:.2} (bounded-clinch saturation)"
    );
    println!(
        "  belief(read_threats) max range: lo={:.2} hi={:.2}  (cross-check — tracks the fear column)",
        threat_max_trace.iter().cloned().fold(f32::INFINITY, f32::min),
        threat_max_trace.iter().cloned().fold(0.0f32, f32::max),
    );
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
