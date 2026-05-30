//! Headless top-down GIF capture of the vampire_survivors sim.
//!
//! Drives `VsViewerApp` (GPU compute only — no window/display needed) and
//! plots each tick's live agents as colored dots on a flat arena, encoding
//! the frames into an animated GIF. This is a poor-man's viewer that works
//! without a display: it renders the actual sim STATE (agent positions +
//! mana-band roles), which is exactly what's needed to review the gameplay
//! and diagnose behavior.
//!
//! Run: `cargo run -p viewer_runtime --example vs_capture --release [SEED] [FRAMES]`
//! Output: /tmp/vs_capture.gif + a per-frame role-count trace on stderr.
//!
//! Colors: player = cyan, enemy = orange-red, spawner = purple
//! (matches VsBridge's MAT_VS_* palette).

use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, Frame, Rgba, RgbaImage};
use viewer_runtime::vs::{VsAgent, VsRole, VsViewerApp};

const GRID: f32 = 96.0; // matches GRID_X/GRID_Y; agents live in [-48,48] world coords
const SCALE: u32 = 4; // px per grid unit -> 384x384 frames
const DEFAULT_FRAMES: usize = 120;

fn world_to_px(p: [f32; 3], dim: u32) -> Option<(i32, i32)> {
    // +GRID/2 centers origin in the grid (the VsBridge +48 offset), then scale.
    let gx = p[0] + GRID / 2.0;
    let gy = p[1] + GRID / 2.0;
    if gx < 0.0 || gy < 0.0 || gx >= GRID || gy >= GRID {
        return None;
    }
    let px = (gx * SCALE as f32) as i32;
    let py = (gy * SCALE as f32) as i32;
    let _ = dim;
    Some((px, py))
}

fn color(a: &VsAgent) -> Rgba<u8> {
    match a.role {
        VsRole::Player => Rgba([0, 220, 220, 255]),    // cyan
        VsRole::Spawner => Rgba([160, 60, 220, 255]),  // purple (unused now)
        // Enemy color by type (move_speed): Swift=yellow, Brute=red, Grunt=orange.
        VsRole::Enemy if a.move_speed > 0.6 => Rgba([240, 230, 40, 255]), // Swift
        VsRole::Enemy if a.move_speed < 0.3 => Rgba([220, 40, 40, 255]),  // Brute
        VsRole::Enemy => Rgba([230, 110, 20, 255]),                       // Grunt
    }
}

fn dot_radius(a: &VsAgent) -> i32 {
    match a.role {
        VsRole::Player => 4,
        VsRole::Spawner => 3,
        VsRole::Enemy if a.move_speed < 0.3 => 3, // Brute bigger
        VsRole::Enemy => 2,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let seed = args
        .next()
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x5_F00D_CAFE_0001);
    let frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_FRAMES);

    let mut app = match VsViewerApp::try_new(seed) {
        Some(a) => a,
        None => {
            eprintln!("[vs_capture] no wgpu adapter — cannot run");
            std::process::exit(1);
        }
    };

    let dim = (GRID as u32) * SCALE;
    let path = "/tmp/vs_capture.gif";
    let file = std::fs::File::create(path).expect("create gif");
    let mut enc = GifEncoder::new_with_speed(file, 20);
    enc.set_repeat(Repeat::Infinite).ok();

    let bg = Rgba([28, 28, 34, 255]);
    let floor = Rgba([44, 44, 52, 255]);

    for f in 0..frames {
        let mut img = RgbaImage::from_pixel(dim, dim, bg);
        // faint floor inset so the arena bounds are visible
        for y in (SCALE)..(dim - SCALE) {
            for x in (SCALE)..(dim - SCALE) {
                img.put_pixel(x, y, floor);
            }
        }

        let (mut np, mut ne, mut ns) = (0u32, 0u32, 0u32);
        for a in app.agents() {
            match a.role {
                VsRole::Player => np += 1,
                VsRole::Enemy => ne += 1,
                VsRole::Spawner => ns += 1,
            }
            if let Some((cx, cy)) = world_to_px(a.pos, dim) {
                let r = dot_radius(a);
                let col = color(a);
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (px, py) = (cx + dx, cy + dy);
                        if px >= 0 && py >= 0 && (px as u32) < dim && (py as u32) < dim {
                            img.put_pixel(px as u32, py as u32, col);
                        }
                    }
                }
            }
        }

        if f % 10 == 0 {
            eprintln!("[vs_capture] frame {f} tick {} : player={np} enemy={ne} spawner={ns}", app.sim_tick());
        }

        // Dump a few mid-swarm stills so the gameplay is viewable as PNGs
        // (a GIF renders only its first frame in most viewers).
        if matches!(f, 12 | 50 | 90) {
            img.save(format!("/tmp/vs_frame_{f:03}.png")).expect("save png still");
        }

        enc.encode_frame(Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(60, 1)))
            .expect("encode frame");

        app.step();
    }

    eprintln!(
        "[vs_capture] wrote {path}  ({frames} frames, seed {seed:#x}, final tick {}, terminated_at={:?})",
        app.sim_tick(),
        app.terminated_at_tick
    );
}
