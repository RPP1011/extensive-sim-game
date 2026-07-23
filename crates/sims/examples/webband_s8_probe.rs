//! S8-prep proof runner (Webband port). Drives ANY allowlisted fixture
//! through the SAME seam the generic `play` binary uses —
//! `sims::make_playable` → `PlayableRuntime` (step / agent_snapshot /
//! render_descriptor) — but presents headlessly: ANSI-color terminal
//! frames plus PNG snapshots, because the windowed path (engine_play →
//! voxel_engine Vulkan renderer) cannot compile against the ~120-line
//! voxel_engine STUB on this machine (see docs/superpowers/plans/
//! webband-port.md, S8-prep report).
//!
//! Colors come from the fixture's own `render {}` descriptor
//! (engine_play_api::RenderDescriptor), so this also proves the
//! descriptor's runtime consumption path end to end.
//!
//! Usage:
//!   cargo run -p sims --example webband_s8_probe [fixture] [seed] [agents] [frames]
//! Defaults: predator_prey, seed 0x9BEEF00D0001, 256 agents, 120 frames.
//! PNGs land in target/webband_s8_probe/.

use engine_play_api::{AgentView, RenderDescriptor};

const VIEW_W: usize = 100;
const VIEW_H: usize = 36;
const FRAME_MS: u64 = 80;

fn color_of(desc: &RenderDescriptor, a: &AgentView) -> [u8; 3] {
    for v in &desc.agents {
        let val = match v.when.field.as_str() {
            "hp" => a.hp,
            "mana" => a.mana,
            "move_speed" => a.move_speed,
            "creature_type" => a.creature_type as f32,
            _ => 0.0,
        };
        if val >= v.when.lo && val < v.when.hi {
            return v.color;
        }
    }
    [160, 160, 160]
}

fn main() {
    // Generated runtimes hold large stack values at construction (the S1
    // report's 2MiB-overflow finding — among_us needs a 64MiB stack). A
    // binary's main thread ignores RUST_MIN_STACK, so run on our own thread.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .expect("spawn probe thread")
        .join()
        .expect("probe thread panicked");
}

fn run() {
    let args: Vec<String> = std::env::args().collect();
    let fixture = args.get(1).cloned().unwrap_or_else(|| "predator_prey".into());
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x9_BEEF_F00D_0001);
    let agents: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let frames: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(120);

    let Some(mut rt) = sims::make_playable(&fixture, seed, agents) else {
        eprintln!(
            "make_playable({fixture:?}) failed (unknown fixture or no GPU adapter).\n\
             known: {:?}",
            sims::PLAYABLE_FIXTURES
        );
        std::process::exit(2);
    };
    let desc = RenderDescriptor::from_json(rt.render_descriptor())
        .expect("render descriptor parses");
    let r = if desc.arena_radius > 1.0 { desc.arena_radius } else { 48.0 };
    eprintln!(
        "[webband_s8_probe] {fixture} seed={seed:#x} agents={agents} \
         arena_radius={r} visuals={} camera={:?}",
        desc.agents.len(),
        desc.camera
    );

    let png_dir = std::path::PathBuf::from("target/webband_s8_probe");
    std::fs::create_dir_all(&png_dir).ok();

    // Movement proof: remember the first snapshot's positions.
    let first: Vec<AgentView> = rt.agent_snapshot();
    let alive0 = first.iter().filter(|a| a.alive).count();

    for f in 0..frames {
        // Hold a "d"-style input: fixtures with an @runtime ctl channel move
        // their player agent; fixtures without one hit the silent no-op arm.
        rt.set_input("ctl.move_x", 1.0);
        rt.step();
        let snap = rt.agent_snapshot();

        // ANSI frame: map world [-r, r]^2 (sim XY plane) onto the grid.
        let mut grid = vec![[0u8; 3]; VIEW_W * VIEW_H];
        let mut hit = vec![false; VIEW_W * VIEW_H];
        for a in snap.iter().filter(|a| a.alive) {
            let gx = ((a.pos[0] + r) / (2.0 * r) * VIEW_W as f32) as i32;
            let gy = ((a.pos[1] + r) / (2.0 * r) * VIEW_H as f32) as i32;
            if gx >= 0 && gy >= 0 && (gx as usize) < VIEW_W && (gy as usize) < VIEW_H {
                let i = gy as usize * VIEW_W + gx as usize;
                grid[i] = color_of(&desc, a);
                hit[i] = true;
            }
        }
        let mut out = String::with_capacity(VIEW_W * VIEW_H * 12);
        out.push_str("\x1b[H\x1b[2J");
        for y in 0..VIEW_H {
            for x in 0..VIEW_W {
                let i = y * VIEW_W + x;
                if hit[i] {
                    let [cr, cg, cb] = grid[i];
                    out.push_str(&format!("\x1b[38;2;{cr};{cg};{cb}m@\x1b[0m"));
                } else {
                    out.push('.');
                }
            }
            out.push('\n');
        }
        let alive = snap.iter().filter(|a| a.alive).count();
        out.push_str(&format!(
            "tick {}  alive {}/{}  fixture {}  (frame {}/{})\n",
            rt.tick(),
            alive,
            snap.len(),
            fixture,
            f + 1,
            frames
        ));
        print!("{out}");

        // PNG snapshots at start / middle / end.
        if f == 0 || f == frames / 2 || f + 1 == frames {
            let size = 512u32;
            let mut img = image::RgbImage::from_pixel(size, size, image::Rgb([24, 24, 28]));
            for a in snap.iter().filter(|a| a.alive) {
                let px = ((a.pos[0] + r) / (2.0 * r) * size as f32) as i32;
                let py = ((a.pos[1] + r) / (2.0 * r) * size as f32) as i32;
                let c = image::Rgb(color_of(&desc, a));
                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        let (x, y) = (px + dx, py + dy);
                        if x >= 0 && y >= 0 && (x as u32) < size && (y as u32) < size {
                            img.put_pixel(x as u32, y as u32, c);
                        }
                    }
                }
            }
            let p = png_dir.join(format!("{fixture}_tick{:04}.png", rt.tick()));
            img.save(&p).expect("write png");
            eprintln!("[webband_s8_probe] wrote {}", p.display());
        }

        std::thread::sleep(std::time::Duration::from_millis(FRAME_MS));
    }

    // Numeric movement proof: mean displacement of surviving agents.
    let last = rt.agent_snapshot();
    let mut moved = 0usize;
    let mut total = 0usize;
    let mut mean_d = 0.0f32;
    for (a, b) in first.iter().zip(last.iter()) {
        if a.alive && b.alive {
            let d = ((b.pos[0] - a.pos[0]).powi(2) + (b.pos[1] - a.pos[1]).powi(2)).sqrt();
            mean_d += d;
            total += 1;
            if d > 0.5 {
                moved += 1;
            }
        }
    }
    if total > 0 {
        mean_d /= total as f32;
    }
    println!(
        "[webband_s8_probe] DONE tick={} alive {}->{}  moved(>0.5u) {}/{}  mean_disp {:.2}u",
        rt.tick(),
        alive0,
        last.iter().filter(|a| a.alive).count(),
        moved,
        total,
        mean_d
    );
}
