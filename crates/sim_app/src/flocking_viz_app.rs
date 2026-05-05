//! Flocking-skirmish with ASCII visualization.
//!
//! Same renderer used by `predator_prey_viz_app`, just pointed at a
//! different sim. Demonstrates the generic viz interface: this binary
//! contains zero per-sim viz logic — glyphs, viewport bounds, and the
//! per-agent snapshot all come through the `CompiledSim` trait.
//!
//! Run with `cargo run --bin flocking_viz_app --features bin-flocking_viz_app --release`.

mod viz;

use engine::CompiledSim;
use flocking_skirmish_runtime::FlockingSkirmishState;
use std::io::Write;
use std::thread::sleep;
use std::time::Duration;
use viz::{render_sim_frame, Viewport};

const SEED: u64 = 0x4B01_D570_47A0_BEEF;
const RED_COUNT: u32 = 100;
const BLUE_COUNT: u32 = 100;
const MAX_TICKS: u64 = 300;
const FRAME_MS: u64 = 60;

fn main() {
    let mut sim = FlockingSkirmishState::new(SEED, RED_COUNT, BLUE_COUNT);
    let viewport = Viewport::for_sim(&sim, 80, 24);

    for tick in 0..=MAX_TICKS {
        if tick > 0 {
            sim.step();
        }

        // Sim-specific status — read counts from HP slice.
        let hp = sim.read_hp();
        let red_alive = hp[..RED_COUNT as usize].iter().filter(|h| **h > 0.0).count();
        let blue_alive = hp[RED_COUNT as usize..].iter().filter(|h| **h > 0.0).count();
        let red_hp: f32 = hp[..RED_COUNT as usize].iter().sum();
        let blue_hp: f32 = hp[RED_COUNT as usize..].iter().sum();

        let title = format!(
            "\x1b[1mFlocking Skirmish  —  tick {:4}  /  R = red  B = blue\x1b[0m",
            tick,
        );
        let status = vec![
            format!(
                " red:  alive \x1b[38;5;196m{:3}\x1b[0m hp \x1b[38;5;196m{:6.0}\x1b[0m   \
                 blue: alive \x1b[38;5;33m{:3}\x1b[0m hp \x1b[38;5;33m{:6.0}\x1b[0m",
                red_alive, red_hp, blue_alive, blue_hp,
            ),
            format!(" seed: 0x{:016X}", SEED),
        ];

        let frame = render_sim_frame(&mut sim, viewport, &title, &status);
        print!("{}", frame);
        std::io::stdout().flush().ok();
        sleep(Duration::from_millis(FRAME_MS));

        if red_alive == 0 || blue_alive == 0 {
            println!(
                "\nWipeout at tick {}: red={}, blue={}",
                tick, red_alive, blue_alive,
            );
            break;
        }
    }
}
