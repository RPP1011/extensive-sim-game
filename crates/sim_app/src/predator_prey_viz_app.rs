//! Predator-prey real with ASCII visualization.
//!
//! Demonstrates the generic `viz::render_sim_frame` driving any sim
//! that implements the `CompiledSim` viz methods (`snapshot`,
//! `glyph_table`, `default_viewport`). The app harness only knows
//! about the runtime's CONSTRUCTOR and the sim-specific stat readouts
//! (kills, births) — the rest comes through the trait.
//!
//! Run with `cargo run --bin predator_prey_viz_app --features bin-predator_prey_viz_app --release`.

mod viz;

use engine::CompiledSim;
use predator_prey_real_runtime::{PredatorPreyRealState, SLOT_CAP};
use std::io::Write;
use std::thread::sleep;
use std::time::Duration;
use viz::{render_sim_frame, Viewport};

const SEED: u64 = 0x707E_DA70_47A0_BEEF;
const MAX_TICKS: u64 = 400;
const FRAME_MS: u64 = 80;

fn main() {
    let mut sim = PredatorPreyRealState::new(SEED, SLOT_CAP);
    let viewport = Viewport::for_sim(&sim, 80, 24);

    for tick in 0..=MAX_TICKS {
        if tick > 0 {
            sim.step();
        }

        let alive_wolves = sim.count_alive_wolves();
        let alive_sheep = sim.count_alive_sheep();
        let title = format!(
            "\x1b[1mPredator-Prey  —  tick {:4}  /  W = wolf  s = sheep\x1b[0m",
            tick,
        );
        let status = vec![
            format!(
                " wolves: \x1b[38;5;196m{:3}\x1b[0m   sheep: \x1b[38;5;47m{:3}\x1b[0m   \
                 kills: {}   wolf_births: {}   sheep_births: {}",
                alive_wolves,
                alive_sheep,
                sim.sheep_kills_so_far(),
                sim.wolf_births_so_far(),
                sim.sheep_births_so_far(),
            ),
            format!(
                " starv: wolves {} sheep {}   total slots: {}/{}   seed: 0x{:016X}",
                sim.wolf_starvations_so_far(),
                sim.sheep_starvations_so_far(),
                alive_wolves + alive_sheep,
                SLOT_CAP,
                SEED,
            ),
        ];

        let frame = render_sim_frame(&mut sim, viewport, &title, &status);
        print!("{}", frame);
        std::io::stdout().flush().ok();
        sleep(Duration::from_millis(FRAME_MS));

        if alive_wolves == 0 && alive_sheep == 0 {
            println!("\nBOTH EXTINCT at tick {}.", tick);
            break;
        }
    }

    println!(
        "\nFinal: wolves={}, sheep={}, kills={}, wolf_births={}, sheep_births={}",
        sim.count_alive_wolves(),
        sim.count_alive_sheep(),
        sim.sheep_kills_so_far(),
        sim.wolf_births_so_far(),
        sim.sheep_births_so_far(),
    );
}
