//! `viewer_app` — Phase A console driver for [`viewer_runtime::ViewerApp`].
//!
//! Drives the App lifecycle against a headless scene with no window
//! and no renderer. Prints a periodic snapshot of the scene's
//! populated entity count + sim tick + score so you can confirm the
//! data flow without standing up Vulkan. The windowed driver is a
//! follow-up phase.
//!
//! ```text
//! viewer_app [seed] [max_ticks]
//! ```
//!
//! Defaults: seed=0, max_ticks=2000.

use viewer_runtime::ViewerApp;
use voxel_engine::app::App;
use voxel_engine::scene::config::SceneConfig;
use voxel_engine::scene::Scene;
use wave_defense_runtime::DEFAULT_MAX_TICKS;

const TICK_LOG_PERIOD: u64 = 50;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let max_ticks: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_TICKS);

    eprintln!(
        "[viewer_app] seed={} max_ticks={} (Phase A: headless, no window)",
        seed, max_ticks,
    );

    let mut app = ViewerApp::new(seed);
    let mut scene = Scene::new_headless(SceneConfig::default());

    if let Err(e) = app.setup(&mut scene) {
        eprintln!("[viewer_app] setup failed: {e}");
        return;
    }
    eprintln!(
        "[viewer_app] setup complete — populated={} entities at tick=0",
        app.populated_entity_count(),
    );

    for _ in 0..max_ticks {
        app.tick(&mut scene, 0.1);
        let t = app.sim_tick();
        if t % TICK_LOG_PERIOD == 0 {
            eprintln!(
                "[viewer_app] tick={} populated={}",
                t,
                app.populated_entity_count(),
            );
        }
        if app.terminated_at_tick.is_some() {
            eprintln!(
                "[viewer_app] sim terminated at tick={}",
                app.terminated_at_tick.unwrap(),
            );
            break;
        }
    }

    eprintln!(
        "[viewer_app] done — final tick={} populated={} terminated_at={:?}",
        app.sim_tick(),
        app.populated_entity_count(),
        app.terminated_at_tick,
    );
}
