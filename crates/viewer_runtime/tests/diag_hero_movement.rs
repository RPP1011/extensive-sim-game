//! Diagnostic test — drive the FULL viewer-side step loop (sim +
//! advance_hero_exploration + refresh_snapshot, in the same order
//! ViewerApp::step does) and report hero positions every 50 ticks
//! plus per-tick position deltas. Quantifies "are heroes actually
//! moving in the viewer pipeline" without needing to launch the
//! windowed binary.

use viewer_runtime::ViewerApp;

#[test]
fn diag_hero_movement_under_full_viewer_loop() {
    // Same seed the viewer_app binary uses by default.
    let seed = 0xD007BEEF57EA1u64;
    let mut app = match ViewerApp::try_new(seed) {
        Some(a) => a,
        None => {
            eprintln!("[diag] no wgpu adapter — skipping");
            return;
        }
    };

    let agents = app.agents();
    let n = agents.len();
    let hero_start = n.saturating_sub(5);
    println!("[diag] N={n} hero_start={hero_start}");
    println!("[diag] tick=0 hero positions:");
    for h in 0..5 {
        let a = &agents[hero_start + h];
        println!("  hero[{h}] pos=({:.2},{:.2},{:.2}) alive={}", a.pos.x, a.pos.y, a.pos.z, a.alive);
    }

    // Track each hero's total path length to detect "stuck" heroes.
    let mut last_pos: Vec<(f32, f32, f32)> = (0..5)
        .map(|h| {
            let a = &agents[hero_start + h];
            (a.pos.x, a.pos.y, a.pos.z)
        })
        .collect();
    let mut path_len: Vec<f32> = vec![0.0; 5];

    for t in 1..=400 {
        app.step();
        let agents = app.agents();
        for h in 0..5 {
            let a = &agents[hero_start + h];
            let dx = a.pos.x - last_pos[h].0;
            let dy = a.pos.y - last_pos[h].1;
            let dz = a.pos.z - last_pos[h].2;
            path_len[h] += (dx * dx + dy * dy + dz * dz).sqrt();
            last_pos[h] = (a.pos.x, a.pos.y, a.pos.z);
        }
        if t % 100 == 0 {
            println!("[diag] tick={t} positions + cumulative path length:");
            let agents = app.agents();
            for h in 0..5 {
                let a = &agents[hero_start + h];
                println!(
                    "  hero[{h}] pos=({:.2},{:.2},{:.2}) path={:.2} alive={}",
                    a.pos.x, a.pos.y, a.pos.z, path_len[h], a.alive,
                );
            }
        }
    }

    println!("[diag] final cumulative path lengths: {:?}", path_len);
    // Expectation: at hero_move_speed=0.40 and 400 ticks, the upper
    // bound on path length is 0.40*400=160 units. Heroes that actually
    // explore the dungeon land near 100-160 (warrior + rogue chase
    // frontier rooms); follower roles (cleric/ranger/mage) cluster
    // with the warrior so they typically score 60-80 (one or two
    // room transitions before warrior outpaces them). The 30 floor
    // catches the bug this test was authored against: heroes wedged
    // at slot boundaries because the `dist > 0.5` cutoff in
    // HeroExplore fired BEFORE crossing the door midpoint, freezing
    // them at y=15.8 with a max path of 8.8 units. Floored at 30 to
    // catch any future regression to "heroes don't cross even one
    // room boundary."
    let max_path = path_len.iter().cloned().fold(0.0_f32, f32::max);
    println!("[diag] max hero path length = {:.2}", max_path);
    assert!(
        max_path > 30.0,
        "no hero crossed the spawn room — max path {max_path:.2} < 30 units in 400 ticks. \
         Likely regression: HeroExplore's dist-cutoff or LoS gate is too aggressive \
         relative to host-written waypoint placement.",
    );
}
