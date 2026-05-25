use viewer_runtime::vs::{VsViewerApp, VsRole};
const SEED: u64 = 0x5_F00D_CAFE_0001;

#[test]
fn vs_viewer_constructs_steps_and_spawns() {
    let mut app = match VsViewerApp::try_new(SEED) {
        Some(a) => a,
        None => { eprintln!("[vs_viewer] skip: no wgpu adapter"); return; }
    };
    let players = app.agents().iter().filter(|a| a.role == VsRole::Player).count();
    let spawners = app.agents().iter().filter(|a| a.role == VsRole::Spawner).count();
    let enemies0 = app.agents().iter().filter(|a| a.role == VsRole::Enemy).count();
    assert_eq!(players, 1, "exactly one player; got {players}");
    assert!(spawners >= 1, "at least one spawner; got {spawners}");
    assert_eq!(enemies0, 0, "no live enemies before any wave; got {enemies0}");

    let mut max_enemies = 0;
    for _ in 0..60 { app.step(); max_enemies = max_enemies.max(app.agents().iter().filter(|a| a.role == VsRole::Enemy).count()); }
    assert_eq!(app.sim_tick(), 60, "stepped 60 ticks; got {}", app.sim_tick());
    assert!(max_enemies > 0, "DSL waves should spawn live enemies through the viewer app; got {max_enemies}");
    assert!(app.agents().iter().all(|a| a.pos.iter().all(|c| c.is_finite())), "no NaN/inf positions");
}
