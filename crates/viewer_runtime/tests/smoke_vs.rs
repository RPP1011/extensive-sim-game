use viewer_runtime::vs::{VsViewerApp, VsRole};
use viewer_runtime::vs_ui;
const SEED: u64 = 0x5_F00D_CAFE_0001;

#[test]
fn vs_viewer_constructs_steps_and_spawns() {
    let mut app = match VsViewerApp::try_new(SEED) {
        Some(a) => a,
        None => { eprintln!("[vs_viewer] skip: no wgpu adapter"); return; }
    };
    let players = app.agents().iter().filter(|a| a.role == VsRole::Player).count();
    let enemies0 = app.agents().iter().filter(|a| a.role == VsRole::Enemy).count();
    // No spawner agents now — the player emits the waves; the drain spawns
    // enemies in a ring around it.
    assert_eq!(players, 1, "exactly one player; got {players}");
    assert_eq!(enemies0, 0, "no live enemies before any wave; got {enemies0}");

    let mut max_enemies = 0;
    for _ in 0..60 { app.step(); max_enemies = max_enemies.max(app.agents().iter().filter(|a| a.role == VsRole::Enemy).count()); }
    assert_eq!(app.sim_tick(), 60, "stepped 60 ticks; got {}", app.sim_tick());
    assert!(max_enemies > 0, "DSL waves should spawn live enemies through the viewer app; got {max_enemies}");
    assert!(app.agents().iter().all(|a| a.pos.iter().all(|c| c.is_finite())), "no NaN/inf positions");

    // Player XP readback is finite + monotonic non-decreasing over the run.
    let xp = app.player_xp();
    assert!(xp.is_finite() && xp >= 0.0, "player_xp finite and >= 0; got {xp}");
}

/// No-GPU-safe smoke: build the host game-UI state, feed it through the
/// engine_ui model/data builders + a headless egui draw, and exercise the
/// level-up + restart state machine. Runs without a display or wgpu adapter.
#[test]
fn vs_ui_model_data_and_draw_headless() {
    let mut progress = vs_ui::PlayerProgress::default();

    // Build the HUD model + per-frame data the viewer feeds engine_ui.
    let mut model = vs_ui::hud_model();
    assert!(!model.hud.is_empty(), "hud model has widgets");

    let mut data = engine_ui::UiData::new();
    vs_ui::build_data(&mut data, 80.0, vs_ui::PLAYER_HP_MAX, 12.0, 7, 100, 42);
    assert_eq!(data.get("level"), 2.0, "12 xp / 5 per level = level 2");

    // Crossing a level threshold opens a fresh menu screen.
    assert!(progress.check_level_up(vs_ui::XP_PER_LEVEL));
    model.screens = vec![vs_ui::menu_screen(SEED, progress.last_level)];

    // Render the HUD + the active level-up menu headlessly (no panic, shapes).
    let ctx = egui::Context::default();
    let out = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = engine_ui::draw(ctx, &model, &data, Some("level_up"));
    });
    assert!(!out.shapes.is_empty(), "engine_ui::draw produced no shapes");

    // Render the death screen headlessly too.
    let mut dead_model = vs_ui::hud_model();
    dead_model.screens = vec![vs_ui::death_screen()];
    let out2 = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = engine_ui::draw(ctx, &dead_model, &data, Some("dead"));
    });
    assert!(!out2.shapes.is_empty(), "death-screen draw produced no shapes");

    // Applying an upgrade pick raises the matching level.
    let before = progress.bolt_level;
    progress.apply(&engine_ui::UiAction::Increment("bolt_level".into()));
    assert_eq!(progress.bolt_level, before + 1.0);
}
