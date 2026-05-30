//! Plan D Task 2 Step 3 — the end-to-end gate. Construct the compiled
//! `play_probe` runtime via the real registry (`sims::make_playable`), drive
//! the generic player's separable per-frame [`engine_play::player::update`]
//! with a held `d` key, and assert the human input reached the runtime: the
//! followed player-band agent moves +X and the runtime recorded a positive
//! `ctl.move_x` write. No window, no panic.
//!
//! This is the GPU-backed half of the Plan D gate (the GPU-free descriptor
//! half lives in `dsl_compiler`'s `play_probe_descriptors_emit`). It needs a
//! GPU adapter — run with `RUST_MIN_STACK=33554432`. Without an adapter
//! `make_playable` returns `None` and the test skips its drive assertions
//! (CI / headless without a device) rather than failing.

use std::collections::HashSet;

use engine_play::bridge::{EngineBridge, Painted};
use engine_play::player::{mock_ui_model, update, HostState, PlayerConfig, UpdateOutput};
use engine_play_api::{ControlsDescriptor, RenderDescriptor};

const SEED: u64 = 0x5151_5151;
const AGENTS: u32 = 64;

#[test]
fn play_probe_constructs_via_registry() {
    // The registry must know `play_probe` regardless of GPU availability.
    assert!(
        sims::PLAYABLE_FIXTURES.contains(&"play_probe"),
        "play_probe must be in the registry: {:?}",
        sims::PLAYABLE_FIXTURES
    );
}

#[test]
fn play_probe_end_to_end() {
    let Some(mut rt) = sims::make_playable("play_probe", SEED, AGENTS) else {
        eprintln!("[play_probe] no GPU adapter; skipping end-to-end drive test");
        return;
    };

    // Descriptors come straight off the runtime — the same path the windowed
    // `Player::new` takes. They MUST parse via the frozen consumer contracts.
    let render = RenderDescriptor::from_json(rt.render_descriptor())
        .expect("play_probe render_descriptor parses via engine_play_api");
    let controls = ControlsDescriptor::from_json(rt.controls_descriptor())
        .expect("play_probe controls_descriptor parses via engine_play_api");
    let bridge = EngineBridge::new_headless(render);
    let mut host = HostState::new(PlayerConfig::default(), mock_ui_model());

    let ectx = egui::Context::default();
    let mut grid = Painted::new();
    let mut last_cells = Vec::new();

    // Hold `d` → the probe's controls bind `d -> ctl.move_x: 1.0`.
    let mut held = HashSet::new();
    held.insert("d".to_string());
    let pressed = HashSet::new();

    // The followed player-band agent's starting X (camera target == player).
    let start_tick = rt.tick();
    let start_x = bridge
        .followed(&rt.agent_snapshot())
        .map(|a| a.pos[0])
        .expect("a player-band agent exists at start (mana ~1.0 from init { mana: slot })");

    // Drive ~20 frames with `d` held.
    let mut last_out = UpdateOutput::default();
    const FRAMES: u32 = 20;
    for _ in 0..FRAMES {
        last_out = update(
            rt.as_mut(),
            &bridge,
            &mut host,
            &controls,
            &held,
            &pressed,
            &ectx,
            &mut grid,
            &mut last_cells,
        );
    }

    // The sim advanced once per frame (no modal froze it).
    assert_eq!(
        rt.tick(),
        start_tick + FRAMES as u64,
        "sim stepped each frame (no modal freeze)"
    );

    // `update` reported the held-`d` write as a positive ctl.move_x input.
    assert!(
        last_out
            .inputs
            .iter()
            .any(|(f, v)| f == "ctl.move_x" && *v > 0.0),
        "held d should produce a positive ctl.move_x write, got inputs {:?}",
        last_out.inputs
    );

    // The input reached the GPU runtime: the followed player moved +X.
    let end_x = bridge
        .followed(&rt.agent_snapshot())
        .map(|a| a.pos[0])
        .expect("player-band agent still followed after the run");
    eprintln!("[play_probe] player x: {start_x} -> {end_x} (held d, {FRAMES} frames)");
    assert!(
        end_x > start_x + 0.5,
        "held d for {FRAMES} frames should drive the player +X (player_speed 1.0/tick); \
         got {start_x} -> {end_x}"
    );

    // The followed-agent paint touched cells (the player splat rendered).
    assert!(!grid.cells.is_empty(), "bridge painted agent cells");
}
