//! Plan A Task 1 — the `make_playable` registry + the generated
//! `PlayableRuntime` impl. A compiled `.sim` is constructable by name and
//! driveable through the frozen `engine_play_api::PlayableRuntime` seam:
//! tick advances on `step`, `set_input` writes a `@runtime` field, and
//! `agent_snapshot` reads back the live agents.
//!
//! Requires a GPU adapter (the runtime is GPU-backed); run with
//! `RUST_MIN_STACK=33554432 cargo test -p sims --test playable_registry`.
//! When no adapter is present `try_new` returns `None` and `make_playable`
//! yields `None`; the test skips its GPU assertions in that case rather than
//! failing (CI / headless without a device).

const SEED: u64 = 0x5151_5151;
const AGENTS: u32 = 512;

#[test]
fn unknown_fixture_is_none() {
    assert!(sims::make_playable("definitely_not_a_fixture", SEED, AGENTS).is_none());
}

#[test]
fn vampire_survivors_is_registered() {
    assert!(
        sims::PLAYABLE_FIXTURES.contains(&"vampire_survivors"),
        "vampire_survivors must be in the registry: {:?}",
        sims::PLAYABLE_FIXTURES
    );
}

#[test]
fn vampire_survivors_descriptors_match_consumer_crates() {
    // Descriptors are static and GPU-independent, but the only way to reach
    // them is a constructed runtime. Skip gracefully without a GPU.
    let Some(rt) = sims::make_playable("vampire_survivors", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping vampire_survivors descriptor checks");
        return;
    };
    let render = engine_play_api::RenderDescriptor::from_json(rt.render_descriptor())
        .expect("render_descriptor parses via engine_play_api");
    // vampire_survivors declares a render block (player + enemy visuals).
    assert!(
        !render.agents.is_empty(),
        "vampire_survivors render descriptor should declare agent visuals"
    );
    let controls = engine_play_api::ControlsDescriptor::from_json(rt.controls_descriptor())
        .expect("controls_descriptor parses via engine_play_api");
    assert!(
        !controls.bindings.is_empty(),
        "vampire_survivors controls descriptor should declare key bindings"
    );
    let ui = engine_ui::UiModel::from_json(rt.ui_descriptor())
        .expect("ui_descriptor parses via engine_ui");
    assert!(
        !ui.hud.is_empty(),
        "vampire_survivors ui descriptor should declare hud widgets"
    );
}

#[test]
fn vampire_survivors_steps_and_takes_input() {
    let Some(mut rt) = sims::make_playable("vampire_survivors", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping vampire_survivors GPU drive test");
        return;
    };
    assert_eq!(rt.tick(), 0, "fresh runtime starts at tick 0");
    rt.step();
    assert_eq!(rt.tick(), 1, "step advances the tick");

    // `ctl.move_x` is a declared `@runtime` field in vampire_survivors.sim;
    // set_input must route to its generated setter without panicking.
    rt.set_input("ctl.move_x", 1.0);
    // Unknown field is a silent no-op (the `_ => {}` arm).
    rt.set_input("ctl.does_not_exist", 1.0);

    rt.step();
    assert_eq!(rt.tick(), 2);

    let snapshot = rt.agent_snapshot();
    assert!(
        !snapshot.is_empty(),
        "agent_snapshot should return the live agents (got {})",
        snapshot.len()
    );
}
