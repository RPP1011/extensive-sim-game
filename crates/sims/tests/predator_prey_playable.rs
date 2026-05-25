//! Plan F — `predator_prey` as a playable game through the generic
//! `make_playable` path with ZERO new Rust. The Hare (slot 0, the lowest-slot
//! Hare, distinguished by the `mana < 0.5` band the `init { mana: slot }` seed
//! stamps) is driven by human input via the `@runtime config ctl` channel +
//! the `HareControl @phase(per_agent)` rule. This is the generality proof:
//! a structurally-different game (evade, entity-subkind identity, no weapons)
//! authored purely in `assets/sim/predator_prey.sim`.
//!
//! Requires a GPU adapter (the runtime is GPU-backed); run with
//! `RUST_MIN_STACK=33554432 cargo test -p sims --test predator_prey_playable`.
//! When no adapter is present `make_playable` yields `None`; the GPU-driving
//! tests skip rather than fail (headless CI without a device).

const SEED: u64 = 0x9_BEEF_F00D_0001;
const AGENTS: u32 = 256;

#[test]
fn predator_prey_is_registered() {
    assert!(
        sims::PLAYABLE_FIXTURES.contains(&"predator_prey"),
        "predator_prey must be in the registry: {:?}",
        sims::PLAYABLE_FIXTURES
    );
}

// Task 2 compile-gate (runs without a GPU too — but the only way to reach the
// descriptors is a constructed runtime, so skip gracefully without an adapter):
// all three player-facing descriptors emit + parse via their consumer crates.
#[test]
fn predator_prey_descriptors_emit_and_parse() {
    let Some(rt) = sims::make_playable("predator_prey", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping predator_prey descriptor checks");
        return;
    };
    // render: arena ring + follow-cam + Hare/Wolf creature_type color bands.
    let render = engine_play_api::RenderDescriptor::from_json(rt.render_descriptor())
        .expect("render_descriptor parses via engine_play_api");
    assert!(
        !render.agents.is_empty(),
        "predator_prey render descriptor should declare agent visuals (Hare/Wolf)"
    );
    // No weapon VFX — a deliberate difference from vampire_survivors (proves
    // the vfx surface is optional).
    assert!(
        render.vfx.is_empty(),
        "predator_prey is an evade game; it declares NO vfx (got {})",
        render.vfx.len()
    );
    // controls: WASD -> ctl.move_{x,y}.
    let controls = engine_play_api::ControlsDescriptor::from_json(rt.controls_descriptor())
        .expect("controls_descriptor parses via engine_play_api");
    assert_eq!(
        controls.bindings.len(),
        4,
        "predator_prey controls should bind WASD (4 keys): {:?}",
        controls.bindings
    );
    // ui: survive-timer text + a view readout + a death screen.
    let ui = engine_ui::UiModel::from_json(rt.ui_descriptor())
        .expect("ui_descriptor parses via engine_ui");
    assert!(
        !ui.hud.is_empty(),
        "predator_prey ui descriptor should declare hud widgets"
    );
    assert!(
        !ui.screens.is_empty(),
        "predator_prey ui descriptor should declare a death screen"
    );
}

// Task 1 runtime gate: the player Hare moves by INPUT, not by autonomous flee.
// Slot 0 is the player Hare (lowest slot, `mana == 0`, the only agent in the
// `mana < 0.5` band). Under `ctl.move_x = 1.0` it must travel +X; reversing the
// input must reverse its travel. This is input-driven movement, not fleeing.
#[test]
fn player_hare_tracks_input() {
    let Some(mut rt) = sims::make_playable("predator_prey", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping predator_prey input-drive test");
        return;
    };
    assert_eq!(rt.tick(), 0, "fresh runtime starts at tick 0");

    // Drive +X.
    rt.set_input("ctl.move_x", 1.0);
    rt.set_input("ctl.move_y", 0.0);
    // Unknown field is a silent no-op (the `_ => {}` arm).
    rt.set_input("ctl.does_not_exist", 1.0);

    let x0 = rt.agent_snapshot()[0].pos[0];
    for _ in 0..10 {
        rt.step();
    }
    let snap1 = rt.agent_snapshot();
    let x1 = snap1[0].pos[0];
    assert_eq!(rt.tick(), 10, "10 steps advance the tick to 10");
    // prey_speed = 0.5/tick × 10 ticks → ~+5.0 (clamped well inside arena 42).
    assert!(
        x1 > x0 + 1.0,
        "player Hare should move +X under move_x=1 (input-driven, not fleeing): {x0} -> {x1}"
    );
    // The player Hare must be alive (init seeds alive=1) — otherwise the move
    // guard would never fire and the assertion above would be vacuous.
    assert!(
        snap1[0].alive,
        "player Hare (slot 0) should be alive under the init seed"
    );

    // Reverse the input: the player Hare must reverse its travel.
    rt.set_input("ctl.move_x", -1.0);
    for _ in 0..10 {
        rt.step();
    }
    let x2 = rt.agent_snapshot()[0].pos[0];
    assert!(
        x2 < x1,
        "player Hare should reverse under move_x=-1: {x1} -> {x2}"
    );
    eprintln!("[predator_prey] PASS: player Hare tracks input ({x0} -> {x1} -> {x2})");
}
