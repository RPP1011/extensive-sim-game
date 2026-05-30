//! Plan D Task 1 — the `play_probe.sim` end-to-end fixture must emit all
//! three non-empty player-facing descriptors that parse back through the
//! frozen consumer-crate `from_json`s (`engine_play_api` for render/controls,
//! `engine_ui` for ui). This is the GPU-free half of the Plan D gate: it
//! pins the *static* descriptor contract straight from the real `.sim`
//! source, independent of whether a GPU adapter is present (the GPU drive
//! half lives in `engine_play`'s `play_probe_end_to_end` test).

use dsl_compiler::cg::emit::{controls, render, ui_model};
use engine_play_api::{CameraSpec, ControlsDescriptor, RenderDescriptor};
use engine_ui::{UiModel, Widget};

/// The actual fixture the registry compiles + `make_playable` constructs.
const PLAY_PROBE_SRC: &str = include_str!("../../../assets/sim/play_probe.sim");

fn program() -> dsl_ast::ast::Program {
    dsl_ast::parse(PLAY_PROBE_SRC).expect("play_probe.sim parses")
}

#[test]
fn play_probe_declares_all_three_blocks() {
    let p = program();
    assert!(p.render.is_some(), "play_probe must declare a render block");
    assert!(p.controls.is_some(), "play_probe must declare a controls block");
    assert!(p.ui.is_some(), "play_probe must declare a ui block");
}

#[test]
fn play_probe_render_parses_and_is_non_empty() {
    let p = program();
    let json = render::render_decl_to_json(
        p.render.as_ref().unwrap(),
        &std::collections::BTreeMap::new(),
    );
    let d = RenderDescriptor::from_json(&json)
        .unwrap_or_else(|e| panic!("render from_json failed ({e}) for:\n{json}"));
    assert_eq!(d.arena_radius, 24.0);
    // Camera follows the player mana band [0.5, 1.5].
    match d.camera {
        CameraSpec::Follow(r) => {
            assert_eq!(r.field, "mana");
            assert_eq!(r.lo, 0.5);
            assert_eq!(r.hi, 1.5);
        }
        _ => panic!("expected Follow camera"),
    }
    // Two agent visuals: player (cyan) + static enemy band (orange).
    assert_eq!(d.agents.len(), 2, "player + static color bands");
    assert_eq!(d.agents[0].color, [0, 220, 220]);
    assert_eq!(d.agents[1].color, [220, 80, 40]);
}

#[test]
fn play_probe_controls_parses_and_is_non_empty() {
    let p = program();
    let json = controls::controls_decl_to_json(p.controls.as_ref().unwrap());
    let d = ControlsDescriptor::from_json(&json)
        .unwrap_or_else(|e| panic!("controls from_json failed ({e}) for:\n{json}"));
    // WASD → the two ctl.move_* fields.
    assert_eq!(d.bindings.len(), 4, "WASD bindings");
    // `d` must write a positive ctl.move_x (the property the end-to-end test
    // leans on: held `d` drives the player +X).
    let d_bind = d
        .bindings
        .iter()
        .find(|b| b.key == "d")
        .expect("a `d` binding exists");
    assert_eq!(d_bind.field, "ctl.move_x");
    assert!(d_bind.value > 0.0, "held d → +ctl.move_x");
}

#[test]
fn play_probe_ui_parses_and_is_non_empty() {
    let p = program();
    let json = ui_model::ui_decl_to_json(p.ui.as_ref().unwrap());
    let m = UiModel::from_json(&json)
        .unwrap_or_else(|e| panic!("ui from_json failed ({e}) for:\n{json}"));
    assert!(!m.hud.is_empty(), "ui declares at least one HUD widget");
    // First widget is the HP bar.
    match &m.hud[0] {
        Widget::Bar { label, value, max, .. } => {
            assert_eq!(label, "HP");
            assert_eq!(value, "hp");
            assert_eq!(max, "hp_max");
        }
        _ => panic!("expected HP Bar first"),
    }
}
