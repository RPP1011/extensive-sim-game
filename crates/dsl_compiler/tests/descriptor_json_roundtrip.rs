//! Plan A — the descriptor JSON the compiler hand-emits MUST parse back
//! through the frozen consumer-crate serde contracts. This test parses a
//! `.sim` with all three player-facing blocks, runs the emitters, and feeds
//! each JSON string to the matching `from_json`. If serde's externally-tagged
//! shape ever drifts from the hand-emitted JSON, this fails (fix the emitter,
//! not the contract).

use dsl_compiler::cg::emit::{controls, render, ui_model};
use engine_play_api::{CameraSpec, ControlsDescriptor, RenderDescriptor, VfxKind};
use engine_ui::{Screen, UiModel, Widget};

const SRC: &str = "\
controls {
  key \"w\" -> ctl.move_y: 1.0
  key \"s\" -> ctl.move_y: -1.0
  key \"a\" -> ctl.move_x: -1.0
  key \"space\" -> ctl.bolt_rate_level: 1.0 press
}
render {
  arena_radius 120.0
  camera follow when mana in [0.5, 1.5]
  agent when mana in [0.5, 1.5] { color (0, 220, 220) }
  agent when mana in [1.5, 9.9] { color (220, 80, 40) }
  vfx on NovaFire period 40 { ring radius 6.0 color (255, 255, 120) }
  vfx on Bolt period 12 { beam_to_nearest when mana in [1.5, 9.9] color (120, 200, 255) }
}
ui {
  hud {
    bar \"HP\" value hp max hp_max color (220, 40, 40)
    bar \"XP\" value xp max xp_per_level color (40, 160, 240)
    text \"Lv {level}  Kills {kills}\"
  }
  menu level_up \"Level Up!\" {
    card \"Bolt Damage +\" -> bolt_level
    card \"Nova +\" -> nova_level
  }
  screen dead \"You Died\" { summary time level kills  restart \"Restart (R)\" }
}
";

fn program() -> dsl_ast::ast::Program {
    dsl_ast::parse(SRC).expect("parse probe sim")
}

#[test]
fn controls_json_parses_via_engine_play_api() {
    let p = program();
    let json = controls::controls_decl_to_json(p.controls.as_ref().unwrap());
    let d = ControlsDescriptor::from_json(&json)
        .unwrap_or_else(|e| panic!("controls from_json failed ({e}) for:\n{json}"));
    assert_eq!(d.bindings.len(), 4);
    assert_eq!(d.bindings[0].key, "w");
    assert_eq!(d.bindings[0].field, "ctl.move_y");
    assert_eq!(d.bindings[0].value, 1.0);
    assert!(matches!(d.bindings[0].mode, engine_play_api::BindMode::Hold));
    assert!(matches!(d.bindings[3].mode, engine_play_api::BindMode::Press));
    assert_eq!(d.bindings[1].value, -1.0);
}

#[test]
fn render_json_parses_via_engine_play_api() {
    let p = program();
    let json = render::render_decl_to_json(p.render.as_ref().unwrap());
    let d = RenderDescriptor::from_json(&json)
        .unwrap_or_else(|e| panic!("render from_json failed ({e}) for:\n{json}"));
    assert_eq!(d.arena_radius, 120.0);
    match d.camera {
        CameraSpec::Follow(r) => {
            assert_eq!(r.field, "mana");
            assert_eq!(r.lo, 0.5);
            assert_eq!(r.hi, 1.5);
        }
        _ => panic!("expected Follow camera"),
    }
    assert_eq!(d.agents.len(), 2);
    assert_eq!(d.agents[0].color, [0, 220, 220]);
    assert_eq!(d.vfx.len(), 2);
    assert_eq!(d.vfx[0].on_rule, "NovaFire");
    assert_eq!(d.vfx[0].period, 40);
    assert!(matches!(d.vfx[0].kind, VfxKind::Ring));
    assert_eq!(d.vfx[0].radius, 6.0);
    match &d.vfx[1].kind {
        VfxKind::BeamToNearest { target } => assert_eq!(target.field, "mana"),
        _ => panic!("expected BeamToNearest vfx"),
    }
}

#[test]
fn ui_json_parses_via_engine_ui() {
    let p = program();
    let json = ui_model::ui_decl_to_json(p.ui.as_ref().unwrap());
    let m = UiModel::from_json(&json)
        .unwrap_or_else(|e| panic!("ui from_json failed ({e}) for:\n{json}"));
    assert_eq!(m.hud.len(), 3);
    match &m.hud[0] {
        Widget::Bar { label, value, max, color } => {
            assert_eq!(label, "HP");
            assert_eq!(value, "hp");
            assert_eq!(max, "hp_max");
            assert_eq!(*color, [220, 40, 40]);
        }
        _ => panic!("expected Bar"),
    }
    match &m.hud[2] {
        Widget::Text { template } => assert_eq!(template, "Lv {level}  Kills {kills}"),
        _ => panic!("expected Text"),
    }
    assert_eq!(m.screens.len(), 2);
    match m.screen("level_up").expect("level_up screen") {
        Screen::Menu { title, cards } => {
            assert_eq!(title, "Level Up!");
            assert_eq!(cards.len(), 2);
            match &cards[0].action {
                engine_ui::UiAction::Increment(f) => assert_eq!(f, "bolt_level"),
                _ => panic!("expected Increment action"),
            }
        }
        _ => panic!("expected Menu"),
    }
    match m.screen("dead").expect("dead screen") {
        Screen::End { title, summary, restart_label } => {
            assert_eq!(title, "You Died");
            assert_eq!(summary.len(), 3);
            assert_eq!(restart_label, "Restart (R)");
        }
        _ => panic!("expected End"),
    }
}

#[test]
fn empty_defaults_parse() {
    ControlsDescriptor::from_json(&controls::empty_controls_json()).unwrap();
    RenderDescriptor::from_json(&render::empty_render_json()).unwrap();
    UiModel::from_json(&ui_model::empty_ui_json()).unwrap();
}
