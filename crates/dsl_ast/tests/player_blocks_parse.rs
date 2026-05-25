//! Plan A — parser coverage for the player-facing descriptor blocks
//! (`controls {}`, `render {}`, `ui {}`). Each is a singleton top-level
//! block parsed onto the `Program` (not a `Decl`). These tests pin the
//! surface grammar + the parsed AST shape; the JSON lowering is covered
//! in `dsl_compiler`.

use dsl_ast::ast::{CameraDecl, UiScreen, UiWidget, VfxKindDecl};

#[test]
fn controls_block_parses_bindings() {
    let src = "\
        controls {\n\
          key \"w\" -> ctl.move_y: 1.0\n\
          key \"s\" -> ctl.move_y: -1.0\n\
          key \"space\" -> ctl.bolt_rate_level: 1.0 press\n\
        }\n";
    let p = dsl_ast::parse(src).expect("parse controls");
    let c = p.controls.expect("controls block present");
    assert_eq!(c.bindings.len(), 3);
    assert_eq!(c.bindings[0].key, "w");
    assert_eq!(c.bindings[0].block, "ctl");
    assert_eq!(c.bindings[0].field, "move_y");
    assert_eq!(c.bindings[0].value, 1.0);
    assert!(!c.bindings[0].press, "default is Hold");
    assert_eq!(c.bindings[1].value, -1.0);
    assert!(c.bindings[2].press, "trailing `press` → Press mode");
    assert_eq!(c.bindings[2].field, "bolt_rate_level");
}

#[test]
fn render_block_parses_camera_agents_vfx() {
    let src = "\
        render {\n\
          arena_radius 120.0\n\
          camera follow when mana in [0.5, 1.5]\n\
          agent when mana in [0.5, 1.5] { color (0, 220, 220) }\n\
          agent when mana in [1.5, 9.9] { color (220, 80, 40) }\n\
          vfx on NovaFire period 40 { ring radius 6.0 color (255, 255, 120) }\n\
          vfx on Bolt period 12 { beam_to_nearest when mana in [1.5, 9.9] color (120, 200, 255) }\n\
        }\n";
    let p = dsl_ast::parse(src).expect("parse render");
    let r = p.render.expect("render block present");
    assert_eq!(r.arena_radius, 120.0);
    match &r.camera {
        CameraDecl::Follow(range) => {
            assert_eq!(range.field, "mana");
            assert_eq!(range.lo, 0.5);
            assert_eq!(range.hi, 1.5);
        }
        _ => panic!("expected follow camera"),
    }
    assert_eq!(r.agents.len(), 2);
    assert_eq!(r.agents[0].color, [0, 220, 220]);
    assert_eq!(r.vfx.len(), 2);
    assert_eq!(r.vfx[0].on_rule, "NovaFire");
    assert_eq!(r.vfx[0].period, 40);
    assert!(matches!(r.vfx[0].kind, VfxKindDecl::Ring));
    assert_eq!(r.vfx[0].radius, 6.0);
    match &r.vfx[1].kind {
        VfxKindDecl::BeamToNearest { target } => assert_eq!(target.field, "mana"),
        _ => panic!("expected beam_to_nearest vfx"),
    }
}

#[test]
fn render_observer_camera_parses() {
    let src = "render { arena_radius 50.0  camera observer }\n";
    let p = dsl_ast::parse(src).expect("parse render observer");
    let r = p.render.expect("render block present");
    assert!(matches!(r.camera, CameraDecl::Observer));
    assert!(r.agents.is_empty());
    assert!(r.vfx.is_empty());
}

#[test]
fn ui_block_parses_hud_menu_screen() {
    let src = "\
        ui {\n\
          hud {\n\
            bar \"HP\" value hp max hp_max color (220, 40, 40)\n\
            bar \"XP\" value xp max xp_per_level color (40, 160, 240)\n\
            text \"Lv {level}  Kills {kills}\"\n\
          }\n\
          menu level_up \"Level Up!\" {\n\
            card \"Bolt Damage +\" -> bolt_level\n\
            card \"Nova +\" -> nova_level\n\
          }\n\
          screen dead \"You Died\" { summary time level kills  restart \"Restart (R)\" }\n\
        }\n";
    let p = dsl_ast::parse(src).expect("parse ui");
    let u = p.ui.expect("ui block present");
    assert_eq!(u.hud.len(), 3);
    match &u.hud[0] {
        UiWidget::Bar { label, value, max, color } => {
            assert_eq!(label, "HP");
            assert_eq!(value, "hp");
            assert_eq!(max, "hp_max");
            assert_eq!(*color, [220, 40, 40]);
        }
        _ => panic!("expected bar widget"),
    }
    match &u.hud[2] {
        UiWidget::Text { template } => assert_eq!(template, "Lv {level}  Kills {kills}"),
        _ => panic!("expected text widget"),
    }
    assert_eq!(u.screens.len(), 2);
    match &u.screens[0] {
        UiScreen::Menu { name, title, cards } => {
            assert_eq!(name, "level_up");
            assert_eq!(title, "Level Up!");
            assert_eq!(cards.len(), 2);
            assert_eq!(cards[0].action_field, "bolt_level");
        }
        _ => panic!("expected menu screen"),
    }
    match &u.screens[1] {
        UiScreen::End { name, title, summary, restart_label } => {
            assert_eq!(name, "dead");
            assert_eq!(title, "You Died");
            assert_eq!(summary.len(), 3);
            assert_eq!(summary[0].0, "time");
            assert_eq!(restart_label, "Restart (R)");
        }
        _ => panic!("expected end screen"),
    }
}

#[test]
fn duplicate_block_rejected() {
    let src = "render { arena_radius 1.0 camera observer }\nrender { arena_radius 2.0 camera observer }\n";
    assert!(dsl_ast::parse(src).is_err(), "duplicate render block must error");
}

#[test]
fn absent_blocks_are_none() {
    let src = "config ctl { move_x: f32 = 0.0 @runtime }\n";
    let p = dsl_ast::parse(src).expect("parse");
    assert!(p.controls.is_none());
    assert!(p.render.is_none());
    assert!(p.ui.is_none());
}
