//! Subkind-seeding (Plan A) parser coverage for the extended `init {}`
//! grammar: f32 init values, `spawn <Subkind> count <N> { … }` population
//! blocks, position builtins (`origin` / `scatter(r)` / `ring(r)`), and the
//! `render {} agent when creature_type is <Subkind>` selector. Pins the
//! surface grammar + parsed AST shape; codegen is covered in `sims`.

use dsl_ast::ast::{CountExpr, Decl, InitExpr, PosBuiltin, RadiusArg};

fn first_init(src: &str) -> dsl_ast::ast::InitDecl {
    let p = dsl_ast::parse(src).expect("parse program");
    for d in p.decls {
        if let Decl::Init(i) = d {
            return i;
        }
    }
    panic!("no init decl parsed");
}

#[test]
fn flat_init_back_compat_int_and_slot() {
    // The pre-existing flat form must still parse with no spawn blocks.
    let init = first_init("init { alive: 1, hp: 100, mana: slot }\n");
    assert!(init.spawns.is_empty(), "flat form has no spawn blocks");
    assert_eq!(init.stmts.len(), 3);
    assert_eq!(init.stmts[0].field, "alive");
    assert_eq!(init.stmts[0].expr, InitExpr::Const(1));
    assert_eq!(init.stmts[1].expr, InitExpr::Const(100));
    assert_eq!(init.stmts[2].expr, InitExpr::Slot);
}

#[test]
fn flat_init_accepts_float_values() {
    // Float fills lower to InitExpr::Float.
    let init = first_init("init { hp: 100.0, mana: 0.5 }\n");
    assert_eq!(init.stmts[0].field, "hp");
    assert_eq!(init.stmts[0].expr, InitExpr::Float(100.0));
    assert_eq!(init.stmts[1].expr, InitExpr::Float(0.5));
}

#[test]
fn spawn_blocks_parse_count_and_fields() {
    // Test two population blocks with literal + config counts.
    let src = "\
        config waves { cap: i32 = 511 }\n\
        init {\n\
          spawn Player count 1    { hp: 100.0, pos: origin }\n\
          spawn Enemy  count config.waves.cap { alive: 0 }\n\
        }\n";
    let init = first_init(src);
    assert_eq!(init.spawns.len(), 2, "two spawn blocks");
    assert!(init.stmts.is_empty(), "no flat stmts alongside spawns here");

    let player = &init.spawns[0];
    assert_eq!(player.subkind, "Player");
    assert_eq!(player.count, CountExpr::Lit(1));
    assert_eq!(player.fields.len(), 2);
    assert_eq!(player.fields[0].field, "hp");
    assert_eq!(player.fields[0].expr, InitExpr::Float(100.0));
    assert_eq!(player.fields[1].field, "pos");
    assert_eq!(player.fields[1].expr, InitExpr::Pos(PosBuiltin::Origin));

    let enemy = &init.spawns[1];
    assert_eq!(enemy.subkind, "Enemy");
    assert_eq!(enemy.count, CountExpr::Config("waves.cap".to_string()));
    assert_eq!(enemy.fields[0].field, "alive");
    assert_eq!(enemy.fields[0].expr, InitExpr::Const(0));
}

#[test]
fn position_builtins_parse() {
    // Position builtins parse as init field values.
    let src = "\
        init {\n\
          spawn A count 1 { pos: origin }\n\
          spawn B count 4 { pos: scatter(40.0) }\n\
          spawn C count 4 { pos: ring(12.5) }\n\
        }\n";
    let init = first_init(src);
    assert_eq!(init.spawns[0].fields[0].expr, InitExpr::Pos(PosBuiltin::Origin));
    assert_eq!(init.spawns[1].fields[0].expr, InitExpr::Pos(PosBuiltin::Scatter(RadiusArg::Lit(40.0))));
    assert_eq!(init.spawns[2].fields[0].expr, InitExpr::Pos(PosBuiltin::Ring(RadiusArg::Lit(12.5))));
}

#[test]
fn config_ref_init_values_parse() {
    // Grammar-gap close: `config.<block>.<field>` as an init field VALUE and
    // as a `scatter`/`ring` radius (resolved to the config default at codegen,
    // the same way a spawn `count config.x` resolves).
    let src = "\
        init {\n\
          spawn A count 1 { hp: config.vs.player_hp, pos: scatter(config.hunt.arena_radius) }\n\
          spawn B count 2 { pos: ring(config.hunt.arena_radius) }\n\
        }\n";
    let init = first_init(src);
    assert_eq!(
        init.spawns[0].fields[0].expr,
        InitExpr::ConfigRef("vs.player_hp".into())
    );
    assert_eq!(
        init.spawns[0].fields[1].expr,
        InitExpr::Pos(PosBuiltin::Scatter(RadiusArg::Config("hunt.arena_radius".into())))
    );
    assert_eq!(
        init.spawns[1].fields[0].expr,
        InitExpr::Pos(PosBuiltin::Ring(RadiusArg::Config("hunt.arena_radius".into())))
    );
}

#[test]
fn render_creature_type_is_subkind_parses() {
    // The `agent when creature_type is <Subkind>` render selector: the parser
    // records the subkind name; the JSON emitter resolves it to the creature_type
    // ordinal (declaration order).
    let src = "\
        entity Hare : Agent {}\n\
        entity Wolf : Agent {}\n\
        render {\n\
          arena_radius 40.0\n\
          camera observer\n\
          agent when creature_type is Hare { color (0, 220, 220) }\n\
          agent when creature_type is Wolf { color (220, 80, 40) }\n\
        }\n";
    let p = dsl_ast::parse(src).expect("parse render with subkind selector");
    let r = p.render.expect("render block present");
    assert_eq!(r.agents.len(), 2);
    assert_eq!(r.agents[0].when.field, "creature_type");
    assert_eq!(r.agents[0].when.subkind.as_deref(), Some("Hare"));
    assert_eq!(r.agents[1].when.subkind.as_deref(), Some("Wolf"));
}
