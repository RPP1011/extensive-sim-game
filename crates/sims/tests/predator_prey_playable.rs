//! Plan C — `predator_prey` as a playable game through the generic
//! `make_playable` path with ZERO new Rust. The whole population is seeded
//! declaratively by the `init { spawn … }` block: 1 `PlayerHare` (the
//! human-driven hare, a distinct subkind) + 199 autonomous `Hare`s + 8
//! `Wolf`s, all gated by `creature_type` (the retired `mana < 0.5` slot-band
//! hack is gone). The PlayerHare is driven by human input via the
//! `@runtime config ctl` channel + the `HareControl @phase(per_agent)` rule
//! (guarded `self.creature_type == PlayerHare`). This is the generality
//! proof: a structurally-different game (evade, entity-subkind identity, no
//! weapons) authored purely in `assets/sim/predator_prey.sim`.
//!
//! Subkind ordinals are declaration order in predator_prey.sim:
//! Hare = 0, Wolf = 1, PlayerHare = 2 (declared after Wolf to keep the
//! existing Hare/Wolf ordinals stable). Seeded slots start at 1 (slot 0 is
//! the AgentId NonZeroU32 sentinel).
//!
//! Requires a GPU adapter (the runtime is GPU-backed); run with
//! `RUST_MIN_STACK=33554432 cargo test -p sims --test predator_prey_playable`.
//! When no adapter is present `make_playable` yields `None`; the GPU-driving
//! tests skip rather than fail (headless CI without a device).

const SEED: u64 = 0x9_BEEF_F00D_0001;
const AGENTS: u32 = 256;

// Declaration order in predator_prey.sim: Hare = 0, Wolf = 1, PlayerHare = 2.
const CT_HARE: u32 = 0;
const CT_WOLF: u32 = 1;
const CT_PLAYER_HARE: u32 = 2;
// Seeded counts (init { spawn … }): 1 PlayerHare + 199 Hare + 8 Wolf.
const PLAYER_HARE_COUNT: usize = 1;
const HARE_COUNT: usize = 199;
const WOLF_COUNT: usize = 8;
// scatter(42.0) radius for the autonomous Hare + Wolf populations.
const SCATTER_R: f32 = 42.0;

#[test]
fn predator_prey_is_registered() {
    assert!(
        sims::PLAYABLE_FIXTURES.contains(&"predator_prey"),
        "predator_prey must be in the registry: {:?}",
        sims::PLAYABLE_FIXTURES
    );
}

// Task 3 compile-gate (runs without a GPU too — but the only way to reach the
// descriptors is a constructed runtime, so skip gracefully without an adapter):
// all three player-facing descriptors emit + parse via their consumer crates.
#[test]
fn predator_prey_descriptors_emit_and_parse() {
    let Some(rt) = sims::make_playable("predator_prey", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping predator_prey descriptor checks");
        return;
    };
    // render: arena ring + PlayerHare follow-cam + PlayerHare/Hare/Wolf
    // creature_type color bands.
    let render = engine_play_api::RenderDescriptor::from_json(rt.render_descriptor())
        .expect("render_descriptor parses via engine_play_api");
    assert_eq!(
        render.agents.len(),
        3,
        "predator_prey render declares 3 creature_type color bands \
         (PlayerHare/Hare/Wolf): {:?}",
        render.agents
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

// Task 3 runtime gate: the `init { spawn … }` block self-seeds the whole
// population by subkind. Assert by `creature_type` (no manual seeding):
// exactly 1 PlayerHare + 199 Hare + 8 Wolf, all alive, with the autonomous
// Hare/Wolf populations scattered within the arena radius (PlayerHare at
// origin).
#[test]
fn seeded_population_by_subkind() {
    let Some(mut rt) = sims::make_playable("predator_prey", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping predator_prey seeding checks");
        return;
    };
    let snap = rt.agent_snapshot();
    assert_eq!(snap.len(), AGENTS as usize, "snapshot covers every slot");

    // Slot 0 is the AgentId NonZeroU32 sentinel — never seeded.
    assert!(!snap[0].alive, "slot 0 (AgentId sentinel) is not seeded");

    // Counts by creature_type over the seeded range (slots >= 1, alive).
    let players: Vec<_> = snap
        .iter()
        .enumerate()
        .filter(|(i, a)| *i >= 1 && a.alive && a.creature_type == CT_PLAYER_HARE)
        .map(|(_, a)| a)
        .collect();
    let hares: Vec<_> = snap
        .iter()
        .enumerate()
        .filter(|(i, a)| *i >= 1 && a.alive && a.creature_type == CT_HARE)
        .map(|(_, a)| a)
        .collect();
    let wolves: Vec<_> = snap
        .iter()
        .enumerate()
        .filter(|(i, a)| *i >= 1 && a.alive && a.creature_type == CT_WOLF)
        .map(|(_, a)| a)
        .collect();

    assert_eq!(
        players.len(),
        PLAYER_HARE_COUNT,
        "expected exactly {PLAYER_HARE_COUNT} live PlayerHare (creature_type=2)"
    );
    assert_eq!(
        hares.len(),
        HARE_COUNT,
        "expected {HARE_COUNT} live autonomous Hares (creature_type=0)"
    );
    assert_eq!(
        wolves.len(),
        WOLF_COUNT,
        "expected {WOLF_COUNT} live Wolves (creature_type=1)"
    );

    // Total seeded population is alive; nothing else is.
    let live = snap.iter().filter(|a| a.alive).count();
    assert_eq!(
        live,
        PLAYER_HARE_COUNT + HARE_COUNT + WOLF_COUNT,
        "only the seeded population is alive"
    );

    // PlayerHare seeded at origin (pos: origin).
    let p = players[0];
    let pr = (p.pos[0] * p.pos[0] + p.pos[1] * p.pos[1]).sqrt();
    assert!(
        pr < 1e-4,
        "PlayerHare seeded at origin, got {:?}",
        p.pos
    );

    // Autonomous Hares + Wolves scattered within the arena radius (42).
    let mut any_nonzero = false;
    for a in hares.iter().chain(wolves.iter()) {
        let r = (a.pos[0] * a.pos[0] + a.pos[1] * a.pos[1]).sqrt();
        assert!(
            r <= SCATTER_R + 1e-3,
            "scattered agent (creature_type={}) at {:?} (r={r}) exceeds scatter radius {SCATTER_R}",
            a.creature_type,
            a.pos
        );
        if r > 1e-4 {
            any_nonzero = true;
        }
    }
    assert!(
        any_nonzero,
        "scatter should place Hares/Wolves off the origin (not all at [0,0,0])"
    );

    eprintln!(
        "[predator_prey] PASS seeding: {} PlayerHare (origin), {} Hare + {} Wolf within r={SCATTER_R}",
        players.len(),
        hares.len(),
        wolves.len()
    );
}

// Task 3 runtime gate: the PlayerHare moves by INPUT, not by autonomous flee.
// The single seeded PlayerHare (creature_type=2) is driven by the
// `HareControl @phase(per_agent)` rule reading `config.ctl.move_*`. Under
// `ctl.move_x = 1.0` it must travel +X; reversing the input must reverse its
// travel. This is input-driven movement, not fleeing.
#[test]
fn player_hare_tracks_input() {
    let Some(mut rt) = sims::make_playable("predator_prey", SEED, AGENTS) else {
        eprintln!("no GPU adapter; skipping predator_prey input-drive test");
        return;
    };
    assert_eq!(rt.tick(), 0, "fresh runtime starts at tick 0");

    // Locate the single PlayerHare slot (creature_type=2). Seeded at slot 1,
    // but resolve it by creature_type rather than assuming the slot index.
    let player_slot = {
        let snap = rt.agent_snapshot();
        let slot = snap
            .iter()
            .position(|a| a.alive && a.creature_type == CT_PLAYER_HARE)
            .expect("a live PlayerHare must be seeded");
        // The PlayerHare must be alive (init seeds alive=1) — otherwise the
        // move guard would never fire and the assertions below are vacuous.
        assert!(snap[slot].alive, "PlayerHare slot should be alive");
        slot
    };

    // Drive +X.
    rt.set_input("ctl.move_x", 1.0);
    rt.set_input("ctl.move_y", 0.0);
    // Unknown field is a silent no-op (the `_ => {}` arm).
    rt.set_input("ctl.does_not_exist", 1.0);

    let x0 = rt.agent_snapshot()[player_slot].pos[0];
    for _ in 0..10 {
        rt.step();
    }
    let x1 = rt.agent_snapshot()[player_slot].pos[0];
    assert_eq!(rt.tick(), 10, "10 steps advance the tick to 10");
    // prey_speed = 0.5/tick × 10 ticks → ~+5.0 (clamped well inside arena 42).
    assert!(
        x1 > x0 + 1.0,
        "PlayerHare should move +X under move_x=1 (input-driven, not fleeing): {x0} -> {x1}"
    );

    // Reverse the input: the PlayerHare must reverse its travel.
    rt.set_input("ctl.move_x", -1.0);
    for _ in 0..10 {
        rt.step();
    }
    let x2 = rt.agent_snapshot()[player_slot].pos[0];
    assert!(
        x2 < x1,
        "PlayerHare should reverse under move_x=-1: {x1} -> {x2}"
    );
    eprintln!("[predator_prey] PASS: PlayerHare (slot {player_slot}) tracks input ({x0} -> {x1} -> {x2})");
}
