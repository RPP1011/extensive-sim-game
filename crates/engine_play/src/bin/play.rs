//! `play` — the generic player binary. It constructs a compiled `.sim`
//! runtime by name via [`sims::make_playable`] and runs the windowed loop,
//! so one binary can play any migrated fixture. The runtime's own
//! `render`/`controls`/`ui` descriptors drive the renderer, input mapper,
//! and HUD (Plan D wires the real registry; Plan B's `MockRuntime` survives
//! behind `--mock` for a windowless smoke check on hosts with no fixture).
//!
//! Usage:
//!   play <fixture> [seed] [agents]   — e.g. `play play_probe`
//!   play --mock                      — the Wave-1 MockRuntime fallback
//!
//! Requires a display; on a headless host the windowed `run` will fail to
//! create a surface. The crate's correctness is verified headlessly via
//! `cargo test -p engine_play` (the `update`/bridge/input tests + the
//! `play_probe_end_to_end` registry gate).

use engine_play::mock::MockRuntime;
use engine_play::player::{mock_ui_model, Player, PlayerConfig};
use engine_ui::UiModel;

/// Default run parameters when the user omits them.
const DEFAULT_SEED: u64 = 0x5151_5151;
const DEFAULT_AGENTS: u32 = 64;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // `--mock`: the Wave-1 fallback (no fixture / no GPU adapter needed for
    // construction). Keeps a runnable smoke path independent of the registry.
    if args.iter().any(|a| a == "--mock") {
        eprintln!("[engine_play] starting `play` over MockRuntime (--mock fallback).");
        let rt = Box::new(MockRuntime::new());
        let player = Player::new(rt, PlayerConfig::default(), mock_ui_model())?;
        return player.run();
    }

    // `play <fixture> [seed] [agents]`.
    let name = match args.get(1) {
        Some(n) => n.clone(),
        None => {
            eprintln!(
                "usage: play <fixture> [seed] [agents]   (or `play --mock`)\n\
                 known fixtures: {:?}",
                sims::PLAYABLE_FIXTURES
            );
            std::process::exit(2);
        }
    };
    let seed = args
        .get(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_SEED);
    let agents = args
        .get(3)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_AGENTS);

    eprintln!("[engine_play] make_playable({name:?}, seed={seed}, agents={agents})");
    let rt = sims::make_playable(&name, seed, agents).unwrap_or_else(|| {
        eprintln!(
            "[engine_play] unknown or failed-to-construct fixture {name:?} \
             (no GPU adapter, or not in the registry).\n\
             known fixtures: {:?}",
            sims::PLAYABLE_FIXTURES
        );
        std::process::exit(2);
    });

    // Build the HUD model from the runtime's own `ui` descriptor; fall back
    // to the default model on a parse error (P10 — never hard-fail the player
    // on a malformed descriptor; degrade to a sane HUD).
    let ui_model = match UiModel::from_json(rt.ui_descriptor()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "[engine_play] ui_descriptor parse failed ({e}); using default HUD model."
            );
            mock_ui_model()
        }
    };

    let player = Player::new(rt, PlayerConfig::default(), ui_model)?;
    player.run()
}
