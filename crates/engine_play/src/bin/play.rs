//! `play` — the generic player binary. For Wave 1 it constructs a
//! [`MockRuntime`] and runs the windowed loop, proving the loop compiles and
//! drives a real runtime. The real registry wire (`sims::make_playable(argv)`)
//! is Plan D; until then this exercises the contract against the mock.
//!
//! Requires a display; on a headless host the windowed `run` will fail to
//! create a surface. The crate's correctness is verified headlessly via
//! `cargo test -p engine_play` (the `update`/bridge/input tests).

use engine_play::mock::MockRuntime;
use engine_play::player::{mock_ui_model, Player, PlayerConfig};

fn main() -> anyhow::Result<()> {
    eprintln!("[engine_play] starting `play` over MockRuntime (Wave 1).");
    eprintln!(
        "[engine_play] What you should see: a top-down flat voxel arena, a cyan \
         player splat, an orange enemy splat, a nova ring every 40 ticks, and a \
         bolt beam every 12 ticks. WASD moves; the HUD shows HP/XP."
    );
    let rt = Box::new(MockRuntime::new());
    let player = Player::new(rt, PlayerConfig::default(), mock_ui_model())?;
    player.run()
}
