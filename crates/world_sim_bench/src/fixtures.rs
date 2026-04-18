//! Fixture loader for committed bincode-serialized WorldState snapshots.
//!
//! Fixtures live in `crates/world_sim_bench/fixtures/world_<scale>.bin`
//! and are regenerated via `xtask world-sim ... --output <path>`.

use anyhow::{anyhow, Result};
use game::world_sim::state::WorldState;
use std::path::Path;

pub fn load(scale: &str) -> Result<WorldState> {
    // Resolve relative to this crate's root, not the CWD of the bench runner.
    let path = format!(
        "{}/fixtures/world_{}.bin",
        env!("CARGO_MANIFEST_DIR"),
        scale
    );
    let bytes = std::fs::read(&path).map_err(|e| {
        anyhow!(
            "fixture missing at {}: {}. Regenerate via: \
             cargo run --release --bin xtask --features profile-systems -- \
             world-sim --ticks 500 --output {}",
            path, e, path
        )
    })?;
    let state: WorldState = bincode::deserialize(&bytes)?;
    Ok(state)
}

#[allow(dead_code)]
pub fn fixture_path(scale: &str) -> String {
    format!("{}/fixtures/world_{}.bin", env!("CARGO_MANIFEST_DIR"), scale)
}

#[allow(dead_code)]
pub fn exists(scale: &str) -> bool {
    Path::new(&fixture_path(scale)).exists()
}
