//! Task 81 — height-bonus scoring gate on Attack.
//!
//! Validates `policy::utility::terrain_height_bonus` — the hardcoded
//! post-step modifier `score_entry` applies on the Attack row. The gate:
//!
//!   `shooter.z > target.z + 2.0 && terrain.line_of_sight(shooter, target)`
//!
//! Bonus value: `+0.35`. Bonus is `0.0` otherwise.
//!
//! FlatPlane (SimState default) returns clear LOS and height=0
//! everywhere — so on the canonical wolves+humans fixture the gate
//! never fires. The z-coordinate still has to be moved explicitly for
//! the bonus to activate. This is the wolves+humans parity story: the
//! seam is in place, runtime-inert for existing content.

use engine::creature::CreatureType;
use engine::policy::utility::{
    terrain_height_bonus, TERRAIN_HEIGHT_BONUS, TERRAIN_HEIGHT_THRESHOLD_M,
};
use engine::state::{AgentSpawn, MovementMode, SimState};
use engine::terrain::TerrainQuery;
use glam::Vec3;
use std::sync::Arc;

/// Hand-authored terrain that blocks every LOS — used to assert the
/// bonus respects the LOS gate.
struct OpaqueTerrain;
impl TerrainQuery for OpaqueTerrain {
    fn height_at(&self, _x: f32, _y: f32) -> f32 {
        0.0
    }
    fn walkable(&self, _pos: Vec3, _mode: MovementMode) -> bool {
        true
    }
    fn line_of_sight(&self, _from: Vec3, _to: Vec3) -> bool {
        false
    }
}

fn spawn(state: &mut SimState, ct: CreatureType, pos: Vec3) -> engine::ids::AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: ct,
            pos,
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        })
        .unwrap()
}

#[test]
fn flat_plane_default_yields_no_bonus_for_coplanar_agents() {
    let mut state = SimState::new(4, 42);
    let a = spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 0.0));
    let b = spawn(&mut state, CreatureType::Wolf, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(terrain_height_bonus(&state, a, b), 0.0);
}

#[test]
fn height_above_threshold_with_clear_los_fires_bonus() {
    let mut state = SimState::new(4, 42);
    // +5 m above the 2 m gate — bonus should fire on the FlatPlane
    // default (LOS clear by construction).
    let shooter = spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 5.0));
    let target = spawn(&mut state, CreatureType::Wolf, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(terrain_height_bonus(&state, shooter, target), TERRAIN_HEIGHT_BONUS);
}

#[test]
fn exactly_at_threshold_does_not_fire_bonus() {
    // Strict inequality: z - target_z must be GREATER than 2.0.
    let mut state = SimState::new(4, 42);
    let shooter = spawn(
        &mut state,
        CreatureType::Human,
        Vec3::new(0.0, 0.0, TERRAIN_HEIGHT_THRESHOLD_M),
    );
    let target = spawn(&mut state, CreatureType::Wolf, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(terrain_height_bonus(&state, shooter, target), 0.0);
}

#[test]
fn shooter_below_target_never_fires_bonus() {
    // Reverse geometry — target is up the hill, shooter at ground.
    let mut state = SimState::new(4, 42);
    let shooter = spawn(&mut state, CreatureType::Wolf, Vec3::new(0.0, 0.0, 0.0));
    let target = spawn(&mut state, CreatureType::Human, Vec3::new(1.0, 0.0, 5.0));
    assert_eq!(terrain_height_bonus(&state, shooter, target), 0.0);
}

#[test]
fn blocked_los_vetoes_bonus() {
    // Elevation is above the gate, but terrain says no — the bonus
    // should withhold. This is the "shooter on far cliff can't
    // exploit elevation through a rock" case.
    let mut state = SimState::new(4, 42);
    let shooter = spawn(&mut state, CreatureType::Human, Vec3::new(0.0, 0.0, 10.0));
    let target = spawn(&mut state, CreatureType::Wolf, Vec3::new(5.0, 0.0, 0.0));
    // Sanity: FlatPlane default fires the bonus.
    assert_eq!(terrain_height_bonus(&state, shooter, target), TERRAIN_HEIGHT_BONUS);
    // Now replace terrain with an opaque backend — bonus withheld.
    state.terrain = Arc::new(OpaqueTerrain);
    assert_eq!(terrain_height_bonus(&state, shooter, target), 0.0);
}

#[test]
fn missing_agent_yields_zero_bonus() {
    // Defensive: a freed / invalid AgentId surfaces as zero rather
    // than NaN / panic.
    let state = SimState::new(4, 42);
    let fake_a = engine::ids::AgentId::new(1).unwrap();
    let fake_b = engine::ids::AgentId::new(2).unwrap();
    assert_eq!(terrain_height_bonus(&state, fake_a, fake_b), 0.0);
}
