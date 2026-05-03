//! Smoke test for `engine::spatial::nearest_k` — the CPU-side top-K
//! topological neighbour query that backs the DSL's
//! `query.nearest_k(self, K, max_radius)` for-iter source.
//!
//! Mirrors the (planned) WGSL emit's per-thread bounded-heap walk: for
//! a ring of kin around a center, the K nearest by `(distance, raw_id)`
//! should win, with ties broken on raw id ascending so the CPU and GPU
//! produce identical neighbour sets.

use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_data::entities::CreatureType;
use glam::Vec3;

fn spawn_wolf(state: &mut SimState, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos,
            hp: 80.0,
            max_hp: 80.0,
            ..Default::default()
        })
        .expect("wolf spawn")
}

/// 7 nearest of 12 candidates — confirms the K-cap is enforced and the
/// surviving set is the K closest by Euclidean distance.
#[test]
fn nearest_k_returns_k_closest_by_distance() {
    let mut state = SimState::new(32, 0xC0FFEE);
    let center = spawn_wolf(&mut state, Vec3::ZERO);
    let mut others: Vec<AgentId> = Vec::with_capacity(12);
    for i in 1..=12 {
        others.push(spawn_wolf(&mut state, Vec3::new(i as f32, 0.0, 0.0)));
    }

    let k = 7u32;
    let max_radius = 50.0;
    let result = engine::spatial::nearest_k(&state, center, k, max_radius);

    assert_eq!(
        result.len(),
        k as usize,
        "nearest_k should cap output at K=7, got {}: {:?}",
        result.len(),
        result
    );
    for i in 0..7 {
        assert_eq!(
            result[i], others[i],
            "slot {} should be the i+1-radius wolf, got {:?}",
            i, result[i]
        );
    }
    for i in 7..12 {
        assert!(
            !result.contains(&others[i]),
            "wolf at radius {} should be dropped by top-K, but appears in {:?}",
            i + 1,
            result
        );
    }
}

/// Fewer than K candidates → return all of them, in distance order.
#[test]
fn nearest_k_returns_fewer_when_pool_smaller_than_k() {
    let mut state = SimState::new(8, 0xC0FFEE);
    let center = spawn_wolf(&mut state, Vec3::ZERO);
    let a = spawn_wolf(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let b = spawn_wolf(&mut state, Vec3::new(1.0, 0.0, 0.0));

    let result = engine::spatial::nearest_k(&state, center, 7, 50.0);
    assert_eq!(result.len(), 2, "expected 2 results, got {:?}", result);
    assert_eq!(result[0], b, "expected b (radius 1) first");
    assert_eq!(result[1], a, "expected a (radius 2) second");
}

/// K=0 short-circuits to an empty result without scanning.
#[test]
fn nearest_k_zero_returns_empty() {
    let mut state = SimState::new(4, 0xC0FFEE);
    let center = spawn_wolf(&mut state, Vec3::ZERO);
    spawn_wolf(&mut state, Vec3::new(1.0, 0.0, 0.0));
    let result = engine::spatial::nearest_k(&state, center, 0, 50.0);
    assert!(result.is_empty(), "K=0 should yield empty: {:?}", result);
}

/// Tie-break on raw `AgentId` ascending — two kin at exactly the same
/// distance must resolve in a deterministic order so the GPU heap walk
/// matches.
#[test]
fn nearest_k_breaks_ties_on_raw_id_ascending() {
    let mut state = SimState::new(8, 0xC0FFEE);
    let center = spawn_wolf(&mut state, Vec3::ZERO);
    let a = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));
    let b = spawn_wolf(&mut state, Vec3::new(-3.0, 0.0, 0.0));
    let result = engine::spatial::nearest_k(&state, center, 1, 50.0);
    assert_eq!(result.len(), 1);
    let (lo, _hi) = if a.raw() < b.raw() { (a, b) } else { (b, a) };
    assert_eq!(
        result[0], lo,
        "tie-break should pick the lower raw id, got {:?} (a={:?}, b={:?})",
        result[0], a, b
    );
}

/// Cross-species kin are filtered out — same contract as `nearby_kin`.
#[test]
fn nearest_k_filters_cross_species() {
    let mut state = SimState::new(8, 0xC0FFEE);
    let center = spawn_wolf(&mut state, Vec3::ZERO);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.5, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        })
        .expect("human spawn");
    let kin = spawn_wolf(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let result = engine::spatial::nearest_k(&state, center, 3, 50.0);
    assert_eq!(result, vec![kin], "cross-species human must be filtered out");
}

/// `max_radius` bounds the candidate sweep — kin outside the radius
/// are not considered even if K isn't full.
#[test]
fn nearest_k_respects_max_radius_bound() {
    let mut state = SimState::new(8, 0xC0FFEE);
    let center = spawn_wolf(&mut state, Vec3::ZERO);
    let near = spawn_wolf(&mut state, Vec3::new(1.0, 0.0, 0.0));
    let far = spawn_wolf(&mut state, Vec3::new(20.0, 0.0, 0.0));
    let _ = far;
    let result = engine::spatial::nearest_k(&state, center, 7, 5.0);
    assert_eq!(
        result,
        vec![near],
        "only kin inside max_radius=5.0 should appear, got {:?}",
        result
    );
}
