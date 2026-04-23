//! Task A2 — `rebuild_and_query_resident` path tests.
//!
//! Two coverage points:
//!
//! 1. Parity between the sync `rebuild_and_query` path and the
//!    GPU-resident `rebuild_and_query_resident` path on a 32-agent
//!    cluster. The resident path uses the GPU prefix-scan kernel (A1)
//!    instead of the CPU exclusive-scan the sync path uses, so the two
//!    paths should produce byte-identical query results.
//!
//! 2. The alive-filter in the query kernel itself: killing an agent
//!    before running the resident path must scrub its id from every
//!    kin list. This used to live in `cascade::filter_dead_from_kin` as
//!    a post-query CPU pass; the WGSL-level check makes that pass
//!    structurally redundant.

#![cfg(feature = "gpu")]

use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::spatial_gpu::SpatialTestHarness;
use glam::Vec3;

#[test]
fn rebuild_and_query_resident_matches_sync_path() {
    // 32 agents in a rough cluster. Sync path (CPU scan) and resident
    // path (GPU scan) results must be byte-identical for the same
    // fixture + radius.
    let mut state = SimState::new(64, 0xCAFE);
    for i in 0..32 {
        let angle = (i as f32) * 0.2;
        let r = 8.0 + (i as f32 % 4.0);
        state
            .spawn_agent(AgentSpawn {
                creature_type: if i % 2 == 0 {
                    CreatureType::Human
                } else {
                    CreatureType::Wolf
                },
                pos: Vec3::new(r * angle.cos(), r * angle.sin(), 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }

    let mut harness = SpatialTestHarness::new().expect("spatial init");
    let sync = harness.run_sync(&state, 12.0).expect("sync query");
    let resident = harness.run_resident(&state, 12.0).expect("resident query");

    assert_eq!(resident.nearest_hostile, sync.nearest_hostile);
    for i in 0..32 {
        assert_eq!(
            resident.nearby_kin[i].count, sync.nearby_kin[i].count,
            "kin count diverged at slot {i}"
        );
        assert_eq!(
            resident.nearby_kin[i].as_slice(),
            sync.nearby_kin[i].as_slice(),
            "kin list diverged at slot {i}"
        );
        assert_eq!(
            resident.within_radius[i].count, sync.within_radius[i].count,
            "within_radius count diverged at slot {i}"
        );
        assert_eq!(
            resident.within_radius[i].as_slice(),
            sync.within_radius[i].as_slice(),
            "within_radius list diverged at slot {i}"
        );
    }
}

#[test]
fn alive_filter_in_query_kernel() {
    // Spawn 4 humans on a line, kill raw id=3 (slot 2), assert the
    // dead id does not appear in any nearby_kin list. Exercises the
    // alive-filter that used to live in `filter_dead_from_kin` on CPU.
    let mut state = SimState::new(8, 0xFEED);
    for i in 0..4 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32 * 2.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }
    let dead = AgentId::new(3).expect("agent id 3");
    state.kill_agent(dead);

    let mut harness = SpatialTestHarness::new().expect("spatial init");
    let res = harness.run_resident(&state, 12.0).expect("resident query");

    for (slot, q) in res.nearby_kin.iter().enumerate() {
        let count = (q.count as usize).min(q.ids.len());
        for i in 0..count {
            assert_ne!(
                q.ids[i], 3,
                "dead id 3 appeared in kin list at slot {slot} (ids={:?})",
                &q.ids[..count],
            );
        }
    }
}
