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

#[test]
fn two_resident_queries_in_one_encoder_do_not_alias() {
    // Two back-to-back resident queries in one encoder, with different
    // radii, must not clobber each other's outputs. This is the C1
    // regression guard: previously `rebuild_and_query_resident`
    // returned a `PoolHandle` into a single per-hash output trio, so
    // both calls landed in the same buffers and the second overwrote
    // the first.
    //
    // Fixture: 8 humans + 8 wolves in a tight cluster so both species
    // have plenty of hostile-in-range candidates for the wider radius
    // but far fewer for the tighter one — the two `nearest` vectors
    // must therefore differ in at least some slots, proving the
    // outputs really are independent.
    let mut state = SimState::new(32, 0xA11A);
    // Humans on a tight cluster around origin.
    for i in 0..8 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new((i as f32) * 0.5, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }
    // Wolves farther out — within 12.0 m but outside 2.0 m for most
    // human slots, so the `engagement` radius=2.0 query will see
    // many slots with no hostile in range while the radius=12.0 query
    // will see a hostile for everyone.
    for i in 0..8 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new((i as f32) * 0.5, 5.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }

    let mut harness = SpatialTestHarness::new().expect("spatial init");

    // Sanity-check the fixture via the sync path first: the wider
    // radius must see hostiles, the tighter one must miss at least
    // some. If this invariant doesn't hold, the aliasing assertion
    // below would pass vacuously.
    let sync_wide = harness.run_sync(&state, 12.0).expect("sync r=12");
    let sync_tight = harness.run_sync(&state, 2.0).expect("sync r=2");
    assert!(
        sync_wide.nearest_hostile.iter().any(|&v| v != u32::MAX),
        "fixture: sync r=12 found no hostiles (got {:?})",
        sync_wide.nearest_hostile,
    );
    let wide_tight_differ = sync_wide
        .nearest_hostile
        .iter()
        .zip(sync_tight.nearest_hostile.iter())
        .any(|(a, b)| a != b);
    assert!(
        wide_tight_differ,
        "fixture: sync r=12 and r=2 produced identical nearest_hostile — \
         test can't detect aliasing (wide:{:?}, tight:{:?})",
        sync_wide.nearest_hostile, sync_tight.nearest_hostile,
    );

    let cap = state.agent_cap();
    let outputs_kin = harness.alloc_output_buffers(cap, "kin");
    let outputs_hostile = harness.alloc_output_buffers(cap, "hostile");

    let (kin_res, hostile_res) = harness
        .run_two_resident_queries_in_one_encoder(&state, 12.0, &outputs_kin, 2.0, &outputs_hostile)
        .expect("two resident queries");

    assert_eq!(kin_res.nearest_hostile.len(), hostile_res.nearest_hostile.len());

    // Find at least one slot where the two resident outputs differ —
    // that's the evidence the outputs aren't aliased. (If aliasing
    // was still present, the second call would overwrite the first
    // and the two vectors would be byte-identical.)
    let differ = kin_res
        .nearest_hostile
        .iter()
        .zip(hostile_res.nearest_hostile.iter())
        .any(|(a, b)| a != b);
    assert!(
        differ,
        "both resident queries produced identical `nearest_hostile` — \
         outputs may still be aliased (radius=12.0 trio:{:?}; radius=2.0 trio:{:?})",
        kin_res.nearest_hostile, hostile_res.nearest_hostile,
    );

    // Cross-check each resident trio matches the sync-path ground
    // truth for its radius — proves we got the *right* independent
    // results, not just two garbage vectors that happen to differ.
    assert_eq!(
        kin_res.nearest_hostile, sync_wide.nearest_hostile,
        "resident r=12 diverged from sync r=12",
    );
    assert_eq!(
        hostile_res.nearest_hostile, sync_tight.nearest_hostile,
        "resident r=2 diverged from sync r=2",
    );
}
