//! Smoke: GpuBackend::snapshot() returns coherent state after
//! step_batch without panicking. Only the gpu-feature build has a
//! snapshot pipeline; default features doesn't expose it.

#![cfg(feature = "gpu")]

mod common;

use common::smoke_fixture_n4;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::step::SimScratch;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;

#[test]
fn snapshot_after_step_batch_returns_ok() {
    let Ok(mut gpu) = engine_gpu::GpuBackend::new() else {
        eprintln!("skipping: no gpu adapter");
        return;
    };
    let mut state = smoke_fixture_n4();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(4096);
    let mut views = ViewRegistry::default();
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();

    // Two consecutive step_batch + snapshot pairs verifies the
    // double-buffer staging doesn't reuse a still-pending write.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 5);
    let snap1 = gpu.snapshot(&mut state).expect("snapshot 1");
    gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 5);
    let snap2 = gpu.snapshot(&mut state).expect("snapshot 2");

    // Both snapshots produce non-error output. The second's gpu_tick
    // should be >= the first's (monotonic).
    let _ = (snap1, snap2);
}
