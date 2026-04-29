//! Smoke: step_batch's encoder doesn't deadlock under async/poll.
//! Skipped on default features — there's no async path; the no-gpu
//! GpuBackend is a synchronous CPU forwarder. Under `--features gpu`,
//! step_batch internally polls the device after submit; this test is
//! mostly about confirming the call returns rather than blocks.

#![cfg(feature = "gpu")]

mod common;

use common::smoke_fixture_n4;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::step::SimScratch;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;

#[test]
fn step_batch_returns_under_gpu_feature() {
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

    // 3 small batches in a row exercises the encoder/submit/poll loop
    // multiple times — would surface device-poll deadlocks.
    for _ in 0..3 {
        gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 1);
    }
    assert!(state.tick >= 3);
}
