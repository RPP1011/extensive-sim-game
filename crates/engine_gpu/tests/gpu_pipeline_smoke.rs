//! GPU-feature integration test: instantiate every kernel + run one
//! tick of step_batch. The compass for the BindingSpec refactor —
//! every BGL/WGSL/bind() drift surfaces here as a wgpu validation
//! panic. Each panic identifies one kernel that needs its three
//! emit-layers reconciled.
//!
//! NOT a parity test (parity_with_cpu plays that role). This test
//! only asserts: every kernel can be instantiated against its BGL,
//! the bind-group can be created, and one full tick of step_batch
//! returns without panic.

#![cfg(feature = "gpu")]

mod common;

use common::smoke_fixture_n4;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::step::SimScratch;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;

#[test]
fn step_batch_instantiates_every_kernel() {
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

    // One step_batch call exercises every kernel in the SCHEDULE.
    // wgpu's eager validation panics on any BGL/WGSL/bind() drift —
    // that's the failure mode this test catches.
    gpu.step_batch(
        &mut state,
        &mut scratch,
        &mut events,
        &mut views,
        &policy,
        &cascade,
        1,
    );
    assert_eq!(state.tick, 1);
}
