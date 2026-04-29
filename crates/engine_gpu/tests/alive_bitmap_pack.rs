//! Smoke for the AlivePackKernel WGSL body landed in commit b4c7b930.
//! Today verifies the dispatch path runs end-to-end without panicking
//! and that step_batch keeps state.tick consistent. Direct readback
//! of the alive_bitmap_buf is gated on snapshot infrastructure that
//! pre-T16 exposed a private accessor for; that readback is deferred
//! to a follow-up plan once the snapshot pipeline grows a public
//! alive-bitmap getter.

#![cfg(feature = "gpu")]

mod common;

use common::smoke_fixture_n4;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::step::SimScratch;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;

#[test]
fn alive_pack_kernel_dispatches_via_step_batch() {
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

    let tick0 = state.tick;
    gpu.step_batch(&mut state, &mut scratch, &mut events, &mut views, &policy, &cascade, 1);
    assert_eq!(state.tick, tick0 + 1);

    // Per-tick alive_pack runs as part of the SCHEDULE-loop dispatch
    // (bound at slot 0/1 with the agents SoA + bitmap output).
    // Today's gate is "doesn't panic + tick advances"; per-bit
    // verification follows once the snapshot pipeline exposes the
    // bitmap.
}
