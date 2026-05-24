//! Mega-crate-hosted port of the cooldown_probe Plan E-A6 behavioural
//! pin. Replaces the test that previously lived inside
//! `crates/cooldown_probe_runtime/src/lib.rs`. The init { alive: 1,
//! cooldown_next_ready_tick: slot } block in
//! `assets/sim/cooldown_probe.sim` lifts the staggered-fire pattern
//! into the DSL surface — this test confirms the GeneratedRuntime
//! preserves it under the mega-crate's `pub mod cooldown_probe { ... }`
//! wrapping.

use sims::cooldown_probe::GeneratedRuntime;
use engine::sim_trait::CompiledSim;
use glam::Vec3;

const N: u32 = 8;
const TICKS: u64 = 16;

#[test]
fn staggered_init_drives_per_slot_fire_pattern() {
    let mut state = match GeneratedRuntime::try_new(0xC001_DA, N) {
        Some(s) => s,
        None => {
            eprintln!("[cooldown_probe init] skipping: no wgpu adapter on host.");
            return;
        }
    };
    for _ in 0..TICKS {
        state.step();
    }
    let r = read_activations(&mut state);
    assert_eq!(r.len(), N as usize);
    for (slot, &count) in r.iter().enumerate() {
        // Producer-then-consumer ordering: with the schedule's topo
        // sort placing the per-agent fire rule BEFORE its accumulator
        // ViewFold (post-T5), the producer atomicAdds into the
        // event_ring BEFORE the fold dispatch. Slot s fires when
        // `tick >= s` (init `cooldown_next_ready_tick: slot`).
        //
        // The fold cfg snapshot (event_tail captured at start-of-tick
        // into each fold's cfg.event_count slot, fixing the
        // over-bounds-walk-into-stale-records drift) means folds now
        // see PRIOR-tick events (the snapshot is taken before this
        // tick's producer atomicAdds run), so each tick's fire is
        // folded one tick LATER. Over TICKS ticks slot s contributes
        // `max(0, TICKS - s - 1)` activations (the very last tick's
        // fire is captured by the next tick's snapshot, but no next
        // tick runs).
        let expected = (TICKS as i64 - slot as i64 - 1).max(0) as f32;
        assert!(
            (count - expected).abs() < 1e-3,
            "slot {slot} activations = {count} (expected {expected})",
        );
    }
    let unique: std::collections::BTreeSet<_> =
        r.iter().map(|f| f.to_bits()).collect();
    assert!(
        unique.len() >= 5,
        "expected staggered values across {} slots, got {} unique: {:?}",
        N, unique.len(), r,
    );
}

/// Per-caster activation count readback. Port of the per-fixture
/// crate's `activations()` impl method — implemented here as a free
/// helper so the mega-crate's auto-generated `pub mod cooldown_probe`
/// stays purely emitted.
fn read_activations(state: &mut GeneratedRuntime) -> Vec<f32> {
    let bytes = (state.agent_count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cooldown_probe::activations_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("cooldown_probe::activations_readback"),
        },
    );
    // Per-view storage (was `view_storage_primary_buf`, now per-view
    // after the aliasing-gap fix). cooldown_probe declares one view
    // (`activations`), so the per-view buffer is the activations one.
    encoder.copy_buffer_to_buffer(
        &state.view_storage_activations_primary_buf,
        0,
        &staging,
        0,
        bytes,
    );
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("activations_staging map_async failed")
    });
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..state.agent_count as usize].to_vec()
    };
    staging.unmap();
    out
}

// CompiledSim shim: drives the GeneratedRuntime via the trait, mirrors
// the per-fixture-crate impl. Kept in-test so the mega-crate's
// generated module doesn't need to know about CompiledSim — that
// trait is engine-side glue, not compiler-emitted.
struct Driver<'a>(&'a mut GeneratedRuntime);

impl CompiledSim for Driver<'_> {
    fn step(&mut self) {
        self.0.step();
    }
    fn agent_count(&self) -> u32 {
        self.0.agent_count
    }
    fn tick(&self) -> u64 {
        self.0.tick
    }
    fn positions(&mut self) -> &[Vec3] {
        &[]
    }
}

// Suppress dead-code lint — Driver isn't wired into this test today
// but is kept as the canonical pattern for any external caller that
// wants to reach the GeneratedRuntime through CompiledSim.
#[allow(dead_code)]
fn _unused() -> Driver<'static> {
    panic!("placeholder")
}
