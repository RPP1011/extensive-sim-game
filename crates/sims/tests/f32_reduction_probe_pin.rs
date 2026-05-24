//! Behavior pin: sort-then-fold delivers byte-equal view storage
//! across same-seed reruns for an N-producer → 1-target f32 reduction.
//!
//! Topology (host-seeded):
//!   slot 0              = Target (creature_type = 1, alive = 1)
//!   slots 1..=N_PRODUCERS = Producer (creature_type = 0, alive = 1)
//!
//! Entity type ordinals (declaration-order alphabetical in the .sim):
//!   Producer = 0, Target = 1.
//!
//! Each Producer emits one Damaged event per tick targeting slot 0.
//! The damage_taken view folds all contributions into slot 0.
//! With N_PRODUCERS=50 and TICKS=100, expected damage_taken[0] = 5.0
//! (50 producers × 100 ticks × 0.1 amount_per_hit = 500 events × 0.1).
//!
//! DETERMINISM GOAL: byte-equal across two same-seed runs (no drift).

use sims::f32_reduction_probe::GeneratedRuntime;

const SEED: u64 = 0xC0FFEE1234;
const TICKS: u32 = 100;
const N_PRODUCERS: u32 = 50;
const N_TOTAL: u32 = N_PRODUCERS + 1; // slot 0 = Target

// Entity type ordinals (alphabetical by entity name in the .sim):
//   Producer = 0, Target = 1.
const CT_PRODUCER: u32 = 0;
const CT_TARGET: u32 = 1;

fn build_and_run(seed: u64) -> Vec<f32> {
    let mut state = match GeneratedRuntime::try_new(seed, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[f32_reduction_probe_pin] skipping: no wgpu adapter on host.");
            return vec![];
        }
    };

    // Seed: set alive=1 for all slots, creature_type = Target for slot 0,
    // Producer for slots 1..=N_PRODUCERS.
    let alive: Vec<u32> = vec![1u32; N_TOTAL as usize];
    let mut ct: Vec<u32> = vec![CT_PRODUCER; N_TOTAL as usize];
    ct[0] = CT_TARGET;

    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&ct),
    );

    for _ in 0..TICKS {
        state.step();
    }

    read_damage_taken(&mut state)
}

fn read_damage_taken(state: &mut GeneratedRuntime) -> Vec<f32> {
    let bytes = (state.agent_count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("f32_reduction_probe::damage_taken_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("f32_reduction_probe::damage_taken_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(
        &state.view_storage_damage_taken_primary_buf,
        0,
        &staging,
        0,
        bytes,
    );
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("damage_taken_staging map_async failed")
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

#[test]
fn f32_reduction_probe_byte_equal_same_seed() {
    let r1 = build_and_run(SEED);
    if r1.is_empty() {
        // No wgpu adapter — skip gracefully.
        return;
    }
    let r2 = build_and_run(SEED);

    // Byte-equal assertion: view storage must be identical across two
    // same-seed runs.  If this fails, the fold path has residual
    // non-determinism in the minimal N-to-1 fan-in shape — investigate
    // the sort-then-fold infrastructure before concluding it's benign.
    assert_eq!(
        r1, r2,
        "f32 reduction must be byte-equal across same-seed reruns. \
         r1[0]={:.6} r2[0]={:.6}",
        r1[0], r2[0],
    );

    // Sanity-check: Target slot accumulated the expected total.
    // The view fold snapshot is taken at start-of-tick, so each tick's
    // events are processed by the NEXT tick's fold pass.  Over TICKS
    // ticks the last tick's emits are never folded (no tick TICKS+1
    // runs), giving 50 producers × (TICKS-1) ticks × 0.1 = 495.0.
    // A small f32 ULP variance is tolerable here — the byte-equal
    // assertion above catches any run-to-run non-determinism.
    let expected = N_PRODUCERS as f32 * (TICKS as f32 - 1.0) * 0.1;
    assert!(
        (r1[0] - expected).abs() < 1.0,
        "damage_taken[0] = {:.3} (expected ≈ {expected:.1})",
        r1[0],
    );

    // Non-Target slots must have zero damage_taken.
    for (i, &v) in r1[1..].iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "damage_taken[{}] = {v} (expected 0.0 — non-Target slot)",
            i + 1,
        );
    }

    println!(
        "  f32_reduction_probe: damage_taken[0] = {:.3} (expected ≈ {:.1}), byte-equal PASS",
        r1[0], expected,
    );
}
