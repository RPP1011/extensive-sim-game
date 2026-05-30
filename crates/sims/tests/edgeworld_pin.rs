//! edgeworld Phase 0 behavioral pin. Phase 0 = hunger + food +
//! forage/eat/starve/regrow. This file grows task-by-task; Task 1
//! only asserts the fixture compiles and the runtime constructs.

use sims::edgeworld::GeneratedRuntime;

const SEED: u64 = 0xED6E_0001;
const N_TOTAL: u32 = 4;

#[test]
fn edgeworld_runtime_constructs() {
    let state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skipping: no wgpu adapter on host.");
            return;
        }
    };
    // Constructing + dropping the runtime is the Task 1 assertion:
    // the fixture compiled and the GPU pipeline built.
    drop(state);
}

#[test]
fn edgeworld_hunger_rises() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skipping: no wgpu adapter.");
            return;
        }
    };
    let n = N_TOTAL as usize;
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&vec![0.0f32; n]));
    for _ in 0..10 {
        state.step();
    }
    let hunger = read_hunger(&mut state, n);
    println!("[edgeworld] hunger after 10 ticks: {hunger:?}");
    assert!(
        hunger[0] > 0.4 && hunger[0] < 0.6,
        "hunger should be ~0.5 after 10 ticks at 0.05/tick, got {}",
        hunger[0]
    );
}

#[test]
fn edgeworld_starvation_kills() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skipping: no wgpu adapter.");
            return;
        }
    };
    let n = N_TOTAL as usize;
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&vec![0.0f32; n]));
    state.gpu.queue.write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&vec![1.0f32; n])); // for fallback path
    // No food → everyone starves. 20 ticks to threshold; run 30.
    for _ in 0..30 {
        state.step();
    }
    let alive = read_alive(&mut state, n);
    let n_alive: u32 = alive.iter().sum();
    println!("[edgeworld] survivors alive after 30 starving ticks: {n_alive}");
    assert_eq!(n_alive, 0, "all survivors should have starved with no food");
}

/// Staging-buffer readback of `state.agent_alive_buf` as u32 — the
/// `read_hunger` helper's integer twin.
fn read_alive(state: &mut GeneratedRuntime, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edgeworld::alive_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("edgeworld::alive_readback"),
        },
    );
    let buf = state.agent_alive_buf.clone();
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

/// Staging-buffer readback of `state.agent_hunger_buf` as f32, modeled
/// on the `readback_u32`/`read_positions` helpers in
/// `detective_investigation_pin.rs`.
fn read_hunger(state: &mut GeneratedRuntime, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edgeworld::hunger_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("edgeworld::hunger_readback"),
        },
    );
    let buf = state.agent_hunger_buf.clone();
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
