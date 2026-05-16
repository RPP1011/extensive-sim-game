//! maze_explorer perf comparison — bare random walk (no belief) vs
//! belief-augmented (event-emitting + fold). Both fixtures use the
//! same RNG keying so they take the same path through the maze
//! (tick_found = 198 either way). The interesting axis is per-tick
//! wall-clock cost — the visited variant pays for one extra event
//! emit per move + one fold_visited kernel dispatch per tick.
//!
//! Question this bench answers: "what does it cost to thread a
//! belief layer through a per-tick rule?" Answer is encoded in the
//! ratio of mean ms/tick across the two fixtures.

#![allow(non_snake_case)]

const SEED: u64 = 0xB1_BB_BB_BB_BB_BB_BB_77u64;
const N_AGENTS: u32 = 1;
const WARMUP_TICKS: usize = 5;
const MEASURE_TICKS: usize = 50;

fn run_bare() -> Option<(f64, f64)> {
    use sims::maze_explorer::GeneratedRuntime;
    let mut state = GeneratedRuntime::try_new(SEED, N_AGENTS)?;
    let alive: [u32; 1] = [1];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
    for _ in 0..WARMUP_TICKS {
        state.step();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(MEASURE_TICKS);
    let total_start = std::time::Instant::now();
    for _ in 0..MEASURE_TICKS {
        let t = std::time::Instant::now();
        state.step();
        let _ = sync_pulse(&state.gpu);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = total_start.elapsed().as_secs_f64() * 1000.0 / MEASURE_TICKS as f64;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = samples[((samples.len() * 95) / 100).min(samples.len() - 1)];
    Some((mean, p95))
}

fn run_belief() -> Option<(f64, f64)> {
    use sims::maze_explorer_visited::GeneratedRuntime;
    let mut state = GeneratedRuntime::try_new(SEED, N_AGENTS)?;
    let alive: [u32; 1] = [1];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
    for _ in 0..WARMUP_TICKS {
        state.step();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(MEASURE_TICKS);
    let total_start = std::time::Instant::now();
    for _ in 0..MEASURE_TICKS {
        let t = std::time::Instant::now();
        state.step();
        let _ = sync_pulse(&state.gpu);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = total_start.elapsed().as_secs_f64() * 1000.0 / MEASURE_TICKS as f64;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = samples[((samples.len() * 95) / 100).min(samples.len() - 1)];
    Some((mean, p95))
}

/// Sync-poll the GPU so the per-tick timer reflects actual
/// completion, not just queue-submit. Reading a 4-byte staging
/// buffer is the cheapest sync we can do.
fn sync_pulse(gpu: &engine::GpuContext) -> u32 {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("maze_perf::sync"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("maze_perf::sync_pulse"),
        });
    gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..4);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let v = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[0]
    };
    staging.unmap();
    v
}

#[test]
fn maze_perf_bare_vs_belief() {
    let bare = match run_bare() {
        Some(x) => x,
        None => {
            eprintln!("[maze_perf] skipping: no wgpu adapter");
            return;
        }
    };
    let belief = match run_belief() {
        Some(x) => x,
        None => {
            eprintln!("[maze_perf] skipping: no wgpu adapter");
            return;
        }
    };
    let overhead_pct = ((belief.0 - bare.0) / bare.0) * 100.0;
    println!("==== maze_explorer perf bench ====");
    println!(
        "  Bench: {N_AGENTS} agent, warmup={WARMUP_TICKS} ticks, \
         measured={MEASURE_TICKS} ticks per fixture."
    );
    println!("  Fixture                  mean ms/tick    p95 ms/tick");
    println!("  bare (random walk)       {:>9.3}      {:>9.3}", bare.0, bare.1);
    println!("  belief (event + fold)    {:>9.3}      {:>9.3}", belief.0, belief.1);
    println!("==================================");
    println!(
        "  Belief layer overhead: {overhead_pct:+.1}% mean ms/tick \
         (= one event-emit + one fold_visited dispatch per tick)."
    );
}
