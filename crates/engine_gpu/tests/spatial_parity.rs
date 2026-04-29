// T16-broken: this test references hand-written kernel modules
// (mask, scoring, physics, apply_actions, movement, spatial_gpu,
// alive_bitmap, cascade, cascade_resident) that were retired in
// commit 4474566c when the SCHEDULE-driven dispatcher in
// `engine_gpu_rules` became authoritative. The test source is
// preserved verbatim below the cfg gate so the SCHEDULE-loop port
// (follow-up: gpu-feature-repair plan) has a reference for what
// behaviour each test asserted.
//
// Equivalent to `#[ignore = "broken by T16 hand-written-kernel
// deletion; needs SCHEDULE-loop port (follow-up)"]` on every
// `#[test]` below — but applied at file scope because the test
// bodies do not compile against the post-T16 surface.
#![cfg(any())]

//! Phase 5 — CPU/GPU spatial hash parity harness.
//!
//! Gated on `--features gpu` so the CPU-only build skips wgpu. Run with
//! `cargo test -p engine_gpu --features gpu --test spatial_parity`.
//!
//! ## What's tested
//!
//! For each agent slot in three density fixtures, assert byte-equality
//! on all three GPU query primitives against CPU references:
//!
//!   * `within_radius` — alive agents in Euclidean range
//!   * `nearest_hostile_to` — closest alive hostile, ties on raw id
//!   * `nearby_kin` — alive same-species neighbours
//!
//! ## Densities
//!
//! | Name     | Agents | World         | Radius | Expected |
//! |----------|--------|---------------|--------|----------|
//! | sparse   | 20     | 100 × 100     | 5      | ~1/query |
//! | medium   | 200    | 50 × 50       | 5      | ~8/query |
//! | dense    | 2000   | 20 × 20       | 5      | ~300/query (truncates at K=32) |
//!
//! Dense exercises the "oversized CPU result truncates to top-K-by-
//! lowest-raw-id on GPU" parity contract — the GPU's `truncated` flag
//! asserts and its result must equal the first K of the CPU's fully-
//! sorted list.

#![cfg(feature = "gpu")]

use engine_data::entities::CreatureType;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::spatial_gpu::{cpu_reference, GpuSpatialHash, NO_HOSTILE};
use glam::Vec3;

const AGENT_CAP_SPARSE: u32 = 32;
const AGENT_CAP_MEDIUM: u32 = 256;
const AGENT_CAP_DENSE: u32 = 2048;
const QUERY_RADIUS: f32 = 5.0;

/// Minimal linear-congruential generator for deterministic fixture
/// positions. One call site, one seed per fixture — avoids pulling a
/// full RNG crate into engine_gpu's dev-dependency surface.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed | 1) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32_01(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

fn spawn_fixture(n_agents: u32, world_side: f32, seed: u64, cap: u32) -> SimState {
    let mut state = SimState::new(cap, seed);
    let mut rng = Lcg::new(seed ^ 0xC0FFEE);
    for i in 0..n_agents {
        // Rotate creature types so we get a mix of hostile/kin
        // relationships: wolves + humans are mutually hostile, deer are
        // kin with deer, dragons are hostile to all.
        let ct = match i % 4 {
            0 => CreatureType::Human,
            1 => CreatureType::Wolf,
            2 => CreatureType::Deer,
            _ => CreatureType::Dragon,
        };
        let px = rng.next_f32_01() * world_side;
        let py = rng.next_f32_01() * world_side;
        state
            .spawn_agent(AgentSpawn {
                creature_type: ct,
                pos: Vec3::new(px, py, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .expect("spawn");
    }
    state
}

fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue, String) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("adapter");
        let label = format!("{:?}", adapter.get_info().backend);
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("spatial_parity::device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");
        (device, queue, label)
    })
}

/// Run full parity check (within_radius + kin + nearest_hostile) for a
/// density fixture. Reports GPU vs CPU rebuild+query timings for each
/// density as a side benefit.
fn check_density(name: &str, n_agents: u32, world_side: f32, cap: u32) {
    let state = spawn_fixture(n_agents, world_side, 0xDEAD_BEEF_1234_5678, cap);
    let (device, queue, backend) = gpu_device_queue();

    let mut hash = GpuSpatialHash::new(&device).expect("spatial hash init");
    // Warm-up — the first dispatch spins up wgpu lazy resources. Avoid
    // polluting the reported time with that.
    let _ = hash.rebuild_and_query(&device, &queue, &state, QUERY_RADIUS).expect("warmup");

    // Per-query perf: run 100 rebuild+query cycles and average the
    // time. 100 calls lets `cargo test` report a steady number without
    // exercising a full benchmark harness.
    const PERF_ITERS: u32 = 100;
    let t0 = std::time::Instant::now();
    let mut gpu_results = hash
        .rebuild_and_query(&device, &queue, &state, QUERY_RADIUS)
        .expect("rebuild+query");
    for _ in 1..PERF_ITERS {
        gpu_results = hash
            .rebuild_and_query(&device, &queue, &state, QUERY_RADIUS)
            .expect("rebuild+query");
    }
    let gpu_total = t0.elapsed();
    let gpu_per = gpu_total / PERF_ITERS;

    let t0 = std::time::Instant::now();
    let cpu_ref = cpu_reference(&state, QUERY_RADIUS);
    let cpu_elapsed = t0.elapsed();

    eprintln!(
        "[{name}] backend={backend} agents={n_agents} world={world_side}x{world_side} radius={QUERY_RADIUS}"
    );
    eprintln!(
        "[{name}] rebuild+query gpu_avg={gpu_per:?} (over {PERF_ITERS})  cpu_reference={cpu_elapsed:?}"
    );

    assert_eq!(gpu_results.within_radius.len(), cpu_ref.within_radius.len());
    assert_eq!(gpu_results.nearby_kin.len(), cpu_ref.nearby_kin.len());
    assert_eq!(gpu_results.nearest_hostile.len(), cpu_ref.nearest_hostile.len());

    let k_cap = engine_gpu::spatial_gpu::K as usize;
    let mut within_sum: u64 = 0;
    let mut within_max: u32 = 0;
    let mut kin_sum: u64 = 0;
    let mut kin_max: u32 = 0;
    let mut within_trunc_slots: u32 = 0;
    let mut kin_trunc_slots: u32 = 0;
    let mut hostile_hits: u32 = 0;

    for slot in 0..cap as usize {
        let gpu_w = gpu_results.within_radius[slot];
        let cpu_w = &cpu_ref.within_radius[slot];
        let w_expected: &[u32] = if cpu_w.len() > k_cap {
            within_trunc_slots += 1;
            &cpu_w[..k_cap]
        } else {
            cpu_w.as_slice()
        };
        let w_trunc_ok = (cpu_w.len() > k_cap) == (gpu_w.truncated != 0);
        assert!(
            w_trunc_ok,
            "[{name}] within_radius slot {slot}: trunc flag mismatch cpu_len={} gpu_trunc={}",
            cpu_w.len(), gpu_w.truncated,
        );
        if gpu_w.as_slice() != w_expected {
            panic!(
                "[{name}] within_radius slot {slot}: GPU differs\n\
                 gpu = {:?}\n\
                 cpu[:K] = {:?}\n\
                 cpu_full_len = {}",
                gpu_w.as_slice(), w_expected, cpu_w.len(),
            );
        }
        within_sum += gpu_w.as_slice().len() as u64;
        if gpu_w.count > within_max { within_max = gpu_w.count; }

        // Kin parity.
        let gpu_k = gpu_results.nearby_kin[slot];
        let cpu_k = &cpu_ref.nearby_kin[slot];
        let k_expected: &[u32] = if cpu_k.len() > k_cap {
            kin_trunc_slots += 1;
            &cpu_k[..k_cap]
        } else {
            cpu_k.as_slice()
        };
        let k_trunc_ok = (cpu_k.len() > k_cap) == (gpu_k.truncated != 0);
        assert!(
            k_trunc_ok,
            "[{name}] kin slot {slot}: trunc flag mismatch cpu_len={} gpu_trunc={}",
            cpu_k.len(), gpu_k.truncated,
        );
        if gpu_k.as_slice() != k_expected {
            panic!(
                "[{name}] kin slot {slot}: GPU differs\n\
                 gpu = {:?}\n\
                 cpu[:K] = {:?}\n\
                 cpu_full_len = {}",
                gpu_k.as_slice(), k_expected, cpu_k.len(),
            );
        }
        kin_sum += gpu_k.as_slice().len() as u64;
        if gpu_k.count > kin_max { kin_max = gpu_k.count; }

        // Nearest hostile parity.
        let gpu_h = gpu_results.nearest_hostile[slot];
        let cpu_h = cpu_ref.nearest_hostile[slot];
        assert_eq!(
            gpu_h, cpu_h,
            "[{name}] nearest_hostile slot {slot}: gpu={} cpu={}",
            gpu_h, cpu_h,
        );
        if gpu_h != NO_HOSTILE {
            hostile_hits += 1;
        }
    }
    eprintln!(
        "[{name}] within: sum={within_sum} max={within_max} trunc_slots={within_trunc_slots}"
    );
    eprintln!(
        "[{name}] kin:    sum={kin_sum} max={kin_max} trunc_slots={kin_trunc_slots}"
    );
    eprintln!(
        "[{name}] nearest_hostile: hit_slots={hostile_hits}"
    );
}

#[test]
fn spatial_parity_sparse() {
    check_density("sparse", 20, 100.0, AGENT_CAP_SPARSE);
}

#[test]
fn spatial_parity_medium() {
    check_density("medium", 200, 50.0, AGENT_CAP_MEDIUM);
}

#[test]
fn spatial_parity_dense() {
    check_density("dense", 2000, 20.0, AGENT_CAP_DENSE);
}

/// Constant sanity check — `NO_HOSTILE` is the u32 sentinel the
/// nearest-hostile kernel returns when nothing's in range.
#[test]
fn no_hostile_sentinel_is_u32_max() {
    assert_eq!(NO_HOSTILE, u32::MAX);
}
