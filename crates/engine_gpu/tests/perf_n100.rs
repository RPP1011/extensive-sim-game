//! Ad-hoc N=100 perf measurement for the CPU↔GPU crossover question.
//! Scratch file — not a regression test. Delete after measuring.

#![cfg(feature = "gpu")]

use std::time::{Duration, Instant};

use engine::backend::{CpuBackend, SimBackend};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

const SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;
const N_AGENTS: u32 = 1000;
const AGENT_CAP: u32 = N_AGENTS + 8;
const TICKS: u32 = 50;
const EVENT_RING_CAP: usize = 1 << 20;

fn spawn_n(n: u32) -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    // 40 humans, 40 wolves, 20 deer — scaled version of showcase ratio.
    let humans = (n * 40 / 100).max(1);
    let wolves = (n * 40 / 100).max(1);
    let deer = n - humans - wolves;
    // Dense grid so agents actually engage. Cluster diameter scales
    // with sqrt(N) to keep density roughly constant at ~1 agent per
    // 4 unit² (close enough for engagement_range=2.0 to fire).
    let side = ((humans as f32).sqrt() + 1.0) as u32;
    let mut idx = 0u32;
    for i in 0..humans {
        let r = (i / side) as f32 * 1.2;
        let c = (i % side) as f32 * 1.2;
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(5.0 + c, 5.0 + r, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .unwrap();
        idx += 1;
    }
    let side_w = ((wolves as f32).sqrt() + 1.0) as u32;
    for i in 0..wolves {
        let r = (i / side_w) as f32 * 1.2;
        let c = (i % side_w) as f32 * 1.2;
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(-5.0 - c, -5.0 - r, 0.0),
                hp: 80.0,
                ..Default::default()
            })
            .unwrap();
    }
    let side_d = ((deer as f32).sqrt() + 1.0) as u32;
    for i in 0..deer {
        let r = (i / side_d) as f32 * 1.5;
        let c = (i % side_d) as f32 * 1.5;
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Deer,
                pos: Vec3::new(c - 5.0, r - 5.0, 0.0),
                hp: 60.0,
                ..Default::default()
            })
            .unwrap();
    }
    let _ = idx;
    state
}

#[test]
fn perf_n100() {
    // CPU
    let mut cpu_backend = CpuBackend;
    let mut state = spawn_n(N_AGENTS);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    let t0 = Instant::now();
    for _ in 0..TICKS {
        cpu_backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    let cpu_total: Duration = t0.elapsed();
    let alive_cpu = state.agents_alive().count();

    // GPU
    let mut gpu_backend = GpuBackend::new().expect("gpu init");
    gpu_backend.set_skip_scoring_sidecar(true);
    let mut state = spawn_n(N_AGENTS);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);

    let t_first = Instant::now();
    gpu_backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    let first_tick = t_first.elapsed();

    let t_rest = Instant::now();
    let mut iter_counts: Vec<u32> = Vec::new();
    for _ in 1..TICKS {
        gpu_backend.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
        if let Some(n) = gpu_backend.last_cascade_iterations() {
            iter_counts.push(n);
        }
    }
    let rest = t_rest.elapsed();
    let avg_iters: f64 = if iter_counts.is_empty() {
        0.0
    } else {
        iter_counts.iter().copied().map(|n| n as f64).sum::<f64>() / iter_counts.len() as f64
    };
    eprintln!("avg cascade iterations/tick: {avg_iters:.2}");
    eprintln!("GPU backend label: {}", gpu_backend.backend_label());
    let alive_gpu = state.agents_alive().count();

    let cpu_per = cpu_total.as_secs_f64() * 1e6 / TICKS as f64;
    let gpu_steady_per = rest.as_secs_f64() * 1e6 / (TICKS - 1) as f64;

    eprintln!("=== N=100 perf ===");
    eprintln!(
        "CPU:        {:.2} ms total ({:.1} µs/tick)  alive={}",
        cpu_total.as_secs_f64() * 1e3,
        cpu_per,
        alive_cpu,
    );
    eprintln!(
        "GPU first:  {:.2} ms  (init + shader compile + 1 tick)",
        first_tick.as_secs_f64() * 1e3,
    );
    eprintln!(
        "GPU steady: {:.2} ms total ({:.1} µs/tick over {} ticks)  alive={}",
        rest.as_secs_f64() * 1e3,
        gpu_steady_per,
        TICKS - 1,
        alive_gpu,
    );
    eprintln!("ratio (steady): GPU is {:.2}× CPU", gpu_steady_per / cpu_per);
}
