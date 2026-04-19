use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

fn bench_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_throughput");
    group.sample_size(10); // 1000-tick iterations are coarse; 10 samples is plenty.
    for &n in &[10usize, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut state = SimState::new((n as u32) + 10, 42);
                let mut scratch = SimScratch::new(state.agent_cap() as usize);
                let mut events = EventRing::with_cap(100_000);
                let cascade = CascadeRegistry::new();
                for i in 0..n {
                    let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
                    state.spawn_agent(AgentSpawn {
                        creature_type: CreatureType::Human,
                        pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
                        hp: 100.0,
                    });
                }
                for _ in 0..1000 {
                    step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
                }
                black_box(&state);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tick);
criterion_main!(benches);
