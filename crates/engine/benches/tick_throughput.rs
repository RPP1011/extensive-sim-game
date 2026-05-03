use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine::mask::{MaskBuffer, MicroKind, TargetMask};
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend, UtilityBackend};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

fn spawn_ring(state: &mut SimState, n: usize) {
    for i in 0..n {
        let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }
}

fn run<B: PolicyBackend>(n: usize, ticks: usize, backend: &B) {
    let mut state = SimState::new((n as u32) + 10, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(100_000);
    let cascade = CascadeRegistry::new();
    spawn_ring(&mut state, n);
    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, backend, &cascade);
    }
    black_box(&state);
}

/// Mixed-action policy that exercises Plan 1's full vocabulary — Eat / Drink /
/// Rest / MoveToward / Communicate / Hold, fan-out based on `id % 5` with a
/// starvation override that forces Eat when hunger < 0.3. Mirrors the policy
/// shape from `tests/acceptance_mixed_actions.rs`.
struct MixedPolicy;

impl PolicyBackend for MixedPolicy {
    fn evaluate(&self, state: &SimState, _mask: &MaskBuffer, _target_mask: &TargetMask, out: &mut Vec<Action>) {
        let tick = state.tick;
        for id in state.agents_alive() {
            let pick = id.raw() % 5;
            let hunger = state.agent_hunger(id).unwrap_or(1.0);
            let action = if hunger < 0.3 {
                Action {
                    agent: id,
                    kind: ActionKind::Micro {
                        kind:   MicroKind::Eat,
                        target: MicroTarget::None,
                    },
                }
            } else if pick == 0 {
                Action {
                    agent: id,
                    kind: ActionKind::Micro {
                        kind:   MicroKind::Drink,
                        target: MicroTarget::None,
                    },
                }
            } else if pick == 1 {
                Action {
                    agent: id,
                    kind: ActionKind::Micro {
                        kind:   MicroKind::Rest,
                        target: MicroTarget::None,
                    },
                }
            } else if pick == 2 {
                Action {
                    agent: id,
                    kind: ActionKind::Micro {
                        kind:   MicroKind::MoveToward,
                        target: MicroTarget::Position(Vec3::new(0.0, 0.0, 10.0)),
                    },
                }
            } else if pick == 3 {
                let target = engine::ids::AgentId::new(1).unwrap();
                if target != id && state.agent_alive(target) {
                    Action {
                        agent: id,
                        kind: ActionKind::Micro {
                            kind:   MicroKind::Communicate,
                            target: MicroTarget::Agent(target),
                        },
                    }
                } else {
                    Action::hold(id)
                }
            } else {
                Action::hold(id)
            };
            out.push(action);
        }
        let _ = tick;
    }
}

fn bench_utility(c: &mut Criterion) {
    let mut group = c.benchmark_group("utility_backend");
    group.sample_size(10);
    for &n in &[10usize, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| run(n, 1000, &UtilityBackend));
        });
    }
    group.finish();
}

fn bench_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_actions");
    group.sample_size(10);
    for &n in &[10usize, 100, 500] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| run(n, 1000, &MixedPolicy));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_utility, bench_mixed);
criterion_main!(benches);
