//! Criterion bench for apply_movement.
//!
//! Loads a fixture at each available scale and benches the ApplyMovementSystem
//! via its apply_inplace entry point. Backend enum is tested with both Scalar
//! and Simd arms — both currently call the same scalar kernel; the Simd arm
//! exists as the hook for future SIMD work.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use game::world_sim::apply::ApplyMovementSystem;
use game::world_sim::delta::MergedDeltas;
use game::world_sim::system::Backend;
use world_sim_bench::fixtures;

fn bench_movement(c: &mut Criterion) {
    for scale in ["2k"] {
        if !fixtures::exists(scale) {
            eprintln!("skip: no fixture for scale={scale}");
            continue;
        }
        let state = match fixtures::load(scale) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip scale={scale}: {e}");
                continue;
            }
        };

        // Build a minimal MergedDeltas so apply_movement has something to
        // chew on. Real-world load seeds ~1 force per alive entity; we
        // simulate that density by adding (0.1, 0.0) for every alive entity.
        let mut merged = MergedDeltas::default();
        for e in state.entities.iter().filter(|e| e.alive) {
            merged.forces_by_entity.insert(e.id, (0.1, 0.0));
        }

        let mut group = c.benchmark_group(format!("movement/{scale}"));
        group.sample_size(30);
        for backend in [Backend::Scalar, Backend::Simd] {
            let sys = ApplyMovementSystem::new(backend);
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{backend:?}")),
                &(sys, state.clone(), merged.clone()),
                |b, (sys, state, merged)| {
                    b.iter_batched(
                        || state.clone(),
                        |mut s| sys.apply_inplace(&mut s, merged),
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_movement);
criterion_main!(benches);
