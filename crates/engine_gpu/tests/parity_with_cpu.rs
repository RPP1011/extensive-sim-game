//! End-to-end CPU vs GPU parity gate. **The most important single
//! test in `engine_gpu`** — catches regressions in any of the
//! Stream-B-landed kernels via byte-equality on per-agent
//! fingerprints after N ticks of `step_batch`.
//!
//! Today (commit dd21f9b9), `step_batch` runs the SCHEDULE-loop
//! dispatcher then CPU-forwards via `engine_rules::step::step`. The
//! CPU forward is authoritative until each emitted WGSL body lands
//! AND parity verifies it. The CPU side of this test runs the same
//! `engine_rules::step::step` directly, so today the test is
//! tautological for state mutation — but it gates encoder regressions
//! (e.g., a kernel's `bind()` lifetime mismatch panicking the
//! SCHEDULE loop), and as Stream B advances + the CPU forward
//! retires, it becomes the gate that proves byte equality.

mod common;

use common::{assert_cpu_gpu_parity, smoke_fixture_n4};
use engine::cascade::CascadeRegistry;
use engine_data::events::Event;
use engine_rules::views::ViewRegistry;

#[test]
fn parity_with_cpu_n4_t1() {
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();
    assert_cpu_gpu_parity(smoke_fixture_n4, &policy, &cascade, 1);
}

#[test]
fn parity_with_cpu_n4_t10() {
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();
    assert_cpu_gpu_parity(smoke_fixture_n4, &policy, &cascade, 10);
}

#[test]
fn parity_with_cpu_n4_t100() {
    let policy = engine::policy::utility::UtilityBackend;
    let cascade = CascadeRegistry::<Event, ViewRegistry>::new();
    assert_cpu_gpu_parity(smoke_fixture_n4, &policy, &cascade, 100);
}
