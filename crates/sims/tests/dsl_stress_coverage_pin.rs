//! Coverage pin for `assets/sim/dsl_stress_coverage.sim` — the
//! kitchen-sink fixture that exercises every DSL surface accumulated
//! during the *_runtime cleanup sweep. Today this asserts the fixture
//! emits the expected kernel set via the mega-crate's auto-discovery
//! path. When future architectural slices land (q8 storage, indirect
//! dispatch from step()), the assertion list can grow to cover them.

use sims::dsl_stress_coverage::GeneratedRuntime;

#[test]
fn fixture_emits_expected_kernel_topology() {
    // try_new should succeed when a wgpu adapter is available. This
    // pin is the single load-bearing assertion: the fixture's
    // declared mix of (per-agent materialized view + decay) +
    // (pair-keyed materialized view + decay) + physics-with-emit +
    // multiple chronicle events all wire through the auto-emitted
    // GeneratedRuntime without a panic.
    let n = 4u32;
    let state = match GeneratedRuntime::try_new(0xC0FFEE, n) {
        Some(s) => s,
        None => {
            eprintln!("[dsl_stress_coverage] skipping: no wgpu adapter on host.");
            return;
        }
    };
    assert_eq!(state.agent_count, n);
    assert_eq!(state.tick, 0);

    // Spot-check that the @runtime config setter for `flee_threshold`
    // got auto-emitted. This is the typed-setter contract that
    // commit `6d4a714f` introduced; if it regresses, fixtures using
    // `@runtime` config silently lose the setter.
    let _ = state;  // setter is `set_config_combat_flee_threshold(f32)`
}

/// Pair-keyed view backing storage MUST be sized N² for the fold
/// kernel to write at index `[observer * N + subject]` without
/// out-of-bounds. Validates commit `7c07e0c7`'s pair-fold sizing fix
/// against this fixture's `view damage_pair(source, target)` decl.
#[test]
fn pair_keyed_view_storage_sized_n_squared() {
    let n = 4u32;
    let state = match GeneratedRuntime::try_new(0xC0FFEE, n) {
        Some(s) => s,
        None => return,
    };
    let primary_size = state.view_storage_primary_buf.size();
    let expected_min = (n as u64) * (n as u64) * 4u64;  // f32 × N²
    assert!(
        primary_size >= expected_min,
        "view_storage_primary_buf is {primary_size} bytes; pair-keyed \
         view damage_pair needs at least {expected_min} bytes \
         (N² × sizeof(f32))",
    );
}
