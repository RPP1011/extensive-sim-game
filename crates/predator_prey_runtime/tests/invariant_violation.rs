//! Stress test for the compiler-emitted `invariants` module.
//!
//! Closes the silent-drop gap on `invariant <name>(...) @<mode> { ... }`
//! declarations: `assets/sim/predator_prey_min.sim`'s
//! `bounded_kill_count(a: Agent) @debug_only { kill_count(a) < 1000.0 }`
//! used to parse + resolve and then go nowhere — no host-side check
//! existed. With `cg::emit::invariants::synthesize_invariants`, the
//! build script now wraps a compiler-generated `check_bounded_kill_count`
//! fn into the runtime crate. This test proves the artifact actually
//! fires by feeding it a deliberately-violating storage slice and
//! asserting the typed `Violation` reports come back with the right
//! payload (slot index, value, bound, mode).
//!
//! The check is scalar and pure (no GPU coupling) so the test runs on
//! the CI bench machine without an adapter — the GPU path is exercised
//! end-to-end by the runtime's `examples/smoke.rs`.

use predator_prey_runtime::invariants::{check_bounded_kill_count, Mode, Violation};

#[test]
fn invariant_fires_on_violating_storage() {
    // Mix of healthy and violating slots. The bound from the .sim
    // declaration is 1000.0; anything `>= 1000.0` violates.
    let storage = vec![
        0.0,        // slot 0 — healthy (Hare slot in the 50/50 init)
        500.0,      // slot 1 — healthy Wolf
        999.999,    // slot 2 — healthy boundary
        1000.0,     // slot 3 — violation (the predicate is `<`, not `<=`)
        2500.0,     // slot 4 — violation (saturation)
        f32::INFINITY, // slot 5 — violation (degenerate fold)
    ];

    let violations = check_bounded_kill_count(&storage);

    assert_eq!(
        violations.len(),
        3,
        "expected 3 violations (slots 3, 4, 5); got {}: {:?}",
        violations.len(),
        violations,
    );

    // Order is slot-ascending — the check iterates the slice in index
    // order and pushes one Violation per failing slot.
    assert_eq!(
        violations[0],
        Violation {
            invariant: "bounded_kill_count",
            agent:     3,
            value:     1000.0,
            bound:     1000.0,
            mode:      Mode::DebugOnly,
        },
    );
    assert_eq!(violations[1].agent, 4);
    assert_eq!(violations[1].value, 2500.0);
    assert_eq!(violations[2].agent, 5);
    assert!(
        violations[2].value.is_infinite(),
        "slot 5 should preserve INF: {:?}",
        violations[2].value,
    );

    // Every violation carries the source-declared mode (`@debug_only`).
    for v in &violations {
        assert_eq!(v.mode, Mode::DebugOnly);
        assert_eq!(v.invariant, "bounded_kill_count");
        assert_eq!(v.bound, 1000.0);
    }
}

#[test]
fn invariant_returns_empty_on_healthy_storage() {
    // Steady-state: every slot well under the bound. The expected
    // return is the empty vec — no allocation noise, no false positives.
    let storage = vec![0.0; 32];
    let violations = check_bounded_kill_count(&storage);
    assert!(
        violations.is_empty(),
        "no violations expected on all-zero storage; got {:?}",
        violations,
    );

    let storage = vec![999.0; 32];
    let violations = check_bounded_kill_count(&storage);
    assert!(
        violations.is_empty(),
        "no violations expected just below the bound; got {:?}",
        violations,
    );
}

#[test]
fn invariant_handles_empty_storage_cleanly() {
    // Defense in depth: a degenerate empty readback (no view storage
    // ever allocated) must not panic — it's a zero-iteration loop.
    let violations = check_bounded_kill_count(&[]);
    assert!(violations.is_empty());
}
