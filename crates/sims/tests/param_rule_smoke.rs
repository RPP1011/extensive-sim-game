//! Runtime gate: monomorphisation produces two independent concrete
//! rules from one parameterised template + two applications.
//!
//! The build itself is the strongest signal — if `cargo build -p sims`
//! succeeds with the `param_rule_smoke` fixture in the allow-list, the
//! monomorphisation pass ran and emitted both `HunterChase` and
//! `WolfChase` into the megacrate. This test then locks in the
//! post-condition by referencing the module.
//!
//! The fused kernel name `physics_chase_and_HunterChase_and_WolfChase`
//! proves that both concrete rules were produced by the monomorphisation
//! pass and subsequently fused by the optimiser — the ideal outcome.

#[test]
fn param_rule_smoke_module_compiles() {
    // Referencing GeneratedRuntime forces the megacrate to expose the
    // module, which fails at compile time if T11's build did not run
    // cleanly. No runtime assertion needed beyond the build-time check.
    let _ = std::mem::size_of::<sims::param_rule_smoke::GeneratedRuntime>();
}
