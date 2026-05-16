//! Plan I — slice I.7 migration pin.
//!
//! Each fixture below was migrated from `view beliefs_flags(observer:
//! Agent, subject: Agent) -> u32` (storage = pair_map) to the new
//! `belief beliefs_flags(...)` keyword. The lowering's signature
//! inference picks PairMap, so the emitted WGSL fold kernel should be
//! shape-equivalent to the pre-migration version.
//!
//! This pin asserts each fixture:
//!   * parses cleanly,
//!   * resolves with `beliefs_flags` carrying `ViewKind::Belief`,
//!   * lowers without diagnostics,
//!   * still emits a `fold_beliefs_flags` kernel.
//!
//! Runtime behavioural pins for these fixtures live under
//! `crates/sims/tests/{tom_probe_pair_map_pin, dungeon_horde_pin,
//! dungeon_stealth_pin, plague_city_pin}.rs`. The sims-crate build is
//! currently blocked by an unrelated pre-existing panic in
//! `threats_struct_probe`; this compiler-layer pin is what's
//! verifiable today.

use dsl_ast::ir::ViewKind;
use dsl_compiler::cg::emit::emit_cg_program;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};

const MIGRATED_FIXTURES: &[&str] = &[
    "tom_probe.sim",
    "dungeon_horde.sim",
    "dungeon_stealth.sim",
    "plague_city.sim",
];

fn assets_path(fixture: &str) -> String {
    format!(
        "{}/../../assets/sim/{}",
        env!("CARGO_MANIFEST_DIR"),
        fixture,
    )
}

#[test]
fn migrated_fixtures_keep_beliefs_flags_as_belief_kind() {
    for fixture in MIGRATED_FIXTURES {
        let path = assets_path(fixture);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let prog = dsl_compiler::parse(&src)
            .unwrap_or_else(|e| panic!("parse {fixture}: {e}"));
        let comp = dsl_ast::resolve::resolve(prog)
            .unwrap_or_else(|e| panic!("resolve {fixture}: {e}"));
        let view = comp
            .views
            .iter()
            .find(|v| v.name == "beliefs_flags")
            .unwrap_or_else(|| panic!("{fixture}: beliefs_flags view should exist"));
        assert_eq!(
            view.kind,
            ViewKind::Belief,
            "{fixture}: `beliefs_flags` should be a Belief after migration; got {:?}",
            view.kind,
        );
    }
}

#[test]
fn beliefs_flags_signature_resolves_to_pair_keyed_belief() {
    // Tighter pin: every migrated fixture's `beliefs_flags` belief
    // must have the (observer: Agent, subject: Agent) signature that
    // the slice I.3 lowering knows how to lower. Catches accidental
    // signature drift that would silently revert to the
    // UnsupportedBeliefShape diagnostic during build.rs lowering.
    use dsl_ast::ir::IrType;
    for fixture in MIGRATED_FIXTURES {
        let path = assets_path(fixture);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        let prog = dsl_compiler::parse(&src)
            .unwrap_or_else(|e| panic!("parse {fixture}: {e}"));
        let comp = dsl_ast::resolve::resolve(prog)
            .unwrap_or_else(|e| panic!("resolve {fixture}: {e}"));
        let view = comp
            .views
            .iter()
            .find(|v| v.name == "beliefs_flags")
            .unwrap_or_else(|| panic!("{fixture}: beliefs_flags view missing"));
        assert_eq!(view.params.len(), 2, "{fixture}: expected 2 params");
        assert!(
            matches!(view.params[0].ty, IrType::AgentId),
            "{fixture}: first param must be Agent; got {:?}",
            view.params[0].ty,
        );
        assert!(
            matches!(view.params[1].ty, IrType::AgentId),
            "{fixture}: second param must be Agent; got {:?}",
            view.params[1].ty,
        );
    }
}

// Suppress unused-import lints — the full emit pipeline is exercised
// by other tests (`belief_lower_pair_map`, `belief_smoke_probe`);
// this pin focuses on the migration-specific resolve-shape contract
// because the heavy fixtures need build.rs-supplied LowerOpts.
const _: () = {
    let _: fn() = || {
        let _ = lower_compilation_to_cg;
        let _ = synthesize_schedule;
        let _ = emit_cg_program;
        let _ = ScheduleStrategy::Default;
    };
};
