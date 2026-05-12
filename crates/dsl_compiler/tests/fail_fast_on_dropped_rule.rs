//! Gap plague_city#P-C — fail-fast on silently-dropped physics rules.
//!
//! When a `.sim`'s physics rule body fails to lower (e.g. references
//! a nonexistent agent field), the legacy build_helper logged the
//! diagnostic as a `cargo:warning=[<fixture> lower diag] ...` and
//! continued — the rule disappeared from the schedule but the build
//! succeeded overall. A fixture author had no compile-time signal
//! that 9 of 12 emitted kernels were missing.
//!
//! `dsl_compiler::build_helper::check_required_rules` is the env-gated
//! diagnostic-promotion helper that fixes this. With the env var
//! `SIM_REQUIRE_ALL_RULES` truthy, any non-empty diagnostic vector
//! returns `Err(message)` (which `emit_into` panics on). With the env
//! var unset/falsy, the helper returns `Ok(())` regardless — the
//! historical warn-and-drop behaviour is preserved.
//!
//! These tests pin both behaviours by:
//!   1. Synthesising a minimal `.sim` whose physics body references
//!      a nonexistent agent field (`agents.nonexistent_field(self)`).
//!      This is exactly the shape that triggered Gap P-C in
//!      plague_city.sim (`agents.infected(...)`).
//!   2. Running it through `parse → resolve → lower` — assertions
//!      confirm the lower step DOES produce diagnostics (otherwise
//!      the test would silently regress).
//!   3. Calling `check_required_rules` with both env-var states and
//!      asserting the gate behaviour.
//!
//! `serial_test` is not used; the env var is set / cleared per-test
//! around the call site, and the helper reads it via `env::var` at
//! call time. Tests are still safe under cargo's default parallel
//! test runner because the `.sim` synthesis + lower step is pure;
//! only the env-var read window needs serialization. Each test's
//! `set_var` / `remove_var` pair brackets a single synchronous call.

use std::sync::Mutex;

use dsl_compiler::build_helper::{check_required_rules, REQUIRE_ALL_RULES_ENV};

// Serialise env-var manipulation across tests in this binary.
// `cargo test` runs tests on multiple threads by default; setting a
// process-global env var in one thread races other threads' reads.
// One mutex held across set/call/unset keeps the env-var window
// single-threaded.
fn env_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

/// Minimal `.sim` source whose `Tick` physics rule body references
/// `agents.nonexistent_field(self)` — a field that is NOT in
/// `AgentFieldId`'s closed enum. Lowering this body produces a
/// `lower diag` that today drops the rule from the schedule.
///
/// The rest of the .sim is the smallest legal preamble (one event,
/// one entity, one config block) the resolve / schedule passes need
/// to walk without their own diagnostics.
const BAD_SIM_SRC: &str = r#"
event Tick { }

@replayable @gpu_amenable
event Logged {
  who: AgentId,
}

entity Citizen : Agent {
  pos: vec3,
  vel: vec3,
}

config probe {
  marker: f32 = 1.0,
}

physics ReadsBadField @phase(per_agent) {
  on Tick {} where (self.alive) {
    let v = agents.nonexistent_field(self);
    if (v > 0.0) {
      emit Logged { who: self }
    }
  }
}
"#;

/// Lower the bad-field `.sim` source and return the diagnostics the
/// driver produced. Asserts the lower step DID produce at least one
/// diagnostic — if a future driver refactor accepts the bad field,
/// this scaffold needs to switch to a different malformed shape so
/// the test stays meaningful.
fn lower_bad_sim_and_collect_diagnostics() -> Vec<String> {
    let program =
        dsl_compiler::parse(BAD_SIM_SRC).expect("parse synthetic bad-field .sim");
    let comp = dsl_ast::resolve::resolve(program)
        .expect("resolve synthetic bad-field .sim");
    let outcome = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp);
    match outcome {
        Ok(_) => panic!(
            "synthetic .sim with `agents.nonexistent_field(self)` lowered \
             cleanly — the test fixture needs a different malformed shape \
             so the env-gate semantics stay testable",
        ),
        Err(o) => o.diagnostics.iter().map(|d| format!("{d}")).collect(),
    }
}

#[test]
fn lower_diagnostics_warn_and_continue_when_env_unset() {
    let _g = env_lock().lock().unwrap_or_else(|p| p.into_inner());
    // SAFETY: this test holds env_lock across the env-var window.
    unsafe {
        std::env::remove_var(REQUIRE_ALL_RULES_ENV);
    }

    let diags = lower_bad_sim_and_collect_diagnostics();
    assert!(
        !diags.is_empty(),
        "synthetic bad-field .sim should produce >=1 lower diagnostic; \
         got an empty Vec — the env-gate test loses its load-bearing \
         input",
    );

    let result = check_required_rules("synthetic_bad_field", &diags);
    assert!(
        result.is_ok(),
        "default behaviour: env unset must continue past diagnostics \
         (warn-and-drop). Got Err: {:?}",
        result.err(),
    );
}

#[test]
fn lower_diagnostics_promote_to_error_when_env_set() {
    let _g = env_lock().lock().unwrap_or_else(|p| p.into_inner());
    // SAFETY: this test holds env_lock across the env-var window.
    unsafe {
        std::env::set_var(REQUIRE_ALL_RULES_ENV, "1");
    }

    let diags = lower_bad_sim_and_collect_diagnostics();
    assert!(
        !diags.is_empty(),
        "synthetic bad-field .sim should produce >=1 lower diagnostic",
    );

    let result = check_required_rules("synthetic_bad_field", &diags);

    // Restore env BEFORE asserting so a panic doesn't leak the var.
    // SAFETY: this test holds env_lock across the env-var window.
    unsafe {
        std::env::remove_var(REQUIRE_ALL_RULES_ENV);
    }

    let msg = result.expect_err(
        "env=1 + non-empty diagnostics must promote to Err(message); \
         got Ok — the gate is failing open",
    );
    assert!(
        msg.contains("synthetic_bad_field"),
        "error message should be tagged with the fixture name; got: {msg}",
    );
    assert!(
        msg.contains(REQUIRE_ALL_RULES_ENV),
        "error message should reference the env-var name so a fixture \
         author can disable the gate or fix the rule; got: {msg}",
    );
    assert!(
        msg.contains("lower diag"),
        "error message should preserve the historical `[fixture lower \
         diag]` prefix so existing tooling still grep-matches; got: \
         {msg}",
    );
}

#[test]
fn empty_diagnostics_are_always_ok_regardless_of_env() {
    // No env-gate — empty diagnostic Vec is the steady-state for any
    // fixture whose .sim lowers cleanly. The promote-to-error path
    // must NEVER fire on an empty Vec, even with the env var set.
    let _g = env_lock().lock().unwrap_or_else(|p| p.into_inner());
    // SAFETY: this test holds env_lock across the env-var window.
    unsafe {
        std::env::set_var(REQUIRE_ALL_RULES_ENV, "1");
    }
    let result = check_required_rules::<&str>("clean_fixture", &[]);
    // SAFETY: this test holds env_lock across the env-var window.
    unsafe {
        std::env::remove_var(REQUIRE_ALL_RULES_ENV);
    }
    assert!(
        result.is_ok(),
        "empty diagnostics + env=1 must still be Ok; got Err: {:?}",
        result.err(),
    );
}

#[test]
fn env_value_truthy_parsing_is_case_insensitive() {
    // The env-gate accepts `1`, `true`, `yes`, `on` (and case
    // variants). Anything else — including `0`, `false`, empty
    // string — is treated as off. Pin both arms so a future "make
    // it stricter / looser" change has to update this test
    // intentionally.
    let _g = env_lock().lock().unwrap_or_else(|p| p.into_inner());
    let diags = vec!["fake diagnostic"];

    for truthy in ["1", "true", "TRUE", "yes", "Yes", "on", "ON"] {
        // SAFETY: this test holds env_lock across the env-var window.
        unsafe {
            std::env::set_var(REQUIRE_ALL_RULES_ENV, truthy);
        }
        let r = check_required_rules("env_parse_test", &diags);
        // SAFETY: this test holds env_lock across the env-var window.
        unsafe {
            std::env::remove_var(REQUIRE_ALL_RULES_ENV);
        }
        assert!(
            r.is_err(),
            "env={truthy:?} should be truthy (Err); got Ok — the gate \
             accepts a value the README documents as enabling fail-fast",
        );
    }

    for falsy in ["0", "false", "no", "off", "", "unset_lookalike"] {
        // SAFETY: this test holds env_lock across the env-var window.
        unsafe {
            std::env::set_var(REQUIRE_ALL_RULES_ENV, falsy);
        }
        let r = check_required_rules("env_parse_test", &diags);
        // SAFETY: this test holds env_lock across the env-var window.
        unsafe {
            std::env::remove_var(REQUIRE_ALL_RULES_ENV);
        }
        assert!(
            r.is_ok(),
            "env={falsy:?} should be falsy (Ok); got Err — the gate \
             promoted on a value documented as preserving the legacy \
             warn-and-drop path",
        );
    }
}
