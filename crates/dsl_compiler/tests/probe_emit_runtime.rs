//! Integration stress test for the compiler-emitted probe runner.
//!
//! Plan §"Slice B" runtime gate (`docs/superpowers/plans/2026-05-03-
//! verb-probe-metric-emit.md`): the emitted `run_<probe>` fn must
//! actually evaluate the assertion at runtime — not just constant-fold
//! the predicate at emit time and discard the result. This test pins
//! that behaviour by constructing two synthetic compilations (one
//! whose const-folded assertion holds, one whose const-folded
//! assertion trips) and inspecting the emitted source string for the
//! presence of both the `Failed { … }` payload literal and the path
//! that returns it.
//!
//! Why string-shape inspection here vs running the probe through a
//! real CompiledSim: the dsl_compiler crate has no engine dependency,
//! so we can't materialise a `&mut dyn CompiledSim` at this layer.
//! The runtime-execution gate lives in `predator_prey_runtime/tests/
//! probe_drives.rs` (which DOES have engine in scope). Together the
//! two tests pin the full emit + drive contract.

use dsl_compiler::cg::emit::probes::synthesize_probes;
use dsl_ast::resolve::resolve;

fn compile_with_probe(probe_body: &str) -> dsl_ast::ir::Compilation {
    let src = format!(
        r#"
event Tick {{ }}
entity Agent_ : Agent {{ pos: vec3, }}
{probe_body}
"#,
        probe_body = probe_body,
    );
    let p = dsl_compiler::parse(&src).expect("parse synthetic source");
    resolve(p).expect("resolve synthetic source")
}

#[test]
fn probe_emits_runner_with_failure_path() {
    // The assertion `count[5.0] == 99.0` constant-folds at compile
    // time to `5.0 == 99.0` → false. The runner still needs to
    // emit the runtime check (the plan's gate), and the Failed
    // payload must carry the right scalars.
    let comp = compile_with_probe(
        r#"
probe gate_fail {
  scenario "synthetic"
  seed 57005
  ticks 50
  assert {
    count[5.0] == 99.0
  }
}
"#,
    );
    let out = synthesize_probes(&comp).expect("non-empty (1 probe declared)");

    // Runner exists.
    assert!(
        out.contains("pub fn run_gate_fail<State: ::engine::sim_trait::CompiledSim>"),
        "expected per-probe runner fn:\n{out}",
    );
    // Tick count baked into the const header.
    assert!(out.contains("const TICKS: u32 = 50;"), "expected ticks=50:\n{out}");
    // Both LHS and RHS literals are emitted as `_f64` constants the
    // runtime compares — proves the constant folder produced the
    // right scalars (not zero, not the wrong slot).
    assert!(out.contains("let lhs: f64 = 5.0_f64;"), "expected LHS=5.0:\n{out}");
    assert!(out.contains("let rhs: f64 = 99.0_f64;"), "expected RHS=99.0:\n{out}");
    // Failure path returns the typed payload with the captured op.
    assert!(out.contains("op:       \"==\""), "expected op=='==':\n{out}");
    assert!(out.contains("ProbeOutcome::Failed"), "expected Failed return:\n{out}");
    assert!(
        out.contains("probe:    \"gate_fail\""),
        "expected probe-name literal in Failed payload:\n{out}",
    );
}

#[test]
fn probe_emits_runner_with_pass_path() {
    // The assertion `count[10.0] >= 5.0` constant-folds to true. The
    // runner still emits the comparison + the Failed branch (the
    // runtime evaluation isn't elided), then falls through to
    // ProbeOutcome::Passed.
    let comp = compile_with_probe(
        r#"
probe gate_pass {
  scenario "synthetic"
  ticks 10
  assert {
    count[10.0] >= 5.0
  }
}
"#,
    );
    let out = synthesize_probes(&comp).expect("non-empty");
    assert!(out.contains("pub fn run_gate_pass"), "expected runner fn:\n{out}");
    assert!(out.contains("ProbeOutcome::Passed"), "expected pass-path return:\n{out}");
    assert!(out.contains("op:       \">=\""), "expected captured op string '>=':\n{out}");
}

#[test]
fn probe_short_run_assertion_uses_declared_ticks() {
    // The plan's intentional-fail gate: drive a probe through fewer
    // ticks than its assertion requires. Today's emit pass keys off
    // the source-side `ticks` value verbatim — the runner drives
    // exactly `ticks` step()s, no more, no less. This test pins that
    // contract by varying `ticks` across probes and asserting the
    // emitted const matches each.
    for declared in [1u32, 50, 500, 1000] {
        let body = format!(
            r#"
probe vary_ticks {{
  scenario "x"
  ticks {declared}
  assert {{
    count[1.0] == 1.0
  }}
}}
"#
        );
        let comp = compile_with_probe(&body);
        let out = synthesize_probes(&comp).expect("non-empty");
        let expected = format!("const TICKS: u32 = {declared};");
        assert!(
            out.contains(&expected),
            "expected `{expected}` in emit; got:\n{out}",
        );
    }
}

#[test]
fn probe_unsupported_shape_emits_typed_skip() {
    // The plan: unsupported assertion shapes get a `// SKIP` comment
    // PLUS a typed `ProbeOutcome::Skipped { reason }` runtime
    // payload. The runner still drives the runtime through `ticks`
    // ticks (so even a SKIPped probe smoke-tests the kernel chain)
    // before returning Skipped.
    let comp = compile_with_probe(
        r#"
probe unsupported_assert {
  scenario "x"
  ticks 5
  assert {
    count[1.0] < max(1.0, 2.0)
  }
}
"#,
    );
    let out = synthesize_probes(&comp).expect("non-empty");
    assert!(
        out.contains("// SKIP `unsupported_assert` assert #0"),
        "expected SKIP comment:\n{out}",
    );
    assert!(
        out.contains("ProbeOutcome::Skipped"),
        "expected typed Skipped return:\n{out}",
    );
    // The runner STILL drives `ticks` step() calls before returning.
    assert!(
        out.contains("for _ in 0..TICKS { state.step(); }"),
        "even SKIPped probes should drive the runtime first:\n{out}",
    );
}

#[test]
fn no_probes_emits_nothing() {
    let src = r#"
event Tick { }
entity Agent_ : Agent { pos: vec3, }
"#;
    let p = dsl_compiler::parse(src).expect("parse");
    let comp = resolve(p).expect("resolve");
    assert!(synthesize_probes(&comp).is_none(), "no probes → no artifact");
}

#[test]
fn multiple_probes_each_get_own_runner() {
    // The plan: every probe declaration becomes its own
    // `run_<name>` fn. Two probes in one fixture → two functions,
    // distinct names, both callable independently.
    let comp = compile_with_probe(
        r#"
probe alpha {
  scenario "x"
  ticks 1
  assert {
    count[1.0] == 1.0
  }
}
probe beta {
  scenario "y"
  ticks 2
  assert {
    count[2.0] >= 1.0
  }
}
"#,
    );
    let out = synthesize_probes(&comp).expect("non-empty");
    assert!(out.contains("pub fn run_alpha"), "missing alpha runner:\n{out}");
    assert!(out.contains("pub fn run_beta"), "missing beta runner:\n{out}");
    // Each runner has its own TICKS const baked in.
    assert!(
        out.contains("const TICKS: u32 = 1;"),
        "alpha's TICKS const should be 1:\n{out}",
    );
    assert!(
        out.contains("const TICKS: u32 = 2;"),
        "beta's TICKS const should be 2:\n{out}",
    );
}
