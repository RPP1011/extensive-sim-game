//! G2 (see docs/superpowers/specs/2026-05-24-dsl-as-engine-scorecard.md):
//! a physics rule that is NOT `@phase(per_agent)` dispatches PerEvent
//! (one thread per event in the ring), where `self` — the per-agent
//! identity — is undefined. Referencing `self` in such a rule used to
//! lower + emit green, producing WGSL that reads an undeclared
//! `agent_id` and panics in naga at runtime (kernel creation). This
//! pins that the compiler now rejects it with a typed error instead.

use dsl_compiler::cg::lower::lower_compilation_to_cg;

fn lower_result(src: &str) -> Result<(), String> {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    match lower_compilation_to_cg(&comp) {
        Ok(_) => Ok(()),
        Err(o) => Err(format!("{:?}", o.diagnostics)),
    }
}

// PerEvent (default phase) physics rule that READS `self` (emits it as
// an event field). `self` → `agent_id`, undeclared in a PerEvent kernel.
const SELF_READ_PER_EVENT: &str = r#"
event Tick { }
event Pinged { who: AgentId }
entity Probe : Agent { }
physics BadSelfEmit {
  on Tick {} {
    emit Pinged { who: self };
  }
}
"#;

// Same rule, correctly phased per-agent: `self` is the swept agent and
// `agent_id` is bound. Must still compile.
const SELF_READ_PER_AGENT: &str = r#"
event Tick { }
event Pinged { who: AgentId }
entity Probe : Agent { }
@phase(per_agent)
physics GoodSelfEmit {
  on Tick {} {
    emit Pinged { who: self };
  }
}
"#;

#[test]
fn self_read_in_per_event_rule_is_rejected() {
    let err = lower_result(SELF_READ_PER_EVENT)
        .expect_err("a PerEvent physics rule reading `self` must be rejected, not lowered to broken WGSL");
    assert!(
        err.contains("SelfRefInPerEventBody"),
        "expected a SelfRefInPerEventBody diagnostic; got: {err}"
    );
}

#[test]
fn self_read_in_per_agent_rule_compiles() {
    lower_result(SELF_READ_PER_AGENT)
        .expect("a @phase(per_agent) rule binds `self`/`agent_id` and must compile");
}
