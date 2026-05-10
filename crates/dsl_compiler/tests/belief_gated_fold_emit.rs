//! Compiler emit pin — the `@belief_gated` annotation flips the
//! PerAgentEventScan fold's source-candidate gate from raw busy
//! lookup to per-(observer, source) belief lookup.
//!
//! This is the load-bearing emit-side test for the E2E belief-gated
//! fold. The host-side spec at
//! `dsl_compiler/tests/threats_belief_gated_fold.rs` defined the
//! semantic contract; the GPU storage pin at
//! `tom_probe_runtime/tests/belief_gated_threat_awareness_gpu.rs`
//! proved the storage layer holds divergent per-cell bits. This
//! test proves the COMPILER actually emits the swapped predicate
//! when the annotation is present, AND falls back to the omniscient
//! gate when it isn't (so existing fixtures stay green).

use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};
use dsl_compiler::cg::emit::emit_cg_program;

fn compile_to_wgsl(src: &str) -> std::collections::BTreeMap<String, String> {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => o.program,
    };
    let schedule = synthesize_schedule(&cg, ScheduleStrategy::Default);
    let arts = emit_cg_program(&schedule.schedule, &cg).expect("emit");
    arts.wgsl_files
}

const BELIEF_GATED_SIM: &str = r#"
event Tick { }
event ScriptedCastBegin { source: AgentId, target: AgentId, ability_id: u32 }

entity Probe : Agent { }

config probe { dummy: u32 = 1 }

@phase(per_agent)
physics MarkBusy {
  on Tick {} where (self.alive && world.tick == 0) {
    agents.set_busy_with_ability_id(self, config.probe.dummy);
  }
}

@materialized(on_event = [ScriptedCastBegin])
@dispatch(per_agent_event_scan)
@belief_gated
view threats(observer: Agent) -> f32 {
  initial: 0.0,
  on ScriptedCastBegin { source: _, target: _, ability_id: _ } { self += 1.0 }
}
"#;

const OMNISCIENT_SIM: &str = r#"
event Tick { }
event ScriptedCastBegin { source: AgentId, target: AgentId, ability_id: u32 }

entity Probe : Agent { }

config probe { dummy: u32 = 1 }

@phase(per_agent)
physics MarkBusy {
  on Tick {} where (self.alive && world.tick == 0) {
    agents.set_busy_with_ability_id(self, config.probe.dummy);
  }
}

@materialized(on_event = [ScriptedCastBegin])
@dispatch(per_agent_event_scan)
view threats(observer: Agent) -> f32 {
  initial: 0.0,
  on ScriptedCastBegin { source: _, target: _, ability_id: _ } { self += 1.0 }
}
"#;

#[test]
fn belief_gated_annotation_emits_belief_cell_predicate() {
    let wgsl = compile_to_wgsl(BELIEF_GATED_SIM);
    let fold = wgsl
        .get("fold_threats.wgsl")
        .expect("fold_threats kernel emitted");

    // The belief-gated predicate appears in the kernel body.
    assert!(
        fold.contains("let belief_cell = beliefs_flags[observer * cfg.event_count + source_candidate];"),
        "expected belief_cell lookup; WGSL was:\n{fold}"
    );
    assert!(
        fold.contains("if ((belief_cell & (1u << 7u)) == 0u) { return; }"),
        "expected belief gate predicate (bit 7); WGSL was:\n{fold}"
    );
    // The omniscient predicate is GONE.
    assert!(
        !fold.contains("if (agent_busy_with_ability_id[source_candidate] == 0u) { return; }"),
        "@belief_gated must REPLACE the omniscient gate; WGSL still contains the busy check:\n{fold}"
    );
    // The beliefs_flags binding is declared.
    assert!(
        fold.contains("var<storage, read> beliefs_flags: array<u32>;"),
        "expected beliefs_flags binding declaration; WGSL was:\n{fold}"
    );
}

#[test]
fn omniscient_default_keeps_busy_predicate() {
    let wgsl = compile_to_wgsl(OMNISCIENT_SIM);
    let fold = wgsl
        .get("fold_threats.wgsl")
        .expect("fold_threats kernel emitted");

    // Without @belief_gated, the omniscient gate is preserved
    // verbatim — every existing fixture that relies on this shape
    // (dodger_probe / threats_view_probe / threats_with_decay_probe)
    // continues to work without source changes.
    assert!(
        fold.contains("if (agent_busy_with_ability_id[source_candidate] == 0u) { return; }"),
        "default fold must preserve the omniscient busy gate; WGSL was:\n{fold}"
    );
    // The belief-gated predicate is NOT emitted.
    assert!(
        !fold.contains("belief_cell"),
        "default fold must NOT emit belief_cell lookup; WGSL was:\n{fold}"
    );
    assert!(
        !fold.contains("beliefs_flags"),
        "default fold must NOT bind beliefs_flags; WGSL was:\n{fold}"
    );
}
