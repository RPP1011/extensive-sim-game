//! Pin: every event payload's last word is the seq trailer, written
//! by the producer's atomicStore alongside other payload fields.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn compile(src: &str) -> EmittedArtifacts {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| o.program);
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

#[test]
fn chronicle_emit_writes_seq_trailer_as_last_payload_word() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Damaged { source: AgentId, target: AgentId, amount: f32 }

@phase(per_agent)
physics Fire {
  on Tick {} where (self.alive) {
    emit Damaged { source: self, target: self, amount: 1.0 }
  }
}
"#;
    let art = compile(src);
    let (_, body) = art.wgsl_files.iter()
        .find(|(name, _)| name.contains("Fire"))
        .expect("Fire kernel emitted");

    // Seq trailer is at offset `stride - 1` = 10 (header 2 + payload 8).
    assert!(
        body.contains("[slot * 11u + 10u]"),
        "expected seq trailer write at offset 10 (stride 11); got body:\n{body}",
    );

    // Seq value is packed: (kernel_id << 24) | (agent_id << 4) | emit_idx.
    // For the single emit in this fixture, expect kernel_id=0, emit_idx=0,
    // agent_id is the per-thread index.
    assert!(
        body.contains("(0u << 24u) | (agent_id << 4u) | 0u"),
        "expected packed seq formula in body; got:\n{body}",
    );
}
