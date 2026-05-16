//! Plan I.4b IR-driven source-agent field offset — verifies the
//! merge kernel reads the source agent from the correct event payload
//! offset when the merge-clause's source field ISN'T the first
//! payload field on the event.
//!
//! Before iter 19, the WGSL emit hardcoded offset 2 (= first payload
//! word, after the 2-word header). Events like `AllyDied { dead:
//! Agent }` worked because `dead` was at offset 2. But events like
//! `event Combat { killer: Agent, victim: Agent }` with
//! `merge from victim: bit_or` would have read `killer` instead of
//! `victim` — silent wrong-source bug.
//!
//! This test compiles a belief with `merge from victim` where
//! `victim` is the SECOND payload field, and asserts the emitted
//! WGSL references the correct event_ring offset (= 2 + 1 = 3).

use dsl_compiler::cg::emit::emit_cg_program;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};

#[test]
fn merge_source_field_offset_reads_from_correct_event_payload_word() {
    // `event Combat { killer: Agent, victim: Agent }` — killer at
    // payload offset 0, victim at payload offset 1. The merge clause
    // takes `victim`, so the emitted kernel must read from event_ring
    // offset 2 + 1 = 3.
    let src = "\
        event Combat { killer: Agent, victim: Agent }\n\
        belief b(observer: Agent, subject: Agent) -> u32 {\n\
          initial: 0,\n\
          on Combat { killer: k, victim: v } merge from v: bit_or\n\
        }\n";
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = lower_compilation_to_cg(&comp)
        .unwrap_or_else(|o| panic!("lower failed: {:?}",
            o.diagnostics.iter().map(|d| format!("{d}")).collect::<Vec<_>>()));
    let sched = synthesize_schedule(&cg, ScheduleStrategy::Default);
    let art = emit_cg_program(&sched.schedule, &cg)
        .unwrap_or_else(|e| panic!("emit: {e:?}"));

    // Find the merge kernel WGSL.
    let merge_kernel = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("merge_b_") && name.ends_with(".wgsl"))
        .expect("merge kernel emitted");
    let body = &merge_kernel.1;

    // The kernel should read the source agent from offset 3 (= 2
    // header + 1 payload index for `victim`), NOT offset 2 (which
    // would be `killer`).
    assert!(
        body.contains("event_ring[event_idx * 10u + 3u]"),
        "expected merge kernel to read source_agent from event payload \
         offset 3 (= header 2 + victim's field index 1); body excerpt:\n{}",
        &body[..body.len().min(1200)],
    );
    // Defensive: should NOT read from offset 2 (the killer field).
    assert!(
        !body.contains("let source_agent = event_ring[event_idx * 10u + 2u]"),
        "merge kernel still reads source_agent from offset 2 (killer field) — \
         IR field-offset lookup didn't propagate. Body excerpt:\n{}",
        &body[..body.len().min(1200)],
    );
}

#[test]
fn merge_source_field_offset_defaults_to_2_for_first_payload_field() {
    // Sanity: the common case (source IS the first payload field)
    // still emits the historic offset 2.
    let src = "\
        event AllyDied { dead: Agent }\n\
        belief b(observer: Agent, subject: Agent) -> u32 {\n\
          initial: 0,\n\
          on AllyDied { dead: d } merge from d: bit_or\n\
        }\n";
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = lower_compilation_to_cg(&comp).expect("lower");
    let sched = synthesize_schedule(&cg, ScheduleStrategy::Default);
    let art = emit_cg_program(&sched.schedule, &cg).expect("emit");

    let merge_kernel = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("merge_b_") && name.ends_with(".wgsl"))
        .expect("merge kernel emitted");
    let body = &merge_kernel.1;

    assert!(
        body.contains("event_ring[event_idx * 10u + 2u]"),
        "single-Agent-payload event (dead at offset 2) should still emit \
         offset 2; body excerpt:\n{}",
        &body[..body.len().min(800)],
    );
}
