//! Pin: `sum(... where pred { 1u } else { 0u })` lowers as a u32
//! accumulator without requiring an explicit f32 cast.
//!
//! Closes the regression-test half of **Gap dungeon_stealth#2** in
//! `docs/architecture/gaps_dungeon_stealth.md` — pre-fix the Sum
//! lowering only seeded an `init` literal for `I32 / F32 / Vec3F32`,
//! so a body whose arms were `1u` / `0u` (typed `U32`) hit the
//! "unsupported sum body type" branch and the rule's host kernel
//! was silently dropped from the emit set with a `lower diag`
//! warning.
//!
//! Post-fix the `FoldKind::Sum` arm has a `CgTy::U32 => LitValue::U32(0)`
//! init case. The WGSL emit's `local_N = local_N + projection` lowers
//! uniformly across U32/I32/F32 since the `+` operator is the same
//! WGSL token at all numeric widths.
//!
//! What this pin asserts:
//!
//!   1. A synthetic `@phase(per_agent)` rule with a u32 sum body
//!      (`sum(other in agents where (other != self) { 1u } else { 0u })`)
//!      lowers + emits without diagnostic.
//!   2. The emitted kernel body declares the accumulator local as `u32`
//!      and seeds the init expression as `0u`.
//!   3. The kernel parses cleanly with naga (catches any latent type
//!      drift the compiler's well-formed pass might miss).

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cg::lower::lower_compilation_to_cg;

fn compile_inline(src: &str) -> EmittedArtifacts {
    let prog = dsl_compiler::parse(src).expect("parse");
    // Intern `field <name>: <ty>` declarations so the lowerer's
    // `agents.set_<name>` resolution can find them.
    let _custom_ids = dsl_compiler::custom_agent_fields::populate(&prog);
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(outcome) => {
            for diag in &outcome.diagnostics {
                eprintln!("[lower diagnostic] {diag}");
            }
            panic!(
                "lower_compilation_to_cg returned {} diagnostic(s) — see stderr above",
                outcome.diagnostics.len()
            );
        }
    };
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit")
}

fn kernel_body_containing<'a>(
    art: &'a EmittedArtifacts,
    needle: &str,
) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle) && name.ends_with(".wgsl"))
        .map(|(_, body)| body.as_str())
}

/// A `sum` body returning u32 lit arms must lower without dropping
/// the host kernel.
#[test]
fn sum_u32_arms_lower_and_emit() {
    let src = r#"
event Tick { }

entity Walker : Agent {
  pos: vec3,
  vel: vec3,
}

field nearby_count_u32: u32

@replayable @gpu_amenable
event NearbyCount { source: AgentId, n: u32 }

@phase(per_agent)
physics CountNearbyU32 {
  on Tick {} where (self.alive) {
    let nearby = sum(other in agents where
      if (other != self) && (other.alive) {
        1u
      } else {
        0u
      });
    emit NearbyCount { source: self, n: nearby }
  }
}

@phase(post)
physics ApplyNearbyCount {
  on NearbyCount { source: s, n: c } {
    agents.set_nearby_count_u32(s, c);
  }
}
"#;
    let art = compile_inline(src);

    let body = kernel_body_containing(&art, "CountNearbyU32")
        .expect("CountNearbyU32 kernel must emit (host kernel was dropped pre-fix)");

    // Accumulator is u32-typed; init expression is the u32 zero literal.
    assert!(
        body.contains("var local_") && body.contains(": u32"),
        "expected u32 accumulator declaration in CountNearbyU32 body — got:\n{body}"
    );
    assert!(
        body.contains("0u"),
        "expected `0u` init literal in CountNearbyU32 body — got:\n{body}"
    );

    // Parse with naga to make sure the WGSL is well-formed.
    let module = naga::front::wgsl::parse_str(body)
        .unwrap_or_else(|e| panic!("naga parse failed for CountNearbyU32:\n{body}\nerror:\n{e:?}"));
    let _validation = naga::valid::Validator::new(
        naga::valid::ValidationFlags::default(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("naga validate failed for CountNearbyU32:\n{body}\nerror:\n{e:?}"));
}
