//! Pin: `spatial.nearby(<non-self origin>)` lowers correctly in a
//! `@phase(post)` chronicle consumer.
//!
//! Closes the regression-test half of **Gap dungeon_horde#1** in
//! `docs/architecture/gaps_dungeon_horde.md` — the auto-injected
//! squared-distance gate from Gap dungeon_layout#1 (commit `375cfc77`)
//! hard-coded `agent_pos[agent_id]` for the cell-window centre, which
//! is unbound in `@phase(post)` event-handler bodies (the implicit
//! per-thread loop is over `event_idx`, not `agent_id`).
//!
//! Pre-fix, migrating dungeon_horde's three scale-hotspot rules
//! (`SoundDetectFromDamage`, `BroadcastAlertOnAllyDeath`,
//! `ScoutBroadcast`) from full-population `for_each_agent` walks to
//! `for slot in spatial.nearby(<event-bound-source>) { … }` was
//! syntactically accepted by the lowerer but produced naga-rejected
//! WGSL referencing an unbound `agent_id` identifier.
//!
//! Post-fix the WGSL emit substitutes the caller's actual origin
//! expression (lowered as a `CgExprId`) into the gate prefix —
//! `spatial.nearby(s)` where `s` is an event-pattern binding now
//! emits `agent_pos[<lowered s>]` instead of `agent_pos[agent_id]`,
//! sidestepping the bug.
//!
//! What this pin asserts:
//!
//!   1. A synthetic `@phase(post)` consumer using
//!      `spatial.nearby(<event-bound binder>)` lowers + emits without
//!      error (no `LoweringError::UnsupportedAstNode`, no
//!      `EmitError::*`).
//!   2. The emitted WGSL's gate origin reads through the bound local's
//!      name (NOT through `agent_id`). The bound local lowers via
//!      `IrStmt::Let { … = EventField { … } }` to a `local_<N>`
//!      identifier; the gate prefix must reference that name.
//!   3. Every emitted kernel parses cleanly with naga (the original
//!      bug surface — unbound `agent_id` was caught by naga, not by
//!      the compiler's well-formed pass).

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cg::lower::lower_compilation_to_cg;

fn compile_inline(src: &str) -> EmittedArtifacts {
    let prog = dsl_compiler::parse(src).expect("parse");
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

/// `@phase(post)` consumer walks `spatial.nearby(<event-bound source>)`.
/// The gate origin must reference the bound local, not `agent_id`.
#[test]
fn spatial_nearby_in_post_phase_uses_event_bound_origin() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event SourcePinged { source: AgentId, marker: AgentId }

@replayable @gpu_amenable
event NeighbourSeen { observer: AgentId, source: AgentId }

entity Walker : Agent {
  pos: vec3,
  vel: vec3,
}

spatial_query nearby_any(self: AgentId, candidate: AgentId) =
  candidate != self

// Source rule: emits a SourcePinged event each tick from slot 0.
// (Synthetic — the test just needs SourcePinged to land on the ring
// so the post-phase consumer below has a chronicle event to walk.)
@phase(per_agent)
physics PingFromSource {
  on Tick {} where (self.alive) {
    emit SourcePinged { source: self, marker: self }
  }
}

// The migration shape from dungeon_horde: a @phase(post) consumer
// walks spatial.nearby() with an event-bound source agent (NOT
// `self`). Pre-fix this lowered cleanly but emitted naga-rejected
// WGSL referencing unbound `agent_id`. Post-fix the gate origin
// reads through the bound local `s`.
@phase(post)
physics NeighbourBroadcast {
  on SourcePinged { source: s, marker: _ } {
    for slot in spatial.nearby_any(s) {
      emit NeighbourSeen { observer: slot, source: s }
    }
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_body_containing(&art, "NeighbourBroadcast").unwrap_or_else(|| {
        let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
        panic!("expected a NeighbourBroadcast kernel; got: {names:?}");
    });

    // 1. The gate origin id binding must reference the lowered event-
    //    bound local — NOT the unbound `agent_id`. The event-bound
    //    pattern binder `s` lowers via `IrStmt::Let { … = EventField
    //    { … } }` to a `local_<N>` name. The exact N varies with
    //    lowering order, so we check the pattern.
    let has_origin_binding = body.contains("_gate_origin_id: u32 = local_");
    assert!(
        has_origin_binding,
        "expected `_gate_origin_id: u32 = local_<N>` referencing the \
         event-bound source binding; body:\n{body}",
    );

    // 2. The gate origin must NOT reference `agent_id` here — the
    //    legacy hard-coded shape is what broke in @phase(post).
    assert!(
        !body.contains("_gate_origin_id: u32 = agent_id"),
        "regression: gate origin fell back to `agent_id` in a \
         @phase(post) consumer; body:\n{body}",
    );

    // 3. The gate body must reference `_gate_origin_pos` (the lowered
    //    origin position) for the squared-distance delta.
    assert!(
        body.contains("_gate_dxyz = agent_pos[per_pair_candidate] - _gate_origin_pos"),
        "expected `_gate_dxyz` delta to subtract `_gate_origin_pos`; \
         body:\n{body}",
    );

    // 4. Naga parses every emitted kernel — the load-bearing assertion
    //    for the gap, since the original bug surfaced as a naga reject.
    let mut failures: Vec<(String, String)> = Vec::new();
    for (name, file_body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(file_body) {
            failures.push((name.clone(), e.emit_to_string(file_body)));
        }
    }
    assert!(
        failures.is_empty(),
        "naga validation failures for @phase(post) spatial.nearby(<event-bound>):\n{:#?}",
        failures,
    );
}

/// Sanity: `spatial.nearby(self)` in a per-agent rule continues to
/// emit `_gate_origin_id: u32 = agent_id` (unchanged behaviour — the
/// fix only redirects non-`self` origins).
#[test]
fn spatial_nearby_with_self_in_per_agent_unchanged() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Pinged { observer: AgentId, target: AgentId }

entity Walker : Agent {
  pos: vec3,
  vel: vec3,
}

spatial_query nearby_others(self: AgentId, candidate: AgentId) =
  candidate != self

@phase(per_agent)
physics PingNeighbors {
  on Tick {} where (self.alive) {
    for other in spatial.nearby_others(self) {
      emit Pinged { observer: self, target: other }
    }
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_body_containing(&art, "PingNeighbors").unwrap_or_else(|| {
        let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
        panic!("expected a PingNeighbors kernel; got: {names:?}");
    });

    // Origin binds to `agent_id` (the per-agent kernel's implicit
    // per-thread slot, unchanged from pre-fix).
    assert!(
        body.contains("_gate_origin_id: u32 = agent_id"),
        "expected `_gate_origin_id: u32 = agent_id` for `self` origin \
         in per-agent rule; body:\n{body}",
    );

    let mut failures: Vec<(String, String)> = Vec::new();
    for (name, file_body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(file_body) {
            failures.push((name.clone(), e.emit_to_string(file_body)));
        }
    }
    assert!(
        failures.is_empty(),
        "naga validation failures for self-origin spatial.nearby:\n{:#?}",
        failures,
    );
}
