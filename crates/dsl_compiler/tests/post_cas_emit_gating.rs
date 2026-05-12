//! Plan G #244 bug 2 regression — post-CAS emit gating for the
//! "first-writer-wins" f32 RMW shape.
//!
//! Background
//! ----------
//!
//! When a physics-rule body has the shape:
//!
//! ```text
//! if (agents.hp(t) >= 100.0 && agents.alive(t)) {
//!     agents.set_hp(t, 99.0);
//!     emit Ignited { tree: t, by: s }
//! }
//! ```
//!
//! …and multiple producer threads concurrently see `hp == 100`, ALL
//! enter the body. The CAS loop runs:
//!   * Winner: `_old_bits = bits(100), _new_bits = bits(99)`,
//!     CAS(100→99) succeeds → real transition.
//!   * Losers: re-load, see `bits(99)`, CAS(99→99) succeeds trivially
//!     → NO real transition.
//!
//! Without gating, ALL threads (winner + N-1 losers) emit `Ignited`.
//! The forest_fire fixture's per-tree `ignition_count` view drifts
//! by N per tick across re-runs (the order in which losers retried is
//! non-deterministic; any per-thread side-effect that flows into a
//! view fold becomes a P5 violation).
//!
//! Fix: when the per-stmt path detects that a literal-F32 Assign sits
//! inside an `If` whose cond reads the same f32 field, emit the CAS
//! loop variant that captures the transition outcome in
//! `_f32_cas_did_transition_<sid>` (declared OUTSIDE the loop) and
//! wrap subsequent stmts in `if (_f32_cas_did_transition_<sid>) { ... }`
//! so only the winning thread fires post-write side-effects.
//!
//! Mirror: alive-CAS handling (Gap N, plague_city / wave_defense)
//! does the same wrapping for `set_alive(t, false)` writes —
//! `wgsl_body.rs::is_alive_cas_site` detection + `if
//! (_alive_cas_<sid>.exchanged) { ... }` wrap. The f32 path is the
//! generalization to literal-F32 writes guarded by an f32 field
//! read.
//!
//! Negative pin: the per-thread RHS shape (`set_hp(t, hp - 1)`,
//! `set_hp(t, hp - amount)`) does NOT trigger the gate — the
//! literal-RHS filter excludes it. This keeps `firebolt_probe`'s
//! chronicle-damage shape bit-identical (the existing CAS-loop
//! retry semantics are correct for per-thread RHS — every retry
//! computes a fresh contribution).

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

fn find_kernel_body<'a>(art: &'a EmittedArtifacts, needle: &str) -> &'a str {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle) && name.ends_with(".wgsl"))
        .map(|(_, b)| b.as_str())
        .unwrap_or_else(|| {
            let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
            panic!("no kernel matching `{needle}`; got: {names:?}")
        })
}

/// Positive pin: the canonical Plan G #244 bug 2 shape — a chronicle
/// consumer that catches a healthy target and transitions hp from
/// `>=100.0` to a literal `99.0`, emitting an event on the
/// transition. The emit MUST be hoisted into the post-CAS gating
/// branch so only the thread that won the actual hp transition
/// fires the event.
#[test]
fn first_writer_wins_hoists_emit_into_transition_gate() {
    // Canonical forest_fire Catch shape (slimmed).
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event EmberLanded {
  source: AgentId,
  target: AgentId,
}

@replayable @gpu_amenable
event Ignited {
  tree: AgentId,
  by:   AgentId,
}

entity Tree : Agent {
  pos: vec3,
  vel: vec3,
}

init {
  alive: 1,
  hp:    100,
}

@phase(post)
physics Catch {
  on EmberLanded { source: s, target: t } {
    if (agents.hp(t) >= 100.0 && agents.alive(t)) {
      agents.set_hp(t, 99.0);
      emit Ignited { tree: t, by: s }
    }
  }
}
"#;
    let art = compile_inline(src);
    let body = find_kernel_body(&art, "Catch");

    // 1. Storage upgrade: agent_hp must be `array<atomic<u32>>`. The
    //    set_hp write triggers the f32 RMW upgrade.
    assert!(
        body.contains("var<storage, read_write> agent_hp: array<atomic<u32>>;"),
        "expected agent_hp atomic upgrade — got:\n{body}"
    );

    // 2. The CAS loop is the "transition-tracking" variant. The marker
    //    is the `var _f32_cas_did_transition_<sid>: bool = false;`
    //    declared OUTSIDE the loop.
    assert!(
        body.contains("var _f32_cas_did_transition_"),
        "expected `var _f32_cas_did_transition_<sid>: bool = false;` outside the CAS loop — got:\n{body}"
    );

    // 3. Inside the loop's success branch, the transition outcome is
    //    captured: `_f32_cas_did_transition_<sid> = (_old_bits_<sid>
    //    != _new_bits_<sid>);` BEFORE the break. Without this, all
    //    CAS retries (including no-op winners that stored 99→99)
    //    would leave `_f32_cas_did_transition_<sid>` false and the
    //    emit would never fire.
    assert!(
        body.contains("_f32_cas_did_transition_") && body.contains(" = (_old_bits_"),
        "expected `_f32_cas_did_transition_<sid> = (_old_bits_<sid> != _new_bits_<sid>);` \
         on the CAS-success branch — got:\n{body}"
    );

    // 4. The emit MUST be inside an `if (_f32_cas_did_transition_<sid>) { ... }`
    //    block. The regression we're closing: emit being placed at
    //    the outer If's source-order position (executes for every
    //    thread that observed the guard, not just the transition
    //    winner).
    assert!(
        body.contains("if (_f32_cas_did_transition_"),
        "expected `if (_f32_cas_did_transition_<sid>) {{ ... }}` wrapping the post-CAS emit — got:\n{body}"
    );

    // 5. Cross-check: the emit slot-add for Ignited (event id resolved
    //    from the outer atomic event-tail bump) must literally appear
    //    inside the gating block. The simplest structural check: the
    //    emit-skeleton substring follows the gate's open brace before
    //    the gate's close brace. We test the loose form: the emit
    //    skeleton (`atomicAdd(&event_tail`) appears AFTER
    //    `if (_f32_cas_did_transition_` in source order.
    let gate_pos = body
        .find("if (_f32_cas_did_transition_")
        .expect("gate substring missing — earlier assertion should've caught this");
    let emit_pos = body
        .rfind("atomicAdd(&event_tail[0], 1u)")
        .expect("emit skeleton missing — Ignited never emitted?");
    assert!(
        emit_pos > gate_pos,
        "emit skeleton at position {emit_pos} appears BEFORE the gate at {gate_pos} — \
         the wrap didn't capture the emit. body:\n{body}"
    );
}

/// Negative pin: per-thread RHS (`set_hp(t, hp - 1)`) does NOT
/// trigger the transition-gating variant. The CAS loop stays in its
/// legacy shape because the natural retry covers correctness — every
/// retry computes a fresh `hp - 1` against the current snapshot, so
/// each thread's contribution is real.
///
/// This pin guards against an over-eager generalization that would
/// gate ALL guarded f32 CAS sites (which would silence valid emits
/// in `if (hp > 0) { set_hp(t, hp - 1); emit Hit }` shapes).
#[test]
fn per_thread_rhs_does_not_trigger_transition_gate() {
    // Conditional decrement with per-thread RHS — emit Hit on every
    // tick the agent has positive hp. This is NOT first-writer-wins;
    // every thread's contribution is a real transition.
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Hit {
  victim: AgentId,
}

entity Critter : Agent {
  pos: vec3,
  vel: vec3,
}

init {
  alive: 1,
  hp:    100,
}

@phase(per_agent)
physics Decay {
  on Tick {} where (self.alive) {
    let cur = agents.hp(self);
    if (cur > 0.0) {
      agents.set_hp(self, cur - 1.0);
      emit Hit { victim: self }
    }
  }
}
"#;
    let art = compile_inline(src);
    let body = find_kernel_body(&art, "Decay");

    // The per-thread RHS shape (`cur - 1.0`) is NOT a literal F32 —
    // the gate predicate excludes it. The CAS loop stays bit-
    // identical to the legacy shape, and NO transition-gating
    // variable is emitted.
    assert!(
        !body.contains("_f32_cas_did_transition_"),
        "transition gate variable should NOT appear for per-thread RHS shape; \
         the literal-RHS filter must exclude `set_hp(t, hp - 1.0)`. body:\n{body}"
    );
    assert!(
        !body.contains("if (_f32_cas_did_transition_"),
        "transition-gate wrap should NOT appear for per-thread RHS shape. body:\n{body}"
    );
}

/// Negative pin: a literal-RHS Assign WITHOUT a guarding If on the
/// same field stays on the legacy CAS shape. Without the cond
/// reading the assigned field, there's no first-writer-wins intent
/// to detect.
///
/// Example: `agents.set_hp(self, 50.0); emit Reset` at the top
/// level of a body — every thread that runs the rule writes 50.0
/// and emits Reset. No gating; no transition-tracking variable.
#[test]
fn literal_rhs_without_field_guard_stays_legacy() {
    // Top-level literal write + emit, with a cond that reads a
    // DIFFERENT field (alive). The `If`-arm pattern detection requires
    // the cond to read the SAME field as the inner Assign.
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Reset {
  victim: AgentId,
}

entity Critter : Agent {
  pos: vec3,
  vel: vec3,
}

init {
  alive: 1,
  hp:    100,
}

@phase(per_agent)
physics Resetter {
  on Tick {} where (self.alive) {
    if (agents.alive(self)) {
      agents.set_hp(self, 50.0);
      emit Reset { victim: self }
    }
  }
}
"#;
    let art = compile_inline(src);
    let body = find_kernel_body(&art, "Resetter");

    // The cond reads alive, the Assign writes hp — different field.
    // No transition gate emitted.
    assert!(
        !body.contains("_f32_cas_did_transition_"),
        "transition gate should NOT fire when the cond reads a different field \
         than the inner Assign. body:\n{body}"
    );
}

/// Cross-check: the alive-CAS hoisting (Gap N) still works for
/// `set_alive(self, false)` patterns — the new f32 gate is
/// orthogonal and must not regress alive-CAS detection.
#[test]
fn alive_cas_hoisting_still_works_alongside_f32_gate() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Died {
  victim: AgentId,
}

entity Citizen : Agent {
  pos: vec3,
  vel: vec3,
}

init {
  alive: 1,
  hp:    100,
}

@phase(per_agent)
physics Sickness {
  on Tick {} where (self.alive) {
    let old_hp = agents.hp(self);
    let new_hp = old_hp - 1.0;
    agents.set_hp(self, new_hp);
    if (old_hp > 0.0 && new_hp <= 0.0) {
      agents.set_alive(self, false);
      emit Died { victim: self }
    }
  }
}
"#;
    let art = compile_inline(src);
    let body = find_kernel_body(&art, "Sickness");

    // Alive-CAS path still detects + emits the wrap.
    assert!(
        body.contains("var<storage, read_write> agent_alive: array<atomic<u32>>;"),
        "alive_cas binding upgrade still required — got:\n{body}"
    );
    assert!(
        body.contains("atomicCompareExchangeWeak(&agent_alive["),
        "alive_cas write still required — got:\n{body}"
    );
    assert!(
        body.contains(", 1u, 0u)"),
        "alive_cas write must be in the kill direction — got:\n{body}"
    );
    assert!(
        body.contains("if (_alive_cas_") && body.contains(".exchanged) {"),
        "alive_cas wrap (`if (_alive_cas_<sid>.exchanged) {{ ... }}`) still required — got:\n{body}"
    );

    // The f32 RMW for set_hp uses per-thread RHS (`new_hp = old_hp -
    // 1.0`) — NOT a literal — so no transition gate.
    assert!(
        !body.contains("_f32_cas_did_transition_"),
        "per-thread RHS set_hp must NOT trigger the f32 transition gate; \
         the alive-CAS path is the only hoisting active here. body:\n{body}"
    );
}
