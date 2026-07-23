//! Pin the `@storage(packed_q8)` annotation: the auto-emitted decay
//! kernel for a q8-packed view processes 4 cells per word via byte-shift
//! unpack/process/repack, mirroring the bespoke
//! `belief_decay_wgsl::decay_kernel_wgsl` shape it subsumes.
//!
//! Coverage:
//!   1. `mode = sub, by = 1` + `@storage(packed_q8)` (no gate) — the
//!      tom_probe shape modulo gate. Asserts the per-WORD preamble, the
//!      per-byte unpack/decay/repack, and the saturating-sub formula
//!      live inside the byte loop (not at the per-cell top level).
//!   2. `mode = sub, by = 1` + `@storage(packed_q8)` + `gate = MaskName`
//!      where the mask reads `agents.beliefs_last_seen_tick` — the real
//!      tom_probe shape. Asserts the gate predicate inlines per-byte
//!      and the BeliefStateColumn handle walk extends the kernel BGL.
//!   3. Resolve-side: invalid `@storage(<name>)` shapes surface typed
//!      errors (unknown packing, non-ident arg, multi-arg).

const TEMPLATE_NO_GATE_Q8: &str = r#"
event Tick { }

@replayable
@gpu_amenable
event Hit {
  target: AgentId,
}

entity Particle : Agent {
  pos: vec3,
  vel: vec3,
}

@materialized(on_event = [Hit])
@decay(per = tick, mode = sub, by = 1)
@storage(packed_q8)
view confidence(observer: Agent, subject: Agent) -> u32 {
  initial: 0,
  on Hit { target: agent } where agent == subject { self += 1 }
  clamp: [0, 255],
}

physics Tickle @phase(per_agent) {
  on Tick {} {
    emit Hit { target: self }
  }
}
"#;

#[test]
fn packed_q8_emits_per_word_byte_loop_in_decay_wgsl() {
    let program = dsl_compiler::parse(TEMPLATE_NO_GATE_Q8).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).expect("lower");
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit");
    let decay_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(k, _)| k.starts_with("decay_"))
        .map(|(_, v)| v.as_str())
        .expect("decay kernel WGSL must be emitted when @decay is set");

    // 1. q8 path uses ONE thread per WORD, not per cell — the preamble
    //    binds `word_idx` (not `k`) and the bound check fires against
    //    the word count.
    assert!(
        decay_wgsl.contains("let word_idx = gid.x + gid.y * 4194240u;"),
        "expected per-WORD preamble; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("if (word_idx >= cfg.slot_count)"),
        "expected per-WORD bounds check; got:\n{decay_wgsl}"
    );
    // 2. Each thread reads its packed word and walks 4 byte slots.
    assert!(
        decay_wgsl.contains("atomicLoad(&view_storage_primary[word_idx])"),
        "expected packed-word atomic load; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("for (var b: u32 = 0u; b < 4u;"),
        "expected per-byte loop; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("(word >> shift) & 0xFFu"),
        "expected byte-shift unpack; got:\n{decay_wgsl}"
    );
    // 3. The saturating-sub formula lives INSIDE the byte loop and
    //    operates on the unpacked u8 (`conf` / `new_conf`), not on the
    //    full word. The `by` constant is still `1u`.
    assert!(
        decay_wgsl.contains("let by: u32 = 1u;"),
        "expected `by` constant from mode=sub, by=1; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("select(conf - by, 0u, conf < by)"),
        "expected per-byte saturating-sub `select(conf - by, 0u, conf < by)`; got:\n{decay_wgsl}"
    );
    // 4. Recompose: the new bytes get OR'd into the new word with the
    //    matching shift, then atomic-stored back.
    assert!(
        decay_wgsl.contains("new_word | ((new_conf & 0xFFu) << shift)")
            || decay_wgsl.contains("new_word = new_word | ((new_conf & 0xFFu) << shift)"),
        "expected per-byte recompose into new_word; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("atomicStore(&view_storage_primary[word_idx]"),
        "expected atomic store of the recomposed word; got:\n{decay_wgsl}"
    );
    // 5. The non-q8 per-cell path's preamble (a bare `let k = gid.x;`
    //    followed by `if (k >= cfg.slot_count)`) is NOT emitted —
    //    the q8 branch fully replaces it. The `k` identifier may still
    //    appear inside namespace-prelude functions injected by the
    //    composer, so we assert the no-q8 PREAMBLE specifically: no
    //    `let k = gid.x;` line at the kernel top level.
    let kernel_top = decay_wgsl
        .split("@compute @workgroup_size(64)")
        .nth(1)
        .unwrap_or("");
    assert!(
        !kernel_top.contains("let k = gid.x;"),
        "q8 path should replace the per-cell `k = gid.x` preamble; got:\n{kernel_top}"
    );
}

const TEMPLATE_WITH_GATE_Q8: &str = r#"
event Tick { }

@replayable
@gpu_amenable
event Hit {
  target: AgentId,
}

entity Knower : Agent {
  pos: vec3,
  vel: vec3,
}

mask BeliefStillFresh(target: Agent) when agents.beliefs_last_seen_tick(self, target) != world.tick

@materialized(on_event = [Hit])
@decay(per = tick, mode = sub, by = 1, gate = BeliefStillFresh)
@storage(packed_q8)
view confidence(observer: Agent, subject: Agent) -> u32 {
  initial: 0,
  on Hit { target: agent } where agent == subject { self += 1 }
  clamp: [0, 255],
}

physics Tickle @phase(per_agent) {
  on Tick {} {
    emit Hit { target: self }
  }
}

// @phase(per_agent), not @phase(post): the body reads `self`, and a
// non-per_agent phase dispatches PerEvent (no per-agent `agent_id` in
// scope), which the G2 well-formed check rejects (SelfRefInPerEventBody).
// per_agent binds `self`/`agent_id`. The rule only feeds beliefs_tick for
// the decay gate below — its phase doesn't affect what this test asserts
// (the decay kernel's packed-q8 predicate inlining). Mirrors the G2
// close-out's fix to the sibling decay_mode_sub_and_gate fixture.
physics SetSomeBelief @phase(per_agent) {
  on Tick {} {
    agents.set_beliefs_last_seen_tick(self, self, world.tick);
  }
}
"#;

#[test]
fn packed_q8_with_gate_inlines_predicate_per_byte() {
    let program = dsl_compiler::parse(TEMPLATE_WITH_GATE_Q8).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg_with_opts(
        &comp,
        dsl_compiler::cg::lower::LowerOpts {
            belief_state: true,
            ..Default::default()
        },
    )
    .expect("lower");
    let schedule = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let artifacts = dsl_compiler::cg::emit::emit_cg_program(&schedule.schedule, &cg)
        .expect("emit");
    let decay_wgsl = artifacts
        .wgsl_files
        .iter()
        .find(|(k, _)| k.starts_with("decay_"))
        .map(|(_, v)| v.as_str())
        .expect("decay kernel WGSL must be emitted");

    // 1. Still per-WORD preamble (gate doesn't change the dispatch shape).
    assert!(
        decay_wgsl.contains("let word_idx = gid.x + gid.y * 4194240u;"),
        "gate + q8 should keep per-WORD preamble; got:\n{decay_wgsl}"
    );
    // 2. The gate-mask BeliefStateColumn handle walk extends the kernel
    //    BGL with `beliefs_tick`.
    assert!(
        decay_wgsl.contains("beliefs_tick"),
        "expected beliefs_tick binding from gate-mask BeliefState handle walk; got:\n{decay_wgsl}"
    );
    // 3. The predicate inlines per-byte (inside the byte loop) so each
    //    cell's gate evaluates against its own (agent_id, per_pair_candidate)
    //    pair.
    assert!(
        decay_wgsl.contains("agents_beliefs_last_seen_tick(agent_id, per_pair_candidate)"),
        "expected per-cell binder mapping in predicate call; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("let decay_gate_value: bool"),
        "expected gate evaluation binding; got:\n{decay_wgsl}"
    );
    assert!(
        decay_wgsl.contains("if (decay_gate_value)"),
        "expected gate-wrapped step; got:\n{decay_wgsl}"
    );
    // 4. The saturating-sub formula still operates per-byte (on `conf`,
    //    not on the full `word`), inside the gate-wrapped block.
    assert!(
        decay_wgsl.contains("select(conf - by, 0u, conf < by)"),
        "expected per-byte saturating-sub inside gate-wrapped block; got:\n{decay_wgsl}"
    );
    // 5. No leftover TODO marker.
    assert!(
        !decay_wgsl.contains("TODO(decay-gate)"),
        "TODO marker should be retired; got:\n{decay_wgsl}"
    );
}

// ---- Resolve-side surface tests ----------------------------------------

const TEMPLATE_FOR_INVALID_STORAGE: &str = r#"
event Tick { }
event Hit { target: AgentId }
entity Particle : Agent { pos: vec3, vel: vec3 }

@materialized(on_event = [Hit])
STORAGE_PLACEHOLDER
view confidence(observer: Agent, subject: Agent) -> u32 {
  initial: 0,
  on Hit { target: agent } where agent == subject { self += 1 }
  clamp: [0, 255],
}

physics Tickle @phase(per_agent) {
  on Tick {} { emit Hit { target: self } }
}
"#;

fn try_resolve_with_storage(storage_ann: &str) -> Result<(), String> {
    let src = TEMPLATE_FOR_INVALID_STORAGE.replace("STORAGE_PLACEHOLDER", storage_ann);
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e}"))?;
    dsl_ast::resolve::resolve(program)
        .map(|_| ())
        .map_err(|e| format!("resolve: {e}"))
}

#[test]
fn storage_packed_q8_resolves() {
    try_resolve_with_storage("@storage(packed_q8)")
        .expect("@storage(packed_q8) should resolve cleanly");
}

#[test]
fn storage_unknown_packing_rejected() {
    let err = try_resolve_with_storage("@storage(packed_q3)")
        .expect_err("unknown packing must surface typed error");
    assert!(
        err.contains("packed_q3") && err.contains("unknown packing"),
        "expected unknown-packing error; got: {err}"
    );
}

#[test]
fn storage_with_keyed_arg_rejected() {
    let err = try_resolve_with_storage("@storage(form = packed_q8)")
        .expect_err("keyed arg must surface typed error");
    assert!(
        err.contains("positional"),
        "expected positional-only error; got: {err}"
    );
}

#[test]
fn storage_with_two_args_rejected() {
    let err = try_resolve_with_storage("@storage(packed_q8, packed_q4)")
        .expect_err("two args must surface typed error");
    assert!(
        err.contains("exactly one"),
        "expected exactly-one-arg error; got: {err}"
    );
}

#[test]
fn storage_omitted_resolves_with_default_none() {
    // No @storage annotation — should resolve cleanly (the field
    // defaults to Packing::None and the per-cell decay path lights up).
    try_resolve_with_storage("")
        .expect("no @storage annotation should resolve cleanly");
}
