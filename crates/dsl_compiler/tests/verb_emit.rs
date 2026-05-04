//! Integration: a `verb` declaration round-trips through the full
//! parse → resolve → CG-lower → schedule → emit pipeline and produces
//! the synthesised mask + scoring artifacts the verb-expand pre-pass
//! injects.
//!
//! Slice A gate of the verb / probe / metric emit plan
//! (`docs/superpowers/plans/2026-05-03-verb-probe-metric-emit.md`).
//!
//! Mirrors the precedent set by
//! `crates/dsl_compiler/tests/predator_prey_invariant.rs` —
//! synthetic-fixture-via-inline-DSL because the aspirational
//! verb-bearing assets (`assets/sim/predator_prey.sim`,
//! `crowd_navigation.sim`) don't parse end-to-end today; the
//! lowering pre-pass has no per-fixture branching, so an inline
//! string exercises the same codepath every shipped runtime
//! eventually hits when a runtime-bound `*_min.sim` grows a verb.

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::op::ComputeOpKind;
use dsl_compiler::cg::program::CgProgram;

/// Drive an inline `.sim` source through parse → resolve → CG-lower
/// → schedule → emit and return both the CG program (for op-level
/// assertions) and the artifacts. Panics with the original error on
/// any failure so the test failure points at the pipeline gap rather
/// than an opaque panic.
fn compile_inline(src: &str) -> (CgProgram, EmittedArtifacts) {
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let cg = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(outcome) => {
            // Surface the diagnostic list so failures pinpoint the
            // first defect rather than just the program shape.
            for diag in &outcome.diagnostics {
                eprintln!("[lower diagnostic] {diag}");
            }
            outcome.program
        }
    };
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit");
    (cg, art)
}

/// Slice A gate: a `verb` with `when` and `score` clauses produces
/// a synthesised mask kernel AND a scoring entry under the
/// `verb_<name>` synthetic identifier. The verb has no `emit` so no
/// cascade SKIP diagnostic surfaces.
#[test]
fn verb_emit_expands_mask_and_scoring() {
    // Pre-existing scoring block + verb. The verb's score expression
    // appends a `verb_Wait = 0.75` row onto the existing block so the
    // existing scoring lowering pipeline picks it up.
    let src = r#"
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
scoring {
  Hold = 0.5
}
verb Wait(self) =
  action Hold
  when  self.alive
  score 0.75
"#;
    let (cg, art) = compile_inline(src);
    assert!(
        !art.kernel_index.is_empty(),
        "expected kernels for a verb-bearing fixture; got none",
    );

    // Op-level assertion 1: the verb expansion injected a MaskPredicate
    // op (the synthesised `verb_Wait` mask). Without verb expansion
    // the program would have ZERO MaskPredicate ops (the source
    // declares no top-level `mask`).
    let mask_ops = cg
        .ops
        .iter()
        .filter(|op| matches!(op.kind, ComputeOpKind::MaskPredicate { .. }))
        .count();
    assert_eq!(
        mask_ops, 1,
        "expected exactly one MaskPredicate op (the verb's synthesised mask); \
         got {} ops in {:?}",
        mask_ops,
        cg.ops.iter().map(|op| format!("{:?}", op.kind)).collect::<Vec<_>>(),
    );

    // Op-level assertion 2: the scoring kernel grew a row for the
    // verb (pre-expansion: 1 row `Hold`; post-expansion: 2 rows
    // `Hold` + `verb_Wait`). The argmax kernel reads both.
    let scoring_op = cg
        .ops
        .iter()
        .find_map(|op| match &op.kind {
            ComputeOpKind::ScoringArgmax { rows, .. } => Some(rows.len()),
            _ => None,
        })
        .expect("expected one ScoringArgmax op");
    assert_eq!(
        scoring_op, 2,
        "expected scoring rows = 2 (`Hold` + verb-injected `verb_Wait`); got {scoring_op}",
    );

    // Artifact-level assertion: the synthesised mask kernel name
    // carries the `verb_<name>` prefix — `verb_Wait` in this
    // fixture. Mask kernel naming varies per emitter (some carry a
    // `fused_mask_` prefix); accept any kernel that mentions
    // `verb_Wait`.
    let any_verb_artifact = art
        .kernel_index
        .iter()
        .any(|name| name.contains("verb_Wait"))
        || art
            .wgsl_files
            .iter()
            .any(|(name, body)| name.contains("verb_Wait") || body.contains("verb_Wait"))
        || art
            .rust_files
            .iter()
            .any(|(_name, body)| body.contains("verb_Wait"));
    assert!(
        any_verb_artifact,
        "expected at least one emitted artifact (kernel / wgsl / rust file) \
         to mention `verb_Wait`; kernels = {:?}",
        art.kernel_index,
    );
    eprintln!(
        "[verb_Wait] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

/// Closes the scoring-kernel ActionSelected emit gap. When a verb
/// declares a non-empty `emit`, the verb expander injects a singleton
/// `ActionSelected` event kind. The driver then records an
/// `EventRing { Append }` write on every `ScoringArgmax` op (via
/// `wire_action_selected_writes`) so the binding scanner declares
/// `event_ring` + `event_tail` bindings on the scoring kernel; the
/// scoring kernel's WGSL body
/// (`cg::emit::kernel::lower_scoring_argmax_body`) inlines the
/// per-agent ring-append after argmax.
///
/// Asserts the round-trip on the CG program (not the schedule):
/// today's stress fixtures don't ship a scoring decl that lands a
/// `scoring` kernel through the full schedule synth, so we drive the
/// kernel emit directly via [`kernel_topology_to_spec_and_body`] for
/// the ScoringArgmax op produced by the lowering.
#[test]
fn scoring_kernel_emits_action_selected_when_verb_present() {
    use dsl_compiler::cg::dispatch::DispatchShape;
    use dsl_compiler::cg::emit::kernel_topology_to_spec_and_body;
    use dsl_compiler::cg::emit::wgsl_body::{EmitCtx, HandleNamingStrategy};
    use dsl_compiler::cg::schedule::synthesis::KernelTopology;

    fn build_emit_ctx<'a>(prog: &'a CgProgram) -> EmitCtx<'a> {
        EmitCtx {
            prog,
            naming: HandleNamingStrategy::Structural,
            tile_walk_index: std::cell::RefCell::new(None),
            dispatch: std::cell::Cell::new(None),
            view_target_locals: std::cell::RefCell::new(Vec::new()),
            pending_target_lets: std::cell::RefCell::new(Vec::new()),
            bound_target_exprs: std::cell::RefCell::new(std::collections::HashSet::new()),
            event_ring_atomic_loads: std::cell::Cell::new(false),
        }
    }

    fn scoring_op_index(cg: &CgProgram) -> usize {
        cg.ops
            .iter()
            .position(|op| matches!(op.kind, ComputeOpKind::ScoringArgmax { .. }))
            .expect("expected exactly one ScoringArgmax op in the lowered program")
    }

    // -- Negative half: verb without `emit` --
    //
    // The verb expander does NOT inject ActionSelected for an
    // emit-less verb, so the scoring kernel stays binding-clean.
    // Confirms the gating works.
    let no_emit_src = r#"
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
scoring {
  Hold = 0.5
}
verb Wait(self) =
  action Hold
  when  self.alive
  score 0.75
"#;
    let (cg_no_emit, _) = compile_inline(no_emit_src);
    let scoring_idx = scoring_op_index(&cg_no_emit);
    let scoring_writes = &cg_no_emit.ops[scoring_idx].writes;
    let has_event_ring_write = scoring_writes.iter().any(|h| {
        matches!(
            h,
            dsl_compiler::cg::data_handle::DataHandle::EventRing { .. }
        )
    });
    assert!(
        !has_event_ring_write,
        "no-verb-emit scoring op must not record EventRing-Append; got writes={:?}",
        scoring_writes,
    );
    let topology_no = KernelTopology::Split {
        op: dsl_compiler::cg::op::OpId(scoring_idx as u32),
        dispatch: DispatchShape::PerAgent,
    };
    let ctx_no = build_emit_ctx(&cg_no_emit);
    let (_spec_no, body_no) =
        kernel_topology_to_spec_and_body(&topology_no, &cg_no_emit, &ctx_no)
            .expect("scoring kernel emits cleanly without verb-emit");
    assert!(
        !body_no.contains("ActionSelected"),
        "no-verb-emit fixture must not emit ActionSelected; body: {body_no}",
    );
    assert!(
        !body_no.contains("event_tail"),
        "no-verb-emit fixture's scoring kernel must not reference event_tail; body: {body_no}",
    );

    // -- Positive half: verb with `emit` --
    //
    // ActionSelected event is injected; the driver records an
    // EventRing-Append on the ScoringArgmax op; the kernel emit
    // inlines the ring-append.
    let emit_src = r#"
event Killed { by: AgentId, prey: AgentId, pos: vec3, }
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
scoring {
  AttackTarget = 1.0
}
verb Strike(self, target: Agent) =
  action AttackTarget(target: target)
  when  self.alive
  emit  Killed { by: self, prey: target, pos: self.pos }
  score 1.0
"#;
    let prog = dsl_compiler::parse(emit_src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    // Tolerate downstream diagnostics — chronicle physics rule's
    // body may surface UnsupportedFieldBase on `self.pos`. The
    // ScoringArgmax op + ActionSelected event still land in the
    // best-effort program.
    let cg_emit = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(outcome) => outcome.program,
    };

    // Driver-wired write must be present.
    let scoring_idx = scoring_op_index(&cg_emit);
    let scoring_writes = &cg_emit.ops[scoring_idx].writes;
    let event_ring_writes: Vec<_> = scoring_writes
        .iter()
        .filter_map(|h| match h {
            dsl_compiler::cg::data_handle::DataHandle::EventRing { ring, kind } => {
                Some((*ring, *kind))
            }
            _ => None,
        })
        .collect();
    assert!(
        event_ring_writes
            .iter()
            .any(|(_, k)| matches!(
                k,
                dsl_compiler::cg::data_handle::EventRingAccess::Append
            )),
        "verb-emit scoring op must record EventRing-Append; got writes={:?}",
        scoring_writes,
    );

    // Drive the kernel-emit directly so we don't depend on the
    // scheduler routing the scoring op into a stage.
    let topology = KernelTopology::Split {
        op: dsl_compiler::cg::op::OpId(scoring_idx as u32),
        dispatch: DispatchShape::PerAgent,
    };
    let ctx = build_emit_ctx(&cg_emit);
    let (spec, body) = kernel_topology_to_spec_and_body(&topology, &cg_emit, &ctx)
        .expect("scoring kernel emits cleanly with verb-emit");
    // Bindings: scanner must declare event_ring + event_tail (the
    // emit body references both unconditionally when a verb cascade
    // is present).
    let binding_names: Vec<&str> = spec
        .bindings
        .iter()
        .map(|b| b.name.as_str())
        .collect();
    assert!(
        binding_names.contains(&"event_ring"),
        "scoring kernel must declare event_ring binding; got {:?}",
        binding_names,
    );
    assert!(
        binding_names.contains(&"event_tail"),
        "scoring kernel must declare event_tail binding; got {:?}",
        binding_names,
    );

    // Argmax skeleton still present.
    assert!(
        body.contains("scoring_output[scoring_base + 0u] = best_action;"),
        "argmax skeleton intact; body: {body}",
    );
    // ActionSelected ring-append shape — mirrors lower_emit_to_wgsl.
    assert!(
        body.contains("// emit ActionSelected"),
        "missing ActionSelected emit comment; body: {body}",
    );
    assert!(
        body.contains("atomicAdd(&event_tail[0], 1u)"),
        "missing event_tail atomicAdd; body: {body}",
    );
    assert!(
        body.contains("agent_id);"),
        "missing agent_id store (actor); body: {body}",
    );
    assert!(
        body.contains("best_action);"),
        "missing best_action store (action_id); body: {body}",
    );
    assert!(
        body.contains("best_target);"),
        "missing best_target store (target); body: {body}",
    );
}

/// A verb with a non-empty `emit` clause expands its cascade to a
/// synthesised `verb_chronicle_<name>` physics rule listening on the
/// (auto-injected) `ActionSelected` event. The expansion no longer
/// surfaces a `VerbExpansionSkipped` diagnostic for the cascade
/// stage — closed 2026-05-03 (see `verb_expand.rs` module docs).
///
/// Other diagnostics MAY surface for downstream lowering of the
/// emit body (e.g., `UnsupportedFieldBase` if the verb references
/// `<bound>.<field>` against the synthesised binder), but the
/// cascade-injection gap itself is gone.
#[test]
fn verb_emit_expands_cascade_no_skip_diagnostic() {
    let src = r#"
event Killed { by: AgentId, prey: AgentId, pos: vec3, }
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
scoring {
  AttackTarget = 1.0
}
verb Strike(self, target: Agent) =
  action AttackTarget(target: target)
  when  self.alive
  emit  Killed { by: self, prey: target, pos: self.pos }
  score 1.0
"#;
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let outcome = lower_compilation_to_cg(&comp);
    // Driver may still return Err if downstream stages flag other
    // unrelated issues (e.g., the verb's positional mask shape, the
    // emit body's `self.pos` access). Tolerate both Ok and Err — we
    // only assert NO `VerbExpansionSkipped` surfaces.
    let diagnostics: Vec<_> = match outcome {
        Ok(_) => Vec::new(),
        Err(o) => o.diagnostics,
    };
    let skipped: Vec<_> = diagnostics
        .iter()
        .filter(|d| {
            matches!(
                d,
                dsl_compiler::cg::lower::LoweringError::VerbExpansionSkipped {
                    reason:
                        dsl_compiler::cg::lower::VerbSkipReason::CascadeNeedsActionEvent,
                    ..
                }
            )
        })
        .collect();
    assert!(
        skipped.is_empty(),
        "expected zero VerbExpansionSkipped diagnostics; got {:?}",
        skipped,
    );
}

/// Gap #1 regression (verb-fire probe, 2026-05-04): a verb with no
/// `target` param (e.g., `verb Pray(self) = action PrayAction ...`)
/// must still synthesise a `verb_chronicle_<name>` rule whose body's
/// `if action_id == <lit>` lowers cleanly. Before the
/// `fresh_local_after` -> stateful `fresh_local` fix in
/// `cg/lower/verb_expand.rs::synthesize_cascade_physics`, the
/// fallback LocalRef for the absent `target` and the freshly
/// allocated `action_id` LocalRef collided on the same id; the
/// pattern-binding-let synthesiser then recorded `action_id: U32`
/// followed by `target: AgentId` against the same local, retyping
/// `action_id` to AgentId, which made the cascade's gate fail to
/// lower with `BinaryOperandTyMismatch (lhs: agent_id, rhs: u32)` —
/// the entire `verb_chronicle_Pray` rule was dropped silently.
///
/// This test asserts: zero `BinaryOperandTyMismatch` diagnostics
/// for a target-less verb, AND the `verb_chronicle_<name>` physics
/// rule survives lowering (was previously dropped).
#[test]
fn target_less_verb_chronicle_lowers_without_operand_ty_mismatch() {
    let src = r#"
event Tick { }
event Prayed { who: AgentId, }
entity Agent_ : Agent { alive: bool, }
scoring {
  PrayAction = 1.0
}
verb Pray(self) =
  action PrayAction
  when  self.alive
  emit  Prayed { who: self }
  score 1.0
"#;
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let outcome = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp);
    let (program, diagnostics) = match outcome {
        Ok(p) => (p, Vec::new()),
        Err(o) => (o.program, o.diagnostics),
    };
    let mismatches: Vec<_> = diagnostics
        .iter()
        .filter(|d| {
            matches!(
                d,
                dsl_compiler::cg::lower::LoweringError::BinaryOperandTyMismatch { .. }
            )
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "expected zero BinaryOperandTyMismatch on target-less verb cascade; got {:?}",
        mismatches,
    );
    // The verb_chronicle_Pray rule must survive lowering all the way
    // to a named physics-rule entry in the interner — before the fix
    // the body failed to lower and the entire rule was dropped.
    let chronicle_present = program
        .interner
        .physics_rules
        .values()
        .any(|n| n == "verb_chronicle_Pray");
    assert!(
        chronicle_present,
        "expected verb_chronicle_Pray to survive lowering; \
         interner.physics_rules: {:?}",
        program.interner.physics_rules.values().collect::<Vec<_>>(),
    );
}

/// Slice 5.7 acceptance probe: with the let-bound-AgentId-local
/// field-access fix landed (Phase 7 follow-up to commit `38c4c969`),
/// the verb cascade's body `pos: self.pos` no longer surfaces an
/// `UnsupportedFieldBase` defect. The synthesised
/// `verb_chronicle_Strike` physics handler binds `actor` as a
/// let-bound LocalRef of type `AgentId`; the body's `self.pos` lowers
/// to `Read(AgentField { Pos, AgentRef::Target(<actor_read>) })` —
/// semantically identical to authoring `agents.pos(self)` by hand —
/// and the emitted WGSL hoists `let target_expr_<N>: u32 = local_<M>;`
/// followed by `agent_pos[target_expr_<N>]`.
///
/// The runtime-side closure (the scoring kernel writing
/// `ActionSelected` per tick when a verb's row wins the per-agent
/// argmax) is a separate slice; this test only asserts the compiler
/// no longer trips on the body shape.
#[test]
fn verb_cascade_self_pos_lowers_without_unsupported_field_base() {
    let src = r#"
event Killed { by: AgentId, prey: AgentId, pos: vec3, }
event Tick { }
entity Agent_ : Agent { pos: vec3, alive: bool, }
scoring {
  AttackTarget = 1.0
}
verb Strike(self, target: Agent) =
  action AttackTarget(target: target)
  when  self.alive
  emit  Killed { by: self, prey: target, pos: self.pos }
  score 1.0
"#;
    let prog = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve");
    let outcome = lower_compilation_to_cg(&comp);
    let diagnostics: Vec<_> = match &outcome {
        Ok(_) => Vec::new(),
        Err(o) => o.diagnostics.clone(),
    };
    let unsupported_field_base: Vec<_> = diagnostics
        .iter()
        .filter(|d| {
            matches!(
                d,
                dsl_compiler::cg::lower::LoweringError::UnsupportedFieldBase { .. }
            )
        })
        .collect();
    assert!(
        unsupported_field_base.is_empty(),
        "expected zero UnsupportedFieldBase diagnostics from the verb cascade body; got {:?}",
        unsupported_field_base,
    );

    // Probe the emitted WGSL: the synthesised `verb_chronicle_Strike`
    // physics kernel should reference `agent_pos[target_expr_<N>]`
    // (the slice-1 stmt-prefix hoisting shape) for `pos: self.pos`.
    // This confirms the lowering produced the cross-agent Target-let
    // form rather than a bare `agent_pos[gid]` (the Self_ shape) or a
    // `agent_pos[per_pair_candidate]` (the per-pair shape).
    // The driver may still surface unrelated downstream diagnostics
    // (e.g., the verb's positional mask shape needing a `from`
    // clause — Task 2.6 gap). Tolerate the partial-Err path and use
    // `outcome.program` so emit + WGSL probing still proceeds.
    let cg = match outcome {
        Ok(cg) => cg,
        Err(o) => o.program,
    };
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, &cg).expect("emit");
    let chronicle_wgsl = art
        .wgsl_files
        .iter()
        .find(|(name, _)| name.contains("verb_chronicle_Strike"))
        .map(|(_, body)| body.clone())
        .expect("expected verb_chronicle_Strike WGSL kernel emitted");
    // The slice-1 stmt-prefix hoisting pattern: the body's `self.pos`
    // read produces a `let target_expr_<N>: u32 = local_<M>;` line
    // (where `local_<M>` is the actor binder from the event-pattern
    // destructuring) followed by `agent_pos[target_expr_<N>]`. This
    // confirms the lowering routed `self.pos` through
    // `AgentRef::Target(<read of let-bound actor>)` rather than the
    // bare `agent_pos[gid]` shape (which would silently address the
    // implicit kernel-row identity).
    assert!(
        chronicle_wgsl.contains("target_expr_") && chronicle_wgsl.contains("agent_pos["),
        "expected `agent_pos[target_expr_<N>]` cross-agent read in chronicle WGSL; got:\n{chronicle_wgsl}",
    );
}
