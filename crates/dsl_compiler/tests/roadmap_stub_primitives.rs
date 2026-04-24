//! Grammar-frontload coverage for the 5 roadmap subsystems whose DSL
//! surface landed ahead of their runtime (Subsystems §1, §3, §6, §7, §12
//! of `docs/superpowers/roadmap.md`). These are parser + IR + resolver
//! stubs only — the physics / physics-WGSL emitters MUST return
//! `EmitError::Unsupported` until the corresponding runtime state
//! (memberships / relationships / theory-of-mind / groups / quests)
//! exists.
//!
//! Each subsystem has:
//!  1. a parse+resolve positive test using at least two of its primitives
//!     in a composite body, verifying the compiler accepts the surface
//!     and lifts the calls to `IrExpr::NamespaceCall`;
//!  2. an emitter test asserting the specific `Unsupported(...)` diagnostic
//!     fires — this catches accidental future over-emit (a subsystem
//!     implementation that lowers primitives before the runtime is wired).

use dsl_compiler::emit_physics::{emit_physics, EmitContext as CpuCtx, EmitError as CpuErr};
use dsl_compiler::emit_physics_wgsl::{
    emit_physics_wgsl, EmitContext as GpuCtx, EmitError as GpuErr,
};
use dsl_compiler::ir::{IrExpr, IrStmt, NamespaceId};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Walks a statement tree and returns every `IrExpr::NamespaceCall` whose
/// `ns` matches `target_ns`. Used to assert a body's primitives lifted
/// into the structured IR form rather than falling through as
/// `UnresolvedCall`.
fn collect_ns_calls(stmts: &[IrStmt], target_ns: NamespaceId) -> Vec<String> {
    fn visit_expr(
        expr: &dsl_compiler::ir::IrExprNode,
        target_ns: NamespaceId,
        out: &mut Vec<String>,
    ) {
        match &expr.kind {
            IrExpr::NamespaceCall { ns, method, args } => {
                if *ns == target_ns {
                    out.push(method.clone());
                }
                for a in args {
                    visit_expr(&a.value, target_ns, out);
                }
            }
            IrExpr::Binary(_, l, r) => {
                visit_expr(l, target_ns, out);
                visit_expr(r, target_ns, out);
            }
            IrExpr::Unary(_, r) => visit_expr(r, target_ns, out),
            IrExpr::BuiltinCall(_, args) => {
                for a in args {
                    visit_expr(&a.value, target_ns, out);
                }
            }
            IrExpr::If { cond, then_expr, else_expr } => {
                visit_expr(cond, target_ns, out);
                visit_expr(then_expr, target_ns, out);
                if let Some(e) = else_expr {
                    visit_expr(e, target_ns, out);
                }
            }
            IrExpr::Field { base, .. } => visit_expr(base, target_ns, out),
            IrExpr::ViewCall(_, args) | IrExpr::VerbCall(_, args) => {
                for a in args {
                    visit_expr(&a.value, target_ns, out);
                }
            }
            IrExpr::UnresolvedCall(_, args) => {
                for a in args {
                    visit_expr(&a.value, target_ns, out);
                }
            }
            _ => {}
        }
    }
    fn visit_stmt(stmt: &IrStmt, target_ns: NamespaceId, out: &mut Vec<String>) {
        match stmt {
            IrStmt::Expr(e) => visit_expr(e, target_ns, out),
            IrStmt::Let { value, .. } => visit_expr(value, target_ns, out),
            IrStmt::If { cond, then_body, else_body, .. } => {
                visit_expr(cond, target_ns, out);
                for s in then_body {
                    visit_stmt(s, target_ns, out);
                }
                if let Some(eb) = else_body {
                    for s in eb {
                        visit_stmt(s, target_ns, out);
                    }
                }
            }
            IrStmt::For { iter, filter, body, .. } => {
                visit_expr(iter, target_ns, out);
                if let Some(f) = filter {
                    visit_expr(f, target_ns, out);
                }
                for s in body {
                    visit_stmt(s, target_ns, out);
                }
            }
            IrStmt::Match { scrutinee, arms, .. } => {
                visit_expr(scrutinee, target_ns, out);
                for a in arms {
                    for s in &a.body {
                        visit_stmt(s, target_ns, out);
                    }
                }
            }
            IrStmt::Emit(e) => {
                for f in &e.fields {
                    visit_expr(&f.value, target_ns, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for s in stmts {
        visit_stmt(s, target_ns, &mut out);
    }
    out
}

/// Assert every method in `methods` resolves to an `IrExpr::NamespaceCall`
/// in `comp.physics[rule_name].handlers[0].body` under `ns`. Implicitly
/// verifies the resolver routed the call through the stdlib method
/// schema rather than falling back to `UnresolvedCall`.
fn assert_methods_resolved(
    comp: &dsl_compiler::ir::Compilation,
    rule_name: &str,
    ns: NamespaceId,
    expected_methods: &[&str],
) {
    let p = comp
        .physics
        .iter()
        .find(|p| p.name == rule_name)
        .unwrap_or_else(|| panic!("`{rule_name}` missing from IR"));
    let seen = collect_ns_calls(&p.handlers[0].body, ns);
    for method in expected_methods {
        assert!(
            seen.iter().any(|m| m == *method),
            "expected `{}::{}` as NamespaceCall under rule `{rule_name}`; saw {:?}",
            ns.name(),
            method,
            seen,
        );
    }
}

// ---------------------------------------------------------------------------
// Subsystem §1 — Memberships (roadmap.md:161-211)
// ---------------------------------------------------------------------------

const SUBSYSTEM_1_SRC: &str = r#"
@replayable event AgentDied { agent_id: AgentId }
event MembershipStubFired { agent_id: AgentId }

// Composite body exercising two of the four Membership primitives.
// `is_group_member(self, Family)` + `is_group_leader(self)` return bool;
// the `!` and `&&` operators compose them into a legal mask-style
// predicate. `Family` is an unresolved upper-case ident; the resolver
// treats it as an `EnumVariant` placeholder (`GroupKind` doesn't exist
// yet) — fine at 1a, resolved when Subsystem §1 lands the enum.
physics membership_grammar_stub @phase(event) {
  on AgentDied { agent_id: a } {
    if membership::is_group_member(a, Family) && !membership::is_group_leader(a) {
      emit MembershipStubFired { agent_id: a }
    }
  }
}
"#;

#[test]
fn subsystem_1_memberships_parses_and_resolves_with_ns_call_ir() {
    let comp = dsl_compiler::compile(SUBSYSTEM_1_SRC)
        .expect("membership:: primitives must parse + resolve cleanly");
    assert_methods_resolved(
        &comp,
        "membership_grammar_stub",
        NamespaceId::Membership,
        &["is_group_member", "is_group_leader"],
    );
}

#[test]
fn subsystem_1_memberships_emit_returns_unsupported_cpu_and_gpu() {
    let comp = dsl_compiler::compile(SUBSYSTEM_1_SRC).expect("compile OK");
    let p = comp
        .physics
        .iter()
        .find(|p| p.name == "membership_grammar_stub")
        .unwrap();

    // CPU emitter.
    let cpu_ctx = CpuCtx { events: &comp.events, event_tags: &comp.event_tags };
    let cpu = emit_physics(p, None, &cpu_ctx);
    let msg = match cpu {
        Err(CpuErr::Unsupported(s)) => s,
        other => panic!("expected CpuErr::Unsupported, got {other:?}"),
    };
    assert!(
        msg.contains("memberships primitive `membership::"),
        "CPU emit error msg should cite the memberships primitive stub; got: {msg}"
    );

    // GPU (WGSL) emitter.
    let gpu_ctx = GpuCtx { events: &comp.events, event_tags: &comp.event_tags };
    let gpu = emit_physics_wgsl(p, &gpu_ctx);
    let msg = match gpu {
        Err(GpuErr::Unsupported(s)) => s,
        other => panic!("expected GpuErr::Unsupported, got {other:?}"),
    };
    assert!(
        msg.contains("memberships primitive `membership::"),
        "GPU emit error msg should cite the memberships primitive stub; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Subsystem §3 — Relationships (roadmap.md:279-311)
// ---------------------------------------------------------------------------

const SUBSYSTEM_3_SRC: &str = r#"
@replayable event AgentDied { agent_id: AgentId }
event RelationshipStubFired { agent_id: AgentId, other_id: AgentId }

// `is_hostile(a, self)` and `knows_well(a, self)` exercise two of the
// three Relationship primitives. The `||` compose gates the follow-up
// emit — the kind of predicate Subsystem §3's gossip / retaliation
// rules will lean on once the cold_relationships SoA exists.
physics relationship_grammar_stub @phase(event) {
  on AgentDied { agent_id: a } {
    if relationship::is_hostile(a, a) || relationship::knows_well(a, a) {
      emit RelationshipStubFired { agent_id: a, other_id: a }
    }
  }
}
"#;

#[test]
fn subsystem_3_relationships_parses_and_resolves_with_ns_call_ir() {
    let comp = dsl_compiler::compile(SUBSYSTEM_3_SRC)
        .expect("relationship:: primitives must parse + resolve cleanly");
    assert_methods_resolved(
        &comp,
        "relationship_grammar_stub",
        NamespaceId::Relationship,
        &["is_hostile", "knows_well"],
    );
}

#[test]
fn subsystem_3_relationships_emit_returns_unsupported_cpu_and_gpu() {
    let comp = dsl_compiler::compile(SUBSYSTEM_3_SRC).expect("compile OK");
    let p = comp
        .physics
        .iter()
        .find(|p| p.name == "relationship_grammar_stub")
        .unwrap();

    let cpu_ctx = CpuCtx { events: &comp.events, event_tags: &comp.event_tags };
    let cpu = emit_physics(p, None, &cpu_ctx);
    let msg = match cpu {
        Err(CpuErr::Unsupported(s)) => s,
        other => panic!("expected CpuErr::Unsupported, got {other:?}"),
    };
    assert!(
        msg.contains("relationships primitive `relationship::"),
        "CPU emit error msg should cite the relationships primitive stub; got: {msg}"
    );

    let gpu_ctx = GpuCtx { events: &comp.events, event_tags: &comp.event_tags };
    let gpu = emit_physics_wgsl(p, &gpu_ctx);
    let msg = match gpu {
        Err(GpuErr::Unsupported(s)) => s,
        other => panic!("expected GpuErr::Unsupported, got {other:?}"),
    };
    assert!(
        msg.contains("relationships primitive `relationship::"),
        "GPU emit error msg should cite the relationships primitive stub; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Subsystem §6 — Theory-of-mind (roadmap.md:447-506)
// ---------------------------------------------------------------------------

const SUBSYSTEM_6_SRC: &str = r#"
@replayable event AgentDied { agent_id: AgentId }
event TomStubFired { agent_id: AgentId }

// `believes_knows(a, a, Combat)` + `can_deceive(a, a, SecretX)` compose
// a deception-gate predicate — representative of the gossip / intrigue
// mechanics Subsystem §6 unlocks. The `Combat` / `SecretX` idents
// resolve as typeless EnumVariant placeholders until the DomainId /
// FactId enums land.
physics theory_of_mind_grammar_stub @phase(event) {
  on AgentDied { agent_id: a } {
    if theory_of_mind::believes_knows(a, a, Combat)
        && theory_of_mind::can_deceive(a, a, SecretX) {
      emit TomStubFired { agent_id: a }
    }
  }
}
"#;

#[test]
fn subsystem_6_theory_of_mind_parses_and_resolves_with_ns_call_ir() {
    let comp = dsl_compiler::compile(SUBSYSTEM_6_SRC)
        .expect("theory_of_mind:: primitives must parse + resolve cleanly");
    assert_methods_resolved(
        &comp,
        "theory_of_mind_grammar_stub",
        NamespaceId::TheoryOfMind,
        &["believes_knows", "can_deceive"],
    );
}

#[test]
fn subsystem_6_theory_of_mind_emit_returns_unsupported_cpu_and_gpu() {
    let comp = dsl_compiler::compile(SUBSYSTEM_6_SRC).expect("compile OK");
    let p = comp
        .physics
        .iter()
        .find(|p| p.name == "theory_of_mind_grammar_stub")
        .unwrap();

    let cpu_ctx = CpuCtx { events: &comp.events, event_tags: &comp.event_tags };
    let cpu = emit_physics(p, None, &cpu_ctx);
    let msg = match cpu {
        Err(CpuErr::Unsupported(s)) => s,
        other => panic!("expected CpuErr::Unsupported, got {other:?}"),
    };
    assert!(
        msg.contains("theory_of_mind primitive `theory_of_mind::"),
        "CPU emit error msg should cite the theory_of_mind primitive stub; got: {msg}"
    );

    let gpu_ctx = GpuCtx { events: &comp.events, event_tags: &comp.event_tags };
    let gpu = emit_physics_wgsl(p, &gpu_ctx);
    let msg = match gpu {
        Err(GpuErr::Unsupported(s)) => s,
        other => panic!("expected GpuErr::Unsupported, got {other:?}"),
    };
    assert!(
        msg.contains("theory_of_mind primitive `theory_of_mind::"),
        "GPU emit error msg should cite the theory_of_mind primitive stub; got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Subsystem §7 — Groups (roadmap.md:510-574)
// ---------------------------------------------------------------------------

const SUBSYSTEM_7_SRC: &str = r#"
// Stub pattern event carrying a GroupId so the body has a typed binding
// to feed the `group::` predicates (without needing the real
// MembershipJoined / GroupDissolved events yet).
event GroupStubPattern { group_id: GroupId, cost: i64 }
event GroupStubFired { group_id: GroupId }

// `exists(g)` + `can_afford_from_treasury(g, cost)` compose the
// treasury-dispensing gate the group economics code will lean on.
physics group_grammar_stub @phase(event) {
  on GroupStubPattern { group_id: g, cost: c } {
    if group::exists(g) && group::can_afford_from_treasury(g, c) {
      emit GroupStubFired { group_id: g }
    }
  }
}
"#;

#[test]
fn subsystem_7_groups_parses_and_resolves_with_ns_call_ir() {
    let comp = dsl_compiler::compile(SUBSYSTEM_7_SRC)
        .expect("group:: primitives must parse + resolve cleanly");
    assert_methods_resolved(
        &comp,
        "group_grammar_stub",
        NamespaceId::Group,
        &["exists", "can_afford_from_treasury"],
    );
}

#[test]
fn subsystem_7_groups_emit_returns_unsupported_cpu_and_gpu() {
    let comp = dsl_compiler::compile(SUBSYSTEM_7_SRC).expect("compile OK");
    let p = comp
        .physics
        .iter()
        .find(|p| p.name == "group_grammar_stub")
        .unwrap();

    let cpu_ctx = CpuCtx { events: &comp.events, event_tags: &comp.event_tags };
    let cpu = emit_physics(p, None, &cpu_ctx);
    let msg = match cpu {
        Err(CpuErr::Unsupported(s)) => s,
        other => panic!("expected CpuErr::Unsupported, got {other:?}"),
    };
    assert!(
        msg.contains("groups primitive `group::"),
        "CPU emit error msg should cite the groups primitive stub; got: {msg}"
    );

    let gpu_ctx = GpuCtx { events: &comp.events, event_tags: &comp.event_tags };
    let gpu = emit_physics_wgsl(p, &gpu_ctx);
    let msg = match gpu {
        Err(GpuErr::Unsupported(s)) => s,
        other => panic!("expected GpuErr::Unsupported, got {other:?}"),
    };
    assert!(
        msg.contains("groups primitive `group::"),
        "GPU emit error msg should cite the groups primitive stub; got: {msg}"
    );
}
