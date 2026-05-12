//! Pin: `agents.set_creature_type(<target>, <new_type>)` lowers + emits
//! through the standard `AgentField` indexed-store path.
//!
//! Closes the regression-test half of **Gap 1** in
//! `docs/architecture/gaps_pirate_fleet.md` ("ownership-transfer
//! primitive silently dropped at lower time because `set_creature_type`
//! is not in the setter allowlist").
//!
//! Background
//! ----------
//!
//! Pre-fix surface: `pirate_fleet.sim`'s `AttemptOwnershipFlip`
//! chronicle consumer body was a single `agents.set_creature_type(t,
//! nt)` expression statement. The `agents_setter_field` allowlist in
//! `cg/lower/physics.rs` did not recognise the method name, so the
//! lowering surfaced `UnsupportedPhysicsStmt { ast_label: "Expr" }`.
//! The build_helper tolerates lower errors → the entire rule body was
//! dropped and no `physics_AttemptOwnershipFlip` kernel emitted. Pin
//! reported `ownership_transfers = 0` despite 500 ticks of boarding
//! cadence.
//!
//! Investigation
//! -------------
//!
//! - `AgentFieldId::CreatureType` exists in
//!   `crates/dsl_compiler/src/cg/data_handle.rs:286` and has type
//!   `OptEnumU32` — which `agent_field_wgsl_ty` maps to plain
//!   `array<u32>` (kernel.rs:1518).
//! - There is no per-kernel atomic upgrade for `agent_creature_type`
//!   (the upgrade path is gated only on `Alive` for the kill-CAS and
//!   on the f32 RMW bitset for the f32 columns — see
//!   `cg/emit/kernel.rs:631+`). The column lands as plain storage.
//! - The `CgStmt::Assign{ DataHandle::AgentField{...}, value }` arm in
//!   `cg/emit/wgsl_body.rs:1962+` produces a plain `agent_<field>[idx]
//!   = <value>;` store. This is the same shape every other u32 SoA
//!   write produces (e.g. `set_stun_expires_at_tick`,
//!   `set_busy_until_tick`, `set_disguise_fake_type`).
//!
//! The gap-doc note about needing `atomicStore` was based on a
//! misreading of `kernel.rs:1462` (which is the `BeliefStateColumn::
//! CreatureType`, not the `AgentField`). The real `AgentField`
//! creature_type column lifts onto the existing plain-store path with
//! no new lowering required — the fix is a single allowlist arm.
//!
//! Pin shape
//! ---------
//!
//! Two assertions on the emitted `physics_AttemptOwnershipFlip`
//! kernel:
//!
//!   1. The kernel exists (the pre-fix bug was that lowering failed
//!      and no kernel was emitted at all).
//!   2. The kernel body contains the plain indexed store
//!      `agent_creature_type[<idx>] = <value>;` — the same shape
//!      `set_stun_expires_at_tick` etc. emit on plain `array<u32>`
//!      bindings.
//!
//! A future regression that, e.g., drops the allowlist arm or routes
//! creature_type through a different (atomic) write path without
//! upgrading the binding accordingly would trip exactly one of these
//! assertions.

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

fn kernel_wgsl<'a>(
    art: &'a EmittedArtifacts,
    needle: &str,
) -> &'a str {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle) && name.ends_with(".wgsl"))
        .map(|(_, body)| body.as_str())
        .unwrap_or_else(|| {
            let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
            panic!("no kernel matched {needle:?}; emitted kernels: {names:?}")
        })
}

/// `agents.set_creature_type(t, 1u)` in a chronicle consumer lowers to
/// a plain indexed store on the `agent_creature_type` SoA column.
/// **This is the load-bearing assertion of Gap 1** — pre-fix, the rule
/// body was dropped at lower time and no kernel emitted at all.
#[test]
fn set_creature_type_chronicle_consumer_emits_indexed_store() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Boarded {
  target: AgentId,
}

@phase(per_agent)
physics SeedBoard {
  on Tick {} where (self.alive) {
    emit Boarded { target: self }
  }
}

@phase(post)
physics AttemptOwnershipFlip {
  on Boarded { target: t } {
    agents.set_creature_type(t, 1u);
  }
}
"#;
    let art = compile_inline(src);

    // 1. The kernel must exist. Pre-fix, the lowering silently dropped
    //    the rule body (UnsupportedPhysicsStmt { ast_label: "Expr" })
    //    and no `physics_AttemptOwnershipFlip` kernel was emitted.
    let body = kernel_wgsl(&art, "physics_AttemptOwnershipFlip");

    // 2. The agent_creature_type binding stays plain `array<u32>` —
    //    `OptEnumU32` lowers to plain storage and there is no atomic
    //    upgrade for this column today (cf. the alive-CAS and f32-RMW
    //    upgrades in `cg/emit/kernel.rs:631+`).
    assert!(
        body.contains("var<storage, read_write> agent_creature_type: array<u32>;"),
        "agent_creature_type binding must be plain array<u32>; got body:\n{body}",
    );

    // 3. The plain indexed store: `agent_creature_type[<idx>] = …;`.
    //    The exact rhs shape (`1u` literal) is renderer-dependent;
    //    asserting only the lvalue + the trailing `=` keeps this pin
    //    tolerant of incidental rhs reshaping while still trapping the
    //    "no store at all" / "wrong column" regressions.
    assert!(
        body.contains("agent_creature_type["),
        "expected an agent_creature_type[<idx>] = <value> store; got body:\n{body}",
    );
    assert!(
        body.contains("] = 1u;"),
        "expected the literal `1u` on the rhs of the store; got body:\n{body}",
    );

    // 4. NO atomic store / CAS — creature_type lives on a plain
    //    `array<u32>` binding, so naked atomicStore would fail WGSL
    //    type checking. This guard protects against a future "upgrade
    //    creature_type to atomic but forget the binding-side upgrade"
    //    half-fix.
    assert!(
        !body.contains("atomicStore(&agent_creature_type"),
        "creature_type write must not emit atomicStore on a plain array<u32> binding; got body:\n{body}",
    );
    assert!(
        !body.contains("atomicCompareExchangeWeak(&agent_creature_type"),
        "creature_type write must not emit a CAS on a plain array<u32> binding; got body:\n{body}",
    );
}

/// `agents.set_creature_type(self, …)` from a per_agent rule body
/// lowers through `AgentRef::Self_` and writes into
/// `agent_creature_type[agent_id]`. Pins the per-agent (non-chronicle)
/// path — the pirate_fleet pin only needs the chronicle path, but the
/// allowlist arm is direction-agnostic and a per_agent surface might
/// land in a future fixture (e.g. faction defection on a self-trigger).
#[test]
fn set_creature_type_per_agent_self_emits_self_indexed_store() {
    let src = r#"
event Tick { }

@phase(per_agent)
physics Defect {
  on Tick {} where (self.alive) {
    agents.set_creature_type(self, 0u);
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_wgsl(&art, "physics_Defect");

    // Plain binding, plain self-indexed store (`agent_id` is the
    // per-agent kernel's bound thread index).
    assert!(
        body.contains("var<storage, read_write> agent_creature_type: array<u32>;"),
        "per-agent set_creature_type must keep agent_creature_type plain; got body:\n{body}",
    );
    assert!(
        body.contains("agent_creature_type[agent_id] = 0u;"),
        "expected `agent_creature_type[agent_id] = 0u;` self-store; got body:\n{body}",
    );
}
