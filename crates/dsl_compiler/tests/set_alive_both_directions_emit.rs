//! Pin: `agents.set_alive(<target>, <bool>)` lowers + emits in BOTH
//! directions — the kill transition (`set_alive(self, false)`) and the
//! revive transition (`set_alive(<target>, true)`).
//!
//! Closes the regression-test half of gap **T3** in
//! `docs/architecture/gaps_observed.md` ("Birth path doesn't propagate").
//!
//! Background
//! ----------
//!
//! Pre-fix surface: chronicle consumers in `trade_caravans.sim` used
//! `agents.set_alive(h, true)` to revive heir slots on inheritance.
//! The fixture pin reported `0/16 heirs revived` — which initially
//! looked like a one-direction setter allowlist or an `alive_pack`
//! kernel that compresses-out only (kill propagates, revive doesn't).
//!
//! Investigation
//! -------------
//!
//! - `agents_setter_field` in `cg/lower/physics.rs` accepts the method
//!   name `set_alive` and maps it to `AgentFieldId::Alive` regardless
//!   of value direction — no allowlist asymmetry.
//! - `lower_agents_setter` in `cg/lower/physics.rs` lowers BOTH `true`
//!   and `false` literals through the same `CgStmt::Assign` path —
//!   no directional bias in the IR lowering.
//! - `alive_pack` (the `PlumbingKind::AliveBitmap` body) reads
//!   `agent_alive[slot] != 0u` and packs into a 32-bit word — fully
//!   bidirectional. Re-runs after every tick, so a 0→1 flip lands in
//!   the bitmap on the next tick's pack pass.
//! - WGSL emit: kernels that contain `set_alive(_, false)` upgrade
//!   `agent_alive` to `array<atomic<u32>>` and emit the
//!   `atomicCompareExchangeWeak` CAS. Kernels that
//!   only contain `set_alive(_, true)` keep `array<u32>` and emit a
//!   plain indexed store (`agent_alive[idx] = select(0u, 1u, true);`).
//!   Both forms write to the SAME underlying buffer (wgpu doesn't
//!   distinguish atomic vs non-atomic at the BGL layer).
//!
//! After the chronicle-consumer Indirect-dispatch + live-event-tail
//! commits (`5d62f9b7`, `96d8c8ae`), the trade_caravans pin reports
//! `15/16 heirs revived` — the missing 1 is a fixture-side off-by-one
//! in the `engaged_with` `+1` sentinel encoding (Heir N's slot index
//! +1 lands on slot N+1; for N=15 that's a market slot which was
//! already alive). T3 is structurally fixed; this pin prevents a
//! future regression in the lowering.
//!
//! Pin shape
//! ---------
//!
//! Two minimal `.sim` sources, one per direction:
//!
//!   1. KILL (per_agent rule): `agents.set_alive(self, false)` →
//!      `agent_alive` declared `array<atomic<u32>>` AND
//!      `atomicCompareExchangeWeak(&agent_alive[<idx>], 1u, 0u)`
//!      appears in the kernel body.
//!   2. REVIVE (chronicle consumer): `agents.set_alive(t, true)` →
//!      `agent_alive` declared `array<u32>` (no atomic upgrade) AND
//!      `agent_alive[<idx>] = select(0u, 1u, true);` appears in the
//!      kernel body.
//!
//! A future regression that, e.g., gates the Alive setter on `false`
//! literals, drops the AliveBitmap repack on revives, or breaks the
//! plain-store emit shape would trip exactly one of these assertions.

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

/// `set_alive(self, false)` in a per_agent rule lowers to the
/// atomicCompareExchangeWeak kill-transition pattern. The
/// `agent_alive` binding is upgraded to `array<atomic<u32>>`.
#[test]
fn set_alive_self_false_emits_atomic_cas_kill_pattern() {
    let src = r#"
event Tick { }

@phase(per_agent)
physics Reaper {
  on Tick {} where (self.alive) {
    agents.set_alive(self, false);
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_wgsl(&art, "physics_Reaper");

    // 1. agent_alive binding upgraded to atomic storage.
    assert!(
        body.contains("var<storage, read_write> agent_alive: array<atomic<u32>>;"),
        "set_alive(self, false) must upgrade agent_alive to array<atomic<u32>>; got body:\n{body}",
    );

    // 2. The CAS itself: 1u → 0u (kill direction).
    assert!(
        body.contains("atomicCompareExchangeWeak(&agent_alive["),
        "expected atomicCompareExchangeWeak on agent_alive; got body:\n{body}",
    );
    assert!(
        body.contains(", 1u, 0u)"),
        "expected CAS args (1u, 0u) for the kill transition; got body:\n{body}",
    );

    // 3. NO plain `agent_alive[idx] = …` assign in the kill kernel —
    //    the CAS is the only valid lvalue write on an atomic.
    assert!(
        !body.contains("agent_alive[agent_id] ="),
        "kill kernel must not emit a plain agent_alive store; got body:\n{body}",
    );
}

/// `set_alive(<target>, true)` in a chronicle consumer (`@phase(post)`
/// + `on <event>`) lowers to a plain indexed store. The `agent_alive`
/// binding stays `array<u32>` (no atomic upgrade) because the kernel
/// body has no `set_alive(_, false)` pattern. **This is the load-bearing
/// half of gap T3** — heir-revive in `trade_caravans.sim::BornRevive`.
#[test]
fn set_alive_target_true_emits_plain_store_revive_pattern() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Born {
  agent: AgentId,
}

@phase(per_agent)
physics SeedBirth {
  on Tick {} where (self.alive) {
    emit Born { agent: self }
  }
}

@phase(post)
physics Revive {
  on Born { agent: h } {
    agents.set_alive(h, true);
  }
}
"#;
    let art = compile_inline(src);
    let body = kernel_wgsl(&art, "physics_Revive");

    // 1. agent_alive binding is PLAIN storage (no atomic upgrade) —
    //    the kernel only writes `true`, no kill pattern in this body.
    assert!(
        body.contains("var<storage, read_write> agent_alive: array<u32>;"),
        "set_alive(_, true)-only kernel must keep agent_alive as array<u32>; got body:\n{body}",
    );
    assert!(
        !body.contains("var<storage, read_write> agent_alive: array<atomic<u32>>;"),
        "revive-only kernel must NOT upgrade agent_alive to atomic; got body:\n{body}",
    );

    // 2. The plain store: `agent_alive[<idx>] = select(0u, 1u, true);`.
    //    The select(0u, 1u, <bool>) wrapper is the bool→u32 coercion
    //    in `wgsl_body.rs::lower_cg_stmt_to_wgsl`.
    assert!(
        body.contains("agent_alive["),
        "expected an agent_alive[<idx>] write; got body:\n{body}",
    );
    assert!(
        body.contains("] = select(0u, 1u, true);"),
        "expected `agent_alive[<idx>] = select(0u, 1u, true);` revive store; got body:\n{body}",
    );

    // 3. NO atomic CAS — the revive shape is a single-writer plain
    //    store; the CAS is only needed when contending threads race
    //    on the kill transition.
    assert!(
        !body.contains("atomicCompareExchangeWeak(&agent_alive["),
        "revive kernel must NOT emit a CAS on agent_alive; got body:\n{body}",
    );
}

/// Both directions in ONE compilation unit — the `Reaper` kernel keeps
/// its CAS upgrade, the `Revive` kernel keeps its plain store. Verifies
/// that per-kernel atomic-mode decisions are independent and don't
/// leak across kernel boundaries (the same `agent_alive` buffer is
/// bound by both kernels — wgpu allows this because the BGL stays
/// `bgl_storage(N, false)` for both).
///
/// This is the configuration `trade_caravans.sim` exercises in
/// production: `LifecycleReaper` (per_agent, kills merchant) +
/// `BornRevive` (chronicle consumer, revives heir).
#[test]
fn set_alive_kill_and_revive_coexist_in_one_program() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Died {
  who: AgentId,
}

@replayable @gpu_amenable
event Born {
  who: AgentId,
}

@phase(per_agent)
physics Reaper {
  on Tick {} where (self.alive) {
    agents.set_alive(self, false);
    emit Died { who: self }
  }
}

@phase(post)
physics OnDied {
  on Died { who: a } {
    emit Born { who: a }
  }
}

@phase(post)
physics Revive {
  on Born { who: h } {
    agents.set_alive(h, true);
  }
}
"#;
    let art = compile_inline(src);
    let reaper = kernel_wgsl(&art, "physics_Reaper");
    let revive = kernel_wgsl(&art, "physics_Revive");

    // Reaper kernel — atomic upgrade, CAS kill.
    assert!(
        reaper.contains("array<atomic<u32>>"),
        "Reaper kernel must upgrade agent_alive to atomic for the kill CAS; got:\n{reaper}",
    );
    assert!(
        reaper.contains("atomicCompareExchangeWeak(&agent_alive["),
        "Reaper kernel must emit the kill-CAS; got:\n{reaper}",
    );

    // Revive kernel — plain storage, plain store.
    assert!(
        revive.contains("var<storage, read_write> agent_alive: array<u32>;"),
        "Revive kernel must keep agent_alive plain; got:\n{revive}",
    );
    assert!(
        revive.contains("] = select(0u, 1u, true);"),
        "Revive kernel must emit the plain revive store; got:\n{revive}",
    );
    assert!(
        !revive.contains("atomicCompareExchangeWeak(&agent_alive["),
        "Revive kernel must NOT emit a CAS on agent_alive; got:\n{revive}",
    );
}

/// AliveBitmap (the `alive_pack` plumbing kernel) is bidirectional —
/// any program that mutates `agent_alive` (in either direction) gets
/// the bitmap repack scheduled. Pins this so a future "alive only ever
/// goes 1→0" optimisation that conditionally drops the pack on
/// revive-only kernels would trip.
#[test]
fn alive_pack_kernel_emitted_for_revive_only_program() {
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Born {
  who: AgentId,
}

@phase(per_agent)
physics SeedBirth {
  on Tick {} where (self.alive) {
    emit Born { who: self }
  }
}

@phase(post)
physics Revive {
  on Born { who: h } {
    agents.set_alive(h, true);
  }
}
"#;
    let art = compile_inline(src);
    // alive_pack is the plumbing kernel that re-packs the alive
    // bitmap from `agent_alive[*] != 0u`. It must run after every
    // alive-write kernel — including the revive-only path.
    let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("alive_pack")),
        "alive_pack plumbing kernel must be emitted when the program writes \
         agent_alive in any direction; got kernels: {names:?}",
    );

    // The bitmap pack body itself is direction-agnostic — packs the
    // CURRENT alive state regardless of how it changed. Confirm the
    // shape is the bidirectional `agent_alive[slot] != 0u` test (the
    // body is intentionally not a "one-direction compress").
    let pack = kernel_wgsl(&art, "alive_pack");
    assert!(
        pack.contains("agent_alive[slot]"),
        "alive_pack must read agent_alive[slot] for bidirectional repack; got:\n{pack}",
    );
    assert!(
        pack.contains("!= 0u"),
        "alive_pack must use `!= 0u` (bidirectional) — not a kill-only `== 0u`; got:\n{pack}",
    );
}
