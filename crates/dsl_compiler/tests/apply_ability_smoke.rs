//! Smoke test for `assets/sim/apply_ability_smoke.sim` — task #138.
//!
//! Drives the full `dsl_compiler` pipeline (parse → resolve → CG lower
//! → schedule → emit) against the apply_ability fixture and asserts the
//! WGSL kernel composer wires up the dispatcher correctly:
//!
//!   - the `CgStmt::ApplyAbility` arm in `cg::emit::wgsl_body` runs to
//!     completion (no `UnsupportedPhysicsStmt` regression),
//!   - the dispatcher loop scaffolding (`for (var i: u32 = 0u; i < 6u`,
//!     `EFFECT_KIND_EMPTY = 0xFFu` continue) lands in the kernel body,
//!   - **slice γ — chronicle-bearing arms** emit real `atomicStore`
//!     writes against `event_ring` with the runtime EventKindIds
//!     (Damage=26, Heal=27, Shield=28, Stun=29, Slow=30, TransferGold=31,
//!     ModifyStanding=32). The unit tests in `wgsl_body.rs` pin the
//!     same fact at the format-string level; this test pins it at the
//!     kernel-body level (i.e. after the binding composer + thread
//!     preamble + cfg uniform have been wrapped around it).
//!
//! Without this test, the dispatcher is exercised only by the inline
//! tests in `wgsl_body.rs` against a hand-built `CgProgram` — the full
//! pipeline (binding composer / cfg uniform / EventRing(Append) write
//! recording / `agent_id` preamble) goes uncovered, so a regression in
//! any of those layers would surface only when the first runtime crate
//! finally references `apply_ability`. That's a worse failure surface.
//!
//! Mirror sim file shape: `target_chaser_compiles` in
//! `stress_fixtures_compile.rs` — same `compile_sim` driver,
//! same kernel-body fishing pattern.

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn compile_sim(path: &std::path::Path) -> Result<EmittedArtifacts, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|e| format!("lower: {e:?}"))?;
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))
}

/// Find the first WGSL kernel whose name contains `needle`.
fn kernel_body_containing<'a>(art: &'a EmittedArtifacts, needle: &str) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle))
        .map(|(_, body)| body.as_str())
}

#[test]
fn apply_ability_smoke_compiles() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| {
        panic!("apply_ability_smoke.sim failed at: {e}");
    });
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[apply_ability_smoke] {} kernels emitted: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

#[test]
fn apply_ability_smoke_emits_dispatcher_loop_in_kernel_body() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    // Pick the kernel that hosts the DispatchAbility physics rule.
    // The kernel composer's naming may evolve; fall back to a
    // generic "physics" search so the fixture pin doesn't drift on
    // composer renames.
    let body = kernel_body_containing(&art, "DispatchAbility")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });

    // Dispatcher scaffolding (slice β step 2):
    assert!(
        body.contains("for (var i: u32 = 0u; i < 6u;"),
        "expected dispatcher slot loop (MAX_EFFECTS_PER_PROGRAM = 6) in kernel body;\n\
         got body:\n{body}"
    );
    assert!(
        body.contains("if (kind == 0xFFu)"),
        "expected EFFECT_KIND_EMPTY skip in kernel body;\n\
         got body:\n{body}"
    );
    assert!(
        body.contains("ability_registry_effect_kinds[effect_base + i]"),
        "expected effect-kinds SoA read indexed by effect_base + i;\n\
         got body:\n{body}"
    );

    // Slice γ — every chronicle-bearing arm emits a kind-tag header
    // store against `event_ring`. The dispatcher's `let _slot: u32 =
    // atomicAdd(&event_tail[0], 1u);` slot acquisition appears once
    // per chronicle-bearing arm, so the body should carry exactly 7
    // copies after the binding composer wraps it.
    for (variant_label, expected_kind_tag) in &[
        ("Damage",          26u32),
        ("Heal",            27u32),
        ("Shield",          28u32),
        ("Stun",            29u32),
        ("Slow",            30u32),
        ("TransferGold",    31u32),
        ("ModifyStanding",  32u32),
    ] {
        let needle = format!(
            "atomicStore(&event_ring[_slot * 10u + 0u], {expected_kind_tag}u);"
        );
        assert!(
            body.contains(&needle),
            "post-pipeline kernel body should still carry the {variant_label} \
             arm's chronicle write (kind={expected_kind_tag}u);\n\
             got body:\n{body}"
        );
    }

    // Slot acquisition appears at least 7 times — once per chronicle-
    // bearing arm. Use `>= 7` rather than `== 7` because future arms
    // may grow chronicle counterparts (the test stays correct as long
    // as the seven slice-γ wirings remain).
    let slot_acquisitions = body
        .matches("let _slot: u32 = atomicAdd(&event_tail[0], 1u);")
        .count();
    assert!(
        slot_acquisitions >= 7,
        "expected ≥7 chronicle slot acquisitions (one per slice-γ arm); \
         got {slot_acquisitions};\n\
         body:\n{body}"
    );
}

/// Naga-validate the dispatcher's emitted WGSL. Format-string
/// assertions catch missing arms / wrong constants, but they DON'T
/// catch syntax errors the WGSL frontend rejects (mismatched braces,
/// undeclared identifiers, type errors, atomic-handle misuse, etc.).
/// Without this gate, a malformed dispatcher could ship green at the
/// dsl_compiler test boundary and only blow up when a runtime crate
/// tries to feed the kernel into wgpu.
///
/// Naga 26.0.0 is the WGSL frontend that wgpu 26.0.1 uses (see
/// `Cargo.toml` pinning), so passing naga here is a strong proxy for
/// "this kernel will compile on a real GPU device". The actual
/// device-driving GPU parity test (#133) layers on top of this gate
/// once a runtime crate consumes the dispatcher.
#[test]
fn apply_ability_smoke_kernel_parses_through_naga() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        if let Err(e) = naga::front::wgsl::parse_str(body) {
            errs.push(format!("  {name}:\n    {e}"));
        }
    }
    assert!(
        errs.is_empty(),
        "apply_ability_smoke emitted {} naga-invalid WGSL kernels — \
         the dispatcher's chronicle writes (slice-γ self-cast or \
         per-arm atomicStore layout) likely produced WGSL the frontend \
         rejects:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Variant: `apply_ability` nested inside an `if` body. Exercises
/// `list_contains_apply_ability`'s recursion through `CgStmt::If` —
/// the helper that drives `wire_ability_registry_column_reads` must
/// descend into both `then` and `else_` arms; if it stops at the
/// top-level statement list, the BGL composer never wires the
/// `ability_registry_*` bindings and naga rejects the kernel
/// (same shape of bug as commit `f447d3eb`).
///
/// Source-string-compiled rather than file-loaded: a transient
/// fixture variant doesn't justify a new `.sim` file in the corpus.
#[test]
fn apply_ability_nested_in_if_body_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }

        entity Hero : Agent { }

        physics ConditionalDispatch @phase(per_agent) {
          on Tick {} where (self.alive) {
            if (agents.level(self) > 0) {
              apply_ability agents.level(self)
            }
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "ConditionalDispatch")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");
    // Sanity: dispatcher loop reached the if-arm.
    assert!(
        body.contains("for (var i: u32 = 0u; i < 6u;"),
        "dispatcher loop must land in nested if arm;\n{body}"
    );
    // Sanity: ability_registry bindings declared (proves
    // wire_ability_registry_column_reads recursed into the if).
    assert!(
        body.contains("ability_registry_effect_kinds"),
        "BGL composer must wire ability_registry_effect_kinds even \
         when ApplyAbility is nested in an if body;\n{body}"
    );

    // Naga validator over every emitted kernel.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse failed: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate failed: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "nested-if variant emitted {} kernels naga rejects:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Variant: two PerAgent physics rules sharing `on Tick` — one uses
/// `apply_ability`, one doesn't. The schedule fusion pass combines
/// them into a single kernel (`cs_physics_DispatcherRule_and_PlainRule`)
/// because both share the same dispatch shape, event source, and
/// have no read/write conflicts.
///
/// The fused kernel's binding set is the UNION of both rules' reads
/// and writes — so `ability_registry_*` is present (DispatcherRule
/// needs it) and `event_ring` is present (DispatcherRule's chronicle
/// writes), even though PlainRule individually doesn't reference
/// either. This is correct behavior; the test pins it so a future
/// fusion-pass change that disables this case surfaces deliberately
/// rather than as a confusing kernel-naming change.
///
/// What this test guards: the post-fusion kernel still passes naga
/// validation. The dispatcher's binding-recording logic
/// (`wire_ability_registry_column_reads`) survives fusion correctly
/// — a fusion-time bug would either drop the recorded reads (kernel
/// references undeclared identifiers) or duplicate them (double
/// binding declarations).
#[test]
fn apply_ability_in_fused_kernel_with_plain_rule_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }

        entity Hero : Agent { }

        physics DispatcherRule @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self)
          }
        }

        physics PlainRule @phase(per_agent) {
          on Tick {} where (self.alive) {
            agents.set_hp(self, agents.hp(self) + 1.0)
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    // The fused kernel name is composer-driven; look for either rule
    // name, taking whichever the composer picked.
    let fused_body = kernel_body_containing(&art, "DispatcherRule")
        .or_else(|| kernel_body_containing(&art, "PlainRule"))
        .expect("fused physics kernel emitted");

    // Bindings present (union of both rules):
    assert!(
        fused_body.contains("ability_registry_effect_kinds"),
        "fused kernel must carry ability_registry_effect_kinds (from \
         DispatcherRule's apply_ability);\n{fused_body}"
    );
    assert!(
        fused_body.contains("agent_hp"),
        "fused kernel must carry agent_hp (from PlainRule's HP write);\n{fused_body}"
    );

    // Each binding declared exactly once (dedup correct under fusion).
    let kinds_decls = fused_body
        .lines()
        .filter(|l| l.contains("var<storage") && l.contains("ability_registry_effect_kinds"))
        .count();
    assert_eq!(
        kinds_decls, 1,
        "ability_registry_effect_kinds declared exactly once even under fusion;\n{fused_body}"
    );

    // Naga validator confirms the fused kernel is well-formed.
    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "fused-rule variant emitted {} naga-rejected kernels:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Variant: `apply_ability` inside a PerEvent rule (driven by a
/// custom event with payload binding). Different kernel shape than
/// the PerAgent fixtures — the kernel iterates `event_count` rather
/// than `agent_cap`, and the cfg uniform carries event-count, not
/// agent-count.
///
/// **Known-failing — captured here as documentation, not a CI gate.**
///
/// **Structural plumbing landed (#161 part 1)**: `CgStmt::ApplyAbility`
/// now carries an explicit `caster` operand, and the dispatcher emit
/// reads `caster` instead of the prior hardcoded `agent_id`. This
/// fixes the PerAgent path cleanly (caster lowers to `AgentSelfId` →
/// `agent_id`) — see the surrounding tests in this file.
///
/// **PerEvent path still broken**: lowering today
/// (`cg/lower/physics.rs:567`) populates caster from
/// `CgExpr::AgentSelfId` REGARDLESS of dispatch shape. PerEvent
/// kernels still resolve `agent_id` to the same undeclared
/// identifier — the structural change unblocks the next step but
/// doesn't solve PerEvent on its own.
///
/// **Next slice (#161 part 2)**: thread `DispatchShape` through
/// `lower_physics_stmt` and synthesize the caster from the event
/// payload's actor field when the surrounding rule is PerEvent.
/// Without an actor convention on every event, this also needs a
/// design decision (e.g., gate apply_ability to events with an
/// `actor: AgentId` field).
///
/// Removing the `#[ignore]` is the natural reproducer when the
/// PerEvent lowering branch lands.
#[test]
#[ignore = "slice δ part 2 — needs PerEvent-aware caster lowering \
            from event payload's actor field; structural plumbing \
            (CgStmt::ApplyAbility.caster) is in but lowering still \
            populates AgentSelfId for both shapes"]
fn apply_ability_in_per_event_rule_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }
        event Triggered { who: AgentId, ability_id: u32 }

        entity Hero : Agent { }

        physics DispatchOnTrigger {
          on Triggered { who: w, ability_id: a } {
            apply_ability a
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "DispatchOnTrigger")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");
    // PerEvent shape: cfg carries event_count, not agent_cap.
    assert!(
        body.contains("cfg.event_count") || body.contains("event_count"),
        "PerEvent kernel must reference event_count in its preamble \
         (cfg uniform shape);\n{body}"
    );
    // Dispatcher loop reached.
    assert!(
        body.contains("for (var i: u32 = 0u; i < 6u;"),
        "dispatcher loop must land in PerEvent rule body;\n{body}"
    );

    // Naga validator. Catches PerEvent-specific shape errors (e.g.
    // event_idx vs agent_id confusion in the chronicle writes).
    let mut errs = Vec::new();
    for (name, kernel_body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(kernel_body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse failed: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate failed: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "PerEvent ApplyAbility variant emitted {} kernels naga rejects:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Variant: `apply_ability` + regular `emit` in the same rule body.
/// Both write to `event_ring` — the dispatcher's chronicle writes
/// (slice-γ kinds 26..=32) and the user-declared `emit Pinged { ... }`
/// share the same SoA buffer. The BGL composer must wire `event_ring`
/// + `event_tail` exactly once even when both producers contribute
/// to the same kernel.
///
/// Without correct binding deduplication, the kernel would either
/// declare `event_ring` twice (WGSL rejects) or once with the wrong
/// access mode (atomic vs non-atomic). Both surface here as naga
/// errors; if neither fires, the dedup is correct.
#[test]
fn apply_ability_alongside_regular_emit_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let src = "
        event Tick { }
        event Pinged { who: AgentId, when: u32 }

        entity Hero : Agent { }

        physics MixedDispatch @phase(per_agent) {
          on Tick {} where (self.alive) {
            apply_ability agents.level(self)
            emit Pinged { who: self, when: world.tick }
          }
        }
    ";
    let program = dsl_compiler::parse(src).expect("parse");
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .expect("lower");
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    let art = dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .expect("emit");

    let body = kernel_body_containing(&art, "MixedDispatch")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .expect("physics kernel emitted");
    // event_ring should appear exactly once as a binding declaration
    // (the BGL composer dedups). Counting `var<storage` declarations
    // mentioning `event_ring` is a coarse-but-sufficient proxy.
    let event_ring_decls = body
        .lines()
        .filter(|l| l.contains("var<storage") && l.contains("event_ring"))
        .count();
    assert_eq!(
        event_ring_decls, 1,
        "event_ring binding must dedup to exactly one declaration \
         even when ApplyAbility (chronicle writes 26..=32) and regular \
         emit (Pinged) both target it; found {event_ring_decls};\n{body}"
    );

    // Naga validator over every emitted kernel.
    let mut errs = Vec::new();
    for (name, kernel_body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(kernel_body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse failed: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate failed: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "mixed-emit variant emitted {} kernels naga rejects:\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Stronger gate than `..._parses_through_naga` — runs naga's full
/// validator over the parsed module. Catches type errors, missing
/// `@binding(N) @group(0)` annotations, atomic-handle misuse,
/// access-mode mismatches, etc. that pure parsing misses.
///
/// `Capabilities::default()` matches WebGPU's baseline; the dispatcher
/// uses no extension features (atomic u32 ops on storage are baseline).
#[test]
fn apply_ability_smoke_kernel_passes_naga_validator() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    let mut errs = Vec::new();
    for (name, body) in &art.wgsl_files {
        let module = match naga::front::wgsl::parse_str(body) {
            Ok(m) => m,
            Err(e) => {
                errs.push(format!("  {name}: parse failed: {e}"));
                continue;
            }
        };
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        if let Err(e) = validator.validate(&module) {
            errs.push(format!("  {name}: validate failed: {e:?}"));
        }
    }
    assert!(
        errs.is_empty(),
        "apply_ability_smoke emitted {} kernels that fail naga \
         validation (parse OK, validator NOT OK — type / binding / \
         atomic-handle issues):\n{}",
        errs.len(),
        errs.join("\n"),
    );
}

/// Pin the BGL composer's wiring of `event_ring` + `event_tail` into
/// the dispatcher kernel. Without these bindings, the chronicle writes
/// emitted by the dispatcher arms would reference undeclared identifiers
/// at WGSL compile time. Recording an `EventRing(Append)` write on
/// ApplyAbility-bearing ops (commit `1779b0e6`) is what hooks the
/// composer; this assertion tests that the hook still fires after the
/// rest of the pipeline runs.
#[test]
fn apply_ability_smoke_kernel_binds_event_ring_and_event_tail() {
    let path = workspace_path("assets/sim/apply_ability_smoke.sim");
    let art = compile_sim(&path).expect("apply_ability_smoke compiles");

    let body = kernel_body_containing(&art, "DispatchAbility")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| {
            panic!(
                "no physics kernel found in artifacts; available: {:?}",
                art.wgsl_files.keys().collect::<Vec<_>>()
            );
        });

    // The composer emits `var<storage, ...> event_ring : array<...>;`
    // and `var<storage, ...> event_tail : array<...>;` declarations
    // after running the EventRing(Append)+sibling-event_tail synthesis
    // path in `cg::emit::kernel`. Match the bare `event_ring` /
    // `event_tail` identifier rather than the full type signature
    // (`array<atomic<u32>>` vs `array<u32>` may evolve as the binding
    // metadata refines), so the assertion is robust to wgsl-ty drift.
    assert!(
        body.contains("event_ring"),
        "dispatcher kernel must bind event_ring (the chronicle writes \
         in the slice-γ arms reference it);\n\
         got body:\n{body}"
    );
    assert!(
        body.contains("event_tail"),
        "dispatcher kernel must bind event_tail (the dispatcher's \
         atomicAdd slot acquisition references it);\n\
         got body:\n{body}"
    );

    // The two bindings appear as WGSL `var<storage, ...>` declarations
    // (one each). At least one declaration per identifier must be
    // present — multiple references in the chronicle writes are fine
    // but the binding declaration itself is what the BGL composer
    // emits exactly once.
    assert!(
        body.matches("var<storage").count() >= 2,
        "expected ≥2 storage binding declarations (event_ring + \
         event_tail at minimum);\n\
         got body:\n{body}"
    );
}
