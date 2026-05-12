//! Pin: `agents.set_alive(self, false)` inside a conditional branch
//! survives PerAgent kernel fusion.
//!
//! Closes the regression-test half of **Gap P-D** in
//! `docs/architecture/gaps_plague_city.md` ("`agents.set_alive(self,
//! false)` inside a fused PerAgent rule doesn't flip the alive bit").
//!
//! Background
//! ----------
//!
//! When the gap was filed, plague_city's `SicknessProgresses` rule had
//! the body:
//!
//! ```text
//! let old_hp = agents.hp(self);
//! let new_hp = old_hp - config.plague.sickness_rate;
//! agents.set_hp(self, new_hp);
//! if (old_hp > 0.0 && new_hp <= 0.0) {
//!   agents.set_alive(self, false);
//!   emit Died { victim: self }
//! }
//! ```
//!
//! …and the schedule fused it with a sibling `ContagionScan` rule into
//! a single PerAgent kernel. The pin reported `D=0` final dead even
//! though every Citizen had accumulated hundreds of negative hp. The
//! gap doc filed three hypotheses; the load-bearing one (1) was that
//! "the conditional branch's `set_alive` Assign gets dropped at fusion
//! time".
//!
//! Investigation (post-T5 schedule fix, commit `d1207fca`)
//! -------------------------------------------------------
//!
//! - `body_ops_have_set_alive_false` in `cg/emit/kernel.rs` walks
//!   every body op of a fused kernel and recursively scans nested
//!   `If` / `Match` / `ForEachNeighborBody` bodies via
//!   `stmt_list_contains_set_alive_false` (`cg/emit/wgsl_body.rs`).
//!   A `set_alive(self, false)` nested inside a conditional in any
//!   sub-body of any fused op is detected and triggers the
//!   `agent_alive` AtomicStorage upgrade.
//! - The per-stmt emit path (`is_alive_cas_site` branch in
//!   `cg/emit/wgsl_body.rs`) emits the
//!   `atomicCompareExchangeWeak(&agent_alive[idx], 1u, 0u)` CAS
//!   regardless of nesting depth — the conditional wrapper around it
//!   is preserved verbatim.
//! - Plague_city's actual symptom was the spatial-build-after-consumer
//!   schedule cycle (Gap T5) — the `physics_ContagionScan` kernel
//!   dispatched BEFORE the `spatial_build_hash_*` chain on tick 1, so
//!   the contagion's `for x in spatial.nearby(self)` walked an empty
//!   grid; no further infections beyond the initial 6 host-seeded
//!   citizens; the 6 originally-infected DID die from the alive flip
//!   but the pin's NOTE described it as "0 deaths".
//!
//! After T5 closed the schedule cycle, the pin reports `D=6` (the
//! initial 6 infections progress to death and flip alive correctly);
//! the fusion-side hypothesis in Gap P-D is structurally falsified
//! — the conditional alive write was never being dropped.
//!
//! Pin shape
//! ---------
//!
//! Two PerAgent rules with overlapping read/write footprints that the
//! fusion analyzer DOES fuse (different SoA columns, no write-after-
//! write hazard) — the second rule contains a conditional alive flip
//! identical in shape to plague_city's `SicknessProgresses` body.
//!
//! Asserts:
//!   1. The fused kernel's WGSL contains `physics_<A>_and_<B>` in
//!      the kernel name (fusion happened).
//!   2. The fused body upgrades `agent_alive` to
//!      `array<atomic<u32>>` (the conditional alive write was
//!      detected by `body_ops_have_set_alive_false`).
//!   3. The fused body contains the
//!      `atomicCompareExchangeWeak(&agent_alive[..], 1u, 0u)` kill
//!      CAS — the conditional wrapper preserves the write.
//!
//! A future regression that, e.g., short-circuits the recursive scan
//! on fused-kernel bodies, drops conditional-arm Assigns at fusion
//! time, or only honours top-level alive writes would trip exactly
//! these assertions.

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

fn find_kernel<'a>(
    art: &'a EmittedArtifacts,
    needle: &str,
) -> Option<(&'a str, &'a str)> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle) && name.ends_with(".wgsl"))
        .map(|(n, b)| (n.as_str(), b.as_str()))
}

/// The load-bearing Gap P-D regression: a PerAgent rule whose body
/// performs a conditional `agents.set_alive(self, false)` MUST keep
/// the alive write through fusion with a sibling PerAgent rule.
///
/// Verifies the fused kernel:
///   - contains both source rule names in its emitted file name
///     (i.e. fusion actually happened),
///   - upgrades `agent_alive` to `array<atomic<u32>>` (the conditional
///     alive write was detected by `body_ops_have_set_alive_false`),
///   - emits the `atomicCompareExchangeWeak(&agent_alive[..], 1u, 0u)`
///     kill CAS inside the fused body.
///
/// If a future fusion-side change strips conditional-arm Assigns at
/// fusion time, this assertion trips.
#[test]
fn fused_per_agent_preserves_conditional_set_alive_false() {
    // Two PerAgent rules sharing the same dispatch shape but writing
    // to disjoint SoA columns (no WAW hazard) — the fusion analyzer
    // joins them into one kernel. The second body contains the gap's
    // exact shape: an `if`-gated alive flip.
    //
    // Note on rule body content:
    //   - `Bleeder` writes agent_hunger only (reads hunger; no alive
    //     write). Picks up agent_hunger as a write to make the fusion
    //     candidate non-trivial.
    //   - `Sickness` mirrors plague_city's SicknessProgresses: reads
    //     hp, writes hp, and conditionally writes alive on the exact
    //     death-crossing tick.
    let src = r#"
event Tick { }

@replayable @gpu_amenable
event Died {
  victim: AgentId,
}

@phase(per_agent)
physics Bleeder {
  on Tick {} where (self.alive) {
    let h = agents.hunger(self);
    agents.set_hunger(self, h - 1.0);
  }
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

    // 1. Fusion happened — find the fused kernel by the `_and_`
    //    naming convention. The two rules share the same dispatch
    //    shape (both PerAgent) and write to disjoint columns
    //    (hunger vs hp/alive), so the fusion analyzer joins them.
    let fused = find_kernel(&art, "_and_").unwrap_or_else(|| {
        let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
        panic!(
            "expected a fused `physics_X_and_Y` kernel; got kernels: {names:?}.\n\
             If the fusion analyzer no longer fuses these rules, the test \
             premise is invalid — pick two siblings with overlapping shape \
             and disjoint writes that DO fuse.",
        );
    });
    let (fused_name, fused_body) = fused;
    assert!(
        fused_name.contains("Bleeder") && fused_name.contains("Sickness"),
        "fused kernel name must mention both source rules; got name: {fused_name}\n\
         body:\n{fused_body}",
    );

    // 2. The conditional alive write was detected — `agent_alive`
    //    binding upgraded to atomic storage. This is the load-bearing
    //    half of the gap: `body_ops_have_set_alive_false` must
    //    recurse into the conditional `then`-arm of the Sickness
    //    body to detect the nested alive write.
    assert!(
        fused_body.contains("var<storage, read_write> agent_alive: array<atomic<u32>>;"),
        "fused kernel must upgrade agent_alive to array<atomic<u32>>; the conditional \
         alive write was missed by the recursive scan. Kernel: {fused_name}\n\
         body:\n{fused_body}",
    );

    // 3. The CAS kill itself survives — the conditional wrapper in
    //    the fused body still hosts the
    //    `atomicCompareExchangeWeak(&agent_alive[..], 1u, 0u)` write.
    //    A regression that drops conditional-arm Assigns at fusion
    //    time would leave the binding upgraded (step 2) but the body
    //    empty (step 3 fails).
    assert!(
        fused_body.contains("atomicCompareExchangeWeak(&agent_alive["),
        "fused kernel must emit the kill CAS inside the conditional; the \
         fusion stripped the conditional-arm alive write. Kernel: {fused_name}\n\
         body:\n{fused_body}",
    );
    assert!(
        fused_body.contains(", 1u, 0u)"),
        "fused kernel CAS must be in the kill direction (1u → 0u); got body:\n{fused_body}",
    );
}
