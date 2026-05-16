//! Plan I — slice I.5 smoke pin (compile-pipeline variant).
//!
//! Drives `assets/sim/belief_smoke_probe.sim` through the full
//! parse → resolve → lower → emit pipeline and pins that the
//! `belief seen(observer: Agent, subject: Agent) -> u32` declaration:
//!
//!   1. parses cleanly (Decl::Belief),
//!   2. resolves into a ViewIR slot with `kind: ViewKind::Belief`,
//!   3. lowers via the slice I.3 PairMap inference path,
//!   4. emits at least one `fold_seen` WGSL kernel with the bit-OR
//!      `atomicOr` body the propagation handler maps to.
//!
//! NOTE: this runs at the compiler-test level (no GPU, no runtime)
//! because the sims-crate build is currently blocked by an unrelated
//! pre-existing panic in `threats_struct_probe` (see
//! `crates/dsl_compiler/src/cg/emit/program.rs:2021`). When that
//! lands a fix the smoke probe should also be opted into
//! `sims/build.rs` for a full runtime smoke pin.

use dsl_ast::ir::ViewKind;
use dsl_compiler::cg::emit::emit_cg_program;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/sim/belief_smoke_probe.sim"
);

#[test]
fn belief_smoke_probe_parse_resolve_lower_emit_round_trip() {
    let src = std::fs::read_to_string(FIXTURE_PATH)
        .unwrap_or_else(|e| panic!("read {FIXTURE_PATH}: {e}"));

    // ---- parse ----
    let prog = dsl_compiler::parse(&src).expect("parse belief_smoke_probe.sim");
    let belief_decl_count = prog
        .decls
        .iter()
        .filter(|d| matches!(d, dsl_compiler::ast::Decl::Belief(_)))
        .count();
    assert_eq!(
        belief_decl_count, 1,
        "expected exactly one `belief` decl in the smoke probe",
    );

    // ---- resolve ----
    let comp = dsl_ast::resolve::resolve(prog).expect("resolve belief_smoke_probe.sim");
    let seen_view = comp
        .views
        .iter()
        .find(|v| v.name == "seen")
        .expect("`seen` belief should appear in comp.views");
    assert_eq!(
        seen_view.kind,
        ViewKind::Belief,
        "the smoke probe's `seen` decl should resolve with kind = Belief",
    );

    // ---- lower ----
    let cg = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(outcome) => {
            let diags: Vec<String> =
                outcome.diagnostics.iter().map(|d| format!("{d}")).collect();
            panic!("lower failed with diagnostics: {diags:?}");
        }
    };

    // ---- emit ----
    let sched = synthesize_schedule(&cg, ScheduleStrategy::Default);
    let art = emit_cg_program(&sched.schedule, &cg).expect("emit succeeds");

    // The propagation handler must produce a `fold_seen` kernel
    // whose body atomically OR-merges the `mark` bit into the
    // pair-keyed cell. Match loosely on filename + body content so a
    // future kernel-fusion pass renaming the kernel still passes as
    // long as the `seen` fold is represented somewhere.
    let names: Vec<&str> = art.wgsl_files.iter().map(|(n, _)| n.as_str()).collect();
    let has_seen_kernel = art
        .wgsl_files
        .iter()
        .any(|(name, _)| name.contains("seen"));
    assert!(
        has_seen_kernel,
        "expected a WGSL kernel referencing `seen`; emitted: {names:?}",
    );
    let fold_body_with_or = art.wgsl_files.iter().find(|(name, body)| {
        name.contains("seen") && (body.contains("atomicOr") || body.contains("|="))
    });
    assert!(
        fold_body_with_or.is_some(),
        "expected the `seen` kernel body to use `atomicOr` (or `|=`); \
         emit may have refactored — files: {names:?}",
    );

    // Plan I.4a — social-merge clause produces a separate kernel
    // (named `merge_<view>_<event>_<op>`). Today the body is a
    // documented stub (`// TODO(plan-I.4b)`); the kernel still emits
    // and the runtime can dispatch it as a no-op until I.4b lands the
    // per-cell merge logic.
    let merge_kernel = art.wgsl_files.iter().find(|(name, _)| {
        name.contains("merge") && name.contains("seen") && name.contains("bit_or")
    });
    assert!(
        merge_kernel.is_some(),
        "expected a `merge_seen_*_bit_or` kernel for the social-merge clause; \
         emitted: {names:?}"
    );
    let (merge_name, merge_body) = merge_kernel.unwrap();
    // Plan I.4b kernel preamble — confirms the event-driven structure
    // is in place: bounds-checked event_idx, kind tag filter, source-
    // agent read from the event payload.
    assert!(
        merge_body.contains("source_agent") && merge_body.contains("event_ring"),
        "expected merge kernel body to read source_agent from event_ring; \
         body excerpt: {}",
        &merge_body[..merge_body.len().min(800)]
    );
    // Plan I.4b — the per-receiver × per-cell merge loop is now in
    // place for bit_or. Spot-check for the atomic primitives the
    // merge uses (atomicLoad from source_agent's row, atomicOr into
    // receiver's row).
    assert!(
        merge_body.contains("atomicLoad") && merge_body.contains("atomicOr"),
        "expected merge kernel body to use atomicLoad + atomicOr for the \
         per-cell bit_or merge; body excerpt: {}",
        &merge_body[..merge_body.len().min(800)]
    );

    // Naga validation — confirms the emitted WGSL is at least
    // syntactically + structurally valid, even if the per-cell merge
    // logic is still a TODO. Catches drift in the hand-emitted
    // skeleton (e.g. a typo in the cfg-field name, or a missing
    // binding declaration that would otherwise only surface at
    // runtime dispatch).
    let module = naga::front::wgsl::parse_str(merge_body).unwrap_or_else(|e| {
        panic!(
            "naga parse failed for merge kernel `{merge_name}`:\n{merge_body}\n\
             error: {e:?}"
        );
    });
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::default(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap_or_else(|e| {
        panic!(
            "naga validate failed for merge kernel `{merge_name}`:\n{merge_body}\n\
             error: {e:?}"
        );
    });
}
