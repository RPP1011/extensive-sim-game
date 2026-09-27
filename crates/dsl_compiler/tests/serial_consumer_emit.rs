//! `@serial(...)` deterministic-contention chronicle consumers — compile
//! side (lowering → schedule → WGSL). The GPU behaviour (bit-exact
//! agreement with a host serial reference, run-to-run and cross-process
//! determinism under ~1000-way contention) is pinned in
//! `crates/sims/tests/contention_probe_pin.rs`; this file pins the
//! compiler contract:
//!
//! * the annotation resolves to a `SerialConsumerSpec` (key word, ordered
//!   columns with direction, payload tiebreak words);
//! * the consumer kernel carries the election + ordered-drain driver and
//!   runs the ORIGINAL body with `event_idx` shadowed per group member;
//! * serial consumers are never fused with a sibling op;
//! * the claim-CAS promotion (a scheduling-dependent winner) is skipped
//!   for serial consumers, while the un-annotated control keeps it;
//! * every emitted kernel passes naga parse + validation;
//! * malformed annotations are typed lowering errors;
//! * programs without the annotation have an empty side table.

use dsl_compiler::cg::emit::EmittedArtifacts;
use dsl_compiler::cg::program::{CgProgram, SerialWordMap};

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn lower_src(src: &str) -> Result<CgProgram, Vec<String>> {
    let program = dsl_compiler::parse(src).map_err(|e| vec![format!("parse: {e:?}")])?;
    let _ = dsl_compiler::custom_agent_fields::populate(&program);
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| vec![format!("resolve: {e:?}")])?;
    dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|o| o.diagnostics.iter().map(|d| d.to_string()).collect())
}

fn emit(cg: &CgProgram) -> EmittedArtifacts {
    let sched = dsl_compiler::cg::schedule::synthesize_schedule(
        cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&sched.schedule, cg).expect("emit")
}

fn fixture(stem: &str) -> (CgProgram, EmittedArtifacts) {
    let src = std::fs::read_to_string(workspace_path(&format!("assets/sim/{stem}.sim"))).unwrap();
    let cg = lower_src(&src).unwrap_or_else(|d| panic!("{stem} failed to lower: {d:#?}"));
    let art = emit(&cg);
    (cg, art)
}

fn kernel<'a>(art: &'a EmittedArtifacts, file: &str) -> &'a str {
    art.wgsl_files.get(file).map(|s| s.as_str()).unwrap_or_else(|| {
        panic!("no kernel {file}; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>())
    })
}

fn naga_validate_all(art: &EmittedArtifacts) {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
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
    assert!(errs.is_empty(), "naga rejected {} kernels:\n{}", errs.len(), errs.join("\n"));
}

#[test]
fn claim_probe_resolves_serial_spec() {
    let (cg, _) = fixture("contention_claim_probe");
    assert_eq!(cg.serial_consumers.len(), 1, "only ApplyClaim is @serial");
    let spec = cg.serial_consumers.values().next().unwrap();
    // ClaimReq { who, what, prio }: header is 2 words (kind, tick).
    assert_eq!(spec.key_name, "t");
    assert_eq!(spec.key_word, 3, "key = t = `what`, payload word 1");
    let cols: Vec<(u32, bool, SerialWordMap)> =
        spec.order.iter().map(|o| (o.word, o.desc, o.map)).collect();
    assert_eq!(
        cols,
        vec![(4, true, SerialWordMap::Unsigned), (2, false, SerialWordMap::Unsigned)],
        "order = [p desc, w asc]"
    );
    assert_eq!(spec.payload_words, vec![2, 3, 4]);
}

#[test]
fn serial_kernel_carries_driver_and_skips_claim_cas() {
    let (_, art) = fixture("contention_claim_probe");
    let serial = kernel(&art, "physics_ApplyClaim.wgsl");
    for needle in [
        "@serial(key = t, order = [p desc, w])",
        "var _sg_lead: bool = true;",
        "var _sg_buf: array<u32, ",
        "let event_idx: u32 = _sg_buf[_sg_q];",
        // Grid-stride over the whole live ring, not one slot per thread
        // (PerEvent kernels are dispatched only agent_cap threads wide).
        "@builtin(num_workgroups) _sg_nwg: vec3<u32>",
        "_sg_ev = _sg_ev + _sg_nwg.x * 64u",
        // p desc is bit-inverted into an ascending key; w stays as-is.
        "let _sg_me0: u32 = (~event_ring[event_idx * 11u + 4u]);",
        "let _sg_me1: u32 = event_ring[event_idx * 11u + 2u];",
        // Emit-free serial kernels read the ring with plain loads.
        "var<storage, read> event_ring: array<u32>;",
    ] {
        assert!(serial.contains(needle), "serial kernel missing `{needle}`:\n{serial}");
    }
    assert!(
        !serial.contains("atomicCompareExchangeWeak"),
        "serial consumer must not be CAS-promoted:\n{serial}"
    );
    let control = kernel(&art, "physics_ApplyClaimUnordered.wgsl");
    assert!(!control.contains("_sg_"), "the un-annotated control must keep the default parallel apply");
}

#[test]
fn serial_consumer_is_never_fused() {
    for stem in ["contention_claim_probe", "contention_stock_probe"] {
        let (cg, art) = fixture(stem);
        assert_eq!(cg.serial_consumers.len(), 1);
        // Each consumer keeps its own singleton kernel file — a fused
        // kernel would be named after both rules.
        let names: Vec<&String> = art.wgsl_files.keys().collect();
        assert!(
            names.iter().all(|n| !(n.contains("ApplyClaim_and") || n.contains("ApplyEat_and"))),
            "{stem}: serial consumer fused with a sibling: {names:?}"
        );
        let serial_file = if stem.contains("claim") { "physics_ApplyClaim.wgsl" } else { "physics_ApplyEat.wgsl" };
        assert_eq!(kernel(&art, serial_file).matches("// op#").count(), 1);
    }
}

#[test]
fn stock_probe_orders_by_hunger_desc_and_keeps_f32_plain() {
    let (cg, art) = fixture("contention_stock_probe");
    let spec = cg.serial_consumers.values().next().unwrap();
    assert_eq!(spec.key_word, 3);
    assert_eq!(spec.order.len(), 2);
    assert!(spec.order[0].desc && !spec.order[1].desc);
    // Eat { who, what, hunger, share }: `share` (f32) is a tiebreak word.
    assert_eq!(spec.payload_words, vec![2, 3, 4, 5]);
    let serial = kernel(&art, "physics_ApplyEat.wgsl");
    assert!(serial.contains("agent_ledger[target_expr_"), "{serial}");
    assert!(serial.contains("} else {"), "the guarded-decrement else branch must survive:\n{serial}");
}

#[test]
fn contention_fixtures_pass_naga_validation() {
    for stem in ["contention_claim_probe", "contention_stock_probe"] {
        let (_, art) = fixture(stem);
        naga_validate_all(&art);
    }
}

const BASE: &str = "
event Tick { }
@replayable @gpu_amenable
event Req { who: AgentId, what: AgentId, prio: u32, spot: vec3, score: f32, rank: i32 }
entity W : Agent { }
field taken: u32
";

fn with_rule(rule: &str) -> String {
    format!("{BASE}\n{rule}\n")
}

#[test]
fn f32_and_i32_order_columns_get_total_order_maps() {
    let cg = lower_src(&with_rule(
        "@phase(post) @serial(key = what, order = [score, rank], desc = [score])
         physics R { on Req { who: w, what: t, score: s, rank: r } { agents.set_taken(t, 1u); } }",
    ))
    .expect("lowers");
    let spec = cg.serial_consumers.values().next().unwrap();
    assert_eq!(spec.key_name, "what", "event field names work as well as binders");
    let maps: Vec<SerialWordMap> = spec.order.iter().map(|o| o.map).collect();
    assert_eq!(maps, vec![SerialWordMap::Float, SerialWordMap::Signed]);
    let art = emit(&cg);
    let body = kernel(&art, "physics_R.wgsl");
    assert!(body.contains("^ 0x80000000u"), "{body}");
    naga_validate_all(&art);
}

fn expect_serial_error(rule: &str, needle: &str) {
    match lower_src(&with_rule(rule)) {
        Ok(_) => panic!("expected an @serial lowering error containing `{needle}`"),
        Err(diags) => assert!(
            diags.iter().any(|d| d.contains("@serial") && d.contains(needle)),
            "expected `{needle}` in diagnostics: {diags:#?}"
        ),
    }
}

#[test]
fn malformed_serial_annotations_are_typed_errors() {
    expect_serial_error(
        "@phase(post) @serial(order = [p])
         physics R { on Req { who: w, what: t, prio: p } { agents.set_taken(t, 1u); } }",
        "missing group key",
    );
    expect_serial_error(
        "@phase(post) @serial(key = nope)
         physics R { on Req { who: w, what: t } { agents.set_taken(t, 1u); } }",
        "`nope` is neither a binder",
    );
    expect_serial_error(
        "@phase(post) @serial(key = t, order = [p], desc = [w])
         physics R { on Req { who: w, what: t, prio: p } { agents.set_taken(t, 1u); } }",
        "listed in `desc` but not in `order`",
    );
    expect_serial_error(
        "@phase(post) @serial(key = spot)
         physics R { on Req { who: w, what: t } { agents.set_taken(t, 1u); } }",
        "one-word scalars",
    );
    expect_serial_error(
        "@phase(post) @serial(key = t, bogus = p)
         physics R { on Req { who: w, what: t, prio: p } { agents.set_taken(t, 1u); } }",
        "unknown argument `bogus`",
    );
    expect_serial_error(
        "@phase(per_agent) @serial(key = self)
         physics R { on Tick {} { agents.set_taken(self, 1u); } }",
        "cannot be combined with `@phase(per_agent)`",
    );
}

#[test]
fn unannotated_programs_have_no_serial_consumers() {
    let cg = lower_src(&with_rule(
        "@phase(post)
         physics R { on Req { who: w, what: t } { if (agents.taken(t) == 0u) { agents.set_taken(t, 1u); } } }",
    ))
    .expect("lowers");
    assert!(cg.serial_consumers.is_empty());
    let art = emit(&cg);
    let body = kernel(&art, "physics_R.wgsl");
    assert!(!body.contains("_sg_"));
    assert!(body.contains("atomicCompareExchangeWeak"), "default claim-CAS promotion unchanged:\n{body}");
}
