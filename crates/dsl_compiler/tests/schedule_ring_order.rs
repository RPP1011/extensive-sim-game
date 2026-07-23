//! S10 — regression pins for the "chronicle consumer scheduled before
//! its producer" defect class.
//!
//! # The class
//!
//! An op that READS this tick's event ring must be dispatched after every
//! op that APPENDS the kind it reads. Break the ordering and NOTHING
//! fails: the consumer scans a segment its producer has not written,
//! matches zero rows, and the feature silently stops existing. Three
//! shipped instances (Gap dungeon_stealth#5; webband_colony S5 and S5b —
//! see `docs/superpowers/plans/webband-port.md`) were each found only
//! because someone happened to keep a numeric pin on the dead feature.
//!
//! Two distinct doors lead into it:
//!
//! * **Ordering policy** — `topological_sort_best_effort`'s cycle-stall
//!   force-pick used to emit the globally smallest un-emitted op, which
//!   can be an innocent ring CONSUMER whose producer is merely stuck
//!   behind an unrelated SCC.
//! * **Missing edges** — a producer→consumer ring edge that never forms
//!   (a wrong event-kind id, an incomplete emitted-kind table). No guard
//!   inside the sort can see this: the constraint it would enforce does
//!   not exist.
//!
//! These tests pin BOTH doors shut, and they pin the loud check that
//! makes either one impossible to ship silently again.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use dsl_compiler::cg::schedule::{
    dependency_graph, ring_order::validate_ring_order, ring_order::RingOrderIssueKind,
    ring_order::RingOrderSeverity, synthesize_schedule, ScheduleStrategy,
};
use dsl_compiler::cg::{CgProgram, CycleEdgeKey, DepGraph, OpId};

// ---------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------

fn lower_src(label: &str, src: &str) -> CgProgram {
    let program = dsl_compiler::parse(src).unwrap_or_else(|e| panic!("parse {label}: {e:?}"));
    // Custom `field` decls must be interned BEFORE resolve or every rule
    // touching one is silently dropped (Gap plague_city#P-A).
    let _ = dsl_compiler::custom_agent_fields::populate(&program);
    let comp =
        dsl_ast::resolve::resolve(program).unwrap_or_else(|e| panic!("resolve {label}: {e:?}"));
    match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => panic!("lower {label} failed: {:?}", o.diagnostics),
    }
}

fn stage_order(prog: &CgProgram) -> Vec<OpId> {
    synthesize_schedule(prog, ScheduleStrategy::Default)
        .schedule
        .stages
        .iter()
        .flat_map(|s| s.kernels.iter().flat_map(|k| k.ops()))
        .collect()
}

fn ring_edges(graph: &DepGraph) -> Vec<(OpId, OpId)> {
    graph
        .edge_reasons
        .iter()
        .filter(|(_, r)| r.iter().any(|k| matches!(k, CycleEdgeKey::Ring(_))))
        .map(|((p, c), _)| (*p, *c))
        .collect()
}

fn order_violations(graph: &DepGraph, order: &[OpId]) -> Vec<(OpId, OpId)> {
    let n = graph.op_count;
    let mut pos = vec![usize::MAX; n];
    for (i, op) in order.iter().enumerate() {
        if (op.0 as usize) < n {
            pos[op.0 as usize] = i;
        }
    }
    ring_edges(graph)
        .into_iter()
        .filter(|(p, c)| pos[c.0 as usize] < pos[p.0 as usize])
        .collect()
}

/// The PRE-FIX force-pick, reproduced verbatim: on a cycle stall, emit
/// the globally smallest un-emitted op with no regard for pending
/// event-ring producers. Kept here (and only here) so "this test would
/// fail under the pre-fix scheduler" is an executable claim rather than
/// an assertion in a report.
fn legacy_best_effort(graph: &DepGraph) -> Vec<OpId> {
    let n = graph.op_count;
    let mut in_degree = vec![0u32; n];
    for succs in graph.edges.values() {
        for s in succs {
            if (s.0 as usize) < n {
                in_degree[s.0 as usize] += 1;
            }
        }
    }
    let mut queue: BinaryHeap<Reverse<OpId>> = BinaryHeap::new();
    let mut emitted = vec![false; n];
    for (i, d) in in_degree.iter().enumerate() {
        if *d == 0 {
            queue.push(Reverse(OpId(i as u32)));
        }
    }
    let mut order = Vec::with_capacity(n);
    let drain = |queue: &mut BinaryHeap<Reverse<OpId>>,
                 emitted: &mut Vec<bool>,
                 in_degree: &mut Vec<u32>,
                 order: &mut Vec<OpId>| {
        while let Some(Reverse(op)) = queue.pop() {
            if emitted[op.0 as usize] {
                continue;
            }
            emitted[op.0 as usize] = true;
            order.push(op);
            if let Some(succs) = graph.edges.get(&op) {
                for &s in succs {
                    let i = s.0 as usize;
                    if i < n && !emitted[i] {
                        in_degree[i] -= 1;
                        if in_degree[i] == 0 {
                            queue.push(Reverse(s));
                        }
                    }
                }
            }
        }
    };
    while order.len() < n {
        drain(&mut queue, &mut emitted, &mut in_degree, &mut order);
        if order.len() == n {
            break;
        }
        let forced = (0..n).find(|&i| !emitted[i]).expect("residual op");
        emitted[forced] = true;
        order.push(OpId(forced as u32));
        if let Some(succs) = graph.edges.get(&OpId(forced as u32)) {
            for &s in succs {
                let i = s.0 as usize;
                if i < n && !emitted[i] {
                    in_degree[i] = in_degree[i].saturating_sub(1);
                    if in_degree[i] == 0 {
                        queue.push(Reverse(s));
                    }
                }
            }
        }
    }
    order
}

// ---------------------------------------------------------------------
// The synthetic fixture
// ---------------------------------------------------------------------

/// A deliberately hostile miniature of webband_colony's supper-gossip
/// shape, small enough to reason about op by op:
///
/// * `rumour` is a pair belief with a `merge from` carrier — a
///   `BeliefSocialMerge` op, and views lower BEFORE physics rules, so it
///   takes a LOW OpId.
/// * `CycleA` / `CycleB` write each other's read field: a genuine 2-op
///   SCC, which is what makes Kahn's queue stall.
/// * `TellTales` — the only emitter of the merge's trigger event — reads
///   a field `CycleA` writes, so it sits DOWNSTREAM of that SCC and can
///   never reach in-degree zero on its own. It is declared last, so it
///   takes a HIGH OpId.
///
/// Under the pre-fix force-pick the stall therefore emits the merge
/// (smallest un-emitted OpId) long before its producer — the exact
/// failure S5 hit. Under the shipped pick the merge is skipped while its
/// ring producer is pending, and the cycle is broken inside the SCC.
const RING_ORDER_FIXTURE: &str = r#"
// Synthetic scheduler fixture (S10). Not a game: every declaration
// exists to shape the dependency graph.

event Tick { }

@replayable @gpu_amenable
event RingProbeProwess {
  who: AgentId,
  whom: AgentId,
}

@replayable @gpu_amenable
event RingProbeTold {
  teller: AgentId,
}

entity Teller : Agent {
  pos: vec3,
  vel: vec3,
}

field ring_probe_a: f32
field ring_probe_b: f32

init {
  spawn Teller count 4 { hp: 100.0, pos: scatter(4.0) }
}

// The gossip carrier. `merge from` lowers a BeliefSocialMerge op that
// reads the LIVE event tail — it MUST run after the op that appends
// RingProbeTold, or it merges an empty ring.
@materialized(on_event = [RingProbeProwess], storage = pair_map)
belief rumour(observer: Agent, subject: Agent) -> f32 {
  initial: 0.0,
  on RingProbeProwess { who: w, whom: s }
    where (w == observer) && (s == subject)
    { self += 1200.0 }
  on RingProbeTold { teller: t } merge from t: max
  clamp: [0.0, 1000000.0],
}

// --- the SCC that makes Kahn's stall -------------------------------
physics CycleA @phase(per_agent) {
  on Tick {} where (self.alive) {
    agents.set_ring_probe_a(self, self.ring_probe_b + 1.0);
  }
}

physics CycleB @phase(per_agent) {
  on Tick {} where (self.alive) {
    agents.set_ring_probe_b(self, self.ring_probe_a + 1.0);
  }
}

// --- the first-hand emitter ----------------------------------------
physics SeeProwess @phase(per_agent) {
  on Tick {} where (self.alive && self.hp > 0.0) {
    emit RingProbeProwess { who: self, whom: self };
  }
}

// --- the merge's producer, DOWNSTREAM of the SCC --------------------
// Reads ring_probe_a (written by CycleA), so its in-degree never
// reaches zero until the cycle is force-broken. Declared last, so its
// OpId is larger than the merge op's.
physics TellTales @phase(per_agent) {
  on Tick {} where (self.alive && self.ring_probe_a > 0.0) {
    emit RingProbeTold { teller: self };
  }
}
"#;

/// Locate `(merge_op, told_producer_op)` in a lowered fixture.
fn merge_and_producer(prog: &CgProgram) -> (OpId, OpId) {
    let merge = prog
        .ops
        .iter()
        .position(|o| o.kind.label().contains("belief_social_merge"))
        .expect("the fixture declares a `merge from` belief");
    let graph = dependency_graph(prog);
    let producers: Vec<OpId> = ring_edges(&graph)
        .into_iter()
        .filter(|(_, c)| c.0 as usize == merge)
        .map(|(p, _)| p)
        .collect();
    assert_eq!(
        producers.len(),
        1,
        "expected exactly one ring producer for the merge op; got {producers:?} — if this \
         is ZERO the kind-refined ring edge failed to form, which is the S5b door and the \
         reason a sort-internal guard cannot close this class"
    );
    (OpId(merge as u32), producers[0])
}

// ---------------------------------------------------------------------
// 1. The class, reproduced and closed
// ---------------------------------------------------------------------

#[test]
fn synthetic_fixture_reproduces_the_class_under_the_legacy_force_pick() {
    let prog = lower_src("ring_order_fixture", RING_ORDER_FIXTURE);
    let graph = dependency_graph(&prog);
    let (merge, producer) = merge_and_producer(&prog);

    // Preconditions that make the reproduction meaningful.
    assert!(
        graph.has_cycle(),
        "the fixture must contain a cycle or the force-pick never runs"
    );
    assert!(
        merge.0 < producer.0,
        "the merge op must have the SMALLER OpId (merge #{}, producer #{}) — that is what \
         made the legacy global-smallest pick choose it",
        merge.0,
        producer.0
    );

    // The pre-fix scheduler mis-orders it...
    let legacy = legacy_best_effort(&graph);
    let legacy_bad = order_violations(&graph, &legacy);
    assert!(
        legacy_bad.contains(&(producer, merge)),
        "the legacy force-pick must schedule the merge ahead of its producer for this test \
         to be a regression test at all; violations = {legacy_bad:?}"
    );

    // ...and the shipped scheduler does not.
    let shipped = stage_order(&prog);
    assert!(
        order_violations(&graph, &shipped).is_empty(),
        "shipped schedule must honour every event-ring edge: {:?}",
        order_violations(&graph, &shipped)
    );
    let pos = |op: OpId| shipped.iter().position(|x| *x == op).unwrap();
    assert!(
        pos(producer) < pos(merge),
        "producer op#{} must be scheduled before merge op#{} (got stages {} and {})",
        producer.0,
        merge.0,
        pos(producer),
        pos(merge),
    );
}

#[test]
fn the_validator_is_loud_about_the_legacy_order() {
    // The point of the whole slice: had this check existed, S5's 65-stage
    // inversion would have been a build warning (and a hard error under
    // SIM_REQUIRE_ALL_RULES) instead of a dead feature nobody noticed.
    let prog = lower_src("ring_order_fixture", RING_ORDER_FIXTURE);
    let graph = dependency_graph(&prog);
    let (merge, producer) = merge_and_producer(&prog);

    let clean = validate_ring_order(&prog, &graph, &stage_order(&prog));
    assert!(
        !clean.iter().any(|i| i.severity == RingOrderSeverity::Bug),
        "the shipped schedule must validate clean: {clean:?}"
    );

    let issues = validate_ring_order(&prog, &graph, &legacy_best_effort(&graph));
    let hit = issues
        .iter()
        .find(|i| {
            matches!(
                i.kind,
                RingOrderIssueKind::ConsumerBeforeProducer { consumer, producer: p, .. }
                    if consumer == merge && p == producer
            )
        })
        .unwrap_or_else(|| panic!("validator missed the inversion: {issues:?}"));
    assert_eq!(
        hit.severity,
        RingOrderSeverity::Bug,
        "no cycle joins the merge to its producer, so this is a scheduler defect, not a \
         forced break: {hit:?}"
    );
    assert!(
        hit.message.contains("RingProbeTold"),
        "the diagnostic must name the event so a reader can find the dead feature: {}",
        hit.message
    );
}

// ---------------------------------------------------------------------
// 2. The S6b hypothesis, tested and corrected
// ---------------------------------------------------------------------

#[test]
fn spawn_statements_do_not_move_the_op_graph() {
    // S6b (webband-port plan) attributed a gossip failure to "spawn
    // statements are OPS, a new one renumbers the op graph". They are
    // not: `init { spawn … }` blocks are extracted host-side by
    // build_helper for the seeder and never become ComputeOps. Pinned
    // here so the next debugger does not re-chase that hypothesis — if
    // this ever fails, spawn lowering changed and the OpId-churn theory
    // becomes live again.
    let base = lower_src("base", RING_ORDER_FIXTURE);
    let grown = lower_src(
        "grown",
        &RING_ORDER_FIXTURE.replace(
            "  spawn Teller count 4 { hp: 100.0, pos: scatter(4.0) }",
            "  spawn Teller count 4 { hp: 100.0, pos: scatter(4.0) }\n  \
             spawn Teller count 1 { hp: 100.0, pos: scatter(5.0) }\n  \
             spawn Teller count 1 { hp: 100.0, pos: scatter(6.0) }\n  \
             spawn Teller count 1 { hp: 100.0, pos: scatter(7.0) }\n  \
             spawn Teller count 1 { hp: 100.0, pos: scatter(8.0) }",
        ),
    );
    assert_eq!(base.ops.len(), grown.ops.len(), "spawn stmts must not add ops");
    assert_eq!(
        stage_order(&base),
        stage_order(&grown),
        "spawn stmts must not move the schedule"
    );
}

// ---------------------------------------------------------------------
// 3. The build gate
// ---------------------------------------------------------------------

#[test]
fn require_all_rules_promotes_ring_order_bugs_to_a_build_error() {
    use dsl_compiler::build_helper::{check_ring_order, REQUIRE_ALL_RULES_ENV};

    // Manufacture the finding the way a broken schedule would.
    let prog = lower_src("ring_order_fixture", RING_ORDER_FIXTURE);
    let graph = dependency_graph(&prog);
    let issues = validate_ring_order(&prog, &graph, &legacy_best_effort(&graph));
    assert!(issues.iter().any(|i| i.severity == RingOrderSeverity::Bug));

    // Default (env unset): printed as cargo:warning, build continues —
    // the historic warn-and-continue posture every fixture relies on.
    std::env::remove_var(REQUIRE_ALL_RULES_ENV);
    assert!(check_ring_order("probe", &issues).is_ok());

    // Opted in: hard error naming the defect.
    std::env::set_var(REQUIRE_ALL_RULES_ENV, "1");
    let err = check_ring_order("probe", &issues).expect_err("must promote");
    std::env::remove_var(REQUIRE_ALL_RULES_ENV);
    assert!(err.contains("probe"), "{err}");
    assert!(err.contains("ring order"), "{err}");

    // A clean schedule never trips the gate, even opted in.
    std::env::set_var(REQUIRE_ALL_RULES_ENV, "1");
    let clean = validate_ring_order(&prog, &graph, &stage_order(&prog));
    let res = check_ring_order("probe", &clean);
    std::env::remove_var(REQUIRE_ALL_RULES_ENV);
    assert!(res.is_ok(), "clean schedule must not fail the build: {res:?}");
}

// ---------------------------------------------------------------------
// 4. The standing net: the whole corpus, every build
// ---------------------------------------------------------------------

#[test]
fn no_fixture_in_the_corpus_schedules_a_ring_consumer_before_its_producer() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets/sim");
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("read assets/sim")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sim"))
        .collect();
    entries.sort();

    let mut checked = 0usize;
    let mut ring_edge_total = 0usize;
    let mut info_total = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for path in entries {
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        // `import`-bearing fixtures need the multi-file driver; they have
        // their own suites.
        if src.lines().any(|l| l.trim_start().starts_with("import ")) {
            continue;
        }
        let Ok(program) = dsl_compiler::parse(&src) else {
            continue;
        };
        let _ = dsl_compiler::custom_agent_fields::populate(&program);
        let Ok(comp) = dsl_ast::resolve::resolve(program) else {
            continue;
        };
        // Partial lowers are still worth checking — a dropped rule does
        // not excuse a mis-ordered ring.
        let prog = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
            Ok(p) => p,
            Err(o) => o.program,
        };
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let graph = dependency_graph(&prog);
        let order = stage_order(&prog);
        ring_edge_total += ring_edges(&graph).len();
        for issue in validate_ring_order(&prog, &graph, &order) {
            match issue.severity {
                RingOrderSeverity::Bug => failures.push(format!("{name}: {}", issue.message)),
                RingOrderSeverity::Forced => {
                    // A genuinely cyclic ring relation. None exists in
                    // the corpus today; if one appears it must be a
                    // deliberate, documented fixture shape — fail here
                    // so it gets that documentation.
                    failures.push(format!("{name} (forced by a cycle): {}", issue.message))
                }
                RingOrderSeverity::Info => info_total += 1,
            }
        }
        checked += 1;
    }

    println!(
        "ring-order sweep: {checked} fixtures, {ring_edge_total} event-ring edges, \
         {info_total} producerless-kind notices"
    );
    assert!(checked > 40, "sweep should cover the corpus, saw {checked}");
    assert!(
        failures.is_empty(),
        "event-ring ordering defects ({}):\n{}",
        failures.len(),
        failures.join("\n"),
    );
}
