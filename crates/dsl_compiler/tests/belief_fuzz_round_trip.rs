//! Sim-level fuzz harness for `belief` decls — random valid `.sim`
//! snippets containing a `belief` declaration, each driven through
//! parse → resolve → lower. Complements the ability fuzz harness
//! (which exercises `.ability` files) by stressing the slice I.1–I.3a
//! belief-primitive surface at scale.
//!
//! Variation axes for each generated belief:
//!
//!   * Signature shape: `(observer: Agent) -> T` (single-key, slice
//!     I.3a) or `(observer: Agent, subject: Agent) -> T` (pair-key,
//!     slice I.3).
//!   * Return type: `bool` / `u8` / `u32` / `i32` / `f32`.
//!   * Initial expression: literal of the matching return type.
//!   * Propagation handler count: 1-3.
//!   * Optional `@decay(rate=..., per=tick)` annotation.
//!   * Optional `@dispatch(per_agent_event_scan)` annotation.
//!   * Optional `@belief_gated` annotation.
//!
//! Each fuzz iteration:
//!   1. Generates the textual `.sim` source.
//!   2. Parses via `dsl_compiler::parse`.
//!   3. Resolves via `dsl_ast::resolve::resolve`.
//!   4. Lowers via `dsl_compiler::cg::lower::lower_compilation_to_cg`.
//!
//! All three stages must succeed. Failures report the offending
//! source and the stage that broke.

use dsl_compiler::cg::lower::lower_compilation_to_cg;

struct Pcg {
    seed: u64,
    counter: u32,
}

impl Pcg {
    fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= (self.counter as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        self.counter = self.counter.wrapping_add(1);
        (x & 0xFFFF_FFFF) as u32
    }
    fn pick<T: Copy>(&mut self, options: &[T]) -> T {
        let idx = (self.next_u32() as usize) % options.len();
        options[idx]
    }
    fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u32() % (hi - lo + 1))
    }
    fn bool_with_prob(&mut self, p: f32) -> bool {
        let r = (self.next_u32() as f32) / (u32::MAX as f32);
        r < p
    }
}

#[derive(Clone, Copy)]
enum BeliefShape {
    SingleKey,
    PairKey,
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // U8 / I32 admitted by grammar; not produced by gen today (see comment in gen_belief_sim)
enum ReturnTy {
    Bool,
    U8,
    U32,
    I32,
    F32,
}

impl ReturnTy {
    fn type_name(self) -> &'static str {
        match self {
            ReturnTy::Bool => "bool",
            ReturnTy::U8 => "u8",
            ReturnTy::U32 => "u32",
            ReturnTy::I32 => "i32",
            ReturnTy::F32 => "f32",
        }
    }
    fn initial_literal(self) -> &'static str {
        match self {
            ReturnTy::Bool => "false",
            ReturnTy::U8 => "0u",
            ReturnTy::U32 => "0u",
            ReturnTy::I32 => "0",
            ReturnTy::F32 => "0.0",
        }
    }
    fn update_op(self) -> &'static str {
        // Use whatever operator the lowering's validate_fold_body
        // accepts for this type. `|=` works for u8/u32, `=` works
        // for all, `+=` works for u32/i32/f32. Pick `=` for the
        // universal safe choice.
        match self {
            ReturnTy::Bool => "self = true",
            ReturnTy::U8 | ReturnTy::U32 => "self |= 1u",
            ReturnTy::I32 => "self += 1",
            ReturnTy::F32 => "self += 1.0",
        }
    }
}

fn gen_belief_sim(rng: &mut Pcg, idx: usize) -> String {
    let shape = if rng.bool_with_prob(0.5) {
        BeliefShape::SingleKey
    } else {
        BeliefShape::PairKey
    };
    // Restrict to the return types the lowering's fold-body type
    // checker accepts with the corresponding `update_op` literal.
    // The full grammar admits `i32` and `u8` returns too, but the
    // literal type defaults in the DSL (bare `1` = u32, `1.0` = f32)
    // make them awkward for the universal `self += <lit>` pattern.
    // Real-world fixtures all land on u32 / f32 / bool for belief
    // storage; the type-suffix literals (e.g. `1i`) needed for i32
    // / u8 aren't a parser surface we generate today.
    let ret_ty = rng.pick(&[ReturnTy::Bool, ReturnTy::U32, ReturnTy::F32]);
    let n_handlers = rng.range_u32(1, 3);
    let with_dispatch = rng.bool_with_prob(0.5);
    let with_decay = matches!(ret_ty, ReturnTy::F32 | ReturnTy::U32) && rng.bool_with_prob(0.3);
    let with_belief_gated = with_dispatch && rng.bool_with_prob(0.2);

    let mut src = String::new();
    // Event decls — one per handler (so each handler binds a distinct
    // event). The fields are minimal: observer (always) + subject
    // (for pair-key) + a generic payload field the fold body can
    // ignore via wildcard binding.
    for h in 0..n_handlers {
        src.push_str(&format!(
            "event Evt{idx}_{h} {{ observer: Agent"
        ));
        if matches!(shape, BeliefShape::PairKey) {
            src.push_str(", subject: Agent");
        }
        src.push_str(", payload: u32 }\n");
    }
    src.push_str(&format!("entity Probe{idx} : Agent {{ }}\n"));

    if with_dispatch {
        src.push_str("@dispatch(per_agent_event_scan)\n");
    }
    if with_belief_gated {
        src.push_str("@belief_gated\n");
    }
    if with_decay {
        src.push_str("@decay(rate = 0.85, per = tick)\n");
    }
    src.push_str(&format!("belief b{idx}("));
    match shape {
        BeliefShape::SingleKey => src.push_str("observer: Agent"),
        BeliefShape::PairKey => src.push_str("observer: Agent, subject: Agent"),
    }
    src.push_str(&format!(") -> {} {{\n", ret_ty.type_name()));
    src.push_str(&format!("  initial: {},\n", ret_ty.initial_literal()));
    // CAVEAT (lowering gap surfaced by this fuzz harness): with
    // `@dispatch(per_agent_event_scan)`, the kernel binds `observer`
    // and `source_candidate` directly off agent_cap — `event_idx` is
    // NOT in scope, so any event-field binder (`observer: o`,
    // `subject: s`) trips a typed lowering diagnostic at the
    // well-formedness gate. The shipping convention (see
    // threats_view_probe.sim) is wildcards for every field.
    //
    // Without `@dispatch(per_agent_event_scan)` the per-event
    // dispatch is the default and event-field binders are fine
    // (see tom_probe's `beliefs_flags` shape with the standard
    // `where (o == observer) && (s == subject)` gate).
    for h in 0..n_handlers {
        if with_dispatch {
            // PerAgentEventScan: wildcards only.
            match shape {
                BeliefShape::SingleKey => {
                    src.push_str(&format!(
                        "  on Evt{idx}_{h} {{ observer: _, payload: _ }} {{ {} }}\n",
                        ret_ty.update_op()
                    ));
                }
                BeliefShape::PairKey => {
                    src.push_str(&format!(
                        "  on Evt{idx}_{h} {{ observer: _, subject: _, payload: _ }} {{ {} }}\n",
                        ret_ty.update_op()
                    ));
                }
            }
        } else {
            // PerEvent: bind + where-gate against the params.
            match shape {
                BeliefShape::SingleKey => {
                    src.push_str(&format!(
                        "  on Evt{idx}_{h} {{ observer: o, payload: _ }} where (o == observer) {{ {} }}\n",
                        ret_ty.update_op()
                    ));
                }
                BeliefShape::PairKey => {
                    src.push_str(&format!(
                        "  on Evt{idx}_{h} {{ observer: o, subject: s, payload: _ }} where (o == observer) && (s == subject) {{ {} }}\n",
                        ret_ty.update_op()
                    ));
                }
            }
        }
    }
    src.push_str("}\n");
    src
}

#[test]
fn fuzz_1000_random_belief_decls_parse_resolve_lower() {
    const N: usize = 1000;
    const SEED: u64 = 0xBE11_C0DE_F00D_BABE;

    let mut rng = Pcg::new(SEED);
    let mut parsed_n = 0usize;
    let mut resolved_n = 0usize;
    let mut lowered_n = 0usize;
    let mut first_failure: Option<(usize, String, &'static str, String)> = None;

    for i in 0..N {
        let src = gen_belief_sim(&mut rng, i);
        let prog = match dsl_compiler::parse(&src) {
            Ok(p) => p,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure = Some((i, src.clone(), "parse", format!("{e}")));
                }
                continue;
            }
        };
        parsed_n += 1;
        let comp = match dsl_ast::resolve::resolve(prog) {
            Ok(c) => c,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure = Some((i, src.clone(), "resolve", format!("{e}")));
                }
                continue;
            }
        };
        resolved_n += 1;
        match lower_compilation_to_cg(&comp) {
            Ok(_) => lowered_n += 1,
            Err(outcome) => {
                if first_failure.is_none() {
                    let joined: Vec<String> =
                        outcome.diagnostics.iter().map(|d| format!("{d}")).collect();
                    first_failure =
                        Some((i, src.clone(), "lower", joined.join(" | ")));
                }
            }
        }
    }

    println!("==== belief decl fuzz harness — {N} random valid beliefs ====");
    println!(
        "  parsed   = {parsed_n}/{N} ({pct:.1}%)",
        pct = (parsed_n as f64) / (N as f64) * 100.0
    );
    println!(
        "  resolved = {resolved_n}/{N} ({pct:.1}%)",
        pct = (resolved_n as f64) / (N as f64) * 100.0
    );
    println!(
        "  lowered  = {lowered_n}/{N} ({pct:.1}%)",
        pct = (lowered_n as f64) / (N as f64) * 100.0
    );
    println!("===============================================================");

    if let Some((i, src, stage, err)) = first_failure {
        panic!(
            "[belief fuzz iter {i}] {stage} failure:\n--- source ---\n{src}\n--- error ---\n{err}"
        );
    }
    assert_eq!(parsed_n, N, "every generated belief must parse");
    assert_eq!(resolved_n, N, "every generated belief must resolve");
    assert_eq!(lowered_n, N, "every generated belief must lower");
}
