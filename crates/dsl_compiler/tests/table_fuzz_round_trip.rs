//! Sim-level fuzz harness for `table` decls — random valid `.sim`
//! snippets containing one or more `table <name>: u32[N] = [...]`
//! decls + a physics rule that reads from each, driven through
//! parse → resolve → lower. Sister to `belief_fuzz_round_trip.rs`;
//! covers the I.3b-era static-table primitive shipped in `df73557a`.
//!
//! Variation axes per generated table:
//!   * Length N ∈ [1, 64]. Spans the trivial single-element case
//!     up to "small lookup table" sizes (factions, ability tiers).
//!   * Element values: random u32 in [0, u32::MAX] (resolver
//!     bounds-checks against the declared `u32` element type).
//!   * Tables-per-file count ∈ [1, 4]. Multi-table fixtures exist
//!     (terrain costs + tile-bitmap + …) so the substring-scan in
//!     the kernel composer needs to handle them deterministically.
//!   * Per-table read pattern in the physics rule: each declared
//!     table gets exactly one `tables.<name>(index)` read so the
//!     kernel composer's prelude-decl emit fires.
//!
//! Each iteration: parse → resolve → lower. All three stages must
//! succeed. Failures report the offending source + stage that broke.

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
    fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u32() % (hi - lo + 1))
    }
}

fn gen_table_sim(rng: &mut Pcg, idx: usize) -> String {
    let n_tables = rng.range_u32(1, 4);
    let mut src = String::new();
    src.push_str("event Tick { }\n");
    // Emit a `Read` event the physics rule can fire each tick with
    // the table-read result as the payload. This keeps the rule
    // body real (no dead-code elimination of the table lookups)
    // without depending on `field <name>` registration via
    // `custom_agent_fields::populate` — which only runs through the
    // full `build_helper::compile_sim_file_to_kernels` path, not
    // the bare `parse → resolve → lower` pipeline the fuzz uses.
    src.push_str(&format!(
        "event Read{idx} {{ value: u32 }}\n"
    ));
    src.push_str(&format!("entity Probe{idx} : Agent {{ }}\n"));

    let mut table_specs: Vec<(String, u32)> = Vec::with_capacity(n_tables as usize);
    for t in 0..n_tables {
        let name = format!("t{idx}_{t}");
        let length = rng.range_u32(1, 64);
        src.push_str(&format!("table {name}: u32[{length}] = ["));
        for i in 0..length {
            if i > 0 {
                src.push_str(", ");
            }
            // Random u32. The resolver bounds-checks 0 ≤ v ≤
            // u32::MAX; rng.next_u32() naturally lands in range.
            src.push_str(&format!("{}", rng.next_u32()));
        }
        src.push_str("]\n");
        table_specs.push((name, length));
    }

    src.push_str("@phase(per_agent)\n");
    src.push_str(&format!("physics Read{idx} {{\n"));
    // No `where (self.alive)` guard: an emit-only rule (no agent-state
    // write) classifies PerEvent, and reading `self` in a PerEvent body is
    // rejected by the G2 well-formedness check (SelfRefInPerEventBody, added
    // in 7d02f20c). The guard was incidental to this fuzz's purpose (table
    // reads); dropping the self-read keeps every generated rule lowerable
    // while the `emit … { value: acc }` below still keeps the table lookups
    // live (no dead-code elimination).
    src.push_str("  on Tick {} {\n");
    src.push_str("    let acc = ");
    for (i, (name, length)) in table_specs.iter().enumerate() {
        if i > 0 {
            src.push_str(" + ");
        }
        // Qualified `tables.<name>(...)` form — bare `<name>(...)`
        // is reserved for view calls (the resolver rewrites
        // `view::<name>(...)` and bare `<name>(...)` into
        // `IrExpr::ViewCall` when the name resolves to a ViewRef);
        // tables aren't in that registry by design.
        src.push_str(&format!("tables.{name}(world.tick % {length}u)"));
    }
    src.push_str(";\n");
    src.push_str(&format!(
        "    emit Read{idx} {{ value: acc }}\n"
    ));
    src.push_str("  }\n");
    src.push_str("}\n");

    src
}

#[test]
fn fuzz_2000_random_table_decls_parse_resolve_lower() {
    const N: usize = 2000;
    const SEED: u64 = 0x7AB1_C0DE_F00D_8881;

    let mut rng = Pcg::new(SEED);
    let mut parsed_n = 0usize;
    let mut resolved_n = 0usize;
    let mut lowered_n = 0usize;
    let mut first_failure: Option<(usize, String, &'static str, String)> = None;

    for i in 0..N {
        let src = gen_table_sim(&mut rng, i);
        let prog = match dsl_compiler::parse(&src) {
            Ok(p) => p,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure =
                        Some((i, src.clone(), "parse", format!("{e}")));
                }
                continue;
            }
        };
        parsed_n += 1;
        let comp = match dsl_ast::resolve::resolve(prog) {
            Ok(c) => c,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure =
                        Some((i, src.clone(), "resolve", format!("{e}")));
                }
                continue;
            }
        };
        resolved_n += 1;
        match lower_compilation_to_cg(&comp) {
            Ok(_) => lowered_n += 1,
            Err(outcome) => {
                if first_failure.is_none() {
                    let joined: Vec<String> = outcome
                        .diagnostics
                        .iter()
                        .map(|d| format!("{d}"))
                        .collect();
                    first_failure =
                        Some((i, src.clone(), "lower", joined.join(" | ")));
                }
            }
        }
    }

    eprintln!(
        "[table_fuzz] {N} iterations: parsed={parsed_n}, resolved={resolved_n}, lowered={lowered_n}"
    );

    if let Some((i, src, stage, err)) = first_failure {
        panic!(
            "fuzz iteration {i} failed at {stage}:\n--- source ---\n{src}\n--- error ---\n{err}"
        );
    }
}

#[test]
fn table_bounds_check_rejects_negative_u32_values() {
    // The resolver bounds-checks each value against the declared
    // element type. A negative literal on a u32 table must surface
    // as a resolve-time error, not silently wrap.
    let src = "table bad: u32[3] = [1, -1, 2]\n";
    let prog = dsl_compiler::parse(src).expect("parse should succeed");
    let result = dsl_ast::resolve::resolve(prog);
    assert!(
        result.is_err(),
        "expected resolve to reject negative value in u32 table"
    );
}

#[test]
fn table_length_mismatch_rejected_at_resolve() {
    // Declared length 4 but only 3 initializer values — resolver
    // must catch this (otherwise we'd OOB-read at kernel time).
    let src = "table bad: u32[4] = [1, 2, 3]\n";
    let prog = dsl_compiler::parse(src).expect("parse should succeed");
    let result = dsl_ast::resolve::resolve(prog);
    assert!(
        result.is_err(),
        "expected resolve to reject length mismatch"
    );
}
