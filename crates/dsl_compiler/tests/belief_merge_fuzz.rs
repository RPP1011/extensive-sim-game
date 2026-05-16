//! Plan I.4b fuzz harness for social-merge — generates random valid
//! `belief` decls with one `merge from <agent>: <op>` clause where
//! the source-agent field can be at ANY payload position in the event
//! (not always first). Validates parse + resolve + lower + emit, plus
//! that the emitted WGSL kernel reads from the correct event_ring
//! offset for each generated source position.
//!
//! Complements `belief_merge_field_offset.rs` (two hand-written
//! cases) by exhaustively varying source position, op, storage shape,
//! and event field count.

use dsl_compiler::cg::emit::emit_cg_program;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::schedule::{synthesize_schedule, ScheduleStrategy};

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
}

#[derive(Clone, Copy)]
enum Shape {
    SingleKey,
    PairKey,
}

/// Generate a `belief` decl with one social-merge clause where the
/// source-agent field is at `source_pos` in the event's payload.
/// Returns `(source_text, expected_event_ring_offset)`.
fn gen_merge_belief(rng: &mut Pcg, idx: usize) -> (String, u32) {
    let shape = if rng.next_u32() % 2 == 0 {
        Shape::SingleKey
    } else {
        Shape::PairKey
    };
    let op = rng.pick(&["bit_or", "max", "min", "replace"]);
    // Total event field count in 1..4 (Agent at source_pos + filler u32s).
    let n_fields = rng.range_u32(1, 4) as usize;
    let source_pos = rng.range_u32(0, (n_fields - 1) as u32) as usize;
    // Expected ring offset = header (2) + source_pos.
    let expected_offset = 2 + source_pos as u32;

    let event_name = format!("Evt{idx}");
    let source_binder = "src_actor";
    let source_field = "actor_id";

    // Build the event field list — Agent at source_pos, u32 fillers
    // elsewhere. Field names are distinct to avoid resolver
    // collisions.
    let mut field_decls: Vec<String> = Vec::with_capacity(n_fields);
    for i in 0..n_fields {
        if i == source_pos {
            field_decls.push(format!("{source_field}: Agent"));
        } else {
            field_decls.push(format!("fill_{i}: u32"));
        }
    }

    // Pattern bindings — same field set, but the source-agent field
    // binds to `src_actor` (the merge clause's source binder name);
    // others are wildcards.
    let mut pattern_bindings: Vec<String> = Vec::with_capacity(n_fields);
    for i in 0..n_fields {
        if i == source_pos {
            pattern_bindings.push(format!("{source_field}: {source_binder}"));
        } else {
            pattern_bindings.push(format!("fill_{i}: _"));
        }
    }

    let params = match shape {
        Shape::SingleKey => "observer: Agent",
        Shape::PairKey => "observer: Agent, subject: Agent",
    };

    let src = format!(
        "\
        event {event_name} {{ {} }}\n\
        belief b{idx}({params}) -> u32 {{\n\
          initial: 0,\n\
          on {event_name} {{ {} }} merge from {source_binder}: {op}\n\
        }}\n",
        field_decls.join(", "),
        pattern_bindings.join(", "),
    );
    (src, expected_offset)
}

#[test]
fn fuzz_500_random_merge_clauses_pick_correct_source_field_offset() {
    const N: usize = 500;
    const SEED: u64 = 0xB0_E_F_DE_AD_C0_DE_F0;

    let mut rng = Pcg::new(SEED);
    let mut total_ok = 0usize;
    let mut first_failure: Option<(usize, String, String)> = None;

    for i in 0..N {
        let (src, expected_offset) = gen_merge_belief(&mut rng, i);
        let prog = match dsl_compiler::parse(&src) {
            Ok(p) => p,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure = Some((i, src.clone(), format!("parse: {e}")));
                }
                continue;
            }
        };
        let comp = match dsl_ast::resolve::resolve(prog) {
            Ok(c) => c,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure = Some((i, src.clone(), format!("resolve: {e}")));
                }
                continue;
            }
        };
        let cg = match lower_compilation_to_cg(&comp) {
            Ok(p) => p,
            Err(o) => {
                if first_failure.is_none() {
                    first_failure = Some((
                        i,
                        src.clone(),
                        format!("lower: {}", o.diagnostics.iter().map(|d| format!("{d}")).collect::<Vec<_>>().join(" | ")),
                    ));
                }
                continue;
            }
        };
        let sched = synthesize_schedule(&cg, ScheduleStrategy::Default);
        let art = match emit_cg_program(&sched.schedule, &cg) {
            Ok(a) => a,
            Err(e) => {
                if first_failure.is_none() {
                    first_failure = Some((i, src.clone(), format!("emit: {e:?}")));
                }
                continue;
            }
        };
        let merge_kernel = art
            .wgsl_files
            .iter()
            .find(|(name, _)| name.starts_with(&format!("merge_b{i}_")));
        let Some((_, body)) = merge_kernel else {
            if first_failure.is_none() {
                first_failure = Some((
                    i,
                    src.clone(),
                    format!("merge kernel not emitted; expected merge_b{i}_*"),
                ));
            }
            continue;
        };
        // The expected event_ring read is the line:
        //   `let source_agent = event_ring[event_idx * 10u + <offset>u];`
        let expected_read = format!(
            "let source_agent = event_ring[event_idx * 10u + {expected_offset}u];"
        );
        if !body.contains(&expected_read) {
            if first_failure.is_none() {
                first_failure = Some((
                    i,
                    src.clone(),
                    format!(
                        "expected `{expected_read}` in body; body excerpt:\n{}",
                        &body[..body.len().min(1500)]
                    ),
                ));
            }
            continue;
        }
        total_ok += 1;
    }

    println!(
        "[merge fuzz] {total_ok}/{N} ({pct:.1}%) random merge beliefs lowered + emitted with correct source offset",
        pct = (total_ok as f64) / (N as f64) * 100.0
    );

    if let Some((i, src, err)) = first_failure {
        panic!(
            "[merge fuzz iter {i}] failed:\n--- source ---\n{src}\n--- error ---\n{err}"
        );
    }
    assert_eq!(
        total_ok, N,
        "every random merge belief must lower + emit with the correct source field offset"
    );
}
