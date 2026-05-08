//! Wave 3 ToM Phase 3.5 closed-loop pin: agent A `scry`s agent B's
//! beliefs about subject C → agent A's BeliefState SoA cells about C
//! match agent B's BeliefState SoA cells about C (verbatim 6-column
//! copy).
//!
//! ## What this exercises
//!
//! Phase 3.5 introduced the runtime-side scry consumer that mirrors the
//! future GPU `EffectScryApplied` chronicle handler. Cross-observer
//! access: the caster reads another agent's belief row and folds it
//! into the caster's own belief row about the same subject.
//!
//! The DSL view-call lowering for kernel-side reads of
//! `agents.beliefs_<field>(observer, subject)` (the spec's struct-then-
//! field access syntax) IS landed in this phase as registered namespace
//! methods that emit WGSL stub functions. The actual cross-observer
//! cell-copy lives in this runtime CPU consumer until a future phase
//! emits a WGSL kernel from the chronicle stream.
//!
//! ## Fixture
//!
//! 4 agents. Agent B (= TARGET_OBSERVER, slot 1) is pre-seeded with
//! belief about agent C (= SUBJECT, slot 2):
//!   - beliefs_pos[1 * 4 + 2] = (10, 20, 30, 1)
//!   - beliefs_type[1 * 4 + 2] = 5
//!   - beliefs_tick[1 * 4 + 2] = 100
//!   - beliefs_confidence[1 * 4 + 2] = 200
//!   - beliefs_suspicion[1 * 4 + 2] = 50
//!   - beliefs_flags[1 * 4 + 2] = 0xCAFE
//!
//! Agent A (= OBSERVER, slot 0) calls `scry(0, 1, 2)` — read agent B's
//! beliefs about agent C, fold into A's beliefs about C.
//!
//! ## Asserts
//!
//! After the scry call: agent A's belief row about agent C
//! (`[0 * N + 2]`) matches agent B's belief row verbatim — every column.
//!
//! Per-cell isolation: every other (observer', subject') cell stays at
//! its pre-seed value (default zero except the source cell at
//! `[1 * 4 + 2]`).
//!
//! ## Skip path
//!
//! GPU adapter unavailable — same convention as `observe_round_trip_pin`.

use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xCAFE_FEED_BEEF_F00D;
const N: u32 = 4;
const OBSERVER: u32 = 0;
const TARGET_OBSERVER: u32 = 1;
const SUBJECT: u32 = 2;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

#[test]
fn scry_copies_target_observer_belief_row_verbatim() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[scry_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell_count = n * n;
    let src_cell = (TARGET_OBSERVER as usize) * n + (SUBJECT as usize);
    let dst_cell = (OBSERVER as usize) * n + (SUBJECT as usize);

    // Seed agent B's belief row about agent C with non-zero values.
    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[src_cell] = [10.0, 20.0, 30.0, 1.0];
    sim.seed_beliefs_pos(&pos_seed);

    let mut type_seed = vec![0u8; cell_count];
    type_seed[src_cell] = 5;
    sim.seed_beliefs_type(&type_seed);

    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[src_cell] = 100;
    sim.seed_beliefs_tick(&tick_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[src_cell] = 200;
    sim.seed_beliefs_confidence(&conf_seed);

    let mut susp_seed = vec![0u8; cell_count];
    susp_seed[src_cell] = 50;
    sim.seed_beliefs_suspicion(&susp_seed);

    let mut flags_seed = vec![0u32; cell_count];
    flags_seed[src_cell] = 0xCAFE;
    sim.seed_beliefs_flags(&flags_seed);

    // Cross-observer access: agent A reads agent B's belief row about C.
    sim.scry(OBSERVER, TARGET_OBSERVER, SUBJECT);

    // -- Asserts on the destination cell (OBSERVER, SUBJECT) --
    let pos = sim.beliefs_pos().to_vec();
    assert_eq!(
        pos[dst_cell], [10.0, 20.0, 30.0, 1.0],
        "scry should copy target_observer's beliefs_pos verbatim",
    );

    let types = sim.beliefs_type().to_vec();
    assert_eq!(
        types[dst_cell], 5,
        "scry should copy target_observer's beliefs_type verbatim",
    );

    let ticks = sim.beliefs_tick().to_vec();
    assert_eq!(
        ticks[dst_cell], 100,
        "scry should copy target_observer's beliefs_tick verbatim",
    );

    let confidence = sim.beliefs_confidence().to_vec();
    assert_eq!(
        confidence[dst_cell], 200,
        "scry should copy target_observer's beliefs_confidence verbatim \
         (NOT peg at 255 — that's observe's behaviour; scry inherits the \
         source's confidence as-is)",
    );

    let suspicion = sim.beliefs_suspicion().to_vec();
    assert_eq!(
        suspicion[dst_cell], 50,
        "scry should copy target_observer's beliefs_suspicion verbatim",
    );

    let flags = sim.beliefs_flags().to_vec();
    assert_eq!(
        flags[dst_cell], 0xCAFE,
        "scry should copy target_observer's beliefs_flags verbatim",
    );

    // -- Source cell is unchanged (scry is read-only on the source row).
    assert_eq!(pos[src_cell], [10.0, 20.0, 30.0, 1.0],
        "scry must not mutate the source cell");
    assert_eq!(types[src_cell], 5);
    assert_eq!(ticks[src_cell], 100);
    assert_eq!(confidence[src_cell], 200);
    assert_eq!(suspicion[src_cell], 50);
    assert_eq!(flags[src_cell], 0xCAFE);

    // -- Per-cell isolation: every other (o, s) cell stays at default 0.
    for o in 0..n {
        for s in 0..n {
            let idx = o * n + s;
            if idx == src_cell || idx == dst_cell {
                continue;
            }
            assert_eq!(
                pos[idx], [0.0; 4],
                "beliefs_pos[{o} * N + {s}] = {:?} (expected [0,0,0,0])",
                pos[idx],
            );
            assert_eq!(types[idx], 0, "beliefs_type[{o} * N + {s}] should stay 0");
            assert_eq!(ticks[idx], 0, "beliefs_tick[{o} * N + {s}] should stay 0");
            assert_eq!(
                confidence[idx], 0,
                "beliefs_confidence[{o} * N + {s}] should stay 0",
            );
            assert_eq!(
                suspicion[idx], 0,
                "beliefs_suspicion[{o} * N + {s}] should stay 0",
            );
            assert_eq!(
                flags[idx], 0,
                "beliefs_flags[{o} * N + {s}] should stay 0",
            );
        }
    }
}

/// Self-scry collapses to a no-op — caster reading their own belief row
/// about subject ends up with the same data they started with. This pin
/// proves the consumer's 6-column copy is correctly addressed (a swap
/// between observer/target_observer would surface here as a value drift).
#[test]
fn self_scry_is_idempotent() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[scry_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell_count = n * n;
    let cell = (OBSERVER as usize) * n + (SUBJECT as usize);

    // Seed agent A's belief row about subject C with a known tuple.
    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[cell] = [7.0, 8.0, 9.0, 1.0];
    sim.seed_beliefs_pos(&pos_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[cell] = 128;
    sim.seed_beliefs_confidence(&conf_seed);

    // Self-scry: agent A reads its own belief row about C.
    sim.scry(OBSERVER, OBSERVER, SUBJECT);

    let pos = sim.beliefs_pos().to_vec();
    assert_eq!(
        pos[cell], [7.0, 8.0, 9.0, 1.0],
        "self-scry must preserve the original belief tuple",
    );
    assert_eq!(sim.beliefs_confidence()[cell], 128);
}
