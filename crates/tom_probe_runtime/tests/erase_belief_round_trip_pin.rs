//! Wave 3 ToM Phase 4 closed-loop pin: caster A `erase_belief`s
//! observer T's beliefs about subject S — after the call, the cells
//! corresponding to the bits in `fields` are cleared to 0; the others
//! retain their pre-erase values.
//!
//! ## What this exercises
//!
//! Phase 4 introduces 3 deception verbs; this pin proves the
//! erase_belief verb's chronicle event → consumer pipeline works
//! end-to-end. The chronicle event payload carries:
//!
//!   * actor=slot 2 (caster A — informational only),
//!   * target=slot 3 (the OBSERVER T whose row gets cleared),
//!   * subject_idx=slot 4 (the agent slot the belief is ABOUT),
//!   * fields=slot 5 (u8-in-u32 bitset; bit 0 = pos, 1 = type, 2 = tick,
//!                    3 = confidence, 4 = suspicion, 5 = flags).
//!
//! The .sim consumer (`physics_ApplyEraseBeliefUpdate`) emits 6 if-blocks
//! — one per BeliefState column — and zeroes the matching cells.
//!
//! ## Fixture
//!
//! 4 agents. Pre-seed every cell of T's row about S with non-zero values
//! so the post-erase zero is unambiguous. Cast erase_belief with FIELDS
//! = `0b00111111` (all 6 columns) at tick = 5. Verify all 6 cells are
//! cleared.
//!
//! ## Asserts
//!
//! After erase_belief with FIELDS = all-bits:
//!   - All 6 BeliefState cells at `[T * N + S]` are zero.
//!   - Other cells (different observer / different subject) retain
//!     whatever pre-seeded values they had.
//!
//! A second test verifies the bit-mask semantic: erase only positions 0
//! and 2 (= pos + tick), confirm the other 4 columns retain their
//! pre-seeded values.

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xC0DE_F00D_FACE_BABE;
const N: u32 = 4;
const CASTER: u32 = 0;
const TARGET: u32 = 1;
const SUBJECT: u32 = 2;
const ERASE_TICK: u64 = 5;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

fn cell_index(observer: u32, subject: u32) -> usize {
    (observer as usize) * (N as usize) + (subject as usize)
}

#[test]
fn erase_belief_with_all_fields_clears_all_six_columns() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[erase_belief_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // Pre-seed every BeliefState column with non-zero values so the
    // post-erase zero is loud. The seed_* helpers take a full N×N row;
    // we set the [TARGET * N + SUBJECT] cell + a neighbour and leave
    // everything else default zero.
    let cell = cell_index(TARGET, SUBJECT);
    let neighbour = cell_index(TARGET, 0); // different subject
    let cell_count = (N * N) as usize;

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[cell] = [1.0, 2.0, 3.0, 1.0];
    pos_seed[neighbour] = [9.0, 8.0, 7.0, 1.0];
    sim.seed_beliefs_pos(&pos_seed);

    let mut type_seed = vec![0u8; cell_count];
    type_seed[cell] = 11;
    type_seed[neighbour] = 22;
    sim.seed_beliefs_type(&type_seed);

    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 100;
    tick_seed[neighbour] = 200;
    sim.seed_beliefs_tick(&tick_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[cell] = 200;
    conf_seed[neighbour] = 100;
    sim.seed_beliefs_confidence(&conf_seed);

    let mut susp_seed = vec![0u8; cell_count];
    susp_seed[cell] = 50;
    susp_seed[neighbour] = 25;
    sim.seed_beliefs_suspicion(&susp_seed);

    let mut flag_seed = vec![0u32; cell_count];
    flag_seed[cell] = 0xCAFE;
    flag_seed[neighbour] = 0xBEEF;
    sim.seed_beliefs_flags(&flag_seed);

    // Tick the sim forward (Phase 1 producer fires; orthogonal to our
    // off-diagonal cells under test).
    for _ in 0..ERASE_TICK {
        sim.step();
    }
    assert_eq!(sim.tick(), ERASE_TICK);

    // Erase ALL 6 columns of [TARGET * N + SUBJECT].
    sim.erase_belief(CASTER, TARGET, SUBJECT, 0b00111111);

    // -- Asserts on the erased cell --
    // The setter `agents.set_beliefs_pos(t, s, vec3(0,0,0))` widens to
    // `vec4<f32>(v, 1.0)` per the std430 storage shape — so the
    // post-erase w-lane is 1.0, not 0.0. The xyz components must be 0.
    assert_eq!(sim.beliefs_pos()[cell], [0.0, 0.0, 0.0, 1.0], "pos xyz cleared");
    assert_eq!(sim.beliefs_type()[cell], 0, "type cleared");
    assert_eq!(sim.beliefs_tick()[cell], 0, "tick cleared");
    assert_eq!(sim.beliefs_confidence()[cell], 0, "confidence cleared");
    assert_eq!(sim.beliefs_suspicion()[cell], 0, "suspicion cleared");
    assert_eq!(sim.beliefs_flags()[cell], 0, "flags cleared");

    // -- Neighbour cell retained its pre-seeded values --
    assert_eq!(
        sim.beliefs_pos()[neighbour],
        [9.0, 8.0, 7.0, 1.0],
        "neighbour cell pos must NOT be touched by erase_belief",
    );
    assert_eq!(sim.beliefs_type()[neighbour], 22, "neighbour type intact");
    assert_eq!(sim.beliefs_tick()[neighbour], 200, "neighbour tick intact");
    assert_eq!(sim.beliefs_confidence()[neighbour], 100, "neighbour confidence intact");
    assert_eq!(sim.beliefs_suspicion()[neighbour], 25, "neighbour suspicion intact");
    assert_eq!(sim.beliefs_flags()[neighbour], 0xBEEF, "neighbour flags intact");
}

#[test]
fn erase_belief_with_partial_fields_clears_only_matching_bits() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[erase_belief_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let cell = cell_index(TARGET, SUBJECT);
    let cell_count = (N * N) as usize;

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[cell] = [1.0, 2.0, 3.0, 1.0];
    sim.seed_beliefs_pos(&pos_seed);

    let mut type_seed = vec![0u8; cell_count];
    type_seed[cell] = 11;
    sim.seed_beliefs_type(&type_seed);

    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 100;
    sim.seed_beliefs_tick(&tick_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[cell] = 200;
    sim.seed_beliefs_confidence(&conf_seed);

    let mut susp_seed = vec![0u8; cell_count];
    susp_seed[cell] = 50;
    sim.seed_beliefs_suspicion(&susp_seed);

    let mut flag_seed = vec![0u32; cell_count];
    flag_seed[cell] = 0xCAFE;
    sim.seed_beliefs_flags(&flag_seed);

    for _ in 0..ERASE_TICK {
        sim.step();
    }

    // Bitset: bit 0 = pos, bit 2 = tick → 0b00000101 = 5.
    sim.erase_belief(CASTER, TARGET, SUBJECT, 0b00000101);

    // pos AND tick cleared:
    assert_eq!(sim.beliefs_pos()[cell], [0.0, 0.0, 0.0, 1.0], "pos xyz cleared (bit 0; w widened by setter)");
    assert_eq!(sim.beliefs_tick()[cell], 0, "tick cleared (bit 2)");
    // type / confidence / suspicion / flags retain seeded values:
    assert_eq!(sim.beliefs_type()[cell], 11, "type intact (bit 1 not set)");
    assert_eq!(sim.beliefs_confidence()[cell], 200, "confidence intact (bit 3 not set)");
    assert_eq!(sim.beliefs_suspicion()[cell], 50, "suspicion intact (bit 4 not set)");
    assert_eq!(sim.beliefs_flags()[cell], 0xCAFE, "flags intact (bit 5 not set)");
}
