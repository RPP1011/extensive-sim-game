//! Wave 3 ToM Phase 3.5 closed-loop pin: agent A `reveal`s its beliefs
//! about subject C → every observer's BeliefState SoA cell about C
//! matches agent A's BeliefState SoA cell about C (verbatim 6-column
//! broadcast).
//!
//! ## What this exercises
//!
//! Phase 3.5 introduced the runtime-side reveal consumer that mirrors
//! the future GPU `EffectRevealApplied` chronicle handler. One-to-many
//! propagation: caster broadcasts its beliefs about a subject to all
//! observers in range (Phase 3.5 picks "all observers" — range-gated
//! reveal lands when spatial query infrastructure for "observers within
//! radius" matures).
//!
//! ## Fixture
//!
//! 4 agents. Agent A (= CASTER, slot 0) is pre-seeded with belief about
//! agent C (= SUBJECT, slot 2):
//!   - beliefs_pos[0 * 4 + 2] = (1, 2, 3, 1)
//!   - beliefs_type[0 * 4 + 2] = 9
//!   - beliefs_tick[0 * 4 + 2] = 50
//!   - beliefs_confidence[0 * 4 + 2] = 180
//!   - beliefs_suspicion[0 * 4 + 2] = 30
//!   - beliefs_flags[0 * 4 + 2] = 0xBEEF
//!
//! Agent A calls `reveal(0, 2)` — broadcast A's beliefs about C to all
//! observers.
//!
//! ## Asserts
//!
//! After the reveal call:
//! - For every observer slot o in 0..N: `beliefs_*[o * N + 2]` matches
//!   the seeded source cell `[0 * N + 2]` verbatim.
//! - Cells about subjects OTHER than C stay at default zero.
//!
//! ## Skip path
//!
//! GPU adapter unavailable — same convention as `observe_round_trip_pin`.

use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xCAFE_FEED_BEEF_F00D;
const N: u32 = 4;
const CASTER: u32 = 0;
const SUBJECT: u32 = 2;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

#[test]
fn reveal_broadcasts_caster_belief_row_to_every_observer() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[reveal_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell_count = n * n;
    let src_cell = (CASTER as usize) * n + (SUBJECT as usize);

    // Seed caster's belief row about subject C.
    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[src_cell] = [1.0, 2.0, 3.0, 1.0];
    sim.seed_beliefs_pos(&pos_seed);

    let mut type_seed = vec![0u8; cell_count];
    type_seed[src_cell] = 9;
    sim.seed_beliefs_type(&type_seed);

    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[src_cell] = 50;
    sim.seed_beliefs_tick(&tick_seed);

    let mut conf_seed = vec![0u8; cell_count];
    conf_seed[src_cell] = 180;
    sim.seed_beliefs_confidence(&conf_seed);

    let mut susp_seed = vec![0u8; cell_count];
    susp_seed[src_cell] = 30;
    sim.seed_beliefs_suspicion(&susp_seed);

    let mut flags_seed = vec![0u32; cell_count];
    flags_seed[src_cell] = 0xBEEF;
    sim.seed_beliefs_flags(&flags_seed);

    // One-to-many propagation: agent A broadcasts its beliefs about C.
    sim.reveal(CASTER, SUBJECT);

    // -- Asserts: every observer's belief row about subject C matches.
    let pos = sim.beliefs_pos().to_vec();
    let types = sim.beliefs_type().to_vec();
    let ticks = sim.beliefs_tick().to_vec();
    let confidence = sim.beliefs_confidence().to_vec();
    let suspicion = sim.beliefs_suspicion().to_vec();
    let flags = sim.beliefs_flags().to_vec();

    for observer in 0..n {
        let cell = observer * n + (SUBJECT as usize);
        assert_eq!(
            pos[cell], [1.0, 2.0, 3.0, 1.0],
            "beliefs_pos[{observer} * N + {SUBJECT}] should equal caster's belief row",
        );
        assert_eq!(
            types[cell], 9,
            "beliefs_type[{observer} * N + {SUBJECT}] should equal caster's belief row",
        );
        assert_eq!(
            ticks[cell], 50,
            "beliefs_tick[{observer} * N + {SUBJECT}] should equal caster's belief row",
        );
        assert_eq!(
            confidence[cell], 180,
            "beliefs_confidence[{observer} * N + {SUBJECT}] should equal caster's belief row",
        );
        assert_eq!(
            suspicion[cell], 30,
            "beliefs_suspicion[{observer} * N + {SUBJECT}] should equal caster's belief row",
        );
        assert_eq!(
            flags[cell], 0xBEEF,
            "beliefs_flags[{observer} * N + {SUBJECT}] should equal caster's belief row",
        );
    }

    // -- Per-cell isolation: cells about subjects OTHER than SUBJECT
    // stay at default zero.
    for o in 0..n {
        for s in 0..n {
            if s == SUBJECT as usize {
                continue;
            }
            let cell = o * n + s;
            assert_eq!(
                pos[cell], [0.0; 4],
                "beliefs_pos[{o} * N + {s}] should stay 0 — reveal is per-subject",
            );
            assert_eq!(types[cell], 0);
            assert_eq!(ticks[cell], 0);
            assert_eq!(confidence[cell], 0);
            assert_eq!(suspicion[cell], 0);
            assert_eq!(flags[cell], 0);
        }
    }
}

/// Reveal from a non-zero source (CASTER ≠ 0) ensures the consumer's
/// source-cell address is computed correctly. A regression that
/// hardcoded `caster=0` would pass the first test but fail here.
#[test]
fn reveal_with_non_zero_caster_addresses_correct_source_cell() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[reveal_round_trip_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell_count = n * n;
    const NON_ZERO_CASTER: u32 = 3;
    let src_cell = (NON_ZERO_CASTER as usize) * n + (SUBJECT as usize);

    let mut pos_seed = vec![[0.0f32; 4]; cell_count];
    pos_seed[src_cell] = [99.0, 88.0, 77.0, 1.0];
    sim.seed_beliefs_pos(&pos_seed);

    sim.reveal(NON_ZERO_CASTER, SUBJECT);

    let pos = sim.beliefs_pos().to_vec();
    for observer in 0..n {
        let cell = observer * n + (SUBJECT as usize);
        assert_eq!(
            pos[cell], [99.0, 88.0, 77.0, 1.0],
            "broadcast must source from caster=3, not caster=0",
        );
    }
}
