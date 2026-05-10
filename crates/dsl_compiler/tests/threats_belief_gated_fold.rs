//! Belief-gated threat fold — architectural pin.
//!
//! Today's threats fold (`build_view_fold_per_agent_event_scan_body`
//! at `crates/dsl_compiler/src/cg/emit/kernel.rs:2645`) hard-codes:
//!
//! ```wgsl
//! if (agent_busy_with_ability_id[source_candidate] == 0u) { return; }
//! ```
//!
//! That makes threat awareness OMNISCIENT — every observer sees every
//! cast regardless of whether they could actually have observed it.
//! The architectural intent (per `BeliefStateColumn::Flags` —
//! `crates/dsl_compiler/src/cg/data_handle.rs:1207`) is for the gate
//! to be observation-shaped:
//!
//! ```wgsl
//! let cell = beliefs_flags[observer * agent_cap + source_candidate];
//! if ((cell & (1u << BELIEF_BIT_OBSERVED_BUSY)) == 0u) { return; }
//! ```
//!
//! That one-line change lets the existing belief primitives
//! (`PlantBelief`, `Scry`, `Reveal`, `Disguise`, `Observe`) gate
//! threat awareness. The fixtures + tests for those verbs already
//! exist (spy_network_runtime, tom_probe_runtime); they write to
//! `beliefs_flags` via atomic-OR on the per-(observer, subject)
//! cell.
//!
//! This test demonstrates the SEMANTICS host-side using the same
//! BeliefStateColumn enum the GPU bindings would consume, with the
//! same fold predicate the modified kernel would use. When the
//! compiler swap lands, this test stays valid as the expected
//! behavioural specification — the GPU dispatch should produce the
//! same observer-by-observer divergence this test asserts.

use dsl_compiler::cg::data_handle::BeliefStateColumn;

/// Bit position in `beliefs_flags[observer * N + subject]` reserved
/// for "observer believes subject is busy casting". One bit per
/// (observer, subject) pair; cleared when the cast resolves /
/// interrupts (or via `EraseBelief`).
const BELIEF_BIT_OBSERVED_BUSY: u32 = 0;

/// Observer-keyed threat count. Mirrors what the GPU's
/// `view_storage_threats_primary[observer]` would hold under the
/// belief-gated fold.
fn run_belief_gated_threats_fold(
    agent_cap: u32,
    beliefs_flags: &[u32],
) -> Vec<f32> {
    let mut out: Vec<f32> = vec![0.0; agent_cap as usize];
    // Mirrors the (observer, source_candidate) double loop in the
    // PerAgentEventScan kernel — just with the belief-gated predicate
    // substituted for the raw busy lookup.
    for observer in 0..agent_cap {
        for source in 0..agent_cap {
            let cell = beliefs_flags[(observer * agent_cap + source) as usize];
            let bit = 1u32 << BELIEF_BIT_OBSERVED_BUSY;
            if (cell & bit) == 0 {
                continue;
            }
            // Body: `self += 1.0` (mirroring the dodger_probe.sim view
            // body verbatim — `self += 1.0` per matching source).
            out[observer as usize] += 1.0;
        }
    }
    out
}

/// Simulates the per-observer scoring kernel: Flee scores
/// `threats[obs]`, Idle scores 0.5, argmax wins. Mirrors the
/// `dodger_probe.sim` verb scoring expressions.
fn would_pick_flee(threats_for_observer: f32) -> bool {
    threats_for_observer > 0.5
}

#[test]
fn belief_gated_fold_produces_per_observer_threat_divergence() {
    const N: u32 = 3;
    // Slot layout:
    //   0: caster (the busy source — actually casting)
    //   1: observer A — knows about the cast (believes caster is busy)
    //   2: observer B — does NOT know (no belief planted)
    let cap = (N * N) as usize;
    let mut beliefs: Vec<u32> = vec![0u32; cap];

    // A's belief about caster = "caster is busy" (bit set).
    let bit = 1u32 << BELIEF_BIT_OBSERVED_BUSY;
    beliefs[(1 * N + 0) as usize] = bit;
    // B's belief about caster = nothing (bit clear).

    let threats = run_belief_gated_threats_fold(N, &beliefs);

    // Observer 0 (the caster itself): no beliefs set → no threats.
    assert_eq!(threats[0], 0.0, "caster's own threat view is empty");
    // Observer A (1): one believed-busy source → threats[1] = 1.0.
    assert_eq!(
        threats[1], 1.0,
        "observer A knows caster is busy → threats[1] = 1.0"
    );
    // Observer B (2): no beliefs about caster → threats[2] = 0.0.
    assert_eq!(
        threats[2], 0.0,
        "observer B does NOT know about cast → threats[2] = 0.0"
    );

    // Behavioural pin: same physical world (caster IS busy), but
    // the two observers' scoring DIVERGES based on their beliefs.
    assert!(
        would_pick_flee(threats[1]),
        "observer A picks Flee (sees the threat)"
    );
    assert!(
        !would_pick_flee(threats[2]),
        "observer B picks Idle (doesn't see the threat — caster could be hidden, disguised, out of LOS)"
    );

    // The divergence IS the proof: belief gating turns omniscient
    // threat awareness into observation-shaped awareness without
    // changing the threats view's signature.
}

/// Plants a belief by setting the bit, just like `EffectOp::PlantBelief`
/// would via atomic-OR on `beliefs_flags[observer * N + subject_idx]`.
/// Uses the engine's `PlantBelief.fact_bit` field semantics.
fn plant_belief(
    beliefs: &mut [u32],
    n: u32,
    observer: u32,
    subject: u32,
    fact_bit: u8,
) {
    let cell = (observer * n + subject) as usize;
    beliefs[cell] |= 1u32 << fact_bit;
}

/// Pin that the BeliefStateColumn::Flags column the engine exposes
/// IS where this gate would read from. If the column re-numbers or
/// renames, this test trips and the author updates the gate kernel
/// to match.
#[test]
fn belief_flags_column_is_the_target_of_the_gate() {
    // Discriminant pinned so the GPU kernel's binding-slot derivation
    // (`BeliefStateColumn::Flags as u8 == 5`) matches what the
    // belief-gated fold would request.
    assert_eq!(BeliefStateColumn::Flags as u8, 5);
    assert_eq!(BeliefStateColumn::Flags.binding_name(), "beliefs_flags");
}

/// Demonstrates the full ToM cycle producing a behavioural change.
/// Setup: caster is hidden → no observer has belief → no threats.
/// PlantBelief informs A → A's threat awareness lights up.
/// Disguise/Erase would clear it; not modelled here (those are
/// separate verbs with their own tests in spy_network_runtime).
#[test]
fn plant_belief_lights_up_observer_threat_awareness() {
    const N: u32 = 3;
    let mut beliefs: Vec<u32> = vec![0u32; (N * N) as usize];

    // Phase 0: caster is busy casting, but no observer knows.
    let threats_initial = run_belief_gated_threats_fold(N, &beliefs);
    assert_eq!(
        threats_initial,
        vec![0.0, 0.0, 0.0],
        "no beliefs planted → all observers see zero threats (caster is hidden)"
    );

    // Phase 1: someone (a herald, a scrier, the caster's own
    // un-disguised cast tell) plants the busy belief in observer A.
    // This is exactly what `EffectOp::PlantBelief { subject_idx: 0,
    // fact_bit: BELIEF_BIT_OBSERVED_BUSY }` does on GPU when applied
    // with caster=A.
    plant_belief(&mut beliefs, N, /*observer*/ 1, /*subject*/ 0,
                 BELIEF_BIT_OBSERVED_BUSY as u8);

    let threats_after_plant = run_belief_gated_threats_fold(N, &beliefs);
    assert_eq!(
        threats_after_plant,
        vec![0.0, 1.0, 0.0],
        "after PlantBelief on (A, caster, BUSY): only A's threats[1] lights up"
    );
    assert!(
        would_pick_flee(threats_after_plant[1]),
        "newly-informed observer A picks Flee"
    );
    assert!(
        !would_pick_flee(threats_after_plant[2]),
        "uninformed observer B still picks Idle"
    );
}
