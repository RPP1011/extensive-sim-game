//! Task 169 — pack-focus mechanic.
//!
//! When an agent commits an engagement, the `pack_focus_on_engagement`
//! DSL physics rule emits a `PackAssist` event once per nearby
//! same-species kin within 12 m of the engaging actor. The `pack_focus`
//! @materialized view folds those events into a per-(observer, target)
//! decayed scalar; scoring's Attack row stacks +0.4 when the observer's
//! pack_focus on a specific target crosses 0.5. Result: kin converge on
//! whichever enemy the first packmate engaged.
//!
//! These tests drive the view's fold directly (no full `step_full`
//! pipeline) because the scoring response is a pure function of view
//! state + tick, and we want to pin the fold + scoring wiring without
//! entangling with mask / cascade / movement. The spatial primitive
//! (`query.nearby_kin`) is exercised by task 167's fear_spread tests,
//! and the physics-rule emit path is smoke-tested by
//! `pipeline_engagement_triggers_pack_assist`.

use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_data::scoring::{
    PredicateDescriptor, ScoringEntry, MAX_MODIFIERS, SCORING_TABLE,
};
use engine::mask::MicroKind;
use glam::Vec3;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn spawn_wolf(state: &mut SimState, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos,
            hp: 80.0,
            max_hp: 80.0,
            ..Default::default()
        })
        .expect("wolf spawn")
}

fn spawn_human(state: &mut SimState, pos: Vec3) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos,
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        })
        .expect("human spawn")
}

/// 2 wolves + 1 human. Wolves at x=3/5, human at x=0. All within 12 m
/// of each other, so pack-focus radius catches both wolves on the
/// human-vs-wolf engagement. Kin-scan from wolf-1 returns [wolf-2];
/// kin-scan from wolf-2 returns [wolf-1]. A single `EngagementCommitted`
/// on (wolf-1, human) should emit one `PackAssist { observer: wolf-2,
/// target: human }`.
fn spawn_pack_fixture() -> (SimState, [AgentId; 2], AgentId) {
    let mut state = SimState::new(16, 0xF00D_D15E);
    let h1 = spawn_human(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let w1 = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));
    let w2 = spawn_wolf(&mut state, Vec3::new(5.0, 0.0, 0.0));
    (state, [w1, w2], h1)
}

/// Fixture for the "does not boost unrelated targets" test: 2 wolves + 2
/// humans. Wolves at x=3/5 (within kin-radius of each other), humans at
/// x=0, x=-3 (both within 12 m of wolf-1).
fn spawn_pack_fixture_two_humans() -> (SimState, [AgentId; 2], [AgentId; 2]) {
    let mut state = SimState::new(16, 0xF00D_D15E);
    let h1 = spawn_human(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let h2 = spawn_human(&mut state, Vec3::new(-3.0, 0.0, 0.0));
    let w1 = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));
    let w2 = spawn_wolf(&mut state, Vec3::new(5.0, 0.0, 0.0));
    (state, [w1, w2], [h1, h2])
}

// ---------------------------------------------------------------------------
// Scoring harness (mirrors fear_spread_rout's — SCORING_TABLE iteration is
// `pub(crate)` on `score_entry`, so we inline the logic here with the
// view-call dispatch covering only the views this test touches).
// ---------------------------------------------------------------------------

fn score_row_for(
    entry: &ScoringEntry,
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
) -> f32 {
    let mut score = entry.base;
    let count = entry.modifier_count as usize;
    let max = count.min(MAX_MODIFIERS);
    for i in 0..max {
        let row = &entry.modifiers[i];
        let pred = &row.predicate;
        match pred.kind {
            PredicateDescriptor::KIND_SCALAR_COMPARE => {
                let lhs = read_field_scalar(state, agent, target, pred.field_id);
                let mut tb = [0u8; 4];
                tb.copy_from_slice(&pred.payload[0..4]);
                let rhs = f32::from_le_bytes(tb);
                if compare(pred.op, lhs, rhs) {
                    score += row.delta;
                }
            }
            PredicateDescriptor::KIND_VIEW_SCALAR_COMPARE => {
                let lhs = eval_view(state, agent, target, pred);
                let mut tb = [0u8; 4];
                tb.copy_from_slice(&pred.payload[0..4]);
                let rhs = f32::from_le_bytes(tb);
                if compare(pred.op, lhs, rhs) {
                    score += row.delta;
                }
            }
            PredicateDescriptor::KIND_VIEW_GRADIENT => {
                let v = eval_view(state, agent, target, pred);
                if v.is_finite() {
                    score += v * row.delta;
                }
            }
            _ => {}
        }
    }
    score
}

fn read_field_scalar(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    field_id: u16,
) -> f32 {
    if field_id >= 0x4000 && field_id < 0x8000 {
        let t = match target {
            Some(x) => x,
            None => return f32::NAN,
        };
        return match field_id {
            0x4000 => state.agent_hp(t).unwrap_or(0.0),
            0x4002 => {
                let hp = state.agent_hp(t).unwrap_or(0.0);
                let mx = state.agent_max_hp(t).unwrap_or(1.0);
                if mx > 0.0 { hp / mx } else { 0.0 }
            }
            _ => f32::NAN,
        };
    }
    match field_id {
        0 => state.agent_hp(agent).unwrap_or(0.0),
        2 => {
            let hp = state.agent_hp(agent).unwrap_or(0.0);
            let mx = state.agent_max_hp(agent).unwrap_or(1.0);
            if mx > 0.0 { hp / mx } else { 0.0 }
        }
        _ => f32::NAN,
    }
}

fn eval_view(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    pred: &PredicateDescriptor,
) -> f32 {
    let slot0 = pred.payload[4];
    let slot1 = pred.payload[5];
    let resolve = |slot: u8| -> Option<AgentId> {
        match slot {
            PredicateDescriptor::ARG_SELF => Some(agent),
            PredicateDescriptor::ARG_TARGET => target,
            _ => None,
        }
    };
    match pred.field_id {
        PredicateDescriptor::VIEW_ID_THREAT_LEVEL => {
            let a = resolve(slot0).unwrap_or(agent);
            if slot1 == PredicateDescriptor::ARG_WILDCARD {
                state.views.threat_level.sum_for_first(a, state.tick)
            } else {
                let b = resolve(slot1).unwrap_or(agent);
                state.views.threat_level.get(a, b, state.tick)
            }
        }
        PredicateDescriptor::VIEW_ID_MY_ENEMIES => {
            let a = resolve(slot0).unwrap_or(agent);
            let b = resolve(slot1).unwrap_or(agent);
            state.views.my_enemies.get(a, b)
        }
        PredicateDescriptor::VIEW_ID_KIN_FEAR => {
            let a = resolve(slot0).unwrap_or(agent);
            if slot1 == PredicateDescriptor::ARG_WILDCARD {
                state.views.kin_fear.sum_for_first(a, state.tick)
            } else {
                let b = resolve(slot1).unwrap_or(agent);
                state.views.kin_fear.get(a, b, state.tick)
            }
        }
        PredicateDescriptor::VIEW_ID_PACK_FOCUS => {
            let a = resolve(slot0).unwrap_or(agent);
            if slot1 == PredicateDescriptor::ARG_WILDCARD {
                state.views.pack_focus.sum_for_first(a, state.tick)
            } else {
                let b = resolve(slot1).unwrap_or(agent);
                state.views.pack_focus.get(a, b, state.tick)
            }
        }
        _ => f32::NAN,
    }
}

fn compare(op: u8, lhs: f32, rhs: f32) -> bool {
    if lhs.is_nan() || rhs.is_nan() {
        return false;
    }
    match op {
        PredicateDescriptor::OP_LT => lhs < rhs,
        PredicateDescriptor::OP_LE => lhs <= rhs,
        PredicateDescriptor::OP_EQ => lhs == rhs,
        PredicateDescriptor::OP_GE => lhs >= rhs,
        PredicateDescriptor::OP_GT => lhs > rhs,
        PredicateDescriptor::OP_NE => lhs != rhs,
        _ => false,
    }
}

fn attack_entry() -> &'static ScoringEntry {
    SCORING_TABLE
        .iter()
        .find(|e| e.action_head == MicroKind::Attack as u16)
        .expect("SCORING_TABLE missing Attack row")
}

/// Drive the `pack_focus` view's fold directly, bypassing the physics
/// rule but matching its emit shape. One call = one `PackAssist` event
/// folded.
fn prime_pack_focus(state: &mut SimState, observer: AgentId, target: AgentId) {
    let ev = Event::PackAssist {
        observer,
        target,
        tick: state.tick,
    };
    state.views.pack_focus.fold_event(&ev, state.tick);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Folding one PackAssist event bumps the observer's pack_focus scalar
/// above the 0.5 threshold the Attack row gates on. Mirror of
/// fear_spread's `one_fear_spread_crosses_threshold`.
#[test]
fn pack_assist_fold_bumps_view() {
    let (mut state, wolves, human) = spawn_pack_fixture();
    let [_w1, w2] = wolves;

    // Baseline: no pack_focus recorded.
    assert!(
        (state.views.pack_focus.get(w2, human, state.tick)).abs() < 1e-5,
        "baseline pack_focus should be 0",
    );

    // One packmate engagement → one fold → value = +1.0, above 0.5 gate.
    prime_pack_focus(&mut state, w2, human);
    let v = state.views.pack_focus.get(w2, human, state.tick);
    assert!(
        v > 0.5,
        "pack_focus {} should cross 0.5 gate after one engagement",
        v,
    );
    assert!(
        (v - 1.0).abs() < 1e-4,
        "pack_focus fold-amount is hardcoded +1.0 per event (emit_fold_arm); got {}",
        v,
    );
}

/// The Attack row gains +0.4 on the pack_focus modifier once the scalar
/// crosses 0.5 — the key behavioral pin for pack hunting. Also confirms
/// the modifier fires through the generated SCORING_TABLE, not just the
/// view.
#[test]
fn pack_focus_boosts_attack_on_same_target() {
    let (mut state, wolves, human) = spawn_pack_fixture();
    let [_w1, w2] = wolves;
    let entry = attack_entry();

    // Baseline Attack(w2, human): self fresh (hp_pct=1.0 ≥ 0.8) → +0.5.
    // human fresh (hp_pct=1.0), no threat_level, no my_enemies, no
    // pack_focus. Base 0.0 + 0.5 = 0.5.
    let s0 = score_row_for(entry, &state, w2, Some(human));
    assert!(
        (s0 - 0.5).abs() < 1e-4,
        "baseline Attack(w2, human) = {s0}, expected ≈0.5",
    );

    // Prime pack_focus: one packmate engagement → scalar = 1.0 > 0.5 → +0.4.
    prime_pack_focus(&mut state, w2, human);
    let s1 = score_row_for(entry, &state, w2, Some(human));
    assert!(
        s1 > s0 + 0.35,
        "Attack should gain ≥+0.35 from pack_focus modifier (+0.4 expected, some epsilon); got {s0} → {s1}",
    );
    assert!(
        (s1 - 0.9).abs() < 1e-3,
        "post-assist Attack(w2, human) = {s1}, expected ≈0.9 (0.5 fresh-self + 0.4 pack_focus)",
    );
}

/// Pack focus is keyed on (observer, *specific* target) — not a wildcard
/// sum. With a second human in the fixture, priming pack_focus on
/// human_a must NOT boost the Attack score on human_b. If the scoring
/// row slipped to wildcard slot, every candidate would tie on the
/// bump and the focus would vanish.
#[test]
fn pack_focus_does_not_boost_other_targets() {
    let (mut state, wolves, humans) = spawn_pack_fixture_two_humans();
    let [_w1, w2] = wolves;
    let [h_engaged, h_other] = humans;
    let entry = attack_entry();

    let attack_other_pre = score_row_for(entry, &state, w2, Some(h_other));

    // Prime pack_focus only on h_engaged.
    prime_pack_focus(&mut state, w2, h_engaged);

    // Attack on the *other* human should be unchanged.
    let attack_other_post = score_row_for(entry, &state, w2, Some(h_other));
    assert_eq!(
        attack_other_post, attack_other_pre,
        "pack_focus on h_engaged should not boost Attack(w2, h_other); \
         got {attack_other_pre} → {attack_other_post}",
    );

    // Sanity: Attack on the *engaged* human DID gain the boost.
    let attack_engaged_post = score_row_for(entry, &state, w2, Some(h_engaged));
    assert!(
        attack_engaged_post > attack_other_post + 0.35,
        "Attack(w2, h_engaged) = {attack_engaged_post} should be \
         ~0.4 higher than Attack(w2, h_other) = {attack_other_post} — \
         only the engaged target got the pack_focus bump",
    );
}

/// Decay: after ~20 ticks (~2 half-lives at rate 0.933 → 0.933^20 ≈
/// 0.25 < 0.5), the pack_focus boost vanishes. Proves pack focus is a
/// transient signal — kin stop converging once the beacon fades.
#[test]
fn pack_focus_decays_below_threshold() {
    let (mut state, wolves, human) = spawn_pack_fixture();
    let [_w1, w2] = wolves;

    prime_pack_focus(&mut state, w2, human);
    let immediate = state.views.pack_focus.get(w2, human, state.tick);
    assert!(immediate > 0.5, "immediate pack_focus {immediate} should be > 0.5");

    // 20 ticks × 0.933 ≈ 0.25 — below the 0.5 gate with margin.
    let decayed = state.views.pack_focus.get(w2, human, state.tick + 20);
    assert!(
        decayed < 0.5,
        "after 20 ticks pack_focus {decayed} should decay below 0.5 (beacon elapsed)",
    );

    // And at that tick Attack loses the +0.4 modifier.
    state.tick += 20;
    let entry = attack_entry();
    let s = score_row_for(entry, &state, w2, Some(human));
    assert!(
        (s - 0.5).abs() < 1e-3,
        "post-decay Attack = {s} should be back to baseline 0.5 \
         (fresh-self +0.5, no pack_focus bump)",
    );
}

/// Symmetric: humans converge on an engaged wolf. The view / physics
/// is species-agnostic (scoping lives in `query.nearby_kin`), so the
/// same mechanic should fire for humans too.
#[test]
fn humans_converge_on_engaged_wolf_same_mechanic() {
    let mut state = SimState::new(16, 0xF00D_D15E);
    let _h1 = spawn_human(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let h2 = spawn_human(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let wolf = spawn_wolf(&mut state, Vec3::new(5.0, 0.0, 0.0));

    // h1 engages wolf → h2 (kin) sees PackAssist → h2's Attack on wolf boosts.
    prime_pack_focus(&mut state, h2, wolf);

    let entry = attack_entry();
    let s = score_row_for(entry, &state, h2, Some(wolf));
    // h2 fresh, wolf fresh → +0.5 from self hp_pct + 0.4 from pack_focus = 0.9.
    assert!(
        (s - 0.9).abs() < 1e-3,
        "human Attack(h2, wolf) after pack_focus = {s}, expected ≈0.9 \
         (same +0.4 as wolves)",
    );
}

/// Primary behavioral assertion: second wolf's Attack score on the
/// already-engaged human is higher than Attack on Nothing and Hold, so
/// the second wolf would pick Attack on that specific human. Directly
/// pins the "convergence" behavior the task calls for.
#[test]
fn wolves_converge_on_engaged_human() {
    let (mut state, wolves, human) = spawn_pack_fixture();
    let [_w1, w2] = wolves;
    let entry_attack = attack_entry();
    let entry_hold = SCORING_TABLE
        .iter()
        .find(|e| e.action_head == MicroKind::Hold as u16)
        .expect("SCORING_TABLE missing Hold row");

    // Before the beacon: Attack(w2, human) has base +0.5 (fresh self),
    // which already beats Hold (0.1). The test here is that the
    // post-beacon score is strictly *higher* than the pre-beacon score —
    // pack_focus is the difference-maker that would break a tie with a
    // different candidate of equal HP.
    let attack_pre = score_row_for(entry_attack, &state, w2, Some(human));
    let hold_score = score_row_for(entry_hold, &state, w2, None);
    assert!(
        attack_pre > hold_score,
        "baseline Attack({attack_pre}) should already beat Hold({hold_score})",
    );

    // w1 engages the human — emit a PackAssist at w2 (the surviving
    // packmate). Shape matches what the physics rule
    // (`pack_focus_on_engagement`) produces from EngagementCommitted.
    prime_pack_focus(&mut state, w2, human);

    // After the beacon: Attack(w2, human) strictly higher than before,
    // by ~+0.4. Any same-hp alternate target (no pack_focus) would
    // stay at the pre-beacon score of 0.5 — so the wolves converge on
    // the engaged human rather than splitting attention.
    let attack_post = score_row_for(entry_attack, &state, w2, Some(human));
    assert!(
        attack_post > attack_pre + 0.35,
        "post-beacon Attack({attack_post}) should be ≥+0.35 over pre({attack_pre}) — \
         +0.4 expected with epsilon",
    );
    assert!(
        attack_post > hold_score,
        "post-beacon Attack({attack_post}) still beats Hold({hold_score})",
    );
}

// ---------------------------------------------------------------------------
// Pipeline smoke — does the generated physics rule actually emit
// PackAssist when called? The direct-fold tests above cover the scoring
// math; this ties it to the emit path so a broken cascade wiring
// surfaces.
// ---------------------------------------------------------------------------

#[test]
fn pipeline_engagement_triggers_pack_assist() {
    let (mut state, wolves, human) = spawn_pack_fixture();
    let [w1, w2] = wolves;
    let mut events = EventRing::with_cap(64);

    // Call the generated physics fn directly. Same signature the
    // dispatcher uses; simulates what happens on an
    // `EngagementCommitted { actor: w1, target: human }`.
    engine::generated::physics::pack_focus_on_engagement::pack_focus_on_engagement(
        w1,
        human,
        &mut state,
        &mut events,
    );

    let assists: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::PackAssist { observer, target, .. } => Some((*observer, *target)),
            _ => None,
        })
        .collect();

    assert_eq!(
        assists.len(),
        1,
        "expected 1 PackAssist event (one kin for w1: w2 only), got {:?}",
        assists,
    );
    let (obs, tgt) = assists[0];
    assert_eq!(obs, w2, "observer should be the surviving packmate w2");
    assert_eq!(tgt, human, "target should be the engaged human");
}

/// Non-same-species agents don't produce PackAssist. Spawn a lone wolf
/// with only humans around; the wolf's kin-scan returns empty, so no
/// PackAssist fires when it engages. Guards against an accidental
/// cross-species leak in `query.nearby_kin`.
#[test]
fn lone_wolf_engagement_emits_no_pack_assist() {
    let mut state = SimState::new(8, 0);
    let lone = spawn_wolf(&mut state, Vec3::ZERO);
    let h1 = spawn_human(&mut state, Vec3::new(2.0, 0.0, 0.0));
    spawn_human(&mut state, Vec3::new(4.0, 0.0, 0.0));
    let mut events = EventRing::with_cap(16);

    engine::generated::physics::pack_focus_on_engagement::pack_focus_on_engagement(
        lone,
        h1,
        &mut state,
        &mut events,
    );

    let assists = events
        .iter()
        .filter(|e| matches!(e, Event::PackAssist { .. }))
        .count();
    assert_eq!(
        assists, 0,
        "lone wolf engagement should emit no PackAssist (no same-species kin)",
    );
}
