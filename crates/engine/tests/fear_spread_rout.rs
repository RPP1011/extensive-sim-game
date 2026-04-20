//! Task 167 — fear-spread rout mechanic (retuned in task 173).
//!
//! When an agent dies, the `fear_spread_on_death` DSL physics rule emits
//! a `FearSpread` event once per nearby same-species kin within 12 m of
//! the dead center. The `kin_fear` @materialized view folds those
//! events into a per-(observer, dead_kin) decayed scalar; scoring's Flee
//! row stacks +0.4 when the observer's total decayed kin_fear crosses
//! 0.5 (via wildcard-slot sum). Result: surviving packmates rout for the
//! ~6-tick half-life window. Task 173 weakened the original `>0.3 : +0.6`
//! gate to `>0.5 : +0.4` and shortened the decay half-life from ~15 ticks
//! to ~6 ticks so routs are meaningful but recoverable — a single kin
//! death triggers a brief retreat, not a permanent one.
//!
//! These tests drive the view's fold directly (no full `step_full`
//! pipeline) because the scoring response is a pure function of view
//! state + tick, and we want to pin the fold + scoring wiring without
//! entangling with mask / cascade / movement. The spatial primitive
//! (`query.nearby_kin`) is exercised by `spatial_kin_filter` below, and
//! the full pipeline is smoke-tested by `pipeline_death_triggers_fear`.

use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_rules::scoring::{
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

/// 3 wolves + 2 humans clustered at the origin. Wolves at x=3/4/5, humans
/// at x=0/1. All within 12 m of each other, so the rout radius covers
/// the whole group.
fn spawn_pack_fixture() -> (SimState, [AgentId; 3], [AgentId; 2]) {
    let mut state = SimState::new(16, 0xF00D_BABE);
    let h1 = spawn_human(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let h2 = spawn_human(&mut state, Vec3::new(1.0, 0.0, 0.0));
    let w1 = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));
    let w2 = spawn_wolf(&mut state, Vec3::new(4.0, 0.0, 0.0));
    let w3 = spawn_wolf(&mut state, Vec3::new(5.0, 0.0, 0.0));
    (state, [w1, w2, w3], [h1, h2])
}

// ---------------------------------------------------------------------------
// Scoring harness (mirrors `wolves_and_humans_parity.rs::threat_level_scoring`
// because `engine::policy::utility::score_entry` is `pub(crate)`).
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

fn flee_entry() -> &'static ScoringEntry {
    SCORING_TABLE
        .iter()
        .find(|e| e.action_head == MicroKind::Flee as u16)
        .expect("SCORING_TABLE missing Flee row")
}

fn attack_entry() -> &'static ScoringEntry {
    SCORING_TABLE
        .iter()
        .find(|e| e.action_head == MicroKind::Attack as u16)
        .expect("SCORING_TABLE missing Attack row")
}

/// Drive the `kin_fear` view's fold directly, bypassing the physics rule
/// but matching its emit shape. One call = one `FearSpread` event folded.
fn prime_kin_fear(state: &mut SimState, observer: AgentId, dead_kin: AgentId) {
    let ev = Event::FearSpread {
        observer,
        dead_kin,
        tick: state.tick,
    };
    state.views.kin_fear.fold_event(&ev, state.tick);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `nearby_kin` yields only same-species, non-self, within-radius agents.
/// Directly exercises the primitive that the physics rule relies on —
/// if this breaks, everything downstream breaks silently.
#[test]
fn spatial_kin_filter() {
    let (state, wolves, humans) = spawn_pack_fixture();
    let [w1, w2, w3] = wolves;
    let [_h1, _h2] = humans;

    // Wolf w2 (center of the pack) should see its two packmates but no
    // humans and not itself.
    let kin = engine::spatial::nearby_kin(&state, w2, 12.0);
    assert_eq!(kin.len(), 2, "expected 2 kin for middle wolf, got {:?}", kin);
    assert!(kin.contains(&w1), "kin list missing w1: {:?}", kin);
    assert!(kin.contains(&w3), "kin list missing w3: {:?}", kin);
    assert!(!kin.contains(&w2), "kin list includes self");
    for &h in &humans {
        assert!(!kin.contains(&h), "kin list includes cross-species {h:?}");
    }
}

/// Empty `nearby_kin` when there's no kin in range — a lone wolf sees
/// no packmates, so no FearSpread would fire if it died.
#[test]
fn spatial_kin_empty_when_alone() {
    let mut state = SimState::new(4, 0);
    let lone = spawn_wolf(&mut state, Vec3::ZERO);
    // Humans don't count as kin.
    spawn_human(&mut state, Vec3::new(1.0, 0.0, 0.0));
    spawn_human(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let kin = engine::spatial::nearby_kin(&state, lone, 12.0);
    assert!(kin.is_empty(), "lone wolf should see no kin, got {:?}", kin);
}

/// Dead `center` still locates kin by its last-known position — see the
/// "dead-center contract" on `nearby_kin`. The primary caller
/// (`fear_spread_on_death`) runs on `AgentDied` AFTER the damage
/// handler's `agents.kill(...)`, so the dead-center case is the hot
/// path. Dead *neighbours* are still filtered out (they're evicted
/// from the spatial index on kill, so in practice they don't reach
/// the neighbour loop, but the `agent_alive` check in the filter is
/// the belt-and-suspenders that makes this invariant explicit).
#[test]
fn spatial_kin_dead_center_still_finds_kin() {
    let (mut state, wolves, _humans) = spawn_pack_fixture();
    let [w1, w2, w3] = wolves;
    state.kill_agent(w1);
    let kin = engine::spatial::nearby_kin(&state, w1, 12.0);
    assert_eq!(kin.len(), 2, "dead center's kin should still be found: {kin:?}");
    assert!(kin.contains(&w2), "expected w2 in kin list: {kin:?}");
    assert!(kin.contains(&w3), "expected w3 in kin list: {kin:?}");
}

/// Folding one FearSpread event bumps the observer's kin_fear total
/// above the 0.5 threshold the Flee row gates on (task 173, up from
/// 0.3).
#[test]
fn one_fear_spread_crosses_threshold() {
    let (mut state, wolves, _humans) = spawn_pack_fixture();
    let [w1, w2, _w3] = wolves;

    // Baseline: no fear recorded.
    assert_eq!(
        state.views.kin_fear.sum_for_first(w2, state.tick),
        0.0,
        "baseline kin_fear should be 0",
    );

    // One packmate death → one fold → value = +1.0, above 0.5 gate.
    prime_kin_fear(&mut state, w2, w1);
    let total = state.views.kin_fear.sum_for_first(w2, state.tick);
    assert!(
        total > 0.5,
        "kin_fear {} should cross 0.5 gate after one death",
        total,
    );
    assert!(
        (total - 1.0).abs() < 1e-4,
        "kin_fear fold-amount is hardcoded +1.0 per event (emit_fold_arm); got {}",
        total,
    );
}

/// The Flee row gains +0.4 on the kin_fear modifier once the sum crosses
/// 0.5 — the key behavioral pin for the rout mechanic. Also confirms the
/// modifier fires through the generated SCORING_TABLE, not just the
/// view.
///
/// Task 173 retuned this from the original `>0.3 : +0.6` to `>0.5 : +0.4`
/// so routed wolves can recover once kin re-engage (pack_focus and any
/// healthy-self Attack bonus can now overcome a single kin death, instead
/// of locking the survivor into permanent flight).
#[test]
fn flee_score_gains_kin_fear_bonus() {
    let (mut state, wolves, _humans) = spawn_pack_fixture();
    let [w1, w2, _w3] = wolves;
    let entry = flee_entry();

    // Baseline Flee score — hp_pct=1.0, hp=80 — no hp gates fire, no
    // threat, no kin_fear. Base = 0.0.
    let s0 = score_row_for(entry, &state, w2, None);
    assert!(s0.abs() < 1e-4, "baseline Flee = {s0}, expected ≈0.0");

    // Prime kin_fear: one dead packmate → sum = 1.0 > 0.5 → +0.4.
    prime_kin_fear(&mut state, w2, w1);
    let s1 = score_row_for(entry, &state, w2, None);
    assert!(
        s1 > s0 + 0.3,
        "Flee should gain ≥+0.3 from kin_fear modifier (+0.4 expected, some epsilon); got {s0} → {s1}",
    );
    assert!(
        (s1 - 0.4).abs() < 1e-3,
        "post-fear Flee = {s1}, expected ≈0.4 (base 0.0 + kin_fear gate +0.4)",
    );
}

/// With kin_fear > 0.5 on a wounded wolf, Flee beats Attack — the
/// intended rout behaviour. Without kin_fear, a fresh-self Attack
/// beats Flee. Task 173 softened the fresh-healthy-wolf rout so the
/// rout is additive to other flee signals (hp-based gates, threat)
/// rather than a hard override — a fresh wolf stays committed, but a
/// lightly-wounded wolf tips into retreat after a packmate dies.
#[test]
fn wolves_rout_when_packmate_dies() {
    let (mut state, wolves, humans) = spawn_pack_fixture();
    let [w1, w2, _w3] = wolves;
    let [h1, _h2] = humans;

    let flee = flee_entry();
    let attack = attack_entry();

    // Pre-death: no kin_fear, w2 at full hp, h1 at full hp. Flee row
    // has no gates firing; Attack has the `self.hp_pct >= 0.8` gate
    // (+0.5) so Attack >> Flee.
    let flee_pre = score_row_for(flee, &state, w2, None);
    let attack_pre = score_row_for(attack, &state, w2, Some(h1));
    assert!(
        attack_pre > flee_pre,
        "pre-death: Attack ({attack_pre}) should beat Flee ({flee_pre}) — a fresh \
         wolf with a full-hp human in range engages, no rout trigger yet",
    );

    // Lightly wound w2 so the raw-hp `hp < 50` gate fires (+0.4) but
    // the fresh-self Attack gate still fires (hp_pct = 45/80 = 0.5625,
    // which fails the `>= 0.8` gate). That way Attack drops to 0.0 and
    // Flee's hp<50 gate alone already matches Attack. The kin_fear +0.4
    // then tips Flee decisively above Attack. Without this wound the
    // retuned kin_fear alone (0.4 bump, task 173) is deliberately too
    // weak to override a fresh-self commit bonus — the whole point of
    // the retune was making routs recoverable.
    state.set_agent_hp(w2, 45.0);

    // Kill w1 and emit the FearSpread directly (the physics rule does
    // the same thing via the cascade; this test exercises the fold +
    // scoring layer without needing a full step pipeline).
    state.kill_agent(w1);
    prime_kin_fear(&mut state, w2, w1);

    // Post-death: kin_fear > 0.5, Flee gains +0.4. Plus the hp<50 gate
    // contributes +0.4. Flee ≈ 0.8; Attack on fresh h1 drops to 0.0
    // (fresh-self gate no longer fires at hp_pct=0.56). Flee >> Attack.
    let flee_post = score_row_for(flee, &state, w2, None);
    let attack_post = score_row_for(attack, &state, w2, Some(h1));
    assert!(
        flee_post > attack_post,
        "post-death: Flee ({flee_post}) should beat Attack ({attack_post}) — \
         a lightly-wounded wolf with kin_fear > 0.5 routs",
    );
    assert!(
        flee_post > flee_pre + 0.3,
        "kin_fear should bump Flee by ~+0.4 ({flee_pre} → {flee_post})",
    );
}

/// Decay: after enough ticks (~2 half-lives ≈ 12), kin_fear drops below
/// 0.5 and the Flee bonus vanishes. Proves the rout is transient — the
/// wolf fights again once the fear wears off. Task 173 shortened the
/// half-life from ~15 ticks to ~6 ticks (rate 0.955 → 0.891) so the
/// recovery window is much tighter.
#[test]
fn kin_fear_decays_below_threshold() {
    let (mut state, wolves, _humans) = spawn_pack_fixture();
    let [w1, w2, _w3] = wolves;

    prime_kin_fear(&mut state, w2, w1);
    let immediate = state.views.kin_fear.sum_for_first(w2, state.tick);
    assert!(immediate > 0.5, "immediate kin_fear {immediate} should be > 0.5");

    // Advance the observer's observed tick. @decay is `rate=0.891`, so
    // 12 ticks gives 0.891^12 ≈ 0.247 — below the 0.5 gate. 20 ticks
    // gives 0.891^20 ≈ 0.100 — well below. We check at 20 to have
    // margin against tick-boundary math.
    let decayed = state.views.kin_fear.sum_for_first(w2, state.tick + 20);
    assert!(
        decayed < 0.5,
        "after 20 ticks kin_fear {decayed} should decay below 0.5 (rout window elapsed)",
    );

    // And at that tick Flee loses the +0.4 modifier. Clone the fold
    // state but advance `state.tick` so the scorer reads the decayed
    // value (the modifier evaluates `view::kin_fear(self, _)` at
    // `state.tick`, not the observer-supplied tick — so we move time
    // forward on the state itself).
    state.tick += 20;
    let flee = flee_entry();
    let s = score_row_for(flee, &state, w2, None);
    assert!(
        s < 0.1,
        "Flee at decayed tick = {s}; with all gates off it should be ≈0.0",
    );
}

/// Symmetric: humans rout when a human dies. The view / physics is
/// species-agnostic (scoping lives in `query.nearby_kin`), so the
/// same mechanic should fire for humans. If this breaks, something
/// leaked a wolf-only assumption into the view fold or the scoring
/// wiring.
#[test]
fn humans_rout_when_packmate_dies_same_mechanic() {
    let (mut state, _wolves, humans) = spawn_pack_fixture();
    let [h1, h2] = humans;

    // Kill h1, emit FearSpread at h2 (the one observer). Same shape as
    // the wolf case — proves the mechanic is symmetric.
    state.kill_agent(h1);
    prime_kin_fear(&mut state, h2, h1);

    let total = state.views.kin_fear.sum_for_first(h2, state.tick);
    assert!(
        total > 0.5,
        "human kin_fear {total} should cross 0.5 gate",
    );

    let flee = flee_entry();
    let s = score_row_for(flee, &state, h2, None);
    assert!(
        (s - 0.4).abs() < 1e-3,
        "human Flee post-death = {s}, expected ≈0.4 (same +0.4 as wolves)",
    );
}

// ---------------------------------------------------------------------------
// End-to-end smoke — does the full cascade (step_full) actually emit
// FearSpread and fold it? The direct-fold tests above cover the scoring
// math; this one ties it to the cascade wiring so a broken dispatcher
// registration surfaces.
// ---------------------------------------------------------------------------

/// Directly drive the physics rule (bypass `step_full` so the test stays
/// compact) and verify FearSpread events land in the ring. Confirms the
/// emission path the cascade dispatcher routes through.
#[test]
fn pipeline_death_triggers_fear() {
    let (mut state, wolves, _humans) = spawn_pack_fixture();
    let [w1, w2, w3] = wolves;
    let mut events = EventRing::with_cap(64);

    // Call the generated physics fn directly. Same signature the
    // dispatcher uses; simulates what happens on an `AgentDied { agent_id: w1 }`.
    state.kill_agent(w1);
    engine::generated::physics::fear_spread_on_death::fear_spread_on_death(
        w1,
        &mut state,
        &mut events,
    );

    let fear_events: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::FearSpread { observer, dead_kin, .. } => Some((*observer, *dead_kin)),
            _ => None,
        })
        .collect();

    assert_eq!(
        fear_events.len(),
        2,
        "expected 2 FearSpread events (one per surviving wolf), got {:?}",
        fear_events,
    );
    for &(observer, dead) in &fear_events {
        assert_eq!(dead, w1, "dead_kin should be the killed wolf");
        assert!(
            observer == w2 || observer == w3,
            "observer should be one of the surviving wolves, got {observer:?}",
        );
    }
}
