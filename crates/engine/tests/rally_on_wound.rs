//! Task 178 — rally mechanic.
//!
//! When a same-species kin is wounded (damaged, still alive, hp_pct
//! drops below 0.5), the `rally_on_wound` DSL physics rule emits a
//! `RallyCall` event once per nearby same-species kin within 12 m of
//! the victim. The `rally_boost` @materialized view folds those events
//! into a per-(observer, wounded_kin) decayed scalar; scoring's Attack
//! row stacks +0.3 when the observer's total decayed rally_boost
//! (wildcard sum) crosses 0.3. Result: survivors fight harder for the
//! ~6-tick half-life window after a packmate takes a hit.
//!
//! Symmetric positive opposite of task 167's fear-spread rout:
//!   FearSpread (kin dies)  → Flee +0.4
//!   RallyCall  (kin hurt)  → Attack +0.3
//!
//! These tests drive the view's fold directly (no full `step_full`
//! pipeline) because the scoring response is a pure function of view
//! state + tick, and we want to pin the fold + scoring wiring without
//! entangling with mask / cascade / movement. The physics rule's emit
//! path is smoke-tested by `pipeline_wound_triggers_rally`.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::invariant::InvariantRegistry;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::NullSink;
use engine::view::MaterializedView;
use engine::mask::MicroKind;
use engine_rules::scoring::{
    PredicateDescriptor, ScoringEntry, MAX_MODIFIERS, SCORING_TABLE,
};
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

/// 3 humans clustered + 1 wolf. Humans at (0,0,0) / (1,0,0) / (-1,0,0),
/// wolf at (3,0,0). All humans within 12 m of each other so a wound
/// on H1 emits RallyCall at H2 and H3. Wolf is 3 m from H1 — inside
/// the wolf's 2 m attack range? No, 3 m > 2 m — outside. The direct-
/// fold tests override this by calling `prime_rally_boost` manually;
/// the pipeline test (`pipeline_wound_triggers_rally`) relies only on
/// the physics rule's emission, not whether the wolf actually lands a
/// hit.
fn spawn_rally_fixture() -> (SimState, [AgentId; 3], AgentId) {
    let mut state = SimState::new(16, 0xF00D_0178);
    let h1 = spawn_human(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let h2 = spawn_human(&mut state, Vec3::new(1.0, 0.0, 0.0));
    let h3 = spawn_human(&mut state, Vec3::new(-1.0, 0.0, 0.0));
    let w1 = spawn_wolf(&mut state, Vec3::new(3.0, 0.0, 0.0));
    (state, [h1, h2, h3], w1)
}

// ---------------------------------------------------------------------------
// Scoring harness — mirrors `fear_spread_rout.rs` / `pack_focus.rs`
// because `engine::policy::utility::score_entry` is `pub(crate)`. Only
// covers the view-calls this test exercises (threat_level / my_enemies
// / kin_fear / pack_focus / rally_boost).
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
        PredicateDescriptor::VIEW_ID_RALLY_BOOST => {
            let a = resolve(slot0).unwrap_or(agent);
            if slot1 == PredicateDescriptor::ARG_WILDCARD {
                state.views.rally_boost.sum_for_first(a, state.tick)
            } else {
                let b = resolve(slot1).unwrap_or(agent);
                state.views.rally_boost.get(a, b, state.tick)
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

/// Drive the `rally_boost` view's fold directly, bypassing the physics
/// rule but matching its emit shape. One call = one `RallyCall` event
/// folded.
fn prime_rally_boost(state: &mut SimState, observer: AgentId, wounded_kin: AgentId) {
    let ev = Event::RallyCall {
        observer,
        wounded_kin,
        tick: state.tick,
    };
    state.views.rally_boost.fold_event(&ev, state.tick);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Folding one RallyCall event bumps the observer's rally_boost
/// wildcard-sum above the 0.3 threshold the Attack row gates on.
/// Mirror of fear_spread's `one_fear_spread_crosses_threshold`.
#[test]
fn one_rally_call_crosses_threshold() {
    let (mut state, humans, _wolf) = spawn_rally_fixture();
    let [h1, h2, _h3] = humans;

    // Baseline: no rally recorded.
    assert_eq!(
        state.views.rally_boost.sum_for_first(h2, state.tick),
        0.0,
        "baseline rally_boost should be 0",
    );

    // One kin wound → one fold → value = +1.0, well above 0.3 gate.
    prime_rally_boost(&mut state, h2, h1);
    let total = state.views.rally_boost.sum_for_first(h2, state.tick);
    assert!(
        total > 0.3,
        "rally_boost {} should cross 0.3 gate after one wound",
        total,
    );
    assert!(
        (total - 1.0).abs() < 1e-4,
        "rally_boost fold-amount is hardcoded +1.0 per event (emit_fold_arm); got {}",
        total,
    );
}

/// The Attack row gains +0.3 on the rally_boost modifier once the sum
/// crosses 0.3 — the key behavioral pin for the rally mechanic. Also
/// confirms the modifier fires through the generated SCORING_TABLE, not
/// just the view. Primary "before/after Attack scores" comparison for
/// the task report: H2 and H3 sitting fresh → rally triggers → Attack
/// score bumps by +0.3.
#[test]
fn humans_rally_when_kin_wounded() {
    let (mut state, humans, wolf) = spawn_rally_fixture();
    let [h1, h2, h3] = humans;
    let entry = attack_entry();

    // Pre-rally Attack scores for H2 and H3 targeting the wolf. With
    // fresh selves (hp_pct=1.0 ≥ 0.8 → +0.5) and a fresh wolf
    // (hp_pct=1.0 → no target bonus) there's no threat_level,
    // my_enemies, pack_focus, or rally_boost. Expected: base 0.0 +
    // 0.5 fresh-self = 0.5.
    let h2_pre = score_row_for(entry, &state, h2, Some(wolf));
    let h3_pre = score_row_for(entry, &state, h3, Some(wolf));
    assert!(
        (h2_pre - 0.5).abs() < 1e-4,
        "baseline Attack(h2, wolf) = {h2_pre}, expected ≈0.5",
    );
    assert!(
        (h3_pre - 0.5).abs() < 1e-4,
        "baseline Attack(h3, wolf) = {h3_pre}, expected ≈0.5",
    );

    // Wound H1: simulate the physics rule emitting RallyCall at H2 and
    // H3 (the two surviving kin within 12 m of the wounded H1). Mirror
    // of `wolves_rout_when_packmate_dies` in fear_spread_rout.rs which
    // does the symmetric prime_kin_fear dance.
    prime_rally_boost(&mut state, h2, h1);
    prime_rally_boost(&mut state, h3, h1);

    // Post-rally Attack scores. Rally bump is +0.3 on top of the
    // 0.5 baseline. Expected: 0.8.
    let h2_post = score_row_for(entry, &state, h2, Some(wolf));
    let h3_post = score_row_for(entry, &state, h3, Some(wolf));
    assert!(
        h2_post > h2_pre + 0.25,
        "Attack(h2, wolf) should gain ≥+0.25 from rally_boost (+0.3 expected, \
         some epsilon); got {h2_pre} → {h2_post}",
    );
    assert!(
        h3_post > h3_pre + 0.25,
        "Attack(h3, wolf) should gain ≥+0.25 from rally_boost (+0.3 expected, \
         some epsilon); got {h3_pre} → {h3_post}",
    );
    assert!(
        (h2_post - 0.8).abs() < 1e-3,
        "post-rally Attack(h2, wolf) = {h2_post}, expected ≈0.8 \
         (0.5 fresh-self + 0.3 rally_boost)",
    );
    assert!(
        (h3_post - 0.8).abs() < 1e-3,
        "post-rally Attack(h3, wolf) = {h3_post}, expected ≈0.8 \
         (0.5 fresh-self + 0.3 rally_boost)",
    );

    // Print the score comparison for the task report (visible with
    // `cargo test -- --nocapture`).
    println!(
        "humans_rally_when_kin_wounded: H2 Attack {h2_pre:.3} → {h2_post:.3}, \
         H3 Attack {h3_pre:.3} → {h3_post:.3} (+0.3 rally_boost modifier)"
    );
}

/// Decay: after ~12 ticks (~2 half-lives at rate 0.891) the rally_boost
/// scalar drops below 0.3 and the Attack bump vanishes. Proves rally is
/// transient — same decay profile as kin_fear (task 173 retuned).
#[test]
fn rally_boost_decays_below_threshold() {
    let (mut state, humans, wolf) = spawn_rally_fixture();
    let [h1, h2, _h3] = humans;

    prime_rally_boost(&mut state, h2, h1);
    let immediate = state.views.rally_boost.sum_for_first(h2, state.tick);
    assert!(immediate > 0.3, "immediate rally_boost {immediate} should be > 0.3");

    // 20 ticks × 0.891 ≈ 0.1 — well below the 0.3 gate.
    let decayed = state.views.rally_boost.sum_for_first(h2, state.tick + 20);
    assert!(
        decayed < 0.3,
        "after 20 ticks rally_boost {decayed} should decay below 0.3 (rally window elapsed)",
    );

    // And at that tick Attack loses the +0.3 modifier — back to base 0.5.
    state.tick += 20;
    let entry = attack_entry();
    let s = score_row_for(entry, &state, h2, Some(wolf));
    assert!(
        (s - 0.5).abs() < 1e-3,
        "post-decay Attack(h2, wolf) = {s} should be back to baseline 0.5 \
         (fresh-self +0.5, no rally_boost bump)",
    );
}

/// Symmetric: wolves rally when a packmate is wounded. The view /
/// physics is species-agnostic (scoping lives in `query.nearby_kin`),
/// so the same mechanic fires for wolves. If this breaks, something
/// leaked a human-only assumption into the view fold or the scoring
/// wiring.
#[test]
fn wolves_rally_when_packmate_wounded_same_mechanic() {
    let mut state = SimState::new(8, 0);
    let w1 = spawn_wolf(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let w2 = spawn_wolf(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let human = spawn_human(&mut state, Vec3::new(5.0, 0.0, 0.0));

    prime_rally_boost(&mut state, w2, w1);

    let entry = attack_entry();
    let s = score_row_for(entry, &state, w2, Some(human));
    // w2 fresh, human fresh → +0.5 from self hp_pct + 0.3 rally = 0.8.
    assert!(
        (s - 0.8).abs() < 1e-3,
        "wolf Attack(w2, human) after rally_boost = {s}, expected ≈0.8 \
         (same +0.3 as humans)",
    );
}

// ---------------------------------------------------------------------------
// Pipeline smoke — does the generated physics rule actually emit
// RallyCall when called? The direct-fold tests above cover the scoring
// math; this ties it to the emit path so a broken cascade wiring
// surfaces.
// ---------------------------------------------------------------------------

/// Directly drive the physics rule (bypass `step_full` so the test
/// stays compact) and verify RallyCall events land in the ring.
/// Confirms the emission path the cascade dispatcher routes through.
#[test]
fn pipeline_wound_triggers_rally() {
    let (mut state, humans, wolf) = spawn_rally_fixture();
    let [h1, h2, h3] = humans;
    let mut events = EventRing::with_cap(64);

    // Drop H1 to 40/100 hp — hp_pct = 0.4 < 0.5 satisfies the rule's
    // "wounded" gate.
    state.set_agent_hp(h1, 40.0);

    // Call the generated physics fn directly — simulates the dispatcher
    // handling an `AgentAttacked { actor: wolf, target: h1, damage: 60 }`
    // event after the `damage` handler set h1.hp = 40.
    engine::generated::physics::rally_on_wound::rally_on_wound(
        wolf,
        h1,
        &mut state,
        &mut events,
    );

    let rallies: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::RallyCall {
                observer,
                wounded_kin,
                ..
            } => Some((*observer, *wounded_kin)),
            _ => None,
        })
        .collect();

    assert_eq!(
        rallies.len(),
        2,
        "expected 2 RallyCall events (one per surviving kin: h2 and h3), got {:?}",
        rallies,
    );
    for &(observer, wounded) in &rallies {
        assert_eq!(wounded, h1, "wounded_kin should be h1");
        assert!(
            observer == h2 || observer == h3,
            "observer should be one of the surviving humans, got {observer:?}",
        );
    }
}

/// Non-wounded hits do not emit RallyCall. H1 at 80/100 (hp_pct=0.8 ≥
/// 0.5) still counts as "not wounded" in the task 166 chronicle_wound
/// sense, so the rule's gate rejects. Guards the threshold.
#[test]
fn rally_rule_skips_non_wounded() {
    let (mut state, humans, wolf) = spawn_rally_fixture();
    let [h1, _h2, _h3] = humans;
    let mut events = EventRing::with_cap(16);

    // H1 still at 80 hp — above the 0.5 hp_pct gate.
    state.set_agent_hp(h1, 80.0);

    engine::generated::physics::rally_on_wound::rally_on_wound(
        wolf,
        h1,
        &mut state,
        &mut events,
    );

    let rallies = events
        .iter()
        .filter(|e| matches!(e, Event::RallyCall { .. }))
        .count();
    assert_eq!(
        rallies, 0,
        "expected 0 RallyCall events (H1 hp=80 is not wounded), got {rallies}",
    );
}

/// Killing-blow ordering: a lethal AgentAttacked does not emit a
/// RallyCall. The cascade orders `damage` to emit `AgentAttacked` before
/// calling `agents.kill(...)`, so by the time `rally_on_wound` runs on
/// a lethal hit the target's `alive` flag is false. The rule's first
/// guard (`agents.alive(t)`) skips the rally; the death's FearSpread
/// fires instead via `fear_spread_on_death`. Guards the mirror-
/// asymmetry: wound rallies, death routs, never both on the same hit.
#[test]
fn rally_rule_skips_dead_target() {
    let (mut state, humans, wolf) = spawn_rally_fixture();
    let [h1, _h2, _h3] = humans;
    let mut events = EventRing::with_cap(16);

    // Simulate the damage handler's lethal branch: it sets hp=0 then
    // calls agents.kill(h1). By the time rally_on_wound runs, h1 is
    // !alive. (The actual damage physics emits AgentAttacked first,
    // then calls kill.)
    state.set_agent_hp(h1, 0.0);
    state.kill_agent(h1);

    engine::generated::physics::rally_on_wound::rally_on_wound(
        wolf,
        h1,
        &mut state,
        &mut events,
    );

    let rallies = events
        .iter()
        .filter(|e| matches!(e, Event::RallyCall { .. }))
        .count();
    assert_eq!(
        rallies, 0,
        "expected 0 RallyCall events (h1 is dead, only FearSpread should fire), got {rallies}",
    );
}

/// Cross-species filter: a wounded human does not emit a RallyCall at
/// nearby wolves. Species scoping is enforced by `query.nearby_kin` in
/// the physics rule.
#[test]
fn rally_rule_is_species_scoped() {
    let mut state = SimState::new(8, 0);
    let wolf = spawn_wolf(&mut state, Vec3::new(0.0, 0.0, 0.0));
    let human = spawn_human(&mut state, Vec3::new(2.0, 0.0, 0.0));
    let mut events = EventRing::with_cap(16);

    state.set_agent_hp(human, 40.0);

    engine::generated::physics::rally_on_wound::rally_on_wound(
        wolf,
        human,
        &mut state,
        &mut events,
    );

    // Only same-species kin get RallyCall; the nearby wolf should NOT
    // rally for a wounded human.
    let rallies = events
        .iter()
        .filter(|e| matches!(e, Event::RallyCall { .. }))
        .count();
    assert_eq!(
        rallies, 0,
        "expected 0 RallyCall events (wolf is not a human's kin), got {rallies}",
    );
}

// ---------------------------------------------------------------------------
// End-to-end smoke — let step_full drive the whole pipeline. Spawns a
// 3-human cluster with 1 wolf in attack range so the wolf's first
// successful hit wounds a human; confirms RallyCall events actually
// flow through the dispatcher and land in the ring.
// ---------------------------------------------------------------------------

/// Full-pipeline smoke: attach humans + 1 wolf, step_full runs for a
/// handful of ticks, assert at least one RallyCall event appears in the
/// event log. Uses a cluster where the wolf is inside attack range so
/// a hit is inevitable on tick 0 or 1.
#[test]
fn pipeline_end_to_end_emits_rally() {
    let mut state = SimState::new(8, 0xF00D_D33D);
    // Humans clustered (all within kin-radius of each other).
    let _h1 = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 20.0, // low hp so first hit crosses the 0.5 gate immediately
            max_hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    let _h2 = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(2.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        })
        .unwrap();
    // Wolf 1 m from H1 (inside 2 m attack range).
    let _w1 = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 80.0,
            max_hp: 80.0,
            ..Default::default()
        })
        .unwrap();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1 << 14);
    let cascade = CascadeRegistry::with_engine_builtins();
    let invariants = InvariantRegistry::new();
    let mut views: Vec<&mut dyn MaterializedView> = Vec::new();
    let telemetry = NullSink;

    for _ in 0..6 {
        step_full(
            &mut state,
            &mut scratch,
            &mut events,
            &UtilityBackend,
            &cascade,
            &mut views[..],
            &invariants,
            &telemetry,
        );
    }

    let rallies: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, Event::RallyCall { .. }))
        .collect();
    assert!(
        !rallies.is_empty(),
        "expected at least one RallyCall over 6 ticks (wolf in attack range of low-hp human); \
         got {:?} events",
        events.iter().count(),
    );
}
